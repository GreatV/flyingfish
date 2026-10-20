//! Fixed-row Prefill verification on the same Engine. The policy controller
//! owns commit/rollback; this adapter owns the CUDA graphs and hidden frame.
use super::*;
use crate::{
    backend::cuda::{linear::LinearWeightRef, ops::grid},
    backend::{Head, Input, Mask, Step},
    mtp::batched::{self as policy, BatchRound, BatchedBackend},
};
use cudarc::driver::{CudaGraph, sys};

fn optional_verify_graphs<G>(
    graph: bool,
    capture: impl FnOnce() -> Result<Vec<G>>,
) -> Result<Vec<G>> {
    if graph { capture() } else { Ok(Vec::new()) }
}

pub(super) struct Batch {
    snapshots: Vec<crate::backend::cuda::state::GdnState>,
    frame_epoch: Option<u64>,
    logits: CudaSlice<bf16>,
    pub(super) status: CudaSlice<u32>,
    host_predictions: [u32; 5],
    row_counts: CudaSlice<i32>,
    meta: CudaFunction,
    pub(super) sequence_check: CudaFunction,
    common: CommonOpsRef,
    capture_checkpoint: StateCheckpoint,
    graphs: Vec<CudaGraph>,
    frame: CudaSlice<bf16>,
    anchor: CudaSlice<bf16>,
    anchor_token: u32,
    frame_position: usize,
    frame_rows: usize,
    #[cfg(test)]
    repair_fail_after: Option<usize>,
    pub(super) calls: usize,
    pub(super) target_restores: usize,
    pub(super) mtp_restores: usize,
}

impl Batch {
    pub(super) fn invalidate_frame(&mut self) {
        self.frame_epoch = None;
        self.frame_position = 0;
        self.frame_rows = 0;
    }
}

impl Engine {
    /// Depth+1 fixed query shapes retain recurrent prefixes.
    pub(in crate::backend::cuda::engine) fn mtp_batch_enable(&mut self) -> Result<()> {
        ensure!(
            self.position == 0 && self.mtp.as_ref().is_some_and(|c| c.batch.is_none()),
            "M2b setup requires empty MTP Engine"
        );
        let verify_rows = self
            .mtp
            .as_ref()
            .context("MTP missing")?
            .options
            .verify_rows(Some(self.workspace_rows))?;
        // This guard rejects late setup before allocating or binding anything.
        self.closure.bind_fixed()?;
        let graph = self.mtp.as_ref().context("MTP missing")?.options.graph;
        let s = self.device.stream.clone();
        let journal = self
            .mtp
            .as_ref()
            .context("MTP missing")?
            .target_checkpoint
            .bytes();
        let buffers = (5 * VOCAB + 6 * HIDDEN) * size_of::<bf16>() + size_of::<[i32; 15]>();
        let snapshot_bytes = crate::mtp::snapshot_bytes(&self.model_spec)?;
        let needed = crate::mtp::batch_bytes(&self.model_spec)?;
        ensure!(
            needed == journal + buffers + snapshot_bytes,
            "M2b budget/attachment mismatch"
        );
        let (free, total) = self.device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| needed <= n),
            "M2b attachment exceeds memory budget: required={needed} free={free}"
        );
        for rows in 1..=verify_rows {
            self.prepare(rows)?;
            self.blas
                .prepare_fixed(rows as u32, VOCAB as u32, HIDDEN as u32)?;
        }
        let snapshots = self
            .state
            .gdn
            .iter()
            .map(|state| {
                Ok(crate::backend::cuda::state::GdnState {
                    s: s.alloc_zeros(4 * state.s.len())?,
                    history: s.alloc_zeros(4 * state.history.len())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            snapshots
                .iter()
                .map(|g| (g.s.len() + g.history.len()) * size_of::<f32>())
                .sum::<usize>()
                == snapshot_bytes,
            "snapshot allocation ledger mismatch"
        );
        self.bind_gdn_snapshots(&snapshots, verify_rows)?;
        let mut batch = Batch {
            snapshots,
            frame_epoch: None,
            logits: s.alloc_zeros(5 * VOCAB)?,
            status: s.alloc_zeros(10)?,
            host_predictions: [0; 5],
            row_counts: s.clone_htod(&[1, 2, 3, 4, 5])?,
            meta: cubin::module(&self.device.ctx, "q38_common")?.load_function("q38_batch_meta")?,
            sequence_check: cubin::module(&self.device.ctx, "q38_common")?
                .load_function("q38_sequence_check")?,
            common: CommonOpsRef::new(&self.device.ctx)?,
            capture_checkpoint: StateCheckpoint::new(&self.state, &self.model_spec, 5, &s)?,
            graphs: Vec::new(),
            frame: s.alloc_zeros(5 * HIDDEN)?,
            anchor: s.alloc_zeros(HIDDEN)?,
            anchor_token: 0,
            frame_position: 0,
            frame_rows: 0,
            #[cfg(test)]
            repair_fail_after: None,
            calls: 0,
            target_restores: 0,
            mtp_restores: 0,
        };
        ensure!(
            batch.capture_checkpoint.bytes() == journal,
            "M2b journal ledger mismatch"
        );
        batch.graphs = optional_verify_graphs(graph, || self.mtp_batch_capture(&mut batch))?;
        let graph_shapes = if batch.graphs.is_empty() {
            "none".to_owned()
        } else {
            (1..=batch.graphs.len())
                .map(|rows| rows.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        self.mtp.as_mut().context("MTP missing")?.batch = Some(batch);
        eprintln!(
            "MTP_M2B_MEMORY additional_device_bytes={needed} capture_checkpoint_bytes={journal} snapshot_bytes={snapshot_bytes} graph_shapes={graph_shapes}"
        );
        Ok(())
    }

    fn mtp_batch_capture(&mut self, batch: &mut Batch) -> Result<Vec<CudaGraph>> {
        batch.invalidate_frame();
        self.closure.capture()?;
        let s = self.device.stream.clone();
        batch
            .capture_checkpoint
            .save(&self.state, &self.work.token, 0, &s, None)?;
        let logits = s.clone_dtoh(&self.work.logits)?;
        let result = (|| -> Result<Vec<CudaGraph>> {
            let mut graphs = Vec::new();
            let verify_rows = self
                .mtp
                .as_ref()
                .context("MTP missing")?
                .options
                .verify_rows(Some(self.workspace_rows))?;
            for rows in 1..=verify_rows {
                if rows > 1 {
                    self.closure.capture()?;
                }
                self.mtp_batch_launch(batch, rows)?;
                s.synchronize()?;
                batch
                    .capture_checkpoint
                    .restore(&mut self.state, &mut self.work.token, &s)?;
                s.memcpy_htod(&logits, &mut self.work.logits)?;
                self.logits_cache.mark_valid();
                s.synchronize()?;
                #[cfg(test)]
                ensure!(
                    self.capture_fail_stage != Some(super::super::super::CaptureFault::Begin),
                    "injected batch capture begin failure"
                );
                s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let launches = self.mtp_batch_launch(batch, rows);
                let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
                launches?;
                #[cfg(test)]
                ensure!(
                    self.capture_fail_stage != Some(super::super::super::CaptureFault::End),
                    "injected batch capture end failure"
                );
                let graph = captured?.context("empty M2b graph")?;
                #[cfg(test)]
                ensure!(
                    self.capture_fail_stage != Some(super::super::super::CaptureFault::Instantiate),
                    "injected batch capture instantiate failure"
                );
                graph.upload()?;
                s.synchronize()?;
                batch
                    .capture_checkpoint
                    .restore(&mut self.state, &mut self.work.token, &s)?;
                s.memcpy_htod(&logits, &mut self.work.logits)?;
                self.logits_cache.mark_valid();
                s.synchronize()?;
                graphs.push(graph);
            }
            Ok(graphs)
        })();
        let restored = (|| -> Result<()> {
            batch
                .capture_checkpoint
                .restore(&mut self.state, &mut self.work.token, &s)?;
            s.memcpy_htod(&logits, &mut self.work.logits)?;
            self.logits_cache.mark_valid();
            s.synchronize()?;
            Ok(())
        })();
        if let Err(error) = restored {
            self.poisoned = true;
            return Err(error.context("M2b capture restore failed"));
        }
        match result {
            Ok(graphs) => Ok(graphs),
            Err(error) => {
                self.poisoned = true;
                Err(error.context("M2b capture transaction failed"))
            }
        }
    }

    /// Capture-safe: no host transfers/allocations, all plans prebound. Metadata
    /// is generated from the device position before Q/K scatter, GDN and FA2.
    fn mtp_batch_launch(&mut self, batch: &mut Batch, rows: usize) -> Result<()> {
        ensure!(
            (1..=crate::mtp::MAX_VERIFY_ROWS).contains(&rows),
            "invalid M2b query rows"
        );
        let s = self.device.stream.clone();
        unsafe {
            s.launch_builder(&batch.meta)
                .arg(&self.work.position)
                .arg(
                    self.state
                        .sequence
                        .as_mut()
                        .context("Q38 sequence missing")?,
                )
                .arg(&(rows as i32))
                .arg(&(self.capacity as i32))
                .launch(grid(1, 1))?;
        }
        ensure!(!self.qwen_device_prefix, "nested device-prefix forward");
        self.qwen_device_prefix = true;
        let forward = self.run_step(
            Step {
                rows,
                input: Input::Ids,
                head: Head::None,
                mask: Mask::Causal,
            },
            None,
        );
        self.qwen_device_prefix = false;
        forward?;
        batch.common.norm.rms_with(
            &s,
            RowDomain {
                meta: self.state.seq_ref(&s)?,
                row_base: 0,
                row_capacity: rows as u32,
            },
            heads(&self.work.x, &s, 0, rows, 1, HIDDEN as u32)?,
            vector(&self.weights.norm, &s)?,
            &mut TensorMut(heads(&self.work.n, &s, 0, rows, 1, HIDDEN as u32)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: self.model_spec.rms_eps,
            },
        )?;
        self.blas.linear_typed(
            ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(matrix(&self.weights.head, &s, 0, VOCAB, HIDDEN)?),
            matrix(&self.work.n, &s, 0, rows, HIDDEN)?,
            &mut TensorMut(matrix(&batch.logits, &s, 0, rows, VOCAB)?),
        )?;
        unsafe {
            s.launch_builder(&self.ops.argmax_rows)
                .arg(&batch.logits)
                .arg(&mut batch.status.slice_mut(4..9))
                .arg(&(VOCAB as i32))
                .arg(&batch.row_counts.slice(rows - 1..rows))
                .launch(grid(rows, 1024))?;
        }
        unsafe {
            s.launch_builder(&self.ops.finite_bf16)
                .arg(&batch.logits)
                .arg(&mut batch.status.slice_mut(9..10))
                .arg(&((rows * VOCAB) as u64))
                .launch(crate::backend::cuda::ops::flat(rows * VOCAB))?;
        }
        s.memcpy_dtod(
            &batch.logits.slice((rows - 1) * VOCAB..rows * VOCAB),
            &mut self.work.logits,
        )?;
        s.memcpy_dtod(
            &batch.status.slice(rows + 3..rows + 4),
            &mut self.work.token,
        )?;
        self.logits_cache.mark_valid();
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn mtp_batch_round(
        &mut self,
        remaining: usize,
        #[cfg(test)] fault: Option<policy::Fault>,
    ) -> Result<BatchRound> {
        self.closure.assert_ready()?;
        ensure!(!self.poisoned, "M2b requires healthy Engine");
        let mut chain = self.mtp.take().context("MTP missing")?;
        let result = (|| {
            ensure!(
                chain.batch.is_some()
                    && self.position > 0
                    && chain.committed_position == self.position
                    && chain.epoch == self.mtp_epoch,
                "M2b prefix changed outside controller or batch not bound"
            );
            let options = chain.options;
            let eos = self.model_spec.eos.clone();
            let r = policy::run(
                &mut Bound {
                    engine: self,
                    chain: &mut chain,
                },
                options,
                remaining,
                &eos,
                #[cfg(test)]
                fault,
            )?;
            self.state.compact_commit(&self.device.stream, true)?;
            chain.committed_position = self.position;
            chain
                .batch
                .as_mut()
                .context("M2b not bound")?
                .invalidate_frame();
            Ok(r)
        })();
        self.mtp = Some(chain);
        result
    }
}

impl Bound<'_> {
    fn queued_draft(&mut self, depth: usize) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        ensure!(
            c.saved.position == Some(e.position) && c.cursor == c.saved.cursor + depth,
            "stale M2b draft checkpoint/cursor"
        );
        let ids = if depth == 0 {
            view(&e.work.token, s, 0, [1], [1])?
        } else {
            view(&c.runtime.prediction, s, 0, [1], [1])?
        };
        let hidden = matrix(
            if depth == 0 {
                &c.last_target
            } else {
                &c.recurrent
            },
            s,
            0,
            1,
            HIDDEN,
        )?;
        let flag = view(
            &c.batch.as_ref().context("M2b not bound")?.status,
            s,
            0,
            [10],
            [1],
        )?;
        c.runtime.run_device(
            Execution {
                device: &e.device,
                ops: &e.ops,
                blas: &e.blas,
                catalog: &c.catalog,
                embedding: &e.weights.embed,
                head: &e.weights.head,
            },
            e.mtp_state.as_mut().context("MTP state missing")?,
            c.cursor,
            ((ids, 0), hidden, flag),
            ExecutionPhase::Decode,
            OutputMode::Queued,
        )?;
        let output = match c.options.recurrent_tap {
            MtpRecurrentTap::PreMtpNorm => &c.runtime.work.x,
            MtpRecurrentTap::PostMtpNorm => &c.runtime.work.n,
        };
        let batch = c.batch.as_mut().context("M2b not bound")?;
        let flag = view(&batch.status, s, 0, [10], [1])?;
        e.ops
            .finite(s, matrix(&c.runtime.logits, s, 0, 1, VOCAB)?, flag)?;
        e.ops
            .finite(s, matrix(&c.runtime.hidden, s, 0, 1, HIDDEN)?, flag)?;
        e.ops
            .finite(s, matrix(&c.runtime.work.n, s, 0, 1, HIDDEN)?, flag)?;
        if c.options.recurrent_tap == MtpRecurrentTap::PreMtpNorm {
            e.ops.finite(s, matrix(output, s, 0, 1, HIDDEN)?, flag)?;
        }
        s.memcpy_dtod(&output.slice(..HIDDEN), &mut c.recurrent)?;
        if depth == 0 {
            s.memcpy_dtod(&c.recurrent, &mut batch.anchor)?;
            batch.anchor_token = c.pending;
        }
        #[cfg(test)]
        if let Some(&token) = c.force_proposals.as_ref().and_then(|v| v.get(depth)) {
            s.memcpy_htod(&[token], &mut c.runtime.prediction.slice_mut(..1))?;
        }
        s.memcpy_dtod(
            &c.runtime.prediction.slice(..1),
            &mut batch.status.slice_mut(depth..depth + 1),
        )?;
        c.cursor += 1;
        e.mtp_state
            .as_mut()
            .context("MTP state missing")?
            .set_sequence(s, c.cursor as u32, 1, 1)?;
        Ok(())
    }

    fn verify_device(&mut self, rows: usize) -> Result<(Vec<u32>, Vec<u32>)> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        ensure!(
            (2..=5).contains(&rows)
                && e.position + rows < e.capacity
                && c.saved.position == Some(e.position),
            "invalid M2b batch extent"
        );
        let s = e.device.stream.clone();
        let batch = c.batch.as_mut().context("M2b not bound")?;
        batch.invalidate_frame();
        ensure!(
            c.runtime.positions.len() >= 3 * rows,
            "M2b input-staging workspace too small"
        );
        let flag = view(&batch.status, &s, 0, [10], [1])?;
        let flag_ptr = flag.element_ptr(9)?;
        unsafe {
            s.launch_builder(&c.runtime.stage_inputs)
                .arg(&e.work.ids)
                .arg(&e.work.ids)
                .arg(&mut c.runtime.positions)
                .arg(
                    e.state
                        .sequence
                        .as_mut()
                        .context("target sequence missing")?,
                )
                .arg(&(rows as u32))
                .arg(&(e.position as i32))
                .arg(&(e.capacity as i32))
                .arg(&(VOCAB as u32))
                .arg(&flag_ptr)
                .launch(crate::backend::cuda::ops::flat(rows))?;
        }
        if c.options.graph {
            batch
                .graphs
                .get(rows - 1)
                .context("M2b graph missing")?
                .launch()?;
        } else {
            e.mtp_batch_launch(batch, rows)?;
        }
        let hidden = match c.options.hidden_tap {
            MtpHiddenTap::PreFinalNorm => &e.work.x,
            MtpHiddenTap::PostFinalNorm => &e.work.n,
        };
        s.memcpy_dtod(
            &hidden.slice(..rows * HIDDEN),
            &mut batch.frame.slice_mut(..rows * HIDDEN),
        )?;
        batch.frame_position = e.position;
        batch.frame_rows = rows;
        batch.frame_epoch = Some(e.mtp_epoch);
        batch.calls += 1;
        let flag = view(&batch.status, &s, 0, [10], [1])?;
        e.ops
            .finite(&s, matrix(&batch.frame, &s, 0, rows, HIDDEN)?, flag)?;
        c.runtime.run_device(
            Execution {
                device: &e.device,
                ops: &e.ops,
                blas: &e.blas,
                catalog: &c.catalog,
                embedding: &e.weights.embed,
                head: &e.weights.head,
            },
            e.mtp_state.as_mut().context("MTP state missing")?,
            c.saved.cursor + 1,
            (
                (view(&e.work.ids, &s, 0, [rows as u32], [1])?, 1),
                matrix(&batch.frame, &s, 0, rows - 1, HIDDEN)?,
                flag,
            ),
            ExecutionPhase::Prefill,
            OutputMode::Repair,
        )?;
        e.ops.finite(
            &s,
            matrix(&c.runtime.work.n, &s, 0, rows - 1, HIDDEN)?,
            flag,
        )?;
        if c.options.recurrent_tap == MtpRecurrentTap::PreMtpNorm {
            e.ops.finite(
                &s,
                matrix(&c.runtime.work.x, &s, 0, rows - 1, HIDDEN)?,
                flag,
            )?;
        }
        let status = s.clone_dtoh(&batch.status)?;
        ensure!(
            status[9] & 0x8000_0000 == 0,
            "M2b checkpoint sequence validation failed"
        );
        ensure!(
            status[9] & 0x4000_0000 == 0,
            "invalid M2b device input token"
        );
        ensure!(
            status[9] == 0,
            "nonfinite M2b draft, target or repair values"
        );
        ensure!(
            status[..rows - 1]
                .iter()
                .chain(&status[4..rows + 4])
                .all(|&t| (t as usize) < VOCAB),
            "invalid M2b prediction ID"
        );
        batch.host_predictions.copy_from_slice(&status[4..9]);
        c.pending = status[rows + 3];
        e.position += rows;
        s.memcpy_htod(&[e.position as i32], &mut e.work.position)?;
        s.memcpy_htod(&[(e.position + 1) as i32], &mut e.work.length)?;
        e.state
            .set_sequence(&s, e.position as u32, 1, e.workspace_rows as u32)?;
        Ok((status[..rows - 1].to_vec(), status[4..rows + 4].to_vec()))
    }

    #[cfg(test)]
    fn verify_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            (2..=5).contains(&inputs.len()) && inputs.iter().all(|&t| (t as usize) < VOCAB),
            "invalid forced verify inputs"
        );
        let e = &mut self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        let batch = c.batch.as_mut().context("M2b not bound")?;
        s.memset_zeros(&mut batch.status)?;
        s.memcpy_htod(inputs, &mut e.work.ids.slice_mut(..inputs.len()))?;
        s.memcpy_htod(
            &inputs[1..],
            &mut batch.status.slice_mut(..inputs.len() - 1),
        )?;
        let flag = view(&batch.status, s, 0, [10], [1])?;
        c.runtime.run_device(
            Execution {
                device: &e.device,
                ops: &e.ops,
                blas: &e.blas,
                catalog: &c.catalog,
                embedding: &e.weights.embed,
                head: &e.weights.head,
            },
            e.mtp_state.as_mut().context("MTP state missing")?,
            c.saved.cursor,
            (
                (view(&e.work.ids, s, 0, [1], [1])?, 0),
                matrix(&c.saved.last_target, s, 0, 1, HIDDEN)?,
                flag,
            ),
            ExecutionPhase::Decode,
            OutputMode::Repair,
        )?;
        let source = if c.options.recurrent_tap == MtpRecurrentTap::PreMtpNorm {
            &c.runtime.work.x
        } else {
            &c.runtime.work.n
        };
        s.memcpy_dtod(&source.slice(..HIDDEN), &mut batch.anchor)?;
        batch.anchor_token = inputs[0];
        let flag = view(&batch.status, s, 0, [10], [1])?;
        e.ops
            .finite(s, matrix(&c.runtime.work.n, s, 0, 1, HIDDEN)?, flag)?;
        e.ops.finite(s, matrix(source, s, 0, 1, HIDDEN)?, flag)?;
        Ok(self.verify_device(inputs.len())?.1)
    }
}

impl BatchedBackend for Bound<'_> {
    fn draft_verify(
        &mut self,
        anchor: u32,
        depth: usize,
        #[cfg(test)] fault: Option<policy::Fault>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        ensure!(
            (1..=4).contains(&depth) && anchor == self.chain.pending,
            "invalid M2b draft anchor/depth"
        );
        for i in 0..depth {
            self.queued_draft(i)?;
            #[cfg(test)]
            policy::inject(fault, policy::Fault::AfterDraft(i))?;
        }
        let s = &self.engine.device.stream;
        let batch = self.chain.batch.as_ref().context("M2b not bound")?;
        s.memcpy_dtod(
            &self.engine.work.token,
            &mut self.engine.work.ids.slice_mut(..1),
        )?;
        s.memcpy_dtod(
            &batch.status.slice(..depth),
            &mut self.engine.work.ids.slice_mut(1..depth + 1),
        )?;
        self.verify_device(depth + 1)
    }
    fn select_prefix(&mut self, rows: usize) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let saved = &c.saved;
        let saved_position = saved.position.context("MTP checkpoint invalid")?;
        let batch = c.batch.as_mut().context("M2b not bound")?;
        let row = policy::snapshot_row(
            rows,
            batch.frame_rows,
            batch.frame_position,
            e.position,
            batch.frame_epoch,
            e.mtp_epoch,
        )?;
        ensure!(
            c.epoch == e.mtp_epoch
                && batch.frame_position == saved_position
                && batch.frame.len() >= batch.frame_rows * HIDDEN,
            "verify frame/checkpoint mismatch"
        );
        let s = &e.device.stream;
        ensure!(
            batch.snapshots.len() == e.state.gdn.len(),
            "snapshot layer count mismatch"
        );
        for (state, snapshot) in e.state.gdn.iter_mut().zip(&batch.snapshots) {
            let n = state.s.len();
            s.memcpy_dtod(&snapshot.s.slice(row * n..(row + 1) * n), &mut state.s)?;
            let n = state.history.len();
            s.memcpy_dtod(
                &snapshot.history.slice(row * n..(row + 1) * n),
                &mut state.history,
            )?;
        }
        s.memcpy_dtod(
            &batch.logits.slice(row * VOCAB..(row + 1) * VOCAB),
            &mut e.work.logits,
        )?;
        s.memcpy_dtod(&batch.status.slice(row + 4..row + 5), &mut e.work.token)?;
        let position = saved_position + rows;
        s.memcpy_htod(&[position as i32], &mut e.work.position)?;
        s.memcpy_htod(&[(position + 1) as i32], &mut e.work.length)?;
        e.state
            .set_sequence(s, position as u32, 1, e.workspace_rows as u32)?;
        batch.frame_rows = rows;
        c.pending = batch.host_predictions[row];
        e.position = position;
        e.logits_cache.mark_valid();
        Ok(())
    }
    fn restore_mtp(&mut self) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let saved = &c.saved;
        saved.position.context("MTP checkpoint invalid")?;
        let batch = c.batch.as_mut().context("M2b not bound")?;
        ensure!(
            batch.frame_epoch == Some(e.mtp_epoch),
            "M2b canonical frame invalid"
        );
        c.cursor = saved.cursor + 1;
        e.device
            .stream
            .memcpy_dtod(&batch.anchor, &mut c.recurrent)?;
        e.mtp_state
            .as_mut()
            .context("MTP state missing")?
            .set_sequence(&e.device.stream, c.cursor as u32, 1, 1)?;
        batch.mtp_restores += 1;
        Ok(())
    }
    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()> {
        let saved = &self.chain.saved;
        let position = saved.position.context("MTP checkpoint invalid")?;
        let batch = self.chain.batch.as_ref().context("M2b not bound")?;
        ensure!(
            batch.frame_epoch == Some(self.engine.mtp_epoch)
                && batch.frame_position == position
                && batch.frame_rows == inputs.len()
                && self.engine.position == position + inputs.len()
                && self.chain.cursor == saved.cursor + 1
                && inputs.first() == Some(&batch.anchor_token),
            "M2b repair frame/partition mismatch"
        );
        #[cfg(test)]
        ensure!(
            !batch
                .repair_fail_after
                .is_some_and(|row| row < inputs.len()),
            "injected M2b canonical prefix failure"
        );
        let s = &self.engine.device.stream;
        if inputs.len() > 1 {
            let source = match self.chain.options.recurrent_tap {
                MtpRecurrentTap::PreMtpNorm => &self.chain.runtime.work.x,
                MtpRecurrentTap::PostMtpNorm => &self.chain.runtime.work.n,
            };
            let row = inputs.len() - 2;
            s.memcpy_dtod(
                &source.slice(row * HIDDEN..(row + 1) * HIDDEN),
                &mut self.chain.recurrent,
            )?;
        }
        let row = inputs.len() - 1;
        s.memcpy_dtod(
            &batch.frame.slice(row * HIDDEN..(row + 1) * HIDDEN),
            &mut self.chain.last_target,
        )?;
        self.chain.cursor = saved.cursor + inputs.len();
        self.engine
            .mtp_state
            .as_mut()
            .context("MTP state missing")?
            .set_sequence(s, self.chain.cursor as u32, 1, 1)?;
        ensure!(
            self.chain.cursor + 1 == self.engine.position,
            "M2b repair cursor mismatch"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_mtp_batched_tests.rs"]
mod tests;
