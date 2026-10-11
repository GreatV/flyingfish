//! Fixed-row Prefill verification on the same Engine. The policy controller
//! owns commit/rollback; this adapter owns the CUDA graphs and hidden frame.
use super::*;
use crate::{
    backend::cuda::{linear::LinearWeightRef, ops::grid},
    backend::{Head, Input, Mask, Step},
    mtp::batched::{self as policy, BatchRound, BatchedBackend},
};
use cudarc::driver::{CudaGraph, sys};

pub(super) struct Batch {
    logits: CudaSlice<bf16>,
    predictions: CudaSlice<u32>,
    row_counts: CudaSlice<i32>,
    meta: CudaFunction,
    common: CommonOpsRef,
    capture_checkpoint: StateCheckpoint,
    graphs: Vec<CudaGraph>,
    frame: Vec<bf16>,
    frame_position: usize,
    frame_rows: usize,
    #[cfg(test)]
    repair_fail_after: Option<usize>,
    pub(super) calls: usize,
    pub(super) target_restores: usize,
    pub(super) mtp_restores: usize,
}

impl Engine {
    /// Setup before Ready; four fixed query shapes cover K1..3 and every partial
    /// replay (including R1 with Prefill phase). No new measured choices.
    pub(in crate::backend::cuda::engine) fn mtp_batch_enable(&mut self) -> Result<()> {
        ensure!(
            self.position == 0 && self.mtp.as_ref().is_some_and(|c| c.batch.is_none()),
            "M2b setup requires empty MTP Engine"
        );
        ensure!(self.workspace_rows >= 4, "M2b requires four workspace rows");
        // This guard rejects late setup before allocating or binding anything.
        self.closure.capture()?;
        let s = self.device.stream.clone();
        let journal = self
            .mtp
            .as_ref()
            .context("MTP missing")?
            .target_checkpoint
            .bytes();
        let buffers = 4 * VOCAB * size_of::<bf16>() + size_of::<[i32; 8]>();
        let needed = journal
            .checked_add(buffers)
            .context("M2b budget overflow")?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| needed <= n),
            "M2b attachment exceeds memory budget: required={needed} free={free}"
        );
        for rows in 1..=4 {
            self.prepare(rows)?;
            self.blas
                .prepare_fixed(rows as u32, VOCAB as u32, HIDDEN as u32)?;
        }
        let mut batch = Batch {
            logits: s.alloc_zeros(4 * VOCAB)?,
            predictions: s.alloc_zeros(4)?,
            row_counts: s.clone_htod(&[1, 2, 3, 4])?,
            meta: cubin::module(&self.device.ctx, "q38_common")?.load_function("q38_batch_meta")?,
            common: CommonOpsRef::new(&self.device.ctx)?,
            capture_checkpoint: StateCheckpoint::new(&self.state, &self.model_spec, 4, &s)?,
            graphs: Vec::new(),
            frame: Vec::new(),
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
        batch
            .capture_checkpoint
            .save(&self.state, &self.work.token, 0, &s)?;
        let logits = s.clone_dtoh(&self.work.logits)?;
        let result = (|| -> Result<Vec<CudaGraph>> {
            let mut graphs = Vec::new();
            for rows in 1..=4 {
                if rows > 1 {
                    self.closure.capture()?;
                }
                self.mtp_batch_launch(&mut batch, rows)?;
                s.synchronize()?;
                batch
                    .capture_checkpoint
                    .restore(&mut self.state, &mut self.work.token, &s)?;
                s.memcpy_htod(&logits, &mut self.work.logits)?;
                s.synchronize()?;
                #[cfg(test)]
                ensure!(
                    self.capture_fail_stage != Some(super::super::super::CaptureFault::Begin),
                    "injected batch capture begin failure"
                );
                s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let launches = self.mtp_batch_launch(&mut batch, rows);
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
            s.synchronize()?;
            Ok(())
        })();
        if let Err(error) = restored {
            self.poisoned = true;
            return Err(error.context("M2b capture restore failed"));
        }
        match result {
            Ok(graphs) => batch.graphs = graphs,
            Err(error) => {
                self.poisoned = true;
                return Err(error.context("M2b capture transaction failed"));
            }
        }
        self.mtp.as_mut().context("MTP missing")?.batch = Some(batch);
        eprintln!(
            "MTP_M2B_MEMORY additional_device_bytes={needed} capture_checkpoint_bytes={journal} graph_shapes=1,2,3,4"
        );
        Ok(())
    }

    /// Capture-safe: no host transfers/allocations, all plans prebound. Metadata
    /// is generated from the device position before Q/K scatter, GDN and FA2.
    fn mtp_batch_launch(&mut self, batch: &mut Batch, rows: usize) -> Result<()> {
        ensure!((1..=4).contains(&rows), "invalid M2b query rows");
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
                .arg(&mut batch.predictions)
                .arg(&(VOCAB as i32))
                .arg(&batch.row_counts.slice(rows - 1..rows))
                .launch(grid(rows, 1024))?;
        }
        s.memcpy_dtod(
            &batch.logits.slice((rows - 1) * VOCAB..rows * VOCAB),
            &mut self.work.logits,
        )?;
        s.memcpy_dtod(
            &batch.predictions.slice(rows - 1..rows),
            &mut self.work.token,
        )?;
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
            chain.committed_position = self.position;
            Ok(r)
        })();
        self.mtp = Some(chain);
        result
    }
}

impl BatchedBackend for Bound<'_> {
    fn verify_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let rows = inputs.len();
        ensure!(
            (1..=4).contains(&rows)
                && e.position + rows < e.capacity
                && inputs.iter().all(|&t| (t as usize) < VOCAB),
            "invalid M2b batch extent/IDs"
        );
        let s = e.device.stream.clone();
        let batch = c.batch.as_mut().context("M2b not bound")?;
        s.memcpy_htod(inputs, &mut e.work.ids.slice_mut(..rows))?;
        s.memcpy_htod(&[e.position as i32], &mut e.work.position)?;
        if c.options.graph {
            batch
                .graphs
                .get(rows - 1)
                .context("M2b graph missing")?
                .launch()?;
        } else {
            e.mtp_batch_launch(batch, rows)?;
        }
        let predictions = s.clone_dtoh(&batch.predictions.slice(..rows))?;
        let logits = s.clone_dtoh(&batch.logits.slice(..rows * VOCAB))?;
        ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite M2b head");
        let hidden = match c.options.hidden_tap {
            MtpHiddenTap::PreFinalNorm => &e.work.x,
            MtpHiddenTap::PostFinalNorm => &e.work.n,
        };
        batch.frame = s.clone_dtoh(&hidden.slice(..rows * HIDDEN))?;
        batch.frame_position = e.position;
        batch.frame_rows = rows;
        batch.calls += 1;
        e.position += rows;
        s.memcpy_htod(&[e.position as i32], &mut e.work.position)?;
        s.memcpy_htod(&[(e.position + 1) as i32], &mut e.work.length)?;
        e.state
            .set_sequence(&s, e.position as u32, 1, e.workspace_rows as u32)?;
        Ok(predictions)
    }
    fn restore_mtp(&mut self) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let saved = c.saved.as_ref().context("MTP checkpoint invalid")?;
        c.mtp_checkpoint.restore(
            e.mtp_state.as_mut().context("MTP state missing")?,
            &mut c.runtime.work.token,
            &e.device.stream,
        )?;
        c.cursor = saved.cursor;
        c.recurrent = saved.recurrent.clone();
        c.batch.as_mut().context("M2b not bound")?.mtp_restores += 1;
        Ok(())
    }
    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()> {
        let saved = self
            .chain
            .saved
            .as_ref()
            .context("MTP checkpoint invalid")?
            .clone();
        let batch = self.chain.batch.as_ref().context("M2b not bound")?;
        ensure!(
            batch.frame_position == saved.position
                && batch.frame_rows == inputs.len()
                && self.engine.position == saved.position + inputs.len()
                && self.chain.cursor == saved.cursor,
            "M2b repair frame/partition mismatch"
        );
        let frame = batch.frame.clone();
        // Independent canonical recurrence over committed target hidden rows.
        // Speculative draft hidden is discarded; target state is untouched.
        for (row, &token) in inputs.iter().enumerate() {
            let hidden = if row == 0 {
                &saved.last_target
            } else {
                &frame[(row - 1) * HIDDEN..row * HIDDEN]
            };
            self.row(token, hidden)?;
            #[cfg(test)]
            ensure!(
                self.chain
                    .batch
                    .as_ref()
                    .context("M2b not bound")?
                    .repair_fail_after
                    != Some(row),
                "injected M2b canonical row failure {row}"
            );
        }
        self.chain.last_target = frame[(inputs.len() - 1) * HIDDEN..].to_vec();
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
