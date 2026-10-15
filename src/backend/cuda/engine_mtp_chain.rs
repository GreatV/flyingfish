//! M2a correctness baseline. Both stores belong to the same Engine; the
//! controller uses its ordinary target decode, including the existing graph.
use super::super::Engine;
use super::*;
use crate::{
    backend::cuda::state::StateCheckpoint,
    mtp::{MtpOptions, transaction::Backend},
};

#[cfg(test)]
use crate::mtp::transaction::{self, Fault, Round};

pub(crate) struct Chain {
    options: MtpOptions,
    runtime: MtpRuntime,
    catalog: WeightCatalog,
    target_checkpoint: StateCheckpoint,
    mtp_checkpoint: StateCheckpoint,
    saved: Saved,
    last_target: CudaSlice<bf16>,
    recurrent: CudaSlice<bf16>,
    pending: u32,
    cursor: usize,
    committed_position: usize,
    epoch: u64,
    #[cfg(test)]
    force_proposals: Option<Vec<u32>>,
    batch: Option<batched::Batch>,
    bootstrap: Bootstrap,
}

struct Bootstrap {
    target: CudaSlice<bf16>,
    pairs: CudaSlice<bf16>,
    carry: CudaSlice<bf16>,
    last_rows: usize,
}
impl Bootstrap {
    fn new(stream: &Arc<CudaStream>, rows: usize) -> Result<Self> {
        Ok(Self {
            target: stream.alloc_zeros(rows * HIDDEN)?,
            pairs: stream.alloc_zeros(rows * HIDDEN)?,
            carry: stream.alloc_zeros(HIDDEN)?,
            last_rows: 0,
        })
    }
    fn bytes(&self) -> usize {
        (self.target.len() + self.pairs.len() + self.carry.len()) * size_of::<bf16>()
    }
}
impl Chain {
    pub(in crate::backend::cuda::engine) fn invalidate_verify(&mut self) {
        if let Some(batch) = self.batch.as_mut() {
            batch.invalidate_frame();
        }
    }
    fn bootstrap_chunk(&mut self, e: &mut Engine, ids: &[u32]) -> Result<()> {
        let rows = ids.len();
        let s = &e.device.stream;
        let b = &mut self.bootstrap;
        match self.options.hidden_tap {
            MtpHiddenTap::PreFinalNorm => s.memcpy_dtod(
                &e.work.x.slice(..rows * HIDDEN),
                &mut b.target.slice_mut(..rows * HIDDEN),
            )?,
            MtpHiddenTap::PostFinalNorm => self.runtime.common.norm.rms_with(
                s,
                RowDomain {
                    meta: e.state.seq_ref(s)?,
                    row_base: 0,
                    row_capacity: rows as u32,
                },
                heads(&e.work.x, s, 0, rows, 1, HIDDEN as u32)?,
                vector(&e.weights.norm, s)?,
                &mut TensorMut(heads(&b.target, s, 0, rows, 1, HIDDEN as u32)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: 1e-6,
                },
            )?,
        }
        let first = e.position == 0;
        let inputs = if first { &ids[1..] } else { ids };
        let paired = inputs.len();
        if paired > 0 {
            let offset = usize::from(!first) * HIDDEN;
            if !first {
                s.memcpy_dtod(&b.carry, &mut b.pairs.slice_mut(..HIDDEN))?;
            }
            if rows > 1 {
                s.memcpy_dtod(
                    &b.target.slice(..(rows - 1) * HIDDEN),
                    &mut b.pairs.slice_mut(offset..offset + (rows - 1) * HIDDEN),
                )?;
            }
            let state = e.mtp_state.as_mut().context("MTP state missing")?;
            self.runtime.run_prefill(
                Execution {
                    device: &e.device,
                    ops: &e.ops,
                    blas: &e.blas,
                    catalog: &self.catalog,
                    embedding: &e.weights.embed,
                    head: &e.weights.head,
                },
                state,
                self.cursor,
                inputs,
                matrix(&b.pairs, s, 0, paired, HIDDEN)?,
            )?;
            self.cursor += paired;
            state.set_sequence(s, self.cursor as u32, 1, 1)?;
            b.last_rows = paired;
        }
        s.memcpy_dtod(
            &b.target.slice((rows - 1) * HIDDEN..rows * HIDDEN),
            &mut b.carry,
        )?;
        Ok(())
    }
}

struct Saved {
    position: Option<usize>,
    work_position: CudaSlice<i32>,
    work_length: CudaSlice<i32>,
    logits: CudaSlice<bf16>,
    last_target: CudaSlice<bf16>,
    recurrent: CudaSlice<bf16>,
    pending: u32,
    cursor: usize,
}

struct Bound<'a> {
    engine: &'a mut Engine,
    chain: &'a mut Chain,
}

impl Bound<'_> {
    fn row(
        &mut self,
        input: u32,
        hidden: Tensor<bf16, 2>,
        mode: OutputMode,
    ) -> Result<Option<u32>> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let state = e.mtp_state.as_mut().context("MTP state missing")?;
        let s = &e.device.stream;
        s.memcpy_htod(&[input], &mut c.runtime.work.ids.slice_mut(..1))?;
        let ids = view(&c.runtime.work.ids, s, 0, [1], [1])?;
        s.memset_zeros(&mut c.runtime.prediction)?;
        let flag = view(&c.runtime.prediction, s, 0, [2], [1])?;
        c.runtime.run_device(
            Execution {
                device: &e.device,
                ops: &e.ops,
                blas: &e.blas,
                catalog: &c.catalog,
                embedding: &e.weights.embed,
                head: &e.weights.head,
            },
            state,
            c.cursor,
            ((ids, 0), hidden, flag),
            ExecutionPhase::Decode,
            if mode == OutputMode::StateOnly {
                OutputMode::Repair
            } else {
                OutputMode::Queued
            },
        )?;
        let output = match c.options.recurrent_tap {
            MtpRecurrentTap::PreMtpNorm => &c.runtime.work.x,
            MtpRecurrentTap::PostMtpNorm => &c.runtime.work.n,
        };
        let flag = view(&c.runtime.prediction, s, 0, [2], [1])?;
        e.ops
            .finite(s, matrix(&c.runtime.work.n, s, 0, 1, HIDDEN)?, flag)?;
        e.ops.finite(s, matrix(output, s, 0, 1, HIDDEN)?, flag)?;
        if mode != OutputMode::StateOnly {
            e.ops
                .finite(s, matrix(&c.runtime.logits, s, 0, 1, VOCAB)?, flag)?;
        }
        s.memcpy_dtod(&output.slice(..HIDDEN), &mut c.recurrent)?;
        let status = s.clone_dtoh(&c.runtime.prediction)?;
        ensure!(status[1] == 0, "nonfinite MTP recurrent hidden or logits");
        c.cursor += 1;
        state.set_sequence(&e.device.stream, c.cursor as u32, 1, 1)?;
        if mode == OutputMode::StateOnly {
            return Ok(None);
        }
        ensure!((status[0] as usize) < VOCAB, "invalid MTP argmax ID");
        Ok(Some(status[0]))
    }
}

impl Backend for Bound<'_> {
    fn position(&self) -> usize {
        self.engine.position
    }
    fn capacity(&self) -> usize {
        // Decode advances the next-row metadata too. Retain that final slot.
        self.engine.capacity - 1
    }
    fn pending(&self) -> Result<u32> {
        Ok(self.chain.pending)
    }
    fn save(&mut self, rows: usize) -> Result<()> {
        let e = &self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        ensure!(
            rows <= 4 && e.position > 0 && c.cursor == e.position - 1,
            "MTP checkpoint cursor mismatch"
        );
        c.saved.position = None;
        let validation = if let Some(batch) = c.batch.as_mut().filter(|_| rows > 1) {
            s.memset_zeros(&mut batch.status)?;
            Some((&batch.sequence_check, view(&batch.status, s, 0, [8], [1])?))
        } else {
            None
        };
        c.target_checkpoint
            .save(&e.state, &e.work.token, e.position, s, validation)?;
        c.mtp_checkpoint.save(
            e.mtp_state.as_ref().context("MTP state missing")?,
            &c.runtime.work.token,
            c.cursor,
            s,
            validation,
        )?;
        let saved = &mut c.saved;
        s.memcpy_dtod(&e.work.position, &mut saved.work_position)?;
        s.memcpy_dtod(&e.work.length, &mut saved.work_length)?;
        s.memcpy_dtod(&e.work.logits, &mut saved.logits)?;
        s.memcpy_dtod(&c.last_target, &mut saved.last_target)?;
        s.memcpy_dtod(&c.recurrent, &mut saved.recurrent)?;
        saved.pending = c.pending;
        saved.cursor = c.cursor;
        saved.position = Some(e.position);
        Ok(())
    }
    fn restore(&mut self, replay: bool) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        let saved = &c.saved;
        let position = saved.position.context("MTP checkpoint not valid")?;
        c.target_checkpoint
            .restore(&mut e.state, &mut e.work.token, s)?;
        c.mtp_checkpoint.restore(
            e.mtp_state.as_mut().context("MTP state missing")?,
            &mut c.runtime.work.token,
            s,
        )?;
        s.memcpy_dtod(&saved.work_position, &mut e.work.position)?;
        s.memcpy_dtod(&saved.work_length, &mut e.work.length)?;
        s.memcpy_dtod(&saved.logits, &mut e.work.logits)?;
        s.synchronize()?;
        e.position = position;
        if replay {
            e.state.compact_defer(s)?;
        }
        e.logits_cache.mark_valid();
        e.poisoned = false;
        s.memcpy_dtod(&saved.last_target, &mut c.last_target)?;
        s.memcpy_dtod(&saved.recurrent, &mut c.recurrent)?;
        c.pending = saved.pending;
        c.cursor = saved.cursor;
        c.committed_position = position;
        if let Some(batch) = c.batch.as_mut() {
            batch.target_restores += 1;
            batch.invalidate_frame();
        }
        Ok(())
    }
    fn poison(&mut self) {
        self.engine.poisoned = true;
    }
    fn draft(&mut self, input: u32, depth: usize) -> Result<u32> {
        let hidden = if depth == 0 {
            matrix(
                &self.chain.last_target,
                &self.engine.device.stream,
                0,
                1,
                HIDDEN,
            )?
        } else {
            matrix(
                &self.chain.recurrent,
                &self.engine.device.stream,
                0,
                1,
                HIDDEN,
            )?
        };
        let token = self
            .row(input, hidden, OutputMode::Predict)?
            .context("MTP prediction missing")?;
        #[cfg(test)]
        let token = self
            .chain
            .force_proposals
            .as_ref()
            .and_then(|v| v.get(depth))
            .copied()
            .unwrap_or(token);
        Ok(token)
    }
    fn target_decode(&mut self, input: u32) -> Result<u32> {
        self.engine.set_token(input)?;
        let token = self.engine.decode(self.chain.options.graph, None)?;
        let hidden = match self.chain.options.hidden_tap {
            MtpHiddenTap::PreFinalNorm => &self.engine.work.x,
            MtpHiddenTap::PostFinalNorm => &self.engine.work.n,
        };
        self.engine
            .device
            .stream
            .memcpy_dtod(&hidden.slice(..HIDDEN), &mut self.chain.last_target)?;
        self.chain.pending = token;
        Ok(token)
    }
    fn canonical_row(&mut self, input: u32) -> Result<()> {
        ensure!(
            self.chain.cursor + 1 == self.engine.position,
            "MTP canonical position mismatch"
        );
        let hidden = matrix(
            &self.chain.last_target,
            &self.engine.device.stream,
            0,
            1,
            HIDDEN,
        )?;
        self.row(input, hidden, OutputMode::StateOnly)?;
        Ok(())
    }
}

/// The MTP row sizes bound before Ready: repair and head rows (1..=16) plus every bucketed tail-chunk size.
fn mtp_bound_rows(chunk: usize) -> std::collections::BTreeSet<usize> {
    let mut rows: std::collections::BTreeSet<usize> = (1..=chunk.min(16)).collect();
    let mut bucket = 32usize;
    while chunk > 16 && bucket < chunk {
        rows.insert(bucket);
        bucket *= 2;
    }
    rows.insert(chunk);
    rows
}

impl Engine {
    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn mtp_enable(
        &mut self,
        options: MtpOptions,
    ) -> Result<()> {
        self.mtp_enable_bound(options)
    }
    pub(in crate::backend::cuda::engine) fn check_mtp_mode(&self, graph: bool) -> Result<()> {
        ensure!(
            self.mtp
                .as_ref()
                .context("MTP not configured")?
                .options
                .graph
                == graph,
            "MTP graph mode differs from setup"
        );
        Ok(())
    }
    pub(crate) fn reset_mtp(&mut self) -> Result<()> {
        let mut chain = self.mtp.take().context("MTP not configured")?;
        let result = (|| -> Result<()> {
            self.reset()?;
            self.mtp_state
                .as_mut()
                .context("MTP state missing")?
                .reset(&self.device.stream)?;
            chain.cursor = 0;
            chain.committed_position = 0;
            chain.epoch = self.mtp_epoch;
            chain.saved.position = None;
            self.device.stream.memset_zeros(&mut chain.last_target)?;
            self.device.stream.memset_zeros(&mut chain.recurrent)?;
            chain.pending = 0;
            chain.bootstrap.last_rows = 0;
            self.device
                .stream
                .memset_zeros(&mut chain.bootstrap.carry)?;
            if let Some(batch) = chain.batch.as_mut() {
                batch.invalidate_frame();
            }
            self.device.stream.synchronize()?;
            Ok(())
        })();
        self.mtp = Some(chain);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub(in crate::backend::cuda::engine) fn mtp_enable_resident(
        &mut self,
        options: MtpOptions,
    ) -> Result<()> {
        self.mtp_enable_bound(options)
    }
    fn mtp_enable_bound(&mut self, options: MtpOptions) -> Result<()> {
        options.validate()?;
        ensure!(
            self.qwen.is_some()
                && self.position == 0
                && self.images.is_none()
                && self.mtp.is_none(),
            "MTP setup requires an empty Qwen text Engine"
        );
        let plan = MtpPlan {
            hidden_tap: options.hidden_tap,
            recurrent_tap: options.recurrent_tap,
            max_rows: self.chunk,
            capacity: self.capacity,
            phase: ExecutionPhase::Prefill,
        };
        let needed =
            crate::mtp::attachment_bytes(&self.model_spec, self.capacity, self.chunk, false)?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| needed <= n),
            "MTP attachment exceeds memory budget: required={needed} free={free}"
        );
        let dir = self
            .qwen
            .as_ref()
            .context("Qwen runtime missing")?
            .checkpoint_path();
        let catalog = WeightCatalog::load_with_policy(
            dir,
            &self.model_spec,
            &self.device,
            LoadPolicy::MtpOnly,
        )?;
        ensure!(
            catalog.bytes == crate::mtp::WEIGHT_BYTES && catalog.tensors.len() == 22,
            "MTP-only ledger mismatch"
        );
        let spec = crate::mtp::component_spec(&self.model_spec);
        let state = StateStore::new(&spec, self.capacity, &self.device.stream)?;
        let mut runtime = MtpRuntime::allocate(
            &self.device,
            &self.model_spec,
            &catalog,
            &state,
            &self.blas,
            plan,
            1,
        )?;
        for rows in mtp_bound_rows(self.chunk) {
            runtime.prepare_rows(&self.device, &catalog, &state, &self.blas, rows)?;
        }
        let bootstrap = Bootstrap::new(&self.device.stream, self.chunk)?;
        let target_checkpoint =
            StateCheckpoint::new(&self.state, &self.model_spec, 4, &self.device.stream)?;
        let mtp_checkpoint = StateCheckpoint::new(&state, &spec, 4, &self.device.stream)?;
        let journal = target_checkpoint.bytes() + mtp_checkpoint.bytes();
        ensure!(
            runtime.allocation_bytes()
                + state.bytes()
                + catalog.bytes
                + journal
                + bootstrap.bytes()
                + (4 * HIDDEN + VOCAB) * size_of::<bf16>()
                + 8
                == needed,
            "MTP attachment allocation/budget mismatch"
        );
        self.device.stream.synchronize()?;
        self.mtp_state = Some(state);
        self.mtp = Some(Chain {
            options,
            runtime,
            catalog,
            target_checkpoint,
            mtp_checkpoint,
            saved: Saved {
                position: None,
                cursor: 0,
                pending: 0,
                work_position: self.device.stream.alloc_zeros(1)?,
                work_length: self.device.stream.alloc_zeros(1)?,
                logits: self.device.stream.alloc_zeros(VOCAB)?,
                last_target: self.device.stream.alloc_zeros(HIDDEN)?,
                recurrent: self.device.stream.alloc_zeros(HIDDEN)?,
            },
            last_target: self.device.stream.alloc_zeros(HIDDEN)?,
            recurrent: self.device.stream.alloc_zeros(HIDDEN)?,
            pending: 0,
            cursor: 0,
            committed_position: 0,
            epoch: self.mtp_epoch,
            #[cfg(test)]
            force_proposals: None,
            batch: None,
            bootstrap,
        });
        eprintln!(
            "MTP_M2A_MEMORY additional_device_bytes={needed} checkpoint_bytes={journal} shared_weights_duplicated=0"
        );
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn mtp_prefill(&mut self, ids: &[u32]) -> Result<u32> {
        ensure!(
            !ids.is_empty() && ids.len() < self.capacity,
            "MTP prefill needs room for decode"
        );
        let mut chain = self.mtp.take().context("MTP not enabled")?;
        let result = (|| -> Result<u32> {
            self.reset()?;
            self.mtp_state
                .as_mut()
                .context("MTP state missing")?
                .reset(&self.device.stream)?;
            chain.cursor = 0;
            chain.committed_position = 0;
            chain.saved.position = None;
            #[cfg(test)]
            {
                chain.force_proposals = None;
            }
            self.device.stream.memset_zeros(&mut chain.last_target)?;
            self.device.stream.memset_zeros(&mut chain.recurrent)?;
            chain.bootstrap.last_rows = 0;
            if let Some(batch) = chain.batch.as_mut() {
                batch.invalidate_frame();
            }
            let token = self.prefill_chunks(ids, None, ids.len(), |engine, chunk| {
                chain.bootstrap_chunk(engine, chunk)
            })?;
            self.device
                .stream
                .memcpy_dtod(&chain.bootstrap.carry, &mut chain.last_target)?;
            if chain.bootstrap.last_rows > 0 {
                let row = chain.bootstrap.last_rows - 1;
                let hidden = match chain.options.recurrent_tap {
                    MtpRecurrentTap::PreMtpNorm => &chain.runtime.work.x,
                    MtpRecurrentTap::PostMtpNorm => &chain.runtime.work.n,
                };
                self.device.stream.memcpy_dtod(
                    &hidden.slice(row * HIDDEN..(row + 1) * HIDDEN),
                    &mut chain.recurrent,
                )?;
            }
            let s = &self.device.stream;
            s.memset_zeros(&mut chain.runtime.prediction)?;
            let flag = view(&chain.runtime.prediction, s, 0, [2], [1])?;
            for buffer in [&chain.last_target, &chain.recurrent] {
                self.ops.finite(s, matrix(buffer, s, 0, 1, HIDDEN)?, flag)?;
            }
            ensure!(
                s.clone_dtoh(&chain.runtime.prediction)?[1] == 0,
                "nonfinite MTP bootstrap boundary"
            );
            chain.pending = token;
            ensure!(
                chain.cursor + 1 == self.position,
                "MTP bootstrap prefix mismatch"
            );
            chain.committed_position = self.position;
            chain.epoch = self.mtp_epoch;
            Ok(token)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        self.mtp = Some(chain);
        result
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn mtp_round(
        &mut self,
        remaining: usize,
        #[cfg(test)] fault: Option<Fault>,
    ) -> Result<Round> {
        ensure!(!self.poisoned, "MTP requires healthy Engine");
        let mut chain = self.mtp.take().context("MTP not enabled")?;
        let result = (|| -> Result<Round> {
            ensure!(
                chain.committed_position == self.position
                    && self.position > 0
                    && chain.epoch == self.mtp_epoch,
                "MTP prefix changed outside its controller"
            );
            let eos = self.model_spec.eos.clone();
            let options = chain.options;
            let round = transaction::run(
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
            Ok(round)
        })();
        self.mtp = Some(chain);
        result
    }
}

#[cfg(test)]
#[path = "engine_mtp_chain_tests.rs"]
pub(super) mod tests;

#[cfg(test)]
mod bound_rows_tests {
    use super::super::bucketed_prefill_rows;
    use super::mtp_bound_rows;
    use anyhow::Result;

    #[test]
    fn the_bound_set_is_the_bucket_closure() {
        let set = |rows: &[usize]| {
            rows.iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(mtp_bound_rows(4), set(&[1, 2, 3, 4]));
        assert_eq!(mtp_bound_rows(16), set(&(1..=16).collect::<Vec<_>>()));
        assert_eq!(
            mtp_bound_rows(2048),
            set(&[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024,
                2048
            ])
        );
        assert_eq!(
            mtp_bound_rows(3000),
            set(&[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024,
                2048, 3000
            ])
        );
    }

    #[test]
    fn every_request_row_maps_into_the_bound_set() -> Result<()> {
        for chunk in [4usize, 16, 3000, 2048] {
            let bound = mtp_bound_rows(chunk);
            for rows in 1..=chunk {
                let bucket = bucketed_prefill_rows(rows as u32, chunk as u32)? as usize;
                assert!(
                    bound.contains(&bucket),
                    "chunk {chunk} rows {rows} map to unbound bucket {bucket}"
                );
            }
            assert!(
                bucketed_prefill_rows(chunk as u32 + 1, chunk as u32).is_err(),
                "rows beyond the chunk must be rejected"
            );
        }
        Ok(())
    }
}

#[path = "engine_mtp_batched.rs"]
mod batched;
