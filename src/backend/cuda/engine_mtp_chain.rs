//! Both stores belong to the same Engine; the controller uses its ordinary target decode and graph.
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
            (1..=crate::mtp::MAX_VERIFY_ROWS).contains(&rows),
            "MTP checkpoint rows {rows} exceed journal capacity {}",
            crate::mtp::MAX_VERIFY_ROWS
        );
        ensure!(
            e.position > 0 && c.cursor == e.position - 1,
            "MTP checkpoint cursor mismatch"
        );
        c.saved.position = None;
        let validation = if let Some(batch) = c.batch.as_mut().filter(|_| rows > 1) {
            s.memset_zeros(&mut batch.status)?;
            Some((&batch.sequence_check, view(&batch.status, s, 0, [10], [1])?))
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
            StateCheckpoint::new(&self.state, &self.model_spec, 5, &self.device.stream)?;
        let mtp_checkpoint = StateCheckpoint::new(&state, &spec, 5, &self.device.stream)?;
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
pub(super) mod tests {
    use super::*;
    use crate::backend::{Settings, SpecBudget, TreeBuilder};
    use std::path::PathBuf;

    pub(super) fn load(write: bool) -> Result<Engine> {
        load_capacity(write, 128)
    }
    pub(super) fn load_capacity(write: bool, capacity: usize) -> Result<Engine> {
        load_format(write, capacity, crate::kv::KvFormat::Bf16)
    }
    pub(super) fn load_format(
        write: bool,
        capacity: usize,
        format: crate::kv::KvFormat,
    ) -> Result<Engine> {
        ensure!(
            std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
            "M2a requires a separate full-model cloud window"
        );
        let dir = PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("FF_Q38_MODEL missing")?);
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        );
        Engine::load_kind(
            &dir,
            0,
            capacity,
            Some(16),
            Settings {
                tree_builder: TreeBuilder::Waves,
                spec_budget: SpecBudget::Chain,
                spec_graph: false,
                draft_model: None,
                runtime_dir: Some(runtime),
                prefix_cache: false,
                kv_format: format,
            },
            write,
        )
    }
    pub(super) fn options(
        k: usize,
        h2: MtpHiddenTap,
        h5: MtpRecurrentTap,
        graph: bool,
    ) -> MtpOptions {
        MtpOptions {
            depth: k,
            hidden_tap: h2,
            recurrent_tap: h5,
            graph,
        }
    }
    pub(super) fn configure(e: &mut Engine, o: MtpOptions) -> Result<()> {
        e.reset()?;
        if let Some(c) = e.mtp.as_mut() {
            o.validate()?;
            c.options = o;
            c.runtime.plan.hidden_tap = o.hidden_tap;
            c.runtime.plan.recurrent_tap = o.recurrent_tap;
        } else {
            e.mtp_enable(o)?;
        }
        Ok(())
    }
    pub(super) fn prompt(case: &str) -> Result<Vec<u32>> {
        let root = PathBuf::from(
            std::env::var_os("FF_Q38_MTP_REF_ROOT").context("FF_Q38_MTP_REF_ROOT missing")?,
        );
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join(format!("inputs/{case}.input_ids.json")),
        )?)?;
        Ok(serde_json::from_value(v["input_ids"].clone())?)
    }
    #[test]
    #[ignore = "exclusive full Qwen model, text_2k/text_16k fixtures and calibrated runtime"]
    fn q38_dense_layer_prefill_gpu() -> Result<()> {
        use crate::backend::cuda::{
            contracts::{RowDomain, TensorMut},
            ops::{CommonOpsRef, RmsMode, RmsOptions},
        };
        use safetensors::{Dtype, SafeTensors};
        ensure!(
            std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
            "full-model GPU window not acknowledged"
        );
        let directory = PathBuf::from(std::env::var("FF_Q38_MODEL")?);
        let fixtures = PathBuf::from(std::env::var("FF_MINICPM_REFERENCE_DIR")?);
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let signature = |e: &Engine, mtp: bool| -> Result<Vec<u8>> {
            let mut bits = Vec::new();
            for (store, prefix) in std::iter::once((&e.state, e.position)).chain(
                mtp.then(|| -> Result<_> {
                    Ok((
                        e.mtp_state.as_ref().context("MTP state missing")?,
                        e.mtp.as_ref().context("MTP missing")?.cursor,
                    ))
                })
                .transpose()?,
            ) {
                for gdn in &store.gdn {
                    for buffer in [&gdn.s, &gdn.history] {
                        let values = e.device.stream.clone_dtoh(buffer)?;
                        ensure!(
                            values.iter().all(|v| v.is_finite()),
                            "Runner FAIL nonfinite GDN state"
                        );
                        bits.extend(values.into_iter().flat_map(|v| v.to_bits().to_le_bytes()));
                    }
                }
                for kv in &store.kv {
                    for buffer in [&kv.k, &kv.v] {
                        for head in 0..e.model_spec.kv_heads as usize {
                            let dim = e.model_spec.head_dim as usize;
                            let values =
                                e.device.stream.clone_dtoh(&buffer.slice(
                                    head * e.capacity * dim..(head * e.capacity + prefix) * dim,
                                ))?;
                            ensure!(
                                values.iter().all(|v| v.to_f32().is_finite()),
                                "Runner FAIL nonfinite live KV"
                            );
                            bits.extend(values.into_iter().flat_map(|v| v.to_bits().to_le_bytes()));
                        }
                    }
                }
                for value in e
                    .device
                    .stream
                    .clone_dtoh(store.sequence.as_ref().context("sequence missing")?)?
                {
                    bits.extend_from_slice(&value.to_le_bytes());
                }
            }
            if mtp {
                let chain = e.mtp.as_ref().context("MTP missing")?;
                for buffer in [&chain.last_target, &chain.recurrent, &chain.bootstrap.carry] {
                    let values = e.device.stream.clone_dtoh(buffer)?;
                    ensure!(
                        values.iter().all(|v| v.to_f32().is_finite()),
                        "Runner FAIL nonfinite bootstrap hidden"
                    );
                    bits.extend(values.into_iter().flat_map(|v| v.to_bits().to_le_bytes()));
                }
                for value in [
                    chain.cursor,
                    chain.committed_position,
                    chain.bootstrap.last_rows,
                ] {
                    bits.extend_from_slice(&value.to_le_bytes());
                }
                bits.extend_from_slice(&chain.pending.to_le_bytes());
            }
            bits.extend_from_slice(&e.position.to_le_bytes());
            bits.extend_from_slice(&e.token()?.to_le_bytes());
            for value in e.decode_logits()? {
                ensure!(
                    value.is_finite(),
                    "Runner FAIL nonfinite dense prefill logits"
                );
                bits.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            Ok(bits)
        };
        let mut defects = 0;
        for (case, length) in [
            ("text_2k", 1000usize),
            ("text_2k", 2100),
            ("text_16k", 16320),
        ] {
            let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
            let source = SafeTensors::deserialize(&bytes)?;
            let tensor = source.tensor("input_ids")?;
            ensure!(
                tensor.dtype() == Dtype::I32
                    && tensor.shape() == [1, if case == "text_2k" { 2100 } else { 16320 }],
                "Runner FAIL dense prefill fixture layout"
            );
            let ids = tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .take(length)
                .map(|x| Ok(u32::try_from(i32::from_le_bytes(*x))?))
                .collect::<Result<Vec<_>>>()?;
            let capacity = (length + 17).next_multiple_of(32);
            let mut e = Engine::load_kind(&directory, 0, capacity, None, settings.clone(), true)?;
            ensure!(
                e.model_spec.hidden == 5120
                    && e.model_spec.layers.len() == 64
                    && e.model_spec.vocab == 248320
                    && e.model_spec.moe.is_none()
                    && e.model_spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Block128,
                "Runner FAIL dense prefill model geometry/format"
            );
            ensure!(
                ids.len() == length && ids.iter().all(|&id| id < e.model_spec.vocab),
                "Runner FAIL dense prefill input extent/vocabulary"
            );
            let common = CommonOpsRef::new(&e.device.ctx)?;
            e.mtp_enable(options(
                4,
                MtpHiddenTap::PostFinalNorm,
                MtpRecurrentTap::PostMtpNorm,
                false,
            ))?;
            e.closure.ready(std::time::Instant::now())?;
            let mut pool = e
                .qwen
                .as_mut()
                .context("Qwen runtime missing")?
                .prompt
                .take();
            ensure!(
                pool.is_some() == (length > e.chunk),
                "Runner FAIL expected dense schedule eligibility"
            );
            if length == 2100 {
                ensure!(
                    e.dense_prefill_chunk(&ids, 0, e.chunk, 0).is_err(),
                    "out-of-order predecessor was not rejected"
                );
                let mut ordered = Vec::new();
                for reverse in [false, true] {
                    e.reset()?;
                    for kv in &mut e.state.kv {
                        e.device.stream.memset_zeros(&mut kv.k)?;
                        e.device.stream.memset_zeros(&mut kv.v)?;
                    }
                    let mut starts: Vec<_> = (0..ids.len()).step_by(e.chunk).collect();
                    if reverse {
                        starts.reverse();
                    }
                    for start in starts {
                        let end = e.dense_prefill_chunk(&ids, 0, start, start)?;
                        e.run_step(
                            crate::backend::Step {
                                rows: end - start,
                                input: crate::backend::Input::Ids,
                                head: crate::backend::Head::None,
                                mask: crate::backend::Mask::Causal,
                            },
                            None,
                        )?;
                    }
                    let values = e
                        .state
                        .gdn
                        .iter()
                        .map(|g| e.device.stream.clone_dtoh(&g.s))
                        .collect::<std::result::Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>();
                    ensure!(
                        values.iter().all(|v| v.is_finite()),
                        "Runner FAIL nonfinite order-control GDN state"
                    );
                    let values: Vec<_> = values.into_iter().map(f32::to_bits).collect();
                    if reverse {
                        ensure!(
                            values != ordered,
                            "Runner FAIL reversed chunks did not perturb GDN state"
                        );
                    } else {
                        ordered = values;
                    }
                }
                println!(
                    "PASS dense_prefill_order_control length=2100 predecessor_rejected=true bypass_changes_gdn_state=true"
                );
            }
            for mtp in [false, true] {
                let mut reference = None;
                for candidate in [false, true] {
                    if candidate {
                        e.qwen.as_mut().context("Qwen runtime missing")?.prompt = pool.take();
                    }
                    e.reset()?;
                    let mut norms = Vec::new();
                    let token = if mtp {
                        e.mtp_prefill(&ids)?
                    } else {
                        e.prefill_chunks(&ids, None, ids.len(), |engine, part| {
                            let stream = &engine.device.stream;
                            let rows = part.len();
                            common.norm.rms_with(
                                stream,
                                RowDomain {
                                    meta: engine.state.seq_ref(stream)?,
                                    row_base: 0,
                                    row_capacity: rows as u32,
                                },
                                heads(&engine.work.x, stream, 0, rows, 1, HIDDEN as u32)?,
                                vector(&engine.weights.norm, stream)?,
                                &mut TensorMut(heads(
                                    &engine.work.n,
                                    stream,
                                    0,
                                    rows,
                                    1,
                                    HIDDEN as u32,
                                )?),
                                RmsOptions {
                                    mode: RmsMode::ZeroCentered,
                                    eps: engine.model_spec.rms_eps,
                                },
                            )?;
                            let mut bits = Vec::new();
                            bits.extend_from_slice(&engine.position.to_le_bytes());
                            for buffer in [&engine.work.x, &engine.work.n] {
                                let values = stream.clone_dtoh(&buffer.slice(..rows * HIDDEN))?;
                                ensure!(
                                    values.iter().all(|v| v.to_f32().is_finite()),
                                    "Runner FAIL nonfinite norm/hidden"
                                );
                                bits.extend(values.into_iter().flat_map(|v| v.to_bits().to_le_bytes()));
                            }
                            norms.push(bits);
                            Ok(())
                        })?
                    };
                    ensure!(
                        e.committed_prefix_ids == ids,
                        "Runner FAIL committed dense prefix differs"
                    );
                    let before = signature(&e, mtp)?;
                    let mut continuation = Vec::new();
                    for index in 0..4 {
                        e.set_token(ids[index * 263 % ids.len()])?;
                        continuation.push(e.decode(false, None)?);
                        continuation.extend(e.decode_logits()?.into_iter().map(f32::to_bits));
                    }
                    let observed = (token, norms, before, continuation, signature(&e, false)?);
                    if let Some(expected) = reference.take() {
                        let equal = expected == observed;
                        defects += usize::from(!equal);
                        println!(
                            "{} dense_layer_prefill length={length} chunk={} mtp={mtp} bitwise_equal={equal} continuation=4",
                            if equal { "PASS" } else { "DEFECT" },
                            e.chunk
                        );
                    } else {
                        reference = Some(observed);
                    }
                }
                pool = e
                    .qwen
                    .as_mut()
                    .context("Qwen runtime missing")?
                    .prompt
                    .take();
            }
        }
        println!(
            "{} dense_layer_prefill_gate cases=6 tolerance_bits=0 defects={defects}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "DEFECT dense layer prefill cases={defects}");
        Ok(())
    }

    pub(super) fn output() -> Result<PathBuf> {
        let p = PathBuf::from(
            std::env::var_os("FF_Q38_MTP_OUTPUT").context("FF_Q38_MTP_OUTPUT missing")?,
        );
        std::fs::create_dir_all(&p)?;
        Ok(p)
    }
    pub(super) fn save(e: &mut Engine) -> Result<()> {
        let mut c = e.mtp.take().context("MTP missing")?;
        let r = Bound {
            engine: e,
            chain: &mut c,
        }
        .save(5);
        e.mtp = Some(c);
        r
    }
    pub(super) fn restore(e: &mut Engine) -> Result<()> {
        let mut c = e.mtp.take().context("MTP missing")?;
        let r = Bound {
            engine: e,
            chain: &mut c,
        }
        .restore(false);
        e.mtp = Some(c);
        r
    }
    pub(super) fn ordinary(e: &mut Engine, count: usize) -> Result<Vec<u32>> {
        let mut c = e.mtp.take().context("MTP missing")?;
        let result = (|| -> Result<Vec<u32>> {
            let mut out = Vec::new();
            let mut b = Bound {
                engine: e,
                chain: &mut c,
            };
            for _ in 0..count {
                let token = b.pending()?;
                // Keep the old head-producing canonical row as the state oracle.
                let hidden = b.chain.last_target.clone();
                b.row(
                    token,
                    matrix(&hidden, &b.engine.device.stream, 0, 1, HIDDEN)?,
                    OutputMode::Predict,
                )?;
                out.push(b.target_decode(token)?);
            }
            b.chain.committed_position = b.engine.position;
            Ok(out)
        })();
        e.mtp = Some(c);
        result
    }
    pub(super) fn state_bits(e: &Engine) -> Result<Vec<u8>> {
        state_bits_with_visibility(e, false)
    }
    pub(super) fn committed_bits(e: &Engine) -> Result<Vec<u8>> {
        state_bits_with_visibility(e, true)
    }
    pub(super) fn committed_difference(
        e: &Engine,
        expected: &[u8],
        actual: &[u8],
    ) -> Result<serde_json::Value> {
        if expected.len() != actual.len() {
            return Ok(
                serde_json::json!([{"field":"packed_extent","expected":expected.len(),"actual":actual.len()}]),
            );
        }
        let c = e.mtp.as_ref().context("MTP missing")?;
        let mut fields = Vec::new();
        for (name, store, prefix) in [
            ("target", &e.state, e.position),
            (
                "mtp",
                e.mtp_state.as_ref().context("MTP state missing")?,
                c.cursor,
            ),
        ] {
            for (layer, g) in store.gdn.iter().enumerate() {
                fields.push((format!("{name}.gdn{layer}.s"), g.s.len() * 4));
                fields.push((format!("{name}.gdn{layer}.history"), g.history.len() * 4));
            }
            for layer in 0..store.kv.len() {
                for part in ["k", "v"] {
                    for head in 0..4 {
                        fields.push((
                            format!("{name}.kv{layer}.{part}.head{head}"),
                            prefix * 256 * 2,
                        ));
                    }
                }
            }
            fields.push((
                format!("{name}.sequence"),
                store.sequence.as_ref().context("metadata missing")?.len() * 4,
            ));
        }
        fields.extend([
            ("target.position".into(), e.work.position.len() * 4),
            ("target.length".into(), e.work.length.len() * 4),
            ("target.token".into(), e.work.token.len() * 4),
        ]);
        if e.position > 0 {
            fields.push(("target.logits".into(), e.work.logits.len() * 2));
        }
        fields.extend([
            ("mtp.pending".into(), 4),
            ("mtp.last_target".into(), c.last_target.len() * 2),
            ("mtp.recurrent".into(), c.recurrent.len() * 2),
            ("target.host_position".into(), size_of::<usize>()),
            ("mtp.cursor".into(), size_of::<usize>()),
            ("mtp.committed_position".into(), size_of::<usize>()),
            ("target.epoch".into(), 8),
            ("mtp.epoch".into(), 8),
        ]);
        let used = fields.iter().map(|(_, bytes)| bytes).sum::<usize>();
        ensure!(
            used <= expected.len(),
            "committed state partition exceeds receipt"
        );
        fields.push(("target.qwen_positions".into(), expected.len() - used));
        let mut offset = 0;
        let mut differences = Vec::new();
        for (field, bytes) in fields {
            let a = &expected[offset..offset + bytes];
            let b = &actual[offset..offset + bytes];
            if let Some(index) = a.iter().zip(b).position(|(a, b)| a != b) {
                let end = (index + 8).min(bytes);
                differences.push(serde_json::json!({"field":field,"byte":index,"expected":a[index..end],"actual":b[index..end]}));
            }
            offset += bytes;
        }
        Ok(serde_json::json!(differences))
    }
    fn state_bits_with_visibility(e: &Engine, visible_only: bool) -> Result<Vec<u8>> {
        let s = &e.device.stream;
        let mut bits = Vec::new();
        for (index, store) in [&e.state, e.mtp_state.as_ref().context("MTP state missing")?]
            .into_iter()
            .enumerate()
        {
            for g in &store.gdn {
                for buf in [&g.s, &g.history] {
                    for v in s.clone_dtoh(buf)? {
                        bits.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
            }
            for kv in &store.kv {
                for buf in [&kv.k, &kv.v] {
                    if visible_only {
                        let prefix = if index == 0 {
                            e.position
                        } else {
                            e.mtp.as_ref().context("MTP missing")?.cursor
                        };
                        for head in 0..4 {
                            for v in s.clone_dtoh(&buf.slice(
                                head * e.capacity * 256..(head * e.capacity + prefix) * 256,
                            ))? {
                                bits.extend_from_slice(&v.to_bits().to_le_bytes());
                            }
                        }
                        continue;
                    }
                    for v in s.clone_dtoh(buf)? {
                        bits.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
            }
            for v in s.clone_dtoh(store.sequence.as_ref().context("metadata missing")?)? {
                bits.extend_from_slice(&v.to_le_bytes());
            }
        }
        for buf in [&e.work.position, &e.work.length] {
            for v in s.clone_dtoh(buf)? {
                bits.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in s.clone_dtoh(&e.work.token)? {
            bits.extend_from_slice(&v.to_le_bytes());
        }
        // Before the first prefill, logits are scratch, including capture warmup.
        if e.position > 0 {
            for v in s.clone_dtoh(&e.work.logits)? {
                bits.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        let c = e.mtp.as_ref().context("MTP missing")?;
        bits.extend_from_slice(&c.pending.to_le_bytes());
        for buf in [&c.last_target, &c.recurrent] {
            for v in s.clone_dtoh(buf)? {
                bits.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for n in [e.position, c.cursor, c.committed_position] {
            bits.extend_from_slice(&n.to_le_bytes());
        }
        bits.extend_from_slice(&e.mtp_epoch.to_le_bytes());
        bits.extend_from_slice(&c.epoch.to_le_bytes());
        for v in e
            .qwen
            .as_ref()
            .context("Qwen missing")?
            .position_values(s)?
        {
            bits.extend_from_slice(&v.to_le_bytes());
        }
        Ok(bits)
    }

    fn load_ready_chain() -> Result<Engine> {
        let mut e = load(true)?;
        configure(
            &mut e,
            options(
                3,
                MtpHiddenTap::PreFinalNorm,
                MtpRecurrentTap::PostMtpNorm,
                true,
            ),
        )?;
        e.capture()?;
        e.closure.ready(std::time::Instant::now())?;
        Ok(e)
    }

    pub(crate) fn q38_mtp_chain_gpu() -> Result<()> {
        let mut e = load_ready_chain()?;
        let out = output()?;
        for k in 1..=3 {
            for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
                for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                    for case in ["math1", "code1", "gen1", "gen2", "math2"] {
                        configure(&mut e, options(k, h2, h5, true))?;
                        let ids = prompt(case)?;
                        let first = e.prefill(&ids, None)?;
                        // The ordinary target consumes the same pending-anchor convention.
                        let mut plain = vec![first];
                        for _ in 1..64 {
                            if e.model_spec
                                .eos
                                .contains(plain.last().context("no target output")?)
                            {
                                break;
                            }
                            let t = e.decode(true, None)?;
                            plain.push(t);
                            if e.model_spec.eos.contains(&t) {
                                break;
                            }
                        }
                        // The four-row transaction journal cannot rewind 64 tokens; re-prefill.
                        ensure!(
                            e.mtp_prefill(&ids)? == first,
                            "MTP bootstrap changed target prefill"
                        );
                        let mut got = vec![first];
                        let mut rounds = Vec::new();
                        while got.len() < plain.len()
                            && !e.model_spec.eos.contains(got.last().context("no output")?)
                        {
                            let r = e.mtp_round(plain.len() - got.len(), None)?;
                            ensure!(!r.tokens.is_empty(), "MTP stopped before target budget");
                            rounds.push(serde_json::json!({"matched":r.matched,"accepted":r.accepted,"consumed":r.consumed,"proposals":r.proposals,"tokens":r.tokens}));
                            got.extend_from_slice(&r.tokens);
                        }
                        ensure!(
                            got == plain,
                            "M2a token divergence {case} K{k} {h2:?}/{h5:?}"
                        );
                        std::fs::write(
                            out.join(format!("{case}-K{k}-{h2:?}-{h5:?}.json")),
                            serde_json::to_vec_pretty(
                                &serde_json::json!({"case":case,"depth":k,"h2":h2,"h5":h5,"tokens":got,"plain_tokens":plain,"rounds":rounds,"target_token_gate":"BIT_EQUAL","recursive_numeric_gate":"requires corrected 6d chain fixture; old 0de341ed not qualified"}),
                            )?,
                        )?;
                        eprintln!("M2A_CHAIN {case} K{k} {h2:?}/{h5:?} target_token_bits PASS");
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn q38_mtp_transaction_gpu() -> Result<()> {
        let mut e = load_ready_chain()?;
        let ids = prompt("math1")?;
        for graph in [false, true] {
            for k in 1..=3 {
                for h2 in [MtpHiddenTap::PreFinalNorm, MtpHiddenTap::PostFinalNorm] {
                    for h5 in [MtpRecurrentTap::PreMtpNorm, MtpRecurrentTap::PostMtpNorm] {
                        configure(&mut e, options(k, h2, h5, graph))?;
                        e.mtp_prefill(&ids)?;
                        save(&mut e)?;
                        let predictions = ordinary(&mut e, k + 1)?;
                        restore(&mut e)?;
                        ensure!(
                            !predictions[..k]
                                .iter()
                                .any(|t| e.model_spec.eos.contains(t)),
                            "transaction fixture hit EOS; use an earlier prefix"
                        );
                        for j in 0..=k {
                            let expected = ordinary(&mut e, j + 1)?;
                            let bits = state_bits(&e)?;
                            restore(&mut e)?;
                            let mut proposals = predictions[..k].to_vec();
                            if j < k {
                                proposals[j] = (proposals[j] + 1) % VOCAB as u32;
                            }
                            e.mtp.as_mut().context("MTP missing")?.force_proposals =
                                Some(proposals);
                            let head_runs =
                                e.mtp.as_ref().context("MTP missing")?.runtime.head_runs;
                            let round = e.mtp_round(64, None)?;
                            ensure!(
                                e.mtp.as_ref().context("MTP missing")?.runtime.head_runs
                                    - head_runs
                                    == k,
                                "M2a canonical repair evaluated an unused MTP head"
                            );
                            ensure!(
                                round.accepted == j && round.tokens == expected,
                                "wrong forced acceptance count/output"
                            );
                            ensure!(
                                state_bits(&e)? == bits,
                                "M2a committed S/history/KV/meta/hidden mismatch at K{k}/j{j}/{h2:?}/{h5:?}/graph{graph}"
                            );
                            restore(&mut e)?;
                            e.mtp.as_mut().context("MTP missing")?.force_proposals = None;
                        }
                    }
                }
            }
        }
        // A snapshot near the last consumable slot must clip its copy window.
        configure(
            &mut e,
            options(
                3,
                MtpHiddenTap::PreFinalNorm,
                MtpRecurrentTap::PostMtpNorm,
                true,
            ),
        )?;
        let mut tail = ids;
        tail.resize(e.capacity - 2, 1);
        e.mtp_prefill(&tail)?;
        save(&mut e)?;
        let expected = ordinary(&mut e, 1)?;
        let bits = state_bits(&e)?;
        restore(&mut e)?;
        let r = e.mtp_round(64, None)?;
        ensure!(
            r.tokens == expected && state_bits(&e)? == bits,
            "last consumable slot mismatch"
        );
        ensure!(
            e.mtp_round(64, None)?.tokens.is_empty(),
            "MTP ran beyond legal next-row metadata"
        );
        // A different prefill with the same length must not revive an old MTP cache.
        let ids = prompt("math1")?;
        e.mtp_prefill(&ids)?;
        e.reset()?;
        e.prefill(&ids, None)?;
        let before = state_bits(&e)?;
        ensure!(
            e.mtp_round(1, None).is_err(),
            "MTP accepted a stale state epoch"
        );
        ensure!(
            state_bits(&e)? == before,
            "stale-epoch rejection mutated state"
        );
        Ok(())
    }

    pub(crate) fn q38_mtp_failure_and_capture_gpu() -> Result<()> {
        let mut e = load(true)?;
        e.mtp_enable(options(
            3,
            MtpHiddenTap::PreFinalNorm,
            MtpRecurrentTap::PostMtpNorm,
            true,
        ))?;
        let before = state_bits(&e)?;
        for occurrence in [1, 2] {
            e.capture_fail_after_layer = Some((0, occurrence));
            ensure!(e.capture().is_err(), "capture fault ignored");
            ensure!(
                e.poisoned && e.graph.is_none() && state_bits(&e)? == before,
                "capture failed to restore state or published graph"
            );
            e.capture_fail_after_layer = None;
            e.poisoned = false;
        }
        use crate::backend::cuda::engine::CaptureFault;
        for fault in [
            CaptureFault::Begin,
            CaptureFault::End,
            CaptureFault::Instantiate,
        ] {
            e.capture_fail_stage = Some(fault);
            ensure!(
                e.capture().is_err(),
                "capture stage fault ignored: {fault:?}"
            );
            ensure!(
                e.poisoned && e.graph.is_none() && state_bits(&e)? == before,
                "capture stage rollback/publication failure: {fault:?}"
            );
            e.capture_fail_stage = None;
            e.poisoned = false;
        }
        e.capture()?;
        ensure!(
            state_bits(&e)? == before,
            "capture changed target/MTP state"
        );
        e.closure.ready(std::time::Instant::now())?;
        e.mtp_prefill(&prompt("math1")?)?;
        save(&mut e)?;
        let predictions = ordinary(&mut e, 4)?;
        restore(&mut e)?;
        e.mtp.as_mut().context("MTP missing")?.force_proposals = Some(predictions[..3].to_vec());
        let mut points = vec![Fault::AfterRestore, Fault::BeforePublish];
        points.extend((0..3).map(Fault::AfterDraft));
        points.extend((0..4).map(Fault::AfterVerify));
        points.extend((0..4).map(Fault::AfterCanonical));
        points.extend((0..4).map(Fault::AfterReplay));
        let before = state_bits(&e)?;
        for fault in points {
            ensure!(
                e.mtp_round(64, Some(fault)).is_err(),
                "MTP fault ignored: {fault:?}"
            );
            ensure!(
                !e.poisoned && state_bits(&e)? == before,
                "rollback state mismatch: {fault:?}"
            );
        }
        let expected = ordinary(&mut e, 4)?;
        let bits = state_bits(&e)?;
        restore(&mut e)?;
        let round = e.mtp_round(64, None)?;
        ensure!(
            round.tokens == expected && state_bits(&e)? == bits,
            "retry after rollback differs from ordinary decode"
        );
        Ok(())
    }
}

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
