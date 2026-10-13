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
    saved: Option<Saved>,
    last_target: Vec<bf16>,
    recurrent: Vec<bf16>,
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

#[derive(Clone)]
struct Saved {
    position: usize,
    work_position: Vec<i32>,
    work_length: Vec<i32>,
    logits: Vec<bf16>,
    last_target: Vec<bf16>,
    recurrent: Vec<bf16>,
    cursor: usize,
}

struct Bound<'a> {
    engine: &'a mut Engine,
    chain: &'a mut Chain,
}

impl Bound<'_> {
    fn row(&mut self, input: u32, hidden: &[bf16], mode: OutputMode) -> Result<Option<u32>> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let state = e.mtp_state.as_mut().context("MTP state missing")?;
        let dump = c.runtime.run_decode(
            Execution {
                device: &e.device,
                ops: &e.ops,
                blas: &e.blas,
                catalog: &c.catalog,
                embedding: &e.weights.embed,
                head: &e.weights.head,
            },
            state,
            RunInput {
                prefix: c.cursor,
                ids: &[input],
                hidden,
                #[cfg(test)]
                control: Control::None,
                detailed: false,
            },
            mode,
        )?;
        let output = match c.options.recurrent_tap {
            MtpRecurrentTap::PreMtpNorm => &c.runtime.work.x,
            MtpRecurrentTap::PostMtpNorm => &c.runtime.work.n,
        };
        c.recurrent = e.device.stream.clone_dtoh(&output.slice(..HIDDEN))?;
        ensure!(
            c.recurrent.iter().all(|h| h.is_finite()),
            "nonfinite MTP recurrent hidden"
        );
        c.cursor += 1;
        state.set_sequence(&e.device.stream, c.cursor as u32, 1, 1)?;
        if mode == OutputMode::StateOnly {
            return Ok(None);
        }
        if let Some(token) = dump.prediction {
            return Ok(Some(token));
        }
        let (_, shape, bytes) = dump.tensors.get("logits").context("MTP head missing")?;
        ensure!(shape == &[1, VOCAB], "MTP head extent mismatch");
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, bits) in bytes.as_chunks::<2>().0.iter().enumerate() {
            let value = bf16::from_bits(u16::from_le_bytes(*bits)).to_f32();
            if value > best.1 {
                best = (i as u32, value);
            }
        }
        Ok(Some(best.0))
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
        self.engine.token()
    }
    fn save(&mut self, rows: usize) -> Result<()> {
        let e = &self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        ensure!(
            rows <= 4 && e.position > 0 && c.cursor == e.position - 1,
            "MTP checkpoint cursor mismatch"
        );
        c.saved = None;
        c.target_checkpoint
            .save(&e.state, &e.work.token, e.position, s)?;
        c.mtp_checkpoint.save(
            e.mtp_state.as_ref().context("MTP state missing")?,
            &c.runtime.work.token,
            c.cursor,
            s,
        )?;
        c.saved = Some(Saved {
            position: e.position,
            work_position: s.clone_dtoh(&e.work.position)?,
            work_length: s.clone_dtoh(&e.work.length)?,
            logits: s.clone_dtoh(&e.work.logits)?,
            last_target: c.last_target.clone(),
            recurrent: c.recurrent.clone(),
            cursor: c.cursor,
        });
        Ok(())
    }
    fn restore(&mut self) -> Result<()> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let s = &e.device.stream;
        let saved = c.saved.as_ref().context("MTP checkpoint not valid")?;
        c.target_checkpoint
            .restore(&mut e.state, &mut e.work.token, s)?;
        c.mtp_checkpoint.restore(
            e.mtp_state.as_mut().context("MTP state missing")?,
            &mut c.runtime.work.token,
            s,
        )?;
        s.memcpy_htod(&saved.work_position, &mut e.work.position)?;
        s.memcpy_htod(&saved.work_length, &mut e.work.length)?;
        s.memcpy_htod(&saved.logits, &mut e.work.logits)?;
        s.synchronize()?;
        e.position = saved.position;
        e.logits_cache.mark_valid();
        e.poisoned = false;
        c.last_target = saved.last_target.clone();
        c.recurrent = saved.recurrent.clone();
        c.cursor = saved.cursor;
        c.committed_position = saved.position;
        if let Some(batch) = c.batch.as_mut() {
            batch.target_restores += 1;
        }
        Ok(())
    }
    fn poison(&mut self) {
        self.engine.poisoned = true;
    }
    fn draft(&mut self, input: u32, depth: usize) -> Result<u32> {
        let hidden = if depth == 0 {
            self.chain.last_target.clone()
        } else {
            self.chain.recurrent.clone()
        };
        let token = self
            .row(input, &hidden, OutputMode::Predict)?
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
        self.chain.last_target = self
            .engine
            .device
            .stream
            .clone_dtoh(&hidden.slice(..HIDDEN))?;
        Ok(token)
    }
    fn canonical_row(&mut self, input: u32) -> Result<()> {
        ensure!(
            self.chain.cursor + 1 == self.engine.position,
            "MTP canonical position mismatch"
        );
        let hidden = self.chain.last_target.clone();
        self.row(input, &hidden, OutputMode::StateOnly)?;
        Ok(())
    }
}

impl Engine {
    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn mtp_enable(
        &mut self,
        options: MtpOptions,
    ) -> Result<()> {
        self.mtp_enable_bound(options, &(1..=self.chunk).collect::<Vec<_>>())
    }
    pub(in crate::backend::cuda::engine) fn mtp_enable_resident(
        &mut self,
        options: MtpOptions,
    ) -> Result<()> {
        self.mtp_enable_bound(options, &(1..=self.chunk).collect::<Vec<_>>())
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
            chain.saved = None;
            chain.last_target.fill(bf16::ZERO);
            chain.recurrent.fill(bf16::ZERO);
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
    pub(in crate::backend::cuda::engine) fn mtp_enable_for_prompt(
        &mut self,
        options: MtpOptions,
        prompt: usize,
    ) -> Result<()> {
        self.mtp_enable_bound(options, &crate::mtp::bootstrap_rows(prompt, self.chunk)?)
    }
    fn mtp_enable_bound(&mut self, options: MtpOptions, rows: &[usize]) -> Result<()> {
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
        for &rows in rows {
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
            saved: None,
            last_target: vec![bf16::ZERO; HIDDEN],
            recurrent: vec![bf16::ZERO; HIDDEN],
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
            chain.saved = None;
            #[cfg(test)]
            {
                chain.force_proposals = None;
            }
            chain.last_target.fill(bf16::ZERO);
            chain.recurrent.fill(bf16::ZERO);
            chain.bootstrap.last_rows = 0;
            if let Some(batch) = chain.batch.as_mut() {
                batch.invalidate_frame();
            }
            let token = self.prefill_chunks(ids, None, ids.len(), |engine, chunk| {
                chain.bootstrap_chunk(engine, chunk)
            })?;
            chain.last_target = self.device.stream.clone_dtoh(&chain.bootstrap.carry)?;
            if chain.bootstrap.last_rows > 0 {
                let row = chain.bootstrap.last_rows - 1;
                let hidden = match chain.options.recurrent_tap {
                    MtpRecurrentTap::PreMtpNorm => &chain.runtime.work.x,
                    MtpRecurrentTap::PostMtpNorm => &chain.runtime.work.n,
                };
                chain.recurrent = self
                    .device
                    .stream
                    .clone_dtoh(&hidden.slice(row * HIDDEN..(row + 1) * HIDDEN))?;
            }
            ensure!(
                chain
                    .last_target
                    .iter()
                    .chain(&chain.recurrent)
                    .all(|x| x.is_finite()),
                "nonfinite MTP bootstrap boundary"
            );
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

#[path = "engine_mtp_batched.rs"]
mod batched;
