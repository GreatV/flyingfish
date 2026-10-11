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
    fn row(&mut self, input: u32, hidden: &[bf16]) -> Result<u32> {
        let e = &mut self.engine;
        let c = &mut self.chain;
        let state = e.mtp_state.as_mut().context("MTP state missing")?;
        let dump = c.runtime.run(
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
        )?;
        let output = match c.options.recurrent_tap {
            MtpRecurrentTap::PreMtpNorm => &c.runtime.work.x,
            MtpRecurrentTap::PostMtpNorm => &c.runtime.work.n,
        };
        c.recurrent = e.device.stream.clone_dtoh(&output.slice(..HIDDEN))?;
        c.cursor += 1;
        state.set_sequence(&e.device.stream, c.cursor as u32, 1, 1)?;
        let (_, shape, bytes) = dump.tensors.get("logits").context("MTP head missing")?;
        ensure!(shape == &[1, VOCAB], "MTP head extent mismatch");
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, bits) in bytes.as_chunks::<2>().0.iter().enumerate() {
            let value = bf16::from_bits(u16::from_le_bytes(*bits)).to_f32();
            if value > best.1 {
                best = (i as u32, value);
            }
        }
        Ok(best.0)
    }

    fn tap_prefill(&mut self, pre: &[bf16]) -> Result<Vec<bf16>> {
        if self.chain.options.hidden_tap == MtpHiddenTap::PreFinalNorm {
            return Ok(pre.to_vec());
        }
        let e = &self.engine;
        let r = &mut self.chain.runtime;
        let s = &e.device.stream;
        s.memcpy_htod(pre, &mut r.hidden)?;
        r.common.norm.rms_with(
            s,
            RowDomain {
                meta: e
                    .mtp_state
                    .as_ref()
                    .context("MTP state missing")?
                    .seq_ref(s)?,
                row_base: 0,
                row_capacity: 1,
            },
            heads(&r.hidden, s, 0, 1, 1, HIDDEN as u32)?,
            vector(&e.weights.norm, s)?,
            &mut TensorMut(heads(&r.embedding, s, 0, 1, 1, HIDDEN as u32)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: 1e-6,
            },
        )?;
        Ok(s.clone_dtoh(&r.embedding)?)
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
        let token = self.row(input, &hidden)?;
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
        self.row(input, &hidden)?;
        Ok(())
    }
}

impl Engine {
    pub(in crate::backend::cuda::engine) fn mtp_enable(
        &mut self,
        options: MtpOptions,
    ) -> Result<()> {
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
            max_rows: 1,
            capacity: self.capacity,
            phase: ExecutionPhase::Decode,
        };
        let memory = plan.memory(&self.model_spec)?;
        let journal = 156_893_184usize + 4 * 65_536 + 36 + 4 * 4096 + 36;
        let needed = memory.total - crate::mtp::PROBE_WEIGHT_BYTES - memory.blas
            + crate::mtp::WEIGHT_BYTES
            + journal;
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
        let mut spec = self.model_spec.clone();
        spec.layers = vec![LayerPlan {
            mixer: Mixer::FullAttention,
            state_index: 0,
        }];
        let state = StateStore::new(&spec, self.capacity, &self.device.stream)?;
        let runtime = MtpRuntime::new(
            &self.device,
            &self.model_spec,
            &catalog,
            &state,
            &self.blas,
            plan,
        )?;
        ensure!(
            runtime.allocation_bytes() + state.bytes()
                == memory.total - memory.weights - memory.blas,
            "MTP attachment allocation/budget mismatch"
        );
        let target_checkpoint =
            StateCheckpoint::new(&self.state, &self.model_spec, 4, &self.device.stream)?;
        let mtp_checkpoint =
            StateCheckpoint::new(&state, &self.model_spec, 4, &self.device.stream)?;
        ensure!(
            target_checkpoint.bytes() + mtp_checkpoint.bytes() == journal,
            "MTP journal budget mismatch"
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
        });
        eprintln!(
            "MTP_M2A_MEMORY additional_device_bytes={needed} checkpoint_bytes={journal} shared_weights_duplicated=0"
        );
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn mtp_prefill(&mut self, ids: &[u32]) -> Result<u32> {
        ensure!(
            !ids.is_empty() && ids.len() < self.capacity - 1,
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
            // The M2a baseline observes the existing prefill without changing its
            // chunk partition or target math. Only the final layer is consumed.
            let mut trace = crate::trace::Trace::default();
            let token = self.prefill(ids, Some(&mut trace))?;
            let chunk = self.chunk;
            let mut bound = Bound {
                engine: self,
                chain: &mut chain,
            };
            let mut seen = 0;
            for (part, tokens) in ids.chunks(chunk).enumerate() {
                let pre = trace.values(&format!("prefill.{part}.layer.63"))?;
                ensure!(
                    pre.len() == tokens.len() * HIDDEN,
                    "MTP prefill hidden extent mismatch"
                );
                for (row, &input) in tokens.iter().enumerate() {
                    if seen > 0 {
                        let hidden = bound.chain.last_target.clone();
                        bound.row(input, &hidden)?;
                    }
                    let hidden: Vec<_> = pre[row * HIDDEN..(row + 1) * HIDDEN]
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect();
                    bound.chain.last_target = bound.tap_prefill(&hidden)?;
                    seen += 1;
                }
            }
            ensure!(
                bound.chain.cursor + 1 == bound.engine.position,
                "MTP bootstrap prefix mismatch"
            );
            bound.chain.committed_position = bound.engine.position;
            bound.chain.epoch = bound.engine.mtp_epoch;
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
