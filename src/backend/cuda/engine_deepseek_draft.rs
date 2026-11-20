use super::*;
use crate::backend::cuda::{
    draft::Compute, latent_window_attention::BlockWindow, linear::Bf16F32Plan, weights::Loader,
};
use crate::backend::gpu::CudaFunction;
use crate::dspark::DraftConfig;
use crate::model::{HcPreSource, HcSublayer};

#[cfg(feature = "cuda")]
pub(crate) struct TargetState {
    pub(in crate::backend::cuda::engine) options: crate::mtp::MtpOptions,
    origin: CudaSlice<i32>,
    length: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    saved_logits: CudaSlice<f32>,
    hc: crate::backend::cuda::hc::HcCheckpoint,
    history: crate::engram::HistoryCheckpoint,
    position: Option<usize>,
    pending: Option<u32>,
    rows: usize,
    kept: Option<usize>,
    prefix_len: usize,
    ids_len: usize,
    input_kind: Option<PrefixLastInputKind>,
}

#[cfg(feature = "cuda")]
impl TargetState {
    fn new(engine: &Engine) -> Result<Self> {
        let s = &engine.device.stream;
        let rows = 4;
        let state = Self {
            options: crate::mtp::MtpOptions {
                depth: 3,
                hidden_tap: crate::mtp::MtpHiddenTap::PreFinalNorm,
                recurrent_tap: crate::mtp::MtpRecurrentTap::PreMtpNorm,
                graph: engine.spec_graph,
            },
            origin: s.alloc_zeros(1)?,
            length: s.alloc_zeros(1)?,
            logits: s.alloc_zeros(rows * engine.config.vocab_size)?,
            saved_logits: s.alloc_zeros(engine.config.vocab_size)?,
            hc: engine
                .hc
                .as_ref()
                .context("staged target mHC missing")?
                .checkpoint(rows, s)?,
            history: crate::engram::HistoryCheckpoint::new(rows)?,
            position: None,
            pending: None,
            rows: 0,
            kept: None,
            prefix_len: 0,
            ids_len: 0,
            input_kind: None,
        };
        eprintln!(
            "STAGED_TARGET_STATE device_bytes={} cpu_history_entries=4",
            (state.origin.len()
                + state.length.len()
                + state.logits.len()
                + state.saved_logits.len())
                * 4
                + state.hc.bytes()
        );
        Ok(state)
    }
}

#[cfg(feature = "cuda")]
impl Engine {
    pub(in crate::backend::cuda::engine) fn staged_prefill(&mut self, ids: &[u32]) -> Result<u32> {
        self.reset()?;
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        spec.draft.reset_setup()?;
        let state = spec
            .staged
            .as_mut()
            .context("staged target state missing")?;
        state.position = None;
        state.rows = 0;
        state.kept = None;
        let first = self.prefill(ids, None)?;
        self.state.set_sequence(
            &self.device.stream,
            self.position as u32,
            1,
            self.workspace_rows as u32,
        )?;
        Ok(first)
    }

    pub(in crate::backend::cuda::engine) fn staged_round(
        &mut self,
        remaining: usize,
        #[cfg(test)] fault: Option<crate::mtp::batched::Fault>,
    ) -> Result<crate::mtp::batched::BatchRound> {
        self.closure.assert_ready()?;
        ensure!(!self.poisoned, "staged round requires healthy state");
        let options = self
            .spec
            .as_ref()
            .and_then(|s| s.staged.as_ref())
            .context("staged target state missing")?
            .options;
        let eos = self.model_spec.eos.clone();
        crate::mtp::batched::run(
            self,
            options,
            remaining,
            &eos,
            #[cfg(test)]
            fault,
        )
    }

    fn staged_verify(&mut self, inputs: &[u32], graph: bool) -> Result<Vec<u32>> {
        ensure!(
            (1..=4).contains(&inputs.len())
                && self
                    .position
                    .checked_add(inputs.len())
                    .is_some_and(|end| end < self.capacity),
            "staged verify capacity"
        );
        let rows = inputs.len();
        let start = self.position;
        self.spec
            .as_mut()
            .context("staged attachment missing")?
            .active_rows = rows;
        self.prepare_deepseek_engram(inputs, None)?;
        let s = self.device.stream.clone();
        s.memcpy_htod(inputs, &mut self.work.ids.slice_mut(..rows))?;
        s.memcpy_htod(&inputs[..1], &mut self.work.token)?;
        s.memcpy_htod(&[i32::try_from(start)?], &mut self.work.position)?;
        s.memcpy_htod(&[i32::try_from(start + rows)?], &mut self.work.length)?;
        if graph {
            self.run_deepseek_graph_rows(rows)?;
        } else {
            self.run_step(
                crate::backend::Step {
                    rows,
                    input: crate::backend::Input::Ids,
                    mask: crate::backend::Mask::Causal,
                    head: Head::All,
                },
                None,
            )?;
        }
        let spec = self.spec.as_ref().context("staged attachment missing")?;
        let predictions = s.clone_dtoh(&spec.predictions.slice(..rows))?;
        self.position = start + rows;
        self.pending_token = predictions.last().copied();
        self.logits_cache.mark_valid();
        Ok(predictions)
    }

    fn staged_select(&mut self, rows: usize) -> Result<()> {
        let s = &self.device.stream;
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        let target = spec
            .staged
            .as_mut()
            .context("staged target state missing")?;
        let origin = target.position.context("staged checkpoint is invalid")?;
        ensure!(
            rows > 0 && rows <= target.rows && self.position == origin + target.rows,
            "stale staged verify prefix"
        );
        spec.checkpoint
            .as_ref()
            .context("staged target checkpoint missing")?
            .select_prefix(&mut self.state, &[], rows, s)?;
        self.hc
            .as_mut()
            .context("staged target mHC missing")?
            .select_checkpoint(&target.hc, rows, s)?;
        let engram = self
            .deepseek_engram
            .as_mut()
            .context("staged target Engram missing")?;
        engram.service.select_history(&mut target.history, rows)?;
        engram.prepared = None;
        let v = self.config.vocab_size;
        s.memcpy_dtod(
            &target.logits.slice((rows - 1) * v..rows * v),
            self.logits_f32
                .as_mut()
                .context("staged target FP32 logits missing")?,
        )?;
        s.memcpy_dtod(
            &spec.predictions.slice(rows - 1..rows),
            &mut self.work.token,
        )?;
        self.position = origin + rows;
        self.pending_token = Some(s.clone_dtoh(&self.work.token)?[0]);
        s.memcpy_htod(&[i32::try_from(self.position)?], &mut self.work.position)?;
        s.memcpy_htod(&[i32::try_from(self.position + 1)?], &mut self.work.length)?;
        self.state
            .set_sequence(s, self.position as u32, 1, self.workspace_rows as u32)?;
        target.kept = Some(rows);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl crate::mtp::transaction::Backend for Engine {
    fn position(&self) -> usize {
        self.position
    }
    fn capacity(&self) -> usize {
        self.capacity - 1
    }
    fn pending(&self) -> Result<u32> {
        self.pending_token.context("staged pending token missing")
    }
    fn poison(&mut self) {
        self.poisoned = true;
    }

    fn save(&mut self, rows: usize) -> Result<()> {
        let s = &self.device.stream;
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        let target = spec
            .staged
            .as_mut()
            .context("staged target state missing")?;
        ensure!(
            (1..=4).contains(&rows) && self.position + rows < self.capacity,
            "staged checkpoint capacity"
        );
        target.position = None;
        spec.checkpoint
            .as_mut()
            .context("staged target checkpoint missing")?
            .save_rows(&self.state, &self.work.token, self.position, rows, s, None)?;
        self.hc
            .as_ref()
            .context("staged target mHC missing")?
            .save_checkpoint(&mut target.hc, s)?;
        self.deepseek_engram
            .as_ref()
            .context("staged target Engram missing")?
            .service
            .save_history(&mut target.history, self.position, rows)?;
        spec.draft.checkpoint_kv(self.position)?;
        s.memcpy_dtod(&self.work.position, &mut target.origin)?;
        s.memcpy_dtod(&self.work.length, &mut target.length)?;
        s.memcpy_dtod(
            self.logits_f32
                .as_ref()
                .context("staged target FP32 logits missing")?,
            &mut target.saved_logits,
        )?;
        target.pending = self.pending_token;
        target.prefix_len = self.committed_prefix_ids.len();
        target.ids_len = self.resident_ids_len;
        target.input_kind = self.last_input_kind;
        target.rows = rows;
        target.kept = None;
        target.position = Some(self.position);
        Ok(())
    }

    fn restore(&mut self, _replay: bool) -> Result<()> {
        let s = &self.device.stream;
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        let target = spec
            .staged
            .as_mut()
            .context("staged target state missing")?;
        let origin = target.position.context("staged checkpoint is invalid")?;
        spec.checkpoint
            .as_ref()
            .context("staged target checkpoint missing")?
            .restore(&mut self.state, &mut self.work.token, s)?;
        self.hc
            .as_mut()
            .context("staged target mHC missing")?
            .select_checkpoint(&target.hc, 0, s)?;
        let engram = self
            .deepseek_engram
            .as_mut()
            .context("staged target Engram missing")?;
        engram.service.select_history(&mut target.history, 0)?;
        engram.prepared = None;
        spec.draft.restore_kv(origin, 0)?;
        s.memcpy_dtod(&target.origin, &mut self.work.position)?;
        s.memcpy_dtod(&target.length, &mut self.work.length)?;
        s.memcpy_dtod(
            &target.saved_logits,
            self.logits_f32
                .as_mut()
                .context("staged target FP32 logits missing")?,
        )?;
        self.position = origin;
        self.pending_token = target.pending;
        self.committed_prefix_ids.truncate(target.prefix_len);
        self.resident_ids_len = target.ids_len;
        self.last_input_kind = target.input_kind;
        target.kept = None;
        Ok(())
    }

    fn draft(&mut self, _input: u32, _depth: usize) -> Result<u32> {
        anyhow::bail!("staged DSpark requires block verification")
    }

    fn target_decode(&mut self, input: u32) -> Result<u32> {
        let graph = self
            .spec
            .as_ref()
            .and_then(|s| s.staged.as_ref())
            .context("staged target state missing")?
            .options
            .graph;
        let prediction = self.staged_verify(&[input], graph)?[0];
        self.staged_select(1)?;
        <Self as crate::mtp::batched::BatchedBackend>::repair_mtp(self, &[input])?;
        Ok(prediction)
    }

    fn canonical_row(&mut self, _input: u32) -> Result<()> {
        ensure!(
            self.position < self.capacity - 1,
            "staged canonical row capacity"
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl crate::mtp::batched::BatchedBackend for Engine {
    fn draft_verify(
        &mut self,
        anchor: u32,
        depth: usize,
        #[cfg(test)] fault: Option<crate::mtp::batched::Fault>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        ensure!(
            (1..=3).contains(&depth),
            "staged DSpark depth must be 1..=3"
        );
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        let target = spec
            .staged
            .as_ref()
            .context("staged target state missing")?;
        ensure!(
            target.position == Some(self.position) && target.rows == depth + 1,
            "staged draft checkpoint mismatch"
        );
        let graph = target.options.graph;
        let mut proposals = spec.draft.propose_joint(
            self.moe_slots
                .as_mut()
                .context("staged target pool missing")?,
            anchor,
            self.position,
            (self.weights.embedding(), &self.weights.head),
            super::super::super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
            },
            graph,
        )?;
        proposals.truncate(depth);
        #[cfg(test)]
        {
            if let Some(forced) = &spec.forced_proposals {
                ensure!(forced.len() >= depth, "staged forced proposal extent");
                proposals.copy_from_slice(&forced[..depth]);
            }
            for i in 0..depth {
                crate::mtp::batched::inject(fault, crate::mtp::batched::Fault::AfterDraft(i))?;
            }
        }
        let inputs: Vec<_> = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect();
        let verify = self.staged_verify(&inputs, graph)?;
        Ok((proposals, verify))
    }

    fn select_prefix(&mut self, rows: usize) -> Result<()> {
        self.staged_select(rows)
    }

    fn restore_mtp(&mut self) -> Result<()> {
        let target = self
            .spec
            .as_ref()
            .and_then(|s| s.staged.as_ref())
            .context("staged target state missing")?;
        if target.kept.is_none() {
            self.staged_select(target.rows)?;
        }
        Ok(())
    }

    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()> {
        let spec = self.spec.as_mut().context("staged attachment missing")?;
        let target = spec
            .staged
            .as_ref()
            .context("staged target state missing")?;
        let origin = target.position.context("staged checkpoint is invalid")?;
        ensure!(
            target.kept == Some(inputs.len()) && self.position == origin + inputs.len(),
            "staged repair prefix mismatch"
        );
        if target.options.graph {
            spec.inject_graphs
                .get(inputs.len() - 1)
                .and_then(Option::as_ref)
                .context("staged injection graph shape missing")?
                .launch()?;
        } else {
            spec.draft.inject_device(
                &self
                    .hidden_capture
                    .as_ref()
                    .context("staged taps missing")?
                    .values,
                inputs.len(),
                origin,
                &target.origin,
                super::super::super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                },
            )?;
        }
        self.committed_prefix_ids.extend_from_slice(inputs);
        self.resident_ids_len = inputs.len();
        self.last_input_kind = Some(PrefixLastInputKind::PrefillChunk);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Engine {
    pub(super) fn attach_staged(&mut self, path: &Path, depth: usize) -> Result<()> {
        ensure!(
            self.spec.is_none() && self.position == 0 && !self.spec_budget.is_tree(),
            "staged attachment requires an empty chain engine"
        );
        let draft = super::super::super::draft::Draft::new(
            path,
            &self.device,
            &self.config,
            self.capacity,
            self.workspace_rows,
            &mut self.blas,
            super::super::super::markov::ScoreRounding::F32,
        )?;
        self.hidden_capture = Some(HiddenCapture::new(
            &draft.config.target_layer_ids,
            &self.config,
            self.workspace_rows,
            &self.device.stream,
        )?);
        let mut spec = Spec::new(draft, self)?;
        let mut target = TargetState::new(self)?;
        target.options.depth = depth;
        spec.staged = Some(target);
        self.spec = Some(spec);
        Ok(())
    }

    pub(super) fn capture_staged(&mut self) -> Result<()> {
        let mut spec = self.spec.take().context("staged attachment missing")?;
        let result = (|| {
            self.device.finish_upload()?;
            for _ in 0..6 {
                self.closure.capture()?;
            }
            spec.draft.capture_staged(
                (self.weights.embedding(), &self.weights.head),
                super::super::super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                },
            )?;
            let state = spec
                .staged
                .as_ref()
                .context("staged target state missing")?;
            let stream = self.device.stream.clone();
            let before = self.device.ctx.mem_get_info()?.0;
            for rows in 1..=4 {
                self.closure.capture()?;
                let graph = super::super::moe::graphs::record_stream(&stream, || {
                    spec.draft.inject_device(
                        &self
                            .hidden_capture
                            .as_ref()
                            .context("staged taps missing")?
                            .values,
                        rows,
                        1,
                        &state.origin,
                        super::super::super::draft::Compute {
                            ops: &self.ops,
                            blas: &mut self.blas,
                        },
                    )
                })?;
                graph.upload()?;
                stream.synchronize()?;
                spec.inject_graphs[rows - 1] = Some(graph);
            }
            let after = self.device.ctx.mem_get_info()?.0;
            eprintln!(
                "{}",
                serde_json::json!({"staged_inject_graph_binding": {
                    "segments": 4, "free_before_bytes": before, "free_after_bytes": after,
                    "allocation_bytes": before.saturating_sub(after)
                }})
            );
            Ok(())
        })();
        self.spec = Some(spec);
        result
    }

    pub(super) fn staged_head(&mut self, rows: usize) -> Result<()> {
        let spec = self.spec.as_mut().context("staged verifier missing")?;
        let state = spec
            .staged
            .as_mut()
            .context("staged verifier state missing")?;
        let s = &self.device.stream;
        let (h, v) = (self.config.hidden_size, self.config.vocab_size);
        ensure!((1..=4).contains(&rows), "staged verification row extent");
        let logits = self
            .logits_f32
            .as_mut()
            .context("staged target FP32 logits missing")?;
        for row in 0..rows {
            self.ops.lm_head_f32(
                s,
                &self.work.n.slice(row * h..(row + 1) * h),
                &self.weights.head,
                logits,
            )?;
            self.ops.argmax_f32(s, logits, &mut self.work.token)?;
            s.memcpy_dtod(logits, &mut state.logits.slice_mut(row * v..(row + 1) * v))?;
            s.memcpy_dtod(
                &self.work.token,
                &mut spec.predictions.slice_mut(row..row + 1),
            )?;
        }
        Ok(())
    }
}

fn projection(loader: &mut Loader<'_>, name: &str, n: usize, k: usize) -> Result<Fp8Projection> {
    Ok(Fp8Projection {
        codes: loader.load_bytes(
            &format!("{name}.weight"),
            &[n, k],
            safetensors::Dtype::F8_E4M3,
        )?,
        scale: loader.load_bytes(
            &format!("{name}.scale"),
            &[n.div_ceil(32), k.div_ceil(32)],
            safetensors::Dtype::F8_E8M0,
        )?,
    })
}

pub(super) fn load_layer(
    loader: &mut Loader<'_>,
    p: &str,
    mix: &MixerGeometry,
    ffn: &FfnGeometry,
    binding: HcBinding,
) -> Result<(
    HcSublayerWeights,
    HcSublayerWeights,
    MixerWeights,
    FfnWeights,
)> {
    let h = mix.hidden;
    let i = ffn.shared_intermediate;
    let hc = |loader: &mut Loader<'_>, role: &str| -> Result<HcSublayerWeights> {
        let basis = usize::try_from(super::super::super::mhc::mix_rows(binding.params.mult))?;
        Ok(HcSublayerWeights {
            fn_w: loader.load_bytes(
                &format!("{p}.hc_{role}_fn"),
                &[basis, binding.params.mult as usize * h],
                safetensors::Dtype::F32,
            )?,
            base: loader.load_bytes(
                &format!("{p}.hc_{role}_base"),
                &[basis],
                safetensors::Dtype::F32,
            )?,
            scale: loader.load_bytes(
                &format!("{p}.hc_{role}_scale"),
                &[3],
                safetensors::Dtype::F32,
            )?,
        })
    };
    let attn_hc = hc(loader, "attn")?;
    let ffn_hc = hc(loader, "ffn")?;
    let mut bytes = |name: &str, shape: &[usize], dtype| {
        loader.load_bytes(&format!("{p}.{name}"), shape, dtype)
    };
    let q_norm = bytes(
        "attn.q_norm.weight",
        &[mix.q_lora],
        safetensors::Dtype::BF16,
    )?;
    let kv_norm = bytes(
        "attn.kv_norm.weight",
        &[mix.head_dim],
        safetensors::Dtype::BF16,
    )?;
    let attn_sink = bytes("attn.attn_sink", &[mix.heads], safetensors::Dtype::F32)?;
    let o_width = mix.o_groups * mix.o_rank;
    let wo_a = bytes(
        "attn.wo_a.weight",
        &[o_width, mix.o_in],
        safetensors::Dtype::F8_E4M3,
    )?;
    let wo_a_scale = bytes(
        "attn.wo_a.scale",
        &[o_width / 32, mix.o_in / 32],
        safetensors::Dtype::F8_E8M0,
    )?;
    let norm = bytes("attn_norm.weight", &[h], safetensors::Dtype::BF16)?;
    let post_norm = bytes("ffn_norm.weight", &[h], safetensors::Dtype::BF16)?;
    let gate_weight = bytes(
        "ffn.gate.weight",
        &[ffn.experts, h],
        safetensors::Dtype::BF16,
    )?;
    let gate_bias = bytes("ffn.gate.bias", &[ffn.experts], safetensors::Dtype::F32)?;
    let shared_gate_codes = bytes(
        "ffn.shared_experts.w1.weight",
        &[i, h],
        safetensors::Dtype::F8_E4M3,
    )?;
    let shared_gate_scales = bytes(
        "ffn.shared_experts.w1.scale",
        &[i.div_ceil(32), h / 32],
        safetensors::Dtype::F8_E8M0,
    )?;
    let shared_up_codes = bytes(
        "ffn.shared_experts.w3.weight",
        &[i, h],
        safetensors::Dtype::F8_E4M3,
    )?;
    let shared_up_scales = bytes(
        "ffn.shared_experts.w3.scale",
        &[i.div_ceil(32), h / 32],
        safetensors::Dtype::F8_E8M0,
    )?;
    let shared_down_codes = bytes(
        "ffn.shared_experts.w2.weight",
        &[h, i],
        safetensors::Dtype::F8_E4M3,
    )?;
    let shared_down_scales = bytes(
        "ffn.shared_experts.w2.scale",
        &[h.div_ceil(32), i / 32],
        safetensors::Dtype::F8_E8M0,
    )?;
    Ok((
        attn_hc,
        ffn_hc,
        MixerWeights {
            wq_a: projection(loader, &format!("{p}.attn.wq_a"), mix.q_lora, h)?,
            q_norm,
            wq_b: projection(
                loader,
                &format!("{p}.attn.wq_b"),
                mix.heads * mix.head_dim,
                mix.q_lora,
            )?,
            wkv: projection(loader, &format!("{p}.attn.wkv"), mix.head_dim, h)?,
            kv_norm,
            attn_sink,
            wo_a,
            wo_a_scale,
            wo_b: projection(loader, &format!("{p}.attn.wo_b"), h, o_width)?,
        },
        FfnWeights {
            gate_weight,
            gate_bias,
            norm,
            post_norm,
            shared_gate_codes,
            shared_gate_scales,
            shared_up_codes,
            shared_up_scales,
            shared_down_codes,
            shared_down_scales,
        },
    ))
}

pub(super) fn routed(
    loader: &mut Loader<'_>,
    p: &str,
    h: usize,
    i: usize,
) -> Result<RoutedExpertWeights> {
    Ok(RoutedExpertWeights {
        gate_codes: loader.load_bytes(
            &format!("{p}.w1.weight"),
            &[i, h / 2],
            safetensors::Dtype::I8,
        )?,
        gate_scales: loader.load_bytes(
            &format!("{p}.w1.scale"),
            &[i, h / 32],
            safetensors::Dtype::F8_E8M0,
        )?,
        up_codes: loader.load_bytes(
            &format!("{p}.w3.weight"),
            &[i, h / 2],
            safetensors::Dtype::I8,
        )?,
        up_scales: loader.load_bytes(
            &format!("{p}.w3.scale"),
            &[i, h / 32],
            safetensors::Dtype::F8_E8M0,
        )?,
        down_codes: loader.load_bytes(
            &format!("{p}.w2.weight"),
            &[h, i / 2],
            safetensors::Dtype::I8,
        )?,
        down_scales: loader.load_bytes(
            &format!("{p}.w2.scale"),
            &[h, i / 32],
            safetensors::Dtype::F8_E8M0,
        )?,
    })
}

pub(crate) struct StagedDraft {
    pub config: DraftConfig,
    stream: Arc<CudaStream>,
    capacity: usize,
    rows: usize,
    hc: HcRuntime,
    mixer: DeepseekMixer,
    ffn: DeepseekFfn,
    router_weights: Vec<CudaSlice<f32>>,
    router_input: CudaSlice<f32>,
    router_cast: CudaFunction,
    slots: Option<super::super::moe::Slots>,
    main_projection: Fp8Projection,
    main_norm: CudaSlice<bf16>,
    final_norm: CudaSlice<bf16>,
    markov_embed: CudaSlice<bf16>,
    markov_head: CudaSlice<bf16>,
    confidence: CudaSlice<bf16>,
    rings: Vec<LatentRing>,
    checkpoint: Vec<LatentRing>,
    block: BlockWindow,
    sequence: CudaSlice<i32>,
    position: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    prepare: CudaFunction,
    ids: CudaSlice<u32>,
    tokens: CudaSlice<u32>,
    main: CudaSlice<bf16>,
    main_codes: CudaSlice<u8>,
    main_scales: CudaSlice<u8>,
    kv: CudaSlice<bf16>,
    x: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    base: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    scores: CudaSlice<f32>,
    parent: CudaSlice<bf16>,
    confidences: CudaSlice<f32>,
    host_confidence: Vec<f32>,
    sum: CudaFunction,
    confidence_dot: CudaFunction,
    head_projection: CudaFunction,
    #[cfg(feature = "cuda")]
    graphs: Vec<(CudaGraph, CudaGraph)>,
}

impl StagedDraft {
    #[cfg(all(test, feature = "cuda"))]
    pub(in crate::backend::cuda) fn test_bits(&self) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for values in [&self.x, &self.n, &self.hc.stream] {
            super::super::super::copy::bits(&self.stream, values, &mut output)?;
        }
        for values in [
            &self.base,
            &self.scores,
            &self.confidences,
            &self.hc.mix_pre,
            &self.hc.mix_post,
            &self.hc.mix_comb,
        ] {
            for value in self.stream.clone_dtoh(values)? {
                output.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for token in self.stream.clone_dtoh(&self.tokens)? {
            output.extend_from_slice(&token.to_le_bytes());
        }
        self.state_bits(0, 128, &mut output)?;
        Ok(output)
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn resident(
        path: &Path,
        d: &Device,
        config: DraftConfig,
        capacity: usize,
        rows: usize,
        blas: &mut Blas,
    ) -> Result<Self> {
        blas.enable_fp8(&d.ctx)?;
        Self::new(path, d, config, capacity, rows, (false, blas))
    }

    pub(crate) fn load(
        path: &Path,
        d: &Device,
        config: DraftConfig,
        capacity: usize,
        rows: usize,
        blas: &Blas,
    ) -> Result<Self> {
        Self::new(path, d, config, capacity, rows, (true, blas))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn capture(
        &mut self,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        mut compute: Compute<'_>,
    ) -> Result<()> {
        ensure!(
            self.ffn.tiered && self.graphs.is_empty(),
            "staged graph binding is not empty/tiered"
        );
        let stream = self.stream.clone();
        let before = stream.context().mem_get_info()?.0;
        let mut graphs = Vec::with_capacity(self.config.num_hidden_layers);
        for stage in 0..self.config.num_hidden_layers {
            let prefix = super::super::moe::graphs::record_stream(&stream, || {
                self.prefix(stage, 1, target.0, &mut compute, None)
            })?;
            let suffix = super::super::moe::graphs::record_stream(&stream, || {
                self.ffn.contribute::<true>(
                    &stream,
                    self.config.block_size,
                    (&self.n, &mut self.out),
                    stage,
                    Accumulator::Fp32,
                    None,
                )?;
                self.suffix(stage, None)?;
                if stage + 1 == self.config.num_hidden_layers {
                    self.heads(target.1, compute.ops, None, None)?;
                }
                Ok(())
            })?;
            prefix.upload()?;
            suffix.upload()?;
            stream.synchronize()?;
            graphs.push((prefix, suffix));
        }
        let after = stream.context().mem_get_info()?.0;
        eprintln!(
            "{}",
            serde_json::json!({"staged_graph_binding": {
                "segments": graphs.len() * 2, "free_before_bytes": before, "free_after_bytes": after,
                "allocation_bytes": before.saturating_sub(after)
            }})
        );
        self.graphs = graphs;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(in crate::backend::cuda) fn bind_slots(
        &mut self,
        slots: super::super::moe::Slots,
    ) -> Result<()> {
        ensure!(
            self.ffn.tiered && self.slots.is_none(),
            "staged expert service already bound or resident"
        );
        self.slots = Some(slots);
        Ok(())
    }

    fn new(
        path: &Path,
        d: &Device,
        config: DraftConfig,
        capacity: usize,
        rows: usize,
        execution: (bool, &Blas),
    ) -> Result<Self> {
        let (tiered, blas) = execution;
        let spec = ModelSpec::read(path)?;
        let staged = config
            .staged
            .as_ref()
            .context("staged draft config missing")?;
        let geometry = staged.catalogue.geometry();
        ensure!(
            rows >= config.block_size && capacity >= rows,
            "staged draft capacity"
        );
        let target_capacity = capacity;
        let capacity = target_capacity
            .checked_add(config.block_size)
            .filter(|&extent| i32::try_from(extent).is_ok())
            .context("staged draft scratch position extent exceeds i32")?;
        let mut binding = deepseek_hc_binding(&spec)?;
        binding.rounding = HcRounding::Fp32ResidualSum;
        let mix = MixerGeometry::new(&spec)?;
        let mut fg = FfnGeometry::new(&spec)?;
        fg.experts = geometry.experts as usize;
        fg.top_k = geometry.top_k as usize;
        let (h, b, rank, stages, window, width) = (
            config.hidden_size,
            config.block_size,
            config.markov_rank,
            config.num_hidden_layers,
            mix.window,
            mix.head_dim,
        );
        let s = &d.stream;
        let mut hc = HcRuntime::new(&d.ctx, s, binding, stages, b)?;
        let mut mixer = DeepseekMixer::new_range(
            &d.ctx,
            s,
            &spec,
            spec.layers.len()..spec.layers.len() + stages,
            rows,
        )?;
        let rotary =
            super::super::super::cubin::module(&d.ctx, "rope")?.load_function("rope_inv_freq")?;
        for inverse in &mut mixer.inv_freq {
            unsafe {
                s.launch_builder(&rotary)
                    .arg(inverse)
                    .arg(&spec.rope_theta)
                    .arg(&(mix.rotary as i32))
                    .launch(super::super::super::ops::flat(mix.rotary / 2))?;
            }
        }
        let mut ffn = DeepseekFfn::new_scoped(&d.ctx, s, &spec, geometry, rows)?;
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let mut manifest = mapped.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        for (name, entry) in &mut manifest.entries {
            entry.load = entry.scope == crate::checkpoint::TensorScope::Mtp
                && !name.ends_with(".ffn.gate.bias_vl")
                && (!tiered || !name.contains(".ffn.experts."));
        }
        let active_nonexpert = manifest
            .entries
            .iter()
            .filter(|(name, entry)| entry.load && !name.contains(".ffn.experts."))
            .count();
        let text_vl_deferred = manifest
            .entries
            .iter()
            .filter(|(name, entry)| {
                entry.scope == crate::checkpoint::TensorScope::Mtp
                    && name.ends_with(".ffn.gate.bias_vl")
                    && !entry.load
            })
            .count();
        ensure!(
            active_nonexpert == 94 && text_vl_deferred == 3,
            "staged text manifest tensor counts differ"
        );
        let expected = staged.resident_weight_bytes
            + if tiered {
                0
            } else {
                staged.catalogue.slot_bytes()
                    * u64::from(geometry.layers)
                    * u64::from(geometry.experts)
            };
        let weights = super::super::super::weights::checkpoint_mapped(
            &mapped,
            d,
            Some(&manifest),
            false,
            |l| {
                let mut ffns = Vec::new();
                let mut routed_layers = Vec::new();
                for stage in 0..stages {
                    let p = format!("mtp.{stage}");
                    let (a, f, m, w) = load_layer(l, &p, &mix, &fg, binding)?;
                    hc.bind_layer(stage, a, f)?;
                    mixer.bind_layer(stage, m)?;
                    ffns.push(w);
                    let mut experts = Vec::new();
                    if !tiered {
                        for expert in 0..fg.experts {
                            experts.push(routed(
                                l,
                                &format!("{p}.ffn.experts.{expert}"),
                                h,
                                fg.expert_intermediate,
                            )?);
                        }
                    }
                    routed_layers.push(experts);
                }
                let main = projection(l, "mtp.0.main_proj", h, config.capture_width())?;
                let mut bf = |name: &str, shape: Vec<usize>| l.load(&[(name.into(), shape)]);
                Ok((
                    ffns,
                    main,
                    bf("mtp.0.main_norm.weight", vec![h])?,
                    bf("mtp.2.norm.weight", vec![h])?,
                    bf(
                        "mtp.2.markov_head.embed.weight",
                        vec![config.vocab_size, rank],
                    )?,
                    bf(
                        "mtp.2.markov_head.head.weight",
                        vec![config.vocab_size, rank],
                    )?,
                    bf("mtp.2.confidence_head.proj.weight", vec![1, h + rank])?,
                    routed_layers,
                    l.bytes,
                ))
            },
        )?
        .0;
        ensure!(
            weights.8 as u64 == expected,
            "staged upload bytes={} expected={expected}",
            weights.8
        );
        eprintln!(
            "PASS STAGED_NATIVE_BIND stages={stages} expert_catalogue={} active_nonexpert={active_nonexpert} text_vl_deferred={text_vl_deferred} uploaded_bytes={} tiered={tiered} target_capacity={target_capacity} draft_capacity={capacity} missing=0 unexpected=0 initialized=0",
            geometry.layers * geometry.experts,
            weights.8
        );
        let mut router_weights = Vec::with_capacity(stages);
        for (stage, w) in weights.0.into_iter().enumerate() {
            let bytes = s.clone_dtoh(&w.gate_weight)?;
            ensure!(
                bytes.len() == fg.experts * h * 2,
                "staged router storage extent"
            );
            let promoted: Vec<_> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect();
            router_weights.push(s.clone_htod(&promoted)?);
            ffn.bind_layer(s, stage, w)?;
        }
        for (stage, experts) in weights.7.into_iter().enumerate() {
            for (expert, weight) in experts.into_iter().enumerate() {
                ffn.bind_routed(stage, expert, weight)?;
            }
        }
        if tiered {
            ffn.bind_tier(s)?;
        } else {
            ffn.finish_routed(s)?;
        }
        d.finish_upload()?;
        hc.prepare_f32_blas(s, blas)?;
        let rings = (0..stages)
            .map(|_| {
                Ok(LatentRing {
                    codes: s.alloc_zeros(window * width)?,
                    scales: s.alloc_zeros(window * width / 32)?,
                })
            })
            .collect::<Result<_>>()?;
        let checkpoint = (0..stages)
            .map(|_| {
                Ok(LatentRing {
                    codes: s.alloc_zeros(4 * width)?,
                    scales: s.alloc_zeros(4 * width / 32)?,
                })
            })
            .collect::<Result<_>>()?;
        let module = super::super::super::cubin::module(&d.ctx, "markov")?;
        let capture = config.capture_width();
        let noise = config.mask_token_id;
        Ok(Self {
            config,
            stream: s.clone(),
            capacity,
            rows,
            hc,
            mixer,
            ffn,
            router_weights,
            router_input: s.alloc_zeros(b * h)?,
            router_cast: super::super::super::cubin::module(&d.ctx, "vision_ops")?
                .load_function("vision_upcast_f32")?,
            slots: None,
            main_projection: weights.1,
            main_norm: weights.2,
            final_norm: weights.3,
            markov_embed: weights.4,
            markov_head: weights.5,
            confidence: weights.6,
            rings,
            checkpoint,
            block: BlockWindow::new(s, window, width, b)?,
            sequence: s.alloc_zeros(8)?,
            position: s.alloc_zeros(1)?,
            positions: s.alloc_zeros(rows * 3)?,
            prepare: super::super::super::cubin::module(&d.ctx, "rope")?
                .load_function("sequence_positions_causal")?,
            ids: s.clone_htod(&vec![noise; b])?,
            tokens: s.alloc_zeros(b + 1)?,
            main: s.alloc_zeros(rows * h)?,
            main_codes: s.alloc_zeros(rows * capture)?,
            main_scales: s.alloc_zeros(rows * capture / 32)?,
            kv: s.alloc_zeros(rows * width)?,
            x: s.alloc_zeros(b * h)?,
            n: s.alloc_zeros(b * h)?,
            qkv: s.alloc_zeros(b * (mix.heads * width + width))?,
            out: s.alloc_zeros(b * h)?,
            base: s.alloc_zeros(b * spec.vocab as usize)?,
            bias: s.alloc_zeros(spec.vocab as usize)?,
            scores: s.alloc_zeros(b * spec.vocab as usize)?,
            parent: s.alloc_zeros(rank)?,
            confidences: s.alloc_zeros(b)?,
            host_confidence: vec![0.0; b],
            sum: module.load_function("sum_argmax_f32")?,
            confidence_dot: module.load_function("concat_dot")?,
            head_projection: super::super::super::cubin::module(&d.ctx, "gemm_skinny")?
                .load_function("gemm_skinny_bf16_f32")?,
            #[cfg(feature = "cuda")]
            graphs: Vec::new(),
        })
    }
}

fn bf_trace(
    s: &Arc<CudaStream>,
    trace: &mut Option<&mut crate::trace::Trace>,
    name: &str,
    shape: Vec<usize>,
    data: &CudaSlice<bf16>,
) -> Result<()> {
    if let Some(t) = trace.as_deref_mut() {
        let count = shape.iter().product::<usize>();
        t.add(
            name.into(),
            shape,
            s.clone_dtoh(&data.slice(..count))?
                .into_iter()
                .map(bf16::to_f32)
                .collect(),
        )?;
    }
    Ok(())
}

fn fp_trace(
    s: &Arc<CudaStream>,
    trace: &mut Option<&mut crate::trace::Trace>,
    name: &str,
    shape: Vec<usize>,
    data: &CudaSlice<f32>,
) -> Result<()> {
    if let Some(t) = trace.as_deref_mut() {
        let count = shape.iter().product::<usize>();
        t.add(name.into(), shape, s.clone_dtoh(&data.slice(..count))?)?;
    }
    Ok(())
}

impl StagedDraft {
    fn sequence(&self, rows: usize) -> Result<SeqRef> {
        ensure!(
            rows > 0 && rows <= self.rows && rows <= 64,
            "staged sequence row extent"
        );
        unsafe {
            self.stream
                .launch_builder(&self.prepare)
                .arg(&self.position)
                .arg(&self.sequence)
                .arg(&self.positions)
                .arg(&(rows as i32))
                .arg(&(self.capacity as i32))
                .arg(&0i32)
                .launch(super::super::super::ops::grid(1, 256))?;
        }
        Ok(SeqRef {
            ptr: self.sequence.device_ptr(&self.stream).0,
        })
    }

    pub(crate) fn inject(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        position: Option<&CudaSlice<i32>>,
        compute: Compute<'_>,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<()> {
        ensure!(
            rows > 0
                && rows <= self.rows
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "staged injection capacity"
        );
        let s = &self.stream;
        if let Some(position) = position {
            s.memcpy_dtod(position, &mut self.position)?;
        } else {
            s.memcpy_htod(&[start as i32], &mut self.position)?;
        }
        let seq = self.sequence(rows)?;
        let (h, capture) = (self.config.hidden_size, self.config.capture_width());
        ensure!(input.len() >= rows * capture, "staged tap extent");
        bf_trace(
            s,
            &mut trace,
            "input.main_hidden.0",
            vec![1, rows, capture],
            input,
        )?;
        self.mixer.ops.act_quant(
            s,
            view(
                input,
                s,
                0,
                [rows as u32, capture as u32],
                [capture as u64, 1],
            )?,
            &mut TensorMut(view(
                &self.main_codes,
                s,
                0,
                [rows as u32, capture as u32],
                [capture as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.main_scales,
                s,
                0,
                [rows as u32, (capture / 32) as u32],
                [(capture / 32) as u64, 1],
            )?),
        )?;
        self.mixer.project::<true>(
            s,
            compute.blas,
            &self.main_projection,
            (
                view(
                    &self.main_codes,
                    s,
                    0,
                    [rows as u32, capture as u32],
                    [capture as u64, 1],
                )?,
                view(
                    &self.main_scales,
                    s,
                    0,
                    [rows as u32, (capture / 32) as u32],
                    [(capture / 32) as u64, 1],
                )?,
            ),
            TensorMut(view(
                &self.main,
                s,
                0,
                [rows as u32, h as u32],
                [h as u64, 1],
            )?),
        )?;
        bf_trace(s, &mut trace, "main_proj.0", vec![1, rows, h], &self.main)?;
        let main = view(
            &self.main,
            s,
            0,
            [rows as u32, 1, h as u32],
            [h as u64, h as u64, 1],
        )?;
        self.mixer.common.norm.rms_with(
            s,
            RowDomain {
                meta: seq,
                row_base: 0,
                row_capacity: rows as u32,
            },
            main,
            view(&self.main_norm, s, 0, [h as u32], [1])?,
            &mut TensorMut(main),
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps: self.config.rms_norm_eps,
            },
        )?;
        bf_trace(s, &mut trace, "main_norm.0", vec![1, rows, h], &self.main)?;
        for stage in 0..self.config.num_hidden_layers {
            self.mixer.seed(
                s,
                compute.blas,
                compute.ops,
                &self.rings[stage],
                MixerStep {
                    position: start,
                    rows,
                    x: &self.main,
                    qkv: &mut self.kv,
                    out: &mut self.out,
                    seq,
                    block: None,
                },
                stage,
            )?;
        }
        Ok(())
    }

    pub(crate) fn body(
        &mut self,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        trace: Option<&mut crate::trace::Trace>,
        forced: Option<&[u32]>,
    ) -> Result<()> {
        let mut slots = self.slots.take();
        let result = self.body_tier(start, target, compute, trace, forced, slots.as_mut());
        self.slots = slots;
        result
    }

    fn body_tier(
        &mut self,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        mut compute: Compute<'_>,
        mut trace: Option<&mut crate::trace::Trace>,
        forced: Option<&[u32]>,
        mut tier: Option<&mut super::super::moe::Slots>,
    ) -> Result<()> {
        let (b, h, v) = (
            self.config.block_size,
            self.config.hidden_size,
            self.config.vocab_size,
        );
        ensure!(
            start > 0 && start.checked_add(b).is_some_and(|end| end <= self.capacity),
            "staged proposal position"
        );
        ensure!(
            target.0.len() == v * h && target.1.len() == v * h,
            "staged shared embedding/head extent"
        );
        if let Some(ids) = forced {
            ensure!(
                ids.len() == b && ids.iter().all(|id| (*id as usize) < v),
                "staged forced ids"
            );
        }
        ensure!(
            !self.ffn.tiered || tier.is_some(),
            "staged expert service is unbound"
        );
        for stage in 0..self.config.num_hidden_layers {
            self.prefix(stage, start, target.0, &mut compute, trace.as_deref_mut())?;
            if let Some(tier) = tier.as_deref_mut() {
                self.ffn.service_tier(&self.stream, b, stage, tier)?;
            }
            self.ffn.contribute::<true>(
                &self.stream,
                b,
                (&self.n, &mut self.out),
                stage,
                Accumulator::Fp32,
                tier.as_deref_mut(),
            )?;
            self.suffix(stage, trace.as_deref_mut())?;
        }
        self.heads(target.1, compute.ops, trace, forced)
    }

    fn prefix(
        &mut self,
        stage: usize,
        start: usize,
        embedding: &CudaSlice<bf16>,
        compute: &mut Compute<'_>,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<()> {
        let (b, h) = (self.config.block_size, self.config.hidden_size);
        let s = &self.stream;
        let seq = if stage == 0 {
            let seq = self.sequence(b)?;
            s.memcpy_dtod(&self.ids.slice(..1), &mut self.tokens.slice_mut(..1))?;
            self.hc.begin_forward(s, b, &self.ids, embedding)?;
            if let Some(t) = trace.as_deref_mut() {
                t.add(
                    "input.anchor.0".into(),
                    vec![1],
                    s.clone_dtoh(&self.ids.slice(..1))?
                        .into_iter()
                        .map(|x| x as f32)
                        .collect(),
                )?;
            }
            seq
        } else {
            SeqRef {
                ptr: self.sequence.device_ptr(s).0,
            }
        };
        self.hc.hc_pre_blas(
            s,
            compute.blas,
            (stage, HcSublayer::Attention, HcPreSource::Carried),
            b,
            &mut self.x,
        )?;
        self.mix_trace(&mut trace, stage, "attn")?;
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.collapse.0"),
            vec![1, b, h],
            &self.x,
        )?;
        self.ffn.norm(
            s,
            NormStep {
                seq,
                rows: b,
                x: &self.x,
                n: &mut self.n,
                layer: stage,
                post: false,
            },
        )?;
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.attn_norm.0"),
            vec![1, b, h],
            &self.n,
        )?;
        #[cfg(test)]
        let query_capture = trace.is_some();
        #[cfg(test)]
        if query_capture {
            self.mixer
                .query_stores
                .borrow_mut()
                .insert("staged".into(), Vec::new());
        }
        self.mixer.run::<true>(
            s,
            compute.blas,
            compute.ops,
            &self.rings[stage],
            MixerStep {
                position: start,
                rows: b,
                x: &self.n,
                qkv: &mut self.qkv,
                out: &mut self.out,
                seq,
                block: Some(&mut self.block),
            },
            stage,
        )?;
        #[cfg(test)]
        if query_capture {
            let g = &self.mixer.geometry;
            let q = b * g.heads * g.head_dim;
            let mut stores = self.mixer.query_stores.borrow_mut();
            stores.remove("staged");
            stores.insert("q_rotary".into(), s.clone_dtoh(&self.qkv.slice(..q))?);
            stores.insert(
                "kv_rotary".into(),
                s.clone_dtoh(&self.qkv.slice(q..q + b * g.head_dim))?,
            );
            stores.insert("o_inverse".into(), s.clone_dtoh(&self.mixer.o.slice(..q))?);
            stores.insert(
                "wo_a".into(),
                s.clone_dtoh(&self.mixer.groups.slice(..b * g.o_groups * g.o_rank))?,
            );
            for name in [
                "q_a",
                "q_norm",
                "q_b",
                "kv_a",
                "kv_norm",
                "q_rotary",
                "kv_rotary",
                "o_inverse",
                "wo_a",
            ] {
                let values = stores
                    .remove(name)
                    .with_context(|| format!("staged attention primitive {name} missing"))?;
                stores.insert(format!("stage{stage}.{name}"), values);
            }
        }
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.attn.0"),
            vec![1, b, h],
            &self.out,
        )?;
        self.hc
            .hc_post(s, stage, HcSublayer::Attention, b, &self.out)?;
        self.hc.hc_pre_blas(
            s,
            compute.blas,
            (stage, HcSublayer::Ffn, HcPreSource::AttentionOwn),
            b,
            &mut self.x,
        )?;
        self.mix_trace(&mut trace, stage, "ffn")?;
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.collapse.1"),
            vec![1, b, h],
            &self.x,
        )?;
        self.ffn.norm(
            s,
            NormStep {
                seq,
                rows: b,
                x: &self.x,
                n: &mut self.n,
                layer: stage,
                post: true,
            },
        )?;
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.ffn_norm.0"),
            vec![1, b, h],
            &self.n,
        )?;
        let count = u32::try_from(b * h)?;
        unsafe {
            s.launch_builder(&self.router_cast)
                .arg(&self.n)
                .arg(&mut self.router_input)
                .arg(&count)
                .launch(super::super::super::ops::flat(b * h))?;
        }
        let experts = self.ffn.geometry.experts;
        compute.blas.linear_f32(
            view(
                &self.router_weights[stage],
                s,
                0,
                [experts as u32, h as u32],
                [h as u64, 1],
            )?,
            view(
                &self.router_input,
                s,
                0,
                [b as u32, h as u32],
                [h as u64, 1],
            )?,
            &mut TensorMut(view(
                &self.ffn.logits,
                s,
                0,
                [b as u32, experts as u32],
                [experts as u64, 1],
            )?),
        )?;
        self.ffn.select_only::<true>(s, b, stage)?;
        #[cfg(test)]
        self.ffn.set_projection_capture(s, trace.is_some())?;
        fp_trace(
            s,
            &mut trace,
            &format!("stage{stage}.route.0.0"),
            vec![b, self.ffn.geometry.top_k],
            &self.ffn.route_weights,
        )?;
        #[cfg(test)]
        fp_trace(
            s,
            &mut trace,
            &format!("stage{stage}.route_logits"),
            vec![b, self.ffn.geometry.experts],
            &self.ffn.logits,
        )?;
        if let Some(t) = trace {
            let ids = s.clone_dtoh(&self.ffn.ids.slice(..b * self.ffn.geometry.top_k))?;
            t.add(
                format!("stage{stage}.route.0.1"),
                vec![b, self.ffn.geometry.top_k],
                ids.into_iter().map(|v| v as f32).collect(),
            )?;
        }
        Ok(())
    }

    fn suffix(&mut self, stage: usize, mut trace: Option<&mut crate::trace::Trace>) -> Result<()> {
        let (b, h) = (self.config.block_size, self.config.hidden_size);
        let s = &self.stream;
        #[cfg(test)]
        if let Some(t) = trace.as_deref_mut() {
            let g = &self.ffn.geometry;
            let routed = self
                .ffn
                .projected_routed
                .as_ref()
                .context("routed projection capture missing")?;
            let shared = self
                .ffn
                .projected_shared
                .as_ref()
                .context("shared projection capture missing")?;
            for (role, shape, data) in [
                ("routed", vec![b, g.top_k, 3, g.expert_intermediate], routed),
                ("shared", vec![b, 3, g.shared_intermediate], shared),
                (
                    "contributions",
                    vec![b, g.top_k, h],
                    &self.ffn.contributions,
                ),
                ("shared_out", vec![b, h], &self.ffn.shared_out),
            ] {
                let count = shape.iter().product::<usize>();
                t.add(
                    format!("stage{stage}.ffn_projections.{role}"),
                    shape,
                    s.clone_dtoh(&data.slice(..count))?
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect(),
                )?;
            }
        }
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.ffn.0"),
            vec![1, b, h],
            &self.out,
        )?;
        self.hc.hc_post(s, stage, HcSublayer::Ffn, b, &self.out)?;
        bf_trace(
            s,
            &mut trace,
            &format!("stage{stage}.block.0.0"),
            vec![1, b, self.hc.binding.params.mult as usize, h],
            &self.hc.stream,
        )?;
        fp_trace(
            s,
            &mut trace,
            &format!("stage{stage}.block.0.1"),
            vec![1, b, self.hc.binding.params.mult as usize],
            &self.hc.mix_pre,
        )?;
        Ok(())
    }

    fn heads(
        &mut self,
        head: &CudaSlice<bf16>,
        ops: &Ops,
        mut trace: Option<&mut crate::trace::Trace>,
        forced: Option<&[u32]>,
    ) -> Result<()> {
        let (b, h, v, rank) = (
            self.config.block_size,
            self.config.hidden_size,
            self.config.vocab_size,
            self.config.markov_rank,
        );
        let s = &self.stream;
        let seq = SeqRef {
            ptr: self.sequence.device_ptr(s).0,
        };
        self.hc.collapse_final(s, b, &mut self.x)?;
        bf_trace(s, &mut trace, "stage2.collapse.2", vec![1, b, h], &self.x)?;
        self.ffn
            .final_norm(s, seq, b, &self.x, &self.final_norm, &mut self.n)?;
        bf_trace(s, &mut trace, "final_norm.0", vec![1, b, h], &self.n)?;
        Bf16F32Plan::bind(
            self.head_projection.clone(),
            view(&self.n, s, 0, [b as u32, h as u32], [h as u64, 1])?,
            view(head, s, 0, [v as u32, h as u32], [h as u64, 1])?,
            view(&self.base, s, 0, [b as u32, v as u32], [v as u64, 1])?,
        )?
        .run(s)?;
        fp_trace(s, &mut trace, "base_head.0", vec![1, b, v], &self.base)?;
        for row in 0..b {
            unsafe {
                s.launch_builder(&ops.embed)
                    .arg(&self.markov_embed)
                    .arg(&self.tokens.slice(row..row + 1))
                    .arg(&mut self.parent)
                    .arg(&1i32)
                    .arg(&(rank as i32))
                    .launch(super::super::super::ops::flat(rank))?;
            }
            Bf16F32Plan::bind(
                self.head_projection.clone(),
                view(&self.parent, s, 0, [1, rank as u32], [rank as u64, 1])?,
                view(
                    &self.markov_head,
                    s,
                    0,
                    [v as u32, rank as u32],
                    [rank as u64, 1],
                )?,
                view(&self.bias, s, 0, [1, v as u32], [v as u64, 1])?,
            )?
            .run(s)?;
            fp_trace(
                s,
                &mut trace,
                &format!("markov.{row}.0"),
                vec![1, v],
                &self.bias,
            )?;
            bf_trace(
                s,
                &mut trace,
                &format!("markov.{row}.1"),
                vec![1, rank],
                &self.parent,
            )?;
            unsafe {
                s.launch_builder(&self.sum)
                    .arg(&self.base.slice(row * v..(row + 1) * v))
                    .arg(&self.bias)
                    .arg(&mut self.scores.slice_mut(row * v..(row + 1) * v))
                    .arg(&mut self.tokens.slice_mut(row + 1..row + 2))
                    .arg(&(v as i32))
                    .launch(super::super::super::ops::grid(1, 1024))?;
            }
            if let Some(ids) = forced {
                s.memcpy_htod(
                    &ids[row..row + 1],
                    &mut self.tokens.slice_mut(row + 1..row + 2),
                )?;
            }
        }
        unsafe {
            s.launch_builder(&self.confidence_dot)
                .arg(&self.x)
                .arg(&self.markov_embed)
                .arg(&self.tokens.slice(..1))
                .arg(&self.tokens.slice(1..))
                .arg(&self.confidence)
                .arg(&mut self.confidences)
                .arg(&(b as i32))
                .arg(&(h as i32))
                .arg(&(rank as i32))
                .arg(&(v as i32))
                .launch(super::super::super::ops::grid(b, 256))?;
        }
        fp_trace(s, &mut trace, "confidence.0", vec![1, b], &self.confidences)?;
        fp_trace(s, &mut trace, "output.0.1", vec![1, b, v], &self.scores)?;
        fp_trace(s, &mut trace, "output.0.2", vec![1, b], &self.confidences)?;
        if let Some(t) = trace.as_deref_mut() {
            t.add(
                "output.0.0".into(),
                vec![1, b + 1],
                s.clone_dtoh(&self.tokens)?
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
            )?;
        }
        self.cache_trace(&mut trace, "cache_after")?;
        Ok(())
    }

    fn mix_trace(
        &self,
        trace: &mut Option<&mut crate::trace::Trace>,
        stage: usize,
        role: &str,
    ) -> Result<()> {
        let (b, m) = (self.config.block_size, self.hc.binding.params.mult as usize);
        fp_trace(
            &self.stream,
            trace,
            &format!("stage{stage}.mix.{role}.0.0"),
            vec![1, b, m],
            &self.hc.mix_pre,
        )?;
        fp_trace(
            &self.stream,
            trace,
            &format!("stage{stage}.mix.{role}.0.1"),
            vec![1, b, m],
            &self.hc.mix_post,
        )?;
        fp_trace(
            &self.stream,
            trace,
            &format!("stage{stage}.mix.{role}.0.2"),
            vec![1, b, m, m],
            &self.hc.mix_comb,
        )
    }

    pub(crate) fn cache_trace(
        &self,
        trace: &mut Option<&mut crate::trace::Trace>,
        tag: &str,
    ) -> Result<()> {
        if let Some(t) = trace.as_deref_mut() {
            let g = &self.mixer.geometry;
            for (stage, ring) in self.rings.iter().enumerate() {
                let codes = self.stream.clone_dtoh(&ring.codes)?;
                let scales = self.stream.clone_dtoh(&ring.scales)?;
                ensure!(
                    scales.iter().all(|s| *s != 255),
                    "nonfinite canonical KV scale"
                );
                let values = codes
                    .iter()
                    .enumerate()
                    .map(|(i, code)| {
                        crate::fp8::decode_e4m3(*code) * 2f32.powi(i32::from(scales[i / 32]) - 127)
                    })
                    .collect();
                t.add(
                    format!("stage{stage}.{tag}.0"),
                    vec![1, g.window, g.head_dim],
                    values,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn propose(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        trace: Option<&mut crate::trace::Trace>,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        ensure!(
            (anchor as usize) < self.config.vocab_size,
            "staged anchor outside vocabulary"
        );
        let mut ids = vec![self.config.mask_token_id; self.config.block_size];
        ids[0] = anchor;
        self.stream.memcpy_htod(&ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[i32::try_from(start)?], &mut self.position)?;
        self.body(start, target, compute, trace, forced)?;
        self.proposals()
    }

    #[cfg(feature = "cuda")]
    pub(in crate::backend::cuda) fn propose_joint(
        &mut self,
        owner: &mut super::super::moe::Slots,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        graph: bool,
    ) -> Result<Vec<u32>> {
        ensure!(
            (anchor as usize) < self.config.vocab_size,
            "staged anchor outside vocabulary"
        );
        let mut ids = vec![self.config.mask_token_id; self.config.block_size];
        ids[0] = anchor;
        self.stream.memcpy_htod(&ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[i32::try_from(start)?], &mut self.position)?;
        let mut slots = self
            .slots
            .take()
            .context("staged joint expert service missing")?;
        let result = (|| {
            if !slots.request_open {
                slots.begin_staging_request()?;
            }
            slots.begin_token()?;
            let executed = slots.with_shared_pool(owner, |tier| {
                #[cfg(feature = "cuda")]
                if graph {
                    ensure!(
                        self.graphs.len() == self.config.num_hidden_layers,
                        "staged graph inventory missing"
                    );
                    for (stage, (prefix, suffix)) in self.graphs.iter().enumerate() {
                        prefix.launch()?;
                        self.ffn
                            .service_tier(&self.stream, self.config.block_size, stage, tier)?;
                        suffix.launch()?;
                        tier.record_all_readers(&self.stream)?;
                    }
                    return Ok(());
                }
                #[cfg(feature = "rocm")]
                ensure!(!graph, "staged segmented graphs require CUDA");
                self.body_tier(start, target, compute, None, None, Some(tier))
            });
            let completed = slots.finish_token();
            executed?;
            completed
        })();
        self.slots = Some(slots);
        result?;
        #[cfg(test)]
        if SUPPLY_CENSUS.with(|records| records.borrow().is_some()) {
            census_record(serde_json::json!({
                "kind": "draft_block", "position": start, "anchor": anchor,
                "tokens": self.stream.clone_dtoh(&self.tokens)?,
                "confidence": self.stream.clone_dtoh(&self.confidences)?,
            }));
        }
        self.proposals()
    }

    pub(crate) fn propose_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream
            .memcpy_dtod(&anchor.slice(..1), &mut self.ids.slice_mut(..1))?;
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.body(start, target, compute, None, None)
    }

    pub(crate) fn proposals(&mut self) -> Result<Vec<u32>> {
        self.stream
            .memcpy_dtoh(&self.confidences, &mut self.host_confidence)?;
        ensure!(
            self.host_confidence.iter().all(|v| v.is_finite()),
            "staged confidence is nonfinite"
        );
        let values = self.stream.clone_dtoh(&self.tokens.slice(1..))?;
        Ok(values)
    }

    pub(crate) fn confidences(&self) -> Vec<f32> {
        self.host_confidence.clone()
    }

    pub(crate) fn copy_proposals(
        &self,
        ids: &mut crate::backend::gpu::CudaViewMut<'_, u32>,
    ) -> Result<()> {
        self.stream.memcpy_dtod(&self.tokens.slice(1..), ids)?;
        Ok(())
    }

    pub(crate) fn reset_setup(&mut self) -> Result<()> {
        if let Some(slots) = &mut self.slots
            && slots.request_open
        {
            slots.end_request()?;
        }
        for ring in &mut self.rings {
            self.stream.memset_zeros(&mut ring.codes)?;
            self.stream.memset_zeros(&mut ring.scales)?;
        }
        self.stream.memcpy_htod(
            &vec![self.config.mask_token_id; self.config.block_size],
            &mut self.ids,
        )?;
        self.stream.memset_zeros(&mut self.position)?;
        Ok(())
    }

    pub(crate) fn grow(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows >= self.rows && rows <= self.capacity,
            "staged workspace growth bound"
        );
        if rows == self.rows {
            return Ok(());
        }
        let (h, width, capture) = (
            self.config.hidden_size,
            self.mixer.geometry.head_dim,
            self.config.capture_width(),
        );
        self.mixer.grow(&self.stream, rows)?;
        self.ffn.grow(&self.stream, rows)?;
        self.main = self.stream.alloc_zeros(rows * h)?;
        self.main_codes = self.stream.alloc_zeros(rows * capture)?;
        self.main_scales = self.stream.alloc_zeros(rows * capture / 32)?;
        self.kv = self.stream.alloc_zeros(rows * width)?;
        self.positions = self.stream.alloc_zeros(rows * 3)?;
        self.rows = rows;
        Ok(())
    }

    pub(crate) fn checkpoint_kv(&mut self, start: usize) -> Result<()> {
        let g = &self.mixer.geometry;
        ensure!(start < self.capacity, "staged checkpoint position");
        for (ring, saved) in self.rings.iter().zip(&mut self.checkpoint) {
            for row in 0..4.min(self.capacity - start) {
                let source = (start + row) % g.window;
                self.stream.memcpy_dtod(
                    &ring
                        .codes
                        .slice(source * g.head_dim..(source + 1) * g.head_dim),
                    &mut saved
                        .codes
                        .slice_mut(row * g.head_dim..(row + 1) * g.head_dim),
                )?;
                let d = g.head_dim / 32;
                self.stream.memcpy_dtod(
                    &ring.scales.slice(source * d..(source + 1) * d),
                    &mut saved.scales.slice_mut(row * d..(row + 1) * d),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn restore_kv(&mut self, start: usize, accepted: usize) -> Result<()> {
        let g = &self.mixer.geometry;
        ensure!(
            accepted <= 4 && start < self.capacity,
            "staged restore prefix"
        );
        for (ring, saved) in self.rings.iter_mut().zip(&self.checkpoint) {
            for row in accepted..4.min(self.capacity - start) {
                let target = (start + row) % g.window;
                self.stream.memcpy_dtod(
                    &saved.codes.slice(row * g.head_dim..(row + 1) * g.head_dim),
                    &mut ring
                        .codes
                        .slice_mut(target * g.head_dim..(target + 1) * g.head_dim),
                )?;
                let d = g.head_dim / 32;
                self.stream.memcpy_dtod(
                    &saved.scales.slice(row * d..(row + 1) * d),
                    &mut ring.scales.slice_mut(target * d..(target + 1) * d),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn state_bits(&self, start: usize, rows: usize, output: &mut Vec<u8>) -> Result<()> {
        ensure!(
            rows > 0
                && start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "staged state range"
        );
        let g = &self.mixer.geometry;
        for ring in &self.rings {
            for row in start..start + rows {
                let slot = row % g.window;
                output.extend(
                    self.stream.clone_dtoh(
                        &ring.codes.slice(slot * g.head_dim..(slot + 1) * g.head_dim),
                    )?,
                );
                let d = g.head_dim / 32;
                output.extend(
                    self.stream
                        .clone_dtoh(&ring.scales.slice(slot * d..(slot + 1) * d))?,
                );
            }
        }
        Ok(())
    }
}

impl DeepseekMixer {
    fn seed(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        rope: &Ops,
        ring: &LatentRing,
        step: MixerStep<'_>,
        layer: usize,
    ) -> Result<()> {
        let MixerStep {
            rows, x, qkv, seq, ..
        } = step;
        let g = &self.geometry;
        ensure!(
            rows > 0
                && rows <= self.rows
                && x.len() >= rows * g.hidden
                && qkv.len() >= rows * g.head_dim,
            "canonical KV seed extent"
        );
        ensure!(
            ring.codes.len() == g.window * g.head_dim
                && ring.scales.len() == g.window * g.head_dim / 32,
            "canonical KV ring extent"
        );
        let w = self
            .weights
            .get(layer)
            .and_then(Option::as_ref)
            .context("canonical KV weights missing")?;
        self.ops.act_quant(
            s,
            view(
                x,
                s,
                0,
                [rows as u32, g.hidden as u32],
                [g.hidden as u64, 1],
            )?,
            &mut TensorMut(view(
                &self.act_hidden_codes,
                s,
                0,
                [rows as u32, g.hidden as u32],
                [g.hidden as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.act_hidden_scales,
                s,
                0,
                [rows as u32, (g.hidden / 32) as u32],
                [(g.hidden / 32) as u64, 1],
            )?),
        )?;
        self.project::<true>(
            s,
            blas,
            &w.wkv,
            (
                view(
                    &self.act_hidden_codes,
                    s,
                    0,
                    [rows as u32, g.hidden as u32],
                    [g.hidden as u64, 1],
                )?,
                view(
                    &self.act_hidden_scales,
                    s,
                    0,
                    [rows as u32, (g.hidden / 32) as u32],
                    [(g.hidden / 32) as u64, 1],
                )?,
            ),
            TensorMut(view(
                qkv,
                s,
                0,
                [rows as u32, g.head_dim as u32],
                [g.head_dim as u64, 1],
            )?),
        )?;
        let kv = view(
            qkv,
            s,
            0,
            [rows as u32, 1, g.head_dim as u32],
            [g.head_dim as u64, g.head_dim as u64, 1],
        )?;
        self.common.norm.rms_with(
            s,
            RowDomain {
                meta: seq,
                row_base: 0,
                row_capacity: rows as u32,
            },
            kv,
            weight_view::<bf16, 1>(&w.kv_norm, s, [g.head_dim as u32], "canonical KV norm")?,
            &mut TensorMut(kv),
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps: self.eps,
            },
        )?;
        let mut fwd = TensorMut(view(
            &self.rotary_fwd,
            s,
            0,
            [rows as u32, g.rotary as u32],
            [g.rotary as u64, 1],
        )?);
        let mut inv = TensorMut(view(
            &self.rotary_inv,
            s,
            0,
            [rows as u32, g.rotary as u32],
            [g.rotary as u64, 1],
        )?);
        rope.rope_pair_rows(
            s,
            view(&self.inv_freq[layer], s, 0, [(g.rotary / 2) as u32], [1])?,
            Position::Sequence { meta: seq, row: 0 },
            rows,
            &mut fwd,
            &mut inv,
        )?;
        rope.rope_pair_tail(s, &mut TensorMut(kv), fwd.0)?;
        for row in 0..rows {
            self.ops.ring_write(
                s,
                view(
                    qkv,
                    s,
                    row * g.head_dim,
                    [1, g.head_dim as u32],
                    [g.head_dim as u64, 1],
                )?,
                Position::Sequence {
                    meta: seq,
                    row: row as u32,
                },
                &mut TensorMut(view(
                    &ring.codes,
                    s,
                    0,
                    [g.window as u32, g.head_dim as u32],
                    [g.head_dim as u64, 1],
                )?),
                &mut TensorMut(view(
                    &ring.scales,
                    s,
                    0,
                    [g.window as u32, (g.head_dim / 32) as u32],
                    [(g.head_dim / 32) as u64, 1],
                )?),
            )?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::backend::cuda::moe::testing::capture;
    use crate::mtp::{batched::BatchedBackend, transaction::Backend};
    use crate::trace::Trace;

    fn canonical(engine: &Engine) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        super::super::tests::composition_state(engine, engine.position, 1, &mut bytes)?;
        let s = &engine.device.stream;
        let state = engine
            .state
            .compressed
            .as_ref()
            .context("compressed state missing")?;
        for value in s.clone_dtoh(&state.generation)? {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for kv in &state.kv {
            for values in [&kv.codes, &kv.scales, &kv.index_codes, &kv.index_scales] {
                bytes.extend(s.clone_dtoh(values)?);
            }
        }
        for index in &state.index {
            for value in s.clone_dtoh(&index.positions)? {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        for value in s.clone_dtoh(&state.candidates)? {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let hc = engine.hc.as_ref().context("mHC missing")?;
        super::super::super::super::copy::bits(s, &hc.stream, &mut bytes)?;
        for values in [&hc.carry, &hc.attn_pre].into_iter().flatten().chain([
            &hc.mix_pre,
            &hc.mix_post,
            &hc.mix_comb,
        ]) {
            for value in s.clone_dtoh(values)? {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        engine
            .spec
            .as_ref()
            .context("staged attachment missing")?
            .draft
            .state_bits(0, 128, &mut bytes)?;
        Ok(bytes)
    }

    fn byte_defects(label: &str, actual: &[u8], expected: &[u8]) -> Result<usize> {
        ensure!(actual.len() == expected.len(), "{label} comparison extent");
        let differences = actual
            .iter()
            .zip(expected)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect::<Vec<_>>();
        println!(
            "DSPARK_STATE_COMPARE boundary={label} bytes={} byte_defects={} first={:?}",
            actual.len(),
            differences.len(),
            differences.first()
        );
        Ok(differences.len())
    }

    fn sequential_row(engine: &mut Engine, input: u32) -> Result<Vec<f32>> {
        let start = engine.position;
        engine.staged_verify(&[input], false)?;
        engine.inject_capture(1, start, None)?;
        engine.committed_prefix_ids.push(input);
        let s = &engine.device.stream;
        s.memcpy_htod(&[engine.position as i32], &mut engine.work.position)?;
        s.memcpy_htod(&[(engine.position + 1) as i32], &mut engine.work.length)?;
        engine
            .state
            .set_sequence(s, engine.position as u32, 1, engine.workspace_rows as u32)?;
        engine.decode_logits()
    }

    fn shared_weights(
        path: &Path,
        device: &Device,
        spec: &ModelSpec,
    ) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let mut manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        for (name, entry) in &mut manifest.entries {
            entry.load = name == "embed.weight" || name == "head.weight";
        }
        Ok(super::super::super::super::weights::checkpoint_mapped(
            &mapped,
            device,
            Some(&manifest),
            false,
            |l| {
                Ok((
                    l.load(&[("embed.weight".into(), vec![129280, 5120])])?,
                    l.load(&[("head.weight".into(), vec![129280, 5120])])?,
                ))
            },
        )?
        .0)
    }

    fn rejected_payload(engine: &Engine) -> Result<[Vec<u8>; 3]> {
        let s = &engine.device.stream;
        let mut payload: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());
        for kv in &engine
            .state
            .compressed
            .as_ref()
            .context("compressed state missing")?
            .kv
        {
            if kv.ratio == 2 {
                for values in [&kv.codes, &kv.scales, &kv.index_codes, &kv.index_scales] {
                    payload[0].extend(s.clone_dtoh(values)?);
                }
                for values in [&kv.tail_values, &kv.tail_scores] {
                    for value in s.clone_dtoh(values)? {
                        payload[0].extend(value.to_bits().to_le_bytes());
                    }
                }
            }
        }
        engine
            .deepseek_engram
            .as_ref()
            .context("Engram missing")?
            .service
            .write_history(&mut payload[1])?;
        let hc = engine.hc.as_ref().context("mHC missing")?;
        super::super::super::super::copy::bits(s, &hc.stream, &mut payload[2])?;
        Ok(payload)
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_MODEL, D1_CENSUS_DOMAIN2, FLYINGFISH_RUNTIME_DIR"]
    fn staged_target_prefix_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let path = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let tokens: Vec<u32> =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_CENSUS_DOMAIN2")?)?)?;
        ensure!(tokens.len() == 600, "frozen target gate token extent");
        let options = crate::spec::Options {
            device: 0,
            capacity: 640,
            chunk: None,
            count: 64,
            ignore_eos: true,
            backend: Settings {
                tree_builder: crate::backend::TreeBuilder::Waves,
                spec_budget: SpecBudget::Chain,
                spec_graph: true,
                draft_model: None,
                prefix_cache: false,
                kv_format: crate::kv::KvFormat::Bf16,
                device_budget_gib: None,
                runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
            },
        };
        let mut engine = Engine::load_mtp(&path, &options, 3, true)?;
        let mut defects = 0usize;
        let mut control = 0usize;
        let s = engine.device.stream.clone();
        for position in [127, 537] {
            engine.staged_prefill(&tokens[..position])?;
            engine.set_token(tokens[position])?;
            let inputs = &tokens[position..position + 4];
            for rows in 2..=4 {
                <Engine as Backend>::save(&mut engine, rows)?;
                let mut states = Vec::new();
                let mut logits = Vec::new();
                let mut taps = Vec::new();
                for &input in &inputs[..rows] {
                    logits.extend(sequential_row(&mut engine, input)?);
                    taps.extend(
                        s.clone_dtoh(
                            &engine
                                .hidden_capture
                                .as_ref()
                                .context("taps missing")?
                                .values
                                .slice(..15360),
                        )?
                        .into_iter()
                        .flat_map(|value| value.to_bits().to_le_bytes()),
                    );
                    states.push(canonical(&engine)?);
                }
                <Engine as Backend>::restore(&mut engine, false)?;
                engine.staged_verify(&inputs[..rows], false)?;
                let eager = super::super::tests::composition_native_state(&engine)?;
                let spec = engine.spec.as_ref().context("staged attachment missing")?;
                let actual = s.clone_dtoh(
                    &spec
                        .staged
                        .as_ref()
                        .context("target state missing")?
                        .logits
                        .slice(..rows * 129280),
                )?;
                let actual: Vec<_> = actual
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                let expected: Vec<_> = logits
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                defects += byte_defects(
                    &format!("p{position}.rows{rows}.logits"),
                    &actual,
                    &expected,
                )?;
                let actual_taps: Vec<_> = s
                    .clone_dtoh(
                        &engine
                            .hidden_capture
                            .as_ref()
                            .context("taps missing")?
                            .values
                            .slice(..rows * 15360),
                    )?
                    .into_iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                defects += byte_defects(
                    &format!("p{position}.rows{rows}.input_taps"),
                    &actual_taps,
                    &taps,
                )?;
                <Engine as Backend>::restore(&mut engine, false)?;
                for kept in 1..=rows {
                    engine.staged_verify(&inputs[..rows], true)?;
                    defects += byte_defects(
                        &format!("p{position}.rows{rows}.eager_graph"),
                        &super::super::tests::composition_native_state(&engine)?,
                        &eager,
                    )?;
                    let wrong = if position == 127 && rows == 4 && kept == 2 {
                        Some(rejected_payload(&engine)?)
                    } else {
                        None
                    };
                    engine.staged_select(kept)?;
                    <Engine as BatchedBackend>::repair_mtp(&mut engine, &inputs[..kept])?;
                    let actual = canonical(&engine)?;
                    defects += byte_defects(
                        &format!("p{position}.rows{rows}.kept{kept}"),
                        &actual,
                        &states[kept - 1],
                    )?;
                    if let Some(wrong) = wrong {
                        let actual = rejected_payload(&engine)?;
                        for ((wrong, actual), component) in
                            wrong
                                .iter()
                                .zip(&actual)
                                .zip(["ratio2", "Engram_history", "mHC"])
                        {
                            let differences =
                                byte_defects(&format!("wrong_prefix_{component}"), wrong, actual)?;
                            ensure!(
                                differences > 0,
                                "wrong-prefix {component} control is not discriminating"
                            );
                            control += differences;
                        }
                    }
                    if kept < rows {
                        let next = sequential_row(&mut engine, inputs[kept])?;
                        let next: Vec<_> = next
                            .iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                            .collect();
                        defects += byte_defects(
                            &format!("p{position}.rows{rows}.kept{kept}.continuation_logits"),
                            &next,
                            &expected[kept * 129280 * 4..(kept + 1) * 129280 * 4],
                        )?;
                        defects += byte_defects(
                            &format!("p{position}.rows{rows}.kept{kept}.continuation_state"),
                            &canonical(&engine)?,
                            &states[kept],
                        )?;
                    }
                    <Engine as Backend>::restore(&mut engine, false)?;
                }
            }
        }
        let anchor = engine.token()?;
        let position = engine.position;
        let target = engine.spec.as_mut().context("staged attachment missing")?;
        let config = target.draft.config.clone();
        let mut cache = Vec::new();
        target.draft.state_bits(0, 128, &mut cache)?;
        target.draft.propose_joint(
            engine.moe_slots.as_mut().context("target pool missing")?,
            anchor,
            engine.position,
            (engine.weights.embedding(), &engine.weights.head),
            Compute {
                ops: &engine.ops,
                blas: &mut engine.blas,
            },
            false,
        )?;
        let eager = target.draft.staged_bits()?;
        target.draft.propose_joint(
            engine.moe_slots.as_mut().context("target pool missing")?,
            anchor,
            engine.position,
            (engine.weights.embedding(), &engine.weights.head),
            Compute {
                ops: &engine.ops,
                blas: &mut engine.blas,
            },
            true,
        )?;
        defects += byte_defects(
            "streamed_staged_eager_graph",
            &target.draft.staged_bits()?,
            &eager,
        )?;
        drop(engine);
        s.synchronize()?;
        let spec = ModelSpec::read(&path)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::super::super::closure::Closure::new(
            false,
            "native DSpark residency identity".into(),
        ));
        let mut blas = Blas::new(
            &device,
            &ops,
            &PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?),
            closure.clone(),
        )?;
        let mut resident = StagedDraft::resident(&path, &device, config, 640, 8, &mut blas)?;
        let (embedding, head) = shared_weights(&path, &device, &spec)?;
        ensure!(
            cache.len() == 3 * 128 * 528,
            "staged canonical cache extent"
        );
        for (ring, bytes) in resident
            .rings
            .iter_mut()
            .zip(cache.as_chunks::<{ 128 * 528 }>().0)
        {
            let mut codes = Vec::with_capacity(128 * 512);
            let mut scales = Vec::with_capacity(128 * 16);
            for row in bytes.as_chunks::<528>().0 {
                codes.extend_from_slice(&row[..512]);
                scales.extend_from_slice(&row[512..]);
            }
            device.stream.memcpy_htod(&codes, &mut ring.codes)?;
            device.stream.memcpy_htod(&scales, &mut ring.scales)?;
        }
        device.finish_upload()?;
        closure.freeze_fixed()?;
        resident.propose(
            anchor,
            position,
            (&embedding, &head),
            Compute {
                ops: &ops,
                blas: &mut blas,
            },
            None,
            None,
        )?;
        let resident_defects =
            byte_defects("staged_resident_streamed", &resident.test_bits()?, &eager)?;
        defects += resident_defects;
        println!(
            "DSPARK_TARGET_PREFIX positions=127,537 widths=3 prefixes=18 numerical_byte_defects={defects} wrong_prefix_control_bytes={control} streamed_staged_graph=true resident_streamed_byte_defects={resident_defects}"
        );
        ensure!(control > 0, "wrong-prefix control is not discriminating");
        if defects != 0 {
            println!("DEFECT DSPARK_TARGET_PREFIX byte_defects={defects}");
        }
        ensure!(defects == 0, "staged target prefix numerical DEFECT");
        println!(
            "PASS DSPARK_TARGET_PREFIX positions=127,537 widths=3 prefixes=18 eager_graph_bits=0 state_bits=0 wrong_prefix_detected=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_MODEL, D1_CENSUS_DOMAIN2, FLYINGFISH_RUNTIME_DIR"]
    fn staged_capacity_boundary_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let path = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let tokens: Vec<u32> =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_CENSUS_DOMAIN2")?)?)?;
        let capacity = 132;
        let prompt = tokens
            .get(..capacity - 4)
            .context("boundary prompt missing")?;
        crate::mtp::check_request(prompt.len(), 3, capacity)?;
        for graph in [false, true] {
            let options = crate::spec::Options {
                device: 0,
                capacity,
                chunk: None,
                count: 3,
                ignore_eos: true,
                backend: Settings {
                    tree_builder: crate::backend::TreeBuilder::Waves,
                    spec_budget: SpecBudget::Chain,
                    spec_graph: graph,
                    draft_model: None,
                    prefix_cache: false,
                    kv_format: crate::kv::KvFormat::Bf16,
                    device_budget_gib: None,
                    runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
                },
            };
            let mut engine = Engine::load_mtp(&path, &options, 3, graph)?;
            ensure!(engine.capacity == capacity, "target capacity changed");
            engine.reset()?;
            let mut plain = vec![engine.prefill(prompt, None)?];
            let mut margin = f64::INFINITY;
            for row in 0..3 {
                if row != 0 {
                    plain.push(engine.decode(false, None)?);
                }
                let (id, gap) =
                    crate::verify::top_two("staged.capacity.plain", &engine.decode_logits()?)?;
                ensure!(Some(&id) == plain.last(), "plain prediction differs");
                margin = margin.min(gap);
            }
            let output = engine.generate_mtp_prepared(
                prompt,
                3,
                true,
                crate::mtp::VerifyPolicy::Batched,
                graph,
                &mut |_, _| true,
            )?;
            ensure!(output.ids == plain, "boundary speculative tokens differ");
            ensure!(
                engine.position == prompt.len() + 2,
                "boundary target position"
            );
            ensure!(margin > 0.0, "boundary prediction has no margin");
            println!(
                "PASS DSPARK_CAPACITY_BOUNDARY graph={graph} capacity={capacity} prompt={} count=3 depth=3 ids={plain:?} plain_min_margin={margin:e}",
                prompt.len()
            );
        }
        Ok(())
    }

    fn census_begin() -> Result<()> {
        SUPPLY_CENSUS.with(|records| -> Result<()> {
            ensure!(records.borrow().is_none(), "supply census already active");
            *records.borrow_mut() = Some(Vec::new());
            Ok(())
        })?;
        crate::experts::disk::READ_CENSUS.with(|records| -> Result<()> {
            ensure!(records.borrow().is_none(), "file census already active");
            *records.borrow_mut() = Some(Vec::new());
            Ok(())
        })
    }

    fn census_take() -> Result<serde_json::Value> {
        let supply = SUPPLY_CENSUS
            .with(|records| records.borrow_mut().take())
            .context("supply census inactive")?;
        let reads = crate::experts::disk::READ_CENSUS
            .with(|records| records.borrow_mut().take())
            .context("file census inactive")?;
        Ok(serde_json::json!({
            "supply": supply, "file_reads": reads, "file_scope": "calling_thread_and_joined_native_disk_workers",
        }))
    }

    fn census_margin(logits: &[f32]) -> Result<serde_json::Value> {
        ensure!(logits.len() == 129280, "census full vocabulary extent");
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "census nonfinite logits"
        );
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.select_nth_unstable_by(1, |&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b))
        });
        let (first, second) = (order[0], order[1]);
        Ok(serde_json::json!({
            "first_id": first, "second_id": second, "first": logits[first],
            "second": logits[second], "margin": logits[first] - logits[second],
        }))
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_MODEL, FF_DS41_CASE, FF_DS41_OUTPUT, FLYINGFISH_RUNTIME_DIR"]
    fn staged_acceptance_census_gpu() -> Result<()> {
        use std::io::Write;
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let case: serde_json::Value =
            serde_json::from_slice(&std::fs::read(std::env::var("FF_DS41_CASE")?)?)?;
        let depth = case["depth"].as_u64().context("census depth missing")? as usize;
        let mode = case["mode"].as_str().context("census mode missing")?;
        ensure!((1..=3).contains(&depth), "census depth outside1..3");
        ensure!(matches!(mode, "candidate" | "ordinary"), "census mode");
        ensure!(
            case["prefix"] == 536
                && case["count"] == 64
                && case["capacity"] == 640
                && case["graph"] == true,
            "frozen census shape differs"
        );
        let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(
            case["tokens"]
                .as_str()
                .context("census token path missing")?,
        )?)?;
        ensure!(tokens.len() == 600, "census frozen600ID extent");
        let output = PathBuf::from(std::env::var("FF_DS41_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let mut journal = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("rounds.jsonl"))?;
        let settings = Settings {
            tree_builder: crate::backend::TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: true,
            draft_model: None,
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            device_budget_gib: None,
            runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        };
        let options = crate::spec::Options {
            device: 0,
            capacity: 640,
            chunk: None,
            count: 64,
            ignore_eos: false,
            backend: settings.clone(),
        };
        let path = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let mut engine = if mode == "candidate" {
            Engine::load_mtp(&path, &options, depth, true)?
        } else {
            Engine::load(&path, 0, 640, None, settings)?
        };
        census_begin()?;
        let seed = if mode == "candidate" {
            engine.staged_prefill(&tokens[..536])?
        } else {
            engine.prefill(&tokens[..536], None)?
        };
        let prefill = census_take()?;
        let mut output_ids = vec![seed];
        let mut consumed_ids = Vec::new();
        let mut records = Vec::new();
        let mut defects = 0usize;
        if mode == "candidate" {
            engine.set_token(seed)?;
            while output_ids.len() < 64 && !engine.model_spec.eos.contains(&engine.token()?) {
                let position = engine.position;
                let anchor = engine.token()?;
                let remaining = 64 - output_ids.len();
                census_begin()?;
                let result = engine.staged_round(remaining, None)?;
                let supply = census_take()?;
                let r = &result.round;
                ensure!(!r.tokens.is_empty(), "census round made no progress");
                ensure!(
                    r.consumed == r.accepted + 1,
                    "census consumed prefix extent"
                );
                ensure!(r.tokens.len() == r.consumed, "census publication extent");
                let inputs: Vec<_> = std::iter::once(anchor)
                    .chain(r.proposals.iter().copied())
                    .take(r.consumed)
                    .collect();
                consumed_ids.extend(inputs);
                output_ids.extend_from_slice(&r.tokens);
                let verify_rows = r.proposals.len() + 1;
                let logits = if result.verify_predictions.is_empty() {
                    engine.decode_logits()?
                } else {
                    engine.device.stream.clone_dtoh(
                        &engine
                            .spec
                            .as_ref()
                            .context("census attachment missing")?
                            .staged
                            .as_ref()
                            .context("census target missing")?
                            .logits
                            .slice(..verify_rows * 129280),
                    )?
                };
                ensure!(
                    logits.len() == verify_rows * 129280,
                    "census verify logit extent"
                );
                let margins = logits
                    .as_chunks::<129280>()
                    .0
                    .iter()
                    .map(|row| census_margin(row))
                    .collect::<Result<Vec<_>>>()?;
                let predictions = if result.verify_predictions.is_empty() {
                    vec![engine.token()?]
                } else {
                    result.verify_predictions.clone()
                };
                for (prediction, margin) in predictions.iter().zip(&margins) {
                    defects += usize::from(margin["first_id"] != *prediction);
                }
                let record = serde_json::json!({
                    "position": position, "anchor": anchor, "remaining": remaining,
                    "proposals": r.proposals, "verify_predictions": predictions,
                    "verify_margins": margins, "matched": r.matched, "accepted": r.accepted,
                    "consumed": r.consumed, "emitted": r.tokens,
                    "kept_verify_state": result.kept_verify_state, "traffic": supply,
                });
                writeln!(journal, "{record}")?;
                journal.flush()?;
                println!(
                    "DSPARK_ACCEPTANCE_ROUND depth={depth} position={position} accepted={} consumed={} emitted={}",
                    r.accepted,
                    r.consumed,
                    output_ids.len()
                );
                records.push(record);
            }
        } else {
            let candidate: serde_json::Value = serde_json::from_slice(&std::fs::read(
                case["candidate"]
                    .as_str()
                    .context("matched candidate path missing")?,
            )?)?;
            let expected: Vec<u32> = serde_json::from_value(candidate["output_ids"].clone())?;
            let consumed: Vec<u32> = serde_json::from_value(candidate["consumed_ids"].clone())?;
            ensure!(
                expected.len() == consumed.len() + 1 && expected.len() <= 64,
                "matched ordinary continuation extent"
            );
            ensure!(
                candidate["case"]["depth"] == depth
                    && candidate["case"]["tokens"] == case["tokens"],
                "matched ordinary case differs"
            );
            defects += usize::from(seed != expected[0]);
            for (index, &input) in consumed.iter().enumerate() {
                let position = engine.position;
                engine.set_token(input)?;
                census_begin()?;
                let prediction = engine.decode(true, None)?;
                let supply = census_take()?;
                let margin = census_margin(&engine.decode_logits()?)?;
                defects += usize::from(prediction != expected[index + 1]);
                let record = serde_json::json!({
                    "position": position, "input": input, "prediction": prediction,
                    "expected": expected[index + 1], "margin": margin, "traffic": supply,
                });
                writeln!(journal, "{record}")?;
                journal.flush()?;
                records.push(record);
                output_ids.push(prediction);
                consumed_ids.push(input);
            }
        }
        let stopped_on_eos = engine.model_spec.eos.contains(&engine.token()?);
        let result = serde_json::json!({
            "case": case, "seed": seed, "prefill": prefill, "output_ids": output_ids,
            "consumed_ids": consumed_ids, "rounds": records, "defects": defects,
            "stopped_on_eos": stopped_on_eos, "timing": false,
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join("result.json"))?;
        serde_json::to_writer(&mut file, &result)?;
        file.write_all(b"\n")?;
        println!(
            "DSPARK_ACCEPTANCE_CASE mode={mode} depth={depth} output_ids={} rounds={} defects={defects} eos={stopped_on_eos}",
            output_ids.len(),
            records.len()
        );
        if defects != 0 {
            println!("DEFECT DSPARK_ACCEPTANCE_CASE mode={mode} depth={depth} defects={defects}");
        }
        ensure!(defects == 0, "census numerical DEFECT");
        println!(
            "PASS DSPARK_ACCEPTANCE_CASE mode={mode} depth={depth} counts_saved=true timing=false"
        );
        Ok(())
    }

    fn taps(start: usize, rows: usize) -> Vec<bf16> {
        (start..start + rows)
            .flat_map(|row| {
                (0..15360).map(move |column| {
                    bf16::from_f32(
                        ((row * 17 + column * 13 + (column / 5120) * 19) % 257) as f32 / 128.0
                            - 1.0,
                    )
                })
            })
            .collect()
    }

    fn outputs(draft: &StagedDraft) -> Result<Trace> {
        let (b, h, v) = (
            draft.config.block_size,
            draft.config.hidden_size,
            draft.config.vocab_size,
        );
        let mut trace = Trace::default();
        let mut t = Some(&mut trace);
        bf_trace(
            &draft.stream,
            &mut t,
            "final_norm.0",
            vec![1, b, h],
            &draft.n,
        )?;
        fp_trace(
            &draft.stream,
            &mut t,
            "base_head.0",
            vec![1, b, v],
            &draft.base,
        )?;
        fp_trace(
            &draft.stream,
            &mut t,
            "output.0.1",
            vec![1, b, v],
            &draft.scores,
        )?;
        fp_trace(
            &draft.stream,
            &mut t,
            "output.0.2",
            vec![1, b],
            &draft.confidences,
        )?;
        trace.add(
            "output.0.0".into(),
            vec![1, b + 1],
            draft
                .stream
                .clone_dtoh(&draft.tokens)?
                .into_iter()
                .map(|id| id as f32)
                .collect(),
        )?;
        Ok(trace)
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_MODEL, FF_MARKOV_FIXTURES, FF_DS41_OUTPUT and FLYINGFISH_RUNTIME_DIR"]
    fn staged_native_body_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let path = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let fixtures = PathBuf::from(std::env::var("FF_MARKOV_FIXTURES")?);
        let output = PathBuf::from(std::env::var("FF_DS41_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(
            fixtures.join("reference-tolerances-r2.json"),
        )?)?;
        ensure!(
            frozen["factor"] == 2
                && frozen["tolerances_frozen"] == true
                && frozen["candidate_seen"] == false,
            "native body requires frozen r2"
        );
        let projection: serde_json::Value = serde_json::from_slice(&std::fs::read(
            fixtures.join("main-projection-contract-r2.json"),
        )?)?;
        ensure!(
            projection["factor"] == 2 && projection["other_predicates_changed"] == false,
            "native projection certificate contract"
        );
        let spec = ModelSpec::read(&path)?;
        let config = DraftConfig::read(&path, &spec.config())?;
        ensure!(
            config.staged.is_some() && config.block_size == 5 && config.num_hidden_layers == 3,
            "native three-stage geometry"
        );
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::super::super::closure::Closure::new(
            false,
            "native DSpark fixed plans".into(),
        ));
        let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        let mut draft = StagedDraft::resident(&path, &device, config, 640, 8, &mut blas)?;
        let (embedding, head) = shared_weights(&path, &device, &spec)?;
        device.finish_upload()?;
        let frequency: serde_json::Value = serde_json::from_slice(&std::fs::read(
            fixtures.join("rotary-frequency-reference-r1.json"),
        )?)?;
        ensure!(
            frequency["theta"].as_f64() == Some(f64::from(spec.rope_theta))
                && frequency["rotary_dim"].as_u64() == Some(u64::from(spec.rotary_dim)),
            "rotary frequency fixture geometry"
        );
        let expected: Vec<u32> = frequency["expected_bits"]
            .as_array()
            .context("rotary frequency fixture values")?
            .iter()
            .map(|value| {
                u32::try_from(value.as_u64().context("rotary frequency bits")?)
                    .context("rotary frequency bit extent")
            })
            .collect::<Result<_>>()?;
        ensure!(expected.len() == 32, "rotary frequency fixture width");
        let old = rotary_inv_freq(
            &spec,
            spec.deepseek_v41.as_ref().context("rotary geometry")?,
            0,
        )?;
        let control = old
            .iter()
            .zip(&expected)
            .filter(|(value, bits)| value.to_bits() != **bits)
            .count();
        ensure!(control == 2, "CPU frequency control did not discriminate");
        for inverse in &draft.mixer.inv_freq {
            let actual = device.stream.clone_dtoh(inverse)?;
            ensure!(
                actual.len() == expected.len()
                    && actual.iter().zip(&expected).all(|(a, b)| a.to_bits() == *b),
                "native rotary frequency bit defect"
            );
        }
        println!(
            "PASS DSPARK_BODY_ROTARY_FREQUENCIES stages=3 values=32 bit_defects=0 old_control={control}"
        );
        closure.freeze_fixed()?;
        let s = &device.stream;
        let mut graph_defects = 0usize;
        let mut projection_defects = 0usize;
        let mut hc_defects = 0usize;
        let mut controls = 0usize;
        for p in [1usize, 127, 128, 129, 537] {
            draft.reset_setup()?;
            for start in (0..p).step_by(8) {
                let rows = 8.min(p - start);
                let input = s.clone_htod(&taps(start, rows))?;
                draft.inject(
                    &input,
                    rows,
                    start,
                    None,
                    Compute {
                        ops: &ops,
                        blas: &mut blas,
                    },
                    None,
                )?;
            }
            let mut trace = Trace::default();
            trace.set_device(&device.info.name)?;
            draft.cache_trace(&mut Some(&mut trace), "cache_before")?;
            let input = s.clone_htod(&taps(p, 1))?;
            draft.inject(
                &input,
                1,
                p,
                None,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some(&mut trace),
            )?;
            let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(
                fixtures.join(format!("fixtures-r1/data/p{p}-repeat0/manifest.json")),
            )?)?;
            let anchor_info = &reference["input.anchor.0"];
            let bytes = std::fs::read(
                fixtures
                    .join(format!("fixtures-r1/data/p{p}-repeat0"))
                    .join(
                        anchor_info["file"]
                            .as_str()
                            .context("anchor fixture path")?,
                    ),
            )?;
            let anchor = u32::try_from(i64::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("anchor fixture extent"))?,
            ))?;
            let proposals = draft.propose(
                anchor,
                p + 1,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some(&mut trace),
                None,
            )?;
            ensure!(
                trace.names().len() == 90,
                "native boundary inventory {}",
                trace.names().len()
            );
            let mut router = Trace::default();
            router.set_device(&device.info.name)?;
            let mut ffn = Trace::default();
            ffn.set_device(&device.info.name)?;
            let mut boundaries = Trace::default();
            boundaries.set_device(&device.info.name)?;
            for name in trace.names() {
                let destination = if name.contains(".ffn_projections.") {
                    &mut ffn
                } else if name.ends_with(".route_logits") {
                    &mut router
                } else {
                    &mut boundaries
                };
                destination.add(
                    name.to_owned(),
                    trace.shape(name)?.to_vec(),
                    trace.values(name)?,
                )?;
            }
            ensure!(router.names().len() == 3, "native router inventory");
            ensure!(ffn.names().len() == 12, "native FFN projection inventory");
            ffn.save(&output.join(format!("p{p}-ffn.safetensors")))?;
            router.save(&output.join(format!("p{p}-router.safetensors")))?;
            trace = boundaries;
            ensure!(
                trace.names().len() == 75,
                "native frozen boundary inventory"
            );
            let limits = frozen["positions"]
                .as_array()
                .context("body numerical positions")?
                .iter()
                .find(|entry| entry["position"].as_u64() == Some(p as u64))
                .context("body numerical position missing")?;
            for component in 0..3 {
                let name = format!("stage0.mix.attn.0.{component}");
                let boundary = &reference[&name];
                let bytes = std::fs::read(
                    fixtures
                        .join(format!("fixtures-r1/data/p{p}-repeat0"))
                        .join(boundary["file"].as_str().context("mHC reference payload")?),
                )?;
                ensure!(bytes.len().is_multiple_of(4), "mHC reference F32 extent");
                let expected: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|value| f32::from_le_bytes(*value))
                    .collect();
                let actual = trace.values(&name)?;
                ensure!(
                    !actual.is_empty()
                        && actual.len() == expected.len()
                        && actual
                            .iter()
                            .chain(&expected)
                            .all(|value| value.is_finite()),
                    "mHC finite/shape gate"
                );
                let maximum = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0.0f64, f64::max);
                let rms = (actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>()
                    / actual.len() as f64)
                    .sqrt();
                let unequal = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let band = &limits["boundaries"][&name];
                let max_limit = band["max_abs"].as_f64().context("mHC maximum band")?;
                let rms_limit = band["rmse"].as_f64().context("mHC RMS band")?;
                let bitwise = band["zero_spread_requires_bitwise_equal"]
                    .as_bool()
                    .context("mHC zero-spread predicate")?;
                let passed = maximum <= max_limit && rms <= rms_limit && (!bitwise || unequal == 0);
                hc_defects += usize::from(!passed);
                println!(
                    "DSPARK_BODY_MHC_CERTIFICATE position={p} component={component} max_abs={maximum} max_limit={max_limit} rmse={rms} rmse_limit={rms_limit} unequal_values={unequal} pass={passed}"
                );
            }
            let expected = projection["positions"]
                .as_array()
                .context("projection certificate positions")?
                .iter()
                .find(|entry| entry["position"].as_u64() == Some(p as u64))
                .context("projection certificate position missing")?["expected"]
                .as_array()
                .context("projection certificate values")?;
            let actual = trace.values("main_proj.0")?;
            ensure!(
                actual.len() == 5120 && expected.len() == actual.len(),
                "projection certificate extent"
            );
            let mut unequal = 0;
            for (actual, expected) in actual.iter().zip(expected) {
                let expected = expected.as_f64().context("projection certificate scalar")? as f32;
                unequal += usize::from(actual.to_bits() != expected.to_bits());
            }
            projection_defects += unequal;
            println!(
                "DSPARK_BODY_MAIN_CERTIFICATE position={p} values=5120 unequal_values={unequal}"
            );
            trace.save(&output.join(format!("p{p}-eager.safetensors")))?;
            let mut primitives = Trace::default();
            primitives.set_device(&device.info.name)?;
            let captured = std::mem::take(&mut *draft.mixer.query_stores.borrow_mut());
            ensure!(captured.len() == 27, "staged attention primitive inventory");
            for (name, values) in captured {
                ensure!(
                    values.len().is_multiple_of(5),
                    "staged primitive row extent"
                );
                primitives.add(
                    name,
                    vec![1, 5, values.len() / 5],
                    values.into_iter().map(bf16::to_f32).collect(),
                )?;
            }
            primitives.save(&output.join(format!("p{p}-attention.safetensors")))?;
            let mut before = Vec::new();
            draft.state_bits(0, 128, &mut before)?;
            let anchor_device = s.clone_htod(&[anchor])?;
            let position = s.clone_htod(&[(p + 1) as i32])?;
            let graph = capture(s, || {
                draft.propose_device(
                    &anchor_device,
                    &position,
                    p + 1,
                    (&embedding, &head),
                    Compute {
                        ops: &ops,
                        blas: &mut blas,
                    },
                )
            })?;
            for buffer in [
                &mut draft.x,
                &mut draft.n,
                &mut draft.qkv,
                &mut draft.out,
                &mut draft.parent,
            ] {
                s.memcpy_htod(&vec![bf16::NAN; buffer.len()], buffer)?;
            }
            for buffer in [
                &mut draft.base,
                &mut draft.bias,
                &mut draft.scores,
                &mut draft.confidences,
            ] {
                s.memcpy_htod(&vec![f32::NAN; buffer.len()], buffer)?;
            }
            s.memcpy_htod(&vec![u32::MAX; draft.tokens.len()], &mut draft.tokens)?;
            graph.launch()?;
            let mut replay = outputs(&draft)?;
            replay.set_device(&device.info.name)?;
            for name in replay.names() {
                let expected = trace.values(name)?;
                let actual = replay.values(name)?;
                let defects = expected
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                graph_defects += defects;
                println!(
                    "DSPARK_BODY_GRAPH position={p} boundary={name} words={} bit_defects={defects}",
                    expected.len()
                );
            }
            replay.save(&output.join(format!("p{p}-graph.safetensors")))?;
            let mut after = Vec::new();
            draft.state_bits(0, 128, &mut after)?;
            graph_defects += before.iter().zip(&after).filter(|(a, b)| a != b).count();
            ensure!(
                before.len() == after.len(),
                "canonical state extent changed"
            );
            if p == 128 {
                let wrong = s.clone_htod(&taps(p - 1, 1))?;
                draft.inject(
                    &wrong,
                    1,
                    p,
                    None,
                    Compute {
                        ops: &ops,
                        blas: &mut blas,
                    },
                    None,
                )?;
                let mut perturbed = Vec::new();
                draft.state_bits(0, 128, &mut perturbed)?;
                let per_stage = before.len() / 3;
                for stage in 0..3 {
                    let lo = stage * per_stage;
                    let differing = before[lo..lo + per_stage]
                        .iter()
                        .zip(&perturbed[lo..lo + per_stage])
                        .filter(|(a, b)| a != b)
                        .count();
                    controls += usize::from(differing > 0);
                    println!("DSPARK_BODY_WRONG_SOURCE stage={stage} differing_bytes={differing}");
                }
            }
            println!(
                "DSPARK_BODY_CASE position={p} proposals={proposals:?} graph_defects={graph_defects}"
            );
        }
        println!(
            "DSPARK_BODY_EXPORTED cases=5 graph_cases=5 graph_bit_defects={graph_defects} controls={controls} body_numerics=external_frozen_validator"
        );
        ensure!(
            graph_defects == 0 && controls == 3,
            "native body graph/control defect"
        );
        println!("PASS DSPARK_BODY_GRAPH cases=5 graph_cases=5 wrong_source_stages=3");
        ensure!(
            projection_defects == 0,
            "native main projection numerical DEFECT"
        );
        println!("PASS DSPARK_BODY_MAIN_CERTIFICATES positions=5 unequal_values=0");
        ensure!(hc_defects == 0, "native mHC original-r2 numerical DEFECT");
        println!("PASS DSPARK_BODY_MHC_CERTIFICATES positions=5 components=3 original_r2=true");
        Ok(())
    }
}
