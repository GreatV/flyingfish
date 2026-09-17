//! Layer-major prefill. Decoder caches are populated once for the full prompt;
//! the ordinary token path continues from the resulting states.

use super::*;
use candle_core::IndexOp;

type TokenRoutes = Vec<(Vec<u32>, Vec<f32>)>;

pub(super) struct PrefillLayerOutput {
    pub(super) streams: Tensor,
    routes: Option<TokenRoutes>,
}

impl StreamedGlm {
    pub(super) fn admit_prefill(&self, tokens: usize) -> Result<()> {
        let text = &self.config.text_config;
        let workspace = prefill_workspace_bytes(text, tokens)?;
        let admission = self
            .admission
            .context("GLM request admission model is unavailable")?;
        let kda_state = checked_bytes(
            &[
                text.linear_num_heads,
                text.linear_head_dim,
                text.linear_head_dim,
                text.linear_attention_layers().count(),
            ],
            4,
        )?
        .checked_add(checked_bytes(
            &[
                3,
                text.linear_qkv_dim()?,
                text.linear_conv_kernel_dim,
                text.linear_attention_layers().count(),
            ],
            self.compute_dtype.size_in_bytes(),
        )?)
        .context("GLM request KDA state bytes overflow")?;
        let cache = self.expert_cache_stats();
        let additional_cache = cache
            .max_bytes
            .checked_sub(cache.bytes)
            .context("GLM cache occupancy exceeds its bound")?;
        let required = [
            workspace,
            kda_state,
            admission.maximum_dsa_cache_bytes,
            admission.streamed_transient_bytes,
            admission.pending_lm_head_bytes,
            admission.live_expert_bytes,
            admission.weight_load_staging_bytes,
            additional_cache,
            admission.safety_bytes,
            // Under the fold these draw from the same pool as the charges above.
            if admission.unified_pool {
                admission.host_charge_bytes
            } else {
                0
            },
            // The mask is charged at its own boundary; readmission omits it.
            if admission.unified_pool {
                admission.prefill_host_charge_bytes
            } else {
                0
            },
            // Charged before the pool exists, and nowhere after. The phase
            // model charges one set per axis, so a folded check needs both.
            admission
                .pinned_slot_bytes
                .checked_mul(if admission.unified_pool { 2 } else { 1 })
                .and_then(|n| n.checked_add(admission.pinned_ring_bytes))
                .context("GLM pinned slot charge overflow")?,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes)
                .context("GLM request admission bytes overflow")
        })?;
        let snapshot = ResourceSnapshot::capture(Some(&self.device));
        let available = if self.device.is_cpu() {
            crate::admission::host_available(&snapshot)
        } else if admission.unified_pool {
            snapshot.unified_pool_available_bytes()
        } else {
            snapshot.device_free_memory_bytes
        }
        .context("cannot measure free memory for GLM batched prefill admission")?;
        ensure!(
            u64::try_from(required)? <= available,
            "GLM {tokens}-token batched prefill needs {required} additional free bytes (workspace {workspace}), but only {available} are free"
        );
        Ok(())
    }

    pub(super) fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &mut DecoderState,
        routing_trace: &mut Option<RoutingTraceBuilder>,
        progress: bool,
    ) -> Result<Tensor> {
        ensure!(
            state.tokens == 0,
            "GLM full prefill requires an empty decoder state"
        );
        let text = &self.config.text_config;
        ensure!(
            !tokens.is_empty() && tokens.len() <= text.index_topk,
            "GLM prefill exceeds its exact DSA token bound"
        );
        let embedding = self
            .weights
            .load_rows(EMBEDDING_WEIGHT, tokens, &self.device)?
            .to_dtype(self.compute_dtype)?;
        let mut streams = embedding.unsqueeze(1)?.repeat((1, text.hc_mult, 1))?;
        let mut routes = Vec::new();
        for (layer, cache) in state.layers.iter_mut().enumerate() {
            let started = Instant::now();
            let output = self.forward_layer_prefill(layer, &streams, cache)?;
            streams = output.streams;
            if let (true, Some(selected)) = (routing_trace.is_some(), output.routes) {
                routes.push((layer, selected));
            }
            self.device.synchronize()?;
            if progress {
                eprintln!(
                    "GLM batched prefill layer {}/{} ({} tokens) completed in {:.2}s",
                    layer + 1,
                    text.num_hidden_layers,
                    tokens.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        if let Some(trace) = routing_trace.as_mut() {
            for token in 0..tokens.len() {
                for (layer, selected) in &routes {
                    let (experts, gate_weights) = &selected[token];
                    trace.record(
                        token,
                        RoutingTracePhase::Prefill,
                        *layer,
                        experts,
                        gate_weights,
                    )?;
                }
            }
        }
        state.tokens = tokens.len();
        let final_streams = streams.i(tokens.len() - 1)?;
        let collapsed = final_streams
            .to_dtype(DType::F32)?
            .mean(0)?
            .to_dtype(self.compute_dtype)?;
        let norm = self.load_tensor(FINAL_NORM_WEIGHT)?;
        math::rms_norm(&collapsed, Some(&norm), text.rms_norm_eps)
    }

    pub(super) fn forward_layer_prefill(
        &self,
        layer: usize,
        streams: &Tensor,
        cache: &mut LayerCache,
    ) -> Result<PrefillLayerOutput> {
        let text = &self.config.text_config;
        let prefix = format!("model.language_model.layers.{layer}");
        let (post, comb, collapsed) = self.hyper_map_prefill(&prefix, "attn", streams)?;
        let norm = self.load_tensor(&format!("{prefix}.input_layernorm.weight"))?;
        let normalized = math::rms_norm(&collapsed, Some(&norm), text.rms_norm_eps)?;
        let attended = match cache {
            LayerCache::Kda(cache) => self.kda_prefill(&prefix, &normalized, cache)?,
            LayerCache::Dsa(cache) => self.mla_prefill(&prefix, &normalized, cache)?,
        };
        let streams = math::apply_mhc_residual_batched(streams, &attended, &post, &comb)?;
        let (post, comb, collapsed) = self.hyper_map_prefill(&prefix, "ffn", &streams)?;
        let norm = self.load_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let normalized = math::rms_norm(&collapsed, Some(&norm), text.rms_norm_eps)?;
        let (output, routes) = match text.mlp_layer_types[layer] {
            MlpKind::Dense => (
                self.mlp_prefill(&format!("{prefix}.mlp"), &normalized)?,
                None,
            ),
            MlpKind::Sparse => {
                let (output, selected) =
                    self.moe_prefill(layer, &format!("{prefix}.mlp"), &normalized)?;
                (output, Some(selected))
            }
        };
        Ok(PrefillLayerOutput {
            streams: math::apply_mhc_residual_batched(&streams, &output, &post, &comb)?,
            routes,
        })
    }

    fn hyper_map_prefill(
        &self,
        prefix: &str,
        site: &str,
        streams: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let text = &self.config.text_config;
        math::mhc_map_batched(
            streams,
            &self.load_tensor(&format!("{prefix}.hc_{site}_fn"))?,
            &self.load_tensor(&format!("{prefix}.hc_{site}_base"))?,
            &self.load_tensor(&format!("{prefix}.hc_{site}_scale"))?,
            text.rms_norm_eps,
            text.hc_eps,
            text.hc_sinkhorn_iters,
        )
    }

    fn project_prefill_convolution(
        &self,
        prefix: &str,
        name: &str,
        input: &Tensor,
        state: &mut Tensor,
    ) -> Result<Tensor> {
        let weight = self.load_linear_weight(&format!("{prefix}.self_attn.{name}_proj.weight"))?;
        let projected = linear(input, &weight)?;
        let weight = self
            .load_tensor(&format!("{prefix}.self_attn.{name}_conv1d.weight"))?
            .squeeze(1)?;
        let (channels, kernel) = weight.dims2()?;
        ensure!(
            state.dims() == [channels, kernel],
            "GLM prefill convolution state has the wrong shape"
        );
        let padded = projected
            .t()?
            .contiguous()?
            .pad_with_zeros(1, kernel - 1, 0)?;
        let tokens = input.dim(0)?;
        let mut windows = Vec::with_capacity(kernel);
        for offset in 0..kernel {
            windows.push(padded.narrow(1, offset, tokens)?.unsqueeze(2)?);
        }
        let convolved = Tensor::cat(&windows, 2)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&weight.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .sum(2)?
            .to_dtype(self.compute_dtype)?;
        let output = math::silu_with_reference_rounding(&convolved)?
            .t()?
            .contiguous()?;
        *state =
            projected
                .t()?
                .contiguous()?
                .pad_with_zeros(1, kernel.saturating_sub(tokens), 0)?;
        *state = state
            .narrow(1, state.dim(1)? - kernel, kernel)?
            .contiguous()?;
        Ok(output)
    }

    fn kda_prefill(&self, prefix: &str, input: &Tensor, cache: &mut KdaCache) -> Result<Tensor> {
        let text = &self.config.text_config;
        let tokens = input.dim(0)?;
        let shape = (tokens, text.linear_num_heads, text.linear_head_dim);
        let q = self
            .project_prefill_convolution(prefix, "q", input, &mut cache.query_conv)?
            .reshape(shape)?;
        let k = self
            .project_prefill_convolution(prefix, "k", input, &mut cache.key_conv)?
            .reshape(shape)?;
        let v = self
            .project_prefill_convolution(prefix, "v", input, &mut cache.value_conv)?
            .reshape(shape)?;
        let g = math::kda_forget_gate(
            input,
            &self.load_linear_weight(&format!("{prefix}.self_attn.f_a_proj.weight"))?,
            &self.load_linear_weight(&format!("{prefix}.self_attn.f_b_proj.weight"))?,
            &self.load_tensor(&format!("{prefix}.self_attn.dt_bias"))?,
            &self.load_tensor(&format!("{prefix}.self_attn.A_log"))?,
            text.linear_lower_bound
                .context("validated KDA lower bound is missing")?,
        )?;
        let beta = math::sigmoid_with_reference_rounding(&linear(
            input,
            &self.load_linear_weight(&format!("{prefix}.self_attn.b_proj.weight"))?,
        )?)?;
        let (attended, state) = math::kda_prefill(&q, &k, &v, &g, &beta, &cache.recurrent)?;
        cache.recurrent = state;
        let gate = linear(
            &linear(
                input,
                &self.load_linear_weight(&format!("{prefix}.self_attn.g_a_proj.weight"))?,
            )?,
            &self.load_linear_weight(&format!("{prefix}.self_attn.g_b_proj.weight"))?,
        )?
        .reshape(shape)?;
        let norm = self.load_tensor(&format!("{prefix}.self_attn.o_norm.weight"))?;
        let attended = math::rms_norm_gated(&attended, &norm, &gate, text.rms_norm_eps)?
            .reshape((tokens, text.linear_qkv_dim()?))?;
        linear(
            &attended,
            &self.load_linear_weight(&format!("{prefix}.self_attn.o_proj.weight"))?,
        )
    }

    fn mla_prefill(&self, prefix: &str, input: &Tensor, cache: &mut DsaCache) -> Result<Tensor> {
        let text = &self.config.text_config;
        let tokens = input.dim(0)?;
        ensure!(
            cache.keys.is_none() && cache.values.is_none(),
            "MLA full prefill requires an empty KV cache"
        );
        ensure!(
            text.qk_rope_head_dim == 0 && tokens <= text.index_topk,
            "MLA prefill requires bounded NoPE attention"
        );
        let q_a = linear(
            input,
            &self.load_linear_weight(&format!("{prefix}.self_attn.q_a_proj.weight"))?,
        )?;
        let q_a = math::rms_norm(
            &q_a,
            Some(&self.load_tensor(&format!("{prefix}.self_attn.q_a_layernorm.weight"))?),
            text.rms_norm_eps,
        )?;
        let q = linear(
            &q_a,
            &self.load_linear_weight(&format!("{prefix}.self_attn.q_b_proj.weight"))?,
        )?
        .reshape((tokens, text.num_attention_heads, text.mla_qk_head_dim()?))?
        .transpose(0, 1)?
        .contiguous()?;
        let kv = linear(
            input,
            &self.load_linear_weight(&format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"))?,
        )?;
        let kv = math::rms_norm(
            &kv,
            Some(&self.load_tensor(&format!("{prefix}.self_attn.kv_a_layernorm.weight"))?),
            text.rms_norm_eps,
        )?;
        let expanded = linear(
            &kv,
            &self.load_linear_weight(&format!("{prefix}.self_attn.kv_b_proj.weight"))?,
        )?
        .reshape((
            tokens,
            text.num_attention_heads,
            text.qk_nope_head_dim + text.v_head_dim,
        ))?
        .transpose(0, 1)?;
        let key = expanded.narrow(2, 0, text.qk_nope_head_dim)?.contiguous()?;
        let value = expanded
            .narrow(2, text.qk_nope_head_dim, text.v_head_dim)?
            .contiguous()?;
        let scores = q
            .matmul(&key.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (text.mla_qk_head_dim()? as f64).sqrt(), 0.0)?;
        let mask = Tensor::from_vec(
            (0..tokens)
                .flat_map(|row| (0..tokens).map(move |column| u8::from(column <= row)))
                .collect::<Vec<_>>(),
            (tokens, tokens),
            &self.device,
        )?
        .broadcast_as(scores.shape())?;
        let negative = Tensor::full(f32::NEG_INFINITY, scores.shape(), &self.device)?
            .to_dtype(self.compute_dtype)?;
        let scores = mask.where_cond(&scores, &negative)?;
        let probabilities =
            ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(self.compute_dtype)?;
        let output = probabilities
            .matmul(&value)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((tokens, text.num_attention_heads * text.v_head_dim))?;
        cache.keys = Some(key);
        cache.values = Some(value);
        linear(
            &output,
            &self.load_linear_weight(&format!("{prefix}.self_attn.o_proj.weight"))?,
        )
    }

    fn mlp_prefill(&self, prefix: &str, input: &Tensor) -> Result<Tensor> {
        let gate = linear(
            input,
            &self.load_linear_weight(&format!("{prefix}.gate_proj.weight"))?,
        )?;
        let up = linear(
            input,
            &self.load_linear_weight(&format!("{prefix}.up_proj.weight"))?,
        )?;
        let activated = math::clamped_swiglu(&gate, &up, self.config.text_config.swiglu_limit)?;
        linear(
            &activated,
            &self.load_linear_weight(&format!("{prefix}.down_proj.weight"))?,
        )
    }

    fn moe_prefill(
        &self,
        layer: usize,
        prefix: &str,
        input: &Tensor,
    ) -> Result<(Tensor, TokenRoutes)> {
        let text = &self.config.text_config;
        let routed = math::topk_router(
            input,
            &self.load_tensor(&format!("{prefix}.gate.weight"))?,
            &self.load_tensor(&format!("{prefix}.gate.e_score_correction_bias"))?,
            text.num_experts_per_tok,
            text.routed_scaling_factor,
            text.norm_topk_prob,
        )?;
        let selected = routed.indices.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let mixtures = routed.weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let mut groups = BTreeMap::<u32, Vec<(u32, f32)>>::new();
        for slot in 0..text.num_experts_per_tok {
            for token in 0..selected.len() {
                groups
                    .entry(selected[token][slot])
                    .or_default()
                    .push((u32::try_from(token)?, mixtures[token][slot]));
            }
        }
        let cache = self.expert_cache.cache_for_layer(layer)?;
        let mut output = Tensor::zeros(input.shape(), self.compute_dtype, &self.device)?;
        for (expert, rows) in groups {
            let expert_prefix = format!("{prefix}.experts.{expert}");
            let load = |suffix: &str| -> Result<Tensor> {
                let name = format!("{expert_prefix}.{suffix}.weight");
                self.complete_cached_weight(cache, name.clone(), cache.get(&name))
            };
            let gate = load("gate_proj")?;
            let up = load("up_proj")?;
            let down = load("down_proj")?;
            let indices = Tensor::from_vec(
                rows.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
                rows.len(),
                &self.device,
            )?;
            let states = input.index_select(&indices, 0)?;
            let gate_up = Tensor::cat(&[&gate, &up], 0)?;
            let projected = self.traced_linear(&states, &gate_up)?;
            let width = gate.dim(0)?;
            let activated = math::clamped_swiglu(
                &projected.narrow(1, 0, width)?,
                &projected.narrow(1, width, width)?,
                text.swiglu_limit,
            )?;
            let contribution = self
                .traced_linear(&activated, &down)?
                .to_dtype(DType::F32)?
                .broadcast_mul(&Tensor::from_vec(
                    rows.iter().map(|(_, weight)| *weight).collect::<Vec<_>>(),
                    (rows.len(), 1),
                    &self.device,
                )?)?
                .to_dtype(self.compute_dtype)?;
            output = output.index_add(&indices, &contribution, 0)?;
        }
        Ok((
            output.add(&self.mlp_prefill(&format!("{prefix}.shared_experts"), input)?)?,
            selected.into_iter().zip(mixtures).collect(),
        ))
    }
}

fn checked_bytes(dimensions: &[usize], element_bytes: usize) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(element_bytes, |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .context("GLM prefill tensor bytes overflow")
        })
}

pub(crate) fn prefill_workspace_bytes(
    text: &super::super::config::GlmTextConfig,
    tokens: usize,
) -> Result<usize> {
    ensure!(
        tokens > 0 && tokens <= text.index_topk,
        "GLM prefill token count exceeds its admission profile"
    );
    let chunks = tokens.div_ceil(64);
    let keys = checked_bytes(
        &[text.linear_num_heads, chunks, 64, text.linear_head_dim],
        4,
    )?;
    let pair = checked_bytes(
        &[text.linear_num_heads, chunks, 64, 64, text.linear_head_dim],
        4,
    )?;
    let scores = checked_bytes(&[text.linear_num_heads, chunks, 64, 64], 4)?;
    let state = checked_bytes(
        &[
            text.linear_num_heads,
            text.linear_head_dim,
            text.linear_head_dim,
        ],
        4,
    )?;
    let kda = [(pair, 4usize), (keys, 36), (scores, 8), (state, 4)]
        .into_iter()
        .try_fold(0usize, |sum, (bytes, count)| {
            sum.checked_add(
                bytes
                    .checked_mul(count)
                    .context("GLM KDA workspace bytes overflow")?,
            )
            .context("GLM KDA workspace sum overflow")
        })?;
    let mla = checked_bytes(&[text.num_attention_heads, tokens, tokens], 16)?;
    let convolution = checked_bytes(
        &[
            3,
            tokens,
            text.linear_qkv_dim()?,
            text.linear_conv_kernel_dim,
        ],
        4,
    )?;
    let widest = text
        .linear_qkv_dim()?
        .max(text.hidden_size)
        .max(text.intermediate_size)
        .max(text.moe_intermediate_size)
        .max(
            text.num_attention_heads
                .checked_mul(text.mla_qk_head_dim()?.max(text.v_head_dim))
                .context("GLM prefill projection width overflow")?,
        );
    let projections = checked_bytes(&[16, tokens, widest], 4)?;
    let streams = checked_bytes(&[8, tokens, text.hc_mult, text.hidden_size], 4)?;
    kda.max(mla)
        .max(convolution)
        .checked_add(projections)
        .and_then(|bytes| bytes.checked_add(streams))
        .context("GLM prefill workspace bound overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        HIDDEN, patterned_bf16, tiny_checkpoint, tiny_checkpoint_with_kda_width,
    };
    use candle_core::safetensors;

    fn check_mhc_residency(device: Device) -> Result<()> {
        let checkpoint = if device.is_cuda() {
            tiny_checkpoint_with_kda_width(128)
        } else {
            tiny_checkpoint()
        };
        let path = checkpoint.path().join("model.safetensors");
        let mut tensors = safetensors::load(&path, &Device::Cpu)?;
        for (name, tensor) in &mut tensors {
            if is_mhc_constant(name) && name.ends_with("_fn") {
                *tensor = patterned_bf16(tensor.shape().clone(), 7);
            }
        }
        safetensors::save(&tensors, path)?;
        let open = |resident| {
            StreamedGlm::open(
                checkpoint.path(),
                StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), device.clone())
                    .with_resident_static(resident),
            )
        };
        let resident = open(true)?;
        let streamed = open(false)?;
        let admitted = resident.admission_breakdown.as_ref().unwrap().static_bytes;
        assert_eq!(resident.resident_static_bytes(), admitted);
        for (name, tensor) in &resident.static_weights {
            if is_mhc_constant(name) {
                assert_eq!(tensor.dtype(), DType::F32, "{name}");
            }
        }
        let streams = patterned_bf16((2, HIDDEN), 3)
            .to_device(&device)?
            .to_dtype(resident.compute_dtype)?;
        let batch = Tensor::stack(&[&streams, &streams], 0)?;
        let values = |(a, b, c): (Tensor, Tensor, Tensor)| -> Result<Vec<Vec<f32>>> {
            [a, b, c]
                .into_iter()
                .map(|t| Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?))
                .collect()
        };
        for layer in 0..resident.config.text_config.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer}");
            for site in ["attn", "ffn"] {
                for _ in 0..2 {
                    assert_eq!(
                        values(resident.hyper_map(&prefix, site, &streams)?)?,
                        values(streamed.hyper_map(&prefix, site, &streams)?)?,
                    );
                    assert_eq!(
                        values(resident.hyper_map_prefill(&prefix, site, &batch)?)?,
                        values(streamed.hyper_map_prefill(&prefix, site, &batch)?)?,
                    );
                }
            }
        }
        assert_eq!(resident.resident_static_bytes(), admitted);
        assert_eq!(streamed.resident_static_bytes(), 0);
        Ok(())
    }

    #[test]
    fn mhc_residency_matches_admission_and_streamed_output() -> Result<()> {
        check_mhc_residency(Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_residency_matches_admission_and_streamed_output_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        check_mhc_residency(device)
    }

    fn close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.dims(), expected.dims());
        for (a, b) in actual
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .zip(expected.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 + b.abs() * 1e-5,
                "batched/serial CPU difference: {a} != {b}"
            );
        }
    }

    #[test]
    fn batched_prefill_populates_caches_and_continues_decode() {
        let checkpoint = crate::test_support::tiny_checkpoint();
        let model = StreamedGlm::open(
            checkpoint.path(),
            StreamedGlmOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu),
        )
        .unwrap();
        let mut batched = DecoderState::new(&model.config, DType::F32, &Device::Cpu).unwrap();
        let mut serial = DecoderState::new(&model.config, DType::F32, &Device::Cpu).unwrap();
        let output = model
            .forward_prefill(&[4, 5, 6], &mut batched, &mut None, false)
            .unwrap();
        let mut old = None;
        for token in [4, 5, 6] {
            old = Some(
                model
                    .forward_token(token, &mut serial, RoutingTracePhase::Prefill, &mut None)
                    .unwrap(),
            );
        }
        close(&output, &old.unwrap());
        assert_eq!(batched.tokens, serial.tokens);
        for (a, b) in batched.layers.iter().zip(&serial.layers) {
            match (a, b) {
                (LayerCache::Kda(a), LayerCache::Kda(b)) => {
                    close(&a.query_conv, &b.query_conv);
                    close(&a.key_conv, &b.key_conv);
                    close(&a.value_conv, &b.value_conv);
                    close(&a.recurrent, &b.recurrent);
                }
                (LayerCache::Dsa(a), LayerCache::Dsa(b)) => {
                    close(a.keys.as_ref().unwrap(), b.keys.as_ref().unwrap());
                    close(a.values.as_ref().unwrap(), b.values.as_ref().unwrap());
                }
                _ => panic!("cache kind differs"),
            }
        }
        close(
            &model
                .forward_token(7, &mut batched, RoutingTracePhase::Decode, &mut None)
                .unwrap(),
            &model
                .forward_token(7, &mut serial, RoutingTracePhase::Decode, &mut None)
                .unwrap(),
        );
        assert!(
            model
                .forward_prefill(&[4], &mut batched, &mut None, false)
                .unwrap_err()
                .to_string()
                .contains("empty decoder")
        );
    }

    #[test]
    fn prefill_workspace_accounts_for_chunk_and_quadratic_growth() {
        let mut text = super::super::super::config::GlmTextConfig::tiny();
        text.index_topk = 2048;
        let small = prefill_workspace_bytes(&text, 33).unwrap();
        let next_chunk = prefill_workspace_bytes(&text, 65).unwrap();
        let full = prefill_workspace_bytes(&text, 2048).unwrap();
        assert!(small > 0 && next_chunk > small && full > next_chunk);
        assert!(prefill_workspace_bytes(&text, 0).is_err());
        assert!(prefill_workspace_bytes(&text, 2049).is_err());
        assert!(checked_bytes(&[usize::MAX, 2], 4).is_err());
    }
}
