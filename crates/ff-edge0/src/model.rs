//! P0 text model: hybrid GDN/full-attention MoE decoder, f32 CPU path.

use crate::config::{Edge0Config, LayerKind};
use crate::int4::GroupQuant;
use crate::weights::Edge0Weights;
#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::ensure;
#[cfg(feature = "cuda")]
use cudarc::driver::sys;
use ff_core::math::{l2norm, rms_norm, silu, softplus};
use std::path::Path;

const P: &str = "language_model.model";

/// (kind, normed input, attention-block output, moe output) — for the
/// reference-diff harnesses.
pub type BlockParts = (String, Vec<f32>, Vec<f32>, Vec<f32>);

/// EDGE0_MAX_CTX override for the resident KV cache, default 4096 (the attn
/// kernel's shared-memory cap is 8192, enforced at upload).
pub fn configured_max_ctx() -> usize {
    std::env::var("EDGE0_MAX_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(4096)
}

struct GdnState {
    conv: Vec<f32>,
    recurrent: Vec<f32>,
}

struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

/// Converted once at load — the hot path used to re-convert these from bf16
/// every token (~1.3M converts + ~150 Vec allocs/token across 40 layers).
struct LayerNorms {
    input: Vec<f32>,
    post: Vec<f32>,
}

pub(crate) struct GdnWeights {
    pub(crate) conv1d: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) norm: Vec<f32>,
}

struct AttnNorms {
    q: Vec<f32>,
    k: Vec<f32>,
}

#[derive(Default)]
pub struct Timing {
    pub forward_ms: f64,
    pub moe_ms: f64,
    pub expert_load_ms: f64,
    pub logits_ms: f64,
    pub gdn_proj_ms: f64,
    pub gdn_recur_ms: f64,
    pub attn_proj_ms: f64,
    pub moe_compute_ms: f64,
    pub gpu_syncs: u64,
    pub layer_inner_ms: f64,
    pub outer_ms: f64,
}

pub struct Edge0Text {
    config: Edge0Config,
    weights: Edge0Weights,
    gdn: Vec<GdnState>,
    kv: Vec<KvCache>,
    layer_norms: Vec<LayerNorms>,
    /// Indexed by GDN-layer order, like `gdn`.
    gdn_weights: Vec<GdnWeights>,
    /// Indexed by attention-layer order, like `kv`.
    attn_norms: Vec<AttnNorms>,
    position: usize,
    embed: GroupQuant,
    final_norm: Vec<f32>,
    projection_cache: std::collections::HashMap<String, std::sync::Arc<GroupQuant>>,
    #[cfg(feature = "cuda")]
    pub gpu: Option<crate::gpu::GpuRuntime>,
    #[cfg(feature = "cuda")]
    pub gpu_experts: Option<crate::gpu::GpuExperts>,
    #[cfg(feature = "cuda")]
    pub gpu_multi: Option<crate::gpu::Edge0Multi>,
    #[cfg(feature = "cuda")]
    decode_graph: Option<crate::gpu::DecodeGraph>,
    pub timing: Timing,
}

impl Edge0Text {
    pub fn load(model_dir: &Path, config: Edge0Config) -> Result<Self> {
        let weights = Edge0Weights::open(model_dir)?;
        let text = &config.text_config;
        let mut gdn = Vec::new();
        let mut kv = Vec::new();
        let mut layer_norms = Vec::new();
        let mut gdn_weights = Vec::new();
        let mut attn_norms = Vec::new();
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{P}.layers.{layer}");
            layer_norms.push(LayerNorms {
                input: weights.f32_named(&format!("{prefix}.input_layernorm.weight"))?,
                post: weights.f32_named(&format!("{prefix}.post_attention_layernorm.weight"))?,
            });
            if text.layer_kind(layer) == LayerKind::LinearAttention {
                let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
                    + text.linear_num_value_heads * text.linear_value_head_dim;
                gdn.push(GdnState {
                    conv: vec![0.0; conv_dim * (text.linear_conv_kernel_dim - 1)],
                    recurrent: vec![
                        0.0;
                        text.linear_num_value_heads
                            * text.linear_key_head_dim
                            * text.linear_value_head_dim
                    ],
                });
                gdn_weights.push(GdnWeights {
                    conv1d: weights.f32_named(&format!("{prefix}.linear_attn.conv1d.weight"))?,
                    dt_bias: weights.f32_named(&format!("{prefix}.linear_attn.dt_bias"))?,
                    a_log: weights.f32_named(&format!("{prefix}.linear_attn.A_log"))?,
                    norm: weights.f32_named(&format!("{prefix}.linear_attn.norm.weight"))?,
                });
            } else {
                kv.push(KvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                    len: 0,
                });
                attn_norms.push(AttnNorms {
                    q: weights.f32_named(&format!("{prefix}.self_attn.q_norm.weight"))?,
                    k: weights.f32_named(&format!("{prefix}.self_attn.k_norm.weight"))?,
                });
            }
        }
        let embed = weights.quant_projection(&format!("{P}.embed_tokens"))?;
        let final_norm = weights.f32_named(&format!("{P}.norm.weight"))?;
        Ok(Self {
            config,
            weights,
            gdn,
            kv,
            layer_norms,
            gdn_weights,
            attn_norms,
            position: 0,
            embed,
            final_norm,
            projection_cache: std::collections::HashMap::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            gpu_experts: None,
            #[cfg(feature = "cuda")]
            gpu_multi: None,
            #[cfg(feature = "cuda")]
            decode_graph: None,
            timing: Timing::default(),
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    #[cfg(feature = "cuda")]
    pub fn is_resident(&self) -> bool {
        self.gpu.as_ref().is_some_and(|rt| rt.res.is_some())
    }

    /// The closed decode loop requires the resident expert set, not just the
    /// static projections.
    #[cfg(feature = "cuda")]
    pub fn has_resident_experts(&self) -> bool {
        self.gpu_experts.is_some()
    }

    /// The resident KV-cache capacity (EDGE0_MAX_CTX, default 4096).
    #[cfg(feature = "cuda")]
    pub fn gpu_max_ctx(&self) -> Option<usize> {
        self.gpu
            .as_ref()
            .and_then(|rt| rt.res.as_ref())
            .map(|res| res.max_ctx)
    }

    /// Device-resident decode step: hidden never leaves the GPU except as
    /// the returned final-norm output. Syncs per token: router top-k and
    /// MoE combine per layer (host, until device top-k lands) + this read.
    #[cfg(feature = "cuda")]
    fn forward_resident(&mut self, token: u32) -> Result<Vec<f32>> {
        // The runtime is taken out so the harness branch can call &mut self
        // (host MoE) without borrowing against it.
        let rt_owned = self.gpu.take().expect("resident runtime");
        let result = self.forward_resident_inner(&rt_owned, token);
        self.gpu = Some(rt_owned);
        result
    }

    #[cfg(feature = "cuda")]
    fn forward_resident_inner(
        &mut self,
        rt: &crate::gpu::GpuRuntime,
        token: u32,
    ) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        let text = self.config.text_config.clone();
        if let Some(res) = &rt.res {
            anyhow::ensure!(
                self.position < res.max_ctx,
                "position {} reached max_ctx {} (KV cache capacity)",
                self.position,
                res.max_ctx
            );
        }
        let mut gdn_index = 0usize;
        let mut kv_index = 0usize;
        rt.embed_into_hidden(token)?;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{P}.layers.{layer}");
            let inner_started = std::time::Instant::now();
            rt.rmsnorm_x1(layer, 0)?;
            let is_gdn = text.layer_kind(layer) == LayerKind::LinearAttention;
            if is_gdn {
                let qkv = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_qkv"))
                    .context("qkv resident")?;
                let z = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_z"))
                    .context("z resident")?;
                let b = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_b"))
                    .context("b resident")?;
                let a = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_a"))
                    .context("a resident")?;
                let out_proj = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.out_proj"))
                    .context("out_proj resident")?;
                rt.gdn_layer_dx(gdn_index, qkv, z, b, a, out_proj, rt.hidden_x1())?;
                rt.add_norm_x1(layer, 1, out_proj.y_ref())?;
            } else {
                let q = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.q_proj"))
                    .context("q resident")?;
                let k = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.k_proj"))
                    .context("k resident")?;
                let v = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.v_proj"))
                    .context("v resident")?;
                let o = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.o_proj"))
                    .context("o resident")?;
                let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
                rt.attn_layer(
                    kv_index,
                    q,
                    k,
                    v,
                    o,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    rotary_dim,
                    text.rope.rope_theta,
                )?;
                rt.add_norm_x1(layer, 1, o.y_ref())?;
            }
            gdn_index += is_gdn as usize;
            kv_index += !is_gdn as usize;
            self.timing.layer_inner_ms += inner_started.elapsed().as_secs_f64() * 1000.0;
            let moe_started = std::time::Instant::now();
            let Some(experts) = self.gpu_experts.as_ref() else {
                // Harness path (no resident experts): host MoE round trip.
                let x1_host = rt.read_hidden_x1()?;
                let moe_out = self.moe_forward(layer, &prefix, &x1_host)?;
                rt.add_moe_residual(&moe_out)?;
                self.timing.moe_ms += moe_started.elapsed().as_secs_f64() * 1000.0;
                if std::env::var_os("EDGE0_DEBUG_RES").is_some() {
                    let h = rt.debug_hidden().unwrap();
                    let norm = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                    println!("gpu layer {layer}: |h| = {norm:.4}");
                }
                continue;
            };
            let router = rt
                .proj
                .get(&format!("{prefix}.mlp.gate"))
                .context("router resident")?;
            let shared_gate = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.gate_proj"))
                .context("sg resident")?;
            let shared_up = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.up_proj"))
                .context("su resident")?;
            let shared_scalar = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert_gate"))
                .context("ss resident")?;
            let shared_down = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.down_proj"))
                .context("sd resident")?;
            rt.moe_closed(
                layer,
                router,
                experts,
                (shared_gate, shared_up, shared_scalar, shared_down),
                rt.hidden_x1(),
                text.effective_top_k(),
            )?;
            if std::env::var_os("EDGE0_LAYER_TIMES").is_some() {
                rt.ctx.stream.synchronize().ok();
                eprintln!(
                    "DBG vec layer {layer} t {:.2}",
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            self.timing.moe_ms += moe_started.elapsed().as_secs_f64() * 1000.0;
            if std::env::var_os("EDGE0_DEBUG_RES").is_some() {
                let h = rt.debug_hidden().unwrap();
                let norm = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                println!("gpu layer {layer}: |h| = {norm:.4} first8 {:?}", &h[..8]);
            }
        }
        self.timing.forward_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.timing.gpu_syncs = rt.ctx.sync_count.load(std::sync::atomic::Ordering::Relaxed);
        self.position += 1;
        rt.bump_position()?;
        rt.final_norm()?;
        rt.read_hidden_x1()
    }

    /// Closed decode step: layers + final norm + lm_head + device argmax.
    /// One dtoh per call (the token id).
    /// Enqueue n decode steps with no intermediate syncs (GPU-time probe).
    #[cfg(feature = "cuda")]
    pub fn run_burst(&mut self, n: usize) -> Result<()> {
        let rt = self.gpu.take().expect("resident runtime");
        let r = (0..n).try_for_each(|_| self.enqueue_decode_step(&rt));
        self.gpu = Some(rt);
        r
    }

    /// Harness: quantized projection + lora pair by name (read-only).
    pub fn weights_quant(&mut self, name: &str) -> Result<std::sync::Arc<crate::int4::GroupQuant>> {
        self.quant_proj(name)
    }

    pub fn weights_lora(&self, name: &str) -> Option<(&[f32], &[f32], usize)> {
        self.weights.lora_for(name)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_token_pub(&mut self, prev: u32) -> Result<u32> {
        self.forward_token(prev)
    }

    #[cfg(feature = "cuda")]
    pub fn set_next_token_harness(&mut self, token: u32) -> Result<()> {
        self.gpu
            .as_ref()
            .expect("resident runtime")
            .set_next_token(token)
    }

    #[cfg(feature = "cuda")]
    pub fn read_next_token_harness(&mut self) -> Result<u32> {
        self.gpu
            .as_ref()
            .expect("resident runtime")
            .read_next_token()
    }

    #[cfg(feature = "cuda")]
    pub fn decode_graph_take(&mut self) -> Option<crate::gpu::DecodeGraph> {
        self.decode_graph.take()
    }

    #[cfg(feature = "cuda")]
    pub fn restore_graph(&mut self, g: Option<crate::gpu::DecodeGraph>) {
        self.decode_graph = g;
    }

    /// First decode token from the post-prefill state (no forward).
    #[cfg(feature = "cuda")]
    pub fn first_token(&mut self) -> Result<u32> {
        self.gpu.as_ref().expect("resident runtime").argmax_x1()
    }

    /// Closed decode step; public for the generate loop.
    #[cfg(feature = "cuda")]
    pub fn forward_token(&mut self, prev: u32) -> Result<u32> {
        let rt_owned = self.gpu.take().expect("resident runtime");
        let result = self.forward_resident_token(&rt_owned, prev);
        self.gpu = Some(rt_owned);
        result
    }

    #[cfg(feature = "cuda")]
    #[cfg(feature = "cuda")]
    fn forward_resident_token(&mut self, rt: &crate::gpu::GpuRuntime, prev: u32) -> Result<u32> {
        let started = std::time::Instant::now();
        if let Some(res) = &rt.res {
            anyhow::ensure!(
                self.position < res.max_ctx,
                "position {} reached max_ctx {} (KV cache capacity)",
                self.position,
                res.max_ctx
            );
        }
        // Graph capture is opt-in (EDGE0_GRAPH): replay measured 26 ms/token
        // vs 2.4 eager — at ~2 us kernels the per-node overhead dominates a
        // ~1200-node graph on this driver. Kept for fusion experiments.
        if std::env::var_os("EDGE0_GRAPH").is_none() {
            self.enqueue_decode_step(rt)?;
            let token = rt.read_next_token()?;
            self.timing.forward_ms += started.elapsed().as_secs_f64() * 1000.0;
            self.timing.gpu_syncs = rt.ctx.sync_count.load(std::sync::atomic::Ordering::Relaxed);
            self.position += 1;
            return Ok(token);
        }
        if let Some(graph) = &self.decode_graph {
            graph.0.launch().context("decode graph replay")?;
        } else {
            rt.ctx
                .stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
                )
                .context("capture begin")?;
            // prev is advisory: the embed kernel reads next_token, which the
            // previous token's argmax (or prefill) already set on device.
            let _ = prev;
            let enq = self.enqueue_decode_step(rt);
            let graph = match enq {
                Ok(()) => rt.ctx.stream.end_capture(sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).context("capture end")?,
                Err(e) => {
                    // Abort capture so the stream is usable again.
                    let _ = rt.ctx.stream.end_capture(sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                    return Err(e);
                }
            };
            let graph = match graph {
                Some(g) => g,
                None => anyhow::bail!("capture produced no graph"),
            };
            graph.launch().context("decode graph first launch")?;
            self.decode_graph = Some(crate::gpu::DecodeGraph(graph));
        }
        let token = rt.read_next_token()?;
        self.timing.forward_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.timing.gpu_syncs = rt.ctx.sync_count.load(std::sync::atomic::Ordering::Relaxed);
        self.position += 1;
        Ok(token)
    }

    /// The pure-launch decode step captured into the graph: embed ->
    /// layers (GDN/attention + closed MoE) -> final norm -> lm_head ->
    /// argmax -> position bump. Every varying value is device-resident.
    #[cfg(feature = "cuda")]
    fn enqueue_decode_step(&mut self, rt: &crate::gpu::GpuRuntime) -> Result<()> {
        let layer_timer = std::time::Instant::now();
        let text = self.config.text_config.clone();
        let mut gdn_index = 0usize;
        let mut kv_index = 0usize;
        rt.embed_from_device_token()?;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{P}.layers.{layer}");
            rt.rmsnorm_x1(layer, 0)?;
            let is_gdn = text.layer_kind(layer) == LayerKind::LinearAttention;
            if is_gdn {
                let qkv = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_qkv"))
                    .context("qkv resident")?;
                let z = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_z"))
                    .context("z resident")?;
                let b = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_b"))
                    .context("b resident")?;
                let a = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.in_proj_a"))
                    .context("a resident")?;
                let out_proj = rt
                    .proj
                    .get(&format!("{prefix}.linear_attn.out_proj"))
                    .context("out_proj resident")?;
                rt.gdn_layer_dx(gdn_index, qkv, z, b, a, out_proj, rt.hidden_x1())?;
                rt.add_norm_x1(layer, 1, out_proj.y_ref())?;
            } else {
                let q = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.q_proj"))
                    .context("q resident")?;
                let k = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.k_proj"))
                    .context("k resident")?;
                let v = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.v_proj"))
                    .context("v resident")?;
                let o = rt
                    .proj
                    .get(&format!("{prefix}.self_attn.o_proj"))
                    .context("o resident")?;
                let rotary_dim = (text.head_dim as f64 * text.rope.partial_rotary_factor) as usize;
                rt.attn_layer(
                    kv_index,
                    q,
                    k,
                    v,
                    o,
                    text.num_attention_heads,
                    text.num_key_value_heads,
                    text.head_dim,
                    rotary_dim,
                    text.rope.rope_theta,
                )?;
                rt.add_norm_x1(layer, 1, o.y_ref())?;
            }
            gdn_index += is_gdn as usize;
            kv_index += !is_gdn as usize;
            let experts = self.gpu_experts.as_ref().context("gpu experts")?;
            let router = rt
                .proj
                .get(&format!("{prefix}.mlp.gate"))
                .context("router resident")?;
            let shared_gate = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.gate_proj"))
                .context("sg resident")?;
            let shared_up = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.up_proj"))
                .context("su resident")?;
            let shared_scalar = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert_gate"))
                .context("ss resident")?;
            let shared_down = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.down_proj"))
                .context("sd resident")?;
            rt.moe_closed(
                layer,
                router,
                experts,
                (shared_gate, shared_up, shared_scalar, shared_down),
                rt.hidden_x1(),
                text.effective_top_k(),
            )?;
            if std::env::var_os("EDGE0_LAYER_TIMES").is_some() {
                rt.ctx.stream.synchronize().ok();
                eprintln!(
                    "DBG layer {layer} t {:.2}",
                    layer_timer.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        rt.final_norm()?;
        rt.finalize_token()?;
        rt.bump_position()?;
        Ok(())
    }

    pub fn embed_row(&self, token: u32) -> Vec<f32> {
        let groups = self.embed.in_dim / crate::int4::GROUP_SIZE;
        (0..self.embed.in_dim)
            .map(|column| {
                let group = column / crate::int4::GROUP_SIZE;
                let scale = self.embed.scales[token as usize * groups + group];
                let bias = self.embed.biases[token as usize * groups + group];
                scale * self.embed_element(token as usize, column) as f32 + bias
            })
            .collect()
    }

    fn embed_element(&self, row: usize, column: usize) -> u32 {
        let per_word = 8;
        let word = self.embed.packed[row * (self.embed.in_dim / per_word) + column / per_word];
        (word >> (4 * (column % per_word))) & 0xF
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if self.gpu.as_ref().is_some_and(|rt| rt.res.is_some()) {
            return self.forward_resident(token);
        }
        let started = std::time::Instant::now();
        let outer_started = std::time::Instant::now();
        let mut hidden = self.embed_row(token);
        let embed_done = outer_started.elapsed();
        let text = self.config.text_config.clone();
        let mut gdn_index = 0;
        let mut kv_index = 0;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{P}.layers.{layer}");
            let inner_started = std::time::Instant::now();
            let normed = rms_norm(
                &hidden,
                &self.layer_norms[layer].input,
                text.rms_norm_eps as f32,
            );
            let attn_out = if text.layer_kind(layer) == LayerKind::LinearAttention {
                self.gdn_forward(gdn_index, &prefix, &normed)?
            } else {
                self.full_attention_forward(kv_index, &prefix, &normed, &text)
            };
            gdn_index += (text.layer_kind(layer) == LayerKind::LinearAttention) as usize;
            kv_index += (text.layer_kind(layer) == LayerKind::FullAttention) as usize;
            for (h, a) in hidden.iter_mut().zip(&attn_out) {
                *h += a;
            }
            let normed = rms_norm(
                &hidden,
                &self.layer_norms[layer].post,
                text.rms_norm_eps as f32,
            );
            self.timing.layer_inner_ms += inner_started.elapsed().as_secs_f64() * 1000.0;
            let moe_started = std::time::Instant::now();
            let moe_out = self.moe_forward(layer, &prefix, &normed)?;
            self.timing.moe_ms += moe_started.elapsed().as_secs_f64() * 1000.0;
            for (h, m) in hidden.iter_mut().zip(&moe_out) {
                *h += m;
            }
            if std::env::var_os("EDGE0_DEBUG_RES").is_some() {
                let norm = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
                println!("cpu layer {layer}: |h| = {norm:.4}");
            }
            if std::env::var_os("EDGE0_DEBUG").is_some() {
                let norm = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
                println!("layer {layer}: |h| = {norm:.3}");
            }
        }
        // Layer-outer: post-loop norm + zero-init are outer; embed already
        // measured. Sum into outer_ms.
        let outer_elapsed = outer_started.elapsed().as_secs_f64() * 1000.0;
        self.timing.outer_ms += embed_done.as_secs_f64() * 1000.0;
        let _ = outer_elapsed;
        self.timing.forward_ms += started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "cuda")]
        if let Some(rt) = &self.gpu {
            self.timing.gpu_syncs = rt.ctx.sync_count.load(std::sync::atomic::Ordering::Relaxed);
        }
        self.position += 1;
        // Final norm is plain RMSNorm (checkpoint norm weights unshifted).
        Ok(rms_norm(
            &hidden,
            &self.final_norm,
            text.rms_norm_eps as f32,
        ))
    }

    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        let out = if self.gpu.as_ref().is_some_and(|rt| rt.res.is_some()) {
            // GPU-resident: lm_head over the final-normed x1 (the closed
            // decode path's input); the host `hidden` arg is stale here.
            self.gpu.as_ref().unwrap().lm_logits_x1()?
        } else {
            self.proj_matvec("language_model.lm_head", hidden)
        };
        #[cfg(not(feature = "cuda"))]
        let out = self.proj_matvec("language_model.lm_head", hidden);
        self.timing.logits_ms += started.elapsed().as_secs_f64() * 1000.0;
        Ok(out)
    }

    /// One layer's parts for reference-diff harnesses: (kind, normed input,
    /// attention-block output, moe output).
    pub fn block_parts(&mut self, layer: usize, x: &[f32]) -> Result<BlockParts> {
        let text = self.config.text_config.clone();
        let prefix = format!("{P}.layers.{layer}");
        let normed = rms_norm(x, &self.layer_norms[layer].input, text.rms_norm_eps as f32);
        let kind = if text.layer_kind(layer) == LayerKind::LinearAttention {
            "gdn"
        } else {
            "attn"
        };
        let gdn_index = (0..layer)
            .filter(|i| text.layer_kind(*i) == LayerKind::LinearAttention)
            .count();
        let kv_index = (0..layer)
            .filter(|i| text.layer_kind(*i) == LayerKind::FullAttention)
            .count();
        let attn_out = if kind == "gdn" {
            self.gdn_forward(gdn_index, &prefix, &normed)?
        } else {
            self.full_attention_forward(kv_index, &prefix, &normed, &text)
        };
        let residual: Vec<f32> = x.iter().zip(&attn_out).map(|(&a, &b)| a + b).collect();
        let moe_in = rms_norm(
            &residual,
            &self.layer_norms[layer].post,
            text.rms_norm_eps as f32,
        );
        let moe_out = self.moe_forward(layer, &prefix, &moe_in)?;
        Ok((kind.to_string(), normed, attn_out, moe_out))
    }

    /// Projection matvec with automatic GPU/CPU dispatch. GPU path
    /// applies LoRA host-side after the kernel (rank 16, negligible).
    fn proj_matvec(&mut self, name: &str, x: &[f32]) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        if let Some(runtime) = &self.gpu
            && let Some(y) = runtime.prepared(name, x)
        {
            return y.unwrap_or_else(|e| panic!("gpu matvec {name}: {e:?}"));
        }
        let quant = self.quant_proj(name).expect("projection load");
        let lora = self.weights.lora_for(name);
        quant.matvec(x, lora)
    }

    fn proj_matvec3_fallback(
        &mut self,
        names: &[String],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let v = self.proj_matvec(&names[2], x);
        let k = self.proj_matvec(&names[1], x);
        let q = self.proj_matvec(&names[0], x);
        (q, k, v)
    }

    fn proj_matvec_fallback(
        &mut self,
        names: &[String],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let a = self.proj_matvec(&names[3], x);
        let b = self.proj_matvec(&names[2], x);
        let z = self.proj_matvec(&names[1], x);
        let q = self.proj_matvec(&names[0], x);
        (q, z, b, a)
    }

    #[cfg(feature = "cuda")]
    fn batch_proj(&self, names: &[String], x: &[f32]) -> Option<Vec<Vec<f32>>> {
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.gpu.as_ref()?.batch_matvec(&refs, x)?.ok()
    }

    /// Upload static projections to the GPU (gdn/attn projections, router,
    /// shared experts, embed, lm_head). Experts stay host-side in v1.
    #[cfg(feature = "cuda")]
    pub fn moe_trace_summary(&self) -> Option<(usize, crate::gpu::MoeTrace)> {
        self.gpu.as_ref()?.moe_trace_summary()
    }

    #[cfg(feature = "cuda")]
    pub fn batch_trace_summary(&self) -> Option<crate::gpu::BatchTrace> {
        self.gpu.as_ref().map(|rt| rt.batch_trace_summary())
    }

    #[cfg(feature = "cuda")]
    pub fn enable_gpu(&mut self, ordinal: usize, experts_resident: bool) -> Result<()> {
        let ctx = crate::gpu::GpuContext::new(ordinal)?;
        if experts_resident {
            // Let the mode.rs planner veto full residency (real VRAM probe)
            // before ~17 GiB of expert uploads OOM mid-way.
            let sizes = crate::mode::WeightSizes::from_weights(&self.weights)?;
            let (free, total) = ctx.context.mem_get_info().context("mem_get_info")?;
            // Same probe contract the ff-core snapshot uses, including the
            // fail-closed half: a failed query warns once and refuses
            // planning (a shared pool cannot be ruled out) rather than
            // guessing a topology.
            static UNIFIED_WARNED: std::sync::Once = std::sync::Once::new();
            let (unified, probe_failed) = match ctx
                .context
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            {
                Ok(value) => (Some(value != 0), false),
                Err(error) => {
                    UNIFIED_WARNED.call_once(|| {
                        eprintln!(
                            "warning: CUDA integrated-topology query failed ({error}); \
                             refusing to plan residency without a topology"
                        );
                    });
                    (None, true)
                }
            };
            // Host-only capture: /proc/meminfo and cgroup views, no CUDA
            // probe. MemAvailable ignores a container's memory ceiling, so
            // the host view is clamped to the cgroup available where one
            // applies (the clamp ff-core's unified-pool helper performs).
            let snapshot = ff_core::probe::ResourceSnapshot::capture(None);
            let host_available = snapshot.host_memory_available_bytes.map(|host| {
                match snapshot.cgroup_v2_memory_available_bytes {
                    Some(cgroup) => host.min(cgroup),
                    None => host,
                }
            });
            let hardware = crate::mode::Hardware {
                total_vram_bytes: Some(total as u64),
                free_vram_bytes: Some(free as u64),
                unified_host_device: unified,
                host_memory_available_bytes: host_available,
                unified_probe_failed: probe_failed,
            };
            let plan =
                crate::mode::plan_mode(&hardware, &crate::mode::ModeOverrides::default(), &sizes)?;
            for line in &plan.provenance {
                println!("[mode] {line}");
            }
            anyhow::ensure!(
                plan.mode == crate::mode::PerformanceMode::FullResident,
                "expert residency was requested but the planner selected {:?} — a \
                 discrete device can run with static weights only; on a unified \
                 pool fix the availability probe first",
                plan.mode
            );
        }
        let mut proj = std::collections::HashMap::new();
        let text = &self.config.text_config;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{P}.layers.{layer}");
            let names: Vec<String> = if text.layer_kind(layer) == LayerKind::LinearAttention {
                vec![
                    format!("{prefix}.linear_attn.in_proj_qkv"),
                    format!("{prefix}.linear_attn.in_proj_z"),
                    format!("{prefix}.linear_attn.in_proj_b"),
                    format!("{prefix}.linear_attn.in_proj_a"),
                    format!("{prefix}.linear_attn.out_proj"),
                ]
            } else {
                vec![
                    format!("{prefix}.self_attn.q_proj"),
                    format!("{prefix}.self_attn.k_proj"),
                    format!("{prefix}.self_attn.v_proj"),
                    format!("{prefix}.self_attn.o_proj"),
                ]
            };
            let mut all = names;
            all.push(format!("{prefix}.mlp.gate"));
            for part in ["gate_proj", "up_proj", "down_proj"] {
                all.push(format!("{prefix}.mlp.shared_expert.{part}"));
            }
            all.push(format!("{prefix}.mlp.shared_expert_gate"));
            for name in all {
                let quant = self.weights.quant_projection(&name)?;
                proj.insert(
                    name.clone(),
                    ctx.upload(&quant, self.weights.lora_for(&name))?,
                );
            }
        }
        proj.insert(format!("{P}.embed_tokens"), ctx.upload(&self.embed, None)?);
        let lm = self.weights.quant_projection("language_model.lm_head")?;
        proj.insert("language_model.lm_head".to_string(), ctx.upload(&lm, None)?);
        let ln_tuples: Vec<(Vec<f32>, Vec<f32>)> = self
            .layer_norms
            .iter()
            .map(|n| (n.input.clone(), n.post.clone()))
            .collect();
        let attn_tuples: Vec<(Vec<f32>, Vec<f32>)> = self
            .attn_norms
            .iter()
            .map(|n| (n.q.clone(), n.k.clone()))
            .collect();
        let res = crate::gpu::ResidentState::upload(
            &ctx,
            text.hidden_size,
            &ln_tuples,
            &attn_tuples,
            &self.final_norm,
            text.num_attention_heads * text.head_dim,
            text.num_key_value_heads * text.head_dim,
            self.kv.len(),
        )?;
        let gdn_devices = self
            .gdn_weights
            .iter()
            .map(|w| {
                crate::gpu::GpuGdn::upload(
                    &ctx,
                    &w.conv1d,
                    &w.a_log,
                    &w.dt_bias,
                    &w.norm,
                    2 * text.linear_num_key_heads * text.linear_key_head_dim
                        + text.linear_num_value_heads * text.linear_value_head_dim,
                    text.linear_conv_kernel_dim,
                    text.linear_num_value_heads,
                    text.linear_num_key_heads,
                    text.linear_key_head_dim,
                    text.linear_value_head_dim,
                    text.rms_norm_eps as f32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if experts_resident {
            let text = &self.config.text_config;
            // Whole-tensor residency ONLY (per-expert GpuQuants would double
            // the 16.9 GiB; only the stacked layout is kept — the batched
            // kernel addresses experts by index).
            let mut stacked = Vec::with_capacity(text.num_hidden_layers);
            let mut stacked_scales = Vec::with_capacity(text.num_hidden_layers);
            let mut stacked_biases = Vec::with_capacity(text.num_hidden_layers);
            // Geometry comes from the checkpoint, not constants.
            let mut rows = [0usize; 3];
            let mut in_dim = [0usize; 3];
            for layer in 0..text.num_hidden_layers {
                // Move into the arrays — CudaSlice's Clone device-duplicates
                // (a full extra expert copy per layer through the mempool).
                let mut row_p = Vec::with_capacity(3);
                let mut row_s = Vec::with_capacity(3);
                let mut row_b = Vec::with_capacity(3);
                for (i, part) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                    let (packed, s, b, r, d) = self.weights.stacked_projection(layer, part)?;
                    if layer == 0 {
                        // edge0_batched_gemv4* stage one word per thread:
                        // words_per_row = in_dim/8 must fit 256 threads.
                        anyhow::ensure!(
                            d / 8 <= 256,
                            "{part}: words_per_row {} exceeds the batched-gemv cap 256",
                            d / 8
                        );
                        rows[i] = r;
                        in_dim[i] = d;
                    } else {
                        anyhow::ensure!(
                            rows[i] == r && in_dim[i] == d,
                            "layer {layer} {part}: expert geometry drifted \
                             ({r}x{d} vs {}x{})",
                            rows[i],
                            in_dim[i]
                        );
                    }
                    row_p.push(ctx.upload_slice(&packed)?);
                    row_s.push(ctx.upload_f32(&s)?);
                    row_b.push(ctx.upload_f32(&b)?);
                }
                stacked.push(row_p.try_into().unwrap());
                stacked_scales.push(row_s.try_into().unwrap());
                stacked_biases.push(row_b.try_into().unwrap());
            }
            self.gpu_experts = Some(crate::gpu::GpuExperts {
                stacked,
                stacked_scales,
                stacked_biases,
                rows,
                in_dim,
            });
            // The mega kernel hardcodes 4 expert slots — fail at startup.
            if std::env::var_os("EDGE0_MEGA").is_some() {
                anyhow::ensure!(
                    text.effective_top_k() == 4,
                    "moe_mega kernel hardcodes 4 expert slots, config top_k={} \
                     — unset EDGE0_MEGA",
                    text.effective_top_k()
                );
            }
        }
        self.gpu = Some(crate::gpu::GpuRuntime::new(
            ctx,
            proj,
            gdn_devices,
            Some(res),
            self.weights.lora_rank,
        ));
        Ok(())
    }

    /// Multi-device orchestration entry point. Mirrors the qwen35 port
    /// (commits 9b6ead8, acc252d).
    ///
    /// `ordinals.len() == 1` routes to the existing single-context path
    /// bitwise-identical to `enable_gpu(ordinals[0], experts_resident)`.
    /// `ordinals.len() > 1` builds one `GpuRuntime` per ordinal for its
    /// byte-balanced layer range, verifies the residency plan fits every
    /// device, and decodes through `forward_multi`/`forward_token_multi`.
    /// Whole-expert residency stays single-device (bails on multiple
    /// ordinals); streamed experts keep their per-layer host round-trip.
    #[cfg(feature = "cuda")]
    pub fn enable_gpu_multi(&mut self, ordinals: &[usize], experts_resident: bool) -> Result<()> {
        ensure!(!ordinals.is_empty(), "at least one ordinal required");
        for (i, &a) in ordinals.iter().enumerate() {
            for &b in &ordinals[i + 1..] {
                ensure!(a != b, "device ordinal {a} listed twice");
            }
        }
        if ordinals.len() == 1 {
            return self.enable_gpu(ordinals[0], experts_resident);
        }
        let config = self.config.clone();
        let mut free_bytes = Vec::with_capacity(ordinals.len());
        for &ordinal in ordinals {
            let ctx = crate::gpu::GpuContext::new(ordinal)
                .with_context(|| format!("init CUDA context on device {ordinal}"))?;
            let (free, _) = ctx
                .context
                .mem_get_info()
                .with_context(|| format!("mem_get_info on cuda:{ordinal}"))?;
            free_bytes.push(free as u64);
        }
        let plans = crate::gpu::plan_residency(
            &self.weights,
            &config,
            ordinals,
            configured_max_ctx(),
            &free_bytes,
            experts_resident,
        )?;
        for plan in &plans {
            eprintln!(
                "edge0_multi_plan cuda:{} range={}..{} projection={} MiB kv={} MiB expert={} MiB \
                 static={} MiB total={} MiB free={} MiB fits={}",
                plan.ordinal,
                plan.range.start,
                plan.range.end,
                plan.projection_bytes / (1024 * 1024),
                plan.kv_bytes / (1024 * 1024),
                plan.expert_bytes / (1024 * 1024),
                plan.static_bytes / (1024 * 1024),
                plan.total_bytes / (1024 * 1024),
                plan.free_bytes / (1024 * 1024),
                plan.fits(),
            );
        }
        for plan in &plans {
            anyhow::ensure!(
                plan.fits(),
                "cuda:{} cannot hold the layer range ({} MiB > {} MiB free; \
                 projection={} MiB kv={} MiB expert={} MiB static={} MiB)",
                plan.ordinal,
                plan.total_bytes / (1024 * 1024),
                plan.free_bytes / (1024 * 1024),
                plan.projection_bytes / (1024 * 1024),
                plan.kv_bytes / (1024 * 1024),
                plan.expert_bytes / (1024 * 1024),
                plan.static_bytes / (1024 * 1024),
            );
        }
        // Flatten the layer-norm buffers (input/post per layer) and the
        // attention-norm buffers (q/k per FullAttention layer) into
        // the slices `Edge0Multi::new` expects.
        let text = &config.text_config;
        let mut layer_norms = Vec::with_capacity(text.num_hidden_layers * 2);
        for n in &self.layer_norms {
            layer_norms.push(n.input.clone());
            layer_norms.push(n.post.clone());
        }
        let mut attn_norms_flat = Vec::with_capacity(self.attn_norms.len() * 2);
        for n in &self.attn_norms {
            attn_norms_flat.push(n.q.clone());
            attn_norms_flat.push(n.k.clone());
        }
        self.gpu_multi = Some(crate::gpu::Edge0Multi::new(
            ordinals,
            &self.weights,
            &config,
            &layer_norms,
            &attn_norms_flat,
            &self.gdn_weights,
            &self.embed,
            &self.final_norm,
            experts_resident,
        )?);
        Ok(())
    }

    /// First decode token from the multi-device prefill state (no forward).
    #[cfg(feature = "cuda")]
    pub fn first_token_multi(&mut self) -> Result<u32> {
        let multi = self.gpu_multi.as_mut().expect("multi-device runtime");
        multi.peers[0].argmax_x1()
    }

    /// Multi-device closed decode step: layers + final norm + lm_head +
    /// argmax + position bump, with `hidden` host-hopping between peers.
    /// Supports both the harness path (host MoE round-trip) and the
    /// resident path (on-device `moe_closed` per peer, scoped to that
    /// peer's layer range).
    #[cfg(feature = "cuda")]
    pub fn forward_token_multi(&mut self, prev: u32) -> Result<u32> {
        let mut multi = self.gpu_multi.take().expect("multi-device runtime");
        if let Some(res) = &multi.peers[0].res {
            anyhow::ensure!(
                self.position < res.max_ctx,
                "position {} reached max_ctx {} (KV cache capacity)",
                self.position,
                res.max_ctx
            );
        }
        multi.peers[0]
            .ctx
            .counted_sync()
            .context("peer0 embed pre-sync")?;
        multi.peers[0].embed_into_hidden(prev)?;
        self.multi_layer_pass(&mut multi)?;
        multi.peers[0].final_norm()?;
        let token = multi.peers[0].argmax_x1()?;
        for peer in &multi.peers {
            peer.bump_position()?;
        }
        self.position += 1;
        self.gpu_multi = Some(multi);
        Ok(token)
    }

    /// Multi-device prefill step: same layer pass as the decode step, but
    /// returns the final-normed hidden instead of an argmaxed token. The
    /// device state (KV, GDN, position counters) advances exactly as in
    /// decode, so the decode chain continues from a warm state.
    #[cfg(feature = "cuda")]
    pub fn forward_multi(&mut self, token: u32) -> Result<Vec<f32>> {
        let mut multi = self.gpu_multi.take().expect("multi-device runtime");
        if let Some(res) = &multi.peers[0].res {
            anyhow::ensure!(
                self.position < res.max_ctx,
                "position {} reached max_ctx {} (KV cache capacity)",
                self.position,
                res.max_ctx
            );
        }
        multi.peers[0]
            .ctx
            .counted_sync()
            .context("peer0 embed pre-sync")?;
        multi.peers[0].embed_into_hidden(token)?;
        self.multi_layer_pass(&mut multi)?;
        multi.peers[0].final_norm()?;
        let mut hidden = std::mem::take(&mut multi.staging_hidden);
        {
            let res = multi.peers[0].res.as_ref().expect("peer0 resident");
            multi.peers[0]
                .ctx
                .stream
                .memcpy_dtoh(&res.x1, &mut hidden)?;
        }
        multi.staging_hidden = vec![0f32; hidden.len()];
        for peer in &multi.peers {
            peer.bump_position()?;
        }
        self.position += 1;
        self.gpu_multi = Some(multi);
        Ok(hidden)
    }

    /// The layer pass shared by multi-device prefill and decode: every peer
    /// runs its range, then `hidden` hops host-mediated to the next peer
    /// (the last peer's hidden hops back to peer 0).
    #[cfg(feature = "cuda")]
    fn multi_layer_pass(&mut self, multi: &mut crate::gpu::Edge0Multi) -> Result<()> {
        let peer_count = multi.peers.len();
        for peer_index in 0..peer_count {
            let range = multi.ranges[peer_index].clone();
            multi.peers[peer_index]
                .ctx
                .counted_sync()
                .context("peer pre-layer sync")?;
            let mut gdn_index_in_peer = 0usize;
            let mut kv_index_in_peer = 0usize;
            for local in range.clone() {
                let peer_layer = local - range.start;
                let prefix = format!("{P}.layers.{local}");
                multi.peers[peer_index].rmsnorm_x1(peer_layer, 0)?;
                let is_gdn =
                    self.config.text_config.layer_kind(local) == LayerKind::LinearAttention;
                if is_gdn {
                    let qkv = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.linear_attn.in_proj_qkv"))
                        .context("qkv resident")?;
                    let z = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.linear_attn.in_proj_z"))
                        .context("z resident")?;
                    let b = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.linear_attn.in_proj_b"))
                        .context("b resident")?;
                    let a = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.linear_attn.in_proj_a"))
                        .context("a resident")?;
                    let out_proj = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.linear_attn.out_proj"))
                        .context("out_proj resident")?;
                    multi.peers[peer_index].gdn_layer_dx(
                        gdn_index_in_peer,
                        qkv,
                        z,
                        b,
                        a,
                        out_proj,
                        multi.peers[peer_index].hidden_x1(),
                    )?;
                    multi.peers[peer_index].add_norm_x1(peer_layer, 1, out_proj.y_ref())?;
                } else {
                    let q = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.self_attn.q_proj"))
                        .context("q resident")?;
                    let k = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.self_attn.k_proj"))
                        .context("k resident")?;
                    let v = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.self_attn.v_proj"))
                        .context("v resident")?;
                    let o = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.self_attn.o_proj"))
                        .context("o resident")?;
                    let rotary_dim = (self.config.text_config.head_dim as f64
                        * self.config.text_config.rope.partial_rotary_factor)
                        as usize;
                    multi.peers[peer_index].attn_layer(
                        kv_index_in_peer,
                        q,
                        k,
                        v,
                        o,
                        self.config.text_config.num_attention_heads,
                        self.config.text_config.num_key_value_heads,
                        self.config.text_config.head_dim,
                        rotary_dim,
                        self.config.text_config.rope.rope_theta,
                    )?;
                    multi.peers[peer_index].add_norm_x1(peer_layer, 1, o.y_ref())?;
                }
                gdn_index_in_peer += is_gdn as usize;
                kv_index_in_peer += !is_gdn as usize;
                // MoE step: closed on device when experts are resident
                // (mirrors `forward_resident_inner`'s Some(experts) branch),
                // otherwise the harness-style host round-trip — same
                // numerics as the single-device path.
                if multi.experts_resident {
                    let experts = multi
                        .peer_experts
                        .get(peer_index)
                        .and_then(Option::as_ref)
                        .context("peer experts missing despite experts_resident")?;
                    let router = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.mlp.gate"))
                        .context("router resident")?;
                    let shared_gate = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.mlp.shared_expert.gate_proj"))
                        .context("sg resident")?;
                    let shared_up = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.mlp.shared_expert.up_proj"))
                        .context("su resident")?;
                    let shared_scalar = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.mlp.shared_expert_gate"))
                        .context("ss resident")?;
                    let shared_down = multi.peers[peer_index]
                        .proj
                        .get(&format!("{prefix}.mlp.shared_expert.down_proj"))
                        .context("sd resident")?;
                    // The peer's `GpuExperts` is indexed by `peer_layer`,
                    // matching the peer's own layer-range slice.
                    multi.peers[peer_index].moe_closed(
                        peer_layer,
                        router,
                        experts,
                        (shared_gate, shared_up, shared_scalar, shared_down),
                        multi.peers[peer_index].hidden_x1(),
                        self.config.text_config.effective_top_k(),
                    )?;
                } else {
                    let x1_host = multi.peers[peer_index].read_hidden_x1()?;
                    // Drop the multi borrow before calling self so the
                    // borrow checker accepts the simultaneous self-borrow.
                    let moe_out = self.moe_forward(local, &prefix, &x1_host)?;
                    multi.peers[peer_index].add_moe_residual(&moe_out)?;
                }
            }
            // Hop: dtoh from this peer's hidden, htod to next peer's hidden.
            // The peer chain is closed by peer 0 doing final_norm + argmax,
            // so the last peer's hidden hops back to peer 0.
            multi.peers[peer_index]
                .ctx
                .counted_sync()
                .context("peer post-layer sync")?;
            // Snapshot staging_hidden so we can release the multi borrow
            // before issuing the two memcpys (which need mutable access
            // to two peers at once).
            let mut staging = std::mem::take(&mut multi.staging_hidden);
            {
                let res = multi.peers[peer_index]
                    .res
                    .as_ref()
                    .expect("resident state");
                multi.peers[peer_index]
                    .ctx
                    .stream
                    .memcpy_dtoh(&res.hidden, &mut staging)?;
            }
            let next_index = if peer_index + 1 < peer_count {
                peer_index + 1
            } else {
                0
            };
            // Hop write: capture the stream borrow before holding the
            // device-buffer borrow.
            let stream = multi.peers[next_index].ctx.stream.clone();
            {
                let dst = multi.peers[next_index]
                    .res
                    .as_mut()
                    .expect("next peer resident");
                stream.memcpy_htod(&staging, &mut dst.hidden)?;
            }
            multi.staging_hidden = staging;
        }
        Ok(())
    }

    fn quant_proj(&mut self, name: &str) -> Result<std::sync::Arc<GroupQuant>> {
        if let Some(hit) = self.projection_cache.get(name) {
            return Ok(std::sync::Arc::clone(hit));
        }
        let loaded = std::sync::Arc::new(self.weights.quant_projection(name)?);
        self.projection_cache
            .insert(name.to_string(), std::sync::Arc::clone(&loaded));
        Ok(loaded)
    }

    fn gdn_forward(&mut self, state_index: usize, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        // Device-resident path: 4 in_proj GEMVs -> conv -> heads -> out_proj
        // on one upload + one sync; the intermediate projections stay on the
        // device.
        #[cfg(feature = "cuda")]
        if let Some(rt) = &self.gpu
            && !rt.gdn.is_empty()
        {
            let qkv = rt
                .proj
                .get(&format!("{prefix}.linear_attn.in_proj_qkv"))
                .context("gdn qkv resident")?;
            let z = rt
                .proj
                .get(&format!("{prefix}.linear_attn.in_proj_z"))
                .context("gdn z resident")?;
            let b = rt
                .proj
                .get(&format!("{prefix}.linear_attn.in_proj_b"))
                .context("gdn b resident")?;
            let a = rt
                .proj
                .get(&format!("{prefix}.linear_attn.in_proj_a"))
                .context("gdn a resident")?;
            let out_proj = rt
                .proj
                .get(&format!("{prefix}.linear_attn.out_proj"))
                .context("gdn out_proj resident")?;
            return rt.gdn_layer_host(state_index, qkv, z, b, a, out_proj, x);
        }
        let text = self.config.text_config.clone();
        let num_v = text.linear_num_value_heads;
        let num_k = text.linear_num_key_heads;
        let dk = text.linear_key_head_dim;
        let dv = text.linear_value_head_dim;
        let key_dim = num_k * dk;
        let value_dim = num_v * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = text.linear_conv_kernel_dim;

        let proj_started = std::time::Instant::now();
        let (qkv, z_proj, b_proj, a_proj) = {
            let names: Vec<String> = ["qkv", "z", "b", "a"]
                .iter()
                .map(|p| format!("{prefix}.linear_attn.in_proj_{p}"))
                .collect();
            #[cfg(feature = "cuda")]
            if let Some(mut outs) = self.batch_proj(&names, x) {
                let a = outs.pop().unwrap();
                let b = outs.pop().unwrap();
                let z = outs.pop().unwrap();
                let q = outs.pop().unwrap();
                (q, z, b, a)
            } else {
                self.proj_matvec_fallback(&names, x)
            }
            #[cfg(not(feature = "cuda"))]
            self.proj_matvec_fallback(&names, x)
        };
        let proj_ms = proj_started.elapsed().as_secs_f64() * 1000.0;
        self.timing.gdn_proj_ms += proj_ms;
        let recur_started = std::time::Instant::now();

        let statics = &self.gdn_weights[state_index];
        let (conv_weight, dt_bias, a_log, norm_weight) = (
            &statics.conv1d,
            &statics.dt_bias,
            &statics.a_log,
            &statics.norm,
        );

        if std::env::var_os("EDGE0_DEBUG").is_some() {
            let w = self
                .weights
                .quant_projection(&format!("{prefix}.linear_attn.in_proj_qkv"))
                .unwrap();
            let r0 = w.row(0);
            println!("gdn qkv_w row0[:8] {:?}", &r0[..8]);
            println!("gdn qkv scales row0[:4] {:?}", &w.scales[..4]);
            println!("gdn qkv biases row0[:4] {:?}", &w.biases[..4]);
            let packed0: Vec<u32> = w.packed[..4].to_vec();
            println!("gdn qkv packed words[:4] {packed0:?}");
            println!("gdn qkv[:6] {:?}", &qkv[..6]);
            println!("gdn z[:6] {:?}", &z_proj[..6]);
            println!("gdn a[:6] {:?} b[:6] {:?}", &a_proj[..6], &b_proj[..6]);
        }
        let state = &mut self.gdn[state_index];
        let mut conv_in = state.conv.clone();
        conv_in.extend_from_slice(&qkv);
        let mut conv_out = vec![0f32; conv_dim];
        for (channel, out) in conv_out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for j in 0..kernel {
                acc += conv_weight[channel * kernel + j] * conv_in[j * conv_dim + channel];
            }
            *out = silu(acc);
        }
        let keep = conv_dim * (kernel - 1);
        state.conv.copy_from_slice(&conv_in[conv_in.len() - keep..]);

        let q_all = &conv_out[..key_dim];
        let k_all = &conv_out[key_dim..2 * key_dim];
        let v_all = &conv_out[2 * key_dim..];

        let mut out = vec![0f32; value_dim];
        let scale = 1.0 / (dk as f32).sqrt();
        // Heads are independent; each head's three S passes (decay, memory
        // read, outer-product write, output read) fuse into ONE pass over
        // the columns — the row of S is touched once per column instead of
        // three full sweeps. Four heads interleaved expose ILP.
        // Heads are independent: recursively split head-aligned slices of
        // BOTH out and the recurrent state, one thread per half — each
        // thread owns disjoint mutable slices, no locks, no atomics, and
        // the per-column loop keeps the update-then-read order that the
        // pre-update-read bug taught us to preserve.
        {
            let ctx = GdnHeadCtx {
                q_all,
                k_all,
                v_all,
                a_proj: &a_proj,
                b_proj: &b_proj,
                a_log,
                dt_bias,
                z_proj: &z_proj,
                norm_weight,
                num_v,
                num_k,
                dk,
                dv,
                scale,
                eps: text.rms_norm_eps as f32,
            };
            recurse_heads(&mut out[..], &mut state.recurrent[..], 0, num_v, &ctx);
        }
        self.timing.gdn_recur_ms += recur_started.elapsed().as_secs_f64() * 1000.0;
        let out_started = std::time::Instant::now();
        let result = self.proj_matvec(&format!("{prefix}.linear_attn.out_proj"), &out);
        self.timing.gdn_proj_ms += out_started.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }

    fn full_attention_forward(
        &mut self,
        kv_index: usize,
        prefix: &str,
        x: &[f32],
        text: &crate::config::TextConfig,
    ) -> Vec<f32> {
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let rotary_dim = (head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let attn_started = std::time::Instant::now();

        let (q_raw, k_raw, v_raw) = {
            let names: Vec<String> = ["q", "k", "v"]
                .iter()
                .map(|p| format!("{prefix}.self_attn.{p}_proj"))
                .collect();
            #[cfg(feature = "cuda")]
            if let Some(mut outs) = self.batch_proj(&names, x) {
                let v = outs.pop().unwrap();
                let k = outs.pop().unwrap();
                let q = outs.pop().unwrap();
                (q, k, v)
            } else {
                self.proj_matvec3_fallback(&names, x)
            }
            #[cfg(not(feature = "cuda"))]
            self.proj_matvec3_fallback(&names, x)
        };
        let norms = &self.attn_norms[kv_index];
        let (q_norm, k_norm) = (&norms.q, &norms.k);

        let q_total = heads * head_dim;
        // q_proj output is per-head interleaved: [q(head_dim) | gate(head_dim)]
        // for each head — not [all q | all gate].
        let mut query = vec![0f32; q_total];
        let mut gate = vec![0f32; q_total];
        for head in 0..heads {
            let base = head * head_dim * 2;
            query[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&q_raw[base..base + head_dim]);
            gate[head * head_dim..(head + 1) * head_dim]
                .copy_from_slice(&q_raw[base + head_dim..base + 2 * head_dim]);
        }

        let position = self.position;
        let cache = &mut self.kv[kv_index];
        let kv_stride = kv_heads * head_dim;
        // Normalize and rotate each key head at its own position before
        // caching; queries rotate at the current position.
        let mut k_stored = vec![0f32; kv_stride];
        for kv_head in 0..kv_heads {
            let mut k = k_raw[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
            k = rms_norm(&k, k_norm, text.rms_norm_eps as f32);
            apply_rope(&mut k, position, rotary_dim, text.rope.rope_theta);
            k_stored[kv_head * head_dim..(kv_head + 1) * head_dim].copy_from_slice(&k);
        }
        cache.keys.extend_from_slice(&k_stored);
        cache.values.extend_from_slice(&v_raw);
        cache.len += 1;

        let mut attn_out = vec![0f32; q_total];
        let scale = 1.0 / (head_dim as f32).sqrt();
        if std::env::var_os("EDGE0_DEBUG").is_some() {
            println!("attn q_raw[:6] {:?}", &q_raw[..6]);
            println!("attn v[:6] {:?}", &v_raw[..6]);
            println!(
                "attn k_normed[:6] {:?}",
                rms_norm(&k_raw[..head_dim], k_norm, text.rms_norm_eps as f32)
            );
        }
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let mut q = query[head * head_dim..(head + 1) * head_dim].to_vec();
            q = rms_norm(&q, q_norm, text.rms_norm_eps as f32);
            apply_rope(&mut q, position, rotary_dim, text.rope.rope_theta);
            let mut scores = Vec::with_capacity(cache.len);
            for step in 0..cache.len {
                let k = &cache.keys[step * kv_stride + kv_head * head_dim..][..head_dim];
                let mut dot = 0f32;
                for (qv, kv) in q.iter().zip(k) {
                    dot += qv * kv;
                }
                scores.push(dot * scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            #[allow(clippy::needless_range_loop)] // step couples scores + kv stride
            for step in 0..cache.len {
                let weight = (scores[step] - max).exp() / sum;
                let v = &cache.values[step * kv_stride + kv_head * head_dim..][..head_dim];
                for (index, out_slot) in attn_out[head * head_dim..(head + 1) * head_dim]
                    .iter_mut()
                    .enumerate()
                {
                    *out_slot += weight * v[index];
                }
            }
        }

        for (slot, gv) in attn_out.iter_mut().zip(gate) {
            *slot *= 1.0 / (1.0 + (-gv).exp());
        }
        let result = self.proj_matvec(&format!("{prefix}.self_attn.o_proj"), &attn_out);
        self.timing.attn_proj_ms += attn_started.elapsed().as_secs_f64() * 1000.0;
        result
    }

    fn moe_forward(&mut self, layer: usize, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let text = self.config.text_config.clone();
        let top_k = text.effective_top_k();
        #[cfg(feature = "cuda")]
        let gpu_serving = self.gpu_experts.is_some();
        #[cfg(not(feature = "cuda"))]
        let gpu_serving = false;
        let mut out = vec![0f32; text.hidden_size];
        // GPU mode: routing happens device-side in moe_router; the host
        // router matvec + expert unpack below are CPU-path-only.
        let (weights, experts): (Vec<f32>, Vec<_>) = if gpu_serving {
            (Vec::new(), Vec::new())
        } else {
            let router = self.quant_proj(&format!("{prefix}.mlp.gate"))?;
            let logits = router.matvec(x, None);
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let chosen = &order[..top_k];
            let max = logits[chosen[0]];
            let exp_sum: f32 = chosen.iter().map(|&e| (logits[e] - max).exp()).sum();
            let weights: Vec<f32> = chosen
                .iter()
                .map(|&e| (logits[e] - max).exp() / exp_sum)
                .collect();
            let load_started = std::time::Instant::now();
            let experts: Vec<_> = chosen
                .iter()
                .map(|&expert| {
                    Ok((
                        self.weights.quant_expert(layer, expert, "gate_proj")?,
                        self.weights.quant_expert(layer, expert, "up_proj")?,
                        self.weights.quant_expert(layer, expert, "down_proj")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            self.timing.expert_load_ms += load_started.elapsed().as_secs_f64() * 1000.0;
            (weights, experts)
        };
        // Serial expert path: the thread scan showed zero gain from threads
        // at this shape (work per call ~= spawn cost), so parallelism here
        // is pure overhead — 79's falsifiable prediction is ~50 ms serial.
        #[cfg(feature = "cuda")]
        let per_expert: Vec<Vec<f32>> = if let Some(experts) = self.gpu_experts.as_ref() {
            let rt = self.gpu.as_ref().expect("gpu runtime");
            let router = rt
                .proj
                .get(&format!("{prefix}.mlp.gate"))
                .expect("router resident");
            let shared_gate = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.gate_proj"))
                .expect("shared gate");
            let shared_up = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.up_proj"))
                .expect("shared up");
            let shared_scalar = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert_gate"))
                .expect("shared scalar gate");
            let shared_down = rt
                .proj
                .get(&format!("{prefix}.mlp.shared_expert.down_proj"))
                .expect("shared down");
            // Two-phase fused: router sync, then everything else (routed
            // gate/up/down + shared gate/up/down, GPU silu) on ONE more
            // sync — the inner never touches the host.
            let dx = rt.ctx.stream.clone_htod(x).expect("moe x upload");
            let logits = rt.moe_router_dx(layer, router, &dx).expect("router");
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let chosen_vec: Vec<usize> = order[..top_k].to_vec();
            let max = logits[chosen_vec[0]];
            let exp_sum: f32 = chosen_vec.iter().map(|&e| (logits[e] - max).exp()).sum();
            let weights: Vec<f32> = chosen_vec
                .iter()
                .map(|&e| (logits[e] - max).exp() / exp_sum)
                .collect();
            let (scalar, all_outs) = rt
                .moe_fused_dx(
                    experts,
                    layer,
                    &chosen_vec,
                    (shared_gate, shared_up, shared_scalar, shared_down),
                    &dx,
                )
                .expect("moe fused");
            let sigmoid = 1.0 / (1.0 + (-scalar).exp());
            let mut final_out = vec![0f32; text.hidden_size];
            for (slot, down) in all_outs[..chosen_vec.len()].iter().enumerate() {
                for (o, &v) in final_out.iter_mut().zip(down) {
                    *o += weights[slot] * v;
                }
            }
            let shared_result = &all_outs[chosen_vec.len()];
            for (o, v) in final_out.iter_mut().zip(shared_result.iter()) {
                *o += sigmoid * v;
            }
            return Ok(final_out);
        } else {
            experts
                .iter()
                .map(|(gate, up, down)| {
                    let g = gate.matvec(x, None);
                    let u = up.matvec(x, None);
                    let inner: Vec<f32> =
                        g.iter().zip(&u).map(|(&gv, &uv)| silu(gv) * uv).collect();
                    down.matvec(&inner, None)
                })
                .collect()
        };
        #[cfg(not(feature = "cuda"))]
        let per_expert: Vec<Vec<f32>> = experts
            .iter()
            .map(|(gate, up, down)| {
                let g = gate.matvec(x, None);
                let u = up.matvec(x, None);
                let inner: Vec<f32> = g.iter().zip(&u).map(|(&gv, &uv)| silu(gv) * uv).collect();
                down.matvec(&inner, None)
            })
            .collect();
        for (slot, expert_out) in per_expert.into_iter().enumerate() {
            for (o, &value) in out.iter_mut().zip(&expert_out) {
                *o += weights[slot] * value;
            }
        }

        let shared_gate = self
            .quant_proj(&format!("{prefix}.mlp.shared_expert_gate"))?
            .matvec(x, None)[0];
        let sg = self
            .quant_proj(&format!("{prefix}.mlp.shared_expert.gate_proj"))?
            .matvec(
                x,
                self.weights
                    .lora_for(&format!("{prefix}.mlp.shared_expert.gate_proj")),
            );
        let su = self
            .quant_proj(&format!("{prefix}.mlp.shared_expert.up_proj"))?
            .matvec(
                x,
                self.weights
                    .lora_for(&format!("{prefix}.mlp.shared_expert.up_proj")),
            );
        let inner: Vec<f32> = sg.iter().zip(&su).map(|(&a, &b)| silu(a) * b).collect();
        let shared = self
            .quant_proj(&format!("{prefix}.mlp.shared_expert.down_proj"))?
            .matvec(
                &inner,
                self.weights
                    .lora_for(&format!("{prefix}.mlp.shared_expert.down_proj")),
            );
        let scalar = 1.0 / (1.0 + (-shared_gate).exp());
        for (o, &value) in out.iter_mut().zip(&shared) {
            *o += scalar * value;
        }
        Ok(out)
    }
}

fn apply_rope(x: &mut [f32], position: usize, rotary_dim: usize, theta: f64) {
    let half = rotary_dim / 2;
    for i in 0..half {
        let freq = theta.powf(-(2.0 * i as f64) / rotary_dim as f64);
        let angle = position as f64 * freq;
        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
        let (x1, x2) = (x[i], x[i + half]);
        x[i] = x1 * cos - x2 * sin;
        x[i + half] = x2 * cos + x1 * sin;
    }
}

struct GdnHeadCtx<'a> {
    q_all: &'a [f32],
    k_all: &'a [f32],
    v_all: &'a [f32],
    a_proj: &'a [f32],
    b_proj: &'a [f32],
    a_log: &'a [f32],
    dt_bias: &'a [f32],
    z_proj: &'a [f32],
    norm_weight: &'a [f32],
    num_v: usize,
    num_k: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    eps: f32,
}

#[inline]
fn gdn_head(out: &mut [f32], s: &mut [f32], head: usize, ctx: &GdnHeadCtx) {
    let GdnHeadCtx {
        q_all,
        k_all,
        v_all,
        a_proj,
        b_proj,
        a_log,
        dt_bias,
        z_proj,
        norm_weight,
        num_v,
        num_k,
        dk,
        dv,
        scale,
        eps,
    } = ctx;
    let k_head = head / (num_v / num_k);
    let q = l2norm(&q_all[k_head * dk..(k_head + 1) * dk]);
    let k = l2norm(&k_all[k_head * dk..(k_head + 1) * dk]);
    let v = &v_all[head * dv..(head + 1) * dv];
    let g = -a_log[head].exp() * softplus(a_proj[head] + dt_bias[head]);
    let beta = 1.0 / (1.0 + (-b_proj[head]).exp());
    let decay = g.exp();
    for element in s.iter_mut() {
        *element *= decay;
    }
    // Row-major SIMD shape: per row, dv contiguous cells take two fmax
    // passes (kv_mem accumulation, then post-update output accumulation) —
    // the old column loop walked stride-dv, defeating vectorization.
    let mut kv_mem = vec![0f32; *dv];
    for row in 0..*dk {
        let kr = k[row];
        let srow = &s[row * dv..(row + 1) * dv];
        for (acc, &cell) in kv_mem.iter_mut().zip(srow) {
            *acc += kr * cell;
        }
    }
    let mut out_acc = vec![0f32; *dv];
    for column in 0..*dv {
        kv_mem[column] = (v[column] - kv_mem[column]) * beta;
    }
    for row in 0..*dk {
        let kr = k[row];
        let qr = q[row];
        let srow = &mut s[row * dv..(row + 1) * dv];
        let row_kv = &kv_mem[..];
        let orow = &mut out_acc[..];
        for c in 0..*dv {
            srow[c] += kr * row_kv[c];
            orow[c] += qr * srow[c];
        }
    }
    for column in 0..*dv {
        out[column] = out_acc[column] * scale;
    }
    let z = &z_proj[head * dv..(head + 1) * dv];
    let mean_sq = out.iter().map(|v| v * v).sum::<f32>() / *dv as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((slot, &zv), &w) in out.iter_mut().zip(z.iter()).zip(norm_weight.iter()) {
        *slot = *slot * inv * w * silu(zv);
    }
}

fn recurse_heads(
    out: &mut [f32],
    state: &mut [f32],
    head_lo: usize,
    head_hi: usize,
    ctx: &GdnHeadCtx,
) {
    let span = head_hi - head_lo;
    if span <= 4 {
        for head in head_lo..head_hi {
            let base = head - head_lo;
            let o = &mut out[base * ctx.dv..][..ctx.dv];
            let s = &mut state[base * ctx.dk * ctx.dv..][..ctx.dk * ctx.dv];
            gdn_head(o, s, head, ctx);
        }
        return;
    }
    let mid = head_lo + span / 2;
    let dv = ctx.dv;
    let sd = ctx.dk * ctx.dv;
    let (out_l, out_r) = out.split_at_mut((mid - head_lo) * dv);
    let (state_l, state_r) = state.split_at_mut((mid - head_lo) * sd);
    std::thread::scope(|scope| {
        scope.spawn(move || recurse_heads(out_r, state_r, mid, head_hi, ctx));
        recurse_heads(out_l, state_l, head_lo, mid, ctx);
    });
}
