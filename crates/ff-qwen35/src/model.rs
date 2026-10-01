//! Dense Qwen3.8-27B text forward (CPU reference path first).
//!
//! Semantics ported from HF `modeling_qwen3_5.py` (transformers 5.8), NOT
//! from ff-edge0: every RMSNorm here is the zero-centered variant
//! (`x * rsqrt(mean(x^2)+eps) * (1 + w)`), attention q/k norms included;
//! the GDN gated norm is plain (`norm(x) * w * silu(z)`). The GDN
//! recurrence is update-then-read: S *= decay; kv = S^T k; delta =
//! (v - kv) * beta; S += k ⊗ delta; out = S^T q.

use crate::config::{LayerKind, Qwen35Config, TEXT_PREFIX};
use crate::weights::Qwen35Weights;
use anyhow::Result;
use ff_core::math::{l2norm, silu};

/// Zero-centered RMSNorm: `x * rsqrt(mean(x^2)+eps) * (1 + w)`.
fn rmsnorm_zc(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| v * inv * (1.0 + w))
        .collect()
}

/// Interleaved mrope: freq i picks axis h (i%3==1 && i<3*sec[1]), w
/// (i%3==2 && i<3*sec[2]), else t. Bit-identical to scalar at [p,p,p].
fn rope_mrope(
    x: &mut [f32],
    pos3: [usize; 3],
    rotary_dim: usize,
    theta: f64,
    sections: [usize; 3],
) {
    let half = rotary_dim / 2;
    for i in 0..half {
        let axis = if i % 3 == 1 && i < 3 * sections[1] {
            1
        } else if i % 3 == 2 && i < 3 * sections[2] {
            2
        } else {
            0
        };
        let freq = theta.powf(-(2.0 * i as f64) / rotary_dim as f64);
        let angle = pos3[axis] as f64 * freq;
        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
        let (x1, x2) = (x[i], x[i + half]);
        x[i] = x1 * cos - x2 * sin;
        x[i + half] = x2 * cos + x1 * sin;
    }
}

/// Decode rope position = KV index + mrope delta (negative for image
/// prompts). Free function so CI can pin the arithmetic without the model.
pub fn decode_rope_pos(position: usize, mrope_delta: i64) -> usize {
    (position as i64 + mrope_delta) as usize
}

struct GdnState {
    /// [k-1][conv_dim] ring (oldest first), as in the edge0 layout.
    conv: Vec<f32>,
    /// Per value head: [dk][dv] (HF orientation: key rows, value cols).
    recurrent: Vec<f32>,
}

struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

pub struct Qwen35Text {
    config: Qwen35Config,
    weights: Qwen35Weights,
    /// Loaded-once quantized projections (a fresh GroupQuant per call would
    /// re-copy ~13.5 GB per token).
    cache: std::cell::RefCell<
        std::collections::HashMap<String, std::sync::Arc<ff_edge0::int4::GroupQuant>>,
    >,
    gdn: Vec<GdnState>,
    kv: Vec<KvCache>,
    position: usize,
    /// Rope position (t,h,w) of the in-flight token; `position` stays the
    /// KV index.
    pos3: [usize; 3],
    /// Decode offset max(prefill)+1 - prompt_len; negative for image
    /// prompts (i64 by necessity).
    mrope_delta: i64,
}

impl Qwen35Text {
    pub fn load(dir: &std::path::Path, config: Qwen35Config) -> Result<Self> {
        let weights = Qwen35Weights::open(dir)?;
        let text = &config.text_config;
        let conv_dim = text.conv_dim();
        let k = text.linear_conv_kernel_dim;
        let mut gdn = Vec::new();
        let mut kv = Vec::new();
        for layer in 0..text.num_hidden_layers {
            match text.layer_kind(layer) {
                LayerKind::LinearAttention => gdn.push(GdnState {
                    conv: vec![0.0; (k - 1) * conv_dim],
                    recurrent: vec![
                        0.0;
                        text.linear_num_value_heads
                            * text.linear_key_head_dim
                            * text.linear_value_head_dim
                    ],
                }),
                LayerKind::FullAttention => kv.push(KvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                    len: 0,
                }),
            }
        }
        Ok(Self {
            config,
            weights,
            cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            gdn,
            kv,
            position: 0,
            pos3: [0; 3],
            mrope_delta: 0,
        })
    }

    fn cached_proj(&self, name: &str) -> Result<std::sync::Arc<ff_edge0::int4::GroupQuant>> {
        let mut cache = self.cache.borrow_mut();
        if !cache.contains_key(name) {
            cache.insert(
                name.to_string(),
                std::sync::Arc::new(self.weights.quant_projection(name)?),
            );
        }
        Ok(std::sync::Arc::clone(cache.get(name).unwrap()))
    }

    fn proj(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        match self.weights.format() {
            crate::weights::INT4 => Ok(self.cached_proj(name)?.matvec(x, None)),
            _ => self.weights.matvec16(name, x),
        }
    }

    fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        let name = format!("{TEXT_PREFIX}.embed_tokens");
        match self.weights.format() {
            crate::weights::INT4 => Ok(self.cached_proj(&name)?.row(token as usize)),
            _ => self.weights.row16(&name, token as usize),
        }
    }

    /// Return the hidden and final-normalized vectors for one token.
    pub fn forward_raw(&mut self, token: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        let pos3 = [decode_rope_pos(self.position, self.mrope_delta); 3];
        self.forward_hidden(self.embed_row(token)?, pos3)
    }

    /// Text token at an explicit mrope position.
    pub fn forward_at(&mut self, token: u32, pos3: [usize; 3]) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = self.embed_row(token)?;
        self.forward_hidden(h, pos3)
    }

    /// Vision-row splice: hidden from the tower, not the embedding table.
    pub fn forward_vision_row(
        &mut self,
        row: Vec<f32>,
        pos3: [usize; 3],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_hidden(row, pos3)
    }

    pub fn set_mrope_delta(&mut self, delta: i64) {
        self.mrope_delta = delta;
    }

    /// Rotary position for the next token.
    pub fn next_rope_pos(&self) -> usize {
        decode_rope_pos(self.position, self.mrope_delta)
    }

    fn forward_hidden(
        &mut self,
        hidden: Vec<f32>,
        pos3: [usize; 3],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut hidden = hidden;
        self.pos3 = pos3;
        let text = self.config.text_config.clone();
        let eps = text.rms_norm_eps as f32;
        let mut gdn_index = 0;
        let mut kv_index = 0;
        for layer in 0..text.num_hidden_layers {
            let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
            let input_norm = self
                .weights
                .f32_named(&format!("{prefix}.input_layernorm.weight"))?;
            let normed = rmsnorm_zc(&hidden, &input_norm, eps);
            let mixer = if text.layer_kind(layer) == LayerKind::LinearAttention {
                let out = self.gdn_step(gdn_index, &prefix, &normed)?;
                gdn_index += 1;
                out
            } else {
                let out = self.attn_step(kv_index, &prefix, &normed)?;
                kv_index += 1;
                out
            };
            for (h, m) in hidden.iter_mut().zip(&mixer) {
                *h += m;
            }
            let post_norm = self
                .weights
                .f32_named(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let normed = rmsnorm_zc(&hidden, &post_norm, eps);
            let gate = self.proj(&format!("{prefix}.mlp.gate_proj"), &normed)?;
            let up = self.proj(&format!("{prefix}.mlp.up_proj"), &normed)?;
            let inner: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let down = self.proj(&format!("{prefix}.mlp.down_proj"), &inner)?;
            for (h, d) in hidden.iter_mut().zip(&down) {
                *h += d;
            }
        }
        self.position += 1;
        let final_norm = self
            .weights
            .f32_named(&format!("{TEXT_PREFIX}.norm.weight"))?;
        let normed = rmsnorm_zc(&hidden, &final_norm, eps);
        Ok((hidden, normed))
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        Ok(self.forward_raw(token)?.1)
    }

    pub fn logits(&self, normed: &[f32]) -> Result<Vec<f32>> {
        self.proj("lm_head", normed)
    }

    fn gdn_step(&mut self, state_index: usize, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let text = &self.config.text_config;
        let num_v = text.linear_num_value_heads;
        let num_k = text.linear_num_key_heads;
        let dk = text.linear_key_head_dim;
        let dv = text.linear_value_head_dim;
        let key_dim = num_k * dk;
        let value_dim = num_v * dv;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = text.linear_conv_kernel_dim;
        let eps = text.rms_norm_eps as f32;

        let qkv = self.proj(&format!("{prefix}.linear_attn.in_proj_qkv"), x)?;
        let z = self.proj(&format!("{prefix}.linear_attn.in_proj_z"), x)?;
        let b = self.proj(&format!("{prefix}.linear_attn.in_proj_b"), x)?;
        let a = self.proj(&format!("{prefix}.linear_attn.in_proj_a"), x)?;

        let conv_w = self
            .weights
            .f32_named(&format!("{prefix}.linear_attn.conv1d.weight"))?;
        let dt_bias = self
            .weights
            .f32_named(&format!("{prefix}.linear_attn.dt_bias"))?;
        let a_log = self
            .weights
            .f32_named(&format!("{prefix}.linear_attn.A_log"))?;
        let norm_w = self
            .weights
            .f32_named(&format!("{prefix}.linear_attn.norm.weight"))?;

        // Causal conv k=4 with silu, ring state.
        let state = &mut self.gdn[state_index];
        let mut conv_in = state.conv.clone();
        conv_in.extend_from_slice(&qkv);
        let mut conv_out = vec![0f32; conv_dim];
        for (channel, out) in conv_out.iter_mut().enumerate() {
            let mut acc = 0f32;
            for j in 0..kernel {
                acc += conv_w[channel * kernel + j] * conv_in[j * conv_dim + channel];
            }
            *out = silu(acc);
        }
        state.conv.copy_from_slice(&conv_in[conv_dim..]);

        let q_all = &conv_out[..key_dim];
        let k_all = &conv_out[key_dim..2 * key_dim];
        let v_all = &conv_out[2 * key_dim..];

        let scale = 1.0 / (dk as f32).sqrt();
        let mut out = vec![0f32; value_dim];
        let ratio = num_v / num_k;
        for h in 0..num_v {
            let kh = h / ratio;
            let q = l2norm(&q_all[kh * dk..(kh + 1) * dk]);
            let kvec = l2norm(&k_all[kh * dk..(kh + 1) * dk]);
            let v = &v_all[h * dv..(h + 1) * dv];
            let ap = a[h] + dt_bias[h];
            let sp = if ap > 20.0 { ap } else { (1.0 + ap.exp()).ln() };
            let decay = (-a_log[h].exp() * sp).exp();
            let beta = 1.0 / (1.0 + (-b[h]).exp());

            // S: [dk][dv] for this head.
            let s = &mut state.recurrent[h * dk * dv..(h + 1) * dk * dv];
            let mut head_out = vec![0f32; dv];
            for j in 0..dv {
                let mut kv = 0f32;
                for i in 0..dk {
                    s[i * dv + j] *= decay;
                    kv += kvec[i] * s[i * dv + j];
                }
                let delta = (v[j] - kv) * beta;
                let mut o = 0f32;
                for i in 0..dk {
                    let sv = s[i * dv + j] + kvec[i] * delta;
                    s[i * dv + j] = sv;
                    o += q[i] * scale * sv;
                }
                head_out[j] = o;
            }
            // Gated RMSNorm over the head: plain weight, then silu(z) gate.
            let inv = (head_out.iter().map(|v| v * v).sum::<f32>() / dv as f32 + eps)
                .sqrt()
                .recip();
            for (j, o) in head_out.iter_mut().enumerate() {
                let zv = z[h * dv + j];
                *o = *o * inv * norm_w[j] * silu(zv);
            }
            out[h * dv..(h + 1) * dv].copy_from_slice(&head_out);
        }
        self.proj(&format!("{prefix}.linear_attn.out_proj"), &out)
    }

    fn attn_step(&mut self, kv_index: usize, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
        let text = &self.config.text_config;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let rotary_dim = (head_dim as f64 * text.rope.partial_rotary_factor) as usize;
        let eps = text.rms_norm_eps as f32;

        let q_raw = self.proj(&format!("{prefix}.self_attn.q_proj"), x)?;
        let k_raw = self.proj(&format!("{prefix}.self_attn.k_proj"), x)?;
        let v_raw = self.proj(&format!("{prefix}.self_attn.v_proj"), x)?;
        let q_norm_w = self
            .weights
            .f32_named(&format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm_w = self
            .weights
            .f32_named(&format!("{prefix}.self_attn.k_norm.weight"))?;

        // q_proj is per-head [q | gate] interleaved.
        let mut query = vec![0f32; heads * head_dim];
        let mut gate = vec![0f32; heads * head_dim];
        for h in 0..heads {
            let base = h * head_dim * 2;
            query[h * head_dim..(h + 1) * head_dim].copy_from_slice(&q_raw[base..base + head_dim]);
            gate[h * head_dim..(h + 1) * head_dim]
                .copy_from_slice(&q_raw[base + head_dim..base + 2 * head_dim]);
        }

        let pos3 = self.pos3;
        let sections = text.rope.mrope_section;
        let theta = text.rope.rope_theta;
        let kv_stride = kv_heads * head_dim;
        let cache = &mut self.kv[kv_index];
        for kh in 0..kv_heads {
            let mut kvec = k_raw[kh * head_dim..(kh + 1) * head_dim].to_vec();
            kvec = rmsnorm_zc(&kvec, &k_norm_w, eps);
            rope_mrope(&mut kvec, pos3, rotary_dim, theta, sections);
            cache.keys.extend_from_slice(&kvec);
        }
        cache.values.extend_from_slice(&v_raw);
        cache.len += 1;

        let mut attn_out = vec![0f32; heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..heads {
            let kv_head = h / (heads / kv_heads);
            let mut q = query[h * head_dim..(h + 1) * head_dim].to_vec();
            q = rmsnorm_zc(&q, &q_norm_w, eps);
            rope_mrope(&mut q, pos3, rotary_dim, theta, sections);
            let mut scores = Vec::with_capacity(cache.len);
            for keys in cache.keys.chunks_exact(kv_stride) {
                let k = &keys[kv_head * head_dim..][..head_dim];
                let dot: f32 = q.iter().zip(k).map(|(qv, kv)| qv * kv).sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            for (step, score) in scores.iter().enumerate() {
                let w = (score - max).exp() / sum;
                let v = &cache.values[step * kv_stride + kv_head * head_dim..][..head_dim];
                for (o, &vv) in attn_out[h * head_dim..(h + 1) * head_dim].iter_mut().zip(v) {
                    *o += w * vv;
                }
            }
        }
        for (o, gv) in attn_out.iter_mut().zip(&gate) {
            *o *= 1.0 / (1.0 + (-gv).exp());
        }
        self.proj(&format!("{prefix}.self_attn.o_proj"), &attn_out)
    }
}
