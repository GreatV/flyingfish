//! Batch-one MiniCPM5 Llama decoder with streamed projections and a KV cache.
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use ff_core::residency::WeightPhase;
use ff_core::weights::ModelWeights;
use serde::Deserialize;
use std::path::Path;

pub mod dspark;
pub mod memory;

struct ForwardOutput {
    logits: Tensor,
    features: Option<Tensor>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub eos_token_id: Vec<u32>,
    pub bos_token_id: u32,
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl Config {
    pub fn read(root: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)
            .context("invalid MiniCPM config")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.architectures == ["LlamaForCausalLM"] && self.model_type == "llama",
            "MiniCPM decoder requires LlamaForCausalLM; DSpark is a paired draft model"
        );
        anyhow::ensure!(
            self.hidden_act == "silu"
                && !self.attention_bias
                && !self.mlp_bias
                && self.rope_scaling.is_none(),
            "unsupported Llama activation, biases or scaled RoPE"
        );
        anyhow::ensure!(
            self.num_hidden_layers > 0
                && self.hidden_size > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0
                && self.max_position_embeddings > 0,
            "MiniCPM dimensions must be positive"
        );
        anyhow::ensure!(
            self.num_key_value_heads > 0
                && self.num_attention_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads)
                && self.head_dim > 0
                && self.head_dim.is_multiple_of(2)
                && self.num_attention_heads.checked_mul(self.head_dim) == Some(self.hidden_size),
            "invalid MiniCPM grouped-query attention geometry"
        );
        anyhow::ensure!(
            self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.
                && self.rope_theta.is_finite()
                && self.rope_theta > 0.,
            "invalid normalization or RoPE constant"
        );
        anyhow::ensure!(
            !self.eos_token_id.is_empty()
                && self
                    .eos_token_id
                    .iter()
                    .chain([&self.bos_token_id])
                    .all(|&id| (id as usize) < self.vocab_size),
            "invalid BOS/EOS token ids"
        );
        Ok(())
    }

    /// Every tensor one forward pass reads. The set does not change between
    /// prefill and decode, although residency can be selected per layer.
    pub fn weight_names(&self) -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".to_owned(),
            "model.norm.weight".to_owned(),
        ];
        if !self.tie_word_embeddings {
            names.push("lm_head.weight".to_owned());
        }
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                names.push(format!("{prefix}.{name}.weight"));
            }
        }
        names
    }

    /// The residency phases for a request that calls `forward` `steps` times,
    /// prefill included.
    ///
    /// Almost every tensor is read once per call. The embedding table is the
    /// exception when the checkpoint ties its word embeddings: `forward`
    /// gathers rows from it at the input *and* runs it again as the output
    /// head, so it is read twice per call. It is declared as its own phase so
    /// that reuse count is the 2× it really is instead of being averaged into
    /// the rest of the decoder, which would under-rank the largest tensor the
    /// model has.
    pub fn decode_phases(&self, steps: u64) -> Vec<WeightPhase> {
        const EMBEDDINGS: &str = "model.embed_tokens.weight";
        if !self.tie_word_embeddings {
            return vec![WeightPhase::new(
                "minicpm.decode",
                self.weight_names(),
                steps,
            )];
        }
        let rest = self
            .weight_names()
            .into_iter()
            .filter(|name| name != EMBEDDINGS);
        vec![
            WeightPhase::new("minicpm.decode", rest, steps),
            WeightPhase::new(
                "minicpm.decode.embeddings",
                [EMBEDDINGS],
                steps.saturating_mul(2),
            ),
        ]
    }

    /// Full tensors eligible for device retention, grouped so a constrained
    /// device can retain individual layers. Input embedding row gathers do
    /// not populate the full-tensor cache; only a tied output head needs the
    /// complete embedding table on the device.
    pub fn device_residency_phases(&self, steps: u64) -> Vec<WeightPhase> {
        let head = if self.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let mut phases = vec![WeightPhase::new(
            "minicpm.target.head",
            [head, "model.norm.weight"],
            steps,
        )];
        let names = self.weight_names();
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.");
            phases.push(WeightPhase::new(
                format!("minicpm.target.layer-{layer:04}"),
                names
                    .iter()
                    .filter(|name| name.starts_with(&prefix))
                    .cloned(),
                steps,
            ));
        }
        phases
    }

    pub fn validate_weights(&self, weights: &ModelWeights) -> Result<()> {
        self.validate()?;
        let h = self.hidden_size;
        let kv = self.num_key_value_heads * self.head_dim;
        let check = |name: &str, shape: &[usize]| -> Result<()> {
            let actual = weights.metadata(name)?;
            anyhow::ensure!(
                actual.shape == shape,
                "{name}: expected {shape:?}, got {:?}",
                actual.shape
            );
            Ok(())
        };
        check("model.embed_tokens.weight", &[self.vocab_size, h])?;
        check("model.norm.weight", &[h])?;
        if !self.tie_word_embeddings {
            check("lm_head.weight", &[self.vocab_size, h])?;
        }
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                check(&format!("{prefix}.{name}.weight"), &[h])?;
            }
            for (name, out, input) in [
                ("self_attn.q_proj", h, h),
                ("self_attn.k_proj", kv, h),
                ("self_attn.v_proj", kv, h),
                ("self_attn.o_proj", h, h),
                ("mlp.gate_proj", self.intermediate_size, h),
                ("mlp.up_proj", self.intermediate_size, h),
                ("mlp.down_proj", h, self.intermediate_size),
            ] {
                check(&format!("{prefix}.{name}.weight"), &[out, input])?;
            }
        }
        Ok(())
    }
}

pub struct Decoder {
    config: Config,
    weights: ModelWeights,
    device: Device,
    dtype: DType,
    query_chunk: usize,
    cache: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
    batch_invariant_decode: bool,
}

impl Decoder {
    pub fn new(
        config: Config,
        weights: ModelWeights,
        device: Device,
        query_chunk: usize,
    ) -> Result<Self> {
        anyhow::ensure!(query_chunk > 0, "attention query chunk must be positive");
        config.validate_weights(&weights)?;
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let cache = vec![None; config.num_hidden_layers];
        Ok(Self {
            config,
            weights,
            device,
            dtype,
            query_chunk,
            cache,
            position: 0,
            batch_invariant_decode: false,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Fix CUDA decode projection geometry and attention reduction extents.
    /// Both target-only and speculative runs must select this profile when
    /// comparing greedy output; it can round differently from native GEMMs.
    pub fn with_batch_invariant_decode(mut self, enabled: bool) -> Result<Self> {
        anyhow::ensure!(
            !enabled || self.device.is_cuda(),
            "batch-invariant decode requires CUDA"
        );
        anyhow::ensure!(
            self.position == 0 || enabled == self.batch_invariant_decode,
            "change decode arithmetic only with an empty request cache"
        );
        self.batch_invariant_decode = enabled;
        Ok(self)
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn clear_cache(&mut self) {
        self.cache.fill(None);
        self.position = 0;
    }

    /// Run an independent greedy request, retaining weights but releasing KV
    /// state on success, validation failure or callback cancellation.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<usize> {
        self.clear_cache();
        let result = (|| {
            anyhow::ensure!(
                !prompt.is_empty() && max_new_tokens > 0,
                "prompt and output limit must be nonempty"
            );
            anyhow::ensure!(
                prompt
                    .len()
                    .checked_add(max_new_tokens)
                    .is_some_and(|end| end <= self.config.max_position_embeddings),
                "request exceeds the target context limit"
            );
            let mut input = prompt.to_vec();
            let mut emitted = 0;
            for _ in 0..max_new_tokens {
                let next = self.forward(&input)?.argmax(0)?.to_scalar::<u32>()?;
                if self.config.eos_token_id.contains(&next) {
                    break;
                }
                emit(next)?;
                emitted += 1;
                input.clear();
                input.push(next);
            }
            Ok(emitted)
        })();
        self.clear_cache();
        result
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .weights
            .load(name, &self.device)?
            .to_dtype(self.dtype)?)
    }

    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let weight = self.get(&format!("{name}.weight"))?.t()?;
        if self.batch_invariant_decode && self.position > 0 && self.device.is_cuda() {
            let rows = x.dim(0)?;
            let mut outputs = Vec::new();
            for start in (0..rows).step_by(8) {
                let count = 8.min(rows - start);
                let input = x.narrow(0, start, count)?.pad_with_zeros(0, 0, 8 - count)?;
                outputs.push(input.matmul(&weight)?.narrow(0, 0, count)?);
            }
            return Ok(Tensor::cat(&outputs, 0)?);
        }
        Ok(x.matmul(&weight)?)
    }

    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let weight = self.get(&format!("{name}.weight"))?;
        let value = x.to_dtype(DType::F32)?;
        let variance = value.sqr()?.mean_keepdim(D::Minus1)?;
        Ok(value
            .broadcast_div(&(variance + self.config.rms_norm_eps)?.sqrt()?)?
            .to_dtype(self.dtype)?
            .broadcast_mul(&weight)?)
    }

    /// Consume prompt/decode tokens and return the next-token logits as F32.
    /// Any compute failure clears the cache so partially updated layers cannot
    /// be reused as if they represented the same prefix.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_capture(tokens, &[], false)?
            .logits
            .squeeze(0)?)
    }

    fn forward_capture(
        &mut self,
        tokens: &[u32],
        layers: &[usize],
        all_logits: bool,
    ) -> Result<ForwardOutput> {
        anyhow::ensure!(
            layers.iter().all(|&id| id < self.config.num_hidden_layers)
                && layers.windows(2).all(|pair| pair[0] < pair[1]),
            "capture layers must be sorted, distinct and within the target model"
        );
        anyhow::ensure!(!tokens.is_empty(), "input tokens cannot be empty");
        anyhow::ensure!(
            tokens
                .iter()
                .all(|&id| (id as usize) < self.config.vocab_size),
            "token id exceeds vocabulary"
        );
        let end = self
            .position
            .checked_add(tokens.len())
            .context("context length overflow")?;
        anyhow::ensure!(
            end <= self.config.max_position_embeddings,
            "context exceeds max_position_embeddings"
        );
        match self.forward_inner(tokens, layers, all_logits) {
            Ok(logits) => {
                self.position = end;
                Ok(logits)
            }
            Err(error) => {
                self.clear_cache();
                Err(error)
            }
        }
    }

    fn truncate_cache(&mut self, position: usize) -> Result<()> {
        anyhow::ensure!(
            position <= self.position,
            "cannot extend a KV cache by truncating it"
        );
        if position == 0 {
            self.clear_cache();
            return Ok(());
        }
        let result = (|| -> Result<()> {
            for entry in &mut self.cache {
                let (k, v) = entry.as_ref().context("missing target KV cache")?;
                let pair = (
                    k.narrow(2, 0, position)?.contiguous()?,
                    v.narrow(2, 0, position)?.contiguous()?,
                );
                *entry = Some(pair);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.clear_cache();
            return Err(error);
        }
        self.position = position;
        Ok(())
    }

    fn forward_inner(
        &mut self,
        tokens: &[u32],
        layers: &[usize],
        all_logits: bool,
    ) -> Result<ForwardOutput> {
        let length = tokens.len();
        let c = self.config.clone();
        let mut hidden = self
            .weights
            .load_rows("model.embed_tokens.weight", tokens, &self.device)?
            .to_dtype(self.dtype)?;
        let half = c.head_dim / 2;
        let angles: Vec<f32> = (self.position..self.position + length)
            .flat_map(|position| {
                (0..half).map(move |i| {
                    position as f32 / (c.rope_theta as f32).powf((2 * i) as f32 / c.head_dim as f32)
                })
            })
            .collect();
        let angles = Tensor::from_vec(angles, (length, half), &self.device)?;
        let cos = angles.cos()?.to_dtype(self.dtype)?;
        let sin = angles.sin()?.to_dtype(self.dtype)?;
        let mut features = Vec::with_capacity(layers.len());
        for layer in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let x = self.norm(&hidden, &format!("{prefix}.input_layernorm"))?;
            let project = |name: &str, heads: usize| -> Result<Tensor> {
                Ok(self
                    .linear(&x, &format!("{prefix}.self_attn.{name}"))?
                    .reshape((1, length, heads, c.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?)
            };
            let q = candle_nn::rotary_emb::rope(
                &project("q_proj", c.num_attention_heads)?,
                &cos,
                &sin,
            )?;
            let k = candle_nn::rotary_emb::rope(
                &project("k_proj", c.num_key_value_heads)?,
                &cos,
                &sin,
            )?;
            let v = project("v_proj", c.num_key_value_heads)?;
            let (k, v) = match &self.cache[layer] {
                Some((past_k, past_v)) => (
                    Tensor::cat(&[past_k, &k], 2)?,
                    Tensor::cat(&[past_v, &v], 2)?,
                ),
                None => (k, v),
            };
            let context =
                if self.batch_invariant_decode && self.position > 0 && self.device.is_cuda() {
                    attention_serial_queries(&q, &k, &v, self.position, self.query_chunk)?
                } else {
                    attention(&q, &k, &v, self.position, self.query_chunk)?
                }
                .transpose(1, 2)?
                .reshape((length, c.hidden_size))?
                .contiguous()?;
            self.cache[layer] = Some((k, v));
            hidden = (&hidden + self.linear(&context, &format!("{prefix}.self_attn.o_proj"))?)?;
            let x = self.norm(&hidden, &format!("{prefix}.post_attention_layernorm"))?;
            let gate = candle_nn::ops::silu(&self.linear(&x, &format!("{prefix}.mlp.gate_proj"))?)?;
            let up = self.linear(&x, &format!("{prefix}.mlp.up_proj"))?;
            hidden =
                (&hidden + self.linear(&(&gate * &up)?, &format!("{prefix}.mlp.down_proj"))?)?;
            if layers.contains(&layer) {
                features.push(hidden.clone());
            }
        }
        let output = if all_logits {
            hidden
        } else {
            hidden.narrow(0, length - 1, 1)?
        };
        let last = self.norm(&output, "model.norm")?;
        let head = if c.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        };
        Ok(ForwardOutput {
            logits: self.linear(&last, head)?.to_dtype(DType::F32)?,
            features: if features.is_empty() {
                None
            } else {
                Some(Tensor::cat(&features, 1)?)
            },
        })
    }
}

/// Grouped-query attention with a bounded tile of score rows. Persistent KV
/// stays grouped, but Candle's broadcast_matmul materializes expanded operands;
/// those temporary copies also contribute to the activation working set.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, offset: usize, chunk: usize) -> Result<Tensor> {
    attention_with_mask(q, k, v, Some(offset), chunk)
}

fn attention_serial_queries(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: usize,
    chunk: usize,
) -> Result<Tensor> {
    let length = q.dim(2)?;
    if length == 1 {
        return attention(q, k, v, offset, chunk);
    }
    let mut rows = Vec::with_capacity(length);
    for row in 0..length {
        let end = offset + row + 1;
        rows.push(attention(
            &q.narrow(2, row, 1)?.contiguous()?,
            &k.narrow(2, 0, end)?.contiguous()?,
            &v.narrow(2, 0, end)?.contiguous()?,
            offset + row,
            chunk,
        )?);
    }
    Ok(Tensor::cat(&rows, 2)?)
}

fn attention_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: Option<usize>,
    chunk: usize,
) -> Result<Tensor> {
    let (_, heads, length, dim) = q.dims4()?;
    let (_, kv_heads, total, _) = k.dims4()?;
    let groups = heads / kv_heads;
    let mut outputs = Vec::with_capacity(kv_heads);
    for head in 0..kv_heads {
        let keys = k
            .narrow(1, head, 1)?
            .to_dtype(DType::F32)?
            .transpose(2, 3)?
            .contiguous()?;
        let values = v.narrow(1, head, 1)?.to_dtype(DType::F32)?.contiguous()?;
        let mut rows = Vec::new();
        for start in (0..length).step_by(chunk) {
            let count = chunk.min(length - start);
            let query = q
                .narrow(1, head * groups, groups)?
                .narrow(2, start, count)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let mask: Vec<f32> = (0..count)
                .flat_map(|row| {
                    (0..total).map(move |col| {
                        if offset.is_none_or(|offset| col <= offset + start + row) {
                            0.
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, count, total), q.device())?;
            let scores =
                (query.broadcast_matmul(&keys)? / (dim as f64).sqrt())?.broadcast_add(&mask)?;
            rows.push(
                candle_nn::ops::softmax_last_dim(&scores)?
                    .broadcast_matmul(&values)?
                    .to_dtype(q.dtype())?,
            );
        }
        outputs.push(Tensor::cat(&rows, 2)?);
    }
    Ok(Tensor::cat(&outputs, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::weights::{CachePolicy, WeightSource};
    use std::collections::HashMap;

    #[test]
    fn attention_preserves_kv_head_groups_and_masks_future_tokens() {
        let q = Tensor::zeros((1, 4, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::zeros((1, 2, 3, 2), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::from_vec(
            vec![1f32, 1., 3., 3., 5., 5., 10., 10., 30., 30., 50., 50.],
            (1, 2, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let values = attention(&q, &k, &v, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = [
            2., 2., 3., 3., 2., 2., 3., 3., 20., 20., 30., 30., 20., 20., 30., 30.,
        ];
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
        );
        let serial = attention_serial_queries(&q, &k, &v, 1, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            serial
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1e-5)
        );
    }

    pub(super) fn fixture() -> (tempfile::TempDir, Config) {
        let config: Config = serde_json::from_value(serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 8, "intermediate_size": 12, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 16, "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
            "rope_theta": 10000., "eos_token_id": [1], "bos_token_id": 0, "hidden_act": "silu"
        }))
        .unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        let mut index = 0usize;
        let mut insert = |name: String, shape: &[usize]| {
            let size: usize = shape.iter().product();
            let values: Vec<f32> = (0..size)
                .map(|i| {
                    if shape.len() == 1 {
                        1.0
                    } else {
                        ((i + index) as f32 * 0.7).sin() * 0.15
                    }
                })
                .collect();
            index += size;
            tensors.insert(name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
        };
        insert("model.embed_tokens.weight".into(), &[16, 8]);
        insert("lm_head.weight".into(), &[16, 8]);
        insert("model.norm.weight".into(), &[8]);
        for layer in 0..2 {
            let p = format!("model.layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                insert(format!("{p}.{name}.weight"), &[8]);
            }
            for (name, shape) in [
                ("self_attn.q_proj", [8, 8]),
                ("self_attn.k_proj", [4, 8]),
                ("self_attn.v_proj", [4, 8]),
                ("self_attn.o_proj", [8, 8]),
                ("mlp.gate_proj", [12, 8]),
                ("mlp.up_proj", [12, 8]),
                ("mlp.down_proj", [8, 12]),
            ] {
                insert(format!("{p}.{name}.weight"), &shape);
            }
        }
        candle_core::safetensors::save(&tensors, temporary.path().join("model.safetensors"))
            .unwrap();
        (temporary, config)
    }

    pub(super) fn decoder(root: &Path, config: Config, chunk: usize) -> Decoder {
        Decoder::new(
            config,
            ModelWeights::open(root, WeightSource::Mmap, CachePolicy::new(1)).unwrap(),
            Device::Cpu,
            chunk,
        )
        .unwrap()
    }

    #[test]
    fn independent_requests_retain_weights_and_clear_kv_after_success_or_cancellation() {
        use ff_core::weights::{DeviceCache, DeviceCachePolicy};
        let (directory, mut config) = fixture();
        let first = decoder(directory.path(), config.clone(), 2)
            .forward(&[3, 5])
            .unwrap()
            .argmax(0)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        config.eos_token_id = vec![(first + 1) % config.vocab_size as u32];
        let cache = DeviceCache::new(DeviceCachePolicy::with_max_bytes(64 << 10));
        let mut weights =
            ModelWeights::open(directory.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        weights.configure_device_cache(cache.clone()).unwrap();
        let mut worker = Decoder::new(config.clone(), weights, Device::Cpu, 2).unwrap();
        for prompt in [&[3, 5][..], &[7, 3, 2][..], &[3, 5][..]] {
            let mut fresh = decoder(directory.path(), config.clone(), 2);
            let mut input = prompt.to_vec();
            let mut expected = Vec::new();
            for _ in 0..4 {
                let token = fresh
                    .forward(&input)
                    .unwrap()
                    .argmax(0)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap();
                if config.eos_token_id.contains(&token) {
                    break;
                }
                expected.push(token);
                input = vec![token];
            }
            let mut actual = Vec::new();
            let count = worker
                .generate_greedy(prompt, 4, |t| {
                    actual.push(t);
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(count, actual.len());
            assert_eq!(worker.position(), 0);
            assert!(worker.cache.iter().all(Option::is_none));
        }
        assert!(cache.stats().resident_bytes > 0 && cache.stats().hits > 0);
        let resident = cache.stats().resident_bytes;
        assert!(
            worker
                .generate_greedy(&[3, 5], 4, |_| anyhow::bail!("cancelled"))
                .is_err()
        );
        assert_eq!(worker.position(), 0);
        assert!(worker.cache.iter().all(Option::is_none));
        assert_eq!(cache.stats().resident_bytes, resident);
        for (prompt, limit) in [(&[][..], 4), (&[3][..], 0), (&[3][..], usize::MAX)] {
            assert!(worker.generate_greedy(prompt, limit, |_| Ok(())).is_err());
            assert_eq!(worker.position(), 0);
        }
        let fresh = decoder(directory.path(), config, 2)
            .forward(&[3, 5])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            worker.forward(&[3, 5]).unwrap().to_vec1::<f32>().unwrap(),
            fresh
        );
    }

    #[test]
    fn cached_decode_and_chunked_prefill_match_full_causal_forward() {
        let (dir, config) = fixture();
        let mut full = decoder(dir.path(), config.clone(), 32);
        let mut cached = decoder(dir.path(), config.clone(), 1);
        let reference = full.forward(&[3, 5, 8, 2]).unwrap();
        cached.forward(&[3, 5]).unwrap();
        cached.forward(&[8]).unwrap();
        let actual = cached.forward(&[2]).unwrap();
        let error = (&reference - actual)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "cache mismatch {error}");
        assert_eq!(cached.position(), 4);
        cached.clear_cache();
        let reset = cached.forward(&[3, 5, 8, 2]).unwrap();
        let error = (&reference - reset)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5, "query chunk mismatch {error}");
    }

    #[test]
    fn invalid_inputs_do_not_advance_cache() {
        let (dir, config) = fixture();
        let mut model = decoder(dir.path(), config, 2);
        model.forward(&[3]).unwrap();
        assert!(model.forward(&[]).is_err());
        assert!(model.forward(&[16]).is_err());
        assert!(model.forward(&[0; 16]).is_err());
        assert_eq!(model.position(), 1);
    }

    #[test]
    fn draft_or_unsupported_config_is_rejected_before_loading() {
        let (_, mut config) = fixture();
        config.architectures = vec!["Qwen3DSparkModel".into()];
        assert!(config.validate().is_err());
        config.architectures = vec!["LlamaForCausalLM".into()];
        config.num_key_value_heads = 3;
        assert!(config.validate().is_err());
    }

    #[test]
    fn device_resident_weights_are_bit_identical_to_streamed() {
        let (dir, config) = fixture();
        let open = |residency: ff_core::weights::DeviceCache| {
            let mut weights =
                ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
            weights.configure_device_cache(residency).unwrap();
            Decoder::new(config.clone(), weights, Device::Cpu, 32).unwrap()
        };
        let mut streamed = open(ff_core::weights::DeviceCache::disabled());
        let mut resident = open(ff_core::weights::DeviceCache::new(
            ff_core::weights::DeviceCachePolicy::with_max_bytes(1 << 20),
        ));
        for tokens in [&[3u32, 5, 8, 2][..], &[7, 1], &[4]] {
            let expected = streamed.forward(tokens).unwrap();
            let actual = resident.forward(tokens).unwrap();
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "resident and streamed logits differ for {tokens:?}"
            );
        }
        let stats = resident.weights.device_cache_stats();
        assert!(stats.hits > 0, "the resident run never hit its cache");
        assert!(stats.resident_bytes > 0);
    }

    #[test]
    fn declared_decode_phase_matches_counted_reads() {
        let (dir, config) = fixture();
        assert!(!config.tie_word_embeddings);
        let mut model = decoder(dir.path(), config.clone(), 32);
        model.weights.count_tensor_reads(true);
        model.forward(&[3, 5]).unwrap();
        model.forward(&[8]).unwrap();
        let reads = model.weights.tensor_reads();
        let [phase] = &config.decode_phases(2)[..] else {
            panic!("an untied checkpoint declares one phase")
        };
        let diff = ff_core::residency::verify_phase_reads(phase, &reads);
        assert!(diff.is_exact(), "{diff:?}");
        assert!(
            reads.values().all(|&count| count == 2),
            "each weight is read once per forward call: {reads:?}"
        );
    }

    /// Tied word embeddings read the table at the input and again as the
    /// output head, so it is the one tensor read twice per call. Declaring it
    /// in the same phase as everything else would under-rank the model's
    /// largest tensor by half.
    #[test]
    fn tied_embeddings_are_declared_as_their_own_twice_per_step_phase() {
        let (dir, mut config) = fixture();
        config.tie_word_embeddings = true;
        let mut model = decoder(dir.path(), config.clone(), 32);
        model.weights.count_tensor_reads(true);
        model.forward(&[3, 5]).unwrap();
        model.forward(&[8]).unwrap();
        let reads = model.weights.tensor_reads();
        assert_eq!(reads["model.embed_tokens.weight"], 4, "twice per forward");
        assert_eq!(reads["model.norm.weight"], 2, "once per forward");
        assert!(!reads.contains_key("lm_head.weight"), "{reads:?}");

        let phases = config.decode_phases(2);
        let [decode, embeddings] = &phases[..] else {
            panic!("a tied checkpoint declares the table separately")
        };
        assert_eq!(decode.reuse_count, 2);
        assert_eq!(embeddings.reuse_count, 4);
        assert_eq!(
            embeddings.tensors,
            std::collections::BTreeSet::from(["model.embed_tokens.weight".to_owned()])
        );
        let combined = ff_core::residency::WeightPhase::new(
            "combined",
            decode.tensors.iter().chain(&embeddings.tensors).cloned(),
            1,
        );
        let diff = ff_core::residency::verify_phase_reads(&combined, &reads);
        assert!(diff.is_exact(), "{diff:?}");
    }
}
