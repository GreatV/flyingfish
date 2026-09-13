use crate::{
    core,
    layout::TEXT_TAG,
    policy::{ExecutionBackendPolicy, H3QwenNumericalContract, H3QwenVisionLinearGeometry},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use ff_core::weights::{CachePolicy, ModelWeights, WeightAccessStats, WeightSource};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path};
use tokenizers::Tokenizer;

/// Whether the pinned text-attention kernel covers this request.
///
/// The kernel was transcribed for exactly the official coffee profile. Callers
/// use this as a dispatch condition, not an admission check: everything else
/// runs on the portable chunked path, which is what the recorded contract then
/// names.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn exact_text_attention_covers(
    batch: usize,
    sequence: usize,
    attention_query_chunk_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> bool {
    batch == 1
        && sequence == 357
        && attention_query_chunk_size == 357
        && num_attention_heads == 64
        && num_key_value_heads == 8
        && head_dim == 128
}

/// Whether this device may run the kernels compiled from this repository.
fn tuned_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::tuned_kernels_available(device)
}

/// Whether this host's vendor libraries are the reference ones, which is what
/// the cuBLASLt and cuDNN operators reproduce.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn reference_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::profile::reference_libraries_available(device)
}

const MAX_CACHED_CAUSAL_MASK_BYTES: usize = 8 * 1024 * 1024;

enum CausalMask {
    Full(Tensor),
    Compact {
        positions: Tensor,
        negative_infinity: Tensor,
    },
}

impl CausalMask {
    fn new(sequence: usize, dtype: DType, device: &Device) -> Result<Self> {
        anyhow::ensure!(sequence > 0, "causal mask sequence must be non-zero");
        if full_causal_mask_construction_bytes(sequence, dtype)? <= MAX_CACHED_CAUSAL_MASK_BYTES {
            Ok(Self::Full(causal_attention_mask(
                sequence, 0, sequence, dtype, device,
            )?))
        } else {
            let end = u32::try_from(sequence)
                .context("causal mask sequence exceeds U32 position indexing")?;
            Ok(Self::Compact {
                positions: Tensor::arange(0u32, end, device)?,
                negative_infinity: Tensor::new(core::qwen_attention_mask_minimum(dtype)?, device)?
                    .to_dtype(dtype)?,
            })
        }
    }

    fn apply(&self, scores: &Tensor, query_start: usize, query_length: usize) -> Result<Tensor> {
        let sequence = match self {
            Self::Full(mask) => mask.dim(3)?,
            Self::Compact { positions, .. } => positions.dim(0)?,
        };
        validate_causal_query_range(sequence, query_start, query_length)?;
        let (_, _, score_query_length, score_sequence) = scores
            .dims4()
            .context("attention scores must be [batch, heads, query, sequence]")?;
        anyhow::ensure!(
            score_query_length == query_length && score_sequence == sequence,
            "attention score shape does not match the causal-mask query range"
        );
        match self {
            Self::Full(mask) => {
                let mask = mask
                    .narrow(2, query_start, query_length)
                    .context("failed to narrow cached causal mask")?;
                scores.broadcast_add(&mask).map_err(Into::into)
            }
            Self::Compact {
                positions,
                negative_infinity,
            } => {
                let future = compact_causal_future_mask(positions, query_start, query_length)?
                    .broadcast_as(scores.shape())?;
                anyhow::ensure!(
                    negative_infinity.dtype() == scores.dtype()
                        && negative_infinity.device().same_device(scores.device()),
                    "attention scores differ from the compact causal-mask dtype or device"
                );
                let negative_infinity = negative_infinity.broadcast_as(scores.shape())?;
                future
                    .where_cond(&negative_infinity, scores)
                    .context("failed to apply compact causal mask")
            }
        }
    }
}

fn full_causal_mask_construction_bytes(sequence: usize, dtype: DType) -> Result<usize> {
    let elements = sequence
        .checked_mul(sequence)
        .context("causal mask element count overflow")?;
    let bytes_per_element = std::mem::size_of::<f32>()
        + if dtype == DType::F32 {
            0
        } else {
            dtype.size_in_bytes()
        };
    elements
        .checked_mul(bytes_per_element)
        .context("causal mask byte count overflow")
}

#[derive(Clone, Debug, Deserialize)]
struct EncoderRootConfig {
    text_config: TextEncoderConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextEncoderConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

pub struct PromptEncoding {
    pub embeddings: Tensor,
    pub text_token_tags: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub numerical_contract: H3QwenNumericalContract,
}

#[cfg(test)]
pub(crate) fn test_qwen_numerical_contract(language_rows: usize) -> H3QwenNumericalContract {
    H3QwenNumericalContract::for_verified_target(
        ExecutionBackendPolicy::Cpu,
        NonZeroUsize::new(language_rows).unwrap(),
        NonZeroUsize::new(language_rows).unwrap(),
        0,
        H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
    )
    .unwrap()
}

pub struct StreamedTextEncoder {
    weights: ModelWeights,
    config: TextEncoderConfig,
    device: Device,
    target_hidden_state: usize,
    attention_query_chunk_size: usize,
}

impl StreamedTextEncoder {
    pub fn open(
        component_dir: impl AsRef<Path>,
        source: WeightSource,
        cache_policy: CachePolicy,
        device: Device,
        target_hidden_state: usize,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        let component_dir = component_dir.as_ref();
        let config_path = component_dir.join("config.json");
        let bytes = fs::read(&config_path).with_context(|| {
            format!(
                "failed to read text encoder config {}",
                config_path.display()
            )
        })?;
        let root: EncoderRootConfig = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid text encoder config {}", config_path.display()))?;
        let weights = ModelWeights::open(component_dir, source, cache_policy)?;
        Self::new(
            weights,
            root.text_config,
            device,
            target_hidden_state,
            attention_query_chunk_size,
        )
    }

    fn new(
        weights: ModelWeights,
        config: TextEncoderConfig,
        device: Device,
        target_hidden_state: usize,
        attention_query_chunk_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            target_hidden_state > 0,
            "target hidden state must follow at least one layer"
        );
        anyhow::ensure!(
            target_hidden_state < config.num_hidden_layers,
            "H3 needs a pre-final-norm hidden state before the encoder's last layer"
        );
        anyhow::ensure!(
            attention_query_chunk_size > 0,
            "attention query chunk size must be non-zero"
        );
        anyhow::ensure!(
            config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads),
            "attention heads are not divisible by KV heads"
        );
        anyhow::ensure!(
            config.head_dim.is_multiple_of(2),
            "attention head dimension must be even"
        );
        Ok(Self {
            weights,
            config,
            device,
            target_hidden_state,
            attention_query_chunk_size,
        })
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.weights.access_stats()
    }

    pub fn encode_prompt(&self, tokenizer: &Tokenizer, prompt: &str) -> Result<PromptEncoding> {
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize prompt: {error}"))?;
        let token_ids = encoding.get_ids().to_vec();
        anyhow::ensure!(
            !token_ids.is_empty(),
            "prompt tokenized to an empty sequence"
        );
        let numerical_contract = self.numerical_contract(token_ids.len())?;
        let embeddings = self.encode_token_ids_after_preflight(&token_ids)?;
        Ok(PromptEncoding {
            text_token_tags: vec![TEXT_TAG; token_ids.len()],
            token_ids,
            embeddings,
            numerical_contract,
        })
    }

    pub fn encode_token_ids(&self, token_ids: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(!token_ids.is_empty(), "token sequence must not be empty");
        if tuned_cuda(&self.device) {
            anyhow::ensure!(
                token_ids.len() <= core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS,
                "Qwen eager CUDA attention has {} token rows, exceeding the verified exact softmax range 1..={}; set {}=0 to run it through Candle's kernels instead",
                token_ids.len(),
                core::QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS,
                crate::cuda::profile::DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE
            );
        }
        anyhow::ensure!(
            token_ids
                .iter()
                .all(|&id| id < self.config.vocab_size as u32),
            "token ID exceeds text encoder vocabulary"
        );
        self.numerical_contract(token_ids.len())?;
        self.encode_token_ids_after_preflight(token_ids)
    }

    fn numerical_contract(&self, language_rows: usize) -> Result<H3QwenNumericalContract> {
        let configured_query_rows = NonZeroUsize::new(self.attention_query_chunk_size)
            .context("Qwen configured query rows must be non-zero")?;
        let language_rows =
            NonZeroUsize::new(language_rows).context("Qwen language rows must be non-zero")?;
        let geometry = H3QwenVisionLinearGeometry::from_patch_rows(0, 0)?;
        let backend = ExecutionBackendPolicy::from_device(&self.device);
        let contract = H3QwenNumericalContract::for_target_with_grids(
            backend,
            backend
                .is_cuda()
                .then(|| crate::policy::CudaCapabilities::from_device(&self.device)),
            configured_query_rows,
            language_rows,
            0,
            geometry,
            &[],
        )?;
        contract.validate_for(
            &self.device,
            configured_query_rows,
            language_rows,
            0,
            geometry,
        )?;
        Ok(contract)
    }

    fn encode_token_ids_after_preflight(&self, token_ids: &[u32]) -> Result<Tensor> {
        let mut hidden = self
            .weights
            .load_rows(
                "model.language_model.embed_tokens.weight",
                token_ids,
                &self.device,
            )?
            .unsqueeze(0)?;
        let (cos, sin) = text_rope(
            token_ids.len(),
            self.config.head_dim,
            self.config.rope_theta,
            hidden.dtype(),
            &self.device,
        )?;
        let causal_mask = CausalMask::new(token_ids.len(), hidden.dtype(), &self.device)?;
        for layer in 0..self.target_hidden_state {
            let prefix = format!("model.language_model.layers.{layer}");
            let attention_names = [
                format!("{prefix}.input_layernorm.weight"),
                format!("{prefix}.self_attn.q_proj.weight"),
                format!("{prefix}.self_attn.k_proj.weight"),
                format!("{prefix}.self_attn.v_proj.weight"),
                format!("{prefix}.self_attn.o_proj.weight"),
                format!("{prefix}.self_attn.q_norm.weight"),
                format!("{prefix}.self_attn.k_norm.weight"),
            ];
            let gate_names = [
                format!("{prefix}.post_attention_layernorm.weight"),
                format!("{prefix}.mlp.gate_proj.weight"),
            ];
            hidden = with_named_group(&self.weights, &attention_names, &self.device, |weights| {
                self.attention(weights, &prefix, &hidden, &cos, &sin, &causal_mask)
            })?;

            let up_name = [format!("{prefix}.mlp.up_proj.weight")];
            let (normalized, gate) =
                with_named_group(&self.weights, &gate_names, &self.device, |weights| {
                    let normalized = core::qwen_rms_norm(
                        &hidden,
                        required(
                            weights,
                            &format!("{prefix}.post_attention_layernorm.weight"),
                        )?,
                        self.config.rms_norm_eps,
                    )?;
                    let gate = core::silu_with_reference_rounding(&linear_no_bias(
                        weights,
                        &format!("{prefix}.mlp.gate_proj.weight"),
                        &normalized,
                    )?)?;
                    Ok((normalized, gate))
                })?;
            let down_name = [format!("{prefix}.mlp.down_proj.weight")];
            let activated = with_named_group(&self.weights, &up_name, &self.device, |weights| {
                let up = linear_no_bias(weights, &up_name[0], &normalized)?;
                gate.mul(&up).map_err(Into::into)
            })?;
            let mlp = with_named_group(&self.weights, &down_name, &self.device, |weights| {
                linear_no_bias(weights, &down_name[0], &activated)
            })?;
            hidden = hidden.add(&mlp)?;
        }
        Ok(hidden)
    }

    fn attention(
        &self,
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal_mask: &CausalMask,
    ) -> Result<Tensor> {
        let normalized = core::qwen_rms_norm(
            hidden,
            required(weights, &format!("{prefix}.input_layernorm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let (batch, sequence, _) = normalized.dims3()?;
        let query = linear_no_bias(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            self.config.num_attention_heads,
            self.config.head_dim,
        ))?;
        let key = linear_no_bias(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ))?;
        let value = linear_no_bias(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &normalized,
        )?
        .reshape((
            batch,
            sequence,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ))?;
        let query = qwen_head_rms_norm(
            &query,
            required(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let key = qwen_head_rms_norm(
            &key,
            required(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
            self.config.rms_norm_eps,
        )?;
        let query = apply_rope(&query, cos, sin)?.transpose(1, 2)?;
        let key = apply_rope(&key, cos, sin)?.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        #[cfg(feature = "cuda")]
        if let (true, CausalMask::Full(causal_mask)) = (
            reference_cuda(&self.device)
                && exact_text_attention_covers(
                    batch,
                    sequence,
                    self.attention_query_chunk_size,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ),
            causal_mask,
        ) {
            let attended =
                crate::cuda::qwen::attention::text_357_exact(&query, &key, &value, causal_mask)?
                    .reshape((
                        batch,
                        sequence,
                        self.config.num_attention_heads * self.config.head_dim,
                    ))?;
            let output = linear_no_bias(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &attended,
            )?;
            return hidden.add(&output).map_err(Into::into);
        }
        let query = query.contiguous()?;
        let key = key.contiguous()?;
        let value = value.contiguous()?;
        let groups = self.config.num_attention_heads / self.config.num_key_value_heads;
        let key = repeat_kv(&key, groups)?.contiguous()?;
        let value = repeat_kv(&value, groups)?.contiguous()?;
        let key_t = key.transpose(2, 3)?;
        let mut chunks = Vec::with_capacity(sequence.div_ceil(self.attention_query_chunk_size));
        for start in (0..sequence).step_by(self.attention_query_chunk_size) {
            let length = self.attention_query_chunk_size.min(sequence - start);
            let query_chunk = query.narrow(2, start, length)?;
            let scores = scaled_attention_scores(&query_chunk, &key_t, self.config.head_dim)?;
            let scores = causal_mask.apply(&scores, start, length)?;
            let probabilities = core::qwen_softmax_last_dim(&scores)?;
            chunks.push(probabilities.matmul(&value)?);
        }
        let chunk_refs = chunks.iter().collect::<Vec<_>>();
        let attended = Tensor::cat(&chunk_refs, 2)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((
                batch,
                sequence,
                self.config.num_attention_heads * self.config.head_dim,
            ))?;
        let output = linear_no_bias(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &attended,
        )?;
        hidden.add(&output).map_err(Into::into)
    }
}

fn scaled_attention_scores(query: &Tensor, key_t: &Tensor, head_dim: usize) -> Result<Tensor> {
    query
        .matmul(key_t)?
        .affine(1. / (head_dim as f64).sqrt(), 0.)
        .map_err(Into::into)
}

fn causal_attention_mask(
    sequence: usize,
    query_start: usize,
    query_length: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    validate_causal_query_range(sequence, query_start, query_length)?;
    let elements = query_length
        .checked_mul(sequence)
        .context("causal mask element count overflow")?;
    let query_end = query_start
        .checked_add(query_length)
        .context("causal mask query range overflows usize")?;
    let mask_minimum = core::qwen_attention_mask_minimum(dtype)?;
    let mut values = Vec::with_capacity(elements);
    for query_position in query_start..query_end {
        values.extend((0..sequence).map(|key_position| {
            if key_position > query_position {
                mask_minimum
            } else {
                0.
            }
        }));
    }
    Tensor::from_vec(values, (1, 1, query_length, sequence), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn compact_causal_future_mask(
    positions: &Tensor,
    query_start: usize,
    query_length: usize,
) -> Result<Tensor> {
    let sequence = positions
        .dims1()
        .context("compact causal positions must be rank one")?;
    anyhow::ensure!(
        positions.dtype() == DType::U32,
        "compact causal positions must be U32"
    );
    validate_causal_query_range(sequence, query_start, query_length)?;
    let keys = positions.reshape((1, 1, 1, sequence))?;
    let queries =
        positions
            .narrow(0, query_start, query_length)?
            .reshape((1, 1, query_length, 1))?;
    keys.broadcast_gt(&queries)
        .context("failed to synthesize compact causal mask")
}

fn validate_causal_query_range(
    sequence: usize,
    query_start: usize,
    query_length: usize,
) -> Result<()> {
    anyhow::ensure!(sequence > 0, "causal mask sequence must be non-zero");
    anyhow::ensure!(
        query_length > 0,
        "causal mask query length must be non-zero"
    );
    let query_end = query_start
        .checked_add(query_length)
        .context("causal mask query range overflows usize")?;
    anyhow::ensure!(
        query_end <= sequence,
        "causal mask query range exceeds the sequence"
    );
    Ok(())
}

fn text_rope(
    sequence: usize,
    head_dim: usize,
    theta: f64,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inverse = qwen_inverse_frequencies(head_dim, theta, device)?.reshape((1, half))?;
    let positions = Tensor::arange(
        0u32,
        u32::try_from(sequence).context("Qwen sequence exceeds U32 RoPE indexing")?,
        device,
    )?
    .to_dtype(DType::F32)?
    .reshape((sequence, 1))?;
    let frequencies = positions.matmul(&inverse)?;
    let frequencies = Tensor::cat(&[&frequencies, &frequencies], 1)?;
    Ok((
        frequencies.cos()?.to_dtype(dtype)?,
        frequencies.sin()?.to_dtype(dtype)?,
    ))
}

/// Loads the exact F32 inverse-frequency bits produced by the pinned Qwen3-VL
/// CPU constructors before their non-persistent buffers are copied to the
/// execution device. Rust/Candle and PyTorch CPU powf differ by an ULP for the
/// released text table, so recomputing algebraically is not an exact contract.
pub(crate) fn qwen_inverse_frequencies(
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<Tensor> {
    const TEXT_BITS: [u32; 64] = [
        1065353216, 1061760040, 1058936413, 1056470439, 1052983098, 1050242642, 1047602217,
        1044217596, 1041557859, 1038748123, 1035463194, 1032881799, 1029907739, 1026719568,
        1024214207, 1021080662, 1017986398, 1015554831, 1012266501, 1009263377, 1006903432,
        1003464873, 1000550206, 998259773, 994675411, 991846595, 989391507, 985897756, 983152262,
        980522405, 977131561, 974466934, 971667455, 968376488, 965790347, 962826240, 959632207,
        957122241, 953998356, 950898404, 948462368, 945183413, 942174768, 939810486, 936381026,
        933461001, 931166358, 927590827, 924756811, 922312630, 918812458, 916061916, 913442645,
        910045568, 907376043, 904586837, 901289821, 898698926, 895744792, 892544887, 890030307,
        886916101, 883810449, 881369935,
    ];
    const VISION_BITS: [u32; 18] = [
        1065353216, 1058633676, 1052246230, 1046256949, 1040466227, 1033802166, 1027481237,
        1021571712, 1015588507, 1008981770, 1002729568, 996902449, 990720361, 984172856, 977991667,
        972249692, 965862113, 959375807,
    ];
    let bits: &[u32] = if head_dim == 128 && (theta as f32).to_bits() == 5_000_000f32.to_bits() {
        &TEXT_BITS
    } else if head_dim == 36 && (theta as f32).to_bits() == 10_000f32.to_bits() {
        &VISION_BITS
    } else {
        anyhow::bail!(
            "unverified Qwen inverse-frequency configuration head_dim={head_dim}, theta={theta}; only released text 128/5000000 and vision 36/10000 are supported"
        )
    };
    let values = bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
    Tensor::from_vec(values, bits.len(), device).map_err(Into::into)
}

fn apply_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = input.dim(candle_core::D::Minus1)?;
    let half = head_dim / 2;
    let first = input.narrow(candle_core::D::Minus1, 0, half)?;
    let second = input.narrow(candle_core::D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], candle_core::D::Minus1)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
    input
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)
        .map_err(Into::into)
}

fn qwen_head_rms_norm(input: &Tensor, weight: &Tensor, epsilon: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device())
        && crate::cuda::qwen::attention::head_rms_norm_width128_covers(input)
    {
        return crate::cuda::qwen::attention::head_rms_norm_width128(input, weight, epsilon)
            .map_err(Into::into);
    }
    core::qwen_rms_norm(input, weight, epsilon)
}

fn repeat_kv(input: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(input.clone());
    }
    let (batch, kv_heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, kv_heads, groups, sequence, head_dim))?
        .contiguous()?
        .reshape((batch, kv_heads * groups, sequence, head_dim))
        .map_err(Into::into)
}

fn with_named_group<T>(
    weights: &ModelWeights,
    names: &[String],
    device: &Device,
    f: impl FnOnce(&BTreeMap<String, Tensor>) -> Result<T>,
) -> Result<T> {
    let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
    weights.with_group(&refs, device, f)
}

fn linear_no_bias(
    weights: &BTreeMap<String, Tensor>,
    name: &str,
    input: &Tensor,
) -> Result<Tensor> {
    let weight = required(weights, name)?;
    Linear::new(weight.clone(), None)
        .forward(&input.to_dtype(weight.dtype())?)
        .with_context(|| format!("linear projection {name}"))
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing text encoder tensor {name}"))
}

#[cfg(test)]
mod tests {

    #[test]
    fn exact_text_attention_covers_only_the_official_coffee_profile() {
        assert!(super::exact_text_attention_covers(1, 357, 357, 64, 8, 128));
        for (batch, sequence, chunk, heads, kv, head_dim, what) in [
            (1, 1, 32, 64, 8, 128, "short prompt with the default chunk"),
            (
                1,
                357,
                32,
                64,
                8,
                128,
                "official rows with the default chunk",
            ),
            (1, 356, 356, 64, 8, 128, "one row short"),
            (2, 357, 357, 64, 8, 128, "batched"),
            (1, 357, 357, 32, 8, 128, "other head count"),
            (1, 357, 357, 64, 4, 128, "other key/value head count"),
            (1, 357, 357, 64, 8, 64, "other head dimension"),
        ] {
            assert!(
                !super::exact_text_attention_covers(batch, sequence, chunk, heads, kv, head_dim),
                "{what} must take the portable path"
            );
        }
    }
    use super::*;
    use candle_core::{Shape, safetensors};
    use serde_json::json;
    use std::collections::HashMap;

    fn ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn patterned(shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| ((index * 17 % 47) as f32 - 23.) / 29.)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    fn score_scaled_attention_reference(query: &Tensor, key_t: &Tensor, head_dim: usize) -> Tensor {
        query
            .matmul(key_t)
            .unwrap()
            .affine(1. / (head_dim as f64).sqrt(), 0.)
            .unwrap()
    }

    #[test]
    fn postscaled_f32_scores_match_reference_scaling() {
        let head_dim = 128;
        let query = patterned((1, 2, 3, head_dim)).affine(0.7, 0.13).unwrap();
        let key_t = patterned((1, 2, head_dim, 5)).affine(-0.4, 0.07).unwrap();
        let expected = score_scaled_attention_reference(&query, &key_t, head_dim);
        let actual = scaled_attention_scores(&query, &key_t, head_dim).unwrap();
        let expected_values = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_values = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual_values, expected_values);
    }

    #[test]
    fn qwen_inverse_frequencies_match_pinned_cpu_constructor_bits() {
        let expected = [
            1065353216u32,
            1061760040,
            1058936413,
            1056470439,
            1052983098,
            1050242642,
            1047602217,
            1044217596,
            1041557859,
            1038748123,
            1035463194,
            1032881799,
            1029907739,
            1026719568,
            1024214207,
            1021080662,
            1017986398,
            1015554831,
            1012266501,
            1009263377,
            1006903432,
            1003464873,
            1000550206,
            998259773,
            994675411,
            991846595,
            989391507,
            985897756,
            983152262,
            980522405,
            977131561,
            974466934,
            971667455,
            968376488,
            965790347,
            962826240,
            959632207,
            957122241,
            953998356,
            950898404,
            948462368,
            945183413,
            942174768,
            939810486,
            936381026,
            933461001,
            931166358,
            927590827,
            924756811,
            922312630,
            918812458,
            916061916,
            913442645,
            910045568,
            907376043,
            904586837,
            901289821,
            898698926,
            895744792,
            892544887,
            890030307,
            886916101,
            883810449,
            881369935,
        ];
        let actual = qwen_inverse_frequencies(128, 5_000_000., &Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn causal_mask_covers_only_the_requested_query_chunk() {
        let mask = causal_attention_mask(5, 2, 2, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
        let values = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&values[..3], &[0., 0., 0.]);
        assert_eq!(values[3], f32::MIN);
        assert_eq!(values[4], f32::MIN);
        assert_eq!(&values[5..9], &[0., 0., 0., 0.]);
        assert_eq!(values[9], f32::MIN);
    }

    #[test]
    fn causal_mask_rejects_invalid_query_ranges() {
        for (sequence, start, length, message) in [
            (0, 0, 1, "sequence must be non-zero"),
            (5, 0, 0, "query length must be non-zero"),
            (5, 4, 2, "query range exceeds the sequence"),
            (5, usize::MAX, 2, "query range overflows usize"),
        ] {
            let error = causal_attention_mask(sequence, start, length, DType::F32, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains(message), "unexpected error: {error}");
        }
    }

    #[test]
    fn causal_mask_policy_caches_small_prompts_and_compacts_large_ones() {
        assert_eq!(
            full_causal_mask_construction_bytes(2_048, DType::BF16).unwrap(),
            24 * 1024 * 1024
        );
        assert!(
            full_causal_mask_construction_bytes(1_182, DType::BF16).unwrap()
                <= MAX_CACHED_CAUSAL_MASK_BYTES
        );
        assert!(
            full_causal_mask_construction_bytes(1_183, DType::BF16).unwrap()
                > MAX_CACHED_CAUSAL_MASK_BYTES
        );
        assert!(
            full_causal_mask_construction_bytes(1_448, DType::F32).unwrap()
                <= MAX_CACHED_CAUSAL_MASK_BYTES
        );
        assert!(
            full_causal_mask_construction_bytes(1_449, DType::F32).unwrap()
                > MAX_CACHED_CAUSAL_MASK_BYTES
        );
        assert_eq!(
            full_causal_mask_construction_bytes(256, DType::BF16).unwrap(),
            384 * 1024
        );

        let common = CausalMask::new(256, DType::BF16, &Device::Cpu).unwrap();
        let common = match &common {
            CausalMask::Full(mask) => mask,
            CausalMask::Compact { .. } => panic!("common BF16 prompt must retain the fast path"),
        };
        assert_eq!(common.dims(), &[1, 1, 256, 256]);
        assert_eq!(common.dtype(), DType::BF16);
        assert_eq!(
            common.elem_count() * common.dtype().size_in_bytes(),
            128 * 1024
        );

        let small = CausalMask::new(16, DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(&small, CausalMask::Full(_)));
        let small_scores = patterned((1, 2, 3, 16));
        let small_chunk = small.apply(&small_scores, 4, 3).unwrap();
        let expected_mask = causal_attention_mask(16, 4, 3, DType::F32, &Device::Cpu).unwrap();
        let expected = small_scores.broadcast_add(&expected_mask).unwrap();
        assert_eq!(
            small_chunk.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let large = CausalMask::new(2_048, DType::F32, &Device::Cpu).unwrap();
        let (positions, negative_infinity) = match &large {
            CausalMask::Compact {
                positions,
                negative_infinity,
            } => (positions, negative_infinity),
            CausalMask::Full(_) => panic!("large F32 mask must use compact positions"),
        };
        assert_eq!(positions.dims(), &[2_048]);
        assert_eq!(positions.dtype(), DType::U32);
        assert!(negative_infinity.dims().is_empty());
        assert_eq!(negative_infinity.dtype(), DType::F32);
        assert_eq!(
            positions.elem_count() * positions.dtype().size_in_bytes()
                + negative_infinity.elem_count() * negative_infinity.dtype().size_in_bytes(),
            8_196
        );

        let predicate = compact_causal_future_mask(positions, 17, 2).unwrap();
        assert_eq!(predicate.dims(), &[1, 1, 2, 2_048]);
        assert_eq!(predicate.dtype(), DType::U8);
        assert_eq!(predicate.elem_count(), 2 * 2_048);
        let predicate = predicate.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert!(predicate[..18].iter().all(|&value| value == 0));
        assert!(predicate[18..2_048].iter().all(|&value| value == 1));
        assert!(predicate[2_048..2_048 + 19].iter().all(|&value| value == 0));
        assert!(predicate[2_048 + 19..].iter().all(|&value| value == 1));

        let large_scores = patterned((1, 2, 2, 2_048));
        let actual = large.apply(&large_scores, 17, 2).unwrap();
        let expected_mask = causal_attention_mask(2_048, 17, 2, DType::F32, &Device::Cpu).unwrap();
        let expected = large_scores.broadcast_add(&expected_mask).unwrap();
        assert_eq!(actual.dims(), &[1, 2, 2, 2_048]);
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let error = CausalMask::new(usize::MAX, DType::F32, &Device::Cpu)
            .err()
            .expect("overflowing mask dimensions must be rejected");
        assert!(error.to_string().contains("element count overflow"));
    }

    #[test]
    fn compact_causal_mask_preserves_bf16_scores_on_cpu() {
        let sequence = 2_048;
        let start = 1_017;
        let length = 3;
        let scores = patterned((1, 2, length, sequence))
            .to_dtype(DType::BF16)
            .unwrap();
        let mask = CausalMask::new(sequence, DType::BF16, &Device::Cpu).unwrap();
        assert!(matches!(&mask, CausalMask::Compact { .. }));
        let actual = mask.apply(&scores, start, length).unwrap();
        let reference_mask =
            causal_attention_mask(sequence, start, length, DType::BF16, &Device::Cpu).unwrap();
        let expected = scores.broadcast_add(&reference_mask).unwrap();
        assert_eq!(actual.dims(), scores.dims());
        assert_eq!(actual.dtype(), DType::BF16);
        assert_eq!(
            actual
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn cpu_encoder_streams_and_rejects_invalid_tokens_before_payload_access() {
        let hidden = 4;
        let head_dim = 128;
        let heads = 2;
        let kv_heads = 1;
        let intermediate = 5;
        let prefix = "model.language_model.layers.0";
        let mut tensors = HashMap::from([(
            "model.language_model.embed_tokens.weight".to_owned(),
            ones((6, hidden)),
        )]);
        for name in ["input_layernorm", "post_attention_layernorm"] {
            tensors.insert(format!("{prefix}.{name}.weight"), ones(hidden));
        }
        for (name, rows) in [
            ("q_proj", heads * head_dim),
            ("k_proj", kv_heads * head_dim),
            ("v_proj", kv_heads * head_dim),
            ("o_proj", hidden),
        ] {
            let columns = if name == "o_proj" {
                heads * head_dim
            } else {
                hidden
            };
            tensors.insert(
                format!("{prefix}.self_attn.{name}.weight"),
                ones((rows, columns)),
            );
        }
        tensors.insert(format!("{prefix}.self_attn.q_norm.weight"), ones(head_dim));
        tensors.insert(format!("{prefix}.self_attn.k_norm.weight"), ones(head_dim));
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            ones((intermediate, hidden)),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            ones((intermediate, hidden)),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            ones((hidden, intermediate)),
        );

        let dir = tempfile::tempdir().unwrap();
        safetensors::save(&tensors, dir.path().join("weights.safetensors")).unwrap();
        let map = tensors
            .keys()
            .map(|name| (name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": tensors.values().map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes()).sum::<usize>()},
                "weight_map": map
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let encoder = StreamedTextEncoder::new(
            weights,
            TextEncoderConfig {
                head_dim,
                vocab_size: 6,
                hidden_size: hidden,
                intermediate_size: intermediate,
                num_hidden_layers: 2,
                num_attention_heads: heads,
                num_key_value_heads: kv_heads,
                rms_norm_eps: 1e-6,
                rope_theta: 5_000_000.,
            },
            Device::Cpu,
            1,
            1,
        )
        .unwrap();
        let access_before = encoder.access_stats();
        let error = encoder.encode_token_ids(&[6]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("token ID exceeds text encoder vocabulary"),
            "unexpected error: {error:#}"
        );
        assert_eq!(encoder.access_stats(), access_before);
        let output = encoder.encode_token_ids(&[2, 3, 2]).unwrap();
        assert_eq!(output.dims(), &[1, 3, hidden]);
        assert!(
            output
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }
}
