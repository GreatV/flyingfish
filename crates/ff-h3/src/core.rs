use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::{Linear, Module, ops};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use std::{collections::BTreeMap, fmt, num::NonZeroUsize, str::FromStr};

/// Whether this device may run the kernels compiled from this repository.
///
/// Their numerics are fixed by their own source and the one `compute_80` PTX
/// it is compiled to, so the question is only whether the hardware can execute
/// those instructions. Every dispatch of a transcribed kernel goes through
/// here.
#[cfg(feature = "cuda")]
fn tuned_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::tuned_kernels_available(device)
}

/// Whether the vendor libraries this host provides are the reference ones.
///
/// cuBLASLt and cuDNN choose different kernels per architecture and per library
/// build, so an operator that reproduces a recorded vendor-library result is
/// making a claim about one host. Compositions that exist to match PyTorch's
/// operator order go through here rather than through [`tuned_cuda`]: they are
/// parity paths, and on this tree's own measurements they are not the faster
/// ones, so declining them elsewhere costs evidence and not speed.
#[cfg(feature = "cuda")]
fn reference_cuda(device: &candle_core::Device) -> bool {
    crate::cuda::profile::reference_libraries_available(device)
}

pub const MODALITY_COUNT: usize = 3;

pub const DEFAULT_FFN_TOKEN_CHUNK_SIZE: usize = 256;

pub const DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE: usize = 32;

pub const DEFAULT_ATTENTION_QUERY_CHUNK_SIZE: usize = 32;

/// The projection chunk to use when FlashAttention is selected.
///
/// On the exact path the projection chunk is a pure memory bound and 32 rows
/// is the conservative choice. Under FlashAttention it is also the query span
/// of each attention call, because the query chunk bounds a score matrix that
/// FlashAttention never materializes. That makes the span decide the kernel's
/// arithmetic intensity: roughly one FLOP per byte per query row, against an
/// RTX 4090 ridge point near 164. At the released 73,743-row T2VA shape a
/// 32-row span measures 16.1 TFLOP/s and a 4,096-row span 157.7, against 162.9
/// for a single whole-sequence call — so this recovers 97% of the achievable
/// rate while holding the query and its normalization buffers to tens of
/// megabytes. Admission still models and can refuse it.
pub const DEFAULT_FLASH_ATTENTION_PROJECTION_CHUNK_SIZE: usize = 4_096;

/// Largest key width covered by PyTorch's CUDA persistent-softmax dispatch.
///
/// The persistent kernel holds a whole softmax row in registers -- at 2048
/// elements across a 32-lane warp that is 64 floats per thread -- so upstream
/// stops dispatching it here and switches to a regular register kernel. This is
/// that kernel's coverage boundary, not a hardware limit and not a choice made
/// here.
pub const CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS: usize = 2048;

/// Largest key width the transcribed softmax pair covers exactly.
///
/// Both kernels come from the same upstream commit and the same vendored `.cu`
/// file: persistent through 2048, regular register above it. Widths are
/// dispatched between them the way upstream dispatches them, so the exact range
/// is the union rather than either kernel's own ceiling.
pub const CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS: usize = 9216;
pub const QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS: usize = CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
pub(crate) const QWEN_RMS_NORM_BACKEND: &str =
    "qwen3-vl-eager-composite-rmsnorm-f32-rsqrt-cast-input-dtype-weight-mul-v1";
pub(crate) const QWEN_ROPE_BACKEND: &str =
    "qwen3-vl-pinned-cpu-f32-invfreq-bits-device-f32-position-math-cos-sin-cast-v1";
pub(crate) const QWEN_ATTENTION_SCORE_BACKEND: &str =
    "qwen3-vl-eager-input-dtype-matmul-then-scale-v1";
pub(crate) const QWEN_ATTENTION_MASK_BACKEND: &str =
    "qwen3-vl-eager-causal-add-finfo-input-dtype-min-v1";
pub(crate) const QWEN_SILU_BACKEND: &str = "qwen3-vl-f32-silu-single-input-dtype-cast-v1";
pub const QWEN_GELU_BACKEND: &str = "qwen3-vl-f32-exact-erf-gelu-single-input-dtype-cast-v1";
pub(crate) const QWEN_SOFTMAX_BACKEND: &str =
    "qwen3-vl-eager-f32-persistent-1..2048+regular-register-2049..9216-v1";

pub(crate) fn qwen_attention_mask_minimum(dtype: candle_core::DType) -> Result<f32> {
    match dtype {
        candle_core::DType::BF16 => Ok(f32::from_bits(0xff7f_0000)),
        candle_core::DType::F16 => Ok(-65_504.0),
        candle_core::DType::F32 => Ok(f32::MIN),
        _ => anyhow::bail!(
            "{QWEN_ATTENTION_MASK_BACKEND} does not support Qwen mask dtype {dtype:?}"
        ),
    }
}

pub const H3_FLASH_ATTENTION_BACKEND: &str = "candle-flash-attn-0.11.0/main-blocks+token-refiner";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionKeyChunkSize(NonZeroUsize);

impl AttentionKeyChunkSize {
    pub const fn new(rows: usize) -> Option<Self> {
        match NonZeroUsize::new(rows) {
            Some(rows) => Some(Self(rows)),
            None => None,
        }
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl fmt::Display for AttentionKeyChunkSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

impl FromStr for AttentionKeyChunkSize {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let rows = value
            .parse::<usize>()
            .map_err(|error| format!("invalid attention key chunk size: {error}"))?;
        Self::new(rows).ok_or_else(|| "attention key chunk size must be positive".to_owned())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AttentionKeyChunkPolicy {
    #[default]
    Full,
    Chunked(AttentionKeyChunkSize),
}

impl AttentionKeyChunkPolicy {
    pub const fn chunked(rows: usize) -> Option<Self> {
        match AttentionKeyChunkSize::new(rows) {
            Some(rows) => Some(Self::Chunked(rows)),
            None => None,
        }
    }

    pub const fn configured_chunk_size(self) -> Option<NonZeroUsize> {
        match self {
            Self::Full => None,
            Self::Chunked(rows) => Some(rows.0),
        }
    }

    pub fn effective_chunk_size(self, total_rows: NonZeroUsize) -> NonZeroUsize {
        match self {
            Self::Full => total_rows,
            Self::Chunked(rows) => rows.0.min(total_rows),
        }
    }

    pub const fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

impl fmt::Display for AttentionKeyChunkPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => formatter.write_str("full"),
            Self::Chunked(rows) => rows.fmt(formatter),
        }
    }
}

impl Serialize for AttentionKeyChunkPolicy {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Full => serializer.serialize_str("full"),
            Self::Chunked(rows) => serializer.serialize_u64(rows.get() as u64),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AttentionKeyChunkPolicyRepr {
    Name(String),
    Rows(u64),
}

impl<'de> Deserialize<'de> for AttentionKeyChunkPolicy {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match AttentionKeyChunkPolicyRepr::deserialize(deserializer)? {
            AttentionKeyChunkPolicyRepr::Name(name) if name == "full" => Ok(Self::Full),
            AttentionKeyChunkPolicyRepr::Name(name) => Err(D::Error::custom(format!(
                "unknown attention key chunk policy {name:?}; expected \"full\" or positive rows"
            ))),
            AttentionKeyChunkPolicyRepr::Rows(rows) => {
                let rows = usize::try_from(rows)
                    .map_err(|_| D::Error::custom("attention key chunk size exceeds usize"))?;
                Self::chunked(rows)
                    .ok_or_else(|| D::Error::custom("attention key chunk size must be positive"))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionChunking {
    pub projection_chunk_size: NonZeroUsize,
    pub query_chunk_size: NonZeroUsize,
    pub key: AttentionKeyChunkPolicy,
}

impl Default for AttentionChunking {
    fn default() -> Self {
        Self {
            projection_chunk_size: NonZeroUsize::new(DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE)
                .expect("default attention projection chunk size is non-zero"),
            query_chunk_size: NonZeroUsize::new(DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)
                .expect("default attention query chunk size is non-zero"),
            key: AttentionKeyChunkPolicy::Full,
        }
    }
}

pub struct AdaLnModulation {
    pub shift_attention: Tensor,
    pub scale_attention: Tensor,
    pub gate_attention: Tensor,
    pub shift_feed_forward: Tensor,
    pub scale_feed_forward: Tensor,
    pub gate_feed_forward: Tensor,
}

/// The per-row tables every transformer-block stage shares.
#[derive(Clone, Copy)]
pub struct BlockContext<'a> {
    pub adaln_indices: &'a Tensor,
    pub rotary_cos: &'a Tensor,
    pub rotary_sin: &'a Tensor,
}

/// Head geometry and normalization epsilons of one attention configuration.
#[derive(Clone, Copy, Debug)]
pub struct AttentionParams {
    pub heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    pub qk_norm_eps: f64,
}

pub fn adaln(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    timestep_embedding: &Tensor,
    hidden_size: usize,
) -> Result<AdaLnModulation> {
    let weight = required(weights, &format!("{prefix}.adaln_proj.linear.weight"))?;
    let bias = optional(weights, &format!("{prefix}.adaln_proj.linear.bias"));
    let activated = ops::silu(timestep_embedding)?.to_dtype(weight.dtype())?;
    let projected = linear_with_reference_bias(&activated, weight, bias)?;
    let timesteps = timestep_embedding.dim(0)?;
    let expected = 6 * hidden_size * MODALITY_COUNT;
    anyhow::ensure!(
        projected.dim(1)? == expected,
        "AdaLN projection has {} outputs, expected {expected}",
        projected.dim(1)?
    );
    let projected = projected.reshape((timesteps * MODALITY_COUNT, 6 * hidden_size))?;
    let part = |index| {
        projected
            .narrow(1, index * hidden_size, hidden_size)?
            .contiguous()
    };
    Ok(AdaLnModulation {
        shift_attention: part(0)?,
        scale_attention: part(1)?,
        gate_attention: part(2)?,
        shift_feed_forward: part(3)?,
        scale_feed_forward: part(4)?,
        gate_feed_forward: part(5)?,
    })
}

/// Matches PyTorch's fused CUDA BF16 linear+bias rounding boundary.
///
/// Candle's `Linear` materializes a BF16 matmul result before adding the bias,
/// introducing a second BF16 rounding that `torch.nn.functional.linear` does
/// not have when cuBLASLt fuses the bias epilogue. The official H3 checkpoint
/// uses biased BF16 projections for context, AdaLN, and output modulation. The
/// verified CUDA path uses the same cuBLASLt compute type and bias epilogue.
/// Unsupported CUDA BF16 shapes fail instead of selecting an approximation;
/// non-CUDA, unbiased, and non-BF16 projections retain Candle's existing path.
pub(crate) fn linear_with_reference_bias(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let input = input.to_dtype(weight.dtype())?;
    #[cfg(feature = "cuda")]
    if reference_cuda(input.device())
        && weight.dtype() == candle_core::DType::BF16
        && let Some(bias) = bias
    {
        return crate::cuda::linear::linear(&input, weight, bias).map_err(Into::into);
    }
    Linear::new(weight.clone(), bias.cloned())
        .forward(&input)
        .map_err(Into::into)
}

pub fn attention_with_projection_chunks(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    modulation: &AdaLnModulation,
    context: &BlockContext<'_>,
    params: AttentionParams,
    chunking: AttentionChunking,
) -> Result<Tensor> {
    let BlockContext {
        adaln_indices,
        rotary_cos,
        rotary_sin,
    } = *context;
    let AttentionParams {
        heads,
        head_dim,
        norm_eps,
        qk_norm_eps,
    } = params;
    let (batch, sequence, _) = hidden_states
        .dims3()
        .context("hidden states must be [batch, sequence, hidden]")?;
    anyhow::ensure!(sequence > 0, "hidden-state sequence must be non-empty");
    anyhow::ensure!(
        adaln_indices.dim(0)? == sequence,
        "AdaLN indices must have one entry per sequence row"
    );
    anyhow::ensure!(
        rotary_cos.dims() == rotary_sin.dims(),
        "rotary cosine/sine shapes differ"
    );
    anyhow::ensure!(
        rotary_cos.dim(0)? == sequence,
        "rotary tables must have one row per sequence row"
    );
    let norm_weight = required(weights, &format!("{prefix}.norm1.weight"))?;
    let projection_chunk_size = chunking.projection_chunk_size.get();
    let score_query_chunk_size = chunking.query_chunk_size.get();

    let key_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_k.weight"))?.clone(),
        None,
    );
    let value_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_v.weight"))?.clone(),
        None,
    );
    let key_norm_weight = required(weights, &format!("{prefix}.attn.norm_k.weight"))?;
    let projection_chunk_count = sequence.div_ceil(projection_chunk_size);
    let mut key_chunks = Vec::with_capacity(projection_chunk_count);
    let mut value_chunks = Vec::with_capacity(projection_chunk_count);
    for start in (0..sequence).step_by(projection_chunk_size) {
        let length = projection_chunk_size.min(sequence - start);
        let (_, normalized, _) = normalized_attention_chunk(
            hidden_states,
            norm_weight,
            modulation,
            adaln_indices,
            start,
            length,
            norm_eps,
        )?;
        let key = key_projection
            .forward(&normalized)
            .with_context(|| format!("{prefix}.attn.to_k"))?
            .reshape((batch, length, heads, head_dim))?;
        let key = rms_norm(&key, key_norm_weight, qk_norm_eps)?;
        let chunk_cos = rotary_cos.narrow(0, start, length)?;
        let chunk_sin = rotary_sin.narrow(0, start, length)?;
        let key = apply_rotary(&key, &chunk_cos, &chunk_sin)?
            .transpose(1, 2)?
            .transpose(2, 3)?
            .contiguous()?;
        key_chunks.push(key);

        let value = value_projection
            .forward(&normalized)
            .with_context(|| format!("{prefix}.attn.to_v"))?
            .reshape((batch, length, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        value_chunks.push(value);
    }
    let key = concatenate_chunks(key_chunks, 3)?;
    let value = concatenate_chunks(value_chunks, 2)?;

    let query_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_q.weight"))?.clone(),
        None,
    );
    let query_norm_weight = required(weights, &format!("{prefix}.attn.norm_q.weight"))?;
    let output_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
        None,
    );
    let mut chunks = Vec::with_capacity(projection_chunk_count);
    for projection_start in (0..sequence).step_by(projection_chunk_size) {
        let projection_length = projection_chunk_size.min(sequence - projection_start);
        let (hidden_chunk, normalized, indices) = normalized_attention_chunk(
            hidden_states,
            norm_weight,
            modulation,
            adaln_indices,
            projection_start,
            projection_length,
            norm_eps,
        )?;
        let query = query_projection
            .forward(&normalized)
            .with_context(|| format!("{prefix}.attn.to_q"))?
            .reshape((batch, projection_length, heads, head_dim))?;
        let query = rms_norm(&query, query_norm_weight, qk_norm_eps)?;
        let chunk_cos = rotary_cos.narrow(0, projection_start, projection_length)?;
        let chunk_sin = rotary_sin.narrow(0, projection_start, projection_length)?;
        let query = apply_rotary(&query, &chunk_cos, &chunk_sin)?;

        let score_chunk_count = projection_length.div_ceil(score_query_chunk_size);
        let mut attended_chunks = Vec::with_capacity(score_chunk_count);
        for score_offset in (0..projection_length).step_by(score_query_chunk_size) {
            let score_length = score_query_chunk_size.min(projection_length - score_offset);
            let query_chunk = query.narrow(1, score_offset, score_length)?.contiguous()?;
            attended_chunks.push(attend_prepared_with_key_chunks(
                &query_chunk,
                &key,
                &value,
                chunking.key,
            )?);
        }
        let attended = concatenate_chunks(attended_chunks, 1)?;
        let output = output_projection.forward(&attended)?;
        let gate = modulation.gate_attention.index_select(&indices, 0)?;
        chunks.push(hidden_chunk.add(&gate.broadcast_mul(&output)?)?);
    }
    concatenate_chunks(chunks, 1)
}

#[cfg(feature = "flash-attn")]
pub fn attention_flash_with_projection_chunks(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    modulation: &AdaLnModulation,
    context: &BlockContext<'_>,
    params: AttentionParams,
    chunking: AttentionChunking,
) -> Result<Tensor> {
    let BlockContext {
        adaln_indices,
        rotary_cos,
        rotary_sin,
    } = *context;
    let AttentionParams {
        heads,
        head_dim,
        norm_eps,
        qk_norm_eps,
    } = params;
    let (batch, sequence, _) = hidden_states
        .dims3()
        .context("hidden states must be [batch, sequence, hidden]")?;
    anyhow::ensure!(sequence > 0, "hidden-state sequence must be non-empty");
    anyhow::ensure!(
        adaln_indices.dim(0)? == sequence,
        "AdaLN indices must have one entry per sequence row"
    );
    anyhow::ensure!(
        rotary_cos.dims() == rotary_sin.dims(),
        "rotary cosine/sine shapes differ"
    );
    anyhow::ensure!(
        rotary_cos.dim(0)? == sequence,
        "rotary tables must have one row per sequence row"
    );
    anyhow::ensure!(
        hidden_states.device().is_cuda(),
        "FlashAttention requires CUDA tensors"
    );
    crate::cuda::validate_flash_attention_device(hidden_states.device())?;
    anyhow::ensure!(
        matches!(
            hidden_states.dtype(),
            candle_core::DType::F16 | candle_core::DType::BF16
        ),
        "FlashAttention requires F16 or BF16 hidden states"
    );
    anyhow::ensure!(
        head_dim <= 512 && head_dim.is_multiple_of(8),
        "FlashAttention head dimension must be a multiple of 8 and at most 512"
    );

    let projection_chunk_size = chunking.projection_chunk_size.get();
    let norm_weight = required(weights, &format!("{prefix}.norm1.weight"))?;
    let key_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_k.weight"))?.clone(),
        None,
    );
    let value_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_v.weight"))?.clone(),
        None,
    );
    let key_norm_weight = required(weights, &format!("{prefix}.attn.norm_k.weight"))?;
    let projection_chunk_count = sequence.div_ceil(projection_chunk_size);
    let mut key_chunks = Vec::with_capacity(projection_chunk_count);
    let mut value_chunks = Vec::with_capacity(projection_chunk_count);

    for start in (0..sequence).step_by(projection_chunk_size) {
        let length = projection_chunk_size.min(sequence - start);
        let (_, normalized, _) = normalized_attention_chunk(
            hidden_states,
            norm_weight,
            modulation,
            adaln_indices,
            start,
            length,
            norm_eps,
        )?;
        let chunk_cos = rotary_cos.narrow(0, start, length)?;
        let chunk_sin = rotary_sin.narrow(0, start, length)?;

        let key = key_projection
            .forward(&normalized)
            .with_context(|| format!("{prefix}.attn.to_k"))?
            .reshape((batch, length, heads, head_dim))?;
        key_chunks.push(
            apply_rotary(
                &rms_norm(&key, key_norm_weight, qk_norm_eps)?,
                &chunk_cos,
                &chunk_sin,
            )?
            .contiguous()?,
        );

        value_chunks.push(
            value_projection
                .forward(&normalized)
                .with_context(|| format!("{prefix}.attn.to_v"))?
                .reshape((batch, length, heads, head_dim))?
                .contiguous()?,
        );
    }

    let key = concatenate_chunks(key_chunks, 1)?.contiguous()?;
    let value = concatenate_chunks(value_chunks, 1)?.contiguous()?;
    let query_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_q.weight"))?.clone(),
        None,
    );
    let query_norm_weight = required(weights, &format!("{prefix}.attn.norm_q.weight"))?;
    let output_projection = Linear::new(
        required(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
        None,
    );
    let mut output_chunks = Vec::with_capacity(projection_chunk_count);
    for projection_start in (0..sequence).step_by(projection_chunk_size) {
        let projection_length = projection_chunk_size.min(sequence - projection_start);
        let (hidden_chunk, normalized, indices) = normalized_attention_chunk(
            hidden_states,
            norm_weight,
            modulation,
            adaln_indices,
            projection_start,
            projection_length,
            norm_eps,
        )?;
        let query = query_projection
            .forward(&normalized)
            .with_context(|| format!("{prefix}.attn.to_q"))?
            .reshape((batch, projection_length, heads, head_dim))?;
        let query = rms_norm(&query, query_norm_weight, qk_norm_eps)?;
        let chunk_cos = rotary_cos.narrow(0, projection_start, projection_length)?;
        let chunk_sin = rotary_sin.narrow(0, projection_start, projection_length)?;
        let query = apply_rotary(&query, &chunk_cos, &chunk_sin)?.contiguous()?;

        let attended = candle_flash_attn::flash_attn(
            &query,
            &key,
            &value,
            1.0 / (head_dim as f32).sqrt(),
            false,
        )
        .context("CUDA FlashAttention failed")?
        .reshape((batch, projection_length, heads * head_dim))?;
        let output = output_projection.forward(&attended)?;
        let gate = modulation.gate_attention.index_select(&indices, 0)?;
        output_chunks.push(hidden_chunk.add(&gate.broadcast_mul(&output)?)?);
    }
    concatenate_chunks(output_chunks, 1)
}

pub fn refiner_attention(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    params: AttentionParams,
    query_chunk_size: NonZeroUsize,
) -> Result<Tensor> {
    let AttentionParams {
        heads,
        head_dim,
        norm_eps,
        qk_norm_eps,
    } = params;
    let normalized = rms_norm(
        hidden_states,
        required(weights, &format!("{prefix}.norm1.weight"))?,
        norm_eps,
    )?;
    let (batch, sequence, _) = normalized.dims3()?;
    let project = |name: &str| -> Result<Tensor> {
        let weight = required(weights, &format!("{prefix}.attn.{name}.weight"))?;
        Linear::new(weight.clone(), None)
            .forward(&normalized)?
            .reshape((batch, sequence, heads, head_dim))
            .map_err(Into::into)
    };
    let query = rms_norm(
        &project("to_q")?,
        required(weights, &format!("{prefix}.attn.norm_q.weight"))?,
        qk_norm_eps,
    )?;
    let key = rms_norm(
        &project("to_k")?,
        required(weights, &format!("{prefix}.attn.norm_k.weight"))?,
        qk_norm_eps,
    )?;
    let value = project("to_v")?;
    let attended = scaled_dot_product(&query, &key, &value, query_chunk_size)?;
    let output = Linear::new(
        required(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
        None,
    )
    .forward(&attended)?;
    hidden_states.add(&output).map_err(Into::into)
}

#[cfg(feature = "flash-attn")]
pub fn refiner_attention_flash(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    params: AttentionParams,
    _query_chunk_size: NonZeroUsize,
) -> Result<Tensor> {
    let AttentionParams {
        heads,
        head_dim,
        norm_eps,
        qk_norm_eps,
    } = params;
    anyhow::ensure!(
        hidden_states.device().is_cuda(),
        "refiner FlashAttention requires CUDA tensors"
    );
    crate::cuda::validate_flash_attention_device(hidden_states.device())?;
    anyhow::ensure!(
        matches!(
            hidden_states.dtype(),
            candle_core::DType::F16 | candle_core::DType::BF16
        ),
        "refiner FlashAttention requires F16 or BF16 hidden states"
    );
    anyhow::ensure!(
        head_dim <= 512 && head_dim.is_multiple_of(8),
        "refiner FlashAttention head dimension must be a multiple of 8 and at most 512"
    );
    let normalized = rms_norm(
        hidden_states,
        required(weights, &format!("{prefix}.norm1.weight"))?,
        norm_eps,
    )?;
    let (batch, sequence, _) = normalized.dims3()?;
    let project = |name: &str| -> Result<Tensor> {
        let weight = required(weights, &format!("{prefix}.attn.{name}.weight"))?;
        Linear::new(weight.clone(), None)
            .forward(&normalized)?
            .reshape((batch, sequence, heads, head_dim))?
            .contiguous()
            .map_err(Into::into)
    };
    let query = rms_norm(
        &project("to_q")?,
        required(weights, &format!("{prefix}.attn.norm_q.weight"))?,
        qk_norm_eps,
    )?
    .contiguous()?;
    let key = rms_norm(
        &project("to_k")?,
        required(weights, &format!("{prefix}.attn.norm_k.weight"))?,
        qk_norm_eps,
    )?
    .contiguous()?;
    let value = project("to_v")?;
    let attended =
        candle_flash_attn::flash_attn(&query, &key, &value, 1.0 / (head_dim as f32).sqrt(), false)
            .context("CUDA refiner FlashAttention failed")?
            .reshape((batch, sequence, heads * head_dim))?;
    let output = Linear::new(
        required(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
        None,
    )
    .forward(&attended)?;
    hidden_states.add(&output).map_err(Into::into)
}

pub fn refiner_feed_forward_chunked(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    token_chunk_size: NonZeroUsize,
    norm_eps: f64,
) -> Result<Tensor> {
    let norm_weight = required(weights, &format!("{prefix}.norm2.weight"))?;
    map_sequence_chunks(hidden_states, token_chunk_size, |_, _, hidden_chunk| {
        let normalized = rms_norm(hidden_chunk, norm_weight, norm_eps)?;
        let output = swiglu(weights, prefix, &normalized)?;
        hidden_chunk.add(&output).map_err(Into::into)
    })
}

pub fn feed_forward_chunked(
    weights: &BTreeMap<String, Tensor>,
    prefix: &str,
    hidden_states: &Tensor,
    modulation: &AdaLnModulation,
    adaln_indices: &Tensor,
    token_chunk_size: NonZeroUsize,
    norm_eps: f64,
) -> Result<Tensor> {
    let sequence = hidden_states
        .dim(1)
        .context("hidden states must be [batch, sequence, hidden]")?;
    anyhow::ensure!(
        adaln_indices.dim(0)? == sequence,
        "AdaLN indices must have one entry per sequence row"
    );
    let norm_weight = required(weights, &format!("{prefix}.norm2.weight"))?;
    map_sequence_chunks(
        hidden_states,
        token_chunk_size,
        |start, length, hidden_chunk| {
            let indices = adaln_indices.narrow(0, start, length)?;
            let normalized = rms_norm(hidden_chunk, norm_weight, norm_eps)?;
            let normalized = modulate(
                &normalized,
                &modulation.shift_feed_forward,
                &modulation.scale_feed_forward,
                &indices,
            )?;
            let output = swiglu(weights, prefix, &normalized)?;
            let gate = modulation.gate_feed_forward.index_select(&indices, 0)?;
            hidden_chunk
                .add(&gate.broadcast_mul(&output)?)
                .map_err(Into::into)
        },
    )
}

fn swiglu(weights: &BTreeMap<String, Tensor>, prefix: &str, normalized: &Tensor) -> Result<Tensor> {
    let input_weight = required(weights, &format!("{prefix}.ff.net.0.proj.weight"))?;
    let projected = Linear::new(input_weight.clone(), None).forward(normalized)?;
    let doubled = projected.dim(candle_core::D::Minus1)?;
    anyhow::ensure!(doubled % 2 == 0, "SwiGLU projection width must be even");
    let inner = doubled / 2;
    let values = projected.narrow(candle_core::D::Minus1, 0, inner)?;
    let gates = projected.narrow(candle_core::D::Minus1, inner, inner)?;
    let activated = values.mul(&silu_with_reference_rounding(&gates)?)?;
    let output_weight = required(weights, &format!("{prefix}.ff.net.2.weight"))?;
    Linear::new(output_weight.clone(), None)
        .forward(&activated)
        .map_err(Into::into)
}

pub(crate) use ff_core::math::silu_with_reference_rounding;

pub(crate) fn qwen_gelu_erf_with_reference_rounding(input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    match dtype {
        candle_core::DType::BF16 | candle_core::DType::F16 => input
            .to_dtype(candle_core::DType::F32)?
            .gelu_erf()?
            .to_dtype(dtype)
            .map_err(Into::into),
        _ => input.gelu_erf().map_err(Into::into),
    }
}

fn normalized_attention_chunk(
    hidden_states: &Tensor,
    norm_weight: &Tensor,
    modulation: &AdaLnModulation,
    adaln_indices: &Tensor,
    start: usize,
    length: usize,
    norm_eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let hidden_chunk = hidden_states.narrow(1, start, length)?.contiguous()?;
    let indices = adaln_indices.narrow(0, start, length)?.contiguous()?;
    let normalized = rms_norm(&hidden_chunk, norm_weight, norm_eps)?;
    let normalized = modulate(
        &normalized,
        &modulation.shift_attention,
        &modulation.scale_attention,
        &indices,
    )?;
    Ok((hidden_chunk, normalized, indices))
}

fn concatenate_chunks(mut chunks: Vec<Tensor>, dimension: usize) -> Result<Tensor> {
    anyhow::ensure!(!chunks.is_empty(), "cannot concatenate an empty chunk set");
    if chunks.len() == 1 {
        return chunks.pop().context("single chunk disappeared");
    }
    let chunk_refs = chunks.iter().collect::<Vec<_>>();
    Tensor::cat(&chunk_refs, dimension).map_err(Into::into)
}

fn map_sequence_chunks<F>(
    input: &Tensor,
    token_chunk_size: NonZeroUsize,
    mut evaluate: F,
) -> Result<Tensor>
where
    F: FnMut(usize, usize, &Tensor) -> Result<Tensor>,
{
    let (_, sequence, _) = input
        .dims3()
        .context("hidden states must be [batch, sequence, hidden]")?;
    anyhow::ensure!(sequence > 0, "hidden-state sequence must be non-empty");
    let token_chunk_size = token_chunk_size.get();
    let mut chunks = Vec::with_capacity(sequence.div_ceil(token_chunk_size));
    for start in (0..sequence).step_by(token_chunk_size) {
        let length = token_chunk_size.min(sequence - start);
        let input_chunk = input.narrow(1, start, length)?.contiguous()?;
        chunks.push(evaluate(start, length, &input_chunk)?);
    }
    concatenate_chunks(chunks, 1)
}

pub(crate) fn scaled_dot_product(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_chunk_size: NonZeroUsize,
) -> Result<Tensor> {
    let (_, sequence, _, _) = query.dims4()?;
    anyhow::ensure!(key.dims() == query.dims(), "query/key shapes differ");
    anyhow::ensure!(value.dims() == query.dims(), "query/value shapes differ");
    let key = key.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
    let value = value.transpose(1, 2)?.contiguous()?;
    let query_chunk_size = query_chunk_size.get();
    let mut chunks = Vec::with_capacity(sequence.div_ceil(query_chunk_size));
    for start in (0..sequence).step_by(query_chunk_size) {
        let length = query_chunk_size.min(sequence - start);
        let query_chunk = query.narrow(1, start, length)?;
        chunks.push(attend_prepared(&query_chunk, &key, &value)?);
    }
    concatenate_chunks(chunks, 1)
}

fn attend_prepared(query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
    let (batch, query_sequence, heads, head_dim) = query.dims4()?;
    let (key_batch, key_heads, key_head_dim, key_sequence) = key.dims4()?;
    anyhow::ensure!(
        (key_batch, key_heads, key_head_dim) == (batch, heads, head_dim),
        "prepared key shape is incompatible with query"
    );
    anyhow::ensure!(
        value.dims() == [batch, heads, key_sequence, head_dim],
        "prepared value shape is incompatible with query/key"
    );
    #[cfg(feature = "cuda")]
    if reference_cuda(query.device()) && query.dtype() == candle_core::DType::BF16 {
        let output_dtype = query.dtype();
        let split_scale = (1.0 / (head_dim as f64).sqrt()).sqrt();
        let query = query
            .transpose(1, 2)?
            .to_dtype(candle_core::DType::F32)?
            .affine(split_scale, 0.)?;
        let key = key
            .to_dtype(candle_core::DType::F32)?
            .affine(split_scale, 0.)?;
        let value = value.to_dtype(candle_core::DType::F32)?;
        let scores = query.matmul(&key)?;
        let width = scores.dim(candle_core::D::Minus1)?;
        let probabilities = if width <= CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS {
            crate::cuda::sdpa_softmax::softmax(&scores)?
        } else if width <= CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS {
            crate::cuda::sdpa_softmax::regular_softmax(&scores)?
        } else {
            anyhow::bail!(
                "H3 native-math CUDA softmax width {width} exceeds the verified exact range \
                 1..={CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS}; enable --flash-attention instead"
            )
        };
        return probabilities
            .matmul(&value)?
            .to_dtype(output_dtype)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, query_sequence, heads * head_dim))
            .map_err(Into::into);
    }
    let query = query
        .transpose(1, 2)?
        .contiguous()?
        .affine(1.0 / (head_dim as f64).sqrt(), 0.)?;
    let scores = query.matmul(key)?;
    let probabilities = softmax_last_dim(&scores)?;
    probabilities
        .matmul(value)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, query_sequence, heads * head_dim))
        .map_err(Into::into)
}

fn attend_prepared_with_key_chunks(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    policy: AttentionKeyChunkPolicy,
) -> Result<Tensor> {
    if policy.is_full() {
        return attend_prepared(query, key, value);
    }
    let output_dtype = query.dtype();
    let (batch, query_sequence, heads, head_dim) = query.dims4()?;
    let (key_batch, key_heads, key_head_dim, key_sequence) = key.dims4()?;
    anyhow::ensure!(query_sequence > 0, "query sequence must be non-empty");
    anyhow::ensure!(key_sequence > 0, "key sequence must be non-empty");
    anyhow::ensure!(
        (key_batch, key_heads, key_head_dim) == (batch, heads, head_dim),
        "prepared key shape is incompatible with query"
    );
    anyhow::ensure!(
        value.dims() == [batch, heads, key_sequence, head_dim],
        "prepared value shape is incompatible with query/key"
    );
    let key_sequence = NonZeroUsize::new(key_sequence).context("key sequence must be non-empty")?;
    let key_chunk_size = policy.effective_chunk_size(key_sequence).get();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let query = query.transpose(1, 2)?.contiguous()?.affine(scale, 0.)?;
    let mut running_max: Option<Tensor> = None;
    let mut running_sum: Option<Tensor> = None;
    let mut running_output: Option<Tensor> = None;

    for start in (0..key_sequence.get()).step_by(key_chunk_size) {
        let length = key_chunk_size.min(key_sequence.get() - start);
        let key_chunk = key.narrow(3, start, length)?.contiguous()?;
        let value_chunk = value
            .narrow(2, start, length)?
            .contiguous()?
            .to_dtype(candle_core::DType::F32)?;
        let scores = query
            .matmul(&key_chunk)?
            .to_dtype(candle_core::DType::F32)?;
        let chunk_max = scores.max_keepdim(candle_core::D::Minus1)?;

        match (
            running_max.take(),
            running_sum.take(),
            running_output.take(),
        ) {
            (Some(previous_max), Some(previous_sum), Some(previous_output)) => {
                let merged_max = previous_max.maximum(&chunk_max)?;
                let previous_scale = previous_max.broadcast_sub(&merged_max)?.exp()?;
                let exponentials = scores.broadcast_sub(&merged_max)?.exp()?;
                let merged_sum = previous_sum
                    .broadcast_mul(&previous_scale)?
                    .add(&exponentials.sum_keepdim(candle_core::D::Minus1)?)?;
                let merged_output = previous_output
                    .broadcast_mul(&previous_scale)?
                    .add(&exponentials.matmul(&value_chunk)?)?;
                running_max = Some(merged_max);
                running_sum = Some(merged_sum);
                running_output = Some(merged_output);
            }
            (None, None, None) => {
                let exponentials = scores.broadcast_sub(&chunk_max)?.exp()?;
                running_sum = Some(exponentials.sum_keepdim(candle_core::D::Minus1)?);
                running_output = Some(exponentials.matmul(&value_chunk)?);
                running_max = Some(chunk_max);
            }
            _ => anyhow::bail!("online attention state became inconsistent"),
        }
    }

    let normalization = running_sum.context("online attention produced no normalization sum")?;
    running_output
        .context("online attention produced no weighted-value output")?
        .broadcast_div(&normalization)?
        .to_dtype(output_dtype)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, query_sequence, heads * head_dim))
        .map_err(Into::into)
}

pub(crate) fn modulate(
    normalized: &Tensor,
    shifts: &Tensor,
    scales: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    let shift = shifts.index_select(indices, 0)?;
    let scale_plus_one = scales.index_select(indices, 0)?.affine(1., 1.)?;
    normalized
        .broadcast_mul(&scale_plus_one)?
        .broadcast_add(&shift)
        .map_err(Into::into)
}

pub(crate) fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotary_dim = cos.dim(1)?;
    anyhow::ensure!(rotary_dim % 2 == 0, "rotary dimension must be even");
    let head_dim = input.dim(candle_core::D::Minus1)?;
    anyhow::ensure!(
        rotary_dim <= head_dim,
        "rotary dimension exceeds attention head dimension"
    );
    let cos = cos.to_dtype(input.dtype())?.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.to_dtype(input.dtype())?.unsqueeze(0)?.unsqueeze(2)?;
    let rotating = input.narrow(candle_core::D::Minus1, 0, rotary_dim)?;
    let half = rotary_dim / 2;
    let first = rotating.narrow(candle_core::D::Minus1, 0, half)?;
    let second = rotating.narrow(candle_core::D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], candle_core::D::Minus1)?;
    let rotating = rotating
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)?;
    if rotary_dim == head_dim {
        Ok(rotating)
    } else {
        let pass = input.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[&rotating, &pass], candle_core::D::Minus1).map_err(Into::into)
    }
}

pub(crate) fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) && input.dtype() == candle_core::DType::BF16 {
        return crate::cuda::rms_norm::rms_norm(input, weight, eps).map_err(Into::into);
    }
    let dtype = input.dtype();
    let input_f32 = input.to_dtype(candle_core::DType::F32)?;
    let variance = input_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    input_f32
        .broadcast_div(&(&variance + eps)?.sqrt()?)?
        .broadcast_mul(&weight.to_dtype(candle_core::DType::F32)?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

/// Qwen3-VL's unfused RMSNorm has two BF16 rounding boundaries: the F32
/// normalization is first cast back to the input dtype, then multiplied by
/// the parameter in that dtype. This is intentionally separate from H3's
/// fused RMSNorm, whose CUDA kernel applies its weight in F32 before one cast.
pub(crate) fn qwen_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    anyhow::ensure!(
        input.dtype() == weight.dtype(),
        "Qwen RMSNorm input and weight dtypes must match"
    );
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) {
        anyhow::ensure!(
            input.dtype() == candle_core::DType::BF16,
            "released Qwen CUDA RMSNorm requires BF16 input and weight"
        );
        if crate::cuda::qwen::attention::head_rms_norm_width128_covers(input) {
            return crate::cuda::qwen::attention::head_rms_norm_width128(input, weight, eps)
                .map_err(Into::into);
        }
        if crate::cuda::qwen::attention::hidden_rms_norm_width5120_covers(input) {
            return crate::cuda::qwen::attention::hidden_rms_norm_width5120(input, weight, eps)
                .map_err(Into::into);
        }
    }
    let dtype = input.dtype();
    let input_f32 = input.to_dtype(candle_core::DType::F32)?;
    let variance = input_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let variance_with_epsilon = (&variance + eps)?;
    #[cfg(feature = "cuda")]
    let inverse_root = if tuned_cuda(input.device()) {
        crate::cuda::rms_norm::qwen_rsqrt(&variance_with_epsilon)?
    } else {
        variance_with_epsilon.sqrt()?.recip()?
    };
    #[cfg(not(feature = "cuda"))]
    let inverse_root = variance_with_epsilon.sqrt()?.recip()?;
    let normalized = input_f32.broadcast_mul(&inverse_root)?.to_dtype(dtype)?;
    weight.broadcast_mul(&normalized).map_err(Into::into)
}

pub(crate) fn qwen_softmax_last_dim(input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let input_f32 = input.to_dtype(candle_core::DType::F32)?;
    #[cfg(feature = "cuda")]
    if tuned_cuda(input.device()) {
        anyhow::ensure!(
            dtype == candle_core::DType::BF16,
            "released Qwen CUDA softmax requires BF16 scores; set {}=0 to run this request through Candle's kernels",
            crate::cuda::profile::DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE
        );
        let width = input_f32.dim(candle_core::D::Minus1)?;
        let probabilities = if width <= CUDA_EXACT_FULL_SOFTMAX_MAX_KEY_ROWS {
            crate::cuda::sdpa_softmax::softmax(&input_f32)?
        } else if width <= QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS {
            crate::cuda::sdpa_softmax::regular_softmax(&input_f32)?
        } else {
            anyhow::bail!(
                "Qwen eager CUDA softmax width {width} exceeds the verified exact range 1..={QWEN_EXACT_SOFTMAX_MAX_KEY_ROWS}; \
                 set {}=0 to run this request through Candle's kernels, which records that it carries no exact-softmax evidence",
                crate::cuda::profile::DISABLE_TUNED_KERNELS_ENVIRONMENT_VARIABLE
            )
        };
        return probabilities.to_dtype(dtype).map_err(Into::into);
    }
    let shifted = input_f32.broadcast_sub(&input_f32.max_keepdim(candle_core::D::Minus1)?)?;
    let exponentials = shifted.exp()?;
    exponentials
        .broadcast_div(&exponentials.sum_keepdim(candle_core::D::Minus1)?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

pub(crate) fn softmax_last_dim(input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let input = input.to_dtype(candle_core::DType::F32)?;
    let shifted = input.broadcast_sub(&input.max_keepdim(candle_core::D::Minus1)?)?;
    let exponentials = shifted.exp()?;
    exponentials
        .broadcast_div(&exponentials.sum_keepdim(candle_core::D::Minus1)?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

pub(crate) fn layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let dtype = input.dtype();
    let input = input.to_dtype(candle_core::DType::F32)?;
    let mean = input.mean_keepdim(candle_core::D::Minus1)?;
    let centered = input.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    centered
        .broadcast_div(&(&variance + eps)?.sqrt()?)?
        .broadcast_mul(&weight.to_dtype(candle_core::DType::F32)?)?
        .broadcast_add(&bias.to_dtype(candle_core::DType::F32)?)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing stage tensor {name}"))
}

fn optional<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Option<&'a Tensor> {
    weights.get(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape};

    fn tensor(shape: impl Into<Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn patterned_tensor(shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| ((index * 11 % 29) as f32 - 14.) / 19.)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    fn non_zero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn attention_chunking(
        projection: usize,
        query: usize,
        key: Option<usize>,
    ) -> AttentionChunking {
        AttentionChunking {
            projection_chunk_size: non_zero(projection),
            query_chunk_size: non_zero(query),
            key: key.map_or(AttentionKeyChunkPolicy::Full, |rows| {
                AttentionKeyChunkPolicy::chunked(rows).unwrap()
            }),
        }
    }

    #[test]
    fn non_cuda_bf16_biased_linear_does_not_add_a_fallback() {
        let input = Tensor::new(&[[0.75f32, -0.5, 0.3125]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::new(
            &[[0.25f32, 0.875, -0.625], [-0.75, 0.5, 0.125]],
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let bias = Tensor::new(&[0.03125f32, -0.0625], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let expected_error = Linear::new(weight.clone(), Some(bias.clone()))
            .forward(&input)
            .unwrap_err()
            .to_string();
        let actual_error = linear_with_reference_bias(&input, &weight, Some(&bias))
            .unwrap_err()
            .to_string();
        assert!(expected_error.contains("unsupported dtype BF16"));
        assert!(actual_error.contains("unsupported dtype BF16"));
    }

    #[test]
    fn half_silu_promotes_before_its_only_rounding_boundary() {
        let input = Tensor::new(&[-8.0f32, -1.25, -0.1, 0.0, 0.75, 9.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let expected = ops::silu(&input.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let actual = silu_with_reference_rounding(&input).unwrap();
        assert_eq!(
            actual
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn qwen_half_gelu_erf_promotes_before_its_only_rounding_boundary() {
        let input = Tensor::new(&[-8.0f32, -1.25, -0.1, 0.0, 0.75, 9.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let expected = input
            .to_dtype(DType::F32)
            .unwrap()
            .gelu_erf()
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let actual = qwen_gelu_erf_with_reference_rounding(&input).unwrap();
        assert_eq!(
            actual
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn rms_norm_matches_pytorch_reference_values() {
        let input = Tensor::new(&[[14.5625f32, 9.5625, -201.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::new(&[1.484375f32, 0.8203125, 0.83984375], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let actual = rms_norm(&input, &weight, 1e-5)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(actual, vec![vec![0.18554688, 0.06738281, -1.4453125]]);
    }

    #[test]
    fn qwen_rms_norm_preserves_the_intermediate_dtype_boundary() {
        let input = Tensor::new(&[[14.5625f32, 9.5625, -201.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::new(&[1.484375f32, 0.8203125, 0.83984375], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let input_f32 = input.to_dtype(DType::F32).unwrap();
        let variance = input_f32
            .sqr()
            .unwrap()
            .mean_keepdim(candle_core::D::Minus1)
            .unwrap();
        let expected = weight
            .broadcast_mul(
                &input_f32
                    .broadcast_mul(&(&variance + 1e-5).unwrap().sqrt().unwrap().recip().unwrap())
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap(),
            )
            .unwrap();
        let actual = qwen_rms_norm(&input, &weight, 1e-5).unwrap();
        assert_eq!(
            actual
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            expected
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
        );
    }

    fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) {
        assert_eq!(left.dims(), right.dims());
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (left, right)) in left.iter().zip(&right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "values differ at {index}: {left} versus {right}"
            );
        }
    }

    fn feed_forward_weights(
        prefix: &str,
        hidden_size: usize,
        inner: usize,
    ) -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        weights.insert(
            format!("{prefix}.norm2.weight"),
            patterned_tensor(hidden_size).affine(0.1, 1.).unwrap(),
        );
        weights.insert(
            format!("{prefix}.ff.net.0.proj.weight"),
            patterned_tensor((2 * inner, hidden_size)),
        );
        weights.insert(
            format!("{prefix}.ff.net.2.weight"),
            patterned_tensor((hidden_size, inner)),
        );
        weights
    }

    fn feed_forward_reference(
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden_states: &Tensor,
        modulation: &AdaLnModulation,
        adaln_indices: &Tensor,
        norm_eps: f64,
    ) -> Result<Tensor> {
        let normalized = rms_norm(
            hidden_states,
            required(weights, &format!("{prefix}.norm2.weight"))?,
            norm_eps,
        )?;
        let normalized = modulate(
            &normalized,
            &modulation.shift_feed_forward,
            &modulation.scale_feed_forward,
            adaln_indices,
        )?;
        let output = swiglu(weights, prefix, &normalized)?;
        let gate = modulation
            .gate_feed_forward
            .index_select(adaln_indices, 0)?;
        hidden_states
            .add(&gate.broadcast_mul(&output)?)
            .map_err(Into::into)
    }

    fn refiner_feed_forward_reference(
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden_states: &Tensor,
        norm_eps: f64,
    ) -> Result<Tensor> {
        let normalized = rms_norm(
            hidden_states,
            required(weights, &format!("{prefix}.norm2.weight"))?,
            norm_eps,
        )?;
        let output = swiglu(weights, prefix, &normalized)?;
        hidden_states.add(&output).map_err(Into::into)
    }

    fn post_matmul_scaled_attention_reference(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor> {
        let (batch, query_sequence, heads, head_dim) = query.dims4()?;
        let (_, _, _, key_sequence) = key.dims4()?;
        let query = query.transpose(1, 2)?.contiguous()?;
        let scores = query
            .matmul(key)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.)?;
        let probabilities = softmax_last_dim(&scores)?;
        probabilities
            .matmul(value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, query_sequence, heads * head_dim))
            .with_context(|| format!("failed to reshape reference with {key_sequence} key rows"))
    }

    fn attention_reference(
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        hidden_states: &Tensor,
        modulation: &AdaLnModulation,
        context: &BlockContext<'_>,
        params: AttentionParams,
    ) -> Result<Tensor> {
        let BlockContext {
            adaln_indices,
            rotary_cos,
            rotary_sin,
        } = *context;
        let AttentionParams {
            heads,
            head_dim,
            norm_eps,
            qk_norm_eps,
        } = params;
        let normalized = rms_norm(
            hidden_states,
            required(weights, &format!("{prefix}.norm1.weight"))?,
            norm_eps,
        )?;
        let normalized = modulate(
            &normalized,
            &modulation.shift_attention,
            &modulation.scale_attention,
            adaln_indices,
        )?;
        let (batch, sequence, _) = normalized.dims3()?;
        let project = |name: &str| -> Result<Tensor> {
            let weight = required(weights, &format!("{prefix}.attn.{name}.weight"))?;
            Linear::new(weight.clone(), None)
                .forward(&normalized)?
                .reshape((batch, sequence, heads, head_dim))
                .map_err(Into::into)
        };
        let query = rms_norm(
            &project("to_q")?,
            required(weights, &format!("{prefix}.attn.norm_q.weight"))?,
            qk_norm_eps,
        )?;
        let key = rms_norm(
            &project("to_k")?,
            required(weights, &format!("{prefix}.attn.norm_k.weight"))?,
            qk_norm_eps,
        )?;
        let query = apply_rotary(&query, rotary_cos, rotary_sin)?;
        let key = apply_rotary(&key, rotary_cos, rotary_sin)?;
        let value = project("to_v")?;
        let prepared_key = key.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
        let prepared_value = value.transpose(1, 2)?.contiguous()?;
        let attended =
            post_matmul_scaled_attention_reference(&query, &prepared_key, &prepared_value)?;
        let output = Linear::new(
            required(weights, &format!("{prefix}.attn.to_out.0.weight"))?.clone(),
            None,
        )
        .forward(&attended)?;
        let gate = modulation.gate_attention.index_select(adaln_indices, 0)?;
        hidden_states
            .add(&gate.broadcast_mul(&output)?)
            .map_err(Into::into)
    }

    #[test]
    fn attention_qkv_chunks_match_unchunked_reference() {
        let prefix = "transformer_blocks.0";
        let hidden_size = 5;
        let sequence = 7;
        let heads = 2;
        let head_dim = 4;
        let mut weights = BTreeMap::new();
        weights.insert(
            format!("{prefix}.norm1.weight"),
            patterned_tensor(hidden_size).affine(0.1, 1.).unwrap(),
        );
        for name in ["to_q", "to_k", "to_v"] {
            weights.insert(
                format!("{prefix}.attn.{name}.weight"),
                patterned_tensor((heads * head_dim, hidden_size)),
            );
        }
        weights.insert(
            format!("{prefix}.attn.norm_q.weight"),
            patterned_tensor(head_dim).affine(0.1, 1.).unwrap(),
        );
        weights.insert(
            format!("{prefix}.attn.norm_k.weight"),
            patterned_tensor(head_dim).affine(0.1, 1.).unwrap(),
        );
        weights.insert(
            format!("{prefix}.attn.to_out.0.weight"),
            patterned_tensor((hidden_size, heads * head_dim)),
        );
        let hidden = patterned_tensor((2, sequence, hidden_size));
        let adaln_indices = Tensor::new(&[0u32, 1, 2, 0, 2, 1, 0], &Device::Cpu).unwrap();
        let modulation = AdaLnModulation {
            shift_attention: patterned_tensor((3, hidden_size)),
            scale_attention: patterned_tensor((3, hidden_size)).affine(0.1, 0.).unwrap(),
            gate_attention: patterned_tensor((3, hidden_size)),
            shift_feed_forward: tensor((3, hidden_size)),
            scale_feed_forward: tensor((3, hidden_size)),
            gate_feed_forward: tensor((3, hidden_size)),
        };
        let rotary_cos = patterned_tensor((sequence, head_dim))
            .affine(0.2, 0.8)
            .unwrap();
        let rotary_sin = patterned_tensor((sequence, head_dim))
            .affine(0.15, 0.1)
            .unwrap();
        let reference = attention_reference(
            &weights,
            prefix,
            &hidden,
            &modulation,
            &BlockContext {
                adaln_indices: &adaln_indices,
                rotary_cos: &rotary_cos,
                rotary_sin: &rotary_sin,
            },
            AttentionParams {
                heads,
                head_dim,
                norm_eps: 1e-5,
                qk_norm_eps: 1e-5,
            },
        )
        .unwrap();

        for query_chunk_size in [1, 2, 3, 4, sequence, sequence + 5] {
            for key_chunk_size in [1, 2, 3, sequence, sequence + 5] {
                let chunked = attention_with_projection_chunks(
                    &weights,
                    prefix,
                    &hidden,
                    &modulation,
                    &BlockContext {
                        adaln_indices: &adaln_indices,
                        rotary_cos: &rotary_cos,
                        rotary_sin: &rotary_sin,
                    },
                    AttentionParams {
                        heads,
                        head_dim,
                        norm_eps: 1e-5,
                        qk_norm_eps: 1e-5,
                    },
                    attention_chunking(query_chunk_size, query_chunk_size, Some(key_chunk_size)),
                )
                .unwrap();
                assert_eq!(chunked.dims(), &[2, sequence, hidden_size]);
                assert_close(&chunked, &reference, 2e-5);
            }
        }

        for (projection_chunk_size, score_query_chunk_size, key_chunk_size) in [
            (1, 1, 1),
            (2, 1, 3),
            (3, 2, 1),
            (4, 3, 2),
            (sequence, 1, sequence),
            (sequence + 5, 2, 3),
            (2, sequence + 5, sequence + 5),
        ] {
            let independently_chunked = attention_with_projection_chunks(
                &weights,
                prefix,
                &hidden,
                &modulation,
                &BlockContext {
                    adaln_indices: &adaln_indices,
                    rotary_cos: &rotary_cos,
                    rotary_sin: &rotary_sin,
                },
                AttentionParams {
                    heads,
                    head_dim,
                    norm_eps: 1e-5,
                    qk_norm_eps: 1e-5,
                },
                attention_chunking(
                    projection_chunk_size,
                    score_query_chunk_size,
                    Some(key_chunk_size),
                ),
            )
            .unwrap();
            assert_eq!(independently_chunked.dims(), &[2, sequence, hidden_size]);
            assert_close(&independently_chunked, &reference, 2e-5);
        }

        let explicit_full_key = attention_with_projection_chunks(
            &weights,
            prefix,
            &hidden,
            &modulation,
            &BlockContext {
                adaln_indices: &adaln_indices,
                rotary_cos: &rotary_cos,
                rotary_sin: &rotary_sin,
            },
            AttentionParams {
                heads,
                head_dim,
                norm_eps: 1e-5,
                qk_norm_eps: 1e-5,
            },
            attention_chunking(3, 3, None),
        )
        .unwrap();
        assert_close(&explicit_full_key, &reference, 2e-5);
    }

    fn emulated_bf16_query_prescaling_score_difference() -> Result<f32> {
        let batch = 2;
        let query_sequence = 5;
        let key_sequence = 7;
        let heads = 2;
        let head_dim = 128;
        let bf16_round = |tensor: Tensor| -> Result<Tensor> {
            Ok(tensor.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)
        };
        let query = bf16_round(
            patterned_tensor((batch, query_sequence, heads, head_dim))
                .transpose(1, 2)?
                .contiguous()?,
        )?;
        let key = bf16_round(
            patterned_tensor((batch, key_sequence, heads, head_dim))
                .affine(0.7, 0.1)?
                .transpose(1, 2)?
                .transpose(2, 3)?
                .contiguous()?,
        )?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let postscaled = bf16_round(bf16_round(query.matmul(&key)?)?.affine(scale, 0.)?)?;
        let prescaled_query = bf16_round(query.affine(scale, 0.)?)?;
        let prescaled = bf16_round(prescaled_query.matmul(&key)?)?;
        prescaled
            .sub(&postscaled)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()
            .map_err(Into::into)
    }

    #[test]
    fn bf16_query_prescaling_rounding_is_bounded_in_cpu_emulation() {
        let difference = emulated_bf16_query_prescaling_score_difference().unwrap();
        assert!(
            difference > 0.,
            "fixture must expose the BF16 rounding change"
        );
        assert!(difference <= 5e-2, "BF16 scaling drift is {difference}");
    }

    #[test]
    fn attention_key_policy_has_typed_json() {
        let full = AttentionKeyChunkPolicy::Full;
        let chunked = AttentionKeyChunkPolicy::chunked(4_096).unwrap();
        assert_eq!(serde_json::to_string(&full).unwrap(), r#""full""#);
        assert_eq!(serde_json::to_string(&chunked).unwrap(), "4096");
        assert_eq!(
            serde_json::from_str::<AttentionKeyChunkPolicy>("4096").unwrap(),
            chunked
        );
        assert!(serde_json::from_str::<AttentionKeyChunkPolicy>("0").is_err());
        assert_eq!(full.effective_chunk_size(non_zero(17)).get(), 17);
        assert_eq!(chunked.effective_chunk_size(non_zero(17)).get(), 17);
        assert_eq!(
            AttentionKeyChunkPolicy::chunked(7)
                .unwrap()
                .effective_chunk_size(non_zero(17))
                .get(),
            7
        );
    }

    #[test]
    fn feed_forward_chunk_sizes_match_unchunked_reference() {
        let prefix = "transformer_blocks.0";
        let hidden_size = 4;
        let sequence = 7;
        let weights = feed_forward_weights(prefix, hidden_size, 5);
        let hidden = patterned_tensor((2, sequence, hidden_size));
        let adaln_indices = Tensor::new(&[0u32, 1, 2, 0, 2, 1, 0], &Device::Cpu).unwrap();
        let modulation = AdaLnModulation {
            shift_attention: tensor((3, hidden_size)),
            scale_attention: tensor((3, hidden_size)),
            gate_attention: tensor((3, hidden_size)),
            shift_feed_forward: patterned_tensor((3, hidden_size)),
            scale_feed_forward: patterned_tensor((3, hidden_size)).affine(0.1, 0.).unwrap(),
            gate_feed_forward: patterned_tensor((3, hidden_size)),
        };
        let reference =
            feed_forward_reference(&weights, prefix, &hidden, &modulation, &adaln_indices, 1e-5)
                .unwrap();

        for token_chunk_size in [1, 2, 3, sequence, sequence + 5] {
            let chunked = feed_forward_chunked(
                &weights,
                prefix,
                &hidden,
                &modulation,
                &adaln_indices,
                non_zero(token_chunk_size),
                1e-5,
            )
            .unwrap();
            assert_eq!(chunked.dims(), &[2, sequence, hidden_size]);
            assert_close(&chunked, &reference, 1e-5);
        }
    }

    #[test]
    fn refiner_feed_forward_chunk_sizes_match_unchunked_reference() {
        let prefix = "token_refiner.refiner_blocks.0";
        let hidden_size = 6;
        let sequence = 9;
        let weights = feed_forward_weights(prefix, hidden_size, 7);
        let hidden = patterned_tensor((2, sequence, hidden_size));
        let reference = refiner_feed_forward_reference(&weights, prefix, &hidden, 1e-5).unwrap();

        for token_chunk_size in [1, 4, sequence, sequence + 5] {
            let chunked = refiner_feed_forward_chunked(
                &weights,
                prefix,
                &hidden,
                non_zero(token_chunk_size),
                1e-5,
            )
            .unwrap();
            assert_eq!(chunked.dims(), &[2, sequence, hidden_size]);
            assert_close(&chunked, &reference, 1e-5);
        }
    }

    #[test]
    fn adaln_supports_four_distinct_timestep_rows() {
        let prefix = "transformer_blocks.0";
        let hidden_size = 4;
        let mut weights = BTreeMap::new();
        weights.insert(
            format!("{prefix}.adaln_proj.linear.weight"),
            patterned_tensor((6 * hidden_size * MODALITY_COUNT, 3)),
        );
        weights.insert(
            format!("{prefix}.adaln_proj.linear.bias"),
            patterned_tensor(6 * hidden_size * MODALITY_COUNT),
        );
        let timestep_embedding = patterned_tensor((4, 3));
        let modulation = adaln(&weights, prefix, &timestep_embedding, hidden_size).unwrap();
        for tensor in [
            modulation.shift_attention,
            modulation.scale_attention,
            modulation.gate_attention,
            modulation.shift_feed_forward,
            modulation.scale_feed_forward,
            modulation.gate_feed_forward,
        ] {
            assert_eq!(tensor.dims(), &[4 * MODALITY_COUNT, hidden_size]);
        }
    }

    #[test]
    fn streamed_block_preserves_shapes_and_finite_values() {
        let hidden_size = 4;
        let heads = 2;
        let head_dim = 3;
        let inner = 5;
        let prefix = "transformer_blocks.0";
        let mut adaln_weights = BTreeMap::new();
        adaln_weights.insert(
            format!("{prefix}.adaln_proj.linear.weight"),
            tensor((6 * hidden_size * MODALITY_COUNT, 2)),
        );
        adaln_weights.insert(
            format!("{prefix}.adaln_proj.linear.bias"),
            Tensor::zeros(6 * hidden_size * MODALITY_COUNT, DType::F32, &Device::Cpu).unwrap(),
        );
        let temb = tensor((1, 2));
        let modulation = adaln(&adaln_weights, prefix, &temb, hidden_size).unwrap();

        let mut attention_weights = BTreeMap::new();
        attention_weights.insert(format!("{prefix}.norm1.weight"), tensor(hidden_size));
        for name in ["to_q", "to_k", "to_v"] {
            attention_weights.insert(
                format!("{prefix}.attn.{name}.weight"),
                tensor((heads * head_dim, hidden_size)),
            );
        }
        attention_weights.insert(format!("{prefix}.attn.norm_q.weight"), tensor(head_dim));
        attention_weights.insert(format!("{prefix}.attn.norm_k.weight"), tensor(head_dim));
        attention_weights.insert(
            format!("{prefix}.attn.to_out.0.weight"),
            tensor((hidden_size, heads * head_dim)),
        );
        let hidden = tensor((1, 2, hidden_size));
        let indices = Tensor::new(&[0u32, 1], &Device::Cpu).unwrap();
        let cos = tensor((2, 2));
        let sin = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let hidden_chunked = attention_with_projection_chunks(
            &attention_weights,
            prefix,
            &hidden,
            &modulation,
            &BlockContext {
                adaln_indices: &indices,
                rotary_cos: &cos,
                rotary_sin: &sin,
            },
            AttentionParams {
                heads,
                head_dim,
                norm_eps: 1e-5,
                qk_norm_eps: 1e-5,
            },
            attention_chunking(1, 1, Some(2)),
        )
        .unwrap();
        let hidden_full = attention_with_projection_chunks(
            &attention_weights,
            prefix,
            &hidden,
            &modulation,
            &BlockContext {
                adaln_indices: &indices,
                rotary_cos: &cos,
                rotary_sin: &sin,
            },
            AttentionParams {
                heads,
                head_dim,
                norm_eps: 1e-5,
                qk_norm_eps: 1e-5,
            },
            attention_chunking(2, 2, None),
        )
        .unwrap();
        assert_eq!(hidden_chunked.dims(), &[1, 2, hidden_size]);
        assert_eq!(
            hidden_chunked
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            hidden_full.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let mut ff_weights = BTreeMap::new();
        ff_weights.insert(format!("{prefix}.norm2.weight"), tensor(hidden_size));
        ff_weights.insert(
            format!("{prefix}.ff.net.0.proj.weight"),
            tensor((2 * inner, hidden_size)),
        );
        ff_weights.insert(
            format!("{prefix}.ff.net.2.weight"),
            tensor((hidden_size, inner)),
        );
        let output = feed_forward_chunked(
            &ff_weights,
            prefix,
            &hidden_chunked,
            &modulation,
            &indices,
            non_zero(DEFAULT_FFN_TOKEN_CHUNK_SIZE),
            1e-5,
        )
        .unwrap();
        assert_eq!(output.dims(), &[1, 2, hidden_size]);
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
