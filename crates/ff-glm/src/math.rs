//! Pure tensor mathematics used by the GLM-5.3-Flash adapter.
//!
//! The checkpoint-facing model code owns tensor names and residency.  This
//! module deliberately contains neither file-system access nor model-specific
//! caching so that the numerically delicate pieces can be tested on small CPU
//! tensors.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Tensor};

pub(crate) mod normalization;
mod prefill;
pub(crate) mod sinkhorn;
pub use prefill::kda_prefill;

/// Output of GLM's sigmoid/noaux top-k router.
#[derive(Debug)]
pub struct RouterOutput {
    /// Unbiased, pre-sigmoid router logits in F32, `[tokens, experts]`.
    pub logits: Tensor,
    /// Normalized and scaled weights for the selected experts, `[tokens, top_k]`.
    pub weights: Tensor,
    /// Selected expert indices, `[tokens, top_k]`.
    pub indices: Tensor,
}

/// RMS normalization with its variance evaluated in F32, matching the
/// reference Transformers implementation.  The unit-normalized value is cast
/// back to the input dtype before the optional affine multiplication.
pub fn rms_norm(input: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    ensure!(
        eps.is_finite() && eps >= 0.0,
        "RMSNorm epsilon must be finite and non-negative"
    );
    let input_dtype = input.dtype();
    let normalized = normalization::normalized(input, eps)?.to_dtype(input_dtype)?;

    if let Some(weight) = weight {
        let width = *input
            .dims()
            .last()
            .context("RMSNorm input must have at least one dimension")?;
        ensure!(
            weight.dims() == [width],
            "RMSNorm weight shape must be [{width}], got {:?}",
            weight.dims()
        );
        normalized
            .broadcast_mul(&weight.to_dtype(input_dtype)?)
            .map_err(Into::into)
    } else {
        Ok(normalized)
    }
}

/// KDA's output RMSNorm followed by its learned sigmoid gate.  Unlike ordinary
/// GLM RMSNorm, the affine and gate multiplication intentionally remain F32
/// until the final cast.
pub fn rms_norm_gated(input: &Tensor, weight: &Tensor, gate: &Tensor, eps: f64) -> Result<Tensor> {
    ensure!(
        eps.is_finite() && eps >= 0.0,
        "gated RMSNorm epsilon must be finite and non-negative"
    );
    let width = *input
        .dims()
        .last()
        .context("gated RMSNorm input must have at least one dimension")?;
    ensure!(
        weight.dims() == [width],
        "gated RMSNorm weight must have shape [{width}], got {:?}",
        weight.dims()
    );
    ensure!(
        gate.shape() == input.shape(),
        "gated RMSNorm gate shape must match input: {:?} versus {:?}",
        gate.dims(),
        input.dims()
    );

    let input_dtype = input.dtype();
    normalization::normalized(input, eps)?
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .mul(&candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
        .to_dtype(input_dtype)
        .map_err(Into::into)
}

pub(crate) use ff_core::math::silu_with_reference_rounding;

/// The native half-precision sigmoid also computes its intermediates in F32.
pub(crate) fn sigmoid_with_reference_rounding(input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    match dtype {
        DType::BF16 | DType::F16 => candle_nn::ops::sigmoid(&input.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
            .map_err(Into::into),
        _ => candle_nn::ops::sigmoid(input).map_err(Into::into),
    }
}

/// GLM's clamped SwiGLU: only the upper side of the gate is clamped, while
/// the up projection is clamped symmetrically.
pub fn clamped_swiglu(gate: &Tensor, up: &Tensor, limit: f64) -> Result<Tensor> {
    ensure!(
        gate.shape() == up.shape(),
        "SwiGLU gate/up shapes differ: {:?} versus {:?}",
        gate.dims(),
        up.dims()
    );
    ensure!(
        limit.is_finite() && limit > 0.0,
        "SwiGLU clamp limit must be finite and positive"
    );

    let gate = gate.minimum(limit)?;
    let up = up.clamp(-limit, limit)?;
    silu_with_reference_rounding(&gate)?
        .mul(&up)
        .map_err(Into::into)
}

/// Compute one mHC collapse/expand mapping for a single token.
///
/// `hidden_streams` is `[streams, hidden]`, `mapping_weight` is
/// `[(2 + streams) * streams, streams * hidden]`, and `mapping_base` has the
/// corresponding output width.  `post` and `comb` stay in F32, as in the
/// reference implementation; `collapsed` is cast back to the input dtype.
pub fn mhc_map(
    hidden_streams: &Tensor,
    mapping_weight: &Tensor,
    mapping_base: &Tensor,
    mapping_scale: &Tensor,
    rms_eps: f64,
    hc_eps: f64,
    sinkhorn_iters: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    hidden_streams
        .dims2()
        .context("mHC hidden streams must have shape [streams, hidden]")?;
    let (post, comb, collapsed) = mhc_map_batched(
        &hidden_streams.unsqueeze(0)?,
        mapping_weight,
        mapping_base,
        mapping_scale,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )?;
    Ok((post.squeeze(0)?, comb.squeeze(0)?, collapsed.squeeze(0)?))
}

/// The same mHC mapping over all prefill rows, `[tokens, streams, hidden]`.
/// Its F32 linear projection keeps the actual token batch rather than looping
/// through matrix-vector products with a different accumulation trajectory.
pub fn mhc_map_batched(
    hidden_streams: &Tensor,
    mapping_weight: &Tensor,
    mapping_base: &Tensor,
    mapping_scale: &Tensor,
    rms_eps: f64,
    hc_eps: f64,
    sinkhorn_iters: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (tokens, streams, hidden) = hidden_streams
        .dims3()
        .context("batched mHC streams must have shape [tokens, streams, hidden]")?;
    ensure!(
        tokens > 0 && streams > 0 && hidden > 0,
        "mHC dimensions must be non-zero"
    );
    ensure!(
        sinkhorn_iters > 0,
        "mHC needs at least one Sinkhorn iteration"
    );
    ensure!(
        hc_eps.is_finite() && hc_eps >= 0.0,
        "mHC epsilon must be finite and non-negative"
    );

    let mix = (2 + streams)
        .checked_mul(streams)
        .context("mHC mapping width overflow")?;
    let flat_width = streams
        .checked_mul(hidden)
        .context("mHC flattened width overflow")?;
    ensure!(
        mapping_weight.dims() == [mix, flat_width],
        "mHC mapping weight must have shape [{mix}, {flat_width}], got {:?}",
        mapping_weight.dims()
    );
    ensure!(
        mapping_base.dims() == [mix],
        "mHC base must have shape [{mix}], got {:?}",
        mapping_base.dims()
    );
    ensure!(
        mapping_scale.dims() == [3],
        "mHC scale must have shape [3], got {:?}",
        mapping_scale.dims()
    );

    let input_dtype = hidden_streams.dtype();
    let flat = hidden_streams
        .reshape((tokens, flat_width))?
        .to_dtype(DType::F32)?;
    let flat = rms_norm(&flat, None, rms_eps)?;
    let mixed = linear_f32(&flat, mapping_weight)?;

    // Reuse cached F32 constants without redundant conversion copies.
    let scale_owned;
    let scale: &Tensor = if mapping_scale.dtype() == DType::F32 {
        mapping_scale
    } else {
        scale_owned = mapping_scale.to_dtype(DType::F32)?;
        &scale_owned
    };
    let base_owned;
    let base: &Tensor = if mapping_base.dtype() == DType::F32 {
        mapping_base
    } else {
        base_owned = mapping_base.to_dtype(DType::F32)?;
        &base_owned
    };
    let comb_logits = mixed.narrow(1, 2 * streams, streams * streams)?;
    let comb_scale = scale.narrow(0, 2, 1)?;
    let comb_base = base.narrow(0, 2 * streams, streams * streams)?;

    // Batch adjacent pre/post gates without changing per-element arithmetic.
    // Apply each half's final constant separately (+eps for pre, *2 for post).
    let pp_scale = Tensor::cat(
        &[
            &scale.narrow(0, 0, 1)?.broadcast_as((streams,))?,
            &scale.narrow(0, 1, 1)?.broadcast_as((streams,))?,
        ],
        0,
    )?;
    let pp = candle_nn::ops::sigmoid(
        &mixed
            .narrow(1, 0, 2 * streams)?
            .broadcast_mul(&pp_scale)?
            .broadcast_add(&base.narrow(0, 0, 2 * streams)?)?,
    )?;
    let pre = (&pp.narrow(1, 0, streams)? + hc_eps)?;
    let post = pp.narrow(1, streams, streams)?.affine(2.0, 0.0)?;

    let comb_logits = comb_logits
        .broadcast_mul(&comb_scale)?
        .broadcast_add(&comb_base)?
        .reshape((tokens, streams, streams))?;
    let mut comb = (&candle_nn::ops::softmax(&comb_logits, D::Minus1)? + hc_eps)?;
    comb = comb.broadcast_div(&(&comb.sum_keepdim(1)? + hc_eps)?)?;
    let comb = sinkhorn::fused_loop(&comb, sinkhorn_iters - 1, hc_eps)?;

    let weighted_streams = pre
        .unsqueeze(2)?
        .broadcast_mul(&hidden_streams.to_dtype(DType::F32)?)?;
    let collapsed = if hidden_streams.device().is_cuda() {
        prefill::sum_short_strided_rows(&weighted_streams.unsqueeze(0)?)?.squeeze(0)?
    } else {
        weighted_streams.sum(1)?
    }
    .to_dtype(input_dtype)?;
    Ok((post, comb, collapsed))
}

/// Apply the mHC expansion and residual-stream mixing after a sublayer.
pub fn apply_mhc_residual(
    residual_streams: &Tensor,
    sublayer_output: &Tensor,
    post: &Tensor,
    comb: &Tensor,
) -> Result<Tensor> {
    let (streams, hidden) = residual_streams
        .dims2()
        .context("mHC residual must have shape [streams, hidden]")?;
    ensure!(
        sublayer_output.dims() == [hidden],
        "mHC sublayer output must have shape [{hidden}], got {:?}",
        sublayer_output.dims()
    );
    ensure!(
        post.dims() == [streams],
        "mHC post weight must have shape [{streams}]"
    );
    ensure!(
        comb.dims() == [streams, streams],
        "mHC combine matrix must have shape [{streams}, {streams}]"
    );
    let dtype = residual_streams.dtype();
    let placed = post
        .to_dtype(dtype)?
        .unsqueeze(1)?
        .broadcast_mul(&sublayer_output.to_dtype(dtype)?.unsqueeze(0)?)?;
    let mixed = comb
        .to_dtype(dtype)?
        .t()?
        .contiguous()?
        .matmul(residual_streams)?;
    placed.add(&mixed).map_err(Into::into)
}

/// Apply mHC placement and residual mixing to `[tokens, streams, hidden]`.
pub fn apply_mhc_residual_batched(
    residual_streams: &Tensor,
    sublayer_output: &Tensor,
    post: &Tensor,
    comb: &Tensor,
) -> Result<Tensor> {
    let (tokens, streams, hidden) = residual_streams
        .dims3()
        .context("batched mHC residual must have shape [tokens, streams, hidden]")?;
    ensure!(
        sublayer_output.dims() == [tokens, hidden],
        "mHC sublayer output must have shape [{tokens}, {hidden}], got {:?}",
        sublayer_output.dims()
    );
    ensure!(
        post.dims() == [tokens, streams],
        "mHC post weight must have shape [{tokens}, {streams}]"
    );
    ensure!(
        comb.dims() == [tokens, streams, streams],
        "mHC combine matrix must have shape [{tokens}, {streams}, {streams}]"
    );

    let dtype = residual_streams.dtype();
    let placed = post
        .to_dtype(dtype)?
        .unsqueeze(2)?
        .broadcast_mul(&sublayer_output.to_dtype(dtype)?.unsqueeze(1)?)?;
    let mixed = comb
        .to_dtype(dtype)?
        .transpose(1, 2)?
        .contiguous()?
        .matmul(residual_streams)?;
    placed.add(&mixed).map_err(Into::into)
}

/// Route flattened token states using GLM's sigmoid/noaux policy.
///
/// GLM-5.3-Flash has one expert group, so its group-selection stage is an
/// identity.  Selection uses the correction bias, while returned mixture
/// weights intentionally come from the uncorrected sigmoid scores.
pub fn topk_router(
    hidden_states: &Tensor,
    router_weight: &Tensor,
    correction_bias: &Tensor,
    top_k: usize,
    routed_scaling_factor: f64,
    normalize_topk: bool,
) -> Result<RouterOutput> {
    let (tokens, hidden) = hidden_states
        .dims2()
        .context("router input must have shape [tokens, hidden]")?;
    let (experts, router_hidden) = router_weight
        .dims2()
        .context("router weight must have shape [experts, hidden]")?;
    ensure!(
        hidden == router_hidden,
        "router input and weight hidden sizes differ"
    );
    ensure!(
        correction_bias.dims() == [experts],
        "router correction bias must have shape [{experts}]"
    );
    ensure!(
        top_k > 0 && top_k <= experts,
        "router top_k must be in 1..={experts}"
    );
    ensure!(
        routed_scaling_factor.is_finite(),
        "router scaling factor must be finite"
    );
    if hidden_states.device().is_cuda() {
        ensure!(
            tokens > 0 && tokens <= 2048 && experts <= 288,
            "GLM CUDA router supports 1..=2048 rows and at most 288 experts"
        );
    }

    let logits = linear_f32(&hidden_states.to_dtype(DType::F32)?, router_weight)?;
    let scores = candle_nn::ops::sigmoid(&logits)?;
    let choice_scores = scores.broadcast_add(&correction_bias.to_dtype(DType::F32)?)?;
    let indices = if hidden_states.device().is_cuda() {
        let rows = choice_scores
            .to_device(&candle_core::Device::Cpu)?
            .to_vec2::<f32>()?;
        let mut indices = Vec::with_capacity(tokens * top_k);
        for row in rows {
            indices.extend(cuda_reference_topk_indices(&row, top_k)?);
        }
        Tensor::from_vec(indices, (tokens, top_k), hidden_states.device())?
    } else {
        let (_, sorted_indices) = choice_scores.contiguous()?.sort_last_dim(false)?;
        sorted_indices.narrow(D::Minus1, 0, top_k)?.contiguous()?
    };
    let mut weights = scores.gather(&indices, D::Minus1)?;
    if normalize_topk {
        weights = weights.broadcast_div(&(&weights.sum_keepdim(D::Minus1)? + 1e-20)?)?;
    }
    weights = weights.affine(routed_scaling_factor, 0.0)?;

    Ok(RouterOutput {
        logits,
        weights,
        indices,
    })
}

/// Finite-F32 selection order of pinned PyTorch CUDA single-block gatherTopK:
/// strict winners in original index order, followed by first-seen cutoff ties.
/// Integer ordering preserves the native distinction between +0 and -0.
fn cuda_reference_topk_indices(scores: &[f32], top_k: usize) -> Result<Vec<u32>> {
    ensure!(
        !scores.is_empty() && scores.len() <= 288 && top_k > 0 && top_k <= scores.len(),
        "GLM reference top-k dimensions are out of bounds"
    );
    ensure!(
        scores.iter().all(|v| v.is_finite()),
        "GLM CUDA router choice scores must be finite"
    );
    let keys = scores
        .iter()
        .map(|v| {
            let bits = v.to_bits();
            bits ^ if bits & 0x8000_0000 != 0 {
                u32::MAX
            } else {
                0x8000_0000
            }
        })
        .collect::<Vec<_>>();
    let mut partition = keys.clone();
    let (_, threshold, _) = partition.select_nth_unstable_by(top_k - 1, |a, b| b.cmp(a));
    let threshold = *threshold;
    let mut selected = keys
        .iter()
        .enumerate()
        .filter(|(_, key)| **key > threshold)
        .map(|(index, _)| index as u32)
        .collect::<Vec<_>>();
    let remaining = top_k - selected.len();
    selected.extend(
        keys.iter()
            .enumerate()
            .filter(|(_, key)| **key == threshold)
            .take(remaining)
            .map(|(index, _)| index as u32),
    );
    ensure!(
        selected.len() == top_k,
        "GLM threshold scan did not fill top-k indices"
    );
    Ok(selected)
}

/// Compute KDA's safe-lower-bound forget gate in F32.
pub fn kda_forget_gate(
    hidden_states: &Tensor,
    f_a_weight: &Tensor,
    f_b_weight: &Tensor,
    dt_bias: &Tensor,
    a_log: &Tensor,
    safe_lower_bound: f64,
) -> Result<Tensor> {
    let (tokens, hidden) = hidden_states
        .dims2()
        .context("KDA forget-gate input must have shape [tokens, hidden]")?;
    let (rank, f_a_hidden) = f_a_weight
        .dims2()
        .context("KDA f_a weight must have shape [rank, hidden]")?;
    let (qkv_dim, f_b_rank) = f_b_weight
        .dims2()
        .context("KDA f_b weight must have shape [heads * head_dim, rank]")?;
    let heads = a_log.dims1().context("KDA A_log must have shape [heads]")?;
    ensure!(
        hidden == f_a_hidden && rank == f_b_rank,
        "KDA forget-gate projection shapes are incompatible"
    );
    ensure!(
        heads > 0 && qkv_dim.is_multiple_of(heads),
        "KDA forget-gate head dimensions are invalid"
    );
    ensure!(
        dt_bias.dims() == [qkv_dim],
        "KDA dt_bias must have shape [{qkv_dim}]"
    );
    ensure!(
        safe_lower_bound.is_finite() && safe_lower_bound <= 0.0,
        "KDA safe gate lower bound must be finite and non-positive"
    );

    let head_dim = qkv_dim / heads;
    let projected = linear(hidden_states, f_a_weight)?;
    let projected = linear(&projected, f_b_weight)?
        .to_dtype(DType::F32)?
        .broadcast_add(&dt_bias.to_dtype(DType::F32)?)?
        .reshape((tokens, heads, head_dim))?;
    let decay_rate = a_log.to_dtype(DType::F32)?.exp()?.reshape((1, heads, 1))?;
    candle_nn::ops::sigmoid(&projected.broadcast_mul(&decay_rate)?)?
        .affine(safe_lower_bound, 0.0)
        .map_err(Into::into)
}

/// One recurrent Kimi Delta Attention update for a single token.
///
/// `query` and `key` are `[heads, key_dim]`, `value` is
/// `[heads, value_dim]`, `log_decay` is `[heads, key_dim]`, `beta` is
/// `[heads]`, and the persistent F32 state is `[heads, key_dim, value_dim]`.
/// Q/K L2 normalization and all recurrence arithmetic are performed in F32.
pub fn kda_single_token(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (heads, key_dim) = query
        .dims2()
        .context("KDA query must have shape [heads, key_dim]")?;
    let (key_heads, key_width) = key
        .dims2()
        .context("KDA key must have shape [heads, key_dim]")?;
    let (value_heads, value_dim) = value
        .dims2()
        .context("KDA value must have shape [heads, value_dim]")?;
    ensure!(
        heads > 0 && key_dim > 0 && value_dim > 0,
        "KDA dimensions must be non-zero"
    );
    ensure!(
        key_heads == heads && key_width == key_dim,
        "KDA query/key shapes differ"
    );
    ensure!(value_heads == heads, "KDA query/value head counts differ");
    ensure!(
        log_decay.dims() == [heads, key_dim],
        "KDA log decay must have shape [{heads}, {key_dim}]"
    );
    ensure!(beta.dims() == [heads], "KDA beta must have shape [{heads}]");
    ensure!(
        state.dims() == [heads, key_dim, value_dim],
        "KDA state must have shape [{heads}, {key_dim}, {value_dim}]"
    );

    let output_dtype = query.dtype();
    let query = l2_normalize_f32(query, 1e-6)?.affine(1.0 / (key_dim as f64).sqrt(), 0.0)?;
    let key = l2_normalize_f32(key, 1e-6)?;
    let value = value.to_dtype(DType::F32)?;
    let decayed_state = state
        .to_dtype(DType::F32)?
        .broadcast_mul(&log_decay.to_dtype(DType::F32)?.exp()?.unsqueeze(2)?)?;

    let recalled = decayed_state.broadcast_mul(&key.unsqueeze(2)?)?.sum(1)?;
    let delta = value
        .sub(&recalled)?
        .broadcast_mul(&beta.to_dtype(DType::F32)?.unsqueeze(1)?)?;
    let update = key.unsqueeze(2)?.broadcast_mul(&delta.unsqueeze(1)?)?;
    let next_state = decayed_state.add(&update)?;
    let output = next_state
        .broadcast_mul(&query.unsqueeze(2)?)?
        .sum(1)?
        .to_dtype(output_dtype)?;
    Ok((output, next_state))
}

fn linear(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let input_width = *input
        .dims()
        .last()
        .context("linear input must have at least one dimension")?;
    let (_, weight_width) = weight
        .dims2()
        .context("linear weight must have shape [output, input]")?;
    ensure!(
        input_width == weight_width,
        "linear input and weight widths differ"
    );
    input.matmul(&weight.t()?).map_err(Into::into)
}

fn linear_f32(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    linear(&input.to_dtype(DType::F32)?, &weight.to_dtype(DType::F32)?)
}

fn l2_normalize_f32(input: &Tensor, eps: f64) -> Result<Tensor> {
    let input = input.to_dtype(DType::F32)?;
    let denominator = (&input.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    input.broadcast_div(&denominator).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::Device;

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn rms_norm_uses_f32_reduction_and_affine_weight() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[3f32, 4.]], &device).unwrap();
        let weight = Tensor::new(&[2f32, 0.5], &device).unwrap();
        let output = rms_norm(&input, Some(&weight), 0.0).unwrap();
        let rms = 12.5f32.sqrt();
        close(
            &output.flatten_all().unwrap().to_vec1().unwrap(),
            &[6.0 / rms, 2.0 / rms],
            1e-6,
        );
    }

    #[test]
    fn gated_rms_norm_keeps_gate_math_in_f32() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[3f32, 4.]], &device).unwrap();
        let weight = Tensor::new(&[2f32, 0.5], &device).unwrap();
        let gate = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
        let output = rms_norm_gated(&input, &weight, &gate, 0.0).unwrap();
        let rms = 12.5f32.sqrt();
        close(
            &output.flatten_all().unwrap().to_vec1().unwrap(),
            &[3.0 / rms, 1.0 / rms],
            1e-6,
        );
    }

    #[test]
    fn half_precision_silu_and_swiglu_round_only_after_activation() {
        let device = Device::Cpu;
        let values = (-1280..=1280).map(|n| n as f32 / 128.0).collect::<Vec<_>>();
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let gate = Tensor::new(values.as_slice(), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let expected = gate
                .to_dtype(DType::F32)
                .unwrap()
                .silu()
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let host = |value: Tensor| {
                value
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                host(silu_with_reference_rounding(&gate).unwrap()),
                host(expected.clone())
            );
            let expected_sigmoid = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32).unwrap())
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            assert_eq!(
                host(sigmoid_with_reference_rounding(&gate).unwrap()),
                host(expected_sigmoid)
            );
            let up = Tensor::ones(gate.shape(), dtype, &device).unwrap();
            assert_eq!(
                host(clamped_swiglu(&gate, &up, 10.0).unwrap()),
                host(expected)
            );
        }
    }

    #[test]
    fn swiglu_applies_asymmetric_checkpoint_clamps() {
        let device = Device::Cpu;
        let gate = Tensor::new(&[-20f32, 20.], &device).unwrap();
        let up = Tensor::new(&[-20f32, 20.], &device).unwrap();
        let output = clamped_swiglu(&gate, &up, 10.0).unwrap();
        let expected = [
            200.0 * (-20.0f32).exp() / (1.0 + (-20.0f32).exp()),
            100.0 / (1.0 + (-10.0f32).exp()),
        ];
        close(&output.to_vec1().unwrap(), &expected, 1e-4);
    }

    #[test]
    fn batched_mhc_keeps_token_rows_independent() {
        use candle_core::{Device, IndexOp};
        let d = Device::Cpu;
        let streams = Tensor::arange(0f32, 18f32, &d)
            .unwrap()
            .reshape((3, 2, 3))
            .unwrap();
        let weight = Tensor::from_vec(
            (0..48)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
                .collect::<Vec<_>>(),
            (8, 6),
            &d,
        )
        .unwrap();
        let base = Tensor::zeros(8, DType::F32, &d).unwrap();
        let scale = Tensor::full(0.5f32, 3, &d).unwrap();
        let (post, comb, collapsed) =
            mhc_map_batched(&streams, &weight, &base, &scale, 1e-5, 1e-6, 20).unwrap();
        let mixed = apply_mhc_residual_batched(&streams, &collapsed, &post, &comb).unwrap();
        for token in 0..3 {
            let input = streams.i(token).unwrap();
            let (p, c, x) = mhc_map(&input, &weight, &base, &scale, 1e-5, 1e-6, 20).unwrap();
            let residual = apply_mhc_residual(&input, &x, &p, &c).unwrap();
            for (batch, one) in [(&post, p), (&comb, c), (&collapsed, x), (&mixed, residual)] {
                close(
                    &batch
                        .i(token)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap(),
                    &one.flatten_all().unwrap().to_vec1().unwrap(),
                    1e-6,
                );
            }
        }
    }

    #[test]
    fn mhc_zero_logits_average_streams() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[2f32], [4.]], &device).unwrap();
        let mapping_weight = Tensor::zeros((8, 2), DType::F32, &device).unwrap();
        let mapping_base = Tensor::zeros(8, DType::F32, &device).unwrap();
        let mapping_scale = Tensor::ones(3, DType::F32, &device).unwrap();
        let (post, comb, collapsed) = mhc_map(
            &hidden,
            &mapping_weight,
            &mapping_base,
            &mapping_scale,
            1e-5,
            0.0,
            2,
        )
        .unwrap();
        close(&post.to_vec1().unwrap(), &[1.0, 1.0], 1e-6);
        close(
            &comb.flatten_all().unwrap().to_vec1().unwrap(),
            &[0.5, 0.5, 0.5, 0.5],
            1e-6,
        );
        close(&collapsed.to_vec1().unwrap(), &[3.0], 1e-6);

        let output = apply_mhc_residual(
            &hidden,
            &Tensor::new(&[10f32], &device).unwrap(),
            &post,
            &comb,
        )
        .unwrap();
        close(
            &output.flatten_all().unwrap().to_vec1().unwrap(),
            &[13.0, 13.0],
            1e-6,
        );
    }

    /// Batching the pre/post gates must preserve bitwise equality.
    #[test]
    fn mhc_batched_gate_chain_matches_split_chain_bitwise() {
        let device = Device::Cpu;
        let (tokens, streams) = (3usize, 4usize);
        let hc_eps = 1e-6f64;
        let mixed = Tensor::from_vec(
            (0..tokens * 4 * streams)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.37)
                .collect::<Vec<_>>(),
            (tokens, 4 * streams),
            &device,
        )
        .unwrap();
        let scale = Tensor::from_vec(vec![0.5f32, 1.5, 2.5], 3, &device).unwrap();
        let base = Tensor::from_vec(
            (0..4 * streams)
                .map(|i| (i as f32 - 8.0) * 0.11)
                .collect::<Vec<_>>(),
            4 * streams,
            &device,
        )
        .unwrap();

        // Split chain (the original two-pass form).
        let pre_old = (&candle_nn::ops::sigmoid(
            &mixed
                .narrow(1, 0, streams)
                .unwrap()
                .broadcast_mul(&scale.narrow(0, 0, 1).unwrap())
                .unwrap()
                .broadcast_add(&base.narrow(0, 0, streams).unwrap())
                .unwrap(),
        )
        .unwrap()
            + hc_eps)
            .unwrap();
        let post_old = candle_nn::ops::sigmoid(
            &mixed
                .narrow(1, streams, streams)
                .unwrap()
                .broadcast_mul(&scale.narrow(0, 1, 1).unwrap())
                .unwrap()
                .broadcast_add(&base.narrow(0, streams, streams).unwrap())
                .unwrap(),
        )
        .unwrap()
        .affine(2.0, 0.0)
        .unwrap();

        // Batched chain (the fused form under test).
        let pp_scale = Tensor::cat(
            &[
                &scale
                    .narrow(0, 0, 1)
                    .unwrap()
                    .broadcast_as((streams,))
                    .unwrap(),
                &scale
                    .narrow(0, 1, 1)
                    .unwrap()
                    .broadcast_as((streams,))
                    .unwrap(),
            ],
            0,
        )
        .unwrap();
        let pp = candle_nn::ops::sigmoid(
            &mixed
                .narrow(1, 0, 2 * streams)
                .unwrap()
                .broadcast_mul(&pp_scale)
                .unwrap()
                .broadcast_add(&base.narrow(0, 0, 2 * streams).unwrap())
                .unwrap(),
        )
        .unwrap();
        let pre_new = (&pp.narrow(1, 0, streams).unwrap() + hc_eps).unwrap();
        let post_new = pp
            .narrow(1, streams, streams)
            .unwrap()
            .affine(2.0, 0.0)
            .unwrap();

        for (batched, split) in [(&pre_new, &pre_old), (&post_new, &post_old)] {
            let batched = batched.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let split = split.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(batched.len(), split.len());
            for (index, (batched, split)) in batched.iter().zip(&split).enumerate() {
                assert_eq!(
                    batched.to_bits(),
                    split.to_bits(),
                    "element {index}: batched {batched:e} != split {split:e}"
                );
            }
        }
    }

    #[test]
    fn router_chooses_with_bias_but_weights_with_raw_scores() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[1f32, 0.]], &device).unwrap();
        let weight = Tensor::new(&[[2f32, 0.], [1., 0.], [0., 0.]], &device).unwrap();
        let bias = Tensor::new(&[-10f32, 0., 10.], &device).unwrap();
        let output = topk_router(&hidden, &weight, &bias, 2, 2.5, true).unwrap();
        assert_eq!(output.indices.to_vec2::<u32>().unwrap(), vec![vec![2, 1]]);
        let raw = [0.5f32, 1.0 / (1.0 + (-1.0f32).exp())];
        let denominator = raw[0] + raw[1];
        close(
            &output.weights.to_vec2::<f32>().unwrap()[0],
            &[2.5 * raw[0] / denominator, 2.5 * raw[1] / denominator],
            1e-6,
        );
    }

    #[test]
    fn forget_gate_matches_safe_lower_bound_formula() {
        let device = Device::Cpu;
        let hidden = Tensor::new(&[[2f32, 3.]], &device).unwrap();
        let f_a = Tensor::new(&[[1f32, 0.]], &device).unwrap();
        let f_b = Tensor::new(&[[1f32], [2.]], &device).unwrap();
        let dt_bias = Tensor::zeros(2, DType::F32, &device).unwrap();
        let a_log = Tensor::zeros(1, DType::F32, &device).unwrap();
        let output = kda_forget_gate(&hidden, &f_a, &f_b, &dt_bias, &a_log, -5.0).unwrap();
        close(
            &output.flatten_all().unwrap().to_vec1().unwrap(),
            &[
                -5.0 / (1.0 + (-2.0f32).exp()),
                -5.0 / (1.0 + (-4.0f32).exp()),
            ],
            1e-6,
        );
    }

    #[test]
    fn cuda_topk_reference_order_preserves_winners_ties_and_signed_zero() {
        assert_eq!(
            cuda_reference_topk_indices(&[3.0, 2.0, 4.0, 1.0], 3).unwrap(),
            [0, 2, 1]
        );
        assert_eq!(
            cuda_reference_topk_indices(&[4.0, 3.0, 4.0, 2.0, 3.0], 4).unwrap(),
            [0, 2, 1, 4]
        );
        assert_eq!(
            cuda_reference_topk_indices(&[0.0, -0.0, -1.0, 1.0], 3).unwrap(),
            [0, 3, 1]
        );
        assert_eq!(
            cuda_reference_topk_indices(&[0.0; 288], 8).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(cuda_reference_topk_indices(&[bad, 0.0], 1).is_err());
        }
        assert!(cuda_reference_topk_indices(&[], 1).is_err());
        assert!(cuda_reference_topk_indices(&[1.0], 0).is_err());
    }

    #[test]
    fn batched_router_materializes_partial_topk_indices() {
        let d = candle_core::Device::Cpu;
        let input = Tensor::zeros((3, 4), DType::F32, &d).unwrap();
        let weight = Tensor::zeros((4, 4), DType::F32, &d).unwrap();
        let bias = Tensor::new(&[0.2f32, -0.2, 0.1, -0.1], &d).unwrap();
        let routed = topk_router(&input, &weight, &bias, 2, 2.5, true).unwrap();
        assert!(routed.indices.is_contiguous());
        assert_eq!(
            routed.indices.to_vec2::<u32>().unwrap(),
            vec![vec![0, 2]; 3]
        );
        assert_eq!(
            routed.weights.to_vec2::<f32>().unwrap(),
            vec![vec![1.25, 1.25]; 3]
        );
    }

    #[test]
    fn single_token_kda_updates_f32_state() {
        let device = Device::Cpu;
        let query = Tensor::new(&[[3f32, 4.]], &device).unwrap();
        let key = Tensor::new(&[[1f32, 0.]], &device).unwrap();
        let value = Tensor::new(&[[2f32, 3.]], &device).unwrap();
        let log_decay = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
        let beta = Tensor::new(&[0.5f32], &device).unwrap();
        let state = Tensor::zeros((1, 2, 2), DType::F32, &device).unwrap();
        let (output, next_state) =
            kda_single_token(&query, &key, &value, &log_decay, &beta, &state).unwrap();

        let inv_sqrt_two = 1.0 / 2.0f32.sqrt();
        close(
            &output.flatten_all().unwrap().to_vec1().unwrap(),
            &[0.6 * inv_sqrt_two, 0.9 * inv_sqrt_two],
            1e-6,
        );
        close(
            &next_state.flatten_all().unwrap().to_vec1().unwrap(),
            &[1.0, 1.5, 0.0, 0.0],
            1e-6,
        );
        assert_eq!(next_state.dtype(), DType::F32);
    }
}
