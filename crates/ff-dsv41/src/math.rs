//! Shared numerics for the V4.1 forward path.
//!
//! Rotary embeddings (with YaRN extrapolation), the block activation
//! quantization round-trips the reference applies in place, the mHC Sinkhorn
//! split, and RMSNorm. Semantics follow `inference/model.py` and
//! `inference/kernel.py` in the checkpoint.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

/// A read guard over a resident tensor's storage; keeps the pages borrowed
/// while a slice is read from them instead of copying the tensor to a Vec.
pub type StorageGuard<'a> = std::sync::RwLockReadGuard<'a, candle_core::Storage>;

/// Borrow a resident tensor without materializing it. The projection must be
/// CPU-resident, contiguous, and F32 (every loader-built weight is).
pub fn resident_f32(tensor: &Tensor) -> Result<(StorageGuard<'_>, usize)> {
    let (guard, layout) = tensor.storage_and_layout();
    anyhow::ensure!(
        layout.is_contiguous() && layout.start_offset() == 0,
        "resident projection must be contiguous"
    );
    Ok((guard, layout.shape().elem_count()))
}

/// The F32 slice behind a guard from `resident_f32`.
pub fn resident_f32_slice<'a>(guard: &'a StorageGuard<'_>) -> Result<&'a [f32]> {
    let candle_core::Storage::Cpu(cpu) = &**guard else {
        anyhow::bail!("resident projection is not CPU storage")
    };
    cpu.as_slice::<f32>().map_err(anyhow::Error::from)
}

pub const FP8_MAX: f32 = 448.0;
pub const FP4_MAX: f32 = 6.0;

/// Rotary frequencies as cos/sin rows, one per position.
///
/// With `original_seq_len > 0` this applies YaRN exactly as the reference
/// does: ramp across `beta_fast..beta_slow`, smooth dimensions keep their
/// frequency and ramped ones are divided by `factor`.
#[allow(clippy::too_many_arguments)]
pub fn precompute_freqs_cis(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f64,
    factor: f64,
    beta_fast: usize,
    beta_slow: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    ensure!(dim.is_multiple_of(2), "rotary dimension {dim} must be even");
    let half = dim / 2;
    let exponent = (0..half as u64)
        .map(|index| 2.0 * index as f64 / dim as f64)
        .collect::<Vec<_>>();
    let mut freqs = exponent
        .into_iter()
        .map(|power| base.powf(-power))
        .collect::<Vec<f64>>();
    if original_seq_len > 0 {
        let corrected = |rotations: f64| {
            dim as f64 * (original_seq_len as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected(beta_fast as f64).floor().max(0.0) as usize;
        let high = corrected(beta_slow as f64).ceil().min(half as f64 - 1.0) as usize;
        let fade = (high as f64 - low as f64).max(1e-3);
        for (index, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((index as f64 - low as f64) / fade).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    let mut rows = Vec::with_capacity(seqlen * half);
    for position in 0..seqlen {
        for freq in &freqs {
            let angle = position as f64 * freq;
            rows.push(angle.cos() as f32);
            rows.push(angle.sin() as f32);
        }
    }
    Tensor::from_vec(rows, (seqlen, half, 2), device).map_err(anyhow::Error::from)
}

/// Rotate adjacent element pairs of the trailing `freqs_cis.shape()[1] * 2`
/// channels by the per-position angles; earlier channels pass through.
/// `inverse` conjugates the rotation, which is how the attention output drops
/// the query's rotation again.
pub fn apply_rotary_emb(x: &Tensor, freqs_cis: &Tensor, inverse: bool) -> Result<Tensor> {
    let dims = x.dims();
    let rotary = freqs_cis.dims()[1] * 2;
    let trailing = *dims.last().context("rotary input must be rank >= 1")?;
    ensure!(
        trailing >= rotary,
        "rotary applies to at most {rotary} channels, found {trailing}"
    );
    let positions = freqs_cis.dims()[0];
    let angles = freqs_cis
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read rotary angles")?;
    let mut values = x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read rotary input")?;
    let channels = values.len() / positions.max(1);
    let rows = values.len().div_ceil(trailing.max(rotary));
    ensure!(
        rows % positions == 0 || positions == 1,
        "rotary batch rows {rows} do not line up with {positions} positions"
    );
    for row in 0..rows {
        let position = if positions == 1 { 0 } else { row % positions };
        let base = row * trailing;
        for pair in 0..rotary / 2 {
            let even = values[base + trailing - rotary + 2 * pair];
            let odd = values[base + trailing - rotary + 2 * pair + 1];
            let cos = angles[(position * rotary / 2 + pair) * 2];
            let sin = if inverse {
                -angles[(position * rotary / 2 + pair) * 2 + 1]
            } else {
                angles[(position * rotary / 2 + pair) * 2 + 1]
            };
            values[base + trailing - rotary + 2 * pair] = even * cos - odd * sin;
            values[base + trailing - rotary + 2 * pair + 1] = even * sin + odd * cos;
        }
    }
    let _ = channels;
    Ok(Tensor::from_vec(values, dims.to_vec(), x.device())
        .map_err(anyhow::Error::from)?
        .to_dtype(x.dtype())?)
}

/// Quantize then dequantize each `block`-element group, E4M3 payload with the
/// scale rounded to a power of two. This is the round trip the reference
/// applies in place to window keys and values.
pub fn fp8_act_round_trip(x: &Tensor, block: usize) -> Result<Tensor> {
    act_round_trip(x, block, Payload::E4M3, ScaleFormat::PowerOfTwo)
}

/// The E2M1 round-trip with E8M0 scales in `block`-element groups (indexer
/// queries and keys) or E4M3 scales (compressed KV latents).
pub fn fp4_act_round_trip(x: &Tensor, block: usize, scale_dtype: DType) -> Result<Tensor> {
    ensure!(
        scale_dtype == DType::F8E8M0 || scale_dtype == DType::F8E4M3,
        "FP4 activation scales must be E8M0 or E4M3"
    );
    let format = if scale_dtype == DType::F8E8M0 {
        ScaleFormat::PowerOfTwo
    } else {
        ScaleFormat::E4M3
    };
    act_round_trip(x, block, Payload::E2M1, format)
}

enum Payload {
    E4M3,
    E2M1,
}

enum ScaleFormat {
    PowerOfTwo,
    E4M3,
}

fn act_round_trip(
    x: &Tensor,
    block: usize,
    payload: Payload,
    format: ScaleFormat,
) -> Result<Tensor> {
    let dims = x.dims();
    let last = dims.last().copied().context("round-trip needs rank >= 1")?;
    ensure!(
        last.is_multiple_of(block),
        "axis of {last} is not a multiple of {block}"
    );
    let mut group_dims = dims.to_vec();
    let last_dim = group_dims.pop().expect("checked above");
    group_dims.push(last_dim / block);
    group_dims.push(block);
    let groups = x.to_dtype(DType::F32)?.reshape(group_dims)?;
    let axis = groups.rank() - 1;
    let amax = groups.abs()?.max_keepdim(axis)?;
    let payload_max = match payload {
        Payload::E4M3 => FP8_MAX,
        Payload::E2M1 => FP4_MAX,
    };
    let floor = match format {
        // A zero group keeps a nonzero scale so the decode never divides by 0.
        ScaleFormat::PowerOfTwo => payload_max * 2f32.powi(-126),
        ScaleFormat::E4M3 => payload_max * 2f32.powi(-9),
    };
    let amax = amax.maximum(&Tensor::full(floor, amax.dims(), x.device())?)?;
    let scales = match format {
        ScaleFormat::PowerOfTwo => {
            let ln2 = std::f64::consts::LN_2;
            (amax
                .affine(1.0 / payload_max as f64, 0.0)?
                .log()?
                .affine(1.0 / ln2, 0.0)?
                .ceil()?
                * ln2)?
                .exp()?
        }
        ScaleFormat::E4M3 => amax
            .affine(1.0 / payload_max as f64, 0.0)?
            .to_dtype(DType::F8E4M3)?
            .to_dtype(DType::F32)?,
    };
    let normalized = groups
        .broadcast_div(&scales)?
        .clamp(-payload_max, payload_max)?;
    let quantized = match payload {
        Payload::E4M3 => normalized.to_dtype(DType::F8E4M3)?.to_dtype(DType::F32)?,
        Payload::E2M1 => round_e2m1(&normalized)?,
    };
    let restored = quantized.broadcast_mul(&scales)?;
    Ok(restored.reshape(dims)?.to_dtype(x.dtype())?)
}

fn round_e2m1(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let values = x
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read E2M1 activation values")?;
    let rounded = values.into_iter().map(round_e2m1_value).collect::<Vec<_>>();
    Tensor::from_vec(rounded, dims, x.device()).map_err(anyhow::Error::from)
}

fn round_e2m1_value(value: f32) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = value.abs();
    // Nearest grid value; exact midpoints take the encoding with the even
    // index, matching the reference float4 cast's round-to-nearest-even.
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, point) in GRID.iter().enumerate() {
        let distance = (magnitude - point).abs();
        if distance < best_distance || (distance == best_distance && index % 2 == 0) {
            best = index;
            best_distance = distance;
        }
    }
    value.signum() * GRID[best]
}

/// The mHC coefficient split: pre and post from the first `2 * streams` mix
/// logits, comb made doubly stochastic by Sinkhorn. Mirrors
/// `hc_split_sinkhorn` in `inference/kernel.py`; ff-glm's
/// `math::sinkhorn::fused_loop` implements the same algorithm.
pub fn hc_split_sinkhorn(
    mixes: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    streams: usize,
    sinkhorn_iters: usize,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mix_width = (2 + streams) * streams;
    let tokens = mixes.dims()[0];
    ensure!(
        mixes.dims() == [tokens, mix_width],
        "mHC mixes must be [tokens, {mix_width}], found {:?}",
        mixes.dims()
    );
    ensure!(
        hc_scale.dims() == [3],
        "hc_scale must hold the pre, post and comb scales"
    );
    ensure!(
        hc_base.dims() == [mix_width],
        "hc_base must hold one bias per mix"
    );
    let eps = eps as f32;
    let device = mixes.device();
    let mixes = mixes
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read mHC mixes")?;
    let hc_scale = hc_scale
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read hc scales")?;
    let hc_base = hc_base
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read hc bases")?;
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());

    let mut pre = Vec::with_capacity(tokens * streams);
    let mut post = Vec::with_capacity(tokens * streams);
    let mut comb = vec![0.0f32; tokens * streams * streams];
    for token in 0..tokens {
        for stream in 0..streams {
            let mix = |slice: usize, offset: usize| -> f32 {
                mixes[token * mix_width + slice * streams + offset]
            };
            let base = |slice: usize, offset: usize| -> f32 { hc_base[slice * streams + offset] };
            pre.push(sigmoid(mix(0, stream) * hc_scale[0] + base(0, stream)) + eps);
            post.push(sigmoid(mix(1, stream) * hc_scale[1] + base(1, stream)) * 2.0);
        }
        let mut matrix = vec![0.0f32; streams * streams];
        for row in 0..streams {
            for column in 0..streams {
                let index = 2 * streams + row * streams + column;
                matrix[row * streams + column] =
                    mixes[token * mix_width + index] * hc_scale[2] + hc_base[index];
            }
        }
        // softmax over each row, then Sinkhorn alternation to doubly stochastic.
        for row in 0..streams {
            let max = matrix[row * streams..(row + 1) * streams]
                .iter()
                .fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let sum: f32 = matrix[row * streams..(row + 1) * streams]
                .iter()
                .map(|value| (value - max).exp())
                .sum();
            for column in 0..streams {
                matrix[row * streams + column] =
                    ((matrix[row * streams + column] - max).exp() / sum) + eps;
            }
        }
        let normalize_columns = |matrix: &mut Vec<f32>| {
            for column in 0..streams {
                let sum: f32 = (0..streams).map(|row| matrix[row * streams + column]).sum();
                for row in 0..streams {
                    matrix[row * streams + column] /= sum + eps;
                }
            }
        };
        let normalize_rows = |matrix: &mut Vec<f32>| {
            for row in 0..streams {
                let sum: f32 = matrix[row * streams..(row + 1) * streams].iter().sum();
                for column in 0..streams {
                    matrix[row * streams + column] /= sum + eps;
                }
            }
        };
        normalize_columns(&mut matrix);
        for _ in 1..sinkhorn_iters {
            normalize_rows(&mut matrix);
            normalize_columns(&mut matrix);
        }
        comb[token * streams * streams..(token + 1) * streams * streams].copy_from_slice(&matrix);
    }
    let pre = Tensor::from_vec(pre, (tokens, streams), device).map_err(anyhow::Error::from)?;
    let post = Tensor::from_vec(post, (tokens, streams), device).map_err(anyhow::Error::from)?;
    let comb =
        Tensor::from_vec(comb, (tokens, streams, streams), device).map_err(anyhow::Error::from)?;
    Ok((pre, post, comb))
}

/// RMSNorm computed in F32 with the checkpoint's own epsilon.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let inner = x.to_dtype(DType::F32)?;
    Ok(
        candle_nn::ops::rms_norm(&inner, &weight.to_dtype(DType::F32)?, eps as f32)?
            .to_dtype(x.dtype())?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_midpoints_round_ties_to_even() {
        // Exact midpoints between adjacent grid values take the encoding
        // with the even index; strict-threshold rounding pushes them upward.
        for (midpoint, expected) in [
            (0.25f32, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
        ] {
            assert_eq!(round_e2m1_value(midpoint), expected, "{midpoint}");
            assert_eq!(round_e2m1_value(-midpoint), -expected, "-{midpoint}");
        }
        assert_eq!(round_e2m1_value(0.6), 0.5);
        assert_eq!(round_e2m1_value(0.9), 1.0);
    }
    use candle_core::Device;

    #[test]
    fn rotary_rotates_pairs_and_inverse_restores() {
        let device = Device::Cpu;
        let freqs = precompute_freqs_cis(4, 3, 0, 10_000.0, 1.0, 32, 1, &device).unwrap();
        assert_eq!(freqs.dims(), [3, 2, 2]);
        let x = Tensor::from_vec(
            vec![
                0.1f32, 1.0, 0.0, 0.0, 1.0, 2.0, //
                0.2, 1.0, 0.0, 0.0, 1.0, 2.0, //
                0.3, 1.0, 0.0, 0.0, 1.0, 2.0,
            ],
            (1, 3, 6),
            &device,
        )
        .unwrap();
        let rotated = apply_rotary_emb(&x, &freqs, false).unwrap();
        let restored = apply_rotary_emb(&rotated, &freqs, true).unwrap();
        let expected = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = restored.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (value, got_value) in expected.iter().zip(got.iter()) {
            assert!((value - got_value).abs() < 1e-5, "{value} vs {got_value}");
        }
    }

    #[test]
    fn rope_frequencies_use_even_exponents_only() {
        // torch computes arange(0, dim, 2)/dim, so dim 4 gives exponents
        // {0, 1/2}, not {0, 1/4}: position 1 of frequency slot 1 must land on
        // angle 10000^-0.5, which differs from the wrong 10000^-0.25.
        let device = Device::Cpu;
        let freqs = precompute_freqs_cis(4, 100, 0, 10_000.0, 1.0, 32, 1, &device).unwrap();
        let values = freqs.to_vec3::<f32>().unwrap();
        // Position 100 of slot 1: correct angle 1.0 rad, wrong exponent gives
        // 100 * 10^-2.5 = 0.316 rad — cos separates them by 0.4.
        let expected = (99.0f64 * 10_000f64.powf(-0.5)).cos() as f32;
        assert!(
            (values[99][1][0] - expected).abs() < 1e-4,
            "{} vs {expected}",
            values[99][1][0]
        );
    }

    #[test]
    fn yarn_fades_only_dimensions_inside_the_transition_band() {
        let device = Device::Cpu;
        let plain = precompute_freqs_cis(8, 2048, 0, 10_000.0, 16.0, 32, 1, &device).unwrap();
        let yarn = precompute_freqs_cis(8, 2048, 65_536, 10_000.0, 16.0, 32, 1, &device).unwrap();
        let plain = plain.to_vec3::<f32>().unwrap();
        let yarn = yarn.to_vec3::<f32>().unwrap();
        // dim 8: the YaRN band covers indices above ~2.5, so index 0 keeps its
        // plain frequency and index 3 is faded toward the factor. The faded
        // frequency is ~1e-3, so the position must be large enough for the
        // angle difference to move cos visibly.
        let position = 2047;
        assert!((yarn[position][0][0] - plain[position][0][0]).abs() < 1e-4);
        assert!((yarn[position][3][0] - plain[position][3][0]).abs() > 1e-2);
    }

    #[test]
    fn fp8_round_trip_snaps_to_the_scaled_e4m3_grid() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![450.0f32, -449.0, 1.0, 2.0], (1, 4), &device).unwrap();
        let round_tripped = fp8_act_round_trip(&x, 4).unwrap();
        let values = round_tripped.to_vec2::<f32>().unwrap()[0].to_vec();
        // amax 450 rounds the scale up to 2.0: 225 snaps to 224 on the E4M3
        // grid, the small values survive at 0.5 and 1.0.
        assert_eq!(values[0], 448.0);
        assert_eq!(values[1], -448.0);
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], 2.0);
    }

    #[test]
    fn fp4_round_trip_snaps_to_the_e2m1_grid() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![0.6f32, 2.6, 4.9, 5.1, -0.3, -1.9, 6.4, 0.1],
            (1, 8),
            &device,
        )
        .unwrap();
        let round_tripped = fp4_act_round_trip(&x, 4, DType::F8E8M0).unwrap();
        let values = round_tripped.to_vec2::<f32>().unwrap()[0].to_vec();
        // Block 1 amax 5.1 keeps scale 1: 0.6 -> 0.5, 2.6 -> 3, 4.9 -> 4,
        // 5.1 -> 6. Block 2 amax 6.4 raises the scale to 2: -0.3 -> 0,
        // -1.9 -> -2, 6.4 -> 6, 0.1 -> 0.
        let expected = [0.5, 3.0, 4.0, 6.0, 0.0, -2.0, 6.0, 0.0];
        for (got, want) in values.iter().zip(expected.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn sinkhorn_output_rows_and_columns_sum_to_one() {
        let device = Device::Cpu;
        let streams = 3;
        let mixes = Tensor::from_vec(
            (0..2 * (2 + streams) * streams)
                .map(|index| index as f32 * 0.25)
                .collect(),
            (2, (2 + streams) * streams),
            &device,
        )
        .unwrap();
        let scale = Tensor::from_vec(vec![0.5f32, 0.7, 0.3], (3,), &device).unwrap();
        let base = Tensor::zeros(((2 + streams) * streams,), DType::F32, &device).unwrap();
        let (pre, post, comb) =
            hc_split_sinkhorn(&mixes, &scale, &base, streams, 20, 1e-6).unwrap();
        assert_eq!(pre.dims(), [2, streams]);
        assert_eq!(post.dims(), [2, streams]);
        assert_eq!(comb.dims(), [2, streams, streams]);
        let rows = comb.sum(2).unwrap().to_vec2::<f32>().unwrap();
        let columns = comb.sum(1).unwrap().to_vec2::<f32>().unwrap();
        for token in 0..2 {
            for value in &rows[token] {
                assert!((value - 1.0).abs() < 1e-3, "row sum {value}");
            }
            for value in &columns[token] {
                assert!((value - 1.0).abs() < 1e-3, "column sum {value}");
            }
        }
    }
}
