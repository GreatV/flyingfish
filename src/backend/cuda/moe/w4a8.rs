//! Routed-expert W4A8 decode: E2M1 weights with one E8M0 scale per 32
//! elements of each row against an E4M3-quantized activation
//! (kernels/moe_ffn.cu).
//! Semantics per the DeepSeek-V4.1-Flash model definition (publisher
//! inference/model.py Expert.forward, kernel.py act_quant and fp4_gemm):
//!   - the packed byte at column j carries the element 2j in its low nibble
//!     and the element 2j+1 in its high nibble; the E2M1 code is one sign
//!     bit, two exponent bits and one mantissa bit;
//!   - the activation is quantized dynamically: the amax over 32 elements is
//!     floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the
//!     codes are E4M3 round-to-nearest-even;
//!   - the 32-element dot is scaled by the activation and the weight scale
//!     outside the dot;
//!   - the gate branch is clamped from above, the up branch on both sides, the
//!     SiLU stays in FP32 and multiplies the up branch in FP32;
//!   - the routed weight multiplies that FP32 product and one BF16 cast makes
//!     the down-input row (RoutedWeightBoundary::BeforeDownQuantization).
#[cfg(test)]
use crate::backend::cuda::contracts::dense_layout;
use crate::backend::cuda::contracts::{Tensor, TensorMut, no_overlap};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

/// Elements per E8M0 scale along K.
pub const W4A8_SCALE_BLOCK: u32 = 32;
/// Threads per W4A8 block; one warp owns one output row.
const W4A8_THREADS: u32 = 256;
/// Output rows per W4A8 block.
const W4A8_ROWS: u32 = 8;
/// Dynamic shared bytes one W4A8 block may request below the 48 KB limit.
const W4A8_SHARED_LIMIT: u64 = 48 * 1024 - 1024;

/// One routed expert's W4A8 weights: E2M1 codes packed two values per byte
/// with one E8M0 scale per 32 elements of each row.
#[derive(Clone, Copy)]
pub struct W4A8ExpertWeights {
    /// `[intermediate, hidden/2]` E2M1 gate codes.
    pub gate_codes: Tensor<u8, 2>,
    /// `[intermediate, hidden/32]` E8M0 gate scales.
    pub gate_scales: Tensor<u8, 2>,
    /// `[intermediate, hidden/2]` E2M1 up codes.
    pub up_codes: Tensor<u8, 2>,
    /// `[intermediate, hidden/32]` E8M0 up scales.
    pub up_scales: Tensor<u8, 2>,
    /// `[hidden, intermediate/2]` E2M1 down codes.
    pub down_codes: Tensor<u8, 2>,
    /// `[hidden, intermediate/32]` E8M0 down scales.
    pub down_scales: Tensor<u8, 2>,
}

/// Inputs of one routed expert's W4A8 decode contribution.
#[derive(Clone, Copy)]
pub struct W4A8Inputs<'a> {
    /// `[hidden]` BF16 token activation.
    pub x: Tensor<bf16, 1>,
    /// The expert's packed weights.
    pub weights: &'a W4A8ExpertWeights,
    /// The routed weight, applied before the down-input quantization.
    pub route_weight: bf16,
    /// The SwiGLU clamp limit.
    pub swiglu_limit: f32,
}

/// Device-side routed dispatch of one decode batch: the ids and route weights
/// stay in device memory and the per-expert launches are sequenced from the
/// host-known `rows`/`top_k` bounds only. `slots` is `[experts, 6]` addresses;
/// a zero `slot_valid` entry marks an unresident expert and traps the launch.
/// Derived extents of one call.
#[derive(Clone, Copy, Debug)]
#[cfg(test)]
struct W4A8Dims {
    hidden: u32,
    intermediate: u32,
}

fn align16(bytes: u64) -> u64 {
    bytes.div_ceil(16) * 16
}

/// Byte offsets of the scratch regions and the total size: the activation
/// codes, the activation scales, the BF16 hidden state, the down-input codes
/// and the down-input scales.
fn scratch_layout(hidden: u32, intermediate: u32) -> ([u64; 5], u64) {
    let codes = 0u64;
    let scales = align16(u64::from(hidden));
    let state = scales + align16(u64::from(hidden / W4A8_SCALE_BLOCK));
    let down_codes = state + align16(2 * u64::from(intermediate));
    let down_scales = down_codes + align16(u64::from(intermediate));
    let total = down_scales + align16(u64::from(intermediate / W4A8_SCALE_BLOCK));
    ([codes, scales, state, down_codes, down_scales], total)
}

/// Scratch bytes of one W4A8 call: the activation codes and scales, the BF16
/// hidden state and the down-input codes and scales.
pub fn scratch_bytes(hidden: u32, intermediate: u32) -> u64 {
    scratch_layout(hidden, intermediate).1
}

#[cfg(test)]
fn require_dense<T: Copy, const R: usize>(
    tensor: &Tensor<T, R>,
    name: &str,
    shape: [u32; R],
) -> Result<()> {
    tensor.validate()?;
    let (stride, _) = dense_layout(shape)?;
    ensure!(
        tensor.shape == shape && tensor.stride == stride,
        "w4a8 expert {name} shape {:?} stride {:?} does not match the expected dense {:?}",
        tensor.shape,
        tensor.stride,
        shape
    );
    Ok(())
}

/// Validate the call extents against each other.
#[cfg(test)]
fn validate(
    input: W4A8Inputs<'_>,
    scratch: &Tensor<u8, 1>,
    out: &Tensor<bf16, 1>,
) -> Result<W4A8Dims> {
    input.x.validate()?;
    let hidden = input.x.shape[0];
    ensure!(
        input.x.stride == [1],
        "w4a8 expert x must be contiguous [hidden] (shape {:?}, stride {:?})",
        input.x.shape,
        input.x.stride
    );
    ensure!(
        hidden >= W4A8_SCALE_BLOCK && hidden.is_multiple_of(W4A8_SCALE_BLOCK),
        "w4a8 expert hidden {hidden} must be a positive multiple of {W4A8_SCALE_BLOCK}"
    );
    let intermediate = input.weights.gate_codes.shape[0];
    ensure!(
        intermediate >= W4A8_SCALE_BLOCK && intermediate.is_multiple_of(W4A8_SCALE_BLOCK),
        "w4a8 expert intermediate {intermediate} must be a positive multiple of {W4A8_SCALE_BLOCK}"
    );
    let weights = input.weights;
    require_dense(
        &weights.gate_codes,
        "gate_codes",
        [intermediate, hidden / 2],
    )?;
    require_dense(
        &weights.gate_scales,
        "gate_scales",
        [intermediate, hidden / W4A8_SCALE_BLOCK],
    )?;
    require_dense(&weights.up_codes, "up_codes", [intermediate, hidden / 2])?;
    require_dense(
        &weights.up_scales,
        "up_scales",
        [intermediate, hidden / W4A8_SCALE_BLOCK],
    )?;
    require_dense(
        &weights.down_codes,
        "down_codes",
        [hidden, intermediate / 2],
    )?;
    require_dense(
        &weights.down_scales,
        "down_scales",
        [hidden, intermediate / W4A8_SCALE_BLOCK],
    )?;
    ensure!(
        input.route_weight.is_finite(),
        "w4a8 expert route weight is not finite"
    );
    ensure!(
        input.swiglu_limit.is_finite() && input.swiglu_limit > 0.0,
        "w4a8 expert swiglu limit {} must be positive and finite",
        input.swiglu_limit
    );
    out.validate()?;
    ensure!(
        out.shape == [hidden] && out.stride == [1],
        "w4a8 expert out must be contiguous [hidden] (shape {:?}, stride {:?})",
        out.shape,
        out.stride
    );
    no_overlap(&input.x, out, "x", "out")?;
    no_overlap(&weights.gate_codes, out, "gate_codes", "out")?;
    no_overlap(&weights.up_codes, out, "up_codes", "out")?;
    no_overlap(&weights.down_codes, out, "down_codes", "out")?;
    no_overlap(scratch, out, "scratch", "out")?;
    scratch.validate()?;
    ensure!(
        scratch.span_bytes >= scratch_bytes(hidden, intermediate),
        "w4a8 expert scratch holds {} bytes, the call needs {}",
        scratch.span_bytes,
        scratch_bytes(hidden, intermediate)
    );
    let shared_gate_up = u64::from(hidden) + u64::from(hidden / W4A8_SCALE_BLOCK);
    let shared_down = u64::from(intermediate) + u64::from(intermediate / W4A8_SCALE_BLOCK);
    ensure!(
        shared_gate_up <= W4A8_SHARED_LIMIT && shared_down <= W4A8_SHARED_LIMIT,
        "w4a8 expert shared staging {shared_gate_up}/{shared_down} B exceeds the {W4A8_SHARED_LIMIT} B limit (hidden {hidden}, intermediate {intermediate})"
    );
    Ok(W4A8Dims {
        hidden,
        intermediate,
    })
}

/// One routed expert's W4A8 decode contribution.
pub struct MoeExpertW4A8 {
    act_quant: CudaFunction,
    #[cfg(test)]
    gate_up: CudaFunction,
    #[cfg(test)]
    down: CudaFunction,
}

impl MoeExpertW4A8 {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            act_quant: module.load_function("moe_act_quant_e4m0")?,
            #[cfg(test)]
            gate_up: module.load_function("moe_ffn_gate_up_e2m1_e8m0")?,
            #[cfg(test)]
            down: module.load_function("moe_ffn_down_e2m1_e8m0")?,
        })
    }

    /// The contribution of one routed expert to one token.
    ///
    /// `scratch` holds the activation codes and scales, the BF16 hidden state
    /// and the down-input codes and scales; `scratch_bytes` sizes it.
    #[cfg(test)]
    pub fn run(
        &self,
        stream: &Arc<CudaStream>,
        input: W4A8Inputs<'_>,
        scratch: &Tensor<u8, 1>,
        out: &mut TensorMut<bf16, 1>,
    ) -> Result<()> {
        let dims = validate(input, scratch, &out.0)?;
        let hidden = dims.hidden;
        let intermediate = dims.intermediate;
        let (offsets, _) = scratch_layout(hidden, intermediate);
        let codes = scratch.ptr + offsets[0];
        let scales = scratch.ptr + offsets[1];
        let state = scratch.ptr + offsets[2];
        let down_codes = scratch.ptr + offsets[3];
        let down_scales = scratch.ptr + offsets[4];
        let groups = hidden / W4A8_SCALE_BLOCK;
        let shared_gate_up = u64::from(hidden) + u64::from(groups);
        let shared_down = u64::from(intermediate) + u64::from(intermediate / W4A8_SCALE_BLOCK);
        let weight_value = input.route_weight.to_f32();
        unsafe {
            stream
                .launch_builder(&self.act_quant)
                .arg(&input.x.ptr)
                .arg(&codes)
                .arg(&scales)
                .arg(&(hidden as i32))
                .launch(LaunchConfig {
                    grid_dim: (groups.div_ceil(W4A8_ROWS), 1, 1),
                    block_dim: (W4A8_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.gate_up)
                .arg(&codes)
                .arg(&scales)
                .arg(&input.weights.gate_codes.ptr)
                .arg(&input.weights.gate_scales.ptr)
                .arg(&input.weights.up_codes.ptr)
                .arg(&input.weights.up_scales.ptr)
                .arg(&state)
                .arg(&(hidden as i32))
                .arg(&(intermediate as i32))
                .arg(&input.swiglu_limit)
                .arg(&weight_value)
                .launch(LaunchConfig {
                    grid_dim: (intermediate.div_ceil(W4A8_ROWS), 1, 1),
                    block_dim: (W4A8_THREADS, 1, 1),
                    shared_mem_bytes: shared_gate_up as u32,
                })?;
            stream
                .launch_builder(&self.act_quant)
                .arg(&state)
                .arg(&down_codes)
                .arg(&down_scales)
                .arg(&(intermediate as i32))
                .launch(LaunchConfig {
                    grid_dim: ((intermediate / W4A8_SCALE_BLOCK).div_ceil(W4A8_ROWS), 1, 1),
                    block_dim: (W4A8_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.down)
                .arg(&down_codes)
                .arg(&down_scales)
                .arg(&input.weights.down_codes.ptr)
                .arg(&input.weights.down_scales.ptr)
                .arg(&out.0.ptr)
                .arg(&(hidden as i32))
                .arg(&(intermediate as i32))
                .launch(LaunchConfig {
                    grid_dim: (hidden.div_ceil(W4A8_ROWS), 1, 1),
                    block_dim: (W4A8_THREADS, 1, 1),
                    shared_mem_bytes: shared_down as u32,
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::Device;
    use crate::backend::cuda::moe::testing;
    use crate::fp8::{decode_e4m3, encode_e4m3};
    use anyhow::{Result, ensure};
    use std::mem::size_of;

    /// DeepSeek-V4.1-Flash layer-0 expert shapes: 5120 hidden, 2304 intermediate.
    const HIDDEN: usize = 5120;
    const INTERMEDIATE: usize = 2304;
    /// The DeepSeek-V4.1-Flash SwiGLU clamp limit.
    const LIMIT: f32 = 10.0;
    /// Routed experts of one decode token.
    const TOP_K: usize = 6;

    /// BF16 ULP distance of two outputs; 1 is the accepted BF16 rounding step.
    const OUTPUT_ULP_TOL: u32 = 1;
    /// Absolute tolerance of the outputs whose BF16 ULP distance exceeds one:
    /// the down dot accumulates 288 chunk sums of 8 terms in FP32 against the
    /// FP64 oracle, and the host simulation of the device loop measures that
    /// difference at 8.9e-4 for the fixture's output scale of 8e3..1.2e4;
    /// 1e-1 covers it with a 110x margin.
    const OUTPUT_ABS_TOL: f64 = 1e-1;

    /// E2M1 magnitudes by nibble: sign, two exponent bits, one mantissa bit.
    const E2M1: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// Golden-ratio stride: the phase sequence never repeats.
    fn pattern(seed: usize, phase: f64) -> f64 {
        (seed as f64 * 0.6180339887498949 + phase).sin()
    }

    /// E8M0 scale value 2^(code - 127); 0xff is NaN.
    fn e8m0_value(code: u8) -> f64 {
        if code == 0xff {
            return f64::NAN;
        }
        let exponent = i64::from(code) - 127;
        f64::from_bits(((exponent + 1023) as u64) << 52)
    }

    /// ceil(log2(v)) of a positive normal float, from its bit pattern.
    fn log2_ceil(value: f32) -> i32 {
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
        exponent + i32::from((bits & 0x7FFFFF) != 0)
    }

    /// Distance in representable BF16 values between two outputs.
    fn bf16_ulp_distance(a: bf16, b: bf16) -> u32 {
        fn key(value: bf16) -> u32 {
            let bits = u32::from(value.to_bits());
            if bits & 0x8000 != 0 {
                (!bits) & 0xffff
            } else {
                bits | 0x8000
            }
        }
        key(a).abs_diff(key(b))
    }

    /// One routed expert's fixture.
    struct Fixture {
        x: Vec<bf16>,
        gate_codes: Vec<u8>,
        gate_scales: Vec<u8>,
        up_codes: Vec<u8>,
        up_scales: Vec<u8>,
        down_codes: Vec<u8>,
        down_scales: Vec<u8>,
        route_weight: bf16,
    }

    /// The nibble of the logical element k of row j. The scrambled index
    /// repeats no period with the 32-element scale block, so a wrong element
    /// index or a wrong block alignment changes the value.
    fn nibble_at(row: usize, k: usize) -> u8 {
        let mut hash = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (row as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 29;
        hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 32;
        (hash % 16) as u8
    }

    /// The E8M0 scale code of block b of row j.
    fn scale_at(row: usize, block: usize) -> u8 {
        124 + ((row * 3 + block * 5) % 4) as u8
    }

    /// Packed E2M1 row: the element 2i is in the low nibble of byte i.
    fn packed_row(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let mut codes = vec![0u8; rows * cols / 2];
        for row in 0..rows {
            for i in 0..cols / 2 {
                let k = 2 * i;
                codes[row * cols / 2 + i] =
                    nibble_at(row + seed, k) | (nibble_at(row + seed, k + 1) << 4);
            }
        }
        codes
    }

    /// E8M0 scale rows: one scale per 32 elements of each row.
    fn scale_rows(rows: usize, blocks: usize, seed: usize) -> Vec<u8> {
        let mut scales = vec![0u8; rows * blocks];
        for row in 0..rows {
            for block in 0..blocks {
                scales[row * blocks + block] = scale_at(row + seed, block);
            }
        }
        scales
    }

    /// The fixture of one routed expert.
    fn fixture(expert: usize) -> Fixture {
        let x: Vec<bf16> = (0..HIDDEN)
            .map(|i| bf16::from_f32((pattern(i + expert * 17, 0.11) * 0.2) as f32))
            .collect();
        Fixture {
            x,
            gate_codes: packed_row(INTERMEDIATE, HIDDEN, expert),
            gate_scales: scale_rows(INTERMEDIATE, HIDDEN / 32, expert),
            up_codes: packed_row(INTERMEDIATE, HIDDEN, expert + 3),
            up_scales: scale_rows(INTERMEDIATE, HIDDEN / 32, expert + 3),
            down_codes: packed_row(HIDDEN, INTERMEDIATE, expert + 5),
            down_scales: scale_rows(HIDDEN, INTERMEDIATE / 32, expert + 5),
            route_weight: bf16::from_f32(0.5 + 0.05 * expert as f32),
        }
    }

    /// The fixture with the packed nibbles swapped.
    fn swapped_nibbles(fx: &Fixture) -> Fixture {
        let swap = |codes: &[u8]| -> Vec<u8> {
            codes
                .iter()
                .map(|byte| (byte >> 4) | ((byte & 0x0F) << 4))
                .collect()
        };
        Fixture {
            x: fx.x.clone(),
            gate_codes: swap(&fx.gate_codes),
            gate_scales: fx.gate_scales.clone(),
            up_codes: swap(&fx.up_codes),
            up_scales: fx.up_scales.clone(),
            down_codes: swap(&fx.down_codes),
            down_scales: fx.down_scales.clone(),
            route_weight: fx.route_weight,
        }
    }

    /// The fixture whose weight scales are read one block late.
    fn shifted_scales(fx: &Fixture) -> Fixture {
        let shift = |scales: &[u8], blocks: usize| -> Vec<u8> {
            let mut out = scales.to_vec();
            for row in 0..out.len() / blocks {
                for block in 0..blocks {
                    out[row * blocks + block] = scales[row * blocks + (block + 1) % blocks];
                }
            }
            out
        };
        Fixture {
            x: fx.x.clone(),
            gate_codes: fx.gate_codes.clone(),
            gate_scales: shift(&fx.gate_scales, HIDDEN / 32),
            up_codes: fx.up_codes.clone(),
            up_scales: shift(&fx.up_scales, HIDDEN / 32),
            down_codes: fx.down_codes.clone(),
            down_scales: shift(&fx.down_scales, INTERMEDIATE / 32),
            route_weight: fx.route_weight,
        }
    }

    /// The packed weight of the logical element k of row j: the element 2i is
    /// in the low nibble of byte i and the element 2i+1 in its high nibble.
    fn packed_weight(codes: &[u8], row_bytes: usize, row: usize, k: usize) -> f64 {
        let byte = codes[row * row_bytes + k / 2];
        let nibble = if k.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        E2M1[nibble as usize]
    }

    /// Activation quantization of one row: per 32 elements the amax is
    /// floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the codes
    /// are E4M3. The scales are per 32-element block along K.
    fn quantize(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / 32);
        for group in 0..values.len() / 32 {
            let block = &values[group * 32..group * 32 + 32];
            let mut amax = 0.0f32;
            for value in block {
                amax = amax.max(value.abs());
            }
            let amax = amax.max(1e-4);
            let exponent = log2_ceil(amax * (1.0f32 / 448.0f32));
            let scale = f32::from_bits(((exponent + 127) as u32) << 23);
            for value in block {
                codes.push(encode_e4m3((*value / scale).clamp(-448.0, 448.0)));
            }
            scales.push((exponent + 127) as u8);
        }
        (codes, scales)
    }

    /// FP64 reference of the W4A8 expert path.
    struct Oracle {
        hidden: usize,
        intermediate: usize,
        limit: f64,
        clamp: bool,
    }

    impl Oracle {
        fn new(hidden: usize, intermediate: usize, limit: f64) -> Self {
            Self {
                hidden,
                intermediate,
                limit,
                clamp: true,
            }
        }

        /// The gate and up dots before the clamp. The activation scale of a
        /// 32-element block multiplies that block's dot once.
        fn gate_up(&self, fx: &Fixture) -> (Vec<f64>, Vec<f64>) {
            let x: Vec<f32> = fx.x.iter().map(|value| value.to_f32()).collect();
            let (x_codes, x_scales) = quantize(&x);
            let blocks = self.hidden / 32;
            let row_bytes = self.hidden / 2;
            let mut gates = vec![0.0f64; self.intermediate];
            let mut ups = vec![0.0f64; self.intermediate];
            for (j, (gate, up)) in gates.iter_mut().zip(ups.iter_mut()).enumerate() {
                let mut gate_sum = 0.0f64;
                let mut up_sum = 0.0f64;
                for (block, block_scale) in x_scales.iter().enumerate() {
                    let mut gate_block = 0.0f64;
                    let mut up_block = 0.0f64;
                    for k in 0..32 {
                        let index = block * 32 + k;
                        let a = f64::from(decode_e4m3(x_codes[index]));
                        gate_block += packed_weight(&fx.gate_codes, row_bytes, j, index) * a;
                        up_block += packed_weight(&fx.up_codes, row_bytes, j, index) * a;
                    }
                    let act_scale = e8m0_value(*block_scale);
                    gate_sum +=
                        gate_block * act_scale * e8m0_value(fx.gate_scales[j * blocks + block]);
                    up_sum += up_block * act_scale * e8m0_value(fx.up_scales[j * blocks + block]);
                }
                *gate = gate_sum;
                *up = up_sum;
            }
            (gates, ups)
        }

        /// The BF16 down-input row after the clamped SwiGLU and the routed
        /// weight: FP32 silu times the up branch, the weight in FP32, one cast
        /// (model definition Expert.forward).
        fn hidden_state(&self, fx: &Fixture) -> Vec<f64> {
            let (gates, ups) = self.gate_up(fx);
            gates
                .iter()
                .zip(ups.iter())
                .map(|(gate, up)| {
                    let g = f64::from(bf16::from_f32(*gate as f32).to_f32());
                    let u = f64::from(bf16::from_f32(*up as f32).to_f32());
                    let (g, u) = if self.clamp {
                        (g.min(self.limit), u.clamp(-self.limit, self.limit))
                    } else {
                        (g, u)
                    };
                    let silu = g / (1.0 + (-g).exp());
                    let product = silu as f32 * u as f32;
                    f64::from(bf16::from_f32(product * fx.route_weight.to_f32()).to_f32())
                })
                .collect()
        }

        /// The expert contribution of one token.
        fn forward(&self, fx: &Fixture) -> Vec<f64> {
            let state = self.hidden_state(fx);
            let scaled: Vec<f32> = state.iter().map(|value| *value as f32).collect();
            let (codes, scales) = quantize(&scaled);
            let blocks = self.intermediate / 32;
            let row_bytes = self.intermediate / 2;
            let mut out = vec![0.0f64; self.hidden];
            for (d, slot) in out.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (block, block_scale) in scales.iter().enumerate() {
                    let mut sum = 0.0f64;
                    for k in 0..32 {
                        let index = block * 32 + k;
                        let a = f64::from(decode_e4m3(codes[index]));
                        sum += packed_weight(&fx.down_codes, row_bytes, d, index) * a;
                    }
                    acc += sum
                        * e8m0_value(*block_scale)
                        * e8m0_value(fx.down_scales[d * blocks + block]);
                }
                *slot = f64::from(bf16::from_f32(acc as f32).to_f32());
            }
            out
        }
    }

    /// Tensor view over host-side storage, for the validation test.
    fn fake<T: Copy, const R: usize>(shape: [u32; R], ptr: u64) -> Tensor<T, R> {
        let (stride, count) = dense_layout(shape).expect("dense layout");
        unsafe {
            Tensor::new(ptr, shape, stride, count * size_of::<T>() as u64).expect("fake tensor")
        }
    }

    fn fake_mut<T: Copy, const R: usize>(shape: [u32; R], ptr: u64) -> TensorMut<T, R> {
        let (stride, count) = dense_layout(shape).expect("dense layout");
        unsafe {
            TensorMut::new(ptr, shape, stride, count * size_of::<T>() as u64).expect("fake tensor")
        }
    }

    #[test]
    fn w4a8_e2m1_nibble_order_matches_the_published_table() {
        // The E2M1 magnitudes of the published table (model definition,
        // convert.py): sign, two exponent bits and one mantissa bit.
        assert_eq!(E2M1[0], 0.0);
        assert_eq!(E2M1[1], 0.5);
        assert_eq!(E2M1[2], 1.0);
        assert_eq!(E2M1[3], 1.5);
        assert_eq!(E2M1[4], 2.0);
        assert_eq!(E2M1[5], 3.0);
        assert_eq!(E2M1[6], 4.0);
        assert_eq!(E2M1[7], 6.0);
        assert_eq!(E2M1[9], -0.5);
        assert_eq!(E2M1[15], -6.0);
        // The packed byte at column i carries the element 2i in its low nibble
        // and the element 2i+1 in its high nibble (model definition,
        // convert.py).
        let fx = fixture(0);
        for row in 0..4 {
            for i in 0..64 {
                let byte = fx.gate_codes[row * HIDDEN / 2 + i];
                let low = byte & 0x0F;
                let high = byte >> 4;
                assert_eq!(low as usize, nibble_at(row, 2 * i) as usize);
                assert_eq!(high as usize, nibble_at(row, 2 * i + 1) as usize);
                // The oracle reads the same byte the same way.
                for k in 2 * i..2 * i + 2 {
                    assert_eq!(
                        packed_weight(&fx.gate_codes, HIDDEN / 2, row, k),
                        E2M1[nibble_at(row, k) as usize],
                        "w4a8 oracle decodes the element {k} of row {row} wrongly"
                    );
                }
            }
        }
        let mut seen = [false; 16];
        for k in 0..256 {
            seen[nibble_at(0, k) as usize] = true;
        }
        assert!(
            seen.iter().all(|hit| *hit),
            "w4a8 fixture does not cover every E2M1 nibble"
        );
        // The two nibbles of a byte differ for most elements, so a swapped
        // packing changes the weights.
        let differing = (0..256)
            .filter(|i| nibble_at(0, 2 * i) != nibble_at(0, 2 * i + 1))
            .count();
        assert!(
            differing * 10 >= 256 * 9,
            "w4a8 fixture packs too many equal nibbles ({differing} of 256)"
        );
    }

    #[test]
    fn w4a8_activation_scale_follows_the_model_definition() {
        // amax 1 -> scale 2^-8, amax 448 -> 1, amax 1e-4 -> 2^-22.
        assert_eq!(log2_ceil(1.0f32 * (1.0f32 / 448.0f32)), -8);
        assert_eq!(log2_ceil(448.0f32 * (1.0f32 / 448.0f32)), 0);
        assert_eq!(log2_ceil(1e-4f32 * (1.0f32 / 448.0f32)), -22);
        assert_eq!(e8m0_value(127), 1.0);
        assert_eq!(e8m0_value(119), 2.0f64.powi(-8));
        // The E4M3 encoder is round-to-nearest-even on the E4M3 grid.
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(-1.0), 0xb8);
        assert_eq!(encode_e4m3(448.0), 0x7e);
        assert_eq!(encode_e4m3(-448.0), 0xfe);
        assert_eq!(encode_e4m3(1.0 / 512.0), 0x01);
        assert_eq!(encode_e4m3(0.0), 0x00);
    }

    #[test]
    fn w4a8_validation_names_every_failed_extent() {
        let hidden = 256u32;
        let intermediate = 128u32;
        // The synthetic buffers are spaced so that no two of them overlap.
        let weights = W4A8ExpertWeights {
            gate_codes: fake::<u8, 2>([intermediate, hidden / 2], 0x1000),
            gate_scales: fake::<u8, 2>([intermediate, hidden / 32], 0x5000),
            up_codes: fake::<u8, 2>([intermediate, hidden / 2], 0x6000),
            up_scales: fake::<u8, 2>([intermediate, hidden / 32], 0xA000),
            down_codes: fake::<u8, 2>([hidden, intermediate / 2], 0xB000),
            down_scales: fake::<u8, 2>([hidden, intermediate / 32], 0xF000),
        };
        let x = fake::<bf16, 1>([hidden], 0x10000);
        let out = fake_mut::<bf16, 1>([hidden], 0x11000);
        let scratch = fake_mut::<u8, 1>([scratch_bytes(hidden, intermediate) as u32], 0x12000);
        let input = W4A8Inputs {
            x,
            weights: &weights,
            route_weight: bf16::from_f32(0.5),
            swiglu_limit: LIMIT,
        };
        validate(input, &scratch.0, &out.0).expect("valid w4a8 extents");
        let check = |error: Result<W4A8Dims>, expected: &str| {
            let error = error
                .expect_err("w4a8 validation accepted a bad extent")
                .to_string();
            assert!(
                error.contains(expected),
                "w4a8 validation error {error:?} does not name {expected}"
            );
        };
        let mut bad = input;
        bad.x = fake::<bf16, 1>([hidden - 1], 0x10000);
        check(validate(bad, &scratch.0, &out.0), "hidden");
        let short_gate = W4A8ExpertWeights {
            gate_codes: fake::<u8, 2>([intermediate, hidden / 4], 0x1000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &short_gate;
        check(validate(bad, &scratch.0, &out.0), "gate_codes");
        let short_scales = W4A8ExpertWeights {
            gate_scales: fake::<u8, 2>([intermediate, hidden / 64], 0x5000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &short_scales;
        check(validate(bad, &scratch.0, &out.0), "gate_scales");
        let long_down = W4A8ExpertWeights {
            down_codes: fake::<u8, 2>([hidden, intermediate / 2 + 8], 0xB000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &long_down;
        check(validate(bad, &scratch.0, &out.0), "down_codes");
        let mut bad = input;
        bad.route_weight = bf16::from_f32(f32::NAN);
        check(validate(bad, &scratch.0, &out.0), "route weight");
        let mut bad = input;
        bad.swiglu_limit = 0.0;
        check(validate(bad, &scratch.0, &out.0), "swiglu limit");
        let short = fake_mut::<u8, 1>([16], 0x12000);
        check(validate(input, &short.0, &out.0), "scratch");
        let bad_out = fake_mut::<bf16, 1>([hidden + 32], 0x11000);
        check(validate(input, &scratch.0, &bad_out.0), "out");
    }

    #[test]
    fn w4a8_swapped_nibbles_change_the_contribution() {
        let fx = fixture(0);
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let straight = oracle.forward(&fx);
        let swapped = oracle.forward(&swapped_nibbles(&fx));
        assert_ne!(straight, swapped, "w4a8 nibble-order control escaped");
    }

    #[test]
    fn w4a8_shifted_scale_blocks_change_the_contribution() {
        let fx = fixture(1);
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let straight = oracle.forward(&fx);
        let shifted = oracle.forward(&shifted_scales(&fx));
        assert_ne!(
            straight, shifted,
            "w4a8 scale-block-alignment control escaped"
        );
    }

    #[test]
    fn w4a8_unclamped_swiglu_changes_the_contribution() {
        let fx = fixture(2);
        let clamped = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let mut unclamped = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        unclamped.clamp = false;
        // The fixture must straddle the clamp: the gate branch above the
        // limit, the gate branch below it and the up branch on both sides.
        let (gates, ups) = clamped.gate_up(&fx);
        let limit = LIMIT as f64;
        assert!(
            gates.iter().any(|gate| *gate > limit),
            "w4a8 fixture gate never exceeds the clamp limit"
        );
        assert!(
            gates.iter().any(|gate| *gate < -limit),
            "w4a8 fixture gate never falls below the clamp limit"
        );
        assert!(
            ups.iter().any(|up| up.abs() > limit),
            "w4a8 fixture up never exceeds the clamp limit"
        );
        assert_ne!(
            clamped.forward(&fx),
            unclamped.forward(&fx),
            "w4a8 clamped-swiglu control escaped"
        );
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn w4a8_expert_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = MoeExpertW4A8::new(&device.ctx)?;
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let bytes = scratch_bytes(HIDDEN as u32, INTERMEDIATE as u32) as usize;
        for expert in 0..TOP_K {
            let fx = fixture(expert);
            let expected = oracle.forward(&fx);
            let xb = testing::upload(&device.upload, fx.x.clone())?;
            let gb = testing::upload(&device.upload, fx.gate_codes.clone())?;
            let gs = testing::upload(&device.upload, fx.gate_scales.clone())?;
            let ub = testing::upload(&device.upload, fx.up_codes.clone())?;
            let us = testing::upload(&device.upload, fx.up_scales.clone())?;
            let db = testing::upload(&device.upload, fx.down_codes.clone())?;
            let ds = testing::upload(&device.upload, fx.down_scales.clone())?;
            let wrong = swapped_nibbles(&fx);
            let wb = testing::upload(&device.upload, wrong.gate_codes)?;
            device.finish_upload()?;
            let out = testing::zeros::<bf16>(s, HIDDEN)?;
            let wrong_out = testing::zeros::<bf16>(s, HIDDEN)?;
            let scratch = testing::zeros::<u8>(s, bytes)?;
            let x = testing::view(&xb, s, [HIDDEN as u32], [1])?;
            let gate_codes = testing::view(
                &gb,
                s,
                [INTERMEDIATE as u32, (HIDDEN / 2) as u32],
                [(HIDDEN / 2) as u64, 1],
            )?;
            let gate_scales = testing::view(
                &gs,
                s,
                [INTERMEDIATE as u32, (HIDDEN / 32) as u32],
                [(HIDDEN / 32) as u64, 1],
            )?;
            let up_codes = testing::view(
                &ub,
                s,
                [INTERMEDIATE as u32, (HIDDEN / 2) as u32],
                [(HIDDEN / 2) as u64, 1],
            )?;
            let up_scales = testing::view(
                &us,
                s,
                [INTERMEDIATE as u32, (HIDDEN / 32) as u32],
                [(HIDDEN / 32) as u64, 1],
            )?;
            let down_codes = testing::view(
                &db,
                s,
                [HIDDEN as u32, (INTERMEDIATE / 2) as u32],
                [(INTERMEDIATE / 2) as u64, 1],
            )?;
            let down_scales = testing::view(
                &ds,
                s,
                [HIDDEN as u32, (INTERMEDIATE / 32) as u32],
                [(INTERMEDIATE / 32) as u64, 1],
            )?;
            let wrong_codes = testing::view(
                &wb,
                s,
                [INTERMEDIATE as u32, (HIDDEN / 2) as u32],
                [(HIDDEN / 2) as u64, 1],
            )?;
            let mut out_view = testing::mut_view(&out, s, [HIDDEN as u32], [1])?;
            let mut wrong_view = testing::mut_view(&wrong_out, s, [HIDDEN as u32], [1])?;
            let scratch_view = testing::mut_view(&scratch, s, [bytes as u32], [1])?;
            let weights = W4A8ExpertWeights {
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                down_codes,
                down_scales,
            };
            let input = W4A8Inputs {
                x,
                weights: &weights,
                route_weight: fx.route_weight,
                swiglu_limit: LIMIT,
            };
            ops.run(s, input, &scratch_view.0, &mut out_view)?;
            let mut wrong_weights = weights;
            wrong_weights.gate_codes = wrong_codes;
            let wrong_input = W4A8Inputs {
                weights: &wrong_weights,
                ..input
            };
            ops.run(s, wrong_input, &scratch_view.0, &mut wrong_view)?;
            s.synchronize()?;
            let host = s.clone_dtoh(&out)?;
            let wrong_host = s.clone_dtoh(&wrong_out)?;
            for d in 0..HIDDEN {
                let got = host[d];
                let want = bf16::from_f32(expected[d] as f32);
                let distance = bf16_ulp_distance(got, want);
                let abs = (f64::from(got.to_f32()) - expected[d]).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "w4a8 expert {expert} dim {d} bf16 ulp distance {distance} abs {abs}"
                );
            }
            let mut caught = 0usize;
            for d in 0..HIDDEN {
                let got = host[d];
                let wrong = wrong_host[d];
                let distance = bf16_ulp_distance(got, wrong);
                let abs = (f64::from(got.to_f32()) - f64::from(wrong.to_f32())).abs();
                if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                    caught += 1;
                }
            }
            ensure!(
                caught > 0,
                "w4a8 expert {expert} planted swapped nibbles were not caught"
            );
        }
        Ok(())
    }
}
