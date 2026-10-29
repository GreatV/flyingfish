//! Shared-expert W8A8 decode: E4M3 weights with one E8M0 scale per 32x32
//! weight block against an E4M3-quantized activation (kernels/moe_ffn.cu).
//! Semantics per the DeepSeek-V4.1-Flash model definition (publisher
//! inference/kernel.py fp8_gemm and act_quant, model.py Expert.forward and
//! MoE.forward):
//!   - the weight scale grid is [ceil(rows/32), ceil(k/32)] and the scale of
//!     the block (row/32, k/32) multiplies the 32-element dot of that block;
//!   - the activation is quantized dynamically: the amax over 32 elements is
//!     floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the
//!     codes are E4M3 round-to-nearest-even;
//!   - the gate branch is clamped from above, the up branch on both sides, the
//!     SiLU stays in FP32 and multiplies the up branch in FP32, one BF16 cast;
//!   - the down input is the BF16 hidden state and the output is BF16;
//!   - the shared expert contributes additively with weight 1 (model
//!     definition MoE.forward: y += self.shared_experts(x)).
use crate::backend::cuda::contracts::{Tensor, TensorMut, dense_layout, no_overlap};
use crate::backend::cuda::cubin;
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

/// Elements per E8M0 activation scale along K and rows per weight scale block.
pub const W8A8_SCALE_BLOCK: u32 = 32;
/// Threads per W8A8 block; one warp owns one output row.
const W8A8_THREADS: u32 = 256;
/// Output rows per W8A8 block.
const W8A8_ROWS: u32 = 8;
/// Dynamic shared bytes one W8A8 block may request below the 48 KB limit.
const W8A8_SHARED_LIMIT: u64 = 48 * 1024 - 1024;

/// The shared expert's W8A8 weights: E4M3 codes with one E8M0 scale per
/// 32x32 block of each weight matrix.
#[derive(Clone, Copy)]
pub struct W8A8SharedWeights {
    /// `[intermediate, hidden]` E4M3 gate codes.
    pub gate_codes: Tensor<u8, 2>,
    /// `[ceil(intermediate/32), hidden/32]` E8M0 gate block scales.
    pub gate_scales: Tensor<u8, 2>,
    /// `[intermediate, hidden]` E4M3 up codes.
    pub up_codes: Tensor<u8, 2>,
    /// `[ceil(intermediate/32), hidden/32]` E8M0 up block scales.
    pub up_scales: Tensor<u8, 2>,
    /// `[hidden, intermediate]` E4M3 down codes.
    pub down_codes: Tensor<u8, 2>,
    /// `[ceil(hidden/32), intermediate/32]` E8M0 down block scales.
    pub down_scales: Tensor<u8, 2>,
}

/// Inputs of the shared expert's W8A8 decode contribution.
#[derive(Clone, Copy)]
pub struct W8A8Inputs<'a> {
    /// `[hidden]` BF16 token activation.
    pub x: Tensor<bf16, 1>,
    /// The shared expert's weights.
    pub weights: &'a W8A8SharedWeights,
    /// The SwiGLU clamp limit.
    pub swiglu_limit: f32,
}

/// Derived extents of one call.
#[derive(Clone, Copy, Debug)]
struct W8A8Dims {
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
    let state = scales + align16(u64::from(hidden / W8A8_SCALE_BLOCK));
    let down_codes = state + align16(2 * u64::from(intermediate));
    let down_scales = down_codes + align16(u64::from(intermediate));
    let total = down_scales + align16(u64::from(intermediate / W8A8_SCALE_BLOCK));
    ([codes, scales, state, down_codes, down_scales], total)
}

/// Scratch bytes of one W8A8 call: the activation codes and scales, the BF16
/// hidden state and the down-input codes and scales.
pub fn scratch_bytes(hidden: u32, intermediate: u32) -> u64 {
    scratch_layout(hidden, intermediate).1
}

fn require_dense<T: Copy, const R: usize>(
    tensor: &Tensor<T, R>,
    name: &str,
    shape: [u32; R],
) -> Result<()> {
    tensor.validate()?;
    let (stride, _) = dense_layout(shape)?;
    ensure!(
        tensor.shape == shape && tensor.stride == stride,
        "w8a8 shared expert {name} shape {:?} stride {:?} does not match the expected dense {:?}",
        tensor.shape,
        tensor.stride,
        shape
    );
    Ok(())
}

/// Validate the call extents against each other.
fn validate(
    input: W8A8Inputs<'_>,
    scratch: &Tensor<u8, 1>,
    out: &Tensor<bf16, 1>,
) -> Result<W8A8Dims> {
    input.x.validate()?;
    let hidden = input.x.shape[0];
    ensure!(
        input.x.stride == [1],
        "w8a8 shared expert x must be contiguous [hidden] (shape {:?}, stride {:?})",
        input.x.shape,
        input.x.stride
    );
    ensure!(
        hidden >= W8A8_SCALE_BLOCK && hidden.is_multiple_of(W8A8_SCALE_BLOCK),
        "w8a8 shared expert hidden {hidden} must be a positive multiple of {W8A8_SCALE_BLOCK}"
    );
    let intermediate = input.weights.gate_codes.shape[0];
    ensure!(
        intermediate >= W8A8_SCALE_BLOCK && intermediate.is_multiple_of(W8A8_SCALE_BLOCK),
        "w8a8 shared expert intermediate {intermediate} must be a positive multiple of {W8A8_SCALE_BLOCK}"
    );
    let weights = input.weights;
    require_dense(&weights.gate_codes, "gate_codes", [intermediate, hidden])?;
    require_dense(
        &weights.gate_scales,
        "gate_scales",
        [
            intermediate.div_ceil(W8A8_SCALE_BLOCK),
            hidden / W8A8_SCALE_BLOCK,
        ],
    )?;
    require_dense(&weights.up_codes, "up_codes", [intermediate, hidden])?;
    require_dense(
        &weights.up_scales,
        "up_scales",
        [
            intermediate.div_ceil(W8A8_SCALE_BLOCK),
            hidden / W8A8_SCALE_BLOCK,
        ],
    )?;
    require_dense(&weights.down_codes, "down_codes", [hidden, intermediate])?;
    require_dense(
        &weights.down_scales,
        "down_scales",
        [
            hidden.div_ceil(W8A8_SCALE_BLOCK),
            intermediate / W8A8_SCALE_BLOCK,
        ],
    )?;
    ensure!(
        input.swiglu_limit.is_finite() && input.swiglu_limit > 0.0,
        "w8a8 shared expert swiglu limit {} must be positive and finite",
        input.swiglu_limit
    );
    out.validate()?;
    ensure!(
        out.shape == [hidden] && out.stride == [1],
        "w8a8 shared expert out must be contiguous [hidden] (shape {:?}, stride {:?})",
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
        "w8a8 shared expert scratch holds {} bytes, the call needs {}",
        scratch.span_bytes,
        scratch_bytes(hidden, intermediate)
    );
    let shared_gate_up = u64::from(hidden) + u64::from(hidden / W8A8_SCALE_BLOCK);
    let shared_down = u64::from(intermediate) + u64::from(intermediate / W8A8_SCALE_BLOCK);
    ensure!(
        shared_gate_up <= W8A8_SHARED_LIMIT && shared_down <= W8A8_SHARED_LIMIT,
        "w8a8 shared expert shared staging {shared_gate_up}/{shared_down} B exceeds the {W8A8_SHARED_LIMIT} B limit (hidden {hidden}, intermediate {intermediate})"
    );
    Ok(W8A8Dims {
        hidden,
        intermediate,
    })
}

/// The shared expert's W8A8 decode contribution.
pub struct MoeSharedW8A8 {
    act_quant: CudaFunction,
    gate_up: CudaFunction,
    down: CudaFunction,
}

pub(crate) struct W8A8BlockCall<'a> {
    pub projections: &'a [super::super::linear::NativeProjection; 3],
    pub fp8: &'a super::super::linear::Fp8Ops,
    pub common: &'a super::super::ops::CommonOpsRef,
    pub rows: super::super::contracts::RowDomain,
    pub x: Tensor<bf16, 2>,
    pub gate: Tensor<bf16, 2>,
    pub up: Tensor<bf16, 2>,
    pub activation: Tensor<bf16, 2>,
    pub out: Tensor<bf16, 2>,
}

impl MoeSharedW8A8 {
    pub(crate) fn run_block(stream: &Arc<CudaStream>, call: W8A8BlockCall<'_>) -> Result<()> {
        let W8A8BlockCall {
            projections,
            fp8,
            common,
            rows,
            x,
            gate,
            up,
            activation,
            out,
        } = call;
        projections[0].run(fp8, stream, x, &TensorMut(gate))?;
        projections[1].run(fp8, stream, x, &TensorMut(up))?;
        let tensor3 = |tensor: Tensor<bf16, 2>| -> Result<Tensor<bf16, 3>> {
            tensor.validate()?;
            let [m, n] = tensor.shape;
            ensure!(
                tensor.stride == [u64::from(n), 1],
                "W8A8 block activation must be dense"
            );
            unsafe {
                Tensor::new(
                    tensor.ptr,
                    [m, 1, n],
                    [u64::from(n), u64::from(n), 1],
                    tensor.span_bytes,
                )
            }
        };
        common.gate.swiglu(
            stream,
            rows,
            tensor3(gate)?,
            tensor3(up)?,
            &mut TensorMut(tensor3(activation)?),
        )?;
        projections[2].run(fp8, stream, activation, &TensorMut(out))
    }

    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "moe_ffn")?;
        Ok(Self {
            act_quant: module.load_function("moe_act_quant_e4m0")?,
            gate_up: module.load_function("moe_ffn_gate_up_e4m3_block32")?,
            down: module.load_function("moe_ffn_down_e4m3_block32")?,
        })
    }

    /// The contribution of the shared expert to one token.
    ///
    /// `scratch` holds the activation codes and scales, the BF16 hidden state
    /// and the down-input codes and scales; `scratch_bytes` sizes it.
    pub fn run(
        &self,
        stream: &Arc<CudaStream>,
        input: W8A8Inputs<'_>,
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
        let groups = hidden / W8A8_SCALE_BLOCK;
        let shared_gate_up = u64::from(hidden) + u64::from(groups);
        let shared_down = u64::from(intermediate) + u64::from(intermediate / W8A8_SCALE_BLOCK);
        unsafe {
            stream
                .launch_builder(&self.act_quant)
                .arg(&input.x.ptr)
                .arg(&codes)
                .arg(&scales)
                .arg(&(hidden as i32))
                .launch(LaunchConfig {
                    grid_dim: (groups.div_ceil(W8A8_ROWS), 1, 1),
                    block_dim: (W8A8_THREADS, 1, 1),
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
                .launch(LaunchConfig {
                    grid_dim: (intermediate.div_ceil(W8A8_ROWS), 1, 1),
                    block_dim: (W8A8_THREADS, 1, 1),
                    shared_mem_bytes: shared_gate_up as u32,
                })?;
            stream
                .launch_builder(&self.act_quant)
                .arg(&state)
                .arg(&down_codes)
                .arg(&down_scales)
                .arg(&(intermediate as i32))
                .launch(LaunchConfig {
                    grid_dim: ((intermediate / W8A8_SCALE_BLOCK).div_ceil(W8A8_ROWS), 1, 1),
                    block_dim: (W8A8_THREADS, 1, 1),
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
                    grid_dim: (hidden.div_ceil(W8A8_ROWS), 1, 1),
                    block_dim: (W8A8_THREADS, 1, 1),
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

    /// DeepSeek-V4.1-Flash layer-0 shapes: 5120 hidden, 2304 shared
    /// intermediate.
    const HIDDEN: usize = 5120;
    const INTERMEDIATE: usize = 2304;
    /// The DeepSeek-V4.1-Flash SwiGLU clamp limit.
    const LIMIT: f32 = 10.0;
    /// Fixtures of the shared expert.
    const FIXTURES: usize = 3;

    /// BF16 ULP distance of two outputs; 1 is the accepted BF16 rounding step.
    const OUTPUT_ULP_TOL: u32 = 1;
    /// Absolute tolerance of the outputs whose BF16 ULP distance exceeds one:
    /// the host simulation of the device loop measures the FP32 accumulation
    /// against the FP64 oracle at 3.9e-3 and one forced BF16 rounding flip
    /// of the silu at 8.0, over the output scale 0..1.3e3; 16.0 covers both.
    const OUTPUT_ABS_TOL: f64 = 16.0;

    /// E4M3 magnitudes of the fixture weights.
    const MAGNITUDES: [u8; 3] = [0x20, 0x28, 0x30];

    /// SplitMix64 finalizer.
    fn hash64(value: u64) -> u64 {
        let mut x = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    /// Hash of one weight element.
    fn element_hash(row: usize, k: usize, salt: u64) -> u64 {
        hash64(
            (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (k as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ salt,
        )
    }

    /// Hash of one weight scale block.
    fn scale_hash(row_block: usize, k_block: usize, salt: u64) -> u64 {
        hash64(
            (row_block as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                ^ (k_block as u64).wrapping_mul(0xA076_1D64_78BD_642F)
                ^ salt,
        )
    }

    /// The token activation: a scrambled pattern of exact BF16 values.
    fn activation() -> Vec<bf16> {
        (0..HIDDEN)
            .map(|i| {
                let t = ((hash64(i as u64) >> 40) & 0xFF) as f32;
                bf16::from_f32((t - 128.0) / 128.0)
            })
            .collect()
    }

    /// The E4M3 code of one weight element: a scrambled magnitude and sign.
    fn weight_code(row: usize, k: usize, salt: u64) -> u8 {
        let hash = element_hash(row, k, salt);
        MAGNITUDES[((hash >> 33) % 3) as usize] | (((hash >> 5) & 1) as u8) << 7
    }

    /// The E8M0 scale code of one weight scale block.
    fn scale_code(row_block: usize, k_block: usize, salt: u64) -> u8 {
        125 + (scale_hash(row_block, k_block, salt) % 3) as u8
    }

    /// One shared expert's fixture.
    struct Fixture {
        x: Vec<bf16>,
        gate_codes: Vec<u8>,
        gate_scales: Vec<u8>,
        up_codes: Vec<u8>,
        up_scales: Vec<u8>,
        down_codes: Vec<u8>,
        down_scales: Vec<u8>,
    }

    /// The fixture of one shared expert.
    fn fixture(index: usize) -> Fixture {
        let salt = (index as u64) * 0x1_0000;
        let gate_codes = (0..INTERMEDIATE)
            .flat_map(|row| (0..HIDDEN).map(move |k| weight_code(row, k, 0x1111 + salt)))
            .collect();
        let up_codes = (0..INTERMEDIATE)
            .flat_map(|row| (0..HIDDEN).map(move |k| weight_code(row, k, 0x2222 + salt)))
            .collect();
        let down_codes = (0..HIDDEN)
            .flat_map(|row| (0..INTERMEDIATE).map(move |k| weight_code(row, k, 0x3333 + salt)))
            .collect();
        let gate_rows = INTERMEDIATE.div_ceil(32);
        let gate_blocks = HIDDEN / 32;
        let down_rows = HIDDEN.div_ceil(32);
        let down_blocks = INTERMEDIATE / 32;
        let gate_scales = (0..gate_rows)
            .flat_map(|row| {
                (0..gate_blocks).map(move |block| scale_code(row, block, 0x4444 + salt))
            })
            .collect();
        let up_scales = (0..gate_rows)
            .flat_map(|row| {
                (0..gate_blocks).map(move |block| scale_code(row, block, 0x5555 + salt))
            })
            .collect();
        let down_scales = (0..down_rows)
            .flat_map(|row| {
                (0..down_blocks).map(move |block| scale_code(row, block, 0x6666 + salt))
            })
            .collect();
        Fixture {
            x: activation(),
            gate_codes,
            gate_scales,
            up_codes,
            up_scales,
            down_codes,
            down_scales,
        }
    }

    /// The weight scale grid read with the two extents swapped: the
    /// published grid is `[rows/32, k/32]` and the transposed read uses the
    /// row count as the row stride.
    fn transposed_scales(grid: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut out = vec![0u8; grid.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[row * cols + col] = grid[col * rows + row];
            }
        }
        out
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

    /// Activation quantization of one row: per `block` elements the amax is
    /// floored at 1e-4, the E8M0 scale is 2^ceil(log2(amax/448)) and the
    /// codes are E4M3.
    fn quantize(values: &[f32], block: usize) -> (Vec<u8>, Vec<u8>) {
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / block);
        for group in 0..values.len() / block {
            let chunk = &values[group * block..group * block + block];
            let mut amax = 0.0f32;
            for value in chunk {
                amax = amax.max(value.abs());
            }
            let amax = amax.max(1e-4);
            let exponent = log2_ceil(amax * (1.0f32 / 448.0f32));
            let scale = f32::from_bits(((exponent + 127) as u32) << 23);
            for value in chunk {
                codes.push(encode_e4m3((*value / scale).clamp(-448.0, 448.0)));
            }
            scales.push((exponent + 127) as u8);
        }
        (codes, scales)
    }

    /// FP64 reference of the shared expert path.
    struct Oracle {
        hidden: usize,
        intermediate: usize,
        limit: f64,
        clamp: bool,
        /// Elements per activation scale.
        quant_block: usize,
        /// Read the weight scale grids with the two extents swapped.
        transposed_scales: bool,
    }

    impl Oracle {
        fn new(hidden: usize, intermediate: usize, limit: f64) -> Self {
            Self {
                hidden,
                intermediate,
                limit,
                clamp: true,
                quant_block: 32,
                transposed_scales: false,
            }
        }

        /// The weight scale of the block (row/32, k/32) in the grid
        /// `[ceil(rows/32), ceil(k/32)]`.
        fn weight_scale(
            &self,
            grid: &[u8],
            rows: usize,
            cols: usize,
            row: usize,
            block: usize,
        ) -> f64 {
            let row_block = row / 32;
            let index = if self.transposed_scales {
                block * rows + row_block
            } else {
                row_block * cols + block
            };
            e8m0_value(grid[index])
        }

        /// The gate and up dots before the clamp. The activation scale of a
        /// block multiplies that block's dot once and the weight block scale
        /// of the grid cell (row/32, block) multiplies the 32-element dot.
        fn gate_up(&self, fx: &Fixture) -> (Vec<f64>, Vec<f64>) {
            let x: Vec<f32> = fx.x.iter().map(|value| value.to_f32()).collect();
            let (x_codes, x_scales) = quantize(&x, self.quant_block);
            let blocks = self.hidden / 32;
            let grid_rows = self.intermediate.div_ceil(32);
            let grid_cols = self.hidden / 32;
            let mut gates = vec![0.0f64; self.intermediate];
            let mut ups = vec![0.0f64; self.intermediate];
            for j in 0..self.intermediate {
                let mut gate_sum = 0.0f64;
                let mut up_sum = 0.0f64;
                for block in 0..blocks {
                    let first = (block * 32) / self.quant_block;
                    let last = ((block + 1) * 32) / self.quant_block;
                    let mut gate_block = 0.0f64;
                    let mut up_block = 0.0f64;
                    for (sub, sub_scale) in x_scales[first..last].iter().enumerate() {
                        let sub = first + sub;
                        let mut gate_sub = 0.0f64;
                        let mut up_sub = 0.0f64;
                        for k in (sub * self.quant_block)..((sub + 1) * self.quant_block) {
                            let index = block * 32 + (k - sub * self.quant_block);
                            let a = f64::from(decode_e4m3(x_codes[index]));
                            gate_sub +=
                                f64::from(decode_e4m3(fx.gate_codes[j * self.hidden + index])) * a;
                            up_sub +=
                                f64::from(decode_e4m3(fx.up_codes[j * self.hidden + index])) * a;
                        }
                        let act_scale = e8m0_value(*sub_scale);
                        gate_block += gate_sub * act_scale;
                        up_block += up_sub * act_scale;
                    }
                    let gate_scale =
                        self.weight_scale(&fx.gate_scales, grid_rows, grid_cols, j, block);
                    let up_scale = self.weight_scale(&fx.up_scales, grid_rows, grid_cols, j, block);
                    gate_sum += gate_block * gate_scale;
                    up_sum += up_block * up_scale;
                }
                gates[j] = gate_sum;
                ups[j] = up_sum;
            }
            (gates, ups)
        }

        /// The BF16 down-input row after the clamped SwiGLU: FP32 silu times
        /// the up branch in FP32 and one BF16 cast (model definition
        /// Expert.forward).
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
                    f64::from(bf16::from_f32(product).to_f32())
                })
                .collect()
        }

        /// The shared expert contribution of one token.
        fn forward(&self, fx: &Fixture) -> Vec<f64> {
            let state = self.hidden_state(fx);
            let scaled: Vec<f32> = state.iter().map(|value| *value as f32).collect();
            let (codes, scales) = quantize(&scaled, self.quant_block);
            let blocks = self.intermediate / 32;
            let grid_rows = self.hidden.div_ceil(32);
            let grid_cols = self.intermediate / 32;
            let mut out = vec![0.0f64; self.hidden];
            for (d, slot) in out.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for block in 0..blocks {
                    let first = (block * 32) / self.quant_block;
                    let last = ((block + 1) * 32) / self.quant_block;
                    let mut sum = 0.0f64;
                    for (sub, sub_scale) in scales[first..last].iter().enumerate() {
                        let sub = first + sub;
                        let mut sub_sum = 0.0f64;
                        for k in (sub * self.quant_block)..((sub + 1) * self.quant_block) {
                            let index = block * 32 + (k - sub * self.quant_block);
                            let a = f64::from(decode_e4m3(codes[index]));
                            sub_sum += f64::from(decode_e4m3(
                                fx.down_codes[d * self.intermediate + index],
                            )) * a;
                        }
                        sum += sub_sum * e8m0_value(*sub_scale);
                    }
                    let weight_scale =
                        self.weight_scale(&fx.down_scales, grid_rows, grid_cols, d, block);
                    acc += sum * weight_scale;
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
    fn w8a8_block_scale_grid_matches_the_published_layout() {
        // The published grid of a [rows, k] weight is [ceil(rows/32),
        // ceil(k/32)] and the scale of the block (row/32, k/32) multiplies
        // that 32x32 block.
        let fx = fixture(0);
        assert_eq!(fx.gate_codes.len(), INTERMEDIATE * HIDDEN);
        assert_eq!(fx.gate_scales.len(), 72 * 160);
        assert_eq!(fx.up_scales.len(), 72 * 160);
        assert_eq!(fx.down_scales.len(), 160 * 72);
        let grid_rows = INTERMEDIATE.div_ceil(32);
        let grid_cols = HIDDEN / 32;
        for row in [0usize, 31, 32, 71, 72, INTERMEDIATE - 1] {
            for block in [0usize, 1, 159] {
                assert_eq!(
                    fx.gate_scales[row / 32 * grid_cols + block],
                    scale_code(row / 32, block, 0x4444),
                    "w8a8 gate scale of the block ({row}, {block}) is misplaced"
                );
            }
        }
        // The transposed read of the same grid lands on other blocks.
        let transposed = transposed_scales(&fx.gate_scales, grid_rows, grid_cols);
        assert_ne!(transposed, fx.gate_scales, "w8a8 scale grid is symmetric");
        let down_rows = HIDDEN.div_ceil(32);
        let down_cols = INTERMEDIATE / 32;
        let transposed_down = transposed_scales(&fx.down_scales, down_rows, down_cols);
        assert_ne!(
            transposed_down, fx.down_scales,
            "w8a8 down scale grid is symmetric"
        );
    }

    #[test]
    fn w8a8_validation_names_every_failed_extent() {
        let hidden = 256u32;
        let intermediate = 128u32;
        // The synthetic buffers are spaced so that no two of them overlap.
        let weights = W8A8SharedWeights {
            gate_codes: fake::<u8, 2>([intermediate, hidden], 0x1_0000),
            gate_scales: fake::<u8, 2>([intermediate / 32, hidden / 32], 0x1_9000),
            up_codes: fake::<u8, 2>([intermediate, hidden], 0x1_A000),
            up_scales: fake::<u8, 2>([intermediate / 32, hidden / 32], 0x2_3000),
            down_codes: fake::<u8, 2>([hidden, intermediate], 0x2_4000),
            down_scales: fake::<u8, 2>([hidden / 32, intermediate / 32], 0x2_D000),
        };
        let x = fake::<bf16, 1>([hidden], 0x3_0000);
        let out = fake_mut::<bf16, 1>([hidden], 0x3_1000);
        let scratch = fake_mut::<u8, 1>([scratch_bytes(hidden, intermediate) as u32], 0x3_2000);
        let input = W8A8Inputs {
            x,
            weights: &weights,
            swiglu_limit: LIMIT,
        };
        validate(input, &scratch.0, &out.0).expect("valid w8a8 extents");
        let check = |error: Result<W8A8Dims>, expected: &str| {
            let error = error
                .expect_err("w8a8 validation accepted a bad extent")
                .to_string();
            assert!(
                error.contains(expected),
                "w8a8 validation error {error:?} does not name {expected}"
            );
        };
        let mut bad = input;
        bad.x = fake::<bf16, 1>([hidden - 1], 0x3_0000);
        check(validate(bad, &scratch.0, &out.0), "hidden");
        let short_gate = W8A8SharedWeights {
            gate_codes: fake::<u8, 2>([intermediate, hidden / 2], 0x1_0000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &short_gate;
        check(validate(bad, &scratch.0, &out.0), "gate_codes");
        let wide_scales = W8A8SharedWeights {
            gate_scales: fake::<u8, 2>([intermediate / 32 + 1, hidden / 32], 0x1_9000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &wide_scales;
        check(validate(bad, &scratch.0, &out.0), "gate_scales");
        let long_down = W8A8SharedWeights {
            down_codes: fake::<u8, 2>([hidden, intermediate + 8], 0x2_4000),
            ..weights
        };
        let mut bad = input;
        bad.weights = &long_down;
        check(validate(bad, &scratch.0, &out.0), "down_codes");
        let mut bad = input;
        bad.swiglu_limit = 0.0;
        check(validate(bad, &scratch.0, &out.0), "swiglu limit");
        let short = fake_mut::<u8, 1>([16], 0x3_2000);
        check(validate(input, &short.0, &out.0), "scratch");
        let bad_out = fake_mut::<bf16, 1>([hidden + 32], 0x3_1000);
        check(validate(input, &scratch.0, &bad_out.0), "out");
    }

    #[test]
    fn w8a8_block_scale_transpose_changes_the_contribution() {
        let fx = fixture(0);
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let straight = oracle.forward(&fx);
        let mut transposed = oracle;
        transposed.transposed_scales = true;
        let swapped = transposed.forward(&fx);
        assert_ne!(
            straight, swapped,
            "w8a8 block-scale transpose control escaped"
        );
    }

    #[test]
    fn w8a8_unclamped_swiglu_changes_the_contribution() {
        let fx = fixture(0);
        let clamped = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let mut unclamped = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        unclamped.clamp = false;
        // The fixture must straddle the clamp: the gate branch above the
        // limit, the gate branch below it and the up branch on both sides.
        let (gates, ups) = clamped.gate_up(&fx);
        let limit = LIMIT as f64;
        assert!(
            gates.iter().any(|gate| *gate > limit),
            "w8a8 fixture gate never exceeds the clamp limit"
        );
        assert!(
            gates.iter().any(|gate| *gate < -limit),
            "w8a8 fixture gate never falls below the clamp limit"
        );
        assert!(
            ups.iter().any(|up| up.abs() > limit),
            "w8a8 fixture up never exceeds the clamp limit"
        );
        assert_ne!(
            clamped.forward(&fx),
            unclamped.forward(&fx),
            "w8a8 clamped-swiglu control escaped"
        );
    }

    #[test]
    fn w8a8_activation_granularity_changes_the_contribution() {
        let fx = fixture(1);
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let straight = oracle.forward(&fx);
        let mut coarse = oracle;
        coarse.quant_block = 16;
        let halved = coarse.forward(&fx);
        assert_ne!(
            straight, halved,
            "w8a8 activation granularity control escaped"
        );
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn w8a8_shared_expert_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = MoeSharedW8A8::new(&device.ctx)?;
        let oracle = Oracle::new(HIDDEN, INTERMEDIATE, LIMIT as f64);
        let bytes = scratch_bytes(HIDDEN as u32, INTERMEDIATE as u32) as usize;
        for index in 0..FIXTURES {
            let fx = fixture(index);
            let expected = oracle.forward(&fx);
            let xb = testing::upload(&device.upload, fx.x.clone())?;
            let gb = testing::upload(&device.upload, fx.gate_codes.clone())?;
            let gs = testing::upload(&device.upload, fx.gate_scales.clone())?;
            let ub = testing::upload(&device.upload, fx.up_codes.clone())?;
            let us = testing::upload(&device.upload, fx.up_scales.clone())?;
            let db = testing::upload(&device.upload, fx.down_codes.clone())?;
            let ds = testing::upload(&device.upload, fx.down_scales.clone())?;
            let swapped =
                transposed_scales(&fx.gate_scales, INTERMEDIATE.div_ceil(32), HIDDEN / 32);
            let sb = testing::upload(&device.upload, swapped)?;
            device.finish_upload()?;
            let out = testing::zeros::<bf16>(s, HIDDEN)?;
            let wrong_out = testing::zeros::<bf16>(s, HIDDEN)?;
            let scratch = testing::zeros::<u8>(s, bytes)?;
            let x = testing::view(&xb, s, [HIDDEN as u32], [1])?;
            let gate_codes = testing::view(
                &gb,
                s,
                [INTERMEDIATE as u32, HIDDEN as u32],
                [HIDDEN as u64, 1],
            )?;
            let gate_scales = testing::view(&gs, s, [72, 160], [160, 1])?;
            let up_codes = testing::view(
                &ub,
                s,
                [INTERMEDIATE as u32, HIDDEN as u32],
                [HIDDEN as u64, 1],
            )?;
            let up_scales = testing::view(&us, s, [72, 160], [160, 1])?;
            let down_codes = testing::view(
                &db,
                s,
                [HIDDEN as u32, INTERMEDIATE as u32],
                [INTERMEDIATE as u64, 1],
            )?;
            let down_scales = testing::view(&ds, s, [160, 72], [72, 1])?;
            let wrong_scales = testing::view(&sb, s, [72, 160], [160, 1])?;
            let mut out_view = testing::mut_view(&out, s, [HIDDEN as u32], [1])?;
            let mut wrong_view = testing::mut_view(&wrong_out, s, [HIDDEN as u32], [1])?;
            let scratch_view = testing::mut_view(&scratch, s, [bytes as u32], [1])?;
            let weights = W8A8SharedWeights {
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                down_codes,
                down_scales,
            };
            let input = W8A8Inputs {
                x,
                weights: &weights,
                swiglu_limit: LIMIT,
            };
            ops.run(s, input, &scratch_view.0, &mut out_view)?;
            let mut wrong_weights = weights;
            wrong_weights.gate_scales = wrong_scales;
            let wrong_input = W8A8Inputs {
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
                    "w8a8 shared expert {index} dim {d} bf16 ulp distance {distance} abs {abs}"
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
                "w8a8 shared expert {index} planted transposed scale grid was not caught"
            );
        }
        Ok(())
    }
}
