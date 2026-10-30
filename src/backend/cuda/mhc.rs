//! mHC hyper-connection mixing op: per-row mixer coefficients, the sinkhorn
//! combination matrix, the stream collapse and the residual update
//! (kernels/mhc.cu). Numerics per the DeepSeek-V4.1-Flash and GLM-5.3 model
//! definitions (publisher inference/model.py Block; modeling_glm5_next.py)
//! and mHC: Manifold-Constrained Hyper-Connections (arXiv 2512.24880); the
//! split/sinkhorn semantics are pinned by the black-box publisher fixtures.
use crate::backend::cuda::contracts::{Tensor, TensorMut, dense_layout, no_overlap};
use crate::backend::cuda::cubin;
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use anyhow::{Result, ensure};
use half::bf16;
use std::sync::Arc;

/// Threads per mHC block.
const MHC_THREADS: u32 = 256;
/// Dynamic shared bytes one block may request below the 48 KB limit.
const MHC_SHARED_LIMIT: u64 = 48 * 1024 - 1024;

/// Rounding of the mHC update coefficients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcRounding {
    /// FP32 coefficients, FP32 accumulation, one BF16 rounding at the output.
    Fp32,
    /// 50e35cc0:1335-1337,1346-1348: post and comb rounded to BF16, BF16
    /// product, comb^T@residual with FP32 accumulation and one BF16 round,
    /// BF16 add.
    PostCombBf16,
}

impl HcRounding {
    /// Kernel argument of the update rounding mode.
    pub fn kernel_arg(self) -> i32 {
        match self {
            HcRounding::Fp32 => 0,
            HcRounding::PostCombBf16 => 1,
        }
    }
}

/// Orientation of the combination-matrix application in the update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcCombOrientation {
    /// `out[j] = post[j]*x + sum_k comb[j,k]*residual[k]`: the stored row index
    /// selects the residual copy.
    Plain,
    /// `out[j] = post[j]*x + sum_k comb[k,j]*residual[k]`: the stored column index
    /// selects the residual copy (model.py Block.hc_post; the same application in
    /// the GLM-5.3 definition). The coefficients stay row-major `[j, k]`.
    Transposed,
}

impl HcCombOrientation {
    /// Kernel argument of the update comb orientation.
    pub fn kernel_arg(self) -> i32 {
        match self {
            HcCombOrientation::Plain => 0,
            HcCombOrientation::Transposed => 1,
        }
    }
}

/// Storage dtype of the fn projection weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcFnDtype {
    /// F32 storage (DeepSeek-V4.1).
    F32,
    /// BF16 storage (GLM-5.3) with exact FP32 register upcast before the dot.
    Bf16,
}

/// fn projection view of one dtype: `[(2+mult)*mult, mult*hidden]` row-major.
#[derive(Clone, Copy, Debug)]
pub enum HcFnW {
    F32(Tensor<f32, 2>),
    Bf16(Tensor<bf16, 2>),
}

impl HcFnW {
    pub fn shape(&self) -> [u32; 2] {
        match self {
            HcFnW::F32(t) => t.shape,
            HcFnW::Bf16(t) => t.shape,
        }
    }
}

/// Mixer parameters of one mHC sublayer instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcParams {
    /// Number of hyper-connection streams.
    pub mult: u32,
    /// Hidden width of one stream.
    pub hidden: u32,
    /// Sinkhorn rounds; the first row softmax is followed by iters-1 rounds.
    pub iters: u32,
    /// Sinkhorn epsilon.
    pub eps: f32,
    /// Epsilon of the unweighted stream RMS.
    pub norm_eps: f32,
}

/// Mixer weights of one mHC sublayer instance.
pub struct HcWeights {
    /// `[(2+mult)*mult, mult*hidden]` row-major, F32 or BF16 storage.
    pub fn_w: HcFnW,
    /// `[(2+mult)*mult]` f32.
    pub base: Tensor<f32, 1>,
    /// `[3]` f32.
    pub scale: Tensor<f32, 1>,
}

/// Mixer coefficients of one mHC row batch.
pub struct HcMixes<'a> {
    /// `[rows, mult]` f32.
    pub pre: &'a mut TensorMut<f32, 2>,
    /// `[rows, mult]` f32.
    pub post: &'a mut TensorMut<f32, 2>,
    /// `[rows, mult, mult]` f32, rows indexed `[j=out, k=in]`.
    pub comb: &'a mut TensorMut<f32, 3>,
}

/// Update coefficients of one mHC row batch.
pub struct HcUpdate {
    /// `[rows, mult]` f32.
    pub post: Tensor<f32, 2>,
    /// `[rows, mult, mult]` f32, rows indexed `[j=out, k=in]`.
    pub comb: Tensor<f32, 3>,
    pub rounding: HcRounding,
    pub orientation: HcCombOrientation,
}

/// Mixer rows: pre, post and the combination matrix.
pub fn mix_rows(mult: u32) -> u64 {
    u64::from(mult) * u64::from(mult) + 2 * u64::from(mult)
}

/// Validate the mixer parameters against the mixer weight extents.
pub fn validate_params(
    params: &HcParams,
    fn_rows: u32,
    fn_cols: u32,
    base_len: u32,
    scale_len: u32,
) -> Result<()> {
    ensure!(params.mult >= 2, "mHC mult {} is below 2", params.mult);
    ensure!(params.iters >= 1, "mHC iters {} is below 1", params.iters);
    ensure!(
        params.eps.is_finite() && params.eps > 0.0,
        "mHC eps {} is not positive",
        params.eps
    );
    ensure!(
        params.norm_eps.is_finite() && params.norm_eps > 0.0,
        "mHC norm_eps {} is not positive",
        params.norm_eps
    );
    let rows = mix_rows(params.mult);
    let cols = u64::from(params.mult) * u64::from(params.hidden);
    ensure!(rows <= u64::from(u32::MAX), "mHC fn rows {rows} exceed u32");
    ensure!(cols <= u64::from(u32::MAX), "mHC fn cols {cols} exceed u32");
    let (rows, cols) = (rows as u32, cols as u32);
    ensure!(fn_rows == rows, "mHC fn rows {fn_rows} != expected {rows}");
    ensure!(fn_cols == cols, "mHC fn cols {fn_cols} != expected {cols}");
    ensure!(
        base_len == rows,
        "mHC base len {base_len} != expected {rows}"
    );
    ensure!(scale_len == 3, "mHC scale len {scale_len} != expected 3");
    Ok(())
}

/// Dynamic shared bytes of one `mhc_mixes` block: the staged stream row, the
/// mixer outputs and the combination matrix scratch.
fn shared_bytes(params: &HcParams) -> u64 {
    let stream = u64::from(params.mult) * u64::from(params.hidden);
    let comb = u64::from(params.mult) * u64::from(params.mult);
    stream * 2 + (mix_rows(params.mult) + comb) * 4
}

fn require_dense<T: Copy, const R: usize>(tensor: &Tensor<T, R>, name: &str) -> Result<()> {
    tensor.validate()?;
    let (stride, _) = dense_layout(tensor.shape)?;
    ensure!(
        tensor.stride == stride,
        "mHC {name} must be dense: stride {:?} != {:?}",
        tensor.stride,
        stride
    );
    Ok(())
}

pub struct MhcOps {
    mixes_f32: CudaFunction,
    mixes_bf16: CudaFunction,
    collapse: CudaFunction,
    update: CudaFunction,
}

impl MhcOps {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = cubin::module(ctx, "mhc")?;
        Ok(Self {
            mixes_f32: module.load_function("mhc_mixes")?,
            mixes_bf16: module.load_function("mhc_mixes_bf16")?,
            collapse: module.load_function("mhc_collapse")?,
            update: module.load_function("mhc_update")?,
        })
    }

    /// Mixer coefficients of every row: pre, post and the combination matrix.
    /// The fn storage dtype of `weights` selects the kernel entry.
    pub fn mixes(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 3>,
        weights: &HcWeights,
        params: &HcParams,
        out: HcMixes<'_>,
    ) -> Result<()> {
        let HcMixes { pre, post, comb } = out;
        let [fn_rows, fn_cols] = weights.fn_w.shape();
        validate_params(
            params,
            fn_rows,
            fn_cols,
            weights.base.shape[0],
            weights.scale.shape[0],
        )?;
        let rows = x.shape[0];
        require_dense(&x, "stream")?;
        match &weights.fn_w {
            HcFnW::F32(t) => require_dense(t, "fn f32")?,
            HcFnW::Bf16(t) => require_dense(t, "fn bf16")?,
        }
        require_dense(&weights.base, "base")?;
        require_dense(&weights.scale, "scale")?;
        require_dense(&pre.0, "pre")?;
        require_dense(&post.0, "post")?;
        require_dense(&comb.0, "comb")?;
        ensure!(
            x.shape == [rows, params.mult, params.hidden],
            "mHC stream shape {:?} != expected [{rows}, {}, {}]",
            x.shape,
            params.mult,
            params.hidden
        );
        ensure!(
            pre.0.shape == [rows, params.mult],
            "mHC pre shape {:?} != expected [{rows}, {}]",
            pre.0.shape,
            params.mult
        );
        ensure!(
            post.0.shape == [rows, params.mult],
            "mHC post shape {:?} != expected [{rows}, {}]",
            post.0.shape,
            params.mult
        );
        ensure!(
            comb.0.shape == [rows, params.mult, params.mult],
            "mHC comb shape {:?} != expected [{rows}, {}, {}]",
            comb.0.shape,
            params.mult,
            params.mult
        );
        ensure!(
            params.hidden.is_multiple_of(8),
            "mHC hidden width {} must be a multiple of 8 for the staged stream row",
            params.hidden
        );
        let shared = shared_bytes(params);
        ensure!(
            shared <= MHC_SHARED_LIMIT,
            "mHC shared staging {shared} B exceeds the {MHC_SHARED_LIMIT} B limit (mult {}, hidden {})",
            params.mult,
            params.hidden
        );
        no_overlap(&x, &pre.0, "stream", "pre")?;
        no_overlap(&x, &post.0, "stream", "post")?;
        no_overlap(&x, &comb.0, "stream", "comb")?;
        match &weights.fn_w {
            HcFnW::F32(t) => {
                no_overlap(t, &pre.0, "fn f32", "pre")?;
                no_overlap(t, &post.0, "fn f32", "post")?;
                no_overlap(t, &comb.0, "fn f32", "comb")?;
            }
            HcFnW::Bf16(t) => {
                no_overlap(t, &pre.0, "fn bf16", "pre")?;
                no_overlap(t, &post.0, "fn bf16", "post")?;
                no_overlap(t, &comb.0, "fn bf16", "comb")?;
            }
        }
        no_overlap(&weights.base, &pre.0, "base", "pre")?;
        no_overlap(&weights.base, &post.0, "base", "post")?;
        no_overlap(&weights.base, &comb.0, "base", "comb")?;
        no_overlap(&weights.scale, &pre.0, "scale", "pre")?;
        no_overlap(&weights.scale, &post.0, "scale", "post")?;
        no_overlap(&weights.scale, &comb.0, "scale", "comb")?;
        no_overlap(&pre.0, &post.0, "pre", "post")?;
        no_overlap(&pre.0, &comb.0, "pre", "comb")?;
        no_overlap(&post.0, &comb.0, "post", "comb")?;
        let (rows, mult, hidden) = (
            i32::try_from(rows)?,
            params.mult as i32,
            params.hidden as i32,
        );
        let (entry, fn_ptr) = match &weights.fn_w {
            HcFnW::F32(t) => (&self.mixes_f32, t.ptr),
            HcFnW::Bf16(t) => (&self.mixes_bf16, t.ptr),
        };
        unsafe {
            stream
                .launch_builder(entry)
                .arg(&x.ptr)
                .arg(&fn_ptr)
                .arg(&weights.base.ptr)
                .arg(&weights.scale.ptr)
                .arg(&mult)
                .arg(&hidden)
                .arg(&(params.iters as i32))
                .arg(&params.eps)
                .arg(&params.norm_eps)
                .arg(&pre.0.ptr)
                .arg(&post.0.ptr)
                .arg(&comb.0.ptr)
                .arg(&rows)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (MHC_THREADS, 1, 1),
                    shared_mem_bytes: shared as u32,
                })?;
        }
        Ok(())
    }

    /// Collapse of the staged stream row with the pre coefficients.
    pub fn collapse(
        &self,
        stream: &Arc<CudaStream>,
        x: Tensor<bf16, 3>,
        pre: Tensor<f32, 2>,
        out: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let rows = x.shape[0];
        let (mult, hidden) = (x.shape[1], x.shape[2]);
        require_dense(&x, "stream")?;
        require_dense(&pre, "pre")?;
        require_dense(&out.0, "collapse")?;
        ensure!(
            pre.shape == [rows, mult],
            "mHC pre shape {:?} != expected [{rows}, {mult}]",
            pre.shape
        );
        ensure!(
            out.0.shape == [rows, hidden],
            "mHC collapse shape {:?} != expected [{rows}, {hidden}]",
            out.0.shape
        );
        no_overlap(&x, &out.0, "stream", "collapse")?;
        no_overlap(&pre, &out.0, "pre", "collapse")?;
        let (rows, mult, hidden) = (i32::try_from(rows)?, mult as i32, hidden as i32);
        unsafe {
            stream
                .launch_builder(&self.collapse)
                .arg(&x.ptr)
                .arg(&pre.ptr)
                .arg(&mult)
                .arg(&hidden)
                .arg(&out.0.ptr)
                .arg(&rows)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (MHC_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Residual update of every row.
    pub fn update(
        &self,
        stream: &Arc<CudaStream>,
        sublayer_out: Tensor<bf16, 2>,
        residual: Tensor<bf16, 3>,
        coefficients: HcUpdate,
        out: &mut TensorMut<bf16, 3>,
    ) -> Result<()> {
        let HcUpdate {
            post,
            comb,
            rounding,
            orientation,
        } = coefficients;
        let rows = sublayer_out.shape[0];
        let hidden = sublayer_out.shape[1];
        let mult = post.shape[1];
        require_dense(&sublayer_out, "sublayer_out")?;
        require_dense(&residual, "residual")?;
        require_dense(&post, "post")?;
        require_dense(&comb, "comb")?;
        require_dense(&out.0, "update")?;
        ensure!(
            residual.shape == [rows, mult, hidden],
            "mHC residual shape {:?} != expected [{rows}, {mult}, {hidden}]",
            residual.shape
        );
        ensure!(
            post.shape == [rows, mult],
            "mHC post shape {:?} != expected [{rows}, {mult}]",
            post.shape
        );
        ensure!(
            comb.shape == [rows, mult, mult],
            "mHC comb shape {:?} != expected [{rows}, {mult}, {mult}]",
            comb.shape
        );
        ensure!(
            out.0.shape == [rows, mult, hidden],
            "mHC update shape {:?} != expected [{rows}, {mult}, {hidden}]",
            out.0.shape
        );
        no_overlap(&sublayer_out, &out.0, "sublayer_out", "update")?;
        no_overlap(&residual, &out.0, "residual", "update")?;
        no_overlap(&post, &out.0, "post", "update")?;
        no_overlap(&comb, &out.0, "comb", "update")?;
        let (rows, mult, hidden) = (i32::try_from(rows)?, mult as i32, hidden as i32);
        unsafe {
            stream
                .launch_builder(&self.update)
                .arg(&sublayer_out.ptr)
                .arg(&residual.ptr)
                .arg(&post.ptr)
                .arg(&comb.ptr)
                .arg(&mult)
                .arg(&hidden)
                .arg(&rounding.kernel_arg())
                .arg(&orientation.kernel_arg())
                .arg(&out.0.ptr)
                .arg(&rows)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (MHC_THREADS, 1, 1),
                    shared_mem_bytes: 0,
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
    use anyhow::ensure;
    use half::bf16;

    /// Golden-ratio stride: the phase sequence never repeats.
    /// FP32-vs-FP64 accumulation over at most 2*mult terms of magnitude <= 2.
    const OUTPUT_ABS_TOL: f64 = 1e-6;

    fn pattern(seed: usize, phase: f64) -> f64 {
        (seed as f64 * 0.6180339887498949 + phase).sin() * 0.75
    }

    fn sigmoid(value: f64) -> f64 {
        1.0 / (1.0 + (-value).exp())
    }

    fn round_bf16(value: f64) -> f64 {
        f64::from(bf16::from_f32(value as f32).to_f32())
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

    /// FP64 reference with the published rounding points.
    #[derive(Clone, Copy, Debug)]
    struct Oracle {
        mult: usize,
        hidden: usize,
        iters: usize,
        eps: f64,
        norm_eps: f64,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Mixes {
        pre: Vec<f64>,
        post: Vec<f64>,
        comb: Vec<f64>,
    }

    impl Oracle {
        /// Row normalization of a row-major combination matrix.
        fn normalize_rows(matrix: &mut [Vec<f64>], eps: f64) {
            for row in matrix.iter_mut() {
                let sum: f64 = row.iter().sum();
                for value in row.iter_mut() {
                    *value /= sum + eps;
                }
            }
        }

        /// Column normalization of a row-major combination matrix.
        fn normalize_columns(matrix: &mut [Vec<f64>], eps: f64) {
            let mut columns: Vec<Vec<f64>> = (0..matrix.len())
                .map(|k| matrix.iter().map(|row| row[k]).collect())
                .collect();
            for column in &mut columns {
                let sum: f64 = column.iter().sum();
                for value in column.iter_mut() {
                    *value /= sum + eps;
                }
            }
            for (j, row) in matrix.iter_mut().enumerate() {
                for (k, value) in row.iter_mut().enumerate() {
                    *value = columns[k][j];
                }
            }
        }

        fn mixes(&self, stream: &[bf16], fn_w: &[f32], base: &[f32], scale: &[f32]) -> Mixes {
            let n = self.mult * self.hidden;
            let x: Vec<f64> = stream.iter().map(|v| f64::from(v.to_f32())).collect();
            let sumsq: f64 = x.iter().map(|v| v * v).sum();
            let rsqrt = 1.0 / (sumsq / n as f64 + self.norm_eps).sqrt();
            let rows = (2 + self.mult) * self.mult;
            let mut m = vec![0.0f64; rows];
            for (c, value) in m.iter_mut().enumerate() {
                let row = &fn_w[c * n..(c + 1) * n];
                let acc: f64 = row.iter().zip(&x).map(|(w, v)| f64::from(*w) * v).sum();
                *value = acc * rsqrt;
            }
            let pre: Vec<f64> = (0..self.mult)
                .map(|j| sigmoid(m[j] * f64::from(scale[0]) + f64::from(base[j])) + self.eps)
                .collect();
            let post: Vec<f64> = (0..self.mult)
                .map(|j| {
                    2.0 * sigmoid(
                        m[self.mult + j] * f64::from(scale[1]) + f64::from(base[self.mult + j]),
                    )
                })
                .collect();
            let mut matrix: Vec<Vec<f64>> = (0..self.mult)
                .map(|j| {
                    (0..self.mult)
                        .map(|k| {
                            let c = 2 * self.mult + j * self.mult + k;
                            m[c] * f64::from(scale[2]) + f64::from(base[c])
                        })
                        .collect()
                })
                .collect();
            for row in &mut matrix {
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                for value in row.iter_mut() {
                    *value = *value / sum + self.eps;
                }
            }
            Self::normalize_columns(&mut matrix, self.eps);
            for _ in 1..self.iters {
                Self::normalize_rows(&mut matrix, self.eps);
                Self::normalize_columns(&mut matrix, self.eps);
            }
            Mixes {
                pre,
                post,
                comb: matrix.into_iter().flatten().collect(),
            }
        }

        fn collapse(&self, stream: &[bf16], pre: &[f64]) -> Vec<bf16> {
            let mut out = vec![bf16::ZERO; self.hidden];
            for (h, value) in out.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (j, p) in pre.iter().enumerate() {
                    acc += p * f64::from(stream[j * self.hidden + h].to_f32());
                }
                *value = bf16::from_f32(acc as f32);
            }
            out
        }

        fn update(
            &self,
            sublayer: &[bf16],
            residual: &[bf16],
            post: &[f64],
            comb: &[f64],
            round_coeffs: bool,
            transposed: bool,
        ) -> Vec<bf16> {
            let mut out = vec![bf16::ZERO; self.mult * self.hidden];
            if round_coeffs {
                for j in 0..self.mult {
                    let p = bf16::from_f32(post[j] as f32);
                    for h in 0..self.hidden {
                        let prod = bf16::from_f32(p.to_f32() * sublayer[h].to_f32());
                        let mut acc = 0.0f32;
                        for k in 0..self.mult {
                            let index = if transposed {
                                k * self.mult + j
                            } else {
                                j * self.mult + k
                            };
                            let c = bf16::from_f32(comb[index] as f32);
                            acc = c
                                .to_f32()
                                .mul_add(residual[k * self.hidden + h].to_f32(), acc);
                        }
                        let mm = bf16::from_f32(acc);
                        out[j * self.hidden + h] = bf16::from_f32(prod.to_f32() + mm.to_f32());
                    }
                }
                return out;
            }
            for j in 0..self.mult {
                let p = if round_coeffs {
                    round_bf16(post[j])
                } else {
                    post[j]
                };
                for h in 0..self.hidden {
                    let mut acc = p * f64::from(sublayer[h].to_f32());
                    for k in 0..self.mult {
                        let index = if transposed {
                            k * self.mult + j
                        } else {
                            j * self.mult + k
                        };
                        let c = if round_coeffs {
                            round_bf16(comb[index])
                        } else {
                            comb[index]
                        };
                        acc += c * f64::from(residual[k * self.hidden + h].to_f32());
                    }
                    out[j * self.hidden + h] = bf16::from_f32(acc as f32);
                }
            }
            out
        }
    }

    struct Fixture {
        stream: Vec<bf16>,
        fn_w: Vec<f32>,
        base: Vec<f32>,
        scale: Vec<f32>,
        sublayer: Vec<bf16>,
        residual: Vec<bf16>,
    }

    /// Fixture whose strides are coprime with the shape strides in play.
    fn fixture(mult: u32, hidden: u32, rows: u32, seed: usize) -> Fixture {
        let mix = mix_rows(mult) as usize;
        let stream: Vec<bf16> = (0..(rows * mult * hidden) as usize)
            .map(|i| bf16::from_f32(pattern(seed + i, 0.11) as f32))
            .collect();
        let fn_w: Vec<f32> = (0..(mix * mult as usize * hidden as usize))
            .map(|i| (pattern(seed * 31 + i, 1.7) * 0.1) as f32)
            .collect();
        let base: Vec<f32> = (0..mix)
            .map(|i| (pattern(seed * 17 + i, 2.3) * 0.5) as f32)
            .collect();
        let scale: Vec<f32> = vec![0.5, 0.75, 1.25];
        let sublayer: Vec<bf16> = (0..(rows * hidden) as usize)
            .map(|i| bf16::from_f32(pattern(seed * 13 + i, 3.9) as f32))
            .collect();
        let residual: Vec<bf16> = (0..(rows * mult * hidden) as usize)
            .map(|i| bf16::from_f32(pattern(seed * 7 + i, 5.1) as f32))
            .collect();
        Fixture {
            stream,
            fn_w,
            base,
            scale,
            sublayer,
            residual,
        }
    }

    #[test]
    fn mhc_hand_case_mult2_matches_named_constants() {
        const MULT: usize = 2;
        const HIDDEN: usize = 2;
        // stream = [[1,2],[3,4]]; x = [1,2,3,4]; sum(x^2) = 30; mean = 7.5;
        // mean + norm_eps = 7.5 + 8.5 = 16; rsqrt = 1/4.
        const NORM_EPS: f64 = 8.5;
        // The mixer rows pick one element of x each: m = [1,3,2,4] * 1/4.
        fn m0() -> f64 {
            1.0 * 0.25
        }
        fn m1() -> f64 {
            3.0 * 0.25
        }
        fn m2() -> f64 {
            2.0 * 0.25
        }
        fn m3() -> f64 {
            4.0 * 0.25
        }
        // pre[j] = sigmoid(m[j] * scale[0] + base[j]) + eps with scale = 1,
        // base = 0 and eps = 1/3.
        fn pre_0() -> f64 {
            sigmoid(m0()) + 1.0 / 3.0
        }
        fn pre_1() -> f64 {
            sigmoid(m1()) + 1.0 / 3.0
        }
        // post[j] = 2 * sigmoid(m[2+j] * scale[1] + base[2+j]).
        fn post_0() -> f64 {
            2.0 * sigmoid(m2())
        }
        fn post_1() -> f64 {
            2.0 * sigmoid(m3())
        }
        // comb_raw = 0 for every (j,k): the row softmax is 1/2 per element,
        // + eps = 1/2 + 1/3 = 5/6, the column sum is 5/3 and the column
        // normalization divides by 5/3 + 1/3 = 2, so every element is 5/12.
        fn comb() -> f64 {
            5.0 / 12.0
        }

        let oracle = Oracle {
            mult: MULT,
            hidden: HIDDEN,
            iters: 1,
            eps: 1.0 / 3.0,
            norm_eps: NORM_EPS,
        };
        let stream = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let mut fn_w = vec![0.0f32; 8 * 4];
        // Mixer row c spans the columns c*4..c*4+4 and picks one element of x.
        fn_w[0] = 1.0;
        fn_w[6] = 1.0;
        fn_w[9] = 1.0;
        fn_w[15] = 1.0;
        let base = vec![0.0f32; 8];
        let scale = vec![1.0f32; 3];
        let mixes = oracle.mixes(&stream, &fn_w, &base, &scale);
        assert!(
            (mixes.pre[0] - pre_0()).abs() <= 1e-12,
            "mHC hand case pre[0] {} != {}",
            mixes.pre[0],
            pre_0()
        );
        assert!(
            (mixes.pre[1] - pre_1()).abs() <= 1e-12,
            "mHC hand case pre[1] {} != {}",
            mixes.pre[1],
            pre_1()
        );
        assert!(
            (mixes.post[0] - post_0()).abs() <= 1e-12,
            "mHC hand case post[0] {} != {}",
            mixes.post[0],
            post_0()
        );
        assert!(
            (mixes.post[1] - post_1()).abs() <= 1e-12,
            "mHC hand case post[1] {} != {}",
            mixes.post[1],
            post_1()
        );
        for (j, row) in mixes.comb.chunks(MULT).enumerate() {
            for (k, value) in row.iter().enumerate() {
                assert!(
                    (value - comb()).abs() <= 1e-12,
                    "mHC hand case comb[{j},{k}] {value} != {}",
                    comb()
                );
            }
        }

        let collapse = oracle.collapse(&stream, &mixes.pre);
        for (h, value) in collapse.iter().enumerate() {
            let mut expected = 0.0f64;
            for (j, p) in mixes.pre.iter().enumerate() {
                expected += p * f64::from(stream[j * HIDDEN + h].to_f32());
            }
            assert_eq!(
                value.to_bits(),
                bf16::from_f32(expected as f32).to_bits(),
                "mHC hand case collapse[{h}] {value} != {expected}"
            );
        }

        let sublayer = vec![bf16::from_f32(2.0), bf16::from_f32(4.0)];
        let residual = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let straight = oracle.update(&sublayer, &residual, &mixes.post, &mixes.comb, false, false);
        for j in 0..MULT {
            let post = if j == 0 { post_0() } else { post_1() };
            for h in 0..HIDDEN {
                let mut expected = post * f64::from(sublayer[h].to_f32());
                expected += comb() * f64::from(residual[h].to_f32());
                expected += comb() * f64::from(residual[HIDDEN + h].to_f32());
                assert_eq!(
                    straight[j * HIDDEN + h].to_bits(),
                    bf16::from_f32(expected as f32).to_bits(),
                    "mHC hand case update[{j},{h}] != {expected}"
                );
            }
        }
        let rounded = oracle.update(&sublayer, &residual, &mixes.post, &mixes.comb, true, false);
        for j in 0..MULT {
            let post = if j == 0 { post_0() } else { post_1() };
            for h in 0..HIDDEN {
                // 50e35cc0:1335-1337,1346-1348: BF16 product, BF16 matmul,
                // BF16 add.
                let prod =
                    bf16::from_f32(bf16::from_f32(post as f32).to_f32() * sublayer[h].to_f32());
                let comb_bf = bf16::from_f32(comb() as f32);
                let mut acc = comb_bf.to_f32().mul_add(residual[h].to_f32(), 0.0);
                acc = comb_bf.to_f32().mul_add(residual[HIDDEN + h].to_f32(), acc);
                let mm = bf16::from_f32(acc);
                let expected = bf16::from_f32(prod.to_f32() + mm.to_f32());
                assert_eq!(
                    rounded[j * HIDDEN + h].to_bits(),
                    expected.to_bits(),
                    "mHC hand case BF16 update[{j},{h}] != {expected:?}"
                );
            }
        }
    }

    #[test]
    fn mhc_validation_rejects_every_failed_parameter() {
        let good = HcParams {
            mult: 4,
            hidden: 4096,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        validate_params(&good, 24, 16384, 24, 3).expect("valid mHC parameters");
        let cases = [
            (HcParams { mult: 1, ..good }, (24, 16384, 24, 3), "mult"),
            (HcParams { iters: 0, ..good }, (24, 16384, 24, 3), "iters"),
            (HcParams { eps: 0.0, ..good }, (24, 16384, 24, 3), "eps"),
            (
                HcParams {
                    norm_eps: 0.0,
                    ..good
                },
                (24, 16384, 24, 3),
                "norm_eps",
            ),
            (good, (23, 16384, 24, 3), "fn rows"),
            (good, (24, 8192, 24, 3), "fn cols"),
            (good, (24, 16384, 23, 3), "base len"),
            (good, (24, 16384, 24, 2), "scale len"),
            (good, (0, 16384, 24, 3), "fn rows"),
        ];
        for (params, (fn_rows, fn_cols, base_len, scale_len), expected) in cases {
            let error = validate_params(&params, fn_rows, fn_cols, base_len, scale_len)
                .expect_err("mHC validation accepted a bad parameter")
                .to_string();
            assert!(
                error.contains(expected),
                "mHC validation error {error:?} does not name {expected}"
            );
        }
    }

    #[test]
    fn mhc_transposed_comb_changes_the_update() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 11);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        let mixes = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        let comb = &mixes.comb;
        let transposed: Vec<f64> = (0..mult)
            .flat_map(|j| (0..mult).map(move |k| comb[k * mult + j]))
            .collect();
        assert_ne!(
            transposed, mixes.comb,
            "mHC transposed comb control needs an asymmetric combination matrix"
        );
        let straight = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &mixes.post,
            &mixes.comb,
            false,
            false,
        );
        let flipped = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &mixes.post,
            &transposed,
            false,
            false,
        );
        assert_ne!(straight, flipped, "mHC transposed comb control escaped");
    }

    #[test]
    fn mhc_sinkhorn_eps_changes_the_combination() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 23);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        let with_eps = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        let without =
            Oracle { eps: 0.0, ..oracle }.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        assert_ne!(
            with_eps.comb, without.comb,
            "mHC sinkhorn eps control escaped"
        );
    }

    #[test]
    fn mhc_norm_eps_changes_the_mixes() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 29);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-20,
        };
        let tight = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        let loose = Oracle {
            norm_eps: 1e-5,
            ..oracle
        }
        .mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        assert_ne!(tight.pre, loose.pre, "mHC norm_eps control escaped");
        assert_ne!(tight.post, loose.post, "mHC norm_eps control escaped");
    }

    #[test]
    fn mhc_scale_before_the_matmul_changes_the_mixes() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 31);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        let mixes = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        // The scale of each mixer row applied before the matmul.
        let cols = mult * hidden;
        let mut scaled = fx.fn_w.clone();
        for c in 0..(2 + mult) * mult {
            let scale = if c < mult {
                fx.scale[0]
            } else if c < 2 * mult {
                fx.scale[1]
            } else {
                fx.scale[2]
            };
            for d in 0..cols {
                scaled[c * cols + d] *= scale;
            }
        }
        let early = oracle.mixes(&fx.stream, &scaled, &fx.base, &fx.scale);
        assert_ne!(
            early.pre, mixes.pre,
            "mHC scale-before-matmul control escaped"
        );
        assert_ne!(
            early.post, mixes.post,
            "mHC scale-before-matmul control escaped"
        );
    }

    #[test]
    fn mhc_fp32_mode_differs_from_pre_rounded_coefficients() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 37);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        let mixes = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        let rounded_post: Vec<f64> = mixes.post.iter().copied().map(round_bf16).collect();
        let rounded_comb: Vec<f64> = mixes.comb.iter().copied().map(round_bf16).collect();
        assert_ne!(
            rounded_post, mixes.post,
            "mHC FP32 control needs coefficients that are not BF16 exact"
        );
        let straight = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &mixes.post,
            &mixes.comb,
            false,
            false,
        );
        let prerounded = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &rounded_post,
            &rounded_comb,
            false,
            false,
        );
        assert_ne!(
            straight, prerounded,
            "mHC FP32 mode vs pre-rounded coefficients control escaped"
        );
    }

    #[test]
    fn mhc_post_comb_bf16_differs_from_unrounded_coefficients() {
        let mult = 4usize;
        let hidden = 8usize;
        let fx = fixture(mult as u32, hidden as u32, 1, 41);
        let oracle = Oracle {
            mult,
            hidden,
            iters: 20,
            eps: 1e-6,
            norm_eps: 1e-5,
        };
        let mixes = oracle.mixes(&fx.stream, &fx.fn_w, &fx.base, &fx.scale);
        let bf16_mode = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &mixes.post,
            &mixes.comb,
            true,
            false,
        );
        let fp32_mode = oracle.update(
            &fx.sublayer,
            &fx.residual,
            &mixes.post,
            &mixes.comb,
            false,
            false,
        );
        assert_ne!(
            bf16_mode, fp32_mode,
            "mHC PostCombBf16 vs unrounded coefficients control escaped"
        );
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn mhc_kernel_matches_oracle_both_modes() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = MhcOps::new(&device.ctx)?;
        const MULT: u32 = 4;
        const ROWS: u32 = 3;
        const ITERS: u32 = 20;
        const EPS: f32 = 1e-6;
        const PRE_ABS_TOL: f32 = 1e-5;
        const POST_ABS_TOL: f32 = 1e-5;
        const COMB_ABS_TOL: f32 = 1e-4;
        const OUTPUT_ULP_TOL: u32 = 1;
        // One leg per bound configuration: the four DeepSeek-V4.1 legs (F32 fn
        // storage, plain comb application, norm_eps 1e-20 with Fp32 rounding)
        // and the GLM-5.3 leg (hidden 4096, norm_eps 1e-5, BF16 fn storage,
        // transposed comb application, PostCombBf16).
        struct Leg {
            hidden: u32,
            norm_eps: f32,
            rounding: HcRounding,
            fn_bf16: bool,
            transposed: bool,
        }
        let legs = [
            Leg {
                hidden: 4096,
                norm_eps: 1e-20,
                rounding: HcRounding::Fp32,
                fn_bf16: false,
                transposed: false,
            },
            Leg {
                hidden: 4096,
                norm_eps: 1e-5,
                rounding: HcRounding::PostCombBf16,
                fn_bf16: false,
                transposed: false,
            },
            Leg {
                hidden: 5120,
                norm_eps: 1e-20,
                rounding: HcRounding::Fp32,
                fn_bf16: false,
                transposed: false,
            },
            Leg {
                hidden: 5120,
                norm_eps: 1e-5,
                rounding: HcRounding::PostCombBf16,
                fn_bf16: false,
                transposed: false,
            },
            Leg {
                hidden: 4096,
                norm_eps: 1e-5,
                rounding: HcRounding::PostCombBf16,
                fn_bf16: true,
                transposed: true,
            },
        ];
        for leg in &legs {
            let (hidden, rounding) = (leg.hidden, leg.rounding);
            let norm_eps = leg.norm_eps;
            let orientation = if leg.transposed {
                HcCombOrientation::Transposed
            } else {
                HcCombOrientation::Plain
            };
            let params = HcParams {
                mult: MULT,
                hidden,
                iters: ITERS,
                eps: EPS,
                norm_eps,
            };
            let oracle = Oracle {
                mult: MULT as usize,
                hidden: hidden as usize,
                iters: ITERS as usize,
                eps: f64::from(EPS),
                norm_eps: f64::from(norm_eps),
            };
            let fx = fixture(MULT, hidden, ROWS, 7);
            // BF16 fn storage: the published fn values rounded to BF16; the
            // oracle consumes the exact BF16->FP32 upcast.
            let fn_bf16_stored: Vec<bf16> = fx.fn_w.iter().map(|v| bf16::from_f32(*v)).collect();
            let fn_oracle: Vec<f32> = if leg.fn_bf16 {
                fn_bf16_stored.iter().map(|v| v.to_f32()).collect()
            } else {
                fx.fn_w.clone()
            };
            let xb = testing::upload_device(&device, fx.stream.clone())?;
            let fnb_f32 = testing::upload_device(&device, fx.fn_w.clone())?;
            let fnb_bf16 = testing::upload_device(&device, fn_bf16_stored)?;
            let baseb = testing::upload_device(&device, fx.base.clone())?;
            let scaleb = testing::upload_device(&device, fx.scale.clone())?;
            let subb = testing::upload_device(&device, fx.sublayer.clone())?;
            let resb = testing::upload_device(&device, fx.residual.clone())?;
            device.finish_upload()?;
            let pre = testing::zeros::<f32>(s, (ROWS * MULT) as usize)?;
            let post = testing::zeros::<f32>(s, (ROWS * MULT) as usize)?;
            let comb = testing::zeros::<f32>(s, (ROWS * MULT * MULT) as usize)?;
            let collapsed = testing::zeros::<bf16>(s, (ROWS * hidden) as usize)?;
            let updated = testing::zeros::<bf16>(s, (ROWS * MULT * hidden) as usize)?;
            let other_mode = testing::zeros::<bf16>(s, (ROWS * MULT * hidden) as usize)?;
            let stream_stride = u64::from(MULT * hidden);
            let comb_stride = u64::from(MULT * MULT);
            let x = testing::view(
                &xb,
                s,
                [ROWS, MULT, hidden],
                [stream_stride, u64::from(hidden), 1],
            )?;
            let fn_w = if leg.fn_bf16 {
                HcFnW::Bf16(testing::view(
                    &fnb_bf16,
                    s,
                    [mix_rows(MULT) as u32, MULT * hidden],
                    [u64::from(MULT * hidden), 1],
                )?)
            } else {
                HcFnW::F32(testing::view(
                    &fnb_f32,
                    s,
                    [mix_rows(MULT) as u32, MULT * hidden],
                    [u64::from(MULT * hidden), 1],
                )?)
            };
            let base = testing::view(&baseb, s, [mix_rows(MULT) as u32], [1])?;
            let scale = testing::view(&scaleb, s, [3], [1])?;
            let residual = testing::view(
                &resb,
                s,
                [ROWS, MULT, hidden],
                [stream_stride, u64::from(hidden), 1],
            )?;
            let pre_view = testing::view(&pre, s, [ROWS, MULT], [u64::from(MULT), 1])?;
            let post_view = testing::view(&post, s, [ROWS, MULT], [u64::from(MULT), 1])?;
            let comb_view = testing::view(
                &comb,
                s,
                [ROWS, MULT, MULT],
                [comb_stride, u64::from(MULT), 1],
            )?;
            ops.mixes(
                s,
                x,
                &HcWeights { fn_w, base, scale },
                &params,
                HcMixes {
                    pre: &mut TensorMut(pre_view),
                    post: &mut TensorMut(post_view),
                    comb: &mut TensorMut(comb_view),
                },
            )?;
            ops.collapse(
                s,
                x,
                pre_view,
                &mut TensorMut(testing::view(
                    &collapsed,
                    s,
                    [ROWS, hidden],
                    [u64::from(hidden), 1],
                )?),
            )?;
            ops.update(
                s,
                testing::view(&subb, s, [ROWS, hidden], [u64::from(hidden), 1])?,
                residual,
                HcUpdate {
                    post: post_view,
                    comb: comb_view,
                    rounding,
                    orientation,
                },
                &mut TensorMut(testing::view(
                    &updated,
                    s,
                    [ROWS, MULT, hidden],
                    [stream_stride, u64::from(hidden), 1],
                )?),
            )?;
            let other_rounding = match rounding {
                HcRounding::Fp32 => HcRounding::PostCombBf16,
                HcRounding::PostCombBf16 => HcRounding::Fp32,
            };
            ops.update(
                s,
                testing::view(&subb, s, [ROWS, hidden], [u64::from(hidden), 1])?,
                residual,
                HcUpdate {
                    post: post_view,
                    comb: comb_view,
                    rounding: other_rounding,
                    orientation,
                },
                &mut TensorMut(testing::view(
                    &other_mode,
                    s,
                    [ROWS, MULT, hidden],
                    [stream_stride, u64::from(hidden), 1],
                )?),
            )?;
            s.synchronize()?;
            let pre_host = s.clone_dtoh(&pre)?;
            let post_host = s.clone_dtoh(&post)?;
            let comb_host = s.clone_dtoh(&comb)?;
            let collapsed_host = s.clone_dtoh(&collapsed)?;
            let updated_host = s.clone_dtoh(&updated)?;
            let other_host = s.clone_dtoh(&other_mode)?;
            let mult = MULT as usize;
            let hidden = hidden as usize;
            for row in 0..ROWS as usize {
                let stream = &fx.stream[row * mult * hidden..(row + 1) * mult * hidden];
                let mixes = oracle.mixes(stream, &fn_oracle, &fx.base, &fx.scale);
                for j in 0..mult {
                    let got = pre_host[row * mult + j];
                    ensure!(
                        (f64::from(got) - mixes.pre[j]).abs() <= f64::from(PRE_ABS_TOL),
                        "mHC pre row={row} j={j} {got:e} != {:e}",
                        mixes.pre[j]
                    );
                    let got = post_host[row * mult + j];
                    ensure!(
                        (f64::from(got) - mixes.post[j]).abs() <= f64::from(POST_ABS_TOL),
                        "mHC post row={row} j={j} {got:e} != {:e}",
                        mixes.post[j]
                    );
                }
                for k in 0..mult * mult {
                    let got = comb_host[row * mult * mult + k];
                    ensure!(
                        (f64::from(got) - mixes.comb[k]).abs() <= f64::from(COMB_ABS_TOL),
                        "mHC comb row={row} k={k} {got:e} != {:e}",
                        mixes.comb[k]
                    );
                }
                let kernel_pre: Vec<f64> = pre_host[row * mult..row * mult + mult]
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let collapse = oracle.collapse(stream, &kernel_pre);
                for (h, want) in collapse.iter().enumerate() {
                    let got = collapsed_host[row * hidden + h];
                    let distance = bf16_ulp_distance(got, *want);
                    let abs = (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs();
                    ensure!(
                        distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                        "mHC collapse row={row} h={h} bf16 ulp distance {distance} abs {abs}"
                    );
                }
                if row == 0 {
                    let mut planted = stream.to_vec();
                    let mut shifted = vec![bf16::ZERO; hidden];
                    shifted[..hidden - 1].copy_from_slice(&stream[2 * hidden + 1..3 * hidden]);
                    shifted[hidden - 1] = stream[2 * hidden];
                    planted[2 * hidden..3 * hidden].copy_from_slice(&shifted);
                    let wrong = oracle.collapse(&planted, &kernel_pre);
                    let mut caught = 0usize;
                    for h in 0..hidden {
                        let got = collapsed_host[row * hidden + h];
                        let distance = bf16_ulp_distance(got, wrong[h]);
                        let abs = (f64::from(got.to_f32()) - f64::from(wrong[h].to_f32())).abs();
                        if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                            caught += 1;
                        }
                    }
                    ensure!(
                        caught > 0,
                        "mHC planted wrong-stream-index collapse mutant was not caught ({hidden} hidden, {rounding:?})"
                    );
                }
                if row == 0 && leg.fn_bf16 {
                    // Planted dtype mutant: a BF16-arithmetic fn dot (every
                    // product and partial sum rounded to BF16) instead of
                    // the exact FP32 register upcast.
                    let n = mult * hidden;
                    let x: Vec<f64> = stream.iter().map(|v| f64::from(v.to_f32())).collect();
                    let sumsq: f64 = x.iter().map(|v| v * v).sum();
                    let rsqrt = 1.0 / (sumsq / n as f64 + f64::from(norm_eps)).sqrt();
                    let mut caught = 0usize;
                    let mut teeth = 0usize;
                    for c in 0..2 * mult {
                        let mut acc = 0.0f64;
                        for d in 0..n {
                            let w = f64::from(fn_oracle[c * n + d]);
                            acc = round_bf16(acc + round_bf16(w * x[d]));
                        }
                        let value = acc * rsqrt;
                        let (scale, exact, got) = if c < mult {
                            (
                                f64::from(fx.scale[0]),
                                mixes.pre[c],
                                pre_host[row * mult + c],
                            )
                        } else {
                            (
                                f64::from(fx.scale[1]),
                                mixes.post[c - mult],
                                post_host[row * mult + c - mult],
                            )
                        };
                        let mutant = if c < mult {
                            sigmoid(value * scale + f64::from(fx.base[c])) + f64::from(EPS)
                        } else {
                            2.0 * sigmoid(value * scale + f64::from(fx.base[c]))
                        };
                        if (mutant - exact).abs() > f64::from(PRE_ABS_TOL) {
                            teeth += 1;
                        }
                        if (f64::from(got) - mutant).abs() > f64::from(PRE_ABS_TOL) {
                            caught += 1;
                        }
                    }
                    ensure!(
                        teeth > 0,
                        "mHC dtype mutant is indistinguishable from the exact upcast on every mixer"
                    );
                    ensure!(
                        caught > 0,
                        "mHC dtype mutant was not caught: the kernel mixes agree with the BF16-arithmetic fn dot"
                    );
                }
                let residual = &fx.residual[row * mult * hidden..(row + 1) * mult * hidden];
                let sublayer = &fx.sublayer[row * hidden..(row + 1) * hidden];
                let kernel_post: Vec<f64> = post_host[row * mult..row * mult + mult]
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let kernel_comb: Vec<f64> = comb_host[row * mult * mult..(row + 1) * mult * mult]
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let update = oracle.update(
                    sublayer,
                    residual,
                    &kernel_post,
                    &kernel_comb,
                    matches!(rounding, HcRounding::PostCombBf16),
                    leg.transposed,
                );
                for j in 0..mult {
                    for h in 0..hidden {
                        let got = updated_host[(row * mult + j) * hidden + h];
                        let distance = bf16_ulp_distance(got, update[j * hidden + h]);
                        let abs = (f64::from(got.to_f32())
                            - f64::from(update[j * hidden + h].to_f32()))
                        .abs();
                        ensure!(
                            distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                            "mHC update row={row} j={j} h={h} bf16 ulp distance {distance} abs {abs}"
                        );
                    }
                }
                if leg.transposed {
                    // Planted orientation mutant: plain comb application on
                    // the GLM leg.
                    let swapped = oracle.update(
                        sublayer,
                        residual,
                        &kernel_post,
                        &kernel_comb,
                        matches!(rounding, HcRounding::PostCombBf16),
                        false,
                    );
                    let mut caught = 0usize;
                    for k in 0..mult * hidden {
                        let got = updated_host[row * mult * hidden + k];
                        let distance = bf16_ulp_distance(got, swapped[k]);
                        let abs = (f64::from(got.to_f32()) - f64::from(swapped[k].to_f32())).abs();
                        if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                            caught += 1;
                        }
                    }
                    ensure!(
                        caught > 0,
                        "mHC orientation mutant was not caught (row {row})"
                    );
                }
            }
            ensure!(
                updated_host != other_host,
                "mHC rounding mode control escaped: both modes produced the same update"
            );
        }
        let error = validate_params(
            &HcParams {
                mult: 4,
                hidden: 4096,
                iters: 20,
                eps: 1e-6,
                norm_eps: 1e-5,
            },
            0,
            16384,
            24,
            3,
        )
        .expect_err("mHC validation accepted a zero-length fn")
        .to_string();
        ensure!(
            error.contains("fn rows"),
            "mHC zero-length fn was not rejected: {error}"
        );
        Ok(())
    }

    fn glm_runtime_policy(device: &Device) -> Result<()> {
        use crate::backend::cuda::hc::{HcRuntime, HcSublayerWeights, glm_binding};
        use crate::model::{HcPreSource, HcSublayer};
        let config = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/glm53-flash-config.json"
        )))?;
        let spec = crate::model_spec::ModelSpec::glm5_next(&config, &serde_json::Value::Null)?;
        let binding = glm_binding(&spec)?;
        let p = binding.params;
        let (rows, mult, hidden) = (3usize, p.mult as usize, p.hidden as usize);
        ensure!(
            mult == 4 && hidden == 4096 && p.iters == 20,
            "GLM mHC production geometry"
        );
        ensure!(
            binding.policy.attention == HcPreSource::Own && binding.policy.ffn == HcPreSource::Own,
            "GLM mHC requires own pre-mixes"
        );
        let s = &device.stream;
        let oracle = Oracle {
            mult,
            hidden,
            iters: p.iters as usize,
            eps: f64::from(p.eps),
            norm_eps: f64::from(p.norm_eps),
        };
        let attn = fixture(p.mult, p.hidden, rows as u32, 7);
        let ffn = fixture(p.mult, p.hidden, rows as u32, 11);
        let upload = |fx: &Fixture| -> Result<HcSublayerWeights> {
            let fn_bytes: Vec<_> = fx
                .fn_w
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
                .collect();
            let base: Vec<_> = fx.base.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let scale: Vec<_> = fx.scale.iter().flat_map(|v| v.to_ne_bytes()).collect();
            Ok(HcSublayerWeights {
                fn_w: s.clone_htod(&fn_bytes)?,
                base: s.clone_htod(&base)?,
                scale: s.clone_htod(&scale)?,
            })
        };
        let mut controls = 0usize;
        for graph in [false, true] {
            let mut runtime = HcRuntime::new(&device.ctx, s, binding, 1, rows)?;
            runtime.bind_layer(0, upload(&attn)?, upload(&ffn)?)?;
            let embedded = s.clone_htod(&attn.sublayer)?;
            runtime.begin_embedded(s, rows, &embedded)?;
            let broadcast = s.clone_dtoh(&runtime.stream)?;
            for row in 0..rows {
                for copy in 0..mult {
                    ensure!(
                        broadcast[(row * mult + copy) * hidden..(row * mult + copy + 1) * hidden]
                            == attn.sublayer[row * hidden..(row + 1) * hidden],
                        "mHC embedded broadcast mismatch"
                    );
                }
            }
            s.memcpy_htod(&attn.stream, &mut runtime.stream)?;
            let mut host_stream = attn.stream.clone();
            let mut branch = s.alloc_zeros(rows * hidden)?;
            let mut prior_pre = vec![0f32; rows * mult];
            for (stage, fx) in [(HcSublayer::Attention, &attn), (HcSublayer::Ffn, &ffn)] {
                let weights: Vec<_> = fx
                    .fn_w
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_f32())
                    .collect();
                let pre_run = || runtime.hc_pre(s, 0, stage, HcPreSource::Own, rows, &mut branch);
                if graph {
                    testing::capture(s, pre_run)?.launch()?;
                } else {
                    let mut run = pre_run;
                    run()?;
                }
                let pre = s.clone_dtoh(&runtime.mix_pre)?;
                let post = s.clone_dtoh(&runtime.mix_post)?;
                let comb = s.clone_dtoh(&runtime.mix_comb)?;
                let collapsed = s.clone_dtoh(&branch)?;
                for row in 0..rows {
                    let stream = &host_stream[row * mult * hidden..(row + 1) * mult * hidden];
                    let mixes = oracle.mixes(stream, &weights, &fx.base, &fx.scale);
                    for j in 0..mult {
                        ensure!(
                            (f64::from(pre[row * mult + j]) - mixes.pre[j]).abs() <= 1e-5,
                            "GLM runtime pre coefficient mismatch"
                        );
                        ensure!(
                            (f64::from(post[row * mult + j]) - mixes.post[j]).abs() <= 1e-5,
                            "GLM runtime post coefficient mismatch"
                        );
                    }
                    for j in 0..mult * mult {
                        ensure!(
                            (f64::from(comb[row * mult * mult + j]) - mixes.comb[j]).abs() <= 1e-4,
                            "GLM runtime comb coefficient mismatch"
                        );
                    }
                    let own: Vec<_> = pre[row * mult..(row + 1) * mult]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect();
                    let expected = oracle.collapse(stream, &own);
                    for (&got, &want) in collapsed[row * hidden..(row + 1) * hidden]
                        .iter()
                        .zip(&expected)
                    {
                        ensure!(
                            bf16_ulp_distance(got, want) <= 1
                                || (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                                    <= OUTPUT_ABS_TOL,
                            "GLM own-policy collapse mismatch"
                        );
                    }
                    if matches!(stage, HcSublayer::Ffn) {
                        let stale: Vec<_> = prior_pre[row * mult..(row + 1) * mult]
                            .iter()
                            .map(|&v| f64::from(v))
                            .collect();
                        let wrong = oracle.collapse(stream, &stale);
                        controls += expected
                            .iter()
                            .zip(wrong)
                            .filter(|(a, b)| {
                                bf16_ulp_distance(**a, *b) > 1
                                    && (a.to_f32() - b.to_f32()).abs() > OUTPUT_ABS_TOL as f32
                            })
                            .count();
                    }
                }
                prior_pre = pre;
                s.memcpy_htod(&fx.sublayer, &mut branch)?;
                let post_run = || runtime.hc_post(s, 0, stage, rows, &branch);
                if graph {
                    testing::capture(s, post_run)?.launch()?;
                } else {
                    let mut run = post_run;
                    run()?;
                }
                let updated = s.clone_dtoh(&runtime.stream)?;
                for row in 0..rows {
                    let ps: Vec<_> = post[row * mult..(row + 1) * mult]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect();
                    let cs: Vec<_> = comb[row * mult * mult..(row + 1) * mult * mult]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect();
                    let expected = oracle.update(
                        &fx.sublayer[row * hidden..(row + 1) * hidden],
                        &host_stream[row * mult * hidden..(row + 1) * mult * hidden],
                        &ps,
                        &cs,
                        true,
                        true,
                    );
                    ensure!(
                        updated[row * mult * hidden..(row + 1) * mult * hidden] == expected,
                        "GLM runtime post rounding/orientation mismatch"
                    );
                }
                host_stream = updated;
            }
            let final_run = || runtime.collapse_final(s, rows, &mut branch);
            if graph {
                testing::capture(s, final_run)?.launch()?;
            } else {
                let mut run = final_run;
                run()?;
            }
            let actual = s.clone_dtoh(&branch)?;
            for row in 0..rows {
                let stream = &host_stream[row * mult * hidden..(row + 1) * mult * hidden];
                let expected = oracle.collapse(stream, &vec![1.0 / mult as f64; mult]);
                ensure!(
                    actual[row * hidden..(row + 1) * hidden] == expected,
                    "GLM uniform head collapse mismatch"
                );
                let stale: Vec<_> = prior_pre[row * mult..(row + 1) * mult]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                controls += expected
                    .iter()
                    .zip(oracle.collapse(stream, &stale))
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
            println!(
                "PASS GLM_MHC_RUNTIME rows={rows} mult={mult} hidden={hidden} graph={graph} own_own=true post_bits=0 uniform_head_bits=0"
            );
        }
        ensure!(controls > 0, "GLM stale-pre policy control escaped");
        println!("PASS GLM_MHC_POLICY_CONTROL differing_elements={controls}");
        Ok(())
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_Q38_GPU_WINDOW=granted"]
    fn mhc_post_comb_matches_definition_reference() -> Result<()> {
        // 50e35cc0:1335-1337,1346-1348 fixture from
        // evidence/G1/roundB/B4/hc-boundary-reference.py; the old fused mode
        // produced 18706 differing bits against this expression.
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = MhcOps::new(&device.ctx)?;
        const ROWS: u32 = 3;
        const MULT: u32 = 4;
        const HIDDEN: u32 = 4096;
        let n_stream = (ROWS * MULT * HIDDEN) as usize;
        let residual: Vec<bf16> = (0..n_stream)
            .map(|i| bf16::from_f32((i as f32 * 73.0 + 19.0) % 257.0 / 32.0 - 4.0))
            .collect();
        let sublayer: Vec<bf16> = (0..(ROWS * HIDDEN) as usize)
            .map(|i| bf16::from_f32((i as f32 * 31.0 + 7.0) % 251.0 / 64.0 - 2.0))
            .collect();
        let post_row = [0.137f32, 0.723, 1.137, 1.823];
        let comb_row = [
            [0.11f32, 0.19, 0.29, 0.41],
            [0.41, 0.11, 0.19, 0.29],
            [0.29, 0.41, 0.11, 0.19],
            [0.19, 0.29, 0.41, 0.11],
        ];
        let post: Vec<f32> = (0..ROWS as usize).flat_map(|_| post_row).collect();
        let comb: Vec<f32> = (0..ROWS as usize)
            .flat_map(|_| comb_row.iter().flat_map(|r| r.iter().copied()))
            .collect();
        let subb = testing::upload_device(&device, sublayer.clone())?;
        let resb = testing::upload_device(&device, residual.clone())?;
        let postb = testing::upload_device(&device, post.clone())?;
        let combb = testing::upload_device(&device, comb.clone())?;
        device.finish_upload()?;
        let updated = testing::zeros::<bf16>(s, n_stream)?;
        ops.update(
            s,
            testing::view(&subb, s, [ROWS, HIDDEN], [u64::from(HIDDEN), 1])?,
            testing::view(
                &resb,
                s,
                [ROWS, MULT, HIDDEN],
                [u64::from(MULT * HIDDEN), u64::from(HIDDEN), 1],
            )?,
            HcUpdate {
                post: testing::view(&postb, s, [ROWS, MULT], [u64::from(MULT), 1])?,
                comb: testing::view(
                    &combb,
                    s,
                    [ROWS, MULT, MULT],
                    [u64::from(MULT * MULT), u64::from(MULT), 1],
                )?,
                rounding: HcRounding::PostCombBf16,
                orientation: HcCombOrientation::Transposed,
            },
            &mut TensorMut(testing::view(
                &updated,
                s,
                [ROWS, MULT, HIDDEN],
                [u64::from(MULT * HIDDEN), u64::from(HIDDEN), 1],
            )?),
        )?;
        let actual = s.clone_dtoh(&updated)?;
        let oracle = Oracle {
            mult: MULT as usize,
            hidden: HIDDEN as usize,
            iters: 1,
            eps: 0.0,
            norm_eps: 1e-5,
        };
        let expected: Vec<_> = (0..ROWS as usize)
            .flat_map(|row| {
                let hidden = HIDDEN as usize;
                let mult = MULT as usize;
                oracle.update(
                    &sublayer[row * hidden..(row + 1) * hidden],
                    &residual[row * mult * hidden..(row + 1) * mult * hidden],
                    &post[row * mult..(row + 1) * mult]
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect::<Vec<_>>(),
                    &comb[row * mult * mult..(row + 1) * mult * mult]
                        .iter()
                        .map(|v| f64::from(*v))
                        .collect::<Vec<_>>(),
                    true,
                    true,
                )
            })
            .collect();
        ensure!(actual.len() == expected.len(), "mHC oracle output extent");
        let mismatches = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            mismatches == 0,
            "DEFECT mhc definition reference: {mismatches} of {} bits differ",
            actual.len()
        );
        eprintln!(
            "PASS MHC_POST_COMB_DEFINITION rows={ROWS} mult={MULT} hidden={HIDDEN} bit_mismatches=0 threshold=0"
        );
        glm_runtime_policy(&device)
    }
}
