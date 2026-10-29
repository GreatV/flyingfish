//! Model-independent mHC runtime: the stream and pre-mix row state, the
//! setup-bound execution semantics and the Hc stage execution of the frozen
//! nine-stage layer program (model.py Block; modeling_glm5_next.py; mHC
//! arXiv 2512.24880).
use crate::backend::cuda::contracts::{Tensor, TensorMut, dense_layout, view};
use crate::backend::cuda::cubin;
use crate::backend::cuda::mhc::{
    HcCombOrientation, HcFnDtype, HcFnW, HcMixes, HcParams, HcRounding, HcUpdate, HcWeights,
    MhcOps, mix_rows, validate_params,
};
use crate::backend::cuda::ops::flat;
use crate::model::{HcPrePolicy, HcPreSource, HcSublayer};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::{mem::size_of, sync::Arc};

/// Setup-bound mHC execution semantics, derived from the model spec once at
/// binding time and fixed for the runtime's lifetime.
#[derive(Clone, Copy, Debug)]
pub struct HcBinding {
    /// Mixer parameters: mult, hidden, iters, eps, norm_eps.
    pub params: HcParams,
    /// Pre-mix source of each HcPre stage.
    pub policy: HcPrePolicy,
    /// Update coefficient rounding.
    pub rounding: HcRounding,
    /// Update comb application orientation.
    pub orientation: HcCombOrientation,
    /// fn projection storage dtype.
    pub fn_dtype: HcFnDtype,
}

/// Owned device weight bytes of one mHC sublayer instance: fn in the bound
/// dtype, base and scale F32; the audited manifest F32/BF16 path
/// (`Loader::load_bytes`) uploads them byte-exact.
pub struct HcSublayerWeights {
    pub fn_w: CudaSlice<u8>,
    pub base: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
}

/// mHC weights of one backbone layer: the attention and FFN instances.
pub struct HcLayerWeights {
    pub attn: HcSublayerWeights,
    pub ffn: HcSublayerWeights,
}

pub struct HcRuntime {
    ops: MhcOps,
    embed_streams: CudaFunction,
    pub(crate) binding: HcBinding,
    weights: Vec<Option<HcLayerWeights>>,
    rows: usize,
    /// `[rows, mult*hidden]` bf16, dense `[rows, mult, hidden]` per row.
    pub(crate) stream: CudaSlice<bf16>,
    stream_next: CudaSlice<bf16>,
    /// `[rows, mult]` f32 pre-mix carried across layers; bound when the
    /// attention HcPre collapses with `HcPreSource::Carried`.
    pub(crate) carry: Option<CudaSlice<f32>>,
    /// `[rows, mult]` f32 attention pre of the current layer; bound when the
    /// FFN HcPre collapses with `HcPreSource::AttentionOwn`.
    pub(crate) attn_pre: Option<CudaSlice<f32>>,
    /// `[rows, mult]` f32 pre mixer output of the in-flight sublayer.
    pub(crate) mix_pre: CudaSlice<f32>,
    /// `[rows, mult]` f32 post mixer output of the in-flight sublayer.
    pub(crate) mix_post: CudaSlice<f32>,
    /// `[rows, mult*mult]` f32 combination matrix, rows indexed `[j=out, k=in]`.
    pub(crate) mix_comb: CudaSlice<f32>,
    /// `[rows, mult]` f32 constant one-hot on copy 0: the carried pre-mix of
    /// the first layer of a forward; bound with the carry.
    one_hot: Option<CudaSlice<f32>>,
}

/// Dense view over one owned weight-byte allocation.
fn weights_view<T: Copy, const R: usize>(
    bytes: &CudaSlice<u8>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
    name: &str,
) -> Result<Tensor<T, R>> {
    let (stride, count) = dense_layout(shape)?;
    let span = count
        .checked_mul(size_of::<T>() as u64)
        .context("mHC weight byte count overflow")?;
    ensure!(
        u64::try_from(bytes.len())? == span,
        "mHC {name} allocation holds {} bytes; shape {shape:?} needs {span}",
        bytes.len()
    );
    let (ptr, _) = bytes.device_ptr(s);
    unsafe { Tensor::new(ptr, shape, stride, span) }
}

/// Constant one-hot pre-mix device buffer: 1.0 on copy 0 of every row.
fn alloc_one_hot(s: &Arc<CudaStream>, rows: usize, mult: usize) -> Result<CudaSlice<f32>> {
    let mut one_hot = vec![0.0f32; rows * mult];
    for row in one_hot.chunks_mut(mult) {
        row[0] = 1.0;
    }
    let mut buffer = s.alloc_zeros(rows * mult)?;
    s.memcpy_htod(&one_hot, &mut buffer.slice_mut(..rows * mult))?;
    Ok(buffer)
}

impl HcRuntime {
    /// Allocates the runtime state for `layers` layers and `rows` rows under
    /// one fixed binding; the per-layer weights are bound separately.
    pub fn new(
        ctx: &Arc<CudaContext>,
        s: &Arc<CudaStream>,
        binding: HcBinding,
        layers: usize,
        rows: usize,
    ) -> Result<Self> {
        let mult = binding.params.mult;
        let basis = u32::try_from(mix_rows(mult)).context("mHC basis rows exceed u32")?;
        validate_params(
            &binding.params,
            basis,
            mult * binding.params.hidden,
            basis,
            3,
        )?;
        ensure!(layers > 0, "mHC runtime needs at least one layer");
        ensure!(
            rows > 0,
            "mHC runtime needs at least one row (workspace capacity)"
        );
        let (mult, hidden) = (mult as usize, binding.params.hidden as usize);
        let carries = binding.policy.attention == HcPreSource::Carried;
        Ok(Self {
            ops: MhcOps::new(ctx)?,
            embed_streams: cubin::module(ctx, "embed")?.load_function("embed_streams")?,
            binding,
            weights: (0..layers).map(|_| None).collect(),
            rows,
            stream: s.alloc_zeros(rows * mult * hidden)?,
            stream_next: s.alloc_zeros(rows * mult * hidden)?,
            carry: if carries {
                Some(s.alloc_zeros(rows * mult)?)
            } else {
                None
            },
            attn_pre: if binding.policy.ffn == HcPreSource::AttentionOwn {
                Some(s.alloc_zeros(rows * mult)?)
            } else {
                None
            },
            mix_pre: s.alloc_zeros(rows * mult)?,
            mix_post: s.alloc_zeros(rows * mult)?,
            mix_comb: s.alloc_zeros(rows * mult * mult)?,
            one_hot: if carries {
                Some(alloc_one_hot(s, rows, mult)?)
            } else {
                None
            },
        })
    }

    /// Binds the weight bytes of one backbone layer.
    pub fn bind_layer(
        &mut self,
        layer: usize,
        attn: HcSublayerWeights,
        ffn: HcSublayerWeights,
    ) -> Result<()> {
        ensure!(
            layer < self.weights.len(),
            "mHC layer {layer} is outside the bound {} layers",
            self.weights.len()
        );
        self.weights[layer] = Some(HcLayerWeights { attn, ffn });
        Ok(())
    }

    fn check_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.rows,
            "mHC row count {rows} is outside the runtime capacity {}",
            self.rows
        );
        Ok(())
    }

    fn sublayer(&self, layer: usize, sublayer: HcSublayer) -> Result<&HcSublayerWeights> {
        let weights = self
            .weights
            .get(layer)
            .with_context(|| format!("mHC layer {layer} has no bound weights"))?
            .as_ref()
            .context("mHC layer {layer} has no bound weights")?;
        Ok(match sublayer {
            HcSublayer::Attention => &weights.attn,
            HcSublayer::Ffn => &weights.ffn,
        })
    }

    fn hc_weights(
        &self,
        s: &Arc<CudaStream>,
        layer: usize,
        sublayer: HcSublayer,
    ) -> Result<HcWeights> {
        let sub = self.sublayer(layer, sublayer)?;
        let mult = self.binding.params.mult;
        let basis = u32::try_from(mix_rows(mult)).context("mHC basis rows exceed u32")?;
        let fn_w = match self.binding.fn_dtype {
            HcFnDtype::F32 => HcFnW::F32(weights_view(
                &sub.fn_w,
                s,
                [basis, mult * self.binding.params.hidden],
                "fn f32",
            )?),
            HcFnDtype::Bf16 => HcFnW::Bf16(weights_view(
                &sub.fn_w,
                s,
                [basis, mult * self.binding.params.hidden],
                "fn bf16",
            )?),
        };
        Ok(HcWeights {
            fn_w,
            base: weights_view(&sub.base, s, [basis], "base")?,
            scale: weights_view(&sub.scale, s, [3], "scale")?,
        })
    }

    /// Host copy of one sublayer's fn, base and scale bytes as uploaded.
    #[cfg(test)]
    pub(crate) fn sublayer_weight_host(
        &self,
        s: &Arc<CudaStream>,
        layer: usize,
        sublayer: HcSublayer,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let weights = self.sublayer(layer, sublayer)?;
        Ok((
            s.clone_dtoh(&weights.fn_w)?,
            s.clone_dtoh(&weights.base)?,
            s.clone_dtoh(&weights.scale)?,
        ))
    }

    fn stream_view(&self, s: &Arc<CudaStream>, rows: usize) -> Result<Tensor<bf16, 3>> {
        let (mult, hidden) = (self.binding.params.mult, self.binding.params.hidden);
        view(
            &self.stream,
            s,
            0,
            [rows as u32, mult, hidden],
            [u64::from(mult * hidden), u64::from(hidden), 1],
        )
    }

    fn mix_views(
        &self,
        s: &Arc<CudaStream>,
        rows: usize,
    ) -> Result<(Tensor<f32, 2>, Tensor<f32, 2>, Tensor<f32, 3>)> {
        let mult = self.binding.params.mult;
        Ok((
            view(
                &self.mix_pre,
                s,
                0,
                [rows as u32, mult],
                [u64::from(mult), 1],
            )?,
            view(
                &self.mix_post,
                s,
                0,
                [rows as u32, mult],
                [u64::from(mult), 1],
            )?,
            view(
                &self.mix_comb,
                s,
                0,
                [rows as u32, mult, mult],
                [u64::from(mult * mult), u64::from(mult), 1],
            )?,
        ))
    }

    /// HcPre: the sublayer's mixes from the pre-collapse stream, the collapse
    /// into the sublayer input, then the pre-mix carry update — under the
    /// Carried policy the FFN pre crosses to the next layer's attention
    /// collapse, under AttentionOwn the attention pre feeds this layer's FFN
    /// collapse, under Own the collapse consumes the just-computed pre itself
    /// and no pre crosses sublayers.
    pub fn hc_pre(
        &mut self,
        s: &Arc<CudaStream>,
        layer: usize,
        sublayer: HcSublayer,
        source: HcPreSource,
        rows: usize,
        x: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        self.check_rows(rows)?;
        let weights = self.hc_weights(s, layer, sublayer)?;
        let (pre, post, comb) = self.mix_views(s, rows)?;
        self.ops.mixes(
            s,
            self.stream_view(s, rows)?,
            &weights,
            &self.binding.params,
            HcMixes {
                pre: &mut TensorMut(pre),
                post: &mut TensorMut(post),
                comb: &mut TensorMut(comb),
            },
        )?;
        let mult = self.binding.params.mult;
        let pre_source = match source {
            HcPreSource::Carried => self
                .carry
                .as_ref()
                .context("mHC carried pre-mix is not bound: the policy allocates no carry")?,
            HcPreSource::AttentionOwn => self
                .attn_pre
                .as_ref()
                .context("mHC attention pre-mix is not bound: the policy allocates no attn_pre")?,
            HcPreSource::Own => &self.mix_pre,
        };
        let pre_view = view(pre_source, s, 0, [rows as u32, mult], [u64::from(mult), 1])?;
        self.ops.collapse(
            s,
            self.stream_view(s, rows)?,
            pre_view,
            &mut TensorMut(view(
                x,
                s,
                0,
                [rows as u32, self.binding.params.hidden],
                [u64::from(self.binding.params.hidden), 1],
            )?),
        )?;
        let count = rows * mult as usize;
        match sublayer {
            HcSublayer::Attention => {
                if let Some(attn_pre) = &mut self.attn_pre {
                    s.memcpy_dtod(
                        &self.mix_pre.slice(..count),
                        &mut attn_pre.slice_mut(..count),
                    )?;
                }
            }
            HcSublayer::Ffn => {
                if let Some(carry) = &mut self.carry {
                    s.memcpy_dtod(&self.mix_pre.slice(..count), &mut carry.slice_mut(..count))?;
                }
            }
        }
        Ok(())
    }

    /// Model-final head collapse: the stream collapses with the last
    /// sublayer's own pre-mix (model.py:1271), no carry update.
    pub fn collapse_final(
        &self,
        s: &Arc<CudaStream>,
        rows: usize,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        self.check_rows(rows)?;
        let mult = self.binding.params.mult;
        let pre_view = view(
            &self.mix_pre,
            s,
            0,
            [rows as u32, mult],
            [u64::from(mult), 1],
        )?;
        self.ops.collapse(
            s,
            self.stream_view(s, rows)?,
            pre_view,
            &mut TensorMut(view(
                out,
                s,
                0,
                [rows as u32, self.binding.params.hidden],
                [u64::from(self.binding.params.hidden), 1],
            )?),
        )
    }

    /// HcPost: residual update of the sublayer output into a new stream.
    pub fn hc_post(
        &mut self,
        s: &Arc<CudaStream>,
        layer: usize,
        sublayer: HcSublayer,
        rows: usize,
        out: &CudaSlice<bf16>,
    ) -> Result<()> {
        self.check_rows(rows)?;
        self.sublayer(layer, sublayer)?;
        let (_, post, comb) = self.mix_views(s, rows)?;
        let (mult, hidden) = (self.binding.params.mult, self.binding.params.hidden);
        self.ops.update(
            s,
            view(out, s, 0, [rows as u32, hidden], [u64::from(hidden), 1])?,
            self.stream_view(s, rows)?,
            HcUpdate {
                post,
                comb,
                rounding: self.binding.rounding,
                orientation: self.binding.orientation,
            },
            &mut TensorMut(view(
                &self.stream_next,
                s,
                0,
                [rows as u32, mult, hidden],
                [u64::from(mult * hidden), u64::from(hidden), 1],
            )?),
        )?;
        let count = rows * (mult * hidden) as usize;
        s.memcpy_dtod(
            &self.stream_next.slice(..count),
            &mut self.stream.slice_mut(..count),
        )?;
        Ok(())
    }

    /// Stream initialization of one forward: `mult` copies of each embedded
    /// token row and the one-hot carried pre-mix on copy 0. Device-only: no
    /// D2H, no host staging, graph-capturable.
    pub fn begin_forward(
        &mut self,
        s: &Arc<CudaStream>,
        rows: usize,
        ids: &CudaSlice<u32>,
        embed: &CudaSlice<bf16>,
    ) -> Result<()> {
        self.check_rows(rows)?;
        ensure!(
            ids.len() >= rows,
            "mHC gather has {} ids for {rows} rows",
            ids.len()
        );
        let mult = self.binding.params.mult as usize;
        let hidden = self.binding.params.hidden as usize;
        let ids_rows = ids.slice(..rows);
        unsafe {
            s.launch_builder(&self.embed_streams)
                .arg(embed)
                .arg(&ids_rows)
                .arg(&mut self.stream.slice_mut(..rows * mult * hidden))
                .arg(&(rows as i32))
                .arg(&(mult as i32))
                .arg(&(hidden as i32))
                .launch(flat(rows * mult * hidden))?;
        }
        if let Some(carry) = &mut self.carry {
            let count = rows * mult;
            let one_hot = self
                .one_hot
                .as_ref()
                .context("mHC carry is bound without its one-hot pre-mix buffer")?;
            s.memcpy_dtod(&one_hot.slice(..count), &mut carry.slice_mut(..count))?;
        }
        Ok(())
    }

    /// Reallocates the per-row state for a larger workspace; the stream and
    /// carry are re-initialized at every forward. The bound weights are
    /// retained.
    pub fn grow(&mut self, s: &Arc<CudaStream>, rows: usize) -> Result<()> {
        ensure!(rows >= self.rows, "mHC runtime rows cannot shrink");
        if rows == self.rows {
            return Ok(());
        }
        let (mult, hidden) = (
            self.binding.params.mult as usize,
            self.binding.params.hidden as usize,
        );
        let carries = self.binding.policy.attention == HcPreSource::Carried;
        self.stream = s.alloc_zeros(rows * mult * hidden)?;
        self.stream_next = s.alloc_zeros(rows * mult * hidden)?;
        if carries {
            self.carry = Some(s.alloc_zeros(rows * mult)?);
            self.one_hot = Some(alloc_one_hot(s, rows, mult)?);
        }
        if self.binding.policy.ffn == HcPreSource::AttentionOwn {
            self.attn_pre = Some(s.alloc_zeros(rows * mult)?);
        }
        self.mix_pre = s.alloc_zeros(rows * mult)?;
        self.mix_post = s.alloc_zeros(rows * mult)?;
        self.mix_comb = s.alloc_zeros(rows * mult * mult)?;
        self.rows = rows;
        Ok(())
    }
}
