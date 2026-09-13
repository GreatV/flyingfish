use crate::sparse::Grid;
use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use ff_core::weights::ModelWeights;

pub(crate) struct Ops {
    pub weights: ModelWeights,
    pub device: Device,
    pub dtype: DType,
    pub query_chunk: usize,
    pub voxel_chunk: usize,
}

impl Ops {
    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        Ok(self.weights.load(name, &self.device)?.to_dtype(dtype)?)
    }
    pub fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let out = x.broadcast_matmul(&self.get(&format!("{prefix}.weight"), x.dtype())?.t()?)?;
        if self.weights.contains(&format!("{prefix}.bias")) {
            Ok(out.broadcast_add(&self.get(&format!("{prefix}.bias"), x.dtype())?)?)
        } else {
            Ok(out)
        }
    }
    pub fn norm(&self, x: &Tensor, prefix: Option<&str>, eps: f64) -> Result<Tensor> {
        let mut out = layer_norm(x, eps)?;
        if let Some(prefix) = prefix {
            out = out
                .broadcast_mul(&self.get(&format!("{prefix}.weight"), DType::F32)?)?
                .broadcast_add(&self.get(&format!("{prefix}.bias"), DType::F32)?)?;
        }
        Ok(out.to_dtype(x.dtype())?)
    }
    fn qk_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let wide = x.to_dtype(DType::F32)?;
        let norm = wide.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
        Ok((wide
            .broadcast_div(&norm)?
            .broadcast_mul(&self.get(&format!("{prefix}.gamma"), DType::F32)?)?
            * (x.dim(D::Minus1)? as f64).sqrt())?
        .to_dtype(x.dtype())?)
    }
    pub fn ff(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        self.linear(
            &self.linear(x, &format!("{prefix}.mlp.0"))?.gelu()?,
            &format!("{prefix}.mlp.2"),
        )
    }
    pub fn self_attention(
        &self,
        x: &Tensor,
        prefix: &str,
        heads: usize,
        qk_norm: bool,
        plan: &AttentionPlan,
    ) -> Result<Tensor> {
        self.self_attention_rotary(x, prefix, heads, qk_norm, plan, None)
    }
    pub fn self_attention_rotary(
        &self,
        x: &Tensor,
        prefix: &str,
        heads: usize,
        qk_norm: bool,
        plan: &AttentionPlan,
        rotary: Option<&Rotary3d>,
    ) -> Result<Tensor> {
        let (rows, width) = x.dims2()?;
        let dim = width / heads;
        let qkv = self
            .linear(x, &format!("{prefix}.to_qkv"))?
            .reshape((rows, 3, heads, dim))?;
        let mut q = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let mut k = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        if qk_norm {
            q = self.qk_norm(&q, &format!("{prefix}.q_rms_norm"))?;
            k = self.qk_norm(&k, &format!("{prefix}.k_rms_norm"))?;
        }
        if let Some(rotary) = rotary {
            q = rotary.apply(&q)?;
            k = rotary.apply(&k)?;
        }
        let q = q.contiguous()?.index_select(&plan.order, 0)?;
        let k = k.contiguous()?.index_select(&plan.order, 0)?;
        let v = v.contiguous()?.index_select(&plan.order, 0)?;
        let mut result = Vec::new();
        for &(start, count) in &plan.ranges {
            result.push(attend(
                &q.narrow(0, start, count)?,
                &k.narrow(0, start, count)?,
                &v.narrow(0, start, count)?,
                self.query_chunk,
            )?);
        }
        let out = Tensor::cat(&result, 0)?
            .reshape((rows, width))?
            .index_select(&plan.restore, 0)?;
        self.linear(&out, &format!("{prefix}.to_out"))
    }
    pub fn cross_attention(
        &self,
        x: &Tensor,
        condition: &Tensor,
        prefix: &str,
        heads: usize,
        qk_norm: bool,
    ) -> Result<Tensor> {
        let (rows, width) = x.dims2()?;
        let dim = width / heads;
        let length = condition.dim(1)?;
        let mut q = self
            .linear(x, &format!("{prefix}.to_q"))?
            .reshape((rows, heads, dim))?;
        let kv = self
            .linear(&condition.squeeze(0)?, &format!("{prefix}.to_kv"))?
            .reshape((length, 2, heads, dim))?;
        let mut k = kv.narrow(1, 0, 1)?.squeeze(1)?;
        let v = kv.narrow(1, 1, 1)?.squeeze(1)?;
        if qk_norm {
            q = self.qk_norm(&q, &format!("{prefix}.q_rms_norm"))?;
            k = self.qk_norm(&k, &format!("{prefix}.k_rms_norm"))?;
        }
        let out = attend(&q, &k, &v, self.query_chunk)?.reshape((rows, width))?;
        self.linear(&out, &format!("{prefix}.to_out"))
    }
    pub fn timestep(&self, time: &Tensor, channels: usize) -> Result<Tensor> {
        Ok(self.timestep_fp32(time, channels)?.to_dtype(self.dtype)?)
    }
    pub fn timestep_fp32(&self, time: &Tensor, channels: usize) -> Result<Tensor> {
        let frequencies = (0..128)
            .map(|i| (-10000f32.ln() * i as f32 / 128.).exp())
            .collect::<Vec<_>>();
        let angles = time
            .to_dtype(DType::F32)?
            .reshape((1, 1))?
            .broadcast_mul(&Tensor::from_vec(frequencies, (1, 128), &self.device)?)?;
        let embedded = Tensor::cat(&[angles.cos()?, angles.sin()?], 1)?;
        let out = self.linear(
            &candle_nn::ops::silu(&self.linear(&embedded, "t_embedder.mlp.0")?)?,
            "t_embedder.mlp.2",
        )?;
        anyhow::ensure!(
            out.dims() == [1, channels],
            "invalid timestep embedding shape"
        );
        Ok(out)
    }
}

pub(crate) fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let wide = x.to_dtype(DType::F32)?;
    let centered = wide.broadcast_sub(&wide.mean_keepdim(D::Minus1)?)?;
    Ok(centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?)
}

pub(crate) fn position(
    grid: &Grid,
    channels: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let freq_dim = channels / 6;
    anyhow::ensure!(
        freq_dim > 0,
        "position embedding needs at least six channels"
    );
    let freq = (0..freq_dim)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / freq_dim as f32))
        .collect::<Vec<_>>();
    let coords = grid
        .coords
        .iter()
        .flat_map(|c| c[1..].iter().map(|&x| x as f32))
        .collect::<Vec<_>>();
    let angles = Tensor::from_vec(coords, (grid.coords.len() * 3, 1), device)?
        .broadcast_mul(&Tensor::from_vec(freq, (1, freq_dim), device)?)?;
    let mut encoded = Tensor::cat(&[angles.sin()?, angles.cos()?], 1)?
        .reshape((grid.coords.len(), 6 * freq_dim))?;
    if 6 * freq_dim < channels {
        encoded = Tensor::cat(
            &[
                encoded,
                Tensor::zeros(
                    (grid.coords.len(), channels - 6 * freq_dim),
                    DType::F32,
                    device,
                )?,
            ],
            1,
        )?;
    }
    Ok(encoded.to_dtype(dtype)?)
}

pub(crate) struct AttentionPlan {
    order: Tensor,
    restore: Tensor,
    ranges: Vec<(usize, usize)>,
}

pub(crate) struct Rotary3d {
    cos: Tensor,
    sin: Tensor,
}
impl Rotary3d {
    pub fn new(grid: &Grid, head_dim: usize, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            head_dim >= 6 && head_dim.is_multiple_of(2),
            "invalid 3D rotary head width"
        );
        let per_axis = head_dim / 6;
        let mut angles = Vec::with_capacity(grid.coords.len() * head_dim / 2);
        for coord in &grid.coords {
            for &axis in &coord[1..] {
                for i in 0..per_axis {
                    angles.push(axis as f32 / 10000f32.powf(i as f32 / per_axis as f32));
                }
            }
            angles.extend(std::iter::repeat_n(0f32, head_dim / 2 - per_axis * 3));
        }
        let angles = Tensor::from_vec(angles, (grid.coords.len(), 1, head_dim / 2), device)?;
        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (rows, heads, dim) = x.dims3()?;
        let pairs = x.to_dtype(DType::F32)?.reshape((rows, heads, dim / 2, 2))?;
        let real = pairs.narrow(3, 0, 1)?.squeeze(3)?;
        let imag = pairs.narrow(3, 1, 1)?.squeeze(3)?;
        let a = (real.broadcast_mul(&self.cos)? - imag.broadcast_mul(&self.sin)?)?;
        let b = (imag.broadcast_mul(&self.cos)? + real.broadcast_mul(&self.sin)?)?;
        Ok(Tensor::stack(&[a, b], 3)?
            .reshape((rows, heads, dim))?
            .to_dtype(x.dtype())?)
    }
}
impl AttentionPlan {
    pub fn new(grid: &Grid, window: Option<(usize, usize)>, device: &Device) -> Result<Self> {
        let mut order = Vec::new();
        let mut restore = vec![0u32; grid.coords.len()];
        let mut ranges = Vec::new();
        for group in grid.partitions(window)? {
            ranges.push((order.len(), group.len()));
            order.extend(group);
        }
        for (i, &row) in order.iter().enumerate() {
            restore[row as usize] = i as u32;
        }
        let rows = order.len();
        Ok(Self {
            order: Tensor::from_vec(order, rows, device)?,
            restore: Tensor::from_vec(restore, rows, device)?,
            ranges,
        })
    }
}

pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, chunk: usize) -> Result<Tensor> {
    anyhow::ensure!(chunk > 0, "attention query chunk must be positive");
    let (rows, heads, dim) = q.dims3()?;
    let k = k.to_dtype(DType::F32)?.permute((1, 2, 0))?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
    let mut tiles = Vec::new();
    for start in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - start);
        let query = q
            .narrow(0, start, n)?
            .to_dtype(DType::F32)?
            .transpose(0, 1)?
            .contiguous()?;
        let scores = (query.matmul(&k)? / (dim as f64).sqrt())?;
        tiles.push(
            candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .to_dtype(q.dtype())?,
        );
    }
    Ok(Tensor::cat(&tiles, 0)?.reshape((rows, heads, dim))?)
}
