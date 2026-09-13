use crate::Component;
use anyhow::Result;
use candle_core::{D, DType, Tensor};

impl Component {
    pub(crate) fn get(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .weights
            .load(name, &self.device)?
            .to_dtype(self.dtype)?)
    }
    pub(crate) fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let weight = self.get(&format!("{prefix}.weight"))?.t()?;
        let result = if x.device().is_cuda() && x.rank() == 3 {
            let (batch, _, _) = x.dims3()?;
            let (input, output) = weight.dims2()?;
            x.matmul(&weight.contiguous()?.broadcast_as((batch, input, output))?)?
        } else {
            x.broadcast_matmul(&weight)?
        };
        if self.weights.contains(&format!("{prefix}.bias")) {
            Ok(result.broadcast_add(&self.get(&format!("{prefix}.bias"))?)?)
        } else {
            Ok(result)
        }
    }
    pub(crate) fn embedding(&self, prefix: &str, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .weights
            .load_rows(&format!("{prefix}.weight"), ids, &self.device)?
            .to_dtype(self.dtype)?)
    }
    pub(crate) fn rms(&self, x: &Tensor, prefix: &str, eps: f64) -> Result<Tensor> {
        let wide = x.to_dtype(DType::F32)?;
        Ok(wide
            .broadcast_div(&(wide.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.get(&format!("{prefix}.weight"))?)?)
    }
    pub(crate) fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let wide = x.to_dtype(DType::F32)?;
        let centered = wide.broadcast_sub(&wide.mean_keepdim(D::Minus1)?)?;
        Ok(centered
            .broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-5)?.sqrt()?)?
            .broadcast_mul(&self.get(&format!("{prefix}.weight"))?)?
            .broadcast_add(&self.get(&format!("{prefix}.bias"))?)?
            .to_dtype(x.dtype())?)
    }
    pub(crate) fn conv(
        &self,
        x: &Tensor,
        prefix: &str,
        padding: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let result = x
            .contiguous()?
            .conv1d(&self.conv_weight(prefix)?, padding, 1, dilation, 1)?;
        self.conv_bias(result, prefix)
    }
    pub(crate) fn conv_weight(&self, prefix: &str) -> Result<Tensor> {
        let name = format!("{prefix}.weight_v");
        if !self.weights.contains(&name) {
            return self.get(&format!("{prefix}.weight"));
        }
        let v = self.get(&name)?;
        let g = self.get(&format!("{prefix}.weight_g"))?;
        let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        Ok(v.broadcast_mul(&g.broadcast_div(&norm)?)?)
    }
    pub(crate) fn conv_bias(&self, x: Tensor, prefix: &str) -> Result<Tensor> {
        let name = format!("{prefix}.bias");
        if !self.weights.contains(&name) {
            return Ok(x);
        }
        Ok(x.broadcast_add(&self.get(&name)?.reshape((1, (), 1))?)?)
    }
}

pub(crate) fn heads(x: &Tensor, count: usize) -> Result<Tensor> {
    let (batch, length, width) = x.dims3()?;
    anyhow::ensure!(
        count > 0 && width.is_multiple_of(count),
        "invalid attention head geometry"
    );
    Ok(x.reshape((batch, length, count, width / count))?
        .transpose(1, 2)?
        .contiguous()?)
}

pub(crate) fn rope(x: &Tensor, dim: usize, theta: f64, offset: usize) -> Result<Tensor> {
    let (_, _, length, width) = x.dims4()?;
    anyhow::ensure!(
        dim > 0 && dim.is_multiple_of(2) && dim <= width && theta.is_finite() && theta > 0.,
        "invalid rotary geometry"
    );
    let angles = (offset..offset + length)
        .flat_map(|p| {
            (0..dim / 2).map(move |i| p as f32 / (theta as f32).powf((2 * i) as f32 / dim as f32))
        })
        .collect::<Vec<_>>();
    let angles = Tensor::from_vec(angles, (length, dim / 2), x.device())?;
    let rotated = candle_nn::rotary_emb::rope(
        &x.narrow(3, 0, dim)?.contiguous()?,
        &angles.cos()?.to_dtype(x.dtype())?,
        &angles.sin()?.to_dtype(x.dtype())?,
    )?;
    if dim == width {
        Ok(rotated)
    } else {
        Ok(Tensor::cat(&[rotated, x.narrow(3, dim, width - dim)?], 3)?)
    }
}

pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: Option<usize>,
    chunk: usize,
) -> Result<Tensor> {
    let (batch, count, length, dim) = q.dims4()?;
    let (kb, kv, total, kd) = k.dims4()?;
    anyhow::ensure!(
        batch == kb
            && dim == kd
            && v.dims() == k.dims()
            && kv > 0
            && count.is_multiple_of(kv)
            && chunk > 0,
        "invalid Music3 attention geometry"
    );
    let mut groups = Vec::new();
    for head in 0..kv {
        let key = k
            .narrow(1, head, 1)?
            .to_dtype(DType::F32)?
            .transpose(2, 3)?
            .contiguous()?;
        let value = v.narrow(1, head, 1)?.to_dtype(DType::F32)?.contiguous()?;
        let mut rows = Vec::new();
        for start in (0..length).step_by(chunk) {
            let n = chunk.min(length - start);
            let query = q
                .narrow(1, head * (count / kv), count / kv)?
                .narrow(2, start, n)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let mut scores = (query.broadcast_matmul(&key)? / (dim as f64).sqrt())?;
            if let Some(offset) = offset {
                let mask = (0..n)
                    .flat_map(|row| {
                        (0..total).map(move |col| {
                            if col <= offset + start + row {
                                0f32
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                scores =
                    scores.broadcast_add(&Tensor::from_vec(mask, (1, 1, n, total), q.device())?)?;
            }
            rows.push(
                candle_nn::ops::softmax_last_dim(&scores)?
                    .broadcast_matmul(&value)?
                    .to_dtype(q.dtype())?,
            );
        }
        groups.push(Tensor::cat(&rows, 2)?);
    }
    Ok(Tensor::cat(&groups, 1)?
        .transpose(1, 2)?
        .reshape((batch, length, count * dim))?
        .contiguous()?)
}
