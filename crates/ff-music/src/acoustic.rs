use crate::{
    Component,
    math::{attention, heads, rope},
};
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

pub(crate) fn condition(model: &Component, frames: &Tensor) -> Result<Tensor> {
    let (batch, length, width) = frames.dims3()?;
    let layers = model.n("num_condition_layers")?;
    let hidden = model.n("condition_hidden_dim")?;
    anyhow::ensure!(
        length > 0 && width == layers * hidden,
        "invalid Music3 frame hidden shape"
    );
    let weights = candle_nn::ops::softmax(&model.get("layer_weight_logits")?, 0)?
        .reshape((1, 1, layers, 1))?;
    let x = frames
        .to_device(&model.device)?
        .to_dtype(model.dtype)?
        .reshape((batch, length, layers, hidden))?
        .broadcast_mul(&weights)?
        .sum(2)?
        .broadcast_mul(&model.get("layer_scale")?)?
        .transpose(1, 2)?
        .contiguous()?;
    let x = model.conv(&x, "proj", 1, 1)?;
    let ratio = model.n("output_sampling_rate")? as f64 / model.n("input_sampling_rate")? as f64
        * model.n("input_hop_length")? as f64
        / model.n("output_hop_length")? as f64;
    let output_length = ((length as f64 * ratio) as usize).max(1);
    let indices = (0..output_length)
        .map(|i| (i * length / output_length) as u32)
        .collect::<Vec<_>>();
    Ok(
        x.index_select(&Tensor::from_vec(indices, output_length, &model.device)?, 2)?
            .transpose(1, 2)?
            .contiguous()?,
    )
}

pub(crate) fn velocity(
    model: &Component,
    latents: &Tensor,
    condition: &Tensor,
    time: f32,
    chunk: usize,
) -> Result<Tensor> {
    let (batch, channels, length) = latents.dims3()?;
    anyhow::ensure!(
        channels == model.n("in_channels")?
            && condition.dims() == [batch, length, model.n("condition_dim")?],
        "Music3 flow latent/condition shape mismatch"
    );
    let input = Tensor::cat(
        &[latents, &latents.zeros_like()?, &condition.transpose(1, 2)?],
        1,
    )?
    .contiguous()?;
    let input = (&input + model.conv(&input, "preprocess_conv", 0, 1)?)?
        .transpose(1, 2)?
        .contiguous()?;
    let times = Tensor::from_vec(vec![time; batch], (batch, 1), &model.device)?;
    let angles =
        (times * (2. * std::f64::consts::PI))?.matmul(&model.get("time_proj.weight")?.t()?)?;
    let fourier = Tensor::cat(&[angles.cos()?, angles.sin()?], 1)?;
    let time_emb = model
        .linear(
            &candle_nn::ops::silu(&model.linear(&fourier, "time_embed.linear_1")?)?,
            "time_embed.linear_2",
        )?
        .unsqueeze(1)?;
    let mut x = Tensor::cat(&[time_emb, model.linear(&input, "proj_in")?], 1)?;
    let h = model.n("num_attention_heads")?;
    let rotary = model.n("rotary_dim")?;
    for layer in 0..model.n("num_layers")? {
        let p = format!("transformer_blocks.{layer}");
        let norm = model.layer_norm(&x, &format!("{p}.norm1"))?;
        let q = rope(
            &heads(&model.linear(&norm, &format!("{p}.attn.to_q"))?, h)?,
            rotary,
            10000.,
            0,
        )?;
        let k = rope(
            &heads(&model.linear(&norm, &format!("{p}.attn.to_k"))?, h)?,
            rotary,
            10000.,
            0,
        )?;
        let v = heads(&model.linear(&norm, &format!("{p}.attn.to_v"))?, h)?;
        x = (&x
            + model.linear(
                &attention(&q, &k, &v, None, chunk)?,
                &format!("{p}.attn.to_out.0"),
            )?)?;
        let ff = model.linear(
            &model.layer_norm(&x, &format!("{p}.norm2"))?,
            &format!("{p}.ff_in"),
        )?;
        let width = model.n("ff_inner_dim")?;
        let gate = candle_nn::ops::silu(&ff.narrow(2, width, width)?)?;
        x = (&x + model.linear(&(&ff.narrow(2, 0, width)? * gate)?, &format!("{p}.ff_out"))?)?;
    }
    let x = model
        .linear(&x.narrow(1, 1, length)?, "proj_out")?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((&x + model.conv(&x, "postprocess_conv", 0, 1)?)?)
}

fn snake(model: &Component, x: &Tensor, prefix: &str) -> Result<Tensor> {
    let alpha = model.get(&format!("{prefix}.alpha"))?;
    Ok((x + x
        .broadcast_mul(&alpha)?
        .sin()?
        .sqr()?
        .broadcast_div(&(alpha + 1e-9)?)?)?)
}

pub(crate) fn vocode(model: &Component, latents: &Tensor) -> Result<Tensor> {
    let (batch, channels, length) = latents.dims3()?;
    anyhow::ensure!(
        length > 0 && channels == model.n("latent_channels")? && channels.is_multiple_of(2),
        "invalid vocoder latents"
    );
    let mut x = latents
        .to_dtype(DType::F32)?
        .reshape((batch * 2, channels / 2, length))?;
    x = model.conv(&model.conv(&x, "dec_in_proj", 0, 1)?, "conv_in", 3, 1)?;
    let strides = model.config["upsampling_ratios"]
        .as_array()
        .context("missing vocoder upsampling ratios")?;
    for (layer, stride) in strides.iter().enumerate() {
        let stride = usize::try_from(stride.as_u64().context("invalid vocoder stride")?)?;
        anyhow::ensure!(stride > 0, "vocoder stride must be positive");
        let p = format!("blocks.{layer}");
        let norm = snake(model, &x, &format!("{p}.snake1"))?;
        let padding = stride.div_ceil(2);
        let leading = (norm.dim(2)? - 1) * stride;
        let extra = (2 * padding).saturating_sub(leading).div_ceil(2 * stride);
        let norm = if extra > 0 {
            norm.pad_with_zeros(2, extra, extra)?
        } else {
            norm
        };
        let prefix = format!("{p}.conv_t1");
        x = model.conv_bias(
            norm.contiguous()?.conv_transpose1d(
                &model.conv_weight(&prefix)?,
                padding,
                0,
                stride,
                1,
                1,
            )?,
            &prefix,
        )?;
        if extra > 0 {
            x = x
                .narrow(2, extra * stride, x.dim(2)? - 2 * extra * stride)?
                .contiguous()?;
        }
        for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
            let p = format!("{p}.res_unit{}", unit + 1);
            let residual = model.conv(
                &snake(model, &x, &format!("{p}.snake1"))?,
                &format!("{p}.conv1"),
                3 * dilation,
                dilation,
            )?;
            let residual = model.conv(
                &snake(model, &residual, &format!("{p}.snake2"))?,
                &format!("{p}.conv2"),
                0,
                1,
            )?;
            x = (&x + residual)?;
        }
    }
    Ok(model
        .conv(&snake(model, &x, "snake_out")?, "conv_out", 3, 1)?
        .tanh()?
        .reshape((batch, 2, ()))?)
}
