use super::{Music3, Result, product, sum, weight_load_reserve};
use crate::Component;
use anyhow::{Context, ensure};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct AcousticMemoryEstimate {
    pub window_frames: usize,
    pub latent_positions: u64,
    pub condition_activation_bytes: u64,
    pub denoise_activation_bytes: u64,
    pub vocoder_activation_bytes: u64,
    pub pipeline_carry_bytes: u64,
    pub condition_device_bytes: u64,
    pub denoise_device_bytes: u64,
    pub vocoder_device_bytes: u64,
    /// Stage maximum including the corresponding transient weight load.
    pub device_peak_bytes: u64,
}

fn conv_shape(model: &Component, prefix: &str) -> Result<[u64; 3]> {
    let name = if model.weights.contains(&format!("{prefix}.weight_v")) {
        format!("{prefix}.weight_v")
    } else {
        format!("{prefix}.weight")
    };
    let metadata = model.weights.raw_tensor_metadata(&name)?;
    ensure!(
        metadata.shape.len() == 3 && metadata.shape.iter().all(|&d| d > 0),
        "invalid convolution geometry for {name}"
    );
    Ok([
        metadata.shape[0] as u64,
        metadata.shape[1] as u64,
        metadata.shape[2] as u64,
    ])
}

#[derive(Default)]
struct ConvWorkingSet {
    plane: u64,
    columns: u64,
}

impl ConvWorkingSet {
    fn plane(&mut self, channels: u64, length: u64) -> Result<()> {
        self.plane = self.plane.max(product(&[2, channels, length, 4])?);
        Ok(())
    }

    fn conv(
        &mut self,
        model: &Component,
        prefix: &str,
        length: u64,
        padding: u64,
        dilation: u64,
    ) -> Result<u64> {
        let [output, input, kernel] = conv_shape(model, prefix)?;
        let padded = sum(&[length, product(&[2, padding])?])?;
        let extent = sum(&[product(&[dilation, kernel - 1])?, 1])?;
        let output_length = padded
            .checked_sub(extent)
            .and_then(|n| n.checked_add(1))
            .context("invalid convolution output length")?;
        self.plane(input, length)?;
        self.plane(output, output_length)?;
        self.columns = self
            .columns
            .max(product(&[2, output_length, input, kernel, 4])?);
        Ok(output_length)
    }
}

fn vocoder(model: &Component, positions: u64) -> Result<(u64, u64)> {
    let mut work = ConvWorkingSet::default();
    let mut length = work.conv(model, "dec_in_proj", positions, 0, 1)?;
    length = work.conv(model, "conv_in", length, 3, 1)?;
    let ratios = model.config["upsampling_ratios"]
        .as_array()
        .context("missing vocoder upsampling ratios")?;
    for (layer, ratio) in ratios.iter().enumerate() {
        let stride = ratio
            .as_u64()
            .filter(|&v| v > 0)
            .context("invalid vocoder stride")?;
        let prefix = format!("blocks.{layer}");
        let [input, output, kernel] = conv_shape(model, &format!("{prefix}.conv_t1"))?;
        work.plane(input, length)?;
        let padding = stride.div_ceil(2);
        let leading = product(&[length - 1, stride])?;
        let twice_padding = product(&[2, padding])?;
        let extra = twice_padding
            .saturating_sub(leading)
            .div_ceil(product(&[2, stride])?);
        let padded = sum(&[length, product(&[2, extra])?])?;
        work.plane(input, padded)?;
        let uncropped = sum(&[product(&[padded - 1, stride])?, kernel])?
            .checked_sub(twice_padding)
            .context("invalid transposed convolution extent")?;
        work.plane(output, uncropped)?;
        length = uncropped
            .checked_sub(product(&[2, extra, stride])?)
            .context("invalid vocoder crop")?;
        for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
            length = work.conv(
                model,
                &format!("{prefix}.res_unit{}.conv1", unit + 1),
                length,
                3 * dilation,
                dilation,
            )?;
            length = work.conv(
                model,
                &format!("{prefix}.res_unit{}.conv2", unit + 1),
                length,
                0,
                1,
            )?;
        }
    }
    length = work.conv(model, "conv_out", length, 3, 1)?;
    Ok((sum(&[product(&[8, work.plane])?, work.columns])?, length))
}

pub(super) fn estimate(
    model: &Music3,
    frames: usize,
    chunk: usize,
) -> Result<AcousticMemoryEstimate> {
    let window_frames = frames.min(crate::pipeline::ACOUSTIC_WINDOW_FRAMES);
    let rows = window_frames as u64;
    let latent_positions = (product(&[rows, 441])? / 128).max(1);
    let hidden = model.condition.n("condition_hidden_dim")? as u64;
    let layers = model.condition.n("num_condition_layers")? as u64;
    let condition_width = model.condition.n("out_dim")? as u64;
    let mut condition_conv = ConvWorkingSet::default();
    let _ = condition_conv.conv(&model.condition, "proj", rows, 1, 1)?;
    let condition_activation_bytes = sum(&[
        product(&[3, rows, layers, hidden, 4])?,
        product(&[8, rows, hidden.max(condition_width), 4])?,
        product(&[3, latent_positions, condition_width, 4])?,
        condition_conv.columns,
    ])?;
    let heads = model.transformer.n("num_attention_heads")? as u64;
    let dim = model.transformer.n("attention_head_dim")? as u64;
    let width = product(&[heads, dim])?;
    let ff = model.transformer.n("ff_inner_dim")? as u64;
    let length = sum(&[latent_positions, 1])?;
    let denoise_activation_bytes = sum(&[
        product(&[12, 2, length, width, 4])?,
        product(&[4, 2, length, ff, 4])?,
        product(&[8, length, dim, 4])?,
        product(&[6, length.min(chunk as u64), length, 4])?,
    ])?;
    let (vocoder_activation_bytes, _) = vocoder(&model.vocoder, latent_positions)?;
    let channels = model.transformer.n("in_channels")? as u64;
    let pipeline_carry_bytes = sum(&[
        product(&[4, latent_positions, condition_width, 4])?,
        product(&[8, latent_positions, channels, 4])?,
    ])?;
    let stages = [
        sum(&[
            condition_activation_bytes,
            weight_load_reserve(&model.condition, false)?,
        ])?,
        sum(&[
            denoise_activation_bytes,
            weight_load_reserve(&model.transformer, true)?,
        ])?,
        sum(&[
            vocoder_activation_bytes,
            weight_load_reserve(&model.vocoder, false)?,
        ])?,
    ];
    let condition_device_bytes = sum(&[stages[0], pipeline_carry_bytes])?;
    let denoise_device_bytes = sum(&[stages[1], pipeline_carry_bytes])?;
    let vocoder_device_bytes = sum(&[stages[2], pipeline_carry_bytes])?;
    let device_peak_bytes = condition_device_bytes
        .max(denoise_device_bytes)
        .max(vocoder_device_bytes);
    Ok(AcousticMemoryEstimate {
        window_frames,
        latent_positions,
        condition_activation_bytes,
        denoise_activation_bytes,
        vocoder_activation_bytes,
        pipeline_carry_bytes,
        condition_device_bytes,
        denoise_device_bytes,
        vocoder_device_bytes,
        device_peak_bytes,
    })
}
