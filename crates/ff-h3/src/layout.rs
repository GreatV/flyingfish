use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

pub const VIDEO_TAG: u32 = 0;
pub const TEXT_TAG: u32 = 1;
pub const AUDIO_TAG: u32 = 2;
const ROPE_FRAME_RESCALE: f64 = 5. / 3.;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1., 4., 4., 4., 4.];
const ROPE_SPATIAL_SCALE: f64 = 32.;

pub struct PackedLayout {
    position_ids: Tensor,
    token_tags: Tensor,
    video_indices: Tensor,
    audio_indices: Tensor,
    text_indices: Tensor,
    sequence_length: usize,
    video_rows: usize,
    audio_rows: usize,
    text_rows: usize,
}

impl PackedLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn t2va(
        text_token_tags: &[u32],
        num_latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        num_audio_latents: usize,
        patch_size: [usize; 3],
        audio_channels: usize,
        device: &Device,
    ) -> Result<Self> {
        let [patch_t, patch_h, patch_w] = patch_size;
        anyhow::ensure!(
            patch_t == 1,
            "T2VA layout currently requires temporal patch size one"
        );
        anyhow::ensure!(
            patch_h > 0 && patch_w > 0,
            "patch dimensions must be non-zero"
        );
        anyhow::ensure!(
            num_latent_frames > 0 && latent_height > 0 && latent_width > 0,
            "video latent frame, height, and width dimensions must be non-zero"
        );
        anyhow::ensure!(
            latent_height.is_multiple_of(patch_h) && latent_width.is_multiple_of(patch_w),
            "latent canvas is not divisible by the transformer patch"
        );
        anyhow::ensure!(audio_channels > 0, "audio channel count must be non-zero");
        anyhow::ensure!(
            text_token_tags.iter().all(|&tag| tag < 3),
            "text stream contains an invalid modality tag"
        );
        let rows_per_frame = (latent_height / patch_h) * (latent_width / patch_w);
        let text_rows = text_token_tags.len();
        let audio_rows = num_audio_latents * audio_channels;
        let video_rows = num_latent_frames * rows_per_frame;
        let sequence_length = text_rows + audio_rows + video_rows;
        let audio_start = text_rows;
        let video_start = audio_start + audio_rows;

        let sqrt_area = ((latent_height * latent_width) as f64).sqrt();
        let height_grid = spatial_grid(latent_height, patch_h, sqrt_area);
        let width_grid = spatial_grid(latent_width, patch_w, sqrt_area);
        let first_width = *width_grid
            .first()
            .context("patch grid produced no horizontal positions")?;
        let last_width = *width_grid
            .last()
            .context("patch grid produced no horizontal positions")?;
        let mut frame_grid = Vec::with_capacity(rows_per_frame);
        for &height in &height_grid {
            for &width in &width_grid {
                frame_grid.push((height, width));
            }
        }

        let mut positions = vec![0f64; sequence_length * 3];
        for row in 0..text_rows {
            positions[row * 3] = row as f64;
        }
        for channel in 0..audio_channels {
            let width = if channel == 0 {
                first_width
            } else {
                last_width
            };
            for latent in 0..num_audio_latents {
                let row = audio_start + channel * num_audio_latents + latent;
                positions[row * 3] = text_rows as f64 + latent as f64;
                positions[row * 3 + 2] = width;
            }
        }
        let mut rotary_time = text_rows as f64;
        for frame in 0..num_latent_frames {
            for (spatial, &(height, width)) in frame_grid.iter().enumerate() {
                let row = video_start + frame * rows_per_frame + spatial;
                positions[row * 3] = rotary_time;
                positions[row * 3 + 1] = height;
                positions[row * 3 + 2] = width;
            }
            rotary_time +=
                ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[frame % ROPE_FRAMES_PER_LATENT.len()];
        }

        let mut tags = Vec::with_capacity(sequence_length);
        tags.extend_from_slice(text_token_tags);
        tags.extend(std::iter::repeat_n(AUDIO_TAG, audio_rows));
        tags.extend(std::iter::repeat_n(VIDEO_TAG, video_rows));
        let text_indices = range_u32(0, text_rows)?;
        let audio_indices = range_u32(audio_start, video_start)?;
        let video_indices = range_u32(video_start, sequence_length)?;
        Ok(Self {
            position_ids: Tensor::from_vec(positions, (sequence_length, 3), device)?,
            token_tags: Tensor::from_vec(tags, sequence_length, device)?,
            video_indices: Tensor::from_vec(video_indices, video_rows, device)?,
            audio_indices: Tensor::from_vec(audio_indices, audio_rows, device)?,
            text_indices: Tensor::from_vec(text_indices, text_rows, device)?,
            sequence_length,
            video_rows,
            audio_rows,
            text_rows,
        })
    }

    pub fn position_ids(&self) -> &Tensor {
        &self.position_ids
    }

    pub fn token_tags(&self) -> &Tensor {
        &self.token_tags
    }

    pub fn video_indices(&self) -> &Tensor {
        &self.video_indices
    }

    pub fn audio_indices(&self) -> &Tensor {
        &self.audio_indices
    }

    pub fn text_indices(&self) -> &Tensor {
        &self.text_indices
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn video_rows(&self) -> usize {
        self.video_rows
    }

    pub fn audio_rows(&self) -> usize {
        self.audio_rows
    }

    pub fn text_rows(&self) -> usize {
        self.text_rows
    }

    pub fn t2va_row_timesteps(
        &self,
        video_timestep: f32,
        audio_timestep: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let distinct = self.t2va_timestep_table(video_timestep, audio_timestep);
        let video_index = timestep_index(&distinct, video_timestep)?;
        let mut inverse = vec![video_index; self.sequence_length];
        if self.audio_rows > 0 {
            let audio_index = timestep_index(&distinct, audio_timestep)?;
            let audio_end = self
                .text_rows
                .checked_add(self.audio_rows)
                .context("audio row range overflow")?;
            inverse
                .get_mut(self.text_rows..audio_end)
                .context("audio row range exceeds packed sequence length")?
                .fill(audio_index);
        }
        Ok((
            Tensor::from_vec(distinct.clone(), distinct.len(), device)?,
            Tensor::from_vec(inverse, self.sequence_length, device)?,
        ))
    }

    pub(crate) fn t2va_timestep_table(&self, video_timestep: f32, audio_timestep: f32) -> Vec<f32> {
        distinct_timestep_table([
            (self.sequence_length - self.audio_rows, video_timestep),
            (self.audio_rows, audio_timestep),
        ])
    }
}

pub fn patchify_video(latents: &Tensor, patch_size: [usize; 3]) -> Result<Tensor> {
    anyhow::ensure!(
        latents.rank() == 5,
        "video latents must be [batch, channels, frames, height, width]"
    );
    let dims = latents.dims();
    let (batch, channels, frames, height, width) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let [patch_t, patch_h, patch_w] = patch_size;
    anyhow::ensure!(
        frames.is_multiple_of(patch_t)
            && height.is_multiple_of(patch_h)
            && width.is_multiple_of(patch_w),
        "video latent shape is not divisible by patch {patch_size:?}"
    );
    latents
        .reshape(&[
            batch,
            channels,
            frames / patch_t,
            patch_t,
            height / patch_h,
            patch_h,
            width / patch_w,
            patch_w,
        ])?
        .permute([0, 2, 4, 6, 1, 3, 5, 7])?
        .contiguous()?
        .reshape((
            batch * (frames / patch_t) * (height / patch_h) * (width / patch_w),
            channels * patch_t * patch_h * patch_w,
        ))
        .map_err(Into::into)
}

pub fn unpatchify_video(
    rows: &Tensor,
    batch: usize,
    channels: usize,
    frames: usize,
    height: usize,
    width: usize,
    patch_size: [usize; 3],
) -> Result<Tensor> {
    let [patch_t, patch_h, patch_w] = patch_size;
    let frame_patches = frames / patch_t;
    let height_patches = height / patch_h;
    let width_patches = width / patch_w;
    anyhow::ensure!(
        rows.dims()
            == [
                batch * frame_patches * height_patches * width_patches,
                channels * patch_t * patch_h * patch_w,
            ],
        "patch rows do not match requested video latent shape"
    );
    rows.reshape(&[
        batch,
        frame_patches,
        height_patches,
        width_patches,
        channels,
        patch_t,
        patch_h,
        patch_w,
    ])?
    .permute([0, 4, 1, 5, 2, 6, 3, 7])?
    .contiguous()?
    .reshape((batch, channels, frames, height, width))
    .map_err(Into::into)
}

pub fn pack_audio(latents: &Tensor) -> Result<Tensor> {
    let (channels, latent_channels, frames) = latents
        .dims3()
        .context("audio latents must be [channels, latent_channels, frames]")?;
    latents
        .permute((0, 2, 1))?
        .contiguous()?
        .reshape((channels * frames, latent_channels))
        .map_err(Into::into)
}

pub fn unpack_audio(rows: &Tensor, channels: usize, frames: usize) -> Result<Tensor> {
    let (row_count, latent_channels) = rows.dims2()?;
    anyhow::ensure!(
        row_count == channels * frames,
        "audio row count differs from requested shape"
    );
    rows.reshape((channels, frames, latent_channels))?
        .permute((0, 2, 1))?
        .contiguous()
        .map_err(Into::into)
}

fn spatial_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1. - ratio) / 2.;
    let count = dim / patch;
    (0..count)
        .map(|index| (left + ratio * index as f64 / count as f64) * ROPE_SPATIAL_SCALE)
        .collect()
}

fn range_u32(start: usize, end: usize) -> Result<Vec<u32>> {
    (start..end)
        .map(|value| u32::try_from(value).context("packed sequence exceeds U32 indexing"))
        .collect()
}

fn distinct_timestep_table<const N: usize>(assignments: [(usize, f32); N]) -> Vec<f32> {
    let mut distinct = assignments
        .into_iter()
        .filter_map(|(count, timestep)| (count > 0).then_some(timestep))
        .collect::<Vec<_>>();
    distinct.sort_by(f32::total_cmp);
    distinct.dedup();
    distinct
}

fn timestep_index(distinct: &[f32], timestep: f32) -> Result<u32> {
    distinct
        .iter()
        .position(|candidate| *candidate == timestep)
        .map(|index| index as u32)
        .context("failed to invert timestep table")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_round_trip_preserves_order() {
        let values = (0..32).map(|value| value as f32).collect::<Vec<_>>();
        let latents = Tensor::from_vec(values.clone(), (1, 2, 2, 2, 4), &Device::Cpu).unwrap();
        let rows = patchify_video(&latents, [1, 2, 2]).unwrap();
        assert_eq!(rows.dims(), &[4, 8]);
        let restored = unpatchify_video(&rows, 1, 2, 2, 2, 4, [1, 2, 2]).unwrap();
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            values
        );
    }

    #[test]
    fn t2va_layout_orders_text_audio_then_video() {
        let layout = PackedLayout::t2va(
            &[TEXT_TAG, TEXT_TAG],
            2,
            2,
            4,
            3,
            [1, 2, 2],
            2,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(layout.text_indices.to_vec1::<u32>().unwrap(), vec![0, 1]);
        assert_eq!(
            layout.audio_indices.to_vec1::<u32>().unwrap(),
            (2..8).collect::<Vec<_>>()
        );
        assert_eq!(
            layout.video_indices.to_vec1::<u32>().unwrap(),
            (8..12).collect::<Vec<_>>()
        );
        assert_eq!(
            layout.token_tags.to_vec1::<u32>().unwrap(),
            vec![1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]
        );
        let (timesteps, inverse) = layout.t2va_row_timesteps(0.25, 0.5, &Device::Cpu).unwrap();
        assert_eq!(timesteps.to_vec1::<f32>().unwrap(), vec![0.25, 0.5]);
        assert_eq!(
            inverse.to_vec1::<u32>().unwrap(),
            vec![0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        );
    }

    #[test]
    fn t2va_layout_rejects_zero_video_geometry() {
        for (frames, height, width) in [(0, 2, 4), (1, 0, 4), (1, 2, 0)] {
            let error = PackedLayout::t2va(
                &[TEXT_TAG],
                frames,
                height,
                width,
                3,
                [1, 2, 2],
                2,
                &Device::Cpu,
            )
            .err()
            .expect("zero video geometry must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("video latent frame, height, and width dimensions must be non-zero")
            );
        }
    }

    #[test]
    fn zero_audio_rows_omit_the_audio_timestep() {
        let layout = PackedLayout::t2va(
            &[TEXT_TAG, TEXT_TAG],
            1,
            2,
            4,
            0,
            [1, 2, 2],
            2,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(layout.audio_rows, 0);
        assert_eq!(layout.t2va_timestep_table(0.25, 0.5), vec![0.25]);
        let (timesteps, inverse) = layout.t2va_row_timesteps(0.25, 0.5, &Device::Cpu).unwrap();
        assert_eq!(timesteps.to_vec1::<f32>().unwrap(), vec![0.25]);
        assert_eq!(inverse.to_vec1::<u32>().unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn audio_row_round_trip_is_channel_major() {
        let latents =
            Tensor::from_vec((0..12).map(|v| v as f32).collect(), (2, 2, 3), &Device::Cpu).unwrap();
        let rows = pack_audio(&latents).unwrap();
        assert_eq!(rows.dims(), &[6, 2]);
        let restored = unpack_audio(&rows, 2, 3).unwrap();
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (0..12).map(|v| v as f32).collect::<Vec<_>>()
        );
    }
}
