//! Dense 3D convolution, which Candle 0.11 does not provide.
//!
//! A `[Cout, Cin, kd, kh, kw]` convolution is the sum over the depth-kernel
//! offsets of `kd` two-dimensional convolutions: output plane `d` reads input
//! plane `d + i - padding` through kernel slice `i`. Expressing it that way
//! reuses Candle's own `conv2d`, which is cuDNN on CUDA, rather than writing a
//! kernel; the depth axis is folded into the batch so all output planes of one
//! offset are convolved at once.

use anyhow::Result;
use candle_core::Tensor;

/// Convolve `[batch, in, depth, height, width]` with
/// `[out, in, kd, kh, kw]`, zero-padded by `padding` on every spatial axis.
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    padding: usize,
) -> Result<Tensor> {
    let (batch, in_channels, depth, height, width) = input.dims5()?;
    let (out_channels, weight_in, kernel_depth, _, _) = weight.dims5()?;
    anyhow::ensure!(
        weight_in == in_channels,
        "convolution weight expects {weight_in} input channels, got {in_channels}"
    );
    let padded_depth = depth + 2 * padding;
    anyhow::ensure!(
        padded_depth >= kernel_depth,
        "input depth {depth} padded to {padded_depth} is shorter than the {kernel_depth}-deep kernel"
    );
    let out_depth = padded_depth - kernel_depth + 1;

    let padded = if padding > 0 {
        input.pad_with_zeros(2, padding, padding)?
    } else {
        input.clone()
    };

    let mut accumulated: Option<Tensor> = None;
    let mut out_height = 0;
    let mut out_width = 0;
    for offset in 0..kernel_depth {
        let planes = padded
            .narrow(2, offset, out_depth)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * out_depth, in_channels, height, width))?;
        let slice = weight.narrow(2, offset, 1)?.squeeze(2)?.contiguous()?;
        let convolved = planes.conv2d(&slice, padding, 1, 1, 1)?;
        out_height = convolved.dim(2)?;
        out_width = convolved.dim(3)?;
        accumulated = Some(match accumulated {
            None => convolved,
            Some(total) => (total + convolved)?,
        });
    }
    let output = accumulated
        .expect("a convolution kernel has at least one depth slice")
        .reshape((batch, out_depth, out_channels, out_height, out_width))?
        .transpose(1, 2)?
        .contiguous()?;
    match bias {
        None => Ok(output),
        Some(bias) => output
            .broadcast_add(&bias.reshape((1, out_channels, 1, 1, 1))?)
            .map_err(Into::into),
    }
}

/// `pixel_shuffle_3d`: fold `scale^3` channels out into the spatial axes.
///
/// Transcribed from `trellis/modules/spatial.py`, whose permutation interleaves
/// each spatial axis with its own scale factor.
pub fn pixel_shuffle_3d(x: &Tensor, scale: usize) -> Result<Tensor> {
    let (batch, channels, height, width, depth) = x.dims5()?;
    let volume = scale.pow(3);
    anyhow::ensure!(
        channels.is_multiple_of(volume),
        "pixel shuffle needs a multiple of {volume} channels, got {channels}"
    );
    let reduced = channels / volume;
    x.reshape(vec![
        batch, reduced, scale, scale, scale, height, width, depth,
    ])?
    .permute([0, 1, 5, 2, 6, 3, 7, 4])?
    .contiguous()?
    .reshape((batch, reduced, height * scale, width * scale, depth * scale))
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp as _};

    /// A one-channel identity kernel reproduces its input, which pins the
    /// padding and the depth alignment at once.
    #[test]
    fn an_identity_kernel_reproduces_its_input() {
        let device = Device::Cpu;
        let input = Tensor::arange(0f32, 2f32 * 3.0 * 4.0, &device)
            .unwrap()
            .reshape((1, 1, 2, 3, 4))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let mut kernel = vec![0f32; 27];
        kernel[13] = 1.0;
        let weight = Tensor::from_vec(kernel, (1, 1, 3, 3, 3), &device).unwrap();
        let output = conv3d(&input, &weight, None, 1).unwrap();
        assert_eq!(output.dims(), input.dims());
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            input.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    /// A kernel that is one everywhere sums the zero-padded neighbourhood, so
    /// every output is a window sum that can be computed independently.
    #[test]
    fn a_ones_kernel_sums_the_padded_neighbourhood() {
        let device = Device::Cpu;
        let (depth, height, width) = (3usize, 3usize, 3usize);
        let values: Vec<f32> = (0..depth * height * width)
            .map(|index| index as f32)
            .collect();
        let input =
            Tensor::from_vec(values.clone(), (1, 1, depth, height, width), &device).unwrap();
        let weight = Tensor::ones((1, 1, 3, 3, 3), DType::F32, &device).unwrap();
        let output = conv3d(&input, &weight, None, 1).unwrap();
        let produced = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let at = |d: i64, h: i64, w: i64| -> f32 {
            if d < 0 || h < 0 || w < 0 {
                return 0.0;
            }
            let (d, h, w) = (d as usize, h as usize, w as usize);
            if d >= depth || h >= height || w >= width {
                return 0.0;
            }
            values[(d * height + h) * width + w]
        };
        for d in 0..depth {
            for h in 0..height {
                for w in 0..width {
                    let mut expected = 0.0;
                    for dd in -1i64..=1 {
                        for dh in -1i64..=1 {
                            for dw in -1i64..=1 {
                                expected += at(d as i64 + dd, h as i64 + dh, w as i64 + dw);
                            }
                        }
                    }
                    let index = (d * height + h) * width + w;
                    assert!(
                        (produced[index] - expected).abs() < 1e-4,
                        "at ({d},{h},{w}): {} != {expected}",
                        produced[index]
                    );
                }
            }
        }
    }

    /// The bias is added once per output channel, not per depth plane.
    #[test]
    fn the_bias_is_added_once_per_output_channel() {
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 1, 2, 2, 2), DType::F32, &device).unwrap();
        let weight = Tensor::zeros((3, 1, 3, 3, 3), DType::F32, &device).unwrap();
        let bias = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let output = conv3d(&input, &weight, Some(&bias), 1).unwrap();
        assert_eq!(output.dims(), [1, 3, 2, 2, 2]);
        for (channel, expected) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            let plane = output.i((0, channel)).unwrap();
            for value in plane.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
                assert!((value - expected).abs() < 1e-6);
            }
        }
    }

    /// Pixel shuffle moves exactly `scale^3` channels into the spatial axes and
    /// keeps every value.
    #[test]
    fn pixel_shuffle_preserves_every_value() {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, 16f32 * 2.0 * 2.0 * 2.0, &device)
            .unwrap()
            .reshape((1, 16, 2, 2, 2))
            .unwrap();
        let shuffled = pixel_shuffle_3d(&x, 2).unwrap();
        assert_eq!(shuffled.dims(), [1, 2, 4, 4, 4]);
        let mut before = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut after = shuffled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        before.sort_by(f32::total_cmp);
        after.sort_by(f32::total_cmp);
        assert_eq!(before, after);
    }
}
