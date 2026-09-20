//! Fused Sinkhorn normalization on `[tokens, streams, streams]` F32 matrices.
//!
//! Preserves each backend's reduction order: serial on CPU, halving tree on CUDA.
//! Each round normalizes dim 2 then dim 1, using F32 epsilon and IEEE division.
//! Unsupported stream counts and backends use the reference sequence.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

/// Maximum stream count supported by the kernel's register matrix.
const MAX_STREAMS: usize = 8;

/// Fallback and test reference using candle operations.
fn reference_loop(comb: &Tensor, rounds: usize, eps: f64) -> Result<Tensor> {
    let mut comb = comb.clone();
    for _ in 0..rounds {
        comb = comb.broadcast_div(&(&comb.sum_keepdim(2)? + eps)?)?;
        comb = comb.broadcast_div(&(&comb.sum_keepdim(1)? + eps)?)?;
    }
    Ok(comb)
}

/// Run `rounds` of Sinkhorn normalization, fused where supported.
pub fn fused_loop(comb: &Tensor, rounds: usize, eps: f64) -> Result<Tensor> {
    if rounds == 0 {
        return Ok(comb.clone());
    }
    ensure!(
        eps.is_finite() && eps >= 0.0,
        "mHC Sinkhorn epsilon must be finite and non-negative"
    );
    let (_, streams, trailing) = comb
        .dims3()
        .context("mHC Sinkhorn input must have shape [tokens, streams, streams]")?;
    ensure!(
        streams == trailing,
        "mHC Sinkhorn matrix must be square, got {streams}x{trailing}"
    );
    ensure!(
        comb.dtype() == DType::F32,
        "mHC Sinkhorn input must be F32, got {:?}",
        comb.dtype()
    );
    // The fused reduction requires power-of-two stream counts.
    if !(comb.device().is_cpu()
        || comb.device().is_cuda() && streams.is_power_of_two() && streams <= MAX_STREAMS)
    {
        return reference_loop(comb, rounds, eps);
    }
    if comb.device().is_cpu() && streams > MAX_STREAMS {
        return reference_loop(comb, rounds, eps);
    }
    let rounds = i32::try_from(rounds).context("mHC Sinkhorn rounds exceed i32")?;
    Ok(comb.contiguous()?.apply_op1_no_bwd(&SinkhornLoop {
        rounds,
        eps: eps as f32,
    })?)
}

struct SinkhornLoop {
    rounds: i32,
    eps: f32,
}

impl SinkhornLoop {
    /// Normalize dim 2 then dim 1, matching the CPU reference order.
    fn run(&self, data: &mut [f32], tokens: usize, streams: usize) {
        let n = streams * streams;
        for token in 0..tokens {
            let m = &mut data[token * n..(token + 1) * n];
            for _ in 0..self.rounds {
                for i in 0..streams {
                    let mut sum = 0.0f32;
                    for j in 0..streams {
                        sum += m[i * streams + j];
                    }
                    let den = sum + self.eps;
                    for j in 0..streams {
                        m[i * streams + j] /= den;
                    }
                }
                for j in 0..streams {
                    let mut sum = 0.0f32;
                    for i in 0..streams {
                        sum += m[i * streams + j];
                    }
                    let den = sum + self.eps;
                    for i in 0..streams {
                        m[i * streams + j] /= den;
                    }
                }
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl candle_core::CustomOp1 for SinkhornLoop {
    fn name(&self) -> &'static str {
        "glm_mhc_sinkhorn_loop_f32"
    }

    fn cpu_fwd(
        &self,
        input: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let dims = layout.shape().dims();
        let candle_core::CpuStorage::F32(data) = input else {
            candle_core::bail!("GLM mHC Sinkhorn loop requires F32 input")
        };
        if dims.len() != 3 || dims[1] != dims[2] || dims[1] > MAX_STREAMS || !layout.is_contiguous()
        {
            candle_core::bail!(
                "GLM mHC Sinkhorn loop requires contiguous [tokens, streams<=8, streams], got {:?}",
                layout.shape()
            );
        }
        let (tokens, streams) = (dims[0], dims[1]);
        let start = layout.start_offset();
        let mut out = data[start..start + tokens * streams * streams].to_vec();
        self.run(&mut out, tokens, streams);
        Ok((candle_core::CpuStorage::F32(out), layout.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for SinkhornLoop {
    fn name(&self) -> &'static str {
        "glm_mhc_sinkhorn_loop_f32"
    }

    fn cpu_fwd(
        &self,
        input: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let dims = layout.shape().dims();
        let candle_core::CpuStorage::F32(data) = input else {
            candle_core::bail!("GLM mHC Sinkhorn loop requires F32 input")
        };
        if dims.len() != 3 || dims[1] != dims[2] || dims[1] > MAX_STREAMS || !layout.is_contiguous()
        {
            candle_core::bail!(
                "GLM mHC Sinkhorn loop requires contiguous [tokens, streams<=8, streams], got {:?}",
                layout.shape()
            );
        }
        let (tokens, streams) = (dims[0], dims[1]);
        let start = layout.start_offset();
        let mut out = data[start..start + tokens * streams * streams].to_vec();
        self.run(&mut out, tokens, streams);
        Ok((candle_core::CpuStorage::F32(out), layout.shape().clone()))
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        let dims = layout.shape().dims();
        if input.dtype() != DType::F32
            || dims.len() != 3
            || dims[1] != dims[2]
            || dims[1] > MAX_STREAMS
            || !dims[1].is_power_of_two()
            || dims[0] == 0
            || !layout.is_contiguous()
        {
            candle_core::bail!(
                "GLM mHC Sinkhorn loop requires nonempty contiguous F32 [tokens, pow2 streams<=8, streams], got {:?}",
                layout.shape()
            );
        }
        let tokens = i32::try_from(dims[0])
            .map_err(|_| candle_core::Error::Msg("GLM mHC Sinkhorn tokens exceed i32".into()))?;
        let streams = i32::try_from(dims[1])
            .map_err(|_| candle_core::Error::Msg("GLM mHC Sinkhorn streams exceed i32".into()))?;
        let count = (tokens as usize) * (streams as usize) * (streams as usize);
        let device = input.device().clone();
        let source = input.as_cuda_slice::<f32>()?;
        let source = source.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(count) }?;
        let function = ff_core::cuda_kernel_assets::load_function(
            &device,
            &crate::kernel_assets::GLM_MHC_SINKHORN_LOOP_F32,
            "glm_mhc_sinkhorn_loop_f32",
            "glm_mhc_sinkhorn_loop_f32_v1",
        )?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&source)
            .arg(&mut output)
            .arg(&tokens)
            .arg(&streams)
            .arg(&self.rounds)
            .arg(&self.eps);
        let threads = (tokens as u32).min(32);
        let config = LaunchConfig {
            grid_dim: ((tokens as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Asymmetric magnitudes expose swapped axes or a changed reduction order.
    fn anisotropic(tokens: usize, streams: usize, device: &Device) -> Tensor {
        let mut values = Vec::with_capacity(tokens * streams * streams);
        for t in 0..tokens {
            for i in 0..streams {
                for j in 0..streams {
                    values.push(
                        ((t * 7 + i * 3 + j * 11) % 13) as f32 * 10f32.powi(i as i32 - 1)
                            + 0.001 * (j as f32 + 1.0)
                            + 0.01,
                    );
                }
            }
        }
        Tensor::from_vec(values, (tokens, streams, streams), device).unwrap()
    }

    fn assert_bit_identical(fused: &Tensor, reference: &Tensor) {
        let fused = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(fused.len(), reference.len());
        for (index, (fused, reference)) in fused.iter().zip(&reference).enumerate() {
            assert_eq!(
                fused.to_bits(),
                reference.to_bits(),
                "element {index}: fused {fused:e} != reference {reference:e}"
            );
        }
    }

    #[test]
    fn fused_loop_matches_reference_bit_exactly_on_cpu() {
        let device = Device::Cpu;
        // Match the 20-iteration schedule and exercise multiple tokens.
        let input = anisotropic(3, 4, &device);
        let fused = fused_loop(&input, 19, 1e-6).unwrap();
        let reference = reference_loop(&input, 19, 1e-6).unwrap();
        assert_bit_identical(&fused, &reference);
    }

    #[test]
    fn fused_loop_with_zero_rounds_returns_input() {
        let device = Device::Cpu;
        let input = anisotropic(2, 4, &device);
        let output = fused_loop(&input, 0, 1e-6).unwrap();
        assert_bit_identical(&output, &input);
    }

    // Requires an sm_80+ GPU; returns early when CUDA is unavailable.
    #[cfg(feature = "cuda")]
    #[test]
    fn fused_loop_matches_reference_bit_exactly_on_cuda() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let input = anisotropic(3, 4, &device);
        let fused = fused_loop(&input, 19, 1e-6)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let reference = reference_loop(&input, 19, 1e-6)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        assert_bit_identical(&fused, &reference);
    }
}
