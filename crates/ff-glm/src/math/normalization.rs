use anyhow::{Result, ensure};
use candle_core::{D, DType, Tensor};

pub(crate) const CUDA_WIDTHS: &[usize] = &[2, 4, 8, 128, 512, 1536, 4096, 16384];

pub(super) fn normalized(input: &Tensor, eps: f64) -> Result<Tensor> {
    let input = input.to_dtype(DType::F32)?;
    if input.device().is_cpu() {
        let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
        return Ok(input.broadcast_div(&(&variance + eps)?.sqrt()?)?);
    }
    ensure!(
        input.device().is_cuda(),
        "GLM normalization requires CPU or CUDA"
    );
    #[cfg(feature = "cuda")]
    {
        let width = input.dim(D::Minus1)?;
        let rows = input.elem_count() / width;
        // Use the fused kernel for contiguous rows needing no padding; otherwise fall back.
        if input.layout().is_contiguous()
            && let Ok(geometry) = mean_geometry(width, rows)
            && geometry.is_exact(width)
        {
            return fused_normalized(&input, geometry, eps);
        }
        let variance = cuda_mean(&input.sqr()?)?;
        let inverse = (&variance + eps)?.contiguous()?.apply_op1_no_bwd(&Rsqrt)?;
        Ok(input.broadcast_mul(&inverse)?)
    }
    #[cfg(not(feature = "cuda"))]
    anyhow::bail!("GLM CUDA normalization requires the cuda feature")
}

/// Shared reduction geometry for the reference and fused kernels.
/// Each step processes `vector` elements per lane across `block_x * groups` lanes.
/// Both kernels must use this geometry to preserve bitwise parity.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct MeanGeometry {
    vector: usize,
    block_x: usize,
    groups: usize,
    iterations: usize,
}

#[cfg(feature = "cuda")]
impl MeanGeometry {
    fn lanes(&self) -> usize {
        self.block_x * self.groups
    }

    /// The fused kernel requires an exact tile; the reference chain handles padding.
    fn is_exact(&self, width: usize) -> bool {
        self.iterations * self.lanes() * self.vector == width
    }
}

#[cfg(feature = "cuda")]
fn mean_geometry(width: usize, rows: usize) -> Result<MeanGeometry> {
    ensure!(
        CUDA_WIDTHS.contains(&width),
        "unverified GLM CUDA normalization width {width}"
    );
    ensure!(
        rows > 0,
        "GLM CUDA normalization does not support empty rows"
    );
    let vector = if width >= 128 { 4 } else { 1 };
    let floor_power = |n: usize| 1usize << (usize::BITS - 1 - n.leading_zeros());
    let dim0 = floor_power((width / vector).min(512));
    let dim1 = floor_power(rows.min(512));
    let mut block_x = dim0.min(32);
    let block_y = dim1.min(512 / block_x);
    block_x = dim0.min(512 / block_y);
    let split_y = width.div_ceil(block_x) >= (block_y * 16).min(256);
    let groups = if split_y { block_y } else { 1 };
    let lanes = block_x * groups;
    ensure!(
        !split_y || width.div_ceil(lanes) < 256,
        "GLM normalization requires an unimplemented cross-CTA reduction"
    );
    let iterations = (width / vector).div_ceil(lanes);
    Ok(MeanGeometry {
        vector,
        block_x,
        groups,
        iterations,
    })
}

#[cfg(feature = "cuda")]
fn fused_normalized(input: &Tensor, geometry: MeanGeometry, eps: f64) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    let rows = input.elem_count() / width;
    let normalized = input
        .reshape((rows, width))?
        .apply_op1_no_bwd(&NormalizedF32 {
            geometry,
            // Match candle affine's F64-to-F32 scalar cast.
            mean_scale: (1.0 / width as f64) as f32,
            eps: eps as f32,
        })?;
    Ok(normalized.reshape(input.shape())?)
}

#[cfg(feature = "cuda")]
fn cuda_mean(input: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    let rows = input.elem_count() / width;
    let MeanGeometry {
        vector,
        block_x,
        groups,
        iterations,
    } = mean_geometry(width, rows)?;
    let lanes = block_x * groups;
    let values = input
        .contiguous()?
        .reshape((rows, width))?
        .pad_with_zeros(1, 0, iterations * lanes * vector - width)?
        .reshape((rows, iterations, lanes, vector))?;
    let mut accumulators = (0..vector)
        .map(|_| Tensor::zeros((rows, lanes, 1), DType::F32, input.device()))
        .collect::<candle_core::Result<Vec<_>>>()?;
    for iteration in 0..iterations {
        let part = values.narrow(1, iteration, 1)?.squeeze(1)?;
        for (slot, accumulator) in accumulators.iter_mut().enumerate() {
            *accumulator = accumulator.add(&part.narrow(2, slot, 1)?)?;
        }
    }
    let mut sum = accumulators[0].clone();
    for accumulator in &accumulators[1..] {
        sum = sum.add(accumulator)?;
    }
    let mut sum = sum.reshape((rows, groups, block_x))?;
    let mut offset = block_x / 2;
    while offset > 0 {
        sum = sum
            .narrow(2, 0, offset)?
            .add(&sum.narrow(2, offset, offset)?)?;
        offset /= 2;
    }
    let mut offset = groups / 2;
    while offset > 0 {
        sum = sum
            .narrow(1, 0, offset)?
            .add(&sum.narrow(1, offset, offset)?)?;
        offset /= 2;
    }
    let mut shape = input.dims().to_vec();
    *shape.last_mut().expect("width checked") = 1;
    Ok(sum.affine(1.0 / width as f64, 0.0)?.reshape(shape)?)
}

#[cfg(feature = "cuda")]
struct NormalizedF32 {
    geometry: MeanGeometry,
    mean_scale: f32,
    eps: f32,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for NormalizedF32 {
    fn name(&self) -> &'static str {
        "glm_reference_normalized_f32"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("GLM reference normalization op is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        if input.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("GLM reference normalization requires contiguous F32 input")
        }
        let (rows, width) = layout.shape().dims2().map_err(|_| {
            candle_core::Error::Msg("GLM normalization expects [rows, width]".into())
        })?;
        let geometry = self.geometry;
        if geometry.lanes() > 512 || !geometry.is_exact(width) || rows == 0 {
            candle_core::bail!("GLM normalization kernel got an unsupported geometry")
        }
        let device = input.device().clone();
        let source = input.as_cuda_slice::<f32>()?;
        let source = source.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(rows * width) }?;
        let function = ff_core::cuda_kernel_assets::load_function(
            &device,
            &crate::kernel_assets::GLM_NORMALIZED_F32,
            "glm_reference_normalized_v1",
            "glm_normalized_f32",
        )?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&function);
        let width_i = width as i32;
        let vector_i = geometry.vector as i32;
        let block_x_i = geometry.block_x as i32;
        let groups_i = geometry.groups as i32;
        let iterations_i = geometry.iterations as i32;
        launch
            .arg(&source)
            .arg(&mut output)
            .arg(&width_i)
            .arg(&vector_i)
            .arg(&block_x_i)
            .arg(&groups_i)
            .arg(&iterations_i)
            .arg(&self.mean_scale)
            .arg(&self.eps);
        let config = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

#[cfg(feature = "cuda")]
struct Rsqrt;

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for Rsqrt {
    fn name(&self) -> &'static str {
        "glm_reference_rsqrt_f32"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("GLM reference rsqrt op is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        if input.dtype() != DType::F32
            || !layout.is_contiguous()
            || layout.shape().elem_count() == 0
        {
            candle_core::bail!("GLM reference rsqrt requires nonempty contiguous F32 input")
        }
        let count = i32::try_from(layout.shape().elem_count())
            .map_err(|_| candle_core::Error::Msg("GLM rsqrt count exceeds i32".into()))?;
        let device = input.device().clone();
        let source = input.as_cuda_slice::<f32>()?;
        let source = source.slice(layout.start_offset()..);
        let mut output = unsafe { device.alloc::<f32>(count as usize) }?;
        let function = ff_core::cuda_kernel_assets::load_function(
            &device,
            &crate::kernel_assets::GLM_RSQRT_F32,
            "glm_reference_rsqrt_v1",
            "glm_rsqrt_f32",
        )?;
        let stream = device.cuda_stream();
        let mut launch = stream.launch_builder(&function);
        launch.arg(&count).arg(&source).arg(&mut output);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::Device;

    /// Reference fallback used as the test oracle.
    fn reference_normalized(input: &Tensor, eps: f64) -> Result<Tensor> {
        let input = input.to_dtype(DType::F32)?;
        let variance = cuda_mean(&input.sqr()?)?;
        let inverse = (&variance + eps)?.contiguous()?.apply_op1_no_bwd(&Rsqrt)?;
        Ok(input.broadcast_mul(&inverse)?)
    }

    /// Vary signs and magnitudes to expose reduction-order changes.
    fn anisotropic(rows: usize, width: usize, device: &Device) -> Tensor {
        let values = (0..rows * width)
            .map(|i| {
                let sign = if i % 3 == 0 { -1f32 } else { 1f32 };
                let magnitude = 10f32.powi((i % 7) as i32 - 3);
                sign * magnitude * (1.0 + 0.001 * (i % 13) as f32)
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, width), device).unwrap()
    }

    // Requires an sm_80+ GPU; returns early when CUDA is unavailable.
    #[test]
    fn fused_normalized_matches_reference_bit_exactly_on_cuda() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        // Cover decode, prefill, grouped reduction, and padded fallback (1, 1536).
        for (rows, width) in [
            (1usize, 4096usize),
            (1, 16384),
            (20, 4096),
            (20, 16384),
            (1, 128),
            (5, 1536),
            (1, 1536),
            (3, 8),
        ] {
            let input = anisotropic(rows, width, &device);
            let fused = normalized(&input, 1e-5)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let reference = reference_normalized(&input, 1e-5)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let fused = fused.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(fused.len(), reference.len());
            for (index, (fused, reference)) in fused.iter().zip(&reference).enumerate() {
                assert_eq!(
                    fused.to_bits(),
                    reference.to_bits(),
                    "[{rows}x{width}] element {index}: fused {fused:e} != reference {reference:e}"
                );
            }
        }
    }

    #[test]
    fn mean_geometry_tiles_exactly_for_deployed_shapes() {
        // Production geometries should use the fused path without padding.
        for (rows, width) in [(1, 4096), (1, 16384), (20, 4096), (20, 16384)] {
            assert!(mean_geometry(width, rows).unwrap().is_exact(width));
        }
    }
}
