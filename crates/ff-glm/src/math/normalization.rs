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
        let variance = cuda_mean(&input.sqr()?)?;
        let inverse = (&variance + eps)?.contiguous()?.apply_op1_no_bwd(&Rsqrt)?;
        Ok(input.broadcast_mul(&inverse)?)
    }
    #[cfg(not(feature = "cuda"))]
    anyhow::bail!("GLM CUDA normalization requires the cuda feature")
}

#[cfg(feature = "cuda")]
fn cuda_mean(input: &Tensor) -> Result<Tensor> {
    let width = input.dim(D::Minus1)?;
    ensure!(
        CUDA_WIDTHS.contains(&width),
        "unverified GLM CUDA normalization width {width}"
    );
    let rows = input.elem_count() / width;
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
        let kernel = device.get_or_load_custom_func(
            "glm_rsqrt_f32",
            "glm_reference_rsqrt_v1",
            include_str!(concat!(env!("OUT_DIR"), "/glm_rsqrt_f32.ptx")),
        )?;
        let mut launch = kernel.builder();
        launch.arg(&count).arg(&source).arg(&mut output);
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            layout.shape().clone(),
        ))
    }
}
