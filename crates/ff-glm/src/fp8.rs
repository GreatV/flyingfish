use anyhow::{Context, Result, bail, ensure};
use candle_core::{CpuStorage, CustomOp2, CustomOp3, DType, Device, Layout, Shape, Tensor};
use ff_core::weights::ModelWeights;
use float8::F8E4M3;
use half::bf16;
use std::num::NonZeroUsize;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
mod staging;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Fp8TransferStats {
    pub uploads: u64,
    pub uploaded_bytes: u64,
    pub slot_allocations: u64,
    pub staging_bytes_per_tier: u64,
    pub trace: Vec<Fp8TransferInterval>,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fp8TransferInterval {
    pub kind: String,
    pub start_ms: f32,
    pub end_ms: f32,
}

/// Block edge used by the official GLM-5.3-Flash fine-grained FP8 checkpoint.
pub const FP8_BLOCK_SIZE: usize = 128;

/// Dequantize a row-major `[out_features, in_features]` E4M3 matrix.
///
/// The checkpoint stores one inverse scale for each 128-by-128 block.  The
/// operation performed here is
///
/// `output[row, column] = weight[row, column] * scale_inv[row / 128, column / 128]`.
///
/// Multiplication is deliberately performed in F32 before the optional BF16
/// cast.  This matches the ordinary reference dequantization path and avoids
/// rounding an F32 checkpoint scale before applying it.
pub fn dequantize_block_fp8(
    weight: &Tensor,
    scale_inv: &Tensor,
    output_dtype: DType,
) -> Result<Tensor> {
    validate_output_dtype(output_dtype)?;
    ensure!(
        weight.dtype() == DType::F8E4M3,
        "block-FP8 weight must have dtype F8E4M3, found {:?}",
        weight.dtype()
    );
    ensure!(
        scale_inv.dtype() == DType::F32,
        "block-FP8 inverse scale must have dtype F32, found {:?}",
        scale_inv.dtype()
    );
    ensure!(
        weight.device().same_device(scale_inv.device()),
        "block-FP8 weight and inverse scale must be on the same device"
    );

    let (out_features, in_features) = weight
        .dims2()
        .context("block-FP8 weight must be a rank-2 [out_features, in_features] matrix")?;
    ensure!(
        out_features > 0 && in_features > 0,
        "block-FP8 weight dimensions must be non-zero, found [{out_features}, {in_features}]"
    );
    let expected_scale_shape = [
        out_features.div_ceil(FP8_BLOCK_SIZE),
        in_features.div_ceil(FP8_BLOCK_SIZE),
    ];
    ensure!(
        scale_inv.dims() == expected_scale_shape,
        "block-FP8 inverse scale shape must be [{}, {}] for weight shape [{out_features}, {in_features}], found {:?}",
        expected_scale_shape[0],
        expected_scale_shape[1],
        scale_inv.dims()
    );

    #[cfg(feature = "cuda")]
    if weight.device().is_cuda() {
        return cuda::dequantize(weight, scale_inv, output_dtype);
    }

    if !weight.device().is_cpu() {
        let cpu_weight = weight
            .to_device(&Device::Cpu)
            .context("failed to stage E4M3 weight on CPU for dequantization")?;
        let cpu_scale = scale_inv
            .to_device(&Device::Cpu)
            .context("failed to stage block-FP8 inverse scale on CPU")?;
        return dequantize_block_fp8(&cpu_weight, &cpu_scale, output_dtype)?
            .to_device(weight.device())
            .context("failed to move dequantized block-FP8 weight to its original device");
    }

    if weight.layout().is_contiguous() && scale_inv.layout().is_contiguous() {
        return weight
            .apply_op2(scale_inv, FusedCpuDequantize { output_dtype })
            .context("failed to dequantize block-FP8 weight on the CPU");
    }

    let weight_f32 = weight
        .to_dtype(DType::F32)
        .context("failed to cast E4M3 weight to F32")?;
    let dequantized = if out_features.is_multiple_of(FP8_BLOCK_SIZE)
        && in_features.is_multiple_of(FP8_BLOCK_SIZE)
    {
        let out_blocks = out_features / FP8_BLOCK_SIZE;
        let in_blocks = in_features / FP8_BLOCK_SIZE;
        let blocked_weight = weight_f32
            .reshape((out_blocks, FP8_BLOCK_SIZE, in_blocks, FP8_BLOCK_SIZE))
            .context("failed to view aligned block-FP8 weight as 128x128 blocks")?;
        let blocked_scale = scale_inv
            .reshape((out_blocks, 1, in_blocks, 1))
            .context("failed to view aligned block-FP8 inverse scales for broadcasting")?;
        blocked_weight
            .broadcast_mul(&blocked_scale)
            .context("failed to apply aligned block-FP8 inverse scales")?
            .reshape((out_features, in_features))
            .context("failed to restore dequantized block-FP8 matrix shape")?
    } else {
        let row_blocks = block_indices(out_features, weight.device())?;
        let column_blocks = block_indices(in_features, weight.device())?;
        let expanded_scale = scale_inv
            .index_select(&row_blocks, 0)
            .context("failed to expand block-FP8 row scales")?
            .index_select(&column_blocks, 1)
            .context("failed to expand block-FP8 column scales")?;
        weight_f32
            .broadcast_mul(&expanded_scale)
            .context("failed to apply partial block-FP8 inverse scales")?
    };

    if output_dtype == DType::F32 {
        Ok(dequantized)
    } else {
        dequantized
            .to_dtype(output_dtype)
            .context("failed to cast dequantized block-FP8 weight to BF16")
    }
}

/// Fused, threaded CPU dequantization.
///
/// The generic path costs four passes over the matrix: an E4M3-to-F32 widen,
/// a scale broadcast, and an optional BF16 narrow, each materializing a full
/// intermediate. At GLM's expert geometry that is 100 MiB of F32 traffic to
/// produce 50 MiB of BF16, single-threaded, and it measured 79.8 ms per expert
/// against 4.3 ms of actual arithmetic.
///
/// E4M3 has 256 distinct values and one inverse scale covers a 128-by-128
/// block, so every product in a block comes from a 256-entry table. Building
/// one table per block costs 256 operations and serves 16,384 elements. The
/// inner loop is then a byte load, a table index and a store — no arithmetic,
/// no intermediate — and the whole matrix is read and written exactly once.
///
/// The table holds `convert(f32(value) * scale)`, which is the same F32
/// multiply and the same single rounding the generic path performs, so the
/// output is bit-for-bit what it produced.
struct FusedCpuDequantize {
    output_dtype: DType,
}

impl FusedCpuDequantize {
    fn fill<T: Copy + Default + Send + Sync>(
        rows: usize,
        cols: usize,
        weight: &[F8E4M3],
        scale: &[f32],
        output: &mut [T],
        convert: impl Fn(f32) -> T + Copy + Send,
    ) {
        let column_blocks = cols.div_ceil(FP8_BLOCK_SIZE);
        let row_blocks = rows.div_ceil(FP8_BLOCK_SIZE);
        let workers = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
            .min(row_blocks)
            .max(1);
        let blocks_per_worker = row_blocks.div_ceil(workers);
        let rows_per_worker = blocks_per_worker * FP8_BLOCK_SIZE;

        let mut output_rest = output;
        let mut weight_rest = weight;
        let mut first_row_block = 0usize;
        std::thread::scope(|scope| {
            while !output_rest.is_empty() {
                let take = (rows_per_worker * cols).min(output_rest.len());
                let (output_share, output_tail) = output_rest.split_at_mut(take);
                let (weight_share, weight_tail) = weight_rest.split_at(take);
                output_rest = output_tail;
                weight_rest = weight_tail;
                let row_block = first_row_block;
                first_row_block += blocks_per_worker;
                scope.spawn(move || {
                    Self::fill_share(
                        row_block,
                        cols,
                        column_blocks,
                        weight_share,
                        scale,
                        output_share,
                        convert,
                    );
                });
            }
        });
    }

    /// One worker's contiguous run of whole row blocks.
    fn fill_share<T: Copy + Default>(
        first_row_block: usize,
        cols: usize,
        column_blocks: usize,
        weight: &[F8E4M3],
        scale: &[f32],
        output: &mut [T],
        convert: impl Fn(f32) -> T,
    ) {
        // Every table for one row block at once: 32 tables of 256 BF16 entries
        // is 16 KiB at GLM's width, so the lookups stay in L1 while the matrix
        // streams past them.
        let mut tables = vec![[T::default(); 256]; column_blocks];
        let block_rows = FP8_BLOCK_SIZE * cols;
        for (index, (output_block, weight_block)) in output
            .chunks_mut(block_rows)
            .zip(weight.chunks(block_rows))
            .enumerate()
        {
            let row_block = first_row_block + index;
            for (column_block, table) in tables.iter_mut().enumerate() {
                let inverse_scale = scale[row_block * column_blocks + column_block];
                for (bits, entry) in table.iter_mut().enumerate() {
                    *entry = convert(F8E4M3::from_bits(bits as u8).to_f32() * inverse_scale);
                }
            }
            for (output_row, weight_row) in
                output_block.chunks_mut(cols).zip(weight_block.chunks(cols))
            {
                for (table, (output_span, weight_span)) in tables.iter().zip(
                    output_row
                        .chunks_mut(FP8_BLOCK_SIZE)
                        .zip(weight_row.chunks(FP8_BLOCK_SIZE)),
                ) {
                    for (element, quantized) in output_span.iter_mut().zip(weight_span) {
                        *element = table[quantized.to_bits() as usize];
                    }
                }
            }
        }
    }
}

impl CustomOp2 for FusedCpuDequantize {
    fn name(&self) -> &'static str {
        "glm-block-fp8-cpu-fused-lut-v1"
    }

    fn cpu_fwd(
        &self,
        weight: &CpuStorage,
        weight_layout: &Layout,
        scale: &CpuStorage,
        scale_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (rows, cols) = weight_layout.shape().dims2()?;
        if !weight_layout.is_contiguous() || !scale_layout.is_contiguous() {
            candle_core::bail!("GLM fused FP8 dequantization requires contiguous inputs")
        }
        let CpuStorage::F8E4M3(weight) = weight else {
            candle_core::bail!("GLM fused FP8 dequantization requires E4M3 storage")
        };
        let CpuStorage::F32(scale) = scale else {
            candle_core::bail!("GLM fused FP8 dequantization requires F32 inverse scales")
        };
        let weight = &weight[weight_layout.start_offset()..][..rows * cols];
        let scale = &scale[scale_layout.start_offset()..]
            [..rows.div_ceil(FP8_BLOCK_SIZE) * cols.div_ceil(FP8_BLOCK_SIZE)];
        let shape = weight_layout.shape().clone();
        match self.output_dtype {
            DType::BF16 => {
                let mut output = vec![bf16::ZERO; rows * cols];
                Self::fill(rows, cols, weight, scale, &mut output, bf16::from_f32);
                Ok((CpuStorage::BF16(output), shape))
            }
            DType::F32 => {
                let mut output = vec![0f32; rows * cols];
                Self::fill(rows, cols, weight, scale, &mut output, |value| value);
                Ok((CpuStorage::F32(output), shape))
            }
            dtype => candle_core::bail!("GLM fused FP8 dequantization cannot produce {dtype:?}"),
        }
    }

    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        candle_core::bail!("GLM fused FP8 dequantization is the CPU path")
    }
}

/// Multiply a block-FP8 matrix by a vector without ever building the matrix.
///
/// `weight` is `[rows, cols]` E4M3 with one inverse scale per 128-by-128
/// block, `input` is `[cols]`, and the result is `[rows]`, computing
/// `output[r] = sum_c weight[r, c] * scale_inv[r / 128, c / 128] * input[c]`.
///
/// Materializing the dequantized matrix first is what this avoids, and at
/// GLM's expert geometry that is the whole cost. Dequantizing writes 50 MiB of
/// BF16 (or 100 MiB of F32) and the matmul reads it back, so a 25 MiB expert
/// moves about 175 MiB through memory to be used once. Fused, the 25 MiB is
/// read and nothing is written, and the dequantized value never leaves a
/// register. Measured on the pinned host, one routed expert's three
/// projections: 9.93 ms materialized against 1.66 ms fused, which is a host
/// evaluation bandwidth of 4.72 GiB/s against 28.27 GiB/s.
///
/// The products are the same products: a table entry is `f32(value) * scale`
/// in F32, exactly the dequantized weight. The sums are not the same sums --
/// this accumulates a row in column order where a blocked GEMM does not -- so
/// results agree to rounding rather than bit for bit, in the direction the
/// tests pin down.
pub fn fused_block_fp8_matvec(
    weight: &Tensor,
    scale_inv: &Tensor,
    input: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = weight
        .dims2()
        .context("block-FP8 matvec weight must be a rank-2 [rows, cols] matrix")?;
    ensure!(
        weight.dtype() == DType::F8E4M3,
        "block-FP8 matvec weight must have dtype F8E4M3, found {:?}",
        weight.dtype()
    );
    ensure!(
        scale_inv.dtype() == DType::F32 && input.dtype() == DType::F32,
        "block-FP8 matvec requires F32 inverse scales and an F32 input"
    );
    ensure!(
        input.dims() == [cols],
        "block-FP8 matvec input must be [{cols}], found {:?}",
        input.dims()
    );
    let expected_scale_shape = [rows.div_ceil(FP8_BLOCK_SIZE), cols.div_ceil(FP8_BLOCK_SIZE)];
    ensure!(
        scale_inv.dims() == expected_scale_shape,
        "block-FP8 matvec inverse scale must be {expected_scale_shape:?}, found {:?}",
        scale_inv.dims()
    );
    ensure!(
        weight.device().is_cpu() && scale_inv.device().is_cpu() && input.device().is_cpu(),
        "the fused block-FP8 matvec is the CPU path"
    );
    weight
        .apply_op3(scale_inv, input, FusedCpuMatvec)
        .context("failed to multiply a block-FP8 matrix by a vector")
}

/// Independent F32 accumulators per 128-wide block. Eight divides the block
/// and every partial block width GLM produces, and is enough lanes to keep the
/// rounding chain short without spilling.
const ACCUMULATOR_LANES: usize = 8;

struct FusedCpuMatvec;

impl FusedCpuMatvec {
    /// One worker's contiguous run of whole row blocks.
    fn rows_of(
        first_row_block: usize,
        cols: usize,
        column_blocks: usize,
        weight: &[F8E4M3],
        scale: &[f32],
        input: &[f32],
        output: &mut [f32],
    ) {
        // Every table for one row block at once, so a row streams past them
        // without rebuilding: 32 tables of 256 F32 entries is 32 KiB at GLM's
        // width, which stays in L1 alongside the input vector.
        //
        // The table looks like the thing to remove -- a load whose address
        // depends on the datum, which does not vectorize -- and it is not.
        // Decoding E4M3 arithmetically instead (re-bias the exponent into F32,
        // handle subnormals as `m * 2^-9`, select NaN) was measured at 3.17 ms
        // per expert against this version's 1.28 ms. At 32 KiB the tables never
        // leave L1, so each lookup is a cheap hit rather than a gather, and
        // replacing it costs about ten operations to save one load.
        let mut tables = vec![[0f32; 256]; column_blocks];
        for (index, row) in output.iter_mut().enumerate() {
            if index % FP8_BLOCK_SIZE == 0 {
                let row_block = first_row_block + index / FP8_BLOCK_SIZE;
                for (column_block, table) in tables.iter_mut().enumerate() {
                    let inverse_scale = scale[row_block * column_blocks + column_block];
                    for (bits, entry) in table.iter_mut().enumerate() {
                        *entry = F8E4M3::from_bits(bits as u8).to_f32() * inverse_scale;
                    }
                }
            }
            let quantized = &weight[index * cols..][..cols];
            let mut total = 0f32;
            for (table, (weight_span, input_span)) in tables.iter().zip(
                quantized
                    .chunks(FP8_BLOCK_SIZE)
                    .zip(input.chunks(FP8_BLOCK_SIZE)),
            ) {
                // Independent lanes rather than one running sum. A single F32
                // accumulator over a row chains every rounding into the next
                // and measured eight times the error of the blocked GEMM this
                // replaces; splitting the chain costs nothing and also gives
                // the multiplies somewhere to go in parallel.
                let mut lanes = [0f32; ACCUMULATOR_LANES];
                let mut weight_lanes = weight_span.chunks_exact(ACCUMULATOR_LANES);
                let mut input_lanes = input_span.chunks_exact(ACCUMULATOR_LANES);
                for (values, elements) in weight_lanes.by_ref().zip(input_lanes.by_ref()) {
                    for (lane, (value, element)) in
                        lanes.iter_mut().zip(values.iter().zip(elements))
                    {
                        *lane += table[value.to_bits() as usize] * element;
                    }
                }
                let mut block = lanes.iter().sum::<f32>();
                for (value, element) in weight_lanes.remainder().iter().zip(input_lanes.remainder())
                {
                    block += table[value.to_bits() as usize] * element;
                }
                total += block;
            }
            *row = total;
        }
    }
}

impl CustomOp3 for FusedCpuMatvec {
    fn name(&self) -> &'static str {
        "glm-block-fp8-cpu-fused-matvec-v1"
    }

    fn cpu_fwd(
        &self,
        weight: &CpuStorage,
        weight_layout: &Layout,
        scale: &CpuStorage,
        scale_layout: &Layout,
        input: &CpuStorage,
        input_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !weight_layout.is_contiguous()
            || !scale_layout.is_contiguous()
            || !input_layout.is_contiguous()
        {
            candle_core::bail!("the fused block-FP8 matvec requires contiguous inputs")
        }
        let (rows, cols) = weight_layout.shape().dims2()?;
        let (CpuStorage::F8E4M3(weight), CpuStorage::F32(scale), CpuStorage::F32(input)) =
            (weight, scale, input)
        else {
            candle_core::bail!("the fused block-FP8 matvec requires E4M3 weights and F32 scales")
        };
        let weight = &weight[weight_layout.start_offset()..][..rows * cols];
        let column_blocks = cols.div_ceil(FP8_BLOCK_SIZE);
        let scale =
            &scale[scale_layout.start_offset()..][..rows.div_ceil(FP8_BLOCK_SIZE) * column_blocks];
        let input = &input[input_layout.start_offset()..][..cols];

        let mut output = vec![0f32; rows];
        let workers = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1)
            .min(rows.div_ceil(FP8_BLOCK_SIZE))
            .max(1);
        // Split on row-block boundaries so a worker owns whole blocks and never
        // shares a scale row with its neighbour.
        let rows_per_worker = rows
            .div_ceil(workers)
            .next_multiple_of(FP8_BLOCK_SIZE)
            .max(FP8_BLOCK_SIZE);

        let mut output_rest = &mut output[..];
        let mut weight_rest = weight;
        let mut first_row_block = 0usize;
        std::thread::scope(|scope| {
            while !output_rest.is_empty() {
                let take = rows_per_worker.min(output_rest.len());
                let (output_share, output_tail) = output_rest.split_at_mut(take);
                let (weight_share, weight_tail) = weight_rest.split_at(take * cols);
                output_rest = output_tail;
                weight_rest = weight_tail;
                let row_block = first_row_block;
                first_row_block += take / FP8_BLOCK_SIZE;
                scope.spawn(move || {
                    Self::rows_of(
                        row_block,
                        cols,
                        column_blocks,
                        weight_share,
                        scale,
                        input,
                        output_share,
                    );
                });
            }
        });
        Ok((CpuStorage::F32(output), (rows,).into()))
    }

    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        candle_core::bail!("the fused block-FP8 matvec is the CPU path")
    }
}

/// Load and dequantize one block-FP8 linear weight from a checkpoint.
pub fn load_block_fp8_weight(
    weights: &ModelWeights,
    weight_name: &str,
    scale_inv_name: &str,
    device: &Device,
    output_dtype: DType,
) -> Result<Tensor> {
    validate_block_fp8_metadata(weights, weight_name, scale_inv_name, output_dtype)?;

    let staging_device = Device::Cpu;
    let load_device = if device.is_cpu() || device.is_cuda() {
        device
    } else {
        &staging_device
    };
    let weight = weights
        .load(weight_name, load_device)
        .with_context(|| format!("failed to load block-FP8 weight {weight_name:?}"))?;
    let scale_inv = weights
        .load(scale_inv_name, load_device)
        .with_context(|| format!("failed to load block-FP8 inverse scale {scale_inv_name:?}"))?;
    let dequantized = dequantize_block_fp8(&weight, &scale_inv, output_dtype)
        .with_context(|| format!("failed to dequantize block-FP8 weight {weight_name:?}"))?;
    if load_device.same_device(device) {
        Ok(dequantized)
    } else {
        dequantized.to_device(device).with_context(|| {
            format!("failed to move dequantized weight {weight_name:?} to {device:?}")
        })
    }
}

fn validate_block_fp8_metadata(
    weights: &ModelWeights,
    weight_name: &str,
    scale_inv_name: &str,
    output_dtype: DType,
) -> Result<()> {
    validate_output_dtype(output_dtype)?;
    let weight_metadata = weights
        .metadata(weight_name)
        .with_context(|| format!("failed to inspect block-FP8 weight {weight_name:?}"))?;
    ensure!(
        weight_metadata.dtype == "F8_E4M3",
        "checkpoint tensor {weight_name:?} must have dtype F8_E4M3, found {}",
        weight_metadata.dtype
    );
    ensure!(
        weight_metadata.shape.len() == 2,
        "checkpoint tensor {weight_name:?} must be rank 2, found shape {:?}",
        weight_metadata.shape
    );
    let scale_metadata = weights
        .metadata(scale_inv_name)
        .with_context(|| format!("failed to inspect block-FP8 inverse scale {scale_inv_name:?}"))?;
    ensure!(
        scale_metadata.dtype == "F32",
        "checkpoint tensor {scale_inv_name:?} must have dtype F32, found {}",
        scale_metadata.dtype
    );
    let expected_scale_shape = [
        weight_metadata.shape[0].div_ceil(FP8_BLOCK_SIZE),
        weight_metadata.shape[1].div_ceil(FP8_BLOCK_SIZE),
    ];
    ensure!(
        scale_metadata.shape == expected_scale_shape,
        "checkpoint tensor {scale_inv_name:?} must have shape [{}, {}] for weight {weight_name:?} with shape {:?}, found {:?}",
        expected_scale_shape[0],
        expected_scale_shape[1],
        weight_metadata.shape,
        scale_metadata.shape
    );

    Ok(())
}

/// Load an unquantized BF16 linear weight, preserving BF16 unless F32 was
/// explicitly requested.
///
/// `ModelWeights` promotes CPU BF16 tensors to F32 because Candle cannot run
/// CPU half-precision matrix multiplication.  This function converts that
/// promoted value back to BF16 when the caller explicitly requests BF16, so
/// its result contract is independent of the selected device.
pub fn load_bf16_weight(
    weights: &ModelWeights,
    weight_name: &str,
    device: &Device,
    output_dtype: DType,
) -> Result<Tensor> {
    validate_output_dtype(output_dtype)?;
    let metadata = weights
        .metadata(weight_name)
        .with_context(|| format!("failed to inspect BF16 weight {weight_name:?}"))?;
    ensure!(
        metadata.dtype == "BF16",
        "checkpoint tensor {weight_name:?} must have dtype BF16, found {}",
        metadata.dtype
    );
    ensure!(
        metadata.shape.len() == 2,
        "checkpoint tensor {weight_name:?} must be rank 2, found shape {:?}",
        metadata.shape
    );
    ensure!(
        metadata.shape.iter().all(|&dimension| dimension > 0),
        "checkpoint tensor {weight_name:?} dimensions must be non-zero, found {:?}",
        metadata.shape
    );

    weights
        .load(weight_name, device)
        .with_context(|| format!("failed to load BF16 weight {weight_name:?}"))?
        .to_dtype(output_dtype)
        .with_context(|| format!("failed to cast BF16 weight {weight_name:?} to {output_dtype:?}"))
}

/// Load either an ordinary BF16 matrix or an official block-FP8 matrix.
///
/// A scale name is mandatory for FP8 and forbidden for BF16, preventing a
/// missing or accidentally ignored quantization scale from silently producing
/// plausible but incorrect model output.
pub fn load_linear_weight(
    weights: &ModelWeights,
    weight_name: &str,
    scale_inv_name: Option<&str>,
    device: &Device,
    output_dtype: DType,
) -> Result<Tensor> {
    let metadata = weights
        .metadata(weight_name)
        .with_context(|| format!("failed to inspect linear weight {weight_name:?}"))?;
    match metadata.dtype.as_str() {
        "F8_E4M3" => {
            let scale_inv_name = scale_inv_name.with_context(|| {
                format!("block-FP8 weight {weight_name:?} requires its weight_scale_inv tensor")
            })?;
            load_block_fp8_weight(weights, weight_name, scale_inv_name, device, output_dtype)
        }
        "BF16" => {
            if let Some(scale_inv_name) = scale_inv_name {
                bail!(
                    "BF16 weight {weight_name:?} must not have an inverse scale, but {scale_inv_name:?} was supplied"
                );
            }
            load_bf16_weight(weights, weight_name, device, output_dtype)
        }
        dtype => {
            bail!("linear weight {weight_name:?} must have dtype BF16 or F8_E4M3, found {dtype}")
        }
    }
}

fn validate_output_dtype(dtype: DType) -> Result<()> {
    ensure!(
        matches!(dtype, DType::BF16 | DType::F32),
        "linear weight output dtype must be BF16 or F32, found {dtype:?}"
    );
    Ok(())
}

fn block_indices(length: usize, device: &Device) -> Result<Tensor> {
    let indices = (0..length)
        .map(|index| {
            u32::try_from(index / FP8_BLOCK_SIZE).context("block-FP8 scale index exceeds U32")
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_vec(indices, length, device).context("failed to build block-FP8 scale indices")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors;
    use ff_core::weights::{CachePolicy, WeightSource};
    use std::collections::HashMap;

    fn fp8(values: Vec<f32>, shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::from_vec(values, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap()
    }

    /// Every E4M3 bit pattern against a real block grid, checked against the
    /// widen-broadcast-narrow chain the fused table replaces. The claim the
    /// table rests on is that one F32 multiply and one rounding are performed
    /// per element either way, so the two must agree exactly rather than
    /// approximately -- these are model weights.
    #[test]
    fn fused_tables_reproduce_the_generic_chain_bit_for_bit() {
        for (rows, cols) in [
            (256usize, 256usize),
            (300, 200),
            (128, 128),
            (1, 1),
            (129, 257),
        ] {
            let row_blocks = rows.div_ceil(FP8_BLOCK_SIZE);
            let column_blocks = cols.div_ceil(FP8_BLOCK_SIZE);
            // Walk all 256 patterns so every table entry is exercised, including
            // the negatives, the subnormals and NaN.
            let patterns = (0..rows * cols)
                .map(|index| F8E4M3::from_bits((index % 256) as u8))
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(patterns, (rows, cols), &Device::Cpu).unwrap();
            let scales = (0..row_blocks * column_blocks)
                .map(|index| 0.125_f32 * (index as f32 + 1.0) - 0.4)
                .collect::<Vec<_>>();
            let scale =
                Tensor::from_vec(scales, (row_blocks, column_blocks), &Device::Cpu).unwrap();

            for output_dtype in [DType::F32, DType::BF16] {
                let fused = dequantize_block_fp8(&weight, &scale, output_dtype).unwrap();
                assert_eq!(fused.dtype(), output_dtype);
                assert_eq!(fused.dims(), &[rows, cols]);

                // The chain the fused path replaces, rebuilt here rather than
                // called, so the test does not depend on the branch it checks.
                let widened = weight.to_dtype(DType::F32).unwrap();
                let row_index = block_indices(rows, &Device::Cpu).unwrap();
                let column_index = block_indices(cols, &Device::Cpu).unwrap();
                let expanded = scale
                    .index_select(&row_index, 0)
                    .unwrap()
                    .index_select(&column_index, 1)
                    .unwrap();
                let reference = widened.broadcast_mul(&expanded).unwrap();
                let reference = if output_dtype == DType::F32 {
                    reference
                } else {
                    reference.to_dtype(output_dtype).unwrap()
                };

                let (fused, reference) = match output_dtype {
                    DType::F32 => (
                        fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        reference.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    ),
                    _ => (
                        fused
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<bf16>()
                            .unwrap()
                            .iter()
                            .map(|value| value.to_f32())
                            .collect(),
                        reference
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<bf16>()
                            .unwrap()
                            .iter()
                            .map(|value| value.to_f32())
                            .collect(),
                    ),
                };
                for (index, (left, right)) in fused.iter().zip(&reference).enumerate() {
                    assert_eq!(
                        left.to_bits(),
                        right.to_bits(),
                        "[{rows}x{cols}] {output_dtype:?} element {index} \
                         diverged: {left} vs {right}"
                    );
                }
            }
        }
    }

    /// The fused matvec multiplies the same numbers a dequantize-then-matmul
    /// does, and sums them in a different order, so the question is not whether
    /// it matches that path bit for bit -- it cannot -- but whether it is at
    /// least as close to the answer. Both are measured against the same
    /// products accumulated in F64.
    #[test]
    fn the_fused_matvec_is_no_less_accurate_than_materializing_first() {
        // The tall shape matters: workers are capped at the row-block count,
        // so only a matrix with more row blocks than this host has threads
        // makes one worker walk several blocks and rebuild its tables midway.
        for (rows, cols) in [
            (256usize, 512usize),
            (128, 128),
            (300, 200),
            (1, 1),
            (FP8_BLOCK_SIZE * 65 + 7, 128),
        ] {
            let row_blocks = rows.div_ceil(FP8_BLOCK_SIZE);
            let column_blocks = cols.div_ceil(FP8_BLOCK_SIZE);
            // Every E4M3 pattern except the two that are NaN. A NaN weight
            // poisons its whole row, and `f64::max` silently returns the other
            // operand when handed one, so leaving them in would make every
            // comparison below vacuous -- which is exactly what it did until a
            // mutation survived and said so.
            let patterns = (0..rows * cols)
                .map(|index| {
                    let bits = ((index * 7 + index / cols) % 256) as u8;
                    F8E4M3::from_bits(if bits & 0x7f == 0x7f { 0x01 } else { bits })
                })
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(patterns.clone(), (rows, cols), &Device::Cpu).unwrap();
            let scales = (0..row_blocks * column_blocks)
                .map(|index| 0.0625_f32 * (index as f32 + 1.0))
                .collect::<Vec<_>>();
            let scale = Tensor::from_vec(scales.clone(), (row_blocks, column_blocks), &Device::Cpu)
                .unwrap();
            let values = (0..cols)
                .map(|index| (index as f32 * 0.013).sin())
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(values.clone(), cols, &Device::Cpu).unwrap();

            let fused = fused_block_fp8_matvec(&weight, &scale, &input)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(fused.len(), rows);
            assert!(
                fused.iter().all(|value| value.is_finite()),
                "[{rows}x{cols}] the fused matvec produced a non-finite row"
            );

            let materialized = dequantize_block_fp8(&weight, &scale, DType::F32)
                .unwrap()
                .matmul(&input.reshape((cols, 1)).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // The same F32 products, accumulated without F32 rounding.
            let mut fused_error = 0f64;
            let mut materialized_error = 0f64;
            for row in 0..rows {
                let mut exact = 0f64;
                for column in 0..cols {
                    let block = (row / FP8_BLOCK_SIZE) * column_blocks + column / FP8_BLOCK_SIZE;
                    let weight = patterns[row * cols + column].to_f32() * scales[block];
                    exact += f64::from(weight) * f64::from(values[column]);
                }
                assert!(
                    exact.is_finite(),
                    "[{rows}x{cols}] row {row} reference is not finite"
                );
                let scale = exact.abs().max(1.0);
                fused_error = fused_error.max((f64::from(fused[row]) - exact).abs() / scale);
                materialized_error =
                    materialized_error.max((f64::from(materialized[row]) - exact).abs() / scale);
            }
            // F32 accumulation of hundreds of terms cannot do better than
            // this, and the blocked GEMM being replaced does not either. The
            // bound that carries the weight is the comparison below.
            assert!(
                fused_error < 1e-4,
                "[{rows}x{cols}] fused relative error {fused_error:e}"
            );
            assert!(
                fused_error <= materialized_error * 4.0,
                "[{rows}x{cols}] fused {fused_error:e} is worse than \
                 materialized {materialized_error:e}"
            );
        }
    }

    #[test]
    fn the_fused_matvec_refuses_shapes_and_dtypes_it_cannot_honour() {
        let weight = fp8(vec![1.; 256 * 128], (256, 128));
        let scale = Tensor::from_vec(vec![2_f32, 3.], (2, 1), &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1_f32; 128], 128, &Device::Cpu).unwrap();
        fused_block_fp8_matvec(&weight, &scale, &input).unwrap();

        let wrong_input = Tensor::from_vec(vec![1_f32; 127], 127, &Device::Cpu).unwrap();
        assert!(fused_block_fp8_matvec(&weight, &scale, &wrong_input).is_err());
        let wrong_scale = Tensor::from_vec(vec![2_f32], (1, 1), &Device::Cpu).unwrap();
        assert!(fused_block_fp8_matvec(&weight, &wrong_scale, &input).is_err());
        let bf16_weight = weight.to_dtype(DType::BF16).unwrap();
        assert!(fused_block_fp8_matvec(&bf16_weight, &scale, &input).is_err());
        let f64_input = input.to_dtype(DType::F64).unwrap();
        assert!(fused_block_fp8_matvec(&weight, &scale, &f64_input).is_err());
    }

    #[test]
    fn dequantizes_complete_blocks_in_f32() {
        let weight = fp8(vec![1.; 256 * 128], (256, 128));
        let scale = Tensor::from_vec(vec![2_f32, 3.], (2, 1), &Device::Cpu).unwrap();
        let output = dequantize_block_fp8(&weight, &scale, DType::F32).unwrap();

        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), &[256, 128]);
        let rows = output.to_vec2::<f32>().unwrap();
        assert_eq!(rows[0][0], 2.);
        assert_eq!(rows[127][127], 2.);
        assert_eq!(rows[128][0], 3.);
        assert_eq!(rows[255][127], 3.);
    }

    #[test]
    fn dequantizes_partial_edge_blocks_to_bf16() {
        let weight = fp8(vec![1.; 129 * 130], (129, 130));
        let scale = Tensor::from_vec(vec![1_f32, 2., 3., 4.], (2, 2), &Device::Cpu).unwrap();
        let output = dequantize_block_fp8(&weight, &scale, DType::BF16).unwrap();

        assert_eq!(output.dtype(), DType::BF16);
        let output = output.to_dtype(DType::F32).unwrap();
        let rows = output.to_vec2::<f32>().unwrap();
        assert_eq!(rows[0][0], 1.);
        assert_eq!(rows[0][129], 2.);
        assert_eq!(rows[128][0], 3.);
        assert_eq!(rows[128][129], 4.);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dequantizes_aligned_blocks_on_cuda() {
        let device = Device::new_cuda(0).unwrap();
        let weight = fp8(vec![1.; 128 * 256], (128, 256))
            .to_device(&device)
            .unwrap()
            .contiguous()
            .unwrap();
        let scale = Tensor::from_vec(vec![2_f32, 3.], (1, 2), &device).unwrap();
        let output = dequantize_block_fp8(&weight, &scale, DType::BF16).unwrap();

        assert_eq!(output.dtype(), DType::BF16);
        let rows = output
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(rows[0][0], 2.);
        assert_eq!(rows[127][127], 2.);
        assert_eq!(rows[0][128], 3.);
        assert_eq!(rows[127][255], 3.);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fp8_matches_cpu_all_encodings_scales_edges_and_offsets() {
        let device = Device::new_cuda(0).unwrap();
        let bytes: Vec<u8> = (0..130 * 259).map(|i| (i % 256) as u8).collect();
        let host =
            Tensor::from_raw_buffer(&bytes, DType::F8E4M3, &[130, 259], &Device::Cpu).unwrap();
        let gpu = host.to_device(&device).unwrap();
        let host = host.narrow(0, 1, 129).unwrap();
        let gpu = gpu.narrow(0, 1, 129).unwrap();
        for factors in [
            [1., -1., 1. + 1. / 256., 0., -0., 0.33333334],
            [f32::MIN_POSITIVE, 1e-38, 1e-35, 1e35, f32::MAX, -1e-38],
        ] {
            let mut scale_data = vec![42f32; 4 * 5];
            for r in 0..2 {
                for c in 0..3 {
                    scale_data[(r + 1) * 5 + c + 1] = factors[r * 3 + c];
                }
            }
            let scales = Tensor::from_vec(scale_data, (4, 5), &Device::Cpu).unwrap();
            let device_scales = scales
                .to_device(&device)
                .unwrap()
                .narrow(0, 1, 2)
                .unwrap()
                .narrow(1, 1, 3)
                .unwrap();
            let scales = scales.narrow(0, 1, 2).unwrap().narrow(1, 1, 3).unwrap();
            for dtype in [DType::F32, DType::BF16] {
                let expected = dequantize_block_fp8(&host, &scales.contiguous().unwrap(), dtype)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let actual = dequantize_block_fp8(&gpu, &device_scales, dtype)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    if b.is_nan() {
                        assert!(a.is_nan(), "{dtype:?} at {i}");
                    } else {
                        assert_eq!(a.to_bits(), b.to_bits(), "{dtype:?} at {i}: {a} != {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_weight_scale_and_output_contracts() {
        let f32_weight = Tensor::ones((128, 128), DType::F32, &Device::Cpu).unwrap();
        let fp8_weight = f32_weight.to_dtype(DType::F8E4M3).unwrap();
        let scale = Tensor::ones((1, 1), DType::F32, &Device::Cpu).unwrap();

        let error = dequantize_block_fp8(&f32_weight, &scale, DType::F32).unwrap_err();
        assert!(error.to_string().contains("must have dtype F8E4M3"));

        let bf16_scale = scale.to_dtype(DType::BF16).unwrap();
        let error = dequantize_block_fp8(&fp8_weight, &bf16_scale, DType::F32).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("inverse scale must have dtype F32")
        );

        let wrong_shape = Tensor::ones((1, 2), DType::F32, &Device::Cpu).unwrap();
        let error = dequantize_block_fp8(&fp8_weight, &wrong_shape, DType::F32).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("inverse scale shape must be [1, 1]")
        );

        let error = dequantize_block_fp8(&fp8_weight, &scale, DType::F16).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("output dtype must be BF16 or F32")
        );

        let rank_one = fp8(vec![1.; 128], 128);
        let error = dequantize_block_fp8(&rank_one, &scale, DType::F32).unwrap_err();
        assert!(error.to_string().contains("must be a rank-2"));
    }

    fn checkpoint(tensors: HashMap<String, Tensor>) -> (tempfile::TempDir, ModelWeights) {
        let directory = tempfile::tempdir().unwrap();
        safetensors::save(&tensors, directory.path().join("model.safetensors")).unwrap();
        let weights =
            ModelWeights::open(directory.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        (directory, weights)
    }

    #[test]
    fn unified_loader_handles_bf16_and_requires_fp8_scale() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "plain.weight".to_owned(),
            Tensor::from_vec(vec![1., 2., 3., 4.], (2, 2), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        tensors.insert(
            "quant.weight".to_owned(),
            fp8(vec![1.; 128 * 128], (128, 128)),
        );
        tensors.insert(
            "quant.weight_scale_inv".to_owned(),
            Tensor::from_vec(vec![2_f32], (1, 1), &Device::Cpu).unwrap(),
        );
        let (_directory, weights) = checkpoint(tensors);

        let plain =
            load_linear_weight(&weights, "plain.weight", None, &Device::Cpu, DType::BF16).unwrap();
        assert_eq!(plain.dtype(), DType::BF16);
        assert_eq!(
            plain
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1., 2.], vec![3., 4.]]
        );

        let missing_scale =
            load_linear_weight(&weights, "quant.weight", None, &Device::Cpu, DType::F32)
                .unwrap_err();
        assert!(
            missing_scale
                .to_string()
                .contains("requires its weight_scale_inv")
        );

        let quantized = load_linear_weight(
            &weights,
            "quant.weight",
            Some("quant.weight_scale_inv"),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        let rows = quantized.to_vec2::<f32>().unwrap();
        assert_eq!(rows[0][0], 2.);
        assert_eq!(rows[127][127], 2.);

        let extra_scale = load_linear_weight(
            &weights,
            "plain.weight",
            Some("quant.weight_scale_inv"),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err();
        assert!(
            extra_scale
                .to_string()
                .contains("must not have an inverse scale")
        );
    }
}
