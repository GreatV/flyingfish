use crate::backend::cuda::contracts::{SeqRef, Tensor, dense_layout, spans_disjoint};
use crate::backend::cuda::cubin;
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

/// F32 projection outputs and the existing request sequence metadata.
#[derive(Clone, Copy)]
pub struct CompressionInputs {
    pub values: Tensor<f32, 2>,
    pub scores: Tensor<f32, 2>,
    pub sequence: SeqRef,
}

/// Caller-owned state and pooled BF16 output before RMS normalization.
#[derive(Clone, Copy)]
pub struct CompressionBuffers {
    pub tail_values: Tensor<f32, 2>,
    pub tail_scores: Tensor<f32, 2>,
    pub cursor: Tensor<u32, 1>,
    pub pooled: Tensor<bf16, 2>,
}

fn dense<T: Copy, const R: usize>(view: Tensor<T, R>) -> Result<()> {
    view.validate()?;
    ensure!(
        view.stride == dense_layout(view.shape)?.0,
        "compression view must be dense"
    );
    Ok(())
}

fn geometry(inputs: CompressionInputs, buffers: CompressionBuffers) -> Result<(u32, u32, u32)> {
    inputs.sequence.validate()?;
    dense(inputs.values)?;
    dense(inputs.scores)?;
    dense(buffers.tail_values)?;
    dense(buffers.tail_scores)?;
    dense(buffers.cursor)?;
    dense(buffers.pooled)?;
    let [rows, channels] = inputs.values.shape;
    ensure!(
        rows <= i32::MAX as u32 && channels <= (i32::MAX / 2) as u32,
        "compression extent exceeds kernel index range"
    );
    ensure!(
        inputs.scores.shape == [rows, channels],
        "compression score shape mismatch"
    );
    ensure!(
        buffers.tail_values.shape == [2, channels] && buffers.tail_scores.shape == [2, channels],
        "compression tail shape mismatch"
    );
    ensure!(
        buffers.cursor.shape == [1],
        "compression cursor shape mismatch"
    );
    ensure!(
        buffers.pooled.shape == [rows.div_ceil(2), channels],
        "compression output shape mismatch"
    );
    let spans = [
        (inputs.values.ptr, inputs.values.span_bytes),
        (inputs.scores.ptr, inputs.scores.span_bytes),
        (buffers.tail_values.ptr, buffers.tail_values.span_bytes),
        (buffers.tail_scores.ptr, buffers.tail_scores.span_bytes),
        (buffers.cursor.ptr, buffers.cursor.span_bytes),
        (buffers.pooled.ptr, buffers.pooled.span_bytes),
        (inputs.sequence.ptr, 32),
    ];
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, other_bytes) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, other_bytes)?,
                "compression views overlap"
            );
        }
    }
    let blocks = rows
        .div_ceil(2)
        .checked_mul(channels.div_ceil(256))
        .context("compression launch extent overflow")?;
    ensure!(
        blocks <= i32::MAX as u32,
        "compression grid exceeds device x extent"
    );
    Ok((rows, channels, blocks))
}

/// Ratio-two pooling; row capacity, addresses and launch geometry bind at setup.
pub struct LatentCompressPlan {
    pool: CudaFunction,
    tail: CudaFunction,
    inputs: CompressionInputs,
    buffers: CompressionBuffers,
    rows: i32,
    channels: i32,
    blocks: u32,
}

impl LatentCompressPlan {
    pub fn new(
        ctx: &Arc<CudaContext>,
        inputs: CompressionInputs,
        buffers: CompressionBuffers,
    ) -> Result<Self> {
        let (rows, channels, blocks) = geometry(inputs, buffers)?;
        let module = cubin::module(ctx, "latent_compress")?;
        Ok(Self {
            pool: module.load_function("latent_compress_pool")?,
            tail: module.load_function("latent_compress_tail")?,
            inputs,
            buffers,
            rows: rows as i32,
            channels: channels as i32,
            blocks,
        })
    }

    /// The caller initializes cursor to zero and includes cursor/tails in request snapshots.
    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let i = self.inputs;
        let b = self.buffers;
        unsafe {
            stream
                .launch_builder(&self.pool)
                .arg(&i.values.ptr)
                .arg(&i.scores.ptr)
                .arg(&b.tail_values.ptr)
                .arg(&b.tail_scores.ptr)
                .arg(&b.cursor.ptr)
                .arg(&i.sequence.ptr)
                .arg(&b.pooled.ptr)
                .arg(&self.rows)
                .arg(&self.channels)
                .launch(LaunchConfig {
                    grid_dim: (self.blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.tail)
                .arg(&i.values.ptr)
                .arg(&i.scores.ptr)
                .arg(&b.tail_values.ptr)
                .arg(&b.tail_scores.ptr)
                .arg(&b.cursor.ptr)
                .arg(&i.sequence.ptr)
                .arg(&self.rows)
                .arg(&self.channels)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::contracts::view;
    use crate::backend::gpu::DevicePtr;

    fn tensor<T: Copy, const R: usize>(ptr: u64, shape: [u32; R]) -> Result<Tensor<T, R>> {
        let (stride, count) = dense_layout(shape)?;
        unsafe { Tensor::new(ptr, shape, stride, count * std::mem::size_of::<T>() as u64) }
    }

    #[test]
    fn compression_bounds_preserve_partial_groups() -> Result<()> {
        for rows in [1, 2, 3, 5, 17, 256] {
            let inputs = CompressionInputs {
                values: tensor(0x100000, [rows, 512])?,
                scores: tensor(0x200000, [rows, 512])?,
                sequence: SeqRef { ptr: 0x800000 },
            };
            let buffers = CompressionBuffers {
                tail_values: tensor(0x300000, [2, 512])?,
                tail_scores: tensor(0x400000, [2, 512])?,
                cursor: tensor(0x500000, [1])?,
                pooled: tensor(0x600000, [rows.div_ceil(2), 512])?,
            };
            assert_eq!(
                geometry(inputs, buffers)?,
                (rows, 512, 2 * rows.div_ceil(2))
            );
            assert!(
                geometry(
                    inputs,
                    CompressionBuffers {
                        tail_scores: buffers.tail_values,
                        ..buffers
                    }
                )
                .is_err()
            );
            assert!(
                geometry(
                    inputs,
                    CompressionBuffers {
                        pooled: tensor(0x600000, [rows.div_ceil(2) + 1, 512])?,
                        ..buffers
                    }
                )
                .is_err()
            );
            assert!(
                geometry(
                    inputs,
                    CompressionBuffers {
                        tail_values: tensor(0x300000, [1, 512])?,
                        ..buffers
                    }
                )
                .is_err()
            );
            assert!(
                geometry(
                    CompressionInputs {
                        sequence: SeqRef {
                            ptr: buffers.cursor.ptr
                        },
                        ..inputs
                    },
                    buffers
                )
                .is_err()
            );
        }
        Ok(())
    }

    fn projected(seed: usize, start: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let indices = start * 512..(start + rows) * 512;
        (
            indices
                .clone()
                .map(|i| (((i * 37 + seed) % 251) as f32 - 125.0) / 16.0)
                .collect(),
            indices
                .map(|i| (((i * 29 + seed) % 47) as f32 - 23.0) / 8.0)
                .collect(),
        )
    }

    fn oracle(seed: usize) -> Vec<f64> {
        let (values, scores) = projected(seed, 0, 20);
        (0..10 * 512)
            .map(|i| {
                let first = (i / 512 * 2) * 512 + i % 512;
                let s0 = f64::from(scores[first]);
                let s1 = f64::from(scores[first + 512]);
                let e0 = (s0 - s0.max(s1)).exp();
                let e1 = (s1 - s0.max(s1)).exp();
                (f64::from(values[first]) * e0 + f64::from(values[first + 512]) * e1) / (e0 + e1)
            })
            .collect()
    }

    struct Trace {
        pooled: Vec<u16>,
        tail_values: Vec<f32>,
        tail_scores: Vec<f32>,
    }

    fn trace(
        ctx: &Arc<CudaContext>,
        seed: usize,
        initial: usize,
        graph: bool,
        control: bool,
    ) -> Result<Trace> {
        let (values, scores) = projected(seed, 0, 21);
        trace_operands(ctx, &values, &scores, initial, [graph, control])
    }

    fn trace_operands(
        ctx: &Arc<CudaContext>,
        values: &[f32],
        scores: &[f32],
        initial: usize,
        mode: [bool; 2],
    ) -> Result<Trace> {
        let [graph, control] = mode;
        let total = values.len() / 512;
        ensure!(
            values.len() == scores.len() && values.len().is_multiple_of(512) && initial <= total,
            "compression recorded operands extent"
        );
        let stream = ctx.new_stream()?;
        let mut tail_values = stream.alloc_zeros::<f32>(1024)?;
        let tail_scores = stream.alloc_zeros::<f32>(1024)?;
        let cursor = stream.alloc_zeros::<u32>(1)?;
        let mut result = Vec::new();
        for start in std::iter::once(0).chain(initial..total) {
            let rows = if start == 0 { initial } else { 1 };
            let (values, scores) = (
                &values[start * 512..(start + rows) * 512],
                &scores[start * 512..(start + rows) * 512],
            );
            let values = stream.clone_htod(values)?;
            let scores = stream.clone_htod(scores)?;
            let metadata = stream.clone_htod(&[
                start as i32,
                rows as i32,
                (start + rows) as i32,
                total as i32,
                1,
                0,
                0,
                0,
            ])?;
            let pointer = metadata.device_ptr(&stream).0;
            let count = rows.div_ceil(2) * 512;
            let sentinel = bf16::from_f32(31.0);
            let output = stream.clone_htod(&vec![sentinel; count + 8])?;
            let input = CompressionInputs {
                values: view(&values, &stream, 0, [rows as u32, 512], [512, 1])?,
                scores: view(&scores, &stream, 0, [rows as u32, 512], [512, 1])?,
                sequence: SeqRef { ptr: pointer },
            };
            let buffers = CompressionBuffers {
                tail_values: view(&tail_values, &stream, 0, [2, 512], [512, 1])?,
                tail_scores: view(&tail_scores, &stream, 0, [2, 512], [512, 1])?,
                cursor: view(&cursor, &stream, 0, [1], [1])?,
                pooled: view(
                    &output,
                    &stream,
                    0,
                    [rows.div_ceil(2) as u32, 512],
                    [512, 1],
                )?,
            };
            let plan = LatentCompressPlan::new(ctx, input, buffers)?;
            if control && start == 1 {
                stream.memcpy_htod(&vec![19.0; 1024], &mut tail_values)?;
            }
            if graph {
                let captured = crate::backend::cuda::moe::testing::capture(&stream, || {
                    plan.run(&stream)?;
                    Ok(())
                })?;
                captured.launch()?;
            } else {
                plan.run(&stream)?;
            }
            let got = stream.clone_dtoh(&output)?;
            let valid = ((start % 2 + rows) / 2) * 512;
            ensure!(
                got[valid..]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits()),
                "compression output guard changed"
            );
            result.extend(got[..valid].iter().map(|v| v.to_bits()));
            ensure!(
                stream.clone_dtoh(&cursor)? == [(start + rows) as u32],
                "compression cursor publication differs"
            );
        }
        Ok(Trace {
            pooled: result,
            tail_values: stream.clone_dtoh(&tail_values)?,
            tail_scores: stream.clone_dtoh(&tail_scores)?,
        })
    }

    #[test]
    #[ignore = "requires a GPU lease and D1_F1_REFERENCE publisher fixture"]
    fn latent_compression_gpu() -> Result<()> {
        let path = std::env::var("D1_F1_REFERENCE").context("D1_F1_REFERENCE is required")?;
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture["reference_only"] == true,
            "compression floor must be reference-only"
        );
        let reference = fixture["compression"]
            .as_array()
            .context("compression reference floor missing")?;
        ensure!(
            reference.len() == 40,
            "compression reference matrix requires two devices, two seeds, five widths and two repeats"
        );
        let mut floor = reference
            .iter()
            .try_fold(0.0f64, |floor, cell| -> Result<f64> {
                let error = cell["fp64_max_abs"]
                    .as_f64()
                    .context("compression FP64 floor absent")?;
                ensure!(
                    error.is_finite() && error > 0.0,
                    "invalid compression reference floor"
                );
                Ok(floor.max(error))
            })?;
        for seed in [11, 29] {
            let mut minimum = vec![f64::INFINITY; 10 * 512];
            let mut maximum = vec![f64::NEG_INFINITY; 10 * 512];
            let cells: Vec<_> = reference.iter().filter(|c| c["seed"] == seed).collect();
            ensure!(cells.len() == 20, "compression seed coverage incomplete");
            for cell in cells {
                let bytes = cell["pooled"]["bytes"]
                    .as_array()
                    .context("reference pooled bytes missing")?;
                ensure!(
                    bytes.len() == 2 * minimum.len(),
                    "reference pooled extent differs"
                );
                for (index, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
                    let low = u8::try_from(pair[0].as_u64().context("invalid reference byte")?)?;
                    let high = u8::try_from(pair[1].as_u64().context("invalid reference byte")?)?;
                    let value =
                        f64::from(bf16::from_bits(u16::from_le_bytes([low, high])).to_f32());
                    ensure!(value.is_finite(), "nonfinite compression reference");
                    minimum[index] = minimum[index].min(value);
                    maximum[index] = maximum[index].max(value);
                }
            }
            for (low, high) in minimum.into_iter().zip(maximum) {
                floor = floor.max(high - low);
            }
        }
        let threshold = 3.0 * floor;
        let ctx = CudaContext::new(0)?;
        let mut cases = 0;
        let mut maximum = 0.0f64;
        for seed in [11, 29] {
            let expected = oracle(seed);
            let baseline = trace(&ctx, seed, 1, false, false)?;
            for initial in [1, 2, 3, 5, 17] {
                for graph in [false, true] {
                    let result = trace(&ctx, seed, initial, graph, false)?;
                    ensure!(
                        result.pooled == baseline.pooled,
                        "compression chunk/graph bit identity differs"
                    );
                    ensure!(
                        result.tail_values == baseline.tail_values
                            && result.tail_scores == baseline.tail_scores,
                        "compression tail state differs"
                    );
                    for (bits, &want) in result.pooled.iter().zip(&expected) {
                        let error = (f64::from(bf16::from_bits(*bits).to_f32()) - want).abs();
                        ensure!(
                            error <= threshold,
                            "compression FP64 error {error} > frozen {threshold}"
                        );
                        maximum = maximum.max(error);
                    }
                    cases += 1;
                }
            }
            let mutant = trace(&ctx, seed, 1, false, true)?;
            ensure!(
                mutant
                    .pooled
                    .iter()
                    .zip(&expected)
                    .any(
                        |(&bits, &want)| (f64::from(bf16::from_bits(bits).to_f32()) - want).abs()
                            > threshold
                    ),
                "wrong-tail control escaped"
            );
        }
        println!(
            "PASS LATENT_COMPRESSION cases={cases} fp64_max={maximum} reference_floor={floor} threshold={threshold} chunk_graph_bits=0 tail_bits=0 wrong_tail_control=true"
        );
        Ok(())
    }
    #[derive(serde::Deserialize)]
    struct RecordedPool {
        owner: usize,
        ratio: usize,
        rows: usize,
        values: Vec<f32>,
        scores: Vec<f32>,
        pooled: Vec<u16>,
        bound: f64,
    }

    #[test]
    #[ignore = "requires GPU lease and D1_OP_REFERENCE isolated publisher operands"]
    fn compressor_publisher_operands_gpu() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Record {
            reference_only: bool,
            candidate_seen: bool,
            compressions: Vec<RecordedPool>,
        }
        let record: Record =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_OP_REFERENCE")?)?)?;
        ensure!(
            record.reference_only && !record.candidate_seen && record.compressions.len() == 4,
            "compressor reference scope"
        );
        let ctx = CudaContext::new(0)?;
        let (mut defects, mut cases, mut controls) = (0, 0, 0);
        for cell in record.compressions.iter().filter(|v| v.ratio == 2) {
            ensure!(
                cell.rows == 129
                    && cell.pooled.len() == 64 * 512
                    && cell.bound.is_finite()
                    && cell.bound >= 0.0,
                "compressor reference dimensions/bound"
            );
            let baseline = trace_operands(&ctx, &cell.values, &cell.scores, 1, [false, false])?;
            for initial in [1, 5, 17, 129] {
                for graph in [false, true] {
                    let result =
                        trace_operands(&ctx, &cell.values, &cell.scores, initial, [graph, false])?;
                    let mut bad = 0;
                    let mut maximum = 0.0f64;
                    for (index, (&got, &want)) in result.pooled.iter().zip(&cell.pooled).enumerate()
                    {
                        let distance = (f64::from(bf16::from_bits(got).to_f32())
                            - f64::from(bf16::from_bits(want).to_f32()))
                        .abs();
                        maximum = maximum.max(distance);
                        if distance > cell.bound {
                            let row = index / 512 * 2;
                            let column = index % 512;
                            println!(
                                "COMPRESSOR_STORE owner={} initial={initial} graph={graph} row={row} column={column} candidate={got} publisher={want} v0={:?} v1={:?} s0={:?} s1={:?}",
                                cell.owner,
                                cell.values[row * 512 + column],
                                cell.values[(row + 1) * 512 + column],
                                cell.scores[row * 512 + column],
                                cell.scores[(row + 1) * 512 + column]
                            );
                            bad += 1;
                        }
                    }
                    let repeat = result
                        .pooled
                        .iter()
                        .zip(&baseline.pooled)
                        .filter(|(a, b)| a != b)
                        .count();
                    let tail = result.tail_values[..512]
                        .iter()
                        .zip(&cell.values[128 * 512..])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                        + result.tail_scores[..512]
                            .iter()
                            .zip(&cell.scores[128 * 512..])
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                    defects += bad + repeat + tail;
                    cases += 1;
                    println!(
                        "COMPRESSOR_OWNER owner={} initial={initial} graph={graph} beyond={bad} repeat_bits={repeat} tail_bits={tail} max={maximum:e} bound={:e}",
                        cell.owner, cell.bound
                    );
                }
            }
            let mutant = trace_operands(&ctx, &cell.values, &cell.scores, 1, [false, true])?;
            let killed = mutant.pooled.iter().zip(&cell.pooled).any(|(&a, &b)| {
                (f64::from(bf16::from_bits(a).to_f32()) - f64::from(bf16::from_bits(b).to_f32()))
                    .abs()
                    > cell.bound
            });
            ensure!(killed, "wrong compression tail escaped");
            controls += 1;
        }
        println!(
            "{} COMPRESSOR_PUBLISHER cases={cases} defects={defects} wrong_tail_controls={controls} ratio1=projection_only",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(
            defects == 0 && cases == 24 && controls == 3,
            "compressor publisher DEFECT"
        );
        Ok(())
    }
}
