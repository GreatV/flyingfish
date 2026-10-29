use super::{
    contracts::{SeqRef, Tensor, dense_layout, spans_disjoint},
    cubin,
};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct GatedResidualViews {
    pub input: Tensor<bf16, 3>,
    pub projection: Tensor<bf16, 2>,
    pub q_weight: Tensor<bf16, 2>,
    pub k_weight: Tensor<bf16, 2>,
    pub mask: Tensor<u8, 1>,
    pub output: Tensor<bf16, 3>,
    pub gates: Tensor<f32, 2>,
    pub sequence: SeqRef,
}

fn dense<T: Copy, const R: usize>(v: Tensor<T, R>) -> Result<(u64, u64)> {
    v.validate()?;
    ensure!(
        v.stride == dense_layout(v.shape)?.0,
        "gated residual views must be dense"
    );
    Ok((v.ptr, v.span_bytes))
}

fn geometry(v: GatedResidualViews) -> Result<([i32; 3], u32)> {
    v.sequence.validate()?;
    let spans = [
        dense(v.input)?,
        dense(v.projection)?,
        dense(v.q_weight)?,
        dense(v.k_weight)?,
        dense(v.mask)?,
        dense(v.output)?,
        dense(v.gates)?,
        (v.sequence.ptr, 32),
    ];
    let [rows, copies, width] = v.input.shape;
    let projected = (u64::from(copies) + 1) * u64::from(width);
    let blocks = u64::from(rows) * u64::from(copies);
    ensure!(
        projected <= i32::MAX as u64 && blocks <= i32::MAX as u64,
        "gated residual extent overflow"
    );
    ensure!(
        v.projection.shape == [rows, projected as u32]
            && v.q_weight.shape == [copies, width]
            && v.k_weight.shape == [copies, width]
            && v.mask.shape == [rows]
            && v.gates.shape == [rows, copies]
            && v.output.shape == v.input.shape,
        "gated residual projection/stream geometry"
    );
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "gated residual views overlap"
            );
        }
    }
    Ok(([rows as i32, copies as i32, width as i32], blocks as u32))
}

/// Weighted per-stream normalization, signed-square-root gate and BF16 residual.
pub struct GatedResidualPlan {
    kernel: CudaFunction,
    views: GatedResidualViews,
    shape: [i32; 3],
    epsilon: f32,
    clamp: f32,
    scale: f32,
    grid: LaunchConfig,
}

impl GatedResidualPlan {
    pub fn new(
        ctx: &Arc<CudaContext>,
        views: GatedResidualViews,
        epsilon: f32,
        clamp: f32,
    ) -> Result<Self> {
        let (shape, blocks) = geometry(views)?;
        ensure!(
            epsilon.is_finite() && epsilon > 0.0 && clamp.is_finite() && clamp > 0.0,
            "gated residual requires positive finite epsilon/clamp"
        );
        Ok(Self {
            kernel: cubin::module(ctx, "gated_residual")?
                .load_function("normalized_gate_residual")?,
            views,
            shape,
            epsilon,
            clamp,
            scale: f64::from(shape[2]).sqrt().recip() as f32,
            grid: LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let v = self.views;
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&v.input.ptr)
                .arg(&v.projection.ptr)
                .arg(&v.q_weight.ptr)
                .arg(&v.k_weight.ptr)
                .arg(&v.mask.ptr)
                .arg(&v.sequence.ptr)
                .arg(&v.output.ptr)
                .arg(&v.gates.ptr)
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .arg(&self.epsilon)
                .arg(&self.clamp)
                .arg(&self.scale)
                .launch(self.grid)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::contracts::view;
    use super::*;
    use anyhow::Context;
    use cudarc::driver::DevicePtr;

    // 2026-10-28: three times max(FP64, repeat, shape, sm86/sm89 reference spread).
    const GATE_LIMIT: f64 = 5.408762987491045e-7;
    const OUTPUT_LIMIT: f64 = 0.02343447780376806;

    #[derive(serde::Deserialize)]
    struct Cell {
        rows: usize,
        seed: u32,
        x: Vec<u8>,
        kv: Vec<u8>,
        q_weight: Vec<u8>,
        k_weight: Vec<u8>,
        mask: Vec<u8>,
        output: Vec<u8>,
        gates: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        reference_only: bool,
        cells: Vec<Cell>,
    }

    fn bf16_values(bytes: &[u8]) -> Result<Vec<bf16>> {
        ensure!(bytes.len().is_multiple_of(2), "BF16 fixture alignment");
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| bf16::from_bits(u16::from_le_bytes(v)))
            .collect())
    }
    fn tensor<T: Copy, const R: usize>(ptr: u64, shape: [u32; R]) -> Result<Tensor<T, R>> {
        let (stride, count) = dense_layout(shape)?;
        unsafe { Tensor::new(ptr, shape, stride, count * std::mem::size_of::<T>() as u64) }
    }
    #[test]
    fn gated_residual_stream_bounds() -> Result<()> {
        let v = GatedResidualViews {
            input: tensor(0x100000, [5, 4, 5120])?,
            projection: tensor(0x200000, [5, 25600])?,
            q_weight: tensor(0x300000, [4, 5120])?,
            k_weight: tensor(0x400000, [4, 5120])?,
            mask: tensor(0x500000, [5])?,
            output: tensor(0x600000, [5, 4, 5120])?,
            gates: tensor(0x700000, [5, 4])?,
            sequence: SeqRef { ptr: 0x800000 },
        };
        assert_eq!(geometry(v)?, ([5, 4, 5120], 20));
        assert!(
            geometry(GatedResidualViews {
                projection: tensor(0x200000, [5, 20480])?,
                ..v
            })
            .is_err()
        );
        assert!(
            geometry(GatedResidualViews {
                output: v.input,
                ..v
            })
            .is_err()
        );
        assert!(
            geometry(GatedResidualViews {
                q_weight: tensor(0x300000, [1, 20480])?,
                ..v
            })
            .is_err()
        );
        Ok(())
    }

    fn gate_cell(ctx: &Arc<CudaContext>, cell: &Cell, graph: bool) -> Result<bool> {
        let stream = ctx.new_stream()?;
        let rows = cell.rows;
        let host = bf16_values(&cell.x)?;
        let expected = bf16_values(&cell.output)?;
        ensure!(
            host.len() == rows * 20480
                && expected.len() == host.len()
                && cell.mask.len() == rows
                && cell.gates.len() == rows * 16,
            "gated residual reference extent"
        );
        let reference_gates: Vec<_> = cell
            .gates
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f32::from_le_bytes(v))
            .collect();
        let input = stream.clone_htod(&host)?;
        let projected = stream.clone_htod(&bf16_values(&cell.kv)?)?;
        let mut qw_host = bf16_values(&cell.q_weight)?;
        let mut qw = stream.clone_htod(&qw_host)?;
        let kw = stream.clone_htod(&bf16_values(&cell.k_weight)?)?;
        let mask = stream.clone_htod(&cell.mask)?;
        let sentinel = bf16::from_bits(0x42f7);
        let mut output = stream.clone_htod(&vec![sentinel; host.len() + 16])?;
        let mut gates = stream.clone_htod(&vec![12345.0f32; rows * 4 + 16])?;
        let mut metadata =
            stream.clone_htod(&[0, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
        let (ptr, record) = metadata.device_ptr(&stream);
        drop(record);
        let views = GatedResidualViews {
            input: view(&input, &stream, 0, [rows as u32, 4, 5120], [20480, 5120, 1])?,
            projection: view(&projected, &stream, 0, [rows as u32, 25600], [25600, 1])?,
            q_weight: view(&qw, &stream, 0, [4, 5120], [5120, 1])?,
            k_weight: view(&kw, &stream, 0, [4, 5120], [5120, 1])?,
            mask: view(&mask, &stream, 0, [rows as u32], [1])?,
            output: view(
                &output,
                &stream,
                0,
                [rows as u32, 4, 5120],
                [20480, 5120, 1],
            )?,
            gates: view(&gates, &stream, 0, [rows as u32, 4], [4, 1])?,
            sequence: SeqRef { ptr },
        };
        let plan = GatedResidualPlan::new(ctx, views, 1e-20, 1e-6)?;
        plan.run(&stream)?;
        let eager = (stream.clone_dtoh(&output)?, stream.clone_dtoh(&gates)?);
        let captured = if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            plan.run(&stream)?;
            Some(
                stream
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("gated residual graph empty")?,
            )
        } else {
            None
        };
        let execute = || -> Result<()> {
            if let Some(g) = &captured {
                g.launch()?;
            } else {
                plan.run(&stream)?;
            }
            Ok(())
        };
        let mut first: Option<(Vec<bf16>, Vec<f32>)> = None;
        for live in [rows, 1] {
            stream.memcpy_htod(
                &[0, live as i32, live as i32, rows as i32, 1, 0, 0, 0],
                &mut metadata,
            )?;
            stream.memcpy_htod(&vec![sentinel; host.len() + 16], &mut output)?;
            stream.memcpy_htod(&vec![12345.0f32; rows * 4 + 16], &mut gates)?;
            execute()?;
            let actual = stream.clone_dtoh(&output)?;
            let gate = stream.clone_dtoh(&gates)?;
            ensure!(
                actual[..live * 20480] == eager.0[..live * 20480]
                    && gate[..live * 4] == eager.1[..live * 4],
                "gated residual eager/graph bits DEFECT"
            );
            let gate_error = gate[..live * 4]
                .iter()
                .zip(&reference_gates)
                .map(|(&a, &b)| f64::from((a - b).abs()))
                .fold(0.0, f64::max);
            let output_error = actual[..live * 20480]
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a.to_f32()) - f64::from(b.to_f32())).abs())
                .fold(0.0, f64::max);
            ensure!(
                gate[..live * 4].iter().all(|v| v.is_finite())
                    && actual[..live * 20480].iter().all(|v| v.is_finite())
                    && gate_error <= GATE_LIMIT
                    && output_error <= OUTPUT_LIMIT,
                "gated residual DEFECT rows={rows} seed={} graph={graph} live={live} gate={gate_error} output={output_error}",
                cell.seed
            );
            for row in 0..live {
                if cell.mask[row] == 0 {
                    ensure!(
                        actual[row * 20480..(row + 1) * 20480]
                            == host[row * 20480..(row + 1) * 20480]
                            && gate[row * 4..(row + 1) * 4].iter().all(|&v| v == 0.0),
                        "masked residual DEFECT"
                    );
                }
            }
            ensure!(
                actual[live * 20480..].iter().all(|&v| v == sentinel)
                    && gate[live * 4..].iter().all(|&v| v == 12345.0),
                "gated residual live guard DEFECT"
            );
            if let Some((a, g)) = &first {
                ensure!(
                    actual[..20480] == a[..] && gate[..4] == g[..],
                    "gated residual live replay bits DEFECT"
                );
            } else {
                first = Some((actual[..20480].to_vec(), gate[..4].to_vec()));
            }
            println!(
                "GATED_RESIDUAL rows={rows} seed={} graph={graph} live={live} gate_max={gate_error} output_max={output_error}",
                cell.seed
            );
        }
        let control = rows == 1 && !graph;
        if control {
            qw_host.rotate_left(5120);
            stream.memcpy_htod(&qw_host, &mut qw)?;
            execute()?;
            ensure!(
                stream.clone_dtoh(&gates)?[..4]
                    .iter()
                    .zip(&reference_gates)
                    .any(|(&a, &b)| f64::from((a - b).abs()) > GATE_LIMIT),
                "wrong-copy weight control escaped"
            );
        }
        Ok(control)
    }

    #[test]
    #[ignore = "requires GPU lease and D1_F1_GATE_REFERENCE model-definition fixture"]
    fn normalized_gate_residual_gpu() -> Result<()> {
        let path =
            std::env::var("D1_F1_GATE_REFERENCE").context("D1_F1_GATE_REFERENCE required")?;
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture.reference_only && fixture.cells.len() == 8,
            "gated residual frozen reference matrix"
        );
        let ctx = CudaContext::new(0)?;
        let mut controls = 0;
        for cell in &fixture.cells {
            for graph in [false, true] {
                controls += usize::from(gate_cell(&ctx, cell, graph)?);
            }
        }
        ensure!(controls == 2, "gated residual control coverage");
        println!(
            "PASS NORMALIZED_GATE_RESIDUAL cases=16 masks_exact=true live_graph_bits=0 guards=true wrong_copy_controls=2 gate_limit={GATE_LIMIT} output_limit={OUTPUT_LIMIT}"
        );
        Ok(())
    }
}
