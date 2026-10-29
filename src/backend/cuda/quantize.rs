use super::{
    contracts::{SeqRef, Tensor, TensorMut, dense_layout, spans_disjoint},
    cubin,
    linear::{Fp8BlockPlan, Fp8Ops, Fp8ScaleRef, LinearWeightRef},
};
use crate::backend::gpu::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use anyhow::{Result, ensure};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct Fp4PackViews {
    pub input: Tensor<bf16, 2>,
    pub codes: Tensor<u8, 2>,
    pub scales: Tensor<u8, 2>,
    pub sequence: SeqRef,
}

#[derive(Clone, Copy)]
pub struct Fp8LookupViews {
    pub records: Tensor<u8, 3>,
    pub codes: Tensor<u8, 2>,
    pub scales: Tensor<u8, 2>,
    pub sequence: SeqRef,
}

/// Native lookup preparation followed by the shared block-scaled projection.
pub struct Fp8LookupProjection {
    prepare: Fp8LookupPlan,
    project: Fp8BlockPlan,
}

impl Fp8LookupProjection {
    pub fn new(
        ctx: &Arc<CudaContext>,
        lookup: Fp8LookupViews,
        weight: Tensor<u8, 2>,
        weight_scales: Tensor<u8, 2>,
        output: Tensor<bf16, 2>,
    ) -> Result<Self> {
        use crate::fp8::{Fp8BlockScale, Fp8ScaleLayout};
        lookup_geometry(lookup)?;
        let spans = [
            (lookup.records.ptr, lookup.records.span_bytes),
            (lookup.codes.ptr, lookup.codes.span_bytes),
            (lookup.scales.ptr, lookup.scales.span_bytes),
            (lookup.sequence.ptr, 32),
            (weight.ptr, weight.span_bytes),
            (weight_scales.ptr, weight_scales.span_bytes),
            (output.ptr, output.span_bytes),
        ];
        for (i, &(p, n)) in spans.iter().enumerate() {
            for &(q, m) in &spans[i + 1..] {
                ensure!(
                    spans_disjoint(p, n, q, m)?,
                    "lookup projection views overlap"
                );
            }
        }
        let layout =
            Fp8ScaleLayout::Block(Fp8BlockScale::new([32, 32], safetensors::Dtype::F8_E8M0)?);
        let project = Fp8Ops::new(ctx)?.bind_block(
            LinearWeightRef::Fp8 {
                codes: weight,
                scale: Fp8ScaleRef::E8m0(weight_scales),
                layout,
            },
            lookup.codes,
            Fp8ScaleRef::E8m0(lookup.scales),
            &TensorMut(output),
        )?;
        Ok(Self {
            prepare: Fp8LookupPlan::new(ctx, lookup)?,
            project,
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        self.prepare.run(stream)?;
        self.project.run(stream)
    }
}

fn lookup_geometry(v: Fp8LookupViews) -> Result<(i32, i32, i32, u32)> {
    v.records.validate()?;
    v.codes.validate()?;
    v.scales.validate()?;
    v.sequence.validate()?;
    let [rows, lookups, record_bytes] = v.records.shape;
    ensure!(
        record_bytes.is_multiple_of(33),
        "native lookup record extent"
    );
    let width = record_bytes / 33 * 32;
    let columns = u64::from(lookups) * u64::from(width);
    ensure!(
        rows <= i32::MAX as u32
            && lookups <= i32::MAX as u32
            && record_bytes <= i32::MAX as u32
            && columns <= u64::from(u32::MAX),
        "native lookup kernel extent"
    );
    ensure!(
        v.records.stride == dense_layout(v.records.shape)?.0,
        "native lookup records must be dense"
    );
    ensure!(
        v.codes.shape == [rows, columns as u32] && v.codes.stride == [columns, 1],
        "native lookup code extent"
    );
    ensure!(
        v.scales.shape == [rows, (columns / 32) as u32] && v.scales.stride == [columns / 32, 1],
        "native lookup scale extent"
    );
    let spans = [
        (v.records.ptr, v.records.span_bytes),
        (v.codes.ptr, v.codes.span_bytes),
        (v.scales.ptr, v.scales.span_bytes),
        (v.sequence.ptr, 32),
    ];
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "native lookup views overlap"
            );
        }
    }
    let grid = (u64::from(rows) * columns).div_ceil(256);
    ensure!(grid <= i32::MAX as u64, "native lookup launch extent");
    Ok((rows as i32, lookups as i32, width as i32, grid as u32))
}

/// Native lookup records decode in registers before shared activation preparation.
pub struct Fp8LookupPlan {
    kernel: CudaFunction,
    views: Fp8LookupViews,
    shape: [i32; 3],
    grid: LaunchConfig,
}

impl Fp8LookupPlan {
    pub fn new(ctx: &Arc<CudaContext>, views: Fp8LookupViews) -> Result<Self> {
        let (rows, lookups, width, blocks) = lookup_geometry(views)?;
        Ok(Self {
            kernel: cubin::module(ctx, "quantize")?.load_function("fp8_lookup_prepare")?,
            views,
            shape: [rows, lookups, width],
            grid: LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&self.views.records.ptr)
                .arg(&self.views.codes.ptr)
                .arg(&self.views.scales.ptr)
                .arg(&self.views.sequence.ptr)
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .launch(self.grid)?;
        }
        Ok(())
    }
}

fn geometry(v: Fp4PackViews, block: u32, group_rows: u32) -> Result<(i32, i32, u32)> {
    ensure!(matches!(block, 16 | 32), "FP4 scale block must be 16 or 32");
    ensure!(
        matches!(group_rows, 1 | 2),
        "FP4 input groups must be one or two rows"
    );
    v.input.validate()?;
    v.codes.validate()?;
    v.scales.validate()?;
    v.sequence.validate()?;
    let [rows, width] = v.input.shape;
    ensure!(
        rows <= i32::MAX as u32 && width <= i32::MAX as u32 && width.is_multiple_of(32),
        "FP4 input extent"
    );
    ensure!(
        v.input.stride == dense_layout(v.input.shape)?.0,
        "FP4 input must be dense"
    );
    ensure!(
        v.codes.shape == [rows, width / 2] && v.codes.stride == [u64::from(width / 2), 1],
        "FP4 code extent"
    );
    ensure!(
        v.scales.shape == [rows, width / block] && v.scales.stride == [u64::from(width / block), 1],
        "FP4 scale extent"
    );
    let spans = [
        (v.input.ptr, v.input.span_bytes),
        (v.codes.ptr, v.codes.span_bytes),
        (v.scales.ptr, v.scales.span_bytes),
        (v.sequence.ptr, 32),
    ];
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "FP4 views overlap"
            );
        }
    }
    let grid = (u64::from(rows) * u64::from(width)).div_ceil(256);
    ensure!(grid <= i32::MAX as u64, "FP4 launch extent");
    Ok((rows as i32, width as i32, grid as u32))
}

/// Block16 uses E4M3 scales; block32 uses E8M0 scales. Views bind at setup.
pub struct Fp4PackPlan {
    kernel: CudaFunction,
    views: Fp4PackViews,
    rows: i32,
    width: i32,
    group_rows: i32,
    grid: LaunchConfig,
}

impl Fp4PackPlan {
    pub fn new(
        ctx: &Arc<CudaContext>,
        views: Fp4PackViews,
        block: u32,
        group_rows: u32,
    ) -> Result<Self> {
        let (rows, width, blocks) = geometry(views, block, group_rows)?;
        let module = cubin::module(ctx, "quantize")?;
        Ok(Self {
            kernel: module.load_function(if block == 16 {
                "fp4_pack_e4m3_16"
            } else {
                "fp4_pack_e8m0_32"
            })?,
            views,
            rows,
            width,
            group_rows: group_rows as i32,
            grid: LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&self.views.input.ptr)
                .arg(&self.views.codes.ptr)
                .arg(&self.views.scales.ptr)
                .arg(&self.views.sequence.ptr)
                .arg(&self.rows)
                .arg(&self.width)
                .arg(&self.group_rows)
                .launch(self.grid)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::contracts::view;
    use super::*;
    use crate::backend::gpu::DevicePtr;
    use anyhow::Context;

    fn tensor<T: Copy>(ptr: u64, shape: [u32; 2]) -> Result<Tensor<T, 2>> {
        let (stride, count) = dense_layout(shape)?;
        unsafe { Tensor::new(ptr, shape, stride, count * std::mem::size_of::<T>() as u64) }
    }

    #[test]
    fn fp4_pack_extents_and_aliases() -> Result<()> {
        for block in [16, 32] {
            let v = Fp4PackViews {
                input: tensor(0x10000, [5, 512])?,
                codes: tensor(0x20000, [5, 256])?,
                scales: tensor(0x30000, [5, 512 / block])?,
                sequence: SeqRef { ptr: 0x40000 },
            };
            assert_eq!(geometry(v, block, 2)?, (5, 512, 10));
            assert!(
                geometry(
                    Fp4PackViews {
                        codes: tensor(0x10000, [5, 256])?,
                        ..v
                    },
                    block,
                    1
                )
                .is_err()
            );
            assert!(
                geometry(
                    Fp4PackViews {
                        scales: tensor(0x30000, [5, 512 / block + 1])?,
                        ..v
                    },
                    block,
                    1
                )
                .is_err()
            );
            assert!(geometry(v, block, 0).is_err());
        }
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct Bytes {
        bytes: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct Native {
        codes: Bytes,
        scales: Bytes,
    }
    #[derive(serde::Deserialize)]
    struct Cell {
        rows: usize,
        width: usize,
        block: u32,
        pattern: String,
        input: Bytes,
        runs: Vec<Native>,
    }
    #[derive(serde::Deserialize)]
    struct Fixture {
        reference_only: bool,
        cases: Vec<Cell>,
    }

    fn gate_cell(ctx: &Arc<CudaContext>, cell: &Cell, graph: bool, group: u32) -> Result<bool> {
        let stream = ctx.new_stream()?;
        let capacity = cell.rows + 1;
        ensure!(
            cell.input.bytes.len() == cell.rows * cell.width * 2 && !cell.runs.is_empty(),
            "FP4 fixture input extent"
        );
        let mut host: Vec<_> = cell
            .input
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
            .collect();
        host.resize(capacity * cell.width, bf16::from_f32(31.0));
        let mut input = stream.clone_htod(&host)?;
        let code_count = capacity * cell.width / 2;
        let scale_count = capacity * cell.width / cell.block as usize;
        let mut codes = stream.clone_htod(&vec![0xa5u8; code_count + 16])?;
        let mut scales = stream.clone_htod(&vec![0xa5u8; scale_count + 16])?;
        let prefix = i32::from(group == 2);
        let valid = cell.rows as i32 * group as i32 - prefix;
        let mut metadata = stream.clone_htod(&[
            prefix,
            valid,
            prefix + valid,
            2 * capacity as i32 + 4,
            1,
            0,
            0,
            0,
        ])?;
        let pointer = metadata.device_ptr(&stream).0;
        let views = Fp4PackViews {
            input: view(
                &input,
                &stream,
                0,
                [capacity as u32, cell.width as u32],
                [cell.width as u64, 1],
            )?,
            codes: view(
                &codes,
                &stream,
                0,
                [capacity as u32, cell.width as u32 / 2],
                [cell.width as u64 / 2, 1],
            )?,
            scales: view(
                &scales,
                &stream,
                0,
                [capacity as u32, cell.width as u32 / cell.block],
                [cell.width as u64 / u64::from(cell.block), 1],
            )?,
            sequence: SeqRef { ptr: pointer },
        };
        let plan = Fp4PackPlan::new(ctx, views, cell.block, group)?;
        let captured = if graph {
            Some(crate::backend::cuda::moe::testing::capture(
                &stream,
                || {
                    plan.run(&stream)?;
                    Ok(())
                },
            )?)
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
        execute()?;
        let c = stream.clone_dtoh(&codes)?;
        let s = stream.clone_dtoh(&scales)?;
        let reference = &cell.runs[0];
        ensure!(
            reference.codes.bytes.len() == cell.rows * cell.width / 2
                && reference.scales.bytes.len() == cell.rows * cell.width / cell.block as usize,
            "FP4 native reference extent"
        );
        ensure!(
            c[..reference.codes.bytes.len()] == reference.codes.bytes
                && s[..reference.scales.bytes.len()] == reference.scales.bytes,
            "FP4 native-byte DEFECT block={} rows={} width={} graph={} group={}",
            cell.block,
            cell.rows,
            cell.width,
            graph,
            group
        );
        ensure!(
            c[reference.codes.bytes.len()..]
                .iter()
                .chain(&s[reference.scales.bytes.len()..])
                .all(|&b| b == 0xa5),
            "FP4 output guard DEFECT"
        );
        if group == 2 {
            stream.memcpy_htod(
                &[0, 1, 1, 2 * capacity as i32 + 4, 1, 0, 0, 0],
                &mut metadata,
            )?;
            stream.memcpy_htod(&vec![0xa5u8; code_count + 16], &mut codes)?;
            stream.memcpy_htod(&vec![0xa5u8; scale_count + 16], &mut scales)?;
            execute()?;
            ensure!(
                stream
                    .clone_dtoh(&codes)?
                    .iter()
                    .chain(&stream.clone_dtoh(&scales)?)
                    .all(|&b| b == 0xa5),
                "FP4 incomplete-group publication DEFECT"
            );
        }
        let control = !graph
            && group == 1
            && cell.rows == 1
            && cell.width == 512
            && cell.pattern == "pattern";
        if control {
            let block = cell.block as usize;
            for lane in 0..block {
                host.swap(lane, lane + block);
            }
            stream.memcpy_htod(&host, &mut input)?;
            execute()?;
            let c = stream.clone_dtoh(&codes)?;
            let s = stream.clone_dtoh(&scales)?;
            ensure!(
                c[..reference.codes.bytes.len()] != reference.codes.bytes
                    || s[..reference.scales.bytes.len()] != reference.scales.bytes,
                "FP4 wrong-group control escaped"
            );
        }
        Ok(control)
    }

    #[test]
    #[ignore = "requires GPU lease and D1_F1_CODEC_REFERENCE native publisher fixture"]
    fn fp4_pack_gpu() -> Result<()> {
        let path =
            std::env::var("D1_F1_CODEC_REFERENCE").context("D1_F1_CODEC_REFERENCE required")?;
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture.reference_only && fixture.cases.len() == 40,
            "FP4 reference matrix requires 40 publisher cells"
        );
        let ctx = CudaContext::new(0)?;
        let mut cases = 0;
        let mut controls = 0;
        for cell in &fixture.cases {
            for graph in [false, true] {
                for group in [1, 2] {
                    controls += usize::from(gate_cell(&ctx, cell, graph, group)?);
                    cases += 1;
                }
            }
        }
        ensure!(controls == 2, "FP4 control coverage");
        println!(
            "PASS FP4_PACK cases={cases} native_bits=0 guards=true graph_replay=true incomplete_group=true wrong_group_controls={controls}"
        );
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct LookupCell {
        rows: usize,
        records: Vec<u8>,
        decoded: Vec<u8>,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct LookupFixture {
        reference_only: bool,
        cells: Vec<LookupCell>,
    }

    #[test]
    fn native_lookup_bounds() -> Result<()> {
        let (stride, count) = dense_layout([5, 24, 264])?;
        let v = Fp8LookupViews {
            records: unsafe { Tensor::new(0x100000, [5, 24, 264], stride, count)? },
            codes: tensor(0x200000, [5, 6144])?,
            scales: tensor(0x300000, [5, 192])?,
            sequence: SeqRef { ptr: 0x400000 },
        };
        assert_eq!(lookup_geometry(v)?, (5, 24, 256, 120));
        assert!(
            lookup_geometry(Fp8LookupViews {
                codes: tensor(0x100000, [5, 6144])?,
                ..v
            })
            .is_err()
        );
        assert!(
            lookup_geometry(Fp8LookupViews {
                scales: tensor(0x300000, [5, 191])?,
                ..v
            })
            .is_err()
        );
        Ok(())
    }

    fn lookup_cell(ctx: &Arc<CudaContext>, c: &LookupCell, graph: bool) -> Result<bool> {
        ensure!(
            c.records.len() == c.rows * 6336
                && c.decoded.len() == c.rows * 12288
                && c.codes.len() == c.rows * 6144
                && c.scales.len() == c.rows * 192,
            "lookup fixture extent"
        );
        let stream = ctx.new_stream()?;
        let capacity = c.rows + 1;
        let mut host = c.records.clone();
        host.resize(capacity * 6336, 0);
        let mut records = stream.clone_htod(&host)?;
        let mut codes = stream.clone_htod(&vec![0xa5u8; capacity * 6144 + 16])?;
        let mut scales = stream.clone_htod(&vec![0xa5u8; capacity * 192 + 16])?;
        let decoded = stream.clone_htod(&vec![bf16::from_bits(0x42f6); capacity * 6144 + 8])?;
        let mut metadata =
            stream.clone_htod(&[0, c.rows as i32, c.rows as i32, capacity as i32, 1, 0, 0, 0])?;
        let sequence = metadata.device_ptr(&stream).0;
        let views = Fp8LookupViews {
            records: view(
                &records,
                &stream,
                0,
                [capacity as u32, 24, 264],
                [6336, 264, 1],
            )?,
            codes: view(&codes, &stream, 0, [capacity as u32, 6144], [6144, 1])?,
            scales: view(&scales, &stream, 0, [capacity as u32, 192], [192, 1])?,
            sequence: SeqRef { ptr: sequence },
        };
        let plan = Fp8LookupPlan::new(ctx, views)?;
        let decode = cubin::module(ctx, "quantize")?.load_function("fp8_lookup_decode")?;
        let decode_ptr = decoded.device_ptr(&stream).0;
        unsafe {
            stream
                .launch_builder(&decode)
                .arg(&views.records.ptr)
                .arg(&decode_ptr)
                .arg(&sequence)
                .arg(&plan.shape[0])
                .arg(&plan.shape[1])
                .arg(&plan.shape[2])
                .launch(plan.grid)?;
        }
        let got = stream.clone_dtoh(&decoded)?;
        for (value, pair) in got.iter().zip(c.decoded.as_chunks::<2>().0) {
            ensure!(
                value.to_bits() == u16::from_le_bytes(*pair),
                "native lookup decode-bit DEFECT"
            );
        }
        ensure!(
            got[c.rows * 6144..].iter().all(|v| v.to_bits() == 0x42f6),
            "lookup decode guard DEFECT"
        );
        let captured = if graph {
            Some(crate::backend::cuda::moe::testing::capture(
                &stream,
                || {
                    plan.run(&stream)?;
                    Ok(())
                },
            )?)
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
        for live in [c.rows, 1] {
            stream.memcpy_htod(
                &[0, live as i32, live as i32, capacity as i32, 1, 0, 0, 0],
                &mut metadata,
            )?;
            stream.memcpy_htod(&vec![0xa5u8; capacity * 6144 + 16], &mut codes)?;
            stream.memcpy_htod(&vec![0xa5u8; capacity * 192 + 16], &mut scales)?;
            execute()?;
            let got = stream.clone_dtoh(&codes)?;
            let scale = stream.clone_dtoh(&scales)?;
            ensure!(
                got[..live * 6144] == c.codes[..live * 6144]
                    && scale[..live * 192] == c.scales[..live * 192],
                "native lookup prepared-bit DEFECT"
            );
            ensure!(
                got[live * 6144..]
                    .iter()
                    .chain(&scale[live * 192..])
                    .all(|&b| b == 0xa5),
                "native lookup active-row guard DEFECT"
            );
        }
        let control = c.rows == 1 && !graph;
        if control {
            host[256 + 7] += 1;
            stream.memcpy_htod(&host, &mut records)?;
            execute()?;
            let scale = stream.clone_dtoh(&scales)?;
            ensure!(
                scale[7] != c.scales[7],
                "native lookup wrong-scale control escaped"
            );
            ensure!(
                scale[..7] == c.scales[..7] && scale[8..192] == c.scales[8..192],
                "native lookup wrong-scale control changed another group"
            );
        }
        Ok(control)
    }

    #[test]
    #[ignore = "requires GPU lease and D1_F1_LOOKUP_REFERENCE model-definition fixture"]
    fn fp8_lookup_prepare_gpu() -> Result<()> {
        let path =
            std::env::var("D1_F1_LOOKUP_REFERENCE").context("D1_F1_LOOKUP_REFERENCE required")?;
        let fixture: LookupFixture = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture.reference_only && fixture.cells.len() == 8,
            "lookup reference matrix"
        );
        let ctx = CudaContext::new(0)?;
        let mut controls = 0;
        for c in &fixture.cells {
            for graph in [false, true] {
                controls += usize::from(lookup_cell(&ctx, c, graph)?);
            }
        }
        ensure!(controls == 2, "lookup control coverage");
        println!(
            "PASS FP8_LOOKUP_PREPARE cases=16 decode_bits=0 prepared_bits=0 live_graph_replay=true guards=true wrong_scale_controls={controls}"
        );
        Ok(())
    }

    fn projection_residual(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        cell: &LookupCell,
        projection: &Fp8LookupProjection,
        projected: &crate::backend::gpu::CudaSlice<bf16>,
        records: &mut crate::backend::gpu::CudaSlice<u8>,
        sequence: SeqRef,
    ) -> Result<()> {
        use super::super::gated_residual::{GatedResidualPlan, GatedResidualViews};
        const WIDTH: usize = 5120;
        // 2026-10-28 score-gate-freeze-r1, unchanged from the isolated residual gate.
        const GATE_LIMIT: f64 = 5.408762987491045e-7;
        let count = cell.rows * 4 * WIDTH;
        let x: Vec<_> = (0..count)
            .map(|i| bf16::from_f32(((i * 37 + 11) % 251) as f32 / 64.0 - 1.953125))
            .collect();
        let qw: Vec<_> = (0..4 * WIDTH)
            .map(|i| bf16::from_f32(0.5 + ((i * 17 + 3) % 31) as f32 / 32.0))
            .collect();
        let kw: Vec<_> = (0..4 * WIDTH)
            .map(|i| bf16::from_f32(0.5 + ((i * 29 + 7) % 47) as f32 / 64.0))
            .collect();
        let mask: Vec<_> = (0..cell.rows).map(|r| u8::from(r % 3 != 1)).collect();
        let input = stream.clone_htod(&x)?;
        let q_weight = stream.clone_htod(&qw)?;
        let k_weight = stream.clone_htod(&kw)?;
        let device_mask = stream.clone_htod(&mask)?;
        let sentinel = bf16::from_bits(0x42f7);
        let out = stream.clone_htod(&vec![sentinel; count + 16])?;
        let gates = stream.clone_htod(&vec![127.0f32; cell.rows * 4 + 16])?;
        let stream_shape = [cell.rows as u32, 4, WIDTH as u32];
        let stream_stride = [(4 * WIDTH) as u64, WIDTH as u64, 1];
        let residual = GatedResidualPlan::new(
            ctx,
            GatedResidualViews {
                input: view(&input, stream, 0, stream_shape, stream_stride)?,
                projection: view(projected, stream, 0, [cell.rows as u32, 25600], [25600, 1])?,
                q_weight: view(&q_weight, stream, 0, [4, WIDTH as u32], [WIDTH as u64, 1])?,
                k_weight: view(&k_weight, stream, 0, [4, WIDTH as u32], [WIDTH as u64, 1])?,
                mask: view(&device_mask, stream, 0, [cell.rows as u32], [1])?,
                output: view(&out, stream, 0, stream_shape, stream_stride)?,
                gates: view(&gates, stream, 0, [cell.rows as u32, 4], [4, 1])?,
                sequence,
            },
            1e-20,
            1e-6,
        )?;
        projection.run(stream)?;
        residual.run(stream)?;
        let normal = stream.clone_dtoh(&out)?;
        let normal_gates = stream.clone_dtoh(&gates)?;
        let kv = stream.clone_dtoh(projected)?;
        crate::backend::cuda::moe::testing::capture(stream, || {
            projection.run(stream)?;
            residual.run(stream)?;
            Ok(())
        })?
        .launch()?;
        ensure!(
            stream.clone_dtoh(&out)? == normal && stream.clone_dtoh(&gates)? == normal_gates,
            "lookup residual composed graph/eager bits DEFECT"
        );
        ensure!(
            normal[count..].iter().all(|&v| v == sentinel)
                && normal_gates[cell.rows * 4..].iter().all(|&v| v == 127.0),
            "lookup residual composed guards DEFECT"
        );
        let mut oracle = vec![0.0f64; count];
        let mut output_bounds = vec![0.0f64; count];
        let mut gate_error = 0.0f64;
        let mut output_error = 0.0f64;
        for row in 0..cell.rows {
            for copy in 0..4 {
                let base = (row * 4 + copy) * WIDTH;
                let key = row * 25600 + copy * WIDTH;
                let mut h2 = 0.0f64;
                let mut k2 = 0.0f64;
                let mut dot = 0.0f64;
                for d in 0..WIDTH {
                    let h = f64::from(x[base + d].to_f32());
                    let k = f64::from(kv[key + d].to_f32());
                    h2 += h * h;
                    k2 += k * k;
                    dot += h
                        * k
                        * f64::from(qw[copy * WIDTH + d].to_f32())
                        * f64::from(kw[copy * WIDTH + d].to_f32());
                }
                let normalized = dot
                    / ((h2 / WIDTH as f64 + 1e-20) * (k2 / WIDTH as f64 + 1e-20)).sqrt()
                    / (WIDTH as f64).sqrt();
                let signed_root = normalized.abs().max(1e-6).sqrt().copysign(normalized);
                let expected_gate = if mask[row] == 0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-signed_root).exp())
                };
                gate_error =
                    gate_error.max((f64::from(normal_gates[row * 4 + copy]) - expected_gate).abs());
                for d in 0..WIDTH {
                    let index = base + d;
                    let h = f64::from(x[index].to_f32());
                    let value = f64::from(kv[row * 25600 + 4 * WIDTH + d].to_f32());
                    oracle[index] = h + expected_gate * value;
                    // Frozen gate error, two FP32 operations and the final BF16 rounding.
                    output_bounds[index] = GATE_LIMIT * value.abs()
                        + oracle[index].abs() / 256.0
                        + 3.0 * f64::from(f32::EPSILON) * (h.abs() + value.abs())
                        + 1e-30;
                    ensure!(
                        (f64::from(normal[index].to_f32()) - oracle[index]).abs()
                            <= output_bounds[index],
                        "lookup residual propagated error DEFECT at {index}"
                    );
                    output_error =
                        output_error.max((f64::from(normal[index].to_f32()) - oracle[index]).abs());
                    ensure!(
                        normal[index].is_finite() && (mask[row] != 0 || normal[index] == x[index]),
                        "lookup residual mask/finite DEFECT"
                    );
                }
            }
        }
        ensure!(
            gate_error <= GATE_LIMIT,
            "lookup residual composed DEFECT gate_error={gate_error} output_error={output_error}"
        );
        let mut reordered = cell.records.clone();
        for row in reordered.as_chunks_mut::<{ 24 * 264 }>().0 {
            row.rotate_left(13 * 264);
        }
        stream.memcpy_htod(&reordered, records)?;
        projection.run(stream)?;
        residual.run(stream)?;
        let wrong = stream.clone_dtoh(&out)?;
        ensure!(
            wrong[..count]
                .iter()
                .zip(&oracle)
                .zip(&output_bounds)
                .any(|((a, b), bound)| (f64::from(a.to_f32()) - b).abs() > *bound),
            "lookup residual wrong-native-row control escaped"
        );
        stream.memcpy_htod(&cell.records, records)?;
        projection.run(stream)?;
        println!(
            "LOOKUP_RESIDUAL rows={} fp64_from_projected_boundary=true gate_max={gate_error} output_max={output_error} graph_bits=0 masks_exact=true wrong_native_row_control=true",
            cell.rows
        );
        Ok(())
    }

    fn projection_cell(ctx: &Arc<CudaContext>, cell: &LookupCell) -> Result<()> {
        const K: usize = 6144;
        const N: usize = 25600;
        let stream = ctx.new_stream()?;
        let table = [0x38u8, 0xb8, 0x30, 0xb0, 0x40, 0xc0, 0];
        let weight: Vec<_> = (0..N * K)
            .map(|i| table[((i / K * 17 + i % K * 29) % 251) % 7])
            .collect();
        let scales: Vec<_> = (0..N / 32 * (K / 32))
            .map(|i| (123 + (i / (K / 32) * 7 + i % (K / 32) * 11) % 5) as u8)
            .collect();
        let w = stream.clone_htod(&weight)?;
        let mut ws = stream.clone_htod(&scales)?;
        let mut records = stream.clone_htod(&cell.records)?;
        let codes = stream.alloc_zeros::<u8>(cell.rows * K)?;
        let prepared_scales = stream.alloc_zeros::<u8>(cell.rows * K / 32)?;
        let sentinel = bf16::from_bits(0x42f7);
        let output = stream.clone_htod(&vec![sentinel; cell.rows * N + 16])?;
        let sequence = stream.clone_htod(&[
            0,
            cell.rows as i32,
            cell.rows as i32,
            cell.rows as i32,
            1,
            0,
            0,
            0,
        ])?;
        let ptr = sequence.device_ptr(&stream).0;
        let lookup = Fp8LookupViews {
            records: view(
                &records,
                &stream,
                0,
                [cell.rows as u32, 24, 264],
                [6336, 264, 1],
            )?,
            codes: view(
                &codes,
                &stream,
                0,
                [cell.rows as u32, K as u32],
                [K as u64, 1],
            )?,
            scales: view(
                &prepared_scales,
                &stream,
                0,
                [cell.rows as u32, K as u32 / 32],
                [(K / 32) as u64, 1],
            )?,
            sequence: SeqRef { ptr },
        };
        let plan = Fp8LookupProjection::new(
            ctx,
            lookup,
            view(&w, &stream, 0, [N as u32, K as u32], [K as u64, 1])?,
            view(
                &ws,
                &stream,
                0,
                [N as u32 / 32, K as u32 / 32],
                [(K / 32) as u64, 1],
            )?,
            view(
                &output,
                &stream,
                0,
                [cell.rows as u32, N as u32],
                [N as u64, 1],
            )?,
        )?;
        plan.run(&stream)?;
        let eager = stream.clone_dtoh(&output)?;
        ensure!(
            stream.clone_dtoh(&codes)? == cell.codes
                && stream.clone_dtoh(&prepared_scales)? == cell.scales,
            "lookup projection preparation DEFECT"
        );
        crate::backend::cuda::moe::testing::capture(&stream, || {
            plan.run(&stream)?;
            Ok(())
        })?
        .launch()?;
        ensure!(
            stream.clone_dtoh(&output)? == eager,
            "lookup projection graph/eager bits DEFECT"
        );
        ensure!(
            eager[..cell.rows * N].iter().all(|v| v.is_finite())
                && eager[cell.rows * N..].iter().all(|&v| v == sentinel),
            "lookup projection finite/guard DEFECT"
        );
        let mut oracle = Vec::new();
        for row in 0..cell.rows {
            for column in [0, 31, 32, 5119, 5120, 8191, 8192, 20479, 20480, 25599] {
                let mut sum = 0.0f64;
                let mut absolute = 0.0f64;
                for inner in 0..K {
                    let a = f64::from(crate::fp8::decode_e4m3(cell.codes[row * K + inner]))
                        * 2.0f64.powi(i32::from(cell.scales[row * (K / 32) + inner / 32]) - 127);
                    let b = f64::from(crate::fp8::decode_e4m3(weight[column * K + inner]))
                        * 2.0f64.powi(i32::from(scales[column / 32 * (K / 32) + inner / 32]) - 127);
                    let product = a * b;
                    sum += product;
                    absolute += product.abs();
                }
                // Shared native-linear bound: FP32 gamma_(K+2G), BF16 output rounding, FP64 roundoff.
                let terms = (K + 2 * (K / 32)) as f64 * f64::from(f32::EPSILON);
                let bound = terms / (1.0 - terms) * absolute + sum.abs() / 256.0 + 1e-30;
                let index = row * N + column;
                ensure!(
                    (f64::from(eager[index].to_f32()) - sum).abs() <= bound,
                    "lookup native projection DEFECT rows={} row={row} column={column} actual={} reference={sum} bound={bound}",
                    cell.rows,
                    eager[index]
                );
                oracle.push((index, sum, bound));
            }
        }
        projection_residual(
            ctx,
            &stream,
            cell,
            &plan,
            &output,
            &mut records,
            lookup.sequence,
        )?;
        let wrong: Vec<_> = scales.iter().map(|&v| v + 1).collect();
        stream.memcpy_htod(&wrong, &mut ws)?;
        plan.run(&stream)?;
        let mutant = stream.clone_dtoh(&output)?;
        ensure!(
            oracle.iter().any(
                |&(i, reference, bound)| (f64::from(mutant[i].to_f32()) - reference).abs() > bound
            ),
            "lookup projection scale control escaped"
        );
        println!(
            "LOOKUP_PROJECTION rows={} weight_codes=[25600,6144] scales=[800,192] fp64_samples={} graph_bits=0 scale_control=true",
            cell.rows,
            oracle.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires GPU lease and D1_F1_LOOKUP_REFERENCE; production-shape native projection"]
    fn fp8_lookup_projection_gpu() -> Result<()> {
        let path =
            std::env::var("D1_F1_LOOKUP_REFERENCE").context("D1_F1_LOOKUP_REFERENCE required")?;
        let fixture: LookupFixture = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture.reference_only && fixture.cells.len() == 8,
            "lookup projection reference matrix"
        );
        let ctx = CudaContext::new(0)?;
        let mut cells = 0;
        for row in [1, 17] {
            let cell = fixture
                .cells
                .iter()
                .find(|c| c.rows == row)
                .context("lookup production row cell missing")?;
            projection_cell(&ctx, cell)?;
            cells += 1;
        }
        println!(
            "PASS FP8_LOOKUP_PROJECTION cases={cells} weight_bytes=157440000 fp64_sampled=true all_outputs_finite=true graph_bits=0 guards=true wrong_scale_controls=2 composed_residual_cells=2 wrong_native_row_controls=2"
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires GPU lease and D1_OP_REFERENCE recorded compressor/indexer quantization operands"]
    fn compressed_state_codec_publisher_gpu() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Codec {
            owner: usize,
            rows: usize,
            width: usize,
            block: u32,
            input: Vec<u16>,
            codes: Vec<u8>,
            scales: Vec<u8>,
        }
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_OP_REFERENCE")?)?)?;
        ensure!(
            root["reference_only"] == true && root["candidate_seen"] == false,
            "codec record provenance"
        );
        let mut records: Vec<Codec> = serde_json::from_value(root["codecs"].clone())?;
        ensure!(records.len() == 4, "compressed KV owner coverage");
        for index in root["indexes"].as_array().context("index records")? {
            let owner = index["owner"].as_u64().context("owner")? as usize;
            for (input, codes, scales) in [
                ("query_input", "query", "query_scales"),
                ("key_input", "keys", "key_scales"),
            ] {
                let input: Vec<u16> = serde_json::from_value(index[input].clone())?;
                records.push(Codec {
                    owner,
                    rows: input.len() / 128,
                    width: 128,
                    block: 32,
                    input,
                    codes: serde_json::from_value(index[codes].clone())?,
                    scales: serde_json::from_value(index[scales].clone())?,
                });
            }
        }
        let ctx = CudaContext::new(0)?;
        let (mut cases, mut defects, mut controls) = (0, 0, 0);
        for record in records {
            let cell = Cell {
                rows: record.rows,
                width: record.width,
                block: record.block,
                pattern: "recorded".into(),
                input: Bytes {
                    bytes: record.input.iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
                runs: vec![Native {
                    codes: Bytes {
                        bytes: record.codes,
                    },
                    scales: Bytes {
                        bytes: record.scales,
                    },
                }],
            };
            for graph in [false, true] {
                match gate_cell(&ctx, &cell, graph, 1) {
                    Ok(_) => {}
                    Err(e) if format!("{e:#}").contains("DEFECT") => {
                        defects += 1;
                        println!(
                            "DEFECT CODEC_OWNER owner={} block={} graph={graph} error={e:#}",
                            record.owner, record.block
                        );
                    }
                    Err(e) => return Err(e),
                }
                cases += 1;
            }
            if record.width == 512 {
                let first = Cell {
                    rows: 1,
                    width: 512,
                    block: record.block,
                    pattern: "pattern".into(),
                    input: Bytes {
                        bytes: cell.input.bytes[..1024].to_vec(),
                    },
                    runs: vec![Native {
                        codes: Bytes {
                            bytes: cell.runs[0].codes.bytes[..256].to_vec(),
                        },
                        scales: Bytes {
                            bytes: cell.runs[0].scales.bytes[..512 / record.block as usize]
                                .to_vec(),
                        },
                    }],
                };
                controls += usize::from(gate_cell(&ctx, &first, false, 1)?);
            }
        }
        println!(
            "{} COMPRESSED_STATE_CODEC cases={cases} defects={defects} wrong_group_controls={controls}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(
            defects == 0 && controls == 4,
            "recorded compressed-state codec DEFECT"
        );
        Ok(())
    }
}
