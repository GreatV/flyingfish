use super::{
    contracts::{SeqRef, Tensor, TensorMut, dense_layout, spans_disjoint},
    cubin,
    selection::{DsaSelectionPlan, DsaTopKInput, DsaTopKOutput},
};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct IndexScoreViews {
    pub query: Tensor<u8, 3>,
    pub query_scales: Tensor<u8, 3>,
    pub keys: Tensor<u8, 2>,
    pub key_scales: Tensor<u8, 2>,
    pub weights: Tensor<bf16, 2>,
    pub committed: Tensor<u32, 1>,
    pub scores: Tensor<f32, 2>,
    pub positions: Tensor<i32, 2>,
    pub sequence: SeqRef,
    pub candidates: Option<Tensor<i32, 2>>,
}

fn dense<T: Copy, const R: usize>(v: Tensor<T, R>) -> Result<(u64, u64)> {
    v.validate()?;
    ensure!(
        v.stride == dense_layout(v.shape)?.0,
        "index score views must be dense"
    );
    Ok((v.ptr, v.span_bytes))
}

fn geometry(v: IndexScoreViews, ratio: u32) -> Result<([i32; 4], u32)> {
    ensure!(matches!(ratio, 1 | 2), "index score compression ratio");
    v.sequence.validate()?;
    let mut spans = vec![
        dense(v.query)?,
        dense(v.query_scales)?,
        dense(v.keys)?,
        dense(v.key_scales)?,
        dense(v.weights)?,
        dense(v.committed)?,
        dense(v.scores)?,
        dense(v.positions)?,
        (v.sequence.ptr, 32),
    ];
    let [rows, heads, packed] = v.query.shape;
    let capacity = v.keys.shape[0];
    ensure!(
        packed == 64
            && v.keys.shape[1] == 64
            && heads <= i32::MAX as u32
            && rows <= i32::MAX as u32
            && capacity <= (i32::MAX - 3) as u32,
        "index score requires stored D128 E2M1 geometry"
    );
    ensure!(
        v.query_scales.shape == [rows, heads, 4] && v.key_scales.shape == [capacity, 4],
        "index E8M0 scale geometry"
    );
    ensure!(
        v.weights.shape == [rows, heads] && v.committed.shape == [1],
        "index weight/count geometry"
    );
    let columns = if let Some(candidates) = v.candidates {
        spans.push(dense(candidates)?);
        ensure!(
            candidates.shape == [rows, 2048],
            "index candidate-block geometry"
        );
        16384
    } else {
        capacity
    };
    ensure!(
        v.scores.shape == [rows, columns] && v.positions.shape == [rows, columns],
        "index score/output geometry"
    );
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "index score views overlap"
            );
        }
    }
    let blocks = u64::from(rows) * u64::from(columns.div_ceil(4));
    ensure!(blocks <= i32::MAX as u64, "index score grid extent");
    Ok((
        [rows as i32, heads as i32, capacity as i32, columns as i32],
        blocks as u32,
    ))
}

/// Native E2M1/E8M0 scoring; raw BF16 head weights are scaled at the model boundary.
pub struct IndexScorePlan {
    kernel: CudaFunction,
    views: IndexScoreViews,
    shape: [i32; 4],
    ratio: i32,
    weight_scale: f32,
    grid: LaunchConfig,
}

impl IndexScorePlan {
    pub fn new(ctx: &Arc<CudaContext>, views: IndexScoreViews, ratio: u32) -> Result<Self> {
        let (shape, blocks) = geometry(views, ratio)?;
        Ok(Self {
            kernel: cubin::module(ctx, "index_score")?.load_function(
                if views.candidates.is_some() {
                    "index_score_fp4_128_candidates"
                } else {
                    "index_score_fp4_128"
                },
            )?,
            views,
            shape,
            ratio: ratio as i32,
            weight_scale: (128.0f64 * f64::from(shape[1])).sqrt().recip() as f32,
            grid: LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let v = self.views;
        let candidates = v.candidates.map_or(0, |v| v.ptr);
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&v.query.ptr)
                .arg(&v.query_scales.ptr)
                .arg(&v.keys.ptr)
                .arg(&v.key_scales.ptr)
                .arg(&v.weights.ptr)
                .arg(&v.sequence.ptr)
                .arg(&v.committed.ptr)
                .arg(&candidates)
                .arg(&v.scores.ptr)
                .arg(&v.positions.ptr)
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.shape[2])
                .arg(&self.shape[3])
                .arg(&self.ratio)
                .arg(&self.weight_scale)
                .launch(self.grid)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct IndexCandidatesViews {
    pub scores: Tensor<f32, 2>,
    pub block_scores: Tensor<f32, 2>,
    pub block_ids: Tensor<i32, 2>,
    pub selected_scores: Tensor<f32, 2>,
    pub selected_ids: Tensor<i32, 2>,
    pub margins: Tensor<f32, 1>,
    pub candidates: Tensor<i32, 2>,
    pub sequence: SeqRef,
}

fn candidate_geometry(v: IndexCandidatesViews, ratio: u32) -> Result<([i32; 2], [u32; 2])> {
    ensure!(matches!(ratio, 1 | 2), "candidate compression ratio");
    v.sequence.validate()?;
    let spans = [
        dense(v.scores)?,
        dense(v.block_scores)?,
        dense(v.block_ids)?,
        dense(v.selected_scores)?,
        dense(v.selected_ids)?,
        dense(v.margins)?,
        dense(v.candidates)?,
        (v.sequence.ptr, 32),
    ];
    let [rows, capacity] = v.scores.shape;
    ensure!(
        rows <= i32::MAX as u32 && capacity <= (i32::MAX - 7) as u32,
        "candidate kernel extent"
    );
    let blocks = capacity.div_ceil(8);
    ensure!(
        v.block_scores.shape == [rows, blocks] && v.block_ids.shape == [rows, blocks],
        "candidate block geometry"
    );
    ensure!(
        v.selected_scores.shape == [rows, 2047]
            && v.selected_ids.shape == [rows, 2047]
            && v.margins.shape == [rows]
            && v.candidates.shape == [rows, 2048],
        "candidate selection geometry"
    );
    for (index, &(ptr, bytes)) in spans.iter().enumerate() {
        for &(other, count) in &spans[index + 1..] {
            ensure!(
                spans_disjoint(ptr, bytes, other, count)?,
                "candidate views overlap"
            );
        }
    }
    let grid = [
        (u64::from(rows) * u64::from(blocks)).div_ceil(256),
        u64::from(rows) * 8,
    ];
    ensure!(
        grid.iter().all(|&v| v <= i32::MAX as u64),
        "candidate launch extent"
    );
    Ok(([rows as i32, capacity as i32], grid.map(|v| v as u32)))
}

/// The shared selector keeps 2047 older blocks; the newest visible block is mandatory.
pub struct IndexCandidatesPlan {
    prepare: CudaFunction,
    publish: CudaFunction,
    selector: DsaSelectionPlan,
    views: IndexCandidatesViews,
    shape: [i32; 2],
    ratio: i32,
    grids: [LaunchConfig; 2],
}

impl IndexCandidatesPlan {
    pub fn new(ctx: &Arc<CudaContext>, views: IndexCandidatesViews, ratio: u32) -> Result<Self> {
        let (shape, grids) = candidate_geometry(views, ratio)?;
        let module = cubin::module(ctx, "index_score")?;
        Ok(Self {
            prepare: module.load_function("index_block_scores")?,
            publish: module.load_function("index_block_publish")?,
            selector: DsaSelectionPlan::new(
                ctx,
                shape[0] as u32,
                (shape[1] as u32).div_ceil(8),
                2047,
            )?,
            views,
            shape,
            ratio: ratio as i32,
            grids: grids.map(|x| LaunchConfig {
                grid_dim: (x, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }),
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let v = self.views;
        unsafe {
            stream
                .launch_builder(&self.prepare)
                .arg(&v.scores.ptr)
                .arg(&v.block_scores.ptr)
                .arg(&v.block_ids.ptr)
                .arg(&v.sequence.ptr)
                .arg(&self.shape[0])
                .arg(&self.shape[1])
                .arg(&self.ratio)
                .launch(self.grids[0])?;
        }
        self.selector.select_scores(
            &DsaTopKInput {
                scores: v.block_scores,
                ids: v.block_ids,
            },
            &DsaTopKOutput {
                scores: TensorMut(v.selected_scores),
                ids: TensorMut(v.selected_ids),
                margins: TensorMut(v.margins),
            },
            stream,
        )?;
        unsafe {
            stream
                .launch_builder(&self.publish)
                .arg(&v.selected_ids.ptr)
                .arg(&v.candidates.ptr)
                .arg(&v.sequence.ptr)
                .arg(&self.shape[0])
                .arg(&self.ratio)
                .launch(self.grids[1])?;
        }
        Ok(())
    }
}

pub struct IndexPositionOrder {
    kernel: CudaFunction,
    input: Tensor<i32, 2>,
    output: Tensor<i32, 2>,
    grid: LaunchConfig,
}

impl IndexPositionOrder {
    pub fn new(
        ctx: &Arc<CudaContext>,
        input: Tensor<i32, 2>,
        output: Tensor<i32, 2>,
    ) -> Result<Self> {
        dense(input)?;
        dense(output)?;
        ensure!(
            input.shape == output.shape
                && input.shape[1] == 512
                && input.shape[0] <= i32::MAX as u32,
            "index position order geometry"
        );
        ensure!(
            spans_disjoint(input.ptr, input.span_bytes, output.ptr, output.span_bytes)?,
            "index position order overlap"
        );
        Ok(Self {
            kernel: cubin::module(ctx, "index_score")?.load_function("index_position_order")?,
            input,
            output,
            grid: LaunchConfig {
                grid_dim: (input.shape[0], 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }

    pub fn run(&self, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(&self.input.ptr)
                .arg(&self.output.ptr)
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

    // 2026-10-28: 3 * 0.0078125 reference shape spread; sm86/sm89 repeat/cross spread is zero.
    const SCORE_LIMIT: f64 = 0.0234375;

    #[derive(serde::Deserialize)]
    struct ScoreBytes {
        bytes: Vec<u8>,
    }
    #[derive(serde::Deserialize)]
    struct ScoreCell {
        rows: usize,
        positions: usize,
        seed: usize,
        repeat: usize,
        scores: ScoreBytes,
    }
    #[derive(serde::Deserialize)]
    struct ScoreFixture {
        reference_only: bool,
        cells: Vec<ScoreCell>,
    }

    fn native_pattern(elements: usize, multiplier: usize, seed: usize) -> (Vec<u8>, Vec<u8>) {
        let codes: Vec<u8> = (0..elements)
            .map(|i| (((i * multiplier + seed) % 251) % 16) as u8)
            .collect();
        let packed = codes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| v[0] | (v[1] << 4))
            .collect();
        let scales = (0..elements / 32)
            .map(|i| (((i * 7 + seed) % 5) + 123) as u8)
            .collect();
        (packed, scales)
    }

    fn score_cell(
        ctx: &Arc<CudaContext>,
        cell: &ScoreCell,
        mode: [bool; 2],
        ratio: usize,
    ) -> Result<bool> {
        let [graph, restrict] = mode;
        let stream = ctx.new_stream()?;
        let (rows, capacity) = (cell.rows, cell.positions);
        let columns = if restrict { 16384 } else { capacity };
        let (q, mut qs) = native_pattern(rows * 32 * 128, 37, cell.seed);
        let (k, ks) = native_pattern(capacity * 128, 29, cell.seed + 7);
        let query = stream.clone_htod(&q)?;
        let mut query_scales = stream.clone_htod(&qs)?;
        let keys = stream.clone_htod(&k)?;
        let key_scales = stream.clone_htod(&ks)?;
        let weights: Vec<_> = (0..rows * 32)
            .map(|i| bf16::from_f32((((i * 13 + cell.seed) % 47) as i32 - 23) as f32 / 32.0))
            .collect();
        let weights = stream.clone_htod(&weights)?;
        let committed = stream.clone_htod(&[capacity as u32])?;
        let scores = stream.clone_htod(&vec![12345.0f32; rows * columns + 16])?;
        let positions = stream.clone_htod(&vec![12345i32; rows * columns + 16])?;
        let prefix = capacity * ratio - rows;
        let metadata = stream.clone_htod(&[
            prefix as i32,
            rows as i32,
            (prefix + rows) as i32,
            (capacity * ratio) as i32,
            1,
            0,
            0,
            0,
        ])?;
        let (ptr, record) = metadata.device_ptr(&stream);
        drop(record);
        let block_count = capacity.div_ceil(8);
        let candidate_ids: Vec<_> = (0..rows * 2048)
            .map(|i| {
                let slot = i % 2048;
                if slot >= block_count {
                    -1
                } else {
                    (block_count - 1 - slot) as i32
                }
            })
            .collect();
        let candidates = stream.clone_htod(&candidate_ids)?;
        let v = IndexScoreViews {
            query: view(&query, &stream, 0, [rows as u32, 32, 64], [2048, 64, 1])?,
            query_scales: view(&query_scales, &stream, 0, [rows as u32, 32, 4], [128, 4, 1])?,
            keys: view(&keys, &stream, 0, [capacity as u32, 64], [64, 1])?,
            key_scales: view(&key_scales, &stream, 0, [capacity as u32, 4], [4, 1])?,
            weights: view(&weights, &stream, 0, [rows as u32, 32], [32, 1])?,
            committed: view(&committed, &stream, 0, [1], [1])?,
            scores: view(
                &scores,
                &stream,
                0,
                [rows as u32, columns as u32],
                [columns as u64, 1],
            )?,
            positions: view(
                &positions,
                &stream,
                0,
                [rows as u32, columns as u32],
                [columns as u64, 1],
            )?,
            sequence: SeqRef { ptr },
            candidates: if restrict {
                Some(view(
                    &candidates,
                    &stream,
                    0,
                    [rows as u32, 2048],
                    [2048, 1],
                )?)
            } else {
                None
            },
        };
        let plan = IndexScorePlan::new(ctx, v, ratio as u32)?;
        plan.run(&stream)?;
        let eager = stream.clone_dtoh(&scores)?;
        if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            plan.run(&stream)?;
            stream
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("score graph empty")?
                .launch()?;
        } else {
            plan.run(&stream)?;
        }
        let actual = stream.clone_dtoh(&scores)?;
        let ids = stream.clone_dtoh(&positions)?;
        ensure!(actual == eager, "index graph/eager bits DEFECT");
        ensure!(
            cell.scores.bytes.len() == rows * capacity * 2,
            "index reference score extent"
        );
        let reference: Vec<_> = cell
            .scores
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| bf16::from_bits(u16::from_le_bytes(v)).to_f32())
            .collect();
        let mut max_error = 0.0f64;
        for row in 0..rows {
            for col in 0..columns {
                let position = if restrict {
                    let block = candidate_ids[row * 2048 + col / 8];
                    if block < 0 {
                        None
                    } else {
                        Some(block as usize * 8 + col % 8)
                    }
                } else {
                    Some(col)
                };
                let visible = (prefix + row + 1) / ratio;
                let index = row * columns + col;
                if let Some(p) = position.filter(|&p| p < visible && p < capacity) {
                    ensure!(
                        ids[index] == p as i32 && actual[index].is_finite(),
                        "index position/finite DEFECT"
                    );
                    max_error = max_error.max(f64::from(
                        (actual[index] - reference[row * capacity + p]).abs(),
                    ));
                } else {
                    ensure!(
                        ids[index] == -1 && actual[index] == f32::NEG_INFINITY,
                        "future/padded index DEFECT"
                    );
                }
            }
        }
        ensure!(
            max_error <= SCORE_LIMIT,
            "index score DEFECT rows={rows} capacity={capacity} seed={} graph={graph} restrict={restrict} ratio={ratio} max={max_error}",
            cell.seed
        );
        ensure!(
            actual[rows * columns..].iter().all(|&v| v == 12345.0)
                && ids[rows * columns..].iter().all(|&v| v == 12345),
            "index output guard DEFECT"
        );
        let control = rows == 1 && !graph && !restrict && ratio == 1;
        if control {
            qs.iter_mut().for_each(|v| *v += 3);
            stream.memcpy_htod(&qs, &mut query_scales)?;
            plan.run(&stream)?;
            ensure!(
                stream.clone_dtoh(&scores)?[..capacity]
                    .iter()
                    .zip(&reference)
                    .any(|(&a, &b)| f64::from((a - b).abs()) > SCORE_LIMIT),
                "index scale control escaped"
            );
        }
        println!(
            "INDEX_SCORE rows={rows} capacity={capacity} seed={} graph={graph} restricted={restrict} ratio={ratio} max_abs={max_error}",
            cell.seed
        );
        Ok(control)
    }

    #[test]
    #[ignore = "requires GPU lease and D1_F1_INDEX_REFERENCE model-definition fixture"]
    fn index_score_gpu() -> Result<()> {
        let path =
            std::env::var("D1_F1_INDEX_REFERENCE").context("D1_F1_INDEX_REFERENCE required")?;
        let fixture: ScoreFixture = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            fixture.reference_only && fixture.cells.len() == 12,
            "index frozen reference matrix"
        );
        let ctx = CudaContext::new(0)?;
        let mut controls = 0;
        for cell in fixture.cells.iter().filter(|c| c.repeat == 0) {
            for ratio in [1, 2] {
                for graph in [false, true] {
                    for restricted in [false, true] {
                        controls +=
                            usize::from(score_cell(&ctx, cell, [graph, restricted], ratio)?);
                    }
                }
            }
        }
        ensure!(controls == 2, "index scale control coverage");
        println!(
            "PASS INDEX_SCORE cases=48 graph_eager_bits=0 causal_ids_exact=true guards=true wrong_scale_controls=2 limit={SCORE_LIMIT}"
        );
        Ok(())
    }
    fn tensor<T: Copy, const R: usize>(ptr: u64, shape: [u32; R]) -> Result<Tensor<T, R>> {
        let (stride, count) = dense_layout(shape)?;
        unsafe { Tensor::new(ptr, shape, stride, count * std::mem::size_of::<T>() as u64) }
    }

    #[test]
    fn index_score_layout_and_alias_bounds() -> Result<()> {
        let v = IndexScoreViews {
            query: tensor(0x100000, [5, 32, 64])?,
            query_scales: tensor(0x200000, [5, 32, 4])?,
            keys: tensor(0x300000, [16385, 64])?,
            key_scales: tensor(0x500000, [16385, 4])?,
            weights: tensor(0x600000, [5, 32])?,
            committed: tensor(0x700000, [1])?,
            scores: tensor(0x800000, [5, 16385])?,
            positions: tensor(0x900000, [5, 16385])?,
            sequence: SeqRef { ptr: 0xa00000 },
            candidates: None,
        };
        assert_eq!(geometry(v, 2)?, ([5, 32, 16385, 16385], 20485));
        assert!(
            geometry(
                IndexScoreViews {
                    query_scales: tensor(0x200000, [5, 32, 8])?,
                    ..v
                },
                2
            )
            .is_err()
        );
        assert!(
            geometry(
                IndexScoreViews {
                    positions: tensor(0x800000, [5, 16385])?,
                    ..v
                },
                1
            )
            .is_err()
        );
        let long = IndexScoreViews {
            query: tensor(0x100000000, [5, 32, 64])?,
            query_scales: tensor(0x200000000, [5, 32, 4])?,
            keys: tensor(0x300000000, [1048576, 64])?,
            key_scales: tensor(0x400000000, [1048576, 4])?,
            weights: tensor(0x500000000, [5, 32])?,
            committed: tensor(0x600000000, [1])?,
            scores: tensor(0x700000000, [5, 1048576])?,
            positions: tensor(0x800000000, [5, 1048576])?,
            sequence: SeqRef { ptr: 0x900000000 },
            candidates: None,
        };
        assert_eq!(geometry(long, 1)?, ([5, 32, 1048576, 1048576], 1310720));
        let restricted = IndexScoreViews {
            scores: tensor(0x700000000, [5, 16384])?,
            positions: tensor(0x800000000, [5, 16384])?,
            candidates: Some(tensor(0xa00000000, [5, 2048])?),
            ..long
        };
        assert_eq!(geometry(restricted, 1)?, ([5, 32, 1048576, 16384], 20480));
        assert!(
            geometry(
                IndexScoreViews {
                    key_scales: tensor(0x400000000, [1048575, 4])?,
                    ..long
                },
                1
            )
            .is_err()
        );
        let blocks = IndexCandidatesViews {
            scores: long.scores,
            block_scores: tensor(0x100000000, [5, 131072])?,
            block_ids: tensor(0x200000000, [5, 131072])?,
            selected_scores: tensor(0x300000000, [5, 2047])?,
            selected_ids: tensor(0x400000000, [5, 2047])?,
            margins: tensor(0x500000000, [5])?,
            candidates: tensor(0x600000000, [5, 2048])?,
            sequence: long.sequence,
        };
        assert_eq!(candidate_geometry(blocks, 1)?, ([5, 1048576], [2560, 40]));
        assert!(
            candidate_geometry(
                IndexCandidatesViews {
                    block_scores: tensor(0x100000000, [5, 131071])?,
                    ..blocks
                },
                1
            )
            .is_err()
        );
        Ok(())
    }

    fn block_oracle(scores: &[f32], visible: usize, reserve: bool) -> Vec<i32> {
        let newest = visible.checked_sub(1).map(|v| v / 8);
        let mut blocks: Vec<_> = scores[..visible]
            .chunks(8)
            .enumerate()
            .filter(|(id, _)| !reserve || Some(*id) != newest)
            .map(|(id, values)| {
                (
                    values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    id as i32,
                )
            })
            .collect();
        blocks.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let keep = if reserve { 2047 } else { 2048 };
        let mut result: Vec<_> = blocks.into_iter().take(keep).map(|v| v.1).collect();
        result.resize(keep, -1);
        if reserve {
            result.push(newest.map_or(-1, |v| v as i32));
        }
        result
    }

    fn candidate_cell(
        ctx: &Arc<CudaContext>,
        capacity: usize,
        ratio: usize,
        graph: bool,
    ) -> Result<bool> {
        let stream = ctx.new_stream()?;
        let rows = 5usize;
        let blocks = capacity.div_ceil(8);
        let prefix = if capacity > 32 {
            capacity * ratio - rows
        } else {
            0
        };
        let mut host: Vec<_> = (0..rows * capacity)
            .map(|i| ((i % capacity * 37 + i / capacity * 13) % 65521) as f32 / 64.0)
            .collect();
        for row in 0..rows {
            let visible = (prefix + row + 1) / ratio;
            if visible > 0 {
                for position in (visible - 1) / 8 * 8..visible {
                    host[row * capacity + position] = -1.0e6;
                }
            }
        }
        let scores = stream.clone_htod(&host)?;
        let block_scores = stream.clone_htod(&vec![12345.0f32; rows * blocks + 16])?;
        let block_ids = stream.clone_htod(&vec![12345i32; rows * blocks + 16])?;
        let selected_scores = stream.clone_htod(&vec![12345.0f32; rows * 2047 + 16])?;
        let selected_ids = stream.clone_htod(&vec![12345i32; rows * 2047 + 16])?;
        let margins = stream.clone_htod(&vec![12345.0f32; rows + 16])?;
        let candidates = stream.clone_htod(&vec![12345i32; rows * 2048 + 16])?;
        let mut metadata = stream.clone_htod(&[
            prefix as i32,
            rows as i32,
            (prefix + rows) as i32,
            (capacity * ratio + rows) as i32,
            1,
            0,
            0,
            0,
        ])?;
        let (ptr, record) = metadata.device_ptr(&stream);
        drop(record);
        let v = IndexCandidatesViews {
            scores: view(
                &scores,
                &stream,
                0,
                [rows as u32, capacity as u32],
                [capacity as u64, 1],
            )?,
            block_scores: view(
                &block_scores,
                &stream,
                0,
                [rows as u32, blocks as u32],
                [blocks as u64, 1],
            )?,
            block_ids: view(
                &block_ids,
                &stream,
                0,
                [rows as u32, blocks as u32],
                [blocks as u64, 1],
            )?,
            selected_scores: view(&selected_scores, &stream, 0, [rows as u32, 2047], [2047, 1])?,
            selected_ids: view(&selected_ids, &stream, 0, [rows as u32, 2047], [2047, 1])?,
            margins: view(&margins, &stream, 0, [rows as u32], [1])?,
            candidates: view(&candidates, &stream, 0, [rows as u32, 2048], [2048, 1])?,
            sequence: SeqRef { ptr },
        };
        let plan = IndexCandidatesPlan::new(ctx, v, ratio as u32)?;
        let captured = if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            plan.run(&stream)?;
            Some(
                stream
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("index graph empty")?,
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
        for live in [rows, 1] {
            stream.memcpy_htod(
                &[
                    prefix as i32,
                    live as i32,
                    (prefix + live) as i32,
                    (capacity * ratio + rows) as i32,
                    1,
                    0,
                    0,
                    0,
                ],
                &mut metadata,
            )?;
            execute()?;
            let actual = stream.clone_dtoh(&candidates)?;
            for row in 0..rows {
                let visible = if row < live {
                    (prefix + row + 1) / ratio
                } else {
                    0
                };
                let expected =
                    block_oracle(&host[row * capacity..(row + 1) * capacity], visible, true);
                ensure!(
                    actual[row * 2048..(row + 1) * 2048] == expected,
                    "index candidate DEFECT capacity={capacity} ratio={ratio} graph={graph} live={live} row={row}"
                );
            }
            ensure!(
                actual[rows * 2048..].iter().all(|&v| v == 12345),
                "candidate output guard DEFECT"
            );
        }
        for (values, count) in [
            (stream.clone_dtoh(&block_ids)?, rows * blocks),
            (stream.clone_dtoh(&selected_ids)?, rows * 2047),
        ] {
            ensure!(
                values[count..].iter().all(|&v| v == 12345),
                "candidate integer guard DEFECT"
            );
        }
        for (values, count) in [
            (stream.clone_dtoh(&block_scores)?, rows * blocks),
            (stream.clone_dtoh(&selected_scores)?, rows * 2047),
            (stream.clone_dtoh(&margins)?, rows),
        ] {
            ensure!(
                values[count..].iter().all(|&v| v == 12345.0),
                "candidate float guard DEFECT"
            );
        }
        let control = capacity > 32768 && !graph;
        if control {
            let visible = (prefix + 1) / ratio;
            let mutant = block_oracle(&host[..capacity], visible, false);
            ensure!(
                !mutant.contains(&((visible - 1) as i32 / 8)),
                "mandatory-newest control escaped"
            );
        }
        Ok(control)
    }

    fn order_cell(ctx: &Arc<CudaContext>, graph: bool) -> Result<()> {
        let stream = ctx.new_stream()?;
        let rows = 5usize;
        let host: Vec<i32> = (0..rows * 512)
            .map(|i| {
                if i % 17 == 0 {
                    -1
                } else {
                    ((i % 512 * 37) % 512 + i / 512 * 32769) as i32
                }
            })
            .collect();
        let input = stream.clone_htod(&host)?;
        let output = stream.clone_htod(&vec![12345i32; rows * 512 + 16])?;
        let plan = IndexPositionOrder::new(
            ctx,
            view(&input, &stream, 0, [rows as u32, 512], [512, 1])?,
            view(&output, &stream, 0, [rows as u32, 512], [512, 1])?,
        )?;
        if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            plan.run(&stream)?;
            stream
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("order graph empty")?
                .launch()?;
        } else {
            plan.run(&stream)?;
        }
        let actual = stream.clone_dtoh(&output)?;
        for row in 0..rows {
            let mut expected = host[row * 512..(row + 1) * 512].to_vec();
            expected.sort_by_key(|&v| if v < 0 { i32::MAX } else { v });
            ensure!(
                actual[row * 512..(row + 1) * 512] == expected,
                "token position order DEFECT row={row} graph={graph}"
            );
        }
        ensure!(
            actual[rows * 512..].iter().all(|&v| v == 12345),
            "token order guard DEFECT"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU lease; exact candidate and position-order gate"]
    fn index_candidates_gpu() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let mut controls = 0;
        for capacity in [17, 32769] {
            for ratio in [1, 2] {
                for graph in [false, true] {
                    controls += usize::from(candidate_cell(&ctx, capacity, ratio, graph)?);
                }
            }
        }
        for graph in [false, true] {
            order_cell(&ctx, graph)?;
        }
        ensure!(controls == 2, "candidate control coverage");
        println!(
            "PASS INDEX_CANDIDATES cases=8 order_cases=2 mismatches=0 graph_live=true guards=true mandatory_newest_controls=2"
        );
        Ok(())
    }
    #[derive(Clone, serde::Deserialize)]
    struct RecordedIndex {
        owner: usize,
        position: usize,
        ratio: usize,
        capacity: usize,
        query: Vec<u8>,
        query_scales: Vec<u8>,
        keys: Vec<u8>,
        key_scales: Vec<u8>,
        weights: Vec<u16>,
        scores: Vec<Option<f32>>,
        selected: Vec<i32>,
        restricted_selected: Vec<i32>,
        candidate_mask: Option<Vec<bool>>,
        bound: f64,
    }

    fn index_native_value(codes: &[u8], scales: &[u8], index: usize) -> f64 {
        let packed = codes[index / 2];
        let nibble = if index.is_multiple_of(2) {
            packed & 15
        } else {
            packed >> 4
        };
        let level = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(nibble & 7)];
        let signed = if nibble & 8 == 0 { level } else { -level };
        f64::from(
            bf16::from_f64(signed * 2.0f64.powi(i32::from(scales[index / 32]) - 127)).to_f32(),
        )
    }

    fn index_score_interval(cell: &RecordedIndex, column: usize) -> [f64; 2] {
        let u = f64::from(f32::EPSILON) / 2.0;
        let gamma = 128.0 * u / (1.0 - 128.0 * u);
        let gamma_heads = 32.0 * u / (1.0 - 32.0 * u);
        let rounded = |v: f64| f64::from(bf16::from_f64(v).to_f32());
        let (mut low, mut high, mut absolute) = (0.0f64, 0.0f64, 0.0f64);
        for head in 0..32 {
            let (mut dot, mut sum_abs) = (0.0f64, 0.0f64);
            for d in 0..128 {
                let term = index_native_value(&cell.query, &cell.query_scales, head * 128 + d)
                    * index_native_value(&cell.keys, &cell.key_scales, column * 128 + d);
                dot += term;
                sum_abs += term.abs();
            }
            let weight = rounded(f64::from(bf16::from_bits(cell.weights[head]).to_f32()) / 64.0);
            let a = rounded(rounded(dot - gamma * sum_abs).max(0.0) * weight);
            let b = rounded(rounded(dot + gamma * sum_abs).max(0.0) * weight);
            low += a.min(b);
            high += a.max(b);
            absolute += a.abs().max(b.abs());
        }
        [
            rounded(low - gamma_heads * absolute),
            rounded(high + gamma_heads * absolute),
        ]
    }

    // 2026-10-29: both swapped scores must lie inside the kth-score FP32 reduction interval.
    fn index_membership(
        cell: &RecordedIndex,
        actual: &[i32],
        scores: &[f32],
    ) -> Result<(usize, usize)> {
        use std::collections::BTreeSet;
        let got: BTreeSet<_> = actual.iter().copied().filter(|&i| i >= 0).collect();
        let want: BTreeSet<_> = cell.selected.iter().copied().collect();
        let visible = (cell.position + 1) / cell.ratio;
        let invalid = got.iter().filter(|&&i| i as usize >= visible).count();
        let duplicate = actual.iter().filter(|&&i| i >= 0).count() - got.len();
        let added: Vec<_> = got.difference(&want).copied().collect();
        let removed: Vec<_> = want.difference(&got).copied().collect();
        let mut defects = invalid + duplicate + added.len().abs_diff(removed.len());
        let mut exempt = 0;
        let kth = want
            .iter()
            .min_by(|&&a, &&b| {
                cell.scores[a as usize]
                    .unwrap_or(f32::NEG_INFINITY)
                    .total_cmp(&cell.scores[b as usize].unwrap_or(f32::NEG_INFINITY))
            })
            .copied();
        if let Some(kth) = kth {
            let interval = index_score_interval(cell, kth as usize);
            for (&a, &b) in added.iter().zip(&removed) {
                let valid = a >= 0 && (a as usize) < visible && b >= 0 && (b as usize) < visible;
                let values = if valid {
                    [
                        f64::from(scores[a as usize]),
                        f64::from(cell.scores[a as usize].context("reachable score")?),
                        f64::from(scores[b as usize]),
                        f64::from(cell.scores[b as usize].context("reachable score")?),
                    ]
                } else {
                    [f64::INFINITY; 4]
                };
                let proven = values.iter().all(|&v| interval[0] <= v && v <= interval[1]);
                exempt += usize::from(proven);
                defects += usize::from(!proven);
                println!(
                    "INDEX_SWAP owner={} position={} added={a} removed={b} scores={values:?} kth_id={kth} kth_score={:?} interval={interval:?} exempt={proven}",
                    cell.owner, cell.position, cell.scores[kth as usize]
                );
            }
        } else {
            defects += added.len();
        }
        println!(
            "INDEX_MEMBERSHIP owner={} position={} exempt_swaps={exempt} defects={defects}",
            cell.owner, cell.position
        );
        Ok((defects, exempt))
    }

    fn select_recorded_scores(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        scores: &cudarc::driver::CudaSlice<f32>,
        ids: &cudarc::driver::CudaSlice<i32>,
        columns: usize,
        graph: bool,
    ) -> Result<Vec<i32>> {
        let selected_scores = stream.alloc_zeros::<f32>(512)?;
        let selected_ids = stream.alloc_zeros::<i32>(512)?;
        let margins = stream.alloc_zeros::<f32>(1)?;
        let ordered = stream.alloc_zeros::<i32>(512)?;
        let plan = DsaSelectionPlan::new(ctx, 1, columns as u32, 512)?;
        let input = DsaTopKInput {
            scores: view(scores, stream, 0, [1, columns as u32], [columns as u64, 1])?,
            ids: view(ids, stream, 0, [1, columns as u32], [columns as u64, 1])?,
        };
        let output = DsaTopKOutput {
            scores: TensorMut(view(&selected_scores, stream, 0, [1, 512], [512, 1])?),
            ids: TensorMut(view(&selected_ids, stream, 0, [1, 512], [512, 1])?),
            margins: TensorMut(view(&margins, stream, 0, [1], [1])?),
        };
        let order = IndexPositionOrder::new(
            ctx,
            output.ids.read(),
            view(&ordered, stream, 0, [1, 512], [512, 1])?,
        )?;
        if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
        }
        plan.select_scores(&input, &output, stream)?;
        order.run(stream)?;
        if graph {
            stream
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("index selection graph")?
                .launch()?;
        }
        let host = stream.clone_dtoh(&ordered)?;
        let valid: Vec<_> = host.iter().copied().filter(|&i| i >= 0).collect();
        ensure!(
            valid.windows(2).all(|v| v[0] < v[1]) && host[valid.len()..].iter().all(|&v| v == -1),
            "selected position order DEFECT"
        );
        Ok(host)
    }

    fn recorded_candidates(
        ctx: &Arc<CudaContext>,
        cell: &RecordedIndex,
        graph: bool,
    ) -> Result<(Vec<i32>, usize, usize)> {
        use std::collections::BTreeSet;
        let s = ctx.new_stream()?;
        let c = cell.capacity;
        let blocks = c.div_ceil(8);
        let visible = (cell.position + 1) / cell.ratio;
        let host: Vec<_> = cell
            .scores
            .iter()
            .map(|v| v.unwrap_or(f32::NEG_INFINITY))
            .collect();
        let scores = s.clone_htod(&host)?;
        let block_scores = s.alloc_zeros::<f32>(blocks)?;
        let block_ids = s.alloc_zeros::<i32>(blocks)?;
        let selected_scores = s.alloc_zeros::<f32>(2047)?;
        let selected_ids = s.alloc_zeros::<i32>(2047)?;
        let margins = s.alloc_zeros::<f32>(1)?;
        let candidates = s.alloc_zeros::<i32>(2048)?;
        let meta = s.clone_htod(&[
            cell.position as i32,
            1,
            (cell.position + 1) as i32,
            (c * cell.ratio + 1) as i32,
            1,
            0,
            0,
            0,
        ])?;
        let (ptr, guard) = meta.device_ptr(&s);
        drop(guard);
        let plan = IndexCandidatesPlan::new(
            ctx,
            IndexCandidatesViews {
                scores: view(&scores, &s, 0, [1, c as u32], [c as u64, 1])?,
                block_scores: view(&block_scores, &s, 0, [1, blocks as u32], [blocks as u64, 1])?,
                block_ids: view(&block_ids, &s, 0, [1, blocks as u32], [blocks as u64, 1])?,
                selected_scores: view(&selected_scores, &s, 0, [1, 2047], [2047, 1])?,
                selected_ids: view(&selected_ids, &s, 0, [1, 2047], [2047, 1])?,
                margins: view(&margins, &s, 0, [1], [1])?,
                candidates: view(&candidates, &s, 0, [1, 2048], [2048, 1])?,
                sequence: SeqRef { ptr },
            },
            cell.ratio as u32,
        )?;
        if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
        }
        plan.run(&s)?;
        if graph {
            s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("recorded candidate graph")?
                .launch()?;
        }
        let actual = s.clone_dtoh(&candidates)?;
        let want: BTreeSet<i32> = if let Some(mask) = &cell.candidate_mask {
            ensure!(mask.len() == c, "publisher candidate-mask shape");
            mask.iter()
                .enumerate()
                .filter(|&(_, v)| *v)
                .map(|(i, _)| (i / 8) as i32)
                .collect()
        } else {
            (0..visible.div_ceil(8) as i32).collect()
        };
        let got: BTreeSet<_> = actual.iter().copied().filter(|&v| v >= 0).collect();
        let newest = visible.checked_sub(1).map(|v| (v / 8) as i32);
        let mut bad = usize::from(newest.is_some_and(|v| !got.contains(&v)))
            + actual.iter().filter(|&&v| v >= 0).count()
            - got.len();
        let added: Vec<_> = got.difference(&want).copied().collect();
        let removed: Vec<_> = want.difference(&got).copied().collect();
        bad += added.len().abs_diff(removed.len());
        let best = |block: i32| -> Option<usize> {
            if block < 0 {
                return None;
            }
            (block as usize * 8..((block as usize + 1) * 8).min(visible))
                .max_by(|&a, &b| host[a].total_cmp(&host[b]))
        };
        let kth = want
            .iter()
            .filter(|&&b| Some(b) != newest)
            .filter_map(|&b| best(b))
            .min_by(|&a, &b| host[a].total_cmp(&host[b]));
        let mut exemptions = 0;
        for (&a, &b) in added.iter().zip(&removed) {
            let (av, bv) = (
                best(a).map_or(f64::INFINITY, |i| f64::from(host[i])),
                best(b).map_or(f64::INFINITY, |i| f64::from(host[i])),
            );
            let interval = kth
                .map(|i| index_score_interval(cell, i))
                .unwrap_or([f64::NAN; 2]);
            let proven = [av, bv]
                .iter()
                .all(|&v| interval[0] <= v && v <= interval[1]);
            exemptions += usize::from(proven);
            bad += usize::from(!proven);
            println!(
                "INDEX_BLOCK_SWAP owner={} position={} added={a} removed={b} scores=[{av:e},{bv:e}] kth={kth:?} interval={interval:?} exempt={proven}",
                cell.owner, cell.position
            );
        }
        let mut reference: Vec<_> = want
            .iter()
            .copied()
            .filter(|&b| Some(b) != newest)
            .collect();
        reference.resize(2047, -1);
        reference.push(newest.unwrap_or(-1));
        if c > 16384 {
            ensure!(
                newest.is_some() && !reference[..2047].contains(&newest.unwrap()),
                "wrong newest-block ticket control escaped"
            );
        }
        println!(
            "INDEX_BLOCKS owner={} position={} graph={graph} exempt_swaps={exemptions} defects={bad}",
            cell.owner, cell.position
        );
        Ok((reference, bad, exemptions))
    }

    fn restricted_recorded_scores(
        ctx: &Arc<CudaContext>,
        cell: &RecordedIndex,
        views: IndexScoreViews,
        reference: &[i32],
        graph: bool,
    ) -> Result<(usize, usize)> {
        let s = ctx.new_stream()?;
        let candidates = s.clone_htod(reference)?;
        let scores = s.alloc_zeros::<f32>(16384)?;
        let ids = s.alloc_zeros::<i32>(16384)?;
        let plan = IndexScorePlan::new(
            ctx,
            IndexScoreViews {
                scores: view(&scores, &s, 0, [1, 16384], [16384, 1])?,
                positions: view(&ids, &s, 0, [1, 16384], [16384, 1])?,
                candidates: Some(view(&candidates, &s, 0, [1, 2048], [2048, 1])?),
                ..views
            },
            cell.ratio as u32,
        )?;
        if graph {
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
        }
        plan.run(&s)?;
        if graph {
            s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("restricted index graph")?
                .launch()?;
        }
        let got = s.clone_dtoh(&scores)?;
        let positions = s.clone_dtoh(&ids)?;
        let mut defects = 0;
        for i in 0..16384 {
            let block = reference[i / 8];
            let p = block as i64 * 8 + (i % 8) as i64;
            if block < 0 || p as usize >= (cell.position + 1) / cell.ratio {
                defects += usize::from(positions[i] != -1 || got[i] != f32::NEG_INFINITY);
                continue;
            }
            defects += usize::from(positions[i] != p as i32);
            let expected = cell.scores[p as usize].context("restricted publisher score")?;
            if f64::from((got[i] - expected).abs()) > cell.bound {
                let interval = index_score_interval(cell, p as usize);
                let proven = [got[i], expected]
                    .iter()
                    .all(|&v| interval[0] <= f64::from(v) && f64::from(v) <= interval[1]);
                defects += usize::from(!proven);
                println!(
                    "INDEX_RESTRICTED_PROOF owner={} position={} column={p} interval={interval:?} candidate={:e} publisher={expected:e} exempt={proven}",
                    cell.owner, cell.position, got[i]
                );
            }
        }
        let selected = select_recorded_scores(ctx, &s, &scores, &ids, 16384, graph)?;
        let mut scoring = cell.clone();
        scoring.selected = cell.restricted_selected.clone();
        let mut dense = vec![f32::NEG_INFINITY; cell.capacity];
        for (&value, &id) in got.iter().zip(&positions) {
            if id >= 0 {
                dense[id as usize] = value;
            }
        }
        let (bad, swaps) = index_membership(&scoring, &selected, &dense)?;
        defects += bad;
        println!(
            "INDEX_RESTRICTED owner={} position={} graph={graph} exempt_swaps={swaps} defects={defects}",
            cell.owner, cell.position
        );
        Ok((defects, swaps))
    }
    fn recorded_index_case(
        ctx: &Arc<CudaContext>,
        cell: &RecordedIndex,
        graph: bool,
    ) -> Result<(usize, usize, bool)> {
        let stream = ctx.new_stream()?;
        let c = cell.capacity;
        ensure!(
            c > 0
                && cell.query.len() == 32 * 64
                && cell.query_scales.len() == 32 * 4
                && cell.keys.len() == c * 64
                && cell.key_scales.len() == c * 4
                && cell.weights.len() == 32
                && cell.scores.len() == c
                && cell.bound.is_finite()
                && cell.bound >= 0.0,
            "index recorded extent/bound"
        );
        let query = stream.clone_htod(&cell.query)?;
        let mut scales = stream.clone_htod(&cell.query_scales)?;
        let keys = stream.clone_htod(&cell.keys)?;
        let key_scales = stream.clone_htod(&cell.key_scales)?;
        let weights = stream.clone_htod(
            &cell
                .weights
                .iter()
                .map(|&v| bf16::from_bits(v))
                .collect::<Vec<_>>(),
        )?;
        let visible = (cell.position + 1) / cell.ratio;
        let committed = stream.clone_htod(&[visible as u32])?;
        let scores = stream.clone_htod(&vec![12345.0f32; c + 16])?;
        let positions = stream.clone_htod(&vec![12345i32; c + 16])?;
        let meta = stream.clone_htod(&[
            cell.position as i32,
            1,
            (cell.position + 1) as i32,
            (c * cell.ratio + 1) as i32,
            1,
            0,
            0,
            0,
        ])?;
        let (ptr, guard) = meta.device_ptr(&stream);
        drop(guard);
        let views = IndexScoreViews {
            query: view(&query, &stream, 0, [1, 32, 64], [2048, 64, 1])?,
            query_scales: view(&scales, &stream, 0, [1, 32, 4], [128, 4, 1])?,
            keys: view(&keys, &stream, 0, [c as u32, 64], [64, 1])?,
            key_scales: view(&key_scales, &stream, 0, [c as u32, 4], [4, 1])?,
            weights: view(&weights, &stream, 0, [1, 32], [32, 1])?,
            committed: view(&committed, &stream, 0, [1], [1])?,
            scores: view(&scores, &stream, 0, [1, c as u32], [c as u64, 1])?,
            positions: view(&positions, &stream, 0, [1, c as u32], [c as u64, 1])?,
            sequence: SeqRef { ptr },
            candidates: None,
        };
        let plan = IndexScorePlan::new(ctx, views, cell.ratio as u32)?;
        plan.run(&stream)?;
        let baseline = stream.clone_dtoh(&scores)?;
        if graph {
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
        }
        plan.run(&stream)?;
        if graph {
            stream
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("recorded score graph")?
                .launch()?;
        }
        let actual = stream.clone_dtoh(&scores)?;
        let ids = stream.clone_dtoh(&positions)?;
        ensure!(
            actual == baseline
                && actual[c..].iter().all(|&v| v == 12345.0)
                && ids[c..].iter().all(|&v| v == 12345),
            "index repeat/guard DEFECT"
        );
        let (mut defects, mut exempt) = (0, 0);
        for i in 0..c {
            if let Some(expected) = cell.scores[i] {
                ensure!(actual[i].is_finite(), "nonfinite recorded score");
                defects += usize::from(ids[i] != i as i32);
                if f64::from((actual[i] - expected).abs()) > cell.bound {
                    let interval = index_score_interval(cell, i);
                    let proven = [actual[i], expected]
                        .iter()
                        .all(|&v| interval[0] <= f64::from(v) && f64::from(v) <= interval[1]);
                    exempt += usize::from(proven);
                    defects += usize::from(!proven);
                    println!(
                        "INDEX_SCORE_PROOF owner={} position={} column={i} n=128 heads=32 interval={interval:?} candidate={:e} publisher={expected:e} exempt={proven}",
                        cell.owner, cell.position, actual[i]
                    );
                }
            } else {
                defects += usize::from(actual[i] != f32::NEG_INFINITY || ids[i] != -1);
            }
        }
        let selected = select_recorded_scores(ctx, &stream, &scores, &positions, c, graph)?;
        let (bad, swaps) = index_membership(cell, &selected, &actual)?;
        defects += bad;
        stream.synchronize()?;
        let (candidate_ids, candidate_bad, block_swaps) = recorded_candidates(ctx, cell, graph)?;
        defects += candidate_bad;
        let (restricted_bad, restricted_swaps) =
            restricted_recorded_scores(ctx, cell, views, &candidate_ids, graph)?;
        defects += restricted_bad;
        let swaps = swaps + block_swaps + restricted_swaps;
        let control = !graph && cell.position == 128;
        if control {
            let changed: Vec<_> = cell
                .query_scales
                .iter()
                .map(|&v| v.saturating_add(1))
                .collect();
            stream.memcpy_htod(&changed, &mut scales)?;
            plan.run(&stream)?;
            let wrong = stream.clone_dtoh(&scores)?;
            ensure!(
                cell.scores
                    .iter()
                    .enumerate()
                    .any(|(i, e)| e.is_some_and(|e| f64::from((wrong[i] - e).abs()) > cell.bound)),
                "wrong index scale escaped"
            );
            let mut wrong = selected.clone();
            wrong[0] = visible as i32;
            ensure!(
                index_membership(cell, &wrong, &actual)?.0 > 0,
                "wrong index ticket escaped"
            );
        }
        println!(
            "INDEX_OWNER owner={} position={} graph={graph} score_exempt={exempt} exempt_swaps={swaps} defects={defects}",
            cell.owner, cell.position
        );
        Ok((defects, swaps, control))
    }

    #[test]
    #[ignore = "requires GPU lease and D1_OP_REFERENCE recorded index operands"]
    fn index_publisher_operands_gpu() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Record {
            reference_only: bool,
            candidate_seen: bool,
            indexes: Vec<RecordedIndex>,
        }
        let record: Record =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_OP_REFERENCE")?)?)?;
        ensure!(
            record.reference_only && !record.candidate_seen && record.indexes.len() == 66,
            "index owner/long matrix"
        );
        let ctx = CudaContext::new(0)?;
        let (mut defects, mut swaps, mut controls) = (0, 0, 0);
        for cell in &record.indexes {
            for graph in [false, true] {
                let (bad, exempt, killed) = recorded_index_case(&ctx, cell, graph)?;
                defects += bad;
                swaps += exempt;
                controls += usize::from(killed);
            }
        }
        println!(
            "{} INDEX_PUBLISHER cases=132 defects={defects} exempt_swaps={swaps} wrong_scale_and_ticket_controls={controls}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0 && controls == 8, "index publisher DEFECT");
        Ok(())
    }
    #[test]
    fn index_cutoff_requires_both_swapped_scores_in_interval() -> Result<()> {
        let cell = RecordedIndex {
            owner: 2,
            position: 1,
            ratio: 1,
            capacity: 2,
            query: vec![0; 2048],
            query_scales: vec![127; 128],
            keys: vec![0; 128],
            key_scales: vec![127; 8],
            weights: vec![0; 32],
            scores: vec![Some(0.0); 2],
            selected: vec![0],
            restricted_selected: vec![0],
            candidate_mask: None,
            bound: 0.0,
        };
        assert_eq!(index_membership(&cell, &[1], &[0.0, 0.0])?, (0, 1));
        assert_eq!(index_membership(&cell, &[1], &[0.0, 1.0])?, (1, 0));
        assert_eq!(index_membership(&cell, &[1], &[1.0, 0.0])?, (1, 0));
        Ok(())
    }
}
