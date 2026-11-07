//! Published sparse [Co,kx,ky,kz,Ci] weights with bounded gather workspace.
use super::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, no_overlap, view},
    cubin,
    ops::{flat, grid},
};
use crate::{
    backend::gpu::{CudaFunction, CudaModule, CudaSlice, DeviceRepr, PushKernelArg},
    sparse::{SparseCorners, SparseMesh, SparseSites},
};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::{rc::Rc, sync::Arc};

pub(crate) struct CornerFields {
    pub layout: SparseCorners,
    pub cube_to_corner: CudaSlice<u32>,
    pub means: CudaSlice<f32>,
    pub positions: CudaSlice<f32>,
    pub scalar: CudaSlice<f32>,
}

pub(crate) fn corner_fields(
    d: &Device,
    cubes: &SparseSites,
    fields: &CudaSlice<f32>,
    frame: [[f32; 4]; 4],
    budget: [usize; 2],
) -> Result<CornerFields> {
    let rows = cubes.coordinates().len();
    ensure!(
        fields.len() == rows.checked_mul(40).context("cube field extent overflow")?
            && frame.iter().flatten().all(|v| v.is_finite()),
        "invalid cube corner fields or matrix"
    );
    let layout = cubes.cube_corners()?;
    let corners = layout.sites.coordinates().len();
    let live = rows
        .checked_mul(64)
        .and_then(|v| v.checked_add(corners.checked_mul(60)?))
        .and_then(|v| v.checked_add(72))
        .context("cube corner workspace overflow")?;
    d.workspace_budget(budget[0], budget[1], live)?;
    let n = u32::try_from(rows)?;
    let u = u32::try_from(corners)?;
    let cube_to_corner = d.stream.clone_htod(&layout.cube_to_corner)?;
    let offsets = d.stream.clone_htod(&layout.contributor_offsets)?;
    let contributors = d.stream.clone_htod(&layout.contributors)?;
    let coordinates = d
        .stream
        .clone_htod(layout.sites.coordinates().as_flattened())?;
    let matrix = d.stream.clone_htod(frame.as_flattened())?;
    let means = d.stream.alloc_zeros::<f32>(corners * 5)?;
    let positions = d.stream.alloc_zeros::<f32>(corners * 3)?;
    let scalar = d.stream.alloc_zeros::<f32>(corners)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("sparse_corner_mean_f32")?)
            .arg(fields)
            .arg(&offsets)
            .arg(&contributors)
            .arg(&n)
            .arg(&u)
            .arg(&means)
            .launch(flat(corners))?;
        d.stream
            .launch_builder(&module.load_function("sparse_corner_world_f32")?)
            .arg(&coordinates)
            .arg(&means)
            .arg(&matrix)
            .arg(&u)
            .arg(&positions)
            .arg(&scalar)
            .launch(flat(corners))?;
    }
    eprintln!(
        "SPARSE_CORNER_PLAN cubes={rows} corners={corners} workspace_bytes={live} contributor_degree_max=8"
    );
    Ok(CornerFields {
        layout,
        cube_to_corner,
        means,
        positions,
        scalar,
    })
}

pub struct BatchGroupNorm {
    device: Device,
    offsets: CudaSlice<u32>,
    partial: CudaSlice<f32>,
    moments: CudaSlice<f32>,
    reduce: [CudaFunction; 2],
    finish: CudaFunction,
    apply: [CudaFunction; 3],
    rows: u32,
    channels: u32,
    groups: u32,
    batches: u32,
    reduce_blocks: u32,
    group_tile: u32,
    apply_blocks: u32,
    eps: f32,
}

impl BatchGroupNorm {
    pub fn new(
        device: &Device,
        offsets: &[u32],
        channels: usize,
        groups: usize,
        eps: f32,
        workspace_budget: usize,
    ) -> Result<Self> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        ensure!(
            offsets.len() >= 2 && offsets[0] == 0 && offsets.windows(2).all(|v| v[0] < v[1]),
            "invalid group norm batch offsets"
        );
        ensure!(
            groups > 0
                && channels > 0
                && channels.is_multiple_of(groups)
                && channels <= u32::MAX as usize
                && eps.is_finite()
                && eps > 0.0,
            "invalid batch group norm geometry"
        );
        let batches = offsets.len() - 1;
        let rows = *offsets.last().context("empty batch offsets")?;
        let maximum = offsets
            .windows(2)
            .map(|v| (v[1] - v[0]) as usize)
            .max()
            .context("empty batch offsets")?;
        let module = cubin::module(&device.ctx, "vision_ops")?;
        let reduce = [
            module.load_function("batch_group_moments_scatter_f32")?,
            module.load_function("batch_group_moments_scatter_f16")?,
        ];
        let apply = [
            module.load_function("batch_group_norm_f32")?,
            module.load_function("batch_group_norm_f16")?,
            module.load_function("batch_group_norm_f16_to_f32")?,
        ];
        let resident = reduce
            .iter()
            .map(|function| {
                function
                    .occupancy_max_active_blocks_per_multiprocessor(256, 0, None)
                    .map(|v| v as usize)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .min()
            .context("group norm occupancy missing")?;
        let sms = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
        let bg = batches
            .checked_mul(groups)
            .context("group norm group count overflow")?;
        ensure!(
            resident > 0
                && sms > 0
                && batches
                    <= device
                        .ctx
                        .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?
                        as usize
                && groups
                    <= device
                        .ctx
                        .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?
                        as usize,
            "group norm grid or occupancy exceeds device limits"
        );
        let group_tile = groups.min(8);
        let reduce_blocks = maximum
            .checked_mul(group_tile)
            .context("group norm reduction extent overflow")?
            .div_ceil(256);
        let max_grid = device
            .ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)? as usize;
        ensure!(
            reduce_blocks <= max_grid,
            "group norm reduction grid exceeds device limit"
        );
        let apply_blocks = maximum
            .checked_mul(channels)
            .context("group norm apply extent overflow")?
            .div_ceil(256)
            .min(sms * resident)
            .min(max_grid)
            .max(1);
        let partial_elements = bg
            .checked_mul(2)
            .context("group norm partial extent overflow")?;
        let moment_elements = bg.checked_mul(2).context("group norm moments overflow")?;
        let bytes = partial_elements
            .checked_add(moment_elements)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .and_then(|n| n.checked_add(size_of_val(offsets)))
            .context("group norm workspace overflow")?;
        let (free, _) = device.ctx.mem_get_info()?;
        ensure!(
            bytes <= workspace_budget.min(free),
            "batch group norm workspace needs {bytes} bytes"
        );
        eprintln!(
            "BATCH_GROUP_NORM_PLAN batches={batches} rows={rows} channels={channels} groups={groups} reduce_blocks={reduce_blocks} group_tile={group_tile} apply_blocks={apply_blocks} workspace_bytes={bytes}"
        );
        Ok(Self {
            device: device.clone(),
            offsets: device.stream.clone_htod(offsets)?,
            partial: device.stream.alloc_zeros(partial_elements)?,
            moments: device.stream.alloc_zeros(moment_elements)?,
            reduce,
            finish: module.load_function("batch_group_moments_finish_f32")?,
            apply,
            rows,
            channels: u32::try_from(channels)?,
            groups: u32::try_from(groups)?,
            batches: u32::try_from(batches)?,
            reduce_blocks: u32::try_from(reduce_blocks)?,
            group_tile: u32::try_from(group_tile)?,
            apply_blocks: u32::try_from(apply_blocks)?,
            eps,
        })
    }
    pub fn run<I: DeviceRepr + Copy + 'static, O: DeviceRepr + Copy + 'static>(
        &self,
        input: Tensor<I, 2>,
        affine: [Tensor<f32, 1>; 2],
        output: &mut TensorMut<O, 2>,
    ) -> Result<()> {
        use std::any::TypeId;
        let half = TypeId::of::<I>() == TypeId::of::<f16>();
        let output_half = TypeId::of::<O>() == TypeId::of::<f16>();
        ensure!(
            (half || TypeId::of::<I>() == TypeId::of::<f32>())
                && (TypeId::of::<O>() == TypeId::of::<f32>() || (half && output_half)),
            "batch group norm requires F16 or F32 features"
        );
        input.validate()?;
        output.0.validate()?;
        ensure!(
            input.shape == [self.rows, self.channels]
                && input.stride == [self.channels as u64, 1]
                && output.0.shape == input.shape
                && output.0.stride == input.stride,
            "batch group norm feature extent mismatch"
        );
        if input.ptr != output.0.ptr {
            no_overlap(&input, &output.0, "group norm input", "group norm output")?;
        }
        for tensor in &affine {
            tensor.validate()?;
            ensure!(
                tensor.shape == [self.channels] && tensor.stride == [1],
                "batch group norm affine extent mismatch"
            );
            no_overlap(tensor, &output.0, "group norm affine", "group norm output")?;
        }
        let stream = &self.device.stream;
        unsafe {
            super::copy::memset_words(
                view(
                    &self.partial,
                    stream,
                    0,
                    [u32::try_from(self.partial.len())?],
                    [1],
                )?
                .ptr,
                0,
                self.partial.len(),
                stream,
            )?;
            for group_first in (0..self.groups).step_by(self.group_tile as usize) {
                let group_count = self.group_tile.min(self.groups - group_first);
                stream
                    .launch_builder(&self.reduce[usize::from(half)])
                    .arg(&input.ptr)
                    .arg(&self.offsets)
                    .arg(&self.partial)
                    .arg(&self.batches)
                    .arg(&self.channels)
                    .arg(&self.groups)
                    .arg(&group_first)
                    .arg(&group_count)
                    .launch(crate::backend::gpu::LaunchConfig {
                        grid_dim: (self.reduce_blocks, self.batches, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream
                .launch_builder(&self.finish)
                .arg(&self.partial)
                .arg(&self.offsets)
                .arg(&self.moments)
                .arg(&self.batches)
                .arg(&self.channels)
                .arg(&self.groups)
                .arg(&self.eps)
                .launch(flat(self.batches as usize * self.groups as usize))?;
            stream
                .launch_builder(
                    &self.apply[if half && !output_half {
                        2
                    } else {
                        usize::from(half)
                    }],
                )
                .arg(&input.ptr)
                .arg(&self.offsets)
                .arg(&self.moments)
                .arg(&affine[0].ptr)
                .arg(&affine[1].ptr)
                .arg(&output.0.ptr)
                .arg(&self.batches)
                .arg(&self.channels)
                .arg(&self.groups)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: (self.apply_blocks, self.batches, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}

pub(crate) fn exclusive_scan(d: &Device, counts: &CudaSlice<u64>) -> Result<CudaSlice<u64>> {
    let count = u32::try_from(counts.len())?;
    ensure!(count > 0, "empty sparse prefix scan");
    let blocks = counts.len().div_ceil(1024);
    let output = d.stream.alloc_zeros::<u64>(counts.len() + 1)?;
    let totals = d.stream.alloc_zeros::<u64>(blocks)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("exclusive_scan_u64")?)
            .arg(counts)
            .arg(&output)
            .arg(&totals)
            .arg(&count)
            .launch(grid(blocks, 256))?;
    }
    if blocks > 1 {
        let offsets = exclusive_scan(d, &totals)?;
        unsafe {
            d.stream
                .launch_builder(&module.load_function("add_scan_offsets_u64")?)
                .arg(&output)
                .arg(&offsets)
                .arg(&count)
                .launch(flat(counts.len() + 1))?;
        }
    }
    Ok(output)
}

fn upload_sites(d: &Device, sites: &SparseSites) -> Result<[CudaSlice<i32>; 2]> {
    let coordinates = d.stream.clone_htod(sites.coordinates().as_flattened())?;
    let sorted: Vec<_> = sites
        .sorted_rows()
        .iter()
        .flat_map(|&(p, row)| [p[0], p[1], p[2], p[3], row])
        .collect();
    Ok([coordinates, d.stream.clone_htod(&sorted)?])
}

pub(crate) fn cube_mesh(
    d: &Device,
    sites: SparseSites,
    cubes: CudaSlice<u32>,
    positions: CudaSlice<f32>,
    scalar: CudaSlice<f32>,
    budget: [usize; 2],
) -> Result<SparseMesh> {
    let rows = sites.coordinates().len();
    ensure!(
        cubes.len()
            == rows
                .checked_mul(8)
                .context("cube incidence extent overflow")?
            && positions.len()
                == scalar
                    .len()
                    .checked_mul(3)
                    .context("corner position extent overflow")?
            && !scalar.is_empty(),
        "invalid cube extraction inputs"
    );
    let n = u32::try_from(rows)?;
    let u = u32::try_from(scalar.len())?;
    let edges = rows.checked_mul(3).context("cube edge count overflow")?;
    u32::try_from(edges)?;
    let live = rows
        .checked_mul(216)
        .and_then(|v| v.checked_add(4096))
        .context("cube extraction workspace overflow")?;
    d.workspace_budget(budget[0], budget[1], live)?;
    let [coordinates, sorted] = upload_sites(d, &sites)?;
    let table = crate::sparse::cube_topology()?;
    let table = d.stream.clone_htod(table.as_flattened())?;
    let masks = d.stream.alloc_zeros::<u8>(rows)?;
    let resolved = d.stream.alloc_zeros::<u8>(rows)?;
    let counts = d.stream.alloc_zeros::<u64>(rows)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("cube_sign_masks_f32")?)
            .arg(&scalar)
            .arg(&cubes)
            .arg(&n)
            .arg(&u)
            .arg(&masks)
            .launch(flat(rows))?;
        d.stream
            .launch_builder(&module.load_function("cube_resolved_components_u8")?)
            .arg(&coordinates)
            .arg(&sorted)
            .arg(&masks)
            .arg(&table)
            .arg(&n)
            .arg(&resolved)
            .arg(&counts)
            .launch(flat(rows))?;
    }
    drop(masks);
    let offsets = exclusive_scan(d, &counts)?;
    drop(counts);
    let vertices_count = d.stream.clone_dtoh(&offsets.slice(rows..rows + 1))?[0];
    ensure!(
        vertices_count <= rows as u64 * 4 && vertices_count <= i32::MAX as u64,
        "cube vertex count exceeds indexing capacity"
    );
    if vertices_count == 0 {
        return Ok(SparseMesh {
            vertices: Vec::new(),
            faces: Vec::new(),
        });
    }
    let vertices = d
        .stream
        .alloc_zeros::<f32>(usize::try_from(vertices_count)? * 3)?;
    let quads = d.stream.alloc_zeros::<i32>(edges * 4)?;
    let face_counts = d.stream.alloc_zeros::<u64>(edges)?;
    let diagonal = d.stream.alloc_zeros::<u8>(edges)?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("cube_dual_vertices_f32")?)
            .arg(&positions)
            .arg(&scalar)
            .arg(&cubes)
            .arg(&resolved)
            .arg(&table)
            .arg(&offsets)
            .arg(&n)
            .arg(&u)
            .arg(&vertices_count)
            .arg(&vertices)
            .launch(flat(rows))?;
        d.stream
            .launch_builder(&module.load_function("sparse_cube_component_edge_ids")?)
            .arg(&coordinates)
            .arg(&sorted)
            .arg(&cubes)
            .arg(&scalar)
            .arg(&resolved)
            .arg(&table)
            .arg(&offsets)
            .arg(&n)
            .arg(&quads)
            .arg(&face_counts)
            .arg(&diagonal)
            .launch(flat(edges))?;
    }
    drop(coordinates);
    drop(sorted);
    drop(cubes);
    drop(positions);
    drop(scalar);
    drop(resolved);
    drop(table);
    drop(offsets);
    drop(sites);
    eprintln!(
        "SPARSE_CUBE_PLAN cubes={rows} corners={u} vertices={vertices_count} edge_slots={edges} workspace_bound_bytes={live}"
    );
    compact_quads(
        d,
        &module,
        vertices,
        quads,
        face_counts,
        QuadSplit::Fixed(diagonal),
        budget,
    )
}

enum QuadSplit {
    Product(CudaSlice<f32>),
    Fixed(CudaSlice<u8>),
}

pub(crate) fn dual_grid_mesh(
    d: &Device,
    sites: SparseSites,
    fields: CudaSlice<f32>,
    resolution: u32,
    budget: [usize; 2],
) -> Result<SparseMesh> {
    let rows = sites.coordinates().len();
    ensure!(
        fields.len() == rows * 7 && resolution > 0,
        "invalid dual grid fields"
    );
    let n = u32::try_from(rows)?;
    d.workspace_budget(budget[0], budget[1], rows * (4 * size_of::<f32>() + 3))?;
    let vertices = d.stream.alloc_zeros::<f32>(rows * 3)?;
    let flags = d.stream.alloc_zeros::<u8>(rows * 3)?;
    let weights = d.stream.alloc_zeros::<f32>(rows)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("dual_grid_head_f32")?)
            .arg(&fields)
            .arg(&n)
            .arg(&vertices)
            .arg(&flags)
            .arg(&weights)
            .launch(flat(rows))?;
        d.stream
            .launch_builder(&module.load_function("activation_f32")?)
            .arg(&weights)
            .arg(&weights)
            .arg(&n)
            .arg(&1u32)
            .arg(&3i32)
            .launch(flat(rows))?;
    }
    drop(fields);
    mesh_from_dual(d, sites, vertices, flags, weights, resolution, budget)
}

pub(crate) fn mesh_from_dual(
    d: &Device,
    sites: SparseSites,
    vertices: CudaSlice<f32>,
    flags: CudaSlice<u8>,
    weights: CudaSlice<f32>,
    resolution: u32,
    budget: [usize; 2],
) -> Result<SparseMesh> {
    let rows = sites.coordinates().len();
    ensure!(
        vertices.len() == rows * 3
            && flags.len() == rows * 3
            && weights.len() == rows
            && resolution > 0,
        "invalid dual grid attributes"
    );
    let edges = rows.checked_mul(3).context("edge count overflow")?;
    u32::try_from(edges)?;
    let n = u32::try_from(rows)?;
    let mut scan_bytes = 0usize;
    let mut blocks = edges;
    loop {
        blocks = blocks.div_ceil(1024);
        scan_bytes = scan_bytes
            .checked_add((2 * blocks + 1) * size_of::<u64>())
            .context("scan workspace overflow")?;
        if blocks == 1 {
            break;
        }
    }
    let row_bytes =
        (4 + 5) * size_of::<i32>() + 3 * 4 * size_of::<i32>() + 3 * 2 * size_of::<u64>();
    let live = rows
        .checked_mul(row_bytes)
        .and_then(|v| v.checked_add(scan_bytes + size_of::<u64>()))
        .context("mesh workspace overflow")?;
    d.workspace_budget(budget[0], budget[1], live)?;
    let [coordinates, sorted] = upload_sites(d, &sites)?;
    let quads = d.stream.alloc_zeros::<i32>(edges * 4)?;
    let counts = d.stream.alloc_zeros::<u64>(edges)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("sparse_grid_world_f32")?)
            .arg(&coordinates)
            .arg(&vertices)
            .arg(&n)
            .arg(&resolution)
            .arg(&-0.5f32)
            .launch(flat(rows * 3))?;
        d.stream
            .launch_builder(&module.load_function("sparse_cube_edge_ids")?)
            .arg(&coordinates)
            .arg(&sorted)
            .arg(&flags)
            .arg(&n)
            .arg(&quads)
            .arg(&counts)
            .launch(flat(edges))?;
    }
    drop(coordinates);
    drop(sorted);
    drop(flags);
    drop(sites);
    compact_quads(
        d,
        &module,
        vertices,
        quads,
        counts,
        QuadSplit::Product(weights),
        budget,
    )
}

fn compact_quads(
    d: &Device,
    module: &Arc<CudaModule>,
    vertices: CudaSlice<f32>,
    quads: CudaSlice<i32>,
    counts: CudaSlice<u64>,
    split: QuadSplit,
    budget: [usize; 2],
) -> Result<SparseMesh> {
    let edges = counts.len();
    let edge_count = u32::try_from(edges)?;
    let n = u32::try_from(vertices.len() / 3)?;
    ensure!(
        vertices.len().is_multiple_of(3) && quads.len() == edges * 4,
        "invalid quad mesh extents"
    );
    let offsets = exclusive_scan(d, &counts)?;
    drop(counts);
    let total = d.stream.clone_dtoh(&offsets.slice(edges..edges + 1))?[0];
    ensure!(
        total <= edges as u64 * 2 && total.is_multiple_of(2),
        "invalid mesh face count"
    );
    if total == 0 {
        return Ok(SparseMesh {
            vertices: d.stream.clone_dtoh(&vertices)?,
            faces: Vec::new(),
        });
    }
    let face_elements = usize::try_from(total)?
        .checked_mul(3)
        .context("face allocation overflow")?;
    let additional = face_elements
        .checked_mul(size_of::<i32>())
        .and_then(|v| v.checked_add(edges))
        .context("face workspace overflow")?;
    d.workspace_budget(budget[0], budget[1], additional)?;
    let diagonal = match split {
        QuadSplit::Fixed(diagonal) => {
            ensure!(diagonal.len() == edges, "quad diagonal extent mismatch");
            diagonal
        }
        QuadSplit::Product(weights) => {
            ensure!(weights.len() == n as usize, "quad weight extent mismatch");
            let diagonal = d.stream.alloc_zeros::<u8>(edges)?;
            unsafe {
                d.stream
                    .launch_builder(&module.load_function("quad_diagonal_f32")?)
                    .arg(&quads)
                    .arg(&offsets)
                    .arg(&weights)
                    .arg(&edge_count)
                    .arg(&n)
                    .arg(&diagonal)
                    .launch(flat(edges))?;
            }
            diagonal
        }
    };
    let faces = d.stream.alloc_zeros::<i32>(face_elements)?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("sparse_face_compact")?)
            .arg(&quads)
            .arg(&offsets)
            .arg(&diagonal)
            .arg(&edge_count)
            .arg(&total)
            .arg(&faces)
            .launch(flat(edges))?;
    }
    Ok(SparseMesh {
        vertices: d.stream.clone_dtoh(&vertices)?,
        faces: d.stream.clone_dtoh(&faces)?,
    })
}

pub struct SparseConv<T = f16> {
    device: Device,
    neighbours: CudaSlice<i32>,
    columns: Rc<CudaSlice<T>>,
    gather: CudaFunction,
    bias: CudaFunction,
    half_gather: Option<CudaFunction>,
    half_store: Option<CudaFunction>,
    rows: usize,
    channels: usize,
    tile: usize,
}
impl SparseConv {
    pub fn new(
        device: &Device,
        sites: &SparseSites,
        channels: usize,
        workspace_budget: usize,
    ) -> Result<Self> {
        let rows = sites.coordinates().len();
        let row_bytes = channels
            .checked_mul(27 * size_of::<f16>())
            .context("sparse gather width overflow")?;
        let map_bytes = rows
            .checked_mul(27 * size_of::<i32>())
            .context("sparse map size overflow")?;
        let (free, _) = device.ctx.mem_get_info()?;
        let available = workspace_budget
            .min(free)
            .checked_sub(map_bytes)
            .context("sparse map exceeds workspace budget")?;
        ensure!(
            channels > 0 && 27 * channels <= i32::MAX as usize,
            "invalid sparse gather width"
        );
        let tile = (available / row_bytes)
            .min(rows)
            .min(i32::MAX as usize / (27 * channels));
        ensure!(tile > 0, "sparse gather cannot fit one row");
        let columns = Rc::new(device.stream.alloc_zeros(tile * 27 * channels)?);
        Self::shared(device, sites, channels, columns)
    }
    pub(crate) fn shared(
        device: &Device,
        sites: &SparseSites,
        channels: usize,
        columns: Rc<CudaSlice<f16>>,
    ) -> Result<Self> {
        let rows = sites.coordinates().len();
        ensure!(
            channels > 0 && channels <= i32::MAX as usize / 27,
            "invalid sparse gather width"
        );
        let tile = (columns.len() / (27 * channels))
            .min(rows)
            .min(i32::MAX as usize / (27 * channels));
        ensure!(tile > 0, "sparse gather cannot fit one row");
        let maps = sites.neighbours_3x3()?;
        let neighbours = device.stream.clone_htod(maps.as_flattened())?;
        Ok(Self {
            device: device.clone(),
            neighbours,
            columns,
            gather: cubin::module(&device.ctx, "vision_ops")?.load_function("sparse_gather_f16")?,
            bias: cubin::module(&device.ctx, "q38_common")?.load_function("q38_bias_fill_f16")?,
            half_gather: None,
            half_store: None,
            rows,
            channels,
            tile,
        })
    }
    pub fn run(
        &self,
        blas: &Blas,
        input: &CudaSlice<f16>,
        weight: Tensor<f16, 2>,
        bias: Tensor<f16, 1>,
        output: &mut CudaSlice<f16>,
    ) -> Result<()> {
        weight.validate()?;
        bias.validate()?;
        let width = 27 * self.channels;
        let out = weight.shape[0] as usize;
        let stream = &self.device.stream;
        ensure!(
            weight.shape[1] as usize == width
                && weight.stride == [width as u64, 1]
                && bias.shape == [out as u32]
                && bias.stride == [1],
            "sparse weights must retain published [Co,27Ci] layout"
        );
        ensure!(
            input.len() == self.rows * self.channels && output.len() == self.rows * out,
            "sparse feature extent mismatch"
        );
        let input_view = view(
            input,
            stream,
            0,
            [self.rows as u32, self.channels as u32],
            [self.channels as u64, 1],
        )?;
        let output_view = view(
            output,
            stream,
            0,
            [self.rows as u32, out as u32],
            [out as u64, 1],
        )?;
        no_overlap(&input_view, &output_view, "sparse input", "sparse output")?;
        no_overlap(&weight, &output_view, "sparse weight", "sparse output")?;
        no_overlap(&bias, &output_view, "sparse bias", "sparse output")?;
        for first in (0..self.rows).step_by(self.tile) {
            let rows = self.tile.min(self.rows - first);
            unsafe {
                stream
                    .launch_builder(&self.gather)
                    .arg(input)
                    .arg(&self.neighbours)
                    .arg(self.columns.as_ref())
                    .arg(&(self.rows as u32))
                    .arg(&(self.channels as u32))
                    .arg(&(first as u32))
                    .arg(&(rows as u32))
                    .launch(flat(rows * width))?;
            }
            let x = view(
                &self.columns,
                stream,
                0,
                [rows as u32, width as u32],
                [width as u64, 1],
            )?;
            let y = view(
                output,
                stream,
                first * out,
                [rows as u32, out as u32],
                [out as u64, 1],
            )?;
            unsafe {
                stream
                    .launch_builder(&self.bias)
                    .arg(&y.ptr)
                    .arg(&bias.ptr)
                    .arg(&((rows * out) as u64))
                    .arg(&(out as u32))
                    .launch(flat(rows * out))?;
            }
            blas.linear_f16(weight, x, &mut TensorMut(y), true)?;
        }
        Ok(())
    }
}

impl<T> SparseConv<T> {
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.neighbours.len() * size_of::<i32>() + self.columns.len() * size_of::<T>()
    }
    #[cfg(test)]
    pub(crate) fn replace_neighbours(&mut self, maps: &[[i32; 27]]) -> Result<()> {
        ensure!(
            maps.len() == self.rows
                && maps
                    .iter()
                    .flatten()
                    .all(|&r| r >= -1 && r < self.rows as i32),
            "invalid sparse neighbor map"
        );
        self.neighbours = self.device.stream.clone_htod(maps.as_flattened())?;
        Ok(())
    }
}

impl SparseConv<f32> {
    pub fn promoted(
        device: &Device,
        sites: &SparseSites,
        channels: [usize; 2],
        workspace_budget: usize,
    ) -> Result<Self> {
        let rows = sites.coordinates().len();
        let [input, output] = channels;
        ensure!(
            input > 0
                && output > 0
                && input <= i32::MAX as usize / 27
                && output <= i32::MAX as usize,
            "invalid promoted sparse channels"
        );
        let width = 27 * input;
        let map_bytes = rows
            .checked_mul(27 * size_of::<i32>())
            .context("sparse map size overflow")?;
        let row_bytes = width
            .checked_add(output)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .context("sparse workspace width overflow")?;
        let (free, _) = device.ctx.mem_get_info()?;
        let available = workspace_budget
            .min(free)
            .checked_sub(map_bytes)
            .context("sparse map exceeds workspace budget")?;
        let tile = (available / row_bytes)
            .min(rows)
            .min(i32::MAX as usize / width)
            .min(i32::MAX as usize / output);
        ensure!(tile > 0, "promoted sparse workspace cannot fit one row");
        let columns = Rc::new(device.stream.alloc_zeros::<f32>(tile * (width + output))?);
        let maps = sites.neighbours_3x3()?;
        let neighbours = device.stream.clone_htod(maps.as_flattened())?;
        let module = cubin::module(&device.ctx, "vision_ops")?;
        Ok(Self {
            device: device.clone(),
            neighbours,
            columns,
            gather: module.load_function("sparse_gather_f32")?,
            bias: module.load_function("sparse_conv_store_f32")?,
            half_gather: Some(module.load_function("sparse_gather_f16_to_f32")?),
            half_store: Some(module.load_function("sparse_conv_store_f16")?),
            rows,
            channels: input,
            tile,
        })
    }
    pub fn run_promoted<I: DeviceRepr + Copy + 'static, O: DeviceRepr + Copy + 'static>(
        &self,
        blas: &Blas,
        input: Tensor<I, 2>,
        weight: Tensor<f32, 2>,
        bias: Tensor<f32, 1>,
        output: &mut TensorMut<O, 2>,
    ) -> Result<()> {
        use std::any::TypeId;
        weight.validate()?;
        bias.validate()?;
        let input_half = TypeId::of::<I>() == TypeId::of::<f16>();
        let output_half = TypeId::of::<O>() == TypeId::of::<f16>();
        ensure!(
            (input_half || TypeId::of::<I>() == TypeId::of::<f32>())
                && (output_half || TypeId::of::<O>() == TypeId::of::<f32>()),
            "promoted sparse features must be F16 or F32"
        );
        let width = weight.shape[1] as usize;
        let out = weight.shape[0] as usize;
        ensure!(
            width > 0
                && width.is_multiple_of(27)
                && width / 27 <= self.channels
                && out > 0
                && weight.stride == [width as u64, 1]
                && bias.shape == [out as u32]
                && bias.stride == [1],
            "promoted sparse weights must retain published [Co,27Ci] layout"
        );
        let channels = width / 27;
        input.validate()?;
        output.0.validate()?;
        ensure!(
            input.shape == [self.rows as u32, channels as u32]
                && input.stride == [channels as u64, 1]
                && output.0.shape == [self.rows as u32, out as u32]
                && output.0.stride == [out as u64, 1],
            "promoted sparse feature extent mismatch"
        );
        let result_offset = self.tile * 27 * self.channels;
        ensure!(
            self.columns.len() - result_offset >= self.tile * out,
            "promoted sparse result tile exceeds workspace"
        );
        let stream = &self.device.stream;
        no_overlap(&input, &output.0, "sparse input", "sparse output")?;
        no_overlap(&weight, &output.0, "sparse weight", "sparse output")?;
        no_overlap(&bias, &output.0, "sparse bias", "sparse output")?;
        let gather = if input_half {
            self.half_gather
                .as_ref()
                .context("promoted half gather missing")?
        } else {
            &self.gather
        };
        let store = if output_half {
            self.half_store
                .as_ref()
                .context("promoted half store missing")?
        } else {
            &self.bias
        };
        for first in (0..self.rows).step_by(self.tile) {
            let rows = self.tile.min(self.rows - first);
            unsafe {
                stream
                    .launch_builder(gather)
                    .arg(&input.ptr)
                    .arg(&self.neighbours)
                    .arg(self.columns.as_ref())
                    .arg(&(self.rows as u32))
                    .arg(&(channels as u32))
                    .arg(&(first as u32))
                    .arg(&(rows as u32))
                    .launch(flat(rows * width))?;
            }
            let x = view(
                &self.columns,
                stream,
                0,
                [rows as u32, width as u32],
                [width as u64, 1],
            )?;
            let product = view(
                &self.columns,
                stream,
                result_offset,
                [rows as u32, out as u32],
                [out as u64, 1],
            )?;
            blas.linear_f32(weight, x, &mut TensorMut(product))?;
            unsafe {
                stream
                    .launch_builder(store)
                    .arg(&product.ptr)
                    .arg(&bias.ptr)
                    .arg(&output.0.ptr)
                    .arg(&(first as u64))
                    .arg(&(rows as u32))
                    .arg(&(out as u32))
                    .launch(flat(rows * out))?;
            }
        }
        Ok(())
    }
}
