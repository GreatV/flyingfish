//! Published sparse [Co,kx,ky,kz,Ci] weights with bounded gather workspace.
use super::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, no_overlap, view},
    cubin,
    ops::{flat, grid},
};
use crate::{
    backend::gpu::{CudaFunction, CudaSlice, PushKernelArg},
    sparse::{SparseMesh, SparseSites},
};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::rc::Rc;

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
    let edge_count = u32::try_from(edges)?;
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
    let coordinates = d.stream.clone_htod(sites.coordinates().as_flattened())?;
    let sorted_host: Vec<_> = sites
        .sorted_rows()
        .iter()
        .flat_map(|&(p, row)| [p[0], p[1], p[2], p[3], row])
        .collect();
    let sorted = d.stream.clone_htod(&sorted_host)?;
    drop(sorted_host);
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
    let offsets = exclusive_scan(d, &counts)?;
    drop(counts);
    let total = d.stream.clone_dtoh(&offsets.slice(edges..edges + 1))?[0];
    ensure!(
        total <= rows as u64 * 6 && total.is_multiple_of(2),
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
    let diagonal = d.stream.alloc_zeros::<u8>(edges)?;
    let faces = d.stream.alloc_zeros::<i32>(face_elements)?;
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

pub struct SparseConv {
    device: Device,
    neighbours: CudaSlice<i32>,
    columns: Rc<CudaSlice<f16>>,
    gather: CudaFunction,
    bias: CudaFunction,
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
            rows,
            channels,
            tile,
        })
    }
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.neighbours.len() * size_of::<i32>() + self.columns.len() * size_of::<f16>()
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
