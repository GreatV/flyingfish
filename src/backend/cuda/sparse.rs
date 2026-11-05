//! Published sparse [Co,kx,ky,kz,Ci] weights with bounded gather workspace.
use super::{
    Device,
    blas::Blas,
    contracts::{Tensor, TensorMut, no_overlap, view},
    cubin,
    ops::flat,
};
use crate::{
    backend::gpu::{CudaFunction, CudaSlice, PushKernelArg},
    sparse::SparseSites,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::rc::Rc;

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
    #[cfg(test)]
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
    #[cfg(test)]
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
