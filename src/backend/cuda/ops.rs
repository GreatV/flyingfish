use super::cubin;
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};
use std::sync::Arc;

pub struct Ops {
    pub embed: CudaFunction,
    pub norm: CudaFunction,
    pub add_norm: CudaFunction,
    pub rope: CudaFunction,
    pub rope_float: CudaFunction,
    pub kv_write: CudaFunction,
    pub act: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_rows: CudaFunction,
    pub advance: CudaFunction,
    pub residual: CudaFunction,
}

impl Ops {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let embed = cubin::module(ctx, "embed")?;
        let norm = cubin::module(ctx, "norm")?;
        let rope = cubin::module(ctx, "rope")?;
        let act = cubin::module(ctx, "act")?;
        let sample = cubin::module(ctx, "sample")?;
        Ok(Self {
            embed: embed.load_function("embed")?,
            norm: norm.load_function("rms_norm")?,
            add_norm: norm.load_function("add_rmsnorm")?,
            rope: rope.load_function("rope")?,
            rope_float: rope.load_function("rope_float")?,
            kv_write: rope.load_function("kv_write")?,
            act: act.load_function("silu_mul")?,
            argmax: sample.load_function("argmax")?,
            argmax_rows: sample.load_function("argmax_rows_bf16")?,
            advance: sample.load_function("advance")?,
            residual: norm.load_function("residual")?,
        })
    }
}

pub fn grid(blocks: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn flat(n: usize) -> LaunchConfig {
    grid(n.div_ceil(256), 256)
}
