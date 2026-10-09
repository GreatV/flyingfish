use super::cubin;
use crate::config::Config;
use anyhow::{Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::bf16;
use std::sync::Arc;

use crate::backend::setup::AttentionPlan;

pub struct Attention {
    first: CudaFunction,
    second: CudaFunction,
    scratch: CudaSlice<f32>,
    first_launch: LaunchConfig,
    second_launch: LaunchConfig,
    head_stride: i32,
    chunk: usize,
    plan: AttentionPlan,
}

impl Attention {
    pub fn phase1_blocks_per_sm(&self) -> Result<usize> {
        Ok(self.first.occupancy_max_active_blocks_per_multiprocessor(
            self.first_launch.block_dim.0,
            self.first_launch.shared_mem_bytes as usize,
            None,
        )? as usize)
    }

    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        c: &Config,
        capacity: usize,
        settings: AttentionPlan,
    ) -> Result<Self> {
        Self::candidate(ctx, stream, c, capacity, settings, true)
    }

    pub fn candidate(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        c: &Config,
        capacity: usize,
        settings: AttentionPlan,
        log: bool,
    ) -> Result<Self> {
        ensure!(
            c.num_attention_heads == 16 && c.num_key_value_heads == 2 && c.head_dim == 128,
            "decode attention requires 16 Q heads, 2 KV heads and head_dim 128"
        );
        ensure!(
            [1, 2, 4, 8].contains(&settings.qpack),
            "attention pack must be 1, 2, 4 or 8"
        );
        let chunk = settings.chunk;
        ensure!(
            chunk > 0 && chunk <= i32::MAX as usize - capacity,
            "attention chunk and cache length must fit a positive int"
        );
        ensure!(
            capacity <= i32::MAX as usize / 128,
            "attention KV head stride exceeds int range"
        );
        ensure!(
            settings.threads >= 32
                && settings.threads <= 1024
                && settings.threads.is_multiple_of(32),
            "attention threads must be a multiple of 32 in 32..=1024"
        );
        ensure!(
            settings.merge_threads >= 128
                && settings.merge_threads <= 1024
                && settings.merge_threads.is_multiple_of(128),
            "attention merge threads must be a multiple of 128 in 128..=1024"
        );
        let chunks = capacity.div_ceil(chunk);
        let first_smem = settings.qpack * (settings.threads as usize / 32) * 130 * 4;
        let second_smem = (2 * chunks + settings.merge_threads as usize) * 4;
        ensure!(
            first_smem <= 48 * 1024 && second_smem <= 48 * 1024,
            "decode attention shared memory exceeds 48 KiB: phase1={first_smem}, phase2={second_smem}"
        );
        let module = cubin::module(ctx, "attention")?;
        let first = module.load_function(&format!("flash_decode_phase1_q{}", settings.qpack))?;
        let second = module.load_function("flash_decode_phase2")?;
        ensure!(
            settings.threads as i32 <= first.max_threads_per_block()?
                && settings.merge_threads as i32 <= second.max_threads_per_block()?,
            "decode attention block size exceeds kernel limit"
        );
        let scratch = stream.alloc_zeros(16 * chunks * 130)?;
        let first_launch = LaunchConfig {
            grid_dim: (chunks as u32, (16 / settings.qpack) as u32, 1),
            block_dim: (settings.threads, 1, 1),
            shared_mem_bytes: first_smem as u32,
        };
        let second_launch = LaunchConfig {
            grid_dim: (16, 1, 1),
            block_dim: (settings.merge_threads, 1, 1),
            shared_mem_bytes: second_smem as u32,
        };
        if log {
            eprintln!(
                "{}",
                serde_json::json!({"decode_attention_plan":{
            "settings":settings,"selected_chunk":chunk,"max_chunks":chunks,"scratch_bytes":16*chunks*130*4,
            "phase1_smem_bytes":first_smem,"phase2_smem_bytes":second_smem,
            "length":"device scalar counts KV positions including current token"}})
            );
        }
        Ok(Self {
            plan: settings,
            first,
            second,
            scratch,
            first_launch,
            second_launch,
            head_stride: (capacity * 128) as i32,
            chunk,
        })
    }

    pub fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        output: &mut CudaSlice<bf16>,
        length: &CudaSlice<i32>,
    ) -> Result<()> {
        let chunk = self.chunk as i32;
        let q = qkv.slice(..2048);
        unsafe {
            stream
                .launch_builder(&self.first)
                .arg(&q)
                .arg(k)
                .arg(v)
                .arg(&mut self.scratch)
                .arg(length)
                .arg(&16i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&128i32)
                .arg(&self.head_stride)
                .arg(&128i32)
                .arg(&self.head_stride)
                .arg(&128i32)
                .arg(&chunk)
                .launch(self.first_launch)?;
            stream
                .launch_builder(&self.second)
                .arg(&mut self.scratch)
                .arg(output)
                .arg(length)
                .arg(&16i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&128i32)
                .arg(&chunk)
                .launch(self.second_launch)?;
        }
        Ok(())
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    pub fn plan(&self) -> AttentionPlan {
        self.plan
    }
}
