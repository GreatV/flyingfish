use super::cubin;
use crate::config::Config;
use anyhow::{Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use half::bf16;
use std::sync::Arc;

use crate::backend::setup::AttentionPlan;

pub struct Attention {
    first: CudaFunction,
    second: CudaFunction,
    window_first: CudaFunction,
    sink_second: CudaFunction,
    scratch: CudaSlice<f32>,
    first_launch: LaunchConfig,
    second_launch: LaunchConfig,
    head_stride: i32,
    chunk: usize,
    max_chunks: usize,
    q_heads: i32,
    kv_heads: i32,
    plan: AttentionPlan,
}

pub struct GqaDecodeInput<'a> {
    pub q: &'a CudaSlice<bf16>,
    pub q_offset: usize,
    pub k: &'a CudaSlice<bf16>,
    pub v: &'a CudaSlice<bf16>,
    pub output: &'a mut CudaSlice<bf16>,
    pub output_offset: usize,
    pub length: &'a CudaSlice<i32>,
    pub sinks: Option<&'a CudaSlice<bf16>>,
    pub window: Option<SlidingWindow>,
}

pub struct SlidingWindow {
    pub first_pos: i32,
    pub tokens: i32,
    pub head_stride: i32,
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
        ensure!(c.head_dim == 128, "decode attention requires head_dim 128");
        let shape = (c.num_attention_heads, c.num_key_value_heads);
        ensure!(
            shape == (16, 2) || shape == (32, 4),
            "decode attention requires Q/KV heads 16/2 or 32/4, got {shape:?}"
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
                && settings.merge_threads.is_power_of_two(),
            "attention merge threads must be a power of two in 128..=1024"
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
        let window_first =
            module.load_function(&format!("flash_decode_window_phase1_q{}", settings.qpack))?;
        let sink_second = module.load_function("flash_decode_sink_phase2")?;
        ensure!(
            settings.threads as i32 <= first.max_threads_per_block()?
                && settings.merge_threads as i32 <= second.max_threads_per_block()?
                && settings.threads as i32 <= window_first.max_threads_per_block()?
                && settings.merge_threads as i32 <= sink_second.max_threads_per_block()?,
            "decode attention block size exceeds kernel limit"
        );
        let scratch = stream.alloc_zeros(c.num_attention_heads * chunks * 130)?;
        let first_launch = LaunchConfig {
            grid_dim: (
                chunks as u32,
                (c.num_attention_heads / settings.qpack) as u32,
                1,
            ),
            block_dim: (settings.threads, 1, 1),
            shared_mem_bytes: first_smem as u32,
        };
        let second_launch = LaunchConfig {
            grid_dim: (c.num_attention_heads as u32, 1, 1),
            block_dim: (settings.merge_threads, 1, 1),
            shared_mem_bytes: second_smem as u32,
        };
        if log {
            eprintln!(
                "{}",
                serde_json::json!({"decode_attention_plan":{
            "settings":settings,"selected_chunk":chunk,"max_chunks":chunks,
            "scratch_bytes":c.num_attention_heads*chunks*130*4,
            "phase1_smem_bytes":first_smem,"phase2_smem_bytes":second_smem,
            "length":"device scalar counts KV positions including current token"}})
            );
        }
        Ok(Self {
            first,
            second,
            window_first,
            sink_second,
            scratch,
            first_launch,
            second_launch,
            head_stride: (capacity * 128) as i32,
            chunk,
            max_chunks: chunks,
            q_heads: c.num_attention_heads as i32,
            kv_heads: c.num_key_value_heads as i32,
            plan: settings,
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
        self.run_with(
            stream,
            GqaDecodeInput {
                q: qkv,
                q_offset: 0,
                k,
                v,
                output,
                output_offset: 0,
                length,
                sinks: None,
                window: None,
            },
        )
    }

    pub fn run_with(&mut self, stream: &Arc<CudaStream>, input: GqaDecodeInput<'_>) -> Result<()> {
        let GqaDecodeInput {
            q,
            q_offset,
            k,
            v,
            output,
            output_offset,
            length,
            sinks,
            window,
        } = input;
        let chunk = self.chunk as i32;
        let gqa = self.q_heads / self.kv_heads;
        ensure!(
            q.len() - q_offset >= self.q_heads as usize * 128
                && output.len() - output_offset >= self.q_heads as usize * 128,
            "decode attention Q/output offsets exceed their buffers"
        );
        let q = q.slice(q_offset..);
        let mut output = output.slice_mut(output_offset..);
        unsafe {
            if let Some(window) = window {
                stream
                    .launch_builder(&self.window_first)
                    .arg(&q)
                    .arg(k)
                    .arg(v)
                    .arg(&mut self.scratch)
                    .arg(length)
                    .arg(&self.q_heads)
                    .arg(&self.kv_heads)
                    .arg(&gqa)
                    .arg(&128i32)
                    .arg(&window.head_stride)
                    .arg(&128i32)
                    .arg(&window.head_stride)
                    .arg(&128i32)
                    .arg(&chunk)
                    .arg(&window.first_pos)
                    .arg(&window.tokens)
                    .launch(self.first_launch)?;
            } else {
                stream
                    .launch_builder(&self.first)
                    .arg(&q)
                    .arg(k)
                    .arg(v)
                    .arg(&mut self.scratch)
                    .arg(length)
                    .arg(&self.q_heads)
                    .arg(&self.kv_heads)
                    .arg(&gqa)
                    .arg(&128i32)
                    .arg(&self.head_stride)
                    .arg(&128i32)
                    .arg(&self.head_stride)
                    .arg(&128i32)
                    .arg(&chunk)
                    .launch(self.first_launch)?;
            }
            if let Some(sinks) = sinks {
                stream
                    .launch_builder(&self.sink_second)
                    .arg(&mut self.scratch)
                    .arg(&mut output)
                    .arg(sinks)
                    .arg(length)
                    .arg(&self.q_heads)
                    .arg(&self.kv_heads)
                    .arg(&gqa)
                    .arg(&128i32)
                    .arg(&chunk)
                    .launch(self.second_launch)?;
            } else {
                stream
                    .launch_builder(&self.second)
                    .arg(&mut self.scratch)
                    .arg(&mut output)
                    .arg(length)
                    .arg(&self.q_heads)
                    .arg(&self.kv_heads)
                    .arg(&gqa)
                    .arg(&128i32)
                    .arg(&chunk)
                    .launch(self.second_launch)?;
            }
        }
        Ok(())
    }

    pub fn set_length(
        &self,
        stream: &Arc<CudaStream>,
        value: i32,
        length: &mut CudaSlice<i32>,
    ) -> Result<()> {
        ensure!(
            value > 0 && usize::try_from(value)? <= self.max_chunks * self.chunk,
            "decode attention length {value} exceeds launched chunks {} chunk {}",
            self.max_chunks,
            self.chunk
        );
        let (ptr, _length) = length.device_ptr_mut(stream);
        stream.context().bind_to_thread()?;
        let bits = u32::try_from(value)?;
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ptr, bits, 1, stream.cu_stream()).result()?;
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
