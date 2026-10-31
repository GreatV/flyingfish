use super::cubin;
use crate::backend::gpu::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

use crate::backend::gpu::{DevicePtr, DevicePtrMut};
use crate::backend::setup::{AttentionPlan, check_decode_attention_shape};

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

pub struct GqaDecodeInput<'a, L: DevicePtr<i32> = CudaSlice<i32>> {
    pub q: &'a CudaSlice<bf16>,
    pub q_offset: usize,
    pub k: &'a CudaSlice<bf16>,
    pub v: &'a CudaSlice<bf16>,
    pub output: &'a mut CudaSlice<bf16>,
    pub output_offset: usize,
    pub length: &'a L,
    pub sinks: Option<&'a CudaSlice<bf16>>,
    pub window: Option<SlidingWindow<'a>>,
}

pub struct SlidingWindow<'a> {
    pub position: &'a CudaSlice<i32>,
    pub head_stride: i32,
}

pub(super) struct RingDecode {
    first: CudaFunction,
    merge: CudaFunction,
    first_launch: LaunchConfig,
    merge_launch: LaunchConfig,
    heads: u32,
    kv_heads: u32,
    ring: u32,
    chunk: u32,
    scale: f32,
}

pub(super) struct RingDecodeInput<'a> {
    pub q: super::contracts::Tensor<bf16, 3>,
    pub k: super::contracts::Tensor<bf16, 3>,
    pub v: super::contracts::Tensor<bf16, 3>,
    pub position: super::contracts::Tensor<i32, 1>,
    pub out: &'a mut super::contracts::TensorMut<bf16, 3>,
    pub lse: &'a mut super::contracts::TensorMut<f32, 1>,
    pub scratch: super::contracts::Scratch,
}

impl RingDecode {
    pub fn workspace(heads: usize, ring: usize) -> Result<usize> {
        ensure!(
            heads > 0 && ring >= 32 && ring.is_multiple_of(32),
            "invalid ring decode geometry"
        );
        heads
            .checked_mul(ring.div_ceil(32))
            .and_then(|n| n.checked_mul(258 * 4))
            .filter(|&n| n <= i32::MAX as usize)
            .context("ring decode workspace overflow")
    }

    pub fn new(
        ctx: &Arc<CudaContext>,
        heads: u32,
        kv_heads: u32,
        ring: u32,
        scale: f32,
    ) -> Result<Self> {
        Self::workspace(heads as usize, ring as usize)?;
        ensure!(
            kv_heads > 0 && heads.is_multiple_of(kv_heads) && ring <= i32::MAX as u32 / 256,
            "invalid ring decode heads/stride"
        );
        ensure!(
            scale.is_finite() && scale > 0.0,
            "invalid ring decode scale"
        );
        #[cfg(feature = "cuda")]
        use cudarc::driver::sys::{CUdevice_attribute as A, CUfunction_attribute as F};
        #[cfg(feature = "cuda")]
        let (sms, warp, max_x, max_y, shared_limit) = (
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
            ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
        );
        #[cfg(feature = "rocm")]
        let (sms, warp, max_x, max_y, shared_limit) = (
            ctx.properties.compute_units,
            ctx.properties.warp,
            ctx.properties.max_grid_x,
            ctx.properties.max_grid_y,
            ctx.properties.shared_bytes,
        );
        ensure!(
            sms > 0 && warp == 32 && max_x > 0 && max_y > 0 && shared_limit > 0,
            "ring decode requires positive SM/CU count and 32-lane warps"
        );
        let sms = sms as usize;
        let module = cubin::module(ctx, "attention")?;
        let merge = module.load_function("flash_decode_ring_d256_merge")?;
        let mut selected = None;
        for pack in [1u32, 2, 4, 8] {
            if !(heads / kv_heads).is_multiple_of(pack) {
                continue;
            }
            let first = module.load_function(&format!("flash_decode_ring_d256_q{pack}"))?;
            let shared = pack * 4640;
            #[cfg(feature = "cuda")]
            let (regs, local, static_shared, threads) = (
                first.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
                first.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
                first.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
                first.get_attribute(F::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            );
            #[cfg(feature = "rocm")]
            let (regs, local, static_shared, threads) = (
                first.registers_per_thread()?,
                first.local_size_bytes()?,
                first.shared_size_bytes()?,
                first.max_threads_per_block()?,
            );
            ensure!(
                regs > 0
                    && regs <= 255
                    && local == 0
                    && threads >= 128
                    && shared as i32 + static_shared <= shared_limit,
                "ring decode resource limit pack={pack} regs={regs} local={local} shared={shared}+{static_shared}"
            );
            let resident =
                first.occupancy_max_active_blocks_per_multiprocessor(128, shared as usize, None)?
                    as usize;
            ensure!(resident > 0, "ring decode zero occupancy pack={pack}");
            let slots = sms
                .checked_mul(resident)
                .context("ring decode occupancy overflow")?;
            eprintln!(
                "{}",
                serde_json::json!({"ring_decode_resources":{
                    "pack":pack,"sms":sms,"registers":regs,"local_bytes":local,
                    "static_shared_bytes":static_shared,"dynamic_shared_bytes":shared,"resident_blocks":resident
                }})
            );
            for chunk in (32..=ring).step_by(32) {
                let parts = ring.div_ceil(chunk);
                let blocks = (heads / pack) as usize * parts as usize;
                ensure!(
                    parts <= max_x as u32 && heads / pack <= max_y as u32,
                    "ring decode grid exceeds device bounds"
                );
                let score = (
                    blocks.min(sms),
                    blocks.min(slots) as f64 / slots as f64,
                    pack,
                    std::cmp::Reverse(parts),
                );
                if selected.as_ref().is_none_or(|(best, _)| score > *best) {
                    selected = Some((
                        score,
                        Self {
                            first: first.clone(),
                            merge: merge.clone(),
                            first_launch: LaunchConfig {
                                grid_dim: (parts, heads / pack, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: shared,
                            },
                            merge_launch: LaunchConfig {
                                grid_dim: (heads, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: (2 * parts + 256) * 4,
                            },
                            heads,
                            kv_heads,
                            ring,
                            chunk,
                            scale,
                        },
                    ));
                }
            }
        }
        let (score, plan) = selected.context("ring decode has no head pack")?;
        #[cfg(feature = "cuda")]
        let (regs, local, shared, threads) = (
            merge.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            merge.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            merge.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
            merge.get_attribute(F::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        );
        #[cfg(feature = "rocm")]
        let (regs, local, shared, threads) = (
            merge.registers_per_thread()?,
            merge.local_size_bytes()?,
            merge.shared_size_bytes()?,
            merge.max_threads_per_block()?,
        );
        ensure!(
            regs > 0
                && regs <= 255
                && local == 0
                && threads >= 256
                && shared + plan.merge_launch.shared_mem_bytes as i32 <= shared_limit,
            "ring decode merge resource limit"
        );
        let merge_resident = merge.occupancy_max_active_blocks_per_multiprocessor(
            256,
            plan.merge_launch.shared_mem_bytes as usize,
            None,
        )?;
        ensure!(merge_resident > 0, "ring decode merge zero occupancy");
        eprintln!(
            "{}",
            serde_json::json!({"ring_decode_plan":{
                "head_dim":256,"heads":heads,"kv_heads":kv_heads,"ring":ring,"scale":scale,
                "pack":score.2,"chunk":plan.chunk,"parts":plan.first_launch.grid_dim.0,
                "blocks":plan.first_launch.grid_dim.0 * plan.first_launch.grid_dim.1,
                "sms":sms,"resident_fraction":score.1,"workspace_bytes":plan.workspace_bytes(),
                "merge_registers":regs,"merge_local_bytes":local,"merge_static_shared_bytes":shared,
                "merge_dynamic_shared_bytes":plan.merge_launch.shared_mem_bytes,"merge_resident_blocks":merge_resident
            }})
        );
        Ok(plan)
    }

    fn workspace_bytes(&self) -> u64 {
        u64::from(self.heads) * u64::from(self.first_launch.grid_dim.0) * 258 * 4
    }

    pub fn run(&self, stream: &Arc<CudaStream>, b: RingDecodeInput<'_>) -> Result<()> {
        use super::contracts::no_overlap;
        for t in [b.q, b.k, b.v, b.out.0] {
            t.validate()?;
        }
        b.position.validate()?;
        b.lse.0.validate()?;
        ensure!(
            b.q.shape == [1, self.heads, 256]
                && b.out.0.shape == b.q.shape
                && b.q.stride[1..] == [256, 1]
                && b.out.0.stride[1..] == [256, 1],
            "ring decode Q/output shape"
        );
        for t in [b.k, b.v] {
            ensure!(
                t.shape == [self.kv_heads, self.ring, 256]
                    && t.stride == [u64::from(self.ring) * 256, 256, 1],
                "ring decode KV shape"
            );
        }
        ensure!(
            b.position.shape == [1] && b.lse.0.shape == [self.heads] && b.lse.0.stride == [1],
            "ring decode position/LSE shape"
        );
        b.scratch.validate(self.workspace_bytes())?;
        for t in [b.q, b.k, b.v] {
            ensure!(
                t.ptr.is_multiple_of(16),
                "ring decode vector input alignment"
            );
            no_overlap(&t, &b.out.0, "ring input", "output")?;
            no_overlap(&t, &b.lse.0, "ring input", "LSE")?;
        }
        no_overlap(&b.out.0, &b.lse.0, "output", "LSE")?;
        no_overlap(&b.position, &b.out.0, "position", "output")?;
        no_overlap(&b.position, &b.lse.0, "position", "LSE")?;
        let spans = [
            (b.q.ptr, b.q.span_bytes),
            (b.k.ptr, b.k.span_bytes),
            (b.v.ptr, b.v.span_bytes),
            (b.position.ptr, b.position.span_bytes),
            (b.out.0.ptr, b.out.0.span_bytes),
            (b.lse.0.ptr, b.lse.0.span_bytes),
        ];
        ensure!(
            spans
                .iter()
                .all(|&(ptr, bytes)| ptr + bytes <= b.scratch.ptr
                    || b.scratch.ptr + b.scratch.bytes <= ptr),
            "ring decode scratch aliases live tensors"
        );
        unsafe {
            stream
                .launch_builder(&self.first)
                .arg(&b.q.ptr)
                .arg(&b.k.ptr)
                .arg(&b.v.ptr)
                .arg(&b.scratch.ptr)
                .arg(&b.position.ptr)
                .arg(&self.heads)
                .arg(&self.kv_heads)
                .arg(&self.ring)
                .arg(&self.chunk)
                .arg(&self.scale)
                .launch(self.first_launch)?;
            stream
                .launch_builder(&self.merge)
                .arg(&b.scratch.ptr)
                .arg(&b.out.0.ptr)
                .arg(&b.lse.0.ptr)
                .arg(&b.position.ptr)
                .arg(&self.heads)
                .arg(&self.ring)
                .arg(&self.chunk)
                .launch(self.merge_launch)?;
        }
        Ok(())
    }
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
        check_decode_attention_shape(c, settings.qpack)?;
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

    pub fn run_with<L: DevicePtr<i32>>(
        &mut self,
        stream: &Arc<CudaStream>,
        input: GqaDecodeInput<'_, L>,
    ) -> Result<()> {
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
        let (len_ptr, _len_guard) = length.device_ptr(stream);
        unsafe {
            if let Some(window) = window {
                stream
                    .launch_builder(&self.window_first)
                    .arg(&q)
                    .arg(k)
                    .arg(v)
                    .arg(&mut self.scratch)
                    .arg(&len_ptr)
                    .arg(&self.q_heads)
                    .arg(&self.kv_heads)
                    .arg(&gqa)
                    .arg(&128i32)
                    .arg(&window.head_stride)
                    .arg(&128i32)
                    .arg(&window.head_stride)
                    .arg(&128i32)
                    .arg(&chunk)
                    .arg(window.position)
                    .arg(&(window.head_stride / 128))
                    .launch(self.first_launch)?;
            } else {
                stream
                    .launch_builder(&self.first)
                    .arg(&q)
                    .arg(k)
                    .arg(v)
                    .arg(&mut self.scratch)
                    .arg(&len_ptr)
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
                    .arg(&len_ptr)
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
                    .arg(&len_ptr)
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
            #[cfg(feature = "cuda")]
            cudarc::driver::sys::cuMemsetD32Async(ptr, bits, 1, stream.cu_stream()).result()?;
            #[cfg(feature = "rocm")]
            stream.memset_words(ptr, bits, 1)?;
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
