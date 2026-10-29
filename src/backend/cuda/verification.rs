use super::cubin;
use crate::backend::gpu::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use crate::{
    backend::setup::{AttentionPlan, MultiImpl, MultiPlan, MultiShape},
    config::Config,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct Shape {
    pub rows: usize,
    pub start: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub capacity: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub window_left: Option<usize>,
}

pub struct Verification {
    stream: Arc<CudaStream>,
    first: CudaFunction,
    second: CudaFunction,
    scratch: CudaSlice<f32>,
    shape: MultiShape,
    implementation: MultiImpl,
    anc: Option<CudaSlice<u64>>,
    plan: AttentionPlan,
}

#[cfg(feature = "rocm")]
pub(super) fn hip_tcmqa_plan(
    ctx: &Arc<CudaContext>,
    capacity: usize,
    rows: usize,
) -> Result<MultiPlan> {
    ctx.require_wmma()?;
    ensure!(
        (1..=64).contains(&rows) && capacity >= rows,
        "HIP TCMQA rows/capacity outside supported extent"
    );
    let function = cubin::module(ctx, "tcmqa")?.load_function("tcmqa_phase1")?;
    let occupancy = function.occupancy_max_active_blocks_per_multiprocessor(
        128,
        MultiImpl::Tcmqa.phase1_smem_bytes(),
        None,
    )?;
    ensure!(occupancy > 0, "HIP TCMQA phase1 has zero occupancy");
    let groups = 2 * (8 * rows).div_ceil(64);
    let partitions = (ctx.device_info()?.sms * occupancy as usize).div_ceil(groups);
    let chunk = capacity.div_ceil(partitions).next_multiple_of(64).max(64);
    Ok(MultiPlan {
        implementation: MultiImpl::Tcmqa,
        launch: AttentionPlan {
            chunk,
            qpack: 8,
            threads: 128,
            merge_threads: 256,
        },
    })
}

pub struct Query<'a> {
    pub qkv: &'a CudaSlice<bf16>,
    pub k: &'a CudaSlice<bf16>,
    pub v: &'a CudaSlice<bf16>,
    pub out: &'a mut CudaSlice<bf16>,
    pub prefix: &'a CudaSlice<i32>,
    pub shape: Shape,
}

/// Resident phase-1 blocks per SM, queried from the driver for each loaded
/// kernel with its real block size and dynamic shared memory. Indexed like
/// `MultiImpl`: V1, Tcmqa, TcmqaW, TcmqaWs.
#[cfg(feature = "cuda")]
pub fn phase1_residency(ctx: &Arc<CudaContext>) -> Result<[usize; 4]> {
    let module = cubin::module(ctx, "tcmqa")?;
    let mut residency = [0usize; 4];
    for (implementation, threads, symbol) in [
        (MultiImpl::Tcmqa, 128u32, "tcmqa_phase1"),
        (MultiImpl::TcmqaW, 256, "tcmqa_phase1_w"),
        (MultiImpl::TcmqaWs, 256, "tcmqa_phase1_ws"),
    ] {
        let first = module.load_function(symbol)?;
        let smem = implementation.phase1_smem_bytes();
        if smem > 48 * 1024 {
            first.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )?;
        }
        residency[implementation as usize] =
            first.occupancy_max_active_blocks_per_multiprocessor(threads, smem, None)? as usize;
    }
    Ok(residency)
}

impl Verification {
    pub fn with_plan(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        c: &Config,
        shape: MultiShape,
        selected: MultiPlan,
    ) -> Result<Self> {
        let MultiShape {
            capacity,
            rows,
            causal,
        } = shape;
        #[cfg(feature = "rocm")]
        if selected.implementation != MultiImpl::V1 {
            ctx.require_wmma()?;
            ensure!(
                selected.implementation == MultiImpl::Tcmqa,
                "HIP TCMQA requires the native 64-pair WMMA plan"
            );
        }
        let plan = selected.launch;
        let implementation = selected.implementation;
        let phase1 = implementation.phase1();
        ensure!(
            c.num_attention_heads == 16 && c.num_key_value_heads == 2 && c.head_dim == 128,
            "MQ attention requires 16 Q heads, 2 KV heads and head_dim 128"
        );
        ensure!(
            rows > 0
                && rows
                    <= if implementation == MultiImpl::V1 {
                        16
                    } else {
                        64
                    },
            "MQ rows exceed implementation support"
        );
        ensure!(
            capacity >= rows && capacity <= i32::MAX as usize / 128,
            "MQ capacity exceeds kernel stride range"
        );
        ensure!(
            [1, 2, 4, 8].contains(&plan.qpack) && plan.chunk > 0,
            "invalid MQ chunk/qpack"
        );
        ensure!(
            plan.threads >= 32 && plan.threads <= 1024 && plan.threads.is_multiple_of(32),
            "invalid MQ phase1 block size"
        );
        ensure!(
            plan.merge_threads >= 128
                && plan.merge_threads <= 1024
                && plan.merge_threads.is_power_of_two(),
            "invalid MQ phase2 block size"
        );
        let chunks = capacity.div_ceil(plan.chunk);
        if implementation.short_merge() {
            #[cfg(feature = "cuda")]
            let warp = ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?
                as usize;
            #[cfg(feature = "rocm")]
            let warp = ctx.properties.warp as usize;
            ensure!(
                chunks <= 2 * warp && matches!(plan.merge_threads, 256 | 512),
                "TCMQA short merge requires at most two chunks per warp lane over the full capacity and 256 or 512 threads"
            );
        }
        let first_smem = if phase1 != MultiImpl::V1 {
            implementation.phase1_smem_bytes()
        } else {
            plan.qpack * (plan.threads as usize / 32) * 130 * 4
        };
        let second_smem = (2 * chunks + plan.merge_threads as usize) * 4;
        // TcmqaW/TcmqaWs phase 1 uses the per-function smem opt-in; phase 2 stays under 48 KiB.
        let first_smem_limit = if matches!(phase1, MultiImpl::TcmqaW | MultiImpl::TcmqaWs) {
            101 * 1024
        } else {
            48 * 1024
        };
        ensure!(
            first_smem <= first_smem_limit,
            "MQ phase-1 shared memory exceeds its limit"
        );
        ensure!(
            second_smem <= 48 * 1024,
            "MQ phase-2 shared memory exceeds 48 KiB"
        );
        let module = cubin::module(
            ctx,
            if implementation == MultiImpl::V1 {
                "attention"
            } else {
                "tcmqa"
            },
        )?;
        let symbol = match implementation {
            MultiImpl::V1 => format!("flash_decode_mq_phase1_q{}", plan.qpack),
            MultiImpl::Tcmqa | MultiImpl::TcmqaShort => "tcmqa_phase1".to_owned(),
            MultiImpl::TcmqaW | MultiImpl::TcmqaWShort => "tcmqa_phase1_w".to_owned(),
            MultiImpl::TcmqaWs => "tcmqa_phase1_ws".to_owned(),
        };
        ensure!(
            implementation != MultiImpl::V1 || causal,
            "v1 MQ only supports causal attention"
        );
        if phase1 == MultiImpl::Tcmqa {
            ensure!(
                plan.threads == 128 && plan.chunk.is_multiple_of(64),
                "TCMQA requires threads128 and chunk multiple64"
            );
        }
        if phase1 == MultiImpl::TcmqaW {
            ensure!(
                plan.threads == 256 && plan.chunk.is_multiple_of(64),
                "TCMQA-W requires threads256 and chunk multiple64"
            );
        }
        if implementation == MultiImpl::TcmqaWs {
            ensure!(
                plan.threads == 256 && plan.chunk.is_multiple_of(64),
                "TCMQA-WS requires threads256 and chunk multiple64"
            );
        }
        let first = module.load_function(&symbol)?;
        #[cfg(feature = "cuda")]
        if matches!(phase1, MultiImpl::TcmqaW | MultiImpl::TcmqaWs) {
            first.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                implementation.phase1_smem_bytes() as i32,
            )?;
        }
        #[cfg(feature = "rocm")]
        ensure!(
            first_smem <= ctx.properties.shared_bytes as usize
                && first.occupancy_max_active_blocks_per_multiprocessor(
                    plan.threads,
                    first_smem,
                    None
                )? > 0,
            "HIP MQ phase1 has zero occupancy or exceeds LDS"
        );
        let second = module.load_function(if implementation == MultiImpl::V1 {
            "flash_decode_mq_phase2"
        } else if implementation.short_merge() {
            "tcmqa_phase2_short"
        } else {
            "tcmqa_phase2"
        })?;
        ensure!(
            plan.threads as i32 <= first.max_threads_per_block()?
                && plan.merge_threads as i32 <= second.max_threads_per_block()?,
            "MQ block size exceeds kernel limit"
        );
        let masks: Vec<u64> = (0..rows)
            .map(|i| {
                let bits = if causal { i + 1 } else { rows };
                if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                }
            })
            .collect();
        Ok(Self {
            stream: stream.clone(),
            first,
            second,
            scratch: stream.alloc_zeros(rows * 16 * chunks * 130)?,
            shape,
            implementation,
            anc: if implementation != MultiImpl::V1 {
                Some(stream.clone_htod(&masks)?)
            } else {
                None
            },
            plan,
        })
    }

    pub fn run(
        &mut self,
        qkv: &CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        prefix: &CudaSlice<i32>,
        shape: Shape,
    ) -> Result<()> {
        self.run_mask(
            Query {
                qkv,
                k,
                v,
                out,
                prefix,
                shape,
            },
            None,
        )
    }

    pub fn run_tree(&mut self, query: Query<'_>, anc: &CudaSlice<u64>) -> Result<()> {
        ensure!(
            self.implementation != MultiImpl::V1 && !query.shape.causal,
            "tree attention requires a TCMQA tree plan"
        );
        ensure!(
            anc.len() >= query.shape.rows,
            "tree ancestor masks do not cover budget rows"
        );
        self.run_mask(query, Some(anc))
    }

    fn run_mask(&mut self, query: Query<'_>, external: Option<&CudaSlice<u64>>) -> Result<()> {
        ensure!(
            query.shape.rows == self.shape.rows && query.prefix.len() == 1,
            "MQ shape does not match its calibrated M/prefix"
        );
        let Query {
            qkv,
            k,
            v,
            out,
            prefix,
            shape,
        } = query;
        ensure!(
            shape.causal == self.shape.causal && shape.rows > 0 && shape.rows <= self.shape.rows,
            "MQ shape does not match its calibrated M/mask"
        );
        ensure!(
            shape.capacity == self.shape.capacity
                && shape
                    .start
                    .checked_add(shape.rows)
                    .is_some_and(|end| end <= self.shape.capacity),
            "MQ range exceeds configured KV cache"
        );
        ensure!(
            shape.q_heads == 16
                && shape.kv_heads == 2
                && shape.head_dim == 128
                && shape.window_left.is_none(),
            "MQ head shape mismatch"
        );
        ensure!(
            shape.rows <= qkv.len() / 2560 && shape.rows <= out.len() / 2048 && prefix.len() == 1,
            "MQ query/output/prefix buffer shape mismatch"
        );
        ensure!(
            k.len() >= self.shape.capacity * 256 && v.len() >= self.shape.capacity * 256,
            "MQ KV buffer shape mismatch"
        );
        let p = self.plan;
        let chunks = self.shape.capacity.div_ceil(p.chunk);
        let rows = shape.rows as i32;
        let stride = (self.shape.capacity * 128) as i32;
        let chunk = p.chunk as i32;
        let qkv = qkv.slice(..shape.rows * 2560);
        let mut out = out.slice_mut(..shape.rows * 2048);
        let prefix = prefix.slice(..1);
        unsafe {
            let mut first = self.stream.launch_builder(&self.first);
            first
                .arg(&qkv)
                .arg(k)
                .arg(v)
                .arg(&mut self.scratch)
                .arg(&prefix)
                .arg(&rows)
                .arg(&16i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&128i32)
                .arg(&2560i32)
                .arg(&128i32)
                .arg(&stride)
                .arg(&128i32)
                .arg(&stride)
                .arg(&128i32)
                .arg(&chunk);
            if let Some(anc) = external.or(self.anc.as_ref()) {
                first.arg(anc);
            }
            first.launch(LaunchConfig {
                grid_dim: if self.implementation != MultiImpl::V1 {
                    (
                        chunks as u32,
                        2,
                        (8 * shape.rows).div_ceil(self.implementation.phase1_tile_rows()) as u32,
                    )
                } else {
                    (chunks as u32, (16 / p.qpack) as u32, rows as u32)
                },
                block_dim: (p.threads, 1, 1),
                shared_mem_bytes: if self.implementation != MultiImpl::V1 {
                    self.implementation.phase1_smem_bytes() as u32
                } else {
                    (p.qpack * (p.threads as usize / 32) * 130 * 4) as u32
                },
            })?;
            self.stream
                .launch_builder(&self.second)
                .arg(&mut self.scratch)
                .arg(&mut out)
                .arg(&prefix)
                .arg(&rows)
                .arg(&16i32)
                .arg(&2i32)
                .arg(&8i32)
                .arg(&128i32)
                .arg(&2048i32)
                .arg(&128i32)
                .arg(&chunk)
                .launch(LaunchConfig {
                    grid_dim: (16 * rows as u32, 1, 1),
                    block_dim: (p.merge_threads, 1, 1),
                    shared_mem_bytes: ((2 * chunks + p.merge_threads as usize) * 4) as u32,
                })?;
        }
        Ok(())
    }

    pub fn implementation(&self) -> &'static str {
        match self.implementation {
            MultiImpl::V1 => "v1 flash_decode_mq",
            MultiImpl::Tcmqa => "TCMQA",
            MultiImpl::TcmqaW => "TCMQA wide (128-row tile)",
            MultiImpl::TcmqaWs => "TCMQA wide, reduced smem (128-row tile)",
            MultiImpl::TcmqaShort => "TCMQA short merge",
            MultiImpl::TcmqaWShort => "TCMQA wide (128-row tile), short merge",
        }
    }
}

#[cfg(feature = "rocm")]
pub(super) fn hip_plan(
    ctx: &Arc<CudaContext>,
    capacity: usize,
    rows: usize,
    multi: bool,
    q_heads: usize,
) -> Result<AttentionPlan> {
    ctx.require_simt("dense attention setup")?;
    ensure!(
        capacity > 0 && rows > 0 && rows <= 16,
        "HIP attention setup extent is invalid"
    );
    let p = &ctx.properties;
    let module = cubin::module(ctx, "attention")?;
    let prefix = if !multi {
        "flash_decode"
    } else {
        "flash_decode_mq"
    };
    let qpack = 8;
    ensure!(
        q_heads > 0 && q_heads.is_multiple_of(qpack),
        "HIP attention query heads must be divisible by qpack"
    );
    let first = module.load_function(&format!("{prefix}_phase1_q{qpack}"))?;
    let second = module.load_function(&format!("{prefix}_phase2"))?;
    let threads = 128u32
        .min(p.max_threads as u32)
        .min(first.max_threads_per_block()? as u32)
        / 32
        * 32;
    let shared = qpack * (threads as usize / 32) * 130 * 4;
    ensure!(
        threads >= 32 && shared <= p.shared_bytes as usize,
        "HIP attention phase1 exceeds detected LDS/thread limits"
    );
    let resident =
        first.occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)? as usize;
    ensure!(
        resident > 0 && p.compute_units > 0,
        "HIP attention reports zero resident blocks/compute units"
    );
    let target = (p.compute_units as usize * resident).div_ceil(rows * q_heads / qpack);
    let chunk = capacity.div_ceil(target).max(128).div_ceil(32) * 32;
    let merge_threads = 128;
    let merge_shared = (2 * capacity.div_ceil(chunk) + merge_threads as usize) * 4;
    ensure!(
        second.max_threads_per_block()? >= merge_threads as i32
            && merge_shared <= (p.shared_bytes as usize).min(48 * 1024),
        "HIP attention merge exceeds detected LDS/thread limits"
    );
    ensure!(
        second.occupancy_max_active_blocks_per_multiprocessor(merge_threads, merge_shared, None)?
            > 0,
        "HIP attention merge has zero occupancy"
    );
    Ok(AttentionPlan {
        chunk,
        threads,
        qpack,
        merge_threads,
    })
}
