use super::{cubin, flash::Shape};
use crate::{
    backend::setup::{AttentionPlan, MultiImpl, MultiPlan, MultiShape},
    config::Config,
};
use anyhow::{Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::bf16;
use std::sync::Arc;

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
/// `MultiImpl`: V1, Tcmqa, TcmqaW.
pub fn phase1_residency(ctx: &Arc<CudaContext>) -> Result<[usize; 3]> {
    let module = cubin::module(ctx, "tcmqa")?;
    let mut residency = [0usize; 3];
    for (implementation, threads) in [(MultiImpl::Tcmqa, 128u32), (MultiImpl::TcmqaW, 256)] {
        let symbol = match implementation {
            MultiImpl::Tcmqa => "tcmqa_phase1",
            MultiImpl::TcmqaW => "tcmqa_phase1_w",
            MultiImpl::V1 => unreachable!(),
        };
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
    pub fn candidate(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        c: &Config,
        capacity: usize,
        rows: usize,
        plan: AttentionPlan,
    ) -> Result<Self> {
        Self::with_plan(
            ctx,
            stream,
            c,
            MultiShape {
                capacity,
                rows,
                causal: true,
            },
            MultiPlan {
                implementation: MultiImpl::V1,
                launch: plan,
            },
        )
    }

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
        let plan = selected.launch;
        let implementation = selected.implementation;
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
            capacity > rows && capacity <= i32::MAX as usize / 128,
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
                && plan.merge_threads.is_multiple_of(128),
            "invalid MQ phase2 block size"
        );
        let chunks = capacity.div_ceil(plan.chunk);
        let first_smem =
            if implementation == MultiImpl::Tcmqa || implementation == MultiImpl::TcmqaW {
                implementation.phase1_smem_bytes()
            } else {
                plan.qpack * (plan.threads as usize / 32) * 130 * 4
            };
        let second_smem = (2 * chunks + plan.merge_threads as usize) * 4;
        // TcmqaW's 52,736 B phase-1 tile needs the per-function opt-in set
        // below; phase 2 has no opt-in anywhere, so it stays under 48 KiB.
        let first_smem_limit = if implementation == MultiImpl::TcmqaW {
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
            MultiImpl::Tcmqa => "tcmqa_phase1".to_owned(),
            MultiImpl::TcmqaW => "tcmqa_phase1_w".to_owned(),
        };
        ensure!(
            implementation != MultiImpl::V1 || causal,
            "v1 MQ only supports causal attention"
        );
        if implementation == MultiImpl::Tcmqa {
            ensure!(
                plan.threads == 128 && plan.chunk.is_multiple_of(64),
                "TCMQA requires threads128 and chunk multiple64"
            );
        }
        if implementation == MultiImpl::TcmqaW {
            ensure!(
                plan.threads == 256 && plan.chunk.is_multiple_of(64),
                "TCMQA-W requires threads256 and chunk multiple64"
            );
        }
        let first = module.load_function(&symbol)?;
        if implementation == MultiImpl::TcmqaW {
            first.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                implementation.phase1_smem_bytes() as i32,
            )?;
        }
        let second = module.load_function(if implementation == MultiImpl::V1 {
            "flash_decode_mq_phase2"
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
        let Query {
            qkv,
            k,
            v,
            out,
            prefix,
            shape,
        } = query;
        ensure!(
            shape.causal == self.shape.causal && shape.rows == self.shape.rows,
            "MQ shape does not match its calibrated M/mask"
        );
        ensure!(
            shape.capacity == self.shape.capacity
                && shape.start + shape.rows <= self.shape.capacity,
            "MQ range exceeds configured KV cache"
        );
        ensure!(
            shape.q_heads == 16 && shape.kv_heads == 2,
            "MQ head shape mismatch"
        );
        ensure!(
            qkv.len() >= shape.rows * 2560 && out.len() >= shape.rows * 2048 && prefix.len() == 1,
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
        unsafe {
            let mut first = self.stream.launch_builder(&self.first);
            first
                .arg(qkv)
                .arg(k)
                .arg(v)
                .arg(&mut self.scratch)
                .arg(prefix)
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
                grid_dim: if self.implementation == MultiImpl::Tcmqa
                    || self.implementation == MultiImpl::TcmqaW
                {
                    (
                        chunks as u32,
                        2,
                        (8 * shape.rows).div_ceil(self.implementation.phase1_tile_rows()) as u32,
                    )
                } else {
                    (chunks as u32, (16 / p.qpack) as u32, rows as u32)
                },
                block_dim: (p.threads, 1, 1),
                shared_mem_bytes: if self.implementation == MultiImpl::Tcmqa
                    || self.implementation == MultiImpl::TcmqaW
                {
                    self.implementation.phase1_smem_bytes() as u32
                } else {
                    (p.qpack * (p.threads as usize / 32) * 130 * 4) as u32
                },
            })?;
            self.stream
                .launch_builder(&self.second)
                .arg(&mut self.scratch)
                .arg(out)
                .arg(prefix)
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
        }
    }
}
