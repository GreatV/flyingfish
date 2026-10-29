//! Indexed-pool selection and sparse latent attention with bounded per-query top-k state.
use super::{
    contracts::{Tensor, TensorMut, no_overlap},
    cubin,
};
use crate::backend::gpu::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use crate::{
    backend::setup::DeviceInfo,
    op_contract::{DsaKvForm, DsaMeta, DsaSpec},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

#[cfg(all(test, feature = "cuda"))]
#[path = "dsa_tests.rs"]
mod tests;

struct Kernels {
    topk512: CudaFunction,
    topk2048: CudaFunction,
    append: CudaFunction,
    pool: CudaFunction,
    index: CudaFunction,
    expand: CudaFunction,
    attention: CudaFunction,
    checkpoint: CudaFunction,
}

#[derive(Clone, Copy, Debug)]
pub struct DsaStateView {
    latent: u64,
    packed: u64,
    pool_keys: u64,
    token_ids: u64,
    pool_ids: u64,
    meta: u64,
}

pub struct DsaState {
    pub latent: CudaSlice<bf16>,
    pub packed: CudaSlice<bf16>,
    pub pool_keys: CudaSlice<bf16>,
    pub token_ids: CudaSlice<i32>,
    pub pool_ids: CudaSlice<i32>,
    pub meta: CudaSlice<i32>,
    spec: DsaSpec,
    capacity: u32,
}
impl DsaState {
    pub fn new(spec: DsaSpec, capacity: u32, stream: &Arc<CudaStream>) -> Result<Self> {
        spec.state_bytes(capacity)?;
        let pools = spec.pool_capacity(capacity) as usize;
        let cap = capacity as usize;
        Ok(Self {
            latent: stream.alloc_zeros(cap * spec.latent_dim as usize)?,
            packed: stream.alloc_zeros(cap * spec.packed_width() as usize)?,
            pool_keys: stream.alloc_zeros(pools * spec.index_dim as usize)?,
            token_ids: stream.alloc_zeros(cap)?,
            pool_ids: stream.alloc_zeros(pools)?,
            meta: stream.alloc_zeros(8)?,
            spec,
            capacity,
        })
    }
    pub fn bytes(&self) -> usize {
        (self.latent.len() + self.packed.len() + self.pool_keys.len()) * 2
            + (self.token_ids.len() + self.pool_ids.len() + self.meta.len()) * 4
    }
    pub fn view(&self, stream: &Arc<CudaStream>) -> DsaStateView {
        DsaStateView {
            latent: self.latent.device_ptr(stream).0,
            packed: self.packed.device_ptr(stream).0,
            pool_keys: self.pool_keys.device_ptr(stream).0,
            token_ids: self.token_ids.device_ptr(stream).0,
            pool_ids: self.pool_ids.device_ptr(stream).0,
            meta: self.meta.device_ptr(stream).0,
        }
    }
    pub fn set_meta(&mut self, stream: &Arc<CudaStream>, meta: DsaMeta) -> Result<()> {
        meta.validate(self.capacity)?;
        stream.memcpy_htod(
            &[
                meta.prefix,
                meta.valid_rows,
                meta.key_count,
                meta.capacity,
                meta.first_key,
                0,
                0,
                0,
            ],
            &mut self.meta,
        )?;
        Ok(())
    }
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        stream.memcpy_htod(&[0, 0, 0, self.capacity as i32, 0, 0, 0, 0], &mut self.meta)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DsaAppendInput {
    pub(crate) latent: Tensor<bf16, 2>,
    pub(crate) packed: Tensor<bf16, 2>,
}
#[derive(Clone, Copy, Debug)]
pub struct DsaIndexInput {
    pub(crate) q: Tensor<bf16, 3>,
    pub(crate) weights: Tensor<bf16, 2>,
    pub(crate) ape: Tensor<bf16, 2>,
}
#[derive(Debug)]
pub struct DsaIndexOutput {
    pub(crate) selected: TensorMut<i32, 2>,
    pub(crate) scores: TensorMut<f32, 2>,
    pub(crate) margins: TensorMut<f32, 1>,
    pub(crate) indices: TensorMut<i32, 2>,
}
#[derive(Clone, Copy, Debug)]
pub struct DsaTopKInput {
    pub(crate) scores: Tensor<f32, 2>,
    pub(crate) ids: Tensor<i32, 2>,
}
#[derive(Debug)]
pub struct DsaTopKOutput {
    pub(crate) scores: TensorMut<f32, 2>,
    pub(crate) ids: TensorMut<i32, 2>,
    pub(crate) margins: TensorMut<f32, 1>,
}
#[derive(Clone, Copy, Debug)]
pub struct DsaAttentionInput<'a> {
    pub(crate) q: Tensor<bf16, 3>,
    pub(crate) kv_b: Option<Tensor<bf16, 2>>,
    pub(crate) indices: Tensor<i32, 2>,
    pub(crate) sinks: Option<Tensor<f32, 1>>,
    pub(crate) out: &'a TensorMut<bf16, 3>,
}

pub struct DsaPlan {
    spec: DsaSpec,
    capacity: u32,
    rows: u32,
    kernels: Arc<Kernels>,
    append_cfg: LaunchConfig,
    pool_cfg: LaunchConfig,
    index_cfg: LaunchConfig,
    expand_cfg: LaunchConfig,
    attention_cfg: LaunchConfig,
}
impl DsaPlan {
    pub fn new(
        ctx: &Arc<CudaContext>,
        info: &DeviceInfo,
        spec: DsaSpec,
        capacity: u32,
        rows: u32,
    ) -> Result<Self> {
        spec.validate()?;
        ensure!(
            capacity > 0 && capacity <= i32::MAX as u32 && rows > 0 && rows <= capacity,
            "invalid DSA plan extents"
        );
        ensure!(
            info.max_threads >= 1024 && info.shared_bytes >= 24 * 1024,
            "DSA indexer requires 1024 threads and 24 KiB shared memory"
        );
        let module = cubin::module(ctx, "dsa")?;
        Ok(Self {
            spec,
            capacity,
            rows,
            kernels: Arc::new(Kernels {
                topk512: module.load_function("dsa_bounded_topk512")?,
                topk2048: module.load_function("dsa_bounded_topk2048")?,
                append: module.load_function("dsa_append")?,
                pool: module.load_function("dsa_pool")?,
                index: module.load_function("dsa_index_score_topk")?,
                expand: module.load_function("dsa_expand_indices")?,
                attention: module.load_function(match spec.kv_form {
                    DsaKvForm::Projected => "dsa_sparse_attention",
                    DsaKvForm::Direct => "dsa_sparse_attention_direct",
                })?,
                checkpoint: module.load_function("dsa_checkpoint")?,
            }),
            append_cfg: LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            pool_cfg: LaunchConfig {
                grid_dim: (rows.div_ceil(4) + 1, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            index_cfg: LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            },
            expand_cfg: LaunchConfig {
                grid_dim: (rows, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            attention_cfg: LaunchConfig {
                grid_dim: (rows, spec.q_heads, 1),
                block_dim: (spec.qk_dim, 1, 1),
                shared_mem_bytes: 0,
            },
        })
    }
    pub fn select_scores(
        &self,
        input: &DsaTopKInput,
        output: &DsaTopKOutput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        input.scores.validate()?;
        input.ids.validate()?;
        output.scores.read().validate()?;
        output.ids.read().validate()?;
        output.margins.read().validate()?;
        let candidates = input.scores.shape[1];
        let keep = output.scores.read().shape[1];
        ensure!(
            candidates > 0 && candidates <= i32::MAX as u32 && keep > 0 && keep <= 2048,
            "DSA bounded selection extents"
        );
        ensure!(
            input.scores.shape == [self.rows, candidates]
                && input.scores.stride == [u64::from(candidates), 1]
                && input.ids.shape == input.scores.shape
                && input.ids.stride == input.scores.stride,
            "DSA candidates must be contiguous [rows,candidates]"
        );
        ensure!(
            output.scores.read().shape == [self.rows, keep]
                && output.scores.read().stride == [u64::from(keep), 1]
                && output.ids.read().shape == output.scores.read().shape
                && output.ids.read().stride == output.scores.read().stride
                && output.margins.read().shape == [self.rows]
                && output.margins.read().stride == [1],
            "DSA selection output geometry"
        );
        no_overlap(
            &input.scores,
            &output.scores.read(),
            "DSA scores",
            "DSA selected scores",
        )?;
        no_overlap(
            &input.ids,
            &output.ids.read(),
            "DSA IDs",
            "DSA selected IDs",
        )?;
        let candidates = candidates as i32;
        let keep = keep as i32;
        let kernel = if keep <= 512 {
            &self.kernels.topk512
        } else {
            &self.kernels.topk2048
        };
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&input.scores.ptr)
                .arg(&input.ids.ptr)
                .arg(&candidates)
                .arg(&keep)
                .arg(&output.ids.0.ptr)
                .arg(&output.scores.0.ptr)
                .arg(&output.margins.0.ptr)
                .launch(LaunchConfig {
                    grid_dim: (self.rows, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    pub fn for_rows(&self, rows: u32) -> Result<Self> {
        ensure!(rows > 0 && rows <= self.rows, "DSA rows outside binding");
        let mut bound = Self {
            spec: self.spec,
            capacity: self.capacity,
            rows,
            kernels: self.kernels.clone(),
            append_cfg: self.append_cfg,
            pool_cfg: self.pool_cfg,
            index_cfg: self.index_cfg,
            expand_cfg: self.expand_cfg,
            attention_cfg: self.attention_cfg,
        };
        bound.append_cfg.grid_dim.0 = rows;
        bound.pool_cfg.grid_dim.0 = rows.div_ceil(4) + 1;
        bound.index_cfg.grid_dim.0 = rows;
        bound.expand_cfg.grid_dim.0 = rows;
        bound.attention_cfg.grid_dim.0 = rows;
        Ok(bound)
    }
    fn state(&self, state: &DsaState, stream: &Arc<CudaStream>) -> Result<DsaStateView> {
        ensure!(
            state.spec == self.spec && state.capacity == self.capacity,
            "DSA state/plan geometry mismatch"
        );
        let view = state.view(stream);
        for ptr in [
            view.latent,
            view.packed,
            view.pool_keys,
            view.token_ids,
            view.pool_ids,
            view.meta,
        ] {
            ensure!(ptr != 0 && ptr.is_multiple_of(16), "DSA state alignment");
        }
        Ok(view)
    }
    fn append_input(&self, input: &DsaAppendInput) -> Result<()> {
        for (view, width, name) in [
            (&input.latent, self.spec.latent_dim, "latent"),
            (&input.packed, self.spec.packed_width(), "packed"),
        ] {
            view.validate()?;
            ensure!(
                view.shape == [self.rows, width] && view.stride == [u64::from(width), 1],
                "DSA {name} input must be contiguous [rows,{width}]"
            );
        }
        Ok(())
    }
    pub fn append(
        &self,
        state: &DsaState,
        input: &DsaAppendInput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        self.append_input(input)?;
        let view = self.state(state, stream)?;
        let latent_width = self.spec.latent_dim as i32;
        unsafe {
            stream
                .launch_builder(&self.kernels.append)
                .arg(&input.latent.ptr)
                .arg(&input.packed.ptr)
                .arg(&view.latent)
                .arg(&view.packed)
                .arg(&view.token_ids)
                .arg(&view.meta)
                .arg(&latent_width)
                .arg(&self.capacity)
                .arg(&self.rows)
                .launch(self.append_cfg)?;
        }
        Ok(())
    }
    pub fn pool(
        &self,
        state: &DsaState,
        input: &DsaIndexInput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        ensure!(
            self.spec.kv_form == DsaKvForm::Projected,
            "DSA pool requires projected pool4 specification"
        );
        let view = self.state(state, stream)?;
        let ape = &input.ape;
        ape.validate()?;
        ensure!(
            ape.shape == [4, 128] && ape.stride == [128, 1],
            "DSA APE must be [4,128] contiguous"
        );
        let pool_capacity = self.spec.pool_capacity(self.capacity) as i32;
        unsafe {
            stream
                .launch_builder(&self.kernels.pool)
                .arg(&view.packed)
                .arg(&view.pool_keys)
                .arg(&view.token_ids)
                .arg(&view.pool_ids)
                .arg(&ape.ptr)
                .arg(&view.meta)
                .arg(&pool_capacity)
                .arg(&self.capacity)
                .arg(&self.rows)
                .launch(self.pool_cfg)?;
        }
        Ok(())
    }
    pub fn index(
        &self,
        state: &DsaState,
        ape: &DsaIndexInput,
        output: &DsaIndexOutput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        ensure!(
            self.spec.kv_form == DsaKvForm::Projected,
            "DSA pooled index requires projected pool4 specification"
        );
        let view = self.state(state, stream)?;
        let rows = self.rows;
        ape.q.validate()?;
        ensure!(
            ape.q.shape == [rows, 32, 128] && ape.q.stride == [4096, 128, 1],
            "DSA index Q must be [rows,32,128] contiguous"
        );
        ape.weights.validate()?;
        ensure!(
            ape.weights.shape == [rows, 32] && ape.weights.stride == [32, 1],
            "DSA index weights must be [rows,32] contiguous"
        );
        ape.ape.validate()?;
        ensure!(
            ape.ape.shape == [4, 128] && ape.ape.stride == [128, 1],
            "DSA APE must be [4,128] contiguous"
        );
        for (view, width, name) in [
            (output.selected.read(), 512, "selected"),
            (output.indices.read(), 2051, "indices"),
        ] {
            view.validate()?;
            ensure!(
                view.shape == [rows, width] && view.stride == [u64::from(width), 1],
                "DSA {name} output must be [rows,{width}] contiguous"
            );
        }
        output.scores.read().validate()?;
        ensure!(
            output.scores.read().shape == [rows, 512] && output.scores.read().stride == [512, 1],
            "DSA scores must be [rows,512] contiguous"
        );
        output.margins.read().validate()?;
        ensure!(
            output.margins.read().shape == [rows] && output.margins.read().stride == [1],
            "DSA margins must be [rows] contiguous"
        );
        let pool_capacity = self.spec.pool_capacity(self.capacity) as i32;
        unsafe {
            stream
                .launch_builder(&self.kernels.index)
                .arg(&ape.q.ptr)
                .arg(&ape.weights.ptr)
                .arg(&view.packed)
                .arg(&view.pool_keys)
                .arg(&view.token_ids)
                .arg(&view.pool_ids)
                .arg(&view.meta)
                .arg(&pool_capacity)
                .arg(&output.selected.0.ptr)
                .arg(&output.scores.0.ptr)
                .arg(&output.margins.0.ptr)
                .arg(&self.capacity)
                .arg(&self.rows)
                .launch(self.index_cfg)?;
            stream
                .launch_builder(&self.kernels.expand)
                .arg(&output.selected.0.ptr)
                .arg(&view.packed)
                .arg(&view.token_ids)
                .arg(&view.pool_ids)
                .arg(&pool_capacity)
                .arg(&view.meta)
                .arg(&output.indices.0.ptr)
                .arg(&self.capacity)
                .arg(&self.rows)
                .launch(self.expand_cfg)?;
        }
        Ok(())
    }
    pub fn attention(
        &self,
        state: &DsaState,
        input: &DsaAttentionInput,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        let view = self.state(state, stream)?;
        let rows = self.rows;
        let dim = self.spec.qk_dim;
        let width = input.indices.shape[1];
        input.q.validate()?;
        ensure!(
            input.q.shape == [rows, 64, dim]
                && input.q.stride == [u64::from(64 * dim), u64::from(dim), 1],
            "DSA attention Q must be [rows,64,dim] contiguous"
        );
        input.out.read().validate()?;
        ensure!(
            input.out.read().shape == [rows, 64, dim]
                && input.out.read().stride == [u64::from(64 * dim), u64::from(dim), 1],
            "DSA attention output must be [rows,64,dim] contiguous"
        );
        no_overlap(&input.q, &input.out.read(), "DSA Q", "DSA output")?;
        input.indices.validate()?;
        ensure!(
            width > 0
                && width <= i32::MAX as u32
                && input.indices.shape == [rows, width]
                && input.indices.stride == [u64::from(width), 1]
                && (self.spec.kv_form == DsaKvForm::Direct || width == 2051),
            "DSA attention indices must be contiguous and unique per row"
        );
        let kv_b = match self.spec.kv_form {
            DsaKvForm::Projected => {
                let weights = input.kv_b.context("projected DSA kv_b is missing")?;
                weights.validate()?;
                ensure!(
                    weights.shape == [32768, 512] && weights.stride == [512, 1],
                    "DSA kv_b must be [32768,512] contiguous"
                );
                weights.ptr
            }
            DsaKvForm::Direct => {
                ensure!(input.kv_b.is_none(), "direct DSA does not accept kv_b");
                0
            }
        };
        let sinks = if let Some(values) = input.sinks {
            values.validate()?;
            ensure!(
                values.shape == [64] && values.stride == [1],
                "DSA sinks must be [64] contiguous"
            );
            values.ptr
        } else {
            0
        };
        let width = width as i32;
        unsafe {
            if self.spec.kv_form == DsaKvForm::Direct {
                stream
                    .launch_builder(&self.kernels.attention)
                    .arg(&input.q.ptr)
                    .arg(&view.latent)
                    .arg(&input.indices.ptr)
                    .arg(&width)
                    .arg(&view.token_ids)
                    .arg(&sinks)
                    .arg(&view.meta)
                    .arg(&input.out.0.ptr)
                    .arg(&self.capacity)
                    .arg(&self.rows)
                    .launch(self.attention_cfg)?;
                return Ok(());
            }
            stream
                .launch_builder(&self.kernels.attention)
                .arg(&input.q.ptr)
                .arg(&view.latent)
                .arg(&kv_b)
                .arg(&input.indices.ptr)
                .arg(&view.token_ids)
                .arg(&sinks)
                .arg(&view.meta)
                .arg(&input.out.0.ptr)
                .arg(&self.capacity)
                .arg(&self.rows)
                .launch(self.attention_cfg)?;
        }
        Ok(())
    }
}

pub struct DsaCheckpoint {
    latent: CudaSlice<bf16>,
    packed: CudaSlice<bf16>,
    pool_keys: CudaSlice<bf16>,
    token_ids: CudaSlice<i32>,
    pool_ids: CudaSlice<i32>,
    meta: CudaSlice<i32>,
    spec: DsaSpec,
    capacity: u32,
    rows: u32,
}
impl DsaCheckpoint {
    pub fn new(spec: DsaSpec, capacity: u32, rows: u32, stream: &Arc<CudaStream>) -> Result<Self> {
        spec.validate()?;
        ensure!(
            rows > 0 && rows <= capacity && capacity > 0 && capacity <= i32::MAX as u32,
            "invalid DSA checkpoint extents"
        );
        let pools = (rows.div_ceil(spec.pool) + 1) as usize;
        Ok(Self {
            latent: stream.alloc_zeros(rows as usize * spec.latent_dim as usize)?,
            packed: stream.alloc_zeros(rows as usize * spec.packed_width() as usize)?,
            pool_keys: stream.alloc_zeros(pools * spec.index_dim as usize)?,
            token_ids: stream.alloc_zeros(rows as usize)?,
            pool_ids: stream.alloc_zeros(pools)?,
            meta: stream.alloc_zeros(8)?,
            spec,
            capacity,
            rows,
        })
    }
    fn launch(
        &mut self,
        plan: &DsaPlan,
        state: &DsaState,
        stream: &Arc<CudaStream>,
        mode: i32,
    ) -> Result<()> {
        ensure!(
            self.spec == plan.spec && self.capacity == plan.capacity && self.rows <= plan.rows,
            "DSA checkpoint/plan geometry mismatch"
        );
        let view = plan.state(state, stream)?;
        let saved_latent = self.latent.device_ptr(stream).0;
        let saved_packed = self.packed.device_ptr(stream).0;
        let saved_pool_keys = self.pool_keys.device_ptr(stream).0;
        let saved_token_ids = self.token_ids.device_ptr(stream).0;
        let saved_pool_ids = self.pool_ids.device_ptr(stream).0;
        let saved_meta = self.meta.device_ptr(stream).0;
        let rows = self.rows as i32;
        let pool_capacity = self.spec.pool_capacity(self.capacity) as i32;
        let latent_width = self.spec.latent_dim as i32;
        let pool_size = self.spec.pool as i32;
        unsafe {
            stream
                .launch_builder(&plan.kernels.checkpoint)
                .arg(&view.latent)
                .arg(&view.packed)
                .arg(&view.pool_keys)
                .arg(&view.token_ids)
                .arg(&view.pool_ids)
                .arg(&view.meta)
                .arg(&saved_latent)
                .arg(&saved_packed)
                .arg(&saved_pool_keys)
                .arg(&saved_token_ids)
                .arg(&saved_pool_ids)
                .arg(&saved_meta)
                .arg(&rows)
                .arg(&pool_capacity)
                .arg(&latent_width)
                .arg(&mode)
                .arg(&self.capacity)
                .arg(&pool_size)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    pub fn save(
        &mut self,
        plan: &DsaPlan,
        state: &DsaState,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        self.launch(plan, state, stream, 0)
    }
    pub fn restore(
        &mut self,
        plan: &DsaPlan,
        state: &DsaState,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        self.launch(plan, state, stream, 1)
    }
}
