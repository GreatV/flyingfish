use super::Device;
use crate::backend::gpu::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::model_spec::BlockGeneration;
use anyhow::{Result, ensure};
use half::bf16;
use std::sync::Arc;

#[cfg(all(test, feature = "cuda"))]
#[path = "sample_tests.rs"]
pub(crate) mod tests;

pub struct BlockSampler {
    stream: Arc<CudaStream>,
    policy: BlockGeneration,
    init: CudaFunction,
    bias: CudaFunction,
    argmax: CudaFunction,
    softmax: CudaFunction,
    fill: CudaFunction,
    commit: CudaFunction,
    parameters: CudaSlice<f32>,
    stops: CudaSlice<u32>,
    pub counts: CudaSlice<u32>,
    pub block: CudaSlice<i32>,
    pub pending: CudaSlice<u8>,
    pub state: CudaSlice<u32>,
    pub picks: CudaSlice<i32>,
    pub probabilities: CudaSlice<f32>,
    probabilities_bf16: CudaSlice<bf16>,
}

impl BlockSampler {
    pub fn work_bytes(rows: u32, vocab: u32, stops: usize) -> Result<u64> {
        ensure!(
            (1..=32).contains(&rows) && vocab > 0 && stops > 0,
            "block sampler workspace geometry mismatch"
        );
        Ok(u64::from(rows) * u64::from(vocab) * 6
            + u64::from(vocab) * 4
            + u64::from(rows) * 9
            + 56
            + u64::try_from(stops)? * 4)
    }

    pub fn prepare(device: &Device, policy: BlockGeneration) -> Result<Self> {
        policy.validate()?;
        let module = super::cubin::module(&device.ctx, "sample")?;
        let s = &device.stream;
        let rows = policy.rows as usize;
        let vocab = policy.vocab as usize;
        let mut sampler = Self {
            stream: s.clone(),
            init: module.load_function("masked_block_init")?,
            bias: module.load_function("token_history_bias")?,
            argmax: module.load_function("argmax_rows_f32")?,
            softmax: module.load_function("softmax_rows_dual")?,
            fill: module.load_function("confidence_threshold_fill")?,
            commit: module.load_function("response_count_commit")?,
            parameters: s.clone_htod(&[
                policy.presence,
                policy.frequency,
                policy.repetition as f32,
                policy.repetition.recip() as f32,
                policy.length_bias,
                policy.length_start as f32,
                (policy.length_window as f32).recip(),
                policy.close_bias,
                policy.close_start as f32,
                (policy.close_window as f32).recip(),
            ])?,
            stops: s.clone_htod(&policy.eos)?,
            counts: s.alloc_zeros(vocab)?,
            block: s.alloc_zeros(rows)?,
            pending: s.alloc_zeros(rows)?,
            state: s.alloc_zeros(4)?,
            picks: s.alloc_zeros(rows)?,
            probabilities: s.alloc_zeros(rows * vocab)?,
            probabilities_bf16: s.alloc_zeros(rows * vocab)?,
            policy,
        };
        sampler.begin_block()?;
        Ok(sampler)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.stream.memset_zeros(&mut self.counts)?;
        self.stream.memset_zeros(&mut self.state)?;
        self.begin_block()
    }

    pub fn begin_block(&mut self) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.init)
                .arg(&mut self.block)
                .arg(&mut self.pending)
                .arg(&mut self.state)
                .arg(&(self.policy.rows as i32))
                .arg(&(self.policy.mask as i32))
                .arg(&self.policy.capacity)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn apply_history(&self, logits: &mut CudaSlice<f32>) -> Result<()> {
        ensure!(
            logits.len() == self.policy.rows as usize * self.policy.vocab as usize,
            "history logits extent mismatch"
        );
        unsafe {
            self.stream
                .launch_builder(&self.bias)
                .arg(logits)
                .arg(&self.counts)
                .arg(&self.block)
                .arg(&self.state)
                .arg(&self.stops)
                .arg(&self.parameters)
                .arg(&(self.policy.rows as i32))
                .arg(&(self.policy.vocab as i32))
                .arg(&(self.policy.mask as i32))
                .arg(&(self.policy.pad as i32))
                .arg(&(self.policy.close_token as i32))
                .arg(&(self.policy.eos.len() as i32))
                .launch(super::ops::flat(
                    self.policy.rows as usize * self.policy.vocab as usize,
                ))?;
        }
        Ok(())
    }

    pub fn step(&mut self, logits: &mut CudaSlice<f32>) -> Result<()> {
        self.apply_history(logits)?;
        let config = LaunchConfig {
            grid_dim: (self.policy.rows, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.argmax)
                .arg(&*logits)
                .arg(&mut self.picks)
                .arg(&(self.policy.rows as i32))
                .arg(&i64::from(self.policy.vocab))
                .arg(&256i32)
                .launch(config)?;
            self.stream
                .launch_builder(&self.softmax)
                .arg(&*logits)
                .arg(&mut self.probabilities)
                .arg(&mut self.probabilities_bf16)
                .arg(&(self.policy.rows as i32))
                .arg(&i64::from(self.policy.vocab))
                .launch(config)?;
        }
        self.fill()
    }

    pub fn fill(&mut self) -> Result<()> {
        unsafe {
            self.stream
                .launch_builder(&self.fill)
                .arg(&self.probabilities)
                .arg(&self.picks)
                .arg(&mut self.block)
                .arg(&mut self.pending)
                .arg(&mut self.state)
                .arg(&(self.policy.rows as i32))
                .arg(&(self.policy.vocab as i32))
                .arg(&self.policy.threshold)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    pub fn commit(&mut self, remaining: u32) -> Result<()> {
        ensure!(
            (1..=self.policy.rows).contains(&remaining),
            "response commit extent mismatch"
        );
        unsafe {
            self.stream
                .launch_builder(&self.commit)
                .arg(&self.block)
                .arg(&self.pending)
                .arg(&mut self.counts)
                .arg(&mut self.state)
                .arg(&self.stops)
                .arg(&(self.policy.eos.len() as i32))
                .arg(&(self.policy.rows as i32))
                .arg(&(self.policy.vocab as i32))
                .arg(&(remaining as i32))
                .arg(&self.policy.capacity)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
