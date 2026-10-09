use super::{Device, cubin, engine::Kv, ops::flat, ops::grid};
use crate::{
    config::Config,
    tree::{Commit, Tree},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, PushKernelArg, sys,
};
use half::bf16;
use std::sync::Arc;

struct Kernels {
    prepare_embed_norm: CudaFunction,
    rope_kv: CudaFunction,
    kv_gather: CudaFunction,
    kv_scatter: CudaFunction,
    hidden_gather: CudaFunction,
    logits: CudaFunction,
}

pub(crate) struct State {
    stream: Arc<CudaStream>,
    kernels: Kernels,
    capacity: usize,
    budget: usize,
    vocab: usize,
    theta: f32,
    tree: Option<Tree>,
    keep: usize,
    tokens: CudaSlice<u32>,
    depth: CudaSlice<i32>,
    anc: CudaSlice<u64>,
    rows: CudaSlice<i32>,
    expected: CudaSlice<i32>,
    prefix: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    slots: CudaSlice<i32>,
    path: CudaSlice<i32>,
    count: CudaSlice<i32>,
    logits_row: CudaSlice<i32>,
    k: CudaSlice<u64>,
    v: CudaSlice<u64>,
    addresses: Vec<(u64, u64)>,
    compact: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
}

/// Workspace buffers the fused tree prepare writes.
pub(crate) struct EmbedOut<'a> {
    pub ids: &'a mut CudaSlice<u32>,
    pub x: &'a mut CudaSlice<bf16>,
    pub y: &'a mut CudaSlice<bf16>,
}

impl State {
    pub(crate) fn reset_setup(&mut self) -> Result<()> {
        self.outside_capture()?;
        self.tree = None;
        self.keep = 0;
        self.stream.memset_zeros(&mut self.tokens)?;
        self.stream.memset_zeros(&mut self.anc)?;
        for value in [
            &mut self.depth,
            &mut self.rows,
            &mut self.expected,
            &mut self.prefix,
            &mut self.positions,
            &mut self.slots,
            &mut self.path,
            &mut self.count,
            &mut self.logits_row,
        ] {
            self.stream.memset_zeros(value)?;
        }
        self.stream.memset_zeros(&mut self.compact)?;
        self.stream.memset_zeros(&mut self.hidden)?;
        Ok(())
    }
    pub(crate) fn new(
        device: &Device,
        config: &Config,
        kv: &[Kv],
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            config.hidden_size == 2048
                && config.num_hidden_layers == 42
                && config.num_attention_heads == 16
                && config.num_key_value_heads == 2
                && config.head_dim == 128
                && config.rope_scaling.is_none(),
            "tree State requires MiniCPM 2048/42/16Q/2KV/D128 without RoPE scaling"
        );
        ensure!(
            matches!(budget, 8 | 16 | 32 | 64),
            "unsupported tree budget"
        );
        ensure!(
            capacity >= budget && capacity <= i32::MAX as usize / 128,
            "tree capacity exceeds positive int stride range"
        );
        ensure!(
            config.vocab_size > 0
                && config.vocab_size <= i32::MAX as usize
                && config.rope_theta.is_finite()
                && config.rope_theta > 0.0,
            "invalid tree vocabulary or RoPE theta"
        );
        let stream = device.stream.clone();
        ensure!(
            stream.capture_status()? == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "tree State allocation is forbidden during stream capture"
        );
        ensure!(kv.len() == 42, "tree target KV requires 42 layers");
        let mut keys = Vec::with_capacity(42);
        let mut values = Vec::with_capacity(42);
        for layer in kv {
            for buffer in [&layer.k, &layer.v] {
                ensure!(
                    buffer.len() == capacity * 256
                        && Arc::ptr_eq(buffer.context(), &device.ctx)
                        && Arc::ptr_eq(buffer.stream(), &stream),
                    "tree KV layout/context/stream does not match State"
                );
            }
            keys.push(layer.k.device_ptr(&stream).0);
            values.push(layer.v.device_ptr(&stream).0);
        }
        let module = cubin::module(&device.ctx, "tree/meta")?;
        let fused = cubin::module(&device.ctx, "tree/tree_rope_kv")?;
        let norm = cubin::module(&device.ctx, "embed_rmsnorm")?;
        let kernels = Kernels {
            prepare_embed_norm: norm.load_function("tree_prepare_embed_norm")?,
            rope_kv: fused.load_function("tree_rope_kv")?,
            kv_gather: module.load_function("tree_kv_gather")?,
            kv_scatter: module.load_function("tree_kv_scatter")?,
            hidden_gather: module.load_function("tree_hidden_gather")?,
            logits: module.load_function("tree_logits")?,
        };
        Ok(Self {
            kernels,
            capacity,
            budget,
            vocab: config.vocab_size,
            theta: config.rope_theta,
            tree: None,
            keep: 0,
            tokens: stream.alloc_zeros(budget)?,
            depth: stream.alloc_zeros(budget)?,
            anc: stream.alloc_zeros(budget)?,
            rows: stream.alloc_zeros(1)?,
            expected: stream.alloc_zeros(1)?,
            prefix: stream.alloc_zeros(1)?,
            positions: stream.alloc_zeros(budget)?,
            slots: stream.alloc_zeros(budget)?,
            path: stream.alloc_zeros(budget)?,
            count: stream.alloc_zeros(1)?,
            logits_row: stream.alloc_zeros(1)?,
            addresses: keys.iter().copied().zip(values.iter().copied()).collect(),
            k: stream.clone_htod(&keys)?,
            v: stream.clone_htod(&values)?,
            compact: stream.alloc_zeros(42 * 512 * budget)?,
            hidden: stream.alloc_zeros(budget * 10240)?,
            stream,
        })
    }

    fn outside_capture(&self) -> Result<()> {
        ensure!(
            self.stream.capture_status()?
                == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "tree metadata upload is forbidden during stream capture"
        );
        Ok(())
    }

    fn buffer<T: DeviceRepr>(&self, buffer: &CudaSlice<T>, len: usize) -> Result<()> {
        ensure!(
            buffer.len() >= len
                && Arc::ptr_eq(buffer.context(), self.stream.context())
                && Arc::ptr_eq(buffer.stream(), &self.stream),
            "tree buffer extent/context/stream mismatch"
        );
        Ok(())
    }

    fn caches(&self, kv: &[Kv]) -> Result<()> {
        ensure!(kv.len() == 42, "tree target KV layer count changed");
        for (layer, &(k, v)) in kv.iter().zip(&self.addresses) {
            self.buffer(&layer.k, self.capacity * 256)?;
            self.buffer(&layer.v, self.capacity * 256)?;
            ensure!(
                layer.k.len() == self.capacity * 256
                    && layer.v.len() == self.capacity * 256
                    && layer.k.device_ptr(&self.stream).0 == k
                    && layer.v.device_ptr(&self.stream).0 == v,
                "tree target KV allocation changed; recreate State and Graphs"
            );
        }
        Ok(())
    }

    pub(crate) fn load(&mut self, tree: &Tree) -> Result<()> {
        self.outside_capture()?;
        let n = tree.nodes().len();
        ensure!(
            n > 0 && n <= self.budget && tree.prefix() <= self.capacity - self.budget,
            "tree rows or padded KV extent exceed State capacity"
        );
        let mut tokens = vec![0u32; self.budget];
        let mut depth = vec![0i32; self.budget];
        let mut anc = vec![0u64; self.budget];
        for (i, node) in tree.nodes().iter().enumerate() {
            ensure!(
                (node.token as usize) < self.vocab
                    && node.depth <= 7
                    && node.position == tree.prefix() + usize::from(node.depth),
                "tree node token/depth/position is unsupported"
            );
            tokens[i] = node.token;
            depth[i] = i32::from(node.depth);
            anc[i] = node.anc;
        }
        self.tree = None;
        self.keep = 0;
        self.stream
            .memcpy_htod(&tokens, &mut self.tokens)
            .context("tree token upload")?;
        self.stream
            .memcpy_htod(&depth, &mut self.depth)
            .context("tree depth upload")?;
        self.stream
            .memcpy_htod(&anc, &mut self.anc)
            .context("tree ancestor upload")?;
        self.stream
            .memcpy_htod(&[n as i32], &mut self.rows)
            .context("tree row count upload")?;
        self.stream
            .memcpy_htod(&[tree.prefix() as i32], &mut self.expected)
            .context("tree prefix upload")?;
        self.stream
            .memcpy_htod(&[0i32], &mut self.count)
            .context("tree commit count reset")?;
        self.tree = Some(tree.clone());
        Ok(())
    }

    pub(crate) fn load_commit(&mut self, commit: &Commit) -> Result<()> {
        self.outside_capture()?;
        let tree = self
            .tree
            .as_ref()
            .context("tree commit requires loaded metadata")?;
        let take = commit.rows.len();
        ensure!(
            take > 0
                && take <= 8
                && take <= tree.nodes().len()
                && commit.output.len() == take
                && commit.rows[0] == 0
                && commit.position == tree.prefix() + take
                && commit.logits_row == commit.rows[take - 1]
                && commit.next == commit.output[take - 1]
                && commit.output.iter().all(|&t| (t as usize) < self.vocab),
            "tree commit extent/cursor/logits row mismatch"
        );
        let mut path = vec![0i32; self.budget];
        for (i, &row) in commit.rows.iter().enumerate() {
            ensure!(
                row < tree.nodes().len(),
                "tree commit row exceeds loaded Tree"
            );
            if i > 0 {
                ensure!(
                    tree.nodes()[row].parent == commit.rows[i - 1] as i32
                        && tree.nodes()[row].token == commit.output[i - 1],
                    "tree commit path is not a matching ancestor chain"
                );
            }
            path[i] = row as i32;
        }
        self.keep = 0;
        self.stream
            .memcpy_htod(&path, &mut self.path)
            .context("tree commit path upload")?;
        self.stream
            .memcpy_htod(&[take as i32], &mut self.count)
            .context("tree commit count upload")?;
        self.stream
            .memcpy_htod(&[commit.logits_row as i32], &mut self.logits_row)
            .context("tree logits row upload")?;
        self.keep = take;
        Ok(())
    }

    /// tree_prepare's metadata writes fused with the embedding gather and the
    /// layer-0 RMSNorm. Replaces (tree_prepare, embed, rms_norm); the metadata
    /// still lands before anything reads it, because this is the same point in
    /// the step where tree_prepare used to run.
    pub(crate) fn prepare_embed_norm(
        &self,
        prefix: &CudaSlice<i32>,
        table: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: EmbedOut<'_>,
        c: &Config,
    ) -> Result<()> {
        ensure!(self.tree.is_some(), "tree prepare requires loaded metadata");
        let EmbedOut { ids, x, y } = out;
        self.buffer(prefix, 1)?;
        self.buffer(ids, self.budget)?;
        self.buffer(x, self.budget * c.hidden_size)?;
        self.buffer(y, self.budget * c.hidden_size)?;
        let budget = self.budget as i32;
        let capacity = self.capacity as i32;
        let dim = c.hidden_size as i32;
        let eps = c.rms_norm_eps;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.prepare_embed_norm)
                .arg(table)
                .arg(&self.tokens)
                .arg(&self.depth)
                .arg(&self.rows)
                .arg(&self.expected)
                .arg(prefix)
                .arg(&self.prefix)
                .arg(ids)
                .arg(&self.positions)
                .arg(&self.slots)
                .arg(weight)
                .arg(x)
                .arg(y)
                .arg(&budget)
                .arg(&capacity)
                .arg(&dim)
                .arg(&eps)
                .launch(grid(self.budget, 256))
                .context("tree metadata preparation, embedding gather and RMSNorm")?;
        }
        Ok(())
    }

    /// Fused tree RoPE and physical KV write: rotates the valid q/k rows in
    /// place, stores K and V into the cache and zeroes the padding rows of both,
    /// replacing (tree_rope, tree_kv_write) with one launch.
    pub(crate) fn rope_kv(
        &self,
        qkv: &mut CudaSlice<bf16>,
        k: &mut CudaSlice<bf16>,
        v: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(self.tree.is_some(), "tree RoPE requires loaded metadata");
        self.buffer(qkv, self.budget * 2560)?;
        self.buffer(k, self.capacity * 256)?;
        self.buffer(v, self.capacity * 256)?;
        ensure!(
            self.addresses
                .contains(&(k.device_ptr(&self.stream).0, v.device_ptr(&self.stream).0)),
            "tree KV write is not a bound target layer"
        );
        let budget = self.budget as i32;
        let capacity = self.capacity as i32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_kv)
                .arg(qkv)
                .arg(&mut *k)
                .arg(&mut *v)
                .arg(&self.positions)
                .arg(&self.rows)
                .arg(&self.prefix)
                .arg(&budget)
                .arg(&capacity)
                .arg(&16i32)
                .arg(&2i32)
                .arg(&128i32)
                .arg(&self.theta)
                .launch(flat(self.budget * 20 * 64))
                .context("tree indexed RoPE and KV write")?;
        }
        Ok(())
    }

    pub(crate) fn gather_path(&mut self, kv: &[Kv], hidden: &CudaSlice<bf16>) -> Result<()> {
        ensure!(self.keep > 0, "tree gather requires loaded commit");
        self.caches(kv)?;
        self.buffer(hidden, self.budget * 10240)?;
        let capacity = self.capacity as i32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_gather)
                .arg(&self.k)
                .arg(&self.v)
                .arg(&mut self.compact)
                .arg(&self.path)
                .arg(&self.count)
                .arg(&self.rows)
                .arg(&self.prefix)
                .arg(&capacity)
                .arg(&42i32)
                .launch(flat(42 * 512 * self.budget))
                .context("tree target KV gather")?;
            self.stream
                .launch_builder(&self.kernels.hidden_gather)
                .arg(hidden)
                .arg(&mut self.hidden)
                .arg(&self.path)
                .arg(&self.count)
                .arg(&self.rows)
                .arg(&10240i32)
                .launch(flat(self.budget * 10240))
                .context("tree five-layer hidden gather")?;
        }
        Ok(())
    }

    pub(crate) fn scatter(&self, kv: &mut [Kv]) -> Result<()> {
        ensure!(self.keep > 0, "tree scatter requires loaded commit");
        self.caches(kv)?;
        let capacity = self.capacity as i32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_scatter)
                .arg(&self.compact)
                .arg(&self.k)
                .arg(&self.v)
                .arg(&self.count)
                .arg(&self.prefix)
                .arg(&capacity)
                .arg(&42i32)
                .launch(flat(42 * 512 * self.budget))
                .context("tree target KV scatter")?;
        }
        Ok(())
    }

    pub(crate) fn select_logits(
        &self,
        source: &CudaSlice<bf16>,
        destination: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            self.keep > 0,
            "tree logits selection requires loaded commit"
        );
        self.buffer(source, self.budget * self.vocab)?;
        self.buffer(destination, self.vocab)?;
        let width = self.vocab as i32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.logits)
                .arg(source)
                .arg(destination)
                .arg(&self.logits_row)
                .arg(&self.rows)
                .arg(&width)
                .launch(flat(self.vocab))
                .context("tree parent-row logits selection")?;
        }
        Ok(())
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }
    pub(crate) fn rows(&self) -> usize {
        self.tree.as_ref().map_or(0, |t| t.nodes().len())
    }
    pub(crate) fn keep(&self) -> usize {
        self.keep
    }
    pub(crate) fn ids(&self) -> &CudaSlice<u32> {
        &self.tokens
    }
    pub(crate) fn ancestors(&self) -> &CudaSlice<u64> {
        &self.anc
    }
    pub(crate) fn positions(&self) -> &CudaSlice<i32> {
        &self.positions
    }
    pub(crate) fn slots(&self) -> &CudaSlice<i32> {
        &self.slots
    }
    pub(crate) fn valid_rows(&self) -> &CudaSlice<i32> {
        &self.rows
    }
    pub(crate) fn prefix(&self) -> &CudaSlice<i32> {
        &self.prefix
    }
    pub(crate) fn compact_hidden(&self) -> &CudaSlice<bf16> {
        &self.hidden
    }
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
