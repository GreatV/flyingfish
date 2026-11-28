//! H3-Image DiT block weight loading, schedule-keyed AdaLN table cache, and pinned-host streaming.

use super::{
    TpuBuffer, TpuDevice, TpuExecutable,
    hlo::{DitBlockHloSpec, emit_bf16_adaln_proj, emit_h3_dit_block},
    pjrt::HostSemantics,
    plan::{BlockPlacement, TpuPlacementPlan},
};
use crate::checkpoint::MappedCheckpoint;
use crate::minimax_h3::{AdalnPlan, AdalnTimeEmbedder, H3Spec, ImageSpec, MODALITIES, adaln_input};
use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{fs::File, path::Path, time::Instant};

/// Cache key of setup-time AdaLN modulation tables derived from the active [`AdalnPlan`].
/// Recompute triggers whenever schedule timesteps, HyperFlow endpoints, or addressed `(timestep, modality)` pairs change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdalnTableKey {
    pub timesteps_bits: Vec<u32>,
    pub endpoints_bits: Vec<u32>,
    pub pairs: Vec<(u32, u8)>,
}

impl AdalnTableKey {
    pub fn from_plan(plan: &AdalnPlan) -> Self {
        Self {
            timesteps_bits: plan.timesteps.iter().map(|v| v.to_bits()).collect(),
            endpoints_bits: plan.endpoints.iter().map(|v| v.to_bits()).collect(),
            pairs: plan.pairs.clone(),
        }
    }
}

/// Schedule-keyed cache holding per-layer compact `[pairs, 6, hidden]` BF16 AdaLN tables
/// and optional `[timesteps, 2, hidden]` `norm_out` table.
#[derive(Default)]
pub struct TpuAdalnCache {
    key: Option<AdalnTableKey>,
    layers: Vec<Vec<bf16>>,
    norm_out: Vec<bf16>,
}

impl TpuAdalnCache {
    pub fn key(&self) -> Option<&AdalnTableKey> {
        self.key.as_ref()
    }

    pub fn layer(&self, index: usize) -> Option<&[bf16]> {
        self.layers.get(index).map(Vec::as_slice)
    }

    pub fn layers(&self) -> &[Vec<bf16>] {
        &self.layers
    }

    pub fn norm_out(&self) -> &[bf16] {
        &self.norm_out
    }

    /// Recomputes cached layer tables when `plan` differs from the stored [`AdalnTableKey`].
    /// Returns `true` when recomputation ran and `false` on a cache hit.
    pub fn ensure_for_plan<F>(&mut self, plan: &AdalnPlan, compute: F) -> Result<bool>
    where
        F: FnOnce(&AdalnPlan) -> Result<Vec<Vec<bf16>>>,
    {
        let next = AdalnTableKey::from_plan(plan);
        if self.key.as_ref() == Some(&next) {
            return Ok(false);
        }
        let layers = compute(plan)?;
        self.key = Some(next);
        self.layers = layers;
        self.norm_out.clear();
        Ok(true)
    }

    /// Recomputes cached per-layer and `norm_out` tables when `plan` differs from [`AdalnTableKey`].
    pub fn ensure_all_for_plan<F>(&mut self, plan: &AdalnPlan, compute: F) -> Result<bool>
    where
        F: FnOnce(&AdalnPlan) -> Result<(Vec<Vec<bf16>>, Vec<bf16>)>,
    {
        let next = AdalnTableKey::from_plan(plan);
        if self.key.as_ref() == Some(&next) && !self.norm_out.is_empty() {
            return Ok(false);
        }
        let (layers, norm_out) = compute(plan)?;
        self.key = Some(next);
        self.layers = layers;
        self.norm_out = norm_out;
        Ok(true)
    }
}

/// Setup-only AdaLN projection weights (`transformer_blocks.{layer}.adaln_proj.linear.{weight,bias}`).
pub struct BlockAdalnProjWeights {
    pub weight: Vec<bf16>,
    pub bias: Vec<bf16>,
}

/// Active BF16 weights of one MiniMax-H3 DiT block (10 tensors, `770,725,376 B` for H3-Image).
pub struct HostDitBlockWeights {
    pub norm1: Vec<bf16>,
    pub to_q: Vec<bf16>,
    pub to_k: Vec<bf16>,
    pub to_v: Vec<bf16>,
    pub norm_q: Vec<bf16>,
    pub norm_k: Vec<bf16>,
    pub to_out: Vec<bf16>,
    pub norm2: Vec<bf16>,
    pub ff_up: Vec<bf16>,
    pub ff_down: Vec<bf16>,
}

impl HostDitBlockWeights {
    pub fn active_bytes(&self) -> u64 {
        let elems = self.norm1.len()
            + self.to_q.len()
            + self.to_k.len()
            + self.to_v.len()
            + self.norm_q.len()
            + self.norm_k.len()
            + self.to_out.len()
            + self.norm2.len()
            + self.ff_up.len()
            + self.ff_down.len();
        (elems as u64) * 2
    }
}

/// Active weights staged in PJRT `"pinned_host"` memory space (or host fallback if absent) for H2D streaming.
enum StagedWeights {
    PinnedHost([TpuBuffer; 10]),
    HostFallback(HostDitBlockWeights),
}

pub struct StreamedDitBlock {
    spec: ImageSpec,
    staged: StagedWeights,
    pub active_bytes: u64,
    pub staged_in_pinned_host: bool,
}

#[derive(Clone, Copy)]
pub struct BlockRunInputs<'a> {
    pub x: &'a TpuBuffer,
    pub adaln: &'a TpuBuffer,
    pub row_pairs: &'a TpuBuffer,
    pub rotary: &'a TpuBuffer,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct StreamRunTiming {
    pub h2d_ms: f64,
    pub execute_ms: f64,
    pub active_bytes: u64,
    pub used_pinned_host_copy: bool,
}

impl StreamedDitBlock {
    /// Stages active weights into `"pinned_host"` when the device exposes it, otherwise retains host buffers.
    pub fn stage(dev: &TpuDevice, spec: &ImageSpec, weights: HostDitBlockWeights) -> Result<Self> {
        let active_bytes = weights.active_bytes();
        let d = spec.hidden as i64;
        let w = (spec.heads * spec.head_dim) as i64;
        let hd = spec.head_dim as i64;
        let f = spec.ffn as i64;
        let f2 = (2 * spec.ffn) as i64;
        if let Some(pinned) = dev.pinned_host_memory() {
            let sem = HostSemantics::ImmutableOnlyDuringCall;
            let bufs = [
                dev.upload_bf16_to_memory(&weights.norm1, &[d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.to_q, &[w, d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.to_k, &[w, d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.to_v, &[w, d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.norm_q, &[hd], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.norm_k, &[hd], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.to_out, &[d, w], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.norm2, &[d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.ff_up, &[f2, d], sem, Some(pinned))?,
                dev.upload_bf16_to_memory(&weights.ff_down, &[d, f], sem, Some(pinned))?,
            ];
            Ok(Self {
                spec: spec.clone(),
                staged: StagedWeights::PinnedHost(bufs),
                active_bytes,
                staged_in_pinned_host: true,
            })
        } else {
            Ok(Self {
                spec: spec.clone(),
                staged: StagedWeights::HostFallback(weights),
                active_bytes,
                staged_in_pinned_host: false,
            })
        }
    }

    /// Streams weights into HBM, runs `exec`, drops all temporary HBM weight buffers, and returns the output.
    pub fn execute_streamed(
        &self,
        dev: &TpuDevice,
        exec: &TpuExecutable,
        inputs: &BlockRunInputs<'_>,
    ) -> Result<(TpuBuffer, StreamRunTiming)> {
        let t_h2d = Instant::now();
        let device_weights: Vec<TpuBuffer> = match &self.staged {
            StagedWeights::PinnedHost(pinned) => pinned
                .iter()
                .map(|b| dev.copy_buffer_to_default_hbm(b))
                .collect::<Result<Vec<_>>>()?,
            StagedWeights::HostFallback(w) => {
                let d = self.spec.hidden as i64;
                let width = (self.spec.heads * self.spec.head_dim) as i64;
                let hd = self.spec.head_dim as i64;
                let f = self.spec.ffn as i64;
                let f2 = (2 * self.spec.ffn) as i64;
                let sem = HostSemantics::ImmutableUntilTransferCompletes;
                vec![
                    dev.upload_bf16(&w.norm1, &[d], sem)?,
                    dev.upload_bf16(&w.to_q, &[width, d], sem)?,
                    dev.upload_bf16(&w.to_k, &[width, d], sem)?,
                    dev.upload_bf16(&w.to_v, &[width, d], sem)?,
                    dev.upload_bf16(&w.norm_q, &[hd], sem)?,
                    dev.upload_bf16(&w.norm_k, &[hd], sem)?,
                    dev.upload_bf16(&w.to_out, &[d, width], sem)?,
                    dev.upload_bf16(&w.norm2, &[d], sem)?,
                    dev.upload_bf16(&w.ff_up, &[f2, d], sem)?,
                    dev.upload_bf16(&w.ff_down, &[d, f], sem)?,
                ]
            }
        };
        let h2d_ms = t_h2d.elapsed().as_secs_f64() * 1e3;

        let mut args: Vec<&TpuBuffer> = Vec::with_capacity(14);
        args.push(inputs.x);
        args.push(inputs.adaln);
        args.push(inputs.row_pairs);
        args.push(inputs.rotary);
        for wb in &device_weights {
            args.push(wb);
        }

        let out_bytes = inputs.rows * self.spec.hidden * 2;
        let t_exec = Instant::now();
        let mut outs = exec.execute(&args, &[out_bytes])?;
        let execute_ms = t_exec.elapsed().as_secs_f64() * 1e3;
        drop(device_weights);

        let out = outs.pop().context("DiT block returned zero outputs")?;
        Ok((
            out,
            StreamRunTiming {
                h2d_ms,
                execute_ms,
                active_bytes: self.active_bytes,
                used_pinned_host_copy: self.staged_in_pinned_host,
            },
        ))
    }
}

/// Compiles one MiniMax-H3-Image DiT main block for `(rows, adaln_pairs, head_chunk)`.
pub fn compile_h3_dit_block(
    dev: &TpuDevice,
    spec: &ImageSpec,
    rows: usize,
    adaln_pairs: usize,
    head_chunk: usize,
) -> Result<TpuExecutable> {
    let hlo_spec = DitBlockHloSpec {
        rows,
        hidden: spec.hidden,
        heads: spec.heads,
        head_dim: spec.head_dim,
        ffn: spec.ffn,
        rotary_dim: 6 * spec.rope_freq_dim,
        head_chunk,
        adaln_pairs,
        norm_eps: spec.norm_eps,
        qk_norm_eps: spec.qk_norm_eps,
    };
    let mlir = emit_h3_dit_block(&hlo_spec)?;
    dev.compile_mlir(&mlir)
}

/// Projects `adaln_input(timesteps)` through `proj` on TPU for the modality tags addressed by `plan`,
/// returning compact `[plan.pairs.len(), 6, hidden]` BF16 rows in `plan.pairs` order.
pub fn compute_block_adaln_table(
    dev: &TpuDevice,
    spec: &ImageSpec,
    time_embedder: &AdalnTimeEmbedder,
    proj: &BlockAdalnProjWeights,
    plan: &AdalnPlan,
) -> Result<Vec<bf16>> {
    ensure!(!plan.timesteps.is_empty(), "AdalnPlan has zero timesteps");
    ensure!(!plan.pairs.is_empty(), "AdalnPlan has zero pairs");
    let width = 6 * spec.hidden;
    let tags = distinct_tags(&plan.pairs);
    ensure!(
        tags.iter().all(|&t| u32::from(t) < MODALITIES),
        "AdalnPlan tag out of range: {tags:?}"
    );

    let x_bf16 = adaln_input(&image_to_h3_spec(spec), time_embedder, &plan.timesteps);
    let compact_out = tags.len() * width;
    let mut compact_w = Vec::with_capacity(compact_out * spec.time_dim);
    let mut compact_b = Vec::with_capacity(compact_out);
    for &tag in &tags {
        let row_start = usize::from(tag) * width;
        let w_start = row_start * spec.time_dim;
        let w_end = (row_start + width) * spec.time_dim;
        compact_w.extend_from_slice(&proj.weight[w_start..w_end]);
        compact_b.extend_from_slice(&proj.bias[row_start..row_start + width]);
    }

    let t = plan.timesteps.len();
    let mlir = emit_bf16_adaln_proj(t, spec.time_dim, compact_out)?;
    let exec = dev.compile_mlir(&mlir)?;
    let sem = HostSemantics::ImmutableUntilTransferCompletes;
    let x_buf = dev.upload_bf16(&x_bf16, &[t as i64, spec.time_dim as i64], sem)?;
    let w_buf = dev.upload_bf16(&compact_w, &[compact_out as i64, spec.time_dim as i64], sem)?;
    let b_buf = dev.upload_bf16(&compact_b, &[compact_out as i64], sem)?;
    let outs = exec.execute(&[&x_buf, &w_buf, &b_buf], &[t * compact_out * 2])?;
    let y = outs[0].download_bf16()?;
    drop((outs, x_buf, w_buf, b_buf, exec));

    let mut table = Vec::with_capacity(plan.pairs.len() * width);
    for &(t_idx, tag) in &plan.pairs {
        let tag_pos = tags
            .iter()
            .position(|&candidate| candidate == tag)
            .with_context(|| format!("tag {tag} missing from compact set {tags:?}"))?;
        let off = (t_idx as usize * tags.len() + tag_pos) * width;
        table.extend_from_slice(&y[off..off + width]);
    }
    Ok(table)
}

/// Projects `adaln_input(timesteps)` through every block's `adaln_proj.linear` and `norm_out.linear`
/// on TPU, reusing one compiled executable for all blocks. Returns `(layer_tables, norm_out_table)`.
pub fn compute_all_adaln_tables(
    dev: &TpuDevice,
    spec: &ImageSpec,
    ckpt: &MappedCheckpoint,
    plan: &AdalnPlan,
) -> Result<(Vec<Vec<bf16>>, Vec<bf16>)> {
    ensure!(!plan.timesteps.is_empty(), "AdalnPlan has zero timesteps");
    ensure!(!plan.pairs.is_empty(), "AdalnPlan has zero pairs");
    let width = 6 * spec.hidden;
    let tags = distinct_tags(&plan.pairs);
    ensure!(
        tags.iter().all(|&t| u32::from(t) < MODALITIES),
        "AdalnPlan tag out of range: {tags:?}"
    );
    let time_embedder = load_time_embedder(ckpt, spec)?;
    let x_bf16 = adaln_input(&image_to_h3_spec(spec), &time_embedder, &plan.timesteps);
    let t = plan.timesteps.len();
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let x_buf = dev.upload_bf16(&x_bf16, &[t as i64, spec.time_dim as i64], sem)?;

    let compact_out = tags.len() * width;
    let block_exec = dev.compile_mlir(&emit_bf16_adaln_proj(t, spec.time_dim, compact_out)?)?;
    let full_out = width * MODALITIES as usize;
    let mut layer_tables = Vec::with_capacity(spec.layers);
    for layer in 0..spec.layers {
        let prefix = format!("transformer_blocks.{layer}.adaln_proj.linear");
        let w_view = ckpt.tensor(&format!("{prefix}.weight"))?;
        let b_view = ckpt.tensor(&format!("{prefix}.bias"))?;
        let (compact_w, compact_b) = extract_tag_adaln_rows(
            &w_view,
            &b_view,
            &tags,
            (full_out, width, spec.time_dim),
            &prefix,
        )?;
        let w_buf =
            dev.upload_bf16(&compact_w, &[compact_out as i64, spec.time_dim as i64], sem)?;
        let b_buf = dev.upload_bf16(&compact_b, &[compact_out as i64], sem)?;
        let outs = block_exec.execute(&[&x_buf, &w_buf, &b_buf], &[t * compact_out * 2])?;
        let y = outs[0].download_bf16()?;
        drop((outs, w_buf, b_buf));
        let mut table = Vec::with_capacity(plan.pairs.len() * width);
        for &(t_idx, tag) in &plan.pairs {
            let tag_pos = tags
                .iter()
                .position(|&candidate| candidate == tag)
                .with_context(|| format!("tag {tag} missing from compact set {tags:?}"))?;
            let off = (t_idx as usize * tags.len() + tag_pos) * width;
            table.extend_from_slice(&y[off..off + width]);
        }
        layer_tables.push(table);
    }
    drop(block_exec);

    let norm_out_table = project_norm_out_adaln(dev, spec, ckpt, &x_buf, t)?;
    drop(x_buf);
    Ok((layer_tables, norm_out_table))
}

fn project_norm_out_adaln(
    dev: &TpuDevice,
    spec: &ImageSpec,
    ckpt: &MappedCheckpoint,
    x_buf: &TpuBuffer,
    t: usize,
) -> Result<Vec<bf16>> {
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    let norm_out_dim = 2 * spec.hidden;
    let norm_w = read_bf16_ckpt(
        ckpt,
        "norm_out.linear.weight",
        &[norm_out_dim, spec.time_dim],
    )?;
    let norm_b = read_bf16_ckpt(ckpt, "norm_out.linear.bias", &[norm_out_dim])?;
    let norm_exec = dev.compile_mlir(&emit_bf16_adaln_proj(t, spec.time_dim, norm_out_dim)?)?;
    let nw_buf = dev.upload_bf16(&norm_w, &[norm_out_dim as i64, spec.time_dim as i64], sem)?;
    let nb_buf = dev.upload_bf16(&norm_b, &[norm_out_dim as i64], sem)?;
    let outs = norm_exec.execute(&[x_buf, &nw_buf, &nb_buf], &[t * norm_out_dim * 2])?;
    let norm_out_table = outs[0].download_bf16()?;
    drop((outs, nw_buf, nb_buf, norm_exec));
    Ok(norm_out_table)
}

fn extract_tag_adaln_rows(
    w_view: &TensorView<'_>,
    b_view: &TensorView<'_>,
    tags: &[u8],
    (full_out, width, time_dim): (usize, usize, usize),
    prefix: &str,
) -> Result<(Vec<bf16>, Vec<bf16>)> {
    ensure!(
        w_view.dtype() == Dtype::BF16 && w_view.shape() == [full_out, time_dim],
        "{prefix}.weight dtype/shape mismatch: {:?} {:?}",
        w_view.dtype(),
        w_view.shape()
    );
    ensure!(
        b_view.dtype() == Dtype::BF16 && b_view.shape() == [full_out],
        "{prefix}.bias dtype/shape mismatch: {:?} {:?}",
        b_view.dtype(),
        b_view.shape()
    );
    let w_bytes = w_view.data();
    let b_bytes = b_view.data();
    let mut compact_w = Vec::with_capacity(tags.len() * width * time_dim);
    let mut compact_b = Vec::with_capacity(tags.len() * width);
    for &tag in tags {
        let row_start = usize::from(tag) * width;
        let w_byte_start = row_start * time_dim * 2;
        let w_byte_end = (row_start + width) * time_dim * 2;
        for &chunk in w_bytes[w_byte_start..w_byte_end].as_chunks::<2>().0 {
            let v = bf16::from_le_bytes(chunk);
            ensure!(v.is_finite(), "{prefix}.weight contains non-finite value");
            compact_w.push(v);
        }
        let b_byte_start = row_start * 2;
        let b_byte_end = (row_start + width) * 2;
        for &chunk in b_bytes[b_byte_start..b_byte_end].as_chunks::<2>().0 {
            let v = bf16::from_le_bytes(chunk);
            ensure!(v.is_finite(), "{prefix}.bias contains non-finite value");
            compact_b.push(v);
        }
    }
    Ok((compact_w, compact_b))
}

/// Deterministic HBM staging action for one block in a denoising pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefetchStep {
    pub block: usize,
    pub load_before: Option<usize>,
    pub prefetch_during: Option<usize>,
    pub release_after: bool,
}

/// Derives per-block load, overlapped prefetch, and HBM release actions bounded by `prefetch_depth`.
pub fn plan_prefetch_schedule(
    placements: &[BlockPlacement],
    prefetch_depth: usize,
) -> Result<Vec<PrefetchStep>> {
    ensure!(!placements.is_empty(), "empty placement schedule");
    let any_streamed = placements.iter().any(|&p| p != BlockPlacement::HbmResident);
    if prefetch_depth == 0 {
        ensure!(
            !any_streamed,
            "prefetch_depth=0 cannot execute streamed blocks"
        );
    }
    let mut steps = Vec::with_capacity(placements.len());
    let mut prefetched: Option<usize> = None;
    for (block, &placement) in placements.iter().enumerate() {
        let is_streamed = placement != BlockPlacement::HbmResident;
        let load_before = if is_streamed && prefetched != Some(block) {
            Some(block)
        } else {
            None
        };
        if prefetched == Some(block) {
            prefetched = None;
        }
        let prefetch_during = if prefetch_depth >= 2
            && block + 1 < placements.len()
            && placements[block + 1] != BlockPlacement::HbmResident
        {
            prefetched = Some(block + 1);
            Some(block + 1)
        } else {
            None
        };
        steps.push(PrefetchStep {
            block,
            load_before,
            prefetch_during,
            release_after: is_streamed,
        });
    }
    Ok(steps)
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct StackStepStats {
    pub hbm_blocks_run: usize,
    pub pinned_blocks_run: usize,
    pub pageable_blocks_run: usize,
    pub disk_blocks_run: usize,
    pub overlapped_prefetches: usize,
    pub peak_staging_slots: usize,
}

enum PlacedDitBlock {
    HbmResident([TpuBuffer; 10]),
    PinnedHost([TpuBuffer; 10]),
    PageableHost(HostDitBlockWeights),
    DiskStreamed { layer: usize },
}

/// Multi-tier 50-block DiT stack executing the setup-derived [`TpuPlacementPlan`] with async prefetch overlap.
pub struct DitBlockStack {
    spec: ImageSpec,
    plan: TpuPlacementPlan,
    schedule: Vec<PrefetchStep>,
    blocks: Vec<PlacedDitBlock>,
    adaln_hbm: Vec<TpuBuffer>,
}

impl DitBlockStack {
    pub fn load(
        dev: &TpuDevice,
        spec: &ImageSpec,
        ckpt: &MappedCheckpoint,
        plan: TpuPlacementPlan,
        adaln_tables: &[Vec<bf16>],
        adaln_pairs: usize,
    ) -> Result<Self> {
        ensure!(
            plan.blocks.len() == spec.layers && adaln_tables.len() == spec.layers,
            "DitBlockStack layer count mismatch: plan={} adaln={} spec={}",
            plan.blocks.len(),
            adaln_tables.len(),
            spec.layers
        );
        let schedule = plan_prefetch_schedule(&plan.blocks, plan.prefetch_depth)?;
        let sem = HostSemantics::ImmutableOnlyDuringCall;
        let mut adaln_hbm = Vec::with_capacity(spec.layers);
        for (layer, table) in adaln_tables.iter().enumerate() {
            adaln_hbm.push(
                dev.upload_bf16(table, &[adaln_pairs as i64, 6, spec.hidden as i64], sem)
                    .with_context(|| format!("upload AdaLN table layer {layer}"))?,
            );
        }
        let mut blocks = Vec::with_capacity(spec.layers);
        for (layer, &tier) in plan.blocks.iter().enumerate() {
            let placed = match tier {
                BlockPlacement::HbmResident => {
                    let w = load_block_active_weights(ckpt, spec, layer)?;
                    let bufs = upload_block_weights(dev, spec, &w, None)?;
                    drop(w);
                    PlacedDitBlock::HbmResident(bufs)
                }
                BlockPlacement::PinnedHostStreamed => {
                    let pinned = dev
                        .pinned_host_memory()
                        .context("plan assigned PinnedHostStreamed without pinned_host memory")?;
                    let w = load_block_active_weights(ckpt, spec, layer)?;
                    let bufs = upload_block_weights(dev, spec, &w, Some(pinned))?;
                    drop(w);
                    PlacedDitBlock::PinnedHost(bufs)
                }
                BlockPlacement::PageableHostStreamed => {
                    PlacedDitBlock::PageableHost(load_block_active_weights(ckpt, spec, layer)?)
                }
                BlockPlacement::DiskStreamed => PlacedDitBlock::DiskStreamed { layer },
            };
            blocks.push(placed);
        }
        Ok(Self {
            spec: spec.clone(),
            plan,
            schedule,
            blocks,
            adaln_hbm,
        })
    }

    pub fn plan(&self) -> &TpuPlacementPlan {
        &self.plan
    }

    fn stage_block_to_hbm_async(
        &self,
        dev: &TpuDevice,
        ckpt: &MappedCheckpoint,
        layer: usize,
    ) -> Result<[TpuBuffer; 10]> {
        match &self.blocks[layer] {
            PlacedDitBlock::PinnedHost(pinned) => {
                let mut out = Vec::with_capacity(10);
                for buf in pinned {
                    out.push(dev.copy_buffer_to_default_hbm_async(buf)?);
                }
                out.try_into()
                    .map_err(|_| anyhow::anyhow!("block weight buffer count != 10"))
            }
            PlacedDitBlock::PageableHost(w) => upload_block_weights(dev, &self.spec, w, None),
            PlacedDitBlock::DiskStreamed { layer: l } => {
                let w = load_block_active_weights(ckpt, &self.spec, *l)?;
                let bufs = upload_block_weights(dev, &self.spec, &w, None)?;
                drop(w);
                Ok(bufs)
            }
            PlacedDitBlock::HbmResident(_) => {
                anyhow::bail!("stage_block_to_hbm_async called on HbmResident layer {layer}")
            }
        }
    }

    /// Runs all 50 blocks for one denoising step, overlapping next-block H2D copies with TPU compute.
    pub fn run_step(
        &self,
        dev: &TpuDevice,
        exec: &TpuExecutable,
        ckpt: &MappedCheckpoint,
        mut x: TpuBuffer,
        (row_pairs, rotary, rows): (&TpuBuffer, &TpuBuffer, usize),
    ) -> Result<(TpuBuffer, StackStepStats)> {
        let out_bytes = rows * self.spec.hidden * 2;
        let mut stats = StackStepStats::default();
        let mut active_slot: Option<(usize, [TpuBuffer; 10])> = None;
        let mut prefetched_slot: Option<(usize, [TpuBuffer; 10])> = None;

        for step in &self.schedule {
            let b = step.block;
            if let Some(load_b) = step.load_before {
                ensure!(
                    load_b == b && active_slot.is_none(),
                    "prefetch schedule load_before mismatch at block {b}"
                );
                let staged = self.stage_block_to_hbm_async(dev, ckpt, b)?;
                active_slot = Some((b, staged));
            } else if step.release_after {
                let (pf_b, staged) = prefetched_slot
                    .take()
                    .with_context(|| format!("missing prefetched weights for block {b}"))?;
                ensure!(pf_b == b, "prefetched block {pf_b} != {b}");
                active_slot = Some((b, staged));
            }

            if let Some((_, ref staged)) = active_slot {
                for buf in staged {
                    buf.await_ready()
                        .with_context(|| format!("await staged weights for block {b}"))?;
                }
            }

            let live_before =
                usize::from(active_slot.is_some()) + usize::from(prefetched_slot.is_some());
            stats.peak_staging_slots = stats.peak_staging_slots.max(live_before);

            let weights_ref: &[TpuBuffer; 10] = match &self.blocks[b] {
                PlacedDitBlock::HbmResident(bufs) => {
                    stats.hbm_blocks_run += 1;
                    bufs
                }
                PlacedDitBlock::PinnedHost(_) => {
                    stats.pinned_blocks_run += 1;
                    &active_slot.as_ref().context("missing active slot")?.1
                }
                PlacedDitBlock::PageableHost(_) => {
                    stats.pageable_blocks_run += 1;
                    &active_slot.as_ref().context("missing active slot")?.1
                }
                PlacedDitBlock::DiskStreamed { .. } => {
                    stats.disk_blocks_run += 1;
                    &active_slot.as_ref().context("missing active slot")?.1
                }
            };

            let mut args: Vec<&TpuBuffer> = Vec::with_capacity(14);
            args.push(&x);
            args.push(&self.adaln_hbm[b]);
            args.push(row_pairs);
            args.push(rotary);
            for wb in weights_ref {
                args.push(wb);
            }

            let (mut outs, ev) = exec.execute_async(&args, &[out_bytes])?;
            if let Some(next_b) = step.prefetch_during {
                ensure!(
                    prefetched_slot.is_none(),
                    "prefetch slot already occupied before block {next_b}"
                );
                let staged = self.stage_block_to_hbm_async(dev, ckpt, next_b)?;
                prefetched_slot = Some((next_b, staged));
                stats.overlapped_prefetches += 1;
                let live_during =
                    usize::from(active_slot.is_some()) + usize::from(prefetched_slot.is_some());
                stats.peak_staging_slots = stats.peak_staging_slots.max(live_during);
            }
            ev.wait()
                .with_context(|| format!("DiT block {b} execute"))?;
            x = outs.pop().context("DiT block returned zero outputs")?;
            if step.release_after {
                active_slot = None;
            }
        }
        ensure!(
            active_slot.is_none() && prefetched_slot.is_none(),
            "DitBlockStack leaked staging slot after step"
        );
        ensure!(
            stats.peak_staging_slots <= self.plan.prefetch_depth,
            "staging slots {} exceeded prefetch_depth {}",
            stats.peak_staging_slots,
            self.plan.prefetch_depth
        );
        Ok((x, stats))
    }
}

pub(crate) fn upload_block_weights(
    dev: &TpuDevice,
    spec: &ImageSpec,
    weights: &HostDitBlockWeights,
    target_memory: Option<&super::pjrt::MemoryKindInfo>,
) -> Result<[TpuBuffer; 10]> {
    let d = spec.hidden as i64;
    let w = (spec.heads * spec.head_dim) as i64;
    let hd = spec.head_dim as i64;
    let f = spec.ffn as i64;
    let f2 = (2 * spec.ffn) as i64;
    let sem = HostSemantics::ImmutableOnlyDuringCall;
    Ok([
        dev.upload_bf16_to_memory(&weights.norm1, &[d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.to_q, &[w, d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.to_k, &[w, d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.to_v, &[w, d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.norm_q, &[hd], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.norm_k, &[hd], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.to_out, &[d, w], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.norm2, &[d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.ff_up, &[f2, d], sem, target_memory)?,
        dev.upload_bf16_to_memory(&weights.ff_down, &[d, f], sem, target_memory)?,
    ])
}

/// Loads `time_embedder.linear_{1,2}.{weight,bias}` (F32) from a multi-shard checkpoint.
pub(crate) fn load_time_embedder(
    ckpt: &MappedCheckpoint,
    spec: &ImageSpec,
) -> Result<AdalnTimeEmbedder> {
    Ok(AdalnTimeEmbedder {
        linear_1_weight: read_f32_ckpt(
            ckpt,
            "time_embedder.linear_1.weight",
            &[spec.time_hidden, spec.freq_dim],
        )?,
        linear_1_bias: read_f32_ckpt(ckpt, "time_embedder.linear_1.bias", &[spec.time_hidden])?,
        linear_2_weight: read_f32_ckpt(
            ckpt,
            "time_embedder.linear_2.weight",
            &[spec.time_dim, spec.time_hidden],
        )?,
        linear_2_bias: read_f32_ckpt(ckpt, "time_embedder.linear_2.bias", &[spec.time_dim])?,
    })
}

/// Loads the 10 active BF16 tensors of `transformer_blocks.{layer}` from a multi-shard checkpoint.
pub(crate) fn load_block_active_weights(
    ckpt: &MappedCheckpoint,
    spec: &ImageSpec,
    layer: usize,
) -> Result<HostDitBlockWeights> {
    load_prefixed_block_weights(ckpt, spec, &format!("transformer_blocks.{layer}"))
}

/// Loads the 10 active BF16 tensors of `token_refiner.refiner_blocks.{layer}` from a multi-shard checkpoint.
pub(crate) fn load_refiner_block_weights(
    ckpt: &MappedCheckpoint,
    spec: &ImageSpec,
    layer: usize,
) -> Result<HostDitBlockWeights> {
    load_prefixed_block_weights(ckpt, spec, &format!("token_refiner.refiner_blocks.{layer}"))
}

fn load_prefixed_block_weights(
    ckpt: &MappedCheckpoint,
    spec: &ImageSpec,
    p: &str,
) -> Result<HostDitBlockWeights> {
    let d = spec.hidden;
    let w = spec.heads * spec.head_dim;
    let hd = spec.head_dim;
    let f = spec.ffn;
    Ok(HostDitBlockWeights {
        norm1: read_bf16_ckpt(ckpt, &format!("{p}.norm1.weight"), &[d])?,
        to_q: read_bf16_ckpt(ckpt, &format!("{p}.attn.to_q.weight"), &[w, d])?,
        to_k: read_bf16_ckpt(ckpt, &format!("{p}.attn.to_k.weight"), &[w, d])?,
        to_v: read_bf16_ckpt(ckpt, &format!("{p}.attn.to_v.weight"), &[w, d])?,
        norm_q: read_bf16_ckpt(ckpt, &format!("{p}.attn.norm_q.weight"), &[hd])?,
        norm_k: read_bf16_ckpt(ckpt, &format!("{p}.attn.norm_k.weight"), &[hd])?,
        to_out: read_bf16_ckpt(ckpt, &format!("{p}.attn.to_out.0.weight"), &[d, w])?,
        norm2: read_bf16_ckpt(ckpt, &format!("{p}.norm2.weight"), &[d])?,
        ff_up: read_bf16_ckpt(ckpt, &format!("{p}.ff.net.0.proj.weight"), &[2 * f, d])?,
        ff_down: read_bf16_ckpt(ckpt, &format!("{p}.ff.net.2.weight"), &[d, f])?,
    })
}

/// Loads `time_embedder.linear_{1,2}.{weight,bias}` (F32) read-only from a safetensors shard.
pub fn load_time_embedder_from_shard(
    shard_path: &Path,
    spec: &ImageSpec,
) -> Result<AdalnTimeEmbedder> {
    let file = File::open(shard_path).with_context(|| format!("open {}", shard_path.display()))?;
    // SAFETY: read-only memory map of an immutable checkpoint shard.
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors {}", shard_path.display()))?;
    Ok(AdalnTimeEmbedder {
        linear_1_weight: read_f32_tensor(
            &st,
            "time_embedder.linear_1.weight",
            &[spec.time_hidden, spec.freq_dim],
        )?,
        linear_1_bias: read_f32_tensor(&st, "time_embedder.linear_1.bias", &[spec.time_hidden])?,
        linear_2_weight: read_f32_tensor(
            &st,
            "time_embedder.linear_2.weight",
            &[spec.time_dim, spec.time_hidden],
        )?,
        linear_2_bias: read_f32_tensor(&st, "time_embedder.linear_2.bias", &[spec.time_dim])?,
    })
}

/// Loads `transformer_blocks.{layer}.adaln_proj.linear.{weight,bias}` (BF16) read-only from a shard.
pub fn load_block_adaln_proj_from_shard(
    shard_path: &Path,
    spec: &ImageSpec,
    layer: usize,
) -> Result<BlockAdalnProjWeights> {
    let file = File::open(shard_path).with_context(|| format!("open {}", shard_path.display()))?;
    // SAFETY: read-only memory map of an immutable checkpoint shard.
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors {}", shard_path.display()))?;
    let out_dim = 6 * spec.hidden * MODALITIES as usize;
    let prefix = format!("transformer_blocks.{layer}.adaln_proj.linear");
    Ok(BlockAdalnProjWeights {
        weight: read_bf16_tensor(&st, &format!("{prefix}.weight"), &[out_dim, spec.time_dim])?,
        bias: read_bf16_tensor(&st, &format!("{prefix}.bias"), &[out_dim])?,
    })
}

/// Loads the 10 active BF16 tensors of `transformer_blocks.{layer}` read-only from a safetensors shard.
pub fn load_block_active_weights_from_shard(
    shard_path: &Path,
    spec: &ImageSpec,
    layer: usize,
) -> Result<HostDitBlockWeights> {
    let file = File::open(shard_path).with_context(|| format!("open {}", shard_path.display()))?;
    // SAFETY: read-only memory map of an immutable checkpoint shard.
    let mmap = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors {}", shard_path.display()))?;
    let d = spec.hidden;
    let w = spec.heads * spec.head_dim;
    let hd = spec.head_dim;
    let f = spec.ffn;
    let p = format!("transformer_blocks.{layer}");
    Ok(HostDitBlockWeights {
        norm1: read_bf16_tensor(&st, &format!("{p}.norm1.weight"), &[d])?,
        to_q: read_bf16_tensor(&st, &format!("{p}.attn.to_q.weight"), &[w, d])?,
        to_k: read_bf16_tensor(&st, &format!("{p}.attn.to_k.weight"), &[w, d])?,
        to_v: read_bf16_tensor(&st, &format!("{p}.attn.to_v.weight"), &[w, d])?,
        norm_q: read_bf16_tensor(&st, &format!("{p}.attn.norm_q.weight"), &[hd])?,
        norm_k: read_bf16_tensor(&st, &format!("{p}.attn.norm_k.weight"), &[hd])?,
        to_out: read_bf16_tensor(&st, &format!("{p}.attn.to_out.0.weight"), &[d, w])?,
        norm2: read_bf16_tensor(&st, &format!("{p}.norm2.weight"), &[d])?,
        ff_up: read_bf16_tensor(&st, &format!("{p}.ff.net.0.proj.weight"), &[2 * f, d])?,
        ff_down: read_bf16_tensor(&st, &format!("{p}.ff.net.2.weight"), &[d, f])?,
    })
}

fn distinct_tags(pairs: &[(u32, u8)]) -> Vec<u8> {
    let mut tags: Vec<u8> = pairs.iter().map(|&(_, tag)| tag).collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

pub(crate) fn image_to_h3_spec(spec: &ImageSpec) -> H3Spec {
    H3Spec {
        hidden: spec.hidden,
        heads: spec.heads,
        head_dim: spec.head_dim,
        layers: spec.layers,
        refiner_layers: spec.refiner_layers,
        ffn: spec.ffn,
        in_channels: spec.in_channels,
        audio_in_channels: 0,
        patch: spec.patch,
        text_dim: spec.text_dim,
        freq_dim: spec.freq_dim,
        time_hidden: spec.time_hidden,
        time_dim: spec.time_dim,
        rope_freq_dim: spec.rope_freq_dim,
        rope_theta: spec.rope_theta,
        norm_eps: spec.norm_eps,
        qk_norm_eps: spec.qk_norm_eps,
        final_norm_eps: spec.final_norm_eps,
        hyperflow_gate: None,
        dropped_blocks: None,
        lowrank_ranks: None,
        int8_convrot: None,
    }
}

pub(crate) fn read_bf16_ckpt(
    ckpt: &MappedCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<bf16>> {
    let view = ckpt.tensor(name)?;
    read_bf16_view(&view, name, expected_shape)
}

pub(crate) fn read_f32_ckpt(
    ckpt: &MappedCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>> {
    let view = ckpt.tensor(name)?;
    read_f32_view(&view, name, expected_shape)
}

pub(crate) fn read_f32_narrow_bf16_ckpt(
    ckpt: &MappedCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<bf16>> {
    let vals = read_f32_ckpt(ckpt, name, expected_shape)?;
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        let b = bf16::from_f32(v);
        ensure!(
            b.is_finite(),
            "tensor {name} narrowed to non-finite BF16 from {v}"
        );
        out.push(b);
    }
    Ok(out)
}

fn read_bf16_tensor(
    st: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<bf16>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    read_bf16_view(&view, name, expected_shape)
}

fn read_bf16_view(
    view: &TensorView<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<bf16>> {
    ensure!(
        view.dtype() == Dtype::BF16,
        "tensor {name} dtype {:?} != BF16",
        view.dtype()
    );
    ensure!(
        view.shape() == expected_shape,
        "tensor {name} shape {:?} != {expected_shape:?}",
        view.shape()
    );
    let data = view.data();
    ensure!(
        data.len().is_multiple_of(2),
        "tensor {name} byte len {} is odd",
        data.len()
    );
    let mut out = Vec::with_capacity(data.len() / 2);
    for &chunk in data.as_chunks::<2>().0 {
        let v = bf16::from_le_bytes(chunk);
        ensure!(v.is_finite(), "tensor {name} contains non-finite value");
        out.push(v);
    }
    Ok(out)
}

fn read_f32_tensor(st: &SafeTensors<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    read_f32_view(&view, name, expected_shape)
}

fn read_f32_view(view: &TensorView<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    ensure!(
        view.dtype() == Dtype::F32,
        "tensor {name} dtype {:?} != F32",
        view.dtype()
    );
    ensure!(
        view.shape() == expected_shape,
        "tensor {name} shape {:?} != {expected_shape:?}",
        view.shape()
    );
    let data = view.data();
    ensure!(
        data.len().is_multiple_of(4),
        "tensor {name} byte len {} not divisible by 4",
        data.len()
    );
    let mut out = Vec::with_capacity(data.len() / 4);
    for &chunk in data.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(chunk);
        ensure!(v.is_finite(), "tensor {name} contains non-finite value");
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimax_h3::{AUDIO_TAG, RowTimesteps, TEXT_TAG, VIDEO_TAG};

    fn plan_for(timesteps: &[f32], tags: &[u8]) -> Result<AdalnPlan> {
        let rows: Vec<RowTimesteps> = timesteps
            .iter()
            .map(|&t| RowTimesteps {
                timesteps: vec![t],
                endpoints: Vec::new(),
                indices: vec![0; tags.len()],
            })
            .collect();
        AdalnPlan::from_tags(tags, &rows)
    }

    fn simulate_prefetch(
        placements: &[BlockPlacement],
        prefetch_depth: usize,
    ) -> Result<(usize, usize)> {
        let steps = plan_prefetch_schedule(placements, prefetch_depth)?;
        assert_eq!(steps.len(), placements.len());
        let mut live: Vec<usize> = Vec::new();
        let mut peak = 0usize;
        let mut overlaps = 0usize;
        for step in &steps {
            if let Some(b) = step.load_before {
                assert_eq!(b, step.block);
                assert!(!live.contains(&b));
                live.push(b);
            }
            if placements[step.block] != BlockPlacement::HbmResident {
                assert!(
                    live.contains(&step.block),
                    "block {} missing from HBM",
                    step.block
                );
            }
            if let Some(nb) = step.prefetch_during {
                assert_eq!(nb, step.block + 1);
                assert!(!live.contains(&nb));
                live.push(nb);
                overlaps += 1;
            }
            peak = peak.max(live.len());
            if step.release_after {
                let pos = live
                    .iter()
                    .position(|&x| x == step.block)
                    .with_context(|| format!("release_after block {} not live", step.block))?;
                live.remove(pos);
            }
        }
        assert!(live.is_empty(), "leaked staging slots: {live:?}");
        assert!(
            peak <= prefetch_depth,
            "peak {peak} > prefetch_depth {prefetch_depth}"
        );
        Ok((peak, overlaps))
    }

    #[test]
    fn prefetch_schedule_bounds_staging_slots_across_tiers() -> Result<()> {
        // v5e1: 17 HBM + 33 PinnedHost, depth=2 -> all 33 streamed blocks overlapped (starting during block 16).
        let mut v5e1 = vec![BlockPlacement::HbmResident; 17];
        v5e1.extend(std::iter::repeat_n(BlockPlacement::PinnedHostStreamed, 33));
        let (peak, overlaps) = simulate_prefetch(&v5e1, 2)?;
        assert_eq!(peak, 2);
        assert_eq!(overlaps, 33);

        // 3-tier constrained host: 17 HBM + 20 PinnedHost + 13 DiskStreamed, depth=2.
        let mut three_tier = vec![BlockPlacement::HbmResident; 17];
        three_tier.extend(std::iter::repeat_n(BlockPlacement::PinnedHostStreamed, 20));
        three_tier.extend(std::iter::repeat_n(BlockPlacement::DiskStreamed, 13));
        let (peak3, overlaps3) = simulate_prefetch(&three_tier, 2)?;
        assert_eq!(peak3, 2);
        assert_eq!(overlaps3, 33);

        // All-streamed (0 HBM) with depth=2: block 0 loads synchronously, 49 blocks overlap.
        let all_disk = vec![BlockPlacement::DiskStreamed; 50];
        let (peak_d, overlaps_d) = simulate_prefetch(&all_disk, 2)?;
        assert_eq!(peak_d, 2);
        assert_eq!(overlaps_d, 49);

        // Single-slot depth=1: zero overlaps, peak=1.
        let (peak1, overlaps1) = simulate_prefetch(&three_tier, 1)?;
        assert_eq!(peak1, 1);
        assert_eq!(overlaps1, 0);

        // All-resident depth=0: peak=0, overlaps=0; depth=0 with streamed blocks errors.
        let all_hbm = vec![BlockPlacement::HbmResident; 50];
        let (peak0, overlaps0) = simulate_prefetch(&all_hbm, 0)?;
        assert_eq!(peak0, 0);
        assert_eq!(overlaps0, 0);
        assert!(plan_prefetch_schedule(&v5e1, 0).is_err());
        Ok(())
    }

    #[test]
    fn adaln_cache_derives_from_schedule_and_recomputes_on_key_change() -> Result<()> {
        let mut cache = TpuAdalnCache::default();
        let mut computes = 0usize;

        // 4-step schedule (not hard-coded 6), tags {image=0, text=1}.
        let p4 = plan_for(&[0.0, 0.3, 0.7, 0.9], &[TEXT_TAG, VIDEO_TAG])?;
        assert_eq!(p4.timesteps.len(), 4);
        assert_eq!(p4.pairs.len(), 8);
        assert!(cache.ensure_for_plan(&p4, |p| {
            computes += 1;
            Ok(vec![vec![bf16::from_f32(p.pairs.len() as f32)]])
        })?);
        assert_eq!(computes, 1);

        // Identical schedule and tags: cache hit, zero recomputation.
        assert!(!cache.ensure_for_plan(&p4, |_| {
            computes += 1;
            Ok(vec![])
        })?);
        assert_eq!(computes, 1);

        // Changed schedule length (6 steps): recompute trigger fires.
        let p6 = plan_for(&[0.0, 0.03, 0.11, 0.25, 0.47, 0.77], &[TEXT_TAG, VIDEO_TAG])?;
        assert!(cache.ensure_for_plan(&p6, |p| {
            computes += 1;
            Ok(vec![vec![bf16::from_f32(p.pairs.len() as f32)]])
        })?);
        assert_eq!(computes, 2);
        assert_eq!(cache.layer(0), Some(&[bf16::from_f32(12.0)][..]));

        // Same count (6 steps) with one altered timestep value: recompute trigger fires.
        let p6_alt = plan_for(&[0.0, 0.04, 0.11, 0.25, 0.47, 0.77], &[TEXT_TAG, VIDEO_TAG])?;
        assert!(cache.ensure_for_plan(&p6_alt, |_| {
            computes += 1;
            Ok(vec![vec![bf16::ONE]])
        })?);
        assert_eq!(computes, 3);

        // Same timesteps with added modality tag (audio=2): recompute trigger fires.
        let p6_audio = plan_for(
            &[0.0, 0.04, 0.11, 0.25, 0.47, 0.77],
            &[TEXT_TAG, VIDEO_TAG, AUDIO_TAG],
        )?;
        assert!(cache.ensure_for_plan(&p6_audio, |p| {
            computes += 1;
            Ok(vec![vec![bf16::from_f32(p.pairs.len() as f32)]])
        })?);
        assert_eq!(computes, 4);
        assert_eq!(cache.layer(0), Some(&[bf16::from_f32(18.0)][..]));
        Ok(())
    }
}
