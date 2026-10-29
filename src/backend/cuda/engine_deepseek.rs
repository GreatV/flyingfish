//! DeepSeek-V4.1 mHC wiring: the setup binding of the shared mHC runtime
//! (F32 fn storage, Fp32 update rounding, plain comb application, the
//! carried/attention-own pre-source policy) and the operator dispatch
//! (model.py Block; mHC arXiv 2512.24880), the layer-0 Mixer stage: latent
//! projections, the sliding-window K ring and the group output projection
//! (model.py Attention.forward), and the engine load path over the audited
//! manifest (TextOnly policy): dense resident weights for every backbone
//! layer, the three runtimes and a memory-admitted routed-expert prefix.
use super::*;
use crate::backend::cuda::contracts::{RowDomain, SeqRef, Tensor, TensorMut, dense_layout, view};
use crate::backend::cuda::hc::{HcBinding, HcRuntime, HcSublayerWeights};
use crate::backend::cuda::latent_window_attention::{LatentWindowAttention, LwaInputs};
use crate::backend::cuda::mhc::{HcCombOrientation, HcFnDtype, HcParams, HcRounding};
use crate::backend::cuda::moe::combine::{Accumulator, CombineOps, MoeCombineAccum};
use crate::backend::cuda::moe::route::RouteOps;
use crate::backend::cuda::moe::w4a8;
use crate::backend::cuda::moe::w8a8;
use crate::backend::cuda::ops::{CommonOpsRef, RmsMode, RmsOptions};
use crate::backend::cuda::state::LatentRing;
use crate::model::hc_pre_policy;
use crate::model_spec::{DeepseekV41Config, ModelSpec, RouterScore};
use cudarc::driver::DevicePtr;

#[path = "engine_deepseek_dsa.rs"]
mod dsa;
#[cfg(test)]
#[path = "deepseek_layer_certificate.rs"]
mod layer_certificate;

struct EngramWeights {
    projection: Fp8Projection,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
}

struct EngramLayer {
    id: usize,
    records: CudaSlice<u8>,
    weights: Option<EngramWeights>,
    plans: Vec<(
        super::super::quantize::Fp8LookupProjection,
        super::super::gated_residual::GatedResidualPlan,
    )>,
}

pub(super) struct DeepseekEngram {
    pub(super) service: crate::engram::EngramService,
    layers: Vec<EngramLayer>,
    codes: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    projection: CudaSlice<bf16>,
    output: CudaSlice<bf16>,
    gates: CudaSlice<f32>,
    mask: CudaSlice<u8>,
    rows: usize,
    width: usize,
    hidden: usize,
    copies: usize,
    pub(super) prepared: Option<(usize, usize)>,
}

impl DeepseekEngram {
    fn load_weights(&mut self, loader: &mut super::super::weights::Loader) -> Result<()> {
        let n = (self.copies + 1) * self.hidden;
        let k = self.width * crate::engram::ENGRAM_HEAD_DIM;
        for layer in &mut self.layers {
            let p = format!("layers.{}.engram", layer.id);
            layer.weights = Some(EngramWeights {
                projection: Fp8Projection {
                    codes: loader.load_bytes(
                        &format!("{p}.wkv.weight"),
                        &[n, k],
                        safetensors::Dtype::F8_E4M3,
                    )?,
                    scale: loader.load_bytes(
                        &format!("{p}.wkv.scale"),
                        &[n / 32, k / 32],
                        safetensors::Dtype::F8_E8M0,
                    )?,
                },
                q: loader.load(&[(format!("{p}.q_weight"), vec![self.copies, self.hidden])])?,
                k: loader.load(&[(format!("{p}.k_weight"), vec![self.copies, self.hidden])])?,
            });
        }
        Ok(())
    }

    fn new(
        device: &Device,
        spec: &ModelSpec,
        manifest: &crate::checkpoint::TensorManifest,
        model: &Path,
        settings: &Settings,
        capacity: usize,
    ) -> Result<Self> {
        use crate::engram::{ENGRAM_NATIVE_ROW_BYTES, EngramHashSetup, EngramService};
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("Engram model geometry missing")?;
        let runtime = settings
            .runtime_dir
            .as_ref()
            .context("DeepSeek Engram requires runtime_dir with engram-hash-v1.json")?;
        let path = runtime.join("engram-hash-v1.json");
        let setup: EngramHashSetup = serde_json::from_slice(
            &std::fs::read(&path)
                .with_context(|| format!("read Engram hash record {}", path.display()))?,
        )
        .context("decode Engram hash record")?;
        setup.validate_model(model)?;
        setup.validate_samples(model)?;
        ensure!(
            setup.layer_ids == ds.engram_layer_ids
                && setup.layer_rows()? == ds.engram_num_embeddings
                && setup.ngram() == ds.engram_max_ngram_size as usize
                && setup.heads() == ds.engram_n_heads as usize
                && ds.engram_head_dim as usize == crate::engram::ENGRAM_HEAD_DIM,
            "Engram hash record differs from model geometry"
        );
        let rows = settings.spec_budget.rows().min(capacity);
        let width = setup.hash_width();
        let available = super::super::calibrate::host_memory(Path::new("/"))? as usize;
        let budget = (available / 64).min(256 << 20);
        let request_bytes = rows * width * setup.layer_ids.len() * ENGRAM_NATIVE_ROW_BYTES * 3;
        let cache_rows = budget
            .checked_sub(request_bytes)
            .context("Engram host budget cannot hold prepared records")?
            / 512;
        ensure!(cache_rows > 0, "Engram LFU host budget admits zero rows");
        let layer_ids = setup.layer_ids.clone();
        let service = EngramService::open(model, manifest, setup, cache_rows, capacity)?;
        eprintln!(
            "{}",
            serde_json::json!({"deepseek_engram_setup": {
            "host_budget_bytes":budget,"cache_rows":cache_rows,"cache_charge_per_row":512,
            "request_bytes":request_bytes,"native_row_bytes":ENGRAM_NATIVE_ROW_BYTES,
            "lookup_columns":width,"layers":layer_ids,"io":"O_DIRECT"}})
        );
        let s = &device.stream;
        let hidden = spec.hidden as usize;
        let copies = ds.hc_mult as usize;
        Ok(Self {
            service,
            layers: layer_ids
                .iter()
                .map(|&id| {
                    Ok(EngramLayer {
                        id: id as usize,
                        records: s.alloc_zeros(rows * width * ENGRAM_NATIVE_ROW_BYTES)?,
                        weights: None,
                        plans: Vec::new(),
                    })
                })
                .collect::<Result<_>>()?,
            codes: s.alloc_zeros(rows * width * crate::engram::ENGRAM_HEAD_DIM)?,
            scales: s.alloc_zeros(rows * width * crate::engram::ENGRAM_SCALES)?,
            projection: s.alloc_zeros(rows * (copies + 1) * hidden)?,
            output: s.alloc_zeros(rows * copies * hidden)?,
            gates: s.alloc_zeros(rows * copies)?,
            mask: s.alloc_zeros(rows)?,
            rows,
            width,
            hidden,
            copies,
            prepared: None,
        })
    }

    fn bind(&mut self, device: &Device, hc: &HcRuntime, seq: SeqRef, epsilon: f32) -> Result<()> {
        use super::super::{
            gated_residual::{GatedResidualPlan, GatedResidualViews},
            quantize::{Fp8LookupProjection, Fp8LookupViews},
        };
        let s = &device.stream;
        let k = self.width * crate::engram::ENGRAM_HEAD_DIM;
        let n = (self.copies + 1) * self.hidden;
        for layer in &mut self.layers {
            let w = layer
                .weights
                .as_ref()
                .context("Engram projection weights missing")?;
            for rows in 1..=self.rows {
                let r = rows as u32;
                let lookup = Fp8LookupProjection::new(
                    &device.ctx,
                    Fp8LookupViews {
                        records: view(
                            &layer.records,
                            s,
                            0,
                            [r, self.width as u32, 264],
                            [(self.width * 264) as u64, 264, 1],
                        )?,
                        codes: view(&self.codes, s, 0, [r, k as u32], [k as u64, 1])?,
                        scales: view(
                            &self.scales,
                            s,
                            0,
                            [r, (k / 32) as u32],
                            [(k / 32) as u64, 1],
                        )?,
                        sequence: seq,
                    },
                    view(
                        &w.projection.codes,
                        s,
                        0,
                        [n as u32, k as u32],
                        [k as u64, 1],
                    )?,
                    view(
                        &w.projection.scale,
                        s,
                        0,
                        [(n / 32) as u32, (k / 32) as u32],
                        [(k / 32) as u64, 1],
                    )?,
                    view(&self.projection, s, 0, [r, n as u32], [n as u64, 1])?,
                )?;
                let shape = [r, self.copies as u32, self.hidden as u32];
                let stride = [(self.copies * self.hidden) as u64, self.hidden as u64, 1];
                let gate = GatedResidualPlan::new(
                    &device.ctx,
                    GatedResidualViews {
                        input: view(&hc.stream, s, 0, shape, stride)?,
                        projection: view(&self.projection, s, 0, [r, n as u32], [n as u64, 1])?,
                        q_weight: view(
                            &w.q,
                            s,
                            0,
                            [self.copies as u32, self.hidden as u32],
                            [self.hidden as u64, 1],
                        )?,
                        k_weight: view(
                            &w.k,
                            s,
                            0,
                            [self.copies as u32, self.hidden as u32],
                            [self.hidden as u64, 1],
                        )?,
                        mask: view(&self.mask, s, 0, [r], [1])?,
                        output: view(&self.output, s, 0, shape, stride)?,
                        gates: view(
                            &self.gates,
                            s,
                            0,
                            [r, self.copies as u32],
                            [self.copies as u64, 1],
                        )?,
                        sequence: seq,
                    },
                    epsilon,
                    1e-6,
                )?;
                layer.plans.push((lookup, gate));
            }
        }
        Ok(())
    }

    fn prepare(
        &mut self,
        stream: &Arc<CudaStream>,
        ids: &[u32],
        position: usize,
        text_mask: Option<&[bool]>,
    ) -> Result<()> {
        self.prepared = None;
        ensure!(
            !ids.is_empty() && ids.len() <= self.rows,
            "Engram request exceeds bound rows"
        );
        let (ticket, _) = self
            .service
            .prepare(ids, u32::try_from(position)?, text_mask)?;
        for layer in &mut self.layers {
            let native = ticket.take(layer.id as u32)?;
            stream.memcpy_htod(
                native.records,
                &mut layer.records.slice_mut(..native.records.len()),
            )?;
        }
        let mask: Vec<u8> = text_mask.map_or_else(
            || vec![1; ids.len()],
            |v| v.iter().map(|&v| u8::from(v)).collect(),
        );
        stream.memcpy_htod(&mask, &mut self.mask.slice_mut(..ids.len()))?;
        self.prepared = Some((position, ids.len()));
        Ok(())
    }

    pub(super) fn run(
        &self,
        stream: &Arc<CudaStream>,
        layer_id: usize,
        rows: usize,
        hc: &mut HcRuntime,
    ) -> Result<()> {
        let Some(layer) = self.layers.iter().find(|v| v.id == layer_id) else {
            return Ok(());
        };
        let (lookup, gate) = layer
            .plans
            .get(rows.checked_sub(1).context("Engram has no rows")?)
            .context("Engram row count is not bound")?;
        lookup.run(stream)?;
        gate.run(stream)?;
        let elements = rows * self.copies * self.hidden;
        stream.memcpy_dtod(
            &self.output.slice(..elements),
            &mut hc.stream.slice_mut(..elements),
        )?;
        Ok(())
    }
}
pub fn deepseek_hc_binding(spec: &crate::model_spec::ModelSpec) -> Result<HcBinding> {
    let ds = spec
        .deepseek_v41
        .as_ref()
        .context("DeepSeek-V4.1 geometry missing")?;
    Ok(HcBinding {
        params: HcParams {
            mult: ds.hc_mult,
            hidden: spec.hidden,
            iters: ds.hc_sinkhorn_iters,
            eps: ds.hc_eps,
            norm_eps: spec.rms_eps,
        },
        policy: hc_pre_policy(spec.architecture)?,
        rounding: HcRounding::Fp32,
        // model.py Block.hc_post: out[m] = post[m]*x + sum_n comb[n,m]*residual[n]
        // (torch.sum over dim 2 after the [b,s,n,m,d] broadcast).
        orientation: HcCombOrientation::Transposed,
        fn_dtype: HcFnDtype::F32,
    })
}

/// Cross-derived mixer geometry of one DeepSeek-V4.1 model, checked against
/// the config once at setup (stored layouts, model definition Attention).
struct MixerGeometry {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
    rotary: usize,
    q_lora: usize,
    o_rank: usize,
    o_groups: usize,
    /// Per-group wo_a input width: `heads * head_dim / o_groups`.
    o_in: usize,
}

impl MixerGeometry {
    fn new(spec: &ModelSpec) -> Result<Self> {
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let geometry = Self {
            hidden: spec.hidden as usize,
            heads: spec.q_heads as usize,
            head_dim: spec.head_dim as usize,
            window: ds.window_size as usize,
            rotary: spec.rotary_dim as usize,
            q_lora: ds.q_lora_rank as usize,
            o_rank: ds.o_lora_rank as usize,
            o_groups: ds.o_groups as usize,
            o_in: (spec.q_heads * spec.head_dim / ds.o_groups) as usize,
        };
        ensure!(
            spec.q_width == spec.q_heads * spec.head_dim,
            "DeepSeek-V4.1 q_width {} != heads {} x head_dim {}",
            spec.q_width,
            spec.q_heads,
            spec.head_dim
        );
        ensure!(
            spec.q_width.is_multiple_of(ds.o_groups),
            "DeepSeek-V4.1 q_width {} is not a multiple of o_groups {}",
            spec.q_width,
            ds.o_groups
        );
        ensure!(
            spec.rotary_dim == ds.qk_rope_head_dim,
            "DeepSeek-V4.1 rotary_dim {} != qk_rope_head_dim {}",
            spec.rotary_dim,
            ds.qk_rope_head_dim
        );
        ensure!(
            geometry.rotary >= 2
                && geometry.rotary.is_multiple_of(2)
                && geometry.rotary <= geometry.head_dim,
            "DeepSeek-V4.1 rotary tail {} is odd, below 2, or exceeds head_dim {}",
            geometry.rotary,
            geometry.head_dim
        );
        ensure!(
            geometry.head_dim.is_multiple_of(32),
            "DeepSeek-V4.1 head_dim {} must be a multiple of 32 for the ring quantization",
            geometry.head_dim
        );
        ensure!(
            geometry.o_in.is_multiple_of(32),
            "DeepSeek-V4.1 group width {} must be a multiple of 32 for the output quantization",
            geometry.o_in
        );
        ensure!(
            geometry.o_rank.is_multiple_of(32),
            "DeepSeek-V4.1 o_lora_rank {} must be a multiple of 32 so one 32x32 weight scale block never spans two groups",
            geometry.o_rank
        );
        ensure!(
            geometry.window >= 1,
            "DeepSeek-V4.1 window {} is below 1",
            geometry.window
        );
        Ok(geometry)
    }
}

/// Inverse rotary frequencies of one layer kind, derived from the config
/// through the spec (model definition precompute_freqs_cis): ratio-0 layers
/// use plain rope_theta, the compressed kinds use compress_rope_theta with the
/// YaRN ramp over the original context.
fn rotary_inv_freq(spec: &ModelSpec, ds: &DeepseekV41Config, ratio: u8) -> Result<Vec<f32>> {
    let dim = spec.rotary_dim;
    let half = dim / 2;
    ensure!(half >= 1, "DeepSeek-V4.1 rotary_dim {dim} is below 2");
    let base = if ratio != 0 {
        ds.compress_rope_theta
    } else {
        spec.rope_theta
    };
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();
    if ratio != 0 {
        let original = f64::from(ds.original_max_position);
        let corrected = |rotations: f64| {
            f64::from(dim) * (original / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * f64::from(base).ln())
        };
        let low = corrected(f64::from(ds.beta_fast)).floor().max(0.0) as usize;
        let high = (corrected(f64::from(ds.beta_slow)).ceil() as usize).min(dim as usize - 1);
        let span = (high as f32 - low as f32).max(1e-3);
        for (i, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / ds.rope_factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    Ok(freqs)
}

/// Owned device weight bytes of one layer's mixer (the audited manifest path
/// uploads them byte-exact). Stored layouts: wq_a `[q_lora, hidden]` bf16,
/// q_norm `[q_lora]` bf16, wq_b `[heads*head_dim, q_lora]` bf16, wkv
/// `[head_dim, hidden]` bf16, kv_norm `[head_dim]` bf16, attn_sink `[heads]`
/// f32, wo_a `[o_groups*o_rank, heads*head_dim/o_groups]` E4M3 codes, wo_a
/// scale `[(o_groups*o_rank)/32, (heads*head_dim/o_groups)/32]` E8M0, wo_b
/// `[hidden, o_groups*o_rank]` bf16.
/// One published FP8 projection: `[n, k]` E4M3 codes with
/// `[ceil(n/32), ceil(k/32)]` E8M0 scales, consumed natively.
pub struct Fp8Projection {
    pub codes: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
}

pub struct MixerWeights {
    pub wq_a: Fp8Projection,
    pub q_norm: CudaSlice<u8>,
    pub wq_b: Fp8Projection,
    pub wkv: Fp8Projection,
    pub kv_norm: CudaSlice<u8>,
    pub attn_sink: CudaSlice<u8>,
    pub wo_a: CudaSlice<u8>,
    pub wo_a_scale: CudaSlice<u8>,
    pub wo_b: Fp8Projection,
}

/// DeepSeek-V4.1 Mixer runtime (model definition Attention.forward, decode
/// path): wq_a/q_norm/wq_b and wkv/kv_norm, the adjacent-pair rotary on the
/// tail with per-layer-kind parameters from the spec, the ring write, the
/// latent window attention over the visible slots oldest first, and the
/// per-32 quantized attention output through the block-diagonal wo_a and wo_b.
/// The per-layer K ring lives in StateStore (`LatentRing`); the mixer holds
/// weights only.
pub struct DeepseekMixer {
    dsa: Option<dsa::Runtime>,
    compress_ratios: Vec<u8>,
    ops: LatentWindowAttention,
    common: CommonOpsRef,
    geometry: MixerGeometry,
    layout: crate::fp8::Fp8ScaleLayout,
    weights: Vec<Option<MixerWeights>>,
    inv_freq: Vec<CudaSlice<f32>>,
    eps: f32,
    slots: CudaSlice<i32>,
    #[cfg(test)]
    query_stores: std::cell::RefCell<std::collections::BTreeMap<String, Vec<bf16>>>,
    rotary_fwd: CudaSlice<f32>,
    rotary_inv: CudaSlice<f32>,
    q_lora: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    o_codes: CudaSlice<u8>,
    o_scales: CudaSlice<u8>,
    groups: CudaSlice<bf16>,
    /// Prepared activation codes/scales per projection K (hidden, q_lora,
    /// o_width): `[rows, k]` codes and `[rows, k/32]` E8M0 scales.
    act_hidden_codes: CudaSlice<u8>,
    act_hidden_scales: CudaSlice<u8>,
    act_lora_codes: CudaSlice<u8>,
    act_lora_scales: CudaSlice<u8>,
    act_o_codes: CudaSlice<u8>,
    act_o_scales: CudaSlice<u8>,
    rows: usize,
}

#[cfg(test)]
impl DeepseekMixer {
    /// Host copy of the attention output before its quantization and of the
    /// post-wo_a groups, both for the first `rows` rows.
    pub(crate) fn attn_output_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        rows: usize,
    ) -> Result<(Vec<bf16>, Vec<bf16>)> {
        let q_width = self.geometry.heads * self.geometry.head_dim;
        let o_width = self.geometry.o_groups * self.geometry.o_rank;
        Ok((
            s.clone_dtoh(&self.o.slice(..rows * q_width))?,
            s.clone_dtoh(&self.groups.slice(..rows * o_width))?,
        ))
    }
}

#[cfg(test)]
impl DeepseekMixer {
    /// Host copy of the hidden activation codes and E8M0 scales the projections consume.
    pub(crate) fn activation_codes_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        rows: usize,
        width: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        Ok((
            s.clone_dtoh(&self.act_hidden_codes.slice(..rows * width))?,
            s.clone_dtoh(&self.act_hidden_scales.slice(..rows * (width / 32)))?,
        ))
    }

    /// Host copy of the wo_b input codes and E8M0 scales of the first row, and
    /// of `wo_b` weight row `dim` with its per-32 block scales.
    pub(crate) fn wo_b_row_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        layer: usize,
        dim: usize,
    ) -> Result<WoBRow> {
        let weights = self
            .weights
            .get(layer)
            .with_context(|| {
                format!("DeepSeek-V4.1 mixer layer {layer} is outside the layer plan")
            })?
            .as_ref()
            .context("DeepSeek-V4.1 mixer layer has no bound weights")?;
        let width = self.geometry.o_groups * self.geometry.o_rank;
        ensure!(
            dim < self.geometry.hidden,
            "wo_b row {dim} is outside the {} hidden rows",
            self.geometry.hidden
        );
        let blocks = width / 32;
        Ok(WoBRow {
            act_codes: s.clone_dtoh(&self.act_o_codes.slice(..width))?,
            act_scales: s.clone_dtoh(&self.act_o_scales.slice(..blocks))?,
            w_codes: s.clone_dtoh(&weights.wo_b.codes.slice(dim * width..(dim + 1) * width))?,
            w_scales: s.clone_dtoh(
                &weights
                    .wo_b
                    .scale
                    .slice((dim / 32) * blocks..(dim / 32 + 1) * blocks),
            )?,
        })
    }

    /// Host copy of the visible ring slot ids the attention consumed.
    pub(crate) fn visible_slots_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        count: usize,
    ) -> Result<Vec<i32>> {
        Ok(s.clone_dtoh(&self.slots.slice(..count))?)
    }

    pub(crate) fn replay_recorded_window(&self, input: RecordedWindowReplay<'_>) -> Result<()> {
        let RecordedWindowReplay {
            s,
            blas,
            ring,
            layer,
            position,
            q,
            codes,
            scales,
            out,
        } = input;
        let g = &self.geometry;
        let q_width = g.heads * g.head_dim;
        let o_width = g.o_groups * g.o_rank;
        let weights = self
            .weights
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 mixer layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 mixer layer {layer} has no bound weights"))?;
        ensure!(
            q.len() >= q_width
                && out.len() >= g.hidden
                && codes.len() == g.window * g.head_dim
                && scales.len() == g.window * (g.head_dim / 32),
            "recorded window replay operand geometry"
        );
        s.memcpy_htod(codes, &mut ring.codes)?;
        s.memcpy_htod(scales, &mut ring.scales)?;
        let visible = (position + 1).min(g.window);
        self.ops
            .window_slots(
                s,
                &mut TensorMut(
                    view(&self.slots, s, 0, [visible as u32], [1])
                        .context("recorded window slot view")?,
                ),
                position as i32,
            )
            .context("recorded window slots")?;
        let sink = weight_view::<f32, 1>(&weights.attn_sink, s, [g.heads as u32], "attn_sink")?;
        self.ops
            .run(
                s,
                LwaInputs {
                    q: view(
                        q,
                        s,
                        0,
                        [g.heads as u32, g.head_dim as u32],
                        [g.head_dim as u64, 1],
                    )
                    .context("recorded window q view")?,
                    k_codes: view(
                        &ring.codes,
                        s,
                        0,
                        [g.window as u32, g.head_dim as u32],
                        [g.head_dim as u64, 1],
                    )
                    .context("recorded window code view")?,
                    k_scales: view(
                        &ring.scales,
                        s,
                        0,
                        [g.window as u32, (g.head_dim / 32) as u32],
                        [(g.head_dim / 32) as u64, 1],
                    )
                    .context("recorded window scale view")?,
                    slot_ids: view(&self.slots, s, 0, [visible as u32], [1])
                        .context("recorded window slot input view")?,
                    sink,
                    inv_rotary: view(&self.rotary_inv, s, 0, [g.rotary as u32], [1])
                        .context("recorded window rotary view")?,
                },
                &mut TensorMut(
                    view(
                        &self.o,
                        s,
                        0,
                        [g.heads as u32, g.head_dim as u32],
                        [g.head_dim as u64, 1],
                    )
                    .context("recorded window output view")?,
                ),
                0,
            )
            .context("recorded window attention")?;
        self.ops
            .act_quant(
                s,
                view(&self.o, s, 0, [1, q_width as u32], [q_width as u64, 1])
                    .context("recorded replay o input view")?,
                &mut TensorMut(
                    view(
                        &self.o_codes,
                        s,
                        0,
                        [1, q_width as u32],
                        [q_width as u64, 1],
                    )
                    .context("recorded replay o code view")?,
                ),
                &mut TensorMut(
                    view(
                        &self.o_scales,
                        s,
                        0,
                        [1, (q_width / 32) as u32],
                        [(q_width / 32) as u64, 1],
                    )
                    .context("recorded replay o scale view")?,
                ),
            )
            .context("recorded replay o quantization")?;
        self.ops
            .group_project(
                s,
                view(
                    &self.o_codes,
                    s,
                    0,
                    [1, g.o_groups as u32, g.o_in as u32],
                    [q_width as u64, g.o_in as u64, 1],
                )
                .context("recorded replay wo_a code view")?,
                view(
                    &self.o_scales,
                    s,
                    0,
                    [1, g.o_groups as u32, (g.o_in / 32) as u32],
                    [(q_width / 32) as u64, (g.o_in / 32) as u64, 1],
                )
                .context("recorded replay wo_a scale view")?,
                weight_view::<u8, 2>(&weights.wo_a, s, [o_width as u32, g.o_in as u32], "wo_a")?,
                weight_view::<u8, 2>(
                    &weights.wo_a_scale,
                    s,
                    [(o_width / 32) as u32, (g.o_in / 32) as u32],
                    "wo_a_scale",
                )?,
                &mut TensorMut(
                    view(&self.groups, s, 0, [1, o_width as u32], [o_width as u64, 1])
                        .context("recorded replay group output view")?,
                ),
            )
            .context("recorded replay wo_a projection")?;
        self.ops
            .act_quant(
                s,
                view(&self.groups, s, 0, [1, o_width as u32], [o_width as u64, 1])
                    .context("recorded replay group input view")?,
                &mut TensorMut(
                    view(
                        &self.act_o_codes,
                        s,
                        0,
                        [1, o_width as u32],
                        [o_width as u64, 1],
                    )
                    .context("recorded replay wo_b code view")?,
                ),
                &mut TensorMut(
                    view(
                        &self.act_o_scales,
                        s,
                        0,
                        [1, (o_width / 32) as u32],
                        [(o_width / 32) as u64, 1],
                    )
                    .context("recorded replay wo_b scale view")?,
                ),
            )
            .context("recorded replay wo_b quantization")?;
        let fp8 = blas.fp8_ops()?;
        let weight_codes = weight_view::<u8, 2>(
            &weights.wo_b.codes,
            s,
            [g.hidden as u32, o_width as u32],
            "wo_b codes",
        )?;
        let weight_scales = weight_view::<u8, 2>(
            &weights.wo_b.scale,
            s,
            [g.hidden.div_ceil(32) as u32, o_width.div_ceil(32) as u32],
            "wo_b scales",
        )?;
        let act_codes = view(
            &self.act_o_codes,
            s,
            0,
            [1, o_width as u32],
            [o_width as u64, 1],
        )
        .context("recorded replay wo_b activation code view")?;
        let act_scales = view(
            &self.act_o_scales,
            s,
            0,
            [1, (o_width / 32) as u32],
            [(o_width / 32) as u64, 1],
        )
        .context("recorded replay wo_b activation scale view")?;
        let output = TensorMut(
            view(out, s, 0, [1, g.hidden as u32], [g.hidden as u64, 1])
                .context("recorded replay output view")?,
        );
        let plan = fp8.bind_block(
            crate::backend::cuda::linear::LinearWeightRef::Fp8 {
                codes: weight_codes,
                scale: crate::backend::cuda::linear::Fp8ScaleRef::E8m0(weight_scales),
                layout: self.layout,
            },
            act_codes,
            crate::backend::cuda::linear::Fp8ScaleRef::E8m0(act_scales),
            &output,
        )?;
        plan.run(s)
    }

    /// Host copy of the query latent after its norm, before the wq_b quantization.
    pub(crate) fn query_latent_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        rows: usize,
    ) -> Result<Vec<bf16>> {
        Ok(s.clone_dtoh(&self.q_lora.slice(..rows * self.geometry.q_lora))?)
    }

    /// Host copy of the post-RoPE query rows and the post-RoPE K row of one step.
    pub(crate) fn query_key_host(
        &self,
        s: &std::sync::Arc<cudarc::driver::CudaStream>,
        qkv: &CudaSlice<bf16>,
        rows: usize,
    ) -> Result<(Vec<bf16>, Vec<bf16>)> {
        let q_width = self.geometry.heads * self.geometry.head_dim;
        let k0 = rows * q_width;
        Ok((
            s.clone_dtoh(&qkv.slice(..k0))?,
            s.clone_dtoh(&qkv.slice(k0..k0 + self.geometry.head_dim))?,
        ))
    }
}

/// One mixer call over the engine work buffers.
pub(crate) struct MixerStep<'a> {
    pub(crate) position: usize,
    pub(crate) rows: usize,
    pub(crate) x: &'a CudaSlice<bf16>,
    pub(crate) qkv: &'a mut CudaSlice<bf16>,
    pub(crate) out: &'a mut CudaSlice<bf16>,
    pub(crate) seq: SeqRef,
}

/// Dense view over owned mixer weight bytes.
fn weight_view<T: Copy, const R: usize>(
    bytes: &CudaSlice<u8>,
    s: &Arc<CudaStream>,
    shape: [u32; R],
    name: &str,
) -> Result<Tensor<T, R>> {
    let (stride, count) = dense_layout(shape)?;
    let span = count
        .checked_mul(std::mem::size_of::<T>() as u64)
        .context("mixer weight byte count overflow")?;
    ensure!(
        u64::try_from(bytes.len())? == span,
        "DeepSeek-V4.1 mixer {name} allocation holds {} bytes; shape {shape:?} needs {span}",
        bytes.len()
    );
    let (ptr, _) = bytes.device_ptr(s);
    unsafe { Tensor::new(ptr, shape, stride, span) }
}

impl DeepseekMixer {
    pub(super) fn check_decode_ready(&self) -> Result<()> {
        ensure!(
            self.dsa.is_some(),
            "DeepSeek-V4.1 decode native DSA runtime is not bound"
        );
        Ok(())
    }

    /// Allocates the runtime for `layers` layers and `rows` decode rows; the
    /// per-layer weights are bound separately.
    pub fn new(
        ctx: &Arc<CudaContext>,
        s: &Arc<CudaStream>,
        spec: &ModelSpec,
        layers: usize,
        rows: usize,
    ) -> Result<Self> {
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let geometry = MixerGeometry::new(spec)?;
        ensure!(
            layers > 0 && layers <= ds.compress_ratios.len(),
            "DeepSeek-V4.1 mixer layer count {layers} is zero or exceeds the compress_ratios extent {}",
            ds.compress_ratios.len()
        );
        ensure!(
            rows > 0,
            "DeepSeek-V4.1 mixer needs at least one row (workspace capacity)"
        );
        let mut inv_freq = Vec::with_capacity(layers);
        for layer in 0..layers {
            let ratio = ds.compress_ratios[layer];
            let freqs = rotary_inv_freq(spec, ds, ratio)?;
            inv_freq.push(s.clone_htod(&freqs)?);
        }
        let q_width = geometry.heads * geometry.head_dim;
        let window = geometry.window;
        let rotary = geometry.rotary;
        let q_lora = geometry.q_lora;
        let o_groups = geometry.o_groups;
        let o_rank = geometry.o_rank;
        let layout = spec
            .linear_format
            .fp8_layout()
            .context("DeepSeek-V4.1 linear format has no FP8 block layout")?;
        ensure!(
            matches!(
                &layout,
                crate::fp8::Fp8ScaleLayout::Block(
                    block
                ) if block.dtype() == safetensors::Dtype::F8_E8M0 && block.block() == [32, 32]
            ),
            "DeepSeek-V4.1 projections require the published [32,32] E8M0 block layout, got {layout:?}"
        );
        let hidden = geometry.hidden;
        let o_width = o_groups * o_rank;
        Ok(Self {
            ops: LatentWindowAttention::new(ctx)?,
            dsa: None,
            compress_ratios: ds.compress_ratios[..layers].to_vec(),
            common: CommonOpsRef::new(ctx)?,
            geometry,
            layout,
            weights: (0..layers).map(|_| None).collect(),
            inv_freq,
            eps: spec.rms_eps,
            slots: s.alloc_zeros(window)?,
            #[cfg(test)]
            query_stores: std::cell::RefCell::default(),
            rotary_fwd: s.alloc_zeros(rows * rotary)?,
            rotary_inv: s.alloc_zeros(rows * rotary)?,
            q_lora: s.alloc_zeros(rows * q_lora)?,
            o: s.alloc_zeros(rows * q_width)?,
            o_codes: s.alloc_zeros(rows * q_width)?,
            o_scales: s.alloc_zeros(rows * (q_width / 32))?,
            groups: s.alloc_zeros(rows * o_groups * o_rank)?,
            act_hidden_codes: s.alloc_zeros(rows * hidden)?,
            act_hidden_scales: s.alloc_zeros(rows * (hidden / 32))?,
            act_lora_codes: s.alloc_zeros(rows * q_lora)?,
            act_lora_scales: s.alloc_zeros(rows * (q_lora / 32))?,
            act_o_codes: s.alloc_zeros(rows * o_width)?,
            act_o_scales: s.alloc_zeros(rows * (o_width / 32))?,
            rows,
        })
    }

    /// Binds the weight bytes of one layer.
    pub fn bind_layer(&mut self, layer: usize, weights: MixerWeights) -> Result<()> {
        ensure!(
            layer < self.weights.len(),
            "DeepSeek-V4.1 mixer layer {layer} is outside the bound {} layers",
            self.weights.len()
        );
        let g = &self.geometry;
        let q_width = g.heads * g.head_dim;
        for (name, bytes, expected) in [
            ("q_norm", &weights.q_norm, 2 * g.q_lora),
            ("kv_norm", &weights.kv_norm, 2 * g.head_dim),
            ("attn_sink", &weights.attn_sink, 4 * g.heads),
            ("wo_a", &weights.wo_a, g.o_groups * g.o_rank * g.o_in),
            (
                "wo_a_scale",
                &weights.wo_a_scale,
                g.o_groups * g.o_rank / 32 * g.o_in / 32,
            ),
        ] {
            ensure!(
                bytes.len() == expected,
                "DeepSeek-V4.1 mixer {name} holds {} bytes; the bound geometry needs {expected}",
                bytes.len()
            );
        }
        for (name, projection, n, k) in [
            ("wq_a", &weights.wq_a, g.q_lora, g.hidden),
            ("wq_b", &weights.wq_b, q_width, g.q_lora),
            ("wkv", &weights.wkv, g.head_dim, g.hidden),
            ("wo_b", &weights.wo_b, g.hidden, g.o_groups * g.o_rank),
        ] {
            ensure!(
                projection.codes.len() == n * k
                    && projection.scale.len() == n.div_ceil(32) * k.div_ceil(32),
                "DeepSeek-V4.1 mixer {name} holds {}+{} bytes; the bound geometry needs {}+{}",
                projection.codes.len(),
                projection.scale.len(),
                n * k,
                n.div_ceil(32) * k.div_ceil(32)
            );
        }
        self.weights[layer] = Some(weights);
        Ok(())
    }

    /// Reallocates the per-row scratch for a larger workspace; the weights,
    /// StateStore rings and rotary tables are retained.
    pub fn grow(&mut self, s: &Arc<CudaStream>, rows: usize) -> Result<()> {
        ensure!(rows >= self.rows, "DeepSeek-V4.1 mixer rows cannot shrink");
        if rows == self.rows {
            return Ok(());
        }
        ensure!(
            self.dsa.is_none(),
            "DSA workspace growth requires rebinding setup plans"
        );
        let g = &self.geometry;
        let q_width = g.heads * g.head_dim;
        self.rotary_fwd = s.alloc_zeros(rows * g.rotary)?;
        self.rotary_inv = s.alloc_zeros(rows * g.rotary)?;
        self.q_lora = s.alloc_zeros(rows * g.q_lora)?;
        self.o = s.alloc_zeros(rows * q_width)?;
        self.o_codes = s.alloc_zeros(rows * q_width)?;
        self.o_scales = s.alloc_zeros(rows * (q_width / 32))?;
        self.groups = s.alloc_zeros(rows * g.o_groups * g.o_rank)?;
        self.act_hidden_codes = s.alloc_zeros(rows * g.hidden)?;
        self.act_hidden_scales = s.alloc_zeros(rows * (g.hidden / 32))?;
        self.act_lora_codes = s.alloc_zeros(rows * g.q_lora)?;
        self.act_lora_scales = s.alloc_zeros(rows * (g.q_lora / 32))?;
        let o_width = g.o_groups * g.o_rank;
        self.act_o_codes = s.alloc_zeros(rows * o_width)?;
        self.act_o_scales = s.alloc_zeros(rows * (o_width / 32))?;
        self.rows = rows;
        Ok(())
    }

    /// One decode step of `step.rows` rows at positions
    /// `step.position..step.position+step.rows` (model definition
    /// Attention.forward with start_pos = position): the projections run over
    /// the row batch, then each row writes its ring slot and attends over the
    /// visible window ending at its own position, oldest first.
    pub(crate) fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        rope: &Ops,
        ring: &LatentRing,
        step: MixerStep<'_>,
        layer: usize,
    ) -> Result<()> {
        let MixerStep {
            position,
            rows,
            x,
            qkv,
            out,
            seq,
        } = step;
        let g = &self.geometry;
        let q_width = g.heads * g.head_dim;
        let o_width = g.o_groups * g.o_rank;
        let weights = self
            .weights
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 mixer layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 mixer layer {layer} has no bound weights"))?;
        ensure!(
            ring.codes.len() == g.window * g.head_dim
                && ring.scales.len() == g.window * (g.head_dim / 32),
            "DeepSeek-V4.1 mixer layer {layer} ring holds {}+{} bytes; the geometry needs {}+{}",
            ring.codes.len(),
            ring.scales.len(),
            g.window * g.head_dim,
            g.window * (g.head_dim / 32)
        );
        ensure!(
            rows >= 1 && rows <= self.rows,
            "DeepSeek-V4.1 mixer rows {rows} are outside the runtime capacity {}",
            self.rows
        );
        ensure!(
            rows <= g.window,
            "DeepSeek-V4.1 mixer rows {rows} exceed the window {}",
            g.window
        );
        let end = position
            .checked_add(rows)
            .context("DeepSeek-V4.1 mixer position extent overflow")?;
        ensure!(
            end <= i32::MAX as usize,
            "DeepSeek-V4.1 mixer position {position} + {rows} exceeds the position extent"
        );
        ensure!(
            x.len() >= rows * g.hidden
                && qkv.len() >= rows * (q_width + g.head_dim)
                && out.len() >= rows * g.hidden,
            "DeepSeek-V4.1 mixer work buffers do not hold {rows} rows"
        );
        let fp8 = blas.fp8_ops()?;
        #[cfg(test)]
        let capture_query = |stage: &str, values: &CudaSlice<bf16>, width: usize| -> Result<()> {
            if layer == 4
                && position == 128
                && rows == 1
                && std::env::var_os("FF_DS41_LAYERS_FIXTURE").is_some()
            {
                self.query_stores
                    .borrow_mut()
                    .insert(stage.to_owned(), s.clone_dtoh(&values.slice(..width))?);
            }
            Ok(())
        };
        let proj = |name: &str,
                    weight: &Fp8Projection,
                    n: usize,
                    k: usize,
                    x_codes: &CudaSlice<u8>,
                    x_scales: &CudaSlice<u8>,
                    y: TensorMut<bf16, 2>|
         -> Result<()> {
            let weight_codes = weight_view::<u8, 2>(
                &weight.codes,
                s,
                [n as u32, k as u32],
                &format!("{name} codes"),
            )?;
            let weight_scales = weight_view::<u8, 2>(
                &weight.scale,
                s,
                [n.div_ceil(32) as u32, k.div_ceil(32) as u32],
                &format!("{name} scales"),
            )?;
            let act_codes = view(x_codes, s, 0, [rows as u32, k as u32], [k as u64, 1])?;
            let act_scales = view(
                x_scales,
                s,
                0,
                [rows as u32, k.div_ceil(32) as u32],
                [k.div_ceil(32) as u64, 1],
            )?;
            let plan = fp8.bind_block(
                crate::backend::cuda::linear::LinearWeightRef::Fp8 {
                    codes: weight_codes,
                    scale: crate::backend::cuda::linear::Fp8ScaleRef::E8m0(weight_scales),
                    layout: self.layout,
                },
                act_codes,
                crate::backend::cuda::linear::Fp8ScaleRef::E8m0(act_scales),
                &y,
            )?;
            plan.run(s)
        };
        let x_view = view(
            x,
            s,
            0,
            [rows as u32, g.hidden as u32],
            [g.hidden as u64, 1],
        )?;
        let lora = view(
            &self.q_lora,
            s,
            0,
            [rows as u32, g.q_lora as u32],
            [g.q_lora as u64, 1],
        )?;
        let q = view(
            qkv,
            s,
            0,
            [rows as u32, q_width as u32],
            [q_width as u64, 1],
        )?;
        let kv = view(
            qkv,
            s,
            rows * q_width,
            [rows as u32, g.head_dim as u32],
            [g.head_dim as u64, 1],
        )?;
        // wq_a -> q_norm (in place) -> wq_b into the q block of qkv; the
        // activation quantizes per-32 K before the native FP8 block gemv.
        self.ops.act_quant(
            s,
            x_view,
            &mut TensorMut(view(
                &self.act_hidden_codes,
                s,
                0,
                [rows as u32, g.hidden as u32],
                [g.hidden as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.act_hidden_scales,
                s,
                0,
                [rows as u32, (g.hidden / 32) as u32],
                [(g.hidden / 32) as u64, 1],
            )?),
        )?;
        proj(
            "wq_a",
            &weights.wq_a,
            g.q_lora,
            g.hidden,
            &self.act_hidden_codes,
            &self.act_hidden_scales,
            TensorMut(view(
                &self.q_lora,
                s,
                0,
                [rows as u32, g.q_lora as u32],
                [g.q_lora as u64, 1],
            )?),
        )?;
        let domain = RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: rows as u32,
        };
        #[cfg(test)]
        capture_query("q_a", &self.q_lora, g.q_lora)?;
        let options = RmsOptions {
            mode: RmsMode::Unrounded,
            eps: self.eps,
        };
        let lora3 = view(
            &self.q_lora,
            s,
            0,
            [rows as u32, 1, g.q_lora as u32],
            [g.q_lora as u64, g.q_lora as u64, 1],
        )?;
        self.common.norm.rms_with(
            s,
            domain,
            lora3,
            weight_view::<bf16, 1>(&weights.q_norm, s, [g.q_lora as u32], "q_norm")?,
            &mut TensorMut(lora3),
            options,
        )?;
        #[cfg(test)]
        capture_query("q_norm", &self.q_lora, g.q_lora)?;
        self.ops.act_quant(
            s,
            lora,
            &mut TensorMut(view(
                &self.act_lora_codes,
                s,
                0,
                [rows as u32, g.q_lora as u32],
                [g.q_lora as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.act_lora_scales,
                s,
                0,
                [rows as u32, (g.q_lora / 32) as u32],
                [(g.q_lora / 32) as u64, 1],
            )?),
        )?;
        proj(
            "wq_b",
            &weights.wq_b,
            q_width,
            g.q_lora,
            &self.act_lora_codes,
            &self.act_lora_scales,
            TensorMut(q),
        )?;
        #[cfg(test)]
        capture_query("q_b", qkv, q_width)?;
        // wkv -> kv_norm (in place) into the kv block of qkv; the hidden
        // activation codes are reused from the wq_a quantization.
        proj(
            "wkv",
            &weights.wkv,
            g.head_dim,
            g.hidden,
            &self.act_hidden_codes,
            &self.act_hidden_scales,
            TensorMut(kv),
        )?;
        let kv3 = view(
            qkv,
            s,
            rows * q_width,
            [rows as u32, 1, g.head_dim as u32],
            [g.head_dim as u64, g.head_dim as u64, 1],
        )?;
        self.common.norm.rms_with(
            s,
            domain,
            kv3,
            weight_view::<bf16, 1>(&weights.kv_norm, s, [g.head_dim as u32], "kv_norm")?,
            &mut TensorMut(kv3),
            options,
        )?;
        // Rotary rows for this step's positions, then the tail rotation on q
        // and kv (model definition apply_rotary_emb, adjacent pairs).
        let mut fwd = TensorMut(view(
            &self.rotary_fwd,
            s,
            0,
            [rows as u32, g.rotary as u32],
            [g.rotary as u64, 1],
        )?);
        let mut inv = TensorMut(view(
            &self.rotary_inv,
            s,
            0,
            [rows as u32, g.rotary as u32],
            [g.rotary as u64, 1],
        )?);
        rope.rope_pair_rows(
            s,
            view(&self.inv_freq[layer], s, 0, [(g.rotary / 2) as u32], [1])?,
            position as i32,
            rows,
            &mut fwd,
            &mut inv,
        )?;
        let rotary = fwd.0;
        rope.rope_pair_tail(
            s,
            &mut TensorMut(view(
                qkv,
                s,
                0,
                [rows as u32, g.heads as u32, g.head_dim as u32],
                [q_width as u64, g.head_dim as u64, 1],
            )?),
            rotary,
        )?;
        rope.rope_pair_tail(s, &mut TensorMut(kv3), rotary)?;
        let indexed = if self.compress_ratios[layer] != 0 {
            let runtime = self
                .dsa
                .as_ref()
                .context("DSA live-state runtime is unbound")?;
            runtime.run(self, s, blas, rope, layer, rows)?;
            Some(runtime)
        } else {
            None
        };
        // Per row: write the ring slot, list the visible slots oldest first,
        // attend. Later rows see earlier rows' writes and never the reverse.
        let mut codes = TensorMut(view(
            &ring.codes,
            s,
            0,
            [g.window as u32, g.head_dim as u32],
            [g.head_dim as u64, 1],
        )?);
        let mut scales = TensorMut(view(
            &ring.scales,
            s,
            0,
            [g.window as u32, (g.head_dim / 32) as u32],
            [(g.head_dim / 32) as u64, 1],
        )?);
        let sink = weight_view::<f32, 1>(&weights.attn_sink, s, [g.heads as u32], "attn_sink")?;
        let p_mode: i32 = 0;
        for row in 0..rows {
            let position = position + row;
            self.ops.ring_write(
                s,
                view(
                    qkv,
                    s,
                    rows * q_width + row * g.head_dim,
                    [1, g.head_dim as u32],
                    [g.head_dim as u64, 1],
                )?,
                position as i32,
                &mut codes,
                &mut scales,
            )?;
            if let Some(runtime) = indexed {
                runtime.attention(s, layer, rows, row)?;
                continue;
            }
            let visible = (position + 1).min(g.window);
            self.ops.window_slots(
                s,
                &mut TensorMut(view(&self.slots, s, 0, [visible as u32], [1])?),
                position as i32,
            )?;
            self.ops.run(
                s,
                LwaInputs {
                    q: view(
                        qkv,
                        s,
                        row * q_width,
                        [g.heads as u32, g.head_dim as u32],
                        [g.head_dim as u64, 1],
                    )?,
                    k_codes: codes.0,
                    k_scales: scales.0,
                    slot_ids: view(&self.slots, s, 0, [visible as u32], [1])?,
                    sink,
                    inv_rotary: view(&self.rotary_inv, s, row * g.rotary, [g.rotary as u32], [1])?,
                },
                &mut TensorMut(view(
                    &self.o,
                    s,
                    row * q_width,
                    [g.heads as u32, g.head_dim as u32],
                    [g.head_dim as u64, 1],
                )?),
                p_mode,
            )?;
        }
        // Attention output: per-32 quantization before the einsum, then the
        // block-diagonal wo_a per group and wo_b (model definition :786-790).
        self.ops.act_quant(
            s,
            view(
                &self.o,
                s,
                0,
                [rows as u32, q_width as u32],
                [q_width as u64, 1],
            )?,
            &mut TensorMut(view(
                &self.o_codes,
                s,
                0,
                [rows as u32, q_width as u32],
                [q_width as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.o_scales,
                s,
                0,
                [rows as u32, (q_width / 32) as u32],
                [(q_width / 32) as u64, 1],
            )?),
        )?;
        self.ops.group_project(
            s,
            view(
                &self.o_codes,
                s,
                0,
                [rows as u32, g.o_groups as u32, g.o_in as u32],
                [q_width as u64, g.o_in as u64, 1],
            )?,
            view(
                &self.o_scales,
                s,
                0,
                [rows as u32, g.o_groups as u32, (g.o_in / 32) as u32],
                [(q_width / 32) as u64, (g.o_in / 32) as u64, 1],
            )?,
            weight_view::<u8, 2>(&weights.wo_a, s, [o_width as u32, g.o_in as u32], "wo_a")?,
            weight_view::<u8, 2>(
                &weights.wo_a_scale,
                s,
                [(o_width / 32) as u32, (g.o_in / 32) as u32],
                "wo_a_scale",
            )?,
            &mut TensorMut(view(
                &self.groups,
                s,
                0,
                [rows as u32, o_width as u32],
                [o_width as u64, 1],
            )?),
        )?;
        self.ops.act_quant(
            s,
            view(
                &self.groups,
                s,
                0,
                [rows as u32, o_width as u32],
                [o_width as u64, 1],
            )?,
            &mut TensorMut(view(
                &self.act_o_codes,
                s,
                0,
                [rows as u32, o_width as u32],
                [o_width as u64, 1],
            )?),
            &mut TensorMut(view(
                &self.act_o_scales,
                s,
                0,
                [rows as u32, (o_width / 32) as u32],
                [(o_width / 32) as u64, 1],
            )?),
        )?;
        proj(
            "wo_b",
            &weights.wo_b,
            g.hidden,
            o_width,
            &self.act_o_codes,
            &self.act_o_scales,
            TensorMut(view(
                out,
                s,
                0,
                [rows as u32, g.hidden as u32],
                [g.hidden as u64, 1],
            )?),
        )?;
        Ok(())
    }
}

/// Cross-derived FFN geometry of one DeepSeek-V4.1 model, checked against
/// the config once at setup (stored layouts, model definition Gate/MoE).
#[derive(Debug)]
struct FfnGeometry {
    hidden: usize,
    experts: usize,
    top_k: usize,
    expert_intermediate: usize,
    shared_intermediate: usize,
    limit: f32,
    routed_scaling: f32,
    score: RouterScore,
}

impl FfnGeometry {
    fn new(spec: &ModelSpec) -> Result<Self> {
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let moe = spec
            .moe
            .as_ref()
            .context("DeepSeek-V4.1 MoE geometry missing")?;
        ensure!(
            moe.score == RouterScore::SqrtSoftplus,
            "DeepSeek-V4.1 router score {:?} is not sqrt-softplus",
            moe.score
        );
        ensure!(
            !moe.shared_gate,
            "DeepSeek-V4.1 shared-expert sigmoid gate is not part of the model definition"
        );
        ensure!(
            (1..=1024).contains(&moe.experts),
            "DeepSeek-V4.1 routed expert count {} is outside [1, 1024]",
            moe.experts
        );
        ensure!(
            moe.experts_per_tok >= 1
                && moe.experts_per_tok <= 32
                && moe.experts_per_tok <= moe.experts,
            "DeepSeek-V4.1 top-k {} is outside [1, {}]",
            moe.experts_per_tok,
            moe.experts
        );
        for (name, dim) in [
            ("hidden", spec.hidden),
            ("expert intermediate", moe.expert_intermediate),
            ("shared intermediate", moe.shared_intermediate),
        ] {
            ensure!(
                dim >= 32 && dim.is_multiple_of(32),
                "DeepSeek-V4.1 FFN {name} {dim} is not a positive multiple of the 32-element scale block"
            );
        }
        ensure!(
            ds.swiglu_limit.is_finite() && ds.swiglu_limit > 0.0,
            "DeepSeek-V4.1 SwiGLU limit {} is not positive and finite",
            ds.swiglu_limit
        );
        ensure!(
            ds.routed_scaling_factor.is_finite() && ds.routed_scaling_factor > 0.0,
            "DeepSeek-V4.1 routed scaling {} is not positive and finite",
            ds.routed_scaling_factor
        );
        Ok(Self {
            hidden: spec.hidden as usize,
            experts: moe.experts as usize,
            top_k: moe.experts_per_tok as usize,
            expert_intermediate: moe.expert_intermediate as usize,
            shared_intermediate: moe.shared_intermediate as usize,
            limit: ds.swiglu_limit,
            routed_scaling: ds.routed_scaling_factor,
            score: moe.score,
        })
    }

    /// Byte extents of the layer FFN weight buffers (stored layouts).
    fn ffn_weight_extents(&self) -> [(&'static str, usize); 10] {
        let blocks = |rows: usize, cols: usize| rows.div_ceil(32) * cols / 32;
        [
            ("gate_weight", 2 * self.experts * self.hidden),
            ("gate_bias", 4 * self.experts),
            ("norm", 2 * self.hidden),
            ("post_norm", 2 * self.hidden),
            ("shared_gate_codes", self.shared_intermediate * self.hidden),
            (
                "shared_gate_scales",
                blocks(self.shared_intermediate, self.hidden),
            ),
            ("shared_up_codes", self.shared_intermediate * self.hidden),
            (
                "shared_up_scales",
                blocks(self.shared_intermediate, self.hidden),
            ),
            ("shared_down_codes", self.hidden * self.shared_intermediate),
            (
                "shared_down_scales",
                blocks(self.hidden, self.shared_intermediate),
            ),
        ]
    }

    /// Byte extents of one routed expert's W4A8 weight buffers (stored
    /// layouts: E2M1 codes packed two per byte, one E8M0 scale per 32 elements
    /// of each row).
    fn routed_weight_extents(&self) -> [(&'static str, usize); 6] {
        let inter = self.expert_intermediate;
        [
            ("gate_codes", inter * self.hidden / 2),
            ("gate_scales", inter * self.hidden / 32),
            ("up_codes", inter * self.hidden / 2),
            ("up_scales", inter * self.hidden / 32),
            ("down_codes", self.hidden * inter / 2),
            ("down_scales", self.hidden * inter / 32),
        ]
    }
}

/// Owned device weight bytes of one layer's shared FFN parts (the audited
/// manifest path uploads them byte-exact). Stored layouts: gate.weight
/// `[experts, hidden]` bf16, gate.bias `[experts]` f32 (the audited name; no
/// e_score_correction_bias in this checkpoint), norm/post_norm `[hidden]` bf16,
/// shared expert E4M3 codes `[intermediate, hidden]` / `[hidden, intermediate]`
/// with `[ceil(rows/32), cols/32]` E8M0 block scales.
pub struct FfnWeights {
    pub gate_weight: CudaSlice<u8>,
    pub gate_bias: CudaSlice<u8>,
    pub norm: CudaSlice<u8>,
    pub post_norm: CudaSlice<u8>,
    pub shared_gate_codes: CudaSlice<u8>,
    pub shared_gate_scales: CudaSlice<u8>,
    pub shared_up_codes: CudaSlice<u8>,
    pub shared_up_scales: CudaSlice<u8>,
    pub shared_down_codes: CudaSlice<u8>,
    pub shared_down_scales: CudaSlice<u8>,
}

/// One routed expert's W4A8 weight bytes: E2M1 codes packed two values per
/// byte with one E8M0 scale per 32 elements of each row.
pub struct RoutedExpertWeights {
    pub gate_codes: CudaSlice<u8>,
    pub gate_scales: CudaSlice<u8>,
    pub up_codes: CudaSlice<u8>,
    pub up_scales: CudaSlice<u8>,
    pub down_codes: CudaSlice<u8>,
    pub down_scales: CudaSlice<u8>,
}

/// Bound FFN state of one layer: the shared parts, the routed expert subset,
/// the host mirror of the router bias (the per-row device expansion is
/// rebuilt from it when the workspace grows) and the device dispatch table
/// of the bound routed experts (`[experts, 6]` weight addresses plus the
/// residency mask; built by `finish_routed`, invalidated by `bind_routed`).
struct LayerFfn {
    weights: FfnWeights,
    routed: Vec<Option<RoutedExpertWeights>>,
    gate_bias: Vec<f32>,
    bias_expanded: CudaSlice<f32>,
    slots: Option<CudaSlice<u64>>,
    slot_valid: CudaSlice<u32>,
}

/// DeepSeek-V4.1 FFN runtime (model definition MoE.forward, decode path):
/// the BF16 router GEMV with FP32 logits on the shared RouteOps cuBLAS
/// handle, the sqrt-softplus noaux_tc top-k with unbiased weights,
/// norm-top-k and routed scaling, the routed W4A8 experts dispatched on
/// device through the per-layer slot tables (no routing data crosses to
/// host), the W8A8 shared expert at weight 1 and the FP32 combine with one
/// BF16 round.
pub struct DeepseekFfn {
    route: RouteOps,
    experts: w4a8::MoeExpertW4A8,
    shared: w8a8::MoeSharedW8A8,
    combine: CombineOps,
    common: CommonOpsRef,
    geometry: FfnGeometry,
    layers: Vec<Option<LayerFfn>>,
    rows: usize,
    logits: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    route_weights: CudaSlice<f32>,
    contributions: CudaSlice<bf16>,
    shared_out: CudaSlice<bf16>,
    routed_scratch: CudaSlice<u8>,
    shared_scratch: CudaSlice<u8>,
    eps: f32,
    tiered: bool,
}

struct DeepseekFfnRun<'a> {
    s: &'a Arc<CudaStream>,
    rows: usize,
    x: &'a CudaSlice<bf16>,
    out: &'a mut CudaSlice<bf16>,
    layer: usize,
    accumulator: Accumulator,
    tier: Option<&'a mut super::moe::Slots>,
    #[cfg(test)]
    fixed_routes: Option<(Vec<i32>, Vec<f32>)>,
}

/// One RMS-norm stage call over the engine work buffers.
pub(crate) struct NormStep<'a> {
    pub(crate) seq: SeqRef,
    pub(crate) rows: usize,
    pub(crate) x: &'a CudaSlice<bf16>,
    pub(crate) n: &'a mut CudaSlice<bf16>,
    pub(crate) layer: usize,
    pub(crate) post: bool,
}

/// Test view of one wo_b weight row and the projection input it consumes.
#[cfg(test)]
pub(crate) struct WoBRow {
    pub act_codes: Vec<u8>,
    pub act_scales: Vec<u8>,
    pub w_codes: Vec<u8>,
    pub w_scales: Vec<u8>,
}

#[cfg(test)]
pub(crate) struct RecordedWindowReplay<'a> {
    pub(crate) s: &'a Arc<CudaStream>,
    pub(crate) blas: &'a Blas,
    pub(crate) ring: &'a mut LatentRing,
    pub(crate) layer: usize,
    pub(crate) position: usize,
    pub(crate) q: &'a CudaSlice<bf16>,
    pub(crate) codes: &'a [u8],
    pub(crate) scales: &'a [u8],
    pub(crate) out: &'a mut CudaSlice<bf16>,
}

/// Test view of the FFN stage buffers.
#[cfg(test)]
pub(crate) struct StageBuffers<'a> {
    pub(crate) logits: &'a CudaSlice<f32>,
    pub(crate) ids: &'a CudaSlice<i32>,
    pub(crate) route_weights: &'a CudaSlice<f32>,
    pub(crate) contributions: &'a CudaSlice<bf16>,
    pub(crate) shared_out: &'a CudaSlice<bf16>,
}

impl DeepseekFfn {
    /// Allocates the runtime for `layers` layers and `rows` decode rows; the
    /// per-layer weights are bound separately.
    pub fn new(
        ctx: &Arc<CudaContext>,
        s: &Arc<CudaStream>,
        spec: &ModelSpec,
        layers: usize,
        rows: usize,
    ) -> Result<Self> {
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let geometry = FfnGeometry::new(spec)?;
        ensure!(
            layers > 0 && layers <= ds.num_hidden_layers as usize,
            "DeepSeek-V4.1 FFN layer count {layers} is zero or exceeds the depth {}",
            ds.num_hidden_layers
        );
        ensure!(
            rows > 0,
            "DeepSeek-V4.1 FFN needs at least one row (workspace capacity)"
        );
        let route = RouteOps::new(ctx, s, geometry.experts as u32, rows)?;
        let experts = geometry.experts;
        let top_k = geometry.top_k;
        let hidden = geometry.hidden;
        let shared_intermediate = geometry.shared_intermediate;
        let expert_intermediate = geometry.expert_intermediate;
        Ok(Self {
            route,
            experts: w4a8::MoeExpertW4A8::new(ctx)?,
            shared: w8a8::MoeSharedW8A8::new(ctx)?,
            combine: CombineOps::new(ctx)?,
            common: CommonOpsRef::new_with_swiglu_rounding(
                ctx,
                crate::model_spec::SwigluRounding::ClampedTwice,
                spec.swiglu_limit(),
            )?,
            geometry,
            layers: (0..layers).map(|_| None).collect(),
            rows,
            logits: s.alloc_zeros(rows * experts)?,
            ids: s.alloc_zeros(rows * top_k)?,
            route_weights: s.alloc_zeros(rows * top_k)?,
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            shared_out: s.alloc_zeros(rows * hidden)?,
            routed_scratch: s.alloc_zeros(
                usize::try_from(w4a8::scratch_bytes(
                    hidden as u32,
                    expert_intermediate as u32,
                ))?
                .max(1),
            )?,
            shared_scratch: s.alloc_zeros(
                usize::try_from(w8a8::scratch_bytes(
                    hidden as u32,
                    shared_intermediate as u32,
                ))?
                .max(1),
            )?,
            tiered: false,
            eps: spec.rms_eps,
        })
    }

    fn expand_bias(
        s: &Arc<CudaStream>,
        bias: &[f32],
        experts: usize,
        rows: usize,
    ) -> Result<CudaSlice<f32>> {
        let expanded: Vec<f32> = bias.iter().copied().cycle().take(rows * experts).collect();
        let mut slice = s.alloc_zeros(rows * experts)?;
        s.memcpy_htod(&expanded, &mut slice.slice_mut(..))?;
        Ok(slice)
    }

    /// Binds the shared FFN weight bytes of one layer.
    pub fn bind_layer(
        &mut self,
        s: &Arc<CudaStream>,
        layer: usize,
        weights: FfnWeights,
    ) -> Result<()> {
        ensure!(
            layer < self.layers.len(),
            "DeepSeek-V4.1 FFN layer {layer} is outside the bound {} layers",
            self.layers.len()
        );
        let g = &self.geometry;
        // The byte list follows ffn_weight_extents order.
        for ((name, expected), bytes) in g.ffn_weight_extents().iter().zip([
            &weights.gate_weight,
            &weights.gate_bias,
            &weights.norm,
            &weights.post_norm,
            &weights.shared_gate_codes,
            &weights.shared_gate_scales,
            &weights.shared_up_codes,
            &weights.shared_up_scales,
            &weights.shared_down_codes,
            &weights.shared_down_scales,
        ]) {
            ensure!(
                bytes.len() == *expected,
                "DeepSeek-V4.1 FFN {name} holds {} bytes; the bound geometry needs {expected}",
                bytes.len()
            );
        }
        let raw: Vec<u8> = s.clone_dtoh(&weights.gate_bias)?;
        let gate_bias: Vec<f32> = (0..raw.len())
            .step_by(4)
            .map(|i| f32::from_le_bytes(raw[i..i + 4].try_into().expect("f32 bytes")))
            .collect();
        ensure!(
            gate_bias.iter().all(|v| v.is_finite()),
            "DeepSeek-V4.1 FFN gate_bias holds a non-finite entry"
        );
        let bias_expanded = Self::expand_bias(s, &gate_bias, g.experts, self.rows)?;
        self.layers[layer] = Some(LayerFfn {
            weights,
            routed: (0..g.experts).map(|_| None).collect(),
            gate_bias,
            bias_expanded,
            slots: None,
            slot_valid: s.alloc_zeros(g.experts)?,
        });
        Ok(())
    }

    /// Binds the W4A8 weight bytes of one routed expert of one layer.
    pub fn bind_routed(
        &mut self,
        layer: usize,
        expert: usize,
        weights: RoutedExpertWeights,
    ) -> Result<()> {
        let g = &self.geometry;
        ensure!(
            expert < g.experts,
            "DeepSeek-V4.1 routed expert id {expert} is outside the catalogue {}",
            g.experts
        );
        let bound = self
            .layers
            .get_mut(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_mut()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        // The byte list follows routed_weight_extents order.
        for ((name, expected), bytes) in g.routed_weight_extents().iter().zip([
            &weights.gate_codes,
            &weights.gate_scales,
            &weights.up_codes,
            &weights.up_scales,
            &weights.down_codes,
            &weights.down_scales,
        ]) {
            ensure!(
                bytes.len() == *expected,
                "DeepSeek-V4.1 FFN routed {name} holds {} bytes; the bound geometry needs {expected}",
                bytes.len()
            );
        }
        bound.routed[expert] = Some(weights);
        bound.slots = None;
        Ok(())
    }

    /// Builds the device dispatch tables (`[experts, 6]` weight addresses and
    /// the residency mask) of every layer with bound routed weights, from the
    /// bound weights alone. Required after the last `bind_routed` of a layer
    /// before the first decode; an unbound routed id keeps a zero valid entry
    /// and traps the decode launch.
    pub fn finish_routed(&mut self, s: &Arc<CudaStream>) -> Result<()> {
        let g = &self.geometry;
        for (layer, bound) in self.layers.iter_mut().enumerate() {
            let Some(bound) = bound.as_mut() else {
                continue;
            };
            ensure!(
                bound.routed.iter().any(|slot| slot.is_some()),
                "DeepSeek-V4.1 FFN layer {layer} has no bound routed experts; the tiered expert service is not bound"
            );
            let mut valid: Vec<u32> = Vec::with_capacity(g.experts);
            let mut table: Vec<u64> = Vec::with_capacity(g.experts * w4a8::W4A8_SLOT_POINTERS);
            for (expert, slot) in bound.routed.iter().enumerate() {
                let Some(weights) = slot.as_ref() else {
                    valid.push(0);
                    table.extend(std::iter::repeat_n(0, w4a8::W4A8_SLOT_POINTERS));
                    continue;
                };
                let weights = Self::routed_views(s, weights, g)?;
                for tensor in [
                    &weights.gate_codes,
                    &weights.gate_scales,
                    &weights.up_codes,
                    &weights.up_scales,
                    &weights.down_codes,
                    &weights.down_scales,
                ] {
                    ensure!(
                        tensor.ptr != 0,
                        "DeepSeek-V4.1 FFN layer {layer} expert {expert} slot holds a null weight address"
                    );
                }
                valid.push(1);
                table.extend([
                    weights.gate_codes.ptr,
                    weights.gate_scales.ptr,
                    weights.up_codes.ptr,
                    weights.up_scales.ptr,
                    weights.down_codes.ptr,
                    weights.down_scales.ptr,
                ]);
            }
            s.memcpy_htod(&valid, &mut bound.slot_valid.slice_mut(..))?;
            bound.slots = Some(s.clone_htod(&table)?);
        }
        Ok(())
    }

    pub fn bind_tier(&mut self, s: &Arc<CudaStream>) -> Result<()> {
        let g = &self.geometry;
        for bound in &mut self.layers {
            let bound = bound
                .as_mut()
                .context("DeepSeek-V4.1 FFN layer has no bound weights")?;
            bound.routed.clear();
            bound.slots = Some(s.alloc_zeros(g.experts * w4a8::W4A8_SLOT_POINTERS)?);
            s.memset_zeros(&mut bound.slot_valid.slice_mut(..))?;
        }
        self.tiered = true;
        Ok(())
    }

    fn sync_tier(
        &mut self,
        s: &Arc<CudaStream>,
        layer: usize,
        slots: &super::moe::Slots,
    ) -> Result<()> {
        let layer_u32 = u32::try_from(layer).context("DeepSeek-V4.1 tier layer overflow")?;
        let (table, valid) = slots.native_dispatch(layer_u32)?;
        let bound = self
            .layers
            .get_mut(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_mut()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        let slot_table = bound
            .slots
            .as_mut()
            .context("DeepSeek-V4.1 tier dispatch table missing")?;
        s.memcpy_htod(&table, &mut slot_table.slice_mut(..))?;
        s.memcpy_htod(&valid, &mut bound.slot_valid.slice_mut(..))?;
        Ok(())
    }

    /// The device dispatch table of one layer; visible to the gates for the
    /// residency and pointer-table controls.
    #[cfg(test)]
    pub(crate) fn routed_slots(&self, layer: usize) -> Result<(&CudaSlice<u64>, &CudaSlice<u32>)> {
        let bound = self
            .layers
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        let slots = bound.slots.as_ref().context(
            "DeepSeek-V4.1 FFN routed dispatch tables are not built; call finish_routed",
        )?;
        Ok((slots, &bound.slot_valid))
    }

    /// Reallocates the per-row scratch for a larger workspace; the weights
    /// are retained and the router bias expansion is rebuilt per layer.
    pub fn grow(&mut self, s: &Arc<CudaStream>, rows: usize) -> Result<()> {
        ensure!(rows >= self.rows, "DeepSeek-V4.1 FFN rows cannot shrink");
        if rows == self.rows {
            return Ok(());
        }
        let g = &self.geometry;
        for layer in self.layers.iter_mut().flatten() {
            layer.bias_expanded = Self::expand_bias(s, &layer.gate_bias, g.experts, rows)?;
        }
        self.logits = s.alloc_zeros(rows * g.experts)?;
        self.ids = s.alloc_zeros(rows * g.top_k)?;
        self.route_weights = s.alloc_zeros(rows * g.top_k)?;
        self.contributions = s.alloc_zeros(rows * g.top_k * g.hidden)?;
        self.shared_out = s.alloc_zeros(rows * g.hidden)?;
        self.rows = rows;
        Ok(())
    }

    fn routed_views(
        s: &Arc<CudaStream>,
        weights: &RoutedExpertWeights,
        g: &FfnGeometry,
    ) -> Result<w4a8::W4A8ExpertWeights> {
        let inter = g.expert_intermediate as u32;
        let hidden = g.hidden as u32;
        Ok(w4a8::W4A8ExpertWeights {
            gate_codes: weight_view::<u8, 2>(
                &weights.gate_codes,
                s,
                [inter, hidden / 2],
                "routed gate_codes",
            )?,
            gate_scales: weight_view::<u8, 2>(
                &weights.gate_scales,
                s,
                [inter, hidden / w4a8::W4A8_SCALE_BLOCK],
                "routed gate_scales",
            )?,
            up_codes: weight_view::<u8, 2>(
                &weights.up_codes,
                s,
                [inter, hidden / 2],
                "routed up_codes",
            )?,
            up_scales: weight_view::<u8, 2>(
                &weights.up_scales,
                s,
                [inter, hidden / w4a8::W4A8_SCALE_BLOCK],
                "routed up_scales",
            )?,
            down_codes: weight_view::<u8, 2>(
                &weights.down_codes,
                s,
                [hidden, inter / 2],
                "routed down_codes",
            )?,
            down_scales: weight_view::<u8, 2>(
                &weights.down_scales,
                s,
                [hidden, inter / w4a8::W4A8_SCALE_BLOCK],
                "routed down_scales",
            )?,
        })
    }

    /// Plain RMS of `x` into `n` with the layer's sublayer norm weight; the
    /// model definition Block.forward norm applies to the hc_pre collapse
    /// with no residual re-add (the stream carries the residual).
    pub(crate) fn norm(&self, s: &Arc<CudaStream>, step: NormStep<'_>) -> Result<()> {
        let NormStep {
            seq,
            rows,
            x,
            n,
            layer,
            post,
        } = step;
        let g = &self.geometry;
        ensure!(
            rows >= 1 && rows <= self.rows,
            "DeepSeek-V4.1 FFN rows {rows} are outside the runtime capacity {}",
            self.rows
        );
        ensure!(
            x.len() >= rows * g.hidden && n.len() >= rows * g.hidden,
            "DeepSeek-V4.1 FFN work buffers do not hold {rows} rows"
        );
        let bound = self
            .layers
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        let (name, bytes) = if post {
            ("post_norm", &bound.weights.post_norm)
        } else {
            ("norm", &bound.weights.norm)
        };
        let weight = weight_view::<bf16, 1>(bytes, s, [g.hidden as u32], name)?;
        let x3 = view(
            x,
            s,
            0,
            [rows as u32, 1, g.hidden as u32],
            [g.hidden as u64, g.hidden as u64, 1],
        )?;
        let mut n3 = TensorMut(view(
            n,
            s,
            0,
            [rows as u32, 1, g.hidden as u32],
            [g.hidden as u64, g.hidden as u64, 1],
        )?);
        let domain = RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: rows as u32,
        };
        self.common.norm.rms_with(
            s,
            domain,
            x3,
            weight,
            &mut n3,
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps: self.eps,
            },
        )
    }

    /// Model-final RMS of the head-collapsed rows with the global norm
    /// weight (model.py:1271 norm after hc_head).
    pub(crate) fn final_norm(
        &self,
        s: &Arc<CudaStream>,
        seq: SeqRef,
        rows: usize,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        n: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let g = &self.geometry;
        ensure!(
            rows >= 1 && rows <= self.rows,
            "DeepSeek-V4.1 final norm rows {rows} are outside the runtime capacity {}",
            self.rows
        );
        ensure!(
            x.len() >= rows * g.hidden && n.len() >= rows * g.hidden,
            "DeepSeek-V4.1 final norm work buffers do not hold {rows} rows"
        );
        let weight = view(weight, s, 0, [g.hidden as u32], [1])?;
        let x3 = view(
            x,
            s,
            0,
            [rows as u32, 1, g.hidden as u32],
            [g.hidden as u64, g.hidden as u64, 1],
        )?;
        let mut n3 = TensorMut(view(
            n,
            s,
            0,
            [rows as u32, 1, g.hidden as u32],
            [g.hidden as u64, g.hidden as u64, 1],
        )?);
        let domain = RowDomain {
            meta: seq,
            row_base: 0,
            row_capacity: rows as u32,
        };
        self.common.norm.rms_with(
            s,
            domain,
            x3,
            weight,
            &mut n3,
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps: self.eps,
            },
        )
    }

    /// One decode step of `rows` rows (model definition MoE.forward over the
    /// normed input): router logits, top-k route, routed experts, shared
    /// expert, FP32 combine.
    fn route_only(
        &mut self,
        s: &Arc<CudaStream>,
        rows: usize,
        x: &CudaSlice<bf16>,
        layer: usize,
    ) -> Result<()> {
        let g = &self.geometry;
        let bound = self
            .layers
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        let x2 = view(
            x,
            s,
            0,
            [rows as u32, g.hidden as u32],
            [g.hidden as u64, 1],
        )?;
        let router = weight_view::<bf16, 2>(
            &bound.weights.gate_weight,
            s,
            [g.experts as u32, g.hidden as u32],
            "gate_weight",
        )?;
        let mut logits = TensorMut(view(
            &self.logits,
            s,
            0,
            [rows as u32, g.experts as u32],
            [g.experts as u64, 1],
        )?);
        self.route.logits_f32(s, x2, router, &mut logits)?;
        let bias = view(
            &bound.bias_expanded,
            s,
            0,
            [rows as u32, g.experts as u32],
            [g.experts as u64, 1],
        )?;
        let mut ids = TensorMut(view(
            &self.ids,
            s,
            0,
            [rows as u32, g.top_k as u32],
            [g.top_k as u64, 1],
        )?);
        let mut route_weights = TensorMut(view(
            &self.route_weights,
            s,
            0,
            [rows as u32, g.top_k as u32],
            [g.top_k as u64, 1],
        )?);
        self.route.topk_noaux_f32(
            logits.0,
            Some(bias),
            &mut ids,
            &mut route_weights,
            g.score,
            g.routed_scaling,
        )
    }

    pub(crate) fn run(
        &mut self,
        s: &Arc<CudaStream>,
        rows: usize,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        layer: usize,
        accumulator: Accumulator,
    ) -> Result<()> {
        self.run_with_tier(DeepseekFfnRun {
            s,
            rows,
            x,
            out,
            layer,
            accumulator,
            tier: None,
            #[cfg(test)]
            fixed_routes: None,
        })
    }

    fn run_with_tier(&mut self, run: DeepseekFfnRun<'_>) -> Result<()> {
        let DeepseekFfnRun {
            s,
            rows,
            x,
            out,
            layer,
            accumulator,
            mut tier,
            #[cfg(test)]
            fixed_routes,
        } = run;
        let (hidden, top_k) = (self.geometry.hidden, self.geometry.top_k);
        ensure!(
            rows >= 1 && rows <= self.rows,
            "DeepSeek-V4.1 FFN rows {rows} are outside the runtime capacity {}",
            self.rows
        );
        ensure!(
            x.len() >= rows * hidden && out.len() >= rows * hidden,
            "DeepSeek-V4.1 FFN work buffers do not hold {rows} rows"
        );
        let x2 = view(x, s, 0, [rows as u32, hidden as u32], [hidden as u64, 1])?;
        self.route_only(s, rows, x, layer)?;
        if let Some(slots) = tier.as_mut() {
            ensure!(
                self.tiered,
                "DeepSeek-V4.1 tier executor is not bound; call bind_tier"
            );
            #[cfg(test)]
            if let Some((ids, weights)) = fixed_routes {
                s.memcpy_htod(&ids, &mut self.ids.slice_mut(..ids.len()))?;
                s.memcpy_htod(&weights, &mut self.route_weights.slice_mut(..weights.len()))?;
            }
            let ids = s.clone_dtoh(&self.ids.slice(..rows * top_k))?;
            let weights = s.clone_dtoh(&self.route_weights.slice(..rows * top_k))?;
            slots.resolve_native(
                u32::try_from(layer).context("DeepSeek-V4.1 tier layer overflow")?,
                &ids,
                &weights,
            )?;
            slots.wait_refill(s)?;
            self.sync_tier(s, layer, slots)?;
        }
        let g = &self.geometry;
        let bound = self
            .layers
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_ref()
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?;
        let ids = TensorMut(view(
            &self.ids,
            s,
            0,
            [rows as u32, g.top_k as u32],
            [g.top_k as u64, 1],
        )?);
        let route_weights = TensorMut(view(
            &self.route_weights,
            s,
            0,
            [rows as u32, g.top_k as u32],
            [g.top_k as u64, 1],
        )?);
        // Routed experts dispatched on device: the ids and route weights stay
        // in device memory and the launches are sequenced from the host-known
        // rows/top-k bounds only; a routed id outside the bound set holds a
        // zero valid entry and traps the launch.
        let slots = bound.slots.as_ref().context(
            "DeepSeek-V4.1 FFN routed dispatch tables are not built; call finish_routed",
        )?;
        let routed_scratch_bytes = usize::try_from(w4a8::scratch_bytes(
            g.hidden as u32,
            g.expert_intermediate as u32,
        ))
        .context("DeepSeek-V4.1 routed scratch extent overflow")?;
        let shared_scratch_bytes = usize::try_from(w8a8::scratch_bytes(
            g.hidden as u32,
            g.shared_intermediate as u32,
        ))
        .context("DeepSeek-V4.1 shared scratch extent overflow")?;
        let routed_scratch = TensorMut(view(
            &self.routed_scratch,
            s,
            0,
            [u32::try_from(routed_scratch_bytes)?],
            [1],
        )?);
        let shared_scratch = TensorMut(view(
            &self.shared_scratch,
            s,
            0,
            [u32::try_from(shared_scratch_bytes)?],
            [1],
        )?);
        {
            let mut contributions = TensorMut(view(
                &self.contributions,
                s,
                0,
                [rows as u32, g.top_k as u32, g.hidden as u32],
                [(g.top_k * g.hidden) as u64, g.hidden as u64, 1],
            )?);
            let dispatch = w4a8::W4A8Dispatch {
                x: x2,
                ids: ids.0,
                route_weights: route_weights.0,
                slots: view(
                    slots,
                    s,
                    0,
                    [g.experts as u32, w4a8::W4A8_SLOT_POINTERS as u32],
                    [w4a8::W4A8_SLOT_POINTERS as u64, 1],
                )?,
                slot_valid: view(&bound.slot_valid, s, 0, [g.experts as u32], [1])?,
                intermediate: g.expert_intermediate as u32,
                swiglu_limit: g.limit,
                contributions: &mut contributions,
            };
            self.experts.run_dispatch(s, dispatch, &routed_scratch.0)?;
        }
        for row in 0..rows {
            let sinter = g.shared_intermediate as u32;
            let hidden = g.hidden as u32;
            let shared_weights = w8a8::W8A8SharedWeights {
                gate_codes: weight_view::<u8, 2>(
                    &bound.weights.shared_gate_codes,
                    s,
                    [sinter, hidden],
                    "shared gate_codes",
                )?,
                gate_scales: weight_view::<u8, 2>(
                    &bound.weights.shared_gate_scales,
                    s,
                    [
                        sinter.div_ceil(w8a8::W8A8_SCALE_BLOCK),
                        hidden / w8a8::W8A8_SCALE_BLOCK,
                    ],
                    "shared gate_scales",
                )?,
                up_codes: weight_view::<u8, 2>(
                    &bound.weights.shared_up_codes,
                    s,
                    [sinter, hidden],
                    "shared up_codes",
                )?,
                up_scales: weight_view::<u8, 2>(
                    &bound.weights.shared_up_scales,
                    s,
                    [
                        sinter.div_ceil(w8a8::W8A8_SCALE_BLOCK),
                        hidden / w8a8::W8A8_SCALE_BLOCK,
                    ],
                    "shared up_scales",
                )?,
                down_codes: weight_view::<u8, 2>(
                    &bound.weights.shared_down_codes,
                    s,
                    [hidden, sinter],
                    "shared down_codes",
                )?,
                down_scales: weight_view::<u8, 2>(
                    &bound.weights.shared_down_scales,
                    s,
                    [
                        hidden.div_ceil(w8a8::W8A8_SCALE_BLOCK),
                        sinter / w8a8::W8A8_SCALE_BLOCK,
                    ],
                    "shared down_scales",
                )?,
            };
            let input = w8a8::W8A8Inputs {
                x: view(x, s, row * g.hidden, [hidden], [1])?,
                weights: &shared_weights,
                swiglu_limit: g.limit,
            };
            let mut shared_row =
                TensorMut(view(&self.shared_out, s, row * g.hidden, [hidden], [1])?);
            self.shared
                .run(s, input, &shared_scratch.0, &mut shared_row)?;
        }
        self.combine.combine_accum(
            s,
            &MoeCombineAccum {
                accumulator,
                contributions: view(
                    &self.contributions,
                    s,
                    0,
                    [rows as u32, g.top_k as u32, g.hidden as u32],
                    [(g.top_k * g.hidden) as u64, g.hidden as u64, 1],
                )?,
                shared_out: view(
                    &self.shared_out,
                    s,
                    0,
                    [rows as u32, g.hidden as u32],
                    [g.hidden as u64, 1],
                )?,
                y: TensorMut(view(
                    out,
                    s,
                    0,
                    [rows as u32, g.hidden as u32],
                    [g.hidden as u64, 1],
                )?),
            },
        )
    }

    /// Stage buffers visible to the gates: router logits, route outputs,
    /// per-choice contributions and the shared row.
    #[cfg(test)]
    pub(crate) fn stage_buffers(&self) -> StageBuffers<'_> {
        StageBuffers {
            logits: &self.logits,
            ids: &self.ids,
            route_weights: &self.route_weights,
            contributions: &self.contributions,
            shared_out: &self.shared_out,
        }
    }
}

impl Engine {
    pub(super) fn prepare_deepseek_engram(
        &mut self,
        ids: &[u32],
        text_mask: Option<&[bool]>,
    ) -> Result<()> {
        if let Some(engram) = &mut self.deepseek_engram {
            engram.prepare(&self.device.stream, ids, self.position, text_mask)?;
            self.state.set_sequence(
                &self.device.stream,
                self.position as u32,
                ids.len() as u32,
                self.workspace_rows as u32,
            )?;
        }
        Ok(())
    }

    /// DeepSeek-V4.1 embedding, normalization and FP32 head dispatch.
    pub(super) fn deepseek_operator(
        &mut self,
        op: crate::backend::Operator,
        i: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        ensure!(
            !matches!(step.mask, crate::backend::Mask::Tree { .. }),
            "DeepSeek-V4.1 tree execution is not bound yet"
        );
        if matches!(op, crate::backend::Operator::Head) {
            ensure!(
                step.head != Head::All,
                "DeepSeek-V4.1 Head::All is not bound"
            );
            if step.head == Head::None {
                return Ok(());
            }
            ensure!(
                (1..=self.workspace_rows).contains(&step.rows),
                "DeepSeek-V4.1 Head rows outside workspace"
            );
            let h = self.config.hidden_size;
            let input = self.work.n.slice((step.rows - 1) * h..step.rows * h);
            let logits = self
                .logits_f32
                .as_mut()
                .context("FP32 head logits not bound")?;
            self.ops
                .lm_head_f32(&self.device.stream, &input, &self.weights.head, logits)?;
            self.ops
                .argmax_f32(&self.device.stream, logits, &mut self.work.token)?;
            return Ok(());
        }
        if matches!(op, crate::backend::Operator::Embed) {
            if let Some(engram) = &self.deepseek_engram {
                ensure!(
                    engram.prepared == Some((self.position, step.rows)),
                    "Engram records were not prepared for this forward"
                );
            }
            let runtime = self
                .hc
                .as_mut()
                .context("DeepSeek-V4.1 mHC runtime is not bound")?;
            let ids = if matches!(step.input, crate::backend::Input::Token) {
                &self.work.token
            } else {
                &self.work.ids
            };
            return runtime.begin_forward(&self.device.stream, step.rows, ids, &self.weights.embed);
        }
        if matches!(
            op,
            crate::backend::Operator::Norm | crate::backend::Operator::PostNorm
        ) {
            let Engine {
                deepseek_ffn,
                work,
                device,
                state,
                ..
            } = self;
            let ffn = deepseek_ffn
                .as_ref()
                .context("DeepSeek-V4.1 FFN runtime is not bound")?;
            let seq = state.seq_ref(&device.stream)?;
            let post = matches!(op, crate::backend::Operator::PostNorm);
            ffn.norm(
                &device.stream,
                NormStep {
                    seq,
                    rows: step.rows,
                    x: &work.x,
                    n: &mut work.n,
                    layer: i,
                    post,
                },
            )
            .with_context(|| {
                format!(
                    "DeepSeek-V4.1 layer {i} {}",
                    if post { "PostNorm" } else { "Norm" }
                )
            })?;
            return Ok(());
        }
        if matches!(op, crate::backend::Operator::NextNorm) {
            // The model-final norm runs once, after the last layer's HcPost:
            // the stream collapses with the last sublayer's own pre-mix
            // (model.py:1271), then one RMS with the global norm weight.
            // Earlier layers carry the stream unchanged.
            let last = self.config.num_hidden_layers - 1;
            if i != last {
                return Ok(());
            }
            let Engine {
                hc,
                deepseek_ffn,
                work,
                device,
                state,
                weights,
                ..
            } = self;
            let hc = hc
                .as_mut()
                .context("DeepSeek-V4.1 mHC runtime is not bound")?;
            hc.collapse_final(&device.stream, step.rows, &mut work.x)?;
            let ffn = deepseek_ffn
                .as_ref()
                .context("DeepSeek-V4.1 FFN runtime is not bound")?;
            let seq = state.seq_ref(&device.stream)?;
            let norm_weight = weights
                .norm
                .as_ref()
                .context("DeepSeek-V4.1 model-final norm weight missing")?;
            ffn.final_norm(
                &device.stream,
                seq,
                step.rows,
                &work.x,
                norm_weight,
                &mut work.n,
            )?;
            return Ok(());
        }
        anyhow::bail!("DeepSeek-V4.1 {op:?} execution is not bound yet (layer {i})")
    }

    /// DeepSeek-V4.1 FFN stage (stage 6 of the nine-stage program, model
    /// definition MoE.forward): the router, the routed W4A8 experts, the W8A8
    /// shared expert and the FP32 combine over the PostNorm rows, decode only.
    pub(super) fn deepseek_ffn_stage(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        ensure!(
            !matches!(step.mask, crate::backend::Mask::Tree { .. }),
            "DeepSeek-V4.1 tree execution is not bound yet"
        );
        let Engine {
            deepseek_ffn,
            moe_slots,
            work,
            device,
            ..
        } = self;
        let ffn = deepseek_ffn
            .as_mut()
            .context("DeepSeek-V4.1 FFN runtime is not bound")?;
        if ffn.tiered {
            let slots = moe_slots
                .as_mut()
                .context("DeepSeek-V4.1 expert tier slots are not bound")?;
            ffn.run_with_tier(DeepseekFfnRun {
                s: &device.stream,
                rows: step.rows,
                x: &work.n,
                out: &mut work.out,
                layer,
                accumulator: Accumulator::Fp32,
                tier: Some(slots),
                #[cfg(test)]
                fixed_routes: None,
            })
        } else {
            ffn.run(
                &device.stream,
                step.rows,
                &work.n,
                &mut work.out,
                layer,
                Accumulator::Fp32,
            )
        }
    }

    /// DeepSeek-V4.1 Mixer stage (stage 2 of the nine-stage program,
    /// model definition Block.forward): the attention sublayer over the
    /// latent window ring, decode rows only.
    pub(super) fn deepseek_mixer_stage(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        ensure!(
            !matches!(step.mask, crate::backend::Mask::Tree { .. }),
            "DeepSeek-V4.1 tree execution is not bound yet"
        );
        let position = self.position;
        let Engine {
            deepseek_mixer,
            blas,
            ops,
            work,
            device,
            state,
            ..
        } = self;
        let seq = state.seq_ref(&device.stream)?;
        let ring = state.latent_ring(layer)?;
        let mixer = deepseek_mixer
            .as_ref()
            .context("DeepSeek-V4.1 mixer runtime is not bound")?;
        mixer.run(
            &device.stream,
            blas,
            ops,
            ring,
            MixerStep {
                position,
                rows: step.rows,
                x: &work.n,
                qkv: &mut work.qkv,
                out: &mut work.out,
                seq,
            },
            layer,
        )
    }
}

/// Classification of one Text-scope manifest tensor under the dense load:
/// `Resident` uploads the published bytes; `Deferred` names a tensor no
/// bound runtime consumes (the four projection code/scale pairs are
/// resident through the native FP8 block-linear path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DenseLoad {
    Resident,
    Deferred,
}

/// `layers.{l}.ffn.experts.{e}.{role}` -> (layer, expert).
fn expert_index(name: &str) -> Option<(usize, usize)> {
    let mut parts = name.split('.');
    if parts.next()? != "layers" {
        return None;
    }
    let layer: usize = parts.next()?.parse().ok()?;
    if parts.next()? != "ffn" || parts.next()? != "experts" {
        return None;
    }
    let expert: usize = parts.next()?.parse().ok()?;
    Some((layer, expert))
}

/// Dense-load class of one Text-scope tensor name (the audited names; the
/// mHC, projection, FFN-shared and router tensors are resident, and the
/// sparse-attention, engram, vision-token and per-expert tensors stay with
/// their unbound runtimes).
pub(crate) fn dense_load(name: &str) -> DenseLoad {
    if name.contains(".engram.embed.")
        || name.contains(".ffn.experts.")
        || name.ends_with("ffn.gate.bias_vl")
        || name.starts_with("image_")
    {
        return DenseLoad::Deferred;
    }
    DenseLoad::Resident
}

/// DeepSeek-V4.1 engine load (the load_recurrent pattern): the TextOnly
/// manifest maps the closed world, the dense resident weights of every
/// backbone layer bind the three runtimes, and a memory-admitted prefix of
/// the routed experts binds per layer. Decode rows only; no draft model, no
/// prefix cache and no Graph capture (the model-final norm and head stages
/// are not bound).
impl Engine {
    pub(super) fn load_deepseek(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        let started = Instant::now();
        ensure!(
            (1..=model_spec.max_position as usize).contains(&capacity),
            "DeepSeek-V4.1 capacity outside 1..={}",
            model_spec.max_position
        );
        ensure!(
            settings.draft_model.is_none() && !settings.prefix_cache,
            "DeepSeek-V4.1 does not support draft or prefix-cache execution"
        );
        ensure!(
            settings.kv_format == crate::kv::KvFormat::Bf16,
            "DeepSeek-V4.1 requires BF16 KV"
        );
        let ds = model_spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let layers = ds.num_hidden_layers as usize;
        let workspace_rows = settings.spec_budget.rows().min(capacity);
        ensure!(
            requested_chunk.is_none_or(|n| n == workspace_rows),
            "DeepSeek-V4.1 decode rows are {workspace_rows}; chunking is not bound"
        );
        let device = Device::new(ordinal)?;
        let s = &device.stream;
        let mapped = crate::checkpoint::MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(&model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let mut engram =
            DeepseekEngram::new(&device, &model_spec, &manifest, dir, &settings, capacity)?;
        let expert_format = model_spec
            .expert_format
            .context("DeepSeek-V4.1 expert_format missing")?;
        let geometry = crate::experts::catalogue::ExpertGeometry::from_model(&model_spec)?;
        let catalogue = crate::experts::catalogue::ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            expert_format,
        )?;
        let slot_bytes = usize::try_from(catalogue.slot_bytes())
            .context("DeepSeek-V4.1 expert slot bytes overflow")?;
        // Resident-prefix admission from the free device memory: device
        // resident bytes (the dequantized projections hold two BF16 bytes per
        // published code) plus one layer's dequantization staging (freed
        // before the next layer), state, workspace and a fixed margin; the
        // remainder admits whole experts across the layers.
        let mut dense_bytes = 0usize;
        let mut planned_upload_bytes = 0usize;
        for (name, entry) in &manifest.entries {
            if !entry.load || entry.scope != crate::checkpoint::TensorScope::Text {
                continue;
            }
            match dense_load(name) {
                DenseLoad::Resident => {
                    dense_bytes = dense_bytes
                        .checked_add(entry.bytes)
                        .context("DeepSeek-V4.1 dense byte budget overflow")?;
                    planned_upload_bytes = planned_upload_bytes
                        .checked_add(entry.bytes)
                        .context("DeepSeek-V4.1 upload plan overflow")?;
                }
                DenseLoad::Deferred => {}
            }
        }
        let mut state = crate::backend::cuda::state::StateStore::new(&model_spec, capacity, s)?;
        state.compressed = Some(crate::backend::cuda::state::LatentCompressedState::new(
            &model_spec,
            capacity,
            workspace_rows,
            s,
        )?);
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model_spec)?.row_bytes()?;
        let dsa_shape = model_spec
            .deepseek_v41
            .as_ref()
            .context("DSA geometry missing")?
            .selection;
        let fixed = dense_bytes
            .checked_add(state.bytes())
            .and_then(|n| n.checked_add(workspace_rows * row_bytes))
            .and_then(|n| n.checked_add(dsa::scratch_bytes(workspace_rows, capacity, &dsa_shape)))
            .and_then(|n| n.checked_add(512 * 1024 * 1024))
            .context("DeepSeek-V4.1 fixed residency overflow")?;
        let (free, total) = device.ctx.mem_get_info()?;
        let available = free.checked_sub(total / 16 + fixed).context(
            "DeepSeek-V4.1 dense resident weights/workspace exceed the device memory budget",
        )?;
        let admitted_total = available / slot_bytes;
        #[cfg(test)]
        let full_residency_layer = match std::env::var("FF_DS41_FULL_RESIDENCY_LAYER") {
            Ok(value) => value
                .parse::<usize>()
                .context("FF_DS41_FULL_RESIDENCY_LAYER must be a layer index")?,
            Err(std::env::VarError::NotPresent) => usize::MAX,
            Err(error) => return Err(error.into()),
        };
        #[cfg(not(test))]
        let full_residency_layer = usize::MAX;
        let tiered = full_residency_layer == usize::MAX;
        if full_residency_layer != usize::MAX {
            ensure!(
                full_residency_layer < layers,
                "FF_DS41_FULL_RESIDENCY_LAYER {full_residency_layer} is outside the {layers} loaded layers"
            );
        }
        let full_residency = full_residency_layer != usize::MAX;
        let (resident_experts, rest) = if tiered {
            eprintln!(
                "{}",
                serde_json::json!({"deepseek_routed_residency": {
                    "mode": "capacity-derived host/disk tiers",
                    "slot_bytes": slot_bytes,
                    "dense_bytes": dense_bytes,
                    "tiered_service": "setup pending"
                }})
            );
            (0, 0)
        } else {
            let leftover = admitted_total.saturating_sub(geometry.experts as usize);
            let rest_prefix = if std::env::var_os("FF_DS41_LAYERS_LAYER").is_some() {
                1
            } else {
                (leftover / layers.max(1)).min(geometry.experts as usize)
            };
            ensure!(
                geometry.experts as usize + rest_prefix <= admitted_total,
                "layer-full residency does not fit: {admitted_total} admitted slots < {} experts + {rest_prefix}",
                geometry.experts
            );
            eprintln!(
                "{}",
                serde_json::json!({"deepseek_routed_residency": {
                    "mode": "layer-full gate scope",
                    "layer": full_residency_layer,
                    "single_layer_gate": std::env::var_os("FF_DS41_LAYERS_LAYER").is_some(),
                    "experts_per_layer": geometry.experts,
                    "rest_layers_prefix": rest_prefix,
                    "of": geometry.experts,
                    "slot_bytes": slot_bytes,
                    "dense_bytes": dense_bytes,
                    "tiered_service": "not bound; decode traps when routed outside the resident prefix"
                }})
            );
            (geometry.experts as usize, rest_prefix)
        };
        if !tiered {
            ensure!(
                resident_experts > 0,
                "DeepSeek-V4.1 routed-expert residency admits zero of the {} bytes/expert; the tiered expert service is not bound (dense fixed bytes {fixed})",
                slot_bytes
            );
        }
        let mix = MixerGeometry::new(&model_spec)?;
        let ffng = FfnGeometry::new(&model_spec)?;
        let experts = geometry.experts as usize;
        let (hidden, shared_i, inter) = (
            ffng.hidden,
            ffng.shared_intermediate,
            ffng.expert_intermediate,
        );
        let basis = usize::try_from(crate::backend::cuda::mhc::mix_rows(ds.hc_mult))
            .context("DeepSeek-V4.1 mHC basis rows overflow")?;
        let hc_width = ds.hc_mult as usize * hidden;
        let mut hc = HcRuntime::new(
            &device.ctx,
            s,
            deepseek_hc_binding(&model_spec)?,
            layers,
            workspace_rows,
        )?;
        let mut mixer = DeepseekMixer::new(&device.ctx, s, &model_spec, layers, workspace_rows)?;
        let mut ffn = DeepseekFfn::new(&device.ctx, s, &model_spec, layers, workspace_rows)?;
        // One closed-world pass: mHC and mixer weights bind as they load; the
        // FFN weights collect and bind after `checkpoint_mapped` returns (its
        // `finish_upload` orders the upload stream before the compute-stream
        // bias pull of `bind_layer`).
        struct DenseLoadOut {
            dsa: dsa::Weights,
            uploaded: usize,
            expert_uploads: usize,
            embed: CudaSlice<bf16>,
            head: CudaSlice<bf16>,
            norm: CudaSlice<bf16>,
            ffn: Vec<FfnWeights>,
            routed: Vec<Vec<RoutedExpertWeights>>,
        }
        let dense = crate::backend::cuda::weights::checkpoint_mapped(
            &mapped,
            &device,
            Some(&manifest),
            false,
            |loader| {
                fn tensor(
                    loader: &mut crate::backend::cuda::weights::Loader,
                    name: &str,
                    shape: &[usize],
                    dtype: safetensors::Dtype,
                ) -> Result<CudaSlice<u8>> {
                    loader.load_bytes(name, shape, dtype)
                }
                let f32_tensor = |loader: &mut crate::backend::cuda::weights::Loader,
                                  name: &str,
                                  shape: &[usize]|
                 -> Result<CudaSlice<u8>> {
                    tensor(loader, name, shape, safetensors::Dtype::F32)
                };
                let e4m3 = |loader: &mut crate::backend::cuda::weights::Loader,
                            name: &str,
                            shape: &[usize]|
                 -> Result<CudaSlice<u8>> {
                    tensor(loader, name, shape, safetensors::Dtype::F8_E4M3)
                };
                let e8m0 = |loader: &mut crate::backend::cuda::weights::Loader,
                            name: &str,
                            shape: &[usize]|
                 -> Result<CudaSlice<u8>> {
                    tensor(loader, name, shape, safetensors::Dtype::F8_E8M0)
                };
                let bf16 = |loader: &mut crate::backend::cuda::weights::Loader,
                            name: &str,
                            shape: &[usize]|
                 -> Result<CudaSlice<u8>> {
                    tensor(loader, name, shape, safetensors::Dtype::BF16)
                };
                let fp8_projection = |loader: &mut crate::backend::cuda::weights::Loader,
                                      name: &str,
                                      n: usize,
                                      k: usize|
                 -> Result<Fp8Projection> {
                    Ok(Fp8Projection {
                        codes: e4m3(loader, &format!("{name}.weight"), &[n, k])?,
                        scale: e8m0(
                            loader,
                            &format!("{name}.scale"),
                            &[n.div_ceil(32), k.div_ceil(32)],
                        )?,
                    })
                };
                let mut ffn_layers = Vec::with_capacity(layers);
                engram.load_weights(loader)?;
                let dsa = dsa::load_weights(loader, &model_spec)?;
                for layer in 0..layers {
                    let p = format!("layers.{layer}");
                    let hc_sublayer = |loader: &mut crate::backend::cuda::weights::Loader,
                                       role: &str|
                     -> Result<HcSublayerWeights> {
                        Ok(HcSublayerWeights {
                            fn_w: f32_tensor(
                                loader,
                                &format!("{p}.hc_{role}_fn"),
                                &[basis, hc_width],
                            )?,
                            base: f32_tensor(loader, &format!("{p}.hc_{role}_base"), &[basis])?,
                            scale: f32_tensor(loader, &format!("{p}.hc_{role}_scale"), &[3])?,
                        })
                    };
                    hc.bind_layer(
                        layer,
                        hc_sublayer(loader, "attn")?,
                        hc_sublayer(loader, "ffn")?,
                    )?;
                    // Mixer: every projection binds its native codes and scales.
                    let attn = |role: &str| format!("{p}.attn.{role}");
                    mixer.bind_layer(
                        layer,
                        MixerWeights {
                            wq_a: fp8_projection(loader, &attn("wq_a"), mix.q_lora, mix.hidden)?,
                            q_norm: bf16(
                                loader,
                                &format!("{p}.attn.q_norm.weight"),
                                &[mix.q_lora],
                            )?,
                            wq_b: fp8_projection(
                                loader,
                                &attn("wq_b"),
                                mix.heads * mix.head_dim,
                                mix.q_lora,
                            )?,
                            wkv: fp8_projection(loader, &attn("wkv"), mix.head_dim, mix.hidden)?,
                            kv_norm: bf16(
                                loader,
                                &format!("{p}.attn.kv_norm.weight"),
                                &[mix.head_dim],
                            )?,
                            attn_sink: f32_tensor(
                                loader,
                                &format!("{p}.attn.attn_sink"),
                                &[mix.heads],
                            )?,
                            wo_a: e4m3(
                                loader,
                                &format!("{p}.attn.wo_a.weight"),
                                &[mix.o_groups * mix.o_rank, mix.o_in],
                            )?,
                            wo_a_scale: e8m0(
                                loader,
                                &format!("{p}.attn.wo_a.scale"),
                                &[mix.o_groups * mix.o_rank / 32, mix.o_in / 32],
                            )?,
                            wo_b: fp8_projection(
                                loader,
                                &attn("wo_b"),
                                mix.hidden,
                                mix.o_groups * mix.o_rank,
                            )?,
                        },
                    )?;
                    let blocks = |rows: usize, cols: usize| [rows.div_ceil(32), cols.div_ceil(32)];
                    ffn_layers.push(FfnWeights {
                        gate_weight: bf16(
                            loader,
                            &format!("{p}.ffn.gate.weight"),
                            &[experts, hidden],
                        )?,
                        gate_bias: f32_tensor(loader, &format!("{p}.ffn.gate.bias"), &[experts])?,
                        norm: bf16(loader, &format!("{p}.attn_norm.weight"), &[hidden])?,
                        post_norm: bf16(loader, &format!("{p}.ffn_norm.weight"), &[hidden])?,
                        shared_gate_codes: e4m3(
                            loader,
                            &format!("{p}.ffn.shared_experts.w1.weight"),
                            &[shared_i, hidden],
                        )?,
                        shared_gate_scales: e8m0(
                            loader,
                            &format!("{p}.ffn.shared_experts.w1.scale"),
                            &blocks(shared_i, hidden),
                        )?,
                        shared_up_codes: e4m3(
                            loader,
                            &format!("{p}.ffn.shared_experts.w3.weight"),
                            &[shared_i, hidden],
                        )?,
                        shared_up_scales: e8m0(
                            loader,
                            &format!("{p}.ffn.shared_experts.w3.scale"),
                            &blocks(shared_i, hidden),
                        )?,
                        shared_down_codes: e4m3(
                            loader,
                            &format!("{p}.ffn.shared_experts.w2.weight"),
                            &[hidden, shared_i],
                        )?,
                        shared_down_scales: e8m0(
                            loader,
                            &format!("{p}.ffn.shared_experts.w2.scale"),
                            &blocks(hidden, shared_i),
                        )?,
                    });
                }
                let mut routed = Vec::with_capacity(layers);
                let mut expert_uploads = 0usize;
                for layer in 0..layers {
                    let layer_resident = if tiered {
                        0
                    } else if layer == full_residency_layer {
                        geometry.experts as usize
                    } else if full_residency {
                        rest
                    } else {
                        resident_experts
                    };
                    let p = format!("layers.{layer}.ffn.experts");
                    let mut slots = Vec::with_capacity(layer_resident);
                    for expert in 0..layer_resident {
                        let e = format!("{p}.{expert}");
                        slots.push(RoutedExpertWeights {
                            gate_codes: tensor(
                                loader,
                                &format!("{e}.w1.weight"),
                                &[inter, hidden.div_ceil(2)],
                                safetensors::Dtype::I8,
                            )?,
                            gate_scales: e8m0(
                                loader,
                                &format!("{e}.w1.scale"),
                                &[inter, hidden.div_ceil(32)],
                            )?,
                            up_codes: tensor(
                                loader,
                                &format!("{e}.w3.weight"),
                                &[inter, hidden.div_ceil(2)],
                                safetensors::Dtype::I8,
                            )?,
                            up_scales: e8m0(
                                loader,
                                &format!("{e}.w3.scale"),
                                &[inter, hidden.div_ceil(32)],
                            )?,
                            down_codes: tensor(
                                loader,
                                &format!("{e}.w2.weight"),
                                &[hidden, inter.div_ceil(2)],
                                safetensors::Dtype::I8,
                            )?,
                            down_scales: e8m0(
                                loader,
                                &format!("{e}.w2.scale"),
                                &[hidden, inter.div_ceil(32)],
                            )?,
                        });
                        expert_uploads += 1;
                    }
                    routed.push(slots);
                }
                let h = model_spec.hidden as usize;
                let embed =
                    loader.load(&[("embed.weight".into(), vec![model_spec.vocab as usize, h])])?;
                let head =
                    loader.load(&[("head.weight".into(), vec![model_spec.vocab as usize, h])])?;
                let norm = loader.load(&[("norm.weight".into(), vec![h])])?;
                // Every Text tensor the load does not upload stays with its
                // unbound runtime; the manifest stays closed-world.
                for (name, entry) in &manifest.entries {
                    if entry.load && dense_load(name) == DenseLoad::Deferred {
                        if let Some((layer, expert)) = expert_index(name) {
                            let layer_resident = if layer == full_residency_layer {
                                geometry.experts as usize
                            } else if full_residency {
                                rest
                            } else {
                                resident_experts
                            };
                            if expert < layer_resident {
                                continue;
                            }
                        }
                        loader.defer(name)?;
                    }
                }
                Ok(DenseLoadOut {
                    dsa,
                    uploaded: loader.bytes,
                    expert_uploads,
                    embed,
                    head,
                    norm,
                    ffn: ffn_layers,
                    routed,
                })
            },
        )?
        .0;
        ensure!(
            dense.uploaded == planned_upload_bytes + dense.expert_uploads * slot_bytes,
            "DeepSeek-V4.1 uploaded {} bytes; the plan held {planned_upload_bytes} dense plus {} experts at {slot_bytes}",
            dense.uploaded,
            dense.expert_uploads
        );
        for (layer, weights) in dense.ffn.into_iter().enumerate() {
            ffn.bind_layer(s, layer, weights)?;
        }
        for (layer, slots) in dense.routed.into_iter().enumerate() {
            for (expert, weights) in slots.into_iter().enumerate() {
                ffn.bind_routed(layer, expert, weights)?;
            }
        }
        if tiered {
            ffn.bind_tier(s)?;
        } else {
            ffn.finish_routed(s)?;
        }
        engram.bind(&device, &hc, state.seq_ref(s)?, model_spec.rms_eps)?;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, model_spec.swiglu_rounding())?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "DeepSeek-V4.1 fixed setup".into(),
        ));
        let runtime = settings.runtime_dir.clone().unwrap_or_default();
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        blas.enable_fp8(&device.ctx)?;
        let mut work = Work::new(s, &model_spec, workspace_rows)?;
        mixer.dsa = Some(dsa::Runtime::new(
            (&device, &blas, &model_spec, &state, &mixer, &work),
            dense.dsa,
        )?);
        s.memcpy_htod(&[1i32], &mut work.length)?;
        let mut engine = Self {
            #[cfg(test)]
            capture_fail_after_layer: None,
            #[cfg(test)]
            capture_fail_stage: None,
            #[cfg(test)]
            flint_window_capture: None,
            forward_rows: (1..=workspace_rows).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: settings.spec_budget,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            runtime,
            spec: None,
            hidden_capture: None,
            hc: Some(hc),
            glm_dsa: None,
            deepseek_mixer: Some(mixer),
            deepseek_ffn: Some(ffn),
            deepseek_engram: Some(engram),
            qwen: None,
            recurrent: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            #[cfg(test)]
            moe_test_ffn: None,
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            weights: Weights {
                embed: dense.embed,
                head: dense.head,
                norm: Some(dense.norm),
                layers: Vec::new(),
                bytes: dense.uploaded,
            },
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash: None,
            decode: None,
            logits_f32: Some(s.alloc_zeros(model_spec.vocab as usize)?),
            qwen_device_prefix: false,
            device,
            config: model_spec.config(),
            model_spec,
            capacity,
            chunk: workspace_rows,
            workspace_rows,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        if tiered {
            engine.moe_slots = Some(super::moe::Slots::load_staging(
                &engine.device,
                &engine.model_spec,
                &manifest,
                dir,
                u32::try_from(workspace_rows)?,
            )?);
            engine.device.finish_upload()?;
        }
        engine.closure.freeze_fixed()?;
        engine.reset()?;
        engine.closure.ready(started)?;
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::hc::{HcRuntime, HcSublayerWeights};
    use crate::backend::cuda::moe::testing;
    use crate::backend::{Head, Input, Mask, Operator, Step};
    use crate::model::{HcPreSource, HcStage, HcSublayer};
    use crate::model_spec::Architecture;
    use half::bf16;
    use std::collections::BTreeMap;

    #[derive(serde::Deserialize)]
    struct CompositionRow {
        top1: usize,
        margin: f64,
        spread: f64,
        nll: Vec<f64>,
    }

    impl CompositionRow {
        fn id_defect(&self, candidate: usize) -> bool {
            self.margin > self.spread && candidate != self.top1
        }
    }

    #[derive(serde::Deserialize)]
    struct CompositionReference {
        revision: String,
        shape: [usize; 2],
        arms: Vec<String>,
        tokens: Vec<u32>,
        rows: Vec<CompositionRow>,
        nll_path_spread: f64,
        nll_limit: f64,
    }

    fn composition_nll(logits: &[f32], target: usize) -> Result<f64> {
        ensure!(
            target < logits.len() && logits.iter().all(|v| v.is_finite()),
            "Runner FAIL composition logit extent/non-finite value"
        );
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        Ok(max
            + logits
                .iter()
                .map(|&v| (f64::from(v) - max).exp())
                .sum::<f64>()
                .ln()
            - f64::from(logits[target]))
    }

    struct CompositionScore<'a> {
        reference: &'a CompositionReference,
        readers: Vec<std::io::BufReader<std::fs::File>>,
        output: std::io::BufWriter<std::fs::File>,
        nll: f64,
        id_defects: usize,
        exempt: usize,
        control_defects: usize,
        first_exit: Option<usize>,
    }

    impl<'a> CompositionScore<'a> {
        fn new(root: &Path, output: &Path, reference: &'a CompositionReference) -> Result<Self> {
            let mut readers = Vec::new();
            for arm in &reference.arms {
                let file = std::fs::File::open(root.join(format!("{arm}.f32")))?;
                ensure!(
                    file.metadata()?.len()
                        == (reference.shape.iter().product::<usize>() * 4) as u64,
                    "Runner FAIL {arm} composition reference byte extent"
                );
                readers.push(std::io::BufReader::new(file));
            }
            Ok(Self {
                reference,
                readers,
                output: std::io::BufWriter::new(std::fs::File::create_new(output)?),
                nll: 0.0,
                id_defects: 0,
                exempt: 0,
                control_defects: 0,
                first_exit: None,
            })
        }

        fn row(&mut self, position: usize, logits: &[f32]) -> Result<()> {
            use std::io::{Read, Write};
            let reference = &self.reference.rows[position];
            ensure!(
                logits.len() == self.reference.shape[1],
                "composition vocabulary extent"
            );
            let nll = composition_nll(logits, self.reference.tokens[position + 1] as usize)?;
            for value in logits {
                self.output.write_all(&value.to_le_bytes())?;
            }
            self.nll += nll;
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(1, |&a, &b| {
                logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
            });
            let top1 = order[0];
            let margin = f64::from(logits[order[0]]) - f64::from(logits[order[1]]);
            let required = reference.margin > reference.spread;
            let defect = reference.id_defect(top1);
            self.id_defects += usize::from(defect);
            self.exempt += usize::from(!required);
            // The control rotates the reference vocabulary by one label.
            self.control_defects +=
                usize::from(reference.id_defect((reference.top1 + 1) % logits.len()));
            let mut distances = Vec::new();
            for reader in &mut self.readers {
                let mut bytes = vec![0u8; logits.len() * 4];
                reader.read_exact(&mut bytes)?;
                let mut max = 0.0f64;
                let mut squares = 0.0;
                for (&got, raw) in logits.iter().zip(bytes.as_chunks::<4>().0) {
                    let want = f32::from_le_bytes(*raw);
                    ensure!(want.is_finite(), "Runner FAIL non-finite reference logits");
                    let delta = (f64::from(got) - f64::from(want)).abs();
                    max = max.max(delta);
                    squares += delta * delta;
                }
                distances.push((max, (squares / logits.len() as f64).sqrt()));
            }
            if distances[0].0 > reference.spread && self.first_exit.is_none() {
                self.first_exit = Some(position);
            }
            println!(
                "COMPOSITION_ROW position={position} nll={nll:.12e} reference_nll={:.12e} id={top1} reference_id={} margin={margin:.12e} reference_margin={:.12e} path_spread={:.12e} required={required} id_defect={defect} distances_max_rms={distances:?}",
                reference.nll[0], reference.top1, reference.margin, reference.spread
            );
            Ok(())
        }

        fn finish(&mut self, informational: bool) -> Result<bool> {
            use std::io::Write;
            self.output.flush()?;
            let rows = self.reference.rows.len() as f64;
            let reference_mean = self.reference.rows.iter().map(|r| r.nll[0]).sum::<f64>() / rows;
            let delta = self.nll / rows - reference_mean;
            let pass = delta.abs() <= self.reference.nll_limit && self.id_defects == 0;
            println!(
                "{} COMPOSITION informational={informational} within_primary_limits={pass} rows={} mean_nll={:.12e} reference_nll={reference_mean:.12e} delta={delta:.12e} limit={:.12e} id_defects={} exempt={} first_logit_exit_diagnostic={:?} vocabulary_rotation_control_detected={}",
                if informational {
                    "DIAGNOSTIC"
                } else if pass {
                    "PASS"
                } else {
                    "DEFECT"
                },
                self.reference.rows.len(),
                self.nll / rows,
                self.reference.nll_limit,
                self.id_defects,
                self.exempt,
                self.first_exit,
                self.control_defects
            );
            Ok(pass && self.control_defects > 0)
        }
    }

    #[test]
    fn composition_score_detects_vocab_shift_and_nll_drift() -> Result<()> {
        let logits = [2.0, 1.0, -1.0];
        let reference = composition_nll(&logits, 0)?;
        let shifted = composition_nll(&[-1.0, 2.0, 1.0], 0)?;
        assert!(shifted - reference > 2.0);
        assert!((composition_nll(&[102.0, 101.0, 99.0], 0)? - reference).abs() < 1e-12);
        let mut row = CompositionRow {
            top1: 0,
            margin: 1.0,
            spread: 0.5,
            nll: vec![reference],
        };
        assert!(row.id_defect(1));
        assert!(!row.id_defect(0));
        row.spread = 1.0;
        assert!(!row.id_defect(1));
        Ok(())
    }

    fn composition_live_state(engine: &Engine, end: usize, rows: usize) -> Result<()> {
        engine
            .deepseek_mixer
            .as_ref()
            .context("composition native mixer unbound")?
            .check_decode_ready()?;
        ensure!(
            engine.decode.is_none(),
            "composition bound a generic decode executor"
        );
        let s = &engine.device.stream;
        let engram = engine
            .deepseek_engram
            .as_ref()
            .context("composition Engram unbound")?;
        ensure!(
            engram.prepared == Some((end - rows, rows)),
            "composition Engram preparation did not advance"
        );
        let state = engine
            .state
            .compressed
            .as_ref()
            .context("composition compressed state unbound")?;
        let generation = s.clone_dtoh(&state.generation)?[0];
        for kv in &state.kv {
            let count = s.clone_dtoh(&kv.count)?[0] as usize;
            let cursor = s.clone_dtoh(&kv.cursor)?[0] as usize;
            let ticket = s.clone_dtoh(&kv.ticket)?;
            ensure!(
                count == end / kv.ratio
                    && cursor == end
                    && ticket == [kv.owner as u32, generation, end as u32, count as u32],
                "composition KV owner {} did not publish live state at {end}",
                kv.owner
            );
        }
        let selection_rows = engine
            .model_spec
            .deepseek_v41
            .as_ref()
            .context("DSA geometry missing")?
            .selection
            .index_topk as usize;
        for selection in &state.index {
            let mapping =
                state.owners[selection.owner].context("composition owner mapping absent")?;
            let ticket = s.clone_dtoh(&selection.ticket)?;
            ensure!(
                ticket
                    == [
                        selection.owner as u32,
                        mapping.kv as u32,
                        generation,
                        (end - rows) as u32,
                        end as u32
                    ],
                "composition selection owner {} did not advance",
                selection.owner
            );
            let ids = s.clone_dtoh(
                &selection
                    .positions
                    .slice((rows - 1) * selection_rows..rows * selection_rows),
            )?;
            let count = end / mapping.ratio;
            ensure!(
                ids.iter()
                    .all(|&id| id == -1 || (id >= 0 && (id as usize) < count)),
                "composition selection references unpublished state"
            );
        }
        println!(
            "COMPOSITION_LIVE_STATE end={end} rows={rows} kv_owners={} index_owners={} engram_prepared=true native_mixer=true generic_decode=false",
            state.kv.len(),
            state.index.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "full-model composition; clean isolated-layer gates and live compressed-state binding required"]
    fn deepseek_composition_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "granted GPU window and live expert tiers required"
        );
        let root = PathBuf::from(
            std::env::var_os("D1_COMPOSITION_REFERENCE")
                .context("composition reference missing")?,
        );
        let reference: CompositionReference =
            serde_json::from_slice(&std::fs::read(root.join("composition.json"))?)?;
        ensure!(
            reference.revision == "1017"
                && reference.shape[0] == 599
                && reference.tokens.len() == 600
                && reference.rows.len() == 599
                && reference.arms == ["R1", "R2", "R3", "R4", "R5"]
                && reference.nll_path_spread.is_finite()
                && reference.nll_path_spread > 0.0
                && reference.nll_limit == 2.0 * reference.nll_path_spread
                && reference.rows.iter().all(|r| r.top1 < reference.shape[1]
                    && r.nll.len() == 5
                    && r.nll.iter().all(|v| v.is_finite())
                    && r.margin.is_finite()
                    && r.margin >= 0.0
                    && r.spread.is_finite()
                    && r.spread >= 0.0),
            "Runner FAIL composition frozen reference contract"
        );
        let model = PathBuf::from(std::env::var_os("F3_MODEL").context("F3_MODEL missing")?);
        let runtime =
            PathBuf::from(std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
        let output = PathBuf::from(
            std::env::var_os("D1_COMPOSITION_OUTPUT").context("composition output missing")?,
        );
        std::fs::create_dir(&output)?;
        let settings = crate::backend::Settings {
            tree_builder: crate::backend::TreeBuilder::Waves,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            runtime_dir: Some(runtime),
        };
        let mut engine = Engine::load(&model, 0, 640, None, settings)?;
        ensure!(
            engine.config.num_hidden_layers == 40
                && engine.deepseek_engram.is_some()
                && engine.moe_slots.is_some()
                && engine.deepseek_ffn.as_ref().is_some_and(|ffn| ffn.tiered),
            "composition engine geometry/Engram/native expert tier binding"
        );
        ensure!(
            engine.config.vocab_size == reference.shape[1],
            "composition model/reference vocabulary"
        );
        let mut score = CompositionScore::new(&root, &output.join("decode.f32"), &reference)?;
        println!("COMPOSITION_ARM singleton_decode primary=true");
        engine.prefill(&reference.tokens[..1], None)?;
        composition_live_state(&engine, 1, 1)?;
        score.row(0, &engine.decode_logits()?)?;
        for position in 1..599 {
            engine.set_token(reference.tokens[position])?;
            engine.decode(false, None)?;
            composition_live_state(&engine, position + 1, 1)?;
            score.row(position, &engine.decode_logits()?)?;
        }
        let accepted = score.finish(false)?;
        drop(score);
        engine.reset()?;
        for kv in &engine
            .state
            .compressed
            .as_ref()
            .context("composition reset state missing")?
            .kv
        {
            ensure!(
                engine.device.stream.clone_dtoh(&kv.count)? == [0]
                    && engine.device.stream.clone_dtoh(&kv.cursor)? == [0],
                "composition reset retained compressed state"
            );
        }
        let mut chunk_score =
            CompositionScore::new(&root, &output.join("chunked.f32"), &reference)?;
        println!("COMPOSITION_ARM chunked_prefill informational=true");
        engine.prefill_chunks(&reference.tokens[..599], None, 599, |e, chunk| {
            composition_live_state(e, e.position + chunk.len(), chunk.len())?;
            for row in 0..chunk.len() {
                e.deepseek_operator(
                    Operator::Head,
                    0,
                    Step {
                        rows: row + 1,
                        input: Input::Ids,
                        head: Head::Last,
                        mask: Mask::Causal,
                    },
                )?;
                e.logits_cache.mark_valid();
                chunk_score.row(e.position + row, &e.decode_logits()?)?;
            }
            Ok(())
        })?;
        let _informational = chunk_score.finish(true)?;
        ensure!(accepted, "DEFECT full-model composition");
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct SourceCell {
        owner: usize,
        position: usize,
        name: String,
        #[serde(default)]
        kind: String,
        input: Vec<u16>,
        #[serde(alias = "width")]
        k: usize,
        #[serde(default)]
        n: usize,
        output: Vec<f32>,
        #[serde(default)]
        epsilon: f32,
        spread: f64,
        analytic: f64,
        bound: f64,
        #[serde(default)]
        act_codes: Vec<u8>,
        #[serde(default)]
        act_scales: Vec<u8>,
    }

    #[derive(serde::Deserialize)]
    struct SourceRotary {
        owner: usize,
        position: usize,
        kind: String,
        heads: usize,
        width: usize,
        input: Vec<u16>,
        output: Vec<u16>,
        rotary: Vec<f32>,
        spread: f64,
        analytic: f64,
        bound: f64,
    }

    struct SourceOps {
        device: Device,
        blas: Blas,
        common: CommonOpsRef,
        latent: LatentWindowAttention,
        weights: BTreeMap<String, Vec<u8>>,
    }

    impl SourceOps {
        fn reduction_proof(&self, cell: &SourceCell, column: usize, got: f32) -> Result<bool> {
            if cell.kind.is_empty() {
                return Ok(false);
            }
            let weight = self.weight(cell, "weight")?;
            let mut exact = 0.0f64;
            let mut absolute = 0.0f64;
            for i in 0..cell.k {
                let (a, b) = if cell.kind == "fp8" {
                    ensure!(
                        cell.act_codes.len() == cell.k && cell.act_scales.len() == cell.k / 32,
                        "source FP8 recorded activation shape"
                    );
                    let scales = self.weight(cell, "scale")?;
                    let a = f64::from(crate::fp8::decode_e4m3(cell.act_codes[i]))
                        * 2.0f64.powi(i32::from(cell.act_scales[i / 32]) - 127);
                    let b = f64::from(crate::fp8::decode_e4m3(weight[column * cell.k + i]))
                        * 2.0f64
                            .powi(i32::from(scales[column / 32 * (cell.k / 32) + i / 32]) - 127);
                    (a, b)
                } else {
                    let offset = (column * cell.k + i) * 2;
                    (
                        f64::from(bf16::from_bits(cell.input[i]).to_f32()),
                        f64::from(
                            bf16::from_bits(u16::from_le_bytes([
                                weight[offset],
                                weight[offset + 1],
                            ]))
                            .to_f32(),
                        ),
                    )
                };
                exact += a * b;
                absolute += (a * b).abs();
            }
            let gamma = cell.k as f64 * 2.0f64.powi(-24) / (1.0 - cell.k as f64 * 2.0f64.powi(-24));
            let radius = gamma * absolute;
            let store = |x: f64| {
                if cell.kind == "f32_output" {
                    f64::from(x as f32)
                } else {
                    f64::from(bf16::from_f64(x).to_f32())
                }
            };
            let interval = [store(exact - radius), store(exact + radius)];
            let want = cell.output[column];
            let proven = [got, want]
                .iter()
                .all(|&v| interval[0] <= f64::from(v) && f64::from(v) <= interval[1]);
            println!(
                "SOURCE_STORE_PROOF owner={} position={} op={} column={column} n={} exact={exact:e} sum_abs={absolute:e} gamma={gamma:e} interval={interval:?} candidate={got:e} publisher={want:e} exempt={proven}",
                cell.owner, cell.position, cell.name, cell.k
            );
            Ok(proven)
        }

        fn weight(&self, cell: &SourceCell, suffix: &str) -> Result<&[u8]> {
            self.weights
                .get(&format!("{}.{suffix}", cell.name))
                .map(Vec::as_slice)
                .context("source operator weight absent")
        }

        fn execute(
            &self,
            cell: &SourceCell,
            rows: usize,
            graph: bool,
            control: bool,
        ) -> Result<Vec<f32>> {
            use crate::backend::cuda::linear::{Bf16F32Plan, Fp8ScaleRef, LinearWeightRef};
            let s = &self.device.stream;
            let (k, n) = (cell.k, if cell.n == 0 { cell.k } else { cell.n });
            let mut input: Vec<_> = cell.input.iter().copied().map(bf16::from_bits).collect();
            if control {
                input.rotate_left(1);
            }
            let x = s.clone_htod(&input.repeat(rows))?;
            let xv = view(&x, s, 0, [rows as u32, k as u32], [k as u64, 1])?;
            let y = s.clone_htod(&vec![bf16::from_f32(127.0); rows * n + 16])?;
            let yf = s.clone_htod(&vec![127.0f32; rows * n + 16])?;
            let mut yv = TensorMut(view(&y, s, 0, [rows as u32, n as u32], [n as u64, 1])?);
            let raw = self.weight(cell, "weight")?;
            let w = s.clone_htod(raw)?;
            let weight_pointer = w.device_ptr(s).0;
            let wv = unsafe {
                Tensor::<bf16, 2>::new(
                    weight_pointer,
                    [n as u32, k as u32],
                    [k as u64, 1],
                    w.len() as u64,
                )
            };
            let float_plan = if cell.kind == "f32_output" {
                Some(Bf16F32Plan::new(
                    &self.device.ctx,
                    xv,
                    wv?,
                    view(&yf, s, 0, [rows as u32, n as u32], [n as u64, 1])?,
                )?)
            } else {
                None
            };
            let codes = s.alloc_zeros::<u8>(rows * k)?;
            let scales = s.alloc_zeros::<u8>(rows * k / 32)?;
            let mut cv = TensorMut(view(&codes, s, 0, [rows as u32, k as u32], [k as u64, 1])?);
            let mut sv = TensorMut(view(
                &scales,
                s,
                0,
                [rows as u32, (k / 32) as u32],
                [(k / 32) as u64, 1],
            )?);
            let ws = if cell.kind == "fp8" {
                Some(s.clone_htod(self.weight(cell, "scale")?)?)
            } else {
                None
            };
            let fp8 = if let Some(ws) = &ws {
                Some(self.blas.fp8_ops()?.bind_block(
                    LinearWeightRef::Fp8 {
                        codes: view(&w, s, 0, [n as u32, k as u32], [k as u64, 1])?,
                        scale: Fp8ScaleRef::E8m0(view(
                            ws,
                            s,
                            0,
                            [n.div_ceil(32) as u32, (k / 32) as u32],
                            [(k / 32) as u64, 1],
                        )?),
                        layout: crate::fp8::Fp8ScaleLayout::Block(crate::fp8::Fp8BlockScale::new(
                            [32, 32],
                            safetensors::Dtype::F8_E8M0,
                        )?),
                    },
                    cv.0,
                    Fp8ScaleRef::E8m0(sv.0),
                    &yv,
                )?)
            } else {
                None
            };
            if cell.kind == "bf16" {
                self.blas
                    .prepare_skinny(crate::backend::setup::LinearShape {
                        rows,
                        output: n,
                        input: k,
                    })?;
            }
            let end = i32::try_from(cell.position + rows)?;
            let meta = s.clone_htod(&[cell.position as i32, rows as i32, end, end, 1, 0, 0, 0])?;
            let seq = SeqRef {
                ptr: meta.device_ptr(s).0,
            };
            let mut launch = || -> Result<()> {
                if let Some(plan) = &float_plan {
                    return plan.run(s);
                }
                if let Some(plan) = &fp8 {
                    self.latent.act_quant(s, xv, &mut cv, &mut sv)?;
                    return plan.run(s);
                }
                if cell.kind == "bf16" {
                    return self.blas.linear_typed_bound(
                        crate::op_contract::ExecutionPhase::Prefill,
                        LinearWeightRef::Bf16(unsafe {
                            Tensor::new(
                                weight_pointer,
                                [n as u32, k as u32],
                                [k as u64, 1],
                                w.len() as u64,
                            )?
                        }),
                        xv,
                        &mut yv,
                        16,
                    );
                }
                ensure!(
                    cell.kind.is_empty() && cell.epsilon > 0.0,
                    "source norm contract"
                );
                self.common.norm.rms_with(
                    s,
                    RowDomain {
                        meta: seq,
                        row_base: 0,
                        row_capacity: rows as u32,
                    },
                    unsafe {
                        Tensor::new(
                            xv.ptr,
                            [rows as u32, 1, k as u32],
                            [k as u64, k as u64, 1],
                            xv.span_bytes,
                        )?
                    },
                    unsafe { Tensor::new(weight_pointer, [k as u32], [1], w.len() as u64)? },
                    &mut TensorMut(unsafe {
                        Tensor::new(
                            yv.0.ptr,
                            [rows as u32, 1, k as u32],
                            [k as u64, k as u64, 1],
                            yv.0.span_bytes,
                        )?
                    }),
                    RmsOptions {
                        mode: RmsMode::Unrounded,
                        eps: cell.epsilon,
                    },
                )
            };
            launch()?;
            if fp8.is_some() && !control {
                ensure!(
                    s.clone_dtoh(&codes)? == cell.act_codes.repeat(rows)
                        && s.clone_dtoh(&scales)? == cell.act_scales.repeat(rows),
                    "source activation codec DEFECT {}",
                    cell.name
                );
            }
            if graph {
                self.device.ctx.synchronize()?;
                s.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                launch()?;
                s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("source operator graph empty")?
                    .launch()?;
            }
            let out = if float_plan.is_some() {
                s.clone_dtoh(&yf)?
            } else {
                s.clone_dtoh(&y)?.into_iter().map(bf16::to_f32).collect()
            };
            ensure!(
                out.iter().all(|v| v.is_finite()) && out[rows * n..].iter().all(|&v| v == 127.0),
                "source operator finite/guard"
            );
            Ok(out[..rows * n].to_vec())
        }
    }

    #[test]
    #[ignore = "requires DeepSeek published weights and frozen source-operator reference"]
    fn deepseek_compressor_indexer_sources_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        #[derive(serde::Deserialize)]
        struct Fixture {
            reference_only: bool,
            candidate_seen: bool,
            projections: Vec<SourceCell>,
            norms: Vec<SourceCell>,
            rotaries: Vec<SourceRotary>,
        }
        let fixture: Fixture =
            serde_json::from_slice(&std::fs::read(std::env::var("D1_SOURCE_REFERENCE")?)?)?;
        ensure!(
            fixture.reference_only
                && !fixture.candidate_seen
                && fixture.projections.len() == 81
                && fixture.norms.len() == 16,
            "source reference scope"
        );
        let mapped =
            crate::checkpoint::MappedCheckpoint::open(&PathBuf::from(std::env::var("F3_MODEL")?))?;
        let views = mapped.views()?;
        let mut weights = BTreeMap::new();
        for cell in fixture.projections.iter().chain(&fixture.norms) {
            for suffix in if cell.kind == "fp8" {
                &['w', 's'][..]
            } else {
                &['w'][..]
            } {
                let name = format!(
                    "{}.{}",
                    cell.name,
                    if *suffix == 'w' { "weight" } else { "scale" }
                );
                if weights.contains_key(&name) {
                    continue;
                }
                let (shard, tensor) = views
                    .iter()
                    .find_map(|(shard, tensors)| tensors.tensor(&name).ok().map(|v| (shard, v)))
                    .context("source weight missing")?;
                let mut data = vec![0u8; tensor.data().len()];
                mapped.read_mapped_range(shard, tensor.data(), &mut data)?;
                let dtype = if *suffix == 's' {
                    safetensors::Dtype::F8_E8M0
                } else if cell.kind == "fp8" {
                    safetensors::Dtype::F8_E4M3
                } else {
                    safetensors::Dtype::BF16
                };
                ensure!(tensor.dtype() == dtype, "source weight native dtype {name}");
                let shape = if *suffix == 's' {
                    vec![cell.n.div_ceil(32), cell.k / 32]
                } else if cell.n == 0 {
                    vec![cell.k]
                } else {
                    vec![cell.n, cell.k]
                };
                ensure!(tensor.shape() == shape, "source weight native shape {name}");
                weights.insert(name, data);
            }
        }
        let device = Device::new(0)?;
        let ops =
            Ops::new_with_swiglu_rounding(&device.ctx, crate::model_spec::SwigluRounding::Twice)?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "source operator gate".into(),
        ));
        let mut blas = Blas::new(
            &device,
            &ops,
            &PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?),
            closure,
        )?;
        blas.enable_fp8(&device.ctx)?;
        let runtime = SourceOps {
            common: CommonOpsRef::new(&device.ctx)?,
            latent: LatentWindowAttention::new(&device.ctx)?,
            device,
            blas,
            weights,
        };
        let (mut defects, mut cases, mut controls) = (0usize, 0usize, 0usize);
        for cell in fixture.projections.iter().chain(&fixture.norms) {
            ensure!(
                cell.bound.is_finite()
                    && cell.bound >= 0.0
                    && cell.bound == (3.0 * cell.spread).max(cell.analytic)
                    && cell.input.len() == cell.k,
                "source frozen band/input"
            );
            let width = cell.output.len();
            ensure!(
                width == if cell.n == 0 { cell.k } else { cell.n },
                "source output shape"
            );
            for rows in [1usize, 5, 16] {
                let eager = runtime.execute(cell, rows, false, false)?;
                let graph = runtime.execute(cell, rows, true, false)?;
                let repeat_bits = eager
                    .iter()
                    .zip(&graph)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let (mut bad, mut exempt) = (0usize, 0usize);
                for (i, &value) in eager.iter().enumerate() {
                    if (f64::from(value) - f64::from(cell.output[i % width])).abs() > cell.bound {
                        if runtime.reduction_proof(cell, i % width, value)? {
                            exempt += 1;
                        } else {
                            bad += 1;
                        }
                    }
                }
                let max = eager
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (f64::from(*v) - f64::from(cell.output[i % width])).abs())
                    .fold(0.0f64, f64::max);
                defects += bad + repeat_bits;
                cases += 1;
                println!(
                    "SOURCE_OWNER owner={} position={} op={} rows={rows} defects={bad} exempt={exempt} repeat_bits={repeat_bits} max={max:e} bound={:e}",
                    cell.owner, cell.position, cell.name, cell.bound
                );
            }
            let control = runtime.execute(cell, 1, false, true)?;
            ensure!(
                control
                    .iter()
                    .zip(&cell.output)
                    .any(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() > cell.bound),
                "source shifted-input control escaped {}",
                cell.name
            );
            controls += 1;
        }
        defects += source_rotary_cases(&runtime.device, &ops, &fixture.rotaries)?;
        println!(
            "{} SOURCE_OPERATORS cases={cases} defects={defects} shifted_input_controls={controls}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(
            defects == 0 && cases == 291 && controls == 97,
            "source operators DEFECT"
        );
        Ok(())
    }

    fn source_rotary_cases(device: &Device, ops: &Ops, cells: &[SourceRotary]) -> Result<usize> {
        ensure!(cells.len() == 48, "source rotary reference coverage");
        let s = &device.stream;
        let mut defects = 0;
        for cell in cells {
            let count = cell.heads * cell.width;
            ensure!(
                cell.input.len() == count
                    && cell.output.len() == count
                    && cell.rotary.len() == 64
                    && cell.bound.is_finite()
                    && cell.bound == (3.0 * cell.spread).max(cell.analytic),
                "source rotary reference shape/band"
            );
            let want: Vec<_> = cell.output.iter().copied().map(bf16::from_bits).collect();
            for rows in [1usize, 5, 16] {
                let original: Vec<_> = cell
                    .input
                    .repeat(rows)
                    .into_iter()
                    .map(bf16::from_bits)
                    .collect();
                let mut x = s.clone_htod(&original)?;
                let rotations = s.clone_htod(&cell.rotary.repeat(rows))?;
                let tv = view(&rotations, s, 0, [rows as u32, 64], [64, 1])?;
                let mut xv = TensorMut(view(
                    &x,
                    s,
                    0,
                    [rows as u32, cell.heads as u32, cell.width as u32],
                    [count as u64, cell.width as u64, 1],
                )?);
                ops.rope_pair_tail(s, &mut xv, tv)?;
                let eager = s.clone_dtoh(&x)?;
                s.memcpy_htod(&original, &mut x)?;
                s.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                ops.rope_pair_tail(s, &mut xv, tv)?;
                let graph = s
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("source rotary graph empty")?;
                graph.launch()?;
                let got = s.clone_dtoh(&x)?;
                let repeat = eager
                    .iter()
                    .zip(&got)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let bad = got
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| {
                        (f64::from(v.to_f32()) - f64::from(want[i % count].to_f32())).abs()
                            > cell.bound
                    })
                    .count();
                let untouched = got
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| {
                        i % cell.width < cell.width - 64 && v.to_bits() != original[*i].to_bits()
                    })
                    .count();
                defects += repeat + bad + untouched;
                println!(
                    "SOURCE_ROTARY owner={} position={} kind={} rows={rows} beyond={bad} repeat_bits={repeat} untouched_bits={untouched} bound={:e}",
                    cell.owner, cell.position, cell.kind, cell.bound
                );
                if rows == 1 && cell.position != 0 {
                    let mut wrong = cell.rotary.clone();
                    for pair in wrong.as_chunks_mut::<2>().0 {
                        pair[1] = -pair[1];
                    }
                    let wrong = s.clone_htod(&wrong)?;
                    s.memcpy_htod(&original, &mut x)?;
                    ops.rope_pair_tail(s, &mut xv, view(&wrong, s, 0, [1, 64], [64, 1])?)?;
                    let mutant = s.clone_dtoh(&x)?;
                    ensure!(
                        mutant.iter().zip(&want).any(|(a, b)| (f64::from(a.to_f32())
                            - f64::from(b.to_f32()))
                        .abs()
                            > cell.bound),
                        "source rotary inverse control escaped"
                    );
                }
            }
        }
        println!(
            "{} SOURCE_ROTARY cells={} executions={} defects={defects}",
            if defects == 0 { "PASS" } else { "DEFECT" },
            cells.len(),
            cells.len() * 3
        );
        Ok(defects)
    }

    #[derive(serde::Deserialize)]
    struct EngramCell {
        key: String,
        layer: usize,
        position: usize,
        ids: Vec<u32>,
        text_mask: Vec<bool>,
        hashes: Vec<u64>,
        input: Vec<i16>,
        output: Vec<i16>,
        projection: Vec<i16>,
    }

    #[derive(serde::Deserialize)]
    struct EngramBand {
        key: String,
        spread: f64,
        analytic: f64,
        bound: f64,
    }

    fn engram_distance(actual: &[bf16], expected: &[bf16], bound: f64) -> Result<(usize, f64)> {
        ensure!(
            actual.len() == expected.len(),
            "Engram boundary extent differs"
        );
        let mut beyond = 0;
        let mut maximum = 0.0f64;
        for (a, b) in actual.iter().zip(expected) {
            ensure!(a.is_finite() && b.is_finite(), "nonfinite Engram boundary");
            let delta = (f64::from(a.to_f32()) - f64::from(b.to_f32())).abs();
            beyond += usize::from(delta > bound);
            maximum = maximum.max(delta);
        }
        Ok((beyond, maximum))
    }

    #[derive(serde::Deserialize)]
    struct EngramOperands {
        key: String,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    fn engram_projection_proof(
        s: &Arc<CudaStream>,
        runtime: &DeepseekEngram,
        cell: &EngramCell,
        operands: &EngramOperands,
    ) -> Result<usize> {
        let k = runtime.width * crate::engram::ENGRAM_HEAD_DIM;
        let codes = s.clone_dtoh(&runtime.codes.slice(..k))?;
        let scales = s.clone_dtoh(&runtime.scales.slice(..k / 32))?;
        ensure!(
            operands.codes.len() == k && operands.scales.len() == k / 32,
            "Engram operand extent"
        );
        let different = codes
            .iter()
            .zip(&operands.codes)
            .filter(|(a, b)| a != b)
            .count()
            + scales
                .iter()
                .zip(&operands.scales)
                .filter(|(a, b)| a != b)
                .count();
        if different > 0 {
            println!(
                "DEFECT ENGRAM_PREPARE key={} operand_bits={different}",
                cell.key
            );
            return Ok(different);
        }
        let weights = &runtime
            .layers
            .iter()
            .find(|v| v.id == cell.layer)
            .and_then(|v| v.weights.as_ref())
            .context("Engram projection weights")?
            .projection;
        let actual = s.clone_dtoh(&runtime.projection.slice(..cell.projection.len()))?;
        let eps = f64::from(f32::EPSILON) * 0.5;
        let operations = k as f64;
        let gamma = operations * eps / (1.0 - operations * eps);
        let mut defects = 0;
        let mut exemptions = 0;
        for (row, (&a, &bits)) in actual.iter().zip(&cell.projection).enumerate() {
            let b = bf16::from_bits(bits as u16);
            if a.to_bits() == b.to_bits() {
                continue;
            }
            let wc = s.clone_dtoh(&weights.codes.slice(row * k..(row + 1) * k))?;
            let ws = s.clone_dtoh(
                &weights
                    .scale
                    .slice((row / 32) * (k / 32)..(row / 32 + 1) * (k / 32)),
            )?;
            let mut sum = 0.0f64;
            let mut absolute = 0.0f64;
            for i in 0..k {
                let term = f64::from(crate::fp8::decode_e4m3(codes[i]))
                    * mix_e8m0(scales[i / 32])
                    * f64::from(crate::fp8::decode_e4m3(wc[i]))
                    * mix_e8m0(ws[i / 32]);
                sum += term;
                absolute += term.abs();
            }
            let bound = (gamma + operations * f64::EPSILON) * absolute;
            let low = f64::from(bf16::from_f64(sum - bound).to_f32());
            let high = f64::from(bf16::from_f64(sum + bound).to_f32());
            let candidate = f64::from(a.to_f32());
            let publisher = f64::from(b.to_f32());
            let exempt =
                low <= candidate && candidate <= high && low <= publisher && publisher <= high;
            defects += usize::from(!exempt);
            exemptions += usize::from(exempt);
            println!(
                "ENGRAM_PROJECTION_PROOF key={} column={row} n={k} gamma={gamma:e} sum_abs={absolute:e} exact={sum:e} radius={bound:e} bf16_interval=[{low:e},{high:e}] candidate={candidate:e} publisher={publisher:e} exempt={exempt}",
                cell.key
            );
        }
        println!(
            "ENGRAM_PROJECTION key={} operand_bits=0 exempt={exemptions} defects={defects}",
            cell.key
        );
        Ok(defects)
    }

    fn engram_store_proof(
        s: &Arc<CudaStream>,
        runtime: &DeepseekEngram,
        cell: &EngramCell,
        actual: &[bf16],
        expected: &[bf16],
        bound: f64,
        norm_eps: f32,
    ) -> Result<usize> {
        let projection = s.clone_dtoh(&runtime.projection.slice(..cell.projection.len()))?;
        let projection_bits = projection
            .iter()
            .zip(&cell.projection)
            .filter(|(a, b)| a.to_bits() != **b as u16)
            .count();
        println!(
            "ENGRAM_ISOLATION key={} projection_bits={projection_bits}",
            cell.key
        );
        if projection_bits != 0 {
            return Ok(0);
        }
        let weights = runtime
            .layers
            .iter()
            .find(|v| v.id == cell.layer)
            .and_then(|v| v.weights.as_ref())
            .context("Engram proof weights")?;
        let q = s.clone_dtoh(&weights.q)?;
        let k = s.clone_dtoh(&weights.k)?;
        let gates = s.clone_dtoh(&runtime.gates.slice(..runtime.copies))?;
        let width = runtime.hidden;
        let eps = f64::from(f32::EPSILON) / 2.0;
        // 2026-10-29: post-candidate rule B includes cancellation stores inside the FP32 reduction interval.
        let n = width as f64;
        let gamma = n * eps / (1.0 - n * eps);
        let mut intervals = Vec::new();
        for (copy, &observed_gate) in gates.iter().enumerate() {
            let (mut h2, mut k2, mut dot, mut absolute) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for d in 0..width {
                let i = copy * width + d;
                let h = f64::from(bf16::from_bits(cell.input[i] as u16).to_f32());
                let key = f64::from(projection[i].to_f32());
                let term = h * f64::from(q[i].to_f32()) * f64::from(k[i].to_f32()) * key;
                h2 += h * h;
                k2 += key * key;
                dot += term;
                absolute += term.abs();
            }
            let norm = |sum: f64, delta: f64| {
                (sum * (1.0 + delta) / (width as f64) + f64::from(norm_eps)).sqrt()
            };
            let denominators = [
                norm(h2, -gamma) * norm(k2, -gamma),
                norm(h2, gamma) * norm(k2, gamma),
            ];
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for numerator in [dot - gamma * absolute, dot + gamma * absolute] {
                for denominator in denominators {
                    let value = numerator / denominator / (width as f64).sqrt();
                    lo = lo.min(value);
                    hi = hi.max(value);
                }
            }
            let sigmoid = |v: f64| 1.0 / (1.0 + (-v.abs().max(1e-6).sqrt().copysign(v)).exp());
            let low = sigmoid(lo);
            let high = sigmoid(hi);
            if !(f64::from(observed_gate) >= low && f64::from(observed_gate) <= high) {
                println!(
                    "DEFECT Engram gate arithmetic outside reduction bound key={} copy={copy} actual={observed_gate:e} range=[{low:e},{high:e}]",
                    cell.key
                );
                return Ok(0);
            }
            let exact_dot = dot
                / ((h2 / (width as f64) + f64::from(norm_eps)).sqrt()
                    * (k2 / (width as f64) + f64::from(norm_eps)).sqrt())
                / (width as f64).sqrt();
            let exact_gate = sigmoid(exact_dot);
            println!(
                "ENGRAM_GATE_PROOF key={} copy={copy} n={width} gamma={gamma:e} dot_sum_abs={absolute:e} dot_exact={dot:e} gate_exact={exact_gate:e} gate_interval=[{low:e},{high:e}]",
                cell.key
            );
            intervals.push((low, high, exact_gate));
        }
        let mut exempt = 0;
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            if (f64::from(a.to_f32()) - f64::from(b.to_f32())).abs() <= bound {
                continue;
            }
            let copy = i / width;
            let d = i % width;
            let h = f64::from(bf16::from_bits(cell.input[i] as u16).to_f32());
            let value = f64::from(projection[runtime.copies * width + d].to_f32());
            let (glo, ghi, exact_gate) = intervals[copy];
            let endpoints = [h + glo * value, h + ghi * value];
            let rounding = 2.0 * eps / (1.0 - 2.0 * eps) * (h.abs() + ghi * value.abs());
            let exact = h + exact_gate * value;
            let radius = (exact - endpoints[0])
                .abs()
                .max((exact - endpoints[1]).abs())
                + rounding;
            let low = f64::from(bf16::from_f64(exact - radius).to_f32());
            let high = f64::from(bf16::from_f64(exact + radius).to_f32());
            let candidate = f64::from(a.to_f32());
            let publisher = f64::from(b.to_f32());
            let proven =
                low <= candidate && candidate <= high && low <= publisher && publisher <= high;
            exempt += usize::from(proven);
            println!(
                "ENGRAM_STORE_PROOF key={} element={i} n={width} gamma={gamma:e} exact={exact:e} propagated_radius={radius:e} bf16_interval=[{low:e},{high:e}] candidate={candidate:e} publisher={publisher:e} exempt={proven}",
                cell.key
            );
        }
        Ok(exempt)
    }

    fn engram_gate_case(
        device: &Device,
        runtime: &mut DeepseekEngram,
        hc: &mut HcRuntime,
        state: &mut super::super::super::state::StateStore,
        cell: &EngramCell,
        bound: f64,
        operands: &EngramOperands,
    ) -> Result<(usize, [usize; 3])> {
        let s = &device.stream;
        let input: Vec<_> = cell
            .input
            .iter()
            .map(|&v| bf16::from_bits(v as u16))
            .collect();
        let expected: Vec<_> = cell
            .output
            .iter()
            .map(|&v| bf16::from_bits(v as u16))
            .collect();
        ensure!(
            input.len() == runtime.copies * runtime.hidden
                && expected.len() == input.len()
                && cell.ids.len() == cell.text_mask.len()
                && cell.position < cell.ids.len(),
            "Engram fixture extent"
        );
        runtime.service.reset();
        for start in (0..cell.position).step_by(runtime.rows) {
            let end = (start + runtime.rows).min(cell.position);
            runtime.prepare(
                s,
                &cell.ids[start..end],
                start,
                Some(&cell.text_mask[start..end]),
            )?;
        }
        let p = cell.position;
        runtime.prepare(s, &cell.ids[p..p + 1], p, Some(&cell.text_mask[p..p + 1]))?;
        let (ticket, _) = runtime.service.prepare(
            &cell.ids[p..p + 1],
            p as u32,
            Some(&cell.text_mask[p..p + 1]),
        )?;
        ensure!(
            ticket.take(cell.layer as u32)?.hashes == cell.hashes,
            "Engram publisher hash IDs differ"
        );
        let index = runtime
            .layers
            .iter()
            .position(|v| v.id == cell.layer)
            .context("Engram fixture layer")?;
        let record_size = runtime.width * crate::engram::ENGRAM_NATIVE_ROW_BYTES;
        let records = s.clone_dtoh(&runtime.layers[index].records.slice(..record_size))?;
        let mut defects = 0;
        for rows in [1, 5] {
            state.set_sequence(s, p as u32, rows as u32, runtime.rows as u32)?;
            s.memcpy_htod(
                &records.repeat(rows),
                &mut runtime.layers[index]
                    .records
                    .slice_mut(..record_size * rows),
            )?;
            s.memcpy_htod(
                &vec![u8::from(cell.text_mask[p]); rows],
                &mut runtime.mask.slice_mut(..rows),
            )?;
            for graph in [false, true] {
                s.memcpy_htod(
                    &input.repeat(rows),
                    &mut hc.stream.slice_mut(..input.len() * rows),
                )?;
                if graph {
                    s.synchronize()?;
                    s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let launched = runtime.run(s, cell.layer, rows, hc);
                    let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
                    launched?;
                    captured?.context("empty Engram graph")?.launch()?;
                } else {
                    runtime.run(s, cell.layer, rows, hc)?;
                }
                let actual = s.clone_dtoh(&hc.stream.slice(..input.len() * rows))?;
                ensure!(
                    actual
                        .chunks_exact(input.len())
                        .all(|row| row == &actual[..input.len()]),
                    "Engram identical rows differ within one call"
                );
                let propagated = engram_distance(&actual, &expected.repeat(rows), bound)?.0;
                defects += engram_projection_proof(s, runtime, cell, operands)?;
                let projected: Vec<_> = cell
                    .projection
                    .iter()
                    .map(|&v| bf16::from_bits(v as u16))
                    .collect();
                s.memcpy_htod(
                    &projected.repeat(rows),
                    &mut runtime.projection.slice_mut(..projected.len() * rows),
                )?;
                s.memcpy_htod(
                    &input.repeat(rows),
                    &mut hc.stream.slice_mut(..input.len() * rows),
                )?;
                let mut execute = || -> Result<()> {
                    runtime.layers[index].plans[rows - 1].1.run(s)?;
                    s.memcpy_dtod(
                        &runtime.output.slice(..input.len() * rows),
                        &mut hc.stream.slice_mut(..input.len() * rows),
                    )?;
                    Ok(())
                };
                if graph {
                    s.synchronize()?;
                    s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let launched = execute();
                    let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
                    launched?;
                    captured?.context("empty isolated Engram graph")?.launch()?;
                } else {
                    execute()?;
                }
                let actual = s.clone_dtoh(&hc.stream.slice(..input.len() * rows))?;
                let (beyond, max) = engram_distance(&actual, &expected.repeat(rows), bound)?;
                let exempt = if beyond > 0 {
                    let per_row = engram_store_proof(
                        s,
                        runtime,
                        cell,
                        &actual[..input.len()],
                        &expected,
                        bound,
                        hc.binding.params.norm_eps,
                    )?;
                    per_row * rows
                } else {
                    0
                };
                defects += beyond - exempt;
                if !cell.text_mask[p] {
                    ensure!(
                        actual == input.repeat(rows),
                        "Engram image-mask changed the residual bits"
                    );
                }
                println!(
                    "ENGRAM_BOUNDARY key={} rows={rows} graph={graph} max={max:e} bound={bound:e} beyond={beyond} exempt={exempt} propagated={propagated} isolation=all_upstream",
                    cell.key
                );
            }
        }
        state.set_sequence(s, p as u32, 1, runtime.rows as u32)?;
        let mut controls = [0; 3];
        if cell.text_mask[p] {
            let mut rotated = records.clone();
            rotated.rotate_left(crate::engram::ENGRAM_NATIVE_ROW_BYTES);
            for (control, count) in controls.iter_mut().enumerate() {
                s.memcpy_htod(&input, &mut hc.stream.slice_mut(..input.len()))?;
                s.memcpy_htod(
                    if control == 0 { &rotated } else { &records },
                    &mut runtime.layers[index].records.slice_mut(..record_size),
                )?;
                s.memcpy_htod(&[u8::from(control != 1)], &mut runtime.mask.slice_mut(..1))?;
                if control != 2 {
                    runtime.run(s, cell.layer, 1, hc)?;
                }
                let actual = s.clone_dtoh(&hc.stream.slice(..input.len()))?;
                *count = engram_distance(&actual, &expected, bound)?.0;
            }
        }
        println!(
            "ENGRAM_CONTROLS key={} rotated={} inverted_mask={} skipped={}",
            cell.key, controls[0], controls[1], controls[2]
        );
        Ok((defects, controls))
    }

    #[test]
    #[ignore = "requires the DeepSeek checkpoint, runtime hash record and frozen Engram reference"]
    fn deepseek_engram_matches_model_definition() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let model = PathBuf::from(std::env::var("F3_MODEL")?);
        let fixture = PathBuf::from(std::env::var("F3_FIXTURE")?);
        #[derive(serde::Deserialize)]
        struct Fixture {
            cells: Vec<EngramCell>,
        }
        #[derive(serde::Deserialize)]
        struct Operands {
            reference_only: bool,
            candidate_values_read: bool,
            cells: Vec<EngramOperands>,
        }
        #[derive(serde::Deserialize)]
        struct Freeze {
            reference_only: bool,
            candidate_seen: bool,
            bands: Vec<EngramBand>,
        }
        let cells: Fixture = serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
        let freeze: Freeze =
            serde_json::from_slice(&std::fs::read(fixture.join("thresholds-engram-r1.json"))?)?;
        ensure!(
            freeze.reference_only
                && !freeze.candidate_seen
                && cells.cells.len() == 32
                && freeze.bands.len() == 32,
            "Engram reference freeze scope"
        );
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        };
        let spec = ModelSpec::read(&model)?;
        let operands: Operands =
            serde_json::from_slice(&std::fs::read(fixture.join("operands.json"))?)?;
        ensure!(
            operands.reference_only
                && !operands.candidate_values_read
                && operands.cells.len() == 32,
            "Engram operand provenance"
        );
        let mapped = crate::checkpoint::MappedCheckpoint::open(&model)?;
        let manifest = mapped.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let mut runtime = DeepseekEngram::new(&device, &spec, &manifest, &model, &settings, 272)?;
        let mut hc = HcRuntime::new(
            &device.ctx,
            s,
            deepseek_hc_binding(&spec)?,
            40,
            runtime.rows,
        )?;
        let mut state = super::super::super::state::StateStore::new(&spec, 272, s)?;
        super::super::super::weights::checkpoint_mapped(
            &mapped,
            &device,
            Some(&manifest),
            false,
            |loader| {
                runtime.load_weights(loader)?;
                for (name, entry) in &manifest.entries {
                    if entry.load
                        && !(name.contains(".engram.") && !name.contains(".engram.embed."))
                    {
                        loader.defer(name)?;
                    }
                }
                Ok(())
            },
        )?;
        runtime.bind(&device, &hc, state.seq_ref(s)?, spec.rms_eps)?;
        let mut defects = 0;
        let mut controls = [[0usize; 3]; 2];
        for cell in &cells.cells {
            let band = freeze
                .bands
                .iter()
                .find(|v| v.key == cell.key)
                .context("missing Engram frozen band")?;
            ensure!(
                band.bound.is_finite()
                    && band.bound >= 0.0
                    && band.bound == (3.0 * band.spread).max(band.analytic),
                "invalid Engram band"
            );
            let (bad, killed) = engram_gate_case(
                &device,
                &mut runtime,
                &mut hc,
                &mut state,
                cell,
                band.bound,
                operands
                    .cells
                    .iter()
                    .find(|v| v.key == cell.key)
                    .context("missing Engram recorded operands")?,
            )?;
            defects += bad;
            let layer = usize::from(cell.layer == 14);
            for (total, count) in controls[layer].iter_mut().zip(killed) {
                *total += count;
            }
        }
        ensure!(
            controls.iter().flatten().all(|&v| v > 0),
            "Engram control escaped: {controls:?}"
        );
        println!(
            "{} deepseek_engram_matches_model_definition cells=32 executions=128 defects={defects} controls={controls:?} image_mask=bit_exact",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(
            defects == 0,
            "Engram post-residual boundary defects={defects}"
        );
        Ok(())
    }

    /// Golden-ratio stride: the phase sequence never repeats (the mhc gate idiom).
    /// FP32-vs-FP64 accumulation over at most 2*mult terms of magnitude <= 2.
    const OUTPUT_ABS_TOL: f64 = 1e-6;
    const PRE_ABS_TOL: f64 = 1e-5;
    const POST_ABS_TOL: f64 = 1e-5;
    const COMB_ABS_TOL: f64 = 1e-4;
    const OUTPUT_ULP_TOL: u32 = 1;

    fn pattern(seed: usize, phase: f64) -> f64 {
        (seed as f64 * 0.6180339887498949 + phase).sin() * 0.75
    }

    fn sigmoid(value: f64) -> f64 {
        1.0 / (1.0 + (-value).exp())
    }

    /// Distance in representable BF16 values between two outputs.
    fn bf16_ulp_distance(a: bf16, b: bf16) -> u32 {
        fn key(value: bf16) -> u32 {
            let bits = u32::from(value.to_bits());
            if bits & 0x8000 != 0 {
                (!bits) & 0xffff
            } else {
                bits | 0x8000
            }
        }
        key(a).abs_diff(key(b))
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Mixes {
        pre: Vec<f64>,
        post: Vec<f64>,
        comb: Vec<f64>,
    }

    /// FP64 reference of the Fp32-rounding mode (the DeepSeek-V4.1 mode).
    struct Oracle {
        mult: usize,
        hidden: usize,
        iters: usize,
        eps: f64,
        norm_eps: f64,
    }

    impl Oracle {
        fn normalize_rows(matrix: &mut [Vec<f64>], eps: f64) {
            for row in matrix.iter_mut() {
                let sum: f64 = row.iter().sum();
                for value in row.iter_mut() {
                    *value /= sum + eps;
                }
            }
        }

        fn normalize_columns(matrix: &mut [Vec<f64>], eps: f64) {
            let mut columns: Vec<Vec<f64>> = (0..matrix.len())
                .map(|k| matrix.iter().map(|row| row[k]).collect())
                .collect();
            for column in &mut columns {
                let sum: f64 = column.iter().sum();
                for value in column.iter_mut() {
                    *value /= sum + eps;
                }
            }
            for (j, row) in matrix.iter_mut().enumerate() {
                for (k, value) in row.iter_mut().enumerate() {
                    *value = columns[k][j];
                }
            }
        }

        fn mixes(&self, stream: &[bf16], fn_w: &[f32], base: &[f32], scale: &[f32]) -> Mixes {
            let n = self.mult * self.hidden;
            let x: Vec<f64> = stream.iter().map(|v| f64::from(v.to_f32())).collect();
            let sumsq: f64 = x.iter().map(|v| v * v).sum();
            let rsqrt = 1.0 / (sumsq / n as f64 + self.norm_eps).sqrt();
            let rows = (2 + self.mult) * self.mult;
            let mut m = vec![0.0f64; rows];
            for (c, value) in m.iter_mut().enumerate() {
                let row = &fn_w[c * n..(c + 1) * n];
                let acc: f64 = row.iter().zip(&x).map(|(w, v)| f64::from(*w) * v).sum();
                *value = acc * rsqrt;
            }
            let pre: Vec<f64> = (0..self.mult)
                .map(|j| sigmoid(m[j] * f64::from(scale[0]) + f64::from(base[j])) + self.eps)
                .collect();
            let post: Vec<f64> = (0..self.mult)
                .map(|j| {
                    2.0 * sigmoid(
                        m[self.mult + j] * f64::from(scale[1]) + f64::from(base[self.mult + j]),
                    )
                })
                .collect();
            let mut matrix: Vec<Vec<f64>> = (0..self.mult)
                .map(|j| {
                    (0..self.mult)
                        .map(|k| {
                            let c = 2 * self.mult + j * self.mult + k;
                            m[c] * f64::from(scale[2]) + f64::from(base[c])
                        })
                        .collect()
                })
                .collect();
            for row in &mut matrix {
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                for value in row.iter_mut() {
                    *value = *value / sum + self.eps;
                }
            }
            Self::normalize_columns(&mut matrix, self.eps);
            for _ in 1..self.iters {
                Self::normalize_rows(&mut matrix, self.eps);
                Self::normalize_columns(&mut matrix, self.eps);
            }
            Mixes {
                pre,
                post,
                comb: matrix.into_iter().flatten().collect(),
            }
        }

        fn collapse(&self, stream: &[bf16], pre: &[f64]) -> Vec<bf16> {
            let mut out = vec![bf16::ZERO; self.hidden];
            for (h, value) in out.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (j, p) in pre.iter().enumerate() {
                    acc += p * f64::from(stream[j * self.hidden + h].to_f32());
                }
                *value = bf16::from_f32(acc as f32);
            }
            out
        }

        fn update(
            &self,
            sublayer: &[bf16],
            residual: &[bf16],
            post: &[f64],
            comb: &[f64],
        ) -> Vec<bf16> {
            let mut out = vec![bf16::ZERO; self.mult * self.hidden];
            for j in 0..self.mult {
                for h in 0..self.hidden {
                    let mut acc = post[j] * f64::from(sublayer[h].to_f32());
                    for k in 0..self.mult {
                        // model.py Block.hc_post: the stored row index selects the residual
                        // copy, so out[j] sums comb[k,j]*residual[k].
                        acc += comb[k * self.mult + j]
                            * f64::from(residual[k * self.hidden + h].to_f32());
                    }
                    out[j * self.hidden + h] = bf16::from_f32(acc as f32);
                }
            }
            out
        }
    }

    struct HcFixture {
        fn_w: Vec<f32>,
        base: Vec<f32>,
        scale: Vec<f32>,
        out: Vec<bf16>,
    }

    fn fixture(seed: usize, mult: usize, hidden: usize, rows: usize) -> HcFixture {
        let mix = (2 + mult) * mult;
        let fn_w: Vec<f32> = (0..mix * mult * hidden)
            .map(|i| (pattern(seed * 31 + i, 1.7) * 0.1) as f32)
            .collect();
        let base: Vec<f32> = (0..mix)
            .map(|i| (pattern(seed * 17 + i, 2.3) * 0.5) as f32)
            .collect();
        let scale: Vec<f32> = vec![0.5, 0.75, 1.25];
        let out: Vec<bf16> = (0..rows * hidden)
            .map(|i| bf16::from_f32(pattern(seed * 13 + i, 3.9) as f32))
            .collect();
        HcFixture {
            fn_w,
            base,
            scale,
            out,
        }
    }

    fn upload_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn uploaded_sublayer(
        fx: &HcFixture,
        upload: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<HcSublayerWeights> {
        Ok(HcSublayerWeights {
            fn_w: testing::upload(upload, upload_f32(&fx.fn_w))?,
            base: testing::upload(upload, upload_f32(&fx.base))?,
            scale: testing::upload(upload, upload_f32(&fx.scale))?,
        })
    }

    #[test]
    fn deepseek_hc_binding_pins_the_shared_runtime_semantics() {
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))
        .unwrap();
        let spec =
            crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null).unwrap();
        let binding = deepseek_hc_binding(&spec).unwrap();
        assert_eq!(
            binding.policy,
            hc_pre_policy(Architecture::DeepseekV41).unwrap()
        );
        assert_eq!(binding.rounding, HcRounding::Fp32);
        assert_eq!(binding.orientation, HcCombOrientation::Transposed);
        assert_eq!(binding.fn_dtype, HcFnDtype::F32);
        assert_eq!(
            binding.params,
            HcParams {
                mult: 4,
                hidden: 5120,
                iters: 20,
                eps: 1e-6,
                norm_eps: 1e-20,
            }
        );
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LayerBand {
        spread: f64,
        analytic: f64,
        bound: f64,
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LayerBands {
        layer: usize,
        positions: Vec<usize>,
        bands: std::collections::BTreeMap<String, Vec<LayerBand>>,
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LayerSample {
        position: usize,
        input: Vec<u16>,
        carried_pre: Vec<f32>,
        attn_norm: Vec<u16>,
        ffn_norm: Vec<u16>,
        router_cutoff: f32,
        router_ids: Vec<i32>,
        route_weights: Vec<f32>,
        wo_b_codes: Vec<u8>,
        wo_b_scales: Vec<u8>,
        boundaries: std::collections::BTreeMap<String, Vec<f32>>,
    }

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct LayerSamples {
        layer: usize,
        revision: String,
        compress_ratio: u8,
        window_codes: Vec<Vec<u8>>,
        window_scales: Vec<Vec<u8>>,
        window_ids: Option<Vec<Vec<i32>>>,
        shared_compress_kv: Option<Vec<Vec<u16>>>,
        shared_index_k: Option<Vec<Vec<u16>>>,
        shared_topk_idxs: Option<Vec<Vec<i32>>>,
        compressed_kv: Option<Vec<Vec<u16>>>,
        compressed_codes: Option<Vec<Vec<u8>>>,
        compressed_scales: Option<Vec<Vec<u8>>>,
        compressed_ids: Option<Vec<Vec<i32>>>,
        samples: Vec<LayerSample>,
    }

    fn layer_bf(values: &[u16]) -> Vec<bf16> {
        values.iter().map(|&v| bf16::from_bits(v)).collect()
    }

    fn f32_below(value: f64) -> f64 {
        let rounded = value as f32;
        if f64::from(rounded) <= value {
            f64::from(f32::from_bits(rounded.to_bits().wrapping_sub(1)))
        } else {
            f64::from(rounded)
        }
    }

    fn f32_above(value: f64) -> f64 {
        let rounded = value as f32;
        if f64::from(rounded) >= value {
            f64::from(f32::from_bits(rounded.to_bits().wrapping_add(1)))
        } else {
            f64::from(rounded)
        }
    }

    fn layer_expected(row: &LayerSample, name: &str, count: usize) -> Result<Vec<bf16>> {
        let values = row
            .boundaries
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        ensure!(
            values.len() == count && values.iter().all(|v| v.is_finite()),
            "invalid {name} shape/values"
        );
        let result: Vec<_> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        ensure!(
            result.iter().zip(values).all(|(a, b)| a.to_f32() == *b),
            "{name} is not BF16"
        );
        Ok(result)
    }

    fn layer_ring(
        engine: &mut Engine,
        fixture: &LayerSamples,
        position: usize,
        planted: bool,
    ) -> Result<()> {
        let s = &engine.device.stream;
        let g = MixerGeometry::new(&engine.model_spec)?;
        ensure!(
            fixture.window_codes.len() > position && fixture.window_scales.len() > position,
            "reference window rows missing"
        );
        let mut codes = vec![0u8; g.window * g.head_dim];
        let mut scales = vec![0u8; g.window * (g.head_dim / 32)];
        for p in position.saturating_sub(g.window)..position {
            ensure!(
                fixture.window_codes[p].len() == g.head_dim
                    && fixture.window_scales[p].len() == g.head_dim / 32,
                "window row extent"
            );
            let slot = p % g.window;
            codes[slot * g.head_dim..(slot + 1) * g.head_dim]
                .copy_from_slice(&fixture.window_codes[p]);
            scales[slot * g.head_dim / 32..(slot + 1) * g.head_dim / 32]
                .copy_from_slice(&fixture.window_scales[p]);
        }
        if planted {
            ensure!(position > 0, "wrong-slot control needs history");
            let slot = (position - 1) % g.window;
            codes[slot * g.head_dim..(slot + 1) * g.head_dim].fill(0x7e);
            scales[slot * g.head_dim / 32..(slot + 1) * g.head_dim / 32].fill(0x89);
        }
        let ring = engine.state.latent_ring_mut(fixture.layer)?;
        s.memcpy_htod(&codes, &mut ring.codes)?;
        s.memcpy_htod(&scales, &mut ring.scales)?;
        ensure!(
            s.clone_dtoh(&ring.codes)? == codes && s.clone_dtoh(&ring.scales)? == scales,
            "latent-ring operand isolation failed at layer {} position {position}",
            fixture.layer
        );
        if fixture.compress_ratio != 0 {
            layer_published_state(engine, fixture, position)?;
        }
        Ok(())
    }

    fn layer_published_state(
        engine: &mut Engine,
        fixture: &LayerSamples,
        position: usize,
    ) -> Result<()> {
        let s = &engine.device.stream;
        let state = engine
            .state
            .compressed
            .as_mut()
            .context("DSA state absent")?;
        let owners = state.owners[fixture.layer].context("DSA owners absent")?;
        let generation = s.clone_dtoh(&state.generation)?[0];
        let count = (position + 1) / owners.ratio;
        let kv = state
            .kv
            .iter_mut()
            .find(|v| v.owner == owners.kv)
            .context("DSA KV owner absent")?;
        let codes = fixture
            .compressed_codes
            .as_ref()
            .context("original compressed codes absent")?;
        let scales = fixture
            .compressed_scales
            .as_ref()
            .context("original compressed scales absent")?;
        let codes: Vec<_> = codes[..count].iter().flatten().copied().collect();
        let scales: Vec<_> = scales[..count].iter().flatten().copied().collect();
        if count != 0 {
            s.memcpy_htod(&codes, &mut kv.codes.slice_mut(..codes.len()))?;
            s.memcpy_htod(&scales, &mut kv.scales.slice_mut(..scales.len()))?;
            ensure!(
                s.clone_dtoh(&kv.codes.slice(..codes.len()))? == codes
                    && s.clone_dtoh(&kv.scales.slice(..scales.len()))? == scales,
                "compressed codec operand isolation failed"
            );
        }
        let kv_ticket = [
            owners.kv as u32,
            generation,
            position as u32 + 1,
            count as u32,
        ];
        s.memcpy_htod(&[count as u32], &mut kv.count)?;
        s.memcpy_htod(&[position as u32 + 1], &mut kv.cursor)?;
        s.memcpy_htod(&kv_ticket, &mut kv.ticket)?;
        let index = state
            .index
            .iter_mut()
            .find(|v| v.owner == owners.index)
            .context("DSA index owner absent")?;
        let ids = &fixture
            .compressed_ids
            .as_ref()
            .context("recorded selections absent")?[position];
        let mut selected = ids
            .iter()
            .copied()
            .filter(|&v| v >= 0)
            .map(|v| v - 129)
            .collect::<Vec<_>>();
        ensure!(
            selected.len() <= 512 && selected.iter().all(|&v| v >= 0 && (v as usize) < count),
            "recorded selection outside committed rows"
        );
        selected.sort_unstable();
        ensure!(
            selected.windows(2).all(|v| v[0] < v[1]),
            "duplicate recorded selection"
        );
        selected.resize(512, -1);
        let index_ticket = [
            owners.index as u32,
            owners.kv as u32,
            generation,
            position as u32,
            position as u32 + 1,
        ];
        s.memcpy_htod(&selected, &mut index.positions.slice_mut(..512))?;
        s.memcpy_htod(&index_ticket, &mut index.ticket)?;
        ensure!(
            s.clone_dtoh(&index.positions.slice(..512))? == selected
                && s.clone_dtoh(&index.ticket)? == index_ticket
                && s.clone_dtoh(&kv.ticket)? == kv_ticket
                && s.clone_dtoh(&kv.count)? == [count as u32],
            "DSA publication operand isolation failed"
        );
        println!(
            "LAYER_DSA_ISOLATION layer={} position={position} kv_owner={} index_owner={} committed={count} native_codec_bytes={} selection_rows=1 input_bits=0",
            fixture.layer,
            owners.kv,
            owners.index,
            codes.len() + scales.len()
        );
        Ok(())
    }

    fn layer_hc_inputs(
        engine: &mut Engine,
        residual: &[bf16],
        pre: &[f32],
        ffn: bool,
    ) -> Result<()> {
        let s = &engine.device.stream;
        let hc = engine.hc.as_mut().context("mHC runtime missing")?;
        ensure!(
            residual.len() == 20480 && pre.len() == 4,
            "mHC operand geometry"
        );
        s.memcpy_htod(residual, &mut hc.stream.slice_mut(..20480))?;
        let carry = if ffn {
            hc.attn_pre.as_mut()
        } else {
            hc.carry.as_mut()
        }
        .context("mHC carry missing")?;
        s.memcpy_htod(pre, &mut carry.slice_mut(..4))?;
        ensure!(
            s.clone_dtoh(&hc.stream.slice(..20480))? == *residual
                && s.clone_dtoh(&carry.slice(..4))? == *pre,
            "mHC operand isolation failed"
        );
        Ok(())
    }

    fn layer_post_inputs(engine: &mut Engine, row: &LayerSample, ffn: bool) -> Result<()> {
        let name = if ffn { "ffn" } else { "attn" };
        let residual = if ffn {
            layer_expected(row, "expand_attn", 20480)?
        } else {
            layer_bf(&row.input)
        };
        let output = layer_expected(row, name, 5120)?;
        let post = row
            .boundaries
            .get(&format!("{name}_post"))
            .context("recorded post missing")?;
        let comb = row
            .boundaries
            .get(&format!("{name}_comb"))
            .context("recorded comb missing")?;
        ensure!(post.len() == 4 && comb.len() == 16, "recorded mix geometry");
        let s = &engine.device.stream;
        let hc = engine.hc.as_mut().context("mHC runtime missing")?;
        s.memcpy_htod(&residual, &mut hc.stream.slice_mut(..20480))?;
        s.memcpy_htod(post, &mut hc.mix_post.slice_mut(..4))?;
        s.memcpy_htod(comb, &mut hc.mix_comb.slice_mut(..16))?;
        s.memcpy_htod(&output, &mut engine.work.out.slice_mut(..5120))?;
        ensure!(
            s.clone_dtoh(&hc.stream.slice(..20480))? == residual
                && s.clone_dtoh(&hc.mix_post.slice(..4))? == *post
                && s.clone_dtoh(&hc.mix_comb.slice(..16))? == *comb
                && s.clone_dtoh(&engine.work.out.slice(..5120))? == output,
            "full-operand isolation failed"
        );
        Ok(())
    }

    fn layer_store_proof(
        engine: &Engine,
        row: &LayerSample,
        position: usize,
        layer: usize,
        element: usize,
        values: [f32; 2],
    ) -> Result<bool> {
        // Rule B uses the FP32 reduction-order interval (4d 2026-10-29T00:15:24Z).
        let [got, want] = values;
        let mixer = engine.deepseek_mixer.as_ref().context("mixer missing")?;
        let dot = mixer.wo_b_row_host(&engine.device.stream, layer, element)?;
        if dot.act_codes != row.wo_b_codes || dot.act_scales != row.wo_b_scales {
            let code = dot
                .act_codes
                .iter()
                .zip(&row.wo_b_codes)
                .position(|(got, want)| got != want);
            let scale = dot
                .act_scales
                .iter()
                .zip(&row.wo_b_scales)
                .position(|(got, want)| got != want);
            println!(
                "LAYER_RULE_B_OPERANDS layer={layer} position={position} element={element} code_mismatch={code:?} scale_mismatch={scale:?}"
            );
            return Ok(false);
        }
        let (mut exact, mut sum_abs) = (0.0f64, 0.0f64);
        for (i, (&w, &x)) in dot.w_codes.iter().zip(&dot.act_codes).enumerate() {
            let scale = mix_e8m0(dot.w_scales[i / 32]) * mix_e8m0(dot.act_scales[i / 32]);
            let product = f64::from(crate::fp8::decode_e4m3(w))
                * f64::from(crate::fp8::decode_e4m3(x))
                * scale;
            exact += product;
            sum_abs += product.abs();
        }
        let n = (dot.act_codes.len() / 16) as f64;
        let gamma = n * f64::exp2(-24.0) / (1.0 - n * f64::exp2(-24.0));
        let bound = gamma * sum_abs;
        let (low, high) = (exact - bound, exact + bound);
        let rounding_radius = |value: f64| -> f64 {
            if value == 0.0 {
                return 0.5 * f64::from(bf16::from_bits(1).to_f32());
            }
            let slot = bf16::from_f32(value as f32);
            let center = f64::from(slot.to_f32());
            let below = f64::from(bf16::from_bits(slot.to_bits().wrapping_sub(1)).to_f32());
            let above = f64::from(bf16::from_bits(slot.to_bits().wrapping_add(1)).to_f32());
            ((value - 0.5 * (below + center)).abs()).min((0.5 * (center + above) - value).abs())
        };
        let (got_value, want_value) = (f64::from(got), f64::from(want));
        let (got_radius, want_radius) = (rounding_radius(got_value), rounding_radius(want_value));
        let proven = got_value + got_radius >= low
            && got_value - got_radius <= high
            && want_value + want_radius >= low
            && want_value - want_radius <= high;
        println!(
            "LAYER_RULE_B layer={layer} position={position} element={element} exact={exact:e} interval=[{low:e},{high:e}] candidate={got_value:e} publisher={want_value:e} gamma={gamma:e} sum_abs={sum_abs:e} exempt={proven}"
        );
        Ok(proven)
    }

    fn layer_collapse_proof(
        engine: &Engine,
        frozen: &LayerBands,
        row: &LayerSample,
        name: &str,
        element: usize,
        values: [f32; 2],
    ) -> Result<bool> {
        let s = &engine.device.stream;
        let hc = engine.hc.as_ref().context("mHC missing")?;
        let (pre, expected_pre, expected_stream) = if name == "collapse_attn" {
            (
                hc.carry.as_ref().context("mHC carry missing")?,
                &row.carried_pre,
                layer_bf(&row.input),
            )
        } else {
            (
                hc.attn_pre.as_ref().context("mHC attention pre missing")?,
                row.boundaries.get("attn_pre").context("attn pre missing")?,
                layer_expected(row, "expand_attn", 20480)?,
            )
        };
        let pre = s.clone_dtoh(&pre.slice(..4))?;
        let stream = s.clone_dtoh(&hc.stream.slice(..20480))?;
        ensure!(
            pre.iter()
                .map(|v| v.to_bits())
                .eq(expected_pre.iter().map(|v| v.to_bits()))
                && stream == expected_stream,
            "collapse certificate operands differ from recorded upstream"
        );
        let operands: Vec<f64> = (0..4)
            .map(|j| f64::from(stream[j * 5120 + element].to_f32()))
            .collect();
        let products: Vec<f64> = pre
            .iter()
            .zip(&operands)
            .map(|(&a, &b)| f64::from(a) * b)
            .collect();
        let exact: f64 = products.iter().sum();
        let sum_abs: f64 = products.iter().map(|v| v.abs()).sum();
        let gamma = 3.0 * f64::exp2(-24.0) / (1.0 - 3.0 * f64::exp2(-24.0));
        let lower = bf16::from_f32((exact - gamma * sum_abs) as f32).to_f32();
        let upper = bf16::from_f32((exact + gamma * sum_abs) as f32).to_f32();
        let exempt = values
            .iter()
            .all(|v| v.is_finite() && *v >= lower && *v <= upper);
        println!(
            "LAYER_COLLAPSE_CERTIFICATE {}",
            serde_json::json!({"layer": frozen.layer, "position": row.position, "op": name,
                "element": element, "pre": pre, "stream": operands, "products": products,
                "exact": exact, "sum_abs": sum_abs, "gamma": gamma, "n": 3,
                "lower": lower, "upper": upper, "candidate": values[0],
                "publisher": values[1], "exempt": exempt})
        );
        Ok(exempt)
    }

    fn layer_score(
        engine: &Engine,
        frozen: &LayerBands,
        row: &LayerSample,
        name: &str,
        got: &[f32],
        chain: Option<&layer_certificate::Certificate>,
    ) -> Result<usize> {
        let index = frozen
            .positions
            .iter()
            .position(|&p| p == row.position)
            .context("unfrozen position")?;
        let entry = frozen
            .bands
            .get(name)
            .and_then(|bands| bands.get(index))
            .context("unfrozen boundary")?;
        ensure!(
            entry.spread.is_finite()
                && entry.analytic.is_finite()
                && entry.spread >= 0.0
                && entry.analytic >= 0.0
                && entry.bound >= (3.0 * entry.spread).max(entry.analytic)
                && entry.bound
                    <= (3.0 * entry.spread).max(entry.analytic) * (1.0 + 4.0 * f64::EPSILON),
            "invalid frozen bound"
        );
        let want = row
            .boundaries
            .get(name)
            .context("reference boundary missing")?;
        ensure!(
            !got.is_empty()
                && got.len() == want.len()
                && got.iter().chain(want).all(|v| v.is_finite()),
            "boundary shape/nonfinite"
        );
        let (mut beyond, mut exempt, mut max_abs) = (0usize, 0usize, 0.0f64);
        let common_projection_operands = if name == "attn"
            && got
                .iter()
                .zip(want)
                .any(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() > entry.bound)
        {
            let mixer = engine.deepseek_mixer.as_ref().context("mixer absent")?;
            let s = &engine.device.stream;
            let codes = s.clone_dtoh(&mixer.act_o_codes.slice(..8192))?;
            let scales = s.clone_dtoh(&mixer.act_o_scales.slice(..256))?;
            let equal = codes == row.wo_b_codes && scales == row.wo_b_scales;
            if !equal {
                println!(
                    "LAYER_RULE_B_CHAIN_REQUIRED layer={} position={} code_mismatches={} scale_mismatches={}",
                    frozen.layer,
                    row.position,
                    codes
                        .iter()
                        .zip(&row.wo_b_codes)
                        .filter(|(a, b)| a != b)
                        .count(),
                    scales
                        .iter()
                        .zip(&row.wo_b_scales)
                        .filter(|(a, b)| a != b)
                        .count()
                );
            }
            equal
        } else {
            false
        };
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            max_abs = max_abs.max(error);
            if error > entry.bound {
                beyond += 1;
                let proven = if let Some(chain) = chain.filter(|_| name == "attn") {
                    chain.covers(i, [a, b])
                } else if common_projection_operands {
                    layer_store_proof(engine, row, row.position, frozen.layer, i, [a, b])?
                } else if matches!(name, "collapse_attn" | "collapse_ffn") {
                    layer_collapse_proof(engine, frozen, row, name, i, [a, b])?
                } else {
                    false
                };
                if proven {
                    exempt += 1;
                }
            }
        }
        let defects = beyond - exempt;
        println!(
            "LAYER_BOUNDARY layer={} position={} op={name} compared={} beyond={beyond} exempt={exempt} defects={defects} max_abs={max_abs:e} bound={:e} isolation=recorded_upstream",
            frozen.layer,
            row.position,
            got.len(),
            entry.bound
        );
        Ok(defects)
    }

    fn layer_attention_record(engine: &Engine, layer: usize, position: usize) -> Result<()> {
        let s = &engine.device.stream;
        let mixer = engine.deepseek_mixer.as_ref().context("mixer absent")?;
        let ring = engine.state.latent_ring(layer)?;
        let bits = |values: Vec<bf16>| values.into_iter().map(bf16::to_bits).collect::<Vec<_>>();
        let (attention, groups) = mixer.attn_output_host(s, 1)?;
        let record = serde_json::json!({
            "layer": layer, "position": position,
            "query": bits(s.clone_dtoh(&engine.work.qkv.slice(..32768))?),
            "raw_codes": s.clone_dtoh(&ring.codes)?,
            "raw_scales": s.clone_dtoh(&ring.scales)?,
            "inverse_rotary": s.clone_dtoh(&mixer.rotary_inv.slice(..64))?,
            "forward_rotary": s.clone_dtoh(&mixer.rotary_fwd.slice(..64))?,
            "source_codes": s.clone_dtoh(&mixer.act_hidden_codes.slice(..5120))?,
            "source_scales": s.clone_dtoh(&mixer.act_hidden_scales.slice(..160))?,
            "query_codes": s.clone_dtoh(&mixer.act_lora_codes.slice(..mixer.geometry.q_lora))?,
            "query_scales": s.clone_dtoh(&mixer.act_lora_scales.slice(..mixer.geometry.q_lora / 32))?,
            "attention_post_rotary": bits(attention), "wo_b_input": bits(groups),
            "attention_codes": s.clone_dtoh(&mixer.o_codes.slice(..32768))?,
            "attention_scales": s.clone_dtoh(&mixer.o_scales.slice(..1024))?,
            "wo_b_codes": s.clone_dtoh(&mixer.act_o_codes.slice(..8192))?,
            "wo_b_scales": s.clone_dtoh(&mixer.act_o_scales.slice(..256))?,
            "output": bits(s.clone_dtoh(&engine.work.out.slice(..5120))?),
        });
        println!("LAYER_ATTENTION_RECORD {record}");
        Ok(())
    }

    fn layer_route_rule_b(
        engine: &Engine,
        layer: usize,
        row: &LayerSample,
        engine_logits: &[f32],
        engine_weights: &[f32],
    ) -> Result<usize> {
        const EXPERTS: usize = 384;
        const TOP_K: usize = 6;
        const HIDDEN: usize = 5120;
        let s = &engine.device.stream;
        let ffn = engine
            .deepseek_ffn
            .as_ref()
            .context("DeepSeek-V4.1 FFN runtime missing")?;
        let bound = ffn
            .layers
            .get(layer)
            .with_context(|| format!("DeepSeek-V4.1 FFN layer {layer} has no bound weights"))?
            .as_ref()
            .context("DeepSeek-V4.1 FFN layer has no bound weights")?;
        let bytes = s.clone_dtoh(&bound.weights.gate_weight)?;
        ensure!(
            bytes.len() == EXPERTS * HIDDEN * 2,
            "router weight extent {}",
            bytes.len()
        );
        let weights: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])))
            .collect();
        let input = layer_bf(&row.ffn_norm);
        let gamma = crate::backend::cuda::gamma(HIDDEN, f64::exp2(-24.0));
        let sqrt_softplus =
            |logit: f64| -> f64 { (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt() };
        let mut invalid = 0usize;
        let mut score_intervals = Vec::with_capacity(TOP_K);
        for &id in &row.router_ids {
            let id = usize::try_from(id).context("router expert id overflow")?;
            let (mut exact, mut sum_abs) = (0.0f64, 0.0f64);
            for (weight, value) in weights[id * HIDDEN..(id + 1) * HIDDEN].iter().zip(&input) {
                let product = f64::from(weight.to_f32()) * f64::from(value.to_f32());
                exact += product;
                sum_abs += product.abs();
            }
            let radius = gamma * sum_abs;
            let logit_interval = (exact - radius, exact + radius);
            let logit_valid = f64::from(engine_logits[id]) >= logit_interval.0
                && f64::from(engine_logits[id]) <= logit_interval.1;
            let score = sqrt_softplus(logit_interval.0).max(0.0);
            let score_high = sqrt_softplus(logit_interval.1).max(0.0);
            let score_interval = (
                f32_below(score.min(score_high)),
                f32_above(score.max(score_high)),
            );
            score_intervals.push((
                id,
                exact,
                sum_abs,
                logit_interval,
                logit_valid,
                score_interval,
            ));
        }
        let sum_gamma = crate::backend::cuda::gamma(TOP_K, f64::exp2(-24.0));
        let op_gamma = 2.0 * crate::backend::cuda::gamma(1, f64::exp2(-24.0));
        let control_id = usize::try_from(row.router_ids[0]).context("router expert id overflow")?;
        let control_logit = engine_logits[control_id] + 1.0;
        let control_interval = score_intervals
            .iter()
            .find(|&&(id, ..)| id == control_id)
            .map(|&(_, _, _, interval, _, _)| interval)
            .context("router control expert absent")?;
        let control_detected = !(f64::from(control_logit) >= control_interval.0
            && f64::from(control_logit) <= control_interval.1);
        invalid += usize::from(!control_detected);
        println!(
            "LAYER_ROUTE_CONTROL layer={layer} position={} id={control_id} logit={:e} interval=[{:e},{:e}] detected={control_detected}",
            row.position,
            f64::from(control_logit),
            control_interval.0,
            control_interval.1
        );
        for (slot, &(id, exact, sum_abs, logit_interval, logit_valid, score_interval)) in
            score_intervals.iter().enumerate()
        {
            let mut low_den = 1e-20f64;
            let mut high_den = 1e-20f64;
            let mut max_den_abs = 1e-20f64;
            for &(_, _, _, _, _, other_score) in &score_intervals {
                high_den += other_score.1;
                low_den += other_score.0;
                max_den_abs += other_score.1;
            }
            let den_radius = sum_gamma * max_den_abs;
            high_den += den_radius;
            low_den = (low_den - den_radius).max(f64::from(f32::from_bits(1)));
            let normalize = |numerator: f64, denominator: f64| -> (f64, f64) {
                let value = f64::from(ffn.geometry.routed_scaling) * numerator / denominator;
                let radius = op_gamma * value.abs();
                (value - radius, value + radius)
            };
            let low_raw = normalize(score_interval.0, high_den).0;
            let high_raw = normalize(score_interval.1, low_den).1;
            let interval = (f32_below(low_raw), f32_above(high_raw));
            let got = f64::from(engine_weights[slot]);
            let want = f64::from(row.route_weights[slot]);
            let valid = logit_valid
                && interval.0 <= got
                && got <= interval.1
                && interval.0 <= want
                && want <= interval.1;
            invalid += usize::from(!valid);
            println!(
                "LAYER_ROUTE_RULE_B layer={layer} position={} slot={slot} id={id} n={HIDDEN} gamma={gamma:e} logit_exact={exact:e} logit_interval=[{:e},{:e}] engine_logit={:e} sum_abs={sum_abs:e} score_interval=[{:e},{:e}] weight_interval=[{:e},{:e}] candidate={got:e} publisher={want:e} exempt={valid}",
                row.position,
                logit_interval.0,
                logit_interval.1,
                f64::from(engine_logits[id]),
                score_interval.0,
                score_interval.1,
                interval.0,
                interval.1
            );
        }
        Ok(invalid)
    }

    fn layer_routing_score(
        engine: &Engine,
        frozen: &LayerBands,
        row: &LayerSample,
        engine_logits: &[f32],
        engine_ids: &[i32],
        engine_weights: &[f32],
    ) -> Result<usize> {
        const EXPERTS: usize = 384;
        const TOP_K: usize = 6;
        let index = frozen
            .positions
            .iter()
            .position(|&p| p == row.position)
            .context("unfrozen routing position")?;
        let bound = frozen.bands["ffn"][index].bound;
        ensure!(
            engine_logits.len() == EXPERTS
                && engine_ids.len() == TOP_K
                && row.router_ids.len() == TOP_K
                && engine_weights.len() == TOP_K
                && row.route_weights.len() == TOP_K
                && row.router_cutoff >= 0.0
                && engine_logits.iter().all(|value| value.is_finite())
                && engine_weights
                    .iter()
                    .chain(&row.route_weights)
                    .all(|value| value.is_finite()),
            "router decision shape or non-finite value"
        );
        let cutoff = |values: &[f32]| -> f32 {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            sorted[TOP_K - 1] - sorted[TOP_K]
        };
        let near_tie =
            f64::from(cutoff(engine_logits)) <= bound || f64::from(row.router_cutoff) <= bound;
        let id_mismatches = engine_ids
            .iter()
            .zip(&row.router_ids)
            .filter(|(got, want)| got != want)
            .count();
        let weight_mismatches = engine_weights
            .iter()
            .zip(&row.route_weights)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        if weight_mismatches != 0 {
            let (index, got, want) = engine_weights
                .iter()
                .zip(&row.route_weights)
                .enumerate()
                .find_map(|(index, (got, want))| {
                    (got.to_bits() != want.to_bits()).then_some((
                        index,
                        f64::from(*got),
                        f64::from(*want),
                    ))
                })
                .unwrap_or((0, 0.0, 0.0));
            println!(
                "LAYER_ROUTING_FIRST layer={} position={} slot={index} engine={got:e} reference={want:e} delta={:e}",
                frozen.layer,
                row.position,
                (got - want).abs(),
            );
        }
        let rule_b_failures = if near_tie || id_mismatches != 0 {
            0
        } else {
            layer_route_rule_b(engine, frozen.layer, row, engine_logits, engine_weights)?
        };
        let defects = if near_tie {
            0
        } else {
            usize::from(id_mismatches != 0) + rule_b_failures
        };
        println!(
            "LAYER_ROUTING layer={} position={} compared={} id_mismatches={} weight_mismatches={} rule_b_failures={rule_b_failures} engine_cutoff={} reference_cutoff={} bound={bound:e} near_tie={near_tie} defects={defects}",
            frozen.layer,
            row.position,
            TOP_K,
            id_mismatches,
            weight_mismatches,
            cutoff(engine_logits),
            row.router_cutoff,
        );
        Ok(defects)
    }

    fn validate_layer_fixture(
        layer: usize,
        fixture: &LayerSamples,
        bands: &LayerBands,
    ) -> Result<()> {
        const POSITIONS: [usize; 8] = [0, 1, 2, 63, 64, 126, 127, 128];
        const NAMES: [&str; 13] = [
            "attn_pre",
            "attn_post",
            "attn_comb",
            "ffn_pre",
            "ffn_post",
            "ffn_comb",
            "collapse_attn",
            "collapse_ffn",
            "attn",
            "ffn",
            "expand_attn",
            "expand_ffn",
            "carried_pre",
        ];
        let boundary_len = |name: &str| -> usize {
            match name {
                "attn_comb" | "ffn_comb" => 16,
                "collapse_attn" | "collapse_ffn" | "attn" | "ffn" => 5120,
                "expand_attn" | "expand_ffn" => 20480,
                _ => 4,
            }
        };
        ensure!(
            bands.layer == layer
                && fixture.layer == layer
                && fixture.revision == "DeepSeek-V4.1-Flash-1017"
                && bands.positions == POSITIONS
                && fixture
                    .samples
                    .iter()
                    .map(|row| row.position)
                    .collect::<Vec<_>>()
                    == POSITIONS
                && bands.bands.len() == NAMES.len(),
            "layer {layer} fixture/threshold identity or position mismatch"
        );
        for name in NAMES {
            let values = bands
                .bands
                .get(name)
                .with_context(|| format!("layer {layer} missing band {name}"))?;
            ensure!(
                values.len() == POSITIONS.len(),
                "layer {layer} band {name} extent"
            );
            for value in values {
                let derived = (3.0 * value.spread).max(value.analytic);
                ensure!(
                    value.spread.is_finite()
                        && value.analytic.is_finite()
                        && value.bound.is_finite()
                        && value.spread >= 0.0
                        && value.analytic >= 0.0
                        && value.bound >= derived
                        && value.bound <= derived * (1.0 + 4.0 * f64::EPSILON),
                    "layer {layer} invalid band {name}: spread={} analytic={} bound={}",
                    value.spread,
                    value.analytic,
                    value.bound
                );
            }
        }
        ensure!(
            fixture.window_codes.len() == 129
                && fixture.window_codes.iter().all(|row| row.len() == 512)
                && fixture.window_scales.len() == 129
                && fixture.window_scales.iter().all(|row| row.len() == 16),
            "layer {layer} latent window layout mismatch"
        );
        if let Some(ids) = fixture.window_ids.as_ref() {
            ensure!(
                ids.len() == 129 && ids.iter().all(|row| row.len() == 128),
                "layer {layer} latent window id layout mismatch"
            );
        }
        if fixture.compress_ratio == 0 {
            ensure!(
                fixture.shared_compress_kv.is_none()
                    && fixture.shared_index_k.is_none()
                    && fixture.shared_topk_idxs.is_none()
                    && fixture.compressed_kv.is_none()
                    && fixture.compressed_ids.is_none(),
                "layer {layer} ratio-0 fixture carries compressed state"
            );
        } else {
            let compressed = fixture
                .compressed_kv
                .as_ref()
                .context("compressed KV missing")?;
            let ids = fixture
                .compressed_ids
                .as_ref()
                .context("compressed indices missing")?;
            ensure!(
                !compressed.is_empty() && compressed.iter().all(|row| row.len() == 512),
                "layer {layer} compressed KV layout mismatch"
            );
            let codes = fixture
                .compressed_codes
                .as_ref()
                .context("original compressed codes missing")?;
            let scales = fixture
                .compressed_scales
                .as_ref()
                .context("original compressed scales missing")?;
            ensure!(
                codes.len() == compressed.len()
                    && codes.iter().all(|row| row.len() == 256)
                    && scales.len() == compressed.len()
                    && scales.iter().all(|row| row.len() == 32),
                "layer {layer} native compressed codec layout mismatch"
            );
            ensure!(
                ids.len() == 129 && ids.iter().all(|row| row.len() == compressed.len()),
                "layer {layer} compressed index layout mismatch"
            );
            if layer == 2 {
                ensure!(
                    fixture.shared_compress_kv.is_none()
                        && fixture.shared_index_k.is_none()
                        && fixture.shared_topk_idxs.is_none(),
                    "layer 2 carries pre-existing shared compressed state"
                );
            } else {
                let compress = fixture
                    .shared_compress_kv
                    .as_ref()
                    .context("shared compressed KV missing")?;
                let index = fixture
                    .shared_index_k
                    .as_ref()
                    .context("shared index K missing")?;
                let topk = fixture
                    .shared_topk_idxs
                    .as_ref()
                    .context("shared top-k indices missing")?;
                ensure!(
                    !compress.is_empty()
                        && compress.iter().all(|row| row.len() == 512)
                        && index.len() == compress.len()
                        && index.iter().all(|row| row.len() == 128)
                        && topk.len() == 129
                        && !topk[0].is_empty()
                        && topk.iter().all(|row| row.len() == topk[0].len()),
                    "layer {layer} shared compressed/index layout mismatch"
                );
            }
        }
        for row in &fixture.samples {
            ensure!(
                row.input.len() == 20480
                    && row.carried_pre.len() == 4
                    && row.attn_norm.len() == 5120
                    && row.ffn_norm.len() == 5120
                    && row.router_cutoff.is_finite()
                    && row.router_cutoff >= 0.0
                    && row.router_ids.len() == 6
                    && row.route_weights.len() == 6
                    && row.wo_b_codes.len() == 8192
                    && row.wo_b_scales.len() == 256
                    && row
                        .carried_pre
                        .iter()
                        .chain(&row.route_weights)
                        .all(|v| { v.is_finite() }),
                "layer {layer} sample {} operand layout mismatch",
                row.position
            );
            ensure!(
                row.boundaries.len() == NAMES.len(),
                "layer {layer} sample {} boundary count",
                row.position
            );
            for name in NAMES {
                let count = boundary_len(name);
                let values = row
                    .boundaries
                    .get(name)
                    .with_context(|| format!("layer {layer} missing boundary {name}"))?;
                ensure!(
                    values.len() == count && values.iter().all(|value| value.is_finite()),
                    "layer {layer} boundary {name} layout/nonfinite value"
                );
                if matches!(
                    name,
                    "collapse_attn"
                        | "collapse_ffn"
                        | "attn"
                        | "ffn"
                        | "expand_attn"
                        | "expand_ffn"
                ) {
                    layer_expected(row, name, count)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only fixture reader; set FF_DS41_LAYERS_FIXTURE to frozen thresholds-r3 plus layer JSON"]
    fn d1_layers_fixture_contract() -> Result<()> {
        let root = PathBuf::from(std::env::var("FF_DS41_LAYERS_FIXTURE")?);
        let frozen: std::collections::BTreeMap<String, LayerBands> =
            serde_json::from_slice(&std::fs::read(root.join("thresholds-r3.json"))?)?;
        for layer in 1..40 {
            let fixture: LayerSamples = serde_json::from_slice(&std::fs::read(
                root.join(format!("layer-{layer:02}.json")),
            )?)?;
            let bands = frozen
                .get(&layer.to_string())
                .with_context(|| format!("layer {layer} has no frozen bands"))?;
            validate_layer_fixture(layer, &fixture, bands)?;
        }
        println!("PASS d1_layers_fixture_contract layers=39 samples=312 boundaries=4056");
        Ok(())
    }

    // 2026-10-28: every judged boundary consumes all recorded upstream operands.
    #[test]
    #[ignore = "GPU lease, FF_DS41_MODEL, FF_DS41_LAYERS_LAYER and FF_DS41_LAYERS_FIXTURE with frozen thresholds-r3"]
    fn d1_layers_match_model_definition() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = PathBuf::from(std::env::var("FF_DS41_LAYERS_FIXTURE")?);
        let layer = std::env::var("FF_DS41_LAYERS_LAYER")?
            .parse::<usize>()
            .context("FF_DS41_LAYERS_LAYER must be a layer index")?;
        ensure!(
            (1..40).contains(&layer),
            "FF_DS41_LAYERS_LAYER {layer} is outside layers 1..39"
        );
        let frozen: std::collections::BTreeMap<String, LayerBands> =
            serde_json::from_slice(&std::fs::read(root.join("thresholds-r3.json"))?)?;
        let bands = frozen
            .get(&layer.to_string())
            .with_context(|| format!("layer {layer} has no frozen bands"))?;
        let fixture: LayerSamples =
            serde_json::from_slice(&std::fs::read(root.join(format!("layer-{layer:02}.json")))?)?;
        validate_layer_fixture(layer, &fixture, bands)?;
        let runtime_dir =
            std::env::temp_dir().join(format!("flyingfish-deepseek-layers-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;
        let hash_record = PathBuf::from(std::env::var_os("FF_DS41_ENGRAM_HASH").context(
            "FF_DS41_ENGRAM_HASH missing: the accepted engram-hash-v1 record is required",
        )?);
        std::fs::copy(&hash_record, runtime_dir.join("engram-hash-v1.json"))
            .with_context(|| format!("copy {}", hash_record.display()))?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(runtime_dir.clone()),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let mut engine = Engine::load(
            &PathBuf::from(std::env::var("FF_DS41_MODEL")?),
            0,
            130,
            None,
            settings,
        )?;
        let device = engine.device.clone();
        let s = &device.stream;
        let step = Step {
            rows: 1,
            input: Input::Ids,
            head: Head::None,
            mask: Mask::Causal,
        };
        let geometry = MixerGeometry::new(&engine.model_spec)?;
        let mut all_defects = 0usize;
        for layer in [layer] {
            let ds = engine
                .model_spec
                .deepseek_v41
                .as_ref()
                .context("DeepSeek-V4.1 geometry missing")?;
            let ratio = ds.compress_ratios[layer];
            ensure!(
                fixture.compress_ratio == ratio,
                "layer {layer} reference compress ratio {} is not {ratio}",
                fixture.compress_ratio
            );
            if ratio != 0 {
                engine
                    .deepseek_mixer
                    .as_mut()
                    .context("mixer absent")?
                    .dsa
                    .as_mut()
                    .context("DSA runtime absent")?
                    .isolate_published_attention(layer)?;
            }
            let mut layer_defects = 0usize;
            for row in &fixture.samples {
                let mut linked_chain = None;
                let computed_chain;
                engine.prepare_step(step)?;
                engine.position = row.position;
                engine.state.set_sequence(
                    s,
                    row.position as u32,
                    1,
                    engine.workspace_rows as u32,
                )?;
                layer_ring(&mut engine, &fixture, row.position, false)?;
                layer_hc_inputs(&mut engine, &layer_bf(&row.input), &row.carried_pre, false)?;
                engine.run_hc_stage(0, HcStage::Pre, layer, step)?;
                for (name, got) in layer_mix_rows(&engine, "attn")? {
                    layer_defects += layer_score(&engine, bands, row, &name, &got, linked_chain)?;
                }
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "collapse_attn",
                    &s.clone_dtoh(&engine.work.x.slice(..5120))?
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                s.memcpy_htod(
                    &layer_expected(row, "collapse_attn", 5120)?,
                    &mut engine.work.x.slice_mut(..5120),
                )?;
                engine.run_operator(Operator::Norm, layer, step, &mut None)?;
                let attn_norm = layer_bf(&row.attn_norm);
                s.memcpy_htod(&attn_norm, &mut engine.work.n.slice_mut(..5120))?;
                ensure!(
                    s.clone_dtoh(&engine.work.n.slice(..5120))? == attn_norm,
                    "attention norm operand isolation failed"
                );
                engine.run_mixer(
                    engine.model_spec.layers[layer].mixer,
                    layer,
                    step,
                    &mut None,
                )?;
                let visible = (row.position + 1).min(128);
                let got_ids = if ratio == 0 {
                    engine
                        .deepseek_mixer
                        .as_ref()
                        .context("DeepSeek-V4.1 mixer runtime is not bound")?
                        .visible_slots_host(s, visible)?
                } else {
                    (row.position + 1 - visible..=row.position)
                        .map(|p| (p % geometry.window) as i32)
                        .collect()
                };
                if let Some(rows) = fixture.window_ids.as_ref() {
                    let want_ids = &rows[row.position][..visible];
                    let ring = engine.state.latent_ring(layer)?;
                    let ring_codes = s.clone_dtoh(&ring.codes)?;
                    let ring_scales = s.clone_dtoh(&ring.scales)?;
                    let nblocks = geometry.head_dim / 32;
                    let mut injected_mismatches = 0usize;
                    let mut current_mismatches = 0usize;
                    for (slot, token) in got_ids.iter().zip(want_ids) {
                        let slot =
                            usize::try_from(*slot).context("latent ring slot index overflow")?;
                        let code_start = slot * geometry.head_dim;
                        let code_end = code_start + geometry.head_dim;
                        let scale_start = slot * nblocks;
                        let scale_end = scale_start + nblocks;
                        let token = usize::try_from(*token)
                            .context("latent window token index overflow")?;
                        let matches = ring_codes[code_start..code_end]
                            == fixture.window_codes[token]
                            && ring_scales[scale_start..scale_end] == fixture.window_scales[token];
                        if !matches {
                            if token == row.position {
                                current_mismatches += 1;
                            } else {
                                injected_mismatches += 1;
                            }
                        }
                    }
                    ensure!(
                        injected_mismatches == 0,
                        "layer {layer} position {} injected window operand mismatches={injected_mismatches}",
                        row.position,
                    );
                    println!(
                        "LAYER_WINDOW_OPERANDS layer={layer} position={} visible={visible} injected_mismatches={injected_mismatches} current_mismatches={current_mismatches}",
                        row.position,
                    );
                }
                println!(
                    "LAYER_WINDOW layer={layer} position={} visible={visible} first={:?} last={:?}",
                    row.position,
                    &got_ids[..visible.min(4)],
                    &got_ids[visible.saturating_sub(4)..],
                );
                let chain_root = root
                    .join("attention-stores")
                    .join(format!("layer{layer}-pos{}", row.position));
                if chain_root.is_dir() {
                    computed_chain = layer_certificate::prove(
                        &engine,
                        layer,
                        row.position,
                        &row.attn_norm,
                        &chain_root,
                    )?;
                    linked_chain = Some(&computed_chain);
                }
                let attention_defects = layer_score(
                    &engine,
                    bands,
                    row,
                    "attn",
                    &s.clone_dtoh(&engine.work.out.slice(..5120))?
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                layer_defects += attention_defects;
                if attention_defects != 0 {
                    layer_attention_record(&engine, layer, row.position)?;
                }
                if ratio == 0 && row.position == 127 && fixture.window_ids.is_some() {
                    let nblocks = geometry.head_dim / 32;
                    let mut replay_codes = vec![0u8; geometry.window * geometry.head_dim];
                    let mut replay_scales = vec![0u8; geometry.window * nblocks];
                    for token in
                        &fixture.window_ids.as_ref().context("window ids missing")?[row.position]
                    {
                        let token = usize::try_from(*token)
                            .context("latent window token index overflow")?;
                        let slot = token % geometry.window;
                        replay_codes[slot * geometry.head_dim..(slot + 1) * geometry.head_dim]
                            .copy_from_slice(&fixture.window_codes[token]);
                        replay_scales[slot * nblocks..(slot + 1) * nblocks]
                            .copy_from_slice(&fixture.window_scales[token]);
                    }
                    {
                        let mixer = engine
                            .deepseek_mixer
                            .as_ref()
                            .context("DeepSeek-V4.1 mixer runtime is not bound")?;
                        let ring = engine.state.latent_ring_mut(layer)?;
                        println!(
                            "LAYER_WINDOW_REPLAY_INPUTS layer={layer} position={} q_len={} out_len={} ring_codes={} ring_scales={} replay_codes={} replay_scales={}",
                            row.position,
                            engine.work.qkv.len(),
                            engine.work.out.len(),
                            ring.codes.len(),
                            ring.scales.len(),
                            replay_codes.len(),
                            replay_scales.len(),
                        );
                        mixer.replay_recorded_window(RecordedWindowReplay {
                            s,
                            blas: &engine.blas,
                            ring,
                            layer,
                            position: row.position,
                            q: &engine.work.qkv,
                            codes: &replay_codes,
                            scales: &replay_scales,
                            out: &mut engine.work.out,
                        })?;
                    }
                    let replay: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..5120))?;
                    let want = layer_expected(row, "attn", 5120)?;
                    let max_abs = replay
                        .iter()
                        .zip(&want)
                        .map(|(got, want)| {
                            (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                        })
                        .fold(0.0f64, f64::max);
                    let bound = bands.bands["attn"][bands
                        .positions
                        .iter()
                        .position(|p| *p == row.position)
                        .context("unfrozen replay position")?]
                    .bound;
                    let beyond = replay
                        .iter()
                        .zip(&want)
                        .filter(|(got, want)| {
                            (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs() > bound
                        })
                        .count();
                    let mixer = engine
                        .deepseek_mixer
                        .as_ref()
                        .context("DeepSeek-V4.1 mixer runtime is not bound")?;
                    let dot = mixer.wo_b_row_host(s, layer, 0)?;
                    let code_mismatches = dot
                        .act_codes
                        .iter()
                        .zip(&row.wo_b_codes)
                        .filter(|(got, want)| got != want)
                        .count();
                    let scale_mismatches = dot
                        .act_scales
                        .iter()
                        .zip(&row.wo_b_scales)
                        .filter(|(got, want)| got != want)
                        .count();
                    println!(
                        "LAYER_WINDOW_REPLAY layer={layer} position={} compared=5120 beyond={beyond} max_abs={max_abs:e} bound={bound:e} wo_b_code_mismatches={code_mismatches} wo_b_scale_mismatches={scale_mismatches}",
                        row.position,
                    );
                    let mixer = engine
                        .deepseek_mixer
                        .as_ref()
                        .context("DeepSeek-V4.1 mixer runtime is not bound")?;
                    let weights = mixer
                        .weights
                        .get(layer)
                        .context("DeepSeek-V4.1 mixer layer weights missing")?
                        .as_ref()
                        .context("DeepSeek-V4.1 mixer layer weights missing")?;
                    let ring = engine.state.latent_ring(layer)?;
                    let sink = weight_view::<f32, 1>(
                        &weights.attn_sink,
                        s,
                        [geometry.heads as u32],
                        "attn_sink",
                    )?;
                    let slots = s.alloc_zeros::<i32>(geometry.window)?;
                    let packed = s.alloc_zeros::<u8>(128 * 256)?;
                    let packed_scales = s.alloc_zeros::<u8>(128 * 32)?;
                    let selected = s.clone_htod(&vec![-1i32; 512])?;
                    let committed = s.clone_htod(&[row.position as u32 + 1])?;
                    let sequence = s.clone_htod(&[
                        row.position as i32,
                        1,
                        row.position as i32 + 1,
                        row.position as i32 + 1,
                        1,
                        0,
                        0,
                        0,
                    ])?;
                    let indexed_out = s.clone_htod(&vec![
                        bf16::from_f32(127.);
                        geometry.heads * geometry.head_dim
                    ])?;
                    use crate::backend::cuda::latent_window_attention::{
                        IndexedAttentionPlan, IndexedInputs,
                    };
                    let plan = IndexedAttentionPlan::new(
                        &engine.device.ctx,
                        IndexedInputs {
                            window: LwaInputs {
                                q: view(
                                    &engine.work.qkv,
                                    s,
                                    0,
                                    [geometry.heads as u32, geometry.head_dim as u32],
                                    [geometry.head_dim as u64, 1],
                                )?,
                                k_codes: view(
                                    &ring.codes,
                                    s,
                                    0,
                                    [geometry.window as u32, geometry.head_dim as u32],
                                    [geometry.head_dim as u64, 1],
                                )?,
                                k_scales: view(
                                    &ring.scales,
                                    s,
                                    0,
                                    [geometry.window as u32, (geometry.head_dim / 32) as u32],
                                    [(geometry.head_dim / 32) as u64, 1],
                                )?,
                                slot_ids: view(&slots, s, 0, [geometry.window as u32], [1])?,
                                sink,
                                inv_rotary: view(
                                    &mixer.rotary_inv,
                                    s,
                                    0,
                                    [geometry.rotary as u32],
                                    [1],
                                )?,
                            },
                            packed: view(&packed, s, 0, [128, 256], [256, 1])?,
                            scales: view(&packed_scales, s, 0, [128, 32], [32, 1])?,
                            selected: view(&selected, s, 0, [1, 512], [512, 1])?,
                            committed: view(&committed, s, 0, [1], [1])?,
                            sequence: crate::backend::cuda::contracts::SeqRef {
                                ptr: sequence.device_ptr(s).0,
                            },
                            ratio: 1,
                        },
                        view(
                            &indexed_out,
                            s,
                            0,
                            [geometry.heads as u32, geometry.head_dim as u32],
                            [geometry.head_dim as u64, 1],
                        )?,
                        0,
                    )?;
                    plan.run(s)?;
                    let indexed = s.clone_dtoh(&indexed_out)?;
                    let window_raw = mixer.attn_output_host(s, 1)?.0;
                    let bit_mismatches = indexed
                        .iter()
                        .zip(&window_raw)
                        .filter(|(got, want)| got.to_bits() != want.to_bits())
                        .count();
                    let max_abs = indexed
                        .iter()
                        .zip(&window_raw)
                        .map(|(got, want)| {
                            (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                        })
                        .fold(0.0f64, f64::max);
                    println!(
                        "LAYER_INDEXED_REPLAY layer={layer} position={} selected=0 compared={} bit_mismatches={bit_mismatches} max_abs={max_abs:e}",
                        row.position,
                        indexed.len(),
                    );
                }
                layer_post_inputs(&mut engine, row, false)?;
                engine.run_hc_stage(3, HcStage::Post, layer, step)?;
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "expand_attn",
                    &s.clone_dtoh(
                        &engine
                            .hc
                            .as_ref()
                            .context("mHC missing")?
                            .stream
                            .slice(..20480),
                    )?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                layer_hc_inputs(
                    &mut engine,
                    &layer_expected(row, "expand_attn", 20480)?,
                    row.boundaries.get("attn_pre").context("attn pre missing")?,
                    true,
                )?;
                engine.run_hc_stage(4, HcStage::Pre, layer, step)?;
                for (name, got) in layer_mix_rows(&engine, "ffn")? {
                    layer_defects += layer_score(&engine, bands, row, &name, &got, linked_chain)?;
                }
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "collapse_ffn",
                    &s.clone_dtoh(&engine.work.x.slice(..5120))?
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                s.memcpy_htod(
                    &layer_expected(row, "collapse_ffn", 5120)?,
                    &mut engine.work.x.slice_mut(..5120),
                )?;
                engine.run_operator(Operator::PostNorm, layer, step, &mut None)?;
                let ffn_norm = layer_bf(&row.ffn_norm);
                s.memcpy_htod(&ffn_norm, &mut engine.work.n.slice_mut(..5120))?;
                ensure!(
                    s.clone_dtoh(&engine.work.n.slice(..5120))? == ffn_norm,
                    "FFN norm operand isolation failed"
                );
                engine.run_ffn(layer, step, &mut None)?;
                let stage = engine
                    .deepseek_ffn
                    .as_ref()
                    .context("DeepSeek-V4.1 FFN runtime missing")?
                    .stage_buffers();
                layer_defects += layer_routing_score(
                    &engine,
                    bands,
                    row,
                    &s.clone_dtoh(&stage.logits.slice(..384))?,
                    &s.clone_dtoh(&stage.ids.slice(..6))?,
                    &s.clone_dtoh(&stage.route_weights.slice(..6))?,
                )?;
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "ffn",
                    &s.clone_dtoh(&engine.work.out.slice(..5120))?
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                layer_post_inputs(&mut engine, row, true)?;
                engine.run_hc_stage(7, HcStage::Post, layer, step)?;
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "expand_ffn",
                    &s.clone_dtoh(
                        &engine
                            .hc
                            .as_ref()
                            .context("mHC missing")?
                            .stream
                            .slice(..20480),
                    )?
                    .iter()
                    .map(|v| v.to_f32())
                    .collect::<Vec<_>>(),
                    linked_chain,
                )?;
                layer_defects += layer_score(
                    &engine,
                    bands,
                    row,
                    "carried_pre",
                    &s.clone_dtoh(
                        &engine
                            .hc
                            .as_ref()
                            .context("mHC missing")?
                            .carry
                            .as_ref()
                            .context("carry missing")?
                            .slice(..4),
                    )?,
                    linked_chain,
                )?;
            }
            layer_controls(&mut engine, &fixture, bands, step)?;
            println!(
                "LAYER_RESULT layer={layer} positions={} defects={layer_defects}",
                fixture.samples.len()
            );
            all_defects += layer_defects;
        }
        println!("D1_LAYERS_RESULT layers=1 boundary_defects={all_defects} dsa_state_defects=0");
        drop(engine);
        std::fs::remove_dir_all(runtime_dir)?;
        ensure!(
            all_defects == 0,
            "non-Engram layer census DEFECT: {all_defects} elements"
        );
        Ok(())
    }

    fn layer_mix_rows(engine: &Engine, prefix: &str) -> Result<Vec<(String, Vec<f32>)>> {
        let s = &engine.device.stream;
        let hc = engine.hc.as_ref().context("mHC missing")?;
        Ok(vec![
            (
                format!("{prefix}_pre"),
                s.clone_dtoh(&hc.mix_pre.slice(..4))?,
            ),
            (
                format!("{prefix}_post"),
                s.clone_dtoh(&hc.mix_post.slice(..4))?,
            ),
            (
                format!("{prefix}_comb"),
                s.clone_dtoh(&hc.mix_comb.slice(..16))?,
            ),
        ])
    }

    fn layer_controls(
        engine: &mut Engine,
        fixture: &LayerSamples,
        bands: &LayerBands,
        step: Step<'_>,
    ) -> Result<()> {
        let row = fixture.samples.last().context("no control row")?;
        let layer = fixture.layer;
        let device = engine.device.clone();
        let s = &device.stream;
        let index = bands.positions.len() - 1;
        let tolerance = bands.bands["collapse_attn"][index].bound;
        let expected = layer_expected(row, "collapse_attn", 5120)?;
        layer_hc_inputs(engine, &layer_bf(&row.input), &row.carried_pre, false)?;
        engine.run_hc_stage(0, HcStage::Pre, layer, step)?;
        let rejected = !layer_collapse_proof(
            engine,
            bands,
            row,
            "collapse_attn",
            0,
            [bf16::from_f32(1.0e30).to_f32(), expected[0].to_f32()],
        )?;
        ensure!(rejected, "collapse certificate planted store escaped");
        println!("LAYER_COLLAPSE_CONTROL layer={layer} rejected={rejected}");
        for control in ["own-row", "wrong-carried"] {
            let mut input = layer_bf(&row.input);
            let mut carry = row.carried_pre.clone();
            if control == "own-row" {
                for v in &mut input {
                    *v = bf16::from_f32(-v.to_f32());
                }
            } else {
                for v in &mut carry {
                    *v = -*v;
                }
            }
            layer_hc_inputs(engine, &input, &carry, false)?;
            engine.run_hc_stage(0, HcStage::Pre, layer, step)?;
            let got = s.clone_dtoh(&engine.work.x.slice(..5120))?;
            let count = got
                .iter()
                .zip(&expected)
                .filter(|(a, b)| (f64::from(a.to_f32()) - f64::from(b.to_f32())).abs() > tolerance)
                .count();
            println!("LAYER_CONTROL layer={layer} control={control} beyond={count}");
            ensure!(count > 0, "layer {layer} {control} control escaped");
        }
        let mut rows = Vec::new();
        for planted in [false, true] {
            layer_ring(engine, fixture, row.position, planted)?;
            s.memcpy_htod(
                &layer_bf(&row.attn_norm),
                &mut engine.work.n.slice_mut(..5120),
            )?;
            engine.run_mixer(
                engine.model_spec.layers[layer].mixer,
                layer,
                step,
                &mut None,
            )?;
            rows.push(s.clone_dtoh(&engine.work.out.slice(..5120))?);
        }
        let tolerance = bands.bands["attn"][index].bound;
        let count = rows[0]
            .iter()
            .zip(&rows[1])
            .filter(|(a, b)| (f64::from(a.to_f32()) - f64::from(b.to_f32())).abs() > tolerance)
            .count();
        println!("LAYER_CONTROL layer={layer} control=wrong-slot beyond={count}");
        ensure!(count > 0, "layer {layer} wrong-slot control escaped");
        Ok(())
    }

    #[test]
    #[ignore = "GPU lease; FF_DS41_GPU_WINDOW=granted, FF_DS41_MODEL, FF_DS41_ENGRAM_HASH"]
    fn deepseek_expert_tier_miss_eviction_refill_gpu() -> Result<()> {
        use crate::experts::plan::Tier;

        fn fixed_output(engine: &mut Engine, ids: &[i32]) -> Result<Vec<bf16>> {
            let weights = vec![1.0 / ids.len() as f32; ids.len()];
            let input: Vec<_> = (0..5120)
                .map(|i| bf16::from_f32(((i * 17) % 251) as f32 / 64.0 - 2.0))
                .collect();
            let Engine {
                deepseek_ffn,
                moe_slots,
                work,
                device,
                ..
            } = engine;
            let ffn = deepseek_ffn.as_mut().context("DeepSeek-V4.1 FFN missing")?;
            let slots = moe_slots
                .as_mut()
                .context("DeepSeek-V4.1 expert tiers missing")?;
            let s = &device.stream;
            slots.begin_token()?;
            s.memcpy_htod(&input, &mut work.n.slice_mut(..5120))?;
            ffn.run_with_tier(DeepseekFfnRun {
                s,
                rows: 1,
                x: &work.n,
                out: &mut work.out,
                layer: 0,
                accumulator: Accumulator::Fp32,
                tier: Some(slots),
                fixed_routes: Some((ids.to_vec(), weights)),
            })?;
            slots.finish_token()?;
            Ok(s.clone_dtoh(&work.out.slice(..5120))?)
        }

        fn evict(engine: &mut Engine, target: u32) -> Result<u32> {
            engine
                .moe_slots
                .as_mut()
                .context("tiers missing")?
                .force_routes(0, &[target as i32])?;
            let mut resident_ids = Vec::new();
            {
                let slots = engine.moe_slots.as_ref().context("tiers missing")?;
                for expert in 0..384u32 {
                    if expert != target && slots.tier(0, expert)? == Tier::Gpu {
                        resident_ids.push(expert as i32);
                    }
                }
            }
            let decay = crate::experts::plan::F_DECAY;
            let chunks = resident_ids.len().div_ceil(6) as i32;
            ensure!(chunks > 0, "no other resident to warm");
            let minimum_warm_score = decay.powi(chunks - 1);
            let mut target_upper = 1.0 / (1.0 - decay);
            while target_upper >= minimum_warm_score {
                for chunk in resident_ids.chunks(6) {
                    let misses = engine
                        .moe_slots
                        .as_mut()
                        .context("tiers missing")?
                        .force_routes(0, chunk)?;
                    ensure!(misses == 0, "resident warming unexpectedly missed");
                }
                target_upper *= decay.powi(chunks);
            }
            let incoming = {
                let slots = engine.moe_slots.as_ref().context("tiers missing")?;
                (0..384u32)
                    .find(|&expert| {
                        expert != target
                            && slots.tier(0, expert).is_ok_and(|tier| tier != Tier::Gpu)
                    })
                    .context("no incoming expert for eviction")?
            };
            let moves = engine
                .moe_slots
                .as_mut()
                .context("tiers missing")?
                .force_routes(0, &[incoming as i32])?;
            ensure!(moves == 1, "target eviction resolved {moves} experts");
            ensure!(
                engine
                    .moe_slots
                    .as_ref()
                    .context("tiers missing")?
                    .tier(0, target)?
                    != Tier::Gpu,
                "target {target} remained resident"
            );
            Ok(incoming)
        }

        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "granted GPU window required"
        );
        let model_dir = PathBuf::from(std::env::var("FF_DS41_MODEL")?);
        let runtime_dir = std::env::temp_dir().join(format!(
            "flyingfish-deepseek-expert-tier-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&runtime_dir)?;
        let hash_record = PathBuf::from(std::env::var_os("FF_DS41_ENGRAM_HASH").context(
            "FF_DS41_ENGRAM_HASH missing: the accepted engram-hash-v1 record is required",
        )?);
        std::fs::copy(&hash_record, runtime_dir.join("engram-hash-v1.json"))?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(runtime_dir.clone()),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let mut engine = Engine::load(&model_dir, 0, 130, None, settings)?;
        let mut miss = None;
        let mut resident_ids = Vec::new();
        {
            let slots = engine.moe_slots.as_ref().context("tiers missing")?;
            for expert in 0..384u32 {
                if slots.tier(0, expert)? != Tier::Gpu {
                    if miss.is_none() {
                        miss = Some(expert);
                    }
                } else if expert != miss.unwrap_or(u32::MAX) {
                    resident_ids.push(expert as i32);
                }
            }
        }
        let miss = miss.context("tier plan has no initial miss")?;
        ensure!(!resident_ids.is_empty(), "no initial resident expert");
        let mut route: Vec<i32> = resident_ids.iter().copied().take(5).collect();
        route.insert(0, miss as i32);
        let misses = engine
            .moe_slots
            .as_mut()
            .context("tiers missing")?
            .force_routes(0, &[miss as i32])?;
        ensure!(
            misses == 1,
            "forced initial miss resolved {} experts",
            misses
        );
        let baseline = fixed_output(&mut engine, &route)?;

        let incoming = evict(&mut engine, miss)?;
        let evicted = engine
            .moe_slots
            .as_ref()
            .context("tiers missing")?
            .tier(0, miss)?;
        ensure!(
            evicted != Tier::Gpu,
            "forced eviction left expert {miss} resident"
        );

        let refill = fixed_output(&mut engine, &route)?;
        let differences = baseline
            .iter()
            .zip(&refill)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            differences == 0,
            "refilled expert output differs at {differences} of 5120 values"
        );

        let route_graph = {
            let Engine {
                deepseek_ffn,
                work,
                device,
                ..
            } = &mut engine;
            let ffn = deepseek_ffn.as_mut().context("DeepSeek-V4.1 FFN missing")?;
            let s = &device.stream;
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            ffn.route_only(s, 1, &work.n, 0)?;
            s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("empty DeepSeek route graph")?
        };
        route_graph
            .launch()
            .context("cuGraphLaunch DeepSeek route")?;
        let graph_ids = {
            let Engine {
                deepseek_ffn,
                device,
                ..
            } = &mut engine;
            device.stream.synchronize()?;
            device.stream.clone_dtoh(
                &deepseek_ffn
                    .as_ref()
                    .context("DeepSeek-V4.1 FFN missing")?
                    .ids
                    .slice(..6),
            )?
        };
        let graph_miss = {
            let slots = engine.moe_slots.as_ref().context("tiers missing")?;
            let selected = graph_ids
                .iter()
                .copied()
                .find(|&id| slots.tier(0, id as u32).is_ok_and(|tier| tier != Tier::Gpu))
                .unwrap_or(graph_ids[0]);
            u32::try_from(selected)?
        };
        let graph_incoming = evict(&mut engine, graph_miss)?;
        let graph_refills = engine
            .moe_slots
            .as_mut()
            .context("tiers missing")?
            .force_routes(0, &graph_ids)?;
        {
            let Engine {
                deepseek_ffn,
                moe_slots,
                device,
                ..
            } = &mut engine;
            let ffn = deepseek_ffn.as_mut().context("DeepSeek-V4.1 FFN missing")?;
            let slots = moe_slots.as_mut().context("tiers missing")?;
            slots.wait_refill(&device.stream)?;
            ffn.sync_tier(&device.stream, 0, slots)?;
        }
        let decode_graph = {
            let Engine {
                deepseek_ffn,
                work,
                device,
                ..
            } = &mut engine;
            let ffn = deepseek_ffn.as_mut().context("DeepSeek-V4.1 FFN missing")?;
            let s = &device.stream;
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            ffn.run(s, 1, &work.n, &mut work.out, 0, Accumulator::Fp32)?;
            s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("empty DeepSeek tier decode graph")?
        };
        decode_graph
            .launch()
            .context("cuGraphLaunch DeepSeek tier decode")?;
        let graph_baseline = {
            let Engine { work, device, .. } = &mut engine;
            device.stream.synchronize()?;
            device.stream.clone_dtoh(&work.out.slice(..5120))?
        };
        let replay_incoming = evict(&mut engine, graph_miss)?;
        let replay_refills = engine
            .moe_slots
            .as_mut()
            .context("tiers missing")?
            .force_routes(0, &graph_ids)?;
        {
            let Engine {
                deepseek_ffn,
                moe_slots,
                device,
                ..
            } = &mut engine;
            let ffn = deepseek_ffn.as_mut().context("DeepSeek-V4.1 FFN missing")?;
            let slots = moe_slots.as_mut().context("tiers missing")?;
            slots.wait_refill(&device.stream)?;
            ffn.sync_tier(&device.stream, 0, slots)?;
        }
        decode_graph
            .launch()
            .context("cuGraphLaunch DeepSeek tier refill replay")?;
        let graph_refill = {
            let Engine { work, device, .. } = &mut engine;
            device.stream.synchronize()?;
            device.stream.clone_dtoh(&work.out.slice(..5120))?
        };
        let graph_differences = graph_baseline
            .iter()
            .zip(&graph_refill)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            graph_differences == 0,
            "graph tier replay differs at {graph_differences} of 5120 values"
        );
        let (saved, wrong) = {
            let ffn = engine.deepseek_ffn.as_mut().context("FFN missing")?;
            let bound = ffn.layers[0].as_mut().context("layer 0 missing")?;
            let table = bound.slots.as_mut().context("slot table missing")?;
            let s = &engine.device.stream;
            let saved = s.clone_dtoh(table)?;
            let mut wrong = saved.clone();
            let dst = graph_ids[0] as usize * 6;
            let src = graph_ids[1] as usize * 6;
            ensure!(dst != src, "wrong-slot control needs distinct experts");
            wrong[dst..dst + 6].copy_from_slice(&saved[src..src + 6]);
            s.memcpy_htod(&wrong, table)?;
            (saved, wrong)
        };
        ensure!(saved != wrong, "wrong-slot control did not change pointers");
        decode_graph.launch()?;
        let corrupted = engine
            .device
            .stream
            .clone_dtoh(&engine.work.out.slice(..5120))?;
        let wrong_slot = graph_baseline
            .iter()
            .zip(&corrupted)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(wrong_slot > 0, "wrong-slot discriminator escaped");
        {
            let ffn = engine.deepseek_ffn.as_mut().context("FFN missing")?;
            let table = ffn.layers[0]
                .as_mut()
                .context("layer 0 missing")?
                .slots
                .as_mut()
                .context("slot table missing")?;
            engine.device.stream.memcpy_htod(&saved, table)?;
        }
        println!("DEEPSEEK_TIER_CONTROL wrong_slot_differences={wrong_slot}/5120");
        println!(
            "PASS DEEPSEEK_EXPERT_TIER_GATE miss={miss} incoming={incoming} evicted_tier={evicted:?} eager=0/5120 graph_miss={graph_miss} graph_incoming={graph_incoming}/{replay_incoming} graph_refills={graph_refills}/{replay_refills} graph=0/5120"
        );
        drop(engine);
        std::fs::remove_dir_all(&runtime_dir)?;
        Ok(())
    }

    // GPU gate: run in a granted window with FF_DS41_GPU_WINDOW=granted via
    //   cargo test deepseek_hc_stages_match_oracle --lib -- --exact --ignored --test-threads=1
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_DS41_GPU_WINDOW=granted"]
    fn deepseek_hc_stages_match_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))?;
        let spec = crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null)?;
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        // The mHC stream RMS epsilon comes from rms_norm_eps (1e-20 here).
        ensure!(
            spec.rms_eps == 1e-20,
            "fixture rms_norm_eps {}",
            spec.rms_eps
        );
        const ROWS: usize = 3;
        const MULT: usize = 4;
        const HIDDEN: usize = 5120;
        const MIX_WIDTH: usize = MULT * HIDDEN;
        let mix_width = MULT * HIDDEN;
        let oracle = Oracle {
            mult: MULT,
            hidden: HIDDEN,
            iters: ds.hc_sinkhorn_iters as usize,
            eps: f64::from(ds.hc_eps),
            norm_eps: f64::from(spec.rms_eps),
        };
        let device = Device::new(0)?;
        let s = &device.stream;
        let attn = fixture(7, MULT, HIDDEN, ROWS);
        let ffn = fixture(11, MULT, HIDDEN, ROWS);
        let table: Vec<bf16> = (0..8 * HIDDEN)
            .map(|i| bf16::from_f32(pattern(3 + i, 0.11) as f32))
            .collect();
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, spec.swiglu_rounding())?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "deepseek hc stages gate".into(),
        ));
        let runtime_dir =
            std::env::temp_dir().join(format!("flyingfish-deepseek-hc-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;
        let blas = Blas::new(&device, &ops, &runtime_dir, closure.clone())?;
        let state = crate::backend::cuda::state::StateStore::new(&spec, 8, s)?;
        let work = Work::new(s, &spec, ROWS)?;
        let mut runtime = HcRuntime::new(&device.ctx, s, deepseek_hc_binding(&spec)?, 1, ROWS)?;
        runtime.bind_layer(
            0,
            uploaded_sublayer(&attn, &device.upload)?,
            uploaded_sublayer(&ffn, &device.upload)?,
        )?;
        let mut engine = Engine {
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            flint_window_capture: None,
            hc: Some(runtime),
            glm_dsa: None,
            deepseek_mixer: None,
            deepseek_ffn: None,
            deepseek_engram: None,
            qwen: None,
            recurrent: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            moe_test_ffn: None,
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            weights: Weights {
                embed: testing::upload(&device.upload, table.clone())?,
                layers: Vec::new(),
                norm: None,
                head: s.alloc_zeros(1)?,
                bytes: 0,
            },
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash: None,
            decode: None,
            logits_f32: None,
            qwen_device_prefix: false,
            forward_rows: (1..=ROWS).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: Default::default(),
            spec_graph: false,
            tree_builder: TreeBuilder::Waves,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime_dir.clone(),
            spec: None,
            hidden_capture: None,
            device: device.clone(),
            config: spec.config(),
            model_spec: spec.clone(),
            capacity: 8,
            chunk: ROWS,
            workspace_rows: ROWS,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        device.finish_upload()?;
        let step = Step {
            rows: ROWS,
            input: Input::Ids,
            head: Head::None,
            mask: Mask::Causal,
        };
        let ids = [0u32, 3, 5];
        s.memcpy_htod(&ids, &mut engine.work.ids.slice_mut(..ROWS))?;
        engine.run_operator(Operator::Embed, 0, step, &mut None)?;
        let stream0: Vec<bf16> = s.clone_dtoh(&engine.hc.as_ref().unwrap().stream)?;
        let carry0: Vec<f32> = s.clone_dtoh(engine.hc.as_ref().unwrap().carry.as_ref().unwrap())?;
        for row in 0..ROWS {
            for (j, got) in stream0[row * mix_width..(row + 1) * mix_width]
                .chunks(HIDDEN)
                .enumerate()
            {
                let expected = &table[ids[row] as usize * HIDDEN..(ids[row] as usize + 1) * HIDDEN];
                ensure!(
                    got.iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "begin_forward stream copy {row}.{j} is not the embedded token"
                );
            }
            ensure!(
                carry0[row * MULT..row * MULT + MULT] == [1.0f32, 0.0, 0.0, 0.0],
                "begin_forward carry {row} is not the one-hot pre-mix"
            );
        }

        engine.run_hc_stage(0, HcStage::Pre, 0, step)?;
        let pre0: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_pre)?;
        let post0: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_post)?;
        let comb0: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_comb)?;
        let attn_pre0: Vec<f32> =
            s.clone_dtoh(engine.hc.as_ref().unwrap().attn_pre.as_ref().unwrap())?;
        let x0: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..ROWS * HIDDEN))?;
        for row in 0..ROWS {
            let stream = &stream0[row * mix_width..(row + 1) * mix_width];
            let mixes = oracle.mixes(stream, &attn.fn_w, &attn.base, &attn.scale);
            for j in 0..MULT {
                let got = pre0[row * MULT + j];
                ensure!(
                    (f64::from(got) - mixes.pre[j]).abs() <= PRE_ABS_TOL,
                    "attn pre row={row} j={j} {got:e} != {:e}",
                    mixes.pre[j]
                );
                let got = post0[row * MULT + j];
                ensure!(
                    (f64::from(got) - mixes.post[j]).abs() <= POST_ABS_TOL,
                    "attn post row={row} j={j} {got:e} != {:e}",
                    mixes.post[j]
                );
                ensure!(
                    attn_pre0[row * MULT + j].to_bits() == pre0[row * MULT + j].to_bits(),
                    "attention pre-mix carry row={row} j={j} was not updated"
                );
            }
            for k in 0..MULT * MULT {
                let got = comb0[row * MULT * MULT + k];
                ensure!(
                    (f64::from(got) - mixes.comb[k]).abs() <= COMB_ABS_TOL,
                    "attn comb row={row} k={k} {got:e} != {:e}",
                    mixes.comb[k]
                );
            }
            // Stage 0 collapses with the CARRIED one-hot pre-mix, not this
            // sublayer's own pre (model.py hc_pre(x, pre_mix)).
            let carried_pre: Vec<f64> = carry0[row * MULT..row * MULT + MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let collapse = oracle.collapse(stream, &carried_pre);
            for h in 0..HIDDEN {
                let got = x0[row * HIDDEN + h];
                let distance = bf16_ulp_distance(got, collapse[h]);
                let abs = (f64::from(got.to_f32()) - f64::from(collapse[h].to_f32())).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "attn collapse row={row} h={h} bf16 ulp distance {distance} abs {abs}"
                );
            }
        }

        s.memcpy_htod(&attn.out, &mut engine.work.out.slice_mut(..ROWS * HIDDEN))?;
        engine.run_hc_stage(3, HcStage::Post, 0, step)?;
        let stream1: Vec<bf16> = s.clone_dtoh(&engine.hc.as_ref().unwrap().stream)?;
        for row in 0..ROWS {
            let residual = &stream0[row * mix_width..(row + 1) * mix_width];
            let sublayer = &attn.out[row * HIDDEN..(row + 1) * HIDDEN];
            let kernel_post: Vec<f64> = post0[row * MULT..row * MULT + MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let kernel_comb: Vec<f64> = comb0[row * MULT * MULT..(row + 1) * MULT * MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let update = oracle.update(sublayer, residual, &kernel_post, &kernel_comb);
            for k in 0..MULT * HIDDEN {
                let got = stream1[row * mix_width + k];
                let distance = bf16_ulp_distance(got, update[k]);
                let abs = (f64::from(got.to_f32()) - f64::from(update[k].to_f32())).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "attn update row={row} k={k} bf16 ulp distance {distance} abs {abs}"
                );
            }
        }

        engine.run_hc_stage(4, HcStage::Pre, 0, step)?;
        let pre4: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_pre)?;
        let post4: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_post)?;
        let comb4: Vec<f32> = s.clone_dtoh(&engine.hc.as_ref().unwrap().mix_comb)?;
        let carry4: Vec<f32> = s.clone_dtoh(engine.hc.as_ref().unwrap().carry.as_ref().unwrap())?;
        let x4: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..ROWS * HIDDEN))?;
        for row in 0..ROWS {
            let stream = &stream1[row * mix_width..(row + 1) * mix_width];
            let mixes = oracle.mixes(stream, &ffn.fn_w, &ffn.base, &ffn.scale);
            for j in 0..MULT {
                let got = pre4[row * MULT + j];
                ensure!(
                    (f64::from(got) - mixes.pre[j]).abs() <= PRE_ABS_TOL,
                    "ffn pre row={row} j={j} {got:e} != {:e}",
                    mixes.pre[j]
                );
                let got = post4[row * MULT + j];
                ensure!(
                    (f64::from(got) - mixes.post[j]).abs() <= POST_ABS_TOL,
                    "ffn post row={row} j={j} {got:e} != {:e}",
                    mixes.post[j]
                );
                ensure!(
                    carry4[row * MULT + j].to_bits() == pre4[row * MULT + j].to_bits(),
                    "ffn pre-mix carry row={row} j={j} was not updated"
                );
            }
            for k in 0..MULT * MULT {
                let got = comb4[row * MULT * MULT + k];
                ensure!(
                    (f64::from(got) - mixes.comb[k]).abs() <= COMB_ABS_TOL,
                    "ffn comb row={row} k={k} {got:e} != {:e}",
                    mixes.comb[k]
                );
            }
            // The FFN collapse consumes the attention's own pre-mix.
            let kernel_attn_pre: Vec<f64> = attn_pre0[row * MULT..row * MULT + MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let collapse = oracle.collapse(stream, &kernel_attn_pre);
            for h in 0..HIDDEN {
                let got = x4[row * HIDDEN + h];
                let distance = bf16_ulp_distance(got, collapse[h]);
                let abs = (f64::from(got.to_f32()) - f64::from(collapse[h].to_f32())).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "ffn collapse row={row} h={h} bf16 ulp distance {distance} abs {abs}"
                );
            }
        }

        s.memcpy_htod(&ffn.out, &mut engine.work.out.slice_mut(..ROWS * HIDDEN))?;
        engine.run_hc_stage(7, HcStage::Post, 0, step)?;
        let stream2: Vec<bf16> = s.clone_dtoh(&engine.hc.as_ref().unwrap().stream)?;
        for row in 0..ROWS {
            let residual = &stream1[row * mix_width..(row + 1) * mix_width];
            let sublayer = &ffn.out[row * HIDDEN..(row + 1) * HIDDEN];
            let kernel_post: Vec<f64> = post4[row * MULT..row * MULT + MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let kernel_comb: Vec<f64> = comb4[row * MULT * MULT..(row + 1) * MULT * MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let update = oracle.update(sublayer, residual, &kernel_post, &kernel_comb);
            for k in 0..MULT * HIDDEN {
                let got = stream2[row * mix_width + k];
                let distance = bf16_ulp_distance(got, update[k]);
                let abs = (f64::from(got.to_f32()) - f64::from(update[k].to_f32())).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "ffn update row={row} k={k} bf16 ulp distance {distance} abs {abs}"
                );
            }
        }

        // Planted control (model.py:978-980): the FFN HcPre collapses with the
        // carried (layer-entry, one-hot) pre-mix instead of the attention's own.
        let runtime = engine.hc.as_mut().context("runtime missing")?;
        s.memcpy_htod(&stream1, &mut runtime.stream.slice_mut(..ROWS * mix_width))?;
        let mut one_hot = vec![0.0f32; ROWS * MULT];
        for row in one_hot.chunks_mut(MULT) {
            row[0] = 1.0;
        }
        s.memcpy_htod(
            &one_hot,
            &mut runtime.carry.as_mut().unwrap().slice_mut(..ROWS * MULT),
        )?;
        runtime.hc_pre(
            s,
            0,
            HcSublayer::Ffn,
            HcPreSource::Carried,
            ROWS,
            &mut engine.work.x,
        )?;
        let x_control: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..ROWS * HIDDEN))?;
        let mut caught = 0usize;
        for i in 0..ROWS * HIDDEN {
            let distance = bf16_ulp_distance(x_control[i], x4[i]);
            let abs = (f64::from(x_control[i].to_f32()) - f64::from(x4[i].to_f32())).abs();
            if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                caught += 1;
            }
        }
        ensure!(
            caught > 0,
            "planted carry-swap control escaped ({caught} of {} elements beyond tolerance)",
            ROWS * HIDDEN
        );

        // Policy control (GLM semantics, modeling_glm5_next.py:1334-1336): the
        // FFN HcPre collapsed with the sublayer's own just-computed pre.
        let runtime = engine.hc.as_mut().context("runtime missing")?;
        runtime.hc_pre(
            s,
            0,
            HcSublayer::Ffn,
            HcPreSource::Own,
            ROWS,
            &mut engine.work.x,
        )?;
        let x_own: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..ROWS * HIDDEN))?;
        let mut caught = 0usize;
        for row in 0..ROWS {
            let stream = &stream1[row * mix_width..(row + 1) * mix_width];
            let kernel_pre: Vec<f64> = pre4[row * MULT..row * MULT + MULT]
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let expected = oracle.collapse(stream, &kernel_pre);
            for h in 0..HIDDEN {
                let got = x_own[row * HIDDEN + h];
                let distance = bf16_ulp_distance(got, expected[h]);
                let abs = (f64::from(got.to_f32()) - f64::from(expected[h].to_f32())).abs();
                ensure!(
                    distance <= OUTPUT_ULP_TOL || abs <= OUTPUT_ABS_TOL,
                    "own-policy ffn collapse row={row} h={h} bf16 ulp distance {distance} abs {abs}"
                );
                let distance = bf16_ulp_distance(got, x4[row * HIDDEN + h]);
                let abs =
                    (f64::from(got.to_f32()) - f64::from(x4[row * HIDDEN + h].to_f32())).abs();
                if distance > OUTPUT_ULP_TOL && abs > OUTPUT_ABS_TOL {
                    caught += 1;
                }
            }
        }
        ensure!(
            caught > 0,
            "planted own-vs-attention-own policy control escaped ({caught} of {} elements beyond tolerance)",
            ROWS * HIDDEN
        );

        // The nine-stage program dispatches its Hc arm, then Norm needs the
        // FFN runtime's bound weights and stops there.
        engine.run_operator(Operator::Embed, 0, step, &mut None)?;
        let error = engine
            .run_layer_program(0, step, &mut None)
            .expect_err("the layer program must stop at the first unbound stage");
        ensure!(
            error
                .to_string()
                .contains("DeepSeek-V4.1 FFN runtime is not bound"),
            "layer program stopped elsewhere: {error}"
        );
        // F5: the model-final head collapse uses the in-flight sublayer's own
        // pre (model.py:1271); the unweighted GLM-style mean must differ.
        {
            let stream_vals: Vec<bf16> = (0..ROWS * MIX_WIDTH)
                .map(|i| bf16::from_f32((pattern(91 + i, 7.7) * 0.75) as f32))
                .collect();
            let pre_vals: Vec<f32> = (0..ROWS * MULT)
                .map(|i| (pattern(53 + i, 1.1) * 0.5 + 0.5) as f32)
                .collect();
            {
                let runtime = engine.hc.as_mut().unwrap();
                s.memcpy_htod(
                    &stream_vals,
                    &mut runtime.stream.slice_mut(..ROWS * MIX_WIDTH),
                )?;
                s.memcpy_htod(&pre_vals, &mut runtime.mix_pre.slice_mut(..ROWS * MULT))?;
            }
            let runtime = engine.hc.as_ref().unwrap();
            let mut collapsed = engine.work.x.clone();
            runtime.collapse_final(s, ROWS, &mut collapsed)?;
            let got: Vec<bf16> = s.clone_dtoh(&collapsed.slice(..ROWS * HIDDEN))?;
            let mut mean_diffs = 0usize;
            for row in 0..ROWS {
                for h in 0..HIDDEN {
                    let mut acc = 0.0f64;
                    for j in 0..MULT {
                        acc += f64::from(pre_vals[row * MULT + j])
                            * f64::from(stream_vals[row * MIX_WIDTH + j * HIDDEN + h].to_f32());
                    }
                    let want = bf16::from_f64(acc);
                    let mean = (0..MULT)
                        .map(|j| f64::from(stream_vals[row * MIX_WIDTH + j * HIDDEN + h].to_f32()))
                        .sum::<f64>()
                        / MULT as f64;
                    ensure!(
                        got[row * HIDDEN + h].to_bits() == want.to_bits(),
                        "final collapse row={row} h={h} wrong"
                    );
                    if (f64::from(got[row * HIDDEN + h].to_f32()) - mean).abs() > 1e-6 {
                        mean_diffs += 1;
                    }
                }
            }
            ensure!(
                mean_diffs > 0,
                "final collapse matches the unweighted mean at every element"
            );
        }
        // NextNorm at a non-final layer is a no-op that leaves work.n zeros.
        engine.run_operator(Operator::NextNorm, 0, step, &mut None)?;
        let n_after: Vec<bf16> = s.clone_dtoh(&engine.work.n.slice(..ROWS * HIDDEN))?;
        ensure!(
            n_after.iter().all(|v| v.to_bits() == bf16::ZERO.to_bits()),
            "non-final NextNorm touched the normed buffer"
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(&runtime_dir);
        Ok(())
    }

    #[test]
    fn deepseek_mixer_geometry_and_rope_derive_from_the_config() {
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))
        .unwrap();
        let spec =
            crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null).unwrap();
        let ds = spec.deepseek_v41.as_ref().unwrap();
        let geometry = MixerGeometry::new(&spec).unwrap();
        // Derived dims against the manifest (config.json text_config): heads
        // 64, head_dim 512, qk_rope_head_dim 64, q_lora_rank 1280,
        // o_lora_rank 1024, o_groups 8, sliding_window 128, hidden 5120.
        assert_eq!(
            (
                geometry.hidden,
                geometry.heads,
                geometry.head_dim,
                geometry.window,
                geometry.rotary,
                geometry.q_lora,
                geometry.o_rank,
                geometry.o_groups,
                geometry.o_in
            ),
            (5120, 64, 512, 128, 64, 1280, 1024, 8, 4096)
        );
        let q_width = geometry.heads * geometry.head_dim;
        // Stored layouts: wq_a [q_lora, hidden], wq_b [q_width, q_lora], wkv
        // [head_dim, hidden], attn_sink [heads] f32, wo_a
        // [o_groups*o_rank, q_width/o_groups] E4M3 codes with one E8M0 scale
        // per 32x32 block, wo_b [hidden, o_groups*o_rank] bf16.
        assert_eq!((geometry.q_lora, geometry.hidden), (1280, 5120));
        assert_eq!((q_width, geometry.q_lora), (32768, 1280));
        assert_eq!(geometry.head_dim, 512);
        assert_eq!(
            (
                geometry.o_groups * geometry.o_rank,
                q_width / geometry.o_groups
            ),
            (8192, 4096)
        );
        assert_eq!(
            (
                geometry.o_groups * geometry.o_rank / 32,
                q_width / geometry.o_groups / 32
            ),
            (256, 128)
        );
        // Ring extents: [window, head_dim] codes, [window, head_dim/32]
        // scales.
        assert_eq!((geometry.window, geometry.head_dim), (128, 512));
        assert_eq!(geometry.head_dim / 32, 16);
        // Layer kinds: layer 0 is ratio 0 (plain rope_theta 10000), the
        // compressed kinds rotate with compress_rope_theta 160000 and the
        // YaRN ramp (factor 16 over the 65536 original context).
        assert_eq!(ds.compress_ratios[0], 0);
        assert_eq!(ds.compress_ratios[2], 2);
        assert_eq!(ds.compress_ratios[40], 0);
        assert_eq!(spec.rope_theta, 10000.0);
        assert_eq!(ds.compress_rope_theta, 160000.0);
        assert_eq!(ds.rope_factor, 16.0);
        assert_eq!(ds.original_max_position, 65536);
        assert_eq!((ds.beta_fast, ds.beta_slow), (32, 1));
        let plain = rotary_inv_freq(&spec, ds, 0).unwrap();
        let yarn = rotary_inv_freq(&spec, ds, 2).unwrap();
        assert_eq!(plain.len(), spec.rotary_dim as usize / 2);
        assert_eq!(yarn.len(), plain.len());
        assert_eq!(plain[0], 1.0);
        assert!((plain[31] - 1.0f32 / 10000.0f32.powf(62.0 / 64.0)).abs() < 1e-9);
        for (i, freq) in yarn.iter().enumerate() {
            let base = 1.0f32 / 160000.0f32.powf(2.0 * i as f32 / 64.0);
            assert!(
                *freq >= base / 16.0 * (1.0 - 1e-6) && *freq <= base * (1.0 + 1e-6),
                "yarn freq {i} = {} outside [base/16, base] = [{}, {}]",
                freq,
                base / 16.0,
                base
            );
        }
    }

    // Production shapes (config.json text_config): heads 64, head_dim 512,
    // window 128, q_lora 1280, o_rank 1024, o_groups 8, hidden 5120.
    const MIX_HIDDEN: usize = 5120;
    const MIX_HEADS: usize = 64;
    const MIX_HEAD_DIM: usize = 512;
    const MIX_WINDOW: usize = 128;
    const MIX_LORA: usize = 1280;
    const MIX_O_RANK: usize = 1024;
    const MIX_O_GROUPS: usize = 8;
    const MIX_O_IN: usize = 4096;
    const MIX_Q_WIDTH: usize = MIX_HEADS * MIX_HEAD_DIM;
    const MIX_ROWS: usize = 3;
    // 131 is prime, so the position period is coprime with the window 128 and
    // the final rows straddle the ring wrap.
    const MIX_SEED_STEPS: usize = 131;

    /// BF16 ULP distance of two outputs; 1 is the accepted BF16 rounding step.
    const MIX_ULP_TOL: u32 = 1;
    /// wq_a/wkv FP32 dot: depth 15 over Σ|w·x| ≤ 5120·0.75·0.015 = 57.6 gives
    /// 15·2^-24·57.6 = 5.2e-4; the RMS amplifies by rstd ≤ 5 and |w| ≤ 1.1
    /// (both asserted in the fixture) → 2.9e-3.
    const MIX_NORM_ABS_TOL: f64 = 4e-3;
    /// wq_b FP32 dot from the matched q_norm input: 13·2^-24·Σ|q_norm·w| ≤
    /// 13·2^-24·1280·6·0.02 = 1.2e-4, plus the FP32 cos/sin at angle ≤ 133
    /// (argument rounding ≤ 8e-6 rad, |q| ≤ 8) → 2.4e-4.
    const MIX_Q_ABS_TOL: f64 = 3e-4;
    /// Score dot 12·2^-24·Σ|q·k| ≤ 12·2^-24·512·3·4 = 4.4e-3, ×512^-0.5 =
    /// 1.9e-4 score error; the softmax weights shift by ≤ 4 score errors over
    /// values ≤ 4 → 3.1e-3, plus the inverse rope 2.4e-5.
    const MIX_O_ABS_TOL: f64 = 4e-3;
    /// Group dot: 13·2^-24·Σ|o·w| ≤ 13·2^-24·4096·4·1.5 = 1.9e-2.
    const MIX_GROUP_ABS_TOL: f64 = 3e-2;
    /// wo_b FP32 dot from the matched group input: 13·2^-24·Σ|g·w| ≤
    /// 13·2^-24·8192·600·0.0015 = 5.7e-3.
    const MIX_OUT_ABS_TOL: f64 = 8e-3;

    /// E8M0FN scale: the value is 2^(code - 127); 0xff is NaN.
    fn mix_e8m0(code: u8) -> f64 {
        if code == 0xff {
            return f64::NAN;
        }
        f64::from_bits(((i64::from(code) - 127 + 1023) as u64) << 52)
    }

    /// ceil(log2(v)) of a positive normal float, from its bit pattern (the
    /// device rule in q38_contract.cuh).
    fn mix_log2_ceil(v: f32) -> i32 {
        let bits = v.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127;
        exponent + i32::from((bits & 0x7f_ffff) != 0)
    }

    /// Host mirror of the shared quantization rule (model definition
    /// kernel.py act_quant, block 32, scale_fmt ue8m0): amax over 32 floored
    /// at 1e-4, E8M0 scale 2^ceil(log2(amax/448)), E4M3-RNE codes.
    fn mix_quant32(vals: &[bf16]) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut codes = Vec::with_capacity(vals.len());
        let mut scales = Vec::with_capacity(vals.len() / 32);
        for group in vals.chunks(32) {
            let mut amax = 0.0f32;
            for v in group {
                amax = amax.max(f32::from(*v).abs());
            }
            amax = amax.max(1e-4f32);
            let exponent = mix_log2_ceil(amax * (1.0f32 / 448.0f32));
            let scale = f32::from_bits(u32::try_from(exponent + 127)? << 23);
            for v in group {
                codes.push(crate::fp8::encode_e4m3(
                    (f32::from(*v) / scale).clamp(-448.0, 448.0),
                ));
            }
            scales.push(u8::try_from(exponent + 127)?);
        }
        Ok((codes, scales))
    }

    // Decoded prepared activations: the E4M3 code times its E8M0 scale,
    // exactly representable.
    fn mix_decoded_acts(vals: &[bf16]) -> Result<Vec<bf16>> {
        let (codes, scales) = mix_quant32(vals)?;
        Ok(codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let scale = f32::from_bits(u32::from(scales[i / 32]) << 23);
                bf16::from_f32(crate::fp8::decode_e4m3(*code) * scale)
            })
            .collect())
    }

    fn mix_rms(vals: &[bf16], weight: &[bf16]) -> Vec<bf16> {
        let sum: f64 = vals
            .iter()
            .map(|v| f64::from(v.to_f32()) * f64::from(v.to_f32()))
            .sum();
        let rstd = 1.0 / (sum / vals.len() as f64 + 1e-20).sqrt();
        vals.iter()
            .zip(weight)
            .map(|(v, w)| bf16::from_f64(f64::from(v.to_f32()) * rstd * f64::from(w.to_f32())))
            .collect()
    }

    fn mix_rope_tail(vals: &mut [bf16], width: usize, inv_freq: &[f32], position: usize) {
        let start = width - 2 * inv_freq.len();
        for vector in 0..vals.len() / width {
            for (i, freq) in inv_freq.iter().enumerate() {
                let angle = position as f64 * f64::from(*freq);
                let (c, s) = (angle.cos(), angle.sin());
                let base = vector * width + start + 2 * i;
                let a = f64::from(vals[base].to_f32());
                let b = f64::from(vals[base + 1].to_f32());
                vals[base] = bf16::from_f64(a * c - b * s);
                vals[base + 1] = bf16::from_f64(a * s + b * c);
            }
        }
    }

    /// FP64 dot of a BF16 weight row into a BF16 input, rounded once to BF16
    /// (the model definition Linear).
    // Decodes one synthetic projection to exact BF16 grids (every E4M3 code
    // times its E8M0 scale is exactly representable; ERRATUM
    // 2026-10-27T17:08:12Z).
    fn mix_fp8_weights(projection: &Fp8Fixture, n: usize, k: usize) -> Vec<bf16> {
        (0..n * k)
            .map(|i| {
                let scale = f64::from(f32::from_bits(
                    u32::from(projection.scales[i / k / 32 * k.div_ceil(32) + (i % k) / 32]) << 23,
                ));
                bf16::from_f32(crate::fp8::decode_e4m3(projection.codes[i]) * scale as f32)
            })
            .collect()
    }

    /// FP64 dot of a BF16 weight row into a BF16 input, rounded once to BF16
    /// (the model definition Linear).
    fn mix_linear(weights: &[bf16], input: &[bf16], rows: usize, cols: usize) -> Vec<bf16> {
        (0..rows)
            .map(|r| {
                let row = &weights[r * cols..(r + 1) * cols];
                let acc: f64 = row
                    .iter()
                    .zip(input)
                    .map(|(w, v)| f64::from(w.to_f32()) * f64::from(v.to_f32()))
                    .sum();
                bf16::from_f64(acc)
            })
            .collect()
    }

    struct MixFixture {
        wq_a: Fp8Fixture,
        q_norm: Vec<bf16>,
        wq_b: Fp8Fixture,
        wkv: Fp8Fixture,
        kv_norm: Vec<bf16>,
        sink: Vec<f32>,
        wo_a_codes: Vec<u8>,
        wo_a_scales: Vec<u8>,
        wo_b: Fp8Fixture,
    }

    /// One projection's synthetic native storage: `[n, k]` E4M3 codes with
    /// `[ceil(n/32), ceil(k/32)]` E8M0 scales (code magnitudes <= 3.0).
    struct Fp8Fixture {
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    fn mix_fp8_fixture(n: usize, k: usize, seed: usize, base: u8) -> Fp8Fixture {
        const W_CODES: [u8; 12] = [
            0x00, 0x08, 0x10, 0x20, 0x30, 0x38, 0x3c, 0x80, 0x90, 0xb0, 0xc0, 0x44,
        ];
        Fp8Fixture {
            codes: (0..n * k)
                .map(|i| W_CODES[(i / k * 29 + i % k * 7 + seed) % W_CODES.len()])
                .collect(),
            scales: (0..n.div_ceil(32) * k.div_ceil(32))
                .map(|i| base + (i % 3) as u8)
                .collect(),
        }
    }

    fn mix_fixture(seed: usize) -> MixFixture {
        // E4M3 codes with magnitude <= 3.0, signs included.
        const W_CODES: [u8; 12] = [
            0x00, 0x08, 0x10, 0x20, 0x30, 0x38, 0x3c, 0x80, 0x90, 0xb0, 0xc0, 0x44,
        ];
        MixFixture {
            wq_a: mix_fp8_fixture(MIX_LORA, MIX_HIDDEN, seed * 29, 122),
            q_norm: (0..MIX_LORA)
                .map(|i| bf16::from_f32(1.0 + (pattern(seed * 13 + i, 2.1) * 0.1) as f32))
                .collect(),
            wq_b: mix_fp8_fixture(MIX_Q_WIDTH, MIX_LORA, seed * 37, 119),
            wkv: mix_fp8_fixture(MIX_HEAD_DIM, MIX_HIDDEN, seed * 41, 122),
            kv_norm: (0..MIX_HEAD_DIM)
                .map(|i| bf16::from_f32(1.0 + (pattern(seed * 7 + i, 4.3) * 0.1) as f32))
                .collect(),
            sink: (0..MIX_HEADS)
                .map(|h| (pattern(seed * 11 + h, 5.1) * 2.0) as f32)
                .collect(),
            wo_a_codes: (0..MIX_O_GROUPS * MIX_O_RANK * MIX_O_IN)
                .map(|i| W_CODES[(i / MIX_O_IN * 29 + i % MIX_O_IN * 7) % W_CODES.len()])
                .collect(),
            wo_a_scales: (0..MIX_O_GROUPS * MIX_O_RANK / 32 * MIX_O_IN / 32)
                .map(|i| 122 + (i % 3) as u8)
                .collect(),
            wo_b: mix_fp8_fixture(MIX_HIDDEN, MIX_O_GROUPS * MIX_O_RANK, seed * 43, 115),
        }
    }

    fn mix_x(seed: usize, position: usize) -> Vec<bf16> {
        (0..MIX_HIDDEN)
            .map(|i| bf16::from_f32((pattern(seed * 31 + position * 101 + i, 0.7) * 0.75) as f32))
            .collect()
    }

    fn mix_bytes(values: &[bf16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect()
    }

    fn mix_bf16_ulp(a: bf16, b: bf16) -> u32 {
        fn key(value: bf16) -> u32 {
            let bits = u32::from(value.to_bits());
            if bits & 0x8000 != 0 {
                (!bits) & 0xffff
            } else {
                bits | 0x8000
            }
        }
        key(a).abs_diff(key(b))
    }

    fn mix_within_tol(got: bf16, want: bf16, abs_tol: f64) -> bool {
        mix_bf16_ulp(got, want) <= MIX_ULP_TOL
            || (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs() <= abs_tol
    }

    fn mix_check(label: &str, got: &[bf16], want: &[bf16], abs_tol: f64) -> Result<usize> {
        ensure!(
            got.len() == want.len(),
            "{label} extents differ: {} != {}",
            got.len(),
            want.len()
        );
        let mut abs_branch = 0usize;
        for (index, (g, w)) in got.iter().zip(want).enumerate() {
            ensure!(
                mix_within_tol(*g, *w, abs_tol),
                "{label} element {index}: bf16 ulp distance {} abs {} (got {}, want {})",
                mix_bf16_ulp(*g, *w),
                (f64::from(g.to_f32()) - f64::from(w.to_f32())).abs(),
                g,
                w
            );
            if mix_bf16_ulp(*g, *w) > 1 {
                abs_branch += 1;
            }
        }
        Ok(abs_branch)
    }

    /// The E4M3 grid step at a decoded value (3 mantissa bits).
    fn mix_e4m3_step(value: f64) -> f64 {
        if value == 0.0 {
            return (1.0 / 512.0) * 2f64.powi(-127 + 9);
        }
        let exponent = value.abs().log2().floor() as i32;
        2f64.powi(exponent - 3)
    }

    // GPU gate: run in a granted window with FF_DS41_GPU_WINDOW=granted via
    //   cargo test deepseek_layer0_mixer_matches_oracle --lib -- --exact --ignored --test-threads=1
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_DS41_GPU_WINDOW=granted"]
    fn deepseek_layer0_mixer_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))?;
        let spec = crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null)?;
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        ensure!(ds.compress_ratios[0] == 0, "layer 0 is not a ratio-0 layer");
        let inv_freq = rotary_inv_freq(&spec, ds, 0)?;
        let fixture = mix_fixture(5);
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, spec.swiglu_rounding())?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "deepseek mixer gate".into(),
        ));
        let runtime_dir =
            std::env::temp_dir().join(format!("flyingfish-deepseek-mixer-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;
        let mut blas = Blas::new(&device, &ops, &runtime_dir, closure.clone())?;
        blas.bind_invariant(&device.ctx, MIX_ROWS)?;
        blas.enable_fp8(&device.ctx)?;
        let mut state =
            crate::backend::cuda::state::StateStore::new(&spec, MIX_SEED_STEPS + MIX_ROWS + 8, s)?;
        state.set_sequence(s, 0, MIX_ROWS as u32, MIX_ROWS as u32)?;
        let work = Work::new(s, &spec, MIX_ROWS)?;
        let mut mixer = DeepseekMixer::new(&device.ctx, s, &spec, 1, MIX_ROWS)?;
        mixer.bind_layer(
            0,
            MixerWeights {
                wq_a: Fp8Projection {
                    codes: testing::upload(&device.upload, fixture.wq_a.codes.clone())?,
                    scale: testing::upload(&device.upload, fixture.wq_a.scales.clone())?,
                },
                q_norm: testing::upload(&device.upload, mix_bytes(&fixture.q_norm))?,
                wq_b: Fp8Projection {
                    codes: testing::upload(&device.upload, fixture.wq_b.codes.clone())?,
                    scale: testing::upload(&device.upload, fixture.wq_b.scales.clone())?,
                },
                wkv: Fp8Projection {
                    codes: testing::upload(&device.upload, fixture.wkv.codes.clone())?,
                    scale: testing::upload(&device.upload, fixture.wkv.scales.clone())?,
                },
                kv_norm: testing::upload(&device.upload, mix_bytes(&fixture.kv_norm))?,
                attn_sink: testing::upload(
                    &device.upload,
                    fixture
                        .sink
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                wo_a: testing::upload(&device.upload, fixture.wo_a_codes.clone())?,
                wo_a_scale: testing::upload(&device.upload, fixture.wo_a_scales.clone())?,
                wo_b: Fp8Projection {
                    codes: testing::upload(&device.upload, fixture.wo_b.codes.clone())?,
                    scale: testing::upload(&device.upload, fixture.wo_b.scales.clone())?,
                },
            },
        )?;
        let mut engine = Engine {
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            flint_window_capture: None,
            hc: None,
            glm_dsa: None,
            deepseek_mixer: Some(mixer),
            deepseek_ffn: None,
            deepseek_engram: None,
            qwen: None,
            recurrent: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            moe_test_ffn: None,
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            weights: Weights {
                embed: testing::upload(&device.upload, vec![bf16::ZERO; 8 * MIX_HIDDEN])?,
                layers: Vec::new(),
                norm: None,
                head: s.alloc_zeros(1)?,
                bytes: 0,
            },
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash: None,
            decode: None,
            logits_f32: None,
            qwen_device_prefix: false,
            forward_rows: (1..=MIX_ROWS).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: Default::default(),
            spec_graph: false,
            tree_builder: TreeBuilder::Waves,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime_dir.clone(),
            spec: None,
            hidden_capture: None,
            device: device.clone(),
            config: spec.config(),
            model_spec: spec.clone(),
            capacity: MIX_SEED_STEPS + MIX_ROWS + 8,
            chunk: MIX_ROWS,
            workspace_rows: MIX_ROWS,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        device.finish_upload()?;
        // Seed the ring past the wrap: positions 0..130 written one row per
        // step, then the final rows 131..133 decode across the wrap.
        for position in 0..MIX_SEED_STEPS {
            let x = mix_x(5, position);
            s.memcpy_htod(&x, &mut engine.work.n.slice_mut(..MIX_HIDDEN))?;
            engine.position = position;
            engine.run_mixer(
                crate::model_spec::Mixer::FullAttention,
                0,
                Step {
                    rows: 1,
                    input: Input::Ids,
                    head: Head::None,
                    mask: Mask::Causal,
                },
                &mut None,
            )?;
        }
        let seed_ring = engine.state.latent_ring(0)?;
        let seed_ring_codes: Vec<u8> = s.clone_dtoh(&seed_ring.codes)?;
        let seed_ring_scales: Vec<u8> = s.clone_dtoh(&seed_ring.scales)?;
        let final_x: Vec<bf16> = (0..MIX_ROWS)
            .flat_map(|row| mix_x(5, MIX_SEED_STEPS + row))
            .collect();
        s.memcpy_htod(
            &final_x,
            &mut engine.work.n.slice_mut(..MIX_ROWS * MIX_HIDDEN),
        )?;
        engine.position = MIX_SEED_STEPS;
        engine.run_mixer(
            crate::model_spec::Mixer::FullAttention,
            0,
            Step {
                rows: MIX_ROWS,
                input: Input::Ids,
                head: Head::None,
                mask: Mask::Causal,
            },
            &mut None,
        )?;
        s.synchronize()?;
        let qkv: Vec<bf16> = s.clone_dtoh(
            &engine
                .work
                .qkv
                .slice(..MIX_ROWS * (MIX_Q_WIDTH + MIX_HEAD_DIM)),
        )?;
        let out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..MIX_ROWS * MIX_HIDDEN))?;
        let mixer_ref = engine.deepseek_mixer.as_ref().unwrap();
        let o: Vec<bf16> = s.clone_dtoh(&mixer_ref.o.slice(..MIX_ROWS * MIX_Q_WIDTH))?;
        let groups: Vec<bf16> = s.clone_dtoh(
            &mixer_ref
                .groups
                .slice(..MIX_ROWS * MIX_O_GROUPS * MIX_O_RANK),
        )?;
        let ring = engine.state.latent_ring(0)?;
        let ring_codes: Vec<u8> = s.clone_dtoh(&ring.codes)?;
        let ring_scales: Vec<u8> = s.clone_dtoh(&ring.scales)?;

        // Oracle: q_lora -> q_norm (the pulled buffer is post-norm).
        let mut oracle_q_norm: Vec<Vec<bf16>> = Vec::with_capacity(MIX_ROWS);
        let mut max_lora = 0.0f64;
        for row in 0..MIX_ROWS {
            let x = &final_x[row * MIX_HIDDEN..(row + 1) * MIX_HIDDEN];
            let quant_x: Vec<bf16> = mix_decoded_acts(x)?;
            let lora = mix_linear(
                &mix_fp8_weights(&fixture.wq_a, MIX_LORA, MIX_HIDDEN),
                &quant_x,
                MIX_LORA,
                MIX_HIDDEN,
            );
            max_lora = max_lora.max(
                lora.iter()
                    .map(|v| f64::from(v.to_f32()).abs())
                    .fold(0.0f64, f64::max),
            );
            let sumsq: f64 = lora
                .iter()
                .map(|v| f64::from(v.to_f32()) * f64::from(v.to_f32()))
                .sum();
            let rms = (sumsq / MIX_LORA as f64).sqrt();
            ensure!(
                rms >= 0.2,
                "fixture q_lora rms {rms} is below the frozen tolerance floor (rstd <= 5)"
            );
            oracle_q_norm.push(mix_rms(&lora, &fixture.q_norm));
        }
        ensure!(
            max_lora <= 24.0,
            "fixture wq_a output {max_lora} exceeds the frozen tolerance amplitudes"
        );
        let max_qn = oracle_q_norm
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| f64::from(v.to_f32()).abs())
            .fold(0.0f64, f64::max);
        ensure!(
            max_qn <= 6.0,
            "fixture q_norm magnitude {max_qn} exceeds the frozen tolerance amplitudes"
        );
        let q_lora_device: Vec<bf16> =
            s.clone_dtoh(&mixer_ref.q_lora.slice(..MIX_ROWS * MIX_LORA))?;
        for row in 0..MIX_ROWS {
            mix_check(
                "q_norm",
                &q_lora_device[row * MIX_LORA..(row + 1) * MIX_LORA],
                &oracle_q_norm[row],
                MIX_NORM_ABS_TOL,
            )?;
        }
        // q: wq_b + rope from the matched q_norm input (the device's own).
        let mut oracle_q: Vec<Vec<bf16>> = Vec::with_capacity(MIX_ROWS);
        for row in 0..MIX_ROWS {
            let quant_lora: Vec<bf16> =
                mix_decoded_acts(&q_lora_device[row * MIX_LORA..(row + 1) * MIX_LORA])?;
            let mut q = mix_linear(
                &mix_fp8_weights(&fixture.wq_b, MIX_Q_WIDTH, MIX_LORA),
                &quant_lora,
                MIX_Q_WIDTH,
                MIX_LORA,
            );
            mix_rope_tail(&mut q, MIX_HEAD_DIM, &inv_freq, MIX_SEED_STEPS + row);
            oracle_q.push(q);
        }
        let max_q = oracle_q
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| f64::from(v.to_f32()).abs())
            .fold(0.0f64, f64::max);
        ensure!(
            max_q <= 8.0,
            "fixture q magnitude {max_q} exceeds the frozen tolerance amplitudes"
        );
        for row in 0..MIX_ROWS {
            mix_check(
                "q post-rope",
                &qkv[row * MIX_Q_WIDTH..(row + 1) * MIX_Q_WIDTH],
                &oracle_q[row],
                MIX_Q_ABS_TOL,
            )?;
        }
        // kv path: wkv -> kv_norm -> rope -> ring quantization, from x
        // directly; the ring check below gates the pre-quant kv values.
        let mut oracle_ring: Vec<(Vec<u8>, Vec<u8>)> =
            (0..MIX_WINDOW).map(|_| (Vec::new(), Vec::new())).collect();
        let mut max_kv = 0.0f64;
        for position in 0..MIX_SEED_STEPS + MIX_ROWS {
            let x = if position < MIX_SEED_STEPS {
                mix_x(5, position)
            } else {
                final_x[(position - MIX_SEED_STEPS) * MIX_HIDDEN
                    ..(position - MIX_SEED_STEPS + 1) * MIX_HIDDEN]
                    .to_vec()
            };
            let quant_kv_x: Vec<bf16> = mix_decoded_acts(&x)?;
            let kv = mix_linear(
                &mix_fp8_weights(&fixture.wkv, MIX_HEAD_DIM, MIX_HIDDEN),
                &quant_kv_x,
                MIX_HEAD_DIM,
                MIX_HIDDEN,
            );
            let mut kv = mix_rms(&kv, &fixture.kv_norm);
            mix_rope_tail(&mut kv, MIX_HEAD_DIM, &inv_freq, position);
            max_kv = max_kv.max(
                kv.iter()
                    .map(|v| f64::from(v.to_f32()).abs())
                    .fold(0.0f64, f64::max),
            );
            let (codes, scales) = mix_quant32(&kv)?;
            oracle_ring[position % MIX_WINDOW] = (codes, scales);
        }
        ensure!(
            max_kv <= 4.0,
            "fixture kv magnitude {max_kv} exceeds the frozen tolerance amplitudes"
        );
        let max_k = oracle_ring
            .iter()
            .flat_map(|(codes, scales)| {
                codes.iter().enumerate().map(move |(d, code)| {
                    f64::from(crate::fp8::decode_e4m3(*code)) * mix_e8m0(scales[d / 32])
                })
            })
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        ensure!(
            max_k <= 4.0,
            "fixture ring value {max_k} exceeds the frozen tolerance amplitudes"
        );
        let nblocks = MIX_HEAD_DIM / 32;
        // One E4M3 grid step at the decoded value (the device and oracle may
        // straddle one rounding boundary where the pre-quant BF16 kv differs
        // by one ulp) plus one BF16 ulp at the asserted |kv| <= 4.
        for slot in 0..MIX_WINDOW {
            let (want_codes, want_scales) = &oracle_ring[slot];
            ensure!(
                !want_codes.is_empty(),
                "oracle ring slot {slot} was never written"
            );
            for d in 0..MIX_HEAD_DIM {
                let got = f64::from(crate::fp8::decode_e4m3(ring_codes[slot * MIX_HEAD_DIM + d]))
                    * mix_e8m0(ring_scales[slot * nblocks + d / 32]);
                let want = f64::from(crate::fp8::decode_e4m3(want_codes[d]))
                    * mix_e8m0(want_scales[d / 32]);
                ensure!(
                    (got - want).abs() <= mix_e4m3_step(want) + 0.032,
                    "ring slot {slot} dim {d} decodes to {got}; the oracle quantization gives {want}"
                );
            }
        }
        // Attention over the matched device q and ring (checked post-rope
        // above), sink and inverse rope from the FP64 table.
        let mut oracle_o: Vec<Vec<bf16>> = Vec::with_capacity(MIX_ROWS);
        for row in 0..MIX_ROWS {
            let position = MIX_SEED_STEPS + row;
            let count = (position + 1).min(MIX_WINDOW);
            let oldest = position + 1 - count;
            // The mixer writes q rows as one dense block, then the kv rows.
            let q_dev = &qkv[row * MIX_Q_WIDTH..(row + 1) * MIX_Q_WIDTH];
            let mut o = vec![bf16::ZERO; MIX_Q_WIDTH];
            for head in 0..MIX_HEADS {
                let q = &q_dev[head * MIX_HEAD_DIM..(head + 1) * MIX_HEAD_DIM];
                // Device ring as of this row: the seed-phase pull plus this
                // step's decode writes (later rows overwrite the oldest slots
                // that earlier rows still attend over, the wrap).
                let mut snap: Vec<(Vec<u8>, Vec<u8>)> = (0..MIX_WINDOW)
                    .map(|slot| {
                        (
                            seed_ring_codes[slot * MIX_HEAD_DIM..(slot + 1) * MIX_HEAD_DIM]
                                .to_vec(),
                            seed_ring_scales[slot * nblocks..(slot + 1) * nblocks].to_vec(),
                        )
                    })
                    .collect();
                for r in 0..=row {
                    let slot = (MIX_SEED_STEPS + r) % MIX_WINDOW;
                    snap[slot] = (
                        ring_codes[slot * MIX_HEAD_DIM..(slot + 1) * MIX_HEAD_DIM].to_vec(),
                        ring_scales[slot * nblocks..(slot + 1) * nblocks].to_vec(),
                    );
                }
                let slot = |p: usize| p % MIX_WINDOW;
                let mut scores = Vec::with_capacity(count);
                for p in oldest..=position {
                    let (codes, scales) = &snap[slot(p)];
                    let mut dot = 0.0f64;
                    for d in 0..MIX_HEAD_DIM {
                        dot += f64::from(q[d].to_f32())
                            * f64::from(crate::fp8::decode_e4m3(codes[d]))
                            * mix_e8m0(scales[d / 32]);
                    }
                    scores.push(dot * (MIX_HEAD_DIM as f64).powf(-0.5));
                }
                let sink = f64::from(fixture.sink[head]);
                let max = scores.iter().copied().fold(sink, f64::max);
                let mut denom = (sink - max).exp();
                for score in &scores {
                    denom += (score - max).exp();
                }
                let mut acc = vec![0.0f64; MIX_HEAD_DIM];
                for (i, score) in scores.iter().enumerate() {
                    let weight = (score - max).exp() / denom;
                    let p = oldest + i;
                    let (codes, scales) = &snap[slot(p)];
                    for d in 0..MIX_HEAD_DIM {
                        acc[d] += weight
                            * f64::from(crate::fp8::decode_e4m3(codes[d]))
                            * mix_e8m0(scales[d / 32]);
                    }
                }
                let tail = MIX_HEAD_DIM - 2 * inv_freq.len();
                for i in 0..inv_freq.len() {
                    let angle = position as f64 * f64::from(inv_freq[i]);
                    let (c, s) = (angle.cos(), angle.sin());
                    let a = acc[tail + 2 * i];
                    let b = acc[tail + 2 * i + 1];
                    acc[tail + 2 * i] = a * c + b * s;
                    acc[tail + 2 * i + 1] = -a * s + b * c;
                }
                for (d, value) in acc.iter().enumerate() {
                    o[head * MIX_HEAD_DIM + d] = bf16::from_f64(*value);
                }
            }
            oracle_o.push(o);
        }
        let max_o = oracle_o
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| f64::from(v.to_f32()).abs())
            .fold(0.0f64, f64::max);
        ensure!(
            max_o <= 4.0,
            "fixture attention output {max_o} exceeds the frozen tolerance amplitudes"
        );
        for row in 0..MIX_ROWS {
            mix_check(
                "attention output",
                &o[row * MIX_Q_WIDTH..(row + 1) * MIX_Q_WIDTH],
                &oracle_o[row],
                MIX_O_ABS_TOL,
            )?;
        }
        // Group projection of the matched device o (identical quantization
        // inputs on both sides), then wo_b of the matched device groups.
        let mut oracle_group_bf16: Vec<Vec<bf16>> = Vec::with_capacity(MIX_ROWS);
        for row in 0..MIX_ROWS {
            let o_row = &o[row * MIX_Q_WIDTH..(row + 1) * MIX_Q_WIDTH];
            let (codes, scales) = mix_quant32(o_row)?;
            let mut projection = vec![0.0f64; MIX_O_GROUPS * MIX_O_RANK];
            for (col, out) in projection.iter_mut().enumerate() {
                let group = col / MIX_O_RANK;
                let mut acc = 0.0f64;
                for block in 0..MIX_O_IN / 32 {
                    let scale = mix_e8m0(scales[group * (MIX_O_IN / 32) + block])
                        * mix_e8m0(fixture.wo_a_scales[col / 32 * (MIX_O_IN / 32) + block]);
                    let mut sum = 0.0f64;
                    for j in 0..32 {
                        let d = block * 32 + j;
                        sum += f64::from(crate::fp8::decode_e4m3(codes[group * MIX_O_IN + d]))
                            * f64::from(crate::fp8::decode_e4m3(
                                fixture.wo_a_codes[col * MIX_O_IN + d],
                            ));
                    }
                    acc += sum * scale;
                }
                *out = acc;
            }
            let max = projection.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            ensure!(
                max <= 600.0,
                "fixture group projection {max} exceeds the frozen tolerance amplitudes"
            );
            oracle_group_bf16.push(projection.iter().map(|v| bf16::from_f64(*v)).collect());
        }
        for row in 0..MIX_ROWS {
            mix_check(
                "group projection",
                &groups[row * MIX_O_GROUPS * MIX_O_RANK..(row + 1) * MIX_O_GROUPS * MIX_O_RANK],
                &oracle_group_bf16[row],
                MIX_GROUP_ABS_TOL,
            )?;
        }
        let mut oracle_out: Vec<Vec<bf16>> = Vec::with_capacity(MIX_ROWS);
        for row in 0..MIX_ROWS {
            let input =
                &groups[row * MIX_O_GROUPS * MIX_O_RANK..(row + 1) * MIX_O_GROUPS * MIX_O_RANK];
            let quant_input: Vec<bf16> = mix_decoded_acts(input)?;
            oracle_out.push(mix_linear(
                &mix_fp8_weights(&fixture.wo_b, MIX_HIDDEN, MIX_O_GROUPS * MIX_O_RANK),
                &quant_input,
                MIX_HIDDEN,
                MIX_O_GROUPS * MIX_O_RANK,
            ));
        }
        let max_out = oracle_out
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| f64::from(v.to_f32()).abs())
            .fold(0.0f64, f64::max);
        ensure!(
            max_out <= 32.0,
            "fixture mixer output {max_out} exceeds the frozen tolerance amplitudes"
        );
        for row in 0..MIX_ROWS {
            mix_check(
                "mixer output",
                &out[row * MIX_HIDDEN..(row + 1) * MIX_HIDDEN],
                &oracle_out[row],
                MIX_OUT_ABS_TOL,
            )?;
        }

        // Planted wrong-slot control: a foreign row lands in one visible ring
        // slot (the wrong-slot-write failure mode), re-run the final row's
        // step (its own slot is rewritten identically) and require the output
        // to move beyond the mixer-output tolerance. A two-row swap cancels
        // under near-uniform weights; a large planted row does not.
        {
            let (mut codes_host, mut scales_host) = {
                let ring = engine.state.latent_ring(0)?;
                (s.clone_dtoh(&ring.codes)?, s.clone_dtoh(&ring.scales)?)
            };
            for d in 0..MIX_HEAD_DIM {
                codes_host[10 * MIX_HEAD_DIM + d] = 0x7Eu8;
            }
            for block in 0..nblocks {
                scales_host[10 * nblocks + block] = 0x89u8;
            }
            let ring = engine.state.latent_ring_mut(0)?;
            s.memcpy_htod(&codes_host, &mut ring.codes)?;
            s.memcpy_htod(&scales_host, &mut ring.scales)?;
        }
        let x = mix_x(5, MIX_SEED_STEPS);
        s.memcpy_htod(&x, &mut engine.work.n.slice_mut(..MIX_HIDDEN))?;
        engine.position = MIX_SEED_STEPS;
        engine.run_mixer(
            crate::model_spec::Mixer::FullAttention,
            0,
            Step {
                rows: 1,
                input: Input::Ids,
                head: Head::None,
                mask: Mask::Causal,
            },
            &mut None,
        )?;
        s.synchronize()?;
        let control: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..MIX_HIDDEN))?;
        let mut caught = 0usize;
        for h in 0..MIX_HIDDEN {
            if !mix_within_tol(control[h], out[h], MIX_OUT_ABS_TOL) {
                caught += 1;
            }
        }
        ensure!(
            caught > 0,
            "planted wrong-slot ring control escaped ({caught} of {MIX_HIDDEN} elements beyond tolerance)"
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(&runtime_dir);
        Ok(())
    }

    /// The dense load partitions the manifest Text scope exactly: the
    /// per-layer mHC/mixer/FFN-shared/router tensors are resident (the four
    /// BF16-consumed projections dequantize from E4M3/E8M0), and the routed
    /// experts, sparse-attention, engram, image-token and bias_vl tensors
    /// stay deferred with their unbound runtimes.
    #[test]
    fn deepseek_dense_load_partitions_the_manifest_text_scope() {
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))
        .unwrap();
        let spec =
            crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null).unwrap();
        let mut expected = std::collections::BTreeMap::new();
        crate::checkpoint::MappedCheckpoint::deepseek_v41_entries(&spec, &mut expected).unwrap();
        let mut counts = std::collections::BTreeMap::new();
        for (name, (_, _, scope)) in &expected {
            if *scope != crate::checkpoint::TensorScope::Text {
                continue;
            }
            *counts.entry(dense_load(name)).or_insert(0usize) += 1;
        }
        let count = |class: DenseLoad| counts.get(&class).copied().unwrap_or(0);
        // 29 resident tensors x 40 layers + embed/head/norm (the four
        // projection code/scale pairs are resident natively); 2304 expert
        // tensors per layer plus 98 sparse/engram/bias_vl/image deferrals.
        assert_eq!(count(DenseLoad::Resident), 29 * 40 + 3 + 8 + 43);
        assert_eq!(count(DenseLoad::Deferred), 2304 * 40 + 98 - 8 - 43);
        let mut layer0 = std::collections::BTreeMap::new();
        for name in expected.keys() {
            if name.starts_with("layers.0.") {
                *layer0.entry(dense_load(name)).or_insert(0usize) += 1;
            }
        }
        assert_eq!(
            (
                layer0.get(&DenseLoad::Resident),
                layer0.get(&DenseLoad::Deferred)
            ),
            (Some(&29), Some(&(2304 + 1)))
        );
        // Names to weights: every bound-runtime family of every layer lands
        // in a loaded class, and the globals classify.
        for layer in 0..40 {
            let p = format!("layers.{layer}");
            for name in [
                format!("{p}.hc_attn_fn"),
                format!("{p}.hc_ffn_base"),
                format!("{p}.hc_attn_scale"),
                format!("{p}.attn.q_norm.weight"),
                format!("{p}.attn.kv_norm.weight"),
                format!("{p}.attn.attn_sink"),
                format!("{p}.attn.wo_a.weight"),
                format!("{p}.attn.wo_a.scale"),
                format!("{p}.attn_norm.weight"),
                format!("{p}.ffn_norm.weight"),
                format!("{p}.ffn.gate.weight"),
                format!("{p}.ffn.gate.bias"),
                format!("{p}.ffn.shared_experts.w1.weight"),
                format!("{p}.ffn.shared_experts.w1.scale"),
            ] {
                assert_eq!(dense_load(&name), DenseLoad::Resident, "{name}");
            }
            for role in ["wq_a", "wq_b", "wkv", "wo_b"] {
                assert_eq!(
                    dense_load(&format!("{p}.attn.{role}.weight")),
                    DenseLoad::Resident
                );
                assert_eq!(
                    dense_load(&format!("{p}.attn.{role}.scale")),
                    DenseLoad::Resident
                );
            }
            assert_eq!(
                dense_load(&format!("{p}.ffn.gate.bias_vl")),
                DenseLoad::Deferred
            );
        }
        for name in ["embed.weight", "head.weight", "norm.weight"] {
            assert_eq!(dense_load(name), DenseLoad::Resident);
        }
        for name in ["image_start", "image_end", "image_newline"] {
            assert_eq!(dense_load(name), DenseLoad::Deferred);
        }
        // Geometry vs the manifest: the resident projection codes carry the
        // MixerGeometry extents and the resident FFN tensors carry the
        // FfnGeometry extents.
        let mix = MixerGeometry::new(&spec).unwrap();
        let ffng = FfnGeometry::new(&spec).unwrap();
        let shape_of = |name: &str| expected.get(name).unwrap().1.clone();
        for (role, n, k) in [
            ("wq_a", mix.q_lora, mix.hidden),
            ("wq_b", mix.heads * mix.head_dim, mix.q_lora),
            ("wkv", mix.head_dim, mix.hidden),
            ("wo_b", mix.hidden, mix.o_groups * mix.o_rank),
        ] {
            assert_eq!(
                shape_of(&format!("layers.0.attn.{role}.weight")),
                vec![n, k],
                "{role}"
            );
        }
        let experts = ffng.experts;
        for (name, shape) in [
            ("layers.0.ffn.gate.weight", vec![experts, ffng.hidden]),
            ("layers.0.ffn.gate.bias", vec![experts]),
            (
                "layers.0.ffn.shared_experts.w1.weight",
                vec![ffng.shared_intermediate, ffng.hidden],
            ),
            (
                "layers.0.ffn.shared_experts.w2.weight",
                vec![ffng.hidden, ffng.shared_intermediate],
            ),
            ("layers.0.attn_norm.weight", vec![ffng.hidden]),
        ] {
            assert_eq!(shape_of(name), shape, "{name}");
        }
    }

    #[test]
    fn deepseek_ffn_geometry_derives_from_the_config() {
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))
        .unwrap();
        let spec =
            crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null).unwrap();
        let g = FfnGeometry::new(&spec).unwrap();
        // Derived dims against the manifest (config.json text_config): 384
        // routed experts, top-6, moe_intermediate_size 2304, one shared
        // expert (shared intermediate 2304), hidden 5120, SwiGLU limit 10,
        // routed scaling 1.5, sqrt-softplus score.
        assert_eq!(
            (
                g.hidden,
                g.experts,
                g.top_k,
                g.expert_intermediate,
                g.shared_intermediate
            ),
            (5120, 384, 6, 2304, 2304)
        );
        assert_eq!((g.limit, g.routed_scaling), (10.0, 1.5));
        assert_eq!(g.score, RouterScore::SqrtSoftplus);
        // Stored layouts: gate.weight [384,5120] bf16, gate.bias [384] f32,
        // norms [5120] bf16; shared expert E4M3 codes with [72,160]/[160,72]
        // E8M0 block-scale grids; routed experts E2M1 codes packed two per
        // byte with one E8M0 scale per 32 elements of each row.
        assert_eq!(
            g.ffn_weight_extents(),
            [
                ("gate_weight", 2 * 384 * 5120),
                ("gate_bias", 4 * 384),
                ("norm", 2 * 5120),
                ("post_norm", 2 * 5120),
                ("shared_gate_codes", 2304 * 5120),
                ("shared_gate_scales", 72 * 160),
                ("shared_up_codes", 2304 * 5120),
                ("shared_up_scales", 72 * 160),
                ("shared_down_codes", 5120 * 2304),
                ("shared_down_scales", 160 * 72),
            ]
        );
        assert_eq!(
            g.routed_weight_extents(),
            [
                ("gate_codes", 2304 * 2560),
                ("gate_scales", 2304 * 160),
                ("up_codes", 2304 * 2560),
                ("up_scales", 2304 * 160),
                ("down_codes", 5120 * 1152),
                ("down_scales", 5120 * 72),
            ]
        );
        let mut bad = spec.clone();
        bad.moe.as_mut().unwrap().score = RouterScore::Softmax;
        let error = FfnGeometry::new(&bad).unwrap_err().to_string();
        assert!(
            error.contains("sqrt-softplus"),
            "geometry error {error:?} does not name the score"
        );
        let mut bad = spec.clone();
        bad.deepseek_v41.as_mut().unwrap().swiglu_limit = 0.0;
        let error = FfnGeometry::new(&bad).unwrap_err().to_string();
        assert!(
            error.contains("SwiGLU limit"),
            "geometry error {error:?} does not name the limit"
        );
    }

    #[test]
    fn deepseek_router_weights_follow_the_unbiased_score() -> Result<()> {
        use crate::backend::cuda::moe::reference::noaux_tc_route;
        let logits = [0.25f32, -1.5, 2.0, 0.0, -0.5, 3.0, 1.25, -2.0];
        let score = |logit: f64| (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt();
        let scores: Vec<f64> = logits.iter().map(|&v| score(f64::from(v))).collect();
        let flat = [0.1f32; 8];
        let (ids, weights) = noaux_tc_route(&logits, &flat, 3, RouterScore::SqrtSoftplus, 1.5)?;
        // Top-3 biased scores descend 5, 2, 6; ties (none here) pick lower ids.
        assert_eq!(ids, vec![5, 2, 6]);
        let selected: Vec<f64> = ids.iter().map(|&id| scores[id as usize]).collect();
        let sum: f64 = selected.iter().sum::<f64>() + 1e-20;
        for (k, &weight) in weights.iter().enumerate() {
            let want = 1.5 * selected[k] / sum;
            assert!(
                (f64::from(weight) - want).abs() <= 1e-6,
                "router weight {k} {weight} != {want}"
            );
        }
        // Positive scores normalize to the routed scaling.
        let total: f32 = weights.iter().sum();
        assert!((total - 1.5).abs() <= 1e-4, "weights sum to {total}");
        // The bias steers selection only: a high bias pulls expert 0 in and
        // its weight still comes from expert 0's unbiased score.
        let mut steered = flat;
        steered[0] = 5.0;
        let (ids2, weights2) =
            noaux_tc_route(&logits, &steered, 3, RouterScore::SqrtSoftplus, 1.5)?;
        assert_eq!(ids2[0], 0);
        let sum2 = scores[0] + scores[5] + scores[2] + 1e-20;
        assert!(
            (f64::from(weights2[0]) - 1.5 * scores[0] / sum2).abs() <= 1e-6,
            "steered weight {} uses the biased score",
            weights2[0]
        );
        // Exact ties select the lower expert id.
        let tied = [1.0f32, 1.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0];
        let (ids3, weights3) = noaux_tc_route(&tied, &flat, 2, RouterScore::SqrtSoftplus, 1.5)?;
        assert_eq!(ids3, vec![0, 1]);
        assert_eq!(weights3[0].to_bits(), weights3[1].to_bits());
        Ok(())
    }

    /// The two consumers of the shared clamped-SwiGLU arm: DeepSeek-V4.1 and
    /// GLM-5.3, each with its spec-driven limit.
    fn clamped_swiglu_specs() -> Result<Vec<crate::model_spec::ModelSpec>> {
        let deepseek: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))?;
        let glm: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/data/glm53-flash-config.json"))?;
        Ok(vec![
            crate::model_spec::ModelSpec::deepseek_v41(&deepseek, &serde_json::Value::Null)?,
            crate::model_spec::ModelSpec::glm5_next(
                &glm,
                &serde_json::json!({"eos_token_id":[154820,154827,154829]}),
            )?,
        ])
    }

    // GPU gate: run in a granted window with FF_DS41_GPU_WINDOW=granted via
    //   cargo test deepseek_clamped_swiglu_matches_oracle --lib -- --exact --ignored --test-threads=1
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_DS41_GPU_WINDOW=granted"]
    fn deepseek_clamped_swiglu_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        const ROWS: usize = 3;
        const WIDTH: usize = 5120;
        // |gate|, |up| <= 1.25*limit asserted; the FP32 silu and product
        // differ from the FP64 oracle by <= 3*2^-23*silu(1.25L)*1.25L =
        // 6.5e-4 at L = 10; 1e-3 covers it 1.5x.
        const SWIGLU_ABS_TOL: f64 = 1e-3;
        let device = Device::new(0)?;
        let s = &device.stream;
        for (spec_index, spec) in clamped_swiglu_specs()?.into_iter().enumerate() {
            ensure!(
                spec.swiglu_rounding() == crate::model_spec::SwigluRounding::ClampedTwice,
                "spec {spec_index} does not select ClampedTwice"
            );
            let limit = spec.swiglu_limit();
            let common = crate::backend::cuda::ops::CommonOpsRef::new_with_swiglu_rounding(
                &device.ctx,
                spec.swiglu_rounding(),
                limit,
            )?;
            // 1.25·limit after the pattern's 0.75 factor, straddling the
            // clamp on both branches.
            let amp = 1.25 * f64::from(limit) / 0.75;
            let gate: Vec<bf16> = (0..ROWS * WIDTH)
                .map(|i| bf16::from_f32((pattern(spec_index * 101 + i, 1.7) * amp) as f32))
                .collect();
            let up: Vec<bf16> = (0..ROWS * WIDTH)
                .map(|i| bf16::from_f32((pattern(spec_index * 211 + i, 3.1) * amp) as f32))
                .collect();
            ensure!(
                f64::from(limit) > 0.0 && amp > f64::from(limit),
                "fixture amplitude {amp} does not straddle the limit {limit}"
            );
            let value = |v: bf16| f64::from(v.to_f32());
            ensure!(
                gate.iter().any(|&v| value(v) > f64::from(limit))
                    && gate.iter().any(|&v| value(v) < -f64::from(limit))
                    && up.iter().any(|&v| value(v) > f64::from(limit))
                    && up.iter().any(|&v| value(v) < -f64::from(limit)),
                "fixture never exceeds the clamp on some branch"
            );
            let meta = s.clone_htod(&[0i32, ROWS as i32, ROWS as i32, ROWS as i32, 1, 0, 0, 0])?;
            let (meta_ptr, _guard) = meta.device_ptr(s);
            let domain = RowDomain {
                meta: SeqRef { ptr: meta_ptr },
                row_base: 0,
                row_capacity: ROWS as u32,
            };
            let gb = testing::upload(&device.upload, mix_bytes(&gate))?;
            let ub = testing::upload(&device.upload, mix_bytes(&up))?;
            let mut yb = testing::zeros::<u8>(s, ROWS * WIDTH * 2)?;
            device.finish_upload()?;
            let dense3 = |slice: &cudarc::driver::CudaSlice<u8>| -> Result<Tensor<bf16, 3>> {
                weight_view::<bf16, 3>(slice, s, [ROWS as u32, 1, WIDTH as u32], "swiglu input")
            };
            common.gate.swiglu(
                s,
                domain,
                dense3(&gb)?,
                dense3(&ub)?,
                &mut TensorMut(dense3(&mut yb)?),
            )?;
            let yb_bytes: Vec<u8> = s.clone_dtoh(&yb)?;
            let got: Vec<bf16> = (0..yb_bytes.len())
                .step_by(2)
                .map(|i| {
                    bf16::from_bits(u16::from_le_bytes(
                        yb_bytes[i..i + 2].try_into().expect("bf16 bytes"),
                    ))
                })
                .collect();
            let silu = |x: f64| x / (1.0 + (-x).exp());
            let limit64 = f64::from(limit);
            let mut clamp_caught = 0usize;
            for i in 0..ROWS * WIDTH {
                let g = value(gate[i]);
                let u = value(up[i]);
                let want = bf16::from_f32(
                    bf16::from_f64(silu(g.min(limit64))).to_f32()
                        * u.clamp(-limit64, limit64) as f32,
                );
                let unclamped = bf16::from_f64(silu(g) * u);
                ensure!(
                    mix_within_tol(got[i], want, SWIGLU_ABS_TOL),
                    "clamped swiglu spec {spec_index} element {i}: ulp {} abs {} (got {}, want {})",
                    mix_bf16_ulp(got[i], want),
                    (f64::from(got[i].to_f32()) - f64::from(want.to_f32())).abs(),
                    got[i],
                    want
                );
                if !mix_within_tol(want, unclamped, SWIGLU_ABS_TOL) {
                    clamp_caught += 1;
                }
            }
            ensure!(
                clamp_caught > 0,
                "planted unclamped control escaped ({clamp_caught} of {} elements beyond tolerance)",
                ROWS * WIDTH
            );
        }
        Ok(())
    }

    // Production shapes (config.json text_config): hidden 5120, 384 experts,
    // top-6, expert/shared intermediate 2304, SwiGLU limit 10, routed
    // scaling 1.5.
    const FF_HIDDEN: usize = 5120;
    const FF_EXPERTS: usize = 384;
    const FF_TOP_K: usize = 6;
    const FF_INTER: usize = 2304;
    /// The six bound routed experts; the bias steers every row's top-6 here.
    const FF_BOUND: [usize; 6] = [17, 83, 149, 215, 301, 370];
    const FF_ROWS: usize = 3;
    /// 131 is prime, so the mutant step's position never repeats a fixture phase.
    const FF_SEED_STEPS: usize = 131;
    /// The router bias gap: bound experts +12, the rest -12. The oracle
    /// asserts |logit| <= 12 (scores <= 3.5), so the 6th/7th biased-score
    /// margin is at least 24 - 2*3.5 = 17; the gate asserts >= 1.0 against
    /// the FP32 route noise 0.5*3e-4 = 1.5e-4.
    const FF_BIAS: f32 = 12.0;
    /// PostNorm FP32 rsqrt/multiply vs FP64: rstd <= 5 (rms >= 0.2 asserted)
    /// and |w| <= 1.1 (asserted) over |x| <= 0.75 (fixture) give 15*2^-24*5*
    /// 1.1*0.75 = 1.5e-3; 4e-3 covers it 2.7x.
    const FF_NORM_ABS_TOL: f64 = 4e-3;
    /// Router logits: the FP32 GEMV differs from the FP64 dot by
    /// 15*2^-24*sum|w*n| <= 15*2^-24*180 = 1.6e-4 (budget asserted); 3e-4.
    const FF_LOGITS_ABS_TOL: f64 = 3e-4;
    /// Route weights: Delta logit <= 3e-4, sqrt-softplus slope <= 0.5 gives
    /// Delta s/s <= 1.5e-4 (relative 1/2 below 0, s >= 0.83 above), so
    /// Delta weight <= 1.5*2*1.5e-4 = 4.5e-4; 1e-3 covers it 2.2x.
    const FF_ROUTE_W_ABS_TOL: f64 = 1e-3;
    /// One BF16-ulp flip of a gate/up dot matters only below the clamp
    /// (<= ulp(10) = 0.0625), moves the hidden state <= 1.3 and crosses at
    /// most one E4M3 step (<= 8 at |state| <= 101 asserted), so the down dot
    /// moves <= 8*6*2^-6 = 0.75 per flip; the FP32 down chunks add
    /// 8*2^-24*4.2e4 = 0.020 (B_down asserted) -> 0.77; 1.0.
    const FF_EXPERT_ABS_TOL: f64 = 1.0;
    /// Six contributions and the shared row within FF_EXPERT_ABS_TOL each,
    /// plus the FP32 combine 7*2^-24*7000 = 2.9e-3 (sum|v| <= 7*1000 asserted):
    /// 7.1 -> 8.0.
    const FF_FINAL_ABS_TOL: f64 = 8.0;

    /// Host mirror of the shared act_quant rule over f32 values: amax per 32
    /// floored at 1e-4, E8M0 scale 2^ceil(log2(amax/448)), E4M3-RNE codes.
    fn ffn_quant32(values: &[f32]) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut codes = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(values.len() / 32);
        for group in values.chunks(32) {
            let mut amax = 0.0f32;
            for &v in group {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4f32);
            let exponent = mix_log2_ceil(amax * (1.0f32 / 448.0f32));
            let scale = f32::from_bits(u32::try_from(exponent + 127)? << 23);
            for &v in group {
                codes.push(crate::fp8::encode_e4m3(
                    ((v as f64 / scale as f64) as f32).clamp(-448.0, 448.0),
                ));
            }
            scales.push(u8::try_from(exponent + 127)?);
        }
        Ok((codes, scales))
    }

    /// E2M1 magnitudes by nibble: sign, two exponent bits, one mantissa bit.
    const FF_E2M1: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// The nibble of the logical element k of row j (the w4a8 fixture idiom:
    /// no period shared with the 32-element scale block).
    fn ff_nibble(row: usize, k: usize) -> u8 {
        let mut hash = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (row as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 29;
        hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 32;
        (hash % 16) as u8
    }

    fn ff_packed(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let mut codes = vec![0u8; rows * cols / 2];
        for row in 0..rows {
            for i in 0..cols / 2 {
                let k = 2 * i;
                codes[row * cols / 2 + i] =
                    ff_nibble(row + seed, k) | (ff_nibble(row + seed, k + 1) << 4);
            }
        }
        codes
    }

    /// E8M0 scale codes: `base + (row*a + block*b) % 4`.
    fn ff_scales(rows: usize, blocks: usize, seed: usize, base: u8, a: usize, b: usize) -> Vec<u8> {
        (0..rows)
            .flat_map(|row| {
                (0..blocks).map(move |block| base + ((row * a + block * b + seed) % 4) as u8)
            })
            .collect()
    }

    /// The E2M1 weight of the logical element k of row j (element 2i in the
    /// low nibble of byte i).
    fn ff_weight(codes: &[u8], row_bytes: usize, row: usize, k: usize) -> f64 {
        let byte = codes[row * row_bytes + k / 2];
        let nibble = if k.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        FF_E2M1[nibble as usize]
    }

    struct FfnFixture {
        gate_weight: Vec<bf16>,
        gate_bias: Vec<f32>,
        norm: Vec<bf16>,
        post_norm: Vec<bf16>,
        routed: Vec<ExpertBytes>,
        shared: ExpertBytes,
    }

    /// w1 codes/scales then w3 codes/scales, then w2 codes/scales.
    type ExpertBytes = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    fn ffn_fixture() -> FfnFixture {
        let mut gate_bias = vec![-FF_BIAS; FF_EXPERTS];
        for &id in &FF_BOUND {
            gate_bias[id] = FF_BIAS;
        }
        // Shared-expert E4M3 codes from |decoded| <= 3.
        let e4m3 = |n: usize, phase: f64| -> Vec<u8> {
            (0..n)
                .map(|i| crate::fp8::encode_e4m3((pattern(37 + i, phase) * 3.0) as f32))
                .collect()
        };
        let blocks_h = FF_HIDDEN / 32;
        let blocks_i = FF_INTER / 32;
        FfnFixture {
            gate_weight: (0..FF_EXPERTS * FF_HIDDEN)
                .map(|i| bf16::from_f32((pattern(41 + i, 0.5) * 0.008) as f32))
                .collect(),
            gate_bias,
            norm: (0..FF_HIDDEN)
                .map(|i| bf16::from_f32(1.0 + (pattern(43 + i, 2.1) * 0.1) as f32))
                .collect(),
            post_norm: (0..FF_HIDDEN)
                .map(|i| bf16::from_f32(1.0 + (pattern(47 + i, 4.3) * 0.1) as f32))
                .collect(),
            routed: FF_BOUND
                .iter()
                .map(|&id| {
                    (
                        ff_packed(FF_INTER, FF_HIDDEN, id),
                        ff_scales(FF_INTER, blocks_h, id, 120, 3, 5),
                        ff_packed(FF_INTER, FF_HIDDEN, id + 3),
                        ff_scales(FF_INTER, blocks_h, id + 3, 120, 3, 5),
                        ff_packed(FF_HIDDEN, FF_INTER, id + 5),
                        ff_scales(FF_HIDDEN, blocks_i, id + 5, 112, 7, 3),
                    )
                })
                .collect(),
            shared: (
                e4m3(FF_INTER * FF_HIDDEN, 1.3),
                ff_scales(FF_INTER.div_ceil(32), blocks_h, 1, 120, 3, 5),
                e4m3(FF_INTER * FF_HIDDEN, 5.7),
                ff_scales(FF_INTER.div_ceil(32), blocks_h, 2, 120, 3, 5),
                e4m3(FF_HIDDEN * FF_INTER, 2.9),
                ff_scales(FF_HIDDEN.div_ceil(32), blocks_i, 3, 112, 7, 3),
            ),
        }
    }

    fn ffn_x(seed: usize, rows: usize) -> Vec<bf16> {
        (0..rows * FF_HIDDEN)
            .map(|i| bf16::from_f32((pattern(seed * 31 + i, 0.7) * 0.75) as f32))
            .collect()
    }

    // GPU gate: run in a granted window with FF_DS41_GPU_WINDOW=granted via
    //   cargo test deepseek_layer0_ffn_matches_oracle --lib -- --exact --ignored --test-threads=1
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_DS41_GPU_WINDOW=granted"]
    fn deepseek_layer0_ffn_matches_oracle() -> Result<()> {
        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/deepseek41-flash-config.json"
        ))?;
        let spec = crate::model_spec::ModelSpec::deepseek_v41(&root, &serde_json::Value::Null)?;
        let limit = spec.swiglu_limit() as f64;
        let routed_scaling = f64::from(
            spec.deepseek_v41
                .as_ref()
                .context("DeepSeek-V4.1 geometry missing")?
                .routed_scaling_factor,
        );
        ensure!(
            (routed_scaling - 1.5).abs() < 1e-6,
            "fixture routed scaling {routed_scaling} is not 1.5"
        );
        let fixture = ffn_fixture();
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, spec.swiglu_rounding())?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "deepseek ffn gate".into(),
        ));
        let runtime_dir =
            std::env::temp_dir().join(format!("flyingfish-deepseek-ffn-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;
        let blas = Blas::new(&device, &ops, &runtime_dir, closure.clone())?;
        let state = crate::backend::cuda::state::StateStore::new(&spec, FF_ROWS + 8, s)?;
        let work = Work::new(s, &spec, FF_ROWS)?;
        let mut ffn = DeepseekFfn::new(&device.ctx, s, &spec, 1, FF_ROWS)?;
        let upload = |bytes: Vec<u8>| testing::upload(&device.upload, bytes);
        let (sg, sgs, ug, ugs, dg, dgs) = &fixture.shared;
        ffn.bind_layer(
            s,
            0,
            FfnWeights {
                gate_weight: upload(mix_bytes(&fixture.gate_weight))?,
                gate_bias: upload(
                    fixture
                        .gate_bias
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                norm: upload(mix_bytes(&fixture.norm))?,
                post_norm: upload(mix_bytes(&fixture.post_norm))?,
                shared_gate_codes: upload(sg.clone())?,
                shared_gate_scales: upload(sgs.clone())?,
                shared_up_codes: upload(ug.clone())?,
                shared_up_scales: upload(ugs.clone())?,
                shared_down_codes: upload(dg.clone())?,
                shared_down_scales: upload(dgs.clone())?,
            },
        )?;
        for (slot, &id) in FF_BOUND.iter().enumerate() {
            let (gc, gs, uc, us, dc, ds) = &fixture.routed[slot];
            ffn.bind_routed(
                0,
                id,
                RoutedExpertWeights {
                    gate_codes: upload(gc.clone())?,
                    gate_scales: upload(gs.clone())?,
                    up_codes: upload(uc.clone())?,
                    up_scales: upload(us.clone())?,
                    down_codes: upload(dc.clone())?,
                    down_scales: upload(ds.clone())?,
                },
            )?;
        }
        ffn.finish_routed(s)?;
        let mut engine = Engine {
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            flint_window_capture: None,
            hc: None,
            glm_dsa: None,
            deepseek_mixer: None,
            deepseek_ffn: Some(ffn),
            deepseek_engram: None,
            qwen: None,
            recurrent: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            moe_test_ffn: None,
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            weights: Weights {
                embed: testing::upload(&device.upload, vec![bf16::ZERO; 8 * FF_HIDDEN])?,
                layers: Vec::new(),
                norm: None,
                head: s.alloc_zeros(1)?,
                bytes: 0,
            },
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash: None,
            decode: None,
            logits_f32: None,
            qwen_device_prefix: false,
            forward_rows: (1..=FF_ROWS).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: Default::default(),
            spec_graph: false,
            tree_builder: TreeBuilder::Waves,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime_dir.clone(),
            spec: None,
            hidden_capture: None,
            device: device.clone(),
            config: spec.config(),
            model_spec: spec.clone(),
            capacity: FF_SEED_STEPS + FF_ROWS,
            chunk: FF_ROWS,
            workspace_rows: FF_ROWS,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        device.finish_upload()?;
        // The production run dispatches on device (no routing data crosses to
        // host); this setup-time pull checks the dispatch tables the kernels
        // read: every bound id holds a valid slot with six non-null
        // addresses, every unbound id holds zero, and no two bound experts
        // share a slot address.
        {
            let ffn_ref = engine.deepseek_ffn.as_ref().unwrap();
            let (slots, valid) = ffn_ref.routed_slots(0)?;
            let valid_host: Vec<u32> = s.clone_dtoh(valid)?;
            ensure!(
                valid_host.len() == FF_EXPERTS,
                "dispatch valid mask holds {} of {} entries",
                valid_host.len(),
                FF_EXPERTS
            );
            ensure!(
                FF_BOUND.iter().all(|&id| valid_host[id] == 1),
                "a bound routed expert holds no dispatch slot"
            );
            ensure!(
                (0..FF_EXPERTS)
                    .filter(|id| !FF_BOUND.contains(id))
                    .all(|id| valid_host[id] == 0),
                "an unbound routed expert holds a dispatch slot"
            );
            let slots_host: Vec<u64> = s.clone_dtoh(slots)?;
            ensure!(
                FF_BOUND
                    .iter()
                    .all(|&id| (0..6).all(|field| slots_host[id * 6 + field] != 0)),
                "a bound routed expert holds a null dispatch address"
            );
            let gate_codes: Vec<u64> = FF_BOUND.iter().map(|&id| slots_host[id * 6]).collect();
            ensure!(
                gate_codes
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == FF_BOUND.len(),
                "two bound routed experts share one dispatch slot address"
            );
        }
        let silu = |x: f64| x / (1.0 + (-x).exp());
        let score_fn = |logit: f64| (logit.max(0.0) + (-logit.abs()).exp().ln_1p()).sqrt();
        let mut flip_sq_total = 0.0f64;
        let mut main_sq_total = 0.0f64;
        let mut control_elements = 0usize;
        let mut final_out: Vec<bf16> = Vec::new();
        let mut final_main_sq = 0.0f64;
        for pass in 0..2u32 {
            let rows = if pass == 0 { 1 } else { FF_ROWS };
            let step = Step {
                rows,
                input: Input::Ids,
                head: Head::None,
                mask: Mask::Causal,
            };
            engine
                .state
                .set_sequence(s, 0, rows as u32, FF_ROWS as u32)?;
            let x_res = ffn_x(5 + pass as usize, rows);
            s.memcpy_htod(&x_res, &mut engine.work.x.slice_mut(..rows * FF_HIDDEN))?;
            // Norm arm (checked on the first pass): plain RMS on x.
            if pass == 0 {
                engine.run_operator(Operator::Norm, 0, step, &mut None)?;
                let n: Vec<bf16> = s.clone_dtoh(&engine.work.n.slice(..rows * FF_HIDDEN))?;
                let lora: Vec<f64> = x_res.iter().map(|v| f64::from(v.to_f32())).collect();
                let sumsq: f64 = lora.iter().map(|v| v * v).sum();
                let rms = (sumsq / lora.len() as f64).sqrt();
                ensure!(
                    rms >= 0.2,
                    "fixture x rms {rms} is below the frozen tolerance floor (rstd <= 5)"
                );
                let rstd = 1.0 / (sumsq / lora.len() as f64 + f64::from(spec.rms_eps)).sqrt();
                let max_weight = fixture
                    .norm
                    .iter()
                    .map(|w| f64::from(w.to_f32()).abs())
                    .fold(0.0f64, f64::max);
                ensure!(
                    max_weight <= 1.1,
                    "fixture norm weight {max_weight} exceeds the frozen tolerance amplitudes"
                );
                for (i, got) in n.iter().enumerate() {
                    let want = bf16::from_f64(lora[i] * rstd * f64::from(fixture.norm[i].to_f32()));
                    ensure!(
                        mix_within_tol(*got, want, FF_NORM_ABS_TOL),
                        "norm element {i}: ulp {} abs {}",
                        mix_bf16_ulp(*got, want),
                        (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                    );
                }
            }
            // PostNorm: plain RMS on x into the FFN input.
            engine.run_operator(Operator::PostNorm, 0, step, &mut None)?;
            let n_dev: Vec<bf16> = s.clone_dtoh(&engine.work.n.slice(..rows * FF_HIDDEN))?;
            let mut max_n = 0.0f64;
            let mut oracle_post: Vec<Vec<bf16>> = Vec::with_capacity(rows);
            for row in 0..rows {
                let x = &x_res[row * FF_HIDDEN..(row + 1) * FF_HIDDEN];
                let values: Vec<f64> = x.iter().map(|v| f64::from(v.to_f32())).collect();
                let sumsq: f64 = values.iter().map(|v| v * v).sum();
                ensure!(
                    (sumsq / values.len() as f64).sqrt() >= 0.2,
                    "fixture x rms is below the frozen tolerance floor"
                );
                let rstd = 1.0 / (sumsq / values.len() as f64 + f64::from(spec.rms_eps)).sqrt();
                oracle_post.push(
                    values
                        .iter()
                        .zip(&fixture.post_norm)
                        .map(|(&v, w)| bf16::from_f64(v * rstd * f64::from(w.to_f32())))
                        .collect(),
                );
                max_n = max_n.max(
                    oracle_post[row]
                        .iter()
                        .map(|v| f64::from(v.to_f32()).abs())
                        .fold(0.0f64, f64::max),
                );
            }
            ensure!(
                max_n <= 4.2,
                "fixture normed input {max_n} exceeds the frozen router budget 4.2"
            );
            for row in 0..rows {
                mix_check(
                    "post_norm",
                    &n_dev[row * FF_HIDDEN..(row + 1) * FF_HIDDEN],
                    &oracle_post[row],
                    FF_NORM_ABS_TOL,
                )?;
            }
            // The FFN consumes the device's own normed rows.
            engine.run_ffn(0, step, &mut None)?;
            s.synchronize()?;
            let ffn_ref = engine.deepseek_ffn.as_ref().unwrap();
            let stage = ffn_ref.stage_buffers();
            let (logits_d, ids_d, weights_d, contributions_d, shared_d) = (
                stage.logits,
                stage.ids,
                stage.route_weights,
                stage.contributions,
                stage.shared_out,
            );
            let logits_dev: Vec<f32> = s.clone_dtoh(&logits_d.slice(..rows * FF_EXPERTS))?;
            let ids_dev: Vec<i32> = s.clone_dtoh(&ids_d.slice(..rows * FF_TOP_K))?;
            let weights_dev: Vec<f32> = s.clone_dtoh(&weights_d.slice(..rows * FF_TOP_K))?;
            // Router oracle over the device normed rows.
            let mut sum_budget = 0.0f64;
            let mut oracle_logits: Vec<Vec<f64>> = Vec::with_capacity(rows);
            for row in 0..rows {
                let n = &n_dev[row * FF_HIDDEN..(row + 1) * FF_HIDDEN];
                let mut row_logits = Vec::with_capacity(FF_EXPERTS);
                let mut row_abs = 0.0f64;
                for e in 0..FF_EXPERTS {
                    let weight = &fixture.gate_weight[e * FF_HIDDEN..(e + 1) * FF_HIDDEN];
                    let mut dot = 0.0f64;
                    let mut abs = 0.0f64;
                    for (w, v) in weight.iter().zip(n) {
                        let term = f64::from(w.to_f32()) * f64::from(v.to_f32());
                        dot += term;
                        abs += term.abs();
                    }
                    row_abs = row_abs.max(abs);
                    row_logits.push(dot);
                }
                sum_budget = sum_budget.max(row_abs);
                oracle_logits.push(row_logits);
            }
            ensure!(
                sum_budget <= 180.0,
                "fixture router dot budget {sum_budget} exceeds the frozen 180"
            );
            let max_logit = oracle_logits
                .iter()
                .flat_map(|row| row.iter())
                .map(|v| v.abs())
                .fold(0.0f64, f64::max);
            ensure!(
                max_logit <= 24.0,
                "fixture router logit {max_logit} exceeds the frozen margin budget 24"
            );
            for row in 0..rows {
                for e in 0..FF_EXPERTS {
                    let got = f64::from(logits_dev[row * FF_EXPERTS + e]);
                    ensure!(
                        (got - oracle_logits[row][e]).abs() <= FF_LOGITS_ABS_TOL,
                        "logits row {row} expert {e}: {got:e} != {:e}",
                        oracle_logits[row][e]
                    );
                }
            }
            // Route: biased selection, unbiased weights, norm over 6, x1.5.
            let mut margin = f64::INFINITY;
            let mut min_weight = 1.0f64;
            let mut oracle_weight: Vec<[f64; FF_EXPERTS]> = Vec::new();
            for logits in &oracle_logits {
                let scores: Vec<f64> = logits.iter().map(|&l| score_fn(l)).collect();
                let mut biased: Vec<(f64, usize)> = scores
                    .iter()
                    .enumerate()
                    .map(|(e, &s)| (s + f64::from(fixture.gate_bias[e]), e))
                    .collect();
                biased.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                margin = margin.min(biased[5].0 - biased[6].0);
                let selected = &biased[..FF_TOP_K];
                let sum: f64 = selected.iter().map(|&(_, e)| scores[e]).sum::<f64>() + 1e-20;
                let mut row_weight = [0.0f64; FF_EXPERTS];
                for &(_, e) in selected {
                    // noaux_tc: the weight uses the unbiased score (model
                    // definition); the bias selects only.
                    row_weight[e] = routed_scaling * scores[e] / sum;
                    min_weight = min_weight.min(row_weight[e]);
                }
                oracle_weight.push(row_weight);
            }
            ensure!(
                margin >= 1.0,
                "fixture top-6 margin {margin} is below the frozen detection threshold"
            );
            ensure!(
                min_weight.is_finite() && min_weight > 0.0,
                "fixture routed weight {min_weight} is not positive and finite"
            );
            for row in 0..rows {
                let mut got: Vec<usize> = (0..FF_TOP_K)
                    .map(|k| ids_dev[row * FF_TOP_K + k] as usize)
                    .collect();
                got.sort_unstable();
                let mut want: Vec<usize> = (0..FF_EXPERTS)
                    .filter(|&e| oracle_weight[row][e] > 0.0)
                    .collect();
                want.sort_unstable();
                ensure!(
                    got == want && got.iter().all(|id| FF_BOUND.contains(id)),
                    "route row {row} selected {got:?}; the oracle selects {want:?}"
                );
                for k in 0..FF_TOP_K {
                    let id = ids_dev[row * FF_TOP_K + k] as usize;
                    let got = weights_dev[row * FF_TOP_K + k];
                    let want = oracle_weight[row][id] as f32;
                    ensure!(
                        (f64::from(got) - f64::from(want)).abs() <= FF_ROUTE_W_ABS_TOL,
                        "route weight row {row} expert {id}: abs {}",
                        (f64::from(got) - f64::from(want)).abs()
                    );
                }
            }
            // Experts and shared expert against the FP64 oracle, from the
            // device's own route outputs and normed rows.
            let mut max_gate = 0.0f64;
            let mut max_state = 0.0f64;
            let mut max_b_down = 0.0f64;
            let mut max_contribution = 0.0f64;
            let mut oracle_final: Vec<Vec<f64>> = Vec::with_capacity(rows);
            for row in 0..rows {
                let n = &n_dev[row * FF_HIDDEN..(row + 1) * FF_HIDDEN];
                let x_f32: Vec<f32> = n.iter().map(|v| v.to_f32()).collect();
                let (xc, xs) = ffn_quant32(&x_f32)?;
                let blocks_h = FF_HIDDEN / 32;
                let blocks_i = FF_INTER / 32;
                let mut before: Vec<Vec<f64>> = Vec::with_capacity(FF_TOP_K + 1);
                let mut after: Vec<Vec<f64>> = Vec::with_capacity(FF_TOP_K + 1);
                for k in 0..FF_TOP_K {
                    let id = ids_dev[row * FF_TOP_K + k] as usize;
                    let rw = weights_dev[row * FF_TOP_K + k];
                    let slot = FF_BOUND.iter().position(|&b| b == id).unwrap();
                    let (gc, gs, uc, us, dc, ds) = &fixture.routed[slot];
                    // Gate/up dots in code units, then clamped SwiGLU.
                    let mut state = vec![0.0f64; FF_INTER];
                    for j in 0..FF_INTER {
                        let mut gate = 0.0f64;
                        let mut up = 0.0f64;
                        for b in 0..blocks_h {
                            let act = mix_e8m0(xs[b]);
                            let wscale = mix_e8m0(gs[j * blocks_h + b]);
                            let uscale = mix_e8m0(us[j * blocks_h + b]);
                            let mut gate_block = 0.0f64;
                            let mut up_block = 0.0f64;
                            for t in 0..32 {
                                let a = f64::from(crate::fp8::decode_e4m3(xc[b * 32 + t]));
                                let index = b * 32 + t;
                                gate_block += a * ff_weight(gc, FF_HIDDEN / 2, j, index);
                                up_block += a * ff_weight(uc, FF_HIDDEN / 2, j, index);
                            }
                            gate += gate_block * act * wscale;
                            up += up_block * act * uscale;
                        }
                        let g = f64::from(bf16::from_f64(gate).to_f32());
                        let u = f64::from(bf16::from_f64(up).to_f32());
                        max_gate = max_gate.max(g.abs()).max(u.abs());
                        let silu_bf16 = bf16::from_f64(silu(g.min(limit)));
                        let product =
                            bf16::from_f32(silu_bf16.to_f32() * u.clamp(-limit, limit) as f32);
                        state[j] = f64::from(product.to_f32());
                    }
                    max_state = max_state.max(state.iter().map(|v| v.abs()).fold(0.0f64, f64::max));
                    // Routed weight BEFORE the down-input quantization.
                    let scaled: Vec<f32> = state
                        .iter()
                        .map(|&v| bf16::from_f32((v * f64::from(rw)) as f32).to_f32())
                        .collect();
                    let (bc, bs) = ffn_quant32(&scaled)?;
                    let mut out = vec![0.0f64; FF_HIDDEN];
                    let mut b_down = 0.0f64;
                    for d in 0..FF_HIDDEN {
                        let mut acc = 0.0f64;
                        for b in 0..blocks_i {
                            let act = mix_e8m0(bs[b]);
                            let wscale = mix_e8m0(ds[d * blocks_i + b]);
                            let mut block = 0.0f64;
                            for t in 0..32 {
                                let a = f64::from(crate::fp8::decode_e4m3(bc[b * 32 + t]));
                                let term = a * ff_weight(dc, FF_INTER / 2, d, b * 32 + t);
                                block += term;
                                b_down += act * wscale * term.abs();
                            }
                            acc += block * act * wscale;
                        }
                        out[d] = f64::from(bf16::from_f64(acc).to_f32());
                    }
                    max_b_down = max_b_down.max(b_down);
                    max_contribution =
                        max_contribution.max(out.iter().map(|v| v.abs()).fold(0.0f64, f64::max));
                    // Control: the same expert with the routed weight applied
                    // AFTER the down-input quantization.
                    let (ac, asx) =
                        ffn_quant32(&state.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
                    let mut out_after = vec![0.0f64; FF_HIDDEN];
                    for d in 0..FF_HIDDEN {
                        let mut acc = 0.0f64;
                        for b in 0..blocks_i {
                            let act = mix_e8m0(asx[b]);
                            let wscale = mix_e8m0(ds[d * blocks_i + b]);
                            let mut block = 0.0f64;
                            for t in 0..32 {
                                let a = f64::from(crate::fp8::decode_e4m3(ac[b * 32 + t]));
                                block += a * ff_weight(dc, FF_INTER / 2, d, b * 32 + t);
                            }
                            acc += block * act * wscale;
                        }
                        out_after[d] =
                            f64::from(bf16::from_f32(bf16::from_f64(acc).to_f32() * rw).to_f32());
                    }
                    before.push(out);
                    after.push(out_after);
                    let got = s.clone_dtoh(&contributions_d.slice(
                        (row * FF_TOP_K + k) * FF_HIDDEN..(row * FF_TOP_K + k + 1) * FF_HIDDEN,
                    ))?;
                    mix_check(
                        "routed contribution",
                        &got,
                        &before[k]
                            .iter()
                            .map(|&v| bf16::from_f64(v))
                            .collect::<Vec<_>>(),
                        FF_EXPERT_ABS_TOL,
                    )?;
                    // Boundary control: the same expert with the routed
                    // weight applied AFTER the down-input quantization. The
                    // E4M3 grids decorrelate (the before path rounds
                    // bf16(state*rw) onto the amax(state*rw) scale, the after
                    // path rounds state onto the amax(state) scale): an
                    // RMS gap of ~4% of the contribution scale against the
                    // 2e-2 FP32 chunk noise.
                    // Self-calibrating control: the aggregate flip RMS
                    // across every expert must sit four times above the
                    // aggregate main-residual RMS; a single expert's flip can
                    // be sub-BF16-ulp when its routed weight is small.
                    let mut gap_sq = 0.0f64;
                    for h in 0..FF_HIDDEN {
                        let gap = (f64::from(got[h].to_f32()) - after[k][h]).abs();
                        gap_sq += gap * gap;
                    }
                    let mut main_sq = 0.0f64;
                    for h in 0..FF_HIDDEN {
                        let residual = f64::from(got[h].to_f32()) - before[k][h];
                        main_sq += residual * residual;
                    }
                    flip_sq_total += gap_sq;
                    main_sq_total += main_sq;
                    control_elements += FF_HIDDEN;
                }
                // Shared expert (E4M3 codes, 32x32 E8M0 weight blocks), no
                // routed weight: before and after coincide.
                let (sg, sgs, ug, ugs, dg, dgs) = &fixture.shared;
                let mut state = vec![0.0f64; FF_INTER];
                for j in 0..FF_INTER {
                    let mut gate = 0.0f64;
                    let mut up = 0.0f64;
                    for b in 0..FF_HIDDEN / 32 {
                        let act = mix_e8m0(xs[b]);
                        let wscale = mix_e8m0(sgs[(j / 32) * blocks_h + b]);
                        let uscale = mix_e8m0(ugs[(j / 32) * blocks_h + b]);
                        let mut gate_block = 0.0f64;
                        let mut up_block = 0.0f64;
                        for t in 0..32 {
                            let a = f64::from(crate::fp8::decode_e4m3(xc[b * 32 + t]));
                            let index = b * 32 + t;
                            gate_block +=
                                a * f64::from(crate::fp8::decode_e4m3(sg[j * FF_HIDDEN + index]));
                            up_block +=
                                a * f64::from(crate::fp8::decode_e4m3(ug[j * FF_HIDDEN + index]));
                        }
                        gate += gate_block * act * wscale;
                        up += up_block * act * uscale;
                    }
                    let g = f64::from(bf16::from_f64(gate).to_f32());
                    let u = f64::from(bf16::from_f64(up).to_f32());
                    max_gate = max_gate.max(g.abs()).max(u.abs());
                    let silu_bf16 = bf16::from_f64(silu(g.min(limit)));
                    state[j] = f64::from(
                        bf16::from_f32(silu_bf16.to_f32() * u.clamp(-limit, limit) as f32).to_f32(),
                    );
                }
                let (bc, bs) = ffn_quant32(&state.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
                let mut out = vec![0.0f64; FF_HIDDEN];
                for d in 0..FF_HIDDEN {
                    let mut acc = 0.0f64;
                    for b in 0..FF_INTER / 32 {
                        let act = mix_e8m0(bs[b]);
                        let wscale = mix_e8m0(dgs[(d / 32) * blocks_i + b]);
                        let mut block = 0.0f64;
                        for t in 0..32 {
                            let a = f64::from(crate::fp8::decode_e4m3(bc[b * 32 + t]));
                            block += a * f64::from(crate::fp8::decode_e4m3(
                                dg[d * FF_INTER + b * 32 + t],
                            ));
                        }
                        acc += block * act * wscale;
                    }
                    out[d] = f64::from(bf16::from_f64(acc).to_f32());
                }
                max_contribution =
                    max_contribution.max(out.iter().map(|v| v.abs()).fold(0.0f64, f64::max));
                mix_check(
                    "shared contribution",
                    &s.clone_dtoh(&shared_d.slice(row * FF_HIDDEN..(row + 1) * FF_HIDDEN))?,
                    &out.iter().map(|&v| bf16::from_f64(v)).collect::<Vec<_>>(),
                    FF_EXPERT_ABS_TOL,
                )?;
                before.push(out.clone());
                oracle_final.push(
                    (0..FF_HIDDEN)
                        .map(|h| before.iter().map(|term| term[h]).sum::<f64>())
                        .collect(),
                );
            }
            ensure!(
                max_gate <= 400.0,
                "fixture gate/up dot {max_gate} exceeds the frozen budget 400"
            );
            ensure!(
                max_gate > limit,
                "fixture gate/up dots never exceed the clamp limit"
            );
            ensure!(
                max_state <= 101.0,
                "fixture hidden state {max_state} exceeds the frozen budget 101"
            );
            ensure!(
                max_b_down <= 4.2e4,
                "fixture down dot envelope {max_b_down} exceeds the frozen budget 4.2e4"
            );
            ensure!(
                max_contribution <= 200.0,
                "fixture contribution {max_contribution} exceeds the frozen budget 200"
            );
            let sum_abs: f64 = oracle_final
                .iter()
                .flat_map(|row| row.iter())
                .map(|v| v.abs())
                .sum();
            ensure!(
                sum_abs <= 7000.0 * rows as f64,
                "fixture combine envelope {sum_abs} exceeds the frozen budget 7000 per row"
            );
            // Final combine: FP32 accumulation of the six contributions and
            // the shared row, one BF16 round.
            let out_dev: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..rows * FF_HIDDEN))?;
            if rows == FF_ROWS {
                final_out = out_dev.clone();
                let mut sq = 0.0f64;
                for row in 0..rows {
                    for h in 0..FF_HIDDEN {
                        let residual =
                            f64::from(out_dev[row * FF_HIDDEN + h].to_f32()) - oracle_final[row][h];
                        sq += residual * residual;
                    }
                }
                final_main_sq = sq;
            }
            for row in 0..rows {
                for h in 0..FF_HIDDEN {
                    let got = out_dev[row * FF_HIDDEN + h];
                    let want = bf16::from_f64(oracle_final[row][h]);
                    ensure!(
                        mix_within_tol(got, want, FF_FINAL_ABS_TOL),
                        "ffn output row {row} dim {h}: ulp {} abs {} (got {}, want {})",
                        mix_bf16_ulp(got, want),
                        (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs(),
                        got,
                        want
                    );
                }
            }
        }
        // Planted off-by-one routing mutant: expert FF_BOUND[i] launches
        // with FF_BOUND[(i+1)%6] weights; the distinct coprime-period
        // fixtures make every pairing wrong, so the output must move beyond
        // the final tolerance.
        {
            let ffn_mut = engine.deepseek_ffn.as_mut().unwrap();
            for (slot, &id) in FF_BOUND.iter().enumerate() {
                let next = &fixture.routed[(slot + 1) % FF_BOUND.len()];
                let (gc, gs, uc, us, dc, ds) = (
                    upload(next.0.clone())?,
                    upload(next.1.clone())?,
                    upload(next.2.clone())?,
                    upload(next.3.clone())?,
                    upload(next.4.clone())?,
                    upload(next.5.clone())?,
                );
                ffn_mut.bind_routed(
                    0,
                    id,
                    RoutedExpertWeights {
                        gate_codes: gc,
                        gate_scales: gs,
                        up_codes: uc,
                        up_scales: us,
                        down_codes: dc,
                        down_scales: ds,
                    },
                )?;
            }
            ffn_mut.finish_routed(s)?;
        }
        let mutant_x: Vec<bf16> = (0..FF_ROWS)
            .flat_map(|row| mix_x(7, FF_SEED_STEPS + row))
            .collect();
        s.memcpy_htod(
            &mutant_x,
            &mut engine.work.x.slice_mut(..FF_ROWS * FF_HIDDEN),
        )?;
        let save_position = engine.position;
        engine.position = FF_SEED_STEPS;
        let step = Step {
            rows: FF_ROWS,
            input: Input::Ids,
            head: Head::None,
            mask: Mask::Causal,
        };
        engine.run_operator(crate::backend::Operator::Norm, 0, step, &mut None)?;
        let Engine {
            deepseek_ffn,
            device,
            work,
            ..
        } = &mut engine;
        deepseek_ffn.as_mut().unwrap().run(
            &device.stream,
            step.rows,
            &work.n,
            &mut work.out,
            0,
            Accumulator::Fp32,
        )?;
        engine.position = save_position;
        let mutant_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..FF_ROWS * FF_HIDDEN))?;
        // Self-calibrating: the off-by-one pairing moves the output by many
        // times the main comparison's own residual (the elementwise tolerance
        // is coarser than one flipped expert pairing by construction).
        let mut mutant_sq = 0.0f64;
        for i in 0..FF_ROWS * FF_HIDDEN {
            let delta = f64::from(mutant_out[i].to_f32()) - f64::from(final_out[i].to_f32());
            mutant_sq += delta * delta;
        }
        let elements = (FF_ROWS * FF_HIDDEN) as f64;
        let mutant_rms = (mutant_sq / elements).sqrt();
        let main_rms = (final_main_sq / elements).sqrt();
        ensure!(
            mutant_rms >= 4.0 * main_rms.max(1e-9),
            "planted off-by-one routing control escaped: mutant RMS {mutant_rms} vs main residual RMS {main_rms}"
        );
        let flip_rms = (flip_sq_total / control_elements as f64).sqrt();
        let main_rms = (main_sq_total / control_elements as f64).sqrt();
        ensure!(
            flip_rms >= 4.0 * main_rms.max(1e-9),
            "routed-weight boundary control escaped: the after-quant oracle sits {flip_rms} RMS from the device, only {:.2}x the main residual RMS {main_rms}",
            flip_rms / main_rms.max(1e-9)
        );
        // Both accumulator boundaries are live: the BF16-routed-sum mode
        // rounds the routed sum once more before the shared join, so every
        // element sits within two BF16 roundings of the FP32 mode. The arm
        // runs after the off-by-one rebind, so the comparison baseline is
        // the mutant output.
        {
            let Engine {
                deepseek_ffn,
                device,
                work,
                ..
            } = &mut engine;
            deepseek_ffn.as_mut().unwrap().run(
                &device.stream,
                FF_ROWS,
                &work.n,
                &mut work.out,
                0,
                Accumulator::Bf16,
            )?;
        }
        let bf16_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..FF_ROWS * FF_HIDDEN))?;
        // The BF16 boundary adds one rounding of the routed sum (magnitude
        // per element = sum_k contributions) and one of the output; the
        // The boundary difference is the two extra BF16 roundings of the
        // routed sum and the shared join; it must be visible (the arm is
        // live) yet stay an order below the output scale.
        let mut boundary_sq = 0.0f64;
        let mut output_sq = 0.0f64;
        for i in 0..FF_ROWS * FF_HIDDEN {
            let delta = f64::from(bf16_out[i].to_f32()) - f64::from(mutant_out[i].to_f32());
            boundary_sq += delta * delta;
            output_sq += f64::from(mutant_out[i].to_f32()) * f64::from(mutant_out[i].to_f32());
        }
        let elements = (FF_ROWS * FF_HIDDEN) as f64;
        let boundary_rms = (boundary_sq / elements).sqrt();
        let output_rms = (output_sq / elements).sqrt();
        ensure!(
            boundary_rms >= 1e-6,
            "BF16 accumulator boundary arm produced identical outputs"
        );
        ensure!(
            boundary_rms <= 0.1 * output_rms,
            "BF16 accumulator boundary diverges: RMS {boundary_rms} vs output RMS {output_rms}"
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(&runtime_dir);
        Ok(())
    }

    // GPU gate: run in a granted window with FF_DS41_GPU_WINDOW=granted via
    //   cargo test d1_layer0_matches_model_definition --lib -- --exact --ignored --test-threads=1
    // FF_DS41_MODEL holds the publisher checkpoint directory and
    // FF_DS41_LAYER0_FIXTURE holds the v4 fixture beside thresholds-r2.json.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_DS41_GPU_WINDOW=granted, FF_DS41_MODEL and FF_DS41_LAYER0_FIXTURE"]
    fn d1_layer0_matches_model_definition() -> Result<()> {
        use serde::Deserialize;
        // 2026-10-28 07:03:37 UTC: isolated-layer band = max(3*spread, analytic).
        const TF_POSITIONS: [usize; 9] = [0, 1, 2, 63, 64, 126, 127, 128, 129];
        // Fixture parameters as recorded by deepseek_reference.py layer0 mode.
        const SEED_POSITIONS: usize = 129;
        const DECODE_POSITIONS: usize = 1;
        const STREAM_COPIES: usize = 4;
        const HIDDEN: usize = 5120;
        const STREAM_WIDTH: usize = STREAM_COPIES * HIDDEN;
        const HC_PRE_CALLS: usize = 2;
        const ATTN_CALLS: usize = 1;
        const FFN_CALLS: usize = 1;
        // layer0_pattern: amplitude * sin(seed * phi + index * phi') in f64,
        // rounded once to bf16.
        const PATTERN_SEED: f64 = 20261026.0;
        const PATTERN_AMPLITUDE: f64 = 0.75;
        // Bit patterns of the fixture generator literals (f64::consts has no
        // PHI and the decimal literals trip the approximate-constant lint).
        const PATTERN_PHI: f64 = f64::from_bits(0x3FF9_E377_9B97_F4A8);
        const PATTERN_PHI_FRAGMENT: f64 = f64::from_bits(0x3FE3_C6EF_372F_E950);

        type F32Map3 = Vec<Vec<Vec<f32>>>;

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct FrozenBand {
            spread: f64,
            analytic: f64,
            bound: f64,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Layer0Thresholds {
            layer: usize,
            positions: Vec<usize>,
            bands: std::collections::BTreeMap<String, Vec<FrozenBand>>,
        }

        /// One hc_mixes invocation: (pre, post, comb).
        #[derive(Deserialize)]
        struct HcMixesCall(F32Map3, F32Map3, Vec<Vec<Vec<Vec<f32>>>>);

        #[derive(Deserialize)]
        struct Provenance {
            revision: String,
            producer: String,
        }

        #[derive(Deserialize)]
        struct Layer0Document {
            layer: u64,
            provenance: Option<Provenance>,
            seed_positions: u64,
            window: u64,
            runs: std::collections::BTreeMap<String, Vec<Layer0Record>>,
        }

        #[derive(Deserialize)]
        struct Layer0Record {
            phase: String,
            start_pos: u64,
            stream: Vec<Vec<i16>>,
            pre_mix: Vec<Vec<Vec<f32>>>,
            carried_pre_mix: Option<Vec<Vec<Vec<f32>>>>,
            // Fixture v2 decode records: the full latent K ring entering
            // position 129 as [window][head_dim] bf16 bits, slot-major at
            // slot = position % window (the publisher ring covers positions
            // 1..128 at slots 1..127,0 after the seed call).
            ring_kv: Option<Vec<Vec<i16>>>,
            // Fixture v4 seed records: the input stream rows at the
            // teacher-forced seed positions, bit-exact, keyed by position.
            tf_streams: Option<std::collections::BTreeMap<String, Vec<Vec<i16>>>>,
            // Fixture v4 seed records: every per-position latent K row the
            // prefill call produced, [seed_positions][head_dim] bf16 bits
            // (the ring entering any position p assembles from rows < p).
            seed_kvs: Option<Vec<Vec<i16>>>,
            // Fixture v7: the publisher's own E4M3 codes and E8M0 scales per position, injected
            // into the ring as bytes.
            seed_k_codes: Option<Vec<Vec<u8>>>,
            seed_k_scales: Option<Vec<Vec<u8>>>,
            // One call: (pre, post, comb) as recorded per hc_mixes invocation.
            mixes: Vec<HcMixesCall>,
            collapse: Vec<Vec<Vec<Vec<i16>>>>,
            attn: Vec<Vec<Vec<Vec<i16>>>>,
            expand: Vec<Vec<Vec<Vec<Vec<i16>>>>>,
            ffn: Vec<Vec<Vec<Vec<i16>>>>,
        }

        /// Flattened per-call boundaries of one fixture record; `stream` is the
        /// single recorded input row of the record (the last seed row, or the
        /// decode row), not one row per position.
        struct StepRecords {
            stream: Vec<bf16>,
            stream_raw: Vec<i16>,
            pre_mix: Vec<f64>,
            carried: Option<Vec<f64>>,
            mixes: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
            collapse: Vec<Vec<bf16>>,
            attn: Vec<Vec<bf16>>,
            expand: Vec<Vec<bf16>>,
            ffn: Vec<Vec<bf16>>,
        }

        fn bits(value: i16) -> bf16 {
            bf16::from_bits(value as u16)
        }

        /// E4M3 code to value; the ring holds finite codes only.
        fn e4m3_value(code: u8) -> f64 {
            let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
            let magnitude = code & 0x7f;
            let exponent = i32::from(magnitude >> 3);
            let mantissa = i32::from(magnitude & 7);
            let value = if exponent == 0 {
                f64::from(mantissa) / 512.0
            } else {
                (1.0 + f64::from(mantissa) / 8.0) * (f64::from(exponent) - 7.0).exp2()
            };
            sign * value
        }

        fn e8m0_value(code: u8) -> f64 {
            (f64::from(code) - 127.0).exp2()
        }

        fn bf16_bits_to_f64(value: u16) -> f64 {
            f64::from(f32::from_bits(u32::from(value) << 16))
        }

        /// Round to BF16 (nearest, ties to even) and return the bits.
        fn f64_to_bf16_bits(value: f64) -> u16 {
            let bits = (value as f32).to_bits();
            ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
        }

        // record.stream: [copies][hidden] bf16 bits.
        fn stream_bits(label: &str, rows: &[Vec<i16>]) -> Result<Vec<bf16>> {
            ensure!(
                rows.len() == STREAM_COPIES && rows.iter().all(|row| row.len() == HIDDEN),
                "{label} stream holds {} rows of {}; the layout is [{STREAM_COPIES}][{HIDDEN}]",
                rows.len(),
                rows.first().map_or(0, |row| row.len()),
            );
            Ok(rows.iter().flatten().map(|&value| bits(value)).collect())
        }

        // collapse/attn/ffn call: [1][seq][hidden] bf16 bits.
        fn hidden_bits(
            label: &str,
            name: &str,
            call: &[Vec<Vec<i16>>],
            seq: usize,
        ) -> Result<Vec<bf16>> {
            ensure!(
                call.len() == 1
                    && call[0].len() == seq
                    && call[0].iter().all(|row| row.len() == HIDDEN),
                "{label} {name} holds {}x{}x{} bits; the layout is [1][{seq}][{HIDDEN}]",
                call.len(),
                call.first().map_or(0, |batch| batch.len()),
                call.first()
                    .and_then(|batch| batch.first())
                    .map_or(0, |row| row.len()),
            );
            Ok(call[0].iter().flatten().map(|&value| bits(value)).collect())
        }

        // expand call: [1][seq][copies][hidden] bf16 bits.
        fn stream_width_bits(
            label: &str,
            name: &str,
            call: &[Vec<Vec<Vec<i16>>>],
            seq: usize,
        ) -> Result<Vec<bf16>> {
            ensure!(
                call.len() == 1 && call[0].len() == seq,
                "{label} {name} holds {} batches of {} rows; the layout starts [1][{seq}]",
                call.len(),
                call.first().map_or(0, |batch| batch.len()),
            );
            let flat: Vec<i16> = call[0].iter().flatten().flatten().copied().collect();
            ensure!(
                flat.len() == seq * STREAM_WIDTH,
                "{label} {name} holds {} bits; the layout needs {seq}x{STREAM_WIDTH}",
                flat.len(),
            );
            Ok(flat.iter().map(|&value| bits(value)).collect())
        }

        // pre_mix, mixes pre/post: [1][seq][copies] f32.
        fn mix_coefficients(
            label: &str,
            name: &str,
            call: &[Vec<Vec<f32>>],
            seq: usize,
        ) -> Result<Vec<f64>> {
            ensure!(
                call.len() == 1
                    && call[0].len() == seq
                    && call[0].iter().all(|row| row.len() == STREAM_COPIES),
                "{label} {name} holds {}x{}x{} values; the layout is [1][{seq}][{STREAM_COPIES}]",
                call.len(),
                call.first().map_or(0, |batch| batch.len()),
                call.first()
                    .and_then(|batch| batch.first())
                    .map_or(0, |row| row.len()),
            );
            Ok(call[0]
                .iter()
                .flatten()
                .map(|&value| f64::from(value))
                .collect())
        }

        // mixes comb: [1][seq][copies][copies] f32, rows indexed [j=out, k=in].
        fn comb_coefficients(
            label: &str,
            name: &str,
            call: &[Vec<Vec<Vec<f32>>>],
            seq: usize,
        ) -> Result<Vec<f64>> {
            ensure!(
                call.len() == 1 && call[0].len() == seq,
                "{label} {name} holds {} batches of {} rows; the layout starts [1][{seq}]",
                call.len(),
                call.first().map_or(0, |batch| batch.len()),
            );
            let flat: Vec<f32> = call[0].iter().flatten().flatten().copied().collect();
            ensure!(
                flat.len() == seq * STREAM_COPIES * STREAM_COPIES,
                "{label} {name} holds {} values; the layout needs {seq}x{STREAM_COPIES}x{STREAM_COPIES}",
                flat.len(),
            );
            Ok(flat.iter().map(|&value| f64::from(value)).collect())
        }

        fn step_records(label: &str, record: &Layer0Record, seq: usize) -> Result<StepRecords> {
            let mut mixes = Vec::with_capacity(record.mixes.len());
            for (call, hc_call) in record.mixes.iter().enumerate() {
                let (pre, post, comb) = (&hc_call.0, &hc_call.1, &hc_call.2);
                mixes.push((
                    mix_coefficients(label, &format!("mixes[{call}].pre"), pre, seq)?,
                    mix_coefficients(label, &format!("mixes[{call}].post"), post, seq)?,
                    comb_coefficients(label, &format!("mixes[{call}].comb"), comb, seq)?,
                ));
            }
            ensure!(
                mixes.len() == HC_PRE_CALLS,
                "{label} records {} hc_mixes calls; the block runs one per HcPre",
                mixes.len(),
            );
            let mut collapse = Vec::with_capacity(record.collapse.len());
            for (call, tensor) in record.collapse.iter().enumerate() {
                collapse.push(hidden_bits(
                    label,
                    &format!("collapse[{call}]"),
                    tensor,
                    seq,
                )?);
            }
            ensure!(
                collapse.len() == HC_PRE_CALLS,
                "{label} records {} hc_pre calls; the block runs {HC_PRE_CALLS}",
                collapse.len(),
            );
            let mut attn = Vec::with_capacity(record.attn.len());
            for (call, tensor) in record.attn.iter().enumerate() {
                attn.push(hidden_bits(label, &format!("attn[{call}]"), tensor, seq)?);
            }
            ensure!(
                attn.len() == ATTN_CALLS,
                "{label} records {} attn calls; the block runs {ATTN_CALLS}",
                attn.len(),
            );
            let mut expand = Vec::with_capacity(record.expand.len());
            for (call, tensor) in record.expand.iter().enumerate() {
                expand.push(stream_width_bits(
                    label,
                    &format!("expand[{call}]"),
                    tensor,
                    seq,
                )?);
            }
            ensure!(
                expand.len() == HC_PRE_CALLS,
                "{label} records {} hc_post calls; the block runs {HC_PRE_CALLS}",
                expand.len(),
            );
            let mut ffn = Vec::with_capacity(record.ffn.len());
            for (call, tensor) in record.ffn.iter().enumerate() {
                ffn.push(hidden_bits(label, &format!("ffn[{call}]"), tensor, seq)?);
            }
            ensure!(
                ffn.len() == FFN_CALLS,
                "{label} records {} ffn calls; the block runs {FFN_CALLS}",
                ffn.len(),
            );
            let carried = record
                .carried_pre_mix
                .as_ref()
                .map(|tensor| mix_coefficients(label, "carried_pre_mix", tensor, 1))
                .transpose()?;
            Ok(StepRecords {
                stream_raw: record.stream.concat(),
                stream: stream_bits(label, &record.stream)?,
                pre_mix: mix_coefficients(label, "pre_mix", &record.pre_mix, seq)?,
                carried,
                mixes,
                collapse,
                attn,
                expand,
                ffn,
            })
        }

        /// Per-boundary margin: max |delta| over the run plus the count and the
        /// first element beyond tolerance.
        struct Margin {
            name: &'static str,
            rule: &'static str,
            compared: usize,
            beyond: usize,
            max_abs: f64,
            first: Option<((usize, usize), f64, f64)>,
        }

        impl Margin {
            fn new(name: &'static str, rule: &'static str) -> Self {
                Self {
                    name,
                    rule,
                    compared: 0,
                    beyond: 0,
                    max_abs: 0.0,
                    first: None,
                }
            }

            fn f64s(&mut self, position: usize, got: &[f32], want: &[f64], tol: f64) {
                self.compare_lengths(got.len(), want.len());
                for (index, (got, want)) in got.iter().zip(want).enumerate() {
                    let (got, want) = (f64::from(*got), *want);
                    self.abs_delta(position, index, got, want, |abs| abs > tol);
                }
            }

            fn bf16s(&mut self, position: usize, got: &[bf16], want: &[bf16], abs_tol: f64) {
                self.compare_lengths(got.len(), want.len());
                for (index, (got, want)) in got.iter().zip(want).enumerate() {
                    assert!(
                        got.is_finite() && want.is_finite(),
                        "non-finite boundary {}",
                        self.name
                    );
                    let abs = (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs();
                    self.compared += 1;
                    self.max_abs = self.max_abs.max(abs);
                    if abs > abs_tol {
                        self.beyond += 1;
                        if self.first.is_none() {
                            self.first = Some((
                                (position, index),
                                f64::from(got.to_f32()),
                                f64::from(want.to_f32()),
                            ));
                        }
                    }
                }
            }

            fn compare_lengths(&self, got: usize, want: usize) {
                assert_eq!(
                    got, want,
                    "boundary {} compares {} against {} elements",
                    self.name, got, want
                );
            }

            fn abs_delta(
                &mut self,
                position: usize,
                index: usize,
                got: f64,
                want: f64,
                beyond: impl Fn(f64) -> bool,
            ) {
                assert!(
                    got.is_finite() && want.is_finite(),
                    "non-finite boundary {}",
                    self.name
                );
                let abs = (got - want).abs();
                self.compared += 1;
                self.max_abs = self.max_abs.max(abs);
                if beyond(abs) {
                    self.beyond += 1;
                    if self.first.is_none() {
                        self.first = Some(((position, index), got, want));
                    }
                }
            }
        }

        // FP64 mixes of one hc call over the operands the engine consumed: the
        // uploaded fn/base/scale bytes and the stream row, in the definition's
        // form. The publisher's value is the third column of `hc_diag`.
        fn hc_mixes_f64(
            stream: &[bf16],
            weights: &(Vec<u8>, Vec<u8>, Vec<u8>),
            params: &HcParams,
        ) -> (Vec<f64>, Vec<f64>) {
            let (mult, hidden) = (params.mult as usize, params.hidden as usize);
            let (eps, norm_eps) = (f64::from(params.eps), f64::from(params.norm_eps));
            let (fn_bytes, base_bytes, scale_bytes) = (&weights.0, &weights.1, &weights.2);
            let n = mult * hidden;
            let basis = 2 * mult + mult * mult;
            assert_eq!(
                stream.len(),
                n,
                "hc mix stream holds {} elements; needs {n}",
                stream.len()
            );
            let f32s = |bytes: &[u8]| -> Vec<f64> {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f64::from(f32::from_le_bytes(*chunk)))
                    .collect()
            };
            let (fn_w, base, scale) = (f32s(fn_bytes), f32s(base_bytes), f32s(scale_bytes));
            assert_eq!(
                fn_w.len(),
                basis * n,
                "hc fn holds {} elements; needs {}",
                fn_w.len(),
                basis * n
            );
            let x: Vec<f64> = stream.iter().map(|v| f64::from(v.to_f32())).collect();
            let mean = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
            let rsqrt = 1.0 / (mean + norm_eps).sqrt();
            let mix: Vec<f64> = (0..basis)
                .map(|c| {
                    let row = &fn_w[c * n..(c + 1) * n];
                    row.iter()
                        .zip(&x)
                        .map(|(weight, value)| weight * value)
                        .sum::<f64>()
                        * rsqrt
                })
                .collect();
            let pre = (0..mult)
                .map(|j| 1.0 / (1.0 + (-(mix[j] * scale[0] + base[j])).exp()) + eps)
                .collect();
            let post = (0..mult)
                .map(|j| {
                    let u = mix[mult + j] * scale[1] + base[mult + j];
                    2.0 * (1.0 / (1.0 + (-u).exp()))
                })
                .collect();
            (pre, post)
        }

        // One line per element beyond its band, carrying the FP64 value of the
        // same operands: engine-vs-FP64 is the reduction, FP64-vs-publisher is
        // the operands.
        fn hc_diag(
            label: &str,
            position: usize,
            engine: &[f32],
            publisher: &[f64],
            reference: &[f64],
            tol: f64,
        ) {
            for (index, ((got, want), reference)) in
                engine.iter().zip(publisher).zip(reference).enumerate()
            {
                let (got, want, reference) = (f64::from(*got), *want, *reference);
                if (got - want).abs() > tol {
                    println!(
                        "LAYER0 diag {label} pos={position} j={index} engine={got:.9} \
                         publisher={want:.9} f64={reference:.9} eng-pub={:.3e} eng-f64={:.3e} \
                         pub-f64={:.3e}",
                        (got - want).abs(),
                        (got - reference).abs(),
                        (want - reference).abs(),
                    );
                }
            }
        }

        // Informational divergence of one informational position: RMS over
        // the f32 values of two bf16 rows; no assert reads it.
        fn rms_divergence(got: &[bf16], want: &[bf16]) -> f64 {
            let sum: f64 = got
                .iter()
                .zip(want)
                .map(|(got, want)| {
                    let delta = f64::from(got.to_f32()) - f64::from(want.to_f32());
                    delta * delta
                })
                .sum();
            (sum / got.len() as f64).sqrt()
        }

        // One boundary's max and RMS divergence: the profile that shows which
        // stage a layer's divergence starts at.
        fn profile(got: &[bf16], want: &[bf16]) -> (f64, f64) {
            let max = got
                .iter()
                .zip(want)
                .map(|(got, want)| (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs())
                .fold(0.0f64, f64::max);
            (max, rms_divergence(got, want))
        }

        ensure!(
            std::env::var("FF_DS41_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let model_dir = PathBuf::from(
            std::env::var_os("FF_DS41_MODEL")
                .context("FF_DS41_MODEL missing: the publisher checkpoint directory is required")?,
        );
        let fixture_path = PathBuf::from(std::env::var_os("FF_DS41_LAYER0_FIXTURE").context(
            "FF_DS41_LAYER0_FIXTURE missing: the recorded publisher layer-0 fixture is required",
        )?);
        let thresholds_path = fixture_path.with_file_name("thresholds-r2.json");
        let reference_dir =
            PathBuf::from(std::env::var_os("FF_DS41_LAYER0_REFERENCE_DIR").context(
                "FF_DS41_LAYER0_REFERENCE_DIR missing: the publisher's exact rows are required",
            )?);
        let thresholds: Layer0Thresholds = serde_json::from_reader(
            std::fs::File::open(&thresholds_path)
                .with_context(|| format!("open frozen bands {}", thresholds_path.display()))?,
        )?;
        ensure!(
            thresholds.layer == 0 && thresholds.positions == TF_POSITIONS,
            "frozen bands do not cover layer0 teacher-forced positions",
        );
        let names = [
            "attn_pre",
            "attn_post",
            "attn_comb",
            "ffn_pre",
            "ffn_post",
            "ffn_comb",
            "collapse_attn",
            "collapse_ffn",
            "attn",
            "ffn",
            "expand_attn",
            "expand_ffn",
            "carried_pre",
        ];
        ensure!(
            thresholds.bands.len() == names.len(),
            "frozen boundary count"
        );
        for name in names {
            let values = thresholds
                .bands
                .get(name)
                .with_context(|| format!("missing frozen boundary {name}"))?;
            ensure!(
                values.len() == TF_POSITIONS.len(),
                "frozen position count for {name}"
            );
            for value in values {
                ensure!(
                    [value.spread, value.analytic, value.bound]
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                        // 2026-10-28 dsflash (D1 lead): the frozen file is data written by an
                        // external tool, and a decimal round-trip can move a value by one ulp
                        // (serde_json's fast float path against the writer's repr). The rule is
                        // unchanged: the tolerance covers the round-trip, not the threshold.
                        && value.bound >= (3.0 * value.spread).max(value.analytic)
                        && value.bound
                            <= (3.0 * value.spread).max(value.analytic)
                                * (1.0 + 4.0 * f64::EPSILON),
                    "invalid max(3*spread, analytic) band for {name}: spread={:?} analytic={:?} bound={:?}",
                    value.spread,
                    value.analytic,
                    value.bound,
                );
            }
        }
        // Extra diagnostic positions have no frozen band: they print and never assert.
        let band = |name: &str, position: usize| {
            if position < TF_POSITIONS.len() {
                thresholds.bands[name][position].bound
            } else {
                f64::INFINITY
            }
        };
        let document: Layer0Document = serde_json::from_reader(
            std::fs::File::open(&fixture_path)
                .with_context(|| format!("open {}", fixture_path.display()))?,
        )
        .with_context(|| format!("parse layer-0 fixture {}", fixture_path.display()))?;
        ensure!(document.layer == 0, "fixture layer {}", document.layer);
        // The recorded per-position K rows must come from the gated revision: a fixture from
        // another revision injects rows the publisher never used.
        let provenance = document
            .provenance
            .as_ref()
            .context("fixture carries no provenance: revision and producer are required")?;
        ensure!(
            provenance.revision == "DeepSeek-V4.1-Flash-1017",
            "fixture revision {} is not DeepSeek-V4.1-Flash-1017",
            provenance.revision,
        );
        ensure!(
            !provenance.producer.is_empty(),
            "fixture provenance carries no producer",
        );
        ensure!(
            document.seed_positions as usize == SEED_POSITIONS,
            "fixture seed_positions {} is not {SEED_POSITIONS}",
            document.seed_positions,
        );
        let runtime_missing = "DeepSeek-V4.1 mHC runtime is not bound";
        let carry_missing = "mHC carried pre-mix is not bound";
        let input_run = document.runs.get("input").with_context(|| {
            format!("fixture runs hold no input run: {:?}", document.runs.keys())
        })?;
        let control_run = document.runs.get("control").with_context(|| {
            format!(
                "fixture runs hold no control run: {:?}",
                document.runs.keys()
            )
        })?;
        ensure!(
            input_run.len() == 2 && control_run.len() == 2,
            "fixture runs hold {} and {} records; each run is one seed and one decode record",
            input_run.len(),
            control_run.len(),
        );
        ensure!(
            input_run[0].phase == "seed"
                && input_run[0].start_pos == 0
                && input_run[1].phase == "decode"
                && input_run[1].start_pos as usize == SEED_POSITIONS,
            "input run phases are {}/{:?} at starts {}/{}",
            input_run[0].phase,
            input_run[1].phase,
            input_run[0].start_pos,
            input_run[1].start_pos,
        );
        let seed_steps = step_records("input seed", &input_run[0], SEED_POSITIONS)?;
        let decode_steps = step_records("input decode", &input_run[1], DECODE_POSITIONS)?;
        let control_decode = step_records("control decode", &control_run[1], DECODE_POSITIONS)?;
        let differing = decode_steps.expand[1]
            .iter()
            .zip(&control_decode.expand[1])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(
            differing > 0,
            "fixture input and control decode final streams are byte-identical",
        );
        // The recording holds the last seed row and the decode row only; the
        // first SEED_POSITIONS-1 rows regenerate from the frozen pattern, and
        // the recorded last row pins the regeneration bit-exactly.
        let pattern_row = |first: usize| -> Vec<bf16> {
            (first..first + STREAM_WIDTH)
                .map(|index| {
                    bf16::from_f64(
                        PATTERN_AMPLITUDE
                            * (PATTERN_SEED * PATTERN_PHI + index as f64 * PATTERN_PHI_FRAGMENT)
                                .sin(),
                    )
                })
                .collect()
        };
        let regenerated = pattern_row((SEED_POSITIONS - 1) * STREAM_WIDTH);
        // Wiring control: the regeneration formula is right when the row
        // matches the recording beyond platform sin() disagreement — the
        // publisher computes sin through torch's libm, the gate through
        // Rust's, and a 1-ulp argument difference can flip one bf16
        // rounding. At most 4 of 20480 elements, each within one bf16 ulp.
        let mut pattern_mismatches = 0usize;
        for (a, raw) in regenerated.iter().zip(&seed_steps.stream_raw) {
            let b = bits(*raw);
            if a.to_bits() != b.to_bits() {
                let (ha, la) = (a.to_bits() as i16, b.to_bits() as i16);
                let flipped = (ha - la).abs();
                ensure!(
                    flipped <= 1,
                    "regenerated seed row differs beyond one bf16 ulp: {a:?} vs {b:?}"
                );
                pattern_mismatches += 1;
            }
        }
        ensure!(
            pattern_mismatches <= 4,
            "regenerated layer0_pattern seed row {} differs from the recorded stream at {pattern_mismatches} of {STREAM_WIDTH} elements",
            SEED_POSITIONS - 1,
        );

        let runtime_dir =
            std::env::temp_dir().join(format!("flyingfish-deepseek-layer0-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;
        let hash_record = PathBuf::from(std::env::var_os("FF_DS41_ENGRAM_HASH").context(
            "FF_DS41_ENGRAM_HASH missing: the accepted engram-hash-v1 record is required",
        )?);
        std::fs::copy(&hash_record, runtime_dir.join("engram-hash-v1.json"))
            .with_context(|| format!("copy {}", hash_record.display()))?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(runtime_dir.clone()),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let mut engine = Engine::load(
            &model_dir,
            0,
            SEED_POSITIONS + DECODE_POSITIONS,
            None,
            settings,
        )?;
        let device = engine.device.clone();
        let s = &device.stream;
        let params = engine.hc.as_ref().context(runtime_missing)?.binding.params;
        ensure!(
            params.mult as usize == STREAM_COPIES,
            "engine hc_mult {} is not {STREAM_COPIES}",
            params.mult,
        );
        ensure!(
            params.hidden as usize == HIDDEN,
            "engine hidden {} is not {HIDDEN}",
            params.hidden,
        );
        let geometry = MixerGeometry::new(&engine.model_spec)?;
        ensure!(
            geometry.window as u64 == document.window,
            "engine attention window {} is not the fixture window {}",
            geometry.window,
            document.window,
        );
        let mixer_kind = engine.model_spec.layers[0].mixer;
        let mut scratch = s.alloc_zeros(HIDDEN)?;
        // Layer-0 hc weights as uploaded: the FP64 mixes reference reads these,
        // not the checkpoint, so the reference shares the engine's operands.
        let hc_attn_bytes = engine
            .hc
            .as_ref()
            .context(runtime_missing)?
            .sublayer_weight_host(s, 0, HcSublayer::Attention)?;
        let hc_ffn_bytes = engine
            .hc
            .as_ref()
            .context(runtime_missing)?
            .sublayer_weight_host(s, 0, HcSublayer::Ffn)?;

        // One margin per frozen band: the attention HcPre (mixes[0]) and the
        // FFN HcPre (mixes[1]) carry separate bands and separate operands.
        let mut attn_pre = Margin::new("attn_pre", "max(3*spread, analytic)");
        let mut attn_post = Margin::new("attn_post", "max(3*spread, analytic)");
        let mut attn_comb = Margin::new("attn_comb", "max(3*spread, analytic)");
        let mut ffn_pre = Margin::new("ffn_pre", "max(3*spread, analytic)");
        let mut ffn_post = Margin::new("ffn_post", "max(3*spread, analytic)");
        let mut ffn_comb = Margin::new("ffn_comb", "max(3*spread, analytic)");
        let mut collapse_margin = Margin::new("collapse", "teacher-forced, freeze table");
        let mut attn_margin = Margin::new("attn", "teacher-forced, freeze table");
        let mut expand_margin = Margin::new("expand", "teacher-forced, freeze table");
        let mut ffn_margin = Margin::new("ffn", "teacher-forced, freeze table");
        let mut final_stream = Margin::new("final_stream", "teacher-forced, freeze table");
        let mut carried_margin = Margin::new("carried_pre_mix", "max(3*spread, analytic)");
        // Informational free-run statistics (ruling (3)): collected over the
        // whole 130-step run, printed, never asserted.
        let mut info_collapse = Margin::new("info_collapse", "free-run");
        let mut info_attn = Margin::new("info_attn", "free-run");
        let mut info_ffn = Margin::new("info_ffn", "free-run");
        let mut info_final = Margin::new("info_final_stream", "free-run");
        let step = Step {
            rows: 1,
            input: Input::Ids,
            head: Head::None,
            mask: Mask::Causal,
        };
        let mut control_compared = 0usize;
        let mut control_beyond = 0usize;
        let window = geometry.window as usize;
        let tf_streams = input_run[0].tf_streams.as_ref().context(
            "layer-0 fixture v4 seed record carries no tf_streams: the bit-exact teacher-forced input rows are required",
        )?;
        let seed_kvs = input_run[0].seed_kvs.as_ref().context(
            "layer-0 fixture v4 seed record carries no seed_kvs: the per-position latent K rows are required",
        )?;
        ensure!(
            seed_kvs.len() == SEED_POSITIONS
                && seed_kvs.iter().all(|row| row.len() == geometry.head_dim),
            "input seed_kvs holds {} rows; the layout is [{SEED_POSITIONS}][{}]",
            seed_kvs.len(),
            geometry.head_dim,
        );
        let seed_k_codes = input_run[0].seed_k_codes.as_ref().context(
            "layer-0 fixture v7 seed record carries no seed_k_codes: the publisher's ring codes are required",
        )?;
        let seed_k_scales = input_run[0].seed_k_scales.as_ref().context(
            "layer-0 fixture v7 seed record carries no seed_k_scales: the publisher's ring scales are required",
        )?;
        ensure!(
            seed_k_codes.len() == SEED_POSITIONS
                && seed_k_codes
                    .iter()
                    .all(|row| row.len() == geometry.head_dim)
                && seed_k_scales.len() == SEED_POSITIONS
                && seed_k_scales
                    .iter()
                    .all(|row| row.len() == geometry.head_dim / 32),
            "seed_k_codes/seed_k_scales layouts are [{SEED_POSITIONS}][{}] / [{SEED_POSITIONS}][{}]",
            geometry.head_dim,
            geometry.head_dim / 32,
        );
        println!(
            "LAYER0 fixture seed0 block15 amax={:.6} fixture={}",
            seed_kvs[0][480..512]
                .iter()
                .map(|&v| f64::from(bits(v).to_f32().abs()))
                .fold(0.0f64, f64::max),
            fixture_path.display(),
        );
        let recorded_ring = input_run[1]
            .ring_kv
            .as_ref()
            .context("input decode record has no ring snapshot")?;
        ensure!(
            recorded_ring.len() == window
                && recorded_ring
                    .iter()
                    .all(|row| row.len() == geometry.head_dim),
            "recorded latent ring layout differs from [{window}, {}]",
            geometry.head_dim,
        );
        for (position, row) in seed_kvs.iter().enumerate() {
            let values: Vec<bf16> = row.iter().map(|&value| bits(value)).collect();
            let decoded = mix_decoded_acts(&values)?;
            ensure!(
                values
                    .iter()
                    .zip(decoded)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "publisher latent row {position} is not idempotent under native re-encoding",
            );
            if position >= SEED_POSITIONS - window {
                ensure!(
                    &recorded_ring[position % window] == row,
                    "assembled latent slot {} differs from the publisher snapshot",
                    position % window,
                );
            }
        }
        let stream_row = |position: usize| -> Result<Vec<bf16>> {
            if position < SEED_POSITIONS - 1 {
                let raw = tf_streams.get(&position.to_string()).with_context(|| {
                    format!("tf_streams carries no row for position {position}")
                })?;
                stream_bits("tf_streams", raw)
            } else if position == SEED_POSITIONS - 1 {
                Ok(seed_steps.stream.clone())
            } else {
                Ok(decode_steps.stream.clone())
            }
        };
        // Teacher forcing (ruling (2)): the publisher's carried state — the
        // latent K ring rows for positions < p at slot pos % window, each
        // re-quantized through the shared block-32 rule into the engine ring
        // (the recorded rows are exact E4M3 grid points, so re-quantization
        // reproduces the publisher codes) — then exactly one engine step.
        macro_rules! inject_ring {
            ($position:expr) => {{
                let nblocks = geometry.head_dim / 32;
                let first = ($position).saturating_sub(window);
                let mut codes = vec![0u8; window * geometry.head_dim];
                let mut scales = vec![0u8; window * nblocks];
                for pos in first..($position) {
                    let slot = pos % window;
                    codes[slot * geometry.head_dim..(slot + 1) * geometry.head_dim]
                        .copy_from_slice(&seed_k_codes[pos]);
                    scales[slot * nblocks..(slot + 1) * nblocks]
                        .copy_from_slice(&seed_k_scales[pos]);
                }
                let ring = engine.state.latent_ring_mut(0)?;
                s.memcpy_htod(&codes, &mut ring.codes)?;
                s.memcpy_htod(&scales, &mut ring.scales)?;
                if ($position) <= 2 {
                    println!(
                        "LAYER0 inject pos={} slot0_scale={} publisher_slot0_scale={}",
                        $position, scales[15], seed_k_scales[0][15],
                    );
                }
            }};
        }
        // Pass 1: free-run over all 130 steps with the engine's own state;
        // informational only.
        for position in 0..SEED_POSITIONS + DECODE_POSITIONS {
            let (records, index): (&StepRecords, usize) = if position < SEED_POSITIONS {
                (&seed_steps, position)
            } else {
                (&decode_steps, position - SEED_POSITIONS)
            };
            let input_row = if position < SEED_POSITIONS - 1 {
                pattern_row(position * STREAM_WIDTH)
            } else {
                stream_row(position)?
            };
            engine.prepare_step(step)?;
            engine.position = position;
            engine
                .state
                .set_sequence(s, position as u32, 1, engine.workspace_rows as u32)?;
            s.memcpy_htod(
                &input_row,
                &mut engine
                    .hc
                    .as_mut()
                    .context(runtime_missing)?
                    .stream
                    .slice_mut(..STREAM_WIDTH),
            )?;
            if position < SEED_POSITIONS {
                let carried: Vec<f32> = records.pre_mix
                    [index * STREAM_COPIES..(index + 1) * STREAM_COPIES]
                    .iter()
                    .map(|&value| value as f32)
                    .collect();
                s.memcpy_htod(
                    &carried,
                    &mut engine
                        .hc
                        .as_mut()
                        .context(runtime_missing)?
                        .carry
                        .as_mut()
                        .context(carry_missing)?
                        .slice_mut(..STREAM_COPIES),
                )?;
            }
            engine.run_hc_stage(0, HcStage::Pre, 0, step)?;
            if position == 0 {
                // Planted wiring control (model.py:978-980): the attention
                // sublayer collapsing with its own pre instead of the carried
                // one must move the collapse beyond the same thresholds.
                engine.hc.as_mut().context(runtime_missing)?.hc_pre(
                    &engine.device.stream,
                    0,
                    HcSublayer::Attention,
                    HcPreSource::Own,
                    1,
                    &mut scratch,
                )?;
                let control_x: Vec<bf16> = s.clone_dtoh(&scratch.slice(..HIDDEN))?;
                control_compared = HIDDEN;
                control_beyond = control_x
                    .iter()
                    .zip(&records.collapse[0][..HIDDEN])
                    .filter(|(got, want)| {
                        (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                            > band("collapse_attn", 0)
                    })
                    .count();
            }
            let x: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..HIDDEN))?;
            info_collapse.bf16s(
                position,
                &x,
                &records.collapse[0][index * HIDDEN..(index + 1) * HIDDEN],
                f64::INFINITY,
            );
            engine.run_operator(Operator::Norm, 0, step, &mut None)?;
            engine.run_mixer(mixer_kind, 0, step, &mut None)?;
            let attn_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?;
            info_attn.bf16s(
                position,
                &attn_out,
                &records.attn[0][index * HIDDEN..(index + 1) * HIDDEN],
                f64::INFINITY,
            );
            engine.run_hc_stage(3, HcStage::Post, 0, step)?;
            engine.run_hc_stage(4, HcStage::Pre, 0, step)?;
            let x: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..HIDDEN))?;
            info_collapse.bf16s(
                position,
                &x,
                &records.collapse[1][index * HIDDEN..(index + 1) * HIDDEN],
                f64::INFINITY,
            );
            engine.run_operator(Operator::PostNorm, 0, step, &mut None)?;
            engine.run_ffn(0, step, &mut None)?;
            let ffn_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?;
            info_ffn.bf16s(
                position,
                &ffn_out,
                &records.ffn[0][index * HIDDEN..(index + 1) * HIDDEN],
                f64::INFINITY,
            );
            engine.run_hc_stage(7, HcStage::Post, 0, step)?;
            let final_row = s.clone_dtoh(&engine.hc.as_ref().context(runtime_missing)?.stream)?;
            let final_row = final_row[..STREAM_WIDTH].to_vec();
            let want = &records.expand[1][index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH];
            info_final.bf16s(position, &final_row, want, f64::INFINITY);
            println!(
                "LAYER0 freerun pos={position} final_stream_rms={:.3e}",
                rms_divergence(&final_row, want)
            );
        }
        ensure!(
            control_beyond > 0,
            "attention-own pre-mix control does not discriminate: {control_beyond} of {control_compared} collapse elements beyond tolerance",
        );
        // Pass 2: teacher-forced single steps at the sampled positions
        // (ruling (2)); positions ascend so the sequence cursor only moves
        // forward.
        let mut tf_positions = TF_POSITIONS.to_vec();
        if let Ok(extra) = std::env::var("FF_DS41_LAYER0_EXTRA_POSITIONS") {
            for field in extra.split(',') {
                let position = field.trim().parse::<usize>()?;
                ensure!(
                    position < SEED_POSITIONS,
                    "extra diagnostic position {position} is not a seed position"
                );
                tf_positions.push(position);
            }
            tf_positions.sort_unstable();
            println!("LAYER0 extra_positions {}", tf_positions.len());
        }
        let isolated_attn = std::env::var_os("FF_DS41_LAYER0_ISOLATED_ATTN").is_some();
        let isolated_ffn = std::env::var_os("FF_DS41_LAYER0_ISOLATED_FFN").is_some();
        println!(
            "LAYER0 gate mode={}",
            if isolated_attn {
                "isolated-attn"
            } else {
                "propagated"
            }
        );
        let (mut attn_exempt, mut attn_defect) = (0usize, 0usize);
        // 2026-10-29: diagnostic positions do not change frozen position indices.
        for &position in &tf_positions {
            let tf = TF_POSITIONS
                .iter()
                .position(|&frozen| frozen == position)
                .unwrap_or(TF_POSITIONS.len());
            let decode = position == SEED_POSITIONS;
            let (records, index): (&StepRecords, usize) = if position < SEED_POSITIONS {
                (&seed_steps, position)
            } else {
                (&decode_steps, position - SEED_POSITIONS)
            };
            let input_row = stream_row(position)?;
            engine.prepare_step(step)?;
            engine.position = position;
            engine
                .state
                .set_sequence(s, position as u32, 1, engine.workspace_rows as u32)?;
            s.memcpy_htod(
                &input_row,
                &mut engine
                    .hc
                    .as_mut()
                    .context(runtime_missing)?
                    .stream
                    .slice_mut(..STREAM_WIDTH),
            )?;
            let carried: Vec<f32> = if decode {
                decode_steps
                    .carried
                    .as_ref()
                    .context("input decode record carries no carried_pre_mix recording")?
                    .iter()
                    .map(|&value| value as f32)
                    .collect()
            } else {
                records.pre_mix[index * STREAM_COPIES..(index + 1) * STREAM_COPIES]
                    .iter()
                    .map(|&value| value as f32)
                    .collect()
            };
            s.memcpy_htod(
                &carried,
                &mut engine
                    .hc
                    .as_mut()
                    .context(runtime_missing)?
                    .carry
                    .as_mut()
                    .context(carry_missing)?
                    .slice_mut(..STREAM_COPIES),
            )?;
            inject_ring!(position);
            println!("LAYER0 teacher_forced pos={position}");
            // Program stage 0 (HcPre): the attention mixes and the collapse
            // with the carried pre-mix.
            engine.run_hc_stage(0, HcStage::Pre, 0, step)?;
            let runtime = engine.hc.as_ref().context(runtime_missing)?;
            let mix_pre: Vec<f32> = s.clone_dtoh(&runtime.mix_pre.slice(..STREAM_COPIES))?;
            let mix_post: Vec<f32> = s.clone_dtoh(&runtime.mix_post.slice(..STREAM_COPIES))?;
            let mix_comb: Vec<f32> =
                s.clone_dtoh(&runtime.mix_comb.slice(..STREAM_COPIES * STREAM_COPIES))?;
            let want = &records.mixes[0];
            let (pre_f64, post_f64) = hc_mixes_f64(&input_row, &hc_attn_bytes, &params);
            let pre_rows = &want.0[index * STREAM_COPIES..(index + 1) * STREAM_COPIES];
            let post_rows = &want.1[index * STREAM_COPIES..(index + 1) * STREAM_COPIES];
            attn_pre.f64s(position, &mix_pre, pre_rows, band("attn_pre", tf));
            attn_post.f64s(position, &mix_post, post_rows, band("attn_post", tf));
            attn_comb.f64s(
                position,
                &mix_comb,
                &want.2[index * STREAM_COPIES * STREAM_COPIES
                    ..(index + 1) * STREAM_COPIES * STREAM_COPIES],
                band("attn_comb", tf),
            );
            hc_diag(
                "attn_pre",
                position,
                &mix_pre,
                pre_rows,
                &pre_f64,
                band("attn_pre", tf),
            );
            hc_diag(
                "attn_post",
                position,
                &mix_post,
                post_rows,
                &post_f64,
                band("attn_post", tf),
            );
            let x: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..HIDDEN))?;
            let want_collapse = &records.collapse[0][index * HIDDEN..(index + 1) * HIDDEN];
            collapse_margin.bf16s(position, &x, want_collapse, band("collapse_attn", tf));
            let (collapse_max, collapse_rms) = profile(&x, want_collapse);
            if position == 0 {
                let carry: Vec<f32> = s.clone_dtoh(
                    &engine
                        .hc
                        .as_ref()
                        .context(runtime_missing)?
                        .carry
                        .as_ref()
                        .context(carry_missing)?
                        .slice(..STREAM_COPIES),
                )?;
                let head = |row: &[bf16]| -> Vec<String> {
                    row[..4]
                        .iter()
                        .map(|v| format!("{:.6}", v.to_f32()))
                        .collect()
                };
                println!("LAYER0 diag collapse_head pos=0 engine={:?}", head(&x));
                println!(
                    "LAYER0 diag collapse_head pos=0 publisher={:?}",
                    head(&records.collapse[0][..4])
                );
                println!("LAYER0 diag carry pos=0 {:?}", carry);
                println!("LAYER0 diag stream_copy0 {:?}", head(&input_row[..4]));
                println!(
                    "LAYER0 diag stream_copy1 {:?}",
                    head(&input_row[HIDDEN..HIDDEN + 4])
                );
            }
            engine.run_operator(Operator::Norm, 0, step, &mut None)?;
            engine.run_mixer(mixer_kind, 0, step, &mut None)?;
            if let Some(dir) = std::env::var_os("FF_DS41_LAYER0_DUMP_DIR") {
                let normed: Vec<bf16> = s.clone_dtoh(&engine.work.n.slice(..HIDDEN))?;
                let encode_normed: Vec<u8> = normed
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect();
                std::fs::write(
                    PathBuf::from(&dir).join(format!("engine-xnorm-{position:03}.bin")),
                    encode_normed,
                )?;
                let (codes, scales) = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?
                    .activation_codes_host(s, 1, HIDDEN)?;
                std::fs::write(
                    PathBuf::from(&dir).join(format!("engine-xcodes-{position:03}.bin")),
                    &codes,
                )?;
                std::fs::write(
                    PathBuf::from(&dir).join(format!("engine-xscales-{position:03}.bin")),
                    &scales,
                )?;
                let dir = PathBuf::from(dir);
                let (o_row, _) = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?
                    .attn_output_host(s, 1)?;
                let raw: Vec<u8> = o_row
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect();
                std::fs::write(dir.join(format!("engine-o-{position:03}.bin")), raw)?;
                let qkv = &engine.work.qkv;
                let (q_row, k_row) = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?
                    .query_key_host(s, qkv, 1)?;
                let encode = |row: &[bf16]| -> Vec<u8> {
                    row.iter()
                        .flat_map(|value| value.to_bits().to_le_bytes())
                        .collect()
                };
                std::fs::write(
                    dir.join(format!("engine-q-{position:03}.bin")),
                    encode(&q_row),
                )?;
                std::fs::write(
                    dir.join(format!("engine-kv-{position:03}.bin")),
                    encode(&k_row),
                )?;
                let lora_row = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?
                    .query_latent_host(s, 1)?;
                std::fs::write(
                    dir.join(format!("engine-qr-{position:03}.bin")),
                    encode(&lora_row),
                )?;
            }
            let attn_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?;
            let want_attn = &records.attn[0][index * HIDDEN..(index + 1) * HIDDEN];
            attn_margin.bf16s(position, &attn_out, want_attn, band("attn", tf));
            // Rule B (4d ruling 2026-10-28T18:43:08Z): an element beyond the attention band is
            // exempt only when some P_j of its head lies within the score-dot analytic bound of
            // a BF16 rounding boundary; every other beyond element is a defect.
            let attn_tol = band("attn", tf);
            if attn_tol.is_finite() {
                let beyond: Vec<usize> = attn_out
                    .iter()
                    .zip(want_attn)
                    .enumerate()
                    .filter(|(_, (got, want))| {
                        (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs() > attn_tol
                    })
                    .map(|(index, _)| index)
                    .collect();
                if !beyond.is_empty() {
                    // Rule B at the projection (4d ruling 2026-10-28T20:04:47Z). The
                    // attention kernel output is bit-exact against the record, so a beyond
                    // element at this boundary is the wo_b store: exempt it when its
                    // distance to the nearest BF16 boundary is within the wo_b
                    // accumulation bound plus the operand term, at most W_BOUND_OPERANDS
                    // elements of the wo_b input differing by one E4M3 code ulp each.
                    const W_BOUND_OPERANDS: f64 = 2.0;
                    let mixer = engine.deepseek_mixer.as_ref().context(runtime_missing)?;
                    let (mut exempt, mut defect) = (0usize, 0usize);
                    let mut first_defect: Option<(usize, f64, f64)> = None;
                    for &element in &beyond {
                        let row = mixer.wo_b_row_host(s, 0, element)?;
                        let (act_codes, act_scales, w_codes, w_scales) =
                            (row.act_codes, row.act_scales, row.w_codes, row.w_scales);
                        let (mut sum_abs, mut operand) = (0.0f64, 0.0f64);
                        for (i, (&wc, &ac)) in w_codes.iter().zip(&act_codes).enumerate() {
                            let w = e4m3_value(wc) * e8m0_value(w_scales[i / 32]);
                            let a = e4m3_value(ac) * e8m0_value(act_scales[i / 32]);
                            let term = (w * a).abs();
                            sum_abs += term;
                            operand = operand.max(term);
                        }
                        let gamma_n = (act_codes.len() / 16) as f64 * f64::exp2(-24.0);
                        let bound =
                            gamma_n * sum_abs + W_BOUND_OPERANDS * f64::exp2(-3.0) * operand;
                        let value = f64::from(attn_out[element].to_f32());
                        let slot = f64_to_bf16_bits(value);
                        let low = bf16_bits_to_f64(slot);
                        let high = bf16_bits_to_f64(slot.wrapping_add(1));
                        let below = bf16_bits_to_f64(slot.wrapping_sub(1));
                        let distance = (value - 0.5 * (below + low))
                            .abs()
                            .min((0.5 * (low + high) - value).abs());
                        if distance <= bound {
                            exempt += 1;
                        } else {
                            defect += 1;
                            first_defect.get_or_insert((
                                element,
                                value,
                                f64::from(want_attn[element].to_f32()),
                            ));
                        }
                    }
                    println!(
                        "LAYER0 attn_projection_bound pos={position} beyond={} exempt={exempt} defect={defect} first_defect={first_defect:?}",
                        beyond.len(),
                    );
                    attn_exempt += exempt;
                    attn_defect += defect;
                }
            }
            let (attn_max, attn_rms) = profile(&attn_out, want_attn);
            if position <= 20 {
                let count = (position + 1).min(window);
                let ids = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?
                    .visible_slots_host(s, count)?;
                println!(
                    "LAYER0 slots pos={position} count={count} first={:?} last={:?}",
                    &ids[..count.min(4)],
                    &ids[count.saturating_sub(4)..]
                );
            }

            if let Some(dir) = std::env::var_os("FF_DS41_LAYER0_DUMP_DIR") {
                let ring = engine.state.latent_ring(0)?;
                let (codes, scales) = (s.clone_dtoh(&ring.codes)?, s.clone_dtoh(&ring.scales)?);
                let mut raw = codes;
                raw.extend_from_slice(&scales);
                std::fs::write(
                    PathBuf::from(dir).join(format!("engine-ring-{position:03}.bin")),
                    raw,
                )?;
            }
            // Exact boundaries: the publisher's own rows on this recorded input, byte for byte.
            // 2026-10-28 dsflash, per 4d ruling 14:00:19Z: the values downstream of the score
            // dot are reported, not asserted; acceptance stays the frozen thresholds-r2 bands.
            let report = |name: &str, got: &[u8]| -> Result<()> {
                let path = reference_dir.join(format!("{name}-{position:03}.bin"));
                let want =
                    std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
                let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                println!(
                    "LAYER0 report {name} pos={position} differing={differing} of {}",
                    got.len()
                );
                Ok(())
            };
            let exact = |name: &str, got: &[u8]| -> Result<()> {
                let path = reference_dir.join(format!("{name}-{position:03}.bin"));
                let want =
                    std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
                ensure!(
                    want.len() == got.len(),
                    "{name} at {position} holds {} bytes; the engine holds {}",
                    want.len(),
                    got.len()
                );
                let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                println!(
                    "LAYER0 exact {name} pos={position} differing={differing} of {}",
                    got.len()
                );
                ensure!(
                    differing == 0,
                    "{name} at {position} differs on {differing} of {} bytes",
                    got.len()
                );
                Ok(())
            };
            let encode = |row: &[bf16]| -> Vec<u8> {
                row.iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect()
            };
            if position < SEED_POSITIONS {
                let normed_row: Vec<bf16> = s.clone_dtoh(&engine.work.n.slice(..HIDDEN))?;
                exact("publisher-xnorm", &encode(&normed_row))?;
                let mixer = engine
                    .deepseek_mixer
                    .as_ref()
                    .context("DeepSeek-V4.1 mixer runtime is not bound")?;
                let (codes, scales) = mixer.activation_codes_host(s, 1, HIDDEN)?;
                exact("publisher-xcodes", &codes)?;
                exact("publisher-xscales", &scales)?;
                report("publisher-qr", &encode(&mixer.query_latent_host(s, 1)?))?;
                let (o_row, _) = mixer.attn_output_host(s, 1)?;
                report("publisher-o", &encode(&o_row))?;
                if position == 18 || position == 19 {
                    let got: Vec<u8> = encode(&o_row);
                    let want = std::fs::read(
                        reference_dir.join(format!("publisher-o-{position:03}.bin")),
                    )?;
                    let per_head: Vec<usize> = (0..64)
                        .map(|head| {
                            let lo = head * 1024;
                            let hi = lo + 1024;
                            got[lo..hi]
                                .iter()
                                .zip(&want[lo..hi])
                                .filter(|(a, b)| a != b)
                                .count()
                        })
                        .collect();
                    println!("LAYER0 o_heads pos={position} {:?}", per_head);
                    let sgot: Vec<String> = (0..4)
                        .map(|d| format!("{:.6}", o_row[d].to_f32()))
                        .collect();
                    let swant: Vec<String> = (0..4)
                        .map(|d| {
                            let lo = d * 2;
                            format!(
                                "{:.6}",
                                bf16::from_bits(u16::from_le_bytes([want[lo], want[lo + 1]]))
                                    .to_f32()
                            )
                        })
                        .collect();
                    println!("LAYER0 o_head0 pos={position} engine={sgot:?} recorded={swant:?}");
                }
            } else {
                println!(
                    "LAYER0 exact skipped pos={position} reason=decode-shape: the reference rows are prefill rows"
                );
            }
            // Rule A (4d ruling 2026-10-28T18:43:08Z): with FF_DS41_LAYER0_ISOLATED_ATTN the
            // sublayer output is the publisher's recorded row, so the boundaries downstream of
            // attention judge our own arithmetic; the default run keeps the propagated numbers
            // as a diagnostic.
            if isolated_attn {
                s.memcpy_htod(want_attn, &mut engine.work.out.slice_mut(..HIDDEN))?;
                // 2026-10-29: rule A includes post/comb and all residual copies.
                let post: Vec<f32> = records.mixes[0].1
                    [index * STREAM_COPIES..(index + 1) * STREAM_COPIES]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let comb: Vec<f32> = records.mixes[0].2[index * STREAM_COPIES * STREAM_COPIES
                    ..(index + 1) * STREAM_COPIES * STREAM_COPIES]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let runtime = engine.hc.as_mut().context(runtime_missing)?;
                s.memcpy_htod(&input_row, &mut runtime.stream.slice_mut(..STREAM_WIDTH))?;
                s.memcpy_htod(&post, &mut runtime.mix_post.slice_mut(..STREAM_COPIES))?;
                s.memcpy_htod(
                    &comb,
                    &mut runtime.mix_comb.slice_mut(..STREAM_COPIES * STREAM_COPIES),
                )?;
            }
            engine.run_hc_stage(3, HcStage::Post, 0, step)?;
            let stream = s.clone_dtoh(&engine.hc.as_ref().context(runtime_missing)?.stream)?;
            let stream = stream[..STREAM_WIDTH].to_vec();
            let want_expand = &records.expand[0][index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH];
            expand_margin.bf16s(position, &stream, want_expand, band("expand_attn", tf));
            let (expand_max, expand_rms) = profile(&stream, want_expand);
            if let Some(dir) = std::env::var_os("FF_DS41_LAYER0_DUMP_DIR") {
                let raw: Vec<u8> = stream
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect();
                std::fs::write(
                    PathBuf::from(dir).join(format!("engine-expand-{position:03}.bin")),
                    raw,
                )?;
            }
            println!(
                "LAYER0 profile pos={position} collapse={collapse_max:.3e}/{collapse_rms:.3e} \
                 attn={attn_max:.3e}/{attn_rms:.3e} expand={expand_max:.3e}/{expand_rms:.3e}"
            );
            // The FFN HcPre consumes the engine's own expanded row, so its FP64
            // reference and its operand difference are read on the same row.
            let expand_differing = stream
                .iter()
                .zip(want_expand)
                .filter(|(got, want)| got.to_bits() != want.to_bits())
                .count();
            let expand_ulps = stream
                .iter()
                .zip(want_expand)
                .map(|(got, want)| {
                    (i32::from(got.to_bits() as i16) - i32::from(want.to_bits() as i16))
                        .unsigned_abs()
                })
                .max()
                .unwrap_or(0);
            println!(
                "LAYER0 expand_operand pos={position} differing={expand_differing} of {STREAM_WIDTH} max_ulp={expand_ulps}"
            );
            // 2026-10-29: rule A isolates FFN mixes/collapse from attention stores.
            let ffn_stream = if isolated_ffn {
                let pre: Vec<f32> = records.mixes[0].0
                    [index * STREAM_COPIES..(index + 1) * STREAM_COPIES]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let runtime = engine.hc.as_mut().context(runtime_missing)?;
                s.memcpy_htod(want_expand, &mut runtime.stream.slice_mut(..STREAM_WIDTH))?;
                s.memcpy_htod(
                    &pre,
                    &mut runtime
                        .attn_pre
                        .as_mut()
                        .context("mHC attention pre-mix missing")?
                        .slice_mut(..STREAM_COPIES),
                )?;
                want_expand
            } else {
                &stream
            };
            // Program stage 4 (HcPre): the FFN mixes and the collapse with the
            // attention's own pre-mix.
            engine.run_hc_stage(4, HcStage::Pre, 0, step)?;
            let runtime = engine.hc.as_ref().context(runtime_missing)?;
            let mix_pre: Vec<f32> = s.clone_dtoh(&runtime.mix_pre.slice(..STREAM_COPIES))?;
            let mix_post: Vec<f32> = s.clone_dtoh(&runtime.mix_post.slice(..STREAM_COPIES))?;
            let mix_comb: Vec<f32> =
                s.clone_dtoh(&runtime.mix_comb.slice(..STREAM_COPIES * STREAM_COPIES))?;
            let want = &records.mixes[1];
            let (pre_f64, post_f64) = hc_mixes_f64(ffn_stream, &hc_ffn_bytes, &params);
            let pre_rows = &want.0[index * STREAM_COPIES..(index + 1) * STREAM_COPIES];
            let post_rows = &want.1[index * STREAM_COPIES..(index + 1) * STREAM_COPIES];
            ffn_pre.f64s(position, &mix_pre, pre_rows, band("ffn_pre", tf));
            ffn_post.f64s(position, &mix_post, post_rows, band("ffn_post", tf));
            ffn_comb.f64s(
                position,
                &mix_comb,
                &want.2[index * STREAM_COPIES * STREAM_COPIES
                    ..(index + 1) * STREAM_COPIES * STREAM_COPIES],
                band("ffn_comb", tf),
            );
            hc_diag(
                "ffn_pre",
                position,
                &mix_pre,
                pre_rows,
                &pre_f64,
                band("ffn_pre", tf),
            );
            hc_diag(
                "ffn_post",
                position,
                &mix_post,
                post_rows,
                &post_f64,
                band("ffn_post", tf),
            );
            let x: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..HIDDEN))?;
            let want_collapse_ffn = &records.collapse[1][index * HIDDEN..(index + 1) * HIDDEN];
            collapse_margin.bf16s(position, &x, want_collapse_ffn, band("collapse_ffn", tf));
            let (collapse_ffn_max, collapse_ffn_rms) = profile(&x, want_collapse_ffn);
            engine.run_operator(Operator::PostNorm, 0, step, &mut None)?;
            engine.run_ffn(0, step, &mut None)?;
            // MoE instrument (4d plan 2026-10-28T19:08:31Z): router logits, chosen ids, applied
            // combine weights, per-choice expert outputs and the shared row.
            if let Some(dir) = std::env::var_os("FF_DS41_LAYER0_DUMP_DIR") {
                let dir = PathBuf::from(dir);
                let stage = engine
                    .deepseek_ffn
                    .as_ref()
                    .context(runtime_missing)?
                    .stage_buffers();
                let logits: Vec<f32> = s.clone_dtoh(&stage.logits.slice(..FF_EXPERTS))?;
                let ids: Vec<i32> = s.clone_dtoh(&stage.ids.slice(..FF_TOP_K))?;
                let weights: Vec<f32> = s.clone_dtoh(&stage.route_weights.slice(..FF_TOP_K))?;
                let contributions: Vec<bf16> =
                    s.clone_dtoh(&stage.contributions.slice(..FF_TOP_K * HIDDEN))?;
                let shared: Vec<bf16> = s.clone_dtoh(&stage.shared_out.slice(..HIDDEN))?;
                let bits = |row: &[bf16]| -> Vec<u8> {
                    row.iter()
                        .flat_map(|value| value.to_bits().to_le_bytes())
                        .collect()
                };
                std::fs::write(
                    dir.join(format!("engine-moe-logits-{position:03}.bin")),
                    logits
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                std::fs::write(
                    dir.join(format!("engine-moe-ids-{position:03}.bin")),
                    ids.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                std::fs::write(
                    dir.join(format!("engine-moe-weights-{position:03}.bin")),
                    weights
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                std::fs::write(
                    dir.join(format!("engine-moe-contrib-{position:03}.bin")),
                    bits(&contributions),
                )?;
                std::fs::write(
                    dir.join(format!("engine-moe-shared-{position:03}.bin")),
                    bits(&shared),
                )?;
            }
            let ffn_out: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?;
            let want_ffn = &records.ffn[0][index * HIDDEN..(index + 1) * HIDDEN];
            ffn_margin.bf16s(position, &ffn_out, want_ffn, band("ffn", tf));
            let (ffn_max, ffn_rms) = profile(&ffn_out, want_ffn);
            // Rule A applied to the FFN chain (4d rulings 2026-10-28T21:19:49Z and 21:24:59Z): an
            // isolated boundary takes ALL its upstream operands from the record. The publisher's
            // recorded FFN row is the sublayer output, and the recorded post/comb are the mix
            // coefficients, so `final_stream` is judged on hc_post's own arithmetic.
            if isolated_ffn {
                let want_ffn_row = &records.ffn[0][index * HIDDEN..(index + 1) * HIDDEN];
                s.memcpy_htod(want_ffn_row, &mut engine.work.out.slice_mut(..HIDDEN))?;
                let mixes = &records.mixes[1];
                let post: Vec<f32> = mixes.1[index * STREAM_COPIES..(index + 1) * STREAM_COPIES]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let comb: Vec<f32> = mixes.2[index * STREAM_COPIES * STREAM_COPIES
                    ..(index + 1) * STREAM_COPIES * STREAM_COPIES]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let runtime = engine.hc.as_mut().context(runtime_missing)?;
                // The residual copies are the record's expanded attention stream.
                s.memcpy_htod(
                    &records.expand[0][index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH],
                    &mut runtime.stream.slice_mut(..STREAM_WIDTH),
                )?;
                s.memcpy_htod(&post, &mut runtime.mix_post.slice_mut(..STREAM_COPIES))?;
                s.memcpy_htod(
                    &comb,
                    &mut runtime.mix_comb.slice_mut(..STREAM_COPIES * STREAM_COPIES),
                )?;
            }
            // Isolated-operand check (4d ruling 2026-10-28T21:24:59Z): the stage-7 launch must
            // see the record's coefficients and residual, or the injection point is wrong.
            if isolated_ffn {
                let runtime = engine.hc.as_ref().context(runtime_missing)?;
                let post_dev: Vec<f32> = s.clone_dtoh(&runtime.mix_post.slice(..STREAM_COPIES))?;
                let comb_dev: Vec<f32> =
                    s.clone_dtoh(&runtime.mix_comb.slice(..STREAM_COPIES * STREAM_COPIES))?;
                let residual_dev: Vec<bf16> =
                    s.clone_dtoh(&runtime.stream.slice(..STREAM_WIDTH))?;
                let want = &records.mixes[1];
                let post_diff = post_dev
                    .iter()
                    .zip(&want.1[index * STREAM_COPIES..(index + 1) * STREAM_COPIES])
                    .filter(|(got, want)| f64::from(**got) != **want)
                    .count();
                let comb_diff = comb_dev
                    .iter()
                    .zip(
                        &want.2[index * STREAM_COPIES * STREAM_COPIES
                            ..(index + 1) * STREAM_COPIES * STREAM_COPIES],
                    )
                    .filter(|(got, want)| f64::from(**got) != **want)
                    .count();
                let residual_diff = residual_dev
                    .iter()
                    .zip(&records.expand[0][index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH])
                    .filter(|(got, want)| *got != *want)
                    .count();
                println!(
                    "LAYER0 isolated_operands pos={position} post_diff={post_diff} comb_diff={comb_diff} residual_diff={residual_diff}"
                );
            }
            engine.run_hc_stage(7, HcStage::Post, 0, step)?;
            let final_row = s.clone_dtoh(&engine.hc.as_ref().context(runtime_missing)?.stream)?;
            let final_row = final_row[..STREAM_WIDTH].to_vec();
            let want_final = &records.expand[1][index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH];
            final_stream.bf16s(position, &final_row, want_final, band("expand_ffn", tf));
            let (final_max, final_rms) = profile(&final_row, want_final);
            println!(
                "LAYER0 profile_ffn pos={position} collapse={collapse_ffn_max:.3e}/{collapse_ffn_rms:.3e} \
                 ffn={ffn_max:.3e}/{ffn_rms:.3e} final={final_max:.3e}/{final_rms:.3e}"
            );
            let carried: Vec<f32> = s.clone_dtoh(
                &engine
                    .hc
                    .as_ref()
                    .context(runtime_missing)?
                    .carry
                    .as_ref()
                    .context(carry_missing)?
                    .slice(..STREAM_COPIES),
            )?;
            carried_margin.f64s(
                position,
                &carried,
                &records.mixes[1].0[index * STREAM_COPIES..(index + 1) * STREAM_COPIES],
                band("carried_pre", tf),
            );
            if position == 1 {
                // Perturbed-code control: one ring code byte moved must change the attention
                // output, or the injection is not reaching the kernel. Runs after this
                // position's comparisons: it rewrites ring slot 0 and work.out.
                let mut codes = vec![0u8; window * geometry.head_dim];
                let mut scales = vec![0u8; (window * geometry.head_dim) / 32];
                for pos in 0..1usize {
                    let slot = pos % window;
                    codes[slot * geometry.head_dim..(slot + 1) * geometry.head_dim]
                        .copy_from_slice(&seed_k_codes[pos]);
                    scales[slot * (geometry.head_dim / 32)..(slot + 1) * (geometry.head_dim / 32)]
                        .copy_from_slice(&seed_k_scales[pos]);
                }
                let slot = 0usize;
                codes[slot * geometry.head_dim] = codes[slot * geometry.head_dim].wrapping_add(1);
                let ring = engine.state.latent_ring_mut(0)?;
                s.memcpy_htod(&codes, &mut ring.codes)?;
                s.memcpy_htod(&scales, &mut ring.scales)?;
                engine.run_mixer(mixer_kind, 0, step, &mut None)?;
                let perturbed: Vec<bf16> = s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?;
                let moved = perturbed
                    .iter()
                    .zip(&attn_out)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!("LAYER0 inject_control pos=1 moved={moved} of {HIDDEN}");
                ensure!(
                    moved > 0,
                    "perturbed ring code did not move the attention output: the injection is not reaching the kernel",
                );
            }
        }
        ensure!(
            control_beyond > 0,
            "attention-own pre-mix control does not discriminate: {control_beyond} of {control_compared} collapse elements beyond tolerance",
        );
        // Planted wrong-slot control at the fully teacher-forced decode
        // position (the mixer-gate teeth): a foreign row lands in ring slot 5
        // (position 5's K, visible in the 128-wide window at position 129;
        // the step itself writes slot 129 % 128 = 1). Stages 0-1 re-run on
        // the re-injected stream row and carry reproduce x and n
        // bit-exactly, the mixer rewrites its own slot identically, and the
        // planted slot must move the attention output beyond the tolerance.
        let wrong_slot_beyond;
        let wrong_carried_beyond;
        {
            let decode_stream = decode_steps.stream.clone();
            let decode_carried: Vec<f32> = decode_steps
                .carried
                .as_ref()
                .context("input decode record carries no carried_pre_mix recording")?
                .iter()
                .map(|&value| value as f32)
                .collect();
            let mut replay_decode =
                |ring_plant: Option<(usize, u8, u8)>, carry: &[f32]| -> Result<Vec<bf16>> {
                    engine.prepare_step(step)?;
                    engine.position = SEED_POSITIONS;
                    engine.state.set_sequence(
                        s,
                        SEED_POSITIONS as u32,
                        1,
                        engine.workspace_rows as u32,
                    )?;
                    inject_ring!(SEED_POSITIONS);
                    if let Some((control_slot, code, scale)) = ring_plant {
                        let nblocks = geometry.head_dim / 32;
                        let (mut codes_host, mut scales_host) = {
                            let ring = engine.state.latent_ring(0)?;
                            (s.clone_dtoh(&ring.codes)?, s.clone_dtoh(&ring.scales)?)
                        };
                        for d in 0..geometry.head_dim {
                            codes_host[control_slot * geometry.head_dim + d] = code;
                        }
                        for block in 0..nblocks {
                            scales_host[control_slot * nblocks + block] = scale;
                        }
                        let ring = engine.state.latent_ring_mut(0)?;
                        s.memcpy_htod(&codes_host, &mut ring.codes)?;
                        s.memcpy_htod(&scales_host, &mut ring.scales)?;
                    }
                    s.memcpy_htod(
                        &decode_stream,
                        &mut engine
                            .hc
                            .as_mut()
                            .context(runtime_missing)?
                            .stream
                            .slice_mut(..STREAM_WIDTH),
                    )?;
                    s.memcpy_htod(
                        carry,
                        &mut engine
                            .hc
                            .as_mut()
                            .context(runtime_missing)?
                            .carry
                            .as_mut()
                            .context(carry_missing)?
                            .slice_mut(..STREAM_COPIES),
                    )?;
                    engine.run_hc_stage(0, HcStage::Pre, 0, step)?;
                    engine.run_operator(Operator::Norm, 0, step, &mut None)?;
                    engine.run_mixer(mixer_kind, 0, step, &mut None)?;
                    Ok(s.clone_dtoh(&engine.work.out.slice(..HIDDEN))?)
                };
            let clean_attn = replay_decode(None, &decode_carried)?;
            let planted_attn = replay_decode(Some((5, 0x7Eu8, 0x89u8)), &decode_carried)?;
            wrong_slot_beyond = planted_attn
                .iter()
                .zip(&clean_attn)
                .zip(&decode_steps.attn[0])
                .filter(|((got, clean), reference)| {
                    (f64::from(got.to_f32()) - f64::from(clean.to_f32())).abs() > band("attn", 8)
                        && (f64::from(got.to_f32()) - f64::from(reference.to_f32())).abs()
                            > band("attn", 8)
                })
                .count();
            // Wrong-carried-state control (ruling (4)): the negated publisher
            // carry must move the collapse beyond the frozen bound; the carry
            // enters hc_pre linearly with the stream copies.
            let negated: Vec<f32> = decode_carried.iter().map(|value| -value).collect();
            engine.prepare_step(step)?;
            engine.position = SEED_POSITIONS;
            engine
                .state
                .set_sequence(s, SEED_POSITIONS as u32, 1, engine.workspace_rows as u32)?;
            inject_ring!(SEED_POSITIONS);
            s.memcpy_htod(
                &decode_stream,
                &mut engine
                    .hc
                    .as_mut()
                    .context(runtime_missing)?
                    .stream
                    .slice_mut(..STREAM_WIDTH),
            )?;
            s.memcpy_htod(
                &negated,
                &mut engine
                    .hc
                    .as_mut()
                    .context(runtime_missing)?
                    .carry
                    .as_mut()
                    .context(carry_missing)?
                    .slice_mut(..STREAM_COPIES),
            )?;
            engine.run_hc_stage(0, HcStage::Pre, 0, step)?;
            let control_x: Vec<bf16> = s.clone_dtoh(&engine.work.x.slice(..HIDDEN))?;
            wrong_carried_beyond = control_x
                .iter()
                .zip(&decode_steps.collapse[0])
                .filter(|(got, want)| {
                    (f64::from(got.to_f32()) - f64::from(want.to_f32())).abs()
                        > band("collapse_attn", 8)
                })
                .count();
        }
        ensure!(
            wrong_slot_beyond > 0,
            "planted wrong-slot ring control escaped ({wrong_slot_beyond} of {HIDDEN} attention elements beyond tolerance)",
        );
        ensure!(
            wrong_carried_beyond > 0,
            "wrong-carried-state control escaped ({wrong_carried_beyond} of {HIDDEN} collapse elements beyond tolerance)",
        );
        let margins = [
            &attn_pre,
            &attn_post,
            &attn_comb,
            &ffn_pre,
            &ffn_post,
            &ffn_comb,
            &collapse_margin,
            &attn_margin,
            &expand_margin,
            &ffn_margin,
            &final_stream,
            &carried_margin,
        ];
        let informational = [&info_collapse, &info_attn, &info_ffn, &info_final];
        // Per-boundary statistics print before the asserts: a failing run
        // still carries the full spread for the threshold freeze.
        for margin in &margins {
            println!(
                "LAYER0 boundary={} compared={} beyond={} max_abs={:.3e} rule={}",
                margin.name, margin.compared, margin.beyond, margin.max_abs, margin.rule,
            );
        }
        for margin in &informational {
            println!(
                "LAYER0 informational={} compared={} max_abs={:.3e}",
                margin.name, margin.compared, margin.max_abs,
            );
        }
        println!(
            "LAYER0 layer=0 run=input steps={} copies={STREAM_COPIES} hidden={HIDDEN} window={} pattern_mismatches={pattern_mismatches} fixture_control_differing_elements={differing}",
            SEED_POSITIONS + DECODE_POSITIONS,
            window,
        );
        // Gate implementation note 2026-10-28T21:19:49Z (4d ruling): rule A also covers the
        // final expand. `final_stream` is judged with the publisher's recorded FFN row injected
        // (FF_DS41_LAYER0_ISOLATED_FFN), where hc_post is bit-replayed and the boundary must hold
        // on its own; the propagated number without the injection is a diagnostic. The FFN row
        // itself is judged at the `ffn` boundary.
        // Gate implementation note 2026-10-28T18:43:08Z (4d ruling, applying the F6 rules).
        // Rule A: the boundaries downstream of attention are judged on the publisher's recorded
        // attention row (FF_DS41_LAYER0_ISOLATED_ATTN, F6 rule A); the propagated numbers stay a
        // diagnostic. Rule B: an attention element beyond its band is exempt only when a P_j of
        // that head lies within the score-dot analytic bound of a BF16 rounding boundary, proven
        // per element in the attn_near_tie line. Every other beyond element is a defect.
        for margin in &margins {
            if margin.name == "attn" {
                ensure!(
                    margin.beyond == attn_exempt,
                    "attention boundary holds {} of {} elements beyond tolerance but only {} carry a BF16 near-tie proof",
                    margin.beyond,
                    margin.compared,
                    attn_exempt,
                );
                continue;
            }
            ensure!(
                margin.beyond == 0,
                "boundary {} holds {} of {} compared elements beyond tolerance, first {:?}",
                margin.name,
                margin.beyond,
                margin.compared,
                margin.first,
            );
        }
        ensure!(
            attn_defect == 0,
            "attention boundary holds {attn_defect} elements beyond tolerance with no BF16 near-tie proof (exempt {attn_exempt})",
        );
        println!(
            "PASS d1_layer0_matches_model_definition mode={} boundaries={} tf_positions={} attn_exempt={} attn_defect={} attention_own_control_beyond={}/{} wrong_slot_control_beyond={}/{} wrong_carried_control_beyond={}/{}",
            if isolated_attn {
                "isolated-attn"
            } else {
                "propagated"
            },
            margins.len(),
            TF_POSITIONS.len(),
            attn_exempt,
            attn_defect,
            control_beyond,
            control_compared,
            wrong_slot_beyond,
            HIDDEN,
            wrong_carried_beyond,
            HIDDEN,
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(&runtime_dir);
        Ok(())
    }
}
