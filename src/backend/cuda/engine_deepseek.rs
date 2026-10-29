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
        setup.validate(&setup.config_sha256, &setup.tokenizer_sha256)?;
        setup.validate_samples()?;
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
/// DeepSeek-V4.1 binding of the shared mHC runtime: the fn projection stays
/// in its published F32 storage, the update rounds once at the output, the
/// comb applies plain, the stream RMS epsilon is rms_eps, and the pre-source
/// policy is the architecture's (carried attention, attention-own FFN).
/// The DeepSeek RMS-norm contract: F32 arithmetic with one BF16 rounding at the store.
/// `FF_DS41_NORM_LEGACY` selects the pre-fix contract for the mode-0 control run.
fn deepseek_rms_mode() -> RmsMode {
    if std::env::var_os("FF_DS41_NORM_LEGACY").is_some() {
        return RmsMode::Standard;
    }
    RmsMode::Unrounded
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
            mode: deepseek_rms_mode(),
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
        // 0 is the definition's attention boundaries; 1 is the pre-fix arithmetic, for the
        // gate's control.
        let p_mode: i32 = if std::env::var_os("FF_DS41_LWA_FP32_P").is_some() {
            1
        } else {
            0
        };
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
        let (table, valid) = slots.deepseek_dispatch(layer_u32)?;
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
                mode: deepseek_rms_mode(),
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
                mode: deepseek_rms_mode(),
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
            let misses = slots.resolve_deepseek(
                u32::try_from(layer).context("DeepSeek-V4.1 tier layer overflow")?,
                &ids,
                &weights,
            )?;
            slots.wait_refill(s)?;
            self.sync_tier(s, layer, slots)?;
            println!(
                "DEEPSEEK_TIER_RESOLVE layer={layer} rows={rows} selected={} misses={misses}",
                ids.len()
            );
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
                .deepseek
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
                deepseek,
                deepseek_ffn,
                work,
                device,
                state,
                weights,
                ..
            } = self;
            let hc = deepseek
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
        let fixed = dense_bytes
            .checked_add(state.bytes())
            .and_then(|n| n.checked_add(workspace_rows * row_bytes))
            .and_then(|n| n.checked_add(dsa::scratch_bytes(workspace_rows, capacity)))
            .and_then(|n| n.checked_add(512 * 1024 * 1024))
            .context("DeepSeek-V4.1 fixed residency overflow")?;
        let (free, total) = device.ctx.mem_get_info()?;
        let available = free.checked_sub(total / 16 + fixed).context(
            "DeepSeek-V4.1 dense resident weights/workspace exceed the device memory budget",
        )?;
        let admitted_total = available / slot_bytes;
        let uniform = (admitted_total / layers.max(1)).min(geometry.experts as usize);
        let tiered = std::env::var("FF_DS41_EXPERT_TIERS").as_deref() == Ok("1");
        // Gate-load scope (not a production placement knob): one selected
        // layer's complete expert set is admitted first; the remaining layers
        // share the leftover admission as a uniform prefix. Requires the total
        // budget to hold one full layer plus at least one expert per
        // remaining layer.
        let legacy_layer0_full =
            std::env::var("FF_DS41_LAYER0_FULL_RESIDENCY").as_deref() == Ok("1");
        let full_residency_layer = match std::env::var("FF_DS41_FULL_RESIDENCY_LAYER") {
            Ok(value) => value
                .parse::<usize>()
                .context("FF_DS41_FULL_RESIDENCY_LAYER must be a layer index")?,
            Err(_) if legacy_layer0_full => 0,
            Err(_) => usize::MAX,
        };
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
        } else if full_residency {
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
        } else {
            (uniform, uniform)
        };
        if !tiered {
            ensure!(
                resident_experts > 0,
                "DeepSeek-V4.1 routed-expert residency admits zero of the {} bytes/expert; the tiered expert service is not bound (dense fixed bytes {fixed})",
                slot_bytes
            );
        }
        if !full_residency && !tiered {
            eprintln!(
                "{}",
                serde_json::json!({"deepseek_routed_residency": {
                    "experts_per_layer": resident_experts,
                    "of": geometry.experts,
                    "slot_bytes": slot_bytes,
                    "dense_bytes": dense_bytes,
                    "tiered_service": "not bound; decode traps when routed outside the resident prefix"
                }})
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
            deepseek: Some(hc),
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
