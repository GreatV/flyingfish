//! Qwen bindings and operators of the shared Engine.
use super::*;
use crate::backend::cuda::contracts::view;
use crate::backend::cuda::{
    attention_d256::{AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings},
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    flash::{Flash, Score},
    gdn::{
        CUDA_GRID_YZ_MAX, GdnBindings, GdnConvWeights, GdnDecayWeights, GdnGateInput, GdnOps,
        GdnPlan,
    },
    linear::{BoundFfn, LayerLinearPlan, LinearWeightRef},
    ops::{CommonOpsRef, GatedRmsInput, ResidualRmsCapture, ResidualRmsInput, RmsMode, RmsOptions},
    weights::{StoredTensor, WeightCatalog},
};
/// The workspace alignment the default-chunk cap must preserve; the same value the budget call passes.
const CHUNK_ALIGNMENT: usize = 32;
/// The default chunk is bounded by the largest alignment multiple at or under the CUDA gridDim.y/z limit the GDN prep pair launches rows on.
pub(super) fn default_chunk(admitted: usize) -> usize {
    admitted.min(CUDA_GRID_YZ_MAX as usize / CHUNK_ALIGNMENT * CHUNK_ALIGNMENT)
}

use crate::backend::cuda::cubin;
use crate::backend::cuda::decode::{GqaDecodeInput, SlidingWindow};
use crate::backend::cuda::dsa::{
    DsaAppendInput, DsaAttentionInput, DsaIndexInput, DsaIndexOutput, DsaPlan,
};
use crate::backend::gpu::LaunchConfig;
use crate::backend::gpu::{CudaFunction, DevicePtr};
use crate::backend::{HcLayerWeights, HcRuntime, HcSublayerWeights};
use crate::gdn_plan::GdnPolicy;
use crate::work_layout::DsaRowLayout;
use crate::{
    linear_plan::LayerDequantPlan,
    model_spec::Mixer,
    op_contract::{AttentionLimits, ExecutionPhase, FullAttentionSpec, GdnSpec},
};
use std::collections::BTreeSet;

fn all_layers(trace: &Option<Target<'_>>) -> bool {
    trace.as_ref().is_some_and(|t| t.trace.all_layers)
}

fn debug_name(trace: &Option<Target<'_>>, layer: usize, point: &str) -> String {
    if all_layers(trace) {
        format!("{layer}.debug.{point}")
    } else {
        format!("debug.{point}")
    }
}

fn norm_mode(spec: &crate::model_spec::ModelSpec) -> RmsMode {
    match spec.architecture {
        crate::model_spec::Architecture::Gemma4 => RmsMode::Unrounded,
        crate::model_spec::Architecture::Flint | crate::model_spec::Architecture::Glm5Next => {
            RmsMode::Standard
        }
        _ => RmsMode::ZeroCentered,
    }
}

#[cfg(test)]
struct FlintWindowInputs<'a> {
    spec: &'a crate::model_spec::ModelSpec,
    layer: usize,
    sliding: bool,
    ring: usize,
    position: &'a CudaSlice<i32>,
    qkv: &'a CudaSlice<bf16>,
    attention: &'a CudaSlice<bf16>,
    kv: &'a Kv,
    sinks: &'a CudaSlice<bf16>,
}

#[cfg(test)]
fn capture_flint_window(
    stream: &Arc<CudaStream>,
    device_name: &str,
    inputs: FlintWindowInputs<'_>,
    capture: &Option<(Vec<usize>, Vec<usize>, PathBuf)>,
) -> Result<()> {
    let FlintWindowInputs {
        spec,
        layer,
        sliding,
        ring,
        position,
        qkv,
        attention,
        kv,
        sinks,
    } = inputs;
    let Some((layers, positions, directory)) = capture else {
        return Ok(());
    };
    if !layers.contains(&layer) {
        return Ok(());
    }
    let host_position = stream.clone_dtoh(position)?[0];
    ensure!(host_position >= 0, "negative Flint window position");
    let logical_position = host_position as usize;
    if !positions.contains(&logical_position) {
        return Ok(());
    }
    let logical_length = if sliding {
        (logical_position + 1).min(ring)
    } else {
        logical_position + 1
    };
    let first = (logical_position + 1 - logical_length) % ring;
    let ring_slots = (0..logical_length)
        .map(|offset| (first + offset) % ring)
        .collect::<Vec<_>>();
    let logical_positions = (logical_position + 1 - logical_length..=logical_position)
        .map(|position| position as f32)
        .collect::<Vec<_>>();
    let head_dim = spec.head_dim as usize;
    let q_heads = spec.q_heads as usize;
    let q_width = spec.q_width as usize;
    let kv_heads = spec.kv_heads as usize;
    let kv_values = ring * kv_heads * head_dim;
    let mut trace = Trace::default();
    trace.set_device(device_name)?;
    trace.add(
        "q".to_owned(),
        vec![q_heads, head_dim],
        stream
            .clone_dtoh(&qkv.slice(..q_width))?
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
    )?;
    trace.add(
        "k".to_owned(),
        vec![kv_heads, ring, head_dim],
        stream
            .clone_dtoh(&kv.k.slice(..kv_values))?
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
    )?;
    trace.add(
        "v".to_owned(),
        vec![kv_heads, ring, head_dim],
        stream
            .clone_dtoh(&kv.v.slice(..kv_values))?
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
    )?;
    trace.add(
        "attention".to_owned(),
        vec![q_heads, head_dim],
        stream
            .clone_dtoh(&attention.slice(..q_width))?
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
    )?;
    trace.add(
        "sinks".to_owned(),
        vec![q_heads],
        stream
            .clone_dtoh(sinks)?
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
    )?;
    trace.add(
        "ring_slots".to_owned(),
        vec![logical_length],
        ring_slots.iter().map(|slot| *slot as f32).collect(),
    )?;
    trace.add(
        "logical_positions".to_owned(),
        vec![logical_length],
        logical_positions,
    )?;
    trace.add(
        "metadata".to_owned(),
        vec![5],
        vec![
            layer as f32,
            logical_position as f32,
            logical_length as f32,
            first as f32,
            u8::from(sliding) as f32,
        ],
    )?;
    std::fs::create_dir_all(directory)?;
    trace.save(&directory.join(format!(
        "layer{layer}_position{logical_position}.safetensors"
    )))?;
    Ok(())
}

fn require_qwen35_geometry(spec: &crate::model_spec::ModelSpec) -> Result<()> {
    use crate::model_spec::{Architecture, LinearFormat};
    let (hidden, intermediate, vocab, hq, hkv, layers, hv, format) = match spec.architecture {
        Architecture::Qwen35Text => (
            5120,
            17408,
            248320,
            24,
            4,
            64,
            48,
            LinearFormat::fp8_block([128, 128], safetensors::Dtype::BF16)?,
        ),
        Architecture::MiniCpmV47 => {
            ensure!(
                matches!(
                    spec.linear_format,
                    LinearFormat::Bf16 | LinearFormat::Fp8E4m3Channel
                ),
                "hybrid engine does not implement linear_format={:?} for {:?}",
                spec.linear_format,
                spec.architecture
            );
            (2048, 512, 248144, 16, 2, 40, 32, spec.linear_format)
        }
        Architecture::Llama
        | Architecture::Flint
        | Architecture::Glm5Next
        | Architecture::LoopVl
        | Architecture::Gemma4 => anyhow::bail!(
            "hybrid engine does not implement architecture={:?}",
            spec.architecture
        ),
        Architecture::DeepseekV41 => anyhow::bail!(
            "hybrid engine does not implement architecture={:?}",
            spec.architecture
        ),
    };
    for (name, value, expected) in [
        ("hidden_size", spec.hidden, hidden),
        ("intermediate_size", spec.intermediate, intermediate),
        ("vocab_size", spec.vocab, vocab),
        ("num_attention_heads", spec.q_heads, hq),
        ("num_key_value_heads", spec.kv_heads, hkv),
        ("head_dim", spec.head_dim, 256),
        ("q_width", spec.q_width, hq * 256),
        ("max_position_embeddings", spec.max_position, 262144),
        ("rotary_dim", spec.rotary_dim, 64),
    ] {
        ensure!(
            value == expected,
            "hybrid engine does not implement {name}={value}"
        );
    }
    ensure!(
        spec.layers.len() == layers,
        "hybrid engine does not implement num_hidden_layers={}",
        spec.layers.len()
    );
    ensure!(
        spec.rms_eps == 1e-6,
        "hybrid engine does not implement rms_norm_eps={}",
        spec.rms_eps
    );
    ensure!(
        spec.rope_theta == 1e7,
        "hybrid engine does not implement rope_theta={}",
        spec.rope_theta
    );
    ensure!(
        spec.linear_format == format,
        "hybrid engine does not implement linear_format={:?} for {:?}",
        spec.linear_format,
        spec.architecture
    );
    let gdn = spec
        .gdn
        .as_ref()
        .context("hybrid engine does not implement gdn=None")?;
    for (name, value, expected) in [
        ("linear_num_key_heads", gdn.key_heads, 16),
        ("linear_num_value_heads", gdn.value_heads, hv),
        ("linear_key_head_dim", gdn.key_head_dim, 128),
        ("linear_value_head_dim", gdn.value_head_dim, 128),
        ("linear_conv_kernel_dim", gdn.conv_kernel, 4),
    ] {
        ensure!(
            value == expected,
            "hybrid engine does not implement {name}={value}"
        );
    }
    Ok(())
}

fn require_glm_geometry(spec: &crate::model_spec::ModelSpec) -> Result<()> {
    use crate::model_spec::{Architecture, LinearFormat};
    ensure!(
        spec.architecture == Architecture::Glm5Next,
        "GLM runtime received architecture={:?}",
        spec.architecture
    );
    let glm = spec
        .glm5_next
        .as_ref()
        .context("GLM-5.3 geometry missing")?;
    for (name, value, expected) in [
        ("hidden_size", spec.hidden, 4096),
        ("intermediate_size", spec.intermediate, 12288),
        ("vocab_size", spec.vocab, 154_880),
        ("num_attention_heads", spec.q_heads, 64),
        ("num_key_value_heads", spec.kv_heads, 64),
        ("head_dim", spec.head_dim, 256),
        ("q_width", spec.q_width, 16_384),
        ("max_position_embeddings", spec.max_position, 1_048_576),
        ("rotary_dim", spec.rotary_dim, 0),
    ] {
        ensure!(value == expected, "GLM runtime implements {name}={value}");
    }
    ensure!(
        spec.layers.len() == 45,
        "GLM runtime implements {} layers",
        spec.layers.len()
    );
    ensure!(
        spec.rms_eps == 1e-5,
        "GLM runtime implements rms_norm_eps={}",
        spec.rms_eps
    );
    ensure!(
        spec.rope_theta == 1.0,
        "GLM runtime implements rope_theta={}",
        spec.rope_theta
    );
    ensure!(
        spec.linear_format == LinearFormat::fp8_block([128, 128], safetensors::Dtype::F32)?,
        "GLM runtime implements linear_format={:?}",
        spec.linear_format
    );
    let moe = spec.moe.context("GLM MoE geometry missing")?;
    ensure!(
        moe.experts == 288
            && moe.experts_per_tok == 8
            && moe.expert_intermediate == 2048
            && moe.shared_intermediate == 2048
            && moe.score == crate::model_spec::RouterScore::Sigmoid
            && !moe.shared_gate,
        "GLM runtime implements MoE geometry {moe:?}"
    );
    let gdn = spec.gdn.context("GLM KDA geometry missing")?;
    ensure!(
        gdn.key_heads == 64
            && gdn.value_heads == 64
            && gdn.key_head_dim == 128
            && gdn.value_head_dim == 128
            && gdn.conv_kernel == 4,
        "GLM runtime implements KDA geometry {gdn:?}"
    );
    ensure!(
        glm.dense_layers == 3
            && glm.nextn_layers == 1
            && glm.hc_mult == 4
            && glm.hc_sinkhorn_iters == 20
            && glm.hc_eps == 1e-6
            && glm.q_lora_rank == 1536
            && glm.kv_lora_rank == 512
            && glm.qk_nope_head_dim == 256
            && glm.v_head_dim == 256
            && glm.index_heads == 32
            && glm.index_head_dim == 128
            && glm.index_topk == 2048
            && glm.index_kpool == 4
            && glm.gate_lower_bound == -5.0
            && glm.swiglu_limit == 10.0
            && glm.routed_scaling_factor == 2.5,
        "GLM runtime implements text geometry {glm:?}"
    );
    crate::op_contract::DsaSpec::from_model(spec)?.validate()?;
    ensure!(
        spec.layers
            .iter()
            .enumerate()
            .all(|(layer, plan)| plan.mixer
                == if (layer + 1).is_multiple_of(4) {
                    Mixer::FullAttention
                } else {
                    Mixer::Gdn
                }),
        "GLM runtime implements the published KDA/DSA schedule"
    );
    Ok(())
}

fn require_flint_geometry(spec: &crate::model_spec::ModelSpec) -> Result<()> {
    use crate::model_spec::{Architecture, FfnKind, LinearFormat, Mixer};
    ensure!(
        spec.architecture == Architecture::Flint
            && spec.hidden == 2048
            && spec.intermediate == 6144
            && spec.vocab == 150144
            && spec.q_heads == 32
            && spec.kv_heads == 4
            && spec.head_dim == 128
            && spec.q_width == 4096
            && spec.max_position == 262144
            && spec.rms_eps == 1e-6
            && spec.rope_theta == 10000.0
            && spec.rotary_dim == 128
            && spec.linear_format == LinearFormat::Bf16
            && spec.gdn.is_none()
            && spec.vision.is_none(),
        "unsupported Flint engine geometry: {spec:?}"
    );
    ensure!(spec.layers.len() == 49, "unsupported Flint layer count");
    ensure!(
        spec.layers.iter().enumerate().all(|(i, layer)| {
            layer.mixer == Mixer::FullAttention
                && layer.state_index as usize == i
                && (layer.rope || layer.sliding_window.is_none())
        }),
        "unsupported Flint layer plan"
    );
    ensure!(
        spec.layers
            .iter()
            .filter(|layer| layer.ffn == FfnKind::Dense)
            .count()
            == 2
            && spec
                .layers
                .iter()
                .filter(|layer| layer.ffn == FfnKind::Moe)
                .count()
                == 47,
        "unsupported Flint FFN layer plan"
    );
    let moe = spec.moe.context("Flint MoE geometry missing")?;
    ensure!(
        moe.experts == 256
            && moe.experts_per_tok == 6
            && moe.expert_intermediate == 512
            && moe.shared_intermediate == 1024
            && moe.score == crate::model_spec::RouterScore::Sigmoid
            && !moe.shared_gate,
        "unsupported Flint MoE geometry: {moe:?}"
    );
    Ok(())
}

fn take_glm_hc_weights(
    catalog: &mut WeightCatalog,
    spec: &crate::model_spec::ModelSpec,
) -> Result<Vec<HcLayerWeights>> {
    let mut weights = Vec::with_capacity(spec.layers.len());
    let hidden = spec.hidden as usize;
    for layer in 0..spec.layers.len() {
        let prefix = format!("model.language_model.layers.{layer}");
        let mut take = |suffix: &str, dtype: safetensors::Dtype, shape: &[usize]| {
            let name = format!("{prefix}.{suffix}");
            let entry = catalog
                .manifest
                .entries
                .get(&name)
                .with_context(|| format!("missing GLM mHC tensor {name}"))?;
            ensure!(
                entry.dtype == dtype && entry.shape == shape,
                "GLM mHC tensor mismatch {name}: dtype={:?} shape={:?}",
                entry.dtype,
                entry.shape
            );
            match catalog.tensors.remove(&name) {
                Some(StoredTensor::Bytes(value)) => Ok(value),
                _ => anyhow::bail!("GLM mHC tensor is not uploaded native bytes: {name}"),
            }
        };
        let mut sublayer = |base: &str| -> Result<HcSublayerWeights> {
            Ok(HcSublayerWeights {
                fn_w: take(
                    &format!("{base}_fn"),
                    safetensors::Dtype::BF16,
                    &[24, 4 * hidden],
                )?,
                base: take(&format!("{base}_base"), safetensors::Dtype::F32, &[24])?,
                scale: take(&format!("{base}_scale"), safetensors::Dtype::F32, &[3])?,
            })
        };
        weights.push(HcLayerWeights {
            attn: sublayer("hc_attn")?,
            ffn: sublayer("hc_ffn")?,
        });
    }
    Ok(weights)
}

pub(super) struct GlmDsaOutputs {
    packed: CudaSlice<bf16>,
    pub(super) selected: CudaSlice<i32>,
    pub(super) scores: CudaSlice<f32>,
    pub(super) margins: CudaSlice<f32>,
    pub(super) indices: CudaSlice<i32>,
}

impl GlmDsaOutputs {
    fn new(rows: usize, s: &Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            packed: s.alloc_zeros(rows * 257)?,
            selected: s.alloc_zeros(rows * 512)?,
            scores: s.alloc_zeros(rows * 512)?,
            margins: s.alloc_zeros(rows)?,
            indices: s.alloc_zeros(rows * 2051)?,
        })
    }

    fn pack_index(
        &mut self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<bf16>,
        starts: [usize; 2],
        rows: usize,
    ) -> Result<()> {
        for (column, start) in starts.into_iter().enumerate() {
            super::super::copy::columns(
                stream,
                &input.slice(start..start + rows * 128),
                &mut self.packed,
                super::super::copy::Rect {
                    rows,
                    width: 128,
                    src_pitch: 128,
                    src_col: 0,
                    dst_pitch: 257,
                    dst_col: column * 128,
                },
            )?;
        }
        Ok(())
    }
}

fn require_gemma_geometry(spec: &crate::model_spec::ModelSpec) -> Result<()> {
    ensure!(
        spec.architecture == crate::model_spec::Architecture::Gemma4
            && spec.linear_format == crate::model_spec::LinearFormat::Bf16
            && spec.moe.is_none()
            && spec.gdn.is_none()
            && spec.per_layer_input == Some(256),
        "unsupported Gemma 4 execution geometry"
    );
    for layer in &spec.layers {
        let g = layer
            .geometry
            .as_ref()
            .context("Gemma 4 layer geometry missing")?;
        let full = g.head_dim == 512;
        ensure!(
            (full || g.head_dim == 256)
                && layer.sliding_window.is_some() != full
                && layer.mixer == Mixer::FullAttention
                && g.intermediate > 0
                && g.rope.pair_stride == g.head_dim / 2
                && g.rope.denominator == g.head_dim
                && g.rope.pairs == if full { 64 } else { 128 },
            "unsupported Gemma 4 layer geometry: {g:?}"
        );
    }
    Ok(())
}

struct GemmaOps {
    scaled_embed: CudaFunction,
    softcap: CudaFunction,
    gated_activation: CudaFunction,
    rope: CudaFunction,
    ring_copy: CudaFunction,
    branch_add: CudaFunction,
    multiply: CudaFunction,
    per_layer: CudaFunction,
    rope_tables: [CudaSlice<bf16>; 2],
    per_layer_table: Tensor<bf16, 2>,
    per_layer_projection: Tensor<bf16, 2>,
    per_layer_norm: Tensor<bf16, 1>,
    per_layer_inputs: CudaSlice<bf16>,
    #[cfg(test)]
    logits: CudaSlice<bf16>,
    #[cfg(test)]
    mlp_ones: CudaSlice<bf16>,
    #[cfg(test)]
    mlp_activation: CudaSlice<bf16>,
}

pub(super) struct Runtime {
    attention: Option<FullAttentionSpec>,
    gdn: Option<GdnSpec>,
    gemma: Option<GemmaOps>,
    checkpoint_dir: PathBuf,
    vision: Option<super::vision_runtime::VisionRuntime>,
    vision_bytes: usize,
    #[cfg(test)]
    pub(super) vision_encode_calls: usize,
    #[cfg(test)]
    pub(super) glm_calls: Option<std::cell::RefCell<Vec<[usize; 10]>>>,
    positions: CudaSlice<i32>,
    active_positions: CudaSlice<i32>,
    gather_positions: CudaFunction,
    catalog: WeightCatalog,
    arena: Option<CudaSlice<u8>>,
    scratch: CudaSlice<u8>,
    kv_scratch: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
    pub(super) prompt: Option<CudaSlice<bf16>>,
    mask: CudaSlice<u64>,
    layers: Vec<Layer>,
    common: CommonOpsRef,
    advance: CudaFunction,
    prepared: BTreeSet<u32>,
    workspace_rows: u32,
    #[cfg(feature = "cuda")]
    verify_rows: usize,
}
struct Layer {
    linear: LayerLinearPlan,
    ffn: BoundFfn,
    native: Option<NativeFfn>,
    input_norm: Tensor<bf16, 1>,
    post_norm: Tensor<bf16, 1>,
    pre_ffn_norm: Option<Tensor<bf16, 1>>,
    post_ffn_norm: Option<Tensor<bf16, 1>>,
    post_ple_norm: Option<Tensor<bf16, 1>>,
    layer_scalar: Option<Tensor<bf16, 1>>,
    mixer: BoundMixer,
}

struct NativeFfn {
    #[cfg(test)]
    wrong_route: bool,
    slot_layer: u32,
    plans: Vec<super::super::moe::decode::NativeMoe>,
    pointers: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    _bias: CudaSlice<u8>,
}

impl NativeFfn {
    fn new(
        engine: &Engine,
        catalog: &WeightCatalog,
        layer: usize,
        ffn: &BoundFfn,
        rows: usize,
    ) -> Result<Option<Self>> {
        use super::super::moe::decode::{NativeMoe, NativeMoeGeometry, NativeMoeWeights};
        let BoundFfn::Native { router, shared } = ffn else {
            return Ok(None);
        };
        let spec = &engine.model_spec;
        let moe = spec.moe.context("native MoE specification missing")?;
        let glm = spec
            .glm5_next
            .as_ref()
            .context("GLM configuration missing")?;
        let stream = &engine.device.stream;
        let experts = moe.experts as usize;
        let name = format!("model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias");
        let geometry = crate::experts::catalogue::ExpertGeometry::from_model(spec)?;
        let slot_layer = u32::try_from(layer)?
            .checked_sub(geometry.first_layer)
            .context("native expert layer precedes first routed layer")?;
        let entry = catalog
            .manifest
            .entries
            .get(&name)
            .context("native router bias manifest missing")?;
        ensure!(
            entry.dtype == safetensors::Dtype::F32 && entry.shape == [experts],
            "native router bias requires FP32 [experts]"
        );
        let StoredTensor::Bytes(source) = catalog
            .tensors
            .get(&name)
            .context("native router bias not uploaded")?
        else {
            anyhow::bail!("native router bias storage is not FP32 bytes");
        };
        let mut bias = stream.alloc_zeros::<u8>(rows * experts * 4)?;
        for row in 0..rows {
            stream.memcpy_dtod(
                source,
                &mut bias.slice_mut(row * experts * 4..(row + 1) * experts * 4),
            )?;
        }
        let pointers = stream.alloc_zeros::<u64>(experts * 6)?;
        let valid = stream.alloc_zeros::<u32>(experts)?;
        let mut plans = Vec::with_capacity(rows);
        for count in 1..=rows {
            let (bias_ptr, _guard) = bias.device_ptr(stream);
            let bias_view = unsafe {
                Tensor::new(
                    bias_ptr,
                    [count as u32, moe.experts],
                    [u64::from(moe.experts), 1],
                    (count * experts * 4) as u64,
                )?
            };
            plans.push(NativeMoe::new(
                &engine.device.ctx,
                stream,
                NativeMoeGeometry {
                    hidden: spec.hidden,
                    intermediate: moe.expert_intermediate,
                    experts: moe.experts,
                    top_k: moe.experts_per_tok,
                    scaling: glm.routed_scaling_factor,
                    limit: glm.swiglu_limit,
                },
                NativeMoeWeights {
                    router: *router,
                    bias: bias_view,
                    slots: view(&pointers, stream, 0, [moe.experts, 6], [6, 1])?,
                    valid: view(&valid, stream, 0, [moe.experts], [1])?,
                    shared: **shared,
                },
                count as u32,
            )?);
        }
        Ok(Some(Self {
            #[cfg(test)]
            wrong_route: false,
            slot_layer,
            plans,
            pointers,
            valid,
            _bias: bias,
        }))
    }
}
enum BoundMixer {
    Attention {
        image_prefill: PrefillPlans<FullAttentionPlan>,
        text_prefill: PrefillPlans<FullAttentionPlan>,
        decode: Box<FullAttentionPlan>,
        #[cfg(feature = "cuda")]
        compact_small: Option<PrefillPlans<FullAttentionPlan>>,
    },
    Gemma {
        flash: Box<Flash>,
        q_norm: Tensor<bf16, 1>,
        k_norm: Option<Tensor<bf16, 1>>,
        full: bool,
    },
    Gdn {
        norm: Tensor<bf16, 1>,
        prefill: PrefillPlans<GdnPlan>,
        verify: Vec<GdnPlan>,
        replay: Vec<GdnPlan>,
        decode: Box<GdnPlan>,
    },
    Dsa {
        q_norm: Tensor<bf16, 1>,
        kv_norm: Tensor<bf16, 1>,
        k_norm: Tensor<bf16, 1>,
        k_bias: Tensor<bf16, 1>,
        ape: Tensor<bf16, 2>,
        kv_b: Tensor<bf16, 2>,
        prefill: PrefillPlans<DsaPlan>,
        decode: Box<DsaPlan>,
    },
    Flint {
        q_norm: Tensor<bf16, 1>,
        k_norm: Tensor<bf16, 1>,
        sinks: CudaSlice<bf16>,
        rope: CudaFunction,
        nope: CudaFunction,
    },
}

/// Prebound logical extents for MTP verify/replay. All plans share the Engine's
/// existing allocations; projection offsets remain based on workspace_rows.
struct PrefillPlans<T> {
    full: Box<T>,
    small: Vec<T>,
    max_rows: usize,
}
/// Either a prebound plan borrowed as-is or a plan narrowed at dispatch.
enum Narrowed<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}
impl<T> std::ops::Deref for Narrowed<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Narrowed::Borrowed(plan) => plan,
            Narrowed::Owned(plan) => plan,
        }
    }
}
impl<T> PrefillPlans<T> {
    fn bind(
        full: T,
        bounds: (usize, usize),
        narrow: impl Fn(&T, u32) -> Result<T>,
    ) -> Result<Self> {
        let (max_rows, small_rows) = bounds;
        ensure!(
            max_rows > 0 && (1..=crate::linear_plan::MAX_SKINNY_ROWS).contains(&small_rows),
            "prefill plan max_rows={max_rows} small_rows={small_rows} exceeds its binding"
        );
        let small = (1..=small_rows.min(max_rows))
            .map(|rows| narrow(&full, rows as u32))
            .collect::<Result<_>>()?;
        Ok(Self {
            full: Box::new(full),
            small,
            max_rows,
        })
    }
    /// Declared small extents borrow prebound plans; larger sizes narrow at dispatch.
    fn select(
        &self,
        rows: usize,
        narrow: &dyn Fn(&T, u32) -> Result<T>,
    ) -> Result<Narrowed<'_, T>> {
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "prefill rows outside binding"
        );
        if let Some(small) = self.small.get(rows - 1) {
            return Ok(Narrowed::Borrowed(small));
        }
        Ok(Narrowed::Owned(narrow(&self.full, rows as u32)?))
    }
}
fn project_kda(
    linear: &LayerLinearPlan,
    blas: &super::super::blas::Blas,
    s: &Arc<CudaStream>,
    phase: ExecutionPhase,
    w: &mut Work,
    spec: GdnSpec,
    shape: [usize; 3],
) -> Result<()> {
    let [linear_rows, capacity, hidden] = shape;
    ensure!(
        matches!(
            spec.gate_mode,
            crate::op_contract::GdnGateMode::PerChannel { .. }
        ),
        "native KDA projection requires per-channel geometry"
    );
    spec.validate()?;
    let value = (spec.hv * spec.dv) as usize;
    let conv = spec.conv_dim()? as usize;
    let hv = spec.hv as usize;
    let a_width = spec.ab_width()? as usize - hv;
    let mw = w.mixer.as_ref().context("KDA mixer workspace missing")?;
    let input = matrix(&w.n, s, 0, linear_rows, hidden)?;
    for (part, role) in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ]
    .into_iter()
    .enumerate()
    {
        linear.run(
            blas,
            role,
            phase,
            input,
            &mut TensorMut(matrix(&w.gu, s, 0, linear_rows, value)?),
        )?;
        super::super::copy::columns(
            s,
            &w.gu,
            &mut w.qkv,
            super::super::copy::Rect {
                rows: linear_rows,
                width: value,
                src_pitch: value,
                src_col: 0,
                dst_pitch: conv,
                dst_col: part * value,
            },
        )?;
    }
    for (low_role, expand_role, output) in [
        (
            "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight",
            &mw.ab,
        ),
        (
            "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight",
            &mw.gate,
        ),
    ] {
        linear.run(
            blas,
            low_role,
            phase,
            input,
            &mut TensorMut(matrix(&w.act, s, 0, linear_rows, spec.dk as usize)?),
        )?;
        linear.run(
            blas,
            expand_role,
            phase,
            matrix(&w.act, s, 0, linear_rows, spec.dk as usize)?,
            &mut TensorMut(matrix(output, s, 0, linear_rows, value)?),
        )?;
    }
    linear.run(
        blas,
        "self_attn.b_proj.weight",
        phase,
        input,
        &mut TensorMut(matrix(&mw.ab, s, capacity * a_width, linear_rows, hv)?),
    )?;
    Ok(())
}

pub(super) fn matrix(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}
pub(super) fn heads(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    h: u32,
    d: u32,
) -> Result<Tensor<bf16, 3>> {
    view(
        b,
        s,
        offset,
        [u32::try_from(rows)?, h, d],
        [u64::from(h) * u64::from(d), u64::from(d), 1],
    )
}
pub(super) fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> Result<Tensor<bf16, 1>> {
    view(b, s, 0, [u32::try_from(b.len())?], [1])
}
pub(super) fn scratch(b: &CudaSlice<u8>, s: &Arc<CudaStream>) -> Scratch {
    let (ptr, _read) = b.device_ptr(s);
    Scratch {
        ptr,
        bytes: b.len() as u64,
    }
}
pub(super) fn limits(rows: usize, capacity: usize) -> Result<AttentionLimits> {
    ensure!(capacity >= 32, "Q38 attention capacity must be at least 32");
    Ok(AttentionLimits {
        max_rows: u32::try_from(rows)?,
        capacity: u32::try_from(capacity)?,
        key_chunk: 256.min((capacity / 32 * 32) as u32),
        causal: 1,
    })
}
#[cfg(feature = "cuda")]
fn nested_verify_tcmqa(device_prefix: bool, rows: usize, bound: usize) -> Result<bool> {
    if !device_prefix {
        return Ok(false);
    }
    ensure!(
        rows > 0 && rows <= bound,
        "nested device-prefix attention rows={rows} exceeds bound={bound}"
    );
    Ok(true)
}
impl Runtime {
    fn native_workspace(
        spec: &crate::model_spec::ModelSpec,
        manifest: &crate::checkpoint::TensorManifest,
        rows: usize,
    ) -> Result<usize> {
        use crate::experts::catalogue::{ExpertGeometry, ExpertLayout};
        let geometry = ExpertGeometry::from_model(spec)?;
        let layout = ExpertLayout::new(
            spec.expert_format.context("native expert format missing")?,
            &geometry,
        )?;
        let (h, i, e, k, layers) = (
            geometry.hidden as u64,
            geometry.intermediate as u64,
            geometry.experts as u64,
            geometry.top_k as u64,
            geometry.layers as u64,
        );
        let rows = u64::try_from(rows)?;
        let live = (rows * k).min(e);
        let slots = layers * live * layout.slot_bytes();
        let quant = |width: u64| width + width.div_ceil(128) * 4;
        let row_buffers = e * 4
            + k * 16
            + quant(h)
            + k * (i * 6 + quant(i) + h * 4)
            + i * 6
            + h * 2
            + 2 * quant(h)
            + quant(i)
            + 4;
        let plans = layers * (rows * (4 * 1024 * 1024 + 64) + rows * (rows + 1) / 2 * row_buffers);
        let tables = layers * (e * 6 * 8 + e * 4 + rows * e * 4);
        let native_inputs = manifest
            .entries
            .values()
            .filter(|entry| {
                entry.load
                    && entry.is_linear_projection()
                    && !entry.is_routed_expert()
                    && entry.dtype == safetensors::Dtype::F8_E4M3
            })
            .try_fold(0u64, |sum, entry| {
                let &width = entry
                    .shape
                    .get(1)
                    .context("native projection is not a matrix")?;
                sum.checked_add(quant(width as u64))
                    .context("native projection scratch overflow")
            })?
            * rows;
        let mult = u64::from(
            spec.glm5_next
                .as_ref()
                .context("GLM configuration missing")?
                .hc_mult,
        );
        let hc = rows * (4 * mult * h + (4 * mult + mult * mult) * 4);
        let dsa = rows * ((512 * 2 + 1 + 2051) * 4 + 257 * 2);
        usize::try_from(slots + plans + tables + native_inputs + hc + dsa)
            .context("native workspace exceeds address space")
    }
    #[cfg(all(test, feature = "cuda"))]
    pub(super) fn allocated_workspace_bytes(&self) -> u64 {
        self.scratch.len() as u64
    }

    fn select_positions(
        &self,
        stream: &Arc<CudaStream>,
        meta: crate::backend::cuda::contracts::SeqRef,
        workspace_rows: usize,
    ) -> Result<()> {
        let (table, _table) = self.positions.device_ptr(stream);
        let (active, _active) = self.active_positions.device_ptr(stream);
        let max_rows = u32::try_from(workspace_rows)?;
        unsafe {
            stream
                .launch_builder(&self.gather_positions)
                .arg(&table)
                .arg(&active)
                .arg(&meta.ptr)
                .arg(&max_rows)
                .launch(grid((workspace_rows * 3).div_ceil(256).min(65535), 256))?;
        }
        Ok(())
    }
    #[cfg(feature = "cuda")]
    pub(super) fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_dir
    }
    #[cfg(test)]
    pub(super) fn position_values(&self, stream: &Arc<CudaStream>) -> Result<Vec<i32>> {
        let values = stream.clone_dtoh(&self.positions)?;
        Ok(values)
    }
    pub(super) fn vision_bytes(&self) -> usize {
        self.vision_bytes
    }
    fn row_bytes(spec: &crate::model_spec::ModelSpec) -> Result<usize> {
        crate::work_layout::RowWorkSpec::new(spec)?
            .row_bytes()?
            .checked_add(spec.q_heads as usize * size_of::<f32>())
            .and_then(|n| n.checked_add(3 * size_of::<i32>()))
            .and_then(|n| {
                if spec.architecture == crate::model_spec::Architecture::Flint {
                    n.checked_add(2 * 4 * 128 * size_of::<bf16>())
                } else {
                    Some(n)
                }
            })
            .context("Q38 row workspace including FA2 LSE overflow")
    }
    fn fixed_bytes(spec: &crate::model_spec::ModelSpec, capacity: usize) -> Result<usize> {
        capacity
            .checked_mul(3 * size_of::<i32>())
            .and_then(|n| n.checked_add(spec.vocab as usize * 2 + 12 + 64 * size_of::<u64>()))
            .and_then(|n| {
                if spec.architecture == crate::model_spec::Architecture::Flint {
                    n.checked_add(spec.vocab as usize * 4)
                        .and_then(|n| n.checked_add(2 * 4 * 512 * 128 * size_of::<bf16>()))
                } else {
                    Some(n)
                }
            })
            .context("Q38 fixed workspace overflow")
    }
    pub(super) fn workspace(
        model: &crate::model_spec::ModelSpec,
        rows: usize,
        capacity: usize,
        policy: GdnPolicy,
        verify_rows: usize,
        stream: Option<&Arc<CudaStream>>,
    ) -> Result<usize> {
        if model.architecture == crate::model_spec::Architecture::Flint {
            let moe = model.moe.context("Flint MoE geometry missing")?;
            let (hidden, intermediate, experts, top_k) = (
                model.hidden as usize,
                moe.expert_intermediate as usize,
                moe.experts as usize,
                moe.experts_per_tok as usize,
            );
            let decode = crate::backend::cuda::moe::decode::decode_scratch_bytes_mixed(
                rows,
                top_k,
                intermediate,
                moe.shared_intermediate as usize,
            );
            let prefill =
                crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
                    rows,
                    rows,
                    top_k,
                    experts,
                    hidden,
                    (intermediate, moe.shared_intermediate as usize),
                    (None, model.routed_weight_boundary()),
                )?;
            return Ok(usize::try_from(decode.max(prefill).div_ceil(256) * 256)?);
        }
        if model.architecture == crate::model_spec::Architecture::Gemma4 {
            let stream = stream.context("Gemma workspace admission requires a device stream")?;
            let layers = model.layers.len();
            let per_layer_input = model.per_layer_input.context("PLE width missing")? as usize;
            let per_layer_inputs = rows
                .checked_mul(layers)
                .and_then(|n| n.checked_mul(per_layer_input))
                .and_then(|n| n.checked_mul(2))
                .context("PLE admission bytes overflow")?;
            let rope_tables = capacity
                .checked_mul(256 + 512)
                .and_then(|n| n.checked_mul(2))
                .context("RoPE admission bytes overflow")?;
            let flash = model.layers.iter().try_fold(0usize, |total, layer| {
                let geometry = layer.geometry.context("Gemma attention geometry missing")?;
                Flash::workspace(
                    stream,
                    rows,
                    model.q_heads as usize,
                    geometry.head_dim as usize,
                    capacity,
                )?
                .checked_add(total)
                .context("Flash admission bytes overflow")
            })?;
            let kv_width = (model.kv_heads * model.head_dim) as usize;
            let staging = (512 + rows)
                .checked_mul(kv_width)
                .and_then(|n| n.checked_mul(4))
                .context("KV staging admission bytes overflow")?;
            return per_layer_inputs
                .checked_add(rope_tables)
                .and_then(|n| n.checked_add(flash))
                .and_then(|n| n.checked_add(staging))
                .context("Gemma admission bytes overflow");
        }
        let attention = if model.architecture == crate::model_spec::Architecture::Glm5Next {
            None
        } else {
            Some(FullAttentionSpec::from_model(model)?)
        };
        let gdn = GdnSpec::from_model(model)?;
        let (attention_prefill, attention_decode, attention_nested_verify) =
            if let Some(attention) = attention.as_ref() {
                (
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::workspace(attention, limits(rows, capacity)?)?
                    } else {
                        FullAttentionOps::preprocess_workspace(attention, limits(rows, capacity)?)?
                    },
                    FullAttentionOps::decode_workspace(attention, 1, u32::try_from(capacity)?)?,
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::decode_workspace(
                            attention,
                            u32::try_from(rows.min(verify_rows))?,
                            u32::try_from(capacity)?,
                        )?
                    } else {
                        FullAttentionOps::workspace(
                            attention,
                            limits(rows.min(verify_rows), capacity)?,
                        )?
                    },
                )
            } else {
                (
                    crate::op_contract::WorkspaceReq {
                        bytes: 0,
                        alignment: 256,
                    },
                    crate::op_contract::WorkspaceReq {
                        bytes: 0,
                        alignment: 256,
                    },
                    crate::op_contract::WorkspaceReq {
                        bytes: 0,
                        alignment: 256,
                    },
                )
            };
        let spec = gdn;
        let prefill = GdnOps::workspace(&spec, u32::try_from(rows)?, policy)?;
        let decode = GdnOps::workspace(&spec, 1, GdnPolicy::P2)?;
        ensure!(
            [
                attention_prefill.alignment,
                attention_decode.alignment,
                attention_nested_verify.alignment,
                prefill.alignment,
                decode.alignment,
            ]
            .iter()
            .all(|&a| a == 256),
            "operator scratch alignment differs from Engine arena"
        );
        Ok(usize::try_from(
            attention_prefill
                .bytes
                .max(attention_decode.bytes)
                .max(attention_nested_verify.bytes)
                .max(prefill.bytes)
                .max(decode.bytes),
        )?)
    }
    pub(super) fn new(
        engine: &Engine,
        checkpoint_dir: &Path,
        catalog: WeightCatalog,
        arena: Option<CudaSlice<u8>>,
        scratch_buffer: CudaSlice<u8>,
        kv_scratch: Option<(CudaSlice<bf16>, CudaSlice<bf16>)>,
        verification: (usize, bool),
    ) -> Result<Self> {
        let (verify_rows, invariant) = verification;
        let s = &engine.device.stream;
        let rows = engine.workspace_rows;
        let flint = engine.model_spec.architecture == crate::model_spec::Architecture::Flint;
        let gemma = engine.model_spec.architecture == crate::model_spec::Architecture::Gemma4;
        let attention = if flint
            || gemma
            || engine.model_spec.architecture == crate::model_spec::Architecture::Glm5Next
        {
            None
        } else {
            Some(FullAttentionSpec::from_model(&engine.model_spec)?)
        };
        let decode_limits = attention
            .as_ref()
            .map(|spec| -> Result<AttentionLimits> {
                let mut bound = limits(1, engine.capacity)?;
                bound.key_chunk = FullAttentionOps::decode_key_chunk(
                    &engine.device.ctx,
                    spec,
                    bound.capacity,
                    engine
                        .state
                        .kv
                        .first()
                        .is_some_and(|kv| kv.compact.is_some()),
                )?;
                Ok(bound)
            })
            .transpose()?;
        let gdn = if flint || gemma {
            None
        } else {
            Some(GdnSpec::from_model(&engine.model_spec)?)
        };
        let q_width = engine.model_spec.q_width;
        let kv_width = engine.model_spec.kv_heads * engine.model_spec.head_dim;
        let gdn_policy = if flint {
            GdnPolicy::P2
        } else {
            GdnPolicy::for_device(&engine.device.info)
        };
        let common = CommonOpsRef::new_with_swiglu_rounding(
            &engine.device.ctx,
            engine.model_spec.swiglu_rounding(),
            engine.model_spec.swiglu_limit(),
        )?;
        let positions = s.clone_htod(
            &crate::vision_input::PositionPlan::text(1, engine.capacity)?.packed_axes()?,
        )?;
        let active_positions = s.alloc_zeros(rows * 3)?;
        let mask = s.clone_htod(
            &(0..64)
                .map(|i| {
                    if i == 63 {
                        u64::MAX
                    } else {
                        (1u64 << (i + 1)) - 1
                    }
                })
                .collect::<Vec<_>>(),
        )?;
        let mut layers = Vec::new();
        let w = &engine.work;
        let mw = if gemma {
            None
        } else {
            Some(w.mixer.as_ref().context("Q38 mixer work missing")?)
        };
        let rope_module = if flint || gemma {
            Some(cubin::module(&engine.device.ctx, "rope")?)
        } else {
            None
        };
        let gemma_ops = if gemma {
            let width = engine
                .model_spec
                .per_layer_input
                .context("PLE width missing")? as usize;
            let depth = engine.model_spec.layers.len();
            let mut tables = Vec::new();
            for full in [false, true] {
                let geometry = engine
                    .model_spec
                    .layers
                    .iter()
                    .find_map(|layer| layer.geometry.filter(|g| (g.head_dim == 512) == full))
                    .context("per-layer RoPE geometry missing")?;
                tables
                    .push(s.clone_htod(&geometry.rope.table(geometry.head_dim, engine.capacity)?)?);
            }
            let rope_tables: [CudaSlice<bf16>; 2] = tables
                .try_into()
                .map_err(|_| anyhow::anyhow!("two per-layer RoPE tables required"))?;
            let act = cubin::module(&engine.device.ctx, "act")?;
            let embed = cubin::module(&engine.device.ctx, "embed")?;
            #[cfg(test)]
            let mlp_elements = rows
                * engine
                    .model_spec
                    .layers
                    .iter()
                    .filter_map(|layer| {
                        layer
                            .geometry
                            .map(|geometry| geometry.intermediate as usize)
                    })
                    .max()
                    .context("Gemma MLP diagnostic geometry missing")?;
            Some(GemmaOps {
                scaled_embed: embed.load_function("embed_scale_broadcast")?,
                softcap: act.load_function("softcap")?,
                gated_activation: act.load_function("silu_mul_split")?,
                rope: rope_module
                    .as_ref()
                    .context("table RoPE module missing")?
                    .load_function("qknorm_rope_table_kv")?,
                ring_copy: rope_module
                    .as_ref()
                    .context("table RoPE module missing")?
                    .load_function("dynamic_ring_copy")?,
                branch_add: cubin::module(&engine.device.ctx, "q38_common")?
                    .load_function("rms_branch_add")?,
                multiply: act.load_function("multiply")?,
                per_layer: embed.load_function("per_layer_input")?,
                rope_tables,
                per_layer_table: matrix(
                    catalog.bf16("model.language_model.embed_tokens_per_layer.weight")?,
                    s,
                    0,
                    engine.model_spec.vocab as usize,
                    depth * width,
                )?,
                per_layer_projection: matrix(
                    catalog.bf16("model.language_model.per_layer_model_projection.weight")?,
                    s,
                    0,
                    depth * width,
                    engine.model_spec.hidden as usize,
                )?,
                per_layer_norm: vector(
                    catalog.bf16("model.language_model.per_layer_projection_norm.weight")?,
                    s,
                )?,
                per_layer_inputs: s.alloc_zeros(rows * depth * width)?,
                #[cfg(test)]
                logits: s.alloc_zeros(rows * engine.model_spec.vocab as usize)?,
                #[cfg(test)]
                mlp_ones: s.clone_htod(&vec![bf16::ONE; mlp_elements])?,
                #[cfg(test)]
                mlp_activation: s.alloc_zeros(mlp_elements)?,
            })
        } else {
            None
        };
        for (i, layout) in engine.model_spec.layers.iter().enumerate() {
            let prefix = if flint {
                format!("model.layers.{i}.")
            } else {
                format!("model.language_model.layers.{i}.")
            };
            let weight = |name: &str| vector(catalog.bf16(&format!("{prefix}{name}"))?, s);
            let plan =
                LayerDequantPlan::from_model(&catalog.manifest, i as u32, &engine.model_spec)?;
            let arena_view = arena.as_ref().map(|b| scratch(b, s));
            let mut linear = LayerLinearPlan::bind(&plan, &catalog, arena_view.as_ref(), s)?;
            let ffn = BoundFfn::bind(&engine.model_spec, &mut linear)?;
            if engine.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                linear.prepare_native(engine.blas.fp8_ops()?, s, rows as u32)?;
            }
            let native = NativeFfn::new(engine, &catalog, i, &ffn, rows)?;
            let mixer = match layout.mixer {
                Mixer::FullAttention => {
                    if engine.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                        let spec = crate::op_contract::DsaSpec::from_model(&engine.model_spec)?;
                        let full = DsaPlan::new(
                            &engine.device.ctx,
                            &engine.device.info,
                            spec,
                            u32::try_from(engine.capacity)?,
                            u32::try_from(rows)?,
                        )?;
                        let decode = Box::new(full.for_rows(1)?);
                        let prefill = PrefillPlans::bind(
                            full,
                            (rows, crate::linear_plan::MAX_SKINNY_ROWS.min(rows)),
                            DsaPlan::for_rows,
                        )?;
                        let aux = |name: &str| {
                            vector(catalog.bf16(&format!("{prefix}self_attn.{name}"))?, s)
                        };
                        let kv_b = matrix(
                            catalog.bf16(&format!("{prefix}self_attn.kv_b_proj.weight"))?,
                            s,
                            0,
                            32_768,
                            512,
                        )?;
                        let ape = matrix(
                            catalog.bf16(&format!(
                                "{prefix}self_attn.indexer.index_kpool_compress_ape"
                            ))?,
                            s,
                            0,
                            4,
                            128,
                        )?;
                        BoundMixer::Dsa {
                            q_norm: aux("q_a_layernorm.weight")?,
                            kv_norm: aux("kv_a_layernorm.weight")?,
                            k_norm: aux("indexer.k_norm.weight")?,
                            k_bias: aux("indexer.k_norm.bias")?,
                            ape,
                            kv_b,
                            prefill,
                            decode,
                        }
                    } else if flint {
                        {
                            let module =
                                rope_module.as_ref().context("Flint rope module missing")?;
                            let offset =
                                catalog.bf16(&format!("{prefix}self_attn.softmax_offset"))?;
                            #[cfg(feature = "cuda")]
                            let sinks = offset.clone();
                            #[cfg(feature = "rocm")]
                            let sinks = {
                                let mut sinks = s.alloc_zeros(offset.len())?;
                                s.memcpy_dtod(offset, &mut sinks)?;
                                sinks
                            };
                            BoundMixer::Flint {
                                q_norm: weight("self_attn.q_norm.weight")?,
                                k_norm: weight("self_attn.k_norm.weight")?,
                                sinks,
                                rope: module.load_function("flint_qknorm_rope_kv")?,
                                nope: module.load_function("flint_qknorm_kv_nope")?,
                            }
                        }
                    } else if gemma {
                        let geometry = layout
                            .geometry
                            .context("Gemma 4 attention geometry missing")?;
                        let full = geometry.head_dim == 512;
                        let mut flash = Flash::new_with_score(
                            s.clone(),
                            rows,
                            engine.model_spec.q_heads as usize,
                            engine.model_spec.kv_heads as usize,
                            geometry.head_dim as usize,
                            Score::from_tokens(
                                1.0,
                                layout.sliding_window.map(|window| window as usize),
                            )?,
                            engine.capacity,
                        )?;
                        let position_limit = if layout.sliding_window.is_some() {
                            engine.state.kv_ring(layout.state_index as usize)?
                        } else {
                            engine.capacity - 1
                        };
                        flash.bind_position(&engine.work.position, position_limit)?;
                        BoundMixer::Gemma {
                            flash: Box::new(flash),
                            q_norm: weight("self_attn.q_norm.weight")?,
                            k_norm: if geometry.kv_reader {
                                None
                            } else {
                                Some(weight("self_attn.k_norm.weight")?)
                            },
                            full,
                        }
                    } else {
                        let attention = *attention
                            .as_ref()
                            .context("full-attention layer has no specification")?;
                        let mw = mw.context("Q38 mixer work missing")?;
                        let kv = &engine.state.kv[layout.state_index as usize];
                        ensure!(
                            kv.compact.is_some() == kv_scratch.is_some(),
                            "KV format/prefill scratch mismatch at setup"
                        );
                        let bind = |bound: AttentionLimits, positions, compact: bool| {
                            let active_rows = bound.max_rows as usize;
                            let (k, v, cap) = if compact {
                                (
                                    &kv.k,
                                    &kv.v,
                                    kv.compact
                                        .as_ref()
                                        .context("compact cache missing")?
                                        .layout
                                        .ring,
                                )
                            } else if let Some((k, v)) = &kv_scratch {
                                (k, v, engine.capacity)
                            } else {
                                (&kv.k, &kv.v, engine.capacity)
                            };
                            FullAttentionOps::prepare(
                                &engine.device.ctx,
                                &attention,
                                bound,
                                AttentionBindings {
                                    compact: if compact {
                                        kv.compact.as_ref().map(|c| c.view(&kv.k, &kv.v, s))
                                    } else {
                                        None
                                    },
                                    q_gate: heads(&w.qkv, s, 0, active_rows, attention.hq, 512)?,
                                    k: heads(
                                        &w.qkv,
                                        s,
                                        rows * (2 * q_width as usize),
                                        active_rows,
                                        attention.hkv,
                                        256,
                                    )?,
                                    v: heads(
                                        &w.qkv,
                                        s,
                                        rows * (2 * q_width as usize + kv_width as usize),
                                        active_rows,
                                        attention.hkv,
                                        256,
                                    )?,
                                    q: TensorMut(heads(
                                        &mw.q,
                                        s,
                                        0,
                                        active_rows,
                                        attention.hq,
                                        256,
                                    )?),
                                    gate: Some(TensorMut(heads(
                                        &mw.gate,
                                        s,
                                        0,
                                        active_rows,
                                        attention.hq,
                                        256,
                                    )?)),
                                    norm_mode: crate::backend::cuda::ops::RmsMode::ZeroCentered,
                                    k_cache: TensorMut(view(
                                        k,
                                        s,
                                        0,
                                        [attention.hkv, cap as u32, 256],
                                        [(cap * 256) as u64, 256, 1],
                                    )?),
                                    v_cache: TensorMut(view(
                                        v,
                                        s,
                                        0,
                                        [attention.hkv, cap as u32, 256],
                                        [(cap * 256) as u64, 256, 1],
                                    )?),
                                    core_out: TensorMut(heads(
                                        &w.attn,
                                        s,
                                        0,
                                        active_rows,
                                        attention.hq,
                                        256,
                                    )?),
                                    q_norm: weight("self_attn.q_norm.weight")?,
                                    k_norm: weight("self_attn.k_norm.weight")?,
                                    causal_mask: view(&mask, s, 0, [64], [1])?,
                                    positions,
                                },
                                common.clone(),
                            )
                        };
                        let decode = bind(
                            decode_limits.context("decode attention limits missing")?,
                            PositionBindings::Axes(view(&active_positions, s, 0, [1, 3], [3, 1])?),
                            kv.compact.is_some(),
                        )?;
                        ensure!(
                            decode.workspace().bytes <= scratch_buffer.len() as u64,
                            "bound decode attention exceeds Engine scratch budget"
                        );
                        if layout.state_index == 0 {
                            eprintln!(
                                "{}",
                                serde_json::json!({"q38_decode_attention_plan": {
                                    "segment_rows":decode.segment_rows(), "segments":decode.segments(),
                                    "max_parts":decode.max_parts(), "phase1_smem_bytes":decode.phase1_smem_bytes(),
                                    "phase2_smem_bytes":decode.phase2_smem_bytes(), "workspace_bytes":decode.workspace().bytes,
                                    "positions":"Axes"
                                }})
                            );
                        }
                        let text_prefill = PrefillPlans::bind(
                            bind(
                                limits(rows, engine.capacity)?,
                                PositionBindings::Linear,
                                false,
                            )?,
                            (rows, verify_rows),
                            |plan, rows| {
                                if cfg!(feature = "rocm") && invariant {
                                    bind(
                                        AttentionLimits {
                                            max_rows: rows,
                                            ..decode_limits
                                                .context("verify attention limits missing")?
                                        },
                                        PositionBindings::Axes(view(
                                            &active_positions,
                                            s,
                                            0,
                                            [rows, 3],
                                            [3, 1],
                                        )?),
                                        false,
                                    )?
                                    .with_single_row_segments()
                                } else {
                                    plan.for_rows(rows)
                                }
                            },
                        )?;
                        ensure!(
                            text_prefill
                                .small
                                .iter()
                                .all(|plan| plan.workspace().bytes <= scratch_buffer.len() as u64),
                            "nested verify attention exceeds Engine scratch budget"
                        );
                        BoundMixer::Attention {
                            image_prefill: PrefillPlans::bind(
                                bind(
                                    limits(rows, engine.capacity)?,
                                    PositionBindings::Axes(view(
                                        &active_positions,
                                        s,
                                        0,
                                        [rows as u32, 3],
                                        [3, 1],
                                    )?),
                                    false,
                                )?,
                                (rows, verify_rows),
                                FullAttentionPlan::for_rows,
                            )?,
                            text_prefill,
                            #[cfg(feature = "cuda")]
                            compact_small: if kv.compact.is_some() {
                                Some(PrefillPlans::bind(
                                    bind(
                                        limits(verify_rows.min(rows), engine.capacity)?,
                                        PositionBindings::Linear,
                                        true,
                                    )?,
                                    (verify_rows.min(rows), verify_rows),
                                    FullAttentionPlan::for_rows,
                                )?)
                            } else {
                                None
                            },
                            decode: Box::new(decode),
                        }
                    }
                }
                Mixer::Gdn => {
                    let spec = gdn.context("GDN layer has no GDN specification")?;
                    let native_kda = matches!(
                        spec.gate_mode,
                        crate::op_contract::GdnGateMode::PerChannel { .. }
                    );
                    let mixer_prefix = if native_kda {
                        "self_attn."
                    } else {
                        "linear_attn."
                    };
                    let mw = mw.context("Q38 mixer work missing")?;
                    let cd = spec.conv_dim()?;
                    let a_width = spec.ab_width()? as usize - spec.hv as usize;
                    let state = &engine.state.gdn[layout.state_index as usize];
                    let bind = |active_rows| {
                        GdnOps::prepare(
                            &engine.device.ctx,
                            &spec,
                            active_rows as u32,
                            GdnBindings {
                                qkv: matrix(&w.qkv, s, 0, active_rows, cd as usize)?,
                                a: if matches!(
                                    spec.gate_mode,
                                    crate::op_contract::GdnGateMode::PerChannel { .. }
                                ) {
                                    GdnGateInput::PerChannel(heads(
                                        &mw.ab,
                                        s,
                                        0,
                                        active_rows,
                                        spec.hv,
                                        128,
                                    )?)
                                } else {
                                    GdnGateInput::PerHead(matrix(
                                        &mw.ab,
                                        s,
                                        0,
                                        active_rows,
                                        spec.hv as usize,
                                    )?)
                                },
                                b: matrix(
                                    &mw.ab,
                                    s,
                                    rows * a_width,
                                    active_rows,
                                    spec.hv as usize,
                                )?,
                                conv_weight: GdnConvWeights::bind(
                                    &catalog,
                                    s,
                                    &format!("{prefix}{mixer_prefix}"),
                                    spec,
                                )?,
                                decay: GdnDecayWeights::bind(
                                    &catalog,
                                    s,
                                    &format!("{prefix}{mixer_prefix}"),
                                    spec,
                                )?,
                                core_out: TensorMut(heads(
                                    &w.attn,
                                    s,
                                    0,
                                    active_rows,
                                    spec.hv,
                                    spec.dv,
                                )?),
                                s: TensorMut(view(
                                    &state.s,
                                    s,
                                    0,
                                    [spec.hv, spec.dk, spec.dv],
                                    [
                                        u64::from(spec.dk) * u64::from(spec.dv),
                                        u64::from(spec.dv),
                                        1,
                                    ],
                                )?),
                                history: TensorMut(view(
                                    &state.history,
                                    s,
                                    0,
                                    [cd, spec.history_width],
                                    [u64::from(spec.history_width), 1],
                                )?),
                            },
                            // The planner's derived chunkwise-P2 split for a plan bound
                            // at `active_rows`; steps of at most MIN_VERIFY_ROWS keep 0.
                            gdn_policy.cw_p2_for_step(active_rows as u32),
                        )
                    };
                    BoundMixer::Gdn {
                        norm: weight(if native_kda {
                            "self_attn.o_norm.weight"
                        } else {
                            "linear_attn.norm.weight"
                        })?,
                        prefill: PrefillPlans::bind(
                            bind(rows)?,
                            (rows, verify_rows),
                            |plan, rows| plan.for_rows(rows, None),
                        )?,
                        verify: Vec::new(),
                        replay: Vec::new(),
                        decode: Box::new(bind(1)?),
                    }
                }
            };
            layers.push(Layer {
                linear,
                ffn,
                native,
                input_norm: weight("input_layernorm.weight")?,
                post_norm: weight("post_attention_layernorm.weight")?,
                pre_ffn_norm: gemma
                    .then(|| weight("pre_feedforward_layernorm.weight"))
                    .transpose()?,
                post_ffn_norm: gemma
                    .then(|| weight("post_feedforward_layernorm.weight"))
                    .transpose()?,
                post_ple_norm: gemma
                    .then(|| weight("post_per_layer_input_norm.weight"))
                    .transpose()?,
                layer_scalar: gemma.then(|| weight("layer_scalar")).transpose()?,
                mixer,
            });
        }
        let module = crate::backend::cuda::cubin::module(&engine.device.ctx, "q38_common")?;
        let runtime = Self {
            attention,
            gdn,
            gemma: gemma_ops,
            checkpoint_dir: checkpoint_dir
                .canonicalize()
                .context("resolve Q38 checkpoint directory")?,
            vision: None,
            vision_bytes: 0,
            #[cfg(test)]
            vision_encode_calls: 0,
            #[cfg(test)]
            glm_calls: (engine.model_spec.architecture
                == crate::model_spec::Architecture::Glm5Next)
                .then(|| std::cell::RefCell::new(vec![[0; 10]; engine.model_spec.layers.len()])),
            positions,
            active_positions,
            gather_positions: crate::backend::cuda::cubin::module(
                &engine.device.ctx,
                "q38_vision_io",
            )?
            .load_function("q38_text_positions")?,
            catalog,
            arena,
            scratch: scratch_buffer,
            kv_scratch,
            prompt: None,
            mask,
            layers,
            common,
            advance: module.load_function("q38_advance")?,
            prepared: BTreeSet::new(),
            workspace_rows: u32::try_from(rows)?,
            #[cfg(feature = "cuda")]
            verify_rows,
        };
        #[cfg(feature = "rocm")]
        runtime.bind_linear_calls(engine, invariant)?;
        Ok(runtime)
    }
    #[cfg(feature = "rocm")]
    fn bind_linear_calls(&self, engine: &Engine, invariant: bool) -> Result<()> {
        if engine.model_spec.architecture == crate::model_spec::Architecture::Glm5Next
            || engine.model_spec.linear_format.fp8_layout().is_none()
        {
            return Ok(());
        }
        let ops = engine.blas.fp8_ops()?;
        let stream = &engine.device.stream;
        let work = &engine.work;
        let mixer = work.mixer.as_ref().context("HIP mixer workspace missing")?;
        let hidden = engine.model_spec.hidden as usize;
        let q_width = engine.model_spec.q_width as usize;
        let kv_width = (engine.model_spec.kv_heads * engine.model_spec.head_dim) as usize;
        let extent = engine.workspace_rows;
        let buckets = (1..=self.workspace_rows)
            .map(|rows| bucketed_prefill_rows(rows, self.workspace_rows))
            .collect::<Result<BTreeSet<_>>>()?;
        for rows in buckets {
            let rows = rows as usize;
            let invariant = invariant && rows <= crate::linear_plan::MAX_SKINNY_ROWS;
            for layer in &self.layers {
                let input = matrix(&work.n, stream, 0, rows, hidden)?;
                let (role, width) = match &layer.mixer {
                    BoundMixer::Dsa { .. } => anyhow::bail!("HIP DSA projections are not bound"),
                    BoundMixer::Flint { .. } => {
                        anyhow::bail!("Flint FP8 mixer binding is unsupported")
                    }
                    BoundMixer::Gemma { .. } => {
                        anyhow::bail!("HIP Gemma FP8 projections are not bound")
                    }
                    BoundMixer::Attention { .. } => {
                        for (role, width, offset) in [
                            ("self_attn.q_proj.weight", 2 * q_width, 0),
                            ("self_attn.k_proj.weight", kv_width, extent * 2 * q_width),
                            (
                                "self_attn.v_proj.weight",
                                kv_width,
                                extent * (2 * q_width + kv_width),
                            ),
                        ] {
                            layer.linear.bind_call(
                                ops,
                                role,
                                input,
                                &TensorMut(matrix(&work.qkv, stream, offset, rows, width)?),
                                invariant,
                            )?;
                        }
                        ("self_attn.o_proj.weight", q_width)
                    }
                    BoundMixer::Gdn { .. } => {
                        let spec = self.gdn.as_ref().context("HIP GDN shape missing")?;
                        let value = (spec.hv * spec.dv) as usize;
                        let heads = spec.hv as usize;
                        for (role, buffer, width, offset) in [
                            (
                                "linear_attn.in_proj_qkv.weight",
                                &work.qkv,
                                spec.conv_dim()? as usize,
                                0,
                            ),
                            ("linear_attn.in_proj_z.weight", &mixer.gate, value, 0),
                            ("linear_attn.in_proj_a.weight", &mixer.ab, heads, 0),
                            (
                                "linear_attn.in_proj_b.weight",
                                &mixer.ab,
                                heads,
                                extent * heads,
                            ),
                        ] {
                            layer.linear.bind_call(
                                ops,
                                role,
                                input,
                                &TensorMut(matrix(buffer, stream, offset, rows, width)?),
                                invariant,
                            )?;
                        }
                        ("linear_attn.out_proj.weight", value)
                    }
                };
                layer.linear.bind_call(
                    ops,
                    role,
                    matrix(&work.attn, stream, 0, rows, width)?,
                    &TensorMut(matrix(&work.out, stream, 0, rows, hidden)?),
                    invariant,
                )?;
                if matches!(layer.ffn, BoundFfn::Dense) {
                    let intermediate = engine.model_spec.intermediate as usize;
                    for (role, offset) in [
                        ("mlp.gate_proj.weight", 0),
                        ("mlp.up_proj.weight", extent * intermediate),
                    ] {
                        layer.linear.bind_call(
                            ops,
                            role,
                            input,
                            &TensorMut(matrix(&work.gu, stream, offset, rows, intermediate)?),
                            invariant,
                        )?;
                    }
                    layer.linear.bind_call(
                        ops,
                        "mlp.down_proj.weight",
                        matrix(&work.act, stream, 0, rows, intermediate)?,
                        &TensorMut(matrix(&work.out, stream, 0, rows, hidden)?),
                        invariant,
                    )?;
                }
            }
        }
        Ok(())
    }
    fn prepare(
        &mut self,
        blas: &Blas,
        rows: u32,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<()> {
        self.prepare_dimensions(blas, rows, spec.vocab, spec.hidden)
    }
    fn prepare_dimensions(
        &mut self,
        blas: &Blas,
        rows: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        if self.prepared.contains(&rows) {
            return Ok(());
        }
        for layer in &mut self.layers {
            layer.linear.prepare(blas, rows)?;
            layer.ffn.prepare(blas, rows)?;
        }
        if let Some(ops) = &self.gemma {
            blas.prepare_fixed(
                rows,
                ops.per_layer_projection.shape[0],
                ops.per_layer_projection.shape[1],
            )?;
        }
        blas.prepare_fixed(rows, vocab, hidden)?;
        self.prepared.insert(rows);
        Ok(())
    }
    pub(super) fn prepare_prefill_rows(
        &mut self,
        blas: &Blas,
        rows: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        let bound = bucketed_prefill_rows(rows, self.workspace_rows)?;
        self.prepare_dimensions(blas, bound, vocab, hidden)
    }
    #[cfg(test)]
    pub(super) fn gemma_logits(
        &self,
        stream: &Arc<CudaStream>,
        row: usize,
        vocab: usize,
    ) -> Result<Vec<bf16>> {
        let ops = self.gemma.as_ref().context("Gemma 4 operations missing")?;
        ensure!(
            row < ops.logits.len() / vocab,
            "Gemma 4 logit row outside binding"
        );
        stream
            .clone_dtoh(&ops.logits.slice(row * vocab..(row + 1) * vocab))
            .context("Gemma device readback failed")
    }

    #[cfg(test)]
    pub(super) fn gemma_rope_table(
        &self,
        stream: &Arc<CudaStream>,
        full: bool,
        positions: &[usize],
        capacity: usize,
    ) -> Result<Vec<bf16>> {
        let ops = self.gemma.as_ref().context("Gemma 4 operations missing")?;
        let d = if full { 512 } else { 256 };
        let table = &ops.rope_tables[usize::from(full)];
        let mut out = Vec::with_capacity(positions.len() * d);
        for &position in positions {
            ensure!(position < capacity, "RoPE table position outside capacity");
            out.extend(stream.clone_dtoh(&table.slice(position * d..(position + 1) * d))?);
        }
        Ok(out)
    }

    fn linear_rows(&self, rows: u32, phase: ExecutionPhase) -> Result<u32> {
        if phase == ExecutionPhase::Decode {
            return Ok(rows);
        }
        let bound = bucketed_prefill_rows(rows, self.workspace_rows)?;
        ensure!(
            self.prepared.contains(&bound),
            "Q38 prefill Lt bucket {bound} was not bound before Ready"
        );
        Ok(bound)
    }
    fn ensure_complete(&self) -> Result<()> {
        // These allocations own every address retained by the operator plans.
        ensure!(
            self.catalog.bytes > 0
                && self.arena.as_ref().is_none_or(|a| !a.is_empty())
                && self.mask.len() == 64,
            "Q38 owners missing"
        );
        Ok(())
    }
}
impl Engine {
    #[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
    pub(super) fn inject_moe_test_positions(&mut self, axes: [&[i32]; 3]) -> Result<()> {
        let rows = axes[0].len();
        let first = *axes[0]
            .first()
            .context("layer injection positions are empty")?;
        ensure!(
            first >= 0
                && axes.iter().all(|a| a.len() == rows
                    && a.iter()
                        .all(|&v| v >= 0 && (v as u32) < self.model_spec.max_position)),
            "invalid layer injection position axes"
        );
        if let Some(runtime) = self.qwen.as_mut() {
            let packed = (0..rows)
                .flat_map(|row| axes.map(|a| a[row]))
                .collect::<Vec<_>>();
            self.device
                .stream
                .memcpy_htod(&packed, &mut runtime.active_positions.slice_mut(..rows * 3))?;
            self.state.set_sequence(
                &self.device.stream,
                first as u32,
                rows as u32,
                self.workspace_rows as u32,
            )?;
        } else {
            ensure!(
                axes.iter()
                    .all(|axis| axis.iter().enumerate().all(|(r, &v)| v == first + r as i32)),
                "legacy layer injection requires contiguous equal position axes"
            );
        }
        self.position = first as usize;
        self.device
            .stream
            .memcpy_htod(&[first], &mut self.work.position)?;
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    pub(super) fn bind_gdn_snapshots(
        &mut self,
        snapshots: &[super::super::state::GdnState],
        verify_rows: usize,
    ) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let stream = &self.device.stream;
        let g = *q
            .gdn
            .as_ref()
            .context("GDN snapshots have no GDN specification")?;
        let cd = g.conv_dim()?;
        ensure!(
            (1..=g.snapshot_limit()? as usize).contains(&verify_rows),
            "GDN hv={} verify_rows={verify_rows} exceeds snapshot bound={}",
            g.hv,
            g.snapshot_limit()?
        );
        let mut states = snapshots.iter();
        for layer in &mut q.layers {
            if let BoundMixer::Gdn {
                prefill, verify, ..
            } = &mut layer.mixer
            {
                let state = states.next().context("missing GDN snapshot allocation")?;
                let state_width = (g.hv * g.dk * g.dv) as usize;
                let history_width = (cd * g.history_width) as usize;
                let slots = state.s.len() / state_width;
                ensure!(
                    slots > 0
                        && slots < g.snapshot_limit()? as usize
                        && slots >= verify_rows - 1
                        && state.s.len() == slots * state_width
                        && state.history.len() == slots * history_width,
                    "GDN snapshot storage hv={} rows={verify_rows} slots={slots} state_words={} history_words={}",
                    g.hv,
                    state.s.len(),
                    state.history.len()
                );
                let s = view(
                    &state.s,
                    stream,
                    0,
                    [slots as u32, g.hv, g.dk, g.dv],
                    [
                        u64::from(g.hv * g.dk * g.dv),
                        u64::from(g.dk * g.dv),
                        u64::from(g.dv),
                        1,
                    ],
                )?;
                let h = view(
                    &state.history,
                    stream,
                    0,
                    [slots as u32, cd, g.history_width],
                    [
                        u64::from(cd * g.history_width),
                        u64::from(g.history_width),
                        1,
                    ],
                )?;
                *verify = (1..=verify_rows)
                    .map(|rows| prefill.full.for_rows(rows as u32, Some((s, h))))
                    .collect::<Result<_>>()?;
            }
        }
        ensure!(states.next().is_none(), "extra GDN snapshot allocation");
        Ok(())
    }

    pub(super) fn bind_gdn_verify_inputs(&mut self, verify_rows: usize) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let inputs = &self.spec.as_ref().context("MoE chain missing")?.gdn_inputs;
        let stream = &self.device.stream;
        let g = q.gdn.as_ref().context("GDN verify runtime missing")?;
        let cd = g.conv_dim()? as usize;
        let hv = g.hv as usize;
        let a_width = g.ab_width()? as usize - hv;
        ensure!(
            (2..=g.snapshot_limit()? as usize).contains(&verify_rows),
            "GDN hv={} verify_rows={verify_rows} exceeds snapshot bound={}",
            g.hv,
            g.snapshot_limit()?
        );
        let slots = verify_rows - 1;
        let mut items = inputs.iter();
        for layer in &mut q.layers {
            if let BoundMixer::Gdn {
                prefill,
                verify,
                replay,
                ..
            } = &mut layer.mixer
            {
                let input = items
                    .next()
                    .context("missing GDN verify input allocation")?;
                ensure!(
                    input.qkv.len() == slots * cd && input.ab.len() == slots * (a_width + hv),
                    "GDN verify input storage hv={} rows={verify_rows} slots={slots} qkv={} ab={}",
                    g.hv,
                    input.qkv.len(),
                    input.ab.len()
                );
                let qkv = matrix(&input.qkv, stream, 0, slots, cd)?;
                let a = match g.gate_mode {
                    crate::op_contract::GdnGateMode::PerHead => {
                        GdnGateInput::PerHead(matrix(&input.ab, stream, 0, slots, hv)?)
                    }
                    crate::op_contract::GdnGateMode::PerChannel { .. } => {
                        GdnGateInput::PerChannel(heads(&input.ab, stream, 0, slots, g.hv, g.dv)?)
                    }
                };
                let b = matrix(&input.ab, stream, slots * a_width, slots, hv)?;
                *verify = (1..=verify_rows)
                    .map(|rows| prefill.full.for_verify(rows as u32, None))
                    .collect::<Result<_>>()?;
                *replay = (1..=slots)
                    .map(|rows| prefill.full.for_verify(rows as u32, Some((qkv, a, b))))
                    .collect::<Result<_>>()?;
            }
        }
        ensure!(items.next().is_none(), "extra GDN verify input allocation");
        Ok(())
    }

    pub(super) fn replay_gdn_prefix(&mut self, start: usize, committed: usize) -> Result<()> {
        ensure!(committed > 0, "GDN replay requires committed > 0");
        let s = &self.device.stream;
        self.state.set_sequence(
            s,
            start as u32,
            committed as u32,
            self.workspace_rows as u32,
        )?;
        let meta = self.state.seq_ref(s)?;
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        for layer in &q.layers {
            if let BoundMixer::Gdn { replay, .. } = &layer.mixer {
                let plan = replay
                    .get(committed - 1)
                    .context("GDN replay plan missing")?;
                GdnOps::run_step(
                    plan,
                    s,
                    meta,
                    ExecutionPhase::Prefill,
                    scratch(&q.scratch, s),
                    true,
                    None,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn reset_qwen_positions(&mut self) -> Result<()> {
        if let Some(q) = self.qwen.as_mut() {
            #[cfg(test)]
            {
                q.vision_encode_calls = 0;
            }
            let positions =
                crate::vision_input::PositionPlan::text(1, self.capacity)?.packed_axes()?;
            self.device
                .stream
                .memcpy_htod(&positions, &mut q.positions)?;
        }
        Ok(())
    }

    pub fn prefill_multimodal(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        mut trace: Option<&mut Trace>,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        if self.recurrent.is_some() {
            return self.prefill_recurrent_media(input, trace);
        }
        ensure!(
            !trace.as_ref().is_some_and(|t| t.all_layers) || input.input_ids.len() <= self.chunk,
            "all-layer multimodal trace requires one complete prompt chunk"
        );
        self.prepare_multimodal(input, trace.as_deref_mut())?;
        let result = self.prefill(input.input_ids, trace);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    pub(super) fn prepare_multimodal(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        trace: Option<&mut Trace>,
    ) -> Result<()> {
        ensure!(self.qwen.is_some(), "packed images require Qwen");
        ensure!(
            self.state.kv.iter().all(|kv| kv.compact.is_none()),
            "8-bit KV is qualified for text only"
        );
        ensure!(
            !self.poisoned && self.position == 0 && self.images.is_none(),
            "multimodal prefill requires a healthy, reset sequence"
        );
        let positions = input
            .validate(&self.model_spec, self.capacity)?
            .packed_axes()?;
        let request_bytes = super::vision_runtime::VisionRuntime::request_bytes(
            &self.model_spec,
            input,
            self.capacity,
        )?;
        self.prepare_vision_weights(request_bytes)?;
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let vision = q.vision.as_ref().context("vision runtime missing")?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|n| request_bytes <= n),
            "vision request exceeds memory budget: required={request_bytes} free={free}"
        );
        self.poisoned = true;
        let request = vision.prepare(&self.device, &mut self.blas, input, self.capacity)?;
        let images = vision.encode(&self.device.stream, &self.blas, request, trace)?;
        #[cfg(test)]
        {
            q.vision_encode_calls += 1;
        }
        self.device
            .stream
            .memcpy_htod(&positions, &mut q.positions)?;
        self.device.stream.synchronize()?;
        self.images = Some(images);
        self.poisoned = false;
        Ok(())
    }

    pub fn prepare_resident_vision(&mut self) -> Result<()> {
        if self.model_spec.vision.is_some() {
            self.prepare_vision_weights(0)?;
        }
        Ok(())
    }

    pub(crate) fn check_resident_vision_budget(
        dir: &Path,
        options: &crate::spec::Options,
    ) -> Result<()> {
        let spec = crate::model_spec::ModelSpec::read(dir)?;
        if spec.vision.is_none() {
            return Ok(());
        }
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(dir)?;
        let main_bytes = checkpoint
            .manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?
            .resident_bytes;
        let vision_bytes = checkpoint
            .manifest(&spec, crate::checkpoint::LoadPolicy::VisionOnly)?
            .resident_bytes;
        let kv_bytes = spec
            .layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::FullAttention)
            .count()
            .checked_mul(options.capacity)
            .and_then(|n| n.checked_mul(spec.kv_heads as usize))
            .and_then(|n| n.checked_mul(spec.head_dim as usize))
            .and_then(|n| n.checked_mul(4))
            .context("resident KV byte count overflow")?;
        let mtp_bytes = crate::mtp::attachment_bytes(&spec, options.capacity, 1, true)?;
        let minimum = [main_bytes, vision_bytes, kv_bytes, mtp_bytes]
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(n).context("resident byte count overflow")
            })?;
        let device = Device::new(options.device)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| minimum <= n),
            "co-resident MTP/vision/KV minimum exceeds memory budget: main_weights_bytes={main_bytes} mtp_minimum_bytes={mtp_bytes} vision_bytes={vision_bytes} kv_bytes={kv_bytes} minimum_required_bytes={minimum} free_bytes={free} reserve_bytes={}",
            total / 16
        );
        Ok(())
    }

    fn prepare_vision_weights(&mut self, request_bytes: usize) -> Result<()> {
        let kv_bytes = self
            .state
            .kv
            .iter()
            .map(|v| (v.k.len() + v.v.len()) * size_of::<bf16>())
            .sum::<usize>();
        #[cfg(feature = "cuda")]
        let mtp_bytes = if self.mtp.is_some() {
            crate::mtp::attachment_bytes(&self.model_spec, self.capacity, self.chunk, true)?
        } else {
            0
        };
        #[cfg(feature = "rocm")]
        let mtp_bytes = 0usize;
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        if q.vision.is_none() {
            let manifest = crate::checkpoint::MappedCheckpoint::open(&q.checkpoint_dir)?
                .manifest(&self.model_spec, crate::checkpoint::LoadPolicy::VisionOnly)?;
            let needed = manifest
                .resident_bytes
                .checked_add(request_bytes)
                .context("vision resident/workspace overflow")?;
            let (free, total) = self.device.ctx.mem_get_info()?;
            eprintln!(
                "{}",
                serde_json::json!({"serve_resident_budget": {
                    "main_weights_bytes": self.weights.bytes, "mtp_attachment_bytes": mtp_bytes,
                    "vision_weights_bytes": manifest.resident_bytes, "kv_bytes": kv_bytes,
                    "request_bytes": request_bytes, "free_bytes": free, "reserve_bytes": total / 16,
                    "additional_required_bytes": needed, "chunk": self.chunk,
                }})
            );
            ensure!(
                free.checked_sub(total / 16).is_some_and(|n| needed <= n),
                "co-resident MTP/vision/KV exceeds memory budget: mtp_bytes={mtp_bytes} vision_bytes={} kv_bytes={kv_bytes} request_bytes={request_bytes} additional_required_bytes={needed} free_bytes={free} reserve_bytes={}",
                manifest.resident_bytes,
                total / 16
            );
            self.poisoned = true;
            let mut catalog = WeightCatalog::load_with_policy(
                &q.checkpoint_dir,
                &self.model_spec,
                &self.device,
                crate::checkpoint::LoadPolicy::VisionOnly,
            )?;
            q.vision = Some(super::vision_runtime::VisionRuntime::from_catalog(
                &self.model_spec,
                &mut catalog,
                &self.device,
                q.common.clone(),
            )?);
            ensure!(catalog.tensors.is_empty(), "vision weights left unbound");
            q.vision_bytes = catalog.bytes;
            self.device.finish_upload()?;
            self.poisoned = false;
        }
        Ok(())
    }

    pub(super) fn load_qwen(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        mode: super::SetupMode,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        match model_spec.architecture {
            crate::model_spec::Architecture::Flint => require_flint_geometry(&model_spec)?,
            crate::model_spec::Architecture::Glm5Next => require_glm_geometry(&model_spec)?,
            crate::model_spec::Architecture::Gemma4 => require_gemma_geometry(&model_spec)?,
            _ => require_qwen35_geometry(&model_spec)?,
        }
        let write = mode.writes();
        #[cfg(feature = "cuda")]
        let reserve_mtp = matches!(mode, super::SetupMode::Mtp { .. });
        #[cfg(feature = "rocm")]
        let reserve_mtp = false;
        #[cfg(feature = "rocm")]
        ensure!(
            matches!(
                model_spec.architecture,
                crate::model_spec::Architecture::MiniCpmV47
                    | crate::model_spec::Architecture::Qwen35Text
                    | crate::model_spec::Architecture::Flint
                    | crate::model_spec::Architecture::Gemma4
                    | crate::model_spec::Architecture::Glm5Next
            ) && !settings.prefix_cache
                && settings.kv_format == crate::kv::KvFormat::Bf16,
            "HIP hybrid setup requires MiniCPM-V-4.7, Qwen3.8, Flint, Gemma 4 or GLM-5.3 with BF16 KV and no persistent prefix cache"
        );
        ensure!(
            model_spec.moe.is_none() || !reserve_mtp,
            "MoE MTP attachment is not bound: architecture={:?}",
            model_spec.architecture
        );
        ensure!(
            model_spec.moe.is_none() || !settings.prefix_cache,
            "MoE prefix_cache=true has no delegated-expert content identity"
        );
        let started = Instant::now();
        let flint = model_spec.architecture == crate::model_spec::Architecture::Flint;
        let glm = model_spec.architecture == crate::model_spec::Architecture::Glm5Next;
        let gemma = model_spec.architecture == crate::model_spec::Architecture::Gemma4;
        let attention = if flint || glm || gemma {
            None
        } else {
            Some(FullAttentionSpec::from_model(&model_spec)?)
        };
        let gdn = if flint || gemma {
            None
        } else {
            Some(GdnSpec::from_model(&model_spec)?)
        };
        let kv_width = model_spec.kv_heads * model_spec.head_dim;
        ensure!(
            !flint || settings.kv_format == crate::kv::KvFormat::Bf16,
            "Flint requires BF16 KV, received {:?}",
            settings.kv_format
        );
        let config = model_spec.config();
        let draft_config = settings
            .draft_model
            .as_deref()
            .filter(|_| model_spec.moe.is_some())
            .map(|path| crate::dspark::DraftConfig::read(path, &config))
            .transpose()?;
        ensure!(
            draft_config.is_none() || settings.kv_format == crate::kv::KvFormat::Bf16,
            "MoE DSpark requires BF16 target KV, received {:?}",
            settings.kv_format
        );
        let verify_rows = match mode {
            #[cfg(feature = "cuda")]
            super::SetupMode::Mtp { verify_rows } => verify_rows,
            _ => draft_config
                .as_ref()
                .map(|draft| draft.target_rows(&model_spec))
                .transpose()?
                .unwrap_or(if glm { 1 } else { crate::mtp::MAX_VERIFY_ROWS }),
        };
        let capture_layers = if let Some(draft) = &draft_config {
            Some(draft.target_layer_ids.clone())
        } else {
            settings
                .draft_model
                .as_deref()
                .map(|path| crate::dspark::read_capture_layers(path, &config))
                .transpose()?
        };
        let target_transaction_bytes = draft_config
            .as_ref()
            .map(|draft| draft.target_state_bytes(&model_spec))
            .transpose()?
            .unwrap_or(0);
        let draft_row_bytes = draft_config
            .as_ref()
            .map(|draft| draft.workspace_row_bytes())
            .transpose()?
            .unwrap_or(0);
        let target_head_bytes = if draft_config.is_some() {
            (model_spec.vocab as usize)
                .checked_mul(size_of::<bf16>())
                .and_then(|n| n.checked_add(size_of::<i32>() + size_of::<u32>()))
                .and_then(|n| n.checked_mul(verify_rows))
                .context("DSpark all-row head byte count overflow")?
        } else {
            0
        };
        ensure!(
            !settings.spec_budget.is_tree(),
            "Qwen DSpark proposal/tree execution is not bound"
        );
        let capture_row_bytes = capture_layers
            .as_ref()
            .map(|ids| {
                ids.len()
                    .checked_mul(config.hidden_size)
                    .and_then(|n| n.checked_mul(size_of::<bf16>()))
                    .context("capture row byte count overflow")
            })
            .transpose()?
            .unwrap_or(0);
        ensure!(
            capacity >= 32 && capacity <= model_spec.max_position as usize,
            "invalid Q38 capacity"
        );
        // Reject an incomplete operator plan before allocating model weights.
        let runtime = settings
            .runtime_dir
            .as_deref()
            .context("Q38 setup requires runtime directory")?
            .to_owned();
        std::fs::create_dir_all(&runtime)?;
        let device = Device::new(ordinal)?;
        #[cfg(feature = "cuda")]
        Runtime::workspace(
            &model_spec,
            1,
            capacity,
            GdnPolicy::P2,
            verify_rows,
            Some(&device.stream),
        )?;
        let closure = Rc::new(super::super::closure::Closure::new(
            write,
            format!(
                "run calibrate with model={dir:?} draft={:?} capacity={capacity} runtime={runtime:?}",
                settings.draft_model
            ),
        ));
        let gdn_policy = GdnPolicy::for_device(&device.info);
        #[cfg(feature = "rocm")]
        Runtime::workspace(
            &model_spec,
            1,
            capacity,
            gdn_policy,
            verify_rows,
            Some(&device.stream),
        )?;
        let prefix_cache = settings.prefix_cache;
        #[cfg(feature = "cuda")]
        let mut catalog = if prefix_cache {
            WeightCatalog::load_for_prefix_cache(dir, &model_spec, &device)?
        } else {
            WeightCatalog::load(dir, &model_spec, &device)?
        };
        #[cfg(feature = "rocm")]
        let mut catalog = WeightCatalog::load(dir, &model_spec, &device)?;
        let glm_hc_weights = if glm {
            take_glm_hc_weights(&mut catalog, &model_spec)?
        } else {
            Vec::new()
        };
        let prefix_identity_load_ms = catalog.load_wall_ms;
        #[cfg(feature = "cuda")]
        let mut prefix_identity = if prefix_cache {
            let upload_sha = catalog
                .upload_content_sha256
                .clone()
                .context("prefix-cache upload hash missing")?;
            let upload_bytes = catalog
                .upload_hashed_bytes
                .context("prefix-cache upload byte count missing")?;
            ensure!(
                upload_bytes == catalog.bytes,
                "prefix-cache upload hash byte count differs from uploaded weights"
            );
            Some(PrefixIdentity::from_prefix_cache(
                dir,
                &text_rope_context_sha256(capacity)?,
                &model_spec,
                &upload_sha,
                upload_bytes,
            )?)
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        if let Some(identity) = prefix_identity.as_mut() {
            identity.attention_kv_dtype = settings.kv_format.tag().into();
        }
        let weights = catalog.take_globals()?;
        let s = &device.stream;
        let state = super::super::state::StateStore::new_with_format(
            &model_spec,
            capacity,
            settings.kv_format,
            s,
        )?;
        let mut kv_scratch = if settings.kv_format == crate::kv::KvFormat::Bf16 {
            None
        } else {
            Some((
                s.alloc_zeros::<bf16>(capacity * kv_width as usize)?,
                s.alloc_zeros::<bf16>(capacity * kv_width as usize)?,
            ))
        };
        let capture_checkpoint = if glm {
            None
        } else {
            Some(super::super::state::StateCheckpoint::new(
                &state,
                &model_spec,
                1,
                s,
            )?)
        };
        #[cfg(feature = "cuda")]
        let arena_bytes = (0..model_spec.layers.len())
            .map(|i| {
                LayerDequantPlan::from_model(&catalog.manifest, i as u32, &model_spec)
                    .map(|p| p.bytes)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing Q38 layers")?;
        #[cfg(feature = "rocm")]
        let arena_bytes = 0u64;
        let arena = if arena_bytes == 0 {
            None
        } else {
            Some(s.alloc_zeros::<u8>(usize::try_from(arena_bytes)?)?)
        };
        let ops = Ops::new(&device.ctx)?;
        #[cfg(feature = "cuda")]
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        #[cfg(feature = "rocm")]
        let mut blas = Blas::new(device.stream.clone())?;
        if model_spec.linear_format.fp8_layout().is_some() {
            blas.enable_fp8(&device.ctx)?;
        }
        let mut draft = if let Some(draft_config) = &draft_config {
            let path = settings
                .draft_model
                .as_deref()
                .context("draft config has no checkpoint path")?;
            let mut draft = super::super::draft::Draft::new(
                path,
                &device,
                &config,
                capacity,
                draft_config.verify_rows()?,
                &mut blas,
                super::super::markov::ScoreRounding::F32,
            )?;
            draft.setup_attention(&device, &ops, &config, &runtime, &closure)?;
            Some(draft)
        } else {
            None
        };
        device.finish_upload()?;
        let (free, total) = device.ctx.mem_get_info()?;
        let mtp_budget = reserve_mtp
            .then(|| crate::mtp::AttachmentBudget::new(&model_spec, capacity))
            .transpose()?;
        let request = crate::prefill::WorkspaceRequest {
            capacity,
            requested: requested_chunk,
            free_after_resident: free,
            total,
            alignment: CHUNK_ALIGNMENT,
            row_bytes: Runtime::row_bytes(&model_spec)?
                .checked_add(mtp_budget.as_ref().map_or(0, |b| b.row_bytes))
                .and_then(|bytes| bytes.checked_add(capture_row_bytes))
                .and_then(|bytes| bytes.checked_add(draft_row_bytes))
                .context("combined row bytes overflow")?,
            fixed_transient_bytes: Runtime::fixed_bytes(&model_spec, capacity)?
                .checked_add(mtp_budget.as_ref().map_or(0, |b| b.fixed_bytes))
                .and_then(|bytes| bytes.checked_add(target_transaction_bytes))
                .and_then(|bytes| bytes.checked_add(target_head_bytes))
                .context("combined fixed bytes overflow")?,
        };
        let expert_setup = if model_spec.moe.is_some() && !glm {
            Some(super::moe::Setup::new(
                &device,
                &model_spec,
                &catalog.manifest,
                dir,
                &runtime,
                write,
                (
                    request,
                    if draft.is_some() { verify_rows } else { 1 },
                    verify_rows,
                    capture_row_bytes
                        .checked_add(draft_row_bytes)
                        .context("draft context row bytes overflow")?,
                    draft.as_ref().map(|d| &d.config),
                ),
            )?)
        } else {
            None
        };
        let native_budget = if glm {
            Some(crate::prefill::WorkspaceBudget::new(
                crate::prefill::WorkspaceRequest {
                    capacity: capacity.min(crate::linear_plan::MAX_SKINNY_ROWS),
                    alignment: 1,
                    ..request
                },
                |rows| {
                    Runtime::workspace(
                        &model_spec,
                        rows,
                        capacity,
                        gdn_policy,
                        verify_rows,
                        Some(s),
                    )?
                    .checked_add(Runtime::native_workspace(
                        &model_spec,
                        &catalog.manifest,
                        rows,
                    )?)
                    .context("native GLM workspace overflow")
                },
                |rows| Ok(1usize << (usize::BITS - 1 - rows.leading_zeros())),
            )?)
        } else {
            None
        };
        let dense_budget;
        let mut dense_prompt_bytes = 0;
        let budget = if let Some(native) = &native_budget {
            native
        } else if let Some(setup) = &expert_setup {
            &setup.workspace.mixer
        } else {
            let workspace = |rows| {
                let mut target = Runtime::workspace(
                    &model_spec,
                    rows,
                    capacity,
                    gdn_policy,
                    verify_rows,
                    Some(s),
                )?;
                if !flint && settings.kv_format != crate::kv::KvFormat::Bf16 {
                    target = target.max(
                        crate::op_contract::AttentionWorkspace::new(
                            attention
                                .as_ref()
                                .context("Q38 attention specification missing")?,
                            verify_rows as u32,
                            capacity as u32,
                            limits(verify_rows, capacity)?.key_chunk,
                        )?
                        .bytes as usize,
                    );
                }
                target
                    .checked_add(
                        mtp_budget
                            .as_ref()
                            .map(|b| b.workspace(rows))
                            .transpose()?
                            .unwrap_or(0),
                    )
                    .context("Q38 combined workspace overflow")
            };
            let initial = crate::prefill::WorkspaceBudget::new(request, workspace, |admitted| {
                Ok(default_chunk(admitted))
            })?;
            dense_prompt_bytes = crate::prefill::dense_prompt_bytes(
                &model_spec,
                capacity,
                initial.plan.chunk,
                settings.kv_format,
            )?;
            dense_budget = if dense_prompt_bytes == 0 {
                initial
            } else {
                crate::prefill::WorkspaceBudget::new(
                    crate::prefill::WorkspaceRequest {
                        fixed_transient_bytes: request.fixed_transient_bytes
                            .checked_add(dense_prompt_bytes)
                            .context("dense prefill fixed bytes overflow")?,
                        ..request
                    },
                    workspace,
                    |admitted| {
                        ensure!(initial.plan.chunk <= admitted,
                            "dense prefill prompt buffers cannot preserve chunk {} within {admitted} rows",
                            initial.plan.chunk);
                        Ok(initial.plan.chunk)
                    },
                ).context("dense prefill prompt-buffer admission")?
            };
            &dense_budget
        };
        let chunk = budget.plan.chunk;
        let rows = expert_setup
            .as_ref()
            .map_or(chunk, |setup| setup.workspace.workspace_rows);
        let context_rows = if draft.is_some() {
            expert_setup
                .as_ref()
                .context("draft context requires expert span setup")?
                .workspace
                .candidate
                .span_rows as usize
        } else {
            rows
        };
        ensure!(
            !(reserve_mtp || draft.is_some()) || rows >= verify_rows,
            "speculative chunk {rows} cannot fit {verify_rows} verification rows"
        );
        if let Some(draft) = &mut draft {
            draft.grow(context_rows)?;
        }
        let target_workspace = Runtime::workspace(
            &model_spec,
            rows,
            capacity,
            gdn_policy,
            verify_rows,
            Some(s),
        )?
        .max(
            if flint || settings.kv_format == crate::kv::KvFormat::Bf16 {
                0
            } else {
                crate::op_contract::AttentionWorkspace::new(
                    attention
                        .as_ref()
                        .context("Q38 attention specification missing")?,
                    verify_rows as u32,
                    capacity as u32,
                    limits(verify_rows, capacity)?.key_chunk,
                )?
                .bytes as usize
            },
        );
        let (attention_prefill_bytes, attention_decode_bytes, attention_verify_bytes) =
            if flint || glm || gemma {
                (0, 0, 0)
            } else {
                let attention = attention
                    .as_ref()
                    .context("Q38 attention specification missing")?;
                (
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::workspace(attention, limits(rows, capacity)?)?.bytes
                    } else {
                        FullAttentionOps::preprocess_workspace(attention, limits(rows, capacity)?)?
                            .bytes
                    },
                    FullAttentionOps::decode_workspace(attention, 1, u32::try_from(capacity)?)?
                        .bytes,
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::decode_workspace(
                            attention,
                            u32::try_from(rows.min(verify_rows))?,
                            u32::try_from(capacity)?,
                        )?
                        .bytes
                    } else {
                        FullAttentionOps::workspace(
                            attention,
                            limits(rows.min(verify_rows), capacity)?,
                        )?
                        .bytes
                    },
                )
            };
        let (gdn_prefill_bytes, gdn_decode_bytes) = if flint || gemma {
            (0, 0)
        } else {
            let gdn = gdn.as_ref().context("Q38 GDN specification missing")?;
            (
                GdnOps::workspace(gdn, u32::try_from(rows)?, gdn_policy)?.bytes,
                GdnOps::workspace(gdn, 1, GdnPolicy::P2)?.bytes,
            )
        };
        let (prefill_backend, decode_backend) = if gemma {
            (
                "native Rust D256-window/D512",
                "native Rust D256-window/D512",
            )
        } else if flint {
            ("native d128 sliding/full", "flash decode d128")
        } else if glm {
            ("native DSA + KDA", "native DSA + KDA")
        } else if cfg!(feature = "rocm") {
            ("native HIP d256", "TCMQA d256")
        } else {
            ("native Rust d256", "TCMQA d256")
        };
        if flint || gemma {
            let elements = (512 + rows)
                .checked_mul(kv_width as usize)
                .context("sliding KV staging extent overflow")?;
            kv_scratch = Some((
                s.alloc_zeros::<bf16>(elements)?,
                s.alloc_zeros::<bf16>(elements)?,
            ));
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "q38_memory_plan": budget,
                "mtp_attachment_planned_bytes": if reserve_mtp { crate::mtp::attachment_bytes(&model_spec, capacity, rows, true)? } else { 0 },
                "resident_weights_bytes": weights.bytes,
                "prefix_cache": {
                    "enabled": prefix_cache,
                    "load_wall_ms": prefix_identity_load_ms,
                    "hashed_bytes": catalog.upload_hashed_bytes.unwrap_or(0)
                },
                "persistent_state_bytes": state.bytes(),
                "kv_format": settings.kv_format.tag(),
                "kv_prefill_scratch_bytes": kv_scratch.as_ref().map_or(0,|(k,v)|(k.len()+v.len())*size_of::<bf16>()),
                "capture_checkpoint_bytes": capture_checkpoint.as_ref().map_or(0, super::super::state::StateCheckpoint::bytes),
                "target_hidden_capture_bytes": context_rows.checked_mul(capture_row_bytes).context("capture byte count overflow")?,
                "draft_context_span_rows": context_rows,
                "draft_workspace_row_bytes": draft_row_bytes,
                "target_transaction_planned_bytes": target_transaction_bytes,
                "target_all_row_head_bytes": target_head_bytes,
                "layer_arena_bytes": arena_bytes,
                "dense_prefill_prompt_bytes": dense_prompt_bytes,
                "attention_prefill_preprocess_bytes": attention_prefill_bytes,
                "attention_prefill_lse_bytes": rows * model_spec.q_heads as usize * size_of::<f32>(),
                "attention_decode_workspace_bytes": attention_decode_bytes,
                "attention_nested_verify_workspace_bytes": attention_verify_bytes,
                "attention_prefill_backend": prefill_backend,
                "attention_decode_backend": decode_backend,
                "gdn_prefill_workspace_bytes": gdn_prefill_bytes,
                "gdn_decode_workspace_bytes": gdn_decode_bytes,
                "bound_prefill_max_rows": rows,
                "bound_decode_max_rows": 1,
                "gdn_plan": gdn_policy.identity(),
                "gdn_workspace_policy": "max_complete_prebound_plan_not_sum",
                "selection_source": "fixed"
            })
        );
        let scratch_buffer = s.alloc_zeros::<u8>(target_workspace)?;
        let peer_tail_elems = expert_setup
            .as_ref()
            .map(super::moe::Setup::peer_tail_elems)
            .transpose()?
            .unwrap_or(0);
        let work = Work::new_with_peer_tail(s, &model_spec, rows, peer_tail_elems)?;
        let glm_dsa = if glm {
            Some(GlmDsaOutputs::new(rows, s)?)
        } else {
            None
        };
        let glm_hc = if glm {
            let mut runtime = HcRuntime::new(
                &device.ctx,
                s,
                crate::backend::cuda::hc::glm_binding(&model_spec)?,
                model_spec.layers.len(),
                rows,
            )?;
            for (layer, weights) in glm_hc_weights.into_iter().enumerate() {
                runtime.bind_layer(layer, weights.attn, weights.ffn)?;
            }
            Some(runtime)
        } else {
            None
        };
        let flash = if glm {
            None
        } else {
            Some(Flash::new(
                s.clone(),
                rows,
                model_spec.q_heads as usize,
                model_spec.kv_heads as usize,
                model_spec.head_dim as usize,
            )?)
        };
        let logits_f32 = if flint {
            Some(s.alloc_zeros(model_spec.vocab as usize)?)
        } else {
            None
        };
        let hidden_capture = capture_layers
            .as_ref()
            .map(|ids| HiddenCapture::new(ids, &config, context_rows, s))
            .transpose()?;
        let mut engine = Self {
            #[cfg(test)]
            capture_fail_after_layer: None,
            #[cfg(test)]
            capture_fail_stage: None,
            #[cfg(test)]
            flint_window_capture: None,
            qwen_device_prefix: false,
            forward_rows: (1..=rows.min(16)).collect(),
            graphs: if expert_setup.is_some() || glm {
                Vec::new()
            } else {
                vec![Graph::Decode]
            },
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
            hidden_capture,
            weights,
            state,
            capture_checkpoint,
            work,
            ops,
            blas,
            flash,
            decode: None,
            logits_f32,
            device,
            config,
            model_spec,
            capacity,
            chunk,
            workspace_rows: rows,
            trace_layer: 0,
            position: 0,
            #[cfg(feature = "cuda")]
            prefix_identity,
            #[cfg(feature = "cuda")]
            prefix_identity_load_ms,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
            qwen: None,
            deepseek_mixer: None,
            deepseek_ffn: None,
            deepseek_engram: None,
            recurrent: None,
            hc: glm_hc,
            glm_dsa,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            #[cfg(test)]
            moe_test_ffn: None,
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            #[cfg(feature = "cuda")]
            mtp: None,
            #[cfg(feature = "cuda")]
            mtp_state: None,
            #[cfg(feature = "cuda")]
            mtp_epoch: 0,
            images: None,
        };
        engine.qwen = Some(Runtime::new(
            &engine,
            dir,
            catalog,
            arena,
            scratch_buffer,
            kv_scratch,
            (verify_rows, draft_config.is_some()),
        )?);
        #[cfg(feature = "cuda")]
        if flint {
            if write {
                engine.ensure_decode()?;
            } else {
                engine.bind_recorded_decode()?;
            }
        }
        if dense_prompt_bytes > 0 {
            engine.qwen.as_mut().context("Qwen runtime missing")?.prompt = Some(
                engine
                    .device
                    .stream
                    .alloc_zeros(dense_prompt_bytes / size_of::<bf16>())?,
            );
        }
        if let Some(draft) = draft {
            engine.spec = Some(Spec::new(draft, &engine)?);
            engine.bind_gdn_verify_inputs(verify_rows)?;
        }
        if let Some(setup) = expert_setup {
            setup.bind(&mut engine)?;
        }
        #[cfg(feature = "cuda")]
        if engine.spec.is_some() {
            engine.bind_invariant_arithmetic()?;
        }
        if glm {
            let manifest = &engine
                .qwen
                .as_ref()
                .context("GLM runtime missing")?
                .catalog
                .manifest;
            let (slots, admitted_rows) = super::moe::Slots::load_staging(
                &engine.device,
                &engine.model_spec,
                manifest,
                dir,
                rows as u32,
            )?;
            engine.moe_slots = Some(slots);
            engine.chunk = engine.chunk.min(admitted_rows as usize);
            #[cfg(feature = "cuda")]
            {
                engine.workspace_rows = admitted_rows as usize;
            }
            engine
                .forward_rows
                .retain(|row| *row <= admitted_rows as usize);
            engine.device.finish_upload()?;
        }
        if engine.model_spec.moe.is_some() && !glm {
            engine.declare_graphs()?;
        }
        #[cfg(feature = "rocm")]
        engine.initialize_hip_hybrid(dir, settings.draft_model.as_deref(), context_rows)?;
        #[cfg(feature = "rocm")]
        if engine.spec.is_some() {
            engine.bind_invariant_arithmetic()?;
        }
        engine.prepare(1)?;
        engine.prepare_prefill_plans()?;
        engine
            .qwen
            .as_ref()
            .context("Q38 runtime missing")?
            .ensure_complete()?;
        #[cfg(feature = "cuda")]
        if let Some(spec) = &engine.spec {
            let profile = super::super::closure::Profile::draft(
                &spec.draft.config,
                capacity,
                context_rows,
                dir,
                settings
                    .draft_model
                    .as_deref()
                    .context("draft calibration path missing")?,
            )?;
            let key = engine.device.key.clone();
            let mut paths: Vec<_> = profile
                .linear
                .iter()
                .map(|&shape| engine.blas.choice_path(shape))
                .collect();
            paths.push(profile.path(&engine.runtime, &key)?);
            engine.closure.preflight(&paths)?;
            spec.draft
                .prepare_linears(&engine.weights.head, &mut engine.blas, &profile, write)?;
            engine.closure.finish(profile, &engine.runtime, key)?;
        } else {
            if !(flint && write) {
                engine.closure.freeze_fixed()?;
            }
        }
        if !write {
            engine.reset()?;
            engine.prepare_graphs()?;
            engine.closure.expect_captures(engine.graphs.len())?;
            engine.check_graphs()?;
            engine.reset()?;
            engine.closure.ready(started)?;
        }
        Ok(engine)
    }
    #[cfg(all(test, feature = "rocm"))]
    pub(super) fn prepare_hip_fixture(&self) -> Result<()> {
        let q = self
            .qwen
            .as_ref()
            .context("fixture hybrid runtime missing")?;
        for rows in 1..=self.workspace_rows as u32 {
            for layer in &q.layers {
                for shape in layer.linear.library_shapes(rows) {
                    self.blas
                        .prepare_fixed(rows, shape.output as u32, shape.input as u32)?;
                }
                if let BoundFfn::Moe { shared, .. } = &layer.ffn {
                    for weight in shared {
                        self.blas
                            .prepare_fixed(rows, weight.shape[0], weight.shape[1])?;
                    }
                }
            }
            self.blas
                .prepare_fixed(rows, self.model_spec.vocab, self.model_spec.hidden)?;
        }
        Ok(())
    }
    #[cfg(feature = "rocm")]
    fn initialize_hip_hybrid(
        &mut self,
        model: &Path,
        draft_path: Option<&Path>,
        context_rows: usize,
    ) -> Result<()> {
        use super::super::closure::{Profile, hash, write_cache};
        use crate::backend::{gpu::blas::Choice, setup::LinearShape};
        let runtime = self.qwen.as_ref().context("HIP hybrid runtime missing")?;
        let buckets = (1..=runtime.workspace_rows)
            .map(|rows| bucketed_prefill_rows(rows, runtime.workspace_rows))
            .collect::<Result<BTreeSet<_>>>()?;
        let mut linear = BTreeSet::from([LinearShape {
            rows: 1,
            output: self.config.vocab_size,
            input: self.config.hidden_size,
        }]);
        if runtime.gemma.is_some() {
            linear.extend((1..=runtime.workspace_rows).map(|rows| LinearShape {
                rows: rows as usize,
                output: self.config.vocab_size,
                input: self.config.hidden_size,
            }));
        }
        for rows in buckets {
            if let Some(ops) = &runtime.gemma {
                linear.insert(LinearShape {
                    rows: rows as usize,
                    output: ops.per_layer_projection.shape[0] as usize,
                    input: ops.per_layer_projection.shape[1] as usize,
                });
            }
            for layer in &runtime.layers {
                linear.extend(layer.linear.library_shapes(rows));
                if let BoundFfn::Moe { shared, .. } = &layer.ffn {
                    linear.extend(shared.iter().map(|weight| LinearShape {
                        rows: rows as usize,
                        output: weight.shape[0] as usize,
                        input: weight.shape[1] as usize,
                    }));
                }
            }
        }
        let draft = self
            .spec
            .as_ref()
            .map(|spec| -> Result<_> {
                Ok((
                    &spec.draft.config,
                    draft_path.context("HIP hybrid draft path missing")?,
                    context_rows,
                ))
            })
            .transpose()?;
        let profile =
            Profile::hip_hybrid(linear, self.capacity, self.workspace_rows, model, draft)?;
        let identity = self.blas.identity().clone();
        let tag = hash(&serde_json::to_vec(&(&identity, &profile))?);
        let path = self.runtime.join(format!("hip-hybrid-{tag}.json"));
        let manifest = self.runtime.join(format!("hip-closure-{tag}.json"));
        self.closure.preflight(&[path.clone(), manifest.clone()])?;
        let cached = self.closure.access(&path)?;
        let choices: Vec<Choice> = if cached {
            let choices: Vec<Choice> = serde_json::from_slice(&std::fs::read(&path)?)?;
            ensure!(
                choices
                    .iter()
                    .map(|choice| choice.shape)
                    .collect::<BTreeSet<_>>()
                    == profile.linear
                    && choices.len() == profile.linear.len(),
                "HIP hybrid linear closure is incomplete or duplicated"
            );
            for choice in &choices {
                ensure!(
                    !self.blas.wmma_candidate(choice.shape) || choice.setup_median_ms.is_some(),
                    "HIP hybrid closure lacks WMMA/library setup selection"
                );
                self.blas.prepare_choice(choice.shape, None, Some(choice))?;
            }
            choices
        } else {
            let mut choices = Vec::with_capacity(profile.linear.len());
            for &shape in &profile.linear {
                if !self.blas.wmma_candidate(shape) {
                    choices.push(self.blas.prepare_choice(shape, None, None)?);
                    continue;
                }
                self.closure.measurement()?;
                let values: Vec<_> = (0..shape.rows * shape.input)
                    .map(|i| bf16::from_f32(((i * 17 + 31) % 257) as f32 / 256.0 - 0.5))
                    .collect();
                let input = self.device.stream.clone_htod(&values)?;
                let weight = if shape.output == self.config.vocab_size
                    && shape.input == self.config.hidden_size
                {
                    Some(&self.weights.head)
                } else {
                    runtime
                        .catalog
                        .manifest
                        .entries
                        .iter()
                        .find_map(|(name, entry)| {
                            if entry.shape != [shape.output, shape.input]
                                || !name.ends_with(".weight")
                            {
                                return None;
                            }
                            match runtime.catalog.tensors.get(name) {
                                Some(super::super::weights::StoredTensor::Bf16(weight)) => {
                                    Some(weight)
                                }
                                _ => None,
                            }
                        })
                };
                let choice = if let Some(weight) = weight {
                    self.blas.calibrate_shape(shape, weight, &input)?
                } else {
                    let draft = &self
                        .spec
                        .as_ref()
                        .context("HIP hybrid calibration shape lacks target/draft weight")?
                        .draft;
                    self.blas
                        .calibrate_shape(shape, &draft.calibration_weight(shape)?, &input)?
                };
                choices.push(choice);
            }
            write_cache(&path, &choices)?;
            choices
        };
        let decode = if self.model_spec.architecture == crate::model_spec::Architecture::Flint {
            let plan = super::super::verification::hip_plan(
                &self.device.ctx,
                self.capacity,
                1,
                false,
                self.config.num_attention_heads,
            )?;
            self.decode = Some(Attention::new(
                &self.device.ctx,
                &self.device.stream,
                &self.config,
                self.capacity,
                plan,
            )?);
            Some(plan)
        } else {
            None
        };
        self.closure.finish_with_key(profile, manifest, identity)?;
        eprintln!(
            "{}",
            serde_json::json!({"hip_hybrid_setup":{
                "cached":cached,
                "measurements":if cached {0} else {choices.iter().filter(|c|c.setup_median_ms.is_some()).count()}, "decode":decode, "calibration":"untuned correctness-only"
            }})
        );
        Ok(())
    }
    pub(super) fn prepare_qwen(&mut self, rows: usize) -> Result<()> {
        self.qwen.as_mut().context("Q38 runtime missing")?.prepare(
            &self.blas,
            u32::try_from(rows)?,
            &self.model_spec,
        )
    }
    pub(super) fn prepare_prefill_rows(&mut self, rows: usize) -> Result<()> {
        if self.qwen.is_some() {
            let vocab = self.model_spec.vocab;
            let hidden = self.model_spec.hidden;
            let q = self.qwen.as_mut().context("Q38 runtime missing")?;
            return q.prepare_prefill_rows(&self.blas, u32::try_from(rows)?, vocab, hidden);
        }
        self.prepare(rows)
    }
    pub(super) fn qwen_advance(&mut self) -> Result<()> {
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        self.state
            .advance_sequence(&self.device.stream, &q.advance)?;
        self.work.advance(&self.ops, &self.device.stream)?;
        self.state.compact_commit(&self.device.stream, false)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prefill_dense_layers(
        &mut self,
        ids: &[u32],
        mut trace: Option<&mut Trace>,
        keep: usize,
        after_chunk: &mut impl FnMut(&mut Self, &[u32]) -> Result<()>,
    ) -> Result<u32> {
        let prefix = self.position;
        let trace_offset = ids
            .len()
            .checked_sub(keep)
            .context("dense prefill trace extent")?;
        let mut prompt = self
            .qwen
            .as_mut()
            .context("Qwen runtime missing")?
            .prompt
            .take()
            .context("dense prefill prompt buffers missing")?;
        let result = (|| {
            ensure!(
                self.model_spec.moe.is_none() && self.images.is_none(),
                "layer-major prefill requires dense text"
            );
            let elements = self
                .capacity
                .checked_mul(self.model_spec.hidden as usize)
                .and_then(|n| n.checked_mul(2))
                .context("dense prefill prompt extent overflow")?;
            ensure!(
                prompt.len() == elements,
                "dense prefill prompt allocation differs from its plan"
            );
            self.dense_prefill_embed(ids, prefix, trace_offset, trace.as_deref_mut(), &mut prompt)?;
            for layer in 0..self.model_spec.layers.len() {
                self.dense_prefill_layer(
                    layer,
                    ids,
                    prefix,
                    trace_offset,
                    trace.as_deref_mut(),
                    &mut prompt,
                )?;
            }
            self.dense_prefill_finish(ids, prefix, trace_offset, trace, &prompt, after_chunk)?;
            let stream = &self.device.stream;
            stream.memcpy_htod(&[self.position as i32], &mut self.work.position)?;
            stream.memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
            if self.position < self.capacity {
                self.state.set_sequence(
                    stream,
                    self.position as u32,
                    1,
                    self.workspace_rows as u32,
                )?;
            }
            let token = self.token()?;
            self.pending_token = Some(token);
            Ok(token)
        })();
        self.qwen.as_mut().context("Qwen runtime missing")?.prompt = Some(prompt);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    #[cfg(feature = "cuda")]
    pub(super) fn dense_prefill_chunk(
        &mut self,
        ids: &[u32],
        prefix: usize,
        start: usize,
        expected: usize,
    ) -> Result<usize> {
        let rows = ids
            .len()
            .checked_sub(start)
            .context("dense prefill chunk start")?
            .min(self.chunk);
        let end = crate::prefill::chunk_end(start, expected, rows, ids.len())?;
        let position = prefix
            .checked_add(start)
            .context("dense prefill position overflow")?;
        ensure!(
            position
                .checked_add(rows)
                .is_some_and(|end| end <= self.capacity),
            "dense prefill exceeds capacity"
        );
        let stream = &self.device.stream;
        self.state.set_sequence(
            stream,
            position as u32,
            rows as u32,
            self.workspace_rows as u32,
        )?;
        stream.memcpy_htod(&[position as i32], &mut self.work.position)?;
        stream.memcpy_htod(&ids[start..end], &mut self.work.ids.slice_mut(..rows))?;
        self.position = position;
        self.resident_ids_len = rows;
        self.last_input_kind = Some(PrefixLastInputKind::PrefillChunk);
        self.prepare_step(crate::backend::Step {
            rows,
            input: crate::backend::Input::Ids,
            head: Head::None,
            mask: crate::backend::Mask::Causal,
        })?;
        Ok(end)
    }

    #[cfg(feature = "cuda")]
    fn dense_prefill_embed(
        &mut self,
        ids: &[u32],
        prefix: usize,
        trace_offset: usize,
        mut trace: Option<&mut Trace>,
        prompt: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        let hidden = self.model_spec.hidden as usize;
        let mut next = 0;
        for start in (0..ids.len()).step_by(self.chunk) {
            next = self.dense_prefill_chunk(ids, prefix, start, next)?;
            let rows = next - start;
            let skip = trace_offset.saturating_sub(start).min(rows);
            let name = format!(
                "prefill.{}",
                (start / self.chunk).saturating_sub(trace_offset / self.chunk)
            );
            let mut target = trace
                .as_deref_mut()
                .filter(|_| skip < rows)
                .map(|trace| Target {
                    trace,
                    prefix: &name,
                    skip,
                });
            self.run_operator(
                crate::backend::Operator::Embed,
                0,
                crate::backend::Step {
                    rows,
                    input: crate::backend::Input::Ids,
                    head: Head::None,
                    mask: crate::backend::Mask::Causal,
                },
                &mut target,
            )?;
            stream.memcpy_dtod(
                &self.work.x.slice(..rows * hidden),
                &mut prompt.slice_mut(start * hidden..next * hidden),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn dense_prefill_layer(
        &mut self,
        layer: usize,
        ids: &[u32],
        prefix: usize,
        trace_offset: usize,
        mut trace: Option<&mut Trace>,
        prompt: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        let hidden = self.model_spec.hidden as usize;
        let normalized = self.capacity * hidden;
        self.qwen_layer_begin(
            layer,
            crate::backend::Step {
                rows: self.chunk,
                input: crate::backend::Input::Ids,
                head: Head::None,
                mask: crate::backend::Mask::Causal,
            },
        )?;
        let mut next = 0;
        for start in (0..ids.len()).step_by(self.chunk) {
            next = self.dense_prefill_chunk(ids, prefix, start, next)?;
            let rows = next - start;
            self.qwen
                .as_ref()
                .context("Qwen runtime missing")?
                .select_positions(&stream, self.state.seq_ref(&stream)?, self.workspace_rows)?;
            stream.memcpy_dtod(
                &prompt.slice(start * hidden..next * hidden),
                &mut self.work.x.slice_mut(..rows * hidden),
            )?;
            stream.memcpy_dtod(
                &prompt.slice(normalized + start * hidden..normalized + next * hidden),
                &mut self.work.n.slice_mut(..rows * hidden),
            )?;
            let skip = trace_offset.saturating_sub(start).min(rows);
            let name = format!(
                "prefill.{}",
                (start / self.chunk).saturating_sub(trace_offset / self.chunk)
            );
            let mut target = trace
                .as_deref_mut()
                .filter(|_| skip < rows)
                .map(|trace| Target {
                    trace,
                    prefix: &name,
                    skip,
                });
            self.run_layer_program(
                layer,
                crate::backend::Step {
                    rows,
                    input: crate::backend::Input::Ids,
                    head: Head::None,
                    mask: crate::backend::Mask::Causal,
                },
                &mut target,
            )?;
            stream.memcpy_dtod(
                &self.work.x.slice(..rows * hidden),
                &mut prompt.slice_mut(start * hidden..next * hidden),
            )?;
            stream.memcpy_dtod(
                &self.work.n.slice(..rows * hidden),
                &mut prompt.slice_mut(normalized + start * hidden..normalized + next * hidden),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn dense_prefill_finish(
        &mut self,
        ids: &[u32],
        prefix: usize,
        trace_offset: usize,
        mut trace: Option<&mut Trace>,
        prompt: &CudaSlice<bf16>,
        after_chunk: &mut impl FnMut(&mut Self, &[u32]) -> Result<()>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        let hidden = self.model_spec.hidden as usize;
        let normalized = self.capacity * hidden;
        let mut next = 0;
        for start in (0..ids.len()).step_by(self.chunk) {
            next = self.dense_prefill_chunk(ids, prefix, start, next)?;
            let rows = next - start;
            stream.memcpy_dtod(
                &prompt.slice(start * hidden..next * hidden),
                &mut self.work.x.slice_mut(..rows * hidden),
            )?;
            stream.memcpy_dtod(
                &prompt.slice(normalized + start * hidden..normalized + next * hidden),
                &mut self.work.n.slice_mut(..rows * hidden),
            )?;
            let skip = trace_offset.saturating_sub(start).min(rows);
            let name = format!(
                "prefill.{}",
                (start / self.chunk).saturating_sub(trace_offset / self.chunk)
            );
            let mut target = trace
                .as_deref_mut()
                .filter(|_| skip < rows)
                .map(|trace| Target {
                    trace,
                    prefix: &name,
                    skip,
                });
            self.run_operator(
                crate::backend::Operator::Head,
                0,
                crate::backend::Step {
                    rows,
                    input: crate::backend::Input::Ids,
                    head: if next == ids.len() {
                        Head::Last
                    } else {
                        Head::None
                    },
                    mask: crate::backend::Mask::Causal,
                },
                &mut target,
            )?;
            if next == ids.len() {
                self.logits_cache.mark_valid();
            } else {
                self.logits_cache.invalidate();
            }
            after_chunk(self, &ids[start..next])?;
            self.committed_prefix_ids
                .extend_from_slice(&ids[start..next]);
            self.position = prefix + next;
        }
        Ok(())
    }

    pub(super) fn qwen_layer_begin(
        &mut self,
        i: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        #[cfg(feature = "rocm")]
        ensure!(
            i < self.model_spec.layers.len() && step.rows > 0,
            "HIP layer begin has invalid layer/rows"
        );
        #[cfg(feature = "cuda")]
        if step.input == crate::backend::Input::Ids
            && !self.blas.direct_fp8_rows().contains(&step.rows)
        {
            self.qwen.as_mut().context("Q38 runtime missing")?.layers[i]
                .linear
                .materialize(&self.blas, &self.device.stream)?;
        }
        Ok(())
    }
    pub(super) fn glm_dsa_mixer(&mut self, i: usize, step: crate::backend::Step<'_>) -> Result<()> {
        use crate::op_contract::ExecutionPhase;
        let rows = step.rows;
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let layout = DsaRowLayout::new(&self.model_spec, self.workspace_rows)?;
        let hidden = self.model_spec.hidden as usize;
        let s = &self.device.stream;
        let q = self.qwen.as_mut().context("GLM runtime missing")?;
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        let layer = &q.layers[i];
        let w = &mut self.work;
        let input = matrix(&w.n, s, 0, linear_rows, hidden)?;
        layer.linear.run(
            &self.blas,
            "self_attn.q_a_proj.weight",
            phase,
            input,
            &mut TensorMut(matrix(
                &w.qkv,
                s,
                layout.query_residual.start,
                linear_rows,
                1536,
            )?),
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.kv_a_proj_with_mqa.weight",
            phase,
            input,
            &mut TensorMut(matrix(&w.qkv, s, layout.latent.start, linear_rows, 512)?),
        )?;
        let domain = RowDomain {
            meta: self.state.seq_ref(s)?,
            row_base: 0,
            row_capacity: rows as u32,
        };
        let residual = heads(&w.qkv, s, layout.query_residual.start, rows, 1, 1536)?;
        let mut residual_mut = TensorMut(heads(
            &w.qkv,
            s,
            layout.query_residual.start,
            rows,
            1,
            1536,
        )?);
        q.common.norm.rms_with(
            s,
            domain,
            residual,
            match &layer.mixer {
                BoundMixer::Dsa { q_norm, .. } => *q_norm,
                _ => anyhow::bail!("GLM DSA layer has no q_a_layernorm"),
            },
            &mut residual_mut,
            RmsOptions {
                mode: RmsMode::Standard,
                eps: self.model_spec.rms_eps,
            },
        )?;
        let latent = heads(&w.qkv, s, layout.latent.start, rows, 1, 512)?;
        let mut latent_mut = TensorMut(heads(&w.qkv, s, layout.latent.start, rows, 1, 512)?);
        q.common.norm.rms_with(
            s,
            domain,
            latent,
            match &layer.mixer {
                BoundMixer::Dsa { kv_norm, .. } => *kv_norm,
                _ => anyhow::bail!("GLM DSA layer has no kv_a_layernorm"),
            },
            &mut latent_mut,
            RmsOptions {
                mode: RmsMode::Standard,
                eps: self.model_spec.rms_eps,
            },
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.q_b_proj.weight",
            phase,
            matrix(&w.qkv, s, layout.query_residual.start, linear_rows, 1536)?,
            &mut TensorMut(matrix(&w.qkv, s, layout.query.start, linear_rows, 16_384)?),
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.indexer.wq_b.weight",
            phase,
            matrix(&w.qkv, s, layout.query_residual.start, linear_rows, 1536)?,
            &mut TensorMut(matrix(
                &w.qkv,
                s,
                layout.index_query.start,
                linear_rows,
                4096,
            )?),
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.indexer.wk.weight",
            phase,
            input,
            &mut TensorMut(matrix(&w.qkv, s, layout.packed.start, linear_rows, 128)?),
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.indexer.index_kpool_compress_gate",
            phase,
            input,
            &mut TensorMut(matrix(
                &w.qkv,
                s,
                layout.packed.start + self.workspace_rows * 128,
                linear_rows,
                128,
            )?),
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.indexer.weights_proj.weight",
            phase,
            input,
            &mut TensorMut(matrix(
                &w.qkv,
                s,
                layout.index_weights.start,
                linear_rows,
                32,
            )?),
        )?;
        let packed = heads(&w.qkv, s, layout.packed.start, rows, 1, 128)?;
        let mut packed_mut = TensorMut(heads(&w.qkv, s, layout.packed.start, rows, 1, 128)?);
        q.common.norm.layer_norm(
            s,
            packed,
            match &layer.mixer {
                BoundMixer::Dsa { k_norm, .. } => *k_norm,
                _ => anyhow::bail!("GLM DSA layer has no indexer k_norm weight"),
            },
            match &layer.mixer {
                BoundMixer::Dsa { k_bias, .. } => *k_bias,
                _ => anyhow::bail!("GLM DSA layer has no indexer k_norm bias"),
            },
            &mut packed_mut,
            1e-6,
        )?;
        let narrowed = if step.input == crate::backend::Input::Token {
            None
        } else {
            Some(match &layer.mixer {
                BoundMixer::Dsa { prefill, .. } => prefill.select(rows, &DsaPlan::for_rows)?,
                _ => anyhow::bail!("GLM DSA layer has no prefill plan"),
            })
        };
        let (plan, state_index, kv_b, ape) = match (&narrowed, &layer.mixer) {
            (
                None,
                BoundMixer::Dsa {
                    decode, kv_b, ape, ..
                },
            ) => (
                decode.as_ref(),
                self.model_spec.layers[i].state_index as usize,
                *kv_b,
                *ape,
            ),
            (Some(narrowed), BoundMixer::Dsa { kv_b, ape, .. }) => (
                &**narrowed,
                self.model_spec.layers[i].state_index as usize,
                *kv_b,
                *ape,
            ),
            _ => anyhow::bail!("GLM DSA layer has no bound DSA plan"),
        };
        let state = self.state.dsa_state_mut(state_index)?;
        let outputs = self
            .glm_dsa
            .as_mut()
            .context("GLM DSA output buffers are not bound")?;
        outputs.pack_index(
            s,
            &w.qkv,
            [
                layout.packed.start,
                layout.packed.start + self.workspace_rows * 128,
            ],
            rows,
        )?;
        let append = DsaAppendInput {
            latent: matrix(&w.qkv, s, layout.latent.start, rows, 512)?,
            packed: matrix(&outputs.packed, s, 0, rows, 257)?,
        };
        plan.append(state, &append, s)?;
        let index = DsaIndexInput {
            q: heads(&w.qkv, s, layout.index_query.start, rows, 32, 128)?,
            weights: matrix(&w.qkv, s, layout.index_weights.start, rows, 32)?,
            ape,
        };
        plan.pool(state, &index, s)?;
        plan.index(
            state,
            &index,
            &DsaIndexOutput {
                selected: TensorMut(view(&outputs.selected, s, 0, [rows as u32, 512], [512, 1])?),
                scores: TensorMut(view(&outputs.scores, s, 0, [rows as u32, 512], [512, 1])?),
                margins: TensorMut(view(&outputs.margins, s, 0, [rows as u32], [1])?),
                indices: TensorMut(view(
                    &outputs.indices,
                    s,
                    0,
                    [rows as u32, 2051],
                    [2051, 1],
                )?),
            },
            s,
        )?;
        let indices_view = view(&outputs.indices, s, 0, [rows as u32, 2051], [2051, 1])?;
        let mut attention_output = TensorMut(heads(&w.attn, s, 0, rows, 64, 256)?);
        plan.attention(
            state,
            &DsaAttentionInput {
                q: heads(&w.qkv, s, layout.query.start, rows, 64, 256)?,
                kv_b: Some(kv_b),
                indices: indices_view,
                sinks: None,
                out: &mut attention_output,
            },
            s,
        )?;
        layer.linear.run(
            &self.blas,
            "self_attn.o_proj.weight",
            phase,
            matrix(&w.attn, s, 0, linear_rows, 16_384)?,
            &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
        )
    }

    pub(super) fn qwen_mixer(
        &mut self,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let layer = &q.layers[i];
        let flint_layer = matches!(layer.mixer, BoundMixer::Flint { .. });
        let gemma_layer = matches!(layer.mixer, BoundMixer::Gemma { .. });
        let glm = self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next;
        let attention = if flint_layer || gemma_layer || glm {
            None
        } else {
            Some(q.attention.context("Q38 attention specification missing")?)
        };
        let gdn = if flint_layer || gemma_layer {
            None
        } else {
            Some(q.gdn.context("Q38 GDN specification missing")?)
        };
        let q_width = self.model_spec.q_width;
        let kv_width = self.model_spec.kv_heads * self.model_spec.head_dim;
        let conv_width = gdn.map_or(0, |g| 2 * g.hk * g.dk + g.hv * g.dv);
        let value_width = gdn.map_or(0, |g| g.hv * g.dv);
        let hidden = self.model_spec.hidden as usize;
        let s = &self.device.stream;
        let rows = step.rows;
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next
            && self.model_spec.layers[i].mixer == Mixer::FullAttention
        {
            return self.glm_dsa_mixer(i, step);
        }
        let layer = &mut q.layers[i];
        let w = &mut self.work;
        let mw = if gemma_layer {
            None
        } else {
            Some(w.mixer.as_ref().context("Q38 mixer work missing")?)
        };
        let meta = self.state.seq_ref(s)?;
        let domain = RowDomain {
            meta,
            row_base: 0,
            row_capacity: rows as u32,
        };
        match &mut layer.mixer {
            BoundMixer::Dsa { .. } => {
                anyhow::bail!("GLM DSA mixer escaped its production dispatch")
            }
            BoundMixer::Flint {
                q_norm,
                k_norm,
                sinks,
                rope,
                nope,
            } => {
                for (role, width, offset) in [
                    ("self_attn.q_proj.weight", q_width as usize, 0),
                    (
                        "self_attn.k_proj.weight",
                        kv_width as usize,
                        linear_rows * q_width as usize,
                    ),
                    (
                        "self_attn.v_proj.weight",
                        kv_width as usize,
                        linear_rows * (q_width as usize + kv_width as usize),
                    ),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, hidden)?,
                        &mut TensorMut(matrix(&w.qkv, s, offset, linear_rows, width)?),
                    )?;
                }
                let state_index = self.model_spec.layers[i].state_index as usize;
                let ring = self.state.kv_ring(state_index)?;
                let kv = &mut self.state.kv[state_index];
                let sliding = self.model_spec.layers[i].sliding_window.is_some();
                let (function, use_rope) = if self.model_spec.layers[i].rope {
                    (rope, true)
                } else {
                    (nope, false)
                };
                let q_source_stride = u64::from(q_width);
                let kv_source_stride = u64::from(kv_width);
                let k_source_offset = u64::from(linear_rows as u32) * u64::from(q_width);
                let v_source_offset = u64::from(linear_rows as u32) * u64::from(q_width + kv_width);
                ensure!(
                    !use_rope || self.model_spec.layers[i].rope,
                    "Flint RoPE function selected for NoPE layer {i}"
                );
                let overflow_sliding =
                    sliding && phase == ExecutionPhase::Prefill && self.position + rows > ring;
                if !overflow_sliding {
                    unsafe {
                        s.launch_builder(function)
                            .arg(&mut w.qkv)
                            .arg(&mut kv.k)
                            .arg(&mut kv.v)
                            .arg(&q_norm.ptr)
                            .arg(&k_norm.ptr)
                            .arg(&w.position)
                            .arg(&-1i32)
                            .arg(&(rows as i32))
                            .arg(&(self.model_spec.q_heads as i32))
                            .arg(&(self.model_spec.kv_heads as i32))
                            .arg(&(self.model_spec.head_dim as i32))
                            .arg(&(ring as i32))
                            .arg(&(if sliding { ring as i32 } else { 0 }))
                            .arg(&(self.model_spec.rope_theta))
                            .arg(&(self.model_spec.rms_eps))
                            .arg(&q_source_stride)
                            .arg(&k_source_offset)
                            .arg(&kv_source_stride)
                            .arg(&v_source_offset)
                            .arg(&kv_source_stride)
                            .launch(LaunchConfig {
                                grid_dim: (
                                    (rows
                                        * (self.model_spec.q_heads + 2 * self.model_spec.kv_heads)
                                            as usize) as u32,
                                    1,
                                    1,
                                ),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                }
                match phase {
                    ExecutionPhase::Prefill => {
                        let shape = Shape {
                            rows,
                            start: if overflow_sliding {
                                self.position.min(512)
                            } else {
                                self.position
                            },
                            q_heads: self.model_spec.q_heads as usize,
                            kv_heads: self.model_spec.kv_heads as usize,
                            capacity: if overflow_sliding {
                                512 + self.workspace_rows
                            } else {
                                ring
                            },
                            head_dim: self.model_spec.head_dim as usize,
                            causal: true,
                            window_left: sliding.then_some(
                                self.model_spec.layers[i].sliding_window.unwrap_or(513) as usize
                                    - 1,
                            ),
                        };
                        if overflow_sliding {
                            let history = self.position.min(512);
                            let capacity = 512 + self.workspace_rows;
                            let (stage_k, stage_v) = q
                                .kv_scratch
                                .as_mut()
                                .context("Flint KV staging is not bound")?;
                            ensure!(
                                stage_k.len() == capacity * kv_width as usize
                                    && stage_v.len() == stage_k.len(),
                                "Flint KV staging extent differs from the setup plan"
                            );
                            if history > 0 {
                                for (source, target) in
                                    [(&kv.k, &mut *stage_k), (&kv.v, &mut *stage_v)]
                                {
                                    super::super::copy::RingCopy {
                                        stream: s,
                                        source,
                                        target,
                                        source_capacity: ring,
                                        target_capacity: capacity,
                                        heads: self.model_spec.kv_heads as usize,
                                        head_dim: self.model_spec.head_dim as usize,
                                    }
                                    .run(
                                        (self.position - history) % ring,
                                        0,
                                        history,
                                        true,
                                        false,
                                    )?;
                                }
                            }
                            unsafe {
                                s.launch_builder(function)
                                    .arg(&mut w.qkv)
                                    .arg(&mut *stage_k)
                                    .arg(&mut *stage_v)
                                    .arg(&q_norm.ptr)
                                    .arg(&k_norm.ptr)
                                    .arg(&w.position)
                                    .arg(&(history as i32))
                                    .arg(&(rows as i32))
                                    .arg(&(self.model_spec.q_heads as i32))
                                    .arg(&(self.model_spec.kv_heads as i32))
                                    .arg(&(self.model_spec.head_dim as i32))
                                    .arg(&(capacity as i32))
                                    .arg(&0i32)
                                    .arg(&(self.model_spec.rope_theta))
                                    .arg(&(self.model_spec.rms_eps))
                                    .arg(&q_source_stride)
                                    .arg(&k_source_offset)
                                    .arg(&kv_source_stride)
                                    .arg(&v_source_offset)
                                    .arg(&kv_source_stride)
                                    .launch(LaunchConfig {
                                        grid_dim: (
                                            (rows
                                                * (self.model_spec.q_heads
                                                    + 2 * self.model_spec.kv_heads)
                                                    as usize)
                                                as u32,
                                            1,
                                            1,
                                        ),
                                        block_dim: (128, 1, 1),
                                        shared_mem_bytes: 0,
                                    })?;
                            }
                            #[cfg(test)]
                            if self.flint_legacy_overflow_prefill {
                                let last = history + rows - 1;
                                for (source, target) in
                                    [(&kv.k, &mut *stage_k), (&kv.v, &mut *stage_v)]
                                {
                                    super::super::copy::RingCopy {
                                        stream: s,
                                        source,
                                        target,
                                        source_capacity: ring,
                                        target_capacity: capacity,
                                        heads: self.model_spec.kv_heads as usize,
                                        head_dim: self.model_spec.head_dim as usize,
                                    }
                                    .run(
                                        last % ring,
                                        0,
                                        1,
                                        false,
                                        false,
                                    )?;
                                }
                            }
                            self.flash
                                .as_mut()
                                .context("Flint prefill attention is not bound")?
                                .run(
                                    &w.qkv,
                                    q_width as usize,
                                    &*stage_k,
                                    &*stage_v,
                                    &mut w.attn,
                                    shape,
                                )?;
                            self.flash
                                .as_mut()
                                .context("Flint prefill attention is not bound")?
                                .apply_sinks(
                                    sinks,
                                    &mut w.attn,
                                    rows,
                                    self.model_spec.q_heads as usize,
                                )?;
                            let end = self.position + rows;
                            let keep = end.min(ring);
                            let source_start = history + rows - keep;
                            let absolute_start = end - keep;
                            for (source, target) in
                                [(&mut *stage_k, &mut kv.k), (&mut *stage_v, &mut kv.v)]
                            {
                                super::super::copy::RingCopy {
                                    stream: s,
                                    source,
                                    target,
                                    source_capacity: capacity,
                                    target_capacity: ring,
                                    heads: self.model_spec.kv_heads as usize,
                                    head_dim: self.model_spec.head_dim as usize,
                                }
                                .run(
                                    source_start,
                                    absolute_start % ring,
                                    keep,
                                    false,
                                    true,
                                )?;
                            }
                        } else {
                            self.flash
                                .as_mut()
                                .context("Flint prefill attention is not bound")?
                                .run(&w.qkv, q_width as usize, &kv.k, &kv.v, &mut w.attn, shape)?;
                            self.flash
                                .as_mut()
                                .context("Flint prefill attention is not bound")?
                                .apply_sinks(
                                    sinks,
                                    &mut w.attn,
                                    rows,
                                    self.model_spec.q_heads as usize,
                                )?;
                        }
                    }
                    ExecutionPhase::Decode => {
                        if !self.flint_frozen_decode_length {
                            self.ops
                                .flint_decode_length(s, &w.position, &mut w.length, ring)?;
                        }
                        let decode = self
                            .decode
                            .as_mut()
                            .context("Flint decode attention is not bound")?;
                        if self.flint_frozen_decode_length {
                            let tokens = (self.position + 1).min(ring);
                            decode.set_length(s, tokens as i32, &mut w.length)?;
                        }
                        decode.run_with(
                            s,
                            GqaDecodeInput {
                                q: &w.qkv,
                                q_offset: 0,
                                k: &kv.k,
                                v: &kv.v,
                                output: &mut w.attn,
                                output_offset: 0,
                                length: &w.length,
                                sinks: Some(sinks),
                                window: sliding.then_some(SlidingWindow {
                                    position: &w.position,
                                    head_stride: (ring * self.model_spec.head_dim as usize) as i32,
                                }),
                            },
                        )?;
                        #[cfg(test)]
                        capture_flint_window(
                            s,
                            &self.device.info.name,
                            FlintWindowInputs {
                                spec: &self.model_spec,
                                layer: i,
                                sliding,
                                ring,
                                position: &w.position,
                                qkv: &w.qkv,
                                attention: &w.attn,
                                kv,
                                sinks,
                            },
                            &self.flint_window_capture,
                        )?;
                    }
                }
                layer.linear.run(
                    &self.blas,
                    "self_attn.o_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, q_width as usize)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
            BoundMixer::Gemma {
                flash,
                q_norm,
                k_norm,
                full,
            } => {
                let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
                let layout = &self.model_spec.layers[i];
                let geometry = layout
                    .geometry
                    .context("Gemma 4 attention geometry missing")?;
                let d = geometry.head_dim as usize;
                let hq = self.model_spec.q_heads as usize;
                let hkv = self.model_spec.kv_heads as usize;
                let q_width = hq * d;
                let kv_width = hkv * d;
                let reader = geometry.kv_reader;
                if reader {
                    layer.linear.run(
                        &self.blas,
                        "self_attn.q_proj.weight",
                        phase,
                        matrix(&w.n, s, 0, linear_rows, hidden)?,
                        &mut TensorMut(matrix(&w.qkv, s, 0, linear_rows, q_width)?),
                    )?;
                } else {
                    for (role, width, offset) in [
                        ("self_attn.q_proj.weight", q_width, 0),
                        ("self_attn.k_proj.weight", kv_width, linear_rows * q_width),
                        (
                            "self_attn.v_proj.weight",
                            kv_width,
                            linear_rows * (q_width + kv_width),
                        ),
                    ] {
                        layer.linear.run(
                            &self.blas,
                            role,
                            phase,
                            matrix(&w.n, s, 0, linear_rows, hidden)?,
                            &mut TensorMut(matrix(&w.qkv, s, offset, linear_rows, width)?),
                        )?;
                    }
                }
                let state_index = layout.state_index as usize;
                let ring = self.state.kv_ring(state_index)?;
                let sliding = layout.sliding_window.is_some();
                let decode = phase == ExecutionPhase::Decode;
                let overflow = sliding && (decode || self.position + rows > ring);
                let history = if overflow { self.position.min(ring) } else { 0 };
                let cache_capacity = if overflow {
                    ring + self.workspace_rows
                } else {
                    ring
                };
                {
                    let kv = &mut self.state.kv[state_index];
                    let (stage_k, stage_v) = if overflow {
                        let (stage_k, stage_v) = q
                            .kv_scratch
                            .as_mut()
                            .context("sliding KV staging is not bound")?;
                        if !reader && (history > 0 || decode) {
                            for (source, target) in [(&kv.k, &mut *stage_k), (&kv.v, &mut *stage_v)]
                            {
                                let copy = super::super::copy::RingCopy {
                                    stream: s,
                                    source,
                                    target,
                                    source_capacity: ring,
                                    target_capacity: cache_capacity,
                                    heads: hkv,
                                    head_dim: d,
                                };
                                if decode {
                                    copy.run_dynamic(&ops.ring_copy, &w.position, rows, false)?;
                                } else {
                                    copy.run(
                                        (self.position - history) % ring,
                                        0,
                                        history,
                                        true,
                                        false,
                                    )?;
                                }
                            }
                        }
                        (stage_k, stage_v)
                    } else {
                        (&mut kv.k, &mut kv.v)
                    };
                    let q_stride = q_width;
                    let prep_heads = if reader { 0 } else { hkv };
                    let cache_start = if overflow && decode {
                        -2
                    } else if overflow {
                        history as i32
                    } else {
                        -1
                    };
                    let ring_tokens = if overflow && decode { ring as i32 } else { 0 };
                    if trace.is_some() && (all_layers(trace) || i == self.trace_layer) {
                        if self.graph.is_none() {
                            s.synchronize()?;
                        }
                        trace_value(
                            trace,
                            s,
                            &debug_name(trace, i, "qkv_projected"),
                            &w.qkv,
                            1,
                            linear_rows
                                * if reader {
                                    q_width
                                } else {
                                    q_width + 2 * kv_width
                                },
                        )?;
                    }
                    unsafe {
                        s.launch_builder(&ops.rope)
                            .arg(&mut w.qkv)
                            .arg(&*stage_k)
                            .arg(&*stage_v)
                            .arg(&q_norm.ptr)
                            .arg(&k_norm.as_ref().unwrap_or(q_norm).ptr)
                            .arg(&ops.rope_tables[usize::from(*full)])
                            .arg(&w.position)
                            .arg(&cache_start)
                            .arg(&(rows as i32))
                            .arg(&(hq as i32))
                            .arg(&(prep_heads as i32))
                            .arg(&(d as i32))
                            .arg(&(cache_capacity as i32))
                            .arg(&ring_tokens)
                            .arg(&self.model_spec.rms_eps)
                            .arg(&(q_stride as u64))
                            .arg(&((linear_rows * q_width) as u64))
                            .arg(&(kv_width as u64))
                            .arg(&((linear_rows * (q_width + kv_width)) as u64))
                            .arg(&(kv_width as u64))
                            .launch(LaunchConfig {
                                grid_dim: ((rows * (hq + 2 * prep_heads)) as u32, 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    if trace.is_some() && (all_layers(trace) || i == self.trace_layer) {
                        if self.graph.is_none() {
                            s.synchronize()?;
                        }
                        trace_value(
                            trace,
                            s,
                            &debug_name(trace, i, "qkv_prepared"),
                            &w.qkv,
                            linear_rows,
                            q_stride,
                        )?;
                        if hkv == 1 && d == 512 {
                            trace_value(
                                trace,
                                s,
                                &debug_name(trace, i, "k_prepared"),
                                stage_k,
                                rows,
                                d,
                            )?;
                            trace_value(
                                trace,
                                s,
                                &debug_name(trace, i, "v_prepared"),
                                stage_v,
                                rows,
                                d,
                            )?;
                        }
                    }
                    flash.run(
                        &w.qkv,
                        q_stride,
                        stage_k,
                        stage_v,
                        &mut w.attn,
                        Shape {
                            rows,
                            start: if overflow && decode {
                                ring
                            } else if overflow {
                                history
                            } else {
                                self.position
                            },
                            q_heads: hq,
                            kv_heads: hkv,
                            capacity: cache_capacity,
                            head_dim: d,
                            causal: true,
                            window_left: layout.sliding_window.map(|window| window as usize - 1),
                        },
                    )?;
                }
                if overflow && !reader {
                    let kv = &mut self.state.kv[state_index];
                    let (stage_k, stage_v) = q
                        .kv_scratch
                        .as_ref()
                        .context("sliding KV staging is not bound")?;
                    let end = history + rows;
                    let keep = end.min(ring);
                    let source_start = end - keep;
                    let absolute_start = self.position + rows - keep;
                    for (source, target) in [(stage_k, &mut kv.k), (stage_v, &mut kv.v)] {
                        let copy = super::super::copy::RingCopy {
                            stream: s,
                            source,
                            target,
                            source_capacity: cache_capacity,
                            target_capacity: ring,
                            heads: hkv,
                            head_dim: d,
                        };
                        if decode {
                            copy.run_dynamic(&ops.ring_copy, &w.position, rows, true)?;
                        } else {
                            copy.run(source_start, absolute_start % ring, keep, false, true)?;
                        }
                    }
                }
                layer.linear.run(
                    &self.blas,
                    "self_attn.o_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, q_width)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
            BoundMixer::Attention {
                image_prefill,
                text_prefill,
                decode,
                #[cfg(feature = "cuda")]
                compact_small,
            } => {
                let mw = mw.context("Q38 mixer work missing")?;
                let attention = *attention
                    .as_ref()
                    .context("full-attention mixer has no specification")?;
                if self.model_spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Channel
                {
                    self.blas.fp8_ops()?.fake_quant(
                        s,
                        &mut TensorMut(matrix(&w.n, s, 0, linear_rows, hidden)?),
                        Some(domain),
                    )?;
                }
                for (role, width, offset) in [
                    ("self_attn.q_proj.weight", 2 * q_width as usize, 0),
                    (
                        "self_attn.k_proj.weight",
                        kv_width as usize,
                        self.workspace_rows * 2 * q_width as usize,
                    ),
                    (
                        "self_attn.v_proj.weight",
                        kv_width as usize,
                        self.workspace_rows * (2 * q_width as usize + kv_width as usize),
                    ),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, hidden)?,
                        &mut TensorMut(matrix(&w.qkv, s, offset, linear_rows, width)?),
                    )?;
                }
                match phase {
                    ExecutionPhase::Prefill => {
                        #[cfg(feature = "rocm")]
                        {
                            ensure!(
                                self.state.kv[self.model_spec.layers[i].state_index as usize]
                                    .compact
                                    .is_none(),
                                "HIP HD256 prefill requires BF16 KV"
                            );
                            let plans = if self.images.is_some() {
                                image_prefill
                            } else {
                                text_prefill
                            };
                            let plan = plans.select(rows, &FullAttentionPlan::for_rows)?;
                            if self.qwen_device_prefix {
                                FullAttentionOps::run(&plan, s, meta, scratch(&q.scratch, s))?;
                            } else {
                                FullAttentionOps::preprocess(
                                    &plan,
                                    s,
                                    meta,
                                    scratch(&q.scratch, s),
                                )?;
                                let kv =
                                    &self.state.kv[self.model_spec.layers[i].state_index as usize];
                                self.flash
                                    .as_mut()
                                    .context("HIP prefill attention is not bound")?
                                    .run(
                                        &mw.q,
                                        q_width as usize,
                                        &kv.k,
                                        &kv.v,
                                        &mut w.attn,
                                        Shape {
                                            rows,
                                            start: self.position,
                                            q_heads: attention.hq as usize,
                                            kv_heads: attention.hkv as usize,
                                            capacity: self.capacity,
                                            head_dim: 256,
                                            causal: true,
                                            window_left: None,
                                        },
                                    )?;
                            }
                        }
                        #[cfg(feature = "cuda")]
                        {
                            if nested_verify_tcmqa(self.qwen_device_prefix, rows, q.verify_rows)? {
                                let kv =
                                    &self.state.kv[self.model_spec.layers[i].state_index as usize];
                                let plans = if kv.compact.is_some() {
                                    compact_small
                                        .as_ref()
                                        .context("compact verify plan missing")?
                                } else if self.model_spec.moe.is_some() {
                                    image_prefill
                                } else {
                                    text_prefill
                                };
                                let narrowed = plans.select(rows, &FullAttentionPlan::for_rows)?;
                                FullAttentionOps::run(&narrowed, s, meta, scratch(&q.scratch, s))?;
                            } else {
                                let kv =
                                    &self.state.kv[self.model_spec.layers[i].state_index as usize];
                                if let Some(c) = &kv.compact {
                                    let (k, v) = q
                                        .kv_scratch
                                        .as_ref()
                                        .context("compact KV prefill scratch missing")?;
                                    ensure!(
                                        self.images.is_none(),
                                        "8-bit KV is qualified for text only"
                                    );
                                    c.dequant_prefix(
                                        c.view(&kv.k, &kv.v, s),
                                        meta,
                                        k,
                                        v,
                                        self.position,
                                        s,
                                    )?;
                                }
                                let plans = if self.images.is_some() {
                                    image_prefill
                                } else {
                                    text_prefill
                                };
                                let narrowed = plans.select(rows, &FullAttentionPlan::for_rows)?;
                                let prefill: &FullAttentionPlan = &narrowed;
                                FullAttentionOps::preprocess(
                                    prefill,
                                    s,
                                    meta,
                                    scratch(&q.scratch, s),
                                )?;
                                let kv =
                                    &self.state.kv[self.model_spec.layers[i].state_index as usize];
                                let (k_cache, v_cache) = if kv.compact.is_some() {
                                    let (k, v) = q
                                        .kv_scratch
                                        .as_ref()
                                        .context("compact KV prefill scratch missing")?;
                                    (k, v)
                                } else {
                                    (&kv.k, &kv.v)
                                };
                                let shape = Shape {
                                    rows,
                                    start: self.position,
                                    q_heads: attention.hq as usize,
                                    kv_heads: attention.hkv as usize,
                                    capacity: self.capacity,
                                    head_dim: 256,
                                    causal: true,
                                    window_left: None,
                                };
                                self.flash
                                    .as_mut()
                                    .context("Q38 prefill attention is not bound")?
                                    .run(
                                        &mw.q,
                                        q_width as usize,
                                        k_cache,
                                        v_cache,
                                        &mut w.attn,
                                        shape,
                                    )?;
                                if let Some(c) = &kv.compact {
                                    let (k, v) = q
                                        .kv_scratch
                                        .as_ref()
                                        .context("compact KV prefill scratch missing")?;
                                    c.finish(
                                        c.view(&kv.k, &kv.v, s),
                                        meta,
                                        Some((k, v, rows)),
                                        true,
                                        s,
                                    )?;
                                }
                            }
                        }
                    }
                    ExecutionPhase::Decode => {
                        FullAttentionOps::run(decode, s, meta, scratch(&q.scratch, s))?;
                    }
                }
                q.common.gate.sigmoid_mul(
                    s,
                    domain,
                    heads(&w.attn, s, 0, rows, attention.hq, 256)?,
                    heads(&mw.gate, s, 0, rows, attention.hq, 256)?,
                    &mut TensorMut(heads(&w.attn, s, 0, rows, attention.hq, 256)?),
                )?;
                if self.model_spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Channel
                {
                    self.blas.fp8_ops()?.fake_quant(
                        s,
                        &mut TensorMut(matrix(&w.attn, s, 0, linear_rows, q_width as usize)?),
                        Some(domain),
                    )?;
                }
                layer.linear.run(
                    &self.blas,
                    "self_attn.o_proj.weight",
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, q_width as usize)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
            BoundMixer::Gdn {
                norm,
                prefill,
                verify,
                decode,
                ..
            } => {
                let mw = mw.context("Q38 mixer work missing")?;
                let spec = gdn.context("GDN mixer has no GDN specification")?;
                let conv = conv_width as usize;
                let value = value_width as usize;
                let hv = spec.hv as usize;
                let a_width = spec.ab_width()? as usize - hv;
                if matches!(
                    spec.gate_mode,
                    crate::op_contract::GdnGateMode::PerChannel { .. }
                ) {
                    project_kda(
                        &layer.linear,
                        &self.blas,
                        s,
                        phase,
                        w,
                        spec,
                        [linear_rows, self.workspace_rows, hidden],
                    )?;
                } else {
                    for (role, output, offset, width) in [
                        ("linear_attn.in_proj_qkv.weight", &w.qkv, 0, conv),
                        ("linear_attn.in_proj_z.weight", &mw.gate, 0, value),
                        ("linear_attn.in_proj_a.weight", &mw.ab, 0, a_width),
                        (
                            "linear_attn.in_proj_b.weight",
                            &mw.ab,
                            self.workspace_rows * a_width,
                            hv,
                        ),
                    ] {
                        layer.linear.run(
                            &self.blas,
                            role,
                            phase,
                            matrix(&w.n, s, 0, linear_rows, hidden)?,
                            &mut TensorMut(matrix(output, s, offset, linear_rows, width)?),
                        )?;
                    }
                }
                let mw = w.mixer.as_ref().context("KDA mixer workspace missing")?;
                if self.qwen_device_prefix
                    && let Some(spec_state) = self.spec.as_mut()
                {
                    let gdn_idx = self.model_spec.layers[i].state_index as usize;
                    let input = spec_state
                        .gdn_inputs
                        .get_mut(gdn_idx)
                        .context("GDN verify input slot missing")?;
                    let slots = input.qkv.len() / conv;
                    let capture_rows = rows.min(slots);
                    if capture_rows > 0 {
                        s.memcpy_dtod(
                            &w.qkv.slice(..capture_rows * conv),
                            &mut input.qkv.slice_mut(..capture_rows * conv),
                        )?;
                        s.memcpy_dtod(
                            &mw.ab.slice(..capture_rows * a_width),
                            &mut input.ab.slice_mut(..capture_rows * a_width),
                        )?;
                        s.memcpy_dtod(
                            &mw.ab.slice(
                                self.workspace_rows * a_width
                                    ..(self.workspace_rows + capture_rows) * a_width,
                            ),
                            &mut input
                                .ab
                                .slice_mut(slots * a_width..(slots + capture_rows) * a_width),
                        )?;
                    }
                }
                let narrowed;
                let plan: &GdnPlan = if self.qwen_device_prefix {
                    ensure!(
                        phase == ExecutionPhase::Prefill,
                        "MTP verify must remain Prefill"
                    );
                    verify
                        .get(rows.checked_sub(1).context("MTP verify requires rows")?)
                        .context("MTP verify GDN plan missing")?
                } else {
                    match phase {
                        ExecutionPhase::Prefill => {
                            narrowed =
                                prefill.select(rows, &|plan, rows| plan.for_rows(rows, None))?;
                            &narrowed
                        }
                        ExecutionPhase::Decode => decode.as_ref(),
                    }
                };
                let gated = GatedRmsInput {
                    mode: if matches!(spec.gate_mode, crate::op_contract::GdnGateMode::PerHead) {
                        super::super::ops::GatedRmsMode::SiluBf16
                    } else {
                        super::super::ops::GatedRmsMode::SigmoidF32
                    },
                    x: heads(&w.attn, s, 0, rows, spec.hv, spec.dv)?,
                    weight: *norm,
                    gate: heads(&mw.gate, s, 0, rows, spec.hv, spec.dv)?,
                };
                if !GdnOps::run_step(
                    plan,
                    s,
                    meta,
                    phase,
                    scratch(&q.scratch, s),
                    self.qwen_device_prefix,
                    Some(&gated),
                )? {
                    q.common.norm.rms_gated(
                        s,
                        domain,
                        gated,
                        &mut TensorMut(heads(&w.attn, s, 0, rows, spec.hv, spec.dv)?),
                        spec.rms_eps,
                    )?;
                }
                layer.linear.run(
                    &self.blas,
                    if matches!(
                        spec.gate_mode,
                        crate::op_contract::GdnGateMode::PerChannel { .. }
                    ) {
                        "self_attn.o_proj.weight"
                    } else {
                        "linear_attn.out_proj.weight"
                    },
                    phase,
                    matrix(&w.attn, s, 0, linear_rows, value)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, hidden)?),
                )?;
            }
        }
        let all = all_layers(trace);
        if all || i == self.trace_layer {
            let name = debug_name(trace, i, "mixer_out");
            let mixer_width = match &layer.mixer {
                BoundMixer::Flint { .. } => q_width,
                BoundMixer::Gemma { full, .. } => {
                    self.model_spec.q_heads * if *full { 512 } else { 256 }
                }
                BoundMixer::Attention { .. } => q_width,
                BoundMixer::Gdn { .. } => value_width,
                BoundMixer::Dsa { .. } => q_width,
            };
            trace_value(trace, s, &name, &w.attn, rows, mixer_width as usize)?;
            let name = debug_name(trace, i, "o_proj");
            trace_value(trace, s, &name, &w.out, rows, hidden)?;
            if matches!(self.model_spec.layers[i].mixer, Mixer::Gdn)
                && (!all || i == 0)
                && let Some(t) = trace.as_mut()
            {
                let gdn = gdn.as_ref().context("GDN trace has no GDN specification")?;
                let state = &self.state.gdn[self.model_spec.layers[i].state_index as usize];
                let prefix = if all {
                    format!("{}.{i}", t.prefix)
                } else {
                    t.prefix.to_owned()
                };
                let suffix = if all { ".e2e" } else { "" };
                t.trace.add(
                    format!("{prefix}.debug.gdn_s{suffix}"),
                    vec![gdn.hv as usize, gdn.dk as usize, gdn.dv as usize],
                    s.clone_dtoh(&state.s)?,
                )?;
                t.trace.add(
                    format!("{prefix}.debug.gdn_history{suffix}"),
                    vec![conv_width as usize, gdn.history_width as usize],
                    s.clone_dtoh(&state.history)?,
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn run_moe_if_bound(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<bool> {
        if self.model_spec.architecture == crate::model_spec::Architecture::DeepseekV41 {
            self.deepseek_ffn_stage(layer, step)?;
            return Ok(true);
        }
        if self
            .qwen
            .as_ref()
            .is_some_and(|runtime| runtime.layers[layer].native.is_some())
        {
            let s = &self.device.stream;
            let runtime = self.qwen.as_mut().context("native MoE runtime missing")?;
            let native = runtime.layers[layer]
                .native
                .as_mut()
                .context("native MoE binding missing")?;
            let plan = native
                .plans
                .get(step.rows.checked_sub(1).context("empty native MoE step")?)
                .context("native MoE rows outside setup closure")?;
            let h = self.model_spec.hidden as usize;
            let x = matrix(&self.work.n, s, 0, step.rows, h)?;
            let y = TensorMut(matrix(&self.work.out, s, 0, step.rows, h)?);
            let (ids, weights) = plan.route(s, x)?;
            let slots = self
                .moe_slots
                .as_mut()
                .context("native expert tier service missing")?;
            let slot_layer = native.slot_layer;
            slots.resolve_native(slot_layer, &ids, &weights)?;
            slots.wait_refill(s)?;
            let (pointers, valid) = slots.native_dispatch(slot_layer)?;
            #[cfg(test)]
            let pointers = {
                let mut pointers = pointers;
                if native.wrong_route {
                    let destination = ids[0] as usize * 6;
                    let source = ids[1] as usize * 6;
                    ensure!(
                        destination != source,
                        "native wrong-route control needs distinct experts"
                    );
                    let replacement: Vec<_> = pointers[source..source + 6].to_vec();
                    pointers[destination..destination + 6].copy_from_slice(&replacement);
                }
                pointers
            };
            s.memcpy_htod(&pointers, &mut native.pointers)?;
            s.memcpy_htod(&valid, &mut native.valid)?;
            plan.run(s, x, &y)?;
            slots.record_all_readers(&self.device)?;
            if all_layers(trace) || layer == self.trace_layer {
                trace_value(
                    trace,
                    s,
                    &format!("{layer}.ffn"),
                    &self.work.out,
                    step.rows,
                    h,
                )?;
            }
            return Ok(true);
        }
        let ffn = self
            .qwen
            .as_ref()
            .map(|runtime| runtime.layers[layer].ffn.clone());
        #[cfg(test)]
        let ffn = ffn.or_else(|| self.moe_test_ffn.clone());
        if let Some(ffn @ BoundFfn::Moe { .. }) = ffn {
            self.run_moe_ffn(layer, step.rows, ffn, trace)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(super) fn moe_ffn_binding(&self, layer: usize) -> Result<BoundFfn> {
        let ffn = self
            .qwen
            .as_ref()
            .context("MoE hybrid runtime missing")?
            .layers
            .get(layer)
            .with_context(|| format!("MoE hybrid layer={layer} is outside runtime"))?
            .ffn
            .clone();
        ensure!(
            matches!(ffn, BoundFfn::Moe { .. }),
            "MoE hybrid layer={layer} has a dense FFN binding"
        );
        Ok(ffn)
    }

    pub(super) fn select_qwen_positions(&self) -> Result<()> {
        let q = self.qwen.as_ref().context("Q38 position runtime missing")?;
        q.select_positions(
            &self.device.stream,
            self.state.seq_ref(&self.device.stream)?,
            self.workspace_rows,
        )
    }

    #[cfg(test)]
    pub(crate) fn gemma_kv_rows(
        &self,
        layer: usize,
        rows: usize,
    ) -> Result<(Vec<bf16>, Vec<bf16>)> {
        let layout = &self.model_spec.layers[layer];
        let d = layout
            .geometry
            .context("Gemma 4 attention geometry missing")?
            .head_dim as usize;
        let hkv = self.model_spec.kv_heads as usize;
        let kv = &self.state.kv[layout.state_index as usize];
        let ring = self.state.kv_ring(layout.state_index as usize)?;
        ensure!(rows <= ring, "KV probe rows exceed the cache ring");
        let s = &self.device.stream;
        let mut k_major = Vec::with_capacity(hkv * rows * d);
        let mut v_major = Vec::with_capacity(hkv * rows * d);
        for head in 0..hkv {
            let base = head * ring * d;
            k_major.extend(s.clone_dtoh(&kv.k.slice(base..base + rows * d))?);
            v_major.extend(s.clone_dtoh(&kv.v.slice(base..base + rows * d))?);
        }
        let mut k = Vec::with_capacity(k_major.len());
        let mut v = Vec::with_capacity(v_major.len());
        for row in 0..rows {
            for head in 0..hkv {
                let base = (head * rows + row) * d;
                k.extend_from_slice(&k_major[base..base + d]);
                v.extend_from_slice(&v_major[base..base + d]);
            }
        }
        Ok((k, v))
    }

    #[cfg(test)]
    pub(crate) fn gemma_linear_probe(&mut self, role: &str, input: &[bf16]) -> Result<Vec<bf16>> {
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        let layer = &q.layers[0];
        let s = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        let intermediate = self.model_spec.layers[0]
            .geometry
            .context("Gemma 4 FFN geometry missing")?
            .intermediate as usize;
        let (input_width, output_width) = match role {
            "self_attn.q_proj.weight" => (hidden, self.model_spec.q_width as usize),
            "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => (
                hidden,
                self.model_spec.kv_heads as usize
                    * self.model_spec.layers[0]
                        .geometry
                        .context("Gemma 4 attention geometry missing")?
                        .head_dim as usize,
            ),
            "mlp.gate_proj.weight" | "mlp.up_proj.weight" => (hidden, intermediate),
            "mlp.down_proj.weight" => (intermediate, hidden),
            _ => anyhow::bail!("unsupported Gemma linear probe role {role}"),
        };
        ensure!(
            input.len().is_multiple_of(input_width),
            "linear probe input extent"
        );
        let rows = input.len() / input_width;
        let x = s.clone_htod(input)?;
        let y = s.alloc_zeros(rows * output_width)?;
        layer.linear.run(
            &self.blas,
            role,
            ExecutionPhase::Prefill,
            matrix(&x, s, 0, rows, input_width)?,
            &mut TensorMut(matrix(&y, s, 0, rows, output_width)?),
        )?;
        s.clone_dtoh(&y).context("Gemma device readback failed")
    }

    #[cfg(test)]
    pub(crate) fn gemma_mlp_activation_probe(
        &self,
        gate: &[bf16],
        up: &[bf16],
    ) -> Result<(Vec<bf16>, Vec<bf16>)> {
        ensure!(
            !gate.is_empty() && gate.len() == up.len(),
            "Gemma MLP activation probe extent"
        );
        let q = self.qwen.as_ref().context("Q38 runtime missing")?;
        let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
        let s = &self.device.stream;
        let gate = s.clone_htod(gate)?;
        let up = s.clone_htod(up)?;
        let ones = s.clone_htod(&vec![bf16::ONE; gate.len()])?;
        let mut activation = s.alloc_zeros::<bf16>(gate.len())?;
        let mut product = s.alloc_zeros::<bf16>(gate.len())?;
        let count = i32::try_from(gate.len())?;
        unsafe {
            s.launch_builder(&ops.gated_activation)
                .arg(&gate)
                .arg(&ones)
                .arg(&mut activation)
                .arg(&count)
                .arg(&2i32)
                .launch(flat(gate.len()))?;
            s.launch_builder(&ops.gated_activation)
                .arg(&gate)
                .arg(&up)
                .arg(&mut product)
                .arg(&count)
                .arg(&2i32)
                .launch(flat(gate.len()))?;
        }
        Ok((s.clone_dtoh(&activation)?, s.clone_dtoh(&product)?))
    }

    pub(super) fn qwen_per_layer_input(
        &mut self,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let s = &self.device.stream;
        let w = &mut self.work;
        let rows = step.rows;
        let h = self.model_spec.hidden as usize;
        let width = self
            .model_spec
            .per_layer_input
            .context("PLE width missing")? as usize;
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        let next_norm = if let Some(next) = q.layers.get(i + 1) {
            next.input_norm
        } else {
            vector(
                self.weights
                    .norm
                    .as_ref()
                    .context("weighted final norm missing")?,
                s,
            )?
        };
        let layer = &mut q.layers[i];
        let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
        let null = 0u64;
        unsafe {
            s.launch_builder(&ops.branch_add)
                .arg(&mut w.x)
                .arg(&w.out)
                .arg(&layer.post_ffn_norm.context("post-FFN norm missing")?.ptr)
                .arg(&null)
                .arg(&null)
                .arg(&mut w.n)
                .arg(&(h as u32))
                .arg(&self.model_spec.rms_eps)
                .launch(grid(rows, 256))?;
        }
        layer.linear.run(
            &self.blas,
            "per_layer_input_gate.weight",
            phase,
            matrix(&w.x, s, 0, linear_rows, h)?,
            &mut TensorMut(matrix(&w.gu, s, 0, linear_rows, width)?),
        )?;
        if all_layers(trace) || i == self.trace_layer {
            trace_value(
                trace,
                s,
                &debug_name(trace, i, "ple_gate"),
                &w.gu,
                rows,
                width,
            )?;
        }
        q.common.gate.gelu(
            s,
            heads(&w.gu, s, 0, rows, 1, width as u32)?,
            &mut TensorMut(heads(&w.act, s, 0, rows, 1, width as u32)?),
            crate::backend::cuda::ops::GeluMode::Tanh,
        )?;
        if all_layers(trace) || i == self.trace_layer {
            trace_value(
                trace,
                s,
                &debug_name(trace, i, "ple_act"),
                &w.act,
                rows,
                width,
            )?;
            trace_value(
                trace,
                s,
                &debug_name(trace, i, "ple_identity"),
                &ops.per_layer_inputs,
                ops.per_layer_inputs.len() / width,
                width,
            )?;
        }
        unsafe {
            s.launch_builder(&ops.multiply)
                .arg(&w.act)
                .arg(
                    &ops.per_layer_inputs
                        .slice(i * linear_rows * width..(i + 1) * linear_rows * width),
                )
                .arg(&mut w.gu)
                .arg(&((rows * width) as i32))
                .launch(flat(rows * width))?;
        }
        if all_layers(trace) || i == self.trace_layer {
            trace_value(
                trace,
                s,
                &debug_name(trace, i, "ple_product"),
                &w.gu,
                rows,
                width,
            )?;
        }
        layer.linear.run(
            &self.blas,
            "per_layer_projection.weight",
            phase,
            matrix(&w.gu, s, 0, linear_rows, width)?,
            &mut TensorMut(matrix(&w.out, s, 0, linear_rows, h)?),
        )?;
        unsafe {
            s.launch_builder(&ops.branch_add)
                .arg(&mut w.x)
                .arg(&w.out)
                .arg(&layer.post_ple_norm.context("post-PLE norm missing")?.ptr)
                .arg(&layer.layer_scalar.context("layer scalar missing")?.ptr)
                .arg(&next_norm.ptr)
                .arg(&mut w.n)
                .arg(&(h as u32))
                .arg(&self.model_spec.rms_eps)
                .launch(grid(rows, 256))?;
        }
        if all_layers(trace) || i == self.trace_layer {
            trace_value(
                trace,
                s,
                &debug_name(trace, i, "per_layer_input"),
                &w.out,
                rows,
                h,
            )?;
            trace_value(trace, s, &debug_name(trace, i, "layer_out"), &w.x, rows, h)?;
        }
        Ok(())
    }

    pub(super) fn qwen_operator(
        &mut self,
        op: crate::backend::Operator,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        use crate::backend::Operator as O;
        let q = self.qwen.as_mut().context("Q38 runtime missing")?;
        let flint = self.model_spec.architecture == crate::model_spec::Architecture::Flint;
        let gemma = self.model_spec.architecture == crate::model_spec::Architecture::Gemma4;
        let layer = &q.layers[i];
        let s = &self.device.stream;
        let w = &mut self.work;
        let rows = step.rows;
        let h = self.model_spec.hidden as usize;
        let f = if gemma {
            self.model_spec.layers[i]
                .geometry
                .context("Gemma 4 FFN geometry missing")?
                .intermediate as usize
        } else {
            self.model_spec.intermediate as usize
        };
        let phase = if step.input == crate::backend::Input::Token {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        };
        let linear_rows = q.linear_rows(rows as u32, phase)? as usize;
        let domain = RowDomain {
            meta: self.state.seq_ref(s)?,
            row_base: 0,
            row_capacity: rows as u32,
        };
        match op {
            O::Embed => {
                q.select_positions(s, domain.meta, self.workspace_rows)?;
                let ids = if step.input == crate::backend::Input::Token {
                    &w.token
                } else {
                    &w.ids
                };
                unsafe {
                    if gemma {
                        let scale = bf16::from_f32((h as f32).sqrt()).to_f32();
                        s.launch_builder(
                            &q.gemma
                                .as_ref()
                                .context("Gemma 4 operations missing")?
                                .scaled_embed,
                        )
                        .arg(self.weights.embedding())
                        .arg(ids)
                        .arg(&w.gu)
                        .arg(&mut w.x)
                        .arg(&mut w.n)
                        .arg(&(rows as i32))
                        .arg(&(h as i32))
                        .arg(&scale)
                        .arg(self.weights.embedding())
                        .arg(&0i32)
                        .arg(&0i32)
                        .launch(flat(rows * h))?;
                    } else {
                        s.launch_builder(&self.ops.embed)
                            .arg(self.weights.embedding())
                            .arg(ids)
                            .arg(&mut w.x)
                            .arg(&(rows as i32))
                            .arg(&(h as i32))
                            .launch(flat(rows * h))?;
                    }
                }
                if step.input == crate::backend::Input::Ids
                    && let Some(images) = &self.images
                {
                    images.splice(
                        s,
                        self.position,
                        &mut TensorMut(matrix(&w.x, s, 0, rows, h)?),
                    )?;
                }
                trace_value(trace, s, "embed", &w.x, rows, h)?;
                if gemma {
                    let ops = q.gemma.as_mut().context("Gemma 4 operations missing")?;
                    let depth = self.model_spec.layers.len();
                    let width = self
                        .model_spec
                        .per_layer_input
                        .context("PLE width missing")? as usize;
                    self.blas.linear_typed(
                        phase,
                        LinearWeightRef::Bf16(ops.per_layer_projection),
                        matrix(&w.x, s, 0, linear_rows, h)?,
                        &mut TensorMut(matrix(&w.gu, s, 0, linear_rows, depth * width)?),
                    )?;
                    let identity_scale = bf16::from_f32((width as f32).sqrt()).to_f32();
                    let context_scale = bf16::from_f32(1.0 / (h as f32).sqrt()).to_f32();
                    let combine_scale = bf16::from_f32(std::f32::consts::FRAC_1_SQRT_2).to_f32();
                    unsafe {
                        s.launch_builder(&ops.per_layer)
                            .arg(ids)
                            .arg(&ops.per_layer_table.ptr)
                            .arg(&w.gu)
                            .arg(&ops.per_layer_norm.ptr)
                            .arg(&mut ops.per_layer_inputs)
                            .arg(&(linear_rows as i32))
                            .arg(&(depth as i32))
                            .arg(&(width as i32))
                            .arg(&identity_scale)
                            .arg(&context_scale)
                            .arg(&combine_scale)
                            .arg(&self.model_spec.rms_eps)
                            .launch(LaunchConfig {
                                grid_dim: ((linear_rows * depth) as u32, 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                }
            }
            O::Norm => {
                if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        layer.input_norm,
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: RmsMode::Standard,
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                    if all_layers(trace) || i == self.trace_layer {
                        let name = debug_name(trace, i, "norm1");
                        trace_value(trace, s, &name, &w.n, rows, h)?;
                    }
                    return Ok(());
                }
                if i == 0
                    || (!flint
                        && ((self.model_spec.moe.is_some()
                            && !self.qwen_device_prefix
                            && step.input == crate::backend::Input::Ids)
                            || !self
                                .hidden_capture
                                .as_ref()
                                .is_some_and(|capture| capture.layer_slots[i - 1].is_some())))
                {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        layer.input_norm,
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: norm_mode(&self.model_spec),
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                }
                if all_layers(trace) || i == self.trace_layer {
                    let name = debug_name(trace, i, "norm1");
                    trace_value(trace, s, &name, &w.n, rows, h)?;
                }
            }
            O::PostNorm | O::NextNorm => {
                if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                    if matches!(op, O::PostNorm) {
                        q.common.norm.rms_with(
                            s,
                            domain,
                            heads(&w.x, s, 0, rows, 1, h as u32)?,
                            layer.post_norm,
                            &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                            RmsOptions {
                                mode: RmsMode::Standard,
                                eps: self.model_spec.rms_eps,
                            },
                        )?;
                        if all_layers(trace) || i == self.trace_layer {
                            let name = debug_name(trace, i, "norm2");
                            trace_value(trace, s, &name, &w.n, rows, h)?;
                        }
                    }
                    return Ok(());
                }
                if gemma && matches!(op, O::NextNorm) {
                    if trace.is_some() {
                        let name = if all_layers(trace) {
                            i.to_string()
                        } else {
                            format!("layer.{i}")
                        };
                        trace_value(trace, s, &name, &w.x, rows, h)?;
                    }
                    return Ok(());
                }
                if matches!(op, O::PostNorm) {
                    if gemma {
                        let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
                        let null = 0u64;
                        unsafe {
                            s.launch_builder(&ops.branch_add)
                                .arg(&mut w.x)
                                .arg(&w.out)
                                .arg(&layer.post_norm.ptr)
                                .arg(&null)
                                .arg(&layer.pre_ffn_norm.context("pre-FFN norm missing")?.ptr)
                                .arg(&mut w.n)
                                .arg(&(h as u32))
                                .arg(&self.model_spec.rms_eps)
                                .launch(grid(rows, 256))?;
                        }
                    } else {
                        let mode = norm_mode(&self.model_spec);
                        if matches!(mode, RmsMode::Standard) {
                            self.ops.rms_add_fp32_with(
                                s,
                                domain,
                                ResidualRmsInput {
                                    x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                                    residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                                    weight: layer.post_norm,
                                },
                                &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                                self.model_spec.rms_eps,
                            )?;
                        } else {
                            q.common.norm.rms_add_with(
                                s,
                                domain,
                                ResidualRmsInput {
                                    x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                                    residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                                    weight: layer.post_norm,
                                },
                                &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                                RmsOptions {
                                    mode,
                                    eps: self.model_spec.rms_eps,
                                },
                            )?;
                        }
                    }
                    if trace.is_some() && (all_layers(trace) || i == self.trace_layer) {
                        let name = debug_name(trace, i, "post_attention");
                        trace_value(trace, s, &name, &w.x, rows, h)?;
                        let name = debug_name(trace, i, "norm2");
                        trace_value(trace, s, &name, &w.n, rows, h)?;
                    }
                } else {
                    let fused_moe_residual = matches!(layer.ffn, BoundFfn::Moe { .. })
                        && matches!(norm_mode(&self.model_spec), RmsMode::ZeroCentered);
                    if let Some(capture) = &mut self.hidden_capture
                        && let Some(slot) = capture.layer_slots[i]
                    {
                        let weight = if let Some(next) = q.layers.get(i + 1) {
                            next.input_norm
                        } else {
                            vector(
                                self.weights
                                    .norm
                                    .as_ref()
                                    .context("weighted final norm missing")?,
                                s,
                            )?
                        };
                        if fused_moe_residual {
                            q.common.norm.rms_with(
                                s,
                                domain,
                                heads(&w.x, s, 0, rows, 1, h as u32)?,
                                weight,
                                &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                                RmsOptions {
                                    mode: norm_mode(&self.model_spec),
                                    eps: self.model_spec.rms_eps,
                                },
                            )?;
                            let rect = copy::Rect {
                                rows,
                                width: h,
                                src_pitch: h,
                                src_col: 0,
                                dst_pitch: capture.width,
                                dst_col: slot as usize * h,
                            };
                            copy::columns(s, &w.x, &mut capture.values.slice_mut(..), rect)?;
                        } else {
                            self.ops.add_rmsnorm_capture(
                                s,
                                Some(domain),
                                ResidualRmsCapture {
                                    input: ResidualRmsInput {
                                        x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                                        residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                                        weight,
                                    },
                                    capture: &mut TensorMut(view(
                                        &capture.values,
                                        s,
                                        0,
                                        [rows as u32, capture.taps as u32, h as u32],
                                        [capture.width as u64, h as u64, 1],
                                    )?),
                                    slot,
                                },
                                &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                                RmsOptions {
                                    mode: norm_mode(&self.model_spec),
                                    eps: self.model_spec.rms_eps,
                                },
                            )?;
                        }
                    } else {
                        if !fused_moe_residual {
                            if matches!(norm_mode(&self.model_spec), RmsMode::Standard) {
                                let weight = if let Some(next) = q.layers.get(i + 1) {
                                    next.input_norm
                                } else {
                                    vector(
                                        self.weights
                                            .norm
                                            .as_ref()
                                            .context("weighted final norm missing")?,
                                        s,
                                    )?
                                };
                                self.ops.rms_add_fp32_with(
                                    s,
                                    domain,
                                    ResidualRmsInput {
                                        x: &mut TensorMut(heads(&w.x, s, 0, rows, 1, h as u32)?),
                                        residual: heads(&w.out, s, 0, rows, 1, h as u32)?,
                                        weight,
                                    },
                                    &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                                    self.model_spec.rms_eps,
                                )?;
                            } else {
                                unsafe {
                                    s.launch_builder(&self.ops.residual)
                                        .arg(&mut w.x)
                                        .arg(&w.out)
                                        .arg(&((rows * h) as i32))
                                        .launch(flat(rows * h))?;
                                }
                            }
                        }
                    }
                    if trace.is_some() {
                        let name = if all_layers(trace) {
                            i.to_string()
                        } else {
                            format!("layer.{i}")
                        };
                        trace_value(trace, s, &name, &w.x, rows, h)?;
                    }
                }
            }
            O::GateUp => {
                for (role, offset) in [
                    ("mlp.gate_proj.weight", 0),
                    ("mlp.up_proj.weight", self.workspace_rows * f),
                ] {
                    layer.linear.run(
                        &self.blas,
                        role,
                        phase,
                        matrix(&w.n, s, 0, linear_rows, h)?,
                        &mut TensorMut(matrix(&w.gu, s, offset, linear_rows, f)?),
                    )?;
                }
                if gemma && (all_layers(trace) || i == self.trace_layer) {
                    trace_value(trace, s, &debug_name(trace, i, "mlp_gate"), &w.gu, rows, f)?;
                    let name = debug_name(trace, i, "mlp_up");
                    if let Some(target) = trace.as_mut() {
                        ensure!(target.skip < rows, "trace selection is empty");
                        let start = self.workspace_rows * f + target.skip * f;
                        let values =
                            s.clone_dtoh(&w.gu.slice(start..start + (rows - target.skip) * f))?;
                        target.trace.add(
                            format!("{}.{name}", target.prefix),
                            vec![1, rows - target.skip, f],
                            values.into_iter().map(|value| value.to_f32()).collect(),
                        )?;
                    }
                }
            }
            O::Activate => {
                let gate = heads(&w.gu, s, 0, rows, 1, f as u32)?;
                let up = heads(&w.gu, s, self.workspace_rows * f, rows, 1, f as u32)?;
                let mut activated = TensorMut(heads(&w.act, s, 0, rows, 1, f as u32)?);
                if gemma {
                    let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
                    unsafe {
                        s.launch_builder(&ops.gated_activation)
                            .arg(&w.gu)
                            .arg(&w.gu.slice(self.workspace_rows * f..))
                            .arg(&mut w.act)
                            .arg(&((rows * f) as i32))
                            .arg(&2i32)
                            .launch(flat(rows * f))?;
                    }
                } else {
                    q.common.gate.swiglu(s, domain, gate, up, &mut activated)?;
                }
                #[cfg(test)]
                if gemma && (all_layers(trace) || i == self.trace_layer) {
                    let ops = q.gemma.as_mut().context("Gemma 4 operations missing")?;
                    ensure!(
                        ops.mlp_ones.len() >= rows * f && ops.mlp_activation.len() >= rows * f,
                        "Gemma MLP diagnostic workspace exceeds setup extent"
                    );
                    let activation = &mut ops.mlp_activation;
                    unsafe {
                        s.launch_builder(&ops.gated_activation)
                            .arg(&w.gu)
                            .arg(&ops.mlp_ones)
                            .arg(&mut *activation)
                            .arg(&((rows * f) as i32))
                            .arg(&2i32)
                            .launch(flat(rows * f))?;
                    }
                    trace_value(
                        trace,
                        s,
                        &debug_name(trace, i, "mlp_activation"),
                        activation,
                        rows,
                        f,
                    )?;
                }
                if gemma && (all_layers(trace) || i == self.trace_layer) {
                    trace_value(
                        trace,
                        s,
                        &debug_name(trace, i, "mlp_product"),
                        &w.act,
                        rows,
                        f,
                    )?;
                }
            }
            O::Down => {
                layer.linear.run(
                    &self.blas,
                    "mlp.down_proj.weight",
                    phase,
                    matrix(&w.act, s, 0, linear_rows, f)?,
                    &mut TensorMut(matrix(&w.out, s, 0, linear_rows, h)?),
                )?;
                if let Some(target) = trace
                    .as_mut()
                    .filter(|target| target.trace.all_layers || i == self.trace_layer)
                {
                    let name = if target.trace.all_layers {
                        format!("{i}.ffn")
                    } else {
                        "ffn".to_owned()
                    };
                    trace_value(trace, s, &name, &w.out, rows, h)?;
                }
            }
            O::Head => {
                if step.head == Head::None {
                    return Ok(());
                }
                if self.model_spec.architecture == crate::model_spec::Architecture::Glm5Next {
                    self.hc
                        .as_mut()
                        .context("GLM-5.3 mHC runtime is not bound")?
                        .collapse_final(s, rows, &mut w.x)?;
                }
                if !flint
                    && ((self.model_spec.moe.is_some()
                        && !self.qwen_device_prefix
                        && step.input == crate::backend::Input::Ids)
                        || !self.hidden_capture.as_ref().is_some_and(|capture| {
                            capture.layer_slots[self.model_spec.layers.len() - 1].is_some()
                        }))
                {
                    q.common.norm.rms_with(
                        s,
                        domain,
                        heads(&w.x, s, 0, rows, 1, h as u32)?,
                        vector(
                            self.weights
                                .norm
                                .as_ref()
                                .context("weighted final norm missing")?,
                            s,
                        )?,
                        &mut TensorMut(heads(&w.n, s, 0, rows, 1, h as u32)?),
                        RmsOptions {
                            mode: norm_mode(&self.model_spec),
                            eps: self.model_spec.rms_eps,
                        },
                    )?;
                }
                trace_value(trace, s, "norm", &w.n, rows, h)?;
                if step.head == Head::All {
                    if gemma {
                        #[cfg(test)]
                        {
                            let ops = q.gemma.as_mut().context("Gemma 4 operations missing")?;
                            let vocab = self.model_spec.vocab as usize;
                            self.blas.linear_typed(
                                phase,
                                LinearWeightRef::Bf16(matrix(&self.weights.head, s, 0, vocab, h)?),
                                matrix(&w.n, s, 0, linear_rows, h)?,
                                &mut TensorMut(matrix(&ops.logits, s, 0, linear_rows, vocab)?),
                            )?;
                            trace_value(trace, s, "head_raw", &ops.logits, linear_rows, vocab)?;
                            unsafe {
                                s.launch_builder(&ops.softcap)
                                    .arg(&mut ops.logits)
                                    .arg(&((linear_rows * vocab) as i64))
                                    .arg(
                                        &self
                                            .model_spec
                                            .logit_softcap
                                            .context("softcap missing")?,
                                    )
                                    .launch(flat(linear_rows * vocab))?;
                            }
                            return Ok(());
                        }
                        #[cfg(not(test))]
                        anyhow::bail!("Gemma 4 all-row head is only bound in model gates");
                    }
                    let spec = self
                        .spec
                        .as_mut()
                        .context("MoE all-row head requires draft state")?;
                    ensure!(
                        self.model_spec.moe.is_some() && (1..=spec.max_verify_rows).contains(&rows),
                        "all-row head rows={rows} outside the loaded MoE verification extent"
                    );
                    self.blas.linear_typed(
                        phase,
                        LinearWeightRef::Bf16(matrix(
                            &self.weights.head,
                            s,
                            0,
                            self.model_spec.vocab as usize,
                            h,
                        )?),
                        matrix(&w.n, s, 0, rows, h)?,
                        &mut TensorMut(matrix(
                            &spec.logits,
                            s,
                            0,
                            rows,
                            self.model_spec.vocab as usize,
                        )?),
                    )?;
                    // SAFETY: the bound logits and predictions cover every complete vocabulary row.
                    unsafe {
                        s.launch_builder(&self.ops.argmax_rows)
                            .arg(&spec.logits)
                            .arg(&mut spec.predictions)
                            .arg(&(self.model_spec.vocab as i32))
                            .arg(&spec.row_counts.slice(rows - 1..rows))
                            .launch(grid(rows, 1024))?;
                    }
                    spec.active_rows = rows;
                    return Ok(());
                }
                if flint {
                    let logits = self
                        .logits_f32
                        .as_mut()
                        .context("Flint FP32 logits are not bound")?;
                    let input = &w.n.slice((rows - 1) * h..rows * h);
                    self.ops.lm_head_f32(s, input, &self.weights.head, logits)?;
                    self.ops.argmax_f32(s, logits, &mut w.token)?;
                    if let Some(t) = trace.as_mut() {
                        let values = s.clone_dtoh(logits)?;
                        t.trace.add(
                            format!("{}.logits", t.prefix),
                            vec![1, 1, self.model_spec.vocab as usize],
                            values,
                        )?;
                    }
                    return Ok(());
                }
                self.blas.linear_typed(
                    phase,
                    LinearWeightRef::Bf16(matrix(
                        &self.weights.head,
                        s,
                        0,
                        self.model_spec.vocab as usize,
                        h,
                    )?),
                    matrix(&w.n, s, (rows - 1) * h, 1, h)?,
                    &mut TensorMut(matrix(&w.logits, s, 0, 1, self.model_spec.vocab as usize)?),
                )?;
                #[cfg(test)]
                if let Some(calls) = &q.glm_calls {
                    calls.borrow_mut()[0][9] += 1;
                }
                if gemma {
                    let ops = q.gemma.as_ref().context("Gemma 4 operations missing")?;
                    unsafe {
                        s.launch_builder(&ops.softcap)
                            .arg(&mut w.logits)
                            .arg(&(self.model_spec.vocab as i64))
                            .arg(&self.model_spec.logit_softcap.context("softcap missing")?)
                            .launch(flat(self.model_spec.vocab as usize))?;
                    }
                }
                unsafe {
                    s.launch_builder(&self.ops.argmax)
                        .arg(&w.logits)
                        .arg(&mut w.token)
                        .arg(&(self.model_spec.vocab as i32))
                        .launch(grid(1, 1024))?;
                }
                if let Some(t) = trace.as_mut() {
                    let values = s.clone_dtoh(&w.logits)?;
                    t.trace.add(
                        format!("{}.logits", t.prefix),
                        vec![1, 1, self.model_spec.vocab as usize],
                        values.into_iter().map(bf16::to_f32).collect(),
                    )?;
                }
            }
            _ => anyhow::bail!("operator {op:?} is not a Q38 common stage"),
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_glm_tests.rs"]
mod composition_tests;

#[cfg(test)]
mod tests {
    use super::default_chunk;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "exclusive CUDA; native KDA projection binding"]
    fn kda_native_projections_gpu() -> anyhow::Result<()> {
        use super::*;
        use crate::backend::cuda::{blas::Blas, closure::Closure, ops::Ops, weights::StoredTensor};
        use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
        use std::collections::BTreeMap;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/glm53-flash-config.json"
        )))?;
        let model = crate::model_spec::ModelSpec::glm5_next(&root, &serde_json::Value::Null)?;
        let spec = GdnSpec::from_model(&model)?;
        let device = crate::backend::cuda::Device::new(0)?;
        let s = &device.stream;
        let width = (spec.hv * spec.dv) as usize;
        let hidden = model.hidden as usize;
        let definitions = [
            ("q_proj", width, hidden),
            ("k_proj", width, hidden),
            ("v_proj", width, hidden),
            ("f_a_proj", 128, hidden),
            ("f_b_proj", width, 128),
            ("g_a_proj", 128, hidden),
            ("g_b_proj", width, 128),
            ("b_proj", 64, hidden),
        ];
        let mut entries = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        let mut bytes = 0;
        for (seed, (role, n, k)) in definitions.iter().copied().enumerate() {
            let mut weight = vec![bf16::ZERO; n * k];
            for row in 0..n {
                weight[row * k + (row * 17 + seed * 29) % k] =
                    bf16::from_f32((row % 7 + 1) as f32 / 8.0);
            }
            let role = format!("self_attn.{role}.weight");
            let name = format!("model.language_model.layers.2.{role}");
            let size = weight.len() * 2;
            entries.insert(
                name.clone(),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape: vec![n, k],
                    shard: "native-kda.safetensors".into(),
                    bytes: size,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, size as u64],
                    scope: TensorScope::Text,
                    layer: Some(2),
                    role,
                    scale_name: None,
                },
            );
            tensors.insert(name, StoredTensor::Bf16(s.clone_htod(&weight)?));
            bytes += size;
        }
        let catalog = WeightCatalog {
            linear_format: crate::model_spec::LinearFormat::Bf16,
            tensors,
            manifest: TensorManifest {
                entries,
                resident_bytes: bytes,
            },
            bytes,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: 0.0,
        };
        let dequant = LayerDequantPlan::from_manifest(&catalog.manifest, 2, catalog.linear_format)?;
        let linear = LayerLinearPlan::bind(&dequant, &catalog, None, s)?;
        let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        let blas = Blas::new(
            &device,
            &Ops::new(&device.ctx)?,
            &runtime,
            std::rc::Rc::new(Closure::new(false, "native KDA binding".into())),
        )?;
        let capacity = 65;
        let mut work = Work::new(s, &model, capacity)?;
        let input: Vec<_> = (0..capacity * hidden)
            .map(|i| bf16::from_f32(((i * 11 + 3) % 251) as f32 / 128.0 - 1.0))
            .collect();
        s.memcpy_htod(&input, &mut work.n)?;
        let project = |seed: usize, values: &[bf16]| -> Vec<bf16> {
            let (_, n, k) = definitions[seed];
            (0..values.len() / k * n)
                .map(|i| {
                    let row = i / n;
                    let output = i % n;
                    bf16::from_f32(
                        values[row * k + (output * 17 + seed * 29) % k].to_f32()
                            * (output % 7 + 1) as f32
                            / 8.0,
                    )
                })
                .collect()
        };
        let expected: Vec<_> = (0..3).map(|seed| project(seed, &input)).collect();
        let a = project(4, &project(3, &input));
        let z = project(6, &project(5, &input));
        let b = project(7, &input);
        let mut defects = 0;
        let mut cells = 0;
        for rows in [1, 2, 5, 31, 33, 65] {
            linear.prepare(&blas, rows as u32)?;
            for phase in [ExecutionPhase::Prefill, ExecutionPhase::Decode] {
                if phase == ExecutionPhase::Decode && rows != 1 {
                    continue;
                }
                for graph in [false, true] {
                    device.finish_upload()?;
                    if graph {
                        s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    }
                    let launched = project_kda(
                        &linear,
                        &blas,
                        s,
                        phase,
                        &mut work,
                        spec,
                        [rows, capacity, hidden],
                    );
                    if graph {
                        let captured =
                            s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                        launched?;
                        captured?
                            .context("empty native projection graph")?
                            .launch()?;
                    } else {
                        launched?;
                    }
                    let qkv = s.clone_dtoh(&work.qkv)?;
                    let mixer = work.mixer.as_ref().context("native mixer workspace")?;
                    let ab = s.clone_dtoh(&mixer.ab)?;
                    let gate = s.clone_dtoh(&mixer.gate)?;
                    let mut mismatches = 0;
                    let mut wrong_segment = 0;
                    for row in 0..rows {
                        for part in 0..3 {
                            for d in 0..width {
                                let actual = qkv[(row * 3 + part) * width + d].to_bits();
                                mismatches += usize::from(
                                    actual != expected[part][row * width + d].to_bits(),
                                );
                                wrong_segment += usize::from(
                                    actual != expected[(part + 1) % 3][row * width + d].to_bits(),
                                );
                            }
                        }
                    }
                    for (actual, expected) in [
                        (&ab[..rows * width], &a[..rows * width]),
                        (&gate[..rows * width], &z[..rows * width]),
                        (
                            &ab[capacity * width..capacity * width + rows * 64],
                            &b[..rows * 64],
                        ),
                    ] {
                        mismatches += actual
                            .iter()
                            .zip(expected)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                    }
                    let passed = mismatches == 0 && wrong_segment > 0;
                    defects += usize::from(!passed);
                    cells += 1;
                    println!(
                        "{} KDA_NATIVE_PROJECTIONS layer=2 rows={rows} graph={graph} phase={phase:?} qkv=[{rows},24576] mismatches={mismatches} wrong_segment={wrong_segment} native_bf16=true copies=3",
                        if passed { "PASS" } else { "DEFECT" }
                    );
                }
            }
        }
        println!("KDA_NATIVE_PROJECTIONS_SUMMARY cells={cells} defects={defects}");
        ensure!(defects == 0, "native projection binding defects={defects}");
        Ok(())
    }

    fn geometry() -> crate::model_spec::ModelSpec {
        crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))
            .unwrap(),
            &serde_json::json!({}),
        )
        .unwrap()
    }

    #[test]
    fn capability_check_matches_compiled_mixer_geometries() -> Result<()> {
        let qwen: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let spec = crate::model_spec::ModelSpec::qwen(&qwen, &serde_json::json!({}))?;
        require_qwen35_geometry(&spec)?;
        let minicpm = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        require_qwen35_geometry(&minicpm)?;
        for original in [&spec, &minicpm] {
            for field in [
                "hidden_size",
                "intermediate_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "q_width",
                "max_position_embeddings",
                "rotary_dim",
                "num_hidden_layers",
                "rms_norm_eps",
                "rope_theta",
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "gdn",
                "linear_format",
                "architecture",
            ] {
                let mut bad = original.clone();
                match field {
                    "hidden_size" => bad.hidden += 1,
                    "intermediate_size" => bad.intermediate += 1,
                    "vocab_size" => bad.vocab += 1,
                    "num_attention_heads" => bad.q_heads += 1,
                    "num_key_value_heads" => bad.kv_heads += 1,
                    "head_dim" => bad.head_dim += 1,
                    "q_width" => bad.q_width += 1,
                    "max_position_embeddings" => bad.max_position += 1,
                    "rotary_dim" => bad.rotary_dim += 1,
                    "num_hidden_layers" => {
                        bad.layers.pop();
                    }
                    "rms_norm_eps" => bad.rms_eps *= 2.0,
                    "rope_theta" => bad.rope_theta *= 2.0,
                    "linear_num_key_heads" => bad.gdn.as_mut().unwrap().key_heads += 1,
                    "linear_num_value_heads" => bad.gdn.as_mut().unwrap().value_heads += 1,
                    "linear_key_head_dim" => bad.gdn.as_mut().unwrap().key_head_dim += 1,
                    "linear_value_head_dim" => bad.gdn.as_mut().unwrap().value_head_dim += 1,
                    "linear_conv_kernel_dim" => bad.gdn.as_mut().unwrap().conv_kernel += 1,
                    "gdn" => bad.gdn = None,
                    "linear_format" => {
                        bad.linear_format = match original.linear_format {
                            crate::model_spec::LinearFormat::Bf16 => {
                                crate::model_spec::LinearFormat::fp8_block(
                                    [128, 128],
                                    safetensors::Dtype::BF16,
                                )?
                            }
                            crate::model_spec::LinearFormat::Fp8E4m3Block { .. } => {
                                crate::model_spec::LinearFormat::Bf16
                            }
                            crate::model_spec::LinearFormat::Fp8E4m3Channel => {
                                crate::model_spec::LinearFormat::Bf16
                            }
                        }
                    }
                    "architecture" => bad.architecture = crate::model_spec::Architecture::Llama,
                    _ => unreachable!(),
                }
                let error = require_qwen35_geometry(&bad).unwrap_err().to_string();
                assert!(error.contains(&format!("{field}=")), "{field}: {error}");
            }
        }
        let error = Engine::load_qwen(
            Path::new("unused-model"),
            0,
            32,
            Some(1),
            Settings {
                tree_builder: crate::backend::TreeBuilder::Waves,
                spec_budget: crate::backend::SpecBudget::Chain,
                spec_graph: false,
                draft_model: None,
                prefix_cache: false,
                kv_format: crate::kv::KvFormat::Bf16,
                runtime_dir: None,
            },
            super::super::SetupMode::Prepare,
            minicpm,
        )
        .err()
        .context("MoE model reached device setup without a runtime directory")?;
        assert!(error.to_string().contains("runtime directory"), "{error}");
        Ok(())
    }

    #[test]
    fn the_default_chunk_is_bounded_by_the_cuda_grid_yz_limit() {
        assert_eq!(default_chunk(100_000), 65_504);
        assert_eq!(default_chunk(100_000) % 32, 0);
        assert_eq!(default_chunk(500), 500);
    }

    /// An admission of 65536 rows selects 65504, a multiple of the budget alignment.
    #[test]
    fn the_default_chunk_cap_preserves_the_budget_alignment() -> Result<()> {
        let budget = WorkspaceBudget::new(
            WorkspaceRequest {
                capacity: 70000,
                requested: None,
                free_after_resident: 1 << 40,
                total: 1 << 40,
                alignment: super::CHUNK_ALIGNMENT,
                row_bytes: 1,
                fixed_transient_bytes: 0,
            },
            |rows| Ok(rows * 4),
            |admitted| Ok(default_chunk(admitted)),
        )?;
        assert_eq!(budget.plan.chunk, 65_504);
        assert_eq!(budget.plan.chunk % super::CHUNK_ALIGNMENT, 0);
        Ok(())
    }

    /// A stand-in plan recording the rows it was narrowed to.
    #[derive(Clone, Debug, PartialEq)]
    struct FakePlan {
        max_rows: usize,
    }
    impl FakePlan {
        fn for_rows(plan: &FakePlan, rows: u32) -> Result<FakePlan> {
            ensure!(
                rows > 0 && rows as usize <= plan.max_rows,
                "rows outside binding"
            );
            Ok(FakePlan {
                max_rows: rows as usize,
            })
        }
    }

    #[test]
    fn select_returns_the_plan_narrowed_to_the_steps_live_rows() {
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, (2048, 5), FakePlan::for_rows)
            .expect("bind");
        // A step size never seen before narrows host-side at dispatch; no preparation step exists.
        assert_eq!(
            plans.select(1021, &FakePlan::for_rows).unwrap().max_rows,
            1021
        );
        assert_eq!(
            plans.select(700, &FakePlan::for_rows).unwrap().max_rows,
            700
        );
        // Rows 1..=5 borrow prebound small plans.
        for rows in 1..=5 {
            assert_eq!(
                plans.select(rows, &FakePlan::for_rows).unwrap().max_rows,
                rows
            );
        }
        assert!(plans.select(0, &FakePlan::for_rows).is_err());
        assert!(plans.select(2049, &FakePlan::for_rows).is_err());
    }

    #[test]
    fn a_step_only_runs_a_plan_narrowed_to_its_own_rows() {
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, (2048, 5), FakePlan::for_rows)
            .expect("bind");
        assert_eq!(
            plans.select(1021, &FakePlan::for_rows).unwrap().max_rows,
            1021,
            "a 1021-row step must not run a plan sized for the admitted chunk"
        );
    }

    #[test]
    fn every_step_size_of_a_chunked_prompt_gets_its_own_plan() {
        // 4093 tokens over a 2048 chunk: two 2048 steps and one 997 tail.
        let plans = PrefillPlans::bind(FakePlan { max_rows: 2048 }, (2048, 5), FakePlan::for_rows)
            .expect("bind");
        for step in [2048usize, 997] {
            assert_eq!(
                plans.select(step, &FakePlan::for_rows).unwrap().max_rows,
                step
            );
        }
    }

    use super::*;
    use crate::prefill::{WorkspaceBudget, WorkspaceRequest};

    #[test]
    fn prefill_linear_rows_use_setup_bound_buckets() -> Result<()> {
        for (rows, workspace, bound) in [
            (1u32, 2048u32, 1u32),
            (7, 2048, 7),
            (16, 2048, 16),
            (17, 2048, 32),
            (31, 2048, 32),
            (2037, 2048, 2048),
            (17, 20, 20),
        ] {
            assert_eq!(bucketed_prefill_rows(rows, workspace)?, bound);
        }
        assert!(bucketed_prefill_rows(0, 2048).is_err());
        assert!(bucketed_prefill_rows(2049, 2048).is_err());
        Ok(())
    }

    #[test]
    fn non_gemma_workspace_admission_is_unchanged() -> Result<()> {
        fn expected(
            spec: &crate::model_spec::ModelSpec,
            rows: usize,
            capacity: usize,
            verify_rows: usize,
        ) -> Result<usize> {
            let attention = if spec.architecture == crate::model_spec::Architecture::Glm5Next {
                None
            } else {
                Some(FullAttentionSpec::from_model(spec)?)
            };
            let (prefill, decode, verify) = if let Some(attention) = attention.as_ref() {
                (
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::workspace(attention, limits(rows, capacity)?)?.bytes
                    } else {
                        FullAttentionOps::preprocess_workspace(attention, limits(rows, capacity)?)?
                            .bytes
                    },
                    FullAttentionOps::decode_workspace(attention, 1, capacity as u32)?.bytes,
                    if cfg!(feature = "rocm") {
                        FullAttentionOps::decode_workspace(
                            attention,
                            rows.min(verify_rows) as u32,
                            capacity as u32,
                        )?
                        .bytes
                    } else {
                        FullAttentionOps::workspace(attention, limits(verify_rows, capacity)?)?
                            .bytes
                    },
                )
            } else {
                (0, 0, 0)
            };
            let gdn = GdnSpec::from_model(spec)?;
            let prefill_gdn = GdnOps::workspace(&gdn, rows as u32, GdnPolicy::P2)?.bytes;
            let decode_gdn = GdnOps::workspace(&gdn, 1, GdnPolicy::P2)?.bytes;
            Ok(usize::try_from(
                prefill
                    .max(decode)
                    .max(verify)
                    .max(prefill_gdn)
                    .max(decode_gdn),
            )?)
        }
        let minicpm = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let loopvl = crate::model_spec::ModelSpec::loopvl(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/loopvl-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        let glm = crate::model_spec::ModelSpec::glm5_next(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/glm53-flash-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        for (spec, rows, capacity) in [(geometry(), 17, 4096), (minicpm, 33, 2048), (glm, 17, 4096)]
        {
            let actual = Runtime::workspace(&spec, rows, capacity, GdnPolicy::P2, 5, None)?;
            let expected = expected(&spec, rows, capacity, 5)?;
            ensure!(
                actual == expected,
                "non-Gemma admission changed for {:?}: actual={actual} expected={expected}",
                spec.architecture
            );
        }
        ensure!(
            Runtime::workspace(&loopvl, 65, 4096, GdnPolicy::P2, 5, None).is_err(),
            "LoopVL full-attention admission was accepted"
        );
        Ok(())
    }

    #[test]
    fn small_prefill_plans_select_structural_rows_without_rebinding() -> Result<()> {
        for capacity in [1, 2, 3, 4, 5, 16, 2048, 4096] {
            let narrow = |&full: &usize, rows: u32| {
                assert_eq!(full, capacity);
                Ok(rows as usize)
            };
            let plans = PrefillPlans::bind(capacity, (capacity, 5), narrow)?;
            assert_eq!(plans.small, (1..=5.min(capacity)).collect::<Vec<_>>());
            for rows in 1..=capacity {
                assert_eq!(*plans.select(rows, &narrow)?, rows);
            }
            for rows in 1..=5.min(capacity) {
                ensure!(
                    matches!(plans.select(rows, &narrow)?, Narrowed::Borrowed(_)),
                    "small extent {rows} did not select a prebound plan"
                );
            }
            assert!(plans.select(0, &narrow).is_err());
            assert!(plans.select(capacity + 1, &narrow).is_err());
            assert!(plans.select(usize::MAX, &narrow).is_err());
            let budget =
                Runtime::workspace(&geometry(), capacity, 4096, GdnPolicy::P2, 5, None)? as u64;
            for rows in 1..=5.min(capacity) {
                assert!(
                    GdnOps::workspace(
                        &crate::op_contract::GdnSpec::q38(),
                        rows as u32,
                        GdnPolicy::P2
                    )?
                    .bytes
                        <= budget
                );
                assert!(
                    FullAttentionOps::preprocess_workspace(
                        &FullAttentionSpec::q38(),
                        limits(rows, 4096)?
                    )?
                    .bytes
                        <= budget
                );
            }
        }
        assert!(PrefillPlans::bind(0, (0, 5), |_, rows| Ok(rows as usize)).is_err());
        let plans = PrefillPlans::bind(4096, (4096, 16), |_, rows| Ok(rows as usize))?;
        for rows in 1..=16 {
            assert!(matches!(
                plans.select(rows, &|_, row| Ok(row as usize))?,
                Narrowed::Borrowed(_)
            ));
        }
        assert!(PrefillPlans::bind(32, (32, 17), |_, rows| Ok(rows as usize)).is_err());
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nested_verify_routes_only_small_rows_to_tcmqa() -> Result<()> {
        for rows in [1, 4, 65, 2048] {
            assert!(!nested_verify_tcmqa(false, rows, 5)?);
        }
        for rows in 1..=5 {
            assert!(nested_verify_tcmqa(true, rows, 5)?);
        }
        assert!(nested_verify_tcmqa(true, 0, 5).is_err());
        assert!(nested_verify_tcmqa(true, 6, 5).is_err());
        for rows in 1..=16 {
            assert!(nested_verify_tcmqa(true, rows, 16)?);
        }
        assert!(nested_verify_tcmqa(true, 17, 16).is_err());
        Ok(())
    }

    #[test]
    fn workspace_binds_nested_verify_tcmqa_rows() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        for capacity in [256, 16_832] {
            let required =
                FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(5, capacity)?)?;
            let allocated = Runtime::workspace(&geometry(), 5, capacity, GdnPolicy::P2, 5, None)?;
            ensure!(
                required.bytes <= allocated as u64,
                "nested verify workspace {required:?} exceeds allocation {allocated}"
            );
            let mtp = crate::mtp::AttachmentBudget::new(&spec, capacity)?;
            let row_bytes = Runtime::row_bytes(&spec)? + mtp.row_bytes;
            let fixed = Runtime::fixed_bytes(&spec, capacity)? + mtp.fixed_bytes;
            let workspace = |rows| {
                Ok(
                    Runtime::workspace(&geometry(), rows, capacity, GdnPolicy::P2, 5, None)?
                        + mtp.workspace(rows)?,
                )
            };
            let total = 24usize << 30;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(5),
                free_after_resident: total / 16 + fixed + 5 * row_bytes + workspace(5)?,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(request, workspace, Ok)?;
            assert_eq!(budget.plan.chunk, 5);
            assert_eq!(budget.plan.row_limit, 5);
            assert_eq!(
                budget.planned_transient_bytes + budget.plan.reserve_bytes,
                request.free_after_resident
            );
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    workspace,
                    Ok,
                )
                .is_err()
            );
            let admitted = WorkspaceBudget::new(
                WorkspaceRequest {
                    requested: None,
                    free_after_resident: total / 4,
                    ..request
                },
                workspace,
                |rows| Ok(default_chunk(rows)),
            )?;
            assert!(admitted.plan.chunk >= 5);
            assert!(admitted.planned_transient_bytes + admitted.plan.reserve_bytes <= total / 4);
            println!(
                "MTP_M5_ADMISSION_PASS capacity={capacity} total={total} free_after_resident={} minimum_free={} chunk={} row_limit={} planned_transient={} reserve={}",
                total / 4,
                request.free_after_resident,
                admitted.plan.chunk,
                admitted.plan.row_limit,
                admitted.planned_transient_bytes,
                admitted.plan.reserve_bytes,
            );
        }
        Ok(())
    }

    #[test]
    fn fa2_workspace_budget_includes_lse_and_decode_scratch() -> Result<()> {
        let root = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/q38-config.json"
        )))?;
        let spec = crate::model_spec::ModelSpec::qwen(&root, &serde_json::json!({}))?;
        let work_row_bytes = crate::work_layout::RowWorkSpec::new(&spec)?.row_bytes()?;
        let capacity = 256;
        let total = 1 << 30;
        let fixed = spec.vocab as usize * 2 + 12 + 64 * size_of::<u64>() + capacity * 12;
        for rows in [1, 65, 128] {
            let scratch = Runtime::workspace(&geometry(), rows, capacity, GdnPolicy::P2, 5, None)?;
            for required in [
                FullAttentionOps::preprocess_workspace(
                    &FullAttentionSpec::q38(),
                    limits(rows, capacity)?,
                )?,
                FullAttentionOps::workspace(&FullAttentionSpec::q38(), limits(1, capacity)?)?,
                GdnOps::workspace(
                    &crate::op_contract::GdnSpec::q38(),
                    rows as u32,
                    GdnPolicy::P2,
                )?,
                GdnOps::workspace(&crate::op_contract::GdnSpec::q38(), 1, GdnPolicy::P2)?,
            ] {
                ensure!(
                    scratch as u64 >= required.bytes,
                    "operator scratch underbudgeted"
                );
            }
            let lse = rows * 24 * 4;
            let allocated = rows * work_row_bytes + lse + rows * 12 + fixed + scratch;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(rows),
                free_after_resident: total / 16 + allocated,
                total,
                alignment: 32,
                row_bytes: Runtime::row_bytes(&spec)?,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(
                request,
                |r| Runtime::workspace(&geometry(), r, capacity, GdnPolicy::P2, 5, None),
                Ok,
            )?;
            assert_eq!(budget.planned_transient_bytes, allocated);
            assert_eq!(budget.plan.row_limit, rows);
            // The same request must fail if even one byte of the allocation is missing.
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(&geometry(), r, capacity, GdnPolicy::P2, 5, None),
                    Ok,
                )
                .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn scan_budget_accounts_for_every_prebound_bucket() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::qwen(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )))?,
            &serde_json::json!({}),
        )?;
        let policy = GdnPolicy::P2;
        let capacity = 8192;
        let total = 8usize << 30;
        let row_bytes = Runtime::row_bytes(&spec)?;
        let fixed = Runtime::fixed_bytes(&spec, capacity)?;
        for rows in [512, 513, 1021, 1024, 2048, 4096, 8192] {
            let scratch = Runtime::workspace(&geometry(), rows, capacity, policy, 5, None)?;
            let gdn = policy.workspace(&crate::op_contract::GdnSpec::q38(), rows as u32)?;
            assert!(scratch as u64 >= gdn.bytes);
            let required = rows * row_bytes + fixed + scratch;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(rows),
                free_after_resident: total / 16 + required,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(
                request,
                |r| Runtime::workspace(&geometry(), r, capacity, policy, 5, None),
                Ok,
            )?;
            assert_eq!(budget.plan.chunk, rows);
            assert_eq!(budget.operator_workspace_bytes, scratch);
            assert_eq!(budget.planned_transient_bytes, required);
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    |r| Runtime::workspace(&geometry(), r, capacity, policy, 5, None),
                    Ok,
                )
                .is_err()
            );
        }
        Ok(())
    }

    fn zero_flint_sinks(engine: &mut Engine) -> Result<()> {
        let stream = engine.device.stream.clone();
        let runtime = engine
            .qwen
            .as_mut()
            .context("Flint runtime missing for sink control")?;
        for layer in &mut runtime.layers {
            if let BoundMixer::Flint { sinks, .. } = &mut layer.mixer {
                let zeros = vec![bf16::ZERO; sinks.len()];
                stream.memcpy_htod(&zeros, &mut sinks.slice_mut(..))?;
            }
        }
        stream.synchronize()?;
        Ok(())
    }

    fn decode_window(
        engine: &mut Engine,
        prefix: &[u32],
        targets: &[u32],
        graph: bool,
    ) -> Result<Vec<Vec<f32>>> {
        engine.reset()?;
        engine.prefill(prefix, None)?;
        let mut rows = Vec::new();
        for target in targets {
            engine.set_token(*target)?;
            engine.decode(graph, None)?;
            rows.push(engine.decode_logits()?);
        }
        Ok(rows)
    }

    fn graph_eager_window(
        engine: &mut Engine,
        prefix: &[u32],
        targets: &[u32],
    ) -> Result<(Vec<Vec<f32>>, u64, u32)> {
        engine.reset()?;
        engine.prefill(prefix, None)?;
        let stream = engine.device.stream.clone();
        let mut checkpoint = super::super::super::state::StateCheckpoint::new(
            &engine.state,
            &engine.model_spec,
            1,
            &stream,
        )?;
        let mut rows = Vec::new();
        let mut bits = 0u64;
        let mut differing_rows = 0u32;
        for target in targets {
            engine.set_token(*target)?;
            let position = engine.position;
            let saved_position = stream.clone_dtoh(&engine.work.position)?;
            let saved_length = stream.clone_dtoh(&engine.work.length)?;
            let saved_pending = engine.pending_token;
            let saved_committed = engine.committed_prefix_ids.clone();
            let saved_resident_len = engine.resident_ids_len;
            let saved_input_kind = engine.last_input_kind;
            checkpoint.save(&engine.state, &engine.work.token, position, &stream, None)?;
            engine
                .moe_prefill
                .as_ref()
                .context("Flint prefill missing for graph/eager reseed")?
                .reseed_request(
                    engine
                        .moe_slots
                        .as_mut()
                        .context("Flint slots missing for graph/eager reseed")?,
                )?;
            engine.refresh_verify_schedule()?;
            engine
                .moe_slots
                .as_mut()
                .context("Flint slots missing for graph/eager freeze")?
                .freeze_probe_admissions()?;
            engine.decode(true, None)?;
            let graph = engine.decode_logits()?;
            checkpoint.restore(&mut engine.state, &mut engine.work.token, &stream)?;
            stream.memcpy_htod(&saved_position, &mut engine.work.position)?;
            stream.memcpy_htod(&saved_length, &mut engine.work.length)?;
            engine.position = position;
            engine.pending_token = saved_pending;
            engine.committed_prefix_ids = saved_committed;
            engine.resident_ids_len = saved_resident_len;
            engine.last_input_kind = saved_input_kind;
            engine.decode(false, None)?;
            let eager = engine.decode_logits()?;
            let row_bits = graph
                .iter()
                .zip(&eager)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count() as u64;
            bits += row_bits;
            if graph
                .iter()
                .zip(&eager)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                differing_rows += 1;
            }
            rows.push(eager);
        }
        Ok((rows, bits, differing_rows))
    }

    fn graph_eager_forced_prefix(
        engine: &mut Engine,
        ids: &[u32],
    ) -> Result<(f64, Vec<Vec<f32>>, u64, u32)> {
        engine.reset()?;
        engine.prefill(&ids[..1], None)?;
        let stream = engine.device.stream.clone();
        let mut checkpoint = super::super::super::state::StateCheckpoint::new(
            &engine.state,
            &engine.model_spec,
            1,
            &stream,
        )?;
        let mut rows = vec![engine.decode_logits()?];
        let mut total_nll = 0f64;
        let mut bits = 0u64;
        let mut differing_rows = 0u32;
        for target in &ids[1..] {
            let logits = engine.decode_logits()?;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let logsumexp = logits
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f32>()
                .ln();
            total_nll += f64::from(max + logsumexp - logits[*target as usize]);
            engine.set_token(*target)?;
            let position = engine.position;
            let saved_position = stream.clone_dtoh(&engine.work.position)?;
            let saved_length = stream.clone_dtoh(&engine.work.length)?;
            let saved_pending = engine.pending_token;
            let saved_committed = engine.committed_prefix_ids.clone();
            let saved_resident_len = engine.resident_ids_len;
            let saved_input_kind = engine.last_input_kind;
            checkpoint.save(&engine.state, &engine.work.token, position, &stream, None)?;
            engine
                .moe_prefill
                .as_ref()
                .context("Flint prefill missing for graph/eager reseed")?
                .reseed_request(
                    engine
                        .moe_slots
                        .as_mut()
                        .context("Flint slots missing for graph/eager reseed")?,
                )?;
            engine.refresh_verify_schedule()?;
            engine
                .moe_slots
                .as_mut()
                .context("Flint slots missing for graph/eager freeze")?
                .freeze_probe_admissions()?;
            engine.decode(true, None)?;
            let graph = engine.decode_logits()?;
            checkpoint.restore(&mut engine.state, &mut engine.work.token, &stream)?;
            stream.memcpy_htod(&saved_position, &mut engine.work.position)?;
            stream.memcpy_htod(&saved_length, &mut engine.work.length)?;
            engine.position = position;
            engine.pending_token = saved_pending;
            engine.committed_prefix_ids = saved_committed;
            engine.resident_ids_len = saved_resident_len;
            engine.last_input_kind = saved_input_kind;
            engine.decode(false, None)?;
            let eager = engine.decode_logits()?;
            let row_bits = graph
                .iter()
                .zip(&eager)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count() as u64;
            if row_bits != 0 {
                differing_rows += 1;
            }
            bits += row_bits;
            rows.push(eager);
        }
        Ok((
            total_nll / (ids.len() - 1) as f64,
            rows,
            bits,
            differing_rows,
        ))
    }

    fn forced_prefix(
        engine: &mut Engine,
        ids: &[u32],
        graph: bool,
        expected: Option<&[Vec<f32>]>,
    ) -> Result<(f64, Vec<Vec<f32>>, u64, u32)> {
        engine.reset()?;
        engine.prefill(&ids[..1], None)?;
        let mut total_nll = 0f64;
        let mut rows = Vec::new();
        let mut bit_mismatches = 0u64;
        let mut max_ulp = 0u32;
        let read_logits = |engine: &Engine| -> Result<Vec<f32>> { engine.decode_logits() };
        for target in &ids[1..] {
            let logits = read_logits(engine)?;
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let logsumexp = logits
                .iter()
                .map(|value| (value - max).exp())
                .sum::<f32>()
                .ln();
            total_nll += f64::from(max + logsumexp - logits[*target as usize]);
            if let Some(expected) = expected {
                let row_bits = logits
                    .iter()
                    .zip(expected[rows.len()].iter())
                    .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
                    .count();
                bit_mismatches += u64::try_from(row_bits)?;
                if row_bits != 0 {
                    max_ulp += 1;
                }
            }
            rows.push(logits);
            engine.set_token(*target)?;
            engine.decode(graph, None)?;
        }
        let final_logits = read_logits(engine)?;
        if let Some(expected) = expected {
            let row_bits = final_logits
                .iter()
                .zip(expected[rows.len()].iter())
                .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
                .count();
            bit_mismatches += u64::try_from(row_bits)?;
            if row_bits != 0 {
                max_ulp += 1;
            }
        }
        rows.push(final_logits);
        Ok((
            total_nll / (ids.len() - 1) as f64,
            rows,
            bit_mismatches,
            max_ulp,
        ))
    }

    #[test]
    #[ignore = "GPU window; FF_MODEL, FLYINGFISH_RUNTIME_DIR and FF_FLINT_PASSAGE required"]
    fn flint_engine_controls_and_graph_eager_nll_gpu() -> Result<()> {
        use crate::backend::{Backend, Settings};
        use std::{env, fs::File};

        let dir = PathBuf::from(env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let runtime =
            PathBuf::from(env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
        let passage = PathBuf::from(env::var_os("FF_FLINT_PASSAGE").context("passage missing")?);
        ensure!(
            env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let reference_metrics = PathBuf::from(
            env::var_os("FF_FLINT_REFERENCE_METRICS")
                .context("FF_FLINT_REFERENCE_METRICS missing")?,
        );
        let settings = Settings {
            runtime_dir: Some(runtime),
            prefix_cache: false,
            draft_model: None,
            tree_builder: crate::backend::TreeBuilder::Serial,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let passage: serde_json::Value =
            serde_json::from_reader(File::open(passage).context("open passage")?)?;
        let mut ids = passage["input_ids"]
            .as_array()
            .context("input_ids missing")?
            .iter()
            .map(|value| value.as_u64().and_then(|value| u32::try_from(value).ok()))
            .collect::<Option<Vec<_>>>()
            .context("invalid passage token id")?;
        ids.truncate(600);
        let reference_document: serde_json::Value = serde_json::from_reader(
            File::open(reference_metrics).context("open reference metrics")?,
        )?;
        let reference_runs = reference_document
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![reference_document]);
        let reference_nlls = reference_runs
            .iter()
            .map(|reference| {
                reference["metrics"]["per_position_nll"]
                    .as_array()
                    .context("reference per-position NLL missing")?
                    .iter()
                    .map(|value| value.as_f64().context("invalid reference per-position NLL"))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            reference_nlls.len() == 4 && reference_nlls.iter().all(|values| values.len() >= 599),
            "expected four measured Flint reference runs"
        );
        let prediction_nll = |rows: &[Vec<f32>], start: usize, end: usize| -> Result<f64> {
            ensure!(
                rows.len() >= end && ids.len() > end,
                "prediction NLL rows={} end={end} differ",
                rows.len()
            );
            let mut total = 0f64;
            for position in start..end {
                let logits = &rows[position];
                let target = ids[position + 1] as usize;
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let logsumexp = logits
                    .iter()
                    .map(|value| (value - max).exp())
                    .sum::<f32>()
                    .ln();
                total += f64::from(max + logsumexp - logits[target]);
            }
            Ok(total / (end - start) as f64)
        };
        let mut ring_engine = Backend::calibrate(&dir, 0, 600, Some(520), settings.clone())?
            .context("Flint ring calibration missing")?;
        let ring_ids = &ids[..520];
        let sliding_state = ring_engine.model_spec.layers[3].state_index as usize;
        let ring = ring_engine.state.kv_ring(sliding_state)?;
        ensure!(
            ring_engine.chunk > ring,
            "ring gate chunk={} does not exceed ring={ring}",
            ring_engine.chunk
        );
        ring_engine.trace_layer = 3;
        let mut ring_trace = Trace::default();
        ring_engine.prefill(ring_ids, Some(&mut ring_trace))?;
        let ring_attention = ring_trace.values("prefill.0.debug.mixer_out")?;
        let prefill_logits = ring_engine.decode_logits()?;
        ensure!(
            prefill_logits.iter().all(|value| value.is_finite()),
            "interleaved sliding-ring prefill has nonfinite logits"
        );
        ring_engine.reset()?;
        ring_engine.flint_legacy_overflow_prefill = true;
        let mut legacy_trace = Trace::default();
        ring_engine.prefill(ring_ids, Some(&mut legacy_trace))?;
        let legacy_attention = legacy_trace.values("prefill.0.debug.mixer_out")?;
        ensure!(
            ring_attention.len() == ring_ids.len() * ring_engine.model_spec.q_width as usize
                && legacy_attention.len() == ring_attention.len(),
            "ring control attention extent differs"
        );
        let attention_bits = legacy_attention
            .iter()
            .zip(&ring_attention)
            .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
            .count();
        let legacy_logits = ring_engine.decode_logits()?;
        let legacy_bits = legacy_logits
            .iter()
            .zip(&prefill_logits)
            .filter(|(actual, reference)| actual.to_bits() != reference.to_bits())
            .count();
        ensure!(
            attention_bits > 0 && legacy_bits > 0,
            "legacy ring-overwrite control does not discriminate: differing_attention={attention_bits} differing_logits={legacy_bits}"
        );
        println!(
            "FLINT_SLIDING_RING_PREFILL chunk={} rows=520 ring={ring} interleaved_finite=true legacy_differing_attention={attention_bits} legacy_differing_logits={legacy_bits} PASS",
            ring_engine.chunk
        );
        drop(ring_engine);

        let mut engine = Backend::calibrate(&dir, 0, 600, None, settings.clone())?
            .context("Flint calibration missing")?;

        let (graph_nll, graph_logits, bit_mismatches, max_ulp) =
            graph_eager_forced_prefix(&mut engine, &ids)?;
        let eager_nll = graph_nll;

        engine.reset()?;
        zero_flint_sinks(&mut engine)?;
        let (sinks_off_nll, _, _, _) = forced_prefix(&mut engine, &ids, true, None)?;
        ensure!(
            (sinks_off_nll - graph_nll).abs() > 0.01 && sinks_off_nll - graph_nll >= 0.001,
            "sinks-off control does not discriminate: candidate={graph_nll} control={sinks_off_nll}"
        );
        drop(engine);
        let window_on_tail = prediction_nll(&graph_logits, 512, 599)?;
        let reference_means = reference_nlls
            .iter()
            .map(|values| values[..599].iter().sum::<f64>() / 599.0)
            .collect::<Vec<_>>();
        let reference_tails = reference_nlls
            .iter()
            .map(|values| values[512..599].iter().sum::<f64>() / 87.0)
            .collect::<Vec<_>>();
        let reference_spread = reference_tails
            .iter()
            .flat_map(|a| reference_tails.iter().map(move |b| (a - b).abs()))
            .fold(0f64, f64::max);
        let tail_tolerance = 0.01f64.max(2.0 * reference_spread);
        for (index, &reference_mean) in reference_means.iter().enumerate() {
            ensure!(
                (graph_nll - reference_mean).abs() <= tail_tolerance,
                "graph NLL differs from reference {index}: graph={graph_nll} reference={reference_mean} tolerance={tail_tolerance}"
            );
        }
        for (index, &reference_tail) in reference_tails.iter().enumerate() {
            ensure!(
                (window_on_tail - reference_tail).abs() <= tail_tolerance,
                "window-on tail NLL differs from reference {index}: on={window_on_tail} reference={reference_tail} tolerance={tail_tolerance}"
            );
        }
        ensure!(
            bit_mismatches == 0 && max_ulp == 0,
            "graph/eager NLL differs: graph={graph_nll} eager={eager_nll} bits={bit_mismatches} max_ulp={max_ulp}"
        );
        println!(
            "FLINT_ENGINE_CONTROLS graph_nll={graph_nll:.9} eager_nll={eager_nll:.9} graph_eager_bits=0 sinks_off_nll={sinks_off_nll:.9} tail_on={window_on_tail:.9} reference_spread={reference_spread:.9} tail_tolerance={tail_tolerance:.9} reference_means={reference_means:?} tail_refs={reference_tails:?} PASS"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU window; FF_MODEL, FLYINGFISH_RUNTIME_DIR, FF_FLINT_PASSAGE and FF_FLINT_WINDOW_DUMP required"]
    fn flint_window_boundary_dump_gpu() -> Result<()> {
        use crate::backend::{Backend, Settings};
        use std::{env, fs::File, path::PathBuf};

        let dir = PathBuf::from(env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let runtime =
            PathBuf::from(env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
        let passage = PathBuf::from(env::var_os("FF_FLINT_PASSAGE").context("passage missing")?);
        let output = PathBuf::from(
            env::var_os("FF_FLINT_WINDOW_DUMP").context("FF_FLINT_WINDOW_DUMP missing")?,
        );
        ensure!(
            env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let settings = Settings {
            runtime_dir: Some(runtime),
            prefix_cache: false,
            draft_model: None,
            tree_builder: crate::backend::TreeBuilder::Serial,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let passage: serde_json::Value =
            serde_json::from_reader(File::open(passage).context("open passage")?)?;
        let ids = passage["input_ids"]
            .as_array()
            .context("input_ids missing")?
            .iter()
            .map(|value| value.as_u64().and_then(|value| u32::try_from(value).ok()))
            .collect::<Option<Vec<_>>>()
            .context("invalid passage token id")?;
        ensure!(ids.len() >= 600, "passage is shorter than the gate");
        let mut engine = Backend::calibrate(&dir, 0, 600, None, settings)?
            .context("Flint calibration missing")?;
        let capture_layers = vec![3, 5];
        let capture_positions = vec![513, 560, 599];
        engine.flint_window_capture = Some((
            capture_layers.clone(),
            capture_positions.clone(),
            output.clone(),
        ));
        engine.prefill(&ids[..1], None)?;
        for (offset, token) in ids[1..600].iter().enumerate() {
            let position = offset + 1;
            engine.set_token(*token)?;
            engine.decode(false, None)?;
            if capture_positions == [560] && position == 560 {
                break;
            }
        }
        let expected = capture_layers
            .iter()
            .flat_map(|layer| {
                capture_positions
                    .iter()
                    .map(move |position| (layer, position))
            })
            .map(|(layer, position)| {
                output.join(format!("layer{layer}_position{position}.safetensors"))
            })
            .collect::<Vec<_>>();
        ensure!(
            expected
                .iter()
                .all(|path| path.is_file() && path.metadata().is_ok_and(|meta| meta.len() > 0)),
            "Flint window dump files are incomplete"
        );
        println!(
            "FLINT_WINDOW_DUMP layers={capture_layers:?} positions={capture_positions:?} files={} PASS",
            expected.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU window; FF_MODEL, FLYINGFISH_RUNTIME_DIR and FF_FLINT_PASSAGE required"]
    fn flint_decode_graph_length_gpu() -> Result<()> {
        use crate::backend::{Backend, Settings};
        use std::{env, fs::File, path::PathBuf};

        let dir = PathBuf::from(env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let runtime =
            PathBuf::from(env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?);
        let passage = PathBuf::from(env::var_os("FF_FLINT_PASSAGE").context("passage missing")?);
        ensure!(
            env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let settings = Settings {
            runtime_dir: Some(runtime),
            prefix_cache: false,
            draft_model: None,
            tree_builder: crate::backend::TreeBuilder::Serial,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: false,
            kv_format: crate::kv::KvFormat::Bf16,
        };
        let passage: serde_json::Value =
            serde_json::from_reader(File::open(passage).context("open passage")?)?;
        let ids = passage["input_ids"]
            .as_array()
            .context("input_ids missing")?
            .iter()
            .map(|value| value.as_u64().and_then(|value| u32::try_from(value).ok()))
            .collect::<Option<Vec<_>>>()
            .context("invalid passage token id")?;
        let mut engine = Backend::calibrate(&dir, 0, 1100, None, settings.clone())?
            .context("Flint calibration missing")?;
        let compare = |graph: Vec<Vec<f32>>, eager: &[Vec<f32>]| -> (u64, u32) {
            let mut bits = 0u64;
            let mut ulp = 0u32;
            for (actual_row, expected_row) in graph.iter().zip(eager) {
                bits += actual_row
                    .iter()
                    .zip(expected_row)
                    .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                    .count() as u64;
                if actual_row
                    .iter()
                    .zip(expected_row)
                    .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
                {
                    ulp += 1;
                }
            }
            (bits, ulp)
        };

        let (eager_512, bits_512, ulp_512) =
            graph_eager_window(&mut engine, &ids[..510], &ids[510..530])?;
        ensure!(
            ulp_512 <= 1,
            "graph/eager decode differs across 512/513: bits={bits_512} max_ulp={ulp_512}"
        );
        let (_eager_1024, bits_1024, ulp_1024) =
            graph_eager_window(&mut engine, &ids[..1020], &ids[1020..1040])?;
        ensure!(
            ulp_1024 <= 1,
            "graph/eager decode differs across 1024: bits={bits_1024} max_ulp={ulp_1024}"
        );

        drop(engine);
        let frozen_spec = crate::model_spec::ModelSpec::read(&dir)?;
        let mut frozen_engine = Engine::load_qwen(
            &dir,
            0,
            1100,
            None,
            settings,
            crate::backend::cuda::engine::SetupMode::Prepare,
            frozen_spec,
        )?;
        frozen_engine.flint_frozen_decode_length = true;
        frozen_engine.finish_flint_calibration(&dir, std::time::Instant::now())?;
        let frozen = decode_window(&mut frozen_engine, &ids[..510], &ids[510..530], true)?;
        let (frozen_bits, frozen_ulp) = compare(frozen, &eager_512);
        ensure!(
            frozen_bits > 0 && frozen_ulp > 1,
            "frozen-length negative control was not detected: bits={frozen_bits} max_ulp={frozen_ulp}"
        );
        println!(
            "FLINT_DECODE_GRAPH_LENGTH rows_512=20 bits_512={bits_512} differing_rows_512={ulp_512} rows_1024=20 bits_1024={bits_1024} differing_rows_1024={ulp_1024} frozen_bits={frozen_bits} frozen_differing_rows={frozen_ulp} PASS"
        );
        Ok(())
    }
}
