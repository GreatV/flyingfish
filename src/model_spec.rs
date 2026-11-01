//! Normalized model geometry. Device execution remains gated by backend capability.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Qwen35Text,
    MiniCpmV47,
    Flint,
    Glm5Next,
    LoopVl,
    DeepseekV41,
    Gemma4,
    EmbeddingGemma2,
}
impl Architecture {
    /// Architectures with a bound MTP decode path.
    pub fn mtp_decode(self) -> bool {
        self == Self::Qwen35Text
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Mixer {
    FullAttention,
    Gdn,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MoeConfig {
    pub experts: u32,
    pub experts_per_tok: u32,
    pub expert_intermediate: u32,
    pub shared_intermediate: u32,
    #[serde(
        default = "RouterScore::softmax",
        skip_serializing_if = "RouterScore::is_softmax"
    )]
    pub score: RouterScore,
    pub shared_gate: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RouterScore {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}
impl RouterScore {
    fn softmax() -> Self {
        Self::Softmax
    }
    fn is_softmax(&self) -> bool {
        *self == Self::Softmax
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwigluRounding {
    Twice,
    Once,
    ClampedTwice,
    ClampedOnce,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedWeightBoundary {
    AfterDownBf16,
    BeforeContributionBf16,
    BeforeDownQuantization,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LinearFormat {
    Bf16,
    Fp8E4m3Block {
        block: [u32; 2],
        scale_dtype: safetensors::Dtype,
    },
    /// compressed-tensors FP8_DYNAMIC per-output-channel weights plus token activations.
    Fp8E4m3Channel,
}
/// Routed-expert checkpoint layout: fused or separate BF16 tensors, per-channel BF16 scales,
/// block scales beside per-expert E4M3 weights, or block scales beside per-expert E2M1 weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExpertFormat {
    Bf16,
    Bf16Separate,
    Fp8W8a16,
    Fp8W8a16Block {
        block: [u32; 2],
        scale_dtype: safetensors::Dtype,
    },
    /// Per-expert E2M1 weights packed two values per byte, with block scales beside them.
    Fp4E2m1Block {
        block: [u32; 2],
        scale_dtype: safetensors::Dtype,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GdnConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextConfig {
    pub dense_layers: u32,
    pub nextn_layers: u32,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub v_head_dim: u32,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
    pub gate_lower_bound: f32,
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeepseekSelectionShape {
    pub index_topk: u32,
    pub candidate_topk_blocks: u32,
    pub candidate_block_size: u32,
}

impl DeepseekSelectionShape {
    pub(crate) const IMPLEMENTED: Self = Self {
        index_topk: 512,
        candidate_topk_blocks: 2048,
        candidate_block_size: 8,
    };
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepseekV41Config {
    pub num_hidden_layers: u32,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub window_size: u32,
    pub qk_rope_head_dim: u32,
    pub q_lora_rank: u32,
    pub o_lora_rank: u32,
    pub o_groups: u32,
    /// Per-layer compress ratio; the nonzero kinds rotate with the YaRN table
    /// built from compress_rope_theta, the zero kinds with plain rope_theta.
    pub compress_ratios: Vec<u8>,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub original_max_position: u32,
    pub beta_fast: u32,
    pub beta_slow: u32,
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
    pub index_heads: u32,
    pub index_head_dim: u32,
    pub selection: DeepseekSelectionShape,
    pub kv_source_layer_ids: Vec<u32>,
    pub index_source_layer_ids: Vec<u32>,
    pub candidate_source_layer_id: u32,
    pub engram_layer_ids: Vec<u32>,
    pub engram_num_embeddings: Vec<u32>,
    pub engram_max_ngram_size: u32,
    pub engram_n_heads: u32,
    pub engram_head_dim: u32,
    pub nextn_layers: u32,
    pub dspark_experts: u32,
    pub dspark_markov_rank: u32,
    pub vision_layers: u32,
    pub vision_hidden: u32,
    pub vision_intermediate: u32,
    pub vision_patch: u32,
    pub vision_downsample_ratio: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerPlan {
    pub mixer: Mixer,
    pub state_index: u32,
    pub rope: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u32>,
    pub ffn: FfnKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<LayerGeometry>,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f32,
    pub pairs: u32,
    pub pair_stride: u32,
    pub denominator: u32,
}
impl RopeSpec {
    /// `[capacity, head_dim]` rows of `cos[0..head_dim/2] | sin[0..head_dim/2]`; frequencies
    /// `theta^(-2i/denominator)` for the first `pairs` indices and zero (cos 1, sin 0) for the rest,
    /// evaluated in F32; element `i` pairs with `i + head_dim/2`.
    pub fn table(&self, head_dim: u32, capacity: usize) -> Result<Vec<half::bf16>> {
        let half = head_dim / 2;
        ensure!(
            head_dim.is_multiple_of(2) && self.pair_stride == half && self.pairs <= half,
            "rope spec {self:?} does not fit head_dim {head_dim}"
        );
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                if i < self.pairs {
                    1.0 / self.theta.powf((2 * i) as f32 / self.denominator as f32)
                } else {
                    0.0
                }
            })
            .collect();
        let mut table = Vec::with_capacity(capacity * head_dim as usize);
        for position in 0..capacity {
            let position = position as f32;
            table.extend(
                inv.iter()
                    .map(|f| half::bf16::from_f32((position * f).cos())),
            );
            table.extend(
                inv.iter()
                    .map(|f| half::bf16::from_f32((position * f).sin())),
            );
        }
        Ok(table)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerGeometry {
    pub head_dim: u32,
    pub intermediate: u32,
    pub rope: RopeSpec,
    pub kv_reader: bool,
    /// Per-layer KV head count; absent means the model-wide `kv_heads`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<u32>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FfnKind {
    Dense,
    Moe,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub abi_version: u32,
    pub architecture: Architecture,
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_width: u32,
    pub max_position: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub linear_format: LinearFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_format: Option<ExpertFormat>,
    pub eos: Vec<u32>,
    pub layers: Vec<LayerPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdn: Option<GdnConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glm5_next: Option<Glm5NextConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deepseek_v41: Option<DeepseekV41Config>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_softcap: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_input: Option<u32>,
    #[serde(default)]
    pub vision: Option<VisionConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioConfig {
    pub hidden: u32,
    pub layers: u32,
    pub heads: u32,
    pub output: u32,
    pub chunk: u32,
    pub context_left: u32,
    pub context_right: u32,
    pub conv_kernel: u32,
    pub subsample_channels: [u32; 2],
    pub rms_eps: f32,
    pub logit_cap: f32,
    pub invalid_logit: f32,
    pub gradient_clip: f32,
    pub residual_weight: f32,
    pub token: u32,
    pub start_token: u32,
    pub end_token: u32,
}

impl AudioConfig {
    fn from_root(root: &Value) -> Result<Option<Self>> {
        let Some(a) = root.get("audio_config").filter(|a| !a.is_null()) else {
            return Ok(None);
        };
        exact_keys(
            a,
            &[
                "_name_or_path",
                "architectures",
                "attention_chunk_size",
                "attention_context_left",
                "attention_context_right",
                "attention_invalid_logits_value",
                "attention_logit_cap",
                "chunk_size_feed_forward",
                "conv_kernel_size",
                "dtype",
                "gradient_clipping",
                "hidden_act",
                "hidden_size",
                "id2label",
                "initializer_range",
                "is_encoder_decoder",
                "label2id",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "output_attentions",
                "output_hidden_states",
                "output_proj_dims",
                "problem_type",
                "residual_weight",
                "return_dict",
                "rms_norm_eps",
                "subsampling_conv_channels",
                "use_clipped_linears",
            ],
            "audio config",
        )?;
        for (key, value) in [
            ("model_type", "gemma4_audio".into()),
            ("hidden_size", 1024.into()),
            ("num_hidden_layers", 12.into()),
            ("num_attention_heads", 8.into()),
            ("output_proj_dims", 1536.into()),
            ("attention_chunk_size", 12.into()),
            ("attention_context_left", 13.into()),
            ("attention_context_right", 0.into()),
            ("conv_kernel_size", 5.into()),
            ("subsampling_conv_channels", serde_json::json!([128, 32])),
            ("rms_norm_eps", 1e-6.into()),
            ("attention_logit_cap", 50.0.into()),
            ("attention_invalid_logits_value", (-1e9).into()),
            ("gradient_clipping", 1e10.into()),
            ("residual_weight", 0.5.into()),
            ("hidden_act", "silu".into()),
            ("use_clipped_linears", true.into()),
            ("dtype", "bfloat16".into()),
            ("chunk_size_feed_forward", 0.into()),
        ] {
            equal(a, key, value)?;
        }
        let embedding = root.get("model_type").and_then(Value::as_str) == Some("embedding_gemma2");
        if embedding {
            ensure!(
                root.get("eoa_token_id").is_none(),
                "unexpected audio end token alias"
            );
        } else {
            ensure!(
                root["eoa_token_id"] == root["eoa_token_index"],
                "audio end token mismatch"
            );
        }
        let spec = Self {
            hidden: 1024,
            layers: 12,
            heads: 8,
            output: 1536,
            chunk: 12,
            context_left: 13,
            context_right: 0,
            conv_kernel: 5,
            subsample_channels: [128, 32],
            rms_eps: 1e-6,
            logit_cap: 50.0,
            invalid_logit: -1e9,
            gradient_clip: 1e10,
            residual_weight: 0.5,
            token: positive(root, "audio_token_id")?,
            start_token: positive(root, "boa_token_id")?,
            end_token: if embedding {
                positive(root, "eoa_token_index")?
            } else {
                positive(root, "eoa_token_id")?
            },
        };
        ensure!(
            spec.token == 258881 && spec.start_token == 256000 && spec.end_token == 258883,
            "unsupported audio token identities"
        );
        Ok(Some(spec))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiniCpmVision {
    pub image_size: u32,
    pub insert_layer: u32,
    pub slice_start_token: u32,
    pub slice_end_token: u32,
    pub newline_token: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxialVision {
    pub head_dim: u32,
    pub soft_tokens: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub standardize: bool,
    pub clipped_linears: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedVision {
    pub soft_tokens: u32,
    pub rms_eps: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub channels: u32,
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub output: u32,
    pub position_side: u32,
    pub image_token: u32,
    pub video_token: u32,
    pub start_token: u32,
    pub end_token: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minicpm: Option<MiniCpmVision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axial: Option<AxialVision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unified: Option<UnifiedVision>,
}

impl VisionConfig {
    pub fn validate(&self, architecture: Architecture) -> Result<()> {
        match architecture {
            Architecture::Qwen35Text => {
                ensure!(
                    [
                        self.depth,
                        self.hidden,
                        self.heads,
                        self.intermediate,
                        self.channels,
                        self.patch,
                        self.temporal_patch,
                        self.merge,
                        self.output,
                        self.position_side
                    ] == [27, 1152, 16, 4304, 3, 16, 2, 2, 5120, 48],
                    "unsupported Q38 vision geometry"
                );
                ensure!(
                    [
                        self.image_token,
                        self.video_token,
                        self.start_token,
                        self.end_token
                    ] == [248056, 248057, 248053, 248054],
                    "unsupported Q38 image token IDs"
                );
                ensure!(self.minicpm.is_none(), "unexpected MiniCPM vision fields");
                ensure!(self.axial.is_none(), "unexpected axial vision fields");
            }
            Architecture::MiniCpmV47 => {
                ensure!(self.axial.is_none(), "unexpected axial vision fields");
                ensure!(
                    [
                        self.depth,
                        self.hidden,
                        self.heads,
                        self.intermediate,
                        self.channels,
                        self.patch,
                        self.temporal_patch,
                        self.merge,
                        self.output,
                        self.position_side
                    ] == [27, 1152, 16, 4304, 3, 14, 1, 4, 2048, 70],
                    "unsupported MiniCPM vision geometry"
                );
                ensure!(
                    [
                        self.image_token,
                        self.video_token,
                        self.start_token,
                        self.end_token
                    ] == [248056, 248057, 248078, 248079],
                    "unsupported MiniCPM image token IDs"
                );
                let v = self
                    .minicpm
                    .as_ref()
                    .context("MiniCPM vision configuration missing")?;
                ensure!(
                    v.image_size == 980,
                    "unsupported image_size: {}",
                    v.image_size
                );
                ensure!(
                    v.insert_layer == 6,
                    "unsupported insert_layer_id: {}",
                    v.insert_layer
                );
                ensure!(
                    v.slice_start_token == 248088,
                    "unsupported slice_start_id: {}",
                    v.slice_start_token
                );
                ensure!(
                    v.slice_end_token == 248089,
                    "unsupported slice_end_id: {}",
                    v.slice_end_token
                );
                ensure!(
                    v.newline_token == 198,
                    "unsupported newline_id: {}",
                    v.newline_token
                );
            }
            Architecture::Gemma4 | Architecture::EmbeddingGemma2 => {
                if let Some(unified) = &self.unified {
                    ensure!(
                        self.depth == 0
                            && self.heads == 0
                            && self.intermediate == 0
                            && self.hidden == 3840
                            && self.output == self.hidden
                            && self.channels == 3
                            && self.patch == 48
                            && self.merge == 1
                            && self.temporal_patch == 1
                            && self.position_side == 1120
                            && unified.soft_tokens == 280
                            && unified.rms_eps == 1e-6
                            && self.axial.is_none()
                            && self.minicpm.is_none()
                            && [self.image_token, self.start_token, self.end_token]
                                == [258880, 255999, 258882],
                        "unsupported unified vision geometry"
                    );
                    return Ok(());
                }
                let axial = self
                    .axial
                    .as_ref()
                    .context("axial vision configuration missing")?;
                ensure!(
                    self.depth > 0
                        && self.hidden > 0
                        && self.heads > 0
                        && self.intermediate > 0
                        && self.output > 0
                        && axial.head_dim > 0
                        && axial.head_dim.is_multiple_of(4)
                        && self.heads.checked_mul(axial.head_dim) == Some(self.hidden)
                        && self.channels == 3
                        && self.patch == 16
                        && self.temporal_patch == 1
                        && self.merge == 3
                        && self.position_side > 0
                        && axial.soft_tokens > 0
                        && axial.soft_tokens.checked_mul(9).is_some()
                        && axial.rms_eps.is_finite()
                        && axial.rms_eps > 0.0
                        && axial.rope_theta.is_finite()
                        && axial.rope_theta > 0.0
                        && self.minicpm.is_none(),
                    "unsupported axial vision geometry"
                );
                ensure!(
                    [self.image_token, self.start_token, self.end_token]
                        == [258880, 255999, 258882],
                    "unsupported image token IDs"
                );
            }
            Architecture::Llama
            | Architecture::Flint
            | Architecture::Glm5Next
            | Architecture::LoopVl => {
                bail!("vision configuration requires a supported multimodal architecture")
            }
            Architecture::DeepseekV41 => {
                bail!("vision configuration requires a supported multimodal architecture")
            }
        }
        Ok(())
    }

    pub fn patch_elements(&self, architecture: Architecture) -> Result<usize> {
        self.validate(architecture)?;
        Ok((self.channels * self.temporal_patch * self.patch * self.patch) as usize)
    }

    pub fn from_root(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config") else {
            return Ok(None);
        };
        match v.get("model_type").and_then(Value::as_str) {
            Some("qwen3_5") => Self::qwen_from_root(root, v).map(Some),
            Some("minicpmv4_7_vision") => Self::minicpm_from_root(root, v).map(Some),
            Some("gemma4_vision") => Self::axial_from_root(root, v).map(Some),
            Some("gemma4_unified_vision") => Self::unified_from_root(root, v).map(Some),
            other => bail!("unsupported vision_config model_type: {other:?}"),
        }
    }

    fn unified_from_root(root: &Value, v: &Value) -> Result<Self> {
        for (key, value) in [
            ("mm_embed_dim", 3840.into()),
            ("mm_posemb_size", 1120.into()),
            ("model_patch_size", 48.into()),
            ("num_soft_tokens", 280.into()),
            ("output_proj_dims", 3840.into()),
            ("patch_size", 16.into()),
            ("pooling_kernel_size", 3.into()),
            ("rms_norm_eps", 1e-6.into()),
        ] {
            equal(v, key, value)?;
        }
        let spec = Self {
            depth: 0,
            hidden: positive(v, "mm_embed_dim")?,
            heads: 0,
            intermediate: 0,
            channels: 3,
            patch: positive(v, "model_patch_size")?,
            temporal_patch: 1,
            merge: 1,
            output: positive(
                root.get("text_config").context("text_config missing")?,
                "hidden_size",
            )?,
            position_side: positive(v, "mm_posemb_size")?,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "boi_token_id")?,
            end_token: integer(root, "eoi_token_id")?,
            minicpm: None,
            axial: None,
            unified: Some(UnifiedVision {
                soft_tokens: positive(v, "num_soft_tokens")?,
                rms_eps: 1e-6,
            }),
        };
        spec.validate(Architecture::Gemma4)?;
        Ok(spec)
    }

    fn axial_from_root(root: &Value, v: &Value) -> Result<Self> {
        if root.get("video_token_id").is_some() {
            equal(root, "video_token_id", 258884.into())?;
        }
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("hidden_activation", "gelu_pytorch_tanh".into()),
            ("dtype", "bfloat16".into()),
        ] {
            equal(v, key, value)?;
        }
        let head_dim = positive(v, "head_dim")?;
        let heads = positive(v, "num_attention_heads")?;
        ensure!(
            integer(v, "global_head_dim")? == head_dim
                && integer(v, "num_key_value_heads")? == heads,
            "axial vision requires equal query/key/value head geometry"
        );
        let rope = v
            .get("rope_parameters")
            .context("vision rope_parameters missing")?;
        equal(
            rope,
            "rope_type",
            if root.get("model_type").and_then(Value::as_str) == Some("embedding_gemma2") {
                "axial".into()
            } else {
                "default".into()
            },
        )?;
        let number = |object: &Value, key: &str| -> Result<f32> {
            Ok(object
                .get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing/invalid vision {key}"))? as f32)
        };
        let flag = |key: &str| -> Result<bool> {
            v.get(key)
                .and_then(Value::as_bool)
                .with_context(|| format!("missing/invalid vision {key}"))
        };
        let soft_tokens = positive(root, "vision_soft_tokens_per_image")?;
        ensure!(
            integer(v, "default_output_length")? == soft_tokens,
            "vision output token budgets differ"
        );
        let spec = Self {
            depth: positive(v, "num_hidden_layers")?,
            hidden: positive(v, "hidden_size")?,
            heads,
            intermediate: positive(v, "intermediate_size")?,
            channels: 3,
            patch: positive(v, "patch_size")?,
            temporal_patch: 1,
            merge: positive(v, "pooling_kernel_size")?,
            output: positive(
                root.get("text_config").context("text_config missing")?,
                "hidden_size",
            )?,
            position_side: positive(v, "position_embedding_size")?,
            image_token: integer(root, "image_token_id")?,
            video_token: 258884,
            start_token: integer(root, "boi_token_id")?,
            end_token: integer(root, "eoi_token_id")?,
            minicpm: None,
            axial: Some(AxialVision {
                head_dim,
                soft_tokens,
                rms_eps: number(v, "rms_norm_eps")?,
                rope_theta: number(rope, "rope_theta")?,
                standardize: flag("standardize")?,
                clipped_linears: flag("use_clipped_linears")?,
            }),
            unified: None,
        };
        spec.validate(Architecture::Gemma4)?;
        Ok(spec)
    }

    fn qwen_from_root(root: &Value, v: &Value) -> Result<Self> {
        equal(v, "hidden_act", "gelu_pytorch_tanh".into())?;
        equal(v, "deepstack_visual_indexes", serde_json::json!([]))?;
        ensure!(
            integer(v, "num_position_embeddings")? == 2304,
            "unsupported vision position table"
        );
        let spec = Self {
            depth: integer(v, "depth")?,
            hidden: integer(v, "hidden_size")?,
            heads: integer(v, "num_heads")?,
            intermediate: integer(v, "intermediate_size")?,
            channels: integer(v, "in_channels")?,
            patch: integer(v, "patch_size")?,
            temporal_patch: integer(v, "temporal_patch_size")?,
            merge: integer(v, "spatial_merge_size")?,
            output: integer(v, "out_hidden_size")?,
            position_side: 48,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "vision_start_token_id")?,
            end_token: integer(root, "vision_end_token_id")?,
            minicpm: None,
            axial: None,
            unified: None,
        };
        spec.validate(Architecture::Qwen35Text)?;
        Ok(spec)
    }

    fn minicpm_from_root(root: &Value, v: &Value) -> Result<Self> {
        equal(v, "hidden_act", "gelu_pytorch_tanh".into())?;
        if let Some(kernel) = v.get("window_kernel_size") {
            ensure!(
                kernel == &serde_json::json!([2, 2]),
                "unsupported window_kernel_size: {kernel:?}"
            );
        }
        let image_size = integer(v, "image_size")?;
        let patch = integer(v, "patch_size")?;
        ensure!(
            patch > 0 && image_size.is_multiple_of(patch),
            "image_size is not a patch multiple"
        );
        // MiniCPMV4_7Config defaults: merge_kernel_size (2, 2), merger_times 1.
        if let Some(kernel) = root.get("merge_kernel_size") {
            ensure!(
                kernel == &serde_json::json!([2, 2]),
                "unsupported merge_kernel_size: {kernel:?}"
            );
        }
        if let Some(times) = root.get("merger_times") {
            ensure!(
                integer(root, "merger_times")? == 1,
                "unsupported merger_times: {times:?}"
            );
        }
        let merge = match root.get("downsample_mode").and_then(Value::as_str) {
            Some("16x") => 4,
            other => bail!("unsupported downsample_mode: {other:?}"),
        };
        let spec = Self {
            depth: integer(v, "num_hidden_layers")?,
            hidden: integer(v, "hidden_size")?,
            heads: integer(v, "num_attention_heads")?,
            intermediate: integer(v, "intermediate_size")?,
            channels: integer(v, "num_channels")?,
            patch,
            temporal_patch: 1,
            merge,
            output: u32::try_from(
                root.get("text_config")
                    .and_then(|t| t.get("hidden_size"))
                    .and_then(Value::as_u64)
                    .context("missing text_config.hidden_size")?,
            )
            .context("hidden_size overflows u32")?,
            position_side: image_size / patch,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "image_start_id")?,
            end_token: integer(root, "image_end_id")?,
            minicpm: Some(MiniCpmVision {
                image_size,
                insert_layer: integer(root, "insert_layer_id")?,
                slice_start_token: integer(root, "slice_start_id")?,
                slice_end_token: integer(root, "slice_end_id")?,
                newline_token: integer(root, "newline_id")?,
            }),
            axial: None,
            unified: None,
        };
        spec.validate(Architecture::MiniCpmV47)?;
        Ok(spec)
    }
}
fn integer(v: &Value, key: &str) -> Result<u32> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing/invalid {key}"))?;
    u32::try_from(n).with_context(|| format!("{key} overflows u32"))
}
fn positive(v: &Value, key: &str) -> Result<u32> {
    let n = integer(v, key)?;
    ensure!(n > 0, "unsupported {key}: 0");
    Ok(n)
}
fn u32_list(v: &Value, key: &str) -> Result<Vec<u32>> {
    let values = v
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing/invalid {key}"))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let n = value
            .as_u64()
            .with_context(|| format!("{key} contains a non-unsigned integer"))?;
        out.push(u32::try_from(n).with_context(|| format!("{key} overflows u32"))?);
    }
    Ok(out)
}
fn block_size(v: &Value, key: &str) -> Result<[u32; 2]> {
    let values = v
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing/invalid {key}"))?;
    ensure!(values.len() == 2, "{key} is not a two-element block size");
    let mut block = [0u32; 2];
    for (index, value) in values.iter().enumerate() {
        let n = value
            .as_u64()
            .with_context(|| format!("{key} contains a non-unsigned integer"))?;
        block[index] = u32::try_from(n).with_context(|| format!("{key} overflows u32"))?;
    }
    ensure!(
        block.iter().all(|&n| n > 0),
        "{key} contains a zero block extent"
    );
    Ok(block)
}
fn integer_array(v: &Value, key: &str, limit: u32) -> Result<Vec<u32>> {
    let values = v
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("missing/invalid {key}"))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let n = value
            .as_u64()
            .with_context(|| format!("{key} contains a non-unsigned integer"))?;
        let n = u32::try_from(n).with_context(|| format!("{key} overflows u32"))?;
        ensure!(n < limit, "{key} contains out-of-range layer {n}");
        out.push(n);
    }
    Ok(out)
}
fn equal(v: &Value, key: &str, expected: Value) -> Result<()> {
    ensure!(
        v.get(key) == Some(&expected),
        "unsupported {key}: {:?}",
        v.get(key)
    );
    Ok(())
}
fn exact_keys(v: &Value, keys: &[&str], label: &str) -> Result<()> {
    let object = v
        .as_object()
        .with_context(|| format!("{label} is not an object"))?;
    if let Some(key) = object.keys().find(|key| !keys.contains(&key.as_str())) {
        bail!("unknown {label} key {key}");
    }
    for key in keys {
        ensure!(object.contains_key(*key), "{label} missing key {key}");
    }
    Ok(())
}
impl LinearFormat {
    pub fn fp8_block(block: [u32; 2], scale_dtype: safetensors::Dtype) -> Result<Self> {
        crate::fp8::Fp8BlockScale::new(block, scale_dtype)?;
        Ok(Self::Fp8E4m3Block { block, scale_dtype })
    }

    fn read(root: &Value) -> Result<Self> {
        let Some(quant) = root.get("quantization_config") else {
            return Ok(Self::Bf16);
        };
        for (key, val) in [
            ("quant_method", "fp8"),
            ("fmt", "e4m3"),
            ("activation_scheme", "dynamic"),
        ] {
            equal(quant, key, val.into())?;
        }
        equal(quant, "weight_block_size", serde_json::json!([128, 128]))?;
        Self::fp8_block([128, 128], safetensors::Dtype::BF16)
    }
}
impl ExpertFormat {
    /// Parses the published compressed-tensors FP8_DYNAMIC form; a partial or unknown config errors.
    fn read(root: &Value) -> Result<Self> {
        let Some(quant) = root
            .get("quantization_config")
            .filter(|value| !value.is_null())
        else {
            return Ok(Self::Bf16);
        };
        exact_keys(
            quant,
            &[
                "config_groups",
                "format",
                "global_compression_ratio",
                "ignore",
                "kv_cache_scheme",
                "quant_method",
                "quantization_status",
                "sparsity_config",
                "transform_config",
                "version",
            ],
            "quantization_config",
        )?;
        for (key, value) in [
            ("quant_method", "compressed-tensors"),
            ("format", "float-quantized"),
            ("quantization_status", "compressed"),
        ] {
            equal(quant, key, value.into())?;
        }
        for key in ["global_compression_ratio", "kv_cache_scheme"] {
            ensure!(
                quant.get(key).is_none_or(Value::is_null),
                "unsupported quantization {key}: {:?}",
                quant.get(key)
            );
        }
        for key in ["sparsity_config", "transform_config"] {
            ensure!(
                quant
                    .get(key)
                    .is_none_or(|value| value == &serde_json::json!({})),
                "unsupported quantization {key}: {:?}",
                quant.get(key)
            );
        }
        ensure!(
            quant
                .get("version")
                .and_then(Value::as_str)
                .is_some_and(|version| !version.is_empty()),
            "quantization version missing"
        );
        ensure!(
            quant
                .get("ignore")
                .and_then(Value::as_array)
                .is_some_and(|ignore| !ignore.is_empty()),
            "quantization ignore list missing"
        );
        let groups = quant
            .get("config_groups")
            .and_then(Value::as_object)
            .context("quantization config_groups missing")?;
        ensure!(
            groups.len() == 1,
            "quantization expects one config group, got {}",
            groups.len()
        );
        let group = groups.values().next().context("empty quantization group")?;
        exact_keys(
            group,
            &[
                "format",
                "input_activations",
                "output_activations",
                "targets",
                "weights",
            ],
            "quantization config group",
        )?;
        equal(group, "format", "float-quantized".into())?;
        ensure!(
            group.get("output_activations").is_none_or(Value::is_null),
            "unsupported quantization output_activations"
        );
        equal(group, "targets", serde_json::json!(["Linear"]))?;
        let weights = group
            .get("weights")
            .context("quantization weights missing")?;
        exact_keys(
            weights,
            &[
                "actorder",
                "block_structure",
                "dynamic",
                "group_size",
                "num_bits",
                "observer",
                "observer_kwargs",
                "scale_dtype",
                "strategy",
                "symmetric",
                "type",
                "zp_dtype",
            ],
            "quantization weights",
        )?;
        equal(weights, "num_bits", 8.into())?;
        equal(weights, "type", "float".into())?;
        equal(weights, "strategy", "channel".into())?;
        equal(weights, "symmetric", true.into())?;
        equal(weights, "dynamic", false.into())?;
        equal(weights, "observer", "memoryless_minmax".into())?;
        for key in [
            "actorder",
            "block_structure",
            "group_size",
            "scale_dtype",
            "zp_dtype",
        ] {
            ensure!(
                weights.get(key).is_none_or(Value::is_null),
                "unsupported quantization weights {key}"
            );
        }
        ensure!(
            weights
                .get("observer_kwargs")
                .is_none_or(|value| value == &serde_json::json!({})),
            "unsupported quantization weights observer_kwargs"
        );
        let activations = group
            .get("input_activations")
            .context("quantization input_activations missing")?;
        exact_keys(
            activations,
            &[
                "actorder",
                "block_structure",
                "dynamic",
                "group_size",
                "num_bits",
                "observer",
                "observer_kwargs",
                "scale_dtype",
                "strategy",
                "symmetric",
                "type",
                "zp_dtype",
            ],
            "quantization input_activations",
        )?;
        equal(activations, "num_bits", 8.into())?;
        equal(activations, "type", "float".into())?;
        equal(activations, "strategy", "token".into())?;
        equal(activations, "symmetric", true.into())?;
        equal(activations, "dynamic", true.into())?;
        for key in [
            "actorder",
            "block_structure",
            "group_size",
            "observer",
            "scale_dtype",
            "zp_dtype",
        ] {
            ensure!(
                activations.get(key).is_none_or(Value::is_null),
                "unsupported quantization input_activations {key}"
            );
        }
        ensure!(
            activations
                .get("observer_kwargs")
                .is_none_or(|value| value == &serde_json::json!({})),
            "unsupported quantization input_activations observer_kwargs"
        );
        Ok(Self::Fp8W8a16)
    }

    /// The published ignore recipe: every tensor that stays in its stored dtype.
    fn expected_ignore(spec: &ModelSpec) -> Result<Vec<String>> {
        let vision = spec
            .vision
            .as_ref()
            .context("quantized checkpoints need the vision tower")?;
        let mut names = vec!["lm_head".to_owned()];
        for (layer, plan) in spec.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}");
            if plan.mixer == Mixer::Gdn {
                names.push(format!("{prefix}.linear_attn"));
                for child in [
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "norm",
                    "out_proj",
                ] {
                    names.push(format!("{prefix}.linear_attn.{child}"));
                }
            }
            for suffix in [
                "mlp.gate",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert_gate",
            ] {
                names.push(format!("{prefix}.{suffix}"));
            }
        }
        names.push("model.merger.mlp.0.linear_1".to_owned());
        names.push("model.merger.mlp.0.linear_2".to_owned());
        for suffix in [
            "linear_1",
            "linear_2",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ] {
            names.push(format!("model.vision_tower.vit_merger.{suffix}"));
        }
        for layer in 0..vision.depth {
            for suffix in [
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
            ] {
                names.push(format!(
                    "model.vision_tower.encoder.layers.{layer}.{suffix}"
                ));
            }
        }
        names.sort_unstable();
        Ok(names)
    }

    /// Requires the published ignore list exactly, so an unexpected quantized role errors.
    fn check_ignore(&self, root: &Value, spec: &ModelSpec) -> Result<()> {
        if *self == Self::Bf16 {
            return Ok(());
        }
        let quant = root
            .get("quantization_config")
            .context("quantization_config missing")?;
        let ignore = quant
            .get("ignore")
            .and_then(Value::as_array)
            .context("quantization ignore missing")?;
        let mut actual = Vec::with_capacity(ignore.len());
        for entry in ignore {
            actual.push(
                entry
                    .as_str()
                    .context("quantization ignore entry is not a string")?
                    .to_owned(),
            );
        }
        actual.sort_unstable();
        let expected = Self::expected_ignore(spec)?;
        if let Some(entry) = expected
            .iter()
            .find(|entry| actual.binary_search(entry).is_err())
        {
            bail!("quantization ignore missing {entry}");
        }
        if let Some(entry) = actual
            .iter()
            .find(|entry| expected.binary_search(entry).is_err())
        {
            bail!("quantization ignore has unexpected entry {entry}");
        }
        ensure!(
            actual.len() == expected.len(),
            "quantization ignore repeats an entry"
        );
        Ok(())
    }
}
fn eos(v: &Value, vocab: u32) -> Result<Vec<u32>> {
    let values = match v {
        Value::Array(a) => a.clone(),
        Value::Number(_) => vec![v.clone()],
        _ => bail!("EOS must be an integer or array"),
    };
    let mut out = Vec::new();
    for value in values {
        let n = value.as_u64().context("EOS must be unsigned integer")?;
        ensure!(n < u64::from(vocab), "EOS outside vocabulary");
        out.push(n as u32);
    }
    out.sort_unstable();
    out.dedup();
    ensure!(!out.is_empty(), "empty EOS set");
    Ok(out)
}

struct TextGeometry {
    hidden: u32,
    intermediate: u32,
    vocab: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    q_width: u32,
    max_position: u32,
    rms_eps: f32,
    rope_theta: f32,
    rotary_dim: u32,
    layers: Vec<LayerPlan>,
    moe: Option<MoeConfig>,
    gdn: GdnConfig,
}

impl TextGeometry {
    fn read(t: &Value, model_type: &str) -> Result<Self> {
        equal(t, "model_type", model_type.into())?;
        if let Some(dtype) = t.get("dtype") {
            ensure!(
                dtype == &Value::String("bfloat16".into()),
                "unsupported dtype: {dtype:?}"
            );
        }
        for (key, val) in [("hidden_act", "silu"), ("mamba_ssm_dtype", "float32")] {
            equal(t, key, val.into())?;
        }
        if let Some(gate) = t.get("output_gate_type") {
            ensure!(
                gate == &Value::String("swish".into()),
                "unsupported output_gate_type: {gate:?}"
            );
        }
        equal(t, "attention_bias", false.into())?;
        equal(t, "attn_output_gate", true.into())?;
        equal(t, "attention_dropout", 0.0.into())?;
        equal(t, "tie_word_embeddings", false.into())?;
        if let Some(bias) = t.get("mlp_bias") {
            ensure!(
                bias == &Value::Bool(false),
                "unsupported mlp_bias: {bias:?}"
            );
        }
        let hidden = positive(t, "hidden_size")?;
        let vocab = positive(t, "vocab_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let head_dim = positive(t, "head_dim")?;
        let max_position = positive(t, "max_position_embeddings")?;
        ensure!(
            q_heads.is_multiple_of(kv_heads),
            "num_attention_heads is not a multiple of num_key_value_heads"
        );
        let interval = positive(t, "full_attention_interval")?;
        let eps = t
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .context("missing/invalid rms_norm_eps")?;
        ensure!(
            eps.is_finite() && eps > 0.0,
            "unsupported rms_norm_eps: {eps}"
        );
        let rope = t
            .get("rope_parameters")
            .context("missing rope_parameters")?;
        equal(rope, "rope_type", "default".into())?;
        equal(rope, "mrope_interleaved", true.into())?;
        equal(rope, "mrope_section", serde_json::json!([11, 11, 10]))?;
        let theta = positive(rope, "rope_theta")?;
        let top_partial = t.get("partial_rotary_factor");
        let rope_partial = rope.get("partial_rotary_factor");
        if let (Some(a), Some(b)) = (top_partial, rope_partial) {
            ensure!(
                a == b,
                "partial_rotary_factor differs between text_config and rope_parameters"
            );
        }
        let partial = top_partial
            .or(rope_partial)
            .and_then(Value::as_f64)
            .context("missing partial_rotary_factor")?;
        ensure!(
            partial > 0.0 && partial <= 1.0,
            "unsupported partial_rotary_factor: {partial}"
        );
        let rotary = f64::from(head_dim) * partial;
        ensure!(
            rotary.fract() == 0.0 && rotary >= 1.0,
            "partial_rotary_factor does not yield an integer rotary_dim: {partial}"
        );
        let types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        ensure!(
            types.len() == depth as usize,
            "layer_types length differs from num_hidden_layers"
        );
        let mut counts = [0u32, 0u32];
        let mut layers = Vec::with_capacity(types.len());
        for (i, kind) in types.iter().enumerate() {
            let full = (i as u32 + 1).is_multiple_of(interval);
            let (mixer, slot, expected) = if full {
                (Mixer::FullAttention, 0, "full_attention")
            } else {
                (Mixer::Gdn, 1, "linear_attention")
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported mixer at layer {i}: {kind:?}"
            );
            layers.push(LayerPlan {
                geometry: None,
                mixer,
                state_index: counts[slot],
                rope: full,
                sliding_window: None,
                ffn: if model_type == "qwen3_5_moe_text" {
                    FfnKind::Moe
                } else {
                    FfnKind::Dense
                },
            });
            counts[slot] += 1;
        }
        let gdn = GdnConfig {
            key_heads: positive(t, "linear_num_key_heads")?,
            value_heads: positive(t, "linear_num_value_heads")?,
            key_head_dim: positive(t, "linear_key_head_dim")?,
            value_head_dim: positive(t, "linear_value_head_dim")?,
            conv_kernel: positive(t, "linear_conv_kernel_dim")?,
        };
        ensure!(
            gdn.value_heads.is_multiple_of(gdn.key_heads),
            "linear_num_value_heads is not a multiple of linear_num_key_heads"
        );
        let moe = if model_type == "qwen3_5_moe_text" {
            equal(t, "output_router_logits", false.into())?;
            ensure!(
                t.get("intermediate_size").is_none(),
                "unexpected intermediate_size on a MoE text tower"
            );
            let experts = positive(t, "num_experts")?;
            let experts_per_tok = positive(t, "num_experts_per_tok")?;
            ensure!(
                experts_per_tok <= experts,
                "num_experts_per_tok exceeds num_experts"
            );
            Some(MoeConfig {
                experts,
                experts_per_tok,
                expert_intermediate: positive(t, "moe_intermediate_size")?,
                shared_intermediate: positive(t, "shared_expert_intermediate_size")?,
                score: RouterScore::Softmax,
                shared_gate: true,
            })
        } else {
            for key in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "output_router_logits",
            ] {
                ensure!(
                    t.get(key).is_none(),
                    "unexpected {key} on a dense text tower"
                );
            }
            None
        };
        let intermediate = match &moe {
            Some(moe) => moe.expert_intermediate,
            None => positive(t, "intermediate_size")?,
        };
        let q_width = q_heads
            .checked_mul(head_dim)
            .context("Q width overflows u32")?;
        Ok(Self {
            hidden,
            intermediate,
            vocab,
            q_heads,
            kv_heads,
            head_dim,
            q_width,
            max_position,
            rms_eps: eps as f32,
            rope_theta: theta as f32,
            rotary_dim: rotary as u32,
            layers,
            moe,
            gdn,
        })
    }
}

impl ModelSpec {
    pub fn swiglu_rounding(&self) -> SwigluRounding {
        match self.architecture {
            Architecture::Flint => SwigluRounding::Once,
            Architecture::Glm5Next | Architecture::DeepseekV41 => SwigluRounding::ClampedTwice,
            Architecture::Llama
            | Architecture::Qwen35Text
            | Architecture::MiniCpmV47
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => SwigluRounding::Twice,
        }
    }

    pub fn swiglu_limit(&self) -> f32 {
        match self.architecture {
            Architecture::Glm5Next => self
                .glm5_next
                .as_ref()
                .map_or(0.0, |config| config.swiglu_limit),
            Architecture::DeepseekV41 => self
                .deepseek_v41
                .as_ref()
                .map_or(0.0, |config| config.swiglu_limit),
            Architecture::Llama
            | Architecture::Qwen35Text
            | Architecture::MiniCpmV47
            | Architecture::Flint
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => 0.0,
        }
    }

    pub fn routed_weight_boundary(&self) -> RoutedWeightBoundary {
        match self.architecture {
            Architecture::Flint | Architecture::Glm5Next => {
                RoutedWeightBoundary::BeforeContributionBf16
            }
            Architecture::DeepseekV41 => RoutedWeightBoundary::BeforeDownQuantization,
            Architecture::Llama
            | Architecture::Qwen35Text
            | Architecture::MiniCpmV47
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => RoutedWeightBoundary::AfterDownBf16,
        }
    }

    /// `Config` view for the model/CLI surface; does not relax Config::read's Llama validation.
    pub fn config(&self) -> crate::config::Config {
        crate::config::Config {
            model_type: match self.architecture {
                Architecture::Llama => "llama",
                Architecture::Qwen35Text => "qwen3_5_text",
                Architecture::MiniCpmV47 => "qwen3_5_moe_text",
                Architecture::Flint => "flint",
                Architecture::Glm5Next => "glm5_next_text",
                Architecture::LoopVl => "hrm_text",
                Architecture::DeepseekV41 => "deepseek_v41_text",
                Architecture::Gemma4 => "gemma4_text",
                Architecture::EmbeddingGemma2 => "embedding_gemma2_text",
            }
            .into(),
            hidden_size: self.hidden as usize,
            intermediate_size: self.intermediate as usize,
            num_hidden_layers: self.layers.len(),
            num_attention_heads: self.q_heads as usize,
            num_key_value_heads: self.kv_heads as usize,
            head_dim: self.head_dim as usize,
            vocab_size: self.vocab as usize,
            max_position_embeddings: self.max_position as usize,
            rms_norm_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            rope_scaling: None,
            hidden_act: if matches!(
                self.architecture,
                Architecture::Gemma4 | Architecture::EmbeddingGemma2
            ) {
                "gelu_pytorch_tanh"
            } else {
                "silu"
            }
            .into(),
            tie_word_embeddings: self.architecture == Architecture::Gemma4,
            torch_dtype: "bfloat16".into(),
            eos_token_id: self.eos.clone(),
            attention_bias: false,
            mlp_bias: false,
        }
    }
    pub fn read(dir: &Path) -> Result<Self> {
        let root: Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
        if root.get("model_type").and_then(Value::as_str) == Some("llama") {
            let c = crate::config::Config::read(dir)?;
            return Self::from_llama(&c);
        }
        let path = dir.join("generation_config.json");
        let generation = if path.exists() {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            Value::Null
        };
        if root.get("model_type").and_then(Value::as_str) == Some("minicpmv4_7") {
            Self::minicpm(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("flint") {
            Self::flint(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("glm5_next") {
            Self::glm5_next(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("loopvl") {
            Self::loopvl(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("deepseek_v41") {
            Self::deepseek_v41(&root, &generation)
        } else if root.get("model_type").and_then(Value::as_str) == Some("embedding_gemma2") {
            Self::embedding_gemma2(&root, &generation)
        } else if matches!(
            root.get("model_type").and_then(Value::as_str),
            Some("gemma4" | "gemma4_unified")
        ) {
            Self::gemma4(&root, &generation)
        } else {
            Self::qwen(&root, &generation)
        }
    }
    pub fn from_llama(c: &crate::config::Config) -> Result<Self> {
        let u = |n: usize| u32::try_from(n).context("Llama dimension overflows u32");
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Llama,
            hidden: u(c.hidden_size)?,
            intermediate: u(c.intermediate_size)?,
            vocab: u(c.vocab_size)?,
            q_heads: u(c.num_attention_heads)?,
            kv_heads: u(c.num_key_value_heads)?,
            head_dim: u(c.head_dim)?,
            q_width: u(c.hidden_size)?,
            max_position: u(c.max_position_embeddings)?,
            rms_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            rotary_dim: u(c.head_dim)?,
            linear_format: LinearFormat::Bf16,
            expert_format: None,
            eos: c.eos_token_id.clone(),
            vision: None,
            moe: None,
            gdn: None,
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            layers: (0..u(c.num_hidden_layers)?)
                .map(|state_index| LayerPlan {
                    geometry: None,
                    mixer: Mixer::FullAttention,
                    state_index,
                    rope: true,
                    sliding_window: None,
                    ffn: FfnKind::Dense,
                })
                .collect(),
        })
    }
    pub fn loopvl(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "loopvl".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["LoopVLForConditionalGeneration"]),
        )?;
        equal(root, "format_version", 2.into())?;
        ensure!(
            root.get("quantization_config").is_none(),
            "LoopVL requires published BF16/F32 tensors"
        );
        equal(
            root,
            "component_dtypes",
            serde_json::json!({"vision":"bfloat16", "language":"bfloat16", "projector":"float32", "visual_gate":"float32", "visual_anchor":"float32"}),
        )?;
        let t = root.get("text_config").context("missing text_config")?;
        for (key, value) in [
            ("model_type", "hrm_text".into()),
            ("dtype", "bfloat16".into()),
            ("hidden_act", "silu".into()),
            ("prefix_lm", true.into()),
            ("tie_word_embeddings", false.into()),
            ("attention_bias", false.into()),
            ("mlp_bias", false.into()),
            ("use_cache", false.into()),
            ("architectures", serde_json::json!(["HrmTextForCausalLM"])),
        ] {
            equal(t, key, value)?;
        }
        for (key, value) in [
            ("hidden_size", 1536),
            ("intermediate_size", 4096),
            ("vocab_size", 65536),
            ("num_attention_heads", 12),
            ("num_key_value_heads", 12),
            ("head_dim", 128),
            ("H_cycles", 2),
            ("L_cycles", 3),
            ("num_layers_per_stack", 16),
            ("num_hidden_layers", 128),
            ("max_position_embeddings", 8192),
            ("bos_token_id", 6),
            ("eos_token_id", 11),
            ("pad_token_id", 5),
        ] {
            equal(t, key, value.into())?;
        }
        for (key, value) in [
            ("rms_norm_eps", 1e-6),
            ("rope_theta", 10000.0),
            ("embedding_scale", 39.191835884530846),
            ("attention_dropout", 0.0),
        ] {
            ensure!(
                t.get(key).and_then(Value::as_f64) == Some(value),
                "unsupported LoopVL {key}"
            );
        }
        equal(
            t,
            "rope_parameters",
            serde_json::json!({"rope_type":"yarn", "rope_theta":10000, "factor":2, "original_max_position_embeddings":4096, "beta_fast":32, "beta_slow":1}),
        )?;
        let language = &root["loopvl_config"]["language"];
        for key in [
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "H_cycles",
            "L_cycles",
            "rope_theta",
            "rope_parameters",
            "max_position_embeddings",
        ] {
            ensure!(
                language.get(key) == t.get(key),
                "LoopVL language/text_config mismatch: {key}"
            );
        }
        let v = root.get("vision_config").context("missing vision_config")?;
        equal(v, "model_type", "penguinvl_vision_encoder".into())?;
        for (key, value) in [
            ("hidden_size", 1024),
            ("intermediate_size", 3072),
            ("num_hidden_layers", 28),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("patch_size", 14),
            ("num_channels", 3),
        ] {
            equal(v, key, value.into())?;
        }
        let eos_value = generation
            .get("eos_token_id")
            .filter(|v| !v.is_null())
            .unwrap_or(&t["eos_token_id"]);
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::LoopVl,
            hidden: positive(t, "hidden_size")?,
            intermediate: positive(t, "intermediate_size")?,
            vocab: positive(t, "vocab_size")?,
            q_heads: positive(t, "num_attention_heads")?,
            kv_heads: positive(t, "num_key_value_heads")?,
            head_dim: positive(t, "head_dim")?,
            q_width: 1536,
            max_position: positive(t, "max_position_embeddings")?,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            rotary_dim: 128,
            linear_format: LinearFormat::Bf16,
            expert_format: None,
            eos: eos(eos_value, 65536)?,
            layers: (0..128)
                .map(|state_index| LayerPlan {
                    geometry: None,
                    mixer: Mixer::FullAttention,
                    state_index,
                    rope: true,
                    sliding_window: None,
                    ffn: FfnKind::Dense,
                })
                .collect(),
            moe: None,
            gdn: None,
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            vision: None,
        })
    }
    pub fn qwen(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "qwen3_5".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["Qwen3_5ForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        let t = root.get("text_config").context("missing text_config")?;
        let text = TextGeometry::read(t, "qwen3_5_text")?;
        Self::from_text(
            Architecture::Qwen35Text,
            text,
            t,
            generation,
            VisionConfig::from_root(root)?,
            LinearFormat::read(root)?,
            None,
        )
    }
    pub fn minicpm(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "minicpmv4_7".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["MiniCPMV4_7ForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        equal(root, "dtype", "bfloat16".into())?;
        equal(root, "mrope_mode", "canvas".into())?;
        let t = root.get("text_config").context("missing text_config")?;
        // Declared by text_config; the checkpoint contains no mtp.* tensors.
        equal(t, "mtp_num_hidden_layers", 1.into())?;
        equal(t, "mtp_use_dedicated_embeddings", false.into())?;
        let text = TextGeometry::read(t, "qwen3_5_moe_text")?;
        let expert_format = ExpertFormat::read(root)?;
        let linear_format = match expert_format {
            ExpertFormat::Bf16 | ExpertFormat::Bf16Separate => LinearFormat::Bf16,
            ExpertFormat::Fp8W8a16 => LinearFormat::Fp8E4m3Channel,
            ExpertFormat::Fp8W8a16Block { .. } => {
                bail!("MiniCPM does not use F32 block-scale experts")
            }
            ExpertFormat::Fp4E2m1Block { .. } => {
                bail!("MiniCPM does not use E2M1 experts")
            }
        };
        let spec = Self::from_text(
            Architecture::MiniCpmV47,
            text,
            t,
            generation,
            VisionConfig::from_root(root)?,
            linear_format,
            Some(expert_format),
        )?;
        expert_format.check_ignore(root, &spec)?;
        Ok(spec)
    }
    pub fn flint(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "flint".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["FlintForCausalLM"]),
        )?;
        exact_keys(
            root,
            &[
                "architectures",
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "decoder_sparse_step",
                "eos_token_id",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "layer_types",
                "max_position_embeddings",
                "model_type",
                "mlp_only_layers",
                "moe_intermediate_size",
                "norm_topk_prob",
                "num_attention_heads",
                "num_experts",
                "num_experts_per_tok",
                "num_hidden_layers",
                "num_key_value_heads",
                "no_rope_layer_indices",
                "output_router_logits",
                "pad_token_id",
                "rms_norm_eps",
                "rope_theta",
                "router_aux_loss_coef",
                "router_score_function",
                "shared_expert_intermediate_size",
                "sliding_window",
                "softmax_type",
                "tie_word_embeddings",
                "torch_dtype",
                "transformers_version",
                "use_cache",
                "use_sliding_window",
                "vocab_size",
            ],
            "flint config",
        )?;
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("bos_token_id", 150000.into()),
            ("decoder_sparse_step", 1.into()),
            ("hidden_act", "silu".into()),
            ("initializer_range", 0.02.into()),
            ("norm_topk_prob", true.into()),
            ("output_router_logits", false.into()),
            ("pad_token_id", 150001.into()),
            ("router_aux_loss_coef", 0.0005.into()),
            ("router_score_function", "sigmoid".into()),
            ("softmax_type", "learnable".into()),
            ("tie_word_embeddings", false.into()),
            ("torch_dtype", "bfloat16".into()),
            ("transformers_version", "5.12.1".into()),
            ("use_cache", true.into()),
            ("use_sliding_window", true.into()),
        ] {
            equal(root, key, value)?;
        }
        let hidden = positive(root, "hidden_size")?;
        let head_dim = positive(root, "head_dim")?;
        let q_heads = positive(root, "num_attention_heads")?;
        let kv_heads = positive(root, "num_key_value_heads")?;
        let rope_theta = root
            .get("rope_theta")
            .and_then(Value::as_f64)
            .context("missing/invalid rope_theta")?;
        ensure!(
            [
                hidden,
                positive(root, "intermediate_size")?,
                positive(root, "max_position_embeddings")?,
                positive(root, "vocab_size")?,
                head_dim,
                q_heads,
                kv_heads,
            ] == [2048, 6144, 262144, 150144, 128, 32, 4]
                && rope_theta == 10000.0,
            "unsupported Flint attention geometry"
        );
        ensure!(
            q_heads.is_multiple_of(kv_heads),
            "Flint Q heads are not a multiple of KV heads"
        );
        let depth = positive(root, "num_hidden_layers")?;
        ensure!(depth == 49, "unsupported Flint depth {depth}");
        let sliding_window = positive(root, "sliding_window")?;
        ensure!(
            sliding_window == 513,
            "unsupported sliding_window {sliding_window}"
        );
        equal(root, "mlp_only_layers", serde_json::json!([0, 1]))?;
        let no_rope = integer_array(root, "no_rope_layer_indices", depth)?;
        ensure!(
            no_rope == [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48],
            "unsupported Flint no-rope layer set"
        );
        let types = root
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing/invalid layer_types")?;
        ensure!(types.len() == depth as usize, "invalid layer_types length");
        let mut layers = Vec::with_capacity(types.len());
        for (i, kind) in types.iter().enumerate() {
            let full = no_rope.contains(&(i as u32));
            let expected = if full {
                "full_attention"
            } else {
                "sliding_attention"
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported Flint layer type at {i}: {kind:?}"
            );
            layers.push(LayerPlan {
                geometry: None,
                mixer: Mixer::FullAttention,
                state_index: u32::try_from(i).context("Flint layer index overflows u32")?,
                rope: !full,
                sliding_window: (!full).then_some(sliding_window),
                ffn: if i < 2 { FfnKind::Dense } else { FfnKind::Moe },
            });
        }
        let moe = MoeConfig {
            experts: positive(root, "num_experts")?,
            experts_per_tok: positive(root, "num_experts_per_tok")?,
            expert_intermediate: positive(root, "moe_intermediate_size")?,
            shared_intermediate: positive(root, "shared_expert_intermediate_size")?,
            score: RouterScore::Sigmoid,
            shared_gate: false,
        };
        ensure!(
            [
                moe.experts,
                moe.experts_per_tok,
                moe.expert_intermediate,
                moe.shared_intermediate
            ] == [256, 6, 512, 1024],
            "unsupported Flint MoE geometry"
        );
        let eps = root
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .context("missing/invalid rms_norm_eps")?;
        ensure!(eps == 1e-6, "unsupported rms_norm_eps {eps}");
        let vocab = positive(root, "vocab_size")?;
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| root.get("eos_token_id"))
            .context("missing Flint EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Flint,
            hidden,
            intermediate: positive(root, "intermediate_size")?,
            vocab,
            q_heads,
            kv_heads,
            head_dim,
            q_width: q_heads * head_dim,
            max_position: positive(root, "max_position_embeddings")?,
            rms_eps: eps as f32,
            rope_theta: rope_theta as f32,
            rotary_dim: head_dim,
            linear_format: LinearFormat::Bf16,
            expert_format: Some(ExpertFormat::Bf16Separate),
            eos: eos(eos_value, vocab)?,
            layers,
            moe: Some(moe),
            gdn: None,
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            vision: None,
        })
    }
    pub fn gemma4(root: &Value, generation: &Value) -> Result<Self> {
        let unified = root.get("model_type").and_then(Value::as_str) == Some("gemma4_unified");
        for (key, value) in [
            (
                "model_type",
                if unified { "gemma4_unified" } else { "gemma4" }.into(),
            ),
            (
                "architectures",
                serde_json::json!([if unified {
                    "Gemma4UnifiedForConditionalGeneration"
                } else {
                    "Gemma4ForConditionalGeneration"
                }]),
            ),
            ("tie_word_embeddings", true.into()),
            ("dtype", "bfloat16".into()),
        ] {
            equal(root, key, value)?;
        }
        exact_keys(
            root,
            &[
                "architectures",
                "audio_config",
                "audio_token_id",
                "boa_token_id",
                "boi_token_id",
                "dtype",
                "eoa_token_id",
                "eoa_token_index",
                "eoi_token_id",
                "image_token_id",
                "initializer_range",
                "model_type",
                "text_config",
                "tie_word_embeddings",
                "transformers_version",
                "video_token_id",
                "vision_config",
                "vision_soft_tokens_per_image",
            ]
            .into_iter()
            .filter(|key| {
                !unified || !matches!(*key, "eoa_token_id" | "vision_soft_tokens_per_image")
            })
            .collect::<Vec<_>>(),
            "gemma4 config",
        )?;
        let t = root.get("text_config").context("missing text_config")?;
        let ple = integer(t, "hidden_size_per_layer_input")?;
        ensure!(
            matches!(ple, 0 | 256),
            "unsupported Gemma per-layer input width {ple}"
        );
        let shared_projection = ple == 0;
        ensure!(
            !unified || shared_projection,
            "unified text does not use per-layer inputs"
        );
        exact_keys(
            t,
            &[
                "attention_bias",
                "attention_dropout",
                "attention_k_eq_v",
                "bos_token_id",
                "dtype",
                "enable_moe_block",
                "eos_token_id",
                if unified {
                    "moe_intermediate_size"
                } else {
                    "expert_intermediate_size"
                },
                "final_logit_softcapping",
                "global_head_dim",
                "head_dim",
                "hidden_activation",
                "hidden_size",
                "hidden_size_per_layer_input",
                "initializer_range",
                "intermediate_size",
                "layer_types",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_experts",
                "num_global_key_value_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_kv_shared_layers",
                "pad_token_id",
                "rms_norm_eps",
                "rope_parameters",
                "sliding_window",
                "tie_word_embeddings",
                "top_k_experts",
                "use_bidirectional_attention",
                "use_cache",
                "use_double_wide_mlp",
                "vocab_size",
                "vocab_size_per_layer_input",
            ]
            .into_iter()
            .filter(|key| !unified || *key != "dtype")
            .collect::<Vec<_>>(),
            "gemma4 text_config",
        )?;
        for (key, value) in [
            (
                "model_type",
                if unified {
                    "gemma4_unified_text"
                } else {
                    "gemma4_text"
                }
                .into(),
            ),
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("attention_k_eq_v", shared_projection.into()),
            ("enable_moe_block", false.into()),
            (
                if unified {
                    "moe_intermediate_size"
                } else {
                    "expert_intermediate_size"
                },
                Value::Null,
            ),
            ("num_experts", Value::Null),
            ("top_k_experts", Value::Null),
            (
                "use_bidirectional_attention",
                if shared_projection {
                    "vision".into()
                } else {
                    Value::Null
                },
            ),
            ("hidden_activation", "gelu_pytorch_tanh".into()),
            ("tie_word_embeddings", true.into()),
            ("final_logit_softcapping", 30.0.into()),
            ("head_dim", 256.into()),
            ("global_head_dim", 512.into()),
            ("vocab_size", 262144.into()),
            ("vocab_size_per_layer_input", 262144.into()),
            ("rms_norm_eps", 1e-6.into()),
            ("bos_token_id", 2.into()),
            ("pad_token_id", 0.into()),
            ("use_cache", true.into()),
        ] {
            equal(t, key, value)?;
        }
        if !unified {
            equal(t, "dtype", "bfloat16".into())?;
        }
        if shared_projection {
            equal(t, "num_kv_shared_layers", 0.into())?;
            equal(t, "use_double_wide_mlp", false.into())?;
        } else {
            equal(t, "num_global_key_value_heads", Value::Null)?;
        }
        let rope = t
            .get("rope_parameters")
            .context("missing rope_parameters")?;
        exact_keys(
            rope,
            &["full_attention", "sliding_attention"],
            "gemma4 rope_parameters",
        )?;
        let full_rope = &rope["full_attention"];
        exact_keys(
            full_rope,
            &["partial_rotary_factor", "rope_theta", "rope_type"],
            "gemma4 full_attention rope",
        )?;
        equal(full_rope, "rope_type", "proportional".into())?;
        equal(full_rope, "partial_rotary_factor", 0.25.into())?;
        equal(full_rope, "rope_theta", 1000000.0.into())?;
        let sliding_rope = &rope["sliding_attention"];
        exact_keys(
            sliding_rope,
            &["rope_theta", "rope_type"],
            "gemma4 sliding_attention rope",
        )?;
        equal(sliding_rope, "rope_type", "default".into())?;
        equal(sliding_rope, "rope_theta", 10000.0.into())?;
        const SLIDING_HEAD_DIM: u32 = 256;
        const GLOBAL_HEAD_DIM: u32 = 512;
        let sliding = RopeSpec {
            theta: 10000.0,
            pairs: SLIDING_HEAD_DIM / 2,
            pair_stride: SLIDING_HEAD_DIM / 2,
            denominator: SLIDING_HEAD_DIM,
        };
        let global = RopeSpec {
            theta: 1000000.0,
            pairs: GLOBAL_HEAD_DIM / 8,
            pair_stride: GLOBAL_HEAD_DIM / 2,
            denominator: GLOBAL_HEAD_DIM,
        };
        let hidden = positive(t, "hidden_size")?;
        let base = positive(t, "intermediate_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        ensure!(
            matches!((hidden, depth), (1536, 35) | (2560, 42)),
            "unsupported Gemma 4 geometry hidden_size={hidden}, num_hidden_layers={depth}; enabled geometries are E2B and E4B"
        );
        let shared = integer(t, "num_kv_shared_layers")?;
        ensure!(
            shared < depth,
            "num_kv_shared_layers {shared} leaves no KV-owning layer"
        );
        let first_shared = depth - shared;
        let window = positive(t, "sliding_window")?;
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let global_kv_heads = shared_projection
            .then(|| positive(t, "num_global_key_value_heads"))
            .transpose()?;
        ensure!(
            q_heads.is_multiple_of(kv_heads)
                && global_kv_heads.is_none_or(|heads| q_heads.is_multiple_of(heads)),
            "num_attention_heads is not a multiple of num_key_value_heads"
        );
        let double_wide = t
            .get("use_double_wide_mlp")
            .and_then(Value::as_bool)
            .context("missing/invalid use_double_wide_mlp")?;
        let types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        ensure!(
            types.len() == depth as usize,
            "layer_types length differs from num_hidden_layers"
        );
        let mut publishers = [None; 2];
        let mut layers = Vec::with_capacity(types.len());
        for (i, kind) in types.iter().enumerate() {
            let i = i as u32;
            let (full, slot) = match kind.as_str() {
                Some("sliding_attention") => (false, 0),
                Some("full_attention") => (true, 1),
                _ => bail!("unsupported Gemma 4 layer type at {i}: {kind:?}"),
            };
            let reader = i >= first_shared;
            if !reader {
                publishers[slot] = Some(i);
            }
            let state_index = if reader {
                publishers[slot].with_context(|| {
                    format!("KV-shared layer {i} has no publisher of its layer type")
                })?
            } else {
                i
            };
            layers.push(LayerPlan {
                mixer: Mixer::FullAttention,
                state_index,
                rope: true,
                sliding_window: (!full).then_some(window),
                ffn: FfnKind::Dense,
                geometry: Some(LayerGeometry {
                    head_dim: if full {
                        GLOBAL_HEAD_DIM
                    } else {
                        SLIDING_HEAD_DIM
                    },
                    intermediate: if reader && double_wide {
                        base.checked_mul(2)
                            .context("double-wide MLP overflows u32")?
                    } else {
                        base
                    },
                    rope: if full { global } else { sliding },
                    kv_reader: reader,
                    kv_heads: if full { global_kv_heads } else { None },
                }),
            });
        }
        let vocab = 262144;
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing Gemma 4 EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Gemma4,
            hidden,
            intermediate: base,
            vocab,
            q_heads,
            kv_heads,
            head_dim: SLIDING_HEAD_DIM,
            q_width: q_heads
                .checked_mul(SLIDING_HEAD_DIM)
                .context("Q width overflows u32")?,
            max_position: positive(t, "max_position_embeddings")?,
            rms_eps: 1e-6,
            rope_theta: sliding.theta,
            rotary_dim: SLIDING_HEAD_DIM,
            linear_format: LinearFormat::Bf16,
            expert_format: None,
            eos: eos(eos_value, vocab)?,
            layers,
            moe: None,
            gdn: None,
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: Some(30.0),
            per_layer_input: (ple != 0).then_some(ple),
            audio: if ple != 0 {
                AudioConfig::from_root(root)?
            } else {
                None
            },
            vision: VisionConfig::from_root(root)?,
        })
    }
    pub fn embedding_gemma2(root: &Value, generation: &Value) -> Result<Self> {
        for (key, value) in [
            ("model_type", "embedding_gemma2".into()),
            ("architectures", serde_json::json!(["EmbeddingGemma2Model"])),
            ("dtype", "bfloat16".into()),
        ] {
            equal(root, key, value)?;
        }
        exact_keys(
            root,
            &[
                "architectures",
                "audio_config",
                "audio_token_id",
                "boa_token_id",
                "boi_token_id",
                "dtype",
                "eoa_token_index",
                "eoi_token_id",
                "image_token_id",
                "initializer_range",
                "model_type",
                "text_config",
                "transformers_version",
                "video_token_id",
                "vision_config",
                "vision_soft_tokens_per_image",
            ],
            "embedding_gemma2 config",
        )?;
        let t = root.get("text_config").context("missing text_config")?;
        exact_keys(
            t,
            &[
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "dtype",
                "embedding_dim",
                "eos_token_id",
                "head_dim",
                "hidden_activation",
                "hidden_size",
                "hidden_size_per_layer_input",
                "initializer_range",
                "intermediate_size",
                "layer_types",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "pad_token_id",
                "per_layer_config",
                "rms_norm_eps",
                "rope_parameters",
                "sliding_window",
                "vocab_size",
            ],
            "embedding_gemma2 text_config",
        )?;
        for (key, value) in [
            ("model_type", "embedding_gemma2_text".into()),
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("hidden_activation", "gelu_pytorch_tanh".into()),
            ("embedding_dim", 768.into()),
            ("hidden_size", 512.into()),
            ("hidden_size_per_layer_input", 512.into()),
            ("intermediate_size", 2048.into()),
            ("num_hidden_layers", 24.into()),
            ("num_attention_heads", 4.into()),
            ("num_key_value_heads", 2.into()),
            ("head_dim", 256.into()),
            ("sliding_window", 512.into()),
            ("vocab_size", 262144.into()),
            ("rms_norm_eps", 1e-6.into()),
            ("bos_token_id", 2.into()),
            ("pad_token_id", 0.into()),
        ] {
            equal(t, key, value)?;
        }
        let rope = t
            .get("rope_parameters")
            .context("missing rope_parameters")?;
        exact_keys(
            rope,
            &["full_attention", "sliding_attention"],
            "embedding_gemma2 rope_parameters",
        )?;
        for (kind, theta) in [
            ("full_attention", 1000000.0),
            ("sliding_attention", 10000.0),
        ] {
            let value = &rope[kind];
            exact_keys(value, &["rope_theta", "rope_type"], "embedding_gemma2 rope")?;
            equal(value, "rope_type", "default".into())?;
            equal(value, "rope_theta", theta.into())?;
        }
        let per_layer = t
            .get("per_layer_config")
            .context("missing per_layer_config")?;
        exact_keys(
            per_layer,
            &["05", "11", "17", "23"],
            "embedding_gemma2 per_layer_config",
        )?;
        for index in ["05", "11", "17", "23"] {
            let value = &per_layer[index];
            exact_keys(
                value,
                &["head_dim", "num_key_value_heads"],
                "embedding_gemma2 full layer",
            )?;
            equal(value, "head_dim", 512.into())?;
            equal(value, "num_key_value_heads", 1.into())?;
        }
        let types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        ensure!(types.len() == 24, "layer_types length differs from 24");
        let sliding = RopeSpec {
            theta: 10000.0,
            pairs: 128,
            pair_stride: 128,
            denominator: 256,
        };
        let global = RopeSpec {
            theta: 1000000.0,
            pairs: 256,
            pair_stride: 256,
            denominator: 512,
        };
        let mut layers = Vec::with_capacity(24);
        for (i, kind) in types.iter().enumerate() {
            let full = match kind.as_str() {
                Some("sliding_attention") => false,
                Some("full_attention") => true,
                _ => bail!("unsupported EmbeddingGemma-2 layer type at {i}: {kind:?}"),
            };
            ensure!(
                full == matches!(i, 5 | 11 | 17 | 23),
                "full layer layout mismatch at {i}"
            );
            layers.push(LayerPlan {
                mixer: Mixer::FullAttention,
                state_index: i as u32,
                rope: true,
                sliding_window: (!full).then_some(512),
                ffn: FfnKind::Dense,
                geometry: Some(LayerGeometry {
                    head_dim: if full { 512 } else { 256 },
                    intermediate: 2048,
                    rope: if full { global } else { sliding },
                    kv_reader: false,
                    kv_heads: full.then_some(1),
                }),
            });
        }
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EmbeddingGemma-2 EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::EmbeddingGemma2,
            hidden: 512,
            intermediate: 2048,
            vocab: 262144,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 256,
            q_width: 1024,
            max_position: positive(t, "max_position_embeddings")?,
            rms_eps: 1e-6,
            rope_theta: 10000.0,
            rotary_dim: 256,
            linear_format: LinearFormat::Bf16,
            expert_format: None,
            eos: eos(eos_value, 262144)?,
            layers,
            moe: None,
            gdn: None,
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: Some(512),
            vision: VisionConfig::from_root(root)?,
            audio: AudioConfig::from_root(root)?,
        })
    }
    pub fn glm5_next(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "glm5_next".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["Glm5NextForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        equal(root, "transformers_version", "5.16.0".into())?;
        let quant = root
            .get("quantization_config")
            .context("GLM-5.3 quantization_config missing")?;
        exact_keys(
            quant,
            &[
                "activation_scheme",
                "fmt",
                "modules_to_not_convert",
                "quant_method",
                "weight_block_size",
            ],
            "quantization_config",
        )?;
        for (key, value) in [
            ("activation_scheme", "dynamic"),
            ("fmt", "e4m3"),
            ("quant_method", "fp8"),
        ] {
            equal(quant, key, value.into())?;
        }
        equal(quant, "weight_block_size", serde_json::json!([128, 128]))?;
        ensure!(
            quant
                .get("modules_to_not_convert")
                .and_then(Value::as_array)
                .is_some_and(|entries| !entries.is_empty() && entries.iter().all(Value::is_string)),
            "quantization modules_to_not_convert must be a nonempty string array"
        );

        let t = root.get("text_config").context("missing text_config")?;
        exact_keys(
            t,
            &[
                "attention_bias",
                "attention_dropout",
                "dtype",
                "eos_token_id",
                "first_k_dense_replace",
                "hc_eps",
                "hc_mult",
                "hc_sinkhorn_iters",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "index_head_dim",
                "index_kpool",
                "index_kpool_always_select_tail",
                "index_kpool_compress",
                "index_n_heads",
                "index_share_for_mtp_iteration",
                "index_topk",
                "indexer_rope_interleave",
                "indexer_types",
                "initializer_range",
                "intermediate_size",
                "kv_lora_rank",
                "layer_types",
                "linear_attn_config",
                "max_position_embeddings",
                "mhc",
                "mla_use_nope",
                "mlp_layer_types",
                "model_type",
                "moe_intermediate_size",
                "moe_router_dtype",
                "n_group",
                "n_routed_experts",
                "n_shared_experts",
                "norm_topk_prob",
                "num_attention_heads",
                "num_experts_per_tok",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_nextn_predict_layers",
                "output_router_logits",
                "pad_token_id",
                "q_lora_rank",
                "qk_head_dim",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "rms_norm_eps",
                "routed_scaling_factor",
                "router_aux_loss_coef",
                "scoring_func",
                "swiglu_limit",
                "tie_word_embeddings",
                "topk_group",
                "topk_method",
                "use_cache",
                "v_head_dim",
                "vocab_size",
            ],
            "text_config",
        )?;
        equal(t, "model_type", "glm5_next_text".into())?;
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("dtype", "bfloat16".into()),
            ("hidden_act", "silu".into()),
            ("mhc", true.into()),
            ("mla_use_nope", true.into()),
            ("moe_router_dtype", "float32".into()),
            ("norm_topk_prob", true.into()),
            ("output_router_logits", false.into()),
            ("scoring_func", "sigmoid".into()),
            ("tie_word_embeddings", false.into()),
            ("topk_method", "noaux_tc".into()),
            ("use_cache", true.into()),
            ("index_kpool_always_select_tail", true.into()),
            ("index_kpool_compress", true.into()),
            ("index_share_for_mtp_iteration", true.into()),
            ("indexer_rope_interleave", true.into()),
        ] {
            equal(t, key, value)?;
        }
        for (key, value) in [
            ("head_dim", 0),
            ("n_group", 1),
            ("n_shared_experts", 1),
            ("num_nextn_predict_layers", 1),
            ("qk_rope_head_dim", 0),
            ("topk_group", 1),
        ] {
            equal(t, key, value.into())?;
        }
        let hidden = positive(t, "hidden_size")?;
        let vocab = positive(t, "vocab_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        let dense_layers = positive(t, "first_k_dense_replace")?;
        ensure!(
            dense_layers < depth,
            "first_k_dense_replace={dense_layers} must be below num_hidden_layers={depth}"
        );
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let qk_head_dim = positive(t, "qk_head_dim")?;
        let qk_nope_head_dim = positive(t, "qk_nope_head_dim")?;
        ensure!(
            qk_head_dim == qk_nope_head_dim && q_heads == kv_heads,
            "GLM-5.3 requires NoPE Q/K and equal Q/KV head counts"
        );
        let max_position = positive(t, "max_position_embeddings")?;
        let q_width = q_heads
            .checked_mul(qk_head_dim)
            .context("GLM-5.3 Q width overflow")?;
        let intermediate = positive(t, "intermediate_size")?;
        let expert_intermediate = positive(t, "moe_intermediate_size")?;
        let experts = positive(t, "n_routed_experts")?;
        let experts_per_tok = positive(t, "num_experts_per_tok")?;
        ensure!(
            experts_per_tok <= experts,
            "num_experts_per_tok exceeds n_routed_experts"
        );
        let shared_intermediate = expert_intermediate
            .checked_mul(positive(t, "n_shared_experts")?)
            .context("shared expert width overflow")?;
        let float = |key: &str| -> Result<f64> {
            t.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing/invalid {key}"))
        };
        let positive_f32 = |key: &str| -> Result<f32> {
            let value = float(key)? as f32;
            ensure!(value.is_finite() && value > 0.0, "invalid {key}");
            Ok(value)
        };
        for key in [
            "hc_eps",
            "initializer_range",
            "rms_norm_eps",
            "routed_scaling_factor",
            "swiglu_limit",
        ] {
            positive_f32(key)?;
        }
        ensure!(
            float("router_aux_loss_coef")?.is_finite() && float("router_aux_loss_coef")? >= 0.0,
            "invalid router_aux_loss_coef"
        );
        let rms_eps = positive_f32("rms_norm_eps")?;
        let hc_eps = positive_f32("hc_eps")?;
        let swiglu_limit = positive_f32("swiglu_limit")?;
        let routed_scaling_factor = positive_f32("routed_scaling_factor")?;
        let hc_mult = positive(t, "hc_mult")?;
        let hc_sinkhorn_iters = positive(t, "hc_sinkhorn_iters")?;
        ensure!(hc_mult == 4, "unsupported GLM-5.3 hc_mult={hc_mult}");
        let q_lora_rank = positive(t, "q_lora_rank")?;
        let kv_lora_rank = positive(t, "kv_lora_rank")?;
        let v_head_dim = positive(t, "v_head_dim")?;
        let index_heads = positive(t, "index_n_heads")?;
        let index_head_dim = positive(t, "index_head_dim")?;
        let index_topk = positive(t, "index_topk")?;
        let index_kpool = positive(t, "index_kpool")?;
        ensure!(
            index_topk.is_multiple_of(index_kpool),
            "index_topk is not an index_kpool multiple"
        );

        let linear = t
            .get("linear_attn_config")
            .context("missing linear_attn_config")?;
        exact_keys(
            linear,
            &[
                "num_heads",
                "gate_lower_bound",
                "head_dim",
                "short_conv_kernel_size",
                "kda_layers",
                "full_attn_layers",
            ],
            "linear_attn_config",
        )?;
        let linear_heads = positive(linear, "num_heads")?;
        let linear_head_dim = positive(linear, "head_dim")?;
        equal(linear, "short_conv_kernel_size", 4.into())?;
        let gate_lower_bound = linear
            .get("gate_lower_bound")
            .and_then(Value::as_f64)
            .context("missing/invalid gate_lower_bound")?;
        ensure!(
            gate_lower_bound.is_finite() && gate_lower_bound <= 0.0,
            "invalid gate_lower_bound"
        );
        let gate_lower_bound = gate_lower_bound as f32;
        ensure!(gate_lower_bound.is_finite(), "invalid gate_lower_bound");
        let layer_types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        let mlp_types = t
            .get("mlp_layer_types")
            .and_then(Value::as_array)
            .context("missing mlp_layer_types")?;
        let indexer_types = t
            .get("indexer_types")
            .and_then(Value::as_array)
            .context("missing indexer_types")?;
        ensure!(
            layer_types.len() == depth as usize
                && mlp_types.len() == depth as usize
                && indexer_types.len() == depth as usize,
            "GLM-5.3 layer type extent differs from num_hidden_layers"
        );
        let mut kda_layers = Vec::new();
        let mut full_attn_layers = Vec::new();
        let mut counts = [0u32; 2];
        let mut layers = Vec::with_capacity(depth as usize);
        for (i, kind) in layer_types.iter().enumerate() {
            let sparse = (i + 1).is_multiple_of(4);
            let expected = if sparse {
                "deepseek_sparse_attention"
            } else {
                "linear_attention"
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported mixer at layer {i}"
            );
            if sparse {
                full_attn_layers.push(i);
            } else {
                kda_layers.push(i);
            }
            let expected_ffn = if i < dense_layers as usize {
                "dense"
            } else {
                "sparse"
            };
            ensure!(
                mlp_types[i].as_str() == Some(expected_ffn),
                "unsupported FFN at layer {i}"
            );
            ensure!(
                indexer_types[i].as_str() == Some("full"),
                "unsupported indexer at layer {i}"
            );
            let (mixer, slot) = if sparse {
                (Mixer::FullAttention, 0)
            } else {
                (Mixer::Gdn, 1)
            };
            layers.push(LayerPlan {
                geometry: None,
                mixer,
                state_index: counts[slot],
                rope: false,
                sliding_window: None,
                ffn: if i < dense_layers as usize {
                    FfnKind::Dense
                } else {
                    FfnKind::Moe
                },
            });
            counts[slot] += 1;
        }
        ensure!(kda_layers.len() + full_attn_layers.len() == depth as usize);
        equal(linear, "kda_layers", serde_json::to_value(&kda_layers)?)?;
        equal(
            linear,
            "full_attn_layers",
            serde_json::to_value(&full_attn_layers)?,
        )?;
        ensure!(
            counts == [11, 34],
            "unsupported GLM-5.3 mixer counts {counts:?}"
        );
        let pad = integer(t, "pad_token_id")?;
        ensure!(pad < vocab, "pad_token_id outside vocabulary");
        let eos_value = generation
            .get("eos_token_id")
            .filter(|value| !value.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Glm5Next,
            hidden,
            intermediate,
            vocab,
            q_heads,
            kv_heads,
            head_dim: qk_head_dim,
            q_width,
            max_position,
            rms_eps,
            rope_theta: 1.0,
            rotary_dim: 0,
            linear_format: LinearFormat::fp8_block([128, 128], safetensors::Dtype::F32)?,
            expert_format: Some(ExpertFormat::Fp8W8a16Block {
                block: [128, 128],
                scale_dtype: safetensors::Dtype::F32,
            }),
            eos: eos(eos_value, vocab)?,
            layers,
            moe: Some(MoeConfig {
                experts,
                experts_per_tok,
                expert_intermediate,
                shared_intermediate,
                score: RouterScore::Sigmoid,
                shared_gate: false,
            }),
            gdn: Some(GdnConfig {
                key_heads: linear_heads,
                value_heads: linear_heads,
                key_head_dim: linear_head_dim,
                value_head_dim: linear_head_dim,
                conv_kernel: 4,
            }),
            glm5_next: Some(Glm5NextConfig {
                dense_layers,
                nextn_layers: positive(t, "num_nextn_predict_layers")?,
                hc_mult,
                hc_sinkhorn_iters,
                hc_eps,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                v_head_dim,
                index_heads,
                index_head_dim,
                index_topk,
                index_kpool,
                gate_lower_bound,
                swiglu_limit,
                routed_scaling_factor,
            }),
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            vision: None,
        })
    }
    pub fn deepseek_v41(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "deepseek_v41".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["DeepseekV41ForCausalLM"]),
        )?;
        equal(root, "dtype", "bfloat16".into())?;
        let quant = root
            .get("quantization_config")
            .context("DeepSeek-V4.1 quantization_config missing")?;
        exact_keys(
            quant,
            &[
                "activation_scheme",
                "expert_dtype",
                "quant_method",
                "scale_fmt",
                "weight_block_size",
            ],
            "quantization_config",
        )?;
        for (key, value) in [
            ("activation_scheme", "dynamic"),
            ("expert_dtype", "fp4"),
            ("quant_method", "fp8"),
            ("scale_fmt", "ue8m0"),
        ] {
            equal(quant, key, value.into())?;
        }
        let block = block_size(quant, "weight_block_size")?;
        ensure!(
            block == [32, 32],
            "unsupported DeepSeek-V4.1 weight block size: {block:?}"
        );

        let t = root.get("text_config").context("missing text_config")?;
        exact_keys(
            t,
            &[
                "attention_bias",
                "attention_dropout",
                "candidate_block_size",
                "candidate_source_layer_id",
                "candidate_topk_blocks",
                "compress_ratios",
                "compress_rope_theta",
                "dspark_block_size",
                "dspark_markov_rank",
                "dspark_n_routed_experts",
                "dspark_noise_token_id",
                "dspark_num_experts_per_tok",
                "dspark_target_layer_ids",
                "engram_compressed_vocab_size",
                "engram_head_dim",
                "engram_layer_ids",
                "engram_max_ngram_size",
                "engram_n_heads",
                "engram_num_embeddings",
                "engram_pad_token_id",
                "engram_vocab_size",
                "hc_eps",
                "hc_mult",
                "hc_sinkhorn_iters",
                "head_dim",
                "hidden_act",
                "hidden_size",
                "index_head_dim",
                "index_n_heads",
                "index_source_layer_ids",
                "index_topk",
                "initializer_range",
                "kv_source_layer_ids",
                "max_position_embeddings",
                "model_type",
                "moe_intermediate_size",
                "n_routed_experts",
                "n_shared_experts",
                "norm_topk_prob",
                "num_attention_heads",
                "num_experts_per_tok",
                "num_hidden_layers",
                "num_key_value_heads",
                "num_nextn_predict_layers",
                "o_groups",
                "o_lora_rank",
                "q_lora_rank",
                "qk_rope_head_dim",
                "rms_norm_eps",
                "rope_scaling",
                "rope_theta",
                "routed_scaling_factor",
                "scoring_func",
                "sliding_window",
                "swiglu_limit",
                "tie_word_embeddings",
                "topk_method",
                "use_cache",
                "vocab_size",
            ],
            "text_config",
        )?;
        equal(t, "model_type", "deepseek_v41_text".into())?;
        for (key, value) in [
            ("attention_bias", false.into()),
            ("attention_dropout", 0.0.into()),
            ("hidden_act", "silu".into()),
            ("norm_topk_prob", true.into()),
            ("scoring_func", "sqrtsoftplus".into()),
            ("tie_word_embeddings", false.into()),
            ("topk_method", "noaux_tc".into()),
            ("use_cache", true.into()),
        ] {
            equal(t, key, value)?;
        }
        let rope = t.get("rope_scaling").context("missing rope_scaling")?;
        exact_keys(
            rope,
            &[
                "beta_fast",
                "beta_slow",
                "factor",
                "original_max_position_embeddings",
                "rope_type",
            ],
            "rope_scaling",
        )?;
        equal(rope, "rope_type", "yarn".into())?;
        equal(rope, "factor", 16.into())?;
        equal(rope, "beta_fast", 32.into())?;
        equal(rope, "beta_slow", 1.into())?;
        equal(rope, "original_max_position_embeddings", 65536.into())?;
        let rope_factor = rope
            .get("factor")
            .and_then(Value::as_f64)
            .context("missing/invalid rope factor")? as f32;
        let original_max_position = positive(rope, "original_max_position_embeddings")?;
        let beta_fast = positive(rope, "beta_fast")?;
        let beta_slow = positive(rope, "beta_slow")?;

        let hidden = positive(t, "hidden_size")?;
        let vocab = positive(t, "vocab_size")?;
        let depth = positive(t, "num_hidden_layers")?;
        let q_heads = positive(t, "num_attention_heads")?;
        let kv_heads = positive(t, "num_key_value_heads")?;
        let head_dim = positive(t, "head_dim")?;
        let qk_rope_head_dim = positive(t, "qk_rope_head_dim")?;
        let q_lora_rank = positive(t, "q_lora_rank")?;
        let o_lora_rank = positive(t, "o_lora_rank")?;
        let o_groups = positive(t, "o_groups")?;
        let window_size = positive(t, "sliding_window")?;
        let max_position = positive(t, "max_position_embeddings")?;
        ensure!(
            q_heads.is_multiple_of(kv_heads),
            "num_attention_heads is not a multiple of num_key_value_heads"
        );
        ensure!(
            qk_rope_head_dim <= head_dim,
            "qk_rope_head_dim exceeds head_dim"
        );
        ensure!(
            q_heads.is_multiple_of(o_groups),
            "num_attention_heads is not a multiple of o_groups"
        );
        let float = |key: &str| -> Result<f64> {
            t.get(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing/invalid {key}"))
        };
        let positive_f32 = |key: &str| -> Result<f32> {
            let value = float(key)? as f32;
            ensure!(value.is_finite() && value > 0.0, "invalid {key}");
            Ok(value)
        };
        for key in [
            "hc_eps",
            "initializer_range",
            "rms_norm_eps",
            "routed_scaling_factor",
            "swiglu_limit",
            "compress_rope_theta",
            "rope_theta",
        ] {
            positive_f32(key)?;
        }
        let rms_eps = positive_f32("rms_norm_eps")?;
        let hc_eps = positive_f32("hc_eps")?;
        let swiglu_limit = positive_f32("swiglu_limit")?;
        let routed_scaling_factor = positive_f32("routed_scaling_factor")?;
        let rope_theta = positive_f32("rope_theta")?;
        let compress_rope_theta = positive_f32("compress_rope_theta")?;
        let hc_mult = positive(t, "hc_mult")?;
        let hc_sinkhorn_iters = positive(t, "hc_sinkhorn_iters")?;
        let experts = positive(t, "n_routed_experts")?;
        let shared_experts = positive(t, "n_shared_experts")?;
        let experts_per_tok = positive(t, "num_experts_per_tok")?;
        ensure!(
            experts_per_tok <= experts,
            "num_experts_per_tok exceeds n_routed_experts"
        );
        let expert_intermediate = positive(t, "moe_intermediate_size")?;
        let shared_intermediate = expert_intermediate
            .checked_mul(shared_experts)
            .context("shared expert width overflow")?;
        let index_heads = positive(t, "index_n_heads")?;
        let index_head_dim = positive(t, "index_head_dim")?;
        let index_topk = positive(t, "index_topk")?;
        let candidate_topk_blocks = positive(t, "candidate_topk_blocks")?;
        let candidate_block_size = positive(t, "candidate_block_size")?;
        let implemented = DeepseekSelectionShape::IMPLEMENTED;
        ensure!(
            index_topk == implemented.index_topk,
            "index_topk {} differs from implemented {}",
            index_topk,
            implemented.index_topk
        );
        ensure!(
            candidate_topk_blocks == implemented.candidate_topk_blocks,
            "candidate_topk_blocks {} differs from implemented {}",
            candidate_topk_blocks,
            implemented.candidate_topk_blocks
        );
        ensure!(
            candidate_block_size == implemented.candidate_block_size,
            "candidate_block_size {} differs from implemented {}",
            candidate_block_size,
            implemented.candidate_block_size
        );
        let selection = DeepseekSelectionShape {
            index_topk,
            candidate_topk_blocks,
            candidate_block_size,
        };
        let kv_source_layer_ids = integer_array(t, "kv_source_layer_ids", depth)?;
        let index_source_layer_ids = integer_array(t, "index_source_layer_ids", depth)?;
        let candidate_source_layer_id = integer(t, "candidate_source_layer_id")?;
        ensure!(
            candidate_source_layer_id < depth,
            "candidate_source_layer_id outside the layer extent"
        );
        let engram_layer_ids = integer_array(t, "engram_layer_ids", depth)?;
        let engram_num_embeddings = u32_list(t, "engram_num_embeddings")?;
        ensure!(
            engram_layer_ids.len() == engram_num_embeddings.len(),
            "engram_num_embeddings length differs from engram_layer_ids"
        );
        let engram_max_ngram_size = positive(t, "engram_max_ngram_size")?;
        let engram_n_heads = positive(t, "engram_n_heads")?;
        let engram_head_dim = positive(t, "engram_head_dim")?;
        positive(t, "engram_vocab_size")?;
        positive(t, "engram_compressed_vocab_size")?;
        let engram_pad_token_id = integer(t, "engram_pad_token_id")?;
        ensure!(
            engram_pad_token_id < vocab,
            "engram_pad_token_id outside vocabulary"
        );
        let nextn_layers = positive(t, "num_nextn_predict_layers")?;
        let dspark_experts = positive(t, "dspark_n_routed_experts")?;
        let dspark_experts_per_tok = positive(t, "dspark_num_experts_per_tok")?;
        ensure!(
            dspark_experts_per_tok <= dspark_experts,
            "dspark_num_experts_per_tok exceeds dspark_n_routed_experts"
        );
        let dspark_markov_rank = positive(t, "dspark_markov_rank")?;
        positive(t, "dspark_block_size")?;
        let dspark_noise_token_id = integer(t, "dspark_noise_token_id")?;
        ensure!(
            dspark_noise_token_id < vocab,
            "dspark_noise_token_id outside vocabulary"
        );
        integer_array(t, "dspark_target_layer_ids", depth)?;
        let ratios = t
            .get("compress_ratios")
            .and_then(Value::as_array)
            .context("missing/invalid compress_ratios")?;
        ensure!(
            ratios.len() == depth as usize + nextn_layers as usize,
            "compress_ratios length differs from the layer extent"
        );
        let compress_ratios: Vec<u8> = ratios
            .iter()
            .map(|ratio| {
                let value = ratio
                    .as_u64()
                    .with_context(|| format!("unsupported compress ratio {ratio:?}"))?;
                ensure!(value <= 2, "unsupported compress ratio {ratio:?}");
                u8::try_from(value).context("compress ratio outside u8")
            })
            .collect::<Result<_>>()?;

        let v = root.get("vision_config").context("missing vision_config")?;
        exact_keys(
            v,
            &[
                "downsample_ratio",
                "hidden_size",
                "intermediate_size",
                "max_image_tokens",
                "max_wh_ratio",
                "min_pixels",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "patch_size",
                "rope_theta",
            ],
            "vision_config",
        )?;
        equal(v, "model_type", "deepseek_v41_vision".into())?;
        let vision_layers = positive(v, "num_hidden_layers")?;
        let vision_hidden = positive(v, "hidden_size")?;
        positive(v, "num_attention_heads")?;
        let vision_intermediate = positive(v, "intermediate_size")?;
        let vision_patch = positive(v, "patch_size")?;
        let vision_downsample_ratio = positive(v, "downsample_ratio")?;
        positive(v, "max_image_tokens")?;
        positive(v, "min_pixels")?;
        ensure!(
            v.get("max_wh_ratio").is_none_or(Value::is_null),
            "unsupported vision max_wh_ratio"
        );
        let vision_rope = v
            .get("rope_theta")
            .and_then(Value::as_f64)
            .context("missing/invalid vision rope_theta")?;
        ensure!(
            vision_rope.is_finite() && vision_rope > 0.0,
            "unsupported vision rope_theta"
        );

        let mut layers = Vec::with_capacity(depth as usize);
        for index in 0..depth {
            layers.push(LayerPlan {
                geometry: None,
                mixer: Mixer::FullAttention,
                state_index: index,
                rope: true,
                sliding_window: Some(window_size),
                ffn: FfnKind::Moe,
            });
        }
        let eos_value = generation
            .get("eos_token_id")
            .filter(|value| !value.is_null())
            .or_else(|| root.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::DeepseekV41,
            hidden,
            intermediate: expert_intermediate,
            vocab,
            q_heads,
            kv_heads,
            head_dim,
            q_width: q_heads
                .checked_mul(head_dim)
                .context("DeepSeek-V4.1 Q width overflows u32")?,
            max_position,
            rms_eps,
            rope_theta,
            rotary_dim: qk_rope_head_dim,
            linear_format: LinearFormat::fp8_block(block, safetensors::Dtype::F8_E8M0)?,
            // Audited FP4 expert scales: one E8M0 scale per 32-element column block of each row.
            expert_format: Some(ExpertFormat::Fp4E2m1Block {
                block: [1, block[1]],
                scale_dtype: safetensors::Dtype::F8_E8M0,
            }),
            eos: eos(eos_value, vocab)?,
            layers,
            moe: Some(MoeConfig {
                experts,
                experts_per_tok,
                expert_intermediate,
                shared_intermediate,
                score: RouterScore::SqrtSoftplus,
                shared_gate: false,
            }),
            gdn: None,
            glm5_next: None,
            deepseek_v41: Some(DeepseekV41Config {
                num_hidden_layers: depth,
                hc_mult,
                hc_sinkhorn_iters,
                hc_eps,
                window_size,
                qk_rope_head_dim,
                q_lora_rank,
                o_lora_rank,
                o_groups,
                compress_ratios,
                compress_rope_theta,
                rope_factor,
                original_max_position,
                beta_fast,
                beta_slow,
                swiglu_limit,
                routed_scaling_factor,
                index_heads,
                index_head_dim,
                selection,
                kv_source_layer_ids,
                index_source_layer_ids,
                candidate_source_layer_id,
                engram_layer_ids,
                engram_num_embeddings,
                engram_max_ngram_size,
                engram_n_heads,
                engram_head_dim,
                nextn_layers,
                dspark_experts,
                dspark_markov_rank,
                vision_layers,
                vision_hidden,
                vision_intermediate,
                vision_patch,
                vision_downsample_ratio,
            }),
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            vision: None,
        })
    }
    fn from_text(
        architecture: Architecture,
        text: TextGeometry,
        t: &Value,
        generation: &Value,
        vision: Option<VisionConfig>,
        linear_format: LinearFormat,
        expert_format: Option<ExpertFormat>,
    ) -> Result<Self> {
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture,
            hidden: text.hidden,
            intermediate: text.intermediate,
            vocab: text.vocab,
            q_heads: text.q_heads,
            kv_heads: text.kv_heads,
            head_dim: text.head_dim,
            q_width: text.q_width,
            max_position: text.max_position,
            rms_eps: text.rms_eps,
            rope_theta: text.rope_theta,
            rotary_dim: text.rotary_dim,
            linear_format,
            expert_format,
            eos: eos(eos_value, text.vocab)?,
            layers: text.layers,
            moe: text.moe,
            gdn: Some(text.gdn),
            glm5_next: None,
            deepseek_v41: None,
            logit_softcap: None,
            per_layer_input: None,
            audio: None,
            vision,
        })
    }
}
