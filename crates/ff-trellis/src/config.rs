//! TRELLIS component and pipeline configuration, as published beside the
//! weights.
//!
//! Every component carries a `<name>.json` sidecar of the shape
//! `{"name": ..., "args": {...}}` next to its `<name>.safetensors`, and every
//! checkpoint directory carries a `pipeline.json` naming the components a
//! pipeline uses. Both are closed here: an unrecognised component name is an
//! error rather than something to guess at, because the name selects which
//! mathematics runs.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Which mathematics a component runs.
///
/// The published sidecars carry one of these in `name`. Adding a variant is how
/// a new component becomes loadable; the argument shapes differ enough between
/// them that one permissive struct would accept configurations no code can run.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "name", content = "args")]
pub enum ComponentConfig {
    /// Dense DiT over a low-resolution occupancy latent.
    SparseStructureFlowModel(SparseStructureFlowArgs),
    /// Dense 3D convolutional encoder from a voxel grid to that latent.
    SparseStructureEncoder(SparseStructureCoderArgs),
    /// Dense 3D convolutional decoder back to a voxel grid.
    SparseStructureDecoder(SparseStructureCoderArgs),
    /// Sparse DiT over the active voxels' structured latents.
    SLatFlowModel(SLatFlowArgs),
    /// Sparse windowed-attention encoder to structured latents.
    SLatEncoder(SLatCoderArgs),
    /// Sparse windowed-attention decoder to 3D Gaussians.
    SLatGaussianDecoder(SLatCoderArgs),
    /// Sparse windowed-attention decoder to a radiance field.
    SLatRadianceFieldDecoder(SLatCoderArgs),
    /// Sparse windowed-attention decoder to a mesh.
    SLatMeshDecoder(SLatCoderArgs),
    /// TRELLIS.2 shape VAE halves.
    FlexiDualGridVaeEncoder(serde_json::Map<String, serde_json::Value>),
    FlexiDualGridVaeDecoder(serde_json::Map<String, serde_json::Value>),
    /// TRELLIS.2 texture VAE halves.
    SparseUnetVaeEncoder(serde_json::Map<String, serde_json::Value>),
    SparseUnetVaeDecoder(serde_json::Map<String, serde_json::Value>),
}

impl ComponentConfig {
    /// The component's name as published, for reporting.
    pub fn name(&self) -> &'static str {
        match self {
            Self::SparseStructureFlowModel(_) => "SparseStructureFlowModel",
            Self::SparseStructureEncoder(_) => "SparseStructureEncoder",
            Self::SparseStructureDecoder(_) => "SparseStructureDecoder",
            Self::SLatFlowModel(_) => "SLatFlowModel",
            Self::SLatEncoder(_) => "SLatEncoder",
            Self::SLatGaussianDecoder(_) => "SLatGaussianDecoder",
            Self::SLatRadianceFieldDecoder(_) => "SLatRadianceFieldDecoder",
            Self::SLatMeshDecoder(_) => "SLatMeshDecoder",
            Self::FlexiDualGridVaeEncoder(_) => "FlexiDualGridVaeEncoder",
            Self::FlexiDualGridVaeDecoder(_) => "FlexiDualGridVaeDecoder",
            Self::SparseUnetVaeEncoder(_) => "SparseUnetVaeEncoder",
            Self::SparseUnetVaeDecoder(_) => "SparseUnetVaeDecoder",
        }
    }

    /// Whether this crate can currently evaluate the component.
    ///
    /// Reported rather than enforced at load time, so `inspect` can describe a
    /// whole checkpoint including the parts that are not executable yet.
    pub fn is_executable(&self) -> bool {
        match self {
            Self::SparseStructureFlowModel(args) => {
                if args.share_mod {
                    args.pe_mode == PositionEmbeddingMode::Rope
                        && matches!(
                            args.dtype.as_deref(),
                            Some("bfloat16" | "float16" | "float32")
                        )
                } else {
                    args.pe_mode == PositionEmbeddingMode::Ape
                }
            }
            Self::SparseStructureDecoder(_) | Self::SLatGaussianDecoder(_) => true,
            Self::SLatFlowModel(args) => {
                if args.share_mod {
                    args.pe_mode == PositionEmbeddingMode::Rope
                        && args.io_block_channels.is_empty()
                        && args.patch_size == 1
                        && matches!(
                            args.dtype.as_deref(),
                            Some("bfloat16" | "float16" | "float32")
                        )
                } else {
                    args.pe_mode == PositionEmbeddingMode::Ape
                        && args.patch_size.is_power_of_two()
                        && args.patch_size.trailing_zeros() as usize == args.io_block_channels.len()
                }
            }
            Self::FlexiDualGridVaeDecoder(args) | Self::SparseUnetVaeDecoder(args) => {
                args.get("block_type")
                    .and_then(|v| v.as_array())
                    .is_some_and(|v| v.iter().all(|x| x == "SparseConvNeXtBlock3d"))
                    && args
                        .get("up_block_type")
                        .and_then(|v| v.as_array())
                        .is_some_and(|v| v.iter().all(|x| x == "SparseResBlockC2S3d"))
            }
            _ => false,
        }
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("invalid TRELLIS component configuration")
    }
}

/// Arguments of the dense sparse-structure DiT.
///
/// The optional fields are optional in the published sidecars, not merely
/// tolerated here: of the five sparse-structure flow models in this family,
/// four name `patch_size` and one does not, two name `qk_rms_norm_cross` and
/// one names `share_mod`. Their defaults below are the values the absent files'
/// weight shapes imply, not guesses.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SparseStructureFlowArgs {
    /// Side length of the cubic latent grid.
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    /// Width of the conditioning the cross-attention reads.
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    /// Absent in the 1.3B model, whose `input_layer.weight` is
    /// `[model_channels, in_channels]` and so patchifies one voxel per token.
    #[serde(default = "unit_patch")]
    pub patch_size: usize,
    /// `ape` carries a learned `pos_emb`; `rope` carries none.
    pub pe_mode: PositionEmbeddingMode,
    /// Whether self-attention query and key are RMS-normalized.
    pub qk_rms_norm: bool,
    /// Whether cross-attention query and key are too.
    #[serde(default)]
    pub qk_rms_norm_cross: bool,
    /// When set, one modulation projection is shared across blocks and each
    /// block carries only its own `modulation` parameter.
    #[serde(default)]
    pub share_mod: bool,
    #[serde(default)]
    pub use_fp16: bool,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub initialization: Option<String>,
}

/// One voxel per token, which is what an absent `patch_size` means here.
fn unit_patch() -> usize {
    1
}

impl SparseStructureFlowArgs {
    /// Tokens the DiT attends over: one per patch of the cubic latent.
    pub fn token_count(&self) -> Result<usize> {
        anyhow::ensure!(
            self.patch_size > 0 && self.resolution.is_multiple_of(self.patch_size),
            "TRELLIS resolution {} is not divisible by patch size {}",
            self.resolution,
            self.patch_size
        );
        let side = self.resolution / self.patch_size;
        side.checked_pow(3)
            .context("TRELLIS token count overflows usize")
    }

    pub fn head_dim(&self) -> Result<usize> {
        anyhow::ensure!(
            self.num_heads > 0 && self.model_channels.is_multiple_of(self.num_heads),
            "TRELLIS model channels {} do not divide into {} heads",
            self.model_channels,
            self.num_heads
        );
        Ok(self.model_channels / self.num_heads)
    }
}

/// Arguments of the sparse structured-latent DiT.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SLatFlowArgs {
    pub resolution: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    #[serde(default = "unit_patch")]
    pub patch_size: usize,
    /// Sparse convolutional residual blocks either side of the transformer.
    /// Absent in the TRELLIS.2 flows, which have none.
    #[serde(default)]
    pub num_io_res_blocks: usize,
    #[serde(default)]
    pub io_block_channels: Vec<usize>,
    pub pe_mode: PositionEmbeddingMode,
    pub qk_rms_norm: bool,
    #[serde(default)]
    pub qk_rms_norm_cross: bool,
    #[serde(default)]
    pub share_mod: bool,
    #[serde(default)]
    pub use_fp16: bool,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub initialization: Option<String>,
}

/// Arguments shared by the dense 3D convolutional structure coders.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SparseStructureCoderArgs {
    #[serde(default)]
    pub in_channels: Option<usize>,
    #[serde(default)]
    pub out_channels: Option<usize>,
    pub latent_channels: usize,
    pub num_res_blocks: usize,
    pub num_res_blocks_middle: usize,
    /// Channel widths, coarsest first for a decoder and last for an encoder.
    pub channels: Vec<usize>,
    pub use_fp16: bool,
}

/// Arguments shared by the sparse windowed-attention coders.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SLatCoderArgs {
    pub resolution: usize,
    #[serde(default)]
    pub in_channels: Option<usize>,
    pub model_channels: usize,
    pub latent_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub attn_mode: AttentionMode,
    pub window_size: usize,
    pub use_fp16: bool,
    /// Shape depends on which representation the decoder emits, so it is kept
    /// as published rather than modelled per decoder.
    #[serde(default)]
    pub representation_config: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionEmbeddingMode {
    /// Learned absolute position embedding.
    Ape,
    /// Rotary position embedding.
    Rope,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMode {
    Full,
    /// Shifted-window attention over 3D voxel windows.
    Swin,
}

/// A `pipeline.json`: which components a pipeline uses, and how it samples them.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PipelineConfig {
    /// `TrellisTextTo3DPipeline`, `TrellisImageTo3DPipeline`, `Trellis2ImageTo3DPipeline`.
    pub name: String,
    pub args: PipelineArgs,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PipelineArgs {
    /// Role to component reference. A reference is either relative to this
    /// checkpoint (`ckpts/name`) or names another published one
    /// (`owner/repo/ckpts/name`).
    pub models: BTreeMap<String, String>,
    /// The conditioner, which lives outside the checkpoint directory. Spelled
    /// as a bare model id in the older pipelines and as a `{name, args}` object
    /// in TRELLIS.2.
    #[serde(default)]
    pub text_cond_model: Option<serde_json::Value>,
    #[serde(default)]
    pub image_cond_model: Option<serde_json::Value>,
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

impl PipelineConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let config: Self =
            serde_json::from_slice(bytes).context("invalid TRELLIS pipeline configuration")?;
        anyhow::ensure!(
            !config.args.models.is_empty(),
            "TRELLIS pipeline names no components"
        );
        Ok(config)
    }

    /// The conditioning model this pipeline needs, which the checkpoint
    /// directory does not contain.
    pub fn conditioner(&self) -> Option<Conditioner> {
        let (kind, value) = match (&self.args.text_cond_model, &self.args.image_cond_model) {
            (Some(value), _) => (ConditionerKind::Text, value),
            (None, Some(value)) => (ConditionerKind::Image, value),
            (None, None) => return None,
        };
        let model = match value {
            serde_json::Value::String(id) => id.clone(),
            serde_json::Value::Object(object) => object
                .get("args")
                .and_then(|args| args.get("model_name"))
                .and_then(|name| name.as_str())
                .unwrap_or_else(|| {
                    object
                        .get("name")
                        .and_then(|name| name.as_str())
                        .unwrap_or("unknown")
                })
                .to_owned(),
            _ => return None,
        };
        Some(Conditioner { kind, model })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConditionerKind {
    Text,
    Image,
}

/// A `pipeline.json` sampler block.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SamplerSpec {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl SamplerSpec {
    /// `sigma_min`, which the published blocks always carry.
    pub fn sigma_min(&self) -> Result<f64> {
        self.args
            .get("sigma_min")
            .and_then(serde_json::Value::as_f64)
            .context("sampler block has no sigma_min")
    }

    /// The scheduling parameters, in TRELLIS-1's spelling.
    ///
    /// TRELLIS.2 renames them — `guidance_strength`, `guidance_interval`, plus
    /// a `guidance_rescale` with no counterpart here — and its sampler is a
    /// different algorithm, so that spelling is refused rather than mapped onto
    /// this one.
    pub fn parameters(&self) -> Result<(usize, f64, (f64, f64), f64)> {
        anyhow::ensure!(
            self.name == "FlowEulerGuidanceIntervalSampler",
            "unsupported TRELLIS sampler {}",
            self.name
        );
        anyhow::ensure!(
            !self.params.contains_key("guidance_strength"),
            "this sampler block uses the TRELLIS.2 spelling ({:?}); only the TRELLIS-1 guidance-interval sampler is implemented",
            self.params.keys().collect::<Vec<_>>()
        );
        let number = |key: &str| -> Result<f64> {
            self.params
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .with_context(|| format!("sampler block has no {key}"))
        };
        let interval = self
            .params
            .get("cfg_interval")
            .and_then(serde_json::Value::as_array)
            .filter(|values| values.len() == 2)
            .context("sampler block has no two-element cfg_interval")?;
        let bound = |index: usize| -> Result<f64> {
            interval[index]
                .as_f64()
                .context("cfg_interval bound is not a number")
        };
        Ok((
            usize::try_from(
                self.params
                    .get("steps")
                    .and_then(|v| v.as_u64())
                    .context("sampler steps must be an unsigned integer")?,
            )?,
            number("cfg_strength")?,
            (bound(0)?, bound(1)?),
            number("rescale_t")?,
        ))
    }
}

impl PipelineConfig {
    /// The sparse-structure sampler block.
    pub fn sparse_structure_sampler(&self) -> Result<SamplerSpec> {
        let value = self
            .args
            .rest
            .get("sparse_structure_sampler")
            .context("pipeline has no sparse_structure_sampler block")?;
        serde_json::from_value(value.clone()).context("invalid sparse_structure_sampler block")
    }
}

/// A conditioning model a pipeline depends on but does not ship.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Conditioner {
    pub kind: ConditionerKind,
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_published_sparse_structure_flow_sidecar() {
        let config = ComponentConfig::from_json(
            br#"{"name":"SparseStructureFlowModel","args":{"resolution":16,"in_channels":8,
                 "out_channels":8,"model_channels":768,"cond_channels":768,"num_blocks":12,
                 "num_heads":12,"mlp_ratio":4,"patch_size":1,"pe_mode":"ape",
                 "qk_rms_norm":true,"use_fp16":true}}"#,
        )
        .unwrap();
        let ComponentConfig::SparseStructureFlowModel(args) = &config else {
            panic!("expected the flow model")
        };
        assert_eq!(args.token_count().unwrap(), 4_096);
        assert_eq!(args.head_dim().unwrap(), 64);
        assert_eq!(args.pe_mode, PositionEmbeddingMode::Ape);
        assert!(config.is_executable());
    }

    #[test]
    fn rejects_an_unknown_component_rather_than_guessing() {
        let error = ComponentConfig::from_json(br#"{"name":"SomethingElse","args":{}}"#)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid TRELLIS component configuration"),
            "{error}"
        );
    }

    #[test]
    fn a_resolution_that_does_not_divide_by_the_patch_is_refused() {
        let args = SparseStructureFlowArgs {
            resolution: 16,
            in_channels: 8,
            out_channels: 8,
            model_channels: 768,
            cond_channels: 768,
            num_blocks: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            patch_size: 3,
            pe_mode: PositionEmbeddingMode::Ape,
            qk_rms_norm: true,
            qk_rms_norm_cross: false,
            share_mod: false,
            use_fp16: true,
            dtype: None,
            initialization: None,
        };
        assert!(
            args.token_count()
                .unwrap_err()
                .to_string()
                .contains("divisible")
        );
    }

    #[test]
    fn reads_the_conditioner_out_of_both_published_spellings() {
        let text = PipelineConfig::from_json(
            br#"{"name":"TrellisTextTo3DPipeline","args":{"models":{"a":"ckpts/b"},
                 "text_cond_model":"openai/clip-vit-large-patch14"}}"#,
        )
        .unwrap();
        assert_eq!(
            text.conditioner().unwrap(),
            Conditioner {
                kind: ConditionerKind::Text,
                model: "openai/clip-vit-large-patch14".into()
            }
        );
        let image = PipelineConfig::from_json(
            br#"{"name":"Trellis2ImageTo3DPipeline","args":{"models":{"a":"ckpts/b"},
                 "image_cond_model":{"name":"DinoV3FeatureExtractor",
                 "args":{"model_name":"facebook/dinov3-vitl16-pretrain-lvd1689m"}}}}"#,
        )
        .unwrap();
        assert_eq!(
            image.conditioner().unwrap(),
            Conditioner {
                kind: ConditionerKind::Image,
                model: "facebook/dinov3-vitl16-pretrain-lvd1689m".into()
            }
        );
    }
}
