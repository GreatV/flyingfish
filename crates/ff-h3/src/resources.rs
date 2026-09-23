use crate::{
    config::TransformerConfig,
    core::{
        AttentionKeyChunkPolicy, DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE,
        DEFAULT_ATTENTION_QUERY_CHUNK_SIZE, DEFAULT_FFN_TOKEN_CHUNK_SIZE,
    },
    model::DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

mod activations;
mod host;
pub use ff_core::bounds;
pub use host::{cache_charge, host_weight_residency_charges};
mod traffic;
mod weights;

pub use bounds::{BudgetReport, BudgetViolation, ResourceBudget, ResourceDomain, format_bytes};

use activations::estimate_activations;
use traffic::estimate_compute_and_traffic;
use weights::estimate_weights;

pub const H3_BASE_CHECKPOINT_BYTES: u64 = 66_280_430_080;

pub const H3_DEFAULT_EVALUATION_COUNT: u64 = 49;
pub const RESOURCE_ESTIMATE_SCHEMA_VERSION: u32 = 1;

const H3_MODALITY_COUNT: u64 = 3;
fn default_timestep_rows() -> u64 {
    2
}
fn is_default_timestep_rows(rows: &u64) -> bool {
    *rows == 2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct T2vaGeometry {
    pub text_rows: usize,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub audio_frames: usize,
    pub audio_channels: usize,
    pub attention_projection_chunk_size: usize,
    pub attention_query_chunk_size: usize,
    pub attention_key_chunk_policy: AttentionKeyChunkPolicy,
    pub ffn_token_chunk_size: usize,
    pub output_token_chunk_size: usize,
}

impl T2vaGeometry {
    pub const fn h3_default(text_rows: usize) -> Self {
        Self {
            text_rows,
            latent_frames: 37,
            latent_height: 48,
            latent_width: 84,
            audio_frames: 207,
            audio_channels: 2,
            attention_projection_chunk_size: DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE,
            attention_query_chunk_size: DEFAULT_ATTENTION_QUERY_CHUNK_SIZE,
            attention_key_chunk_policy: AttentionKeyChunkPolicy::Full,
            ffn_token_chunk_size: DEFAULT_FFN_TOKEN_CHUNK_SIZE,
            output_token_chunk_size: DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE,
        }
    }

    pub fn validate(&self, patch_size: [usize; 3]) -> Result<()> {
        for (name, value) in [
            ("latent_frames", self.latent_frames),
            ("latent_height", self.latent_height),
            ("latent_width", self.latent_width),
            ("audio_frames", self.audio_frames),
            ("audio_channels", self.audio_channels),
            (
                "attention_projection_chunk_size",
                self.attention_projection_chunk_size,
            ),
            (
                "attention_query_chunk_size",
                self.attention_query_chunk_size,
            ),
            ("ffn_token_chunk_size", self.ffn_token_chunk_size),
            ("output_token_chunk_size", self.output_token_chunk_size),
        ] {
            anyhow::ensure!(value > 0, "{name} must be non-zero");
        }
        anyhow::ensure!(
            patch_size.iter().all(|&value| value > 0),
            "patch_size must be non-zero"
        );
        anyhow::ensure!(
            self.latent_frames.is_multiple_of(patch_size[0])
                && self.latent_height.is_multiple_of(patch_size[1])
                && self.latent_width.is_multiple_of(patch_size[2]),
            "video latent geometry is not divisible by patch {patch_size:?}"
        );
        Ok(())
    }

    pub fn sequence_rows(&self, patch_size: [usize; 3]) -> Result<SequenceRows> {
        self.validate(patch_size)?;
        let latent_frames = as_u64(self.latent_frames, "latent frame count")?;
        let latent_height = as_u64(self.latent_height, "latent height")?;
        let latent_width = as_u64(self.latent_width, "latent width")?;
        let patch_t = as_u64(patch_size[0], "temporal patch")?;
        let patch_h = as_u64(patch_size[1], "height patch")?;
        let patch_w = as_u64(patch_size[2], "width patch")?;
        let video_rows_per_frame = checked_product(
            "video rows per patched frame",
            &[latent_height / patch_h, latent_width / patch_w],
        )?;
        let patched_frames = latent_frames / patch_t;
        let video = checked_product("video rows", &[patched_frames, video_rows_per_frame])?;
        let audio = checked_product(
            "audio rows",
            &[
                as_u64(self.audio_frames, "audio frame count")?,
                as_u64(self.audio_channels, "audio channel count")?,
            ],
        )?;
        let text = as_u64(self.text_rows, "text row count")?;
        let total = checked_sum("total sequence rows", &[text, audio, video])?;
        Ok(SequenceRows {
            text,
            audio,
            video,
            total,
            patched_video_frames: patched_frames,
            video_rows_per_frame,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceRows {
    pub text: u64,
    pub audio: u64,
    pub video: u64,
    pub total: u64,
    pub patched_video_frames: u64,
    pub video_rows_per_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformerShape {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub hidden_size: usize,
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_embed_hidden_dim: usize,
    pub time_embed_dim: usize,
    pub rope_freq_dim: usize,
}

impl TransformerShape {
    pub const fn h3_base() -> Self {
        Self {
            num_layers: 50,
            num_attention_heads: 56,
            attention_head_dim: 128,
            hidden_size: 5_376,
            ffn_dim: 14_336,
            in_channels: 24,
            audio_in_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5_120,
            freq_dim: 256,
            time_embed_hidden_dim: 5_376,
            time_embed_dim: 2_688,
            rope_freq_dim: 16,
        }
    }

    pub fn from_config(config: &TransformerConfig) -> Self {
        Self {
            num_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads,
            attention_head_dim: config.attention_head_dim,
            hidden_size: config.hidden_size,
            ffn_dim: config.ffn_dim,
            in_channels: config.in_channels,
            audio_in_channels: config.audio_in_channels,
            patch_size: config.patch_size,
            text_dim: config.text_dim,
            freq_dim: config.freq_dim,
            time_embed_hidden_dim: config.time_embed_hidden_dim,
            time_embed_dim: config.time_embed_dim,
            rope_freq_dim: config.rope_freq_dim,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("num_layers", self.num_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("attention_head_dim", self.attention_head_dim),
            ("hidden_size", self.hidden_size),
            ("ffn_dim", self.ffn_dim),
            ("in_channels", self.in_channels),
            ("audio_in_channels", self.audio_in_channels),
            ("text_dim", self.text_dim),
            ("freq_dim", self.freq_dim),
            ("time_embed_hidden_dim", self.time_embed_hidden_dim),
            ("time_embed_dim", self.time_embed_dim),
            ("rope_freq_dim", self.rope_freq_dim),
        ] {
            anyhow::ensure!(value > 0, "{name} must be non-zero");
        }
        anyhow::ensure!(
            self.patch_size.iter().all(|&value| value > 0),
            "patch_size must be non-zero"
        );
        Ok(())
    }
}

impl From<&TransformerConfig> for TransformerShape {
    fn from(config: &TransformerConfig) -> Self {
        Self::from_config(config)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceAssumptions {
    pub weight_element_bytes: u64,
    pub activation_element_bytes: u64,
    pub io_weight_element_bytes: u64,
    pub accumulator_element_bytes: u64,
    pub device_memory_is_host: bool,
    pub use_flash_attention: bool,
    pub normalization_f32_buffer_count: u64,
    pub softmax_f32_buffer_count: u64,
    pub modulation_buffer_count: u64,
    pub pipeline_latent_buffer_count: u64,
    pub evaluation_count: u64,
    pub precompute_adaln_steps: u64,
    /// T2VA has two timestep rows; conditioned modes may require three/four.
    /// Omitting the default preserves existing T2VA estimate serialization.
    #[serde(
        default = "default_timestep_rows",
        skip_serializing_if = "is_default_timestep_rows"
    )]
    pub timestep_rows: u64,
    /// Host weight-cache bytes charged to the host peak. Under the
    /// unified-memory fold (`device_memory_is_host` on a CUDA device) this
    /// also carries the once-only device residency reserve, routed here via
    /// `additional_host_allowance_bytes` — do not read the field as a cache
    /// sizing recommendation without checking that allowance first.
    pub host_weight_cache_bytes: u64,
    pub mapped_weight_residency_bytes: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub checkpoint_weight_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub peak_materialized_weight_bytes_override: Option<u64>,
    /// Full cache ceiling in addition to live stage weights and scratch.
    #[serde(default, skip_serializing_if = "device_cache_bytes_is_zero")]
    pub device_weight_cache_bytes: u64,
    pub backend_workspace_bytes: u64,
}

fn device_cache_bytes_is_zero(bytes: &u64) -> bool {
    *bytes == 0
}

impl ResourceAssumptions {
    pub const fn h3_bf16_mmap() -> Self {
        Self {
            weight_element_bytes: 2,
            activation_element_bytes: 2,
            io_weight_element_bytes: 4,
            accumulator_element_bytes: 4,
            device_memory_is_host: false,
            use_flash_attention: false,
            normalization_f32_buffer_count: 3,
            softmax_f32_buffer_count: 4,
            modulation_buffer_count: 5,
            pipeline_latent_buffer_count: 3,
            evaluation_count: H3_DEFAULT_EVALUATION_COUNT,
            precompute_adaln_steps: H3_DEFAULT_EVALUATION_COUNT,
            timestep_rows: 2,
            host_weight_cache_bytes: 0,
            mapped_weight_residency_bytes: 0,
            checkpoint_weight_bytes: Some(H3_BASE_CHECKPOINT_BYTES),
            peak_materialized_weight_bytes_override: None,
            device_weight_cache_bytes: 0,
            backend_workspace_bytes: 0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("weight_element_bytes", self.weight_element_bytes),
            ("timestep_rows", self.timestep_rows),
            ("activation_element_bytes", self.activation_element_bytes),
            ("io_weight_element_bytes", self.io_weight_element_bytes),
            ("accumulator_element_bytes", self.accumulator_element_bytes),
            (
                "normalization_f32_buffer_count",
                self.normalization_f32_buffer_count,
            ),
            ("softmax_f32_buffer_count", self.softmax_f32_buffer_count),
            ("modulation_buffer_count", self.modulation_buffer_count),
            (
                "pipeline_latent_buffer_count",
                self.pipeline_latent_buffer_count,
            ),
        ] {
            anyhow::ensure!(value > 0, "{name} must be non-zero");
        }
        anyhow::ensure!(
            self.accumulator_element_bytes >= self.activation_element_bytes,
            "accumulator elements must not be narrower than activations"
        );
        anyhow::ensure!(
            self.evaluation_count > 0,
            "evaluation_count must be non-zero"
        );
        Ok(())
    }
}

impl Default for ResourceAssumptions {
    fn default() -> Self {
        Self::h3_bf16_mmap()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightMemoryEstimate {
    #[serde(deserialize_with = "crate::required_option")]
    pub checkpoint_bytes: Option<u64>,
    pub context_projection_stage_bytes: u64,
    pub time_input_stage_bytes: u64,
    pub latent_input_stage_bytes: u64,
    pub attention_stage_bytes: u64,
    pub feed_forward_stage_bytes: u64,
    pub adaln_stage_bytes: u64,
    pub output_stage_bytes: u64,
    pub derived_peak_materialized_bytes: u64,
    pub peak_materialized_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationMemoryEstimate {
    pub hidden_state_bytes: u64,
    pub normalization_f32_workspace_bytes: u64,
    pub modulation_workspace_bytes: u64,
    pub qkv_projected_bytes: u64,
    pub qkv_transposed_bytes: u64,
    pub attention_score_chunk_bytes: u64,
    pub attention_softmax_workspace_bytes: u64,
    pub flash_attention_backend_workspace_bytes: u64,
    pub attention_output_bytes: u64,
    pub ffn_projected_bytes: u64,
    pub ffn_activated_bytes: u64,
    pub ffn_output_bytes: u64,
    pub ffn_normalization_f32_workspace_bytes: u64,
    pub ffn_modulation_workspace_bytes: u64,
    pub adaln_modulation_bytes: u64,
    pub output_head_chunk_rows: u64,
    pub output_head_hidden_chunk_bytes: u64,
    pub output_head_normalization_f32_workspace_bytes: u64,
    pub output_head_modulation_f32_workspace_bytes: u64,
    pub output_head_modulation_table_bytes: u64,
    pub output_head_projected_bytes: u64,
    pub output_head_working_set_bytes: u64,
    pub adaln_schedule_cache_bytes: u64,
    pub prompt_embedding_bytes: u64,
    pub refined_text_cache_bytes: u64,
    pub rotary_cache_bytes: u64,
    pub packed_layout_bytes: u64,
    pub latent_state_bytes: u64,
    pub pipeline_latent_working_set_bytes: u64,
    pub static_context_bytes: u64,
    pub persistent_pipeline_bytes: u64,
    pub attention_working_set_bytes: u64,
    pub feed_forward_working_set_bytes: u64,
    pub adaln_working_set_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputeTrafficEstimate {
    pub evaluation_count: u64,
    pub attention_qk_flops_per_evaluation: u64,
    pub attention_pv_flops_per_evaluation: u64,
    pub attention_projection_flops_per_evaluation: u64,
    pub ffn_flops_per_evaluation: u64,
    pub total_flops_per_evaluation: u64,
    pub total_schedule_flops: u64,
    pub transformer_weight_bytes_per_evaluation_without_adaln_precompute: u64,
    pub transformer_weight_bytes_per_evaluation_with_adaln_precompute: u64,
    pub transformer_weight_materialization_bytes_without_adaln_precompute: u64,
    pub transformer_weight_materialization_bytes_with_adaln_precompute: u64,
    pub configured_transformer_weight_materialization_bytes: u64,
    pub adaln_precompute_saved_weight_materialization_bytes: u64,
    pub configured_precomputed_evaluation_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEstimate {
    pub schema_version: u32,
    pub geometry: T2vaGeometry,
    pub model: TransformerShape,
    pub assumptions: ResourceAssumptions,
    pub sequence_rows: SequenceRows,
    pub weights: WeightMemoryEstimate,
    pub activations: ActivationMemoryEstimate,
    pub compute_and_traffic: ComputeTrafficEstimate,
    pub host_runtime_workspace_bytes: u64,
    pub peak_host_bytes: u64,
    pub attention_stage_peak_device_bytes: u64,
    pub feed_forward_stage_peak_device_bytes: u64,
    pub adaln_stage_peak_device_bytes: u64,
    pub output_stage_peak_device_bytes: u64,
    pub peak_device_bytes: u64,
}

impl ResourceEstimate {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == RESOURCE_ESTIMATE_SCHEMA_VERSION,
            "unsupported resource-estimate schema {}; this build supports schema {}",
            self.schema_version,
            RESOURCE_ESTIMATE_SCHEMA_VERSION
        );
        let recomputed = Self::for_shape(self.model, self.geometry, self.assumptions)?;
        anyhow::ensure!(
            recomputed == *self,
            "resource estimate does not match a fresh recomputation"
        );
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let estimate: Self =
            serde_json::from_slice(bytes).context("invalid resource-estimate JSON")?;
        estimate.validate()?;
        Ok(estimate)
    }
    pub fn for_t2va(
        config: &TransformerConfig,
        geometry: T2vaGeometry,
        assumptions: ResourceAssumptions,
    ) -> Result<Self> {
        config.validate()?;
        Self::for_shape(TransformerShape::from_config(config), geometry, assumptions)
    }

    pub fn for_shape(
        model: TransformerShape,
        geometry: T2vaGeometry,
        assumptions: ResourceAssumptions,
    ) -> Result<Self> {
        let sequence_rows = geometry.sequence_rows(model.patch_size)?;
        Self::for_shape_rows(model, geometry, assumptions, sequence_rows)
    }

    /// Extend the target geometry with actual conditioned packed rows. Input
    /// latents already loaded before capture are not charged twice; the packed
    /// hidden/attention/output workspaces cover all target and condition rows.
    pub fn for_shape_rows(
        model: TransformerShape,
        geometry: T2vaGeometry,
        assumptions: ResourceAssumptions,
        sequence_rows: SequenceRows,
    ) -> Result<Self> {
        model.validate()?;
        assumptions.validate()?;
        let target = geometry.sequence_rows(model.patch_size)?;
        anyhow::ensure!(
            sequence_rows.text == target.text
                && sequence_rows.video >= target.video
                && sequence_rows.audio >= target.audio
                && sequence_rows.patched_video_frames == target.patched_video_frames
                && sequence_rows.video_rows_per_frame == target.video_rows_per_frame,
            "packed rows disagree with target geometry"
        );
        anyhow::ensure!(
            sequence_rows
                .text
                .checked_add(sequence_rows.video)
                .and_then(|v| v.checked_add(sequence_rows.audio))
                == Some(sequence_rows.total),
            "packed row counts do not sum to the declared total"
        );
        let weights = estimate_weights(model, assumptions)?;
        let activations = estimate_activations(model, geometry, sequence_rows, assumptions)?;
        let compute_and_traffic =
            estimate_compute_and_traffic(model, sequence_rows, weights, assumptions)?;

        let host_runtime_workspace_bytes = checked_sum(
            "host layout workspace",
            &[
                bytes("host positions", sequence_rows.total, 3 * 8)?,
                bytes("host token tags", sequence_rows.total, 4)?,
                bytes("host modality indices", sequence_rows.total, 4)?,
                bytes("host timestep rows", sequence_rows.total, 4)?,
                bytes("host timestep indices", sequence_rows.total, 4)?,
                bytes("host text token tags", sequence_rows.text, 4)?,
            ],
        )?;
        let host_only_peak_bytes = checked_sum(
            "host-only peak memory",
            &[
                assumptions.host_weight_cache_bytes,
                assumptions.mapped_weight_residency_bytes,
                host_runtime_workspace_bytes,
                activations.adaln_schedule_cache_bytes,
            ],
        )?;

        let retained_and_workspace = checked_sum(
            "device weight retention plus backend workspace",
            &[
                assumptions.device_weight_cache_bytes,
                assumptions.backend_workspace_bytes,
            ],
        )?;
        let attention_stage_peak_device_bytes = stage_peak(
            "attention stage peak",
            activations.persistent_pipeline_bytes,
            weights.attention_stage_bytes,
            activations.attention_working_set_bytes,
            retained_and_workspace,
        )?;
        let feed_forward_stage_peak_device_bytes = stage_peak(
            "feed-forward stage peak",
            activations.persistent_pipeline_bytes,
            weights.feed_forward_stage_bytes,
            activations.feed_forward_working_set_bytes,
            retained_and_workspace,
        )?;
        let adaln_stage_peak_device_bytes = stage_peak(
            "AdaLN stage peak",
            activations.persistent_pipeline_bytes,
            weights.adaln_stage_bytes,
            activations.adaln_working_set_bytes,
            retained_and_workspace,
        )?;
        let output_stage_peak_device_bytes = stage_peak(
            "output stage peak",
            activations.persistent_pipeline_bytes,
            weights.output_stage_bytes,
            activations.output_head_working_set_bytes,
            retained_and_workspace,
        )?;
        let weight_only_stage_peak_device_bytes = stage_peak(
            "weight-only stage peak",
            activations.persistent_pipeline_bytes,
            weights.peak_materialized_bytes,
            0,
            retained_and_workspace,
        )?;
        let peak_device_bytes = [
            attention_stage_peak_device_bytes,
            feed_forward_stage_peak_device_bytes,
            adaln_stage_peak_device_bytes,
            output_stage_peak_device_bytes,
            weight_only_stage_peak_device_bytes,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let peak_host_bytes = if assumptions.device_memory_is_host {
            checked_sum(
                "combined CPU host memory",
                &[host_only_peak_bytes, peak_device_bytes],
            )?
        } else {
            host_only_peak_bytes
        };

        Ok(Self {
            schema_version: RESOURCE_ESTIMATE_SCHEMA_VERSION,
            geometry,
            model,
            assumptions,
            sequence_rows,
            weights,
            activations,
            compute_and_traffic,
            host_runtime_workspace_bytes,
            peak_host_bytes,
            attention_stage_peak_device_bytes,
            feed_forward_stage_peak_device_bytes,
            adaln_stage_peak_device_bytes,
            output_stage_peak_device_bytes,
            peak_device_bytes,
        })
    }

    pub fn to_pretty_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn check_budget(&self, budget: ResourceBudget) -> BudgetReport {
        budget.check(self)
    }
}

/// Adapter-owned bridge from H3 estimates to model-independent byte budgets.
pub trait H3ResourceBudgetExt {
    fn check(self, estimate: &ResourceEstimate) -> BudgetReport;
    fn validate(self, estimate: &ResourceEstimate) -> Result<()>;
}

impl H3ResourceBudgetExt for ResourceBudget {
    fn check(self, estimate: &ResourceEstimate) -> BudgetReport {
        self.check_peaks(estimate.peak_host_bytes, estimate.peak_device_bytes)
    }

    fn validate(self, estimate: &ResourceEstimate) -> Result<()> {
        self.validate_peaks(estimate.peak_host_bytes, estimate.peak_device_bytes)
    }
}

impl fmt::Display for ResourceEstimate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "T2VA rows: {} total ({} text + {} audio + {} video)",
            self.sequence_rows.total,
            self.sequence_rows.text,
            self.sequence_rows.audio,
            self.sequence_rows.video
        )?;
        writeln!(
            formatter,
            "Peak materialized weights: {}",
            format_bytes(self.weights.peak_materialized_bytes)
        )?;
        writeln!(
            formatter,
            "Attention projection/query chunks: {} / {} rows; key chunk: {}",
            self.geometry.attention_projection_chunk_size,
            self.geometry.attention_query_chunk_size,
            self.geometry.attention_key_chunk_policy,
        )?;
        if self.assumptions.use_flash_attention {
            writeln!(
                formatter,
                "Attention strategy: streamed FlashAttention (complete K/V, chunked Q/output)"
            )?;
            writeln!(
                formatter,
                "Complete K/V + Q chunk: {}",
                format_bytes(self.activations.qkv_projected_bytes)
            )?;
            writeln!(
                formatter,
                "FlashAttention backend statistics workspace: {}",
                format_bytes(self.activations.flash_attention_backend_workspace_bytes)
            )?;
        } else {
            if self.geometry.attention_key_chunk_policy.is_full() {
                writeln!(formatter, "Attention strategy: materialized full softmax")?;
            } else {
                writeln!(formatter, "Attention strategy: online key-chunked softmax")?;
            }
            writeln!(
                formatter,
                "Chunked K/V/Q projections / prepared K/V + Q chunk: {} / {}",
                format_bytes(self.activations.qkv_projected_bytes),
                format_bytes(self.activations.qkv_transposed_bytes)
            )?;
            writeln!(
                formatter,
                "Attention score chunk/softmax workspace: {} / {}",
                format_bytes(self.activations.attention_score_chunk_bytes),
                format_bytes(self.activations.attention_softmax_workspace_bytes)
            )?;
        }
        writeln!(
            formatter,
            "Attention output-side residency: {}",
            format_bytes(self.activations.attention_output_bytes)
        )?;
        writeln!(
            formatter,
            "FFN projected/activated: {} / {}",
            format_bytes(self.activations.ffn_projected_bytes),
            format_bytes(self.activations.ffn_activated_bytes)
        )?;
        writeln!(
            formatter,
            "Output head configured/effective chunk, working set, stage peak: {}/{} rows / {} / {}",
            self.geometry.output_token_chunk_size,
            self.activations.output_head_chunk_rows,
            format_bytes(self.activations.output_head_working_set_bytes),
            format_bytes(self.output_stage_peak_device_bytes),
        )?;
        writeln!(
            formatter,
            "Modeled FLOPs/evaluation: {} (QK {} + PV {} + attention projections {} + FFN {})",
            format_flops(self.compute_and_traffic.total_flops_per_evaluation),
            format_flops(self.compute_and_traffic.attention_qk_flops_per_evaluation),
            format_flops(self.compute_and_traffic.attention_pv_flops_per_evaluation),
            format_flops(
                self.compute_and_traffic
                    .attention_projection_flops_per_evaluation
            ),
            format_flops(self.compute_and_traffic.ffn_flops_per_evaluation),
        )?;
        writeln!(
            formatter,
            "Modeled full-attention schedule: {} across {} evaluations (chunking does not reduce FLOPs)",
            format_flops(self.compute_and_traffic.total_schedule_flops),
            self.compute_and_traffic.evaluation_count,
        )?;
        writeln!(
            formatter,
            "Transformer block weight materialization without/with full AdaLN precompute: {} / {}",
            format_bytes(
                self.compute_and_traffic
                    .transformer_weight_materialization_bytes_without_adaln_precompute
            ),
            format_bytes(
                self.compute_and_traffic
                    .transformer_weight_materialization_bytes_with_adaln_precompute
            ),
        )?;
        writeln!(
            formatter,
            "Configured block weight materialization ({} cached evaluations): {}",
            self.compute_and_traffic
                .configured_precomputed_evaluation_count,
            format_bytes(
                self.compute_and_traffic
                    .configured_transformer_weight_materialization_bytes
            ),
        )?;
        writeln!(
            formatter,
            "Host AdaLN schedule cache: {}",
            format_bytes(self.activations.adaln_schedule_cache_bytes)
        )?;
        writeln!(
            formatter,
            "{}: {} / {}",
            if self.assumptions.device_memory_is_host {
                "Conservative CPU host peak / execution working set"
            } else {
                "Conservative peak host/device"
            },
            format_bytes(self.peak_host_bytes),
            format_bytes(self.peak_device_bytes)
        )?;
        if let Some(checkpoint_bytes) = self.weights.checkpoint_bytes {
            write!(
                formatter,
                "Full mapped checkpoint payload (configured mapping-residency allowance: {}): {}",
                format_bytes(self.assumptions.mapped_weight_residency_bytes),
                format_bytes(checkpoint_bytes)
            )?;
        }
        Ok(())
    }
}

pub fn format_flops(flops: u64) -> String {
    const UNITS: [&str; 7] = [
        "FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs",
    ];
    if flops < 1_000 {
        return format!("{flops} FLOPs");
    }
    let mut value = flops as f64;
    let mut unit = 0usize;
    while value >= 1_000. && unit + 1 < UNITS.len() {
        value /= 1_000.;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

fn stage_peak(
    label: &'static str,
    persistent: u64,
    weight: u64,
    working: u64,
    backend_workspace: u64,
) -> Result<u64> {
    checked_sum(label, &[persistent, weight, working, backend_workspace])
}

fn bytes(label: &'static str, elements: u64, element_bytes: u64) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .with_context(|| format!("{label} byte count overflow"))
}

fn checked_product(label: &'static str, factors: &[u64]) -> Result<u64> {
    factors.iter().try_fold(1u64, |product, &factor| {
        product
            .checked_mul(factor)
            .with_context(|| format!("{label} overflow"))
    })
}

fn checked_sum(label: &'static str, values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |sum, &value| {
        sum.checked_add(value)
            .with_context(|| format!("{label} overflow"))
    })
}

fn as_u64(value: usize, label: &'static str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{label} does not fit in u64"))
}

/// The `ModelRequirement` view of one T2VA request: the per-evaluation re-read
/// set is the steady demand, the AdaLN projections and the text encoder
/// stream once.
pub struct H3T2vaRequirement {
    steady_weight_bytes: u64,
    single_pass_weight_bytes: u64,
    memory_materialization_bytes: u64,
    activation_peak_bytes: u64,
    flops_per_evaluation: u64,
    chunk_ladder: Vec<(ff_core::configure::ChunkPlan, u64)>,
}

/// The chunk ladder the deriver searches, largest last. The top entry is the
/// measured performance optimum of the 2026-09 chunk scan; larger entries fit
/// more devices but compute slower.
const CHUNK_LADDER: [(usize, usize, usize); 4] = [
    (512, 128, 512),
    (1024, 256, 1024),
    (2048, 512, 1024),
    (4096, 1024, 1024),
];

impl H3T2vaRequirement {
    pub fn from_estimate(
        estimate: &ResourceEstimate,
        encoder_weight_bytes: u64,
        flash_attention: bool,
        backend_workspace_bytes: u64,
        materialization_override: Option<u64>,
    ) -> Result<Self> {
        let traffic = &estimate.compute_and_traffic;
        let steady = traffic.transformer_weight_bytes_per_evaluation_with_adaln_precompute;
        let single_pass = traffic
            .transformer_weight_bytes_per_evaluation_without_adaln_precompute
            .saturating_sub(steady)
            .saturating_add(encoder_weight_bytes);
        let mut assumptions = estimate.assumptions;
        assumptions.use_flash_attention = flash_attention;
        let mut chunk_ladder = Vec::with_capacity(CHUNK_LADDER.len());
        for (projection, feed_forward, output) in CHUNK_LADDER {
            let mut geometry = estimate.geometry;
            geometry.attention_projection_chunk_size = projection;
            geometry.attention_query_chunk_size = projection;
            geometry.ffn_token_chunk_size = feed_forward;
            geometry.output_token_chunk_size = output;
            let entry = ResourceEstimate::for_shape(estimate.model, geometry, assumptions)?;
            chunk_ladder.push((
                ff_core::configure::ChunkPlan {
                    attention_projection: projection,
                    feed_forward,
                    output,
                },
                entry
                    .peak_device_bytes
                    .saturating_add(backend_workspace_bytes),
            ));
        }
        Ok(Self {
            steady_weight_bytes: steady,
            single_pass_weight_bytes: single_pass,
            memory_materialization_bytes: materialization_override.unwrap_or(
                estimate
                    .weights
                    .checkpoint_bytes
                    .context("resource estimate carries no transformer checkpoint size")?,
            ),
            activation_peak_bytes: chunk_ladder
                .last()
                .map(|(_, peak)| *peak)
                .context("the chunk ladder is empty")?,
            flops_per_evaluation: traffic.total_flops_per_evaluation,
            chunk_ladder,
        })
    }
}

impl ff_core::configure::ModelRequirement for H3T2vaRequirement {
    fn steady_weight_bytes(&self) -> Result<u64> {
        Ok(self.steady_weight_bytes)
    }

    fn activation_peak_bytes(&self) -> Result<u64> {
        Ok(self.activation_peak_bytes)
    }

    fn single_pass_weight_bytes(&self) -> Result<u64> {
        Ok(self.single_pass_weight_bytes)
    }

    fn flops_per_evaluation(&self) -> Result<u64> {
        Ok(self.flops_per_evaluation)
    }

    fn largest_chunk_plan_within(
        &self,
        activation_budget_bytes: u64,
    ) -> Result<Option<ff_core::configure::SelectedChunkPlan>> {
        Ok(self
            .chunk_ladder
            .iter()
            .filter(|(_, peak)| *peak <= activation_budget_bytes)
            .max_by_key(|(plan, _)| plan.feed_forward)
            // Nothing fits: hand over the smallest plan rather than the
            // caller's static defaults, which are the most expensive ones.
            // Admission still sees the real peak and refuses honestly.
            .or_else(|| self.chunk_ladder.first())
            .map(|(plan, peak)| ff_core::configure::SelectedChunkPlan {
                plan: *plan,
                peak_device_bytes: *peak,
            }))
    }

    fn memory_materialization_bytes(&self) -> Result<u64> {
        Ok(self.memory_materialization_bytes)
    }
}

#[cfg(test)]
mod tests;
