use crate::core::AttentionKeyChunkPolicy;
#[cfg(test)]
use crate::core::{DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE, DEFAULT_ATTENTION_QUERY_CHUNK_SIZE};
use crate::{
    config::TransformerConfig,
    core::{self, AttentionChunking, DEFAULT_FFN_TOKEN_CHUNK_SIZE, MODALITY_COUNT},
    embeddings,
    execution::{ExecutionStage, H3ExecutionPlan, StageKind},
    layout::{AUDIO_TAG, VIDEO_TAG},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use ff_core::{
    residency::{DeviceResidencyPlan, PhaseResidencyDemand, WeightPhase, plan_device_residency},
    weights::{
        CachePolicy, CacheStats, DeviceCache, DeviceCachePolicy, DeviceCacheStats, ModelWeights,
        WeightAccessStats, WeightSource,
    },
};
use std::{collections::BTreeMap, num::NonZeroUsize, ops::Range, path::Path};

pub const DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE: usize = 256;
pub const H3_FLASH_ATTENTION_BACKEND: &str = core::H3_FLASH_ATTENTION_BACKEND;
pub const H3_OUTPUT_HEAD_ORDER_BACKEND: &str =
    "both-heads-on-contiguous-full-packed-chunks-then-modality-select-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformerChunking {
    pub attention: AttentionChunking,
    pub feed_forward_chunk_size: NonZeroUsize,
    pub output_chunk_size: NonZeroUsize,
}

impl Default for TransformerChunking {
    fn default() -> Self {
        Self {
            attention: AttentionChunking::default(),
            feed_forward_chunk_size: NonZeroUsize::new(DEFAULT_FFN_TOKEN_CHUNK_SIZE)
                .expect("default feed-forward chunk size is non-zero"),
            output_chunk_size: NonZeroUsize::new(DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE)
                .expect("default output chunk size is non-zero"),
        }
    }
}

pub struct TransformerStaticInputs<'a> {
    pub encoder_hidden_states: &'a Tensor,
    pub token_tags: &'a Tensor,
    pub position_ids: &'a Tensor,
    pub video_indices: &'a Tensor,
    pub audio_indices: &'a Tensor,
    pub text_indices: &'a Tensor,
}

pub struct TransformerStepInputs<'a> {
    pub video_hidden_states: &'a Tensor,
    pub audio_hidden_states: &'a Tensor,
    pub timestep: &'a Tensor,
    pub timestep_indices: &'a Tensor,
}

#[derive(Clone)]
pub struct PreparedTransformerContext {
    refined_text: Tensor,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
    token_tags: Tensor,
    video_indices: Tensor,
    audio_indices: Tensor,
    text_indices: Tensor,
    sequence_length: usize,
    canonical_layout_order: bool,
}

pub struct PreparedDenoiseSchedule {
    steps: Vec<PreparedDenoiseStep>,
}

struct PreparedDenoiseStep {
    timesteps: Vec<f32>,
    timestep_embedding: Tensor,
    modulations: Vec<core::AdaLnModulation>,
}

impl PreparedDenoiseSchedule {
    pub fn steps(&self) -> usize {
        self.steps.len()
    }
}

impl PreparedTransformerContext {
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn text_rows(&self) -> usize {
        self.text_indices.elem_count()
    }
}

/// Whether the host could still be holding `bytes` by the time a scan of that
/// size comes back to its start. An unreadable memory figure answers `true`,
/// which leaves the kernel's ordinary behaviour in place rather than acting on
/// a number this process does not have.
fn host_can_retain(bytes: u64) -> bool {
    let snapshot = ff_core::probe::ResourceSnapshot::capture(None);
    snapshot
        .cgroup_v2_memory_available_bytes
        .into_iter()
        .chain(snapshot.host_memory_available_bytes)
        .min()
        .is_none_or(|available| bytes <= available)
}

pub struct TransformerOutput {
    pub video: Tensor,
    pub audio: Tensor,
}

struct ProjectedStepInputs {
    video: Tensor,
    audio: Tensor,
    timestep: Tensor,
}

struct BlockExecutionContext<'a> {
    adaln_indices: &'a Tensor,
    rotary_cos: &'a Tensor,
    rotary_sin: &'a Tensor,
}

pub struct StreamedTransformer {
    weights: ModelWeights,
    config: TransformerConfig,
    plan: H3ExecutionPlan,
    device: Device,
    chunking: TransformerChunking,
    #[cfg(feature = "flash-attn")]
    flash_attention: bool,
    device_residency_plan: Option<DeviceResidencyPlan>,
    host_residency_plan: Option<DeviceResidencyPlan>,
}

#[derive(Clone, Debug)]
pub struct StreamedTransformerOptions {
    pub weight_source: WeightSource,
    pub cache_policy: CachePolicy,
    pub device: Device,
    pub chunking: TransformerChunking,
    pub flash_attention: bool,
    /// Explicit optional retention. The caller must reserve activation and
    /// workspace capacity separately; default remains streaming.
    pub device_cache_policy: DeviceCachePolicy,
    /// Protect recurrent raw weights in a bounded tensor-granularity host cache.
    pub host_phase_priority: bool,
}

#[cfg(feature = "cuda")]
fn validate_h3_cuda_numerical_ranges(
    flash_attention: bool,
    key_policy: AttentionKeyChunkPolicy,
    sequence_length: usize,
    context_projection_rows: usize,
    timestep_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        (1..=crate::cuda::linear::MAX_CONTEXT_ROWS).contains(&context_projection_rows),
        "CUDA BF16 H3 context projection supports 1..={} total projection rows with the \
         verified cuBLASLt contract; request has {context_projection_rows}",
        crate::cuda::linear::MAX_CONTEXT_ROWS
    );
    anyhow::ensure!(
        (1..=crate::cuda::linear::MAX_TIMESTEP_ROWS).contains(&timestep_rows),
        "CUDA BF16 H3 timestep projections support 1..={} distinct rows with the verified \
         cuBLASLt contract; request may use {timestep_rows}",
        crate::cuda::linear::MAX_TIMESTEP_ROWS
    );
    if !flash_attention {
        anyhow::ensure!(
            context_projection_rows <= core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS,
            "CUDA BF16 token refiner supports at most {} context projection rows with the \
             verified exact-softmax backends; request has {context_projection_rows}. \
             Enable --flash-attention instead",
            core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS
        );
    }
    if !flash_attention && key_policy.is_full() {
        anyhow::ensure!(
            sequence_length <= core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS,
            "CUDA BF16 full-softmax supports at most {} packed rows with the verified \
             PyTorch exact backends; request has {sequence_length}. Enable \
             --flash-attention instead",
            core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn validate_h3_numerical_backend(
    device: &Device,
    flash_attention: bool,
    key_policy: AttentionKeyChunkPolicy,
    sequence_length: usize,
    context_projection_rows: usize,
    timestep_rows: usize,
) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    if flash_attention {
        crate::cuda::validate_flash_attention_device(device)?;
    }
    if !crate::cuda::profile::reference_libraries_available(device) {
        return Ok(());
    }
    crate::cuda::profile::validate_exact_profile(device)?;
    validate_h3_cuda_numerical_ranges(
        flash_attention,
        key_policy,
        sequence_length,
        context_projection_rows,
        timestep_rows,
    )
}

#[cfg(not(feature = "cuda"))]
pub fn validate_h3_numerical_backend(
    device: &Device,
    _flash_attention: bool,
    _key_policy: AttentionKeyChunkPolicy,
    _sequence_length: usize,
    _context_projection_rows: usize,
    _timestep_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        !device.is_cuda(),
        "H3 CUDA numerical backend validation requires the cuda build feature"
    );
    Ok(())
}

impl StreamedTransformerOptions {
    pub fn new(weight_source: WeightSource, cache_policy: CachePolicy, device: Device) -> Self {
        Self {
            weight_source,
            cache_policy,
            device,
            chunking: TransformerChunking::default(),
            flash_attention: false,
            device_cache_policy: DeviceCachePolicy::DISABLED,
            host_phase_priority: false,
        }
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.host_phase_priority
                || (self.cache_policy.granularity == ff_core::weights::CacheGranularity::Tensor
                    && self.cache_policy.max_bytes.is_some()),
            "host phase priority requires a byte-bounded tensor cache"
        );
        if self.flash_attention {
            anyhow::ensure!(
                self.chunking.attention.key.is_full(),
                "FlashAttention and key-chunked online softmax are mutually exclusive"
            );
            crate::cuda::validate_flash_attention_device(&self.device)?;
        }
        Ok(())
    }
}

impl StreamedTransformer {
    pub fn open(
        component_dir: impl AsRef<Path>,
        options: StreamedTransformerOptions,
    ) -> Result<Self> {
        options.validate()?;
        let component_dir = component_dir.as_ref();
        let config = TransformerConfig::from_file(component_dir.join("config.json"))?;
        config.validate()?;
        #[cfg(feature = "cuda")]
        if options.device.is_cuda() {
            anyhow::ensure!(
                config.hidden_size == 5_376 && config.attention_head_dim == 128,
                "exact H3 CUDA RMSNorm requires released hidden/head widths 5376/128, got {}/{}",
                config.hidden_size,
                config.attention_head_dim
            );
        }
        let weights =
            ModelWeights::open(component_dir, options.weight_source, options.cache_policy)?;
        Self::from_validated_parts(weights, config, options)
    }

    fn from_validated_parts(
        mut weights: ModelWeights,
        config: TransformerConfig,
        options: StreamedTransformerOptions,
    ) -> Result<Self> {
        let plan = H3ExecutionPlan::from_config(&weights, &config)?;
        // Drop evicted pages when evaluation weights exceed available host memory.
        weights.drop_evicted_pages(!host_can_retain(plan.evaluation_weight_bytes()));
        let phases = if options.device_cache_policy.is_enabled() || options.host_phase_priority {
            plan.stages()
                .iter()
                .filter(|stage| {
                    matches!(
                        stage.kind,
                        StageKind::LatentInput
                            | StageKind::BlockAttention(_)
                            | StageKind::BlockFeedForward(_)
                            | StageKind::Output
                    )
                })
                .map(|stage| {
                    WeightPhase::new(stage.kind.to_string(), stage.tensor_names.clone(), 1)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let demands = phases
            .iter()
            .map(|phase| PhaseResidencyDemand::from_phase(phase, &weights, &options.device))
            .collect::<Result<Vec<_>>>()?;
        let device_residency_plan = if options.device_cache_policy.is_enabled() {
            let placement =
                plan_device_residency(&demands, options.device_cache_policy.max_bytes, 0)?;
            let names = phases
                .iter()
                .filter(|phase| placement.placed.contains(&phase.name))
                .flat_map(|phase| phase.tensors.iter().cloned())
                .collect::<Vec<_>>();
            weights.configure_device_cache_with_priority(
                DeviceCache::new(options.device_cache_policy),
                names,
            )?;
            Some(placement)
        } else {
            None
        };
        let host_residency_plan = if options.host_phase_priority {
            let budget = options
                .cache_policy
                .max_bytes
                .context("host phase priority requires a byte-bounded tensor cache")?;
            let raw_demands = demands
                .iter()
                .filter(|demand| {
                    !options.device.is_cuda()
                        || device_residency_plan
                            .as_ref()
                            .is_none_or(|plan| !plan.placed.contains(&demand.name))
                })
                .cloned()
                .map(|mut demand| {
                    demand.resident_bytes = demand.transfer_bytes;
                    demand
                })
                .collect::<Vec<_>>();
            let placement = plan_device_residency(&raw_demands, budget, 0)?;
            let names = phases
                .iter()
                .filter(|phase| placement.placed.contains(&phase.name))
                .flat_map(|phase| phase.tensors.iter().cloned())
                .collect::<Vec<_>>();
            if options.device.is_cuda() && device_residency_plan.is_some() {
                weights.configure_complementary_host_cache(names)?;
            } else {
                weights.configure_host_cache_priority(names)?;
            }
            Some(placement)
        } else {
            None
        };
        Ok(Self {
            weights,
            config,
            plan,
            device: options.device,
            chunking: options.chunking,
            #[cfg(feature = "flash-attn")]
            flash_attention: options.flash_attention,
            device_residency_plan,
            host_residency_plan,
        })
    }

    pub fn plan(&self) -> &H3ExecutionPlan {
        &self.plan
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.weights.cache_stats()
    }

    pub fn access_stats(&self) -> WeightAccessStats {
        self.weights.access_stats()
    }

    pub fn device_cache_stats(&self) -> DeviceCacheStats {
        self.weights.device_cache_stats()
    }

    pub fn device_residency_plan(&self) -> Option<&DeviceResidencyPlan> {
        self.device_residency_plan.as_ref()
    }

    pub fn host_residency_plan(&self) -> Option<&DeviceResidencyPlan> {
        self.host_residency_plan.as_ref()
    }

    pub fn prepare_context(
        &self,
        inputs: &TransformerStaticInputs<'_>,
    ) -> Result<PreparedTransformerContext> {
        let canonical_layout_order = self.validate_static_inputs(inputs)?;
        let sequence_length = inputs.position_ids.dim(0)?;
        let (context_batch, context_rows, _) = inputs.encoder_hidden_states.dims3()?;
        let context_projection_rows = context_batch
            .checked_mul(context_rows)
            .context("H3 context projection row count overflows usize")?;
        let flash_attention = {
            #[cfg(feature = "flash-attn")]
            {
                self.flash_attention
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                false
            }
        };
        validate_h3_numerical_backend(
            &self.device,
            flash_attention,
            self.chunking.attention.key,
            sequence_length,
            context_projection_rows,
            1,
        )?;
        let (rotary_cos, rotary_sin) = embeddings::rotary(
            inputs.position_ids,
            self.config.rope_freq_dim,
            self.config.rope_theta,
        )?;

        let context_stage = self.required_stage(&StageKind::ContextInput)?;
        let mut refined_text = self.execute_stage(context_stage, |_, tensors| {
            linear(tensors, "context_embedder", inputs.encoder_hidden_states)
        })?;
        for layer in 0..self.config.num_refiner_layers {
            let prefix = format!("token_refiner.refiner_blocks.{layer}");
            let attention_stage = self.required_stage(&StageKind::RefinerAttention(layer))?;
            refined_text = self.execute_stage(attention_stage, |_, tensors| {
                #[cfg(feature = "flash-attn")]
                if self.flash_attention {
                    core::refiner_attention_flash(
                        tensors,
                        &prefix,
                        &refined_text,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        self.chunking.attention.query_chunk_size,
                        self.config.norm_eps,
                        self.config.qk_norm_eps,
                    )
                } else {
                    core::refiner_attention(
                        tensors,
                        &prefix,
                        &refined_text,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        self.chunking.attention.query_chunk_size,
                        self.config.norm_eps,
                        self.config.qk_norm_eps,
                    )
                }
                #[cfg(not(feature = "flash-attn"))]
                core::refiner_attention(
                    tensors,
                    &prefix,
                    &refined_text,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    self.chunking.attention.query_chunk_size,
                    self.config.norm_eps,
                    self.config.qk_norm_eps,
                )
            })?;
            let feed_forward_stage = self.required_stage(&StageKind::RefinerFeedForward(layer))?;
            refined_text = self.execute_stage(feed_forward_stage, |_, tensors| {
                core::refiner_feed_forward_chunked(
                    tensors,
                    &prefix,
                    &refined_text,
                    self.chunking.feed_forward_chunk_size,
                    self.config.norm_eps,
                )
            })?;
        }
        let final_refiner_stage = self.required_stage(&StageKind::RefinerOutputNorm)?;
        refined_text = self.execute_stage(final_refiner_stage, |_, tensors| {
            core::rms_norm(
                &refined_text,
                required(tensors, "token_refiner.final_norm.weight")?,
                self.config.final_norm_eps,
            )
        })?;

        Ok(PreparedTransformerContext {
            refined_text,
            rotary_cos,
            rotary_sin,
            token_tags: inputs.token_tags.clone(),
            video_indices: inputs.video_indices.clone(),
            audio_indices: inputs.audio_indices.clone(),
            text_indices: inputs.text_indices.clone(),
            sequence_length,
            canonical_layout_order,
        })
    }

    pub fn prepare_denoise_schedule(
        &self,
        timestep_tables: &[Vec<f32>],
    ) -> Result<PreparedDenoiseSchedule> {
        anyhow::ensure!(
            !timestep_tables.is_empty(),
            "denoising schedule must contain at least one evaluation"
        );
        for table in timestep_tables {
            anyhow::ensure!(!table.is_empty(), "timestep table must not be empty");
            anyhow::ensure!(
                table.iter().all(|value| value.is_finite()),
                "timestep table contains a non-finite value"
            );
            #[cfg(feature = "cuda")]
            if self.device.is_cuda() {
                anyhow::ensure!(
                    table.len() <= crate::cuda::linear::MAX_TIMESTEP_ROWS,
                    "CUDA BF16 H3 timestep projection supports at most {} distinct rows with \
                     the verified cuBLASLt contract; request has {}",
                    crate::cuda::linear::MAX_TIMESTEP_ROWS,
                    table.len()
                );
            }
        }

        let time_stage = self.required_stage(&StageKind::TimeInput)?;
        let timestep_embeddings = self.execute_stage(time_stage, |_, weights| {
            timestep_tables
                .iter()
                .map(|values| {
                    let timestep = Tensor::from_vec(values.clone(), values.len(), &self.device)?;
                    self.project_timestep_with_weights(weights, &timestep)?
                        .to_device(&Device::Cpu)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut steps = timestep_tables
            .iter()
            .cloned()
            .zip(timestep_embeddings)
            .map(|(timesteps, timestep_embedding)| PreparedDenoiseStep {
                timesteps,
                timestep_embedding,
                modulations: Vec::with_capacity(self.config.num_layers),
            })
            .collect::<Vec<_>>();

        for block in 0..self.config.num_layers {
            let prefix = format!("transformer_blocks.{block}");
            let stage = self.required_stage(&StageKind::BlockAdaLn(block))?;
            self.execute_stage(stage, |_, weights| {
                for step in &mut steps {
                    let timestep_embedding = step.timestep_embedding.to_device(&self.device)?;
                    let modulation = core::adaln(
                        weights,
                        &prefix,
                        &timestep_embedding,
                        self.config.hidden_size,
                    )?;
                    step.modulations
                        .push(modulation_to_device(&modulation, &Device::Cpu)?);
                }
                Ok(())
            })?;
        }
        Ok(PreparedDenoiseSchedule { steps })
    }

    pub(crate) fn forward_step(
        &self,
        prepared: &PreparedTransformerContext,
        inputs: &TransformerStepInputs<'_>,
    ) -> Result<TransformerOutput> {
        self.validate_step_inputs(prepared, inputs)?;
        let latent_stage = self.required_stage(&StageKind::LatentInput)?;
        let (video, audio) = self.execute_stage(latent_stage, |_, tensors| {
            self.project_latents(tensors, inputs)
        })?;
        let timestep = self.project_timestep(inputs.timestep)?;
        self.forward_projected(
            prepared,
            inputs,
            ProjectedStepInputs {
                video,
                audio,
                timestep,
            },
            None,
        )
    }

    pub fn forward_precomputed_step(
        &self,
        prepared: &PreparedTransformerContext,
        schedule: &PreparedDenoiseSchedule,
        step_index: usize,
        inputs: &TransformerStepInputs<'_>,
    ) -> Result<TransformerOutput> {
        self.validate_step_inputs(prepared, inputs)?;
        let step = schedule
            .steps
            .get(step_index)
            .with_context(|| format!("denoising step {step_index} is out of range"))?;
        anyhow::ensure!(
            inputs.timestep.to_vec1::<f32>()? == step.timesteps,
            "runtime timestep table differs from the prepared schedule"
        );
        let latent_stage = self.required_stage(&StageKind::LatentInput)?;
        let (video, audio) = self.execute_stage(latent_stage, |_, tensors| {
            self.project_latents(tensors, inputs)
        })?;
        let timestep = step.timestep_embedding.to_device(&self.device)?;
        self.forward_projected(
            prepared,
            inputs,
            ProjectedStepInputs {
                video,
                audio,
                timestep,
            },
            Some(&step.modulations),
        )
    }

    fn forward_projected(
        &self,
        prepared: &PreparedTransformerContext,
        inputs: &TransformerStepInputs<'_>,
        projected: ProjectedStepInputs,
        precomputed_modulations: Option<&[core::AdaLnModulation]>,
    ) -> Result<TransformerOutput> {
        let video = projected.video.to_dtype(prepared.refined_text.dtype())?;
        let audio = projected.audio.to_dtype(prepared.refined_text.dtype())?;
        let packed = pack_projected_modalities(
            &prepared.refined_text,
            &audio,
            &video,
            &prepared.text_indices,
            &prepared.audio_indices,
            &prepared.video_indices,
            prepared.sequence_length,
            prepared.canonical_layout_order,
        )?;

        let modality_count = Tensor::new(MODALITY_COUNT as u32, &self.device)?;
        let adaln_indices = inputs
            .timestep_indices
            .broadcast_mul(&modality_count)?
            .broadcast_add(&prepared.token_tags)?;
        let hidden = if let Some(modulations) = precomputed_modulations {
            self.forward_blocks_precomputed_range(
                0..self.config.num_layers,
                &packed,
                modulations,
                BlockExecutionContext {
                    adaln_indices: &adaln_indices,
                    rotary_cos: &prepared.rotary_cos,
                    rotary_sin: &prepared.rotary_sin,
                },
            )?
        } else {
            self.forward_blocks_range_in_context(
                0..self.config.num_layers,
                &packed,
                &projected.timestep,
                BlockExecutionContext {
                    adaln_indices: &adaln_indices,
                    rotary_cos: &prepared.rotary_cos,
                    rotary_sin: &prepared.rotary_sin,
                },
            )?
        };

        let output_stage = self.required_stage(&StageKind::Output)?;
        self.execute_stage(output_stage, |_, tensors| {
            self.output_heads(tensors, &hidden, &projected.timestep, prepared, inputs)
        })
    }

    pub fn forward_blocks_range(
        &self,
        block_range: Range<usize>,
        hidden_states: &Tensor,
        timestep_embedding: &Tensor,
        adaln_indices: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<Tensor> {
        self.forward_blocks_range_in_context(
            block_range,
            hidden_states,
            timestep_embedding,
            BlockExecutionContext {
                adaln_indices,
                rotary_cos,
                rotary_sin,
            },
        )
    }

    fn forward_blocks_range_in_context(
        &self,
        block_range: Range<usize>,
        hidden_states: &Tensor,
        timestep_embedding: &Tensor,
        context: BlockExecutionContext<'_>,
    ) -> Result<Tensor> {
        self.validate_block_range(&block_range)?;
        self.validate_block_inputs(
            hidden_states,
            context.adaln_indices,
            context.rotary_cos,
            context.rotary_sin,
        )?;
        self.validate_timestep_embedding(timestep_embedding)?;

        let mut hidden_states = hidden_states.clone();
        for block in block_range {
            let prefix = format!("transformer_blocks.{block}");
            let adaln_stage = self.required_stage(&StageKind::BlockAdaLn(block))?;
            let modulation = self.execute_stage(adaln_stage, |_, tensors| {
                core::adaln(
                    tensors,
                    &prefix,
                    timestep_embedding,
                    self.config.hidden_size,
                )
            })?;
            hidden_states = self.forward_block(
                block,
                &hidden_states,
                &modulation,
                context.adaln_indices,
                context.rotary_cos,
                context.rotary_sin,
            )?;
        }
        Ok(hidden_states)
    }

    fn forward_blocks_precomputed_range(
        &self,
        block_range: Range<usize>,
        hidden_states: &Tensor,
        modulations: &[core::AdaLnModulation],
        context: BlockExecutionContext<'_>,
    ) -> Result<Tensor> {
        self.validate_block_range(&block_range)?;
        self.validate_block_inputs(
            hidden_states,
            context.adaln_indices,
            context.rotary_cos,
            context.rotary_sin,
        )?;
        anyhow::ensure!(
            modulations.len() == self.config.num_layers,
            "prepared AdaLN block count differs from transformer config"
        );
        let mut hidden_states = hidden_states.clone();
        for block in block_range {
            let modulation = &modulations[block];
            let modulation = modulation_to_device(modulation, &self.device)?;
            hidden_states = self.forward_block(
                block,
                &hidden_states,
                &modulation,
                context.adaln_indices,
                context.rotary_cos,
                context.rotary_sin,
            )?;
        }
        Ok(hidden_states)
    }

    fn validate_block_range(&self, block_range: &Range<usize>) -> Result<()> {
        anyhow::ensure!(
            block_range.start < block_range.end,
            "transformer block range start {} must be smaller than end {}",
            block_range.start,
            block_range.end
        );
        anyhow::ensure!(
            block_range.end <= self.config.num_layers,
            "transformer block range [{}, {}) exceeds {} blocks",
            block_range.start,
            block_range.end,
            self.config.num_layers
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_block(
        &self,
        block: usize,
        hidden_states: &Tensor,
        modulation: &core::AdaLnModulation,
        adaln_indices: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<Tensor> {
        let prefix = format!("transformer_blocks.{block}");
        let attention_stage = self.required_stage(&StageKind::BlockAttention(block))?;
        let hidden_states = self.execute_stage(attention_stage, |_, tensors| {
            #[cfg(feature = "flash-attn")]
            if self.flash_attention {
                core::attention_flash_with_projection_chunks(
                    tensors,
                    &prefix,
                    hidden_states,
                    modulation,
                    adaln_indices,
                    rotary_cos,
                    rotary_sin,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    self.chunking.attention,
                    self.config.norm_eps,
                    self.config.qk_norm_eps,
                )
            } else {
                core::attention_with_projection_chunks(
                    tensors,
                    &prefix,
                    hidden_states,
                    modulation,
                    adaln_indices,
                    rotary_cos,
                    rotary_sin,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    self.chunking.attention,
                    self.config.norm_eps,
                    self.config.qk_norm_eps,
                )
            }
            #[cfg(not(feature = "flash-attn"))]
            core::attention_with_projection_chunks(
                tensors,
                &prefix,
                hidden_states,
                modulation,
                adaln_indices,
                rotary_cos,
                rotary_sin,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                self.chunking.attention,
                self.config.norm_eps,
                self.config.qk_norm_eps,
            )
        })?;
        let feed_forward_stage = self.required_stage(&StageKind::BlockFeedForward(block))?;
        self.execute_stage(feed_forward_stage, |_, tensors| {
            core::feed_forward_chunked(
                tensors,
                &prefix,
                &hidden_states,
                modulation,
                adaln_indices,
                self.chunking.feed_forward_chunk_size,
                self.config.norm_eps,
            )
        })
    }

    fn validate_block_inputs(
        &self,
        hidden_states: &Tensor,
        adaln_indices: &Tensor,
        rotary_cos: &Tensor,
        rotary_sin: &Tensor,
    ) -> Result<()> {
        let (batch, sequence, hidden) = hidden_states
            .dims3()
            .context("hidden states must be [batch, sequence, hidden]")?;
        anyhow::ensure!(batch > 0 && sequence > 0, "hidden states must be non-empty");
        anyhow::ensure!(
            hidden == self.config.hidden_size,
            "hidden-state width is {hidden}, expected {}",
            self.config.hidden_size
        );
        anyhow::ensure!(
            adaln_indices.dims() == [sequence] && adaln_indices.dtype() == DType::U32,
            "AdaLN indices must have one entry per sequence row"
        );
        anyhow::ensure!(
            rotary_cos.dims() == rotary_sin.dims(),
            "rotary cosine/sine shapes differ"
        );
        anyhow::ensure!(
            rotary_cos.dim(0)? == sequence,
            "rotary tables must have one row per sequence row"
        );
        anyhow::ensure!(
            rotary_cos.rank() == 2
                && rotary_cos.dim(1)?.is_multiple_of(2)
                && rotary_cos.dim(1)? <= self.config.attention_head_dim,
            "rotary tables have an invalid width"
        );
        anyhow::ensure!(
            rotary_cos.dtype() == rotary_sin.dtype()
                && matches!(rotary_cos.dtype(), DType::F32 | DType::F16 | DType::BF16),
            "rotary tables must share a floating-point dtype"
        );
        anyhow::ensure!(
            matches!(hidden_states.dtype(), DType::F32 | DType::F16 | DType::BF16),
            "hidden states must use a floating-point dtype"
        );
        for (name, tensor) in [
            ("hidden states", hidden_states),
            ("AdaLN indices", adaln_indices),
            ("rotary cosine", rotary_cos),
            ("rotary sine", rotary_sin),
        ] {
            anyhow::ensure!(
                tensor.device().same_device(&self.device),
                "{name} are on the wrong device"
            );
        }
        Ok(())
    }

    fn validate_timestep_embedding(&self, timestep_embedding: &Tensor) -> Result<()> {
        let (rows, width) = timestep_embedding
            .dims2()
            .context("timestep embedding must be [timesteps, width]")?;
        anyhow::ensure!(rows > 0, "timestep embedding must be non-empty");
        anyhow::ensure!(
            width == self.config.time_embed_dim,
            "timestep embedding width is {width}, expected {}",
            self.config.time_embed_dim
        );
        anyhow::ensure!(
            matches!(
                timestep_embedding.dtype(),
                DType::F32 | DType::F16 | DType::BF16
            ),
            "timestep embedding must use a floating-point dtype"
        );
        anyhow::ensure!(
            timestep_embedding.device().same_device(&self.device),
            "timestep embedding is on the wrong device"
        );
        Ok(())
    }

    fn project_latents(
        &self,
        weights: &BTreeMap<String, Tensor>,
        inputs: &TransformerStepInputs<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let video = linear(weights, "proj_in", inputs.video_hidden_states)?;
        let audio = linear(weights, "audio_proj_in", inputs.audio_hidden_states)?;
        Ok((video, audio))
    }

    fn project_timestep(&self, timestep: &Tensor) -> Result<Tensor> {
        let stage = self.required_stage(&StageKind::TimeInput)?;
        self.execute_stage(stage, |_, weights| {
            self.project_timestep_with_weights(weights, timestep)
        })
    }

    fn project_timestep_with_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let timestep = embeddings::timestep_projection(timestep, self.config.freq_dim)?;
        let timestep = linear(weights, "time_embedder.linear_1", &timestep)?;
        let timestep = ops::silu(&timestep)?;
        linear(weights, "time_embedder.linear_2", &timestep)
    }

    fn output_heads(
        &self,
        weights: &BTreeMap<String, Tensor>,
        hidden: &Tensor,
        timestep: &Tensor,
        prepared: &PreparedTransformerContext,
        inputs: &TransformerStepInputs<'_>,
    ) -> Result<TransformerOutput> {
        let modulation = linear(weights, "norm_out.linear", &ops::silu(timestep)?)?;
        anyhow::ensure!(
            modulation.dim(1)? == 2 * self.config.hidden_size,
            "output modulation has wrong width"
        );
        let shift = modulation
            .narrow(1, 0, self.config.hidden_size)?
            .contiguous()?;
        let scale = modulation
            .narrow(1, self.config.hidden_size, self.config.hidden_size)?
            .contiguous()?;
        let norm_weight = required(weights, "norm_out.norm.weight")?;
        project_output_full_sequence_chunks(
            weights,
            hidden,
            &prepared.video_indices,
            &prepared.audio_indices,
            inputs.timestep_indices,
            &shift,
            &scale,
            norm_weight,
            self.chunking.output_chunk_size,
            self.config.final_norm_eps,
        )
    }

    fn validate_static_inputs(&self, inputs: &TransformerStaticInputs<'_>) -> Result<bool> {
        let (text_batch, text_rows, text_width) = inputs.encoder_hidden_states.dims3()?;
        anyhow::ensure!(text_batch > 0, "text batch must be non-zero");
        anyhow::ensure!(
            text_width == self.config.text_dim,
            "text width is {text_width}, expected {}",
            self.config.text_dim
        );
        let (sequence, axes) = inputs.position_ids.dims2()?;
        anyhow::ensure!(axes == 3, "position IDs must be [sequence, 3]");
        anyhow::ensure!(
            inputs.token_tags.dims() == [sequence],
            "token tags must be [sequence]"
        );
        anyhow::ensure!(
            inputs.text_indices.dims() == [text_rows],
            "text index count differs from text rows"
        );
        for (name, tensor) in [
            ("token_tags", inputs.token_tags),
            ("video_indices", inputs.video_indices),
            ("audio_indices", inputs.audio_indices),
            ("text_indices", inputs.text_indices),
        ] {
            anyhow::ensure!(tensor.dtype() == DType::U32, "{name} must use U32 indices");
            anyhow::ensure!(
                tensor.device().same_device(&self.device),
                "{name} is on the wrong device"
            );
        }
        let token_tags = inputs.token_tags.to_vec1::<u32>()?;
        let video_indices = inputs.video_indices.to_vec1::<u32>()?;
        let audio_indices = inputs.audio_indices.to_vec1::<u32>()?;
        let text_indices = inputs.text_indices.to_vec1::<u32>()?;
        anyhow::ensure!(
            token_tags.iter().all(|&tag| tag < MODALITY_COUNT as u32),
            "a token modality tag is outside H3's three modalities"
        );
        let mut occupied = vec![false; sequence];
        for (name, indices) in [
            ("video_indices", video_indices.as_slice()),
            ("audio_indices", audio_indices.as_slice()),
            ("text_indices", text_indices.as_slice()),
        ] {
            for &index in indices {
                let index = index as usize;
                anyhow::ensure!(index < sequence, "{name} contains an out-of-range row");
                anyhow::ensure!(
                    !occupied[index],
                    "packed modality row {index} occurs more than once"
                );
                occupied[index] = true;
            }
        }
        anyhow::ensure!(
            occupied.iter().all(|value| *value),
            "packed modality indices leave an unassigned row"
        );
        anyhow::ensure!(
            video_indices
                .iter()
                .all(|&index| token_tags[index as usize] == VIDEO_TAG),
            "a video index points to a non-video token tag"
        );
        anyhow::ensure!(
            audio_indices
                .iter()
                .all(|&index| token_tags[index as usize] == AUDIO_TAG),
            "an audio index points to a non-audio token tag"
        );
        for (name, tensor) in [
            ("text", inputs.encoder_hidden_states),
            ("position_ids", inputs.position_ids),
        ] {
            anyhow::ensure!(
                tensor.device().same_device(&self.device),
                "{name} is on the wrong device"
            );
        }
        anyhow::ensure!(
            inputs.video_indices.dim(0)? + inputs.audio_indices.dim(0)? + text_rows == sequence,
            "modality rows do not exactly cover the packed sequence"
        );
        let audio_start = text_indices.len();
        let video_start = audio_start + audio_indices.len();
        Ok(indices_are_contiguous(&text_indices, 0)
            && indices_are_contiguous(&audio_indices, audio_start)
            && indices_are_contiguous(&video_indices, video_start))
    }

    fn validate_step_inputs(
        &self,
        prepared: &PreparedTransformerContext,
        inputs: &TransformerStepInputs<'_>,
    ) -> Result<()> {
        let video_patch_dim =
            self.config.in_channels * self.config.patch_size.iter().product::<usize>();
        let (video_batch, video_rows, video_width) = inputs.video_hidden_states.dims3()?;
        let (audio_batch, audio_rows, audio_width) = inputs.audio_hidden_states.dims3()?;
        let (text_batch, _, _) = prepared.refined_text.dims3()?;
        anyhow::ensure!(
            video_batch == audio_batch && video_batch == text_batch,
            "video, audio, and text batch sizes differ"
        );
        anyhow::ensure!(
            video_rows == prepared.video_indices.dim(0)?,
            "video index count differs from video rows"
        );
        anyhow::ensure!(
            audio_rows == prepared.audio_indices.dim(0)?,
            "audio index count differs from audio rows"
        );
        anyhow::ensure!(
            video_width == video_patch_dim,
            "video patch width is {video_width}, expected {video_patch_dim}"
        );
        anyhow::ensure!(
            audio_width == self.config.audio_in_channels,
            "audio width is {audio_width}, expected {}",
            self.config.audio_in_channels
        );
        anyhow::ensure!(
            inputs.timestep.rank() == 1,
            "timestep must be one-dimensional"
        );
        anyhow::ensure!(
            inputs.timestep_indices.dims() == [prepared.sequence_length],
            "timestep indices must be [sequence]"
        );
        anyhow::ensure!(
            inputs.timestep_indices.dtype() == DType::U32,
            "timestep_indices must use U32 indices"
        );
        anyhow::ensure!(
            inputs.timestep_indices.device().same_device(&self.device),
            "timestep_indices are on the wrong device"
        );
        let timestep_count = inputs.timestep.dim(0)? as u32;
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            anyhow::ensure!(
                timestep_count > 0
                    && timestep_count as usize <= crate::cuda::linear::MAX_TIMESTEP_ROWS,
                "CUDA BF16 H3 timestep projection supports 1..={} distinct rows with the \
                 verified cuBLASLt contract; request has {timestep_count}",
                crate::cuda::linear::MAX_TIMESTEP_ROWS
            );
        }
        anyhow::ensure!(
            inputs
                .timestep_indices
                .to_vec1::<u32>()?
                .iter()
                .all(|&index| index < timestep_count),
            "a timestep index is outside the distinct timestep table"
        );
        for (name, tensor) in [
            ("video", inputs.video_hidden_states),
            ("audio", inputs.audio_hidden_states),
            ("timestep", inputs.timestep),
        ] {
            anyhow::ensure!(
                tensor.device().same_device(&self.device),
                "{name} is on the wrong device"
            );
        }
        Ok(())
    }

    fn execute_stage<T>(
        &self,
        stage_index: usize,
        mut f: impl FnMut(&ExecutionStage, &BTreeMap<String, Tensor>) -> Result<T>,
    ) -> Result<T> {
        self.plan.with_stage(
            &self.weights,
            stage_index,
            &self.device,
            |stage, tensors| self.compute_with_oom_recovery(stage, tensors, &mut f),
        )
    }

    fn compute_with_oom_recovery<T>(
        &self,
        stage: &ExecutionStage,
        tensors: &BTreeMap<String, Tensor>,
        f: &mut impl FnMut(&ExecutionStage, &BTreeMap<String, Tensor>) -> Result<T>,
    ) -> Result<T> {
        let result = f(stage, tensors);
        #[cfg(feature = "cuda")]
        if let Err(error) = &result
            && self.device.is_cuda()
            && ff_core::weights::is_cuda_allocation_error(error.as_ref())
            && let Some(cache) = self.weights.device_cache()
            && !cache.is_demoted()
            && cache.stats().resident_bytes > 0
        {
            let removed = cache.demote_after_allocation_failure(
                &format!("compute:{}", stage.kind),
                &self.device,
                error,
            );
            self.device
                .synchronize()
                .context("finish cache release before H3 stage retry")?;
            eprintln!(
                "H3 {}: cleared {removed} cache entries after CUDA OOM; retrying compute once",
                stage.kind
            );
            return f(stage, tensors).with_context(|| {
                format!(
                    "H3 {} retry failed after cache release; initial error: {error:#}",
                    stage.kind,
                )
            });
        }
        result
    }

    fn required_stage(&self, kind: &StageKind) -> Result<usize> {
        self.plan
            .stage_index(kind)
            .with_context(|| format!("execution plan is missing {kind}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_projected_modalities(
    refined_text: &Tensor,
    audio: &Tensor,
    video: &Tensor,
    text_indices: &Tensor,
    audio_indices: &Tensor,
    video_indices: &Tensor,
    sequence_length: usize,
    canonical_layout_order: bool,
) -> Result<Tensor> {
    let (batch, text_rows, hidden) = refined_text
        .dims3()
        .context("refined text must be [batch, rows, hidden]")?;
    let (audio_batch, audio_rows, audio_hidden) = audio
        .dims3()
        .context("projected audio must be [batch, rows, hidden]")?;
    let (video_batch, video_rows, video_hidden) = video
        .dims3()
        .context("projected video must be [batch, rows, hidden]")?;
    anyhow::ensure!(
        audio_batch == batch && video_batch == batch,
        "projected modality batch sizes differ"
    );
    anyhow::ensure!(
        audio_hidden == hidden && video_hidden == hidden,
        "projected modality hidden widths differ"
    );
    anyhow::ensure!(
        text_indices.elem_count() == text_rows
            && audio_indices.elem_count() == audio_rows
            && video_indices.elem_count() == video_rows,
        "projected modality row counts differ from layout indices"
    );
    anyhow::ensure!(
        text_rows + audio_rows + video_rows == sequence_length,
        "projected modalities do not cover the packed sequence"
    );

    if canonical_layout_order {
        return Tensor::cat(&[refined_text, audio, video], 1)
            .context("failed to concatenate canonical text/audio/video rows");
    }

    let mut packed = Tensor::zeros(
        (batch, sequence_length, hidden),
        refined_text.dtype(),
        refined_text.device(),
    )?;
    packed = packed.index_add(text_indices, refined_text, 1)?;
    packed = packed.index_add(audio_indices, audio, 1)?;
    packed
        .index_add(video_indices, video, 1)
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn project_output_full_sequence_chunks(
    weights: &BTreeMap<String, Tensor>,
    hidden: &Tensor,
    video_indices: &Tensor,
    audio_indices: &Tensor,
    timestep_indices: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    norm_weight: &Tensor,
    chunk_size: NonZeroUsize,
    norm_eps: f64,
) -> Result<TransformerOutput> {
    let (batch, sequence, hidden_size) = hidden
        .dims3()
        .context("output hidden states must be [batch, sequence, hidden]")?;
    anyhow::ensure!(
        timestep_indices.dims() == [sequence],
        "output timestep indices must be [sequence]"
    );
    anyhow::ensure!(
        video_indices.dtype() == DType::U32
            && audio_indices.dtype() == DType::U32
            && timestep_indices.dtype() == DType::U32,
        "output row and timestep indices must use U32"
    );
    anyhow::ensure!(
        norm_weight.dims() == [hidden_size],
        "output norm weight has the wrong width"
    );
    let (timestep_count, shift_width) = shift
        .dims2()
        .context("output shift table must be [timesteps, hidden]")?;
    anyhow::ensure!(
        scale.dims() == [timestep_count, shift_width] && shift_width == hidden_size,
        "output modulation tables have the wrong shape"
    );
    for head_prefix in ["proj_out", "audio_proj_out"] {
        let head_weight = required(weights, &format!("{head_prefix}.weight"))?;
        let (_, head_input_width) = head_weight
            .dims2()
            .with_context(|| format!("{head_prefix}.weight must be a matrix"))?;
        anyhow::ensure!(
            head_input_width == hidden_size,
            "{head_prefix} input width is {head_input_width}, expected {hidden_size}"
        );
    }

    let chunk_size = chunk_size.get();
    let mut video_chunks = Vec::with_capacity(sequence.div_ceil(chunk_size));
    let mut audio_chunks = Vec::with_capacity(sequence.div_ceil(chunk_size));
    for start in (0..sequence).step_by(chunk_size) {
        let length = chunk_size.min(sequence - start);
        let hidden_chunk = hidden.narrow(1, start, length)?.contiguous()?;
        let timestep_chunk = timestep_indices.narrow(0, start, length)?;
        let normalized = core::rms_norm(&hidden_chunk, norm_weight, norm_eps)?;
        let shift_chunk = shift.index_select(&timestep_chunk, 0)?;
        let scale_chunk = scale.index_select(&timestep_chunk, 0)?.affine(1., 1.)?;
        let modulated = normalized
            .broadcast_mul(&scale_chunk)?
            .broadcast_add(&shift_chunk)?;
        video_chunks.push(linear(weights, "proj_out", &modulated)?);
        audio_chunks.push(linear(weights, "audio_proj_out", &modulated)?);
    }
    let video_chunks = video_chunks.iter().collect::<Vec<_>>();
    let audio_chunks = audio_chunks.iter().collect::<Vec<_>>();
    let video_all = Tensor::cat(&video_chunks, 1)
        .context("failed to concatenate full-packed video-head output chunks")?;
    let audio_all = Tensor::cat(&audio_chunks, 1)
        .context("failed to concatenate full-packed audio-head output chunks")?;
    let video = video_all.index_select(video_indices, 1)?;
    let audio = audio_all.index_select(audio_indices, 1)?;
    anyhow::ensure!(
        video.dim(0)? == batch && audio.dim(0)? == batch,
        "output head batch changed during modality selection"
    );
    Ok(TransformerOutput { video, audio })
}

fn indices_are_contiguous(indices: &[u32], start: usize) -> bool {
    indices.iter().enumerate().all(|(offset, &index)| {
        start
            .checked_add(offset)
            .is_some_and(|expected| index as usize == expected)
    })
}

fn modulation_to_device(
    modulation: &core::AdaLnModulation,
    device: &Device,
) -> Result<core::AdaLnModulation> {
    Ok(core::AdaLnModulation {
        shift_attention: modulation.shift_attention.to_device(device)?,
        scale_attention: modulation.scale_attention.to_device(device)?,
        gate_attention: modulation.gate_attention.to_device(device)?,
        shift_feed_forward: modulation.shift_feed_forward.to_device(device)?,
        scale_feed_forward: modulation.scale_feed_forward.to_device(device)?,
        gate_feed_forward: modulation.gate_feed_forward.to_device(device)?,
    })
}

fn linear(weights: &BTreeMap<String, Tensor>, prefix: &str, input: &Tensor) -> Result<Tensor> {
    let weight = required(weights, &format!("{prefix}.weight"))?;
    let bias = weights.get(&format!("{prefix}.bias"));
    core::linear_with_reference_bias(input, weight, bias)
        .with_context(|| format!("linear projection {prefix}"))
}

fn required<'a>(weights: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    weights
        .get(name)
        .with_context(|| format!("missing stage tensor {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MODALITY_COUNT;
    use ::safetensors::tensor::View as SafeTensorView;
    use candle_core::{DType, Shape, safetensors};
    use serde_json::json;
    use std::{
        collections::{BTreeMap, HashMap},
        fs,
    };

    fn ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn patterned_bf16(shape: impl Into<Shape>, seed: usize, center: f32) -> Tensor {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| center + (((index * 7 + seed) % 17) as f32 - 8.0) / 32.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    fn tensor_bytes(tensor: &Tensor) -> Vec<u8> {
        tensor.device().synchronize().unwrap();
        SafeTensorView::data(tensor).into_owned()
    }

    fn non_zero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn from_values(values: Vec<f32>, shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_full_sequence_output_heads(
        weights: &BTreeMap<String, Tensor>,
        hidden: &Tensor,
        timestep: &Tensor,
        timestep_indices: &Tensor,
        video_indices: &Tensor,
        audio_indices: &Tensor,
        hidden_size: usize,
        norm_eps: f64,
    ) -> Result<TransformerOutput> {
        let modulation = linear(weights, "norm_out.linear", &ops::silu(timestep)?)?;
        let shift = modulation.narrow(1, 0, hidden_size)?.contiguous()?;
        let scale = modulation
            .narrow(1, hidden_size, hidden_size)?
            .contiguous()?;
        let normalized =
            core::rms_norm(hidden, required(weights, "norm_out.norm.weight")?, norm_eps)?;
        let shift = shift.index_select(timestep_indices, 0)?;
        let scale = scale.index_select(timestep_indices, 0)?.affine(1., 1.)?;
        let modulated = normalized.broadcast_mul(&scale)?.broadcast_add(&shift)?;
        let video = linear(weights, "proj_out", &modulated)?.index_select(video_indices, 1)?;
        let audio =
            linear(weights, "audio_proj_out", &modulated)?.index_select(audio_indices, 1)?;
        Ok(TransformerOutput { video, audio })
    }

    #[allow(clippy::too_many_arguments)]
    fn chunked_output_heads_for_test(
        weights: &BTreeMap<String, Tensor>,
        hidden: &Tensor,
        timestep: &Tensor,
        timestep_indices: &Tensor,
        video_indices: &Tensor,
        audio_indices: &Tensor,
        hidden_size: usize,
        chunk_size: usize,
        norm_eps: f64,
    ) -> Result<TransformerOutput> {
        let modulation = linear(weights, "norm_out.linear", &ops::silu(timestep)?)?;
        let shift = modulation.narrow(1, 0, hidden_size)?.contiguous()?;
        let scale = modulation
            .narrow(1, hidden_size, hidden_size)?
            .contiguous()?;
        let norm_weight = required(weights, "norm_out.norm.weight")?;
        project_output_full_sequence_chunks(
            weights,
            hidden,
            video_indices,
            audio_indices,
            timestep_indices,
            &shift,
            &scale,
            norm_weight,
            non_zero(chunk_size),
            norm_eps,
        )
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "maximum absolute difference {max_abs} exceeds {tolerance}"
        );
    }

    #[test]
    fn transformer_options_use_shared_typed_defaults() {
        let options =
            StreamedTransformerOptions::new(WeightSource::Mmap, CachePolicy::new(1), Device::Cpu);
        assert_eq!(
            options.chunking.attention.projection_chunk_size.get(),
            DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE
        );
        assert_eq!(
            options.chunking.attention.query_chunk_size.get(),
            DEFAULT_ATTENTION_QUERY_CHUNK_SIZE
        );
        assert_eq!(
            options.chunking.attention.key,
            AttentionKeyChunkPolicy::Full
        );
        assert_eq!(
            options.chunking.feed_forward_chunk_size.get(),
            DEFAULT_FFN_TOKEN_CHUNK_SIZE
        );
        assert_eq!(
            options.chunking.output_chunk_size.get(),
            DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE
        );
        assert!(!options.device_cache_policy.is_enabled());
        assert!(!options.host_phase_priority);
        let mut invalid_host_priority = options.clone();
        invalid_host_priority.host_phase_priority = true;
        assert!(invalid_host_priority.validate().is_err());
        assert!(format!("{options:?}").contains("StreamedTransformerOptions"));

        let mut incompatible = options.clone();
        incompatible.flash_attention = true;
        incompatible.chunking.attention.key = AttentionKeyChunkPolicy::chunked(8).unwrap();
        assert!(
            incompatible
                .validate()
                .unwrap_err()
                .to_string()
                .contains("mutually exclusive")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_full_softmax_range_is_rejected_before_stage_execution() {
        // 2048 is where the persistent kernel hands over to the regular-register
        // one, not where exact full softmax stops: the two together cover
        // 1..=9216, and that pair is what the range check admits.
        let maximum = core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
        validate_h3_cuda_numerical_ranges(false, AttentionKeyChunkPolicy::Full, maximum, 1, 1)
            .unwrap();
        let error = validate_h3_cuda_numerical_ranges(
            false,
            AttentionKeyChunkPolicy::Full,
            maximum + 1,
            1,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("at most 9216 packed rows"));
        assert!(error.to_string().contains("--flash-attention"));
        validate_h3_cuda_numerical_ranges(true, AttentionKeyChunkPolicy::Full, maximum + 1, 1, 4)
            .unwrap();
        validate_h3_cuda_numerical_ranges(
            false,
            AttentionKeyChunkPolicy::chunked(512).unwrap(),
            maximum + 1,
            1,
            1,
        )
        .unwrap();
        let error =
            validate_h3_cuda_numerical_ranges(false, AttentionKeyChunkPolicy::Full, maximum, 0, 1)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("context projection supports 1..=16384")
        );
        let error = validate_h3_cuda_numerical_ranges(
            false,
            AttentionKeyChunkPolicy::chunked(512).unwrap(),
            maximum + 1,
            maximum + 1,
            4,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("token refiner supports at most 9216")
        );
        validate_h3_cuda_numerical_ranges(true, AttentionKeyChunkPolicy::Full, 125_510, 8_620, 4)
            .unwrap();
        let error =
            validate_h3_cuda_numerical_ranges(true, AttentionKeyChunkPolicy::Full, maximum, 1, 5)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("timestep projections support 1..=4")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn retained_real_h3_geometries_require_and_accept_explicit_flash() {
        for (name, context_rows, packed_rows, timestep_rows) in [
            ("t2va", 357, 73_743, 2),
            ("fl2va", 1_935, 76_329, 3),
            ("ref2va", 8_620, 125_510, 4),
        ] {
            validate_h3_cuda_numerical_ranges(
                true,
                AttentionKeyChunkPolicy::Full,
                packed_rows,
                context_rows,
                timestep_rows,
            )
            .unwrap_or_else(|error| panic!("{name} Flash profile was rejected: {error}"));
            let error = validate_h3_cuda_numerical_ranges(
                false,
                AttentionKeyChunkPolicy::Full,
                packed_rows,
                context_rows,
                timestep_rows,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("--flash-attention"),
                "{name} full-softmax rejection did not prescribe FlashAttention"
            );
        }
    }

    #[test]
    fn chunked_output_heads_match_the_full_sequence_reference() {
        let hidden_size = 4;
        let time_dim = 2;
        let mut weights = BTreeMap::new();
        weights.insert(
            "norm_out.norm.weight".to_owned(),
            from_values(vec![0.75, 1.0, 1.25, 1.5], hidden_size),
        );
        weights.insert(
            "norm_out.linear.weight".to_owned(),
            from_values(
                (0..2 * hidden_size * time_dim)
                    .map(|index| index as f32 * 0.025 - 0.15)
                    .collect(),
                (2 * hidden_size, time_dim),
            ),
        );
        weights.insert(
            "norm_out.linear.bias".to_owned(),
            from_values(
                (0..2 * hidden_size)
                    .map(|index| index as f32 * 0.01 - 0.03)
                    .collect(),
                2 * hidden_size,
            ),
        );
        for (prefix, output_width, offset) in [("proj_out", 3, -0.2f32), ("audio_proj_out", 2, 0.1)]
        {
            weights.insert(
                format!("{prefix}.weight"),
                from_values(
                    (0..output_width * hidden_size)
                        .map(|index| offset + index as f32 * 0.035)
                        .collect(),
                    (output_width, hidden_size),
                ),
            );
            weights.insert(
                format!("{prefix}.bias"),
                from_values(
                    (0..output_width)
                        .map(|index| offset * 0.25 + index as f32 * 0.02)
                        .collect(),
                    output_width,
                ),
            );
        }

        let hidden = from_values(
            (0..7 * hidden_size)
                .map(|index| index as f32 * 0.04 - 0.45)
                .collect(),
            (1, 7, hidden_size),
        );
        let timestep = from_values(vec![0.2, -0.4, 0.7, 0.1], (2, time_dim));
        let timestep_indices = Tensor::new(&[0u32, 1, 0, 1, 1, 0, 1], &Device::Cpu).unwrap();
        let video_indices = Tensor::new(&[6u32, 2, 4], &Device::Cpu).unwrap();
        let audio_indices = Tensor::new(&[1u32, 5], &Device::Cpu).unwrap();
        let reference = reference_full_sequence_output_heads(
            &weights,
            &hidden,
            &timestep,
            &timestep_indices,
            &video_indices,
            &audio_indices,
            hidden_size,
            1e-5,
        )
        .unwrap();

        for chunk_size in [1, 2, 3, 64] {
            let chunked = chunked_output_heads_for_test(
                &weights,
                &hidden,
                &timestep,
                &timestep_indices,
                &video_indices,
                &audio_indices,
                hidden_size,
                chunk_size,
                1e-5,
            )
            .unwrap();
            assert_eq!(chunked.video.dims(), &[1, 3, 3]);
            assert_eq!(chunked.audio.dims(), &[1, 2, 2]);
            assert_tensor_close(&chunked.video, &reference.video, 1e-5);
            assert_tensor_close(&chunked.audio, &reference.audio, 1e-5);
        }
    }

    #[test]
    fn canonical_packing_uses_cat_and_arbitrary_partitions_fall_back_safely() {
        let text = from_values(vec![10., 11., 12., 13.], (1, 2, 2));
        let audio = from_values(vec![20., 21., 22., 23.], (1, 2, 2));
        let video = from_values(vec![30., 31., 32., 33., 34., 35.], (1, 3, 2));
        let canonical_text = Tensor::new(&[0u32, 1], &Device::Cpu).unwrap();
        let canonical_audio = Tensor::new(&[2u32, 3], &Device::Cpu).unwrap();
        let canonical_video = Tensor::new(&[4u32, 5, 6], &Device::Cpu).unwrap();
        let canonical = pack_projected_modalities(
            &text,
            &audio,
            &video,
            &canonical_text,
            &canonical_audio,
            &canonical_video,
            7,
            true,
        )
        .unwrap();
        assert_eq!(
            canonical.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                10., 11., 12., 13., 20., 21., 22., 23., 30., 31., 32., 33., 34., 35.
            ]
        );

        let arbitrary_text = Tensor::new(&[1u32, 4], &Device::Cpu).unwrap();
        let arbitrary_audio = Tensor::new(&[0u32, 5], &Device::Cpu).unwrap();
        let arbitrary_video = Tensor::new(&[2u32, 3, 6], &Device::Cpu).unwrap();
        let fallback = pack_projected_modalities(
            &text,
            &audio,
            &video,
            &arbitrary_text,
            &arbitrary_audio,
            &arbitrary_video,
            7,
            false,
        )
        .unwrap();
        assert_eq!(
            fallback.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                20., 21., 10., 11., 30., 31., 32., 33., 12., 13., 22., 23., 34., 35.
            ]
        );
        assert!(indices_are_contiguous(&[0, 1], 0));
        assert!(!indices_are_contiguous(&[1, 4], 0));
    }

    fn exercise_block_ranges(device: Device, flash_attention: bool) {
        let hidden = if device.is_cuda() { 5376 } else { 4 };
        let heads = 1;
        let head_dim = if device.is_cuda() { 128 } else { 32 };
        let ffn = 5;
        let time = if device.is_cuda() { 2688 } else { 2 };
        let text_width = if device.is_cuda() { 5120 } else { 2 };
        let layers = 3;
        let mut tensors = HashMap::new();
        for (name, input_width) in [("proj_in", 1), ("audio_proj_in", 1)] {
            tensors.insert(format!("{name}.weight"), ones((hidden, input_width)));
            tensors.insert(format!("{name}.bias"), ones(hidden));
        }
        tensors.insert(
            "context_embedder.weight".to_owned(),
            patterned_bf16((hidden, text_width), 1, 0.0),
        );
        tensors.insert(
            "context_embedder.bias".to_owned(),
            patterned_bf16(hidden, 2, 0.0),
        );
        tensors.insert("time_embedder.linear_1.weight".to_owned(), ones((time, 2)));
        tensors.insert("time_embedder.linear_1.bias".to_owned(), ones(time));
        tensors.insert(
            "time_embedder.linear_2.weight".to_owned(),
            ones((time, time)),
        );
        tensors.insert("time_embedder.linear_2.bias".to_owned(), ones(time));

        let refiner = "token_refiner.refiner_blocks.0";
        tensors.insert(
            format!("{refiner}.norm1.weight"),
            patterned_bf16(hidden, 3, 1.0),
        );
        tensors.insert(
            format!("{refiner}.norm2.weight"),
            patterned_bf16(hidden, 4, 1.0),
        );
        for (offset, name) in ["to_q", "to_k", "to_v"].into_iter().enumerate() {
            tensors.insert(
                format!("{refiner}.attn.{name}.weight"),
                patterned_bf16((heads * head_dim, hidden), 5 + offset, 0.0),
            );
        }
        tensors.insert(
            format!("{refiner}.attn.norm_q.weight"),
            patterned_bf16(head_dim, 8, 1.0),
        );
        tensors.insert(
            format!("{refiner}.attn.norm_k.weight"),
            patterned_bf16(head_dim, 9, 1.0),
        );
        tensors.insert(
            format!("{refiner}.attn.to_out.0.weight"),
            patterned_bf16((hidden, heads * head_dim), 10, 0.0),
        );
        tensors.insert(
            format!("{refiner}.ff.net.0.proj.weight"),
            patterned_bf16((2 * ffn, hidden), 11, 0.0),
        );
        tensors.insert(
            format!("{refiner}.ff.net.2.weight"),
            patterned_bf16((hidden, ffn), 12, 0.0),
        );
        tensors.insert(
            "token_refiner.final_norm.weight".to_owned(),
            patterned_bf16(hidden, 13, 1.0),
        );

        for block in 0..layers {
            let prefix = format!("transformer_blocks.{block}");
            let seed = 100 + block * 20;
            tensors.insert(
                format!("{prefix}.adaln_proj.linear.weight"),
                patterned_bf16((6 * hidden * MODALITY_COUNT, time), seed, 0.0),
            );
            tensors.insert(
                format!("{prefix}.norm1.weight"),
                patterned_bf16(hidden, seed + 1, 1.0),
            );
            for (offset, name) in ["to_q", "to_k", "to_v"].into_iter().enumerate() {
                tensors.insert(
                    format!("{prefix}.attn.{name}.weight"),
                    patterned_bf16((heads * head_dim, hidden), seed + 2 + offset, 0.0),
                );
            }
            tensors.insert(
                format!("{prefix}.attn.norm_q.weight"),
                patterned_bf16(head_dim, seed + 5, 1.0),
            );
            tensors.insert(
                format!("{prefix}.attn.norm_k.weight"),
                patterned_bf16(head_dim, seed + 6, 1.0),
            );
            tensors.insert(
                format!("{prefix}.attn.to_out.0.weight"),
                patterned_bf16((hidden, heads * head_dim), seed + 7, 0.0),
            );
            tensors.insert(
                format!("{prefix}.norm2.weight"),
                patterned_bf16(hidden, seed + 8, 1.0),
            );
            tensors.insert(
                format!("{prefix}.ff.net.0.proj.weight"),
                patterned_bf16((2 * ffn, hidden), seed + 9, 0.0),
            );
            tensors.insert(
                format!("{prefix}.ff.net.2.weight"),
                patterned_bf16((hidden, ffn), seed + 10, 0.0),
            );
        }
        tensors.insert(
            "norm_out.norm.weight".to_owned(),
            patterned_bf16(hidden, 200, 1.0),
        );
        tensors.insert(
            "norm_out.linear.weight".to_owned(),
            patterned_bf16((2 * hidden, time), 201, 0.0),
        );
        tensors.insert(
            "norm_out.linear.bias".to_owned(),
            patterned_bf16(2 * hidden, 202, 0.0),
        );
        for name in ["proj_out", "audio_proj_out"] {
            tensors.insert(format!("{name}.weight"), ones((1, hidden)));
            tensors.insert(format!("{name}.bias"), ones(1));
        }

        let dir = tempfile::tempdir().unwrap();
        safetensors::save(&tensors, dir.path().join("weights.safetensors")).unwrap();
        let map = tensors
            .keys()
            .map(|name| (name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": tensors.values().map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes()).sum::<usize>()},
                "weight_map": map
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let config = TransformerConfig {
            class_name: "MiniMaxH3Transformer3DModel".to_owned(),
            num_attention_heads: heads,
            attention_head_dim: head_dim,
            hidden_size: hidden,
            num_layers: layers,
            num_refiner_layers: 1,
            ffn_dim: ffn,
            in_channels: 1,
            audio_in_channels: 1,
            patch_size: [1, 1, 1],
            text_dim: text_width,
            freq_dim: 2,
            time_embed_hidden_dim: time,
            time_embed_dim: time,
            rope_freq_dim: 1,
            rope_theta: 10_000.,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        };
        let mut options = StreamedTransformerOptions::new(
            WeightSource::Mmap,
            CachePolicy::new(1),
            device.clone(),
        );
        options.chunking.attention.projection_chunk_size = non_zero(1);
        options.chunking.attention.query_chunk_size = non_zero(1);
        options.chunking.feed_forward_chunk_size = non_zero(1);
        options.flash_attention = flash_attention;
        config.validate().unwrap();
        options.validate().unwrap();
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DriverError, result, sys};
            let mut recovery_options = options.clone();
            recovery_options.device_cache_policy = DeviceCachePolicy::with_max_bytes(256 << 20)
                .with_cuda_allocator(ff_core::weights::CudaWeightAllocator::Direct);
            let recovery_stack = StreamedTransformer::from_validated_parts(
                ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap(),
                config.clone(),
                recovery_options,
            )
            .unwrap();
            let stage = recovery_stack
                .required_stage(&StageKind::BlockFeedForward(0))
                .unwrap();
            let mut calls = 0;
            let invalid = recovery_stack.execute_stage(stage, |_, _| -> Result<()> {
                calls += 1;
                anyhow::bail!("invalid compute input")
            });
            assert!(invalid.is_err());
            assert_eq!(calls, 1);
            assert!(!recovery_stack.device_cache_stats().demoted);
            recovery_stack
                .weights
                .load("context_embedder.weight", &device)
                .unwrap();
            device.synchronize().unwrap();
            let stream = device.as_cuda_device().unwrap().cuda_stream();
            let mut requested = None;
            let mut observed_free = Vec::new();
            calls = 0;
            let allocation = recovery_stack
                .execute_stage(stage, |_, _| -> Result<CudaSlice<u8>> {
                    calls += 1;
                    stream.synchronize()?;
                    let free = stream.context().mem_get_info()?.0;
                    observed_free.push(free);
                    let bytes = *requested.get_or_insert(free.checked_add(16 << 20).unwrap());
                    stream.context().bind_to_thread()?;
                    Ok(unsafe {
                        stream.upgrade_device_ptr::<u8>(result::malloc_sync(bytes)?, bytes)
                    })
                })
                .unwrap();
            assert_eq!(calls, 2);
            assert!(observed_free[0] < requested.unwrap());
            assert!(observed_free[1] >= requested.unwrap());
            assert!(recovery_stack.device_cache_stats().demoted);
            assert_eq!(recovery_stack.device_cache_stats().resident_bytes, 0);
            drop(allocation);
            stream.synchronize().unwrap();
            println!(
                "{}",
                serde_json::json!({
                    "physical_oom_recovery":true,"attempts":calls,"requested_bytes":requested,
                    "free_before_bytes":observed_free[0],"free_after_release_bytes":observed_free[1],
                })
            );
            calls = 0;
            let exhausted = recovery_stack.execute_stage(stage, |_, _| -> Result<()> {
                calls += 1;
                Err(DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY).into())
            });
            assert!(exhausted.is_err());
            assert_eq!(calls, 1);
        }
        let normal_stack = StreamedTransformer::from_validated_parts(
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap(),
            config.clone(),
            options.clone(),
        )
        .unwrap();
        let cached_stack = if device.is_cpu() {
            let mut cached_options = options.clone();
            cached_options.device_cache_policy = DeviceCachePolicy::with_max_bytes(4096);
            cached_options.host_phase_priority = true;
            cached_options.cache_policy = CachePolicy::unbounded_units()
                .with_max_bytes(4096)
                .with_granularity(ff_core::weights::CacheGranularity::Tensor);
            Some(
                StreamedTransformer::from_validated_parts(
                    ModelWeights::open(dir.path(), WeightSource::Mmap, cached_options.cache_policy)
                        .unwrap(),
                    config.clone(),
                    cached_options,
                )
                .unwrap(),
            )
        } else {
            None
        };
        let stack = StreamedTransformer::from_validated_parts(weights, config, options).unwrap();
        let expected_dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let input = patterned_bf16((1, 2, hidden), 300, 0.25)
            .to_dtype(expected_dtype)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let temb = ones((1, time)).to_device(&device).unwrap();
        let indices = Tensor::new(&[0u32, 1], &device).unwrap();
        let cos = ones((2, 2)).to_device(&device).unwrap();
        let sin = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let output = stack
            .forward_blocks_range(0..layers, &input, &temb, &indices, &cos, &sin)
            .unwrap();
        assert_eq!(output.dims(), &[1, 2, hidden]);
        assert_eq!(output.dtype(), expected_dtype);
        let normal_output = normal_stack
            .forward_blocks_range(0..layers, &input, &temb, &indices, &cos, &sin)
            .unwrap();
        let reference_bytes = tensor_bytes(&normal_output);
        assert_eq!(tensor_bytes(&output), reference_bytes);
        if let Some(cached_stack) = &cached_stack {
            let placement = cached_stack.device_residency_plan().unwrap();
            assert!(!placement.placed.is_empty());
            assert!(
                placement
                    .placed
                    .iter()
                    .all(|name| !name.contains("adaln") && !name.starts_with("refiner"))
            );
            for _ in 0..2 {
                let cached_output = cached_stack
                    .forward_blocks_range(0..layers, &input, &temb, &indices, &cos, &sin)
                    .unwrap();
                assert_eq!(tensor_bytes(&cached_output), reference_bytes);
            }
            let stats = cached_stack.device_cache_stats();
            assert!(stats.hits > 0 && stats.prioritized_bytes > 0);
            assert!(stats.resident_bytes <= 4096 && !stats.demoted);
            let host = cached_stack.cache_stats();
            assert!(host.resident_bytes <= 4096);
            let priority = host.tensor_retention.unwrap().priority.unwrap();
            assert_eq!(
                priority.reserved_bytes,
                cached_stack.host_residency_plan().unwrap().resident_bytes
            );
            assert!(
                priority.resident_bytes > 0 && priority.resident_bytes <= priority.reserved_bytes
            );
        }
        let input_bytes = tensor_bytes(&input);
        for block in 0..layers {
            let block_output = normal_stack
                .forward_blocks_range(block..block + 1, &input, &temb, &indices, &cos, &sin)
                .unwrap();
            assert_ne!(tensor_bytes(&block_output), input_bytes);
        }
        for cut in 1..layers {
            let prefix = normal_stack
                .forward_blocks_range(0..cut, &input, &temb, &indices, &cos, &sin)
                .unwrap();
            let split = normal_stack
                .forward_blocks_range(cut..layers, &prefix, &temb, &indices, &cos, &sin)
                .unwrap();
            assert_eq!(split.dtype(), expected_dtype);
            assert_eq!(tensor_bytes(&split), reference_bytes);
        }
        let access_before = normal_stack.access_stats();
        let reversed = layers..layers - 1;
        for invalid in [0..0, reversed, 0..layers + 1] {
            assert!(
                normal_stack
                    .forward_blocks_range(invalid, &input, &temb, &indices, &cos, &sin)
                    .is_err()
            );
        }
        assert_eq!(normal_stack.access_stats(), access_before);

        let video = ones((1, 1, 1)).to_device(&device).unwrap();
        let audio = ones((1, 1, 1)).to_device(&device).unwrap();
        let text = ones((1, 1, text_width)).to_device(&device).unwrap();
        let timestep = Tensor::new(&[0.5f32], &device).unwrap();
        let timestep_indices = Tensor::new(&[0u32, 0, 0], &device).unwrap();
        let token_tags = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let position_ids = Tensor::zeros((3, 3), DType::I64, &device).unwrap();
        let video_indices = Tensor::new(&[0u32], &device).unwrap();
        let text_indices = Tensor::new(&[1u32], &device).unwrap();
        let audio_indices = Tensor::new(&[2u32], &device).unwrap();
        let prepared = stack
            .prepare_context(&TransformerStaticInputs {
                encoder_hidden_states: &text,
                token_tags: &token_tags,
                position_ids: &position_ids,
                video_indices: &video_indices,
                audio_indices: &audio_indices,
                text_indices: &text_indices,
            })
            .unwrap();
        let output = stack
            .forward_step(
                &prepared,
                &TransformerStepInputs {
                    video_hidden_states: &video,
                    audio_hidden_states: &audio,
                    timestep: &timestep,
                    timestep_indices: &timestep_indices,
                },
            )
            .unwrap();
        assert_eq!(output.video.dims(), &[1, 1, 1]);
        assert_eq!(output.audio.dims(), &[1, 1, 1]);
        let schedule = stack.prepare_denoise_schedule(&[vec![0.5]]).unwrap();
        assert_eq!(schedule.steps(), 1);
        let prepared_step = &schedule.steps[0];
        let prepared_timestep = prepared_step.timestep_embedding.to_device(&device).unwrap();
        let prepared_dynamic = normal_stack
            .forward_blocks_range(0..layers, &input, &prepared_timestep, &indices, &cos, &sin)
            .unwrap();
        let prepared_reference = normal_stack
            .forward_blocks_precomputed_range(
                0..layers,
                &input,
                &prepared_step.modulations,
                BlockExecutionContext {
                    adaln_indices: &indices,
                    rotary_cos: &cos,
                    rotary_sin: &sin,
                },
            )
            .unwrap();
        let prepared_reference_bytes = tensor_bytes(&prepared_reference);
        assert_eq!(tensor_bytes(&prepared_dynamic), prepared_reference_bytes);
        for cut in 1..layers {
            let prefix = normal_stack
                .forward_blocks_precomputed_range(
                    0..cut,
                    &input,
                    &prepared_step.modulations,
                    BlockExecutionContext {
                        adaln_indices: &indices,
                        rotary_cos: &cos,
                        rotary_sin: &sin,
                    },
                )
                .unwrap();
            let split = normal_stack
                .forward_blocks_precomputed_range(
                    cut..layers,
                    &prefix,
                    &prepared_step.modulations,
                    BlockExecutionContext {
                        adaln_indices: &indices,
                        rotary_cos: &cos,
                        rotary_sin: &sin,
                    },
                )
                .unwrap();
            assert_eq!(split.dtype(), expected_dtype);
            assert_eq!(tensor_bytes(&split), prepared_reference_bytes);
        }
        let cached = stack
            .forward_precomputed_step(
                &prepared,
                &schedule,
                0,
                &TransformerStepInputs {
                    video_hidden_states: &video,
                    audio_hidden_states: &audio,
                    timestep: &timestep,
                    timestep_indices: &timestep_indices,
                },
            )
            .unwrap();
        assert_eq!(cached.video.dims(), output.video.dims());
        assert_eq!(cached.audio.dims(), output.audio.dims());
        assert_eq!(tensor_bytes(&cached.video), tensor_bytes(&output.video));
        assert_eq!(tensor_bytes(&cached.audio), tensor_bytes(&output.audio));
        stack
            .forward_blocks_range(1..2, &input, &temb, &indices, &cos, &sin)
            .unwrap();
        stack
            .forward_blocks_precomputed_range(
                1..2,
                &input,
                &prepared_step.modulations,
                BlockExecutionContext {
                    adaln_indices: &indices,
                    rotary_cos: &cos,
                    rotary_sin: &sin,
                },
            )
            .unwrap();
        assert!(
            stack
                .forward_blocks_range(2..2, &input, &temb, &indices, &cos, &sin)
                .is_err()
        );
    }

    #[test]
    fn block_ranges_match_uncut_stack_for_every_internal_cut() {
        exercise_block_ranges(Device::Cpu, false);
    }
}
