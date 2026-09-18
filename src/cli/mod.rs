mod adapters;
mod routing;

use adapters::{GlmCommand, H3Command, H3DecodeCommand, TrellisCommand};

mod calibrate;
mod calibrate_io;
mod checkpoint;
mod clip;
mod collective;
mod conditioned;
mod decode;
mod denoise;
mod generate;
mod glm;
mod glm_multi;
mod identity;
mod initialize;
mod inspect;
mod minicpm;
mod models;
mod music;
mod plan;
mod probe;
mod progress;
mod prompt;
mod resource;
mod trellis;
mod trellis_generate;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors};
use clap::{Parser, Subcommand};
use flyingfish::{
    glm::{ExpertCacheLayout, ExpertCacheReplacementPolicy},
    h3::audio_vae::{StreamedAudioVae, WavSampleFormat, write_wav},
    h3::conditioning_provenance::{ExternalPromptProvenance, H3ConditioningProvenance},
    h3::config::TransformerConfig,
    h3::core::{
        AttentionChunking, AttentionKeyChunkPolicy, AttentionKeyChunkSize,
        CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS, DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE,
        DEFAULT_ATTENTION_QUERY_CHUNK_SIZE, DEFAULT_FFN_TOKEN_CHUNK_SIZE,
        DEFAULT_FLASH_ATTENTION_PROJECTION_CHUNK_SIZE,
    },
    h3::cuda::profile::validate_selected_device as validate_h3_selected_cuda_profile,
    h3::execution::H3ExecutionPlan,
    h3::model::{
        DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE, StreamedTransformer, StreamedTransformerOptions,
        TransformerChunking, validate_h3_numerical_backend,
    },
    h3::pipeline::{
        DenoiseCheckpointEvent, DenoiseObserver, DenoisePreparationEvent, DenoiseStepEvent,
        T2vaExecutionOptions, T2vaLatents, T2vaSchedule, denoise_t2va_with_options_and_observer,
    },
    h3::policy::{
        ExecutionPolicy, H3QwenNumericalContract, H3QwenVisionGridModality,
        H3QwenVisionLinearGeometry,
    },
    h3::resources::{
        ResourceAssumptions, ResourceBudget, ResourceEstimate, T2vaGeometry, TransformerShape,
    },
    h3::solver::{AttentionBackendAllowlist, PolicySearchSpace, solve_feasible_policies},
    h3::target_geometry::{H3AspectRatio, H3TargetGeometry, resolve_h3_target_geometry},
    h3::text_encoder::StreamedTextEncoder,
    h3::video_vae::StreamedVideoVae,
    recovery::{CheckpointIdentity, PolicyHistory, take_t2va_checkpoint_metadata},
    runtime::artifact::sync_parent_directory,
    runtime::frame_manifest::PngFrameSetManifest,
    runtime::parity::{ParityTolerance, compare_safetensors},
    runtime::residency::ResidencyAuthorization,
    runtime::telemetry::TelemetryMonitor,
    runtime::weights::{
        CacheGranularity, CachePolicy, DeviceCache, DeviceCachePolicy, ModelWeights, WeightSource,
    },
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::{
    collections::HashMap,
    num::{NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

const DEFAULT_NON_FLASH_BACKEND_WORKSPACE_MIB: u64 = 1536;
const DEFAULT_FLASH_BACKEND_WORKSPACE_MIB: u64 = 3584;

fn build_qwen_numerical_contract(
    device: &Device,
    configured_query_rows: usize,
    language_rows: usize,
    image_total_patch_rows: usize,
    video_total_patch_rows: usize,
    max_vision_segment_rows: usize,
) -> Result<H3QwenNumericalContract> {
    let configured_query_rows = NonZeroUsize::new(configured_query_rows)
        .context("Qwen configured query rows must be non-zero")?;
    let language_rows =
        NonZeroUsize::new(language_rows).context("Qwen language rows must be non-zero")?;
    let geometry = H3QwenVisionLinearGeometry::from_patch_rows(
        image_total_patch_rows,
        video_total_patch_rows,
    )?;
    let backend = flyingfish::h3::policy::ExecutionBackendPolicy::from_device(device);
    let contract = H3QwenNumericalContract::for_target_with_grids(
        backend,
        backend
            .is_cuda()
            .then(|| flyingfish::h3::policy::CudaCapabilities::from_device(device)),
        configured_query_rows,
        language_rows,
        max_vision_segment_rows,
        geometry,
        &[],
    )?;
    contract.validate_for(
        device,
        configured_query_rows,
        language_rows,
        max_vision_segment_rows,
        geometry,
    )?;
    Ok(contract)
}

fn validate_qwen_numerical_contract(
    contract: &H3QwenNumericalContract,
    device: &Device,
    language_rows: usize,
    image_total_patch_rows: usize,
    video_total_patch_rows: usize,
    max_vision_segment_rows: usize,
) -> Result<()> {
    contract.validate_for(
        device,
        NonZeroUsize::new(
            usize::try_from(contract.configured_query_rows)
                .context("recorded Qwen configured query rows exceed usize")?,
        )
        .context("recorded Qwen configured query rows must be non-zero")?,
        NonZeroUsize::new(language_rows).context("Qwen language rows must be non-zero")?,
        max_vision_segment_rows,
        H3QwenVisionLinearGeometry::from_patch_rows(
            image_total_patch_rows,
            video_total_patch_rows,
        )?,
    )
}

#[derive(Debug, Parser)]
#[command(name = "ff", about = "Checkpoint and inference toolkit")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// MiniMax-H3's own request target: `short_edge`, `aspect_ratio`, and
/// `duration_seconds`.
///
/// The official API takes a target in these terms; every command here takes it
/// in latents and aligned frame counts. Passing this group converts one into
/// the other instead of making the caller apply the 16x VAE ratio, the patch-2
/// divisibility, and `17 * n + 5` frame alignment by hand. It is an
/// alternative to the explicit geometry flags, never a modifier of them.
#[derive(Clone, Debug, clap::Args)]
struct TargetGeometryArgs {
    #[arg(help = "Official target: short side in pixels, a multiple of 32. Implies the geometry")]
    #[arg(long, requires = "duration_seconds")]
    short_edge: Option<usize>,
    #[arg(
        help = "Official target aspect ratio, written width:height, or `auto` to keep the \
                  source ratio. Defaults to 16:9, which is what the released 768p assets use"
    )]
    #[arg(long, requires = "short_edge")]
    aspect_ratio: Option<String>,
    #[arg(help = "Official target duration in seconds, 4 through 15")]
    #[arg(long, requires = "short_edge")]
    duration_seconds: Option<usize>,
}

/// The ratio spelling that keeps a conditioning source's own shape.
const AUTO_ASPECT_RATIO: &str = "auto";

impl TargetGeometryArgs {
    /// Resolve the official target, or `None` when the caller spelled the
    /// geometry out instead.
    ///
    /// `source_canvas` is the `(width, height)` an `auto` ratio is taken from.
    /// Commands with no conditioning input pass `None` and reject `auto`.
    fn resolve(&self, source_canvas: Option<(usize, usize)>) -> Result<Option<H3TargetGeometry>> {
        let (Some(short_edge), Some(duration_seconds)) = (self.short_edge, self.duration_seconds)
        else {
            anyhow::ensure!(
                self.aspect_ratio.is_none(),
                "--aspect-ratio only applies to the official --short-edge target"
            );
            return Ok(None);
        };
        let aspect_ratio = match self.aspect_ratio.as_deref() {
            None => H3AspectRatio::WIDESCREEN,
            Some(AUTO_ASPECT_RATIO) => {
                let (width, height) = source_canvas.context(
                    "--aspect-ratio auto needs a conditioning input to take its ratio from; pass an explicit ratio such as 16:9",
                )?;
                H3AspectRatio::of_canvas(width, height)?
            }
            Some(text) => text.parse()?,
        };
        let target = resolve_h3_target_geometry(short_edge, aspect_ratio, duration_seconds)?;
        Ok(Some(target))
    }

    /// The latent geometry to run, preferring an official target over the
    /// explicitly spelled flags. Announces the conversion so the operator sees
    /// the canvas and aligned duration the target actually produced.
    fn resolve_latent_geometry(
        &self,
        explicit: (usize, usize, usize, usize),
    ) -> Result<(usize, usize, usize, usize)> {
        let Some(target) = self.resolve(None)? else {
            return Ok(explicit);
        };
        eprintln!("{}", describe_h3_target(&target));
        Ok((
            target.latent_frames,
            target.latent_height,
            target.latent_width,
            target.audio_frames,
        ))
    }

    /// The same, for commands whose geometry flags are required rather than
    /// defaulted. Clap's `required_unless_present` already guarantees one form
    /// is complete; the error covers the case it cannot see.
    fn resolve_required_latent_geometry(
        &self,
        explicit: (Option<usize>, Option<usize>, Option<usize>, Option<usize>),
    ) -> Result<(usize, usize, usize, usize)> {
        if let Some(target) = self.resolve(None)? {
            eprintln!("{}", describe_h3_target(&target));
            return Ok((
                target.latent_frames,
                target.latent_height,
                target.latent_width,
                target.audio_frames,
            ));
        }
        let (Some(latent_frames), Some(latent_height), Some(latent_width), Some(audio_frames)) =
            explicit
        else {
            bail!(
                "pass either the official --short-edge target or all of --latent-frames, --latent-height, --latent-width, and --audio-frames"
            );
        };
        Ok((latent_frames, latent_height, latent_width, audio_frames))
    }

    /// The pixel canvas and *requested* frame count for the conditioned
    /// commands, which align the request themselves.
    ///
    /// `source_canvas` is the conditioning input's own `(width, height)`, which
    /// an `auto` ratio is taken from.
    fn resolve_canvas_geometry(
        &self,
        explicit_canvas: (Option<usize>, Option<usize>),
        explicit_num_frames: usize,
        source_canvas: Option<(usize, usize)>,
    ) -> Result<(Option<(usize, usize)>, usize)> {
        let Some(target) = self.resolve(source_canvas)? else {
            let (height, width) = explicit_canvas;
            anyhow::ensure!(
                height.is_some() == width.is_some(),
                "--height and --width must be passed together"
            );
            return Ok((height.zip(width), explicit_num_frames));
        };
        eprintln!("{}", describe_h3_target(&target));
        Ok((
            Some((target.canvas_height, target.canvas_width)),
            target.requested_num_frames,
        ))
    }
}

/// One line stating what an official target resolved to, including the aligned
/// duration, which frame alignment can push past the request.
fn describe_h3_target(target: &H3TargetGeometry) -> String {
    format!(
        "official target: short_edge {} aspect_ratio {} duration_seconds {} -> {}x{}, {} frames ({:.3} s)",
        target.short_edge,
        target.aspect_ratio,
        target.duration_seconds,
        target.canvas_width,
        target.canvas_height,
        target.num_frames,
        target.aligned_duration_seconds()
    )
}

#[derive(Clone, Copy, Debug, clap::Args)]
struct TransformerChunkArgs {
    #[arg(help = "Number of attention query rows evaluated per bounded score chunk")]
    #[arg(
        long,
        default_value_t = NonZeroUsize::new(DEFAULT_ATTENTION_QUERY_CHUNK_SIZE).unwrap()
    )]
    attention_query_chunk_size: NonZeroUsize,
    #[arg(help = "Rows per Q/K/V projection GEMM, independent of score workspace")]
    #[arg(
        long,
        default_value_t = NonZeroUsize::new(DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE).unwrap()
    )]
    attention_projection_chunk_size: NonZeroUsize,
    #[arg(help = "Experimental online-softmax key tile; omit to use full softmax")]
    #[arg(long, conflicts_with = "flash_attention")]
    attention_key_chunk_size: Option<AttentionKeyChunkSize>,
    #[arg(help = "Number of sequence rows evaluated per bounded FFN activation chunk")]
    #[arg(
        long,
        default_value_t = NonZeroUsize::new(DEFAULT_FFN_TOKEN_CHUNK_SIZE).unwrap()
    )]
    ffn_token_chunk_size: NonZeroUsize,
    #[arg(help = "Number of modality rows evaluated per output-head chunk")]
    #[arg(
        long,
        default_value_t = NonZeroUsize::new(DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE).unwrap()
    )]
    output_token_chunk_size: NonZeroUsize,
}

impl TransformerChunkArgs {
    fn attention_key_policy(self) -> AttentionKeyChunkPolicy {
        self.attention_key_chunk_size.map_or(
            AttentionKeyChunkPolicy::Full,
            AttentionKeyChunkPolicy::Chunked,
        )
    }

    fn model_chunking(self) -> TransformerChunking {
        TransformerChunking {
            attention: AttentionChunking {
                projection_chunk_size: self.attention_projection_chunk_size,
                query_chunk_size: self.attention_query_chunk_size,
                key: self.attention_key_policy(),
            },
            feed_forward_chunk_size: self.ffn_token_chunk_size,
            output_chunk_size: self.output_token_chunk_size,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, clap::Args)]
struct DenoiseChunkArgs {
    #[arg(
        help = "Number of attention query rows evaluated per bounded score chunk. \
                  Inert under --flash-attention, which materializes no score chunk"
    )]
    #[arg(long)]
    attention_query_chunk_size: Option<NonZeroUsize>,
    #[arg(
        help = "Rows per Q/K/V projection GEMM, independent of score workspace. \
                Also the query span of each FlashAttention call, so it defaults \
                to 32 on the exact path and 4096 under --flash-attention"
    )]
    #[arg(long)]
    attention_projection_chunk_size: Option<NonZeroUsize>,
    #[arg(help = "Experimental online-softmax key tile; omit to use full softmax")]
    #[arg(long, conflicts_with = "flash_attention")]
    attention_key_chunk_size: Option<AttentionKeyChunkSize>,
    #[arg(help = "Number of sequence rows evaluated per bounded FFN activation chunk")]
    #[arg(long)]
    ffn_token_chunk_size: Option<NonZeroUsize>,
    #[arg(help = "Number of modality rows evaluated per output-head chunk")]
    #[arg(long)]
    output_token_chunk_size: Option<NonZeroUsize>,
}

impl DenoiseChunkArgs {
    fn is_explicit(self) -> bool {
        self.attention_query_chunk_size.is_some()
            || self.attention_projection_chunk_size.is_some()
            || self.attention_key_chunk_size.is_some()
            || self.ffn_token_chunk_size.is_some()
            || self.output_token_chunk_size.is_some()
    }

    /// Resolve every unstated chunk to its documented default.
    ///
    /// The projection chunk has two defaults because it means two things. On
    /// the exact path it only bounds the Q/K/V projection activations, so 32
    /// rows costs nothing. Under FlashAttention it is additionally the query
    /// span of each attention call, and a 32-row span runs that call an order
    /// of magnitude below the device's achievable rate. An explicit
    /// `--attention-projection-chunk-size` is always honoured as given.
    /// The FlashAttention path also uses 1024-row FFN/output chunks, based on
    /// the retained local H3 operator screening and two-evaluation execution.
    /// These geometries are recorded in the execution policy, as chunk sizes
    /// can affect vendor GEMM rounding. Explicit sizes always take precedence.
    fn configured(self, flash_attention: bool) -> TransformerChunkArgs {
        let default_projection_chunk_size = if flash_attention {
            DEFAULT_FLASH_ATTENTION_PROJECTION_CHUNK_SIZE
        } else {
            DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE
        };
        let (default_ffn_chunk_size, default_output_chunk_size) = if flash_attention {
            (1024, 1024)
        } else {
            (
                DEFAULT_FFN_TOKEN_CHUNK_SIZE,
                DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE,
            )
        };
        TransformerChunkArgs {
            attention_query_chunk_size: self.attention_query_chunk_size.unwrap_or_else(|| {
                NonZeroUsize::new(DEFAULT_ATTENTION_QUERY_CHUNK_SIZE)
                    .expect("default attention query chunk is non-zero")
            }),
            attention_projection_chunk_size: self.attention_projection_chunk_size.unwrap_or_else(
                || {
                    NonZeroUsize::new(default_projection_chunk_size)
                        .expect("default attention projection chunk is non-zero")
                },
            ),
            attention_key_chunk_size: self.attention_key_chunk_size,
            ffn_token_chunk_size: self.ffn_token_chunk_size.unwrap_or_else(|| {
                NonZeroUsize::new(default_ffn_chunk_size)
                    .expect("default feed-forward chunk is non-zero")
            }),
            output_token_chunk_size: self.output_token_chunk_size.unwrap_or_else(|| {
                NonZeroUsize::new(default_output_chunk_size)
                    .expect("default output chunk is non-zero")
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, clap::Args)]
struct WeightCacheArgs {
    #[arg(help = "Keep weight shards file-backed or read them into host memory")]
    #[arg(long, value_enum, default_value = "mmap")]
    weight_source: WeightSource,
    #[arg(
        help = "Host residency ceiling in MiB; replaces the default one-shard bound. A single oversized unit may still reside"
    )]
    #[arg(long)]
    host_cache_mib: Option<u64>,
    #[arg(
        long,
        value_enum,
        default_value = "shard",
        help = "Retain whole source shards or independently bounded raw tensor ranges"
    )]
    host_cache_granularity: CacheGranularity,
}

impl WeightCacheArgs {
    fn cache_policy(self) -> Result<CachePolicy> {
        let policy = match self.host_cache_mib {
            Some(value) => CachePolicy::unbounded_units().with_max_bytes(mib_to_bytes(value)?),
            None => CachePolicy::new(1),
        };
        Ok(policy.with_granularity(self.host_cache_granularity))
    }
}

/// Device residency for adapters whose loaders stream weights per use. Kept off
/// the shared `WeightCacheArgs` because H3's commands route residency through
/// their own policy machinery.
#[derive(Clone, Copy, Debug, Default, clap::Args)]
struct DeviceCacheArgs {
    #[arg(
        long,
        help = "Device residency ceiling in MiB per device, shared by the request's checkpoints on that device; zero disables retention, absent uses the adapter's default"
    )]
    device_cache_mib: Option<u64>,
    #[arg(
        long,
        value_enum,
        default_value = "pool",
        help = "CUDA allocator for retained weights; direct is experimental and may synchronize uploads"
    )]
    device_cache_allocator: flyingfish::runtime::weights::CudaWeightAllocator,
}

impl DeviceCacheArgs {
    /// What the operator authorized, if anything. A ceiling is an upper bound
    /// the ladder may spend under, not the budget itself: what is actually
    /// spent is decided against measured device capacity.
    fn authorization(self) -> Result<ResidencyAuthorization> {
        Ok(match self.device_cache_mib.filter(|mib| *mib > 0) {
            Some(mib) => ResidencyAuthorization::OperatorExplicit {
                ceiling_bytes: mib_to_bytes(mib)?,
            },
            None => ResidencyAuthorization::NotAuthorized,
        })
    }
}

/// Headroom the residency ladder leaves free on the device for activations,
/// workspaces and allocator slack: the shared admission-reserve shape (5% of
/// the pool total, capped at 1 GiB), so small and unified pools are not
/// over-reserved — a 24 GiB discrete card keeps the historical flat 1 GiB
/// exactly.
///
/// A declared margin, not a measured requirement. Adapters add known request
/// allocations separately. Runtime cache demotion can recover weight-load
/// allocation failures; this margin does not guarantee that every activation
/// allocation will fit.
fn device_residency_reserve_bytes(snapshot: &flyingfish::runtime::probe::ResourceSnapshot) -> u64 {
    let total = if snapshot.host_device_memory_is_unified == Some(true) {
        [
            snapshot.host_pool_total_bytes(),
            snapshot.device_total_memory_bytes,
        ]
        .into_iter()
        .flatten()
        .min()
    } else {
        snapshot.device_total_memory_bytes
    };
    // The free-view fallback below is for legacy recorded snapshots only: a
    // live capture produces the device total and free together (both come
    // from the same CUDA device handle).
    flyingfish::runtime::probe::admission_reserve_bytes(total.or(snapshot.device_free_memory_bytes))
}

/// Default CUDA placement for adapters with request tensor estimates. Explicit
/// ceilings (including zero) and non-CUDA devices keep their existing behavior.
/// `required_device_bytes` is one caller's device charge and is multiplied by
/// `unified_share`, the number of callers drawing from this pool.
/// `unified_host_bytes` is already the total across every concurrent host
/// allocation, because those exist per caller whether or not the caller shares
/// this pool — the two counts differ and only the caller knows the second. All
/// of it applies under the fold alone.
fn decide_auto_residency_with_required_memory(
    demands: &[flyingfish::runtime::residency::PhaseResidencyDemand],
    device: &candle_core::Device,
    args: DeviceCacheArgs,
    required_device_bytes: u64,
    unified_host_bytes: u64,
    unified_share: u64,
) -> Result<DeviceCache> {
    use flyingfish::runtime::{probe::ResourceSnapshot, residency::plan_device_residency};
    if !device.is_cuda() || args.device_cache_mib.is_some() {
        return decide_residency_with_required_memory(
            demands,
            device,
            args,
            required_device_bytes,
            unified_host_bytes,
            unified_share,
        );
    }
    let snapshot = ResourceSnapshot::capture(Some(device));
    // Each concurrent caller needs its own fixed margin, but only where the
    // pool is actually shared; a discrete device owes nothing to the others.
    let shared = snapshot.unified_pool_available_bytes().is_some();
    let share = if shared { unified_share.max(1) } else { 1 };
    let reserve = required_device_bytes
        .saturating_mul(share)
        .checked_add(device_residency_reserve_bytes(&snapshot).saturating_mul(share))
        .context("device residency reserve overflow")?;
    let capacity = if snapshot.unified_accounting_is_undecidable() {
        0
    } else {
        snapshot
            .unified_pool_available_bytes()
            .or(snapshot.device_free_memory_bytes)
            .unwrap_or(0)
    };
    let reserve = if shared {
        reserve
            .checked_add(unified_host_bytes)
            .context("unified host reserve overflow")?
    } else {
        reserve
    };
    // Reserving each concurrent caller's charges is not enough: the remainder
    // is also shared, so one caller may retain only its portion of it.
    let available = capacity.saturating_sub(reserve) / share;
    let plan = plan_device_residency(demands, available, 0)?;
    eprintln!(
        "device residency: auto retains {} MiB in {}/{} weight groups after reserving {} MiB for request tensors and workspace",
        plan.resident_bytes / (1 << 20),
        plan.placed.len(),
        demands.len(),
        reserve / (1 << 20),
    );
    Ok(DeviceCache::with_selected_phases(
        DeviceCachePolicy::with_max_bytes(plan.resident_bytes)
            .with_cuda_allocator(args.device_cache_allocator),
        plan.placed,
    ))
}

fn decide_residency_with_required_memory(
    demands: &[flyingfish::runtime::residency::PhaseResidencyDemand],
    device: &candle_core::Device,
    args: DeviceCacheArgs,
    required_device_bytes: u64,
    unified_host_bytes: u64,
    unified_share: u64,
) -> Result<DeviceCache> {
    use flyingfish::runtime::{probe::ResourceSnapshot, residency::decide_device_residency};
    let snapshot = ResourceSnapshot::capture(Some(device));
    let shared = snapshot.unified_pool_available_bytes().is_some();
    let share = if shared { unified_share.max(1) } else { 1 };
    let reserve = device_residency_reserve_bytes(&snapshot)
        .saturating_mul(share)
        .checked_add(required_device_bytes.saturating_mul(share))
        .context("device residency reserve overflow")?;
    // An explicit ceiling is still clamped against the shared pool, so the
    // caller's separately modelled host allocation is reserved here too.
    let reserve = if shared {
        reserve
            .checked_add(unified_host_bytes)
            .context("unified host reserve overflow")?
    } else {
        reserve
    };
    // The share is taken before planning, by withholding the other callers'
    // portions as reserve: dividing the decision afterwards would leave a plan
    // built against the whole remainder, selecting phases that can never be
    // retained. The operator's own ceiling still clamps on top, so a ceiling
    // that already fits the share is not reduced.
    let reserve = match snapshot.unified_pool_available_bytes() {
        Some(pool) if share > 1 => pool.saturating_sub(pool.saturating_sub(reserve) / share),
        _ => reserve,
    };
    let decision = decide_device_residency(&snapshot, demands, reserve, args.authorization()?)?;
    if decision.authorized_budget_bytes > 0 {
        eprintln!(
            "device residency: {} MiB authorized, {}/{} phase working sets fit, {} MiB planned weight peak",
            decision.authorized_budget_bytes / (1 << 20),
            decision.plan.placed.len(),
            demands.len(),
            decision.plan.resident_bytes / (1 << 20),
        );
        for phase in &decision.plan.spilled {
            eprintln!("device residency: {phase} cannot be retained in full within this budget");
        }
    }
    Ok(DeviceCache::with_selected_phases(
        DeviceCachePolicy::with_max_bytes(decision.authorized_budget_bytes)
            .with_cuda_allocator(args.device_cache_allocator),
        decision.plan.placed,
    ))
}

#[derive(Clone, Copy, Debug, Default, clap::Args)]
struct OptionalWeightCacheArgs {
    #[arg(help = "Keep weight shards file-backed or read them into host memory")]
    #[arg(long, value_enum)]
    weight_source: Option<WeightSource>,
    #[arg(
        help = "Host residency ceiling in MiB; replaces the default one-shard bound. A single oversized unit may still reside"
    )]
    #[arg(long)]
    host_cache_mib: Option<u64>,
    #[arg(
        long,
        value_enum,
        help = "Retain whole source shards or independently bounded raw tensor ranges"
    )]
    host_cache_granularity: Option<CacheGranularity>,
}

impl OptionalWeightCacheArgs {
    fn is_explicit(self) -> bool {
        self.weight_source.is_some()
            || self.host_cache_mib.is_some()
            || self.host_cache_granularity.is_some()
    }

    fn configured(self) -> WeightCacheArgs {
        WeightCacheArgs {
            weight_source: self.weight_source.unwrap_or(WeightSource::Mmap),
            host_cache_mib: self.host_cache_mib,
            host_cache_granularity: self.host_cache_granularity.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Default, clap::Args)]
struct H3ResourceArgs {
    #[arg(long, value_enum, default_value_t = flyingfish::runtime::resource_selection::ResourcePolicyMode::Performance)]
    resource_policy: flyingfish::runtime::resource_selection::ResourcePolicyMode,
    #[arg(long)]
    resource_evidence: Option<PathBuf>,
    #[arg(
        long,
        help = "Linux benchmark only: evict and verify checkpoint file-cache pages before payload loading"
    )]
    resource_cold_cache: bool,
}

#[derive(Clone, Copy, Debug, Default, clap::Args)]
struct H3AdmissionArgs {
    #[arg(long)]
    max_host_mib: Option<u64>,
    #[arg(long)]
    max_device_mib: Option<u64>,
    #[arg(long)]
    backend_workspace_mib: Option<u64>,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(subcommand, about = "Score image/text similarity with CLIP")]
    Clip(clip::ClipCommand),
    #[command(
        subcommand,
        about = "Generate music from lyrics and a caption with MiniMax Music3"
    )]
    Music(music::MusicCommand),
    #[command(subcommand, about = "Run disk-streamed MiniCPM5 text generation")]
    Minicpm(minicpm::MiniCpmCommand),
    #[command(
        subcommand,
        about = "Discover, inspect and load local model components"
    )]
    Models(models::ModelsCommand),
    #[command(
        name = "verify-resource-evidence",
        about = "Verify retained paired resource observations and output identities"
    )]
    VerifyResourceEvidence {
        #[arg(long)]
        input: PathBuf,
    },
    #[command(
        subcommand,
        about = "Inspect or explicitly repack H3 Base checkpoint storage"
    )]
    Checkpoint(CheckpointCommand),
    #[command(about = "Read a best-effort hardware descriptor and point-in-time availability")]
    Probe {
        #[arg(help = "Device to inspect: auto, cpu, cuda:N, or metal:N")]
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(help = "Emit a stable machine-readable report")]
        #[arg(long)]
        json: bool,
    },
    #[command(about = "Inspect a safetensors checkpoint index")]
    Inspect {
        #[arg(help = "Directory containing a safetensors index")]
        #[arg(long)]
        checkpoint: PathBuf,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[arg(help = "Cross-check every index entry with safetensors shard headers")]
        #[arg(long)]
        verify: bool,
    },
    #[command(about = "Materialize one named tensor onto a device")]
    Tensor {
        #[arg(help = "Directory containing a safetensors index")]
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[command(flatten)]
        weights: WeightCacheArgs,
    },
    #[command(about = "Record one checkpoint tree's local size and modification identity")]
    Identify {
        #[arg(help = "Directory containing a safetensors index")]
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, help = "New JSON file written outside the checkpoint directory")]
        output: PathBuf,
    },
    #[command(about = "Compare two inference safetensors files and emit a numerical parity report")]
    Diff {
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        actual: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        atol: f64,
        #[arg(long, default_value_t = 1e-4)]
        rtol: f64,
        #[arg(help = "Accept extra tensors in the actual file (for checkpoint metadata)")]
        #[arg(long)]
        allow_unexpected: bool,
    },
    #[command(subcommand)]
    #[command(about = "Measure explicit sequential and local-interconnect I/O profiles")]
    Bench(BenchCommand),
    #[command(subcommand)]
    #[command(about = "Run MiniMax-H3 text-to-video-and-audio commands")]
    H3(H3Command),
    #[command(subcommand)]
    #[command(about = "Run disk-streamed GLM-5.3-Flash text commands")]
    Glm(GlmCommand),
    #[command(subcommand)]
    #[command(about = "Run TRELLIS structured-3D-latent commands")]
    Trellis(TrellisCommand),
    #[command(about = "Internal one-trial worker. The request is read from bounded stdin")]
    #[command(name = "__calibrate-t2va-trial", hide = true)]
    CalibrateT2vaTrial,
}

#[derive(Debug, Subcommand)]
enum CheckpointCommand {
    #[command(
        about = "Report Base transformer stages and storage capacity without reading tensor payloads"
    )]
    StageLayout {
        #[arg(long, help = "Released Base transformer component directory")]
        checkpoint: PathBuf,
        #[arg(
            long,
            help = "New JSON report outside the original model; stdout if omitted"
        )]
        output: Option<PathBuf>,
    },
    #[command(
        about = "Copy Base tensors byte-for-byte into explicit stage-sized safetensors files"
    )]
    RepackExactStages {
        #[arg(long, help = "Released Base transformer component directory")]
        checkpoint: PathBuf,
        #[arg(long, help = "New destination directory outside the original model")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum IoBenchmarkProfile {
    Sequential,
    LocalInterconnect,
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    #[command(about = "Measure explicit sequential or local-interconnect I/O profiles")]
    Io {
        #[arg(long, value_enum, default_value = "sequential")]
        profile: IoBenchmarkProfile,
        #[arg(
            long,
            help = "Explicit local payload file for the sequential profile",
            required_unless_present = "model",
            conflicts_with = "model"
        )]
        payload: Option<PathBuf>,
        #[arg(
            long,
            help = "GLM-5.3-Flash root for the local-interconnect expert profile",
            required_unless_present = "payload",
            conflicts_with = "payload"
        )]
        model: Option<PathBuf>,
        #[arg(help = "New path receiving the versioned I/O calibration report")]
        #[arg(long)]
        output: PathBuf,
        #[arg(help = "Device for optional host-to-device copy: cpu, cuda:N, or metal:N")]
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(help = "Payload format recorded in the hardware-and-format key")]
        #[arg(long, default_value = "raw_sequential")]
        format: String,
        #[arg(
            long,
            help = "Explicit peer cuda:N; required with the local profile when multiple GPUs are visible"
        )]
        peer_device: Option<String>,
        #[arg(long, default_value_t = flyingfish::interconnect_benchmark::DEFAULT_LOCAL_IO_WARMUP_ITERATIONS)]
        warmups: usize,
        #[arg(long, default_value_t = flyingfish::interconnect_benchmark::DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS)]
        samples: usize,
        #[arg(long, default_value_t = 3)]
        expert_layer: usize,
        #[arg(long, default_value_t = 0)]
        expert_index: usize,
    },
    #[command(
        about = "Measure a concurrent ring all-reduce of the H3 block cut across explicit CUDA \
                 devices"
    )]
    Collective {
        #[arg(help = "New path receiving the versioned collective benchmark report")]
        #[arg(long)]
        output: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            help = "The ring, in order, as explicit cuda:N devices. Rank r receives from rank r-1"
        )]
        devices: Vec<String>,
        #[arg(
            long,
            value_delimiter = ',',
            default_values_t = flyingfish::collective_benchmark::DEFAULT_COLLECTIVE_RANK_COUNTS,
            help = "Rank counts to measure; one that exceeds the named devices is recorded as \
                    unavailable rather than approximated"
        )]
        ranks: Vec<usize>,
        #[arg(
            long,
            help = "One single-device evaluation, in seconds, measured on this host. Without it \
                    no tensor-split projection is made: the time belongs to one GPU running one \
                    model and cannot be assumed"
        )]
        single_device_evaluation_seconds: Option<f64>,
        #[arg(
            long,
            requires = "single_device_evaluation_seconds",
            help = "Interconnect rate the projection charges communication at. Defaults to this \
                    run's own measured point-to-point reference"
        )]
        charged_bytes_per_second: Option<f64>,
        #[arg(long, default_value_t = flyingfish::interconnect_benchmark::DEFAULT_LOCAL_IO_WARMUP_ITERATIONS)]
        warmups: usize,
        #[arg(long, default_value_t = flyingfish::interconnect_benchmark::DEFAULT_LOCAL_IO_SAMPLE_ITERATIONS)]
        samples: usize,
    },
}

struct CliDenoiseObserver {
    progress: progress::DenoiseProgress,
}

impl CliDenoiseObserver {
    fn new(enabled: bool) -> Self {
        Self {
            progress: progress::DenoiseProgress::for_stderr(enabled),
        }
    }
}

fn build_transformer_options(
    device: Device,
    policy: &ExecutionPolicy,
) -> Result<StreamedTransformerOptions> {
    policy.validate_device(&device)?;
    Ok(StreamedTransformerOptions {
        weight_source: policy.weight_source(),
        cache_policy: policy.cache_policy()?,
        device,
        chunking: policy.transformer_chunking()?,
        flash_attention: policy.flash_attention(),
        device_cache_policy: policy.weights.device_cache,
        host_phase_priority: policy.weights.host_phase_priority,
    })
}

fn report_h3_device_cache(transformer: &StreamedTransformer) {
    if let Some(plan) = transformer.host_residency_plan() {
        eprintln!(
            "H3 host cache: {}",
            serde_json::json!({"planned_resident_bytes":plan.resident_bytes,"stats":transformer.cache_stats()})
        );
    }
    if let Some(plan) = transformer.device_residency_plan() {
        eprintln!(
            "H3 device cache: {}",
            serde_json::json!({
                "planned_resident_bytes": plan.resident_bytes,
                "stats": transformer.device_cache_stats(),
            })
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_execution_policy(
    path: Option<&Path>,
    device: &Device,
    weight_source: WeightSource,
    cache_policy: CachePolicy,
    chunks: TransformerChunkArgs,
    flash_attention: bool,
    precompute_adaln: bool,
) -> Result<ExecutionPolicy> {
    let policy = match path {
        Some(path) => ExecutionPolicy::load(path)?,
        None => ExecutionPolicy::from_runtime(
            device,
            weight_source,
            cache_policy,
            chunks.model_chunking(),
            flash_attention,
            precompute_adaln,
        )?,
    };
    validate_executable_policy(&policy, device)?;
    Ok(policy)
}

fn validate_executable_policy(policy: &ExecutionPolicy, device: &Device) -> Result<()> {
    policy.validate_device(device)?;
    anyhow::ensure!(
        !policy.flash_attention() || cfg!(feature = "flash-attn"),
        "execution policy selects FlashAttention, but this binary was not compiled with --features flash-attn"
    );
    Ok(())
}

/// Choose an attention backend the request can actually run on.
///
/// The exact full-softmax path covers a bounded number of packed rows. Past
/// that bound it is not that the operator prefers another backend -- full
/// softmax does not run the request at all -- and the row count is known here.
/// Online softmax processes keys in blocks instead, so the bound applies per
/// block rather than to the sequence.
///
/// The block is the largest the verified exact-softmax kernels cover, which is
/// the same bound the full path ran out of: larger blocks mean fewer passes,
/// and the modelled peak grows slowly enough with block size that memory does
/// not decide this. A pinned or replayed policy is left alone, and so is one
/// that already names a backend without the bound.
fn promote_attention_backend_for_rows(
    policy: &mut ExecutionPolicy,
    on_cuda: bool,
    packed_rows: u64,
    operator_named_the_backend: bool,
) -> Result<()> {
    use flyingfish::h3::core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
    use flyingfish::h3::policy::AttentionBackendPolicy;

    let bound = u64::try_from(CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS)
        .context("exact softmax row bound exceeds u64")?;
    if operator_named_the_backend
        || !on_cuda
        || policy.attention.backend != AttentionBackendPolicy::FullSoftmax
        || packed_rows <= bound
    {
        return Ok(());
    }
    policy.attention.backend = AttentionBackendPolicy::OnlineSoftmax;
    policy.attention.configured_key_rows = Some(bound);
    policy.rebind_attention_numerics()?;
    eprintln!(
        "attention: {packed_rows} packed rows exceed the {bound} full softmax covers; \
         using online softmax over {bound}-row key blocks"
    );
    Ok(())
}

fn select_resume_execution_policy(
    mut requested: ExecutionPolicy,
    recorded: Option<&ExecutionPolicy>,
    explicit_policy: bool,
    explicit_policy_settings: bool,
    requires_recorded_policy: bool,
) -> Result<ExecutionPolicy> {
    match recorded {
        Some(recorded) => {
            if explicit_policy || explicit_policy_settings {
                // Device residency is planned against whatever the card has
                // free at that moment, so the recorded ceiling describes the
                // first run's machine rather than anything the operator asked
                // for. Two runs minutes apart can plan differently and agree on
                // every choice a person made. Carry the recorded ceiling
                // forward -- keeping the run's placement stable across a
                // resume -- and let admission decide whether it still fits;
                // refusing the resume for it would name a field no H3 command
                // even exposes as a flag.
                if !explicit_policy {
                    requested.weights.device_cache = recorded.weights.device_cache;
                }
                if let Some(field) = recorded.first_difference(&requested) {
                    bail!(
                        "the checkpoint's execution policy disagrees with the requested one at {field}"
                    );
                }
                Ok(requested)
            } else {
                Ok(recorded.clone())
            }
        }
        None if requires_recorded_policy => bail!("checkpoint has no execution policy"),
        None => Ok(requested),
    }
}

#[cfg(test)]
fn default_execution_policy(device: &Device) -> ExecutionPolicy {
    ExecutionPolicy::from_runtime(
        device,
        WeightSource::Mmap,
        CachePolicy::new(1),
        TransformerChunking::default(),
        false,
        true,
    )
    .expect("default execution policy is valid")
}

impl DenoiseObserver for CliDenoiseObserver {
    fn synchronize_device_timings(&self) -> bool {
        self.progress.synchronize_device_timings()
    }

    fn on_preparation_completed(&mut self, event: DenoisePreparationEvent) -> Result<()> {
        self.progress
            .on_preparation_completed(event, &mut std::io::stderr().lock())
    }

    fn on_step_completed(&mut self, event: DenoiseStepEvent) -> Result<()> {
        self.progress
            .on_step_completed(event, &mut std::io::stderr().lock())
    }
}

pub(crate) fn run() -> Result<()> {
    routing::run()
}

fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Clip(command) => clip::run(command),
        Command::Music(command) => music::run(command),
        Command::Minicpm(command) => minicpm::run(command),
        Command::Models(command) => models::run(command),
        Command::VerifyResourceEvidence { input } => resource::verify_evidence(&input),
        Command::Checkpoint(command) => checkpoint::run(command),
        command @ Command::Probe { .. } => probe::run_probe(command),
        command @ Command::Inspect { .. } => inspect::run_inspect(command),
        command @ Command::Tensor { .. } => inspect::run_tensor(command),
        command @ Command::Identify { .. } => identity::run_identify_model(command),
        command @ Command::Diff { .. } => plan::run_compare_tensors(command),
        Command::Bench(BenchCommand::Collective {
            output,
            devices,
            ranks,
            single_device_evaluation_seconds,
            charged_bytes_per_second,
            warmups,
            samples,
        }) => collective::run_collective(
            output,
            devices,
            ranks,
            single_device_evaluation_seconds,
            charged_bytes_per_second,
            warmups,
            samples,
        ),
        Command::Bench(command) => calibrate_io::run_calibrate_io(command),
        Command::H3(command) => command.run(),
        Command::Trellis(command) => command.run(),
        Command::Glm(command) => command.run(),
        Command::CalibrateT2vaTrial => calibrate::run_calibrate_t2va_trial(),
    }
}

fn take_input(values: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    values
        .remove(name)
        .with_context(|| format!("input safetensors is missing {name}"))
}

fn sorted_tensor_names(values: &HashMap<String, Tensor>) -> Vec<String> {
    let mut names = values.keys().cloned().collect::<Vec<_>>();
    names.sort_unstable();
    names
}

#[allow(clippy::too_many_arguments)]
fn make_t2va_noise(
    config: &TransformerConfig,
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    audio_frames: usize,
    audio_channels: usize,
    seed: u64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let video_count = config
        .in_channels
        .checked_mul(latent_frames)
        .and_then(|value| value.checked_mul(latent_height))
        .and_then(|value| value.checked_mul(latent_width))
        .context("video latent element count overflow")?;
    let video_values = (0..video_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    let video_latents = Tensor::from_vec(
        video_values,
        (
            1,
            config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        ),
        device,
    )?;
    let audio_count = audio_channels
        .checked_mul(config.audio_in_channels)
        .and_then(|value| value.checked_mul(audio_frames))
        .context("audio latent element count overflow")?;
    let audio_values = (0..audio_count)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect::<Vec<f32>>();
    let audio_latents = Tensor::from_vec(
        audio_values,
        (audio_channels, config.audio_in_channels, audio_frames),
        device,
    )?;
    Ok((video_latents, audio_latents))
}

fn mib_to_bytes(value: u64) -> Result<u64> {
    value
        .checked_mul(1024 * 1024)
        .context("MiB resource limit exceeds u64")
}

fn resolve_optional_new_output(path: Option<PathBuf>, model: &Path) -> Result<Option<PathBuf>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let path = resolve_output_outside_model(&path, model)?;
    ensure_new_output(&path, "telemetry output")?;
    Ok(Some(path))
}

fn ensure_optional_output_is_distinct(path: Option<&Path>, reserved: &[&Path]) -> Result<()> {
    if let Some(path) = path {
        anyhow::ensure!(
            reserved.iter().all(|reserved| *reserved != path),
            "telemetry output conflicts with a primary output: {}",
            path.display()
        );
    }
    Ok(())
}

/// The directory a new output will be created in.
///
/// `Path::parent` answers `Some("")` for a bare relative file name, not `None`,
/// so `unwrap_or(".")` never fires there and leaves an empty path that every
/// later `is_dir` or `canonicalize` rejects. Both spellings name the working
/// directory. `ArtifactStaging` already resolves this the same way.
fn output_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn ensure_new_output(path: &Path, label: &str) -> Result<()> {
    anyhow::ensure!(!path.exists(), "{label} already exists: {}", path.display());
    let parent = output_parent(path);
    anyhow::ensure!(
        parent.is_dir(),
        "{label} parent directory does not exist: {}",
        parent.display()
    );
    Ok(())
}

fn create_new_directory(path: &Path, label: &str) -> Result<()> {
    std::fs::create_dir(path)
        .with_context(|| format!("failed to create {label} {}", path.display()))?;
    let parent = output_parent(path);
    sync_parent_directory(parent)
        .with_context(|| format!("failed to synchronize {label} parent {}", parent.display()))?;
    Ok(())
}

fn resolve_output_outside_model(path: &Path, model: &Path) -> Result<PathBuf> {
    let model = std::fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let candidate = if std::fs::symlink_metadata(path).is_ok() {
        std::fs::canonicalize(path)
            .with_context(|| format!("failed to resolve existing output {}", path.display()))?
    } else {
        let parent = output_parent(path);
        let parent = std::fs::canonicalize(parent)
            .with_context(|| format!("failed to resolve output parent {}", parent.display()))?;
        match path.file_name() {
            Some(name) => parent.join(name),
            None => parent,
        }
    };
    anyhow::ensure!(
        !candidate.starts_with(&model),
        "refusing to write output inside the model directory: {}",
        candidate.display()
    );
    Ok(candidate)
}

fn write_telemetry(path: &Path, monitor: TelemetryMonitor) -> Result<()> {
    let report = monitor.finish()?;
    let json = serde_json::to_vec_pretty(&report)?;
    let staging = flyingfish::runtime::artifact::ArtifactStaging::new(path)
        .with_context(|| format!("failed to stage telemetry report {}", path.display()))?;
    publish_staged_bytes(staging, &json)?;
    println!("saved runtime telemetry to {}", path.display());
    Ok(())
}

fn publish_staged_bytes(
    staging: flyingfish::runtime::artifact::ArtifactStaging,
    bytes: &[u8],
) -> Result<flyingfish::runtime::artifact::PublishedArtifact> {
    staging.write_bytes(bytes)?;
    Ok(staging.publish()?)
}

fn publish_png_frame_manifest(directory: &Path, frame_count: usize) -> Result<()> {
    const FILE_NAME: &str = "frames.manifest.json";
    let manifest = PngFrameSetManifest::collect(directory, FILE_NAME, frame_count)?;
    let json = manifest.canonical_json()?;
    let path = directory.join(FILE_NAME);
    let staging_parent = directory
        .parent()
        .context("PNG frame directory has no staging parent")?;
    let staging = flyingfish::runtime::artifact::ArtifactStaging::new_with_staging_parent(
        &path,
        staging_parent,
    )
    .with_context(|| format!("failed to stage PNG frame-set manifest {}", path.display()))?;
    publish_staged_bytes(staging, &json)?;
    manifest.verify_completed_directory(directory, FILE_NAME)?;
    Ok(())
}

fn resolve_component(model: &Path, component: &Path) -> Result<PathBuf> {
    anyhow::ensure!(
        model.is_dir(),
        "model directory does not exist: {}",
        model.display()
    );
    anyhow::ensure!(
        !component.is_absolute(),
        "component must be relative to the model directory"
    );
    anyhow::ensure!(
        component
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))),
        "component path may not contain . or .."
    );
    let model = std::fs::canonicalize(model)
        .with_context(|| format!("failed to resolve model directory {}", model.display()))?;
    let unresolved = model.join(component);
    let result = std::fs::canonicalize(&unresolved).with_context(|| {
        format!(
            "failed to resolve component directory {}",
            unresolved.display()
        )
    })?;
    anyhow::ensure!(
        result.is_dir(),
        "component directory does not exist: {}",
        result.display()
    );
    anyhow::ensure!(
        result.starts_with(&model),
        "component directory escapes the model root: {}",
        result.display()
    );
    Ok(result)
}

fn parse_device(value: &str) -> Result<Device> {
    if value == "cpu" {
        return Ok(Device::Cpu);
    }
    if value == "auto" {
        return Device::cuda_if_available(0).context("failed to initialize automatic device");
    }
    if let Some(ordinal) = value.strip_prefix("cuda:") {
        let ordinal = ordinal
            .parse::<usize>()
            .context("invalid CUDA device ordinal")?;
        return Device::new_cuda(ordinal).context("failed to initialize CUDA device");
    }
    if let Some(ordinal) = value.strip_prefix("metal:") {
        let ordinal = ordinal
            .parse::<usize>()
            .context("invalid Metal device ordinal")?;
        return Device::new_metal(ordinal).context("failed to initialize Metal device");
    }
    bail!("unknown device {value:?}; use cpu, auto, cuda:N, or metal:N")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn snap(
        unified: Option<bool>,
        host_total: Option<u64>,
        device_total: Option<u64>,
        device_free: Option<u64>,
    ) -> flyingfish::runtime::probe::ResourceSnapshot {
        flyingfish::runtime::probe::ResourceSnapshot {
            schema_version: flyingfish::runtime::probe::RESOURCE_SNAPSHOT_SCHEMA_VERSION,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: Some(host_total.unwrap_or(0)),
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: device_free,
            host_device_memory_is_unified: unified,
            device_topology_probe_failed: false,
            host_memory_total_bytes: host_total,
            device_total_memory_bytes: device_total,
            measurement_scope: flyingfish::runtime::probe::ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        }
    }

    #[test]
    fn device_residency_reserve_matches_the_flat_margin_on_a_24gib_card() {
        let snapshot = snap(Some(false), Some(64 << 30), Some(24 << 30), Some(20 << 30));
        // 24 GiB / 20 = 1.2 GiB > the 1 GiB cap: the historical flat margin,
        // pinned so the baseline cannot drift.
        assert_eq!(device_residency_reserve_bytes(&snapshot), 1 << 30);
    }

    #[test]
    fn device_residency_reserve_scales_down_on_small_unified_pools() {
        let snapshot = snap(Some(true), Some(6 << 30), Some(6 << 30), Some(4 << 30));
        assert_eq!(device_residency_reserve_bytes(&snapshot), 512 << 20);
    }

    #[test]
    fn device_residency_reserve_keeps_the_cap_without_totals() {
        // No totals at all: the historical flat cap, not a free-view guess.
        let snapshot = snap(None, None, None, None);
        assert_eq!(device_residency_reserve_bytes(&snapshot), 1 << 30);
        // A free view without a total is a LEGACY-RECORD shape only (a live
        // capture produces the total and free together); it mirrors the GLM
        // axis fallback so replayed selections reproduce their era's reserve.
        let snapshot = snap(None, None, None, Some(4 << 30));
        assert_eq!(device_residency_reserve_bytes(&snapshot), 512 << 20);
    }

    #[test]
    fn parses_calibrate_io_command() {
        let args = Args::try_parse_from([
            "ff",
            "bench",
            "io",
            "--payload",
            "payload.bin",
            "--output",
            "io.json",
            "--device",
            "cpu",
        ])
        .unwrap();
        match args.command {
            Command::Bench(BenchCommand::Io {
                profile,
                payload,
                output,
                device,
                format,
                ..
            }) => {
                assert_eq!(profile, IoBenchmarkProfile::Sequential);
                assert_eq!(payload, Some(PathBuf::from("payload.bin")));
                assert_eq!(output, PathBuf::from("io.json"));
                assert_eq!(device, "cpu");
                assert_eq!(format, "raw_sequential");
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn parses_local_interconnect_io_profile() {
        let args = Args::try_parse_from([
            "ff",
            "bench",
            "io",
            "--profile",
            "local-interconnect",
            "--model",
            "GLM-5.3-Flash",
            "--output",
            "local-io.json",
            "--device",
            "cuda:0",
            "--samples",
            "7",
            "--peer-device",
            "cuda:1",
        ])
        .unwrap();
        match args.command {
            Command::Bench(BenchCommand::Io {
                profile,
                model,
                device,
                peer_device,
                samples,
                ..
            }) => {
                assert_eq!(profile, IoBenchmarkProfile::LocalInterconnect);
                assert_eq!(model, Some(PathBuf::from("GLM-5.3-Flash")));
                assert_eq!(device, "cuda:0");
                assert_eq!(peer_device.as_deref(), Some("cuda:1"));
                assert_eq!(samples, 7);
            }
            _ => panic!("parsed the wrong command"),
        }

        assert!(
            Args::try_parse_from([
                "ff",
                "bench",
                "io",
                "--profile",
                "local-interconnect",
                "--device",
                "cuda:0",
                "--output",
                "local-io.json",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from(["ff", "bench", "io", "--output", "sequential.json",]).is_err()
        );
    }

    #[test]
    fn parses_resource_plan_command() {
        let args = Args::try_parse_from([
            "ff",
            "h3",
            "plan",
            "--model",
            "model",
            "--text-rows",
            "12",
            "--max-device-mib",
            "8192",
            "--json",
        ])
        .unwrap();
        match args.command {
            Command::H3(H3Command::PlanT2va {
                text_rows,
                chunks,
                max_device_mib,
                json,
                ..
            }) => {
                assert_eq!(text_rows, 12);
                let exact = chunks.configured(false);
                assert_eq!(
                    exact.attention_query_chunk_size.get(),
                    DEFAULT_ATTENTION_QUERY_CHUNK_SIZE
                );
                assert_eq!(
                    exact.attention_projection_chunk_size.get(),
                    DEFAULT_ATTENTION_PROJECTION_CHUNK_SIZE
                );
                assert_eq!(
                    exact.ffn_token_chunk_size.get(),
                    DEFAULT_FFN_TOKEN_CHUNK_SIZE
                );
                assert_eq!(
                    exact.output_token_chunk_size.get(),
                    DEFAULT_OUTPUT_TOKEN_CHUNK_SIZE
                );
                assert_eq!(exact.attention_key_policy(), AttentionKeyChunkPolicy::Full);
                assert_eq!(
                    chunks
                        .configured(true)
                        .attention_projection_chunk_size
                        .get(),
                    DEFAULT_FLASH_ATTENTION_PROJECTION_CHUNK_SIZE
                );
                assert_eq!(max_device_mib, Some(8192));
                assert!(json);
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn shared_chunk_args_reject_zero_and_flash_conflicts() {
        assert!(
            Args::try_parse_from([
                "ff",
                "h3",
                "plan",
                "--model",
                "model",
                "--text-rows",
                "1",
                "--attention-query-chunk-size",
                "0",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "ff",
                "h3",
                "denoise",
                "--model",
                "model",
                "--inputs",
                "inputs.safetensors",
                "--output",
                "output.safetensors",
                "--attention-key-chunk-size",
                "128",
                "--flash-attention",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "ff",
                "h3",
                "generate",
                "--model",
                "model",
                "--prompt",
                "test",
                "--output-dir",
                "output",
                "--attention-key-chunk-size",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn shared_weight_cache_args_preserve_required_and_optional_defaults() {
        let args = Args::try_parse_from([
            "ff",
            "inspect",
            "--checkpoint",
            "model",
            "--weight-source",
            "memory",
            "--host-cache-mib",
            "4",
        ])
        .unwrap();
        let Command::Inspect { weights, .. } = args.command else {
            panic!("expected inspect command")
        };
        assert_eq!(weights.weight_source, WeightSource::Memory);
        assert_eq!(weights.host_cache_mib, Some(4));

        let args = Args::try_parse_from([
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "inputs.safetensors",
            "--output",
            "output.safetensors",
        ])
        .unwrap();
        let Command::H3(H3Command::DenoiseT2va { weights, .. }) = args.command else {
            panic!("expected denoise command")
        };
        assert!(!weights.is_explicit());
        let weights = weights.configured();
        assert_eq!(weights.weight_source, WeightSource::Mmap);
        assert_eq!(weights.host_cache_mib, None);
    }

    #[test]
    fn tensor_cache_granularity_is_explicit_and_conflicts_with_policy_replay() {
        let args = Args::try_parse_from([
            "ff",
            "tensor",
            "--checkpoint",
            "model",
            "--name",
            "weight",
            "--host-cache-granularity",
            "tensor",
            "--host-cache-mib",
            "2",
        ])
        .unwrap();
        let Command::Tensor { weights, .. } = args.command else {
            panic!("expected tensor command")
        };
        let policy = weights.cache_policy().unwrap();
        assert_eq!(policy.granularity, CacheGranularity::Tensor);
        assert_eq!(policy.max_bytes, Some(2 * 1024 * 1024));
        assert_eq!(policy.max_shards, usize::MAX);
        assert!(
            Args::try_parse_from([
                "ff",
                "h3",
                "denoise",
                "--model",
                "model",
                "--inputs",
                "input",
                "--output",
                "output",
                "--policy",
                "policy.json",
                "--host-cache-granularity",
                "tensor"
            ])
            .is_err()
        );
    }

    #[test]
    fn host_cache_mib_replaces_the_one_shard_bound() {
        let args = Args::try_parse_from(["ff", "inspect", "--checkpoint", "model"]).unwrap();
        let Command::Inspect { weights, .. } = args.command else {
            panic!("expected inspect command")
        };
        let policy = weights.cache_policy().unwrap();
        assert_eq!(policy.max_shards, 1);
        assert_eq!(policy.max_bytes, None);

        let args =
            Args::try_parse_from(["ff", "tensor", "--checkpoint", "model", "--name", "weight"])
                .unwrap();
        let Command::Tensor { weights, .. } = args.command else {
            panic!("expected tensor command")
        };
        let policy = weights.cache_policy().unwrap();
        assert_eq!(policy.max_shards, 1);
        assert_eq!(policy.max_bytes, None);

        let args = Args::try_parse_from([
            "ff",
            "inspect",
            "--checkpoint",
            "model",
            "--host-cache-mib",
            "8",
        ])
        .unwrap();
        let Command::Inspect { weights, .. } = args.command else {
            panic!("expected inspect command")
        };
        let policy = weights.cache_policy().unwrap();
        assert_eq!(
            policy,
            CachePolicy::unbounded_units().with_max_bytes(8 * 1024 * 1024)
        );

        let args = Args::try_parse_from([
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "inputs.safetensors",
            "--output",
            "output.safetensors",
            "--host-cache-mib",
            "4",
        ])
        .unwrap();
        let Command::H3(H3Command::DenoiseT2va { weights, .. }) = args.command else {
            panic!("expected denoise command")
        };
        let policy = weights.configured().cache_policy().unwrap();
        assert_eq!(
            policy,
            CachePolicy::unbounded_units().with_max_bytes(4 * 1024 * 1024)
        );
    }

    #[test]
    fn host_cache_shards_is_rejected_and_hidden_component_still_parses() {
        assert!(
            Args::try_parse_from([
                "ff",
                "inspect",
                "--checkpoint",
                "model",
                "--host-cache-shards",
                "2",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "ff",
                "h3",
                "denoise",
                "--model",
                "model",
                "--inputs",
                "inputs.safetensors",
                "--output",
                "output.safetensors",
                "--host-cache-shards",
                "2",
            ])
            .is_err()
        );
        let args = Args::try_parse_from([
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "inputs.safetensors",
            "--output",
            "output.safetensors",
            "--component",
            "FL2VA/transformer",
        ])
        .unwrap();
        let Command::H3(H3Command::DenoiseT2va { component, .. }) = args.command else {
            panic!("expected denoise command")
        };
        assert_eq!(component, PathBuf::from("FL2VA/transformer"));
    }

    #[test]
    fn every_command_has_a_nonempty_about() {
        fn walk(command: &clap::Command) {
            let name = command.get_name();
            if name != "help" {
                let about = command
                    .get_about()
                    .map(|text| text.to_string())
                    .unwrap_or_default();
                assert!(
                    !about.trim().is_empty(),
                    "command `{name}` is missing a non-empty about string"
                );
            }
            for subcommand in command.get_subcommands() {
                walk(subcommand);
            }
        }
        walk(&Args::command());
    }

    #[test]
    fn pinned_policy_conflicts_with_resource_flags_but_allows_device_selection() {
        let base = [
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "inputs.safetensors",
            "--output",
            "output.safetensors",
            "--policy",
            "policy.json",
        ];
        let mut with_chunk = base.to_vec();
        with_chunk.extend(["--ffn-token-chunk-size", "64"]);
        assert!(Args::try_parse_from(with_chunk).is_err());

        let mut with_weights = base.to_vec();
        with_weights.extend(["--host-cache-mib", "2"]);
        assert!(Args::try_parse_from(with_weights).is_err());

        let mut with_device = base.to_vec();
        with_device.extend(["--device", "cpu"]);
        assert!(Args::try_parse_from(with_device).is_ok());
    }

    #[test]
    fn recorded_nondefault_policy_is_automatically_replayed_from_cli_defaults() {
        let defaults = default_execution_policy(&Device::Cpu);
        let mut recorded = defaults.clone();
        recorded.configured_output_rows = 17;
        let selected =
            select_resume_execution_policy(defaults.clone(), Some(&recorded), false, false, true)
                .unwrap();
        assert_eq!(selected, recorded);

        let mut wrong_backend = recorded;
        wrong_backend.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        let selected = select_resume_execution_policy(
            defaults.clone(),
            Some(&wrong_backend),
            false,
            false,
            true,
        )
        .unwrap();
        assert!(validate_executable_policy(&selected, &Device::Cpu).is_err());
    }

    /// A request past what full softmax covers is not a preference for another
    /// backend: full softmax does not run it. The row count decides, and the
    /// operator only has to name a backend when they want a particular one.
    #[test]
    fn a_request_past_the_full_softmax_bound_selects_online_softmax() {
        use flyingfish::h3::core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS as BOUND;
        use flyingfish::h3::policy::AttentionBackendPolicy;

        let bound = BOUND as u64;
        let cuda = |rows: u64, pinned: bool| {
            let mut policy = default_execution_policy(&Device::Cpu);
            policy.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
            *policy.numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
                flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
                AttentionBackendPolicy::FullSoftmax,
            )
            .unwrap();
            promote_attention_backend_for_rows(&mut policy, false, rows, pinned).unwrap();
            policy
        };

        // The device decides too: this is a CUDA bound, and a CPU run is left
        // alone whatever the row count.
        let on_cpu = cuda(bound + 1, false);
        assert_eq!(
            on_cpu.attention.backend,
            AttentionBackendPolicy::FullSoftmax
        );

        let mut past = default_execution_policy(&Device::Cpu);
        past.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        *past.numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
            AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        let mut within = past.clone();

        // Exactly at the bound is still covered; one row past it is not.
        promote_attention_backend_for_rows(&mut within, true, bound, false).unwrap();
        assert_eq!(
            within.attention.backend,
            AttentionBackendPolicy::FullSoftmax
        );

        promote_attention_backend_for_rows(&mut past, true, bound + 1, false).unwrap();
        assert_eq!(
            past.attention.backend,
            AttentionBackendPolicy::OnlineSoftmax
        );
        assert_eq!(past.attention.configured_key_rows, Some(bound));
        past.validate().unwrap();
        assert_eq!(
            past.transformer_chunking().unwrap().attention.key,
            flyingfish::h3::core::AttentionKeyChunkPolicy::chunked(BOUND).unwrap(),
            "the chunking a run derives must follow the promoted policy"
        );

        // A pinned policy named its backend on purpose.
        let mut pinned = default_execution_policy(&Device::Cpu);
        pinned.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        *pinned.numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
            AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        promote_attention_backend_for_rows(&mut pinned, true, bound + 1, true).unwrap();
        assert_eq!(
            pinned.attention.backend,
            AttentionBackendPolicy::FullSoftmax
        );
    }

    /// A resume must not be refused for a residency ceiling the operator never
    /// asked for. `ff video generate` exposes no device-cache flag at all: the
    /// value in a recorded policy came from planning against the free memory
    /// that one card had at that one moment.
    #[test]
    fn resume_carries_the_recorded_device_residency_instead_of_replanning_it() {
        use flyingfish::runtime::weights::DeviceCachePolicy;

        let defaults = default_execution_policy(&Device::Cpu);
        let mut recorded = defaults.clone();
        recorded.weights.device_cache = DeviceCachePolicy::with_max_bytes(16_632_512_512);
        assert_eq!(
            defaults.first_difference(&recorded),
            Some("weights.device_cache"),
            "the fixture must differ in exactly the field under test"
        );

        // A flag like --flash-attention marks the settings explicit, which is
        // what drives the comparison. The residency ceiling still resumes.
        let selected =
            select_resume_execution_policy(defaults.clone(), Some(&recorded), false, true, true)
                .unwrap();
        assert_eq!(selected.weights.device_cache, recorded.weights.device_cache);
        assert_eq!(selected, recorded);

        // Anything the operator did choose is still compared.
        let mut other_rows = defaults.clone();
        other_rows.configured_output_rows += 1;
        other_rows.weights.device_cache = DeviceCachePolicy::with_max_bytes(1 << 30);
        let error = select_resume_execution_policy(other_rows, Some(&recorded), false, true, true)
            .unwrap_err();
        assert!(
            error.to_string().contains("configured_output_rows"),
            "unexpected error: {error}"
        );

        // A pinned policy file names every field on purpose, so it is compared
        // whole -- including the ceiling it pins.
        let error = select_resume_execution_policy(defaults, Some(&recorded), true, false, true)
            .unwrap_err();
        assert!(
            error.to_string().contains("weights.device_cache"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn explicit_policy_mismatch_and_unproven_resume_are_rejected() {
        let defaults = default_execution_policy(&Device::Cpu);
        let mut recorded = defaults.clone();
        recorded.configured_output_rows = 17;
        let mismatch =
            select_resume_execution_policy(defaults.clone(), Some(&recorded), true, false, true)
                .unwrap_err()
                .to_string();
        assert!(mismatch.contains("disagrees"));
        assert!(mismatch.contains("configured_output_rows"));

        let mut online_chunks = TransformerChunking::default();
        online_chunks.attention.key = AttentionKeyChunkPolicy::chunked(16).unwrap();
        let online = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            online_chunks,
            false,
            true,
        )
        .unwrap();
        let numerical_mismatch =
            select_resume_execution_policy(defaults.clone(), Some(&online), true, false, true)
                .unwrap_err()
                .to_string();
        assert!(numerical_mismatch.contains("numerics.attention"));

        select_resume_execution_policy(defaults.clone(), None, false, false, false).unwrap();
        assert!(
            select_resume_execution_policy(defaults.clone(), None, false, false, true)
                .unwrap_err()
                .to_string()
                .contains("no execution policy")
        );

        let mut nondefault_cli = defaults.clone();
        nondefault_cli.configured_output_rows = 1;
        assert!(
            select_resume_execution_policy(nondefault_cli, Some(&recorded), false, true, true,)
                .unwrap_err()
                .to_string()
                .contains("disagrees")
        );

        let args = Args::try_parse_from([
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "inputs.safetensors",
            "--output",
            "output.safetensors",
            "--ffn-token-chunk-size",
            "256",
        ])
        .unwrap();
        let Command::H3(H3Command::DenoiseT2va { chunks, .. }) = args.command else {
            panic!("expected denoise command")
        };
        assert!(
            chunks.is_explicit(),
            "an explicitly supplied default must not be mistaken for an implicit default"
        );
        assert!(
            select_resume_execution_policy(
                defaults,
                Some(&recorded),
                false,
                chunks.is_explicit(),
                true,
            )
            .unwrap_err()
            .to_string()
            .contains("disagrees")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resume_mismatch_names_the_changed_cuda_capabilities() {
        let mut recorded = default_execution_policy(&Device::Cpu);
        recorded.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        *recorded.numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
            flyingfish::h3::policy::AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        recorded.validate().unwrap();
        // Every CUDA artifact section is a pure function of the capabilities and
        // the attention backend, and both are compared before the artifacts are,
        // so no pair of policies that `validate` accepts can differ first inside
        // `cuda_artifacts`. What a resume actually reports for a device that
        // afforded different capabilities is the capability set itself.
        let mut requested = recorded.clone();
        *requested.numerics = flyingfish::h3::policy::H3NumericalContract::for_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
            Some(flyingfish::h3::policy::CudaCapabilities {
                tuned_kernels: false,
                reference_libraries: true,
            }),
            flyingfish::h3::policy::AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        requested.validate().unwrap();

        let error = select_resume_execution_policy(requested, Some(&recorded), true, false, true)
            .unwrap_err();
        assert!(error.to_string().contains("numerics.cuda_capabilities"));
    }

    #[test]
    fn show_policy_history_extracts_canonical_json_without_overwriting() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("checkpoint.safetensors");
        let output = directory.path().join("policy.json");
        let policy = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            TransformerChunking::default(),
            false,
            true,
        )
        .unwrap();
        let mut history = PolicyHistory::new();
        history
            .append_successful_evaluations(1, policy.clone())
            .unwrap();
        let mut tensors = HashMap::from([
            (
                "video_latents",
                Tensor::zeros((1, 1, 1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "audio_latents",
                Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "prompt_embeddings",
                Tensor::zeros((1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "text_token_tags",
                Tensor::zeros(1, candle_core::DType::U32, &Device::Cpu).unwrap(),
            ),
            ("completed_steps", Tensor::new(1u32, &Device::Cpu).unwrap()),
            ("sigma_points", Tensor::new(2u32, &Device::Cpu).unwrap()),
            ("video_shift", Tensor::new(12f32, &Device::Cpu).unwrap()),
            ("audio_shift", Tensor::new(3f32, &Device::Cpu).unwrap()),
        ]);
        history
            .insert_checkpoint_tensors(&mut tensors, &Device::Cpu)
            .unwrap();
        safetensors::save(&tensors, &checkpoint).unwrap();

        denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
            checkpoint: checkpoint.clone(),
            output: Some(output.clone()),
        })
        .unwrap();
        assert_eq!(
            std::fs::read(&output).unwrap(),
            history.canonical_json().unwrap()
        );

        // A history recorded on another backend: still auditable, still
        // refused when a device that cannot run it is named.
        let legacy_checkpoint = directory
            .path()
            .join("other-backend-checkpoint.safetensors");
        let legacy_output = directory.path().join("other-backend-policy.json");
        let mut legacy_policy = policy;
        legacy_policy.execution_backend = flyingfish::h3::policy::ExecutionBackendPolicy::Cuda;
        *legacy_policy.numerics = flyingfish::h3::policy::H3NumericalContract::for_verified_target(
            flyingfish::h3::policy::ExecutionBackendPolicy::Cuda,
            legacy_policy.attention.backend,
        )
        .unwrap();
        legacy_policy.validate().unwrap();
        let mut legacy_history = PolicyHistory::new();
        legacy_history
            .append_successful_evaluations(1, legacy_policy.clone())
            .unwrap();
        for name in ["ff_policy_history_json", "ff_policy_history_schema"] {
            tensors.remove(name).unwrap();
        }
        legacy_history
            .insert_checkpoint_tensors(&mut tensors, &Device::Cpu)
            .unwrap();
        safetensors::save(&tensors, &legacy_checkpoint).unwrap();
        denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
            checkpoint: legacy_checkpoint,
            output: Some(legacy_output.clone()),
        })
        .unwrap();
        assert_eq!(
            std::fs::read(legacy_output).unwrap(),
            legacy_history.canonical_json().unwrap()
        );

        let selected = select_resume_execution_policy(
            default_execution_policy(&Device::Cpu),
            Some(&legacy_policy),
            false,
            false,
            true,
        )
        .unwrap();
        let error = validate_executable_policy(&selected, &Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("execution policy requires"));
        assert!(
            denoise::run_show_policy_history(H3Command::ShowPolicyHistory {
                checkpoint,
                output: Some(output),
            })
            .unwrap_err()
            .to_string()
            .contains("already exists")
        );
    }

    #[test]
    fn parses_unified_generation_command() {
        let args = Args::try_parse_from([
            "ff",
            "h3",
            "generate",
            "--model",
            "model",
            "--prompt",
            "a kite",
            "--output-dir",
            "output",
            "--target-hidden-state",
            "37",
            "--attention-projection-chunk-size",
            "64",
            "--ffn-token-chunk-size",
            "64",
            "--output-token-chunk-size",
            "128",
            "--wav-format",
            "float32",
        ])
        .unwrap();
        match args.command {
            Command::H3(H3Command::GenerateT2va {
                prompt,
                target_hidden_state,
                chunks,
                wav_format,
                ..
            }) => {
                assert_eq!(prompt, "a kite");
                assert_eq!(target_hidden_state, 37);
                assert_eq!(chunks.attention_projection_chunk_size.unwrap().get(), 64);
                assert_eq!(chunks.ffn_token_chunk_size.unwrap().get(), 64);
                assert_eq!(chunks.output_token_chunk_size.unwrap().get(), 128);
                assert_eq!(wav_format, WavSampleFormat::Float32);
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn parses_glm_generation_and_residency_policy() {
        let args = Args::try_parse_from([
            "ff",
            "glm",
            "generate",
            "--model",
            "GLM-5.3-Flash",
            "--prompt",
            "17*23=",
            "--max-new-tokens",
            "32",
            "--max-context-tokens",
            "64",
            "--reasoning-effort",
            "low",
            "--temperature",
            "0",
            "--resident-static",
            "--expert-cache-mib",
            "4096",
            "--expert-cache-layout",
            "shared-pool",
            "--expert-cache-replacement",
            "lfu",
            "--expert-cache-readmit",
            "--expert-cache-min-mib",
            "1024",
            "--execution-manifest",
            "output/glm-execution.json",
            "--json",
            "--telemetry-json",
            "output/glm-telemetry.json",
            "--routing-trace",
            "output/glm-routing.json",
            "--routing-trace-domain",
            "arithmetic",
        ])
        .unwrap();
        match args.command {
            Command::Glm(GlmCommand::Generate {
                prompt,
                max_new_tokens,
                max_context_tokens,
                reasoning_effort,
                temperature,
                top_p,
                resident_static,
                expert_cache_mib,
                expert_cache_layout,
                expert_cache_replacement,
                expert_cache_readmit,
                expert_cache_min_mib,
                json,
                telemetry_json,
                routing_trace,
                routing_trace_domain,
                execution_manifest,
                ..
            }) => {
                assert_eq!(prompt, "17*23=");
                assert_eq!(max_new_tokens.get(), 32);
                assert_eq!(max_context_tokens.get(), 64);
                assert_eq!(reasoning_effort, "low");
                assert_eq!(temperature, 0.0);
                assert_eq!(top_p, 0.95);
                assert_eq!(resident_static, Some(true));
                assert_eq!(expert_cache_mib, Some(4096));
                assert_eq!(expert_cache_layout, Some(ExpertCacheLayout::SharedPool));
                assert_eq!(
                    expert_cache_replacement,
                    Some(ExpertCacheReplacementPolicy::Lfu)
                );
                assert!(expert_cache_readmit);
                assert_eq!(expert_cache_min_mib, Some(1024));
                assert!(json);
                assert_eq!(
                    telemetry_json,
                    Some(PathBuf::from("output/glm-telemetry.json"))
                );
                assert_eq!(
                    routing_trace,
                    Some(PathBuf::from("output/glm-routing.json"))
                );
                assert_eq!(routing_trace_domain.as_deref(), Some("arithmetic"));
                assert_eq!(
                    execution_manifest,
                    Some(PathBuf::from("output/glm-execution.json"))
                );
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn glm_residency_flags_distinguish_absent_from_explicit_zero_and_false() {
        let base = [
            "ff",
            "glm",
            "generate",
            "--model",
            "GLM-5.3-Flash",
            "--prompt",
            "hello",
        ];
        let defaults = Args::try_parse_from(base).unwrap();
        let Command::Glm(GlmCommand::Generate {
            weights,
            resource_policy,
            resident_static,
            no_resident_static,
            expert_cache_mib,
            expert_cache_layout,
            expert_cache_replacement,
            ..
        }) = defaults.command
        else {
            panic!("wrong command")
        };
        assert!(!weights.is_explicit());
        assert_eq!(
            resource_policy,
            flyingfish::runtime::resource_selection::ResourcePolicyMode::Performance
        );
        assert_eq!(resident_static, None);
        assert!(!no_resident_static);
        assert_eq!(expert_cache_mib, None);
        assert_eq!(expert_cache_layout, None);
        assert_eq!(expert_cache_replacement, None);

        let args = Args::try_parse_from(base.into_iter().chain([
            "--no-resident-static",
            "--expert-cache-mib",
            "0",
            "--weight-source",
            "mmap",
            "--expert-cache-layout",
            "shared-pool",
        ]))
        .unwrap();
        let Command::Glm(GlmCommand::Generate {
            weights,
            resident_static,
            no_resident_static,
            expert_cache_mib,
            expert_cache_layout,
            ..
        }) = args.command
        else {
            panic!("wrong command")
        };
        assert_eq!(resident_static, None);
        assert!(no_resident_static);
        assert_eq!(expert_cache_mib, Some(0));
        assert_eq!(weights.weight_source, Some(WeightSource::Mmap));
        assert_eq!(expert_cache_layout, Some(ExpertCacheLayout::SharedPool));
        assert!(
            Args::try_parse_from(
                base.into_iter()
                    .chain(["--resident-static", "--no-resident-static"])
            )
            .is_err()
        );
        let explicit_false =
            Args::try_parse_from(base.into_iter().chain(["--resident-static=false"])).unwrap();
        let Command::Glm(GlmCommand::Generate {
            resident_static, ..
        }) = explicit_false.command
        else {
            panic!("wrong command")
        };
        assert_eq!(resident_static, Some(false));
    }

    #[test]
    fn parses_glm_routing_replay_lists() {
        let args = Args::try_parse_from([
            "ff",
            "glm",
            "replay-routing",
            "--trace",
            "routing.json",
            "--output",
            "replay.json",
            "--segment-lengths",
            "2,8",
            "--cache-mib",
            "64,128",
        ])
        .unwrap();
        match args.command {
            Command::Glm(GlmCommand::ReplayRouting {
                trace,
                output,
                segment_lengths,
                cache_mib,
            }) => {
                assert_eq!(trace, PathBuf::from("routing.json"));
                assert_eq!(output, PathBuf::from("replay.json"));
                assert_eq!(
                    segment_lengths
                        .into_iter()
                        .map(NonZeroUsize::get)
                        .collect::<Vec<_>>(),
                    [2, 8]
                );
                assert_eq!(
                    cache_mib
                        .into_iter()
                        .map(NonZeroU64::get)
                        .collect::<Vec<_>>(),
                    [64, 128]
                );
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn parses_bounded_glm_parity_capture() {
        let args = Args::try_parse_from([
            "ff",
            "glm",
            "capture-parity",
            "--model",
            "GLM-5.3-Flash",
            "--prompt",
            "17*23=",
            "--max-context-tokens",
            "128",
            "--reasoning-effort",
            "max",
            "--resident-static",
            "--output",
            "capture.safetensors",
        ])
        .unwrap();
        match args.command {
            Command::Glm(GlmCommand::CaptureParity {
                prompt,
                max_context_tokens,
                reasoning_effort,
                resident_static,
                output,
                ..
            }) => {
                assert_eq!(prompt, "17*23=");
                assert_eq!(max_context_tokens.get(), 128);
                assert_eq!(reasoning_effort, "max");
                assert!(resident_static);
                assert_eq!(output, PathBuf::from("capture.safetensors"));
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn adaptive_glm_cache_requires_its_audit_manifest_and_gate() {
        let base = [
            "ff", "glm", "generate", "--model", "model", "--prompt", "prompt",
        ];
        let mut readmit_without_manifest = base.to_vec();
        readmit_without_manifest.push("--expert-cache-readmit");
        assert!(Args::try_parse_from(readmit_without_manifest).is_err());

        let mut floor_without_readmit = base.to_vec();
        floor_without_readmit.extend(["--expert-cache-min-mib", "1"]);
        assert!(Args::try_parse_from(floor_without_readmit).is_err());
    }

    #[test]
    fn parses_resumable_denoise_and_online_attention() {
        let args = Args::try_parse_from([
            "ff",
            "h3",
            "denoise",
            "--model",
            "model",
            "--inputs",
            "checkpoint.safetensors",
            "--output",
            "next.safetensors",
            "--checkpoint-dir",
            "checkpoints",
            "--max-steps",
            "2",
            "--attention-key-chunk-size",
            "1024",
        ])
        .unwrap();
        match args.command {
            Command::H3(H3Command::DenoiseT2va {
                max_steps,
                chunks,
                checkpoint_dir,
                ..
            }) => {
                assert_eq!(max_steps.map(NonZeroUsize::get), Some(2));
                assert_eq!(checkpoint_dir, Some(PathBuf::from("checkpoints")));
                assert_eq!(
                    chunks
                        .attention_key_chunk_size
                        .map(AttentionKeyChunkSize::get),
                    Some(1024)
                );
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn builds_transformer_options_from_run_settings() {
        let mut policy = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(2),
            (TransformerChunkArgs {
                attention_projection_chunk_size: NonZeroUsize::new(11).unwrap(),
                attention_query_chunk_size: NonZeroUsize::new(12).unwrap(),
                attention_key_chunk_size: None,
                ffn_token_chunk_size: NonZeroUsize::new(13).unwrap(),
                output_token_chunk_size: NonZeroUsize::new(14).unwrap(),
            })
            .model_chunking(),
            false,
            true,
        )
        .unwrap();
        policy.weights.device_cache = DeviceCachePolicy::with_max_bytes(4096);
        let options = build_transformer_options(Device::Cpu, &policy).unwrap();
        assert_eq!(options.device_cache_policy, policy.weights.device_cache);
        assert_eq!(options.weight_source, WeightSource::Mmap);
        assert_eq!(options.cache_policy, CachePolicy::new(2));
        assert_eq!(options.chunking.attention.projection_chunk_size.get(), 11);
        assert_eq!(options.chunking.attention.query_chunk_size.get(), 12);
        assert_eq!(
            options.chunking.attention.key,
            AttentionKeyChunkPolicy::Full
        );
        assert_eq!(options.chunking.feed_forward_chunk_size.get(), 13);
        assert_eq!(options.chunking.output_chunk_size.get(), 14);
        assert!(!options.flash_attention);
        policy.weights.granularity = CacheGranularity::Tensor;
        policy.weights.cache_bytes = Some(4096);
        policy.weights.host_phase_priority = true;
        assert!(
            build_transformer_options(Device::Cpu, &policy)
                .unwrap()
                .host_phase_priority
        );
    }

    #[test]
    fn parses_tensor_parity_command() {
        let args = Args::try_parse_from([
            "ff",
            "diff",
            "--reference",
            "gold.safetensors",
            "--actual",
            "actual.safetensors",
            "--atol",
            "0.001",
        ])
        .unwrap();
        match args.command {
            Command::Diff { atol, rtol, .. } => {
                assert_eq!(atol, 0.001);
                assert_eq!(rtol, 0.0001);
            }
            _ => panic!("parsed the wrong command"),
        }
    }

    #[test]
    fn dispatches_to_a_handler_without_loading_weights() {
        let root = tempfile::tempdir().unwrap();
        let missing_model = root.path().join("missing-model");
        let args = Args::try_parse_from([
            "ff",
            "inspect",
            "--checkpoint",
            missing_model.to_str().unwrap(),
        ])
        .unwrap();

        let error = dispatch(args.command).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("checkpoint directory does not exist"),
            "unexpected handler error: {error:#}"
        );
    }

    #[test]
    fn extracted_handler_can_be_tested_directly() {
        let root = tempfile::tempdir().unwrap();
        let missing_model = root.path().join("missing-model");
        let command = Args::try_parse_from([
            "ff",
            "tensor",
            "--checkpoint",
            missing_model.to_str().unwrap(),
            "--name",
            "probe.weight",
        ])
        .unwrap()
        .command;

        let error = inspect::run_tensor(command).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("checkpoint directory does not exist")
        );
    }

    #[test]
    fn refuses_outputs_inside_model_directory() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        std::fs::create_dir(&model).unwrap();
        assert!(resolve_output_outside_model(&model.join("result.bin"), &model).is_err());
        assert!(resolve_output_outside_model(&root.path().join("result.bin"), &model).is_ok());
        #[cfg(unix)]
        {
            let target = model.join("existing.bin");
            std::fs::write(&target, b"model").unwrap();
            let link = root.path().join("linked-output.bin");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(resolve_output_outside_model(&link, &model).is_err());
        }
    }

    #[test]
    fn new_output_validation_rejects_existing_targets_and_missing_parents() {
        let directory = tempfile::tempdir().unwrap();
        let available = directory.path().join("available.bin");
        ensure_new_output(&available, "test output").unwrap();

        std::fs::write(&available, b"existing").unwrap();
        assert!(ensure_new_output(&available, "test output").is_err());
        assert!(
            ensure_new_output(&directory.path().join("missing/target.bin"), "test output").is_err()
        );
    }

    #[test]
    fn telemetry_reports_are_atomically_published_without_clobbering() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model");
        std::fs::create_dir(&model).unwrap();
        let path = directory.path().join("telemetry.json");
        let monitor = TelemetryMonitor::start(None, Duration::from_millis(1)).unwrap();
        write_telemetry(&path, monitor).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        flyingfish::runtime::telemetry::RuntimeTelemetryReport::from_json(&bytes).unwrap();

        let monitor = TelemetryMonitor::start(None, Duration::from_millis(1)).unwrap();
        assert!(write_telemetry(&path, monitor).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(std::fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".ff-stage-dir-")
        }));
    }

    #[cfg(unix)]
    #[test]
    fn component_symlink_may_not_escape_the_model_root() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        let outside = root.path().join("outside-transformer");
        std::fs::create_dir(&model).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, model.join("transformer")).unwrap();

        let error = resolve_component(&model, Path::new("transformer")).unwrap_err();
        assert!(error.to_string().contains("escapes the model root"));
    }

    fn official_target(
        short_edge: Option<usize>,
        aspect_ratio: Option<&str>,
        duration_seconds: Option<usize>,
    ) -> TargetGeometryArgs {
        TargetGeometryArgs {
            short_edge,
            aspect_ratio: aspect_ratio.map(str::to_owned),
            duration_seconds,
        }
    }

    #[test]
    fn the_official_target_replaces_the_explicit_latent_geometry() {
        let resolved = official_target(Some(768), None, Some(10))
            .resolve_latent_geometry((1, 2, 3, 4))
            .unwrap();
        assert_eq!(resolved, (72, 48, 84, 405));
    }

    #[test]
    fn the_explicit_latent_geometry_survives_an_absent_target() {
        let resolved = official_target(None, None, None)
            .resolve_latent_geometry((1, 2, 3, 4))
            .unwrap();
        assert_eq!(resolved, (1, 2, 3, 4));
    }

    #[test]
    fn an_auto_ratio_keeps_the_conditioning_shape_at_the_requested_short_edge() {
        let (canvas, num_frames) = official_target(Some(768), Some(AUTO_ASPECT_RATIO), Some(10))
            .resolve_canvas_geometry((None, None), 240, Some((1920, 1080)))
            .unwrap();
        assert_eq!(canvas, Some((768, 1344)));
        assert_eq!(num_frames, 240);
    }

    #[test]
    fn an_auto_ratio_without_a_conditioning_shape_says_so() {
        let error = official_target(Some(768), Some(AUTO_ASPECT_RATIO), Some(10))
            .resolve_canvas_geometry((None, None), 240, None)
            .unwrap_err();
        assert!(error.to_string().contains("conditioning input"), "{error}");
    }

    #[test]
    fn a_half_stated_explicit_canvas_is_still_refused() {
        let error = official_target(None, None, None)
            .resolve_canvas_geometry((Some(768), None), 240, None)
            .unwrap_err();
        assert!(
            error.to_string().contains("must be passed together"),
            "{error}"
        );
    }

    #[test]
    fn the_official_and_explicit_generate_targets_are_mutually_exclusive() {
        let base = [
            "ff",
            "h3",
            "generate",
            "--model",
            "model",
            "--prompt",
            "a",
            "--output-dir",
            "out",
        ];
        assert!(
            Args::try_parse_from(base.iter().copied().chain([
                "--short-edge",
                "768",
                "--duration-seconds",
                "10",
                "--latent-frames",
                "72",
            ]))
            .is_err()
        );
        assert!(Args::try_parse_from(base.iter().copied().chain(["--short-edge", "768"])).is_err());
        assert!(
            Args::try_parse_from(base.iter().copied().chain(["--duration-seconds", "10"])).is_err()
        );
        assert!(
            Args::try_parse_from(base.iter().copied().chain([
                "--short-edge",
                "768",
                "--duration-seconds",
                "10"
            ]))
            .is_ok()
        );
    }

    #[test]
    fn prepare_commands_accept_either_geometry_form_but_require_one() {
        let prepare = [
            "ff",
            "h3",
            "prepare",
            "--model",
            "model",
            "--prompt-encoding",
            "prompt.safetensors",
            "--output",
            "out.safetensors",
        ];
        assert!(Args::try_parse_from(prepare).is_err());
        assert!(
            Args::try_parse_from(prepare.iter().copied().chain([
                "--short-edge",
                "768",
                "--duration-seconds",
                "10"
            ]))
            .is_ok()
        );

        let ref2va = [
            "ff",
            "h3",
            "prepare-ref2va",
            "--model",
            "model",
            "--prompt",
            "a",
            "--references-json",
            "refs.json",
            "--output",
            "out",
        ];
        assert!(Args::try_parse_from(ref2va).is_err());
        assert!(
            Args::try_parse_from(ref2va.iter().copied().chain([
                "--short-edge",
                "768",
                "--duration-seconds",
                "10"
            ]))
            .is_ok()
        );
    }
}
