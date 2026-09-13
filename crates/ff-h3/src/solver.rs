use crate::{
    core::{AttentionKeyChunkPolicy, CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS},
    policy::{
        AttentionBackendPolicy, AttentionExecutionPolicy, ExecutionBackendPolicy, ExecutionPolicy,
    },
    resources::{
        H3ResourceBudgetExt, ResourceAssumptions, ResourceBudget, ResourceEstimate, T2vaGeometry,
        TransformerShape,
    },
};
use anyhow::{Context, Result};
use ff_core::weights::accounting::CacheInventory;
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionBackendAllowlist {
    pub full_softmax: bool,
    pub online_softmax: bool,
    pub flash_attention: bool,
}

impl AttentionBackendAllowlist {
    pub const fn parity_safe() -> Self {
        Self {
            full_softmax: true,
            online_softmax: false,
            flash_attention: false,
        }
    }

    fn enabled_count(self) -> usize {
        [self.full_softmax, self.online_softmax, self.flash_attention]
            .into_iter()
            .filter(|enabled| *enabled)
            .count()
    }
}

impl Default for AttentionBackendAllowlist {
    fn default() -> Self {
        Self::parity_safe()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicySearchSpace {
    pub attention_backends: AttentionBackendAllowlist,
    pub attention_projection_rows: Vec<NonZeroUsize>,
    pub attention_query_rows: Vec<NonZeroUsize>,
    pub attention_key_rows: Vec<NonZeroUsize>,
    pub feed_forward_rows: Vec<NonZeroUsize>,
    pub output_rows: Vec<NonZeroUsize>,
    pub precompute_adaln: Vec<bool>,
}

impl PolicySearchSpace {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.attention_backends.enabled_count() > 0,
            "attention backend allowlist must enable at least one backend per solve"
        );
        for (name, values) in [
            (
                "attention projection rows",
                self.attention_projection_rows.as_slice(),
            ),
            ("attention query rows", self.attention_query_rows.as_slice()),
            ("feed-forward rows", self.feed_forward_rows.as_slice()),
            ("output rows", self.output_rows.as_slice()),
        ] {
            anyhow::ensure!(!values.is_empty(), "{name} search space must not be empty");
        }
        anyhow::ensure!(
            !self.precompute_adaln.is_empty(),
            "AdaLN precompute search space must not be empty"
        );
        if self.attention_backends.online_softmax {
            anyhow::ensure!(
                !self.attention_key_rows.is_empty(),
                "online softmax requires at least one key-row candidate"
            );
        }
        Ok(())
    }
}

impl Default for PolicySearchSpace {
    fn default() -> Self {
        fn rows(values: &[usize]) -> Vec<NonZeroUsize> {
            values
                .iter()
                .copied()
                .map(|value| NonZeroUsize::new(value).expect("default chunk rows are non-zero"))
                .collect()
        }

        Self {
            attention_backends: AttentionBackendAllowlist::default(),
            attention_projection_rows: rows(&[16, 32, 64, 128, 256, 512, 1024]),
            attention_query_rows: rows(&[1, 2, 4, 8, 16, 32, 64, 128, 256]),
            attention_key_rows: rows(&[128, 256, 512, 1024, 2048, 4096]),
            feed_forward_rows: rows(&[32, 64, 128, 256, 512, 1024]),
            output_rows: rows(&[32, 64, 128, 256, 512, 1024]),
            precompute_adaln: vec![true, false],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeasiblePolicyCandidate {
    pub policy: ExecutionPolicy,
    /// Position in this enumeration: `candidate-0` is the first the ordering
    /// below admits. It names a row in one result, not a policy.
    pub candidate_id: String,
    pub estimate: ResourceEstimate,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CanonicalAttentionChoice {
    backend_rank: u8,
    key_rows: Option<usize>,
}

impl CanonicalAttentionChoice {
    const FULL: Self = Self {
        backend_rank: 0,
        key_rows: None,
    };
    const FLASH: Self = Self {
        backend_rank: 2,
        key_rows: None,
    };

    const fn online(key_rows: usize) -> Self {
        Self {
            backend_rank: 1,
            key_rows: Some(key_rows),
        }
    }

    const fn policy(self) -> (AttentionBackendPolicy, Option<u64>) {
        match self.backend_rank {
            0 => (AttentionBackendPolicy::FullSoftmax, None),
            1 => (
                AttentionBackendPolicy::OnlineSoftmax,
                Some(self.key_rows.expect("online choice carries key rows") as u64),
            ),
            2 => (AttentionBackendPolicy::FlashAttention, None),
            _ => unreachable!(),
        }
    }
}

/// Metadata and a separately declared legacy planning allowance. The latter
/// is not a cache bound and does not change the execution policy.
pub struct SolverHostWeights<'a> {
    pub inventory: &'a CacheInventory,
    pub additional_host_allowance_bytes: u64,
}

pub fn solve_feasible_policies(
    base_policy: &ExecutionPolicy,
    model: TransformerShape,
    geometry: T2vaGeometry,
    assumptions: ResourceAssumptions,
    budget: ResourceBudget,
    search_space: &PolicySearchSpace,
    host_weights: SolverHostWeights<'_>,
) -> Result<Vec<FeasiblePolicyCandidate>> {
    base_policy.validate()?;
    model.validate()?;
    assumptions.validate()?;
    search_space.validate()?;
    match base_policy.execution_backend {
        ExecutionBackendPolicy::Cpu => anyhow::ensure!(
            assumptions.device_memory_is_host,
            "CPU policy solving requires assumptions.device_memory_is_host=true"
        ),
        ExecutionBackendPolicy::Cuda | ExecutionBackendPolicy::Metal => {
            anyhow::ensure!(
                !assumptions.device_memory_is_host,
                "non-CPU policy solving requires assumptions.device_memory_is_host=false"
            );
            anyhow::ensure!(
                assumptions.backend_workspace_bytes > 0,
                "non-CPU policy solving requires a non-zero backend workspace allowance"
            );
        }
    }
    anyhow::ensure!(
        assumptions.device_weight_cache_bytes == base_policy.weights.device_cache.max_bytes,
        "device weight assumptions disagree with cache policy"
    );
    let host =
        crate::resources::host_weight_residency_charges(base_policy, host_weights.inventory)?;
    let expected_owned = host
        .owned_weight_bytes
        .checked_add(host_weights.additional_host_allowance_bytes)
        .context("host cache plus planning allowance overflow")?;
    anyhow::ensure!(
        assumptions.host_weight_cache_bytes == expected_owned
            && assumptions.mapped_weight_residency_bytes == host.mapped_weight_bytes,
        "host weight assumptions disagree with the actual cache policy: expected owned {} and mapped {} bytes",
        expected_owned,
        host.mapped_weight_bytes
    );
    anyhow::ensure!(
        !search_space.attention_backends.flash_attention || base_policy.execution_backend.is_cuda(),
        "FlashAttention is allowlisted but the base policy is not CUDA"
    );
    if search_space.attention_backends.flash_attention {
        anyhow::ensure!(
            model.attention_head_dim.is_multiple_of(8) && model.attention_head_dim <= 512,
            "FlashAttention requires attention_head_dim to be divisible by 8 and at most 512, got {}",
            model.attention_head_dim
        );
        anyhow::ensure!(
            cfg!(feature = "flash-attn"),
            "FlashAttention is allowlisted, but this build was not compiled with --features flash-attn"
        );
    }

    let mut dimensions_only = geometry;
    dimensions_only.attention_projection_chunk_size = 1;
    dimensions_only.attention_query_chunk_size = 1;
    dimensions_only.attention_key_chunk_policy = AttentionKeyChunkPolicy::Full;
    dimensions_only.ffn_token_chunk_size = 1;
    dimensions_only.output_token_chunk_size = 1;
    let sequence_rows = dimensions_only.sequence_rows(model.patch_size)?;
    let total_rows =
        usize::try_from(sequence_rows.total).context("packed sequence row count exceeds usize")?;
    let output_modality_rows = usize::try_from(sequence_rows.video.max(sequence_rows.audio))
        .context("output modality row count exceeds usize")?;
    anyhow::ensure!(
        total_rows > 0,
        "packed sequence must contain at least one row"
    );
    anyhow::ensure!(
        output_modality_rows > 0,
        "output modalities must contain at least one row"
    );

    let projection_query_rows = canonical_projection_query_rows(
        &search_space.attention_projection_rows,
        &search_space.attention_query_rows,
        total_rows,
    );
    let feed_forward_rows = canonical_rows(&search_space.feed_forward_rows, total_rows);
    let output_rows = canonical_rows(&search_space.output_rows, output_modality_rows);
    let precompute_adaln = search_space
        .precompute_adaln
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut attention_choices = canonical_attention_choices(
        search_space.attention_backends,
        &search_space.attention_key_rows,
        total_rows,
    );
    if base_policy.execution_backend == ExecutionBackendPolicy::Cuda {
        let full_is_out_of_range = total_rows > CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
        let refiner_is_out_of_range = geometry.text_rows > CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS;
        attention_choices.retain(|choice| {
            !((full_is_out_of_range && *choice == CanonicalAttentionChoice::FULL)
                || (refiner_is_out_of_range && choice.backend_rank == 1))
        });
        anyhow::ensure!(
            !attention_choices.is_empty(),
            "CUDA BF16 full-softmax supports at most {CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS} \
             packed rows and Online attention's persistent token refiner supports at most \
             {CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS} text rows; request has {total_rows} packed \
             rows and {} text rows. Select FlashAttention instead",
            geometry.text_rows
        );
    }

    let mut canonical_policies = BTreeMap::<Vec<u8>, (ExecutionPolicy, T2vaGeometry)>::new();
    for &(projection_rows, query_rows) in &projection_query_rows {
        for &ffn_rows in &feed_forward_rows {
            for &output_rows in &output_rows {
                for &precompute_adaln in &precompute_adaln {
                    for &attention_choice in &attention_choices {
                        let (backend, configured_key_rows) = attention_choice.policy();
                        let mut policy = base_policy.clone();
                        policy.attention = AttentionExecutionPolicy {
                            backend,
                            configured_projection_rows: projection_rows as u64,
                            configured_query_rows: query_rows as u64,
                            configured_key_rows,
                        };
                        policy.rebind_attention_numerics()?;
                        policy.configured_ffn_rows = ffn_rows as u64;
                        policy.configured_output_rows = output_rows as u64;
                        policy.precompute_adaln = precompute_adaln;
                        let canonical_json = policy.canonical_json()?;

                        let mut candidate_geometry = geometry;
                        candidate_geometry.attention_projection_chunk_size = projection_rows;
                        candidate_geometry.attention_query_chunk_size = query_rows;
                        candidate_geometry.attention_key_chunk_policy = match configured_key_rows {
                            Some(rows) => AttentionKeyChunkPolicy::chunked(
                                usize::try_from(rows).context("canonical key rows exceed usize")?,
                            )
                            .context("canonical key rows are invalid")?,
                            None => AttentionKeyChunkPolicy::Full,
                        };
                        candidate_geometry.ffn_token_chunk_size = ffn_rows;
                        candidate_geometry.output_token_chunk_size = output_rows;
                        canonical_policies
                            .entry(canonical_json)
                            .or_insert((policy, candidate_geometry));
                    }
                }
            }
        }
    }

    let mut feasible = Vec::new();
    for (_, (policy, candidate_geometry)) in canonical_policies {
        let mut candidate_assumptions = assumptions;
        candidate_assumptions.use_flash_attention =
            policy.attention.backend == AttentionBackendPolicy::FlashAttention;
        candidate_assumptions.precompute_adaln_steps = if policy.precompute_adaln {
            candidate_assumptions.evaluation_count
        } else {
            0
        };
        let estimate =
            ResourceEstimate::for_shape(model, candidate_geometry, candidate_assumptions)?;
        if budget.check(&estimate).within_budget {
            feasible.push(FeasiblePolicyCandidate {
                candidate_id: String::new(),
                policy,
                estimate,
            });
        }
    }
    feasible.sort_by(|left, right| {
        left.estimate
            .peak_device_bytes
            .cmp(&right.estimate.peak_device_bytes)
            .then_with(|| {
                left.estimate
                    .peak_host_bytes
                    .cmp(&right.estimate.peak_host_bytes)
            })
            // Two policies with the same modelled peaks still need a total
            // order, and their own canonical form is what provides one.
            .then_with(|| canonical_or_empty(&left.policy).cmp(&canonical_or_empty(&right.policy)))
    });
    for (index, candidate) in feasible.iter_mut().enumerate() {
        candidate.candidate_id = format!("candidate-{index}");
    }
    Ok(feasible)
}

/// A policy's canonical bytes, for ordering only. Every policy here has
/// already been validated, so the fallback never decides an ordering.
fn canonical_or_empty(policy: &ExecutionPolicy) -> Vec<u8> {
    policy.canonical_json().unwrap_or_default()
}

fn canonical_rows(values: &[NonZeroUsize], maximum: usize) -> Vec<usize> {
    values
        .iter()
        .map(|rows| rows.get().min(maximum))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn canonical_projection_query_rows(
    projection_rows: &[NonZeroUsize],
    query_rows: &[NonZeroUsize],
    maximum: usize,
) -> Vec<(usize, usize)> {
    let projections = canonical_rows(projection_rows, maximum);
    let queries = canonical_rows(query_rows, maximum);
    projections
        .into_iter()
        .flat_map(|projection| {
            queries
                .iter()
                .copied()
                .map(move |query| (projection, query.min(projection)))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn canonical_attention_choices(
    allowlist: AttentionBackendAllowlist,
    key_rows: &[NonZeroUsize],
    sequence_rows: usize,
) -> Vec<CanonicalAttentionChoice> {
    let mut choices = BTreeSet::new();
    if allowlist.full_softmax {
        choices.insert(CanonicalAttentionChoice::FULL);
    }
    if allowlist.online_softmax {
        for rows in key_rows {
            if rows.get() < sequence_rows {
                choices.insert(CanonicalAttentionChoice::online(rows.get()));
            } else if allowlist.full_softmax {
                choices.insert(CanonicalAttentionChoice::FULL);
            }
        }
    }
    if allowlist.flash_attention {
        choices.insert(CanonicalAttentionChoice::FLASH);
    }
    choices.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{AttentionChunking, AttentionKeyChunkPolicy},
        model::TransformerChunking,
        policy::H3NumericalContract,
    };
    use candle_core::Device;
    use ff_core::weights::{CachePolicy, WeightSource};
    use std::collections::HashSet;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn base_policy() -> ExecutionPolicy {
        ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(2).with_max_bytes(64 * 1024 * 1024),
            TransformerChunking {
                attention: AttentionChunking {
                    projection_chunk_size: nz(32),
                    query_chunk_size: nz(32),
                    key: AttentionKeyChunkPolicy::Full,
                },
                feed_forward_chunk_size: nz(256),
                output_chunk_size: nz(256),
            },
            false,
            true,
        )
        .unwrap()
    }

    fn cuda_policy() -> ExecutionPolicy {
        let mut policy = base_policy();
        policy.execution_backend = ExecutionBackendPolicy::Cuda;
        policy.numerics = Box::new(
            H3NumericalContract::for_verified_target(
                ExecutionBackendPolicy::Cuda,
                AttentionBackendPolicy::FullSoftmax,
            )
            .unwrap(),
        );
        policy.validate().unwrap();
        policy
    }

    fn cpu_assumptions() -> ResourceAssumptions {
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.device_memory_is_host = true;
        assumptions
    }

    fn tiny_shape() -> TransformerShape {
        TransformerShape {
            num_layers: 2,
            num_attention_heads: 2,
            attention_head_dim: 3,
            hidden_size: 4,
            ffn_dim: 5,
            in_channels: 1,
            audio_in_channels: 1,
            patch_size: [1, 1, 1],
            text_dim: 1,
            freq_dim: 1,
            time_embed_hidden_dim: 1,
            time_embed_dim: 2,
            rope_freq_dim: 1,
        }
    }

    fn cache_inventory() -> CacheInventory {
        use ff_core::weights::accounting::CacheShardInventory;
        CacheInventory {
            shards: ["a", "b"]
                .into_iter()
                .map(|name| CacheShardInventory {
                    name: format!("{name}.safetensors"),
                    file_bytes: 108,
                    header_bytes: 8,
                    selected_tensor_bytes: 100,
                    selected_tensor_count: 1,
                    largest_tensor_bytes: 100,
                })
                .collect(),
        }
    }

    fn solve_fixture(
        policy: &ExecutionPolicy,
        model: TransformerShape,
        geometry: T2vaGeometry,
        mut assumptions: ResourceAssumptions,
        budget: ResourceBudget,
        search: &PolicySearchSpace,
    ) -> Result<Vec<FeasiblePolicyCandidate>> {
        let inventory = cache_inventory();
        let host = crate::resources::host_weight_residency_charges(policy, &inventory)?;
        assumptions.host_weight_cache_bytes = host.owned_weight_bytes;
        assumptions.mapped_weight_residency_bytes = host.mapped_weight_bytes;
        super::solve_feasible_policies(
            policy,
            model,
            geometry,
            assumptions,
            budget,
            search,
            SolverHostWeights {
                inventory: &inventory,
                additional_host_allowance_bytes: 0,
            },
        )
    }

    fn tiny_geometry() -> T2vaGeometry {
        T2vaGeometry {
            text_rows: 1,
            latent_frames: 1,
            latent_height: 1,
            latent_width: 1,
            audio_frames: 1,
            audio_channels: 1,
            attention_projection_chunk_size: 1,
            attention_query_chunk_size: 1,
            attention_key_chunk_policy: AttentionKeyChunkPolicy::Full,
            ffn_token_chunk_size: 1,
            output_token_chunk_size: 1,
        }
    }

    fn compact_space() -> PolicySearchSpace {
        PolicySearchSpace {
            attention_backends: AttentionBackendAllowlist::default(),
            attention_projection_rows: vec![nz(32), nz(64)],
            attention_query_rows: vec![nz(16), nz(32)],
            attention_key_rows: vec![nz(1024)],
            feed_forward_rows: vec![nz(128), nz(256)],
            output_rows: vec![nz(128), nz(256)],
            precompute_adaln: vec![true, false],
        }
    }

    #[test]
    fn standard_geometry_returns_only_budgeted_full_softmax_candidates() {
        let candidates = solve_fixture(
            &base_policy(),
            TransformerShape::h3_base(),
            T2vaGeometry::h3_default(1),
            cpu_assumptions(),
            ResourceBudget::default(),
            &compact_space(),
        )
        .unwrap();

        assert_eq!(candidates.len(), 32);
        for candidate in candidates {
            assert_eq!(
                candidate.policy.attention.backend,
                AttentionBackendPolicy::FullSoftmax
            );
            assert!(
                ResourceBudget::default()
                    .check(&candidate.estimate)
                    .within_budget
            );
            assert_eq!(candidate.estimate.sequence_rows.total, 37_711);
            assert_eq!(candidate.policy.weights, base_policy().weights);
        }
    }

    #[test]
    fn cuda_full_softmax_rejects_rows_above_the_verified_exact_range() {
        let policy = cuda_policy();
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.backend_workspace_bytes = 1;
        let error = solve_fixture(
            &policy,
            TransformerShape::h3_base(),
            T2vaGeometry::h3_default(1),
            assumptions,
            ResourceBudget::default(),
            &compact_space(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("at most 9216 packed rows"));
        assert!(error.to_string().contains("Select FlashAttention"));
    }

    #[cfg(feature = "flash-attn")]
    #[test]
    fn cuda_flash_solver_accepts_rows_above_persistent_softmax_range() {
        let policy = cuda_policy();
        let mut assumptions = ResourceAssumptions::h3_bf16_mmap();
        assumptions.backend_workspace_bytes = 1;
        let mut search = compact_space();
        search.attention_backends = AttentionBackendAllowlist {
            full_softmax: true,
            online_softmax: false,
            flash_attention: true,
        };
        let candidates = solve_fixture(
            &policy,
            TransformerShape::h3_base(),
            T2vaGeometry::h3_default(1),
            assumptions,
            ResourceBudget::default(),
            &search,
        )
        .unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| {
            candidate.policy.attention.backend == AttentionBackendPolicy::FlashAttention
        }));
    }

    #[test]
    fn tiny_geometry_canonicalizes_bounds_and_deduplicates_equivalent_policies() {
        let mut search = PolicySearchSpace {
            attention_backends: AttentionBackendAllowlist::parity_safe(),
            attention_projection_rows: vec![nz(2), nz(4), nz(8)],
            attention_query_rows: vec![nz(1), nz(4), nz(16)],
            attention_key_rows: vec![nz(1), nz(3), nz(8)],
            feed_forward_rows: vec![nz(2), nz(8)],
            output_rows: vec![nz(2), nz(8)],
            precompute_adaln: vec![true],
        };
        let candidates = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget::default(),
            &search,
        )
        .unwrap();

        assert_eq!(candidates.len(), 8);
        let mut ids = HashSet::new();
        let mut policies = HashSet::new();
        for candidate in &candidates {
            assert!(ids.insert(candidate.candidate_id.as_str()));
            assert!(policies.insert(candidate.policy.canonical_json().unwrap()));
            assert!(
                candidate.policy.attention.configured_query_rows
                    <= candidate.policy.attention.configured_projection_rows
            );
            assert!(candidate.policy.attention.configured_projection_rows <= 3);
            assert!(candidate.policy.configured_ffn_rows <= 3);
            assert_eq!(candidate.policy.configured_output_rows, 1);
            if let Some(key_rows) = candidate.policy.attention.configured_key_rows {
                assert!(key_rows < 3);
            }
        }

        search.attention_projection_rows = vec![nz(8), nz(2), nz(4), nz(8)];
        search.attention_query_rows = vec![nz(16), nz(4), nz(1), nz(1)];
        search.attention_key_rows = vec![nz(8), nz(1), nz(3), nz(1)];
        search.feed_forward_rows = vec![nz(8), nz(2), nz(2)];
        search.output_rows = vec![nz(8), nz(2), nz(8)];
        search.precompute_adaln = vec![true, true];
        let repeated = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget::default(),
            &search,
        )
        .unwrap();
        assert_eq!(repeated, candidates);
    }

    #[test]
    fn zero_budget_has_no_solution() {
        let candidates = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget {
                max_host_bytes: Some(0),
                max_device_bytes: Some(0),
            },
            &compact_space(),
        )
        .unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn experimental_backends_are_not_in_the_default_allowlist() {
        let mut search = compact_space();
        search.attention_projection_rows = vec![nz(1)];
        search.attention_query_rows = vec![nz(1)];
        search.feed_forward_rows = vec![nz(1)];
        search.output_rows = vec![nz(1)];
        search.precompute_adaln = vec![false];
        let candidates = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget::default(),
            &search,
        )
        .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].policy.attention.backend,
            AttentionBackendPolicy::FullSoftmax
        );
    }

    #[test]
    fn rejects_empty_attention_backend_allowlist() {
        let mut search = compact_space();
        search.attention_backends = AttentionBackendAllowlist {
            full_softmax: false,
            online_softmax: false,
            flash_attention: false,
        };
        let error = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget::default(),
            &search,
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least one backend"));
    }

    #[test]
    fn backend_policy_requires_matching_memory_domain_and_workspace() {
        let error = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            ResourceAssumptions::h3_bf16_mmap(),
            ResourceBudget::default(),
            &compact_space(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("device_memory_is_host=true"));

        let gpu_policy = cuda_policy();
        let mut host_charged = cpu_assumptions();
        host_charged.backend_workspace_bytes = 1;
        let error = solve_fixture(
            &gpu_policy,
            tiny_shape(),
            tiny_geometry(),
            host_charged,
            ResourceBudget::default(),
            &compact_space(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("device_memory_is_host=false"));

        let error = solve_fixture(
            &gpu_policy,
            tiny_shape(),
            tiny_geometry(),
            ResourceAssumptions::h3_bf16_mmap(),
            ResourceBudget::default(),
            &compact_space(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-zero backend workspace"));

        let mut gpu_assumptions = ResourceAssumptions::h3_bf16_mmap();
        gpu_assumptions.backend_workspace_bytes = 1;
        assert!(
            solve_fixture(
                &gpu_policy,
                tiny_shape(),
                tiny_geometry(),
                gpu_assumptions,
                ResourceBudget::default(),
                &compact_space(),
            )
            .is_ok()
        );
    }

    #[test]
    fn flash_attention_rejects_unsupported_head_dimensions() {
        let gpu_policy = cuda_policy();
        let mut gpu_assumptions = ResourceAssumptions::h3_bf16_mmap();
        gpu_assumptions.backend_workspace_bytes = 1;
        let mut search = compact_space();
        search.attention_backends = AttentionBackendAllowlist {
            full_softmax: false,
            online_softmax: false,
            flash_attention: true,
        };
        let error = solve_fixture(
            &gpu_policy,
            tiny_shape(),
            tiny_geometry(),
            gpu_assumptions,
            ResourceBudget::default(),
            &search,
        )
        .unwrap_err();
        assert!(error.to_string().contains("divisible by 8 and at most 512"));
    }

    #[cfg(not(feature = "flash-attn"))]
    #[test]
    fn flash_attention_requires_the_compiled_feature() {
        let gpu_policy = cuda_policy();
        let mut gpu_assumptions = ResourceAssumptions::h3_bf16_mmap();
        gpu_assumptions.backend_workspace_bytes = 1;
        let mut search = compact_space();
        search.attention_backends = AttentionBackendAllowlist {
            full_softmax: false,
            online_softmax: false,
            flash_attention: true,
        };
        let error = solve_fixture(
            &gpu_policy,
            TransformerShape::h3_base(),
            tiny_geometry(),
            gpu_assumptions,
            ResourceBudget::default(),
            &search,
        )
        .unwrap_err();
        assert!(error.to_string().contains("--features flash-attn"));
    }

    #[test]
    fn online_tiles_cannot_cross_an_unallowlisted_full_backend() {
        let search = PolicySearchSpace {
            attention_backends: AttentionBackendAllowlist {
                full_softmax: false,
                online_softmax: true,
                flash_attention: false,
            },
            attention_projection_rows: vec![nz(1)],
            attention_query_rows: vec![nz(1)],
            attention_key_rows: vec![nz(3), nz(30)],
            feed_forward_rows: vec![nz(1)],
            output_rows: vec![nz(1)],
            precompute_adaln: vec![false],
        };
        let candidates = solve_fixture(
            &base_policy(),
            tiny_shape(),
            tiny_geometry(),
            cpu_assumptions(),
            ResourceBudget::default(),
            &search,
        )
        .unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn legacy_additional_allowance_is_explicit_and_does_not_rewrite_cache_knobs() {
        let policy = base_policy();
        let inventory = cache_inventory();
        let charge = crate::resources::host_weight_residency_charges(&policy, &inventory).unwrap();
        let mut assumptions = cpu_assumptions();
        assumptions.host_weight_cache_bytes = charge.owned_weight_bytes + 4096;
        assumptions.mapped_weight_residency_bytes = charge.mapped_weight_bytes;
        let solve = |allowance| {
            super::solve_feasible_policies(
                &policy,
                tiny_shape(),
                tiny_geometry(),
                assumptions,
                ResourceBudget::default(),
                &compact_space(),
                SolverHostWeights {
                    inventory: &inventory,
                    additional_host_allowance_bytes: allowance,
                },
            )
        };
        assert!(solve(0).is_err());
        let candidates = solve(4096).unwrap();
        assert!(!candidates.is_empty());
        for candidate in candidates {
            assert_eq!(candidate.policy.weights, policy.weights);
            assert_eq!(candidate.estimate.assumptions.host_weight_cache_bytes, 4096);
        }
    }

    #[test]
    fn accepts_charged_memory_but_rejects_mismatched_assumptions() {
        for memory in [false, true] {
            {
                let mut policy = base_policy();
                policy.weights.source = if memory {
                    crate::policy::WeightSourcePolicy::Memory
                } else {
                    crate::policy::WeightSourcePolicy::Mmap
                };
                let candidates = solve_fixture(
                    &policy,
                    tiny_shape(),
                    tiny_geometry(),
                    cpu_assumptions(),
                    ResourceBudget::default(),
                    &compact_space(),
                )
                .unwrap();
                assert!(!candidates.is_empty());
                assert!(
                    candidates
                        .iter()
                        .all(|candidate| candidate.policy.weights == policy.weights)
                );
                let error = super::solve_feasible_policies(
                    &policy,
                    tiny_shape(),
                    tiny_geometry(),
                    cpu_assumptions(),
                    ResourceBudget::default(),
                    &compact_space(),
                    SolverHostWeights {
                        inventory: &cache_inventory(),
                        additional_host_allowance_bytes: 0,
                    },
                )
                .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("host weight assumptions disagree")
                );
            }
        }
    }
}
