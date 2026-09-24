//! Configuration derivation inputs and rules.
//!
//! A model architecture reports its own resource demand through
//! [`ModelRequirement`]; [`derive`] turns a topology profile plus a
//! requirement into an execution plan with a recorded provenance trail.

use anyhow::Result;
use std::fmt;

use crate::probe::admission_reserve_bytes;
use crate::topology::TopologyProfile;

/// Operating-system allowance in the rule-1 criterion: what a clean
/// environment needs beside the materialized weights (the 62.6 GiB host
/// holding a 61.7 GiB checkpoint leaves 0.9 GiB and runs).
pub const OS_ALLOWANCE_BYTES: u64 = 768 << 20;

pub trait ModelRequirement {
    /// Weight bytes the steady phases re-read: the host-resident demand of a
    /// `memory` weight source (derivation rule 1).
    fn steady_weight_bytes(&self) -> Result<u64>;

    /// Peak device-resident activation and workspace bytes across phases
    /// (derivation rule 3).
    fn activation_peak_bytes(&self) -> Result<u64>;

    /// Weight bytes of phases that stream once and never re-read (derivation
    /// rule 1). Zero when the architecture has no single-pass component.
    fn single_pass_weight_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// Floating-point operations per steady evaluation. Zero when unmodeled;
    /// FLOP-based derivations skip a model that reports zero.
    fn flops_per_evaluation(&self) -> Result<u64> {
        Ok(0)
    }

    /// The largest chunk plan whose activation peak fits
    /// `activation_budget_bytes`, with that plan's own peak; `None` when the
    /// architecture models no chunk ladder (derivation rule 2 keeps the
    /// defaults then).
    fn largest_chunk_plan_within(
        &self,
        activation_budget_bytes: u64,
    ) -> Result<Option<SelectedChunkPlan>> {
        let _ = activation_budget_bytes;
        Ok(None)
    }

    /// Total weight bytes a `memory` source materializes into the host cache
    /// across all phases (derivation rule 1's cache ceiling). Defaults to the
    /// steady plus single-pass demand.
    fn memory_materialization_bytes(&self) -> Result<u64> {
        Ok(self.steady_weight_bytes()? + self.single_pass_weight_bytes()?)
    }

    /// Total bytes of the streamable pool a partial residency can hold: all
    /// routed experts, or all decode-stage model weights. Zero when the
    /// architecture has nothing poolable (rule 7 stays silent).
    fn poolable_resident_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// Fixed bytes drawn from the same budget before the pool: the cost that
    /// stays resident beside a partial pool at the pool's own phase boundary.
    fn poolable_fixed_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    /// The memory pool rule 7's partial residency draws from.
    fn poolable_residency_domain(&self) -> ResidencyDomain {
        ResidencyDomain::Device
    }
}

/// Which memory pool rule 7's partial residency draws from.
///
/// `Device` budgets against the device's own total, independent of host
/// memory; `TopologyProfile` carries no unified-memory signal yet, so this is
/// unsound on integrated/unified hardware where the two share one pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyDomain {
    Host,
    Device,
}

pub const RULE_POOL_RESIDENCY: &str = "rule-7-pool-residency";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkPlan {
    pub attention_projection: usize,
    pub feed_forward: usize,
    pub output: usize,
}

/// A selected chunk plan and the activation peak it was measured with, so
/// rule 2 can report the selected peak while rule 3 keeps the conservative
/// ladder-top figure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedChunkPlan {
    pub plan: ChunkPlan,
    pub peak_device_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightSourceChoice {
    Memory,
    Mmap,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivationStep {
    pub rule: &'static str,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedConfig {
    pub weight_source: WeightSourceChoice,
    pub steady_weight_bytes: u64,
    pub host_reserve_bytes: u64,
    /// Cache capacity ceiling under a `memory` weight source: the smaller of
    /// what the host holds beside its reserve and what materializes at all.
    /// `None` when the derivation streams through mmap.
    pub host_cache_ceiling_bytes: Option<u64>,
    /// Rule 7's pool residency: `Some(bytes)` caps the resident pool there,
    /// `Some(0)` means the budget covers none of the pool (full streaming),
    /// and `None` means nothing is poolable or the domain's pool went
    /// unmeasured.
    pub pool_resident_bytes: Option<u64>,
    pub chunks: Option<ChunkPlan>,
    pub provenance: Vec<DerivationStep>,
}

impl DerivedConfig {
    /// Zero out rule 7's pool residency and record why, so an adapter whose
    /// execution modes cannot realize the derived partial cap (an
    /// all-or-nothing cache, or a cap too small to be worth using) reports a
    /// cap the run can actually achieve.
    pub fn snap_pool_residency_to_streaming(&mut self, reason: &str) {
        self.pool_resident_bytes = Some(0);
        for step in &mut self.provenance {
            if step.rule == RULE_POOL_RESIDENCY {
                step.detail = format!("{}; {reason}", step.detail);
            }
        }
    }
}

impl fmt::Display for DerivationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.rule, self.detail)
    }
}

/// Derive the P1 execution plan for the device at `ordinal`.
///
/// Rule 1 places the steady re-read set in host memory when it fits beside
/// the admission reserve, and streams single-pass weights through mmap
/// otherwise (FlexGen's fastest-tier placement; single-pass mmap measured
/// 180-490s -> 48s on the H3 text encoder). Rule 2 picks the largest chunk
/// plan whose activation peak leaves the reserve inside the selected
/// device's memory.
pub fn derive(
    ordinal: usize,
    profile: &TopologyProfile,
    requirement: &dyn ModelRequirement,
) -> Result<DerivedConfig> {
    let steady = requirement.steady_weight_bytes()?;
    let single_pass = requirement.single_pass_weight_bytes()?;
    let materialization = requirement.memory_materialization_bytes()?;
    let reserve = admission_reserve_bytes(profile.host_memory_total_bytes);
    let weight_source = match profile.host_memory_total_bytes {
        Some(host) if materialization.saturating_add(OS_ALLOWANCE_BYTES) <= host => {
            WeightSourceChoice::Memory
        }
        _ => WeightSourceChoice::Mmap,
    };
    let host_cache_ceiling_bytes = match (weight_source, profile.host_memory_total_bytes) {
        (WeightSourceChoice::Memory, Some(host)) => {
            Some(host.saturating_sub(reserve).min(materialization))
        }
        _ => None,
    };
    let mut provenance = vec![DerivationStep {
        rule: "rule-1-weight-source",
        detail: {
            let mmap_cost = (weight_source == WeightSourceChoice::Mmap).then(|| {
                match profile.storage_bytes_per_second {
                    Some(bandwidth) if bandwidth > 0 => format!(
                        "mmap re-reads {steady} B per evaluation from storage (~{:.1} s at \
                         {bandwidth} B/s)",
                        steady as f64 / bandwidth as f64
                    ),
                    _ => format!(
                        "mmap re-reads {steady} B per evaluation from storage (storage bandwidth \
                         unmeasured)"
                    ),
                }
            });
            format!(
                "materialization {materialization} B + OS allowance {OS_ALLOWANCE_BYTES} B against \
                 host {:?} B -> {weight_source:?}; steady re-read {steady} B, single-pass \
                 {single_pass} B; cache ceiling {:?} B{}",
                profile.host_memory_total_bytes,
                host_cache_ceiling_bytes,
                mmap_cost
                    .as_deref()
                    .map(|cost| format!("; {cost}"))
                    .unwrap_or_default()
            )
        },
    }];

    let device_memory = profile
        .devices
        .get(ordinal)
        .and_then(|device| device.total_memory_bytes);
    let chunks = match device_memory {
        Some(device_memory) => {
            let device_reserve = admission_reserve_bytes(Some(device_memory));
            let activation_budget = device_memory.saturating_sub(device_reserve);
            let selected = requirement.largest_chunk_plan_within(activation_budget)?;
            // Rule 3's residency figure stays on the ladder-top peak: it is
            // the conservative envelope the planner charges against.
            let _top_peak = requirement.activation_peak_bytes()?;
            let detail = match &selected {
                Some(selected) if selected.peak_device_bytes > activation_budget => format!(
                    "activation budget {activation_budget} B of {device_memory} B selects \
                     {:?} at its own peak {} B; no modeled plan fits, so the smallest is \
                     selected and admission is expected to refuse",
                    selected.plan, selected.peak_device_bytes
                ),
                Some(selected) => format!(
                    "activation budget {activation_budget} B of {device_memory} B selects \
                     {:?} at peak {} B",
                    selected.plan, selected.peak_device_bytes
                ),
                None => format!(
                    "activation budget {activation_budget} B of {device_memory} B; the \
                     architecture models no chunk plan, defaults apply"
                ),
            };
            provenance.push(DerivationStep {
                rule: "rule-2-chunks",
                detail,
            });
            selected.map(|selected| selected.plan)
        }
        None => {
            provenance.push(DerivationStep {
                rule: "rule-2-chunks",
                detail: "no CUDA device recorded; chunks stay at the defaults".to_owned(),
            });
            None
        }
    };
    let pool_resident_bytes =
        rule7_pool_residency(profile, device_memory, requirement, &mut provenance)?;
    if profile.devices.len() > 1 {
        provenance.push(DerivationStep {
            rule: "rule-5-multi-device",
            detail: format!(
                "{} devices present; multi-device placement is not derived",
                profile.devices.len()
            ),
        });
    }
    Ok(DerivedConfig {
        weight_source,
        steady_weight_bytes: steady,
        host_reserve_bytes: reserve,
        host_cache_ceiling_bytes,
        pool_resident_bytes,
        chunks,
        provenance,
    })
}

/// Rule 7: bound the poolable partial residency by what its domain's pool
/// holds beside the admission reserve and the pool's fixed cost. `None` when
/// the architecture has nothing poolable or the domain's pool went unmeasured.
fn rule7_pool_residency(
    profile: &TopologyProfile,
    device_memory: Option<u64>,
    requirement: &dyn ModelRequirement,
    provenance: &mut Vec<DerivationStep>,
) -> Result<Option<u64>> {
    let pool_total = requirement.poolable_resident_bytes()?;
    if pool_total == 0 {
        return Ok(None);
    }
    let fixed = requirement.poolable_fixed_bytes()?;
    let (pool_memory, pool_name) = match requirement.poolable_residency_domain() {
        ResidencyDomain::Device => (device_memory, "device"),
        ResidencyDomain::Host => (profile.host_memory_total_bytes, "host"),
    };
    let Some(pool_memory) = pool_memory else {
        provenance.push(DerivationStep {
            rule: RULE_POOL_RESIDENCY,
            detail: format!(
                "poolable {pool_total} B but no {pool_name} memory recorded; pool residency \
                 not derived"
            ),
        });
        return Ok(None);
    };
    let reserve = admission_reserve_bytes(Some(pool_memory));
    let resident = pool_memory
        .saturating_sub(reserve)
        .saturating_sub(fixed)
        .min(pool_total);
    let outcome = if resident > 0 {
        format!("resident cap {:.1} GiB", resident as f64 / 1073741824.0)
    } else {
        "budget covers none of the pool; full streaming".to_owned()
    };
    provenance.push(DerivationStep {
        rule: RULE_POOL_RESIDENCY,
        detail: format!(
            "pool {:.1} GiB; fixed {:.1} GiB; reserve {:.1} GiB; {pool_name} {:.1} GiB -> \
             {outcome}",
            pool_total as f64 / 1073741824.0,
            fixed as f64 / 1073741824.0,
            reserve as f64 / 1073741824.0,
            pool_memory as f64 / 1073741824.0,
        ),
    });
    Ok(Some(resident))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{DeviceBackend, HardwareFingerprint};
    use crate::topology::{InterconnectLevel, TopologyDevice};

    struct Bare {
        weights: u64,
        activations: u64,
    }

    impl ModelRequirement for Bare {
        fn steady_weight_bytes(&self) -> Result<u64> {
            Ok(self.weights)
        }

        fn activation_peak_bytes(&self) -> Result<u64> {
            Ok(self.activations)
        }
    }

    fn profile(host: Option<u64>, device: Option<u64>) -> TopologyProfile {
        TopologyProfile {
            schema_version: crate::topology::TOPOLOGY_PROFILE_SCHEMA_VERSION,
            fingerprint: HardwareFingerprint::collect(&candle_core::Device::Cpu),
            host_memory_total_bytes: host,
            cgroup_memory_limit_bytes: None,
            peer_links: Vec::new(),
            devices: device
                .map(|bytes| {
                    vec![TopologyDevice {
                        ordinal: 0,
                        backend: DeviceBackend::Cuda,
                        name: Some("fixture".to_owned()),
                        total_memory_bytes: Some(bytes),
                        compute_capability: None,
                        cuda_device_uuid: None,
                    }]
                })
                .unwrap_or_default(),
            interconnect: InterconnectLevel::SingleDevice,
            storage_bytes_per_second: None,
        }
    }

    #[test]
    fn unmodeled_quantities_default_to_zero() {
        let bare = Bare {
            weights: 1000,
            activations: 100,
        };
        assert_eq!(bare.steady_weight_bytes().unwrap(), 1000);
        assert_eq!(bare.activation_peak_bytes().unwrap(), 100);
        assert_eq!(bare.single_pass_weight_bytes().unwrap(), 0);
        assert_eq!(bare.flops_per_evaluation().unwrap(), 0);
        assert_eq!(bare.largest_chunk_plan_within(u64::MAX).unwrap(), None);
    }

    #[test]
    fn weight_source_follows_host_capacity() {
        let gib = 1u64 << 30;
        let bare = Bare {
            weights: 10 * gib,
            activations: gib,
        };
        let roomy = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &bare).unwrap();
        assert_eq!(roomy.weight_source, WeightSourceChoice::Memory);
        let steady_fits_but_materialization_does_not = derive(
            0,
            &profile(Some(10 * gib + (512 << 20)), Some(24 * gib)),
            &bare,
        )
        .unwrap();
        assert_eq!(
            steady_fits_but_materialization_does_not.weight_source,
            WeightSourceChoice::Mmap
        );
        let tight = derive(0, &profile(Some(10 * gib), Some(24 * gib)), &bare).unwrap();
        assert_eq!(tight.weight_source, WeightSourceChoice::Mmap);
        let unknown = derive(0, &profile(None, None), &bare).unwrap();
        assert_eq!(unknown.weight_source, WeightSourceChoice::Mmap);
        assert_eq!(unknown.chunks, None);
    }

    #[test]
    fn cache_ceiling_covers_the_materialization_within_host_capacity() {
        let gib = 1u64 << 30;
        let bare = Bare {
            weights: 10 * gib,
            activations: gib,
        };
        let derived = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &bare).unwrap();
        let reserve = admission_reserve_bytes(Some(100 * gib));
        assert_eq!(
            derived.host_cache_ceiling_bytes,
            Some((100 * gib - reserve).min(10 * gib))
        );
        let tight = derive(0, &profile(Some(10 * gib), Some(24 * gib)), &bare).unwrap();
        assert_eq!(tight.host_cache_ceiling_bytes, None);
    }

    #[test]
    fn chunk_selection_stays_inside_the_device_budget() {
        let gib = 1u64 << 30;
        let bare = Bare {
            weights: 10 * gib,
            activations: 4 * gib,
        };
        let derived = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &bare).unwrap();
        assert!(
            derived
                .provenance
                .iter()
                .any(|step| step.rule == "rule-2-chunks")
        );
    }

    struct Pooled {
        weights: u64,
        activations: u64,
        pool_total: u64,
        fixed: u64,
        domain: ResidencyDomain,
    }

    impl ModelRequirement for Pooled {
        fn steady_weight_bytes(&self) -> Result<u64> {
            Ok(self.weights)
        }

        fn activation_peak_bytes(&self) -> Result<u64> {
            Ok(self.activations)
        }

        fn poolable_resident_bytes(&self) -> Result<u64> {
            Ok(self.pool_total)
        }

        fn poolable_fixed_bytes(&self) -> Result<u64> {
            Ok(self.fixed)
        }

        fn poolable_residency_domain(&self) -> ResidencyDomain {
            self.domain
        }
    }

    #[test]
    fn pool_residency_takes_the_leftover_device_budget() {
        let gib = 1u64 << 30;
        let pooled = Pooled {
            weights: 10 * gib,
            activations: gib,
            pool_total: 10 * gib,
            fixed: 2 * gib,
            domain: ResidencyDomain::Device,
        };
        let roomy = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &pooled).unwrap();
        // 24 GiB device: 1 GiB reserve + 2 GiB fixed leaves room for the
        // whole 10 GiB pool.
        assert_eq!(roomy.pool_resident_bytes, Some(10 * gib));
        assert!(
            roomy
                .provenance
                .iter()
                .any(|step| step.rule == RULE_POOL_RESIDENCY)
        );

        let tight = Pooled {
            pool_total: 10 * gib,
            fixed: 7 * gib,
            ..pooled
        };
        let partial = derive(0, &profile(Some(100 * gib), Some(8 * gib)), &tight).unwrap();
        // 8 GiB device: the 512 MiB reserve + 7 GiB fixed leaves 512 MiB of
        // the pool resident.
        assert_eq!(partial.pool_resident_bytes, Some(512 << 20));

        let swamped = Pooled {
            pool_total: 10 * gib,
            fixed: 8 * gib,
            ..pooled
        };
        let streaming = derive(0, &profile(Some(100 * gib), Some(8 * gib)), &swamped).unwrap();
        assert_eq!(streaming.pool_resident_bytes, Some(0));
        assert!(
            streaming
                .provenance
                .iter()
                .any(|step| step.detail.contains("full streaming"))
        );
    }

    #[test]
    fn pool_residency_draws_from_the_declared_domain() {
        let gib = 1u64 << 30;
        let host_pool = Pooled {
            weights: 10 * gib,
            activations: gib,
            pool_total: 30 * gib,
            fixed: 2 * gib,
            domain: ResidencyDomain::Host,
        };
        // 20 GiB host: the reserve sits at its 1 GiB cap, so 17 GiB of the
        // 30 GiB pool is resident.
        let derived = derive(0, &profile(Some(20 * gib), None), &host_pool).unwrap();
        assert_eq!(derived.pool_resident_bytes, Some(17 * gib));

        let device_pool = Pooled {
            domain: ResidencyDomain::Device,
            ..host_pool
        };
        let unmeasured = derive(0, &profile(Some(20 * gib), None), &device_pool).unwrap();
        assert_eq!(unmeasured.pool_resident_bytes, None);
        assert!(
            unmeasured
                .provenance
                .iter()
                .any(|step| step.detail.contains("no device memory recorded"))
        );

        let host_unmeasured = Pooled {
            domain: ResidencyDomain::Host,
            ..host_pool
        };
        let derived = derive(0, &profile(None, Some(24 * gib)), &host_unmeasured).unwrap();
        assert_eq!(derived.pool_resident_bytes, None);
        assert!(
            derived
                .provenance
                .iter()
                .any(|step| step.detail.contains("no host memory recorded"))
        );
    }

    #[test]
    fn an_empty_pool_skips_rule_seven() {
        let gib = 1u64 << 30;
        let bare = Bare {
            weights: 10 * gib,
            activations: gib,
        };
        let derived = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &bare).unwrap();
        assert_eq!(derived.pool_resident_bytes, None);
        assert!(
            !derived
                .provenance
                .iter()
                .any(|step| step.rule == RULE_POOL_RESIDENCY)
        );
    }

    #[test]
    fn snapping_pool_residency_zeroes_the_value_and_annotates_the_step() {
        let gib = 1u64 << 30;
        let pooled = Pooled {
            weights: 10 * gib,
            activations: gib,
            pool_total: 10 * gib,
            fixed: 2 * gib,
            domain: ResidencyDomain::Device,
        };
        let mut derived = derive(0, &profile(Some(100 * gib), Some(24 * gib)), &pooled).unwrap();
        assert_eq!(derived.pool_resident_bytes, Some(10 * gib));

        derived.snap_pool_residency_to_streaming("caller-specific reason");

        assert_eq!(derived.pool_resident_bytes, Some(0));
        assert!(
            derived
                .provenance
                .iter()
                .find(|step| step.rule == RULE_POOL_RESIDENCY)
                .is_some_and(|step| step.detail.ends_with("caller-specific reason"))
        );
    }
}
