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

    /// Weight bytes of the decode-stage model (the visual VAE for H3), which
    /// runs after the denoise phases end. Zero when the architecture has no
    /// separate decode-stage model (derivation rule 6 then stays silent).
    fn vae_weight_bytes(&self) -> Result<u64> {
        Ok(0)
    }
}

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

/// Rule 6's outcome for the decode-stage model: `Some(bytes)` reserves that
/// many device bytes for a resident decoder; `None` streams it per layer
/// group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VaeResidency {
    pub weight_bytes: u64,
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
    pub vae_resident_bytes: Option<u64>,
    pub chunks: Option<ChunkPlan>,
    pub provenance: Vec<DerivationStep>,
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
    let vae_weight_bytes = requirement.vae_weight_bytes()?;
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
    let mut vae_resident_bytes = None;
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
            // Rule 6: the decode-stage model runs after the denoise phases
            // end, so the device it sees is free of the denoise residency.
            // The cache cap must leave room for the decode workspace and the
            // admission reserve beside the VAE weights.
            let decode_workspace = vae_weight_bytes / 4;
            let vae_resident = if vae_weight_bytes > 0 {
                let available_for_vae = device_memory
                    .saturating_sub(device_reserve)
                    .saturating_sub(decode_workspace);
                (available_for_vae >= vae_weight_bytes).then_some(vae_weight_bytes)
            } else {
                None
            };
            let vae_detail = if vae_weight_bytes == 0 {
                None
            } else {
                let device_reserve = admission_reserve_bytes(Some(device_memory));
                let resident_note = vae_resident
                    .map(|cap| {
                        format!(
                            "resident; decoder cache capped at {:.1} GiB",
                            cap as f64 / 1073741824.0
                        )
                    })
                    .unwrap_or_else(|| {
                        "device cannot hold the decoder beside its workspace; streaming per \
                         layer-group"
                            .to_owned()
                    });
                Some(format!(
                    "vae {:.1} GiB; decode workspace {:.1} GiB; reserve {:.1} GiB; device \
                     {:.1} GiB at the decode boundary (transformer residency dropped) -> {}",
                    vae_weight_bytes as f64 / 1073741824.0,
                    decode_workspace as f64 / 1073741824.0,
                    device_reserve as f64 / 1073741824.0,
                    device_memory as f64 / 1073741824.0,
                    resident_note
                ))
            };
            if let Some(vae_detail) = vae_detail {
                provenance.push(DerivationStep {
                    rule: "rule-6-vae-residency",
                    detail: vae_detail,
                });
            }
            vae_resident_bytes = vae_resident;
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
        vae_resident_bytes,
        chunks,
        provenance,
    })
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
}
