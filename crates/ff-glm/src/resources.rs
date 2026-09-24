//! Topology-driven configuration inputs for GLM.
//!
//! [`GlmRequirement`] reports the routed-expert pool to the shared
//! derivation: the dequantized expert working set is the partial residency
//! and the phase device charges at a zero expert cache are its fixed cost.

use crate::admission::{ExpertCacheBound, GlmAdmissionBreakdown};
use anyhow::{Context, Result};
use ff_core::configure::{DerivedConfig, ModelRequirement, ResidencyDomain};
use ff_core::topology::TopologyProfile;
use ff_core::weights::CachePolicy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmRequirement {
    expert_pool_bytes: u64,
    fixed_bytes: u64,
}

impl GlmRequirement {
    /// `resident_static` and `cache_policy` select which streamed-load
    /// charges the phases carry; the expert cache itself enters as the pool,
    /// so the fixed cost is folded at a zero cache bound. The phase device
    /// reserves stay out of the fold — rule 7 charges the admission reserve
    /// once, from the topology profile.
    pub fn from_breakdown(
        breakdown: &GlmAdmissionBreakdown,
        resident_static: bool,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let phases = breakdown.phases(
            resident_static,
            ExpertCacheBound::new(0, crate::expert_cache_manager::ExpertCacheLayout::default()),
            cache_policy,
        )?;
        let fixed_bytes = phases
            .iter()
            .try_fold(0u64, |peak, phase| {
                Ok::<_, anyhow::Error>(peak.max(phase.required_device_bytes.unwrap_or(0)))
            })?;
        let entry_bytes = breakdown.expert_cache_entry_bytes();
        let expert_pool_bytes = entry_bytes
            .checked_mul(3)
            .and_then(|n| n.checked_mul(breakdown.num_experts as u64))
            .and_then(|n| n.checked_mul(breakdown.sparse_layers.len() as u64))
            .context("GLM expert working set overflow")?;
        Ok(Self {
            expert_pool_bytes,
            fixed_bytes,
        })
    }
}

impl ModelRequirement for GlmRequirement {
    fn steady_weight_bytes(&self) -> Result<u64> {
        Ok(self.expert_pool_bytes)
    }

    fn activation_peak_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    fn memory_materialization_bytes(&self) -> Result<u64> {
        Ok(self.expert_pool_bytes + self.fixed_bytes)
    }

    fn poolable_resident_bytes(&self) -> Result<u64> {
        Ok(self.expert_pool_bytes)
    }

    fn poolable_fixed_bytes(&self) -> Result<u64> {
        Ok(self.fixed_bytes)
    }

    fn poolable_residency_domain(&self) -> ResidencyDomain {
        ResidencyDomain::Device
    }
}

/// Derive the zero-flag configuration for the device at CUDA `ordinal`.
pub fn derive_glm_configuration(
    breakdown: &GlmAdmissionBreakdown,
    resident_static: bool,
    cache_policy: CachePolicy,
    profile: &TopologyProfile,
    ordinal: usize,
) -> Result<DerivedConfig> {
    let requirement = GlmRequirement::from_breakdown(breakdown, resident_static, cache_policy)?;
    let index = profile
        .devices
        .iter()
        .position(|device| device.ordinal == ordinal)
        .unwrap_or(ordinal);
    ff_core::configure::derive(index, profile, &requirement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::configure::RULE_POOL_RESIDENCY;
    use ff_core::probe::{DeviceBackend, HardwareFingerprint};
    use ff_core::topology::{InterconnectLevel, TopologyDevice};

    fn requirement() -> GlmRequirement {
        GlmRequirement {
            expert_pool_bytes: 30 << 30,
            fixed_bytes: 12 << 30,
        }
    }

    fn profile(device_bytes: u64) -> TopologyProfile {
        TopologyProfile {
            schema_version: ff_core::topology::TOPOLOGY_PROFILE_SCHEMA_VERSION,
            fingerprint: HardwareFingerprint::collect(&candle_core::Device::Cpu),
            host_memory_total_bytes: Some(62 << 30),
            cgroup_memory_limit_bytes: None,
            peer_links: Vec::new(),
            devices: vec![TopologyDevice {
                ordinal: 0,
                backend: DeviceBackend::Cuda,
                name: Some("fixture".to_owned()),
                total_memory_bytes: Some(device_bytes),
                compute_capability: None,
                cuda_device_uuid: None,
            }],
            interconnect: InterconnectLevel::SingleDevice,
            storage_bytes_per_second: None,
        }
    }

    #[test]
    fn the_expert_pool_covers_the_device_budget_left_over_by_the_phases() {
        let derived = ff_core::configure::derive(0, &profile(24 << 30), &requirement()).unwrap();
        let reserve = ff_core::probe::admission_reserve_bytes(Some(24 << 30));
        assert_eq!(
            derived.pool_resident_bytes,
            Some((24 << 30) - reserve - (12 << 30))
        );
        assert!(
            derived
                .provenance
                .iter()
                .any(|step| step.rule == RULE_POOL_RESIDENCY)
        );

        // A budget smaller than the fixed cost leaves none of the pool
        // resident.
        let swamped = ff_core::configure::derive(0, &profile(8 << 30), &requirement()).unwrap();
        assert_eq!(swamped.pool_resident_bytes, Some(0));
    }
}
