//! Topology-driven configuration inputs for edge0.
//!
//! [`Edge0Requirement`] reports the weighed checkpoint split from
//! [`crate::mode::WeightSizes`] to the shared derivation: the routed-expert
//! pool is the partial residency, the static skeleton plus embed/lm_head is
//! its fixed cost, and both stream through the device pool.

use crate::mode::WeightSizes;
use anyhow::Result;
use ff_core::configure::{DerivedConfig, ModelRequirement, ResidencyDomain};
use ff_core::topology::TopologyProfile;

pub struct Edge0Requirement {
    expert_total_bytes: u64,
    fixed_bytes: u64,
}

impl Edge0Requirement {
    pub fn from_sizes(sizes: &WeightSizes) -> Self {
        Self {
            expert_total_bytes: sizes.expert_packed + sizes.expert_scale_bias,
            fixed_bytes: sizes.static_packed + sizes.static_scale_bias + sizes.embed_lm_head,
        }
    }
}

impl ModelRequirement for Edge0Requirement {
    fn steady_weight_bytes(&self) -> Result<u64> {
        Ok(self.expert_total_bytes)
    }

    fn activation_peak_bytes(&self) -> Result<u64> {
        Ok(0)
    }

    fn memory_materialization_bytes(&self) -> Result<u64> {
        Ok(self.expert_total_bytes + self.fixed_bytes)
    }

    fn poolable_resident_bytes(&self) -> Result<u64> {
        Ok(self.expert_total_bytes)
    }

    fn poolable_fixed_bytes(&self) -> Result<u64> {
        Ok(self.fixed_bytes)
    }

    fn poolable_residency_domain(&self) -> ResidencyDomain {
        ResidencyDomain::Device
    }
}

/// Derive the zero-flag configuration for the device at CUDA `ordinal`.
///
/// Edge0 has no partial expert cache: `--resident-experts` uploads every
/// expert or the planner refuses (`crates/ff-edge0/src/model.rs`
/// `enable_gpu`'s `FullResident`-only gate). A partial rule-7 cap is snapped
/// down to zero so the reported residency matches an achievable mode.
pub fn derive_edge0_configuration(
    sizes: &WeightSizes,
    profile: &TopologyProfile,
    ordinal: usize,
) -> Result<DerivedConfig> {
    let requirement = Edge0Requirement::from_sizes(sizes);
    let index = profile
        .devices
        .iter()
        .position(|device| device.ordinal == ordinal)
        .unwrap_or(ordinal);
    let mut derived = ff_core::configure::derive(index, profile, &requirement)?;
    if let Some(bytes) = derived.pool_resident_bytes
        && bytes < requirement.expert_total_bytes
    {
        derived.snap_pool_residency_to_streaming(
            "edge0 has no partial expert cache, so residency snaps to full host streaming",
        );
    }
    Ok(derived)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::configure::WeightSourceChoice;
    use ff_core::probe::{DeviceBackend, HardwareFingerprint};
    use ff_core::topology::{InterconnectLevel, TopologyDevice};

    fn synthetic_sizes() -> WeightSizes {
        WeightSizes {
            expert_packed: 15 << 30,
            expert_scale_bias: 15 << 27,
            static_packed: 684 << 20,
            static_scale_bias: 684 << 17,
            embed_lm_head: 545 << 20,
        }
    }

    fn profile(host_bytes: u64, device_bytes: u64) -> TopologyProfile {
        TopologyProfile {
            schema_version: ff_core::topology::TOPOLOGY_PROFILE_SCHEMA_VERSION,
            fingerprint: HardwareFingerprint::collect(&candle_core::Device::Cpu),
            host_memory_total_bytes: Some(host_bytes),
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
    fn the_derived_residency_matches_the_mode_planner_arithmetic() {
        let sizes = synthetic_sizes();
        let expert_total = sizes.expert_packed + sizes.expert_scale_bias;
        let fixed = sizes.static_packed + sizes.static_scale_bias + sizes.embed_lm_head;
        let requirement = Edge0Requirement::from_sizes(&sizes);

        // The 24 GiB discrete fixture from mode.rs: full residency, and the
        // derived rule-7 cap covers the whole expert pool.
        let hardware = crate::mode::Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(60 << 30),
            unified_probe_failed: false,
        };
        let plan = crate::mode::plan_mode(&hardware, &Default::default(), &sizes).unwrap();
        let derived = derive_edge0_configuration(&sizes, &profile(60 << 30, 24 << 30), 0).unwrap();
        assert_eq!(plan.mode, crate::mode::PerformanceMode::FullResident);
        assert_eq!(derived.pool_resident_bytes, Some(expert_total));

        // A budget too small for full residency: mode.rs's planner still
        // computes an advisory partial figure (it refuses the run instead of
        // executing a partial cache), so the derivation snaps its own
        // partial cap to zero rather than echoing an unachievable number.
        let small_hardware = crate::mode::Hardware {
            total_vram_bytes: Some(8 << 30),
            free_vram_bytes: Some(7 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(60 << 30),
            unified_probe_failed: false,
        };
        let plan = crate::mode::plan_mode(&small_hardware, &Default::default(), &sizes).unwrap();
        let derived = derive_edge0_configuration(&sizes, &profile(60 << 30, 8 << 30), 0).unwrap();
        assert_eq!(plan.mode, crate::mode::PerformanceMode::StreamingExperts);
        assert!(plan.expert_bytes_resident > 0 && plan.expert_bytes_resident < expert_total);
        assert_eq!(derived.pool_resident_bytes, Some(0));
        assert!(
            derived
                .provenance
                .iter()
                .find(|step| step.rule == ff_core::configure::RULE_POOL_RESIDENCY)
                .is_some_and(|step| step.detail.contains("no partial expert cache")
                    && step.detail.contains("full host streaming"))
        );
        assert_eq!(
            requirement.poolable_residency_domain(),
            ResidencyDomain::Device
        );
        assert_eq!(requirement.steady_weight_bytes().unwrap(), expert_total);
        assert_eq!(
            requirement.memory_materialization_bytes().unwrap(),
            expert_total + fixed
        );
        assert_eq!(derived.weight_source, WeightSourceChoice::Memory);
    }
}
