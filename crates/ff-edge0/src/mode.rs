//! Performance-mode planner: senses hardware, applies user configuration,
//! and derives the best residency mode — re-deriving whenever configuration
//! changes (e.g. a VRAM budget unset defaults to all VRAM; setting one
//! re-runs the whole derivation). Every decision carries a provenance line
//! so a run's mode is attributable after the fact.

use anyhow::{Context, Result, ensure};
use ff_core::probe::admission_reserve_bytes;

#[derive(Clone, Debug, PartialEq)]
pub struct Hardware {
    pub total_vram_bytes: Option<u64>,
    pub free_vram_bytes: Option<u64>,
    /// From the unified-memory three-tier probe; None = unprobed.
    pub unified_host_device: Option<bool>,
    /// Host availability already clamped to the cgroup limit where one
    /// applies (MemAvailable alone ignores a container's memory ceiling).
    pub host_memory_available_bytes: Option<u64>,
    /// A failed CUDA topology query; planning requires a successful probe.
    pub unified_probe_failed: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModeOverrides {
    /// User VRAM budget in bytes. `None` = default (all VRAM minus
    /// reserve); `Some(n)` re-derives every downstream decision against n.
    pub vram_budget_bytes: Option<u64>,
    /// Force a mode regardless of hardware (escape hatch; recorded in
    /// provenance so a forced run is never mistaken for a sensed one).
    pub force_host_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PerformanceMode {
    /// All experts resident as packed int4 plus static weights in VRAM.
    FullResident,
    /// Static weights + embed/lm_head resident; routed experts stream per
    /// token through pinned staging (the mode bounded by PCIe, not VRAM).
    StreamingExperts,
    /// Everything on host (the P0 CPU path).
    HostOnly,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModePlan {
    pub mode: PerformanceMode,
    pub budget_bytes: u64,
    pub expert_bytes_resident: u64,
    pub provenance: Vec<String>,
}

/// Weight byte totals WEIGHED from the checkpoint shards (the
/// `bucket_bytes` index scan) — hand-derived geometry formulas were
/// retired after one overestimated static weights by 1.8×.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeightSizes {
    pub expert_packed: u64,
    pub expert_scale_bias: u64,
    pub static_packed: u64,
    pub static_scale_bias: u64,
    pub embed_lm_head: u64,
}

impl WeightSizes {
    pub fn from_weights(weights: &crate::weights::Edge0Weights) -> Result<Self> {
        let (ep, esb, sp, ssb, embed_packed, embed_sb) = weights.bucket_bytes()?;
        Ok(Self {
            expert_packed: ep,
            expert_scale_bias: esb,
            static_packed: sp,
            static_scale_bias: ssb,
            embed_lm_head: embed_packed + embed_sb,
        })
    }
}

/// Derive the performance mode from hardware plus user overrides. Pure:
/// same inputs give the same plan, and every branch appends a provenance
/// line naming the inputs that selected it.
pub fn plan_mode(
    hardware: &Hardware,
    overrides: &ModeOverrides,
    sizes: &WeightSizes,
) -> Result<ModePlan> {
    let mut provenance = Vec::new();
    let expert_total = sizes.expert_packed + sizes.expert_scale_bias;
    let fixed = sizes.static_packed + sizes.static_scale_bias + sizes.embed_lm_head;
    provenance.push(format!(
        "weights (weighed): experts {expert_total} B (packed {} + s/b {}), fixed {fixed} B",
        sizes.expert_packed, sizes.expert_scale_bias
    ));

    ensure!(
        !hardware.unified_probe_failed,
        "unified_host_device probe failed; run ff probe --device cuda:N --json"
    );
    if overrides.force_host_only {
        provenance.push("mode: HOST-ONLY forced by user override".into());
        return Ok(ModePlan {
            mode: PerformanceMode::HostOnly,
            budget_bytes: 0,
            expert_bytes_resident: 0,
            provenance,
        });
    }

    let Some(total) = hardware.total_vram_bytes else {
        ensure!(
            hardware.free_vram_bytes.is_none() && hardware.unified_host_device.is_none(),
            "total_vram_bytes unavailable; run ff probe --device cuda:N --json"
        );
        provenance.push("mode: HOST-ONLY — no VRAM reported by probe".into());
        return Ok(ModePlan {
            mode: PerformanceMode::HostOnly,
            budget_bytes: 0,
            expert_bytes_resident: 0,
            provenance,
        });
    };

    let reserve = admission_reserve_bytes(Some(total));

    let unified = hardware
        .unified_host_device
        .context("unified_host_device unavailable; run ff probe --device cuda:N --json")?;
    let free = hardware
        .free_vram_bytes
        .context("free_vram_bytes unavailable; run ff probe --device cuda:N --json")?;
    let unified_pool = if unified {
        let host = hardware
            .host_memory_available_bytes
            .context("host_memory_available_bytes unavailable; run ff probe --json")?;
        let pool = host.min(free);
        provenance.push(format!(
            "unified pool: host view {host} B, device free view {free} B -> pool {pool} B"
        ));
        Some(pool)
    } else {
        None
    };

    let budget = match overrides.vram_budget_bytes {
        Some(user) => {
            // User budgets are ASSUMED to already include runtime overhead;
            // a unified pool still caps what the shared memory can supply.
            let cap = unified_pool.map_or(total, |pool| total.min(pool));
            let b = user.min(cap);
            provenance.push(format!(
                "budget: user {user} B capped to {b} B — user budget is ASSUMED to already include runtime overhead (context, activations, KV/GDN state)"
            ));
            b
        }
        None => {
            let source = unified_pool.unwrap_or(total);
            let b = source.saturating_sub(reserve);
            if unified_pool.is_none()
                && let Some(free) = hardware.free_vram_bytes
            {
                provenance.push(format!("free VRAM at probe: {free} B"));
            }
            provenance.push(format!(
                "budget: unset → pool {source} B − scaled reserve {reserve} B = {b} B"
            ));
            b
        }
    };

    if budget >= fixed + expert_total {
        provenance.push(format!(
            "mode: FULL-RESIDENT — budget {budget} B ≥ fixed {fixed} + experts {expert_total}"
        ));
        Ok(ModePlan {
            mode: PerformanceMode::FullResident,
            budget_bytes: budget,
            expert_bytes_resident: expert_total,
            provenance,
        })
    } else if budget >= fixed {
        let resident = (budget - fixed).min(expert_total);
        provenance.push(format!(
            "mode: STREAMING-EXPERTS — budget {budget} B covers fixed {fixed} B; {resident} of {expert_total} B experts resident, rest streams"
        ));
        Ok(ModePlan {
            mode: PerformanceMode::StreamingExperts,
            budget_bytes: budget,
            expert_bytes_resident: resident,
            provenance,
        })
    } else {
        provenance.push(format!(
            "mode: HOST-ONLY — budget {budget} B < fixed weights {fixed} B"
        ));
        Ok(ModePlan {
            mode: PerformanceMode::HostOnly,
            budget_bytes: budget,
            expert_bytes_resident: 0,
            provenance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Edge0Weights;
    use ff_core::paths::checkpoint_dir;

    /// The real checkpoint's weighed sizes as five constants; the audit test
    /// holds this fixture to the checkpoint.
    fn synthetic_sizes() -> WeightSizes {
        WeightSizes {
            expert_packed: 15 << 30,
            expert_scale_bias: 15 << 27,
            static_packed: 684 << 20,
            static_scale_bias: 684 << 17,
            embed_lm_head: 545 << 20,
        }
    }

    /// Weighed against the real checkpoint.
    fn sizes() -> WeightSizes {
        let dir = &checkpoint_dir("Edge0/Edge0-35B-A3B-preview")
            .filter(|dir| dir.is_dir())
            .expect("requires FF_MODELS_DIR with Edge0/Edge0-35B-A3B-preview");
        let weights = Edge0Weights::open(dir).unwrap();
        WeightSizes::from_weights(&weights).unwrap()
    }

    fn hw24g() -> Hardware {
        Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(60 << 30),
            unified_probe_failed: false,
        }
    }

    #[test]
    #[ignore = "requires FF_MODELS_DIR with Edge0/Edge0-35B-A3B-preview"]
    fn weighed_sizes_match_the_index_audit() {
        let s = sizes();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        assert!((gib(s.expert_packed + s.expert_scale_bias) - 16.88).abs() < 0.05);
        assert!((s.expert_scale_bias as f64 / s.expert_packed as f64 - 0.125).abs() < 0.001);
        assert!((gib(s.static_packed + s.static_scale_bias) - 0.76).abs() < 0.05);
        assert!((gib(s.embed_lm_head) - 0.53).abs() < 0.05);
        let synth = synthetic_sizes();
        for (field, synthetic, real) in [
            ("expert_packed", synth.expert_packed, s.expert_packed),
            (
                "expert_scale_bias",
                synth.expert_scale_bias,
                s.expert_scale_bias,
            ),
            ("static_packed", synth.static_packed, s.static_packed),
            (
                "static_scale_bias",
                synth.static_scale_bias,
                s.static_scale_bias,
            ),
            ("embed_lm_head", synth.embed_lm_head, s.embed_lm_head),
        ] {
            assert!(
                (gib(synthetic) - gib(real)).abs() < 0.05,
                "synthetic_sizes {field} drifts from the checkpoint"
            );
        }
    }

    #[test]
    fn unset_budget_defaults_to_all_vram_and_fits_full_resident() {
        let sizes = synthetic_sizes();
        let plan = plan_mode(&hw24g(), &ModeOverrides::default(), &sizes).unwrap();
        assert_eq!(plan.mode, PerformanceMode::FullResident);
        assert!(plan.provenance.iter().any(|l| l.contains("scaled reserve")));
    }

    #[test]
    fn user_budget_rederives_to_streaming() {
        let sizes = synthetic_sizes();
        let plan = plan_mode(
            &hw24g(),
            &ModeOverrides {
                vram_budget_bytes: Some(18 << 30),
                force_host_only: false,
            },
            &sizes,
        )
        .unwrap();
        assert_eq!(plan.mode, PerformanceMode::StreamingExperts);
        assert!(
            plan.provenance
                .iter()
                .any(|l| l.contains("ASSUMED to already include"))
        );
    }

    #[test]
    fn reserve_scales_below_the_crossover_on_small_pools() {
        let sizes = synthetic_sizes();
        let hw = Hardware {
            total_vram_bytes: Some(8 << 30),
            free_vram_bytes: Some(7 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(16 << 30),
            unified_probe_failed: false,
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes).unwrap();
        assert_eq!(plan.budget_bytes, (8 << 30) - (512 << 20));
        assert_eq!(plan.mode, PerformanceMode::StreamingExperts);
    }

    #[test]
    fn tight_budget_falls_to_host_only() {
        let sizes = synthetic_sizes();
        let plan = plan_mode(
            &hw24g(),
            &ModeOverrides {
                vram_budget_bytes: Some(1 << 30),
                force_host_only: false,
            },
            &sizes,
        )
        .unwrap();
        assert_eq!(plan.mode, PerformanceMode::HostOnly);
    }

    #[test]
    fn unified_pool_budgets_against_measured_availability() {
        let sizes = synthetic_sizes();
        let hw = Hardware {
            total_vram_bytes: Some(6 << 30),
            free_vram_bytes: Some(7 << 29), // 3.5 GiB
            unified_host_device: Some(true),
            host_memory_available_bytes: Some(4 << 30),
            unified_probe_failed: false,
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes).unwrap();
        let pool = 7 << 29; // min(4 GiB host view, 3.5 GiB device free view)
        let reserve = 512 << 20; // the shared formula's floor at a 6 GiB total
        assert_eq!(plan.budget_bytes, pool - reserve);
        assert_eq!(plan.mode, PerformanceMode::StreamingExperts);
        assert!(
            plan.provenance
                .iter()
                .any(|l| l.contains("unified pool: host view"))
        );
        assert!(
            plan.provenance
                .iter()
                .any(|l| l.contains("device free view"))
        );
    }

    #[test]
    fn unmeasurable_unified_pool_errors() {
        let sizes = synthetic_sizes();
        let hw = Hardware {
            total_vram_bytes: Some(16 << 30),
            free_vram_bytes: Some(15 << 30),
            unified_host_device: Some(true),
            host_memory_available_bytes: None,
            unified_probe_failed: false,
        };
        let error = plan_mode(&hw, &ModeOverrides::default(), &sizes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("host_memory_available_bytes") && error.contains("ff probe"));
    }

    #[test]
    fn failed_topology_probe_errors() {
        let sizes = synthetic_sizes();
        // A failed probe cannot rule out a shared pool: planning as discrete
        // would reintroduce the very blind spot this closed (review finding,
        // mirroring ff-core's admission consumers).
        let hw = Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: None,
            host_memory_available_bytes: None,
            unified_probe_failed: true,
        };
        let error = plan_mode(&hw, &ModeOverrides::default(), &sizes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unified_host_device") && error.contains("ff probe"));
    }

    #[test]
    fn unified_pool_without_a_device_free_view_fails_closed() {
        let sizes = synthetic_sizes();
        // Symmetric with the missing-host-view case: both availability views
        // are binding for a shared pool (the ff-core contract).
        let hw = Hardware {
            total_vram_bytes: Some(16 << 30),
            free_vram_bytes: None,
            unified_host_device: Some(true),
            host_memory_available_bytes: Some(8 << 30),
            unified_probe_failed: false,
        };
        let error = plan_mode(&hw, &ModeOverrides::default(), &sizes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("free_vram_bytes") && error.contains("ff probe"));
    }

    #[test]
    fn unprobed_unified_topology_errors() {
        let sizes = synthetic_sizes();
        let hw = Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: None,
            host_memory_available_bytes: None,
            unified_probe_failed: false,
        };
        let error = plan_mode(&hw, &ModeOverrides::default(), &sizes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unified_host_device") && error.contains("ff probe"));
    }

    #[test]
    fn unified_pool_with_a_user_budget_caps_to_the_measured_pool() {
        let sizes = synthetic_sizes();
        let hw = Hardware {
            total_vram_bytes: Some(6 << 30),
            free_vram_bytes: Some(4 << 30),
            unified_host_device: Some(true),
            host_memory_available_bytes: Some(4 << 30),
            unified_probe_failed: false,
        };
        let plan = plan_mode(
            &hw,
            &ModeOverrides {
                vram_budget_bytes: Some(6 << 30),
                force_host_only: false,
            },
            &sizes,
        )
        .unwrap();
        // The user asked for the whole 6 GiB total; the shared pool only
        // measures 4 GiB free, and no reserve is charged (operator contract).
        assert_eq!(plan.budget_bytes, 4 << 30);
    }

    #[test]
    fn no_vram_probe_is_host_only_with_provenance() {
        let sizes = synthetic_sizes();
        let plan = plan_mode(
            &Hardware {
                total_vram_bytes: None,
                free_vram_bytes: None,
                unified_host_device: None,
                host_memory_available_bytes: None,
                unified_probe_failed: false,
            },
            &ModeOverrides::default(),
            &sizes,
        )
        .unwrap();
        assert_eq!(plan.mode, PerformanceMode::HostOnly);
        assert!(
            plan.provenance
                .iter()
                .any(|l| l.contains("no VRAM reported"))
        );
    }
}
