//! Performance-mode planner: senses hardware, applies user configuration,
//! and derives the best residency mode — re-deriving whenever configuration
//! changes (e.g. a VRAM budget unset defaults to all VRAM; setting one
//! re-runs the whole derivation). Every decision carries a provenance line
//! so a run's mode is attributable after the fact.

use anyhow::Result;
use ff_core::probe::admission_reserve_bytes;

/// Scale/bias bytes as a fraction of packed bytes — geometry-fixed by
/// group_size 64 and the BF16 scales+biases pair, and WEIGHED exactly
/// 12.5% from the checkpoint index (2026-09-17 audit). Single source for
/// every traffic and residency budget in this crate.
pub const SCALE_BIAS_OVERHEAD: f64 = 0.125;

#[derive(Clone, Debug, PartialEq)]
pub struct Hardware {
    pub total_vram_bytes: Option<u64>,
    pub free_vram_bytes: Option<u64>,
    /// From the unified-memory three-tier probe; None = unprobed.
    pub unified_host_device: Option<bool>,
    pub host_memory_available_bytes: Option<u64>,
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
        provenance.push("mode: HOST-ONLY — no VRAM reported by probe".into());
        return Ok(ModePlan {
            mode: PerformanceMode::HostOnly,
            budget_bytes: 0,
            expert_bytes_resident: 0,
            provenance,
        });
    };

    let reserve = admission_reserve_bytes(Some(total));

    // Unified pools share one physical memory with the host, so the budget
    // comes from the measured pool — the smaller of the two availability
    // views — not from the device total alone (which on such devices IS the
    // host's memory too). A confirmed-unified pool whose host view is missing
    // cannot be budgeted honestly: fail closed to host-only rather than plan
    // against a pool that may already be spent.
    let unified_pool: Option<u64> = match hardware.unified_host_device {
        None | Some(false) => None,
        Some(true) => match hardware.host_memory_available_bytes {
            Some(host_available) => {
                let pool = host_available.min(hardware.free_vram_bytes.unwrap_or(host_available));
                let device_view = hardware
                    .free_vram_bytes
                    .map(|free| format!("{free} B"))
                    .unwrap_or_else(|| "unmeasured".into());
                provenance.push(format!(
                    "unified pool: host view {host_available} B, device free view {device_view} → pool {pool} B"
                ));
                Some(pool)
            }
            None => {
                provenance.push(
                    "mode: HOST-ONLY — unified pool of unknown size (host availability unmeasured)"
                        .into(),
                );
                return Ok(ModePlan {
                    mode: PerformanceMode::HostOnly,
                    budget_bytes: 0,
                    expert_bytes_resident: 0,
                    provenance,
                });
            }
        },
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
            let b = unified_pool.unwrap_or(total).saturating_sub(reserve);
            if let Some(free) = hardware.free_vram_bytes {
                provenance.push(format!("free VRAM at probe: {free} B"));
            }
            let source = unified_pool.unwrap_or(total);
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
    use std::path::Path;

    fn sizes() -> WeightSizes {
        let weights = Edge0Weights::open(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/Edge0/Edge0-35B-A3B-preview"
        )))
        .unwrap();
        WeightSizes::from_weights(&weights).unwrap()
    }

    fn hw24g() -> Hardware {
        Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(60 << 30),
        }
    }

    #[test]
    fn weighed_sizes_match_the_index_audit() {
        let s = sizes();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        assert!((gib(s.expert_packed + s.expert_scale_bias) - 16.88).abs() < 0.05);
        assert!((s.expert_scale_bias as f64 / s.expert_packed as f64 - 0.125).abs() < 0.001);
        assert!((gib(s.static_packed + s.static_scale_bias) - 0.76).abs() < 0.05);
        assert!((gib(s.embed_lm_head) - 0.53).abs() < 0.05);
    }

    #[test]
    fn unset_budget_defaults_to_all_vram_and_fits_full_resident() {
        let plan = plan_mode(&hw24g(), &ModeOverrides::default(), &sizes()).unwrap();
        assert_eq!(plan.mode, PerformanceMode::FullResident);
        assert!(plan.provenance.iter().any(|l| l.contains("scaled reserve")));
    }

    #[test]
    fn user_budget_rederives_to_streaming() {
        let plan = plan_mode(
            &hw24g(),
            &ModeOverrides {
                vram_budget_bytes: Some(18 << 30),
                force_host_only: false,
            },
            &sizes(),
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
        let hw = Hardware {
            total_vram_bytes: Some(8 << 30),
            free_vram_bytes: Some(7 << 30),
            unified_host_device: Some(false),
            host_memory_available_bytes: Some(16 << 30),
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes()).unwrap();
        assert_eq!(plan.budget_bytes, (8 << 30) - (512 << 20));
        assert_eq!(plan.mode, PerformanceMode::StreamingExperts);
    }

    #[test]
    fn tight_budget_falls_to_host_only() {
        let plan = plan_mode(
            &hw24g(),
            &ModeOverrides {
                vram_budget_bytes: Some(1 << 30),
                force_host_only: false,
            },
            &sizes(),
        )
        .unwrap();
        assert_eq!(plan.mode, PerformanceMode::HostOnly);
    }

    #[test]
    fn unified_pool_budgets_against_measured_availability() {
        let hw = Hardware {
            total_vram_bytes: Some(6 << 30),
            free_vram_bytes: Some(7 << 29), // 3.5 GiB
            unified_host_device: Some(true),
            host_memory_available_bytes: Some(4 << 30),
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes()).unwrap();
        let pool = 7 << 29; // min(4 GiB host view, 3.5 GiB device free view)
        let reserve = (6 << 30) / 20;
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
    fn unmeasurable_unified_pool_fails_closed_to_host_only() {
        let hw = Hardware {
            total_vram_bytes: Some(16 << 30),
            free_vram_bytes: Some(15 << 30),
            unified_host_device: Some(true),
            host_memory_available_bytes: None,
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes()).unwrap();
        assert_eq!(plan.mode, PerformanceMode::HostOnly);
        assert_eq!(plan.budget_bytes, 0);
        assert!(
            plan.provenance
                .iter()
                .any(|l| l.contains("unified pool of unknown size"))
        );
    }

    #[test]
    fn unprobed_unified_topology_keeps_the_discrete_plan() {
        // Today's production values on a discrete card: the integrated probe
        // failed or was skipped, so legacy split-axis planning must hold.
        let hw = Hardware {
            total_vram_bytes: Some(24 << 30),
            free_vram_bytes: Some(23 << 30),
            unified_host_device: None,
            host_memory_available_bytes: None,
        };
        let plan = plan_mode(&hw, &ModeOverrides::default(), &sizes()).unwrap();
        assert_eq!(plan.mode, PerformanceMode::FullResident);
        assert_eq!(plan.budget_bytes, (24 << 30) - (1 << 30));
        assert!(!plan.provenance.iter().any(|l| l.contains("unified pool")));
    }

    #[test]
    fn unified_pool_with_a_user_budget_caps_to_the_measured_pool() {
        let hw = Hardware {
            total_vram_bytes: Some(6 << 30),
            free_vram_bytes: Some(4 << 30),
            unified_host_device: Some(true),
            host_memory_available_bytes: Some(4 << 30),
        };
        let plan = plan_mode(
            &hw,
            &ModeOverrides {
                vram_budget_bytes: Some(6 << 30),
                force_host_only: false,
            },
            &sizes(),
        )
        .unwrap();
        // The user asked for the whole 6 GiB total; the shared pool only
        // measures 4 GiB free, and no reserve is charged (operator contract).
        assert_eq!(plan.budget_bytes, 4 << 30);
    }

    #[test]
    fn no_vram_probe_is_host_only_with_provenance() {
        let plan = plan_mode(
            &Hardware {
                total_vram_bytes: None,
                free_vram_bytes: None,
                unified_host_device: None,
                host_memory_available_bytes: None,
            },
            &ModeOverrides::default(),
            &sizes(),
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
