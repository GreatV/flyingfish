use crate::{
    probe::ResourceSnapshot,
    resource_selection::{
        CandidateDisposition, ResourceCandidateObservation, ResourcePhaseEstimate,
        SelectedResourceAxis, SelectionOrigin,
    },
    weights::ModelWeights,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

pub fn materialize(
    weights: &ModelWeights,
    names: &[&str],
    device: &Device,
) -> Result<BTreeMap<String, Tensor>> {
    let mut tensors = BTreeMap::new();
    for &name in names {
        tensors.insert(name.to_owned(), weights.load(name, device)?);
    }
    Ok(tensors)
}

pub fn with_tensors<T>(
    weights: &ModelWeights,
    names: &[&str],
    device: &Device,
    f: impl FnOnce(&BTreeMap<String, Tensor>) -> Result<T>,
) -> Result<T> {
    f(&materialize(weights, names, device)?)
}

/// A maximal interval of execution over which the set of tensors read does not
/// change. Adapters declare their phases; the residency ladder places each one
/// independently.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightPhase {
    pub name: String,
    /// Every tensor the phase reads, as named by the checkpoint index.
    pub tensors: BTreeSet<String>,
    /// How many times one request reads each of the phase's tensors. This is
    /// the reuse term of the density ranking.
    pub reuse_count: u64,
}

impl WeightPhase {
    pub fn new(
        name: impl Into<String>,
        tensors: impl IntoIterator<Item = impl Into<String>>,
        reuse_count: u64,
    ) -> Self {
        Self {
            name: name.into(),
            tensors: tensors.into_iter().map(Into::into).collect(),
            reuse_count,
        }
    }

    /// Bytes the phase's tensors would occupy on `device` once materialized.
    /// Fails when a declared tensor is absent from the checkpoint, so a stale
    /// declaration surfaces at planning time rather than at first read.
    pub fn device_bytes(&self, weights: &ModelWeights, device: &Device) -> Result<u64> {
        let mut total = 0u64;
        for name in &self.tensors {
            total = total
                .checked_add(weights.produced_bytes(name, device)?)
                .context("phase working set exceeds u64 bytes")?;
        }
        Ok(total)
    }
}

/// How the tensors a phase actually read differ from its declaration. Both
/// directions are reported: reading an undeclared tensor means the ladder
/// cannot have placed it, and declaring an unread tensor overstates the
/// working set.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseReadDiff {
    /// Read during the phase but absent from the declaration.
    pub undeclared_reads: BTreeSet<String>,
    /// Declared but never read during the phase.
    pub unread_declarations: BTreeSet<String>,
}

impl PhaseReadDiff {
    pub fn is_exact(&self) -> bool {
        self.undeclared_reads.is_empty() && self.unread_declarations.is_empty()
    }
}

/// Diff a phase declaration against the reads the loader counted over the
/// phase's interval. `reads` is the delta of `ModelWeights::tensor_reads()`
/// across the interval, which the caller snapshots before and after.
pub fn verify_phase_reads(phase: &WeightPhase, reads: &BTreeMap<String, u64>) -> PhaseReadDiff {
    let read: BTreeSet<&String> = reads.keys().collect();
    let declared: BTreeSet<&String> = phase.tensors.iter().collect();
    PhaseReadDiff {
        undeclared_reads: read
            .difference(&declared)
            .map(|name| (*name).clone())
            .collect(),
        unread_declarations: declared
            .difference(&read)
            .map(|name| (*name).clone())
            .collect(),
    }
}

/// The outcome of asking the device cache to hold a phase. Placement is a
/// runtime fallback, never a precondition: tensors in `spilled` will be
/// streamed from the host tiers when the phase runs, exactly as they are
/// without any cache.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PhaseRetention {
    pub retained: BTreeSet<String>,
    pub spilled: BTreeSet<String>,
}

/// Pre-warm the device cache with a phase's tensors. A disabled cache retains
/// nothing; a placement failure (for example device-memory exhaustion after
/// the cache has already evicted everything it holds) spills that tensor
/// instead of failing the request.
pub fn retain_phase_on_device(
    weights: &ModelWeights,
    phase: &WeightPhase,
    device: &Device,
) -> PhaseRetention {
    let mut outcome = PhaseRetention::default();
    if !weights.device_cache_policy().is_enabled() {
        outcome.spilled = phase.tensors.clone();
        return outcome;
    }
    for name in &phase.tensors {
        let _ = weights.load(name, device);
    }
    for name in &phase.tensors {
        match weights.device_resident(name, device) {
            Ok(true) => {
                outcome.retained.insert(name.clone());
            }
            _ => {
                outcome.spilled.insert(name.clone());
            }
        }
    }
    outcome
}

/// How a phase's working set behaves when one request is split across ranks.
/// A phase is the unit of placement, so it is also the unit this question is
/// asked of.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PhasePartition {
    /// Every rank holds the whole phase. This is what a rank count of one
    /// always means, and what the norms, AdaLN parameters and rotary tables of
    /// a Megatron-shaped split mean at any rank count.
    #[default]
    Replicated,
    /// The phase's tensors are split across ranks, so one rank holds its share:
    /// the QKV, attention-output, and feed-forward projections.
    Sharded,
}

/// One phase's claim on the device tier: `value = (transfer_bytes +
/// transform_cost) × reuse_count`, ranked by `density = value /
/// resident_bytes`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhaseResidencyDemand {
    pub name: String,
    /// Bytes the phase's tensors occupy once materialized on the device.
    pub resident_bytes: u64,
    /// Bytes one read of the phase transfers up the ladder without residency.
    pub transfer_bytes: u64,
    /// Per-read transform cost, in the same units as `transfer_bytes`.
    pub transform_cost: u64,
    /// Reads of the phase per request; the reuse term of the ranking.
    pub reuse_count: u64,
    /// Half-open lifetime on a caller-declared execution timeline. None means
    /// retained for the whole request. Only declare disjoint intervals when
    /// the runtime releases the earlier store before opening the next one.
    pub lifetime: Option<std::ops::Range<u64>>,
    /// Whether a rank holds this phase whole or holds a share of it. Ignored
    /// at one rank, which is why declaring it changes no existing behaviour.
    pub partition: PhasePartition,
}

impl PhaseResidencyDemand {
    /// Derive a demand from a declared phase: resident bytes are the
    /// post-promotion device bytes, transfer bytes the checkpoint's raw bytes.
    /// A stale declaration fails here, at planning time, not at first read.
    pub fn from_phase(
        phase: &WeightPhase,
        weights: &ModelWeights,
        device: &Device,
    ) -> Result<Self> {
        let mut resident_bytes = 0u64;
        let mut transfer_bytes = 0u64;
        for name in &phase.tensors {
            resident_bytes = resident_bytes
                .checked_add(weights.produced_bytes(name, device)?)
                .context("phase working set exceeds u64 bytes")?;
            transfer_bytes = transfer_bytes
                .checked_add(u64::try_from(weights.raw_tensor_metadata(name)?.bytes)?)
                .context("phase transfer bytes exceed u64")?;
        }
        Ok(Self {
            name: phase.name.clone(),
            resident_bytes,
            transfer_bytes,
            transform_cost: 0,
            reuse_count: phase.reuse_count,
            lifetime: None,
            partition: PhasePartition::Replicated,
        })
    }

    /// This demand as one rank of `ranks` sees it. A replicated phase is
    /// returned unchanged; a sharded one divides its resident and transfer
    /// bytes, rounding *up* so the ranks' shares cover the whole phase and no
    /// rank is planned against a working set smaller than the one it will
    /// actually hold. Reuse and transform cost are per read and do not divide.
    ///
    /// At `ranks == 1` this is the identity for every phase, sharded or not,
    /// which is what keeps a single-device request on exactly the plan it had.
    pub fn per_rank(&self, ranks: NonZeroUsize) -> Result<Self> {
        let ranks = u64::try_from(ranks.get()).context("rank count exceeds u64")?;
        if ranks == 1 || self.partition == PhasePartition::Replicated {
            return Ok(self.clone());
        }
        Ok(Self {
            resident_bytes: self.resident_bytes.div_ceil(ranks),
            transfer_bytes: self.transfer_bytes.div_ceil(ranks),
            ..self.clone()
        })
    }

    /// `(transfer_bytes + transform_cost) × reuse_count`, checked.
    pub fn value(&self) -> Result<u128> {
        let per_read = self
            .transfer_bytes
            .checked_add(self.transform_cost)
            .context("residency demand per-read cost overflows u64")?;
        Ok(u128::from(per_read) * u128::from(self.reuse_count))
    }
}

/// Compare two densities `value / resident_bytes` without leaving integer
/// arithmetic. Cross-multiplication is exact for any demand a real device can
/// express; a product that would exceed u128 falls back to truncated division,
/// which still orders anything that large correctly enough to fill a cache. A
/// zero-resident demand has infinite density when its value is positive.
fn density_ordering(
    a_value: u128,
    a_resident: u64,
    b_value: u128,
    b_resident: u64,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a_resident, b_resident) {
        (0, 0) => return a_value.cmp(&b_value),
        (0, _) => {
            return if a_value > 0 {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        (_, 0) => {
            return if b_value > 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        _ => {}
    }
    match a_value
        .checked_mul(u128::from(b_resident))
        .zip(b_value.checked_mul(u128::from(a_resident)))
    {
        Some((left, right)) => left.cmp(&right),
        None => (a_value / u128::from(a_resident)).cmp(&(b_value / u128::from(b_resident))),
    }
}

/// What the density fill decided for a budget, before any authorization is
/// applied. `placed` and `spilled` are in density order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeviceResidencyPlan {
    pub budget_bytes: u64,
    pub placed: Vec<String>,
    pub spilled: Vec<String>,
    /// Peak retained weight bytes across the selected phase lifetimes.
    pub resident_bytes: u64,
}

fn residency_peak(
    demands: &[PhaseResidencyDemand],
    selected: impl Iterator<Item = usize>,
) -> Option<u64> {
    let mut active = 0u64;
    let mut events = Vec::new();
    for index in selected {
        let demand = &demands[index];
        if let Some(lifetime) = &demand.lifetime {
            events.push((lifetime.start, true, demand.resident_bytes));
            events.push((lifetime.end, false, demand.resident_bytes));
        } else {
            active = active.checked_add(demand.resident_bytes)?;
        }
    }
    events.sort_unstable_by_key(|&(time, start, _)| (time, start));
    let mut peak = active;
    for (_, start, bytes) in events {
        active = if start {
            active.checked_add(bytes)?
        } else {
            active.checked_sub(bytes)?
        };
        peak = peak.max(active);
    }
    Some(peak)
}

/// Fill the device tier greedily by descending reuse density. The unit of
/// placement is the phase: a phase is placed whole or spilled whole, and a
/// phase that does not fit never blocks a smaller, less dense one behind it.
/// Equal densities are broken by phase name so the fill is deterministic.
/// `capacity_bytes` is what the tier can hold; `reserve_bytes` stays free of
/// the cache. Explicitly disjoint lifetimes reuse capacity; demands without
/// lifetimes remain concurrent.
pub fn plan_device_residency(
    demands: &[PhaseResidencyDemand],
    capacity_bytes: u64,
    reserve_bytes: u64,
) -> Result<DeviceResidencyPlan> {
    let mut names = BTreeSet::new();
    let mut values = Vec::with_capacity(demands.len());
    for demand in demands {
        if let Some(lifetime) = &demand.lifetime {
            anyhow::ensure!(
                lifetime.start < lifetime.end,
                "invalid residency lifetime for phase {}",
                demand.name
            );
        }
        anyhow::ensure!(
            names.insert(&demand.name),
            "duplicate residency demand for phase {}",
            demand.name
        );
        values.push(demand.value()?);
    }
    let mut order: Vec<usize> = (0..demands.len()).collect();
    order.sort_by(|&a, &b| {
        density_ordering(
            values[a],
            demands[a].resident_bytes,
            values[b],
            demands[b].resident_bytes,
        )
        .reverse()
        .then_with(|| demands[a].name.cmp(&demands[b].name))
    });

    let budget = capacity_bytes.saturating_sub(reserve_bytes);
    let mut plan = DeviceResidencyPlan {
        budget_bytes: budget,
        ..Default::default()
    };
    let mut selected = Vec::new();
    for index in order {
        let demand = &demands[index];
        let peak = residency_peak(
            demands,
            selected.iter().copied().chain(std::iter::once(index)),
        );
        if let Some(peak) = peak.filter(|&peak| peak <= budget) {
            plan.resident_bytes = peak;
            selected.push(index);
            plan.placed.push(demand.name.clone());
        } else {
            plan.spilled.push(demand.name.clone());
        }
    }
    Ok(plan)
}

/// Whether the ladder may place anything at all. Retention is gated on
/// qualified paired evidence: without it, or without an explicit operator
/// setting, the streaming baseline stays selected no matter what the density
/// fill finds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidencyAuthorization {
    NotAuthorized,
    /// The operator named a residency ceiling (`--device-cache-mib`).
    OperatorExplicit {
        ceiling_bytes: u64,
    },
    /// Qualified paired evidence per the charter's rules, and the ceiling that
    /// evidence supports.
    MeasuredEvidence {
        ceiling_bytes: u64,
    },
}

impl ResidencyAuthorization {
    /// The ceiling this authorization carries, or `None` when it authorizes
    /// nothing. A ceiling bounds the fill itself, not just the number recorded
    /// afterwards: clamping only the record would publish a plan naming phases
    /// the cache was never allowed to hold.
    pub const fn ceiling_bytes(self) -> Option<u64> {
        match self {
            Self::NotAuthorized => None,
            Self::OperatorExplicit { ceiling_bytes } | Self::MeasuredEvidence { ceiling_bytes } => {
                Some(ceiling_bytes)
            }
        }
    }
}

/// The ladder's decision: the density fill, the budget it was authorized to
/// spend, and the fragments a caller merges into a
/// `ResourceSelectionProvenance` — one axis, the two candidates (exactly one
/// selected), and one phase charge per demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceResidencyDecision {
    pub plan: DeviceResidencyPlan,
    pub authorized_budget_bytes: u64,
    pub axes: Vec<SelectedResourceAxis>,
    pub candidates: Vec<ResourceCandidateObservation>,
    pub phase_charges: Vec<ResourcePhaseEstimate>,
}

/// Decide device residency from a captured snapshot. Capacity is the
/// snapshot's free device memory (a host-only capture places nothing); the
/// reserve stays free of the cache. The fill is always computed and recorded,
/// but only an authorized decision selects it — `NotAuthorized` records the
/// streaming baseline as selected and charges no device bytes. There is no
/// automatic enablement without qualified paired evidence.
pub fn decide_device_residency(
    snapshot: &ResourceSnapshot,
    demands: &[PhaseResidencyDemand],
    reserve_bytes: u64,
    authorization: ResidencyAuthorization,
) -> Result<DeviceResidencyDecision> {
    // On a probed unified-memory device, host allocations draw from the same
    // pool, so the binding capacity is the smaller pool view rather than the
    // device view alone. Discrete and unprobed captures use the device view
    // exactly as before.
    let capacity = snapshot
        .unified_pool_available_bytes()
        .or(snapshot.device_free_memory_bytes)
        .unwrap_or(0);
    decide_residency_within(capacity, demands, reserve_bytes, authorization)
}

/// One device's capacity, as probed for that device. `ResourceSnapshot` reports
/// a scalar `device_free_memory_bytes`, so a request spread over several
/// devices captures one snapshot per device rather than reusing one.
pub fn probe_rank_capacities(devices: &[Device]) -> Vec<u64> {
    devices
        .iter()
        .map(|device| {
            ResourceSnapshot::capture(Some(device))
                .device_free_memory_bytes
                .unwrap_or(0)
        })
        .collect()
}

/// What the ladder decided for one rank: the capacity its own device reported,
/// and the fill planned against that rank's share of every sharded phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankResidencyDecision {
    pub rank: u32,
    pub capacity_bytes: u64,
    /// The demands as this rank sees them, in the order they were given.
    pub demands: Vec<PhaseResidencyDemand>,
    pub decision: DeviceResidencyDecision,
}

/// Decide device residency for a request spread across ranks: one decision per
/// rank, each against that rank's own measured capacity and that rank's own
/// share of every sharded phase. `capacities` is in rank order, one entry per
/// rank, and its length is the rank count.
///
/// A one-rank call is `decide_device_residency` against that one capacity —
/// same demands, same fill, same record — which is the "no behaviour change at
/// N=1" half of T1's gate. The reserve and the authorized ceiling apply to each
/// rank, because both describe one device: a ceiling divided among ranks would
/// be the defect this rewrite exists to avoid.
pub fn decide_rank_residency(
    capacities: &[u64],
    demands: &[PhaseResidencyDemand],
    reserve_bytes: u64,
    authorization: ResidencyAuthorization,
) -> Result<Vec<RankResidencyDecision>> {
    let ranks = NonZeroUsize::new(capacities.len())
        .context("a rank-aware residency plan needs at least one rank capacity")?;
    capacities
        .iter()
        .enumerate()
        .map(|(rank, capacity)| {
            let demands = demands
                .iter()
                .map(|demand| demand.per_rank(ranks))
                .collect::<Result<Vec<_>>>()?;
            Ok(RankResidencyDecision {
                rank: u32::try_from(rank).context("rank index exceeds u32")?,
                capacity_bytes: *capacity,
                decision: decide_residency_within(
                    *capacity,
                    &demands,
                    reserve_bytes,
                    authorization,
                )?,
                demands,
            })
        })
        .collect()
}

fn decide_residency_within(
    capacity: u64,
    demands: &[PhaseResidencyDemand],
    reserve_bytes: u64,
    authorization: ResidencyAuthorization,
) -> Result<DeviceResidencyDecision> {
    let capacity = match authorization.ceiling_bytes() {
        None => capacity,
        Some(ceiling) => capacity.min(ceiling.saturating_add(reserve_bytes)),
    };
    let plan = plan_device_residency(demands, capacity, reserve_bytes)?;
    let authorized = match authorization.ceiling_bytes() {
        None => 0,
        Some(_) => plan.budget_bytes,
    };
    let active = authorized > 0 && !plan.placed.is_empty();

    let origin = match authorization {
        ResidencyAuthorization::NotAuthorized => SelectionOrigin::Baseline,
        ResidencyAuthorization::OperatorExplicit { .. } => SelectionOrigin::OperatorExplicit,
        ResidencyAuthorization::MeasuredEvidence { .. } => SelectionOrigin::MeasuredEvidence,
    };
    let axes = vec![SelectedResourceAxis {
        axis: "weights.device_cache_bytes".into(),
        value: authorized.to_string(),
        origin,
    }];

    let placed: BTreeSet<&str> = plan.placed.iter().map(String::as_str).collect();
    let phase_charges = demands
        .iter()
        .map(|demand| ResourcePhaseEstimate {
            phase: demand.name.clone(),
            required_host_bytes: 0,
            optional_host_bytes: 0,
            reclaimable_host_bytes: 0,
            host_promotion_reserve_bytes: 0,
            required_device_bytes: Some(0),
            optional_device_bytes: Some(if active && placed.contains(demand.name.as_str()) {
                demand.resident_bytes
            } else {
                0
            }),
            device_reserve_bytes: if active { reserve_bytes } else { 0 },
        })
        .collect();

    let residency = ResourceCandidateObservation {
        candidate_id: "device-residency".into(),
        disposition: if active {
            CandidateDisposition::Selected
        } else if authorization.ceiling_bytes().is_none() {
            CandidateDisposition::NoBenefitEvidence
        } else {
            CandidateDisposition::CapacityRejected
        },
        reason: match (authorization, active) {
            (ResidencyAuthorization::NotAuthorized, _) => {
                "no qualified paired evidence authorizes device residency".into()
            }
            (_, true) => format!(
                "density fill places {} phases ({} peak bytes) within {} authorized bytes",
                plan.placed.len(),
                plan.resident_bytes,
                authorized
            ),
            (_, false) => "device headroom holds no declared phase after the reserve".into(),
        },
        expected_cost: None,
        evidence: vec![],
    };
    let baseline = ResourceCandidateObservation {
        candidate_id: "streaming-baseline".into(),
        disposition: if active {
            match authorization {
                ResidencyAuthorization::OperatorExplicit { .. } => {
                    CandidateDisposition::OperatorExcluded
                }
                _ => CandidateDisposition::Slower,
            }
        } else {
            CandidateDisposition::Selected
        },
        reason: if active {
            "device residency supersedes streaming for the placed phases".into()
        } else {
            "baseline_no_benefit_evidence".into()
        },
        expected_cost: None,
        evidence: vec![],
    };

    Ok(DeviceResidencyDecision {
        plan,
        authorized_budget_bytes: authorized,
        axes,
        candidates: vec![baseline, residency],
        phase_charges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{CachePolicy, WeightSource};
    use candle_core::{DType, Device, Tensor, safetensors};
    use serde_json::json;
    use std::{collections::BTreeMap as Map, collections::HashMap, fs};

    fn open_named(names: &[&str]) -> (tempfile::TempDir, ModelWeights) {
        let dir = tempfile::tempdir().unwrap();
        let tensors = names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                (
                    (*name).to_owned(),
                    Tensor::new(&[index as f32], &Device::Cpu).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        safetensors::save(&tensors, dir.path().join("weights.safetensors")).unwrap();
        let weight_map = names
            .iter()
            .map(|name| (*name, "weights.safetensors"))
            .collect::<Map<_, _>>();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": names.len() * 4},
                "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        (dir, weights)
    }

    #[test]
    fn materializes_an_explicit_name_set_without_an_h3_plan() {
        let (_dir, weights) = open_named(&["router.weight", "experts.0.w1"]);
        let loaded = materialize(&weights, &["experts.0.w1"], &Device::Cpu).unwrap();
        assert_eq!(loaded.len(), 1);
        let tensor = loaded.get("experts.0.w1").unwrap();
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.to_vec1::<f32>().unwrap(), vec![1.0]);
        let doubled = with_tensors(&weights, &["router.weight"], &Device::Cpu, |tensors| {
            assert_eq!(tensors.len(), 1);
            tensors["router.weight"].affine(2., 0.).map_err(Into::into)
        })
        .unwrap();
        assert_eq!(doubled.to_vec1::<f32>().unwrap(), vec![0.0]);
    }

    #[test]
    fn phase_declaration_diffs_against_counted_reads() {
        let (_dir, weights) = open_named(&["a", "b", "c"]);
        weights.count_tensor_reads(true);
        let phase = WeightPhase::new("decode", ["a", "b"], 7);
        assert_eq!(phase.device_bytes(&weights, &Device::Cpu).unwrap(), 8);
        weights.load("a", &Device::Cpu).unwrap();
        weights.load("c", &Device::Cpu).unwrap();
        let diff = verify_phase_reads(&phase, &weights.tensor_reads());
        assert!(!diff.is_exact());
        assert_eq!(diff.undeclared_reads, BTreeSet::from(["c".to_owned()]));
        assert_eq!(diff.unread_declarations, BTreeSet::from(["b".to_owned()]));
        weights.load("b", &Device::Cpu).unwrap();
        let exact = WeightPhase::new("decode", ["a", "b", "c"], 1);
        assert!(verify_phase_reads(&exact, &weights.tensor_reads()).is_exact());
    }

    #[test]
    fn phase_device_bytes_reject_an_unknown_tensor() {
        let (_dir, weights) = open_named(&["a"]);
        let phase = WeightPhase::new("broken", ["a", "missing"], 1);
        let error = phase
            .device_bytes(&weights, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing"), "{error}");
    }

    #[test]
    fn retain_phase_never_fails_the_request() {
        use crate::weights::{DeviceCache, DeviceCachePolicy};
        let (_dir, mut weights) = open_named(&["a", "b"]);
        weights.count_tensor_reads(true);
        let phase = WeightPhase::new("decode", ["a", "b"], 4);
        let outcome = retain_phase_on_device(&weights, &phase, &Device::Cpu);
        assert!(outcome.retained.is_empty());
        assert_eq!(outcome.spilled, phase.tensors);
        assert!(weights.tensor_reads().is_empty());
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(1 << 20)))
            .unwrap();
        let outcome = retain_phase_on_device(&weights, &phase, &Device::Cpu);
        assert_eq!(outcome.retained, phase.tensors);
        assert!(outcome.spilled.is_empty());
        assert_eq!(weights.device_cache_stats().resident_bytes, 8);
        let broken = WeightPhase::new("broken", ["a", "missing"], 1);
        let outcome = retain_phase_on_device(&weights, &broken, &Device::Cpu);
        assert_eq!(outcome.retained, BTreeSet::from(["a".to_owned()]));
        assert_eq!(outcome.spilled, BTreeSet::from(["missing".to_owned()]));
    }

    fn demand(name: &str, resident: u64, transfer: u64, reuse: u64) -> PhaseResidencyDemand {
        PhaseResidencyDemand {
            name: name.into(),
            resident_bytes: resident,
            transfer_bytes: transfer,
            transform_cost: 0,
            reuse_count: reuse,
            lifetime: None,
            partition: PhasePartition::Replicated,
        }
    }

    #[test]
    fn sequential_phases_reuse_capacity_at_the_boundary() {
        let mut a = demand("a", 8, 8, 2);
        let mut b = demand("b", 8, 8, 2);
        assert_eq!(
            plan_device_residency(&[a.clone(), b.clone()], 8, 0)
                .unwrap()
                .placed,
            ["a"]
        );
        a.lifetime = Some(0..1);
        b.lifetime = Some(1..2);
        let plan = plan_device_residency(&[a, b], 8, 0).unwrap();
        assert_eq!(plan.placed, ["a", "b"]);
        assert!(plan.spilled.is_empty());
        assert_eq!(plan.resident_bytes, 8);
    }

    #[test]
    fn lifetime_peak_includes_persistent_and_overlapping_phases() {
        let persistent = demand("persistent", 3, 3, 100);
        let mut a = demand("a", 5, 5, 10);
        let mut b = demand("b", 5, 5, 9);
        let mut c = demand("c", 5, 5, 8);
        a.lifetime = Some(0..2);
        b.lifetime = Some(1..3);
        c.lifetime = Some(2..4);
        let plan = plan_device_residency(&[persistent, a, b, c], 10, 2).unwrap();
        assert_eq!(plan.placed, ["persistent", "a", "c"]);
        assert_eq!(plan.spilled, ["b"]);
        assert_eq!(plan.resident_bytes, 8);
    }

    #[test]
    fn lifetime_peak_handles_large_disjoint_allocations_and_rejects_empty_ranges() {
        let mut a = demand("a", u64::MAX, 1, 1);
        let mut b = demand("b", u64::MAX, 1, 1);
        a.lifetime = Some(0..1);
        b.lifetime = Some(1..2);
        let plan = plan_device_residency(&[a.clone(), b], u64::MAX, 0).unwrap();
        assert_eq!(plan.placed, ["a", "b"]);
        assert_eq!(plan.resident_bytes, u64::MAX);
        a.lifetime = Some(1..1);
        assert!(plan_device_residency(&[a], u64::MAX, 0).is_err());
    }

    #[test]
    fn density_fill_ranks_by_reuse_not_by_size() {
        let language = demand("music3.autoregressive", 9 << 20, 9 << 20, 210);
        let vocoder = demand("music3.vocode", 1 << 20, 1 << 20, 2);
        let plan = plan_device_residency(&[vocoder.clone(), language.clone()], 9 << 20, 0).unwrap();
        assert_eq!(plan.placed, ["music3.autoregressive"]);
        assert_eq!(plan.spilled, ["music3.vocode"]);
        assert_eq!(plan.resident_bytes, 9 << 20);
        let plan = plan_device_residency(&[vocoder, language], 16 << 20, 0).unwrap();
        assert_eq!(plan.placed, ["music3.autoregressive", "music3.vocode"]);
    }

    #[test]
    fn density_fill_skips_an_oversized_phase_and_keeps_filling() {
        let huge = demand("huge", 100, 100, 10);
        let small = demand("small", 10, 1, 1);
        let plan = plan_device_residency(&[huge, small], 50, 0).unwrap();
        assert_eq!(plan.placed, ["small"]);
        assert_eq!(plan.spilled, ["huge"]);
    }

    #[test]
    fn density_fill_is_deterministic_and_honors_the_reserve() {
        let a = demand("alpha", 10, 10, 2);
        let b = demand("bravo", 10, 10, 2);
        let plan = plan_device_residency(&[b.clone(), a.clone()], 20, 0).unwrap();
        assert_eq!(plan.placed, ["alpha", "bravo"]);
        let plan = plan_device_residency(&[a, b], 20, 0).unwrap();
        assert_eq!(plan.placed, ["alpha", "bravo"]);
        let plan = plan_device_residency(&[demand("alpha", 10, 10, 2)], 20, 11).unwrap();
        assert_eq!(plan.budget_bytes, 9);
        assert_eq!(plan.spilled, ["alpha"]);
        let plan = plan_device_residency(&[demand("alpha", 1, 1, 1)], 4, 10).unwrap();
        assert_eq!(plan.budget_bytes, 0);
        assert!(plan.placed.is_empty());
    }

    #[test]
    fn demand_values_are_checked_and_transform_cost_counts() {
        let mut d = demand("phase", 10, 5, 3);
        assert_eq!(d.value().unwrap(), 15);
        d.transform_cost = 5;
        assert_eq!(d.value().unwrap(), 30);
        d.transfer_bytes = u64::MAX;
        assert!(d.value().is_err());
        let plain = demand("plain", 10, 10, 1);
        let mut transformed = demand("transformed", 10, 10, 1);
        transformed.transform_cost = 1;
        let plan = plan_device_residency(&[plain, transformed], 10, 0).unwrap();
        assert_eq!(plan.placed, ["transformed"]);
        assert!(
            plan_device_residency(&[demand("dup", 1, 1, 1), demand("dup", 1, 1, 1)], 10, 0)
                .is_err()
        );
    }

    #[test]
    fn demand_from_phase_charges_post_promotion_residency_and_raw_transfer() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "half".to_owned(),
            Tensor::new(&[1.5f32, 2.5], &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        safetensors::save(&tensors, dir.path().join("weights.safetensors")).unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 4},
                "weight_map": {"half": "weights.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let weights =
            ModelWeights::open(dir.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
        let phase = WeightPhase::new("phase", ["half"], 7);
        let demand = PhaseResidencyDemand::from_phase(&phase, &weights, &Device::Cpu).unwrap();
        assert_eq!(demand.resident_bytes, 8);
        assert_eq!(demand.transfer_bytes, 4);
        assert_eq!(demand.reuse_count, 7);
        let stale = WeightPhase::new("stale", ["missing"], 1);
        assert!(PhaseResidencyDemand::from_phase(&stale, &weights, &Device::Cpu).is_err());
    }

    fn snapshot(device_free: Option<u64>) -> ResourceSnapshot {
        use crate::probe::ResourceMeasurementScopes;
        ResourceSnapshot {
            schema_version: 1,
            measured_at_unix_ms: 1,
            host_memory_available_bytes: None,
            cgroup_v2_memory_limit: None,
            cgroup_v2_memory_current_bytes: None,
            cgroup_v2_memory_available_bytes: None,
            device_free_memory_bytes: device_free,
            host_device_memory_is_unified: None,
            measurement_scope: ResourceMeasurementScopes {
                host_memory: None,
                cgroup_memory: None,
                device_memory: None,
            },
        }
    }

    fn provenance(
        decision: &DeviceResidencyDecision,
    ) -> crate::resource_selection::ResourceSelectionProvenance {
        use crate::{
            resource_selection::{RESOURCE_SELECTION_SCHEMA_VERSION, ResourceSelectionProvenance},
            weights::accounting::{CacheInventory, CacheShardInventory},
        };
        use serde_json::json;
        ResourceSelectionProvenance {
            schema_version: RESOURCE_SELECTION_SCHEMA_VERSION,
            policy: json!({"weight_source": "mmap"}),
            selector_revision: "weight-residency-v1".into(),
            request: json!({"prompt_tokens": 10}),
            input: None,
            model: json!({"component": "transformer"}),
            hardware: None,
            executable: None,
            environment: None,
            mode: crate::resource_selection::ResourcePolicyMode::Performance,
            selection_snapshot: snapshot(Some(1 << 20)),
            final_admission_snapshot: None,
            already_present_at_capture: vec![],
            inventory: CacheInventory {
                shards: vec![CacheShardInventory {
                    name: "a.safetensors".into(),
                    file_bytes: 108,
                    header_bytes: 8,
                    selected_tensor_bytes: 100,
                    selected_tensor_count: 1,
                    largest_tensor_bytes: 100,
                }],
            },
            phases: decision.phase_charges.clone(),
            workload: Default::default(),
            axes: decision.axes.clone(),
            candidates: decision.candidates.clone(),
        }
    }

    #[test]
    fn unauthorized_decision_keeps_the_streaming_baseline_selected() {
        let demands = vec![demand("decode", 100, 100, 50)];
        let decision = decide_device_residency(
            &snapshot(Some(1 << 20)),
            &demands,
            0,
            ResidencyAuthorization::NotAuthorized,
        )
        .unwrap();
        assert_eq!(decision.plan.placed, ["decode"]);
        assert_eq!(decision.authorized_budget_bytes, 0);
        assert_eq!(decision.axes[0].value, "0");
        assert_eq!(decision.axes[0].origin, SelectionOrigin::Baseline);
        let [baseline, residency] = &decision.candidates[..] else {
            panic!("two candidates")
        };
        assert_eq!(baseline.disposition, CandidateDisposition::Selected);
        assert_eq!(
            residency.disposition,
            CandidateDisposition::NoBenefitEvidence
        );
        assert_eq!(decision.phase_charges[0].optional_device_bytes, Some(0));
        assert_eq!(decision.phase_charges[0].device_reserve_bytes, 0);
        provenance(&decision).validate().unwrap();
    }

    #[test]
    fn authorized_decision_selects_the_fill_and_charges_each_phase() {
        let demands = vec![
            demand("music3.autoregressive", 90, 90, 210),
            demand("music3.vocode", 10, 10, 2),
        ];
        let decision = decide_device_residency(
            &snapshot(Some(100)),
            &demands,
            8,
            ResidencyAuthorization::MeasuredEvidence {
                ceiling_bytes: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(decision.authorized_budget_bytes, 92);
        assert_eq!(decision.plan.placed, ["music3.autoregressive"]);
        assert_eq!(decision.axes[0].value, "92");
        assert_eq!(decision.axes[0].origin, SelectionOrigin::MeasuredEvidence);
        let [baseline, residency] = &decision.candidates[..] else {
            panic!("two candidates")
        };
        assert_eq!(residency.disposition, CandidateDisposition::Selected);
        assert_eq!(baseline.disposition, CandidateDisposition::Slower);
        assert_eq!(decision.phase_charges[0].optional_device_bytes, Some(90));
        assert_eq!(decision.phase_charges[0].device_reserve_bytes, 8);
        assert_eq!(decision.phase_charges[1].optional_device_bytes, Some(0));
        assert_eq!(decision.phase_charges[1].device_reserve_bytes, 8);
        assert_eq!(
            decision.phase_charges[0].device_peak_bytes().unwrap(),
            Some(98)
        );
        provenance(&decision).validate().unwrap();
    }

    #[test]
    fn an_operator_ceiling_bounds_the_fill_not_just_the_record() {
        let demands = vec![demand("big", 32, 32, 100), demand("small", 16, 16, 1)];
        let decision = decide_device_residency(
            &snapshot(Some(64)),
            &demands,
            0,
            ResidencyAuthorization::OperatorExplicit { ceiling_bytes: 16 },
        )
        .unwrap();
        assert_eq!(decision.authorized_budget_bytes, 16);
        assert_eq!(decision.plan.placed, ["small"]);
        assert_eq!(decision.plan.spilled, ["big"]);
        assert_eq!(decision.axes[0].value, "16");
        provenance(&decision).validate().unwrap();

        let decision = decide_device_residency(
            &snapshot(Some(20)),
            &demands,
            8,
            ResidencyAuthorization::OperatorExplicit {
                ceiling_bytes: 1 << 30,
            },
        )
        .unwrap();
        assert_eq!(decision.authorized_budget_bytes, 12);
    }

    #[test]
    fn an_unauthorized_decision_still_records_the_fill_it_declined() {
        let demands = vec![demand("decode", 32, 32, 100)];
        let decision = decide_device_residency(
            &snapshot(Some(64)),
            &demands,
            0,
            ResidencyAuthorization::NotAuthorized,
        )
        .unwrap();
        assert_eq!(decision.authorized_budget_bytes, 0);
        assert_eq!(decision.plan.placed, ["decode"]);
        assert_eq!(decision.axes[0].value, "0");
        assert_eq!(decision.axes[0].origin, SelectionOrigin::Baseline);
        provenance(&decision).validate().unwrap();
    }

    #[test]
    fn retention_reports_what_the_cache_holds_not_what_loaded() {
        use crate::weights::{DeviceCache, DeviceCachePolicy};
        let (_dir, mut weights) = open_named(&["a", "b"]);
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(4)))
            .unwrap();
        let outcome = retain_phase_on_device(
            &weights,
            &WeightPhase::new("decode", ["a", "b"], 1),
            &Device::Cpu,
        );
        assert_eq!(outcome.retained.len(), 1);
        assert_eq!(outcome.spilled.len(), 1);
        assert_eq!(weights.device_cache_stats().resident_tensors, 1);

        let (_dir, mut weights) = open_named(&["a"]);
        weights
            .configure_device_cache(DeviceCache::new(DeviceCachePolicy::with_max_bytes(1)))
            .unwrap();
        let outcome = retain_phase_on_device(
            &weights,
            &WeightPhase::new("decode", ["a"], 1),
            &Device::Cpu,
        );
        assert!(outcome.retained.is_empty(), "{outcome:?}");
        assert_eq!(outcome.spilled, BTreeSet::from(["a".to_owned()]));
    }

    #[test]
    fn operator_explicit_decision_marks_the_baseline_excluded() {
        let demands = vec![demand("decode", 64, 64, 100)];
        let decision = decide_device_residency(
            &snapshot(Some(128)),
            &demands,
            0,
            ResidencyAuthorization::OperatorExplicit {
                ceiling_bytes: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(decision.axes[0].origin, SelectionOrigin::OperatorExplicit);
        let [baseline, residency] = &decision.candidates[..] else {
            panic!("two candidates")
        };
        assert_eq!(residency.disposition, CandidateDisposition::Selected);
        assert_eq!(baseline.disposition, CandidateDisposition::OperatorExcluded);
        provenance(&decision).validate().unwrap();
    }

    #[test]
    fn authorized_decision_without_headroom_is_capacity_rejected() {
        let demands = vec![demand("decode", 1 << 30, 1 << 30, 10)];
        let decision = decide_device_residency(
            &snapshot(None),
            &demands,
            0,
            ResidencyAuthorization::MeasuredEvidence {
                ceiling_bytes: u64::MAX,
            },
        )
        .unwrap();
        assert_eq!(decision.authorized_budget_bytes, 0);
        assert!(decision.plan.placed.is_empty());
        let [baseline, residency] = &decision.candidates[..] else {
            panic!("two candidates")
        };
        assert_eq!(baseline.disposition, CandidateDisposition::Selected);
        assert_eq!(
            residency.disposition,
            CandidateDisposition::CapacityRejected
        );
        assert_eq!(decision.phase_charges[0].optional_device_bytes, Some(0));
        provenance(&decision).validate().unwrap();
    }

    /// Capacity is probed for one device and the demand is a whole-phase
    /// quantity. Both become per-rank, and at one rank the result is the plan
    /// the single-device path makes.
    #[test]
    fn one_rank_plans_exactly_what_the_single_device_path_plans() {
        let mut sharded = demand("projections", 64, 64, 8);
        sharded.partition = PhasePartition::Sharded;
        let demands = [sharded, demand("norms", 8, 8, 8)];
        let snapshot = ResourceSnapshot {
            device_free_memory_bytes: Some(96),
            host_device_memory_is_unified: None,
            ..ResourceSnapshot::capture(None)
        };
        let authorization = ResidencyAuthorization::OperatorExplicit {
            ceiling_bytes: 1 << 20,
        };
        let single = decide_device_residency(&snapshot, &demands, 0, authorization).unwrap();
        let ranked = decide_rank_residency(&[96], &demands, 0, authorization).unwrap();
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].rank, 0);
        assert_eq!(ranked[0].capacity_bytes, 96);
        assert_eq!(ranked[0].decision, single);
        assert_eq!(ranked[0].demands[0].resident_bytes, 64);
    }

    #[test]
    fn a_sharded_phase_divides_per_rank_and_a_replicated_one_does_not() {
        let mut sharded = demand("projections", 64, 64, 8);
        sharded.partition = PhasePartition::Sharded;
        let replicated = demand("norms", 9, 9, 8);
        let ranks = NonZeroUsize::new(4).unwrap();
        let share = sharded.per_rank(ranks).unwrap();
        assert_eq!(share.resident_bytes, 16);
        assert_eq!(share.transfer_bytes, 16);
        assert_eq!(share.reuse_count, sharded.reuse_count);
        assert_eq!(replicated.per_rank(ranks).unwrap(), replicated);
        let mut odd = demand("odd", 10, 10, 1);
        odd.partition = PhasePartition::Sharded;
        let share = odd.per_rank(NonZeroUsize::new(4).unwrap()).unwrap();
        assert_eq!(share.resident_bytes, 3);
        assert!(share.resident_bytes * 4 >= odd.resident_bytes);
    }

    /// Capacity is per device: a rank on a smaller device places less, and it
    /// does so without changing what a larger rank places. This is the failure
    /// a single scalar `device_free_memory_bytes` could not express.
    #[test]
    fn each_rank_is_planned_against_its_own_device_capacity() {
        let mut projections = demand("projections", 64, 64, 8);
        projections.partition = PhasePartition::Sharded;
        let norms = demand("norms", 8, 8, 1);
        let decisions = decide_rank_residency(
            &[64, 16],
            &[projections, norms],
            0,
            ResidencyAuthorization::OperatorExplicit {
                ceiling_bytes: 1 << 20,
            },
        )
        .unwrap();
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].demands[0].resident_bytes, 32);
        assert_eq!(decisions[0].decision.plan.placed, ["projections", "norms"]);
        assert_eq!(decisions[1].capacity_bytes, 16);
        assert_eq!(decisions[1].decision.plan.placed, ["norms"]);
        assert_eq!(decisions[1].decision.plan.spilled, ["projections"]);
    }

    #[test]
    fn a_rank_aware_plan_needs_at_least_one_capacity() {
        let error = decide_rank_residency(
            &[],
            &[demand("a", 1, 1, 1)],
            0,
            ResidencyAuthorization::NotAuthorized,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("at least one rank capacity"), "{error}");
    }
}
