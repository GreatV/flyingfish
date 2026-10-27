//! Setup-derived expert residency capacity and per-request placement policy.
//! Per-layer slots are equal, with the remainder assigned to low layer ids.
use crate::experts::arena::SlotLayout;
use crate::experts::catalogue::ExpertGeometry;
use anyhow::{Context, Result, ensure};

/// Per-token admission-score decay.
pub const F_DECAY: f64 = 0.98;
/// Minimum decayed frequency for admission.
pub const THETA_MIN: f64 = 1.05;
/// Copies of the request's normalized prefill frequency in the initial score.
pub const F_SEED_SCALE: f64 = 12.0;

/// One measured CPU expert-service cell.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceCell {
    pub workers: u32,
    pub rows: u32,
    pub experts: u32,
    pub seconds: f64,
}

/// Measured CPU miss service table.
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceTable {
    isa: String,
    cells: Vec<ServiceCell>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantTable {
    max_rows: u32,
    measured_rows: u32,
    max_groups: u32,
    default_participants: u32,
    mixed: std::collections::BTreeMap<(u32, u32), u32>,
}

impl ServiceTable {
    pub fn new(isa: impl Into<String>, cells: Vec<ServiceCell>) -> Result<Self> {
        ensure!(!cells.is_empty(), "empty expert service table");
        for cell in &cells {
            ensure!(
                cell.workers > 0 && cell.rows > 0 && cell.experts > 0,
                "service cell has a zero extent: {cell:?}"
            );
            ensure!(
                cell.seconds.is_finite() && cell.seconds > 0.0,
                "service cell has an invalid time: {cell:?}"
            );
        }
        for (index, cell) in cells.iter().enumerate() {
            ensure!(
                !cells[..index].iter().any(|other| {
                    other.workers == cell.workers
                        && other.rows == cell.rows
                        && other.experts == cell.experts
                }),
                "duplicate service cell {cell:?}"
            );
        }
        Ok(Self {
            isa: isa.into(),
            cells,
        })
    }

    pub fn isa(&self) -> &str {
        &self.isa
    }

    /// Worker count with the smallest mean all-miss decode time; ties keep fewer workers.
    pub fn worker_count(&self) -> Result<u32> {
        let workers: std::collections::BTreeSet<u32> = self
            .cells
            .iter()
            .filter(|cell| cell.rows == 1)
            .map(|cell| cell.workers)
            .collect();
        ensure!(!workers.is_empty(), "service table has no rows=1 cells");
        let mut best: Option<(f64, u32)> = None;
        for workers in workers {
            let experts: Vec<u32> = self
                .cells
                .iter()
                .filter(|cell| cell.rows == 1 && cell.workers == workers)
                .map(|cell| cell.experts)
                .collect();
            let mut total = 0.0;
            let mut count = 0u32;
            for experts in experts {
                total += self.seconds(1, experts, workers)?;
                count += 1;
            }
            ensure!(count > 0, "no rows=1 cells for {workers} workers");
            let mean = total / f64::from(count);
            if best.is_none_or(|(time, chosen)| mean < time || (mean == time && workers < chosen)) {
                best = Some((mean, workers));
            }
        }
        Ok(best.map(|(_, workers)| workers).expect("best is set"))
    }

    pub fn pool_worker_count(&self) -> Result<u32> {
        let groups = self
            .cells
            .iter()
            .map(|cell| cell.experts)
            .max()
            .expect("nonempty service table");
        Ok(self
            .participant_table(self.max_rows(), groups)?
            .worker_count())
    }

    pub fn participant_table(&self, max_rows: u32, max_groups: u32) -> Result<ParticipantTable> {
        ensure!(max_rows > 0, "zero participant rows");
        ensure!(max_groups > 0, "zero participant groups");
        let workers: Vec<u32> = self
            .cells
            .iter()
            .map(|cell| cell.workers)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut default_participants = workers[0];
        let mut mixed = std::collections::BTreeMap::new();
        let measured_rows = self.max_rows();
        for rows in 1..=measured_rows {
            for groups in 1..=max_groups {
                let mut best: Option<(f64, u32)> = None;
                for worker in &workers {
                    let seconds =
                        self.estimate_seconds(f64::from(rows), f64::from(groups), *worker)?;
                    if best.is_none_or(|(time, chosen)| {
                        seconds < time || (seconds == time && *worker < chosen)
                    }) {
                        best = Some((seconds, *worker));
                    }
                }
                let participants = best.map(|(_, workers)| workers).expect("best is set");
                if rows == 1 && groups == 1 {
                    default_participants = participants;
                }
                if participants != default_participants {
                    mixed.insert((rows, groups), participants);
                }
            }
        }
        Ok(ParticipantTable {
            max_rows,
            measured_rows,
            max_groups,
            default_participants,
            mixed,
        })
    }

    /// Largest measured row count.
    pub fn max_rows(&self) -> u32 {
        self.cells.iter().map(|cell| cell.rows).max().unwrap_or(0)
    }

    /// Exact cell or linear interpolation along the rows axis; rows outside the measured extent error.
    pub fn rows_time(&self, rows: u32, experts: u32, workers: u32) -> Result<f64> {
        ensure!(rows > 0, "zero service rows");
        if let Some(cell) = self
            .cells
            .iter()
            .find(|cell| cell.rows == rows && cell.experts == experts && cell.workers == workers)
        {
            return Ok(cell.seconds);
        }
        let mut lower: Option<(u32, f64)> = None;
        let mut upper: Option<(u32, f64)> = None;
        for cell in &self.cells {
            if cell.experts != experts || cell.workers != workers {
                continue;
            }
            if cell.rows < rows && lower.is_none_or(|(seen, _)| cell.rows > seen) {
                lower = Some((cell.rows, cell.seconds));
            }
            if cell.rows > rows && upper.is_none_or(|(seen, _)| cell.rows < seen) {
                upper = Some((cell.rows, cell.seconds));
            }
        }
        match (lower, upper) {
            (Some((lo, lo_s)), Some((hi, hi_s))) => {
                Ok(lo_s + (hi_s - lo_s) * f64::from(rows - lo) / f64::from(hi - lo))
            }
            _ => anyhow::bail!(
                "rows={rows} outside the measured rows extent {}..={} for experts={experts} workers={workers} (isa {})",
                self.cells.iter().map(|cell| cell.rows).min().unwrap_or(0),
                self.max_rows(),
                self.isa
            ),
        }
    }

    /// Measured or linearly interpolated service time over the experts axis.
    pub fn seconds(&self, rows: u32, experts: u32, workers: u32) -> Result<f64> {
        ensure!(
            rows > 0 && experts > 0 && workers > 0,
            "zero service extent"
        );
        if let Some(cell) = self
            .cells
            .iter()
            .find(|cell| cell.rows == rows && cell.experts == experts && cell.workers == workers)
        {
            return Ok(cell.seconds);
        }
        let mut lower: Option<(u32, f64)> = None;
        let mut upper: Option<(u32, f64)> = None;
        for cell in &self.cells {
            if cell.rows != rows || cell.workers != workers {
                continue;
            }
            if cell.experts < experts && lower.is_none_or(|(seen, _)| cell.experts > seen) {
                lower = Some((cell.experts, cell.seconds));
            }
            if cell.experts > experts && upper.is_none_or(|(seen, _)| cell.experts < seen) {
                upper = Some((cell.experts, cell.seconds));
            }
        }
        match (lower, upper) {
            (Some((lo, lo_s)), Some((hi, hi_s))) => {
                Ok(lo_s + (hi_s - lo_s) * f64::from(experts - lo) / f64::from(hi - lo))
            }
            _ => anyhow::bail!(
                "service table has no bracketing cell for rows={rows} experts={experts} workers={workers} (isa {})",
                self.isa
            ),
        }
    }

    /// Continuous bilinear interpolation over calibrated `(rows, experts)` knots, linear above the top `experts` knot.
    pub fn estimate_seconds(&self, rows: f64, experts: f64, workers: u32) -> Result<f64> {
        ensure!(
            rows.is_finite()
                && rows >= 1.0
                && rows <= f64::from(self.max_rows())
                && experts.is_finite()
                && experts >= 0.0
                && workers > 0,
            "invalid continuous service extent rows={rows} experts={experts} workers={workers}"
        );
        if experts == 0.0 {
            return Ok(0.0);
        }
        let knots: Vec<u32> = self
            .cells
            .iter()
            .filter(|cell| cell.workers == workers)
            .map(|cell| cell.experts)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        ensure!(
            !knots.is_empty(),
            "service table has no cells for workers={workers} (isa {})",
            self.isa
        );
        let at_experts = |expert_knot: u32| -> Result<f64> {
            let lo = rows.floor() as u32;
            let hi = rows.ceil() as u32;
            if lo == hi {
                self.rows_time(lo, expert_knot, workers)
            } else {
                let lo_s = self.rows_time(lo, expert_knot, workers)?;
                let hi_s = self.rows_time(hi, expert_knot, workers)?;
                Ok(lo_s + (hi_s - lo_s) * (rows - f64::from(lo)))
            }
        };
        let first = knots[0];
        if experts <= f64::from(first) {
            return Ok(at_experts(first)? * (experts / f64::from(first)));
        }
        let last = knots[knots.len() - 1];
        if experts >= f64::from(last) {
            return Ok(at_experts(last)? * (experts / f64::from(last)));
        }
        for window in knots.windows(2) {
            let (lo, hi) = (window[0], window[1]);
            if f64::from(lo) <= experts && experts <= f64::from(hi) {
                let lo_s = at_experts(lo)?;
                let hi_s = at_experts(hi)?;
                return Ok(lo_s + (hi_s - lo_s) * (experts - f64::from(lo)) / f64::from(hi - lo));
            }
        }
        anyhow::bail!(
            "service table failed to bracket continuous experts={experts} for workers={workers}"
        )
    }
}

impl ParticipantTable {
    pub fn worker_count(&self) -> u32 {
        self.mixed
            .values()
            .copied()
            .fold(self.default_participants, u32::max)
    }

    pub fn uniform(max_rows: u32, max_groups: u32, participants: u32) -> Result<Self> {
        ensure!(
            max_rows > 0 && max_groups > 0 && participants > 0,
            "zero participant table extent rows={max_rows} groups={max_groups} workers={participants}"
        );
        Ok(Self {
            max_rows,
            measured_rows: max_rows,
            max_groups,
            default_participants: participants,
            mixed: std::collections::BTreeMap::new(),
        })
    }

    pub fn participant(&self, rows: usize, groups: usize) -> Result<u32> {
        let rows = u32::try_from(rows).context("participant rows exceed u32")?;
        let groups = u32::try_from(groups).context("participant groups exceed u32")?;
        let groups = groups.max(1);
        ensure!(
            rows > 0 && rows <= self.max_rows && groups <= self.max_groups,
            "participant extent rows={rows} groups={groups} outside rows 1..={} groups 1..={}",
            self.max_rows,
            self.max_groups
        );
        let rows = rows.min(self.measured_rows);
        Ok(self
            .mixed
            .get(&(rows, groups))
            .copied()
            .unwrap_or(self.default_participants))
    }
}

/// Setup reserves subtracted before any expert slot or host block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reserves {
    pub device_fixed_bytes: u64,
    pub host_fixed_bytes: u64,
    pub residual_row_bytes: u64,
    pub normalized_row_bytes: u64,
    pub contribution_row_bytes: u64,
    pub done_row_bytes: u64,
    pub route_row_bytes: u64,
    pub prefill_staging_slots: u32,
    pub disk_ring_slots: u32,
}

impl Reserves {
    /// Device bytes reserved outside the expert slots: fixed profile plus staging slots.
    pub fn persistent_bytes(&self, slot_bytes: u64) -> Result<u64> {
        let staging = u64::from(self.prefill_staging_slots)
            .checked_mul(slot_bytes)
            .context("prefill staging overflow")?;
        self.device_fixed_bytes
            .checked_add(staging)
            .context("device reserve overflow")
    }

    /// Whole-span buffer bytes for `span_rows`; time-shared with the expert slot arena.
    pub fn span_bytes(&self, span_rows: u32) -> Result<u64> {
        let per_row = self
            .residual_row_bytes
            .checked_add(self.normalized_row_bytes)
            .and_then(|n| n.checked_add(self.contribution_row_bytes))
            .and_then(|n| n.checked_add(self.done_row_bytes))
            .and_then(|n| n.checked_add(self.route_row_bytes))
            .context("span reserve overflow")?;
        u64::from(span_rows)
            .checked_mul(per_row)
            .context("span buffer overflow")
    }

    pub fn host_bytes(&self, slot_bytes: u64) -> Result<u64> {
        let ring = u64::from(self.disk_ring_slots)
            .checked_mul(slot_bytes)
            .context("disk ring overflow")?;
        self.host_fixed_bytes
            .checked_add(ring)
            .context("host reserve overflow")
    }
}

/// Planner inputs; every quantity is measured or derived by setup.
#[derive(Clone, Debug)]
pub struct PlanInputs<'a> {
    pub geometry: ExpertGeometry,
    pub layout: crate::experts::catalogue::ExpertLayout,
    pub device_free_bytes: u64,
    pub host_available_bytes: u64,
    pub h2d_bytes_per_second: f64,
    pub d2h_bytes_per_second: f64,
    /// Device async copy engines available to expert transfers.
    pub copy_engines: u32,
    pub disk_readers: u32,
    /// Successfully registered host span measured at setup; 0 when unsupported.
    pub host_registration_bytes: u64,
    pub service: &'a ServiceTable,
    pub reserves: &'a Reserves,
    pub span_rows: u32,
    pub wave_rows: u32,
}

/// Measured peer capacity, captured 4 KiB decode bounce rate, and eager 8/68 MiB prefill rates.
#[derive(Clone, Copy, Debug)]
pub struct PeerInputs<'a> {
    pub device_free_bytes: u64,
    pub device_reserved_bytes: u64,
    pub decode_transfer: (u64, f64, f64),
    pub prefill_transfers: &'a [(u64, f64, f64)],
}

/// Setup capacity derived for one admission cap.
#[derive(Clone, Debug)]
pub struct Capacity {
    pub layout: crate::experts::catalogue::ExpertLayout,
    pub span_rows: u32,
    pub wave_rows: u32,
    pub device_expert_bytes: u64,
    pub peer_expert_bytes: u64,
    pub host_expert_bytes: u64,
    pub gpu0_layers: u32,
    pub slots_per_layer: Vec<u32>,
    /// Usable host blocks after reserving the per-token spare blocks.
    pub host_blocks: u32,
    /// Reserve host blocks for swap write-backs: the admission cap, at least one when the host
    /// tier is non-empty.
    pub spare_blocks: u32,
    pub disk_bytes: u64,
    pub worker_count: u32,
    pub disk_workers: u32,
    pub prefill_staging_slots: u32,
    /// Allocated host arena bytes fit the measured page-lock registration.
    pub pinned: bool,
}

impl Capacity {
    pub fn slots_total(&self) -> u32 {
        self.slots_per_layer.iter().sum()
    }

    pub fn gpu0_slots(&self) -> u32 {
        self.slots_per_layer[..self.gpu0_layers as usize]
            .iter()
            .sum()
    }

    pub fn peer_slots(&self) -> u32 {
        self.slots_per_layer[self.gpu0_layers as usize..]
            .iter()
            .sum()
    }

    pub fn slot_bytes(&self) -> u64 {
        self.layout.slot_bytes()
    }

    pub fn host_backed(&self, geometry: &ExpertGeometry) -> bool {
        u64::from(self.host_blocks) == u64::from(geometry.layers) * u64::from(geometry.experts)
            || (self.slots_per_layer[self.gpu0_layers as usize..]
                .iter()
                .all(|&s| s == geometry.experts)
                && u64::from(self.host_blocks)
                    == u64::from(self.gpu0_layers) * u64::from(geometry.experts))
    }
}

/// One span/wave candidate with the caller's exact reserves.
#[derive(Clone, Debug)]
pub struct SpanCandidate {
    pub span_rows: u32,
    pub wave_rows: u32,
    pub reserves: Reserves,
}

/// Tail slots borrowed by a prefill of `prefill_rows` at `span_rows`.
pub fn span_borrow_slots(
    reserves: &Reserves,
    slot_bytes: u64,
    prefill_rows: u32,
    span_rows: u32,
) -> Result<u32> {
    ensure!(span_rows > 0, "zero span rows");
    let rows = prefill_rows.min(span_rows);
    ensure!(rows > 0, "zero prefill rows");
    let bytes = reserves.span_bytes(rows)?;
    let slot = slot_bytes;
    u32::try_from(bytes.div_ceil(slot)).context("span borrow exceeds u32 slots")
}

/// Largest span whose borrowed tail slots leave one slot per layer; ties keep fewer borrowed slots.
pub fn select_span(
    inputs: &PlanInputs<'_>,
    admission_cap: u32,
    prefill_rows: u32,
    candidates: &[SpanCandidate],
) -> Result<(usize, Capacity)> {
    select_span_with_peer(inputs, admission_cap, prefill_rows, candidates, None)
}

pub fn select_span_with_peer(
    inputs: &PlanInputs<'_>,
    admission_cap: u32,
    prefill_rows: u32,
    candidates: &[SpanCandidate],
    peer: Option<PeerInputs<'_>>,
) -> Result<(usize, Capacity)> {
    ensure!(!candidates.is_empty(), "empty span candidate list");
    let mut best: Option<(usize, u32, u32, Capacity)> = None;
    let mut failures: Vec<String> = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let candidate_inputs = PlanInputs {
            span_rows: candidate.span_rows,
            wave_rows: candidate.wave_rows,
            reserves: &candidate.reserves,
            ..inputs.clone()
        };
        let capacity = match capacity_with_peer(&candidate_inputs, admission_cap, peer) {
            Ok(capacity) => capacity,
            Err(error) => {
                failures.push(format!("candidate {index}: {error}"));
                continue;
            }
        };
        let borrow = span_borrow_slots(
            &candidate.reserves,
            inputs.layout.slot_bytes(),
            prefill_rows,
            candidate.span_rows,
        )?;
        if borrow > capacity.gpu0_slots() - capacity.gpu0_layers {
            failures.push(format!(
                "candidate {index}: borrow {borrow} leaves fewer than {} slots",
                capacity.gpu0_layers
            ));
            continue;
        }
        let better = best.as_ref().is_none_or(|(_, span, seen, _)| {
            candidate.span_rows > *span || (candidate.span_rows == *span && borrow < *seen)
        });
        if better {
            best = Some((index, candidate.span_rows, borrow, capacity));
        }
    }
    match best {
        Some((index, _, _, capacity)) => Ok((index, capacity)),
        None => anyhow::bail!(
            "no feasible span candidate; {}",
            failures.first().map(String::as_str).unwrap_or("empty")
        ),
    }
}

/// Per-request initial residency, ranked by the request's prefill frequency.
pub fn initial_residency(
    frequency: &[Vec<f64>],
    slots_per_layer: &[u32],
    geometry: &ExpertGeometry,
) -> Result<Vec<Vec<u32>>> {
    validate_frequency(frequency, geometry)?;
    ensure!(
        slots_per_layer.len() == geometry.layers as usize,
        "slot plan has {} layers, expected {}",
        slots_per_layer.len(),
        geometry.layers
    );
    let mut initial = Vec::with_capacity(geometry.layers as usize);
    for (layer, slots) in slots_per_layer.iter().enumerate() {
        ensure!(
            *slots <= geometry.experts,
            "layer {layer} requests {slots} slots for {} experts",
            geometry.experts
        );
        let mut ranked: Vec<u32> = (0..geometry.experts).collect();
        ranked.sort_by(|&a, &b| {
            frequency[layer][b as usize]
                .total_cmp(&frequency[layer][a as usize])
                .then(a.cmp(&b))
        });
        ranked.truncate(*slots as usize);
        initial.push(ranked);
    }
    Ok(initial)
}

fn validate_frequency(frequency: &[Vec<f64>], geometry: &ExpertGeometry) -> Result<()> {
    ensure!(
        frequency.len() == geometry.layers as usize,
        "frequency has {} layers, expected {}",
        frequency.len(),
        geometry.layers
    );
    for (layer, row) in frequency.iter().enumerate() {
        ensure!(
            row.len() == geometry.experts as usize,
            "layer {layer} frequency has {} experts, expected {}",
            row.len(),
            geometry.experts
        );
        ensure!(
            row.iter().all(|f| f.is_finite() && *f >= 0.0),
            "layer {layer} frequency has a non-finite or negative entry"
        );
    }
    Ok(())
}

/// Equal split of `total_slots` over layers; the remainder goes to the lowest layer ids.
pub fn equal_slots(total_slots: u64, layers: u32) -> Result<Vec<u32>> {
    ensure!(layers > 0, "zero layers");
    let base = total_slots / u64::from(layers);
    let remainder = total_slots % u64::from(layers);
    ensure!(
        base <= u64::from(u32::MAX),
        "slot split exceeds u32 per layer"
    );
    Ok((0..layers)
        .map(|layer| base as u32 + u32::from(u64::from(layer) < remainder))
        .collect())
}

fn validate_peer_transfers(transfers: &[(u64, f64, f64)]) -> Result<()> {
    ensure!(!transfers.is_empty(), "empty peer transfer calibration");
    for (index, &(b, r01, r10)) in transfers.iter().enumerate() {
        ensure!(
            b > 0
                && r01.is_finite()
                && r01 > 0.0
                && r10.is_finite()
                && r10 > 0.0
                && (index == 0 || b > transfers[index - 1].0),
            "invalid peer transfer calibration cell ({b}, {r01}, {r10})"
        );
    }
    Ok(())
}

fn peer_rate(transfers: &[(u64, f64, f64)], bytes: u64) -> Result<(f64, f64)> {
    validate_peer_transfers(transfers)?;
    if bytes <= transfers[0].0 {
        return Ok((transfers[0].1, transfers[0].2));
    }
    let last = transfers.len() - 1;
    if bytes >= transfers[last].0 {
        return Ok((transfers[last].1, transfers[last].2));
    }
    for window in transfers.windows(2) {
        let (b0, r01_0, r10_0) = window[0];
        let (b1, r01_1, r10_1) = window[1];
        if bytes >= b0 && bytes <= b1 {
            let t = (bytes - b0) as f64 / (b1 - b0) as f64;
            return Ok((r01_0 + (r01_1 - r01_0) * t, r10_0 + (r10_1 - r10_0) * t));
        }
    }
    anyhow::bail!("no bracketing peer transfer cell for bytes={bytes}")
}

fn peer_hop_seconds(
    transfers: &[(u64, f64, f64)],
    bytes_0_to_1: u64,
    bytes_1_to_0: u64,
) -> Result<f64> {
    let (r01, _) = peer_rate(transfers, bytes_0_to_1)?;
    let (_, r10) = peer_rate(transfers, bytes_1_to_0)?;
    Ok(bytes_0_to_1 as f64 / r01 + bytes_1_to_0 as f64 / r10)
}

/// Round-trip peer bounce time interpolated linearly in transfer duration across calibrated byte knots.
pub fn peer_bounce_seconds(
    transfers: &[(u64, f64, f64)],
    bytes_0_to_1: u64,
    bytes_1_to_0: u64,
) -> Result<f64> {
    validate_peer_transfers(transfers)?;
    let one_way = |bytes: u64, forward: bool| -> Result<f64> {
        let rate_of = |cell: (u64, f64, f64)| if forward { cell.1 } else { cell.2 };
        if bytes <= transfers[0].0 {
            return Ok(bytes as f64 / rate_of(transfers[0]));
        }
        let last = transfers.len() - 1;
        if bytes >= transfers[last].0 {
            return Ok(bytes as f64 / rate_of(transfers[last]));
        }
        for window in transfers.windows(2) {
            let c0 = window[0];
            let c1 = window[1];
            if bytes >= c0.0 && bytes <= c1.0 {
                let s0 = c0.0 as f64 / rate_of(c0);
                let s1 = c1.0 as f64 / rate_of(c1);
                let t = (bytes - c0.0) as f64 / (c1.0 - c0.0) as f64;
                return Ok(s0 + (s1 - s0) * t);
            }
        }
        anyhow::bail!("no bracketing peer transfer cell for bytes={bytes}")
    };
    Ok(one_way(bytes_0_to_1, true)? + one_way(bytes_1_to_0, false)?)
}

/// Setup capacity from the reserves-first subtraction for one admission cap.
pub fn capacity(inputs: &PlanInputs<'_>, admission_cap: u32) -> Result<Capacity> {
    capacity_with_peer(inputs, admission_cap, None)
}

pub fn capacity_with_peer(
    inputs: &PlanInputs<'_>,
    admission_cap: u32,
    peer: Option<PeerInputs<'_>>,
) -> Result<Capacity> {
    let geometry = inputs.geometry;
    ensure!(inputs.span_rows > 0, "zero span rows");
    ensure!(
        inputs.wave_rows > 0 && inputs.wave_rows <= inputs.span_rows,
        "wave rows {} outside the span {}",
        inputs.wave_rows,
        inputs.span_rows
    );
    ensure!(
        inputs.h2d_bytes_per_second.is_finite() && inputs.h2d_bytes_per_second > 0.0,
        "invalid H2D rate {}",
        inputs.h2d_bytes_per_second
    );
    ensure!(
        inputs.d2h_bytes_per_second.is_finite() && inputs.d2h_bytes_per_second > 0.0,
        "invalid D2H rate {}",
        inputs.d2h_bytes_per_second
    );
    ensure!(
        inputs.copy_engines > 0 && inputs.copy_engines <= 16,
        "invalid copy engine count {}",
        inputs.copy_engines
    );
    let slot_bytes = inputs.layout.slot_bytes();
    let device_reserved = inputs.reserves.persistent_bytes(slot_bytes)?;
    let host_reserved = inputs.reserves.host_bytes(slot_bytes)?;
    ensure!(
        inputs.device_free_bytes >= device_reserved,
        "persistent reserves {device_reserved} B exceed device free bytes {}",
        inputs.device_free_bytes
    );
    ensure!(
        inputs.host_available_bytes >= host_reserved,
        "host reserves {host_reserved} B exceed host available bytes {}",
        inputs.host_available_bytes
    );
    let device_expert_bytes = inputs.device_free_bytes - device_reserved;
    let host_expert_bytes = inputs.host_available_bytes - host_reserved;
    let routed_slots = u64::from(geometry.layers)
        .checked_mul(u64::from(geometry.experts))
        .context("routed expert count overflow")?;
    let gpu0_max_slots = (device_expert_bytes / slot_bytes).min(routed_slots);
    ensure!(
        gpu0_max_slots >= u64::from(geometry.layers),
        "device expert bytes {device_expert_bytes} hold {gpu0_max_slots} slots, below one per layer ({})",
        geometry.layers
    );
    let headroom = span_borrow_slots(
        inputs.reserves,
        slot_bytes,
        inputs.span_rows,
        inputs.span_rows,
    )?;
    let worker_count = inputs.service.pool_worker_count()?;
    let mut gpu0_layers = geometry.layers;
    let mut peer_expert_bytes = 0u64;
    let mut total_slots = gpu0_max_slots;
    let mut slots_per_layer = equal_slots(total_slots, geometry.layers)?;
    if gpu0_max_slots < routed_slots
        && let Some(peer) = peer
    {
        let peer_free_expert_bytes = peer
            .device_free_bytes
            .saturating_sub(peer.device_reserved_bytes);
        let peer_max_slots = peer_free_expert_bytes / slot_bytes;
        let max_peer_layers = (peer_max_slots / u64::from(geometry.experts))
            .min(u64::from(geometry.layers.saturating_sub(1))) as u32;
        let needed_peer_layers =
            (routed_slots - gpu0_max_slots).div_ceil(u64::from(geometry.experts)) as u32;
        let peer_layers = needed_peer_layers.min(max_peer_layers);
        let candidate_gpu0_layers = geometry.layers - peer_layers;
        let gpu0_slots =
            gpu0_max_slots.min(u64::from(candidate_gpu0_layers) * u64::from(geometry.experts));
        let peer_slots = u64::from(peer_layers) * u64::from(geometry.experts);
        if peer_layers > 0 && gpu0_slots >= u64::from(candidate_gpu0_layers) + u64::from(headroom) {
            let row_0_to_1 = u64::from(geometry.hidden) * 2 + u64::from(geometry.top_k) * 6;
            let row_1_to_0 = u64::from(geometry.hidden) * 2;
            let decode_hop_seconds = f64::from(peer_layers)
                * peer_hop_seconds(
                    std::slice::from_ref(&peer.decode_transfer),
                    row_0_to_1,
                    row_1_to_0,
                )?;
            let removed_misses = (gpu0_slots + peer_slots).saturating_sub(gpu0_max_slots);
            let decode_miss_seconds = (removed_misses as f64 / f64::from(geometry.experts))
                * inputs
                    .service
                    .seconds(1, 1, inputs.service.worker_count()?)?;
            let prefill_hop_seconds = f64::from(peer_layers)
                * peer_hop_seconds(
                    peer.prefill_transfers,
                    u64::from(inputs.span_rows) * row_0_to_1,
                    u64::from(inputs.span_rows) * row_1_to_0,
                )?;
            let prefill_stream_seconds =
                (removed_misses * slot_bytes) as f64 / inputs.h2d_bytes_per_second;
            if decode_hop_seconds < decode_miss_seconds
                && prefill_hop_seconds < prefill_stream_seconds
            {
                gpu0_layers = candidate_gpu0_layers;
                peer_expert_bytes = peer_free_expert_bytes;
                total_slots = gpu0_slots + peer_slots;
                slots_per_layer = equal_slots(gpu0_slots, gpu0_layers)?;
                slots_per_layer.resize(geometry.layers as usize, geometry.experts);
            }
        }
    }
    let host_block_slots = host_expert_bytes / slot_bytes;
    let non_gpu = routed_slots - total_slots;
    ensure!(
        non_gpu == 0 || host_block_slots >= u64::from(admission_cap),
        "host blocks {host_block_slots} cannot hold {admission_cap} reserve blocks"
    );
    let host_target = non_gpu
        .checked_add(u64::from(headroom))
        .context("host headroom overflow")?;
    let (host_blocks, reserve) = if non_gpu == 0 {
        (host_block_slots.min(host_target), 0)
    } else {
        let mut reserve = admission_cap;
        let mut host_blocks = (host_block_slots - u64::from(reserve)).min(host_target);
        if host_blocks > 0 && reserve == 0 {
            if host_block_slots > 1 {
                reserve = 1;
                host_blocks = (host_block_slots - 1).min(host_target);
            } else {
                host_blocks = 0;
            }
        }
        let partial_bytes = host_blocks
            .checked_add(u64::from(reserve))
            .and_then(|blocks| blocks.checked_mul(slot_bytes))
            .context("partial host arena bytes overflow")?;
        let swappable_slots = u64::from(gpu0_layers)
            .checked_mul(u64::from(geometry.experts))
            .context("swappable expert slot count overflow")?;
        let backed_bytes = swappable_slots
            .checked_mul(slot_bytes)
            .context("complete host arena bytes overflow")?;
        if swappable_slots <= host_block_slots
            && (partial_bytes > inputs.host_registration_bytes
                || backed_bytes <= inputs.host_registration_bytes)
        {
            host_blocks = swappable_slots;
            reserve = 0;
        }
        (host_blocks, reserve)
    };
    let host_blocks = u32::try_from(host_blocks).context("host block count exceeds u32")?;
    let hosted = u64::from(host_blocks).min(non_gpu);
    let unique_slots = total_slots
        .checked_add(hosted)
        .context("unique expert count overflow")?;
    ensure!(
        unique_slots <= routed_slots,
        "unique expert slots {unique_slots} exceed the routed total {routed_slots}"
    );
    let disk_bytes = (routed_slots - unique_slots)
        .checked_mul(slot_bytes)
        .context("disk tier bytes overflow")?;
    let arena_bytes = u64::from(host_blocks)
        .checked_add(u64::from(reserve))
        .and_then(|blocks| blocks.checked_mul(slot_bytes))
        .context("host arena bytes overflow")?;
    let pinned = arena_bytes > 0 && arena_bytes <= inputs.host_registration_bytes;
    let disk_workers = if u64::from(host_blocks) < host_target {
        ensure!(inputs.disk_readers > 0, "no measured disk readers");
        inputs.disk_readers
    } else {
        0
    };
    Ok(Capacity {
        layout: inputs.layout.clone(),
        span_rows: inputs.span_rows,
        wave_rows: inputs.wave_rows,
        device_expert_bytes,
        peer_expert_bytes,
        host_expert_bytes,
        gpu0_layers,
        slots_per_layer,
        host_blocks,
        spare_blocks: reserve,
        disk_bytes,
        worker_count,
        disk_workers,
        prefill_staging_slots: inputs.reserves.prefill_staging_slots,
        pinned,
    })
}

/// Evicted tail slots of a prefill borrow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Borrow {
    /// Borrowed physical positions, descending.
    pub positions: Vec<u32>,
    pub experts: Vec<(u32, u32)>,
    pub to_host: u32,
    pub to_disk: u32,
    pub d2h_bytes: u64,
    pub disk_bytes: u64,
    /// One flag per victim; true when its host backing must be written.
    pub write_back: Vec<bool>,
    /// Device events each host victim's D2H waits on; filled by the residency.
    pub host_waits: Vec<Option<u64>>,
}

/// Tier of one expert in the provenance ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Gpu,
    Host,
    Disk,
}

/// Sentinel of a slot position not holding a GPU-resident expert.
const EMPTY_SLOT: u32 = u32::MAX;

/// Exact tier membership and byte accounting; the capacity/provenance control. Slot positions are
/// the compact `SlotLayout` ids: rank-major, layer-minor, staging excluded.
pub struct TierLedger {
    layout: SlotLayout,
    host_blocks: u32,
    host_backing: bool,
    gpu: Vec<Vec<bool>>,
    host: Vec<Vec<bool>>,
    /// Physical slot id -> resident expert, or `EMPTY_SLOT`.
    slots: Vec<u32>,
    /// Physical slot id -> layer.
    layer_of: Vec<u32>,
    /// Positions lent to the span buffers; unavailable to admission until the refill.
    borrowed: Vec<bool>,
}

impl TierLedger {
    /// Starts with every slot free and the host tier empty.
    pub fn empty(layout: SlotLayout, host_blocks: u32) -> Result<Self> {
        let geometry = layout.geometry();
        let (layer_of, count) = Self::positions(&layout)?;
        let routed = u64::from(geometry.layers) * u64::from(geometry.experts);
        let peer_complete = (layout.gpu0_layers()..geometry.layers)
            .all(|layer| layout.slot_count(layer).ok() == Some(geometry.experts));
        let swappable = u64::from(layout.gpu0_layers()) * u64::from(geometry.experts);
        let ledger = Self {
            host_backing: u64::from(host_blocks) == routed
                || (peer_complete && u64::from(host_blocks) == swappable),
            gpu: vec![vec![false; geometry.experts as usize]; geometry.layers as usize],
            host: vec![vec![false; geometry.experts as usize]; geometry.layers as usize],
            slots: vec![EMPTY_SLOT; count],
            layer_of,
            borrowed: vec![false; count],
            layout,
            host_blocks,
        };
        ledger.check()?;
        Ok(ledger)
    }

    /// Starts from a request's initial GPU set, one expert per rank; the host tier starts empty.
    pub fn new(layout: SlotLayout, host_blocks: u32, initial: &[Vec<u32>]) -> Result<Self> {
        let geometry = layout.geometry();
        ensure!(
            initial.len() == geometry.layers as usize,
            "ledger layer count mismatch"
        );
        let mut ledger = Self::empty(layout, host_blocks)?;
        for (layer, set) in initial.iter().enumerate() {
            let count = ledger.slot_count(layer as u32)?;
            ensure!(
                set.len() as u32 == count,
                "layer {layer} initial set has {} experts for {count} slots",
                set.len()
            );
            for (rank, &expert) in set.iter().enumerate() {
                ensure!(
                    expert < geometry.experts,
                    "initial expert {expert} outside {}",
                    geometry.experts
                );
                ensure!(
                    !ledger.gpu[layer][expert as usize],
                    "duplicate initial expert {expert}"
                );
                let position = ledger.position(layer as u32, rank as u32)?;
                ledger.gpu[layer][expert as usize] = true;
                ledger.slots[position] = expert;
            }
        }
        ledger.check()?;
        Ok(ledger)
    }

    fn positions(layout: &SlotLayout) -> Result<(Vec<u32>, usize)> {
        let geometry = layout.geometry();
        let bytes = layout.slot_bytes();
        let mut count = 0usize;
        for layer in 0..geometry.layers {
            count = count
                .checked_add(layout.slot_count(layer)? as usize)
                .context("slot position count overflow")?;
        }
        let mut layer_of = vec![u32::MAX; count];
        for layer in 0..geometry.layers {
            for rank in 0..layout.slot_count(layer)? {
                let start = layout.resident_slot(layer, rank)?.start;
                ensure!(
                    start % bytes == 0,
                    "resident slot start {start} is not slot-aligned"
                );
                let position = start / bytes;
                ensure!(
                    position < count && layer_of[position] == u32::MAX,
                    "slot layout resolves two ranks to position {position}"
                );
                layer_of[position] = layer;
            }
        }
        Ok((layer_of, count))
    }

    fn slot_count(&self, layer: u32) -> Result<u32> {
        self.layout.slot_count(layer)
    }

    fn position(&self, layer: u32, rank: u32) -> Result<usize> {
        Ok(self.layout.resident_slot(layer, rank)?.start / self.layout.slot_bytes())
    }

    pub fn tier(&self, layer: u32, expert: u32) -> Result<Tier> {
        self.check_index(layer, expert)?;
        let (layer, expert) = (layer as usize, expert as usize);
        Ok(if self.gpu[layer][expert] {
            Tier::Gpu
        } else if self.host[layer][expert] {
            Tier::Host
        } else {
            Tier::Disk
        })
    }

    pub fn has_full_host_backing(&self) -> bool {
        self.host_backing
    }

    fn check_index(&self, layer: u32, expert: u32) -> Result<()> {
        let geometry = self.layout.geometry();
        ensure!(
            layer < geometry.layers,
            "expert layer {layer} outside {}",
            geometry.layers
        );
        ensure!(
            expert < geometry.experts,
            "expert {expert} outside {}",
            geometry.experts
        );
        Ok(())
    }

    fn free_host_blocks(&self) -> u32 {
        let used: u32 = self
            .host
            .iter()
            .flat_map(|layer| layer.iter())
            .filter(|resident| **resident)
            .count() as u32;
        self.host_blocks.saturating_sub(used)
    }

    /// Admits a host expert into a GPU slot; complete backing retains both host copies.
    pub fn admit_host(&mut self, layer: u32, expert: u32, victim: u32) -> Result<()> {
        self.check_index(layer, expert)?;
        self.check_index(layer, victim)?;
        let (l, e, v) = (layer as usize, expert as usize, victim as usize);
        ensure!(
            self.host[l][e],
            "expert {expert} of layer {layer} is not host resident"
        );
        ensure!(
            self.gpu[l][v],
            "victim {victim} of layer {layer} is not GPU resident"
        );
        ensure!(expert != victim, "candidate {expert} equals its victim");
        let position = self.slot_position(layer, victim)?;
        ensure!(
            !self.borrowed[position],
            "victim slot {position} is borrow-claimed"
        );
        self.host[l][e] = self.host_backing;
        self.gpu[l][e] = true;
        self.gpu[l][v] = false;
        self.host[l][v] = true;
        self.slots[position] = expert;
        debug_assert!(self.check().is_ok());
        Ok(())
    }

    /// Admits a disk expert; the victim goes to a spare host block or to the disk tier.
    pub fn admit_disk(
        &mut self,
        layer: u32,
        expert: u32,
        victim: u32,
        host_block_free: bool,
    ) -> Result<()> {
        self.check_index(layer, expert)?;
        self.check_index(layer, victim)?;
        let (l, e, v) = (layer as usize, expert as usize, victim as usize);
        ensure!(
            self.tier(layer, expert)? == Tier::Disk,
            "expert {expert} is not disk resident"
        );
        ensure!(
            self.gpu[l][v],
            "victim {victim} of layer {layer} is not GPU resident"
        );
        ensure!(expert != victim, "candidate {expert} equals its victim");
        ensure!(
            !host_block_free || self.free_host_blocks() > 0,
            "no free host block for the victim"
        );
        let position = self.slot_position(layer, victim)?;
        ensure!(
            !self.borrowed[position],
            "victim slot {position} is borrow-claimed"
        );
        self.gpu[l][e] = true;
        self.gpu[l][v] = false;
        if host_block_free {
            self.host[l][v] = true;
        }
        self.slots[position] = expert;
        debug_assert!(self.check().is_ok());
        Ok(())
    }

    /// Drops a GPU-only expert to the disk tier without a host copy.
    #[cfg(test)]
    pub fn evict_gpu(&mut self, layer: u32, expert: u32) -> Result<()> {
        self.check_index(layer, expert)?;
        let (l, e) = (layer as usize, expert as usize);
        ensure!(
            self.gpu[l][e],
            "expert {expert} of layer {layer} is not GPU resident"
        );
        let position = self.slot_position(layer, expert)?;
        self.gpu[l][e] = false;
        self.slots[position] = EMPTY_SLOT;
        debug_assert!(self.check().is_ok());
        Ok(())
    }

    /// Places a non-GPU expert into the first empty, non-borrowed slot of its layer.
    pub fn install(&mut self, layer: u32, expert: u32) -> Result<u32> {
        self.check_index(layer, expert)?;
        let (l, e) = (layer as usize, expert as usize);
        ensure!(
            !self.gpu[l][e],
            "expert {expert} of layer {layer} is already GPU resident"
        );
        let position = self
            .empty_slot(layer)?
            .with_context(|| format!("layer {layer} has no empty slot for expert {expert}"))?;
        let position = usize::try_from(position).context("slot position exceeds usize")?;
        self.host[l][e] &= self.host_backing;
        self.gpu[l][e] = true;
        self.slots[position] = expert;
        debug_assert!(self.check().is_ok());
        u32::try_from(position).context("slot position exceeds u32")
    }

    /// Physical position of the first empty, non-borrowed slot of a layer.
    pub fn empty_slot(&self, layer: u32) -> Result<Option<u32>> {
        for rank in 0..self.slot_count(layer)? {
            let position = self.position(layer, rank)?;
            if self.slots[position] == EMPTY_SLOT && !self.borrowed[position] {
                return Ok(Some(
                    u32::try_from(position).context("slot position exceeds u32")?,
                ));
            }
        }
        Ok(None)
    }

    /// Slot position of a GPU-resident expert.
    pub fn slot(&self, layer: u32, expert: u32) -> Result<u32> {
        u32::try_from(self.slot_position(layer, expert)?).context("slot position exceeds u32")
    }

    fn slot_position(&self, layer: u32, expert: u32) -> Result<usize> {
        for rank in 0..self.slot_count(layer)? {
            let position = self.position(layer, rank)?;
            if self.slots[position] == expert {
                return Ok(position);
            }
        }
        anyhow::bail!("expert {expert} of layer {layer} has no slot")
    }

    /// Marks a non-GPU expert host-resident within the host block capacity.
    pub fn place_host(&mut self, layer: u32, expert: u32) -> Result<()> {
        self.check_index(layer, expert)?;
        let (l, e) = (layer as usize, expert as usize);
        ensure!(
            !self.gpu[l][e] || self.host_backing,
            "expert {expert} of layer {layer} is GPU resident"
        );
        ensure!(
            !self.host[l][e],
            "expert {expert} of layer {layer} is already host resident"
        );
        ensure!(
            self.free_host_blocks() > 0,
            "host tier holds {} experts for {} blocks",
            self.host_blocks - self.free_host_blocks(),
            self.host_blocks
        );
        self.host[l][e] = true;
        debug_assert!(self.check().is_ok());
        Ok(())
    }

    /// Physical positions of the layer's empty, non-borrowed slots.
    pub fn empty_slots(&self, layer: u32) -> Result<u32> {
        let mut empty = 0u32;
        for rank in 0..self.slot_count(layer)? {
            let position = self.position(layer, rank)?;
            if self.slots[position] == EMPTY_SLOT && !self.borrowed[position] {
                empty += 1;
            }
        }
        Ok(empty)
    }

    /// Borrows the last `slots` existing physical positions of the resident prefix
    /// (SlotLayout `borrowed_tail`, staging excluded, every layer's first slot retained).
    /// At most `to_host_limit` new host copies are written; existing backing is retained.
    /// Empty positions are only marked borrowed. The position list is validated before
    /// mutating. Reached only through `Residency::borrow`.
    pub(crate) fn borrow_tail(&mut self, slots: u32, to_host_limit: u32) -> Result<Borrow> {
        ensure!(slots > 0, "zero borrowed slots");
        let range = self.layout.borrowed_tail(slots)?;
        let bytes = self.layout.slot_bytes();
        let slot_bytes = bytes as u64;
        ensure!(
            range.start % bytes == 0 && range.end % bytes == 0,
            "borrowed tail {range:?} is not slot-aligned"
        );
        let first = range.start / bytes;
        let end = range.end / bytes;
        ensure!(
            end <= self.slots.len(),
            "borrowed tail {range:?} exceeds {} positions",
            self.slots.len()
        );
        for position in first..end {
            ensure!(
                !self.borrowed[position],
                "slot position {position} is already borrowed"
            );
            let expert = self.slots[position];
            if expert != EMPTY_SLOT {
                ensure!(
                    self.gpu[self.layer_of[position] as usize][expert as usize],
                    "slot owner {expert} is not GPU resident"
                );
            }
        }
        let mut positions: Vec<u32> = Vec::with_capacity(slots as usize);
        let mut victims: Vec<(u32, u32)> = Vec::with_capacity(slots as usize);
        let mut to_host = 0u32;
        let mut to_disk = 0u32;
        let mut write_back = Vec::with_capacity(slots as usize);
        for position in (first..end).rev() {
            self.borrowed[position] = true;
            positions.push(u32::try_from(position).context("slot position exceeds u32")?);
            let expert = self.slots[position];
            if expert == EMPTY_SLOT {
                continue;
            }
            let layer = self.layer_of[position];
            let (l, e) = (layer as usize, expert as usize);
            self.gpu[l][e] = false;
            let backed = self.host[l][e];
            let copy = !backed && to_host < to_host_limit;
            if backed || copy {
                self.host[l][e] = true;
                to_host += 1;
            } else {
                to_disk += 1;
            }
            self.slots[position] = EMPTY_SLOT;
            victims.push((layer, expert));
            write_back.push(copy);
        }
        self.check()?;
        Ok(Borrow {
            d2h_bytes: write_back.iter().filter(|&&copy| copy).count() as u64 * slot_bytes,
            disk_bytes: u64::from(to_disk) * slot_bytes,
            positions,
            experts: victims,
            to_host,
            to_disk,
            write_back,
            host_waits: Vec::new(),
        })
    }

    /// Returns every borrowed position to admission; the decode target repopulates them.
    /// Reached only through `Residency::refill`.
    pub(crate) fn refill_tail(&mut self) -> Result<Vec<u32>> {
        let mut returned = Vec::new();
        for (position, borrowed) in self.borrowed.iter_mut().enumerate() {
            if *borrowed {
                *borrowed = false;
                returned.push(u32::try_from(position).context("slot position exceeds u32")?);
            }
        }
        self.check()?;
        Ok(returned)
    }

    pub fn unique_bytes(&self) -> Result<u64> {
        let slots = self
            .gpu
            .iter()
            .zip(&self.host)
            .flat_map(|(gpu, host)| gpu.iter().zip(host))
            .filter(|(gpu, host)| **gpu || **host)
            .count();
        u64::try_from(slots)?
            .checked_mul(self.layout.slot_bytes() as u64)
            .context("unique bytes overflow")
    }

    pub fn disk_bytes(&self) -> Result<u64> {
        let geometry = self.layout.geometry();
        let routed = u64::from(geometry.layers)
            .checked_mul(u64::from(geometry.experts))
            .context("routed expert count overflow")?;
        let slot = self.layout.slot_bytes() as u64;
        let used = self.unique_bytes()? / slot;
        routed
            .checked_sub(used)
            .and_then(|n| n.checked_mul(slot))
            .context("disk bytes overflow")
    }

    pub fn check(&self) -> Result<()> {
        let geometry = self.layout.geometry();
        ensure!(
            self.layer_of.len() == self.slots.len() && self.borrowed.len() == self.slots.len(),
            "ledger position arrays disagree"
        );
        ensure!(
            !self.layer_of.contains(&u32::MAX),
            "slot layout leaves a position unmapped"
        );
        let mut host_used = 0u32;
        for layer in 0..geometry.layers {
            let count = self.slot_count(layer)? as usize;
            let gpu = self.gpu[layer as usize]
                .iter()
                .filter(|resident| **resident)
                .count();
            ensure!(
                gpu <= count,
                "layer {layer} holds {gpu} GPU experts for {count} slots"
            );
            let mut slotted = Vec::with_capacity(count);
            for rank in 0..count {
                let position = self.position(layer, rank as u32)?;
                ensure!(
                    self.layer_of[position] == layer,
                    "slot {position} does not belong to layer {layer}"
                );
                if self.borrowed[position] {
                    ensure!(
                        self.slots[position] == EMPTY_SLOT,
                        "borrowed position {position} still holds an expert"
                    );
                }
                let expert = self.slots[position];
                if expert != EMPTY_SLOT {
                    ensure!(
                        (expert as usize) < geometry.experts as usize,
                        "slot rank {rank} of layer {layer} holds expert {expert} outside the layer"
                    );
                    ensure!(
                        self.gpu[layer as usize][expert as usize],
                        "slot rank {rank} of layer {layer} holds non-resident expert {expert}"
                    );
                    slotted.push(expert);
                }
            }
            slotted.sort_unstable();
            let before = slotted.len();
            slotted.dedup();
            ensure!(
                slotted.len() == before,
                "layer {layer} holds one expert in two slots"
            );
            for expert in 0..geometry.experts as usize {
                ensure!(
                    self.host_backing
                        || !(self.gpu[layer as usize][expert] && self.host[layer as usize][expert]),
                    "expert {expert} of layer {layer} is both GPU and host resident"
                );
                if self.gpu[layer as usize][expert] {
                    ensure!(
                        slotted.binary_search(&(expert as u32)).is_ok(),
                        "GPU expert {expert} of layer {layer} has no slot"
                    );
                }
            }
            host_used += self.host[layer as usize]
                .iter()
                .filter(|resident| **resident)
                .count() as u32;
        }
        ensure!(
            host_used <= self.host_blocks,
            "host tier holds {host_used} experts for {} blocks",
            self.host_blocks
        );
        Ok(())
    }
}

/// Admission budget parameters derived from one budget in bytes per token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Admission {
    pub budget_bytes_per_token: u64,
}

impl Admission {
    /// Experts per token under the hard step cap; the token bucket is withdrawn.
    pub fn hard_cap(&self, slot_bytes: u64) -> Result<u32> {
        ensure!(slot_bytes > 0, "zero expert slot bytes");
        u32::try_from(self.budget_bytes_per_token / slot_bytes)
            .context("admission cap exceeds u32 experts")
    }
}

/// F seed: 12 copies of the request's normalized prefill frequency.
pub fn seed_scores(frequency: &[Vec<f64>], geometry: &ExpertGeometry) -> Result<Vec<Vec<f64>>> {
    validate_frequency(frequency, geometry)?;
    let mut scores = Vec::with_capacity(frequency.len());
    for (layer, row) in frequency.iter().enumerate() {
        let total: f64 = row.iter().sum();
        ensure!(total > 0.0, "layer {layer} prefill frequency is all zero");
        scores.push(
            row.iter()
                .map(|value| F_SEED_SCALE * value * f64::from(geometry.top_k) / total)
                .collect(),
        );
    }
    Ok(scores)
}

/// Closed-form admission budget: the per-token copy time may not exceed the CPU time of one
/// missed expert in every layer, priced by the measured copy rates and async engine count.
pub fn admission_budget(
    geometry: &ExpertGeometry,
    service: &ServiceTable,
    h2d_bytes_per_second: f64,
    d2h_bytes_per_second: f64,
    copy_engines: u32,
) -> Result<u64> {
    ensure!(
        h2d_bytes_per_second.is_finite() && h2d_bytes_per_second > 0.0,
        "invalid H2D rate {h2d_bytes_per_second}"
    );
    ensure!(
        d2h_bytes_per_second.is_finite() && d2h_bytes_per_second > 0.0,
        "invalid D2H rate {d2h_bytes_per_second}"
    );
    ensure!(
        copy_engines > 0 && copy_engines <= 16,
        "invalid copy engine count {copy_engines}"
    );
    let workers = service.worker_count()?;
    let cpu_seconds = service.seconds(1, 1, workers)? * f64::from(geometry.layers);
    let bytes_per_second = if copy_engines >= 2 {
        h2d_bytes_per_second.min(d2h_bytes_per_second)
    } else {
        1.0 / (1.0 / h2d_bytes_per_second + 1.0 / d2h_bytes_per_second)
    };
    let bytes = cpu_seconds * bytes_per_second;
    ensure!(
        bytes.is_finite() && bytes >= 0.0 && bytes <= u64::MAX as f64,
        "admission budget {bytes} outside the byte range"
    );
    Ok(bytes.floor() as u64)
}

/// Host-bounce packet layout for peer-GPU routed-expert decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerSendLayout {
    pub ids_off: usize,
    pub ids_bytes: usize,
    pub weights_off: usize,
    pub weights_bytes: usize,
    pub send_bytes: usize,
    pub recv_bytes: usize,
}

impl PeerSendLayout {
    pub fn for_rows(rows: usize, hidden: usize, top_k: usize) -> Result<Self> {
        ensure!(
            rows > 0 && hidden > 0 && top_k > 0,
            "zero peer send dimension rows={rows} hidden={hidden} top_k={top_k}"
        );
        let x_bytes = rows
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(size_of::<half::bf16>()))
            .context("peer send x byte overflow")?;
        ensure!(
            x_bytes.is_multiple_of(16),
            "peer send x bytes={x_bytes} is not 16-byte aligned"
        );
        let ids_off = x_bytes;
        let ids_bytes = rows
            .checked_mul(top_k)
            .and_then(|v| v.checked_mul(size_of::<i32>()))
            .context("peer send ids byte overflow")?;
        let weights_off = ids_off
            .checked_add(ids_bytes)
            .and_then(|v| v.checked_next_multiple_of(16))
            .context("peer send weights offset overflow")?;
        let weights_bytes = rows
            .checked_mul(top_k)
            .and_then(|v| v.checked_mul(size_of::<half::bf16>()))
            .context("peer send weights byte overflow")?;
        let send_bytes = weights_off
            .checked_add(weights_bytes)
            .context("peer send total byte overflow")?;
        Ok(Self {
            ids_off,
            ids_bytes,
            weights_off,
            weights_bytes,
            send_bytes,
            recv_bytes: x_bytes,
        })
    }

    pub fn tail_bf16_elems(max_rows: usize, hidden: usize, top_k: usize) -> Result<usize> {
        let layout = Self::for_rows(max_rows, hidden, top_k)?;
        Ok((layout.send_bytes - layout.ids_off).div_ceil(size_of::<half::bf16>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            first_layer: 0,
            layers: 2,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn service() -> ServiceTable {
        let mut cells = Vec::new();
        for workers in [1u32, 2] {
            for experts in 1..=2u32 {
                cells.push(ServiceCell {
                    workers,
                    rows: 1,
                    experts,
                    seconds: f64::from(experts) * (0.2 / f64::from(workers)),
                });
            }
        }
        ServiceTable::new("test", cells).unwrap()
    }

    fn frequency() -> Vec<Vec<f64>> {
        (0..2)
            .map(|layer| {
                (0..8)
                    .map(|expert| 1.0 + f64::from(layer) + f64::from(7 - expert) * 0.5)
                    .collect()
            })
            .collect()
    }

    fn reserves() -> Reserves {
        Reserves {
            device_fixed_bytes: 1_000,
            host_fixed_bytes: 100,
            residual_row_bytes: 16,
            normalized_row_bytes: 16,
            contribution_row_bytes: 32,
            done_row_bytes: 8,
            route_row_bytes: 40,
            prefill_staging_slots: 1,
            disk_ring_slots: 1,
        }
    }

    fn slot() -> u64 {
        crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry(),
        )
        .unwrap()
        .slot_bytes()
    }

    fn capacity_ref<'a>(inputs: &PlanInputs<'a>) -> PlanInputs<'a> {
        PlanInputs {
            geometry: inputs.geometry,
            layout: inputs.layout.clone(),
            device_free_bytes: inputs.device_free_bytes,
            host_available_bytes: inputs.host_available_bytes,
            h2d_bytes_per_second: inputs.h2d_bytes_per_second,
            d2h_bytes_per_second: inputs.d2h_bytes_per_second,
            copy_engines: inputs.copy_engines,
            disk_readers: inputs.disk_readers,
            host_registration_bytes: inputs.host_registration_bytes,
            service: inputs.service,
            reserves: inputs.reserves,
            span_rows: inputs.span_rows,
            wave_rows: inputs.wave_rows,
        }
    }

    fn inputs<'a>(service: &'a ServiceTable, reserves: &'a Reserves) -> PlanInputs<'a> {
        let geometry = geometry();
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )
        .unwrap();
        let slot = layout.slot_bytes();
        PlanInputs {
            geometry,
            layout,
            device_free_bytes: reserves.persistent_bytes(slot).unwrap() + 4 * slot,
            host_available_bytes: reserves.host_bytes(slot).unwrap() + 26 * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 5e8,
            copy_engines: 2,
            disk_readers: 2,
            host_registration_bytes: u64::MAX,
            service,
            reserves,
            span_rows: 4,
            wave_rows: 2,
        }
    }

    #[test]
    fn reserves_reduce_slots_and_over_reserve_names_the_quantity() {
        let service = service();
        let reserves = reserves();
        let inputs = inputs(&service, &reserves);
        let derived = capacity(&inputs, 0).unwrap();
        assert_eq!(derived.slots_per_layer, vec![2, 2]);
        assert_eq!(derived.host_blocks, 16);
        assert_eq!(derived.disk_bytes, 0);
        let small_host = PlanInputs {
            host_available_bytes: reserves.host_bytes(slot()).unwrap() + 8 * slot(),
            ..capacity_ref(&inputs)
        };
        let small = capacity(&small_host, 0).unwrap();
        assert_eq!(
            small.host_blocks, 7,
            "one block is reserved for swap write-backs"
        );
        assert_eq!(small.disk_bytes, (16 - 4 - 7) * slot());
        let tight = PlanInputs {
            device_free_bytes: reserves.persistent_bytes(slot()).unwrap() + 2 * slot() - 1,
            ..capacity_ref(&inputs)
        };
        let error = capacity(&tight, 0).unwrap_err().to_string();
        assert!(error.contains("device expert bytes"), "{error}");
        let over = PlanInputs {
            device_free_bytes: reserves.persistent_bytes(slot()).unwrap() - 1,
            ..tight
        };
        let error = capacity(&over, 0).unwrap_err().to_string();
        assert!(error.contains("persistent reserves"), "{error}");
        let spares = capacity(&inputs, 28).unwrap_err().to_string();
        assert!(spares.contains("reserve blocks"), "{spares}");
    }

    #[test]
    fn pinned_is_decided_from_the_measured_registration() {
        let service = service();
        let reserves = reserves();
        let inputs = inputs(&service, &reserves);
        let arena_bytes = 16 * slot();
        let eligible = PlanInputs {
            host_registration_bytes: arena_bytes,
            ..capacity_ref(&inputs)
        };
        assert!(capacity(&eligible, 0).unwrap().pinned);
        let short = PlanInputs {
            host_registration_bytes: arena_bytes - 1,
            ..capacity_ref(&inputs)
        };
        assert!(!capacity(&short, 0).unwrap().pinned);
        let none = PlanInputs {
            host_registration_bytes: 0,
            ..capacity_ref(&inputs)
        };
        assert!(!capacity(&none, 0).unwrap().pinned);
        let complete = PlanInputs {
            device_free_bytes: reserves.persistent_bytes(slot()).unwrap() + 8 * slot(),
            host_available_bytes: reserves.host_bytes(slot()).unwrap() + 16 * slot(),
            host_registration_bytes: 16 * slot(),
            ..capacity_ref(&inputs)
        };
        let full = capacity(&complete, 0).unwrap();
        assert_eq!(full.host_blocks, 16);
        assert_eq!(full.spare_blocks, 0);
        assert!(full.pinned);
        for (host_available_bytes, host_registration_bytes) in [
            (
                complete.host_available_bytes - 1,
                complete.host_registration_bytes,
            ),
            (
                complete.host_available_bytes,
                complete.host_registration_bytes - 1,
            ),
        ] {
            let bounded = capacity(
                &PlanInputs {
                    host_available_bytes,
                    host_registration_bytes,
                    ..capacity_ref(&complete)
                },
                0,
            )
            .unwrap();
            assert_eq!(bounded.host_blocks, 11);
            assert_eq!(bounded.spare_blocks, 1);
            assert_eq!(bounded.slots_per_layer, full.slots_per_layer);
            assert_eq!(bounded.span_rows, full.span_rows);
            assert!(bounded.pinned);
        }
    }

    #[test]
    fn capacity_sizes_from_the_fp8_layout_slot_bytes() {
        let service = service();
        let reserves = reserves();
        let geometry = geometry();
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Fp8W8a16,
            &geometry,
        )
        .unwrap();
        let slot = layout.slot_bytes();
        assert_eq!(slot, 128);
        let inputs = PlanInputs {
            geometry,
            layout,
            device_free_bytes: reserves.persistent_bytes(slot).unwrap() + 4 * slot,
            host_available_bytes: reserves.host_bytes(slot).unwrap() + 10 * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 5e8,
            copy_engines: 2,
            disk_readers: 2,
            host_registration_bytes: u64::MAX,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, 0).unwrap();
        assert_eq!(capacity.slot_bytes(), 128);
        assert_eq!(capacity.slots_per_layer, vec![2, 2]);
        assert_eq!(capacity.host_blocks, 9);
        assert_eq!(capacity.disk_bytes, 3 * 128);
    }

    #[test]
    fn host_blocks_cover_the_worst_case_span_borrow() {
        let service = service();
        let reserves = reserves();
        let inputs = inputs(&service, &reserves);
        assert_eq!(span_borrow_slots(&reserves, slot(), 4, 4).unwrap(), 3);
        let derived = capacity(&inputs, 0).unwrap();
        assert_eq!(derived.host_blocks, 16, "complete immutable host backing");
        assert_eq!(derived.spare_blocks, 0);
        assert_eq!(derived.disk_bytes, 0);
        assert_eq!(derived.disk_workers, 0);
        let partial = PlanInputs {
            host_available_bytes: reserves.host_bytes(slot()).unwrap() + 14 * slot(),
            ..capacity_ref(&inputs)
        };
        let borrowed_disk = capacity(&partial, 0).unwrap();
        assert_eq!(borrowed_disk.host_blocks, 13);
        assert_eq!(borrowed_disk.disk_bytes, 0);
        assert_eq!(borrowed_disk.disk_workers, inputs.disk_readers);
        let no_readers = PlanInputs {
            disk_readers: 0,
            ..capacity_ref(&partial)
        };
        assert!(capacity(&no_readers, 0).is_err());
        let small_host = PlanInputs {
            host_available_bytes: reserves.host_bytes(slot()).unwrap() + 8 * slot(),
            ..capacity_ref(&inputs)
        };
        let small = capacity(&small_host, 0).unwrap();
        assert_eq!(small.host_blocks, 7, "host bytes below the headroom target");
        assert_eq!(small.disk_bytes, 5 * slot());
    }

    #[test]
    fn fully_resident_capacity_is_clamped_to_the_routed_count() {
        let service = service();
        let reserves = reserves();
        let slot = slot();
        let mut inputs = inputs(&service, &reserves);
        inputs.device_free_bytes = reserves.persistent_bytes(slot).unwrap() + 16 * slot + 4_096;
        inputs.host_available_bytes = reserves.host_bytes(slot).unwrap() + 2 * slot;
        let derived = capacity(&inputs, 0).unwrap();
        assert_eq!(derived.slots_per_layer, vec![8, 8]);
        assert_eq!(
            derived.host_blocks, 2,
            "the borrow headroom keeps the available blocks without a decode reserve"
        );
        assert_eq!(derived.spare_blocks, 0);
        assert_eq!(derived.disk_bytes, 0);
        assert_eq!(derived.disk_workers, inputs.disk_readers);
        inputs.host_available_bytes = reserves.host_bytes(slot).unwrap() + 16 * slot;
        inputs.host_registration_bytes = 16 * slot;
        let headroom =
            span_borrow_slots(&reserves, slot, inputs.span_rows, inputs.span_rows).unwrap();
        let ample = capacity(&inputs, 0).unwrap();
        assert_eq!(ample.slots_per_layer, vec![8, 8]);
        assert_eq!(ample.host_blocks, headroom);
        assert_eq!(ample.spare_blocks, 0);
        assert_eq!(ample.disk_bytes, 0);
        assert_eq!(ample.disk_workers, 0);
        assert!(ample.pinned);
    }

    #[test]
    fn initial_residency_ranks_by_frequency_and_ties_by_id() {
        let frequency = vec![vec![0.0; 8], vec![1.0; 8]];
        let initial = initial_residency(&frequency, &[2, 3], &geometry()).unwrap();
        assert_eq!(initial[0], vec![0, 1]);
        assert_eq!(initial[1], vec![0, 1, 2]);
        let mut bad = frequency.clone();
        bad[1][2] = f64::NAN;
        assert!(initial_residency(&bad, &[2, 3], &geometry()).is_err());
        assert!(initial_residency(&frequency, &[2], &geometry()).is_err());
    }

    #[test]
    fn service_table_interpolates_and_selects_workers() {
        let table = service();
        assert_eq!(table.seconds(1, 1, 1).unwrap(), 0.2);
        assert_eq!(table.seconds(1, 2, 1).unwrap(), 0.4);
        assert!((table.seconds(1, 3, 1).unwrap_err().to_string()).contains("no bracketing cell"));
        assert_eq!(table.worker_count().unwrap(), 2);
        let mut mixed = Vec::new();
        for (workers, experts, seconds) in [
            (1u32, 1u32, 0.10),
            (1, 2, 0.16),
            (1, 3, 0.24),
            (1, 4, 0.32),
            (2, 1, 0.08),
            (2, 2, 0.14),
            (2, 3, 0.15),
            (2, 4, 0.24),
            (4, 1, 0.14),
            (4, 2, 0.12),
            (4, 3, 0.16),
            (4, 4, 0.20),
        ] {
            mixed.push(ServiceCell {
                workers,
                rows: 1,
                experts,
                seconds,
            });
        }
        let mixed = ServiceTable::new("mixed", mixed).unwrap();
        assert_eq!(mixed.worker_count().unwrap(), 2);
        assert_eq!(mixed.pool_worker_count().unwrap(), 4);
        let participants = mixed.participant_table(1, 4).unwrap();
        assert_eq!(participants.worker_count(), 4);
        assert_eq!(participants.participant(1, 1).unwrap(), 2);
        assert_eq!(participants.participant(1, 2).unwrap(), 4);
        assert_eq!(participants.participant(1, 3).unwrap(), 2);
        assert_eq!(participants.participant(1, 4).unwrap(), 4);
        let extended = mixed.participant_table(4, 4).unwrap();
        assert_eq!(extended.participant(4, 1).unwrap(), 2);
        assert_eq!(extended.participant(4, 3).unwrap(), 2);
        assert!(extended.participant(5, 1).is_err());
        assert!(
            ParticipantTable::uniform(2, 4, 2)
                .unwrap()
                .participant(3, 1)
                .is_err()
        );
        assert!(
            ServiceTable::new(
                "x",
                vec![ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: f64::NAN
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn f_seed_is_twelve_normalized_prefill_copies() {
        let scores = seed_scores(&frequency(), &geometry()).unwrap();
        for layer in &scores {
            let sum: f64 = layer.iter().sum();
            assert!((sum - F_SEED_SCALE * f64::from(geometry().top_k)).abs() < 1e-9);
        }
        let zero = vec![vec![0.0; 8], vec![0.0; 8]];
        assert!(seed_scores(&zero, &geometry()).is_err());
    }

    #[test]
    fn admission_budget_is_one_miss_per_layer_priced_by_the_copy_rates() {
        let geometry = geometry();
        let table = ServiceTable::new(
            "test",
            vec![
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 0.0002,
                },
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 2,
                    seconds: 0.0004,
                },
            ],
        )
        .unwrap();
        let budget = admission_budget(&geometry, &table, 1e9, 1e9, 2).unwrap();
        assert_eq!(budget, 400_000);
        assert_eq!(
            Admission {
                budget_bytes_per_token: budget
            }
            .hard_cap(slot())
            .unwrap(),
            400_000 / slot() as u32
        );
        let serial = admission_budget(&geometry, &table, 1e9, 1e9, 1).unwrap();
        assert_eq!(serial, 200_000);
        assert!(admission_budget(&geometry, &table, 0.0, 1e9, 2).is_err());
        assert!(admission_budget(&geometry, &table, 1e9, 1e9, 0).is_err());
    }

    #[test]
    fn span_borrow_uses_the_prompt_and_span_minimum() {
        let reserves = reserves();
        assert_eq!(span_borrow_slots(&reserves, slot(), 4, 4).unwrap(), 3);
        assert_eq!(span_borrow_slots(&reserves, slot(), 2, 4).unwrap(), 2);
        assert_eq!(span_borrow_slots(&reserves, slot(), 8, 4).unwrap(), 3);
        assert!(span_borrow_slots(&reserves, slot(), 0, 4).is_err());
        assert!(span_borrow_slots(&reserves, slot(), 4, 0).is_err());
    }

    #[test]
    fn select_span_picks_the_largest_feasible_and_breaks_ties_by_borrow() {
        let service = service();
        let reserves = reserves();
        let mut inputs = inputs(&service, &reserves);
        inputs.device_free_bytes = reserves.persistent_bytes(slot()).unwrap() + 4 * slot();
        let mut tight = reserves;
        tight.residual_row_bytes = 128;
        let candidates = vec![
            SpanCandidate {
                span_rows: 4,
                wave_rows: 2,
                reserves,
            },
            SpanCandidate {
                span_rows: 2,
                wave_rows: 2,
                reserves: tight,
            },
            SpanCandidate {
                span_rows: 2,
                wave_rows: 2,
                reserves,
            },
        ];
        let (index, capacity) = select_span(&inputs, 0, 4, &candidates).unwrap();
        assert_eq!(index, 2);
        assert_eq!(capacity.slots_total(), 4);
        let error = select_span(&inputs, 0, 4, &candidates[..1])
            .unwrap_err()
            .to_string();
        assert!(error.contains("no feasible span candidate"), "{error}");
    }

    fn slot_layout(
        geometry: ExpertGeometry,
        slots_per_layer: Vec<u32>,
        host_blocks: u32,
    ) -> SlotLayout {
        let slot = slot();
        let capacity = Capacity {
            layout: crate::experts::catalogue::ExpertLayout::new(
                crate::model_spec::ExpertFormat::Bf16,
                &geometry,
            )
            .unwrap(),
            span_rows: 4,
            wave_rows: 2,
            device_expert_bytes: slots_per_layer.iter().map(|&n| u64::from(n)).sum::<u64>() * slot,
            peer_expert_bytes: 0,
            host_expert_bytes: u64::from(host_blocks) * slot,
            gpu0_layers: geometry.layers,
            slots_per_layer,
            host_blocks,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 1,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: false,
        };
        SlotLayout::new(geometry, &capacity).unwrap()
    }

    #[test]
    fn peer_capacity_splits_layers_and_prices_measured_hops() {
        let service = service();
        let reserves = reserves();
        let slot = slot();
        let mut inputs = inputs(&service, &reserves);
        inputs.device_free_bytes = reserves.persistent_bytes(slot).unwrap() + 11 * slot;
        let fast_prefill = [
            (8 * 1024 * 1024, 23e9, 23e9),
            (68 * 1024 * 1024, 23e9, 23e9),
        ];
        let peer = PeerInputs {
            device_free_bytes: 10 * slot,
            device_reserved_bytes: 2 * slot,
            decode_transfer: (4_096, 200e6, 200e6),
            prefill_transfers: &fast_prefill,
        };
        let derived = capacity_with_peer(&inputs, 0, Some(peer)).unwrap();
        assert_eq!(derived.gpu0_layers, 1);
        assert_eq!(derived.slots_per_layer, vec![8, 8]);
        assert_eq!(derived.gpu0_slots(), 8);
        assert_eq!(derived.peer_slots(), 8);
        assert_eq!(derived.disk_bytes, 0);
        let slow_decode = capacity_with_peer(
            &inputs,
            0,
            Some(PeerInputs {
                decode_transfer: (4_096, 100.0, 100.0),
                ..peer
            }),
        )
        .unwrap();
        assert_eq!(slow_decode.gpu0_layers, 2);
        assert_eq!(slow_decode.slots_per_layer, vec![6, 5]);
        assert_eq!(slow_decode.peer_slots(), 0);
        let slow_prefill = [
            (8 * 1024 * 1024, 100.0, 100.0),
            (68 * 1024 * 1024, 100.0, 100.0),
        ];
        let rejected = capacity_with_peer(
            &inputs,
            0,
            Some(PeerInputs {
                prefill_transfers: &slow_prefill,
                ..peer
            }),
        )
        .unwrap();
        assert_eq!(rejected.gpu0_layers, 2);
        assert_eq!(rejected.slots_per_layer, vec![6, 5]);
        assert_eq!(rejected.peer_slots(), 0);

        let a4000_geom = ExpertGeometry {
            first_layer: 0,
            layers: 40,
            experts: 256,
            top_k: 8,
            hidden: 2048,
            intermediate: 512,
        };
        let a4000_layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Fp8W8a16,
            &a4000_geom,
        )
        .unwrap();
        let fp8_slot = a4000_layout.slot_bytes();
        assert_eq!(fp8_slot, 3_151_872);
        let a4000_service = ServiceTable::new(
            "a4000-7950x",
            vec![
                ServiceCell {
                    workers: 8,
                    rows: 1,
                    experts: 1,
                    seconds: 0.00025,
                },
                ServiceCell {
                    workers: 8,
                    rows: 1,
                    experts: 8,
                    seconds: 0.002,
                },
            ],
        )
        .unwrap();
        let a4000_inputs = PlanInputs {
            geometry: a4000_geom,
            layout: a4000_layout,
            device_free_bytes: reserves.persistent_bytes(fp8_slot).unwrap() + 3_500 * fp8_slot,
            host_available_bytes: reserves.host_bytes(fp8_slot).unwrap() + 4_096 * fp8_slot,
            h2d_bytes_per_second: 26.82e9,
            d2h_bytes_per_second: 24.0e9,
            copy_engines: 2,
            disk_readers: 2,
            host_registration_bytes: 4_096 * fp8_slot,
            service: &a4000_service,
            reserves: &reserves,
            span_rows: 4096,
            wave_rows: 256,
        };
        let a4000_prefill = [
            (8 * 1024 * 1024, 5.1e9, 5.1e9),
            (68 * 1024 * 1024, 5.1e9, 5.1e9),
        ];
        let a4000_peer = PeerInputs {
            device_free_bytes: 19 * 256 * fp8_slot + 1_300_000_000,
            device_reserved_bytes: 1_300_000_000 + 128 * fp8_slot,
            decode_transfer: (4_096, 330e6, 330e6),
            prefill_transfers: &a4000_prefill,
        };
        let a4000 = capacity_with_peer(&a4000_inputs, 0, Some(a4000_peer)).unwrap();
        assert_eq!(a4000.gpu0_layers, 22);
        assert_eq!(a4000.gpu0_slots(), 3_500);
        assert_eq!(a4000.peer_slots(), 18 * 256);
        assert_eq!(a4000.slots_total(), 3_500 + 18 * 256);
        assert_eq!(a4000.disk_bytes, 0);
        assert!(a4000.slots_per_layer[..22].iter().all(|&s| s < 256));
        assert!(a4000.slots_per_layer[22..].iter().all(|&s| s == 256));

        let a4000_30g_inputs = PlanInputs {
            host_available_bytes: reserves.host_bytes(fp8_slot).unwrap() + 8_000 * fp8_slot,
            host_registration_bytes: 8_000 * fp8_slot,
            ..a4000_inputs.clone()
        };
        let a4000_30g = capacity_with_peer(&a4000_30g_inputs, 13, Some(a4000_peer)).unwrap();
        assert_eq!(a4000_30g.gpu0_layers, 22);
        assert_eq!(a4000_30g.host_blocks, 22 * 256);
        assert_eq!(a4000_30g.spare_blocks, 0);
        assert!(a4000_30g.pinned);
        let a4000_30g_ledger = TierLedger::empty(
            SlotLayout::new(a4000_geom, &a4000_30g).unwrap(),
            a4000_30g.host_blocks,
        )
        .unwrap();
        assert!(a4000_30g_ledger.has_full_host_backing());

        // 18 removed misses * 0.25 ms = 4.5 ms/tok; old single_gpu_misses (6740/256 * 0.25 ms) = 6.58 ms/tok.
        // At 28 MB/s bounce (18 * 8240 / 28e6 = 5.30 ms/tok), exact removed_misses rejects peer while old overcount would accept.
        let overpriced_partial = capacity_with_peer(
            &a4000_inputs,
            0,
            Some(PeerInputs {
                decode_transfer: (4_096, 28e6, 28e6),
                ..a4000_peer
            }),
        )
        .unwrap();
        assert_eq!(overpriced_partial.gpu0_layers, 40);
        assert_eq!(overpriced_partial.peer_slots(), 0);
    }

    #[test]
    fn peer_free_below_reserve_selects_gpu0_plan() {
        let reserves = reserves();
        let geom = ExpertGeometry {
            first_layer: 0,
            layers: 40,
            experts: 256,
            top_k: 8,
            hidden: 2048,
            intermediate: 512,
        };
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Fp8W8a16,
            &geom,
        )
        .unwrap();
        let fp8_slot = layout.slot_bytes();
        let service = ServiceTable::new(
            "a4000-7950x",
            vec![
                ServiceCell {
                    workers: 8,
                    rows: 1,
                    experts: 1,
                    seconds: 0.00025,
                },
                ServiceCell {
                    workers: 8,
                    rows: 1,
                    experts: 8,
                    seconds: 0.002,
                },
            ],
        )
        .unwrap();
        let inputs = PlanInputs {
            geometry: geom,
            layout,
            device_free_bytes: reserves.persistent_bytes(fp8_slot).unwrap() + 3_500 * fp8_slot,
            host_available_bytes: reserves.host_bytes(fp8_slot).unwrap() + 4_096 * fp8_slot,
            h2d_bytes_per_second: 26.82e9,
            d2h_bytes_per_second: 24.0e9,
            copy_engines: 2,
            disk_readers: 2,
            host_registration_bytes: 4_096 * fp8_slot,
            service: &service,
            reserves: &reserves,
            span_rows: 4096,
            wave_rows: 256,
        };
        let prefill = [
            (8 * 1024 * 1024, 5.1e9, 5.1e9),
            (68 * 1024 * 1024, 5.1e9, 5.1e9),
        ];
        let peer_reserved = 1_300_000_000 + 128 * fp8_slot;
        let candidate = SpanCandidate {
            span_rows: inputs.span_rows,
            wave_rows: inputs.wave_rows,
            reserves,
        };
        let (_, single_gpu) =
            select_span(&inputs, 13, 4096, std::slice::from_ref(&candidate)).unwrap();
        assert_eq!(single_gpu.gpu0_layers, 40);
        assert_eq!(single_gpu.peer_expert_bytes, 0);
        assert_eq!(single_gpu.peer_slots(), 0);
        for free_bytes in [
            0,
            peer_reserved - 1,
            peer_reserved,
            peer_reserved + 255 * fp8_slot,
        ] {
            let (index, selected) = select_span_with_peer(
                &inputs,
                13,
                4096,
                std::slice::from_ref(&candidate),
                Some(PeerInputs {
                    device_free_bytes: free_bytes,
                    device_reserved_bytes: peer_reserved,
                    decode_transfer: (4_096, 330e6, 330e6),
                    prefill_transfers: &prefill,
                }),
            )
            .unwrap();
            assert_eq!(index, 0);
            assert_eq!(selected.gpu0_layers, single_gpu.gpu0_layers);
            assert_eq!(selected.peer_expert_bytes, 0);
            assert_eq!(selected.peer_slots(), 0);
            assert_eq!(selected.slots_per_layer, single_gpu.slots_per_layer);
            assert_eq!(selected.host_blocks, single_gpu.host_blocks);
            assert_eq!(selected.spare_blocks, single_gpu.spare_blocks);
            assert_eq!(selected.disk_bytes, single_gpu.disk_bytes);
            assert_eq!(selected.pinned, single_gpu.pinned);
            assert_eq!(
                SlotLayout::new(geom, &selected)
                    .unwrap()
                    .peer_resident_bytes(),
                0
            );
        }
        let (_, one_peer_layer) = select_span_with_peer(
            &inputs,
            13,
            4096,
            std::slice::from_ref(&candidate),
            Some(PeerInputs {
                device_free_bytes: peer_reserved + 256 * fp8_slot,
                device_reserved_bytes: peer_reserved,
                decode_transfer: (4_096, 330e6, 330e6),
                prefill_transfers: &prefill,
            }),
        )
        .unwrap();
        assert_eq!(one_peer_layer.gpu0_layers, 39);
        assert_eq!(one_peer_layer.peer_slots(), 256);
    }

    fn ledger_for(
        geometry: ExpertGeometry,
        slots: Vec<u32>,
        host_blocks: u32,
        initial: &[Vec<u32>],
    ) -> TierLedger {
        TierLedger::new(
            slot_layout(geometry, slots, host_blocks),
            host_blocks,
            initial,
        )
        .unwrap()
    }

    #[test]
    fn ledger_slot_ids_follow_the_rank_major_layout() {
        let geometry = geometry();
        let initial = initial_residency(&frequency(), &[2, 2], &geometry).unwrap();
        let ledger = ledger_for(geometry, vec![2, 2], 5, &initial);
        assert_eq!(ledger.slot(0, initial[0][0]).unwrap(), 0);
        assert_eq!(ledger.slot(1, initial[1][0]).unwrap(), 1);
        assert_eq!(ledger.slot(0, initial[0][1]).unwrap(), 2);
        assert_eq!(ledger.slot(1, initial[1][1]).unwrap(), 3);
        assert!(ledger.check().is_ok());
    }

    #[test]
    fn ledger_install_fills_empty_positions_in_layout_order() {
        let geometry = geometry();
        let layout = slot_layout(geometry, vec![2, 2], 5);
        let mut ledger = TierLedger::empty(layout, 5).unwrap();
        assert_eq!(ledger.empty_slot(0).unwrap(), Some(0));
        assert_eq!(ledger.install(0, 3).unwrap(), 0);
        assert_eq!(ledger.install(1, 3).unwrap(), 1);
        assert_eq!(ledger.install(0, 4).unwrap(), 2);
        assert_eq!(ledger.install(1, 4).unwrap(), 3);
        assert_eq!(ledger.empty_slot(0).unwrap(), None);
        assert!(ledger.install(0, 5).is_err());
        assert!(ledger.check().is_ok());
    }

    #[test]
    fn ledger_borrows_an_empty_tail_position_without_evicting_an_occupant() {
        let geometry = geometry();
        let initial = initial_residency(&frequency(), &[2, 2], &geometry).unwrap();
        let mut ledger = ledger_for(geometry, vec![2, 2], 5, &initial);
        let tail = initial[1][1];
        ledger.evict_gpu(1, tail).unwrap();
        let borrow = ledger.borrow_tail(1, 5).unwrap();
        assert!(borrow.experts.is_empty());
        assert_eq!((borrow.to_host, borrow.to_disk), (0, 0));
        assert_eq!(ledger.tier(0, initial[0][0]).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(0, initial[0][1]).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(1, initial[1][0]).unwrap(), Tier::Gpu);
        assert!(ledger.check().is_ok());
        assert_eq!(ledger.refill_tail().unwrap().len(), 1);
        assert!(ledger.check().is_ok());
        let again = ledger.borrow_tail(1, 5).unwrap();
        assert!(again.experts.is_empty());
    }

    #[test]
    fn ledger_borrow_retains_every_layers_first_slot() {
        let geometry = geometry();
        let initial = initial_residency(&frequency(), &[2, 1], &geometry).unwrap();
        let mut ledger = ledger_for(geometry, vec![2, 1], 5, &initial);
        let borrow = ledger.borrow_tail(1, 5).unwrap();
        assert_eq!(borrow.experts, vec![(0, initial[0][1])]);
        assert_eq!(ledger.tier(0, initial[0][0]).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(1, initial[1][0]).unwrap(), Tier::Gpu);
        let before = ledger.unique_bytes().unwrap();
        assert!(ledger.borrow_tail(2, 5).is_err());
        assert_eq!(ledger.unique_bytes().unwrap(), before);
        assert!(ledger.check().is_ok());
    }

    #[test]
    fn ledger_borrow_takes_the_layout_physical_tail() {
        let geometry = geometry();
        let slot = slot();
        let initial = initial_residency(&frequency(), &[2, 2], &geometry).unwrap();
        let mut ledger = ledger_for(geometry, vec![2, 2], 2, &initial);
        let borrow = ledger.borrow_tail(2, 2).unwrap();
        let mut evicted = borrow.experts.clone();
        evicted.sort_unstable();
        assert_eq!(evicted, vec![(0, initial[0][1]), (1, initial[1][1])]);
        assert_eq!((borrow.to_host, borrow.to_disk), (2, 0));
        assert_eq!(borrow.d2h_bytes, 2 * slot);
        let before = ledger.unique_bytes().unwrap();
        let error = ledger.borrow_tail(1, 2).unwrap_err().to_string();
        assert!(error.contains("already borrowed"), "{error}");
        assert_eq!(ledger.unique_bytes().unwrap(), before);
        assert_eq!(ledger.refill_tail().unwrap().len(), 2);
        assert!(ledger.check().is_ok());
        let again = ledger.borrow_tail(2, 2).unwrap();
        assert!(again.experts.is_empty());
        assert_eq!(ledger.refill_tail().unwrap().len(), 2);
        let mut starved = ledger_for(geometry, vec![2, 2], 0, &initial);
        let borrow = starved.borrow_tail(2, 0).unwrap();
        assert_eq!((borrow.to_host, borrow.to_disk), (0, 2));
        assert_eq!(borrow.disk_bytes, 2 * slot);
        assert!(starved.check().is_ok());
    }

    #[test]
    fn ledger_keeps_tiers_disjoint_and_bytes_exact() {
        let geometry = geometry();
        let slot = slot();
        let initial = initial_residency(&frequency(), &[2, 2], &geometry).unwrap();
        let mut ledger = ledger_for(geometry, vec![2, 2], 5, &initial);
        let routed = 16 * slot;
        assert_eq!(
            ledger.unique_bytes().unwrap() + ledger.disk_bytes().unwrap(),
            routed
        );
        assert_eq!(ledger.tier(0, initial[0][0]).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(0, 7).unwrap(), Tier::Disk);
        ledger.admit_disk(0, 7, initial[0][0], true).unwrap();
        assert_eq!(ledger.tier(0, 7).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(0, initial[0][0]).unwrap(), Tier::Host);
        ledger.admit_host(0, initial[0][0], initial[0][1]).unwrap();
        assert_eq!(ledger.tier(0, initial[0][0]).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(0, initial[0][1]).unwrap(), Tier::Host);
        ledger.admit_disk(1, 7, initial[1][0], true).unwrap();
        assert_eq!(ledger.tier(1, 7).unwrap(), Tier::Gpu);
        assert_eq!(ledger.tier(1, initial[1][0]).unwrap(), Tier::Host);
        ledger.evict_gpu(0, 7).unwrap();
        assert_eq!(ledger.tier(0, 7).unwrap(), Tier::Disk);
        assert_eq!(
            ledger.unique_bytes().unwrap() + ledger.disk_bytes().unwrap(),
            routed
        );
        for layer in 0..2u32 {
            for expert in 0..8u32 {
                let tier = ledger.tier(layer, expert).unwrap();
                assert!(matches!(tier, Tier::Gpu | Tier::Host | Tier::Disk));
            }
        }
        assert!(ledger.check().is_ok());
        assert!(ledger.admit_host(1, initial[1][1], initial[1][1]).is_err());
    }
}
