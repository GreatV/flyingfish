//! Setup-derived expert residency capacity and per-request placement policy.
use crate::experts::catalogue::ExpertGeometry;
use anyhow::{Context, Result, ensure};

/// Decayed-frequency decay of the admission score.
pub const F_DECAY: f64 = 0.98;
/// Minimum decayed frequency of an admitted candidate.
pub const THETA_MIN: f64 = 1.05;
/// F seed multiplier over the request's normalized prefill frequency.
pub const F_SEED_SCALE: f64 = 12.0;
/// Normalized prefill frequency sums to the top-k count per layer.
pub const FREQUENCY_SUM: f64 = 8.0;

/// One measured CPU expert-service cell.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub fn device_bytes(&self, geometry: &ExpertGeometry, span_rows: u32) -> Result<u64> {
        let per_row = self
            .residual_row_bytes
            .checked_add(self.normalized_row_bytes)
            .and_then(|n| n.checked_add(self.contribution_row_bytes))
            .and_then(|n| n.checked_add(self.done_row_bytes))
            .and_then(|n| n.checked_add(self.route_row_bytes))
            .context("span reserve overflow")?;
        let span = u64::from(span_rows)
            .checked_mul(per_row)
            .context("span buffer overflow")?;
        let staging = u64::from(self.prefill_staging_slots)
            .checked_mul(geometry.slot_bytes()?)
            .context("prefill staging overflow")?;
        self.device_fixed_bytes
            .checked_add(span)
            .and_then(|n| n.checked_add(staging))
            .context("device reserve overflow")
    }

    pub fn host_bytes(&self, geometry: &ExpertGeometry) -> Result<u64> {
        let ring = u64::from(self.disk_ring_slots)
            .checked_mul(geometry.slot_bytes()?)
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
    pub device_free_bytes: u64,
    pub host_available_bytes: u64,
    pub h2d_bytes_per_second: f64,
    pub d2h_bytes_per_second: f64,
    /// Device async copy engines available to expert transfers.
    pub copy_engines: u32,
    pub disk_readers: u32,
    pub service: &'a ServiceTable,
    pub reserves: &'a Reserves,
    pub span_rows: u32,
    pub wave_rows: u32,
}

/// Setup capacity derived for one admission cap.
#[derive(Clone, Debug)]
pub struct Capacity {
    pub span_rows: u32,
    pub wave_rows: u32,
    pub device_expert_bytes: u64,
    pub host_expert_bytes: u64,
    pub slots_per_layer: Vec<u32>,
    /// Usable host blocks after reserving the per-token spare blocks.
    pub host_blocks: u32,
    /// Spare host blocks used by admission write-backs.
    pub spare_blocks: u32,
    pub disk_bytes: u64,
    pub worker_count: u32,
    pub disk_workers: u32,
    pub prefill_staging_slots: u32,
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

/// Setup capacity from the reserves-first subtraction for one admission cap.
pub fn capacity(inputs: &PlanInputs<'_>, admission_cap: u32) -> Result<Capacity> {
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
    let device_reserved = inputs.reserves.device_bytes(&geometry, inputs.span_rows)?;
    let host_reserved = inputs.reserves.host_bytes(&geometry)?;
    ensure!(
        inputs.device_free_bytes >= device_reserved,
        "span/staging reserves {device_reserved} B exceed device free bytes {}",
        inputs.device_free_bytes
    );
    ensure!(
        inputs.host_available_bytes >= host_reserved,
        "host reserves {host_reserved} B exceed host available bytes {}",
        inputs.host_available_bytes
    );
    let slot_bytes = geometry.slot_bytes()?;
    let device_expert_bytes = inputs.device_free_bytes - device_reserved;
    let host_expert_bytes = inputs.host_available_bytes - host_reserved;
    let total_slots = device_expert_bytes / slot_bytes;
    ensure!(
        total_slots >= u64::from(geometry.layers),
        "device expert bytes {device_expert_bytes} hold {total_slots} slots, below one per layer ({})",
        geometry.layers
    );
    let slots_per_layer = equal_slots(total_slots, geometry.layers)?;
    let routed_slots = u64::from(geometry.layers)
        .checked_mul(u64::from(geometry.experts))
        .context("routed expert count overflow")?;
    let host_block_slots = host_expert_bytes / slot_bytes;
    ensure!(
        host_block_slots >= u64::from(admission_cap),
        "host blocks {host_block_slots} cannot hold {admission_cap} spare blocks"
    );
    let host_blocks = (host_block_slots - u64::from(admission_cap)).min(routed_slots - total_slots);
    let host_blocks = u32::try_from(host_blocks).context("host block count exceeds u32")?;
    let unique_slots = total_slots
        .checked_add(u64::from(host_blocks))
        .context("unique expert count overflow")?;
    ensure!(
        unique_slots <= routed_slots,
        "unique expert slots {unique_slots} exceed the routed total {routed_slots}"
    );
    let disk_bytes = (routed_slots - unique_slots)
        .checked_mul(slot_bytes)
        .context("disk tier bytes overflow")?;
    let worker_count = inputs.service.worker_count()?;
    let disk_workers = if disk_bytes > 0 {
        ensure!(inputs.disk_readers > 0, "no measured disk readers");
        inputs.disk_readers
    } else {
        0
    };
    Ok(Capacity {
        span_rows: inputs.span_rows,
        wave_rows: inputs.wave_rows,
        device_expert_bytes,
        host_expert_bytes,
        slots_per_layer,
        host_blocks,
        spare_blocks: admission_cap,
        disk_bytes,
        worker_count,
        disk_workers,
        prefill_staging_slots: inputs.reserves.prefill_staging_slots,
    })
}

/// Tier of one expert in the provenance ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Gpu,
    Host,
    Disk,
}

/// Exact tier membership and byte accounting; the capacity/provenance control.
pub struct TierLedger {
    geometry: ExpertGeometry,
    slots_per_layer: Vec<u32>,
    host_blocks: u32,
    gpu: Vec<Vec<bool>>,
    host: Vec<Vec<bool>>,
}

impl TierLedger {
    /// Starts from a request's initial GPU set; the host tier starts empty.
    pub fn new(
        geometry: ExpertGeometry,
        slots_per_layer: Vec<u32>,
        host_blocks: u32,
        initial: &[Vec<u32>],
    ) -> Result<Self> {
        ensure!(
            slots_per_layer.len() == geometry.layers as usize
                && initial.len() == geometry.layers as usize,
            "ledger layer count mismatch"
        );
        let mut gpu = vec![vec![false; geometry.experts as usize]; geometry.layers as usize];
        for (layer, set) in initial.iter().enumerate() {
            ensure!(
                set.len() as u32 == slots_per_layer[layer],
                "layer {layer} initial set has {} experts for {} slots",
                set.len(),
                slots_per_layer[layer]
            );
            for &expert in set {
                ensure!(
                    expert < geometry.experts,
                    "initial expert {expert} outside {}",
                    geometry.experts
                );
                ensure!(
                    !gpu[layer][expert as usize],
                    "duplicate initial expert {expert}"
                );
                gpu[layer][expert as usize] = true;
            }
        }
        let ledger = Self {
            geometry,
            slots_per_layer,
            host_blocks,
            gpu,
            host: vec![vec![false; geometry.experts as usize]; geometry.layers as usize],
        };
        ledger.check()?;
        Ok(ledger)
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

    fn check_index(&self, layer: u32, expert: u32) -> Result<()> {
        ensure!(
            layer < self.geometry.layers,
            "expert layer {layer} outside {}",
            self.geometry.layers
        );
        ensure!(
            expert < self.geometry.experts,
            "expert {expert} outside {}",
            self.geometry.experts
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

    /// Admits a host expert into a GPU slot, writing the victim back into its block.
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
        self.host[l][e] = false;
        self.gpu[l][e] = true;
        self.gpu[l][v] = false;
        self.host[l][v] = true;
        self.check()
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
        self.gpu[l][e] = true;
        self.gpu[l][v] = false;
        if host_block_free {
            self.host[l][v] = true;
        }
        self.check()
    }

    /// Drops a GPU-only expert to the disk tier without a host copy.
    pub fn evict_gpu(&mut self, layer: u32, expert: u32) -> Result<()> {
        self.check_index(layer, expert)?;
        let (l, e) = (layer as usize, expert as usize);
        ensure!(
            self.gpu[l][e],
            "expert {expert} of layer {layer} is not GPU resident"
        );
        self.gpu[l][e] = false;
        self.check()
    }

    pub fn free_host_block(&self) -> bool {
        self.free_host_blocks() > 0
    }

    pub fn unique_bytes(&self) -> Result<u64> {
        let slots = self
            .gpu
            .iter()
            .chain(self.host.iter())
            .flat_map(|layer| layer.iter())
            .filter(|resident| **resident)
            .count();
        u64::try_from(slots)?
            .checked_mul(self.geometry.slot_bytes()?)
            .context("unique bytes overflow")
    }

    pub fn disk_bytes(&self) -> Result<u64> {
        let routed = u64::from(self.geometry.layers)
            .checked_mul(u64::from(self.geometry.experts))
            .context("routed expert count overflow")?;
        let slot = self.geometry.slot_bytes()?;
        let used = self.unique_bytes()? / slot;
        routed
            .checked_sub(used)
            .and_then(|n| n.checked_mul(slot))
            .context("disk bytes overflow")
    }

    /// Duplicate residency is zero by construction.
    pub fn duplicate_bytes(&self) -> u64 {
        0
    }

    pub fn check(&self) -> Result<()> {
        let mut host_used = 0u32;
        for (layer, slots) in self.slots_per_layer.iter().enumerate() {
            let gpu = self.gpu[layer].iter().filter(|resident| **resident).count();
            ensure!(
                gpu <= *slots as usize,
                "layer {layer} holds {gpu} GPU experts for {slots} slots"
            );
            for expert in 0..self.geometry.experts as usize {
                ensure!(
                    !(self.gpu[layer][expert] && self.host[layer][expert]),
                    "expert {expert} of layer {layer} is both GPU and host resident"
                );
            }
            host_used += self.host[layer]
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
    pub fn hard_cap(&self, slot_bytes: u64) -> u32 {
        (self.budget_bytes_per_token / slot_bytes) as u32
    }
}

/// Predicted decode latency of one admission budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub hit: f64,
    pub cpu_seconds: f64,
    pub h2d_seconds: f64,
    pub d2h_seconds: f64,
    pub overlap_seconds: f64,
    pub serial_seconds: f64,
    pub admitted_per_token: f64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample_row(frequency: &[f64], top_k: u32, rng: &mut Rng, chosen: &mut Vec<u32>) {
    chosen.clear();
    let mut weights = frequency.to_vec();
    for _ in 0..top_k {
        let total: f64 = weights.iter().sum();
        let pick = if total > 0.0 {
            let point = rng.unit() * total;
            let mut cumulative = 0.0;
            let mut picked = None;
            for (expert, weight) in weights.iter().enumerate() {
                if *weight <= 0.0 {
                    continue;
                }
                cumulative += weight;
                if point <= cumulative {
                    picked = Some(expert as u32);
                    break;
                }
            }
            picked.unwrap_or_else(|| {
                (0..weights.len() as u32)
                    .rev()
                    .find(|&expert| weights[expert as usize] > 0.0)
                    .expect("a positive weight exists")
            })
        } else {
            (0..weights.len() as u32)
                .find(|&expert| !chosen.contains(&expert))
                .expect("an unchosen expert exists")
        };
        chosen.push(pick);
        weights[pick as usize] = 0.0;
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
                .map(|value| F_SEED_SCALE * value * FREQUENCY_SUM / total)
                .collect(),
        );
    }
    Ok(scores)
}

/// One decode simulation request.
pub struct Simulation<'a> {
    pub geometry: ExpertGeometry,
    pub slots_per_layer: &'a [u32],
    pub host_blocks: u32,
    pub spare_blocks: u32,
    pub initial: &'a [Vec<u32>],
    pub frequency: &'a [Vec<f64>],
    pub admission: Admission,
    pub service: &'a ServiceTable,
    pub h2d_bytes_per_second: f64,
    pub d2h_bytes_per_second: f64,
    pub copy_engines: u32,
    pub steps: u32,
    pub warmup: u32,
    pub seed: u64,
}

/// Decode-latency prediction over a deterministic synthetic stream drawn from `frequency`.
pub fn simulate(input: Simulation<'_>) -> Result<Prediction> {
    let Simulation {
        geometry,
        slots_per_layer,
        host_blocks,
        spare_blocks,
        initial,
        frequency,
        admission,
        service,
        h2d_bytes_per_second,
        d2h_bytes_per_second,
        copy_engines,
        steps,
        warmup,
        seed,
    } = input;
    validate_frequency(frequency, &geometry)?;
    ensure!(
        slots_per_layer.len() == geometry.layers as usize
            && initial.len() == geometry.layers as usize,
        "slot/initial layer count mismatch"
    );
    ensure!(steps > warmup && warmup > 0, "invalid simulation window");
    ensure!(
        geometry.top_k <= geometry.experts,
        "top_k {} exceeds {} experts",
        geometry.top_k,
        geometry.experts
    );
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
    let layers = geometry.layers as usize;
    let experts = geometry.experts as usize;
    let workers = service.worker_count()?;
    let slot_bytes = geometry.slot_bytes()?;
    let hard_cap = admission.hard_cap(slot_bytes);
    let mut resident: Vec<Vec<bool>> = Vec::with_capacity(layers);
    let mut host: Vec<Vec<bool>> = Vec::with_capacity(layers);
    let mut score = seed_scores(frequency, &geometry)?;
    let mut host_used = 0u32;
    for (layer, set) in initial.iter().enumerate() {
        ensure!(
            set.len() as u32 == slots_per_layer[layer],
            "layer {layer} initial set has {} experts for {} slots",
            set.len(),
            slots_per_layer[layer]
        );
        let mut flags = vec![false; experts];
        for &expert in set {
            ensure!(
                expert < geometry.experts,
                "initial expert {expert} out of range"
            );
            flags[expert as usize] = true;
        }
        resident.push(flags);
        host.push(vec![false; experts]);
    }
    let mut spares = spare_blocks;
    let mut rng = Rng(seed);
    let mut sampled: Vec<u32> = Vec::with_capacity(geometry.top_k as usize);
    let mut pairs: Vec<(f64, u32, u32, u32, bool)> = Vec::new();
    let mut measured = 0u32;
    let mut hit_sum = 0.0f64;
    let mut cpu_sum = 0.0f64;
    let mut h2d_sum = 0.0f64;
    let mut d2h_sum = 0.0f64;
    let mut overlap_sum = 0.0f64;
    let mut serial_sum = 0.0f64;
    let mut admitted_sum = 0.0f64;
    for step in 0..steps {
        let mut sampled_rows: Vec<Vec<u32>> = Vec::with_capacity(layers);
        for row in frequency {
            sample_row(row, geometry.top_k, &mut rng, &mut sampled);
            sampled_rows.push(sampled.clone());
        }
        let mut hits_per_layer = vec![0u32; layers];
        for layer in 0..layers {
            for value in score[layer].iter_mut() {
                *value *= F_DECAY;
            }
            for &expert in &sampled_rows[layer] {
                score[layer][expert as usize] += 1.0;
                if resident[layer][expert as usize] {
                    hits_per_layer[layer] += 1;
                }
            }
        }
        pairs.clear();
        for layer in 0..layers {
            let row = &sampled_rows[layer];
            let active = |expert: u32| row.contains(&expert);
            let victim = (0..geometry.experts)
                .filter(|&expert| resident[layer][expert as usize] && !active(expert))
                .min_by(|&a, &b| {
                    score[layer][a as usize]
                        .total_cmp(&score[layer][b as usize])
                        .then(a.cmp(&b))
                });
            let Some(victim) = victim else { continue };
            for &candidate in row {
                if resident[layer][candidate as usize] {
                    continue;
                }
                let candidate_score = score[layer][candidate as usize];
                let delta = candidate_score - score[layer][victim as usize];
                if candidate_score >= THETA_MIN && delta > 0.0 {
                    pairs.push((
                        delta,
                        layer as u32,
                        candidate,
                        victim,
                        host[layer][candidate as usize],
                    ));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut admitted = 0u32;
        let mut h2d_bytes = 0u64;
        let mut d2h_bytes = 0u64;
        for &(_, layer, candidate, victim, from_host) in &pairs {
            if admitted >= hard_cap {
                break;
            }
            let (l, e, v) = (layer as usize, candidate as usize, victim as usize);
            if !resident[l][v] || resident[l][e] {
                continue;
            }
            let host_block_free = spares > 0 && host_used < host_blocks;
            resident[l][e] = true;
            resident[l][v] = false;
            host[l][e] = false;
            if from_host {
                host[l][v] = true;
            } else if host_block_free {
                host[l][v] = true;
                host_used += 1;
                spares -= 1;
            }
            h2d_bytes += slot_bytes;
            if from_host || host_block_free {
                d2h_bytes += slot_bytes;
            }
            admitted += 1;
        }
        let mut cpu = 0.0f64;
        for hits in &hits_per_layer {
            let misses = geometry.top_k - hits;
            if misses > 0 {
                cpu += service.seconds(1, misses, workers)?;
            }
        }
        let h2d = h2d_bytes as f64 / h2d_bytes_per_second;
        let d2h = d2h_bytes as f64 / d2h_bytes_per_second;
        let copy = if copy_engines >= 2 {
            h2d.max(d2h)
        } else {
            h2d + d2h
        };
        if step >= warmup {
            measured += 1;
            hit_sum += f64::from(hits_per_layer.iter().sum::<u32>());
            cpu_sum += cpu;
            h2d_sum += h2d;
            d2h_sum += d2h;
            overlap_sum += cpu.max(copy);
            serial_sum += cpu + copy;
            admitted_sum += f64::from(admitted);
        }
    }
    let measured = f64::from(measured);
    let rows_total = f64::from(geometry.layers * geometry.top_k);
    Ok(Prediction {
        hit: hit_sum / measured / rows_total,
        cpu_seconds: cpu_sum / measured,
        h2d_seconds: h2d_sum / measured,
        d2h_seconds: d2h_sum / measured,
        overlap_seconds: overlap_sum / measured,
        serial_seconds: serial_sum / measured,
        admitted_per_token: admitted_sum / measured,
    })
}

/// Admission ladder: whole expert slots per token, zero first.
pub fn budget_ladder(slot_bytes: u64, max_experts_per_token: u32) -> Result<Vec<u64>> {
    ensure!(
        max_experts_per_token > 0 && max_experts_per_token <= 1024,
        "invalid admission ladder extent {max_experts_per_token}"
    );
    let mut ladder = Vec::with_capacity(max_experts_per_token as usize + 1);
    ladder.push(0);
    for experts in 1..=max_experts_per_token {
        ladder.push(
            u64::from(experts)
                .checked_mul(slot_bytes)
                .context("admission budget overflow")?,
        );
    }
    Ok(ladder)
}

/// One budget search over a fixed setup.
pub struct BudgetSearch<'a> {
    pub inputs: &'a PlanInputs<'a>,
    pub initial: &'a [Vec<u32>],
    pub frequency: &'a [Vec<f64>],
    pub ladder: &'a [u64],
    pub steps: u32,
    pub warmup: u32,
    pub seed: u64,
}

/// Capacity and budget with the smallest predicted overlap-aware decode time.
pub fn choose_budget(search: BudgetSearch<'_>) -> Result<(Capacity, u64, Prediction)> {
    ensure!(!search.ladder.is_empty(), "empty admission ladder");
    let slot_bytes = search.inputs.geometry.slot_bytes()?;
    let mut best: Option<(Capacity, u64, Prediction)> = None;
    for &budget in search.ladder {
        let cap = (budget / slot_bytes) as u32;
        let capacity = capacity(search.inputs, cap)?;
        let prediction = simulate(Simulation {
            geometry: search.inputs.geometry,
            slots_per_layer: &capacity.slots_per_layer,
            host_blocks: capacity.host_blocks,
            spare_blocks: capacity.spare_blocks,
            initial: search.initial,
            frequency: search.frequency,
            admission: Admission {
                budget_bytes_per_token: budget,
            },
            service: search.inputs.service,
            h2d_bytes_per_second: search.inputs.h2d_bytes_per_second,
            d2h_bytes_per_second: search.inputs.d2h_bytes_per_second,
            copy_engines: search.inputs.copy_engines,
            steps: search.steps,
            warmup: search.warmup,
            seed: search.seed,
        })?;
        if best
            .as_ref()
            .is_none_or(|(_, _, seen)| prediction.overlap_seconds < seen.overlap_seconds)
        {
            best = Some((capacity, budget, prediction));
        }
    }
    Ok(best.expect("ladder is not empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
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

    fn capacity_ref<'a>(inputs: &PlanInputs<'a>) -> PlanInputs<'a> {
        PlanInputs {
            geometry: inputs.geometry,
            device_free_bytes: inputs.device_free_bytes,
            host_available_bytes: inputs.host_available_bytes,
            h2d_bytes_per_second: inputs.h2d_bytes_per_second,
            d2h_bytes_per_second: inputs.d2h_bytes_per_second,
            copy_engines: inputs.copy_engines,
            disk_readers: inputs.disk_readers,
            service: inputs.service,
            reserves: inputs.reserves,
            span_rows: inputs.span_rows,
            wave_rows: inputs.wave_rows,
        }
    }

    fn simulation<'a>(
        slots: &'a [u32],
        initial: &'a [Vec<u32>],
        frequency: &'a [Vec<f64>],
        service: &'a ServiceTable,
        admission: Admission,
    ) -> Simulation<'a> {
        Simulation {
            geometry: geometry(),
            slots_per_layer: slots,
            host_blocks: 5,
            spare_blocks: 2,
            initial,
            frequency,
            admission,
            service,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            steps: 64,
            warmup: 8,
            seed: 7,
        }
    }

    fn search<'a>(
        inputs: &'a PlanInputs<'a>,
        initial: &'a [Vec<u32>],
        frequency: &'a [Vec<f64>],
        ladder: &'a [u64],
    ) -> BudgetSearch<'a> {
        BudgetSearch {
            inputs,
            initial,
            frequency,
            ladder,
            steps: 32,
            warmup: 4,
            seed: 3,
        }
    }

    fn inputs<'a>(service: &'a ServiceTable, reserves: &'a Reserves) -> PlanInputs<'a> {
        let geometry = geometry();
        let slot = geometry.slot_bytes().unwrap();
        PlanInputs {
            geometry,
            device_free_bytes: reserves.device_bytes(&geometry, 4).unwrap() + 4 * slot,
            host_available_bytes: reserves.host_bytes(&geometry).unwrap() + 26 * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 5e8,
            copy_engines: 2,
            disk_readers: 2,
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
        assert_eq!(derived.host_blocks, 12);
        assert_eq!(derived.disk_bytes, 0);
        let small_host = PlanInputs {
            host_available_bytes: reserves.host_bytes(&geometry()).unwrap()
                + 8 * geometry().slot_bytes().unwrap(),
            ..capacity_ref(&inputs)
        };
        let small = capacity(&small_host, 0).unwrap();
        assert_eq!(small.host_blocks, 8);
        assert_eq!(
            small.disk_bytes,
            (16 - 4 - 8) * geometry().slot_bytes().unwrap()
        );
        let tight = PlanInputs {
            device_free_bytes: reserves.device_bytes(&geometry(), 4).unwrap()
                + 2 * geometry().slot_bytes().unwrap()
                - 1,
            ..inputs
        };
        let error = capacity(&tight, 0).unwrap_err().to_string();
        assert!(error.contains("device expert bytes"), "{error}");
        let over = PlanInputs {
            device_free_bytes: reserves.device_bytes(&geometry(), 4).unwrap() - 1,
            ..tight
        };
        let error = capacity(&over, 0).unwrap_err().to_string();
        assert!(error.contains("span/staging reserves"), "{error}");
        let spares = capacity(&inputs, 28).unwrap_err().to_string();
        assert!(spares.contains("spare blocks"), "{spares}");
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
    fn f_seed_is_twelve_normalized_prefill_copies() -> Result<()> {
        let scores = seed_scores(&frequency(), &geometry())?;
        for layer in &scores {
            let sum: f64 = layer.iter().sum();
            assert!((sum - F_SEED_SCALE * FREQUENCY_SUM).abs() < 1e-9);
        }
        let zero = vec![vec![0.0; 8], vec![0.0; 8]];
        assert!(seed_scores(&zero, &geometry()).is_err());
        Ok(())
    }

    #[test]
    fn ledger_keeps_tiers_disjoint_and_bytes_exact() -> Result<()> {
        let geometry = geometry();
        let slot = geometry.slot_bytes()?;
        let initial = initial_residency(&frequency(), &[2, 2], &geometry)?;
        let mut ledger = TierLedger::new(geometry, vec![2, 2], 5, &initial)?;
        let routed = 16 * slot;
        assert_eq!(ledger.unique_bytes()? + ledger.disk_bytes()?, routed);
        assert_eq!(ledger.tier(0, initial[0][0])?, Tier::Gpu);
        assert_eq!(ledger.tier(0, 7)?, Tier::Disk);
        ledger.admit_disk(0, 7, initial[0][0], true)?;
        assert_eq!(ledger.tier(0, 7)?, Tier::Gpu);
        assert_eq!(ledger.tier(0, initial[0][0])?, Tier::Host);
        ledger.admit_host(0, initial[0][0], initial[0][1])?;
        assert_eq!(ledger.tier(0, initial[0][0])?, Tier::Gpu);
        assert_eq!(ledger.tier(0, initial[0][1])?, Tier::Host);
        ledger.admit_disk(1, 7, initial[1][0], true)?;
        assert_eq!(ledger.tier(1, 7)?, Tier::Gpu);
        assert_eq!(ledger.tier(1, initial[1][0])?, Tier::Host);
        ledger.evict_gpu(0, 7)?;
        assert_eq!(ledger.tier(0, 7)?, Tier::Disk);
        assert_eq!(ledger.duplicate_bytes(), 0);
        assert_eq!(ledger.unique_bytes()? + ledger.disk_bytes()?, routed);
        for layer in 0..2u32 {
            for expert in 0..8u32 {
                let tier = ledger.tier(layer, expert)?;
                assert!(matches!(tier, Tier::Gpu | Tier::Host | Tier::Disk));
            }
        }
        assert!(ledger.check().is_ok());
        assert!(ledger.admit_host(1, initial[1][1], initial[1][1]).is_err());
        Ok(())
    }

    #[test]
    fn admission_hard_cap_only_and_deterministic() -> Result<()> {
        let geometry = geometry();
        let frequency = frequency();
        let slots = vec![2u32, 2u32];
        let initial = initial_residency(&frequency, &slots, &geometry)?;
        let slot_bytes = geometry.slot_bytes()?;
        let service = service();
        let zero = simulate(simulation(
            &slots,
            &initial,
            &frequency,
            &service,
            Admission {
                budget_bytes_per_token: 0,
            },
        ))?;
        assert_eq!(zero.admitted_per_token, 0.0);
        assert_eq!(zero.h2d_seconds, 0.0);
        assert_eq!(zero.d2h_seconds, 0.0);
        let budget = Admission {
            budget_bytes_per_token: slot_bytes * 2,
        };
        assert_eq!(budget.hard_cap(slot_bytes), 2);
        let two = simulate(simulation(&slots, &initial, &frequency, &service, budget))?;
        assert!(two.admitted_per_token > 0.0 && two.admitted_per_token <= 2.0);
        assert_eq!(
            two,
            simulate(simulation(&slots, &initial, &frequency, &service, budget))?
        );
        Ok(())
    }

    #[test]
    fn d2h_pricing_matches_the_engine_count() -> Result<()> {
        let geometry = geometry();
        let frequency = frequency();
        let slots = vec![2u32, 2u32];
        let initial = initial_residency(&frequency, &slots, &geometry)?;
        let slot_bytes = geometry.slot_bytes()?;
        let reserves = reserves();
        let service = service();
        let base = inputs(&service, &reserves);
        let single_inputs = PlanInputs {
            copy_engines: 1,
            ..capacity_ref(&base)
        };
        let dual_inputs = PlanInputs {
            copy_engines: 2,
            ..capacity_ref(&base)
        };
        let ladder = [0, slot_bytes];
        let (_, _, single) = choose_budget(search(&single_inputs, &initial, &frequency, &ladder))?;
        let (_, _, dual) = choose_budget(search(&dual_inputs, &initial, &frequency, &ladder))?;
        assert!(single.serial_seconds >= single.overlap_seconds);
        assert!(dual.serial_seconds >= dual.overlap_seconds);
        Ok(())
    }

    #[test]
    fn budget_choice_is_deterministic_and_respects_the_ladder() -> Result<()> {
        let geometry = geometry();
        let frequency = frequency();
        let slots = vec![2u32, 2u32];
        let initial = initial_residency(&frequency, &slots, &geometry)?;
        let slot_bytes = geometry.slot_bytes()?;
        let ladder = budget_ladder(slot_bytes, 2)?;
        assert_eq!(ladder, vec![0, slot_bytes, slot_bytes * 2]);
        let reserves = reserves();
        let service = service();
        let inputs = inputs(&service, &reserves);
        let (_, budget, _) = choose_budget(search(&inputs, &initial, &frequency, &ladder))?;
        assert!(ladder.contains(&budget));
        let (_, again, _) = choose_budget(search(&inputs, &initial, &frequency, &ladder))?;
        assert_eq!(budget, again);
        Ok(())
    }
}
