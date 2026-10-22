//! Prefill span service: cohort classes and the wave plan for one layer's whole-span routing.
use crate::experts::arena::SlotEvents;
use crate::experts::catalogue::ExpertGeometry;
use crate::experts::plan::ServiceTable;
use anyhow::{Context, Result, ensure};

/// Measured service rates and the setup-bound CPU row constraint.
pub struct SpanRates<'a> {
    pub service: &'a ServiceTable,
    pub h2d_bytes_per_second: f64,
    pub cpu_row_limit: u32,
}

/// Service class of one routed expert in a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanClass {
    Resident,
    Streamed,
    Cpu,
}

/// Per-layer span execution plan; `update` rewrites an instance in place without allocating.
/// `staged` is one flat expert list whose `staging_slots`-expert chunks are the cohorts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanPlan {
    span_rows: u32,
    wave_rows: u32,
    waves: u32,
    streamed_bytes: u64,
    geometry: ExpertGeometry,
    slot_bytes: u64,
    staging_slots: u32,
    workers: u32,
    resident: Vec<u32>,
    staged: Vec<u32>,
    cpu: Vec<u32>,
    classes: Vec<SpanClass>,
    expert_cohort: Vec<u16>,
    cohort_offsets: Vec<usize>,
    cohort_cursor: Vec<usize>,
    assignment_rows: Vec<u32>,
    wave_offsets: Vec<usize>,
    packed_waves: Vec<(u32, u32, u32)>,
}

impl SpanPlan {
    /// Setup-time allocation for one geometry's experts and `staging_slots`-expert cohorts;
    /// `workers` is the service table's selected worker count.
    pub fn new(
        geometry: &ExpertGeometry,
        slot_bytes: u64,
        staging_slots: u32,
        workers: u32,
    ) -> Result<Self> {
        ensure!(slot_bytes > 0, "zero expert slot bytes");
        ensure!(staging_slots > 0, "zero prefill staging slots");
        ensure!(workers > 0, "zero service workers");
        let experts = geometry.experts as usize;
        let max_cohorts = experts.div_ceil(staging_slots as usize) + 1;
        Ok(Self {
            span_rows: 0,
            wave_rows: 0,
            waves: 0,
            streamed_bytes: 0,
            geometry: *geometry,
            slot_bytes,
            staging_slots,
            workers,
            resident: Vec::with_capacity(experts),
            staged: Vec::with_capacity(experts),
            cpu: Vec::with_capacity(experts),
            classes: vec![SpanClass::Cpu; experts],
            expert_cohort: vec![u16::MAX; experts],
            cohort_offsets: Vec::with_capacity(max_cohorts + 1),
            cohort_cursor: Vec::with_capacity(max_cohorts),
            assignment_rows: Vec::new(),
            wave_offsets: Vec::with_capacity(max_cohorts + 1),
            packed_waves: Vec::new(),
        })
    }

    /// Rewrites the reusable cohorts from routed row counts and current residency.
    pub fn update(
        &mut self,
        counts: &[u32],
        resident: &[bool],
        span_rows: u32,
        wave_rows: u32,
        rates: &SpanRates<'_>,
    ) -> Result<()> {
        let geometry = &self.geometry;
        ensure!(
            counts.len() == geometry.experts as usize,
            "expert counts hold {} experts, expected {}",
            counts.len(),
            geometry.experts
        );
        ensure!(
            resident.len() == geometry.experts as usize,
            "residency holds {} experts, expected {}",
            resident.len(),
            geometry.experts
        );
        ensure!(span_rows > 0, "zero span rows");
        ensure!(
            wave_rows > 0 && wave_rows <= span_rows,
            "wave rows {wave_rows} outside the span {span_rows}"
        );
        ensure!(
            rates.h2d_bytes_per_second.is_finite() && rates.h2d_bytes_per_second > 0.0,
            "invalid H2D rate {}",
            rates.h2d_bytes_per_second
        );
        let slot_bytes = self.slot_bytes;
        let copy_seconds = slot_bytes as f64 / rates.h2d_bytes_per_second;
        let workers = self.workers;
        let max_rows = rates.service.max_rows();
        let mut streamed = 0usize;
        for (expert, &rows) in counts.iter().enumerate() {
            let class = if rows == 0 {
                continue;
            } else if resident[expert] {
                SpanClass::Resident
            } else if rows > rates.cpu_row_limit {
                SpanClass::Streamed
            } else if rows > max_rows {
                let at_max = rates.service.rows_time(max_rows, 1, workers)?;
                ensure!(
                    copy_seconds < at_max,
                    "expert {expert} rows={rows} exceed the measured rows extent ..={max_rows} and the CPU route wins at {max_rows}"
                );
                SpanClass::Streamed
            } else {
                let cpu_seconds = rates.service.rows_time(rows, 1, workers)?;
                if copy_seconds < cpu_seconds {
                    SpanClass::Streamed
                } else {
                    SpanClass::Cpu
                }
            };
            self.classes[expert] = class;
            if class == SpanClass::Streamed {
                streamed += 1;
            }
        }
        let streamed_bytes = u64::try_from(streamed)?
            .checked_mul(slot_bytes)
            .context("streamed span bytes overflow")?;
        self.resident.clear();
        self.staged.clear();
        self.cpu.clear();
        for (expert, &rows) in counts.iter().enumerate() {
            if rows == 0 {
                continue;
            }
            match self.classes[expert] {
                SpanClass::Resident => self.resident.push(expert as u32),
                SpanClass::Streamed => self.staged.push(expert as u32),
                SpanClass::Cpu => self.cpu.push(expert as u32),
            }
        }
        self.span_rows = span_rows;
        self.wave_rows = wave_rows;
        self.waves = span_rows.div_ceil(wave_rows);
        self.streamed_bytes = streamed_bytes;
        self.wave_offsets.clear();
        self.packed_waves.clear();
        Ok(())
    }

    /// Buckets whole-span assignments by cohort in one pass and cuts each cohort into
    /// contiguous `(base, rows, active)` waves with `0 < active <= wave_rows * top_k`.
    pub fn pack_waves(&mut self, counts: &[u32], ids: &[i32]) -> Result<()> {
        let experts = self.geometry.experts as usize;
        let top_k = self.geometry.top_k as usize;
        let choices = (self.span_rows as usize)
            .checked_mul(top_k)
            .context("span choice count overflow")?;
        ensure!(
            self.span_rows > 0 && self.wave_rows > 0,
            "span plan is not initialized"
        );
        ensure!(
            counts.len() == experts,
            "expert counts hold {} experts, expected {experts}",
            counts.len()
        );
        ensure!(
            ids.len() == choices,
            "routed ids hold {} choices, expected {choices}",
            ids.len()
        );
        let num_cohorts = 1 + self.cohort_count() as usize;
        self.expert_cohort.fill(u16::MAX);
        for &expert in &self.resident {
            self.expert_cohort[expert as usize] = 0;
        }
        for (cohort, chunk) in self.staged.chunks(self.staging_slots as usize).enumerate() {
            let tag = u16::try_from(cohort + 1).context("cohort index exceeds u16")?;
            for &expert in chunk {
                self.expert_cohort[expert as usize] = tag;
            }
        }
        self.cohort_offsets.clear();
        self.cohort_offsets.push(0);
        let resident_total: usize = self
            .resident
            .iter()
            .map(|&expert| counts[expert as usize] as usize)
            .sum();
        self.cohort_offsets.push(resident_total);
        let mut total = resident_total;
        for chunk in self.staged.chunks(self.staging_slots as usize) {
            let cohort_total: usize = chunk
                .iter()
                .map(|&expert| counts[expert as usize] as usize)
                .sum();
            total = total
                .checked_add(cohort_total)
                .context("cohort assignment offset overflow")?;
            self.cohort_offsets.push(total);
        }
        self.cohort_cursor.clear();
        self.cohort_cursor
            .extend_from_slice(&self.cohort_offsets[..num_cohorts]);
        if self.assignment_rows.len() < total {
            self.assignment_rows.resize(total, 0);
        }
        for (choice, &expert) in ids.iter().enumerate() {
            let index = usize::try_from(expert)
                .ok()
                .filter(|&index| index < experts)
                .with_context(|| format!("routed expert {expert} outside 0..{experts}"))?;
            let cohort = self.expert_cohort[index];
            if cohort != u16::MAX {
                let c = cohort as usize;
                let pos = self.cohort_cursor[c];
                ensure!(
                    pos < self.cohort_offsets[c + 1],
                    "cohort {c} assignment count exceeds span count"
                );
                self.assignment_rows[pos] = (choice / top_k) as u32;
                self.cohort_cursor[c] = pos + 1;
            }
        }
        for c in 0..num_cohorts {
            ensure!(
                self.cohort_cursor[c] == self.cohort_offsets[c + 1],
                "cohort {c} bucketed {} of {} assignments",
                self.cohort_cursor[c] - self.cohort_offsets[c],
                self.cohort_offsets[c + 1] - self.cohort_offsets[c]
            );
        }
        let active_limit = (self.wave_rows as usize)
            .checked_mul(top_k)
            .context("wave active capacity overflow")?;
        self.packed_waves.clear();
        self.wave_offsets.clear();
        self.wave_offsets.push(0);
        for c in 0..num_cohorts {
            let rows = &self.assignment_rows[self.cohort_offsets[c]..self.cohort_offsets[c + 1]];
            let mut idx = 0usize;
            while idx < rows.len() {
                let base = rows[idx];
                let mut end = (idx + active_limit).min(rows.len());
                if end < rows.len() {
                    let split_row = rows[end];
                    while end > idx && rows[end - 1] == split_row {
                        end -= 1;
                    }
                }
                ensure!(
                    end > idx,
                    "wave packing made no progress at cohort {c} row {base}"
                );
                let wave_rows = rows[end - 1] - base + 1;
                let active = u32::try_from(end - idx)?;
                self.packed_waves.push((base, wave_rows, active));
                idx = end;
            }
            self.wave_offsets.push(self.packed_waves.len());
        }
        Ok(())
    }

    pub fn cohort_waves(&self, cohort: usize) -> Result<&[(u32, u32, u32)]> {
        ensure!(
            cohort + 1 < self.wave_offsets.len(),
            "cohort {cohort} outside packed wave count {}",
            self.wave_offsets.len().saturating_sub(1)
        );
        Ok(&self.packed_waves[self.wave_offsets[cohort]..self.wave_offsets[cohort + 1]])
    }

    pub fn span_rows(&self) -> u32 {
        self.span_rows
    }

    pub fn wave_rows(&self) -> u32 {
        self.wave_rows
    }

    pub fn waves(&self) -> u32 {
        self.waves
    }

    pub fn streamed_bytes(&self) -> u64 {
        self.streamed_bytes
    }

    pub fn resident(&self) -> &[u32] {
        &self.resident
    }

    /// Flat staged expert list in routing order; `cohorts()` chunks it by staging slot count.
    pub fn staged(&self) -> &[u32] {
        &self.staged
    }

    pub fn cohorts(&self) -> impl Iterator<Item = &[u32]> {
        self.staged.chunks(self.staging_slots as usize)
    }

    pub fn cohort_count(&self) -> u32 {
        u32::try_from(self.staged.len().div_ceil(self.staging_slots as usize))
            .expect("cohort count fits u32")
    }

    pub fn cpu(&self) -> &[u32] {
        &self.cpu
    }
}

/// Prefill staging leases: one cohort copy per round, pinned through every row wave.
/// The event bookkeeping, the slot list and the released wait list are allocated once.
pub struct StagingLeases {
    slots: u32,
    waves: u32,
    events: SlotEvents,
    ids: Vec<u32>,
    recorded: Vec<u32>,
    loaded: u32,
    waits: Vec<u64>,
}

impl StagingLeases {
    pub fn new(slots: u32) -> Result<Self> {
        ensure!(slots > 0, "zero prefill staging slots");
        Ok(Self {
            slots,
            waves: 0,
            events: SlotEvents::new(slots as usize),
            ids: Vec::with_capacity(slots as usize),
            recorded: vec![0; slots as usize],
            loaded: 0,
            waits: Vec::with_capacity(slots as usize),
        })
    }

    /// Sets the row-wave count of the next cohort; legal only with no cohort loaded.
    pub fn reset(&mut self, waves: u32) -> Result<()> {
        ensure!(waves > 0, "zero span row waves");
        ensure!(
            self.loaded == 0,
            "staging cohort {} is still loaded",
            self.loaded
        );
        self.waves = waves;
        Ok(())
    }

    /// Records the cohort copy into staging slots `0..cohort.len()` under one event.
    pub fn load(&mut self, cohort: &[u32], event: u64) -> Result<()> {
        ensure!(
            self.loaded == 0,
            "staging cohort {} is still loaded",
            self.loaded
        );
        ensure!(self.waves > 0, "staging waves are not set");
        ensure!(!cohort.is_empty(), "empty staging cohort");
        ensure!(
            cohort.len() as u32 <= self.slots,
            "cohort {} exceeds {} staging slots",
            cohort.len(),
            self.slots
        );
        self.ids.clear();
        self.ids.extend(0..cohort.len() as u32);
        self.events.record_copy(&self.ids, event)?;
        self.recorded[..cohort.len()].fill(0);
        self.loaded = cohort.len() as u32;
        Ok(())
    }

    /// Records one wave's reader event on one staging slot.
    pub fn record_wave(&mut self, slot: u32, event: u64) -> Result<()> {
        ensure!(
            slot < self.loaded,
            "staging slot {slot} outside the loaded cohort {}",
            self.loaded
        );
        self.events.record_readers(&[slot], event)?;
        self.recorded[slot as usize] += 1;
        Ok(())
    }

    /// Releases a cohort that recorded every row wave and returns the retained reader-event slice
    /// that the next copy stream waits on.
    pub fn release(&mut self) -> Result<&[u64]> {
        ensure!(self.loaded > 0, "no staging cohort to release");
        for slot in 0..self.loaded {
            ensure!(
                self.recorded[slot as usize] == self.waves,
                "staging slot {slot} recorded {} of {} waves",
                self.recorded[slot as usize],
                self.waves
            );
        }
        self.waits.clear();
        for slot in 0..self.loaded {
            if let [_, Some(reader)] = self.events.reclaim(slot)? {
                self.waits.push(reader);
            }
        }
        self.loaded = 0;
        Ok(&self.waits)
    }
}

/// One prefetch entry: layer, expert and whether a demand read asked for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefetchEntry {
    layer: u32,
    expert: u32,
    demand: bool,
}

/// A retired prefetch; a loaded entry's ring slot is returned for the caller to release after its
/// copy event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retired {
    Queued,
    Loaded(u32),
}

/// Outcome of a prefetch request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requested {
    Queued,
    Present,
    Deferred,
}

/// Bounded layer-ahead disk prefetch over the ring; demand entries take priority.
pub struct Prefetch {
    depth: u32,
    queued: Vec<PrefetchEntry>,
    deferred: Vec<PrefetchEntry>,
    in_flight: Option<PrefetchEntry>,
    loaded: Vec<(PrefetchEntry, u32)>,
    taken: Vec<(PrefetchEntry, u32)>,
}

impl Prefetch {
    pub fn new(depth: u32, ring_slots: u32) -> Result<Self> {
        ensure!(depth > 0, "zero prefetch depth");
        ensure!(
            depth <= ring_slots,
            "prefetch depth {depth} exceeds {ring_slots} ring slots"
        );
        Ok(Self {
            depth,
            queued: Vec::with_capacity(depth as usize),
            deferred: Vec::with_capacity(depth as usize),
            in_flight: None,
            loaded: Vec::with_capacity(depth as usize),
            taken: Vec::with_capacity(depth as usize),
        })
    }

    /// Queued, in-flight, loaded and taken entries: every entry holds or waits for a ring slot.
    /// Deferred demands hold no entry until they are admitted.
    pub fn entries(&self) -> u32 {
        let in_flight = usize::from(self.in_flight.is_some());
        u32::try_from(self.queued.len() + in_flight + self.loaded.len() + self.taken.len())
            .expect("prefetch entry count fits u32")
    }

    fn contains(&self, layer: u32, expert: u32) -> bool {
        self.queued
            .iter()
            .chain(self.deferred.iter())
            .chain(self.in_flight.iter())
            .chain(self.loaded.iter().map(|(entry, _)| entry))
            .chain(self.taken.iter().map(|(entry, _)| entry))
            .any(|entry| entry.layer == layer && entry.expert == expert)
    }

    /// Admits deferred demands first whenever an entry frees.
    fn admit_deferred(&mut self) {
        while self.entries() < self.depth && !self.deferred.is_empty() {
            let entry = self.deferred.remove(0);
            if !self.contains(entry.layer, entry.expert) {
                self.queued.insert(0, entry);
            }
        }
    }

    /// Queues one disk read. A demand that cannot be queued is deferred and admitted first when an
    /// entry frees; a prefetch of a host-tier expert is rejected. The second value is an evicted
    /// loaded entry's ring slot for `discard`.
    pub fn request(
        &mut self,
        layer: u32,
        expert: u32,
        demand: bool,
        host: bool,
    ) -> Result<(Requested, Option<u32>)> {
        ensure!(
            !(host && !demand),
            "host-tier expert {expert} of layer {layer} is not a disk prefetch"
        );
        self.push(layer, expert, demand)
    }

    fn push(&mut self, layer: u32, expert: u32, demand: bool) -> Result<(Requested, Option<u32>)> {
        if self.contains(layer, expert) {
            return Ok((Requested::Present, None));
        }
        let mut evicted_slot = None;
        if self.entries() >= self.depth {
            ensure!(demand, "prefetch queue full at depth {}", self.depth);
            if let Some(index) = self.queued.iter().position(|entry| !entry.demand) {
                self.queued.remove(index);
            } else if let Some(index) = self.loaded.iter().position(|(entry, _)| !entry.demand) {
                evicted_slot = Some(self.loaded.remove(index).1);
            } else {
                ensure!(
                    self.deferred.len() < self.depth as usize,
                    "pending demand list full at depth {}",
                    self.depth
                );
                self.deferred.push(PrefetchEntry {
                    layer,
                    expert,
                    demand,
                });
                return Ok((Requested::Deferred, None));
            }
        }
        let entry = PrefetchEntry {
            layer,
            expert,
            demand,
        };
        if demand {
            self.queued.insert(0, entry);
        } else {
            self.queued.push(entry);
        }
        Ok((Requested::Queued, evicted_slot))
    }

    /// Starts the oldest request when a ring slot is free; it stays in flight until `complete`
    /// and is never evicted.
    pub fn begin_read(&mut self) -> Option<(u32, u32)> {
        self.admit_deferred();
        if self.in_flight.is_some() {
            return None;
        }
        let in_use = self.loaded.len() + self.taken.len();
        if in_use >= self.depth as usize {
            return None;
        }
        let entry = self.queued.first().copied()?;
        self.queued.remove(0);
        self.in_flight = Some(entry);
        Some((entry.layer, entry.expert))
    }

    /// Records the ring slot of the read in flight.
    pub fn complete(&mut self, layer: u32, expert: u32, slot: u32) -> Result<()> {
        let entry = self.in_flight.context("no prefetch read in flight")?;
        ensure!(
            (entry.layer, entry.expert) == (layer, expert),
            "completed request ({layer}, {expert}) is not the read in flight"
        );
        self.in_flight = None;
        self.loaded.push((entry, slot));
        Ok(())
    }

    /// Retires a non-demand queued or loaded request; a loaded entry's ring slot is returned so
    /// the caller can release it after its copy event. Cancelling a demand names it.
    pub fn cancel(&mut self, layer: u32, expert: u32) -> Result<Retired> {
        if let Some(index) = self
            .queued
            .iter()
            .position(|entry| entry.layer == layer && entry.expert == expert)
        {
            ensure!(
                !self.queued[index].demand,
                "demand {expert} of layer {layer} cannot be cancelled"
            );
            self.queued.remove(index);
            self.admit_deferred();
            return Ok(Retired::Queued);
        }
        let index = self
            .loaded
            .iter()
            .position(|(entry, _)| entry.layer == layer && entry.expert == expert)
            .with_context(|| format!("no queued or loaded prefetch for ({layer}, {expert})"))?;
        ensure!(
            !self.loaded[index].0.demand,
            "demand {expert} of layer {layer} cannot be cancelled"
        );
        let retired = Retired::Loaded(self.loaded.remove(index).1);
        self.admit_deferred();
        Ok(retired)
    }

    /// Takes a loaded request's ring slot; the slot keeps counting toward the depth until
    /// `release`.
    pub fn take(&mut self, layer: u32, expert: u32) -> Option<u32> {
        let index = self
            .loaded
            .iter()
            .position(|(entry, _)| entry.layer == layer && entry.expert == expert)?;
        let (entry, slot) = self.loaded.remove(index);
        self.taken.push((entry, slot));
        Some(slot)
    }

    /// Frees a taken ring slot after the consumer released it and admits deferred demands.
    pub fn release(&mut self, slot: u32) -> bool {
        let released = self
            .taken
            .iter()
            .position(|(_, taken)| *taken == slot)
            .map(|index| self.taken.remove(index))
            .is_some();
        if released {
            self.admit_deferred();
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::plan::{ServiceCell, ServiceTable};

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn table() -> ServiceTable {
        let mut cells = Vec::new();
        for workers in [1u32, 2] {
            for rows in [1u32, 2, 4, 8] {
                for experts in [1u32, 2] {
                    cells.push(ServiceCell {
                        workers,
                        rows,
                        experts,
                        seconds: f64::from(rows) * 0.01 * f64::from(experts) / f64::from(workers),
                    });
                }
            }
        }
        ServiceTable::new("test", cells).unwrap()
    }

    #[test]
    fn classes_follow_the_measured_copy_and_cpu_times() {
        let geometry = geometry();
        let counts = [4u32, 1, 1, 0, 2, 1, 0, 0];
        let resident = [true, false, false, false, false, false, false, false];
        let mut rates = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        // 192 B at 1 GB/s is 0.192 us, below every measured CPU cell here.
        let mut plan = planner(&geometry, 2, &rates).unwrap();
        plan.update(&counts, &resident, 4, 2, &rates).unwrap();
        assert_eq!(plan.resident(), &[0]);
        assert_eq!(plan.staged(), &[1, 2, 4, 5]);
        assert_eq!(
            plan.cohorts().collect::<Vec<_>>(),
            vec![&[1, 2][..], &[4, 5][..]]
        );
        assert_eq!(plan.cohort_count(), 2);
        assert!(plan.cpu().is_empty());
        assert_eq!(plan.streamed_bytes(), 4 * 192);
        assert_eq!(plan.waves(), 2);
        // 192 B at 100 B/s is 1.92 s, above every measured CPU cell here.
        rates.h2d_bytes_per_second = 100.0;
        let mut plan = planner(&geometry, 2, &rates).unwrap();
        plan.update(&counts, &resident, 4, 2, &rates).unwrap();
        assert!(plan.staged().is_empty());
        assert_eq!(plan.cohort_count(), 0);
        assert_eq!(plan.cpu(), &[1, 2, 4, 5]);
        assert_eq!(plan.streamed_bytes(), 0);
    }

    #[test]
    fn expert_rows_interpolate_between_measured_cells() {
        let table = table();
        assert_eq!(table.rows_time(2, 1, 1).unwrap(), 0.02);
        assert_eq!(table.rows_time(3, 1, 1).unwrap(), 0.03);
        assert_eq!(table.rows_time(6, 1, 1).unwrap(), 0.06);
        let error = table.rows_time(16, 1, 1).unwrap_err().to_string();
        assert!(
            error.contains("outside the measured rows extent"),
            "{error}"
        );
        assert!(table.rows_time(0, 1, 1).is_err());
        assert!(table.rows_time(10, 3, 1).is_err());
    }

    #[test]
    fn rows_above_the_measured_window_need_the_gpu_class_to_win_at_the_maximum() {
        let geometry = geometry();
        let counts = [0u32, 16, 0, 0, 0, 0, 0, 0];
        let resident = [false; 8];
        let fast = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        let mut plan = planner(&geometry, 2, &fast).unwrap();
        plan.update(&counts, &resident, 16, 2, &fast).unwrap();
        assert_eq!(plan.staged(), &[1]);
        assert_eq!(plan.cohorts().collect::<Vec<_>>(), vec![&[1][..]]);
        assert!(plan.cpu().is_empty());
        let slow = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 100.0,
        };
        let error = plan
            .update(&counts, &resident, 16, 2, &slow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceed the measured rows extent"), "{error}");
    }

    #[test]
    fn plans_validate_their_extents() {
        let geometry = geometry();
        let counts = [0u32; 8];
        let resident = [false; 8];
        let rates = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        let mut plan = planner(&geometry, 2, &rates).unwrap();
        assert!(plan.update(&counts[..7], &resident, 4, 2, &rates).is_err());
        assert!(plan.update(&counts, &resident[..7], 4, 2, &rates).is_err());
        assert!(plan.update(&counts, &resident, 4, 8, &rates).is_err());
        plan.update(&counts, &resident, 4, 2, &rates).unwrap();
        assert!(plan.resident().is_empty() && plan.staged().is_empty() && plan.cpu().is_empty());
        assert_eq!(plan.waves(), 2);
    }

    #[test]
    fn staging_leases_pin_a_cohort_through_every_wave() -> Result<()> {
        let mut leases = StagingLeases::new(3)?;
        leases.reset(2)?;
        let error = leases.record_wave(0, 10).unwrap_err().to_string();
        assert!(error.contains("outside the loaded cohort"), "{error}");
        leases.load(&[7, 8, 9], 1)?;
        let error = leases.load(&[4], 2).unwrap_err().to_string();
        assert!(error.contains("is still loaded"), "{error}");
        for slot in 0..3u32 {
            leases.record_wave(slot, 10 + u64::from(slot))?;
        }
        let error = leases.release().unwrap_err().to_string();
        assert!(error.contains("recorded 1 of 2 waves"), "{error}");
        for slot in 0..3u32 {
            leases.record_wave(slot, 20 + u64::from(slot))?;
        }
        let waits = leases.release()?;
        assert_eq!(waits, &[20, 21, 22], "the last reader event per slot wins");
        leases.load(&[1, 2], 30)?;
        let error = leases.record_wave(2, 31).unwrap_err().to_string();
        assert!(error.contains("outside the loaded cohort 2"), "{error}");
        Ok(())
    }

    #[test]
    fn staging_leases_validate_cohorts_events_and_waves() {
        assert!(StagingLeases::new(0).is_err());
        let mut leases = StagingLeases::new(2).unwrap();
        assert!(leases.reset(0).is_err());
        assert!(leases.load(&[1], 1).is_err(), "waves are not set");
        leases.reset(1).unwrap();
        assert!(leases.load(&[1, 2, 3], 1).is_err());
        assert!(leases.load(&[], 1).is_err());
        leases.load(&[1], 4).unwrap();
        leases.record_wave(0, 6).unwrap();
        assert!(leases.record_wave(1, 7).is_err());
        let waits = leases.release().unwrap();
        assert_eq!(waits, &[6]);
        assert!(leases.release().is_err());
    }

    #[test]
    fn prefetch_defers_a_demand_and_admits_it_first() -> Result<()> {
        assert!(Prefetch::new(0, 2).is_err());
        assert!(Prefetch::new(3, 2).is_err());
        let mut prefetch = Prefetch::new(1, 2)?;
        assert_eq!(prefetch.request(0, 1, true, false)?.0, Requested::Queued);
        assert_eq!(
            prefetch.request(0, 2, true, false)?,
            (Requested::Deferred, None),
            "a demand that cannot be queued is held"
        );
        assert_eq!(prefetch.request(0, 1, true, false)?.0, Requested::Present);
        assert_eq!(prefetch.entries(), 1);
        assert_eq!(prefetch.begin_read(), Some((0, 1)));
        prefetch.complete(0, 1, 5)?;
        assert_eq!(prefetch.take(0, 1), Some(5));
        assert!(prefetch.release(5));
        assert_eq!(
            prefetch.begin_read(),
            Some((0, 2)),
            "the deferred demand is issued once D1 completes"
        );
        prefetch.complete(0, 2, 5)?;
        let error = prefetch.request(0, 3, false, true).unwrap_err().to_string();
        assert!(error.contains("not a disk prefetch"), "{error}");
        Ok(())
    }

    #[test]
    fn prefetch_read_waits_for_a_free_ring_slot() -> Result<()> {
        let mut prefetch = Prefetch::new(2, 2)?;
        prefetch.request(0, 0, false, false)?;
        prefetch.request(0, 1, false, false)?;
        assert_eq!(prefetch.begin_read(), Some((0, 0)));
        prefetch.complete(0, 0, 5)?;
        assert_eq!(prefetch.begin_read(), Some((0, 1)));
        prefetch.complete(0, 1, 7)?;
        assert_eq!(prefetch.take(0, 0), Some(5));
        assert_eq!(prefetch.take(0, 1), Some(7));
        assert_eq!(
            prefetch.request(0, 2, true, false)?,
            (Requested::Deferred, None),
            "the demand waits while both ring slots are taken"
        );
        assert_eq!(prefetch.begin_read(), None, "no ring slot is free");
        assert!(prefetch.release(5));
        assert_eq!(prefetch.begin_read(), Some((0, 2)));
        prefetch.complete(0, 2, 5)?;
        Ok(())
    }

    #[test]
    fn prefetch_cancel_retires_only_non_demand_entries() -> Result<()> {
        let mut prefetch = Prefetch::new(1, 2)?;
        prefetch.request(0, 0, false, false)?;
        assert_eq!(prefetch.begin_read(), Some((0, 0)));
        prefetch.complete(0, 0, 5)?;
        assert_eq!(prefetch.entries(), 1);
        assert_eq!(prefetch.cancel(0, 0)?, Retired::Loaded(5));
        assert_eq!(prefetch.entries(), 0);
        assert!(prefetch.cancel(0, 0).is_err());
        prefetch.request(0, 1, false, false)?;
        assert_eq!(prefetch.cancel(0, 1)?, Retired::Queued);
        assert_eq!(prefetch.entries(), 0);
        prefetch.request(0, 2, true, false)?;
        let error = prefetch.cancel(0, 2).unwrap_err().to_string();
        assert!(
            error.contains("demand 2 of layer 0 cannot be cancelled"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn prefetch_taken_slots_count_toward_the_depth() -> Result<()> {
        let mut prefetch = Prefetch::new(1, 2)?;
        prefetch.request(0, 0, false, false)?;
        assert_eq!(prefetch.begin_read(), Some((0, 0)));
        prefetch.complete(0, 0, 5)?;
        assert_eq!(prefetch.take(0, 0), Some(5));
        assert_eq!(prefetch.take(0, 0), None);
        assert_eq!(prefetch.entries(), 1);
        let error = prefetch
            .request(0, 1, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("prefetch queue full at depth 1"), "{error}");
        assert!(prefetch.release(5));
        assert!(!prefetch.release(5));
        prefetch.request(0, 1, false, false)?;
        assert_eq!(prefetch.entries(), 1);
        Ok(())
    }

    fn planner(
        geometry: &ExpertGeometry,
        staging_slots: u32,
        rates: &SpanRates<'_>,
    ) -> Result<SpanPlan> {
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            geometry,
        )?
        .slot_bytes();
        SpanPlan::new(
            geometry,
            slot_bytes,
            staging_slots,
            rates.service.worker_count()?,
        )
    }

    #[test]
    fn failed_update_leaves_the_previous_plan_intact() -> Result<()> {
        let geometry = geometry();
        let fast = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        let slow = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 100.0,
        };
        let counts = [4u32, 1, 1, 0, 2, 1, 0, 0];
        let resident = [true, false, false, false, false, false, false, false];
        let mut plan = planner(&geometry, 2, &fast)?;
        plan.update(&counts, &resident, 4, 2, &fast)?;
        let snapshot = plan.clone();
        let mut wide = counts;
        wide[1] = 16;
        let error = plan
            .update(&wide, &resident, 4, 2, &slow)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exceed the measured rows extent"), "{error}");
        assert_eq!(
            plan, snapshot,
            "a failed update must leave the previous plan intact"
        );
        Ok(())
    }

    #[test]
    fn span_plan_update_reuses_its_buffers() -> Result<()> {
        let geometry = geometry();
        let rates = SpanRates {
            cpu_row_limit: u32::MAX,
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        let counts_a = [4u32, 1, 1, 0, 2, 1, 0, 0];
        let resident_a = [true, false, false, false, false, false, false, false];
        let counts_b = [0u32, 1, 0, 0, 0, 0, 0, 0];
        let resident_b = [false; 8];
        let mut plan = planner(&geometry, 2, &rates)?;
        plan.update(&counts_a, &resident_a, 4, 2, &rates)?;
        let (resident, staged, cpu) = (
            plan.resident().as_ptr(),
            plan.staged().as_ptr(),
            plan.cpu().as_ptr(),
        );
        assert_eq!(plan.resident(), &[0]);
        assert_eq!(plan.staged(), &[1, 2, 4, 5]);
        plan.update(&counts_b, &resident_b, 4, 2, &rates)?;
        assert_eq!(plan.staged(), &[1]);
        plan.update(&counts_a, &resident_a, 4, 2, &rates)?;
        assert_eq!(plan.resident().as_ptr(), resident);
        assert_eq!(plan.staged().as_ptr(), staged);
        assert_eq!(plan.cpu().as_ptr(), cpu);
        Ok(())
    }

    #[test]
    fn staging_leases_release_reuses_its_wait_slice() -> Result<()> {
        let mut leases = StagingLeases::new(2)?;
        leases.reset(1)?;
        leases.load(&[1, 2], 1)?;
        leases.record_wave(0, 2)?;
        leases.record_wave(1, 3)?;
        let first = leases.release()?.as_ptr();
        leases.reset(1)?;
        leases.load(&[3, 4], 4)?;
        leases.record_wave(0, 5)?;
        leases.record_wave(1, 6)?;
        let waits = leases.release()?;
        assert_eq!(waits, &[5, 6]);
        assert_eq!(waits.as_ptr(), first);
        Ok(())
    }

    #[test]
    fn packed_waves_cover_cohort_assignments_within_active_capacity() -> Result<()> {
        let geometry = ExpertGeometry {
            layers: 1,
            experts: 256,
            hidden: 2048,
            intermediate: 512,
            top_k: 8,
        };
        let mut cells = Vec::new();
        for workers in [1u32, 2] {
            for rows in [1u32, 16, 256, 1024] {
                for experts in [1u32, 2] {
                    cells.push(ServiceCell {
                        workers,
                        rows,
                        experts,
                        seconds: f64::from(rows) * 0.01 * f64::from(experts) / f64::from(workers),
                    });
                }
            }
        }
        let service = ServiceTable::new("test_minicpm", cells)?;
        let rates = SpanRates {
            cpu_row_limit: 16,
            service: &service,
            h2d_bytes_per_second: 1e10,
        };
        let span_rows = 1024u32;
        let wave_rows = 256u32;
        let top_k = geometry.top_k as usize;
        let mut ids = Vec::with_capacity(span_rows as usize * top_k);
        let mut counts = vec![0u32; geometry.experts as usize];
        for r in 0..span_rows as usize {
            for c in 0..top_k {
                let expert = if c < 3 {
                    // 3 choices per row in 0..16 (resident): 3072 total assignments (> 2048).
                    ((r * 3 + c) % 16) as u32
                } else if c < 7 {
                    // 4 choices per row in 16..32 (staged, > 16 rows each).
                    16 + ((r * 5 + c * 3) % 16) as u32
                } else if r < 64 {
                    // 1 choice per row in 32..40 on first 64 rows (8 rows each <= 16 -> CPU).
                    32 + (r % 8) as u32
                } else {
                    16 + ((r * 7 + 13) % 16) as u32
                };
                ids.push(expert as i32);
                counts[expert as usize] += 1;
            }
        }
        let mut resident = vec![false; geometry.experts as usize];
        for flag in &mut resident[..16] {
            *flag = true;
        }
        let mut plan = planner(&geometry, 2, &rates)?;
        plan.update(&counts, &resident, span_rows, wave_rows, &rates)?;
        plan.pack_waves(&counts, &ids)?;

        // Resident cohort (3072 assignments, 3 per row) splits on a row boundary into 2046 + 1026.
        let res_waves = plan.cohort_waves(0)?;
        assert_eq!(res_waves, &[(0, 682, 2046), (682, 342, 1026)]);

        let total_cohorts = 1 + plan.cohort_count() as usize;
        for cohort in 0..total_cohorts {
            let experts: Vec<u32> = if cohort == 0 {
                plan.resident().to_vec()
            } else {
                plan.cohorts().nth(cohort - 1).unwrap().to_vec()
            };
            let waves = plan.cohort_waves(cohort)?;
            assert!(!waves.is_empty(), "non-empty cohort {cohort} has no waves");
            let mut prev_end = 0u32;
            for (w_idx, &(base, rows, active)) in waves.iter().enumerate() {
                assert!(rows > 0, "cohort {cohort} wave {w_idx} has zero rows");
                assert!(
                    active > 0 && active <= wave_rows * geometry.top_k,
                    "cohort {cohort} wave {w_idx} active={active} outside 1..=2048"
                );
                if w_idx > 0 {
                    assert!(
                        base >= prev_end,
                        "cohort {cohort} wave {w_idx} overlaps previous wave"
                    );
                }
                prev_end = base + rows;
                assert!(prev_end <= span_rows);
            }
            for r in 0..span_rows {
                let row_ids = &ids[r as usize * top_k..(r as usize + 1) * top_k];
                let row_active = row_ids
                    .iter()
                    .filter(|&&e| experts.contains(&(e as u32)))
                    .count();
                let covering: Vec<_> = waves
                    .iter()
                    .enumerate()
                    .filter(|(_, (base, rows, _))| r >= *base && r < *base + *rows)
                    .collect();
                if row_active > 0 {
                    assert_eq!(
                        covering.len(),
                        1,
                        "cohort {cohort} row {r} with {row_active} assignments covered {} times",
                        covering.len()
                    );
                } else {
                    assert!(covering.len() <= 1);
                }
            }
            for &(base, rows, active) in waves {
                let start = base as usize * top_k;
                let end = (base + rows) as usize * top_k;
                let actual = ids[start..end]
                    .iter()
                    .filter(|&&e| experts.contains(&(e as u32)))
                    .count() as u32;
                assert_eq!(actual, active, "cohort {cohort} wave ({base}, {rows})");
            }
        }

        let span_16k = 16_320u32;
        let mut ids_16k = Vec::with_capacity(span_16k as usize * top_k);
        let mut counts_16k = vec![0u32; geometry.experts as usize];
        for r in 0..span_16k as usize {
            for c in 0..top_k {
                let expert = ((r * 13 + c * 31) % geometry.experts as usize) as u32;
                ids_16k.push(expert as i32);
                counts_16k[expert as usize] += 1;
            }
        }
        let mut resident_16k = vec![false; geometry.experts as usize];
        for flag in &mut resident_16k[..118] {
            *flag = true;
        }
        plan.update(&counts_16k, &resident_16k, span_16k, wave_rows, &rates)?;
        let t0 = std::time::Instant::now();
        for _ in 0..40 {
            plan.pack_waves(&counts_16k, &ids_16k)?;
        }
        let elapsed = t0.elapsed();
        assert!(!plan.cohort_waves(0)?.is_empty());
        eprintln!(
            "pack_waves_16k span_rows={span_16k} layers=40 total_us={} per_layer_us={:.2}",
            elapsed.as_micros(),
            elapsed.as_secs_f64() * 1e6 / 40.0
        );
        Ok(())
    }
}
