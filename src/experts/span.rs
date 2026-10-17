//! Prefill span service: cohort classes and the wave plan for one layer's whole-span routing.
use crate::experts::arena::SlotEvents;
use crate::experts::catalogue::ExpertGeometry;
use crate::experts::plan::ServiceTable;
use anyhow::{Context, Result, ensure};

/// Measured rates the span classes are chosen from.
pub struct SpanRates<'a> {
    pub service: &'a ServiceTable,
    pub h2d_bytes_per_second: f64,
}

/// Service class of one routed expert in a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanClass {
    Resident,
    Streamed,
    Cpu,
}

/// Per-layer span execution plan: classes, staged cohorts and the wave count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanPlan {
    pub span_rows: u32,
    pub wave_rows: u32,
    pub waves: u32,
    pub resident: Vec<u32>,
    pub staged: Vec<Vec<u32>>,
    pub cpu: Vec<u32>,
    pub streamed_bytes: u64,
}

/// Plans one layer's span from the routed row counts and the current residency.
pub fn plan_span(
    geometry: &ExpertGeometry,
    counts: &[u32],
    resident: &[bool],
    span_rows: u32,
    wave_rows: u32,
    staging_slots: u32,
    rates: &SpanRates<'_>,
) -> Result<SpanPlan> {
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
    ensure!(staging_slots > 0, "zero prefill staging slots");
    ensure!(
        rates.h2d_bytes_per_second.is_finite() && rates.h2d_bytes_per_second > 0.0,
        "invalid H2D rate {}",
        rates.h2d_bytes_per_second
    );
    let slot_bytes = geometry.slot_bytes()?;
    let copy_seconds = slot_bytes as f64 / rates.h2d_bytes_per_second;
    let workers = rates.service.worker_count()?;
    let max_rows = rates.service.max_rows();
    let mut resident_experts = Vec::new();
    let mut streamed = Vec::new();
    let mut cpu = Vec::new();
    for (expert, &rows) in counts.iter().enumerate() {
        if rows == 0 {
            continue;
        }
        if resident[expert] {
            resident_experts.push(expert as u32);
            continue;
        }
        if rows > max_rows {
            let at_max = rates.service.rows_time(max_rows, 1, workers)?;
            ensure!(
                copy_seconds < at_max,
                "expert {expert} rows={rows} exceed the measured rows extent ..={max_rows} and the CPU route wins at {max_rows}"
            );
            streamed.push(expert as u32);
            continue;
        }
        let cpu_seconds = rates.service.rows_time(rows, 1, workers)?;
        if copy_seconds < cpu_seconds {
            streamed.push(expert as u32);
        } else {
            cpu.push(expert as u32);
        }
    }
    let mut staged = Vec::new();
    for cohort in streamed.chunks(staging_slots as usize) {
        staged.push(cohort.to_vec());
    }
    let streamed_bytes = u64::try_from(streamed.len())?
        .checked_mul(slot_bytes)
        .context("streamed span bytes overflow")?;
    Ok(SpanPlan {
        span_rows,
        wave_rows,
        waves: span_rows.div_ceil(wave_rows),
        resident: resident_experts,
        staged,
        cpu,
        streamed_bytes,
    })
}

/// Prefill staging leases: one cohort copy per round, pinned through every row wave.
pub struct StagingLeases {
    slots: u32,
    waves: u32,
    events: SlotEvents,
    ids: Vec<u32>,
    recorded: Vec<u32>,
    loaded: u32,
}

impl StagingLeases {
    pub fn new(slots: u32, waves: u32) -> Result<Self> {
        ensure!(slots > 0, "zero prefill staging slots");
        ensure!(waves > 0, "zero span row waves");
        Ok(Self {
            slots,
            waves,
            events: SlotEvents::new(slots as usize),
            ids: Vec::with_capacity(slots as usize),
            recorded: vec![0; slots as usize],
            loaded: 0,
        })
    }

    /// Records the cohort copy into staging slots `0..cohort.len()` under one event.
    pub fn load(&mut self, cohort: &[u32], event: u64) -> Result<()> {
        ensure!(
            self.loaded == 0,
            "staging cohort {} is still loaded",
            self.loaded
        );
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

    /// Releases a cohort that recorded every row wave and returns the reader events the next copy
    /// stream waits on.
    pub fn release(&mut self) -> Result<Vec<u64>> {
        ensure!(self.loaded > 0, "no staging cohort to release");
        for slot in 0..self.loaded {
            ensure!(
                self.recorded[slot as usize] == self.waves,
                "staging slot {slot} recorded {} of {} waves",
                self.recorded[slot as usize],
                self.waves
            );
        }
        let mut waits = Vec::new();
        for slot in 0..self.loaded {
            waits.append(&mut self.events.reclaim(slot)?);
        }
        self.loaded = 0;
        Ok(waits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::plan::{ServiceCell, ServiceTable};

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
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
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        // 192 B at 1 GB/s is 0.192 us, below every measured CPU cell here.
        let plan = plan_span(&geometry, &counts, &resident, 4, 2, 2, &rates).unwrap();
        assert_eq!(plan.resident, vec![0]);
        assert_eq!(plan.staged, vec![vec![1, 2], vec![4, 5]]);
        assert!(plan.cpu.is_empty());
        assert_eq!(plan.streamed_bytes, 4 * 192);
        assert_eq!(plan.waves, 2);
        // 192 B at 100 B/s is 1.92 s, above every measured CPU cell here.
        rates.h2d_bytes_per_second = 100.0;
        let plan = plan_span(&geometry, &counts, &resident, 4, 2, 2, &rates).unwrap();
        assert!(plan.staged.is_empty());
        assert_eq!(plan.cpu, vec![1, 2, 4, 5]);
        assert_eq!(plan.streamed_bytes, 0);
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
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        let plan = plan_span(&geometry, &counts, &resident, 16, 2, 2, &fast).unwrap();
        assert_eq!(plan.staged, vec![vec![1]]);
        assert!(plan.cpu.is_empty());
        let slow = SpanRates {
            service: &table(),
            h2d_bytes_per_second: 100.0,
        };
        let error = plan_span(&geometry, &counts, &resident, 16, 2, 2, &slow)
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
            service: &table(),
            h2d_bytes_per_second: 1e9,
        };
        assert!(plan_span(&geometry, &counts[..7], &resident, 4, 2, 2, &rates).is_err());
        assert!(plan_span(&geometry, &counts, &resident[..7], 4, 2, 2, &rates).is_err());
        assert!(plan_span(&geometry, &counts, &resident, 4, 8, 2, &rates).is_err());
        assert!(plan_span(&geometry, &counts, &resident, 4, 2, 0, &rates).is_err());
        let plan = plan_span(&geometry, &counts, &resident, 4, 2, 2, &rates).unwrap();
        assert!(plan.resident.is_empty() && plan.staged.is_empty() && plan.cpu.is_empty());
        assert_eq!(plan.waves, 2);
    }

    #[test]
    fn staging_leases_pin_a_cohort_through_every_wave() -> Result<()> {
        let mut leases = StagingLeases::new(3, 2)?;
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
        assert_eq!(waits, vec![10, 20, 11, 21, 12, 22]);
        leases.load(&[1, 2], 30)?;
        let error = leases.record_wave(2, 31).unwrap_err().to_string();
        assert!(error.contains("outside the loaded cohort 2"), "{error}");
        Ok(())
    }

    #[test]
    fn staging_leases_validate_cohorts_events_and_waves() {
        assert!(StagingLeases::new(0, 1).is_err());
        assert!(StagingLeases::new(1, 0).is_err());
        let mut leases = StagingLeases::new(2, 1).unwrap();
        assert!(leases.load(&[1, 2, 3], 1).is_err());
        assert!(leases.load(&[], 1).is_err());
        leases.load(&[1], 4).unwrap();
        leases.record_wave(0, 6).unwrap();
        assert!(leases.record_wave(1, 7).is_err());
        let waits = leases.release().unwrap();
        assert_eq!(waits, vec![6]);
        assert!(leases.release().is_err());
    }
}
