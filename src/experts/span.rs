//! Prefill span service: cohort classes and the wave plan for one layer's whole-span routing.
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
        ServiceTable::new(
            "test",
            vec![
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 0.01,
                },
                ServiceCell {
                    workers: 1,
                    rows: 2,
                    experts: 1,
                    seconds: 0.02,
                },
            ],
        )
        .unwrap()
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
        // 192 B at 1 GB/s is 0.192 us; rows=4 costs 40 ms from the 2-row cell scaled linearly.
        let plan = plan_span(&geometry, &counts, &resident, 4, 2, 2, &rates).unwrap();
        assert_eq!(plan.resident, vec![0]);
        assert_eq!(plan.staged, vec![vec![1, 2], vec![4, 5]]);
        assert!(plan.cpu.is_empty());
        assert_eq!(plan.streamed_bytes, 4 * 192);
        assert_eq!(plan.waves, 2);
        // 192 B at 100 B/s is 1.92 s, above every CPU cell here.
        rates.h2d_bytes_per_second = 100.0;
        let plan = plan_span(&geometry, &counts, &resident, 4, 2, 2, &rates).unwrap();
        assert!(plan.staged.is_empty());
        assert_eq!(plan.cpu, vec![1, 2, 4, 5]);
        assert_eq!(plan.streamed_bytes, 0);
    }

    #[test]
    fn expert_rows_scale_linearly_above_the_measured_window() {
        let table = table();
        assert_eq!(table.rows_time(2, 1, 1).unwrap(), 0.02);
        assert_eq!(table.rows_time(10, 1, 1).unwrap(), 0.1);
        assert!(table.rows_time(0, 1, 1).is_err());
        assert!(table.rows_time(10, 3, 1).is_err());
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
}
