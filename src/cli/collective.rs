//! `ff bench collective`.

use super::{ensure_new_output, publish_staged_bytes};
use anyhow::{Context, Result, bail};
use flyingfish::{
    collective_benchmark::{
        CollectiveBenchmarkOptions, CollectiveBenchmarkReport, CostModelBaseline, CostModelVerdict,
        PointToPointReference, RankCountMeasurement, measure_collective_benchmark,
    },
    runtime::artifact::ArtifactStaging,
};
use std::path::PathBuf;

pub(super) fn run_collective(
    output: PathBuf,
    devices: Vec<String>,
    ranks: Vec<usize>,
    single_device_evaluation_seconds: Option<f64>,
    charged_bytes_per_second: Option<f64>,
    warmups: usize,
    samples: usize,
) -> Result<()> {
    let cuda_ordinals = devices
        .iter()
        .map(|value| explicit_cuda_ordinal(value, "--devices"))
        .collect::<Result<Vec<_>>>()?;
    let options = CollectiveBenchmarkOptions {
        cuda_ordinals,
        rank_counts: ranks,
        warmup_iterations: warmups,
        sample_iterations: samples,
    };
    options.validate()?;
    ensure_new_output(&output, "collective benchmark output")?;
    let staging = ArtifactStaging::new(&output).with_context(|| {
        format!(
            "failed to stage collective benchmark report {}",
            output.display()
        )
    })?;
    let report = measure_collective_benchmark(options, |point_to_point| {
        // The projection is made only when the operator supplies the one rate
        // this run cannot measure for itself. The link rate it can measure, so
        // that is where the default comes from rather than a constant.
        let single_device_evaluation_seconds = single_device_evaluation_seconds?;
        let charged_bytes_per_second = charged_bytes_per_second.or(point_to_point)?;
        Some(CostModelBaseline {
            single_device_evaluation_seconds,
            charged_bytes_per_second,
        })
    })?;
    let json = serde_json::to_vec_pretty(&report)
        .context("failed to serialize the collective benchmark report")?;
    let _ = CollectiveBenchmarkReport::from_json(&json)?;
    let published = publish_staged_bytes(staging, &json)?;
    println!(
        "wrote collective benchmark schema {} report to {}",
        report.schema_version,
        published.destination.display()
    );
    report_to_stdout(&report);
    Ok(())
}

fn report_to_stdout(report: &CollectiveBenchmarkReport) {
    match &report.point_to_point_reference {
        PointToPointReference::Measured {
            source_cuda_ordinal,
            destination_cuda_ordinal,
            series,
        } => println!(
            "point to point, one pair active: cuda:{source_cuda_ordinal} to \
             cuda:{destination_cuda_ordinal} {:.3} GiB/s",
            gib_per_second(series.statistics.median_bytes_per_second)
        ),
        PointToPointReference::Unavailable {
            reason,
            available_cuda_devices,
        } => println!(
            "point to point unavailable ({reason:?}; {available_cuda_devices} usable CUDA devices)"
        ),
    }
    for measurement in &report.rank_counts {
        match measurement {
            RankCountMeasurement::Measured(ring) => {
                let rank_count = ring.rank_count;
                println!(
                    "N={rank_count}: concurrent neighbour {:.3} GiB/s per rank; ring transfer \
                     {:.3} GiB/s; ring all-reduce {:.3} GiB/s",
                    gib_per_second(
                        ring.concurrent_neighbor_d2d
                            .statistics
                            .median_bytes_per_second
                    ),
                    gib_per_second(
                        ring.ring_all_reduce_transfer_only
                            .statistics
                            .median_bytes_per_second
                    ),
                    gib_per_second(ring.ring_all_reduce.statistics.median_bytes_per_second),
                );
                println!(
                    "N={rank_count}: reduced output verified over {} elements",
                    ring.verification.verified_elements,
                );
            }
            RankCountMeasurement::Unavailable {
                rank_count,
                reason,
                available_cuda_devices,
            } => println!(
                "N={rank_count}: unavailable ({reason:?}; {available_cuda_devices} usable CUDA \
                 devices)"
            ),
        }
    }
    for row in &report.cost_model.rows {
        let derived = row.derived.map_or_else(
            || "no projection (no baseline supplied)".to_owned(),
            |derived| {
                format!(
                    "projected {:.2} s ({:.2}x)",
                    derived.total_seconds, derived.speedup
                )
            },
        );
        match &row.measured {
            Some(measured) => {
                let against = match (
                    measured.total_seconds,
                    measured.speedup,
                    measured.relative_total_error,
                ) {
                    (Some(total), Some(speedup), Some(error)) => format!(
                        ", measured {total:.2} s ({speedup:.2}x), relative error {:.1}%",
                        error * 100.0
                    ),
                    _ => format!(
                        ", measured {:.2} s of communication",
                        measured.communication_seconds
                    ),
                };
                println!(
                    "N={}: {derived}{against}{}",
                    row.rank_count,
                    measured
                        .concurrent_fraction_of_point_to_point
                        .map(|fraction| format!(
                            "; concurrent link at {:.1}% of the single-pair rate",
                            fraction * 100.0
                        ))
                        .unwrap_or_default(),
                );
            }
            None => println!("N={}: {derived}, not measured on this host", row.rank_count),
        }
    }
    println!(
        "cost-model verdict: {}",
        match report.cost_model.verdict {
            CostModelVerdict::Unmeasured =>
                "unmeasured — this host could not run the collective, so T0 is not answered",
            CostModelVerdict::Stands => "stands — the derived table predicts the measured totals",
            CostModelVerdict::Replaced =>
                "replaced — the measured totals differ but still finish sooner than one device",
            CostModelVerdict::Refuted =>
                "refuted — a measured rank count is no faster than one device",
        }
    );
}

fn explicit_cuda_ordinal(value: &str, flag: &str) -> Result<usize> {
    let ordinal = value.strip_prefix("cuda:").with_context(|| {
        format!("{flag} must name explicit cuda:N devices; the collective never selects hardware")
    })?;
    if ordinal.is_empty() || !ordinal.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{flag} has an invalid CUDA ordinal: {value:?}");
    }
    ordinal
        .parse::<usize>()
        .with_context(|| format!("{flag} CUDA ordinal exceeds usize"))
}

fn gib_per_second(bytes_per_second: f64) -> f64 {
    bytes_per_second / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_is_named_explicitly_and_never_inferred() {
        assert_eq!(explicit_cuda_ordinal("cuda:3", "--devices").unwrap(), 3);
        for value in ["auto", "cpu", "0", "metal:0", "cuda:", "cuda:-1", "cuda:1x"] {
            let error = explicit_cuda_ordinal(value, "--devices")
                .unwrap_err()
                .to_string();
            assert!(error.contains("--devices"), "{error}");
        }
    }

    #[test]
    fn an_invalid_ring_writes_nothing() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("must-not-exist.json");
        let error = run_collective(
            output.clone(),
            vec!["cuda:0".to_owned(), "cuda:0".to_owned()],
            vec![2],
            None,
            None,
            1,
            5,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("twice"), "{error}");
        assert!(!output.exists());
    }
}
