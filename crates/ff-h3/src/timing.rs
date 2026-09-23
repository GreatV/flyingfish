//! Optional synchronized per-stage timing for the transformer hot path.
//!
//! `FF_H3_TIMING=1` writes one JSON record per phase to stderr;
//! `FF_H3_TIMING=<path>` appends the same records as JSON lines to `<path>`.
//! Unset, empty or `0` disables collection; the disabled path is one atomic
//! load per stage and no lock, allocation or device synchronization.

use anyhow::{Context, Result};
use serde::Serialize;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub const STAGE_TIMING_SCHEMA_VERSION: u32 = 1;
pub const STAGE_TIMING_ENVIRONMENT_VARIABLE: &str = "FF_H3_TIMING";

#[derive(Debug, Eq, PartialEq)]
enum Destination {
    Disabled,
    Stderr,
    File(Cow<'static, Path>),
}

fn destination_from_value(value: &str) -> Destination {
    match value {
        "" | "0" => Destination::Disabled,
        "1" => Destination::Stderr,
        other => Destination::File(Cow::Owned(Path::new(other).to_path_buf())),
    }
}

enum Output {
    Stderr,
    File(BufWriter<File>),
}

#[derive(Default)]
struct BucketTotals {
    load: Duration,
    compute: Duration,
    stages: u64,
}

struct Collector {
    output: Output,
    records: u64,
    buckets: BTreeMap<&'static str, BucketTotals>,
    prefetch_stages: u64,
    prefetch_fallback_tensors: u64,
    prefetch_skipped: u64,
    prefetch_fill: Duration,
}

#[derive(Serialize)]
struct BucketSummary {
    stages: u64,
    load_s: f64,
    compute_s: f64,
}

#[derive(Serialize)]
struct StageRecord {
    schema_version: u32,
    phase: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<usize>,
    records: u64,
    stages: BucketSummary,
    buckets: BTreeMap<&'static str, BucketSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_cache: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefetch: Option<PrefetchSummary>,
}

#[derive(Serialize)]
struct PrefetchSummary {
    stages: u64,
    fallback_tensors: u64,
    skipped_resident: u64,
    fill_s: f64,
}

#[derive(Serialize)]
struct CheckpointRecord {
    schema_version: u32,
    phase: &'static str,
    completed_evaluations: u64,
    checkpoint_write_s: f64,
}

impl Collector {
    fn open(destination: Destination) -> Option<Self> {
        let output = match destination {
            Destination::Disabled => return None,
            Destination::Stderr => Output::Stderr,
            Destination::File(path) => match open_append(&path) {
                Ok(file) => Output::File(BufWriter::new(file)),
                Err(error) => {
                    eprintln!(
                        "{STAGE_TIMING_ENVIRONMENT_VARIABLE} path {} cannot be opened ({}); \
                         stage timing falls back to stderr",
                        path.display(),
                        error
                    );
                    Output::Stderr
                }
            },
        };
        Some(Self {
            output,
            records: 0,
            buckets: BTreeMap::new(),
            prefetch_stages: 0,
            prefetch_fallback_tensors: 0,
            prefetch_skipped: 0,
            prefetch_fill: Duration::ZERO,
        })
    }

    fn record_stage(&mut self, bucket: &'static str, load: Duration, compute: Duration) {
        let totals = self.buckets.entry(bucket).or_default();
        totals.load += load;
        totals.compute += compute;
        totals.stages += 1;
    }

    fn take_stage_record(
        &mut self,
        phase: &'static str,
        step: Option<usize>,
        device_cache: Option<serde_json::Value>,
    ) -> StageRecord {
        let mut stages = BucketSummary {
            stages: 0,
            load_s: 0.0,
            compute_s: 0.0,
        };
        let mut buckets = BTreeMap::new();
        for (name, totals) in std::mem::take(&mut self.buckets) {
            let summary = BucketSummary {
                stages: totals.stages,
                load_s: totals.load.as_secs_f64(),
                compute_s: totals.compute.as_secs_f64(),
            };
            stages.stages += summary.stages;
            stages.load_s += summary.load_s;
            stages.compute_s += summary.compute_s;
            buckets.insert(name, summary);
        }
        self.records += 1;
        let prefetch = (self.prefetch_stages > 0).then_some(PrefetchSummary {
            stages: self.prefetch_stages,
            fallback_tensors: self.prefetch_fallback_tensors,
            skipped_resident: std::mem::take(&mut self.prefetch_skipped),
            fill_s: std::mem::take(&mut self.prefetch_fill).as_secs_f64(),
        });
        self.prefetch_stages = 0;
        self.prefetch_fallback_tensors = 0;
        StageRecord {
            schema_version: STAGE_TIMING_SCHEMA_VERSION,
            phase,
            step,
            records: self.records,
            stages,
            buckets,
            device_cache,
            prefetch,
        }
    }

    fn emit(&mut self, record: &impl Serialize) {
        let line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(_) => return,
        };
        match &mut self.output {
            Output::Stderr => eprintln!("h3 stage timing: {line}"),
            Output::File(writer) => {
                let _ = writeln!(writer, "{line}");
                let _ = writer.flush();
            }
        }
    }
}

fn open_append(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))
}

fn collector() -> Option<&'static Mutex<Collector>> {
    static COLLECTOR: OnceLock<Option<Mutex<Collector>>> = OnceLock::new();
    COLLECTOR
        .get_or_init(|| {
            let destination = std::env::var(STAGE_TIMING_ENVIRONMENT_VARIABLE)
                .map(|value| destination_from_value(&value))
                .unwrap_or(Destination::Disabled);
            Collector::open(destination).map(Mutex::new)
        })
        .as_ref()
}

pub(crate) fn enabled() -> bool {
    collector().is_some()
}

pub(crate) fn record_stage(bucket: &'static str, load: Duration, compute: Duration) {
    let Some(collector) = collector().and_then(|collector| collector.lock().ok()) else {
        return;
    };
    let mut collector = collector;
    collector.record_stage(bucket, load, compute);
}

#[cfg(feature = "cuda")]
pub(crate) fn record_prefetch(
    stages: u64,
    fallback_tensors: u64,
    skipped_resident: u64,
    fill: Duration,
) {
    let Some(mut collector) = collector().and_then(|collector| collector.lock().ok()) else {
        return;
    };
    collector.prefetch_stages += stages;
    collector.prefetch_fallback_tensors += fallback_tensors;
    collector.prefetch_skipped += skipped_resident;
    collector.prefetch_fill += fill;
}

pub(crate) fn finish_phase_with_cache(
    phase: &'static str,
    step: Option<usize>,
    device_cache: Option<serde_json::Value>,
) {
    let Some(mut collector) = collector().and_then(|collector| collector.lock().ok()) else {
        return;
    };
    let record = collector.take_stage_record(phase, step, device_cache);
    collector.emit(&record);
}

pub fn record_checkpoint_write(completed_evaluations: u64, elapsed: Duration) {
    let Some(mut collector) = collector().and_then(|collector| collector.lock().ok()) else {
        return;
    };
    collector.emit(&CheckpointRecord {
        schema_version: STAGE_TIMING_SCHEMA_VERSION,
        phase: "checkpoint",
        completed_evaluations,
        checkpoint_write_s: elapsed.as_secs_f64(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_collector() -> Collector {
        Collector {
            output: Output::Stderr,
            records: 0,
            buckets: BTreeMap::new(),
            prefetch_stages: 0,
            prefetch_fallback_tensors: 0,
            prefetch_skipped: 0,
            prefetch_fill: Duration::ZERO,
        }
    }

    #[test]
    fn environment_values_select_the_destination() {
        assert_eq!(destination_from_value(""), Destination::Disabled);
        assert_eq!(destination_from_value("0"), Destination::Disabled);
        assert_eq!(destination_from_value("1"), Destination::Stderr);
        assert_eq!(
            destination_from_value("h3-timing.jsonl"),
            Destination::File(Cow::Owned(Path::new("h3-timing.jsonl").to_path_buf()))
        );
    }

    #[test]
    fn stage_record_sums_buckets_and_resets_totals() {
        let mut collector = test_collector();
        collector.record_stage("adaln", Duration::from_millis(10), Duration::from_millis(5));
        collector.record_stage("adaln", Duration::from_millis(1), Duration::from_millis(2));
        collector.record_stage(
            "attention",
            Duration::from_millis(3),
            Duration::from_millis(7),
        );
        let record = collector.take_stage_record("eval", Some(12), None);
        assert_eq!(record.step, Some(12));
        assert_eq!(record.stages.stages, 3);
        assert!((record.stages.load_s - 0.014).abs() < 1e-9);
        assert!((record.stages.compute_s - 0.014).abs() < 1e-9);
        assert_eq!(record.buckets["adaln"].stages, 2);
        assert_eq!(record.buckets["attention"].stages, 1);
        let next = collector.take_stage_record("eval", None, None);
        assert_eq!(next.stages.stages, 0);
        assert_eq!(next.records, 2);
        assert!(next.buckets.is_empty());
    }

    #[test]
    fn records_serialize_with_the_documented_field_names() {
        let mut collector = test_collector();
        collector.record_stage("other", Duration::from_secs(2), Duration::from_secs(3));
        let json =
            serde_json::to_value(collector.take_stage_record("context", None, None)).unwrap();
        assert_eq!(json["schema_version"], STAGE_TIMING_SCHEMA_VERSION);
        assert_eq!(json["phase"], "context");
        assert!(json.get("step").is_none());
        assert_eq!(json["stages"]["stages"], 1);
        assert_eq!(json["stages"]["load_s"], 2.0);
        assert_eq!(json["buckets"]["other"]["compute_s"], 3.0);
        let json = serde_json::to_value(CheckpointRecord {
            schema_version: STAGE_TIMING_SCHEMA_VERSION,
            phase: "checkpoint",
            completed_evaluations: 7,
            checkpoint_write_s: 0.5,
        })
        .unwrap();
        assert_eq!(json["phase"], "checkpoint");
        assert_eq!(json["completed_evaluations"], 7);
        assert_eq!(json["checkpoint_write_s"], 0.5);
    }
}
