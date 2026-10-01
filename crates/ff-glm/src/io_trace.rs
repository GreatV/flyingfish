use ff_core::telemetry::{
    ProcessWideIoFaultCounters, ProcessWideIoFaultDelta, TraceMeasurement,
    process_wide_io_fault_delta, process_wide_io_fault_sample,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FillReads {
    pub direct: u64,
    pub buffered: u64,
    pub mmap: u64,
}

#[cfg(feature = "cuda")]
static DIRECT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda")]
static BUFFERED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda")]
static MMAP: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "cuda")]
pub(crate) enum FillKind {
    #[cfg(target_os = "linux")]
    Direct,
    #[cfg(unix)]
    Buffered,
    Mmap,
}

fn enabled() -> anyhow::Result<bool> {
    static ENABLED: std::sync::OnceLock<std::result::Result<usize, String>> =
        std::sync::OnceLock::new();
    Ok(ff_core::probe::cached_env_usize(&ENABLED, "FF_GLM_IO_TRACE", 0, 0, 1)? == 1)
}

#[cfg(feature = "cuda")]
pub(crate) fn record_fill(kind: FillKind) -> anyhow::Result<()> {
    if !enabled()? {
        return Ok(());
    }
    let counter = match kind {
        #[cfg(target_os = "linux")]
        FillKind::Direct => &DIRECT,
        #[cfg(unix)]
        FillKind::Buffered => &BUFFERED,
        FillKind::Mmap => &MMAP,
    };
    counter.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn fills() -> FillReads {
    #[cfg(feature = "cuda")]
    {
        FillReads {
            direct: DIRECT.load(Ordering::Relaxed),
            buffered: BUFFERED.load(Ordering::Relaxed),
            mmap: MMAP.load(Ordering::Relaxed),
        }
    }
    #[cfg(not(feature = "cuda"))]
    FillReads::default()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IoStage {
    pub phase: String,
    pub index: Option<usize>,
    pub elapsed_ms: f64,
    pub counters: TraceMeasurement<ProcessWideIoFaultDelta>,
    pub pinned_weight_reads: FillReads,
}

pub struct IoTrace {
    started: Instant,
    previous: TraceMeasurement<ProcessWideIoFaultCounters>,
    stages: Vec<IoStage>,
    fills: FillReads,
}

impl IoTrace {
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        Ok(enabled()?.then(Self::new))
    }

    fn new() -> Self {
        Self {
            started: Instant::now(),
            previous: process_wide_io_fault_sample(),
            stages: Vec::new(),
            fills: fills(),
        }
    }

    pub fn record(&mut self, phase: &'static str, index: Option<usize>) {
        let now = Instant::now();
        let current = process_wide_io_fault_sample();
        let current_fills = fills();
        self.stages.push(IoStage {
            phase: phase.into(),
            index,
            elapsed_ms: (now - self.started).as_secs_f64() * 1000.0,
            counters: process_wide_io_fault_delta(&self.previous, &current),
            pinned_weight_reads: FillReads {
                direct: current_fills.direct.saturating_sub(self.fills.direct),
                buffered: current_fills.buffered.saturating_sub(self.fills.buffered),
                mmap: current_fills.mmap.saturating_sub(self.fills.mmap),
            },
        });
        self.previous = current;
        self.started = now;
        self.fills = current_fills;
    }

    pub fn finish(self) -> Vec<IoStage> {
        self.stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_are_ordered_and_serialize_availability() {
        let mut trace = IoTrace::new();
        trace.record("prefill", None);
        trace.record("decode_token", Some(0));
        let stages = trace.finish();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].phase, "prefill");
        assert_eq!(stages[1].index, Some(0));
        for stage in stages {
            assert!(stage.elapsed_ms.is_finite() && stage.elapsed_ms >= 0.0);
            let json = serde_json::to_value(&stage).unwrap();
            assert!(json["counters"]["availability"].is_string());
        }
    }
}
