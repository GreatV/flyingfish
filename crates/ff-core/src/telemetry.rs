use anyhow::{Context, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const TELEMETRY_THREAD_NAME: &str = "ff-telemetry";
pub const RUNTIME_TELEMETRY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case", deny_unknown_fields)]
pub enum TraceMeasurement<T> {
    Available { value: T },
    Unavailable { reason: String },
}

impl<T> TraceMeasurement<T> {
    pub fn available(value: T) -> Self {
        Self::Available { value }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageBoundaryMemorySample {
    pub process_rss_bytes: TraceMeasurement<u64>,
    pub device_used: TraceMeasurement<DeviceUsedMemorySample>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceUsedMemorySample {
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub measurement_scope: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessWideIoFaultCounters {
    minor_page_faults: u64,
    major_page_faults: u64,
    rchar_bytes: u64,
    read_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessWideIoFaultDelta {
    pub measurement_scope: String,
    pub minor_page_faults: u64,
    pub major_page_faults: u64,
    pub rchar_bytes: u64,
    pub read_bytes: u64,
}

impl StageBoundaryMemorySample {
    pub fn validate(&self) -> Result<()> {
        validate_trace_measurement(&self.process_rss_bytes)?;
        validate_trace_measurement(&self.device_used)?;
        if let TraceMeasurement::Available { value } = &self.device_used {
            anyhow::ensure!(
                value.total_bytes > 0 && value.used_bytes <= value.total_bytes,
                "stage trace device-used memory exceeds total memory"
            );
            anyhow::ensure!(
                value.measurement_scope == "device_wide_point_sample",
                "stage trace has an unknown device-memory measurement scope"
            );
        }
        Ok(())
    }
}

impl ProcessWideIoFaultDelta {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.measurement_scope == "process_wide_proxy",
            "stage trace has an unknown process-counter measurement scope"
        );
        Ok(())
    }
}

fn validate_trace_measurement<T>(measurement: &TraceMeasurement<T>) -> Result<()> {
    if let TraceMeasurement::Unavailable { reason } = measurement {
        anyhow::ensure!(
            !reason.trim().is_empty(),
            "unavailable trace measurement requires a reason"
        );
    }
    Ok(())
}

impl ProcessWideIoFaultCounters {
    fn delta_since(&self, before: &Self) -> Option<ProcessWideIoFaultDelta> {
        Some(ProcessWideIoFaultDelta {
            measurement_scope: "process_wide_proxy".to_owned(),
            minor_page_faults: self
                .minor_page_faults
                .checked_sub(before.minor_page_faults)?,
            major_page_faults: self
                .major_page_faults
                .checked_sub(before.major_page_faults)?,
            rchar_bytes: self.rchar_bytes.checked_sub(before.rchar_bytes)?,
            read_bytes: self.read_bytes.checked_sub(before.read_bytes)?,
        })
    }
}

pub fn process_wide_io_fault_delta(
    before: &TraceMeasurement<ProcessWideIoFaultCounters>,
    after: &TraceMeasurement<ProcessWideIoFaultCounters>,
) -> TraceMeasurement<ProcessWideIoFaultDelta> {
    match (before, after) {
        (
            TraceMeasurement::Available { value: before },
            TraceMeasurement::Available { value: after },
        ) => after.delta_since(before).map_or_else(
            || TraceMeasurement::unavailable("a process-wide counter moved backwards"),
            TraceMeasurement::available,
        ),
        (TraceMeasurement::Unavailable { reason }, _) => {
            TraceMeasurement::unavailable(format!("start sample unavailable: {reason}"))
        }
        (_, TraceMeasurement::Unavailable { reason }) => {
            TraceMeasurement::unavailable(format!("end sample unavailable: {reason}"))
        }
    }
}

pub fn stage_boundary_memory_sample(device: &Device) -> StageBoundaryMemorySample {
    let process_rss_bytes = match process_memory() {
        Ok((rss, _)) => TraceMeasurement::available(rss),
        Err(()) => TraceMeasurement::unavailable("process RSS is unavailable on this platform"),
    };
    let device_used = match device_memory_result(device) {
        Ok(Some((used_bytes, total_bytes))) => {
            TraceMeasurement::available(DeviceUsedMemorySample {
                used_bytes,
                total_bytes,
                measurement_scope: "device_wide_point_sample".to_owned(),
            })
        }
        Ok(None) => TraceMeasurement::unavailable(
            "device-used memory point sampling is unsupported for this backend",
        ),
        Err(reason) => TraceMeasurement::unavailable(reason),
    };
    StageBoundaryMemorySample {
        process_rss_bytes,
        device_used,
    }
}

#[cfg(target_os = "linux")]
pub fn process_wide_io_fault_sample() -> TraceMeasurement<ProcessWideIoFaultCounters> {
    let stat = match std::fs::read_to_string("/proc/self/stat") {
        Ok(stat) => stat,
        Err(error) => {
            return TraceMeasurement::unavailable(format!(
                "failed to read /proc/self/stat: {error}"
            ));
        }
    };
    let io = match std::fs::read_to_string("/proc/self/io") {
        Ok(io) => io,
        Err(error) => {
            return TraceMeasurement::unavailable(format!("failed to read /proc/self/io: {error}"));
        }
    };
    match (parse_linux_stat_faults(&stat), parse_linux_io(&io)) {
        (Some((minor_page_faults, major_page_faults)), Some((rchar_bytes, read_bytes))) => {
            TraceMeasurement::available(ProcessWideIoFaultCounters {
                minor_page_faults,
                major_page_faults,
                rchar_bytes,
                read_bytes,
            })
        }
        _ => TraceMeasurement::unavailable("failed to parse Linux process counters"),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn process_wide_io_fault_sample() -> TraceMeasurement<ProcessWideIoFaultCounters> {
    TraceMeasurement::unavailable("process-wide I/O and fault counters require Linux procfs")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTelemetryReport {
    pub schema_version: u32,
    pub elapsed_ms: u64,
    pub samples: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub peak_process_rss_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub process_high_watermark_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_total_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_baseline_used_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_peak_used_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_peak_delta_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_measurement_scope: Option<String>,
    pub process_sampling_errors: u64,
    pub device_sampling_errors: u64,
}

impl RuntimeTelemetryReport {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "unsupported runtime-telemetry schema {}; this build supports schema {}",
            self.schema_version,
            RUNTIME_TELEMETRY_SCHEMA_VERSION
        );
        anyhow::ensure!(self.samples > 0, "runtime telemetry has no samples");
        anyhow::ensure!(
            self.process_sampling_errors <= self.samples
                && self.device_sampling_errors <= self.samples,
            "runtime telemetry per-source errors exceed sample count"
        );
        match (
            self.cuda_total_bytes,
            self.cuda_baseline_used_bytes,
            self.cuda_peak_used_bytes,
            self.cuda_peak_delta_bytes,
            self.cuda_measurement_scope.as_deref(),
        ) {
            (None, None, None, None, None) => {}
            (Some(total), Some(baseline), Some(peak), Some(delta), Some(scope)) => {
                anyhow::ensure!(
                    baseline <= peak && peak <= total,
                    "runtime telemetry CUDA baseline/peak/total are inconsistent"
                );
                anyhow::ensure!(
                    delta == peak - baseline,
                    "runtime telemetry CUDA peak delta is inconsistent"
                );
                anyhow::ensure!(
                    scope == "device_wide_baseline_delta",
                    "runtime telemetry has an unknown CUDA measurement scope"
                );
            }
            _ => anyhow::bail!("runtime telemetry has incomplete CUDA measurements"),
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let report: Self =
            serde_json::from_slice(bytes).context("invalid runtime-telemetry JSON")?;
        report.validate()?;
        Ok(report)
    }
}

#[derive(Default)]
struct SampleState {
    samples: u64,
    peak_process_rss_bytes: Option<u64>,
    process_high_watermark_bytes: Option<u64>,
    cuda_total_bytes: Option<u64>,
    cuda_baseline_used_bytes: Option<u64>,
    cuda_peak_used_bytes: Option<u64>,
    process_sampling_errors: u64,
    device_sampling_errors: u64,
}

pub struct TelemetryMonitor {
    started: Instant,
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<SampleState>>,
    device: Option<Device>,
    worker: Option<JoinHandle<()>>,
}

impl TelemetryMonitor {
    pub fn start(device: Option<Device>, interval: Duration) -> Result<Self> {
        anyhow::ensure!(!interval.is_zero(), "telemetry interval must be non-zero");
        let started = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(SampleState::default()));
        if let Some(device) = device.as_ref() {
            device.synchronize()?;
        }
        sample(device.as_ref(), &state, true)
            .map_err(anyhow::Error::msg)
            .context("failed to establish telemetry device-memory baseline")?;
        let worker_stop = Arc::clone(&stop);
        let worker_state = Arc::clone(&state);
        let worker_device = device.clone();
        let worker = thread::Builder::new()
            .name(TELEMETRY_THREAD_NAME.to_owned())
            .spawn(move || {
                while !worker_stop.load(Ordering::Relaxed) {
                    let _ = sample(worker_device.as_ref(), &worker_state, false);
                    thread::sleep(interval);
                }
                let _ = sample(worker_device.as_ref(), &worker_state, false);
            })
            .context("failed to start telemetry sampler")?;
        Ok(Self {
            started,
            stop,
            state,
            device,
            worker: Some(worker),
        })
    }

    pub fn finish(mut self) -> Result<RuntimeTelemetryReport> {
        if let Some(device) = self.device.as_ref() {
            device.synchronize()?;
        }
        let _ = sample(self.device.as_ref(), &self.state, false);
        self.stop_and_join()?;
        let state = self.state.lock().expect("telemetry mutex poisoned");
        let cuda_peak_delta_bytes =
            match (state.cuda_peak_used_bytes, state.cuda_baseline_used_bytes) {
                (Some(peak), Some(baseline)) => Some(peak.saturating_sub(baseline)),
                _ => None,
            };
        let report = RuntimeTelemetryReport {
            schema_version: RUNTIME_TELEMETRY_SCHEMA_VERSION,
            elapsed_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            samples: state.samples,
            peak_process_rss_bytes: state.peak_process_rss_bytes,
            process_high_watermark_bytes: state.process_high_watermark_bytes,
            cuda_total_bytes: state.cuda_total_bytes,
            cuda_baseline_used_bytes: state.cuda_baseline_used_bytes,
            cuda_peak_used_bytes: state.cuda_peak_used_bytes,
            cuda_peak_delta_bytes,
            cuda_measurement_scope: state
                .cuda_total_bytes
                .map(|_| "device_wide_baseline_delta".to_owned()),
            process_sampling_errors: state.process_sampling_errors,
            device_sampling_errors: state.device_sampling_errors,
        };
        report.validate()?;
        Ok(report)
    }

    fn stop_and_join(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("telemetry sampler panicked"))?;
        }
        Ok(())
    }
}

impl Drop for TelemetryMonitor {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

fn sample(
    device: Option<&Device>,
    state: &Mutex<SampleState>,
    establish_device_baseline: bool,
) -> std::result::Result<(), String> {
    let process = process_memory();
    let device_memory = device.map(device_memory_result);
    let mut state = state.lock().expect("telemetry mutex poisoned");
    state.samples += 1;
    match process {
        Ok((rss, high_watermark)) => {
            state.peak_process_rss_bytes = max_optional(state.peak_process_rss_bytes, rss);
            state.process_high_watermark_bytes =
                max_optional(state.process_high_watermark_bytes, high_watermark);
        }
        Err(()) => {
            state.process_sampling_errors += 1;
        }
    }
    match device_memory {
        Some(Ok(Some((used, total)))) => {
            state.cuda_total_bytes = Some(total);
            if establish_device_baseline {
                state.cuda_baseline_used_bytes = Some(used);
            }
            state.cuda_peak_used_bytes = max_optional(state.cuda_peak_used_bytes, used);
        }
        Some(Err(reason)) => {
            state.device_sampling_errors += 1;
            return Err(reason);
        }
        Some(Ok(None)) | None => {}
    }
    Ok(())
}

fn max_optional(current: Option<u64>, value: u64) -> Option<u64> {
    Some(current.map_or(value, |current| current.max(value)))
}

#[cfg(target_os = "linux")]
fn process_memory() -> std::result::Result<(u64, u64), ()> {
    let status = std::fs::read_to_string("/proc/self/status").map_err(|_| ())?;
    parse_linux_status(&status).ok_or(())
}

/// Current and peak working set, Windows' counterpart to `VmRSS`/`VmHWM`.
///
/// Both name the resident portion of the process. The kernels manage
/// residency differently — Windows trims working sets under pressure more
/// eagerly than Linux reclaims — so these are comparable across runs on one
/// host, not across platforms.
#[cfg(windows)]
fn process_memory() -> std::result::Result<(u64, u64), ()> {
    use windows_sys::Win32::System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };

    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    counters.cb = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS>()).map_err(|_| ())?;
    let read = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &raw mut counters, counters.cb) };
    if read == 0 {
        return Err(());
    }
    Ok((
        u64::try_from(counters.WorkingSetSize).map_err(|_| ())?,
        u64::try_from(counters.PeakWorkingSetSize).map_err(|_| ())?,
    ))
}

#[cfg(not(any(target_os = "linux", windows)))]
fn process_memory() -> std::result::Result<(u64, u64), ()> {
    Err(())
}

#[cfg(target_os = "linux")]
fn parse_linux_status(status: &str) -> Option<(u64, u64)> {
    let value = |name: &str| {
        status.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?;
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            kib.checked_mul(1024)
        })
    };
    Some((value("VmRSS:")?, value("VmHWM:")?))
}

#[cfg(target_os = "linux")]
fn parse_linux_stat_faults(stat: &str) -> Option<(u64, u64)> {
    let (_, fields) = stat.trim().rsplit_once(") ")?;
    let fields = fields.split_whitespace().collect::<Vec<_>>();
    Some((fields.get(7)?.parse().ok()?, fields.get(9)?.parse().ok()?))
}

#[cfg(target_os = "linux")]
fn parse_linux_io(io: &str) -> Option<(u64, u64)> {
    let value = |name: &str| {
        io.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?;
            rest.trim().parse::<u64>().ok()
        })
    };
    Some((value("rchar:")?, value("read_bytes:")?))
}

#[cfg(feature = "cuda")]
fn device_memory_result(device: &Device) -> std::result::Result<Option<(u64, u64)>, String> {
    if !device.is_cuda() {
        return Ok(None);
    }
    let cuda = device
        .as_cuda_device()
        .map_err(|error| format!("failed to access CUDA device for memory sample: {error}"))?;
    let stream = cuda.cuda_stream();
    let (free, total) = stream
        .context()
        .mem_get_info()
        .map_err(|error| format!("CUDA mem_get_info failed: {error}"))?;
    let used = total
        .checked_sub(free)
        .ok_or_else(|| "CUDA free memory exceeds total memory".to_owned())?;
    let used = u64::try_from(used).map_err(|_| "CUDA used-memory sample exceeds u64".to_owned())?;
    let total =
        u64::try_from(total).map_err(|_| "CUDA total-memory sample exceeds u64".to_owned())?;
    Ok(Some((used, total)))
}

#[cfg(not(feature = "cuda"))]
fn device_memory_result(_device: &Device) -> std::result::Result<Option<(u64, u64)>, String> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_name_fits_linux_comm_limit() {
        assert!(TELEMETRY_THREAD_NAME.len() <= 15);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_process_memory() {
        let status = "Name:\ttest\nVmHWM:\t  2048 kB\nVmRSS:\t 1024 kB\n";
        assert_eq!(parse_linux_status(status), Some((1 << 20, 2 << 20)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_faults_after_a_complex_command_name() {
        let stat = "42 (worker ) name) R 1 2 3 4 5 6 70 8 90 10 11 12\n";
        assert_eq!(parse_linux_stat_faults(stat), Some((70, 90)));
        assert_eq!(parse_linux_stat_faults("malformed"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_linux_process_io_counters_by_name() {
        let io = "rchar: 123\nwchar: 8\nsyscr: 2\nread_bytes: 456\n";
        assert_eq!(parse_linux_io(io), Some((123, 456)));
        assert_eq!(parse_linux_io("rchar: 1\n"), None);
    }

    #[test]
    fn process_counter_delta_is_explicitly_a_process_wide_proxy() {
        let before = TraceMeasurement::available(ProcessWideIoFaultCounters {
            minor_page_faults: 10,
            major_page_faults: 2,
            rchar_bytes: 100,
            read_bytes: 40,
        });
        let after = TraceMeasurement::available(ProcessWideIoFaultCounters {
            minor_page_faults: 13,
            major_page_faults: 3,
            rchar_bytes: 150,
            read_bytes: 48,
        });
        assert_eq!(
            process_wide_io_fault_delta(&before, &after),
            TraceMeasurement::available(ProcessWideIoFaultDelta {
                measurement_scope: "process_wide_proxy".to_owned(),
                minor_page_faults: 3,
                major_page_faults: 1,
                rchar_bytes: 50,
                read_bytes: 8,
            })
        );
    }

    #[test]
    fn cpu_boundary_sample_does_not_claim_device_memory() {
        let sample = stage_boundary_memory_sample(&Device::Cpu);
        assert!(!sample.device_used.is_available());
        #[cfg(target_os = "linux")]
        assert!(sample.process_rss_bytes.is_available());
    }

    #[test]
    fn monitor_stops_and_reports_samples() {
        let monitor = TelemetryMonitor::start(None, Duration::from_millis(1)).unwrap();
        thread::sleep(Duration::from_millis(3));
        let report = monitor.finish().unwrap();
        assert_eq!(report.schema_version, RUNTIME_TELEMETRY_SCHEMA_VERSION);
        assert!(report.samples >= 1);
        assert!(report.process_sampling_errors <= report.samples);
        assert!(report.device_sampling_errors <= report.samples);
        #[cfg(target_os = "linux")]
        assert!(report.peak_process_rss_bytes.is_some());
    }

    #[test]
    fn rejects_zero_interval() {
        assert!(TelemetryMonitor::start(None, Duration::ZERO).is_err());
    }

    #[test]
    fn runtime_report_requires_split_error_fields() {
        let report = TelemetryMonitor::start(None, Duration::from_millis(1))
            .unwrap()
            .finish()
            .unwrap();
        let mut value = serde_json::to_value(report).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("device_sampling_errors");
        assert!(RuntimeTelemetryReport::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
