use super::{
    Device, calibrate,
    closure::Closure,
    ops::{Ops, grid},
};
use crate::backend::setup::{CacheKey, LinearChoice, LinearImpl, LinearShape};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg, sys};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Serialize, Deserialize)]
struct Trial {
    implementation: LinearImpl,
    samples: usize,
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    choice: LinearChoice,
    trials: Vec<Trial>,
    rel_rmse: f64,
    wall_ms: f64,
}

pub(super) fn supports(shape: LinearShape) -> bool {
    shape.rows > 0
        && shape.rows <= 16
        && shape.output > 0
        && shape.input > 0
        && shape.input.is_multiple_of(256)
}

pub(super) fn check_extent(shape: LinearShape, weights: usize, inputs: usize) -> Result<()> {
    ensure!(
        supports(shape),
        "invalid linear calibration shape: {shape:?}"
    );
    let required_w = shape
        .output
        .checked_mul(shape.input)
        .context("calibration weight extent overflow")?;
    let required_x = shape
        .rows
        .checked_mul(shape.input)
        .context("calibration input extent overflow")?;
    ensure!(
        weights == required_w && inputs == required_x,
        "calibration matrix extent mismatch: {shape:?}; weights={weights} required={required_w}, inputs={inputs} required={required_x}"
    );
    Ok(())
}

pub(super) struct Environment {
    pub key: CacheKey,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    residual: CudaFunction,
    l2_bytes: usize,
    runtime: PathBuf,
    closure: Rc<Closure>,
}

impl Environment {
    pub fn new(d: &Device, ops: &Ops, runtime: &Path, closure: Rc<Closure>) -> Result<Self> {
        ensure!(
            d.info.shared_bytes >= 40960,
            "skinny GEMM requires 40960 bytes shared memory"
        );
        Ok(Self {
            key: calibrate::key(d)?,
            ctx: d.ctx.clone(),
            stream: d.stream.clone(),
            residual: ops.residual.clone(),
            l2_bytes: d.info.l2_bytes,
            runtime: runtime.to_path_buf(),
            closure,
        })
    }

    pub fn path(&self, shape: LinearShape) -> PathBuf {
        self.runtime.join(format!(
            "linear-m{}-o{}-i{}-{}-{}-{}.json",
            shape.rows,
            shape.output,
            shape.input,
            self.key.device_uuid,
            self.key.driver_version,
            self.key.binary_version
        ))
    }
    pub fn read(&self, shape: LinearShape) -> Result<LinearChoice> {
        let path = self.path(shape);
        ensure!(
            self.closure.access(&path)?,
            "missing linear cache {}",
            path.display()
        );
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)?;
        check(&cache, &self.key, shape)?;
        Ok(cache.choice)
    }
    pub fn load(
        &self,
        shape: LinearShape,
        launch: impl Fn(&LinearImpl, &mut CudaSlice<bf16>) -> Result<()>,
    ) -> Result<LinearChoice> {
        ensure!(
            self.stream.capture_status()?
                == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "linear calibration must finish before Graph capture: {shape:?}"
        );
        let path = self.path(shape);
        let (cache, hit) = if self.closure.access(&path)? {
            let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)?;
            check(&cache, &self.key, shape)?;
            (cache, true)
        } else {
            let cache = self.measure(shape, launch)?;
            check(&cache, &self.key, shape)?;
            let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            f.write_all(&serde_json::to_vec_pretty(&cache)?)?;
            f.sync_all()?;
            std::fs::rename(tmp, &path)?;
            (cache, false)
        };
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"linear":{
            "shape":shape,"selection":cache.choice,"cached":hit,"calibration":cache,
            "path":path,"method":"single-op CUDA Graph; cold L2 outside timed events"}}})
        );
        Ok(cache.choice)
    }

    fn measure(
        &self,
        shape: LinearShape,
        launch: impl Fn(&LinearImpl, &mut CudaSlice<bf16>) -> Result<()>,
    ) -> Result<Cache> {
        self.closure.measurement()?;
        let started = Instant::now();
        let s = &self.stream;
        let mut out = s.alloc_zeros::<bf16>(shape.rows * shape.output)?;
        launch(&LinearImpl::Cublas, &mut out)?;
        let reference = s.clone_dtoh(&out)?;
        launch(&LinearImpl::Skinny, &mut out)?;
        let actual = s.clone_dtoh(&out)?;
        ensure!(
            reference.iter().chain(&actual).all(|v| v.is_finite()),
            "nonfinite linear calibration output: {shape:?}"
        );
        let signal: f64 = reference.iter().map(|v| (v.to_f32() as f64).powi(2)).sum();
        let error: f64 = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a.to_f32() as f64 - b.to_f32() as f64).powi(2))
            .sum();
        let rel_rmse = (error / signal.max(1e-30)).sqrt();
        ensure!(
            rel_rmse <= 0.02,
            "skinny/cuBLAS calibration rel_rmse={rel_rmse} exceeds 0.02 for {shape:?}"
        );
        let count = self.l2_bytes.max(1024 * 1024);
        let mut flush = s.alloc_zeros::<bf16>(count)?;
        let zeros = s.alloc_zeros::<bf16>(count)?;
        let tile = vec![bf16::ONE; 512 * 1024];
        for offset in (0..count).step_by(tile.len()) {
            let n = tile.len().min(count - offset);
            s.memcpy_htod(&tile[..n], &mut flush.slice_mut(offset..offset + n))?;
        }
        let begin = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let mut trials = Vec::new();
        for implementation in [LinearImpl::Cublas, LinearImpl::Skinny] {
            launch(&implementation, &mut out)?;
            s.synchronize()?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = launch(&implementation, &mut out);
            let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = captured?.context("empty linear candidate Graph")?;
            graph.upload()?;
            graph.launch()?;
            s.synchronize()?;
            let timer = Instant::now();
            let mut samples = Vec::new();
            while timer.elapsed() < Duration::from_millis(20) || samples.len() < 12 {
                unsafe {
                    s.launch_builder(&self.residual)
                        .arg(&mut flush)
                        .arg(&zeros)
                        .arg(&(count as i32))
                        .launch(grid(count.div_ceil(2048), 256))?;
                }
                begin.record(s)?;
                graph.launch()?;
                end.record(s)?;
                end.synchronize()?;
                samples.push(begin.elapsed_ms(&end)? as f64 * 1000.0);
            }
            trials.push(Trial {
                implementation,
                samples: samples.len(),
                median_us: calibrate::percentile(&samples, 0.5),
                p10_us: calibrate::percentile(&samples, 0.1),
                p90_us: calibrate::percentile(&samples, 0.9),
            });
        }
        let best = best(&trials)?;
        Ok(Cache {
            choice: LinearChoice {
                shape,
                key: self.key.clone(),
                implementation: best.implementation.clone(),
                median_us: best.median_us,
            },
            trials,
            rel_rmse,
            wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

fn best(trials: &[Trial]) -> Result<&Trial> {
    trials
        .iter()
        .min_by(|a, b| {
            a.median_us
                .total_cmp(&b.median_us)
                .then(a.p90_us.total_cmp(&b.p90_us))
        })
        .context("empty linear candidate measurements")
}

fn check(cache: &Cache, key: &CacheKey, shape: LinearShape) -> Result<()> {
    ensure!(
        cache.choice.key == *key && cache.choice.shape == shape,
        "linear cache identity mismatch"
    );
    ensure!(
        cache.trials.len() == 2
            && cache
                .trials
                .iter()
                .filter(|t| matches!(t.implementation, LinearImpl::Cublas))
                .count()
                == 1
            && cache
                .trials
                .iter()
                .filter(|t| matches!(t.implementation, LinearImpl::Skinny))
                .count()
                == 1,
        "linear cache candidate list mismatch"
    );
    ensure!(
        cache.rel_rmse.is_finite()
            && cache.rel_rmse <= 0.02
            && cache.rel_rmse >= 0.0
            && cache.trials.iter().all(|t| t.samples >= 12
                && t.median_us.is_finite()
                && t.median_us > 0.0
                && t.p90_us.is_finite()
                && t.p90_us >= t.median_us
                && t.p10_us.is_finite()
                && t.p10_us > 0.0),
        "invalid linear calibration measurements"
    );
    let best = best(&cache.trials)?;
    ensure!(
        std::mem::discriminant(&cache.choice.implementation)
            == std::mem::discriminant(&best.implementation)
            && cache.choice.median_us == best.median_us,
        "linear cache winner mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_rejects_stale_identity_and_wrong_winner() {
        let valid = LinearShape {
            rows: 1,
            output: 2048,
            input: 10240,
        };
        check_extent(valid, 2048 * 10240, 10240).unwrap();
        assert!(
            check_extent(
                LinearShape {
                    input: 86016,
                    ..valid
                },
                2048 * 10240,
                86016
            )
            .is_err()
        );
        assert!(check_extent(valid, 2048 * 10240 - 1, 10240).is_err());
        assert!(check_extent(valid, 2048 * 10240, 10239).is_err());
        let key = CacheKey {
            device_uuid: "device".into(),
            driver_version: "driver".into(),
            binary_version: "binary".into(),
        };
        let shape = LinearShape {
            rows: 8,
            output: 2048,
            input: 2048,
        };
        let mut cache = Cache {
            choice: LinearChoice {
                shape,
                key: key.clone(),
                implementation: LinearImpl::Skinny,
                median_us: 10.0,
            },
            trials: vec![
                Trial {
                    implementation: LinearImpl::Cublas,
                    samples: 12,
                    median_us: 20.0,
                    p10_us: 18.0,
                    p90_us: 22.0,
                },
                Trial {
                    implementation: LinearImpl::Skinny,
                    samples: 12,
                    median_us: 10.0,
                    p10_us: 9.0,
                    p90_us: 11.0,
                },
            ],
            rel_rmse: 0.003,
            wall_ms: 50.0,
        };
        check(&cache, &key, shape).unwrap();
        let mut stale = key.clone();
        stale.driver_version = "new driver".into();
        assert!(
            check(&cache, &stale, shape)
                .unwrap_err()
                .to_string()
                .contains("identity")
        );
        cache.choice.implementation = LinearImpl::Cublas;
        assert!(
            check(&cache, &key, shape)
                .unwrap_err()
                .to_string()
                .contains("winner")
        );
        cache.choice.implementation = LinearImpl::Skinny;
        cache.trials[1].median_us = f64::NAN;
        assert!(
            check(&cache, &key, shape)
                .unwrap_err()
                .to_string()
                .contains("measurements")
        );
    }
}
