use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::{
    Device, cubin,
    ops::{Ops, grid},
};
use crate::backend::setup::CacheKey;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{PushKernelArg, sys};
use half::bf16;
use serde::{Deserialize, Serialize};

// Markov phase-1 small-body selection: F1 (cp.async pipeline) vs preF1
// (LDG.32 direct read), measured per P bucket {1, 8}, independent of
// context length. Selection metric: per-launch mean of a 7-launch burst
// after one L2 flush; the burst's first launch is stored as cold-launch
// evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Body {
    F1,
    PreF1,
}

impl Body {
    pub(super) fn symbol(self, p: usize) -> &'static str {
        match (self, p) {
            (Body::F1, 1) => "markov2_top4_phase1_p1",
            (Body::F1, 8) => "markov2_top4_phase1_p8",
            (Body::PreF1, 1) => "markov2_top4_phase1_preF1_p1",
            (Body::PreF1, 8) => "markov2_top4_phase1_preF1_p8",
            _ => unreachable!(),
        }
    }
}

const BUCKETS: [usize; 2] = [1, 8];
const BURST: usize = 7;

#[derive(Serialize, Deserialize)]
struct Trial {
    body: Body,
    bursts: usize,
    burst_mean_us: f64,
    burst_p90_us: f64,
    cold_first_us: f64,
}

#[derive(Serialize, Deserialize)]
struct Choice {
    p: usize,
    winner: Body,
    trials: Vec<Trial>,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    key: CacheKey,
    schema: u32,
    choices: Vec<Choice>,
    wall_ms: f64,
}

pub(super) struct Calibrated {
    cache: Cache,
    hit: bool,
    path: PathBuf,
}

impl Calibrated {
    pub(super) fn cached(&self) -> bool {
        self.hit
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn choice(&self, p: usize) -> Result<Body> {
        let choice = self
            .cache
            .choices
            .iter()
            .find(|c| c.p == p)
            .with_context(|| format!("Markov calibration has no P={p} bucket"))?;
        Ok(choice.winner)
    }
}

fn check(cache: &Cache, key: &CacheKey) -> Result<()> {
    ensure!(
        cache.schema == 1 && cache.key == *key,
        "markov phase-1 cache identity mismatch"
    );
    ensure!(
        cache.choices.len() == BUCKETS.len(),
        "markov phase-1 cache bucket count mismatch"
    );
    for &p in &BUCKETS {
        let choice = cache
            .choices
            .iter()
            .find(|c| c.p == p)
            .with_context(|| format!("markov phase-1 cache lacks P={p}"))?;
        ensure!(
            choice.trials.len() == 2
                && choice.trials.iter().filter(|t| t.body == Body::F1).count() == 1
                && choice
                    .trials
                    .iter()
                    .filter(|t| t.body == Body::PreF1)
                    .count()
                    == 1,
            "markov phase-1 cache P={p} trials must cover both bodies once"
        );
        ensure!(
            choice.trials.iter().all(|t| {
                t.bursts > 0
                    && t.burst_mean_us.is_finite()
                    && t.burst_mean_us > 0.0
                    && t.cold_first_us.is_finite()
                    && t.cold_first_us > 0.0
            }),
            "invalid markov phase-1 timing"
        );
        let best = choice
            .trials
            .iter()
            .min_by(|a, b| {
                a.burst_mean_us
                    .total_cmp(&b.burst_mean_us)
                    .then(a.burst_p90_us.total_cmp(&b.burst_p90_us))
            })
            .context("empty markov phase-1 bucket")?;
        ensure!(
            choice.winner == best.body,
            "markov phase-1 cache winner mismatch for P={p}"
        );
    }
    Ok(())
}

pub(super) fn load(d: &Device, ops: &Ops, dir: &Path) -> Result<Calibrated> {
    let key = super::calibrate::key(d)?;
    let path = dir.join(format!(
        "markov-phase1-v1-{}-{}-{}.json",
        key.device_uuid, key.driver_version, key.binary_version
    ));
    let (cache, hit) = if path.exists() {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)?;
        check(&cache, &key)?;
        (cache, true)
    } else {
        eprintln!(
            "{}",
            serde_json::json!({"markov_phase1_calibration":{"schema":1,"buckets":BUCKETS,"path":path,"cached":false,"action":"measure both bodies"}})
        );
        let cache = measure(d, ops, key.clone())?;
        check(&cache, &key)?;
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        f.write_all(&serde_json::to_vec_pretty(&cache)?)?;
        f.sync_all()?;
        std::fs::rename(temporary, &path)?;
        (cache, false)
    };
    Ok(Calibrated { cache, hit, path })
}

fn random(n: usize, mut state: u32) -> Vec<bf16> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32((state >> 8) as f32 / 16777216.0 - 0.5)
        })
        .collect()
}

fn measure(d: &Device, ops: &Ops, key: CacheKey) -> Result<Cache> {
    let start = Instant::now();
    let s = &d.stream;
    let module = cubin::module(&d.ctx, "markov")?;
    let base = s.clone_htod(&random(7 * 130560, 11))?;
    let w1 = s.clone_htod(&random(130560 * 256, 12))?;
    let w2 = s.clone_htod(&random(130560 * 256, 13))?;
    let flush_n = d.info.l2_bytes.max(1024 * 1024);
    let mut flush = s.clone_htod(&random(flush_n, 14))?;
    let zero = s.alloc_zeros::<bf16>(flush_n)?;
    let (begin, end) = (
        d.ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
        d.ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
    );
    let mut choices = Vec::new();
    for &p in &BUCKETS {
        let tokens: Vec<u32> = (0..p as u32).map(|i| 1000 + 7 * i).collect();
        let rows: Vec<i32> = (0..p).map(|i| (i % 7) as i32).collect();
        let tokens = s.clone_htod(&tokens)?;
        let rows = s.clone_htod(&rows)?;
        let partial_top = s.alloc_zeros::<f32>(510 * 64 * 8)?;
        let partial_lse = s.alloc_zeros::<f32>(510 * 64 * 2)?;
        let mut functions = Vec::new();
        for body in [Body::F1, Body::PreF1] {
            functions.push((body, module.load_function(body.symbol(p))?));
        }
        let state = super::markov::Phase1State {
            base: &base,
            w1: &w1,
            w2: &w2,
            tokens: &tokens,
            rows: &rows,
            partial_top: &partial_top,
            partial_lse: &partial_lse,
            p,
        };
        let reference = super::markov::phase1_bits(s, &functions[0].1, &state)?;
        let candidate = super::markov::phase1_bits(s, &functions[1].1, &state)?;
        ensure!(
            reference == candidate,
            "markov phase-1 bodies diverge bitwise at P={p}"
        );
        let mut trials = Vec::new();
        for &(body, ref function) in functions.iter() {
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = unsafe { super::markov::phase1_launch(s, function, &state) };
            let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = captured?.context("empty markov phase-1 candidate graph")?;
            graph.upload()?;
            graph.launch()?;
            s.synchronize()?;
            let timer = Instant::now();
            let mut burst_means = Vec::new();
            let mut cold_firsts = Vec::new();
            while timer.elapsed() < Duration::from_millis(40) || burst_means.len() < 12 {
                unsafe {
                    s.launch_builder(&ops.residual)
                        .arg(&mut flush)
                        .arg(&zero)
                        .arg(&(flush_n as i32))
                        .launch(grid(flush_n.div_ceil(2048), 256))?;
                }
                let mut launch_us = Vec::with_capacity(BURST);
                for _ in 0..BURST {
                    begin.record(s)?;
                    graph.launch()?;
                    end.record(s)?;
                    s.synchronize()?;
                    launch_us.push(begin.elapsed_ms(&end)? as f64 * 1000.0);
                }
                cold_firsts.push(launch_us[0]);
                burst_means.push(launch_us.iter().sum::<f64>() / BURST as f64);
            }
            trials.push(Trial {
                body,
                bursts: burst_means.len(),
                burst_mean_us: super::calibrate::percentile(&burst_means, 0.5),
                burst_p90_us: super::calibrate::percentile(&burst_means, 0.9),
                cold_first_us: super::calibrate::percentile(&cold_firsts, 0.5),
            });
        }
        let winner = trials
            .iter()
            .min_by(|a, b| {
                a.burst_mean_us
                    .total_cmp(&b.burst_mean_us)
                    .then(a.burst_p90_us.total_cmp(&b.burst_p90_us))
            })
            .context("empty markov phase-1 bucket")?
            .body;
        choices.push(Choice { p, winner, trials });
    }
    Ok(Cache {
        key,
        schema: 1,
        choices,
        wall_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}
