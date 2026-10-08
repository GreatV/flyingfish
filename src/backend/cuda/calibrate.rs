use super::{
    Device,
    decode::Attention,
    flash::Shape,
    ops::{Ops, grid},
    verification::Verification,
};
use crate::{
    backend::setup::{AttentionPlan, CacheKey, Selection},
    config::Config,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{PushKernelArg, sys};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    fs::File,
    hash::Hasher,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

const LENGTHS: [usize; 3] = [1024, 8192, 32768];
const CAPACITIES: [usize; 3] = [1280, 8448, 33024];

#[derive(Clone, Serialize, Deserialize)]
struct Trial {
    length: usize,
    plan: AttentionPlan,
    samples: usize,
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
    wall_ms: f64,
    max_abs_error: f64,
    evaluation_lengths: [usize; 4],
}

#[derive(Serialize, Deserialize)]
struct Cache {
    #[serde(default)]
    query_rows: Option<usize>,
    schema: u32,
    key: CacheKey,
    winners: [AttentionPlan; 3],
    trials: Vec<Trial>,
    wall_ms: f64,
}

#[derive(Serialize)]
pub struct Calibrated {
    cache: Cache,
    hit: bool,
    path: PathBuf,
}

impl Calibrated {
    pub fn choose(&self, capacity: usize) -> Result<(AttentionPlan, Selection)> {
        let bucket = if capacity <= 2048 {
            0
        } else if capacity <= 16384 {
            1
        } else {
            2
        };
        Ok((
            self.cache.winners[bucket],
            Selection::Calibrated {
                key: self.cache.key.clone(),
                cached: self.hit,
            },
        ))
    }
}

pub fn runtime_dir(path: &Path, model: &Path) -> Result<PathBuf> {
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut parent = full.as_path();
    let mut suffix = Vec::new();
    while !parent.exists() {
        suffix.push(
            parent
                .file_name()
                .context("invalid runtime directory")?
                .to_os_string(),
        );
        parent = parent
            .parent()
            .context("runtime directory has no existing ancestor")?;
    }
    let mut resolved = parent.canonicalize()?;
    for part in suffix.iter().rev() {
        resolved.push(part);
    }
    let model = model.canonicalize()?;
    ensure!(
        !resolved.starts_with(&model)
            && !resolved
                .components()
                .any(|c| matches!(c, Component::Normal(p) if p == "models")),
        "runtime directory must be outside all readonly models directories"
    );
    std::fs::create_dir_all(&resolved)?;
    ensure!(resolved.is_dir(), "runtime path is not a directory");
    Ok(resolved)
}

pub(super) fn key(d: &Device) -> Result<CacheKey> {
    let bytes = d.ctx.uuid()?.bytes;
    let hex: String = bytes.iter().map(|&b| format!("{:02x}", b as u8)).collect();
    let uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    let result = Command::new("nvidia-smi")
        .args(["--query-gpu=uuid,driver_version", "--format=csv,noheader"])
        .output()
        .context("query NVIDIA driver version for calibration cache")?;
    ensure!(
        result.status.success(),
        "nvidia-smi driver query failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout)?;
    let version = stdout
        .lines()
        .find_map(|line| {
            let (id, version) = line.split_once(',')?;
            (id.trim() == uuid).then(|| version.trim().to_owned())
        })
        .context("CUDA UUID was not found in nvidia-smi driver query")?;
    ensure!(
        !version.is_empty() && version.chars().all(|c| c.is_ascii_digit() || c == '.'),
        "invalid driver version in calibration cache key"
    );
    let mut file = File::open(std::env::current_exe()?)?;
    let mut hash = DefaultHasher::new();
    let mut block = [0u8; 65536];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.write(&block[..n]);
    }
    Ok(CacheKey {
        device_uuid: uuid,
        driver_version: version,
        binary_version: format!("hash64-{:016x}", hash.finish()),
    })
}

fn rank(t: &Trial) -> (f64, f64, usize) {
    (t.median_us, t.p90_us, usize::MAX - t.plan.chunk)
}

fn check(cache: &Cache, key: &CacheKey, query_rows: Option<usize>) -> Result<()> {
    let packs: &[usize] = if query_rows.is_some() {
        &[1, 2, 4, 8]
    } else {
        &[2, 4]
    };
    let candidates = packs.len() * 8;
    ensure!(
        cache.schema == 1
            && cache.key == *key
            && cache.query_rows == query_rows
            && cache.trials.len() == candidates * 3,
        "invalid attention calibration cache schema/key/trial count"
    );
    for (bucket, length) in LENGTHS.iter().enumerate() {
        let trials: Vec<_> = cache
            .trials
            .iter()
            .filter(|t| t.length == *length)
            .collect();
        ensure!(
            trials.len() == candidates,
            "calibration cache lacks {candidates} trials for length {length}"
        );
        for chunk in [128, 256, 512, 1024] {
            for &qpack in packs {
                for threads in [128, 256] {
                    ensure!(
                        trials
                            .iter()
                            .filter(|t| t.plan.chunk == chunk
                                && t.plan.qpack == qpack
                                && t.plan.threads == threads
                                && t.plan.merge_threads == 256)
                            .count()
                            == 1,
                        "calibration cache candidate lattice is incomplete"
                    );
                }
            }
        }
        ensure!(
            trials
                .iter()
                .all(|t| t.samples > 0 && t.median_us.is_finite() && t.median_us > 0.0),
            "invalid calibration sample"
        );
        let best = trials
            .into_iter()
            .min_by(|a, b| {
                rank(a)
                    .partial_cmp(&rank(b))
                    .expect("finite calibration ranks")
            })
            .context("empty calibration bucket")?;
        ensure!(
            cache.winners[bucket] == best.plan,
            "cache winner is not the measured minimum"
        );
    }
    Ok(())
}

pub fn load(d: &Device, ops: &Ops, c: &Config, dir: &Path) -> Result<Calibrated> {
    load_kind(d, ops, c, dir, None)
}

fn load_kind(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    query_rows: Option<usize>,
) -> Result<Calibrated> {
    let key = key(d).context("calibration cache key")?;
    let name = query_rows.map_or_else(
        || "attention".to_owned(),
        |rows| format!("attention-mq{rows}"),
    );
    let path = dir.join(format!(
        "{name}-{}-{}-{}.json",
        key.device_uuid, key.driver_version, key.binary_version
    ));
    let (cache, hit) = if path.exists() {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("read calibration cache {}", path.display()))?;
        check(&cache, &key, query_rows)?;
        (cache, true)
    } else {
        let cache = measure(d, ops, c, key, query_rows).context("measure attention candidates")?;
        check(&cache, &cache.key, query_rows)?;
        let temp = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(&cache)?)?;
        file.sync_all()?;
        std::fs::rename(temp, &path)?;
        (cache, false)
    };
    Ok(Calibrated { cache, hit, path })
}

pub(super) fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let k = (v.len() - 1) as f64 * q;
    let lo = k.floor() as usize;
    let hi = k.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (k - lo as f64)
}

enum Candidate {
    Decode(Attention),
    Multi(Verification),
}

struct Buffers<'a> {
    q: &'a cudarc::driver::CudaSlice<bf16>,
    k: &'a cudarc::driver::CudaSlice<bf16>,
    v: &'a cudarc::driver::CudaSlice<bf16>,
    out: &'a mut cudarc::driver::CudaSlice<bf16>,
    length: &'a cudarc::driver::CudaSlice<i32>,
}

impl Candidate {
    fn run(&mut self, d: &Device, b: Buffers<'_>, capacity: usize, rows: usize) -> Result<()> {
        match self {
            Self::Decode(a) => a.run(&d.stream, b.q, b.k, b.v, b.out, b.length),
            Self::Multi(a) => a.run(
                b.q,
                b.k,
                b.v,
                b.out,
                b.length,
                Shape {
                    rows,
                    start: 0,
                    q_heads: 16,
                    kv_heads: 2,
                    capacity,
                    causal: true,
                },
            ),
        }
    }
}

fn measure(
    d: &Device,
    ops: &Ops,
    c: &Config,
    key: CacheKey,
    query_rows: Option<usize>,
) -> Result<Cache> {
    let start = Instant::now();
    let s = &d.stream;
    let rows = query_rows.unwrap_or(1);
    let packs: &[usize] = if query_rows.is_some() {
        &[1, 2, 4, 8]
    } else {
        &[2, 4]
    };
    let mut q = s.alloc_zeros::<bf16>(rows * c.qkv_dim())?;
    let random = |count: usize, mut seed: u32| {
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                bf16::from_f32((seed >> 8) as f32 / 16777216.0 - 0.5)
            })
            .collect::<Vec<_>>()
    };
    let zero_q = vec![bf16::ZERO; rows * c.qkv_dim()];
    let queries = random(rows * c.qkv_dim(), 4);
    let capacities = CAPACITIES.map(|n| n + query_rows.unwrap_or(0));
    let k = s.clone_htod(&random(capacities[2] * c.kv_dim(), 1))?;
    let v_values = random(capacities[2] * c.kv_dim(), 2);
    let v = s.clone_htod(&v_values)?;
    let mut out = s.alloc_zeros::<bf16>(rows * c.hidden_size)?;
    let mut length = s.alloc_zeros::<i32>(1)?;
    let flush_count = d.info.l2_bytes.max(1024 * 1024);
    let mut flush = s.alloc_zeros::<bf16>(flush_count)?;
    let zeros = s.alloc_zeros::<bf16>(flush_count)?;
    let flush_tile = random(512 * 1024, 3);
    for offset in (0..flush_count).step_by(flush_tile.len()) {
        let count = flush_tile.len().min(flush_count - offset);
        s.memcpy_htod(
            &flush_tile[..count],
            &mut flush.slice_mut(offset..offset + count),
        )?;
    }
    let pairs: Vec<_> = (0..8)
        .map(|_| {
            Ok((
                d.ctx
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
                d.ctx
                    .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
            ))
        })
        .collect::<Result<_>>()?;
    s.synchronize()?;
    let mut trials = Vec::new();
    let mut winners = [AttentionPlan {
        chunk: 128,
        qpack: 2,
        threads: 128,
        merge_threads: 256,
    }; 3];
    for (bucket, &l) in LENGTHS.iter().enumerate() {
        s.memcpy_htod(&[l as i32], &mut length)?;
        let mut expected = vec![0f64; rows * c.kv_dim()];
        for head in 0..c.num_key_value_heads {
            let mut sum = vec![0f64; c.head_dim];
            for row in 0..l {
                for (dim, value) in sum.iter_mut().enumerate() {
                    *value += v_values[(head * capacities[bucket] + row) * c.head_dim + dim]
                        .to_f32() as f64;
                }
            }
            for qi in 0..rows {
                let visible = if query_rows.is_some() { l + qi + 1 } else { l };
                for (dim, value) in sum.iter_mut().enumerate() {
                    if query_rows.is_some() {
                        *value += v_values[(head * capacities[bucket] + l + qi) * c.head_dim + dim]
                            .to_f32() as f64;
                    }
                    expected[(qi * c.num_key_value_heads + head) * c.head_dim + dim] =
                        *value / visible as f64;
                }
            }
        }
        let mut best = (f64::INFINITY, f64::INFINITY, usize::MAX);
        for chunk in [128, 256, 512, 1024] {
            for &qpack in packs {
                for threads in [128, 256] {
                    let plan = AttentionPlan {
                        chunk,
                        qpack,
                        threads,
                        merge_threads: 256,
                    };
                    let mut attention = match query_rows {
                        Some(rows) => Candidate::Multi(Verification::candidate(
                            &d.ctx,
                            s,
                            c,
                            capacities[bucket],
                            rows,
                            plan,
                        )?),
                        None => Candidate::Decode(Attention::candidate(
                            &d.ctx,
                            s,
                            c,
                            capacities[bucket],
                            plan,
                            false,
                        )?),
                    };
                    s.memcpy_htod(&zero_q, &mut q)?;
                    s.memcpy_htod(&[l as i32], &mut length)?;
                    attention
                        .run(
                            d,
                            Buffers {
                                q: &q,
                                k: &k,
                                v: &v,
                                out: &mut out,
                                length: &length,
                            },
                            capacities[bucket],
                            rows,
                        )
                        .context("warm attention candidate")?;
                    s.synchronize()?;
                    let actual = s.clone_dtoh(&out)?;
                    ensure!(
                        actual.iter().all(|v| v.is_finite()),
                        "nonfinite calibration output"
                    );
                    let max_error = actual
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let qi = i / c.hidden_size;
                            let head = (i % c.hidden_size)
                                / c.head_dim
                                / (c.num_attention_heads / c.num_key_value_heads);
                            (v.to_f32() as f64
                                - expected[(qi * c.num_key_value_heads + head) * c.head_dim
                                    + i % c.head_dim])
                                .abs()
                        })
                        .fold(0f64, f64::max);
                    ensure!(
                        max_error <= 0.0001,
                        "calibration uniform-attention reference mismatch: {max_error}"
                    );
                    s.memcpy_htod(&queries, &mut q)?;
                    s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    let launch = attention.run(
                        d,
                        Buffers {
                            q: &q,
                            k: &k,
                            v: &v,
                            out: &mut out,
                            length: &length,
                        },
                        capacities[bucket],
                        rows,
                    );
                    let graph = s.end_capture(sys::CUgraphInstantiate_flags(0));
                    launch?;
                    let graph = graph?.context("empty calibration graph")?;
                    graph.upload().context("upload candidate graph")?;
                    graph.launch().context("launch candidate graph")?;
                    s.synchronize()?;
                    let timer = Instant::now();
                    let mut samples = Vec::new();
                    let trial_ms = if query_rows.is_some() { 15 } else { 30 };
                    while timer.elapsed() < Duration::from_millis(trial_ms) {
                        for (index, (begin, end)) in pairs.iter().enumerate() {
                            let growth = [0, 1, 128, 256][index % 4];
                            s.memcpy_htod(&[(l + growth) as i32], &mut length)?;
                            unsafe {
                                s.launch_builder(&ops.residual)
                                    .arg(&mut flush)
                                    .arg(&zeros)
                                    .arg(&(flush_count as i32))
                                    .launch(grid(flush_count.div_ceil(2048), 256))?;
                            }
                            begin.record(s).context("record calibration begin event")?;
                            graph.launch().context("launch candidate graph")?;
                            end.record(s).context("record calibration end event")?;
                        }
                        s.synchronize()?;
                        for (begin, end) in &pairs {
                            samples.push(
                                begin
                                    .elapsed_ms(end)
                                    .context("calibration elapsed event time")?
                                    as f64
                                    * 1000.0,
                            );
                        }
                    }
                    let wall_ms = timer.elapsed().as_secs_f64() * 1000.0;
                    let median_us = percentile(&samples, 0.5);
                    ensure!(
                        s.clone_dtoh(&out)?.iter().all(|v| v.is_finite()),
                        "nonfinite calibration output"
                    );
                    let p90_us = percentile(&samples, 0.9);
                    let score = (median_us, p90_us, usize::MAX - chunk);
                    if score < best {
                        best = score;
                        winners[bucket] = plan;
                    }
                    trials.push(Trial {
                        length: l,
                        plan,
                        samples: samples.len(),
                        median_us,
                        p10_us: percentile(&samples, 0.1),
                        p90_us: percentile(&samples, 0.9),
                        wall_ms,
                        max_abs_error: max_error,
                        evaluation_lengths: [l, l + 1, l + 128, l + 256],
                    });
                }
            }
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        wall_ms <= 2000.0,
        "attention calibration exceeded 2 s: {wall_ms:.3} ms"
    );
    Ok(Cache {
        query_rows,
        schema: 1,
        key,
        winners,
        trials,
        wall_ms,
    })
}
