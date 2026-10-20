use super::{
    Device,
    decode::Attention,
    ops::{Ops, grid},
};
use crate::{
    backend::setup::{
        ATTENTION_LENGTHS, AttentionPlan, CacheKey, Selection, attention_bucket, attention_chunks,
    },
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

const LENGTHS: [usize; 5] = ATTENTION_LENGTHS;
const CAPACITIES: [usize; 5] = [1280, 8448, 33024, 65792, 131008];

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
struct Occupancy {
    plan: AttentionPlan,
    blocks: usize,
    active_per_sm: usize,
}

impl Occupancy {
    fn fits(&self, sms: usize) -> bool {
        self.blocks <= sms * self.active_per_sm
    }
}

fn plans(bucket: usize) -> Vec<AttentionPlan> {
    let mut plans = Vec::new();
    for &chunk in attention_chunks(bucket) {
        for qpack in [2, 4] {
            for threads in [128, 256] {
                for merge_threads in [256, 512] {
                    plans.push(AttentionPlan {
                        chunk,
                        qpack,
                        threads,
                        merge_threads,
                    });
                }
            }
        }
    }
    plans
}

#[derive(Serialize, Deserialize)]
struct Cache {
    schema: u32,
    key: CacheKey,
    winner: AttentionPlan,
    bucket: usize,
    lengths: [usize; 5],
    trials: Vec<Trial>,
    sms: usize,
    phase1: Vec<Occupancy>,
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
        let bucket = attention_bucket(capacity)?;
        ensure!(
            self.cache.bucket == bucket,
            "decode calibration bucket does not match capacity"
        );
        eprintln!(
            "{}",
            serde_json::json!({"attention_calibration_selection":{"capacity":capacity,"bucket":bucket,"measured_length":LENGTHS[bucket],"plan":self.cache.winner}})
        );
        Ok((
            self.cache.winner,
            Selection::Calibrated {
                key: self.cache.key.clone(),
                cached: self.hit,
            },
        ))
    }
}

pub fn runtime_dir(path: &Path, model: &Path, create: bool) -> Result<PathBuf> {
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
    if create {
        std::fs::create_dir_all(&resolved)?;
    }
    ensure!(
        !resolved.exists() || resolved.is_dir(),
        "runtime path is not a directory"
    );
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

fn check(cache: &Cache, key: &CacheKey, bucket: usize) -> Result<()> {
    let expected = plans(bucket);
    let filtered = bucket >= 3;
    ensure!(
        cache.schema == 2
            && cache.lengths == LENGTHS
            && cache.key == *key
            && bucket < LENGTHS.len()
            && cache.bucket == bucket
            && cache.sms > 0,
        "invalid attention calibration cache identity"
    );
    ensure!(
        cache.phase1.len() == if filtered { expected.len() } else { 0 },
        "invalid phase1 occupancy lattice"
    );
    let length = &LENGTHS[bucket];
    let trials: Vec<_> = cache
        .trials
        .iter()
        .filter(|t| t.length == *length)
        .collect();
    ensure!(
        trials.len() == cache.trials.len(),
        "cache contains another length bucket"
    );
    let mut timed = 0;
    for plan in &expected {
        let eligible = if filtered {
            let records: Vec<_> = cache.phase1.iter().filter(|o| o.plan == *plan).collect();
            ensure!(records.len() == 1, "missing or duplicate phase1 occupancy");
            let o = records[0];
            ensure!(
                o.blocks == length.div_ceil(plan.chunk) * (16 / plan.qpack) && o.active_per_sm > 0,
                "invalid phase1 occupancy record"
            );
            o.fits(cache.sms)
        } else {
            true
        };
        ensure!(
            trials.iter().filter(|t| t.plan == *plan).count() == usize::from(eligible),
            "attention candidate timing/filter lattice mismatch"
        );
        timed += usize::from(eligible);
    }
    ensure!(
        cache.trials.len() == timed,
        "unknown attention calibration candidate"
    );
    ensure!(
        trials.iter().all(|t| t.samples > 0
            && t.median_us.is_finite()
            && t.median_us > 0.0
            && t.evaluation_lengths == [*length, *length + 1, *length + 128, *length + 256]),
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
        cache.winner == best.plan,
        "cache winner is not the measured minimum"
    );
    Ok(())
}

pub(super) fn cache_path(dir: &Path, key: &CacheKey, bucket: usize) -> PathBuf {
    dir.join(format!(
        "attention-long-v2-b{bucket}-{}-{}-{}.json",
        key.device_uuid, key.driver_version, key.binary_version
    ))
}

pub(super) fn load(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    capacity: usize,
    closure: &super::closure::Closure,
) -> Result<Calibrated> {
    let bucket = attention_bucket(capacity)?;
    let key = key(d).context("calibration cache key")?;
    let path = cache_path(dir, &key, bucket);
    let (cache, hit) = if closure.access(&path)? {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("read calibration cache {}", path.display()))?;
        ensure!(cache.sms == d.info.sms, "attention cache SM count mismatch");
        check(&cache, &key, bucket)?;
        (cache, true)
    } else {
        eprintln!(
            "{}",
            serde_json::json!({"attention_calibration_cache":{"schema":2,"lengths":LENGTHS,"bucket":bucket,"path":path,"cached":false,"action":"measure requested bucket"}})
        );
        let cache =
            measure(d, ops, c, key, bucket, closure).context("measure attention candidates")?;
        check(&cache, &cache.key, bucket)?;
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

/// Uniform-attention max |actual − expected| and max error / bound over elements.
/// Bound: a + half_ulp_bf16(|expected| + a), a = γ_{chunk + ⌈keys/chunk⌉}(2^-23)·max|V| for the
/// split-KV sums of at most `chunk` and ⌈keys/chunk⌉ FP32 terms, the reciprocal and the product.
pub(super) fn uniform_attention_error(
    c: &Config,
    actual: &[bf16],
    expected: &[f64],
    chunk: usize,
    keys: usize,
    v_max: f64,
) -> (f64, f64) {
    let accumulation = super::gamma(chunk + keys.div_ceil(chunk), 2f64.powi(-23)) * v_max;
    actual
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let row = i / c.hidden_size;
            let head =
                (i % c.hidden_size) / c.head_dim / (c.num_attention_heads / c.num_key_value_heads);
            let e = expected[(row * c.num_key_value_heads + head) * c.head_dim + i % c.head_dim];
            let error = (v.to_f32() as f64 - e).abs();
            (
                error,
                error / (accumulation + crate::verify::bf16_half_ulp(e.abs() + accumulation)),
            )
        })
        .fold((0f64, 0f64), |(a, r), (e, q)| (a.max(e), r.max(q)))
}

fn measure(
    d: &Device,
    ops: &Ops,
    c: &Config,
    key: CacheKey,
    selected: usize,
    closure: &super::closure::Closure,
) -> Result<Cache> {
    closure.measurement()?;
    let start = Instant::now();
    let s = &d.stream;
    let trial_ms = 30;
    let mut q = s.alloc_zeros::<bf16>(c.qkv_dim())?;
    let random = |count: usize, mut seed: u32| {
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                bf16::from_f32((seed >> 8) as f32 / 16777216.0 - 0.5)
            })
            .collect::<Vec<_>>()
    };
    let zero_q = vec![bf16::ZERO; c.qkv_dim()];
    let queries = random(c.qkv_dim(), 4);
    let k = s.clone_htod(&random(CAPACITIES[selected] * c.kv_dim(), 1))?;
    let v_values = random(CAPACITIES[selected] * c.kv_dim(), 2);
    let v_max = v_values
        .iter()
        .map(|v| v.to_f32().abs() as f64)
        .fold(0f64, f64::max);
    let v = s.clone_htod(&v_values)?;
    let mut out = s.alloc_zeros::<bf16>(c.hidden_size)?;
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
    let mut phase1 = Vec::new();
    let mut winner = None;
    for (bucket, &l) in LENGTHS.iter().enumerate().filter(|(b, _)| *b == selected) {
        s.memcpy_htod(&[l as i32], &mut length)?;
        let mut expected = vec![0f64; c.kv_dim()];
        for head in 0..c.num_key_value_heads {
            let mut sum = vec![0f64; c.head_dim];
            for row in 0..l {
                for (dim, value) in sum.iter_mut().enumerate() {
                    *value += v_values[(head * CAPACITIES[bucket] + row) * c.head_dim + dim]
                        .to_f32() as f64;
                }
            }
            for (dim, value) in sum.iter().enumerate() {
                expected[head * c.head_dim + dim] = *value / l as f64;
            }
        }
        let mut best = (f64::INFINITY, f64::INFINITY, usize::MAX);
        for plan in plans(bucket) {
            let chunk = plan.chunk;
            let mut attention =
                Attention::candidate(&d.ctx, s, c, CAPACITIES[bucket], plan, false)?;
            if bucket >= 3 {
                let record = Occupancy {
                    plan,
                    blocks: l.div_ceil(chunk) * (16 / plan.qpack),
                    active_per_sm: attention
                        .phase1_blocks_per_sm()
                        .context("query decode phase1 occupancy")?,
                };
                ensure!(record.active_per_sm > 0, "decode phase1 has zero occupancy");
                let fits = record.fits(d.info.sms);
                eprintln!(
                    "{}",
                    serde_json::json!({"phase1_calibration_filter":{"length":l,"plan":plan,"sms":d.info.sms,"blocks":record.blocks,"active_per_sm":record.active_per_sm,"timed":fits}})
                );
                phase1.push(record);
                if !fits {
                    continue;
                }
            }
            s.memcpy_htod(&zero_q, &mut q)?;
            s.memcpy_htod(&[l as i32], &mut length)?;
            attention
                .run(s, &q, &k, &v, &mut out, &length)
                .context("warm attention candidate")?;
            s.synchronize()?;
            let actual = s.clone_dtoh(&out)?;
            ensure!(
                actual.iter().all(|v| v.is_finite()),
                "nonfinite calibration output"
            );
            let (max_error, ratio) =
                uniform_attention_error(c, &actual, &expected, chunk, l, v_max);
            ensure!(
                ratio <= 1.0,
                "calibration uniform-attention reference mismatch: max_error={max_error} error/bound={ratio}"
            );
            s.memcpy_htod(&queries, &mut q)?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launch = attention.run(s, &q, &k, &v, &mut out, &length);
            let graph = s.end_capture(sys::CUgraphInstantiate_flags(0));
            launch?;
            let graph = graph?.context("empty calibration graph")?;
            graph.upload().context("upload candidate graph")?;
            graph.launch().context("launch candidate graph")?;
            s.synchronize()?;
            let timer = Instant::now();
            let mut samples = Vec::new();
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
                winner = Some(plan);
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
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Cache {
        schema: 2,
        lengths: LENGTHS,
        key,
        winner: winner.context("no measured decode attention candidate")?,
        bucket: selected,
        trials,
        sms: d.info.sms,
        phase1,
        wall_ms,
    })
}

pub(super) fn host_memory(root: &Path) -> Result<(u64, u64)> {
    let meminfo = std::fs::read_to_string(root.join("proc/meminfo"))?;
    let mut available = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|line| line.split_whitespace().next())
        .context("expert host profile has no MemAvailable")?
        .parse::<u64>()?
        .checked_mul(1024)
        .context("expert available host bytes overflow")?;
    let groups = std::fs::read_to_string(root.join("proc/self/cgroup"))?;
    let memory = groups.lines().find_map(|line| {
        let (_, rest) = line.split_once(':')?;
        let (controllers, group) = rest.split_once(':')?;
        controllers
            .split(',')
            .any(|controller| controller == "memory")
            .then_some(group)
    });
    let (mount, mut directory, maximum_name, current_name) = if let Some(group) = memory {
        let mounts = std::fs::read_to_string(root.join("proc/self/mountinfo"))?;
        let decode = |field: &str| {
            field
                .replace("\\040", " ")
                .replace("\\011", "\t")
                .replace("\\012", "\n")
                .replace("\\134", "\\")
        };
        let (_, point, relative) = mounts
            .lines()
            .filter_map(|line| {
                let (paths, options) = line.split_once(" - ")?;
                let mut options = options.split_whitespace();
                if options.next()? != "cgroup" {
                    return None;
                }
                options.next()?;
                if !options.next()?.split(',').any(|name| name == "memory") {
                    return None;
                }
                let mut paths = paths.split_whitespace().skip(3);
                let base = decode(paths.next()?);
                let point = decode(paths.next()?);
                let relative = Path::new(group).strip_prefix(&base).ok()?.to_path_buf();
                Some((Path::new(&base).components().count(), point, relative))
            })
            .min_by_key(|(depth, _, _)| *depth)
            .with_context(|| format!("expert v1 memory controller mount missing for {group}"))?;
        let mount = root.join(point.trim_start_matches('/'));
        let directory = mount.join(relative);
        (
            mount,
            directory,
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
        )
    } else {
        let group = groups
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .context("expert memory controller missing from v1 and v2 cgroup memberships")?;
        let mount = root.join("sys/fs/cgroup");
        let directory = mount.join(group.trim_start_matches('/'));
        (mount, directory, "memory.max", "memory.current")
    };
    loop {
        let maximum = directory.join(maximum_name);
        if directory == mount && !maximum.try_exists()? {
            break;
        }
        let limit = std::fs::read_to_string(&maximum)
            .with_context(|| format!("read expert cgroup limit {}", maximum.display()))?;
        if memory.is_some() || limit.trim() != "max" {
            let current = directory.join(current_name);
            let used = std::fs::read_to_string(&current)
                .with_context(|| format!("read expert cgroup usage {}", current.display()))?
                .trim()
                .parse::<u64>()?;
            let statistics = directory.join("memory.stat");
            let statistics = std::fs::read_to_string(&statistics).with_context(|| {
                format!("read expert cgroup statistics {}", statistics.display())
            })?;
            let counter = |name: &str| -> Result<u64> {
                statistics
                    .lines()
                    .find_map(|line| {
                        let mut fields = line.split_whitespace();
                        (fields.next()? == name).then(|| fields.next()).flatten()
                    })
                    .with_context(|| {
                        format!(
                            "expert cgroup statistics missing {name} in {}",
                            directory.display()
                        )
                    })?
                    .parse::<u64>()
                    .with_context(|| format!("invalid expert cgroup counter {name}"))
            };
            let reclaimable = if memory.is_some() {
                counter("total_active_file")?
                    .checked_add(counter("total_inactive_file")?)
                    .context("expert cgroup file LRU bytes overflow")?
            } else {
                counter("file")?.saturating_sub(counter("shmem")?)
            };
            let used = used.saturating_sub(reclaimable);
            available = available.min(limit.trim().parse::<u64>()?.saturating_sub(used));
        }
        if directory == mount {
            break;
        }
        ensure!(
            directory.pop(),
            "expert cgroup has no parent under {}",
            mount.display()
        );
    }
    let limits = std::fs::read_to_string(root.join("proc/self/limits"))?;
    let locked = limits
        .lines()
        .find_map(|line| line.strip_prefix("Max locked memory"))
        .and_then(|line| line.split_whitespace().next())
        .context("expert host profile has no locked-memory limit")?;
    let locked = if locked == "unlimited" {
        available
    } else {
        locked.parse::<u64>()?.min(available)
    };
    Ok((available, locked))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ExpertHost {
    cpu: String,
    isa: String,
    affinity: String,
    threads: u32,
    llc_bytes: u64,
}

impl ExpertHost {
    fn read() -> Result<Self> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").context("read host CPU identity")?;
        let cpu = cpuinfo
            .split("\n\n")
            .next()
            .context("empty host CPU identity")?
            .lines()
            .filter(|line| {
                line.split_once(':').is_some_and(|(name, _)| {
                    [
                        "vendor_id",
                        "cpu family",
                        "model",
                        "stepping",
                        "model name",
                        "flags",
                    ]
                    .contains(&name.trim())
                })
            })
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(!cpu.is_empty(), "host CPU identity has no hardware fields");
        let status = std::fs::read_to_string("/proc/self/status").context("read CPU affinity")?;
        let affinity = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .context("missing Cpus_allowed_list")?
            .trim()
            .to_owned();
        let mut caches = std::collections::BTreeMap::new();
        for range in affinity.split(',') {
            let (first, last) = range.split_once('-').unwrap_or((range, range));
            let first: u32 = first.parse().context("invalid CPU affinity start")?;
            let last: u32 = last.parse().context("invalid CPU affinity end")?;
            ensure!(first <= last, "reversed CPU affinity range {range}");
            for cpu in first..=last {
                let directory = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cache"));
                for entry in std::fs::read_dir(&directory)
                    .with_context(|| format!("read cache topology {}", directory.display()))?
                {
                    let entry = entry?.path();
                    if !entry.is_dir() {
                        continue;
                    }
                    let read = |field| {
                        std::fs::read_to_string(entry.join(field)).with_context(|| {
                            format!("read host cache {field} at {}", entry.display())
                        })
                    };
                    let kind = read("type")?;
                    if !matches!(kind.trim(), "Data" | "Unified") {
                        continue;
                    }
                    let level: u32 = read("level")?.trim().parse().with_context(|| {
                        format!("parse host cache level at {}", entry.display())
                    })?;
                    let shared = read("shared_cpu_list")?;
                    let size = read("size")?;
                    let size = size.trim();
                    let digits = size.chars().take_while(char::is_ascii_digit).count();
                    let number: u64 = size[..digits].parse().context("invalid host cache size")?;
                    let multiplier = match &size[digits..] {
                        "" => 1,
                        "K" => 1 << 10,
                        "M" => 1 << 20,
                        "G" => 1 << 30,
                        unit => anyhow::bail!("invalid host cache size unit {unit}"),
                    };
                    let bytes = number
                        .checked_mul(multiplier)
                        .context("host cache size overflow")?;
                    caches.insert((level, shared), bytes);
                }
            }
        }
        let level = caches
            .keys()
            .map(|(level, _)| *level)
            .max()
            .context("no host data cache")?;
        let llc_bytes = caches
            .iter()
            .filter(|((l, _), _)| *l == level)
            .try_fold(0u64, |sum, (_, &bytes)| sum.checked_add(bytes))
            .context("host LLC extent overflow")?;
        ensure!(llc_bytes > 0, "zero host LLC bytes");
        Ok(Self {
            cpu,
            isa: crate::cpu::expert_ffn::Isa::detect().name().into(),
            affinity,
            threads: u32::try_from(std::thread::available_parallelism()?.get())?,
            llc_bytes,
        })
    }

    fn workers(&self) -> Vec<u32> {
        let limit = self.threads.min(256);
        (0..=8)
            .map(|shift| 1 << shift)
            .chain([limit - limit / 4, limit])
            .filter(|&n| n > 0 && n <= limit)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn ring_experts(
        &self,
        geometry: &crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
    ) -> Result<u32> {
        let bytes = self
            .llc_bytes
            .checked_mul(8)
            .context("CPU service ring extent overflow")?;
        u32::try_from(
            bytes
                .div_ceil(slot_bytes)
                .max(u64::from(geometry.top_k) * 6),
        )
        .context("CPU service ring exceeds u32 experts")
    }
}

#[derive(Serialize, Deserialize)]
struct ExpertTransfer {
    bytes: u64,
    h2d_seconds: [f64; 5],
    d2h_seconds: [f64; 5],
}

const EXPERT_SERVICE_ROWS: [u32; 7] = [1, 2, 5, 8, 16, 32, 64];

#[derive(Serialize, Deserialize)]
struct ExpertCache {
    schema: u32,
    expert_format: crate::model_spec::ExpertFormat,
    slot_bytes: u64,
    key: CacheKey,
    host: ExpertHost,
    geometry: crate::experts::catalogue::ExpertGeometry,
    copy_engines: u32,
    ring_experts: u32,
    transfers: Vec<ExpertTransfer>,
    service: Vec<crate::experts::plan::ServiceCell>,
}

pub(super) struct ExpertCalibration {
    pub h2d_bytes_per_second: f64,
    pub d2h_bytes_per_second: f64,
    pub copy_engines: u32,
    pub service: crate::experts::plan::ServiceTable,
}

impl ExpertCache {
    fn check(
        &self,
        key: &CacheKey,
        host: &ExpertHost,
        geometry: &crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
    ) -> Result<ExpertCalibration> {
        ensure!(
            self.schema == 2,
            "expert calibration schema {} expected 2",
            self.schema
        );
        ensure!(
            self.slot_bytes == slot_bytes
                && crate::experts::catalogue::ExpertLayout::new(self.expert_format, geometry)?
                    .slot_bytes()
                    == slot_bytes,
            "expert calibration format={:?} slot_bytes={} expected={slot_bytes}",
            self.expert_format,
            self.slot_bytes
        );
        ensure!(
            self.key == *key,
            "expert calibration key {:?} expected {key:?}",
            self.key
        );
        ensure!(
            self.host == *host,
            "expert calibration host {:?} expected {host:?}",
            self.host
        );
        ensure!(
            self.geometry == *geometry,
            "expert calibration geometry {:?} expected {geometry:?}",
            self.geometry
        );
        ensure!(
            self.copy_engines > 0 && self.copy_engines <= 16,
            "invalid expert async engine count {}",
            self.copy_engines
        );
        ensure!(
            self.ring_experts == host.ring_experts(geometry, slot_bytes)?,
            "expert ring count {} differs from host-derived count",
            self.ring_experts
        );
        let sizes = [1, geometry.top_k]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|n| {
                u64::from(n)
                    .checked_mul(slot_bytes)
                    .context("expert transfer extent overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            self.transfers.len() == sizes.len(),
            "expert transfer lattice size {} expected {}",
            self.transfers.len(),
            sizes.len()
        );
        for (cell, bytes) in self.transfers.iter().zip(sizes) {
            ensure!(
                cell.bytes == bytes
                    && cell
                        .h2d_seconds
                        .iter()
                        .chain(&cell.d2h_seconds)
                        .all(|&n| n.is_finite() && n > 0.0),
                "invalid expert transfer cell bytes={} expected={bytes}",
                cell.bytes
            );
        }
        let missed: Vec<_> = [1, 2, 4, geometry.top_k]
            .into_iter()
            .filter(|&n| n <= geometry.top_k)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let workers = host.workers();
        let expected = workers.len() * EXPERT_SERVICE_ROWS.len() * missed.len();
        ensure!(
            self.service.len() == expected,
            "expert CPU service cells {} expected {expected}",
            self.service.len()
        );
        for &workers in &workers {
            for rows in EXPERT_SERVICE_ROWS {
                for &experts in &missed {
                    ensure!(
                        self.service
                            .iter()
                            .filter(|c| c.workers == workers
                                && c.rows == rows
                                && c.experts == experts)
                            .count()
                            == 1,
                        "missing or duplicate expert CPU cell workers={workers} rows={rows} experts={experts}"
                    );
                }
            }
        }
        let service = crate::experts::plan::ServiceTable::new(&host.isa, self.service.clone())?;
        let slot = self
            .transfers
            .first()
            .context("missing native-slot transfer cell")?;
        let h2d_bytes_per_second = slot.bytes as f64 / percentile(&slot.h2d_seconds, 0.5);
        let d2h_bytes_per_second = slot.bytes as f64 / percentile(&slot.d2h_seconds, 0.5);
        ensure!(
            h2d_bytes_per_second.is_finite() && d2h_bytes_per_second.is_finite(),
            "non-finite expert transfer rate h2d={h2d_bytes_per_second} d2h={d2h_bytes_per_second}"
        );
        Ok(ExpertCalibration {
            h2d_bytes_per_second,
            d2h_bytes_per_second,
            copy_engines: self.copy_engines,
            service,
        })
    }
}

fn check_expert_transfer(bytes: &[u8]) -> Result<()> {
    if let Some((index, &actual)) = bytes
        .iter()
        .enumerate()
        .find(|(index, actual)| **actual != (index % 251) as u8)
    {
        anyhow::bail!(
            "expert transfer byte {index}: actual={actual} expected={} bytes={}",
            index % 251,
            bytes.len()
        );
    }
    Ok(())
}

fn measure_expert_transfer(d: &Device, bytes: usize) -> Result<ExpertTransfer> {
    let mut host = unsafe { d.ctx.alloc_pinned_with_flags::<u8>(bytes, 0) }
        .with_context(|| format!("allocate expert transfer pinned bytes={bytes}"))?;
    unsafe {
        std::ptr::write_bytes(host.as_mut_ptr()?, 0, bytes);
    }
    for (index, value) in host.as_mut_slice()?.iter_mut().enumerate() {
        *value = (index % 251) as u8;
    }
    let mut device = d
        .upload
        .alloc_zeros::<u8>(bytes)
        .with_context(|| format!("allocate expert transfer device bytes={bytes}"))?;
    d.upload
        .memcpy_htod(&host, &mut device)
        .with_context(|| format!("expert transfer roundtrip H2D bytes={bytes}"))?;
    d.upload
        .memcpy_dtoh(&device, &mut host)
        .with_context(|| format!("expert transfer roundtrip D2H bytes={bytes}"))?;
    d.upload
        .synchronize()
        .with_context(|| format!("synchronize expert transfer roundtrip bytes={bytes}"))?;
    check_expert_transfer(host.as_slice()?)?;
    let mut cell = ExpertTransfer {
        bytes: bytes as u64,
        h2d_seconds: [0.0; 5],
        d2h_seconds: [0.0; 5],
    };
    for download in [false, true] {
        for trial in 0..6 {
            let begin = d
                .upload
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| {
                    format!(
                        "record expert transfer begin bytes={bytes} d2h={download} trial={trial}"
                    )
                })?;
            for _ in 0..16 {
                if download {
                    d.upload.memcpy_dtoh(&device, &mut host).with_context(|| {
                        format!("expert transfer timed D2H bytes={bytes} trial={trial}")
                    })?;
                } else {
                    d.upload.memcpy_htod(&host, &mut device).with_context(|| {
                        format!("expert transfer timed H2D bytes={bytes} trial={trial}")
                    })?;
                }
            }
            let end = d
                .upload
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| {
                    format!("record expert transfer end bytes={bytes} d2h={download} trial={trial}")
                })?;
            end.synchronize().with_context(|| {
                format!(
                    "synchronize expert transfer end bytes={bytes} d2h={download} trial={trial}"
                )
            })?;
            let seconds = f64::from(begin.elapsed_ms(&end).with_context(|| {
                format!("time expert transfer events bytes={bytes} d2h={download} trial={trial}")
            })?) / 16000.0;
            ensure!(
                seconds.is_finite() && seconds > 0.0,
                "invalid expert transfer sample seconds={seconds} bytes={bytes} d2h={download}"
            );
            if trial > 0 {
                if download {
                    cell.d2h_seconds[trial - 1] = seconds;
                } else {
                    cell.h2d_seconds[trial - 1] = seconds;
                }
            }
        }
    }
    Ok(cell)
}

pub(super) fn load_experts(
    d: &Device,
    dir: &Path,
    geometry: crate::experts::catalogue::ExpertGeometry,
    expert_format: crate::model_spec::ExpertFormat,
    closure: &super::closure::Closure,
) -> Result<ExpertCalibration> {
    let slot_bytes =
        crate::experts::catalogue::ExpertLayout::new(expert_format, &geometry)?.slot_bytes();
    let key = key(d)?;
    let host = ExpertHost::read()?;
    let identity = super::closure::hash(&serde_json::to_vec(&(&host, geometry, expert_format))?);
    let path = dir.join(format!(
        "experts-v2-{}-{}-{}-{identity}.json",
        key.device_uuid, key.driver_version, key.binary_version
    ));
    let (cache, hit) = if closure.access(&path)? {
        (
            serde_json::from_slice::<ExpertCache>(&std::fs::read(&path)?)
                .with_context(|| format!("read expert calibration {}", path.display()))?,
            true,
        )
    } else {
        closure.measurement()?;
        let ring_experts = host.ring_experts(&geometry, slot_bytes)?;
        let ring_bytes = u64::from(ring_experts)
            .checked_mul(slot_bytes)
            .context("expert ring bytes overflow")?;
        let (available, _) = host_memory(Path::new("/"))?;
        ensure!(
            ring_bytes <= available / 2,
            "expert service ring {ring_bytes} B exceeds half of effective available host memory {available} B"
        );
        let copy_engines = u32::try_from(
            d.ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
        )?;
        let transfers = [1, geometry.top_k]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|n| {
                measure_expert_transfer(
                    d,
                    usize::try_from(
                        u64::from(n)
                            .checked_mul(slot_bytes)
                            .context("expert transfer extent overflow")?,
                    )?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let missed: Vec<_> = [1, 2, 4, geometry.top_k]
            .into_iter()
            .filter(|&n| n <= geometry.top_k)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let service = crate::cpu::expert_ffn::measure_service_table(
            crate::cpu::expert_ffn::Shape::new(
                geometry.hidden as usize,
                geometry.intermediate as usize,
            )?,
            ring_experts,
            &host.workers(),
            &EXPERT_SERVICE_ROWS,
            &missed,
            expert_format,
        )?;
        let cache = ExpertCache {
            schema: 2,
            expert_format,
            slot_bytes,
            key: key.clone(),
            host: host.clone(),
            geometry,
            copy_engines,
            ring_experts,
            transfers,
            service,
        };
        cache.check(&key, &host, &geometry, slot_bytes)?;
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
    let measured_engines = u32::try_from(
        d.ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
    )?;
    ensure!(
        cache.copy_engines == measured_engines,
        "expert cache engine count {} differs from driver {measured_engines}",
        cache.copy_engines
    );
    let measured = cache.check(&key, &host, &geometry, slot_bytes)?;
    eprintln!(
        "{}",
        serde_json::json!({"expert_calibration":{"path":path,"cached":hit,"cells":cache,"h2d_bytes_per_second":measured.h2d_bytes_per_second,"d2h_bytes_per_second":measured.d2h_bytes_per_second}})
    );
    Ok(measured)
}

#[cfg(test)]
mod long_tests {
    use super::*;
    #[test]
    fn expert_cells_guard_capacity_inputs() -> Result<()> {
        use crate::experts::{catalogue::ExpertGeometry, plan::ServiceCell};
        let mut bytes: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
        check_expert_transfer(&bytes)?;
        bytes[2048] ^= 1;
        assert!(check_expert_transfer(&bytes).is_err());
        let key = CacheKey {
            device_uuid: "fixture".into(),
            driver_version: "fixture".into(),
            binary_version: "fixture".into(),
        };
        let host = ExpertHost {
            cpu: "fixture".into(),
            isa: "fixture".into(),
            affinity: "0-1".into(),
            threads: 2,
            llc_bytes: 128,
        };
        let geometry = ExpertGeometry {
            layers: 2,
            experts: 8,
            hidden: 16,
            intermediate: 8,
            top_k: 2,
        };
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?
        .slot_bytes();
        let mut cache = ExpertCache {
            schema: 2,
            expert_format: crate::model_spec::ExpertFormat::Bf16,
            slot_bytes,
            key: key.clone(),
            host: host.clone(),
            geometry,
            copy_engines: 2,
            ring_experts: host.ring_experts(&geometry, slot_bytes)?,
            transfers: [768, 1536]
                .map(|bytes| ExpertTransfer {
                    bytes,
                    h2d_seconds: [0.001; 5],
                    d2h_seconds: [0.002; 5],
                })
                .into(),
            service: [1, 2]
                .into_iter()
                .flat_map(|workers| {
                    EXPERT_SERVICE_ROWS.into_iter().flat_map(move |rows| {
                        [1, 2].into_iter().map(move |experts| ServiceCell {
                            workers,
                            rows,
                            experts,
                            seconds: 0.001,
                        })
                    })
                })
                .collect(),
        };
        let measured = cache.check(&key, &host, &geometry, slot_bytes)?;
        cache.expert_format = crate::model_spec::ExpertFormat::Fp8W8a16;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.expert_format = crate::model_spec::ExpertFormat::Bf16;
        cache.slot_bytes += 16;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.slot_bytes = slot_bytes;
        assert_eq!(measured.h2d_bytes_per_second, 768000.0);
        assert_eq!(measured.d2h_bytes_per_second, 384000.0);
        assert_eq!(measured.copy_engines, 2);
        assert_eq!(measured.service.seconds(1, 1, 1)?, 0.001);
        assert_eq!(measured.service.seconds(64, 1, 1)?, 0.001);
        let large = cache.service.iter().position(|c| c.rows == 64).unwrap();
        let cell = cache.service.remove(large);
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.service.insert(large, cell);
        cache.transfers[0].d2h_seconds[0] = f64::NAN;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.transfers[0].d2h_seconds[0] = 0.0;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.transfers[0].d2h_seconds[0] = 0.002;
        cache.service[0].experts = 2;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.service[0].experts = 1;
        cache.copy_engines = 0;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.copy_engines = 2;
        cache.host.affinity = "2-3".into();
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.host = host.clone();
        cache.geometry.hidden += 4;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.geometry = geometry;
        cache.key.binary_version = "changed".into();
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        cache.key = key.clone();
        cache.transfers[0].bytes -= 1;
        assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive CUDA calibration"]
    fn expert_setup_smoke_gpu() -> Result<()> {
        let device = Device::new(0)?;
        let untimed_begin = device
            .upload
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        let untimed_end = device
            .upload
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        let untimed_error = untimed_begin
            .elapsed_ms(&untimed_end)
            .expect_err("untimed events produced elapsed time");
        ensure!(
            untimed_error.0 == sys::cudaError_enum::CUDA_ERROR_INVALID_HANDLE,
            "untimed expert transfer events returned {untimed_error:?}, expected CUDA_ERROR_INVALID_HANDLE"
        );
        println!("PASS expert_setup_event_control untimed=CUDA_ERROR_INVALID_HANDLE");
        measure_expert_transfer(&device, 4096)?;
        let host = ExpertHost::read()?;
        let cells = crate::cpu::expert_ffn::measure_service_table(
            crate::cpu::expert_ffn::Shape::new(16, 8)?,
            12,
            &[1],
            &[1],
            &[1],
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        let table = crate::experts::plan::ServiceTable::new(&host.isa, cells)?;
        ensure!(
            table.seconds(1, 1, 1)? > 0.0,
            "invalid CPU smoke service time"
        );
        println!(
            "PASS expert_setup_smoke transfer_bytes=4096 directions=2 CPU_cells=1 host={}",
            serde_json::to_string(&host)?
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive CUDA calibration and an empty runtime directory"]
    fn expert_setup_measurements_gpu() -> Result<()> {
        let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let runtime = PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        );
        let runtime = runtime_dir(&runtime, &model, true)?;
        ensure!(
            std::fs::read_dir(&runtime)?.next().is_none(),
            "expert measurement gate runtime directory must be empty: {}",
            runtime.display()
        );
        let spec = crate::model_spec::ModelSpec::read(&model)?;
        let geometry = crate::experts::catalogue::ExpertGeometry::from_model(&spec)?;
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            spec.expert_format.context("MoE expert_format missing")?,
            &geometry,
        )?
        .slot_bytes();
        super::super::engine::Engine::calibrate(
            &model,
            0,
            32,
            None,
            crate::backend::Settings {
                tree_builder: crate::backend::TreeBuilder::Waves,
                spec_budget: crate::backend::SpecBudget::Chain,
                spec_graph: false,
                draft_model: None,
                prefix_cache: false,
                kv_format: crate::kv::KvFormat::Bf16,
                runtime_dir: Some(runtime.clone()),
            },
        )?;
        let entries = std::fs::read_dir(&runtime)?.collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            entries.len() == 1,
            "expert measurement gate wrote {} files",
            entries.len()
        );
        let path = entries[0].path();
        let bytes = std::fs::read(&path)?;
        let modified = std::fs::metadata(&path)?.modified()?;
        let device = Device::new(0)?;
        let measured: ExpertCache = serde_json::from_slice(&bytes)?;
        let measured =
            measured.check(&key(&device)?, &ExpertHost::read()?, &geometry, slot_bytes)?;
        let reader = super::super::closure::Closure::new(false, "calibrate".into());
        let reused = load_experts(
            &device,
            &runtime,
            geometry,
            spec.expert_format.context("MoE expert_format missing")?,
            &reader,
        )?;
        assert_eq!(measured.h2d_bytes_per_second, reused.h2d_bytes_per_second);
        assert_eq!(measured.d2h_bytes_per_second, reused.d2h_bytes_per_second);
        assert_eq!(measured.copy_engines, reused.copy_engines);
        assert_eq!(measured.service, reused.service);
        assert_eq!(std::fs::read(&path)?, bytes);
        assert_eq!(std::fs::metadata(&path)?.modified()?, modified);
        let mut invalid: ExpertCache = serde_json::from_slice(&bytes)?;
        invalid.transfers[0].d2h_seconds[0] = 0.0;
        std::fs::write(&path, serde_json::to_vec(&invalid)?)?;
        let rejected = load_experts(
            &device,
            &runtime,
            geometry,
            spec.expert_format.context("MoE expert_format missing")?,
            &reader,
        );
        std::fs::write(&path, &bytes)?;
        assert!(rejected.is_err());
        device.upload.synchronize()?;
        device.stream.synchronize()?;
        println!(
            "PASS expert_setup_measurements cold_cells=true cache_reuse=byte_identical invalid_d2h=rejected"
        );
        Ok(())
    }

    #[test]
    fn cache_requires_measured_bucket_and_complete_candidates() -> Result<()> {
        for (bucket, &length) in LENGTHS.iter().enumerate() {
            let key = CacheKey {
                device_uuid: "test".into(),
                driver_version: "test".into(),
                binary_version: "test".into(),
            };
            let mut trials = Vec::new();
            let mut phase1 = Vec::new();
            for plan in plans(bucket) {
                if bucket >= 3 {
                    let o = Occupancy {
                        plan,
                        blocks: length.div_ceil(plan.chunk) * (16 / plan.qpack),
                        active_per_sm: 8,
                    };
                    let fits = o.fits(128);
                    phase1.push(o);
                    if !fits {
                        continue;
                    }
                }
                let median_us = (trials.len() + 1) as f64;
                trials.push(Trial {
                    length,
                    plan,
                    samples: 1,
                    median_us,
                    p10_us: median_us,
                    p90_us: median_us,
                    wall_ms: 1.0,
                    max_abs_error: 0.0,
                    evaluation_lengths: [length, length + 1, length + 128, length + 256],
                });
            }
            let mut cache = Cache {
                schema: 2,
                key,
                winner: trials[0].plan,
                bucket,
                lengths: LENGTHS,
                trials,
                sms: 128,
                phase1,
                wall_ms: 1.0,
            };
            check(&cache, &cache.key, bucket)?;
            let winner = cache.winner;
            cache.winner = cache.trials[1].plan;
            assert!(check(&cache, &cache.key, bucket).is_err());
            cache.winner = winner;
            assert!(check(&cache, &cache.key, (bucket + 1) % LENGTHS.len()).is_err());
            if bucket >= 3 {
                cache.phase1[0].blocks += 1;
                assert!(check(&cache, &cache.key, bucket).is_err());
                cache.phase1[0].blocks -= 1;
            }
            cache.schema = 1;
            assert!(check(&cache, &cache.key, bucket).is_err());
            cache.schema = 2;
            cache.lengths[4] = 32768;
            assert!(check(&cache, &cache.key, bucket).is_err());
            cache.lengths = LENGTHS;
            cache.trials.pop();
            assert!(check(&cache, &cache.key, bucket).is_err());
        }
        Ok(())
    }
}
