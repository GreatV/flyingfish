use super::{
    Device,
    decode::Attention,
    ops::{Ops, grid},
};
use crate::{
    backend::setup::{
        ATTENTION_LENGTHS, AttentionPlan, CacheKey, Selection, attention_bucket, attention_chunks,
        attention_lengths,
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
    io::Read,
    path::{Component, Path, PathBuf},
    sync::OnceLock,
    time::{Duration, Instant},
};

const LENGTHS: [usize; 5] = ATTENTION_LENGTHS;

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
    capacity: usize,
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
            self.cache.bucket == bucket && self.cache.capacity == capacity,
            "decode calibration bucket does not match capacity"
        );
        eprintln!(
            "{}",
            serde_json::json!({"attention_calibration_selection":{"capacity":capacity,"bucket":bucket,"measured_length":self.cache.trials[0].length,"plan":self.cache.winner}})
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

fn driver_release(text: &str) -> Result<String> {
    text.lines()
        .find_map(|line| line.strip_prefix("NVRM version:"))
        .and_then(|line| {
            line.split_whitespace().find(|part| {
                let pieces: Vec<_> = part.split('.').collect();
                pieces.len() == 3
                    && pieces
                        .iter()
                        .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
            })
        })
        .map(str::to_owned)
        .context("NVRM version line has no NVIDIA driver release")
}

pub(super) fn key(ctx: &cudarc::driver::CudaContext) -> Result<CacheKey> {
    static DRIVER: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let version = DRIVER
        .get_or_init(|| {
            std::fs::read_to_string("/proc/driver/nvidia/version")
                .context("read NVIDIA kernel-module release")
                .and_then(|text| driver_release(&text))
                .map_err(|error| format!("{error:#}"))
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let bytes = ctx.uuid()?.bytes;
    let hex: Vec<String> = bytes.iter().map(|&v| format!("{:02x}", v as u8)).collect();
    let uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        hex[0..4].concat(),
        hex[4..6].concat(),
        hex[6..8].concat(),
        hex[8..10].concat(),
        hex[10..16].concat()
    );
    let mut file = File::open(std::env::current_exe()?)?;
    let mut hash = DefaultHasher::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.write(&buffer[..count]);
    }
    let cublaslt_version = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
    ensure!(
        cublaslt_version > 0,
        "cuBLASLt returned an invalid library version"
    );
    Ok(CacheKey {
        device_uuid: uuid,
        driver_version: version.clone(),
        binary_version: format!("hash64-{:016x}", hash.finish()),
        cublaslt_version,
    })
}

fn rank(t: &Trial) -> (f64, f64, usize) {
    (t.median_us, t.p90_us, usize::MAX - t.plan.chunk)
}

fn check(cache: &Cache, key: &CacheKey, capacity: usize) -> Result<()> {
    let bucket = attention_bucket(capacity)?;
    let evaluation_lengths = attention_lengths(capacity, 0)?;
    let expected = plans(bucket);
    let filtered = bucket >= 3;
    ensure!(
        cache.schema == 3
            && cache.capacity == capacity
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
    let length = &evaluation_lengths[0];
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
                o.blocks == capacity.div_ceil(plan.chunk) * (16 / plan.qpack)
                    && o.active_per_sm > 0,
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
            && t.evaluation_lengths == evaluation_lengths),
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

pub(super) fn cache_path(dir: &Path, key: &CacheKey, bucket: usize, capacity: usize) -> PathBuf {
    dir.join(format!(
        "attention-long-v3-b{bucket}-c{capacity}-{}.json",
        key.tag()
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
    let key = d.key.clone();
    let path = cache_path(dir, &key, bucket, capacity);
    let (cache, hit) = if closure.access(&path)? {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("read calibration cache {}", path.display()))?;
        ensure!(cache.sms == d.info.sms, "attention cache SM count mismatch");
        check(&cache, &key, capacity)?;
        (cache, true)
    } else {
        eprintln!(
            "{}",
            serde_json::json!({"attention_calibration_cache":{"schema":3,"lengths":LENGTHS,"bucket":bucket,"capacity":capacity,"path":path,"cached":false,"action":"measure requested bucket"}})
        );
        let cache =
            measure(d, ops, c, key, capacity, closure).context("measure attention candidates")?;
        check(&cache, &cache.key, capacity)?;
        super::closure::write_cache(&path, &cache)?;
        (cache, false)
    };
    Ok(Calibrated { cache, hit, path })
}

pub(super) fn load_recorded(d: &Device, dir: &Path, capacity: usize) -> Result<Calibrated> {
    let bucket = attention_bucket(capacity)?;
    let key = d.key.clone();
    let path = cache_path(dir, &key, bucket, capacity);
    let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("read calibration cache {}", path.display()))?;
    ensure!(cache.sms == d.info.sms, "attention cache SM count mismatch");
    check(&cache, &key, capacity)?;
    Ok(Calibrated {
        cache,
        hit: true,
        path,
    })
}

pub(super) use crate::verify::percentile;

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
            let width = c.num_attention_heads * c.head_dim;
            let row = i / width;
            let head = (i % width) / c.head_dim;
            let kv_head = head / (c.num_attention_heads / c.num_key_value_heads);
            let e = expected[row * c.kv_dim() + kv_head * c.head_dim + i % c.head_dim];
            let error = (v.to_f32() as f64 - e).abs();
            (
                error,
                error / (accumulation + crate::verify::bf16_half_ulp(e.abs() + accumulation)),
            )
        })
        .fold((0f64, 0f64), |(a, r), (e, q)| (a.max(e), r.max(q)))
}

pub(super) fn random(n: usize, mut state: u32) -> Vec<bf16> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bf16::from_f32((state >> 8) as f32 / 16777216.0 - 0.5)
        })
        .collect()
}

fn measure(
    d: &Device,
    ops: &Ops,
    c: &Config,
    key: CacheKey,
    capacity: usize,
    closure: &super::closure::Closure,
) -> Result<Cache> {
    let selected = attention_bucket(capacity)?;
    let evaluation_lengths = attention_lengths(capacity, 0)?;
    closure.measurement()?;
    let start = Instant::now();
    let s = &d.stream;
    let trial_ms = 30;
    let attention_width = c.num_attention_heads * c.head_dim;
    let mut q = s.alloc_zeros::<bf16>(attention_width)?;
    let zero_q = vec![bf16::ZERO; attention_width];
    let queries = random(attention_width, 4);
    let k = s.clone_htod(&random(capacity * c.kv_dim(), 1))?;
    let v_values = random(capacity * c.kv_dim(), 2);
    let v_max = v_values
        .iter()
        .map(|v| v.to_f32().abs() as f64)
        .fold(0f64, f64::max);
    let v = s.clone_htod(&v_values)?;
    let mut out = s.alloc_zeros::<bf16>(attention_width)?;
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
    let bucket = selected;
    let l = evaluation_lengths[0];
    {
        s.memcpy_htod(&[l as i32], &mut length)?;
        let mut expected = vec![0f64; c.kv_dim()];
        for kv_head in 0..c.num_key_value_heads {
            let mut sum = vec![0f64; c.head_dim];
            for row in 0..l {
                for (dim, value) in sum.iter_mut().enumerate() {
                    *value +=
                        v_values[(kv_head * capacity + row) * c.head_dim + dim].to_f32() as f64;
                }
            }
            for (dim, value) in sum.iter().enumerate() {
                expected[kv_head * c.head_dim + dim] = *value / l as f64;
            }
        }
        let mut best = (f64::INFINITY, f64::INFINITY, usize::MAX);
        for plan in plans(bucket) {
            let chunk = plan.chunk;
            let mut attention = Attention::candidate(&d.ctx, s, c, capacity, plan, false)?;
            if bucket >= 3 {
                let record = Occupancy {
                    plan,
                    blocks: capacity.div_ceil(chunk) * (16 / plan.qpack),
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
                    let measured_length = evaluation_lengths[index % 4];
                    s.memcpy_htod(&[measured_length as i32], &mut length)?;
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
                evaluation_lengths,
            });
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Cache {
        schema: 3,
        capacity,
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

pub(super) fn host_memory(root: &Path) -> Result<u64> {
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
    Ok(available)
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
    host_registration_bytes: u64,
    host_registration_seconds: [f64; 3],
    transfers: Vec<ExpertTransfer>,
    service: Vec<crate::experts::plan::ServiceCell>,
}

pub(super) struct ExpertCalibration {
    pub h2d_bytes_per_second: f64,
    pub d2h_bytes_per_second: f64,
    pub copy_engines: u32,
    pub host_registration_bytes: u64,
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
            self.schema == 3,
            "expert calibration schema {} expected 3",
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
        let expert_bytes = u64::from(geometry.layers)
            .checked_mul(u64::from(geometry.experts))
            .and_then(|n| n.checked_mul(slot_bytes))
            .context("expert registration extent overflow")?;
        ensure!(
            self.host_registration_bytes <= expert_bytes
                && self.host_registration_bytes.is_multiple_of(slot_bytes)
                && if self.host_registration_bytes == 0 {
                    self.host_registration_seconds[0].is_finite()
                        && self.host_registration_seconds[0] >= 0.0
                        && self.host_registration_seconds[1..] == [0.0; 2]
                } else {
                    self.host_registration_seconds
                        .iter()
                        .all(|&n| n.is_finite() && n > 0.0)
                },
            "invalid measured host registration bytes={} seconds={:?}",
            self.host_registration_bytes,
            self.host_registration_seconds
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
            host_registration_bytes: self.host_registration_bytes,
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

fn registration_extent<T>(
    maximum: u64,
    mut register: impl FnMut(u64) -> Result<Option<T>>,
) -> Result<Option<(u64, T)>> {
    let (mut accepted, mut end, mut probe) = (0, maximum, maximum);
    let mut measured = None;
    while accepted < end {
        match register(probe)? {
            Some(value) => {
                accepted = probe;
                measured = Some(value);
            }
            None => end = probe - 1,
        }
        probe = accepted + (end - accepted).div_ceil(2);
    }
    Ok(measured.map(|value| (accepted, value)))
}

fn measure_host_registration(d: &Device, bytes: u64, slot_bytes: u64) -> Result<(u64, [f64; 3])> {
    if d.ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED)?
        == 0
    {
        return Ok((0, [0.0; 3]));
    }
    ensure!(
        bytes > 0 && slot_bytes > 0 && bytes.is_multiple_of(slot_bytes),
        "invalid host registration measurement bytes={bytes} slot_bytes={slot_bytes}"
    );
    let start = Instant::now();
    let mut allocation = memmap2::MmapMut::map_anon(usize::try_from(bytes)?)
        .with_context(|| format!("allocate host registration measurement: bytes={bytes}"))?;
    allocation.fill(0);
    let allocated = start.elapsed().as_secs_f64();
    d.ctx.bind_to_thread()?;
    let pointer = allocation.as_mut_ptr().cast::<std::ffi::c_void>();
    let measured = registration_extent(bytes / slot_bytes, |slots| {
        let extent = usize::try_from(slots * slot_bytes)?;
        let start = Instant::now();
        // SAFETY: every prefix stays within the owned mapping and has no DMA users.
        let result = unsafe { sys::cuMemHostRegister_v2(pointer, extent, 0).result() };
        let registered = start.elapsed().as_secs_f64();
        if let Err(error) = result {
            if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                eprintln!(
                    "{}",
                    serde_json::json!({"host_registration_capacity_probe": {
                        "bytes":extent,"cuda_result":"CUDA_ERROR_OUT_OF_MEMORY",
                        "register_seconds":registered
                    }})
                );
                return Ok(None);
            }
            return Err(error)
                .with_context(|| format!("measure CUDA host registration: bytes={extent}"));
        }
        let start = Instant::now();
        // SAFETY: pointer is the successfully registered prefix and has no DMA users.
        unsafe { sys::cuMemHostUnregister(pointer).result() }
            .with_context(|| format!("unregister host registration measurement: bytes={extent}"))?;
        Ok(Some([registered, start.elapsed().as_secs_f64()]))
    })?;
    let (extent, [registered, unregistered]) = measured
        .map(|(slots, seconds)| (slots * slot_bytes, seconds))
        .unwrap_or((0, [0.0; 2]));
    eprintln!(
        "{}",
        serde_json::json!({"host_registration_measurement": {
            "bytes":extent,"allocation_bytes":bytes,"allocation_seconds":allocated,
            "cuda_result":if extent > 0 {"CUDA_SUCCESS"} else {"CUDA_ERROR_OUT_OF_MEMORY"},
            "register_seconds":registered,"unregister_seconds":unregistered
        }})
    );
    Ok((extent, [allocated, registered, unregistered]))
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
    let key = d.key.clone();
    let host = ExpertHost::read()?;
    let identity = super::closure::hash(&serde_json::to_vec(&(&host, geometry, expert_format))?);
    let path = dir.join(format!("experts-v3-{}-{identity}.json", key.tag()));
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
        let available = host_memory(Path::new("/"))?;
        ensure!(
            ring_bytes <= available / 2,
            "expert service ring {ring_bytes} B exceeds half of effective available host memory {available} B"
        );
        let copy_engines = u32::try_from(
            d.ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)?,
        )?;
        let registration_slots = ((available - available / 16) / slot_bytes)
            .min(u64::from(geometry.layers) * u64::from(geometry.experts));
        let (host_registration_bytes, host_registration_seconds) =
            measure_host_registration(d, registration_slots * slot_bytes, slot_bytes)?;
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
            schema: 3,
            expert_format,
            slot_bytes,
            key: key.clone(),
            host: host.clone(),
            geometry,
            copy_engines,
            ring_experts,
            host_registration_bytes,
            host_registration_seconds,
            transfers,
            service,
        };
        cache.check(&key, &host, &geometry, slot_bytes)?;
        super::closure::write_cache(&path, &cache)?;
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
    let registration_supported = d
        .ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED)?
        != 0;
    ensure!(
        registration_supported || cache.host_registration_bytes == 0,
        "expert host registration profile differs from driver support={registration_supported} bytes={}",
        cache.host_registration_bytes
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
    fn uniform_attention_error_maps_query_heads_to_kv_heads() {
        let c = Config {
            model_type: "llama".into(),
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            vocab_size: 128,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: vec![1],
            attention_bias: false,
            mlp_bias: false,
        };
        let expected: Vec<_> = (0..c.num_key_value_heads)
            .flat_map(|head| vec![f64::from(head as u32); c.head_dim])
            .collect();
        let actual: Vec<_> = (0..c.num_attention_heads)
            .flat_map(|head| vec![bf16::from_f32(head as f32); c.head_dim])
            .collect();
        let (max_error, _) = uniform_attention_error(&c, &actual, &expected, c.head_dim, 1, 1.0);
        assert_eq!(max_error, 14.0);
    }

    #[test]
    fn expert_cells_guard_capacity_inputs() -> Result<()> {
        use crate::experts::{catalogue::ExpertGeometry, plan::ServiceCell};
        for limit in 0..=17 {
            let measured = registration_extent(17, |slots| Ok((slots <= limit).then_some(slots)))?;
            assert_eq!(measured.map(|(slots, _)| slots).unwrap_or(0), limit);
            assert!(measured.is_none_or(|(slots, value)| slots == value));
        }
        let mut attempts = 0;
        assert_eq!(
            registration_extent(17, |_| {
                attempts += 1;
                Ok(Some(()))
            })?,
            Some((17, ()))
        );
        assert_eq!(attempts, 1);
        assert!(
            registration_extent::<()>(0, |_| panic!("zero extent must not register"))?.is_none()
        );
        assert!(registration_extent::<()>(17, |_| anyhow::bail!("register failed")).is_err());
        println!(
            "PASS host_registration_search maximum_slots=17 exact_limits=0..17 fatal_error=propagated"
        );
        let mut bytes: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
        check_expert_transfer(&bytes)?;
        bytes[2048] ^= 1;
        assert!(check_expert_transfer(&bytes).is_err());
        let key = CacheKey {
            device_uuid: "fixture".into(),
            driver_version: "fixture".into(),
            binary_version: "fixture".into(),
            cublaslt_version: 130201,
        };
        let host = ExpertHost {
            cpu: "fixture".into(),
            isa: "fixture".into(),
            affinity: "0-1".into(),
            threads: 2,
            llc_bytes: 128,
        };
        let geometry = ExpertGeometry {
            first_layer: 0,
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
            schema: 3,
            expert_format: crate::model_spec::ExpertFormat::Bf16,
            slot_bytes,
            key: key.clone(),
            host: host.clone(),
            geometry,
            copy_engines: 2,
            ring_experts: host.ring_experts(&geometry, slot_bytes)?,
            host_registration_bytes: 16 * slot_bytes,
            host_registration_seconds: [0.001; 3],
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
        cache.host_registration_bytes = 0;
        cache.host_registration_seconds = [0.001, 0.0, 0.0];
        assert_eq!(
            cache
                .check(&key, &host, &geometry, slot_bytes)?
                .host_registration_bytes,
            0
        );
        cache.host_registration_bytes = 16 * slot_bytes;
        cache.host_registration_seconds = [0.001; 3];
        let mut invalid_registration = serde_json::to_value(&cache)?;
        invalid_registration["host_registration_bytes"] = serde_json::json!(u64::MAX);
        let invalid_registration: ExpertCache = serde_json::from_value(invalid_registration)?;
        assert!(
            invalid_registration
                .check(&key, &host, &geometry, slot_bytes)
                .is_err()
        );
        assert_eq!(measured.host_registration_bytes, 16 * slot_bytes);
        for seconds in [[f64::NAN, 0.001, 0.001], [0.001, 0.0, 0.001]] {
            cache.host_registration_seconds = seconds;
            assert!(cache.check(&key, &host, &geometry, slot_bytes).is_err());
        }
        cache.host_registration_seconds = [0.001; 3];
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
        let measured = measured.check(&device.key, &ExpertHost::read()?, &geometry, slot_bytes)?;
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
        assert_eq!(
            measured.host_registration_bytes,
            reused.host_registration_bytes
        );
        assert_eq!(measured.service, reused.service);
        assert_eq!(std::fs::read(&path)?, bytes);
        assert_eq!(std::fs::metadata(&path)?.modified()?, modified);
        let (device_free, device_total) = device.ctx.mem_get_info()?;
        let available = host_memory(Path::new("/"))?;
        let copies = measured.copy_engines.min(2);
        let reserves = crate::experts::plan::Reserves {
            device_fixed_bytes: device_total as u64 / 16,
            host_fixed_bytes: available / 16,
            residual_row_bytes: 0,
            normalized_row_bytes: 0,
            contribution_row_bytes: 0,
            done_row_bytes: 0,
            route_row_bytes: 0,
            prefill_staging_slots: copies,
            disk_ring_slots: copies,
        };
        let capacity = crate::experts::plan::capacity(
            &crate::experts::plan::PlanInputs {
                geometry,
                layout: crate::experts::catalogue::ExpertLayout::new(
                    spec.expert_format.context("MoE expert_format missing")?,
                    &geometry,
                )?,
                device_free_bytes: device_free as u64,
                host_available_bytes: available,
                h2d_bytes_per_second: measured.h2d_bytes_per_second,
                d2h_bytes_per_second: measured.d2h_bytes_per_second,
                copy_engines: measured.copy_engines,
                disk_readers: copies,
                host_registration_bytes: measured.host_registration_bytes,
                service: &measured.service,
                reserves: &reserves,
                span_rows: 1,
                wave_rows: 1,
            },
            1,
        )?;
        println!(
            "PASS expert_setup_capacity_probe {}",
            serde_json::json!({"pinned":capacity.pinned,"geometry":geometry,
                "slot_bytes":slot_bytes,"registered_bytes":measured.host_registration_bytes,
                "host_blocks":capacity.host_blocks,"spare_blocks":capacity.spare_blocks,
                "disk_bytes":capacity.disk_bytes,"device_free_bytes":device_free,
                "device_fixed_bytes":reserves.device_fixed_bytes,
                "host_available_bytes":available,"host_fixed_bytes":reserves.host_fixed_bytes,
                "prefill_staging_slots":copies,"disk_ring_slots":copies,
                "span_rows":1,"wave_rows":1,"admission_cap":1,
                "scope":"calibration capacity probe; model weights and inference workspace not loaded"})
        );
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

    fn calibration_expected(
        c: &Config,
        values: &[bf16],
        shape: crate::backend::setup::MultiShape,
        prefix: usize,
        tree: bool,
    ) -> Vec<f64> {
        let mut expected = vec![0.0; shape.rows * c.kv_dim()];
        let mut masks = vec![1u64; shape.rows];
        for row in 1..shape.rows {
            masks[row] = masks[(row - 1) / 4] | (1u64 << row);
        }
        for head in 0..c.num_key_value_heads {
            for dim in 0..c.head_dim {
                let value = |position: usize| {
                    values[(head * shape.capacity + position) * c.head_dim + dim].to_f32() as f64
                };
                let sum: f64 = (0..prefix).map(value).sum();
                for row in 0..shape.rows {
                    let (extra, count) = if shape.rows == 1 {
                        (0.0, 0)
                    } else if tree {
                        let positions =
                            (0..shape.rows).filter(|&col| masks[row] & (1u64 << col) != 0);
                        (
                            positions.map(|col| value(prefix + col)).sum(),
                            masks[row].count_ones() as usize,
                        )
                    } else {
                        let count = if shape.causal { row + 1 } else { shape.rows };
                        ((0..count).map(|col| value(prefix + col)).sum(), count)
                    };
                    expected[(row * c.num_key_value_heads + head) * c.head_dim + dim] =
                        (sum + extra) / (prefix + count) as f64;
                }
            }
        }
        expected
    }

    fn calibration_attention_cell(
        d: &Device,
        ops: &Ops,
        c: &Config,
        dir: &Path,
        shape: crate::backend::setup::MultiShape,
        tree: bool,
    ) -> Result<()> {
        use super::super::{flash::Shape, verification::Verification};
        use crate::backend::setup::attention_lengths;
        let crate::backend::setup::MultiShape {
            capacity,
            rows,
            causal,
        } = shape;
        let lengths = attention_lengths(capacity, if rows == 1 { 0 } else { rows })?;
        let bucket = attention_bucket(capacity)?;
        let writer = super::super::closure::Closure::new(true, "capacity calibration gate".into());
        let reader = super::super::closure::Closure::new(false, "capacity calibration gate".into());
        let cold = Instant::now();
        let (plan, mut decode, mut multi, path) = if rows == 1 {
            let calibrated = load(d, ops, c, dir, capacity, &writer)?;
            let plan = calibrated.choose(capacity)?.0;
            let path = cache_path(dir, &d.key, bucket, capacity);
            let bytes = std::fs::read(&path)?;
            let hit = load(d, ops, c, dir, capacity, &reader)?;
            ensure!(
                hit.hit && std::fs::read(&path)? == bytes,
                "decode cache hit changed bytes"
            );
            let wrong = if capacity == 131072 {
                capacity - 1
            } else {
                capacity + 1
            };
            ensure!(
                hit.choose(wrong).is_err(),
                "decode accepted another capacity"
            );
            (
                plan,
                Some(Attention::candidate(
                    &d.ctx, &d.stream, c, capacity, plan, false,
                )?),
                None,
                path,
            )
        } else {
            let calibrated = if tree {
                super::super::multi_calibrate::load_tree(d, ops, c, dir, rows, capacity, &writer)?
            } else {
                super::super::multi_calibrate::load(d, ops, c, dir, shape, &writer)?
            };
            let plan = calibrated.choose(capacity)?.0;
            let path = super::super::multi_calibrate::cache_path(dir, &d.key, bucket, shape, tree);
            let bytes = std::fs::read(&path)?;
            let hit = if tree {
                super::super::multi_calibrate::load_tree(d, ops, c, dir, rows, capacity, &reader)?
            } else {
                super::super::multi_calibrate::load(d, ops, c, dir, shape, &reader)?
            };
            ensure!(
                std::fs::read(&path)? == bytes,
                "multi cache hit changed bytes"
            );
            let wrong = if capacity == 131072 {
                capacity - 1
            } else {
                capacity + 1
            };
            ensure!(
                hit.choose(wrong).is_err(),
                "multi accepted another capacity"
            );
            (
                plan.launch,
                None,
                Some(Verification::with_plan(&d.ctx, &d.stream, c, shape, plan)?),
                path,
            )
        };
        let cold_and_reload_s = cold.elapsed().as_secs_f64();
        let cache: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            cache["capacity"].as_u64() == Some(capacity as u64),
            "calibration capacity differs from allocation"
        );
        let cold_ms = cache["wall_ms"]
            .as_f64()
            .context("calibration wall_ms missing")?;
        let s = &d.stream;
        let q = s.alloc_zeros::<bf16>(rows * c.qkv_dim())?;
        let k = s.alloc_zeros::<bf16>(capacity * c.kv_dim())?;
        let values: Vec<_> = (0..capacity * c.kv_dim())
            .map(|i| bf16::from_f32(((i * 37 + i / 128 * 11) % 263) as f32 / 512.0 - 131.0 / 512.0))
            .collect();
        let v = s.clone_htod(&values)?;
        let mut out = s.alloc_zeros::<bf16>(rows * c.hidden_size)?;
        let mut prefix = s.alloc_zeros::<i32>(1)?;
        let mut masks = vec![1u64; rows];
        for row in 1..rows {
            masks[row] = masks[(row - 1) / 4] | (1u64 << row);
        }
        let ancestors = s.clone_htod(&masks)?;
        for length in lengths {
            ensure!(
                length + if rows == 1 { 0 } else { rows } <= capacity,
                "calibration evaluation exceeds capacity"
            );
            s.memcpy_htod(&[length as i32], &mut prefix)?;
            let mut launch = |out: &mut cudarc::driver::CudaSlice<bf16>| -> Result<()> {
                if let Some(op) = &mut decode {
                    return op.run(s, &q, &k, &v, out, &prefix);
                }
                let op = multi.as_mut().context("multi operation missing")?;
                let geometry = Shape {
                    rows,
                    start: 0,
                    q_heads: c.num_attention_heads,
                    kv_heads: c.num_key_value_heads,
                    capacity,
                    head_dim: c.head_dim,
                    causal,
                    window_left: None,
                };
                if tree {
                    op.run_tree(
                        super::super::verification::Query {
                            qkv: &q,
                            k: &k,
                            v: &v,
                            out,
                            prefix: &prefix,
                            shape: geometry,
                        },
                        &ancestors,
                    )
                } else {
                    op.run(&q, &k, &v, out, &prefix, geometry)
                }
            };
            launch(&mut out)?;
            let eager = s.clone_dtoh(&out)?;
            ensure!(
                eager.iter().all(|v| v.is_finite()),
                "nonfinite selected attention output"
            );
            let expected = calibration_expected(c, &values, shape, length, tree);
            let (error, ratio) = uniform_attention_error(
                c,
                &eager,
                &expected,
                plan.chunk,
                length + if rows == 1 { 0 } else { rows },
                0.5,
            );
            ensure!(
                ratio <= 1.0,
                "DEFECT selected calibration attention error={error} error/bound={ratio} C={capacity} R={rows} tree={tree}"
            );
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            launch(&mut out)?;
            let graph = s
                .end_capture(sys::CUgraphInstantiate_flags(0))?
                .context("selected attention graph missing")?;
            graph.launch()?;
            let actual = s.clone_dtoh(&out)?;
            ensure!(
                actual
                    .iter()
                    .zip(&eager)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "DEFECT selected attention graph/eager bits differ"
            );
        }
        println!(
            "PASS calibration_attention_capacity C={capacity} R={rows} causal={causal} tree={tree} lengths={lengths:?} kv=[2,{capacity},128] cold_ms={cold_ms} cold_and_reload_s={cold_and_reload_s} cache_reuse=byte_identical uniform_bound=1 graph_eager_bits=0"
        );
        Ok(())
    }

    #[test]
    #[ignore = "exclusive CUDA, FF_ARGMAX_MODEL and an empty FLYINGFISH_RUNTIME_DIR"]
    fn calibration_identity_capacity_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let model =
            PathBuf::from(std::env::var_os("FF_ARGMAX_MODEL").context("FF_ARGMAX_MODEL missing")?);
        let c = Config::read(&model)?;
        ensure!(
            c.hidden_size == 2048
                && c.num_attention_heads == 16
                && c.num_key_value_heads == 2
                && c.head_dim == 128,
            "calibration gate requires native MiniCPM5-2B geometry"
        );
        let dir = runtime_dir(
            &PathBuf::from(
                std::env::var_os("FLYINGFISH_RUNTIME_DIR")
                    .context("FLYINGFISH_RUNTIME_DIR missing")?,
            ),
            &model,
            true,
        )?;
        let d = Device::new(0)?;
        let ops = Ops::new(&d.ctx)?;
        d.finish_upload()?;
        let mut defects = 0;
        let mut cells = 0;
        for capacity in [1, 64, 257, 1023, 1312, 2049, 16385, 49153, 98305, 131072] {
            for (rows, causal, tree) in [
                (1, false, false),
                (7, false, false),
                (8, true, false),
                (16, false, true),
                (64, false, true),
            ] {
                if capacity < rows {
                    continue;
                }
                cells += 1;
                let shape = crate::backend::setup::MultiShape {
                    capacity,
                    rows,
                    causal,
                };
                if let Err(error) = calibration_attention_cell(&d, &ops, &c, &dir, shape, tree) {
                    let message = format!("{error:#}");
                    ensure!(
                        message.contains("DEFECT") || message.contains("reference mismatch"),
                        "RUNNER FAIL calibration C={capacity} R={rows} tree={tree}: {message}"
                    );
                    defects += 1;
                    println!(
                        "DEFECT calibration_attention_capacity C={capacity} R={rows} causal={causal} tree={tree} error={error:#}"
                    );
                }
            }
        }
        if let Err(error) = super::super::blas::Blas::calibration_identity_gate(&d, &ops, &dir) {
            ensure!(
                format!("{error:#}").contains("DEFECT"),
                "RUNNER FAIL Lt identity: {error:#}"
            );
            defects += 1;
            println!("DEFECT calibration_lt_identity error={error:#}");
        }
        d.stream.synchronize()?;
        println!(
            "{} calibration_identity_capacity cells={cells} defects={defects}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "DEFECT calibration gate cells={defects}");
        Ok(())
    }

    #[test]
    fn cache_requires_measured_bucket_and_complete_candidates() -> Result<()> {
        assert_eq!(
            driver_release(
                "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 595.99.02 Release Build"
            )?,
            "595.99.02"
        );
        assert!(driver_release("NVRM version: unavailable").is_err());
        assert!(driver_release("GCC version: 13.3.0").is_err());
        for (bucket, &capacity) in crate::backend::setup::ATTENTION_CAPACITIES
            .iter()
            .enumerate()
        {
            let lengths = attention_lengths(capacity, 0)?;
            let length = lengths[0];
            let key = CacheKey {
                device_uuid: "test".into(),
                driver_version: "test".into(),
                binary_version: "test".into(),
                cublaslt_version: 130201,
            };
            let mut trials = Vec::new();
            let mut phase1 = Vec::new();
            for plan in plans(bucket) {
                if bucket >= 3 {
                    let o = Occupancy {
                        plan,
                        blocks: capacity.div_ceil(plan.chunk) * (16 / plan.qpack),
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
                    evaluation_lengths: lengths,
                });
            }
            let mut cache = Cache {
                schema: 3,
                key,
                winner: trials[0].plan,
                bucket,
                capacity,
                lengths: LENGTHS,
                trials,
                sms: 128,
                phase1,
                wall_ms: 1.0,
            };
            check(&cache, &cache.key, capacity)?;
            assert!(check(&cache, &cache.key, capacity - 1).is_err());
            let mut stale = cache.key.clone();
            stale.cublaslt_version += 1;
            assert!(check(&cache, &stale, capacity).is_err());
            let winner = cache.winner;
            cache.winner = cache.trials[1].plan;
            assert!(check(&cache, &cache.key, capacity).is_err());
            cache.winner = winner;
            if bucket >= 3 {
                cache.phase1[0].blocks += 1;
                assert!(check(&cache, &cache.key, capacity).is_err());
                cache.phase1[0].blocks -= 1;
            }
            cache.schema = 2;
            assert!(check(&cache, &cache.key, capacity).is_err());
            cache.schema = 3;
            cache.trials[0].evaluation_lengths[3] = capacity + 1;
            assert!(check(&cache, &cache.key, capacity).is_err());
            cache.trials[0].evaluation_lengths = lengths;
            cache.trials.pop();
            assert!(check(&cache, &cache.key, capacity).is_err());
        }
        Ok(())
    }
}
