use super::{
    Device, calibrate,
    flash::Shape,
    ops::{Ops, grid},
    verification::{Query, Verification},
};
use crate::{
    backend::setup::{AttentionPlan, CacheKey, MultiImpl, MultiPlan, MultiShape, Selection},
    config::Config,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{PushKernelArg, sys};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const LENGTHS: [usize; 3] = [1024, 8192, 32768];

#[derive(Serialize, Deserialize)]
struct Trial {
    bucket: usize,
    plan: MultiPlan,
    samples: usize,
    median_us: f64,
    p90_us: f64,
    p10_us: f64,
    max_abs_error: f64,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    key: CacheKey,
    rows: usize,
    causal: bool,
    #[serde(default)]
    tree: bool,
    winners: [MultiPlan; 3],
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
    pub fn choose(&self, capacity: usize) -> (MultiPlan, Selection) {
        let bucket = if capacity <= 2048 {
            0
        } else if capacity <= 16384 {
            1
        } else {
            2
        };
        (
            self.cache.winners[bucket],
            Selection::Calibrated {
                key: self.cache.key.clone(),
                cached: self.hit,
            },
        )
    }
}

fn candidates(bucket: usize, rows: usize, causal: bool, tree: bool) -> Vec<MultiPlan> {
    let mut plans = Vec::new();
    if causal && !tree && rows <= 16 {
        for chunk in [128, 256, 512, 1024] {
            for qpack in [1, 2, 4, 8] {
                for threads in [128, 256] {
                    plans.push(MultiPlan {
                        implementation: MultiImpl::V1,
                        launch: AttentionPlan {
                            chunk,
                            qpack,
                            threads,
                            merge_threads: 256,
                        },
                    });
                }
            }
        }
    }
    let chunks = match (bucket, rows) {
        (0, 1..=32) => [64, 128],
        (0, _) => [128, 64],
        (1, 1..=8) => [128, 256],
        (1, 9..=32) => [256, 128],
        (1, _) => [512, 256],
        (2, 1..=8) => [256, 512],
        (2, 9..=16) => [512, 1024],
        (2, 17..=32) => [1024, 512],
        (2, _) => [2048, 1024],
        _ => unreachable!(),
    };
    for chunk in chunks {
        plans.push(MultiPlan {
            implementation: MultiImpl::Tcmqa,
            launch: AttentionPlan {
                chunk,
                qpack: 8,
                threads: 128,
                merge_threads: 256,
            },
        });
    }
    plans
}

fn check(cache: &Cache, key: &CacheKey, rows: usize, causal: bool, tree: bool) -> Result<()> {
    ensure!(
        cache.key == *key && cache.rows == rows && cache.causal == causal && cache.tree == tree,
        "multi attention cache identity mismatch"
    );
    for bucket in 0..3 {
        let expected = candidates(bucket, rows, causal, tree);
        let trials: Vec<_> = cache.trials.iter().filter(|t| t.bucket == bucket).collect();
        ensure!(
            trials.len() == expected.len(),
            "multi attention cache candidate count mismatch"
        );
        for plan in expected {
            ensure!(
                trials.iter().filter(|t| t.plan == plan).count() == 1,
                "multi attention cache lacks a candidate"
            );
        }
        ensure!(
            trials.iter().all(|t| t.samples > 0
                && t.median_us.is_finite()
                && t.median_us > 0.0
                && t.p90_us.is_finite()),
            "invalid multi attention timing"
        );
        let best = trials
            .into_iter()
            .min_by(|a, b| {
                a.median_us
                    .total_cmp(&b.median_us)
                    .then(a.p90_us.total_cmp(&b.p90_us))
            })
            .context("empty multi attention bucket")?;
        ensure!(
            cache.winners[bucket] == best.plan,
            "multi attention cache winner mismatch"
        );
    }
    Ok(())
}

pub fn load(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    rows: usize,
    causal: bool,
) -> Result<Calibrated> {
    load_mask(d, ops, c, dir, rows, causal, false)
}

pub fn load_tree(d: &Device, ops: &Ops, c: &Config, dir: &Path, rows: usize) -> Result<Calibrated> {
    ensure!(
        [16, 32, 64].contains(&rows),
        "tree calibration requires budget16/32/64"
    );
    load_mask(d, ops, c, dir, rows, false, true)
}

fn load_mask(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    rows: usize,
    causal: bool,
    tree: bool,
) -> Result<Calibrated> {
    ensure!(
        (1..=64).contains(&rows),
        "multi attention M must be within1..64"
    );
    let key = calibrate::key(d)?;
    let path = dir.join(format!(
        "multi-m{rows}-{}-{}-{}-{}.json",
        if tree {
            "tree"
        } else if causal {
            "causal"
        } else {
            "full"
        },
        key.device_uuid,
        key.driver_version,
        key.binary_version
    ));
    let (cache, hit) = if path.exists() {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)?;
        check(&cache, &key, rows, causal, tree)?;
        (cache, true)
    } else {
        let cache = measure(d, ops, c, key, rows, causal, tree)?;
        check(&cache, &cache.key, rows, causal, tree)?;
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

fn measure(
    d: &Device,
    ops: &Ops,
    c: &Config,
    key: CacheKey,
    rows: usize,
    causal: bool,
    tree: bool,
) -> Result<Cache> {
    let start = Instant::now();
    let s = &d.stream;
    let capacities = LENGTHS.map(|n| n + 256 + rows);
    let q_values = random(rows * c.qkv_dim(), 4);
    let zero_q = vec![bf16::ZERO; q_values.len()];
    let mut q = s.clone_htod(&q_values)?;
    let k = s.clone_htod(&random(capacities[2] * c.kv_dim(), 1))?;
    let v_values = random(capacities[2] * c.kv_dim(), 2);
    let v = s.clone_htod(&v_values)?;
    let mut out = s.alloc_zeros::<bf16>(rows * c.hidden_size)?;
    let mut length = s.alloc_zeros::<i32>(1)?;
    let mut masks = vec![1u64; rows];
    for i in 1..rows {
        masks[i] = masks[(i - 1) / 4] | (1u64 << i);
    }
    let anc = if tree {
        Some(s.clone_htod(&masks)?)
    } else {
        None
    };
    let flush_n = d.info.l2_bytes.max(1024 * 1024);
    let mut flush = s.clone_htod(&random(flush_n, 3))?;
    let zero = s.alloc_zeros::<bf16>(flush_n)?;
    let events: Vec<_> = (0..4)
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
    let mut winners = [candidates(0, rows, causal, tree)[0]; 3];
    for (bucket, &l) in LENGTHS.iter().enumerate() {
        let capacity = capacities[bucket];
        let mut expected = vec![0f64; rows * c.kv_dim()];
        for head in 0..c.num_key_value_heads {
            let mut sum = vec![0f64; c.head_dim];
            for pos in 0..l {
                for (dim, a) in sum.iter_mut().enumerate() {
                    *a += v_values[(head * capacity + pos) * c.head_dim + dim].to_f32() as f64;
                }
            }
            if !causal && !tree {
                for pos in l..l + rows {
                    for (dim, a) in sum.iter_mut().enumerate() {
                        *a += v_values[(head * capacity + pos) * c.head_dim + dim].to_f32() as f64;
                    }
                }
            }
            for row in 0..rows {
                if tree {
                    for (dim, &prefix_sum) in sum.iter().enumerate() {
                        let mut value = prefix_sum;
                        for col in 0..rows {
                            if masks[row] & (1u64 << col) != 0 {
                                value += v_values[(head * capacity + l + col) * c.head_dim + dim]
                                    .to_f32() as f64;
                            }
                        }
                        expected[(row * c.num_key_value_heads + head) * c.head_dim + dim] =
                            value / (l + masks[row].count_ones() as usize) as f64;
                    }
                    continue;
                }
                for (dim, a) in sum.iter_mut().enumerate() {
                    if causal {
                        *a += v_values[(head * capacity + l + row) * c.head_dim + dim].to_f32()
                            as f64;
                    }
                    expected[(row * c.num_key_value_heads + head) * c.head_dim + dim] = *a
                        / if causal {
                            (l + row + 1) as f64
                        } else {
                            (l + rows) as f64
                        };
                }
            }
        }
        let mut best = (f64::INFINITY, f64::INFINITY);
        for plan in candidates(bucket, rows, causal, tree) {
            let mut op = Verification::with_plan(
                &d.ctx,
                s,
                c,
                MultiShape {
                    capacity,
                    rows,
                    causal,
                },
                plan,
            )?;
            let shape = || Shape {
                rows,
                start: 0,
                q_heads: 16,
                kv_heads: 2,
                capacity,
                causal,
            };
            s.memcpy_htod(&zero_q, &mut q)?;
            s.memcpy_htod(&[l as i32], &mut length)?;
            execute(
                &mut op,
                Query {
                    qkv: &q,
                    k: &k,
                    v: &v,
                    out: &mut out,
                    prefix: &length,
                    shape: shape(),
                },
                anc.as_ref(),
            )?;
            s.synchronize()?;
            let actual = s.clone_dtoh(&out)?;
            ensure!(
                actual.iter().all(|x| x.is_finite()),
                "nonfinite multi attention calibration output"
            );
            let max_error = actual
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let row = i / c.hidden_size;
                    let head = (i % c.hidden_size)
                        / c.head_dim
                        / (c.num_attention_heads / c.num_key_value_heads);
                    (x.to_f32() as f64
                        - expected
                            [(row * c.num_key_value_heads + head) * c.head_dim + i % c.head_dim])
                        .abs()
                })
                .fold(0f64, f64::max);
            ensure!(
                max_error <= 0.0001,
                "multi attention uniform reference mismatch: {max_error}"
            );
            s.memcpy_htod(&q_values, &mut q)?;
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let launched = execute(
                &mut op,
                Query {
                    qkv: &q,
                    k: &k,
                    v: &v,
                    out: &mut out,
                    prefix: &length,
                    shape: shape(),
                },
                anc.as_ref(),
            );
            let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = captured?.context("empty multi attention candidate graph")?;
            graph.upload()?;
            graph.launch()?;
            s.synchronize()?;
            let timer = Instant::now();
            let mut samples = Vec::new();
            while timer.elapsed() < Duration::from_millis(if causal { 10 } else { 30 }) {
                for (index, (begin, end)) in events.iter().enumerate() {
                    s.memcpy_htod(&[(l + [0, 1, 128, 256][index]) as i32], &mut length)?;
                    unsafe {
                        s.launch_builder(&ops.residual)
                            .arg(&mut flush)
                            .arg(&zero)
                            .arg(&(flush_n as i32))
                            .launch(grid(flush_n.div_ceil(2048), 256))?;
                    }
                    begin.record(s)?;
                    graph.launch()?;
                    end.record(s)?;
                }
                s.synchronize()?;
                for (begin, end) in &events {
                    samples.push(begin.elapsed_ms(end)? as f64 * 1000.0);
                }
            }
            let median = calibrate::percentile(&samples, 0.5);
            let p90 = calibrate::percentile(&samples, 0.9);
            if (median, p90) < best {
                best = (median, p90);
                winners[bucket] = plan;
            }
            trials.push(Trial {
                bucket,
                plan,
                samples: samples.len(),
                median_us: median,
                p90_us: p90,
                p10_us: calibrate::percentile(&samples, 0.1),
                max_abs_error: max_error,
            });
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        wall_ms <= 2000.0,
        "multi attention calibration exceeds2s: {wall_ms}ms"
    );
    Ok(Cache {
        key,
        rows,
        causal,
        tree,
        winners,
        trials,
        wall_ms,
    })
}

fn execute(
    op: &mut Verification,
    query: Query<'_>,
    anc: Option<&cudarc::driver::CudaSlice<u64>>,
) -> Result<()> {
    if let Some(anc) = anc {
        op.run_tree(query, anc)
    } else {
        op.run(
            query.qkv,
            query.k,
            query.v,
            query.out,
            query.prefix,
            query.shape,
        )
    }
}
