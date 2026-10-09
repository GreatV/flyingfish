use super::{
    Device, calibrate,
    flash::Shape,
    ops::{Ops, grid},
    verification::{Query, Verification, phase1_residency},
};
use crate::{
    backend::setup::{
        ATTENTION_LENGTHS, AttentionPlan, CacheKey, MultiImpl, MultiPlan, MultiShape, Selection,
        attention_bucket, attention_chunks,
    },
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

const LENGTHS: [usize; 5] = ATTENTION_LENGTHS;

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
    winner: MultiPlan,
    bucket: usize,
    schema: u32,
    lengths: [usize; 5],
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
    pub fn choose(&self, capacity: usize) -> Result<(MultiPlan, Selection)> {
        let bucket = attention_bucket(capacity)?;
        ensure!(
            self.cache.bucket == bucket,
            "multi calibration bucket does not match capacity"
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

#[derive(Clone, Copy)]
struct Budget {
    kv_bytes_per_token: usize,
    l2_bytes: usize,
    sms: usize,
    /// Resident phase-1 blocks per SM, indexed like `MultiImpl`.
    phase1_resident: [usize; 3],
    kv_heads: usize,
}

fn budget(d: &Device, c: &Config) -> Result<Budget> {
    Ok(Budget {
        kv_bytes_per_token: c.kv_dim() * std::mem::size_of::<bf16>() * 2,
        l2_bytes: d.info.l2_bytes,
        sms: d.info.sms,
        phase1_resident: phase1_residency(&d.ctx)?,
        kv_heads: c.num_key_value_heads,
    })
}

/// Phase-1 CTAs that fit on the whole device: SM count times the driver-reported
/// resident blocks per SM for that implementation's block size and dynamic
/// shared memory.
fn phase1_slots(budget: Budget, implementation: MultiImpl) -> usize {
    budget.sms * budget.phase1_resident[implementation as usize].max(1)
}

/// KV capacity a calibration trial for this length bucket and row count
/// allocates and launches with.
fn calibration_capacity(length: usize, rows: usize) -> usize {
    length + 256 + rows
}

/// Chunks whose phase-1 grid fills one and two whole waves of resident CTAs:
/// rounding the chunk up to a 64-position step keeps the grid at or under the
/// wave, so no tail block is launched. The grid is
/// ceil(calibration_capacity / chunk) chunks per KV head times the row tiles.
fn wave_chunks(
    length: usize,
    rows: usize,
    kv_heads: usize,
    budget: Budget,
    implementation: MultiImpl,
) -> Vec<usize> {
    let tiles = (8 * rows)
        .div_ceil(implementation.phase1_tile_rows())
        .max(1);
    let per_wave = (phase1_slots(budget, implementation) / (kv_heads * tiles)).max(1);
    let work = calibration_capacity(length, rows);
    let mut chunks: Vec<usize> = [1usize, 2]
        .into_iter()
        .map(|waves| work.div_ceil(per_wave * waves).div_ceil(64) * 64)
        .collect();
    chunks.sort_unstable();
    chunks.dedup();
    chunks
}

/// Baseline chunk candidates, always offered alongside the derived ones.
fn baseline_chunks(bucket: usize, rows: usize) -> [usize; 2] {
    match (bucket, rows) {
        (0, 1..=32) => [64, 128],
        (0, _) => [128, 64],
        (1, 1..=8) => [128, 256],
        (1, 9..=32) => [256, 128],
        (1, _) => [512, 256],
        (2, 1..=8) => [256, 512],
        (2, 9..=16) => [512, 1024],
        (2, 17..=32) => [1024, 512],
        (2, _) => [2048, 1024],
        (3, 1..=8) => [512, 1024],
        (3, 9..=16) => [1024, 2048],
        (3, 17..=32) => [2048, 1024],
        (3, _) => [4096, 2048],
        (4, 1..=8) => [1024, 2048],
        (4, 9..=16) => [2048, 4096],
        (4, 17..=32) => [4096, 2048],
        (4, _) => [4096, 2048],
        _ => unreachable!(),
    }
}

fn candidates(
    bucket: usize,
    rows: usize,
    causal: bool,
    tree: bool,
    budget: Budget,
) -> Vec<MultiPlan> {
    let mut plans = Vec::new();
    if causal && !tree && rows <= 16 {
        for &chunk in attention_chunks(bucket) {
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
    let chunks = baseline_chunks(bucket, rows);
    // Each implementation sweeps its baseline seeds plus the two whole-wave
    // chunks derived from its own tile height and residency.
    let with_waves = |implementation: MultiImpl| -> Vec<usize> {
        let mut derived = wave_chunks(
            LENGTHS[bucket],
            rows,
            budget.kv_heads,
            budget,
            implementation,
        );
        derived.extend_from_slice(&chunks);
        derived.sort_unstable();
        derived.dedup();
        derived
    };
    for chunk in with_waves(MultiImpl::Tcmqa) {
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
    // 128-row tile variant. It reads each K/V tile once per launch; the 64-row
    // variant reads it once per row tile, so it is offered only when the query
    // rows span two or more 64-row tiles.
    // One layer's K/V prefix at the bucket's measured length is
    // LENGTHS[bucket] * kv_bytes_per_token bytes; the extra 64-row tile re-reads
    // that prefix, so the variant is offered only where it exceeds the device L2.
    if 8 * rows > 64 && LENGTHS[bucket] * budget.kv_bytes_per_token > budget.l2_bytes {
        for chunk in with_waves(MultiImpl::TcmqaW) {
            plans.push(MultiPlan {
                implementation: MultiImpl::TcmqaW,
                launch: AttentionPlan {
                    chunk,
                    qpack: 8,
                    threads: 256,
                    merge_threads: 256,
                },
            });
        }
    }
    plans
        .into_iter()
        .flat_map(|p| {
            [
                p,
                MultiPlan {
                    launch: AttentionPlan {
                        merge_threads: 512,
                        ..p.launch
                    },
                    ..p
                },
            ]
        })
        .collect()
}

fn check(
    cache: &Cache,
    key: &CacheKey,
    rows: usize,
    causal: bool,
    tree: bool,
    bucket: usize,
    budget: Budget,
) -> Result<()> {
    ensure!(
        cache.schema == 2
            && cache.lengths == LENGTHS
            && cache.key == *key
            && cache.rows == rows
            && cache.causal == causal
            && cache.tree == tree
            && bucket < LENGTHS.len()
            && cache.bucket == bucket
            && cache.trials.len() == candidates(bucket, rows, causal, tree, budget).len(),
        "multi attention cache identity mismatch"
    );
    let expected = candidates(bucket, rows, causal, tree, budget);
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
        cache.winner == best.plan,
        "multi attention cache winner mismatch"
    );
    Ok(())
}

pub(super) fn load(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    shape: MultiShape,
    closure: &super::closure::Closure,
) -> Result<Calibrated> {
    load_mask(d, ops, c, dir, shape, false, closure)
}

pub(super) fn load_tree(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    rows: usize,
    capacity: usize,
    closure: &super::closure::Closure,
) -> Result<Calibrated> {
    ensure!(
        [16, 32, 64].contains(&rows),
        "tree calibration requires budget16/32/64"
    );
    load_mask(
        d,
        ops,
        c,
        dir,
        MultiShape {
            rows,
            causal: false,
            capacity,
        },
        true,
        closure,
    )
}

pub(super) fn cache_path(
    dir: &Path,
    key: &CacheKey,
    bucket: usize,
    rows: usize,
    causal: bool,
    tree: bool,
) -> PathBuf {
    dir.join(format!(
        "multi-long-v2-b{bucket}-m{rows}-{}-{}-{}-{}.json",
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
    ))
}

fn load_mask(
    d: &Device,
    ops: &Ops,
    c: &Config,
    dir: &Path,
    shape: MultiShape,
    tree: bool,
    closure: &super::closure::Closure,
) -> Result<Calibrated> {
    let rows = shape.rows;
    let causal = shape.causal;
    let bucket = attention_bucket(shape.capacity)?;
    ensure!(
        (1..=64).contains(&rows),
        "multi attention M must be within1..64"
    );
    let key = calibrate::key(d)?;
    let path = cache_path(dir, &key, bucket, rows, causal, tree);
    let (cache, hit) = if closure.access(&path)? {
        let cache: Cache = serde_json::from_slice(&std::fs::read(&path)?)?;
        check(&cache, &key, rows, causal, tree, bucket, budget(d, c)?)?;
        (cache, true)
    } else {
        eprintln!(
            "{}",
            serde_json::json!({"attention_calibration_cache":{"schema":2,"lengths":LENGTHS,"bucket":bucket,"path":path,"cached":false,"action":"measure requested bucket"}})
        );
        let cache = measure(d, ops, c, key, shape, tree, closure)?;
        check(
            &cache,
            &cache.key,
            rows,
            causal,
            tree,
            bucket,
            budget(d, c)?,
        )?;
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
    shape: MultiShape,
    tree: bool,
    closure: &super::closure::Closure,
) -> Result<Cache> {
    closure.measurement()?;
    let rows = shape.rows;
    let causal = shape.causal;
    let selected = attention_bucket(shape.capacity)?;
    let start = Instant::now();
    let trial_ms = if causal { 10 } else { 30 };
    let s = &d.stream;
    let capacities = LENGTHS.map(|n| calibration_capacity(n, rows));
    let q_values = random(rows * c.qkv_dim(), 4);
    let zero_q = vec![bf16::ZERO; q_values.len()];
    let mut q = s.clone_htod(&q_values)?;
    let k = s.clone_htod(&random(capacities[selected] * c.kv_dim(), 1))?;
    let v_values = random(capacities[selected] * c.kv_dim(), 2);
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
    let mut winner = None;
    for (bucket, &l) in LENGTHS.iter().enumerate().filter(|(b, _)| *b == selected) {
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
        for plan in candidates(bucket, rows, causal, tree, budget(d, c)?) {
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
            while timer.elapsed() < Duration::from_millis(trial_ms) {
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
                winner = Some(plan);
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
    Ok(Cache {
        schema: 2,
        lengths: LENGTHS,
        key,
        rows,
        causal,
        tree,
        winner: winner.context("no measured multi attention candidate")?,
        bucket: selected,
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

#[cfg(test)]
mod long_tests {
    use super::*;

    #[test]
    fn measured_bucket_has_complete_candidates_and_valid_launches() -> Result<()> {
        let budget = Budget {
            kv_bytes_per_token: 2 * 128 * std::mem::size_of::<bf16>() * 2,
            l2_bytes: 4 << 20,
            sms: 48,
            phase1_resident: [0, 2, 1],
            kv_heads: 2,
        };
        for (rows, causal, tree) in [
            (7, false, false),
            (8, true, false),
            (16, false, true),
            (64, false, true),
        ] {
            for (bucket, &length) in LENGTHS.iter().enumerate() {
                let key = CacheKey {
                    device_uuid: "test".into(),
                    driver_version: "test".into(),
                    binary_version: "test".into(),
                };
                let mut trials = Vec::new();
                for (i, plan) in candidates(bucket, rows, causal, tree, budget)
                    .into_iter()
                    .enumerate()
                {
                    let capacity = calibration_capacity(length, rows);
                    assert!(capacity <= 131072);
                    assert!(
                        (2 * capacity.div_ceil(plan.launch.chunk)
                            + plan.launch.merge_threads as usize)
                            * 4
                            <= 49152
                    );
                    let median_us = (i + 1) as f64;
                    trials.push(Trial {
                        bucket,
                        plan,
                        samples: 1,
                        median_us,
                        p90_us: median_us,
                        p10_us: median_us,
                        max_abs_error: 0.0,
                    });
                }
                let mut cache = Cache {
                    key,
                    rows,
                    causal,
                    tree,
                    winner: trials[0].plan,
                    bucket,
                    schema: 2,
                    lengths: LENGTHS,
                    trials,
                    wall_ms: 1.0,
                };
                check(&cache, &cache.key, rows, causal, tree, bucket, budget)?;
                let winner = cache.winner;
                cache.winner = cache.trials[1].plan;
                assert!(check(&cache, &cache.key, rows, causal, tree, bucket, budget).is_err());
                cache.winner = winner;
                assert!(
                    check(
                        &cache,
                        &cache.key,
                        rows,
                        causal,
                        tree,
                        (bucket + 1) % LENGTHS.len(),
                        budget,
                    )
                    .is_err()
                );
                cache.schema = 1;
                assert!(check(&cache, &cache.key, rows, causal, tree, bucket, budget).is_err());
                cache.schema = 2;
                cache.trials.pop();
                assert!(check(&cache, &cache.key, rows, causal, tree, bucket, budget).is_err());
            }
        }
        Ok(())
    }
}
