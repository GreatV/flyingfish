use super::*;
use crate::backend::cuda::{Device, moe::testing};
use safetensors::{Dtype, SafeTensors};

fn round_bf16(value: f64) -> bf16 {
    assert!(value.is_finite());
    let initial = bf16::from_f32(value as f32);
    let mut best = initial;
    let mut distance = (value - best.to_f64()).abs();
    for bits in [
        initial.to_bits().wrapping_sub(1),
        initial.to_bits().wrapping_add(1),
    ] {
        let candidate = bf16::from_bits(bits);
        let delta = (value - candidate.to_f64()).abs();
        if delta < distance || (delta == distance && bits & 1 == 0) {
            best = candidate;
            distance = delta;
        }
    }
    best
}

fn pattern(count: usize, period: usize, multiplier: usize, divisor: f32) -> Vec<bf16> {
    (0..count)
        .map(|i| {
            bf16::from_f32(
                ((i * multiplier + 13) % period) as f32 / divisor - (period / 2) as f32 / divisor,
            )
        })
        .collect()
}

struct Fixture {
    keys: usize,
    capacity: usize,
    rows: usize,
    first: usize,
    packed: Vec<bf16>,
    q: Vec<bf16>,
    weights: Vec<bf16>,
    ape: Vec<bf16>,
}

impl Fixture {
    fn new(keys: usize, first: usize, invalid: bool) -> Self {
        let rows = keys.min(5);
        let k = pattern(keys * 128, 1009, 37, 256.0);
        let gate = pattern(keys * 128, 509, 17, 128.0);
        let mut packed = vec![bf16::ZERO; keys * 257];
        for token in 0..keys {
            packed[token * 257..token * 257 + 128]
                .copy_from_slice(&k[token * 128..(token + 1) * 128]);
            packed[token * 257 + 128..token * 257 + 256]
                .copy_from_slice(&gate[token * 128..(token + 1) * 128]);
            packed[token * 257 + 256] = bf16::from_f32(f32::from(!invalid && token >= first));
        }
        Self {
            keys,
            capacity: keys,
            rows,
            first: if invalid { keys } else { first },
            packed,
            q: pattern(rows * 32 * 128, 1031, 29, 512.0),
            weights: pattern(rows * 32, 127, 19, 128.0)
                .iter()
                .map(|v| bf16::from_f32(v.to_f32().abs() + 0.125))
                .collect(),
            ape: pattern(4 * 128, 509, 17, 256.0),
        }
    }

    fn valid(&self, token: usize) -> bool {
        token >= self.keys.saturating_sub(self.capacity)
            && token < self.keys
            && self.packed[token * 257 + 256] != bf16::ZERO
    }

    fn meta(&self, prefix: usize, rows: usize) -> DsaMeta {
        DsaMeta {
            prefix: prefix as i32,
            valid_rows: rows as i32,
            key_count: (prefix + rows) as i32,
            capacity: self.capacity as i32,
            first_key: self.first.min(prefix + rows) as i32,
            reserved: [0; 3],
        }
    }
}

struct IndexReference {
    pools: Vec<(usize, Vec<bf16>)>,
    scores: Vec<f64>,
    selected: Vec<i32>,
    indices: Vec<i32>,
    margin: f64,
}

fn index_reference(f: &Fixture, pool_end: usize) -> IndexReference {
    let mut pools = Vec::new();
    for pool in 0..(f.keys - f.first).div_ceil(4) {
        let begin = f.first + pool * 4;
        if !(begin..begin + 4).all(|i| f.valid(i)) {
            continue;
        }
        let mut key = Vec::with_capacity(128);
        for channel in 0..128 {
            let logits: Vec<f64> = (0..4)
                .map(|j| {
                    f.packed[(begin + j) * 257 + 128 + channel].to_f64()
                        + f.ape[j * 128 + channel].to_f64()
                })
                .collect();
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|x| (x - maximum).exp()).collect();
            let total: f64 = exp.iter().sum();
            let value: f64 = exp
                .iter()
                .enumerate()
                .map(|(j, &p)| {
                    let probability = round_bf16(p / total).to_f64();
                    round_bf16(probability * f.packed[(begin + j) * 257 + channel].to_f64())
                        .to_f64()
                })
                .sum();
            key.push(round_bf16(value));
        }
        pools.push((pool, key));
    }
    let mut scores = Vec::new();
    let mut selected = vec![-1; f.rows * 512];
    let mut indices = vec![-1; f.rows * 2051];
    let mut margin = f64::INFINITY;
    let width = pools.len().min(512) * 4;
    for row in 0..f.rows {
        let position = f.keys - f.rows + row;
        let mut candidates = Vec::new();
        for (pool, key) in &pools {
            let score: f64 = (0..32)
                .map(|head| {
                    let dot: f64 = key
                        .iter()
                        .enumerate()
                        .map(|(d, k)| f.q[(row * 32 + head) * 128 + d].to_f64() * k.to_f64())
                        .sum();
                    (dot / 128.0f64.sqrt()).max(0.0) * f.weights[row * 32 + head].to_f64()
                        / 32.0f64.sqrt()
                })
                .sum();
            scores.push(score);
            if f.valid(position) && f.first + pool * 4 + pool_end <= position {
                candidates.push((*pool, score));
            }
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if candidates.len() > 512 {
            margin = margin.min(candidates[511].1 - candidates[512].1);
        }
        for (slot, &(pool, _)) in candidates.iter().take(512).enumerate() {
            selected[row * 512 + slot] = pool as i32;
            for j in 0..4 {
                indices[row * 2051 + slot * 4 + j] = (f.first + pool * 4 + j) as i32;
            }
        }
        if f.valid(position) {
            let visible = (0..=position)
                .filter(|&i| f.packed[i * 257 + 256] != bf16::ZERO)
                .count();
            let tail = visible % 4;
            let begin = f.first + visible - tail;
            for j in 0..tail {
                if begin + j <= position && f.valid(begin + j) {
                    indices[row * 2051 + width + j] = (begin + j) as i32;
                }
            }
        }
    }
    IndexReference {
        pools,
        scores,
        selected,
        indices,
        margin,
    }
}

fn stored_bf16(source: &SafeTensors<'_>, name: &str) -> Result<Vec<bf16>> {
    let tensor = source.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::BF16, "Runner FAIL {name} dtype");
    Ok(tensor
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])))
        .collect())
}

fn stored_i32(source: &SafeTensors<'_>, name: &str) -> Result<Vec<i32>> {
    let tensor = source.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::I32, "Runner FAIL {name} dtype");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| i32::from_le_bytes(*x))
        .collect())
}

#[test]
#[ignore = "CPU-only pinned model-definition fixtures; FF_DSA_REFERENCE_DIR"]
fn dsa_definition_reference_cpu() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("FF_DSA_REFERENCE_DIR")?);
    let mut defects = 0;
    let mut max_score = 0.0f64;
    for (keys, first, invalid) in [
        (3, 0, false),
        (4, 0, false),
        (2047, 0, false),
        (2048, 0, false),
        (2051, 0, false),
        (4096, 0, false),
        (7, 3, false),
        (2051, 3, false),
        (4, 0, true),
    ] {
        let f = Fixture::new(keys, first, invalid);
        let name = format!("n{keys}-p{first}-invalid{}", usize::from(invalid));
        let bytes = std::fs::read(root.join(format!("{name}.safetensors")))?;
        let stored = SafeTensors::deserialize(&bytes)?;
        ensure!(
            stored_bf16(&stored, "packed")? == f.packed,
            "Runner FAIL fixture packed {name}"
        );
        ensure!(
            stored_bf16(&stored, "index_q")? == f.q,
            "Runner FAIL fixture Q {name}"
        );
        let expected = index_reference(&f, 3);
        if keys == 4 && !invalid {
            let premature = index_reference(&f, 2);
            let changed = expected
                .indices
                .iter()
                .zip(premature.indices)
                .filter(|(a, b)| **a != *b)
                .count();
            println!("DSA_POOL_END_CONTROL premature_end=2 actual_end=3 changed_raw_ids={changed}");
            ensure!(changed > 0, "Runner FAIL pool-end control is insensitive");
        }
        let pools: Vec<_> = expected
            .pools
            .iter()
            .flat_map(|(_, key)| key)
            .copied()
            .collect();
        ensure!(
            stored.tensor("pool_keys")?.shape() == [expected.pools.len(), 128],
            "Runner FAIL pool shape"
        );
        ensure!(
            stored.tensor("indices")?.shape() == [f.rows, 2051],
            "Runner FAIL index shape"
        );
        let pool_errors = pools
            .iter()
            .zip(stored_bf16(&stored, "pool_keys")?)
            .filter(|(a, b)| **a != *b)
            .count();
        let index_errors = expected
            .indices
            .iter()
            .zip(stored_i32(&stored, "indices")?)
            .filter(|(a, b)| **a != *b)
            .count();
        let scores = stored.tensor("scores")?;
        ensure!(scores.dtype() == Dtype::F32, "Runner FAIL score dtype");
        ensure!(
            scores.shape() == [f.rows, expected.pools.len()],
            "Runner FAIL score shape"
        );
        for (want, bytes) in expected.scores.iter().zip(scores.data().as_chunks::<4>().0) {
            let value = f32::from_le_bytes(*bytes);
            max_score = max_score.max((want - f64::from(value)).abs());
        }
        defects += pool_errors + index_errors;
        println!(
            "DSA_REFERENCE case={name} pool_bits={pool_errors} raw_ids={index_errors} margin={}",
            expected.margin
        );
    }
    println!("DSA_REFERENCE_SUMMARY defects={defects} max_score_abs={max_score:e} oracle=fp64");
    ensure!(
        defects == 0,
        "DEFECT DSA independent reference count={defects}"
    );
    Ok(())
}

fn attention_error(actual: &[bf16], expected: &[bf16]) -> Result<(f64, f64)> {
    ensure!(
        actual.len() == expected.len(),
        "Runner FAIL attention extent"
    );
    let mut maximum = 0.0f64;
    let mut square = 0.0;
    for (a, b) in actual.iter().zip(expected) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "Runner FAIL nonfinite attention"
        );
        let delta = a.to_f64() - b.to_f64();
        maximum = maximum.max(delta.abs());
        square += delta * delta;
    }
    Ok((maximum, (square / actual.len() as f64).sqrt()))
}

fn projected_reference(
    q: &[bf16],
    latent: &[bf16],
    weights: &[bf16],
    indices: &[i32],
) -> Vec<bf16> {
    let keys = latent.len() / 512;
    let rows = q.len() / (64 * 256);
    let mut output = vec![bf16::ZERO; q.len()];
    for head in 0..64 {
        let mut kv = vec![0.0f64; keys * 512];
        for token in 0..keys {
            for channel in 0..512 {
                let value: f64 = (0..512)
                    .map(|d| {
                        latent[token * 512 + d].to_f64()
                            * weights[(head * 512 + channel) * 512 + d].to_f64()
                    })
                    .sum();
                kv[token * 512 + channel] = round_bf16(value).to_f64();
            }
        }
        for row in 0..rows {
            let mut scores = Vec::new();
            for &id in &indices[row * 2051..(row + 1) * 2051] {
                if id >= 0 {
                    let score: f64 = (0..256)
                        .map(|d| {
                            q[(row * 64 + head) * 256 + d].to_f64() * kv[id as usize * 512 + d]
                        })
                        .sum();
                    scores.push((id as usize, score / 16.0));
                }
            }
            let maximum = scores.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = scores.iter().map(|x| (x.1 - maximum).exp()).sum();
            if total > 0.0 {
                for d in 0..256 {
                    let value: f64 = scores
                        .iter()
                        .map(|&(id, score)| (score - maximum).exp() * kv[id * 512 + 256 + d])
                        .sum();
                    output[(row * 64 + head) * 256 + d] = round_bf16(value / total);
                }
            }
        }
    }
    output
}

#[test]
#[ignore = "CPU-only pinned model-definition fixtures; FF_DSA_REFERENCE_DIR"]
fn dsa_attention_reference_cpu() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("FF_DSA_REFERENCE_DIR")?);
    let bytes = std::fs::read(root.join("weights.safetensors"))?;
    let weights = stored_bf16(&SafeTensors::deserialize(&bytes)?, "kv_b")?;
    for name in [
        "n3-p0-invalid0",
        "n4-p0-invalid0",
        "n7-p3-invalid0",
        "n4-p0-invalid1",
    ] {
        let bytes = std::fs::read(root.join(format!("{name}.safetensors")))?;
        let stored = SafeTensors::deserialize(&bytes)?;
        let actual = projected_reference(
            &stored_bf16(&stored, "attention_q")?,
            &stored_bf16(&stored, "latent")?,
            &weights,
            &stored_i32(&stored, "indices")?,
        );
        let (maximum, rms) = attention_error(&actual, &stored_bf16(&stored, "attention")?)?;
        println!("DSA_ATTENTION_REFERENCE case={name} max_abs={maximum:e} rms={rms:e} oracle=fp64");
        ensure!(
            maximum <= 3.0517578125e-5 && rms <= 2.0e-7,
            "DEFECT attention reference"
        );
    }
    Ok(())
}

#[test]
#[ignore = "exclusive GPU; FF_Q38_GPU_WINDOW=granted; FF_DSA_REFERENCE_DIR"]
fn dsa_attention_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root = std::path::PathBuf::from(std::env::var("FF_DSA_REFERENCE_DIR")?);
    let device = Device::new(0)?;
    let s = &device.stream;
    let bytes = std::fs::read(root.join("weights.safetensors"))?;
    let weight_values = stored_bf16(&SafeTensors::deserialize(&bytes)?, "kv_b")?;
    ensure!(weight_values.len() == 32768 * 512, "Runner FAIL kv_b shape");
    let weights = s.clone_htod(&weight_values)?;
    let kv_b = testing::dense(&weights, s, 32768, 512)?;
    let mut defects = 0;
    for (keys, first, invalid) in [
        (3, 0, false),
        (4, 0, false),
        (2047, 0, false),
        (2048, 0, false),
        (2051, 0, false),
        (4096, 0, false),
        (7, 3, false),
        (2051, 3, false),
        (4, 0, true),
    ] {
        let f = Fixture::new(keys, first, invalid);
        let name = format!("n{keys}-p{first}-invalid{}", usize::from(invalid));
        let bytes = std::fs::read(root.join(format!("{name}.safetensors")))?;
        let stored = SafeTensors::deserialize(&bytes)?;
        let expected = stored_bf16(&stored, "attention")?;
        ensure!(
            expected.len() == f.rows * 64 * 256,
            "Runner FAIL attention fixture shape"
        );
        let q = s.clone_htod(&stored_bf16(&stored, "attention_q")?)?;
        let latent = s.clone_htod(&stored_bf16(&stored, "latent")?)?;
        let packed = s.clone_htod(&f.packed)?;
        let ids = stored_i32(&stored, "indices")?;
        ensure!(
            ids == index_reference(&f, 3).indices,
            "Runner FAIL reference selected IDs"
        );
        let indices = s.clone_htod(&ids)?;
        let output = s.alloc_zeros::<bf16>(expected.len())?;
        let out = testing::mut_view(&output, s, [f.rows as u32, 64, 256], [16384, 256, 1])?;
        let spec = DsaSpec::glm53();
        let plan = DsaPlan::new(&device.ctx, &device.info, spec, keys as u32, keys as u32)?;
        let mut state = DsaState::new(spec, keys as u32, s)?;
        state.set_meta(s, f.meta(0, keys))?;
        plan.append(
            &state,
            &DsaAppendInput {
                latent: testing::dense(&latent, s, keys as u32, 512)?,
                packed: testing::dense(&packed, s, keys as u32, 257)?,
            },
            s,
        )?;
        state.set_meta(s, f.meta(keys - f.rows, f.rows))?;
        let bound = plan.for_rows(f.rows as u32)?;
        let input = DsaAttentionInput {
            q: testing::view(&q, s, [f.rows as u32, 64, 256], [16384, 256, 1])?,
            kv_b: Some(kv_b),
            indices: testing::dense(&indices, s, f.rows as u32, 2051)?,
            sinks: None,
            out: &out,
        };
        device.finish_upload()?;
        for graph in [false, true] {
            if graph {
                s.synchronize()?;
                testing::capture(s, || bound.attention(&state, &input, s))?.launch()?;
            } else {
                bound.attention(&state, &input, s)?;
            }
            let (maximum, rms) = attention_error(&s.clone_dtoh(&output)?, &expected)?;
            let pass = maximum <= 3.0517578125e-5 && rms <= 2.0e-7;
            defects += usize::from(!pass);
            println!(
                "{} DSA_ATTENTION case={name} graph={graph} max_abs={maximum:e} rms={rms:e}",
                if pass { "PASS" } else { "DEFECT" }
            );
        }
        if keys == 4 && !invalid {
            let mut wrong = ids.clone();
            wrong[0] = 1;
            let wrong = s.clone_htod(&wrong)?;
            let wrong_input = DsaAttentionInput {
                indices: testing::dense(&wrong, s, f.rows as u32, 2051)?,
                ..input
            };
            bound.attention(&state, &wrong_input, s)?;
            let (maximum, _) = attention_error(&s.clone_dtoh(&output)?, &expected)?;
            println!("DSA_ATTENTION_WRONG_ID max_abs={maximum:e}");
            defects += usize::from(maximum <= 3.0517578125e-5);
        }
    }
    println!("DSA_ATTENTION_SUMMARY cases=9 modes=2 defects={defects}");
    ensure!(defects == 0, "DEFECT sparse attention count={defects}");
    Ok(())
}

fn index_gpu_case(device: &Device, f: &Fixture) -> Result<usize> {
    let s = &device.stream;
    let spec = DsaSpec::glm53();
    let plan = DsaPlan::new(
        &device.ctx,
        &device.info,
        spec,
        f.capacity as u32,
        f.capacity as u32,
    )?;
    let mut state = DsaState::new(spec, f.capacity as u32, s)?;
    let latent = s.alloc_zeros::<bf16>(f.keys * 512)?;
    let q = s.clone_htod(&f.q)?;
    let weights = s.clone_htod(&f.weights)?;
    let ape = s.clone_htod(&f.ape)?;
    let input = DsaIndexInput {
        q: testing::view(&q, s, [f.rows as u32, 32, 128], [4096, 128, 1])?,
        weights: testing::dense(&weights, s, f.rows as u32, 32)?,
        ape: testing::dense(&ape, s, 4, 128)?,
    };
    let mut chunks = Vec::new();
    let mut prefix = 0;
    while prefix < f.keys - 1 {
        let rows = (f.keys - 1 - prefix).min(f.capacity.saturating_sub(2).max(1));
        chunks.push((prefix, rows));
        prefix += rows;
    }
    chunks.push((f.keys - 1, 1));
    for (prefix, rows) in chunks {
        let packed = s.clone_htod(&f.packed[prefix * 257..(prefix + rows) * 257])?;
        state.set_meta(s, f.meta(prefix, rows))?;
        let bound = plan.for_rows(rows as u32)?;
        let append = DsaAppendInput {
            latent: super::super::contracts::view(&latent, s, 0, [rows as u32, 512], [512, 1])?,
            packed: super::super::contracts::view(&packed, s, 0, [rows as u32, 257], [257, 1])?,
        };
        bound.append(&state, &append, s)?;
        bound.pool(&state, &input, s)?;
    }
    state.set_meta(s, f.meta(f.keys - f.rows, f.rows))?;
    let bound = plan.for_rows(f.rows as u32)?;
    let selected = s.alloc_zeros::<i32>(f.rows * 512)?;
    let scores = s.alloc_zeros::<f32>(f.rows * 512)?;
    let margins = s.alloc_zeros::<f32>(f.rows)?;
    let indices = s.alloc_zeros::<i32>(f.rows * 2051)?;
    let output = DsaIndexOutput {
        selected: testing::dense_mut(&selected, s, f.rows as u32, 512)?,
        scores: testing::dense_mut(&scores, s, f.rows as u32, 512)?,
        margins: testing::mut_view(&margins, s, [f.rows as u32], [1])?,
        indices: testing::dense_mut(&indices, s, f.rows as u32, 2051)?,
    };
    let expected = index_reference(f, 3);
    let mut defects = 0;
    device.finish_upload()?;
    for graph in [false, true] {
        if graph {
            s.synchronize()?;
            testing::capture(s, || bound.index(&state, &input, &output, s))?.launch()?;
        } else {
            bound.index(&state, &input, &output, s)?;
        }
        s.synchronize()?;
        let actual_pools = s.clone_dtoh(&state.pool_keys)?;
        let actual_ids = s.clone_dtoh(&state.pool_ids)?;
        let mut pool_bits = 0;
        for (pool, key) in &expected.pools {
            let slot = pool % actual_ids.len();
            pool_bits += usize::from(actual_ids[slot] != *pool as i32);
            pool_bits += key
                .iter()
                .zip(&actual_pools[slot * 128..(slot + 1) * 128])
                .filter(|(a, b)| a != b)
                .count();
        }
        let actual_selected = s.clone_dtoh(&selected)?;
        let actual_indices = s.clone_dtoh(&indices)?;
        let actual_scores = s.clone_dtoh(&scores)?;
        let mut score_errors = 0;
        let mut max_score_abs = 0.0f64;
        for (index, &pool) in actual_selected.iter().enumerate() {
            if pool < 0 {
                score_errors += usize::from(actual_scores[index] != f32::NEG_INFINITY);
            } else if let Some(at) = expected
                .pools
                .iter()
                .position(|(id, _)| *id == pool as usize)
            {
                let want = expected.scores[index / 512 * expected.pools.len() + at];
                let error = (f64::from(actual_scores[index]) - want).abs();
                max_score_abs = max_score_abs.max(error);
                score_errors += usize::from(!error.is_finite() || error > 9.5367431640625e-7);
            } else {
                score_errors += 1;
            }
        }
        let selected_errors = actual_selected
            .iter()
            .zip(&expected.selected)
            .filter(|(a, b)| a != b)
            .count();
        let index_errors = actual_indices
            .iter()
            .zip(&expected.indices)
            .filter(|(a, b)| a != b)
            .count();
        defects += pool_bits + selected_errors + index_errors + score_errors;
        println!(
            "{} DSA_INDEX keys={} capacity={} rows={} first={} graph={graph} pool_bits={pool_bits} selected_ids={selected_errors} raw_ids={index_errors} score_errors={score_errors} max_score_abs={max_score_abs:e} margin={}",
            if pool_bits + selected_errors + index_errors + score_errors == 0 {
                "PASS"
            } else {
                "DEFECT"
            },
            f.keys,
            f.capacity,
            f.rows,
            f.first,
            expected.margin
        );
    }
    Ok(defects)
}

#[test]
#[ignore = "exclusive GPU; FF_Q38_GPU_WINDOW=granted"]
fn dsa_index_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let mut defects = 0;
    for (keys, first, invalid) in [
        (3, 0, false),
        (4, 0, false),
        (2047, 0, false),
        (2048, 0, false),
        (2051, 0, false),
        (4096, 0, false),
        (7, 3, false),
        (2051, 3, false),
        (4, 0, true),
    ] {
        defects += index_gpu_case(&device, &Fixture::new(keys, first, invalid))?;
    }
    println!("DSA_INDEX_SUMMARY cases=9 modes=2 defects={defects} oracle=fp64");
    ensure!(defects == 0, "DEFECT DSA index count={defects}");
    Ok(())
}

#[derive(PartialEq)]
struct Snapshot {
    values: [Vec<bf16>; 3],
    ids: [Vec<i32>; 3],
}

fn snapshot(state: &DsaState, s: &Arc<CudaStream>) -> Result<Snapshot> {
    Ok(Snapshot {
        values: [
            s.clone_dtoh(&state.latent)?,
            s.clone_dtoh(&state.packed)?,
            s.clone_dtoh(&state.pool_keys)?,
        ],
        ids: [
            s.clone_dtoh(&state.token_ids)?,
            s.clone_dtoh(&state.pool_ids)?,
            s.clone_dtoh(&state.meta)?,
        ],
    })
}

struct Chunk {
    meta: CudaSlice<i32>,
    latent: CudaSlice<bf16>,
    packed: CudaSlice<bf16>,
}

impl Chunk {
    fn new(f: &Fixture, prefix: usize, rows: usize, s: &Arc<CudaStream>) -> Result<Self> {
        let meta = f.meta(prefix, rows);
        let latent = pattern(f.keys * 512, 1021, 41, 1024.0);
        Ok(Self {
            meta: s.clone_htod(&[
                meta.prefix,
                meta.valid_rows,
                meta.key_count,
                meta.capacity,
                meta.first_key,
                0,
                0,
                0,
            ])?,
            latent: s.clone_htod(&latent[prefix * 512..(prefix + rows) * 512])?,
            packed: s.clone_htod(&f.packed[prefix * 257..(prefix + rows) * 257])?,
        })
    }

    fn run(
        &self,
        plan: &DsaPlan,
        state: &mut DsaState,
        input: &DsaIndexInput,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        let rows = (self.latent.len() / 512) as u32;
        let bound = plan.for_rows(rows)?;
        s.memcpy_dtod(&self.meta, &mut state.meta)?;
        bound.append(
            state,
            &DsaAppendInput {
                latent: testing::dense(&self.latent, s, rows, 512)?,
                packed: testing::dense(&self.packed, s, rows, 257)?,
            },
            s,
        )?;
        bound.pool(state, input, s)
    }
}

#[test]
#[ignore = "exclusive GPU; FF_Q38_GPU_WINDOW=granted"]
fn dsa_state_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let spec = DsaSpec::glm53();
    let mut defects = 0;
    let plan_overflow = DsaPlan::new(&device.ctx, &device.info, spec, 11, 12).is_ok();
    let checkpoint_overflow = DsaCheckpoint::new(spec, 11, 12, s).is_ok();
    defects += usize::from(plan_overflow) + usize::from(checkpoint_overflow);
    println!(
        "DSA_CAPACITY plan_overflow={plan_overflow} checkpoint_overflow={checkpoint_overflow}"
    );
    for base in [6, 17] {
        for graph in [false, true] {
            let mut f = Fixture::new(base + 5, 3, false);
            f.capacity = 11;
            let plan = DsaPlan::new(&device.ctx, &device.info, spec, 11, 11)?;
            let mut state = DsaState::new(spec, 11, s)?;
            let q = s.alloc_zeros::<bf16>(4096)?;
            let weights = s.alloc_zeros::<bf16>(32)?;
            let ape = s.clone_htod(&f.ape)?;
            let input = DsaIndexInput {
                q: testing::view(&q, s, [1, 32, 128], [4096, 128, 1])?,
                weights: testing::dense(&weights, s, 1, 32)?,
                ape: testing::dense(&ape, s, 4, 128)?,
            };
            let mut prefix = 0;
            while prefix < base {
                let rows = (base - prefix).min(5);
                Chunk::new(&f, prefix, rows, s)?.run(&plan, &mut state, &input, s)?;
                prefix += rows;
            }
            let initial = snapshot(&state, s)?;
            let first = Chunk::new(&f, base, 3, s)?;
            let second = Chunk::new(&f, base + 3, 2, s)?;
            let mut outer = DsaCheckpoint::new(spec, 11, 5, s)?;
            let mut inner = DsaCheckpoint::new(spec, 11, 2, s)?;
            device.finish_upload()?;
            let mut run = || -> Result<()> {
                outer.save(&plan, &state, s)?;
                first.run(&plan, &mut state, &input, s)?;
                let nested = if graph {
                    None
                } else {
                    Some(snapshot(&state, s)?)
                };
                inner.save(&plan, &state, s)?;
                second.run(&plan, &mut state, &input, s)?;
                inner.restore(&plan, &state, s)?;
                if let Some(nested) = nested {
                    let pass = snapshot(&state, s)? == nested;
                    println!("DSA_CHECKPOINT inner base={base} equal={pass}");
                    defects += usize::from(!pass);
                }
                outer.restore(&plan, &state, s)?;
                Ok(())
            };
            if graph {
                testing::capture(s, run)?.launch()?;
            } else {
                run()?;
            }
            let pass = snapshot(&state, s)? == initial;
            println!("DSA_CHECKPOINT outer base={base} graph={graph} equal={pass}");
            defects += usize::from(!pass);
            state.reset(s)?;
            let restarted = Chunk::new(&f, 0, 5, s)?;
            restarted.run(&plan, &mut state, &input, s)?;
            let mut fresh = DsaState::new(spec, 11, s)?;
            restarted.run(&plan, &mut fresh, &input, s)?;
            let reset = snapshot(&state, s)?;
            let fresh = snapshot(&fresh, s)?;
            let pass = reset.values[0][..5 * 512] == fresh.values[0][..5 * 512]
                && reset.values[1][..5 * 257] == fresh.values[1][..5 * 257]
                && reset.values[2][..128] == fresh.values[2][..128]
                && reset.ids[0][..5] == fresh.ids[0][..5]
                && reset.ids[2] == fresh.ids[2];
            println!("DSA_RESET base={base} graph={graph} live_equal={pass}");
            defects += usize::from(!pass);
        }
    }
    for (keys, capacity, first) in [(8, 7, 0), (19, 11, 3), (31, 13, 3)] {
        let mut f = Fixture::new(keys, first, false);
        f.capacity = capacity;
        defects += index_gpu_case(&device, &f)?;
    }
    println!("DSA_STATE_SUMMARY defects={defects}");
    ensure!(defects == 0, "DEFECT DSA state count={defects}");
    Ok(())
}

fn direct_spec() -> DsaSpec {
    DsaSpec {
        kv_form: DsaKvForm::Direct,
        pool: 1,
        index_topk: 512,
        qk_dim: 512,
        v_dim: 512,
        ..DsaSpec::glm53()
    }
}

fn direct_reference(q: &[bf16], kv: &[bf16], sinks: &[f32], ids: &[i32]) -> Vec<bf16> {
    let rows = q.len() / (64 * 512);
    let mut output = vec![bf16::ZERO; q.len()];
    for row in 0..rows {
        for head in 0..64 {
            let mut maximum = -1e30f64;
            let mut total = 0.0f64;
            let mut accumulated = [0.0f64; 512];
            for block in ids[row * 640..(row + 1) * 640].chunks(64) {
                let mut scores = [f64::NEG_INFINITY; 64];
                for (slot, &id) in block.iter().enumerate() {
                    if id >= 0 {
                        scores[slot] = (0..512)
                            .map(|d| {
                                q[(row * 64 + head) * 512 + d].to_f64()
                                    * kv[id as usize * 512 + d].to_f64()
                            })
                            .sum::<f64>()
                            / 512.0f64.sqrt();
                    }
                }
                let next = scores.iter().copied().fold(maximum, f64::max);
                let scale = (maximum - next).exp();
                total *= scale;
                for value in &mut accumulated {
                    *value *= scale;
                }
                for (slot, &id) in block.iter().enumerate() {
                    if id >= 0 {
                        let probability = (scores[slot] - next).exp();
                        total += probability;
                        let probability = round_bf16(probability).to_f64();
                        for (d, value) in accumulated.iter_mut().enumerate() {
                            *value += probability * kv[id as usize * 512 + d].to_f64();
                        }
                    }
                }
                maximum = next;
            }
            total += (f64::from(sinks[head]) - maximum).exp();
            for (d, value) in accumulated.into_iter().enumerate() {
                output[(row * 64 + head) * 512 + d] = round_bf16(value / total);
            }
        }
    }
    output
}

fn stored_f32(source: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = source.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::F32, "Runner FAIL {name} dtype");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect())
}

#[test]
#[ignore = "CPU-only direct attention reference; FF_DSA_REFERENCE_DIR"]
fn dsa_direct_reference_cpu() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("FF_DSA_REFERENCE_DIR")?);
    for (keys, invalid) in [
        (3, false),
        (4, false),
        (127, false),
        (128, false),
        (129, false),
        (4096, false),
        (4, true),
    ] {
        let name = format!("n{keys}-invalid{}", usize::from(invalid));
        let bytes = std::fs::read(root.join(format!("{name}.safetensors")))?;
        let f = SafeTensors::deserialize(&bytes)?;
        let result = direct_reference(
            &stored_bf16(&f, "q")?,
            &stored_bf16(&f, "kv")?,
            &stored_f32(&f, "sinks")?,
            &stored_i32(&f, "indices")?,
        );
        let (maximum, rms) = attention_error(&result, &stored_bf16(&f, "fp64")?)?;
        println!("DSA_DIRECT_REFERENCE case={name} max_abs={maximum:e} rms={rms:e}");
        ensure!(
            maximum <= 9.765625e-4 && rms <= 8.0e-6,
            "DEFECT direct FP64 reference"
        );
    }
    Ok(())
}

#[test]
#[ignore = "exclusive GPU; FF_Q38_GPU_WINDOW=granted; FF_DSA_REFERENCE_DIR"]
fn dsa_direct_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root = std::path::PathBuf::from(std::env::var("FF_DSA_REFERENCE_DIR")?);
    let device = Device::new(0)?;
    let s = &device.stream;
    let mut defects = 0;
    for (keys, invalid) in [
        (3, false),
        (4, false),
        (127, false),
        (128, false),
        (129, false),
        (4096, false),
        (4, true),
    ] {
        let rows = keys.min(5);
        let name = format!("n{keys}-invalid{}", usize::from(invalid));
        let bytes = std::fs::read(root.join(format!("{name}.safetensors")))?;
        let f = SafeTensors::deserialize(&bytes)?;
        ensure!(
            f.tensor("q")?.shape() == [rows, 64, 512]
                && f.tensor("kv")?.shape() == [keys, 512]
                && f.tensor("indices")?.shape() == [rows, 640],
            "Runner FAIL direct fixture shape"
        );
        let q = s.clone_htod(&stored_bf16(&f, "q")?)?;
        let kv = s.clone_htod(&stored_bf16(&f, "kv")?)?;
        let sinks = s.clone_htod(&stored_f32(&f, "sinks")?)?;
        let ids = s.clone_htod(&stored_i32(&f, "indices")?)?;
        let expected = stored_bf16(&f, "attention")?;
        let output = s.alloc_zeros::<bf16>(rows * 64 * 512)?;
        let out = testing::mut_view(&output, s, [rows as u32, 64, 512], [32768, 512, 1])?;
        let spec = direct_spec();
        let plan = DsaPlan::new(&device.ctx, &device.info, spec, keys as u32, keys as u32)?;
        let mut state = DsaState::new(spec, keys as u32, s)?;
        let packed = s.alloc_zeros::<bf16>(keys * 257)?;
        state.set_meta(
            s,
            DsaMeta {
                prefix: 0,
                valid_rows: keys as i32,
                key_count: keys as i32,
                capacity: keys as i32,
                first_key: 0,
                reserved: [0; 3],
            },
        )?;
        plan.append(
            &state,
            &DsaAppendInput {
                latent: testing::dense(&kv, s, keys as u32, 512)?,
                packed: testing::dense(&packed, s, keys as u32, 257)?,
            },
            s,
        )?;
        state.set_meta(
            s,
            DsaMeta {
                prefix: (keys - rows) as i32,
                valid_rows: rows as i32,
                key_count: keys as i32,
                capacity: keys as i32,
                first_key: 0,
                reserved: [0; 3],
            },
        )?;
        let bound = plan.for_rows(rows as u32)?;
        let input = DsaAttentionInput {
            q: testing::view(&q, s, [rows as u32, 64, 512], [32768, 512, 1])?,
            kv_b: None,
            indices: testing::dense(&ids, s, rows as u32, 640)?,
            sinks: Some(testing::view(&sinks, s, [64], [1])?),
            out: &out,
        };
        device.finish_upload()?;
        for graph in [false, true] {
            if graph {
                testing::capture(s, || bound.attention(&state, &input, s))?.launch()?;
            } else {
                bound.attention(&state, &input, s)?;
            }
            let (maximum, rms) = attention_error(&s.clone_dtoh(&output)?, &expected)?;
            let pass = maximum <= 9.765625e-4 && rms <= 8.0e-6;
            defects += usize::from(!pass);
            println!(
                "{} DSA_DIRECT case={name} graph={graph} max_abs={maximum:e} rms={rms:e}",
                if pass { "PASS" } else { "DEFECT" }
            );
        }
    }
    println!("DSA_DIRECT_SUMMARY cases=7 modes=2 defects={defects}");
    ensure!(defects == 0, "DEFECT direct attention count={defects}");
    Ok(())
}

#[test]
#[ignore = "exclusive GPU; FF_Q38_GPU_WINDOW=granted"]
fn dsa_bounded_topk_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let plan = DsaPlan::new(&device.ctx, &device.info, direct_spec(), 65536, 3)?;
    let mut defects = 0;
    for (candidates, keep) in [
        (7, 7),
        (512, 512),
        (2048, 512),
        (2048, 2048),
        (65536, 512),
        (65536, 2048),
    ] {
        let mut scores = vec![0.0f32; 3 * candidates];
        let mut ids = vec![0i32; 3 * candidates];
        let mut expected_ids = vec![-1; 3 * keep];
        let mut expected_scores = vec![f32::NEG_INFINITY; 3 * keep];
        let mut expected_margin = [f32::INFINITY; 3];
        for row in 0..3 {
            let mut sorted = Vec::new();
            for candidate in 0..candidates {
                let score = if row == 1 {
                    if candidate % 2 == 0 { 0.0 } else { -0.0 }
                } else if row == 2 && candidate % 11 == 0 {
                    f32::NEG_INFINITY
                } else {
                    ((candidate * 29 + 13) % 1031) as f32 / 128.0
                };
                let id = if row == 2 && candidate % 13 == 0 {
                    -1
                } else {
                    ((candidate * 37) % candidates) as i32
                };
                scores[row * candidates + candidate] = score;
                ids[row * candidates + candidate] = id;
                if id >= 0 && score.is_finite() {
                    sorted.push((id, score));
                }
            }
            sorted.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .expect("finite score")
                    .then(a.0.cmp(&b.0))
            });
            for (slot, &(id, score)) in sorted.iter().take(keep).enumerate() {
                expected_ids[row * keep + slot] = id;
                expected_scores[row * keep + slot] = score;
            }
            if sorted.len() > keep {
                expected_margin[row] = sorted[keep - 1].1 - sorted[keep].1;
            }
        }
        let scores = s.clone_htod(&scores)?;
        let ids = s.clone_htod(&ids)?;
        let selected = s.alloc_zeros::<i32>(3 * keep)?;
        let values = s.alloc_zeros::<f32>(3 * keep)?;
        let margins = s.alloc_zeros::<f32>(3)?;
        let input = DsaTopKInput {
            scores: testing::dense(&scores, s, 3, candidates as u32)?,
            ids: testing::dense(&ids, s, 3, candidates as u32)?,
        };
        let output = DsaTopKOutput {
            scores: testing::dense_mut(&values, s, 3, keep as u32)?,
            ids: testing::dense_mut(&selected, s, 3, keep as u32)?,
            margins: testing::mut_view(&margins, s, [3], [1])?,
        };
        device.finish_upload()?;
        for graph in [false, true] {
            if graph {
                testing::capture(s, || plan.select_scores(&input, &output, s))?.launch()?;
            } else {
                plan.select_scores(&input, &output, s)?;
            }
            let pass = s.clone_dtoh(&selected)? == expected_ids
                && s.clone_dtoh(&values)? == expected_scores
                && s.clone_dtoh(&margins)? == expected_margin;
            defects += usize::from(!pass);
            println!(
                "{} DSA_BOUNDED_TOPK candidates={candidates} keep={keep} rows=3 graph={graph} exact_ids_scores_margin={pass}",
                if pass { "PASS" } else { "DEFECT" }
            );
        }
    }
    println!("DSA_BOUNDED_TOPK_SUMMARY cases=6 modes=2 defects={defects}");
    ensure!(defects == 0, "DEFECT bounded topk count={defects}");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "exclusive GPU; metadata traps execute in isolated child processes"]
fn dsa_metadata_bounds_gpu() -> Result<()> {
    use std::io::Write;
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let Ok(case) = std::env::var("FF_DSA_GUARD_CHILD") else {
        for case in ["valid", "rows", "capacity", "checkpoint_overflow"] {
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "backend::cuda::dsa::tests::dsa_metadata_bounds_gpu",
                    "--exact",
                    "--include-ignored",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env("FF_DSA_GUARD_CHILD", case)
                .output()?;
            let log = String::from_utf8_lossy(&output.stdout);
            ensure!(
                output.status.success() && log.contains(&format!("PASS DSA_BOUND case={case}")),
                "DEFECT DSA metadata child={case} status={:?} stdout={log} stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            );
            print!("{log}");
        }
        return Ok(());
    };
    let device = Device::new(0)?;
    let s = &device.stream;
    let spec = DsaSpec::glm53();
    let plan = DsaPlan::new(&device.ctx, &device.info, spec, 4, 1)?;
    let mut state = DsaState::new(spec, 4, s)?;
    let metadata = match case.as_str() {
        "valid" => [0, 1, 1, 4, 0, 0, 0, 0],
        "rows" => [0, 2, 2, 4, 0, 0, 0, 0],
        "capacity" => [0, 1, 1, 3, 0, 0, 0, 0],
        "checkpoint_overflow" => [i32::MAX - 1, 1, i32::MAX, 4, 0, 0, 0, 0],
        _ => anyhow::bail!("unknown DSA metadata case"),
    };
    s.memcpy_htod(&metadata, &mut state.meta)?;
    let latent = s.alloc_zeros::<bf16>(512)?;
    let packed = s.alloc_zeros::<bf16>(257)?;
    if case == "checkpoint_overflow" {
        let mut saved = DsaCheckpoint::new(spec, 4, 1, s)?;
        saved.save(&plan, &state, s)?;
    } else {
        plan.append(
            &state,
            &DsaAppendInput {
                latent: testing::dense(&latent, s, 1, 512)?,
                packed: testing::dense(&packed, s, 1, 257)?,
            },
            s,
        )?;
    }
    let result = s.synchronize();
    if case == "valid" {
        result?;
    } else {
        let error = result.expect_err("DEFECT metadata outside binding did not trap");
        ensure!(
            matches!(
                error.0,
                cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
                    | cudarc::driver::sys::CUresult::CUDA_ERROR_LAUNCH_FAILED
            ),
            "Runner FAIL unrelated CUDA error {error:?}"
        );
    }
    println!("PASS DSA_BOUND case={case}");
    std::io::stdout().flush()?;
    std::process::exit(0);
}
