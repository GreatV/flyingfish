//! Offline requant: bf16 checkpoint -> groupwise affine int4
//! (s*q + b, group 64, 8x int4 per U32 low-nibble-first, the edge0 byte
//! layout). Per-group scale/bias by iterated least squares, deterministic.
//!
//! Usage: requant `<src_dir> <dst_dir> [threads]`

use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::Mmap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// 2-D `.weight` tensors quantize; norms/conv/A_log/dt_bias/vision copy.
fn quantizable(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name.ends_with(".weight")
        && shape[1].is_multiple_of(64)
        && !name.starts_with("model.visual.")
}

/// Iterated affine LSQ: q from clip-round((w-b)/s), then s,b refit on
/// (q, w) by the 2x2 normal equations. Four rounds, then a small scale
/// search around the LSQ s (clip-round is insensitive to b within a
/// round; s is the lever that moves SSE).
fn quantize_group(w: &[f32], out: &mut [u8; 64], bias_scratch: usize) -> (f32, f32) {
    let _ = bias_scratch;
    let fit = |w: &[f32], q: &[u8; 64]| -> (f32, f32) {
        let n = w.len() as f32;
        let qw: f32 = q.iter().map(|&v| v as f32).sum::<f32>() / n;
        let ww: f32 = w.iter().sum::<f32>() / n;
        let mut cov = 0.0f32;
        let mut var = 0.0f32;
        for (i, &v) in w.iter().enumerate() {
            let dq = q[i] as f32 - qw;
            cov += dq * (v - ww);
            var += dq * dq;
        }
        if var > 0.0 {
            (cov / var, ww - cov / var * qw)
        } else {
            (1.0, ww)
        }
    };
    let assign = |w: &[f32], s: f32, b: f32| -> [u8; 64] {
        let mut q = [0u8; 64];
        for (i, &v) in w.iter().enumerate() {
            q[i] = ((v - b) / s).round().clamp(0.0, 15.0) as u8;
        }
        q
    };
    let sse = |w: &[f32], q: &[u8; 64], s: f32, b: f32| -> f32 {
        w.iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = v - (s * q[i] as f32 + b);
                d * d
            })
            .sum()
    };
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in w {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let mut s = (mx - mn) / 15.0;
    // Degenerate group (all-equal weights): scale 1, bias = w keeps q at 0.
    if s <= 0.0 || s.is_nan() {
        s = 1.0;
    }
    // Score and assign every candidate with the bf16-rounded (s, b) that
    // dequant will actually use.
    let round = |v: f32| bf16::from_f32(v).to_f32();
    let mut s = round(s);
    let mut b = round(mn);
    let mut q = assign(w, s, b);
    for _ in 0..4 {
        let (ns, nb) = fit(w, &q);
        if ns > 0.0 {
            s = round(ns);
            b = round(nb);
        }
        q = assign(w, s, b);
    }
    // Local scale search around the LSQ solution.
    let mut best = (sse(w, &q, s, b), s, b, q);
    for f in [0.85f32, 0.925, 1.075, 1.15] {
        let sc = round(s * f);
        let qc = assign(w, sc, b);
        let (fc, bc) = fit(w, &qc);
        let (sc, bc) = if fc > 0.0 {
            (round(fc), round(bc))
        } else {
            (sc, b)
        };
        let qc = assign(w, sc, bc);
        let e = sse(w, &qc, sc, bc);
        if e < best.0 {
            best = (e, sc, bc, qc);
        }
    }
    *out = best.3;
    (best.1, best.2)
}

struct OutTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Returns the tensors plus (sse, sum_w2) against the stored bf16-rounded
/// (s, b) — the error dequant actually sees.
fn quantize_tensor(name: &str, shape: &[usize], bf: &[bf16]) -> Result<(Vec<OutTensor>, f64, f64)> {
    let (rows, cols) = (shape[0], shape[1]);
    let groups_per_row = cols / 64;
    let words_per_row = cols / 8;
    let mut packed = vec![0u32; rows * words_per_row];
    let mut scales = Vec::with_capacity(rows * groups_per_row);
    let mut biases = Vec::with_capacity(rows * groups_per_row);
    let mut sse = 0.0f64;
    let mut w2 = 0.0f64;
    for r in 0..rows {
        let row = &bf[r * cols..(r + 1) * cols];
        for g in 0..groups_per_row {
            let w: Vec<f32> = row[g * 64..(g + 1) * 64]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let mut q = [0u8; 64];
            let (s, b) = quantize_group(&w, &mut q, 0);
            for (j, &wv) in w.iter().enumerate() {
                let d = wv - (s * q[j] as f32 + b);
                sse += (d * d) as f64;
                w2 += (wv * wv) as f64;
            }
            scales.push(bf16::from_f32(s));
            biases.push(bf16::from_f32(b));
            for wdx in 0..8 {
                let mut word = 0u32;
                for (j, &qv) in q[wdx * 8..(wdx + 1) * 8].iter().enumerate() {
                    word |= (qv as u32) << (4 * j);
                }
                packed[r * words_per_row + g * 8 + wdx] = word;
            }
        }
    }
    let base = name.strip_suffix(".weight").unwrap_or(name);
    Ok((
        vec![
            OutTensor {
                name: format!("{base}.weight"),
                dtype: "U32",
                shape: vec![rows, words_per_row],
                data: bytemuck_cast_u32(&packed),
            },
            OutTensor {
                name: format!("{base}.scales"),
                dtype: "BF16",
                shape: vec![rows, groups_per_row],
                data: bytemuck_cast_bf16(&scales),
            },
            OutTensor {
                name: format!("{base}.biases"),
                dtype: "BF16",
                shape: vec![rows, groups_per_row],
                data: bytemuck_cast_bf16(&biases),
            },
        ],
        sse,
        w2,
    ))
}

fn bytemuck_cast_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytemuck_cast_bf16(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// Minimal safetensors writer: u64 header len + JSON + raw data.
fn write_shard(path: &Path, tensors: &[OutTensor]) -> Result<()> {
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (i, t) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let end = offset + t.data.len() as u64;
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            t.name, t.dtype, t.shape, offset, end
        ));
        offset = end;
    }
    header.push('}');
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for t in tensors {
        f.write_all(&t.data)?;
    }
    Ok(())
}

struct Job {
    shard: PathBuf,
    name: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// Absolute form of a possibly-nonexistent path (nearest existing ancestor
/// canonicalized, missing tail re-attached).
fn absolutize(p: &Path) -> Result<PathBuf> {
    let mut ancestor = p;
    let mut missing = Vec::new();
    while !ancestor.exists() {
        match ancestor.file_name() {
            Some(name) => {
                missing.push(name);
                // A bare relative name ("out") has parent "" — treat as ".".
                ancestor = match ancestor.parent() {
                    Some(p) if p.as_os_str().is_empty() => Path::new("."),
                    Some(p) => p,
                    None => break,
                };
            }
            None => break,
        }
    }
    let mut abs = ancestor.canonicalize()?;
    for name in missing.iter().rev() {
        abs = abs.join(name);
    }
    Ok(abs)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = PathBuf::from(
        args.next()
            .context("usage: requant <src> <dst> [threads]")?,
    );
    let dst = PathBuf::from(
        args.next()
            .context("usage: requant <src> <dst> [threads]")?,
    );
    let threads: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    // Checkpoints are read-only inputs. Equality is not enough: dst nested
    // inside src (e.g. requant model model/out) is still a write into the
    // checkpoint. Compare absolute ancestry — dst may not exist yet, so
    // canonicalize its nearest existing ancestor and re-attach the tail.
    let src_abs = src.canonicalize()?;
    let dst_abs = absolutize(&dst)?;
    ensure!(
        !dst_abs.starts_with(&src_abs),
        "dst {} is inside src {} (checkpoints are read-only)",
        dst.display(),
        src.display()
    );
    if dst.exists() {
        ensure!(
            dst.read_dir()?.next().is_none(),
            "dst {} exists and is not empty",
            dst.display()
        );
    }
    std::fs::create_dir_all(&dst)?;

    // Collect jobs from every shard's header.
    let mut jobs = Vec::new();
    let mut copies = Vec::new();
    let mut entries = Vec::new();
    for e in std::fs::read_dir(&src)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "safetensors") {
            entries.push(p);
        }
    }
    entries.sort();
    ensure!(
        !entries.is_empty(),
        "no safetensors under {}",
        src.display()
    );
    for shard in &entries {
        let map = unsafe { Mmap::map(&std::fs::File::open(shard)?)? };
        let hlen = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + hlen])?;
        for (name, meta) in header.as_object().context("header object")? {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
            let offs: Vec<u64> = serde_json::from_value(meta["data_offsets"].clone())?;
            let dtype = meta["dtype"].as_str().context("dtype")?;
            ensure!(dtype == "BF16", "{name}: expected BF16, got {dtype}");
            let job = Job {
                shard: shard.clone(),
                name: name.clone(),
                shape: shape.clone(),
                data_offsets: (8 + hlen + offs[0] as usize, 8 + hlen + offs[1] as usize),
            };
            if quantizable(name, &shape) {
                jobs.push(job);
            } else {
                copies.push(job);
            }
        }
    }
    println!(
        "{} tensors to quantize, {} to copy",
        jobs.len(),
        copies.len()
    );

    // Quantize in parallel over tensors (each maps its shard on demand).
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<(Vec<OutTensor>, f64, f64)>> = (0..jobs.len())
        .map(|_| std::sync::Mutex::new((Vec::new(), 0.0, 0.0)))
        .collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        return;
                    }
                    let job = &jobs[i];
                    let map =
                        unsafe { Mmap::map(&std::fs::File::open(&job.shard).unwrap()).unwrap() };
                    let raw = &map[job.data_offsets.0..job.data_offsets.1];
                    let bf: Vec<bf16> = raw
                        .chunks_exact(2)
                        .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    match quantize_tensor(&job.name, &job.shape, &bf) {
                        Ok(ts) => *results[i].lock().unwrap() = ts,
                        Err(e) => panic!("quantize {}: {e:?}", job.name),
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d.is_multiple_of(50) {
                        println!("{d}/{} quantized", jobs.len());
                    }
                }
            });
        }
    });

    // Ordered output: quantized tensors + verbatim copies, sharded ~3.5 GiB.
    let mut out: Vec<OutTensor> = Vec::new();
    let (mut sse, mut w2) = (0.0f64, 0.0f64);
    for r in &results {
        let mut g = r.lock().unwrap();
        sse += g.1;
        w2 += g.2;
        out.append(&mut g.0);
    }
    println!(
        "rms_ratio = {:.5} (sqrt(sse/w2), over {} quantized tensors)",
        (sse / w2).sqrt(),
        jobs.len()
    );
    for job in &copies {
        let map = unsafe { Mmap::map(&std::fs::File::open(&job.shard)?)? };
        out.push(OutTensor {
            name: job.name.clone(),
            dtype: "BF16",
            shape: job.shape.clone(),
            data: map[job.data_offsets.0..job.data_offsets.1].to_vec(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));

    const SHARD_CAP: u64 = 3_500_000_000;
    let mut weight_map = serde_json::Map::new();
    let mut shard_idx = 1usize;
    let total_shards_est =
        (out.iter().map(|t| t.data.len() as u64).sum::<u64>() / SHARD_CAP + 1) as usize;
    let mut cur: Vec<OutTensor> = Vec::new();
    let mut cur_bytes = 0u64;
    let total = out.len();
    for t in out {
        if cur_bytes + t.data.len() as u64 > SHARD_CAP && !cur.is_empty() {
            let fname = format!("model-{shard_idx:05}-of-{total_shards_est:05}.safetensors");
            for ct in &cur {
                weight_map.insert(ct.name.clone(), fname.clone().into());
            }
            write_shard(&dst.join(&fname), &cur)?;
            println!("wrote {fname} ({} tensors)", cur.len());
            shard_idx += 1;
            cur = Vec::new();
            cur_bytes = 0;
        }
        cur_bytes += t.data.len() as u64;
        cur.push(t);
    }
    if !cur.is_empty() {
        let fname = format!("model-{shard_idx:05}-of-{total_shards_est:05}.safetensors");
        for ct in &cur {
            weight_map.insert(ct.name.clone(), fname.clone().into());
        }
        write_shard(&dst.join(&fname), &cur)?;
        println!("wrote {fname} ({} tensors)", cur.len());
    }
    ensure!(
        shard_idx == total_shards_est,
        "shard count drifted: {shard_idx} != {total_shards_est}"
    );

    let index = serde_json::json!({
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    });
    std::fs::write(
        dst.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index)?,
    )?;
    // Provenance + config passthrough.
    std::fs::copy(src.join("config.json"), dst.join("config.json"))?;
    for extra in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "merges.txt",
        "vocab.json",
    ] {
        let p = src.join(extra);
        if p.exists() {
            std::fs::copy(&p, dst.join(extra))?;
        }
    }
    println!("done: {total} tensors -> {}", dst.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_dst_is_detected() {
        let tmp = std::env::temp_dir().join(format!("requant-test-{}", std::process::id()));
        let src = tmp.join("model");
        std::fs::create_dir_all(&src).unwrap();
        let src_abs = src.canonicalize().unwrap();
        // Equal, nested, and deeply-nested all start with src_abs.
        assert!(absolutize(&src).unwrap().starts_with(&src_abs));
        assert!(absolutize(&src.join("out")).unwrap().starts_with(&src_abs));
        assert!(
            absolutize(&src.join("sub/deep"))
                .unwrap()
                .starts_with(&src_abs)
        );
        assert!(!absolutize(&tmp.join("out")).unwrap().starts_with(&src_abs));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn bare_relative_dst_works() {
        // `requant <src> out` — the usage-line form: parent is "", not ".".
        let tmp = std::env::temp_dir().join(format!("requant-rel-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cwd = std::env::current_dir().unwrap();
        // CWD is process-global; restore on panic too, or every later
        // relative-path consumer in this test process runs in the wrong dir.
        struct RestoreCwd(PathBuf);
        impl Drop for RestoreCwd {
            fn drop(&mut self) {
                std::env::set_current_dir(&self.0).ok();
            }
        }
        let _guard = RestoreCwd(cwd);
        std::env::set_current_dir(&tmp).unwrap();
        let abs = absolutize(Path::new("out")).unwrap();
        assert_eq!(abs, tmp.canonicalize().unwrap().join("out"));
        drop(_guard);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn group_roundtrip_and_packing() {
        // Ascending values: q must come out ascending; packing is low
        // nibble first.
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut q = [0u8; 64];
        let (s, b) = quantize_group(&w, &mut q, 0);
        assert!(s > 0.0);
        let err: f32 = w
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - (s * q[i] as f32 + b)).abs())
            .fold(0.0, f32::max);
        // Half the quantum is the floor for any uniform 4-bit fit.
        assert!(err < s * 0.5 + 1e-3, "max abs err {err} vs quantum {s}");
        assert!(q.windows(2).all(|p| p[0] <= p[1] + 1), "q not ~ascending");
        // Constant group: degenerate scale guard.
        let w = [1.25f32; 64];
        let mut q = [9u8; 64];
        let (s, b) = quantize_group(&w, &mut q, 0);
        for (i, &v) in w.iter().enumerate() {
            let rec = s * q[i] as f32 + b;
            assert!((v - rec).abs() < 1e-3, "const group off by {}", v - rec);
        }
    }
}
