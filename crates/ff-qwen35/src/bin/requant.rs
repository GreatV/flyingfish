//! Groupwise affine int4 conversion, with RTN or least-squares refit.
//! Usage: requant <source> <destination> [threads] [refit|rtn].

use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::Mmap;
use serde::Deserializer;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Method {
    Refit,
    Rtn,
}

fn quantize_group(w: &[f32], out: &mut [u8; 64], method: Method) -> (f32, f32) {
    if method == Method::Rtn {
        let min = w.iter().copied().fold(f32::INFINITY, f32::min);
        let max = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let magnitude = ((max - min) * (1.0f32 / 15.0)).max(1e-7);
        let (mut scale, edge) = if min.abs() > max.abs() {
            (magnitude, min)
        } else {
            (-magnitude, max)
        };
        let zero = (edge / scale).round_ties_even();
        let bias = if zero == 0.0 {
            0.0
        } else {
            scale = edge / zero;
            edge
        };
        for (q, &value) in out.iter_mut().zip(w) {
            *q = ((value - bias) / scale).round_ties_even().clamp(0.0, 15.0) as u8;
        }
        return (
            bf16::from_f32(scale).to_f32(),
            bf16::from_f32(bias).to_f32(),
        );
    }
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
    dtype: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Returns the tensors plus (sse, sum_w2) against the stored bf16-rounded
/// (s, b) — the error dequant actually sees.
fn quantize_tensor(
    name: &str,
    shape: &[usize],
    bf: &[bf16],
    method: Method,
) -> Result<(Vec<OutTensor>, f64, f64)> {
    let (rows, cols) = (shape[0], shape[1]);
    ensure!(
        rows.checked_mul(cols) == Some(bf.len()),
        "{name}: shape and payload disagree"
    );
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
            ensure!(
                w.iter().all(|v| v.is_finite()),
                "{name}: nonfinite weights at row {r}, group {g}"
            );
            let mut q = [0u8; 64];
            let (s, b) = quantize_group(&w, &mut q, method);
            ensure!(
                s.is_finite() && b.is_finite(),
                "{name}: nonfinite quantization parameters"
            );
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
                dtype: "U32".into(),
                shape: vec![rows, words_per_row],
                data: bytemuck_cast_u32(&packed),
            },
            OutTensor {
                name: format!("{base}.scales"),
                dtype: "BF16".into(),
                shape: vec![rows, groups_per_row],
                data: bytemuck_cast_bf16(&scales),
            },
            OutTensor {
                name: format!("{base}.biases"),
                dtype: "BF16".into(),
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
        let end = offset
            .checked_add(t.data.len() as u64)
            .context("shard size overflow")?;
        header.push_str(&format!(
            "{}:{{\"dtype\":{},\"shape\":{},\"data_offsets\":[{},{}]}}",
            serde_json::to_string(&t.name)?,
            serde_json::to_string(&t.dtype)?,
            serde_json::to_string(&t.shape)?,
            offset,
            end
        ));
        offset = end;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let staging = ff_core::artifact::ArtifactStaging::new(path)?;
    staging.write_with(|file| {
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(header.as_bytes())?;
        for tensor in tensors {
            writer.write_all(&tensor.data)?;
        }
        writer.flush()?;
        Ok(())
    })?;
    staging.publish()?;
    Ok(())
}

struct Job {
    shard: PathBuf,
    name: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
    dtype: String,
}

struct HeaderOrder;

impl<'de> serde::de::Visitor<'de> for HeaderOrder {
    type Value = Vec<(String, serde_json::Value)>;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a safetensors header object")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut entries = Vec::new();
        while let Some(entry) = map.next_entry()? {
            entries.push(entry);
        }
        Ok(entries)
    }
}

fn shard_jobs(shard: &Path) -> Result<Vec<Job>> {
    let map = unsafe { Mmap::map(&std::fs::File::open(shard)?)? };
    ensure!(
        map.len() >= 8,
        "truncated safetensors header: {}",
        shard.display()
    );
    let len = usize::try_from(u64::from_le_bytes(map[..8].try_into()?))?;
    let start = len.checked_add(8).context("header size overflow")?;
    ensure!(start <= map.len(), "header exceeds {}", shard.display());
    let header =
        serde_json::Deserializer::from_slice(&map[8..start]).deserialize_map(HeaderOrder)?;
    let mut jobs = Vec::new();
    for (name, meta) in header {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = serde_json::from_value(meta["shape"].clone())?;
        let offsets: [usize; 2] = serde_json::from_value(meta["data_offsets"].clone())?;
        ensure!(
            offsets[0] <= offsets[1] && offsets[1] <= map.len() - start,
            "{name}: invalid data offsets"
        );
        let dtype = meta["dtype"]
            .as_str()
            .context("missing tensor dtype")?
            .to_owned();
        if quantizable(&name, &shape) {
            ensure!(dtype == "BF16", "{name}: expected BF16, got {dtype}");
        }
        jobs.push(Job {
            shard: shard.to_owned(),
            name,
            shape,
            data_offsets: (start + offsets[0], start + offsets[1]),
            dtype,
        });
    }
    Ok(jobs)
}

fn convert(jobs: &[Job], threads: usize, method: Method) -> Result<(Vec<OutTensor>, f64, f64)> {
    let next = AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<(Vec<OutTensor>, f64, f64)>> = (0..jobs.len())
        .map(|_| std::sync::Mutex::new((Vec::new(), 0.0, 0.0)))
        .collect();
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for _ in 0..threads.min(jobs.len()) {
            workers.push(scope.spawn(|| -> Result<()> {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let map = unsafe { Mmap::map(&std::fs::File::open(&job.shard)?)? };
                    let raw = &map[job.data_offsets.0..job.data_offsets.1];
                    let converted = if quantizable(&job.name, &job.shape) {
                        ensure!(
                            raw.len().is_multiple_of(2),
                            "{}: odd BF16 payload",
                            job.name
                        );
                        let bf = raw
                            .chunks_exact(2)
                            .map(|c| bf16::from_le_bytes([c[0], c[1]]))
                            .collect::<Vec<_>>();
                        quantize_tensor(&job.name, &job.shape, &bf, method)?
                    } else {
                        (
                            vec![OutTensor {
                                name: job.name.clone(),
                                dtype: job.dtype.clone(),
                                shape: job.shape.clone(),
                                data: raw.to_vec(),
                            }],
                            0.0,
                            0.0,
                        )
                    };
                    *results[i].lock().expect("quantization slot poisoned") = converted;
                }
                Ok(())
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("quantization worker panicked"))??;
        }
        Ok(())
    })?;
    let mut out = Vec::new();
    let (mut sse, mut w2) = (0.0, 0.0);
    for result in results {
        let (mut tensors, error, energy) = result
            .into_inner()
            .map_err(|_| anyhow::anyhow!("quantization slot poisoned"))?;
        out.append(&mut tensors);
        sse += error;
        w2 += energy;
    }
    Ok((out, sse, w2))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let src = PathBuf::from(
        args.next()
            .context("usage: requant <src> <dst> [threads] [refit|rtn]")?,
    )
    .canonicalize()?;
    let dst = PathBuf::from(
        args.next()
            .context("usage: requant <src> <dst> [threads] [refit|rtn]")?,
    );
    let threads = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()
        .context("invalid quantization worker count")?
        .unwrap_or(8);
    ensure!(threads > 0, "quantization worker count must be positive");
    let method = match args.next().as_deref() {
        None | Some("refit") => Method::Refit,
        Some("rtn") => Method::Rtn,
        Some(value) => anyhow::bail!("unknown quantization method {value}; expected refit or rtn"),
    };
    ensure!(
        args.next().is_none(),
        "usage: requant <src> <dst> [threads] [refit|rtn]"
    );
    let parent = dst
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("destination parent must exist")?;
    let dst = parent.join(
        dst.file_name()
            .context("destination must name a directory")?,
    );
    let dst = if dst.exists() {
        dst.canonicalize()?
    } else {
        dst
    };
    ensure!(
        !src.starts_with(&dst) && !dst.starts_with(&src),
        "source and destination overlap"
    );
    std::fs::create_dir_all(&dst)?;
    println!(
        "source: {}\ndestination: {}\nmethod: {method:?}, threads: {threads}",
        src.display(),
        dst.display()
    );
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("config.json"))?)?;
    config["quantization"] =
        serde_json::json!({"format": ff_qwen35::config::QUANTIZATION_FORMAT, "group_size":64});
    let config_path = dst.join("config.json");
    if config_path.exists() {
        let existing: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        ensure!(
            existing == config,
            "destination config differs from this source"
        );
    } else {
        std::fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;
    }
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
    ] {
        let source = src.join(name);
        let target = dst.join(name);
        if source.exists() {
            if target.exists() {
                ensure!(
                    std::fs::read(&source)? == std::fs::read(&target)?,
                    "destination {name} differs from source"
                );
            } else {
                std::fs::copy(source, target)?;
            }
        }
    }
    let index_path = dst.join("model.safetensors.index.json");
    let (mut weight_map, mut total): (serde_json::Map<String, serde_json::Value>, u64) =
        if index_path.exists() {
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
            let map = index["weight_map"]
                .as_object()
                .context("destination index missing weight_map")?
                .clone();
            let total = index["metadata"]["total_size"]
                .as_u64()
                .context("destination index missing total_size")?;
            ensure!(
                total > 0 || map.is_empty(),
                "destination index has zero size for nonempty weights"
            );
            (map, total)
        } else {
            (serde_json::Map::new(), 0)
        };
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(&src)? {
        let path = entry?.path();
        if path.extension().is_some_and(|v| v == "safetensors") {
            shards.push(path);
        }
    }
    shards.sort();
    ensure!(!shards.is_empty(), "no safetensors under {}", src.display());
    for shard in shards {
        let name = shard
            .file_name()
            .context("missing shard name")?
            .to_str()
            .context("shard name is not UTF-8")?;
        let target = dst.join(name);
        ensure!(
            !target.exists(),
            "destination shard already exists: {}",
            target.display()
        );
        let jobs = shard_jobs(&shard)?;
        let (tensors, sse, w2) = convert(&jobs, threads, method)?;
        for tensor in &tensors {
            ensure!(
                !weight_map.contains_key(&tensor.name),
                "duplicate destination tensor {}",
                tensor.name
            );
            weight_map.insert(tensor.name.clone(), name.into());
            total = total
                .checked_add(tensor.data.len() as u64)
                .context("checkpoint size overflow")?;
        }
        write_shard(&target, &tensors)?;
        let index = serde_json::json!({"metadata":{"total_size":total},"weight_map":weight_map});
        let staging = ff_core::artifact::ArtifactStaging::new(&index_path)?;
        staging.write_bytes(&serde_json::to_vec_pretty(&index)?)?;
        ff_core::artifact::replace_file_durably(staging.producer_path(), &index_path)?;
        println!(
            "wrote {name}: {} tensors, cumulative {total} B, rms_ratio {:.5}",
            tensors.len(),
            if w2 > 0.0 { (sse / w2).sqrt() } else { 0.0 }
        );
    }
    println!("done: {} tensors -> {}", weight_map.len(), dst.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtn_uses_signed_scale_even_rounding_and_low_nibbles() {
        let mut w = [0.0; 64];
        w[1..6].copy_from_slice(&[15.0, 14.5, 13.5, 12.5, 11.5]);
        let mut q = [0; 64];
        let (scale, bias) = quantize_group(&w, &mut q, Method::Rtn);
        assert_eq!((scale, bias), (-1.0, 15.0));
        assert_eq!(&q[..6], &[15, 0, 0, 2, 2, 4]);
        let bf = w.map(bf16::from_f32);
        let (tensors, _, _) = quantize_tensor("test.weight", &[1, 64], &bf, Method::Rtn).unwrap();
        assert_eq!(
            u32::from_le_bytes(tensors[0].data[..4].try_into().unwrap()),
            0xff42_200f
        );
        for value in [0.0, 1.25] {
            let (scale, bias) = quantize_group(&[value; 64], &mut q, Method::Rtn);
            assert!(scale.is_finite() && scale < 0.0);
            assert_eq!(bias, value);
            assert_eq!(q, [0; 64]);
        }
    }

    #[test]
    fn group_roundtrip_and_packing() {
        // Ascending values: q must come out ascending; packing is low
        // nibble first.
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut q = [0u8; 64];
        let (s, b) = quantize_group(&w, &mut q, Method::Refit);
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
        let (s, b) = quantize_group(&w, &mut q, Method::Refit);
        for (i, &v) in w.iter().enumerate() {
            let rec = s * q[i] as f32 + b;
            assert!((v - rec).abs() < 1e-3, "const group off by {}", v - rec);
        }
    }
}
