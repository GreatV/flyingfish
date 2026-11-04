use super::*;
use crate::checkpoint::MappedCheckpoint;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, path::PathBuf};

type CellId = (String, String, Option<String>);
struct HostTensor {
    shape: Vec<usize>,
    values: Vec<f64>,
}
fn tensor(path: &Path, key: &str) -> Result<HostTensor> {
    let checkpoint = MappedCheckpoint::open_component(
        path.parent().context("tensor parent missing")?,
        path.file_stem()
            .and_then(|x| x.to_str())
            .context("tensor stem missing")?,
    )?;
    let t = checkpoint.tensor(key)?;
    let values = match t.dtype() {
        Dtype::F32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect(),
        Dtype::F64 => t
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        Dtype::BF16 => t
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f64())
            .collect(),
        Dtype::F16 => t
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f64())
            .collect(),
        Dtype::I32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f64)
            .collect(),
        Dtype::I64 => t
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f64)
            .collect(),
        Dtype::BOOL => t.data().iter().map(|&x| x as f64).collect(),
        dtype => anyhow::bail!("unsupported reference dtype {dtype:?}"),
    };
    Ok(HostTensor {
        shape: t.shape().to_vec(),
        values,
    })
}
fn f32_tensor(path: &Path, key: &str) -> Result<Vec<f32>> {
    Ok(tensor(path, key)?
        .values
        .into_iter()
        .map(|x| x as f32)
        .collect())
}
fn id(table: &str, name: &str, key: Option<&str>) -> CellId {
    (table.into(), name.into(), key.map(str::to_owned))
}
fn env_path(name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(
        std::env::var_os(name).with_context(|| format!("{name} required"))?,
    ))
}
fn metrics(x: &[f64], y: &[f64]) -> Result<BTreeMap<&'static str, f64>> {
    ensure!(!x.is_empty() && x.len() == y.len(), "metric shape mismatch");
    let (mut max, mut square, mut dot, mut xx, mut yy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in x.iter().zip(y) {
        ensure!(a.is_finite() && b.is_finite(), "non-finite gate input");
        let d = a - b;
        max = max.max(d.abs());
        square += d * d;
        dot += a * b;
        xx += a * a;
        yy += b * b;
    }
    let cosine = if xx > 0.0 && yy > 0.0 {
        (1.0 - dot / (xx * yy).sqrt()).clamp(0.0, 2.0)
    } else if xx == yy {
        0.0
    } else {
        1.0
    };
    Ok(BTreeMap::from([
        ("max_abs", max),
        ("rmse", (square / x.len() as f64).sqrt()),
        ("cosine_error", cosine),
    ]))
}

struct Gate {
    cells: BTreeMap<CellId, Value>,
    anchors: BTreeMap<CellId, Value>,
    seen: BTreeSet<CellId>,
    defects: usize,
}
impl Gate {
    fn new() -> Result<Self> {
        let path = env_path("FF_PIXAL_FREEZE")?;
        let bytes = std::fs::read(&path)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes))
                == "231d9f8eba81398b442c6eebfbee947c1d9e43c9df4b983e11a538d75dc47e76",
            "M3 freeze integrity mismatch"
        );
        let freeze: Value = serde_json::from_slice(&bytes)?;
        ensure!(freeze["reference_valid"] == true, "invalid M3 freeze");
        let selected: BTreeSet<_> = freeze["split"]["M3b"]
            .as_array()
            .context("M3b split missing")?
            .iter()
            .map(|c| {
                id(
                    c["table"].as_str().unwrap(),
                    c["name"].as_str().unwrap(),
                    c["key"].as_str(),
                )
            })
            .collect();
        let mut cells = BTreeMap::new();
        let mut anchors = BTreeMap::new();
        for table in freeze["tables"].as_array().context("tables missing")? {
            let name = table["source_table"]
                .as_str()
                .context("table name missing")?;
            for c in table["cells"].as_array().context("cells missing")? {
                let k = id(
                    name,
                    c["name"].as_str().context("cell name missing")?,
                    c["key"].as_str(),
                );
                if selected.contains(&k) {
                    cells.insert(k, c.clone());
                }
            }
        }
        for c in freeze["controls"].as_array().context("controls missing")? {
            let k = id(
                c["table"].as_str().unwrap(),
                c["name"].as_str().unwrap(),
                c["key"].as_str(),
            );
            if selected.contains(&k) {
                ensure!(c["valid"] == true, "invalid control");
                if let Some(ratios) = c["ratios"].as_object() {
                    for (m, v) in ratios {
                        if c["gating"][m] == true {
                            ensure!(v.as_f64().context("ratio missing")? >= 10.0, "weak control");
                        }
                    }
                }
                anchors.insert(k, c["anchor"].clone());
            }
        }
        ensure!(
            cells.len() == 41 && anchors.len() == 41,
            "M3b primary coverage mismatch"
        );
        let parent = path.parent().context("freeze parent missing")?;
        let bytes = std::fs::read(parent.join("M3-PREFIX-FREEZE.json"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes))
                == "aeebbf22ae6031391f85e1be7dec572cf9dd2cae6ee44cf02ec0ecccba025d27",
            "prefix freeze integrity mismatch"
        );
        let prefix: Value = serde_json::from_slice(&bytes)?;
        ensure!(prefix["reference_valid"] == true, "invalid prefix freeze");
        for table in prefix["tables"]
            .as_array()
            .context("prefix tables missing")?
        {
            let case = table["case"].as_str().context("prefix case missing")?;
            ensure!(table["anchor"] == "r3math", "unexpected prefix anchor");
            for c in table["cells"].as_array().context("prefix cells missing")? {
                ensure!(c["reference_valid"] == true, "invalid prefix cell");
                let name = c["name"].as_str().context("prefix name missing")?;
                let (stage, key) = name.split_once('-').context("prefix stage/key missing")?;
                let k = id(case, name, None);
                let mut c = c.clone();
                c["frozen_gating_metrics"] = c["gating_metrics"].clone();
                if let Some(ratios) = c["ratios"].as_object() {
                    for (m, v) in ratios {
                        if c["gating_metrics"][m] == true {
                            ensure!(
                                v.as_f64().context("prefix ratio missing")? >= 10.0,
                                "weak prefix control"
                            );
                        }
                    }
                }
                cells.insert(k.clone(), c);
                anchors.insert(k,serde_json::json!({"path":parent.join(format!("prefix-{case}-r3math/{stage}.safetensors")),"key":if key=="coordinates"{"coords"}else{key}}));
            }
        }
        ensure!(
            cells.len() == 101 && anchors.len() == 101,
            "M3b total coverage mismatch"
        );
        println!(
            "P1_M3B_FREEZE cells=101 minimum_control_ratio=10.015377390521339 thresholds_unchanged=true prefix_unqualified_signs=391402"
        );
        Ok(Self {
            cells,
            anchors,
            seen: BTreeSet::new(),
            defects: 0,
        })
    }
    fn anchor(&self, k: &CellId) -> Result<HostTensor> {
        let a = self.anchors.get(k).context("unknown gate cell")?;
        tensor(
            Path::new(a["path"].as_str().context("anchor path missing")?),
            a["key"].as_str().context("anchor key missing")?,
        )
    }
    fn score(&mut self, k: CellId, shape: &[usize], values: &[f64]) -> Result<()> {
        ensure!(self.seen.insert(k.clone()), "duplicate score {k:?}");
        let reference = self.anchor(&k)?;
        let c = &self.cells[&k];
        ensure!(
            shape == reference.shape && values.len() == reference.values.len(),
            "gate shape mismatch {k:?}"
        );
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite candidate {k:?}"
        );
        if let Some(taus) = c["tau"].as_object() {
            let result = metrics(values, &reference.values)?;
            for (name, tau) in taus {
                let tau = tau.as_f64().context("tau missing")?;
                let gating = c["frozen_gating_metrics"][name] == true;
                let defect = gating && result[name.as_str()] > tau;
                self.defects += usize::from(defect);
                println!(
                    "P1_M3B_METRIC {}",
                    serde_json::json!({"table":k.0,"name":k.1,"key":k.2,"metric":name,"value":result[name.as_str()],"tau":tau,"gating":gating,"result":if defect{"DEFECT"}else{"PASS"}})
                );
            }
        } else {
            let (reference, values) = if k.1.ends_with("-coarse") {
                (pool(&reference.values, 64, 32), pool(values, 64, 32))
            } else {
                (reference.values, values.to_vec())
            };
            let margin = c["margin_tau"].as_f64();
            let quant = k.1.ends_with("stable-quantization");
            let mut stable = 0;
            let mut mismatches = 0;
            for (&a, &b) in reference.iter().zip(&values) {
                let valid = margin.is_none_or(|tau| {
                    if quant {
                        (a - (a.floor() + 0.5)).abs() > tau
                    } else {
                        a.abs() > tau
                    }
                });
                if valid {
                    stable += 1;
                    mismatches += usize::from(if quant {
                        a.round_ties_even() != b.round_ties_even()
                    } else if margin.is_some() {
                        (a > 0.0) != (b > 0.0)
                    } else {
                        a != b
                    });
                }
            }
            let gating = c["frozen_gating_metrics"]["exact"] == true;
            self.defects += usize::from(gating && mismatches > 0);
            println!(
                "P1_M3B_EXACT {}",
                serde_json::json!({"table":k.0,"name":k.1,"stable":stable,"mismatches":mismatches,"gating":gating})
            );
        }
        Ok(())
    }
    fn score_f32(&mut self, k: CellId, shape: &[usize], values: &[f32]) -> Result<()> {
        self.score(
            k,
            shape,
            &values.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        )
    }
    fn finish(self) -> Result<()> {
        ensure!(
            self.seen.len() == self.cells.len(),
            "unscored M3b cells: {:?}",
            self.cells
                .keys()
                .filter(|k| !self.seen.contains(*k))
                .collect::<Vec<_>>()
        );
        println!(
            "P1_M3B_GATE {} cells={} defects={} timing=false eager=true",
            if self.defects == 0 { "PASS" } else { "DEFECT" },
            self.seen.len(),
            self.defects
        );
        ensure!(self.defects == 0, "M3b candidate defects");
        Ok(())
    }
}
fn pool(values: &[f64], side: usize, target: usize) -> Vec<f64> {
    let f = side / target;
    let mut out = vec![f64::NEG_INFINITY; target.pow(3)];
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                let i = (x / f * target + y / f) * target + z / f;
                out[i] = out[i].max(values[(x * side + y) * side + z]);
            }
        }
    }
    out
}
fn coordinates(path: &Path, key: &str) -> Result<Vec<[i32; 4]>> {
    let t = tensor(path, key)?;
    ensure!(
        t.shape.len() == 2 && t.shape[1] == 4,
        "coordinate layout mismatch"
    );
    t.values
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| {
            ensure!(
                v.iter()
                    .all(|&x| x.fract() == 0.0 && x >= 0.0 && x <= i32::MAX as f64),
                "invalid coordinate"
            );
            Ok(v.map(|x| x as i32))
        })
        .collect()
}
fn host(d: &Device, x: &CudaSlice<f16>) -> Result<Vec<f32>> {
    Ok(d.stream
        .clone_dtoh(x)?
        .into_iter()
        .map(f16::to_f32)
        .collect())
}

fn dense(g: &mut Gate, root: &Path, model: &Path, runtime: &Path) -> Result<()> {
    let decoder = PixalDecoder::open(model, DecoderKind::Structure, runtime, 4_000_000_000)?;
    let mut first = None;
    for variant in ["single", "mv"] {
        for fixture in ["cottage", "robot_crab"] {
            for fixed in [false, true] {
                let arm = if fixed { "r2" } else { "r3math" };
                let path=root.join(format!("evidence/P1-M3-freeze-r1/recovered/ss-{variant}-{arm}/{fixture}-flow.safetensors"));
                let latent = f32_tensor(&path, "final")?;
                let logits = decoder.structure(&latent)?;
                let table = if fixed {
                    format!("floors-ss-fixed-{variant}-angular-r1.json")
                } else {
                    format!("floors-ss-{variant}.json")
                };
                let name = if fixed {
                    format!("{fixture}-decoder")
                } else {
                    fixture.into()
                };
                g.score_f32(
                    id(&table, &name, if fixed { None } else { Some("logits") }),
                    &[1, 1, 64, 64, 64],
                    &logits,
                )?;
                let table = format!("floors-decisions-{variant}.json");
                let name = format!(
                    "{fixture}-{}-occupancy",
                    if fixed { "fixed-decoder" } else { "unforced" }
                );
                g.score_f32(id(&table, &name, None), &[1, 1, 64, 64, 64], &logits)?;
                if !fixed {
                    g.score_f32(
                        id(&table, &format!("{fixture}-unforced-coarse"), None),
                        &[1, 1, 64, 64, 64],
                        &logits,
                    )?;
                }
                let sites = PixalDecoder::structure_sites(&logits, 64, 32)?;
                let pooled = pool(
                    &logits.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                    64,
                    32,
                );
                ensure!(
                    sites.coordinates().len() == pooled.iter().filter(|&&x| x > 0.0).count(),
                    "occupancy/maxpool/argwhere count mismatch"
                );
                if first.is_none() {
                    first = Some((latent, logits));
                }
            }
        }
    }
    let (input, expected) = first.context("dense fixture missing")?;
    ensure!(
        decoder.structure(&input)? == expected,
        "dense A-B-A isolation defect"
    );
    println!("P1_M3B_ISOLATION PASS dense_A_B_A_bit_exact=true");
    Ok(())
}
fn pointwise(g: &mut Gate, decoder: &PixalDecoder) -> Result<()> {
    let d = &decoder.device;
    for (stage, c) in [1024usize, 512, 256, 128].into_iter().enumerate() {
        let n = 4099;
        let mut values = Vec::with_capacity(n * c);
        for row in 0..n {
            for channel in 0..c {
                let a = ((row * 7 + channel * 11) % 17) as f32 - 8.0;
                let b = ((row * 13 + channel * 5) % 19) as f32 - 9.0;
                let mut x = f16::from_f32(a / 32.0 + b / 64.0);
                if row == 0 {
                    x = f16::from_f32(0.25);
                } else if row == 1 {
                    x = f16::from_f32(0.25 + ((channel % 7) as f32 - 3.0) / 4096.0);
                } else if row == 2 {
                    x = f16::from_f32(x.to_f32() * 16.0);
                }
                values.push(x);
            }
        }
        let x = d.stream.clone_htod(&values)?;
        let norm = d.stream.alloc_zeros::<f16>(n * c)?;
        let name = format!("blocks.{stage}.0");
        decoder.norm16(Some(&format!("{name}.norm")), &x, &norm, n, c, 1e-6)?;
        g.score_f32(
            id(
                "floors-pointwise-published.json",
                &format!("shape-{c}-norm"),
                None,
            ),
            &[n, c],
            &host(d, &norm)?,
        )?;
        let mut wide = d.stream.alloc_zeros::<f16>(n * c * 4)?;
        let mut output = d.stream.alloc_zeros::<f16>(n * c)?;
        decoder.linear16(&format!("{name}.mlp.0"), &norm, &mut wide, [n, c, 4 * c])?;
        decoder.k.act(d, ptr(&wide, d), n, 4 * c, false)?;
        decoder.linear16(&format!("{name}.mlp.2"), &wide, &mut output, [n, 4 * c, c])?;
        let k = id(
            "floors-pointwise-published.json",
            &format!("shape-{c}-mlp_residual"),
            None,
        );
        if stage == 0 {
            let wrong = host(d, &output)?;
            let reference = g.anchor(&k)?;
            let separation = metrics(
                &wrong.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                &reference.values,
            )?;
            for metric in ["max_abs", "rmse"] {
                let tau = g.cells[&k]["tau"][metric]
                    .as_f64()
                    .context("control tau missing")?;
                ensure!(
                    separation[metric] >= 10.0 * tau,
                    "MLP no-residual control weak"
                );
                println!(
                    "P1_M3B_CONTROL omitted_residual metric={metric} ratio={}",
                    separation[metric] / tau
                );
            }
        }
        decoder.k.add(d, &output, &x, &decoder.ones, [n, c])?;
        g.score_f32(k, &[n, c], &host(d, &output)?)?;
    }
    Ok(())
}
fn sparse_ops(g: &mut Gate, decoder: &PixalDecoder, root: &Path) -> Result<()> {
    let d = &decoder.device;
    for c2s in [false, true] {
        let path = root.join(if c2s {
            "evidence/P1-M1-r1/sparse-c2s-r2/operator.safetensors"
        } else {
            "evidence/P1-M1-r1/sparse-conv-r2/operator.safetensors"
        });
        let sites = SparseSites::new(coordinates(&path, "coords")?)?;
        let n = sites.coordinates().len();
        let input = f32_tensor(&path, "features")?;
        let input: Vec<_> = input.into_iter().map(f16::from_f32).collect();
        let x = d.stream.clone_htod(&input)?;
        let co = if c2s { 4096 } else { 1024 };
        let conv = SparseConv::new(d, &sites, 1024, 512 * 1024 * 1024)?;
        let mut y = d.stream.alloc_zeros::<f16>(n * co)?;
        decoder.sparse_conv(
            &conv,
            if c2s {
                "blocks.0.4.conv1"
            } else {
                "blocks.0.0.conv"
            },
            &x,
            &mut y,
            1024,
        )?;
        drop(conv);
        let table = if c2s {
            "floors-sparse-c2s-published-r2.json"
        } else {
            "floors-sparse-conv-published.json"
        };
        g.score_f32(id(table, "output", None), &[n, co], &host(d, &y)?)?;
        drop(y);
        if !c2s {
            continue;
        }
        let norm = d.stream.alloc_zeros::<f16>(n * 1024)?;
        decoder.norm16(Some("blocks.0.4.norm1"), &x, &norm, n, 1024, 1e-6)?;
        g.score_f32(id(table, "norm", None), &[n, 1024], &host(d, &norm)?)?;
        let mut sub = d.stream.alloc_zeros::<f16>(n * 8)?;
        decoder.linear16("blocks.0.4.to_subdiv", &x, &mut sub, [n, 1024, 8])?;
        let sub = host(d, &sub)?;
        g.score_f32(id(table, "subdivision", None), &[n, 8], &sub)?;
        g.score_f32(id(table, "stable_subdivision_sign", None), &[n, 8], &sub)?;
        let mut wide = d.stream.alloc_zeros::<f16>(n * 4096)?;
        let mut y = d.stream.alloc_zeros::<f16>(n * 1024)?;
        decoder.linear16("blocks.0.0.mlp.0", &norm, &mut wide, [n, 1024, 4096])?;
        decoder.k.act(d, ptr(&wide, d), n, 4096, false)?;
        decoder.linear16("blocks.0.0.mlp.2", &wide, &mut y, [n, 4096, 1024])?;
        decoder.k.add(d, &y, &x, &decoder.ones, [n, 1024])?;
        g.score_f32(id(table, "mlp_residual", None), &[n, 1024], &host(d, &y)?)?;
        drop(wide);
        drop(y);
        drop(norm);
        drop(x);
        let guide = f32_tensor(&path, "subdivision")?;
        let masks: Vec<_> = guide
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| v.map(|x| x > 0.0))
            .collect();
        let (children, pairs) = sites.subdivide(&masks, n * 8)?;
        let coords = children
            .coordinates()
            .as_flattened()
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>();
        g.score(
            id(table, "channel_to_spatial", None),
            &[children.coordinates().len(), 4],
            &coords,
        )?;
        let common = f32_tensor(&path, "output")?
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
        let input = d.stream.clone_htod(&common)?;
        let pairs = d.stream.clone_htod(pairs.as_flattened())?;
        let features = d
            .stream
            .alloc_zeros::<f16>(children.coordinates().len() * 512)?;
        decoder.c2s(
            &input,
            &pairs,
            &features,
            [n, children.coordinates().len(), 512, 512],
            false,
        )?;
        ensure!(
            host(d, &features)? == f32_tensor(&path, "child_features")?,
            "C2S fixed-feature indexing defect"
        );
        println!(
            "P1_M3B_C2S PASS fixed_features_bit_exact=true rows={}",
            children.coordinates().len()
        );
    }
    Ok(())
}
fn transitions(g: &mut Gate, root: &Path) -> Result<()> {
    for (fixture, resolution, capture) in [
        ("cottage", 1536, "pipeline-single-1536-cottage-r0a"),
        ("robot_crab", 1408, "pipeline-single-1536-robot_crab-r0a"),
        ("mv4", 1024, "pipeline-mv-1024-mv4-r0a"),
    ] {
        let base = root.join("evidence/P1-M1-r1").join(capture);
        let sites = SparseSites::new(coordinates(
            &base.join("shape_slat_decoder.blocks.3.4-call0-fixed-input.safetensors"),
            "output.0.coords",
        )?)?;
        let scaled = sites.scaled_coordinates(512, resolution / 16)?;
        for suffix in ["quantization", "stable-quantization"] {
            g.score_f32(
                id(
                    "floors-transition-published.json",
                    &format!("{fixture}-{suffix}"),
                    None,
                ),
                &[scaled.len(), 3],
                scaled.as_flattened(),
            )?;
        }
        let result = sites.round_unique(512, resolution / 16)?;
        let expected = coordinates(
            &base.join("shape_slat-1-inputs.safetensors"),
            "noise.coords",
        )?;
        ensure!(
            result.coordinates() == expected,
            "cascade round/unique coordinate defect"
        );
        println!(
            "P1_M3B_TRANSITION PASS fixture={fixture} input_rows={} output_rows={} exact=true",
            sites.coordinates().len(),
            result.coordinates().len()
        );
    }
    Ok(())
}
fn prefix(g: &mut Gate, decoder: &PixalDecoder, root: &Path, model: &Path) -> Result<()> {
    let pipeline: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let norm = &pipeline["args"]["shape_slat_normalization"];
    let mean = norm["mean"].as_array().context("shape mean missing")?;
    let std = norm["std"].as_array().context("shape std missing")?;
    ensure!(
        mean.len() == 32 && std.len() == 32,
        "shape normalization width mismatch"
    );
    for (case, capture) in [
        ("cottage1024", "pipeline-single-1024-cottage-r2"),
        ("crab1024", "pipeline-single-1024-robot_crab-r2"),
        ("cottage1536", "pipeline-single-1536-cottage-r0a"),
        ("crab1408", "pipeline-single-1536-robot_crab-r0a"),
        ("mv41024", "pipeline-mv-1024-mv4-r0a"),
    ] {
        let base = root.join("evidence/P1-M1-r1").join(capture);
        let input = base.join("shape_slat-0-outputs.safetensors");
        let coords = coordinates(&input, "samples.coords")?;
        let mut latent = f32_tensor(&input, "samples.features")?;
        for (i, x) in latent.iter_mut().enumerate() {
            *x = *x * std[i % 32].as_f64().context("invalid std")? as f32
                + mean[i % 32].as_f64().context("invalid mean")? as f32;
        }
        let mut guide_coords = Vec::new();
        let mut guide_logits = Vec::new();
        for (stage, block) in [4, 16, 8, 4].into_iter().enumerate() {
            let suffix = if case == "cottage1024" {
                "-fixed-input.safetensors"
            } else {
                "-call0-fixed-input.safetensors"
            };
            let path = base.join(format!("shape_slat_decoder.blocks.{stage}.{block}{suffix}"));
            guide_coords.push(coordinates(&path, "output.1.coords")?);
            guide_logits.push(f32_tensor(&path, "output.1.features")?);
        }
        let guides: Vec<_> = guide_coords
            .iter()
            .zip(&guide_logits)
            .map(|(coordinates, logits)| SubdivisionGuide {
                coordinates,
                logits,
            })
            .collect();
        decoder.shape_prefix(coords, &latent, &guides, |stage, out, sub| {
            g.score_f32(
                id(case, &format!("stage{stage}-features"), None),
                &[out.coordinates.len(), out.channels],
                &out.features,
            )?;
            g.score_f32(
                id(case, &format!("stage{stage}-subdivision"), None),
                &[sub.len() / 8, 8],
                sub,
            )?;
            g.score(
                id(case, &format!("stage{stage}-coordinates"), None),
                &[out.coordinates.len(), 4],
                &out.coordinates
                    .as_flattened()
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
            )?;
            println!(
                "P1_M3B_PREFIX_PROGRESS case={case} stage={stage} rows={} channels={}",
                out.coordinates.len(),
                out.channels
            );
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU, published checkpoints, and the frozen Pixal3D captures"]
fn pixal3d_decoders_match_frozen_definition_gpu() -> Result<()> {
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let mut g = Gate::new()?;
    dense(&mut g, &root, &model, &runtime)?;
    let decoder = PixalDecoder::open(&model, DecoderKind::Shape, &runtime, 10_000_000_000)?;
    pointwise(&mut g, &decoder)?;
    sparse_ops(&mut g, &decoder, &root)?;
    transitions(&mut g, &root)?;
    prefix(&mut g, &decoder, &root, &model)?;
    g.finish()
}

#[test]
fn conditioning_rejects_another_flow_with_the_same_row_count() -> Result<()> {
    let coordinates = [[1, 3, 7], [2, 5, 11], [4, 9, 13]];
    let origin = Rc::new(());
    validate_conditioning_origin(&origin.clone(), &origin)?;
    for foreign in [
        [[1, 3, 8], [2, 5, 11], [4, 9, 13]],
        [[4, 9, 13], [1, 3, 7], [2, 5, 11]],
        coordinates,
    ] {
        assert_eq!(foreign.len(), coordinates.len());
        let other_flow = Rc::new(());
        assert!(validate_conditioning_origin(&other_flow, &origin).is_err());
    }
    Ok(())
}
