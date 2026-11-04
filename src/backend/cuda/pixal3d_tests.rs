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
    let tensors = checkpoint.component_tensors()?;
    let t = tensors
        .get(key)
        .with_context(|| format!("{} lacks {key}", path.display()))?;
    let values = match t.dtype.as_str() {
        "F32" => t
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect(),
        "F64" => t
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect(),
        "BF16" => t
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b).to_f64())
            .collect(),
        "F16" => t
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes(*b).to_f64())
            .collect(),
        "I32" => t
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f64)
            .collect(),
        "I64" => t
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f64)
            .collect(),
        "BOOL" => t.data.iter().map(|&x| x as f64).collect(),
        dtype => anyhow::bail!("unsupported reference dtype {dtype}"),
    };
    Ok(HostTensor {
        shape: t.shape.clone(),
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
        let bytes = std::fs::read(env_path("FF_PIXAL_FREEZE")?)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes))
                == "231d9f8eba81398b442c6eebfbee947c1d9e43c9df4b983e11a538d75dc47e76",
            "M3 freeze integrity mismatch"
        );
        let freeze: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            freeze["reference_valid"] == true,
            "invalid reference freeze"
        );
        let selected: BTreeSet<_> = freeze["split"]["M3a"]
            .as_array()
            .context("M3a split missing")?
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
        for table in freeze["tables"]
            .as_array()
            .context("freeze tables missing")?
        {
            let name = table["source_table"]
                .as_str()
                .context("table source missing")?;
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
        let mut minimum = f64::INFINITY;
        for c in freeze["controls"].as_array().context("controls missing")? {
            let k = id(
                c["table"].as_str().context("control table missing")?,
                c["name"].as_str().context("control name missing")?,
                c["key"].as_str(),
            );
            if selected.contains(&k) {
                ensure!(c["valid"] == true, "invalid frozen control");
                if let Some(ratios) = c["ratios"].as_object() {
                    for (metric, value) in ratios {
                        if c["gating"][metric] == true {
                            let ratio = value.as_f64().context("control ratio missing")?;
                            ensure!(ratio >= 10.0, "weak frozen control");
                            minimum = minimum.min(ratio);
                        }
                    }
                }
                anchors.insert(k, c["anchor"].clone());
            }
        }
        ensure!(
            cells.len() == 764 && anchors.len() == 764,
            "M3a frozen coverage mismatch"
        );
        println!(
            "P1_M3A_FROZEN cells=764 minimum_control_ratio={minimum:.15} thresholds_unchanged=true"
        );
        Ok(Self {
            cells,
            anchors,
            seen: BTreeSet::new(),
            defects: 0,
        })
    }
    fn score(&mut self, key: CellId, shape: &[usize], candidate: &[f64]) -> Result<()> {
        ensure!(
            self.seen.insert(key.clone()),
            "duplicate frozen cell score: {key:?}"
        );
        let cell = self
            .cells
            .get(&key)
            .context("cell outside M3a frozen split")?;
        let anchor = &self.anchors[&key];
        let reference = tensor(
            Path::new(anchor["path"].as_str().context("anchor path missing")?),
            anchor["key"].as_str().context("anchor key missing")?,
        )?;
        ensure!(
            shape == reference.shape && candidate.len() == reference.values.len(),
            "candidate/anchor layout mismatch {key:?}: {shape:?} {:?}",
            reference.shape
        );
        if let Some(taus) = cell["tau"].as_object() {
            let values = metrics(candidate, &reference.values)?;
            for (name, tau) in taus {
                let tau = tau.as_f64().context("invalid frozen tau")?;
                let value = values[name.as_str()];
                let gating = cell["frozen_gating_metrics"][name] == true;
                let defect = gating && value > tau;
                self.defects += usize::from(defect);
                println!(
                    "P1_M3A_METRIC {}",
                    serde_json::json!({"table":key.0,"name":key.1,"key":key.2,"metric":name,"value":value,"tau":tau,"gating":gating,"result":if defect {"DEFECT"} else {"PASS"}})
                );
            }
        } else {
            ensure!(
                candidate.iter().all(|x| x.is_finite()),
                "non-finite exact cell"
            );
            let mismatches = candidate
                .iter()
                .zip(&reference.values)
                .filter(|(a, b)| a != b)
                .count();
            let gating = cell["frozen_gating_metrics"]["exact"] == true;
            self.defects += usize::from(gating && mismatches > 0);
            println!(
                "P1_M3A_EXACT {}",
                serde_json::json!({"table":key.0,"name":key.1,"mismatches":mismatches,"gating":gating})
            );
        }
        Ok(())
    }
    fn score_f32(&mut self, key: CellId, shape: &[usize], values: &[f32]) -> Result<()> {
        self.score(
            key,
            shape,
            &values.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        )
    }
    fn finish(self) -> Result<()> {
        let missing: Vec<_> = self
            .cells
            .keys()
            .filter(|k| !self.seen.contains(*k))
            .collect();
        ensure!(missing.is_empty(), "unscored frozen cells: {missing:?}");
        println!(
            "P1_M3A_GATE {} cells={} defects={} timing=false eager=true",
            if self.defects == 0 { "PASS" } else { "DEFECT" },
            self.seen.len(),
            self.defects
        );
        ensure!(
            self.defects == 0,
            "finite candidate defects: {}",
            self.defects
        );
        Ok(())
    }
}

#[test]
fn sampler_schedule_preserves_cfg_branch_counts() {
    for (sparse, expected) in [(false, 22), (true, 21)] {
        let steps = 12;
        let times = schedule(sparse, steps);
        assert_eq!(times[0], 1.0);
        assert_eq!(times[steps], 0.0);
        assert!(times.windows(2).all(|t| t[0] > t[1]));
        assert_eq!(
            times[..steps]
                .iter()
                .map(|t| 1 + usize::from((0.6..=1.0).contains(t)))
                .sum::<usize>(),
            expected
        );
    }
    assert!(!(0.6..=1.0).contains(&0.599));
    assert!((0.6..=1.0).contains(&0.6));
}

fn download_bf(d: &Device, values: &CudaSlice<bf16>) -> Result<Vec<f32>> {
    Ok(d.stream
        .clone_dtoh(values)?
        .into_iter()
        .map(bf16::to_f32)
        .collect())
}
fn upload_bf(d: &Device, values: &[f32]) -> Result<CudaSlice<bf16>> {
    Ok(d.stream.clone_htod(
        &values
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
    )?)
}
fn ss_coordinates() -> Vec<[i32; 3]> {
    (0..4096)
        .map(|i| [i / 256, (i / 16) % 16, i % 16])
        .collect()
}
fn coordinates(t: &HostTensor) -> Result<Vec<[i32; 3]>> {
    ensure!(
        t.shape.len() == 2 && t.shape[1] == 4 && t.values.len() == t.shape[0] * 4,
        "sparse coordinates must be stored [N,4]"
    );
    t.values
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            ensure!(
                p[0] == 0.0
                    && p.iter()
                        .all(|&v| v >= 0.0 && v <= i32::MAX as f64 && v.fract() == 0.0),
                "invalid captured coordinate"
            );
            Ok([p[1] as i32, p[2] as i32, p[3] as i32])
        })
        .collect()
}
fn sampler_cells(gate: &mut Gate) -> Result<()> {
    let d = Device::new(0)?;
    for (sparse, kind, shape) in [
        (false, "sparse_structure", vec![1, 8, 16, 16, 16]),
        (true, "shape_slat", vec![20749, 32]),
    ] {
        let count = shape.iter().product();
        let noise: Vec<_> = (0..count)
            .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let positive: Vec<_> = (0..count)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 9.0)
            .collect();
        let negative: Vec<_> = (0..count)
            .map(|i| ((i * 13 % 23) as f32 - 11.0) / 13.0)
            .collect();
        for text in ["1.0", "0.9", "0.6", "0.599"] {
            let t: f64 = text.parse()?;
            let mut sampler = Sampler::new(&d, &noise)?;
            d.stream.memcpy_htod(&positive, &mut sampler.positive)?;
            d.stream.memcpy_htod(&negative, &mut sampler.negative)?;
            sampler.step(&d, t, t - 0.037, sparse)?;
            gate.score_f32(
                id(
                    "floors-sampler-published.json",
                    &format!("{kind}-{text}"),
                    None,
                ),
                &shape,
                &d.stream.clone_dtoh(&sampler.state)?,
            )?;
        }
    }
    Ok(())
}
struct FlowCase<'a> {
    table: &'a str,
    fixture: Option<&'a str>,
    shape: Vec<usize>,
}
impl FlowCase<'_> {
    fn cell(&self, name: &str) -> CellId {
        match self.fixture {
            Some(f) => id(self.table, f, Some(name)),
            None => id(self.table, name, None),
        }
    }
}
fn flow_cells(
    flow: &mut PixalFlow,
    gate: &mut Gate,
    case: FlowCase<'_>,
    noise: &[f32],
    positive: &Conditioning,
    negative: &Conditioning,
) -> Result<()> {
    let sparse = flow.spec.sparse();
    let steps = flow.steps;
    let times = schedule(sparse, steps);
    flow.set_times(
        &times[..steps]
            .iter()
            .map(|t| (1000.0 * t) as f32)
            .collect::<Vec<_>>(),
    )?;
    let mut sampler = Sampler::new(&flow.device, noise)?;
    let mut call = 0;
    for step in 0..steps {
        for (index, condition) in [positive, negative].into_iter().enumerate() {
            if index == 1 && !(0.6..=1.0).contains(&times[step]) {
                continue;
            }
            let output = if index == 0 {
                &mut sampler.positive
            } else {
                &mut sampler.negative
            };
            flow.forward_observe(&sampler.state, step, condition, output, |f, block| {
                if sparse && [0, 1, 10, 11, 20].contains(&call) && [0, 15, 29].contains(&block) {
                    gate.score_f32(
                        case.cell(&format!("call{call}-blocks-block{block}")),
                        &[f.rows, 1536],
                        &download_bf(&f.device, &f.work.hidden)?,
                    )?;
                }
                Ok(())
            })?;
            gate.score_f32(
                case.cell(&format!("velocity{call}")),
                &case.shape,
                &flow.device.stream.clone_dtoh(output)?,
            )?;
            call += 1;
        }
        sampler.step(&flow.device, times[step], times[step + 1], sparse)?;
        gate.score_f32(
            case.cell(&format!("state{step}")),
            &case.shape,
            &flow.device.stream.clone_dtoh(&sampler.state)?,
        )?;
    }
    ensure!(
        call == if sparse { 21 } else { 22 },
        "CFG branch call count mismatch"
    );
    gate.score_f32(
        case.cell("final"),
        &case.shape,
        &flow.device.stream.clone_dtoh(&sampler.state)?,
    )?;
    println!(
        "P1_M3A_FLOW_COMPLETE table={} fixture={:?} forwards={call} steps={steps}",
        case.table, case.fixture
    );
    Ok(())
}
fn fixed_ss(flow: &mut PixalFlow, gate: &mut Gate, root: &Path, variant: &str) -> Result<()> {
    let base = root.join(format!("evidence/P1-M1-r1/ss-operators-{variant}"));
    let table = format!("floors-ss-fixed-{variant}-angular-r1.json");
    for fixture in ["cottage", "robot_crab"] {
        for call in [0, 1, 12, 13, 21] {
            for block in [0, 15, 29] {
                let name = format!("{fixture}-call{call}-block{block}");
                let input = base.join(format!("{name}-inputs.safetensors"));
                let hidden = f32_tensor(&input, "h")?;
                flow.device.stream.memcpy_htod(
                    &hidden.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
                    &mut flow.work.hidden,
                )?;
                let modulation = upload_bf(&flow.device, &f32_tensor(&input, "t")?)?;
                let condition = flow.conditioning(
                    &f32_tensor(&input, "global_cond")?,
                    &f32_tensor(&input, "proj_cond")?,
                    &flow.coordinates,
                )?;
                flow.block(block, ptr(&modulation, &flow.device), &condition)?;
                gate.score_f32(
                    id(&table, &name, None),
                    &[1, 4096, 1536],
                    &download_bf(&flow.device, &flow.work.hidden)?,
                )?;
                for (part, kind) in ["q", "k"].into_iter().enumerate() {
                    let name = format!("{name}-{kind}_rms_norm");
                    let input = upload_bf(
                        &flow.device,
                        &f32_tensor(&base.join(format!("{name}-mixed.safetensors")), "input")?,
                    )?;
                    flow.kernels.l2(
                        &flow.device,
                        ptr(&input, &flow.device),
                        ptr(&flow.work.q, &flow.device),
                        flow.blocks[block].gamma[part],
                        4096,
                        false,
                    )?;
                    gate.score_f32(
                        id(&table, &name, None),
                        &[1, 4096, 12, 128],
                        &download_bf(&flow.device, &flow.work.q)?,
                    )?;
                }
            }
        }
    }
    Ok(())
}
fn ss_flows(gate: &mut Gate, model: &Path, runtime: &Path, root: &Path) -> Result<()> {
    let conditioning = env_path("FF_PIXAL_FREEZE")?
        .parent()
        .context("freeze parent missing")?
        .join("recovered/conditioning-ss-r2");
    for variant in ["single", "mv"] {
        let stem = format!(
            "ss_flow_img_dit_1_3B_64_bf16{}",
            if variant == "mv" { "_mv" } else { "" }
        );
        let mut flow = PixalFlow::open(
            &model.join("ckpts"),
            &stem,
            runtime,
            &ss_coordinates(),
            10_000_000_000,
        )?;
        fixed_ss(&mut flow, gate, root, variant)?;
        attention_ss(&mut flow, gate, root, variant)?;
        let mut initial = Vec::new();
        for fixture in ["cottage", "robot_crab"] {
            let noise = f32_tensor(&root.join("fixtures/pixal3d/ss-noise.safetensors"), fixture)?;
            let input = conditioning.join(format!("{fixture}.safetensors"));
            let positive = flow.conditioning(
                &f32_tensor(&input, "global")?,
                &f32_tensor(&input, "projection")?,
                &flow.coordinates,
            )?;
            let negative =
                flow.conditioning(&vec![0.0; 5120], &vec![0.0; 4096 * 1024], &flow.coordinates)?;
            flow.set_times(&[1000.0])?;
            let x = flow.device.stream.clone_htod(&noise)?;
            let mut out = flow.device.stream.alloc_zeros(noise.len())?;
            flow.forward(&x, 0, &positive, &mut out)?;
            initial.push(flow.device.stream.clone_dtoh(&out)?);
            flow_cells(
                &mut flow,
                gate,
                FlowCase {
                    table: &format!("floors-ss-{variant}.json"),
                    fixture: Some(fixture),
                    shape: vec![8, 16, 16, 16],
                },
                &noise,
                &positive,
                &negative,
            )?;
        }
        let input = conditioning.join("cottage.safetensors");
        let positive = flow.conditioning(
            &f32_tensor(&input, "global")?,
            &f32_tensor(&input, "projection")?,
            &flow.coordinates,
        )?;
        let noise = f32_tensor(
            &root.join("fixtures/pixal3d/ss-noise.safetensors"),
            "cottage",
        )?;
        flow.set_times(&[1000.0])?;
        let x = flow.device.stream.clone_htod(&noise)?;
        let mut out = flow.device.stream.alloc_zeros(noise.len())?;
        flow.forward(&x, 0, &positive, &mut out)?;
        ensure!(
            initial[0] == flow.device.stream.clone_dtoh(&out)? && initial[0] != initial[1],
            "two-sample workspace isolation failed"
        );
        println!(
            "P1_M3A_ISOLATION PASS variant={variant} order=A_B_A bit_exact=true two_outputs_differ=true"
        );
    }
    Ok(())
}
fn shape_flows(gate: &mut Gate, model: &Path, runtime: &Path, root: &Path) -> Result<()> {
    for (variant, resolution, fixture, stages) in [
        ("single", 1024, "cottage", vec![0, 1]),
        ("single", 1024, "robot_crab", vec![0, 1]),
        ("single", 1536, "cottage", vec![1]),
        ("single", 1536, "robot_crab", vec![1]),
        ("mv", 1024, "mv4", vec![0, 1]),
    ] {
        let arm = if variant == "single" && resolution == 1024 {
            "r2"
        } else {
            "r0a"
        };
        let base = root.join(format!(
            "evidence/P1-M1-r1/pipeline-{variant}-{resolution}-{fixture}-{arm}"
        ));
        for stage in stages {
            let input = base.join(format!("shape_slat-{stage}-inputs.safetensors"));
            let coords = tensor(&input, "noise.coords")?;
            let points = coordinates(&coords)?;
            for key in ["cond.proj.coords", "neg_cond.proj.coords"] {
                ensure!(
                    tensor(&input, key)?.values == coords.values,
                    "captured conditioning coordinate alignment differs"
                );
            }
            let stem = format!(
                "slat_flow_img2shape_dit_1_3B_{}_bf16{}",
                if stage == 0 { 512 } else { 1024 },
                if variant == "mv" { "_mv" } else { "" }
            );
            let mut flow = PixalFlow::open(
                &model.join("ckpts"),
                &stem,
                runtime,
                &points,
                10_000_000_000,
            )?;
            let global = f32_tensor(&input, "cond.global")?;
            let projected = f32_tensor(&input, "cond.proj.features")?;
            let aligned = coordinates(&tensor(&input, "cond.proj.coords")?)?;
            let positive = flow.conditioning(&global, &projected, &aligned)?;
            let mut wrong = aligned.clone();
            wrong.rotate_left(1);
            ensure!(
                flow.conditioning(&global, &projected, &wrong).is_err(),
                "misaligned projection rows were accepted"
            );
            drop(global);
            drop(projected);
            let negative = flow.conditioning(
                &f32_tensor(&input, "neg_cond.global")?,
                &f32_tensor(&input, "neg_cond.proj.features")?,
                &coordinates(&tensor(&input, "neg_cond.proj.coords")?)?,
            )?;
            let actual = if resolution == 1536 && fixture == "robot_crab" {
                1408
            } else {
                resolution
            };
            let table = format!(
                "floors-slat-{variant}-{actual}-{fixture}-{}-angular-r1.json",
                if stage == 0 { "lr" } else { "hr" }
            );
            gate.score(
                id(&table, "coordinate_alignment", None),
                &coords.shape,
                &flow
                    .coordinates
                    .iter()
                    .flat_map(|p| [0.0, p[0] as f64, p[1] as f64, p[2] as f64])
                    .collect::<Vec<_>>(),
            )?;
            flow_cells(
                &mut flow,
                gate,
                FlowCase {
                    table: &table,
                    fixture: None,
                    shape: vec![points.len(), 32],
                },
                &f32_tensor(&input, "noise.features")?,
                &positive,
                &negative,
            )?;
        }
    }
    Ok(())
}

fn head_major(values: &[f32], rows: usize) -> Vec<f32> {
    (0..12)
        .flat_map(|head| {
            (0..rows).flat_map(move |row| {
                (0..128).map(move |channel| values[(row * 12 + head) * 128 + channel])
            })
        })
        .collect()
}
fn row_major(values: &[f32], rows: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..12).flat_map(move |head| {
                (0..128).map(move |channel| values[(head * rows + row) * 128 + channel])
            })
        })
        .collect()
}
fn rotate(
    flow: &PixalFlow,
    values: &CudaSlice<bf16>,
    phases: &CudaSlice<f32>,
    head_major: bool,
) -> Result<()> {
    let kernel = if head_major {
        &flow.kernels.rope_hm
    } else {
        &flow.kernels.rope
    };
    unsafe {
        flow.device
            .stream
            .launch_builder(kernel)
            .arg(values)
            .arg(phases)
            .arg(&(flow.rows as i32))
            .arg(&12i32)
            .arg(&128i32)
            .arg(&128i32)
            .launch(flat(flow.rows * 12 * 64))?;
    }
    Ok(())
}
impl Gate {
    fn control(&self, key: CellId, values: &[f32]) -> Result<()> {
        let cell = &self.cells[&key];
        let anchor = &self.anchors[&key];
        let reference = tensor(
            Path::new(
                anchor["path"]
                    .as_str()
                    .context("control anchor path missing")?,
            ),
            anchor["key"]
                .as_str()
                .context("control anchor key missing")?,
        )?;
        let distances = metrics(
            &values.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &reference.values,
        )?;
        for (metric, tau) in cell["tau"]
            .as_object()
            .context("control needs float cell")?
        {
            if cell["frozen_gating_metrics"][metric] != true {
                continue;
            }
            let ratio = distances[metric.as_str()] / tau.as_f64().context("control tau missing")?;
            println!(
                "P1_M3A_NATIVE_CONTROL table={} name={} metric={metric} ratio={ratio}",
                key.0, key.1
            );
            ensure!(
                ratio >= 10.0,
                "native phase-index control fails to separate at 10 tau"
            );
        }
        Ok(())
    }
}
fn attention_ss(flow: &mut PixalFlow, gate: &mut Gate, root: &Path, variant: &str) -> Result<()> {
    let table = "floors-attention-index-published.json";
    let n = 4096;
    let base = root.join(format!("evidence/P1-M1-r1/ss-operators-{variant}"));
    let values: Vec<_> = (0..n)
        .flat_map(|r| {
            (0..12).flat_map(move |h| {
                (0..128).map(move |c| {
                    ((r * 7 + h * 11 + c * 3) % 17) as f32 / 16.0 - 8.0 / 16.0
                        + ((r * 13 + h * 5 + c * 7) % 19) as f32 / 32.0
                        - 9.0 / 32.0
                })
            })
        })
        .collect();
    let values = head_major(&values, n);
    flow.device.stream.memcpy_htod(
        &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        &mut flow.work.v,
    )?;
    for block in [0, 15, 29] {
        let prefix = format!("cottage-call0-block{block}");
        let q = f32_tensor(
            &base.join(format!("{prefix}-q_rms_norm-mixed.safetensors")),
            "output",
        )?;
        let k = f32_tensor(
            &base.join(format!("{prefix}-k_rms_norm-mixed.safetensors")),
            "output",
        )?;
        flow.device.stream.memcpy_htod(
            &q.iter().copied().map(bf16::from_f32).collect::<Vec<_>>(),
            &mut flow.work.q,
        )?;
        flow.device.stream.memcpy_htod(
            &head_major(&k, n)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            &mut flow.work.kn,
        )?;
        rotate(flow, &flow.work.q, &flow.phases, false)?;
        rotate(flow, &flow.work.kn, &flow.phases, true)?;
        gate.score_f32(
            id(table, &format!("{variant}-{prefix}-q"), None),
            &[1, n, 12, 128],
            &download_bf(&flow.device, &flow.work.q)?,
        )?;
        gate.score_f32(
            id(table, &format!("{variant}-{prefix}-k"), None),
            &[1, n, 12, 128],
            &row_major(&download_bf(&flow.device, &flow.work.kn)?, n),
        )?;
        flow.flash.run(
            &flow.work.q,
            1536,
            &flow.work.kn,
            &flow.work.v,
            &mut flow.work.attention,
            Shape {
                rows: n,
                start: 0,
                capacity: n,
                q_heads: 12,
                kv_heads: 12,
                head_dim: 128,
                causal: false,
                window_left: None,
            },
        )?;
        gate.score_f32(
            id(table, &format!("{variant}-{prefix}-attention"), None),
            &[1, n, 12, 128],
            &download_bf(&flow.device, &flow.work.attention)?,
        )?;
        if block == 0 && variant == "single" {
            let phases = flow.device.stream.clone_dtoh(&flow.phases)?;
            let wrong: Vec<_> = ss_coordinates()
                .iter()
                .flat_map(|p| {
                    let other = (p[2] * 256 + p[1] * 16 + p[0]) as usize;
                    phases[other * 128..(other + 1) * 128].iter().copied()
                })
                .collect();
            let wrong = flow.device.stream.clone_htod(&wrong)?;
            let q = upload_bf(&flow.device, &q)?;
            rotate(flow, &q, &wrong, false)?;
            gate.control(
                id(table, &format!("{variant}-{prefix}-q"), None),
                &download_bf(&flow.device, &q)?,
            )?;
        }
    }
    Ok(())
}
fn selected_rows(values: &[f32], rows: usize) -> Vec<f32> {
    (0..127)
        .flat_map(|i| {
            let row = i * 131 % rows;
            values[row * 1536..(row + 1) * 1536].iter().copied()
        })
        .collect()
}
fn sparse_attention(gate: &mut Gate, model: &Path, runtime: &Path, root: &Path) -> Result<()> {
    let table = "floors-sparse-attention-published.json";
    for (resolution, fixture, stage) in [
        (1024, "cottage", 0),
        (1024, "robot_crab", 1),
        (1536, "robot_crab", 1),
    ] {
        let arm = if resolution == 1024 { "r2" } else { "r0a" };
        let input=root.join(format!("evidence/P1-M1-r1/pipeline-single-{resolution}-{fixture}-{arm}/shape_slat-{stage}-inputs.safetensors"));
        let coordinates = coordinates(&tensor(&input, "noise.coords")?)?;
        let n = coordinates.len();
        let mut flow = PixalFlow::open(
            &model.join("ckpts"),
            "slat_flow_img2shape_dit_1_3B_1024_bf16",
            runtime,
            &coordinates,
            10_000_000_000,
        )?;
        for (part, kind) in ["q", "k"].into_iter().enumerate() {
            let values: Vec<_> = (0..n)
                .flat_map(|r| {
                    (0..12).flat_map(move |h| {
                        (0..128).map(move |c| {
                            if part == 0 {
                                (((r * 7 + h * 11 + c * 3) % 17) as f32 - 8.0) / 8.0
                                    + (((r * 13 + h * 5 + c * 7) % 19) as f32 - 9.0) / 16.0
                            } else {
                                (((r * 11 + h * 3 + c * 7) % 23) as f32 - 11.0) / 8.0
                            }
                        })
                    })
                })
                .collect();
            let input = upload_bf(&flow.device, &values)?;
            let output = if part == 0 {
                &flow.work.q
            } else {
                &flow.work.kn
            };
            flow.kernels.l2(
                &flow.device,
                ptr(&input, &flow.device),
                ptr(output, &flow.device),
                flow.blocks[15].gamma[part],
                n,
                part == 1,
            )?;
            let norm = download_bf(&flow.device, output)?;
            let norm = if part == 1 { row_major(&norm, n) } else { norm };
            gate.score_f32(
                id(table, &format!("{n}-{kind}_norm"), None),
                &[127, 12, 128],
                &selected_rows(&norm, n),
            )?;
            rotate(&flow, output, &flow.phases, part == 1)?;
            let rotated = download_bf(&flow.device, output)?;
            let rotated = if part == 1 {
                row_major(&rotated, n)
            } else {
                rotated
            };
            gate.score_f32(
                id(table, &format!("{n}-{kind}_rope"), None),
                &[127, 12, 128],
                &selected_rows(&rotated, n),
            )?;
        }
        let values: Vec<_> = (0..n)
            .flat_map(|r| {
                (0..12).flat_map(move |h| {
                    (0..128).map(move |c| (((r * 5 + h * 7 + c * 11) % 29) as f32 - 14.0) / 16.0)
                })
            })
            .collect();
        flow.device.stream.memcpy_htod(
            &head_major(&values, n)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            &mut flow.work.v,
        )?;
        let q = upload_bf(
            &flow.device,
            &selected_rows(&download_bf(&flow.device, &flow.work.q)?, n),
        )?;
        let mut output = flow.device.stream.alloc_zeros::<bf16>(127 * 1536)?;
        flow.flash.run_masked(super::super::flash::Query {
            q: &q,
            q_row_stride: 1536,
            k: &flow.work.kn,
            v: &flow.work.v,
            out: &mut output,
            keys: n,
            prefix: None,
            shape: Shape {
                rows: 127,
                start: 0,
                capacity: n,
                q_heads: 12,
                kv_heads: 12,
                head_dim: 128,
                causal: false,
                window_left: None,
            },
        })?;
        gate.score_f32(
            id(table, &format!("{n}-attention"), None),
            &[127, 12, 128],
            &download_bf(&flow.device, &output)?,
        )?;
    }
    Ok(())
}

fn transitions(gate: &mut Gate, model: &Path, root: &Path) -> Result<()> {
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let normalization = &config["args"]["shape_slat_normalization"];
    let coefficients = |name: &str| -> Result<Vec<f32>> {
        normalization[name]
            .as_array()
            .context("normalization missing")?
            .iter()
            .map(|v| Ok(v.as_f64().context("normalization value invalid")? as f32))
            .collect()
    };
    let channels = FlowSpec::read(
        &model.join("ckpts"),
        "slat_flow_img2shape_dit_1_3B_1024_bf16",
    )?
    .out_channels;
    let mean = coefficients("mean")?;
    let std = coefficients("std")?;
    let d = Device::new(0)?;
    for (variant, resolution, fixture) in [
        ("single", 1536, "cottage"),
        ("single", 1536, "robot_crab"),
        ("mv", 1024, "mv4"),
    ] {
        let input=root.join(format!("evidence/P1-M1-r1/pipeline-{variant}-{resolution}-{fixture}-r0a/shape_slat-1-outputs.safetensors"));
        let host = tensor(&input, "samples.features")?;
        let mut values = d
            .stream
            .clone_htod(&host.values.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
        denormalize_shape(&d, &mut values, &mean, &std, channels)?;
        gate.score_f32(
            id(
                "floors-transition-published.json",
                &format!("{fixture}-shape"),
                None,
            ),
            &host.shape,
            &d.stream.clone_dtoh(&values)?,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires admitted GPU, published checkpoints and frozen P1 M3 reference captures"]
fn pixal3d_flows_match_frozen_definition_gpu() -> Result<()> {
    let mut gate = Gate::new()?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    sampler_cells(&mut gate)?;
    transitions(&mut gate, &model, &root)?;
    ss_flows(&mut gate, &model, &runtime, &root)?;
    sparse_attention(&mut gate, &model, &runtime, &root)?;
    shape_flows(&mut gate, &model, &runtime, &root)?;
    gate.finish()
}
