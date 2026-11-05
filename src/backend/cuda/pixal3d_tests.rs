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
        "U64" => t
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b) as f64)
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
            image_blocks: None,
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

struct DecoderGate {
    package: &'static str,
    cells: BTreeMap<CellId, Value>,
    anchors: BTreeMap<CellId, Value>,
    seen: BTreeSet<CellId>,
    defects: usize,
}
impl DecoderGate {
    fn surface(geometry_only: bool) -> Result<Self> {
        let bytes = std::fs::read(env_path("FF_PIXAL_FREEZE")?)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes))
                == "bc06c2394b677124a2041bacd29167d93a9a679e0205c4e229fd0afcefc0a3db",
            "M4 freeze integrity mismatch"
        );
        let freeze: Value = serde_json::from_slice(&bytes)?;
        ensure!(freeze["reference_valid"] == true, "invalid M4 reference");
        let mut cells = BTreeMap::new();
        let mut anchors = BTreeMap::new();
        for table in freeze["tables"].as_array().context("M4 tables missing")? {
            let family = table["family"].as_str().context("M4 family missing")?;
            if family == "slat" || (family == "geometry") != geometry_only {
                continue;
            }
            let table_name = table["source_table"]
                .as_str()
                .context("M4 table name missing")?;
            for c in table["cells"].as_array().context("M4 cells missing")? {
                let name = c["name"].as_str().context("M4 cell name missing")?;
                if name == "guided-texture-coordinate-identity"
                    || (family == "decoder-fixed" && name == "tex-fields")
                    || name.starts_with("normalization-")
                {
                    continue;
                }
                let key = id(table_name, name, c["key"].as_str());
                let mut anchor = c["anchor"].clone();
                if anchor.is_null() && c["sources"].is_object() {
                    anchor = c["sources"][c["comparison_anchor"]
                        .as_str()
                        .context("comparison arm missing")?]
                    .clone();
                }
                if family == "geometry" && c["sources"].is_array() {
                    let name = if let Some(suffix) = name.strip_prefix("compaction-") {
                        suffix.to_owned()
                    } else if name == "head-strict-edge-sign" {
                        "f64.flags".into()
                    } else {
                        let (case, field) =
                            name.rsplit_once('-').context("analytic field missing")?;
                        format!("{case}.{field}")
                    };
                    anchor = serde_json::json!({"path":c["sources"][0], "key":name});
                }
                if !anchor.is_null() {
                    anchors.insert(key.clone(), anchor);
                }
                ensure!(cells.insert(key, c.clone()).is_none(), "duplicate M4 cell");
            }
        }
        println!(
            "P1_M4A_FREEZE cells={} minimum_control_ratio={} unforced_topology=measurement reference_jaccard=0.5926020937353391",
            cells.len(),
            freeze["minimum_gated_control_ratio"]
        );
        Ok(Self {
            package: if geometry_only {
                "P1_M4A_GEOMETRY"
            } else {
                "P1_M4A_SHAPE"
            },
            cells,
            anchors,
            seen: BTreeSet::new(),
            defects: 0,
        })
    }
    fn exact_count(&mut self, key: CellId, tested: usize, mismatches: usize) -> Result<()> {
        ensure!(
            self.cells.contains_key(&key) && self.seen.insert(key.clone()),
            "unknown or repeated exact cell {key:?}"
        );
        ensure!(tested > 0, "empty exact support {key:?}");
        self.defects += usize::from(mismatches > 0);
        println!(
            "{}_EXACT {}",
            self.package,
            serde_json::json!({"table":key.0,"name":key.1,"tested":tested,"mismatches":mismatches,"result":if mismatches==0{"PASS"}else{"DEFECT"}})
        );
        Ok(())
    }
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
        Ok(Self {
            package: "P1_M3B",
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
        let reference = self.anchor(&k)?;
        self.score_against(k, shape, values, reference)
    }
    fn score_against(
        &mut self,
        k: CellId,
        shape: &[usize],
        values: &[f64],
        reference: HostTensor,
    ) -> Result<()> {
        ensure!(self.seen.insert(k.clone()), "duplicate score {k:?}");
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
                    "{}_METRIC {}",
                    self.package,
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
                "{}_EXACT {}",
                self.package,
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
            "unscored decoder cells: {:?}",
            self.cells
                .keys()
                .filter(|k| !self.seen.contains(*k))
                .collect::<Vec<_>>()
        );
        println!(
            "{}_GATE {} cells={} defects={} timing=false eager=true",
            self.package,
            if self.defects == 0 { "PASS" } else { "DEFECT" },
            self.seen.len(),
            self.defects
        );
        ensure!(self.defects == 0, "decoder candidate defects");
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
fn decoder_coordinates(path: &Path, key: &str) -> Result<Vec<[i32; 4]>> {
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

fn dense_decoder_cells(
    g: &mut DecoderGate,
    root: &Path,
    model: &Path,
    runtime: &Path,
) -> Result<()> {
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
                let key = id(&table, &name, if fixed { None } else { Some("logits") });
                if g.cells.contains_key(&key) {
                    g.score_f32(key, &[1, 1, 64, 64, 64], &logits)?;
                }
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
                let expected: Vec<_> = pooled
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| **value > 0.0)
                    .map(|(i, _)| {
                        [
                            0,
                            (i / 1024) as i32,
                            ((i / 32) % 32) as i32,
                            (i % 32) as i32,
                        ]
                    })
                    .collect();
                ensure!(
                    sites.coordinates() == expected,
                    "occupancy/maxpool/argwhere coordinate mismatch"
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
fn pointwise(g: &mut DecoderGate, decoder: &PixalDecoder) -> Result<()> {
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
fn sparse_ops(g: &mut DecoderGate, decoder: &PixalDecoder, root: &Path) -> Result<()> {
    let d = &decoder.device;
    let sites = SparseSites::new(vec![[0, 0, 0, 0]])?;
    let conv = SparseConv::new(d, &sites, 1, 27 * 6)?;
    let v = f16::from_f32(1.0 + 1.0 / 1024.0);
    let bias = f16::from_f32(1.0 / 2048.0);
    let mut weights = vec![f16::ZERO; 27];
    weights[13] = v;
    let weights = d.stream.clone_htod(&weights)?;
    let bias_device = d.stream.clone_htod(&[bias])?;
    let input = d.stream.clone_htod(&[v])?;
    let mut output = d.stream.alloc_zeros::<f16>(1)?;
    conv.run(
        &decoder.blas,
        &input,
        dense(&weights, d, [1, 27])?,
        dense(&bias_device, d, [1])?,
        &mut output,
    )?;
    let one_round = f16::from_f32(v.to_f32() * v.to_f32() + bias.to_f32());
    let two_round = f16::from_f32(f16::from_f32(v.to_f32() * v.to_f32()).to_f32() + bias.to_f32());
    ensure!(
        one_round != two_round && d.stream.clone_dtoh(&output)? == [one_round],
        "sparse bias rounded twice"
    );
    println!(
        "P1_M3B_BIAS PASS single_round={} double_round_control={} bit_exact=true",
        one_round.to_f32(),
        two_round.to_f32()
    );
    drop(conv);

    for c2s in [false, true] {
        let path = root.join(if c2s {
            "evidence/P1-M1-r1/sparse-c2s-r2/operator.safetensors"
        } else {
            "evidence/P1-M1-r1/sparse-conv-r2/operator.safetensors"
        });
        let sites = SparseSites::new(decoder_coordinates(&path, "coords")?)?;
        let n = sites.coordinates().len();
        let input = f32_tensor(&path, "features")?;
        let input: Vec<_> = input.into_iter().map(f16::from_f32).collect();
        let x = d.stream.clone_htod(&input)?;
        let co = if c2s { 4096 } else { 1024 };
        let live = n
            .checked_mul(co * size_of::<f16>())
            .context("sparse operator output overflow")?;
        let budget = d.workspace_budget(decoder.initial_free, decoder.device_budget, live)?;
        let conv = SparseConv::new(d, &sites, 1024, budget)?;
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
fn decoder_transitions(g: &mut DecoderGate, root: &Path) -> Result<()> {
    for (fixture, resolution, capture) in [
        ("cottage", 1536, "pipeline-single-1536-cottage-r0a"),
        ("robot_crab", 1408, "pipeline-single-1536-robot_crab-r0a"),
        ("mv4", 1024, "pipeline-mv-1024-mv4-r0a"),
    ] {
        let base = root.join("evidence/P1-M1-r1").join(capture);
        let sites = SparseSites::new(decoder_coordinates(
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
        let expected = decoder_coordinates(
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
fn prefix(
    g: &mut DecoderGate,
    decoder: &PixalDecoder,
    root: &Path,
    model: &Path,
    linked: Option<&[f32]>,
) -> Result<()> {
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
        if linked.is_some() && case != "cottage1024" {
            continue;
        }
        let base = root.join("evidence/P1-M1-r1").join(capture);
        let input = base.join("shape_slat-0-outputs.safetensors");
        let coords = decoder_coordinates(&input, "samples.coords")?;
        let mut latent = match linked {
            Some(values) => values.to_vec(),
            None => f32_tensor(&input, "samples.features")?,
        };
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
            guide_coords.push(decoder_coordinates(&path, "output.1.coords")?);
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
            if linked.is_some() {
                let anchor = g.anchor(&id(case, &format!("stage{stage}-features"), None))?;
                let values: Vec<_> = out.features.iter().map(|&x| x as f64).collect();
                let errors = metrics(&values, &anchor.values);
                let reference = g.anchor(&id(case, &format!("stage{stage}-coordinates"), None))?;
                let coordinates_equal = out.coordinates.as_flattened().iter().map(|&x| x as f64).eq(reference.values);
                println!("P1_M3B_LINKED_DIAGNOSTIC case={case} stage={stage} rows={} metrics={errors:?} coordinates_equal={coordinates_equal} recorded_guides=true", out.coordinates.len());
                return Ok(());
            }
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

fn linked_lr(root: &Path, model: &Path, runtime: &Path) -> Result<Vec<f32>> {
    let input = root
        .join("evidence/P1-M1-r1/pipeline-single-1024-cottage-r2/shape_slat-0-inputs.safetensors");
    let points = coordinates(&tensor(&input, "noise.coords")?)?;
    let mut flow = PixalFlow::open(
        &model.join("ckpts"),
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        runtime,
        &points,
        10_000_000_000,
    )?;
    let positive = flow.conditioning(
        &f32_tensor(&input, "cond.global")?,
        &f32_tensor(&input, "cond.proj.features")?,
        &coordinates(&tensor(&input, "cond.proj.coords")?)?,
    )?;
    let negative = flow.conditioning(
        &f32_tensor(&input, "neg_cond.global")?,
        &f32_tensor(&input, "neg_cond.proj.features")?,
        &coordinates(&tensor(&input, "neg_cond.proj.coords")?)?,
    )?;
    let noise = f32_tensor(&input, "noise.features")?;
    let latent = flow.sample(&noise, &positive, &negative)?;
    let times = schedule(true, flow.steps);
    let forwards = flow.steps
        + times[..flow.steps]
            .iter()
            .filter(|t| (0.6..=1.0).contains(*t))
            .count();
    println!(
        "P1_M3B_LINKED_LR rows={} steps={} forwards={forwards} finite={} diagnostics_only=true",
        points.len(),
        flow.steps,
        latent.iter().all(|x| x.is_finite())
    );
    Ok(latent)
}

#[test]
#[ignore = "requires a CUDA GPU, published checkpoints, and the frozen Pixal3D captures"]
fn pixal3d_decoders_match_frozen_definition_gpu() -> Result<()> {
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let mut g = DecoderGate::new()?;
    dense_decoder_cells(&mut g, &root, &model, &runtime)?;
    let decoder = PixalDecoder::open(&model, DecoderKind::Shape, &runtime, 10_000_000_000)?;
    pointwise(&mut g, &decoder)?;
    sparse_ops(&mut g, &decoder, &root)?;
    decoder_transitions(&mut g, &root)?;
    prefix(&mut g, &decoder, &root, &model, None)?;
    drop(decoder);
    let latent = linked_lr(&root, &model, &runtime)?;
    let decoder = PixalDecoder::open(&model, DecoderKind::Shape, &runtime, 10_000_000_000)?;
    prefix(&mut g, &decoder, &root, &model, Some(&latent))?;
    g.finish()
}

#[test]
fn structure_sites_keep_xyz_order() -> Result<()> {
    let mut logits = vec![-1.0; 8usize.pow(3)];
    logits[(8 + 3) * 8 + 6] = 1.0;
    logits[(5 * 8 + 7) * 8] = 2.0;
    let expected = vec![[0, 0, 1, 3], [0, 2, 3, 0]];
    let sites = PixalDecoder::structure_sites(&logits, 8, 4)?;
    assert_eq!(sites.coordinates(), expected);
    let swapped: Vec<_> = sites
        .coordinates()
        .iter()
        .map(|p| [p[0], p[3], p[2], p[1]])
        .collect();
    assert_eq!(swapped.len(), expected.len());
    assert_ne!(swapped, expected);
    let line = SparseSites::new((0..512).map(|x| [0, x, 0, 0]).collect())?;
    assert_eq!(line.round_unique(512, 96)?.coordinates().len(), 96);
    let (resolution, sites) = PixalDecoder::cascade_grid(&line, 1536, 96)?;
    assert_eq!((resolution, sites.coordinates().len()), (1408, 88));
    let (resolution, sites) = PixalDecoder::cascade_grid(&line, 1536, 88)?;
    assert_eq!((resolution, sites.coordinates().len()), (1280, 80));
    let (resolution, sites) = PixalDecoder::cascade_grid(&line, 1536, 1)?;
    assert_eq!((resolution, sites.coordinates().len()), (1024, 64));
    Ok(())
}

fn geometry_compaction(g: &mut DecoderGate, d: &Device, input: &Path) -> Result<()> {
    let sites = SparseSites::new(decoder_coordinates(input, "coords")?)?;
    let logits = f32_tensor(input, "logits")?;
    let masks: Vec<_> = logits
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| v.map(|x| x > 0.0))
        .collect();
    let (children, pairs) = sites.subdivide(&masks, sites.coordinates().len() * 8)?;
    g.score(
        id("m4-geometry-exact", "compaction-coords", None),
        &[children.coordinates().len(), 4],
        &children
            .coordinates()
            .as_flattened()
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>(),
    )?;
    for (name, key) in [
        ("compaction-offsets", "counts"),
        ("compaction-capacity_prefix", "capacity_counts"),
    ] {
        let values: Vec<u64> = tensor(input, key)?
            .values
            .into_iter()
            .map(|x| x as u64)
            .collect();
        let counts = d.stream.clone_htod(&values)?;
        let prefix = super::super::sparse::exclusive_scan(d, &counts)?;
        let actual = d.stream.clone_dtoh(&prefix)?;
        g.score(
            id("m4-geometry-exact", name, None),
            &[actual.len()],
            &actual.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        )?;
        if key == "capacity_counts" {
            ensure!(
                actual.last().copied().unwrap() > u32::MAX as u64,
                "wide-scan discriminator lost"
            );
        }
    }
    let rows = sites.coordinates().len();
    let values: Vec<_> = (0..rows * 128)
        .flat_map(|i| {
            let bits = ((((i * 29 + 7) % 251) as f32 - 125.0) / 127.0).to_bits();
            [
                f16::from_bits(bits as u16),
                f16::from_bits((bits >> 16) as u16),
            ]
        })
        .collect();
    let x = d.stream.clone_htod(&values)?;
    let pairs = d.stream.clone_htod(pairs.as_flattened())?;
    let y = d
        .stream
        .alloc_zeros::<f16>(children.coordinates().len() * 32)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("channel_to_spatial_f16")?)
            .arg(&x)
            .arg(&pairs)
            .arg(&y)
            .arg(&(rows as u32))
            .arg(&(children.coordinates().len() as u32))
            .arg(&32u32)
            .arg(&32u32)
            .arg(&0i32)
            .launch(flat(y.len()))?;
    }
    // Two adjacent F16 payloads preserve every bit of each F32 indexing fixture.
    let raw = d.stream.clone_dtoh(&y)?;
    let actual: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f32::from_bits(v[0].to_bits() as u32 | ((v[1].to_bits() as u32) << 16)))
        .collect();
    g.score_f32(
        id("m4-geometry-exact", "compaction-features", None),
        &[children.coordinates().len(), 16],
        &actual,
    )?;
    let swapped = tensor(input, "wrong_axis_coords")?;
    ensure!(
        swapped.values
            != children
                .coordinates()
                .as_flattened()
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>(),
        "axis control vanished"
    );
    Ok(())
}

fn geometry_heads(g: &mut DecoderGate, d: &Device, input: &Path) -> Result<()> {
    let fields = f32_tensor(input, "head_fields")?;
    let rows = fields.len() / 7;
    let values = d.stream.clone_htod(&fields)?;
    let dual = d.stream.alloc_zeros::<f32>(rows * 3)?;
    let flags = d.stream.alloc_zeros::<u8>(rows * 3)?;
    let weights = d.stream.alloc_zeros::<f32>(rows)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    unsafe {
        d.stream
            .launch_builder(&module.load_function("dual_grid_head_f32")?)
            .arg(&values)
            .arg(&(rows as u32))
            .arg(&dual)
            .arg(&flags)
            .arg(&weights)
            .launch(flat(rows))?;
        d.stream
            .launch_builder(&module.load_function("activation_f32")?)
            .arg(&weights)
            .arg(&weights)
            .arg(&(rows as u32))
            .arg(&1u32)
            .arg(&3i32)
            .launch(flat(rows))?;
    }
    let host_dual = d.stream.clone_dtoh(&dual)?;
    g.score_f32(
        id("m4-geometry-pointwise", "head-dual", None),
        &[rows, 3],
        &host_dual,
    )?;
    g.score_f32(
        id("m4-geometry-pointwise", "head-weight", None),
        &[rows, 1],
        &d.stream.clone_dtoh(&weights)?,
    )?;
    g.score(
        id("m4-geometry-exact", "head-strict-edge-sign", None),
        &[rows, 3],
        &d.stream
            .clone_dtoh(&flags)?
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<_>>(),
    )?;
    let all_coords = decoder_coordinates(input, "coords")?;
    let coordinates = d.stream.clone_htod(all_coords[..rows].as_flattened())?;
    for resolution in [64u32, 1024, 1152, 1280, 1408, 1536] {
        let vertices = d.stream.clone_htod(&host_dual)?;
        unsafe {
            d.stream
                .launch_builder(&module.load_function("sparse_grid_world_f32")?)
                .arg(&coordinates)
                .arg(&vertices)
                .arg(&(rows as u32))
                .arg(&resolution)
                .arg(&-0.5f32)
                .launch(flat(rows * 3))?;
        }
        let name = if resolution == 64 {
            "head-world".to_owned()
        } else {
            format!("head-world{resolution}")
        };
        g.score_f32(
            id("m4-geometry-pointwise", &name, None),
            &[rows, 3],
            &d.stream.clone_dtoh(&vertices)?,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA and frozen geometry fixtures"]
fn sparse_geometry_matches_frozen_definition_gpu() -> Result<()> {
    let mut g = DecoderGate::surface(true)?;
    let input = env_path("FF_PIXAL_TOPOLOGY")?;
    let d = Device::new(0)?;
    let (free, _) = d.ctx.mem_get_info()?;
    geometry_compaction(&mut g, &d, &input)?;
    geometry_heads(&mut g, &d, &input)?;
    for axis in 0..3 {
        for resolution in [8u32, 16] {
            for case in [
                "disconnected",
                "open_boundary",
                "tie",
                "positive_split",
                "negative_split",
                "degenerate",
            ] {
                let name = format!("axis{axis}-grid{resolution}-{case}");
                let vertex_key = id("m4-geometry-exact", &format!("{name}-vertices"), None);
                let path = PathBuf::from(
                    g.anchors[&vertex_key]["path"]
                        .as_str()
                        .context("analytic source missing")?,
                );
                let c = tensor(&path, &format!("{name}.coords"))?;
                let coordinates = c
                    .values
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|v| [0, v[0] as i32, v[1] as i32, v[2] as i32])
                    .collect();
                let sites = SparseSites::new(coordinates)?;
                let dual = d
                    .stream
                    .clone_htod(&f32_tensor(&path, &format!("{name}.dual"))?)?;
                let weights = d
                    .stream
                    .clone_htod(&f32_tensor(&path, &format!("{name}.weights"))?)?;
                let flags = tensor(&path, &format!("{name}.flags"))?
                    .values
                    .into_iter()
                    .map(|x| x as u8)
                    .collect::<Vec<_>>();
                let flags = d.stream.clone_htod(&flags)?;
                let mesh = super::super::sparse::mesh_from_dual(
                    &d,
                    sites,
                    dual,
                    flags,
                    weights,
                    resolution,
                    [free, free],
                )?;
                g.score_f32(vertex_key, &[mesh.vertices.len() / 3, 3], &mesh.vertices)?;
                g.score(
                    id("m4-geometry-exact", &format!("{name}-faces"), None),
                    &[mesh.faces.len() / 3, 3],
                    &mesh.faces.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                )?;
                if case != "degenerate" {
                    ensure!(
                        mesh.faces.iter().map(|&x| x as f64).collect::<Vec<_>>()
                            != tensor(&path, &format!("{name}.wrong_winding"))?.values,
                        "winding control vanished"
                    );
                }
            }
        }
    }
    let counts: Vec<u64> = (0..1_048_589)
        .map(|i| ((i * 7) % 1031) as u64 + if i % 4099 == 0 { 1u64 << 32 } else { 0 })
        .collect();
    let mut expected = Vec::with_capacity(counts.len() + 1);
    expected.push(0u64);
    for &v in &counts {
        expected.push(
            expected
                .last()
                .unwrap()
                .checked_add(v)
                .context("CPU prefix overflow")?,
        );
    }
    let prefix = super::super::sparse::exclusive_scan(&d, &d.stream.clone_htod(&counts)?)?;
    let actual = d.stream.clone_dtoh(&prefix)?;
    ensure!(actual == expected, "hierarchical wide prefix defect");
    ensure!(
        actual.iter().any(|&x| x != x as u32 as u64),
        "32-bit prefix control vanished"
    );
    println!(
        "P1_M4A_SCAN PASS elements={} levels=3 exact=true u32_control=true",
        counts.len()
    );
    let sites = SparseSites::new(vec![
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])?;
    let mut flags = vec![0u8; 21];
    flags[0] = 1;
    flags[12] = 1;
    let mesh = super::super::sparse::mesh_from_dual(
        &d,
        sites,
        d.stream.clone_htod(&[0.5f32; 21])?,
        d.stream.clone_htod(&flags)?,
        d.stream.clone_htod(&[1f32; 7])?,
        8,
        [free, free],
    )?;
    ensure!(
        mesh.faces == [0, 1, 2, 2, 1, 3],
        "cross-batch edge lookup defect"
    );
    let sites = SparseSites::new(vec![[0, 0, 0, 0]])?;
    let mesh = super::super::sparse::mesh_from_dual(
        &d,
        sites,
        d.stream.clone_htod(&[0.5f32; 3])?,
        d.stream.clone_htod(&[0u8; 3])?,
        d.stream.clone_htod(&[1f32])?,
        8,
        [free, free],
    )?;
    ensure!(
        mesh.faces.is_empty() && mesh.vertices.len() == 3,
        "empty edge set defect"
    );
    println!("P1_M4A_MESH_BOUNDARY PASS batch_isolation=true missing_corner=true empty_edges=true");
    g.finish()
}

fn surface_signs(
    g: &mut DecoderGate,
    case: &str,
    name: &str,
    values: &[f32],
    reference: &[f64],
) -> Result<()> {
    let key = id(&format!("m4-decisions-{case}.json"), name, None);
    let cell = &g.cells[&key];
    let margin = cell["margin_tau"].as_f64().context("sign margin missing")?;
    ensure!(values.len() == reference.len(), "sign extent mismatch");
    let mut selected = 0;
    let mut mismatches = 0;
    for (&value, &anchor) in values.iter().zip(reference) {
        ensure!(value.is_finite(), "nonfinite sign input");
        if anchor.abs() > margin {
            selected += 1;
            mismatches += usize::from((value > 0.0) != (anchor > 0.0));
        }
    }
    ensure!(
        Some(selected as u64) == cell["stable"].as_u64(),
        "frozen sign support mismatch"
    );
    g.exact_count(key, selected, mismatches)
}

fn surface_count(case: &str, phase: &str, plan: [usize; 3], rows: [usize; 2]) -> Result<()> {
    let [stage, parent_bytes, child_bytes] = plan;
    let channels = 1024 >> stage;
    let [parent, child] = rows;
    let parent_map = parent * 27 * size_of::<i32>();
    let child_map = child * 27 * size_of::<i32>();
    let parent_columns = parent_bytes
        .checked_sub(parent_map)
        .context("parent map bound")?;
    let child_columns = child_bytes
        .checked_sub(child_map)
        .context("child map bound")?;
    let parent_tile = parent_columns / (27 * channels * size_of::<f16>());
    let child_tile = child_columns / (27 * (channels / 2) * size_of::<f16>());
    ensure!(parent_tile > 0 && child_tile > 0, "empty gather tile");
    let calls =
        ([4, 16, 8, 4][stage] + 1) * parent.div_ceil(parent_tile) + child.div_ceil(child_tile);
    println!(
        "P1_M4A_COUNT {}",
        serde_json::json!({"case":case,"phase":phase,"stage":stage,"parent_rows":parent,"child_rows":child,"channels":channels,"parent_map_bytes":parent_map,"child_map_bytes":child_map,"parent_columns_bytes":parent_columns,"child_columns_bytes":child_columns,"parent_tile_rows":parent_tile,"child_tile_rows":child_tile,"gather_launches":calls,"bias_launches":calls,"gemm_dispatches":calls,"subdivision_d2h_bytes":parent*8*size_of::<f16>(),"child_pair_h2d_bytes":child*2*size_of::<u32>()})
    );
    Ok(())
}

fn surface_fixed(
    g: &mut DecoderGate,
    decoder: &PixalDecoder,
    root: &Path,
    case: (&str, &str, &str),
) -> Result<()> {
    let (case, capture, table) = case;
    let capture = if case == "crab1408" {
        format!("{capture}-decoder-r2")
    } else {
        capture.to_owned()
    };
    let base = root.join("evidence/P1-M1-r1").join(capture);
    let path = base.join("result.safetensors");
    let coordinates = decoder_coordinates(&path, "result.1.0.coords")?;
    let latent = f32_tensor(&path, "result.1.0.features")?;
    let topology = base.join("raw-geometry-material.safetensors");
    let mut guide_coords = Vec::new();
    let mut guide_logits = Vec::new();
    for stage in 0..4 {
        guide_coords.push(decoder_coordinates(
            &topology,
            &format!("subdivisions.{stage}.coords"),
        )?);
        guide_logits.push(f32_tensor(
            &topology,
            &format!("subdivisions.{stage}.features"),
        )?);
    }
    let guides: Vec<_> = guide_coords
        .iter()
        .zip(&guide_logits)
        .map(|(coordinates, logits)| SubdivisionGuide {
            coordinates,
            logits,
        })
        .collect();
    let (sites, features) = decoder.sparse_decode(
        coordinates,
        &latent,
        &guides,
        |plan @ [stage, _, _], (parents, sub), (next, _)| {
            surface_count(
                case,
                "fixed",
                plan,
                [parents.coordinates().len(), next.coordinates().len()],
            )?;
            let key = id(table, &format!("shape-subdivision{stage}"), None);
            g.score_f32(key.clone(), &[sub.len() / 8, 8], sub)?;
            let source = &g.cells[&key]["sources"]["r0a"];
            let anchor = tensor(
                Path::new(source["path"].as_str().context("sign anchor missing")?),
                source["key"].as_str().context("sign tensor missing")?,
            )?;
            surface_signs(g, case, &format!("subdivision{stage}"), sub, &anchor.values)?;
            println!(
                "P1_M4A_FIXED_PROGRESS case={case} stage={stage} rows={}",
                next.coordinates().len()
            );
            Ok(())
        },
    )?;
    let output = decoder.shape_head(&sites, features)?;
    let fields = decoder.device.stream.clone_dtoh(&output)?;
    let expected = decoder_coordinates(
        &base.join("shape_slat_decoder-fields.safetensors"),
        "output.coords",
    )?;
    ensure!(
        sites.coordinates() == expected,
        "forced shape coordinate identity mismatch"
    );
    let key = id(table, "shape-fields", None);
    g.score_f32(key.clone(), &[sites.coordinates().len(), 7], &fields)?;
    let source = &g.cells[&key]["sources"]["r0a"];
    let anchor = tensor(
        Path::new(source["path"].as_str().context("edge anchor missing")?),
        source["key"].as_str().context("edge key missing")?,
    )?;
    let reference: Vec<_> = anchor
        .values
        .as_chunks::<7>()
        .0
        .iter()
        .flat_map(|v| v[3..6].iter().copied())
        .collect();
    let candidate: Vec<_> = fields
        .as_chunks::<7>()
        .0
        .iter()
        .flat_map(|v| v[3..6].iter().copied())
        .collect();
    surface_signs(g, case, "edges", &candidate, &reference)?;
    println!(
        "P1_M4A_FIXED_COMPLETE case={case} rows={} sparse_convolutions=40",
        sites.coordinates().len()
    );
    Ok(())
}

fn surface_stable(
    g: &mut DecoderGate,
    case: &str,
    phase: &str,
    level: usize,
    parents: &SparseSites,
    logits: &[f32],
) -> Result<()> {
    let key = id(
        &format!("m4-stable-decisions-{case}"),
        &format!("{phase}-stage{level}"),
        None,
    );
    let cell = &g.cells[&key];
    let path = Path::new(
        cell["selection_fixture"]
            .as_str()
            .context("selection path missing")?,
    );
    let bytes = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == cell["selection_fixture_sha256"]
                .as_str()
                .context("selection digest missing")?,
        "selection integrity mismatch"
    );
    drop(bytes);
    let coordinates = decoder_coordinates(path, "coords")?;
    let selected = tensor(path, "selected")?;
    let signs = tensor(path, "signs")?;
    ensure!(
        selected.shape == [coordinates.len(), 8] && signs.shape == selected.shape,
        "invalid selection layout"
    );
    let mut tested = 0;
    let mut mismatches = 0;
    let mut missing = 0;
    let mut selected_parents = 0;
    for (i, &point) in coordinates.iter().enumerate() {
        let row = parents.row(point);
        let active = selected.values[i * 8..i * 8 + 8].iter().any(|&v| v != 0.0);
        selected_parents += usize::from(active);
        missing += usize::from(active && row.is_none());
        for bit in 0..8 {
            if selected.values[i * 8 + bit] != 0.0 {
                tested += 1;
                mismatches += usize::from(row.is_none_or(|row| {
                    (logits[row as usize * 8 + bit] > 0.0) != (signs.values[i * 8 + bit] != 0.0)
                }));
            }
        }
    }
    ensure!(
        cell["selected_decisions"].as_u64() == Some(tested as u64)
            && cell["selected_parents"].as_u64() == Some(selected_parents as u64)
            && cell["common_parents"].as_u64() == Some(coordinates.len() as u64),
        "stable support changed"
    );
    println!(
        "P1_M4A_STABLE_SUPPORT case={case} phase={phase} level={level} parents={} missing={missing}",
        coordinates.len()
    );
    g.exact_count(key, tested, mismatches)
}

fn surface_own(
    g: &mut DecoderGate,
    core: &str,
    phase: &str,
    level: usize,
    parents: &SparseSites,
    logits: &[f32],
    children: &SparseSites,
) -> Result<()> {
    let mut row = 0;
    let mut mismatches = 0;
    for (&parent, values) in parents.coordinates().iter().zip(logits.as_chunks::<8>().0) {
        for (bit, &value) in values.iter().enumerate() {
            if value > 0.0 {
                let expected = [
                    parent[0],
                    parent[1] * 2 + (bit as i32 & 1),
                    parent[2] * 2 + ((bit as i32 >> 1) & 1),
                    parent[3] * 2 + ((bit as i32 >> 2) & 1),
                ];
                mismatches += usize::from(children.coordinates().get(row) != Some(&expected));
                row += 1;
            }
        }
    }
    mismatches += children.coordinates().len().abs_diff(row);
    g.exact_count(
        id(
            &format!("m4-own-branch-contract-{core}"),
            &format!("{phase}-stage{level}-own-mask-compaction"),
            None,
        ),
        row.max(children.coordinates().len()),
        mismatches,
    )
}

fn surface_generated(
    g: &mut DecoderGate,
    decoder: &PixalDecoder,
    root: &Path,
    model: &Path,
    case: (&str, &str, u32),
) -> Result<()> {
    let (case, core, requested) = case;
    let base = root
        .join("evidence/P1-M4-freeze-r1")
        .join(format!("core-{core}-r0a-r2"));
    let metadata: Value = serde_json::from_slice(&std::fs::read(base.join("metadata.json"))?)?;
    let resolution = metadata["trace"]
        .as_array()
        .context("capture trace missing")?
        .iter()
        .find(|v| v["stage"] == "decode")
        .context("decode stage missing")?["actual_resolution"]
        .as_u64()
        .context("actual grid missing")? as u32;
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let norm = &config["args"]["shape_slat_normalization"];
    let mean: Vec<f32> = norm["mean"]
        .as_array()
        .context("shape mean missing")?
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let std: Vec<f32> = norm["std"]
        .as_array()
        .context("shape std missing")?
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let d = &decoder.device;
    for (call, phase) in ["LR", "HR"].into_iter().enumerate() {
        let input = base.join(format!("shape_slat-{call}-outputs.safetensors"));
        let coords = decoder_coordinates(&input, "samples.coords")?;
        let mut latent = d
            .stream
            .clone_htod(&f32_tensor(&input, "samples.features")?)?;
        denormalize_shape(d, &mut latent, &mean, &std, mean.len())?;
        let physical = d.stream.clone_dtoh(&latent)?;
        drop(latent);
        let (sites, features) = decoder.sparse_decode(coords, &physical, &[], |plan @ [level, _, _], (parents, logits), (next, _)| {
            surface_count(case, phase, plan, [parents.coordinates().len(), next.coordinates().len()])?;
            surface_stable(g, case, phase, level, parents, logits)?;
            surface_own(g, core, phase, level, parents, logits, next)?;
            let reference = base.join(format!("shape_slat_decoder.blocks.{level}.{}-call{call}-topology.safetensors", [4,16,8,4][level]));
            let reference = decoder_coordinates(&reference, "child_coords")?;
            let common = reference.iter().filter(|&&p| next.row(p).is_some()).count();
            let union = reference.len() + next.coordinates().len() - common;
            let disagreement = 1.0 - common as f64 / union as f64;
            println!("P1_M4A_COORDINATE_MEASUREMENT case={case} phase={phase} level={level} candidate_rows={} reference_rows={} jaccard_disagreement={disagreement} gating=false input=recorded_flow_output reference_pair_cottage1024=0.5926020937353391", next.coordinates().len(), reference.len());
            Ok(())
        })?;
        if phase == "LR" {
            drop(features);
            let (actual, quantized) = PixalDecoder::cascade_grid(&sites, requested, 49152)?;
            println!(
                "P1_M4A_CASCADE_MEASUREMENT case={case} requested={requested} actual={actual} rows={} recorded_actual={resolution} gating=false",
                quantized.coordinates().len()
            );
        } else {
            let fields = decoder.shape_head(&sites, features)?;
            let mesh = super::super::sparse::dual_grid_mesh(
                d,
                sites,
                fields,
                resolution,
                [decoder.initial_free, decoder.device_budget],
            )?;
            println!(
                "P1_M4A_RAW_MESH_MEASUREMENT case={case} actual_grid={resolution} vertices={} triangles={} gating=false scope=generated_decoder_on_recorded_flow",
                mesh.vertices.len() / 3,
                mesh.faces.len() / 3
            );
            ensure!(
                mesh.vertices.iter().all(|x| x.is_finite()),
                "nonfinite generated raw mesh"
            );
            ensure!(
                mesh.faces
                    .iter()
                    .all(|&x| x >= 0 && (x as usize) < mesh.vertices.len() / 3),
                "raw mesh index outside vertices"
            );
        }
    }
    Ok(())
}

fn surface_normalization(g: &mut DecoderGate, model: &Path, root: &Path) -> Result<()> {
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let d = Device::new(0)?;
    for (fixture, capture) in [
        ("cottage", "pipeline-single-1536-cottage-r0a"),
        ("robot_crab", "pipeline-single-1536-robot_crab-r0a"),
        ("mv4", "pipeline-mv-1024-mv4-r0a"),
    ] {
        for (kind, stage) in [("shape", "shape_slat-1"), ("tex", "tex_slat-0")] {
            let norm = &config["args"][format!("{kind}_slat_normalization")];
            let mean: Vec<_> = norm["mean"]
                .as_array()
                .context("normalization mean missing")?
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let std: Vec<_> = norm["std"]
                .as_array()
                .context("normalization scale missing")?
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let input = root
                .join("evidence/P1-M1-r1")
                .join(capture)
                .join(format!("{stage}-outputs.safetensors"));
            let host = tensor(&input, "samples.features")?;
            let mut values = d
                .stream
                .clone_htod(&host.values.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
            denormalize_shape(&d, &mut values, &mean, &std, mean.len())?;
            g.score_f32(
                id(
                    "floors-transition-published.json",
                    &format!("{fixture}-{kind}"),
                    None,
                ),
                &host.shape,
                &d.stream.clone_dtoh(&values)?,
            )?;
        }
    }
    Ok(())
}

fn canonical_quad(faces: &[i32]) -> Option<([i32; 4], [i32; 6])> {
    let mut sorted: [i32; 6] = faces.try_into().ok()?;
    sorted.sort_unstable();
    let mut key = [-1; 4];
    let mut count = 0;
    for (i, &v) in sorted.iter().enumerate() {
        if i == 0 || v != sorted[i - 1] {
            if count == 4 {
                return None;
            }
            key[count] = v;
            count += 1;
        }
    }
    if count != 4 {
        return None;
    }
    let mut triangles = [[0; 3]; 2];
    for (i, face) in faces.as_chunks::<3>().0.iter().enumerate() {
        let first = (0..3).min_by_key(|&j| face[j]).unwrap();
        triangles[i] = [face[first], face[(first + 1) % 3], face[(first + 2) % 3]];
    }
    triangles.sort_unstable();
    Some((key, triangles.as_flattened().try_into().unwrap()))
}

fn surface_mesh(g: &mut DecoderGate, root: &Path) -> Result<()> {
    let d = Device::new(0)?;
    let (free, _) = d.ctx.mem_get_info()?;
    let base = root.join("evidence/P1-M1-r1/pipeline-single-1024-cottage-r2");
    let path = base.join("shape_slat_decoder-fields.safetensors");
    let sites = SparseSites::new(decoder_coordinates(&path, "output.coords")?)?;
    let fields = f32_tensor(&path, "output.features")?;
    let flags: Vec<u8> = fields
        .as_chunks::<7>()
        .0
        .iter()
        .flat_map(|v| v[3..6].iter().map(|&x| u8::from(x > 0.0)))
        .collect();
    drop(fields);
    let mesh_path = root.join("evidence/P1-M1-r1/mesh-local/r0a.safetensors");
    let dual = d
        .stream
        .clone_htod(&f32_tensor(&mesh_path, "dual_vertices")?)?;
    let weights = d
        .stream
        .clone_htod(&f32_tensor(&mesh_path, "split_weight")?)?;
    let mesh = super::super::sparse::mesh_from_dual(
        &d,
        sites,
        dual,
        d.stream.clone_htod(&flags)?,
        weights,
        1024,
        [free, free],
    )?;
    g.score_f32(
        id("floors-mesh-published-r2.json", "vertices", None),
        &[mesh.vertices.len() / 3, 3],
        &mesh.vertices,
    )?;
    let reference = tensor(&mesh_path, "faces")?
        .values
        .into_iter()
        .map(|v| v as i32)
        .collect::<Vec<_>>();
    let weight = tensor(
        &root.join(
            "evidence/P1-M4-freeze-r1/replay/evidence/P1-M1-r1/mesh-local/r3math.safetensors",
        ),
        "split_weight",
    )?
    .values;
    let key = id(
        "floors-mesh-published-r2.json",
        "stable_split_topology",
        None,
    );
    let cell = &g.cells[&key];
    let margin = cell["margin_tau"].as_f64().context("mesh margin missing")?;
    let mut expected = Vec::new();
    for pair in reference.as_chunks::<6>().0 {
        let split = pair[3] == pair[0];
        let quad = [
            pair[0],
            pair[1],
            if split { pair[2] } else { pair[5] },
            if split { pair[5] } else { pair[2] },
        ];
        let delta = weight[quad[0] as usize] * weight[quad[2] as usize]
            - weight[quad[1] as usize] * weight[quad[3] as usize];
        if delta.abs() > margin {
            expected.push(canonical_quad(pair).context("invalid frozen quad")?);
        }
    }
    ensure!(
        cell["stable_quads"].as_u64() == Some(expected.len() as u64),
        "frozen mesh support mismatch"
    );
    let mut actual: Vec<_> = mesh
        .faces
        .as_chunks::<6>()
        .0
        .iter()
        .filter_map(|p| canonical_quad(p))
        .collect();
    let invalid = mesh.faces.len() / 6 - actual.len();
    expected.sort_unstable();
    actual.sort_unstable();
    let mut cursor = 0;
    let mut mismatches = invalid;
    for pair in &expected {
        while cursor < actual.len() && actual[cursor].0 < pair.0 {
            cursor += 1;
        }
        if actual.get(cursor) == Some(pair) {
            cursor += 1;
        } else {
            mismatches += 1;
        }
    }
    g.exact_count(key, expected.len(), mismatches)?;
    println!(
        "P1_M4A_MESH_SUPPORT total_quads={} stable_quads={} outside_stable={} canonical_winding=true multiplicity=true",
        reference.len() / 6,
        expected.len(),
        reference.len() / 6 - expected.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA, published checkpoints and frozen decoder captures"]
fn pixal3d_shape_mesh_matches_frozen_definition_gpu() -> Result<()> {
    let mut g = DecoderGate::surface(false)?;
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    dense_decoder_cells(&mut g, &root, &model, &runtime)?;
    decoder_transitions(&mut g, &root)?;
    surface_normalization(&mut g, &model, &root)?;
    surface_mesh(&mut g, &root)?;
    let decoder = PixalDecoder::open(&model, DecoderKind::Shape, &runtime, 22_000_000_000)?;
    for (case, capture, table, core, requested) in [
        (
            "cottage1024",
            "pipeline-single-1024-cottage-r2",
            "m4-floors-decoder-fixed-cottage-angular-r1.json",
            "single-1024-cottage",
            1024,
        ),
        (
            "crab1024",
            "pipeline-single-1024-robot_crab-r2",
            "m4-floors-decoder-fixed-robot_crab-angular-r1.json",
            "single-1024-robot_crab",
            1024,
        ),
        (
            "cottage1536",
            "pipeline-single-1536-cottage-r0a",
            "floors-decoder-cottage1536-full-m4.json",
            "single-1536-cottage",
            1536,
        ),
        (
            "crab1408",
            "pipeline-single-1536-robot_crab-r0a",
            "floors-decoder-crab1408-full-m4.json",
            "single-1536-robot_crab",
            1536,
        ),
        (
            "mv41024",
            "pipeline-mv-1024-mv4-r0a",
            "floors-decoder-mv41024-full-m4.json",
            "mv-1024-mv4",
            1024,
        ),
    ] {
        surface_fixed(&mut g, &decoder, &root, (case, capture, table))?;
        surface_generated(&mut g, &decoder, &root, &model, (case, core, requested))?;
    }
    g.finish()
}
