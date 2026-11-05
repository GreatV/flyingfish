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

struct OperationGate {
    root: PathBuf,
    kind: String,
    cells: BTreeMap<String, Value>,
    seen: BTreeSet<String>,
    defects: usize,
}
impl OperationGate {
    fn new(kind: &str, count: usize) -> Result<Self> {
        let root = env_path("FF_HI3D_FIXTURES")?;
        let freeze: Value =
            serde_json::from_slice(&std::fs::read(root.join(kind).join("FROZEN.json"))?)?;
        ensure!(
            freeze["all_declared_arms_present"] == true,
            "missing reference arm"
        );
        let cells: BTreeMap<String, Value> = serde_json::from_value(freeze["cells"].clone())?;
        ensure!(cells.len() == count, "frozen operation coverage mismatch");
        for cell in cells.values() {
            ensure!(
                cell["score_anchor"] == "sm89/anchor",
                "candidate anchor is not released F16"
            );
            for control in cell["controls"]
                .as_object()
                .context("frozen controls missing")?
                .values()
            {
                for metric in ["max_abs", "rmse"] {
                    ensure!(
                        control["ratios"][metric]
                            .as_f64()
                            .context("control ratio missing")?
                            >= 10.0,
                        "weak frozen control"
                    );
                }
            }
        }
        Ok(Self {
            root,
            kind: kind.into(),
            cells,
            seen: BTreeSet::new(),
            defects: 0,
        })
    }
    fn score(&mut self, name: &str, values: &[f64], control: bool) -> Result<()> {
        if !control {
            ensure!(self.seen.insert(name.into()), "duplicate operation score");
        }
        let cell = self.cells.get(name).context("cell outside frozen scope")?;
        let anchor = tensor(
            &self.root.join(&self.kind).join("anchor.safetensors"),
            &format!("{name}.anchor"),
        )?;
        let errors = metrics(values, &anchor.values)?;
        for metric in ["max_abs", "rmse"] {
            let bound = cell["bounds"][metric].as_f64().context("bound missing")?;
            let value = errors[metric];
            let defect = if control {
                value < 10.0 * bound
            } else {
                value > bound
            };
            self.defects += usize::from(defect);
            println!(
                "N5_M1_METRIC kind={} cell={name} metric={metric} value={value:.15e} bound={bound:.15e} control={control} result={}",
                self.kind,
                if defect { "DEFECT" } else { "PASS" }
            );
        }
        Ok(())
    }
    fn finish(self) -> Result<()> {
        ensure!(
            self.seen.len() == self.cells.len(),
            "operation cells unexecuted"
        );
        println!(
            "N5_M1_GATE {} kind={} cells={} defects={} timing=false",
            if self.defects == 0 { "PASS" } else { "DEFECT" },
            self.kind,
            self.seen.len(),
            self.defects
        );
        ensure!(self.defects == 0, "finite operation candidate defects");
        Ok(())
    }
}

fn selected_half(
    d: &Device,
    values: &CudaSlice<f16>,
    rows: &[usize],
    width: usize,
) -> Result<Vec<f64>> {
    let values = d.stream.clone_dtoh(values)?;
    ensure!(
        rows.iter().all(|&r| r < values.len() / width),
        "selected row outside output"
    );
    Ok(rows
        .iter()
        .flat_map(|&r| {
            values[r * width..(r + 1) * width]
                .iter()
                .map(|x| x.to_f64())
        })
        .collect())
}
fn frozen_rows(cell: &Value) -> Result<Vec<usize>> {
    cell["compared_rows"]
        .as_array()
        .context("compared rows missing")?
        .iter()
        .map(|v| usize::try_from(v.as_u64().context("invalid compared row")?).map_err(Into::into))
        .collect()
}
fn lattice(path: &Path, key: &str) -> Result<Vec<[i32; 3]>> {
    let t = tensor(path, key)?;
    ensure!(
        t.shape.len() == 2
            && t.shape[1] == 4
            && t.values.as_chunks::<4>().0.iter().all(|p| p[0] == 0.0),
        "single-sample lattice required"
    );
    Ok(t.values
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[1] as i32, p[2] as i32, p[3] as i32])
        .collect())
}

#[test]
#[ignore = "requires published flow and frozen fixtures; CUDA"]
fn hi3d_flow_components_gpu() -> Result<()> {
    let mut gate = OperationGate::new("components", 66)?;
    let root = env_path("FF_HI3D_MODEL")?;
    let common = gate.root.join("common.safetensors");
    let coordinates = lattice(&common, "input_blocks.1.updown.coordinates")?;
    let input_path = gate.root.join("component-inputs.safetensors");
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    for component in ["transformer_512", "transformer_1024"] {
        let flow = Flow::<f16>::open_hi3d(&root, component, &runtime, &coordinates, 8 << 30)?;
        for (name, cell) in gate.cells.clone() {
            if !name.starts_with(component) {
                continue;
            }
            let (_, rest) = name.split_once('.').context("component cell malformed")?;
            let (block, op) = rest
                .split_once('.')
                .context("component operation missing")?;
            let block = block
                .strip_prefix("block")
                .context("block prefix missing")?
                .parse::<usize>()?;
            let b = &flow.blocks[block];
            let input = tensor(&input_path, &format!("{name}.input"))?;
            let width = input.shape[1];
            let n = cell["input_shape"][0]
                .as_u64()
                .context("input rows missing")? as usize;
            let count = input.shape[0];
            let out = cell["shape"][1].as_u64().context("output width missing")? as usize;
            let selected: Vec<_> = (0..count)
                .map(|row| {
                    (0..count)
                        .find(|&i| (i * 127) % count == row)
                        .context("motif inverse missing")
                })
                .collect::<Result<_>>()?;
            for control in [false, true] {
                let values: Vec<_> = (0..n)
                    .flat_map(|r| {
                        let source = (r * 127) % count;
                        let values = &input.values;
                        (0..width).map(move |c| {
                            f16::from_f64(
                                values[source * width
                                    + if control {
                                        (c + width - 127 % width) % width
                                    } else {
                                        c
                                    }],
                            )
                        })
                    })
                    .collect();
                let x = flow.device.stream.clone_htod(&values)?;
                let y = flow.device.stream.alloc_zeros::<f16>(n * out)?;
                if op.starts_with("norm") {
                    flow.kernels.norm(
                        &flow.device,
                        ptr(&x, &flow.device),
                        ptr(&y, &flow.device),
                        [n, width],
                        if op == "norm2" {
                            [b.norm_weight, b.norm_bias]
                        } else {
                            [0, 0]
                        },
                        (1e-6, false),
                    )?;
                } else if op == "mlp.mlp.1" {
                    unsafe {
                        flow.device
                            .stream
                            .launch_builder(&flow.kernels.gelu)
                            .arg(&x)
                            .arg(&y)
                            .arg(&(n as u32))
                            .arg(&1u32)
                            .arg(&(width as u32))
                            .arg(&(width as u64))
                            .arg(&0u64)
                            .arg(&1u64)
                            .arg(&(width as u64))
                            .arg(&0u64)
                            .arg(&1u64)
                            .arg(&1i32)
                            .launch(grid((n * width).div_ceil(256).min(65535), 256))?;
                    }
                } else {
                    let linear = match op {
                        "self_attn.to_qkv" => b.qkv,
                        "self_attn.to_out" => b.self_out,
                        "cross_attn.to_q" => b.cross_q,
                        "cross_attn.to_kv" => b.cross_kv,
                        "cross_attn.to_out" => b.cross_out,
                        "mlp.mlp.0" => b.fc1,
                        "mlp.mlp.2" => b.fc2,
                        _ => anyhow::bail!("unsupported component operation {op}"),
                    };
                    linear.run(&flow.blas, n, ptr(&x, &flow.device), ptr(&y, &flow.device))?;
                }
                gate.score(
                    &name,
                    &selected_half(&flow.device, &y, &selected, out)?,
                    control,
                )?;
            }
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires published flow and frozen fixtures; CUDA"]
fn hi3d_flow_qk_rope_gpu() -> Result<()> {
    let mut gate = OperationGate::new("qk", 72)?;
    let root = env_path("FF_HI3D_MODEL")?;
    let common = gate.root.join("common.safetensors");
    let coordinates = lattice(&common, "input_blocks.1.updown.coordinates")?;
    let anchor = gate.root.join("qk/anchor.safetensors");
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    for component in ["transformer_512", "transformer_1024"] {
        let flow = Flow::<f16>::open_hi3d(&root, component, &runtime, &coordinates, 8 << 30)?;
        for (name, cell) in gate.cells.clone() {
            if !name.starts_with(component) {
                continue;
            }
            let fields: Vec<_> = name.split('.').collect();
            ensure!(fields.len() == 5, "QK cell malformed");
            let tag = fields[2];
            let role = fields[3];
            let mode = fields[4];
            let coords = tensor(&anchor, &format!("{tag}.coordinates"))?;
            ensure!(
                coords.shape.len() == 2 && coords.shape[1] == 3,
                "QK coordinate shape"
            );
            let n = coords.shape[0];
            let gain = cell["gamma_checkpoint_key"]
                .as_str()
                .context("gain key missing")?;
            let gamma = &flow._f32[gain];
            let values: Vec<_> = (0..n * 2112)
                .map(|i| {
                    f16::from_f32(
                        ((i * if role == "q" { 127 } else { 61 }) % 251) as f32 / 128.0
                            - 125.0 / 128.0,
                    )
                })
                .collect();
            let selected = frozen_rows(&cell)?;
            for control in [false, true] {
                let input = flow.device.stream.clone_htod(&values)?;
                let output = flow.device.stream.alloc_zeros::<f16>(values.len())?;
                let mut gamma_ptr = ptr(gamma, &flow.device);
                let modified = if control && mode == "norm" {
                    let values: Vec<_> = flow
                        .device
                        .stream
                        .clone_dtoh(gamma)?
                        .into_iter()
                        .map(|x| x + 1.0)
                        .collect();
                    Some(flow.device.stream.clone_htod(&values)?)
                } else {
                    None
                };
                if let Some(gain) = &modified {
                    gamma_ptr = ptr(gain, &flow.device);
                }
                if mode == "rope_norm" {
                    let sites = SparseSites::new(
                        coords
                            .values
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .map(|p| {
                                if control {
                                    [0, p[2] as i32, p[1] as i32, p[0] as i32]
                                } else {
                                    [0, p[0] as i32, p[1] as i32, p[2] as i32]
                                }
                            })
                            .collect(),
                    )?;
                    let phases = spatial_phases(&flow.device, &flow.kernels, &sites)?;
                    unsafe {
                        flow.device
                            .stream
                            .launch_builder(&flow.kernels.rope)
                            .arg(&input)
                            .arg(&phases)
                            .arg(&(n as i32))
                            .arg(&22i32)
                            .arg(&96i32)
                            .arg(&96i32)
                            .arg(&1i32)
                            .launch(flat(n * 22 * 48))?;
                    }
                }
                flow.kernels.l2(
                    &flow.device,
                    ptr(&input, &flow.device),
                    ptr(&output, &flow.device),
                    gamma_ptr,
                    n,
                    role == "k",
                )?;
                let values = flow.device.stream.clone_dtoh(&output)?;
                let mut compared = Vec::with_capacity(selected.len() * 2112);
                for &row in &selected {
                    for head in 0..22 {
                        for dim in 0..96 {
                            let at = if role == "k" {
                                (head * n + row) * 96 + dim
                            } else {
                                (row * 22 + head) * 96 + dim
                            };
                            compared.push(values[at].to_f64());
                        }
                    }
                }
                gate.score(&name, &compared, control)?;
            }
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires published flow and frozen fixtures; CUDA"]
fn hi3d_flow_blocks_gpu() -> Result<()> {
    let mut gate = OperationGate::new("blocks", 6)?;
    let root = env_path("FF_HI3D_MODEL")?;
    let common = gate.root.join("common.safetensors");
    let coordinates = lattice(&common, "input_blocks.1.updown.coordinates")?;
    let context = f32_tensor(&common, "context")?;
    let modulation: Vec<_> = f32_tensor(&common, "modulation")?
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    for component in ["transformer_512", "transformer_1024"] {
        let mut flow = Flow::<f16>::open_hi3d(&root, component, &runtime, &coordinates, 8 << 30)?;
        let conditioning = flow.conditioning(&context, &[], &coordinates)?;
        let modulation = flow.device.stream.clone_htod(&modulation)?;
        for index in [0, 12, 23] {
            let name = format!("{component}.block{index}");
            let rows = frozen_rows(&gate.cells[&name])?;
            let values: Vec<_> = f32_tensor(&common, &format!("blocks.{index}.features"))?
                .into_iter()
                .map(f16::from_f32)
                .collect();
            ensure!(
                values.len() == flow.work.hidden.len(),
                "block input shape mismatch"
            );
            for control in [false, true] {
                flow.work.hidden = flow.device.stream.clone_htod(&values)?;
                if control {
                    flow.block::<true>(index, ptr(&modulation, &flow.device), &conditioning)?;
                } else {
                    flow.block::<false>(index, ptr(&modulation, &flow.device), &conditioning)?;
                }
                gate.score(
                    &name,
                    &selected_half(&flow.device, &flow.work.hidden, &rows, 2112)?,
                    control,
                )?;
            }
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires published flow and frozen fixtures; CUDA"]
fn hi3d_flow_pool_inverse_gpu() -> Result<()> {
    let mut gate = OperationGate::new("pool", 2)?;
    let common = gate.root.join("common.safetensors");
    let coordinates = lattice(&common, "input_blocks.1.updown.coordinates")?;
    let root = env_path("FF_HI3D_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let flow = Flow::<f16>::open_hi3d(&root, "transformer_512", &runtime, &coordinates, 8 << 30)?;
    let io = flow.io.as_ref().context("pool IO missing")?;
    let input: Vec<_> = f32_tensor(&common, "input_blocks.1.updown.features")?
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let input = flow.device.stream.clone_htod(&input)?;
    let pooled = flow.device.stream.alloc_zeros::<f16>(2548 * 256)?;
    unsafe {
        flow.device
            .stream
            .launch_builder(&io.pool)
            .arg(&input)
            .arg(&io.offsets)
            .arg(&io.children)
            .arg(&pooled)
            .arg(&2548u32)
            .arg(&256u32)
            .launch(flat(pooled.len()))?;
    }
    let inverse = flow.device.stream.clone_dtoh(&io.inverse)?;
    let anchor_path = gate.root.join("pool/anchor.safetensors");
    ensure!(
        inverse
            .iter()
            .map(|&x| x as f64)
            .eq(tensor(&anchor_path, "unpool_inverse")?.values),
        "paired inverse differs from published row order"
    );
    let unpooled = flow.device.stream.alloc_zeros::<f16>(11295 * 256)?;
    io.copy(
        &flow.device,
        ptr(&pooled, &flow.device),
        ptr(&io.inverse, &flow.device),
        ptr(&unpooled, &flow.device),
        11295,
        256,
    )?;
    gate.score(
        "pool",
        &selected_half(&flow.device, &pooled, &(0..2548).collect::<Vec<_>>(), 256)?,
        false,
    )?;
    gate.score(
        "unpool",
        &selected_half(
            &flow.device,
            &unpooled,
            &(0..11295).collect::<Vec<_>>(),
            256,
        )?,
        false,
    )?;
    let rank: Vec<_> = tensor(&anchor_path, "wrong_xz_rank")?
        .values
        .into_iter()
        .map(|v| v as u32)
        .collect();
    for (name, index) in [
        ("pool", rank.clone()),
        (
            "unpool",
            inverse.iter().map(|&i| rank[i as usize]).collect(),
        ),
    ] {
        let index = flow.device.stream.clone_htod(&index)?;
        let output = flow.device.stream.alloc_zeros::<f16>(index.len() * 256)?;
        io.copy(
            &flow.device,
            ptr(&pooled, &flow.device),
            ptr(&index, &flow.device),
            ptr(&output, &flow.device),
            index.len(),
            256,
        )?;
        gate.score(
            name,
            &selected_half(
                &flow.device,
                &output,
                &(0..index.len()).collect::<Vec<_>>(),
                256,
            )?,
            true,
        )?;
    }
    gate.finish()
}

#[test]
#[ignore = "requires published flow and frozen fixtures; CUDA"]
fn hi3d_flow_sparse_convolutions_gpu() -> Result<()> {
    let mut gate = OperationGate::new("io-conv", 16)?;
    let root = env_path("FF_HI3D_MODEL")?;
    let common = gate.root.join("common.safetensors");
    let coordinates = lattice(&common, "input_blocks.1.updown.coordinates")?;
    let inputs = gate.root.join("io-inputs.safetensors");
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    for component in ["transformer_512", "transformer_1024"] {
        let flow = Flow::<f16>::open_hi3d(&root, component, &runtime, &coordinates, 8 << 30)?;
        for (name, cell) in gate.cells.clone() {
            let Some(op) = name.strip_prefix(&format!("{component}.")) else {
                continue;
            };
            let coords = lattice(&inputs, &format!("{op}.coordinates"))?;
            let sites = SparseSites::new(coords.iter().map(|p| [0, p[0], p[1], p[2]]).collect())?;
            let data = tensor(&inputs, &format!("{op}.features"))?;
            let width = data.shape[1];
            let output_width = cell["shape"][1]
                .as_u64()
                .context("conv output width missing")? as usize;
            let data: Vec<_> = data.values.into_iter().map(f16::from_f64).collect();
            let data = flow.device.stream.clone_htod(&data)?;
            let mut convolution = SparseConv::new(&flow.device, &sites, width, 128 << 20)?;
            let weight = dense(
                &flow._inner_weights[&format!("{op}.conv.weight")],
                &flow.device,
                [output_width as u32, (27 * width) as u32],
            )?;
            let bias = dense(
                &flow._inner_weights[&format!("{op}.conv.bias")],
                &flow.device,
                [output_width as u32],
            )?;
            let mut output = flow
                .device
                .stream
                .alloc_zeros::<f16>(coords.len() * output_width)?;
            let selected = frozen_rows(&cell)?;
            convolution.run(&flow.blas, &data, weight, bias, &mut output)?;
            gate.score(
                &name,
                &selected_half(&flow.device, &output, &selected, output_width)?,
                false,
            )?;
            let maps: Vec<[i32; 27]> = sites
                .coordinates()
                .iter()
                .map(|p| {
                    std::array::from_fn(|tap| {
                        sites
                            .row([
                                p[0],
                                p[3] + (tap / 9) as i32 - 1,
                                p[2] + ((tap / 3) % 3) as i32 - 1,
                                p[1] + (tap % 3) as i32 - 1,
                            ])
                            .unwrap_or(-1)
                    })
                })
                .collect();
            convolution.replace_neighbours(&maps)?;
            convolution.run(&flow.blas, &data, weight, bias, &mut output)?;
            gate.score(
                &name,
                &selected_half(&flow.device, &output, &selected, output_width)?,
                true,
            )?;
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires frozen Heun common inputs; CUDA"]
fn hi3d_heun_updates_gpu() -> Result<()> {
    let mut gate = OperationGate::new("heun", 50)?;
    let source = gate.root.join("trajectory.safetensors");
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(gate.root.join("trajectory.json"))?)?;
    let schedule = heun_schedule(11295, 25)?;
    let device = Device::new(0)?;
    let kernel = cubin::module(&device.ctx, "flow")?.load_function("heun_update_f32")?;
    let selected: Vec<_> = tensor(&source, "selected_rows")?
        .values
        .into_iter()
        .map(|v| v as usize)
        .collect();
    ensure!(selected.len() == 31, "Heun recorded rows mismatch");
    for (step, &(time, next)) in schedule.iter().enumerate() {
        let first_time = metadata["steps"][step]["first_time"]
            .as_f64()
            .context("first time missing")? as f32;
        let next_time = metadata["steps"][step]["next_time"]
            .as_f64()
            .context("next time missing")? as f32;
        ensure!(
            time == first_time && next == next_time,
            "published schedule mismatch at {step}: {time}/{first_time} {next}/{next_time}"
        );
        let state = if step == 0 {
            let noise = f32_tensor(&source, "noise")?;
            selected
                .iter()
                .flat_map(|&r| noise[r * 32..(r + 1) * 32].iter().copied())
                .collect()
        } else {
            f32_tensor(&source, &format!("step{}.updated", step - 1))?
        };
        let first = f32_tensor(&source, &format!("step{step}.first_velocity"))?;
        let second = f32_tensor(&source, &format!("step{step}.second_velocity"))?;
        let expand = |x: &[f32]| -> Result<_> {
            ensure!(x.len() == 31 * 32, "Heun recorded extent mismatch");
            let values: Vec<_> = (0..11295)
                .flat_map(|i| {
                    x[(i * 127 % 31) * 32..(i * 127 % 31 + 1) * 32]
                        .iter()
                        .copied()
                })
                .collect();
            Ok(device.stream.clone_htod(&values)?)
        };
        let state = expand(&state)?;
        let first = expand(&first)?;
        let second = expand(&second)?;
        let output = device.stream.alloc_zeros::<f32>(11295 * 32)?;
        for (kind, second_ptr, control) in [
            ("predictor", 0, false),
            ("updated", ptr(&second, &device), false),
            ("updated", ptr(&first, &device), true),
        ] {
            if control && step != 12 {
                continue;
            }
            unsafe {
                device
                    .stream
                    .launch_builder(&kernel)
                    .arg(&state)
                    .arg(&first)
                    .arg(&second_ptr)
                    .arg(&output)
                    .arg(&((11295 * 32) as u64))
                    .arg(&(time - next))
                    .launch(flat(11295 * 32))?;
            }
            let values = device.stream.clone_dtoh(&output)?;
            let rows: Vec<_> = (0..31)
                .map(|r| (0..31).find(|&i| i * 127 % 31 == r).unwrap())
                .collect();
            let compact: Vec<_> = rows
                .iter()
                .flat_map(|&r| values[r * 32..(r + 1) * 32].iter().map(|&v| v as f64))
                .collect();
            ensure!(
                (0..11295).all(
                    |i| values[i * 32..(i + 1) * 32]
                        .iter()
                        .map(|&v| v as f64)
                        .eq(compact[(i * 127 % 31) * 32..(i * 127 % 31 + 1) * 32]
                            .iter()
                            .copied())
                ),
                "pointwise production compression differs"
            );
            gate.score(&format!("step{step}.{kind}"), &compact, control)?;
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires published flow and recorded conditioning/noise; CUDA"]
fn hi3d_flow_fixed_coordinates_heun_gpu() -> Result<()> {
    let fixtures = env_path("FF_HI3D_FIXTURES")?;
    let root = env_path("FF_HI3D_MODEL")?;
    let source = fixtures.join("trajectory.safetensors");
    let coordinates = lattice(&source, "coordinates")?;
    let noise = f32_tensor(&source, "noise")?;
    let context = f32_tensor(&fixtures.join("common.safetensors"), "context")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let mut flow =
        Flow::<f16>::open_hi3d(&root, "transformer_512", &runtime, &coordinates, 8 << 30)?;
    let positive = flow.conditioning(&context, &[], &coordinates)?;
    let negative = flow.conditioning(&vec![0.0; context.len()], &[], &coordinates)?;
    let result = flow.sample_heun(&noise, &positive, &negative)?;
    let reference = tensor(&source, "final_latents")?;
    let repeat = tensor(
        &fixtures.join("trajectory-repeat.safetensors"),
        "final_latents",
    )?;
    let candidate: Vec<f64> = if reference.values.len() == result.len() {
        result.iter().map(|&v| v as f64).collect()
    } else {
        let selected = tensor(&source, "selected_rows")?;
        selected
            .values
            .iter()
            .flat_map(|&r| {
                result[(r as usize) * 32..(r as usize + 1) * 32]
                    .iter()
                    .map(|&v| v as f64)
            })
            .collect()
    };
    let errors = metrics(&candidate, &reference.values)?;
    let variation = metrics(&repeat.values, &reference.values)?;
    println!(
        "N5_M1_TRAJECTORY MEASUREMENT updates=25 coordinates={} channels=32 native_max_abs={:.15e} native_rmse={:.15e} source_repeat_max_abs={:.15e} source_repeat_rmse={:.15e} equivalence_gate=false timing=false",
        coordinates.len(),
        errors["max_abs"],
        errors["rmse"],
        variation["max_abs"],
        variation["rmse"]
    );
    Ok(())
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
    flow: &mut Flow,
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
fn fixed_ss(flow: &mut Flow, gate: &mut Gate, root: &Path, variant: &str) -> Result<()> {
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
                flow.block::<false>(block, ptr(&modulation, &flow.device), &condition)?;
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
        let mut flow = Flow::open(
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
            let mut flow = Flow::open(
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
    flow: &Flow,
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
fn attention_ss(flow: &mut Flow, gate: &mut Gate, root: &Path, variant: &str) -> Result<()> {
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
        let mut flow = Flow::open(
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
    cells: BTreeMap<CellId, Value>,
    anchors: BTreeMap<CellId, Value>,
    seen: BTreeSet<CellId>,
    defects: usize,
}
impl DecoderGate {
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
    let mut flow = Flow::open(
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
    Ok(())
}
