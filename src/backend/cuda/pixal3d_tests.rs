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
fn half_tensor(path: &Path, key: &str) -> Result<Vec<f16>> {
    let checkpoint = MappedCheckpoint::open_component(
        path.parent().context("feature bank parent missing")?,
        path.file_stem()
            .and_then(|v| v.to_str())
            .context("feature bank stem missing")?,
    )?;
    let t = checkpoint.tensor(key)?;
    ensure!(
        t.dtype() == Dtype::F16,
        "feature bank must retain F16 storage"
    );
    Ok(t.data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f16::from_le_bytes(*v))
        .collect())
}
fn indices(value: &Value) -> Result<Vec<usize>> {
    value
        .as_array()
        .context("selected rows missing")?
        .iter()
        .map(|v| usize::try_from(v.as_u64().context("invalid selected row")?).map_err(Into::into))
        .collect()
}
fn selected_values<T: DeviceRepr + Copy>(
    d: &Device,
    values: &CudaSlice<T>,
    rows: &[usize],
    width: usize,
    convert: impl Fn(T) -> f64,
) -> Result<Vec<f64>> {
    ensure!(
        width > 0
            && values.len().is_multiple_of(width)
            && rows.iter().all(|&r| r < values.len() / width),
        "selected rows outside feature bank"
    );
    let mut output = Vec::with_capacity(rows.len() * width);
    for &row in rows {
        output.extend(
            d.stream
                .clone_dtoh(&values.slice(row * width..(row + 1) * width))?
                .into_iter()
                .map(&convert),
        );
    }
    Ok(output)
}
fn bank_sites(path: &Path) -> Result<SparseSites> {
    let grid = tensor(path, "grid_coord")?;
    let batch = tensor(path, "batch")?;
    ensure!(
        grid.shape == [batch.values.len(), 3],
        "feature bank grid/batch extent mismatch"
    );
    SparseSites::new(
        grid.values
            .as_chunks::<3>()
            .0
            .iter()
            .zip(batch.values)
            .map(|(p, b)| [b as i32, p[0] as i32, p[1] as i32, p[2] as i32])
            .collect(),
    )
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
    decoder_anchor: Option<PathBuf>,
    decoder_key: Option<String>,
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
            decoder_anchor: None,
            decoder_key: None,
        })
    }
    fn decoder(kind: &str, freeze_dir: &str, anchor_dir: &str, count: usize) -> Result<Self> {
        let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
        let freeze: Value =
            serde_json::from_slice(&std::fs::read(root.join(freeze_dir).join("FROZEN.json"))?)?;
        ensure!(
            freeze["reference_protocol"] == "PASS",
            "decoder reference protocol is invalid"
        );
        let cells: BTreeMap<String, Value> = serde_json::from_value(freeze["cells"].clone())?;
        ensure!(
            cells.len() == count
                && cells
                    .values()
                    .all(|cell| cell["reference_protocol"] == "PASS"),
            "decoder cell coverage or protocol mismatch"
        );
        let decoder_anchor = Some(root.join(anchor_dir).join("anchor.safetensors"));
        Ok(Self {
            root,
            kind: kind.into(),
            cells,
            seen: BTreeSet::new(),
            defects: 0,
            decoder_anchor,
            decoder_key: None,
        })
    }
    fn score(&mut self, name: &str, values: &[f64], control: bool) -> Result<()> {
        if !control {
            ensure!(self.seen.insert(name.into()), "duplicate operation score");
        }
        self.measure(name, values, control)
    }
    fn replay(&mut self, name: &str, values: &[f64], mode: &str) -> Result<()> {
        ensure!(self.seen.contains(name), "unscored operation replay");
        println!("N5_M2B_MODE cell={name} mode={mode}");
        self.measure(name, values, false)
    }
    fn measure(&mut self, name: &str, values: &[f64], control: bool) -> Result<()> {
        let cell = self.cells.get(name).context("cell outside frozen scope")?;
        let anchor = if let Some(path) = &self.decoder_anchor {
            if let Some(source) = cell["source_cell"].as_str() {
                let source = tensor(path, source)?;
                let width = source.shape[1];
                let rows = cell["selector"]["rows"]
                    .as_array()
                    .context("selector rows missing")?;
                let mut values = Vec::with_capacity(rows.len() * width);
                for row in rows {
                    let row = usize::try_from(row.as_u64().context("invalid selector row")?)?;
                    ensure!(row < source.shape[0], "selector exceeds anchor rows");
                    values.extend_from_slice(&source.values[row * width..(row + 1) * width]);
                }
                HostTensor {
                    shape: vec![rows.len(), width],
                    values,
                }
            } else {
                tensor(path, self.decoder_key.as_deref().unwrap_or(name))?
            }
        } else {
            tensor(
                &self.root.join(&self.kind).join("anchor.safetensors"),
                &format!("{name}.anchor"),
            )?
        };
        let errors = metrics(values, &anchor.values)?;
        for metric in ["max_abs", "rmse"] {
            let bound = cell[if self.decoder_anchor.is_some() {
                "tau"
            } else {
                "bounds"
            }][metric]
                .as_f64()
                .context("bound missing")?;
            let value = errors[metric];
            if !control && let Some(bound) = cell["original_tau"][metric].as_f64() {
                println!(
                    "N5_M2B_ORIGINAL_FIELD_METRIC cell={name} metric={metric} value={value:.15e} bound={bound:.15e} result={}",
                    if value > bound { "DEFECT" } else { "PASS" }
                );
            }
            let defect = if control {
                if self.decoder_anchor.is_some() {
                    value <= 10.0 * bound
                } else {
                    value < 10.0 * bound
                }
            } else {
                value > bound
            };
            self.defects += usize::from(defect);
            println!(
                "{}_METRIC kind={} cell={name} metric={metric} value={value:.15e} bound={bound:.15e} control={control} result={}",
                if self.decoder_anchor.is_some() {
                    "N5_M2B"
                } else {
                    "N5_M1"
                },
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
            "{}_GATE {} kind={} cells={} defects={} timing=false",
            if self.decoder_anchor.is_some() {
                "N5_M2B"
            } else {
                "N5_M1"
            },
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
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_sparse_convolution_definition_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_MODEL")?;
    let checkpoint = DecoderCheckpoint::open(&root.join("vae"), DecoderKind::PointTransformer)?;
    let reference = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let common = reference.join("sm89-r1/additional-common-inputs.safetensors");
    let sites = SparseSites::new(decoder_coordinates(&common, "conv.coordinates")?)?;
    let wrong_maps = sites
        .coordinates()
        .iter()
        .map(|&p| {
            std::array::from_fn(|tap| {
                let query = [
                    p[0],
                    p[3] + (tap / 9) as i32 - 1,
                    p[2] + ((tap / 3) % 3) as i32 - 1,
                    p[1] + (tap % 3) as i32 - 1,
                ];
                sites.row(query).unwrap_or(-1)
            })
        })
        .collect::<Vec<[i32; 27]>>();
    let d = Device::new(0)?;
    let ops = Ops::new(&d.ctx)?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let blas = Blas::new(
        &d,
        &ops,
        &runtime,
        Rc::new(Closure::new(false, "sparse convolution".into())),
    )?;
    let n = sites.coordinates().len();
    let mut defects = 0;
    let mut results = Vec::new();
    for tile in [127, 251] {
        let mut gate =
            OperationGate::decoder("sparse-convolution", "conv-freeze-r1", "sm89-r1", 54)?;
        for (name, cell) in gate.cells.clone() {
            let definition = &cell["definition"];
            let ci = definition["input_shape"][1]
                .as_u64()
                .context("input width missing")? as usize;
            let co = definition["stored_weight_shape"][0]
                .as_u64()
                .context("output width missing")? as usize;
            let module = name
                .strip_prefix("conv.")
                .context("conv prefix missing")?
                .rsplit_once('.')
                .context("feature dtype missing")?
                .0;
            let weight = checkpoint.mapped.tensor(&format!("{module}.weight"))?;
            let weight: Vec<_> = weight
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let weight = d.stream.clone_htod(&weight)?;
            let bias = checkpoint.mapped.tensor(&format!("{module}.bias"))?;
            let bias: Vec<_> = bias
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let bias = d.stream.clone_htod(&bias)?;
            let rows: Vec<_> = definition["sampled_output_rows"]
                .as_array()
                .context("sample rows missing")?
                .iter()
                .map(|v| v.as_u64().context("invalid sample row").map(|v| v as usize))
                .collect::<Result<_>>()?;
            let input = f32_tensor(&common, &format!("conv.c{ci}.input"))?;
            let workspace = n * 27 * size_of::<i32>() + tile * (27 * ci + co) * size_of::<f32>();
            let mut conv = SparseConv::<f32>::promoted(&d, &sites, [ci, co], workspace)?;
            for control in [false, true] {
                if control {
                    conv.replace_neighbours(&wrong_maps)?;
                }
                let weight = matrix(ptr(&weight, &d), co, 27 * ci)?;
                let bias = view(&bias, &d.stream, 0, [co as u32], [1])?;
                let values = if name.ends_with(".float16") {
                    let input: Vec<_> = input.iter().map(|&x| f16::from_f32(x)).collect();
                    let input = d.stream.clone_htod(&input)?;
                    let output = d.stream.alloc_zeros::<f16>(n * co)?;
                    conv.run_promoted(
                        &blas,
                        matrix::<f16>(ptr(&input, &d), n, ci)?,
                        weight,
                        bias,
                        &mut TensorMut(matrix::<f16>(ptr(&output, &d), n, co)?),
                    )?;
                    selected_half(&d, &output, &rows, co)?
                } else {
                    let input = d.stream.clone_htod(&input)?;
                    let output = d.stream.alloc_zeros::<f32>(n * co)?;
                    conv.run_promoted(
                        &blas,
                        matrix::<f32>(ptr(&input, &d), n, ci)?,
                        weight,
                        bias,
                        &mut TensorMut(matrix::<f32>(ptr(&output, &d), n, co)?),
                    )?;
                    let output = d.stream.clone_dtoh(&output)?;
                    rows.iter()
                        .flat_map(|&r| output[r * co..(r + 1) * co].iter().map(|&x| x as f64))
                        .collect()
                };
                gate.score(&name, &values, control)?;
            }
        }
        defects += gate.defects;
        println!(
            "N5_M2B_CONV_TILE tile={tile} cells={} defects={} bounded=true stored_layout=KRSC input_rows={n}",
            gate.seen.len(),
            gate.defects
        );
        results.push(gate.finish());
    }
    for result in results {
        result?;
    }
    println!(
        "PASS N5_M2B_SPARSE_CONV_GATE cells=54 tiles=2 executions=108 control_executions=108 defects={defects} eager=true timing=false"
    );
    Ok(())
}

fn normalized_features<
    I: DeviceRepr + Copy + 'static,
    O: DeviceRepr + Copy + ValidAsZeroBits + 'static,
>(
    d: &Device,
    plan: &crate::backend::cuda::sparse::BatchGroupNorm,
    input: Vec<I>,
    affine: [&CudaSlice<f32>; 2],
    shape: [usize; 2],
    selected: &[usize],
    convert: impl Fn(O) -> f64,
) -> Result<Vec<f64>> {
    let [n, c] = shape;
    let x = d.stream.clone_htod(&input)?;
    drop(input);
    let output = d.stream.alloc_zeros::<O>(n * c)?;
    plan.run::<I, O>(
        matrix(ptr(&x, d), n, c)?,
        [
            view(affine[0], &d.stream, 0, [c as u32], [1])?,
            view(affine[1], &d.stream, 0, [c as u32], [1])?,
        ],
        &mut TensorMut(matrix(ptr(&output, d), n, c)?),
    )?;
    ensure!(
        selected.iter().all(|&row| row < n),
        "norm selected row exceeds support"
    );
    selected_values(d, &output, selected, c, convert)
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_production_norm_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let model = env_path("FF_HI3D_MODEL")?.join("vae/model.safetensors");
    let d = Device::new(0)?;
    let mut gate = OperationGate::decoder(
        "actual-support-group-norm",
        "production-norm-freeze-r1",
        "production-norm-sm89-r1",
        4,
    )?;
    for (name, cell) in gate.cells.clone() {
        let definition = &cell["definition"];
        let n = definition["full_input_shape"][0]
            .as_u64()
            .context("norm extent missing")? as usize;
        let c = definition["full_input_shape"][1]
            .as_u64()
            .context("norm width missing")? as usize;
        let rows = indices(&definition["selected_output_rows"])?;
        let bank_name = Path::new(
            definition["common_input_bank"]
                .as_str()
                .context("norm bank missing")?,
        )
        .file_name()
        .context("norm bank filename missing")?;
        let case = name.split('.').nth(1).context("norm case missing")?;
        let bank = root.join(format!("{case}-decode-sm86-r1")).join(bank_name);
        let module = definition["module"]
            .as_str()
            .context("norm module missing")?;
        let weight = d
            .stream
            .clone_htod(&f32_tensor(&model, &format!("{module}.weights"))?)?;
        let bias = d
            .stream
            .clone_htod(&f32_tensor(&model, &format!("{module}.bias"))?)?;
        for control in [false, true] {
            let offsets = if control {
                (0..n)
                    .step_by(47)
                    .chain(std::iter::once(n))
                    .map(|v| v as u32)
                    .collect()
            } else {
                vec![0, n as u32]
            };
            let plan = BatchGroupNorm::new(&d, &offsets, c, 32, 1e-6, 64 << 20)?;
            let x = half_tensor(&bank, "feat")?;
            ensure!(x.len() == n * c, "actual support feature count mismatch");
            let values = if name.ends_with(".field") {
                normalized_features::<f16, f32>(
                    &d,
                    &plan,
                    x,
                    [&weight, &bias],
                    [n, c],
                    &rows,
                    f64::from,
                )?
            } else {
                normalized_features::<f16, f16>(
                    &d,
                    &plan,
                    x,
                    [&weight, &bias],
                    [n, c],
                    &rows,
                    f16::to_f64,
                )?
            };
            gate.score(&name, &values, control)?;
        }
    }
    gate.finish()
}

fn head_case<T: DeviceRepr + Copy + ValidAsZeroBits + 'static>(
    decoder: &PixalDecoder,
    gate: &mut OperationGate,
    case: &str,
    stage: usize,
    input: Vec<T>,
) -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let name = format!("decoder.dec.dec{stage}.up.upsample");
    let bank = root.join(format!("{case}-decode-sm86-r1/{name}.input.safetensors"));
    let sites = bank_sites(&bank)?;
    let n = sites.coordinates().len();
    ensure!(input.len().is_multiple_of(n), "head input extent mismatch");
    let c = input.len() / n;
    let d = &decoder.device;
    let input = d.stream.clone_htod(&input)?;
    let x = matrix::<T>(ptr(&input, d), n, c)?;
    let workspace = PointWorkspace::new(decoder, &sites, [c, c], false)?;
    let maps = root.join(format!("{case}-heads-sm89-r1/maps.safetensors"));
    let rows: Vec<_> = tensor(&maps, &format!("{name}.selected_parents"))?
        .values
        .into_iter()
        .map(|v| v as usize)
        .collect();
    gate.decoder_anchor = Some(root.join(format!("{case}-heads-sm89-r1/anchor.safetensors")));
    let w = decoder
        .weights
        .float
        .get(&format!("{name}.cls_head.fc.0.weights"))
        .context("head norm weight missing")?;
    let b = decoder
        .weights
        .float
        .get(&format!("{name}.cls_head.fc.0.bias"))
        .context("head norm bias missing")?;
    let normalized = d.stream.alloc_zeros::<f32>(n * c)?;
    decoder.point_norm(
        &workspace.norms,
        &format!("{name}.cls_head.fc.0"),
        x,
        &mut TensorMut(matrix::<f32>(ptr(&normalized, d), n, c)?),
    )?;
    gate.decoder_key = Some(format!("{name}.normalized"));
    gate.score(
        &format!("{case}.{name}.normalized"),
        &selected_values(d, &normalized, &rows, c, f64::from)?,
        false,
    )?;
    drop(normalized);
    let offsets: Vec<_> = (0..n)
        .step_by(47)
        .chain(std::iter::once(n))
        .map(|v| v as u32)
        .collect();
    let plan = BatchGroupNorm::new(d, &offsets, c, 32, 1e-6, 64 << 20)?;
    let wrong = d.stream.alloc_zeros::<f32>(n * c)?;
    plan.run::<T, f32>(
        x,
        [
            view(w, &d.stream, 0, [c as u32], [1])?,
            view(b, &d.stream, 0, [c as u32], [1])?,
        ],
        &mut TensorMut(matrix(ptr(&wrong, d), n, c)?),
    )?;
    gate.score(
        &format!("{case}.{name}.normalized"),
        &selected_values(d, &wrong, &rows, c, f64::from)?,
        true,
    )?;
    drop(wrong);
    let logits = decoder.point_head(&workspace, x, stage)?;
    let values = d.stream.clone_dtoh(&logits)?;
    drop(logits);
    let scores: Vec<_> = values.iter().map(|&v| f64::from(v)).collect();
    gate.decoder_key = Some(format!("{name}.logits"));
    gate.score(&format!("{case}.{name}.logits"), &scores, false)?;
    let permutation = [0, 4, 2, 6, 1, 5, 3, 7];
    let wrong: Vec<_> = values
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|row| permutation.map(|i| f64::from(row[i])))
        .collect();
    gate.score(&format!("{case}.{name}.logits"), &wrong, true)?;
    let selectors = root.join("heads-freeze-r1/selectors.safetensors");
    let key = format!("{case}.stage{stage}");
    let stable = tensor(&selectors, &format!("{key}.stable"))?.values;
    let expected = tensor(&selectors, &format!("{key}.expected_occupied"))?.values;
    ensure!(
        stable.len() == values.len() && expected.len() == values.len(),
        "stable selector extent mismatch"
    );
    let decision_defects = values
        .iter()
        .zip(&stable)
        .zip(&expected)
        .filter(|((v, s), e)| **s != 0.0 && ((**v > 0.0) != (**e != 0.0)))
        .count();
    gate.defects += decision_defects;
    let active: Vec<_> = values
        .as_chunks::<8>()
        .0
        .iter()
        .map(|row| std::array::from_fn(|i| row[i] > 0.0))
        .collect();
    let offsets =
        std::array::from_fn(|i| [((i >> 2) & 1) as i32, ((i >> 1) & 1) as i32, (i & 1) as i32]);
    let (children, source) = sites.subdivide_with_offsets(&active, 12_000_000, &offsets)?;
    let source_map: BTreeMap<_, _> = source
        .iter()
        .enumerate()
        .map(|(r, p)| ((p[0] as usize, p[1] as usize), r))
        .collect();
    let mut associations = 0;
    for (i, (&s, &e)) in stable.iter().zip(&expected).enumerate() {
        if s == 0.0 || e == 0.0 {
            continue;
        }
        let parent = i / 8;
        let child = i % 8;
        if let Some(&row) = source_map.get(&(parent, child)) {
            let p = sites.coordinates()[parent];
            ensure!(
                children.coordinates()[row]
                    == [
                        p[0],
                        2 * p[1] + offsets[child][0],
                        2 * p[2] + offsets[child][1],
                        2 * p[3] + offsets[child][2]
                    ],
                "stable child coordinate/source mismatch"
            );
            associations += 1;
        }
    }
    println!(
        "N5_M2B_STABLE {} selector={key} decision_defects={decision_defects} physical_associations={associations} near_zero_propagated=true",
        if decision_defects == 0 {
            "PASS"
        } else {
            "DEFECT"
        }
    );
    drop(workspace);
    let projection = decoder.point_project(x, stage)?;
    let projected = selected_values(d, &projection, &rows, 8 * c, f16::to_f64)?;
    gate.decoder_key = Some(format!("{name}.projection"));
    gate.score(&format!("{case}.{name}.projection"), &projected, false)?;
    let wrong: Vec<_> = projected
        .chunks_exact(8 * c)
        .flat_map(|row| {
            permutation
                .iter()
                .flat_map(move |&i| row[i * c..(i + 1) * c].iter().copied())
        })
        .collect();
    gate.score(&format!("{case}.{name}.projection"), &wrong, true)?;
    let pairs = d.stream.clone_htod(source.as_flattened())?;
    let features = d.stream.alloc_zeros::<f16>(source.len() * c)?;
    decoder.c2s(
        &projection,
        &pairs,
        &features,
        [n, source.len(), c, c],
        false,
    )?;
    for &parent in &rows {
        for child in 0..8 {
            if let Some(&row) = source_map.get(&(parent, child)) {
                let actual = d
                    .stream
                    .clone_dtoh(&features.slice(row * c..(row + 1) * c))?;
                let wanted = d.stream.clone_dtoh(
                    &projection.slice((parent * 8 + child) * c..(parent * 8 + child + 1) * c),
                )?;
                ensure!(
                    actual == wanted,
                    "physical child feature association mismatch"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_heads_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let decoder = PixalDecoder::open(
        &env_path("FF_HI3D_MODEL")?,
        DecoderKind::PointTransformer,
        &env_path("FLYINGFISH_RUNTIME_DIR")?,
        6 << 30,
    )?;
    let mut gate = OperationGate::decoder(
        "occupancy-heads",
        "heads-freeze-r1",
        "cottage-heads-sm89-r1",
        18,
    )?;
    for case in ["cottage", "robot"] {
        for stage in [3, 2, 1] {
            let bank = root.join(format!(
                "{case}-decode-sm86-r1/decoder.dec.dec{stage}.up.upsample.input.safetensors"
            ));
            if stage == 3 {
                head_case(&decoder, &mut gate, case, stage, f32_tensor(&bank, "feat")?)?;
            } else {
                head_case(
                    &decoder,
                    &mut gate,
                    case,
                    stage,
                    half_tensor(&bank, "feat")?,
                )?;
            }
        }
    }
    gate.finish()
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_field_affine_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let mut decoder = PixalDecoder::open(
        &env_path("FF_HI3D_MODEL")?,
        DecoderKind::PointTransformer,
        &env_path("FLYINGFISH_RUNTIME_DIR")?,
        6 << 30,
    )?;
    let d = &decoder.device;
    let mut field = OperationGate::decoder("corner-attributes", "fields-freeze-r1", "sm89-r1", 1)?;
    let raw = f32_tensor(
        &root.join("CPU-r3/common-inputs.safetensors"),
        "fields.raw40",
    )?;
    ensure!(raw.len() == 97 * 40, "corner field fixture shape mismatch");
    let values = d.stream.clone_htod(&raw)?;
    unsafe {
        d.stream
            .launch_builder(
                decoder
                    .k
                    .corner_fields
                    .as_ref()
                    .context("corner activation not loaded")?,
            )
            .arg(&values)
            .arg(&97u32)
            .launch(flat(97 * 32))?;
    }
    let actual = d.stream.clone_dtoh(&values)?;
    let selected: Vec<_> = actual
        .as_chunks::<40>()
        .0
        .iter()
        .flat_map(|row| row[8..].iter().map(|&v| f64::from(v)))
        .collect();
    ensure!(
        actual
            .as_chunks::<40>()
            .0
            .iter()
            .zip(raw.as_chunks::<40>().0.iter())
            .all(|(a, b)| a[..8] == b[..8]),
        "sign logits changed during corner activation"
    );
    field.score("fields.activated32", &selected, false)?;
    let wrong: Vec<_> = raw
        .as_chunks::<40>()
        .0
        .iter()
        .flat_map(|row| row[..32].iter().map(|v| f64::from(v.tanh())))
        .collect();
    field.score("fields.activated32", &wrong, true)?;
    let field_result = field.finish();
    let mut gate = OperationGate::decoder(
        "single-round-affine",
        "affine-freeze-r1",
        "affine-rounding-sm89-r1",
        1,
    )?;
    gate.decoder_anchor = Some(root.join("affine-rounding-sm89-r1/outputs.safetensors"));
    gate.decoder_key = Some("anchor".into());
    let rows = indices(&gate.cells["affine.production_qkv"]["definition"]["selected_rows"])?;
    let (n, ci, co) = (11295usize, 512usize, 1536usize);
    let unit = 2f32.powi(-10);
    let input: Vec<_> = (0..n * ci)
        .map(|i| {
            f16::from_f32(1.0 + unit * (1 + (((i / ci) * 127 + (i % ci) * 61) % 251) % 2) as f32)
        })
        .collect();
    let mut weight = vec![f16::ZERO; ci * co];
    let mut bias = vec![f16::ZERO; co];
    for column in 0..co {
        let sign = if (column * 61 % 127).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        weight[column * ci + column * 127 % ci] = f16::from_f32(sign * (1.0 + unit));
        bias[column] = f16::from_f32(sign * unit / 2.0);
    }
    let name = "decoder.mid.block0.attn.qkv";
    decoder
        .weights
        .half
        .insert(format!("{name}.weight"), d.stream.clone_htod(&weight)?);
    decoder
        .weights
        .half
        .insert(format!("{name}.bias"), d.stream.clone_htod(&bias)?);
    let input = d.stream.clone_htod(&input)?;
    let output = d.stream.alloc_zeros::<f16>(n * co)?;
    decoder.point_linear(
        name,
        matrix(ptr(&input, d), n, ci)?,
        &mut TensorMut(matrix(ptr(&output, d), n, co)?),
        true,
    )?;
    gate.score(
        "affine.production_qkv",
        &selected_values(d, &output, &rows, co, f16::to_f64)?,
        false,
    )?;
    decoder.point_linear(
        name,
        matrix(ptr(&input, d), n, ci)?,
        &mut TensorMut(matrix(ptr(&output, d), n, co)?),
        false,
    )?;
    let early = selected_values(d, &output, &rows, co, f16::to_f64)?;
    let wrong: Vec<_> = early
        .into_iter()
        .enumerate()
        .map(|(i, v)| f16::from_f32(v as f32 + bias[i % co].to_f32()).to_f64())
        .collect();
    gate.score("affine.production_qkv", &wrong, true)?;
    field_result?;
    gate.finish()
}

fn stage_case<I, O>(
    decoder: &PixalDecoder,
    gate: &mut OperationGate,
    name: &str,
    bank: &Path,
    input: Vec<I>,
    execute: impl Fn(&PointWorkspace<I>, &SparseConv<f32>, CudaSlice<I>) -> Result<CudaSlice<O>>,
    convert: impl Fn(O) -> f64,
) -> Result<()>
where
    I: DeviceRepr + Copy + ValidAsZeroBits + 'static,
    O: DeviceRepr + Copy + 'static,
{
    let cell = &gate.cells[name];
    let rows = indices(&cell["definition"]["selected_rows"])?;
    let ci = cell["definition"]["full_input_shape"][1]
        .as_u64()
        .context("stage input width missing")? as usize;
    let co = cell["definition"]["output_shape"][1]
        .as_u64()
        .context("stage output width missing")? as usize;
    let sites = bank_sites(bank)?;
    let n = sites.coordinates().len();
    ensure!(input.len() == n * ci, "stage input extent mismatch");
    let d = &decoder.device;
    let workspace = PointWorkspace::new(
        decoder,
        &sites,
        [
            ci.max(co),
            co.max(if name.ends_with(".field") { 128 } else { co }),
        ],
        name.ends_with(".field") || name.ends_with(".dec0"),
    )?;
    let max_width = ci.max(co);
    let wrong_budget = n * 108 + 127 * (27 * max_width + max_width) * 4;
    let mut wrong_conv =
        SparseConv::<f32>::promoted(d, &sites, [max_width, max_width], wrong_budget)?;
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
    wrong_conv.replace_neighbours(&maps)?;
    drop(maps);
    for control in [false, true] {
        let x = d.stream.clone_htod(&input)?;
        let output = execute(
            &workspace,
            if control {
                &wrong_conv
            } else {
                &workspace.conv
            },
            x,
        )?;
        ensure!(output.len() == n * co, "stage output extent mismatch");
        gate.score(
            name,
            &selected_values(d, &output, &rows, co, &convert)?,
            control,
        )?;
        if name.ends_with(".field") {
            ensure!(
                std::any::TypeId::of::<O>() == std::any::TypeId::of::<f32>(),
                "raw corner field dtype mismatch"
            );
            unsafe {
                d.stream
                    .launch_builder(
                        decoder
                            .k
                            .corner_fields
                            .as_ref()
                            .context("corner activation missing")?,
                    )
                    .arg(&output)
                    .arg(&u32::try_from(n)?)
                    .launch(flat(n * 32))?;
            }
            let values = selected_values(d, &output, &rows, co, &convert)?;
            let activated: Vec<_> = values
                .as_chunks::<40>()
                .0
                .iter()
                .flat_map(|r| r[8..].iter().copied())
                .collect();
            gate.score(&format!("{name}.activated"), &activated, control)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_stages_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let decoder = PixalDecoder::open(
        &env_path("FF_HI3D_MODEL")?,
        DecoderKind::PointTransformer,
        &env_path("FLYINGFISH_RUNTIME_DIR")?,
        6 << 30,
    )?;
    let d = &decoder.device;
    let mut gate = OperationGate::decoder(
        "whole-decoder-stages",
        "stages-freeze-r1",
        "stages-sm89-r1",
        16,
    )?;
    let supplement: Value = serde_json::from_slice(&std::fs::read(
        root.join("stages-field-supplement-20261106-r1/FROZEN.json"),
    )?)?;
    ensure!(
        supplement["reference_protocol"] == "PASS"
            && supplement["formula"] == "tau=max(2*max(all declared pair errors),epsilon)",
        "invalid field supplement protocol"
    );
    let cells = supplement["cells"]
        .as_object()
        .context("field supplement cells missing")?;
    ensure!(cells.len() == 4, "field supplement coverage mismatch");
    for (name, cell) in cells {
        ensure!(
            [
                "stage.cottage.field",
                "stage.cottage.field.activated",
                "stage.robot.field",
                "stage.robot.field.activated"
            ]
            .contains(&name.as_str())
                && cell["reference_protocol"] == "PASS",
            "supplement outside field scope"
        );
        let original = gate
            .cells
            .get_mut(name)
            .context("supplement cell outside original freeze")?;
        original["original_tau"] = original["tau"].clone();
        original["tau"] = cell["tau"].clone();
        println!(
            "N5_M2B_FIELD_SUPPLEMENT cell={name} original_floor_retained=true dated=2026-11-06"
        );
    }
    for case in ["cottage", "robot"] {
        let directory = root.join(format!("{case}-decode-sm86-r1"));
        let bank = directory.join("decoder.mid.input.safetensors");
        let n = bank_sites(&bank)?.coordinates().len();
        let latents = root.join(format!("{case}-latents-sm86-r1/trajectory.safetensors"));
        stage_case(
            &decoder,
            &mut gate,
            &format!("stage.{case}.latent"),
            &bank,
            f32_tensor(&latents, "final_latents")?,
            |_, conv, input| {
                let output = d.stream.alloc_zeros::<f32>(n * 512)?;
                decoder.point_conv(
                    conv,
                    "decoder.z_to_f.conv.0",
                    matrix::<f32>(ptr(&input, d), n, 32)?,
                    &mut TensorMut(matrix::<f32>(ptr(&output, d), n, 512)?),
                )?;
                Ok(output)
            },
            f64::from,
        )?;
        let maps = bank_sites(&bank)?.serialized_windows(48)?;
        stage_case(
            &decoder,
            &mut gate,
            &format!("stage.{case}.mid"),
            &bank,
            f32_tensor(&bank, "feat")?,
            |workspace, conv, input| decoder.point_mid(workspace, &maps, input, conv),
            f64::from,
        )?;
        for (stage, ci, co) in [(3, 512, 512), (2, 512, 256), (1, 256, 128), (0, 128, 128)] {
            let bank = directory.join(format!("decoder.dec.dec{stage}.input.safetensors"));
            let name = format!("stage.{case}.dec{stage}");
            if stage == 3 {
                stage_case(
                    &decoder,
                    &mut gate,
                    &name,
                    &bank,
                    f32_tensor(&bank, "feat")?,
                    |workspace, conv, input| {
                        let x = decoder.point_residual(
                            workspace,
                            &format!("decoder.dec.dec{stage}.block0"),
                            input,
                            [ci, co],
                            &workspace.conv,
                        )?;
                        decoder.point_residual(
                            workspace,
                            &format!("decoder.dec.dec{stage}.block1"),
                            x,
                            [co, co],
                            conv,
                        )
                    },
                    f64::from,
                )?;
            } else {
                stage_case(
                    &decoder,
                    &mut gate,
                    &name,
                    &bank,
                    half_tensor(&bank, "feat")?,
                    |workspace, conv, input| {
                        let x = decoder.point_residual(
                            workspace,
                            &format!("decoder.dec.dec{stage}.block0"),
                            input,
                            [ci, co],
                            &workspace.conv,
                        )?;
                        decoder.point_residual(
                            workspace,
                            &format!("decoder.dec.dec{stage}.block1"),
                            x,
                            [co, co],
                            conv,
                        )
                    },
                    f16::to_f64,
                )?;
            }
        }
        let bank = directory.join("decoder.conv_out.input.safetensors");
        stage_case(
            &decoder,
            &mut gate,
            &format!("stage.{case}.field"),
            &bank,
            half_tensor(&bank, "feat")?,
            |workspace, conv, input| decoder.point_field(workspace, &input, conv),
            f64::from,
        )?;
    }
    gate.finish()
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_complete_gpu() -> Result<()> {
    let root = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let decoder = PixalDecoder::open(
        &env_path("FF_HI3D_MODEL")?,
        DecoderKind::PointTransformer,
        &env_path("FLYINGFISH_RUNTIME_DIR")?,
        6 << 30,
    )?;
    for case in ["cottage", "robot"] {
        let trajectory = root.join(format!("{case}-latents-sm86-r1/trajectory.safetensors"));
        let coordinates = decoder_coordinates(&trajectory, "coordinates")?;
        let latent = f32_tensor(&trajectory, "final_latents")?;
        let bank = root.join(format!(
            "{case}-decode-sm86-r1/decoder.mid.input.safetensors"
        ));
        let matrix = tensor(&bank, "matrix")?;
        ensure!(
            matrix.values.len() == 16,
            "inherited matrix extent mismatch"
        );
        let frame =
            std::array::from_fn(|r| std::array::from_fn(|c| matrix.values[r * 4 + c] as f32));
        let output = decoder.points(coordinates, &latent, frame, 64)?;
        let n = output.fields.coordinates.len();
        ensure!(
            n > 0
                && n <= 12_000_000
                && output.fields.channels == 32
                && output.fields.features.len() == n * 32
                && output.sign_logits.len() == n * 8,
            "complete decoder output extent mismatch"
        );
        ensure!(
            output.matrix == frame && output.resolution == 512,
            "complete decoder changed inherited frame"
        );
        ensure!(
            output
                .fields
                .coordinates
                .iter()
                .all(|p| p[0] == 0 && p[1..].iter().all(|&v| (0..512).contains(&v))),
            "complete decoder coordinates outside output frame"
        );
        ensure!(
            output.sign_logits.iter().all(|v| v.is_finite())
                && output
                    .fields
                    .features
                    .iter()
                    .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
            "complete decoder nonfinite or unactivated output"
        );
        println!(
            "PASS N5_M2B_COMPLETE case={case} rows={n} channels=32 signs=8 resolution=512 inherited_matrix_exact=true convolutions=27 residuals=10 unforced_support_measurement_only=true timing=false"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_decoder_norm_activation_gpu() -> Result<()> {
    use crate::backend::cuda::sparse::BatchGroupNorm;
    let reference = env_path("FF_HI3D_DECODER_REFERENCE")?;
    let model = env_path("FF_HI3D_MODEL")?.join("vae/model.safetensors");
    let d = Device::new(0)?;
    let mut gate = OperationGate::decoder("batch-group-norm", "norm-freeze-r1", "sm89-r1", 12)?;
    for (name, cell) in gate.cells.clone() {
        let definition = &cell["definition"];
        let input = tensor(
            &reference.join("CPU-r3/common-inputs.safetensors"),
            &format!("{name}.input"),
        )?;
        let [n, c] = <[usize; 2]>::try_from(input.shape)
            .map_err(|_| anyhow::anyhow!("invalid group norm shape"))?;
        let module = definition["parameter_module"]
            .as_str()
            .context("norm parameter module missing")?;
        let weight = d
            .stream
            .clone_htod(&f32_tensor(&model, &format!("{module}.weights"))?)?;
        let bias = d
            .stream
            .clone_htod(&f32_tensor(&model, &format!("{module}.bias"))?)?;
        let counts: Vec<u32> = definition["batch_counts"]
            .as_array()
            .context("batch counts missing")?
            .iter()
            .map(|n| u32::try_from(n.as_u64().context("batch count invalid")?).map_err(Into::into))
            .collect::<Result<_>>()?;
        let groups = definition["groups"]
            .as_u64()
            .context("group count missing")? as usize;
        let eps = definition["eps"].as_f64().context("norm epsilon missing")? as f32;
        let selected: Vec<_> = (0..n).collect();
        for control in [false, true] {
            let mut offsets = vec![0u32];
            for &count in &counts {
                let first = *offsets.last().context("empty norm offsets")?;
                if control {
                    for row in (0..count).step_by(47) {
                        offsets.push(first + (row + 47).min(count));
                    }
                } else {
                    offsets.push(first + count);
                }
            }
            let plan = BatchGroupNorm::new(&d, &offsets, c, groups, eps, 16 << 20)?;
            let values = if name.ends_with(".float16") {
                normalized_features::<f16, f16>(
                    &d,
                    &plan,
                    input.values.iter().map(|&x| f16::from_f64(x)).collect(),
                    [&weight, &bias],
                    [n, c],
                    &selected,
                    f16::to_f64,
                )?
            } else {
                normalized_features::<f32, f32>(
                    &d,
                    &plan,
                    input.values.iter().map(|&x| x as f32).collect(),
                    [&weight, &bias],
                    [n, c],
                    &selected,
                    f64::from,
                )?
            };
            gate.score(&name, &values, control)?;
        }
    }
    let norm_result = gate.finish();
    let mut gate =
        OperationGate::decoder("rounded-sigmoid-swish", "swish-freeze-r2", "sm89-r1", 4)?;
    let module = cubin::module(&d.ctx, "vision_ops")?;
    for (name, cell) in gate.cells.clone() {
        let input = tensor(
            &reference.join("CPU-r3/common-inputs.safetensors"),
            &format!("{name}.input"),
        )?;
        let n = input.shape[0];
        let c = input.shape[1];
        let half = cell["dtype"] == "torch.float16";
        let rows_arg = n as u32;
        let width_arg = c as u32;
        let count_arg = (n * c) as u64;
        let control_mode = if half { 1i32 } else { 0i32 };
        for control in [false, true] {
            let function = module.load_function(if control {
                if half {
                    "activation_f16"
                } else {
                    "activation_f32"
                }
            } else if half {
                "swish_rounded_f16"
            } else {
                "swish_rounded_f32"
            })?;
            let output = if half {
                let input = d.stream.clone_htod(
                    &input
                        .values
                        .iter()
                        .map(|&x| f16::from_f64(x))
                        .collect::<Vec<_>>(),
                )?;
                let output = d.stream.alloc_zeros::<f16>(n * c)?;
                let mut launch = d.stream.launch_builder(&function);
                launch.arg(&input).arg(&output);
                if control {
                    launch.arg(&rows_arg).arg(&width_arg).arg(&control_mode);
                } else {
                    launch.arg(&count_arg);
                }
                unsafe {
                    launch.launch(flat(n * c))?;
                }
                d.stream
                    .clone_dtoh(&output)?
                    .into_iter()
                    .map(f16::to_f64)
                    .collect::<Vec<_>>()
            } else {
                let input = d
                    .stream
                    .clone_htod(&input.values.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
                let output = d.stream.alloc_zeros::<f32>(n * c)?;
                let mut launch = d.stream.launch_builder(&function);
                launch.arg(&input).arg(&output);
                if control {
                    launch.arg(&rows_arg).arg(&width_arg).arg(&control_mode);
                } else {
                    launch.arg(&count_arg);
                }
                unsafe {
                    launch.launch(flat(n * c))?;
                }
                d.stream
                    .clone_dtoh(&output)?
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>()
            };
            gate.score(&name, &output, control)?;
        }
    }
    let activation_result = gate.finish();
    norm_result?;
    activation_result
}

#[test]
#[ignore = "requires admitted GPU, published decoder and frozen reference inputs"]
fn hi3d_serialized_attention_definition_gpu() -> Result<()> {
    let model = env_path("FF_HI3D_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let decoder = PixalDecoder::open(&model, DecoderKind::PointTransformer, &runtime, 6 << 30)?;
    let d = &decoder.device;
    let mut gate = OperationGate::decoder(
        "serialized-window-attention",
        "attention-freeze-r2",
        "sm89-r1",
        20,
    )?;
    let common = gate
        .root
        .join("sm89-r1/additional-common-inputs.safetensors");
    let map_file = gate.root.join("CPU-r3/maps.safetensors");
    let freeze: Value = serde_json::from_slice(&std::fs::read(gate.root.join("FROZEN.json"))?)?;
    let cases = freeze["exact_serialization_maps"]
        .as_object()
        .context("serialization cases missing")?;
    ensure!(cases.len() == 9, "serialization case count mismatch");
    for name in cases.keys() {
        let name = name
            .strip_prefix("serialization.")
            .context("serialization prefix missing")?;
        let grid = tensor(&map_file, &format!("{name}.grid"))?;
        let batch = tensor(&map_file, &format!("{name}.batch"))?;
        let coordinates = grid
            .values
            .as_chunks::<3>()
            .0
            .iter()
            .zip(batch.values)
            .map(|(p, b)| [b as i32, p[0] as i32, p[1] as i32, p[2] as i32])
            .collect();
        let sites = SparseSites::new(coordinates)?;
        let mut maps = sites.serialized_windows(48)?;
        let original_pad = maps.pad.clone();
        let frozen_unpad = tensor(&map_file, &format!("{name}.unpad"))?;
        let frozen_inverse = tensor(&map_file, &format!("{name}.inverse"))?;
        let control_pad = tensor(&map_file, &format!("{name}.control_pad"))?;
        let n = sites.coordinates().len();
        for order in 0..2 {
            let key = format!("attention.{name}.order{order}");
            let packed = tensor(&common, &format!("{key}.qkv"))?;
            let mut original = Vec::with_capacity(n * 1536);
            for row in 0..n {
                let rank = frozen_inverse.values[order * n + row] as usize;
                let packed_row = frozen_unpad.values[rank] as usize;
                original.extend(
                    packed.values[packed_row * 1536..(packed_row + 1) * 1536]
                        .iter()
                        .map(|&x| f16::from_f64(x)),
                );
            }
            let qkv = d.stream.clone_htod(&original)?;
            let selector = format!("{key}.final_window");
            let selector_rows: Option<Vec<usize>> = gate
                .cells
                .get(&selector)
                .map(|cell| {
                    cell["selector"]["rows"]
                        .as_array()
                        .context("attention selector rows missing")
                        .and_then(|rows| {
                            rows.iter()
                                .map(|row| {
                                    row.as_u64()
                                        .context("invalid selector row")
                                        .map(|row| row as usize)
                                })
                                .collect()
                        })
                })
                .transpose()?;
            for control in [false, true] {
                if control {
                    maps.pad = control_pad.values.iter().map(|&v| v as u32).collect();
                }
                let output =
                    decoder.serialized_qkv(&maps, matrix::<f16>(ptr(&qkv, d), n, 1536)?, order)?;
                let values: Vec<_> = d
                    .stream
                    .clone_dtoh(&output)?
                    .into_iter()
                    .map(f16::to_f64)
                    .collect();
                let has_full_control =
                    !gate.cells[&key]["control_margin"].is_null() && selector_rows.is_none();
                if !control || has_full_control {
                    gate.score(&key, &values, control)?;
                }
                if let Some(rows) = &selector_rows {
                    let selected: Vec<_> = rows
                        .iter()
                        .flat_map(|&row| values[row * 512..(row + 1) * 512].iter().copied())
                        .collect();
                    gate.score(&selector, &selected, control)?;
                }
                maps.pad.clone_from(&original_pad);
            }
        }
    }
    serialized_attention_modes(&mut gate, &common, &map_file)?;
    gate.finish()
}

fn serialized_attention_modes(gate: &mut OperationGate, common: &Path, maps: &Path) -> Result<()> {
    let context = crate::backend::gpu::CudaContext::new(0)?;
    // Replay buffers and metadata belong to this isolated stream.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream()?;
    let first = tensor(common, "attention.17.order0.qkv")?;
    let second = tensor(common, "attention.49.order0.qkv")?;
    let combined: Vec<_> = first
        .values
        .into_iter()
        .chain(second.values)
        .map(f16::from_f64)
        .collect();
    let input_rows = combined.len() / 1536;
    ensure!(input_rows == 113, "mixed serialized packet extent mismatch");
    let input = stream.clone_htod(&combined)?;
    let module = cubin::module(&context, "vision_ops")?;
    for grouped in [false, true] {
        let rows = if grouped { 113usize } else { 17usize };
        let count = rows * 512;
        let source = stream.clone_htod(&(0..rows as u32).collect::<Vec<_>>())?;
        let q = stream.alloc_zeros::<f16>(count)?;
        let k = stream.alloc_zeros::<f16>(count)?;
        let v = stream.alloc_zeros::<f16>(count)?;
        let sentinel = f16::from_f32(-123.0);
        let mut output = stream.clone_htod(&vec![sentinel; count + 8])?;
        unsafe {
            stream
                .launch_builder(&module.load_function("qkv_gather_head_major_f16")?)
                .arg(&input)
                .arg(&source)
                .arg(&q)
                .arg(&k)
                .arg(&v)
                .arg(&(input_rows as u32))
                .arg(&(rows as u32))
                .arg(&32u32)
                .arg(&16u32)
                .launch(flat(count))?;
        }
        let mut attention = Flash::new_f16(
            stream.clone(),
            rows,
            32,
            32,
            16,
            Score {
                scale: 0.25,
                window_left: None,
            },
        )?;
        if grouped {
            attention.bind_f16_windows(&[0, 17, 65, 113])?;
        }
        let graph_equal;
        let guard_unchanged;
        {
            let mut run = || {
                attention.run_f16(
                    &q.slice(..),
                    512,
                    &k.slice(..),
                    &v.slice(..),
                    &mut output.slice_mut(..count),
                    (rows, rows, rows),
                )
            };
            run()?;
            stream.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            run()?;
            let graph = stream
                .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                .context("serialized attention graph missing")?;
            let eager = stream.clone_dtoh(&output)?;
            graph.launch()?;
            let captured = stream.clone_dtoh(&output)?;
            ensure!(
                eager[..count]
                    .iter()
                    .chain(&captured[..count])
                    .all(|v| v.is_finite()),
                "serialized F16 output is nonfinite"
            );
            graph_equal = eager == captured;
            guard_unchanged = eager[count..]
                .iter()
                .chain(&captured[count..])
                .all(|&v| v == sentinel);
            gate.defects += usize::from(!graph_equal) + usize::from(!guard_unchanged);
            for (case, base) in [("17", 0usize), ("49", 17usize)] {
                if !grouped && case == "49" {
                    continue;
                }
                let inverse = tensor(maps, &format!("{case}.inverse"))?;
                let unpad = tensor(maps, &format!("{case}.unpad"))?;
                let n = unpad.values.len();
                let values: Vec<_> = inverse.values[..n]
                    .iter()
                    .flat_map(|&rank| {
                        let row = base + unpad.values[rank as usize] as usize;
                        captured[row * 512..(row + 1) * 512]
                            .iter()
                            .map(|v| v.to_f64())
                    })
                    .collect();
                gate.replay(
                    &format!("attention.{case}.order0"),
                    &values,
                    if grouped {
                        "mixed17_49_grouped_graph_and_eager"
                    } else {
                        "scalar_graph_and_eager"
                    },
                )?;
            }
        }
        let isolation = if grouped {
            let expected = stream.clone_dtoh(&output)?;
            attention.bind_f16_windows(&[0, 48, 96, 113])?;
            attention.run_f16(
                &q.slice(..),
                512,
                &k.slice(..),
                &v.slice(..),
                &mut output.slice_mut(..count),
                (rows, rows, rows),
            )?;
            attention.bind_f16_windows(&[0, 17, 65, 113])?;
            attention.run_f16(
                &q.slice(..),
                512,
                &k.slice(..),
                &v.slice(..),
                &mut output.slice_mut(..count),
                (rows, rows, rows),
            )?;
            let restored = expected == stream.clone_dtoh(&output)?;
            gate.defects += usize::from(!restored);
            Some(restored)
        } else {
            None
        };
        let valid = graph_equal && guard_unchanged && isolation != Some(false);
        println!(
            "{} N5_M2B_ATTENTION_MODE grouped={grouped} graph_eager_bit_identical={graph_equal} guard_unchanged={guard_unchanged} rows={rows} metadata_rebind={grouped} isolation_bit_identical={isolation:?} timing=false",
            if valid { "PASS" } else { "DEFECT" }
        );
    }
    Ok(())
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
    let mut exact_cases = 0;
    for rows in [17, 11295] {
        for width in [256, 2112] {
            let first: Vec<_> = (0..rows * width)
                .map(|i| f16::from_f32(((i * 127 % 251) as f32 - 125.0) / 128.0))
                .collect();
            let second: Vec<_> = first
                .iter()
                .map(|x| f16::from_f32(x.to_f32() + 4.0))
                .collect();
            let a = flow.device.stream.clone_htod(&first)?;
            let b = flow.device.stream.clone_htod(&second)?;
            let output = flow.device.stream.alloc_zeros::<f16>(2 * rows * width)?;
            for control in [false, true] {
                let (left, right) = if control { (&b, &a) } else { (&a, &b) };
                io.concat(
                    &flow.device,
                    ptr(left, &flow.device),
                    ptr(right, &flow.device),
                    ptr(&output, &flow.device),
                    rows,
                    width,
                )?;
                let actual = flow.device.stream.clone_dtoh(&output)?;
                let mismatches = actual
                    .iter()
                    .enumerate()
                    .filter(|&(i, value)| {
                        let row = i / (2 * width);
                        let col = i % (2 * width);
                        let expected = if col < width {
                            first[row * width + col]
                        } else {
                            second[row * width + col - width]
                        };
                        value.to_bits() != expected.to_bits()
                    })
                    .count();
                ensure!(
                    mismatches == if control { actual.len() } else { 0 },
                    "concat source/column indexing mismatch"
                );
            }
            let copied = flow.device.stream.alloc_zeros::<f16>(rows * width)?;
            io.copy(
                &flow.device,
                ptr(&a, &flow.device),
                0,
                ptr(&copied, &flow.device),
                rows,
                width,
            )?;
            ensure!(
                flow.device
                    .stream
                    .clone_dtoh(&copied)?
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(first.iter().map(|x| x.to_bits())),
                "identity copy changed stored bits"
            );
            let indices: Vec<_> = (0..rows).map(|r| (r * 127 % rows) as u32).collect();
            let index = flow.device.stream.clone_htod(&indices)?;
            io.copy(
                &flow.device,
                ptr(&a, &flow.device),
                ptr(&index, &flow.device),
                ptr(&copied, &flow.device),
                rows,
                width,
            )?;
            let expected = indices.iter().flat_map(|&r| {
                first[r as usize * width..(r as usize + 1) * width]
                    .iter()
                    .map(|x| x.to_bits())
            });
            ensure!(
                flow.device
                    .stream
                    .clone_dtoh(&copied)?
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected),
                "indexed copy changed row/column order"
            );
            exact_cases += 1;
            println!(
                "N5_M1_SPARSE_ROWS PASS rows={rows} width={width} identity=true indexed=true concat=true wrong_source_control_mismatches={} byte_exact=true",
                rows * width * 2
            );
        }
    }
    ensure!(exact_cases == 4, "sparse row coverage missing");
    gate.finish()
}

#[test]
#[ignore = "requires frozen SiLU reference cells; CUDA"]
fn hi3d_flow_silu_gpu() -> Result<()> {
    let mut gate = OperationGate::new("silu", 12)?;
    let device = Device::new(0)?;
    let kernel = cubin::module(&device.ctx, "vision_ops")?.load_function("vision_silu_f16")?;
    for (name, cell) in gate.cells.clone() {
        let rows = cell["shape"][0].as_u64().context("SiLU rows missing")? as usize;
        let width = cell["shape"][1].as_u64().context("SiLU width missing")? as usize;
        let count = rows.min(47);
        let selected: Vec<_> = (0..count)
            .map(|r| (0..count).find(|&i| i * 127 % count == r).unwrap())
            .collect();
        for control in [false, true] {
            let input: Vec<_> = (0..rows)
                .flat_map(|row| {
                    (0..width).map(move |col| {
                        let c = if control {
                            (col + width - 127 % width) % width
                        } else {
                            col
                        };
                        f16::from_f32(
                            ((((row * 127 % count) * 61 + c * 127) % 251) as f32 - 125.0) / 16.0,
                        )
                    })
                })
                .collect();
            let input = device.stream.clone_htod(&input)?;
            let output = device.stream.alloc_zeros::<f16>(rows * width)?;
            unsafe {
                device
                    .stream
                    .launch_builder(&kernel)
                    .arg(&input)
                    .arg(&output)
                    .arg(&u32::try_from(rows * width)?)
                    .launch(flat(rows * width))?;
            }
            gate.score(
                &name,
                &selected_half(&device, &output, &selected, width)?,
                control,
            )?;
        }
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
        let mut flow = Flow::<f16>::open_hi3d(&root, component, &runtime, &coordinates, 8 << 30)?;
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
            let (block, conv) = op
                .rsplit_once('.')
                .context("convolution component missing")?;
            let block = match block {
                "input_blocks.0" => 0,
                "input_blocks.1" => 1,
                "out_blocks.0" => 2,
                "out_blocks.1" => 3,
                _ => anyhow::bail!("unknown IO block"),
            };
            let conv = match conv {
                "conv1" => 0,
                "conv2" => 1,
                _ => anyhow::bail!("unknown IO convolution"),
            };
            let convolution =
                &mut flow.io.as_mut().context("sparse IO missing")?.blocks[block].conv[conv];
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
    let context_stream = flow.device.ctx.new_stream()?;
    let context_device = context_stream.clone_htod(&context)?;
    let device_positive = flow.conditioning_device(&context_device, &coordinates)?;
    ensure!(
        flow.device.stream.clone_dtoh(&device_positive.global)?
            == flow.device.stream.clone_dtoh(&positive.global)?,
        "device conditioning conversion mismatch"
    );
    let mut wrong_coordinates = coordinates.clone();
    wrong_coordinates[0][0] += 1;
    ensure!(
        flow.conditioning_device(&context_device, &wrong_coordinates)
            .is_err(),
        "device conditioning coordinate control accepted"
    );
    println!(
        "N5_M2A_HANDOFF PASS rows=2917 channels=2048 exact_f16=true cross_stream=true coordinate_control=rejected production_context_D2H_H2D=0 inspection_D2H=true"
    );
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
    package: &'static str,
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
            package: "P1_M3A",
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
                    "{}_METRIC {}",
                    self.package,
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
                "{}_EXACT {}",
                self.package,
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
            "{}_GATE {} cells={} defects={} timing=false eager=true",
            self.package,
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
            let mut sampler = Sampler::new(&d, &noise, true)?;
            d.stream.memcpy_htod(&positive, &mut sampler.positive)?;
            d.stream.memcpy_htod(
                &negative,
                sampler.negative.as_mut().context("CFG buffer missing")?,
            )?;
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
    negative: Option<&Conditioning>,
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
    let mut sampler = Sampler::new(&flow.device, noise, !flow.spec.texture())?;
    let mut call = 0;
    let mut forwards = [0usize; 2];
    for step in 0..steps {
        for (index, condition) in [Some(positive), negative].into_iter().enumerate() {
            let Some(condition) = condition else { continue };
            if index == 1 && !(0.6..=1.0).contains(&times[step]) {
                continue;
            }
            let output = if index == 0 {
                &mut sampler.positive
            } else {
                sampler.negative.as_mut().context("CFG buffer missing")?
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
            forwards[index] += 1;
        }
        sampler.step(&flow.device, times[step], times[step + 1], sparse)?;
        gate.score_f32(
            case.cell(&format!("state{step}")),
            &case.shape,
            &flow.device.stream.clone_dtoh(&sampler.state)?,
        )?;
    }
    ensure!(
        call == if flow.spec.texture() {
            12
        } else if sparse {
            21
        } else {
            22
        },
        "CFG branch call count mismatch"
    );
    if flow.spec.texture() {
        texture_launch_contract(forwards[0], forwards[1], 0)?;
        ensure!(
            texture_launch_contract(forwards[0], forwards[1] + 1, 0).is_err(),
            "negative branch mutant survived"
        );
        println!(
            "P1_M4B_BRANCHES positive={} negative={} mutant=rejected",
            forwards[0], forwards[1]
        );
    }
    gate.score_f32(
        case.cell("final"),
        &case.shape,
        &flow.device.stream.clone_dtoh(&sampler.state)?,
    )?;
    println!(
        "{}_FLOW_COMPLETE table={} fixture={:?} forwards={call} steps={steps}",
        gate.package, case.table, case.fixture
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
            &Device::new(0)?,
            &model.join("ckpts"),
            &stem,
            runtime,
            (&ss_coordinates(), 16),
            &Rc::new(()),
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
                Some(&negative),
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
                &Device::new(0)?,
                &model.join("ckpts"),
                &stem,
                runtime,
                (
                    &points,
                    if stage == 0 {
                        32
                    } else if resolution == 1536 && fixture == "robot_crab" {
                        88
                    } else {
                        resolution / 16
                    },
                ),
                &Rc::new(()),
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
                Some(&negative),
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
            &Device::new(0)?,
            &model.join("ckpts"),
            "slat_flow_img2shape_dit_1_3B_1024_bf16",
            runtime,
            (
                &coordinates,
                if stage == 0 {
                    32
                } else if resolution == 1536 {
                    88
                } else {
                    64
                },
            ),
            &Rc::new(()),
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
    ensure!(
        mean.len() == channels && std.len() == channels,
        "normalization width differs from flow"
    );
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
        channel_transform(&d, &mut values, [&mean, &std], false)?;
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
    let mut camera = crate::backend::ProjectionCamera {
        world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0],
        horizontal_fov: 0.7,
        mesh_scale: 1.0,
    };
    let first = camera;
    camera.world_to_camera[3] = 0.25;
    let views = [first, camera];
    let shape = crate::backend::PixalConditioningShape {
        image_side: 1024,
        grid_side: 88,
        naf_side: Some(512),
    };
    validate_projection_binding(
        [&origin, &origin],
        [&coordinates, &coordinates],
        [shape, shape],
        [&views, &views],
    )?;
    let foreign = Rc::new(());
    let reordered = [coordinates[2], coordinates[0], coordinates[1]];
    assert!(
        validate_projection_binding(
            [&origin, &foreign],
            [&coordinates, &coordinates],
            [shape, shape],
            [&views, &views]
        )
        .is_err()
    );
    assert!(
        validate_projection_binding(
            [&origin, &origin],
            [&coordinates, &reordered],
            [shape, shape],
            [&views, &views]
        )
        .is_err()
    );
    assert!(
        validate_projection_binding(
            [&origin, &origin],
            [&coordinates, &coordinates],
            [
                shape,
                crate::backend::PixalConditioningShape {
                    grid_side: 96,
                    ..shape
                }
            ],
            [&views, &views]
        )
        .is_err()
    );
    assert!(
        validate_projection_binding(
            [&origin, &origin],
            [&coordinates, &coordinates],
            [shape, shape],
            [&views, &[camera, first]]
        )
        .is_err()
    );
    assert!(
        validate_projection_binding(
            [&origin, &origin],
            [&coordinates, &coordinates],
            [
                shape,
                crate::backend::PixalConditioningShape {
                    naf_side: Some(1024),
                    ..shape
                }
            ],
            [&views, &views]
        )
        .is_err()
    );
    assert_eq!(coordinates.len(), reordered.len());
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
                == "4382f7d999e51118220fcc32710a10a533b0d4b8ea294b332f272af4d43ac3ca",
            "M4 forced-stage freeze integrity mismatch"
        );
        let sibling: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            sibling["reference_valid"] == true,
            "invalid stage reference"
        );
        let bytes = std::fs::read(
            sibling["prior_freeze"]["path"]
                .as_str()
                .context("prior freeze missing")?,
        )?;
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
                if anchor.is_null() && c["sources"].is_object() && family != "stable-decisions" {
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
        if !geometry_only {
            for table in sibling["tables"]
                .as_array()
                .context("stage tables missing")?
            {
                let table_name = table["source_table"]
                    .as_str()
                    .context("stage table missing")?;
                for cell in table["cells"].as_array().context("stage cells missing")? {
                    let name = cell["name"].as_str().context("stage cell missing")?;
                    let key = id(table_name, name, None);
                    ensure!(cells.contains_key(&key), "unknown stable stage");
                    cells.insert(key, cell.clone());
                    let key = id(table_name, &format!("{name}-logits"), None);
                    cells.insert(
                        key.clone(),
                        serde_json::json!({
                            "tau":cell["numeric_tau"],
                            "frozen_gating_metrics":cell["numeric_gating_metrics"]
                        }),
                    );
                    anchors.insert(
                        key,
                        serde_json::json!({"path":cell["sources"]["fp16"],"key":"logits"}),
                    );
                    cells.insert(
                        id(table_name, &format!("{name}-own-mask-compaction"), None),
                        serde_json::json!({"tau":0}),
                    );
                }
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
    let decoder = PixalDecoder::open(
        &Device::new(0)?,
        model,
        DecoderKind::Structure,
        runtime,
        4_000_000_000,
    )?;
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
    let device_input = decoder.device.stream.clone_htod(&input)?;
    ensure!(
        decoder.structure_device(device_input)? == expected,
        "device CHW decoder handoff differs from host entry"
    );
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
        &Device::new(0)?,
        &model.join("ckpts"),
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        runtime,
        (&points, 32),
        &Rc::new(()),
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
    let latent = flow.sample(&noise, &positive, Some(&negative))?;
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
    let decoder = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Shape,
        &runtime,
        10_000_000_000,
    )?;
    pointwise(&mut g, &decoder)?;
    sparse_ops(&mut g, &decoder, &root)?;
    decoder_transitions(&mut g, &root)?;
    prefix(&mut g, &decoder, &root, &model, None)?;
    drop(decoder);
    let latent = linked_lr(&root, &model, &runtime)?;
    let decoder = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Shape,
        &runtime,
        10_000_000_000,
    )?;
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
    let texture = phase.starts_with("texture-");
    println!(
        "{}_COUNT {}",
        if texture { "P1_M4B" } else { "P1_M4A" },
        serde_json::json!({"case":case,"phase":phase,"stage":stage,"parent_rows":parent,"child_rows":child,"channels":channels,"parent_map_bytes":parent_map,"child_map_bytes":child_map,"parent_columns_bytes":parent_columns,"child_columns_bytes":child_columns,"parent_tile_rows":parent_tile,"child_tile_rows":child_tile,"gather_launches":calls,"bias_launches":calls,"gemm_dispatches":calls,"subdivision_d2h_bytes":if texture{0}else{parent*8*size_of::<f16>()},"child_pair_h2d_bytes":child*2*size_of::<u32>()})
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
            let anchor = g.anchor(&key)?;
            surface_signs(g, case, &format!("subdivision{stage}"), sub, &anchor.values)?;
            println!(
                "P1_M4A_FIXED_PROGRESS case={case} stage={stage} rows={}",
                next.coordinates().len()
            );
            Ok(())
        },
    )?;
    let output = decoder.output_head(&sites, features)?;
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
    let anchor = g.anchor(&key)?;
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

fn surface_selection(cell: &Value) -> Result<(Vec<[i32; 4]>, HostTensor, HostTensor)> {
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
    let tested = selected.values.iter().filter(|&&v| v != 0.0).count();
    ensure!(
        tested > 0
            && selected
                .values
                .as_chunks::<8>()
                .0
                .iter()
                .all(|row| row.iter().any(|&v| v != 0.0))
            && cell["selected_decisions"].as_u64() == Some(tested as u64)
            && cell["selected_parents"].as_u64() == Some(coordinates.len() as u64)
            && cell["common_parents"]
                .as_u64()
                .is_some_and(|n| n >= coordinates.len() as u64),
        "stable support changed"
    );
    Ok((coordinates, selected, signs))
}

fn surface_stable(
    g: &mut DecoderGate,
    case: &str,
    phase: &str,
    level: usize,
    parents: &SparseSites,
    logits: &[f32],
    gating: bool,
) -> Result<()> {
    let key = id(
        &format!("m4-stable-decisions-{case}"),
        &format!("{phase}-stage{level}"),
        None,
    );
    let (coordinates, selected, signs) = surface_selection(&g.cells[&key])?;
    let mut tested = 0;
    let mut mismatches = 0;
    let mut missing = 0;
    for (i, &point) in coordinates.iter().enumerate() {
        let row = parents.row(point);
        missing += usize::from(row.is_none());
        for bit in 0..8 {
            if selected.values[i * 8 + bit] != 0.0 {
                tested += 1;
                let mismatch = row.is_none_or(|row| {
                    (logits[row as usize * 8 + bit] > 0.0) != (signs.values[i * 8 + bit] != 0.0)
                });
                if mismatch && mismatches < 4 {
                    println!(
                        "P1_M4A_STABLE_DETAIL {}",
                        serde_json::json!({"case":case,"phase":phase,"level":level,"parent":point,"bit":bit,"candidate":row.map(|r| logits[r as usize * 8 + bit]),"expected_positive":signs.values[i * 8 + bit] != 0.0})
                    );
                }
                mismatches += usize::from(mismatch);
            }
        }
    }
    println!(
        "P1_M4A_STABLE_SUPPORT case={case} phase={phase} level={level} parents={} missing={missing}",
        coordinates.len()
    );
    if gating {
        g.exact_count(key, tested, mismatches)
    } else {
        println!(
            "P1_M4A_GENERATED_STABLE_MEASUREMENT case={case} phase={phase} level={level} tested={tested} mismatches={mismatches} gating=false"
        );
        Ok(())
    }
}

fn surface_own(
    g: &mut DecoderGate,
    table: &str,
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
            table,
            &format!("{phase}-stage{level}-own-mask-compaction"),
            None,
        ),
        row.max(children.coordinates().len()),
        mismatches,
    )
}

fn surface_divergence(
    label: [&str; 2],
    level: usize,
    parents: &SparseSites,
    logits: &[f32],
    path: &Path,
) -> Result<()> {
    let [case, phase] = label;
    let coordinates = decoder_coordinates(path, "subdivision.coords")?;
    let reference = f32_tensor(path, "subdivision.features")?;
    ensure!(
        reference.len() == coordinates.len() * 8,
        "reference subdivision extent mismatch"
    );
    let mut flips = 0;
    let mut missing = 0;
    let mut maximum = 0.0f32;
    for (point, values) in coordinates.iter().zip(reference.as_chunks::<8>().0) {
        let Some(row) = parents.row(*point) else {
            missing += 1;
            continue;
        };
        for (bit, &expected) in values.iter().enumerate() {
            let actual = logits[row as usize * 8 + bit];
            maximum = maximum.max((actual - expected).abs());
            if (actual > 0.0) != (expected > 0.0) {
                if flips < 4 {
                    println!(
                        "P1_M4A_BRANCH_DETAIL {}",
                        serde_json::json!({"case":case,"phase":phase,"level":level,"parent":point,"bit":bit,"candidate":actual,"r0a":expected,"gating":false})
                    );
                }
                flips += 1;
            }
        }
    }
    println!(
        "P1_M4A_BRANCH_MEASUREMENT {}",
        serde_json::json!({"case":case,"phase":phase,"level":level,"r0a_parents":coordinates.len(),"missing_parents":missing,"common_parent_max_abs":maximum,"sign_flips":flips,"gating":false})
    );
    Ok(())
}

fn surface_forced(g: &mut DecoderGate, decoder: &PixalDecoder, case: &str) -> Result<()> {
    let table = format!("m4-stable-decisions-{case}");
    let d = &decoder.device;
    for phase in ["LR", "HR"] {
        for (level, blocks) in [4, 16, 8, 4].into_iter().enumerate() {
            let name = format!("{phase}-stage{level}");
            let cell = &g.cells[&id(&table, &name, None)];
            let path = Path::new(
                cell["forced_common_input"]["path"]
                    .as_str()
                    .context("stage input missing")?,
            );
            let parents = SparseSites::new(decoder_coordinates(path, "coords")?)?;
            let anchor_path = Path::new(
                cell["sources"]["fp16"]
                    .as_str()
                    .context("stage anchor missing")?,
            );
            ensure!(
                decoder_coordinates(anchor_path, "coords")? == parents.coordinates(),
                "stage anchor row identity mismatch"
            );
            let host = tensor(path, "features")?;
            let n = parents.coordinates().len();
            let channels = 1024 >> level;
            ensure!(
                host.shape == [n, channels] && host.values.iter().all(|x| x.is_finite()),
                "stage input layout or finite check failed"
            );
            let input: Vec<_> = host.values.iter().map(|&x| f16::from_f64(x)).collect();
            drop(host);
            let x = d.stream.clone_htod(&input)?;
            drop(input);
            let budget = d.workspace_budget(
                decoder.initial_free,
                decoder.device_budget,
                n * channels * 14,
            )?;
            let conv = SparseConv::new(d, &parents, channels, budget)?;
            let parent_bytes = conv.workspace_bytes();
            let x = decoder.next_blocks(level, &conv, x, channels, blocks)?;
            let (children, features, logits, child_conv) =
                decoder.up_stage(level, &parents, x, conv, None)?;
            surface_count(
                case,
                &format!("forced-{phase}"),
                [level, parent_bytes, child_conv.workspace_bytes()],
                [n, children.coordinates().len()],
            )?;
            g.score_f32(
                id(&table, &format!("{name}-logits"), None),
                &[n, 8],
                &logits,
            )?;
            surface_stable(g, case, phase, level, &parents, &logits, true)?;
            surface_own(g, &table, phase, level, &parents, &logits, &children)?;
            drop(features);
        }
    }
    Ok(())
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
        channel_transform(d, &mut latent, [&mean, &std], false)?;
        let physical = d.stream.clone_dtoh(&latent)?;
        drop(latent);
        let (sites, features) = decoder.sparse_decode(coords, &physical, &[], |plan @ [level, _, _], (parents, logits), (next, _)| {
            surface_count(case, phase, plan, [parents.coordinates().len(), next.coordinates().len()])?;
            surface_stable(g, case, phase, level, parents, logits, false)?;
            surface_own(g, &format!("m4-own-branch-contract-{core}"), phase, level, parents, logits, next)?;
            let reference = base.join(format!("shape_slat_decoder.blocks.{level}.{}-call{call}-topology.safetensors", [4,16,8,4][level]));
            surface_divergence([case, phase], level, parents, logits, &reference)?;
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
            let fields = decoder.output_head(&sites, features)?;
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
            channel_transform(&d, &mut values, [&mean, &std], false)?;
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

const SURFACE_CASES: [(&str, &str, &str, &str, u32); 5] = [
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
];

#[test]
#[ignore = "requires frozen surface captures; CPU only"]
fn pixal3d_surface_freeze_cpu() -> Result<()> {
    for (geometry, expected) in [(true, 85), (false, 236)] {
        let gate = DecoderGate::surface(geometry)?;
        ensure!(
            gate.cells.len() == expected,
            "surface cell coverage mismatch"
        );
        for (key, cell) in &gate.cells {
            if cell["selection_fixture"].is_string() {
                ensure!(
                    !gate.anchors.contains_key(key)
                        && cell["selection_fixture"].is_string()
                        && cell["selected_decisions"].as_u64().is_some_and(|n| n > 0),
                    "invalid stable-decision support"
                );
                surface_selection(cell)?;
                for fixture in cell["reference_fixtures"]
                    .as_object()
                    .context("stage fixtures missing")?
                    .values()
                {
                    let bytes =
                        std::fs::read(fixture["path"].as_str().context("fixture path missing")?)?;
                    ensure!(
                        format!("{:x}", Sha256::digest(&bytes))
                            == fixture["sha256"]
                                .as_str()
                                .context("fixture digest missing")?,
                        "stage fixture integrity mismatch"
                    );
                }
            }
        }
    }
    let mut gate = DecoderGate::surface(false)?;
    for (case, _, table, _, _) in SURFACE_CASES {
        for level in 0..5 {
            let name = if level == 4 {
                "edges".to_owned()
            } else {
                format!("subdivision{level}")
            };
            let field = if level == 4 {
                "shape-fields".to_owned()
            } else {
                format!("shape-{name}")
            };
            let key = id(table, &field, None);
            let sign_key = id(&format!("m4-decisions-{case}.json"), &name, None);
            ensure!(
                gate.cells[&key]["comparison_anchor"] == gate.cells[&sign_key]["comparison_anchor"],
                "sign and numeric comparison anchors differ"
            );
            let anchor = gate.anchor(&key)?;
            let reference = if level == 4 {
                anchor
                    .values
                    .as_chunks::<7>()
                    .0
                    .iter()
                    .flat_map(|row| row[3..6].iter().copied())
                    .collect()
            } else {
                anchor.values
            };
            let candidate: Vec<_> = reference.iter().map(|&v| v as f32).collect();
            surface_signs(&mut gate, case, &name, &candidate, &reference)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, published checkpoints and frozen decoder captures"]
fn pixal3d_generated_topology_isolation_gpu() -> Result<()> {
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let base = root.join("evidence/P1-M4-freeze-r1/core-single-1024-cottage-r0a-r2");
    let input = base.join("shape_slat-0-outputs.safetensors");
    let coordinates = decoder_coordinates(&input, "samples.coords")?;
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let norm = &config["args"]["shape_slat_normalization"];
    let channels = |key: &str| -> Result<Vec<f32>> {
        norm[key]
            .as_array()
            .context("shape normalization array missing")?
            .iter()
            .map(|x| Ok(x.as_f64().context("shape normalization value missing")? as f32))
            .collect()
    };
    let mean = channels("mean")?;
    let std = channels("std")?;
    let decoder = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Shape,
        &runtime,
        12_000_000_000,
    )?;
    let d = &decoder.device;
    let mut latent = d
        .stream
        .clone_htod(&f32_tensor(&input, "samples.features")?)?;
    channel_transform(d, &mut latent, [&mean, &std], false)?;
    let physical = d.stream.clone_dtoh(&latent)?;
    drop(latent);
    let mut guide_coords = Vec::new();
    let mut guide_logits = Vec::new();
    let mut paths = Vec::new();
    for (level, block) in [4, 16, 8, 4].into_iter().enumerate() {
        let path = base.join(format!(
            "shape_slat_decoder.blocks.{level}.{block}-call0-topology.safetensors"
        ));
        guide_coords.push(decoder_coordinates(&path, "subdivision.coords")?);
        guide_logits.push(f32_tensor(&path, "subdivision.features")?);
        paths.push(path);
    }
    let guides: Vec<_> = guide_coords
        .iter()
        .zip(&guide_logits)
        .map(|(coordinates, logits)| SubdivisionGuide {
            coordinates,
            logits,
        })
        .collect();
    let mut baseline_logits = Vec::new();
    let mut baseline_parents = Vec::new();
    for forced_prefix in [0, 2, 4] {
        println!("P1_M4A_TOPOLOGY_ISOLATION_BEGIN forced_prefix={forced_prefix} gating=false");
        let mut gate = DecoderGate::surface(false)?;
        gate.package = "P1_M4A_TOPOLOGY_ISOLATION";
        let (sites, features) = decoder.sparse_decode(
            coordinates.clone(),
            &physical,
            &guides[..forced_prefix],
            |[level, _, _], (parents, logits), (next, _)| {
                if level < 2 {
                    if forced_prefix == 0 {
                        baseline_logits.push(logits.to_vec());
                        baseline_parents.push(parents.coordinates().to_vec());
                    } else {
                        ensure!(
                            logits == baseline_logits[level]
                                && parents.coordinates() == baseline_parents[level],
                            "topology intervention changed an upstream input or logit"
                        );
                    }
                }
                let reference_children = decoder_coordinates(&paths[level], "child_coords")?;
                if level == 0 || level < forced_prefix {
                    ensure!(
                        next.coordinates() == reference_children,
                        "recorded topology intervention differs from reference"
                    );
                }
                println!(
                    "P1_M4A_TOPOLOGY_ISOLATION_STAGE forced_prefix={forced_prefix} level={level} parents={} children={} reference_children={} gating=false",
                    parents.coordinates().len(), next.coordinates().len(), reference_children.len()
                );
                surface_stable(&mut gate, "cottage1024", "LR", level, parents, logits, false)?;
                surface_divergence(["cottage1024", "LR"], level, parents, logits, &paths[level])?;
                let point = [0, 38, 27, 60];
                if level == 2 {
                    println!(
                        "P1_M4A_TOPOLOGY_ISOLATION_POINT {}",
                        serde_json::json!({"forced_prefix":forced_prefix,"parent":point,"bit":6,"logit":parents.row(point).map(|row| logits[row as usize * 8 + 6]),"gating":false})
                    );
                }
                Ok(())
            },
        )?;
        drop(features);
        println!(
            "P1_M4A_TOPOLOGY_ISOLATION_END forced_prefix={forced_prefix} scored_cells={} mismatching_cells={} final_rows={} gating=false",
            gate.seen.len(),
            gate.defects,
            sites.coordinates().len()
        );
    }
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
    let decoder = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Shape,
        &runtime,
        22_000_000_000,
    )?;
    for (case, capture, table, core, requested) in SURFACE_CASES {
        surface_forced(&mut g, &decoder, case)?;
        surface_fixed(&mut g, &decoder, &root, (case, capture, table))?;
        surface_generated(&mut g, &decoder, &root, &model, (case, core, requested))?;
    }
    g.finish()
}

impl DecoderGate {
    fn texture(flow: bool) -> Result<Self> {
        let bytes = std::fs::read(env_path("FF_PIXAL_FREEZE")?)?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes))
                == "bc06c2394b677124a2041bacd29167d93a9a679e0205c4e229fd0afcefc0a3db",
            "texture freeze integrity mismatch"
        );
        let freeze: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            freeze["reference_valid"] == true,
            "invalid texture reference"
        );
        let mut cells = BTreeMap::new();
        let mut anchors = BTreeMap::new();
        for table in freeze["tables"]
            .as_array()
            .context("texture tables missing")?
        {
            let family = table["family"].as_str().context("family missing")?;
            let table_name = table["source_table"].as_str().context("table missing")?;
            for cell in table["cells"].as_array().context("cells missing")? {
                let name = cell["name"].as_str().context("name missing")?;
                let selected = if flow {
                    family == "slat"
                } else {
                    (family == "decoder-fixed" && name == "tex-fields")
                        || name == "guided-texture-coordinate-identity"
                        || name.starts_with("normalization-")
                        || (family == "transition" && name.ends_with("-tex"))
                };
                if !selected {
                    continue;
                }
                let key = id(table_name, name, cell["key"].as_str());
                let mut anchor = cell["anchor"].clone();
                if anchor.is_null() && cell["sources"].is_object() {
                    anchor = cell["sources"][cell["comparison_anchor"]
                        .as_str()
                        .context("texture anchor missing")?]
                    .clone();
                }
                if !anchor.is_null() {
                    anchors.insert(key.clone(), anchor);
                }
                for (metric, ratio) in cell["ratios"].as_object().into_iter().flatten() {
                    if cell["frozen_gating_metrics"][metric] == true {
                        ensure!(
                            ratio.as_f64().context("control ratio missing")? >= 10.0,
                            "weak frozen texture control"
                        );
                    }
                }
                ensure!(
                    cells.insert(key, cell.clone()).is_none(),
                    "duplicate texture cell"
                );
            }
        }
        println!(
            "P1_M4B_FREEZE flow={flow} cells={} minimum_control_ratio={}",
            cells.len(),
            freeze["minimum_gated_control_ratio"]
        );
        Ok(Self {
            package: if flow {
                "P1_M4B_FLOW"
            } else {
                "P1_M4B_MATERIAL"
            },
            cells,
            anchors,
            seen: BTreeSet::new(),
            defects: 0,
        })
    }
}

fn texture_launch_contract(positive: usize, negative: usize, subdivision: usize) -> Result<()> {
    ensure!(
        positive == 12 && negative == 0 && subdivision == 0,
        "texture launch contract defect"
    );
    Ok(())
}

#[test]
fn guided_material_contract_rejects_mutants() -> Result<()> {
    let coordinates = [[0, 1, 3, 5], [0, 7, 11, 13]];
    let logits: Vec<_> = (0..16).map(|i| (i as f32 - 7.0) / 3.0).collect();
    let guide = SubdivisionGuide {
        coordinates: &coordinates,
        logits: &logits,
    };
    guide.validate(&coordinates)?;
    let mut foreign = coordinates;
    foreign.swap(0, 1);
    ensure!(
        guide.validate(&foreign).is_err(),
        "foreign guide mutant survived"
    );
    validate_guide_count(DecoderKind::Texture, 4)?;
    ensure!(
        validate_guide_count(DecoderKind::Texture, 3).is_err(),
        "missing guide mutant survived"
    );
    texture_launch_contract(12, 0, 0)?;
    for mutant in [(11, 0, 0), (12, 1, 0), (12, 0, 1)] {
        ensure!(
            texture_launch_contract(mutant.0, mutant.1, mutant.2).is_err(),
            "launch mutant survived"
        );
    }
    ensure!(
        !DecoderKind::Texture
            .inventory()
            .keys()
            .any(|k| k.contains("to_subdiv")),
        "texture predictor weights present"
    );
    println!(
        "P1_M4B_CONTRACT_CPU PASS foreign_guide=rejected missing_guide=rejected positive_count_mutant=rejected negative_count_mutant=rejected subdivision_launch_mutant=rejected"
    );
    Ok(())
}

#[test]
#[ignore = "requires frozen texture captures; CPU only"]
fn pixal3d_texture_freeze_cpu() -> Result<()> {
    for flow in [true, false] {
        let gate = DecoderGate::texture(flow)?;
        ensure!(
            gate.cells.len() == if flow { 190 } else { 16 },
            "texture subset mismatch"
        );
        for anchor in gate.anchors.values() {
            let path = Path::new(anchor["path"].as_str().context("anchor path missing")?);
            ensure!(path.is_file(), "texture anchor absent: {}", path.display());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, published texture checkpoints and frozen captures"]
fn pixal3d_texture_flow_matches_frozen_definition_gpu() -> Result<()> {
    let selected = DecoderGate::texture(true)?;
    let mut gate = Gate {
        package: selected.package,
        cells: selected.cells,
        anchors: selected.anchors,
        seen: BTreeSet::new(),
        defects: 0,
    };
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    for (case, capture, _, core, requested) in SURFACE_CASES {
        let base = root.join("evidence/P1-M1-r1").join(capture);
        let input = base.join("tex_slat-0-inputs.safetensors");
        let points = coordinates(&tensor(&input, "noise.coords")?)?;
        let stem = format!(
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16{}",
            if core.starts_with("mv-") { "_mv" } else { "" }
        );
        let mut flow = Flow::open(
            &Device::new(0)?,
            &model.join("ckpts"),
            &stem,
            &runtime,
            (
                &points,
                if case == "crab1408" {
                    88
                } else {
                    requested / 16
                },
            ),
            &Rc::new(()),
            12_000_000_000,
        )?;
        let mut positive = flow.conditioning(
            &f32_tensor(&input, "cond.global")?,
            &f32_tensor(&input, "cond.proj.features")?,
            &coordinates(&tensor(&input, "cond.proj.coords")?)?,
        )?;
        let physical_base = if case == "crab1408" {
            root.join("evidence/P1-M1-r1")
                .join(format!("{capture}-decoder-r2"))
        } else {
            base.clone()
        };
        let physical_path = physical_base.join("result.safetensors");
        let physical = f32_tensor(&physical_path, "result.1.0.features")?;
        let shape_coords = coordinates(&tensor(&physical_path, "result.1.0.coords")?)?;
        flow.condition_shape(&mut positive, &physical, &shape_coords)?;
        let mut wrong = shape_coords.clone();
        wrong.rotate_left(1);
        ensure!(
            flow.condition_shape(&mut positive, &physical, &wrong)
                .is_err(),
            "foreign concat mutant survived"
        );
        let actual_core = if case == "crab1408" {
            "single-1408-robot_crab"
        } else {
            core
        };
        let table = format!("floors-slat-{actual_core}-tex-angular-r1.json");
        let coords = tensor(&input, "noise.coords")?;
        gate.score(
            id(&table, "coordinate_alignment", None),
            &coords.shape,
            &flow
                .coordinates
                .iter()
                .flat_map(|p| [0.0, p[0] as f64, p[1] as f64, p[2] as f64])
                .collect::<Vec<_>>(),
        )?;
        println!(
            "P1_M4B_ATTENTION case={case} rows={} heads={} lse_bytes={} quadratic_workspace_bytes=0 route=online_softmax",
            flow.rows,
            flow.spec.num_heads,
            flow.flash.lse.len() * size_of::<f32>()
        );
        flow_cells(
            &mut flow,
            &mut gate,
            FlowCase {
                table: &table,
                fixture: None,
                shape: vec![points.len(), 32],
            },
            &f32_tensor(&input, "noise.features")?,
            &positive,
            None,
        )?;
    }
    gate.finish()
}

fn material_pointwise(g: &mut DecoderGate, model: &Path, root: &Path) -> Result<()> {
    let d = Device::new(0)?;
    let path = env_path("FF_PIXAL_NORMALIZATION")?;
    let shape = tensor(&path, "physical_shape")?;
    let noise = f32_tensor(&path, "noise")?;
    let mut normalized = d
        .stream
        .clone_htod(&shape.values.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
    let mean = f32_tensor(&path, "shape_mean")?;
    let std = f32_tensor(&path, "shape_std")?;
    channel_transform(&d, &mut normalized, [&mean, &std], true)?;
    g.score_f32(
        id("m4-geometry-pointwise", "normalization-shape_inverse", None),
        &shape.shape,
        &d.stream.clone_dtoh(&normalized)?,
    )?;
    let mut denorm = d.stream.clone_htod(&noise)?;
    channel_transform(
        &d,
        &mut denorm,
        [
            &f32_tensor(&path, "tex_mean")?,
            &f32_tensor(&path, "tex_std")?,
        ],
        false,
    )?;
    g.score_f32(
        id("m4-geometry-pointwise", "normalization-tex_denorm", None),
        &shape.shape,
        &d.stream.clone_dtoh(&denorm)?,
    )?;
    drop(denorm);
    let noise_device = d.stream.clone_htod(&noise)?;
    let concat = d.stream.alloc_zeros::<f32>(noise.len() * 2)?;
    let kernel = cubin::module(&d.ctx, "q38_common")?.load_function("concat_rows_f32")?;
    unsafe {
        d.stream
            .launch_builder(&kernel)
            .arg(&noise_device)
            .arg(&normalized)
            .arg(&concat)
            .arg(&(shape.shape[0] as i64))
            .arg(&32i32)
            .arg(&32i32)
            .launch(flat(noise.len() * 2))?;
    }
    let actual = d.stream.clone_dtoh(&concat)?;
    let normalized_host = d.stream.clone_dtoh(&normalized)?;
    let cpu: Vec<f32> = noise
        .as_chunks::<32>()
        .0
        .iter()
        .zip(normalized_host.as_chunks::<32>().0.iter())
        .flat_map(|(a, b)| a.iter().chain(b).copied())
        .collect();
    let valid_concat = |values: &[f32]| -> Result<()> {
        ensure!(values == cpu, "row concat defect");
        Ok(())
    };
    valid_concat(&actual)?;
    let reversed: Vec<f32> = normalized_host
        .as_chunks::<32>()
        .0
        .iter()
        .zip(noise.as_chunks::<32>().0.iter())
        .flat_map(|(a, b)| a.iter().chain(b).copied())
        .collect();
    ensure!(
        valid_concat(&reversed).is_err(),
        "reversed halves mutant survived"
    );
    g.score_f32(
        id("m4-geometry-pointwise", "normalization-concat", None),
        &[shape.shape[0], 64],
        &actual,
    )?;
    drop((concat, noise_device, normalized));
    let raw: Vec<f32> = (0..257 * 6)
        .map(|i| ((i * 37 % 103) - 51) as f32 / 13.0)
        .collect();
    let mut material = d.stream.clone_htod(&raw)?;
    channel_transform(&d, &mut material, [&[0.5; 6], &[0.5; 6]], false)?;
    let expected: Vec<f32> = raw.iter().map(|&v| v * 0.5 + 0.5).collect();
    let validate = |values: &[f32]| -> Result<()> {
        ensure!(values == expected, "material channel/affine defect");
        Ok(())
    };
    validate(&d.stream.clone_dtoh(&material)?)?;
    let mut swapped = expected.clone();
    for row in swapped.as_chunks_mut::<6>().0 {
        row.swap(3, 4);
    }
    ensure!(validate(&swapped).is_err(), "PBR channel mutant survived");
    let clipped: Vec<_> = expected.iter().map(|&x| x.clamp(0.0, 1.0)).collect();
    ensure!(
        validate(&clipped).is_err(),
        "material clamp mutant survived"
    );
    println!(
        "P1_M4B_POINTWISE_CONTRACT PASS reversed_halves=rejected channel_order=rejected clamp=rejected"
    );
    let config: Value = serde_json::from_slice(&std::fs::read(model.join("pipeline.json"))?)?;
    let norm = &config["args"]["tex_slat_normalization"];
    let mean: Vec<f32> = serde_json::from_value(norm["mean"].clone())?;
    let std: Vec<f32> = serde_json::from_value(norm["std"].clone())?;
    for (fixture, capture) in [
        ("cottage", "pipeline-single-1536-cottage-r0a"),
        ("robot_crab", "pipeline-single-1536-robot_crab-r0a"),
        ("mv4", "pipeline-mv-1024-mv4-r0a"),
    ] {
        let input = root
            .join("evidence/P1-M1-r1")
            .join(capture)
            .join("tex_slat-0-outputs.safetensors");
        let host = tensor(&input, "samples.features")?;
        let mut values = d
            .stream
            .clone_htod(&host.values.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
        channel_transform(&d, &mut values, [&mean, &std], false)?;
        g.score_f32(
            id(
                "floors-transition-published.json",
                &format!("{fixture}-tex"),
                None,
            ),
            &host.shape,
            &d.stream.clone_dtoh(&values)?,
        )?;
    }
    Ok(())
}

fn material_fixed(
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
    let input = base.join("result.safetensors");
    let coords = decoder_coordinates(&input, "result.1.1.coords")?;
    let latent = f32_tensor(&input, "result.1.1.features")?;
    let topology = base.join("raw-geometry-material.safetensors");
    let mut stored = Vec::new();
    for stage in 0..4 {
        stored.push(SparseDecoderOutput {
            coordinates: decoder_coordinates(&topology, &format!("subdivisions.{stage}.coords"))?,
            features: f32_tensor(&topology, &format!("subdivisions.{stage}.features"))?,
            channels: 8,
        });
    }
    let guides: Vec<_> = stored
        .iter()
        .map(|s| SubdivisionGuide {
            coordinates: &s.coordinates,
            logits: &s.features,
        })
        .collect();
    let before = decoder.subdivision_launches.get();
    let (sites, features) = decoder.sparse_decode(
        coords,
        &latent,
        &guides,
        |plan, (parents, predicted), (next, _)| {
            ensure!(predicted.is_empty(), "texture computed subdivision logits");
            surface_count(
                case,
                "texture-fixed",
                plan,
                [parents.coordinates().len(), next.coordinates().len()],
            )?;
            Ok(())
        },
    )?;
    let output = decoder.output_head(&sites, features)?;
    let actual = decoder.device.stream.clone_dtoh(&output)?;
    let expected_coords = decoder_coordinates(
        &base.join("tex_slat_decoder-fields.safetensors"),
        "output.coords",
    )?;
    ensure!(
        sites.coordinates() == expected_coords,
        "fixed texture coordinate mismatch"
    );
    g.score_f32(
        id(table, "tex-fields", None),
        &[sites.coordinates().len(), 6],
        &actual,
    )?;
    texture_launch_contract(12, 0, decoder.subdivision_launches.get() - before)?;
    ensure!(
        texture_launch_contract(12, 0, 1).is_err(),
        "texture subdivision mutant survived"
    );
    println!(
        "P1_M4B_MATERIAL_FIXED case={case} rows={} subdivision_launches=0",
        sites.coordinates().len()
    );
    Ok(())
}

fn material_generated(
    g: &mut DecoderGate,
    shape: &PixalDecoder,
    texture: &PixalDecoder,
    root: &Path,
    case: (&str, &str),
) -> Result<()> {
    let (case, capture) = case;
    let capture = if case == "crab1408" {
        format!("{capture}-decoder-r2")
    } else {
        capture.to_owned()
    };
    let input = root
        .join("evidence/P1-M1-r1")
        .join(capture)
        .join("result.safetensors");
    let coords = decoder_coordinates(&input, "result.1.0.coords")?;
    let latent = f32_tensor(&input, "result.1.0.features")?;
    let (shape_output, guides) = shape.decode(coords.clone(), &latent, &[])?;
    let expected = shape_output.coordinates;
    drop(shape_output.features);
    let latent = f32_tensor(&input, "result.1.1.features")?;
    ensure!(
        decoder_coordinates(&input, "result.1.1.coords")? == coords,
        "shape/texture latent identity mismatch"
    );
    let borrowed: Vec<_> = guides
        .iter()
        .map(|s| SubdivisionGuide {
            coordinates: &s.coordinates,
            logits: &s.features,
        })
        .collect();
    let mut tested = 0;
    let mut mismatches = 0;
    let before = texture.subdivision_launches.get();
    let (sites, features) = texture.sparse_decode(
        coords,
        &latent,
        &borrowed,
        |plan @ [stage, _, _], (parents, predicted), (next, _)| {
            ensure!(predicted.is_empty(), "texture predicted subdivision");
            let guide = &guides[stage];
            ensure!(
                parents.coordinates() == guide.coordinates,
                "generated parent identity mismatch"
            );
            let mask: Vec<[bool; 8]> = guide
                .features
                .as_chunks::<8>()
                .0
                .iter()
                .map(|r| r.map(|x| x > 0.0))
                .collect();
            let (expected, _) = parents.subdivide(&mask, usize::MAX)?;
            tested += expected.coordinates().len();
            mismatches += expected
                .coordinates()
                .iter()
                .zip(next.coordinates())
                .filter(|(a, b)| a != b)
                .count()
                + expected
                    .coordinates()
                    .len()
                    .abs_diff(next.coordinates().len());
            surface_count(
                case,
                "texture-generated",
                plan,
                [parents.coordinates().len(), next.coordinates().len()],
            )?;
            Ok(())
        },
    )?;
    drop(features);
    tested += expected.len();
    mismatches += expected
        .iter()
        .zip(sites.coordinates())
        .filter(|(a, b)| a != b)
        .count()
        + expected.len().abs_diff(sites.coordinates().len());
    texture_launch_contract(12, 0, texture.subdivision_launches.get() - before)?;
    let core = SURFACE_CASES
        .iter()
        .find(|c| c.0 == case)
        .context("unknown material case")?
        .3;
    g.exact_count(
        id(
            &format!("m4-own-branch-contract-{core}"),
            "guided-texture-coordinate-identity",
            None,
        ),
        tested,
        mismatches,
    )
}

#[test]
#[ignore = "requires CUDA, published material checkpoints and frozen captures"]
fn pixal3d_material_matches_frozen_definition_gpu() -> Result<()> {
    let mut g = DecoderGate::texture(false)?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    material_pointwise(&mut g, &model, &root)?;
    let shape = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Shape,
        &runtime,
        22_000_000_000,
    )?;
    let texture = PixalDecoder::open(
        &Device::new(0)?,
        &model,
        DecoderKind::Texture,
        &runtime,
        22_000_000_000,
    )?;
    for (case, capture, table, _, _) in SURFACE_CASES {
        material_fixed(&mut g, &texture, &root, (case, capture, table))?;
        material_generated(&mut g, &shape, &texture, &root, (case, capture))?;
    }
    g.finish()
}

fn indexed_projection(
    device: &Device,
    input: u64,
    output: &CudaSlice<f32>,
    camera: &CudaSlice<f32>,
    coordinates: &CudaSlice<i32>,
    geometry: [u32; 6],
) -> Result<()> {
    let [side, height, width, channels, output_channels, offset] = geometry;
    let rows = u32::try_from(coordinates.len() / 3)?;
    let kernel =
        cubin::module(&device.ctx, "vision_ops")?.load_function("project_bilinear_indexed_f32")?;
    for first in (0..rows).step_by(8191) {
        let count = (rows - first).min(8191);
        unsafe {
            device
                .stream
                .launch_builder(&kernel)
                .arg(&input)
                .arg(output)
                .arg(camera)
                .arg(coordinates)
                .arg(&side)
                .arg(&height)
                .arg(&width)
                .arg(&channels)
                .arg(&output_channels)
                .arg(&offset)
                .arg(&u64::from(first))
                .arg(&count)
                .arg(&0u32)
                .arg(&1.0f32)
                .launch(grid(
                    (u64::from(count) * u64::from(channels))
                        .div_ceil(256)
                        .min(65535) as usize,
                    256,
                ))?;
        }
    }
    Ok(())
}
fn projection_features(
    device: &Device,
    source: &Path,
    output: &CudaSlice<f32>,
    camera: &CudaSlice<f32>,
    coords: &CudaSlice<i32>,
    side: u32,
) -> Result<()> {
    let s = &device.stream;
    let hidden = f32_tensor(source, "dino_final_block")?;
    ensure!(hidden.len() == 4101 * 1024, "projection DINO input shape");
    let input = s.clone_htod(&hidden)?;
    let normalized = s.alloc_zeros::<f32>(hidden.len())?;
    let common = cubin::module(&device.ctx, "q38_common")?;
    let norm = common.load_function("q38_layer_norm_f32")?;
    unsafe {
        s.launch_builder(&norm)
            .arg(&input)
            .arg(&0u64)
            .arg(&0u64)
            .arg(&normalized)
            .arg(&4101u32)
            .arg(&1u32)
            .arg(&1024u32)
            .arg(&1024u64)
            .arg(&1024u64)
            .arg(&1u64)
            .arg(&1024u64)
            .arg(&1024u64)
            .arg(&1u64)
            .arg(&1e-5f32)
            .launch(grid(4101, 256))?;
    }
    indexed_projection(
        device,
        ptr(&normalized, device) + 5 * 1024 * 4,
        output,
        camera,
        coords,
        [side, 64, 64, 1024, 2048, 0],
    )?;
    let checkpoint = MappedCheckpoint::open_component(
        source.parent().context("feature parent")?,
        source
            .file_stem()
            .and_then(|x| x.to_str())
            .context("feature stem")?,
    )?;
    let tensors = checkpoint.component_tensors()?;
    let naf = tensors.get("naf_output").context("NAF features missing")?;
    ensure!(
        naf.dtype == "F32"
            && naf.shape.len() == 3
            && naf.shape[0] == 1024
            && naf.shape[1] == naf.shape[2],
        "projection NAF input shape"
    );
    let extent = naf.shape[1];
    let plane = extent * extent;
    let transpose = common.load_function("transpose_f32")?;
    for channel in (0..1024).step_by(32) {
        let values: Vec<_> = naf.data[channel * plane * 4..(channel + 32) * plane * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let chw = s.clone_htod(&values)?;
        let hwc = s.alloc_zeros::<f32>(values.len())?;
        unsafe {
            s.launch_builder(&transpose)
                .arg(&chw)
                .arg(&hwc)
                .arg(&32i64)
                .arg(&(plane as i64))
                .launch(flat(values.len()))?;
        }
        indexed_projection(
            device,
            ptr(&hwc, device),
            output,
            camera,
            coords,
            [
                side,
                extent as u32,
                extent as u32,
                32,
                2048,
                (1024 + channel) as u32,
            ],
        )?;
        s.synchronize()?;
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA and frozen projection captures"]
fn pixal3d_indexed_projection_gpu() -> Result<()> {
    let freeze_path = env_path("FF_PIXAL_M4C_FREEZE")?;
    let bytes = std::fs::read(&freeze_path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "301489b487d9ee4fccbae6dac7764612fd70807535184731bef92d79c490a6cf",
        "projection freeze integrity"
    );
    let freeze: Value = serde_json::from_slice(&bytes)?;
    let root = freeze_path.parent().context("freeze parent")?;
    let project = env_path("FF_PIXAL_REFERENCE")?;
    let synthetic: Value = serde_json::from_slice(&std::fs::read(
        root.join("projection-synthetic-cpu/metadata.json"),
    )?)?;
    let real: Value = serde_json::from_slice(&std::fs::read(
        root.join("projection-real-cpu/metadata.json"),
    )?)?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let module = cubin::module(&device.ctx, "vision_ops")?;
    let focal = module.load_function("perspective_focal_f32")?;
    let cells = freeze["projection"]["cells"]
        .as_array()
        .context("projection cells")?;
    ensure!(cells.len() == 33, "projection coverage");
    let mut defects = 0;
    let mut seen = BTreeSet::new();
    for cell in cells {
        let name = cell["name"].as_str().context("cell name")?;
        ensure!(seen.insert(name), "duplicate projection cell");
        let is_synthetic = cell["kind"] == "synthetic";
        let metadata = if is_synthetic { &synthetic } else { &real };
        let record = metadata["records"]
            .as_array()
            .context("projection metadata")?
            .iter()
            .find(|r| r["name"] == name)
            .context("projection record")?;
        let side = u32::try_from(record["grid"].as_u64().context("projection grid")?)?;
        let fixture = record["fixture"].as_str().context("projection fixture")?;
        let camera: Value = serde_json::from_slice(&std::fs::read(
            project.join("fixtures/pixal3d").join(format!(
                "{}-camera.json",
                if fixture == "near" {
                    "cottage"
                } else {
                    fixture
                }
            )),
        )?)?;
        let mut parameters: Vec<f32> =
            serde_json::from_value::<Vec<Vec<f32>>>(camera["world_to_camera"].clone())?
                .into_iter()
                .take(3)
                .flatten()
                .collect();
        if fixture == "near" {
            parameters[11] = -0.125;
        }
        parameters.extend([
            camera["camera_angle_x"].as_f64().context("fov")? as f32,
            camera["mesh_scale"].as_f64().context("scale")? as f32,
            1024.0,
        ]);
        ensure!(parameters.len() == 15, "camera extent");
        let camera = s.clone_htod(&parameters)?;
        unsafe {
            s.launch_builder(&focal).arg(&camera).launch(grid(1, 1))?;
        }
        let path = Path::new(
            cell["sources"]["cpu"]
                .as_str()
                .context("projection source")?,
        );
        let coordinates = coordinates(&tensor(path, "coords")?)?;
        ensure!(
            coordinates
                .iter()
                .flatten()
                .all(|&v| v >= 0 && v < side as i32),
            "projection coordinate bound"
        );
        let coords = s.clone_htod(coordinates.as_flattened())?;
        let channels = if is_synthetic { 17 } else { 2048 };
        let count = coordinates.len() * channels;
        let mut output = s.alloc_zeros::<f32>(count + 64)?;
        s.memcpy_htod(&[12345.25f32; 64], &mut output.slice_mut(count..))?;
        let synthetic_input = if is_synthetic {
            let mut data = Vec::with_capacity(31 * 37 * 17);
            for y in 0..31 {
                for x in 0..37 {
                    for c in 0..17 {
                        data.push(
                            ((y * 7 + x * 11 + c * 3) % 19 - 9) as f32 / 16.0
                                + ((y * 13 + x * 5 + c * 7) % 23 - 11) as f32 / 32.0,
                        );
                    }
                }
            }
            let input = s.clone_htod(&data)?;
            indexed_projection(
                &device,
                ptr(&input, &device),
                &output,
                &camera,
                &coords,
                [side, 31, 37, 17, 17, 0],
            )?;
            Some(input)
        } else {
            projection_features(
                &device,
                Path::new(
                    record["sources"]["r0a"]
                        .as_str()
                        .context("feature source")?,
                ),
                &output,
                &camera,
                &coords,
                side,
            )?;
            None
        };
        s.synchronize()?;
        ensure!(
            s.clone_dtoh(&output.slice(count..))? == [12345.25f32; 64],
            "projection canary"
        );
        let actual: Vec<f64> = s
            .clone_dtoh(&output.slice(..count))?
            .into_iter()
            .map(f64::from)
            .collect();
        let reference = tensor(path, if is_synthetic { "anchor" } else { "r0a" })?;
        let measured = metrics(&actual, &reference.values)?;
        let mut pass = true;
        for metric in ["max_abs", "rmse"] {
            let tau = cell["tau"][metric].as_f64().context("projection tau")?;
            if measured[metric] > tau {
                pass = false;
                defects += 1;
            }
            println!(
                "P1_M4C_PROJECTION_CELL {} name={name} metric={metric} value={} tau={tau}",
                if measured[metric] <= tau {
                    "PASS"
                } else {
                    "DEFECT"
                },
                measured[metric]
            );
        }
        let control = if let Some(input) = synthetic_input {
            let swapped: Vec<_> = coordinates.iter().map(|p| [p[2], p[1], p[0]]).collect();
            let coords = s.clone_htod(swapped.as_flattened())?;
            indexed_projection(
                &device,
                ptr(&input, &device),
                &output,
                &camera,
                &coords,
                [side, 31, 37, 17, 17, 0],
            )?;
            s.clone_dtoh(&output.slice(..count))?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        } else {
            let mut wrong = actual.clone();
            wrong.rotate_right(channels);
            wrong
        };
        let wrong = metrics(&control, &reference.values)?;
        for metric in ["max_abs", "rmse"] {
            let ratio = wrong[metric] / cell["tau"][metric].as_f64().context("control tau")?;
            ensure!(
                ratio >= 10.0,
                "projection control weak: {name} {metric} {ratio}"
            );
            println!("P1_M4C_PROJECTION_CONTROL PASS name={name} metric={metric} ratio={ratio}");
        }
        println!(
            "P1_M4C_PROJECTION_ROWS name={name} rows={} grid={side} indexed_output_bytes={} dense_output_bytes={} coordinate_bytes={} pass={pass}",
            coordinates.len(),
            count * 4,
            u64::from(side).pow(3) * channels as u64 * 4,
            coordinates.len() * 12
        );
    }
    println!(
        "P1_M4C_PROJECTION_GATE {} cells={} metrics={} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" },
        seen.len(),
        seen.len() * 2
    );
    ensure!(defects == 0, "indexed projection candidate defects");
    Ok(())
}

#[test]
#[ignore = "requires CUDA; conditioning stream, context and capacity controls"]
fn pixal3d_device_handoff_gpu() -> Result<()> {
    use std::io::Write;
    if let Ok(case) = std::env::var("FF_PIXAL_HANDOFF_CHILD") {
        ensure!(
            matches!(case.as_str(), "copy_nan" | "cast_nan" | "cast_overflow"),
            "invalid handoff control"
        );
        let device = Device::new(0)?;
        let mut values = vec![1.0f32; 257];
        values[128] = if case == "cast_overflow" {
            f32::MAX
        } else {
            f32::NAN
        };
        let input = device.stream.clone_htod(&values)?;
        if case == "copy_nan" {
            let output = device.stream.alloc_zeros::<f32>(257)?;
            finite_copy(&device, &input, &output)?;
            std::mem::forget(output);
        } else {
            let output = device.stream.alloc_zeros::<bf16>(257)?;
            let kernel = cubin::module(&device.ctx, "vision_ops")?
                .load_function("vision_downcast_f32_to_bf16_checked")?;
            unsafe {
                device
                    .stream
                    .launch_builder(&kernel)
                    .arg(&input)
                    .arg(&output)
                    .arg(&257u32)
                    .launch(flat(257))?;
            }
            std::mem::forget(output);
        }
        let error = device
            .stream
            .synchronize()
            .expect_err("invalid handoff value did not trap");
        ensure!(
            matches!(
                error.0,
                cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
                    | cudarc::driver::sys::CUresult::CUDA_ERROR_LAUNCH_FAILED
            ),
            "unexpected handoff failure: {error:?}"
        );
        println!("P1_M4C_HANDOFF_TRAP PASS case={case}");
        std::io::stdout().flush()?;
        std::process::exit(0);
    }
    for case in ["copy_nan", "cast_nan", "cast_overflow"] {
        let result = std::process::Command::new(std::env::current_exe()?)
            .args([
                "backend::cuda::pixal3d::tests::pixal3d_device_handoff_gpu",
                "--exact",
                "--include-ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("FF_PIXAL_HANDOFF_CHILD", case)
            .output()?;
        let output = String::from_utf8_lossy(&result.stdout);
        ensure!(
            result.status.success()
                && output.contains(&format!("P1_M4C_HANDOFF_TRAP PASS case={case}")),
            "handoff control {case} failed: {output} {}",
            String::from_utf8_lossy(&result.stderr)
        );
        print!("{output}");
    }
    let device = Device::new(0)?;
    let producer = device.ctx.new_stream()?;
    let mut source = producer.alloc_zeros::<f32>(257)?;
    producer.synchronize()?;
    let mut values: Vec<f32> = (0..257)
        .map(|i| ((i * 37 % 97) + 1) as f32 / 97.0)
        .collect();
    values[..6].copy_from_slice(
        &[
            0x3f807fff, 0x3f808000, 0x3f808001, 0x3f817fff, 0x3f818000, 0x3f818001,
        ]
        .map(f32::from_bits),
    );
    let stale = device.stream.alloc_zeros::<f32>(257)?;
    let copy = cubin::module(&device.ctx, "q38_common")?.load_function("copy_finite_f32")?;
    let source_ptr = source.device_ptr(&producer).0;
    let stale_ptr = ptr(&stale, &device);
    unsafe {
        device
            .stream
            .launch_builder(&copy)
            .arg(&source_ptr)
            .arg(&stale_ptr)
            .arg(&257u64)
            .launch(flat(257))?;
    }
    let release = device.stream.record_event(Some(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
    ))?;
    producer.wait(&release)?;
    producer.memcpy_htod(&values, &mut source)?;
    let output = device.stream.alloc_zeros::<f32>(257)?;
    finite_copy(&device, &source, &output)?;
    drop(source);
    drop(producer);
    ensure!(
        device.stream.clone_dtoh(&output)? == values,
        "producer handoff differs"
    );
    let stale_values = device.stream.clone_dtoh(&stale)?;
    ensure!(
        stale_values.iter().all(|&v| v == 0.0) && stale_values != values,
        "missing-dependency control did not expose stale contents"
    );
    let cast = cubin::module(&device.ctx, "vision_ops")?
        .load_function("vision_downcast_f32_to_bf16_checked")?;
    let cast_output = device.stream.alloc_zeros::<bf16>(257)?;
    unsafe {
        device
            .stream
            .launch_builder(&cast)
            .arg(&output)
            .arg(&cast_output)
            .arg(&257u32)
            .launch(flat(257))?;
    }
    ensure!(
        device.stream.clone_dtoh(&cast_output)?
            == values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        "device conditioning BF16 rounding differs"
    );
    ensure!(
        values
            .iter()
            .any(|&value| bf16::from_f32(value) != bf16::from_bits((value.to_bits() >> 16) as u16)),
        "truncation control did not discriminate"
    );
    let ordinal = cudarc::driver::result::device::get(0)?;
    let mut context = std::ptr::null_mut();
    let foreign = unsafe {
        cudarc::driver::sys::cuCtxCreate_v4(&mut context, std::ptr::null_mut(), 0, ordinal)
            .result()?;
        match cudarc::driver::CudaContext::from_raw_context(0, ordinal, context) {
            Ok(context) => context,
            Err(error) => {
                cudarc::driver::sys::cuCtxDestroy_v2(context).result()?;
                return Err(error.into());
            }
        }
    };
    let other_stream = foreign.new_stream()?;
    let other = other_stream.alloc_zeros::<f32>(257)?;
    ensure!(
        wait_input(&device, &other).is_err(),
        "foreign context accepted"
    );
    drop(other);
    drop(other_stream);
    drop(foreign);
    device.ctx.bind_to_thread()?;
    let (free, _) = device.ctx.mem_get_info()?;
    let needed = 257 * size_of::<f32>();
    ensure!(
        device.workspace_budget(free, needed - 1, needed).is_err(),
        "undersized live-set budget accepted"
    );
    ensure!(
        device
            .workspace_budget(free, needed - 1, needed - 4)
            .is_ok(),
        "one-word omission control did not discriminate"
    );
    println!(
        "P1_M4C_HANDOFF PASS values=257 cast_bit_exact=true producer_lifetime=true foreign_context_rejected=true stale_read_mutant=257 capacity_omission_bytes=4 timing=false"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA and frozen normalization captures"]
fn pixal3d_composed_normalization_gpu() -> Result<()> {
    let path = env_path("FF_PIXAL_M4C_FREEZE")?;
    let bytes = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "301489b487d9ee4fccbae6dac7764612fd70807535184731bef92d79c490a6cf",
        "M4c freeze integrity"
    );
    let freeze: Value = serde_json::from_slice(&bytes)?;
    let cells = freeze["normalization"]
        .as_array()
        .context("normalization cells")?;
    ensure!(cells.len() == 18, "normalization coverage");
    let device = Device::new(0)?;
    let mut defects = 0;
    for cell in cells {
        let name = cell["name"].as_str().context("normalization name")?;
        let path = Path::new(cell["path"].as_str().context("normalization path")?);
        let input = f32_tensor(path, "normalized")?;
        let mean = f32_tensor(path, "mean")?;
        let std = f32_tensor(path, "std")?;
        let reference = tensor(path, "physical")?;
        let mut output = device.stream.clone_htod(&input)?;
        channel_transform(&device, &mut output, [&mean, &std], false)?;
        let observed: Vec<_> = device
            .stream
            .clone_dtoh(&output)?
            .into_iter()
            .map(f64::from)
            .collect();
        let distance = metrics(&observed, &reference.values)?;
        for metric in ["max_abs", "rmse"] {
            let tau = cell["tau"][metric].as_f64().context("normalization tau")?;
            let pass = distance[metric] <= tau;
            defects += usize::from(!pass);
            println!(
                "P1_M4C_NORMALIZATION_CELL {} name={name} metric={metric} value={} tau={tau}",
                if pass { "PASS" } else { "DEFECT" },
                distance[metric]
            );
        }
        channel_transform(&device, &mut output, [&mean, &std], false)?;
        let doubled: Vec<_> = device
            .stream
            .clone_dtoh(&output)?
            .into_iter()
            .map(f64::from)
            .collect();
        for (control, values) in [
            (
                "omitted",
                input.into_iter().map(f64::from).collect::<Vec<_>>(),
            ),
            ("doubled", doubled),
        ] {
            let distance = metrics(&values, &reference.values)?;
            for metric in ["max_abs", "rmse"] {
                let ratio = distance[metric]
                    / cell["tau"][metric]
                        .as_f64()
                        .context("normalization control tau")?;
                ensure!(
                    ratio >= 10.0,
                    "normalization control weak {name} {control} {metric}"
                );
                println!(
                    "P1_M4C_NORMALIZATION_CONTROL PASS name={name} control={control} metric={metric} ratio={ratio}"
                );
            }
        }
    }
    println!(
        "P1_M4C_NORMALIZATION {} cells=18 metrics=36 controls=36 defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "normalization defects");
    Ok(())
}

#[test]
fn coordinate_noise_preserves_batch_axes_and_channel_layout() -> Result<()> {
    let shape = [2usize, 3, 5, 7, 11];
    let values: Vec<_> = (0..shape.iter().product()).map(|i| i as f32).collect();
    let field = CoordinateNoise {
        values: &values,
        shape,
        batch: 1,
    };
    let coordinates = [[2, 1, 6], [0, 4, 1], [1, 3, 0]];
    let rows = field.gather(&coordinates, 11, false)?;
    let columns = field.gather(&coordinates, 11, true)?;
    for (row, [x, y, z]) in coordinates.into_iter().enumerate() {
        for channel in 0..11 {
            let expected =
                ((((3 + x as usize) * 5 + y as usize) * 7 + z as usize) * 11 + channel) as f32;
            assert_eq!(rows[row * 11 + channel], expected);
            assert_eq!(columns[channel * coordinates.len() + row], expected);
        }
    }
    let wrong_batch = CoordinateNoise { batch: 0, ..field };
    assert_ne!(wrong_batch.gather(&coordinates, 11, false)?, rows);
    assert_ne!(columns, rows);
    assert!(field.gather(&[[3, 0, 0]], 11, false).is_err());
    assert!(field.gather(&[[-1, 0, 0]], 11, false).is_err());
    assert!(field.gather(&coordinates, 10, false).is_err());
    Ok(())
}

#[test]
#[ignore = "requires frozen coordinate noise fixtures; CPU only"]
fn pixal3d_coordinate_noise_fixture_cpu() -> Result<()> {
    let bytes = std::fs::read(env_path("FF_PIXAL_M4C_FREEZE")?)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "301489b487d9ee4fccbae6dac7764612fd70807535184731bef92d79c490a6cf",
        "M4c freeze integrity"
    );
    let freeze: Value = serde_json::from_slice(&bytes)?;
    let cells = freeze["noise"].as_array().context("noise cells")?;
    let root = env_path("FF_PIXAL_M4C_NOISE")?;
    let mut scored = 0;
    for stage in ["ss", "lr", "hr", "tex"] {
        let data = tensor(&root.join(format!("{stage}.safetensors")), "noise")?;
        let shape: [usize; 5] = data
            .shape
            .try_into()
            .map_err(|_| anyhow::anyhow!("noise field rank"))?;
        let values: Vec<_> = data.values.into_iter().map(|v| v as f32).collect();
        let field = CoordinateNoise {
            values: &values,
            shape,
            batch: 0,
        };
        for cell in cells.iter().filter(|cell| cell["stage"] == stage) {
            let source = Path::new(cell["source"].as_str().context("noise source")?);
            let dense = stage == "ss";
            let coordinates = if dense {
                (0..16usize.pow(3))
                    .map(|i| [(i / 256) as i32, ((i / 16) % 16) as i32, (i % 16) as i32])
                    .collect::<Vec<_>>()
            } else {
                coordinates(&tensor(source, "noise.coords")?)?
            };
            let expected = f32_tensor(source, if dense { "noise" } else { "noise.features" })?;
            let actual = field.gather(&coordinates, shape[4], dense)?;
            ensure!(
                actual == expected,
                "coordinate noise mismatch {} {stage}",
                cell["case"]
            );
            let swapped: Vec<_> = coordinates.iter().map(|p| [p[2], p[1], p[0]]).collect();
            let wrong = field.gather(&swapped, shape[4], dense)?;
            let flips = wrong.iter().zip(&expected).filter(|(a, b)| a != b).count();
            ensure!(
                flips > 0
                    && flips as u64
                        == cell["axis_swap_control_changed_components"]
                            .as_u64()
                            .context("noise control count")?,
                "noise axis control mismatch"
            );
            println!(
                "P1_M4C_NOISE_CELL PASS case={} stage={stage} values={} axis_control_flips={flips}",
                cell["case"],
                actual.len()
            );
            scored += 1;
        }
    }
    ensure!(scored == 24, "noise fixture coverage");
    println!("P1_M4C_NOISE PASS cells={scored} axis_controls={scored} exact=true CPU_only=true");
    Ok(())
}

#[test]
fn generation_rejects_extra_decoder_passes_and_forward_branches() -> Result<()> {
    let mut spec: FlowSpec = serde_json::from_value(serde_json::json!({
        "resolution":16,"in_channels":8,"out_channels":8,"model_channels":1536,
        "cond_channels":1024,"num_blocks":30,"num_heads":12,"mlp_ratio":5.333333333333333,
        "pe_mode":"rope","share_mod":true,"qk_rms_norm":true,"qk_rms_norm_cross":true,
        "image_attn_mode":"proj","dtype":"bfloat16","proj_in_channels":null
    }))?;
    for (channels, negative) in [(8, 10), (32, 9), (64, 0)] {
        spec.in_channels = channels;
        flow_call_contract(&spec, [12, negative])?;
        assert!(flow_call_contract(&spec, [11, negative]).is_err());
        assert!(flow_call_contract(&spec, [12, negative + 1]).is_err());
    }
    spec.out_channels = 32;
    ensure!(
        Work::<bf16>::planned_bytes(43632, &spec, 5)? == 3_010_590_144,
        "published texture work count"
    );
    ensure!(
        Work::<bf16>::planned_bytes(usize::MAX, &spec, 5).is_err(),
        "work count overflow accepted"
    );
    ensure!(
        F32Attention::required_bytes([12, 43632, 5, 128, 128])? == 847_872,
        "cross attention tile count"
    );
    decoder_pass_contract(1)?;
    assert!(decoder_pass_contract(0).is_err());
    assert!(decoder_pass_contract(2).is_err());
    Ok(())
}

fn generation_capture(
    path: &Path,
    coordinates: &[[i32; 3]],
    values: &[f32],
    shape: Vec<usize>,
) -> Result<()> {
    use safetensors::tensor::TensorView;
    ensure!(!path.exists(), "generation capture already exists");
    let points: Vec<u8> = coordinates
        .iter()
        .flat_map(|p| [0, p[0], p[1], p[2]])
        .flat_map(i32::to_le_bytes)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    safetensors::serialize_to_file(
        [
            (
                "coordinates",
                TensorView::new(Dtype::I32, vec![coordinates.len(), 4], &points)?,
            ),
            ("features", TensorView::new(Dtype::F32, shape, &bytes)?),
        ],
        None,
        path,
    )?;
    Ok(())
}

fn generation_measurement(
    case: &Value,
    stage: usize,
    coordinates: &[[i32; 3]],
    actual: &[f32],
    freeze: &Value,
) -> Result<()> {
    if stage == 0 {
        let source = Path::new(case["capture"].as_str().context("chain capture")?)
            .join("sparse_structure-0-outputs.safetensors");
        let expected = tensor(&source, "samples")?;
        let distance = metrics(
            &actual.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &expected.values,
        )?;
        println!(
            "P1_M4C_CHAIN_MEASUREMENT case={} stage=SS rows={} max_abs={} rmse={} gating=false",
            case["case"],
            coordinates.len(),
            distance["max_abs"],
            distance["rmse"]
        );
        return Ok(());
    }
    let name = format!(
        "{}-{}",
        case["case"].as_str().context("case name")?,
        ["SS", "LR", "HR", "texture"][stage]
    );
    let cell = freeze["normalization"]
        .as_array()
        .context("normalization cells")?
        .iter()
        .find(|c| c["name"] == name)
        .context("chain normalization capture")?;
    let path = Path::new(cell["path"].as_str().context("normalization path")?);
    let expected_coordinates = decoder_coordinates(path, "coords")?;
    let expected = f32_tensor(path, "physical")?;
    let sites = SparseSites::new(expected_coordinates)?;
    let mut common = 0;
    let mut maximum = 0.0f64;
    let mut squares = 0.0f64;
    for (row, p) in coordinates.iter().enumerate() {
        if let Some(index) = sites.row([0, p[0], p[1], p[2]]) {
            common += 1;
            for channel in 0..32 {
                let delta = f64::from(actual[row * 32 + channel])
                    - f64::from(expected[index as usize * 32 + channel]);
                maximum = maximum.max(delta.abs());
                squares += delta * delta;
            }
        }
    }
    let union = coordinates.len() + sites.coordinates().len() - common;
    println!(
        "P1_M4C_CHAIN_MEASUREMENT name={name} native_rows={} reference_rows={} common={common} jaccard_disagreement={} prior_decoder_full_grid_reference_pair_jaccard_disagreement=0.5926020937353391 common_max_abs={maximum} common_rmse={} gating=false",
        coordinates.len(),
        sites.coordinates().len(),
        1.0 - common as f64 / union as f64,
        if common > 0 {
            (squares / (common * 32) as f64).sqrt()
        } else {
            f64::NAN
        }
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA, published Pixal3D components, RGBA/cameras and coordinate noise"]
fn pixal3d_generated_request_gpu() -> Result<()> {
    let bytes = std::fs::read(env_path("FF_PIXAL_M4C_FREEZE")?)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes))
            == "301489b487d9ee4fccbae6dac7764612fd70807535184731bef92d79c490a6cf",
        "M4c freeze integrity"
    );
    let freeze: Value = serde_json::from_slice(&bytes)?;
    let name = std::env::var("FF_PIXAL_M4C_CASE").context("FF_PIXAL_M4C_CASE")?;
    let case = freeze["cases"]
        .as_array()
        .context("chain cases")?
        .iter()
        .find(|case| case["case"] == name)
        .context("unfrozen generation case")?;
    let (fixture, resolution) = name.rsplit_once('-').context("case resolution")?;
    let resolution: u32 = resolution.parse()?;
    let view_count = match fixture {
        "mv4" => 4,
        "mv1" => 1,
        "cottage" | "robot_crab" => 1,
        _ => anyhow::bail!("case fixture"),
    };
    let multiview = fixture.starts_with("mv");
    let inputs = env_path("FF_P1_M2_ROOT")?
        .join("gate-inputs.safetensors")
        .canonicalize()?;
    let input_file = std::fs::File::open(&inputs)?;
    let input_map = unsafe { memmap2::MmapOptions::new().map(&input_file)? };
    let archive = safetensors::SafeTensors::deserialize(&input_map)?;
    let root = env_path("FF_PIXAL_REFERENCE")?;
    let mut views = Vec::new();
    for index in 0..view_count {
        let prefix = if multiview {
            format!("mv-{index}")
        } else {
            fixture.to_owned()
        };
        let rgba = archive.tensor(&format!("{prefix}-rgba"))?;
        let camera = if multiview {
            let matrices = f32_tensor(&inputs, "mv-world-to-camera")?;
            crate::backend::ProjectionCamera {
                world_to_camera: matrices[index * 12..(index + 1) * 12].try_into()?,
                horizontal_fov: f32_tensor(&inputs, "mv-angle")?[index],
                mesh_scale: f32_tensor(&inputs, "mv-mesh-scale")?[0],
            }
        } else {
            let meta: Value = serde_json::from_slice(&std::fs::read(
                root.join("fixtures/pixal3d")
                    .join(format!("{fixture}-camera.json")),
            )?)?;
            crate::backend::ProjectionCamera {
                world_to_camera: f32_tensor(&inputs, &format!("{fixture}-world-to-camera"))?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("camera extent"))?,
                horizontal_fov: meta["camera_angle_x"].as_f64().context("camera FOV")? as f32,
                mesh_scale: meta["mesh_scale"].as_f64().context("camera scale")? as f32,
            }
        };
        views.push(PixalView {
            rgba: rgba.data(),
            dimensions: [rgba.shape()[0], rgba.shape()[1]],
            camera,
        });
    }
    let noise_root = env_path("FF_PIXAL_M4C_NOISE")?;
    let noise = ["ss", "lr", "hr", "tex"]
        .map(|stage| -> Result<_> {
            let data = tensor(&noise_root.join(format!("{stage}.safetensors")), "noise")?;
            let shape: [usize; 5] = data
                .shape
                .try_into()
                .map_err(|_| anyhow::anyhow!("noise rank"))?;
            Ok((
                data.values
                    .into_iter()
                    .map(|v| v as f32)
                    .collect::<Vec<_>>(),
                shape,
            ))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let model = env_path("FF_PIXAL_MODEL")?;
    let dino = env_path("FF_DINOV3_MODEL")?;
    let naf = env_path("FF_NAF_MODEL")?;
    let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
    let generation = PixalGeneration {
        model: &model,
        dino: &dino,
        naf: &naf,
        runtime: &runtime,
        views: &views,
        multiview,
        requested_resolution: resolution,
        noise: std::array::from_fn(|index| CoordinateNoise {
            values: &noise[index].0,
            shape: noise[index].1,
            batch: 0,
        }),
    };
    let output_dir = env_path("FF_PIXAL_M4C_OUTPUT")?;
    std::fs::create_dir(&output_dir)?;
    let device = Device::new(0)?;
    if fixture == "mv4" {
        for (label, changed) in [("A0", false), ("B", true), ("A1", false)] {
            let mut current: Vec<_> = views
                .iter()
                .map(|v| PixalView {
                    rgba: v.rgba,
                    dimensions: v.dimensions,
                    camera: v.camera,
                })
                .collect();
            if changed {
                let rgba = archive.tensor("robot_crab-rgba")?;
                current[1].rgba = rgba.data();
                current[1].dimensions = [rgba.shape()[0], rgba.shape()[1]];
            }
            let current_generation = PixalGeneration {
                views: &current,
                noise: std::array::from_fn(|i| CoordinateNoise {
                    values: generation.noise[i].values,
                    shape: generation.noise[i].shape,
                    batch: generation.noise[i].batch,
                }),
                ..generation
            };
            let destination = output_dir.join(label);
            std::fs::create_dir(&destination)?;
            generation_run(&current_generation, &device, case, &freeze, &destination)?;
        }
        for file in ["stage0", "stage1", "stage2", "stage3", "mesh", "material"] {
            let suffix = format!("{file}.safetensors");
            let equal = generation_equal(
                &output_dir.join("A0").join(&suffix),
                &output_dir.join("A1").join(&suffix),
            )?;
            ensure!(equal, "A->B->A changed {file}");
        }
        let different = !generation_equal(
            &output_dir.join("A0/stage0.safetensors"),
            &output_dir.join("B/stage0.safetensors"),
        )?;
        ensure!(different, "non-anchor media input did not reach generation");
        println!(
            "P1_M4C_ISOLATION PASS requests=3 exact_A_files=6 changed_non_anchor=1 different_B=true"
        );
    } else {
        generation_run(&generation, &device, case, &freeze, &output_dir)?;
    }
    Ok(())
}

fn generation_equal(left: &Path, right: &Path) -> Result<bool> {
    let a = std::fs::File::open(left)?;
    let b = std::fs::File::open(right)?;
    let a = unsafe { memmap2::MmapOptions::new().map(&a)? };
    let b = unsafe { memmap2::MmapOptions::new().map(&b)? };
    let a = safetensors::SafeTensors::deserialize(&a)?;
    let b = safetensors::SafeTensors::deserialize(&b)?;
    let names: BTreeSet<_> = a.names().into_iter().collect();
    if names != b.names().into_iter().collect() {
        return Ok(false);
    }
    for name in names {
        let x = a.tensor(name)?;
        let y = b.tensor(name)?;
        if x.dtype() != y.dtype() || x.shape() != y.shape() || x.data() != y.data() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn generation_run(
    generation: &PixalGeneration<'_>,
    device: &Device,
    case: &Value,
    freeze: &Value,
    output_dir: &Path,
) -> Result<()> {
    use safetensors::tensor::TensorView;
    let name = case["case"].as_str().context("case name")?;
    let mut seen = Vec::new();
    let mut grids = Vec::new();
    let retention_limit = if name == "cottage-1024" {
        0
    } else {
        usize::MAX
    };
    let result =
        generation.run_observed(device, retention_limit, |stage, grid, coordinates, data| {
            ensure!(stage == seen.len(), "missing or repeated flow stage");
            seen.push(stage);
            grids.push(grid);
            let actual = device.stream.clone_dtoh(data)?;
            ensure!(
                actual.iter().all(|v| v.is_finite()),
                "nonfinite stage output"
            );
            let shape = if stage == 0 {
                vec![1, 8, 16, 16, 16]
            } else {
                vec![coordinates.len(), 32]
            };
            generation_capture(
                &output_dir.join(format!("stage{stage}.safetensors")),
                coordinates,
                &actual,
                shape,
            )?;
            generation_measurement(case, stage, coordinates, &actual, freeze)?;
            println!(
                "P1_M4C_STAGE PASS case={name} stage={stage} grid={grid} rows={} values={}",
                coordinates.len(),
                actual.len()
            );
            Ok(())
        })?;
    ensure!(
        seen == [0, 1, 2, 3] && grids == [16, 32, result.resolution / 16, result.resolution / 16],
        "stage/grid integration mismatch"
    );
    ensure!(
        result.origin == [-0.5; 3] && result.voxel_size == 1.0 / result.resolution as f32,
        "output world grid mismatch"
    );
    ensure!(
        result.mesh.vertices.len().is_multiple_of(3)
            && result.mesh.vertices.iter().all(|x| x.is_finite())
            && result.mesh.faces.len().is_multiple_of(3)
            && result
                .mesh
                .faces
                .iter()
                .all(|&i| i >= 0 && (i as usize) < result.mesh.vertices.len() / 3),
        "invalid raw mesh"
    );
    ensure!(
        result.material.channels == 6
            && result.material.features.len() == result.material.coordinates.len() * 6
            && result.material.features.iter().all(|x| x.is_finite())
            && result.material.coordinates.iter().all(|p| p[0] == 0
                && p[1..]
                    .iter()
                    .all(|&v| v >= 0 && v < result.resolution as i32)),
        "invalid material voxel fields"
    );
    let coordinates: Vec<_> = result
        .material
        .coordinates
        .iter()
        .map(|p| [p[1], p[2], p[3]])
        .collect();
    generation_capture(
        &output_dir.join("material.safetensors"),
        &coordinates,
        &result.material.features,
        vec![coordinates.len(), 6],
    )?;
    let vertices: Vec<u8> = result
        .mesh
        .vertices
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let faces: Vec<u8> = result
        .mesh
        .faces
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let world: Vec<u8> = [
        result.origin[0],
        result.origin[1],
        result.origin[2],
        result.voxel_size,
    ]
    .into_iter()
    .flat_map(f32::to_le_bytes)
    .collect();
    safetensors::serialize_to_file(
        [
            ("world_grid", TensorView::new(Dtype::F32, vec![4], &world)?),
            (
                "vertices",
                TensorView::new(
                    Dtype::F32,
                    vec![result.mesh.vertices.len() / 3, 3],
                    &vertices,
                )?,
            ),
            (
                "faces",
                TensorView::new(Dtype::I32, vec![result.mesh.faces.len() / 3, 3], &faces)?,
            ),
        ],
        None,
        &output_dir.join("mesh.safetensors"),
    )?;
    println!(
        "P1_M4C_GENERATED_REQUEST PASS case={name} actual_resolution={} vertices={} faces={} material_rows={} generated_reference_comparisons=measurements timing=false",
        result.resolution,
        result.mesh.vertices.len() / 3,
        result.mesh.faces.len() / 3,
        result.material.coordinates.len()
    );
    Ok(())
}
