//! N1-M1 engine-cell gates. GPU gates are `#[ignore]`d and read every fixture
//! path from `FF_N1_*` environment variables under a coordinator-granted window.

#[cfg(feature = "cuda")]
mod gates {
    use crate::backend::cuda::engine::Engine;
    use crate::backend::gpu::{CudaSlice, CudaStream};
    use crate::backend::{Head, Input, Mask, Operator, Settings, SpecBudget, Step, TreeBuilder};
    use anyhow::{Context, Result, bail, ensure};
    use half::bf16;
    use safetensors::{Dtype, SafeTensors};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        collections::{BTreeMap, BTreeSet},
        path::{Path, PathBuf},
        sync::Arc,
    };

    const LAYERS: [usize; 5] = [0, 1, 25, 50, 51];
    const TAGS: [&str; 3] = ["mixer", "oproj", "h"];

    fn env_path(name: &str) -> Result<PathBuf> {
        std::env::var(name)
            .with_context(|| format!("{name} not set"))
            .map(PathBuf::from)
    }

    fn read_json(path: &Path) -> Result<Value> {
        serde_json::from_slice(&std::fs::read(path)?)
            .with_context(|| format!("frozen JSON {}", path.display()))
    }

    fn f32_field(value: &Value, name: &str, cell: &str) -> Result<f64> {
        value
            .get(name)
            .and_then(Value::as_f64)
            .with_context(|| format!("frozen {cell} missing {name}"))
    }

    fn cell_bound(cell: &Value, key: &str) -> Result<(f64, f64)> {
        let bound = cell
            .get("bound")
            .with_context(|| format!("frozen {key} missing bound"))?;
        Ok((
            f32_field(bound, "max_abs", key)?,
            f32_field(bound, "rmse", key)?,
        ))
    }

    fn cell_numel(cell: &Value, key: &str) -> Result<usize> {
        cell.get("numel")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .with_context(|| format!("frozen {key} missing numel"))
    }

    fn cell_arms(cell: &Value, key: &str) -> Result<Vec<String>> {
        cell.get("arms")
            .and_then(Value::as_array)
            .with_context(|| format!("frozen {key} missing arms"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("frozen {key} has non-string arm"))
            })
            .collect()
    }

    fn parse_cell(key: &str) -> Result<(String, Option<usize>, usize)> {
        let (prefix, pos) = key
            .rsplit_once("_p")
            .with_context(|| format!("cell {key} has no position"))?;
        let pos: usize = pos
            .parse()
            .with_context(|| format!("cell {key} has invalid position"))?;
        let (tag, layer) = match prefix.rsplit_once("_l") {
            Some((tag, layer)) => (
                tag.to_owned(),
                Some(
                    layer
                        .parse::<usize>()
                        .with_context(|| format!("cell {key} has invalid layer"))?,
                ),
            ),
            None => (prefix.to_owned(), None),
        };
        Ok((tag, layer, pos))
    }

    fn f32_taps(bytes: &[u8]) -> Result<BTreeMap<String, Vec<f32>>> {
        let view = SafeTensors::deserialize(bytes)?;
        let mut out = BTreeMap::new();
        for (name, tensor) in view.tensors() {
            ensure!(
                tensor.dtype() == Dtype::F32,
                "reference tap {name} is not F32"
            );
            let data = tensor.data();
            ensure!(data.len() % 4 == 0, "reference tap {name} is unaligned F32");
            out.insert(
                name,
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            );
        }
        Ok(out)
    }

    fn bf16_rows(bytes: &[u8], name: &str, rows: usize, width: usize) -> Result<Vec<bf16>> {
        ensure!(
            bytes.len() == rows * width * 2,
            "input {name} byte extent differs"
        );
        let values: Vec<bf16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
            .collect();
        ensure!(
            values.iter().all(|v| v.to_f32().is_finite()),
            "input {name} has non-finite values"
        );
        Ok(values)
    }

    fn metrics(candidate: &[f32], reference: &[f32]) -> Result<(f64, f64)> {
        ensure!(
            candidate.len() == reference.len() && !candidate.is_empty(),
            "tap extent differs"
        );
        let mut max_abs = 0.0f64;
        let mut squared = 0.0f64;
        for (&a, &b) in candidate.iter().zip(reference) {
            ensure!(a.is_finite() && b.is_finite(), "non-finite tap value");
            let d = (f64::from(a) - f64::from(b)).abs();
            max_abs = max_abs.max(d);
            squared += d * d;
        }
        Ok((max_abs, (squared / candidate.len() as f64).sqrt()))
    }

    fn arm_dirs(frozen: &Value) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        if let Some(arms) = frozen.get("arms").and_then(Value::as_object) {
            for (name, meta) in arms {
                let dir = meta
                    .get("dir")
                    .and_then(Value::as_str)
                    .with_context(|| format!("frozen arm {name} missing dir"))?;
                out.insert(name.clone(), dir.to_owned());
            }
        }
        if let Some(shape) = frozen.get("shape_arm")
            && let (Some(name), Some(dir)) = (
                shape.get("name").and_then(Value::as_str),
                shape.get("dir").and_then(Value::as_str),
            )
        {
            out.insert(name.to_owned(), dir.to_owned());
        }
        ensure!(!out.is_empty(), "frozen file has no reference arms");
        Ok(out)
    }

    fn load_arms(
        root: &Path,
        frozen: &Value,
    ) -> Result<BTreeMap<String, BTreeMap<String, Vec<f32>>>> {
        let mut out = BTreeMap::new();
        for (name, dir) in arm_dirs(frozen)? {
            let base = Path::new(&dir)
                .file_name()
                .with_context(|| format!("frozen arm {name} has no directory name"))?;
            let path = root.join(base).join("taps.safetensors");
            let bytes = std::fs::read(&path)
                .with_context(|| format!("reference arm {name} at {}", path.display()))?;
            out.insert(name, f32_taps(&bytes)?);
        }
        Ok(out)
    }

    fn tap_positions(cells: &Value, layer: Option<usize>, tag: &str) -> Result<BTreeSet<usize>> {
        let mut out = BTreeSet::new();
        for key in cells
            .as_object()
            .context("frozen cells is not an object")?
            .keys()
        {
            let (cell_tag, cell_layer, pos) = parse_cell(key)?;
            if cell_tag == tag && cell_layer == layer {
                out.insert(pos);
            }
        }
        Ok(out)
    }

    fn read_rows(
        s: &Arc<CudaStream>,
        buffer: &CudaSlice<bf16>,
        width: usize,
        positions: &BTreeSet<usize>,
        tag: &str,
        layer: usize,
        out: &mut BTreeMap<String, Vec<f32>>,
    ) -> Result<()> {
        for &pos in positions {
            let end = pos
                .checked_add(1)
                .and_then(|n| n.checked_mul(width))
                .context("tap position overflow")?;
            ensure!(end <= buffer.len(), "tap position outside workspace buffer");
            let row = s.clone_dtoh(&buffer.slice(pos * width..end))?;
            let values: Vec<f32> = row.into_iter().map(bf16::to_f32).collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "candidate tap {tag}_l{layer}_p{pos} has non-finite values"
            );
            out.insert(format!("{tag}_l{layer}_p{pos}"), values);
        }
        Ok(())
    }

    fn replay_layer(
        engine: &mut Engine,
        layer: usize,
        hidden: &[bf16],
        rows: usize,
        width: usize,
        positions: &BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<BTreeMap<String, Vec<f32>>> {
        let s = engine.device.stream.clone();
        let step = Step {
            rows,
            input: Input::Ids,
            head: Head::None,
            mask: Mask::Causal,
        };
        engine.position = 0;
        s.memcpy_htod(hidden, &mut engine.work.x.slice_mut(..rows * width))?;
        s.memset_zeros(&mut engine.work.out.slice_mut(..rows * width))?;
        {
            let catalog = engine
                .catalog
                .as_ref()
                .context("classic streamed catalog missing")?;
            let input_norm = catalog.classic_norm(layer, "input_layernorm")?;
            engine
                .work
                .add_norm(&engine.ops, &s, input_norm, rows, &engine.config)?;
        }
        s.memcpy_htod(&[0i32], &mut engine.work.position)?;
        s.memcpy_htod(&[rows as i32], &mut engine.work.length)?;
        if let Some(dense) = engine.catalog.as_mut().and_then(|c| c.dense.as_mut()) {
            dense.begin(layer, &s)?;
        }
        let mut taps = BTreeMap::new();
        for op in [
            Operator::Qkv,
            Operator::RopeKv,
            Operator::Attention,
            Operator::Output,
        ] {
            engine.run_operator(op, layer, step, &mut None)?;
            match op {
                Operator::Attention => read_rows(
                    &s,
                    &engine.work.attn,
                    width,
                    &positions["mixer"],
                    "mixer",
                    layer,
                    &mut taps,
                )?,
                Operator::Output => read_rows(
                    &s,
                    &engine.work.out,
                    width,
                    &positions["oproj"],
                    "oproj",
                    layer,
                    &mut taps,
                )?,
                _ => {}
            }
        }
        for op in [
            Operator::PostNorm,
            Operator::GateUp,
            Operator::Activate,
            Operator::Down,
            Operator::NextNorm,
        ] {
            engine.run_operator(op, layer, step, &mut None)?;
        }
        read_rows(
            &s,
            &engine.work.x,
            width,
            &positions["h"],
            "h",
            layer,
            &mut taps,
        )?;
        if let Some(dense) = engine.catalog.as_mut().and_then(|c| c.dense.as_mut()) {
            dense.end(layer, &s)?;
        }
        Ok(taps)
    }

    fn final_norm_and_logits(
        engine: &mut Engine,
        candidates: &mut BTreeMap<String, Vec<f32>>,
        positions: &BTreeSet<usize>,
        width: usize,
    ) -> Result<()> {
        let s = engine.device.stream.clone();
        let vocab = engine.config.vocab_size;
        let mut rows = BTreeMap::new();
        for &pos in positions {
            let end = (pos + 1) * width;
            let row = s.clone_dtoh(&engine.work.n.slice(pos * width..end))?;
            let values: Vec<f32> = row.iter().map(|&v| bf16::to_f32(v)).collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "final norm at p{pos} has non-finite values"
            );
            candidates.insert(format!("final_norm_p{pos}"), values);
            rows.insert(pos, row);
        }
        let step = Step {
            rows: 1,
            input: Input::Ids,
            head: Head::Last,
            mask: Mask::Causal,
        };
        for (pos, row) in rows {
            s.memcpy_htod(&row[..], &mut engine.work.n.slice_mut(..width))?;
            engine.run_operator(Operator::Head, 0, step, &mut None)?;
            let logits = s.clone_dtoh(&engine.work.logits.slice(..vocab))?;
            let values: Vec<f32> = logits.into_iter().map(bf16::to_f32).collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "logits at p{pos} have non-finite values"
            );
            candidates.insert(format!("logits_p{pos}"), values);
        }
        Ok(())
    }

    fn compare_cells(
        source: &str,
        cells: &Value,
        arms: &BTreeMap<String, BTreeMap<String, Vec<f32>>>,
        candidate: &BTreeMap<String, Vec<f32>>,
        gated: bool,
        outcomes: &mut Vec<Value>,
    ) -> Result<usize> {
        let mut defects = 0usize;
        for (key, cell) in cells
            .as_object()
            .context("frozen cells is not an object")?
            .iter()
        {
            let measured = if let Some(values) = candidate.get(key) {
                let numel = cell_numel(cell, key)?;
                ensure!(values.len() == numel, "candidate {key} extent differs");
                let bound_arms = cell_arms(cell, key)?;
                let mut worst = (0.0f64, 0.0f64);
                for arm in bound_arms {
                    let reference = arms
                        .get(&arm)
                        .and_then(|taps| taps.get(key))
                        .with_context(|| format!("{source} {key} missing reference arm {arm}"))?;
                    ensure!(reference.len() == numel, "reference {key} extent differs");
                    let pair = metrics(values, reference)?;
                    worst.0 = worst.0.max(pair.0);
                    worst.1 = worst.1.max(pair.1);
                }
                Some(worst)
            } else {
                ensure!(!gated, "gated {source} cell {key} has no candidate tap");
                None
            };
            let (bound_abs, bound_rmse) = cell_bound(cell, key)?;
            let over = measured.is_some_and(|(a, r)| a > bound_abs || r > bound_rmse);
            if gated && over {
                defects += 1;
            }
            let pairs: Vec<Value> = cell
                .get("pairs")
                .and_then(Value::as_object)
                .map(|p| {
                    p.iter()
                        .map(|(name, v)| json!({"pair": name, "values": v}))
                        .collect()
                })
                .unwrap_or_default();
            outcomes.push(json!({
                "source": source,
                "cell": key,
                "kind": if gated { "gate" } else { "measurement" },
                "max_abs": measured.map(|m| m.0),
                "rmse": measured.map(|m| m.1),
                "bound": {"max_abs": bound_abs, "rmse": bound_rmse},
                "reference_pairs": pairs,
                "defect": gated && over,
            }));
        }
        Ok(defects)
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_N1_MODEL, FF_N1_INPUTS, FF_N1_LAYER_FREEZE, FF_N1_WHOLE_FREEZE, FF_N1_REFERENCE_ARMS, FF_N1_EVIDENCE, FLYINGFISH_RUNTIME_DIR"]
    fn n1_streamed_classic_layer_cells_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not granted"
        );
        let model = env_path("FF_N1_MODEL")?;
        let inputs_path = env_path("FF_N1_INPUTS")?;
        let arms_root = env_path("FF_N1_REFERENCE_ARMS")?;
        let evidence = env_path("FF_N1_EVIDENCE")?;
        let runtime = env_path("FLYINGFISH_RUNTIME_DIR")?;
        let layer_frozen = read_json(&env_path("FF_N1_LAYER_FREEZE")?)?;
        let whole_frozen = read_json(&env_path("FF_N1_WHOLE_FREEZE")?)?;

        let rows = layer_frozen
            .get("common_L")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .context("r3 frozen common_L missing")?;
        let layer_cells = layer_frozen
            .get("cells")
            .context("r3 frozen cells missing")?;
        let whole_cells = whole_frozen
            .get("cells")
            .context("whole frozen cells missing")?;
        let layer_arms = load_arms(&arms_root, &layer_frozen)?;
        let whole_arms = load_arms(&arms_root, &whole_frozen)?;

        let inputs_bytes = std::fs::read(&inputs_path)?;
        let want_sha = layer_frozen
            .get("common_inputs_sha256")
            .and_then(Value::as_str)
            .context("r3 frozen common_inputs_sha256 missing")?;
        ensure!(
            format!("{:x}", Sha256::digest(&inputs_bytes)) == want_sha,
            "recorded inputs sha256 differs from the frozen fixture"
        );
        let inputs = SafeTensors::deserialize(&inputs_bytes)?;

        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            runtime_dir: Some(runtime),
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            device_budget_gib: None,
        };
        let mut engine = Engine::load_kind(&model, 0, rows + 8, Some(rows), settings, true)?;
        ensure!(engine.qwen.is_none(), "N1 gate requires the classic engine");
        ensure!(
            engine.catalog.as_ref().is_some_and(|c| c.dense.is_some()),
            "N1 gate requires the streamed classic catalog"
        );
        let width = engine.config.hidden_size;
        ensure!(
            rows <= engine.workspace_rows,
            "workspace cannot hold L rows"
        );
        engine.forward_rows.insert(1);
        engine.forward_rows.insert(rows);

        let mut positions = BTreeMap::new();
        for tag in TAGS {
            let mut set = BTreeSet::new();
            for layer in LAYERS {
                set.extend(tap_positions(layer_cells, Some(layer), tag)?);
            }
            for layer in [0usize, 1] {
                set.extend(tap_positions(whole_cells, Some(layer), tag)?);
            }
            ensure!(!set.is_empty(), "no tap positions for {tag}");
            ensure!(
                set.iter().copied().max().is_some_and(|p| p < rows),
                "tap position outside L"
            );
            positions.insert(tag.to_owned(), set);
        }

        let mut candidate = BTreeMap::new();
        let mut inputs_by_layer = BTreeMap::new();
        for layer in LAYERS {
            let name = format!("in_l{layer}");
            let tensor = inputs.tensor(&name)?;
            ensure!(tensor.dtype() == Dtype::BF16, "input {name} is not BF16");
            ensure!(
                tensor.shape() == [rows, width],
                "input {name} shape differs from [L, hidden]"
            );
            let hidden = bf16_rows(tensor.data(), &name, rows, width)?;
            inputs_by_layer.insert(layer, hidden);
            candidate.extend(replay_layer(
                &mut engine,
                layer,
                inputs_by_layer[&layer].as_slice(),
                rows,
                width,
                &positions,
            )?);
        }
        let deep_positions = tap_positions(whole_cells, None, "final_norm")?;
        final_norm_and_logits(&mut engine, &mut candidate, &deep_positions, width)?;

        let mut outcomes = Vec::new();
        let mut defects = compare_cells(
            "r3",
            layer_cells,
            &layer_arms,
            &candidate,
            true,
            &mut outcomes,
        )?;
        let early: serde_json::Map<String, Value> = whole_cells
            .as_object()
            .context("whole cells is not an object")?
            .iter()
            .filter(|(key, _)| {
                parse_cell(key).is_ok_and(|(_, layer, _)| layer.is_some_and(|l| l < 2))
            })
            .map(|(key, cell)| (key.clone(), cell.clone()))
            .collect();
        defects += compare_cells(
            "whole_early",
            &Value::Object(early),
            &whole_arms,
            &candidate,
            true,
            &mut outcomes,
        )?;
        let deep: serde_json::Map<String, Value> = whole_cells
            .as_object()
            .context("whole cells is not an object")?
            .iter()
            .filter(|(key, _)| {
                parse_cell(key).is_ok_and(|(tag, layer, _)| {
                    (layer.is_none() || layer.is_some_and(|l| l >= 2))
                        && matches!(
                            tag.as_str(),
                            "final_norm" | "logits" | "h" | "mixer" | "oproj"
                        )
                })
            })
            .map(|(key, cell)| (key.clone(), cell.clone()))
            .collect();
        compare_cells(
            "whole_deep_measurement",
            &Value::Object(deep),
            &whole_arms,
            &candidate,
            false,
            &mut outcomes,
        )?;

        let control_min_ratio = layer_frozen
            .get("control_thresholds")
            .and_then(|c| c.get("control_min_ratio"))
            .and_then(Value::as_f64)
            .context("r3 control_min_ratio missing")?;
        ensure!(
            layer_frozen
                .get("control_thresholds")
                .and_then(|c| c.get("control_requires_both_metrics"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "r3 controls require both metrics"
        );
        let wrong_input = inputs_by_layer[&51].as_slice();
        let control_taps = replay_layer(&mut engine, 0, wrong_input, rows, width, &positions)?;
        let mut control_worst = (0.0f64, 0.0f64);
        let mut bound_worst = (0.0f64, 0.0f64);
        let mut control_cells = 0usize;
        for (key, cell) in layer_cells
            .as_object()
            .context("r3 cells is not an object")?
            .iter()
        {
            let (_, layer, _) = parse_cell(key)?;
            if layer != Some(0) {
                continue;
            }
            control_cells += 1;
            let values = control_taps
                .get(key)
                .with_context(|| format!("control tap {key} missing"))?;
            let (bound_abs, bound_rmse) = cell_bound(cell, key)?;
            bound_worst.0 = bound_worst.0.max(bound_abs);
            bound_worst.1 = bound_worst.1.max(bound_rmse);
            for arm in cell_arms(cell, key)? {
                let reference = layer_arms
                    .get(&arm)
                    .and_then(|taps| taps.get(key))
                    .with_context(|| format!("control {key} missing reference arm {arm}"))?;
                let pair = metrics(values, reference)?;
                control_worst.0 = control_worst.0.max(pair.0);
                control_worst.1 = control_worst.1.max(pair.1);
            }
        }
        ensure!(control_cells > 0, "control produced no layer-0 cells");
        let control_ratio = (
            control_worst.0 / bound_worst.0,
            control_worst.1 / bound_worst.1,
        );
        let control_ok =
            control_ratio.0 >= control_min_ratio && control_ratio.1 >= control_min_ratio;

        std::fs::create_dir_all(&evidence)?;
        let report = json!({
            "gate": "N1-M1-LAYER-CELLS",
            "model": model.display().to_string(),
            "L": rows,
            "hidden": width,
            "layers": LAYERS,
            "gated_cells": outcomes
                .iter()
                .filter(|o| o.get("kind").and_then(Value::as_str) == Some("gate"))
                .count(),
            "defects": defects,
            "control": {
                "ratio_max_abs": control_ratio.0,
                "ratio_rmse": control_ratio.1,
                "required": control_min_ratio,
                "passed": control_ok,
            },
            "cells": outcomes,
        });
        std::fs::write(
            evidence.join("n1_layer_cells_report.json"),
            serde_json::to_string_pretty(&report)? + "\n",
        )?;
        println!(
            "N1_LAYER_CELLS cells={} defects={} control_ratio=({:.3},{:.3}) required={} evidence={}",
            report["gated_cells"].as_u64().unwrap_or(0),
            defects,
            control_ratio.0,
            control_ratio.1,
            control_min_ratio,
            evidence.display()
        );
        if defects > 0 || !control_ok {
            bail!("N1_LAYER_CELLS defects={defects} control_passed={control_ok}");
        }
        println!(
            "PASS N1_LAYER_CELLS cells={} defects=0 control_ratio=({:.3},{:.3})",
            report["gated_cells"].as_u64().unwrap_or(0),
            control_ratio.0,
            control_ratio.1
        );
        Ok(())
    }
}
