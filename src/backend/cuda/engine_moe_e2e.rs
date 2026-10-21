//! Published-checkpoint text, media and teacher-forced decode acceptance gate.
use super::*;
use crate::vision_input::{ImageGrid, MultimodalInput};
use safetensors::{Dtype, SafeTensors};

#[derive(Debug)]
struct Bounds {
    layers: std::collections::BTreeMap<usize, [f64; 4]>,
    norm: [f64; 4],
    logits: [f64; 4],
    decode: [f64; 4],
    media: [f64; 2],
    cross_media: [f64; 2],
}

fn ceil_bf16(value: f64) -> Result<f64> {
    ensure!(
        value.is_finite() && value >= 0.0,
        "invalid noise value={value}"
    );
    let rounded = bf16::from_f32(value as f32);
    let bits = rounded.to_bits() + u16::from(f64::from(rounded.to_f32()) < value);
    let result = f64::from(bf16::from_bits(bits).to_f32());
    ensure!(result.is_finite(), "noise exceeds BF16 range value={value}");
    Ok(result)
}

fn measured_bounds(cells: &[&serde_json::Value]) -> Result<[f64; 4]> {
    ensure!(!cells.is_empty(), "reference noise has no cells");
    let mut bound = [0.0f64; 4];
    for cell in cells {
        for (index, key) in ["max_abs", "p99_abs", "mean_abs", "cosine_sim"]
            .iter()
            .enumerate()
        {
            let value = cell[key]
                .as_f64()
                .with_context(|| format!("missing noise {key}"))?;
            ensure!(value.is_finite(), "nonfinite noise {key}={value}");
            let error = if index == 3 {
                value.clamp(-1.0, 1.0).acos()
            } else {
                value
            };
            bound[index] = bound[index].max(error);
        }
    }
    for value in &mut bound[..3] {
        // Two-arm error budget: ceil_bf16(delta_ref + delta_ref).
        *value = ceil_bf16(*value + *value)?;
    }
    // Angular triangle bound: cos(min(2 * max(acos(reference cosine)), pi)).
    bound[3] = (2.0 * bound[3]).min(std::f64::consts::PI).cos();
    Ok(bound)
}

fn media_boundary(measured: &serde_json::Value) -> Result<[f64; 2]> {
    let separation = measured["max_abs"]
        .as_f64()
        .context("media max_abs missing")?;
    let cosine = measured["cosine_sim"]
        .as_f64()
        .context("media cosine missing")?;
    ensure!(
        separation.is_finite() && separation > 0.0 && cosine.is_finite() && cosine < 1.0,
        "invalid measured media separation={separation} cosine={cosine}"
    );
    // Midpoint between identical media and the measured reference contrast.
    Ok([separation / 2.0, 1.0 - (1.0 - cosine) / 2.0])
}

impl Bounds {
    fn read(fixtures: &Path, case: &str, format: crate::model_spec::ExpertFormat) -> Result<Self> {
        let directory = fixtures.parent().context("reference parent missing")?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("fixtures_manifest.json"))?)?;
        let video = case.starts_with("video_");
        let controls = &manifest[if video { "video" } else { "image" }]["media_reach_controls"];
        let same_grid = if video {
            "same_grid_1280x720_football_vs_boat_pan_logits_diff"
        } else {
            "same_grid_400x267_hockey_vs_boat_logits_diff"
        };
        let cross_media = match (format, video) {
            (_, true) => "football_16x_vs_sample_demo_1_16x_logits_diff",
            (crate::model_spec::ExpertFormat::Bf16, false) => "hockey_16x_vs_boat_16x_logits_diff",
            (crate::model_spec::ExpertFormat::Fp8W8a16, false) => same_grid,
        };
        let media = media_boundary(&controls[same_grid])?;
        let cross_media = media_boundary(&controls[cross_media])?;
        if format == crate::model_spec::ExpertFormat::Fp8W8a16 {
            return Self::read_fp8(fixtures, case, media, cross_media);
        }
        let noise: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("noise_floor_r2.json"))?)?;
        let floor = &noise["per_case_reference_floor"]["cases"][case];
        let comparisons = floor["comparisons"]
            .as_object()
            .with_context(|| format!("missing per-case reference noise {case}"))?;
        ensure!(
            comparisons.len() == 4,
            "reference noise {case} needs four comparisons"
        );
        let cross_gpu = &noise["official_cross_gpu_streamed"];
        ensure!(
            cross_gpu["sm89_primary_bitwise"].as_bool() == Some(true),
            "streamed official sm89 reference did not reproduce the frozen primary"
        );
        let cross_gpu = &cross_gpu["cases"][case];
        ensure!(
            cross_gpu.is_object(),
            "missing official cross-GPU reference noise {case}"
        );
        let batch_shape = &noise["official_expert_batch_shape"];
        ensure!(
            batch_shape["validated"].as_bool() == Some(true),
            "official expert batch-shape arms did not pass validation"
        );
        let batch_shape = ["sm89", "sm86"].map(|arch| &batch_shape["cases"][case][arch]);
        ensure!(
            batch_shape.iter().all(|cell| cell.is_object()),
            "missing official expert batch-shape noise {case}"
        );
        let vision = (case.starts_with("image_") || case.starts_with("video_"))
            .then(|| &noise["official_vision_implementation_propagation"]["cases"][case]);
        if let Some(cell) = vision {
            ensure!(
                cell.is_object(),
                "missing official vision propagation noise {case}"
            );
        }
        let mut layers = std::collections::BTreeMap::new();
        for layer in [0, 3, 6, 19, 39] {
            let key = format!("layer_{layer:02}_hidden_last8");
            let mut cells: Vec<_> = comparisons
                .values()
                .map(|cell| &cell["layers"][&key])
                .collect();
            cells.push(&cross_gpu["layers"][&key]);
            cells.extend(batch_shape.iter().map(|cell| &cell["layers"][&key]));
            if layer == 0 {
                cells.push(&floor["fp32_layer0"]);
            }
            if let Some(cell) = vision {
                cells.push(&cell["layers"][&key]);
            }
            layers.insert(layer, measured_bounds(&cells)?);
        }
        let cells = |key: &str| {
            let mut cells = comparisons
                .values()
                .map(|cell| &cell[key])
                .collect::<Vec<_>>();
            cells.push(&cross_gpu[key]);
            cells.extend(batch_shape.iter().map(|cell| &cell[key]));
            if let Some(cell) = vision {
                cells.push(&cell[key]);
            }
            cells
        };
        Ok(Self {
            layers,
            norm: measured_bounds(&cells("norm"))?,
            logits: measured_bounds(&cells("logits"))?,
            decode: measured_bounds(&cells("teacher_top2"))?,
            media,
            cross_media,
        })
    }

    fn read_fp8(
        fixtures: &Path,
        case: &str,
        media: [f64; 2],
        cross_media: [f64; 2],
    ) -> Result<Self> {
        let directory = fixtures.parent().context("FP8 reference parent missing")?;
        let noise: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("fp8-floor.json"))?)?;
        ensure!(
            noise["schema"] == 1
                && noise["format"] == "fp8_w8a16"
                && noise["state"] == "frozen_before_candidate_measurement"
                && noise["declaration"]["candidate_results_seen"] == false,
            "FP8 reference floor is not valid and frozen"
        );
        let components = noise["cases"][case]["components"]
            .as_object()
            .with_context(|| format!("FP8 floor case missing: {case}"))?;
        ensure!(components.len() == 6, "FP8 floor requires six components");
        for name in [
            "expert_implementation_order",
            "layer0_mixed_precision",
            "official_vision_implementation_propagation",
            "same_input_cross_gpu",
            "teacher_forced_per_step",
            "official_fp8_fake_quant_cross_gpu",
        ] {
            ensure!(
                components.contains_key(name),
                "FP8 floor component missing: {case}/{name}"
            );
        }
        let mut arms = vec![
            "expert_implementation_order",
            "layer0_mixed_precision",
            "same_input_cross_gpu",
        ];
        let vision = &components["official_vision_implementation_propagation"];
        let visual = case.starts_with("image_") || case.starts_with("video_");
        ensure!(
            vision["applicable"] == visual,
            "FP8 vision floor applicability differs from case={case}"
        );
        if visual {
            arms.push("official_vision_implementation_propagation");
        }
        for arm in &arms {
            ensure!(
                components[*arm]["applicable"] == true,
                "missing FP8 arm {case}/{arm}"
            );
        }
        let cells = |key: &str| {
            arms.iter()
                .map(|arm| &components[*arm]["tensors"][key])
                .collect::<Vec<_>>()
        };
        let mut layers = std::collections::BTreeMap::new();
        for layer in [0, 3, 6, 19, 39] {
            let key = format!("layer_{layer:02}_hidden_last8");
            let mut values = cells(&key);
            if layer == 0 {
                values.push(
                    &components["layer0_mixed_precision"]["tensors"]["layer0_fp32_raw_last8"],
                );
            }
            layers.insert(layer, measured_bounds(&values)?);
        }
        let teacher = &components["teacher_forced_per_step"];
        ensure!(
            teacher["steps"] == 32,
            "FP8 teacher floor requires 32 steps"
        );
        for arm in &arms {
            ensure!(
                teacher["comparisons"][*arm]
                    .as_array()
                    .is_some_and(|steps| steps.len() == 32),
                "FP8 teacher floor extent differs: {case}/{arm}"
            );
        }
        for arm in &arms {
            for step in 0..32 {
                measured_bounds(&[&teacher["comparisons"][*arm][step]["top2"]])?;
            }
        }
        let quant = &components["official_fp8_fake_quant_cross_gpu"];
        ensure!(
            quant["form"] == "P2_official_loader_token_fake_quant",
            "FP8 fake-quant form missing"
        );
        for layer in [3, 39] {
            for role in ["attn", "o"] {
                for kind in ["scale", "codes", "fake"] {
                    let key = format!("fixed_layer_{layer:02}_{role}_{kind}");
                    measured_bounds(&[&quant["cells"][key]])?;
                }
            }
        }
        Ok(Self {
            layers,
            norm: measured_bounds(&cells("final_norm_last"))?,
            logits: measured_bounds(&cells("prefill_last_logits_bf16"))?,
            decode: measured_bounds(&cells("teacher_top2"))?,
            media,
            cross_media,
        })
    }
}

fn floats(source: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = source.tensor(name)?;
    let values = match tensor.dtype() {
        Dtype::BF16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
            .collect::<Vec<_>>(),
        Dtype::F32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        dtype => anyhow::bail!("reference {name} has floating dtype {dtype:?}"),
    };
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite reference {name}"
    );
    Ok(values)
}

fn integers(source: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
    let tensor = source.tensor(name)?;
    match tensor.dtype() {
        Dtype::I64 => tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i64::from_le_bytes(*v))?))
            .collect(),
        Dtype::I32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i32::from_le_bytes(*v))?))
            .collect(),
        dtype => anyhow::bail!("reference {name} has integer dtype {dtype:?}"),
    }
}

fn metrics(a: &[f32], b: &[f32]) -> Result<[f64; 4]> {
    ensure!(
        !a.is_empty() && a.len() == b.len(),
        "comparison lengths {}/{}",
        a.len(),
        b.len()
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "nonfinite comparison"
    );
    if a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()) {
        return Ok([0.0, 0.0, 0.0, 1.0]);
    }
    let mut errors = Vec::with_capacity(a.len());
    let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (f64::from(a), f64::from(b));
        errors.push((a - b).abs());
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    errors.sort_unstable_by(f64::total_cmp);
    Ok([
        errors[errors.len() - 1],
        errors[(errors.len() * 99).div_ceil(100).saturating_sub(1)],
        errors.iter().sum::<f64>() / errors.len() as f64,
        if aa == 0.0 && bb == 0.0 {
            1.0
        } else {
            dot / (aa * bb).sqrt()
        },
    ])
}

fn score(name: &str, a: &[f32], b: &[f32], bound: [f64; 4]) -> Result<usize> {
    let m = metrics(a, b)?;
    ensure!(m.iter().all(|v| v.is_finite()), "nonfinite metrics {name}");
    let pass = m[0] <= bound[0]
        && m[1] <= bound[1]
        && m[2] <= bound[2]
        && m[3] >= bound[3]
        && (bound[0] != 0.0 || a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    println!(
        "{} minicpm_e2e {name} max={} p99={} mean={} cosine={} bound={bound:?}",
        if pass { "PASS" } else { "DEFECT" },
        m[0],
        m[1],
        m[2],
        m[3]
    );
    Ok(usize::from(!pass))
}

fn tail(trace: &Trace, name: &str, words: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for part in 0..trace.prefill_parts()? {
        values.extend(trace.values(&format!("prefill.{part}.{name}"))?);
    }
    ensure!(
        values.len() >= words,
        "trace {name} has {}/{} tail words",
        values.len(),
        words
    );
    Ok(values.split_off(values.len() - words))
}

fn score_chain(
    name: &str,
    got: &[f32],
    expected: &[f32],
    rounds: u32,
    reductions: &[usize],
) -> Result<usize> {
    let values = metrics(got, expected)?;
    let magnitude = got
        .iter()
        .chain(expected)
        .map(|v| f64::from(v.abs()))
        .fold(0.0, f64::max);
    let relative = super::super::super::bf16_chain_bound(rounds, reductions);
    let pass = values[0] <= relative * magnitude;
    println!(
        "{} minicpm_fp8_layer0 {name} max={} magnitude={magnitude} relative_bound={relative}",
        if pass { "PASS" } else { "DEFECT" },
        values[0]
    );
    Ok(usize::from(!pass))
}

fn native_layer0(engine: &mut Engine, source: &SafeTensors<'_>) -> Result<usize> {
    let h = engine.model_spec.hidden as usize;
    let moe = engine
        .model_spec
        .moe
        .context("FP8 layer0 MoE configuration missing")?;
    let (k, experts, intermediate) = (
        moe.experts_per_tok as usize,
        moe.experts as usize,
        moe.expert_intermediate as usize,
    );
    let normalized = source.tensor("layer_00_moe_in_norm")?;
    ensure!(
        normalized.shape().len() == 2,
        "FP8 layer0 normalized input is not a matrix"
    );
    let rows = normalized.shape()[0];
    for (name, shape) in [
        ("layer_00_moe_in_norm", vec![rows, h]),
        ("layer_00_router_logits", vec![rows, experts]),
        ("layer_00_router_topk_ids", vec![rows, k]),
        ("layer_00_router_topk_weights", vec![rows, k]),
        ("layer_00_routed_per_choice_out", vec![rows, k, h]),
    ] {
        ensure!(
            source.tensor(name)?.shape() == shape,
            "FP8 layer0 reference shape: {name}"
        );
    }
    let input = floats(source, "layer_00_moe_in_norm")?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let reference_ids = integers(source, "layer_00_router_topk_ids")?;
    let reference_weights = floats(source, "layer_00_router_topk_weights")?;
    let reference_logits = floats(source, "layer_00_router_logits")?;
    let expected = floats(source, "layer_00_routed_per_choice_out")?;
    ensure!(
        reference_ids.iter().all(|&id| (id as usize) < experts),
        "FP8 layer0 reference expert outside model"
    );
    engine.reset()?;
    let ffn = engine.moe_ffn_binding(0)?;
    let BoundFfn::Moe { router, .. } = ffn else {
        anyhow::bail!("FP8 layer0 FFN is dense");
    };
    let mut decode = engine.moe.take().context("FP8 layer0 decoder missing")?;
    let mut slots = engine
        .moe_slots
        .take()
        .context("FP8 layer0 residency missing")?;
    let result = (|| -> Result<usize> {
        let stream = &engine.device.stream;
        let batch = decode.max_rows.min(8);
        ensure!(batch > 0 && rows > 0, "FP8 layer0 input is empty");
        let logits = stream.alloc_zeros::<bf16>(batch * experts)?;
        slots.begin_request(&vec![vec![1.0; experts]; engine.model_spec.layers.len()])?;
        let mut defects = 0;
        for base in (0..rows).step_by(batch) {
            let count = batch.min(rows - base);
            let choices = base * k..(base + count) * k;
            stream.memcpy_htod(
                &input[base * h..(base + count) * h],
                &mut engine.work.n.slice_mut(..count * h),
            )?;
            let x = qwen::matrix(&engine.work.n, stream, 0, count, h)?;
            decode.route.route_prefill(
                stream,
                x,
                router,
                &mut TensorMut(view(
                    &decode.ids,
                    stream,
                    0,
                    [count as u32, k as u32],
                    [k as u64, 1],
                )?),
                &mut TensorMut(view(
                    &decode.weights,
                    stream,
                    0,
                    [count as u32, k as u32],
                    [k as u64, 1],
                )?),
                &mut TensorMut(qwen::matrix(&logits, stream, 0, count, experts)?),
            )?;
            defects += score_chain(
                "router_logits",
                &stream
                    .clone_dtoh(&logits.slice(..count * experts))?
                    .into_iter()
                    .map(bf16::to_f32)
                    .collect::<Vec<_>>(),
                &reference_logits[base * experts..(base + count) * experts],
                1,
                &[h],
            )?;
            decode.route(&engine.device, ffn, x)?;
            let (actual_ids, actual_weights) = decode.routed()?;
            let mut routed = Vec::new();
            let mut reference = Vec::new();
            let mut selection_errors = 0;
            for row in 0..count {
                let ids = &reference_ids[(base + row) * k..(base + row + 1) * k];
                let row_logits =
                    &reference_logits[(base + row) * experts..(base + row + 1) * experts];
                let cutoff = ids
                    .iter()
                    .map(|&id| row_logits[id as usize])
                    .fold(f32::INFINITY, f32::min);
                let magnitude = row_logits
                    .iter()
                    .map(|v| f64::from(v.abs()))
                    .fold(0.0, f64::max);
                let limit = super::super::super::bf16_chain_bound(1, &[h]) * magnitude;
                for choice in 0..k {
                    let id = actual_ids[row * k + choice];
                    ensure!(
                        id >= 0 && (id as usize) < experts,
                        "FP8 layer0 route expert outside model"
                    );
                    selection_errors +=
                        usize::from(f64::from(row_logits[id as usize]) + limit < f64::from(cutoff));
                    if let Some(index) = ids.iter().position(|&expected| expected == id as u32) {
                        routed.push(actual_weights[row * k + choice].to_f32());
                        reference.push(reference_weights[(base + row) * k + index]);
                    }
                }
            }
            defects += usize::from(selection_errors != 0);
            println!(
                "{} minicpm_fp8_layer0 router_selection base={base} rows={count} outside_margin={selection_errors} matched_weights={}",
                if selection_errors == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                },
                routed.len()
            );
            if routed.is_empty() {
                defects += 1;
                println!("DEFECT minicpm_fp8_layer0 router_weights overlap=0");
            } else {
                defects += score_chain("router_weights", &routed, &reference, 2, &[h, experts])?;
            }
            let ids = reference_ids[choices.clone()]
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>();
            let weights = reference_weights[choices.clone()]
                .iter()
                .map(|&v| bf16::from_f32(v))
                .collect::<Vec<_>>();
            stream.memcpy_htod(&ids, &mut decode.ids.slice_mut(..count * k))?;
            stream.memcpy_htod(&weights, &mut decode.weights.slice_mut(..count * k))?;
            decode.host_ids.as_mut_slice()?[..count * k].copy_from_slice(&ids);
            decode.host_weights.as_mut_slice()?[..count * k].copy_from_slice(&weights);
            slots.begin_token()?;
            slots.finish_routed(
                &engine.device,
                &mut decode,
                0,
                ffn,
                (
                    x,
                    &mut TensorMut(qwen::matrix(&engine.work.out, stream, 0, count, h)?),
                ),
            )?;
            let contributions = stream.clone_dtoh(&decode.contributions.slice(..count * k * h))?;
            // Gate/up, SiLU, product, down and routing apply six BF16 roundings.
            defects += score_chain(
                "native_contributions",
                &contributions
                    .iter()
                    .copied()
                    .map(bf16::to_f32)
                    .collect::<Vec<_>>(),
                &expected[base * k * h..(base + count) * k * h],
                6,
                &[h, h, intermediate],
            )?;
            let shared = stream.clone_dtoh(&decode.shared.slice(..count * h))?;
            let scale = stream.clone_dtoh(&decode.scale.slice(..count))?;
            let got = stream.clone_dtoh(&engine.work.out.slice(..count * h))?;
            let mut mismatches = 0;
            for row in 0..count {
                for column in 0..h {
                    let routed = bf16::from_f32(
                        (0..k)
                            .map(|choice| contributions[(row * k + choice) * h + column].to_f32())
                            .sum(),
                    );
                    let shared =
                        bf16::from_f32(shared[row * h + column].to_f32() * scale[row].to_f32());
                    let value = bf16::from_f32(routed.to_f32() + shared.to_f32());
                    mismatches += usize::from(value.to_bits() != got[row * h + column].to_bits());
                }
            }
            defects += usize::from(mismatches != 0);
            println!(
                "{} minicpm_fp8_layer0 combine_boundaries base={base} rows={count} bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
            slots.finish_token()?;
        }
        slots.end_request()?;
        Ok(defects)
    })();
    engine.moe = Some(decode);
    engine.moe_slots = Some(slots);
    result
}

struct Input {
    ids: Vec<u32>,
    modalities: Vec<u8>,
    pixels: Vec<f32>,
    grids: Vec<ImageGrid>,
}

impl Input {
    fn read(source: &SafeTensors<'_>, case: &str) -> Result<Self> {
        let ids = integers(source, "input_ids")?;
        ensure!(
            source.tensor("input_ids")?.shape() == [1, ids.len()],
            "input ids layout"
        );
        let mut input = Self {
            ids,
            modalities: Vec::new(),
            pixels: Vec::new(),
            grids: Vec::new(),
        };
        let (pixels, sizes) = match case.split('_').next() {
            Some("text") => return Ok(input),
            Some("image") => ("pixel_values", "target_sizes"),
            Some("video") => ("pixel_values_videos", "target_sizes_videos"),
            _ => anyhow::bail!("unknown native fixture case={case}"),
        };
        input.modalities = integers(source, "mm_token_type_ids")?
            .into_iter()
            .map(|v| Ok(u8::try_from(v)?))
            .collect::<Result<_>>()?;
        input.pixels = floats(source, pixels)?;
        let sizes = integers(source, sizes)?;
        ensure!(
            sizes.len().is_multiple_of(2),
            "reference target_sizes layout"
        );
        input.grids = sizes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| ImageGrid {
                t: 1,
                h: s[0],
                w: s[1],
            })
            .collect();
        Ok(input)
    }

    fn video(source: &SafeTensors<'_>, model: &Path, case: &str) -> Result<Self> {
        use crate::serve::{
            detok::Shared,
            image::{Processor, VideoOptions},
        };
        let reference = Self::read(source, case)?;
        let directory = PathBuf::from(
            std::env::var_os("FF_VIDEO_REF_DIR").context("FF_VIDEO_REF_DIR missing")?,
        );
        let clip = match case {
            "video_football_16x" => "football_4s",
            "video_control_clip2_16x" => "sample_demo_1",
            "video_control_same_grid_16x" => "boat_pan_4s",
            _ => anyhow::bail!("unknown video fixture case={case}"),
        };
        let floor: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join("preprocess_video_noise_floor.json"),
        )?)?;
        let measured = floor["cases"]
            .as_object()
            .context("video pixel floor cases missing")?;
        ensure!(!measured.is_empty(), "video pixel floor has no cases");
        let tolerance = measured
            .iter()
            .try_fold(0.0f64, |bound, (name, cell)| -> Result<f64> {
                Ok(bound.max(
                    cell["tv_vs_pil_diff"]["max_abs"]
                        .as_f64()
                        .with_context(|| format!("video pixel floor {name} max_abs missing"))?,
                ))
            })? as f32
            + 4.0 * f32::EPSILON;
        ensure!(
            tolerance.is_finite() && tolerance > 0.0,
            "invalid video pixel floor {case}"
        );
        let spec = crate::model_spec::ModelSpec::read(model)?;
        let vision = spec.vision.as_ref().context("video vision spec missing")?;
        let first = reference
            .ids
            .iter()
            .position(|&id| id == vision.start_token)
            .context("video opening marker missing")?;
        let last = reference
            .ids
            .iter()
            .rposition(|&id| id == vision.end_token)
            .context("video closing marker missing")?;
        ensure!(first < last, "video marker extent {case}");
        let mut prompt = reference.ids[..first].to_vec();
        prompt.push(vision.video_token);
        prompt.extend_from_slice(&reference.ids[last + 1..]);
        let tokenizer = Shared::load(model)?;
        let processor =
            Processor::load(model, Some(&tokenizer))?.context("video processor missing")?;
        let bytes = std::fs::read(directory.join("inputs").join(format!("{clip}.mp4")))?;
        let url = format!("data:video/mp4;base64,{}", base64::encode(bytes));
        let planned = processor.plan_video(
            &url,
            VideoOptions {
                max_num_frames: 4,
                stack_frames: 1,
                max_slice_nums: Some(1),
            },
            4352,
        )?;
        let video = processor.prepare_planned_video(planned)?;
        let frames = video.frames.len();
        let (ids, prepared) = processor.expand(&prompt, Vec::new(), vec![video], 4352)?;
        let input = Self {
            ids,
            modalities: prepared.modality_ids,
            pixels: prepared.pixel_values,
            grids: prepared.grids,
        };
        ensure!(
            input.ids == reference.ids
                && input.modalities == reference.modalities
                && input.grids == reference.grids,
            "video preprocessing ids/modality/grid mismatch {case}"
        );
        ensure!(
            input.pixels.len() == reference.pixels.len()
                && input.pixels.iter().all(|value| value.is_finite()),
            "video pixel extent/finite check {case}"
        );
        let error = input
            .pixels
            .iter()
            .zip(&reference.pixels)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(
            error <= tolerance,
            "video preprocess {case} pixel max={error} > {tolerance}"
        );
        let axes = input.packed().validate(&spec, 4352)?.packed_axes()?;
        let expected = integers(source, "canvas_position_ids")?;
        ensure!(
            expected.len() == 3 * input.ids.len(),
            "video reference canvas extent {case}"
        );
        ensure!(
            (0..3).all(
                |axis| (0..input.ids.len()).all(|row| i64::from(axes[3 * row + axis])
                    == i64::from(expected[axis * input.ids.len() + row]))
            ),
            "video canvas mismatch {case}"
        );
        ensure!(
            frames == integers(source, "sampled_frame_indices")?.len(),
            "video frame count mismatch {case}"
        );
        println!(
            "PASS minicpm_video_preprocess {case} frames={frames} modality=2 ids={} pixel_max={error} measured_floor={tolerance}",
            input.ids.len()
        );
        Ok(input)
    }

    fn packed(&self) -> MultimodalInput<'_> {
        MultimodalInput {
            input_ids: &self.ids,
            modality_ids: &self.modalities,
            pixel_values: &self.pixels,
            grids: &self.grids,
        }
    }
}

fn teacher_steps(
    source: &SafeTensors<'_>,
    case: &str,
    format: crate::model_spec::ExpertFormat,
) -> Result<usize> {
    let steps = if format == crate::model_spec::ExpertFormat::Bf16 && case.starts_with("text_") {
        64
    } else {
        32
    };
    let rows = [steps];
    let pairs = [steps, 2];
    for (key, expected) in [
        ("greedy_token_ids", rows.as_slice()),
        ("step_top2_ids", pairs.as_slice()),
        ("step_top2_logits", pairs.as_slice()),
        ("step_top2_margin", rows.as_slice()),
    ] {
        let actual = source.tensor(key)?;
        ensure!(
            actual.shape() == expected,
            "teacher fixture {case}/{key} shape={:?}, expected={expected:?}",
            actual.shape()
        );
    }
    Ok(steps)
}

fn native_cases(format: crate::model_spec::ExpertFormat) -> &'static [&'static str] {
    match format {
        crate::model_spec::ExpertFormat::Bf16 => &[
            "text_32",
            "text_1k",
            "text_4k",
            "image_hockey_16x",
            "image_boat_16x",
            "video_football_16x",
            "video_control_clip2_16x",
            "video_control_same_grid_16x",
        ],
        crate::model_spec::ExpertFormat::Fp8W8a16 => &[
            "text_32",
            "text_1k",
            "text_4k",
            "image_hockey_16x",
            "video_football_16x",
            "video_control_clip2_16x",
            "video_control_same_grid_16x",
        ],
    }
}

fn check_fixtures(fixtures: &Path, model_dir: &Path, cases: &[&str]) -> Result<()> {
    let model = crate::model_spec::ModelSpec::read(model_dir)?;
    let format = model
        .expert_format
        .context("native fixture expert_format missing")?;
    for &case in cases {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        teacher_steps(&source, case, format)?;
        Bounds::read(fixtures, case, format)?;
        let input = if case.starts_with("video_") {
            Input::video(&source, model_dir, case)?
        } else {
            Input::read(&source, case)?
        };
        ensure!(
            input.ids.len() >= 8
                && input.ids.len() <= 4096
                && input.ids.iter().all(|&id| id < model.vocab),
            "fixture prompt extent/vocabulary {case}"
        );
        if !input.grids.is_empty() {
            input.packed().validate(&model, 4352)?;
        }
        if case == "image_hockey_16x" {
            let bytes =
                std::fs::read(fixtures.join("control_boat_at_hockey_size_16x.safetensors"))?;
            let tensors = SafeTensors::deserialize(&bytes)?;
            let control = Input::read(&tensors, "image_control")?;
            control.packed().validate(&model, 4352)?;
            ensure!(
                control.ids == input.ids
                    && control.grids == input.grids
                    && control.modalities == input.modalities,
                "reference media control differs outside pixels"
            );
        }
        for (name, tensor) in source.tensors() {
            if matches!(tensor.dtype(), Dtype::BF16 | Dtype::F32) {
                floats(&source, &name)?;
            }
        }
        for key in ["greedy_token_ids", "step_top2_ids"] {
            ensure!(
                integers(&source, key)?.iter().all(|&id| id < model.vocab),
                "fixture decode vocabulary {case}/{key}"
            );
        }
        ensure!(
            source.tensor("prefill_last_logits_bf16")?.shape() == [1, model.vocab as usize]
                && source.tensor("final_norm_last")?.shape() == [1, model.hidden as usize],
            "fixture logits/norm layout {case}"
        );
    }
    Ok(())
}

fn chain_state(model: &Engine) -> Result<Vec<u8>> {
    let stream = &model.device.stream;
    let mut values = Vec::new();
    for layer in &model.state.gdn {
        for buffer in [&layer.s, &layer.history] {
            for value in stream.clone_dtoh(buffer)? {
                ensure!(value.is_finite(), "nonfinite chain recurrent state");
                values.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    for cache in &model.state.kv {
        for buffer in [&cache.k, &cache.v] {
            for value in stream.clone_dtoh(buffer)? {
                ensure!(value.is_finite(), "nonfinite chain KV");
                values.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(values)
}

fn restore_chain_state(model: &mut Engine, mut values: &[u8]) -> Result<()> {
    let stream = &model.device.stream;
    for layer in &mut model.state.gdn {
        for buffer in [&mut layer.s, &mut layer.history] {
            let (saved, tail) = values
                .split_at_checked(buffer.len() * size_of::<f32>())
                .context("chain recurrent state restore extent")?;
            let saved: Vec<_> = saved
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect();
            stream.memcpy_htod(&saved, buffer)?;
            values = tail;
        }
    }
    for cache in &mut model.state.kv {
        for buffer in [&mut cache.k, &mut cache.v] {
            let (saved, tail) = values
                .split_at_checked(buffer.len() * size_of::<bf16>())
                .context("chain KV restore extent")?;
            let saved: Vec<_> = saved
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| bf16::from_bits(u16::from_le_bytes(*value)))
                .collect();
            stream.memcpy_htod(&saved, buffer)?;
            values = tail;
        }
    }
    ensure!(values.is_empty(), "chain state restore has trailing bytes");
    Ok(())
}

#[test]
#[ignore = "requires CUDA, published target/draft and an owned FLYINGFISH_RUNTIME_DIR"]
fn minicpm_chain_verify_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: Some(draft),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    Engine::calibrate(&target, 0, 256, Some(32), settings.clone())?;
    let mut model = Engine::load(&target, 0, 256, Some(32), settings)?;
    let stream = model.device.stream.clone();
    let (rows, taps) = {
        let spec = model.spec.as_ref().context("chain draft missing")?;
        (
            spec.draft.config.verify_rows()?,
            spec.draft.config.target_layer_ids.clone(),
        )
    };
    ensure!(
        rows == 16 && model.model_spec.moe.is_some(),
        "production block15 MoE fixture required"
    );
    let h = model.config.hidden_size;
    let width = h * taps.len();
    ensure!(
        chain_state(&model)?.iter().all(|&value| value == 0),
        "graph setup changed initial target state"
    );
    let mut initial_draft = Trace::default();
    model
        .spec
        .as_ref()
        .context("chain missing")?
        .draft
        .snapshot_kv(&mut initial_draft, 0, rows)?;
    for layer in 0..model
        .spec
        .as_ref()
        .context("chain missing")?
        .draft
        .config
        .num_hidden_layers
    {
        for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
            ensure!(
                initial_draft
                    .values(&name)?
                    .iter()
                    .all(|value| value.to_bits() == 0),
                "graph setup changed initial draft KV layer={layer}"
            );
        }
    }
    let mut defects = 0;
    let mut wrong_layer = 0;
    for prefix in [17usize, 65] {
        model.reset()?;
        let ids: Vec<_> = (0..prefix).map(|row| (row * 263 + 17) as u32).collect();
        let mut trace = Trace::all_layers();
        if prefix <= model.chunk {
            model.prefill_tail(&ids, &mut trace, ids.len())?;
            let capture = model
                .hidden_capture
                .as_ref()
                .context("prefill tap binding missing")?;
            let values = stream.clone_dtoh(&capture.values.slice(..prefix * width))?;
            for (slot, layer) in taps.iter().enumerate() {
                let expected = trace.values(&format!("layer.{layer}"))?;
                ensure!(
                    expected.len() == prefix * h,
                    "prefill trace extent mismatch"
                );
                defects += values
                    .chunks_exact(width)
                    .enumerate()
                    .map(|(row, values)| {
                        values[slot * h..(slot + 1) * h]
                            .iter()
                            .zip(&expected[row * h..(row + 1) * h])
                            .filter(|(a, b)| a.to_bits() != bf16::from_f32(**b).to_bits())
                            .count()
                    })
                    .sum::<usize>();
            }
            let saved = values;
            let first = taps[0];
            let capture = model
                .hidden_capture
                .as_mut()
                .context("tap control binding missing")?;
            ensure!(
                capture.layer_slots[first + 1].is_none(),
                "wrong-layer control overlaps a tap"
            );
            capture.layer_slots[first] = None;
            capture.layer_slots[first + 1] = Some(0);
            model.reset()?;
            model.prefill(&ids, None)?;
            let capture = model
                .hidden_capture
                .as_mut()
                .context("tap control binding missing")?;
            let changed = stream.clone_dtoh(&capture.values.slice(..prefix * width))?;
            wrong_layer += changed
                .chunks_exact(width)
                .zip(saved.chunks_exact(width))
                .map(|(a, b)| {
                    a[..h]
                        .iter()
                        .zip(&b[..h])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                })
                .sum::<usize>();
            capture.layer_slots[first + 1] = None;
            capture.layer_slots[first] = Some(0);
            model.reset()?;
            model.prefill(&ids, None)?;
        } else {
            model.prefill(&ids, None)?;
        }
        let inputs: Vec<_> = (0..rows)
            .map(|row| ((prefix + row) * 541 + 37) as u32)
            .collect();
        model.set_token(inputs[0])?;
        model
            .spec
            .as_mut()
            .context("chain missing")?
            .checkpoint
            .as_mut()
            .context("chain checkpoint missing")?
            .save(&model.state, &model.work.token, prefix, &stream, None)?;
        let mut expected_states = vec![chain_state(&model)?];
        let mut logits = Vec::new();
        let mut captures = Vec::new();
        let mut predictions = Vec::new();
        for token in &inputs {
            model.set_token(*token)?;
            model.advance(false, None)?;
            logits.extend(stream.clone_dtoh(&model.work.logits)?);
            predictions.push(model.token()?);
            captures.extend(
                stream.clone_dtoh(
                    &model
                        .hidden_capture
                        .as_ref()
                        .context("decode taps missing")?
                        .values
                        .slice(..width),
                )?,
            );
            expected_states.push(chain_state(&model)?);
        }
        model
            .spec
            .as_ref()
            .context("chain missing")?
            .checkpoint
            .as_ref()
            .context("chain checkpoint missing")?
            .restore(&mut model.state, &mut model.work.token, &stream)?;
        model.position = prefix;
        model.committed_prefix_ids.truncate(prefix);
        model.set_token(inputs[0])?;
        stream.memcpy_htod(&[prefix as i32], &mut model.work.position)?;
        stream.memcpy_htod(&[prefix as i32 + 1], &mut model.work.length)?;
        stream.memcpy_htod(&inputs, &mut model.work.ids.slice_mut(..rows))?;
        model.run_step(
            crate::backend::Step {
                rows,
                input: crate::backend::Input::Ids,
                mask: crate::backend::Mask::Causal,
                head: Head::All,
            },
            None,
        )?;
        let spec = model.spec.as_ref().context("chain missing")?;
        let actual = stream.clone_dtoh(&spec.logits)?;
        let logit_mismatches = actual
            .iter()
            .zip(&logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let actual = stream.clone_dtoh(
            &model
                .hidden_capture
                .as_ref()
                .context("verify taps missing")?
                .values
                .slice(..rows * width),
        )?;
        let tap_mismatches = actual
            .iter()
            .zip(&captures)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let id_mismatches = stream
            .clone_dtoh(&spec.predictions)?
            .iter()
            .zip(&predictions)
            .filter(|(a, b)| a != b)
            .count();
        defects += logit_mismatches + tap_mismatches + id_mismatches;
        for accepted in (0..=rows).rev() {
            let spec = model.spec.as_ref().context("chain missing")?;
            spec.checkpoint
                .as_ref()
                .context("chain checkpoint missing")?
                .select_prefix(&mut model.state, &spec.snapshots, accepted, &stream)?;
            let actual = chain_state(&model)?;
            let mismatches = actual
                .iter()
                .zip(&expected_states[accepted])
                .filter(|(a, b)| a != b)
                .count();
            defects += mismatches;
            println!(
                "{} minicpm_chain_prefix prefix={prefix} accepted={accepted} state_bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
        }
        println!(
            "{} minicpm_chain_verify prefix={prefix} rows={rows} logits_bit_mismatches={logit_mismatches} capture_bit_mismatches={tap_mismatches} id_mismatches={id_mismatches}",
            if logit_mismatches + tap_mismatches + id_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
    }
    let ids: Vec<_> = (0..17).map(|row| (row * 263 + 17) as u32).collect();
    model.reset()?;
    model.prefill(&ids, None)?;
    let mut greedy = Vec::new();
    let mut greedy_states = vec![chain_state(&model)?];
    let mut greedy_logits = Vec::new();
    let mut greedy_taps = Vec::new();
    model
        .spec
        .as_mut()
        .context("chain missing")?
        .checkpoint
        .as_mut()
        .context("checkpoint missing")?
        .save(&model.state, &model.work.token, ids.len(), &stream, None)?;
    for _ in 0..rows {
        greedy.push(model.decode(false, None)?);
        greedy_states.push(chain_state(&model)?);
        greedy_logits.push(stream.clone_dtoh(&model.work.logits)?);
        greedy_taps.extend(
            stream.clone_dtoh(
                &model
                    .hidden_capture
                    .as_ref()
                    .context("greedy taps missing")?
                    .values
                    .slice(..width),
            )?,
        );
    }
    let greedy_taps = stream.clone_htod(&greedy_taps)?;
    for graph in [false, true] {
        model.set_spec_graph(graph)?;
        for (matched, limit, eos) in [
            (0usize, 16, None),
            (7, 16, None),
            (15, 16, None),
            (15, 1, None),
            (15, 8, None),
            (15, 16, Some(greedy[3])),
            (16, 16, None),
        ] {
            model.reset()?;
            model.prefill(&ids, None)?;
            model
                .spec
                .as_ref()
                .context("chain missing")?
                .checkpoint
                .as_ref()
                .context("checkpoint missing")?
                .restore(&mut model.state, &mut model.work.token, &stream)?;
            let anchor = model.token()?;
            model.set_token(anchor)?;
            model.set_spec_eos(&eos.into_iter().collect::<Vec<_>>())?;
            if matched < rows {
                let mut proposals = greedy[..rows - 1].to_vec();
                if matched < rows - 1 {
                    proposals[matched] = (proposals[matched] + 1) % model.model_spec.vocab;
                }
                model
                    .spec
                    .as_mut()
                    .context("chain missing")?
                    .forced_proposals = Some(proposals);
            }
            let graph_range =
                super::super::super::profile::Range::new(graph, c"minicpm_chain_graph");
            let round = model.spec_round(limit, None, "chain")?;
            drop(graph_range);
            let committed = round.output.len();
            ensure!(
                committed > 0 && committed <= rows,
                "Runner FAIL transaction committed extent"
            );
            let mut expected = greedy[..(matched + 1).min(rows).min(limit)].to_vec();
            if matched == rows {
                expected.truncate(committed);
            }
            if let Some(index) = expected.iter().position(|id| Some(*id) == eos) {
                expected.truncate(index + 1);
            }
            let output_mismatches = usize::from(round.output != expected)
                + usize::from(round.accepted + 1 != committed);
            let boundary_mismatches = usize::from(model.position != ids.len() + committed)
                + usize::from(model.pending_token != expected.last().copied())
                + usize::from(
                    model.committed_prefix_ids
                        != ids
                            .iter()
                            .copied()
                            .chain(std::iter::once(anchor))
                            .chain(expected.iter().copied().take(committed - 1))
                            .collect::<Vec<_>>(),
                );
            let actual = chain_state(&model)?;
            let state_mismatches = actual
                .iter()
                .zip(&greedy_states[committed])
                .filter(|(a, b)| a != b)
                .count();
            let actual = stream.clone_dtoh(&model.work.logits)?;
            let logit_mismatches = actual
                .iter()
                .zip(&greedy_logits[committed - 1])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let draft = &mut model.spec.as_mut().context("chain missing")?.draft;
            let mut actual = Trace::default();
            draft.snapshot_kv(&mut actual, ids.len(), rows)?;
            draft.restore_kv(ids.len(), 0)?;
            draft.inject(
                &greedy_taps,
                committed,
                ids.len(),
                super::super::super::draft::Compute {
                    ops: &model.ops,
                    blas: &mut model.blas,
                },
                None,
            )?;
            let mut expected_kv = Trace::default();
            draft.snapshot_kv(&mut expected_kv, ids.len(), rows)?;
            let mut draft_mismatches = 0;
            for layer in 0..draft.config.num_hidden_layers {
                for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
                    draft_mismatches += actual
                        .values(&name)?
                        .iter()
                        .zip(expected_kv.values(&name)?)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
            }
            let mismatches = output_mismatches
                + boundary_mismatches
                + state_mismatches
                + logit_mismatches
                + draft_mismatches;
            defects += mismatches;
            println!(
                "{} minicpm_chain_commit graph={graph} matched={matched} limit={limit} eos={eos:?} accepted={} committed={committed} output_mismatches={output_mismatches} boundary_mismatches={boundary_mismatches} state_bit_mismatches={state_mismatches} logits_bit_mismatches={logit_mismatches} draft_kv_bit_mismatches={draft_mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" },
                round.accepted
            );
        }
    }
    for graph in [false, true] {
        model.set_spec_graph(graph)?;
        model.reset()?;
        model.prefill(&ids, None)?;
        let before = chain_state(&model)?;
        let mut draft_before = Trace::default();
        model
            .spec
            .as_ref()
            .context("chain missing")?
            .draft
            .snapshot_kv(&mut draft_before, ids.len(), rows)?;
        let pending = model.pending_token;
        let missing = if graph {
            model
                .spec
                .as_mut()
                .context("chain missing")?
                .forced_proposals = Some(greedy[..rows - 1].to_vec());
            model.spec.as_mut().context("chain missing")?.inject_graphs[7].take()
        } else {
            model.capture_fail_after_layer = Some((0, 1));
            None
        };
        let error = model
            .spec_round(8, None, "chain_error")
            .err()
            .context("Runner FAIL transaction fault was ignored")?;
        if graph {
            ensure!(
                error
                    .to_string()
                    .contains("MoE commit graph rows=8 missing before Ready"),
                "Runner FAIL graph fault reached a different failure: {error}"
            );
            model.spec.as_mut().context("chain missing")?.inject_graphs[7] = missing;
        } else {
            ensure!(
                error.to_string().contains("injected capture failure"),
                "Runner FAIL eager fault reached a different failure: {error}"
            );
        }
        model.capture_fail_after_layer = None;
        ensure!(
            model.poisoned,
            "Runner FAIL transaction fault was not exposed"
        );
        let mut error_mismatches = usize::from(chain_state(&model)? != before)
            + usize::from(
                model.position != ids.len()
                    || model.pending_token != pending
                    || model.committed_prefix_ids != ids,
            );
        let mut draft_after = Trace::default();
        let draft = &model.spec.as_ref().context("chain missing")?.draft;
        draft.snapshot_kv(&mut draft_after, ids.len(), rows)?;
        for layer in 0..draft.config.num_hidden_layers {
            for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
                error_mismatches += draft_before
                    .values(&name)?
                    .iter()
                    .zip(draft_after.values(&name)?)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
        }
        defects += error_mismatches;
        println!(
            "{} minicpm_chain_error graph={graph} restored_bit_mismatches={error_mismatches} poisoned=true",
            if error_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        model.reset()?;
    }
    ensure!(
        wrong_layer > 0,
        "Runner FAIL wrong-layer capture control was not detected"
    );
    println!(
        "{} minicpm_chain_verify_end rows=16 prefixes=17,65 accepted=0..16 wrong_layer_mismatches={wrong_layer} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(
        defects == 0,
        "DEFECT production chain verification cells={defects}"
    );
    Ok(())
}

#[test]
#[ignore = "CPU-only native fixtures; requires FF_MODEL, FF_MINICPM_REFERENCE_DIR and FF_VIDEO_REF_DIR"]
fn minicpm_native_checkpoint_fixtures_cpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("reference directory missing")?,
    );
    let format = crate::model_spec::ModelSpec::read(&model)?
        .expert_format
        .context("native fixture expert_format missing")?;
    let cases = native_cases(format);
    check_fixtures(&fixtures, &model, cases)?;
    println!(
        "PASS minicpm_native_fixture_preflight format={format:?} cases={} video_decode=ffmpeg canvas=true floors_complete=true",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "exclusive CUDA profiling, native FP8 FF_MODEL, FF_MINICPM_REFERENCE_DIR, FF_VIDEO_REF_DIR and FLYINGFISH_RUNTIME_DIR required"]
fn minicpm_native_prefill_profile_gpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FP8 model missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("profile input directory missing")?,
    );
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("profile runtime missing")?,
    );
    let spec = crate::model_spec::ModelSpec::read(&model)?;
    ensure!(
        spec.expert_format == Some(crate::model_spec::ExpertFormat::Fp8W8a16)
            && spec.hidden == 2048
            && spec.layers.len() == 40,
        "native FP8 profile requires the published MiniCPM geometry"
    );
    let settings = Settings {
        runtime_dir: Some(runtime),
        prefix_cache: false,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    Engine::calibrate(&model, 0, 4352, None, settings.clone())?;
    let mut engine = Engine::load(&model, 0, 4352, None, settings)?;
    let cases = native_cases(crate::model_spec::ExpertFormat::Fp8W8a16);
    check_fixtures(&fixtures, &model, cases)?;
    for &case in cases {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let input = if case.starts_with("video_") {
            Input::video(&source, &model, case)?
        } else {
            Input::read(&source, case)?
        };
        for capture in [false, true] {
            engine.reset()?;
            if capture {
                engine.profiler_start()?;
            }
            let started = Instant::now();
            let result = if input.grids.is_empty() {
                engine.prefill(&input.ids, None)
            } else {
                engine.prefill_multimodal(&input.packed(), None)
            };
            engine.device.stream.synchronize()?;
            let seconds = started.elapsed().as_secs_f64();
            if capture {
                engine.profiler_stop()?;
            }
            result?;
            ensure!(
                engine
                    .decode_logits()?
                    .iter()
                    .all(|value| value.is_finite()),
                "nonfinite native prefill output {case}"
            );
            if capture {
                println!(
                    "{}",
                    serde_json::json!({"minicpm_native_prefill_profile": {
                        "case":case,"input_layout":[1,input.ids.len()],"hidden":spec.hidden,
                        "graphs":false,"seconds":seconds,"warm_requests":1,
                        "region":"native full-model media and text prefill; setup and warm request excluded"
                    }})
                );
            }
        }
    }
    println!(
        "PASS minicpm_native_prefill_profile cases={} graphs=false finite=true capture=cudaProfilerApi",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "exclusive CUDA timing, BF16 FF_MODEL, FF_MINICPM_REFERENCE_DIR and pinned FLYINGFISH_RUNTIME_DIR required"]
fn minicpm_bf16_timing_arm_gpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("BF16 timing model missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("BF16 timing reference missing")?,
    );
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("BF16 timing runtime missing")?,
    );
    let spec = crate::model_spec::ModelSpec::read(&model)?;
    ensure!(
        spec.expert_format == Some(crate::model_spec::ExpertFormat::Bf16)
            && spec.hidden == 2048
            && spec.layers.len() == 40,
        "BF16 timing requires the published MiniCPM geometry"
    );
    let settings = Settings {
        runtime_dir: Some(runtime),
        prefix_cache: false,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let mut engine = Engine::load(&model, 0, 4352, None, settings)?;
    ensure!(
        !engine.graphs.is_empty(),
        "BF16 timing requires segmented graphs"
    );
    let read_bytes = || -> Result<u64> {
        std::fs::read_to_string("/proc/self/io")?
            .lines()
            .find_map(|line| line.strip_prefix("read_bytes:"))
            .context("BF16 timing process has no physical read counter")?
            .trim()
            .parse()
            .context("invalid BF16 timing physical read counter")
    };
    for case in ["text_32", "text_1k"] {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let input = Input::read(&source, case)?;
        let forced = integers(&source, "greedy_token_ids")?;
        let margins = floats(&source, "step_top2_margin")?;
        ensure!(
            forced.len() >= 32 && margins.len() >= 32 && input.ids.len() <= 4096,
            "BF16 timing fixture extent: {case}"
        );
        for run in 0..3 {
            engine.device.stream.synchronize()?;
            let reads_before = read_bytes()?;
            let started = Instant::now();
            engine.reset()?;
            let mut predictions = vec![engine.prefill(&input.ids, None)?];
            for &token in &forced[..31] {
                engine.set_token(token)?;
                predictions.push(engine.decode(true, None)?);
            }
            engine.device.stream.synchronize()?;
            let seconds = started.elapsed().as_secs_f64();
            let reads = read_bytes()?
                .checked_sub(reads_before)
                .context("BF16 timing physical read counter decreased")?;
            ensure!(
                engine.decode_logits()?.iter().all(|v| v.is_finite()),
                "BF16 timing nonfinite output: {case}"
            );
            println!(
                "{}",
                serde_json::json!({"minicpm_bf16_timing_arm":{
                    "case":case,"run":run,"measured":run>0,"seconds":seconds,
                    "read_bytes":reads,
                    "prompt_tokens":input.ids.len(),"teacher_steps":32,"predictions":predictions,
                    "reference_top2_margins":&margins[..32],"graph_segments":engine.graphs.len(),
                    "region":"reset+prefill+31 forced graph decodes; synchronization and token transfers included; load/calibration excluded"
                }})
            );
        }
    }
    let mut stats = Trace::default();
    engine.record_test_stats(&mut stats)?;
    println!(
        "{}",
        serde_json::json!({"minicpm_bf16_timing_modes":{
            "resident_hits":stats.values("stats.gpu_resident_hits")?[0],
            "cpu_misses":stats.values("stats.cpu_worker_misses")?[0],
            "disk_experts":stats.values("residency.tier")?.iter().filter(|&&tier| tier==2.0).count(),
            "graph_segments":engine.graphs.len()
        }})
    );
    Ok(())
}

fn chain_replay(
    model: &mut Engine,
    round: &crate::dspark::Round,
    origin: usize,
    anchor: u32,
) -> Result<usize> {
    let stream = model.device.stream.clone();
    let actual = chain_state(model)?;
    model.check_logits()?;
    let verified = stream.clone_dtoh(&model.spec.as_ref().context("chain missing")?.logits)?;
    ensure!(
        verified.iter().all(|value| value.is_finite()),
        "nonfinite chain verification logits"
    );
    let logits = stream.clone_dtoh(&model.work.logits)?;
    let sequence = stream.clone_dtoh(
        model
            .state
            .sequence
            .as_ref()
            .context("chain sequence missing")?,
    )?;
    let position = stream.clone_dtoh(&model.work.position)?;
    let length = stream.clone_dtoh(&model.work.length)?;
    let token = model.token()?;
    let boundary = (
        model.position,
        model.pending_token,
        model.resident_ids_len,
        model.last_input_kind,
        model.committed_prefix_ids.clone(),
    );
    model
        .spec
        .as_ref()
        .context("chain missing")?
        .checkpoint
        .as_ref()
        .context("chain checkpoint missing")?
        .restore(&mut model.state, &mut model.work.token, &stream)?;
    model.position = origin;
    model.committed_prefix_ids.truncate(origin);
    stream.memcpy_htod(&[origin as i32], &mut model.work.position)?;
    stream.memcpy_htod(&[origin as i32 + 1], &mut model.work.length)?;
    let mut predictions = Vec::new();
    for input in
        std::iter::once(anchor).chain(round.output.iter().copied().take(round.committed - 1))
    {
        model.set_token(input)?;
        predictions.push(model.decode(false, None)?);
    }
    let expected = chain_state(model)?;
    ensure!(actual.len() == expected.len(), "chain replay state extent");
    let state_mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| **a != *b)
        .count();
    let replay_logits = stream.clone_dtoh(&model.work.logits)?;
    let logit_mismatches = logits
        .iter()
        .zip(replay_logits)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let metadata_mismatches = usize::from(
        boundary.0 != model.position
            || boundary.1 != model.pending_token
            || boundary.4 != model.committed_prefix_ids
            || token != model.token()?
            || position != stream.clone_dtoh(&model.work.position)?
            || length != stream.clone_dtoh(&model.work.length)?
            || sequence
                != stream.clone_dtoh(
                    model
                        .state
                        .sequence
                        .as_ref()
                        .context("chain sequence missing")?,
                )?,
    );
    let output_mismatches = usize::from(predictions != round.output);
    let acceptance_mismatches = usize::from(round.accepted + 1 != round.committed);
    let mismatches = state_mismatches
        + logit_mismatches
        + metadata_mismatches
        + output_mismatches
        + acceptance_mismatches;
    println!(
        "{} minicpm_chain_boundary origin={origin} committed={} state_bit_mismatches={state_mismatches} logits_bit_mismatches={logit_mismatches} metadata_mismatches={metadata_mismatches} output_mismatches={output_mismatches} acceptance_mismatches={acceptance_mismatches}",
        if mismatches == 0 { "PASS" } else { "DEFECT" },
        round.committed
    );
    restore_chain_state(model, &actual)?;
    stream.memcpy_htod(
        &sequence,
        model
            .state
            .sequence
            .as_mut()
            .context("chain sequence missing")?,
    )?;
    stream.memcpy_htod(&position, &mut model.work.position)?;
    stream.memcpy_htod(&length, &mut model.work.length)?;
    stream.memcpy_htod(&logits, &mut model.work.logits)?;
    model.set_token(token)?;
    model.position = boundary.0;
    model.pending_token = boundary.1;
    model.resident_ids_len = boundary.2;
    model.last_input_kind = boundary.3;
    model.committed_prefix_ids = boundary.4;
    Ok(mismatches)
}

#[test]
#[ignore = "requires CUDA, published target/draft, five T0 fixtures and owned runtime"]
fn minicpm_chain_decode_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let fixtures = PathBuf::from(std::env::var("FF_MINICPM_REFERENCE_DIR")?);
    let cases = [
        "text_32",
        "text_1k",
        "text_4k",
        "image_hockey_16x",
        "image_boat_16x",
    ];
    let mut inputs = Vec::new();
    for case in cases {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        inputs.push(Input::read(&source, case)?);
    }
    let rows = crate::dspark::DraftConfig::read(
        &draft,
        &crate::model_spec::ModelSpec::read(&target)?.config(),
    )?
    .verify_rows()?;
    let capacity = (inputs
        .iter()
        .map(|input| input.ids.len())
        .max()
        .context("no chain inputs")?
        + 256
        + rows)
        .next_multiple_of(32);
    let mut settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    Engine::calibrate(&target, 0, capacity, None, settings.clone())?;
    let mut plain = Engine::load(&target, 0, capacity, None, settings.clone())?;
    ensure!(plain.spec.is_none(), "plain baseline loaded a draft");
    let layout = plain
        .moe_slots
        .as_ref()
        .context("plain slots missing")?
        .storage
        .layout();
    let plain_slots = (0..plain.model_spec.layers.len())
        .map(|layer| layout.slot_count(layer as u32))
        .collect::<Result<Vec<_>>>()?;
    let eos = plain.model_spec.eos.clone();
    let mut plain_only = Vec::new();
    for (case, input) in cases.iter().zip(&inputs) {
        plain.reset()?;
        if !input.grids.is_empty() {
            plain.prepare_multimodal(&input.packed(), None)?;
        }
        let mut ids = vec![plain.prefill(&input.ids, None)?];
        while ids.len() < 256 && !eos.contains(ids.last().context("plain output missing")?) {
            ids.push(plain.decode(false, None)?);
        }
        plain.check_logits()?;
        ensure!(
            plain
                .qwen
                .as_ref()
                .context("plain runtime missing")?
                .vision_encode_calls
                == usize::from(!input.grids.is_empty()),
            "plain encoder count {case}"
        );
        println!(
            "minicpm_chain_plain_only case={case} tokens={} ids={ids:?} decoded={:?}",
            ids.len(),
            crate::tokenizer::decode(&target, &ids)?
        );
        plain_only.push(ids);
    }
    plain.synchronize()?;
    drop(plain);
    settings.draft_model = Some(draft);
    Engine::calibrate(&target, 0, capacity, None, settings.clone())?;
    let mut model = Engine::load(&target, 0, capacity, None, settings.clone())?;
    ensure!(model.spec.is_some(), "chain target has no attached draft");
    model.set_spec_eos(&eos)?;
    let mut baseline = Vec::new();
    for (case, input) in cases.iter().zip(&inputs) {
        model.reset()?;
        if !input.grids.is_empty() {
            model.prepare_multimodal(&input.packed(), None)?;
        }
        let mut ids = vec![model.prefill(&input.ids, None)?];
        while ids.len() < 256 && !eos.contains(ids.last().context("M1 output missing")?) {
            ids.push(model.decode(false, None)?);
        }
        model.check_logits()?;
        ensure!(
            model
                .qwen
                .as_ref()
                .context("M1 runtime missing")?
                .vision_encode_calls
                == usize::from(!input.grids.is_empty()),
            "M1 encoder count {case}"
        );
        println!(
            "PASS minicpm_chain_m1 case={case} draft_attached=true tokens={} ids={ids:?} decoded={:?}",
            ids.len(),
            crate::tokenizer::decode(&target, &ids)?
        );
        baseline.push(ids);
    }
    let layout = model
        .moe_slots
        .as_ref()
        .context("chain slots missing")?
        .storage
        .layout();
    let chain_slots = (0..model.model_spec.layers.len())
        .map(|layer| layout.slot_count(layer as u32))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "minicpm_chain_deployment capacity={capacity} rows={rows} plain_slots={plain_slots:?} chain_slots={chain_slots:?} slot_delta={}",
        i64::from(chain_slots.iter().sum::<u32>()) - i64::from(plain_slots.iter().sum::<u32>())
    );
    let mut defects = 0;
    for graph in [false, true] {
        model.set_spec_graph(graph)?;
        let mut media_outputs = Vec::new();
        for (((case, input), expected), ordinary) in
            cases.iter().zip(&inputs).zip(&baseline).zip(&plain_only)
        {
            model.reset()?;
            if !input.grids.is_empty() {
                model.prepare_multimodal(&input.packed(), None)?;
            }
            let mut ids = vec![model.prefill(&input.ids, None)?];
            let (mut rounds, mut accepted, mut committed) = (0, 0, 0);
            while ids.len() < 256 && !eos.contains(ids.last().context("chain output missing")?) {
                let origin = model.position;
                let anchor = model.token()?;
                let round = model.spec_round(256 - ids.len(), None, case)?;
                ensure!(
                    round.committed > 0
                        && round.committed == round.output.len()
                        && round.committed <= rows,
                    "chain committed extent {case}"
                );
                defects += chain_replay(&mut model, &round, origin, anchor)?;
                println!(
                    "minicpm_chain_round case={case} graph={graph} round={rounds} accepted={} committed={} output={:?}",
                    round.accepted, round.committed, round.output
                );
                accepted += round.accepted;
                committed += round.committed;
                rounds += 1;
                ids.extend(&round.output);
            }
            let calls = model
                .qwen
                .as_ref()
                .context("chain runtime missing")?
                .vision_encode_calls;
            ensure!(
                calls == usize::from(!input.grids.is_empty()),
                "chain encoder count {case}"
            );
            let mismatches = ids.iter().zip(expected).filter(|(a, b)| a != b).count()
                + ids.len().abs_diff(expected.len());
            defects += mismatches;
            println!(
                "{} minicpm_chain_lossless case={case} graph={graph} tokens={} expected_tokens={} token_mismatches={mismatches} rounds={rounds} accepted={accepted} committed={committed} encoder_calls={calls} ids={ids:?} decoded={:?} baseline=draft_attached_m1",
                if mismatches == 0 { "PASS" } else { "DEFECT" },
                ids.len(),
                expected.len(),
                crate::tokenizer::decode(&target, &ids)?
            );
            let agreement = ids.iter().zip(ordinary).filter(|(a, b)| a == b).count();
            println!(
                "minicpm_chain_plain_only_agreement case={case} graph={graph} exact={} matching_positions={agreement} dspark_tokens={} plain_only_tokens={} descriptive=true",
                ids == *ordinary,
                ids.len(),
                ordinary.len()
            );
            if !input.grids.is_empty() {
                media_outputs.push(ids);
            }
        }
        let distinct = media_outputs.len() == 2 && media_outputs[0] != media_outputs[1];
        defects += usize::from(!distinct);
        println!(
            "{} minicpm_chain_two_media graph={graph} distinct_ids={distinct}",
            if distinct { "PASS" } else { "DEFECT" }
        );
    }
    model.synchronize()?;
    drop(model);
    let options = crate::spec::Options {
        device: 0,
        capacity,
        chunk: None,
        backend: settings,
        count: rows + 1,
        ignore_eos: false,
    };
    let mut public =
        crate::model::Model::load(&target, 0, capacity, None, options.backend.clone())?;
    let first = baseline[0][0];
    let eos_index = baseline[0]
        .iter()
        .take(rows)
        .position(|&token| token != first)
        .context("Runner FAIL EOS fixture has no distinct decode token")?;
    let committed_counts = |output: &crate::spec::Output| {
        output.rounds.iter().all(|round| {
            round.accepted + 1 == round.committed && round.committed == round.output.len()
        })
    };
    for graph in [false, true] {
        public.set_spec_graph(graph)?;
        public.config.eos_token_id = vec![baseline[0][eos_index]];
        let output = crate::spec::produce_events(
            &mut public,
            &inputs[0].ids,
            None,
            &options,
            None,
            &mut |_| true,
        )?;
        let pass = output.ids == baseline[0][..=eos_index] && committed_counts(&output);
        defects += usize::from(!pass);
        println!(
            "{} minicpm_chain_production_eos graph={graph} tokens={} expected_tokens={} ids={:?}",
            if pass { "PASS" } else { "DEFECT" },
            output.ids.len(),
            eos_index + 1,
            output.ids
        );
        ensure!(
            public.rewind(public.position(), public.token()?).is_err(),
            "Runner FAIL old scalar EOS rewind control did not reject recurrent state"
        );
        public.config.eos_token_id = eos.clone();
        for index in [3, 4] {
            let input = &inputs[index];
            let output = crate::spec::produce_events(
                &mut public,
                &input.ids,
                Some(&input.packed()),
                &options,
                None,
                &mut |_| true,
            )?;
            let expected = &baseline[index][..options.count.min(baseline[index].len())];
            let pass = output.ids == expected && committed_counts(&output);
            defects += usize::from(!pass);
            println!(
                "{} minicpm_chain_production_media case={} graph={graph} tokens={} token_equal={pass} decoded={:?}",
                if pass { "PASS" } else { "DEFECT" },
                cases[index],
                output.ids.len(),
                crate::tokenizer::decode(&target, &output.ids)?
            );
        }
    }
    println!(
        "{} minicpm_chain_decode_end cases=5 modes=2 horizon=256 natural_eos=true exact_tokens=true exact_boundaries=true production_eos_and_media=true defects={defects} lossless_scope=draft_attached_m1",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "DEFECT MiniCPM chain decode count={defects}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA, T0 fixtures, pinned calibration and explicit draft/graph timing arms"]
fn minicpm_chain_decode_timing_gpu() -> Result<()> {
    // One deployment arm runs text1k/text4k with one warmup and three measured decodes each.
    chain_decode_timing(&["text_1k", "text_4k"], 4, false)
}

#[test]
#[ignore = "exclusive CUDA, native FF_MODEL, text_1k reference and pinned calibration required"]
fn minicpm_expert_swap_identity_gpu() -> Result<()> {
    chain_decode_timing(&["text_1k"], 1, true)
}

#[test]
#[ignore = "exclusive CUDA timing, native FF_MODEL, text_1k reference and pinned calibration required"]
fn minicpm_expert_swap_timing_gpu() -> Result<()> {
    chain_decode_timing(&["text_1k"], 2, false)
}

fn chain_decode_timing(cases: &[&str], runs: usize, margins: bool) -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_MODEL")?);
    let fixtures = PathBuf::from(std::env::var("FF_MINICPM_REFERENCE_DIR")?);
    let graph = match std::env::var("FF_MINICPM_DECODE_GRAPH")?.as_str() {
        "graph" => true,
        "eager" => false,
        value => anyhow::bail!("chain timing graph mode={value}"),
    };
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: std::env::var_os("FF_ARGMAX_DRAFT").map(PathBuf::from),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let mut inputs = Vec::new();
    for &case in cases {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        inputs.push((case, integers(&tensors, "input_ids")?));
    }
    let capacity = (inputs
        .iter()
        .map(|(_, ids)| ids.len())
        .max()
        .context("no timing inputs")?
        + 256
        + crate::linear_plan::MAX_SKINNY_ROWS)
        .next_multiple_of(32);
    let setup = Instant::now();
    let mut model = Engine::load(&target, 0, capacity, None, settings)?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let eos = model.model_spec.eos.clone();
    let with_draft = model.spec.is_some();
    ensure!(
        !margins || !with_draft,
        "swap identity requires plain decode"
    );
    if with_draft {
        model.set_spec_graph(graph)?;
        model.set_spec_eos(&eos)?;
    }
    let slots = model
        .moe_slots
        .as_ref()
        .context("chain timing requires MoE slots")?
        .storage
        .layout();
    let slots = (0..model.model_spec.layers.len())
        .map(|layer| slots.slot_count(layer as u32))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "minicpm_chain_timing_plan {}",
        serde_json::json!({"with_draft":with_draft,
        "capacity":capacity,"slots_per_layer":slots,"setup_seconds":setup_seconds})
    );
    let mut defects = 0;
    let margin = |model: &mut Engine| -> Result<f32> {
        let logits = model.decode_logits()?;
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite swap identity logits"
        );
        let mut top = [f32::NEG_INFINITY; 2];
        for value in logits {
            if value > top[0] {
                top[1] = top[0];
                top[0] = value;
            } else if value > top[1] {
                top[1] = value;
            }
        }
        Ok(top[0] - top[1])
    };
    for (case, input) in inputs {
        let mut expected = None;
        for run in 0..runs {
            model.reset()?;
            let prefill = Instant::now();
            let first = model.prefill(&input, None)?;
            model.synchronize()?;
            let prefill_seconds = prefill.elapsed().as_secs_f64();
            let mut ids = vec![first];
            let mut top2_margins = Vec::new();
            if margins {
                top2_margins.push(margin(&mut model)?);
            }
            let mut rounds = Vec::new();
            let range = super::super::super::profile::Range::new(run > 0, c"minicpm_chain_decode");
            let timer = Instant::now();
            while ids.len() < 256 && !eos.contains(ids.last().context("timing output missing")?) {
                if with_draft {
                    let round = model.spec_round(256 - ids.len(), None, "timing")?;
                    ids.extend(&round.output);
                    rounds.push(round);
                } else {
                    ids.push(model.decode(graph, None)?);
                    if margins {
                        top2_margins.push(margin(&mut model)?);
                    }
                }
            }
            model.synchronize()?;
            let seconds = timer.elapsed().as_secs_f64();
            drop(range);
            model.check_logits()?;
            ensure!(
                seconds.is_finite() && seconds > 0.0,
                "nonfinite/nonpositive chain timing"
            );
            let same = expected.as_ref().is_none_or(|expected| *expected == ids);
            ensure!(
                !margins || top2_margins.len() == ids.len(),
                "swap identity margin extent"
            );
            let decode_tokens = ids.len() - 1;
            let ms_per_token = seconds * 1000.0 / decode_tokens as f64;
            defects += usize::from(!same);
            println!(
                "{} minicpm_chain_decode_timing {}",
                if same { "PASS" } else { "DEFECT" },
                serde_json::json!({"case":case,"run":run,"measured":run>0,"with_draft":with_draft,
                    "graph":graph,"seconds":seconds,"prefill_seconds":prefill_seconds,
                    "tokens":ids.len(),"ids":ids,"rounds":rounds,"decode_tokens":decode_tokens,
                    "ms_per_token":ms_per_token,"margin_collection":margins,"top2_margins":top2_margins})
            );
            if expected.is_none() {
                expected = Some(ids);
            }
        }
    }
    ensure!(
        defects == 0,
        "DEFECT chain timing output changed across repeats count={defects}"
    );
    Ok(())
}

fn prefill_scores(
    case: &str,
    source: &SafeTensors<'_>,
    trace: &Trace,
    logits: &[f32],
    bounds: &Bounds,
) -> Result<usize> {
    let media = case.starts_with("image_") || case.starts_with("video_");
    let mut defects = 0;
    if !media {
        defects += score(
            &format!("{case}/embed"),
            &tail(trace, "embed", 8 * 2048)?,
            &floats(source, "embed_last8")?,
            [0.0, 0.0, 0.0, 1.0],
        )?;
    }
    for (&layer, &bound) in &bounds.layers {
        let key = format!("layer_{layer:02}_hidden_last8");
        ensure!(
            source.tensor(&key)?.shape() == [8, 2048],
            "reference {key} layout"
        );
        defects += score(
            &format!("{case}/{key}"),
            &tail(trace, &format!("layer.{layer}"), 8 * 2048)?,
            &floats(source, &key)?,
            bound,
        )?;
    }
    let part = trace.prefill_parts()? - 1;
    let norm = trace.values(&format!("prefill.{part}.norm"))?;
    defects += score(
        &format!("{case}/norm"),
        &norm[norm.len() - 2048..],
        &floats(source, "final_norm_last")?,
        bounds.norm,
    )?;
    defects += score(
        &format!("{case}/logits"),
        logits,
        &floats(source, "prefill_last_logits_bf16")?,
        bounds.logits,
    )?;
    Ok(defects)
}

fn teacher_force(
    engine: &mut Engine,
    source: &SafeTensors<'_>,
    case: &str,
    model: &Path,
    bounds: &Bounds,
) -> Result<(usize, Vec<Vec<f32>>)> {
    let ids = integers(source, "greedy_token_ids")?;
    let top = integers(source, "step_top2_ids")?;
    let expected = floats(source, "step_top2_logits")?;
    let margins = floats(source, "step_top2_margin")?;
    let steps = teacher_steps(
        source,
        case,
        engine
            .model_spec
            .expert_format
            .context("native teacher expert_format missing")?,
    )?;
    let mut outputs = Vec::with_capacity(steps);
    let mut errors = Vec::with_capacity(steps * 2);
    let (mut defects, mut scored) = (0, 0);
    let mut predictions = Vec::new();
    for step in 0..steps {
        let values = engine.decode_logits()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite decode {case}/{step}"
        );
        let prediction = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("empty logits")?
            .0 as u32;
        predictions.push(prediction);
        let error = (0..2)
            .map(|j| {
                let error = (values[top[2 * step + j] as usize] - expected[2 * step + j]).abs();
                errors.push(error);
                error
            })
            .fold(0.0f32, f32::max);
        // Either of the two logits may move by the measured absolute-error bound.
        let id_margin = bounds.decode[0] + bounds.decode[0];
        let above_margin = f64::from(margins[step]) > id_margin;
        let id_ok = !above_margin || prediction == ids[step];
        scored += usize::from(above_margin);
        let pass = f64::from(error) <= bounds.decode[0] && id_ok;
        defects += usize::from(!pass);
        println!(
            "{} minicpm_teacher {case} step={step} top2_max={error} reference_margin={} id_scored={} predicted={prediction} reference={}",
            if pass { "PASS" } else { "DEFECT" },
            margins[step],
            above_margin,
            ids[step]
        );
        outputs.push(values);
        if step + 1 < steps {
            engine.set_token(ids[step])?;
            let range = super::super::super::profile::Range::new(true, c"minicpm_e2e_eager");
            engine.decode(false, None)?;
            drop(range);
        }
    }
    errors.sort_unstable_by(f32::total_cmp);
    let median = errors[errors.len() / 2];
    defects += usize::from(f64::from(median) > bounds.decode[1]);
    println!(
        "{} minicpm_teacher_summary {case} steps={steps} ids_scored={scored} median_top2_error={median}",
        if f64::from(median) <= bounds.decode[1] {
            "PASS"
        } else {
            "DEFECT"
        }
    );
    if case.starts_with("image_") || case.starts_with("video_") {
        println!(
            "minicpm_decoded {case} {:?}",
            crate::tokenizer::decode(model, &predictions)?
        );
    }
    Ok((defects, outputs))
}

fn replay(
    engine: &mut Engine,
    source: &SafeTensors<'_>,
    saved: &super::super::super::state::StateCheckpoint,
    logits: &[Vec<f32>],
    start: usize,
    pending: u32,
) -> Result<usize> {
    engine
        .moe_prefill
        .as_ref()
        .context("prefill missing")?
        .reseed_request(engine.moe_slots.as_mut().context("slots missing")?)?;
    saved.restore(
        &mut engine.state,
        &mut engine.work.token,
        &engine.device.stream,
    )?;
    engine.position = start;
    engine.committed_prefix_ids.truncate(start);
    engine.pending_token = Some(pending);
    engine
        .device
        .stream
        .memcpy_htod(&[start as i32], &mut engine.work.position)?;
    engine
        .device
        .stream
        .memcpy_htod(&[start as i32 + 1], &mut engine.work.length)?;
    let ids = integers(source, "greedy_token_ids")?;
    let mut defects = 0;
    for step in 1..logits.len() {
        engine.set_token(ids[step - 1])?;
        let range = super::super::super::profile::Range::new(true, c"minicpm_e2e_graph");
        engine.decode(true, None)?;
        drop(range);
        let got = engine.decode_logits()?;
        ensure!(
            got.iter().all(|v| v.is_finite()),
            "nonfinite graph replay step={step}"
        );
        let mismatches = got
            .iter()
            .zip(&logits[step])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        defects += usize::from(mismatches != 0);
        println!(
            "{} minicpm_graph step={step} logits_bit_mismatches={mismatches}",
            if mismatches == 0 { "PASS" } else { "DEFECT" }
        );
    }
    Ok(defects)
}

fn media_control(
    engine: &mut Engine,
    input: &mut Input,
    fixtures: &Path,
    baseline: &[f32],
    bound: [f64; 2],
) -> Result<usize> {
    let bytes = std::fs::read(fixtures.join("control_boat_at_hockey_size_16x.safetensors"))?;
    let source = SafeTensors::deserialize(&bytes)?;
    let control = Input::read(&source, "image_control")?;
    ensure!(
        control.ids == input.ids
            && control.modalities == input.modalities
            && control.grids == input.grids,
        "media control ids/modalities/grid differ"
    );
    input.pixels = control.pixels;
    engine.reset()?;
    engine.prefill_multimodal(&input.packed(), None)?;
    let changed = engine.decode_logits()?;
    let m = metrics(&changed, baseline)?;
    let calls = engine
        .qwen
        .as_ref()
        .context("runtime missing")?
        .vision_encode_calls;
    let pass = calls == 1 && m[0] >= bound[0] && m[3] <= bound[1];
    println!(
        "{} minicpm_media_control same_ids=true same_grid=true changed_pixels=boat_reference encoder_calls={calls} max={} cosine={} bound={bound:?}",
        if pass { "PASS" } else { "DEFECT" },
        m[0],
        m[3]
    );
    Ok(usize::from(!pass))
}

fn video_caption(
    engine: &mut Engine,
    input: &Input,
    case: &str,
    model: &Path,
    steps: usize,
) -> Result<Vec<u32>> {
    engine.reset()?;
    let range = super::super::super::profile::Range::new(true, c"minicpm_video_prefill");
    let mut token = engine.prefill_multimodal(&input.packed(), None)?;
    drop(range);
    let mut ids = Vec::with_capacity(steps);
    for step in 0..steps {
        ids.push(token);
        if engine.model_spec.eos.contains(&token) || step + 1 == steps {
            break;
        }
        token = engine.decode(false, None)?;
    }
    println!(
        "minicpm_video_caption {case} ids={ids:?} text={:?}",
        crate::tokenizer::decode(model, &ids)?
    );
    Ok(ids)
}

fn video_splice_control(
    engine: &mut Engine,
    input: &Input,
    source: &SafeTensors<'_>,
    bounds: &Bounds,
) -> Result<()> {
    engine.reset()?;
    engine.prepare_multimodal(&input.packed(), None)?;
    engine.images = None;
    engine.prefill(&input.ids, None)?;
    let metrics = metrics(
        &engine.decode_logits()?,
        &floats(source, "prefill_last_logits_bf16")?,
    )?;
    let detected = metrics[..3]
        .iter()
        .zip(&bounds.logits[..3])
        .any(|(value, bound)| value > bound)
        || metrics[3] < bounds.logits[3];
    ensure!(
        detected,
        "omitted video splice was not detected: {metrics:?}"
    );
    println!(
        "PASS minicpm_video_splice_control omitted=true detected=true metrics={metrics:?} bound={:?}",
        bounds.logits
    );
    Ok(())
}

#[test]
fn cosine_budget_covers_two_opposite_reference_errors() -> Result<()> {
    let reference = [1.0, 0.0];
    let a = [3.0, 4.0];
    let b = [3.0, -4.0];
    let floor = metrics(&reference, &a)?;
    let cell = serde_json::json!({"max_abs":floor[0], "p99_abs":floor[1],
        "mean_abs":floor[2], "cosine_sim":floor[3]});
    let bound = measured_bounds(&[&cell])?;
    let actual = metrics(&a, &b)?[3];
    let linear = 1.0 - 2.0 * (1.0 - floor[3]);
    ensure!(
        actual < linear,
        "counterexample did not reject linear cosine scaling"
    );
    ensure!(
        actual >= bound[3],
        "two-arm angular triangle bound rejected a valid pair: {actual} < {}",
        bound[3]
    );
    Ok(())
}

#[test]
fn reference_floors_are_selected_per_input() -> Result<()> {
    let directory = std::env::temp_dir().join(format!(
        "flyingfish-layer0-floor-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&directory)?;
    let result = (|| -> Result<()> {
        let zero = serde_json::json!({"max_abs":0.0, "p99_abs":0.0,
            "mean_abs":0.0, "cosine_sim":1.0});
        let layers = [0, 3, 6, 19, 39]
            .map(|layer| (format!("layer_{layer:02}_hidden_last8"), zero.clone()))
            .into_iter()
            .collect::<serde_json::Map<_, _>>();
        let comparison = serde_json::json!({"layers":layers, "norm":zero,
            "logits":zero, "teacher_top2":zero});
        let comparisons = (0..4)
            .map(|index| (index.to_string(), comparison.clone()))
            .collect::<serde_json::Map<_, _>>();
        let cases = [
            ("text_32", 0.125),
            ("image_boat_16x", 0.25),
            ("image_hockey_16x", 0.375),
            ("video_football_16x", 0.5),
        ]
        .map(|(case, error)| {
            (
                case.to_owned(),
                serde_json::json!({"comparisons":comparisons,
                    "fp32_layer0":{"max_abs":error, "p99_abs":error/2.0,
                        "mean_abs":error/4.0, "cosine_sim":1.0}}),
            )
        })
        .into_iter()
        .collect::<serde_json::Map<_, _>>();
        let vision_error = serde_json::json!({"max_abs":0.0625, "p99_abs":0.03125,
            "mean_abs":0.015625, "cosine_sim":1.0});
        let vision_layers = [0, 3, 6, 19, 39]
            .map(|layer| {
                (
                    format!("layer_{layer:02}_hidden_last8"),
                    vision_error.clone(),
                )
            })
            .into_iter()
            .collect::<serde_json::Map<_, _>>();
        let vision = serde_json::json!({"layers":vision_layers, "norm":vision_error,
            "logits":vision_error, "teacher_top2":vision_error});
        let cross_error = serde_json::json!({"max_abs":0.03125, "p99_abs":0.015625,
            "mean_abs":0.0078125, "cosine_sim":1.0});
        let cross_layers = [0, 3, 6, 19, 39]
            .map(|layer| {
                (
                    format!("layer_{layer:02}_hidden_last8"),
                    cross_error.clone(),
                )
            })
            .into_iter()
            .collect::<serde_json::Map<_, _>>();
        let cross = serde_json::json!({"layers":cross_layers, "norm":cross_error,
            "logits":cross_error, "teacher_top2":cross_error});
        let batch_shape = cases
            .iter()
            .map(|(case, floor)| {
                let mut shape = comparison.clone();
                shape["logits"] = floor["fp32_layer0"].clone();
                (
                    case.clone(),
                    serde_json::json!({"sm89":shape,"sm86":comparison}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        std::fs::write(
            directory.join("noise_floor_r2.json"),
            serde_json::to_vec(
                &serde_json::json!({"per_case_reference_floor":{"cases":cases},
                "official_vision_implementation_propagation":{"cases":{
                    "image_boat_16x":vision,"image_hockey_16x":vision,"video_football_16x":vision}},
                "official_expert_batch_shape":{"validated":true,"cases":batch_shape},
                "official_cross_gpu_streamed":{"sm89_primary_bitwise":true,"cases":{
                    "text_32":cross,"image_boat_16x":cross,"image_hockey_16x":cross,"video_football_16x":cross}}}),
            )?,
        )?;
        let media = serde_json::json!({"max_abs":1.0, "cosine_sim":0.0});
        let video_media = serde_json::json!({"max_abs":2.0, "cosine_sim":0.5});
        let video_cross = serde_json::json!({"max_abs":4.0, "cosine_sim":-0.5});
        std::fs::write(
            directory.join("fixtures_manifest.json"),
            serde_json::to_vec(&serde_json::json!({"image":{"media_reach_controls":{
                "same_grid_400x267_hockey_vs_boat_logits_diff":media,
                "hockey_16x_vs_boat_16x_logits_diff":media}},"video":{"media_reach_controls":{
                "same_grid_1280x720_football_vs_boat_pan_logits_diff":video_media,
                "football_16x_vs_sample_demo_1_16x_logits_diff":video_cross}}}))?,
        )?;
        for (case, expected) in [
            ("text_32", 0.25),
            ("image_boat_16x", 0.5),
            ("image_hockey_16x", 0.75),
            ("video_football_16x", 1.0),
        ] {
            let bounds = Bounds::read(
                &directory.join("fixtures"),
                case,
                crate::model_spec::ExpertFormat::Bf16,
            )?;
            ensure!(
                bounds.layers[&0] == [expected, expected / 2.0, expected / 4.0, 1.0],
                "wrong layer0 precision floor for {case}: {:?}",
                bounds.layers[&0]
            );
            let other = if !case.starts_with("text_") {
                [0.125, 0.0625, 0.03125, 1.0]
            } else {
                [0.0625, 0.03125, 0.015625, 1.0]
            };
            for layer in [3, 6, 19, 39] {
                assert_eq!(bounds.layers[&layer], other);
            }
            assert_eq!(bounds.norm, other);
            assert_eq!(
                bounds.logits,
                [expected, expected / 2.0, expected / 4.0, 1.0]
            );
            assert_eq!(bounds.decode, other);
        }
        let manifest_path = directory.join("fixtures_manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        manifest["image"]["media_reach_controls"]
            .as_object_mut()
            .context("image media controls missing")?
            .remove("hockey_16x_vs_boat_16x_logits_diff")
            .context("two-image control missing")?;
        std::fs::write(manifest_path, serde_json::to_vec(&manifest)?)?;
        let quant_cells = [3, 39]
            .into_iter()
            .flat_map(|layer| {
                ["attn", "o"].into_iter().flat_map(move |role| {
                    ["scale", "codes", "fake"]
                        .into_iter()
                        .map(move |kind| format!("fixed_layer_{layer:02}_{role}_{kind}"))
                })
            })
            .map(|key| (key, zero.clone()))
            .collect::<serde_json::Map<_, _>>();
        let fp8_cases = ["text_32", "text_1k", "image_hockey_16x", "text_4k",
            "video_football_16x", "video_control_clip2_16x", "video_control_same_grid_16x"].into_iter().enumerate().map(|(index, case)| {
            let mut tensors = layers.clone();
            for key in ["final_norm_last", "prefill_last_logits_bf16", "teacher_top2", "layer0_fp32_raw_last8"] {
                tensors.insert(key.to_owned(), zero.clone());
            }
            let error = (index + 1) as f64 / 8.0;
            let aggregate = serde_json::json!({"max_abs":error,"p99_abs":error,
                "mean_abs":error,"cosine_sim":1.0});
            tensors.insert("teacher_top2".to_owned(), aggregate.clone());
            let arm = serde_json::json!({"applicable":true, "tensors":tensors});
            let teacher = (0..32).map(|step| {
                serde_json::json!({"top2":if step == 0 {zero.clone()} else {aggregate.clone()}})
            }).collect::<Vec<_>>();
            let image = case.starts_with("image_") || case.starts_with("video_");
            let mut teacher_arms = serde_json::json!({
                "expert_implementation_order":teacher,"layer0_mixed_precision":teacher,
                "same_input_cross_gpu":teacher});
            if image {
                teacher_arms["official_vision_implementation_propagation"] = serde_json::json!(teacher);
            }
            (case.to_owned(), serde_json::json!({"components":{
                "expert_implementation_order":arm,"layer0_mixed_precision":arm,
                "same_input_cross_gpu":arm,
                "official_vision_implementation_propagation":if image {arm.clone()} else {serde_json::json!({"applicable":false})},
                "teacher_forced_per_step":{"steps":32,"comparisons":teacher_arms},
                "official_fp8_fake_quant_cross_gpu":{"form":"P2_official_loader_token_fake_quant","cells":quant_cells}
            }}))
        }).collect::<serde_json::Map<_, _>>();
        let mut fp8 = serde_json::json!({"schema":1,"format":"fp8_w8a16",
            "state":"frozen_before_candidate_measurement","declaration":{"candidate_results_seen":false},"cases":fp8_cases});
        let path = directory.join("fp8-floor.json");
        std::fs::write(&path, serde_json::to_vec(&fp8)?)?;
        for (index, case) in [
            "text_32",
            "text_1k",
            "image_hockey_16x",
            "text_4k",
            "video_football_16x",
            "video_control_clip2_16x",
            "video_control_same_grid_16x",
        ]
        .into_iter()
        .enumerate()
        {
            let bound = Bounds::read(
                &directory.join("fixtures"),
                case,
                crate::model_spec::ExpertFormat::Fp8W8a16,
            )?;
            let expected = (index + 1) as f64 / 4.0;
            assert_eq!(bound.decode, [expected, expected, expected, 1.0]);
            if case.starts_with("video_") {
                assert_eq!(bound.media, [1.0, 0.75]);
                assert_eq!(bound.cross_media, [2.0, 0.25]);
            } else {
                assert_eq!(bound.media, [0.5, 0.5]);
                assert_eq!(bound.cross_media, bound.media);
            }
        }
        fp8["cases"]["text_32"]["components"]["teacher_forced_per_step"]["comparisons"]["same_input_cross_gpu"] =
            serde_json::json!([]);
        std::fs::write(&path, serde_json::to_vec(&fp8)?)?;
        assert!(
            Bounds::read(
                &directory.join("fixtures"),
                "text_32",
                crate::model_spec::ExpertFormat::Fp8W8a16
            )
            .is_err()
        );
        assert!(
            Bounds::read(
                &directory.join("fixtures"),
                "missing_case",
                crate::model_spec::ExpertFormat::Fp8W8a16
            )
            .is_err()
        );
        Ok(())
    })();
    std::fs::remove_dir_all(&directory)?;
    result
}

#[test]
#[ignore = "requires CUDA, FF_MODEL, FF_MINICPM_REFERENCE_DIR, FF_VIDEO_REF_DIR and FLYINGFISH_RUNTIME_DIR"]
fn minicpm_native_checkpoint_e2e_gpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("reference directory missing")?,
    );
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime directory missing")?,
    );
    let settings = Settings {
        runtime_dir: Some(runtime),
        prefix_cache: false,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let spec = crate::model_spec::ModelSpec::read(&model)?;
    let format = spec
        .expert_format
        .context("native checkpoint expert_format missing")?;
    let names = native_cases(format);
    let cases = names
        .iter()
        .copied()
        .map(|case| Ok((case, Bounds::read(&fixtures, case, format)?)))
        .collect::<Result<Vec<_>>>()?;
    let cross_media = cases
        .first()
        .context("reference cases missing")?
        .1
        .cross_media;
    println!("minicpm_reference_bounds {cases:?}");
    check_fixtures(&fixtures, &model, names)?;
    Engine::calibrate(&model, 0, 4352, None, settings.clone())?;
    let mut engine = Engine::load(&model, 0, 4352, None, settings.clone())?;
    ensure!(
        engine.model_spec.hidden == 2048
            && engine.model_spec.vocab == 248144
            && engine.model_spec.layers.len() == 40,
        "published MiniCPM geometry required"
    );
    let mut defects = 0;
    if format == crate::model_spec::ExpertFormat::Fp8W8a16 {
        let bytes = std::fs::read(fixtures.join("text_32_layer00_moe.safetensors"))?;
        defects += native_layer0(&mut engine, &SafeTensors::deserialize(&bytes)?)?;
    }
    let mut media_logits = Vec::new();
    let mut videos = Vec::new();
    let mut splice_controls = 0;
    let mut image_controls = 0;
    for (case, bounds) in &cases {
        let case = *case;
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let video = case.starts_with("video_");
        let media = case.starts_with("image_") || video;
        let mut input = if video {
            Input::video(&source, &model, case)?
        } else {
            Input::read(&source, case)?
        };
        engine.reset()?;
        let mut trace = Trace::default();
        if media {
            engine.prepare_multimodal(&input.packed(), Some(&mut trace))?;
        }
        engine.prefill_tail(&input.ids, &mut trace, 8)?;
        let first_logits = engine.decode_logits()?;
        defects += prefill_scores(case, &source, &trace, &first_logits, bounds)?;
        let calls = engine
            .qwen
            .as_ref()
            .context("runtime missing")?
            .vision_encode_calls;
        ensure!(
            calls == usize::from(media),
            "encoder call count {case}={calls}"
        );
        println!("PASS minicpm_media_calls {case} encoder_calls={calls}");
        let mut checkpoint = super::super::super::state::StateCheckpoint::new(
            &engine.state,
            &engine.model_spec,
            64,
            &engine.device.stream,
        )?;
        checkpoint.save(
            &engine.state,
            &engine.work.token,
            engine.position,
            &engine.device.stream,
            None,
        )?;
        let pending = engine.token()?;
        let (errors, logits) = teacher_force(&mut engine, &source, case, &model, bounds)?;
        defects += errors;
        defects += replay(
            &mut engine,
            &source,
            &checkpoint,
            &logits,
            input.ids.len(),
            pending,
        )?;
        drop(checkpoint);
        let mut stats = Trace::default();
        engine.record_test_stats(&mut stats)?;
        println!(
            "minicpm_modes {case} resident_hits={} cpu_misses={} disk_experts={} graph_segments={}",
            stats.values("stats.gpu_resident_hits")?[0],
            stats.values("stats.cpu_worker_misses")?[0],
            stats
                .values("residency.tier")?
                .iter()
                .filter(|&&tier| tier == 2.0)
                .count(),
            engine.graphs.len()
        );
        if case.starts_with("image_") {
            media_logits.push(first_logits.clone());
        }
        if video {
            let ids = video_caption(
                &mut engine,
                &input,
                case,
                &model,
                integers(&source, "greedy_token_ids")?.len(),
            )?;
            videos.push((case, first_logits.clone(), ids));
            if case == "video_football_16x" {
                video_splice_control(&mut engine, &input, &source, bounds)?;
                splice_controls += 1;
            }
        }
        if case == "image_hockey_16x" {
            defects += media_control(
                &mut engine,
                &mut input,
                &fixtures,
                &first_logits,
                bounds.media,
            )?;
            image_controls += 1;
        }
    }
    if format == crate::model_spec::ExpertFormat::Bf16 {
        let media = metrics(&media_logits[0], &media_logits[1])?;
        defects += usize::from(media[0] < cross_media[0] || media[3] > cross_media[1]);
        println!(
            "minicpm_two_media max={} cosine={} defects={defects}",
            media[0], media[3]
        );
    }
    let video_count = videos.len();
    ensure!(video_count == 3, "native gate requires three video cases");
    let video_bounds = &cases
        .iter()
        .find(|(case, _)| *case == "video_football_16x")
        .context("video bounds missing")?
        .1;
    for (index, bound) in [(1, video_bounds.cross_media), (2, video_bounds.media)] {
        let metrics = metrics(&videos[0].1, &videos[index].1)?;
        let pass =
            metrics[0] >= bound[0] && metrics[3] <= bound[1] && videos[0].2 != videos[index].2;
        defects += usize::from(!pass);
        println!(
            "{} minicpm_two_videos pair={} max={} cosine={} distinct_ids={} bound={bound:?}",
            if pass { "PASS" } else { "DEFECT" },
            videos[index].0,
            metrics[0],
            metrics[3],
            videos[0].2 != videos[index].2
        );
    }
    for (case, _, expected) in videos.iter().rev() {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let input = Input::video(&source, &model, case)?;
        let ids = video_caption(&mut engine, &input, case, &model, expected.len())?;
        defects += usize::from(ids != *expected);
        println!(
            "{} minicpm_video_request_order {case} reverse_order=true same_ids={}",
            if ids == *expected { "PASS" } else { "DEFECT" },
            ids == *expected
        );
    }
    println!("minicpm_native_end defects={defects}");
    ensure!(defects == 0, "MiniCPM end-to-end DEFECT count={defects}");
    println!(
        "PASS minicpm_native_checkpoint format={format:?} cases={} bitwise_graphs=true media_controls={image_controls} video_cases={video_count} video_splice_controls={splice_controls}",
        names.len(),
    );
    Ok(())
}
