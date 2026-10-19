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
    fn read(fixtures: &Path, case: &str) -> Result<Self> {
        let directory = fixtures.parent().context("reference parent missing")?;
        let noise: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("noise_floor_r2.json"))?)?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("fixtures_manifest.json"))?)?;
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
        let video = case.starts_with("video_");
        let media = &manifest[if video { "video" } else { "image" }]["media_reach_controls"];
        let (same_grid, cross_media) = if video {
            (
                "same_grid_1280x720_football_vs_boat_pan_logits_diff",
                "football_16x_vs_sample_demo_1_16x_logits_diff",
            )
        } else {
            (
                "same_grid_400x267_hockey_vs_boat_logits_diff",
                "hockey_16x_vs_boat_16x_logits_diff",
            )
        };
        Ok(Self {
            layers,
            norm: measured_bounds(&cells("norm"))?,
            logits: measured_bounds(&cells("logits"))?,
            decode: measured_bounds(&cells("teacher_top2"))?,
            media: media_boundary(&media[same_grid])?,
            cross_media: media_boundary(&media[cross_media])?,
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
        let (clip, processor_case) = match case {
            "video_football_16x" => ("football_4s", "football_4s_16x_s1"),
            "video_control_clip2_16x" => ("sample_demo_1", "sample_demo_1_16x_s1"),
            "video_control_same_grid_16x" => ("boat_pan_4s", "boat_pan_4s_16x_s1"),
            _ => anyhow::bail!("unknown video fixture case={case}"),
        };
        let floor: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join("preprocess_video_noise_floor.json"),
        )?)?;
        let measured = &floor["cases"][processor_case];
        let tolerance = measured["tv_vs_pil_diff"]["max_abs"]
            .as_f64()
            .context("video pixel floor missing")? as f32
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

fn check_fixtures(fixtures: &Path, model_dir: &Path) -> Result<()> {
    let model = crate::model_spec::ModelSpec::read(model_dir)?;
    for case in [
        "text_32",
        "text_1k",
        "text_4k",
        "image_hockey_16x",
        "image_boat_16x",
        "video_football_16x",
        "video_control_clip2_16x",
        "video_control_same_grid_16x",
    ] {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        Bounds::read(fixtures, case)?;
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

#[test]
#[ignore = "CPU-only native fixtures; requires FF_MODEL, FF_MINICPM_REFERENCE_DIR and FF_VIDEO_REF_DIR"]
fn minicpm_native_checkpoint_fixtures_cpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("reference directory missing")?,
    );
    check_fixtures(&fixtures, &model)?;
    println!(
        "PASS minicpm_native_fixture_preflight cases=8 video_decode=ffmpeg canvas=true floors_complete=true"
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
    let steps = if case.starts_with("image_") { 32 } else { 64 };
    ensure!(
        ids.len() >= steps
            && top.len() == steps * 2
            && expected.len() == top.len()
            && margins.len() == steps,
        "teacher-forced fixture extents {case}"
    );
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
    let mut token = engine.prefill_multimodal(&input.packed(), None)?;
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
        std::fs::write(
            directory.join("fixtures_manifest.json"),
            serde_json::to_vec(&serde_json::json!({"image":{"media_reach_controls":{
                "same_grid_400x267_hockey_vs_boat_logits_diff":media,
                "hockey_16x_vs_boat_16x_logits_diff":media}},"video":{"media_reach_controls":{
                "same_grid_1280x720_football_vs_boat_pan_logits_diff":media,
                "football_16x_vs_sample_demo_1_16x_logits_diff":media}}}))?,
        )?;
        for (case, expected) in [
            ("text_32", 0.25),
            ("image_boat_16x", 0.5),
            ("image_hockey_16x", 0.75),
            ("video_football_16x", 1.0),
        ] {
            let bounds = Bounds::read(&directory.join("fixtures"), case)?;
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
    let cases = [
        "text_32",
        "text_1k",
        "text_4k",
        "image_hockey_16x",
        "image_boat_16x",
        "video_football_16x",
        "video_control_clip2_16x",
        "video_control_same_grid_16x",
    ]
    .into_iter()
    .map(|case| Ok((case, Bounds::read(&fixtures, case)?)))
    .collect::<Result<Vec<_>>>()?;
    let cross_media = cases
        .first()
        .context("reference cases missing")?
        .1
        .cross_media;
    println!("minicpm_reference_bounds {cases:?}");
    check_fixtures(&fixtures, &model)?;
    Engine::calibrate(&model, 0, 4352, None, settings.clone())?;
    let mut engine = Engine::load(&model, 0, 4352, None, settings.clone())?;
    ensure!(
        engine.model_spec.hidden == 2048
            && engine.model_spec.vocab == 248144
            && engine.model_spec.layers.len() == 40,
        "published MiniCPM geometry required"
    );
    let mut defects = 0;
    let mut media_logits = Vec::new();
    let mut videos = Vec::new();
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
        }
    }
    let media = metrics(&media_logits[0], &media_logits[1])?;
    defects += usize::from(media[0] < cross_media[0] || media[3] > cross_media[1]);
    println!(
        "minicpm_two_media max={} cosine={} defects={defects}",
        media[0], media[3]
    );
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
        "PASS minicpm_native_checkpoint cases=8 teacher_steps=352 bitwise_graphs=true media_control=boat_pixels,video_pixels video_splice_control=true"
    );
    Ok(())
}
