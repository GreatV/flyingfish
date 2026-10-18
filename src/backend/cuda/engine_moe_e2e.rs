//! Published-checkpoint text, media and teacher-forced decode acceptance gate.
use super::*;
use crate::vision_input::{ImageGrid, MultimodalInput};
use safetensors::{Dtype, SafeTensors};

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
    let pass = m[0] <= bound[0] && m[1] <= bound[1] && m[2] <= bound[2] && m[3] >= bound[3];
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
    fn read(source: &SafeTensors<'_>, media: bool) -> Result<Self> {
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
        if media {
            input.modalities = integers(source, "mm_token_type_ids")?
                .into_iter()
                .map(|v| Ok(u8::try_from(v)?))
                .collect::<Result<_>>()?;
            input.pixels = floats(source, "pixel_values")?;
            let sizes = integers(source, "target_sizes")?;
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
        }
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

fn check_fixtures(fixtures: &Path, model: &crate::model_spec::ModelSpec) -> Result<()> {
    for case in [
        "text_32",
        "text_1k",
        "text_4k",
        "image_hockey_16x",
        "image_boat_16x",
    ] {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let input = Input::read(&source, case.starts_with("image_"))?;
        ensure!(
            input.ids.len() >= 8
                && input.ids.len() <= 4096
                && input.ids.iter().all(|&id| id < model.vocab),
            "fixture prompt extent/vocabulary {case}"
        );
        if !input.grids.is_empty() {
            input.packed().validate(model, 4352)?;
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

fn prefill_scores(
    case: &str,
    source: &SafeTensors<'_>,
    trace: &Trace,
    logits: &[f32],
) -> Result<usize> {
    let image = case.starts_with("image_");
    let mut defects = 0;
    if !image {
        defects += score(
            &format!("{case}/embed"),
            &tail(trace, "embed", 8 * 2048)?,
            &floats(source, "embed_last8")?,
            [0.0, 0.0, 0.0, 1.0 - 1e-12],
        )?;
    }
    for (layer, bound) in [
        (0, [0.015625, 0.002, 0.0003, 0.9999]),
        (3, [0.0234375, 0.0035, 0.0006, 0.9998]),
        (6, [0.03125, 0.004, 0.0008, 0.9998]),
        (19, [0.09375, 0.045, 0.009, 0.996]),
        (39, [0.4375, 0.1875, 0.045, 0.991]),
    ] {
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
        if image {
            [0.75, 0.5, 0.15, 0.998]
        } else {
            [0.625, 0.4375, 0.130, 0.9985]
        },
    )?;
    defects += score(
        &format!("{case}/logits"),
        logits,
        &floats(source, "prefill_last_logits_bf16")?,
        if image {
            [1.5, 0.45, 0.125, 0.9985]
        } else {
            [1.25, 0.375, 0.105, 0.999]
        },
    )?;
    Ok(defects)
}

fn teacher_force(
    engine: &mut Engine,
    source: &SafeTensors<'_>,
    case: &str,
    model: &Path,
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
        let id_ok = margins[step] <= 1.5 || prediction == ids[step];
        scored += usize::from(margins[step] > 1.5);
        let pass = error
            <= if case.starts_with("image_") {
                1.75
            } else {
                1.5
            }
            && id_ok;
        defects += usize::from(!pass);
        println!(
            "{} minicpm_teacher {case} step={step} top2_max={error} reference_margin={} id_scored={} predicted={prediction} reference={}",
            if pass { "PASS" } else { "DEFECT" },
            margins[step],
            margins[step] > 1.5,
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
    defects += usize::from(median > 0.375);
    println!(
        "{} minicpm_teacher_summary {case} steps={steps} ids_scored={scored} median_top2_error={median}",
        if median <= 0.375 { "PASS" } else { "DEFECT" }
    );
    if case.starts_with("image_") {
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
    model: &Path,
    fixtures: &Path,
    baseline: &[f32],
) -> Result<usize> {
    use crate::serve::{
        detok::Shared,
        image::{Processor, RgbImage, load_image},
    };
    let bytes = std::fs::read(
        fixtures
            .parent()
            .context("fixture parent missing")?
            .join("inputs/boat.jpg"),
    )?;
    let boat = load_image(&format!("data:image/jpeg;base64,{}", base64::encode(bytes)))?;
    let mut resized = RgbImage {
        width: 400,
        height: 267,
        pixels: vec![0; 400 * 267 * 3],
    };
    for y in 0..267 {
        for x in 0..400 {
            let source = ((y * boat.height / 267) * boat.width + x * boat.width / 400) * 3;
            resized.pixels[(y * 400 + x) * 3..][..3].copy_from_slice(&boat.pixels[source..][..3]);
        }
    }
    let tokenizer = Shared::load(model)?;
    let packed = Processor::load(model, Some(&tokenizer))?
        .context("processor missing")?
        .prepare_rgb(&resized)?;
    ensure!(
        packed.grids == input.grids && packed.pixel_values.len() == input.pixels.len(),
        "media control grid differs"
    );
    input.pixels = packed.pixel_values;
    engine.reset()?;
    engine.prefill_multimodal(&input.packed(), None)?;
    let changed = engine.decode_logits()?;
    let m = metrics(&changed, baseline)?;
    let calls = engine
        .qwen
        .as_ref()
        .context("runtime missing")?
        .vision_encode_calls;
    let pass = calls == 1 && m[0] >= 5.0 && m[3] <= 0.93;
    println!(
        "{} minicpm_media_control same_ids=true same_grid=true changed_pixels=boat_nearest_resize encoder_calls={calls} max={} cosine={}",
        if pass { "PASS" } else { "DEFECT" },
        m[0],
        m[3]
    );
    Ok(usize::from(!pass))
}

#[test]
#[ignore = "requires CUDA, FF_MODEL, FF_MINICPM_REFERENCE_DIR and FLYINGFISH_RUNTIME_DIR"]
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
    check_fixtures(&fixtures, &crate::model_spec::ModelSpec::read(&model)?)?;
    Engine::calibrate(&model, 0, 4352, None, settings.clone())?;
    let mut engine = Engine::load(&model, 0, 4352, None, settings)?;
    ensure!(
        engine.model_spec.hidden == 2048
            && engine.model_spec.vocab == 248144
            && engine.model_spec.layers.len() == 40,
        "published MiniCPM geometry required"
    );
    let mut defects = 0;
    let mut media_logits = Vec::new();
    for case in [
        "text_32",
        "text_1k",
        "text_4k",
        "image_hockey_16x",
        "image_boat_16x",
    ] {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        let media = case.starts_with("image_");
        let mut input = Input::read(&source, media)?;
        engine.reset()?;
        let mut trace = Trace::default();
        if media {
            engine.prepare_multimodal(&input.packed(), Some(&mut trace))?;
        }
        engine.prefill_tail(&input.ids, &mut trace, 8)?;
        let first_logits = engine.decode_logits()?;
        defects += prefill_scores(case, &source, &trace, &first_logits)?;
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
        let (errors, logits) = teacher_force(&mut engine, &source, case, &model)?;
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
        if media {
            media_logits.push(first_logits.clone());
        }
        if case == "image_hockey_16x" {
            defects += media_control(&mut engine, &mut input, &model, &fixtures, &first_logits)?;
        }
    }
    let media = metrics(&media_logits[0], &media_logits[1])?;
    defects += usize::from(media[0] == 0.0);
    println!(
        "minicpm_two_media max={} cosine={} defects={defects}",
        media[0], media[3]
    );
    ensure!(defects == 0, "MiniCPM end-to-end DEFECT count={defects}");
    println!(
        "PASS minicpm_native_checkpoint cases=5 teacher_steps=256 bitwise_graphs=true media_control=boat_pixels"
    );
    Ok(())
}
