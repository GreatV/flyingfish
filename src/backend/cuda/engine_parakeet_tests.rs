#![cfg(feature = "cuda")]

use super::{
    Device,
    engine_parakeet::{NemotronStream, ParakeetEncoder, StreamChunk, StreamOmit},
};
use crate::{audio::LogMel, nemotron::NemotronSpec};
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::path::{Path, PathBuf};

struct Fixture {
    features: Vec<f32>,
    mask: Vec<u8>,
    subsample: Vec<f32>,
    subsample_mask: Vec<u8>,
    layer0: Vec<f32>,
    control: Vec<f32>,
    encoder: Vec<f32>,
    batch: usize,
    frames: usize,
    rows: usize,
}

fn f32_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::F32, "{name} must be F32");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn read_fixture(path: &Path) -> Result<Fixture> {
    read_case(path, true)
}

fn read_nemotron_fixture(path: &Path) -> Result<Fixture> {
    read_case(path, false)
}

fn read_case(path: &Path, require_control: bool) -> Result<Fixture> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let shape = tensors.tensor("input_features")?.shape().to_vec();
    ensure!(shape.len() == 3 && shape[2] == 128, "mel fixture shape");
    let batch = shape[0];
    let frames = shape[1];
    let rows = tensors.tensor("subsample")?.shape()[1];
    let mask = tensors.tensor("attention_mask")?;
    let subsample_mask = tensors.tensor("subsample_mask")?;
    ensure!(
        mask.dtype() == Dtype::U8 && subsample_mask.dtype() == Dtype::U8,
        "fixture masks must be U8"
    );
    let control = if require_control {
        f32_values(&tensors, "layer0_control")?
    } else {
        ensure!(
            tensors.tensor("layer0_control").is_err(),
            "Nemotron fixture stores a reference-side control tensor"
        );
        Vec::new()
    };
    Ok(Fixture {
        features: f32_values(&tensors, "input_features")?,
        mask: mask.data().to_vec(),
        subsample: f32_values(&tensors, "subsample")?,
        subsample_mask: subsample_mask.data().to_vec(),
        layer0: f32_values(&tensors, "layer0")?,
        control,
        encoder: f32_values(&tensors, "encoder")?,
        batch,
        frames,
        rows,
    })
}

fn difference(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(
        actual.len() == expected.len(),
        "reference element count mismatch"
    );
    ensure!(
        actual.iter().all(|value| value.is_finite()),
        "non-finite candidate"
    );
    let mut maximum = 0.0f64;
    let mut squares = 0.0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        let error = f64::from(a - b).abs();
        maximum = maximum.max(error);
        squares += error * error;
    }
    Ok((maximum, (squares / actual.len() as f64).sqrt()))
}

fn bounds(floors: &Value, case: &str, tap: &str) -> Result<(f64, f64)> {
    Ok((
        floors["cases"][case][tap]["max_abs"]
            .as_f64()
            .with_context(|| format!("missing {case}/{tap} max_abs"))?,
        floors["cases"][case][tap]["rmse"]
            .as_f64()
            .with_context(|| format!("missing {case}/{tap} rmse"))?,
    ))
}

fn match_encoder(
    engine: &ParakeetEncoder,
    fixtures: &Path,
    floors_path: &Path,
    label: &str,
) -> Result<()> {
    let floors: Value = serde_json::from_slice(&std::fs::read(floors_path)?)?;
    for case in ["short", "edges", "production"] {
        let fixture = read_fixture(&fixtures.join(format!("{case}.safetensors")))?;
        for batch in 0..fixture.batch {
            let valid = fixture.mask[batch * fixture.frames..(batch + 1) * fixture.frames]
                .iter()
                .filter(|&&value| value != 0)
                .count();
            let input =
                &fixture.features[batch * fixture.frames * 128..(batch + 1) * fixture.frames * 128];
            let expected_mask =
                &fixture.subsample_mask[batch * fixture.rows..(batch + 1) * fixture.rows];
            let (subsample, mask) = engine.subsample(input, valid)?;
            ensure!(mask == expected_mask, "{case}/{batch} subsample mask");
            let width = fixture.rows * 1024;
            let expected_subsample = &fixture.subsample[batch * width..(batch + 1) * width];
            let (maximum, rmse) = difference(&subsample, expected_subsample)?;
            let (max_bound, rmse_bound) = bounds(&floors, case, "subsample")?;
            ensure!(
                maximum <= max_bound && rmse <= rmse_bound,
                "{case}/{batch} subsample DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} actual={:?} expected={:?}",
                &subsample[..4],
                &expected_subsample[..4]
            );
            println!(
                "{label}_SUBSAMPLE case={case} batch={batch} rows={} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS",
                fixture.rows
            );
            let layer0 = engine.layer(0, expected_subsample, expected_mask)?;
            let expected_layer0 = &fixture.layer0[batch * width..(batch + 1) * width];
            let (maximum, rmse) = difference(&layer0, expected_layer0)?;
            let (max_bound, rmse_bound) = bounds(&floors, case, "layer0")?;
            ensure!(
                maximum <= max_bound && rmse <= rmse_bound,
                "{case}/{batch} layer0 DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound}"
            );
            println!(
                "{label}_LAYER0 case={case} batch={batch} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS"
            );
            let expected_control = &fixture.control[batch * width..(batch + 1) * width];
            let (control_max, control_rmse) = difference(&layer0, expected_control)?;
            ensure!(
                control_max > 10.0 * max_bound && control_rmse > 10.0 * rmse_bound,
                "{case}/{batch} identity-shift control not discriminating: max_abs={control_max} rmse={control_rmse}"
            );
            let (encoded, encoded_mask, encoded_valid) = engine.encode(input, valid)?;
            ensure!(
                encoded_mask == expected_mask
                    && encoded_valid == expected_mask.iter().filter(|&&v| v != 0).count(),
                "{case}/{batch} encoder mask"
            );
            let expected_encoder = &fixture.encoder[batch * width..(batch + 1) * width];
            let (maximum, rmse) = difference(&encoded, expected_encoder)?;
            let (max_bound, rmse_bound) = bounds(&floors, case, "encoder")?;
            ensure!(
                maximum <= max_bound && rmse <= rmse_bound,
                "{case}/{batch} encoder DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound}"
            );
            println!(
                "{label}_ENCODER case={case} batch={batch} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS"
            );
        }
    }
    println!("PASS {label}_ENCODER cells=6 control=identity_relative_shift");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease and external fixtures"]
fn parakeet_subsample_layer_and_encoder_match_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_PARAKEET_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_PARAKEET_MODEL").context("FF_PARAKEET_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_PARAKEET_ENCODER_FIXTURES")
            .context("FF_PARAKEET_ENCODER_FIXTURES missing")?,
    );
    let floors = PathBuf::from(
        std::env::var_os("FF_PARAKEET_ENCODER_FLOORS")
            .context("FF_PARAKEET_ENCODER_FLOORS missing")?,
    );
    let device = Device::new(0)?;
    let engine = ParakeetEncoder::load(&model, &device)?;
    match_encoder(&engine, &fixtures, &floors, "PARAKEET")
}

#[test]
#[ignore = "requires GPU lease and external fixtures"]
fn canary_subsample_layer_and_encoder_match_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_CANARY_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_CANARY_MODEL").context("FF_CANARY_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_CANARY_ENCODER_FIXTURES")
            .context("FF_CANARY_ENCODER_FIXTURES missing")?,
    );
    let floors = PathBuf::from(
        std::env::var_os("FF_CANARY_ENCODER_FLOORS").context("FF_CANARY_ENCODER_FLOORS missing")?,
    );
    let device = Device::new(0)?;
    let engine = ParakeetEncoder::load_canary(&model, &device)?;
    match_encoder(&engine, &fixtures, &floors, "CANARY")
}

fn slice_f32(values: &[f32], batch: usize, rows: usize, width: usize) -> &[f32] {
    let start = batch * rows * width;
    &values[start..start + rows * width]
}

fn record(defects: &mut Vec<String>, line: String) {
    println!("{line}");
    defects.push(line);
}

fn score_cell(
    defects: &mut Vec<String>,
    label: &str,
    case: &str,
    batch: usize,
    actual: &[f32],
    expected: &[f32],
    bounds: (f64, f64),
) -> Result<()> {
    let (max_bound, rmse_bound) = bounds;
    ensure!(
        expected.iter().all(|value| value.is_finite()),
        "{case}/{batch} {label} reference is non-finite"
    );
    let (maximum, rmse) = difference(actual, expected)?;
    if maximum <= max_bound && rmse <= rmse_bound {
        println!(
            "NEMOTRON_{label} case={case} batch={batch} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS"
        );
    } else {
        record(
            defects,
            format!(
                "NEMOTRON_{label} case={case} batch={batch} DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound}"
            ),
        );
    }
    Ok(())
}

fn match_nemotron(engine: &ParakeetEncoder, fixtures: &Path, floors_path: &Path) -> Result<()> {
    let floors: Value = serde_json::from_slice(&std::fs::read(floors_path)?)?;
    let eps_max = floors["eps_min"]["max_abs"]
        .as_f64()
        .context("missing eps_min max_abs")?;
    let eps_rmse = floors["eps_min"]["rmse"]
        .as_f64()
        .context("missing eps_min rmse")?;
    ensure!(
        eps_max == 1e-6 && eps_rmse == 1e-7,
        "eps_min moved: max_abs={eps_max} rmse={eps_rmse}"
    );
    let mut defects = Vec::new();
    let short = read_nemotron_fixture(&fixtures.join("short.safetensors"))?;
    ensure!(
        short.batch == 2 && short.rows == 3,
        "short fixture is not two 3-row clips"
    );
    ParakeetEncoder::reset_encode_counters();
    let (first, first_mask, first_valid) = {
        let valid = short.mask[..short.frames]
            .iter()
            .filter(|&&value| value != 0)
            .count();
        engine.encode(&short.features[..short.frames * 128], valid)?
    };
    ensure!(
        ParakeetEncoder::encode_launches() == 663 && ParakeetEncoder::encode_call_count() == 1,
        "short clip 0 launches={} calls={}",
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::encode_call_count()
    );
    let (second, second_mask, second_valid) = {
        let valid = short.mask[short.frames..short.frames * 2]
            .iter()
            .filter(|&&value| value != 0)
            .count();
        let start = short.frames * 128;
        engine.encode(&short.features[start..start + short.frames * 128], valid)?
    };
    ensure!(
        ParakeetEncoder::encode_launches() == 1326 && ParakeetEncoder::encode_call_count() == 2,
        "short clip 1 launches={} calls={}",
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::encode_call_count()
    );
    ensure!(
        first_mask.len() == 3
            && second_mask.len() == 3
            && first_valid == 3
            && second_valid == 3
            && first != second,
        "two-clip encode did not return distinct 3-row outputs"
    );
    println!("NEMOTRON_CALLS calls=2 rows=3 distinct=true launches_each=663 PASS");
    let mut production_encoded = Vec::new();
    for case in ["short", "edges", "production"] {
        let fixture = read_nemotron_fixture(&fixtures.join(format!("{case}.safetensors")))?;
        let rows = match case {
            "short" => 3,
            "edges" => 9,
            "production" => 376,
            _ => unreachable!("case"),
        };
        let batch_count = match case {
            "short" | "production" => 2,
            "edges" => 1,
            _ => unreachable!("case"),
        };
        ensure!(
            fixture.rows == rows && fixture.batch == batch_count,
            "{case} fixture shape rows={} batch={}",
            fixture.rows,
            fixture.batch
        );
        if case == "production" {
            ensure!(
                fixture.frames == 3001,
                "production mel frames {}",
                fixture.frames
            );
            let sums: Vec<usize> = (0..2)
                .map(|batch| {
                    fixture.mask[batch * fixture.frames..(batch + 1) * fixture.frames]
                        .iter()
                        .filter(|&&value| value != 0)
                        .count()
                })
                .collect();
            ensure!(
                sums == [3000, 2999],
                "production mask sums {sums:?} are not 3000 and 2999"
            );
            let width = rows * 1024;
            ensure!(
                slice_f32(&fixture.encoder, 0, rows, 1024)
                    != slice_f32(&fixture.encoder, 1, rows, 1024)
                    && fixture.encoder.len() == 2 * width,
                "frozen dropped-frame encoder outputs are identical"
            );
        }
        for batch in 0..fixture.batch {
            let valid = fixture.mask[batch * fixture.frames..(batch + 1) * fixture.frames]
                .iter()
                .filter(|&&value| value != 0)
                .count();
            let input =
                &fixture.features[batch * fixture.frames * 128..(batch + 1) * fixture.frames * 128];
            let expected_mask =
                &fixture.subsample_mask[batch * fixture.rows..(batch + 1) * fixture.rows];
            ensure!(
                expected_mask.iter().filter(|&&value| value != 0).count() == rows,
                "{case}/{batch} reference row mask is not fully valid"
            );
            let (subsample, mask) = engine.subsample(input, valid)?;
            ensure!(
                mask == expected_mask,
                "{case}/{batch} subsample mask mismatch"
            );
            let (max_bound, rmse_bound) = bounds(&floors, case, "subsample")?;
            ensure!(
                max_bound >= 1e-6 && rmse_bound >= 1e-7,
                "{case} subsample floor is below eps_min"
            );
            score_cell(
                &mut defects,
                "SUBSAMPLE",
                case,
                batch,
                &subsample,
                slice_f32(&fixture.subsample, batch, rows, 1024),
                (max_bound, rmse_bound),
            )?;
            let expected_subsample = slice_f32(&fixture.subsample, batch, rows, 1024);
            let layer0 = engine.layer(0, expected_subsample, expected_mask)?;
            let (max_bound, rmse_bound) = bounds(&floors, case, "layer0")?;
            ensure!(
                max_bound >= 1e-6 && rmse_bound >= 1e-7,
                "{case} layer0 floor is below eps_min"
            );
            score_cell(
                &mut defects,
                "LAYER0",
                case,
                batch,
                &layer0,
                slice_f32(&fixture.layer0, batch, rows, 1024),
                (max_bound, rmse_bound),
            )?;
            ParakeetEncoder::reset_encode_counters();
            let (encoded, encoded_mask, encoded_valid) = engine.encode(input, valid)?;
            ensure!(
                ParakeetEncoder::encode_launches() == 663
                    && ParakeetEncoder::encode_call_count() == 1,
                "{case}/{batch} launches={} calls={}",
                ParakeetEncoder::encode_launches(),
                ParakeetEncoder::encode_call_count()
            );
            ensure!(
                encoded_mask == expected_mask && encoded_valid == rows,
                "{case}/{batch} encoder mask"
            );
            println!(
                "NEMOTRON_LAUNCHES case={case} batch={batch} launches=663 graphs_counter=issued PASS"
            );
            let (max_bound, rmse_bound) = bounds(&floors, case, "encoder")?;
            ensure!(
                max_bound >= 1e-6 && rmse_bound >= 1e-7,
                "{case} encoder floor is below eps_min"
            );
            score_cell(
                &mut defects,
                "ENCODER",
                case,
                batch,
                &encoded,
                slice_f32(&fixture.encoder, batch, rows, 1024),
                (max_bound, rmse_bound),
            )?;
            if case == "production" {
                production_encoded.push(encoded);
            }
        }
    }
    ensure!(
        production_encoded.len() == 2 && production_encoded[0] != production_encoded[1],
        "candidate dropped-frame encoder outputs are identical"
    );
    let production = read_nemotron_fixture(&fixtures.join("production.safetensors"))?;
    for batch in 0..production.batch {
        let expected_mask =
            &production.subsample_mask[batch * production.rows..(batch + 1) * production.rows];
        let expected_subsample = slice_f32(&production.subsample, batch, production.rows, 1024);
        let wide_layer = engine.layer_right(0, expected_subsample, expected_mask, 1)?;
        let (layer_bound, layer_rmse) = bounds(&floors, "production", "layer0")?;
        let (maximum, rmse) = difference(
            &wide_layer,
            slice_f32(&production.layer0, batch, production.rows, 1024),
        )?;
        ensure!(
            maximum >= 10.0 * layer_bound && rmse >= 10.0 * layer_rmse,
            "production/{batch} wide-chunk layer0 control not discriminating: max_abs={maximum} rmse={rmse} tau_max={layer_bound} tau_rmse={layer_rmse}"
        );
        println!(
            "NEMOTRON_CONTROL case=production batch={batch} tap=layer0 kind=right_chunks_1 max_abs={maximum} tau={layer_bound} rmse={rmse} tau={layer_rmse} PASS"
        );
        let valid = production.mask[batch * production.frames..(batch + 1) * production.frames]
            .iter()
            .filter(|&&value| value != 0)
            .count();
        let input = &production.features
            [batch * production.frames * 128..(batch + 1) * production.frames * 128];
        ParakeetEncoder::reset_encode_counters();
        let (wide_encoded, _, _) = engine.encode_right(input, valid, 1)?;
        ensure!(
            ParakeetEncoder::encode_launches() == 663,
            "production/{batch} wide encode launches={}",
            ParakeetEncoder::encode_launches()
        );
        let (encoder_bound, encoder_rmse) = bounds(&floors, "production", "encoder")?;
        let (maximum, rmse) = difference(
            &wide_encoded,
            slice_f32(&production.encoder, batch, production.rows, 1024),
        )?;
        ensure!(
            maximum >= 10.0 * encoder_bound && rmse >= 10.0 * encoder_rmse,
            "production/{batch} wide-chunk encoder control not discriminating: max_abs={maximum} rmse={rmse} tau_max={encoder_bound} tau_rmse={encoder_rmse}"
        );
        println!(
            "NEMOTRON_CONTROL case=production batch={batch} tap=encoder kind=right_chunks_1 max_abs={maximum} tau={encoder_bound} rmse={rmse} tau={encoder_rmse} PASS"
        );
    }
    ensure!(
        defects.is_empty(),
        "candidate defects: {}",
        defects.join(" | ")
    );
    println!("PASS NEMOTRON_ENCODER cells=15 control=right_chunks_1 launches=663");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease and external fixtures"]
fn nemotron_subsample_layer_and_encoder_match_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_NEMOTRON_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_NEMOTRON_ENCODER_FIXTURES")
            .context("FF_NEMOTRON_ENCODER_FIXTURES missing")?,
    );
    let floors = PathBuf::from(
        std::env::var_os("FF_NEMOTRON_ENCODER_FLOORS")
            .context("FF_NEMOTRON_ENCODER_FLOORS missing")?,
    );
    let device = Device::new(0)?;
    let engine = ParakeetEncoder::load_nemotron(&model, &device)?;
    match_nemotron(&engine, &fixtures, &floors)
}

struct StreamFixture {
    pcm: Vec<f32>,
    samples: usize,
    mel: Vec<f32>,
    mel_frames: usize,
    features: Vec<f32>,
    feature_frames: usize,
    subsample: Vec<f32>,
    layer0: Vec<f32>,
    encoder: Vec<f32>,
    batch: usize,
}

fn read_stream(path: &Path) -> Result<StreamFixture> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let pcm_shape = tensors.tensor("pcm")?.shape().to_vec();
    let mel_shape = tensors.tensor("mel")?.shape().to_vec();
    let feature_shape = tensors.tensor("input_features")?.shape().to_vec();
    let rows = tensors.tensor("subsample")?.shape().to_vec();
    ensure!(
        pcm_shape.len() == 2
            && mel_shape.len() == 3
            && feature_shape.len() == 3
            && rows.len() == 3
            && mel_shape[2] == 128
            && feature_shape[2] == 128
            && rows[1] == 4
            && rows[2] == 1024
            && pcm_shape[0] == mel_shape[0]
            && feature_shape[0] == mel_shape[0]
            && rows[0] == mel_shape[0]
            && tensors.tensor("layer0")?.shape() == rows
            && tensors.tensor("encoder")?.shape() == rows,
        "stream fixture shape {}",
        path.display()
    );
    let batch = mel_shape[0];
    let mask = tensors.tensor("subsample_mask")?;
    ensure!(
        mask.dtype() == Dtype::U8
            && mask.data().len() == batch * 4
            && mask.data().iter().all(|value| *value == 1),
        "stream fixture subsample mask {}",
        path.display()
    );
    let pcm = f32_values(&tensors, "pcm")?;
    let mel = f32_values(&tensors, "mel")?;
    let features = f32_values(&tensors, "input_features")?;
    let subsample = f32_values(&tensors, "subsample")?;
    ensure!(
        pcm.len() == batch * pcm_shape[1]
            && mel.len() == batch * mel_shape[1] * 128
            && features.len() == batch * feature_shape[1] * 128
            && subsample.len() == batch * 4 * 1024,
        "stream fixture element count {}",
        path.display()
    );
    Ok(StreamFixture {
        pcm,
        samples: pcm_shape[1],
        mel,
        mel_frames: mel_shape[1],
        features,
        feature_frames: feature_shape[1],
        subsample,
        layer0: f32_values(&tensors, "layer0")?,
        encoder: f32_values(&tensors, "encoder")?,
        batch,
    })
}

fn require_floor(floors: &Value, case: &str, tap: &str) -> Result<(f64, f64)> {
    let (max_bound, rmse_bound) = bounds(floors, case, tap)?;
    ensure!(
        max_bound >= 1e-6 && rmse_bound >= 1e-7,
        "{case} {tap} floor is below eps_min"
    );
    Ok((max_bound, rmse_bound))
}

struct CellScore<'a> {
    defects: &'a mut Vec<String>,
    cells: &'a mut usize,
    floors: &'a Value,
}

fn score_stream_tap(
    score: &mut CellScore<'_>,
    label: &str,
    tap: &str,
    case: &str,
    batch: usize,
    actual: &[f32],
    expected: &[f32],
) -> Result<()> {
    let bounds = require_floor(score.floors, case, tap)?;
    score_cell(score.defects, label, case, batch, actual, expected, bounds)?;
    *score.cells += 1;
    Ok(())
}

fn encode_checked(
    engine: &ParakeetEncoder,
    session: &mut NemotronStream,
    input: &[f32],
    chunk: usize,
) -> Result<StreamChunk> {
    ParakeetEncoder::reset_encode_counters();
    let output = engine.encode_chunk(session, input)?;
    ensure!(
        ParakeetEncoder::encode_launches() == 663
            && ParakeetEncoder::encode_call_count() == 1
            && ParakeetEncoder::attention_query_rows() == 4,
        "chunk {chunk} launches={} calls={} grid={}",
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::encode_call_count(),
        ParakeetEncoder::attention_query_rows()
    );
    let cumulative = (chunk as u32 + 1) * 4;
    ensure!(
        session.cumulative() == cumulative
            && session.key_frames() == usize::min(56, usize::try_from(cumulative)?)
            && session.chunks() == cumulative / 4
            && output.subsample.len() == 4 * 1024
            && output.layer0.len() == 4 * 1024
            && output.encoder.len() == 4 * 1024,
        "chunk {chunk} cache stored={} cumulative={} chunks={}",
        session.key_frames(),
        session.cumulative(),
        session.chunks()
    );
    Ok(output)
}

struct MelCase<'a> {
    wave: &'a [f32],
    expected: &'a [f32],
    center: bool,
    last: bool,
    frames: usize,
}

fn score_mel_chunk(
    score: &mut CellScore<'_>,
    frontend: &LogMel,
    case: &str,
    batch: usize,
    mel: MelCase<'_>,
) -> Result<()> {
    let extracted = frontend.extract(&[mel.wave], false, mel.center, mel.last)?;
    let valid = extracted.mask.iter().filter(|value| **value).count();
    ensure!(
        valid == mel.frames && extracted.features.len() >= valid * 128,
        "{case}/{batch} mel frames {valid} require {}",
        mel.frames
    );
    ensure!(
        extracted.features[valid * 128..]
            .iter()
            .all(|value| *value == 0.0),
        "{case}/{batch} masked mel tail is nonzero"
    );
    score_stream_tap(
        score,
        "MEL",
        "mel",
        case,
        batch,
        &extracted.features[..valid * 128],
        mel.expected,
    )
}

fn score_encoder_chunk(
    score: &mut CellScore<'_>,
    case: &str,
    batch: usize,
    output: &StreamChunk,
    fixture: &StreamFixture,
) -> Result<()> {
    score_stream_tap(
        score,
        "SUBSAMPLE",
        "subsample",
        case,
        batch,
        &output.subsample,
        slice_f32(&fixture.subsample, batch, 4, 1024),
    )?;
    score_stream_tap(
        score,
        "LAYER0",
        "layer0",
        case,
        batch,
        &output.layer0,
        slice_f32(&fixture.layer0, batch, 4, 1024),
    )?;
    score_stream_tap(
        score,
        "ENCODER",
        "encoder",
        case,
        batch,
        &output.encoder,
        slice_f32(&fixture.encoder, batch, 4, 1024),
    )
}

fn chunk_plan(name: &str, chunk: usize) -> Result<(bool, bool, usize, usize)> {
    let last = name == "padded" && chunk == 2;
    let center = chunk == 0;
    let mel_frames = if chunk == 0 {
        25
    } else if last {
        10
    } else {
        32
    };
    let feature_frames = if chunk == 0 { 25 } else { 32 };
    Ok((center, last, mel_frames, feature_frames))
}

fn load_chunk(
    fixtures: &Path,
    name: &str,
    chunk: usize,
    batch_count: usize,
) -> Result<StreamFixture> {
    let fixture = read_stream(&fixtures.join(format!("{name}-c{chunk}.safetensors")))?;
    let (_, last, mel_frames, feature_frames) = chunk_plan(name, chunk)?;
    ensure!(
        fixture.batch == batch_count
            && fixture.mel_frames == mel_frames
            && fixture.feature_frames == feature_frames,
        "{name}-c{chunk} frames mel={} features={} batch={}",
        fixture.mel_frames,
        fixture.feature_frames,
        fixture.batch
    );
    if last {
        ensure!(
            fixture.features[mel_frames * 128..]
                .iter()
                .all(|value| *value == 0.0)
                && fixture.features.len() == feature_frames * 128,
            "{name}-c{chunk} encoder tail is not zero"
        );
    }
    Ok(fixture)
}

fn control_gap(
    actual: &[f32],
    expected: &[f32],
    tau: (f64, f64),
    case: &str,
    batch: usize,
    tap: &str,
    kind: &str,
) -> Result<()> {
    let (tau_max, tau_rmse) = tau;
    let (maximum, rmse) = difference(actual, expected)?;
    ensure!(
        maximum >= 10.0 * tau_max && rmse >= 10.0 * tau_rmse,
        "{case}/{batch} {kind} {tap} control not discriminating: max_abs={maximum} rmse={rmse} tau_max={tau_max} tau_rmse={tau_rmse}"
    );
    println!(
        "NEMOTRON_STREAM_CONTROL case={case} batch={batch} tap={tap} kind={kind} max_abs={maximum} tau={tau_max} rmse={rmse} tau={tau_rmse} PASS"
    );
    Ok(())
}

fn match_nemotron_stream(
    engine: &ParakeetEncoder,
    frontend: &LogMel,
    fixtures: &Path,
    floors_path: &Path,
) -> Result<()> {
    let floors: Value = serde_json::from_slice(&std::fs::read(floors_path)?)?;
    let eps_max = floors["eps_min"]["max_abs"]
        .as_f64()
        .context("missing eps_min max_abs")?;
    let eps_rmse = floors["eps_min"]["rmse"]
        .as_f64()
        .context("missing eps_min rmse")?;
    ensure!(
        eps_max == 1e-6 && eps_rmse == 1e-7,
        "eps_min moved: max_abs={eps_max} rmse={eps_rmse}"
    );
    let mut defects = Vec::new();
    let mut cells = 0usize;
    {
        let mut score = CellScore {
            defects: &mut defects,
            cells: &mut cells,
            floors: &floors,
        };
        for chunk in 0..3 {
            let fixture = load_chunk(fixtures, "short", chunk, 2)?;
            let (center, last, mel_frames, _) = chunk_plan("short", chunk)?;
            let case = format!("short-c{chunk}");
            let mut encoded = Vec::new();
            for batch in 0..2 {
                let wave = &fixture.pcm[batch * fixture.samples..(batch + 1) * fixture.samples];
                score_mel_chunk(
                    &mut score,
                    frontend,
                    &case,
                    batch,
                    MelCase {
                        wave,
                        expected: slice_f32(&fixture.mel, batch, mel_frames, 128),
                        center,
                        last,
                        frames: mel_frames,
                    },
                )?;
                let mut session = engine.stream()?;
                if chunk > 0 {
                    for earlier in 0..chunk {
                        let prior = load_chunk(fixtures, "short", earlier, 2)?;
                        let input = slice_f32(&prior.features, batch, prior.feature_frames, 128);
                        encode_checked(engine, &mut session, input, earlier)?;
                    }
                }
                let input = slice_f32(&fixture.features, batch, fixture.feature_frames, 128);
                let output = encode_checked(engine, &mut session, input, chunk)?;
                score_encoder_chunk(&mut score, &case, batch, &output, &fixture)?;
                encoded.push(output.encoder);
            }
            ensure!(
                encoded.len() == 2 && encoded[0] != encoded[1],
                "short-c{chunk} encoder outputs are byte-identical"
            );
        }
        ParakeetEncoder::reset_position_uploads();
        let mut window = engine.stream()?;
        ensure!(
            ParakeetEncoder::position_uploads() == 15,
            "position uploads {}",
            ParakeetEncoder::position_uploads()
        );
        for chunk in 0..16 {
            let fixture = load_chunk(fixtures, "window", chunk, 1)?;
            let input = slice_f32(&fixture.features, 0, fixture.feature_frames, 128);
            let output = encode_checked(engine, &mut window, input, chunk)?;
            if chunk == 14 || chunk == 15 {
                let case = format!("window-c{chunk}");
                score_encoder_chunk(&mut score, &case, 0, &output, &fixture)?;
            }
        }
        ensure!(
            ParakeetEncoder::position_uploads() == 15,
            "position uploads grew to {}",
            ParakeetEncoder::position_uploads()
        );
        let mut padded = engine.stream()?;
        for chunk in 0..3 {
            let fixture = load_chunk(fixtures, "padded", chunk, 1)?;
            let (center, last, mel_frames, _) = chunk_plan("padded", chunk)?;
            let input = slice_f32(&fixture.features, 0, fixture.feature_frames, 128);
            let output = encode_checked(engine, &mut padded, input, chunk)?;
            if chunk == 2 {
                let wave = &fixture.pcm[..fixture.samples];
                score_mel_chunk(
                    &mut score,
                    frontend,
                    "padded-c2",
                    0,
                    MelCase {
                        wave,
                        expected: slice_f32(&fixture.mel, 0, mel_frames, 128),
                        center,
                        last,
                        frames: mel_frames,
                    },
                )?;
                score_encoder_chunk(&mut score, "padded-c2", 0, &output, &fixture)?;
            }
        }
        ensure!(
            *score.cells == 34,
            "scored {} streaming cells",
            *score.cells
        );
    }
    let mut left = engine.stream()?;
    let mut right = engine.stream()?;
    let first = load_chunk(fixtures, "short", 0, 2)?;
    ParakeetEncoder::reset_encode_counters();
    let left_out = engine.encode_chunk(
        &mut left,
        slice_f32(&first.features, 0, first.feature_frames, 128),
    )?;
    let right_out = engine.encode_chunk(
        &mut right,
        slice_f32(&first.features, 1, first.feature_frames, 128),
    )?;
    ensure!(
        ParakeetEncoder::encode_call_count() == 2
            && ParakeetEncoder::encode_launches() == 1326
            && ParakeetEncoder::attention_query_rows() == 4
            && left_out.encoder != right_out.encoder,
        "short clips calls={} launches={} grid={}",
        ParakeetEncoder::encode_call_count(),
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::attention_query_rows()
    );
    println!("NEMOTRON_STREAM_CALLS calls=2 launches_each=663 grid=4 distinct=true PASS");
    for batch in 0..2 {
        let mut session = engine.stream()?;
        let chunk0 = load_chunk(fixtures, "short", 0, 2)?;
        encode_checked(
            engine,
            &mut session,
            slice_f32(&chunk0.features, batch, chunk0.feature_frames, 128),
            0,
        )?;
        let chunk1 = load_chunk(fixtures, "short", 1, 2)?;
        let stored = session.key_frames();
        let cumulative = session.cumulative();
        ParakeetEncoder::reset_encode_counters();
        let peeked = engine.encode_chunk_omitting(
            &mut session,
            slice_f32(&chunk1.features, batch, chunk1.feature_frames, 128),
            StreamOmit::Conv,
        )?;
        ensure!(
            ParakeetEncoder::encode_launches() == 663
                && ParakeetEncoder::attention_query_rows() == 4
                && session.key_frames() == stored
                && session.cumulative() == cumulative
                && peeked.encoder.len() == 4 * 1024,
            "short-c1 omit-conv changed the session"
        );
        control_gap(
            &peeked.layer0,
            slice_f32(&chunk1.layer0, batch, 4, 1024),
            require_floor(&floors, "short-c1", "layer0")?,
            "short-c1",
            batch,
            "layer0",
            "omit_conv",
        )?;
        control_gap(
            &peeked.encoder,
            slice_f32(&chunk1.encoder, batch, 4, 1024),
            require_floor(&floors, "short-c1", "encoder")?,
            "short-c1",
            batch,
            "encoder",
            "omit_conv",
        )?;
    }
    let mut session = engine.stream()?;
    for chunk in 0..15 {
        let fixture = load_chunk(fixtures, "window", chunk, 1)?;
        encode_checked(
            engine,
            &mut session,
            slice_f32(&fixture.features, 0, fixture.feature_frames, 128),
            chunk,
        )?;
    }
    let chunk15 = load_chunk(fixtures, "window", 15, 1)?;
    let stored = session.key_frames();
    let cumulative = session.cumulative();
    ParakeetEncoder::reset_encode_counters();
    let peeked = engine.encode_chunk_omitting(
        &mut session,
        slice_f32(&chunk15.features, 0, chunk15.feature_frames, 128),
        StreamOmit::Keys,
    )?;
    ensure!(
        ParakeetEncoder::encode_launches() == 663
            && ParakeetEncoder::attention_query_rows() == 4
            && session.key_frames() == stored
            && session.cumulative() == cumulative
            && peeked.layer0.len() == 4 * 1024,
        "window-c15 omit-keys changed the session"
    );
    control_gap(
        &peeked.layer0,
        slice_f32(&chunk15.layer0, 0, 4, 1024),
        require_floor(&floors, "window-c15", "layer0")?,
        "window-c15",
        0,
        "layer0",
        "omit_keys",
    )?;
    control_gap(
        &peeked.encoder,
        slice_f32(&chunk15.encoder, 0, 4, 1024),
        require_floor(&floors, "window-c15", "encoder")?,
        "window-c15",
        0,
        "encoder",
        "omit_keys",
    )?;
    ensure!(
        defects.is_empty(),
        "candidate defects: {}",
        defects.join(" | ")
    );
    println!("PASS NEMOTRON_STREAM cells=34 control=omit_keys,omit_conv launches=663 grid=4");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease and external fixtures"]
fn nemotron_streaming_chunks_match_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_NEMOTRON_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_NEMOTRON_STREAM_FIXTURES")
            .context("FF_NEMOTRON_STREAM_FIXTURES missing")?,
    );
    let floors = PathBuf::from(
        std::env::var_os("FF_NEMOTRON_STREAM_FLOORS")
            .context("FF_NEMOTRON_STREAM_FLOORS missing")?,
    );
    let spec = NemotronSpec::read(&model)?;
    let frontend = LogMel::new(spec.mel)?;
    let device = Device::new(0)?;
    let engine = ParakeetEncoder::load_nemotron(&model, &device)?;
    match_nemotron_stream(&engine, &frontend, &fixtures, &floors)
}
