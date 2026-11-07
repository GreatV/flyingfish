#![cfg(feature = "cuda")]

use super::{Device, engine_parakeet::ParakeetEncoder};
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
        ParakeetEncoder::encode_launches() == 663 && ParakeetEncoder::encode_calls() == 1,
        "short clip 0 launches={} calls={}",
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::encode_calls()
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
        ParakeetEncoder::encode_launches() == 1326 && ParakeetEncoder::encode_calls() == 2,
        "short clip 1 launches={} calls={}",
        ParakeetEncoder::encode_launches(),
        ParakeetEncoder::encode_calls()
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
                ParakeetEncoder::encode_launches() == 663 && ParakeetEncoder::encode_calls() == 1,
                "{case}/{batch} launches={} calls={}",
                ParakeetEncoder::encode_launches(),
                ParakeetEncoder::encode_calls()
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
