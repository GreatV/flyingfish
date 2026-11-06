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
    Ok(Fixture {
        features: f32_values(&tensors, "input_features")?,
        mask: mask.data().to_vec(),
        subsample: f32_values(&tensors, "subsample")?,
        subsample_mask: subsample_mask.data().to_vec(),
        layer0: f32_values(&tensors, "layer0")?,
        control: f32_values(&tensors, "layer0_control")?,
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
