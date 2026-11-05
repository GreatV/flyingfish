use anyhow::{Context, Result, ensure};
use flyingfish::{audio::LogMel, parakeet::ParakeetSpec};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn spec_from_data() -> Result<ParakeetSpec> {
    ParakeetSpec::from_json(
        &serde_json::from_str(include_str!("data/parakeet-tdt-config.json"))?,
        &serde_json::from_str(include_str!("data/parakeet-tdt-processor.json"))?,
        &serde_json::from_str(include_str!("data/parakeet-tdt-generation.json"))?,
    )
}

#[test]
fn parakeet_spec_rejects_changed_geometry() -> Result<()> {
    let spec = spec_from_data()?;
    ensure!(
        [spec.hidden, spec.intermediate, spec.layers, spec.heads] == [1024, 4096, 24, 8]
            && [
                spec.decoder_hidden,
                spec.decoder_layers,
                spec.vocab,
                spec.blank
            ] == [640, 2, 8193, 8192]
            && spec.durations == [0, 1, 2, 3, 4]
            && spec.mel.n_fft == 512
            && spec.mel.hop_length == 160,
        "published Parakeet geometry changed"
    );
    let mut root: Value = serde_json::from_str(include_str!("data/parakeet-tdt-config.json"))?;
    let processor = serde_json::from_str(include_str!("data/parakeet-tdt-processor.json"))?;
    let generation = serde_json::from_str(include_str!("data/parakeet-tdt-generation.json"))?;
    root["encoder_config"]["subsampling_conv_stride"] = 1.into();
    ensure!(
        ParakeetSpec::from_json(&root, &processor, &generation).is_err(),
        "changed subsampling stride accepted"
    );
    root["encoder_config"]["subsampling_conv_stride"] = 2.into();
    root["architectures"] = serde_json::json!(["ParakeetForRNNT"]);
    ensure!(
        ParakeetSpec::from_json(&root, &processor, &generation).is_err(),
        "changed architecture accepted"
    );
    Ok(())
}

#[test]
#[ignore = "published checkpoint; set FF_PARAKEET_MODEL"]
fn parakeet_published_tensor_contract() -> Result<()> {
    let directory =
        PathBuf::from(std::env::var_os("FF_PARAKEET_MODEL").context("FF_PARAKEET_MODEL missing")?);
    let spec = ParakeetSpec::read(&directory)?;
    let tensors =
        spec.validate_checkpoint(&flyingfish::checkpoint::MappedCheckpoint::open(&directory)?)?;
    ensure!(tensors == 723, "published tensor count changed");
    println!("PASS PARAKEET_TENSOR_CONTRACT tensors=723 dtypes=native shapes=exact");
    Ok(())
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

fn i64_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i64>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::I64, "{name} must be I64");
    Ok(tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .collect())
}

fn difference(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(
        actual.len() == expected.len(),
        "reference element count mismatch"
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

struct ReferenceFeatures {
    features: Vec<f32>,
    mask: Vec<u8>,
    lengths: Vec<i64>,
    shape: Vec<usize>,
}

fn read_features(path: &Path) -> Result<ReferenceFeatures> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let features = f32_values(&tensors, "input_features")?;
    let shape = tensors.tensor("input_features")?.shape().to_vec();
    let mask = tensors.tensor("attention_mask")?;
    ensure!(mask.dtype() == Dtype::U8, "attention_mask must be U8");
    Ok(ReferenceFeatures {
        features,
        mask: mask.data().to_vec(),
        lengths: i64_values(&tensors, "audio_lengths")?,
        shape,
    })
}

#[test]
#[ignore = "external CPU fixtures; set FF_PARAKEET_MODEL, FF_PARAKEET_FIXTURES and FF_PARAKEET_FLOORS"]
fn parakeet_mel_matches_frozen_reference() -> Result<()> {
    let model =
        PathBuf::from(std::env::var_os("FF_PARAKEET_MODEL").context("FF_PARAKEET_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_PARAKEET_FIXTURES").context("FF_PARAKEET_FIXTURES missing")?,
    );
    let floors: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(
        std::env::var_os("FF_PARAKEET_FLOORS").context("FF_PARAKEET_FLOORS missing")?,
    ))?)?;
    let frontend = LogMel::new(ParakeetSpec::read(&model)?.mel)?;
    for case in ["short", "edges", "production"] {
        let path = fixtures.join(format!("{case}.safetensors"));
        let ReferenceFeatures {
            features: expected,
            mask: expected_mask,
            lengths,
            shape,
        } = read_features(&path)?;
        ensure!(shape.len() == 3 && shape[2] == 128, "{case} feature shape");
        let bytes = std::fs::read(&path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let pcm = f32_values(&tensors, "pcm")?;
        let pcm_shape = tensors.tensor("pcm")?.shape().to_vec();
        ensure!(
            pcm_shape.len() == 2 && pcm_shape[0] == shape[0] && lengths.len() == shape[0],
            "{case} PCM shape"
        );
        let waves: Vec<&[f32]> = lengths
            .iter()
            .enumerate()
            .map(|(batch, &length)| {
                let length = usize::try_from(length).expect("validated audio length");
                &pcm[batch * pcm_shape[1]..batch * pcm_shape[1] + length]
            })
            .collect();
        let actual = frontend.extract(&waves)?;
        ensure!(
            actual.frames == shape[1]
                && actual.mel_bins == shape[2]
                && actual.features.len() == expected.len()
                && actual.mask.len() == expected_mask.len(),
            "{case} output shape mismatch"
        );
        ensure!(
            actual
                .mask
                .iter()
                .zip(expected_mask.iter())
                .all(|(&a, &b)| a == (b != 0)),
            "{case} attention mask mismatch"
        );
        let (maximum, rmse) = difference(&actual.features, &expected)?;
        let max_bound = floors["cases"][case]["max_abs"]
            .as_f64()
            .with_context(|| format!("missing {case} max_abs floor"))?;
        let rmse_bound = floors["cases"][case]["rmse"]
            .as_f64()
            .with_context(|| format!("missing {case} rmse floor"))?;
        ensure!(
            maximum <= max_bound && rmse <= rmse_bound,
            "{case} DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} actual={:?} expected={:?}",
            &actual.features[..4],
            &expected[..4]
        );
        println!(
            "PARAKEET_MEL case={case} frames={} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS",
            actual.frames
        );
        if case == "short" {
            let control = read_features(&fixtures.join("short-control.safetensors"))?.features;
            let (control_max, control_rmse) = difference(&actual.features, &control)?;
            ensure!(
                control_max > 10.0 * max_bound && control_rmse > 10.0 * rmse_bound,
                "periodic-window control is not discriminating: max_abs={control_max} rmse={control_rmse}"
            );
            println!(
                "PARAKEET_MEL_CONTROL periodic_hann max_abs={control_max} rmse={control_rmse} PASS"
            );
        }
    }
    println!("PASS PARAKEET_MEL cells=3 control=periodic_hann");
    Ok(())
}
