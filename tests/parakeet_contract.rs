use anyhow::{Context, Result, ensure};
use flyingfish::{
    audio::LogMel, canary::CanarySpec, nemotron::NemotronSpec, parakeet::ParakeetSpec,
};
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
    let conformer = spec.conformer();
    ensure!(
        !conformer.attention_bias
            && !conformer.convolution_bias
            && conformer.prefix == "encoder"
            && conformer.layers == 24
            && conformer.hidden == 1024
            && conformer.heads == 8
            && conformer.conv_kernel == 9
            && conformer.conv_batch_norm
            && conformer.pad_left == 1
            && conformer.pad_right == 1
            && conformer.depthwise_left == 4
            && conformer.linear_in == 4096
            && conformer.chunk_size == 0
            && conformer.left_chunks == 0
            && conformer.max_position == spec.max_position,
        "Parakeet conformer bias or prefix changed"
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

fn match_mel(
    frontend: &LogMel,
    fixtures: &Path,
    floors: &Value,
    label: &str,
    normalize: bool,
) -> Result<()> {
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
        let actual = frontend.extract(&waves, normalize)?;
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
            "{label}_MEL case={case} frames={} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS",
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
                "{label}_MEL_CONTROL periodic_hann max_abs={control_max} rmse={control_rmse} PASS"
            );
        }
    }
    println!("PASS {label}_MEL cells=3 control=periodic_hann");
    Ok(())
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
    match_mel(&frontend, &fixtures, &floors, "PARAKEET", true)
}

fn canary_spec_from_data() -> Result<CanarySpec> {
    CanarySpec::from_json(
        &serde_json::from_str(include_str!("data/canary-1b-v2-config.json"))?,
        &serde_json::from_str(include_str!("data/canary-1b-v2-processor.json"))?,
        &serde_json::from_str(include_str!("data/canary-1b-v2-generation.json"))?,
    )
}

#[test]
fn canary_spec_rejects_changed_geometry() -> Result<()> {
    let spec = canary_spec_from_data()?;
    let conformer = spec.conformer();
    ensure!(
        [
            spec.hidden,
            spec.layers,
            spec.heads,
            spec.vocab,
            spec.decoder_layers
        ] == [1024, 32, 8, 16384, 8]
            && spec.mel.n_fft == 512
            && spec.mel.hop_length == 160
            && conformer.attention_bias
            && conformer.convolution_bias
            && conformer.prefix == "model.encoder"
            && conformer.layers == 32
            && conformer.conv_kernel == 9
            && conformer.conv_batch_norm
            && conformer.pad_left == 1
            && conformer.pad_right == 1
            && conformer.depthwise_left == 4
            && conformer.linear_in == 4096
            && conformer.chunk_size == 0
            && conformer.left_chunks == 0
            && conformer.max_position == 5000,
        "published Canary geometry changed"
    );
    let mut root: Value = serde_json::from_str(include_str!("data/canary-1b-v2-config.json"))?;
    let processor = serde_json::from_str(include_str!("data/canary-1b-v2-processor.json"))?;
    let generation = serde_json::from_str(include_str!("data/canary-1b-v2-generation.json"))?;
    root["encoder_config"]["num_hidden_layers"] = 24.into();
    ensure!(
        CanarySpec::from_json(&root, &processor, &generation).is_err(),
        "changed encoder layer count accepted"
    );
    root["encoder_config"]["num_hidden_layers"] = 32.into();
    root["architectures"] = serde_json::json!(["ParakeetForTDT"]);
    ensure!(
        CanarySpec::from_json(&root, &processor, &generation).is_err(),
        "changed architecture accepted"
    );
    Ok(())
}

#[test]
#[ignore = "published checkpoint; set FF_CANARY_MODEL"]
fn canary_published_tensor_contract() -> Result<()> {
    let directory =
        PathBuf::from(std::env::var_os("FF_CANARY_MODEL").context("FF_CANARY_MODEL missing")?);
    let spec = CanarySpec::read(&directory)?;
    let tensors =
        spec.validate_checkpoint(&flyingfish::checkpoint::MappedCheckpoint::open(&directory)?)?;
    ensure!(tensors == 1507, "published tensor count changed");
    println!(
        "PASS CANARY_TENSOR_CONTRACT tensors=1507 dtypes=native shapes=exact tie=proj_out.weight==model.decoder.embed_tokens.weight kept=neither"
    );
    Ok(())
}

#[test]
#[ignore = "external CPU fixtures; set FF_CANARY_MODEL, FF_CANARY_FIXTURES and FF_CANARY_FLOORS"]
fn canary_mel_matches_frozen_reference() -> Result<()> {
    let model =
        PathBuf::from(std::env::var_os("FF_CANARY_MODEL").context("FF_CANARY_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_CANARY_FIXTURES").context("FF_CANARY_FIXTURES missing")?,
    );
    let floors: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(
        std::env::var_os("FF_CANARY_FLOORS").context("FF_CANARY_FLOORS missing")?,
    ))?)?;
    let frontend = LogMel::new(CanarySpec::read(&model)?.mel)?;
    match_mel(&frontend, &fixtures, &floors, "CANARY", true)
}

#[test]
#[ignore = "external CPU fixtures; set FF_NEMOTRON_MODEL, FF_NEMOTRON_FIXTURES and FF_NEMOTRON_FLOORS"]
fn nemotron_mel_matches_frozen_reference() -> Result<()> {
    let model =
        PathBuf::from(std::env::var_os("FF_NEMOTRON_MODEL").context("FF_NEMOTRON_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_NEMOTRON_FIXTURES").context("FF_NEMOTRON_FIXTURES missing")?,
    );
    let floors: Value = serde_json::from_slice(&std::fs::read(PathBuf::from(
        std::env::var_os("FF_NEMOTRON_FLOORS").context("FF_NEMOTRON_FLOORS missing")?,
    ))?)?;
    let frontend = LogMel::new(NemotronSpec::read(&model)?.mel)?;
    match_mel(&frontend, &fixtures, &floors, "NEMOTRON", false)
}
