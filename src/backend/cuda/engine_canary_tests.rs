#![cfg(feature = "cuda")]

use super::{Device, engine_canary::Canary};
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::path::{Path, PathBuf};

const SEQUENCE: usize = 26;
const HIDDEN: usize = 1024;
const VOCAB: usize = 16384;

struct Item {
    features: Vec<f32>,
    valid_frames: usize,
    encoder: Vec<f32>,
    rows: usize,
    valid_rows: usize,
    tokens: Vec<u32>,
    tokens_de: Vec<u32>,
    embeddings: Vec<f32>,
    layer0: Vec<f32>,
    logits: Vec<f32>,
    embeddings_de: Vec<f32>,
    layer0_de: Vec<f32>,
    logits_de: Vec<f32>,
    layer0_causal: Vec<f32>,
    logits_causal: Vec<f32>,
    layer0_kept: Vec<f32>,
    logits_kept: Vec<f32>,
    layer0_padded: Vec<f32>,
    logits_padded: Vec<f32>,
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

fn i32_values(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::I32, "{name} must be I32");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect())
}

fn read_items(path: &Path, with_mel: bool) -> Result<Vec<Item>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let encoder_shape = tensors.tensor("encoder")?.shape().to_vec();
    ensure!(
        encoder_shape.len() == 3 && encoder_shape[2] == HIDDEN,
        "encoder fixture shape"
    );
    let batch = encoder_shape[0];
    let rows = encoder_shape[1];
    let mask = tensors.tensor("encoder_mask")?;
    ensure!(mask.dtype() == Dtype::U8, "encoder_mask must be U8");
    let (frames, features, frame_mask) = if with_mel {
        let shape = tensors.tensor("input_features")?.shape().to_vec();
        ensure!(
            shape.len() == 3 && shape[0] == batch && shape[2] == 128,
            "mel fixture shape"
        );
        let frame_mask = tensors.tensor("attention_mask")?;
        ensure!(frame_mask.dtype() == Dtype::U8, "attention_mask must be U8");
        (
            shape[1],
            f32_values(&tensors, "input_features")?,
            frame_mask.data().to_vec(),
        )
    } else {
        (0, Vec::new(), Vec::new())
    };
    let tokens = i32_values(&tensors, "input_ids")?;
    let tokens_de = i32_values(&tensors, "input_ids_de")?;
    ensure!(
        tokens.len() == batch * SEQUENCE && tokens_de.len() == tokens.len(),
        "token fixture length"
    );
    let mut items = Vec::with_capacity(batch);
    for index in 0..batch {
        let valid_rows = mask.data()[index * rows..(index + 1) * rows]
            .iter()
            .filter(|&&value| value != 0)
            .count();
        let valid_frames = if with_mel {
            frame_mask[index * frames..(index + 1) * frames]
                .iter()
                .filter(|&&value| value != 0)
                .count()
        } else {
            0
        };
        let row = index * rows * HIDDEN;
        let sequence = index * SEQUENCE * HIDDEN;
        let logit = index * SEQUENCE * VOCAB;
        let ids = index * SEQUENCE;
        items.push(Item {
            features: if with_mel {
                features[index * frames * 128..(index + 1) * frames * 128].to_vec()
            } else {
                Vec::new()
            },
            valid_frames,
            encoder: f32_values(&tensors, "encoder")?[row..row + rows * HIDDEN].to_vec(),
            rows,
            valid_rows,
            tokens: tokens[ids..ids + SEQUENCE].to_vec(),
            tokens_de: tokens_de[ids..ids + SEQUENCE].to_vec(),
            embeddings: f32_values(&tensors, "embeddings")?[sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            layer0: f32_values(&tensors, "layer0")?[sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            logits: f32_values(&tensors, "logits")?[logit..logit + SEQUENCE * VOCAB].to_vec(),
            embeddings_de: f32_values(&tensors, "embeddings_de")?
                [sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            layer0_de: f32_values(&tensors, "layer0_de")?[sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            logits_de: f32_values(&tensors, "logits_de")?[logit..logit + SEQUENCE * VOCAB].to_vec(),
            layer0_causal: f32_values(&tensors, "layer0_causal")?
                [sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            logits_causal: f32_values(&tensors, "logits_causal")?[logit..logit + SEQUENCE * VOCAB]
                .to_vec(),
            layer0_kept: f32_values(&tensors, "layer0_kept")?
                [sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            logits_kept: f32_values(&tensors, "logits_kept")?[logit..logit + SEQUENCE * VOCAB]
                .to_vec(),
            layer0_padded: f32_values(&tensors, "layer0_padded")?
                [sequence..sequence + SEQUENCE * HIDDEN]
                .to_vec(),
            logits_padded: f32_values(&tensors, "logits_padded")?[logit..logit + SEQUENCE * VOCAB]
                .to_vec(),
        });
    }
    Ok(items)
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
    for (&left, &right) in actual.iter().zip(expected) {
        let error = f64::from(left - right).abs();
        maximum = maximum.max(error);
        squares += error * error;
    }
    Ok((maximum, (squares / actual.len() as f64).sqrt()))
}

fn bit_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn perturb_row(row: &mut [f32]) {
    for (index, value) in row.iter_mut().enumerate() {
        let pattern = ((index * 17 + 3) % 251) as f32;
        *value = pattern / 125.0 - 1.0;
    }
}

fn bound(floors: &Value, case: &str, path: &str, tap: &str) -> Result<(f64, f64)> {
    Ok((
        floors["cases"][case][path][tap]["max_abs"]
            .as_f64()
            .with_context(|| format!("missing {case}/{path}/{tap} max_abs"))?,
        floors["cases"][case][path][tap]["rmse"]
            .as_f64()
            .with_context(|| format!("missing {case}/{path}/{tap} rmse"))?,
    ))
}

fn within(
    defects: &mut Vec<String>,
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_bound: f64,
    rmse_bound: f64,
) -> Result<()> {
    let (maximum, rmse) = difference(actual, expected)?;
    if maximum > max_bound || rmse > rmse_bound {
        defects.push(format!(
            "{label} DEFECT max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound}"
        ));
    } else {
        println!("{label} max_abs={maximum} bound={max_bound} rmse={rmse} bound={rmse_bound} PASS");
    }
    Ok(())
}

fn far(
    defects: &mut Vec<String>,
    label: &str,
    actual: &[f32],
    control: &[f32],
    max_bound: f64,
    rmse_bound: f64,
) -> Result<()> {
    let (maximum, rmse) = difference(actual, control)?;
    if maximum <= 10.0 * max_bound || rmse <= 10.0 * rmse_bound {
        defects.push(format!(
            "{label} control DEFECT max_abs={maximum} rmse={rmse} bound={max_bound}/{rmse_bound}"
        ));
    } else {
        println!("{label} control max_abs={maximum} rmse={rmse} ratio_bound=10 PASS");
    }
    Ok(())
}

fn argmax_rows(logits: &[f32]) -> Vec<(usize, f32)> {
    logits
        .chunks(VOCAB)
        .map(|row| {
            let mut best = 0usize;
            let mut top = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (index, &value) in row.iter().enumerate() {
                if value > top {
                    second = top;
                    top = value;
                    best = index;
                } else if value > second {
                    second = value;
                }
            }
            (best, top - second)
        })
        .collect()
}

fn check_ids(defects: &mut Vec<String>, label: &str, logits: &[f32], spec: &Value) -> Result<()> {
    let assert_ids = spec["assert"]
        .as_bool()
        .with_context(|| format!("missing {label} id assert"))?;
    if !assert_ids {
        println!("{label} ids not asserted");
        return Ok(());
    }
    let expected = spec["ids"]
        .as_array()
        .with_context(|| format!("missing {label} ids"))?;
    ensure!(expected.len() == SEQUENCE, "{label} id count");
    let actual = argmax_rows(logits);
    for (row, (expected_id, (actual_id, margin))) in expected.iter().zip(actual).enumerate() {
        let expected_id = expected_id
            .as_u64()
            .with_context(|| format!("{label} row {row} id is not an integer"))?;
        let Ok(expected_index) = usize::try_from(expected_id) else {
            defects.push(format!(
                "{label} row {row} id DEFECT actual={actual_id} expected={expected_id} margin={margin}"
            ));
            continue;
        };
        if actual_id != expected_index {
            defects.push(format!(
                "{label} row {row} id DEFECT actual={actual_id} expected={expected_id} margin={margin}"
            ));
        }
    }
    println!("{label} ids PASS");
    Ok(())
}

fn check_mask(
    defects: &mut Vec<String>,
    engine: &Canary,
    item: &Item,
    label: &str,
    layer_bound: (f64, f64),
    logit_bound: (f64, f64),
) -> Result<()> {
    let (max_bound, rmse_bound) = layer_bound;
    let (logit_max, logit_rmse) = logit_bound;
    ensure!(
        bit_equal(&item.layer0_padded, &item.layer0)
            && bit_equal(&item.logits_padded, &item.logits),
        "{label} padded-row fixture is not bit-equal"
    );
    let clean = engine.decode(&item.encoder, item.valid_rows, &item.tokens)?;
    if item.valid_rows < item.rows {
        let mut changed = item.encoder.clone();
        let start = item.valid_rows * HIDDEN;
        perturb_row(&mut changed[start..start + HIDDEN]);
        let perturbed = engine.decode(&changed, item.valid_rows, &item.tokens)?;
        if bit_equal(&clean.0, &perturbed.0)
            && bit_equal(&clean.1, &perturbed.1)
            && bit_equal(&clean.2, &perturbed.2)
        {
            println!("{label} padded-row bit-equal PASS");
        } else {
            defects.push(format!(
                "{label} padded-row candidate is not bit-equal DEFECT"
            ));
        }
    }
    far(
        defects,
        &format!("{label} kept-row layer0"),
        &clean.1,
        &item.layer0_kept,
        max_bound,
        rmse_bound,
    )?;
    far(
        defects,
        &format!("{label} kept-row logits"),
        &clean.2,
        &item.logits_kept,
        logit_max,
        logit_rmse,
    )?;
    Ok(())
}

fn match_decoder(engine: &Canary, fixtures: &Path, floors: &Value) -> Result<()> {
    let mut defects = Vec::new();
    for case in ["short", "edges", "production"] {
        let full = read_items(&fixtures.join(format!("full/{case}.safetensors")), true)?;
        let isolated = read_items(
            &fixtures.join(format!("isolated/{case}.safetensors")),
            false,
        )?;
        ensure!(full.len() == isolated.len(), "{case} full/isolated batch");
        let (embed_max, embed_rmse) = bound(floors, case, "full", "embeddings")?;
        let (layer_max, layer_rmse) = bound(floors, case, "full", "layer0")?;
        let (logit_max, logit_rmse) = bound(floors, case, "full", "logits")?;
        let (i_embed_max, i_embed_rmse) = bound(floors, case, "isolated", "embeddings")?;
        let (i_layer_max, i_layer_rmse) = bound(floors, case, "isolated", "layer0")?;
        let (i_logit_max, i_logit_rmse) = bound(floors, case, "isolated", "logits")?;
        for (index, (item, isolated_item)) in full.iter().zip(&isolated).enumerate() {
            ensure!(
                item.encoder.len() == isolated_item.encoder.len()
                    && item.valid_rows == isolated_item.valid_rows
                    && item.tokens == isolated_item.tokens,
                "{case}/{index} isolated fixture does not match the full token and mask contract"
            );
            let (embeddings, layer0, logits) =
                engine.logits_from_mel(&item.features, item.valid_frames, &item.tokens)?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} embeddings"),
                &embeddings,
                &item.embeddings,
                embed_max,
                embed_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} layer0"),
                &layer0,
                &item.layer0,
                layer_max,
                layer_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} logits"),
                &logits,
                &item.logits,
                logit_max,
                logit_rmse,
            )?;
            let de = engine.logits_from_mel(&item.features, item.valid_frames, &item.tokens_de)?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} embeddings_de"),
                &de.0,
                &item.embeddings_de,
                embed_max,
                embed_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} layer0_de"),
                &de.1,
                &item.layer0_de,
                layer_max,
                layer_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} logits_de"),
                &de.2,
                &item.logits_de,
                logit_max,
                logit_rmse,
            )?;
            far(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} language layer0"),
                &layer0,
                &item.layer0_de,
                layer_max,
                layer_rmse,
            )?;
            far(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} language logits"),
                &logits,
                &item.logits_de,
                logit_max,
                logit_rmse,
            )?;
            far(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} causal layer0"),
                &layer0,
                &item.layer0_causal,
                layer_max,
                layer_rmse,
            )?;
            far(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index} causal logits"),
                &logits,
                &item.logits_causal,
                logit_max,
                logit_rmse,
            )?;
            check_mask(
                &mut defects,
                engine,
                item,
                &format!("CANARY_DECODER full {case}/{index}"),
                (layer_max, layer_rmse),
                (logit_max, logit_rmse),
            )?;
            let spec = &floors["cases"][case]["items"][index]["full_ids"];
            check_ids(
                &mut defects,
                &format!("CANARY_DECODER full {case}/{index}"),
                &logits,
                spec,
            )?;
            let decoded = engine.decode(
                &isolated_item.encoder,
                isolated_item.valid_rows,
                &isolated_item.tokens,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER isolated {case}/{index} embeddings"),
                &decoded.0,
                &isolated_item.embeddings,
                i_embed_max,
                i_embed_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER isolated {case}/{index} layer0"),
                &decoded.1,
                &isolated_item.layer0,
                i_layer_max,
                i_layer_rmse,
            )?;
            within(
                &mut defects,
                &format!("CANARY_DECODER isolated {case}/{index} logits"),
                &decoded.2,
                &isolated_item.logits,
                i_logit_max,
                i_logit_rmse,
            )?;
            check_mask(
                &mut defects,
                engine,
                isolated_item,
                &format!("CANARY_DECODER isolated {case}/{index}"),
                (i_layer_max, i_layer_rmse),
                (i_logit_max, i_logit_rmse),
            )?;
            let spec = &floors["cases"][case]["items"][index]["isolated_ids"];
            check_ids(
                &mut defects,
                &format!("CANARY_DECODER isolated {case}/{index}"),
                &decoded.2,
                spec,
            )?;
        }
    }
    for defect in &defects {
        println!("{defect}");
    }
    ensure!(
        defects.is_empty(),
        "CANARY_DECODER defects={}",
        defects.len()
    );
    println!("PASS CANARY_DECODER cells=5 controls=language+causal+kept-row+padded-row");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease, frozen floors, and external fixtures"]
fn canary_decoder_matches_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_CANARY_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_CANARY_MODEL").context("FF_CANARY_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_CANARY_DECODER_FIXTURES")
            .context("FF_CANARY_DECODER_FIXTURES missing")?,
    );
    let floors: Value = serde_json::from_slice(&std::fs::read(
        std::env::var_os("FF_CANARY_DECODER_FLOORS").context("FF_CANARY_DECODER_FLOORS missing")?,
    )?)?;
    ensure!(
        floors["schema"] == "n6c-m2-decoder-v1",
        "decoder floor schema"
    );
    let device = Device::new(0)?;
    let engine = Canary::load(&model, &device)?;
    match_decoder(&engine, &fixtures, &floors)
}
