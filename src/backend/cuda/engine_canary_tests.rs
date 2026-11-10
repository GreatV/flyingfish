#![cfg(feature = "cuda")]

use super::{
    Device,
    engine_canary::{Canary, StepTally},
};
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
            compare_cached(
                engine,
                &isolated_item.encoder,
                isolated_item.rows,
                isolated_item.valid_rows,
                &isolated_item.tokens,
                &format!("CANARY_DECODER isolated {case}/{index}"),
            )?;
            compare_cached(
                engine,
                &isolated_item.encoder,
                isolated_item.rows,
                isolated_item.valid_rows,
                &isolated_item.tokens_de,
                &format!("CANARY_DECODER isolated {case}/{index} de"),
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

const PREFILL: usize = 9;

struct NumericItem {
    encoder: Vec<f32>,
    valid_rows: usize,
    tokens: Vec<u32>,
    logits: Vec<f32>,
}

struct Clip {
    features: Vec<f32>,
    valid_frames: usize,
    encoder: Vec<f32>,
    valid_rows: usize,
    prompt: Vec<u32>,
    scores: Vec<f32>,
}

fn prefix_valid(mask: &[u8]) -> Result<usize> {
    let valid = mask.iter().filter(|&&value| value != 0).count();
    ensure!(valid > 0, "encoder valid count is 0");
    ensure!(
        mask[..valid].iter().all(|&value| value != 0),
        "encoder mask is not a prefix"
    );
    ensure!(
        mask[valid..].iter().all(|&value| value == 0),
        "encoder mask is not a right-padding prefix"
    );
    Ok(valid)
}

fn read_numeric(path: &Path) -> Result<Vec<NumericItem>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let shape = tensors.tensor("encoder")?.shape().to_vec();
    ensure!(
        shape.len() == 3 && shape[2] == HIDDEN,
        "encoder fixture shape"
    );
    let (batch, rows) = (shape[0], shape[1]);
    let mask = tensors.tensor("encoder_mask")?;
    ensure!(
        mask.dtype() == Dtype::U8 && mask.shape() == [batch, rows],
        "encoder_mask fixture"
    );
    let tokens = i32_values(&tensors, "input_ids")?;
    let logits = f32_values(&tensors, "logits")?;
    let encoder = f32_values(&tensors, "encoder")?;
    ensure!(tokens.len() == batch * SEQUENCE, "token fixture length");
    ensure!(
        logits.len() == batch * SEQUENCE * VOCAB,
        "logit fixture length"
    );
    let mut items = Vec::with_capacity(batch);
    for index in 0..batch {
        let mask_row = &mask.data()[index * rows..(index + 1) * rows];
        items.push(NumericItem {
            encoder: encoder[index * rows * HIDDEN..(index + 1) * rows * HIDDEN].to_vec(),
            valid_rows: prefix_valid(mask_row)?,
            tokens: tokens[index * SEQUENCE..(index + 1) * SEQUENCE].to_vec(),
            logits: logits[index * SEQUENCE * VOCAB..(index + 1) * SEQUENCE * VOCAB].to_vec(),
        });
    }
    Ok(items)
}

fn read_clip(path: &Path) -> Result<Clip> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let frames = tensors.tensor("input_features")?.shape().to_vec();
    ensure!(frames.len() == 2 && frames[1] == 128, "mel fixture shape");
    let frame_mask = tensors.tensor("attention_mask")?;
    ensure!(
        frame_mask.dtype() == Dtype::U8 && frame_mask.shape() == [frames[0]],
        "attention_mask fixture"
    );
    let encoder_shape = tensors.tensor("encoder")?.shape().to_vec();
    ensure!(
        encoder_shape.len() == 2 && encoder_shape[1] == HIDDEN,
        "clip encoder shape"
    );
    let encoder_mask = tensors.tensor("encoder_mask")?;
    ensure!(
        encoder_mask.dtype() == Dtype::U8 && encoder_mask.shape() == [encoder_shape[0]],
        "clip encoder_mask"
    );
    let scores = f32_values(&tensors, "scores")?;
    ensure!(
        scores.len().is_multiple_of(VOCAB) && !scores.is_empty(),
        "clip score length"
    );
    let ids = i32_values(&tensors, "ids")?;
    let prompt = i32_values(&tensors, "prompt")?;
    ensure!(prompt.len() == PREFILL, "clip prompt length");
    ensure!(
        ids.len() == PREFILL + scores.len() / VOCAB,
        "clip id length"
    );
    ensure!(
        ids[..PREFILL] == prompt,
        "clip ids do not start with the prompt"
    );
    ensure!(ids[ids.len() - 1] == 3, "clip ids do not end on eos");
    Ok(Clip {
        features: f32_values(&tensors, "input_features")?,
        valid_frames: prefix_valid(frame_mask.data())?,
        encoder: f32_values(&tensors, "encoder")?,
        valid_rows: prefix_valid(encoder_mask.data())?,
        prompt,
        scores,
    })
}

fn cross_delta(engine: &Canary, before: u64) -> Result<u64> {
    let writes = engine.cross_writes();
    ensure!(writes >= before, "cross-write counter moved backwards");
    Ok(writes - before)
}

fn check_generated(
    defects: &mut Vec<String>,
    label: &str,
    actual: &[u32],
    spec: &Value,
) -> Result<()> {
    ensure!(
        spec["assert"].as_bool() == Some(true),
        "{label} ids are not asserted"
    );
    let expected = spec["ids"]
        .as_array()
        .with_context(|| format!("missing {label} ids"))?;
    let margin = spec["margin"]
        .as_f64()
        .with_context(|| format!("missing {label} margin"))?;
    let mut expected_ids = Vec::with_capacity(expected.len());
    for (row, value) in expected.iter().enumerate() {
        let id = value
            .as_u64()
            .with_context(|| format!("{label} row {row} id is not an integer"))?;
        expected_ids.push(u32::try_from(id).with_context(|| format!("{label} row {row} id"))?);
    }
    if actual == expected_ids {
        println!("{label} ids margin={margin} PASS");
    } else {
        defects.push(format!(
            "{label} ids DEFECT actual={actual:?} expected={expected_ids:?} margin={margin}"
        ));
    }
    Ok(())
}

const TEACHER_H2D: u64 = 4 + 8;
const DEVICE_H2D: u64 = 8;
const ARGMAX_D2H: u64 = 4;
const DEVICE_D2D: u64 = 4;

fn attention_parts(keys: u32) -> u32 {
    let chunk = keys.div_ceil(32).max(512).next_multiple_of(4);
    keys.div_ceil(chunk)
}

fn body_launches(cross_rows: u32, cross_valid: u32) -> StepTally {
    let strided =
        u64::from(8 * (1 + attention_parts(1024)) + 8 * (1 + attention_parts(cross_rows)));
    let exclude = 8 + if cross_valid == cross_rows { 0 } else { 8 };
    StepTally {
        gemm_ex: 65,
        bias: 65,
        norm: 26,
        residual: 25,
        activation: 8,
        gather: 2,
        place: 16,
        exclude,
        softmax: 16,
        sum_parts: 16,
        strided,
        argmax: 0,
        finite: 0,
        h2d: 0,
        d2h: 0,
        d2d: 0,
        graph_launches: 0,
        graph_captures: 0,
    }
}

fn launch_total(step: &StepTally) -> u64 {
    step.gemm_ex
        + step.bias
        + step.norm
        + step.residual
        + step.activation
        + step.gather
        + step.place
        + step.exclude
        + step.softmax
        + step.sum_parts
        + step.strided
}

fn capture_tally(cross_rows: u32, cross_valid: u32, teacher: bool) -> StepTally {
    let mut step = body_launches(cross_rows, cross_valid);
    step.argmax = 1;
    step.finite = 1;
    step.h2d = if teacher { TEACHER_H2D } else { DEVICE_H2D };
    step.d2h = ARGMAX_D2H;
    step.d2d = if teacher { 0 } else { DEVICE_D2D };
    step.graph_captures = 1;
    step
}

fn eager_tally(cross_rows: u32, cross_valid: u32, teacher: bool) -> StepTally {
    let mut step = capture_tally(cross_rows, cross_valid, teacher);
    step.graph_captures = 0;
    step
}

fn replay_tally(teacher: bool) -> StepTally {
    StepTally {
        gemm_ex: 0,
        bias: 0,
        norm: 0,
        residual: 0,
        activation: 0,
        gather: 0,
        place: 0,
        exclude: 0,
        softmax: 0,
        sum_parts: 0,
        strided: 0,
        argmax: 1,
        finite: 1,
        h2d: if teacher { TEACHER_H2D } else { DEVICE_H2D },
        d2h: ARGMAX_D2H,
        d2d: if teacher { 0 } else { DEVICE_D2D },
        graph_launches: 1,
        graph_captures: 0,
    }
}

fn assert_named(label: &str, step: &StepTally, valid_eq_rows: bool) -> Result<()> {
    ensure!(step.gather == 2, "{label} gather {}", step.gather);
    ensure!(step.softmax == 16, "{label} softmax {}", step.softmax);
    ensure!(step.sum_parts == 16, "{label} sum_parts {}", step.sum_parts);
    ensure!(
        step.exclude == if valid_eq_rows { 8 } else { 16 },
        "{label} exclude {}",
        step.exclude
    );
    Ok(())
}

fn assert_graph_steps(
    label: &str,
    steps: &[StepTally],
    rows: u32,
    valid: u32,
    teacher: bool,
) -> Result<()> {
    ensure!(!steps.is_empty(), "{label} has no 1-row step");
    let expected = capture_tally(rows, valid, teacher);
    ensure!(
        steps[0] == expected,
        "{label} capture {steps:?} expected {expected:?}"
    );
    assert_named(&format!("{label} capture"), &steps[0], valid == rows)?;
    let replay = replay_tally(teacher);
    for (index, step) in steps.iter().enumerate().skip(1) {
        ensure!(
            *step == replay,
            "{label} replay {index} {step:?} expected {replay:?}"
        );
    }
    Ok(())
}

fn assert_eager_steps(
    label: &str,
    steps: &[StepTally],
    rows: u32,
    valid: u32,
    teacher: bool,
) -> Result<()> {
    let expected = eager_tally(rows, valid, teacher);
    for (index, step) in steps.iter().enumerate() {
        ensure!(
            *step == expected,
            "{label} eager {index} {step:?} expected {expected:?}"
        );
        assert_named(&format!("{label} eager {index}"), step, valid == rows)?;
    }
    Ok(())
}

fn first_mismatch(left: &[f32], right: &[f32]) -> usize {
    if left.len() != right.len() {
        return left.len().min(right.len());
    }
    left.iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
        .unwrap_or(left.len())
}

fn compare_cached(
    engine: &Canary,
    encoder: &[f32],
    rows: usize,
    valid: usize,
    tokens: &[u32],
    label: &str,
) -> Result<()> {
    ensure!(
        tokens.len() == SEQUENCE && rows > 0 && encoder.len() == rows * HIDDEN,
        "{label} cached shape"
    );
    let rows_u = u32::try_from(rows).context("encoder rows")?;
    let valid_u = u32::try_from(valid).context("encoder valid")?;
    engine.set_graph(true);
    let graph = engine.cached_logits(encoder, valid, tokens, PREFILL)?;
    let steps = engine.take_steps();
    ensure!(
        steps.len() == SEQUENCE - PREFILL,
        "{label} graph steps {}",
        steps.len()
    );
    assert_graph_steps(label, &steps, rows_u, valid_u, true)?;
    engine.set_graph(false);
    let eager = engine.cached_logits(encoder, valid, tokens, PREFILL)?;
    ensure!(
        bit_equal(&graph, &eager),
        "{label} graph/eager logit mismatch at {}",
        first_mismatch(&graph, &eager)
    );
    let eager_steps = engine.take_steps();
    ensure!(
        eager_steps.len() == SEQUENCE - PREFILL,
        "{label} eager steps {}",
        eager_steps.len()
    );
    assert_eager_steps(label, &eager_steps, rows_u, valid_u, true)?;
    engine.set_graph(true);
    Ok(())
}

fn check_device_steps(
    engine: &Canary,
    label: &str,
    ids: &[u32],
    prompt_len: usize,
    encoder: &[f32],
    valid_rows: usize,
) -> Result<()> {
    let steps = engine.take_steps();
    ensure!(
        ids.len() > prompt_len && ids.last() == Some(&3),
        "{label} ids"
    );
    ensure!(
        steps.len() == ids.len() - prompt_len - 1,
        "{label} steps {} ids {}",
        steps.len(),
        ids.len()
    );
    ensure!(!steps.is_empty(), "{label} device step");
    let rows = encoder.len() / HIDDEN;
    ensure!(
        encoder.len() == rows * HIDDEN && rows > 0,
        "{label} encoder rows"
    );
    assert_graph_steps(
        label,
        &steps,
        u32::try_from(rows).context("encoder rows")?,
        u32::try_from(valid_rows).context("encoder valid")?,
        false,
    )
}

#[test]
fn one_row_launch_inventory() {
    assert_eq!(attention_parts(1024), 2);
    assert_eq!(attention_parts(376), 1);
    assert_eq!(attention_parts(512), 1);
    assert_eq!(attention_parts(513), 2);
    assert_eq!(attention_parts(1), 1);
    let full = body_launches(376, 376);
    assert_eq!(
        full,
        StepTally {
            gemm_ex: 65,
            bias: 65,
            norm: 26,
            residual: 25,
            activation: 8,
            gather: 2,
            place: 16,
            exclude: 8,
            softmax: 16,
            sum_parts: 16,
            strided: 40,
            argmax: 0,
            finite: 0,
            h2d: 0,
            d2h: 0,
            d2d: 0,
            graph_launches: 0,
            graph_captures: 0,
        }
    );
    assert_eq!(launch_total(&full), 287);
    let wide = body_launches(513, 513);
    assert_eq!(wide.strided, 48);
    assert_eq!(wide.exclude, 8);
    assert_eq!(launch_total(&wide), 295);
    let dropped = body_launches(376, 375);
    assert_eq!(dropped.exclude, 16);
    assert_eq!(dropped.strided, 40);
    assert_eq!(launch_total(&dropped), 295);
    assert_eq!(TEACHER_H2D, 12);
    assert_eq!(DEVICE_H2D, 8);
    assert_eq!(ARGMAX_D2H, 4);
    assert_eq!(DEVICE_D2D, 4);
    assert_eq!(capture_tally(376, 376, true).h2d, 12);
    assert_eq!(capture_tally(376, 376, true).d2d, 0);
    assert_eq!(capture_tally(376, 376, true).d2h, 4);
    assert_eq!(capture_tally(376, 376, true).graph_captures, 1);
    assert_eq!(capture_tally(376, 376, true).graph_launches, 0);
    assert_eq!(capture_tally(376, 376, false).h2d, 8);
    assert_eq!(capture_tally(376, 376, false).d2d, 4);
    assert_eq!(replay_tally(true).graph_launches, 1);
    assert_eq!(replay_tally(true).gemm_ex, 0);
    assert_eq!(replay_tally(false).d2d, 4);
    assert_eq!(eager_tally(376, 376, true).graph_captures, 0);
    assert_eq!(eager_tally(376, 376, true).gemm_ex, 65);
    assert_eq!(launch_total(&eager_tally(513, 513, true)), 295);
    assert_eq!(launch_total(&eager_tally(376, 375, true)), 295);
}

fn match_greedy(engine: &Canary, fixtures: &Path, floors: &Value, model: &Path) -> Result<()> {
    engine.reset_encode_calls();
    engine.reset_cross_writes();
    engine.set_graph(true);
    ensure!(
        engine.encode_calls() == 0 && engine.cross_writes() == 0,
        "counters did not reset"
    );
    let mut defects = Vec::new();
    let mut eager_opens = 0u64;
    for case in ["short", "edges", "production"] {
        let items = read_numeric(&fixtures.join("numeric").join(format!("{case}.safetensors")))?;
        let (max_bound, rmse_bound) = (
            floors["cases"][case]["max_abs"]
                .as_f64()
                .with_context(|| format!("missing {case} max_abs"))?,
            floors["cases"][case]["rmse"]
                .as_f64()
                .with_context(|| format!("missing {case} rmse"))?,
        );
        for (index, item) in items.iter().enumerate() {
            ensure!(
                item.logits.iter().all(|value| value.is_finite()),
                "{case}/{index} non-finite fixture"
            );
            let rows = item.encoder.len() / HIDDEN;
            ensure!(
                item.encoder.len() == rows * HIDDEN && rows > 0,
                "{case}/{index} encoder rows"
            );
            if case == "production" {
                ensure!(rows == 376, "{case}/{index} encoder rows {rows}");
            }
            let rows_u = u32::try_from(rows).context("encoder rows")?;
            let valid_u = u32::try_from(item.valid_rows).context("encoder valid")?;
            let before = engine.cross_writes();
            let logits =
                engine.cached_logits(&item.encoder, item.valid_rows, &item.tokens, PREFILL)?;
            ensure!(engine.encode_calls() == 0, "{case}/{index} encode calls");
            ensure!(
                cross_delta(engine, before)? == 16,
                "{case}/{index} cross writes"
            );
            let steps = engine.take_steps();
            ensure!(
                steps.len() == SEQUENCE - PREFILL,
                "{case}/{index} graph steps {}",
                steps.len()
            );
            assert_graph_steps(
                &format!("CANARY_GREEDY {case}/{index}"),
                &steps,
                rows_u,
                valid_u,
                true,
            )?;
            within(
                &mut defects,
                &format!("CANARY_GREEDY {case}/{index}"),
                &logits,
                &item.logits,
                max_bound,
                rmse_bound,
            )?;
            check_ids(
                &mut defects,
                &format!("CANARY_GREEDY {case}/{index}"),
                &logits,
                &floors["cases"][case]["items"][index],
            )?;
            let before = engine.cross_writes();
            engine.set_graph(false);
            let eager =
                engine.cached_logits(&item.encoder, item.valid_rows, &item.tokens, PREFILL)?;
            ensure!(
                cross_delta(engine, before)? == 16,
                "{case}/{index} eager cross writes"
            );
            ensure!(
                bit_equal(&logits, &eager),
                "{case}/{index} graph/eager logit mismatch at {}",
                first_mismatch(&logits, &eager)
            );
            let eager_steps = engine.take_steps();
            ensure!(
                eager_steps.len() == SEQUENCE - PREFILL,
                "{case}/{index} eager steps {}",
                eager_steps.len()
            );
            assert_eager_steps(
                &format!("CANARY_GREEDY {case}/{index}"),
                &eager_steps,
                rows_u,
                valid_u,
                true,
            )?;
            engine.set_graph(true);
            eager_opens += 1;
        }
    }
    let production = read_numeric(&fixtures.join("numeric/production.safetensors"))?;
    ensure!(production.len() == 1, "production batch");
    let shift_bytes = std::fs::read(fixtures.join("numeric/production_shift.safetensors"))?;
    let shift_tensors = SafeTensors::deserialize(&shift_bytes)?;
    let clean = f32_values(&shift_tensors, "logits_clean")?;
    ensure!(
        clean.len() == VOCAB && clean.iter().all(|value| value.is_finite()),
        "shift clean logits"
    );
    let before = engine.cross_writes();
    let shifted = engine.cached_step_with_shifted_cross_k(
        &production[0].encoder,
        production[0].valid_rows,
        &production[0].tokens[..PREFILL + 1],
        PREFILL,
    )?;
    ensure!(engine.encode_calls() == 0, "shift encode calls");
    ensure!(cross_delta(engine, before)? == 16, "shift cross writes");
    let shift_steps = engine.take_steps();
    let shift_rows = production[0].encoder.len() / HIDDEN;
    ensure!(shift_rows == 376, "production shift rows {shift_rows}");
    ensure!(shift_steps.len() == 1, "shift steps {}", shift_steps.len());
    assert_graph_steps(
        "CANARY_GREEDY production cross-k-shift",
        &shift_steps,
        u32::try_from(shift_rows).context("encoder rows")?,
        u32::try_from(production[0].valid_rows).context("encoder valid")?,
        true,
    )?;
    far(
        &mut defects,
        "CANARY_GREEDY production cross-k-shift",
        &shifted,
        &clean,
        floors["cases"]["production"]["max_abs"]
            .as_f64()
            .context("production max_abs")?,
        floors["cases"]["production"]["rmse"]
            .as_f64()
            .context("production rmse")?,
    )?;
    let clip0 = read_clip(&fixtures.join("clips/0.safetensors"))?;
    let clip1 = read_clip(&fixtures.join("clips/1.safetensors"))?;
    let clip_de = read_clip(&fixtures.join("clips/0_de.safetensors"))?;
    let encodes = engine.encode_calls();
    let cross = engine.cross_writes();
    let (ids0, _) = engine.transcribe(&clip0.features, clip0.valid_frames, &clip0.prompt)?;
    ensure!(cross_delta(engine, cross)? == 16, "clip 0 cross writes");
    check_device_steps(
        engine,
        "CANARY_GREEDY clip0",
        &ids0,
        clip0.prompt.len(),
        &clip0.encoder,
        clip0.valid_rows,
    )?;
    let cross = engine.cross_writes();
    let (ids1, _) = engine.transcribe(&clip1.features, clip1.valid_frames, &clip1.prompt)?;
    ensure!(engine.encode_calls() - encodes == 2, "clip encode delta");
    ensure!(cross_delta(engine, cross)? == 16, "clip 1 cross writes");
    check_device_steps(
        engine,
        "CANARY_GREEDY clip1",
        &ids1,
        clip1.prompt.len(),
        &clip1.encoder,
        clip1.valid_rows,
    )?;
    check_generated(
        &mut defects,
        "CANARY_GREEDY clip0",
        &ids0,
        &floors["clips"]["0"],
    )?;
    check_generated(
        &mut defects,
        "CANARY_GREEDY clip1",
        &ids1,
        &floors["clips"]["1"],
    )?;
    let text0 = crate::tokenizer::decode(model, &ids0)?;
    let text1 = crate::tokenizer::decode(model, &ids1)?;
    println!("CANARY_GREEDY clip0 text={text0:?}");
    println!("CANARY_GREEDY clip1 text={text1:?}");
    if text0.as_bytes() == text1.as_bytes() {
        defects.push("CANARY_GREEDY clip texts are byte-identical DEFECT".to_string());
    }
    if text0
        != floors["clips"]["0"]["text"]
            .as_str()
            .context("clip 0 text")?
    {
        defects.push(format!("CANARY_GREEDY clip0 text DEFECT actual={text0:?}"));
    }
    if text1
        != floors["clips"]["1"]["text"]
            .as_str()
            .context("clip 1 text")?
    {
        defects.push(format!("CANARY_GREEDY clip1 text DEFECT actual={text1:?}"));
    }
    let encodes = engine.encode_calls();
    let cross = engine.cross_writes();
    let (ids1_again, _) = engine.transcribe(&clip1.features, clip1.valid_frames, &clip1.prompt)?;
    ensure!(
        cross_delta(engine, cross)? == 16,
        "swapped clip 1 cross writes"
    );
    check_device_steps(
        engine,
        "CANARY_GREEDY clip1 order",
        &ids1_again,
        clip1.prompt.len(),
        &clip1.encoder,
        clip1.valid_rows,
    )?;
    let cross = engine.cross_writes();
    let (ids0_again, _) = engine.transcribe(&clip0.features, clip0.valid_frames, &clip0.prompt)?;
    ensure!(engine.encode_calls() - encodes == 2, "swapped encode delta");
    ensure!(
        cross_delta(engine, cross)? == 16,
        "swapped clip 0 cross writes"
    );
    check_device_steps(
        engine,
        "CANARY_GREEDY clip0 order",
        &ids0_again,
        clip0.prompt.len(),
        &clip0.encoder,
        clip0.valid_rows,
    )?;
    if ids0_again != ids0 || ids1_again != ids1 {
        defects.push(format!(
            "CANARY_GREEDY order DEFECT clip0={ids0_again:?} clip1={ids1_again:?}"
        ));
    } else {
        println!("CANARY_GREEDY order PASS");
    }
    let encodes = engine.encode_calls();
    let cross = engine.cross_writes();
    let (de_ids, de_prefill) =
        engine.greedy(&clip_de.encoder, clip_de.valid_rows, &clip_de.prompt)?;
    ensure!(engine.encode_calls() == encodes, "de-prompt encode calls");
    ensure!(cross_delta(engine, cross)? == 16, "de-prompt cross writes");
    check_device_steps(
        engine,
        "CANARY_GREEDY clip0_de",
        &de_ids,
        clip_de.prompt.len(),
        &clip_de.encoder,
        clip_de.valid_rows,
    )?;
    check_generated(
        &mut defects,
        "CANARY_GREEDY clip0_de",
        &de_ids,
        &floors["clips"]["de"],
    )?;
    if de_ids == ids0 {
        defects.push("CANARY_GREEDY language ids are identical DEFECT".to_string());
    }
    let de_text = crate::tokenizer::decode(model, &de_ids)?;
    println!("CANARY_GREEDY clip0_de text={de_text:?}");
    far(
        &mut defects,
        "CANARY_GREEDY language prefill",
        &de_prefill,
        &clip0.scores[..VOCAB],
        floors["clips"]["0"]["prefill_max_abs"]
            .as_f64()
            .context("prefill max_abs")?,
        floors["clips"]["0"]["prefill_rmse"]
            .as_f64()
            .context("prefill rmse")?,
    )?;
    ensure!(
        engine.cross_writes() == (11 + eager_opens) * 16,
        "cross-write total {} eager_opens {eager_opens}",
        engine.cross_writes()
    );
    for defect in &defects {
        println!("{defect}");
    }
    ensure!(
        defects.is_empty(),
        "CANARY_GREEDY defects={}",
        defects.len()
    );
    println!("PASS CANARY_GREEDY cells=5 clips=2 controls=language+cross-k-shift+order");
    Ok(())
}

#[test]
#[ignore = "requires GPU lease, frozen floors, and external fixtures"]
fn canary_greedy_matches_reference() -> Result<()> {
    ensure!(
        std::env::var("FF_CANARY_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model =
        PathBuf::from(std::env::var_os("FF_CANARY_MODEL").context("FF_CANARY_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_CANARY_GREEDY_FIXTURES")
            .context("FF_CANARY_GREEDY_FIXTURES missing")?,
    );
    let floors: Value = serde_json::from_slice(&std::fs::read(
        std::env::var_os("FF_CANARY_GREEDY_FLOORS").context("FF_CANARY_GREEDY_FLOORS missing")?,
    )?)?;
    ensure!(
        floors["schema"] == "n6c-m3-greedy-v1",
        "greedy floor schema"
    );
    let device = Device::new(0)?;
    let engine = Canary::load(&model, &device)?;
    match_greedy(&engine, &fixtures, &floors, &model)
}

