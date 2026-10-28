//! Gemma 4 model-contract gates.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::Operator,
    model::{LayerStage, layer_program},
    model_spec::{Architecture, FfnKind, LayerGeometry, Mixer, ModelSpec, RopeSpec},
};
use serde_json::{Value, json};

const E2B: &str = include_str!("data/gemma4-e2b-config.json");
const E4B: &str = include_str!("data/gemma4-e4b-config.json");

fn read(text: &str) -> Result<(Value, ModelSpec)> {
    let root: Value = serde_json::from_str(text)?;
    let spec = ModelSpec::gemma4(&root, &json!({"eos_token_id": 1}))?;
    Ok((root, spec))
}

fn geometry(spec: &ModelSpec, layer: usize) -> Result<LayerGeometry> {
    spec.layers[layer]
        .geometry
        .with_context(|| format!("layer {layer} has no geometry"))
}

struct Expect {
    hidden: u32,
    intermediate: u32,
    layers: usize,
    kv_heads: u32,
    first_shared: usize,
    double_wide: bool,
    full: &'static [usize],
    publishers: [usize; 2],
}

fn check(spec: &ModelSpec, e: &Expect) -> Result<()> {
    ensure!(
        spec.architecture == Architecture::Gemma4
            && spec.hidden == e.hidden
            && spec.intermediate == e.intermediate
            && spec.vocab == 262144
            && spec.q_heads == 8
            && spec.kv_heads == e.kv_heads
            && spec.max_position == 131072
            && spec.eos == [1]
            && spec.layers.len() == e.layers,
        "normalized Gemma 4 geometry: {spec:?}"
    );
    ensure!(
        spec.logit_softcap == Some(30.0) && spec.per_layer_input == Some(256),
        "Gemma 4 head configuration"
    );
    ensure!(
        spec.config().tie_word_embeddings && spec.config().hidden_act == "gelu_pytorch_tanh",
        "Gemma 4 Config view"
    );
    let sliding = RopeSpec {
        theta: 10000.0,
        pairs: 128,
        pair_stride: 128,
        denominator: 256,
    };
    let global = RopeSpec {
        theta: 1000000.0,
        pairs: 64,
        pair_stride: 256,
        denominator: 512,
    };
    for (i, layer) in spec.layers.iter().enumerate() {
        let full = e.full.contains(&i);
        let reader = i >= e.first_shared;
        let g = geometry(spec, i)?;
        let publisher = e.publishers[usize::from(full)];
        ensure!(
            layer.rope
                && layer.ffn == FfnKind::Dense
                && layer.sliding_window == (!full).then_some(512)
                && g.head_dim == if full { 512 } else { 256 }
                && g.rope == if full { global } else { sliding }
                && g.kv_reader == reader
                && layer.state_index == if reader { publisher as u32 } else { i as u32 }
                && g.intermediate
                    == if reader && e.double_wide {
                        2 * e.intermediate
                    } else {
                        e.intermediate
                    },
            "layer {i}: {layer:?}"
        );
    }
    Ok(())
}

#[test]
fn gemma4_e2b_config_normalizes_the_published_layer_plan() -> Result<()> {
    let (root, spec) = read(E2B)?;
    check(
        &spec,
        &Expect {
            hidden: 1536,
            intermediate: 6144,
            layers: 35,
            kv_heads: 1,
            first_shared: 15,
            double_wide: true,
            full: &[4, 9, 14, 19, 24, 29, 34],
            publishers: [13, 14],
        },
    )?;
    let mut shifted = root.clone();
    shifted["text_config"]["num_kv_shared_layers"] = json!(19);
    let moved = ModelSpec::gemma4(&shifted, &Value::Null)?;
    ensure!(
        moved.layers[16].state_index == 15 && spec.layers[16].state_index == 13,
        "shared-layer count must move the sliding publisher"
    );
    Ok(())
}

#[test]
fn gemma4_e4b_config_normalizes_the_published_layer_plan() -> Result<()> {
    let (_, spec) = read(E4B)?;
    check(
        &spec,
        &Expect {
            hidden: 2560,
            intermediate: 10240,
            layers: 42,
            kv_heads: 2,
            first_shared: 24,
            double_wide: false,
            full: &[5, 11, 17, 23, 29, 35, 41],
            publishers: [22, 23],
        },
    )
}

#[test]
fn gemma4_reader_rejects_unsupported_configurations() -> Result<()> {
    let (root, _) = read(E2B)?;
    let text = |key: &str, value: Value| {
        let mut mutant = root.clone();
        mutant["text_config"][key] = value;
        mutant
    };
    let mut proportional = root.clone();
    proportional["text_config"]["rope_parameters"]["full_attention"]["rope_type"] =
        json!("default");
    let mut orphan = root.clone();
    for kind in orphan["text_config"]["layer_types"]
        .as_array_mut()
        .context("layer_types")?
        .iter_mut()
        .take(15)
    {
        *kind = json!("sliding_attention");
    }
    for (label, mutant) in [
        ("k_eq_v", text("attention_k_eq_v", json!(true))),
        ("moe", text("enable_moe_block", json!(true))),
        ("activation", text("hidden_activation", json!("silu"))),
        ("softcap", text("final_logit_softcapping", json!(50.0))),
        ("no owner", text("num_kv_shared_layers", json!(35))),
        ("unknown key", text("unknown_field", json!(1))),
        ("rope type", proportional),
        ("no publisher", orphan),
    ] {
        ensure!(
            ModelSpec::gemma4(&mutant, &Value::Null).is_err(),
            "{label} mutant was accepted"
        );
    }
    Ok(())
}

#[test]
fn existing_specs_serialize_without_the_gemma4_fields() -> Result<()> {
    let root: Value = serde_json::from_str(include_str!("data/flint-config.json"))?;
    let spec = ModelSpec::flint(&root, &json!({"eos_token_id": [150001]}))?;
    let text = serde_json::to_string(&spec)?;
    ensure!(
        !text.contains("geometry")
            && !text.contains("logit_softcap")
            && !text.contains("per_layer_input"),
        "Flint spec serialization gained Gemma 4 fields"
    );
    Ok(())
}

#[test]
fn per_layer_input_stage_follows_the_ffn_only_when_the_spec_carries_it() -> Result<()> {
    for text in [E2B, E4B] {
        let (_, mut spec) = read(text)?;
        for layer in 0..spec.layers.len() {
            let program = layer_program(&spec, layer)?;
            ensure!(
                program.len() == 6
                    && matches!(program[0], LayerStage::Common(Operator::Norm))
                    && matches!(program[1], LayerStage::Mixer(Mixer::FullAttention))
                    && matches!(program[2], LayerStage::Common(Operator::PostNorm))
                    && matches!(program[3], LayerStage::Ffn)
                    && matches!(program[4], LayerStage::PerLayerInput)
                    && matches!(program[5], LayerStage::Common(Operator::NextNorm)),
                "layer {layer}: {program:?}"
            );
        }
        spec.per_layer_input = None;
        let program = layer_program(&spec, 0)?;
        ensure!(
            program.len() == 5
                && !program
                    .iter()
                    .any(|stage| matches!(stage, LayerStage::PerLayerInput))
                && matches!(program[4], LayerStage::Common(Operator::NextNorm)),
            "specs without a per-layer input width keep the five-stage program: {program:?}"
        );
    }
    let root: Value = serde_json::from_str(include_str!("data/flint-config.json"))?;
    let flint = ModelSpec::flint(&root, &json!({"eos_token_id": [150001]}))?;
    ensure!(
        layer_program(&flint, 0)?.len() == 5,
        "Flint program arity changed"
    );
    Ok(())
}
