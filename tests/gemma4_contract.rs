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
        pairs: 256,
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

struct Fixtures {
    bytes: Vec<u8>,
}

impl Fixtures {
    fn read(file: &str) -> Result<Self> {
        let dir = std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?;
        Ok(Self {
            bytes: std::fs::read(std::path::Path::new(&dir).join(file))?,
        })
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        let tensors = safetensors::SafeTensors::deserialize(&self.bytes)?;
        let view = tensors.tensor(name)?;
        ensure!(
            view.dtype() == safetensors::Dtype::BF16,
            "{name} is not BF16"
        );
        Ok(view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b).to_f32())
            .collect())
    }
}

struct Layout {
    heads: usize,
    kv_heads: usize,
    d: usize,
}

/// REFERENCE_ATTENTION: test-side F32/F64 attention over one layer's bound K/V `[kv_heads, keys, d]`; no attention kernel is launched.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layout: &Layout,
    positions: &[usize],
    window: Option<usize>,
    scale: f64,
) -> Vec<f32> {
    let Layout { heads, kv_heads, d } = *layout;
    let group = heads / kv_heads;
    let span = k.len() / (kv_heads * d);
    let mut out = vec![0f32; positions.len() * heads * d];
    for (row, &i) in positions.iter().enumerate() {
        let lo = window.map_or(0, |w| (i + 1).saturating_sub(w));
        for h in 0..heads {
            let kv = h / group;
            let query = &q[(i * heads + h) * d..(i * heads + h + 1) * d];
            let scores: Vec<f64> = (lo..=i)
                .map(|j| {
                    let key = &k[(kv * span + j) * d..(kv * span + j + 1) * d];
                    query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f64 = weights.iter().sum();
            for e in 0..d {
                let acc: f64 = (lo..=i)
                    .zip(&weights)
                    .map(|(j, w)| w * f64::from(v[(kv * span + j) * d + e]))
                    .sum();
                out[(row * heads + h) * d + e] = (acc / total) as f32;
            }
        }
    }
    out
}

/// Frozen bound (PLAN G4-S2a §4): cosine >= 0.9999 and max abs error <= 2^-6 * max|reference|, per head.
fn heads_pass(got: &[f32], reference: &[f32], layout: &Layout, rows: usize) -> Vec<bool> {
    let Layout { heads, d, .. } = *layout;
    (0..heads)
        .map(|h| {
            let (mut dot, mut a, mut b, mut worst, mut peak) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for row in 0..rows {
                for e in 0..d {
                    let (x, y) = (
                        f64::from(got[(row * heads + h) * d + e]),
                        f64::from(reference[(row * heads + h) * d + e]),
                    );
                    dot += x * y;
                    a += x * x;
                    b += y * y;
                    worst = worst.max((x - y).abs());
                    peak = peak.max(y.abs());
                }
            }
            let cosine = dot / (a.sqrt() * b.sqrt());
            cosine.is_finite() && cosine >= 0.9999 && worst <= peak / 64.0
        })
        .collect()
}

#[test]
#[ignore = "CPU-only; requires FF_GEMMA4_FIXTURES"]
fn kv_sharing_window_and_scale_match_the_definition_reference_attention() -> Result<()> {
    let fixtures = Fixtures::read("attn.safetensors")?;
    let positions: Vec<usize> = [0usize, 1, 255, 511, 512, 513, 1023, 1024, 1025, 1535]
        .into_iter()
        .chain((0..17).map(|k| 520 + 61 * k))
        .collect();
    const T: usize = 1536;
    let (mut defects, mut undetected, mut cases) = (0, 0, 0);
    for (tag, text, sample) in [
        ("e2b", E2B, [0usize, 4, 13, 14, 15, 19, 34]),
        ("e4b", E4B, [0usize, 5, 22, 23, 24, 29, 41]),
    ] {
        let (_, spec) = read(text)?;
        let heads = spec.q_heads as usize;
        let kv_heads = spec.kv_heads as usize;
        let mut slots: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; spec.layers.len()];
        for &l in &sample {
            if !geometry(&spec, l)?.kv_reader {
                slots[l] = Some((
                    fixtures.tensor(&format!("att.{tag}.l{l}.k"))?,
                    fixtures.tensor(&format!("att.{tag}.l{l}.v"))?,
                ));
            }
        }
        for &l in &sample {
            let g = geometry(&spec, l)?;
            let layout = Layout {
                heads,
                kv_heads,
                d: g.head_dim as usize,
            };
            let q = fixtures.tensor(&format!("att.{tag}.l{l}.q"))?;
            let reference = fixtures.tensor(&format!("att.{tag}.l{l}.out"))?;
            ensure!(q.len() == T * heads * layout.d, "{tag} l{l} q extent");
            let window = spec.layers[l].sliding_window.map(|w| w as usize);
            let bound = |slot: usize| -> Result<&(Vec<f32>, Vec<f32>)> {
                slots[slot]
                    .as_ref()
                    .with_context(|| format!("{tag} layer {l}: slot {slot} was never written"))
            };
            let run = |slot: usize, window: Option<usize>, scale: f64| -> Result<Vec<bool>> {
                let (k, v) = bound(slot)?;
                let got = reference_attention(&q, k, v, &layout, &positions, window, scale);
                Ok(heads_pass(&got, &reference, &layout, positions.len()))
            };
            let slot = spec.layers[l].state_index as usize;
            let ok = run(slot, window, 1.0)?;
            println!(
                "GEMMA4_ATTENTION REFERENCE_ATTENTION {tag} layer={l} kind={} reader={} slot={slot} heads_passing={}/{}",
                if window.is_some() { "sliding" } else { "full" },
                g.kv_reader,
                ok.iter().filter(|x| **x).count(),
                heads
            );
            defects += usize::from(!ok.iter().all(|x| *x));
            let mut mutants: Vec<(&str, Vec<bool>)> = vec![(
                "scale_inv_sqrt_d",
                run(slot, window, 1.0 / (layout.d as f64).sqrt())?,
            )];
            if let Some(w) = window {
                mutants.push(("window_minus_1", run(slot, Some(w - 1), 1.0)?));
                mutants.push(("window_plus_1", run(slot, Some(w + 1), 1.0)?));
            }
            if g.kv_reader {
                let other = sample
                    .iter()
                    .copied()
                    .find(|&o| {
                        o != slot
                            && geometry(&spec, o)
                                .is_ok_and(|x| !x.kv_reader && x.head_dim == g.head_dim)
                    })
                    .context("no second owner of the same layer type in the sample")?;
                mutants.push(("reader_bound_to_other_owner", run(other, window, 1.0)?));
                let zeros = (
                    vec![0f32; T * kv_heads * layout.d],
                    vec![0f32; T * kv_heads * layout.d],
                );
                let got =
                    reference_attention(&q, &zeros.0, &zeros.1, &layout, &positions, window, 1.0);
                mutants.push((
                    "reader_bound_to_own_unwritten_slot",
                    heads_pass(&got, &reference, &layout, positions.len()),
                ));
            }
            for (name, passes) in mutants {
                let failing = passes.iter().filter(|x| !**x).count();
                println!(
                    "GEMMA4_ATTENTION_MUTANT {name} {tag} layer={l} heads_failing={failing}/{heads}"
                );
                undetected += usize::from(failing == 0);
            }
            cases += 1;
        }
    }
    println!("GEMMA4_ATTENTION cases={cases} defects={defects} undetected_mutants={undetected}");
    ensure!(defects == 0 && undetected == 0, "GEMMA4_ATTENTION FAIL");
    Ok(())
}
