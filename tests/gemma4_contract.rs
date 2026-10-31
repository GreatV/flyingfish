//! Gemma 4 model-contract gates.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::Operator,
    model::{LayerStage, layer_program},
    model_spec::{Architecture, FfnKind, LayerGeometry, Mixer, ModelSpec, RopeSpec, VisionConfig},
    vision_input::{AudioClip, ImageGrid, MultimodalInput},
};
use serde_json::{Value, json};

const E2B: &str = include_str!("data/gemma4-e2b-config.json");
const E4B: &str = include_str!("data/gemma4-e4b-config.json");

#[test]
fn large_gemma_geometry_preserves_global_kv_and_absent_per_layer_inputs() -> Result<()> {
    for (text, hidden, depth, heads, kv, global_kv) in [
        (
            include_str!("data/gemma4-12b-config.json"),
            3840,
            48,
            16,
            8,
            1,
        ),
        (
            include_str!("data/gemma4-31b-config.json"),
            5376,
            60,
            32,
            16,
            4,
        ),
    ] {
        let (root, spec) = read(text)?;
        ensure!(
            spec.hidden == hidden
                && spec.layers.len() == depth
                && spec.q_heads == heads
                && spec.kv_heads == kv
                && spec.per_layer_input.is_none()
                && spec.audio.is_none(),
            "large Gemma geometry"
        );
        let vision = spec.vision.as_ref().context("large Gemma vision missing")?;
        if hidden == 3840 {
            ensure!(
                vision.unified.is_some()
                    && vision.axial.is_none()
                    && vision.patch_elements(spec.architecture)? == 6912,
                "unified raw patch geometry"
            );
            let audio = spec.audio.as_ref().context("raw audio missing")?;
            ensure!(
                [audio.frame, audio.output] == [640, 3840],
                "raw frame projection geometry"
            );
        } else {
            let axial = vision.axial.as_ref().context("31B axial config missing")?;
            ensure!(
                [vision.hidden, vision.depth, vision.heads, axial.head_dim] == [1152, 27, 16, 72]
                    && axial.standardize
                    && !axial.clipped_linears
                    && spec.audio.is_none(),
                "31B media geometry"
            );
        }
        let work = flyingfish::work_layout::RowWorkSpec::new(&spec)?;
        let expected_projection = if hidden == 3840 { 8704 } else { 18432 };
        ensure!(
            work.projection == expected_projection && work.mixer == heads as usize * 512,
            "large Gemma projection/work bounds: {work:?}"
        );
        for (index, layer) in spec.layers.iter().enumerate() {
            let geometry = geometry(&spec, index)?;
            let full = root["text_config"]["layer_types"][index] == "full_attention";
            ensure!(
                geometry.kv_heads == full.then_some(global_kv)
                    && geometry.head_dim == if full { 512 } else { 256 }
                    && !geometry.kv_reader
                    && layer.state_index as usize == index
                    && layer.sliding_window == (!full).then_some(1024)
                    && layer_program(&spec, index)?.len() == 5,
                "layer {index}: {layer:?}"
            );
        }
        for (key, value) in [
            ("num_global_key_value_heads", json!(3)),
            ("num_kv_shared_layers", json!(1)),
            ("attention_k_eq_v", json!(false)),
            ("hidden_size_per_layer_input", json!(256)),
        ] {
            let mut mutant = root.clone();
            mutant["text_config"][key] = value;
            ensure!(
                ModelSpec::gemma4(&mutant, &Value::Null).is_err(),
                "accepted {key} mutant"
            );
        }
    }
    Ok(())
}

#[test]
fn audio_masks_and_mixed_splices_preserve_token_identity() -> Result<()> {
    for text in [E2B, E4B] {
        let (root, spec) = read(text)?;
        let a = spec.audio.as_ref().context("audio config")?;
        for frames in [1usize, 3, 4, 5, 47, 48, 49, 3000] {
            let features = vec![0.25; frames * 128];
            let mut mask = vec![true; frames];
            if frames > 1 {
                mask[frames - 1] = false;
            }
            let count = mask.iter().step_by(4).filter(|&&v| v).count();
            let audio = [AudioClip {
                features: &features,
                mask: &mask,
            }];
            let mut ids = vec![a.start_token];
            ids.extend(std::iter::repeat_n(a.token, count));
            ids.push(a.end_token);
            let mut modalities = vec![3; ids.len()];
            modalities[0] = 0;
            modalities[count + 1] = 0;
            let input = MultimodalInput {
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &[],
                grids: &[],
                audio: &audio,
            };
            let plan = input.validate(&spec, 1024)?;
            ensure!(
                plan.feature_rows() == count && plan.audio_slots()[0].tokens == (1..count + 1),
                "audio mask stride changed"
            );
            let ranges = plan.splices(1..count + 1)?;
            ensure!(
                ranges.len() == 1 && ranges[0].feature_rows == (0..count),
                "audio splice changed"
            );
            let mut wrong = modalities.clone();
            wrong[1] = 0;
            ensure!(
                MultimodalInput {
                    modality_ids: &wrong,
                    ..input
                }
                .validate(&spec, 1024)
                .is_err(),
                "audio placeholder marked text was accepted"
            );
        }
        let v = spec.vision.as_ref().context("vision config")?;
        let features = vec![0.5; 9 * 128];
        let mask = [true; 9];
        let audio = [AudioClip {
            features: &features,
            mask: &mask,
        }];
        let grid = [ImageGrid { t: 1, h: 3, w: 3 }];
        let pixels = vec![0.5; 9 * 768];
        let ids = [
            a.start_token,
            a.token,
            a.token,
            a.token,
            a.end_token,
            v.start_token,
            v.image_token,
            v.end_token,
            10,
        ];
        let modalities = [0, 3, 3, 3, 0, 0, 1, 0, 0];
        let input = MultimodalInput {
            input_ids: &ids,
            modality_ids: &modalities,
            pixel_values: &pixels,
            grids: &grid,
            audio: &audio,
        };
        let plan = input.validate(&spec, 32)?;
        ensure!(
            plan.audio_slots()[0].features == (0..3)
                && plan.slots()[0].features == (3..4)
                && plan.feature_rows() == 4
                && plan.axes()[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "mixed media order changed"
        );
        let splices = plan.splices(2..8)?;
        ensure!(
            splices.contains(&flyingfish::vision_input::SpliceRange {
                chunk_rows: 0..2,
                feature_rows: 1..3
            }) && splices.contains(&flyingfish::vision_input::SpliceRange {
                chunk_rows: 4..5,
                feature_rows: 3..4
            }),
            "mixed chunk splice misaligned"
        );
        ensure!(
            MultimodalInput {
                audio: &[],
                ..input
            }
            .validate(&spec, 32)
            .is_err(),
            "missing audio accepted"
        );
        for key in [
            "conv_kernel_size",
            "attention_context_left",
            "output_proj_dims",
        ] {
            let mut wrong = root.clone();
            wrong["audio_config"][key] = json!(7);
            ensure!(
                ModelSpec::gemma4(&wrong, &Value::Null).is_err(),
                "unsupported audio geometry accepted"
            );
        }
#[test]
fn unified_media_shapes_and_packed_patch_bounds_are_strict() -> Result<()> {
    let (root, spec) = read(include_str!("data/gemma4-12b-config.json"))?;
    let ids = [2, 255999, 258880, 258880, 258882, 4];
    let modality = [0, 0, 1, 1, 0, 0];
    let pixels = vec![0.5; 2 * 6912];
    let grid = [ImageGrid { t: 1, h: 1, w: 2 }];
    let input = MultimodalInput {
        input_ids: &ids,
        modality_ids: &modality,
        pixel_values: &pixels,
        grids: &grid,
    };
    let plan = input.validate(&spec, 32)?;
    ensure!(
        plan.patch_rows() == 2 && plan.feature_rows() == 2,
        "unified patch rows were pooled twice"
    );
    let blocks = plan.image_blocks()?;
    ensure!(
        blocks[..12] == [1, 1, 2, 2, 2, 4, 2, 4, 5, 5, 6, 6],
        "image-block query bounds"
    );
    ensure!(
        plan.chunk_end(0, 3, 6)? == 2
            && plan.chunk_end(2, 3, 6)? == 5
            && plan.chunk_end(2, 1, 6).is_err()
            && plan.chunk_end(3, 3, 6).is_err(),
        "image blocks split across chunks"
    );
    let short = MultimodalInput {
        pixel_values: &pixels[..pixels.len() - 1],
        ..input
    };
    ensure!(
        short.validate(&spec, 32).is_err(),
        "short raw patch accepted"
    );
    for (scope, key, value) in [
        ("vision_config", "mm_embed_dim", 1152),
        ("vision_config", "model_patch_size", 16),
        ("vision_config", "mm_posemb_size", 280),
        ("audio_config", "audio_samples_per_token", 320),
        ("audio_config", "output_proj_dims", 3840),
    ] {
        let mut mutant = root.clone();
        mutant[scope][key] = json!(value);
        ensure!(
            ModelSpec::gemma4(&mutant, &Value::Null).is_err(),
            "accepted {scope}.{key} mutation"
        );
    }
    Ok(())
}

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
fn axial_vision_preserves_geometry_and_image_slot_bounds() -> Result<()> {
    for text in [E2B, E4B] {
        let (_, spec) = read(text)?;
        let vision = spec.vision.as_ref().context("missing vision spec")?;
        let axial = vision.axial.as_ref().context("missing axial spec")?;
        ensure!(
            [
                vision.hidden,
                vision.depth,
                vision.heads,
                axial.head_dim,
                vision.intermediate
            ] == [768, 16, 12, 64, 3072]
                && vision.output == spec.hidden
                && vision.position_side == 10240
                && axial.soft_tokens == 280
                && axial.clipped_linears
                && !axial.standardize,
            "published vision geometry changed"
        );
        for (h, w) in [(3u32, 9u32), (48, 48), (42, 60)] {
            let features = (h * w / 9) as usize;
            let mut ids = vec![2, vision.start_token];
            ids.extend(std::iter::repeat_n(vision.image_token, features));
            ids.extend([vision.end_token, 17, 19]);
            let mut modalities = vec![0; ids.len()];
            modalities[2..features + 2].fill(1);
            let pixels = vec![0.5; (h * w * 768) as usize];
            let grids = [ImageGrid { t: 1, h, w }];
            let input = MultimodalInput {
                audio: &[],
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &pixels,
                grids: &grids,
            };
            let capacity = ids.len() + 11;
            let plan = input.validate(&spec, capacity)?;
            ensure!(
                plan.feature_rows() == features
                    && plan.patch_rows() == (h * w) as usize
                    && plan.slots()[0].tokens == (2..features + 2)
                    && plan.rope_delta() == 0
                    && plan
                        .axes()
                        .iter()
                        .all(|axis| axis.iter().enumerate().all(|(i, &p)| p == i as i32))
                    && plan.decode_position(ids.len())? == ids.len() as i32,
                "image placement or scalar text positions changed"
            );
            let splice = plan.splices(3..features + 3)?;
            ensure!(
                splice.len() == 1
                    && splice[0].chunk_rows == (0..features - 1)
                    && splice[0].feature_rows == (1..features),
                "chunk splice bounds changed"
            );
            let mut wrong_ids = ids.clone();
            wrong_ids[features + 1] = 17;
            ensure!(
                MultimodalInput {
                    audio: &[],
                    input_ids: &wrong_ids,
                    ..input
                }
                .validate(&spec, capacity)
                .is_err(),
                "wrong image placeholder was accepted"
            );
            ensure!(
                MultimodalInput {
                    audio: &[],
                    pixel_values: &pixels[..pixels.len() - 1],
                    ..input
                }
                .validate(&spec, capacity)
                .is_err(),
                "short pixel buffer was accepted"
            );
            let wrong_grid = [ImageGrid {
                h: h + 1,
                ..grids[0]
            }];
            ensure!(
                MultimodalInput {
                    audio: &[],
                    grids: &wrong_grid,
                    ..input
                }
                .validate(&spec, capacity)
                .is_err(),
                "misaligned pooling grid was accepted"
            );
            ensure!(
                input.validate(&spec, ids.len() - 1).is_err(),
                "short KV capacity was accepted"
            );
        }
    }
    Ok(())
}

#[test]
fn axial_vision_parser_accepts_diffusion_geometry_without_text_support() -> Result<()> {
    let (mut root, _) = read(E2B)?;
    let vision = root.get_mut("vision_config").context("vision_config")?;
    for (key, value) in [
        ("hidden_size", 1152),
        ("intermediate_size", 4304),
        ("num_hidden_layers", 27),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 16),
        ("head_dim", 72),
        ("global_head_dim", 72),
    ] {
        vision[key] = json!(value);
    }
    vision["standardize"] = json!(true);
    vision["use_clipped_linears"] = json!(false);
    root["text_config"]["hidden_size"] = json!(2816);
    root.as_object_mut()
        .context("config object")?
        .remove("video_token_id");
    let parsed = VisionConfig::from_root(&root)?.context("vision spec")?;
    let axial = parsed.axial.as_ref().context("axial spec")?;
    ensure!(
        [
            parsed.hidden,
            parsed.intermediate,
            parsed.depth,
            parsed.heads,
            axial.head_dim,
            parsed.output
        ] == [1152, 4304, 27, 16, 72, 2816]
            && axial.standardize
            && !axial.clipped_linears,
        "DiffusionGemma vision parameters were specialized away"
    );
    for (key, value) in [
        ("num_key_value_heads", json!(8)),
        ("head_dim", json!(64)),
        ("pooling_kernel_size", json!(0)),
        ("rms_norm_eps", json!(0.0)),
        ("standardize", json!(null)),
    ] {
        let mut mutant = root.clone();
        mutant["vision_config"][key] = value;
        ensure!(
            VisionConfig::from_root(&mutant).is_err(),
            "invalid vision geometry was accepted: {key}"
        );
    }
    Ok(())
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
