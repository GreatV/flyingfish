//! MiniMax-H3 FL2VA geometry, layout, schedule and request-producer gates.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    checkpoint::MappedCheckpoint,
    minimax_h3::{
        AUDIO_TAG, AdalnPlan, DecodePlan, Geometry, H3Spec, KeyframeAnchor, PackedLayout,
        RowTimesteps, Schedule, TEXT_TAG, VIDEO_TAG, VaeSpec, VaeTiles, row_timestep_plan,
    },
};
use safetensors::{Dtype, SafeTensors};
use std::{env, fs, path::PathBuf};

/// Turbo 8-NFE recipe: 9 grid points, video shift 6, audio shift 3.
const GRID_POINTS: usize = 9;
const VIDEO_SHIFT: f64 = 6.0;
const AUDIO_SHIFT: f64 = 3.0;

fn spec() -> Result<H3Spec> {
    H3Spec::parse(&serde_json::from_str(include_str!(
        "data/minimax-h3-transformer-config.json"
    ))?)
}

/// 6 label rows, a 1,010-row vision block tagged video, and a 311-token prompt.
fn text_tags() -> Vec<u8> {
    let mut tags = vec![TEXT_TAG; 6];
    tags.extend([VIDEO_TAG; 1010]);
    tags.extend([TEXT_TAG; 311]);
    tags
}

fn f64_bits(position: [f64; 3]) -> [u64; 3] {
    position.map(f64::to_bits)
}

#[test]
fn minimax_h3_config_normalizes_the_published_transformer() -> Result<()> {
    let spec = spec()?;
    ensure!(
        spec.hidden == 5376
            && spec.heads == 56
            && spec.head_dim == 128
            && spec.layers == 50
            && spec.refiner_layers == 2
            && spec.ffn == 14336
            && spec.video_row_width() == 96
            && spec.audio_in_channels == 32
            && spec.text_dim == 5120
            && spec.time_dim == 2688
            && spec.rope_freq_dim == 16,
        "unexpected MiniMax-H3 geometry: {spec:?}"
    );
    ensure!(spec.block_linears().len() == 312, "block linear count");
    let mut unknown: serde_json::Value =
        serde_json::from_str(include_str!("data/minimax-h3-transformer-config.json"))?;
    unknown["sliding_window"] = 1.into();
    ensure!(H3Spec::parse(&unknown).is_err(), "accepted an unknown key");
    Ok(())
}

/// Video VAE decode geometry equals the model definition's `_split_tiles` / `_decode` (tile starts and overlaps at
/// 768 x 1344; output frames per latent length, read from `_decode` on shape-only clips, including padded lengths).
#[test]
fn minimax_h3_vae_decode_geometry_equals_the_model_definition() -> Result<()> {
    let vae = VaeSpec::parse(&serde_json::from_str(include_str!(
        "data/minimax-h3-vae-config.json"
    ))?)?;
    assert_eq!(
        (
            vae.dim,
            vae.layers,
            vae.heads,
            vae.head_dim,
            vae.registers,
            vae.ffn,
            vae.rope_dims
        ),
        (2048, 36, 32, 64, 4, 8192, 48)
    );
    assert_eq!(vae.patch_width(), 3072);
    assert_eq!(
        (
            vae.chunk_latents(),
            vae.overlap_latents(),
            vae.frame_pre_padding(),
            vae.frame_overlap()
        ),
        (5, 2, 3, 5)
    );
    let tiles = VaeTiles::released(&vae, 768, 1344)?;
    assert_eq!((tiles.height, tiles.width, tiles.count()), (256, 256, 28));
    assert_eq!(tiles.rows, (vec![0, 160, 336, 512], vec![96, 80, 80]));
    assert_eq!(
        tiles.cols,
        (
            vec![0, 176, 352, 528, 704, 896, 1088],
            vec![80, 80, 80, 80, 64, 64]
        )
    );
    for (latents, frames) in [
        (3, 9),
        (7, 22),
        (12, 39),
        (36, 120),
        (37, 124),
        (38, 128),
        (39, 132),
        (40, 136),
        (41, 137),
        (57, 192),
    ] {
        assert_eq!(
            DecodePlan::new(&vae, latents)?.frames,
            frames,
            "{latents} latents"
        );
    }
    assert_eq!(DecodePlan::new(&vae, 37)?.chunks, 7);
    assert_eq!(DecodePlan::new(&vae, 57)?.chunks, 11);
    assert!(DecodePlan::new(&vae, 2).is_err());
    let spec = spec()?;
    assert_eq!(Geometry::resolve(&spec, 768, 1344, 120)?.frames, 124);
    for requested in 120..=345 {
        let geometry = Geometry::resolve(&spec, 768, 1344, requested)?;
        assert_eq!(
            DecodePlan::new(&vae, geometry.latent_frames)?.frames,
            geometry.frames,
            "{requested} requested frames"
        );
    }
    Ok(())
}

#[test]
fn minimax_h3_schedule_reproduces_the_reference_grids() -> Result<()> {
    let expected = |bits: [u32; 9]| bits.map(f32::from_bits).to_vec();
    let video = Schedule::new(GRID_POINTS, VIDEO_SHIFT)?;
    let audio = Schedule::new(GRID_POINTS, AUDIO_SHIFT)?;
    ensure!(
        video.sigmas
            == expected([
                0x3f800000, 0x3f7a0be8, 0x3f7286bd, 0x3f68ba2f, 0x3f5b6db7, 0x3f48590b, 0x3f2aaaab,
                0x3eec4ec5, 0
            ]),
        "video sigmas {:?}",
        video.sigmas
    );
    ensure!(
        audio.sigmas
            == expected([
                0x3f800000, 0x3f745d17, 0x3f666666, 0x3f555555, 0x3f400000, 0x3f249249, 0x3f000000,
                0x3e99999a, 0
            ]),
        "audio sigmas {:?}",
        audio.sigmas
    );
    ensure!(
        Schedule::new(GRID_POINTS, 12.0)?.sigmas[1] == f32::from_bits(0x3f7cfcfd),
        "shift 12 grid"
    );
    ensure!(
        Schedule::new(16, VIDEO_SHIFT).is_err(),
        "accepted a grid outside the scalar linspace form"
    );
    Ok(())
}

#[test]
fn minimax_h3_layout_reproduces_the_reference_rows() -> Result<()> {
    let spec = spec()?;
    let tags = text_tags();
    let small = Geometry::resolve(&spec, 768, 1344, 124)?;
    let production = Geometry::resolve(&spec, 768, 1344, 192)?;
    ensure!(
        (small.frames, small.latent_frames, small.audio_latents) == (124, 37, 207)
            && (
                production.frames,
                production.latent_frames,
                production.audio_latents
            ) == (192, 57, 320)
            && (small.latent_height, small.latent_width) == (48, 84),
        "geometry {small:?} {production:?}"
    );
    ensure!(
        Geometry::resolve(&spec, 768, 1344, 22).is_err(),
        "accepted a clip shorter than 5 s"
    );
    let (h0, h23) = (0x400f3db865602068u64, 0x403b1642df0cfc49u64);
    let (w0, w5, w41) = (
        0xc014a9fea74be3a4u64,
        0x4027efcf5dc802b2u64,
        0x4042143ccac5fc9eu64,
    );
    let origin = 0x4094bc0000000000u64;
    let zero = 0u64;

    let first = PackedLayout::build(&spec, &small, &tags, &[KeyframeAnchor::First])?;
    let p = |layout: &PackedLayout, row: usize| f64_bits(layout.position_ids[row]);
    ensure!(
        first.rows() == 40045
            && first.condition_video_rows == 1008
            && first.audio_indices.first() == Some(&2335)
            && first.video_indices[1008] == 2749
            && first.video_indices.len() == 1008 + 37 * 1008,
        "small layout extents"
    );
    ensure!(
        p(&first, 1326) == [0x4094b80000000000, zero, zero]
            && p(&first, 1327) == [origin, h0, w0]
            && p(&first, 2334) == [origin, h23, w41]
            && p(&first, 2541) == [0x4097f40000000000, zero, w0]
            && p(&first, 2542) == [origin, zero, w41]
            && p(&first, 3757) == [0x4094c2aaaaaaaaab, h0, w0]
            && p(&first, 7806) == [0x40952d5555555555, h0, w5]
            && p(&first, 40044) == [0x4097dc0000000000, h23, w41],
        "small layout coordinates"
    );
    ensure!(
        first.token_tags[1327] == VIDEO_TAG
            && first.token_tags[2335] == AUDIO_TAG
            && first.token_tags[2748] == AUDIO_TAG
            && first.token_tags[2749] == VIDEO_TAG,
        "small layout tags"
    );

    let both = PackedLayout::build(
        &spec,
        &small,
        &tags,
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
    )?;
    ensure!(
        both.rows() == 41053
            && both.condition_video_rows == 2016
            && p(&both, 2335) == [0x4097efffffffffff, h0, w0]
            && p(&both, 3342) == [0x4097efffffffffff, h23, w41]
            && p(&both, 3343) == [origin, zero, w0]
            && p(&both, 41052) == [0x4097dc0000000000, h23, w41],
        "last-anchor layout"
    );
    // At 42 latent frames the pairwise and sequential span sums round to different anchor times.
    let pairwise = PackedLayout::build(
        &spec,
        &Geometry::resolve(&spec, 768, 1344, 141)?,
        &tags,
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
    )?;
    ensure!(
        p(&pairwise, 2335)[0] == 0x4098615555555555,
        "pairwise last-anchor time"
    );

    let long = PackedLayout::build(&spec, &production, &tags, &[KeyframeAnchor::First])?;
    ensure!(
        long.rows() == 60431
            && p(&long, 2654) == [0x4099b80000000000, zero, w0]
            && p(&long, 60430) == [0x4099a15555555556, h23, w41],
        "production layout"
    );
    Ok(())
}

#[test]
fn minimax_h3_row_timesteps_reproduce_the_reference_plan() -> Result<()> {
    let spec = spec()?;
    let layout = PackedLayout::build(
        &spec,
        &Geometry::resolve(&spec, 768, 1344, 124)?,
        &text_tags(),
        &[KeyframeAnchor::First],
    )?;
    let plan = row_timestep_plan(
        &layout,
        &Schedule::new(GRID_POINTS, VIDEO_SHIFT)?,
        &Schedule::new(GRID_POINTS, AUDIO_SHIFT)?,
    )?;
    let bits = |step: &RowTimesteps| {
        step.timesteps
            .iter()
            .map(|t| t.to_bits())
            .collect::<Vec<_>>()
    };
    let rows = |step: &RowTimesteps| [0usize, 1327, 2335, 40044].map(|row| step.indices[row]);
    ensure!(plan.len() == 8, "evaluation count");
    ensure!(
        bits(&plan[0]) == [0, 0x3f7fbe77] && rows(&plan[0]) == [0, 1, 0, 0],
        "step 0 plan"
    );
    ensure!(
        bits(&plan[1]) == [0x3cbe8300, 0x3d3a2e90, 0x3f7fbe77]
            && bits(&plan[7]) == [0x3f09d89e, 0x3f333333, 0x3f7fbe77]
            && plan[1..].iter().all(|step| rows(step) == [0, 2, 1, 0]),
        "steps 1-7 plan"
    );
    let adaln = plan[1].adaln_indices(&layout);
    ensure!(
        [adaln[0], adaln[6], adaln[1327], adaln[2335], adaln[40044]] == [1, 0, 6, 5, 0],
        "AdaLN row indices"
    );
    Ok(())
}

fn tensor<T>(
    file: &SafeTensors<'_>,
    name: &str,
    dtype: Dtype,
    decode: fn(&[u8]) -> T,
) -> Result<Vec<T>> {
    let view = file
        .tensor(name)
        .with_context(|| format!("capture lacks {name}"))?;
    ensure!(view.dtype() == dtype, "{name} dtype {:?}", view.dtype());
    let width = dtype.bitsize() / 8;
    Ok(view.data().chunks_exact(width).map(decode).collect())
}

fn le_i64(b: &[u8]) -> i64 {
    i64::from_le_bytes(b.try_into().expect("8 bytes"))
}
fn le_f64(b: &[u8]) -> f64 {
    f64::from_le_bytes(b.try_into().expect("8 bytes"))
}
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes(b.try_into().expect("4 bytes"))
}

fn indices(values: &[u32]) -> Vec<i64> {
    values.iter().map(|&v| i64::from(v)).collect()
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Every layout tensor, both sigma grids and every step's `(timestep, timestep_indices)` equal the capture bit for bit.
fn check_capture(spec: &H3Spec, run: &std::path::Path, video_shift: f64) -> Result<()> {
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(run.join("meta.json"))?)?;
    let resolved = &meta["resolved"];
    let field = |v: &serde_json::Value, key: &str| {
        v[key]
            .as_u64()
            .map(|n| n as usize)
            .with_context(|| format!("meta lacks {key}"))
    };
    let geometry = Geometry::resolve(
        spec,
        field(resolved, "height")?,
        field(resolved, "width")?,
        field(&meta["cell"], "frames")?,
    )?;
    ensure!(
        geometry.frames == field(resolved, "num_frames")?
            && geometry.latent_frames == field(resolved, "num_latent_frames")?
            && geometry.audio_latents == field(resolved, "num_audio_latents")?,
        "geometry {geometry:?} vs {resolved}"
    );
    let anchors: &[KeyframeAnchor] = if meta["cell"]["last_image"].is_null() {
        &[KeyframeAnchor::First]
    } else {
        &[KeyframeAnchor::First, KeyframeAnchor::Last]
    };
    let bytes = fs::read(run.join("layout.safetensors"))?;
    let captured = SafeTensors::deserialize(&bytes)?;
    let tags = tensor(&captured, "token_tags", Dtype::I64, le_i64)?;
    let text = tensor(&captured, "text_indices", Dtype::I64, le_i64)?;
    let text_tags = text
        .iter()
        .map(|&row| u8::try_from(tags[row as usize]))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = PackedLayout::build(spec, &geometry, &text_tags, anchors)?;
    ensure!(
        layout.condition_video_rows == field(resolved, "num_condition_video_rows")?,
        "condition rows"
    );
    let positions = tensor(&captured, "position_ids", Dtype::F64, le_f64)?;
    ensure!(
        positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            == layout
                .position_ids
                .iter()
                .flat_map(|p| f64_bits(*p))
                .collect::<Vec<_>>(),
        "position_ids differ"
    );
    ensure!(
        tags == layout
            .token_tags
            .iter()
            .map(|&t| i64::from(t))
            .collect::<Vec<_>>()
            && tensor(&captured, "video_indices", Dtype::I64, le_i64)?
                == indices(&layout.video_indices)
            && tensor(&captured, "audio_indices", Dtype::I64, le_i64)?
                == indices(&layout.audio_indices)
            && text == indices(&layout.text_indices),
        "tags or indices differ"
    );
    let bytes = fs::read(run.join("final.safetensors"))?;
    let finals = SafeTensors::deserialize(&bytes)?;
    let video = Schedule::new(GRID_POINTS, video_shift)?;
    let audio = Schedule::new(GRID_POINTS, AUDIO_SHIFT)?;
    ensure!(
        f32_bits(&tensor(&finals, "video_sigmas", Dtype::F32, le_f32)?) == f32_bits(&video.sigmas)
            && f32_bits(&tensor(&finals, "audio_sigmas", Dtype::F32, le_f32)?)
                == f32_bits(&audio.sigmas),
        "sigma grids differ"
    );
    for (k, step) in row_timestep_plan(&layout, &video, &audio)?
        .iter()
        .enumerate()
    {
        let bytes = fs::read(run.join(format!("step{k}.safetensors")))?;
        let captured = SafeTensors::deserialize(&bytes)?;
        ensure!(
            f32_bits(&tensor(&captured, "timestep", Dtype::F32, le_f32)?)
                == f32_bits(&step.timesteps)
                && tensor(&captured, "timestep_indices", Dtype::I64, le_i64)?
                    == indices(&step.indices),
            "step {k} row timesteps differ"
        );
    }
    Ok(())
}

#[test]
#[ignore = "CPU-only Stage 0 capture equality; set FF_MODEL and FF_H3_CAPTURES (comma-separated run dirs)"]
fn minimax_h3_layout_and_schedule_equal_the_reference_captures() -> Result<()> {
    let spec = H3Spec::read(&PathBuf::from(
        env::var_os("FF_MODEL").context("FF_MODEL missing")?,
    ))?;
    let runs = env::var("FF_H3_CAPTURES").context("FF_H3_CAPTURES missing")?;
    for run in runs.split(',').map(PathBuf::from) {
        check_capture(&spec, &run, VIDEO_SHIFT).with_context(|| format!("{}", run.display()))?;
        ensure!(
            check_capture(&spec, &run, 12.0).is_err(),
            "{}: shift 12 control matched the capture",
            run.display()
        );
        println!("H3_CAPTURE_EQUAL {}", run.display());
    }
    Ok(())
}

#[test]
#[ignore = "CPU-only published checkpoint audit; set FF_MODEL and FF_H3_LORA"]
fn minimax_h3_manifests_match_the_published_checkpoints() -> Result<()> {
    let model = PathBuf::from(env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = H3Spec::read(&model)?;
    let manifest =
        MappedCheckpoint::open_component(&model.join("transformer"), "diffusion_pytorch_model")?
            .minimax_h3_manifest(&spec)?;
    let f32_tensors = manifest
        .entries
        .values()
        .filter(|entry| entry.dtype == Dtype::F32)
        .count();
    let dormant = manifest
        .entries
        .values()
        .filter(|entry| !entry.load)
        .count();
    ensure!(
        manifest.entries.len() == 638
            && f32_tensors == 12
            && dormant == 100
            && manifest.resident_bytes == 66_280_430_080 - 26_020_915_200,
        "transformer manifest: tensors={} f32={f32_tensors} dormant={dormant} loaded bytes={}",
        manifest.entries.len(),
        manifest.resident_bytes
    );
    let mut layer_matrices = vec![0usize; spec.layers];
    for entry in manifest.entries.values() {
        if let Some(layer) = entry.layer
            && entry.load
            && entry.shape.len() == 2
        {
            layer_matrices[layer as usize] += entry.bytes;
        }
    }
    ensure!(
        layer_matrices.iter().all(|&bytes| bytes == 770_703_360),
        "streamed block matrices per layer: {layer_matrices:?}"
    );
    let lora = PathBuf::from(env::var_os("FF_H3_LORA").context("FF_H3_LORA missing")?);
    let stem = lora
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".safetensors"))
        .context("FF_H3_LORA is not a .safetensors file")?;
    let (manifest, adapter) =
        MappedCheckpoint::open_component(lora.parent().context("LoRA directory")?, stem)?
            .minimax_h3_lora_manifest(&spec)?;
    ensure!(
        manifest.entries.len() == 624 && adapter.rank == 128 && adapter.scale == 0.0625,
        "LoRA manifest: tensors={} {adapter:?}",
        manifest.entries.len()
    );
    let mut shallow = spec.clone();
    shallow.layers -= 1;
    ensure!(
        MappedCheckpoint::open_component(&model.join("transformer"), "diffusion_pytorch_model")?
            .minimax_h3_manifest(&shallow)
            .is_err(),
        "accepted a 49-block manifest"
    );
    println!(
        "H3_MANIFEST tensors=638 f32=12 dormant=100 loaded=40259514880 block_matrices=770703360x50 lora=624 rank=128 scale=0.0625"
    );
    Ok(())
}

#[test]
fn minimax_h3_adaln_plan_addresses_every_row_pair() -> Result<()> {
    let spec = spec()?;
    let geometry = Geometry::resolve(&spec, 768, 1344, 124)?;
    let video = Schedule::new(GRID_POINTS, VIDEO_SHIFT)?;
    let audio = Schedule::new(GRID_POINTS, AUDIO_SHIFT)?;
    for anchors in [
        &[KeyframeAnchor::First][..],
        &[KeyframeAnchor::First, KeyframeAnchor::Last],
    ] {
        let layout = PackedLayout::build(&spec, &geometry, &text_tags(), anchors)?;
        let plan = row_timestep_plan(&layout, &video, &audio)?;
        let adaln = AdalnPlan::build(&layout, &plan);
        ensure!(
            adaln.timesteps.len() == 16 && adaln.pairs.len() == 25,
            "AdaLN extent: timesteps={} pairs={}",
            adaln.timesteps.len(),
            adaln.pairs.len()
        );
        for (k, step) in plan.iter().enumerate() {
            for row in 0..layout.rows() {
                let t = adaln.row_timesteps[k][row];
                ensure!(
                    adaln.timesteps[t as usize] == step.timesteps[step.indices[row] as usize]
                        && adaln.pairs[adaln.row_pairs[k][row] as usize]
                            == (t, layout.token_tags[row]),
                    "step {k} row {row} AdaLN pair"
                );
            }
        }
    }
    Ok(())
}

struct ConditionerCell {
    tokens: Vec<u32>,
    grids: Vec<flyingfish::vision_input::ImageGrid>,
    pixel_values: Vec<f32>,
}

/// Packs the u8 keyframes `k*.pixels [1, 3, 1, h, w]` of `runs` through the published processor configuration.
fn conditioner_pixels(
    processor: &flyingfish::serve::image::Processor,
    frames: &[(PathBuf, usize)],
) -> Result<(Vec<flyingfish::vision_input::ImageGrid>, Vec<f32>)> {
    let (mut grids, mut values) = (Vec::new(), Vec::new());
    for (run, index) in frames {
        let bytes = fs::read(run.join("keyframes.safetensors"))?;
        let file = SafeTensors::deserialize(&bytes)?;
        let name = format!("k{index}.pixels");
        let view = file
            .tensor(&name)
            .with_context(|| format!("capture lacks {name}"))?;
        ensure!(view.dtype() == Dtype::U8, "{name} dtype {:?}", view.dtype());
        let &[1, 3, 1, height, width] = view.shape() else {
            anyhow::bail!("{name} shape {:?}", view.shape());
        };
        let plane = height * width;
        let chw = view.data();
        let image = flyingfish::serve::image::RgbImage {
            width,
            height,
            pixels: (0..plane)
                .flat_map(|p| (0..3).map(move |c| chw[c * plane + p]))
                .collect(),
        };
        let packed = processor.prepare_rgb(&image)?;
        grids.extend(packed.grids);
        values.extend(packed.pixel_values);
    }
    Ok((grids, values))
}

fn check_conditioner_cell(
    spec: &flyingfish::model_spec::ModelSpec,
    cell: &ConditionerCell,
    reference: &SafeTensors<'_>,
) -> Result<()> {
    use flyingfish::vision_input::{MultimodalInput, VisionPositions};
    let vision = spec
        .vision
        .as_ref()
        .context("vision configuration missing")?;
    let section = vision.mrope_section.context("mRoPE section missing")?;
    let rows = cell.tokens.len();
    let modality: Vec<u8> = cell
        .tokens
        .iter()
        .map(|&id| u8::from(id == vision.image_token))
        .collect();
    let plan = MultimodalInput {
        input_ids: &cell.tokens,
        modality_ids: &modality,
        pixel_values: &cell.pixel_values,
        grids: &cell.grids,
        audio: &[],
    }
    .validate(spec, rows)?;
    let axes = plan.chunk(0..rows)?;
    let positions = tensor(reference, "rope_position_ids", Dtype::I64, le_i64)?;
    for (axis, values) in axes.iter().enumerate() {
        ensure!(
            values
                .iter()
                .map(|&v| i64::from(v))
                .eq(positions[axis * rows..(axis + 1) * rows].iter().copied()),
            "rope position axis {axis} differs"
        );
    }
    let half = spec.head_dim as usize / 2;
    let table =
        flyingfish::minimax_h3::mrope_table(&plan, section, spec.rope_theta, spec.head_dim)?;
    let mut worst = 0f32;
    for (name, offset) in [("text_cos", 0), ("text_sin", half)] {
        let expected = tensor(reference, name, Dtype::F32, le_f32)?;
        ensure!(expected.len() == rows * 2 * half, "{name} extent");
        for row in 0..rows {
            for j in 0..2 * half {
                let got = table[row * 2 * half + offset + j % half];
                worst = worst.max((got - expected[row * 2 * half + j]).abs());
            }
        }
    }
    ensure!(worst <= 4e-6, "mRoPE table max_abs {worst:e} above 4e-6");
    let patches: usize = cell.grids.iter().map(|g| (g.t * g.h * g.w) as usize).sum();
    let positions = VisionPositions::new(vision, spec.architecture, &cell.grids, patches)?;
    let coordinates = tensor(reference, "vision_position_ids", Dtype::I64, le_i64)?;
    ensure!(
        positions
            .coordinates
            .iter()
            .flat_map(|c| c.map(i64::from))
            .eq(coordinates.iter().copied()),
        "vision coordinates differ"
    );
    let indices = tensor(reference, "interp_indices", Dtype::I64, le_i64)?;
    let weights = tensor(reference, "interp_weights", Dtype::F32, le_f32)?;
    for k in 0..4 {
        ensure!(
            positions.bilinear_indices[k]
                .iter()
                .enumerate()
                .all(|(n, &i)| i64::from(i) == indices[n * 4 + k]),
            "interpolation corner {k} indices differ"
        );
        ensure!(
            positions.bilinear_weights[k]
                .iter()
                .enumerate()
                .all(|(n, w)| w.to_bits() == weights[n * 4 + k].to_bits()),
            "interpolation corner {k} weights differ"
        );
    }
    println!("H3_N6A_TABLE rows={rows} patches={patches} mrope_max_abs={worst:e}");
    Ok(())
}

/// Qwen3-VL conditioner inputs on the CPU: processor pixels bit-equal to the captured `pixel_values`; mRoPE
/// positions, vision coordinates and bilinear tables equal to the model-definition dump; the mRoPE table within 4e-6.
#[test]
#[ignore = "CPU-only N6a gate; set FF_H3_TEXT_ENCODER, FF_H3_S0 (Stage 0 run root) and FF_H3_N6A_REF"]
fn minimax_h3_conditioner_inputs_equal_the_reference() -> Result<()> {
    let encoder =
        PathBuf::from(env::var_os("FF_H3_TEXT_ENCODER").context("FF_H3_TEXT_ENCODER missing")?);
    let s0 = PathBuf::from(env::var_os("FF_H3_S0").context("FF_H3_S0 missing")?);
    let dumps = PathBuf::from(env::var_os("FF_H3_N6A_REF").context("FF_H3_N6A_REF missing")?);
    let spec = flyingfish::model_spec::ModelSpec::read(&encoder)?;
    ensure!(
        spec.architecture == flyingfish::model_spec::Architecture::Qwen3Vl,
        "text encoder parsed as {:?}",
        spec.architecture
    );
    let processor = flyingfish::serve::image::Processor::load(&encoder, None)?
        .context("text encoder has no processor")?;
    let cells = [
        ("A", "A-bf16", vec![("A-bf16", 0)], vec!["A-bf16"]),
        (
            "D",
            "D-bf16",
            vec![("D-bf16", 0), ("D-bf16", 1)],
            vec!["D-bf16"],
        ),
        (
            "E",
            "D-bf16",
            vec![("A-bf16", 0), ("B-bf16", 0)],
            vec!["A-bf16", "B-bf16"],
        ),
    ];
    for (name, tokens_from, frames, captured) in cells {
        let bytes = fs::read(s0.join(tokens_from).join("text.safetensors"))?;
        let text = SafeTensors::deserialize(&bytes)?;
        let tokens = tensor(&text, "token_ids", Dtype::I64, le_i64)?
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let frames: Vec<_> = frames
            .into_iter()
            .map(|(run, k)| (s0.join(run), k))
            .collect();
        let (grids, pixel_values) = conditioner_pixels(&processor, &frames)?;
        let mut expected = Vec::new();
        for run in captured {
            let bytes = fs::read(s0.join(run).join("text.safetensors"))?;
            expected.extend(f32_bits(&tensor(
                &SafeTensors::deserialize(&bytes)?,
                "pixel_values",
                Dtype::F32,
                le_f32,
            )?));
        }
        ensure!(
            f32_bits(&pixel_values) == expected,
            "cell {name}: pixel_values differ from the capture"
        );
        let dump = fs::read(dumps.join(format!("{name}.safetensors")))?;
        let reference = SafeTensors::deserialize(&dump)?;
        let cell = ConditionerCell {
            tokens,
            grids,
            pixel_values,
        };
        check_conditioner_cell(&spec, &cell, &reference).with_context(|| format!("cell {name}"))?;
        let swapped = ConditionerCell {
            grids: cell
                .grids
                .iter()
                .map(|g| flyingfish::vision_input::ImageGrid {
                    t: g.t,
                    h: g.w,
                    w: g.h,
                })
                .collect(),
            ..cell
        };
        ensure!(
            check_conditioner_cell(&spec, &swapped, &reference).is_err(),
            "cell {name}: h/w swapped grid control matched the reference"
        );
        println!("H3_N6A_CELL cell={name} PASS");
    }
    println!("H3_N6A_GATE PASS cells=3 control=FAIL_AS_REQUIRED");
    Ok(())
}

/// `FF_MODEL`, `FF_H3_S0` (Stage 0 runs) and `FF_H3_N9` (the N9 freeze) for the producer gates.
fn producer_env() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let var = |key: &str| -> Result<PathBuf> {
        Ok(PathBuf::from(
            env::var_os(key).with_context(|| format!("{key} missing"))?,
        ))
    };
    Ok((var("FF_MODEL")?, var("FF_H3_S0")?, var("FF_H3_N9")?))
}

/// The tensor `name` of a recorded safetensors file as raw little-endian bytes with its dtype and shape.
fn recorded(path: &std::path::Path, name: &str) -> Result<(Dtype, Vec<usize>, Vec<u8>)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let file = SafeTensors::deserialize(&bytes)?;
    let t = file.tensor(name)?;
    Ok((t.dtype(), t.shape().to_vec(), t.data().to_vec()))
}

fn i64s(data: &[u8]) -> Vec<i64> {
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect()
}

/// The recorded run `cell`'s prompt and its keyframe image tokens (`t * h * w / 4` per image grid).
fn recorded_presentation(s0: &std::path::Path, cell: &str) -> Result<(String, Vec<usize>)> {
    let run = s0.join(format!("{cell}-bf16"));
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(run.join("meta.json"))?)?;
    let prompt = meta["prompt"].as_str().context("meta prompt")?.to_owned();
    let (_, shape, grid) = recorded(&run.join("text.safetensors"), "image_grid_thw")?;
    ensure!(
        shape.len() == 2 && shape[1] == 3,
        "image_grid_thw {shape:?}"
    );
    let tokens = i64s(&grid)
        .chunks(3)
        .map(|g| (g[0] * g[1] * g[2]) as usize / 4)
        .collect();
    Ok((prompt, tokens))
}

/// N9a: the native presentation of the recorded prompt and image grids equals the recorded token ids and text-row
/// tags of cells A, P, B and D (D: two keyframes); control: one prompt character changed changes the ids.
#[test]
#[ignore = "FF_MODEL, FF_H3_S0, FF_H3_N9"]
fn minimax_h3_presentation_matches_the_recorded_tokens() -> Result<()> {
    let (model, s0, _) = producer_env()?;
    let tokenizer = flyingfish::serve::detok::Shared::load(&model.join("text_encoder"))?;
    for cell in ["A", "P", "B", "D"] {
        let (prompt, image_tokens) = recorded_presentation(&s0, cell)?;
        let (ids, tags) = flyingfish::minimax_h3::presentation(&tokenizer, &prompt, &image_tokens)?;
        let run = s0.join(format!("{cell}-bf16"));
        let want: Vec<u32> = i64s(&recorded(&run.join("text.safetensors"), "token_ids")?.2)
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let token_tags = i64s(&recorded(&run.join("layout.safetensors"), "token_tags")?.2);
        let rows = i64s(&recorded(&run.join("layout.safetensors"), "text_indices")?.2);
        let want_tags: Vec<u8> = rows.iter().map(|&r| token_tags[r as usize] as u8).collect();
        println!(
            "H3_N9A cell={cell} keyframes={} tokens={} ids_equal={} tags_equal={}",
            image_tokens.len(),
            ids.len(),
            ids == want,
            tags == want_tags
        );
        ensure!(
            ids == want && tags == want_tags,
            "cell {cell}: presentation differs from the recording"
        );
        let mut changed = prompt.clone();
        changed.replace_range(0..1, "f");
        let (control, _) =
            flyingfish::minimax_h3::presentation(&tokenizer, &changed, &image_tokens)?;
        ensure!(
            control != want,
            "cell {cell}: a changed prompt kept the recorded ids"
        );
    }
    println!("H3_N9A_GATE PASS cells=4 control=FAIL_AS_REQUIRED");
    Ok(())
}

/// N9b: the native canvas equals the model definition's `resolve_canvas_size` table of the N9 freeze.
#[test]
#[ignore = "FF_MODEL, FF_H3_S0, FF_H3_N9"]
fn minimax_h3_canvas_matches_the_definition_table() -> Result<()> {
    let (model, _, n9) = producer_env()?;
    let spec = H3Spec::read(&model)?;
    let table: serde_json::Value = serde_json::from_slice(&fs::read(n9.join("canvas.json"))?)?;
    let rows = table["rows"].as_array().context("canvas rows")?;
    for row in rows {
        let v = |i: usize| row[i].as_f64().context("canvas row value");
        let got = flyingfish::minimax_h3::canvas_size(&spec, v(0)?, v(1)?)?;
        let want = (v(2)? as usize, v(3)? as usize);
        ensure!(
            got == want,
            "aspect {}:{} gives {got:?}, the definition {want:?}",
            v(0)?,
            v(1)?
        );
    }
    println!("H3_N9B_CANVAS PASS rows={}", rows.len());
    Ok(())
}

/// N9b: the text encoder's image processor on the recorded canvas u8 reproduces the recorded `pixel_values` and grid
/// (cells A and B, exact); the native keyframe (decode + Lanczos-3 of the source JPEG) lies within the frozen floor of
/// the recorded canvas (max_abs at most twice the freeze's combined floor; the differing fraction is diagnostic).
/// Control: the native keyframe shifted one pixel horizontally must exceed the max_abs bound by at least 10x in every
/// cell; otherwise the keyframe comparison is a measurement only (the pixel_values check still gates).
#[test]
#[ignore = "FF_MODEL, FF_H3_S0, FF_H3_N9"]
fn minimax_h3_image_producers_match_the_recording() -> Result<()> {
    let (model, s0, n9) = producer_env()?;
    let floor: serde_json::Value = serde_json::from_slice(&fs::read(n9.join("floor.json"))?)?;
    let combined = |key: &str| -> Result<f64> {
        ["i2va_test_1_chef.jpg", "i2va_test_11_rally_car.jpg"]
            .iter()
            .map(|name| {
                floor[name]["combined_floor"][key]
                    .as_f64()
                    .context("combined floor")
            })
            .try_fold(0f64, |m, v| Ok(m.max(v?)))
    };
    let (max_bound, fraction_bound) = (
        2.0 * combined("max_abs")?,
        2.0 * combined("differing_fraction")?,
    );
    let processor = flyingfish::serve::image::Processor::load(&model.join("text_encoder"), None)?
        .context("text encoder image processor")?;
    let (mut detected_all, mut within_all) = (true, true);
    for cell in ["A", "B"] {
        let run = s0.join(format!("{cell}-bf16"));
        let (_, shape, planes) = recorded(&run.join("keyframes.safetensors"), "k0.pixels")?;
        let (height, width) = (shape[3], shape[4]);
        let plane = height * width;
        let pixels: Vec<u8> = (0..plane)
            .flat_map(|p| (0..3).map(move |c| (c, p)))
            .map(|(c, p)| planes[c * plane + p])
            .collect();
        let canvas = flyingfish::serve::image::RgbImage {
            width,
            height,
            pixels,
        };
        let packed = processor.prepare_rgb(&canvas)?;
        let (_, _, want) = recorded(&run.join("text.safetensors"), "pixel_values")?;
        let want: Vec<f32> = want
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let grid = i64s(&recorded(&run.join("text.safetensors"), "image_grid_thw")?.2);
        let grid_equal = packed
            .grids
            .iter()
            .flat_map(|g| [g.t as i64, g.h as i64, g.w as i64])
            .collect::<Vec<_>>()
            == grid;
        println!(
            "H3_N9B cell={cell} pixel_values_equal={} grid_equal={grid_equal}",
            packed.pixel_values == want
        );
        ensure!(
            packed.pixel_values == want && grid_equal,
            "cell {cell}: pixel_values differ from the recording"
        );
        let meta: serde_json::Value = serde_json::from_slice(&fs::read(run.join("meta.json"))?)?;
        let source = flyingfish::serve::image::decode_image(
            &fs::read(meta["cell"]["image"].as_str().context("meta image")?)?,
            Some("image/jpeg"),
        )?;
        let empty = flyingfish::minimax_h3::H3Draws {
            posterior_eps: Vec::new(),
            anchor_noise: Vec::new(),
            video_noise: Vec::new(),
            audio_rows: Vec::new(),
        };
        let (prompt, _) = recorded_presentation(&s0, cell)?;
        let request = flyingfish::minimax_h3::prepare_i2va(
            &model,
            &prompt,
            &source,
            (124, empty),
            (8, (6.0, 3.0)),
        )?;
        ensure!(
            request.height == height && request.width == width,
            "canvas {}x{}",
            request.height,
            request.width
        );
        let diff: Vec<u8> = request
            .keyframe
            .iter()
            .zip(&planes)
            .map(|(&a, &b)| a.abs_diff(b))
            .collect();
        let max = f64::from(*diff.iter().max().context("empty keyframe")?);
        let fraction = diff.iter().filter(|&&d| d > 0).count() as f64 / diff.len() as f64;
        let mean = diff.iter().map(|&d| f64::from(d)).sum::<f64>() / diff.len() as f64;
        let ok = max <= max_bound;
        println!(
            "H3_N9B cell={cell} keyframe max_abs={max} bound={max_bound} within={ok} differing_fraction={fraction:.6} (diagnostic, floor x2 {fraction_bound:.6}) mean_abs={mean:.6} native_pixel_values_equal={}",
            request.pixel_values == want
        );
        within_all &= ok;
        let shifted_max = (0..request.keyframe.len())
            .map(|i| {
                let x = i % width;
                let from = if x + 1 < width { i + 1 } else { i };
                request.keyframe[from].abs_diff(planes[i])
            })
            .max()
            .context("empty keyframe")?;
        let detected = f64::from(shifted_max) > 10.0 * max_bound;
        println!(
            "H3_N9B cell={cell} one-pixel-shift control max_abs={shifted_max} required>{} detected={detected}",
            10.0 * max_bound
        );
        detected_all &= detected;
    }
    if detected_all {
        ensure!(within_all, "native keyframe outside the frozen floor");
        println!("H3_N9B_GATE PASS cells=2 keyframe=GATED control=DETECTED");
    } else {
        println!(
            "H3_N9B_GATE PASS cells=2 keyframe=MEASUREMENT within={within_all} control=NOT_DETECTED_AT_10X"
        );
    }
    Ok(())
}
