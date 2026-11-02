//! MiniMax-H3 FL2VA geometry, layout and schedule gates.
use anyhow::{Context, Result, ensure};
use flyingfish::{
    checkpoint::MappedCheckpoint,
    minimax_h3::{
        AUDIO_TAG, Geometry, H3Spec, KeyframeAnchor, PackedLayout, RowTimesteps, Schedule,
        TEXT_TAG, VIDEO_TAG, row_timestep_plan,
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
    ensure!(
        manifest.entries.len() == 638
            && f32_tensors == 12
            && manifest.resident_bytes == 66_280_430_080,
        "transformer manifest: tensors={} f32={f32_tensors} bytes={}",
        manifest.entries.len(),
        manifest.resident_bytes
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
    println!("H3_MANIFEST tensors=638 f32=12 bytes=66280430080 lora=624 rank=128 scale=0.0625");
    Ok(())
}
