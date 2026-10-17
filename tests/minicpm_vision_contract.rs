use anyhow::{Context, Result, ensure};
use flyingfish::{
    model_spec::{Architecture, ModelSpec},
    serve::detok::Shared,
    serve::image::{PreparedImages, Processor, RgbImage},
    vision_input::ImageGrid,
};
use safetensors::{Dtype, SafeTensors};
use std::path::PathBuf;

const CASES: [&str; 10] = [
    "no-slice-224",
    "wide-1xn",
    "tall-nx1",
    "grid-3x3",
    "aspect-20x1",
    "extreme-80x1",
    "odd-979x653",
    "odd-1000x700",
    "multi-two",
    "many-11",
];

/// Reference-only preprocess noise floor: torchvision vs PIL processor, max 0.0156863257 normalized
/// (2 u8 levels), `evidence/T0-ref/preprocess_noise_floor.json` sha256
/// `3431df00de14adfbd0b50c5ef650319523412b8278cce25fa2fb9ef115ea02c1`; the 4·EPS slack covers
/// f32 accumulation, not candidate measurements.
const PIXEL_TOLERANCE: f32 = 4.0 / 255.0 + 4.0 * f32::EPSILON;

fn integers(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i64>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::I64, "{name} must be I64");
    Ok(tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| i64::from_le_bytes(*v))
        .collect())
}

fn floats(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = tensors.tensor(name)?;
    ensure!(tensor.dtype() == Dtype::F32, "{name} must be F32");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect())
}

#[test]
#[ignore = "CPU-only external fixtures; set FF_MODEL and FF_MINICPM_FIXTURE_DIR"]
fn processor_layout_pixels_and_canvas_match_reference() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = PathBuf::from(
        std::env::var_os("FF_MINICPM_FIXTURE_DIR").context("FF_MINICPM_FIXTURE_DIR missing")?,
    );
    let spec = ModelSpec::read(&model)?;
    ensure!(
        spec.architecture == Architecture::MiniCpmV47,
        "fixtures require the MiniCPM spec"
    );
    let tokenizer = Shared::load(&model)?;
    let processor =
        Processor::load(&model, Some(&tokenizer))?.context("MiniCPM image processor")?;
    let mut worst_pixels = 0.0_f32;
    for case in CASES {
        let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let count = usize::try_from(integers(&tensors, "image_count")?[0])?;
        let widths = integers(&tensors, "image_widths")?;
        let heights = integers(&tensors, "image_heights")?;
        let mut images = Vec::with_capacity(count);
        for index in 0..count {
            let tensor = tensors.tensor(&format!("rgb_{index}"))?;
            ensure!(tensor.dtype() == Dtype::U8, "{case} rgb_{index} must be U8");
            let width = usize::try_from(widths[index])?;
            let height = usize::try_from(heights[index])?;
            ensure!(
                tensor.shape() == [height, width, 3],
                "{case} rgb_{index} shape"
            );
            images.push(processor.prepare_rgb(&RgbImage {
                width,
                height,
                pixels: tensor.data().to_vec(),
            })?);
        }
        let prompt: Vec<u32> = integers(&tensors, "prompt_ids")?
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let expected_ids: Vec<u32> = integers(&tensors, "input_ids")?
            .into_iter()
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        let capacity = expected_ids.len() + 64;
        let (expanded, prepared) = processor.expand(&prompt, images, capacity)?;
        assert_eq!(expanded, expected_ids, "{case} ids");

        let expected_modality: Vec<u8> = integers(&tensors, "mm_token_type_ids")?
            .into_iter()
            .map(u8::try_from)
            .collect::<std::result::Result<_, _>>()?;
        assert_eq!(prepared.modality_ids, expected_modality, "{case} modality");

        let target_sizes = integers(&tensors, "target_sizes")?;
        let expected_grids: Vec<ImageGrid> = target_sizes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|size| {
                Ok(ImageGrid {
                    t: 1,
                    h: u32::try_from(size[0])?,
                    w: u32::try_from(size[1])?,
                })
            })
            .collect::<Result<_>>()?;
        assert_eq!(prepared.grids, expected_grids, "{case} grids");

        let reference = floats(&tensors, "pixel_values_fp32")?;
        assert_eq!(
            prepared.pixel_values.len(),
            reference.len(),
            "{case} pixel count"
        );
        let mut worst = 0.0_f32;
        let mut differing = 0usize;
        for (actual, expected) in prepared.pixel_values.iter().zip(&reference) {
            worst = worst.max((actual - expected).abs());
            differing += usize::from(actual.to_bits() != expected.to_bits());
        }
        worst_pixels = worst_pixels.max(worst);
        println!(
            "{}",
            serde_json::json!({
                "case": case,
                "ids": expanded.len(),
                "crops": prepared.grids.len(),
                "pixels": reference.len(),
                "pixel_max_abs": worst,
                "pixel_different": differing,
            })
        );
        assert!(
            worst <= PIXEL_TOLERANCE,
            "{case} pixel error {worst} exceeds {PIXEL_TOLERANCE}"
        );

        let plan = prepared.input(&expanded).validate(&spec, capacity)?;
        let positions = integers(&tensors, "positions")?;
        let len = expanded.len();
        ensure!(
            positions.len() == 3 * len,
            "{case} reference position count"
        );
        for axis in 0..3 {
            for row in 0..len {
                assert_eq!(
                    i64::from(plan.axes()[axis][row]),
                    positions[axis * len + row],
                    "{case} axis {axis} row {row}"
                );
            }
        }
        assert_eq!(
            plan.rope_delta(),
            integers(&tensors, "rope_delta")?[0],
            "{case} rope delta"
        );
        if case == CASES[0] {
            let run = prepared
                .modality_ids
                .iter()
                .position(|&m| m == 1)
                .context("no image tokens")?;
            let mut bad_ids = expanded.clone();
            let mut bad_modality = prepared.modality_ids.clone();
            bad_ids.remove(run);
            bad_modality.remove(run);
            let bad = PreparedImages {
                modality_ids: bad_modality,
                pixel_values: prepared.pixel_values.clone(),
                grids: prepared.grids.clone(),
            };
            assert!(
                bad.input(&bad_ids).validate(&spec, capacity).is_err(),
                "{case} short image run must be rejected"
            );
            let mut bad_ids = expanded.clone();
            bad_ids[run - 1] = 0;
            let bad = PreparedImages {
                modality_ids: prepared.modality_ids.clone(),
                pixel_values: prepared.pixel_values.clone(),
                grids: prepared.grids.clone(),
            };
            assert!(
                bad.input(&bad_ids).validate(&spec, capacity).is_err(),
                "{case} missing opening marker must be rejected"
            );
        }
    }
    println!("{{\"worst_pixel_max_abs\":{worst_pixels}}}");
    Ok(())
}
