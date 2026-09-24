use super::H3Command;
use super::checkpoint::resolve_component;
use super::device_parse::parse_device_single;
use super::generate::{LatentShape, make_t2va_noise};
use super::output_hygiene::{ensure_new_output, resolve_output_outside_model};
use super::qwen_numerical::validate_qwen_numerical_contract;
use anyhow::{Context, Result, bail};
use candle_core::{Tensor, safetensors};
use flyingfish::h3::config::TransformerConfig;
use flyingfish::h3::cuda::profile::validate_selected_device as validate_h3_selected_cuda_profile;
use flyingfish::h3::policy::H3QwenNumericalContract;
use flyingfish::h3::text_encoder::StreamedTextEncoder;
use flyingfish::runtime::artifact::ArtifactStaging;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub(crate) fn take_input(values: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    values
        .remove(name)
        .with_context(|| format!("input safetensors is missing {name}"))
}

pub(crate) fn sorted_tensor_names(values: &HashMap<String, Tensor>) -> Vec<String> {
    let mut names = values.keys().cloned().collect::<Vec<_>>();
    names.sort_unstable();
    names
}

pub(super) fn run_encode_prompt(command: H3Command) -> Result<()> {
    let H3Command::EncodePrompt {
        model,
        component,
        tokenizer,
        prompt,
        output,
        device,
        weights: weight_args,
        attention_query_chunk_size,
        target_hidden_state,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for encode-prompt");
    };
    let component_dir = resolve_component(&model, &component)?;
    let output = resolve_output_outside_model(&output, &model)?;
    let tokenizer_path = model.join(&tokenizer);
    anyhow::ensure!(
        tokenizer_path.is_file(),
        "tokenizer does not exist: {}",
        tokenizer_path.display()
    );
    ensure_new_output(&output, "prompt encoding output")?;
    let output_staging = ArtifactStaging::new_for_path_producer(&output)
        .with_context(|| format!("failed to stage prompt encoding {}", output.display()))?;
    let device = parse_device_single(&device)?;
    validate_h3_selected_cuda_profile(&device)
        .context("prompt encoding exact CUDA profile preflight")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let encoder = StreamedTextEncoder::open(
        component_dir,
        weight_args.weight_source,
        weight_args.cache_policy()?,
        device.clone(),
        target_hidden_state,
        attention_query_chunk_size,
    )?;
    let encoded = encoder.encode_prompt(&tokenizer, &prompt)?;
    let tag_count = encoded.text_token_tags.len();
    let token_count = encoded.token_ids.len();
    let tags = Tensor::from_vec(encoded.text_token_tags, tag_count, &device)?;
    let token_ids = Tensor::from_vec(encoded.token_ids, token_count, &device)?;
    let contract = encoded.numerical_contract;
    let mut tensors = HashMap::from([
        ("prompt_embeddings", encoded.embeddings),
        ("text_token_tags", tags),
        ("token_ids", token_ids),
    ]);
    contract.insert_artifact_tensors(&mut tensors, &device)?;
    safetensors::save(&tensors, output_staging.producer_path())
        .with_context(|| format!("failed to save prompt encoding {}", output.display()))?;
    output_staging.publish()?;
    println!(
        "encoded {token_count} H3 prompt tokens to {}",
        output.display()
    );
    Ok(())
}

pub(super) fn run_prepare_t2va_inputs(command: H3Command) -> Result<()> {
    let H3Command::PrepareT2vaInputs {
        model,
        component,
        prompt_encoding,
        output,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        audio_channels,
        target,
        seed,
        device,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for prepare-t2va-inputs");
    };
    let (latent_frames, latent_height, latent_width, audio_frames) = target
        .resolve_required_latent_geometry((
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
        ))?;
    let component_dir = resolve_component(&model, &component)?;
    let output = resolve_output_outside_model(&output, &model)?;
    let config = TransformerConfig::from_file(component_dir.join("config.json"))?;
    anyhow::ensure!(
        prompt_encoding.is_file(),
        "prompt encoding does not exist: {}",
        prompt_encoding.display()
    );
    ensure_new_output(&output, "prepared T2VA output")?;
    let output_staging = ArtifactStaging::new_for_path_producer(&output)
        .with_context(|| format!("failed to stage prepared T2VA inputs {}", output.display()))?;
    for (name, value) in [
        ("latent_frames", latent_frames),
        ("latent_height", latent_height),
        ("latent_width", latent_width),
        ("audio_frames", audio_frames),
        ("audio_channels", audio_channels),
    ] {
        anyhow::ensure!(value > 0, "{name} must be non-zero");
    }
    let device = parse_device_single(&device)?;
    let mut values = safetensors::load(&prompt_encoding, &device).with_context(|| {
        format!(
            "failed to load prompt encoding {}",
            prompt_encoding.display()
        )
    })?;
    let qwen_contract = H3QwenNumericalContract::take_artifact_tensors(&mut values)?
        .context("prompt encoding is missing the required Qwen numerical contract")?;
    let prompt_embeddings = take_input(&mut values, "prompt_embeddings")?;
    let text_token_tags = take_input(&mut values, "text_token_tags")?;
    let token_ids = take_input(&mut values, "token_ids")?;
    let (prompt_batch, prompt_rows, _) = prompt_embeddings
        .dims3()
        .context("prompt_embeddings must be [1, rows, width]")?;
    anyhow::ensure!(
        prompt_batch == 1 && prompt_rows > 0,
        "prompt_embeddings must contain one non-empty batch"
    );
    anyhow::ensure!(
        matches!(
            prompt_embeddings.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16
        ),
        "prompt_embeddings must use a floating-point dtype"
    );
    anyhow::ensure!(
        text_token_tags.dtype() == candle_core::DType::U32
            && text_token_tags.dims() == [prompt_rows]
            && text_token_tags
                .to_vec1::<u32>()?
                .iter()
                .all(|tag| *tag == 1),
        "text_token_tags must be a U32 text-tag vector matching prompt rows"
    );
    anyhow::ensure!(
        token_ids.dtype() == candle_core::DType::U32 && token_ids.dims() == [prompt_rows],
        "token_ids must be a U32 vector matching prompt rows"
    );
    anyhow::ensure!(
        values.is_empty(),
        "prompt encoding contains unexpected tensors: {}",
        sorted_tensor_names(&values).join(", ")
    );
    validate_qwen_numerical_contract(&qwen_contract, &device, prompt_rows, 0, 0, 0)?;
    let (video_latents, audio_latents) = make_t2va_noise(
        &config,
        LatentShape {
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            audio_channels,
        },
        seed,
        &device,
    )?;
    let mut tensors = HashMap::from([
        ("prompt_embeddings", prompt_embeddings),
        ("text_token_tags", text_token_tags),
        ("video_latents", video_latents),
        ("audio_latents", audio_latents),
    ]);
    qwen_contract.insert_artifact_tensors(&mut tensors, &device)?;
    safetensors::save(&tensors, output_staging.producer_path())
        .with_context(|| format!("failed to save prepared T2VA inputs {}", output.display()))?;
    output_staging.publish()?;
    println!("saved prepared T2VA inputs to {}", output.display());
    Ok(())
}
