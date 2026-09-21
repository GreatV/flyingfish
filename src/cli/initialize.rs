use super::H3Command;
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, safetensors};
use flyingfish::h3::conditioning_provenance::H3ConditioningProvenance;
use flyingfish::recovery::{
    CheckpointIdentity, MAX_RECOVERY_CHECKPOINT_BYTES, PolicyHistory,
    take_t2va_checkpoint_metadata, validate_t2va_schedule,
};
use flyingfish::runtime::artifact::{ArtifactStaging, read_artifact_snapshot};
use std::collections::HashMap;

pub(super) fn run_initialize_t2va_checkpoint(command: H3Command) -> Result<()> {
    let H3Command::InitializeT2vaCheckpoint {
        inputs,
        output,
        sigma_points,
        video_shift,
        audio_shift,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for initialize-t2va-checkpoint");
    };
    anyhow::ensure!(
        inputs.is_file(),
        "initial T2VA inputs do not exist: {}",
        inputs.display()
    );
    anyhow::ensure!(
        !output.exists(),
        "step-zero checkpoint output already exists: {}",
        output.display()
    );
    validate_t2va_schedule(
        u64::try_from(sigma_points.get()).context("sigma_points exceeds u64")?,
        video_shift,
        audio_shift,
    )?;
    let snapshot = read_artifact_snapshot(&inputs, MAX_RECOVERY_CHECKPOINT_BYTES)
        .with_context(|| format!("failed to read initial T2VA inputs {}", inputs.display()))?;
    let mut tensors = safetensors::load_buffer(&snapshot.bytes, &Device::Cpu)
        .with_context(|| format!("failed to load initial T2VA inputs {}", inputs.display()))?;
    let conditioning_provenance =
        match H3ConditioningProvenance::take_artifact_tensors(&mut tensors)? {
            Some(H3ConditioningProvenance::FlyingfishQwen(contract)) => {
                H3ConditioningProvenance::FlyingfishQwen(contract)
            }
            Some(H3ConditioningProvenance::ExternallyValidated(_)) => bail!(
                "initialize-t2va-checkpoint does not accept external official-fixture provenance"
            ),
            None => bail!("initial T2VA inputs are missing Flyingfish Qwen provenance"),
        };
    anyhow::ensure!(
        take_t2va_checkpoint_metadata(&mut tensors)?.is_none(),
        "initialize-t2va-checkpoint requires raw initial inputs"
    );
    tensors.insert(
        "completed_steps".to_owned(),
        Tensor::new(0u32, &Device::Cpu)?,
    );
    tensors.insert(
        "sigma_points".to_owned(),
        Tensor::new(
            u32::try_from(sigma_points.get()).context("sigma_points exceeds u32")?,
            &Device::Cpu,
        )?,
    );
    tensors.insert(
        "video_shift".to_owned(),
        Tensor::new(video_shift, &Device::Cpu)?,
    );
    tensors.insert(
        "audio_shift".to_owned(),
        Tensor::new(audio_shift, &Device::Cpu)?,
    );
    let mut policy_tensors = HashMap::new();
    PolicyHistory::new().insert_checkpoint_tensors(&mut policy_tensors, &Device::Cpu)?;
    conditioning_provenance.insert_artifact_tensors(&mut policy_tensors, &Device::Cpu)?;
    tensors.extend(
        policy_tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor)),
    );
    let staging = ArtifactStaging::new_for_path_producer(&output)
        .with_context(|| format!("failed to stage step-zero checkpoint {}", output.display()))?;
    safetensors::save(&tensors, staging.producer_path())
        .with_context(|| format!("failed to save step-zero checkpoint {}", output.display()))?;
    let published = staging.publish()?;
    let identity = CheckpointIdentity::collect(&published.destination)?;
    println!(
        "initialized step-zero checkpoint {} of {} bytes",
        published.destination.display(),
        identity.checkpoint_bytes
    );
    Ok(())
}
