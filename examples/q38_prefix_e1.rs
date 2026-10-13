//! Exact Qwen3.8 hybrid-prefix save/poison/restore experiment.
//!
//! Build with the CUDA feature and run only under a 4d GPU grant.  The CPU
//! legs still compile this example, but exit with an explicit backend error.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(feature = "cuda"))]
use anyhow::bail;
#[cfg(feature = "cuda")]
use anyhow::ensure;
use clap::Parser;
#[cfg(feature = "cuda")]
use flyingfish::prefix_state::sha256;
#[cfg(feature = "cuda")]
use flyingfish::{
    backend::{
        Backend, Settings, SpecBudget, TextPrefixRestoreIntent, TextPrefixSaveRequest, TreeBuilder,
    },
    model_spec::ModelSpec,
    prefix_state::{PrefixIdentity, PrefixSnapshot},
};
#[cfg(feature = "cuda")]
use serde_json::json;
#[cfg(feature = "cuda")]
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prefix_ids: PathBuf,
    #[arg(long)]
    suffix_ids: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    package_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value_t = 32768)]
    capacity: usize,
    #[arg(long, default_value_t = 256)]
    block_tokens: usize,
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    transfer_bytes: usize,
    #[arg(long)]
    runtime_dir: Option<PathBuf>,
    #[arg(long, default_value = "Qwen3.8-27B-FP8")]
    model_id: String,
    #[arg(long, default_value = "1f7800145ce7ae0f59a6306d33c90cf80bb0f170")]
    model_revision: String,
    #[arg(long, default_value = "qwen3.8-official")]
    tokenizer: String,
    #[arg(long, default_value = "mrope_interleaved/theta1e7/context-v1")]
    rope: String,
}

#[cfg(feature = "cuda")]
fn read_ids(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 4 == 0 {
        // Use the raw path only when it is explicitly named .u32.
        if path.extension().is_some_and(|x| x == "u32") {
            return Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect());
        }
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let values = value
        .as_array()
        .with_context(|| format!("{} must contain a JSON token array", path.display()))?;
    values
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .with_context(|| format!("{} contains a non-u32 token", path.display()))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        bail!("q38_prefix_e1 requires --features cuda");
    }
    #[cfg(feature = "cuda")]
    {
        let prefix = read_ids(&args.prefix_ids)?;
        let suffix = read_ids(&args.suffix_ids)?;
        ensure_counts(&prefix, &suffix, args.capacity)?;
        let spec = ModelSpec::read(&args.model)?;
        let identity = PrefixIdentity::new(
            args.model_id.clone(),
            args.model_revision.clone(),
            args.tokenizer.clone(),
            args.rope.clone(),
            &spec,
        )?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: true,
            draft_model: None,
            runtime_dir: args.runtime_dir.clone(),
        };
        let mut engine = Backend::load(&args.model, args.device, args.capacity, None, settings)?;
        engine.prefill(&prefix, None)?;
        let saved = engine.save_text_prefix(TextPrefixSaveRequest {
            identity: identity.clone(),
            prefix_ids: &prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        engine.step(false)?;
        let baseline_pending_token = engine.token()?;
        let baseline_pending_logits = engine.decode_logits()?;
        let baseline_token = engine.prefill(&suffix, None)?;
        let baseline_logits = engine.decode_logits()?;
        engine.poison_state_for_prefix_e1()?;
        let restore = engine.restore_text_prefix(
            &saved.snapshot,
            TextPrefixRestoreIntent {
                prefix_ids: &prefix,
                pending_next_token: saved.snapshot.manifest.boundary.pending_next_token,
                transfer_bytes: args.transfer_bytes,
            },
        )?;
        let readback = engine.save_text_prefix(TextPrefixSaveRequest {
            identity,
            prefix_ids: &prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        ensure!(
            saved.snapshot.manifest.state_root == readback.snapshot.manifest.state_root,
            "post-restore readback state root differs"
        );
        ensure!(
            saved.snapshot.manifest.records == readback.snapshot.manifest.records,
            "post-restore readback receipts differ"
        );
        drop(readback);
        engine.step(false)?;
        let restored_pending_token = engine.token()?;
        let restored_pending_logits = engine.decode_logits()?;
        ensure!(
            baseline_pending_token == restored_pending_token,
            "restored pending next token differs"
        );
        ensure!(
            baseline_pending_logits
                .iter()
                .zip(&restored_pending_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "restored first decode logits are not bitwise identical"
        );
        let restored_token = engine.prefill(&suffix, None)?;
        let restored_logits = engine.decode_logits()?;
        ensure!(
            baseline_token == restored_token,
            "restored suffix token differs: {baseline_token} != {restored_token}"
        );
        ensure!(
            baseline_logits.len() == restored_logits.len(),
            "restored logits extent differs"
        );
        ensure!(
            baseline_logits
                .iter()
                .zip(&restored_logits)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "restored suffix logits are not bitwise identical"
        );
        let document = json!({
            "schema": "q38-prefix-e1-v1",
            "status": "PASS",
            "capacity": args.capacity,
            "prefix_tokens": prefix.len(),
            "suffix_tokens": suffix.len(),
            "block_tokens": args.block_tokens,
            "transfer_bytes": args.transfer_bytes,
            "save": saved.receipt,
            "restore": restore,
            "baseline_pending_next_token": baseline_pending_token,
            "restored_pending_next_token": restored_pending_token,
            "pending_decode_logits_bit_equal": true,
            "post_restore_readback_state_root": saved.snapshot.manifest.state_root,
            "post_restore_readback_pass": true,
            "baseline_suffix_token": baseline_token,
            "restored_suffix_token": restored_token,
            "baseline_logits_sha256": sha256_bytemuck(&baseline_logits),
            "restored_logits_sha256": sha256_bytemuck(&restored_logits),
            "logits_bit_equal": true,
            "package_dir": args.package_dir,
        });
        let manifest_sha256 = if let Some(package) = &args.package_dir {
            saved.snapshot.write_package(package)?;
            let bytes = std::fs::read(PrefixSnapshot::manifest_path(package))?;
            Some(sha256(&bytes))
        } else {
            None
        };
        let document = {
            let mut value = document;
            value["manifest_sha256"] = json!(manifest_sha256);
            value
        };
        std::fs::write(&args.output, serde_json::to_vec_pretty(&document)?)?;
        println!("{}", serde_json::to_string_pretty(&document)?);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn ensure_counts(prefix: &[u32], suffix: &[u32], capacity: usize) -> Result<()> {
    anyhow::ensure!(!prefix.is_empty(), "prefix is empty");
    anyhow::ensure!(!suffix.is_empty(), "suffix is empty");
    anyhow::ensure!(
        prefix
            .len()
            .checked_add(1)
            .and_then(|x| x.checked_add(suffix.len()))
            .is_some_and(|x| x <= capacity),
        "prefix plus one pending decode token plus suffix exceeds capacity"
    );
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::ensure_counts;

    #[test]
    fn capacity_includes_prefix_pending_token_and_suffix() {
        let prefix = vec![1; 32_768];
        let suffix = vec![2; 1_024];
        ensure_counts(&prefix, &suffix, 33_793).unwrap();
        assert!(ensure_counts(&prefix, &suffix, 33_792).is_err());

        let prefix_128k = vec![1; 131_072];
        ensure_counts(&prefix_128k, &suffix, 132_097).unwrap();
        assert!(ensure_counts(&prefix_128k, &suffix, 132_096).is_err());
    }
}

#[cfg(feature = "cuda")]
fn sha256_bytemuck(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256(&bytes)
}
