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
    model_spec::{Architecture, ModelSpec},
    prefix_state::{
        PrefixBoundary, PrefixBuilder, PrefixGeometry, PrefixIdentity, PrefixRecordKind,
        PrefixSnapshot, token_ids_sha256,
    },
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
    /// Prefill chunk size in rows; None uses the planner's admitted chunk.
    #[arg(long)]
    chunk: Option<usize>,
    #[arg(long)]
    prefix_cache: bool,
    #[arg(long)]
    expected_identity: Option<PathBuf>,
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

#[cfg(feature = "cuda")]
fn internally_consistent_foreign_snapshot(
    source: &PrefixSnapshot,
    identity: PrefixIdentity,
    spec: &ModelSpec,
    capacity: usize,
) -> Result<PrefixSnapshot> {
    let boundary: PrefixBoundary = source.manifest.boundary.clone();
    let block_tokens = source.manifest.block_tokens;
    let geometry = PrefixGeometry::new(
        spec,
        capacity,
        boundary.prefix_tokens,
        block_tokens,
        &identity,
    )?;
    let mut builder = PrefixBuilder::new(geometry, boundary)?;
    for (receipt, payload) in source.records() {
        let block = if receipt.key.kind == PrefixRecordKind::AttentionKvBlock {
            receipt.key.token_start / block_tokens
        } else {
            0
        };
        builder.add(receipt.key.kind, receipt.key.layer, block, payload.to_vec())?;
    }
    builder.finish()
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
        ensure!(
            spec.architecture == Architecture::Qwen35Text,
            "q38_prefix_e1 requires the Qwen3.8 text architecture"
        );
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: true,
            draft_model: None,
            runtime_dir: args.runtime_dir.clone(),
            prefix_cache: args.prefix_cache,
        };
        let mut engine = Backend::load(
            &args.model,
            args.device,
            args.capacity,
            args.chunk,
            settings,
        )?;
        engine.prefill(&prefix, None)?;
        let saved = engine.save_text_prefix(TextPrefixSaveRequest {
            prefix_ids: &prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        if let Some(path) = &args.expected_identity {
            let expected: PrefixIdentity = serde_json::from_slice(&std::fs::read(path)?)?;
            ensure!(
                saved.snapshot.manifest.identity == expected,
                "engine-derived prefix identity differs from expected identity file"
            );
        }
        let mut manifest_only_foreign = saved.snapshot.clone();
        manifest_only_foreign
            .manifest
            .identity
            .model_id
            .push_str("-foreign-negative-control");
        let manifest_only_error = engine
            .restore_text_prefix(
                &manifest_only_foreign,
                TextPrefixRestoreIntent {
                    prefix_ids: &prefix,
                    pending_next_token: saved.snapshot.manifest.boundary.pending_next_token,
                    transfer_bytes: args.transfer_bytes,
                },
            )
            .expect_err("foreign identity snapshot must be rejected");
        ensure!(
            manifest_only_error
                .to_string()
                .contains("prefix record identity differs from manifest"),
            "manifest-only foreign snapshot was not rejected by internal consistency: {manifest_only_error}"
        );

        let mut foreign_identity = saved.snapshot.manifest.identity.clone();
        foreign_identity
            .model_id
            .push_str("-foreign-negative-control");
        let consistent_foreign = internally_consistent_foreign_snapshot(
            &saved.snapshot,
            foreign_identity,
            &spec,
            args.capacity,
        )?;
        let foreign_error = engine
            .restore_text_prefix(
                &consistent_foreign,
                TextPrefixRestoreIntent {
                    prefix_ids: &prefix,
                    pending_next_token: saved.snapshot.manifest.boundary.pending_next_token,
                    transfer_bytes: args.transfer_bytes,
                },
            )
            .expect_err("internally consistent foreign snapshot must be rejected");
        ensure!(
            foreign_error
                .to_string()
                .contains("prefix snapshot identity differs"),
            "foreign snapshot was not rejected by engine identity: {foreign_error}"
        );
        engine.check_ready()?;
        let negative_readback = engine.save_text_prefix(TextPrefixSaveRequest {
            prefix_ids: &prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        ensure!(
            saved.snapshot.manifest.state_root == negative_readback.snapshot.manifest.state_root
                && saved.snapshot.manifest.records == negative_readback.snapshot.manifest.records,
            "foreign identity rejection changed resident prefix state"
        );
        drop(negative_readback);
        let committed_decode_token = saved.snapshot.manifest.boundary.pending_next_token;
        engine.step(true)?;
        let mut decoded_prefix = prefix.clone();
        decoded_prefix.push(committed_decode_token);
        let decoded_saved = engine.save_text_prefix(TextPrefixSaveRequest {
            prefix_ids: &decoded_prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        let decoded_boundary_pending_token = engine.token()?;
        ensure!(
            decoded_boundary_pending_token
                == decoded_saved.snapshot.manifest.boundary.pending_next_token,
            "decoded boundary pending token changed before save completed"
        );
        let decoded_boundary_logits = engine.decode_logits()?;
        ensure!(
            decoded_boundary_logits.iter().all(|v| v.is_finite()),
            "E1 nonfinite logits at decoded_boundary_logits"
        );
        engine.step(true)?;
        let baseline_pending_token = engine.token()?;
        let baseline_pending_logits = engine.decode_logits()?;
        ensure!(
            baseline_pending_logits.iter().all(|v| v.is_finite()),
            "E1 nonfinite logits at baseline_pending_logits"
        );
        let baseline_token = engine.prefill(&suffix, None)?;
        let baseline_logits = engine.decode_logits()?;
        ensure!(
            baseline_logits.iter().all(|v| v.is_finite()),
            "E1 nonfinite logits at baseline_logits"
        );
        engine.poison_state_for_prefix_e1()?;
        let restore = engine.restore_text_prefix(
            &decoded_saved.snapshot,
            TextPrefixRestoreIntent {
                prefix_ids: &decoded_prefix,
                pending_next_token: decoded_saved.snapshot.manifest.boundary.pending_next_token,
                transfer_bytes: args.transfer_bytes,
            },
        )?;
        let readback = engine.save_text_prefix(TextPrefixSaveRequest {
            prefix_ids: &decoded_prefix,
            block_tokens: args.block_tokens,
            transfer_bytes: args.transfer_bytes,
        })?;
        ensure!(
            decoded_saved.snapshot.manifest.state_root == readback.snapshot.manifest.state_root,
            "post-restore readback state root differs"
        );
        ensure!(
            decoded_saved.snapshot.manifest.records == readback.snapshot.manifest.records,
            "post-restore readback receipts differ"
        );
        drop(readback);
        let restored_boundary_pending_token = engine.token()?;
        ensure!(
            restored_boundary_pending_token == decoded_boundary_pending_token,
            "restored boundary pending token differs before the compared decode step"
        );
        let stale_logits_error = engine
            .decode_logits()
            .expect_err("decode logits must be invalid immediately after restore");
        ensure!(
            stale_logits_error
                .to_string()
                .contains("decode logits are invalid after prefix restore"),
            "unexpected stale-logits rejection: {stale_logits_error}"
        );
        engine.step(true)?;
        let restored_pending_token = engine.token()?;
        let restored_pending_logits = engine.decode_logits()?;
        ensure!(
            restored_pending_logits.iter().all(|v| v.is_finite()),
            "E1 nonfinite logits at restored_pending_logits"
        );
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
            restored_logits.iter().all(|v| v.is_finite()),
            "E1 nonfinite logits at restored_logits"
        );
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
        let mut baseline_after_step = decoded_prefix.clone();
        baseline_after_step.push(decoded_boundary_pending_token);
        let mut restored_after_step = decoded_prefix.clone();
        restored_after_step.push(restored_boundary_pending_token);
        let baseline_after_step_sha256 = token_ids_sha256(&baseline_after_step)?;
        let restored_after_step_sha256 = token_ids_sha256(&restored_after_step)?;
        let decoded_boundary_logits_sha256 = sha256_bytemuck(&decoded_boundary_logits);
        let token_trace = json!({
            "prefill_return_pending_t1": committed_decode_token,
            "decode_boundary_committed_prefix_ids": decoded_prefix.clone(),
            "decode_boundary_committed_prefix_sha256": token_ids_sha256(&decoded_prefix)?,
            "decode_boundary_pending_t2": decoded_boundary_pending_token,
            "decode_boundary_logits_sha256": decoded_boundary_logits_sha256,
            "baseline_step_committed_t2": decoded_boundary_pending_token,
            "baseline_post_step_pending_t3": baseline_pending_token,
            "baseline_committed_after_step_sha256": baseline_after_step_sha256,
            "baseline_suffix_ids": suffix.clone(),
            "baseline_suffix_first_pending": baseline_token,
            "restored_boundary_pending_t2": restored_boundary_pending_token,
            "restored_step_committed_t2": restored_boundary_pending_token,
            "restored_post_step_pending_t3": restored_pending_token,
            "restored_committed_after_step_sha256": restored_after_step_sha256,
            "restored_suffix_ids": suffix.clone(),
            "restored_suffix_first_pending": restored_token,
        });
        let document = json!({
            "schema": "q38-prefix-e1-v1",
            "status": "PASS",
            "capacity": args.capacity,
            "prefix_tokens": prefix.len(),
            "suffix_tokens": suffix.len(),
            "decode_committed_prefix_tokens": decoded_prefix.len(),
            "block_tokens": args.block_tokens,
            "transfer_bytes": args.transfer_bytes,
            "save": saved.receipt,
            "decode_boundary_save": decoded_saved.receipt,
            "foreign_identity_negative_control": true,
            "manifest_only_foreign_rejected_by_internal_consistency": true,
            "internally_consistent_foreign_rejected_by_engine_identity": true,
            "foreign_identity_rejected_before_mutation": true,
            "restored_logits_invalid_until_step": true,
            "post_restore_step_used_graph": true,
            "post_foreign_identity_readback_state_root":
                saved.snapshot.manifest.state_root.clone(),
            "restore": restore,
            "baseline_pending_next_token": baseline_pending_token,
            "restored_pending_next_token": restored_pending_token,
            "pending_decode_logits_bit_equal": true,
            "post_decode_restore_readback_state_root": decoded_saved.snapshot.manifest.state_root,
            "post_restore_readback_pass": true,
            "baseline_suffix_token": baseline_token,
            "restored_suffix_token": restored_token,
            "baseline_logits_sha256": sha256_bytemuck(&baseline_logits),
            "restored_logits_sha256": sha256_bytemuck(&restored_logits),
            "logits_bit_equal": true,
            "finite_logits": true,
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
            value["token_trace"] = token_trace;
            value
        };
        std::fs::write(&args.output, serde_json::to_vec_pretty(&document)?)?;
        println!("{}", serde_json::to_string_pretty(&document)?);
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
fn ensure_counts(prefix: &[u32], suffix: &[u32], capacity: usize) -> Result<()> {
    anyhow::ensure!(!prefix.is_empty(), "prefix is empty");
    anyhow::ensure!(!suffix.is_empty(), "suffix is empty");
    anyhow::ensure!(
        prefix
            .len()
            .checked_add(2)
            .and_then(|x| x.checked_add(suffix.len()))
            .is_some_and(|x| x <= capacity),
        "prefix plus two decode tokens plus suffix exceeds capacity"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ensure_counts;

    #[test]
    fn capacity_includes_prefix_pending_token_and_suffix() {
        let prefix = vec![1; 32_768];
        let suffix = vec![2; 1_024];
        ensure_counts(&prefix, &suffix, 33_794).unwrap();
        assert!(ensure_counts(&prefix, &suffix, 33_793).is_err());

        let prefix_128k = vec![1; 131_072];
        ensure_counts(&prefix_128k, &suffix, 132_098).unwrap();
        assert!(ensure_counts(&prefix_128k, &suffix, 132_097).is_err());
    }

    #[test]
    fn cpu_mock_dry_run_aligns_decode_boundary_sequences() {
        struct Mock {
            committed: Vec<u32>,
            pending: u32,
            next: u32,
        }
        impl Mock {
            fn token(&self) -> u32 {
                self.pending
            }
            fn step(&mut self) {
                self.committed.push(self.pending);
                self.next += 1;
                self.pending = self.next;
            }
            fn prefill_suffix(&mut self, suffix: &[u32]) -> u32 {
                self.committed.extend_from_slice(suffix);
                self.next += 1;
                self.pending = self.next;
                self.pending
            }
        }

        let prefix = [10u32, 11, 12];
        let suffix = [80u32, 81, 82];
        let mut baseline = Mock {
            committed: prefix.to_vec(),
            pending: 20,
            next: 20,
        };

        // Prefill returned T1=20; one step commits T1 and yields T2=21.
        baseline.step();
        let saved_committed = baseline.committed.clone();
        let saved_pending = baseline.token();
        assert_eq!(saved_committed, [10, 11, 12, 20]);
        assert_eq!(saved_pending, 21);

        // Baseline and restore both consume T2 at the saved boundary, then suffix.
        baseline.step();
        let baseline_post_step = baseline.token();
        let baseline_suffix = baseline.prefill_suffix(&suffix);
        let mut restored = Mock {
            committed: saved_committed,
            pending: saved_pending,
            next: 21,
        };
        assert_eq!(restored.token(), 21);
        restored.step();
        let restored_post_step = restored.token();
        let restored_suffix = restored.prefill_suffix(&suffix);

        assert_eq!(baseline_post_step, restored_post_step);
        assert_eq!(baseline_suffix, restored_suffix);
        assert_eq!(baseline.committed, restored.committed);
        assert_eq!(baseline.committed, [10, 11, 12, 20, 21, 80, 81, 82]);
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
