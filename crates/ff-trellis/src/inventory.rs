//! What a component's weight file actually contains, checked against what its
//! configuration says it should.
//!
//! The point is to catch a checkpoint whose sidecar and weights disagree before
//! any mathematics runs, and to give `inspect` something to report.

use anyhow::{Context, Result};
use ff_core::weights::ModelWeights;

use crate::config::{ComponentConfig, PositionEmbeddingMode, SparseStructureFlowArgs};

/// A component's tensors, summarized.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInventory {
    pub tensor_count: usize,
    pub payload_bytes: u64,
}

impl ComponentInventory {
    pub fn read(weights: &ModelWeights) -> Self {
        Self {
            tensor_count: weights.tensor_names().count(),
            payload_bytes: weights.indexed_payload_bytes(),
        }
    }
}

/// Check a component's weights against its configuration.
///
/// Only the components this crate can evaluate are checked in detail; the rest
/// report their inventory without a structural claim, because asserting a
/// layout nothing reads would be a guess.
pub fn verify(config: &ComponentConfig, weights: &ModelWeights) -> Result<ComponentInventory> {
    let inventory = ComponentInventory::read(weights);
    if let ComponentConfig::SparseStructureFlowModel(args) = config {
        verify_sparse_structure_flow(args, weights)
            .context("sparse-structure flow weights disagree with their configuration")?;
    }
    Ok(inventory)
}

/// The dense sparse-structure DiT.
///
/// Two structural variants are published and both are checked here. With
/// `share_mod` unset each block carries its own `adaLN_modulation`; with it
/// set, one projection is shared at the top level and each block carries only
/// its own `modulation` parameter. A learned `pos_emb` is present exactly when
/// `pe_mode` is `ape`.
fn verify_sparse_structure_flow(
    args: &SparseStructureFlowArgs,
    weights: &ModelWeights,
) -> Result<()> {
    let channels = args.model_channels;
    let patch_volume = args
        .patch_size
        .checked_pow(3)
        .context("TRELLIS patch volume overflows usize")?;

    require(
        weights,
        "input_layer.weight",
        &[channels, args.in_channels * patch_volume],
    )?;
    require(weights, "input_layer.bias", &[channels])?;
    require(
        weights,
        "out_layer.weight",
        &[args.out_channels * patch_volume, channels],
    )?;
    require(
        weights,
        "out_layer.bias",
        &[args.out_channels * patch_volume],
    )?;
    require(weights, "t_embedder.mlp.0.weight", &[channels, 256])?;
    require(weights, "t_embedder.mlp.2.weight", &[channels, channels])?;

    match args.pe_mode {
        PositionEmbeddingMode::Ape => {
            require(weights, "pos_emb", &[args.token_count()?, channels])?;
        }
        PositionEmbeddingMode::Rope => {
            anyhow::ensure!(
                !weights.contains("pos_emb"),
                "a rope sparse-structure flow model carries a learned pos_emb"
            );
        }
    }

    if args.share_mod {
        require(
            weights,
            "adaLN_modulation.1.weight",
            &[channels * 6, channels],
        )?;
    }

    let hidden = (channels as f64 * args.mlp_ratio) as usize;
    let head_dim = args.head_dim()?;
    for block in 0..args.num_blocks {
        let prefix = format!("blocks.{block}");
        require(
            weights,
            &format!("{prefix}.self_attn.to_qkv.weight"),
            &[channels * 3, channels],
        )?;
        require(
            weights,
            &format!("{prefix}.self_attn.to_out.weight"),
            &[channels, channels],
        )?;
        require(
            weights,
            &format!("{prefix}.cross_attn.to_q.weight"),
            &[channels, channels],
        )?;
        require(
            weights,
            &format!("{prefix}.cross_attn.to_kv.weight"),
            &[channels * 2, args.cond_channels],
        )?;
        require(
            weights,
            &format!("{prefix}.cross_attn.to_out.weight"),
            &[channels, channels],
        )?;
        require(
            weights,
            &format!("{prefix}.mlp.mlp.0.weight"),
            &[hidden, channels],
        )?;
        require(
            weights,
            &format!("{prefix}.mlp.mlp.2.weight"),
            &[channels, hidden],
        )?;
        require(weights, &format!("{prefix}.norm2.weight"), &[channels])?;
        if args.share_mod {
            require(weights, &format!("{prefix}.modulation"), &[channels * 6])?;
        } else {
            require(
                weights,
                &format!("{prefix}.adaLN_modulation.1.weight"),
                &[channels * 6, channels],
            )?;
        }
        let gamma = [args.num_heads, head_dim];
        if args.qk_rms_norm {
            require(
                weights,
                &format!("{prefix}.self_attn.q_rms_norm.gamma"),
                &gamma,
            )?;
            require(
                weights,
                &format!("{prefix}.self_attn.k_rms_norm.gamma"),
                &gamma,
            )?;
        }
        if args.qk_rms_norm_cross {
            require(
                weights,
                &format!("{prefix}.cross_attn.q_rms_norm.gamma"),
                &gamma,
            )?;
            require(
                weights,
                &format!("{prefix}.cross_attn.k_rms_norm.gamma"),
                &gamma,
            )?;
        }
    }
    Ok(())
}

fn require(weights: &ModelWeights, name: &str, shape: &[usize]) -> Result<()> {
    let metadata = weights
        .metadata(name)
        .with_context(|| format!("missing tensor {name}"))?;
    anyhow::ensure!(
        metadata.shape == shape,
        "tensor {name} has shape {:?}, expected {shape:?}",
        metadata.shape
    );
    Ok(())
}
