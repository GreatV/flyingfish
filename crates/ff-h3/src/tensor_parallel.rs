//! Does the H3 head layout admit the column split?
//!
//! A Megatron-shaped split for H3 cannot simply be assumed safe: the
//! wider-than-residual QKV width and the three-axis rotary application are
//! H3-specific and have to be checked against the head layout first.
//!
//! This module makes the check, and it is a check rather than an assumption in
//! two senses: the split is refused for a rank count the layout does not admit,
//! and every boundary it produces is derived from the released configuration
//! rather than written down.
//!
//! # What the layout actually is
//!
//! `to_q`, `to_k` and `to_v` are `[heads × head_dim, hidden]`, and the released
//! transformer has 56 heads of 128 against a hidden size of 5,376: the
//! projection is **7,168 wide against a 5,376-wide residual**. That asymmetry
//! is the first thing to check, and it turns out not to obstruct the split,
//! because the split runs along the projection's *output* axis and the residual
//! is only rejoined by `to_out.0`, whose output stays 5,376 wide on every rank
//! and is what the all-reduce sums. What the wider width does change is the
//! arithmetic of the share: a rank holds `7168 / N` rows of each of the three
//! projections, not `5376 / N`.
//!
//! # Why the boundary must be a head boundary
//!
//! After the projection the result is reshaped to
//! `[batch, length, heads, head_dim]` and two things happen inside one head's
//! `head_dim` lanes:
//!
//! * `norm_q` / `norm_k` are RMSNorm weights of shape `[head_dim]`, applied to
//!   the last axis. One weight vector serves every head, so it is replicated,
//!   not split — but only because no rank ever holds a fraction of a head.
//! * `apply_rotary` rotates the leading `6 × rope_freq_dim` lanes of each head
//!   (96 of 128 in the released configuration, with the remaining 32 passed
//!   through) by splitting them into halves and mixing the two. The cosine and
//!   sine tables are `[sequence, 6 × rope_freq_dim]` and are broadcast across
//!   the head axis: every head sees the same table. That is the three-axis
//!   rotary application — the three position axes are concatenated into that
//!   width — and it is head-independent.
//!
//! So both operations are per-head and identical across heads. A split that
//! lands on a multiple of `head_dim` gives each rank whole heads, and each rank
//! then performs exactly the normalization and rotation the single-device path
//! performs on those same heads. A split that cut a head would do neither, and
//! is refused here.
//!
//! # The correction this check forced
//!
//! A column split of "FFN gate and up" reads as two tensors. The released
//! checkpoint fuses them:
//! `ff.net.0.proj.weight` is `[28672, 5376]` — two 14,336-wide halves in one
//! tensor, values above gates, which `core::swiglu` separates by narrowing the
//! projected output in half. A contiguous row split of that tensor is therefore
//! **not** a column-parallel shard of the feed-forward: at two ranks it would
//! give rank 0 all 14,336 values and rank 1 all 14,336 gates, and neither rank
//! could compute the activation at all. The split has to take each rank's share
//! of *each* half, which is `TensorPartition::SegmentedShard`.
//!
//! This is the one substantive thing the check changed, and it is why the
//! layout has to be read rather than assumed: the arithmetic all works out on
//! the config's `ffn_dim`, and the checkpoint stores twice that.
//!
//! # What this does not claim
//!
//! That the split is numerically free. It is not: the two row-parallel
//! projections sum partial products across ranks, and that sum is not the
//! single-device sum. That cost is the subject of `numerics.collective`, and
//! measuring the divergence is its own exercise. This module says only that
//! the partition is *expressible* on this layout, and at which rank counts.

use crate::config::TransformerConfig;
use anyhow::{Context, Result};
use ff_core::weights::{TensorAxis, TensorPartition};

/// The released feed-forward input projection fuses the SwiGLU values and gates
/// into one tensor, which `core::swiglu` separates by halving the projected
/// width.
pub const SWIGLU_SEGMENTS: u32 = 2;

/// AdaLN emits eighteen residual-width modulation terms per block, which is
/// what makes `adaln_proj.linear` 96,768 wide against a 5,376-wide residual.
const ADALN_MODULATION_TERMS: usize = 18;

/// One tensor of an H3 transformer block under a Megatron-shaped split, with
/// the reason it is split the way it is.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitTensor {
    /// The block-relative suffix, as `core.rs` names it. A caller joins its own
    /// `transformer_blocks.{index}` prefix.
    pub suffix: &'static str,
    /// The released tensor's shape, before any split. The per-head norms are
    /// one-dimensional in the checkpoint and are recorded that way.
    pub shape: Vec<usize>,
    /// What this rank holds. `Whole` is a replicated tensor.
    pub partition: TensorPartition,
    /// Whether the block's output needs a cross-rank sum after this tensor.
    pub all_reduce_after: bool,
    /// Rows or columns this rank holds along the split axis, and the whole
    /// count when replicated.
    pub rank_extent: usize,
}

impl SplitTensor {
    /// The length of the axis this tensor is split along, or of its own last
    /// axis when it is replicated. A replicated tensor has no split axis, so
    /// asking for one would be a category error; this is what a caller means.
    pub fn axis_length(&self) -> Result<usize> {
        match self.partition {
            TensorPartition::Whole => self
                .shape
                .last()
                .copied()
                .context("a split tensor has at least one axis"),
            TensorPartition::Shard { axis, .. } | TensorPartition::SegmentedShard { axis, .. } => {
                let index = match axis {
                    TensorAxis::Rows => 0,
                    TensorAxis::Columns => 1,
                };
                self.shape
                    .get(index)
                    .copied()
                    .with_context(|| format!("{} has no axis {index} to split", self.suffix))
            }
        }
    }
}

/// The Megatron-shaped split of one H3 transformer block, resolved against a
/// released configuration and a rank count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3BlockSplit {
    pub rank: u32,
    pub ranks: u32,
    /// Attention heads this rank owns. Every rank owns the same number: the
    /// rank count divides the head count or the split is refused.
    pub heads_per_rank: usize,
    /// The projection width one rank holds, `heads_per_rank × head_dim`.
    pub attention_width_per_rank: usize,
    /// The feed-forward width one rank holds.
    pub feed_forward_width_per_rank: usize,
    /// Cross-rank sums one block performs: one after the attention output
    /// projection, one after the feed-forward down projection.
    pub all_reduces_per_block: usize,
    pub tensors: Vec<SplitTensor>,
}

impl H3BlockSplit {
    /// Resolve the split for `rank` of `ranks` against `config`, or explain why
    /// this layout does not admit that rank count.
    ///
    /// Refusals are deliberate rather than rounded. A rank count that does not
    /// divide the head count would put a fraction of a head on some rank, which
    /// breaks the per-head normalization and rotation the layout depends on; a
    /// rank count that does not divide the feed-forward width would make the
    /// ring's chunks unequal. Either would still produce numbers, and they
    /// would be the wrong ones.
    pub fn resolve(config: &TransformerConfig, rank: u32, ranks: u32) -> Result<Self> {
        anyhow::ensure!(ranks > 0, "a tensor-parallel split needs at least one rank");
        anyhow::ensure!(
            rank < ranks,
            "rank {rank} is outside a {ranks}-rank split of an H3 block"
        );
        let rank_count = usize::try_from(ranks).context("rank count exceeds usize")?;
        let heads = config.num_attention_heads;
        let head_dim = config.attention_head_dim;
        let hidden = config.hidden_size;
        let feed_forward = config.ffn_dim;
        anyhow::ensure!(
            heads.is_multiple_of(rank_count),
            "a {ranks}-rank column split does not divide {heads} attention heads; splitting \
             inside a head would break the per-head QK normalization and rotary application \
             that make each rank's arithmetic the single-device arithmetic on its own heads"
        );
        anyhow::ensure!(
            feed_forward.is_multiple_of(rank_count),
            "a {ranks}-rank column split does not divide the {feed_forward}-wide feed-forward \
             intermediate"
        );
        let attention_width = heads
            .checked_mul(head_dim)
            .context("H3 attention projection width overflows usize")?;
        let fused_feed_forward = feed_forward
            .checked_mul(SWIGLU_SEGMENTS as usize)
            .context("H3 fused feed-forward width overflows usize")?;
        let adaln_width = hidden
            .checked_mul(ADALN_MODULATION_TERMS)
            .context("H3 AdaLN projection width overflows usize")?;
        let heads_per_rank = heads / rank_count;
        let attention_width_per_rank = attention_width / rank_count;
        debug_assert_eq!(attention_width_per_rank, heads_per_rank * head_dim);
        let feed_forward_width_per_rank = feed_forward / rank_count;

        let column = TensorPartition::Shard {
            axis: TensorAxis::Rows,
            rank,
            ranks,
        };
        let row = TensorPartition::Shard {
            axis: TensorAxis::Columns,
            rank,
            ranks,
        };
        let tensors = vec![
            SplitTensor {
                suffix: "attn.to_q.weight",
                shape: vec![attention_width, hidden],
                partition: column,
                all_reduce_after: false,
                rank_extent: attention_width_per_rank,
            },
            SplitTensor {
                suffix: "attn.to_k.weight",
                shape: vec![attention_width, hidden],
                partition: column,
                all_reduce_after: false,
                rank_extent: attention_width_per_rank,
            },
            SplitTensor {
                suffix: "attn.to_v.weight",
                shape: vec![attention_width, hidden],
                partition: column,
                all_reduce_after: false,
                rank_extent: attention_width_per_rank,
            },
            SplitTensor {
                suffix: "attn.norm_q.weight",
                shape: vec![head_dim],
                partition: TensorPartition::Whole,
                all_reduce_after: false,
                rank_extent: head_dim,
            },
            SplitTensor {
                suffix: "attn.norm_k.weight",
                shape: vec![head_dim],
                partition: TensorPartition::Whole,
                all_reduce_after: false,
                rank_extent: head_dim,
            },
            SplitTensor {
                suffix: "attn.to_out.0.weight",
                shape: vec![hidden, attention_width],
                partition: row,
                all_reduce_after: true,
                rank_extent: attention_width_per_rank,
            },
            SplitTensor {
                suffix: "ff.net.0.proj.weight",
                shape: vec![fused_feed_forward, hidden],
                partition: TensorPartition::SegmentedShard {
                    axis: TensorAxis::Rows,
                    rank,
                    ranks,
                    segments: SWIGLU_SEGMENTS,
                },
                all_reduce_after: false,
                rank_extent: feed_forward_width_per_rank * SWIGLU_SEGMENTS as usize,
            },
            SplitTensor {
                suffix: "ff.net.2.weight",
                shape: vec![hidden, feed_forward],
                partition: row,
                all_reduce_after: true,
                rank_extent: feed_forward_width_per_rank,
            },
            SplitTensor {
                suffix: "norm1.weight",
                shape: vec![hidden],
                partition: TensorPartition::Whole,
                all_reduce_after: false,
                rank_extent: hidden,
            },
            SplitTensor {
                suffix: "norm2.weight",
                shape: vec![hidden],
                partition: TensorPartition::Whole,
                all_reduce_after: false,
                rank_extent: hidden,
            },
            SplitTensor {
                suffix: "adaln_proj.linear.weight",
                shape: vec![adaln_width, config.time_embed_dim],
                partition: TensorPartition::Whole,
                all_reduce_after: false,
                rank_extent: config.time_embed_dim,
            },
        ];
        let all_reduces_per_block = tensors
            .iter()
            .filter(|tensor| tensor.all_reduce_after)
            .count();
        let split = Self {
            rank,
            ranks,
            heads_per_rank,
            attention_width_per_rank,
            feed_forward_width_per_rank,
            all_reduces_per_block,
            tensors,
        };
        split.validate(config)?;
        Ok(split)
    }

    /// Every rank count this layout admits, in increasing order. A caller that
    /// wants to know what it may ask for asks this rather than guessing.
    pub fn admitted_rank_counts(config: &TransformerConfig, maximum: u32) -> Vec<u32> {
        (1..=maximum)
            .filter(|ranks| Self::resolve(config, 0, *ranks).is_ok())
            .collect()
    }

    /// The half-open row or column ranges this rank holds of a split tensor, in
    /// the order its slice concatenates them. A replicated tensor's range is
    /// the whole of it, on every rank; a fused projection names one range per
    /// segment.
    pub fn ranges(&self, tensor: &SplitTensor) -> Result<Vec<std::ops::Range<usize>>> {
        tensor.partition.ranges(tensor.axis_length()?)
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let rank_count = usize::try_from(self.ranks).context("rank count exceeds usize")?;
        anyhow::ensure!(
            self.heads_per_rank * rank_count == config.num_attention_heads,
            "resolved head shares do not cover the head count"
        );
        anyhow::ensure!(
            self.all_reduces_per_block == 2,
            "a Megatron-shaped H3 block sums once after the attention output projection and \
             once after the feed-forward down projection"
        );
        for tensor in &self.tensors {
            let ranges = self.ranges(tensor)?;
            let mut extent = 0;
            for range in &ranges {
                anyhow::ensure!(
                    range.start < range.end,
                    "an empty share of {} is not a split",
                    tensor.suffix
                );
                extent += range.end - range.start;
                if tensor.suffix.starts_with("attn.")
                    && tensor.partition != TensorPartition::Whole
                    && tensor
                        .shape
                        .contains(&(config.num_attention_heads * config.attention_head_dim))
                {
                    anyhow::ensure!(
                        range.start.is_multiple_of(config.attention_head_dim)
                            && range.end.is_multiple_of(config.attention_head_dim),
                        "the share {range:?} of {} cuts an attention head",
                        tensor.suffix
                    );
                }
            }
            anyhow::ensure!(
                extent == tensor.rank_extent,
                "the recorded extent of {} disagrees with its ranges",
                tensor.suffix
            );
            if let TensorPartition::SegmentedShard { segments, .. } = tensor.partition {
                anyhow::ensure!(
                    ranges.len() == segments as usize,
                    "the split of {} lost a fused segment",
                    tensor.suffix
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released transformer: 56 heads of 128 against a 5,376-wide residual,
    /// a 14,336-wide feed-forward intermediate, and a rotary table built from
    /// three position axes of 16 frequencies each.
    fn released() -> TransformerConfig {
        let config = TransformerConfig::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/h3-transformer/config.json"
        ))
        .expect("the released H3 transformer configuration fixture is valid");
        assert_eq!(config.num_attention_heads, 56);
        assert_eq!(config.attention_head_dim, 128);
        assert_eq!(config.hidden_size, 5_376);
        assert_eq!(config.ffn_dim, 14_336);
        assert_eq!(config.rope_freq_dim, 16);
        config
    }

    /// The layout admits the column split at the two rank counts the cost
    /// table models.
    #[test]
    fn the_released_head_layout_admits_the_column_split_at_two_and_four_ranks() {
        let config = released();
        for ranks in [2u32, 4] {
            for rank in 0..ranks {
                let split = H3BlockSplit::resolve(&config, rank, ranks).unwrap();
                assert_eq!(split.heads_per_rank, 56 / ranks as usize);
                assert_eq!(split.attention_width_per_rank, 7_168 / ranks as usize);
                assert_eq!(split.feed_forward_width_per_rank, 14_336 / ranks as usize);
                assert_eq!(split.all_reduces_per_block, 2);
            }
        }
        let split = H3BlockSplit::resolve(&config, 0, 4).unwrap();
        assert_eq!(
            split.all_reduces_per_block * (config.num_layers + config.num_refiner_layers),
            104
        );
    }

    /// The projection is wider than the residual, and the split is of the
    /// projection, not of the residual.
    #[test]
    fn the_wider_than_residual_projection_is_what_divides() {
        let config = released();
        let split = H3BlockSplit::resolve(&config, 1, 4).unwrap();
        let attention_width = config.num_attention_heads * config.attention_head_dim;
        assert_eq!(attention_width, 7_168);
        assert!(attention_width > config.hidden_size);
        let qkv = split
            .tensors
            .iter()
            .find(|tensor| tensor.suffix == "attn.to_q.weight")
            .unwrap();
        assert_eq!(qkv.shape, [7_168, 5_376]);
        assert_eq!(split.ranges(qkv).unwrap(), vec![1_792..3_584]);
        let out = split
            .tensors
            .iter()
            .find(|tensor| tensor.suffix == "attn.to_out.0.weight")
            .unwrap();
        assert_eq!(out.shape, [5_376, 7_168]);
        assert!(out.all_reduce_after);
        assert_eq!(split.ranges(out).unwrap(), vec![1_792..3_584]);
    }

    /// Every attention boundary is a multiple of `head_dim`, which is what
    /// keeps the per-head QK norms replicated and the rotary application
    /// identical to the single-device one.
    #[test]
    fn no_admitted_rank_count_cuts_a_head() {
        let config = released();
        for ranks in H3BlockSplit::admitted_rank_counts(&config, 8) {
            for rank in 0..ranks {
                let split = H3BlockSplit::resolve(&config, rank, ranks).unwrap();
                for tensor in &split.tensors {
                    for range in split.ranges(tensor).unwrap() {
                        if tensor.suffix.starts_with("attn.")
                            && tensor.partition != TensorPartition::Whole
                        {
                            assert_eq!(range.start % config.attention_head_dim, 0);
                            assert_eq!(range.end % config.attention_head_dim, 0);
                        }
                    }
                }
                for suffix in ["attn.norm_q.weight", "attn.norm_k.weight"] {
                    let norm = split
                        .tensors
                        .iter()
                        .find(|tensor| tensor.suffix == suffix)
                        .unwrap();
                    assert_eq!(norm.partition, TensorPartition::Whole);
                    assert_eq!(norm.rank_extent, config.attention_head_dim);
                }
            }
        }
    }

    /// The shapes this module splits are the released ones, read from the
    /// checkpoint's own safetensors header rather than derived from the config.
    /// This is the test that caught the fused feed-forward: `ffn_dim` is
    /// 14,336 and `ff.net.0.proj.weight` is 28,672 rows.
    #[test]
    fn the_modelled_shapes_are_the_released_checkpoint_shapes() {
        let Some(root) = std::env::var_os("FF_H3_TRANSFORMER_CHECKPOINT") else {
            eprintln!("skipping: set FF_H3_TRANSFORMER_CHECKPOINT to check released weights");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let index = root.join("diffusion_pytorch_model.safetensors.index.json");
        let map: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
        let weight_map = map["weight_map"].as_object().unwrap();
        let config = released();
        let split = H3BlockSplit::resolve(&config, 0, 4).unwrap();
        let mut headers: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        for tensor in &split.tensors {
            let name = format!("transformer_blocks.0.{}", tensor.suffix);
            let shard = weight_map[&name].as_str().unwrap().to_owned();
            let header = headers.entry(shard.clone()).or_insert_with(|| {
                let bytes = std::fs::read(root.join(&shard)).unwrap();
                let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
                serde_json::from_slice(&bytes[8..8 + length]).unwrap()
            });
            let shape = header[&name]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(
                shape, tensor.shape,
                "{name} is {shape:?} in the checkpoint, modelled as {:?}",
                tensor.shape
            );
        }
        let fused = split
            .tensors
            .iter()
            .find(|tensor| tensor.suffix == "ff.net.0.proj.weight")
            .unwrap();
        assert_eq!(fused.shape, [28_672, 5_376]);
        assert_eq!(fused.shape[0], config.ffn_dim * SWIGLU_SEGMENTS as usize);
        assert!(matches!(
            fused.partition,
            TensorPartition::SegmentedShard { segments: 2, .. }
        ));
    }

    /// The fused feed-forward gives each rank its share of the values *and* its
    /// share of the gates. A contiguous split would give rank 0 all 14,336
    /// values and rank 1 all 14,336 gates, and neither could compute SwiGLU.
    #[test]
    fn the_fused_feed_forward_splits_values_and_gates_separately() {
        let config = released();
        let fused_of = |rank: u32, ranks: u32| {
            let split = H3BlockSplit::resolve(&config, rank, ranks).unwrap();
            let tensor = split
                .tensors
                .iter()
                .find(|tensor| tensor.suffix == "ff.net.0.proj.weight")
                .unwrap()
                .clone();
            (split.ranges(&tensor).unwrap(), tensor)
        };
        let (ranges, tensor) = fused_of(0, 2);
        assert_eq!(ranges, vec![0..7_168, 14_336..21_504]);
        assert_eq!(tensor.rank_extent, 14_336);
        let (ranges, _) = fused_of(1, 2);
        assert_eq!(ranges, vec![7_168..14_336, 21_504..28_672]);
        let (ranges, tensor) = fused_of(3, 4);
        assert_eq!(ranges, vec![10_752..14_336, 25_088..28_672]);
        let values = ranges[0].end - ranges[0].start;
        let gates = ranges[1].end - ranges[1].start;
        assert_eq!(values, gates);
        assert_eq!(values + gates, tensor.rank_extent);
        assert!(ranges[0].end <= config.ffn_dim);
        assert!(ranges[1].start >= config.ffn_dim);
        let split = H3BlockSplit::resolve(&config, 3, 4).unwrap();
        let down = split
            .tensors
            .iter()
            .find(|tensor| tensor.suffix == "ff.net.2.weight")
            .unwrap();
        assert_eq!(down.shape, [5_376, 14_336]);
        assert_eq!(down.rank_extent, values);
        assert!(down.all_reduce_after);
    }

    /// A rank count the head layout does not admit is refused, not rounded.
    #[test]
    fn a_rank_count_that_would_cut_a_head_is_refused() {
        let config = released();
        assert_eq!(
            H3BlockSplit::admitted_rank_counts(&config, 8),
            vec![1, 2, 4, 7, 8],
            "56 heads and a 14,336-wide feed-forward admit exactly these counts up to eight"
        );
        for ranks in [3u32, 5, 6] {
            let error = H3BlockSplit::resolve(&config, 0, ranks)
                .unwrap_err()
                .to_string();
            assert!(error.contains("attention heads"), "{error}");
            assert!(error.contains("inside a head"), "{error}");
        }
        assert!(H3BlockSplit::resolve(&config, 4, 4).is_err());
        assert!(H3BlockSplit::resolve(&config, 0, 0).is_err());
    }

    /// One rank is the whole block: the shares are the released shapes, and
    /// nothing is partitioned.
    #[test]
    fn one_rank_holds_the_whole_block() {
        let config = released();
        let split = H3BlockSplit::resolve(&config, 0, 1).unwrap();
        assert_eq!(split.heads_per_rank, 56);
        assert_eq!(split.attention_width_per_rank, 7_168);
        assert_eq!(split.feed_forward_width_per_rank, 14_336);
        for tensor in &split.tensors {
            let ranges = split.ranges(tensor).unwrap();
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(
                ranges.last().unwrap().end,
                tensor.axis_length().unwrap(),
                "{}",
                tensor.suffix
            );
            let covered: usize = ranges.iter().map(|range| range.end - range.start).sum();
            assert_eq!(covered, tensor.axis_length().unwrap(), "{}", tensor.suffix);
        }
    }

    /// The shares of every rank cover each split tensor exactly once, with no
    /// gap and no overlap.
    #[test]
    fn the_ranks_shares_tile_every_split_tensor() {
        let config = released();
        for ranks in [2u32, 4, 7, 8] {
            let splits = (0..ranks)
                .map(|rank| H3BlockSplit::resolve(&config, rank, ranks).unwrap())
                .collect::<Vec<_>>();
            for (index, tensor) in splits[0].tensors.iter().enumerate() {
                if tensor.partition == TensorPartition::Whole {
                    for split in &splits {
                        let ranges = split.ranges(&split.tensors[index]).unwrap();
                        assert_eq!(ranges, vec![0..tensor.axis_length().unwrap()]);
                    }
                    continue;
                }
                let mut covered = splits
                    .iter()
                    .flat_map(|split| split.ranges(&split.tensors[index]).unwrap())
                    .flat_map(|range| range.collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                covered.sort_unstable();
                assert_eq!(
                    covered,
                    (0..tensor.axis_length().unwrap()).collect::<Vec<_>>(),
                    "{}",
                    tensor.suffix
                );
            }
        }
    }
}
