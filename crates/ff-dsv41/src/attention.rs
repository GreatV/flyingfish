//! Sliding-window attention with the compressed-KV sparse indexer.
//!
//! Reference: `inference/model.py` (`Compressor`, `Indexer`, `Attention`,
//! `select_candidate_blocks`). These are host-reference scalar
//! implementations over materialized F32 tensors.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::math::{apply_rotary_emb, fp4_act_round_trip, fp8_act_round_trip, rms_norm};

/// Sparse multi-head attention over gathered KV rows with an attention sink.
///
/// `topk_idxs` is `[batch, queries, k]` with `-1` marking an empty slot; a
/// query whose slots are all empty produces zeros, matching the training
/// kernel's finite-lower-bound convention.
pub fn sparse_attn(
    q: &Tensor,
    kv: &Tensor,
    attn_sink: &Tensor,
    topk_idxs: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let dims = q.dims();
    ensure!(dims.len() == 4, "queries must be [batch, seq, heads, dim]");
    let [batch, seq, heads, head_dim] = [dims[0], dims[1], dims[2], dims[3]];
    ensure!(
        kv.dims().len() == 3 && kv.dims()[0] == batch && kv.dims()[2] == head_dim,
        "kv must be [batch, positions, dim] sharing the query's batch and head dim"
    );
    let index_dims = topk_idxs.dims();
    ensure!(
        index_dims.len() == 3 && index_dims[0] == batch && index_dims[1] == seq,
        "topk indices must be [batch, seq, k]"
    );
    let k = index_dims[2];
    let q_values = q
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read queries")?;
    let kv_values = kv
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read kv")?;
    let sink_values = attn_sink
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read attention sinks")?;
    let indexes = topk_idxs
        .flatten_all()?
        .to_vec1::<i64>()
        .context("read topk indices")?;
    let positions = kv.dims()[1];
    let mut output = vec![0.0f32; batch * seq * heads * head_dim];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let query = &q_values[((b * seq + s) * heads + h) * head_dim..][..head_dim];
                let mut scores = Vec::with_capacity(k);
                let mut rows = Vec::with_capacity(k);
                for slot in 0..k {
                    let index = indexes[(b * seq + s) * k + slot];
                    if index < 0 {
                        continue;
                    }
                    let index = index as usize;
                    ensure!(
                        index < positions,
                        "topk index {index} beyond {positions} kv rows"
                    );
                    let row = &kv_values[(b * positions + index) * head_dim..][..head_dim];
                    let score: f32 = query
                        .iter()
                        .zip(row.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale as f32;
                    scores.push(score);
                    rows.push(index);
                }
                let sink = sink_values[h];
                let max = scores
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
                    .max(sink)
                    .max(-1.0e30);
                let mut total = (sink - max).exp();
                let mut weighted = vec![0.0f32; head_dim];
                for (score, index) in scores.iter().zip(rows.iter()) {
                    let weight = (score - max).exp();
                    total += weight;
                    let row = &kv_values[(b * positions + index) * head_dim..][..head_dim];
                    for (accumulator, value) in weighted.iter_mut().zip(row.iter()) {
                        *accumulator += weight * value;
                    }
                }
                let target = &mut output[((b * seq + s) * heads + h) * head_dim..][..head_dim];
                for (slot, value) in target.iter_mut().zip(weighted.iter()) {
                    *slot = value / total;
                }
            }
        }
    }
    Ok(Tensor::from_vec(output, (batch, seq, heads, head_dim), q.device())?.to_dtype(q.dtype())?)
}

/// Level one of the two-level top-k: keep the `topk_blocks` highest-scoring
/// blocks per query. Positions the query cannot reach carry `-inf` scores, so
/// an unreachable block scores `-inf` and is dropped; the block holding the
/// query's newest reachable position is pinned in.
///
/// Returns a U8 mask shaped like `scores`, 1 where the position survives.
pub fn select_candidate_blocks(
    scores: &Tensor,
    compress_lens: usize,
    topk_blocks: usize,
    block_size: usize,
) -> Result<Tensor> {
    let dims = scores.dims();
    ensure!(dims.len() >= 2, "candidate scores must be [..., positions]");
    let width = dims[dims.len() - 1];
    let values = scores
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read candidate scores")?;
    let rows = values.len() / width;
    let blocks = width.div_ceil(block_size);
    let mut mask = vec![0u8; values.len()];
    for row in 0..rows {
        let mut block_scores = vec![f32::NEG_INFINITY; blocks];
        for (block, score) in block_scores.iter_mut().enumerate() {
            let start = block * block_size;
            let end = ((block + 1) * block_size).min(width);
            for column in start..end {
                *score = score.max(values[row * width + column]);
            }
        }
        let last = compress_lens as isize - 1;
        if last >= 0 {
            block_scores[(last / block_size as isize) as usize] = f32::INFINITY;
        }
        let mut order = (0..blocks).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            block_scores[*b]
                .partial_cmp(&block_scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let wanted = topk_blocks.min(blocks);
        let mut chosen = 0;
        for block in order {
            if chosen == wanted {
                break;
            }
            if block_scores[block] > f32::NEG_INFINITY {
                let start = block * block_size;
                let end = ((block + 1) * block_size).min(width);
                for column in start..end {
                    mask[row * width + column] = 1;
                }
                chosen += 1;
            }
        }
    }
    Ok(Tensor::from_vec(mask, dims.to_vec(), scores.device())?)
}

/// One compressor: pools `ratio` consecutive tokens into one KV latent with a
/// per-channel softmax gate. A trailing partial group survives across decode
/// steps in the carried state.
pub struct Compressor {
    pub ratio: usize,
    pub head_dim: usize,
    pub norm_weight: Tensor,
    pub wkv: Tensor,
    pub wgate: Option<Tensor>,
    pub kv_state: Tensor,
    pub score_state: Tensor,
    batch: usize,
}

impl Compressor {
    pub fn new(
        ratio: usize,
        head_dim: usize,
        norm_weight: Tensor,
        wkv: Tensor,
        wgate: Option<Tensor>,
        batch: usize,
    ) -> Result<Self> {
        ensure!(ratio > 0, "compress ratio must be positive");
        ensure!(
            wkv.dims().len() == 2 && wkv.dims()[0] == head_dim,
            "compressor wkv must be [head_dim, hidden]"
        );
        if ratio > 1 {
            ensure!(
                wgate.as_ref().is_some_and(|gate| gate.dims() == wkv.dims()),
                "ratio {ratio} needs a gate projection shaped like wkv"
            );
        }
        let device = norm_weight.device();
        let kv_state = Tensor::zeros((batch, ratio, head_dim), DType::F32, device)?;
        let score_state = Tensor::full(f32::NEG_INFINITY, (batch, ratio), device)?;
        Ok(Self {
            ratio,
            head_dim,
            norm_weight,
            wkv,
            wgate,
            kv_state,
            score_state,
            batch,
        })
    }

    /// Returns the pre-RoPE latents for the groups this step completes, or
    /// `None` while a group is still filling up.
    pub fn forward(&mut self, x: &Tensor, start_pos: usize) -> Result<Option<Tensor>> {
        let dims = x.dims();
        ensure!(
            dims.len() == 3,
            "compressor input must be [batch, seq, dim]"
        );
        let [batch, seqlen, hidden] = [dims[0], dims[1], dims[2]];
        ensure!(batch == self.batch, "compressor batch changed mid-run");
        ensure!(
            self.wkv.dims()[1] == hidden,
            "compressor wkv expects {} channels, found {hidden}",
            self.wkv.dims()[1]
        );
        let (ratio, head_dim) = (self.ratio, self.head_dim);
        let x_values = x
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read compressor input")?;
        let wkv = self
            .wkv
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read compressor wkv")?;
        let wgate = match &self.wgate {
            Some(gate) => Some(
                gate.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .context("read compressor wgate")?,
            ),
            None => None,
        };
        let project = |weights: &[f32], token: usize, b: usize| -> Vec<f32> {
            let base = (b * seqlen + token) * hidden;
            (0..head_dim)
                .map(|row| {
                    let mut sum = 0.0;
                    for column in 0..hidden {
                        sum += weights[row * hidden + column] * x_values[base + column];
                    }
                    sum
                })
                .collect()
        };
        let mut kv_state = self.kv_state.flatten_all()?.to_vec1::<f32>()?;
        let mut score_state = self.score_state.flatten_all()?.to_vec1::<f32>()?;
        let mut pooled: Vec<f32> = Vec::new();
        let mut produced = 0usize;
        let norm = |pooled: Vec<f32>, produced: usize| -> Result<Tensor> {
            let tensor = Tensor::from_vec(pooled, (batch, produced, head_dim), x.device())
                .map_err(anyhow::Error::from)?;
            let normalized = rms_norm(&tensor, &self.norm_weight, 1e-20)?;
            Ok(normalized.to_dtype(x.dtype())?)
        };

        if start_pos == 0 {
            let remainder = seqlen % ratio;
            let complete = seqlen - remainder;
            for token in 0..complete {
                let group = token / ratio;
                let slot = token % ratio;
                for b in 0..batch {
                    let kv = project(&wkv, token, b);
                    let gate = wgate
                        .as_ref()
                        .map(|wgate| project(wgate, token, b))
                        .unwrap_or_else(|| vec![0.0; head_dim]);
                    let base = ((b * (complete / ratio) + group) * ratio + slot) * head_dim;
                    // Scores and latents for one group are collected first, then
                    // pooled with a per-channel softmax over the group's tokens.
                    stash(&mut kv_state, &mut score_state, b, ratio, slot, &kv, &gate);
                    let _ = base;
                }
                if slot + 1 == ratio {
                    for b in 0..batch {
                        pooled.extend(pool_group(&kv_state, &score_state, b, ratio, head_dim));
                    }
                    produced += 1;
                }
            }
            // Reset the carried state to just the trailing partial group.
            let mut tail_state = vec![0.0f32; kv_state.len()];
            let mut tail_scores = vec![f32::NEG_INFINITY; score_state.len()];
            if remainder > 0 {
                for (slot, token) in (complete..seqlen).enumerate() {
                    for b in 0..batch {
                        let kv = project(&wkv, token, b);
                        let gate = wgate
                            .as_ref()
                            .map(|wgate| project(wgate, token, b))
                            .unwrap_or_else(|| vec![0.0; head_dim]);
                        stash(
                            &mut tail_state,
                            &mut tail_scores,
                            b,
                            ratio,
                            slot,
                            &kv,
                            &gate,
                        );
                    }
                }
            }
            kv_state.copy_from_slice(&tail_state);
            score_state.copy_from_slice(&tail_scores);
        } else {
            ensure!(
                seqlen == 1,
                "decode compression handles one token at a time"
            );
            for b in 0..batch {
                let kv = project(&wkv, 0, b);
                let gate = wgate
                    .as_ref()
                    .map(|wgate| project(wgate, 0, b))
                    .unwrap_or_else(|| vec![0.0; head_dim]);
                stash(
                    &mut kv_state,
                    &mut score_state,
                    b,
                    ratio,
                    start_pos % ratio,
                    &kv,
                    &gate,
                );
            }
            if (start_pos + 1).is_multiple_of(ratio) {
                for b in 0..batch {
                    pooled.extend(pool_group(&kv_state, &score_state, b, ratio, head_dim));
                }
                produced = 1;
            }
        }

        self.kv_state = Tensor::from_vec(kv_state, (batch, ratio, head_dim), x.device())
            .map_err(anyhow::Error::from)?;
        self.score_state = Tensor::from_vec(score_state, (batch, ratio), x.device())
            .map_err(anyhow::Error::from)?;
        if produced == 0 {
            return Ok(None);
        }
        Ok(Some(norm(pooled, produced)?))
    }
}

fn stash(
    kv_state: &mut [f32],
    score_state: &mut [f32],
    batch: usize,
    ratio: usize,
    slot: usize,
    kv: &[f32],
    gate: &[f32],
) {
    let base = (batch * ratio + slot) * kv.len();
    for (target, value) in kv_state[base..base + kv.len()].iter_mut().zip(kv.iter()) {
        *target = *value;
    }
    for (target, value) in score_state[batch * ratio + slot..batch * ratio + slot + 1]
        .iter_mut()
        .zip(gate.iter())
    {
        *target = *value;
    }
}

fn pool_group(
    kv_state: &[f32],
    score_state: &[f32],
    batch: usize,
    ratio: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut pooled = Vec::with_capacity(head_dim);
    let scores = (0..ratio)
        .map(|slot| score_state[batch * ratio + slot])
        .collect::<Vec<_>>();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = scores.iter().map(|score| (score - max).exp()).sum();
    for d in 0..head_dim {
        let mut value = 0.0;
        for slot in 0..ratio {
            let weight = (scores[slot] - max).exp() / total;
            value += weight * kv_state[(batch * ratio + slot) * head_dim + d];
        }
        pooled.push(value);
    }
    pooled
}

/// The rotary helper the attention paths apply to the RoPE tail channels.
pub fn rotate_tail(x: &Tensor, freqs_cis: &Tensor, inverse: bool) -> Result<Tensor> {
    apply_rotary_emb(x, freqs_cis, inverse)
}

/// The in-place quantization round-trips the reference applies to the KV
/// caches before they are written.
pub fn quantize_window_kv(kv: &Tensor) -> Result<Tensor> {
    fp8_act_round_trip(kv, 32)
}

pub fn quantize_compressed_kv(latent: &Tensor) -> Result<Tensor> {
    fp4_act_round_trip(latent, 16, DType::F8E4M3)
}

pub fn quantize_indexer(vector: &Tensor) -> Result<Tensor> {
    fp4_act_round_trip(vector, 32, DType::F8E8M0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn sparse_attention_matches_a_dense_reference_with_masking() {
        let device = Device::Cpu;
        // One head, dim 2: attention over two rows plus the sink.
        let q = Tensor::from_vec(vec![1.0f32, 0.0], (1, 1, 1, 2), &device).unwrap();
        let kv =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0], (1, 3, 2), &device).unwrap();
        let sink = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let indexes = Tensor::from_vec(vec![0i64, 2, -1], (1, 1, 3), &device).unwrap();
        let output = sparse_attn(&q, &kv, &sink, &indexes, 0.5f64.sqrt()).unwrap();
        // Dense reference over rows {0, 2} and the unscaled sink logit 0.5.
        let scores = [
            1.0f32 * 0.5f64.sqrt() as f32,
            2.0 * 0.5f64.sqrt() as f32,
            0.5,
        ];
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores.map(|score| (score - max).exp());
        let total = weights.iter().sum::<f32>();
        let expected_row0 = weights[0] / total;
        let expected_row2 = weights[1] / total;
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected0 = expected_row0 * 1.0 + expected_row2 * 2.0;
        let expected1 = expected_row0 * 0.0 + expected_row2 * 2.0;
        assert!(
            (values[0] - expected0).abs() < 1e-5,
            "{} vs {}",
            values[0],
            expected0
        );
        assert!(
            (values[1] - expected1).abs() < 1e-5,
            "{} vs {}",
            values[1],
            expected1
        );
    }

    #[test]
    fn candidate_blocks_pin_the_newest_block_and_drop_unreachable_ones() {
        let device = Device::Cpu;
        // 8 positions, block size 4; position 7 is the newest reachable one.
        let scores = Tensor::from_vec(
            vec![
                1.0f32,
                0.0,
                0.0,
                0.0, //
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            (1, 8),
            &device,
        )
        .unwrap();
        let mask = select_candidate_blocks(&scores, 8, 1, 4).unwrap();
        let values = mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        // The second block is pinned even though its scores are -inf.
        assert_eq!(values, [0, 0, 0, 0, 1, 1, 1, 1]);
        // With two blocks wanted, the high-scoring first block joins it.
        let mask = select_candidate_blocks(&scores, 8, 2, 4).unwrap();
        let values = mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(values, [1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn compressor_pools_groups_and_carries_the_tail() {
        let device = Device::Cpu;
        let head_dim = 2;
        let hidden = 3;
        let norm = Tensor::ones(head_dim, DType::F32, &device).unwrap();
        let wkv = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
            (head_dim, hidden),
            &device,
        )
        .unwrap();
        let wgate = Tensor::from_vec(
            vec![0.3f32, 0.2, 0.1, 0.1, 0.2, 0.3],
            (head_dim, hidden),
            &device,
        )
        .unwrap();
        let mut compressor = Compressor::new(2, head_dim, norm, wkv, Some(wgate), 1).unwrap();
        // Prefill five tokens: two complete groups plus a one-token tail.
        let x = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 4.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0,
            ],
            (1, 5, hidden),
            &device,
        )
        .unwrap();
        let latents = compressor.forward(&x, 0).unwrap().unwrap();
        assert_eq!(latents.dims(), [1, 2, head_dim]);
        // The token at position 5 fills slot 1 of group 2 and completes it.
        let step = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (1, 1, hidden), &device).unwrap();
        let latents = compressor.forward(&step, 5).unwrap().unwrap();
        assert_eq!(latents.dims(), [1, 1, head_dim]);
        // Position 6 opens group 3 and produces nothing.
        let step = Tensor::from_vec(vec![2.0f32, 2.0, 2.0], (1, 1, hidden), &device).unwrap();
        assert!(compressor.forward(&step, 6).unwrap().is_none());
    }
}
