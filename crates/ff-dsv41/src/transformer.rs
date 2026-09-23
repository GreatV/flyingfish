//! The CED transformer top level: embed, hyper-connection plumbing across the
//! block stack, the optional engram insertions, final collapse and head.
//!
//! Reference: `Transformer.forward` and `Block.forward` in
//! `inference/model.py`. Host-reference scalar implementation.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::block::{hc_mixes, hc_post, hc_pre};
use crate::engram::{NgramHashState, engram_forward};
use crate::math::rms_norm;
use crate::model::{
    AttentionCore, SharedAttentionRuntime, expand_hc, greedy_token_from_logits, identity_pre_mix,
};
use crate::moe::{Expert, Gate};

/// The routed plus shared FFN of one block.
pub struct FfnWeights {
    pub gate: Gate,
    pub experts: Vec<Expert>,
    pub shared: Expert,
}

/// The MoE forward: route each token to its top-k experts, evaluate them on
/// the selected rows, and add the shared expert every token goes through.
pub fn moe_forward(ffn: &FfnWeights, x: &Tensor, image_mask: Option<&[bool]>) -> Result<Tensor> {
    let dims = x.dims();
    ensure!(dims.len() == 3, "MoE input must be [batch, seq, hidden]");
    let [batch, seq, hidden] = [dims[0], dims[1], dims[2]];
    let tokens = batch * seq;
    let flat = x
        .to_dtype(DType::F32)?
        .reshape((tokens, hidden))?
        .contiguous()?;
    let x_values = flat
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read MoE input")?;
    let (weights, indices) = ffn.gate.forward(&flat, image_mask)?;
    let topk = ffn.gate.topk;
    ensure!(
        indices.len() == tokens * topk,
        "routing table does not cover {tokens} tokens"
    );
    let mut output = vec![0.0f32; tokens * hidden];
    for (expert_index, expert) in ffn.experts.iter().enumerate() {
        // Gather every row routed to this expert so its resident weights are
        // read once per layer instead of once per route.
        let mut routed_tokens = Vec::new();
        let mut routed_weights = Vec::new();
        for token in 0..tokens {
            for slot in 0..topk {
                if indices[token * topk + slot] as usize == expert_index {
                    routed_tokens.push(token);
                    routed_weights.push(weights[token * topk + slot]);
                }
            }
        }
        if routed_tokens.is_empty() {
            continue;
        }
        let rows = Tensor::from_vec(
            routed_tokens
                .iter()
                .flat_map(|token| x_values[token * hidden..(token + 1) * hidden].to_vec())
                .collect::<Vec<_>>(),
            (routed_weights.len(), hidden),
            x.device(),
        )?;
        let evaluated = expert.forward_weighted(&rows, &routed_weights)?;
        let values = evaluated
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (row, token) in routed_tokens.iter().enumerate() {
            for (slot, value) in output[token * hidden..(token + 1) * hidden]
                .iter_mut()
                .zip(values[row * hidden..(row + 1) * hidden].iter())
            {
                *slot += value;
            }
        }
    }
    let shared = ffn.shared.forward(&flat, 1.0)?;
    let shared = shared
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (slot, value) in output.iter_mut().zip(shared.iter()) {
        *slot += value;
    }
    Tensor::from_vec(output, (batch, seq, hidden), x.device()).map_err(anyhow::Error::from)
}

/// The engram row lookup: hash ids in, dequantized `[cols, head_dim]` rows
/// out. Content-addressed, so the loader provides an mmap-backed callback.
pub type EngramLookup = Box<dyn Fn(&[i64]) -> Vec<f32> + Send + Sync>;

/// The engram of one block: the hash tables are content-addressed, so the
/// row lookup is a callback the loader provides (mmap-backed in production).
pub struct EngramCore {
    pub layer_hash_index: usize,
    pub head_dim: usize,
    pub wkv: Tensor,
    pub q_weight: Tensor,
    pub k_weight: Tensor,
    pub lookup: EngramLookup,
}

/// The static weights of one block around its attention core.
pub struct BlockWeights {
    pub attention: AttentionCore,
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub hc_attn_fn: Tensor,
    pub hc_attn_scale: Tensor,
    pub hc_attn_base: Tensor,
    pub hc_ffn_fn: Tensor,
    pub hc_ffn_scale: Tensor,
    pub hc_ffn_base: Tensor,
    pub ffn: FfnWeights,
    pub engram: Option<EngramCore>,
    pub ratio: usize,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub uses_candidates: bool,
    pub candidate_source: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TransformerParams {
    pub heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub norm_eps: f64,
    pub vocab: usize,
    pub hidden: usize,
}

/// One block's forward: attention and FFN each wrapped in their own
/// hyper-connection expand/collapse, the coefficients of a sublayer consumed
/// by the next one. Returns the stream and the FFN's `pre` mix.
#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    block: &mut BlockWeights,
    x: &Tensor,
    start_pos: usize,
    pre_mix: &Tensor,
    runtime: &mut SharedAttentionRuntime,
    params: TransformerParams,
    image_mask: Option<&[bool]>,
) -> Result<(Tensor, Tensor)> {
    let attention = &mut block.attention;
    let attn_mixes = hc_mixes(
        x,
        &block.hc_attn_fn,
        &block.hc_attn_scale,
        &block.hc_attn_base,
        params.hc_mult,
        params.hc_sinkhorn_iters,
        params.hc_eps,
        params.norm_eps,
    )?;
    let mut collapsed = hc_pre(x, pre_mix)?;
    let normed = rms_norm(&collapsed, &block.attn_norm, params.norm_eps)?;
    let attended = attention.forward(
        &normed,
        start_pos,
        runtime,
        block.ratio,
        block.is_kv_source,
        block.is_index_source,
        block.uses_candidates,
        block.candidate_source,
        (params.head_dim as f64).recip().sqrt(),
        params.rope_head_dim,
        params.o_lora_rank,
        params.o_groups,
        params.heads,
        params.head_dim,
        params.hidden,
    )?;
    let mut stream = hc_post(&attended, x, &attn_mixes.post, &attn_mixes.comb)?;

    let ffn_mixes = hc_mixes(
        &stream,
        &block.hc_ffn_fn,
        &block.hc_ffn_scale,
        &block.hc_ffn_base,
        params.hc_mult,
        params.hc_sinkhorn_iters,
        params.hc_eps,
        params.norm_eps,
    )?;
    collapsed = hc_pre(&stream, &attn_mixes.pre)?;
    let normed = rms_norm(&collapsed, &block.ffn_norm, params.norm_eps)?;
    let fed = moe_forward(&block.ffn, &normed, image_mask)?;
    stream = hc_post(&fed, &stream, &ffn_mixes.post, &ffn_mixes.comb)?;
    Ok((stream, ffn_mixes.pre))
}

pub struct Transformer {
    pub params: TransformerParams,
    pub embed: Tensor,
    pub blocks: Vec<BlockWeights>,
    pub norm_weight: Tensor,
    pub head: Tensor,
    pub ngram: Option<NgramHashState>,
}

impl Transformer {
    /// One forward over `[batch, seq]` token ids. Returns the greedy token
    /// from the last position's logits and the logits themselves.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<(u32, Tensor)> {
        self.forward_with_capture(input_ids, start_pos, None, None)
    }

    /// The same forward, optionally snapshotting the residual stream after
    /// every block for parity capture.
    pub fn forward_with_capture(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        mut capture: Option<&mut Vec<Tensor>>,
        mut observer: Option<&mut dyn FnMut(usize)>,
    ) -> Result<(u32, Tensor)> {
        let dims = input_ids.dims();
        ensure!(dims.len() == 2, "input ids must be [batch, seq]");
        let [batch, seq, hidden, hc] = [dims[0], dims[1], self.params.hidden, self.params.hc_mult];
        ensure!(
            self.embed.dims() == [self.params.vocab, hidden],
            "embedding table must be [vocab, hidden]"
        );
        let device = input_ids.device();
        let ids = input_ids
            .flatten_all()?
            .to_vec1::<u32>()
            .context("read input ids")?;
        let (embed_guard, _) = crate::math::resident_f32(&self.embed)?;
        let table = crate::math::resident_f32_slice(&embed_guard)?;
        let mut embedded = Vec::with_capacity(ids.len() * hidden);
        for id in &ids {
            ensure!(*id < self.params.vocab as u32, "token id {id} out of range");
            embedded.extend_from_slice(&table[*id as usize * hidden..][..hidden]);
        }
        let embedded = Tensor::from_vec(embedded, (batch, seq, hidden), device)?;
        let mut stream = expand_hc(&embedded, hc)?;
        let mut pre_mix = identity_pre_mix(batch, seq, hc, device)?;
        let mut runtime = SharedAttentionRuntime::default();
        let engram_hashes = match &mut self.ngram {
            Some(state) => Some(state.forward(input_ids, start_pos, None)?),
            None => None,
        };
        for block in &mut self.blocks {
            if let (Some(core), Some(hashes)) = (&block.engram, &engram_hashes) {
                let cols = hashes.dims()[3];
                let layer = core.layer_hash_index;
                let flat = hashes
                    .narrow(2, layer, 1)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<i64>()?;
                let embedded_rows = (core.lookup)(&flat);
                let embedded_rows =
                    Tensor::from_vec(embedded_rows, (batch, seq, cols, core.head_dim), device)?;
                stream = engram_forward(
                    &stream,
                    &embedded_rows,
                    &core.wkv,
                    &core.q_weight,
                    &core.k_weight,
                    None,
                )?;
            }
            let (next_stream, next_pre) = block_forward(
                block,
                &stream,
                start_pos,
                &pre_mix,
                &mut runtime,
                self.params,
                None,
            )?;
            stream = next_stream;
            pre_mix = next_pre;
            if let Some(capture) = capture.as_deref_mut() {
                capture.push(stream.contiguous()?);
                if let Some(observer) = observer.as_deref_mut() {
                    observer(capture.len());
                }
            }
        }
        let collapsed = hc_pre(&stream, &pre_mix)?;
        let normed = rms_norm(&collapsed, &self.norm_weight, self.params.norm_eps)?;
        let last = normed.narrow(1, seq - 1, 1)?;
        let (head_guard, _) = crate::math::resident_f32(&self.head)?;
        let head = crate::math::resident_f32_slice(&head_guard)?;
        ensure!(
            head.len() == self.params.vocab * hidden,
            "head must be [vocab, hidden]"
        );
        let last_values = last.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut logits = vec![0.0f32; batch * self.params.vocab];
        for b in 0..batch {
            for word in 0..self.params.vocab {
                let mut sum = 0.0;
                for channel in 0..hidden {
                    sum += head[word * hidden + channel] * last_values[b * hidden + channel];
                }
                logits[b * self.params.vocab + word] = sum;
            }
        }
        let logits = Tensor::from_vec(logits, (batch, self.params.vocab), device)?;
        let token = greedy_token_from_logits(&logits)?;
        Ok((token, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::Compressor;
    use crate::math::precompute_freqs_cis;
    use crate::model::WindowRing;
    use crate::moe::Gate;
    use candle_core::Device;

    fn ones(rows: usize, columns: usize, device: &Device) -> Tensor {
        Tensor::from_vec(
            (0..rows * columns)
                .map(|index| ((index % 7) as f32 - 3.0) / 97.0)
                .collect(),
            (rows, columns),
            device,
        )
        .unwrap()
    }

    #[test]
    fn transformer_decodes_after_prefill() {
        let device = Device::Cpu;
        let heads = 2;
        let head_dim = 32;
        let hidden = 32;
        let rope = 8;
        let window = 4;
        let hc = 2;
        let mix = (2 + hc) * hc;
        let params = TransformerParams {
            heads,
            head_dim,
            rope_head_dim: rope,
            o_lora_rank: 16,
            o_groups: 2,
            hc_mult: hc,
            hc_sinkhorn_iters: 4,
            hc_eps: 1e-6,
            norm_eps: 1e-20,
            vocab: 32,
            hidden,
        };
        let expert = || Expert {
            w1: ones(8, hidden, &device),
            w2: ones(hidden, 8, &device),
            w3: ones(8, hidden, &device),
            swiglu_limit: 10.0,
        };
        let ffn = || FfnWeights {
            gate: Gate {
                weight: ones(2, hidden, &device),
                bias: Tensor::from_vec(vec![0.5f32, 0.0], (2,), &device).unwrap(),
                bias_vl: None,
                topk: 2,
                gate_temp: 1.0,
                norm_topk_prob: true,
                route_scale: 1.5,
            },
            experts: vec![expert(), expert()],
            shared: expert(),
        };
        let attention = |ratio: usize| AttentionCore {
            sink: Tensor::zeros(heads, DType::F32, &device).unwrap(),
            wq_a: ones(16, hidden, &device),
            q_norm: Tensor::ones(16, DType::F32, &device).unwrap(),
            wq_b: ones(heads * head_dim, 16, &device),
            wkv: ones(head_dim, hidden, &device),
            kv_norm: Tensor::ones(head_dim, DType::F32, &device).unwrap(),
            wo_a: ones(2 * 16, head_dim, &device),
            wo_b: ones(hidden, 2 * 16, &device),
            compressor: if ratio > 1 {
                Some(
                    Compressor::new(
                        ratio,
                        head_dim,
                        Tensor::ones(head_dim, DType::F32, &device).unwrap(),
                        ones(head_dim, hidden, &device),
                        Some(ones(head_dim, hidden, &device)),
                        1,
                        1e-20,
                    )
                    .unwrap(),
                )
            } else {
                None
            },
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            ring: WindowRing::new(1, window, head_dim),
            indexer_wq_b: if ratio > 1 {
                Some(ones(2 * 32, 16, &device))
            } else {
                None
            },
            indexer_wk: if ratio > 1 {
                Some(ones(32, head_dim, &device))
            } else {
                None
            },
            indexer_k_norm: if ratio > 1 {
                Some(Tensor::ones(32, DType::F32, &device).unwrap())
            } else {
                None
            },
            weights_proj: if ratio > 1 {
                Some(ones(2, hidden, &device))
            } else {
                None
            },
            freqs: precompute_freqs_cis(rope, 32, 0, 1600.0, 1.0, 32, 1, &device).unwrap(),
            index_head_dim: 32,
            index_heads: 2,
            index_topk: 2,
            candidate_topk_blocks: 4,
            candidate_block_size: 4,
            norm_eps: 1e-20,
        };
        let block = |ratio: usize, kv: bool, index: bool| BlockWeights {
            attention: attention(ratio),
            attn_norm: Tensor::ones(hidden, DType::F32, &device).unwrap(),
            ffn_norm: Tensor::ones(hidden, DType::F32, &device).unwrap(),
            hc_attn_fn: ones(mix, hc * hidden, &device),
            hc_attn_scale: Tensor::from_vec(vec![0.3f32, 0.4, 0.2], (3,), &device).unwrap(),
            hc_attn_base: Tensor::zeros(mix, DType::F32, &device).unwrap(),
            hc_ffn_fn: ones(mix, hc * hidden, &device),
            hc_ffn_scale: Tensor::from_vec(vec![0.3f32, 0.4, 0.2], (3,), &device).unwrap(),
            hc_ffn_base: Tensor::zeros(mix, DType::F32, &device).unwrap(),
            ffn: ffn(),
            engram: None,
            ratio,
            is_kv_source: kv,
            is_index_source: index,
            uses_candidates: false,
            candidate_source: false,
        };
        let mut transformer = Transformer {
            params,
            embed: ones(params.vocab, hidden, &device),
            blocks: vec![block(0, false, false), block(2, true, true)],
            norm_weight: Tensor::ones(hidden, DType::F32, &device).unwrap(),
            head: ones(params.vocab, hidden, &device),
            ngram: None,
        };

        let prefill = Tensor::from_vec(vec![1u32, 5, 9, 13], (1, 4), &device).unwrap();
        let (token, logits) = transformer.forward(&prefill, 0).unwrap();
        assert_eq!(logits.dims(), [1, params.vocab]);
        let values = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((token as usize) < params.vocab);

        let mut decoded = token;
        for step in 4..7 {
            let step_input = Tensor::from_vec(vec![decoded], (1, 1), &device).unwrap();
            let (next, logits) = transformer.forward(&step_input, step).unwrap();
            assert_eq!(logits.dims(), [1, params.vocab]);
            let values = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                values.iter().all(|value| value.is_finite()),
                "non-finite logits at step {step}"
            );
            decoded = next;
        }
        assert!((decoded as usize) < params.vocab);
    }
}
