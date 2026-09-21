//! The CED transformer: a single 40-layer stack whose attention layers share
//! compressed KV, index keys, top-k picks and candidate masks through one
//! runtime carrier instead of a true encoder/decoder split.
//!
//! Reference: `Transformer`, `Attention`, `Indexer`, `SharedAttentionRuntime`
//! in `inference/model.py`. Host-reference scalar implementation.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::attention::{
    Compressor, quantize_compressed_kv, quantize_indexer, quantize_window_kv,
    select_candidate_blocks, sparse_attn,
};
use crate::config::TextConfig;
use crate::math::precompute_freqs_cis;

/// The four slots attention layers hand down the stack. Sources write before
/// their consumers read, so one slot each is enough.
#[derive(Default)]
pub struct SharedAttentionRuntime {
    pub compress_kv: Option<Tensor>,
    pub index_k: Option<Tensor>,
    pub topk_idxs: Option<Tensor>,
    pub candidates: Option<Tensor>,
}

/// The sliding-window KV ring: prefill seeds it with the trailing window of
/// the chunk, decode writes one slot per step and attends over the whole
/// ring, oldest first.
pub struct WindowRing {
    pub cache: Tensor,
    pub window: usize,
    batch: usize,
}

impl WindowRing {
    pub fn new(
        batch: usize,
        window: usize,
        head_dim: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Ok(Self {
            cache: Tensor::zeros((batch, window, head_dim), DType::F32, device)?,
            window,
            batch,
        })
    }

    /// Seed or step the ring; returns the KV rows this step attends over.
    pub fn write(&mut self, kv: &Tensor, start_pos: usize) -> Result<Tensor> {
        let dims = kv.dims();
        ensure!(dims.len() == 3, "window kv must be [batch, seq, head_dim]");
        let seqlen = dims[1];
        let mut values = self.cache.flatten_all()?.to_vec1::<f32>()?;
        let incoming = kv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let head_dim = dims[2];
        let (batch, window) = (self.batch, self.window);
        if start_pos == 0 {
            if seqlen <= window {
                values[..incoming.len()].copy_from_slice(&incoming);
            } else {
                let cutoff = seqlen % window;
                let tail = &incoming[(seqlen - window) * head_dim..];
                values[cutoff * head_dim..].copy_from_slice(&tail[..(window - cutoff) * head_dim]);
                values[..cutoff * head_dim].copy_from_slice(&tail[(window - cutoff) * head_dim..]);
            }
            self.cache = Tensor::from_vec(values, (batch, window, head_dim), kv.device())
                .map_err(anyhow::Error::from)?;
            Ok(kv.clone())
        } else {
            ensure!(seqlen == 1, "decode writes one token at a time");
            let slot = start_pos % window;
            for b in 0..batch {
                let base = (b * window + slot) * head_dim;
                values[base..base + head_dim]
                    .copy_from_slice(&incoming[b * head_dim..(b + 1) * head_dim]);
            }
            // The returned view is oldest-first; the caller's indices address
            // this rank order, matching the reference's ring layout.
            let mut ordered = Vec::with_capacity(batch * window * head_dim);
            for b in 0..batch {
                let oldest = slot + 1;
                for position in oldest..window {
                    ordered.extend_from_slice(
                        &values[(b * window + position) * head_dim..][..head_dim],
                    );
                }
                for position in 0..oldest {
                    ordered.extend_from_slice(
                        &values[(b * window + position) * head_dim..][..head_dim],
                    );
                }
            }
            self.cache = Tensor::from_vec(values, (batch, window, head_dim), kv.device())
                .map_err(anyhow::Error::from)?;
            Ok(
                Tensor::from_vec(ordered, (batch, window, head_dim), kv.device())
                    .map_err(anyhow::Error::from)?,
            )
        }
    }
}

/// Which sliding-window slots each query attends to; `-1` marks an empty slot.
pub fn window_topk_idxs(window: usize, batch: usize, seqlen: usize, start_pos: usize) -> Tensor {
    let mut idxs = Vec::with_capacity(batch * seqlen * window);
    if start_pos == 0 {
        for _ in 0..batch {
            for end in 0..seqlen {
                for slot in 0..window {
                    let index = end as isize - window as isize + 1 + slot as isize;
                    idxs.push(if index >= 0 { index as i64 } else { -1 });
                }
            }
        }
        Tensor::from_vec(
            idxs,
            (batch, seqlen, window.min(seqlen).max(window)),
            &candle_core::Device::Cpu,
        )
        .unwrap()
    } else {
        // Rank r in the rotated view is the r-th oldest row; the physical
        // slot it came from is (oldest + r) mod window, valid once the
        // sequence has reached that slot.
        let oldest = start_pos % window + 1;
        for _ in 0..batch {
            for _ in 0..seqlen {
                for rank in 0..window {
                    let position = (oldest + rank) % window;
                    idxs.push(if position <= start_pos {
                        rank as i64
                    } else {
                        -1
                    });
                }
            }
        }
        Tensor::from_vec(idxs, (batch, seqlen, window), &candle_core::Device::Cpu).unwrap()
    }
}

/// One attention layer's weights, materialized on the host.
pub struct AttentionCore {
    pub sink: Tensor,
    pub wq_a: Tensor,
    pub q_norm: Tensor,
    pub wq_b: Tensor,
    pub wkv: Tensor,
    pub kv_norm: Tensor,
    pub wo_a: Tensor,
    pub wo_b: Tensor,
    pub compressor: Option<Compressor>,
    pub compress_cache: Vec<f32>,
    pub index_k_cache: Vec<f32>,
    pub ring: WindowRing,
    pub indexer_wq_b: Option<Tensor>,
    pub indexer_wk: Option<Tensor>,
    pub indexer_k_norm: Option<Tensor>,
    pub weights_proj: Option<Tensor>,
    pub freqs: Tensor,
    pub index_head_dim: usize,
    pub index_heads: usize,
    pub index_topk: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub norm_eps: f64,
}

fn linear_rows(weight: &Tensor, input: &[f32], out: usize, inn: usize) -> Result<Vec<f32>> {
    ensure!(
        input.len() == inn,
        "activation of {} does not match the {inn}-wide projection",
        input.len()
    );
    let weights = weight
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read projection weight")?;
    Ok(linear_rows_from_slice(&weights, input, out, inn))
}

fn linear_rows_from_slice(weights: &[f32], input: &[f32], out: usize, inn: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out];
    for (row, slot) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for column in 0..inn {
            sum += weights[row * inn + column] * input[column];
        }
        *slot = sum;
    }
    output
}

/// The indexer scores compressed positions and returns the top-k picks each
/// query attends to, in position order, offset past the window rows.
#[allow(clippy::too_many_arguments)]
pub fn indexer_forward(
    x: &Tensor,
    qr: &Tensor,
    latent: Option<&Tensor>,
    start_pos: usize,
    offset: usize,
    wq_b: &Tensor,
    weights_proj: &Tensor,
    index_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    compress_ratio: usize,
    index_k: &Tensor,
    candidates: Option<&Tensor>,
    freqs: &Tensor,
    rope_head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let dims = x.dims();
    ensure!(dims.len() == 3, "indexer input must be [batch, seq, dim]");
    let [batch, seqlen, hidden] = [dims[0], dims[1], dims[2]];
    let compress_len = (start_pos + seqlen) / compress_ratio;
    let x_values = x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read indexer input")?;
    let qr_values = qr
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read indexer query base")?;
    let qlora = qr_values.len() / (batch * seqlen);
    let keys = index_k
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read index keys")?;
    let k_positions = keys.len() / (batch * index_head_dim);
    let mut queries = Vec::with_capacity(batch * seqlen * index_heads * index_head_dim);
    for token in 0..batch * seqlen {
        let mut projected = linear_rows(
            wq_b,
            &qr_values[token * qlora..(token + 1) * qlora],
            index_heads * index_head_dim,
            qlora,
        )?;
        // Queries rotate at their own positions before quantization, sharing
        // the layer's rotary table (model.py:546-547).
        let position = start_pos + token % seqlen;
        apply_rotary_slice(
            &mut projected,
            1,
            index_heads,
            index_head_dim,
            freqs,
            position,
            rope_head_dim,
            false,
        );
        queries.extend(projected);
    }
    let query_tensor = quantize_indexer(
        &Tensor::from_vec(
            queries.clone(),
            (batch * seqlen, index_heads * index_head_dim),
            x.device(),
        )
        .map_err(anyhow::Error::from)?,
    )?;
    queries = query_tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read quantized indexer queries")?;
    let mut weight_values = Vec::with_capacity(batch * seqlen * index_heads);
    {
        let projection = weights_proj
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read weights_proj")?;
        for token in 0..batch * seqlen {
            for head in 0..index_heads {
                let mut sum = 0.0;
                for column in 0..hidden {
                    sum += projection[head * hidden + column] * x_values[token * hidden + column];
                }
                weight_values.push(sum);
            }
        }
    }
    let scale = (index_head_dim as f64).recip().sqrt() * (index_heads as f64).recip().sqrt();
    let mut scores = vec![f32::NEG_INFINITY; batch * seqlen * k_positions];
    for token in 0..batch * seqlen {
        for position in 0..k_positions {
            let mut total = 0.0f32;
            for head in 0..index_heads {
                let mut dot = 0.0f32;
                for d in 0..index_head_dim {
                    dot += queries[(token * index_heads + head) * index_head_dim + d]
                        * keys[((token / seqlen) * k_positions + position) * index_head_dim + d];
                }
                total += dot.max(0.0) * weight_values[token * index_heads + head] * scale as f32;
            }
            scores[token * k_positions + position] = total;
        }
    }
    // A compressed position becomes visible once the query has passed its
    // last token.
    for token in 0..batch * seqlen {
        let visible = if start_pos == 0 {
            (token % seqlen + 1) / compress_ratio
        } else {
            compress_len
        };
        for position in 0..k_positions {
            if position >= visible {
                scores[token * k_positions + position] = f32::NEG_INFINITY;
            }
        }
    }
    let scores_tensor = Tensor::from_vec(scores.clone(), (batch, seqlen, k_positions), x.device())
        .map_err(anyhow::Error::from)?;
    // `candidates` is the bool mask the source layer published; consumers
    // mask by it without recomputing.
    if let Some(mask) = candidates {
        let mask_values = mask.flatten_all()?.to_vec1::<u8>()?;
        for (slot, keep) in scores.iter_mut().zip(mask_values.iter()) {
            if *keep == 0 {
                *slot = f32::NEG_INFINITY;
            }
        }
    }
    let topk = index_topk.min(compress_len);
    let mut idxs = vec![-1i64; batch * seqlen * topk.max(1)];
    for token in 0..batch * seqlen {
        let visible = if start_pos == 0 {
            (token % seqlen + 1) / compress_ratio
        } else {
            compress_len
        };
        let mut order = (0..k_positions).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            scores[token * k_positions + *b]
                .partial_cmp(&scores[token * k_positions + *a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut chosen: Vec<usize> = Vec::new();
        for position in order {
            if chosen.len() == topk {
                break;
            }
            if position < visible {
                chosen.push(position);
            }
        }
        chosen.sort_unstable();
        for (slot, position) in idxs[token * topk.max(1)..(token + 1) * topk.max(1)]
            .iter_mut()
            .zip(chosen.iter())
        {
            *slot = *position as i64 + offset as i64;
        }
    }
    let idx_tensor = Tensor::from_vec(idxs, (batch, seqlen, topk.max(1)), x.device())
        .map_err(anyhow::Error::from)?;
    let _ = latent;
    Ok((idx_tensor, scores_tensor))
}

/// One full attention layer forward: window KV ring plus, when this layer
/// compresses, the shared compressed KV and indexer picks, concatenated into
/// one sparse-attention call. Reference: `Attention.forward`.
impl AttentionCore {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        x: &Tensor,
        start_pos: usize,
        runtime: &mut SharedAttentionRuntime,
        ratio: usize,
        is_kv_source: bool,
        is_index_source: bool,
        uses_candidates: bool,
        candidate_source: bool,
        softmax_scale: f64,
        rope_head_dim: usize,
        o_lora_rank: usize,
        o_groups: usize,
        heads: usize,
        head_dim: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        let dims = x.dims();
        ensure!(dims.len() == 3, "attention input must be [batch, seq, dim]");
        let [batch, seqlen, _] = [dims[0], dims[1], dims[2]];
        let device = x.device();
        let x_values = x
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("read attention input")?;

        let qlora = self.wq_a.dims()[0];
        let wq_a = self
            .wq_a
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut qr = Vec::with_capacity(batch * seqlen * qlora);
        for token in 0..batch * seqlen {
            let projected = linear_rows_from_slice(
                &wq_a,
                &x_values[token * hidden..(token + 1) * hidden],
                qlora,
                hidden,
            );
            qr.extend(rms_norm_slice(&projected, &self.q_norm, self.norm_eps));
        }
        let wq_b = self
            .wq_b
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut q = Vec::with_capacity(batch * seqlen * heads * head_dim);
        for token in 0..batch * seqlen {
            let projected = linear_rows_from_slice(
                &wq_b,
                &qr[token * qlora..(token + 1) * qlora],
                heads * head_dim,
                qlora,
            );
            q.extend(projected);
        }
        apply_rotary_slice(
            &mut q,
            seqlen,
            heads,
            head_dim,
            &self.freqs,
            start_pos,
            rope_head_dim,
            false,
        );

        // Window KV: shared latent, normalized, rotated, FP8-round-tripped,
        // written into the ring.
        let wkv = self
            .wkv
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut window = Vec::with_capacity(batch * seqlen * head_dim);
        for token in 0..batch * seqlen {
            let projected = linear_rows_from_slice(
                &wkv,
                &x_values[token * hidden..(token + 1) * hidden],
                head_dim,
                hidden,
            );
            window.extend(rms_norm_slice(&projected, &self.kv_norm, self.norm_eps));
        }
        apply_rotary_slice(
            &mut window,
            seqlen,
            1,
            head_dim,
            &self.freqs,
            start_pos,
            rope_head_dim,
            false,
        );
        let window_tensor = quantize_window_kv(&Tensor::from_vec(
            window.clone(),
            (batch, seqlen, head_dim),
            device,
        )?)?;
        let window_rows = window_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let window_tensor = Tensor::from_vec(window_rows, (batch, seqlen, head_dim), device)
            .map_err(anyhow::Error::from)?;
        let attended_kv = self.ring.write(&window_tensor, start_pos)?;
        let mut kv_values = attended_kv
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .to_vec();
        let window_len = kv_values.len() / (batch * head_dim);
        let ring_window = if start_pos == 0 {
            self.ring.window.min(seqlen).max(1)
        } else {
            self.ring.window
        };
        let window_picks = window_topk_idxs(ring_window, batch, seqlen, start_pos)
            .flatten_all()?
            .to_vec1::<i64>()?;
        let window_width = window_picks.len() / (batch * seqlen);
        let mut compress_picks: Vec<i64> = Vec::new();
        let mut topk: Vec<i64> = window_picks;

        let mut latents = None;
        if ratio > 0 {
            let latent = match (is_kv_source, &mut self.compressor) {
                (true, Some(compressor)) => compressor.forward(x, start_pos)?,
                _ => None,
            };
            if let Some(latent) = latent {
                latents = Some(
                    latent
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .to_vec(),
                );
            }
            let compress_len = (start_pos + seqlen).checked_div(ratio).unwrap_or(0);
            if !self.index_k_cache.is_empty() {
                let key_rows = self.index_k_cache.len() / self.index_head_dim;
                runtime.index_k = Some(Tensor::from_vec(
                    self.index_k_cache[..compress_len.min(key_rows) * self.index_head_dim].to_vec(),
                    (batch, compress_len.min(key_rows), self.index_head_dim),
                    device,
                )?);
            }
            if let Some(latent) = &latents {
                // Index-key owners publish their keys before the latent is
                // rotated for the KV cache; the indexer needs the unrotated
                // form, then applies its own rotation and FP4 round-trip.
                if let (true, true, Some(wk), Some(k_norm)) = (
                    is_kv_source,
                    is_index_source,
                    self.indexer_wk.as_ref(),
                    self.indexer_k_norm.as_ref(),
                ) {
                    let produced = latent.len() / (batch * head_dim);
                    let mut keys = Vec::with_capacity(produced * self.index_head_dim);
                    for group in 0..produced {
                        let projected = linear_rows(
                            wk,
                            &latent[group * head_dim..(group + 1) * head_dim],
                            self.index_head_dim,
                            head_dim,
                        )?;
                        keys.extend(rms_norm_slice(&projected, k_norm, self.norm_eps));
                    }
                    for group in 0..produced {
                        apply_rotary_slice(
                            &mut keys
                                [group * self.index_head_dim..(group + 1) * self.index_head_dim],
                            1,
                            1,
                            self.index_head_dim,
                            &self.freqs.clone(),
                            if start_pos == 0 {
                                group * ratio
                            } else {
                                start_pos + 1 - ratio
                            },
                            rope_head_dim,
                            false,
                        );
                    }
                    let quantized = quantize_indexer(
                        &Tensor::from_vec(keys, (batch, produced, self.index_head_dim), device)
                            .map_err(anyhow::Error::from)?,
                    )?;
                    self.index_k_cache.extend(
                        quantized
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                    let key_rows = self.index_k_cache.len() / self.index_head_dim;
                    runtime.index_k = Some(Tensor::from_vec(
                        self.index_k_cache[..compress_len.min(key_rows) * self.index_head_dim]
                            .to_vec(),
                        (batch, compress_len.min(key_rows), self.index_head_dim),
                        device,
                    )?);
                }
            }
            if is_index_source {
                let qr_tensor = Tensor::from_vec(qr.clone(), (batch, seqlen, qlora), device)
                    .map_err(anyhow::Error::from)?;
                let (idxs, scores) = indexer_forward(
                    x,
                    &qr_tensor,
                    None,
                    start_pos,
                    window_len,
                    self.indexer_wq_b
                        .as_ref()
                        .context("index source needs indexer wq_b")?,
                    self.weights_proj
                        .as_ref()
                        .context("index source needs weights_proj")?,
                    self.index_heads,
                    self.index_head_dim,
                    self.index_topk,
                    ratio,
                    runtime
                        .index_k
                        .as_ref()
                        .context("index keys missing before first index source")?,
                    if uses_candidates {
                        runtime.candidates.as_ref()
                    } else {
                        None
                    },
                    &self.freqs,
                    rope_head_dim,
                )?;
                if candidate_source {
                    runtime.candidates = Some(select_candidate_blocks(
                        &scores,
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                    )?);
                }
                runtime.topk_idxs = Some(idxs.clone());
                compress_picks = idxs.flatten_all()?.to_vec1::<i64>()?;
            } else {
                let shared = runtime
                    .topk_idxs
                    .as_ref()
                    .context("non-index layer read picks before any source wrote them")?;
                compress_picks = shared.flatten_all()?.to_vec1::<i64>()?;
            }
            if let Some(latent) = latents {
                // The latent stands for the first token of its group; decode
                // yields one group at position start_pos + 1 - ratio.
                let produced = latent.len() / (batch * head_dim);
                let mut rotated = latent;
                for group in 0..produced {
                    apply_rotary_slice(
                        &mut rotated[group * head_dim..(group + 1) * head_dim],
                        1,
                        1,
                        head_dim,
                        &self.freqs,
                        if start_pos == 0 {
                            group * ratio
                        } else {
                            start_pos + 1 - ratio
                        },
                        rope_head_dim,
                        false,
                    );
                }
                let quantized = quantize_compressed_kv(&Tensor::from_vec(
                    rotated,
                    (batch, produced, head_dim),
                    device,
                )?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
                self.compress_cache.extend(quantized);
            }
            // Consumers append the source's published rows into their own
            // attention input so the compressed picks index real rows;
            // sources publish from their own cache instead of reading.
            let shared_rows: Vec<f32> = if is_kv_source {
                let cache_rows = self.compress_cache.len() / head_dim;
                let rows = cache_rows.min(compress_len * batch);
                runtime.compress_kv = Some(Tensor::from_vec(
                    self.compress_cache[..rows * head_dim].to_vec(),
                    (batch, rows / batch.max(1), head_dim),
                    device,
                )?);
                self.compress_cache[..rows * head_dim].to_vec()
            } else {
                runtime
                    .compress_kv
                    .as_ref()
                    .context("non-source layer read compressed KV before any source wrote it")?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            };
            kv_values.extend(shared_rows);
        }

        // sparse_attn reads one pick row per query: interleave the window and
        // compressed picks per (batch, query) instead of concatenating blocks.
        if !compress_picks.is_empty() {
            let compress_width = compress_picks.len() / (batch * seqlen);
            let mut interleaved =
                Vec::with_capacity(batch * seqlen * (window_width + compress_width));
            for row in 0..batch * seqlen {
                interleaved.extend_from_slice(&topk[row * window_width..(row + 1) * window_width]);
                interleaved.extend_from_slice(
                    &compress_picks[row * compress_width..(row + 1) * compress_width],
                );
            }
            topk = interleaved;
        }

        let total_rows = kv_values.len() / (batch * head_dim);
        let kv_tensor = Tensor::from_vec(kv_values, (batch, total_rows, head_dim), device)
            .map_err(anyhow::Error::from)?;
        let q_tensor = Tensor::from_vec(q, (batch, seqlen, heads, head_dim), device)
            .map_err(anyhow::Error::from)?;
        let picks = total_rows - window_len + seqlen.max(1);
        let topk_width = topk.len() / (batch * seqlen);
        let _ = picks;
        let idx_tensor = Tensor::from_vec(topk, (batch, seqlen, topk_width.max(1)), device)
            .map_err(anyhow::Error::from)?;
        let mut output = sparse_attn(
            &q_tensor,
            &kv_tensor,
            &self.sink,
            &idx_tensor,
            softmax_scale,
        )?;
        let mut o = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .to_vec();
        apply_rotary_slice(
            &mut o,
            seqlen,
            heads,
            head_dim,
            &self.freqs,
            start_pos,
            rope_head_dim,
            true,
        );
        // wo_a is block-diagonal over groups; each group projects its own heads.
        let group_heads = heads / o_groups;
        let group_width = group_heads * head_dim;
        let mut collapsed = Vec::with_capacity(batch * seqlen * o_groups * o_lora_rank);
        let wo_a = self
            .wo_a
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .to_vec();
        let _ = &wo_a;
        for token in 0..batch * seqlen {
            for group in 0..o_groups {
                for rank in 0..o_lora_rank {
                    let mut sum = 0.0f32;
                    for channel in 0..group_width {
                        let head = group * group_heads + channel / head_dim;
                        let inner = channel % head_dim;
                        sum += o[(token * heads + head) * head_dim + inner]
                            * wo_a[(group * o_lora_rank + rank) * group_width + channel];
                    }
                    collapsed.push(sum);
                }
            }
        }
        let mut projected = Vec::with_capacity(batch * seqlen * hidden);
        let wo_b = self
            .wo_b
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for token in 0..batch * seqlen {
            projected.extend(linear_rows_from_slice(
                &wo_b,
                &collapsed[token * o_groups * o_lora_rank..(token + 1) * o_groups * o_lora_rank],
                hidden,
                o_groups * o_lora_rank,
            ));
        }
        output = Tensor::from_vec(projected, (batch, seqlen, hidden), device)
            .map_err(anyhow::Error::from)?;
        Ok(output)
    }
}

fn rms_norm_slice(values: &[f32], weight: &Tensor, eps: f64) -> Vec<f32> {
    let weight = weight
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let width = weight.len();
    let squares = values.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let rstd = 1.0 / (squares + eps as f32).sqrt();
    values
        .iter()
        .zip(weight.iter())
        .map(|(value, scale)| value * rstd * scale)
        .collect()
}

/// In-place rotary over the trailing `rope_dim` channels of each head, using
/// rows `[offset, offset + seqlen)` of `freqs`.
#[allow(clippy::too_many_arguments)]
fn apply_rotary_slice(
    values: &mut [f32],
    seqlen: usize,
    heads: usize,
    head_dim: usize,
    freqs: &Tensor,
    offset: usize,
    rope_dim: usize,
    inverse: bool,
) {
    assert!(
        head_dim >= rope_dim,
        "rotary span {rope_dim} exceeds the {head_dim}-wide channel"
    );
    let rows = values.len() / (seqlen * heads * head_dim);
    let angles = freqs
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let half = rope_dim / 2;
    for row in 0..rows {
        for token in 0..seqlen {
            let angle_base = (offset + token) * half * 2;
            for head in 0..heads {
                let base = ((row * seqlen + token) * heads + head) * head_dim;
                for pair in 0..half {
                    let even = values[base + head_dim - rope_dim + 2 * pair];
                    let odd = values[base + head_dim - rope_dim + 2 * pair + 1];
                    let cos = angles[angle_base + 2 * pair];
                    let raw_sin = angles[angle_base + 2 * pair + 1];
                    let sin = if inverse { -raw_sin } else { raw_sin };
                    values[base + head_dim - rope_dim + 2 * pair] = even * cos - odd * sin;
                    values[base + head_dim - rope_dim + 2 * pair + 1] = even * sin + odd * cos;
                }
            }
        }
    }
}

/// Greedy pick from the last position's logits.
pub fn greedy_token_from_logits(logits: &Tensor) -> Result<u32> {
    let dims = logits.dims();
    ensure!(dims.last().is_some(), "logits must have a vocab axis");
    let values = logits
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read logits")?;
    let vocab = *dims.last().expect("checked above");
    let tail = &values[values.len() - vocab..];
    ensure!(
        tail.iter().all(|value| value.is_finite()),
        "model produced non-finite logits"
    );
    let best = tail
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
        .context("empty logits")?;
    Ok(best.0 as u32)
}

/// Prefill/decode helper shared by the model: expand the embedding into
/// `hc_mult` parallel copies.
pub fn expand_hc(embedding: &Tensor, hc_mult: usize) -> Result<Tensor> {
    let dims = embedding.dims();
    ensure!(dims.len() == 3, "embedding must be [batch, seq, dim]");
    let values = embedding
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read embedding")?;
    let [batch, seq, hidden] = [dims[0], dims[1], dims[2]];
    let mut expanded = Vec::with_capacity(values.len() * hc_mult);
    for token in 0..batch * seq {
        for _ in 0..hc_mult {
            expanded.extend_from_slice(&values[token * hidden..(token + 1) * hidden]);
        }
    }
    Tensor::from_vec(expanded, (batch, seq, hc_mult, hidden), embedding.device())
        .map_err(anyhow::Error::from)
}

/// The initial one-hot hc mix.
pub fn identity_pre_mix(
    batch: usize,
    seq: usize,
    hc_mult: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut values = vec![0.0f32; batch * seq * hc_mult];
    for token in 0..batch * seq {
        values[token * hc_mult] = 1.0;
    }
    Tensor::from_vec(values, (batch, seq, hc_mult), device).map_err(anyhow::Error::from)
}

/// The one rotary table a layer uses for every path: window KV, compressed
/// KV at its group positions, and the indexer keys it publishes. Compressing
/// layers run at `compress_rope_theta` with YaRN extrapolation; pure
/// sliding-window layers use plain `rope_theta` without it.
pub fn layer_freqs(
    config: &TextConfig,
    rope_head_dim: usize,
    compress_ratio: u8,
    max_seq: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if compress_ratio == 0 {
        return precompute_freqs_cis(
            rope_head_dim,
            max_seq,
            0,
            config.rope_theta,
            1.0,
            config
                .rope_scaling
                .as_ref()
                .map_or(32, |scaling| scaling.beta_fast),
            config
                .rope_scaling
                .as_ref()
                .map_or(1, |scaling| scaling.beta_slow),
            device,
        );
    }
    let scaling = config
        .rope_scaling
        .as_ref()
        .context("compressing layers need rope_scaling for YaRN")?;
    precompute_freqs_cis(
        rope_head_dim,
        max_seq,
        scaling.original_max_position_embeddings,
        config.compress_rope_theta,
        scaling.factor,
        scaling.beta_fast,
        scaling.beta_slow,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn window_ring_seeds_prefill_and_rotates_decode() {
        let device = Device::Cpu;
        let mut ring = WindowRing::new(1, 4, 2, &device).unwrap();
        // Prefill six tokens: the ring keeps the last four.
        let prefill = Tensor::from_vec(
            (0..12).map(|index| index as f32).collect(),
            (1, 6, 2),
            &device,
        )
        .unwrap();
        let over = ring.write(&prefill, 0).unwrap();
        assert_eq!(over.dims(), [1, 6, 2]);
        // Decode at position 6 lands in slot 2; the ordered view starts with
        // slot 3 (the oldest).
        let step = Tensor::from_vec(vec![100.0f32, 101.0], (1, 1, 2), &device).unwrap();
        let ordered = ring.write(&step, 6).unwrap();
        assert_eq!(ordered.dims(), [1, 4, 2]);
        let values = ordered.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Slots after prefill hold [t4, t5, t2, t3]; position 6 overwrites
        // slot 2, so the oldest-first order is [t3, t4, t5, new].
        assert_eq!(values[0], 6.0);
        assert_eq!(values[6], 100.0);
        // Prefill top-k: every query sees its own causal window, the earliest
        // queries see nothing, the last sees the trailing four positions.
        let idxs = window_topk_idxs(4, 1, 6, 0);
        assert_eq!(idxs.dims(), [1, 6, 4]);
        let flat = idxs.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(flat[..4], [-1, -1, -1, 0]);
        assert_eq!(flat[5 * 4..6 * 4], [2, 3, 4, 5]);
    }

    #[test]
    fn indexer_queries_rotate_at_their_positions_before_scoring() {
        let device = Device::Cpu;
        let hidden = 32usize;
        let qlora = 16usize;
        let x = Tensor::from_vec(
            (0..2 * hidden)
                .map(|index| (index % 5) as f32 / 31.0)
                .collect(),
            (1, 2, hidden),
            &device,
        )
        .unwrap();
        let qr = Tensor::ones((1, 2, qlora), DType::F32, &device).unwrap();
        // One head, head_dim 32, one rope pair in the last two channels: the
        // projection pins that pair to (1, 0); the two keys disagree on the
        // sign of channel 30, so the winning pick flips exactly when the
        // angle flips. All non-rope channels stay zero.
        let wq_b = {
            let mut values = vec![0.0f32; 32 * qlora];
            values[30 * qlora] = 1.0;
            Tensor::from_vec(values, (32, qlora), &device).unwrap()
        };
        let weights_proj = Tensor::from_vec(vec![0.5f32; hidden], (1, hidden), &device).unwrap();
        let mut key_values = vec![0.0f32; 2 * 32];
        key_values[30] = 1.0;
        key_values[32 + 30] = -1.0;
        let index_k = Tensor::from_vec(key_values, (1, 2, 32), &device).unwrap();
        let freqs = |flip: bool| {
            let angle = if flip { std::f32::consts::PI } else { 0.0f32 };
            Tensor::from_vec(
                [angle.cos(), angle.sin()]
                    .iter()
                    .cycle()
                    .take(16)
                    .copied()
                    .collect::<Vec<f32>>(),
                (4, 1, 2),
                &device,
            )
            .unwrap()
        };
        let run = |flip: bool| {
            // Decode position: the causal visibility mask then allows every
            // key, so the pick reflects the score sign alone.
            let (idxs, _) = indexer_forward(
                &x,
                &qr,
                None,
                1,
                4,
                &wq_b,
                &weights_proj,
                1,
                32,
                1,
                1,
                &index_k,
                None,
                &freqs(flip),
                2,
            )
            .unwrap();
            idxs.flatten_all().unwrap().to_vec1::<i64>().unwrap()
        };
        let identity = run(false);
        let flipped = run(true);
        assert!(
            identity != flipped,
            "picks identical across flipped angles: {identity:?}"
        );
        assert_eq!(identity[0], 4, "identity angles pick key 0: {identity:?}");
        assert_eq!(flipped[0], 5, "flipped angles pick key 1: {flipped:?}");
    }

    #[test]
    fn candidate_consumers_mask_by_the_published_mask() {
        let device = Device::Cpu;
        let hidden = 32usize;
        let qlora = 16usize;
        let x = Tensor::from_vec(
            (0..4 * hidden)
                .map(|index| (index % 5) as f32 / 31.0)
                .collect(),
            (1, 4, hidden),
            &device,
        )
        .unwrap();
        let qr = Tensor::zeros((1, 4, qlora), DType::F32, &device).unwrap();
        let wq_b = {
            let mut values = vec![0.0f32; 2 * 32 * qlora];
            for row in 0..2 * 32 {
                values[row * qlora + (row % qlora)] = 0.1;
            }
            Tensor::from_vec(values, (2 * 32, qlora), &device).unwrap()
        };
        let weights_proj =
            Tensor::from_vec(vec![0.5f32; 2 * hidden], (2, hidden), &device).unwrap();
        // Position 1 scores higher than position 0; masking position 1 must
        // flip the top pick to position 0 — the mask steers selection, and a
        // consumer that recomputed its own candidates would ignore it.
        let index_k = Tensor::zeros((1, 2, 32), DType::F32, &device).unwrap();
        let freqs =
            crate::math::precompute_freqs_cis(8, 8, 0, 1600.0, 1.0, 32, 1, &device).unwrap();
        let open = Tensor::from_vec(vec![1u8; 4 * 2], (1, 4, 2), &device).unwrap();
        let mut masked = vec![1u8; 4 * 2];
        for row in masked.chunks_mut(2) {
            row[1] = 0;
        }
        let masked = Tensor::from_vec(masked, (1, 4, 2), &device).unwrap();
        let (open_picks, _) = indexer_forward(
            &x,
            &qr,
            None,
            0,
            4,
            &wq_b,
            &weights_proj,
            2,
            32,
            2,
            2,
            &index_k,
            Some(&open),
            &freqs,
            8,
        )
        .unwrap();
        let open_picks = open_picks.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        let (masked_picks, _) = indexer_forward(
            &x,
            &qr,
            None,
            0,
            4,
            &wq_b,
            &weights_proj,
            2,
            32,
            2,
            2,
            &index_k,
            Some(&masked),
            &freqs,
            8,
        )
        .unwrap();
        let masked_picks = masked_picks
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(open_picks.len(), 4 * 2);
        assert_eq!(masked_picks.len(), 4 * 2);
        assert!(
            open_picks.iter().any(|pick| *pick != masked_picks[0]),
            "mask did not steer selection: {open_picks:?} vs {masked_picks:?}"
        );
    }

    #[test]
    fn consumer_layers_attend_over_the_published_compressed_rows() {
        use crate::math::precompute_freqs_cis;
        let device = Device::Cpu;
        let heads = 2usize;
        let head_dim = 32usize;
        let hidden = 32usize;
        let qlora = 16usize;
        let ones = |rows: usize, columns: usize| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| ((index % 7) as f32 - 3.0) / 97.0)
                    .collect(),
                (rows, columns),
                &device,
            )
            .unwrap()
        };
        let build = || AttentionCore {
            sink: Tensor::zeros(heads, DType::F32, &device).unwrap(),
            wq_a: ones(qlora, hidden),
            q_norm: Tensor::ones(qlora, DType::F32, &device).unwrap(),
            wq_b: ones(heads * head_dim, qlora),
            wkv: ones(head_dim, hidden),
            kv_norm: Tensor::ones(head_dim, DType::F32, &device).unwrap(),
            wo_a: ones(2 * 16, head_dim),
            wo_b: ones(hidden, 2 * 16),
            compressor: Some(
                Compressor::new(
                    2,
                    head_dim,
                    Tensor::ones(head_dim, DType::F32, &device).unwrap(),
                    ones(head_dim, hidden),
                    Some(ones(head_dim, hidden)),
                    1,
                    1e-20,
                )
                .unwrap(),
            ),
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            ring: WindowRing::new(1, 4, head_dim, &device).unwrap(),
            indexer_wq_b: Some(ones(2 * 32, qlora)),
            indexer_wk: Some(ones(32, head_dim)),
            indexer_k_norm: Some(Tensor::ones(32, DType::F32, &device).unwrap()),
            weights_proj: Some(ones(2, hidden)),
            freqs: precompute_freqs_cis(8, 16, 0, 1600.0, 1.0, 32, 1, &device).unwrap(),
            index_head_dim: 32,
            index_heads: 2,
            index_topk: 2,
            candidate_topk_blocks: 4,
            candidate_block_size: 4,
            norm_eps: 1e-20,
        };
        let x = Tensor::from_vec(
            (0..2 * hidden)
                .map(|index| (index % 5) as f32 / 31.0)
                .collect(),
            (1, 2, hidden),
            &device,
        )
        .unwrap();
        let run = |shared: f32| {
            let mut source = build();
            let mut source_runtime = SharedAttentionRuntime::default();
            source
                .forward(
                    &x,
                    0,
                    &mut source_runtime,
                    2,
                    true,
                    true,
                    false,
                    false,
                    0.5f64.sqrt(),
                    8,
                    16,
                    2,
                    heads,
                    head_dim,
                    hidden,
                )
                .unwrap();
            let published = source_runtime.compress_kv.clone().unwrap();
            let published_dims = published.dims().to_vec();
            let published_count = published
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .len();
            let mut consumer = build();
            let mut runtime = SharedAttentionRuntime {
                compress_kv: Some(published),
                index_k: source_runtime.index_k.clone(),
                topk_idxs: source_runtime.topk_idxs.clone(),
                candidates: None,
            };
            // Overwrite every published row with a constant: the consumer's
            // attention output must move with it. A layer that ignored the
            // shared rows (or sized its pick row against an empty kv)
            // either crashes or stays identical.
            let shape = published_dims;
            let count = published_count;
            runtime.compress_kv =
                Some(Tensor::from_vec(vec![shared; count], shape, &device).unwrap());
            let output = consumer
                .forward(
                    &x,
                    0,
                    &mut runtime,
                    2,
                    false,
                    false,
                    false,
                    false,
                    0.5f64.sqrt(),
                    8,
                    16,
                    2,
                    heads,
                    head_dim,
                    hidden,
                )
                .unwrap();
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let quiet = run(0.0);
        let loud = run(9.0);
        assert_eq!(quiet.len(), 2 * hidden);
        assert!(
            quiet
                .iter()
                .zip(loud.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "consumer output ignored the shared compressed rows: {quiet:?} vs {loud:?}"
        );
    }

    #[test]
    fn window_and_compress_picks_interleave_per_query() {
        use crate::math::precompute_freqs_cis;
        let device = Device::Cpu;
        let heads = 2usize;
        let head_dim = 32usize;
        let hidden = 32usize;
        let qlora = 16usize;
        let ones = |rows: usize, columns: usize| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| ((index % 7) as f32 - 3.0) / 97.0)
                    .collect(),
                (rows, columns),
                &device,
            )
            .unwrap()
        };
        let build = || AttentionCore {
            sink: Tensor::zeros(heads, DType::F32, &device).unwrap(),
            wq_a: ones(qlora, hidden),
            q_norm: Tensor::ones(qlora, DType::F32, &device).unwrap(),
            wq_b: ones(heads * head_dim, qlora),
            wkv: ones(head_dim, hidden),
            kv_norm: Tensor::ones(head_dim, DType::F32, &device).unwrap(),
            wo_a: ones(2 * 16, head_dim),
            wo_b: ones(hidden, 2 * 16),
            compressor: Some(
                Compressor::new(
                    2,
                    head_dim,
                    Tensor::ones(head_dim, DType::F32, &device).unwrap(),
                    ones(head_dim, hidden),
                    Some(ones(head_dim, hidden)),
                    1,
                    1e-20,
                )
                .unwrap(),
            ),
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            ring: WindowRing::new(1, 4, head_dim, &device).unwrap(),
            indexer_wq_b: Some(ones(2 * 32, qlora)),
            indexer_wk: Some(ones(32, head_dim)),
            indexer_k_norm: Some(Tensor::ones(32, DType::F32, &device).unwrap()),
            weights_proj: Some(ones(2, hidden)),
            freqs: precompute_freqs_cis(8, 16, 0, 1600.0, 1.0, 32, 1, &device).unwrap(),
            index_head_dim: 32,
            index_heads: 2,
            index_topk: 2,
            candidate_topk_blocks: 4,
            candidate_block_size: 4,
            norm_eps: 1e-20,
        };
        let base = |scale: f32| {
            Tensor::from_vec(
                (0..4 * hidden)
                    .map(|index| if index >= 3 * hidden { scale } else { 0.0 })
                    .collect(),
                (1, 4, hidden),
                &device,
            )
            .unwrap()
        };
        // Two identical runs except token 3's magnitude: query 0's window is
        // only itself (-1 sentinels for the rest), so its output row must be
        // identical across the two runs once picks interleave per query.
        let mut quiet = build();
        let mut loud = build();
        let out_quiet = quiet
            .forward(
                &base(0.0),
                0,
                &mut SharedAttentionRuntime::default(),
                2,
                true,
                true,
                false,
                false,
                0.5f64.sqrt(),
                8,
                16,
                2,
                heads,
                head_dim,
                hidden,
            )
            .unwrap();
        let out_loud = loud
            .forward(
                &base(5.0),
                0,
                &mut SharedAttentionRuntime::default(),
                2,
                true,
                true,
                false,
                false,
                0.5f64.sqrt(),
                8,
                16,
                2,
                heads,
                head_dim,
                hidden,
            )
            .unwrap();
        let quiet = out_quiet.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let loud = out_loud.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for channel in 0..hidden {
            assert!(
                (quiet[channel] - loud[channel]).abs() < 1e-4,
                "query 0 changed with token 3: {quiet:?} vs {loud:?}"
            );
        }
    }

    #[test]
    fn non_source_layers_leave_the_shared_compress_cache_intact() {
        use crate::math::precompute_freqs_cis;
        let device = Device::Cpu;
        let heads = 2usize;
        let head_dim = 32usize;
        let hidden = 32usize;
        let qlora = 16usize;
        let ones = |rows: usize, columns: usize| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| ((index % 7) as f32 - 3.0) / 97.0)
                    .collect(),
                (rows, columns),
                &device,
            )
            .unwrap()
        };
        let core = AttentionCore {
            sink: Tensor::zeros(heads, DType::F32, &device).unwrap(),
            wq_a: ones(qlora, hidden),
            q_norm: Tensor::ones(qlora, DType::F32, &device).unwrap(),
            wq_b: ones(heads * head_dim, qlora),
            wkv: ones(head_dim, hidden),
            kv_norm: Tensor::ones(head_dim, DType::F32, &device).unwrap(),
            wo_a: ones(2 * 16, head_dim),
            wo_b: ones(hidden, 2 * 16),
            compressor: None,
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            ring: WindowRing::new(1, 4, head_dim, &device).unwrap(),
            indexer_wq_b: None,
            indexer_wk: None,
            indexer_k_norm: None,
            weights_proj: None,
            freqs: precompute_freqs_cis(8, 16, 0, 1600.0, 1.0, 32, 1, &device).unwrap(),
            index_head_dim: 32,
            index_heads: 2,
            index_topk: 2,
            candidate_topk_blocks: 4,
            candidate_block_size: 4,
            norm_eps: 1e-20,
        };
        let mut core = core;
        let published = Tensor::ones((1, 2, head_dim), DType::F32, &device).unwrap();
        let expected = published.clone();
        let mut runtime = SharedAttentionRuntime {
            compress_kv: Some(published),
            index_k: Some(Tensor::zeros((1, 2, 32), DType::F32, &device).unwrap()),
            topk_idxs: Some(Tensor::from_vec(vec![0i64, 0], (1, 1, 2), &device).unwrap()),
            candidates: None,
        };
        let x = Tensor::from_vec(
            (0..4 * hidden)
                .map(|index| (index % 5) as f32 / 31.0)
                .collect(),
            (1, 4, hidden),
            &device,
        )
        .unwrap();
        core.forward(
            &x,
            0,
            &mut runtime,
            2,
            false,
            false,
            false,
            false,
            0.5f64.sqrt(),
            8,
            16,
            2,
            heads,
            head_dim,
            hidden,
        )
        .unwrap();
        let after = runtime
            .compress_kv
            .expect("non-source layer must not drop the shared cache");
        assert_eq!(after.dims(), expected.dims());
        let kept = after.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            kept.iter().all(|value| *value == 1.0),
            "shared cache overwritten: {kept:?}"
        );
    }

    #[test]
    fn attention_layer_forward_runs_prefill_and_decode() {
        use crate::math::precompute_freqs_cis;
        let device = Device::Cpu;
        let heads = 2usize;
        let head_dim = 32usize;
        let hidden = 32usize;
        let qlora = 16usize;
        let o_lora = 16usize;
        let groups = 2usize;
        let window = 4usize;
        let rope = 8usize;
        let ones = |rows: usize, columns: usize| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| ((index % 7) as f32 - 3.0) / 97.0)
                    .collect(),
                (rows, columns),
                &device,
            )
            .unwrap()
        };
        let freqs = precompute_freqs_cis(rope, 16, 0, 1600.0, 1.0, 32, 1, &device).unwrap();
        let core = AttentionCore {
            sink: Tensor::zeros(heads, DType::F32, &device).unwrap(),
            wq_a: ones(qlora, hidden),
            q_norm: Tensor::ones(qlora, DType::F32, &device).unwrap(),
            wq_b: ones(heads * head_dim, qlora),
            wkv: ones(head_dim, hidden),
            kv_norm: Tensor::ones(head_dim, DType::F32, &device).unwrap(),
            wo_a: ones(groups * o_lora, (heads / groups) * head_dim),
            wo_b: ones(hidden, groups * o_lora),
            compress_cache: Vec::new(),
            index_k_cache: Vec::new(),
            compressor: Some(
                Compressor::new(
                    2,
                    head_dim,
                    Tensor::ones(head_dim, DType::F32, &device).unwrap(),
                    ones(head_dim, hidden),
                    Some(ones(head_dim, hidden)),
                    1,
                    1e-20,
                )
                .unwrap(),
            ),
            ring: WindowRing::new(1, window, head_dim, &device).unwrap(),
            indexer_wq_b: Some(ones(2 * 32, qlora)),
            indexer_wk: Some(ones(32, head_dim)),
            indexer_k_norm: Some(Tensor::ones(32, DType::F32, &device).unwrap()),
            weights_proj: Some(ones(2, hidden)),
            freqs,
            index_head_dim: 32,
            index_heads: 2,
            index_topk: 2,
            candidate_topk_blocks: 4,
            candidate_block_size: 4,
            norm_eps: 1e-20,
        };
        let mut runtime = SharedAttentionRuntime {
            compress_kv: Some(Tensor::zeros((1, 8, head_dim), DType::F32, &device).unwrap()),
            index_k: None,
            topk_idxs: None,
            candidates: None,
        };
        let mut core = core;
        let x = Tensor::from_vec(
            (0..6 * hidden)
                .map(|index| ((index % 5) as f32 - 2.0) / 31.0)
                .collect(),
            (1, 6, hidden),
            &device,
        )
        .unwrap();
        let output = core
            .forward(
                &x,
                0,
                &mut runtime,
                2,
                true,
                true,
                false,
                false,
                0.5f64.sqrt(),
                rope,
                o_lora,
                groups,
                heads,
                head_dim,
                hidden,
            )
            .unwrap();
        assert_eq!(output.dims(), [1, 6, hidden]);
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));

        let step = Tensor::from_vec(
            (0..hidden).map(|index| index as f32 / 41.0).collect(),
            (1, 1, hidden),
            &device,
        )
        .unwrap();
        let output = core
            .forward(
                &step,
                6,
                &mut runtime,
                2,
                true,
                true,
                false,
                false,
                0.5f64.sqrt(),
                rope,
                o_lora,
                groups,
                heads,
                head_dim,
                hidden,
            )
            .unwrap();
        assert_eq!(output.dims(), [1, 1, hidden]);
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn decode_picks_address_rank_order_with_sentinels_for_unfilled_slots() {
        // start_pos 7 with window 4: slots hold positions 4..7 and the view is
        // oldest-first, so every rank is filled and in range.
        let idxs = window_topk_idxs(4, 1, 1, 7);
        let flat = idxs.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(flat, [0, 1, 2, 3]);
        // start_pos 2 (early decode, ring still filling): rank 0 maps to
        // physical slot 3, which the sequence never reached, so it is -1; the
        // remaining ranks cover slots 0..2.
        let idxs = window_topk_idxs(4, 1, 1, 2);
        let flat = idxs.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(flat, [-1, 1, 2, 3]);
    }

    #[test]
    fn prefill_window_picks_stay_within_the_configured_window() {
        let idxs = window_topk_idxs(4, 1, 6, 0);
        // Six-token prefill with a 4-row window: each query row carries at
        // most 4 picks and the earliest queries see -1 for slots before the
        // sequence started.
        assert_eq!(idxs.dims(), [1, 6, 4]);
        let flat = idxs.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(flat[..4], [-1, -1, -1, 0]);
        assert_eq!(flat[5 * 4..6 * 4], [2, 3, 4, 5]);
    }

    #[test]
    fn window_ring_persists_decoded_rows_across_steps() {
        let device = Device::Cpu;
        let mut ring = WindowRing::new(1, 4, 2, &device).unwrap();
        let prefill = Tensor::from_vec(
            (0..8).map(|index| index as f32).collect(),
            (1, 4, 2),
            &device,
        )
        .unwrap();
        ring.write(&prefill, 0).unwrap();
        // Two decode steps: position 4 -> slot 0, position 5 -> slot 1.
        for (position, tag) in [(4usize, 100.0f32), (5, 200.0)] {
            let step = Tensor::from_vec(vec![tag, tag + 1.0], (1, 1, 2), &device).unwrap();
            ring.write(&step, position).unwrap();
        }
        // The third decode read must still see position 4's row somewhere in
        // the ring; a cache that never persisted writes would show zeros.
        let step = Tensor::from_vec(vec![300.0f32, 301.0], (1, 1, 2), &device).unwrap();
        let view = ring.write(&step, 6).unwrap();
        let values = view.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            values
                .windows(2)
                .any(|pair| pair[0] == 100.0 && pair[1] == 101.0),
            "position 4 lost from the ring: {values:?}"
        );
        assert!(
            values
                .windows(2)
                .any(|pair| pair[0] == 200.0 && pair[1] == 201.0),
            "position 5 lost from the ring: {values:?}"
        );
    }

    #[test]
    fn identity_mix_keeps_the_first_copy() {
        let mix = identity_pre_mix(2, 3, 4, &Device::Cpu).unwrap();
        let values = mix.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[4], 1.0);
    }

    #[test]
    fn greedy_selection_rejects_non_finite_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0], (1, 3), &device).unwrap();
        let error = greedy_token_from_logits(&logits).unwrap_err();
        assert!(error.to_string().contains("non-finite"), "{error:#}");
    }

    #[test]
    fn expand_hc_repeats_each_token() {
        let device = Device::Cpu;
        let embedding = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &device).unwrap();
        let expanded = expand_hc(&embedding, 3).unwrap();
        assert_eq!(expanded.dims(), [1, 2, 3, 2]);
        let values = expanded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values[..4], [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(values[4..8], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(values[8..12], [3.0, 4.0, 3.0, 4.0]);
    }
}
