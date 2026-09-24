//! Engram n-gram conditional memory.
//!
//! Reference: `inference/engram.py`. A position is hashed by the
//! `(max_ngram_size - 1)` n-grams ending there, each split over `n_heads`
//! prime-sized buckets; every (n-gram size, head) pair owns a disjoint prime
//! range drawn in order from just above `engram_vocab_size`.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};

use crate::config::TextConfig;
use crate::numpy_random::engram_multipliers;

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn next_prime_above(start: u64, seen: &mut Vec<u64>) -> u64 {
    let mut candidate = start + 1;
    loop {
        if is_prime(candidate) && !seen.contains(&candidate) {
            seen.push(candidate);
            return candidate;
        }
        candidate += 1;
    }
}

/// Build the compressed token map: every vocabulary id onto a smaller id
/// space where tokens that normalize alike collapse together. Mirrors
/// `build_compressed_token_map` in `inference/engram.py` with the same
/// normalizer chain, including the private-use sentinel that keeps a lone
/// space from being stripped away.
pub fn build_compressed_token_map(tokenizer: &tokenizers::Tokenizer) -> Result<(Vec<u32>, u32)> {
    use tokenizers::normalizers::replace::ReplacePattern;
    use tokenizers::normalizers::{Lowercase, NFD, NFKC, Replace, Sequence, Strip, StripAccents};
    use tokenizers::{NormalizedString, Normalizer as _};

    let sentinel = '\u{e000}';
    let make = |pattern: ReplacePattern, content: String| -> anyhow::Result<_> {
        Replace::new(pattern, content).map_err(|error| anyhow::anyhow!("normalizer: {error}"))
    };
    let sequence: Vec<tokenizers::normalizers::NormalizerWrapper> = vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        make(
            ReplacePattern::Regex("[ \\t\\r\\n]+".to_owned()),
            " ".to_owned(),
        )?
        .into(),
        make(
            ReplacePattern::Regex("^ $".to_owned()),
            sentinel.to_string(),
        )?
        .into(),
        Strip::new(true, true).into(),
        make(ReplacePattern::String(sentinel.to_string()), " ".to_owned())?.into(),
    ];
    let normalizer = Sequence::new(sequence);
    let mut key_to_new = std::collections::HashMap::new();
    let mut lookup = vec![0u32; tokenizer.get_vocab_size(false)];
    for (token_id, slot) in lookup.iter_mut().enumerate() {
        let text = tokenizer
            .decode(&[token_id as u32], false)
            .unwrap_or_default();
        let key = if text.contains('\u{fffd}') {
            tokenizer.id_to_token(token_id as u32).unwrap_or_default()
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|error| anyhow::anyhow!("normalize token text: {error}"))?;
            let normalized = normalized.get().to_owned();
            if normalized.is_empty() {
                text
            } else {
                normalized
            }
        };
        let next = key_to_new.len() as u32;
        *slot = *key_to_new.entry(key).or_insert(next);
    }
    Ok((lookup, key_to_new.len() as u32))
}

/// The per-layer hash multipliers, reproducing numpy's
/// `default_rng(10007 * layer_id).integers(0, bound, max_ngram_size)` stream
/// bit-for-bit; see `numpy_random`. The bound keeps `token * multiplier`
/// inside int64.
pub fn hash_multipliers(
    layer_id: usize,
    compressed_vocab: u32,
    max_ngram_size: usize,
) -> Result<Vec<u64>> {
    ensure!(compressed_vocab > 0, "compressed vocab must be positive");
    let bound = std::cmp::max(1, i64::MAX / compressed_vocab as i64 / 2) as u64;
    Ok(engram_multipliers(layer_id, bound, max_ngram_size))
}

#[derive(Clone, Debug)]
pub struct EngramLayout {
    pub max_ngram_size: usize,
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<u64>,
    pub primes: Vec<Vec<Vec<u64>>>,
    pub offsets: Vec<Vec<u64>>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl EngramLayout {
    pub fn from_config(config: &TextConfig) -> Result<Self> {
        ensure!(
            !config.engram_layer_ids.is_empty(),
            "engram layout needs at least one layer"
        );
        let mut seen = Vec::new();
        let mut primes = Vec::new();
        let mut offsets = Vec::new();
        for _ in &config.engram_layer_ids {
            let mut per_ngram = Vec::new();
            for _ in 1..config.engram_max_ngram_size {
                let mut sizes = Vec::new();
                for _ in 0..config.engram_n_heads {
                    sizes.push(next_prime_above(config.engram_vocab_size - 1, &mut seen));
                }
                per_ngram.push(sizes);
            }
            let flat: Vec<u64> = per_ngram.iter().flatten().copied().collect();
            let mut running = vec![0u64];
            let mut offset = 0u64;
            for size in &flat[..flat.len().saturating_sub(1)] {
                offset += size;
                running.push(offset);
            }
            primes.push(per_ngram);
            offsets.push(running);
        }
        Ok(Self {
            max_ngram_size: config.engram_max_ngram_size,
            layer_ids: config.engram_layer_ids.clone(),
            num_embeddings: config.engram_num_embeddings.clone(),
            primes,
            offsets,
            n_heads: config.engram_n_heads,
            head_dim: config.engram_head_dim,
        })
    }

    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_heads
    }
}

/// Maps each position to the hash ids of the n-grams ending there.
///
/// Look-back stops at the start of the sequence and at any dead token (an
/// image span, stored as `DEAD`), so an n-gram never spans one. The cache
/// carries compressed ids across the prefill/decode split.
pub struct NgramHashState {
    layout: EngramLayout,
    multipliers: Vec<Vec<u64>>,
    token_map: Vec<u32>,
    pad_id: u32,
    cache: Vec<i64>,
    max_seq: usize,
    batch: usize,
}

impl NgramHashState {
    pub const DEAD: i64 = -1;

    pub fn new(
        layout: EngramLayout,
        token_map: Vec<u32>,
        batch: usize,
        max_seq: usize,
        compressed_vocab_size: u32,
        pad_token_id: u32,
    ) -> Result<Self> {
        let multipliers = layout
            .layer_ids
            .iter()
            .map(|layer| {
                hash_multipliers(*layer, compressed_vocab_size, layout.max_ngram_size)
                    .with_context(|| format!("hash multipliers for engram layer {layer}"))
            })
            .collect::<Result<Vec<Vec<u64>>>>()?;
        let pad_id = *token_map
            .get(pad_token_id as usize)
            .with_context(|| format!("pad token id {pad_token_id} is not in the token map"))?;
        Ok(Self {
            layout,
            multipliers,
            token_map,
            pad_id,
            cache: vec![0; batch * max_seq],
            max_seq,
            batch,
        })
    }

    /// Returns hash ids shaped `[batch, seq, layers, n_hash_cols]`.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        token_mask: Option<&[bool]>,
    ) -> Result<Tensor> {
        let dims = input_ids.dims();
        ensure!(dims.len() == 2, "input ids must be [batch, seq]");
        let [batch, seqlen] = [dims[0], dims[1]];
        ensure!(batch == self.batch, "hash batch changed mid-run");
        ensure!(start_pos + seqlen <= self.max_seq, "hash cache overflow");
        let ids = input_ids
            .flatten_all()?
            .to_vec1::<u32>()
            .context("read input ids")?;
        let compressed: Vec<i64> = ids
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let flat = index / seqlen * self.max_seq + start_pos + index % seqlen;
                let compressed = self.token_map[*id as usize] as i64;
                match token_mask {
                    Some(mask) if !mask[index] => Self::DEAD,
                    _ => compressed,
                }
                .pipe(|value| {
                    self.cache[flat] = value;
                    value
                })
            })
            .collect();
        let layers = self.layout.layer_ids.len();
        let ngram = self.layout.max_ngram_size;
        let heads = self.layout.n_heads;
        let cols = self.layout.n_hash_cols();
        let mut hashes = vec![0i64; batch * seqlen * layers * cols];
        for b in 0..batch {
            for s in 0..seqlen {
                let position = start_pos + s;
                let mut tokens = vec![self.pad_id as i64; ngram];
                let mut blocked = false;
                for (shift, token) in tokens.iter_mut().enumerate() {
                    if position < shift {
                        blocked = true;
                        continue;
                    }
                    let source = if shift == 0 {
                        compressed[b * seqlen + s]
                    } else {
                        self.cache[b * self.max_seq + position - shift]
                    };
                    blocked = blocked || source == Self::DEAD;
                    *token = if blocked { self.pad_id as i64 } else { source };
                }
                for (layer, multipliers) in self.multipliers.iter().enumerate() {
                    let mut rolling = (tokens[0] as i64).wrapping_mul(multipliers[0] as i64);
                    for i in 1..ngram {
                        let product = (tokens[i] as i64).wrapping_mul(multipliers[i] as i64);
                        rolling ^= product;
                        // One rolling hash broadcasts across every head bucket of
                        // this n-gram size; each head reads its own prime range.
                        let ngram_index = i - 1;
                        for head in 0..heads {
                            let bucket = self.layout.primes[layer][ngram_index][head];
                            let offset = self.layout.offsets[layer]
                                .get(ngram_index * heads + head)
                                .copied()
                                .unwrap_or(0);
                            let column = ngram_index * heads + head;
                            hashes[((b * seqlen + s) * layers + layer) * cols + column] =
                                rolling.rem_euclid(bucket as i64) + offset as i64;
                        }
                    }
                }
            }
        }
        Tensor::from_vec(hashes, (batch, seqlen, layers, cols), input_ids.device())
            .map_err(anyhow::Error::from)
    }
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

/// Writes an n-gram lookup into the residual stream, gated by how well it
/// matches that stream. Reference: `Engram` in `inference/model.py`.
pub fn engram_forward(
    x: &Tensor,
    embedded: &Tensor,
    wkv: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    token_mask: Option<&[bool]>,
) -> Result<Tensor> {
    let dims = x.dims();
    ensure!(
        dims.len() == 4,
        "engram input must be [batch, seq, hc, dim]"
    );
    let [batch, seq, hc, hidden] = [dims[0], dims[1], dims[2], dims[3]];
    let embed_dims = embedded.dims();
    ensure!(
        embed_dims.len() == 4 && embed_dims[0] == batch && embed_dims[1] == seq,
        "embedded hashes must be [batch, seq, cols, head_dim]"
    );
    let cols = embed_dims[2];
    let head_dim = embed_dims[3];
    let x_values = x
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read engram stream")?;
    let embedded = embedded
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read engram embeddings")?;
    let (wkv_guard, _) = crate::math::resident_f32(wkv)?;
    let wkv = crate::math::resident_f32_slice(&wkv_guard)?;
    let q_weight = q_weight
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read engram q weight")?;
    let k_weight = k_weight
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("read engram k weight")?;
    let kv_width = hidden * (hc + 1);
    ensure!(
        wkv.len() == kv_width * cols * head_dim,
        "engram wkv must project {cols} x {head_dim} channels onto {kv_width}"
    );
    let mut output = x_values.clone();
    for b in 0..batch {
        for s in 0..seq {
            let mut kv = vec![0.0f32; kv_width];
            for row in 0..kv_width {
                let mut sum = 0.0;
                for source in 0..cols * head_dim {
                    sum += wkv[row * cols * head_dim + source]
                        * embedded[((b * seq + s) * cols) * head_dim + source];
                }
                kv[row] = sum;
            }
            let key = &kv[..hc * hidden];
            let value = &kv[hc * hidden..];
            for copy in 0..hc {
                let stream = &x_values[((b * seq + s) * hc + copy) * hidden..][..hidden];
                let key_row = &key[copy * hidden..][..hidden];
                let stream_rstd = 1.0
                    / (stream.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + 1e-20).sqrt();
                let key_rstd = 1.0
                    / (key_row.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + 1e-20).sqrt();
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += q_weight[copy * hidden + d]
                        * k_weight[copy * hidden + d]
                        * stream[d]
                        * key_row[d];
                }
                let dot = dot * stream_rstd * key_rstd * (hidden as f32).recip().sqrt();
                let signed = dot.abs().max(1e-6).sqrt().copysign(dot);
                let mut gate = 1.0 / (1.0 + (-signed).exp());
                if token_mask.is_some_and(|mask| !mask[b * seq + s]) {
                    gate = 0.0;
                }
                for d in 0..hidden {
                    output[((b * seq + s) * hc + copy) * hidden + d] += gate * value[d];
                }
            }
        }
    }
    Tensor::from_vec(output, dims.to_vec(), x.device()).map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff_core::paths::checkpoint_dir;

    #[test]
    fn prime_chain_starts_above_the_vocabulary() {
        let mut seen = Vec::new();
        let first = next_prime_above(15_999_999, &mut seen);
        let second = next_prime_above(first, &mut seen);
        assert!(first > 15_999_999);
        assert!(second > first);
        assert!(seen.contains(&first) && seen.contains(&second));
    }

    #[test]
    fn derived_multipliers_match_the_training_values() {
        let one = hash_multipliers(1, 99_092, 4).unwrap();
        let fourteen = hash_multipliers(14, 99_092, 4).unwrap();
        assert_eq!(
            one,
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391
            ]
        );
        assert_eq!(fourteen[3], 82_619_226_485_591);
    }

    #[test]
    fn layout_sizes_match_the_configuration() {
        let Some(dir) = checkpoint_dir("deepseek-ai/DeepSeek-V4.1-Flash") else {
            return;
        };
        let dir = &dir;
        if !dir.exists() {
            return;
        }
        let config = crate::config::DeepseekV41Config::from_model_dir(dir).unwrap();
        let layout = EngramLayout::from_config(&config.text_config).unwrap();
        assert_eq!(layout.n_hash_cols(), 24);
        assert_eq!(layout.primes.len(), 2);
        assert_eq!(layout.primes[0].len(), 3);
        assert_eq!(layout.primes[0][0].len(), 8);
        // Offsets lead with zero, then accumulate the earlier prime sizes.
        assert_eq!(layout.offsets[0][0], 0);
        assert_eq!(layout.offsets[0][1], layout.primes[0][0][0]);
        assert_eq!(
            layout.offsets[0][2],
            layout.primes[0][0][0] + layout.primes[0][0][1]
        );
    }

    #[test]
    fn engram_value_broadcasts_across_hc_copies() {
        let device = candle_core::Device::Cpu;
        let (batch, seq, hc, hidden, head_dim, cols) = (1, 1, 2, 4, 3, 2);
        let x = Tensor::zeros((batch, seq, hc, hidden), DType::F32, &device).unwrap();
        // Embedded hash rows with a distinct value scale per column.
        let mut embedded = vec![0.0f32; batch * seq * cols * head_dim];
        for c in 0..cols {
            for d in 0..head_dim {
                embedded[c * head_dim + d] = (c as f32 + 1.0) * 0.1;
            }
        }
        let embedded = Tensor::from_vec(embedded, (batch, seq, cols, head_dim), &device).unwrap();
        // wkv: key rows zero, one shared value row = column sums.
        let mut wkv = vec![0.0f32; hidden * (hc + 1) * cols * head_dim];
        let value_base = hc * hidden;
        for c in 0..cols {
            for d in 0..head_dim {
                for channel in 0..hidden {
                    wkv[(value_base + channel) * cols * head_dim + c * head_dim + d] =
                        (c as f32 + 1.0) * 0.1;
                }
            }
        }
        let wkv = Tensor::from_vec(wkv, (hidden * (hc + 1), cols * head_dim), &device).unwrap();
        let q_weight = Tensor::ones(hc * hidden, DType::F32, &device).unwrap();
        let k_weight = Tensor::ones(hc * hidden, DType::F32, &device).unwrap();
        let out = engram_forward(&x, &embedded, &wkv, &q_weight, &k_weight, None).unwrap();
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Zero stream and zero gate input (sigmoid(0)-ish via clamped dot at 0):
        // gate = sigmoid(sqrt(clamp(0))) = sigmoid(0) = 0.5. Both hc copies add
        // the same shared value row; an offset-into-value indexing (copy*hidden)
        // would read zeros for copy 0 (value starts at hc*hidden).
        for copy in 0..hc {
            for channel in 0..hidden {
                let slot = copy * hidden + channel;
                assert!(
                    (values[slot] - values[channel]).abs() < 1e-5,
                    "copy {copy} diverged: {values:?}"
                );
            }
        }
        assert!(values[0].abs() > 1e-3, "value row never added: {values:?}");
    }

    #[test]
    fn every_head_bucket_gets_a_hash_column() {
        // One layer, three heads, 2-gram: the single rolling hash must land in
        // all three head buckets, each with its own prime and offset.
        let layout = EngramLayout {
            max_ngram_size: 2,
            layer_ids: vec![1],
            num_embeddings: vec![10],
            primes: vec![vec![vec![7, 11, 13]]],
            offsets: vec![vec![0, 0, 0]],
            n_heads: 3,
            head_dim: 2,
        };
        let token_map = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut state = NgramHashState::new(layout, token_map, 1, 8, 99_092, 2).unwrap();
        let ids = Tensor::from_vec(vec![5u32], (1, 1), &candle_core::Device::Cpu).unwrap();
        let hashes = state.forward(&ids, 0, None).unwrap();
        assert_eq!(hashes.dims(), [1, 1, 1, 3]);
        let values = hashes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        let m = hash_multipliers(1, 99_092, 2).unwrap();
        let rolling = (5i64).wrapping_mul(m[0] as i64) ^ (2i64).wrapping_mul(m[1] as i64);
        let residue = rolling.rem_euclid(7);
        assert_eq!(values[0], residue);
        assert_eq!(values[1], rolling.rem_euclid(11));
        assert_eq!(values[2], rolling.rem_euclid(13));
    }

    #[test]
    fn construction_uses_the_configured_vocab_and_pad() {
        let layout = || EngramLayout {
            max_ngram_size: 2,
            layer_ids: vec![1],
            num_embeddings: vec![10],
            primes: vec![vec![vec![7, 11, 13]]],
            offsets: vec![vec![0, 0, 0]],
            n_heads: 3,
            head_dim: 2,
        };
        let state =
            NgramHashState::new(layout(), vec![10u32, 11, 12, 13, 14, 15], 1, 8, 77, 3).unwrap();
        assert_eq!(state.pad_id, 13);
        assert!(
            NgramHashState::new(layout(), vec![10u32], 1, 8, 99_092, 9).is_err(),
            "an out-of-map pad id must be rejected"
        );
    }

    #[test]
    fn dead_tokens_block_the_lookback() {
        // A tiny hand-checked hash: one layer, one head, 2-grams only.
        let layout = EngramLayout {
            max_ngram_size: 2,
            layer_ids: vec![1],
            num_embeddings: vec![10],
            primes: vec![vec![vec![7]]],
            offsets: vec![vec![0]],
            n_heads: 1,
            head_dim: 2,
        };
        let token_map = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut state = NgramHashState::new(layout, token_map, 1, 8, 99_092, 2).unwrap();
        let ids = Tensor::from_vec(vec![3u32, 3], (1, 2), &candle_core::Device::Cpu).unwrap();
        let hashes = state.forward(&ids, 0, None).unwrap();
        let values = hashes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        // Position 1: 2-gram (3,3) -> (3*m0 ^ 3*m1) % 7 with pinned multipliers.
        let m = hash_multipliers(1, 99_092, 2).unwrap();
        let rolling = (3i64).wrapping_mul(m[0] as i64) ^ (3i64).wrapping_mul(m[1] as i64);
        assert_eq!(values[1], rolling.rem_euclid(7));
        // Position 0 has no history: pad id 2 enters the 2-gram instead.
        let rolling_pad = (3i64).wrapping_mul(m[0] as i64) ^ (2i64).wrapping_mul(m[1] as i64);
        assert_eq!(values[0], rolling_pad.rem_euclid(7));
        // Masking the second token pads its whole n-gram, history included.
        let hashes = state.forward(&ids, 0, Some(&[true, false])).unwrap();
        let values = hashes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        let rolling_dead = (2i64).wrapping_mul(m[0] as i64) ^ (2i64).wrapping_mul(m[1] as i64);
        assert_eq!(values[1], rolling_dead.rem_euclid(7));
    }
}
