use anyhow::{Context, Result, ensure};
use std::path::Path;
use std::sync::Arc;
use tokenizers::{Tokenizer, decoders::DecoderWrapper};

/// One tokenizer shared by all requests.
pub struct Shared {
    tokenizer: Arc<Tokenizer>,
    byte_level: bool,
    byte_fallback: bool,
}

pub struct Detok {
    tokenizer: Arc<Tokenizer>,
    full: String,
    ids: Vec<u32>,
    context: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    deferred: Vec<u32>,
    byte_level: bool,
    byte_fallback: bool,
    #[cfg(test)]
    decode_visits: std::cell::Cell<usize>,
}

fn has_byte_fallback(decoder: &DecoderWrapper) -> bool {
    match decoder {
        DecoderWrapper::ByteFallback(_) => true,
        DecoderWrapper::Sequence(sequence) => sequence.get_decoders().iter().any(has_byte_fallback),
        _ => false,
    }
}

impl Shared {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))
            .context("detok init")?;
        Ok(Self::new(tokenizer))
    }

    pub(super) fn new(tokenizer: Tokenizer) -> Self {
        Self {
            byte_level: matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))),
            byte_fallback: tokenizer.get_decoder().is_some_and(has_byte_fallback),
            tokenizer: Arc::new(tokenizer),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with(text, true)
    }

    /// Encodes rendered chat template output, which already contains
    /// bos_token, without adding special tokens.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with(text, false)
    }

    fn encode_with(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))
    }

    pub fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .with_context(|| format!("tokenizer lacks token {token:?}"))
    }

    pub fn detok(&self) -> Detok {
        Detok {
            tokenizer: self.tokenizer.clone(),
            full: String::new(),
            ids: Vec::new(),
            context: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            deferred: Vec::new(),
            byte_level: self.byte_level,
            byte_fallback: self.byte_fallback,
            #[cfg(test)]
            decode_visits: std::cell::Cell::new(0),
        }
    }
}

impl Detok {
    /// Returns stable text from the new IDs; incomplete UTF-8 stays in context.
    pub fn push(&mut self, ids: &[u32]) -> Result<String> {
        let tokens = ids
            .iter()
            .map(|&id| {
                self.tokenizer
                    .id_to_token(id)
                    .with_context(|| format!("token id {id} outside vocabulary"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.ids.extend_from_slice(ids);
        let mut suffix = String::new();
        for (&id, token) in ids.iter().zip(tokens) {
            if self
                .tokenizer
                .get_added_vocabulary()
                .is_special_token(&token)
                || (self.byte_level && token.is_empty())
            {
                continue;
            }
            self.deferred.push(id);
            if self.byte_fallback
                && token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
                && u8::from_str_radix(&token[3..5], 16).is_ok()
            {
                continue;
            }
            suffix.push_str(&self.step()?);
            suffix.push_str(&self.compact_replacements()?);
        }
        self.full.push_str(&suffix);
        Ok(suffix)
    }

    /// Flush everything held back; call once when generation is complete.
    pub fn finish(&mut self) -> Result<String> {
        let full = self.decode(&self.ids)?;
        ensure!(
            full.starts_with(&self.full),
            "tokenizer changed the committed streaming prefix"
        );
        let suffix = full[self.full.len()..].to_string();
        self.full = full;
        self.context.clear();
        self.deferred.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        Ok(suffix)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        #[cfg(test)]
        self.decode_visits.set(self.decode_visits.get() + ids.len());
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
    }

    fn step(&mut self) -> Result<String> {
        #[cfg(test)]
        let (old, empty, extended) = (
            self.context.len(),
            self.prefix.is_empty(),
            self.context.len() + self.deferred.len(),
        );
        let result = tokenizers::tokenizer::step_decode_stream(
            &*self.tokenizer,
            std::mem::take(&mut self.deferred),
            true,
            &mut self.context,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| anyhow::anyhow!("tokenizer stream: {error}"))?;
        #[cfg(test)]
        self.decode_visits.set(
            self.decode_visits.get()
                + extended
                + if empty { old } else { 0 }
                + if result.is_some() {
                    self.context.len()
                } else {
                    0
                },
        );
        Ok(result.unwrap_or_default())
    }

    fn compact_replacements(&mut self) -> Result<String> {
        if !self.byte_level || self.context.len() <= 8 {
            return Ok(String::new());
        }
        let decoded = self.decode(&self.context)?;
        ensure!(
            decoded.starts_with(&self.prefix),
            "tokenizer changed the rolling streaming prefix"
        );
        for first in (1..=self.context.len() - 3).rev() {
            let tail = self.decode(&self.context[first..])?;
            let stable = decoded.len().saturating_sub(tail.len());
            if decoded.ends_with(&tail) && stable >= self.prefix.len() {
                let suffix = decoded[self.prefix.len()..stable].to_string();
                self.context.drain(..first);
                self.prefix.clear();
                self.prefix_index = 0;
                return Ok(suffix);
            }
        }
        anyhow::bail!("byte-level tokenizer has no stable UTF-8 streaming boundary")
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn text(&self) -> &str {
        &self.full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streaming_fixture(byte_fallback: bool) -> Result<Shared> {
        let mut tokenizer = if byte_fallback {
            let vocab = (0..256)
                .map(|byte| (format!("<0x{byte:02X}>"), byte))
                .chain([
                    ("a".into(), 256),
                    ("b".into(), 257),
                    (" ".into(), 258),
                    ("[UNK]".into(), 259),
                ])
                .collect();
            let model = tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .map_err(|error| anyhow::anyhow!("fixture model: {error}"))?;
            let mut tokenizer = Tokenizer::new(model);
            tokenizer.with_decoder(Some(tokenizers::decoders::sequence::Sequence::new(vec![
                tokenizers::decoders::byte_fallback::ByteFallback::new().into(),
                tokenizers::decoders::fuse::Fuse::new().into(),
            ])));
            tokenizer
        } else {
            let mut alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
                .into_iter()
                .collect::<Vec<_>>();
            alphabet.sort_unstable();
            let vocab: tokenizers::models::bpe::Vocab = alphabet
                .into_iter()
                .enumerate()
                .map(|(id, character)| (character.to_string(), id as u32))
                .chain([("[UNK]".into(), 256)])
                .collect();
            let model = tokenizers::models::bpe::BPE::builder()
                .vocab_and_merges(vocab, Vec::new())
                .unk_token("[UNK]".into())
                .build()
                .map_err(|error| anyhow::anyhow!("fixture model: {error}"))?;
            let mut tokenizer = Tokenizer::new(model);
            tokenizer.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, false, false),
            ));
            tokenizer.with_decoder(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
            ));
            tokenizer
        };
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("<end>", true)]);
        Ok(Shared::new(tokenizer))
    }

    fn check_stream(shared: &Shared, ids: &[u32], grouped: bool) -> Result<usize> {
        let expected = shared
            .tokenizer
            .decode(ids, true)
            .map_err(|error| anyhow::anyhow!("fixture decode: {error}"))?;
        let mut detok = shared.detok();
        let mut emitted = String::new();
        let mut first = 0;
        let mut turn = 0;
        while first < ids.len() {
            let count = if grouped { [1, 5, 2, 13][turn % 4] } else { 1 };
            let last = (first + count).min(ids.len());
            emitted.push_str(&detok.push(&ids[first..last])?);
            first = last;
            turn += 1;
        }
        emitted.push_str(&detok.finish()?);
        assert_eq!(emitted, expected);
        assert_eq!(detok.text(), expected);
        assert_eq!(detok.ids(), ids);
        let visits = detok.decode_visits.get();
        assert!(
            visits <= 16 * ids.len(),
            "visits={visits} tokens={}",
            ids.len()
        );
        Ok(visits)
    }

    #[test]
    fn streaming_detok_matches_batch_with_linear_work() -> Result<()> {
        for byte_fallback in [false, true] {
            let shared = streaming_fixture(byte_fallback)?;
            let special = shared.token_id("<end>")?;
            let pattern = if byte_fallback {
                vec![256, 0xC3, 0xA9, 257, 0xF0, 0x9F, 0x8C, 0x8D, 258, special]
            } else {
                shared.encode("ab é 中🌍 ")?
            };
            for count in [4096, 8192] {
                let ids = pattern
                    .iter()
                    .copied()
                    .cycle()
                    .take(count)
                    .collect::<Vec<_>>();
                for grouped in [false, true] {
                    let visits = check_stream(&shared, &ids, grouped)?;
                    eprintln!(
                        "PASS detok_linear fallback={byte_fallback} grouped={grouped} tokens={count} visits={visits} bound={}",
                        16 * count
                    );
                }
                let visits = check_stream(&shared, &vec![special; count], false)?;
                eprintln!(
                    "PASS detok_specials fallback={byte_fallback} tokens={count} visits={visits}"
                );
                let invalid = if byte_fallback {
                    255
                } else {
                    shared.token_id("ÿ")?
                };
                let mut tail = vec![invalid; count];
                tail.push(if byte_fallback {
                    0xC3
                } else {
                    shared.token_id("Ã")?
                });
                let visits = check_stream(&shared, &tail, false)?;
                eprintln!(
                    "PASS detok_replacements fallback={byte_fallback} tokens={} visits={visits}",
                    tail.len()
                );
                if !byte_fallback {
                    let mixed = (0..count)
                        .map(|i| ((i * 197 + i / 17) % 256) as u32)
                        .collect::<Vec<_>>();
                    check_stream(&shared, &mixed, false)?;
                    check_stream(&shared, &mixed, true)?;
                }
            }
            let mut detok = shared.detok();
            assert!(detok.push(&[u32::MAX]).is_err());
            assert!(detok.ids().is_empty());
            if !byte_fallback {
                let ids = shared.encode("é")?;
                assert_eq!(ids.len(), 2);
                assert_eq!(detok.push(&ids[..1])?, "");
                assert_eq!(detok.push(&ids[1..])?, "é");
                assert_eq!(detok.finish()?, "");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_DETOK_MODEL pointing at a model directory with tokenizer.json"]
    fn push_holds_back_and_releases_multibyte() {
        let dir = std::env::var("FF_DETOK_MODEL").expect("FF_DETOK_MODEL");
        let shared = Shared::load(Path::new(&dir)).expect("load");
        let text = "你好，世界！ emoji 🌍 test";
        let ids = shared.encode(text).expect("encode");
        let mut detok = shared.detok();
        let mut emitted = String::new();
        for id in &ids {
            emitted.push_str(&detok.push(std::slice::from_ref(id)).expect("push"));
        }
        emitted.push_str(&detok.finish().expect("finish"));
        assert_eq!(detok.ids(), ids.as_slice());
        assert_eq!(detok.text(), text);
        assert_eq!(emitted, text);
    }
}
