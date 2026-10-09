use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Tokenizer loaded once and shared; incremental detokenization decodes the
/// accumulated ids and emits only the stable new suffix. Runs on the server
/// side, never on the engine thread.
pub struct Shared {
    tokenizer: Arc<Tokenizer>,
}

pub struct Detok {
    tokenizer: Arc<Tokenizer>,
    emitted: String,
    full: String,
    ids: Vec<u32>,
}

impl Shared {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))
            .context("detok init")?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
        })
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

    pub fn detok(&self) -> Detok {
        Detok {
            tokenizer: self.tokenizer.clone(),
            emitted: String::new(),
            full: String::new(),
            ids: Vec::new(),
        }
    }
}

impl Detok {
    /// Feed newly emitted ids; returns the newly stable text suffix. A
    /// trailing run of U+FFFD replacement characters marks an incomplete
    /// multi-byte sequence (byte-fallback tokens split across rounds) and
    /// is held back until it completes or the stream finishes.
    pub fn push(&mut self, ids: &[u32]) -> Result<String> {
        self.ids.extend_from_slice(ids);
        let full = self.decode()?;
        let stable = trim_trailing_replacements(&full).to_string();
        self.full = full;
        Ok(self.advance_to(&stable))
    }

    /// Flush everything held back; call once when generation is complete.
    pub fn finish(&mut self) -> Result<String> {
        let full = self.full.clone();
        Ok(self.advance_to(&full))
    }

    fn decode(&self) -> Result<String> {
        self.tokenizer
            .decode(&self.ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
    }

    fn advance_to(&mut self, stable: &str) -> String {
        if stable.starts_with(&self.emitted) {
            let suffix = stable[self.emitted.len()..].to_string();
            self.emitted = stable.to_string();
            suffix
        } else {
            let common = self
                .emitted
                .char_indices()
                .zip(stable.chars())
                .take_while(|((_, a), b)| a == b)
                .last()
                .map_or(0, |((i, c), _)| i + c.len_utf8());
            let suffix = stable[common..].to_string();
            self.emitted = stable.to_string();
            suffix
        }
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn text(&self) -> &str {
        &self.full
    }
}

fn trim_trailing_replacements(text: &str) -> &str {
    let mut cut = text.len();
    for c in text.chars().rev() {
        if c == '\u{FFFD}' {
            cut -= c.len_utf8();
        } else {
            break;
        }
    }
    &text[..cut]
}

#[cfg(test)]
mod tests {
    use super::*;

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
