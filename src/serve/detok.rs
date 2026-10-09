use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Tokenizer loaded once and shared; incremental detokenization decodes
/// the accumulated ids and emits only the new suffix. Runs on the server
/// side, never on the engine thread.
pub struct Shared {
    tokenizer: Arc<Tokenizer>,
}

pub struct Detok {
    tokenizer: Arc<Tokenizer>,
    decoded: String,
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
        self.tokenizer
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))
    }

    pub fn detok(&self) -> Detok {
        Detok {
            tokenizer: self.tokenizer.clone(),
            decoded: String::new(),
            ids: Vec::new(),
        }
    }
}

impl Detok {
    /// Feed newly emitted ids; returns the newly stable text suffix.
    pub fn push(&mut self, ids: &[u32]) -> Result<String> {
        self.ids.extend_from_slice(ids);
        let full = self
            .tokenizer
            .decode(&self.ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        if full.len() <= self.decoded.len() {
            return Ok(String::new());
        }
        ensure_prefix(&self.decoded, &full)?;
        let suffix = full[self.decoded.len()..].to_string();
        self.decoded = full;
        Ok(suffix)
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn text(&self) -> &str {
        &self.decoded
    }
}

fn ensure_prefix(shorter: &str, longer: &str) -> Result<()> {
    anyhow::ensure!(
        longer.starts_with(shorter),
        "detokenizer output is not append-only: {longer:?} does not extend {shorter:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Shared {
        Shared::load(Path::new(
            "/home/greatx/repos/flyingfish/models/openbmb/MiniCPM5-2B",
        ))
        .expect("load")
    }

    #[test]
    fn push_yields_append_only_suffixes() {
        let shared = shared();
        let ids = shared.encode("Large language models are").expect("encode");
        let mut detok = shared.detok();
        let half = detok.push(&ids[..2]).expect("push");
        let rest = detok.push(&ids[2..]).expect("push");
        assert_eq!(detok.ids(), ids.as_slice());
        assert_eq!(format!("{half}{rest}"), detok.text());
    }
}
