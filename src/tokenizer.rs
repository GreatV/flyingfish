use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) fn load(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer load: {e}"))
}

pub fn encode(dir: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer = load(dir)?;
    Ok(tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("tokenizer encode: {e}"))?
        .get_ids()
        .to_vec())
}

pub fn decode(dir: &Path, ids: &[u32]) -> Result<String> {
    decode_with(&load(dir)?, ids)
}

pub(crate) fn decode_with(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow!("tokenizer decode: {e}"))
}
