use anyhow::{Result, anyhow};
use std::path::Path;
use tokenizers::Tokenizer;

pub fn encode(dir: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer load: {e}"))?;
    Ok(tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("tokenizer encode: {e}"))?
        .get_ids()
        .to_vec())
}

pub fn decode(dir: &Path, ids: &[u32]) -> Result<String> {
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer load: {e}"))?;
    tokenizer
        .decode(ids, true)
        .map_err(|e| anyhow!("tokenizer decode: {e}"))
}
