//! Music3's special-token caption/lyrics contract.
use anyhow::{Context, Result};
use regex::{Captures, Regex};
use tokenizers::Tokenizer;

pub fn format_prompt(caption: &str, lyrics: &str) -> Result<String> {
    anyhow::ensure!(
        !caption.trim().is_empty() && !lyrics.trim().is_empty(),
        "caption and lyrics must be nonempty"
    );
    let special = Regex::new(r"<\|([^|]*)\|>")?;
    let caption = special.replace_all(caption, |captures: &Captures<'_>| {
        let inner = captures[1].trim();
        match inner.split_once(char::is_whitespace) {
            Some((key, value)) => format!("{key} is {}", value.trim_start()),
            None => inner.to_string(),
        }
    });
    let heading = Regex::new(r"^\s{0,3}#{1,6}\s+")?;
    let bullet = Regex::new(r"^\s*[*+-]\s+")?;
    let bold = Regex::new(r"\*\*([^*]+)\*\*")?;
    let italic = Regex::new(r"(^|[^*])\*([^*\n]+)\*([^*]|$)")?;
    let mut lines = Vec::new();
    for line in caption.lines() {
        let mut line = bullet.replace(&heading.replace(line, ""), "").to_string();
        loop {
            let next = bold.replace_all(&line, "$1").to_string();
            if next == line {
                break;
            }
            line = next;
        }
        loop {
            let next = italic.replace_all(&line, "$1$2$3").to_string();
            if next == line {
                break;
            }
            line = next;
        }
        lines.push(line.trim_end().to_string());
    }
    let caption = Regex::new(r"(?m)^\s*[-*_]{3,}\s*$")?
        .replace_all(&lines.join("\n"), "")
        .to_string();
    let caption = caption.replace("• ", "").replace("    ", "");
    let caption = Regex::new(r"\n{2,}")?.replace_all(&caption, "\n");
    let leading = Regex::new(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")?;
    let lyrics = lyrics
        .split('\n')
        .map(|line| {
            leading
                .captures(line)
                .map_or_else(|| line.to_string(), |c| c[1].trim().to_string())
        })
        .collect::<Vec<_>>()
        .join("\n");
    let lyrics = lyrics
        .replace("] ", "]\n")
        .replace(" [", "\n[")
        .replace(" ^ ", "\n");
    let lyrics = Regex::new(r"\[([^\]]+)\]")?
        .replace_all(&lyrics, |c: &Captures<'_>| {
            format!("[{}]", c[1].to_lowercase())
        })
        .to_string();
    Ok(format!(
        "<|im_start|><|caption_start|>{caption}<|caption_end|><|lyrics_start|>[start]\n{lyrics}<|lyrics_end|><|im_end|><|audio_start|>"
    ))
}

pub(crate) fn tokenize(
    tokenizer: &Tokenizer,
    caption: &str,
    lyrics: &str,
) -> Result<(Vec<u32>, usize)> {
    for (name, id) in [("<|audio_end|>", 151670), ("<|audio_cfg|>", 151654)] {
        anyhow::ensure!(
            tokenizer.token_to_id(name) == Some(id),
            "Music3 tokenizer has unexpected {name} token id"
        );
    }
    let encoded = tokenizer
        .encode(format_prompt(caption, lyrics)?, false)
        .map_err(|error| anyhow::anyhow!("Music3 tokenization failed: {error}"))?;
    let mut ids = encoded.get_ids().to_vec();
    let length = ids.len();
    anyhow::ensure!(
        (3..=5000).contains(&length),
        "Music3 prompt must contain 3..=5000 tokens"
    );
    let mut unconditional = ids.clone();
    unconditional
        .get_mut(1..length - 2)
        .context("invalid prompt structure")?
        .fill(151654);
    ids.extend(unconditional);
    Ok((ids, length))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caption_and_lyrics_match_checkpoint_template() {
        assert_eq!(
            format_prompt(
                "# **Warm** pop\n- <|bpm 90|>",
                "[Verse] discard\nHello ^ world\n[Chorus]"
            )
            .unwrap(),
            "<|im_start|><|caption_start|>Warm pop\nbpm is 90<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello\nworld\n[chorus]<|lyrics_end|><|im_end|><|audio_start|>"
        );
        assert!(format_prompt("", "lyrics").is_err());
    }
}
