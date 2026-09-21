use super::device_parse::parse_device;
use super::{DeviceCacheArgs, WeightCacheArgs, kit};
use anyhow::{Context, Result, bail};
use clap::Subcommand;
use flyingfish::minicpm::Config;
use session::{Worker, WorkerOptions};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod batch;
mod session;

#[derive(Debug, Subcommand)]
pub(super) enum MiniCpmCommand {
    #[command(about = "Run queued requests with one reusable model worker per device")]
    Batch(batch::Args),
    #[command(about = "Generate text with the MiniCPM5 target model using greedy decoding")]
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(
            long,
            help = "Paired MiniCPM5 DSpark checkpoint for greedy speculative decoding"
        )]
        draft_model: Option<PathBuf>,
        #[arg(
            long,
            conflicts_with = "prompt_file",
            required_unless_present = "prompt_file"
        )]
        prompt: Option<String>,
        #[arg(long, conflicts_with = "prompt")]
        prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = NonZeroUsize::new(128).unwrap())]
        max_new_tokens: NonZeroUsize,
        #[command(flatten)]
        device: kit::DeviceArgs,
        #[arg(long, default_value_t = 32)]
        attention_query_chunk_size: usize,
        #[arg(
            long,
            help = "Use consistent CUDA arithmetic for target-only and speculative decoding (experimental)"
        )]
        batch_invariant_decode: bool,
        #[arg(long, help = "Treat prompt as an already formatted completion prefix")]
        raw: bool,
        #[command(flatten)]
        weights: WeightCacheArgs,
        #[command(flatten)]
        device_cache: DeviceCacheArgs,
    },
}

pub(super) fn run(command: MiniCpmCommand) -> Result<()> {
    let command = match command {
        MiniCpmCommand::Batch(args) => return batch::run(args),
        generate => generate,
    };
    let MiniCpmCommand::Generate {
        model,
        draft_model,
        prompt,
        prompt_file,
        max_new_tokens,
        device,
        attention_query_chunk_size,
        batch_invariant_decode,
        raw,
        weights,
        device_cache,
    } = command
    else {
        unreachable!()
    };
    let kit::DeviceArgs { device } = device;
    let max_new_tokens = max_new_tokens.get();
    let prompt = match (prompt, prompt_file) {
        (Some(text), None) => text,
        (None, Some(path)) => std::fs::read_to_string(path).context("read prompt file")?,
        _ => bail!("supply --prompt or --prompt-file"),
    };
    let config = Config::read(&model)?;
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("load tokenizer: {error}"))?;
    let input = encode_prompt(&tokenizer, &config, &prompt, raw, max_new_tokens)?;
    let mut worker = Worker::open(
        WorkerOptions {
            model: &model,
            draft_model: draft_model.as_deref(),
            config,
            device: parse_device(&device)?,
            weights,
            device_cache,
            query_chunk: attention_query_chunk_size,
            batch_invariant: batch_invariant_decode,
        },
        &[flyingfish::minicpm::memory::RequestGeometry {
            prompt_tokens: input.len(),
            max_new_tokens,
            attention_query_chunk_size,
            batch_invariant_decode,
        }],
    )?;
    if batch_invariant_decode {
        eprintln!("decode arithmetic: fixed CUDA batch geometry");
    }
    let mut generated = Vec::new();
    let mut stream = tokenizer.decode_stream(true);
    let mut streamed = String::new();
    let mut stdout = std::io::stdout().lock();
    let mut emit = |next| -> Result<()> {
        generated.push(next);
        if let Some(text) = stream
            .step(next)
            .map_err(|error| anyhow::anyhow!("decode output: {error}"))?
        {
            write!(stdout, "{text}")?;
            stdout.flush()?;
            streamed.push_str(&text);
        }
        Ok(())
    };
    if let Some(stats) = worker.generate(&input, max_new_tokens, &mut emit)? {
        eprintln!(
            "DSpark: {} draft forwards, {} target verification forwards, {}/{} draft tokens accepted, {} rejected blocks",
            stats.draft_forwards,
            stats.target_verify_forwards,
            stats.accepted_draft_tokens,
            stats.proposed_tokens,
            stats.rejected_blocks
        );
        if stats.target_decode_forwards > 0 {
            eprintln!(
                "DSpark scheduling: {} direct target forwards, target-only fallback: {}",
                stats.target_decode_forwards, stats.fell_back_to_target
            );
        }
        if let (Some(target), Some(speculative)) = (
            stats.observed_target_ms_per_token,
            stats.observed_speculative_ms_per_token,
        ) {
            eprintln!(
                "DSpark observed compute: target {target:.2} ms/token, speculative {speculative:.2} ms/token"
            );
        }
    }
    write_decode_tail(&tokenizer, &generated, &streamed, &mut stdout)?;
    writeln!(stdout)?;
    eprintln!("generated {} tokens (greedy)", generated.len());
    Ok(())
}

/// Emit whatever the streaming decoder was still holding back.
///
/// `DecodeStream` withholds a piece whose bytes do not yet form a character, so
/// a generation that stops mid-sequence -- at the token limit, at EOS, or on a
/// truncated byte-fallback run -- leaves the last character unwritten. Decoding
/// the whole sequence resolves it the way the request's own token ids define
/// it: a run that never completes decodes to U+FFFD, which is a character the
/// caller is owed rather than one to hide.
fn write_decode_tail(
    tokenizer: &Tokenizer,
    generated: &[u32],
    streamed: &str,
    stdout: &mut impl Write,
) -> Result<()> {
    let full = tokenizer
        .decode(generated, true)
        .map_err(|error| anyhow::anyhow!("decode output: {error}"))?;
    // Each chunk a `DecodeStream` hands back is the next piece of the decode of
    // everything stepped so far, so what was streamed is a prefix of the whole.
    let tail = full
        .strip_prefix(streamed)
        .context("the complete decode does not extend the streamed text")?;
    if !tail.is_empty() {
        write!(stdout, "{tail}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn encode_prompt(
    tokenizer: &Tokenizer,
    config: &Config,
    prompt: &str,
    raw: bool,
    limit: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(limit > 0, "max_new_tokens must be positive");
    let text = if raw {
        prompt.to_owned()
    } else {
        let bos = tokenizer
            .id_to_token(config.bos_token_id)
            .context("tokenizer is missing BOS")?;
        format!(
            "{bos}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    };
    let input = tokenizer
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!("tokenize prompt: {error}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!input.is_empty(), "prompt tokenized to an empty sequence");
    anyhow::ensure!(
        input
            .len()
            .checked_add(limit)
            .is_some_and(|total| total <= config.max_position_embeddings),
        "prompt and requested output exceed model context"
    );
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{decoders::byte_fallback::ByteFallback, models::wordlevel::WordLevel};

    /// A byte-fallback vocabulary, which is the shape that makes a streaming
    /// decoder withhold: "你" is three tokens, and stopping inside it leaves
    /// bytes that are not yet a character.
    fn byte_fallback_tokenizer() -> Tokenizer {
        let vocab = ["<unk>", "A", "<0xE4>", "<0xBD>", "<0xA0>"]
            .into_iter()
            .enumerate()
            .map(|(index, token)| (token.to_owned(), index as u32))
            .collect();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(ByteFallback::new()));
        tokenizer
    }

    /// Everything the caller sees: what the stream emitted token by token, plus
    /// whatever the tail write adds once generation stops.
    fn streamed_then_finished(tokenizer: &Tokenizer, generated: &[u32]) -> String {
        let mut stream = tokenizer.decode_stream(true);
        let mut streamed = String::new();
        for &token in generated {
            if let Some(piece) = stream.step(token).unwrap() {
                streamed.push_str(&piece);
            }
        }
        let mut out = Vec::new();
        write_decode_tail(tokenizer, generated, &streamed, &mut out).unwrap();
        streamed + &String::from_utf8(out).unwrap()
    }

    #[test]
    fn generated_text_reads_back_the_tokens_it_was_generated_from() {
        let tokenizer = byte_fallback_tokenizer();
        for generated in [
            vec![],
            vec![1],          // plain text, nothing withheld
            vec![1, 2, 3, 4], // a character spanning three tokens, completed
            vec![1, 2],       // stopped one byte into that character
            vec![1, 2, 3],    // stopped two bytes in
            vec![2],          // the sequence never starts a character at all
        ] {
            assert_eq!(
                streamed_then_finished(&tokenizer, &generated),
                tokenizer.decode(&generated, true).unwrap(),
                "output disagrees with its own tokens: {generated:?}"
            );
        }
    }

    /// An unfinished byte run is U+FFFD, which the caller is owed: the token
    /// limit cutting a character short is a visible outcome, not a silent one.
    #[test]
    fn a_sequence_cut_mid_character_ends_in_a_replacement_character() {
        let tokenizer = byte_fallback_tokenizer();
        assert_eq!(streamed_then_finished(&tokenizer, &[1, 2]), "A\u{fffd}");
        assert_eq!(
            streamed_then_finished(&tokenizer, &[1, 2, 3, 4]),
            "A\u{4f60}"
        );
    }
}
