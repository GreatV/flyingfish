use super::{
    Args,
    http::{Error, Generator, Input},
};
use anyhow::{Result, ensure};
use flyingfish::qwen35::{
    config::{Qwen35Config, chat_prompt},
    gpu::QwenGpu,
    weights::Qwen35Weights,
};
use serde_json::{Value, json};
use std::time::Instant;
use tokenizers::Tokenizer;

pub(super) struct Engine {
    gpu: QwenGpu,
    tokenizer: Tokenizer,
    config: Qwen35Config,
    max_context: usize,
}

impl Engine {
    pub fn load(args: Args, devices: Vec<usize>) -> Result<Self> {
        let started = Instant::now();
        let path = args.model.canonicalize()?;
        eprintln!("model: {}", path.display());
        eprintln!("devices: {devices:?}");
        let config = Qwen35Config::from_model_dir(&path)?;
        ensure!(
            args.max_context.get() <= config.text_config.max_position_embeddings,
            "--max-context exceeds model context"
        );
        let tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let weights = Qwen35Weights::open(&path)?;
        let mut gpu = QwenGpu::with_max_ctx(
            &devices,
            &weights,
            &config,
            args.max_context.get(),
            ff_qwen35::gpu::force_stream_requested(),
        )?;
        let (records, _) = flyingfish::host_profile::HostProfile::group_records(
            args.host_profile.as_deref(),
            devices[0],
            gpu.forced_group() || weights.format().is_16bit(),
        )?;
        gpu.bind_groups(&records, &flyingfish::collect_binary_identity()?)?;
        eprintln!(
            "model loaded in {:.3} s; prefill {:?}",
            started.elapsed().as_secs_f64(),
            gpu.prefill_modes()
        );
        Ok(Self {
            gpu,
            tokenizer,
            config,
            max_context: args.max_context.get(),
        })
    }

    fn infer(&mut self, ids: &[u32], max_tokens: usize, started: Instant) -> Result<Value> {
        let reset_start = Instant::now();
        self.gpu.reset()?;
        let prefill_start = Instant::now();
        self.gpu.push_tokens(ids)?;
        let first = self.gpu.read_token()?;
        let decode_start = Instant::now();
        let output = self.gpu.decode(first, max_tokens)?;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let text = self
            .tokenizer
            .decode(&output, true)
            .map_err(anyhow::Error::msg)?;
        let stopped = output
            .last()
            .is_some_and(|id| self.config.text_config.eos_token_id.contains(id));
        Ok(json!({
            "text": text,
            "generated_ids": output,
            "prompt_tokens": ids.len(),
            "completion_tokens": output.len(),
            "finish_reason": if stopped { "stop" } else { "length" },
            "prefill_device_modes": self.gpu.prefill_modes(),
            "timings": {
                "reset_ms": (prefill_start - reset_start).as_secs_f64() * 1000.0,
                "prefill_ms": (decode_start - prefill_start).as_secs_f64() * 1000.0,
                "decode_ms": decode_ms,
                "decode_tokens": output.len().saturating_sub(1),
                "total_ms": started.elapsed().as_secs_f64() * 1000.0
            }
        }))
    }
}

impl Generator for Engine {
    fn generate(&mut self, input: Input) -> std::result::Result<Value, Error> {
        let started = Instant::now();
        let ids = self
            .tokenizer
            .encode(chat_prompt(&input.prompt), false)
            .map_err(Error::invalid)?
            .get_ids()
            .to_vec();
        if ids.is_empty()
            || ids
                .len()
                .checked_add(input.max_new_tokens.get())
                .is_none_or(|n| n > self.max_context)
        {
            return Err(Error::invalid(format!(
                "prompt plus requested output must fit {} tokens",
                self.max_context
            )));
        }
        self.infer(&ids, input.max_new_tokens.get(), started)
            .map_err(|error| {
                eprintln!("inference failed: {error:#}");
                Error::failed("inference failed; restart the server")
            })
    }
}
