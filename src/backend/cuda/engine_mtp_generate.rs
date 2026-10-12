//! Shared MTP setup and generation; resident serving binds all tails before Ready.
use super::*;
use crate::mtp::{MtpHiddenTap, MtpOptions, MtpRecurrentTap, Output, VerifyPolicy};

impl Engine {
    pub(crate) fn generate_mtp(
        dir: &Path,
        ids: &[u32],
        options: &crate::spec::Options,
        depth: usize,
        policy: VerifyPolicy,
        graph: bool,
    ) -> Result<Output> {
        crate::mtp::check_request(ids.len(), options.count, options.capacity)?;
        let (mut engine, load_ms) = Self::load_mtp_inner(dir, options, depth, graph, Some(ids))?;
        let mut output = engine.generate_mtp_prepared(
            ids,
            options.count,
            options.ignore_eos,
            policy,
            graph,
            &mut |_, _| true,
        )?;
        output.load_ms = load_ms;
        Ok(output)
    }

    pub(crate) fn load_mtp(
        dir: &Path,
        options: &crate::spec::Options,
        depth: usize,
        graph: bool,
    ) -> Result<Self> {
        Self::load_mtp_inner(dir, options, depth, graph, None).map(|(engine, _)| engine)
    }

    fn load_mtp_inner(
        dir: &Path,
        options: &crate::spec::Options,
        depth: usize,
        graph: bool,
        prompt: Option<&[u32]>,
    ) -> Result<(Self, f64)> {
        let mtp = MtpOptions {
            depth,
            hidden_tap: MtpHiddenTap::PostFinalNorm,
            recurrent_tap: MtpRecurrentTap::PostMtpNorm,
            graph,
        };
        mtp.validate()?;
        ensure!(
            options.backend.draft_model.is_none(),
            "MTP cannot use a separate draft model"
        );
        ensure!(
            options.chunk.is_none_or(|n| n >= 4),
            "MTP prefill chunk must be at least four"
        );
        let spec = crate::model_spec::ModelSpec::read(dir)?;
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen35Text,
            "MTP generation is supported only for Qwen3.8 text"
        );
        if let Some(ids) = prompt {
            ensure!(
                ids.iter().all(|&t| t < spec.vocab),
                "MTP prompt token exceeds vocabulary"
            );
        }
        let prompt = prompt.map(<[u32]>::len);
        let load_start = Instant::now();
        let mut settings = options.backend.clone();
        settings.spec_budget = SpecBudget::Chain;
        settings.spec_graph = false;
        let mut engine = Self::load_kind_for(
            dir,
            options.device,
            options.capacity,
            options.chunk,
            settings,
            SetupMode::Mtp,
        )?;
        if let Some(prompt) = prompt {
            engine.mtp_enable_for_prompt(mtp, prompt)?;
        } else {
            engine.mtp_enable_resident(mtp)?;
        }
        engine.mtp_batch_enable()?;
        if graph {
            engine.capture()?;
        }
        let rows: std::collections::BTreeSet<_> = if let Some(prompt) = prompt {
            let mut rows = std::collections::BTreeSet::from([prompt.min(engine.chunk)]);
            if prompt > engine.chunk && prompt % engine.chunk != 0 {
                rows.insert(prompt % engine.chunk);
            }
            rows
        } else {
            (1..=engine.chunk).collect()
        };
        for rows in rows {
            engine.prepare(rows)?;
        }
        engine.closure.ready(load_start)?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        Ok((engine, load_ms))
    }

    pub(crate) fn generate_mtp_prepared(
        &mut self,
        ids: &[u32],
        count: usize,
        ignore_eos: bool,
        policy: VerifyPolicy,
        graph: bool,
        on_tokens: &mut dyn FnMut(&[u32], usize) -> bool,
    ) -> Result<Output> {
        crate::mtp::check_request(ids.len(), count, self.capacity)?;
        ensure!(
            ids.iter().all(|&t| t < self.model_spec.vocab),
            "MTP prompt token exceeds vocabulary"
        );
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "engine state is poisoned; reset/reload is required"
        );
        self.check_mtp_mode(graph)?;
        if graph {
            self.check_ready()?;
        }
        let eos = self.model_spec.eos.clone();
        if ignore_eos {
            self.model_spec.eos.clear();
        }
        let result = (|| -> Result<Output> {
            let origin = Instant::now();
            let first = self.mtp_prefill(ids)?;
            let first_ns = origin.elapsed().as_nanos();
            let mut last_ns = first_ns;
            let mut generated = Vec::with_capacity(count);
            generated.push(first);
            let mut running = on_tokens(&[first], 0);
            while running
                && generated.len() < count
                && !self
                    .model_spec
                    .eos
                    .contains(generated.last().context("MTP output missing")?)
            {
                let remaining = count - generated.len();
                let round = match policy {
                    #[cfg(test)]
                    VerifyPolicy::Decode => self.mtp_round(
                        remaining,
                        #[cfg(test)]
                        None,
                    )?,
                    VerifyPolicy::Batched => {
                        self.mtp_batch_round(
                            remaining,
                            #[cfg(test)]
                            None,
                        )?
                        .round
                    }
                };
                ensure!(
                    !round.tokens.is_empty() && round.tokens.len() <= remaining,
                    "MTP publication exceeds request budget"
                );
                last_ns = origin.elapsed().as_nanos();
                generated.extend_from_slice(&round.tokens);
                running = on_tokens(&round.tokens, round.accepted);
            }
            Ok(Output {
                ids: generated,
                load_ms: 0.0,
                prefill_ms: first_ns as f64 / 1e6,
                decode_ms: (last_ns - first_ns) as f64 / 1e6,
            })
        })();
        self.model_spec.eos = eos;
        result
    }
}
