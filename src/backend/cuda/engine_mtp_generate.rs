//! Production request entry point; numerical work stays in the MTP controllers.
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
        let mtp = MtpOptions {
            depth,
            hidden_tap: MtpHiddenTap::PostFinalNorm,
            recurrent_tap: MtpRecurrentTap::PostMtpNorm,
            graph,
        };
        mtp.validate()?;
        ensure!(
            !ids.is_empty()
                && options.count > 0
                && ids
                    .len()
                    .checked_add(options.count)
                    .is_some_and(|n| n <= options.capacity),
            "MTP request needs a nonempty prompt, positive output count and a reserved next-token slot"
        );
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
        ensure!(
            ids.iter().all(|&t| t < spec.vocab),
            "MTP prompt token exceeds vocabulary"
        );
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
        engine.mtp_enable_for_prompt(mtp, ids.len())?;
        engine.mtp_batch_enable()?;
        if graph {
            engine.capture()?;
        }
        for rows in ids
            .chunks(engine.chunk)
            .map(<[u32]>::len)
            .collect::<std::collections::BTreeSet<_>>()
        {
            engine.prepare(rows)?;
        }
        engine.closure.ready(load_start)?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        if options.ignore_eos {
            engine.model_spec.eos.clear();
        }
        let origin = Instant::now();
        let first = engine.mtp_prefill(ids)?;
        let first_ns = origin.elapsed().as_nanos();
        let mut last_ns = first_ns;
        let mut generated = Vec::with_capacity(options.count);
        generated.push(first);
        while generated.len() < options.count
            && !engine
                .model_spec
                .eos
                .contains(generated.last().context("MTP output missing")?)
        {
            let remaining = options.count - generated.len();
            let round = match policy {
                #[cfg(test)]
                VerifyPolicy::Decode => engine.mtp_round(
                    remaining,
                    #[cfg(test)]
                    None,
                )?,
                VerifyPolicy::Batched => {
                    engine
                        .mtp_batch_round(
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
        }
        Ok(Output {
            ids: generated,
            load_ms,
            prefill_ms: first_ns as f64 / 1e6,
            decode_ms: (last_ns - first_ns) as f64 / 1e6,
        })
    }
}
