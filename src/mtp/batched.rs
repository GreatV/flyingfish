//! M2b policy: retain full verification state, replay partial prefixes once.
use super::{
    MtpOptions,
    transaction::{self, Backend, Round},
};
use anyhow::{Result, ensure};

/// Batch calls use an explicitly selected prefill phase, including one-row
/// partial replays. The backend retains their hidden rows for MTP repair.
pub trait BatchedBackend: Backend {
    fn verify_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>>;
    /// Preserve the verified target state and its hidden-row buffer.
    fn restore_mtp(&mut self) -> Result<()>;
    /// Rebuild canonical MTP rows from the saved boundary and latest target batch.
    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()>;
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    AfterDraft(usize),
    AfterVerify,
    AfterMtpRestore,
    AfterTargetRestore,
    AfterReplay,
    AfterRepair,
    BeforePublish,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BatchRound {
    pub round: Round,
    pub kept_verify_state: bool,
    pub verify_predictions: Vec<u32>,
    pub replay_predictions: Option<Vec<u32>>,
}

#[cfg(test)]
fn inject(fault: Option<Fault>, point: Fault) -> Result<()> {
    ensure!(
        fault != Some(point),
        "injected batched MTP failure at {point:?}"
    );
    Ok(())
}

pub fn run<B: BatchedBackend>(
    b: &mut B,
    options: MtpOptions,
    remaining: usize,
    eos: &[u32],
    #[cfg(test)] fault: Option<Fault>,
) -> Result<BatchRound> {
    options.validate()?;
    let position = b.position();
    ensure!(position <= b.capacity(), "MTP position exceeds capacity");
    let anchor = b.pending()?;
    if remaining == 0 || position == b.capacity() || eos.contains(&anchor) {
        return Ok(BatchRound {
            round: Round::default(),
            kept_verify_state: false,
            verify_predictions: vec![],
            replay_predictions: None,
        });
    }
    let k = options
        .depth
        .min(remaining - 1)
        .min(b.capacity() - position - 1);
    if k == 0 {
        #[cfg(test)]
        let fault = match fault {
            Some(Fault::BeforePublish) => Some(transaction::Fault::BeforePublish),
            Some(Fault::AfterRepair) => Some(transaction::Fault::AfterCanonical(0)),
            Some(Fault::AfterReplay) => Some(transaction::Fault::AfterReplay(0)),
            _ => None,
        };
        let round = transaction::run(
            b,
            options,
            remaining,
            eos,
            #[cfg(test)]
            fault,
        )?;
        return Ok(BatchRound {
            round,
            kept_verify_state: false,
            verify_predictions: vec![],
            replay_predictions: None,
        });
    }
    if let Err(e) = b.save(k + 1) {
        b.poison();
        return Err(e.context("M2b checkpoint save failed"));
    }
    let result = (|| -> Result<BatchRound> {
        let mut proposals = Vec::with_capacity(k);
        let mut token = anchor;
        for depth in 0..k {
            token = b.draft(token, depth)?;
            proposals.push(token);
            #[cfg(test)]
            inject(fault, Fault::AfterDraft(depth))?;
        }
        let inputs: Vec<_> = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect();
        let verify = b.verify_batch(&inputs)?;
        ensure!(
            verify.len() == inputs.len(),
            "batch verifier prediction count mismatch"
        );
        #[cfg(test)]
        inject(fault, Fault::AfterVerify)?;
        let matched = proposals
            .iter()
            .zip(&verify)
            .take_while(|(a, b)| a == b)
            .count();
        let accepted = proposals[..matched]
            .iter()
            .position(|t| eos.contains(t))
            .unwrap_or(matched);
        let keep = accepted == k;
        let committed = &inputs[..accepted + 1];
        let replay;
        let pending;
        if keep {
            b.restore_mtp()?;
            #[cfg(test)]
            inject(fault, Fault::AfterMtpRestore)?;
            replay = None;
            pending = verify[k];
        } else {
            b.restore()?;
            #[cfg(test)]
            inject(fault, Fault::AfterTargetRestore)?;
            let predictions = b.verify_batch(committed)?;
            ensure!(
                predictions.len() == committed.len(),
                "batch replay prediction count mismatch"
            );
            pending = predictions[accepted];
            replay = Some(predictions);
            #[cfg(test)]
            inject(fault, Fault::AfterReplay)?;
        }
        b.repair_mtp(committed)?;
        #[cfg(test)]
        inject(fault, Fault::AfterRepair)?;
        ensure!(
            b.position() == position + committed.len() && b.pending()? == pending,
            "M2b committed metadata/pending mismatch"
        );
        let mut tokens = proposals[..accepted].to_vec();
        tokens.push(pending);
        ensure!(tokens.len() <= remaining, "M2b output exceeds budget");
        #[cfg(test)]
        inject(fault, Fault::BeforePublish)?;
        Ok(BatchRound {
            round: Round {
                tokens,
                proposals,
                matched,
                accepted,
                consumed: accepted + 1,
            },
            kept_verify_state: keep,
            verify_predictions: verify,
            replay_predictions: replay,
        })
    })();
    match result {
        Ok(r) => Ok(r),
        Err(e) => {
            if let Err(restore) = b.restore() {
                b.poison();
                return Err(restore.context(format!("M2b rollback failed after: {e:#}")));
            }
            Err(e)
        }
    }
}

#[cfg(test)]
#[path = "batched_tests.rs"]
mod tests;
