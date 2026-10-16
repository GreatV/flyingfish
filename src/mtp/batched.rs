//! Batched verification commits a retained recurrent prefix without target replay.
use super::{
    MtpOptions,
    transaction::{self, Backend, Round},
};
use anyhow::{Result, ensure};

/// Verification retains each row's recurrent state and hidden rows for MTP repair.
pub trait BatchedBackend: Backend {
    fn draft_verify(
        &mut self,
        anchor: u32,
        depth: usize,
        #[cfg(test)] fault: Option<Fault>,
    ) -> Result<(Vec<u32>, Vec<u32>)>;
    /// Select the committed target prefix and its pending prediction.
    fn select_prefix(&mut self, rows: usize) -> Result<()>;
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
    AfterPrefixSelect,
    AfterRepair,
    BeforePublish,
}

#[derive(Debug, PartialEq, Eq)]
pub struct BatchRound {
    pub round: Round,
    pub kept_verify_state: bool,
    pub verify_predictions: Vec<u32>,
}

#[cfg(test)]
pub(crate) fn inject(fault: Option<Fault>, point: Fault) -> Result<()> {
    ensure!(
        fault != Some(point),
        "injected batched MTP failure at {point:?}"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn snapshot_row(
    rows: usize,
    live: usize,
    base: usize,
    position: usize,
    epoch: Option<u64>,
    expected_epoch: u64,
) -> Result<usize> {
    ensure!(
        (1..=5).contains(&live)
            && rows > 0
            && rows < live
            && base.checked_add(live) == Some(position)
            && epoch == Some(expected_epoch),
        "stale or invalid verify snapshot extent/epoch"
    );
    Ok(rows - 1)
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
        });
    }
    if let Err(e) = b.save(k + 1) {
        b.poison();
        return Err(e.context("M2b checkpoint save failed"));
    }
    let result = (|| -> Result<BatchRound> {
        let (proposals, verify) = b.draft_verify(
            anchor,
            k,
            #[cfg(test)]
            fault,
        )?;
        ensure!(proposals.len() == k, "draft prediction count mismatch");
        let inputs: Vec<_> = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect();
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
        let pending = verify[accepted];
        if keep {
            b.restore_mtp()?;
            #[cfg(test)]
            inject(fault, Fault::AfterMtpRestore)?;
        } else {
            b.select_prefix(committed.len())?;
            #[cfg(test)]
            inject(fault, Fault::AfterPrefixSelect)?;
            b.restore_mtp()?;
            #[cfg(test)]
            inject(fault, Fault::AfterMtpRestore)?;
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
        })
    })();
    match result {
        Ok(r) => Ok(r),
        Err(e) => {
            if let Err(restore) = b.restore(false) {
                b.poison();
                return Err(restore.context(format!("M2b rollback failed after: {e:#}")));
            }
            b.poison();
            Err(e)
        }
    }
}

#[cfg(test)]
#[path = "batched_tests.rs"]
mod tests;
