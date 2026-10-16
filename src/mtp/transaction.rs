//! M2a: verify with ordinary decode, restore, then replay before publication.
use super::MtpOptions;
use anyhow::{Result, ensure};

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    AfterDraft(usize),
    AfterVerify(usize),
    AfterRestore,
    AfterCanonical(usize),
    AfterReplay(usize),
    BeforePublish,
}

/// The Engine owns both state stores and the preallocated checkpoint. No token
/// escapes through this interface; only a completed round returns new tokens.
pub trait Backend {
    fn position(&self) -> usize;
    /// Number of positions that can be consumed while retaining legal next-step metadata.
    fn capacity(&self) -> usize;
    fn pending(&self) -> Result<u32>;
    fn save(&mut self, rows: usize) -> Result<()>;
    fn restore(&mut self, replay: bool) -> Result<()>;
    fn poison(&mut self);
    fn draft(&mut self, input: u32, depth: usize) -> Result<u32>;
    fn target_decode(&mut self, input: u32) -> Result<u32>;
    fn canonical_row(&mut self, input: u32) -> Result<()>;
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Round {
    pub tokens: Vec<u32>,
    pub proposals: Vec<u32>,
    pub matched: usize,
    pub accepted: usize,
    pub consumed: usize,
}

#[cfg(test)]
fn inject(fault: Option<Fault>, point: Fault) -> Result<()> {
    ensure!(fault != Some(point), "injected MTP failure at {point:?}");
    Ok(())
}

pub fn run<B: Backend>(
    backend: &mut B,
    options: MtpOptions,
    remaining: usize,
    eos: &[u32],
    #[cfg(test)] fault: Option<Fault>,
) -> Result<Round> {
    options.validate()?;
    let anchor = backend.pending()?;
    ensure!(
        backend.position() <= backend.capacity(),
        "MTP position exceeds capacity"
    );
    if remaining == 0 || eos.contains(&anchor) || backend.position() == backend.capacity() {
        return Ok(Round::default());
    }
    let available = backend.capacity() - backend.position();
    let k = options.depth.min(remaining - 1).min(available - 1);
    if let Err(error) = backend.save(k + 1) {
        backend.poison();
        return Err(error.context("MTP checkpoint save failed"));
    }
    let result = (|| -> Result<Round> {
        if k == 0 {
            backend.canonical_row(anchor)?;
            #[cfg(test)]
            inject(fault, Fault::AfterCanonical(0))?;
            let pending = backend.target_decode(anchor)?;
            #[cfg(test)]
            inject(fault, Fault::AfterReplay(0))?;
            #[cfg(test)]
            inject(fault, Fault::BeforePublish)?;
            return Ok(Round {
                tokens: vec![pending],
                consumed: 1,
                ..Round::default()
            });
        }
        let mut proposals = Vec::with_capacity(k);
        let mut input = anchor;
        for depth in 0..k {
            input = backend.draft(input, depth)?;
            proposals.push(input);
            #[cfg(test)]
            inject(fault, Fault::AfterDraft(depth))?;
        }
        let mut matched = 0;
        let mut matching = true;
        for (row, input) in std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .enumerate()
        {
            let expected = backend.target_decode(input)?;
            if row < k {
                matching &= expected == proposals[row];
                matched += usize::from(matching);
            }
            #[cfg(test)]
            inject(fault, Fault::AfterVerify(row))?;
        }
        let accepted = proposals[..matched]
            .iter()
            .position(|t| eos.contains(t))
            .unwrap_or(matched);
        backend.restore(true)?;
        #[cfg(test)]
        inject(fault, Fault::AfterRestore)?;
        let mut pending = anchor;
        for row in 0..=accepted {
            let input = if row == 0 { anchor } else { proposals[row - 1] };
            backend.canonical_row(input)?;
            #[cfg(test)]
            inject(fault, Fault::AfterCanonical(row))?;
            pending = backend.target_decode(input)?;
            if row < accepted {
                ensure!(
                    pending == proposals[row],
                    "M2a verify/replay greedy mismatch at row {row}"
                );
            }
            #[cfg(test)]
            inject(fault, Fault::AfterReplay(row))?;
        }
        let mut tokens = proposals[..accepted].to_vec();
        tokens.push(pending);
        ensure!(
            tokens.len() <= remaining && tokens.len() == accepted + 1,
            "MTP output accounting mismatch"
        );
        #[cfg(test)]
        inject(fault, Fault::BeforePublish)?;
        Ok(Round {
            tokens,
            proposals,
            matched,
            accepted,
            consumed: accepted + 1,
        })
    })();
    match result {
        Ok(round) => Ok(round),
        Err(error) => {
            if let Err(restore) = backend.restore(false) {
                backend.poison();
                return Err(restore.context(format!("MTP rollback failed after: {error:#}")));
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::{MtpHiddenTap as H2, MtpRecurrentTap as H5};
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct State {
        p: usize,
        pending: u32,
        s: u64,
        history: [u32; 3],
        kv: Vec<u32>,
        mtp: Vec<u32>,
        hidden: u32,
    }
    struct Mock {
        state: State,
        snapshot: Option<State>,
        draft: Vec<u32>,
        capacity: usize,
        poisoned: bool,
        restore_fails: bool,
        seal_locked: bool,
    }
    impl Mock {
        fn new(draft: Vec<u32>) -> Self {
            Self {
                state: State {
                    p: 1,
                    pending: 10,
                    s: 9,
                    history: [1, 2, 3],
                    kv: vec![77; 16],
                    mtp: vec![88; 16],
                    hidden: 5,
                },
                snapshot: None,
                draft,
                capacity: 16,
                poisoned: false,
                restore_fails: false,
                seal_locked: false,
            }
        }
        fn ordinary(&mut self, count: usize) {
            for _ in 0..count {
                let t = self.state.pending;
                self.canonical_row(t).unwrap();
                self.target_decode(t).unwrap();
            }
        }
    }
    impl Backend for Mock {
        fn position(&self) -> usize {
            self.state.p
        }
        fn capacity(&self) -> usize {
            self.capacity
        }
        fn pending(&self) -> Result<u32> {
            Ok(self.state.pending)
        }
        fn save(&mut self, rows: usize) -> Result<()> {
            ensure!(self.state.p + rows <= self.capacity, "capacity");
            self.snapshot = Some(self.state.clone());
            self.seal_locked = true;
            Ok(())
        }
        fn restore(&mut self, replay: bool) -> Result<()> {
            ensure!(!self.restore_fails, "restore failure");
            self.state = self.snapshot.clone().unwrap();
            self.seal_locked = replay;
            Ok(())
        }
        fn poison(&mut self) {
            self.poisoned = true;
        }
        fn draft(&mut self, input: u32, depth: usize) -> Result<u32> {
            self.state.mtp[self.state.p - 1 + depth] = input.wrapping_mul(37);
            Ok(self.draft[depth])
        }
        fn target_decode(&mut self, input: u32) -> Result<u32> {
            self.state.kv[self.state.p] = input;
            self.state.s = self.state.s.wrapping_mul(73).wrapping_add(input as u64);
            self.state.history.rotate_left(1);
            self.state.history[2] = input;
            self.state.hidden = input ^ self.state.p as u32;
            self.state.p += 1;
            self.state.pending = input + 1;
            Ok(input + 1)
        }
        fn canonical_row(&mut self, input: u32) -> Result<()> {
            self.state.mtp[self.state.p - 1] = self.state.hidden.wrapping_mul(19) ^ input;
            Ok(())
        }
    }
    fn options(depth: usize, h2: H2, h5: H5) -> MtpOptions {
        MtpOptions {
            depth,
            hidden_tap: h2,
            recurrent_tap: h5,
            graph: false,
        }
    }
    #[test]
    fn every_acceptance_length_commits_the_plain_decode_state() {
        for k in 1..=4 {
            for accepted in 0..=k {
                for h2 in [H2::PreFinalNorm, H2::PostFinalNorm] {
                    for h5 in [H5::PreMtpNorm, H5::PostMtpNorm] {
                        let mut draft: Vec<u32> = (11..11 + k as u32).collect();
                        if accepted < k {
                            draft[accepted] = 999;
                        }
                        let mut target = Mock::new(draft);
                        let mut ordinary = Mock::new(vec![]);
                        ordinary.ordinary(accepted + 1);
                        let round = run(&mut target, options(k, h2, h5), 64, &[], None).unwrap();
                        assert_eq!(round.accepted, accepted);
                        assert_eq!(round.tokens, (11..12 + accepted as u32).collect::<Vec<_>>());
                        assert_eq!(target.state, ordinary.state);
                        assert!(!target.poisoned);
                    }
                }
            }
        }
    }
    #[test]
    fn failures_roll_back_without_publication_and_failed_restore_poisons() {
        let opt = options(3, H2::PreFinalNorm, H5::PostMtpNorm);
        let mut points = vec![Fault::AfterRestore, Fault::BeforePublish];
        points.extend((0..3).map(Fault::AfterDraft));
        points.extend((0..4).map(Fault::AfterVerify));
        points.extend((0..4).map(Fault::AfterCanonical));
        points.extend((0..4).map(Fault::AfterReplay));
        for point in points {
            let mut b = Mock::new(vec![11, 12, 13]);
            let before = b.state.clone();
            assert!(run(&mut b, opt, 64, &[], Some(point)).is_err(), "{point:?}");
            assert_eq!(b.state, before, "{point:?}");
            assert!(
                !b.seal_locked,
                "rollback must leave sealing enabled: {point:?}"
            );
            assert!(!b.poisoned);
            let retry = run(&mut b, opt, 64, &[], None).unwrap();
            assert_eq!(retry.tokens, [11, 12, 13, 14]);
        }
        let mut b = Mock::new(vec![11, 12, 13]);
        b.restore_fails = true;
        assert!(run(&mut b, opt, 64, &[], Some(Fault::AfterDraft(0))).is_err());
        assert!(b.poisoned);
    }
    #[test]
    fn failed_single_row_round_unlocks_before_plain_decode() {
        let opt = options(3, H2::PreFinalNorm, H5::PostMtpNorm);
        for point in [
            Fault::AfterCanonical(0),
            Fault::AfterReplay(0),
            Fault::BeforePublish,
        ] {
            let mut b = Mock::new(vec![]);
            let mut plain = Mock::new(vec![]);
            assert!(run(&mut b, opt, 1, &[], Some(point)).is_err());
            assert!(!b.seal_locked);
            b.ordinary(8);
            plain.ordinary(8);
            assert_eq!(b.state, plain.state);
        }
    }
    #[test]
    fn eos_output_budget_and_capacity_do_not_consume_pending_eos() {
        let opt = options(3, H2::PostFinalNorm, H5::PreMtpNorm);
        for (remaining, eos, capacity, want) in [
            (64, vec![12], 16, vec![11, 12]),
            (1, vec![], 16, vec![11]),
            (64, vec![], 2, vec![11]),
            (0, vec![], 16, vec![]),
            (64, vec![10], 16, vec![]),
            (64, vec![], 1, vec![]),
        ] {
            let mut b = Mock::new(vec![11, 12, 13]);
            b.capacity = capacity;
            let r = run(&mut b, opt, remaining, &eos, None).unwrap();
            assert_eq!(r.tokens, want);
            let mut plain = Mock::new(vec![]);
            plain.ordinary(want.len());
            assert_eq!(b.state, plain.state);
        }
        let mut b = Mock::new(vec![]);
        assert!(run(&mut b, MtpOptions { depth: 5, ..opt }, 1, &[], None).is_err());
        assert!(b.snapshot.is_none());
    }
}
