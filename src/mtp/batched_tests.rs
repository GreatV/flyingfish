use super::*;
use crate::mtp::{MtpHiddenTap, MtpRecurrentTap};

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    position: usize,
    pending: u32,
    accumulator: u64,
    target_kv: Vec<u32>,
    mtp_kv: Vec<u32>,
    hidden: u32,
    partitions: Vec<(usize, usize)>,
}
struct Mock {
    state: State,
    saved: Option<State>,
    proposals: Vec<u32>,
    hidden_rows: Vec<u32>,
    events: Vec<&'static str>,
    replay_bias: u32,
    poisoned: bool,
    restore_fails: bool,
}
impl Mock {
    fn new(proposals: Vec<u32>) -> Self {
        Self {
            state: State {
                position: 1,
                pending: 10,
                accumulator: 17,
                target_kv: vec![77; 32],
                mtp_kv: vec![88; 32],
                hidden: 5,
                partitions: vec![],
            },
            saved: None,
            proposals,
            hidden_rows: vec![],
            events: vec![],
            replay_bias: 0,
            poisoned: false,
            restore_fails: false,
        }
    }
    fn append(&mut self, token: u32, phase: usize) {
        self.state.accumulator = self
            .state
            .accumulator
            .wrapping_mul(71)
            .wrapping_add(token as u64 + phase as u64);
        self.state.hidden = token ^ self.state.position as u32;
        self.state.target_kv[self.state.position] = token;
        self.state.position += 1;
    }
}
impl Backend for Mock {
    fn position(&self) -> usize {
        self.state.position
    }
    fn capacity(&self) -> usize {
        31
    }
    fn pending(&self) -> Result<u32> {
        Ok(self.state.pending)
    }
    fn save(&mut self, _: usize) -> Result<()> {
        self.events.push("save");
        self.saved = Some(self.state.clone());
        Ok(())
    }
    fn restore(&mut self) -> Result<()> {
        self.events.push("restore_all");
        ensure!(!self.restore_fails, "restore failure");
        self.state = self.saved.clone().unwrap();
        Ok(())
    }
    fn poison(&mut self) {
        self.poisoned = true;
    }
    fn draft(&mut self, input: u32, depth: usize) -> Result<u32> {
        self.events.push("draft");
        self.state.mtp_kv[self.state.position - 1 + depth] = input.wrapping_mul(33);
        Ok(self.proposals[depth])
    }
    fn target_decode(&mut self, input: u32) -> Result<u32> {
        self.events.push("decode");
        self.state.partitions.push((self.state.position, 0));
        self.append(input, 0);
        self.state.pending = input + 1;
        Ok(input + 1)
    }
    fn canonical_row(&mut self, input: u32) -> Result<()> {
        self.events.push("canonical_row");
        self.state.mtp_kv[self.state.position - 1] = self.state.hidden ^ input;
        Ok(())
    }
}
impl BatchedBackend for Mock {
    fn verify_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>> {
        self.events.push("batch");
        self.state
            .partitions
            .push((self.state.position, inputs.len()));
        self.hidden_rows.clear();
        for &token in inputs {
            self.append(token, inputs.len());
            self.hidden_rows.push(self.state.hidden);
        }
        let mut predictions: Vec<_> = inputs.iter().map(|t| t + 1).collect();
        if inputs.len() < 4 {
            *predictions.last_mut().unwrap() += self.replay_bias;
        }
        self.state.pending = *predictions.last().unwrap();
        Ok(predictions)
    }
    fn restore_mtp(&mut self) -> Result<()> {
        self.events.push("restore_mtp");
        self.state.mtp_kv = self.saved.as_ref().unwrap().mtp_kv.clone();
        Ok(())
    }
    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()> {
        self.events.push("repair");
        let before = self.saved.as_ref().unwrap();
        for (row, &token) in inputs.iter().enumerate() {
            let hidden = if row == 0 {
                before.hidden
            } else {
                self.hidden_rows[row - 1]
            };
            self.state.mtp_kv[before.position - 1 + row] = hidden ^ token;
        }
        Ok(())
    }
}
fn options(k: usize) -> MtpOptions {
    MtpOptions {
        depth: k,
        hidden_tap: MtpHiddenTap::PreFinalNorm,
        recurrent_tap: MtpRecurrentTap::PostMtpNorm,
        graph: true,
    }
}

#[test]
fn all_accept_keeps_one_batch_and_partial_accept_replays_one_batch() {
    for k in 1..=3 {
        for j in 0..=k {
            let mut proposals: Vec<_> = (11..11 + k as u32).collect();
            if j < k {
                proposals[j] = 999;
            }
            let mut b = Mock::new(proposals);
            let r = run(&mut b, options(k), 64, &[], None).unwrap();
            assert_eq!((r.round.accepted, r.kept_verify_state), (j, j == k));
            assert_eq!(
                b.events.iter().filter(|&&e| e == "batch").count(),
                if j == k { 1 } else { 2 }
            );
            assert_eq!(
                b.events.iter().filter(|&&e| e == "restore_all").count(),
                usize::from(j != k)
            );
            assert!(!b.events.contains(&"decode"));
            let mut canonical = Mock::new(vec![]);
            canonical.save(j + 1).unwrap();
            let inputs: Vec<_> = (10..11 + j as u32).collect();
            let expected = canonical.verify_batch(&inputs).unwrap();
            canonical.repair_mtp(&inputs).unwrap();
            assert_eq!(b.state, canonical.state);
            assert_eq!(r.round.tokens.last(), expected.last());
        }
    }
}

#[test]
fn partial_replay_uses_its_own_correction_and_same_phase_state() {
    let mut b = Mock::new(vec![11, 999, 13]);
    b.replay_bias = 7;
    let r = run(&mut b, options(3), 64, &[], None).unwrap();
    assert_eq!(r.round.tokens, [11, 19]);
    assert_ne!(r.round.tokens.last(), r.verify_predictions.get(1));
    let mut canonical = Mock::new(vec![]);
    canonical.replay_bias = 7;
    canonical.save(2).unwrap();
    canonical.verify_batch(&[10, 11]).unwrap();
    canonical.repair_mtp(&[10, 11]).unwrap();
    assert_eq!(b.state, canonical.state);
    let mut decode = Mock::new(vec![]);
    decode.target_decode(10).unwrap();
    decode.target_decode(11).unwrap();
    assert_ne!(b.state.accumulator, decode.state.accumulator);
}

#[test]
fn both_commit_paths_roll_back_on_failure_and_never_publish_early() {
    for partial in [false, true] {
        let mut points = vec![
            Fault::AfterDraft(0),
            Fault::AfterDraft(2),
            Fault::AfterVerify,
            Fault::AfterRepair,
            Fault::BeforePublish,
        ];
        if partial {
            points.extend([Fault::AfterTargetRestore, Fault::AfterReplay]);
        } else {
            points.push(Fault::AfterMtpRestore);
        }
        for fault in points {
            let mut b = Mock::new(if partial {
                vec![11, 999, 13]
            } else {
                vec![11, 12, 13]
            });
            let before = b.state.clone();
            assert!(run(&mut b, options(3), 64, &[], Some(fault)).is_err());
            assert_eq!(b.state, before);
            assert!(!b.poisoned);
        }
    }
    let mut b = Mock::new(vec![11, 12, 13]);
    b.restore_fails = true;
    assert!(run(&mut b, options(3), 64, &[], Some(Fault::AfterRepair)).is_err());
    assert!(b.poisoned);
}

#[test]
fn eos_and_tail_budget_do_not_consume_the_published_pending_token() {
    let mut b = Mock::new(vec![11, 12, 13]);
    let r = run(&mut b, options(3), 64, &[12], None).unwrap();
    assert_eq!(r.round.tokens, [11, 12]);
    assert_eq!(b.state.position, 3);
    assert_eq!(b.state.pending, 12);
    assert!(!r.kept_verify_state);
    let mut b = Mock::new(vec![]);
    let r = run(&mut b, options(3), 1, &[], None).unwrap();
    assert_eq!(r.round.tokens, [11]);
    assert_eq!(b.events.iter().filter(|&&e| e == "decode").count(), 1);
    assert!(!b.events.contains(&"batch"));
}

#[test]
fn effective_depth_clips_to_both_capacity_and_publication_budget() {
    for depth in 1..=3 {
        for remaining in 0..=5 {
            for room in 0..=4 {
                let mut b = Mock::new(vec![11, 12, 13]);
                b.state.position = 31 - room;
                let before = b.state.clone();
                let r = run(&mut b, options(depth), remaining, &[], None).unwrap();
                assert!(r.round.consumed <= room.min(remaining));
                assert_eq!(r.round.tokens.len(), r.round.consumed);
                assert_eq!(b.state.position, before.position + r.round.consumed);
                if room == 0 || remaining == 0 {
                    assert_eq!(b.state, before);
                    assert!(b.events.is_empty());
                } else {
                    let effective = depth.min(remaining - 1).min(room - 1);
                    assert_eq!(r.round.proposals.len(), effective);
                    assert_eq!(r.kept_verify_state, effective > 0);
                }
            }
        }
    }
}
