use super::*;
use crate::mtp::{MtpHiddenTap, MtpRecurrentTap};

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    position: usize,
    pending: u32,
    accumulator: u64,
    history: [u32; 3],
    visible: usize,
    epoch: u64,
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
    snapshots: Vec<State>,
    snapshot_epoch: Option<u64>,
    mutant: Option<usize>,
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
                history: [91, 92, 93],
                visible: 1,
                epoch: 7,
                target_kv: vec![77; 32],
                mtp_kv: vec![88; 32],
                hidden: 5,
                partitions: vec![],
            },
            saved: None,
            proposals,
            hidden_rows: vec![],
            events: vec![],
            snapshots: vec![],
            snapshot_epoch: None,
            mutant: None,
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
        self.state.visible = self.state.position;
        self.state.history.rotate_left(1);
        self.state.history[2] = token;
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
    fn restore(&mut self, _replay: bool) -> Result<()> {
        self.events.push("restore_all");
        ensure!(!self.restore_fails, "restore failure");
        self.state = self.saved.clone().unwrap();
        self.snapshot_epoch = None;
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

impl Mock {
    fn verify_batch(&mut self, inputs: &[u32]) -> Result<Vec<u32>> {
        self.events.push("batch");
        self.state
            .partitions
            .push((self.state.position, inputs.len()));
        self.hidden_rows.clear();
        self.snapshots.clear();
        self.snapshot_epoch = Some(self.state.epoch);
        for &token in inputs {
            self.append(token, inputs.len());
            self.hidden_rows.push(self.state.hidden);
            self.state.pending = token + 1;
            self.snapshots.push(self.state.clone());
        }
        Ok(inputs.iter().map(|t| t + 1).collect())
    }
}

impl BatchedBackend for Mock {
    fn draft_verify(
        &mut self,
        anchor: u32,
        depth: usize,
        fault: Option<Fault>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut proposals = Vec::new();
        let mut token = anchor;
        for i in 0..depth {
            token = self.draft(token, i)?;
            proposals.push(token);
            inject(fault, Fault::AfterDraft(i))?;
        }
        let inputs: Vec<_> = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect();
        let verify = self.verify_batch(&inputs)?;
        Ok((proposals, verify))
    }
    fn select_prefix(&mut self, rows: usize) -> Result<()> {
        self.events.push("select_prefix");
        let mut row = snapshot_row(
            rows,
            self.snapshots.len(),
            self.saved.as_ref().map_or(1, |s| s.position),
            self.state.position,
            self.snapshot_epoch,
            self.state.epoch,
        )?;
        if self.mutant == Some(0) {
            row += 1;
        }
        if self.mutant == Some(1) {
            row = row.saturating_sub(1);
        }
        let tail = self.state.target_kv.clone();
        let visible = self.state.visible;
        self.state = self.snapshots[row].clone();
        self.state.target_kv = tail;
        if self.mutant == Some(2) {
            self.state.history = [self.state.hidden; 3];
        }
        if self.mutant == Some(3) {
            self.state.visible = visible;
        }
        if self.mutant != Some(4) {
            self.hidden_rows.truncate(rows);
        }
        Ok(())
    }
    fn restore_mtp(&mut self) -> Result<()> {
        self.events.push("restore_mtp");
        self.state.mtp_kv = self.saved.as_ref().unwrap().mtp_kv.clone();
        Ok(())
    }
    fn repair_mtp(&mut self, inputs: &[u32]) -> Result<()> {
        self.events.push("repair");
        ensure!(
            self.hidden_rows.len() == inputs.len(),
            "untruncated hidden frame"
        );
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

// The oracle consumes the full verify shape but commits only the causal prefix.
fn oracle(proposals: &[u32], accepted: usize) -> State {
    let mut expected = Mock::new(vec![]);
    expected.save(proposals.len() + 1).unwrap();
    expected.state.partitions.push((1, proposals.len() + 1));
    let inputs: Vec<_> = std::iter::once(10)
        .chain(proposals.iter().copied())
        .collect();
    for (row, &token) in inputs.iter().enumerate() {
        if row <= accepted {
            expected.append(token, inputs.len());
            expected.hidden_rows.push(expected.state.hidden);
        } else {
            expected.state.target_kv[1 + row] = token;
        }
    }
    expected.state.pending = inputs[accepted] + 1;
    expected.repair_mtp(&inputs[..accepted + 1]).unwrap();
    expected.state
}

#[test]
fn every_acceptance_branch_selects_the_full_verify_prefix_without_replay() {
    for k in 1..=3 {
        for accepted in 0..=k {
            let mut proposals: Vec<_> = (11..11 + k as u32).collect();
            if accepted < k {
                proposals[accepted] = 999;
            }
            let expected = oracle(&proposals, accepted);
            let mut b = Mock::new(proposals);
            let r = run(&mut b, options(k), 64, &[], None).unwrap();
            assert_eq!(
                (r.round.accepted, r.kept_verify_state),
                (accepted, accepted == k)
            );
            assert_eq!(b.events.iter().filter(|&&e| e == "batch").count(), 1);
            assert!(!b.events.contains(&"restore_all"));
            assert!(!b.events.contains(&"decode"));
            assert_eq!(b.state, expected);
            assert_eq!(r.round.tokens.last(), r.verify_predictions.get(accepted));
        }
    }
}

#[test]
fn state_oracle_detects_wrong_row_history_visibility_frame_and_stale_epoch() {
    for (rows, live, base, position) in [
        (0, 4, 1, 5),
        (4, 4, 1, 5),
        (1, 0, 1, 1),
        (1, 5, 1, 6),
        (1, 4, usize::MAX, 3),
        (1, 4, 1, 6),
    ] {
        assert!(snapshot_row(rows, live, base, position, Some(7), 7).is_err());
    }
    for mutant in 0..5 {
        let proposals = vec![11, 12, 999];
        let expected = oracle(&proposals, 2);
        let mut b = Mock::new(proposals);
        b.mutant = Some(mutant);
        let result = run(&mut b, options(3), 64, &[], None);
        assert!(
            result.is_err() || b.state != expected,
            "mutant {mutant} escaped"
        );
    }
    let mut b = Mock::new(vec![]);
    b.verify_batch(&[10, 11, 12, 13]).unwrap();
    b.state.epoch += 1;
    assert!(b.select_prefix(2).is_err());
    b.state.epoch -= 1;
    b.restore_fails = false;
    b.save(4).unwrap();
    b.restore(false).unwrap();
    assert!(b.select_prefix(2).is_err());
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
            points.extend([Fault::AfterPrefixSelect, Fault::AfterMtpRestore]);
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
            assert!(b.poisoned);
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
