use anyhow::{Result, ensure};
use std::{
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap, HashMap},
};

pub const MAX_ROWS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub token: u32,
    pub logp: f64,
}

#[derive(Clone, Debug)]
pub struct Distribution {
    choices: Vec<Choice>,
}

impl Distribution {
    pub fn logits(logits: &[f32], branches: usize) -> Result<Self> {
        ensure!(
            !logits.is_empty() && logits.len() <= u32::MAX as usize,
            "invalid vocabulary size"
        );
        ensure!(
            branches > 0 && branches <= logits.len(),
            "invalid branch count"
        );
        ensure!(
            logits
                .iter()
                .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
            "nonfinite draft logits"
        );
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        ensure!(max.is_finite(), "draft distribution has no finite logits");
        let total: f32 = logits.iter().map(|v| (*v - max).exp()).sum();
        ensure!(
            total.is_finite() && total > 0.0,
            "invalid draft normalization"
        );
        let logz = total.ln();
        let mut choices: Vec<Choice> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(token, value)| Choice {
                token: token as u32,
                logp: f64::from((*value - max) - logz),
            })
            .collect();
        choices.sort_by(|a, b| b.logp.total_cmp(&a.logp).then(a.token.cmp(&b.token)));
        choices.truncate(branches);
        Self::top(choices)
    }

    pub fn top(mut choices: Vec<Choice>) -> Result<Self> {
        ensure!(!choices.is_empty(), "draft top candidates are empty");
        ensure!(
            choices.iter().all(|c| c.logp.is_finite() && c.logp <= 0.0),
            "invalid full-vocabulary normalized log probability"
        );
        let mass: f64 = choices.iter().map(|c| c.logp.exp()).sum();
        ensure!(
            mass <= 1.000001,
            "draft top candidates exceed unit probability mass"
        );
        for (i, c) in choices.iter().enumerate() {
            ensure!(
                !choices[..i].iter().any(|v| v.token == c.token),
                "duplicate draft top candidate"
            );
        }
        choices.sort_by(|a, b| b.logp.total_cmp(&a.logp).then(a.token.cmp(&b.token)));
        Ok(Self { choices })
    }

    pub fn choices(&self) -> &[Choice] {
        &self.choices
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub token: u32,
    pub parent: i32,
    pub depth: u8,
    pub position: usize,
    pub anc: u64,
    pub logp: f64,
}

#[derive(Clone, Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    prefix: usize,
    capacity: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WaveStats {
    pub waves: usize,
    pub requests: usize,
    pub batch_sizes: Vec<usize>,
    pub used_requests: usize,
}

struct Candidate {
    token: u32,
    parent: Option<usize>,
    depth: u8,
    logp: f64,
}

impl Candidate {
    fn order(nodes: &[Self], a: usize, b: usize) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let left = &nodes[a];
        let right = &nodes[b];
        left.logp
            .total_cmp(&right.logp)
            .then_with(|| right.depth.cmp(&left.depth))
            .then_with(|| match (left.parent, right.parent) {
                (Some(a), Some(b)) => Self::order(nodes, a, b),
                _ => Ordering::Equal,
            })
            .then_with(|| right.token.cmp(&left.token))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Walk {
    pub rows: Vec<usize>,
    pub bonus: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub position: usize,
    pub token: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commit {
    pub matched: usize,
    pub output: Vec<u32>,
    pub rows: Vec<usize>,
    pub logits_row: usize,
    pub next: u32,
    pub position: usize,
}

#[derive(Clone, Debug)]
struct Pending {
    parent: usize,
    depth: u8,
    token: u32,
    logp: f64,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        self.logp
            .total_cmp(&other.logp)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.parent.cmp(&self.parent))
            .then_with(|| other.token.cmp(&self.token))
    }
}

impl Tree {
    fn extent(prefix: usize, capacity: usize, rows: usize) -> Result<()> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "tree requires 1..=64 rows including anchor"
        );
        ensure!(
            prefix
                .checked_add(rows)
                .is_some_and(|n| n <= capacity && n <= i32::MAX as usize),
            "tree temporary KV extent exceeds capacity or device position range"
        );
        Ok(())
    }

    pub fn edges(tokens: &[u32], parents: &[i32], prefix: usize, capacity: usize) -> Result<Self> {
        ensure!(
            tokens.len() == parents.len(),
            "tree token/parent extent mismatch"
        );
        Self::extent(prefix, capacity, tokens.len())?;
        ensure!(parents[0] == -1, "tree root must have parent -1");
        let mut tree = Self {
            nodes: vec![Node {
                token: tokens[0],
                parent: -1,
                depth: 0,
                position: prefix,
                anc: 1,
                logp: 0.0,
            }],
            prefix,
            capacity,
        };
        for row in 1..tokens.len() {
            ensure!(
                parents[row] >= 0 && (parents[row] as usize) < row,
                "tree parents must precede their children"
            );
            tree.push(parents[row] as usize, tokens[row], 0.0)?;
        }
        Ok(tree)
    }

    fn push(&mut self, parent: usize, token: u32, logp: f64) -> Result<()> {
        ensure!(
            self.nodes.len() < MAX_ROWS && parent < self.nodes.len(),
            "invalid tree insertion"
        );
        ensure!(
            logp.is_finite() && logp <= 0.0,
            "invalid cumulative log probability"
        );
        ensure!(
            !self
                .nodes
                .iter()
                .any(|n| n.parent == parent as i32 && n.token == token),
            "duplicate token among siblings"
        );
        let row = self.nodes.len();
        Self::extent(self.prefix, self.capacity, row + 1)?;
        let p = &self.nodes[parent];
        let depth = p
            .depth
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("tree depth overflow"))?;
        let position = self
            .prefix
            .checked_add(usize::from(depth))
            .ok_or_else(|| anyhow::anyhow!("tree position overflow"))?;
        self.nodes.push(Node {
            token,
            parent: parent as i32,
            depth,
            position,
            anc: p.anc | (1u64 << row),
            logp,
        });
        Ok(())
    }

    pub fn best_first<F>(
        anchor: u32,
        prefix: usize,
        capacity: usize,
        budget: usize,
        depth: u8,
        mut distribution: F,
    ) -> Result<Self>
    where
        F: FnMut(u8, u32) -> Result<Distribution>,
    {
        Self::extent(prefix, capacity, budget)?;
        ensure!((1..=7).contains(&depth), "DSpark tree depth must be 1..=7");
        let mut tree = Self::edges(&[anchor], &[-1], prefix, capacity)?;
        let mut heap = BinaryHeap::new();
        if budget > 1 {
            tree.expand(0, &mut heap, &mut distribution)?;
        }
        while tree.nodes.len() < budget {
            let Some(p) = heap.pop() else { break };
            let row = tree.nodes.len();
            tree.push(p.parent, p.token, p.logp)?;
            if p.depth < depth && tree.nodes.len() < budget {
                tree.expand(row, &mut heap, &mut distribution)?;
            }
        }
        Ok(tree)
    }

    pub fn best_first_waves<F>(
        anchor: u32,
        prefix: usize,
        capacity: usize,
        budget: usize,
        depth: u8,
        mut batch: F,
    ) -> Result<(Self, WaveStats)>
    where
        F: FnMut(&[(u8, u32)]) -> Result<Vec<Distribution>>,
    {
        Self::extent(prefix, capacity, budget)?;
        ensure!((1..=7).contains(&depth), "DSpark tree depth must be 1..=7");
        let mut nodes = vec![Candidate {
            token: anchor,
            parent: None,
            depth: 0,
            logp: 0.0,
        }];
        let mut active = vec![0usize];
        let mut stats = WaveStats::default();
        for level in 0..depth {
            let eligible: Vec<usize> = active
                .iter()
                .take(budget - 1)
                .copied()
                .filter(|&i| nodes[i].depth == level)
                .collect();
            if eligible.is_empty() {
                break;
            }
            let mut index = HashMap::new();
            let mut requests = Vec::new();
            for &parent in &eligible {
                let key = (level, nodes[parent].token);
                index.entry(key).or_insert_with(|| {
                    let slot = requests.len();
                    requests.push(key);
                    slot
                });
            }
            let distributions = batch(&requests)?;
            ensure!(
                distributions.len() == requests.len(),
                "tree batch returned {} distributions for {} ordered requests",
                distributions.len(),
                requests.len()
            );
            stats.waves += 1;
            stats.requests += requests.len();
            stats.batch_sizes.push(requests.len());
            for parent in eligible {
                let d = &distributions[index[&(level, nodes[parent].token)]];
                for choice in d.choices() {
                    let logp = nodes[parent].logp + choice.logp;
                    ensure!(
                        logp.is_finite() && logp <= nodes[parent].logp,
                        "invalid draft path probability"
                    );
                    active.push(nodes.len());
                    nodes.push(Candidate {
                        token: choice.token,
                        parent: Some(parent),
                        depth: level + 1,
                        logp,
                    });
                }
            }
            active.sort_by(|&a, &b| Candidate::order(&nodes, b, a));
            active.truncate(budget);
        }
        let mut rows = vec![None; nodes.len()];
        let mut tokens = Vec::with_capacity(active.len());
        let mut parents = Vec::with_capacity(active.len());
        let mut used = BTreeSet::new();
        for (row, &i) in active.iter().enumerate() {
            rows[i] = Some(row);
            tokens.push(nodes[i].token);
            parents.push(match nodes[i].parent {
                Some(parent) => rows[parent]
                    .ok_or_else(|| anyhow::anyhow!("tree wave selection lost an ancestor"))?
                    as i32,
                None => -1,
            });
            if row < budget - 1 && nodes[i].depth < depth {
                used.insert((nodes[i].depth, nodes[i].token));
            }
        }
        stats.used_requests = used.len();
        let mut tree = Self::edges(&tokens, &parents, prefix, capacity)?;
        for (node, &i) in tree.nodes.iter_mut().zip(&active) {
            node.logp = nodes[i].logp;
        }
        Ok((tree, stats))
    }

    fn expand<F>(
        &self,
        parent: usize,
        heap: &mut BinaryHeap<Pending>,
        distribution: &mut F,
    ) -> Result<()>
    where
        F: FnMut(u8, u32) -> Result<Distribution>,
    {
        let p = &self.nodes[parent];
        let d = distribution(p.depth, p.token)?;
        for c in d.choices {
            let logp = p.logp + c.logp;
            ensure!(
                logp.is_finite() && logp <= p.logp,
                "invalid draft path probability"
            );
            heap.push(Pending {
                parent,
                depth: p.depth + 1,
                token: c.token,
                logp,
            });
        }
        Ok(())
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn prefix(&self) -> usize {
        self.prefix
    }

    pub fn walk(&self, predictions: &[u32]) -> Result<Walk> {
        ensure!(
            predictions.len() == self.nodes.len(),
            "tree prediction extent mismatch"
        );
        let mut rows = vec![0];
        loop {
            let row = *rows.last().expect("root path is nonempty");
            let bonus = predictions[row];
            let child = self
                .nodes
                .iter()
                .enumerate()
                .find(|(_, n)| n.parent == row as i32 && n.token == bonus);
            if let Some((i, _)) = child {
                rows.push(i);
            } else {
                return Ok(Walk { rows, bonus });
            }
        }
    }

    pub fn select(&self, predictions: &[u32], limit: usize, eos: &[u32]) -> Result<Commit> {
        ensure!(limit > 0, "tree output limit must be positive");
        let walk = self.walk(predictions)?;
        let mut output: Vec<u32> = walk.rows[1..]
            .iter()
            .map(|&i| self.nodes[i].token)
            .collect();
        output.push(walk.bonus);
        let matched = walk.rows.len() - 1;
        let end = eos
            .iter()
            .filter_map(|token| output.iter().position(|v| v == token))
            .min()
            .map_or(output.len(), |i| i + 1);
        output.truncate(limit.min(end));
        let rows = walk.rows[..output.len()].to_vec();
        let next = *output.last().expect("positive output limit");
        let logits_row = *rows.last().expect("root is committed");
        let position = self
            .prefix
            .checked_add(output.len())
            .ok_or_else(|| anyhow::anyhow!("tree commit position overflow"))?;
        ensure!(position <= self.capacity, "tree commit exceeds capacity");
        Ok(Commit {
            matched,
            output,
            rows,
            logits_row,
            next,
            position,
        })
    }

    pub fn check_commit(
        &self,
        plan: &Commit,
        predictions: &[u32],
        eos: &[u32],
        cursor: &Cursor,
    ) -> Result<()> {
        ensure!(
            cursor.position == self.prefix && cursor.token == self.nodes[0].token,
            "tree transaction cursor is stale"
        );
        let expected = self.select(predictions, plan.output.len(), eos)?;
        ensure!(
            *plan == expected,
            "tree transaction commit plan is inconsistent"
        );
        Ok(())
    }

    /// KV and hidden work must complete before the caller publishes this logical cursor.
    pub fn finish(
        &self,
        plan: &Commit,
        predictions: &[u32],
        eos: &[u32],
        cursor: &mut Cursor,
    ) -> Result<()> {
        self.check_commit(plan, predictions, eos, cursor)?;
        *cursor = Cursor {
            position: plan.position,
            token: plan.next,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tree() -> Tree {
        Tree::edges(&[10, 20, 30, 21, 31], &[-1, 0, 0, 1, 2], 100, 105).unwrap()
    }
    #[test]
    fn siblings_can_exchange_rows_without_changing_output() {
        let a = tree().select(&[20, 21, 31, 99, 98], 8, &[]).unwrap();
        let b = Tree::edges(&[10, 30, 20, 31, 21], &[-1, 0, 0, 1, 2], 100, 105)
            .unwrap()
            .select(&[20, 31, 21, 98, 99], 8, &[])
            .unwrap();
        assert_eq!(a.output, vec![20, 21, 99]);
        assert_eq!(a.output, b.output);
        assert_eq!(a.position, b.position);
        assert_eq!(a.rows, vec![0, 1, 3]);
        assert_eq!(b.rows, vec![0, 2, 4]);
    }
    #[test]
    fn changing_an_ancestor_changes_descendant_context() {
        let a = tree();
        let b = Tree::edges(&[10, 20, 30, 21, 31], &[-1, 0, 0, 2, 2], 100, 105).unwrap();
        let values = [1, 2, 4, 8, 16];
        let sum = |t: &Tree| {
            values
                .iter()
                .enumerate()
                .filter(|(i, _)| t.nodes()[3].anc & (1u64 << i) != 0)
                .map(|(_, v)| v)
                .sum::<i32>()
        };
        assert_eq!(sum(&a), 11);
        assert_eq!(sum(&b), 13);
        assert_eq!(a.nodes()[3].position, b.nodes()[3].position);
    }
    #[test]
    fn rejection_commits_root_and_leaves_prediction_pending() {
        let c = tree().select(&[42, 21, 31, 99, 98], 8, &[]).unwrap();
        assert_eq!(c.output, vec![42]);
        assert_eq!(c.rows, vec![0]);
        assert_eq!(c.matched, 0);
        assert_eq!(c.next, 42);
        assert_eq!(c.logits_row, 0);
        assert_eq!(c.position, 101);
    }
    #[test]
    fn leaf_acceptance_keeps_bonus_out_of_kv() {
        let c = tree().select(&[20, 21, 31, 99, 98], 8, &[]).unwrap();
        assert_eq!(c.matched, 2);
        assert_eq!(c.rows, vec![0, 1, 3]);
        assert_eq!(c.next, 99);
        assert_eq!(c.logits_row, 3);
        assert_eq!(c.position, 103);
    }
    #[test]
    fn eos_and_limit_leave_last_emitted_token_pending() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        let c = t.select(&p, 8, &[21]).unwrap();
        assert_eq!(c.output, vec![20, 21]);
        assert_eq!(c.rows, vec![0, 1]);
        assert_eq!(c.next, 21);
        assert_eq!(c.logits_row, 1);
        let c = t.select(&p, 1, &[]).unwrap();
        assert_eq!(c.output, vec![20]);
        assert_eq!(c.rows, vec![0]);
        assert_eq!(c.next, 20);
        assert_eq!(c.position, 101);
        let c = t.select(&p, 8, &[20]).unwrap();
        assert_eq!(c.rows, vec![0]);
        assert_eq!(c.output, vec![20]);
        assert!(t.select(&p, 0, &[]).is_err());
    }
    #[test]
    fn masks_cover_only_parent_path_and_self() {
        let t = tree();
        assert_eq!(t.nodes()[3].anc, 0b01011);
        assert_eq!(t.nodes()[4].anc, 0b10101);
        assert_eq!(t.nodes()[1].position, 101);
        assert_eq!(t.nodes()[2].position, 101);
        assert_eq!(t.nodes()[4].position, 102);
    }
    #[test]
    fn row_63_uses_high_bit_without_shift_overflow() {
        let tokens: Vec<_> = (0..64).collect();
        let parents: Vec<_> = (0..64).map(|i| i - 1).collect();
        let t = Tree::edges(&tokens, &parents, 0, 64).unwrap();
        assert_eq!(t.nodes()[63].anc, u64::MAX);
    }
    #[test]
    fn capacity_and_topology_errors_are_explicit() {
        assert!(Tree::edges(&[1, 2], &[-1, 0], 9, 10).is_err());
        assert!(Tree::edges(&[1], &[-1], 10, 10).is_err());
        assert!(Tree::edges(&[1], &[-1], usize::MAX, usize::MAX).is_err());
        assert!(Tree::edges(&[1, 2], &[-1, -1], 0, 2).is_err());
        assert!(Tree::edges(&[1, 2], &[-1, 1], 0, 2).is_err());
        assert!(Tree::edges(&[1, 2, 2], &[-1, 0, 0], 0, 3).is_err());
        assert!(Tree::edges(&[], &[], 0, 0).is_err());
        assert!(tree().select(&[1], 1, &[]).is_err());
        let t = Tree::edges(&[1, 2], &[-1, 0], 8, 10).unwrap();
        assert_eq!(t.select(&[2, 3], 2, &[]).unwrap().position, 10);
    }
    #[test]
    fn best_first_uses_path_probability_and_parent_conditioning() {
        let mut calls = Vec::new();
        let t = Tree::best_first(3, 10, 15, 5, 3, |depth, parent| {
            calls.push((depth, parent));
            Distribution::logits(
                if parent == 3 {
                    &[1.0, 0.0, -9.0]
                } else if parent == 0 {
                    &[-9.0, 0.0, 2.0]
                } else {
                    &[0.0, 0.0, 0.0]
                },
                2,
            )
        })
        .unwrap();
        assert_eq!(t.nodes()[1].token, 0);
        assert_eq!(t.nodes()[1].parent, 0);
        assert_eq!(t.nodes()[2].token, 2);
        assert_eq!(t.nodes()[2].parent, 1);
        assert!(calls.contains(&(1, 0)));
        assert!(calls.contains(&(2, 2)));
        for n in t.nodes().iter().skip(1) {
            assert!(n.logp <= t.nodes()[n.parent as usize].logp);
        }
    }
    #[test]
    fn distribution_is_normalized_over_full_vocabulary() {
        let d = Distribution::logits(&[0., 0., 0., 0.], 2).unwrap();
        assert_eq!(d.choices()[0].token, 0);
        assert_eq!(d.choices()[1].token, 1);
        assert!((d.choices()[0].logp.exp() - 0.25).abs() < 1e-7);
        assert!(Distribution::logits(&[f32::NAN], 1).is_err());
        assert!(Distribution::logits(&[f32::INFINITY], 1).is_err());
        assert!(Distribution::logits(&[f32::NEG_INFINITY], 1).is_err());
        assert!(Distribution::logits(&[0.], 0).is_err());
    }
    #[test]
    fn gpu_top_candidates_keep_full_vocabulary_probabilities() {
        let d = Distribution::top(vec![
            Choice {
                token: 2,
                logp: 0.2f64.ln(),
            },
            Choice {
                token: 1,
                logp: 0.3f64.ln(),
            },
        ])
        .unwrap();
        assert_eq!(d.choices()[0].token, 1);
        assert!((d.choices()[0].logp.exp() - 0.3).abs() < 1e-12);
        assert!(Distribution::top(vec![]).is_err());
        assert!(
            Distribution::top(vec![
                Choice {
                    token: 1,
                    logp: 0.0
                },
                Choice {
                    token: 2,
                    logp: 0.0
                }
            ])
            .is_err()
        );
        assert!(
            Distribution::top(vec![
                Choice {
                    token: 1,
                    logp: -1.0
                },
                Choice {
                    token: 1,
                    logp: -1.0
                }
            ])
            .is_err()
        );
        assert!(
            Distribution::top(vec![Choice {
                token: 1,
                logp: f64::NAN
            }])
            .is_err()
        );
    }

    #[test]
    fn full_walk_keeps_the_leaf_bonus_separate_from_commit_limits() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        assert_eq!(
            t.walk(&p).unwrap(),
            Walk {
                rows: vec![0, 1, 3],
                bonus: 99
            }
        );
        assert_eq!(t.select(&p, 1, &[]).unwrap().next, 20);
    }
    #[test]
    fn stale_cursor_cannot_publish_a_commit() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        let plan = t.select(&p, 8, &[]).unwrap();
        for state in [
            Cursor {
                position: 101,
                token: 10,
            },
            Cursor {
                position: 100,
                token: 11,
            },
        ] {
            let mut cursor = state;
            assert!(t.finish(&plan, &p, &[], &mut cursor).is_err());
            assert_eq!(cursor, state);
        }
    }
    #[test]
    fn corrupt_commit_does_not_change_the_cursor() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        let plan = t.select(&p, 8, &[]).unwrap();
        let mut bad = plan.clone();
        bad.rows = vec![0, 2, 4];
        let mut cursor = Cursor {
            position: 100,
            token: 10,
        };
        let original = cursor;
        assert!(t.finish(&bad, &p, &[], &mut cursor).is_err());
        assert_eq!(cursor, original);
        let mut bad = plan.clone();
        bad.position = 106;
        assert!(t.finish(&bad, &p, &[], &mut cursor).is_err());
        assert_eq!(cursor, original);
        let mut bad = plan;
        bad.next = 42;
        assert!(t.finish(&bad, &p, &[], &mut cursor).is_err());
        assert_eq!(cursor, original);
    }
    #[test]
    fn finish_commits_inputs_and_keeps_last_output_pending() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        let mut cursor = Cursor {
            position: 100,
            token: 10,
        };
        let c = t.select(&p, 8, &[21]).unwrap();
        t.finish(&c, &p, &[21], &mut cursor).unwrap();
        assert_eq!(c.rows, vec![0, 1]);
        assert_eq!(
            cursor,
            Cursor {
                position: 102,
                token: 21
            }
        );
        assert!(t.finish(&c, &p, &[21], &mut cursor).is_err());
    }

    #[test]
    fn checking_a_commit_does_not_publish_the_pending_token() {
        let t = tree();
        let p = [20, 21, 31, 99, 98];
        let plan = t.select(&p, 8, &[]).unwrap();
        let cursor = Cursor {
            position: 100,
            token: 10,
        };
        t.check_commit(&plan, &p, &[], &cursor).unwrap();
        assert_eq!(
            cursor,
            Cursor {
                position: 100,
                token: 10
            }
        );
        let mut bad = plan;
        bad.rows = vec![0, 2, 4];
        assert!(t.check_commit(&bad, &p, &[], &cursor).is_err());
    }

    #[test]
    fn best_first_children_are_a_rank_prefix_for_all_budgets() {
        for seed in 0u32..16 {
            for branches in [4, 8] {
                for budget in [16, 32, 64] {
                    let scores = |depth: u8, parent: u32| -> Result<Distribution> {
                        let logits: Vec<f32> = (0u32..23)
                            .map(|token| {
                                let value = seed
                                    .wrapping_mul(747796405)
                                    .wrapping_add(parent.wrapping_mul(2891336453))
                                    .wrapping_add(token.wrapping_mul(277803737))
                                    .wrapping_add(u32::from(depth) * 103);
                                (value % 16384) as f32 / 2048.0
                            })
                            .collect();
                        Distribution::logits(&logits, branches)
                    };
                    let t = Tree::best_first(22, 0, budget, budget, 7, scores).unwrap();
                    let (wave, stats) =
                        Tree::best_first_waves(22, 0, budget, budget, 7, |requests| {
                            requests
                                .iter()
                                .map(|&(depth, token)| scores(depth, token))
                                .collect()
                        })
                        .unwrap();
                    assert_eq!(
                        wave.nodes(),
                        t.nodes(),
                        "seed={seed} branches={branches} budget={budget}"
                    );
                    assert!(stats.waves <= 7);
                    assert!(stats.batch_sizes.iter().all(|&n| n > 0 && n < budget));
                    assert_eq!(stats.requests, stats.batch_sizes.iter().sum::<usize>());
                    assert!(stats.used_requests <= stats.requests);
                    assert_eq!(t.nodes().len(), budget);
                    for (index, parent) in t.nodes().iter().enumerate() {
                        let children: Vec<u32> = t
                            .nodes()
                            .iter()
                            .filter(|n| n.parent == index as i32)
                            .map(|n| n.token)
                            .collect();
                        if children.is_empty() {
                            continue;
                        }
                        let candidates = scores(parent.depth, parent.token).unwrap();
                        let expected: Vec<u32> = candidates
                            .choices()
                            .iter()
                            .take(children.len())
                            .map(|c| c.token)
                            .collect();
                        assert_eq!(
                            children, expected,
                            "seed={seed} budget={budget} parent={index}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn tied_scores_keep_a_deterministic_rank_prefix() {
        let t =
            Tree::best_first(9, 0, 64, 64, 7, |_, _| Distribution::logits(&[0.0; 16], 8)).unwrap();
        for (index, _) in t.nodes().iter().enumerate() {
            let children: Vec<u32> = t
                .nodes()
                .iter()
                .filter(|n| n.parent == index as i32)
                .map(|n| n.token)
                .collect();
            assert_eq!(children, (0..children.len() as u32).collect::<Vec<_>>());
        }
        let (wave, _) = Tree::best_first_waves(9, 0, 64, 64, 7, |requests| {
            requests
                .iter()
                .map(|_| Distribution::logits(&[0.0; 16], 8))
                .collect()
        })
        .unwrap();
        assert_eq!(wave.nodes(), t.nodes());
    }

    #[test]
    fn wave_batch_extent_and_capacity_errors_are_explicit() {
        assert!(Tree::best_first_waves(0, 0, 16, 16, 7, |_| Ok(Vec::new())).is_err());
        assert!(
            Tree::best_first_waves(0, 1, 16, 16, 7, |_| {
                panic!("capacity must be checked before requests")
            })
            .is_err()
        );
        assert!(
            Tree::best_first_waves(0, 0, 16, 16, 0, |_| {
                panic!("depth must be checked before requests")
            })
            .is_err()
        );
    }

    #[test]
    fn depth_waves_preserve_a_deep_first_multibranch_tree() {
        let scores = |_: u8, token: u32| {
            Distribution::top(vec![
                Choice {
                    token: token * 2 + 1,
                    logp: 0.9f64.ln(),
                },
                Choice {
                    token: token * 2 + 2,
                    logp: 0.1f64.ln(),
                },
            ])
        };
        let mut serial_requests = 0;
        let serial = Tree::best_first(0, 100, 116, 16, 7, |row, token| {
            serial_requests += 1;
            scores(row, token)
        })
        .unwrap();
        let (wave, stats) = Tree::best_first_waves(0, 100, 116, 16, 7, |requests| {
            requests
                .iter()
                .map(|&(row, token)| scores(row, token))
                .collect()
        })
        .unwrap();
        assert_eq!(wave.nodes(), serial.nodes());
        assert!(serial_requests > 7);
        assert_eq!(stats.waves, 7);
    }

    #[test]
    fn wave_ties_use_final_parent_order_instead_of_arena_order() {
        let scores = |depth: u8, token: u32| {
            let choices = match (depth, token) {
                (0, 0) => vec![(1, -1.0), (2, -2.0)],
                (1, 1) => vec![(3, -3.0)],
                (1, 2) => vec![(4, -1.0)],
                (2, 3) => vec![(5, -1.0)],
                (2, 4) => vec![(6, -2.0)],
                _ => vec![(token + 10, -2.0)],
            };
            Distribution::top(
                choices
                    .into_iter()
                    .map(|(token, logp)| Choice { token, logp })
                    .collect(),
            )
        };
        let serial = Tree::best_first(0, 0, 7, 7, 7, scores).unwrap();
        let (wave, _) = Tree::best_first_waves(0, 0, 7, 7, 7, |requests| {
            requests
                .iter()
                .map(|&(row, token)| scores(row, token))
                .collect()
        })
        .unwrap();
        assert_eq!(
            serial.nodes().iter().map(|n| n.token).collect::<Vec<_>>(),
            [0, 1, 2, 4, 3, 6, 5]
        );
        assert_eq!(wave.nodes(), serial.nodes());
    }
}
