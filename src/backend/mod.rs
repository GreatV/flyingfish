use anyhow::Result;
use serde::Serialize;
pub mod setup;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::engine::Engine as Backend;
#[cfg(not(feature = "cuda"))]
mod missing;
#[cfg(not(feature = "cuda"))]
pub use missing::Backend;

#[derive(Clone, Debug)]
pub struct Top4 {
    pub tokens: [u32; 4],
    pub logp: [f32; 4],
    pub lse: f32,
}

impl Top4 {
    pub fn distribution(&self) -> Result<crate::tree::Distribution> {
        anyhow::ensure!(self.lse.is_finite(), "nonfinite Markov logsumexp");
        crate::tree::Distribution::top(
            self.tokens
                .iter()
                .zip(self.logp)
                .map(|(&token, logp)| crate::tree::Choice {
                    token,
                    logp: f64::from(logp),
                })
                .collect(),
        )
    }

    pub fn bits_equal(&self, other: &Self) -> bool {
        self.tokens == other.tokens
            && self.logp.map(f32::to_bits) == other.logp.map(f32::to_bits)
            && self.lse.to_bits() == other.lse.to_bits()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpecBudget {
    Chain,
    #[default]
    Tree16,
    Tree32,
    Tree64,
}

impl SpecBudget {
    pub const fn rows(self) -> usize {
        match self {
            Self::Chain => 8,
            Self::Tree16 => 16,
            Self::Tree32 => 32,
            Self::Tree64 => 64,
        }
    }

    pub const fn is_tree(self) -> bool {
        !matches!(self, Self::Chain)
    }

    pub fn workspace_rows(self, chunk: usize) -> usize {
        chunk.max(self.rows())
    }

    pub fn check_tree(self, step: Step<'_>, prefix: usize, capacity: usize) -> Result<()> {
        anyhow::ensure!(
            self.is_tree(),
            "tree forward requires --spec-budget tree16, tree32 or tree64"
        );
        step.check(self.rows())?;
        anyhow::ensure!(
            step.input == Input::Ids && step.head == Head::All,
            "tree verification requires token IDs and all-row logits"
        );
        anyhow::ensure!(
            prefix
                .checked_add(step.rows)
                .is_some_and(|end| end <= capacity),
            "tree temporary slots exceed KV capacity"
        );
        let Mask::Tree { parents, positions } = step.mask else {
            anyhow::bail!("tree preparation requires tree metadata");
        };
        anyhow::ensure!(
            parents[0] == -1 && positions[0] == prefix,
            "tree root must have parent -1 and position equal to committed prefix"
        );
        for row in 1..step.rows {
            anyhow::ensure!(parents[row] >= 0, "tree row {row} has an extra root");
            anyhow::ensure!(
                positions[row]
                    .checked_sub(prefix)
                    .is_some_and(|depth| depth <= 7),
                "tree row {row} exceeds draft depth7"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, clap::Args, Serialize)]
pub struct Settings {
    #[arg(long, global = true, value_enum, default_value_t = TreeBuilder::Waves)]
    pub tree_builder: TreeBuilder,
    #[arg(long, global = true, value_enum, default_value_t = SpecBudget::default(), requires = "draft_model")]
    pub spec_budget: SpecBudget,
    #[arg(long, global = true, default_value_t = true, action = clap::ArgAction::Set)]
    pub spec_graph: bool,
    #[arg(skip)]
    pub linear_choices: Vec<setup::LinearChoice>,
    #[arg(skip)]
    pub draft_model: Option<std::path::PathBuf>,
    #[arg(long, global = true, env = "FLYINGFISH_RUNTIME_DIR")]
    pub runtime_dir: Option<std::path::PathBuf>,
}

impl Settings {
    pub fn with_draft(&self, path: &std::path::Path) -> Self {
        let mut settings = self.clone();
        settings.draft_model = Some(path.to_owned());
        settings
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TreeBuilder {
    Serial,
    Waves,
}

#[cfg(feature = "cuda")]
pub struct Event(cudarc::driver::CudaEvent);
#[cfg(not(feature = "cuda"))]
pub enum Event {}

#[cfg(feature = "cuda")]
pub struct Logits(cudarc::driver::CudaSlice<half::bf16>);
#[cfg(not(feature = "cuda"))]
pub enum Logits {}

impl Event {
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        return Ok(self.0.synchronize()?);
        #[cfg(not(feature = "cuda"))]
        match *self {}
    }
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        #[cfg(feature = "cuda")]
        return Ok(self.0.elapsed_ms(&end.0)?);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = end;
            match *self {}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Ids,
    Token,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    None,
    Last,
    All,
}

#[derive(Clone, Copy, Debug)]
pub enum Mask<'a> {
    Causal,
    Tree {
        parents: &'a [i32],
        positions: &'a [usize],
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    pub rows: usize,
    pub input: Input,
    pub head: Head,
    pub mask: Mask<'a>,
}

impl Step<'_> {
    pub fn check(&self, available_rows: usize) -> Result<()> {
        anyhow::ensure!(
            self.rows > 0 && self.rows <= available_rows,
            "forward rows {} exceed workspace {}",
            self.rows,
            available_rows
        );
        anyhow::ensure!(
            self.input != Input::Token || self.rows == 1,
            "device token input requires exactly one row"
        );
        if let Mask::Tree { parents, positions } = self.mask {
            anyhow::ensure!(
                self.rows <= 128 && parents.len() == self.rows && positions.len() == self.rows,
                "tree verification requires 1..=128 rows with matching parent and position arrays"
            );
            for (row, &parent) in parents.iter().enumerate() {
                anyhow::ensure!(
                    parent >= -1 && parent < row as i32,
                    "invalid tree parent {parent} at row {row}"
                );
                if parent >= 0 {
                    anyhow::ensure!(
                        positions[parent as usize].checked_add(1) == Some(positions[row]),
                        "tree position does not follow parent at row {row}"
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Operator {
    Embed,
    Norm,
    Qkv,
    RopeKv,
    Attention,
    Output,
    PostNorm,
    GateUp,
    Activate,
    Down,
    NextNorm,
    Head,
}

pub struct Target<'a> {
    pub trace: &'a mut crate::trace::Trace,
    pub prefix: &'a str,
    pub skip: usize,
}

#[derive(Debug, Serialize)]
pub struct TreeMetadata {
    pub ids: Vec<u32>,
    pub ancestors: Vec<u64>,
    pub positions: Vec<i32>,
    pub slots: Vec<i32>,
    pub valid_rows: i32,
    pub prefix: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tree_budget_limits_rows_depth_roots_and_physical_slots() {
        let parents = [-1, 0, 0, 1];
        let positions = [100, 101, 101, 102];
        let step = Step {
            rows: 4,
            input: Input::Ids,
            head: Head::All,
            mask: Mask::Tree {
                parents: &parents,
                positions: &positions,
            },
        };
        assert!(SpecBudget::Tree16.check_tree(step, 100, 104).is_ok());
        assert!(SpecBudget::Chain.check_tree(step, 100, 104).is_err());
        assert!(SpecBudget::Tree16.check_tree(step, 100, 103).is_err());
        let roots = [-1, 0, -1, 1];
        assert!(
            SpecBudget::Tree16
                .check_tree(
                    Step {
                        mask: Mask::Tree {
                            parents: &roots,
                            positions: &positions
                        },
                        ..step
                    },
                    100,
                    104
                )
                .is_err()
        );
        let chain: Vec<_> = (0..9).map(|i| i - 1).collect();
        let deep: Vec<_> = (100..109).collect();
        assert!(
            SpecBudget::Tree16
                .check_tree(
                    Step {
                        rows: 9,
                        mask: Mask::Tree {
                            parents: &chain,
                            positions: &deep
                        },
                        ..step
                    },
                    100,
                    128
                )
                .is_err()
        );
        assert_eq!(SpecBudget::Tree64.workspace_rows(8), 64);
        assert_eq!(SpecBudget::Tree64.workspace_rows(256), 256);
        assert_eq!(SpecBudget::Chain.workspace_rows(8), 8);
    }

    #[test]
    fn workspace_holds_validation_rows_for_small_prefill_chunks() {
        for budget in [
            SpecBudget::Chain,
            SpecBudget::Tree16,
            SpecBudget::Tree32,
            SpecBudget::Tree64,
        ] {
            for chunk in 1..budget.rows() {
                assert_eq!(budget.workspace_rows(chunk), budget.rows());
            }
            assert_eq!(budget.workspace_rows(256), 256);
        }
    }
    #[test]
    fn verification_accepts_128_rows_and_rejects_cycles() {
        let parents: Vec<i32> = (0..128).map(|i| i - 1).collect();
        let positions: Vec<usize> = (0..128).collect();
        let step = Step {
            rows: 128,
            input: Input::Ids,
            head: Head::All,
            mask: Mask::Tree {
                parents: &parents,
                positions: &positions,
            },
        };
        assert!(step.check(128).is_ok());
        assert!(step.check(127).is_err());
        let mut invalid = parents.clone();
        invalid[64] = 64;
        assert!(
            Step {
                mask: Mask::Tree {
                    parents: &invalid,
                    positions: &positions
                },
                ..step
            }
            .check(128)
            .is_err()
        );
    }
}
