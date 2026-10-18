use crate::{
    backend::{
        SpecBudget,
        setup::{CacheKey, LinearShape},
    },
    config::Config,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, hash_map::DefaultHasher},
    fs,
    hash::Hasher,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Setup,
    Frozen,
    Ready,
}

pub(crate) struct Closure {
    write: bool,
    stage: Cell<Stage>,
    measurements: Cell<usize>,
    captures: Cell<usize>,
    late_attempts: Cell<usize>,
    files: RefCell<BTreeSet<PathBuf>>,
    command: String,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Profile {
    pub linear: BTreeSet<LinearShape>,
    pub pairs: Pairs,
    pub attention: Vec<Attention>,
    capacity: usize,
    small_rows: usize,
    budget: Option<usize>,
    model: String,
    draft: Option<String>,
}

type Pairs = BTreeSet<(usize, usize)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Attention {
    Decode,
    Draft,
    Verify,
    Tree,
}

fn attention(budget: Option<SpecBudget>) -> Vec<Attention> {
    let mut kinds = vec![Attention::Decode];
    if let Some(budget) = budget {
        kinds.push(Attention::Draft);
        kinds.push(if budget.is_tree() {
            Attention::Tree
        } else {
            Attention::Verify
        });
    }
    kinds
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
struct Manifest {
    schema: u32,
    key: CacheKey,
    profile: Profile,
    files: Vec<(String, String)>,
}

pub(super) fn hash(bytes: &[u8]) -> String {
    let mut hash = DefaultHasher::new();
    hash.write(bytes);
    format!("{:016x}", hash.finish())
}

pub(super) fn small_rows(chunk: usize, capacity: usize) -> usize {
    chunk.min(capacity).min(16)
}

pub(super) fn injection_limit(small_rows: usize) -> usize {
    small_rows.max(8)
}

pub(super) fn forward_rows(small_rows: usize, budget: Option<SpecBudget>) -> BTreeSet<usize> {
    (1..=small_rows)
        .chain(budget.into_iter().flat_map(|b| [7, b.rows()]))
        .collect()
}

fn shapes(
    c: &Config,
    small_rows: usize,
    budget: Option<SpecBudget>,
    draft: Option<&crate::dspark::DraftConfig>,
) -> Result<(BTreeSet<LinearShape>, Pairs)> {
    let mut linear = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    let mut add = |rows, output, input| {
        pairs.insert((output, input));
        let shape = LinearShape {
            rows,
            output,
            input,
        };
        if super::linear_calibrate::supports(shape) {
            linear.insert(shape);
        }
    };
    for rows in forward_rows(small_rows, budget) {
        for (output, input) in [
            (c.qkv_dim(), c.hidden_size),
            (c.hidden_size, c.hidden_size),
            (2 * c.intermediate_size, c.hidden_size),
            (c.hidden_size, c.intermediate_size),
        ] {
            add(rows, output, input);
        }
    }
    add(1, c.vocab_size, c.hidden_size);
    if let Some(budget) = budget {
        let draft = draft.context("draft config missing from closure declaration")?;
        for (output, input) in [
            (draft.qkv_dim(), draft.hidden_size),
            (draft.hidden_size, draft.hidden_size),
            (2 * draft.intermediate_size, draft.hidden_size),
            (draft.hidden_size, draft.intermediate_size),
            (draft.vocab_size, draft.hidden_size),
        ] {
            add(draft.block_size, output, input);
        }
        add(budget.rows(), c.vocab_size, c.hidden_size);
        let max = injection_limit(small_rows);
        for rows in 1..=max {
            for (output, input) in [
                (draft.hidden_size, draft.capture_width()),
                (draft.kv_dim(), draft.hidden_size),
            ] {
                add(rows, output, input);
            }
        }
        add(1, draft.vocab_size, draft.markov_rank);
    }
    Ok((linear, pairs))
}

impl Profile {
    pub fn new(
        c: &Config,
        capacity: usize,
        chunk: usize,
        budget: Option<SpecBudget>,
        model: &Path,
        draft: Option<&Path>,
        draft_config: Option<&crate::dspark::DraftConfig>,
    ) -> Result<Self> {
        let small_rows = small_rows(chunk, capacity);
        ensure!(small_rows > 0, "empty profile row bound");
        let (linear, pairs) = shapes(c, small_rows, budget, draft_config)?;
        Ok(Self {
            linear,
            pairs,
            attention: attention(budget),
            capacity,
            small_rows,
            budget: budget.map(SpecBudget::rows),
            model: hash(&fs::read(model.join("config.json"))?),
            draft: draft
                .map(|p| fs::read(p.join("config.json")).map(|b| hash(&b)))
                .transpose()?,
        })
    }
    pub fn path(&self, dir: &Path, key: &CacheKey) -> Result<PathBuf> {
        Ok(dir.join(format!(
            "closure-v1-{}-{}-{}-{}.json",
            key.device_uuid,
            key.driver_version,
            key.binary_version,
            hash(&serde_json::to_vec(self)?)
        )))
    }
}

impl Closure {
    pub fn new(write: bool, command: String) -> Self {
        Self {
            write,
            stage: Cell::new(Stage::Setup),
            measurements: Cell::new(0),
            captures: Cell::new(0),
            late_attempts: Cell::new(0),
            files: RefCell::new(BTreeSet::new()),
            command,
        }
    }
    pub fn preflight(&self, paths: &[PathBuf]) -> Result<()> {
        if self.write {
            return Ok(());
        }
        let missing: Vec<_> = paths
            .iter()
            .filter(|p| !p.is_file())
            .map(|p| p.display().to_string())
            .collect();
        ensure!(
            missing.is_empty(),
            "missing calibration keys/files:\n{}\nrun: {}",
            missing.join("\n"),
            self.command
        );
        Ok(())
    }
    pub fn access(&self, path: &Path) -> Result<bool> {
        ensure!(
            self.stage.get() == Stage::Setup,
            "calibration cache access after choices were frozen: {}",
            path.display()
        );
        self.files.borrow_mut().insert(path.to_owned());
        if path.is_file() {
            return Ok(true);
        }
        ensure!(
            self.write,
            "missing calibration key {}\nrun: {}",
            path.display(),
            self.command
        );
        Ok(false)
    }
    pub(super) fn measurement(&self) -> Result<()> {
        if self.stage.get() != Stage::Setup || !self.write {
            if self.stage.get() == Stage::Ready {
                self.late_attempts.set(self.late_attempts.get() + 1);
            }
            anyhow::bail!(
                "calibration measurement is forbidden in ordinary startup or after freeze; run: {}",
                self.command
            );
        }
        self.measurements.set(self.measurements.get() + 1);
        Ok(())
    }
    fn before_ready(&self, action: &str) -> Result<()> {
        if self.stage.get() == Stage::Ready {
            self.late_attempts.set(self.late_attempts.get() + 1);
        }
        ensure!(
            self.stage.get() == Stage::Frozen,
            "{action} requires frozen choices and must precede Ready"
        );
        Ok(())
    }
    pub fn bind_fixed(&self) -> Result<()> {
        if self.stage.get() == Stage::Ready {
            self.late_attempts.set(self.late_attempts.get() + 1);
        }
        ensure!(
            self.stage.get() == Stage::Setup || self.stage.get() == Stage::Frozen,
            "fixed operator binding must precede Ready"
        );
        Ok(())
    }
    pub fn capture(&self) -> Result<()> {
        self.before_ready("production Graph capture")?;
        self.captures.set(self.captures.get() + 1);
        Ok(())
    }
    /// A fixed operator plan has no measured choices or calibration files.
    pub fn freeze_fixed(&self) -> Result<()> {
        ensure!(
            self.stage.get() == Stage::Setup
                && self.measurements.get() == 0
                && self.files.borrow().is_empty(),
            "fixed plan cannot reuse measured calibration"
        );
        self.stage.set(Stage::Frozen);
        eprintln!(
            "{}",
            serde_json::json!({"fixed_operator_closure":{"selection_source":"fixed","measurements":0,"files":0}})
        );
        Ok(())
    }
    pub(super) fn finish(&self, profile: Profile, dir: &Path, key: CacheKey) -> Result<()> {
        ensure!(
            self.stage.get() == Stage::Setup,
            "closure was already frozen"
        );
        let path = profile.path(dir, &key)?;
        let files = self
            .files
            .borrow()
            .iter()
            .map(|p| {
                Ok((
                    p.file_name()
                        .context("cache filename missing")?
                        .to_str()
                        .context("cache filename is not UTF-8")?
                        .to_owned(),
                    hash(&fs::read(p)?),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = Manifest {
            schema: 1,
            key,
            profile,
            files,
        };
        if self.write {
            use std::io::Write;
            let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&tmp)?;
            file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
            file.sync_all()?;
            fs::rename(tmp, &path)?;
        } else {
            let expected: Manifest = serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("read closure {}", path.display()))?,
            )?;
            ensure!(
                expected == manifest,
                "calibration closure identity/files mismatch: {}\nrun: {}",
                path.display(),
                self.command
            );
        }
        self.stage.set(Stage::Frozen);
        eprintln!(
            "{}",
            serde_json::json!({"calibration_closure":{"path":path,"linear_keys":manifest.profile.linear.len(),"measurements":self.measurements.get(),"files":manifest.files.len(),"write":self.write}})
        );
        Ok(())
    }
    pub fn ready(&self, started: Instant) -> Result<()> {
        ensure!(
            self.stage.get() == Stage::Frozen && self.late_attempts.get() == 0,
            "invalid Ready transition"
        );
        ensure!(
            self.write || self.measurements.get() == 0,
            "ordinary startup measured candidates"
        );
        self.stage.set(Stage::Ready);
        self.assert_ready()?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_ready":{"wall_ms":started.elapsed().as_secs_f64()*1000.0,"measurements":self.measurements.get(),"production_captures":self.captures.get(),"late_attempts":self.late_attempts.get()}})
        );
        Ok(())
    }
    pub fn assert_ready(&self) -> Result<()> {
        ensure!(
            self.stage.get() == Stage::Ready && self.late_attempts.get() == 0,
            "backend is not Ready or attempted late setup"
        );
        Ok(())
    }
    pub(super) fn expect_captures(&self, expected: usize) -> Result<()> {
        ensure!(
            self.captures.get() == expected,
            "production Graph closure incomplete: expected {expected}, captured {}",
            self.captures.get()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_value() -> serde_json::Value {
        serde_json::json!({
            "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
            "num_hidden_layers":42,"num_attention_heads":16,"num_key_value_heads":2,
            "head_dim":128,"vocab_size":130560,"max_position_embeddings":131072,
            "rms_norm_eps":0.00001,"rope_theta":10000.0,"hidden_act":"silu",
            "tie_word_embeddings":false,"torch_dtype":"bfloat16","eos_token_id":[2]
        })
    }

    fn config() -> Result<Config> {
        Ok(serde_json::from_value(config_value())?)
    }

    fn draft_value() -> serde_json::Value {
        serde_json::json!({
            "model_type":"qwen3", "architectures":["Qwen3DSparkModel"], "dtype":"bfloat16",
            "hidden_size":2048, "intermediate_size":6144, "num_hidden_layers":5,
            "num_attention_heads":16, "num_key_value_heads":2, "head_dim":128,
            "vocab_size":130560, "draft_vocab_size":130560, "rms_norm_eps":1e-6,
            "block_size":7, "mask_token_id":75982, "target_layer_ids":[1,10,20,30,39],
            "num_target_layers":42, "markov_rank":256, "markov_head_type":"vanilla",
            "projector_type":"dspark", "rope_parameters":{"rope_theta":5000000.0,"rope_type":"default"}
        })
    }

    #[test]
    fn draft_linear_closure_covers_all_execution_sites() -> Result<()> {
        let source = include_str!("draft.rs");
        for (function, sites) in [("inject_block", 3), ("block", 6)] {
            let body = source
                .split_once(&format!("    fn {function}("))
                .context("draft execution function missing")?
                .1
                .split_once("\n    }\n")
                .context("draft execution function end missing")?
                .0;
            assert_eq!(body.matches("blas.linear(").count(), sites);
        }
        let mut target_value = config_value();
        target_value["num_key_value_heads"] = 4.into();
        let c: Config = serde_json::from_value(target_value.clone())?;
        let mut draft_value = draft_value();
        draft_value["intermediate_size"] = 8192.into();
        let root = std::env::temp_dir().join(format!(
            "flyingfish-draft-linears-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let model = root.join("target");
            let draft_dir = root.join("draft");
            fs::create_dir(&model)?;
            fs::create_dir(&draft_dir)?;
            fs::write(
                model.join("config.json"),
                serde_json::to_vec(&target_value)?,
            )?;
            fs::write(
                draft_dir.join("config.json"),
                serde_json::to_vec(&draft_value)?,
            )?;
            let draft = crate::dspark::DraftConfig::read(&draft_dir, &c)?;
            let (h, f, kv, rows) = (
                draft.hidden_size,
                draft.intermediate_size,
                draft.num_key_value_heads * draft.head_dim,
                draft.block_size,
            );
            let (mut cells, mut checks, mut defects) = (0, 0, 0);
            for budget in [
                SpecBudget::Chain,
                SpecBudget::Tree16,
                SpecBudget::Tree32,
                SpecBudget::Tree64,
            ] {
                for chunk in [1, 16, 256] {
                    let profile = Profile::new(
                        &c,
                        4096,
                        chunk,
                        Some(budget),
                        &model,
                        Some(&draft_dir),
                        Some(&draft),
                    )?;
                    let mut missing = BTreeSet::new();
                    for injected in 1..=injection_limit(small_rows(chunk, 4096)) {
                        for (site, rows, output, input) in [
                            (
                                "inject_block/fc",
                                injected,
                                h,
                                draft.target_layer_ids.len() * h,
                            ),
                            ("inject_block/k", injected, kv, h),
                            ("inject_block/v", injected, kv, h),
                            (
                                "block/qkv",
                                rows,
                                (draft.num_attention_heads + 2 * draft.num_key_value_heads)
                                    * draft.head_dim,
                                h,
                            ),
                            ("block/o", rows, h, h),
                            ("block/gate_up", rows, 2 * f, h),
                            ("block/down", rows, h, f),
                            ("block/target_head", rows, draft.vocab_size, h),
                            ("block/markov_w2", 1, draft.vocab_size, draft.markov_rank),
                        ] {
                            let shape = LinearShape {
                                rows,
                                output,
                                input,
                            };
                            if !profile.linear.contains(&shape)
                                || !profile.pairs.contains(&(output, input))
                            {
                                missing.insert((site, rows, output, input));
                                defects += 1;
                            }
                            checks += 1;
                        }
                    }
                    if !missing.is_empty() {
                        println!(
                            "DEFECT PR61_DRAFT_LINEAR budget={budget:?} chunk={chunk} missing={missing:?}"
                        );
                    }
                    cells += 1;
                }
            }
            println!(
                "PR61_DRAFT_LINEAR_CLOSURE sites=9 profile_cells={cells} shape_checks={checks} target_kv=4 draft_kv=2 target_intermediate=6144 draft_intermediate=8192 defects={defects} {}",
                if defects == 0 { "PASS" } else { "DEFECT" }
            );
            ensure!(defects == 0, "draft execution linears missing from closure");
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn draft_mlp_closure_covers_distinct_intermediate_size() -> Result<()> {
        let c = config()?;
        let mut draft_value = draft_value();
        let root = std::env::temp_dir().join(format!(
            "flyingfish-draft-closure-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let model = root.join("target");
            let draft_dir = root.join("draft");
            fs::create_dir(&model)?;
            fs::create_dir(&draft_dir)?;
            fs::write(
                model.join("config.json"),
                serde_json::to_vec(&config_value())?,
            )?;
            let mut cells = 0;
            for budget in [
                SpecBudget::Chain,
                SpecBudget::Tree16,
                SpecBudget::Tree32,
                SpecBudget::Tree64,
            ] {
                for chunk in [1, 16, 256] {
                    let make = |d: &crate::dspark::DraftConfig| {
                        Profile::new(
                            &c,
                            4096,
                            chunk,
                            Some(budget),
                            &model,
                            Some(&draft_dir),
                            Some(d),
                        )
                    };
                    draft_value["intermediate_size"] = c.intermediate_size.into();
                    fs::write(
                        draft_dir.join("config.json"),
                        serde_json::to_vec(&draft_value)?,
                    )?;
                    let draft = crate::dspark::DraftConfig::read(&draft_dir, &c)?;
                    let base = make(&draft)?;
                    draft_value["intermediate_size"] = 8192.into();
                    fs::write(
                        draft_dir.join("config.json"),
                        serde_json::to_vec(&draft_value)?,
                    )?;
                    let draft = crate::dspark::DraftConfig::read(&draft_dir, &c)?;
                    let distinct = make(&draft)?;
                    let mut expected_linear = base.linear.clone();
                    let mut expected_pairs = base.pairs.clone();
                    for (output, input) in [
                        (2 * draft.intermediate_size, c.hidden_size),
                        (c.hidden_size, draft.intermediate_size),
                    ] {
                        expected_linear.insert(LinearShape {
                            rows: 7,
                            output,
                            input,
                        });
                        expected_pairs.insert((output, input));
                    }
                    assert_eq!(distinct.linear, expected_linear);
                    assert_eq!(distinct.pairs, expected_pairs);
                    assert_eq!(distinct.linear.len(), base.linear.len() + 2);
                    assert_eq!(distinct.attention, base.attention);
                    cells += 1;
                }
            }
            println!(
                "PR61_DRAFT_MLP profile_cells={cells} target_intermediate=6144 draft_intermediate=8192 added_choices=2 added_pairs=2 PASS"
            );
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn shape_closure_covers_prefill_and_all_commits() -> Result<()> {
        let c = config()?;
        let draft_value = draft_value();
        let draft: crate::dspark::DraftConfig = serde_json::from_value(draft_value.clone())?;
        assert_eq!(draft.num_target_layers, 42);
        assert_eq!(draft.capture_width(), 10240);
        let draft_config = Some(&draft);
        let root = std::env::current_exe()?
            .parent()
            .context("test executable directory missing")?
            .join(format!(
                "profile-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ));
        fs::create_dir(&root)?;
        let identity = (|| -> Result<()> {
            let model = root.join("target");
            let draft_dir = root.join("draft");
            fs::create_dir(&model)?;
            fs::create_dir(&draft_dir)?;
            fs::write(
                model.join("config.json"),
                serde_json::to_vec(&config_value())?,
            )?;
            fs::write(
                draft_dir.join("config.json"),
                serde_json::to_vec(&draft_value)?,
            )?;
            let key = CacheKey {
                device_uuid: "test".into(),
                driver_version: "test".into(),
                binary_version: "test".into(),
            };
            for budget in [
                None,
                Some(SpecBudget::Chain),
                Some(SpecBudget::Tree16),
                Some(SpecBudget::Tree32),
                Some(SpecBudget::Tree64),
            ] {
                let make = |chunk| {
                    Profile::new(
                        &c,
                        33824,
                        chunk,
                        budget,
                        &model,
                        budget.map(|_| draft_dir.as_path()),
                        budget.map(|_| &draft),
                    )
                };
                let p16 = make(16)?;
                for chunk in [256, 32768] {
                    let other = make(chunk)?;
                    assert_eq!(p16, other);
                    assert_eq!(p16.path(&root, &key)?, other.path(&root, &key)?);
                }
                let p8 = make(8)?;
                assert_ne!(p8, p16);
                assert_ne!(p8.path(&root, &key)?, p16.path(&root, &key)?);
            }
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        identity?;

        assert_eq!(
            shapes(&c, small_rows(32768, 33824), None, None)?.0.len(),
            65
        );
        assert_eq!(
            shapes(
                &c,
                small_rows(32768, 33824),
                Some(SpecBudget::Chain),
                draft_config
            )?
            .0
            .len(),
            100
        );
        assert_eq!(
            shapes(
                &c,
                small_rows(32768, 33824),
                Some(SpecBudget::Tree16),
                draft_config
            )?
            .0
            .len(),
            100
        );
        for budget in [
            SpecBudget::Chain,
            SpecBudget::Tree16,
            SpecBudget::Tree32,
            SpecBudget::Tree64,
        ] {
            let (linear, pairs) = shapes(&c, small_rows(32768, 33824), Some(budget), draft_config)?;
            assert!(linear.contains(&LinearShape {
                rows: 1,
                output: c.vocab_size,
                input: draft.markov_rank
            }));
            assert!(pairs.contains(&(c.vocab_size, draft.markov_rank)));
            let kinds = attention(Some(budget));
            assert_eq!(kinds.contains(&Attention::Verify), !budget.is_tree());
            assert_eq!(
                kinds,
                vec![
                    Attention::Decode,
                    Attention::Draft,
                    if budget.is_tree() {
                        Attention::Tree
                    } else {
                        Attention::Verify
                    }
                ]
            );
        }
        assert_eq!(attention(None), vec![Attention::Decode]);
        for budget in [
            SpecBudget::Chain,
            SpecBudget::Tree16,
            SpecBudget::Tree32,
            SpecBudget::Tree64,
        ] {
            for chunk in [1, 8, 9, 15, 16, 32] {
                let limit = injection_limit(small_rows(chunk, 4096));
                let (linear, pairs) =
                    shapes(&c, small_rows(chunk, 4096), Some(budget), draft_config)?;
                let allowed = forward_rows(small_rows(chunk, 4096), Some(budget));
                let workspace = budget.workspace_rows(chunk);
                for rows in 1..=workspace {
                    let step = crate::backend::Step {
                        rows,
                        input: crate::backend::Input::Ids,
                        head: crate::backend::Head::Last,
                        mask: crate::backend::Mask::Causal,
                    };
                    let declared = rows > 16
                        || linear.contains(&LinearShape {
                            rows,
                            output: c.qkv_dim(),
                            input: c.hidden_size,
                        });
                    assert_eq!(step.check(workspace, rows, &allowed).is_ok(), declared);
                    assert_eq!(rows > 16 || allowed.contains(&rows), declared);
                }
                if budget.is_tree() {
                    for rows in 1..=budget.rows() {
                        let parents: Vec<_> =
                            (0..rows).map(|r| if r == 0 { -1 } else { 0 }).collect();
                        let positions: Vec<_> = (0..rows).map(|r| usize::from(r != 0)).collect();
                        let step = crate::backend::Step {
                            rows,
                            input: crate::backend::Input::Ids,
                            head: crate::backend::Head::All,
                            mask: crate::backend::Mask::Tree {
                                parents: &parents,
                                positions: &positions,
                            },
                        };
                        assert!(step.check(workspace, budget.rows(), &allowed).is_ok());
                    }
                }
                for rows in 1..=16 {
                    assert_eq!(
                        linear.contains(&LinearShape {
                            rows,
                            output: 2048,
                            input: 10240
                        }),
                        rows <= limit
                    );
                    assert_eq!(
                        linear.contains(&LinearShape {
                            rows,
                            output: 256,
                            input: 2048
                        }),
                        rows <= limit
                    );
                }
                assert!(pairs.contains(&(2048, 10240)) && pairs.contains(&(256, 2048)));
            }
        }
        let (small, _) = shapes(
            &c,
            small_rows(1, 4096),
            Some(SpecBudget::Tree16),
            draft_config,
        )?;
        for rows in 1..=8 {
            for (output, input) in [(2048, 10240), (256, 2048)] {
                assert!(small.contains(&LinearShape {
                    rows,
                    output,
                    input
                }));
            }
        }
        assert!(!small.contains(&LinearShape {
            rows: 9,
            output: 2048,
            input: 10240
        }));
        assert!(small.contains(&LinearShape {
            rows: 16,
            output: 2560,
            input: 2048
        }));
        let (wide, _) = shapes(
            &c,
            small_rows(32768, 33824),
            Some(SpecBudget::Tree64),
            draft_config,
        )?;
        assert!(wide.iter().all(|s| s.rows <= 16));
        Ok(())
    }

    #[test]
    fn calibration_excludes_unaligned_inputs() -> Result<()> {
        let mut c = config()?;
        for (hidden, intermediate) in [(384, 768), (512, 640)] {
            c.hidden_size = hidden;
            c.intermediate_size = intermediate;
            c.num_attention_heads = hidden / c.head_dim;
            c.num_key_value_heads = 1;
            let mut draft: crate::dspark::DraftConfig = serde_json::from_value(draft_value())?;
            draft.hidden_size = hidden;
            draft.intermediate_size = intermediate;
            draft.num_attention_heads = c.num_attention_heads;
            draft.num_key_value_heads = c.num_key_value_heads;
            draft.markov_rank = 128;
            for budget in [
                None,
                Some(SpecBudget::Chain),
                Some(SpecBudget::Tree16),
                Some(SpecBudget::Tree64),
            ] {
                let (linear, _) = shapes(&c, small_rows(32768, 33824), budget, Some(&draft))?;
                assert!(!linear.is_empty());
                for &shape in &linear {
                    super::super::linear_calibrate::check_extent(
                        shape,
                        shape.output * shape.input,
                        shape.rows * shape.input,
                    )?;
                }
                if !hidden.is_multiple_of(256) {
                    assert!(linear.iter().all(|s| s.output != c.vocab_size));
                }
                if !intermediate.is_multiple_of(256) {
                    assert!(
                        !linear
                            .iter()
                            .any(|s| s.output == hidden && s.input == intermediate)
                    );
                }
                assert!(!linear.iter().any(|s| s.input == 128));
            }
        }
        Ok(())
    }

    #[test]
    fn coverage_keeps_all_uncalibrated_pairs() -> Result<()> {
        let mut c = config()?;
        c.hidden_size = 384;
        c.intermediate_size = 640;
        c.num_attention_heads = 3;
        c.num_key_value_heads = 1;
        let mut draft: crate::dspark::DraftConfig = serde_json::from_value(draft_value())?;
        draft.hidden_size = c.hidden_size;
        draft.intermediate_size = c.intermediate_size;
        draft.num_attention_heads = c.num_attention_heads;
        draft.num_key_value_heads = c.num_key_value_heads;
        draft.markov_rank = 128;
        let base = BTreeSet::from([
            (c.qkv_dim(), 384),
            (384, 384),
            (1280, 384),
            (384, 640),
            (c.vocab_size, 384),
        ]);
        for budget in [
            None,
            Some(SpecBudget::Chain),
            Some(SpecBudget::Tree16),
            Some(SpecBudget::Tree64),
        ] {
            let (linear, pairs) = shapes(&c, small_rows(32768, 33824), budget, Some(&draft))?;
            assert!(linear.is_empty());
            let mut expected = base.clone();
            if budget.is_some() {
                expected.extend([(384, 1920), (128, 384)]);
                expected.insert((c.vocab_size, 128));
            }
            assert_eq!(pairs, expected);
        }
        Ok(())
    }
    #[test]
    fn missing_cache_is_read_only() -> Result<()> {
        let closure = Closure::new(false, "flyingfish calibrate".into());
        let missing = std::env::current_exe()?.join("missing-choice.json");
        let error = closure
            .preflight(std::slice::from_ref(&missing))
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing-choice.json") && error.contains("flyingfish calibrate"));
        assert!(closure.access(&missing).is_err());
        assert_eq!(closure.measurements.get(), 0);
        assert!(!missing.exists());
        Ok(())
    }
    #[test]
    fn complete_manifest_rejects_changed_cache() -> Result<()> {
        let exe = std::env::current_exe()?;
        let root = exe
            .parent()
            .context("test executable directory missing")?
            .join(format!(
                "closure-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let path = root.join("choice.json");
            fs::write(&path, b"original")?;
            let key = CacheKey {
                device_uuid: "test".into(),
                driver_version: "test".into(),
                binary_version: "test".into(),
            };
            let profile = || Profile {
                linear: BTreeSet::new(),
                pairs: BTreeSet::new(),
                attention: attention(Some(SpecBudget::Tree16)),
                capacity: 16,
                small_rows: 16,
                budget: Some(16),
                model: "test".into(),
                draft: Some("test".into()),
            };
            let writer = Closure::new(true, "calibrate".into());
            assert!(writer.access(&path)?);
            writer.finish(profile(), &root, key.clone())?;
            let reader = Closure::new(false, "calibrate".into());
            assert!(reader.access(&path)?);
            reader.finish(profile(), &root, key.clone())?;
            fs::write(&path, b"changed")?;
            let reader = Closure::new(false, "calibrate".into());
            assert!(reader.access(&path)?);
            assert!(reader.finish(profile(), &root, key).is_err());
            assert_eq!(reader.measurements.get(), 0);
            Ok(())
        })();
        fs::remove_dir_all(root)?;
        result
    }
    #[test]
    fn frozen_and_ready_reject_setup() -> Result<()> {
        let closure = Closure::new(false, "calibrate".into());
        assert!(closure.measurement().is_err());
        assert!(closure.capture().is_err());
        closure.stage.set(Stage::Frozen);
        closure.capture()?;
        assert!(closure.measurement().is_err());
        closure.ready(Instant::now())?;
        assert!(closure.capture().is_err());
        assert!(closure.assert_ready().is_err());
        Ok(())
    }
    #[test]
    fn fixed_binding_covers_setup_and_frozen_without_counting_captures() -> Result<()> {
        let closure = Closure::new(false, "fixed setup".into());
        closure.bind_fixed()?;
        closure.freeze_fixed()?;
        closure.bind_fixed()?;
        closure.expect_captures(0)?;
        closure.ready(Instant::now())?;
        assert!(closure.bind_fixed().is_err());
        assert!(closure.assert_ready().is_err());
        closure.expect_captures(0)?;
        Ok(())
    }
}
