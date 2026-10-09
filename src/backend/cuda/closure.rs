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
    dimensions: Option<(usize, usize)>,
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
        let (context, rank) =
            dimensions.context("draft dimensions missing from closure declaration")?;
        add(7, c.vocab_size, c.hidden_size);
        add(budget.rows(), c.vocab_size, c.hidden_size);
        let max = injection_limit(small_rows);
        for rows in 1..=max {
            for (output, input) in [(c.hidden_size, context), (c.kv_dim(), c.hidden_size)] {
                add(rows, output, input);
            }
        }
        add(1, c.vocab_size, rank);
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
        let dimensions = draft_config.map(|d| (d.capture_width(), d.markov_rank));
        let small_rows = small_rows(chunk, capacity);
        ensure!(small_rows > 0, "empty profile row bound");
        let (linear, pairs) = shapes(c, small_rows, budget, dimensions)?;
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
    pub fn capture(&self) -> Result<()> {
        if self.stage.get() == Stage::Ready {
            self.late_attempts.set(self.late_attempts.get() + 1);
        }
        ensure!(
            self.stage.get() == Stage::Frozen,
            "production Graph capture requires frozen choices and must precede Ready"
        );
        self.captures.set(self.captures.get() + 1);
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

    #[test]
    fn shape_closure_covers_prefill_and_all_commits() -> Result<()> {
        let c = config()?;
        let draft_value = serde_json::json!({
            "model_type":"dflash","architectures":[],"dtype":"bfloat16",
            "hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":5,
            "num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,
            "vocab_size":130560,"draft_vocab_size":130560,"rms_norm_eps":0.00001,
            "block_size":8,"mask_token_id":0,"target_layer_ids":[1,10,20,30,39],
            "num_target_layers":42,"markov_rank":256,"markov_head_type":"markov",
            "projector_type":"linear","rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
        });
        let draft: crate::dspark::DraftConfig = serde_json::from_value(draft_value.clone())?;
        assert_eq!(draft.num_target_layers, 42);
        assert_eq!(draft.capture_width(), 10240);
        let dimensions = Some((draft.capture_width(), draft.markov_rank));
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
                dimensions
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
                dimensions
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
            let (linear, pairs) = shapes(&c, small_rows(32768, 33824), Some(budget), dimensions)?;
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
                    shapes(&c, small_rows(chunk, 4096), Some(budget), dimensions)?;
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
            dimensions,
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
            dimensions,
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
            for budget in [
                None,
                Some(SpecBudget::Chain),
                Some(SpecBudget::Tree16),
                Some(SpecBudget::Tree64),
            ] {
                let (linear, _) = shapes(
                    &c,
                    small_rows(32768, 33824),
                    budget,
                    Some((5 * hidden, 128)),
                )?;
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
            let (linear, pairs) = shapes(&c, small_rows(32768, 33824), budget, Some((1920, 128)))?;
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
}
