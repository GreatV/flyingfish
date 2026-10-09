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
    capacity: usize,
    chunk: usize,
    budget: Option<usize>,
    model: String,
    draft: Option<String>,
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

fn shapes(
    c: &Config,
    capacity: usize,
    chunk: usize,
    budget: Option<SpecBudget>,
    dimensions: Option<(usize, usize)>,
) -> Result<BTreeSet<LinearShape>> {
    let mut linear = BTreeSet::new();
    let mut layers = |rows| {
        for (output, input) in [
            (c.qkv_dim(), c.hidden_size),
            (c.hidden_size, c.hidden_size),
            (2 * c.intermediate_size, c.hidden_size),
            (c.hidden_size, c.intermediate_size),
        ] {
            if rows <= 16 && input.is_multiple_of(256) {
                linear.insert(LinearShape {
                    rows,
                    output,
                    input,
                });
            }
        }
    };
    for rows in 1..=chunk.min(capacity).min(16) {
        layers(rows);
    }
    if let Some(budget) = budget {
        layers(7);
        layers(budget.rows());
    }
    let head = |rows| LinearShape {
        rows,
        output: c.vocab_size,
        input: c.hidden_size,
    };
    linear.insert(head(1));
    if let Some(budget) = budget {
        let (context, rank) =
            dimensions.context("draft dimensions missing from closure declaration")?;
        linear.insert(head(7));
        if budget.rows() <= 16 {
            linear.insert(head(budget.rows()));
        }
        let max = chunk.min(capacity).clamp(8, 16);
        for rows in 1..=max {
            for (output, input) in [(c.hidden_size, context), (c.kv_dim(), c.hidden_size)] {
                linear.insert(LinearShape {
                    rows,
                    output,
                    input,
                });
            }
        }
        if !budget.is_tree() {
            linear.insert(LinearShape {
                rows: 1,
                output: c.vocab_size,
                input: rank,
            });
        }
    }
    Ok(linear)
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
        let linear = shapes(c, capacity, chunk, budget, dimensions)?;
        Ok(Self {
            linear,
            capacity,
            chunk,
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
    #[test]
    fn shape_closure_covers_prefill_and_all_commits() -> Result<()> {
        let c: Config = serde_json::from_value(serde_json::json!({
            "model_type":"llama", "hidden_size":2048, "intermediate_size":6144,
            "num_hidden_layers":42,"num_attention_heads":16,"num_key_value_heads":2,
            "head_dim":128,"vocab_size":130560,"max_position_embeddings":131072,
            "rms_norm_eps":0.00001,"rope_theta":10000.0,"hidden_act":"silu",
            "tie_word_embeddings":false,"torch_dtype":"bfloat16","eos_token_id":[2]
        }))?;
        let draft: crate::dspark::DraftConfig = serde_json::from_value(serde_json::json!({
            "model_type":"dflash","architectures":[],"dtype":"bfloat16",
            "hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":5,
            "num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,
            "vocab_size":130560,"draft_vocab_size":130560,"rms_norm_eps":0.00001,
            "block_size":8,"mask_token_id":0,"target_layer_ids":[1,10,20,30,39],
            "num_target_layers":42,"markov_rank":256,"markov_head_type":"markov",
            "projector_type":"linear","rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
        }))?;
        assert_eq!(draft.num_target_layers, 42);
        assert_eq!(draft.capture_width(), 10240);
        let dimensions = Some((draft.capture_width(), draft.markov_rank));
        assert_eq!(shapes(&c, 33824, 32768, None, None)?.len(), 65);
        assert_eq!(
            shapes(&c, 33824, 32768, Some(SpecBudget::Chain), dimensions)?.len(),
            100
        );
        assert_eq!(
            shapes(&c, 33824, 32768, Some(SpecBudget::Tree16), dimensions)?.len(),
            99
        );
        let small = shapes(&c, 4096, 1, Some(SpecBudget::Tree16), dimensions)?;
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
        let wide = shapes(&c, 33824, 32768, Some(SpecBudget::Tree64), dimensions)?;
        assert!(wide.iter().all(|s| s.rows <= 16));
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
                capacity: 16,
                chunk: 16,
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
