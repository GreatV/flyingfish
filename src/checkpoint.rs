//! Shared mmap/index audit, independent of device upload and model execution.
use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::{
    Deserialize, Deserializer,
    de::{Error, MapAccess, Visitor},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    path::{Component, Path},
};

fn unique_map<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> std::result::Result<BTreeMap<String, T>, D::Error> {
    struct Unique<T>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Unique<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("object with unique keys")
        }
        fn visit_map<A: MapAccess<'de>>(
            self,
            mut a: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((k, v)) = a.next_entry::<String, T>()? {
                if map.insert(k.clone(), v).is_some() {
                    return Err(A::Error::custom(format!("duplicate key {k}")));
                }
            }
            Ok(map)
        }
    }
    d.deserialize_map(Unique(std::marker::PhantomData))
}
#[derive(Clone, Deserialize)]
pub(crate) struct Index {
    #[serde(deserialize_with = "unique_map")]
    pub weight_map: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Header(#[serde(deserialize_with = "unique_map")] BTreeMap<String, serde_json::Value>);

pub struct MappedCheckpoint {
    pub(crate) index: Index,
    maps: BTreeMap<String, Mmap>,
    #[cfg(feature = "cuda")]
    files: BTreeMap<String, File>,
}

pub fn has_mtp_component(dir: &Path) -> Result<bool> {
    let path = dir.join("model.safetensors.index.json");
    let index = if path.is_file() {
        serde_json::from_slice::<Index>(&std::fs::read(path)?)?
    } else {
        MappedCheckpoint::open(dir)?.index
    };
    Ok(index.weight_map.keys().any(|name| name.starts_with("mtp.")))
}

impl MappedCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let indexed = dir.join("model.safetensors.index.json");
        let index = if indexed.is_file() {
            serde_json::from_slice::<Index>(&std::fs::read(indexed)?)?
        } else {
            let map =
                unsafe { MmapOptions::new().map(&File::open(dir.join("model.safetensors"))?)? };
            let tensors = SafeTensors::deserialize(&map)?;
            Index {
                weight_map: tensors
                    .names()
                    .into_iter()
                    .map(|s| (s.to_owned(), "model.safetensors".to_owned()))
                    .collect(),
            }
        };
        ensure!(!index.weight_map.is_empty(), "empty checkpoint index");
        let base = dir.canonicalize()?;
        let mut maps = BTreeMap::new();
        #[cfg(feature = "cuda")]
        let mut files = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for shard in index.weight_map.values().collect::<BTreeSet<_>>() {
            ensure!(
                Path::new(shard)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "unsafe shard path {shard}"
            );
            let path = base.join(shard).canonicalize()?;
            ensure!(
                path.starts_with(&base),
                "shard symlink escapes model directory"
            );
            let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let map = unsafe { MmapOptions::new().map(&file)? };
            ensure!(map.len() >= 8, "truncated safetensors header");
            let hlen = usize::try_from(u64::from_le_bytes(map[..8].try_into()?))?;
            let end = 8usize.checked_add(hlen).context("header size overflow")?;
            ensure!(
                hlen <= 16 * 1024 * 1024 && end <= map.len(),
                "invalid safetensors header size"
            );
            let header: Header = serde_json::from_slice(&map[8..end])?;
            let tensors = SafeTensors::deserialize(&map)?;
            ensure!(
                header.0.len()
                    == tensors.len() + usize::from(header.0.contains_key("__metadata__")),
                "header tensor count mismatch"
            );
            for name in tensors.names() {
                ensure!(
                    index.weight_map.get(name) == Some(shard),
                    "index/header mismatch for {name}"
                );
                ensure!(
                    seen.insert(name.to_owned()),
                    "duplicate tensor across shards: {name}"
                );
            }
            maps.insert(shard.clone(), map);
            #[cfg(feature = "cuda")]
            files.insert(shard.clone(), file);
        }
        ensure!(
            seen.len() == index.weight_map.len(),
            "index names missing from shard headers"
        );
        Ok(Self {
            index,
            maps,
            #[cfg(feature = "cuda")]
            files,
        })
    }
    /// Reads a validated mapped range through its file descriptor without faulting mapping pages.
    #[cfg(feature = "cuda")]
    pub(crate) fn read_mapped_range(
        &self,
        shard: &str,
        range: &[u8],
        destination: &mut [u8],
    ) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let map = self.maps.get(shard).context("unknown checkpoint shard")?;
        ensure!(
            range.len() == destination.len(),
            "checkpoint read length mismatch"
        );
        let offset = (range.as_ptr() as usize)
            .checked_sub(map.as_ptr() as usize)
            .context("range outside checkpoint mapping")?;
        ensure!(
            offset
                .checked_add(range.len())
                .is_some_and(|end| end <= map.len()),
            "checkpoint range exceeds mapping"
        );
        self.files
            .get(shard)
            .context("checkpoint file handle missing")?
            .read_exact_at(destination, u64::try_from(offset)?)?;
        Ok(())
    }
    pub(crate) fn views(&self) -> Result<BTreeMap<String, SafeTensors<'_>>> {
        self.maps
            .iter()
            .map(|(n, m)| Ok((n.clone(), SafeTensors::deserialize(m)?)))
            .collect()
    }
    pub fn q38_manifest(&self, spec: &crate::model_spec::ModelSpec) -> Result<TensorManifest> {
        self.q38_manifest_with_policy(spec, LoadPolicy::TextOnly)
    }

    pub fn q38_manifest_with_policy(
        &self,
        spec: &crate::model_spec::ModelSpec,
        policy: LoadPolicy,
    ) -> Result<TensorManifest> {
        use crate::model_spec::{Architecture, Mixer};
        ensure!(
            spec.architecture == Architecture::Qwen35Text,
            "Q38 manifest requires text spec"
        );
        if matches!(policy, LoadPolicy::TextWithVision | LoadPolicy::VisionOnly) {
            spec.vision
                .as_ref()
                .context("vision load policy requires vision configuration")?
                .validate()?;
        }
        let mut expected = BTreeMap::new();
        let mut put = |name: String, dtype: Dtype, shape: Vec<usize>, load: bool| {
            expected.insert(name, (dtype, shape, load));
        };
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            put(name.into(), Dtype::BF16, vec![248320, 5120], true);
        }
        put(
            "model.language_model.norm.weight".into(),
            Dtype::BF16,
            vec![5120],
            true,
        );
        for (i, layer) in spec.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                put(format!("{p}.{name}.weight"), Dtype::BF16, vec![5120], true);
            }
            let mut projections = vec![
                ("mlp.gate_proj", 17408usize, 5120usize),
                ("mlp.up_proj", 17408, 5120),
                ("mlp.down_proj", 5120, 17408),
            ];
            match layer.mixer {
                Mixer::FullAttention => {
                    projections.extend([
                        ("self_attn.q_proj", 12288, 5120),
                        ("self_attn.k_proj", 1024, 5120),
                        ("self_attn.v_proj", 1024, 5120),
                        ("self_attn.o_proj", 5120, 6144),
                    ]);
                    for name in ["q_norm", "k_norm"] {
                        put(
                            format!("{p}.self_attn.{name}.weight"),
                            Dtype::BF16,
                            vec![256],
                            true,
                        );
                    }
                }
                Mixer::Gdn => {
                    projections.extend([
                        ("linear_attn.in_proj_qkv", 10240, 5120),
                        ("linear_attn.in_proj_z", 6144, 5120),
                        ("linear_attn.out_proj", 5120, 6144),
                    ]);
                    for name in ["in_proj_a", "in_proj_b"] {
                        put(
                            format!("{p}.linear_attn.{name}.weight"),
                            Dtype::BF16,
                            vec![48, 5120],
                            true,
                        );
                    }
                    put(
                        format!("{p}.linear_attn.conv1d.weight"),
                        Dtype::BF16,
                        vec![10240, 1, 4],
                        true,
                    );
                    put(
                        format!("{p}.linear_attn.norm.weight"),
                        Dtype::BF16,
                        vec![128],
                        true,
                    );
                    for name in ["A_log", "dt_bias"] {
                        put(
                            format!("{p}.linear_attn.{name}"),
                            Dtype::BF16,
                            vec![48],
                            true,
                        );
                    }
                }
            }
            for (name, n, k) in projections {
                put(
                    format!("{p}.{name}.weight"),
                    Dtype::F8_E4M3,
                    vec![n, k],
                    true,
                );
                put(
                    format!("{p}.{name}.weight_scale_inv"),
                    Dtype::BF16,
                    vec![n.div_ceil(128), k.div_ceil(128)],
                    true,
                );
            }
        }
        #[derive(Deserialize)]
        struct Skipped {
            name: String,
            dtype: Dtype,
            shape: Vec<usize>,
        }
        // The closed-world skipped-tensor manifest is a required build input.
        for item in
            serde_json::from_str::<Vec<Skipped>>(include_str!("checkpoint/q38_skipped.json"))?
        {
            put(item.name, item.dtype, item.shape, false);
        }
        ensure!(
            expected.len() == self.index.weight_map.len(),
            "Q38 tensor count mismatch"
        );
        let views = self.views()?;
        let mut entries = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for (name, shard) in &self.index.weight_map {
            let (dtype, shape, text_tensor) = expected
                .remove(name)
                .with_context(|| format!("unknown Q38 tensor {name}"))?;
            let scope = if text_tensor {
                TensorScope::Text
            } else if name.starts_with("mtp.") {
                TensorScope::Mtp
            } else {
                ensure!(name.starts_with("model.visual."), "unknown non-text scope");
                TensorScope::Vision
            };
            let load = policy.selects_tensor(scope, name);
            let t = views[shard].tensor(name)?;
            ensure!(
                t.dtype() == dtype && t.shape() == shape,
                "Q38 dtype/shape mismatch: {name}"
            );
            if load {
                resident_bytes = resident_bytes
                    .checked_add(t.data().len())
                    .context("resident size overflow")?;
            }
            let map = &self.maps[shard];
            let data_start = 8u64
                .checked_add(u64::from_le_bytes(map[..8].try_into()?))
                .context("data start overflow")?;
            let file_offset = (t.data().as_ptr() as usize)
                .checked_sub(map.as_ptr() as usize)
                .context("tensor outside mapping")? as u64;
            let start = file_offset
                .checked_sub(data_start)
                .context("tensor overlaps header")?;
            let end = start
                .checked_add(t.data().len() as u64)
                .context("tensor offset overflow")?;
            let layer = name
                .strip_prefix("model.language_model.layers.")
                .and_then(|s| s.split('.').next())
                .map(str::parse::<u32>)
                .transpose()?;
            let role = if let Some(layer) = layer {
                name.strip_prefix(&format!("model.language_model.layers.{layer}."))
                    .unwrap_or(name)
                    .to_owned()
            } else {
                name.clone()
            };
            let scale_name = if dtype == Dtype::F8_E4M3 {
                Some(format!(
                    "{}.weight_scale_inv",
                    name.strip_suffix(".weight")
                        .context("invalid FP8 tensor suffix")?
                ))
            } else {
                None
            };
            entries.insert(
                name.clone(),
                TensorEntry {
                    dtype,
                    shape,
                    shard: shard.clone(),
                    bytes: t.data().len(),
                    load,
                    data_start,
                    data_offsets: [start, end],
                    scope,
                    layer,
                    role,
                    scale_name,
                },
            );
        }
        let expected_bytes = match policy {
            LoadPolicy::TextOnly => 29_468_003_328,
            LoadPolicy::TextWithVision => 30_389_463_520,
            LoadPolicy::VisionOnly => 921_460_192,
            LoadPolicy::TextWithMtp => 29_945_203_072,
            #[cfg(test)]
            LoadPolicy::MtpProbe => 5_562_793_344,
            LoadPolicy::MtpOnly => 477_199_744,
        };
        ensure!(
            expected.is_empty() && resident_bytes == expected_bytes,
            "Q38 manifest incomplete"
        );
        Ok(TensorManifest {
            entries,
            resident_bytes,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TensorScope {
    Text,
    Vision,
    Mtp,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadPolicy {
    #[default]
    TextOnly,
    TextWithVision,
    /// For the standalone vision component; cannot construct a text Engine.
    VisionOnly,
    TextWithMtp,
    /// Only MTP and the shared embedding/head; no target layers or final norm.
    #[cfg(test)]
    MtpProbe,
    /// Attach to an Engine that already owns the shared embedding/head.
    MtpOnly,
}

impl LoadPolicy {
    pub fn selects(self, scope: TensorScope) -> bool {
        #[cfg(test)]
        if self == Self::MtpProbe {
            return scope == TensorScope::Mtp;
        }
        matches!(
            (self, scope),
            (
                Self::TextOnly | Self::TextWithVision | Self::TextWithMtp,
                TensorScope::Text
            ) | (Self::TextWithVision | Self::VisionOnly, TensorScope::Vision)
                | (Self::TextWithMtp | Self::MtpOnly, TensorScope::Mtp)
        )
    }

    pub fn selects_tensor(self, scope: TensorScope, _name: &str) -> bool {
        #[cfg(test)]
        if self == Self::MtpProbe
            && scope == TensorScope::Text
            && matches!(
                _name,
                "model.language_model.embed_tokens.weight" | "lm_head.weight"
            )
        {
            return true;
        }
        self.selects(scope)
    }
}
#[derive(Debug, serde::Serialize)]
pub struct TensorEntry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub shard: String,
    pub bytes: usize,
    pub load: bool,
    pub data_start: u64,
    pub data_offsets: [u64; 2],
    pub scope: TensorScope,
    pub layer: Option<u32>,
    pub role: String,
    pub scale_name: Option<String>,
}
#[derive(Debug, serde::Serialize)]
pub struct TensorManifest {
    pub entries: BTreeMap<String, TensorEntry>,
    pub resident_bytes: usize,
}

#[cfg(test)]
mod component_tests {
    use super::*;

    #[test]
    fn mtp_presence_comes_from_checkpoint_tensor_names() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("flyingfish-mtp-component-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("model.safetensors.index.json");
        std::fs::write(
            &path,
            br#"{"weight_map":{"model.layers.0.weight":"model.safetensors"}}"#,
        )?;
        assert!(!has_mtp_component(&root)?);
        std::fs::write(
            &path,
            br#"{"weight_map":{"mtp.fc.weight":"model.safetensors"}}"#,
        )?;
        assert!(has_mtp_component(&root)?);
        std::fs::write(
            &path,
            br#"{"weight_map":{"mtp.fc.weight":"a","mtp.fc.weight":"b"}}"#,
        )?;
        assert!(has_mtp_component(&root).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
