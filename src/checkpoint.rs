//! Shared mmap/index audit, independent of device upload and model execution.
use anyhow::{Context, Result, bail, ensure};
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
    pub fn manifest(
        &self,
        spec: &crate::model_spec::ModelSpec,
        policy: LoadPolicy,
    ) -> Result<TensorManifest> {
        use crate::model_spec::Architecture;
        let label = match spec.architecture {
            Architecture::Qwen35Text => "Q38",
            Architecture::MiniCpmV47 => "MiniCPM",
            Architecture::Llama => bail!("checkpoint manifest requires a Qwen3.5-family spec"),
        };
        let mut expected = BTreeMap::new();
        Self::text_entries(spec, &mut expected)?;
        match spec.architecture {
            Architecture::Qwen35Text => {
                if policy.selects(TensorScope::Vision) {
                    spec.vision
                        .as_ref()
                        .context("vision load policy requires vision configuration")?
                        .validate(Architecture::Qwen35Text)?;
                }
                #[derive(Deserialize)]
                struct Skipped {
                    name: String,
                    dtype: Dtype,
                    shape: Vec<usize>,
                }
                // The closed-world skipped-tensor manifest is a required build input.
                for item in serde_json::from_str::<Vec<Skipped>>(include_str!(
                    "checkpoint/q38_skipped.json"
                ))? {
                    let scope = if item.name.starts_with("mtp.") {
                        TensorScope::Mtp
                    } else {
                        ensure!(
                            item.name.starts_with("model.visual."),
                            "unknown non-text scope"
                        );
                        TensorScope::Vision
                    };
                    put(&mut expected, item.name, item.dtype, item.shape, scope);
                }
            }
            Architecture::MiniCpmV47 => Self::minicpm_vision_entries(spec, &mut expected)?,
            Architecture::Llama => unreachable!(),
        }
        if policy.selects(TensorScope::Mtp)
            && !expected
                .values()
                .any(|(_, _, scope)| *scope == TensorScope::Mtp)
        {
            bail!("{label} checkpoint has no MTP component");
        }
        let planned = expected
            .iter()
            .try_fold(0usize, |sum, (name, (dtype, shape, scope))| {
                if !policy.selects_tensor(*scope, name) {
                    return Ok(sum);
                }
                sum.checked_add(tensor_bytes(*dtype, shape)?)
                    .context("resident size overflow")
            })?;
        self.assemble_manifest(
            label,
            expected,
            policy,
            planned,
            spec.linear_format.fp8_layout(),
        )
    }

    fn text_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        use crate::model_spec::Mixer;
        let hidden = spec.hidden as usize;
        let vocab = spec.vocab as usize;
        let head_dim = spec.head_dim as usize;
        let q_width = spec.q_width as usize;
        let kv_width = spec.kv_heads as usize * head_dim;
        let fp8 = spec.linear_format.fp8_layout();
        let dense_fp8 = fp8.filter(|layout| *layout == crate::fp8::Fp8ScaleLayout::Block128);
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            put(
                expected,
                name.into(),
                Dtype::BF16,
                vec![vocab, hidden],
                TensorScope::Text,
            );
        }
        put(
            expected,
            "model.language_model.norm.weight".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Text,
        );
        for (i, layer) in spec.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                put(
                    expected,
                    format!("{p}.{name}.weight"),
                    Dtype::BF16,
                    vec![hidden],
                    TensorScope::Text,
                );
            }
            match &spec.moe {
                Some(moe) => {
                    let experts = moe.experts as usize;
                    let expert_i = moe.expert_intermediate as usize;
                    let shared_i = moe.shared_intermediate as usize;
                    put(
                        expected,
                        format!("{p}.mlp.gate.weight"),
                        Dtype::BF16,
                        vec![experts, hidden],
                        TensorScope::Text,
                    );
                    if spec.expert_format == Some(crate::model_spec::ExpertFormat::Fp8W8a16) {
                        for expert in 0..experts {
                            for (role, n, k) in [
                                ("gate_proj", expert_i, hidden),
                                ("up_proj", expert_i, hidden),
                                ("down_proj", hidden, expert_i),
                            ] {
                                projection(
                                    expected,
                                    format!("{p}.mlp.experts.{expert}.{role}"),
                                    n,
                                    k,
                                    Some(crate::fp8::Fp8ScaleLayout::PerChannel),
                                );
                            }
                        }
                    } else {
                        put(
                            expected,
                            format!("{p}.mlp.experts.gate_up_proj"),
                            Dtype::BF16,
                            vec![experts, 2 * expert_i, hidden],
                            TensorScope::Text,
                        );
                        put(
                            expected,
                            format!("{p}.mlp.experts.down_proj"),
                            Dtype::BF16,
                            vec![experts, hidden, expert_i],
                            TensorScope::Text,
                        );
                    }
                    put(
                        expected,
                        format!("{p}.mlp.shared_expert.gate_proj.weight"),
                        Dtype::BF16,
                        vec![shared_i, hidden],
                        TensorScope::Text,
                    );
                    put(
                        expected,
                        format!("{p}.mlp.shared_expert.up_proj.weight"),
                        Dtype::BF16,
                        vec![shared_i, hidden],
                        TensorScope::Text,
                    );
                    put(
                        expected,
                        format!("{p}.mlp.shared_expert.down_proj.weight"),
                        Dtype::BF16,
                        vec![hidden, shared_i],
                        TensorScope::Text,
                    );
                    put(
                        expected,
                        format!("{p}.mlp.shared_expert_gate.weight"),
                        Dtype::BF16,
                        vec![1, hidden],
                        TensorScope::Text,
                    );
                }
                None => {
                    let intermediate = spec.intermediate as usize;
                    projection(
                        expected,
                        format!("{p}.mlp.gate_proj"),
                        intermediate,
                        hidden,
                        dense_fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.mlp.up_proj"),
                        intermediate,
                        hidden,
                        dense_fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.mlp.down_proj"),
                        hidden,
                        intermediate,
                        dense_fp8,
                    );
                }
            }
            match layer.mixer {
                Mixer::FullAttention => {
                    projection(
                        expected,
                        format!("{p}.self_attn.q_proj"),
                        2 * q_width,
                        hidden,
                        fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.self_attn.k_proj"),
                        kv_width,
                        hidden,
                        fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.self_attn.v_proj"),
                        kv_width,
                        hidden,
                        fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.self_attn.o_proj"),
                        hidden,
                        q_width,
                        fp8,
                    );
                    for name in ["q_norm", "k_norm"] {
                        put(
                            expected,
                            format!("{p}.self_attn.{name}.weight"),
                            Dtype::BF16,
                            vec![head_dim],
                            TensorScope::Text,
                        );
                    }
                }
                Mixer::Gdn => {
                    let g = spec.gdn.as_ref().context("GDN geometry missing")?;
                    let key_dim = g.key_heads as usize * g.key_head_dim as usize;
                    let value_heads = g.value_heads as usize;
                    let value_dim = value_heads * g.value_head_dim as usize;
                    let conv_dim = 2 * key_dim + value_dim;
                    projection(
                        expected,
                        format!("{p}.linear_attn.in_proj_qkv"),
                        conv_dim,
                        hidden,
                        dense_fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.linear_attn.in_proj_z"),
                        value_dim,
                        hidden,
                        dense_fp8,
                    );
                    projection(
                        expected,
                        format!("{p}.linear_attn.out_proj"),
                        hidden,
                        value_dim,
                        dense_fp8,
                    );
                    for name in ["in_proj_a", "in_proj_b"] {
                        put(
                            expected,
                            format!("{p}.linear_attn.{name}.weight"),
                            Dtype::BF16,
                            vec![value_heads, hidden],
                            TensorScope::Text,
                        );
                    }
                    put(
                        expected,
                        format!("{p}.linear_attn.conv1d.weight"),
                        Dtype::BF16,
                        vec![conv_dim, 1, g.conv_kernel as usize],
                        TensorScope::Text,
                    );
                    put(
                        expected,
                        format!("{p}.linear_attn.norm.weight"),
                        Dtype::BF16,
                        vec![g.value_head_dim as usize],
                        TensorScope::Text,
                    );
                    for name in ["A_log", "dt_bias"] {
                        put(
                            expected,
                            format!("{p}.linear_attn.{name}"),
                            Dtype::BF16,
                            vec![value_heads],
                            TensorScope::Text,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn minicpm_vision_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let vision = spec
            .vision
            .as_ref()
            .context("MiniCPM vision configuration missing")?;
        vision.validate(crate::model_spec::Architecture::MiniCpmV47)?;
        let vh = vision.hidden as usize;
        let vi = vision.intermediate as usize;
        let patch = vision.patch as usize;
        // MiniCPMV4_7ViTWindowAttentionMerger windows are 2x2; the DownsampleMLP stacks two 2x2 merges.
        let window_hidden = vh * 4;
        let window_intermediate = vi * 4;
        let merged = vh * 4;
        put(
            expected,
            "model.vision_tower.embeddings.patch_embedding.weight".into(),
            Dtype::BF16,
            vec![vh, vision.channels as usize, patch, patch],
            TensorScope::Vision,
        );
        put(
            expected,
            "model.vision_tower.embeddings.patch_embedding.bias".into(),
            Dtype::BF16,
            vec![vh],
            TensorScope::Vision,
        );
        put(
            expected,
            "model.vision_tower.embeddings.position_embedding.weight".into(),
            Dtype::BF16,
            vec![(vision.position_side as usize).pow(2), vh],
            TensorScope::Vision,
        );
        for name in ["post_layernorm.weight", "post_layernorm.bias"] {
            put(
                expected,
                format!("model.vision_tower.{name}"),
                Dtype::BF16,
                vec![vh],
                TensorScope::Vision,
            );
        }
        for i in 0..vision.depth as usize {
            let p = format!("model.vision_tower.encoder.layers.{i}");
            for name in ["layer_norm1", "layer_norm2"] {
                for suffix in ["weight", "bias"] {
                    put(
                        expected,
                        format!("{p}.{name}.{suffix}"),
                        Dtype::BF16,
                        vec![vh],
                        TensorScope::Vision,
                    );
                }
            }
            for (name, shape) in [
                ("mlp.fc1.weight", vec![vi, vh]),
                ("mlp.fc1.bias", vec![vi]),
                ("mlp.fc2.weight", vec![vh, vi]),
                ("mlp.fc2.bias", vec![vh]),
            ] {
                put(
                    expected,
                    format!("{p}.{name}"),
                    Dtype::BF16,
                    shape,
                    TensorScope::Vision,
                );
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                for (suffix, shape) in [("weight", vec![vh, vh]), ("bias", vec![vh])] {
                    put(
                        expected,
                        format!("{p}.self_attn.{name}.{suffix}"),
                        Dtype::BF16,
                        shape,
                        TensorScope::Vision,
                    );
                }
            }
        }
        let p = "model.vision_tower.vit_merger";
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            for (suffix, shape) in [("weight", vec![vh, vh]), ("bias", vec![vh])] {
                put(
                    expected,
                    format!("{p}.self_attn.{name}.{suffix}"),
                    Dtype::BF16,
                    shape,
                    TensorScope::Vision,
                );
            }
        }
        for (name, width) in [("layer_norm1", vh), ("pre_norm", window_hidden)] {
            for suffix in ["weight", "bias"] {
                put(
                    expected,
                    format!("{p}.{name}.{suffix}"),
                    Dtype::BF16,
                    vec![width],
                    TensorScope::Vision,
                );
            }
        }
        put(
            expected,
            format!("{p}.linear_1.weight"),
            Dtype::BF16,
            vec![window_intermediate, window_hidden],
            TensorScope::Vision,
        );
        put(
            expected,
            format!("{p}.linear_1.bias"),
            Dtype::BF16,
            vec![window_intermediate],
            TensorScope::Vision,
        );
        put(
            expected,
            format!("{p}.linear_2.weight"),
            Dtype::BF16,
            vec![vh, window_intermediate],
            TensorScope::Vision,
        );
        put(
            expected,
            format!("{p}.linear_2.bias"),
            Dtype::BF16,
            vec![vh],
            TensorScope::Vision,
        );
        let p = "model.merger.mlp.0";
        for suffix in ["weight", "bias"] {
            put(
                expected,
                format!("{p}.pre_norm.{suffix}"),
                Dtype::BF16,
                vec![merged],
                TensorScope::Vision,
            );
        }
        for (name, n, k) in [
            ("linear_1.weight", merged, merged),
            ("linear_2.weight", vision.output as usize, merged),
        ] {
            put(
                expected,
                format!("{p}.{name}"),
                Dtype::BF16,
                vec![n, k],
                TensorScope::Vision,
            );
        }
        for (name, width) in [
            ("linear_1.bias", merged),
            ("linear_2.bias", vision.output as usize),
        ] {
            put(
                expected,
                format!("{p}.{name}"),
                Dtype::BF16,
                vec![width],
                TensorScope::Vision,
            );
        }
        Ok(())
    }

    fn assemble_manifest(
        &self,
        label: &str,
        mut expected: BTreeMap<String, ExpectedTensor>,
        policy: LoadPolicy,
        expected_bytes: usize,
        fp8_layout: Option<crate::fp8::Fp8ScaleLayout>,
    ) -> Result<TensorManifest> {
        ensure!(
            expected.len() == self.index.weight_map.len(),
            "{label} tensor count mismatch"
        );
        let views = self.views()?;
        let mut entries = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for (name, shard) in &self.index.weight_map {
            let (dtype, shape, scope) = expected
                .remove(name)
                .with_context(|| format!("unknown {label} tensor {name}"))?;
            let load = policy.selects_tensor(scope, name);
            let t = views[shard].tensor(name)?;
            ensure!(
                t.dtype() == dtype && t.shape() == shape,
                "{label} dtype/shape mismatch: {name}"
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
                    "{}.{}",
                    name.strip_suffix(".weight")
                        .context("invalid FP8 tensor suffix")?,
                    fp8_layout
                        .context("FP8 tensor without scale layout")?
                        .scale_suffix()
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
        ensure!(
            expected.is_empty() && resident_bytes == expected_bytes,
            "{label} manifest incomplete"
        );
        Ok(TensorManifest {
            entries,
            resident_bytes,
        })
    }
}

type ExpectedTensor = (Dtype, Vec<usize>, TensorScope);

fn put(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    scope: TensorScope,
) {
    expected.insert(name, (dtype, shape, scope));
}

fn projection(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    base: String,
    n: usize,
    k: usize,
    fp8: Option<crate::fp8::Fp8ScaleLayout>,
) {
    let dtype = if fp8.is_some() {
        Dtype::F8_E4M3
    } else {
        Dtype::BF16
    };
    put(
        expected,
        format!("{base}.weight"),
        dtype,
        vec![n, k],
        TensorScope::Text,
    );
    if let Some(layout) = fp8 {
        put(
            expected,
            format!("{base}.{}", layout.scale_suffix()),
            Dtype::BF16,
            layout.scale_shape(n as u32, k as u32).to_vec(),
            TensorScope::Text,
        );
    }
}

fn tensor_bytes(dtype: Dtype, shape: &[usize]) -> Result<usize> {
    let bits = shape.iter().try_fold(dtype.bitsize(), |n, &d| {
        n.checked_mul(d).context("tensor size overflow")
    })?;
    ensure!(bits % 8 == 0, "tensor size is not a byte multiple");
    Ok(bits / 8)
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
impl TensorEntry {
    pub(crate) fn is_routed_expert(&self) -> bool {
        self.role.starts_with("mlp.experts.")
    }
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
