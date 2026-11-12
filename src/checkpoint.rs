//! Shared mmap/index audit, independent of device upload and model execution.
use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::TensorView;
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

pub(crate) struct PublishedTensor<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

fn published_tensors(map: &[u8]) -> Result<BTreeMap<String, PublishedTensor<'_>>> {
    ensure!(map.len() >= 8, "truncated safetensors header");
    let length = usize::try_from(u64::from_le_bytes(map[..8].try_into()?))?;
    let end = 8usize.checked_add(length).context("header size overflow")?;
    ensure!(
        length <= 16 * 1024 * 1024 && end <= map.len(),
        "invalid safetensors header size"
    );
    let header: Header = serde_json::from_slice(&map[8..end])?;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in header.0 {
        if name == "__metadata__" {
            let _: BTreeMap<String, String> = serde_json::from_value(value)?;
            continue;
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Entry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [usize; 2],
        }
        let entry: Entry = serde_json::from_value(value)?;
        let bits = if entry.dtype == "C64" {
            64
        } else {
            serde_json::from_value::<Dtype>(serde_json::Value::String(entry.dtype.clone()))?
                .bitsize()
        };
        let elements = entry.shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).context("tensor shape overflow")
        })?;
        let total_bits = elements
            .checked_mul(bits)
            .context("tensor byte size overflow")?;
        let [start, stop] = entry.data_offsets;
        ensure!(
            total_bits.is_multiple_of(8) && stop.checked_sub(start) == Some(total_bits / 8),
            "invalid tensor byte span for {name}"
        );
        let data = map[end..]
            .get(start..stop)
            .with_context(|| format!("tensor range outside file: {name}"))?;
        ranges.push([start, stop]);
        tensors.insert(
            name,
            PublishedTensor {
                dtype: entry.dtype,
                shape: entry.shape,
                data,
            },
        );
    }
    ranges.sort_unstable();
    let mut next = 0;
    for [start, stop] in ranges {
        ensure!(start == next, "tensor ranges overlap or contain holes");
        next = stop;
    }
    ensure!(next == map.len() - end, "unindexed bytes in checkpoint");
    Ok(tensors)
}

pub struct MappedCheckpoint {
    pub(crate) index: Index,
    maps: BTreeMap<String, Mmap>,
    zip_tensors: Option<BTreeMap<String, StoredTensor>>,
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    files: BTreeMap<String, File>,
}

/// MTP tensors are present and the checkpoint architecture has a bound MTP decode path.
pub fn has_mtp_component(dir: &Path) -> Result<bool> {
    let path = dir.join("model.safetensors.index.json");
    let index = if path.is_file() {
        serde_json::from_slice::<Index>(&std::fs::read(path)?)?
    } else {
        MappedCheckpoint::open(dir)?.index
    };
    if !index.weight_map.keys().any(|name| {
        name.starts_with("mtp.")
            || (name.starts_with("model.language_model.layers.") && name.contains(".eh_proj."))
    }) {
        return Ok(false);
    }
    Ok(crate::model_spec::ModelSpec::read(dir)?
        .architecture
        .mtp_decode())
}

impl MappedCheckpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_component(dir, "model")
    }
    /// Opens a safetensors component stem or an explicit stored-ZIP `.pth` component.
    pub fn open_component(dir: &Path, stem: &str) -> Result<Self> {
        ensure!(
            Path::new(stem).components().count() == 1
                && matches!(
                    Path::new(stem).components().next(),
                    Some(Component::Normal(_))
                ),
            "checkpoint stem must be a single path component"
        );
        if Path::new(stem).extension().is_some_and(|s| s == "pth") {
            return Self::open_stored_zip(dir, stem);
        }
        let indexed = dir.join(format!("{stem}.safetensors.index.json"));
        let index = if indexed.is_file() {
            serde_json::from_slice::<Index>(&std::fs::read(indexed)?)?
        } else {
            let map = unsafe {
                MmapOptions::new().map(&File::open(dir.join(format!("{stem}.safetensors")))?)?
            };
            let tensors = published_tensors(&map)?;
            Index {
                weight_map: tensors
                    .keys()
                    .map(|s| (s.to_owned(), format!("{stem}.safetensors")))
                    .collect(),
            }
        };
        ensure!(!index.weight_map.is_empty(), "empty checkpoint index");
        let base = dir.canonicalize()?;
        let mut maps = BTreeMap::new();
        #[cfg(any(feature = "cuda", feature = "rocm"))]
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
            let tensors = published_tensors(&map)?;
            for name in tensors.keys() {
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
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            files.insert(shard.clone(), file);
        }
        ensure!(
            seen.len() == index.weight_map.len(),
            "index names missing from shard headers"
        );
        Ok(Self {
            index,
            maps,
            zip_tensors: None,
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            files,
        })
    }
    fn open_stored_zip(dir: &Path, name: &str) -> Result<Self> {
        let base = dir.canonicalize()?;
        let path = base.join(name).canonicalize()?;
        ensure!(
            path.starts_with(&base),
            "checkpoint symlink escapes directory"
        );
        let file = File::open(path)?;
        let map = unsafe { MmapOptions::new().map(&file)? };
        let tensors = stored_zip_tensors(&map)?;
        let index = Index {
            weight_map: tensors
                .keys()
                .map(|key| (key.clone(), name.to_owned()))
                .collect(),
        };
        Ok(Self {
            index,
            maps: BTreeMap::from([(name.to_owned(), map)]),
            zip_tensors: Some(tensors),
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            files: BTreeMap::from([(name.to_owned(), file)]),
        })
    }

    /// Opens the stored-ZIP `model_weights.ckpt` member of a `.nemo` tar archive.
    pub fn open_nemo(path: &Path) -> Result<Self> {
        ensure!(path.is_file(), "nemo archive missing: {}", path.display());
        let file = File::open(path)?;
        let map = unsafe { MmapOptions::new().map(&file)? };
        let member = tar_member(&map, "model_weights.ckpt")?;
        let mut tensors = stored_zip_tensors(&map[member.clone()])?;
        for tensor in tensors.values_mut() {
            tensor.range.start += member.start;
            tensor.range.end += member.start;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("nemo archive name")?
            .to_owned();
        let index = Index {
            weight_map: tensors
                .keys()
                .map(|key| (key.clone(), name.clone()))
                .collect(),
        };
        Ok(Self {
            index,
            maps: BTreeMap::from([(name.clone(), map)]),
            zip_tensors: Some(tensors),
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            files: BTreeMap::from([(name, file)]),
        })
    }
    pub(crate) fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let shard = self
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("checkpoint lacks {name}"))?;
        let map = self.maps.get(shard).context("unknown checkpoint shard")?;
        if let Some(tensors) = &self.zip_tensors {
            let tensor = tensors.get(name).context("ZIP tensor missing")?;
            return Ok(TensorView::new(
                Dtype::F32,
                tensor.shape.clone(),
                &map[tensor.range.clone()],
            )?);
        }
        Ok(SafeTensors::deserialize(map)?.tensor(name)?)
    }

    pub(crate) fn component_tensors(&self) -> Result<BTreeMap<String, PublishedTensor<'_>>> {
        ensure!(
            self.zip_tensors.is_none(),
            "component range inventory requires safetensors"
        );
        let mut result = BTreeMap::new();
        for map in self.maps.values() {
            result.extend(published_tensors(map)?);
        }
        Ok(result)
    }
    /// Reads a validated mapped range through its file descriptor without faulting mapping pages.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
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
    /// Host copy of one published F32 tensor with the given shape.
    #[cfg(any(feature = "cuda", feature = "rocm", test))]
    pub(crate) fn f32_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::F32 && tensor.shape() == shape,
            "{name} is {:?} {:?}, expected F32 {shape:?}",
            tensor.dtype(),
            tensor.shape()
        );
        Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub(crate) fn views(&self) -> Result<BTreeMap<String, SafeTensors<'_>>> {
        ensure!(
            self.zip_tensors.is_none(),
            "safetensors shard views requested for stored ZIP"
        );
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
        if spec.architecture == Architecture::LoopVl {
            ensure!(
                matches!(
                    policy,
                    LoadPolicy::TextOnly | LoadPolicy::TextWithVision | LoadPolicy::VisionOnly
                ),
                "LoopVL load policy requires text and/or vision"
            );
            let mut expected = BTreeMap::new();
            Self::loopvl_entries(&mut expected);
            let planned = expected
                .iter()
                .filter(|(name, (_, _, scope))| policy.selects_tensor(*scope, name))
                .try_fold(0usize, |sum, (_, (dtype, shape, _))| {
                    sum.checked_add(tensor_bytes(*dtype, shape)?)
                        .context("LoopVL resident size overflow")
                })?;
            return self.assemble_manifest("LoopVL", expected, policy, planned, None);
        }
        if spec.architecture == Architecture::EmbeddingGemma2 {
            let expected = self.gemma4_expected(spec)?;
            let planned = planned_bytes(&expected, policy)?;
            return self.assemble_manifest("EmbeddingGemma2", expected, policy, planned, None);
        }
        if spec.architecture == Architecture::Gemma4 {
            ensure!(
                matches!(
                    policy,
                    LoadPolicy::TextOnly
                        | LoadPolicy::TextWithVision
                        | LoadPolicy::VisionOnly
                        | LoadPolicy::AudioOnly
                ),
                "Gemma 4 load policy requires text, vision or audio"
            );
            let expected = self.gemma4_expected(spec)?;
            let planned = Self::planned_bytes(&expected, policy)?;
            return self.assemble_manifest("Gemma4", expected, policy, planned, None);
        }
        if spec.architecture == Architecture::DiffusionGemma {
            ensure!(
                policy == LoadPolicy::TextOnly,
                "DiffusionGemma text-only loading is required for DG1a"
            );
            let expected = self.diffusion_gemma_expected(spec)?;
            let planned = Self::planned_bytes(&expected, policy)?;
            return self.assemble_manifest("DiffusionGemma", expected, policy, planned, None);
        }
        if spec.architecture == Architecture::Glm5Next {
            let mut expected = BTreeMap::new();
            Self::glm5_next_entries(spec, &mut expected)?;
            let views = self.views()?;
            let mut vision_count = 0usize;
            let mut vision_bytes = 0usize;
            for (name, shard) in self
                .index
                .weight_map
                .iter()
                .filter(|(name, _)| name.starts_with("model.visual."))
            {
                let tensor = views[shard].tensor(name)?;
                put(
                    &mut expected,
                    name.clone(),
                    tensor.dtype(),
                    tensor.shape().to_vec(),
                    TensorScope::Vision,
                );
                vision_count += 1;
                vision_bytes = vision_bytes
                    .checked_add(tensor.data().len())
                    .context("vision payload size overflow")?;
            }
            ensure!(
                vision_count == 347 && vision_bytes == 1_127_254_016,
                "GLM-5.3 vision scope mismatch: tensors={vision_count} bytes={vision_bytes}"
            );
            let planned = planned_bytes(&expected, policy)?;
            return self.assemble_manifest(
                "GLM53",
                expected,
                policy,
                planned,
                spec.linear_format
                    .fp8_layout()
                    .map(|layout| layout.scale_suffix()),
            );
        }
        if spec.architecture == Architecture::DeepseekV41 {
            let mut expected = BTreeMap::new();
            Self::deepseek_v41_entries(spec, &mut expected)?;
            let planned = planned_bytes(&expected, policy)?;
            return self.assemble_manifest("DeepSeek41", expected, policy, planned, Some("scale"));
        }
        let label = match spec.architecture {
            Architecture::Qwen3 => "Qwen3",
            Architecture::Qwen35Text => "Q38",
            Architecture::MiniCpmV47 => "MiniCPM",
            Architecture::Flint => "Flint",
            Architecture::Clover => "Clover",
            Architecture::Qwen3Vl => "Qwen3-VL",
            Architecture::Llama => "Llama",
            Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => {
                bail!("checkpoint manifest requires a Qwen3.5-family spec")
            }
            Architecture::DeepseekV41 | Architecture::DiffusionGemma => {
                unreachable!("architecture has a dedicated manifest path")
            }
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
                let declared_mtp_layers = spec.mtp_layers;
                // The merger's output projection feeds the text hidden size:
                // its shapes derive from the spec, not the frozen file.
                if spec.vision.is_some() {
                    put(
                        &mut expected,
                        "model.visual.merger.linear_fc2.weight".into(),
                        Dtype::BF16,
                        vec![spec.hidden as usize, 4608],
                        TensorScope::Vision,
                    );
                    put(
                        &mut expected,
                        "model.visual.merger.linear_fc2.bias".into(),
                        Dtype::BF16,
                        vec![spec.hidden as usize],
                        TensorScope::Vision,
                    );
                }
                for item in serde_json::from_str::<Vec<Skipped>>(include_str!(
                    "checkpoint/q38_skipped.json"
                ))? {
                    if item.name.starts_with("mtp.") {
                        // The MTP skip set exists only when the publisher
                        // declares the decoder. Undeclared entries are
                        // omitted; the census still fails loudly if the
                        // checkpoint ships them (unexpected) or omits
                        // declared ones (missing).
                        if declared_mtp_layers == 0 {
                            continue;
                        }
                        put(
                            &mut expected,
                            item.name,
                            item.dtype,
                            item.shape,
                            TensorScope::Mtp,
                        );
                    } else {
                        ensure!(
                            item.name.starts_with("model.visual."),
                            "unknown non-text scope"
                        );
                        if spec.vision.is_some() {
                            put(
                                &mut expected,
                                item.name,
                                item.dtype,
                                item.shape,
                                TensorScope::Vision,
                            );
                        }
                    }
                }
            }
            Architecture::MiniCpmV47 => Self::minicpm_vision_entries(spec, &mut expected)?,
            Architecture::Qwen3Vl => Self::qwen3_vl_vision_entries(spec, &mut expected)?,
            Architecture::Flint
            | Architecture::Clover
            | Architecture::Qwen3
            | Architecture::Llama => {}
            Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2
            | Architecture::DiffusionGemma => unreachable!(),
            Architecture::DeepseekV41 => {
                unreachable!("DeepSeek-V4.1 has a dedicated manifest path")
            }
        }
        if policy.selects(TensorScope::Mtp)
            && !expected
                .values()
                .any(|(_, _, scope)| *scope == TensorScope::Mtp)
        {
            bail!("{label} checkpoint has no MTP component");
        }
        let planned = planned_bytes(&expected, policy)?;
        self.assemble_manifest(
            label,
            expected,
            policy,
            planned,
            spec.linear_format
                .fp8_layout()
                .map(|layout| layout.scale_suffix()),
        )
    }

    fn loopvl_entries(expected: &mut BTreeMap<String, ExpectedTensor>) {
        use TensorScope::{Text, Vision};
        let mut add = |name: String, shape: Vec<usize>, dtype, scope| {
            put(expected, name, dtype, shape, scope)
        };
        for stack in ["L", "H"] {
            for layer in 0..16 {
                let p = format!("language_model.model.{stack}_module.layers.{layer}");
                for role in ["q", "k", "v", "o", "gate"] {
                    add(
                        format!("{p}.self_attn.{role}_proj.weight"),
                        vec![1536, 1536],
                        Dtype::BF16,
                        Text,
                    );
                }
                for role in ["gate", "up"] {
                    add(
                        format!("{p}.mlp.{role}_proj.weight"),
                        vec![4096, 1536],
                        Dtype::BF16,
                        Text,
                    );
                }
                add(
                    format!("{p}.mlp.down_proj.weight"),
                    vec![1536, 4096],
                    Dtype::BF16,
                    Text,
                );
            }
        }
        for (name, shape) in [
            (
                "language_model.model.embed_tokens.weight",
                vec![65536, 1536],
            ),
            ("language_model.lm_head.weight", vec![65536, 1536]),
            ("language_model.model.z_L_init", vec![1536]),
        ] {
            add(name.into(), shape, Dtype::BF16, Text);
        }
        for (name, shape) in [
            ("language_model.model.visual_anchor.raw_alpha", vec![2]),
            ("language_model.model.visual_gate.out_proj.bias", vec![1]),
            (
                "language_model.model.visual_gate.out_proj.weight",
                vec![1, 256],
            ),
            (
                "language_model.model.visual_gate.query_proj.weight",
                vec![256, 1536],
            ),
            (
                "language_model.model.visual_gate.visual_proj.weight",
                vec![256, 1536],
            ),
            ("projector.fc1.weight", vec![1536, 1024]),
            ("projector.fc1.bias", vec![1536]),
            ("projector.fc2.weight", vec![1536, 1536]),
            ("projector.fc2.bias", vec![1536]),
            ("projector.visual_embedding_scale", vec![]),
        ] {
            add(name.into(), shape, Dtype::F32, Vision);
        }
        for layer in 0..28 {
            let p = format!("vision_encoder.model.encoder.layers.{layer}");
            for (role, shape) in [
                ("input_layernorm.weight", vec![1024]),
                ("post_attention_layernorm.weight", vec![1024]),
                ("self_attn.q_norm.weight", vec![128]),
                ("self_attn.k_norm.weight", vec![128]),
                ("self_attn.q_proj.weight", vec![2048, 1024]),
                ("self_attn.k_proj.weight", vec![1024, 1024]),
                ("self_attn.v_proj.weight", vec![1024, 1024]),
                ("self_attn.o_proj.weight", vec![1024, 2048]),
                ("mlp.gate_proj.weight", vec![3072, 1024]),
                ("mlp.up_proj.weight", vec![3072, 1024]),
                ("mlp.down_proj.weight", vec![1024, 3072]),
            ] {
                add(format!("{p}.{role}"), shape, Dtype::BF16, Vision);
            }
        }
        for (name, shape) in [
            ("embeddings.patch_embedding.weight", vec![1024, 3, 14, 14]),
            ("embeddings.patch_embedding.bias", vec![1024]),
            ("encoder.norm.weight", vec![1024]),
        ] {
            add(
                format!("vision_encoder.model.{name}"),
                shape,
                Dtype::BF16,
                Vision,
            );
        }
    }

    fn gemma4_expected(
        &self,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<BTreeMap<String, ExpectedTensor>> {
        let mut expected = BTreeMap::new();
        Self::gemma4_entries(spec, &mut expected)?;
        if let Some(vision) = &spec.vision {
            if vision.unified.is_some() {
                Self::unified_vision_entries(vision, &mut expected)?;
            } else {
                let (tower, projection) =
                    if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
                        ("vision_tower", "embed_vision.embedding_projection.weight")
                    } else {
                        (
                            "model.vision_tower",
                            "model.embed_vision.embedding_projection.weight",
                        )
                    };
                Self::axial_vision_entries(vision, tower, projection, &mut expected)?;
            }
        }
        Self::audio_entries(spec, &mut expected)?;
        let views = self.views()?;
        for (name, shard) in &self.index.weight_map {
            let scope = if spec.vision.is_none()
                && (name.starts_with("model.vision_tower.")
                    || name.starts_with("model.embed_vision.")
                    || name.starts_with("model.vision_embedder.")
                    || name.starts_with("vision_tower.")
                    || name.starts_with("embed_vision."))
            {
                TensorScope::Vision
            } else if spec.audio.is_none()
                && (name.starts_with("model.audio_tower.")
                    || name.starts_with("model.embed_audio.")
                    || name.starts_with("audio_tower.")
                    || name.starts_with("embed_audio."))
            {
                TensorScope::Audio
            } else {
                continue;
            };
            let tensor = views[shard].tensor(name)?;
            put(
                &mut expected,
                name.clone(),
                tensor.dtype(),
                tensor.shape().to_vec(),
                scope,
            );
        }
        Ok(expected)
    }

    fn audio_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let Some(audio) = &spec.audio else {
            return Ok(());
        };
        let a = match audio {
            crate::model_spec::AudioConfig::Raw(a) => {
                put(
                    expected,
                    "model.embed_audio.embedding_projection.weight".into(),
                    Dtype::BF16,
                    vec![a.output as usize, a.frame as usize],
                    TensorScope::Audio,
                );
                return Ok(());
            }
            crate::model_spec::AudioConfig::Encoder(a) => a,
        };
        let h = a.hidden as usize;
        let mut add = |name, shape| put(expected, name, Dtype::BF16, shape, TensorScope::Audio);
        let p = if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
            "audio_tower"
        } else {
            "model.audio_tower"
        };
        let [c0, c1] = a.subsample_channels.map(|n| n as usize);
        for (i, input, output) in [(0, 1, c0), (1, c0, c1)] {
            add(
                format!("{p}.subsample_conv_projection.layer{i}.conv.weight"),
                vec![output, input, 3, 3],
            );
            add(
                format!("{p}.subsample_conv_projection.layer{i}.norm.weight"),
                vec![output],
            );
        }
        add(
            format!("{p}.subsample_conv_projection.input_proj_linear.weight"),
            vec![h, 32 * c1],
        );
        add(
            format!("{p}.output_proj.weight"),
            vec![a.output as usize, h],
        );
        add(format!("{p}.output_proj.bias"), vec![a.output as usize]);
        add(
            if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
                "embed_audio.embedding_projection.weight".into()
            } else {
                "model.embed_audio.embedding_projection.weight".into()
            },
            vec![spec.hidden as usize, a.output as usize],
        );
        for layer in 0..a.layers {
            let p = format!("{p}.layers.{layer}");
            for (name, shape) in [
                ("feed_forward1.ffw_layer_1", vec![4 * h, h]),
                ("feed_forward1.ffw_layer_2", vec![h, 4 * h]),
                ("feed_forward2.ffw_layer_1", vec![4 * h, h]),
                ("feed_forward2.ffw_layer_2", vec![h, 4 * h]),
                ("self_attn.q_proj", vec![h, h]),
                ("self_attn.k_proj", vec![h, h]),
                ("self_attn.v_proj", vec![h, h]),
                ("self_attn.post", vec![h, h]),
                ("lconv1d.linear_start", vec![2 * h, h]),
                ("lconv1d.linear_end", vec![h, h]),
            ] {
                add(format!("{p}.{name}.linear.weight"), shape);
                for bound in ["input_min", "input_max", "output_min", "output_max"] {
                    add(format!("{p}.{name}.{bound}"), vec![]);
                }
            }
            for name in [
                "feed_forward1.pre_layer_norm",
                "feed_forward1.post_layer_norm",
                "feed_forward2.pre_layer_norm",
                "feed_forward2.post_layer_norm",
                "lconv1d.pre_layer_norm",
                "lconv1d.conv_norm",
                "norm_pre_attn",
                "norm_post_attn",
                "norm_out",
            ] {
                add(format!("{p}.{name}.weight"), vec![h]);
            }
            add(format!("{p}.self_attn.relative_k_proj.weight"), vec![h, h]);
            add(
                format!("{p}.self_attn.per_dim_scale"),
                vec![h / a.heads as usize],
            );
            add(
                format!("{p}.lconv1d.depthwise_conv1d.weight"),
                vec![h, 1, a.conv_kernel as usize],
            );
        }
        Ok(())
    }

    fn unified_vision_entries(
        vision: &crate::model_spec::VisionConfig,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        vision.validate(crate::model_spec::Architecture::Gemma4)?;
        ensure!(
            vision.unified.is_some(),
            "unified vision configuration missing"
        );
        let h = vision.hidden as usize;
        let patch = vision.patch_elements(crate::model_spec::Architecture::Gemma4)?;
        let mut add = |name: &str, shape: Vec<usize>| {
            put(
                expected,
                name.into(),
                Dtype::BF16,
                shape,
                TensorScope::Vision,
            );
        };
        for (name, shape) in [
            ("model.vision_embedder.patch_dense.weight", vec![h, patch]),
            ("model.vision_embedder.patch_dense.bias", vec![h]),
            (
                "model.vision_embedder.pos_embedding",
                vec![vision.position_side as usize, 2, h],
            ),
            (
                "model.embed_vision.embedding_projection.weight",
                vec![vision.output as usize, h],
            ),
        ] {
            add(name, shape);
        }
        for (name, width) in [("patch_ln1", patch), ("patch_ln2", h), ("pos_norm", h)] {
            for suffix in ["weight", "bias"] {
                add(
                    &format!("model.vision_embedder.{name}.{suffix}"),
                    vec![width],
                );
            }
        }
        Ok(())
    }

    pub(crate) fn axial_vision_entries(
        vision: &crate::model_spec::VisionConfig,
        tower_prefix: &str,
        projection_name: &str,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        vision.validate(crate::model_spec::Architecture::Gemma4)?;
        let axial = vision
            .axial
            .as_ref()
            .context("axial vision configuration missing")?;
        let h = vision.hidden as usize;
        let d = axial.head_dim as usize;
        let i = vision.intermediate as usize;
        let mut add = |name: String, shape: Vec<usize>| {
            put(expected, name, Dtype::BF16, shape, TensorScope::Vision);
        };
        for (name, shape) in [
            (
                format!("{tower_prefix}.patch_embedder.input_proj.weight"),
                vec![
                    h,
                    vision.patch_elements(crate::model_spec::Architecture::Gemma4)?,
                ],
            ),
            (
                format!("{tower_prefix}.patch_embedder.position_embedding_table"),
                vec![2, vision.position_side as usize, h],
            ),
            (projection_name.to_owned(), vec![vision.output as usize, h]),
        ] {
            add(name, shape);
        }
        if axial.standardize {
            for name in ["std_bias", "std_scale"] {
                add(format!("{tower_prefix}.{name}"), vec![h]);
            }
        }
        for layer in 0..vision.depth {
            let prefix = format!("{tower_prefix}.encoder.layers.{layer}");
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                add(format!("{prefix}.{name}.weight"), vec![h]);
            }
            for name in ["q_norm", "k_norm"] {
                add(format!("{prefix}.self_attn.{name}.weight"), vec![d]);
            }
            for (name, shape) in [
                ("self_attn.q_proj", vec![h, h]),
                ("self_attn.k_proj", vec![h, h]),
                ("self_attn.v_proj", vec![h, h]),
                ("self_attn.o_proj", vec![h, h]),
                ("mlp.gate_proj", vec![i, h]),
                ("mlp.up_proj", vec![i, h]),
                ("mlp.down_proj", vec![h, i]),
            ] {
                add(format!("{prefix}.{name}.linear.weight"), shape);
                if axial.clipped_linears {
                    for bound in ["input_min", "input_max", "output_min", "output_max"] {
                        add(format!("{prefix}.{name}.{bound}"), vec![]);
                    }
                }
            }
        }
        Ok(())
    }

    fn planned_bytes(
        expected: &BTreeMap<String, ExpectedTensor>,
        policy: LoadPolicy,
    ) -> Result<usize> {
        expected
            .iter()
            .try_fold(0usize, |sum, (name, (dtype, shape, scope))| {
                if !policy.selects_tensor(*scope, name) {
                    return Ok(sum);
                }
                sum.checked_add(tensor_bytes(*dtype, shape)?)
                    .context("resident size overflow")
            })
    }

    fn gemma4_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let hidden = spec.hidden as usize;
        let depth = spec.layers.len();
        let ple = spec.per_layer_input.map(|width| width as usize);
        let q_heads = spec.q_heads as usize;
        let text = if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
            "language_model"
        } else {
            "model.language_model"
        };
        let mut add = |name: String, shape: Vec<usize>, scope| {
            put(expected, name, Dtype::BF16, shape, scope);
        };
        add(
            format!("{text}.embed_tokens.weight"),
            vec![spec.vocab as usize, hidden],
            TensorScope::Text,
        );
        if let Some(ple) = ple {
            let module = if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
                "ple."
            } else {
                ""
            };
            if spec.architecture != crate::model_spec::Architecture::EmbeddingGemma2 {
                add(
                    format!("{text}.embed_tokens_per_layer.weight"),
                    vec![spec.vocab as usize, depth * ple],
                    TensorScope::Text,
                );
            }
            for (name, shape) in [
                ("per_layer_model_projection", vec![depth * ple, hidden]),
                ("per_layer_projection_norm", vec![ple]),
            ] {
                add(
                    format!("{text}.{module}{name}.weight"),
                    shape,
                    TensorScope::Text,
                );
            }
        }
        add(
            format!("{text}.norm.weight"),
            vec![hidden],
            TensorScope::Text,
        );
        if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
            add(
                format!("{text}.embedding_projection.weight"),
                vec![768, hidden],
                TensorScope::Text,
            );
        }
        for (i, layer) in spec.layers.iter().enumerate() {
            let g = layer
                .geometry
                .with_context(|| format!("Gemma 4 layer {i} has no geometry"))?;
            let (head_dim, inter) = (g.head_dim as usize, g.intermediate as usize);
            let kv_heads = g.kv_heads.unwrap_or(spec.kv_heads) as usize;
            let p = format!("{text}.layers.{i}");
            let own = if g.kv_reader {
                TensorScope::Dormant
            } else {
                TensorScope::Text
            };
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                add(
                    format!("{p}.{name}.weight"),
                    vec![hidden],
                    TensorScope::Text,
                );
            }
            if let Some(ple) = ple {
                let block = if spec.architecture == crate::model_spec::Architecture::EmbeddingGemma2
                {
                    ".ple_block"
                } else {
                    ""
                };
                for (name, shape) in [
                    ("post_per_layer_input_norm", vec![hidden]),
                    ("per_layer_input_gate", vec![ple, hidden]),
                    ("per_layer_projection", vec![hidden, ple]),
                ] {
                    add(
                        format!("{p}{block}.{name}.weight"),
                        shape,
                        TensorScope::Text,
                    );
                }
            }
            add(format!("{p}.layer_scalar"), vec![1], TensorScope::Text);
            for (name, shape) in [
                ("mlp.gate_proj", vec![inter, hidden]),
                ("mlp.up_proj", vec![inter, hidden]),
                ("mlp.down_proj", vec![hidden, inter]),
                ("self_attn.q_proj", vec![q_heads * head_dim, hidden]),
                ("self_attn.o_proj", vec![hidden, q_heads * head_dim]),
            ] {
                add(format!("{p}.{name}.weight"), shape, TensorScope::Text);
            }
            add(
                format!("{p}.self_attn.q_norm.weight"),
                vec![head_dim],
                TensorScope::Text,
            );
            for (name, shape) in [
                ("self_attn.k_proj", vec![kv_heads * head_dim, hidden]),
                ("self_attn.k_norm", vec![head_dim]),
            ] {
                add(format!("{p}.{name}.weight"), shape, own);
            }
            if ple.is_some() || layer.sliding_window.is_some() {
                add(
                    format!("{p}.self_attn.v_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                    own,
                );
            }
        }
        Ok(())
    }

    fn diffusion_gemma_expected(
        &self,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<BTreeMap<String, ExpectedTensor>> {
        let mut expected = BTreeMap::new();
        Self::diffusion_gemma_entries(spec, &mut expected)?;
        let views = self.views()?;
        for (name, shard) in &self.index.weight_map {
            let scope = if name.starts_with("model.encoder.vision_tower.")
                || name.starts_with("model.encoder.embed_vision.")
            {
                TensorScope::Vision
            } else {
                continue;
            };
            let tensor = views[shard].tensor(name)?;
            put(
                &mut expected,
                name.clone(),
                tensor.dtype(),
                tensor.shape().to_vec(),
                scope,
            );
        }
        Ok(expected)
    }

    fn diffusion_gemma_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let moe = spec
            .moe
            .as_ref()
            .context("DiffusionGemma spec is missing its MoE configuration")?;
        let hidden = spec.hidden as usize;
        let dense = spec.intermediate as usize;
        let experts = moe.experts as usize;
        let expert_i = moe.expert_intermediate as usize;
        let canvas = spec
            .canvas_length
            .context("DiffusionGemma canvas length missing from the spec")?
            as usize;
        ensure!(
            canvas > 0 && spec.vocab == 262_144,
            "DiffusionGemma manifest requires a positive canvas and the published vocabulary"
        );
        let q_heads = spec.q_heads as usize;
        let text = "model.decoder";
        let mut add = |name: String, shape: Vec<usize>, scope| {
            put(expected, name, Dtype::BF16, shape, scope);
        };
        add(
            format!("{text}.embed_tokens.weight"),
            vec![spec.vocab as usize, hidden],
            TensorScope::Text,
        );
        add(
            format!("{text}.norm.weight"),
            vec![hidden],
            TensorScope::Text,
        );
        for (name, shape) in [
            ("self_conditioning.pre_norm", vec![hidden]),
            ("self_conditioning.gate_proj", vec![dense, hidden]),
            ("self_conditioning.up_proj", vec![dense, hidden]),
            ("self_conditioning.down_proj", vec![hidden, dense]),
        ] {
            add(format!("{text}.{name}.weight"), shape, TensorScope::Text);
        }
        for (i, layer) in spec.layers.iter().enumerate() {
            let g = layer
                .geometry
                .with_context(|| format!("DiffusionGemma layer {i} has no geometry"))?;
            let head_dim = g.head_dim as usize;
            let kv_heads = g.kv_heads.unwrap_or(spec.kv_heads) as usize;
            let p = format!("{text}.layers.{i}");
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
            ] {
                add(
                    format!("{p}.{name}.weight"),
                    vec![hidden],
                    TensorScope::Text,
                );
            }
            add(format!("{p}.layer_scalar"), vec![1], TensorScope::Text);
            add(
                format!("model.encoder.language_model.layers.{i}.layer_scalar"),
                vec![1],
                TensorScope::Text,
            );
            for (name, shape) in [
                ("self_attn.q_proj", vec![q_heads * head_dim, hidden]),
                ("self_attn.o_proj", vec![hidden, q_heads * head_dim]),
                ("self_attn.q_norm", vec![head_dim]),
                ("self_attn.k_proj", vec![kv_heads * head_dim, hidden]),
                ("self_attn.k_norm", vec![head_dim]),
                ("mlp.gate_proj", vec![dense, hidden]),
                ("mlp.up_proj", vec![dense, hidden]),
                ("mlp.down_proj", vec![hidden, dense]),
                ("router.proj", vec![experts, hidden]),
            ] {
                add(format!("{p}.{name}.weight"), shape, TensorScope::Text);
            }
            for (name, shape) in [
                ("router.scale", vec![hidden]),
                ("router.per_expert_scale", vec![experts]),
                ("experts.gate_up_proj", vec![experts, 2 * expert_i, hidden]),
                ("experts.down_proj", vec![experts, hidden, expert_i]),
            ] {
                add(format!("{p}.{name}"), shape, TensorScope::Text);
            }
            if layer.sliding_window.is_some() {
                add(
                    format!("{p}.self_attn.v_proj.weight"),
                    vec![kv_heads * head_dim, hidden],
                    TensorScope::Text,
                );
            }
        }
        Ok(())
    }

    fn text_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        use crate::model_spec::Architecture;
        use crate::model_spec::Mixer;
        let hidden = spec.hidden as usize;
        let vocab = spec.vocab as usize;
        let head_dim = spec.head_dim as usize;
        let q_width = spec.q_width as usize;
        let kv_width = spec.kv_heads as usize * head_dim;
        let fp8 = spec.linear_format.fp8_layout();
        let dense_fp8 = fp8.filter(|layout| layout.block_scale().is_some());
        let nvfp4 = spec.linear_format == crate::model_spec::LinearFormat::NvFp4;
        let global_prefix = match spec.architecture {
            Architecture::Flint
            | Architecture::Clover
            | Architecture::Qwen3
            | Architecture::Llama => "model",
            Architecture::Qwen35Text | Architecture::MiniCpmV47 | Architecture::Qwen3Vl => {
                "model.language_model"
            }
            Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::DeepseekV41
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2
            | Architecture::DiffusionGemma => {
                unreachable!("architecture has a dedicated manifest path")
            }
        };
        put(
            expected,
            format!("{global_prefix}.embed_tokens.weight"),
            Dtype::BF16,
            vec![vocab, hidden],
            TensorScope::Text,
        );
        if nvfp4 {
            nvfp4_projection(expected, "lm_head".to_owned(), vocab, hidden)?;
        } else {
            put(
                expected,
                "lm_head.weight".to_owned(),
                Dtype::BF16,
                vec![vocab, hidden],
                TensorScope::Text,
            );
        }
        put(
            expected,
            format!("{global_prefix}.norm.weight"),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Text,
        );
        for (i, layer) in spec.layers.iter().enumerate() {
            let p = format!("{global_prefix}.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                put(
                    expected,
                    format!("{p}.{name}.weight"),
                    Dtype::BF16,
                    vec![hidden],
                    TensorScope::Text,
                );
            }
            match layer.ffn {
                crate::model_spec::FfnKind::Moe => {
                    let moe = spec
                        .moe
                        .as_ref()
                        .context("MoE layer has no global MoE geometry")?;
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
                    } else if spec.expert_format
                        == Some(crate::model_spec::ExpertFormat::Bf16Separate)
                    {
                        for expert in 0..experts {
                            for (role, n, k) in [
                                ("gate_proj", expert_i, hidden),
                                ("up_proj", expert_i, hidden),
                                ("down_proj", hidden, expert_i),
                            ] {
                                put(
                                    expected,
                                    format!("{p}.mlp.experts.{expert}.{role}.weight"),
                                    Dtype::BF16,
                                    vec![n, k],
                                    TensorScope::Text,
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
                    if moe.shared_gate {
                        put(
                            expected,
                            format!("{p}.mlp.shared_expert_gate.weight"),
                            Dtype::BF16,
                            vec![1, hidden],
                            TensorScope::Text,
                        );
                    }
                }
                crate::model_spec::FfnKind::Dense => {
                    let intermediate = spec.intermediate as usize;
                    for (role, n, k) in [
                        ("gate_proj", intermediate, hidden),
                        ("up_proj", intermediate, hidden),
                        ("down_proj", hidden, intermediate),
                    ] {
                        let base = format!("{p}.mlp.{role}");
                        if nvfp4 {
                            nvfp4_projection(expected, base, n, k)?;
                        } else {
                            projection(expected, base, n, k, dense_fp8);
                        }
                    }
                }
            }
            match layer.mixer {
                Mixer::FullAttention => {
                    let q_out = match spec.architecture {
                        Architecture::Flint
                        | Architecture::Qwen3
                        | Architecture::Qwen3Vl
                        | Architecture::Llama => q_width,
                        Architecture::Qwen35Text
                        | Architecture::Clover
                        | Architecture::MiniCpmV47 => 2 * q_width,
                        Architecture::Glm5Next
                        | Architecture::LoopVl
                        | Architecture::Gemma4
                        | Architecture::EmbeddingGemma2
                        | Architecture::DiffusionGemma => {
                            unreachable!("architecture has a dedicated manifest path")
                        }
                        Architecture::DeepseekV41 => {
                            unreachable!("DeepSeek-V4.1 has a dedicated manifest path")
                        }
                    };
                    for (role, n, k) in [
                        ("q_proj", q_out, hidden),
                        ("k_proj", kv_width, hidden),
                        ("v_proj", kv_width, hidden),
                        ("o_proj", hidden, q_width),
                    ] {
                        let base = format!("{p}.self_attn.{role}");
                        if nvfp4 {
                            nvfp4_projection(expected, base, n, k)?;
                        } else {
                            projection(expected, base, n, k, fp8);
                        }
                    }
                    if spec.architecture != Architecture::Llama {
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
                    if spec.architecture == Architecture::Flint {
                        put(
                            expected,
                            format!("{p}.self_attn.softmax_offset"),
                            Dtype::BF16,
                            vec![spec.q_heads as usize],
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
                    for (role, n, k) in [
                        ("in_proj_qkv", conv_dim, hidden),
                        ("in_proj_z", value_dim, hidden),
                        ("out_proj", hidden, value_dim),
                    ] {
                        let base = format!("{p}.linear_attn.{role}");
                        if nvfp4 {
                            nvfp4_projection(expected, base, n, k)?;
                        } else {
                            projection(expected, base, n, k, dense_fp8);
                        }
                    }
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
                    if spec.architecture == Architecture::Clover {
                        put(
                            expected,
                            format!("{p}.linear_attn.conv_right_weight"),
                            Dtype::BF16,
                            vec![conv_dim, 1, g.conv_kernel as usize - 1],
                            TensorScope::Text,
                        );
                        for name in [
                            "bd_softmax_logit_scale",
                            "bd_softmax_history_bias",
                            "bd_softmax_history_c",
                        ] {
                            put(
                                expected,
                                format!("{p}.linear_attn.{name}"),
                                Dtype::BF16,
                                vec![value_heads],
                                TensorScope::Text,
                            );
                        }
                    }
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

    /// Closed-world expectation of the DeepSeek-V4.1 checkpoint.
    pub fn deepseek_v41_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let ds = spec
            .deepseek_v41
            .as_ref()
            .context("DeepSeek-V4.1 geometry missing")?;
        let moe = spec
            .moe
            .as_ref()
            .context("DeepSeek-V4.1 MoE geometry missing")?;
        let layout = spec
            .linear_format
            .fp8_layout()
            .context("DeepSeek-V4.1 linear format has no scale layout")?;
        let expert_layout = match spec.expert_format {
            Some(crate::model_spec::ExpertFormat::Fp4E2m1Block { block, scale_dtype }) => {
                crate::fp8::Fp8BlockScale::new(block, scale_dtype)?
            }
            other => bail!("unsupported DeepSeek-V4.1 expert format {other:?}"),
        };
        let hidden = spec.hidden as usize;
        let vocab = spec.vocab as usize;
        let head_dim = spec.head_dim as usize;
        let q_width = spec.q_width as usize;
        let expert_i = moe.expert_intermediate as usize;
        let shared_i = moe.shared_intermediate as usize;
        let q_lora = ds.q_lora_rank as usize;
        let hc_width = (ds.hc_mult as usize) * hidden;
        let o_width = (ds.o_lora_rank as usize) * (ds.o_groups as usize);
        let o_input = (ds.o_groups as usize) * head_dim;

        for name in ["embed.weight", "head.weight"] {
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
            "norm.weight".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Text,
        );
        for name in ["image_start", "image_end", "image_newline"] {
            put(
                expected,
                name.into(),
                Dtype::BF16,
                vec![hidden],
                TensorScope::Text,
            );
        }

        let mut layers =
            Vec::with_capacity(ds.num_hidden_layers as usize + ds.nextn_layers as usize);
        for index in 0..ds.num_hidden_layers {
            layers.push((
                format!("layers.{index}"),
                TensorScope::Text,
                moe.experts as usize,
            ));
        }
        for index in 0..ds.nextn_layers {
            layers.push((
                format!("mtp.{index}"),
                TensorScope::Mtp,
                ds.dspark_experts as usize,
            ));
        }
        for (prefix, scope, expert_count) in layers {
            put(
                expected,
                format!("{prefix}.attn.attn_sink"),
                Dtype::F32,
                vec![spec.q_heads as usize],
                scope,
            );
            put(
                expected,
                format!("{prefix}.attn.kv_norm.weight"),
                Dtype::BF16,
                vec![head_dim],
                scope,
            );
            put(
                expected,
                format!("{prefix}.attn.q_norm.weight"),
                Dtype::BF16,
                vec![q_lora],
                scope,
            );
            for name in ["attn_norm", "ffn_norm"] {
                put(
                    expected,
                    format!("{prefix}.{name}.weight"),
                    Dtype::BF16,
                    vec![hidden],
                    scope,
                );
            }
            for name in ["hc_attn_base", "hc_ffn_base"] {
                put(
                    expected,
                    format!("{prefix}.{name}"),
                    Dtype::F32,
                    vec![HC_BASIS],
                    scope,
                );
            }
            for name in ["hc_attn_scale", "hc_ffn_scale"] {
                put(
                    expected,
                    format!("{prefix}.{name}"),
                    Dtype::F32,
                    vec![HC_SCALE],
                    scope,
                );
            }
            for name in ["hc_attn_fn", "hc_ffn_fn"] {
                put(
                    expected,
                    format!("{prefix}.{name}"),
                    Dtype::F32,
                    vec![HC_BASIS, hc_width],
                    scope,
                );
            }
            for (name, n, k) in [
                ("wkv", head_dim, hidden),
                ("wq_a", q_lora, hidden),
                ("wq_b", q_width, q_lora),
                ("wo_a", o_width, o_input),
                ("wo_b", hidden, o_width),
            ] {
                deepseek_projection(
                    expected,
                    &format!("{prefix}.attn.{name}"),
                    n,
                    k,
                    layout,
                    scope,
                );
            }
            put(
                expected,
                format!("{prefix}.ffn.gate.weight"),
                Dtype::BF16,
                vec![expert_count, hidden],
                scope,
            );
            for name in ["ffn.gate.bias", "ffn.gate.bias_vl"] {
                put(
                    expected,
                    format!("{prefix}.{name}"),
                    Dtype::F32,
                    vec![expert_count],
                    scope,
                );
            }
            for (name, n, k) in [
                ("w1", shared_i, hidden),
                ("w2", hidden, shared_i),
                ("w3", shared_i, hidden),
            ] {
                deepseek_projection(
                    expected,
                    &format!("{prefix}.ffn.shared_experts.{name}"),
                    n,
                    k,
                    layout,
                    scope,
                );
            }
            for expert in 0..expert_count {
                for (name, n, k) in [
                    ("w1", expert_i, hidden),
                    ("w2", hidden, expert_i),
                    ("w3", expert_i, hidden),
                ] {
                    deepseek_fp4_expert(
                        expected,
                        &format!("{prefix}.ffn.experts.{expert}.{name}"),
                        n,
                        k,
                        expert_layout,
                        scope,
                    );
                }
            }
        }

        for layer in 0..ds.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            if ds.kv_source_layer_ids.contains(&layer) {
                put(
                    expected,
                    format!("{prefix}.attn.compressor.norm.weight"),
                    Dtype::BF16,
                    vec![head_dim],
                    TensorScope::Text,
                );
                put(
                    expected,
                    format!("{prefix}.attn.compressor.wkv.weight"),
                    Dtype::BF16,
                    vec![head_dim, hidden],
                    TensorScope::Text,
                );
                if layer != ds.candidate_source_layer_id {
                    put(
                        expected,
                        format!("{prefix}.attn.compressor.wgate.weight"),
                        Dtype::BF16,
                        vec![head_dim, hidden],
                        TensorScope::Text,
                    );
                }
                put(
                    expected,
                    format!("{prefix}.attn.indexer.k_norm.weight"),
                    Dtype::BF16,
                    vec![ds.index_head_dim as usize],
                    TensorScope::Text,
                );
                put(
                    expected,
                    format!("{prefix}.attn.indexer.wk.weight"),
                    Dtype::BF16,
                    vec![ds.index_head_dim as usize, head_dim],
                    TensorScope::Text,
                );
            }
            if ds.index_source_layer_ids.contains(&layer) {
                put(
                    expected,
                    format!("{prefix}.attn.indexer.weights_proj.weight"),
                    Dtype::BF16,
                    vec![ds.index_heads as usize, hidden],
                    TensorScope::Text,
                );
                deepseek_projection(
                    expected,
                    &format!("{prefix}.attn.indexer.wq_b"),
                    (ds.index_heads as usize) * (ds.index_head_dim as usize),
                    q_lora,
                    layout,
                    TensorScope::Text,
                );
            }
            if let Some(position) = ds.engram_layer_ids.iter().position(|id| *id == layer) {
                let rows = ds.engram_num_embeddings[position] as usize;
                put(
                    expected,
                    format!("{prefix}.engram.embed.weight"),
                    Dtype::F8_E4M3,
                    vec![rows, ds.engram_head_dim as usize],
                    TensorScope::Text,
                );
                put(
                    expected,
                    format!("{prefix}.engram.embed.scale"),
                    Dtype::F8_E8M0,
                    vec![rows, ds.engram_n_heads as usize],
                    TensorScope::Text,
                );
                for name in ["engram.q_weight", "engram.k_weight"] {
                    put(
                        expected,
                        format!("{prefix}.{name}"),
                        Dtype::BF16,
                        vec![ds.engram_max_ngram_size as usize, hidden],
                        TensorScope::Text,
                    );
                }
                deepseek_projection(
                    expected,
                    &format!("{prefix}.engram.wkv"),
                    ENGRAM_WKV_ROWS,
                    hidden + (ds.engram_max_ngram_size as usize) * (ds.engram_head_dim as usize),
                    layout,
                    TensorScope::Text,
                );
            }
        }

        deepseek_projection(
            expected,
            "mtp.0.main_proj",
            hidden,
            MAIN_PROJECTION_WIDTH,
            layout,
            TensorScope::Mtp,
        );
        put(
            expected,
            "mtp.0.main_norm.weight".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Mtp,
        );
        put(
            expected,
            "mtp.2.norm.weight".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Mtp,
        );
        put(
            expected,
            "mtp.2.confidence_head.proj.weight".into(),
            Dtype::BF16,
            vec![1, hidden + ds.dspark_markov_rank as usize],
            TensorScope::Mtp,
        );
        for name in [
            "mtp.2.markov_head.embed.weight",
            "mtp.2.markov_head.head.weight",
        ] {
            put(
                expected,
                name.into(),
                Dtype::BF16,
                vec![vocab, ds.dspark_markov_rank as usize],
                TensorScope::Mtp,
            );
        }

        let vision_hidden = ds.vision_hidden as usize;
        let vision_i = ds.vision_intermediate as usize;
        let patch = ds.vision_patch as usize;
        let merged = (ds.vision_downsample_ratio as usize) * (ds.vision_downsample_ratio as usize);
        put(
            expected,
            "vision.patch_embed.proj.weight".into(),
            Dtype::BF16,
            vec![vision_hidden, VISION_CHANNELS * patch * patch],
            TensorScope::Vision,
        );
        put(
            expected,
            "vision.patch_embed.proj.bias".into(),
            Dtype::BF16,
            vec![vision_hidden],
            TensorScope::Vision,
        );
        put(
            expected,
            "vision.norm.weight".into(),
            Dtype::BF16,
            vec![vision_hidden],
            TensorScope::Vision,
        );
        for block in 0..ds.vision_layers {
            let prefix = format!("vision.blocks.{block}");
            for name in ["norm1", "norm2"] {
                put(
                    expected,
                    format!("{prefix}.{name}.weight"),
                    Dtype::BF16,
                    vec![vision_hidden],
                    TensorScope::Vision,
                );
            }
            put(
                expected,
                format!("{prefix}.attn.wqkv.weight"),
                Dtype::BF16,
                vec![3 * vision_hidden, vision_hidden],
                TensorScope::Vision,
            );
            put(
                expected,
                format!("{prefix}.attn.wqkv.bias"),
                Dtype::BF16,
                vec![3 * vision_hidden],
                TensorScope::Vision,
            );
            put(
                expected,
                format!("{prefix}.attn.wo.weight"),
                Dtype::BF16,
                vec![vision_hidden, vision_hidden],
                TensorScope::Vision,
            );
            put(
                expected,
                format!("{prefix}.attn.wo.bias"),
                Dtype::BF16,
                vec![vision_hidden],
                TensorScope::Vision,
            );
            put(
                expected,
                format!("{prefix}.mlp.w1.weight"),
                Dtype::BF16,
                vec![2 * vision_i, vision_hidden],
                TensorScope::Vision,
            );
            put(
                expected,
                format!("{prefix}.mlp.w2.weight"),
                Dtype::BF16,
                vec![vision_hidden, vision_i],
                TensorScope::Vision,
            );
        }
        put(
            expected,
            "aligner.w1.weight".into(),
            Dtype::BF16,
            vec![hidden, vision_hidden * merged],
            TensorScope::Vision,
        );
        put(
            expected,
            "aligner.w1.bias".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Vision,
        );
        put(
            expected,
            "aligner.w2.weight".into(),
            Dtype::BF16,
            vec![hidden, hidden],
            TensorScope::Vision,
        );
        put(
            expected,
            "aligner.w2.bias".into(),
            Dtype::BF16,
            vec![hidden],
            TensorScope::Vision,
        );
        Ok(())
    }

    pub(crate) fn glm5_next_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        use crate::model_spec::Mixer;
        let glm = spec
            .glm5_next
            .as_ref()
            .context("GLM-5.3 geometry missing")?;
        let hidden = spec.hidden as usize;
        let depth = spec.layers.len();
        let hc_width = (glm.hc_mult as usize)
            .checked_mul(hidden)
            .context("GLM HC input width overflow")?;
        let hc_coefficients = glm
            .hc_mult
            .checked_add(2)
            .and_then(|n| n.checked_mul(glm.hc_mult))
            .context("GLM HC coefficient width overflow")? as usize;
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            put(
                expected,
                name.into(),
                Dtype::BF16,
                vec![spec.vocab as usize, hidden],
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
        for i in 0..depth + glm.nextn_layers as usize {
            let scope = if i < depth {
                TensorScope::Text
            } else {
                TensorScope::Mtp
            };
            let p = format!("model.language_model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                put(
                    expected,
                    format!("{p}.{name}.weight"),
                    Dtype::BF16,
                    vec![hidden],
                    scope,
                );
            }
            if i < depth {
                for name in ["hc_attn_base", "hc_ffn_base"] {
                    put(
                        expected,
                        format!("{p}.{name}"),
                        Dtype::F32,
                        vec![hc_coefficients],
                        scope,
                    );
                }
                for name in ["hc_attn_scale", "hc_ffn_scale"] {
                    put(expected, format!("{p}.{name}"), Dtype::F32, vec![3], scope);
                }
                for name in ["hc_attn_fn", "hc_ffn_fn"] {
                    put(
                        expected,
                        format!("{p}.{name}"),
                        Dtype::BF16,
                        vec![hc_coefficients, hc_width],
                        scope,
                    );
                }
            }
            if i < glm.dense_layers as usize {
                for (role, n, k) in [
                    ("gate_proj", spec.intermediate as usize, hidden),
                    ("up_proj", spec.intermediate as usize, hidden),
                    ("down_proj", hidden, spec.intermediate as usize),
                ] {
                    projection_scoped(
                        expected,
                        format!("{p}.mlp.{role}"),
                        n,
                        k,
                        spec.linear_format.fp8_layout(),
                        scope,
                    );
                }
            } else {
                Self::glm5_next_moe(spec, &p, expected, scope)?;
            }
            let sparse = i >= depth || spec.layers[i].mixer == Mixer::FullAttention;
            if sparse {
                Self::glm5_next_dsa(spec, &p, expected, scope)?;
            } else {
                Self::glm5_next_kda(spec, &p, expected, scope)?;
            }
            if i >= depth {
                for (name, shape) in [
                    ("eh_proj.weight", vec![hidden, 2 * hidden]),
                    ("enorm.weight", vec![hidden]),
                    ("hnorm.weight", vec![hidden]),
                    ("shared_head.norm.weight", vec![hidden]),
                ] {
                    put(expected, format!("{p}.{name}"), Dtype::BF16, shape, scope);
                }
            }
        }
        Ok(())
    }

    fn glm5_next_kda(
        spec: &crate::model_spec::ModelSpec,
        prefix: &str,
        expected: &mut BTreeMap<String, ExpectedTensor>,
        scope: TensorScope,
    ) -> Result<()> {
        let g = spec.gdn.as_ref().context("GLM-5.3 KDA geometry missing")?;
        let hidden = spec.hidden as usize;
        let width = g
            .value_heads
            .checked_mul(g.value_head_dim)
            .context("GLM KDA projection width overflow")? as usize;
        let heads = g.value_heads as usize;
        let low_rank = g.key_head_dim as usize;
        for (role, n, k) in [
            ("b_proj", heads, hidden),
            ("f_a_proj", low_rank, hidden),
            ("f_b_proj", width, low_rank),
            ("g_a_proj", low_rank, hidden),
            ("g_b_proj", width, low_rank),
            ("k_proj", width, hidden),
            ("q_proj", width, hidden),
            ("v_proj", width, hidden),
            ("o_proj", hidden, width),
        ] {
            put(
                expected,
                format!("{prefix}.self_attn.{role}.weight"),
                Dtype::BF16,
                vec![n, k],
                scope,
            );
        }
        for role in ["k_conv1d", "q_conv1d", "v_conv1d"] {
            put(
                expected,
                format!("{prefix}.self_attn.{role}.weight"),
                Dtype::BF16,
                vec![width, 1, g.conv_kernel as usize],
                scope,
            );
        }
        put(
            expected,
            format!("{prefix}.self_attn.o_norm.weight"),
            Dtype::BF16,
            vec![g.value_head_dim as usize],
            scope,
        );
        put(
            expected,
            format!("{prefix}.self_attn.A_log"),
            Dtype::F32,
            vec![heads],
            scope,
        );
        put(
            expected,
            format!("{prefix}.self_attn.dt_bias"),
            Dtype::F32,
            vec![width],
            scope,
        );
        Ok(())
    }

    fn glm5_next_dsa(
        spec: &crate::model_spec::ModelSpec,
        prefix: &str,
        expected: &mut BTreeMap<String, ExpectedTensor>,
        scope: TensorScope,
    ) -> Result<()> {
        let glm = spec
            .glm5_next
            .as_ref()
            .context("GLM-5.3 DSA geometry missing")?;
        let hidden = spec.hidden as usize;
        let q_lora = glm.q_lora_rank as usize;
        let kv_lora = glm.kv_lora_rank as usize;
        let index_dim = glm.index_head_dim as usize;
        let kv_width = glm
            .qk_nope_head_dim
            .checked_add(glm.v_head_dim)
            .and_then(|n| n.checked_mul(spec.q_heads))
            .context("GLM DSA expanded KV width overflow")? as usize;
        let output_width = spec
            .q_heads
            .checked_mul(glm.v_head_dim)
            .context("GLM DSA output width overflow")? as usize;
        for (suffix, shape) in [
            (
                "index_kpool_compress_ape",
                vec![glm.index_kpool as usize, index_dim],
            ),
            ("index_kpool_compress_gate", vec![index_dim, hidden]),
            ("k_norm.bias", vec![index_dim]),
            ("k_norm.weight", vec![index_dim]),
            (
                "weights_proj.weight",
                vec![glm.index_heads as usize, hidden],
            ),
            ("wk.weight", vec![index_dim, hidden]),
            (
                "wq_b.weight",
                vec![glm.index_heads as usize * index_dim, q_lora],
            ),
        ] {
            put(
                expected,
                format!("{prefix}.self_attn.indexer.{suffix}"),
                Dtype::BF16,
                shape,
                scope,
            );
        }
        for (name, width) in [("kv_a_layernorm", kv_lora), ("q_a_layernorm", q_lora)] {
            put(
                expected,
                format!("{prefix}.self_attn.{name}.weight"),
                Dtype::BF16,
                vec![width],
                scope,
            );
        }
        put(
            expected,
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            Dtype::BF16,
            vec![kv_width, kv_lora],
            scope,
        );
        for (role, n, k) in [
            ("q_a_proj", q_lora, hidden),
            ("q_b_proj", spec.q_width as usize, q_lora),
            ("kv_a_proj_with_mqa", kv_lora, hidden),
            ("o_proj", hidden, output_width),
        ] {
            projection_scoped(
                expected,
                format!("{prefix}.self_attn.{role}"),
                n,
                k,
                spec.linear_format.fp8_layout(),
                scope,
            );
        }
        Ok(())
    }

    fn glm5_next_moe(
        spec: &crate::model_spec::ModelSpec,
        prefix: &str,
        expected: &mut BTreeMap<String, ExpectedTensor>,
        scope: TensorScope,
    ) -> Result<()> {
        let moe = spec.moe.context("GLM-5.3 MoE geometry missing")?;
        let hidden = spec.hidden as usize;
        put(
            expected,
            format!("{prefix}.mlp.gate.weight"),
            Dtype::BF16,
            vec![moe.experts as usize, hidden],
            scope,
        );
        put(
            expected,
            format!("{prefix}.mlp.gate.e_score_correction_bias"),
            Dtype::F32,
            vec![moe.experts as usize],
            scope,
        );
        for (role, n, k) in [
            ("gate_proj", moe.expert_intermediate as usize, hidden),
            ("up_proj", moe.expert_intermediate as usize, hidden),
            ("down_proj", hidden, moe.expert_intermediate as usize),
        ] {
            projection_scoped(
                expected,
                format!("{prefix}.mlp.shared_experts.{role}"),
                n,
                k,
                spec.linear_format.fp8_layout(),
                scope,
            );
            for expert in 0..moe.experts {
                projection_scoped(
                    expected,
                    format!("{prefix}.mlp.experts.{expert}.{role}"),
                    n,
                    k,
                    spec.linear_format.fp8_layout(),
                    scope,
                );
            }
        }
        Ok(())
    }

    /// Qwen3-VL vision tower: patch embedding, learned positions, `depth` LayerNorm blocks, the final merger (LayerNorm
    /// per patch) and one deepstack merger per tap (LayerNorm over the merged row), all BF16.
    fn qwen3_vl_vision_entries(
        spec: &crate::model_spec::ModelSpec,
        expected: &mut BTreeMap<String, ExpectedTensor>,
    ) -> Result<()> {
        let v = spec
            .vision
            .as_ref()
            .context("Qwen3-VL vision configuration missing")?;
        v.validate(crate::model_spec::Architecture::Qwen3Vl)?;
        let (hidden, mlp, output) = (
            v.hidden as usize,
            v.intermediate as usize,
            v.output as usize,
        );
        let merged = hidden * (v.merge * v.merge) as usize;
        let mut add = |name: String, shape: Vec<usize>| {
            put(expected, name, Dtype::BF16, shape, TensorScope::Vision)
        };
        add(
            "model.visual.patch_embed.proj.weight".into(),
            vec![
                hidden,
                v.channels as usize,
                v.temporal_patch as usize,
                v.patch as usize,
                v.patch as usize,
            ],
        );
        add("model.visual.patch_embed.proj.bias".into(), vec![hidden]);
        add(
            "model.visual.pos_embed.weight".into(),
            vec![(v.position_side * v.position_side) as usize, hidden],
        );
        for i in 0..v.depth {
            let p = format!("model.visual.blocks.{i}");
            for (name, shape) in [
                ("attn.qkv.weight", vec![3 * hidden, hidden]),
                ("attn.qkv.bias", vec![3 * hidden]),
                ("attn.proj.weight", vec![hidden, hidden]),
                ("attn.proj.bias", vec![hidden]),
                ("mlp.linear_fc1.weight", vec![mlp, hidden]),
                ("mlp.linear_fc1.bias", vec![mlp]),
                ("mlp.linear_fc2.weight", vec![hidden, mlp]),
                ("mlp.linear_fc2.bias", vec![hidden]),
                ("norm1.weight", vec![hidden]),
                ("norm1.bias", vec![hidden]),
                ("norm2.weight", vec![hidden]),
                ("norm2.bias", vec![hidden]),
            ] {
                add(format!("{p}.{name}"), shape);
            }
        }
        let mergers = std::iter::once(("model.visual.merger".to_owned(), hidden)).chain(
            (0..v.deepstack.len())
                .map(|k| (format!("model.visual.deepstack_merger_list.{k}"), merged)),
        );
        for (p, norm) in mergers {
            add(format!("{p}.norm.weight"), vec![norm]);
            add(format!("{p}.norm.bias"), vec![norm]);
            add(format!("{p}.linear_fc1.weight"), vec![merged, merged]);
            add(format!("{p}.linear_fc1.bias"), vec![merged]);
            add(format!("{p}.linear_fc2.weight"), vec![output, merged]);
            add(format!("{p}.linear_fc2.bias"), vec![output]);
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

    pub(crate) fn assemble_manifest(
        &self,
        label: &str,
        mut expected: BTreeMap<String, ExpectedTensor>,
        policy: LoadPolicy,
        expected_bytes: usize,
        scale_suffix: Option<&str>,
    ) -> Result<TensorManifest> {
        if expected.len() != self.index.weight_map.len() {
            let missing: Vec<_> = expected
                .keys()
                .filter(|name| !self.index.weight_map.contains_key(*name))
                .take(3)
                .collect();
            let unexpected: Vec<_> = self
                .index
                .weight_map
                .keys()
                .filter(|name| !expected.contains_key(*name))
                .take(3)
                .collect();
            bail!(
                "{label} tensor count mismatch: expected={} actual={} missing={missing:?} unexpected={unexpected:?}",
                expected.len(),
                self.index.weight_map.len()
            );
        }
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
            let (layer, role) = split_layer_role(name)?;
            let scale_name = if dtype == Dtype::F8_E4M3 {
                if scale_suffix.is_none() && name.ends_with(".weight_scale") {
                    None
                } else {
                    Some(format!(
                        "{}.{}",
                        name.strip_suffix(".weight")
                            .context("invalid FP8 tensor suffix")?,
                        scale_suffix.context("FP8 tensor without scale layout")?
                    ))
                }
            } else if dtype == Dtype::U8 {
                let stem = name
                    .strip_suffix(".weight")
                    .context("invalid NVFP4 tensor suffix")?;
                Some(format!("{stem}.weight_scale"))
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
                    quant: None,
                    swap_halves: false,
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

impl MappedCheckpoint {
    /// MiniMax-H3 transformer: F32 patch, time-embedding and output projections, BF16 elsewhere. The per-block AdaLN
    /// projections are Dormant: read once at setup into the modulation table, never part of the per-step layer set.
    pub fn minimax_h3_manifest(&self, spec: &crate::minimax_h3::H3Spec) -> Result<TensorManifest> {
        let expected = minimax_h3_expected(spec)?;
        let planned = planned_bytes(&expected, LoadPolicy::TextOnly)?;
        self.assemble_manifest("MiniMax-H3", expected, LoadPolicy::TextOnly, planned, None)
    }

    /// MiniMax-H3-Image transformer: F32 patch, time-embedding and per-step heads, BF16 elsewhere. Per-block AdaLN
    /// projections are Dormant.
    pub fn minimax_h3_image_manifest(
        &self,
        spec: &crate::minimax_h3::ImageSpec,
    ) -> Result<TensorManifest> {
        let expected = minimax_h3_image_expected(spec)?;
        let planned = planned_bytes(&expected, LoadPolicy::TextOnly)?;
        self.assemble_manifest(
            "MiniMax-H3-Image",
            expected,
            LoadPolicy::TextOnly,
            planned,
            None,
        )
    }

    /// MiniMax-H3 transformer weights from one community GGUF file. The expected set is `minimax_h3_expected`;
    /// every GGUF tensor is matched by name, checked against the official shape and stored type, and mapped to its
    /// official entry. `rope.inv_freq` has no official counterpart and is ignored; any other unknown tensor is an
    /// error, as is any expected tensor the GGUF does not produce.
    pub fn minimax_h3_gguf_manifest(
        spec: &crate::minimax_h3::H3Spec,
        gguf: &crate::gguf::GgufFile,
    ) -> Result<TensorManifest> {
        let expected = minimax_h3_expected(spec)?;
        let shard = gguf
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .context("GGUF path has no file name")?
            .to_owned();
        let mut entries = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for tensor in gguf.tensors() {
            let (block_len, block_bytes) = ggml_block(tensor.ggml_type)?;
            let dims = &tensor.dims;
            let count = dims.iter().product::<u64>();
            let official: Vec<u64> = match gguf.orig_shape(&tensor.name) {
                Some(shape) => shape.iter().map(|&v| u64::from(v)).collect(),
                None => dims.iter().rev().copied().collect(),
            };
            ensure!(
                official.iter().product::<u64>() == count,
                "GGUF tensor {} official shape {official:?} does not cover {count} elements",
                tensor.name
            );
            let input = *official.last().context("GGUF official shape is empty")?;
            ensure!(
                input % block_len == 0,
                "GGUF tensor {} has input {input}, not a multiple of the {}-element {:?} block",
                tensor.name,
                block_len,
                tensor.ggml_type
            );
            let full_bytes = count / block_len * block_bytes;
            ensure!(
                full_bytes == tensor.nbytes as u64,
                "GGUF tensor {} stores {} B, its {count} elements require {full_bytes} B",
                tensor.name,
                tensor.nbytes
            );
            let base = gguf
                .data_start()
                .checked_add(tensor.offset)
                .context("GGUF tensor offset overflow")?;
            for map in map_gguf_tensor(spec, &tensor.name)? {
                let (dtype, shape, scope) = expected.get(&map.official).with_context(|| {
                    format!(
                        "GGUF tensor {} maps to unknown official tensor {}",
                        tensor.name, map.official
                    )
                })?;
                let (data_start, nbytes) = match map.range {
                    Some((start, end)) => {
                        ensure!(
                            official.len() == 2
                                && official[1] == shape[1] as u64
                                && official[0] == 3 * shape[0] as u64
                                && end <= official[0]
                                && start < end
                                && end - start == shape[0] as u64,
                            "GGUF tensor {} fused range {start}..{end} does not match official {} shape {shape:?}",
                            tensor.name,
                            map.official
                        );
                        let row_blocks = input / block_len;
                        let offset = start
                            .checked_mul(row_blocks)
                            .and_then(|n| n.checked_mul(block_bytes))
                            .context("GGUF tensor offset overflow")?;
                        let nbytes = (end - start)
                            .checked_mul(row_blocks)
                            .and_then(|n| n.checked_mul(block_bytes))
                            .context("GGUF tensor size overflow")?;
                        (
                            base.checked_add(offset)
                                .context("GGUF tensor offset overflow")?,
                            nbytes,
                        )
                    }
                    None => {
                        let matches = official.len() == shape.len()
                            && official
                                .iter()
                                .zip(shape)
                                .all(|(got, want)| *got == *want as u64);
                        ensure!(
                            matches,
                            "GGUF tensor {} official shape {official:?} does not match {} shape {shape:?}",
                            tensor.name,
                            map.official
                        );
                        (base, tensor.nbytes as u64)
                    }
                };
                ensure!(
                    !entries.contains_key(&map.official),
                    "GGUF tensor {} maps to already-produced official tensor {}",
                    tensor.name,
                    map.official
                );
                ensure!(
                    !map.swap_halves || tensor.ggml_type == crate::gguf::GgmlType::Q4_0,
                    "GGUF tensor {} stores the fused fc1 as {:?}, which has no half-swap path",
                    tensor.name,
                    tensor.ggml_type
                );
                let pass_through = match tensor.ggml_type {
                    crate::gguf::GgmlType::F32 => Some(Dtype::F32),
                    crate::gguf::GgmlType::Bf16 => Some(Dtype::BF16),
                    _ => None,
                };
                if let Some(stored) = pass_through {
                    ensure!(
                        stored == *dtype,
                        "GGUF tensor {} stores {:?} bytes for the {:?} tensor {}",
                        tensor.name,
                        tensor.ggml_type,
                        dtype,
                        map.official
                    );
                }
                let (layer, role) = split_layer_role(&map.official)?;
                let load = LoadPolicy::TextOnly.selects_tensor(*scope, &map.official);
                let nbytes = usize::try_from(nbytes)?;
                if load {
                    resident_bytes = resident_bytes
                        .checked_add(nbytes)
                        .context("resident size overflow")?;
                }
                entries.insert(
                    map.official.clone(),
                    TensorEntry {
                        dtype: *dtype,
                        shape: shape.clone(),
                        shard: shard.clone(),
                        bytes: nbytes,
                        load,
                        data_start,
                        data_offsets: [0, nbytes as u64],
                        scope: *scope,
                        layer,
                        role,
                        scale_name: None,
                        quant: Some(tensor.ggml_type),
                        swap_halves: map.swap_halves,
                    },
                );
            }
        }
        ensure!(
            entries.len() == expected.len(),
            "MiniMax-H3 GGUF manifest incomplete: expected {} official tensors, produced {}",
            expected.len(),
            entries.len()
        );
        Ok(TensorManifest {
            entries,
            resident_bytes,
        })
    }

    /// MiniMax-H3 video VAE (published F32): every tensor's name and shape from `vae/config.json`. The decoder side
    /// (post_quant_conv and the ViT decoder) is Text scope, the keyframe encoder side (encoder and quant_conv) Vision
    /// scope; both load.
    pub fn minimax_h3_vae_manifest(
        &self,
        vae: &crate::minimax_h3::VaeSpec,
    ) -> Result<TensorManifest> {
        let mut expected = BTreeMap::new();
        let mut entry = |name: String, shape: Vec<usize>, scope: TensorScope| {
            put(&mut expected, name, Dtype::F32, shape, scope)
        };
        let (latent, dim) = (vae.latent_channels, vae.dim);
        let decoder = TensorScope::Text;
        for (name, shape) in [
            ("post_quant_conv.weight", vec![latent, latent, 1, 1, 1]),
            ("post_quant_conv.bias", vec![latent]),
            ("decoder.proj_in.weight", vec![dim, latent]),
            ("decoder.proj_in.bias", vec![dim]),
            ("decoder.register_tokens", vec![1, vae.registers, dim]),
            ("decoder.norm_out.weight", vec![dim]),
            ("decoder.norm_out.bias", vec![dim]),
            ("decoder.proj_out.weight", vec![vae.patch_width(), dim]),
            ("decoder.proj_out.bias", vec![vae.patch_width()]),
        ] {
            entry(name.to_owned(), shape, decoder);
        }
        for layer in 0..vae.layers {
            let block = format!("decoder.transformer_blocks.{layer}");
            for (name, shape) in [
                ("attn.to_q.weight", vec![dim, dim]),
                ("attn.to_k.weight", vec![dim, dim]),
                ("attn.to_v.weight", vec![dim, dim]),
                ("attn.to_out.0.weight", vec![dim, dim]),
                ("ff.net.0.proj.weight", vec![2 * vae.ffn, dim]),
                ("ff.net.2.weight", vec![dim, vae.ffn]),
                ("attn.to_q.bias", vec![dim]),
                ("attn.to_k.bias", vec![dim]),
                ("attn.to_v.bias", vec![dim]),
                ("attn.to_out.0.bias", vec![dim]),
                ("ff.net.0.proj.bias", vec![2 * vae.ffn]),
                ("ff.net.2.bias", vec![dim]),
                ("norm1.weight", vec![dim]),
                ("norm2.weight", vec![dim]),
                ("scale1", vec![dim]),
                ("scale2", vec![dim]),
            ] {
                entry(format!("{block}.{name}"), shape, decoder);
            }
        }
        let encoder = TensorScope::Vision;
        let conv = |out: usize, input: usize, k: usize| vec![out, input, k, k, k];
        let first = *vae
            .block_out_channels
            .first()
            .context("empty VAE encoder")?;
        let last = *vae.block_out_channels.last().context("empty VAE encoder")?;
        entry(
            "encoder.conv_in.weight".into(),
            conv(first, vae.in_channels, 3),
            encoder,
        );
        entry("encoder.conv_in.bias".into(), vec![first], encoder);
        let mut input = first;
        for (i, &out) in vae.block_out_channels.iter().enumerate() {
            for j in 0..vae.layers_per_block {
                let resnet = format!("encoder.down_blocks.{i}.resnets.{j}");
                let channels = if j == 0 { input } else { out };
                for (name, shape) in [
                    ("norm1.weight", vec![channels]),
                    ("norm1.bias", vec![channels]),
                    ("conv1.weight", conv(out, channels, 3)),
                    ("conv1.bias", vec![out]),
                    ("norm2.weight", vec![out]),
                    ("norm2.bias", vec![out]),
                    ("conv2.weight", conv(out, out, 3)),
                    ("conv2.bias", vec![out]),
                ] {
                    entry(format!("{resnet}.{name}"), shape, encoder);
                }
                if channels != out {
                    entry(
                        format!("{resnet}.conv_shortcut.weight"),
                        conv(out, channels, 1),
                        encoder,
                    );
                    entry(format!("{resnet}.conv_shortcut.bias"), vec![out], encoder);
                }
            }
            if vae.spatial_factors[i] * vae.temporal_factors[i] > 1 {
                let down = format!("encoder.down_blocks.{i}.downsamplers.0.conv");
                entry(format!("{down}.weight"), conv(out, out, 3), encoder);
                entry(format!("{down}.bias"), vec![out], encoder);
            }
            input = out;
        }
        entry("encoder.norm_out.weight".into(), vec![last], encoder);
        entry("encoder.norm_out.bias".into(), vec![last], encoder);
        entry(
            "encoder.conv_out.weight".into(),
            conv(2 * latent, last, 3),
            encoder,
        );
        entry("encoder.conv_out.bias".into(), vec![2 * latent], encoder);
        entry(
            "quant_conv.weight".into(),
            conv(2 * latent, 2 * latent, 1),
            encoder,
        );
        entry("quant_conv.bias".into(), vec![2 * latent], encoder);
        let planned = planned_bytes(&expected, LoadPolicy::TextWithVision)?;
        self.assemble_manifest(
            "MiniMax-H3 video VAE",
            expected,
            LoadPolicy::TextWithVision,
            planned,
            None,
        )
    }

    /// MiniMax-H3 audio VAE (published F32): every tensor's name and shape from `audio_vae/config.json`. The decoder
    /// side (dec_in_proj and the BigVGAN decoder) is Audio scope and loads; the DAC encoder, pre_block, mean_proj and
    /// logs_proj are Dormant.
    pub fn minimax_h3_audio_vae_manifest(
        &self,
        a: &crate::minimax_h3::AudioVaeSpec,
    ) -> Result<TensorManifest> {
        type Entry<'a> = dyn FnMut(String, Vec<usize>, TensorScope) + 'a;
        /// A weight-normed convolution: `weight_g [dim0, 1, 1]`, `weight_v [dim0, dim1, k]`, optional `bias [width]`.
        fn normed(
            e: &mut Entry<'_>,
            base: &str,
            v: [usize; 3],
            bias: Option<usize>,
            scope: TensorScope,
        ) {
            e(format!("{base}.weight_g"), vec![v[0], 1, 1], scope);
            e(format!("{base}.weight_v"), v.to_vec(), scope);
            if let Some(width) = bias {
                e(format!("{base}.bias"), vec![width], scope);
            }
        }
        let mut expected = BTreeMap::new();
        let e: &mut Entry<'_> = &mut |name: String, shape: Vec<usize>, scope: TensorScope| {
            put(&mut expected, name, Dtype::F32, shape, scope)
        };
        let (encoder, decoder) = (TensorScope::Dormant, TensorScope::Audio);
        let snake = |e: &mut Entry<'_>, base: String, dim: usize| {
            e(format!("{base}.alpha"), vec![1, dim, 1], encoder)
        };
        normed(
            e,
            "encoder.block.0",
            [a.encoder_dim, 1, 7],
            Some(a.encoder_dim),
            encoder,
        );
        let mut dim = a.encoder_dim;
        for (i, &stride) in a.encoder_rates.iter().enumerate() {
            let block = format!("encoder.block.{}", i + 1);
            for unit in 0..3 {
                let base = format!("{block}.block.{unit}.block");
                snake(e, format!("{base}.0"), dim);
                normed(e, &format!("{base}.1"), [dim, dim, 7], Some(dim), encoder);
                snake(e, format!("{base}.2"), dim);
                normed(e, &format!("{base}.3"), [dim, dim, 1], Some(dim), encoder);
            }
            snake(e, format!("{block}.block.3"), dim);
            normed(
                e,
                &format!("{block}.block.4"),
                [2 * dim, dim, 2 * stride],
                Some(2 * dim),
                encoder,
            );
            dim *= 2;
        }
        let tail = a.encoder_rates.len() + 1;
        snake(e, format!("encoder.block.{tail}"), dim);
        normed(
            e,
            &format!("encoder.block.{}", tail + 1),
            [a.latent_dim, dim, 3],
            Some(a.latent_dim),
            encoder,
        );
        let (latent, lc) = (a.latent_dim, a.latent_channels);
        for (name, shape) in [
            ("pre_block.norm1.weight", vec![latent]),
            ("pre_block.norm1.bias", vec![latent]),
            ("pre_block.norm3.weight", vec![latent]),
            ("pre_block.norm3.bias", vec![latent]),
            ("pre_block.norm2.weight", vec![lc]),
            ("pre_block.norm2.bias", vec![lc]),
            ("pre_block.attn.qkv.weight", vec![3 * latent, latent]),
            ("pre_block.attn.q_bias", vec![latent]),
            ("pre_block.attn.v_bias", vec![latent]),
            ("pre_block.attn.zero_k_bias", vec![latent]),
            ("pre_block.attn.proj.weight", vec![lc, lc]),
            ("pre_block.attn.proj.bias", vec![lc]),
            ("pre_block.proj.weight", vec![lc, latent]),
            ("pre_block.proj.bias", vec![lc]),
            ("pre_block.mlp.norm.weight", vec![lc]),
            ("pre_block.mlp.norm.bias", vec![lc]),
            ("pre_block.mlp.w0.weight", vec![2 * lc, lc]),
            ("pre_block.mlp.w0.bias", vec![2 * lc]),
            ("pre_block.mlp.w1.weight", vec![2 * lc, lc]),
            ("pre_block.mlp.w1.bias", vec![2 * lc]),
            ("pre_block.mlp.w2.weight", vec![lc, 2 * lc]),
            ("pre_block.mlp.w2.bias", vec![lc]),
            ("mean_proj.weight", vec![lc, lc, 1]),
            ("mean_proj.bias", vec![lc]),
            ("logs_proj.weight", vec![lc, lc, 1]),
            ("logs_proj.bias", vec![lc]),
        ] {
            e(name.to_owned(), shape, encoder);
        }
        e("dec_in_proj.weight".into(), vec![latent, lc, 1], decoder);
        e("dec_in_proj.bias".into(), vec![latent], decoder);
        normed(
            e,
            "decoder.conv_pre",
            [a.decoder_dim, latent, 7],
            Some(a.decoder_dim),
            decoder,
        );
        let activation = |e: &mut Entry<'_>, base: String, channels: usize| {
            let taps = vec![1, 1, crate::minimax_h3::AUDIO_RESAMPLE_TAPS];
            e(format!("{base}.act.alpha"), vec![channels], decoder);
            e(format!("{base}.act.beta"), vec![channels], decoder);
            e(format!("{base}.upsample.filter"), taps.clone(), decoder);
            e(format!("{base}.downsample.lowpass.filter"), taps, decoder);
        };
        let blocks = a.resblock_kernels.len();
        let mut channels = a.decoder_dim;
        for (i, &kernel) in a.decoder_kernels.iter().enumerate() {
            let half = channels / 2;
            normed(
                e,
                &format!("decoder.ups.{i}.0"),
                [channels, half, kernel],
                Some(half),
                decoder,
            );
            channels = half;
            for (j, (&k, dilations)) in a
                .resblock_kernels
                .iter()
                .zip(&a.resblock_dilations)
                .enumerate()
            {
                let block = format!("decoder.resblocks.{}", i * blocks + j);
                for m in 0..dilations.len() {
                    for convs in ["convs1", "convs2"] {
                        normed(
                            e,
                            &format!("{block}.{convs}.{m}"),
                            [channels, channels, k],
                            Some(channels),
                            decoder,
                        );
                    }
                }
                for m in 0..2 * dilations.len() {
                    activation(e, format!("{block}.activations.{m}"), channels);
                }
            }
        }
        activation(e, "decoder.activation_post".into(), channels);
        normed(e, "decoder.conv_post", [1, channels, 7], None, decoder);
        let planned = planned_bytes(&expected, LoadPolicy::AudioOnly)?;
        self.assemble_manifest(
            "MiniMax-H3 audio VAE",
            expected,
            LoadPolicy::AudioOnly,
            planned,
            None,
        )
    }

    /// MiniMax-H3 Qwen3-VL conditioner: decoder layers `0..layers` load; the embedding table (read by row), the later
    /// layers, the final norm and the LM head are Dormant; the vision tower is Vision scope.
    pub fn minimax_h3_conditioner_manifest(
        &self,
        spec: &crate::model_spec::ModelSpec,
        layers: usize,
    ) -> Result<TensorManifest> {
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Qwen3Vl
                && layers <= spec.layers.len(),
            "conditioner manifest needs Qwen3-VL with at least {layers} decoder layers"
        );
        let mut expected = BTreeMap::new();
        Self::text_entries(spec, &mut expected)?;
        Self::qwen3_vl_vision_entries(spec, &mut expected)?;
        for (name, (_, _, scope)) in expected.iter_mut() {
            let layer = name
                .strip_prefix("model.language_model.layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|index| index.parse::<usize>().ok());
            if *scope == TensorScope::Text && layer.is_none_or(|layer| layer >= layers) {
                *scope = TensorScope::Dormant;
            }
        }
        let planned = planned_bytes(&expected, LoadPolicy::TextOnly)?;
        self.assemble_manifest(
            "MiniMax-H3 conditioner",
            expected,
            LoadPolicy::TextOnly,
            planned,
            None,
        )
    }

    /// MiniMax-H3 LoRA file: `{linear}.lora_A.default.weight [rank, in]` and `{linear}.lora_B.default.weight
    /// [out, rank]` in BF16 for every block linear; scale is the metadata `alpha` over the rank.
    pub fn minimax_h3_lora_manifest(
        &self,
        spec: &crate::minimax_h3::H3Spec,
    ) -> Result<(TensorManifest, crate::minimax_h3::LoraAdapter)> {
        ensure!(
            self.maps.len() == 1,
            "MiniMax-H3 LoRA spans {} files",
            self.maps.len()
        );
        let (shard, map) = self.maps.iter().next().context("empty LoRA checkpoint")?;
        let (_, metadata) = SafeTensors::read_metadata(map)?;
        let metadata = metadata
            .metadata()
            .as_ref()
            .context("MiniMax-H3 LoRA metadata missing")?;
        let field = |key: &str| metadata.get(key).map(String::as_str);
        ensure!(
            field("key_format") == Some("minimax-h3-diffusers")
                && field("floating_dtype") == Some("bfloat16"),
            "unsupported MiniMax-H3 LoRA metadata {metadata:?}"
        );
        let alpha: f32 = field("alpha")
            .context("MiniMax-H3 LoRA alpha missing")?
            .parse()
            .context("MiniMax-H3 LoRA alpha")?;
        let linears = spec.block_linears();
        let probe = format!("{}.lora_A.default.weight", linears[0].0);
        let rank = *SafeTensors::deserialize(map)?
            .tensor(&probe)
            .with_context(|| format!("{shard} lacks {probe}"))?
            .shape()
            .first()
            .context("LoRA rank")?;
        ensure!(
            rank > 0 && alpha.is_finite() && alpha > 0.0,
            "invalid MiniMax-H3 LoRA rank {rank} alpha {alpha}"
        );
        let mut expected = BTreeMap::new();
        for (base, n, k) in linears {
            let a = format!("{base}.lora_A.default.weight");
            let b = format!("{base}.lora_B.default.weight");
            put(
                &mut expected,
                a,
                Dtype::BF16,
                vec![rank, k],
                TensorScope::Text,
            );
            put(
                &mut expected,
                b,
                Dtype::BF16,
                vec![n, rank],
                TensorScope::Text,
            );
        }
        let planned = planned_bytes(&expected, LoadPolicy::TextOnly)?;
        let manifest = self.assemble_manifest(
            "MiniMax-H3 LoRA",
            expected,
            LoadPolicy::TextOnly,
            planned,
            None,
        )?;
        Ok((
            manifest,
            crate::minimax_h3::LoraAdapter {
                rank,
                scale: alpha / rank as f32,
            },
        ))
    }
}

/// The official MiniMax-H3 tensor set: F32 patch, time-embedding and output projections, BF16 elsewhere. The
/// per-block AdaLN projections are Dormant: read once at setup into the modulation table, never part of the
/// per-step layer set.
pub(crate) fn minimax_h3_expected(
    spec: &crate::minimax_h3::H3Spec,
) -> Result<BTreeMap<String, ExpectedTensor>> {
    let scope = TensorScope::Text;
    let mut expected = BTreeMap::new();
    for (base, n, k) in [
        ("proj_in", spec.hidden, spec.video_row_width()),
        ("audio_proj_in", spec.hidden, spec.audio_in_channels),
        ("time_embedder.linear_1", spec.time_hidden, spec.freq_dim),
        ("time_embedder.linear_2", spec.time_dim, spec.time_hidden),
        ("proj_out", spec.video_row_width(), spec.hidden),
        ("audio_proj_out", spec.audio_in_channels, spec.hidden),
    ] {
        put(
            &mut expected,
            format!("{base}.weight"),
            Dtype::F32,
            vec![n, k],
            scope,
        );
        put(
            &mut expected,
            format!("{base}.bias"),
            Dtype::F32,
            vec![n],
            scope,
        );
    }
    let modulation = 6 * spec.hidden * crate::minimax_h3::MODALITIES as usize;
    for (name, shape) in [
        ("context_embedder.weight", vec![spec.hidden, spec.text_dim]),
        ("context_embedder.bias", vec![spec.hidden]),
        ("token_refiner.final_norm.weight", vec![spec.hidden]),
        ("norm_out.norm.weight", vec![spec.hidden]),
        (
            "norm_out.linear.weight",
            vec![2 * spec.hidden, spec.time_dim],
        ),
        ("norm_out.linear.bias", vec![2 * spec.hidden]),
    ] {
        put(&mut expected, name.to_owned(), Dtype::BF16, shape, scope);
    }
    let blocks = (0..spec.refiner_layers)
        .map(|i| (format!("token_refiner.refiner_blocks.{i}"), false))
        .chain((0..spec.layers).map(|i| (format!("transformer_blocks.{i}"), true)));
    for (block, adaln) in blocks {
        for (name, shape) in [
            ("norm1.weight", vec![spec.hidden]),
            ("norm2.weight", vec![spec.hidden]),
            ("attn.norm_q.weight", vec![spec.head_dim]),
            ("attn.norm_k.weight", vec![spec.head_dim]),
        ] {
            put(
                &mut expected,
                format!("{block}.{name}"),
                Dtype::BF16,
                shape,
                scope,
            );
        }
        if adaln {
            let base = format!("{block}.adaln_proj.linear");
            put(
                &mut expected,
                format!("{base}.weight"),
                Dtype::BF16,
                vec![modulation, spec.time_dim],
                TensorScope::Dormant,
            );
            put(
                &mut expected,
                format!("{base}.bias"),
                Dtype::BF16,
                vec![modulation],
                TensorScope::Dormant,
            );
        }
    }
    for (base, n, k) in spec.block_linears() {
        put(
            &mut expected,
            format!("{base}.weight"),
            Dtype::BF16,
            vec![n, k],
            scope,
        );
    }
    Ok(expected)
}

/// Official MiniMax-H3-Image tensor set. F32 `proj_in`, the time MLP and `pdd_proj_out.{i}`; no audio projection and
/// no single `proj_out`. Per-block AdaLN projections are Dormant.
pub(crate) fn minimax_h3_image_expected(
    spec: &crate::minimax_h3::ImageSpec,
) -> Result<BTreeMap<String, ExpectedTensor>> {
    let scope = TensorScope::Text;
    let mut expected = BTreeMap::new();
    let row = spec.row_width();
    for (base, n, k) in [
        ("proj_in", spec.hidden, row),
        ("time_embedder.linear_1", spec.time_hidden, spec.freq_dim),
        ("time_embedder.linear_2", spec.time_dim, spec.time_hidden),
    ] {
        put(
            &mut expected,
            format!("{base}.weight"),
            Dtype::F32,
            vec![n, k],
            scope,
        );
        put(
            &mut expected,
            format!("{base}.bias"),
            Dtype::F32,
            vec![n],
            scope,
        );
    }
    for head in 0..spec.pdd_heads {
        let base = format!("pdd_proj_out.{head}");
        put(
            &mut expected,
            format!("{base}.weight"),
            Dtype::F32,
            vec![row, spec.hidden],
            scope,
        );
        put(
            &mut expected,
            format!("{base}.bias"),
            Dtype::F32,
            vec![row],
            scope,
        );
    }
    let modulation = 6 * spec.hidden * crate::minimax_h3::MODALITIES as usize;
    for (name, shape) in [
        ("context_embedder.weight", vec![spec.hidden, spec.text_dim]),
        ("context_embedder.bias", vec![spec.hidden]),
        ("token_refiner.final_norm.weight", vec![spec.hidden]),
        ("norm_out.norm.weight", vec![spec.hidden]),
        (
            "norm_out.linear.weight",
            vec![2 * spec.hidden, spec.time_dim],
        ),
        ("norm_out.linear.bias", vec![2 * spec.hidden]),
    ] {
        put(&mut expected, name.to_owned(), Dtype::BF16, shape, scope);
    }
    let blocks = (0..spec.refiner_layers)
        .map(|i| (format!("token_refiner.refiner_blocks.{i}"), false))
        .chain((0..spec.layers).map(|i| (format!("transformer_blocks.{i}"), true)));
    for (block, adaln) in blocks {
        for (name, shape) in [
            ("norm1.weight", vec![spec.hidden]),
            ("norm2.weight", vec![spec.hidden]),
            ("attn.norm_q.weight", vec![spec.head_dim]),
            ("attn.norm_k.weight", vec![spec.head_dim]),
        ] {
            put(
                &mut expected,
                format!("{block}.{name}"),
                Dtype::BF16,
                shape,
                scope,
            );
        }
        if adaln {
            let base = format!("{block}.adaln_proj.linear");
            put(
                &mut expected,
                format!("{base}.weight"),
                Dtype::BF16,
                vec![modulation, spec.time_dim],
                TensorScope::Dormant,
            );
            put(
                &mut expected,
                format!("{base}.bias"),
                Dtype::BF16,
                vec![modulation],
                TensorScope::Dormant,
            );
        }
    }
    for (base, n, k) in spec.block_linears() {
        put(
            &mut expected,
            format!("{base}.weight"),
            Dtype::BF16,
            vec![n, k],
            scope,
        );
    }
    Ok(expected)
}

/// One official tensor produced by a GGUF tensor.
struct GgufTensorMap {
    official: String,
    /// Row range `[start, end)` of the official `[out, in]` tensor; `None` for the whole tensor.
    range: Option<(u64, u64)>,
    /// The source rows are the two halves of the official tensor in the opposite order (fused gate+up).
    swap_halves: bool,
}

/// GGUF tensors with no official MiniMax-H3 counterpart.
const UNUSED_GGUF_TENSORS: &[&str] = &["rope.inv_freq"];

/// Community GGUF names to official MiniMax-H3 names. The GGUF data is the official row-major flat buffer; a 2-D
/// tensor's declared dims are `[ne0=in, ne1=out]` (or a packed `[64, N]`/`[256, N]` view with the original input
/// dim in `comfy.gguf.orig_shape.<name>`), so no transpose is needed. `blocks.{i}` is a transformer block,
/// `token_refiner.blocks.{i}` a refiner block. The fused `attn.qkv_proj` is `[3*inner, hidden]` with q, k, v
/// contiguous along rows; the fused `mlp.fc1` is `[2*ffn, hidden]` with the two halves in the opposite order to the
/// official `ff.net.0.proj`.
fn map_gguf_tensor(spec: &crate::minimax_h3::H3Spec, name: &str) -> Result<Vec<GgufTensorMap>> {
    let width = (spec.heads * spec.head_dim) as u64;
    let mut out = Vec::new();
    if UNUSED_GGUF_TENSORS.contains(&name) {
        return Ok(out);
    }
    for (weight, bias, official) in [
        (
            "video_patch_proj.weight",
            "video_patch_proj.bias",
            "proj_in",
        ),
        (
            "audio_patch_proj.weight",
            "audio_patch_proj.bias",
            "audio_proj_in",
        ),
        (
            "condition_proj.weight",
            "condition_proj.bias",
            "context_embedder",
        ),
        (
            "time_embedder.proj_in.weight",
            "time_embedder.proj_in.bias",
            "time_embedder.linear_1",
        ),
        (
            "time_embedder.proj_out.weight",
            "time_embedder.proj_out.bias",
            "time_embedder.linear_2",
        ),
        (
            "final_layer.video_out.weight",
            "final_layer.video_out.bias",
            "proj_out",
        ),
        (
            "final_layer.audio_out.weight",
            "final_layer.audio_out.bias",
            "audio_proj_out",
        ),
        (
            "final_layer.adaln_proj.linear.weight",
            "final_layer.adaln_proj.linear.bias",
            "norm_out.linear",
        ),
    ] {
        if name == weight {
            out.push(GgufTensorMap {
                official: format!("{official}.weight"),
                range: None,
                swap_halves: false,
            });
            return Ok(out);
        }
        if name == bias {
            out.push(GgufTensorMap {
                official: format!("{official}.bias"),
                range: None,
                swap_halves: false,
            });
            return Ok(out);
        }
    }
    match name {
        "final_layer.norm.weight" => {
            out.push(GgufTensorMap {
                official: "norm_out.norm.weight".into(),
                range: None,
                swap_halves: false,
            });
            return Ok(out);
        }
        "token_refiner.final_norm.weight" => {
            out.push(GgufTensorMap {
                official: "token_refiner.final_norm.weight".into(),
                range: None,
                swap_halves: false,
            });
            return Ok(out);
        }
        _ => {}
    }
    for (prefix, official, layers, adaln) in [
        ("blocks.", "transformer_blocks.", spec.layers, true),
        (
            "token_refiner.blocks.",
            "token_refiner.refiner_blocks.",
            spec.refiner_layers,
            false,
        ),
    ] {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        let (index, suffix) = rest
            .split_once('.')
            .with_context(|| format!("GGUF block tensor {name} has no suffix"))?;
        let index: usize = index
            .parse()
            .with_context(|| format!("GGUF block index in {name}"))?;
        ensure!(
            index < layers,
            "GGUF block index {index} out of range in {name}"
        );
        let block = format!("{official}{index}");
        let mapped = match suffix {
            "attn.qkv_proj.weight" => {
                for (suffix, range) in [
                    ("to_q", (0, width)),
                    ("to_k", (width, 2 * width)),
                    ("to_v", (2 * width, 3 * width)),
                ] {
                    out.push(GgufTensorMap {
                        official: format!("{block}.attn.{suffix}.weight"),
                        range: Some(range),
                        swap_halves: false,
                    });
                }
                return Ok(out);
            }
            "attn.out_proj.weight" => format!("{block}.attn.to_out.0.weight"),
            "mlp.fc1.weight" => {
                out.push(GgufTensorMap {
                    official: format!("{block}.ff.net.0.proj.weight"),
                    range: None,
                    swap_halves: true,
                });
                return Ok(out);
            }
            "mlp.fc2.weight" => format!("{block}.ff.net.2.weight"),
            "norm1.weight" => format!("{block}.norm1.weight"),
            "norm2.weight" => format!("{block}.norm2.weight"),
            "attn.q_norm.weight" => format!("{block}.attn.norm_q.weight"),
            "attn.k_norm.weight" => format!("{block}.attn.norm_k.weight"),
            "adaln_proj.linear.weight" if adaln => {
                format!("{block}.adaln_proj.linear.weight")
            }
            "adaln_proj.linear.bias" if adaln => format!("{block}.adaln_proj.linear.bias"),
            _ => bail!("unknown MiniMax-H3 GGUF tensor {name}"),
        };
        out.push(GgufTensorMap {
            official: mapped,
            range: None,
            swap_halves: false,
        });
        return Ok(out);
    }
    bail!("unknown MiniMax-H3 GGUF tensor {name}")
}

/// `(block_len, block_bytes)` of a supported GGUF type; the block runs along `ne0`.
fn ggml_block(ggml_type: crate::gguf::GgmlType) -> Result<(u64, u64)> {
    match ggml_type {
        crate::gguf::GgmlType::F32 => Ok((1, 4)),
        crate::gguf::GgmlType::F16 | crate::gguf::GgmlType::Bf16 => Ok((1, 2)),
        crate::gguf::GgmlType::Q4_0 => Ok((32, 18)),
        other => bail!("unsupported MiniMax-H3 GGUF type {other:?}"),
    }
}

/// Layer name prefixes of the published checkpoints.
const LAYER_PREFIXES: [&str; 7] = [
    "model.layers.",
    "model.language_model.layers.",
    "language_model.layers.",
    "layers.",
    "model.decoder.layers.",
    "model.encoder.language_model.layers.",
    "transformer_blocks.",
];

pub type ExpectedTensor = (Dtype, Vec<usize>, TensorScope);

fn planned_bytes(expected: &BTreeMap<String, ExpectedTensor>, policy: LoadPolicy) -> Result<usize> {
    expected
        .iter()
        .try_fold(0usize, |sum, (name, (dtype, shape, scope))| {
            if !policy.selects_tensor(*scope, name) {
                return Ok(sum);
            }
            sum.checked_add(tensor_bytes(*dtype, shape)?)
                .context("resident size overflow")
        })
}

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
    projection_scoped(expected, base, n, k, fp8, TensorScope::Text);
}

fn nvfp4_projection(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    base: String,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "NVFP4 projection {base} requires positive N and K divisible by 16, got [{n}, {k}]"
    );
    put(
        expected,
        format!("{base}.weight"),
        Dtype::U8,
        vec![n, k / 2],
        TensorScope::Text,
    );
    put(
        expected,
        format!("{base}.weight_scale"),
        Dtype::F8_E4M3,
        vec![n, k / 16],
        TensorScope::Text,
    );
    put(
        expected,
        format!("{base}.weight_scale_2"),
        Dtype::F32,
        vec![],
        TensorScope::Text,
    );
    put(
        expected,
        format!("{base}.input_scale"),
        Dtype::F32,
        vec![],
        TensorScope::Text,
    );
    Ok(())
}

fn projection_scoped(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    base: String,
    n: usize,
    k: usize,
    fp8: Option<crate::fp8::Fp8ScaleLayout>,
    scope: TensorScope,
) {
    let dtype = if fp8.is_some() {
        Dtype::F8_E4M3
    } else {
        Dtype::BF16
    };
    put(expected, format!("{base}.weight"), dtype, vec![n, k], scope);
    if let Some(layout) = fp8 {
        put(
            expected,
            format!("{base}.{}", layout.scale_suffix()),
            layout.scale_dtype(),
            layout.scale_shape(n as u32, k as u32).to_vec(),
            scope,
        );
    }
}

/// Hyper-connection basis width of the published DeepSeek-V4.1 checkpoint.
const HC_BASIS: usize = 24;
/// Hyper-connection group count of the published DeepSeek-V4.1 checkpoint.
const HC_SCALE: usize = 3;
/// Rows of the published engram WKV matrix `[rows, hidden + ngram_size * head_dim]`.
const ENGRAM_WKV_ROWS: usize = 25_600;
/// Columns of the published MTP main projection `[hidden, columns]`.
const MAIN_PROJECTION_WIDTH: usize = 15_360;
/// Colour channels of the published DeepSeek-V4.1 patch embedding.
const VISION_CHANNELS: usize = 3;

fn deepseek_projection(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    base: &str,
    n: usize,
    k: usize,
    layout: crate::fp8::Fp8ScaleLayout,
    scope: TensorScope,
) {
    put(
        expected,
        format!("{base}.weight"),
        Dtype::F8_E4M3,
        vec![n, k],
        scope,
    );
    put(
        expected,
        format!("{base}.scale"),
        layout.scale_dtype(),
        layout.scale_shape(n as u32, k as u32).to_vec(),
        scope,
    );
}

fn deepseek_fp4_expert(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    base: &str,
    n: usize,
    k: usize,
    layout: crate::fp8::Fp8BlockScale,
    scope: TensorScope,
) {
    put(
        expected,
        format!("{base}.weight"),
        Dtype::I8,
        vec![n, k / 2],
        scope,
    );
    put(
        expected,
        format!("{base}.scale"),
        layout.dtype(),
        layout.scale_shape(n as u32, k as u32).to_vec(),
        scope,
    );
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
    Audio,
    Dormant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadPolicy {
    #[default]
    TextOnly,
    TextWithVision,
    /// For the standalone vision component; cannot construct a text Engine.
    VisionOnly,
    /// Audio component attached to an existing text engine.
    AudioOnly,
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
                | (Self::AudioOnly, TensorScope::Audio)
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
#[derive(Clone, Debug, serde::Serialize)]
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
    pub quant: Option<crate::gguf::GgmlType>,
    /// The GGUF source rows are the two halves of the official tensor in the opposite order (fused gate+up).
    pub swap_halves: bool,
}
impl TensorEntry {
    #[cfg(any(feature = "cuda", feature = "rocm", test))]
    pub(crate) fn native_byte_storage(&self) -> bool {
        matches!(self.dtype, Dtype::F8_E4M3 | Dtype::U8 | Dtype::F32)
            || (self.dtype == Dtype::BF16
                && matches!(self.role.as_str(), "hc_attn_fn" | "hc_ffn_fn"))
    }

    pub(crate) fn is_linear_projection(&self) -> bool {
        self.is_non_expert_loaded()
            && self.shape.len() == 2
            && (self.role.ends_with(".weight")
                || self.role == "self_attn.indexer.index_kpool_compress_gate")
    }

    pub(crate) fn is_routed_expert(&self) -> bool {
        // DiffusionGemma publishes the fused experts without the mlp. segment.
        self.role.starts_with("mlp.experts.") || self.role.starts_with("experts.")
    }

    pub(crate) fn is_non_expert_loaded(&self) -> bool {
        self.load && !self.is_routed_expert()
    }
}

fn split_layer_role(name: &str) -> Result<(Option<u32>, String)> {
    let layer = LAYER_PREFIXES
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
        .and_then(|s| s.split('.').next())
        .map(str::parse::<u32>)
        .transpose()?;
    let role = if let Some(layer) = layer {
        LAYER_PREFIXES
            .iter()
            .find_map(|prefix| name.strip_prefix(&format!("{prefix}{layer}.")))
            .or_else(|| name.strip_prefix(&format!("model.language_model.layers.{layer}.")))
            .unwrap_or(name)
            .to_owned()
    } else {
        name.to_owned()
    };
    Ok((layer, role))
}

fn is_embed_tokens_role(name: &str) -> bool {
    name == "embed_tokens.weight" || name.ends_with(".embed_tokens.weight")
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
/// Classic fused projection order `[q, k, v, gate, up, o, down]`; q/k/v and gate/up are contiguous.
pub(crate) fn classic_layer_names(layer: usize) -> [String; 7] {
    let p = format!("model.layers.{layer}");
    [
        format!("{p}.self_attn.q_proj.weight"),
        format!("{p}.self_attn.k_proj.weight"),
        format!("{p}.self_attn.v_proj.weight"),
        format!("{p}.mlp.gate_proj.weight"),
        format!("{p}.mlp.up_proj.weight"),
        format!("{p}.self_attn.o_proj.weight"),
        format!("{p}.mlp.down_proj.weight"),
    ]
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn validate_classic_order(layer: usize, order: &[String]) -> Result<()> {
    let want = classic_layer_names(layer);
    ensure!(
        order.len() == want.len() && order.iter().eq(want.iter()),
        "classic layer {layer} fused projection order differs"
    );
    Ok(())
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TensorManifest {
    pub entries: BTreeMap<String, TensorEntry>,
    pub resident_bytes: usize,
}

impl TensorManifest {
    pub(crate) fn validate_dsa_layer(
        &self,
        spec: &crate::model_spec::ModelSpec,
        layer: u32,
        prefix: &str,
    ) -> Result<()> {
        let mut expected = BTreeMap::new();
        MappedCheckpoint::glm5_next_dsa(
            spec,
            prefix.trim_end_matches('.'),
            &mut expected,
            TensorScope::Text,
        )?;
        for (name, (dtype, shape, scope)) in expected {
            let entry = self
                .entries
                .get(&name)
                .with_context(|| format!("DSA layer {layer} missing binding {name}"))?;
            ensure!(
                entry.load
                    && entry.scope == scope
                    && entry.layer == Some(layer)
                    && entry.role == name[prefix.len()..]
                    && entry.dtype == dtype
                    && entry.shape == shape
                    && entry.bytes == tensor_bytes(dtype, &shape)?
                    && entry.data_offsets[1].checked_sub(entry.data_offsets[0])
                        == Some(u64::try_from(entry.bytes)?),
                "DSA layer {layer} native binding mismatch: {name}"
            );
        }
        Ok(())
    }

    /// Loaded text-tower non-expert tensor bytes scanned during decode (excludes token embedding lookup).
    pub fn non_expert_decode_bytes(&self) -> Result<u64> {
        self.entries
            .values()
            .filter(|entry| {
                entry.is_non_expert_loaded()
                    && entry.scope == TensorScope::Text
                    && !is_embed_tokens_role(&entry.role)
            })
            .try_fold(0u64, |sum, entry| {
                sum.checked_add(u64::try_from(entry.bytes)?)
                    .context("non-expert decode byte count overflow")
            })
    }
}

impl MappedCheckpoint {
    #[cfg(test)]
    pub(crate) fn planned_non_expert_decode_bytes(
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<u64> {
        let mut expected = BTreeMap::new();
        if spec.architecture == crate::model_spec::Architecture::Glm5Next {
            Self::glm5_next_entries(spec, &mut expected)?;
        } else {
            Self::text_entries(spec, &mut expected)?;
        }
        let mut entries = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for (name, (dtype, shape, scope)) in expected {
            let bytes = tensor_bytes(dtype, &shape)?;
            let load = LoadPolicy::TextOnly.selects_tensor(scope, &name);
            if load {
                resident_bytes = resident_bytes
                    .checked_add(bytes)
                    .context("planned resident byte count overflow")?;
            }
            let (layer, role) = split_layer_role(&name)?;
            entries.insert(
                name,
                TensorEntry {
                    dtype,
                    shape,
                    shard: String::new(),
                    bytes,
                    load,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope,
                    layer,
                    role,
                    scale_name: None,
                    quant: None,
                    swap_halves: false,
                },
            );
        }
        TensorManifest {
            entries,
            resident_bytes,
        }
        .non_expert_decode_bytes()
    }
}

#[cfg(test)]
mod component_tests {
    use super::*;

    #[test]
    fn flat_llama_manifest_census_is_derived_from_config() -> Result<()> {
        let c = crate::config::Config {
            model_type: "llama".into(),
            hidden_size: 7168,
            intermediate_size: 22016,
            num_hidden_layers: 52,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 166912,
            max_position_embeddings: 524288,
            rms_norm_eps: 1e-5,
            rope_theta: 120000000.0,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: vec![166101],
            attention_bias: false,
            mlp_bias: false,
        };
        let spec = crate::model_spec::ModelSpec::from_llama(&c)?;
        let mut expected = BTreeMap::new();
        MappedCheckpoint::text_entries(&spec, &mut expected)?;
        ensure!(
            expected.len() == 471,
            "flat Llama census: {} tensors",
            expected.len()
        );
        ensure!(
            expected["model.embed_tokens.weight"].1 == [166912, 7168]
                && expected["lm_head.weight"].1 == [166912, 7168],
            "untied embed and lm_head"
        );
        ensure!(
            expected["model.layers.0.self_attn.q_proj.weight"].1 == [7168, 7168],
            "q_proj out width is q_width, not 2*q_width"
        );
        ensure!(
            expected["model.layers.0.mlp.down_proj.weight"].1 == [7168, 22016],
            "down_proj stored layout"
        );
        ensure!(
            !expected.keys().any(|k| k.contains("language_model")
                || k.contains("q_norm")
                || k.contains("k_norm")),
            "flat Llama has no language_model prefix or q/k norms"
        );
        ensure!(
            planned_bytes(&expected, LoadPolicy::TextOnly)? == 66237954048,
            "flat Llama published BF16 payload"
        );
        Ok(())
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    fn classic_fused_order_rejects_swapped_groups() -> Result<()> {
        let layer = 3;
        let correct = classic_layer_names(layer).to_vec();
        validate_classic_order(layer, &correct)?;
        let mut qk = correct.clone();
        qk.swap(0, 1);
        assert!(
            validate_classic_order(layer, &qk).is_err(),
            "q/k swap must be rejected"
        );
        let mut kv = correct.clone();
        kv.swap(1, 2);
        assert!(
            validate_classic_order(layer, &kv).is_err(),
            "k/v swap must be rejected"
        );
        let mut gu = correct.clone();
        gu.swap(3, 4);
        assert!(
            validate_classic_order(layer, &gu).is_err(),
            "gate/up swap must be rejected"
        );
        let mut qv = correct.clone();
        qv.swap(0, 2);
        assert!(
            validate_classic_order(layer, &qv).is_err(),
            "q/v swap must be rejected"
        );
        assert!(
            validate_classic_order(layer + 1, &correct).is_err(),
            "layer mismatch must be rejected"
        );
        Ok(())
    }

    #[test]
    fn music3_qwen3_manifest_geometry() -> Result<()> {
        let root = serde_json::from_str(include_str!("../tests/data/music3-qwen3-config.json"))?;
        let spec = crate::model_spec::ModelSpec::qwen3(&root)?;
        let mut expected = BTreeMap::new();
        MappedCheckpoint::text_entries(&spec, &mut expected)?;
        ensure!(
            expected.len() == 399
                && planned_bytes(&expected, LoadPolicy::TextOnly)? == 17_168_951_296,
            "Qwen3 manifest census"
        );
        ensure!(
            expected["model.layers.0.self_attn.q_proj.weight"].1 == [4096, 4096]
                && expected["model.layers.0.self_attn.k_norm.weight"].1 == [128],
            "Qwen3 ungated Q/QK norm layout"
        );
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only published checkpoint; set FF_MODEL to Music3"]
    fn music3_qwen3_manifest_rejects_missing_norm() -> Result<()> {
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?)
                .join("language_model");
        let spec = crate::model_spec::ModelSpec::read(&dir)?;
        let mapped = MappedCheckpoint::open(&dir)?;
        mapped.manifest(&spec, LoadPolicy::TextOnly)?;
        let mut expected = BTreeMap::new();
        MappedCheckpoint::text_entries(&spec, &mut expected)?;
        expected
            .remove("model.layers.0.self_attn.q_norm.weight")
            .context("missing Q norm")?;
        let bytes = planned_bytes(&expected, LoadPolicy::TextOnly)?;
        ensure!(
            mapped
                .assemble_manifest(
                    "Qwen3 norm control",
                    expected,
                    LoadPolicy::TextOnly,
                    bytes,
                    None
                )
                .is_err(),
            "Qwen3 missing Q norm was accepted"
        );
        println!("PASS music3_qwen3_manifest missing_norm_control=rejected");
        Ok(())
    }

    fn binding_literals<'a>(source: &'a str, begin: &str, end: &str) -> Result<Vec<&'a str>> {
        let body = source
            .split_once(begin)
            .with_context(|| format!("binding census scope missing: {begin}"))?
            .1
            .split_once(end)
            .with_context(|| format!("binding census scope end missing: {end}"))?
            .0;
        Ok(body.split('"').skip(1).step_by(2).collect())
    }

    fn glm_consumer_roles(
        mixer: crate::model_spec::Mixer,
        dense: bool,
    ) -> Result<BTreeSet<String>> {
        let engine = include_str!("backend/cuda/engine_qwen.rs");
        let linear = include_str!("backend/cuda/linear.rs");
        let gdn = include_str!("backend/cuda/gdn.rs");
        let mut roles = BTreeSet::new();
        for role in binding_literals(engine, "layers.push(Layer {", "pre_ffn_norm: gemma")? {
            if role.ends_with(".weight") {
                roles.insert(role.to_owned());
            }
        }
        let hc = binding_literals(
            engine,
            "fn take_glm_hc_weights(",
            "pub(super) struct GlmDsaOutputs",
        )?;
        for base in hc.iter().filter(|name| name.starts_with("hc_")) {
            for suffix in hc.iter().filter_map(|name| name.strip_prefix("{base}")) {
                roles.insert(format!("{base}{suffix}"));
            }
        }
        let (begin, end) = if mixer == crate::model_spec::Mixer::FullAttention {
            ("fn glm_dsa_mixer(", "pub(super) fn qwen_mixer(")
        } else {
            ("fn project_kda(", "pub(super) fn matrix(")
        };
        for role in binding_literals(engine, begin, end)? {
            if role.starts_with("self_attn.") {
                roles.insert(role.to_owned());
            }
        }
        if mixer == crate::model_spec::Mixer::FullAttention {
            let aux = binding_literals(engine, "let aux = |name: &str|", "} else if flint {")?;
            for role in aux {
                if let Some(role) = role.strip_prefix("{prefix}") {
                    if role != "self_attn.{name}" {
                        roles.insert(role.to_owned());
                    }
                } else if role.ends_with(".weight") || role.ends_with(".bias") {
                    roles.insert(format!("self_attn.{role}"));
                }
            }
        } else {
            for role in binding_literals(
                engine,
                "pub(super) fn qwen_mixer(",
                "pub(super) fn run_moe_if_bound(",
            )? {
                if role.starts_with("self_attn.") {
                    roles.insert(role.to_owned());
                }
            }
            for role in binding_literals(engine, "Mixer::Gdn => {", "layers.push(Layer {")? {
                if role.starts_with("self_attn.") && role.ends_with(".weight") {
                    roles.insert(role.to_owned());
                }
            }
            for role in binding_literals(
                gdn,
                "impl GdnDecayWeights {",
                "pub(crate) enum GdnConvWeights",
            )? {
                if let Some(role) = role.strip_prefix("{prefix}") {
                    roles.insert(format!("self_attn.{role}"));
                }
            }
            for role in binding_literals(
                gdn,
                "impl GdnConvWeights {",
                "pub(crate) struct GdnBindings",
            )? {
                if role.ends_with("_conv1d") {
                    roles.insert(format!("self_attn.{role}.weight"));
                }
            }
        }
        let ffn = if dense {
            binding_literals(engine, "O::GateUp => {", "O::Head => {")?
        } else {
            binding_literals(linear, "pub fn bind(spec:", "let shared =")?
        };
        for role in ffn {
            if role.starts_with("mlp.") {
                roles.insert(role.to_owned());
            }
        }
        if !dense {
            for role in binding_literals(engine, "impl NativeFfn {", "enum BoundMixer")? {
                if let Some(role) = role.strip_prefix("model.language_model.layers.{layer}.") {
                    roles.insert(role.to_owned());
                }
            }
        }
        ensure!(!roles.is_empty(), "empty GLM consumer role census");
        Ok(roles)
    }

    #[test]
    fn glm_consumer_roles_exclude_conditional_gemma_norms() -> Result<()> {
        use crate::model_spec::Mixer;
        for mixer in [Mixer::FullAttention, Mixer::Gdn] {
            for dense in [false, true] {
                let roles = glm_consumer_roles(mixer, dense)?;
                for role in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                    ensure!(roles.contains(role), "GLM norm role missing: {role}");
                }
                for role in [
                    "pre_feedforward_layernorm.weight",
                    "post_feedforward_layernorm.weight",
                    "post_per_layer_input_norm.weight",
                ] {
                    ensure!(
                        !roles.contains(role),
                        "Gemma-only norm in GLM census: {role}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires the published GLM safetensors index"]
    fn glm_production_binding_index() -> Result<()> {
        let directory = std::env::var_os("FF_GLM_MODEL").context("FF_GLM_MODEL is required")?;
        let directory = Path::new(&directory);
        let spec = crate::model_spec::ModelSpec::read(directory)?;
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Glm5Next,
            "GLM census architecture mismatch"
        );
        let index: Index = serde_json::from_slice(&std::fs::read(
            directory.join("model.safetensors.index.json"),
        )?)?;
        let mut inventory = BTreeMap::new();
        MappedCheckpoint::glm5_next_entries(&spec, &mut inventory)?;
        let mut requested = BTreeSet::new();
        let mut missing = Vec::new();
        let mut resolve = |name: String| {
            if requested.insert(name.clone()) && !index.weight_map.contains_key(&name) {
                missing.push(name);
            }
        };
        for role in binding_literals(
            include_str!("backend/cuda/weights.rs"),
            "pub fn take_globals(",
            "pub fn load(",
        )? {
            if role.starts_with("model.language_model.") || role == "lm_head.weight" {
                resolve(role.to_owned());
            }
        }
        for (layer, plan) in spec.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{layer}.");
            for role in
                glm_consumer_roles(plan.mixer, plan.ffn == crate::model_spec::FfnKind::Dense)?
            {
                resolve(format!("{prefix}{role}"));
            }
            for (name, (_, _, scope)) in &inventory {
                if *scope == TensorScope::Text && name.starts_with(&prefix) {
                    resolve(name.clone());
                }
            }
        }
        let first = missing.first().map_or("none", String::as_str);
        println!(
            "GLM_BINDING_CENSUS requested={} resolved={} missing={} first_missing={first} layers={} upload_bytes=0",
            requested.len(),
            requested.len() - missing.len(),
            missing.len(),
            spec.layers.len()
        );
        ensure!(
            missing.is_empty(),
            "GLM binding requests are missing from the official index: {first}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires the published GLM checkpoint"]
    fn glm_published_native_storage() -> Result<()> {
        let root = std::env::var_os("FF_GLM_MODEL").context("FF_GLM_MODEL is required")?;
        let root = std::path::Path::new(&root);
        let spec = crate::model_spec::ModelSpec::read(root)?;
        let mapped = MappedCheckpoint::open(root)?;
        let manifest = mapped.manifest(&spec, LoadPolicy::TextOnly)?;
        let mut counts = [0usize; 2];
        for (name, entry) in &manifest.entries {
            if !entry.load || !entry.role.starts_with("hc_") {
                continue;
            }
            ensure!(
                entry.native_byte_storage(),
                "mHC native storage missing: {name}"
            );
            match entry.dtype {
                Dtype::BF16 => counts[0] += 1,
                Dtype::F32 => counts[1] += 1,
                dtype => anyhow::bail!("unexpected published mHC dtype {dtype:?}: {name}"),
            }
        }
        ensure!(
            counts == [90, 180],
            "published mHC census mismatch: {counts:?}"
        );
        println!("PASS glm_published_native_storage bf16_fn=90 f32=180 conversion=0");
        Ok(())
    }

    #[test]
    fn glm_native_manifest_preserves_roles_and_projection_extents() -> Result<()> {
        let root = serde_json::from_str(include_str!("../tests/data/glm53-flash-config.json"))?;
        let mut spec = crate::model_spec::ModelSpec::glm5_next(&root, &serde_json::Value::Null)?;
        let mut tensors = BTreeMap::new();
        MappedCheckpoint::glm5_next_entries(&spec, &mut tensors)?;
        assert_eq!(tensors.len(), 75_761);
        for (scope, count, bytes) in [
            (TensorScope::Text, 74_001, 319_706_118_392usize),
            (TensorScope::Mtp, 1_760, 7_493_399_168),
        ] {
            let entries: Vec<_> = tensors.values().filter(|entry| entry.2 == scope).collect();
            assert_eq!(entries.len(), count);
            let total = entries.iter().try_fold(0usize, |sum, (dtype, shape, _)| {
                sum.checked_add(tensor_bytes(*dtype, shape)?)
                    .context("manifest byte overflow")
            })?;
            assert_eq!(total, bytes);
        }
        let mut manifest = TensorManifest {
            entries: BTreeMap::new(),
            resident_bytes: 0,
        };
        for (name, (dtype, shape, scope)) in &tensors {
            let (layer, role) = split_layer_role(name)?;
            if !layer.is_some_and(|i| (i + 1).is_multiple_of(4)) || !role.starts_with("self_attn.")
            {
                continue;
            }
            let bytes = tensor_bytes(*dtype, shape)?;
            manifest.entries.insert(
                name.clone(),
                TensorEntry {
                    dtype: *dtype,
                    shape: shape.clone(),
                    shard: "native.safetensors".into(),
                    bytes,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope: *scope,
                    layer,
                    role,
                    scale_name: (*dtype == Dtype::F8_E4M3).then(|| format!("{}_scale_inv", name)),
                    quant: None,
                    swap_halves: false,
                },
            );
            manifest.resident_bytes += bytes;
        }
        for layer in (3..45).step_by(4) {
            let plan = crate::linear_plan::LayerDequantPlan::from_model(&manifest, layer, &spec)?;
            assert!(plan.slots.is_empty());
            assert_eq!(plan.bytes, 0);
            assert_eq!(
                manifest
                    .entries
                    .iter()
                    .filter(|(name, entry)| name.starts_with(&plan.prefix)
                        && entry.dtype == Dtype::F8_E4M3)
                    .count(),
                4
            );
            let admitted: Vec<_> = manifest
                .entries
                .iter()
                .filter(|(name, e)| name.starts_with(&plan.prefix) && e.is_linear_projection())
                .collect();
            assert_eq!(admitted.len(), 9);
            let old_suffix_count = admitted
                .iter()
                .filter(|(name, _)| name.ends_with(".weight"))
                .count();
            assert_eq!(old_suffix_count, 8);
            let gate = format!("{}self_attn.indexer.index_kpool_compress_gate", plan.prefix);
            assert!(manifest.entries[&gate].is_linear_projection());
            let entry = manifest.entries.remove(&gate).unwrap();
            assert!(
                crate::linear_plan::LayerDequantPlan::from_model(&manifest, layer, &spec).is_err()
            );
            manifest.entries.insert(gate.clone(), entry);
            manifest.entries.get_mut(&gate).unwrap().shape.swap(0, 1);
            assert!(
                crate::linear_plan::LayerDequantPlan::from_model(&manifest, layer, &spec).is_err()
            );
            manifest.entries.get_mut(&gate).unwrap().shape.swap(0, 1);
            manifest.entries.get_mut(&gate).unwrap().load = false;
            assert!(
                crate::linear_plan::LayerDequantPlan::from_model(&manifest, layer, &spec).is_err()
            );
            manifest.entries.get_mut(&gate).unwrap().load = true;
        }
        println!(
            "PASS glm_dsa_native_binding layers=11 projections_per_layer=9 fp8_slots=4 old_suffix_control=8 missing_shape_unloaded_controls=33"
        );
        for (layer, role, dtype, shape) in [
            (2, "hc_attn_fn", Dtype::BF16, vec![24, 16384]),
            (2, "hc_attn_base", Dtype::F32, vec![24]),
            (2, "self_attn.q_proj.weight", Dtype::BF16, vec![8192, 4096]),
            (
                2,
                "self_attn.q_conv1d.weight",
                Dtype::BF16,
                vec![8192, 1, 4],
            ),
            (2, "self_attn.A_log", Dtype::F32, vec![64]),
            (2, "self_attn.dt_bias", Dtype::F32, vec![8192]),
            (
                3,
                "self_attn.q_a_proj.weight",
                Dtype::F8_E4M3,
                vec![1536, 4096],
            ),
            (
                3,
                "self_attn.q_a_proj.weight_scale_inv",
                Dtype::F32,
                vec![12, 32],
            ),
            (
                3,
                "self_attn.kv_b_proj.weight",
                Dtype::BF16,
                vec![32768, 512],
            ),
            (
                3,
                "self_attn.o_proj.weight",
                Dtype::F8_E4M3,
                vec![4096, 16384],
            ),
            (3, "mlp.gate.weight", Dtype::BF16, vec![288, 4096]),
            (3, "mlp.gate.e_score_correction_bias", Dtype::F32, vec![288]),
        ] {
            let name = format!("model.language_model.layers.{layer}.{role}");
            assert_eq!(tensors[&name], (dtype, shape, TensorScope::Text), "{name}");
            assert_eq!(split_layer_role(&name)?, (Some(layer), role.to_owned()));
        }
        let mut hc_native = 0;
        let mut old_dtype_control = 0;
        for (name, (dtype, shape, scope)) in &tensors {
            let (layer, role) = split_layer_role(name)?;
            if !role.starts_with("hc_") {
                continue;
            }
            let bytes = tensor_bytes(*dtype, shape)?;
            let entry = TensorEntry {
                dtype: *dtype,
                shape: shape.clone(),
                shard: "native.safetensors".into(),
                bytes,
                load: *scope == TensorScope::Text,
                data_start: 0,
                data_offsets: [0, bytes as u64],
                scope: *scope,
                layer,
                role,
                scale_name: None,
                quant: None,
                swap_halves: false,
            };
            assert!(
                entry.native_byte_storage(),
                "mHC native storage missing: {name}"
            );
            hc_native += 1;
            old_dtype_control += usize::from(*dtype == Dtype::BF16);
            if *dtype == Dtype::BF16 {
                let mut ordinary = entry;
                ordinary.role = "self_attn.q_proj.weight".into();
                assert!(
                    !ordinary.native_byte_storage(),
                    "ordinary BF16 changed storage"
                );
            }
        }
        assert_eq!(hc_native, 270);
        assert_eq!(old_dtype_control, 90);
        println!("PASS glm_hc_native_storage entries=270 bf16_fn=90 f32=180 old_dtype_control=90");
        spec.glm5_next.as_mut().unwrap().v_head_dim = 128;
        tensors.clear();
        MappedCheckpoint::glm5_next_dsa(&spec, "layer", &mut tensors, TensorScope::Text)?;
        assert_eq!(tensors["layer.self_attn.kv_b_proj.weight"].1, [24576, 512]);
        assert_eq!(tensors["layer.self_attn.o_proj.weight"].1, [4096, 8192]);
        spec.glm5_next.as_mut().unwrap().v_head_dim = u32::MAX;
        assert!(
            MappedCheckpoint::glm5_next_dsa(&spec, "layer", &mut tensors, TensorScope::Text)
                .is_err()
        );
        spec.glm5_next.as_mut().unwrap().hc_mult = u32::MAX;
        assert!(MappedCheckpoint::glm5_next_entries(&spec, &mut tensors).is_err());
        println!(
            "PASS glm_native_manifest text=74001 mtp=1760 native_roles=12 asymmetric_value_control=true overflow_controls=2"
        );
        Ok(())
    }

    #[test]
    fn complex_checkpoint_ranges_are_checked() -> Result<()> {
        let make = |shape: usize, start: usize, stop: usize| -> Result<Vec<u8>> {
            let header = serde_json::to_vec(&serde_json::json!({
                "phases": {"dtype":"C64", "shape":[shape], "data_offsets":[start,stop]}
            }))?;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            for value in [0.0f32, 1.0, -1.0, 0.0] {
                bytes.extend(value.to_le_bytes());
            }
            Ok(bytes)
        };
        let bytes = make(2, 0, 16)?;
        let tensors = published_tensors(&bytes)?;
        assert_eq!(tensors["phases"].shape, [2]);
        assert_eq!(tensors["phases"].data.len(), 16);
        assert_eq!(
            f32::from_le_bytes(tensors["phases"].data[8..12].try_into()?),
            -1.0
        );
        assert!(published_tensors(&make(2, 0, 8)?).is_err());
        assert!(published_tensors(&make(2, 8, 24)?).is_err());
        assert!(published_tensors(&make(usize::MAX, 0, 16)?).is_err());
        let root = std::env::temp_dir().join(format!("ff-indexed-packed-{}", std::process::id()));
        std::fs::create_dir(&root)?;
        let write_shard = |name: &str, dtype: &str, bytes: usize| -> Result<()> {
            let header = serde_json::to_vec(&serde_json::json!({
                name: {"dtype":dtype,"shape":[4],"data_offsets":[0,bytes]}
            }))?;
            let mut data = (header.len() as u64).to_le_bytes().to_vec();
            data.extend(header);
            data.resize(data.len() + bytes, 0);
            std::fs::write(root.join(format!("{name}.safetensors")), data)?;
            Ok(())
        };
        write_shard("packed", "F4", 2)?;
        write_shard("float8", "F8_E4M3", 4)?;
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":{
                "packed":"packed.safetensors","float8":"float8.safetensors"
            }}))?,
        )?;
        let checkpoint = MappedCheckpoint::open(&root)?;
        let tensors = checkpoint.component_tensors()?;
        assert_eq!(tensors["packed"].data.len(), 2);
        assert_eq!(tensors["float8"].data.len(), 4);
        drop(tensors);
        drop(checkpoint);
        write_shard("packed", "F4", 3)?;
        assert!(MappedCheckpoint::open(&root).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

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
        std::fs::write(
            root.join("config.json"),
            include_str!("../tests/data/q38-config.json"),
        )?;
        assert!(has_mtp_component(&root)?);
        std::fs::write(
            root.join("config.json"),
            include_str!("../tests/data/deepseek41-flash-config.json"),
        )?;
        assert!(
            !has_mtp_component(&root)?,
            "DeepSeek-V4.1 mtp.* tensors selected an unbound MTP decode"
        );
        std::fs::write(
            &path,
            br#"{"weight_map":{"mtp.fc.weight":"a","mtp.fc.weight":"b"}}"#,
        )?;
        assert!(has_mtp_component(&root).is_err());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    fn diffusion_gemma_spec() -> Result<crate::model_spec::ModelSpec> {
        let root: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/data/diffusiongemma-26b-a4b-config.json"
        ))?;
        crate::model_spec::ModelSpec::diffusion_gemma(
            &root,
            &serde_json::json!({
                "eos_token_id": [1, 106, 50],
                "max_denoising_steps": 48,
                "t_min": 0.4,
                "t_max": 0.8,
                "confidence_threshold": 0.005,
                "sampler_config": {"entropy_bound": 0.1},
                "pad_token_id": 0,
            }),
        )
    }

    #[test]
    fn diffusion_gemma_plan_matches_the_published_tensor_counts() -> Result<()> {
        let spec = diffusion_gemma_spec()?;
        ensure!(
            spec.canvas_length == Some(256)
                && spec.moe.as_ref().is_some_and(|moe| {
                    moe.experts == 128
                        && moe.experts_per_tok == 8
                        && moe.expert_intermediate == 704
                        && moe.shared_intermediate == 2112
                        && !moe.shared_gate
                })
                && spec.layers.len() == 30
                && spec
                    .layers
                    .iter()
                    .filter(|l| l.sliding_window.is_some())
                    .count()
                    == 25
                && spec.layers[5].geometry.as_ref().unwrap().kv_heads == Some(2)
                && spec.layers[0].geometry.as_ref().unwrap().kv_heads.is_none()
                && spec.layers[5].geometry.as_ref().unwrap().head_dim == 512
                && spec.layers[0].geometry.as_ref().unwrap().head_dim == 256
                && spec.eos == [1, 50, 106],
            "DiffusionGemma spec drift: {spec:?}"
        );
        let mut expected = BTreeMap::new();
        MappedCheckpoint::diffusion_gemma_entries(&spec, &mut expected)?;
        let count = |scope| expected.values().filter(|e| e.2 == scope).count();
        ensure!(
            count(TensorScope::Text) == 691
                && MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?
                    == 50_501_973_624usize,
            "DiffusionGemma plan: text={} resident={}",
            count(TensorScope::Text),
            MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?,
        );
        for name in [
            "model.decoder.embed_tokens.weight",
            "model.decoder.self_conditioning.down_proj.weight",
            "model.decoder.layers.0.router.scale",
            "model.decoder.layers.0.router.per_expert_scale",
            "model.decoder.layers.0.experts.gate_up_proj",
            "model.decoder.layers.0.experts.down_proj",
            "model.decoder.layers.0.layer_scalar",
            "model.encoder.language_model.layers.29.layer_scalar",
            "model.decoder.layers.0.self_attn.v_proj.weight",
        ] {
            ensure!(
                expected.contains_key(name),
                "missing expected tensor {name}"
            );
        }
        for name in [
            "lm_head.weight",
            "model.encoder.language_model.embed_tokens.weight",
            "model.encoder.language_model.layers.0.mlp.up_proj.weight",
            "model.decoder.layers.5.self_attn.v_proj.weight",
            "model.decoder.layers.0.mlp.experts.gate_up_proj",
        ] {
            ensure!(
                !expected.contains_key(name),
                "tied or absent tensor must not be expected: {name}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "published checkpoint; set FF_MODEL; CPU only"]
    fn published_diffusion_gemma_manifest_accounts_for_all_components() -> Result<()> {
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let spec = crate::model_spec::ModelSpec::read(&dir)?;
        let checkpoint = MappedCheckpoint::open(&dir)?;
        let manifest = checkpoint.manifest(&spec, LoadPolicy::TextOnly)?;
        let count = |scope| {
            manifest
                .entries
                .values()
                .filter(|entry| entry.scope == scope)
                .count()
        };
        ensure!(
            manifest.entries.len() == 1047
                && count(TensorScope::Text) == 691
                && count(TensorScope::Vision) == 356
                && manifest.resident_bytes == 50_501_973_624usize
                && manifest
                    .entries
                    .values()
                    .all(|e| e.load == (e.scope == TensorScope::Text)),
            "DiffusionGemma manifest: entries={} vision={} resident={}",
            manifest.entries.len(),
            count(TensorScope::Vision),
            manifest.resident_bytes
        );
        // Tier conservation (4595b03e form against the DiffusionGemma
        // manifest): every planned byte is accounted exactly once, both
        // capacity bounds hold, and the zero-expert-space control errors.
        for (device_gib, host_gib, compute_units) in [(22u64, 40u64, 128), (10, 8, 48), (3, 3, 16)]
        {
            let resources = crate::experts::plan::BlockResources {
                device_free_bytes: device_gib << 30,
                host_available_bytes: host_gib << 30,
                device_work_bytes: 512 << 20,
                host_work_bytes: 128 << 20,
                compute_units,
                copy_engines: 2,
                rows: 32,
                staging_slots_max: 16,
            };
            let plan = crate::experts::plan::BlockResidency::new(&spec, &manifest, resources)?;
            let slot = plan.experts.slot_bytes();
            let gpu = plan.device_fixed_bytes
                + plan.dense.resident_bytes as u64
                + 2 * plan.dense.slot_bytes as u64
                + u64::from(plan.experts.slots_total() + plan.experts.prefill_staging_slots) * slot;
            let host = plan.host_fixed_bytes
                + plan.dense.host_bytes as u64
                + u64::from(plan.experts.host_blocks + plan.experts.spare_blocks) * slot;
            ensure!(
                gpu <= resources.device_free_bytes && host <= resources.host_available_bytes,
                "DiffusionGemma model tier capacity exceeded"
            );
            let unique = plan.global_dense_bytes
                + plan.dense.resident_bytes as u64
                + plan.dense.host_bytes as u64
                + plan.dense.disk_bytes as u64
                + u64::from(plan.experts.slots_total() + plan.experts.host_blocks) * slot
                + plan.experts.disk_bytes;
            ensure!(
                unique == manifest.resident_bytes as u64,
                "DiffusionGemma model tier conservation"
            );
            let mut invalid = resources;
            invalid.device_free_bytes = plan.global_dense_bytes + resources.device_work_bytes;
            ensure!(
                crate::experts::plan::BlockResidency::new(&spec, &manifest, invalid).is_err(),
                "DiffusionGemma capacity control accepted zero expert space"
            );
        }
        let layer = "model.decoder.layers.0.experts.gate_up_proj".to_owned();
        type Mutation = fn(&mut BTreeMap<String, ExpectedTensor>, &str);
        let mutants: [(&str, Mutation); 3] = [
            ("missing", |e, name| {
                e.remove(name);
            }),
            ("shape", |e, name| {
                e.get_mut(name).expect("expected tensor").1[0] += 1;
            }),
            ("extra", |e, _| {
                put(
                    e,
                    "model.decoder.layers.0.extra.weight".into(),
                    Dtype::BF16,
                    vec![1],
                    TensorScope::Text,
                );
            }),
        ];
        for (label, mutate) in mutants {
            let mut expected = checkpoint.diffusion_gemma_expected(&spec)?;
            mutate(&mut expected, &layer);
            let planned = MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?;
            ensure!(
                checkpoint
                    .assemble_manifest(
                        "DiffusionGemma",
                        expected,
                        LoadPolicy::TextOnly,
                        planned,
                        None,
                    )
                    .is_err(),
                "{label} mutant was accepted"
            );
        }
        Ok(())
    }

    #[test]
    fn clover_manifest_counts_all_side_weights() -> Result<()> {
        let root = serde_json::from_str(include_str!("../tests/fixtures/clover_config.json"))?;
        let spec = crate::model_spec::ModelSpec::clover(
            &root,
            &serde_json::json!({"eos_token_id": [248046, 248044]}),
        )?;
        let mut expected = BTreeMap::new();
        MappedCheckpoint::text_entries(&spec, &mut expected)?;
        ensure!(expected.len() == 813, "Clover tensor count");
        ensure!(
            MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)? == 69_322_701_696,
            "Clover published BF16 payload"
        );
        for (suffix, shape, count) in [
            ("conv_right_weight", vec![8192, 1, 3], 30),
            ("bd_softmax_logit_scale", vec![32], 30),
            ("bd_softmax_history_bias", vec![32], 30),
            ("bd_softmax_history_c", vec![32], 30),
        ] {
            let entries: Vec<_> = expected
                .iter()
                .filter(|(name, _)| name.ends_with(suffix))
                .collect();
            ensure!(entries.len() == count, "Clover side tensor count: {suffix}");
            ensure!(
                entries
                    .iter()
                    .all(|(_, entry)| entry.0 == Dtype::BF16 && entry.1 == shape),
                "Clover side tensor layout: {suffix}"
            );
        }
        ensure!(
            spec.rotary_dim == 64 && spec.head_dim == 256 && spec.q_width == 4096,
            "Clover attention geometry"
        );
        let work = crate::work_layout::RowWorkSpec::new(&spec)?;
        ensure!(
            work.row_bytes()? == 6 * spec.hidden as usize + 4,
            "Clover common workspace contains only base rows"
        );
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL to the pinned Clover directory"]
    fn clover_manifest_matches_checkpoint_and_rejects_side_weight_mutants() -> Result<()> {
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let spec = crate::model_spec::ModelSpec::read(&dir)?;
        ensure!(
            spec.architecture == crate::model_spec::Architecture::Clover,
            "expected Clover"
        );
        let checkpoint = MappedCheckpoint::open(&dir)?;
        let manifest = checkpoint.manifest(&spec, LoadPolicy::TextOnly)?;
        ensure!(
            manifest.entries.len() == 813 && manifest.resident_bytes == 69_322_701_696,
            "Clover manifest extent"
        );
        let catalogue = crate::experts::catalogue::ExpertCatalogue::from_manifest(
            &manifest,
            crate::experts::catalogue::ExpertGeometry::from_model(&spec)?,
            spec.expert_format.context("Clover expert format missing")?,
        )?;
        ensure!(
            catalogue.slot_bytes() == 6_291_456
                && catalogue.geometry().layers == 40
                && catalogue.geometry().experts == 256,
            "Clover native expert catalogue extent"
        );
        for (device_gib, host_gib, compute_units) in [(22u64, 40u64, 128), (10, 8, 48), (3, 3, 16)]
        {
            let resources = crate::experts::plan::BlockResources {
                device_free_bytes: device_gib << 30,
                host_available_bytes: host_gib << 30,
                device_work_bytes: 512 << 20,
                host_work_bytes: 128 << 20,
                compute_units,
                copy_engines: 2,
                rows: 32,
                staging_slots_max: spec.moe.as_ref().context("block MoE missing")?.experts,
            };
            let plan = crate::experts::plan::BlockResidency::new(&spec, &manifest, resources)?;
            let slot = plan.experts.slot_bytes();
            let gpu = plan.device_fixed_bytes
                + plan.dense.resident_bytes as u64
                + 2 * plan.dense.slot_bytes as u64
                + u64::from(plan.experts.slots_total() + plan.experts.prefill_staging_slots) * slot;
            let host = plan.host_fixed_bytes
                + plan.dense.host_bytes as u64
                + u64::from(plan.experts.host_blocks + plan.experts.spare_blocks) * slot;
            ensure!(
                gpu <= resources.device_free_bytes && host <= resources.host_available_bytes,
                "Clover model tier capacity exceeded"
            );
            let unique = plan.global_dense_bytes
                + plan.dense.resident_bytes as u64
                + plan.dense.host_bytes as u64
                + plan.dense.disk_bytes as u64
                + u64::from(plan.experts.slots_total() + plan.experts.host_blocks) * slot
                + plan.experts.disk_bytes;
            ensure!(
                unique == manifest.resident_bytes as u64,
                "Clover model tier conservation"
            );
            ensure!(
                plan.experts.span_rows == 32 && plan.experts.disk_bytes > 0,
                "Clover bounded tiers changed block extent or lost disk tier"
            );
            ensure!(
                (device_gib == 3) == (plan.dense.slot_bytes > 0),
                "Clover dense slot derivation"
            );
            let mut invalid = resources;
            invalid.device_free_bytes = plan.global_dense_bytes + resources.device_work_bytes;
            ensure!(
                crate::experts::plan::BlockResidency::new(&spec, &manifest, invalid).is_err(),
                "Clover capacity control accepted zero expert space"
            );
            println!(
                "PASS clover_model_capacity_fixture device_gib={device_gib} host_gib={host_gib} gpu_bytes={gpu} host_bytes={host} dense_slot_bytes={} unique_bytes={unique}",
                plan.dense.slot_bytes
            );
        }
        for mutation in 0..3 {
            let mut expected = BTreeMap::new();
            MappedCheckpoint::text_entries(&spec, &mut expected)?;
            let name = "model.layers.0.linear_attn.conv_right_weight";
            match mutation {
                0 => {
                    expected
                        .remove(name)
                        .context("side weight missing from plan")?;
                }
                1 => {
                    expected
                        .get_mut(name)
                        .context("side weight missing from plan")?
                        .1[2] += 1
                }
                _ => {
                    put(
                        &mut expected,
                        "model.extra.weight".into(),
                        Dtype::BF16,
                        vec![1],
                        TensorScope::Text,
                    );
                }
            }
            let planned = MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?;
            ensure!(
                checkpoint
                    .assemble_manifest("Clover", expected, LoadPolicy::TextOnly, planned, None)
                    .is_err(),
                "Clover accepted side-weight mutant {mutation}"
            );
        }
        println!(
            "PASS clover_manifest tensors=813 bytes=69322701696 side_tensors=120 catalogue_slots=10240 controls=3"
        );
        Ok(())
    }

    fn gemma4_spec(text: &str, shared: Option<u32>) -> Result<crate::model_spec::ModelSpec> {
        let mut root: serde_json::Value = serde_json::from_str(text)?;
        if let Some(shared) = shared {
            root["text_config"]["num_kv_shared_layers"] = shared.into();
        }
        crate::model_spec::ModelSpec::gemma4(&root, &serde_json::json!({"eos_token_id": 1}))
    }

    #[test]
    fn gemma4_plan_matches_the_published_tensor_counts() -> Result<()> {
        for (text, text_tensors, dormant, resident) in [
            (
                include_str!("../tests/data/gemma4-e2b-config.json"),
                600usize,
                60usize,
                9_257_138_758usize,
            ),
            (
                include_str!("../tests/data/gemma4-e4b-config.json"),
                719,
                54,
                14_926_026_836,
            ),
            (
                include_str!("../tests/data/gemma4-12b-config.json"),
                666,
                0,
                23_814_700_640,
            ),
            (
                include_str!("../tests/data/gemma4-31b-config.json"),
                832,
                0,
                61_394_690_680,
            ),
        ] {
            let spec = gemma4_spec(text, None)?;
            let mut expected = BTreeMap::new();
            MappedCheckpoint::gemma4_entries(&spec, &mut expected)?;
            let count = |scope| expected.values().filter(|e| e.2 == scope).count();
            ensure!(
                count(TensorScope::Text) + count(TensorScope::Dormant) == text_tensors
                    && count(TensorScope::Dormant) == dormant
                    && MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?
                        == resident,
                "Gemma 4 plan: text={} dormant={}",
                count(TensorScope::Text),
                count(TensorScope::Dormant)
            );
            ensure!(
                !expected.contains_key("lm_head.weight"),
                "tied head must not expect lm_head"
            );
            if spec.per_layer_input.is_none() {
                ensure!(
                    !expected.keys().any(|key| key.contains("per_layer")),
                    "large Gemma manifest contains per-layer input weights"
                );
                for (index, layer) in spec.layers.iter().enumerate() {
                    let name =
                        format!("model.language_model.layers.{index}.self_attn.v_proj.weight");
                    ensure!(
                        expected.contains_key(&name) == layer.sliding_window.is_some(),
                        "shared projection binding: {name}"
                    );
                }
                let vision = spec.vision.as_ref().context("large vision spec")?;
                let (vision_count, vision_bytes) = if vision.unified.is_some() {
                    MappedCheckpoint::unified_vision_entries(vision, &mut expected)?;
                    (10, 99_844_608)
                } else {
                    MappedCheckpoint::axial_vision_entries(
                        vision,
                        "model.vision_tower",
                        "model.embed_vision.embedding_projection.weight",
                        &mut expected,
                    )?;
                    (356, 1_151_487_072)
                };
                ensure!(
                    expected
                        .values()
                        .filter(|e| e.2 == TensorScope::Vision)
                        .count()
                        == vision_count
                        && MappedCheckpoint::planned_bytes(&expected, LoadPolicy::VisionOnly)?
                            == vision_bytes,
                    "large vision manifest count or allocation size changed"
                );
                if spec.hidden == 5376 {
                    let mut encoder_vision = vision.clone();
                    encoder_vision.output = 2816;
                    let mut encoder = BTreeMap::new();
                    MappedCheckpoint::axial_vision_entries(
                        &encoder_vision,
                        "model.encoder.vision_tower",
                        "model.encoder.embed_vision.embedding_projection.weight",
                        &mut encoder,
                    )?;
                    ensure!(encoder.len() == 356
                        && encoder.contains_key("model.encoder.vision_tower.encoder.layers.26.self_attn.q_proj.linear.weight")
                        && encoder["model.encoder.embed_vision.embedding_projection.weight"].1 == [2816, 1152]
                        && MappedCheckpoint::planned_bytes(&encoder, LoadPolicy::VisionOnly)? == 1_145_588_832,
                        "shared axial manifest lost its binding prefix or output width");
                }
                continue;
            }
            MappedCheckpoint::axial_vision_entries(
                spec.vision.as_ref().context("vision spec")?,
                "model.vision_tower",
                "model.embed_vision.embedding_projection.weight",
                &mut expected,
            )?;
            let vision_bytes = if spec.hidden == 1536 {
                337_089_408
            } else {
                338_662_272
            };
            ensure!(
                expected
                    .values()
                    .filter(|e| e.2 == TensorScope::Vision)
                    .count()
                    == 659
                    && MappedCheckpoint::planned_bytes(&expected, LoadPolicy::VisionOnly)?
                        == vision_bytes
                    && MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextWithVision)?
                        == resident + vision_bytes,
                "vision manifest count or allocation size changed"
            );
            let mut shifted = BTreeMap::new();
            let moved = gemma4_spec(text, Some(3))?;
            MappedCheckpoint::gemma4_entries(&moved, &mut shifted)?;
            ensure!(
                shifted
                    .values()
                    .filter(|e| e.2 == TensorScope::Dormant)
                    .count()
                    == 9,
                "shared-layer count must move the Dormant set"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL to a Gemma 4 directory"]
    fn gemma4_manifest_matches_the_checkpoint_and_rejects_mutants() -> Result<()> {
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let spec = crate::model_spec::ModelSpec::read(&dir)?;
        let checkpoint = MappedCheckpoint::open(&dir)?;
        let manifest = checkpoint.manifest(&spec, LoadPolicy::TextOnly)?;
        if spec.vision.is_some() {
            let vision_manifest = checkpoint.manifest(&spec, LoadPolicy::VisionOnly)?;
            let expected_vision_bytes = match spec.hidden {
                1536 => 337_089_408,
                2560 => 338_662_272,
                3840 => 99_844_608,
                5376 => 1_151_487_072,
                hidden => bail!("unexpected vision hidden size {hidden}"),
            };
            ensure!(
                vision_manifest.resident_bytes == expected_vision_bytes
                    && vision_manifest
                        .entries
                        .values()
                        .all(|e| e.load == (e.scope == TensorScope::Vision)),
                "vision allocation plan differs from published tensors"
            );
            let mut wrong_geometry = spec.clone();
            wrong_geometry
                .vision
                .as_mut()
                .context("vision spec")?
                .intermediate += 16;
            ensure!(
                checkpoint
                    .manifest(&wrong_geometry, LoadPolicy::VisionOnly)
                    .is_err(),
                "checkpoint-derived vision shapes hid a config/weight mismatch"
            );
        }
        if spec.audio.is_some() {
            let audio_manifest = checkpoint.manifest(&spec, LoadPolicy::AudioOnly)?;
            ensure!(
                audio_manifest.resident_bytes
                    == match spec.hidden {
                        1536 => 614_368_768,
                        2560 => 617_514_496,
                        3840 => 4_915_200,
                        hidden => bail!("unexpected audio hidden size {hidden}"),
                    }
                    && audio_manifest.entries.values().filter(|e| e.load).count()
                        == if spec.hidden == 3840 { 1 } else { 752 }
                    && audio_manifest
                        .entries
                        .values()
                        .all(|e| e.load == (e.scope == TensorScope::Audio)),
                "audio manifest residency differs from published shapes"
            );
            let mut wrong_audio = spec.clone();
            match wrong_audio.audio.as_mut().context("audio spec")? {
                crate::model_spec::AudioConfig::Encoder(a) => a.conv_kernel = 3,
                crate::model_spec::AudioConfig::Raw(a) => a.frame += 1,
            }
            ensure!(
                checkpoint
                    .manifest(&wrong_audio, LoadPolicy::TextOnly)
                    .is_err(),
                "checkpoint-derived audio shapes hid a wrong convolution kernel"
            );
        }
        let (total, vision, audio, dormant, resident) = match spec.hidden {
            1536 => (2011, 659, 752, 60, 9_257_138_758),
            2560 => (2130, 659, 752, 54, 14_926_026_836),
            3840 => (677, 10, 1, 0, 23_814_700_640),
            5376 => (1188, 356, 0, 0, 61_394_690_680),
            hidden => bail!("unexpected Gemma 4 hidden size {hidden}"),
        };
        let count = |scope| {
            manifest
                .entries
                .values()
                .filter(|entry| entry.scope == scope)
                .count()
        };
        ensure!(
            manifest.entries.len() == total
                && count(TensorScope::Vision) == vision
                && count(TensorScope::Audio) == audio
                && count(TensorScope::Dormant) == dormant
                && manifest.resident_bytes == resident
                && manifest
                    .entries
                    .values()
                    .all(|e| e.load == (e.scope == TensorScope::Text)),
            "Gemma 4 manifest: entries={} vision={} audio={} dormant={} resident={}",
            manifest.entries.len(),
            count(TensorScope::Vision),
            count(TensorScope::Audio),
            count(TensorScope::Dormant),
            manifest.resident_bytes
        );
        let layer = format!("model.language_model.layers.{}.mlp.up_proj.weight", 3);
        type Mutation = fn(&mut BTreeMap<String, ExpectedTensor>, &str);
        let mutants: [(&str, Mutation); 3] = [
            ("missing", |e, name| {
                e.remove(name);
            }),
            ("shape", |e, name| {
                e.get_mut(name).expect("expected tensor").1[0] += 1;
            }),
            ("extra", |e, _| {
                put(
                    e,
                    "model.language_model.extra.weight".into(),
                    Dtype::BF16,
                    vec![1],
                    TensorScope::Text,
                );
            }),
        ];
        for (label, mutate) in mutants {
            let mut expected = checkpoint.gemma4_expected(&spec)?;
            mutate(&mut expected, &layer);
            let planned = MappedCheckpoint::planned_bytes(&expected, LoadPolicy::TextOnly)?;
            ensure!(
                checkpoint
                    .assemble_manifest("Gemma4", expected, LoadPolicy::TextOnly, planned, None)
                    .is_err(),
                "{label} mutant was accepted"
            );
        }
        for (index, layout) in spec.layers.iter().enumerate() {
            let geometry = layout.geometry.context("Gemma 4 layer geometry missing")?;
            let plan = crate::linear_plan::LayerDequantPlan::from_manifest(
                &manifest,
                u32::try_from(index)?,
                spec.linear_format,
            )?;
            plan.validate_contiguous_pair(&manifest, "mlp.gate_proj.weight", "mlp.up_proj.weight")?;
            ensure!(
                plan.validate_contiguous_pair(
                    &manifest,
                    "self_attn.k_proj.weight",
                    "self_attn.q_proj.weight",
                )
                .is_err(),
                "disjoint k_proj and q_proj accepted as contiguous on layer {index}"
            );
            let shared_projection = geometry.head_dim == 512 && spec.per_layer_input.is_none();
            if !geometry.kv_reader && !shared_projection {
                plan.validate_contiguous_pair(
                    &manifest,
                    "self_attn.q_proj.weight",
                    "self_attn.v_proj.weight",
                )?;
            } else {
                ensure!(
                    plan.validate_contiguous_pair(
                        &manifest,
                        "self_attn.q_proj.weight",
                        "self_attn.v_proj.weight",
                    )
                    .is_err(),
                    "dormant or k_eq_v v_proj accepted on layer {index}"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct StoredTensor {
    shape: Vec<usize>,
    range: std::ops::Range<usize>,
}

struct CheckpointBytes<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> CheckpointBytes<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .context("checkpoint offset overflow")?;
        let bytes = self
            .data
            .get(self.position..end)
            .context("truncated checkpoint record")?;
        self.position = end;
        Ok(bytes)
    }
    fn number(&mut self, width: usize) -> Result<usize> {
        ensure!(width <= 8, "checkpoint integer width");
        let mut bytes = [0u8; 8];
        bytes[..width].copy_from_slice(self.take(width)?);
        Ok(usize::try_from(u64::from_le_bytes(bytes))?)
    }
    fn signature(&mut self, expected: usize) -> Result<()> {
        ensure!(self.number(4)? == expected, "invalid ZIP signature");
        Ok(())
    }
}

pub(crate) fn tar_member_bytes<'a>(data: &'a [u8], name: &str) -> Result<&'a [u8]> {
    Ok(&data[tar_member(data, name)?])
}

pub(crate) fn tar_member(data: &[u8], name: &str) -> Result<std::ops::Range<usize>> {
    let mut position = 0usize;
    while position + 512 <= data.len() {
        let header = &data[position..position + 512];
        if header.iter().all(|&byte| byte == 0) {
            break;
        }
        ensure!(
            &header[257..262] == b"ustar",
            "tar member header is not ustar"
        );
        let field = |range: std::ops::Range<usize>| -> Result<&str> {
            let bytes = &header[range];
            let end = bytes
                .iter()
                .position(|&byte| byte == 0)
                .unwrap_or(bytes.len());
            Ok(std::str::from_utf8(&bytes[..end])?)
        };
        let member = field(0..100)?;
        let size_text = field(124..136)?;
        ensure!(
            !size_text.is_empty() && size_text.bytes().all(|b| b.is_ascii_digit() || b == b' '),
            "tar member size is not octal: {member}"
        );
        let size = usize::from_str_radix(size_text.trim_matches(' '), 8)
            .with_context(|| format!("tar member size parse: {member}"))?;
        let typeflag = header[156];
        ensure!(
            typeflag == b'0' || typeflag == 0,
            "tar member is not a regular file: {member}"
        );
        let data_start = position + 512;
        let data_stop = data_start
            .checked_add(size)
            .context("tar member size overflow")?;
        ensure!(data_stop <= data.len(), "tar member truncated: {member}");
        if member == name {
            return Ok(data_start..data_stop);
        }
        position = data_start + size.div_ceil(512) * 512;
    }
    bail!("tar member missing: {name}")
}

fn stored_zip_entries(data: &[u8]) -> Result<BTreeMap<String, std::ops::Range<usize>>> {
    ensure!(data.len() >= 22, "truncated ZIP end record");
    let end = (data.len().saturating_sub(65557)..=data.len() - 22)
        .rev()
        .find(|&i| data[i..i + 4] == [0x50, 0x4b, 5, 6])
        .context("ZIP end record missing")?;
    let mut tail = CheckpointBytes {
        data,
        position: end + 4,
    };
    ensure!(
        tail.number(2)? == 0 && tail.number(2)? == 0,
        "multi-disk ZIP unsupported"
    );
    let entries = tail.number(2)?;
    ensure!(
        entries > 0 && entries < 4096 && tail.number(2)? == entries,
        "ZIP entry count"
    );
    let directory_bytes = tail.number(4)?;
    let directory_offset = tail.number(4)?;
    let comment = tail.number(2)?;
    tail.take(comment)?;
    ensure!(tail.position == data.len(), "trailing ZIP bytes");
    let directory_end = directory_offset
        .checked_add(directory_bytes)
        .context("ZIP directory overflow")?;
    ensure!(directory_end <= end, "ZIP directory exceeds end record");
    if directory_end != end {
        let mut zip64 = CheckpointBytes {
            data,
            position: directory_end,
        };
        zip64.signature(0x06064b50)?;
        ensure!(zip64.number(8)? == 44, "ZIP64 record length");
        zip64.take(4)?;
        ensure!(
            zip64.number(4)? == 0 && zip64.number(4)? == 0,
            "ZIP64 disk count"
        );
        ensure!(
            zip64.number(8)? == entries && zip64.number(8)? == entries,
            "ZIP64 entry count mismatch"
        );
        ensure!(
            zip64.number(8)? == directory_bytes && zip64.number(8)? == directory_offset,
            "ZIP64 directory mismatch"
        );
        zip64.signature(0x07064b50)?;
        ensure!(
            zip64.number(4)? == 0 && zip64.number(8)? == directory_end && zip64.number(4)? == 1,
            "ZIP64 locator mismatch"
        );
        ensure!(zip64.position == end, "ZIP64 trailer length");
    }
    let mut directory = CheckpointBytes {
        data: &data[..directory_end],
        position: directory_offset,
    };
    let mut result = BTreeMap::new();
    let mut occupied = Vec::new();
    for _ in 0..entries {
        directory.signature(0x02014b50)?;
        directory.take(4)?;
        let flags = directory.number(2)?;
        ensure!(
            flags & !0x808 == 0 && directory.number(2)? == 0,
            "compressed or encrypted ZIP member"
        );
        directory.take(4)?;
        let crc = directory.number(4)?;
        let bytes = directory.number(4)?;
        ensure!(directory.number(4)? == bytes, "stored ZIP size mismatch");
        let name_len = directory.number(2)?;
        let extra_len = directory.number(2)?;
        let comment_len = directory.number(2)?;
        ensure!(directory.number(2)? == 0, "ZIP member on another disk");
        directory.take(6)?;
        let local_offset = directory.number(4)?;
        let name = std::str::from_utf8(directory.take(name_len)?)?.to_owned();
        ensure!(
            !name.is_empty()
                && name
                    .split('/')
                    .all(|p| !p.is_empty() && p != "." && p != ".." && !p.contains('\\')),
            "unsafe ZIP member name"
        );
        ensure!(extra_len == 0, "unsupported central ZIP extra fields");
        directory.take(comment_len)?;
        let mut local = CheckpointBytes {
            data: &data[..directory_offset],
            position: local_offset,
        };
        local.signature(0x04034b50)?;
        local.take(2)?;
        ensure!(
            local.number(2)? == flags && local.number(2)? == 0,
            "ZIP local flags/method mismatch"
        );
        local.take(4)?;
        let local_crc = local.number(4)?;
        let compressed = local.number(4)?;
        let uncompressed = local.number(4)?;
        let local_name_len = local.number(2)?;
        let local_extra_len = local.number(2)?;
        ensure!(
            local.take(local_name_len)? == name.as_bytes(),
            "ZIP local name mismatch"
        );
        let extra = local.take(local_extra_len)?;
        if !extra.is_empty() {
            let mut padding = CheckpointBytes {
                data: extra,
                position: 0,
            };
            ensure!(
                padding.number(2)? == 0x4246,
                "unsupported local ZIP extra field"
            );
            let count = padding.number(2)?;
            ensure!(
                padding.take(count)?.iter().all(|&b| b == b'Z') && padding.position == extra.len(),
                "invalid ZIP alignment padding"
            );
        }
        let start = local.position;
        let payload = local.take(bytes)?;
        let stop = local.position;
        let mut computed = !0u32;
        for &byte in payload {
            computed ^= u32::from(byte);
            for _ in 0..8 {
                computed = (computed >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(computed & 1));
            }
        }
        ensure!(
            (!computed) as usize == crc,
            "ZIP payload checksum mismatch: {name}"
        );
        if flags & 8 != 0 {
            ensure!(
                local_crc == 0 && compressed == 0 && uncompressed == 0,
                "ZIP descriptor local fields"
            );
            local.signature(0x08074b50)?;
            ensure!(
                local.number(4)? == crc && local.number(4)? == bytes && local.number(4)? == bytes,
                "ZIP data descriptor mismatch"
            );
        } else {
            ensure!(
                local_crc == crc && compressed == bytes && uncompressed == bytes,
                "ZIP local size/checksum mismatch"
            );
        }
        occupied.push(local_offset..local.position);
        ensure!(
            result.insert(name, start..stop).is_none(),
            "duplicate ZIP entry"
        );
    }
    ensure!(
        directory.position == directory_end,
        "ZIP directory size mismatch"
    );
    occupied.sort_by_key(|r| r.start);
    ensure!(
        occupied.first().is_some_and(|r| r.start == 0)
            && occupied.windows(2).all(|w| w[0].end == w[1].start)
            && occupied.last().is_some_and(|r| r.end == directory_offset),
        "ZIP member overlap or unindexed bytes"
    );
    Ok(result)
}

#[derive(Clone)]
enum CheckpointValue {
    Mark,
    Integer(usize),
    String(String),
    Global(String),
    Tuple(Vec<std::rc::Rc<CheckpointValue>>),
    Dictionary(BTreeMap<String, std::rc::Rc<CheckpointValue>>),
    False,
    Storage(String, usize),
    Tensor(StoredTensor),
}

impl CheckpointValue {
    fn text(&self) -> Result<&str> {
        if let Self::String(s) = self {
            Ok(s)
        } else {
            bail!("pickle expected string")
        }
    }
    fn integer(&self) -> Result<usize> {
        if let Self::Integer(i) = self {
            Ok(*i)
        } else {
            bail!("pickle expected integer")
        }
    }
    fn tuple(&self) -> Result<&[std::rc::Rc<Self>]> {
        if let Self::Tuple(v) = self {
            Ok(v)
        } else {
            bail!("pickle expected tuple")
        }
    }
    fn dictionary(&self) -> Result<&BTreeMap<String, std::rc::Rc<Self>>> {
        if let Self::Dictionary(v) = self {
            Ok(v)
        } else {
            bail!("pickle expected dictionary")
        }
    }
}

struct CheckpointPickle<'a> {
    bytes: CheckpointBytes<'a>,
    stack: Vec<std::rc::Rc<CheckpointValue>>,
    memo: BTreeMap<usize, std::rc::Rc<CheckpointValue>>,
    entries: &'a BTreeMap<String, std::ops::Range<usize>>,
    prefix: &'a str,
}

impl CheckpointPickle<'_> {
    fn pop(&mut self) -> Result<std::rc::Rc<CheckpointValue>> {
        self.stack.pop().context("pickle stack underflow")
    }
    fn marked(&mut self) -> Result<Vec<std::rc::Rc<CheckpointValue>>> {
        let index = self
            .stack
            .iter()
            .rposition(|v| matches!(**v, CheckpointValue::Mark))
            .context("pickle MARK missing")?;
        let values = self.stack.split_off(index + 1);
        self.stack.pop();
        Ok(values)
    }
    fn reduce(&mut self) -> Result<CheckpointValue> {
        use CheckpointValue as V;
        let args = self.pop()?;
        let args = args.tuple()?;
        let callable = self.pop()?;
        let V::Global(tag) = &*callable else {
            bail!("pickle REDUCE callable")
        };
        if tag == "collections.OrderedDict" {
            ensure!(args.is_empty(), "OrderedDict constructor arguments");
            return Ok(V::Dictionary(BTreeMap::new()));
        }
        ensure!(
            tag == "torch._utils._rebuild_tensor_v2" && args.len() == 6,
            "unsupported tensor constructor"
        );
        let V::Storage(storage, count) = &*args[0] else {
            bail!("tensor storage missing")
        };
        ensure!(
            matches!(*args[4], V::False) && args[5].dictionary()?.is_empty(),
            "tensor grad/hooks unsupported"
        );
        let offset = args[1].integer()?;
        let shape = args[2]
            .tuple()?
            .iter()
            .map(|v| v.integer())
            .collect::<Result<Vec<_>>>()?;
        let strides = args[3]
            .tuple()?
            .iter()
            .map(|v| v.integer())
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !shape.is_empty() && shape.len() <= 8 && shape.len() == strides.len(),
            "tensor rank mismatch"
        );
        let mut elements = 1usize;
        for (&dim, &stride) in shape.iter().zip(&strides).rev() {
            ensure!(dim > 0 && stride == elements, "noncontiguous stored tensor");
            elements = elements
                .checked_mul(dim)
                .context("tensor element count overflow")?;
        }
        ensure!(
            offset == 0 && elements == *count,
            "tensor storage offset/size mismatch"
        );
        let range = self
            .entries
            .get(&format!("{}data/{storage}", self.prefix))
            .context("tensor ZIP storage missing")?;
        ensure!(
            range.len()
                == elements
                    .checked_mul(4)
                    .context("tensor byte count overflow")?,
            "tensor storage byte count mismatch"
        );
        Ok(V::Tensor(StoredTensor {
            shape,
            range: range.clone(),
        }))
    }
    fn persistent(&mut self) -> Result<CheckpointValue> {
        use CheckpointValue as V;
        let value = self.pop()?;
        let fields = value.tuple()?;
        ensure!(
            fields.len() == 5 && fields[0].text()? == "storage",
            "unsupported persistent ID"
        );
        ensure!(
            matches!(&*fields[1], V::Global(s) if s == "torch.FloatStorage"),
            "unsupported storage type"
        );
        let name = fields[2].text()?;
        ensure!(
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit()),
            "invalid storage key"
        );
        ensure!(
            fields[3].text()? == "cpu" || fields[3].text()? == "cuda:0",
            "unsupported stored location tag"
        );
        Ok(V::Storage(name.to_owned(), fields[4].integer()?))
    }
    fn set_items(&mut self, multiple: bool) -> Result<()> {
        let values = if multiple {
            self.marked()?
        } else {
            let value = self.pop()?;
            vec![self.pop()?, value]
        };
        ensure!(values.len().is_multiple_of(2), "pickle key/value count");
        let mut map = self.pop()?.dictionary()?.clone();
        for pair in values.as_chunks::<2>().0.iter() {
            ensure!(
                map.insert(pair[0].text()?.to_owned(), pair[1].clone())
                    .is_none(),
                "duplicate pickle dictionary key"
            );
        }
        self.stack
            .push(std::rc::Rc::new(CheckpointValue::Dictionary(map)));
        Ok(())
    }
    fn parse(mut self) -> Result<BTreeMap<String, StoredTensor>> {
        use CheckpointValue as V;
        use std::rc::Rc;
        ensure!(
            self.bytes.take(2)? == [0x80, 2],
            "checkpoint requires pickle protocol 2"
        );
        loop {
            ensure!(
                self.stack.len() < 8192 && self.memo.len() < 8192,
                "pickle metadata capacity"
            );
            let opcode = u8::try_from(self.bytes.number(1)?)?;
            let value = match opcode {
                b'(' => Some(V::Mark),
                b')' => Some(V::Tuple(Vec::new())),
                b'}' => Some(V::Dictionary(BTreeMap::new())),
                b'K' => Some(V::Integer(self.bytes.number(1)?)),
                b'M' => Some(V::Integer(self.bytes.number(2)?)),
                b'J' => {
                    let i = self.bytes.number(4)?;
                    ensure!(i <= i32::MAX as usize, "negative pickle integer");
                    Some(V::Integer(i))
                }
                0x89 => Some(V::False),
                b'X' => {
                    let len = self.bytes.number(4)?;
                    Some(V::String(
                        std::str::from_utf8(self.bytes.take(len)?)?.to_owned(),
                    ))
                }
                b'c' => {
                    let mut names = Vec::new();
                    for _ in 0..2 {
                        let start = self.bytes.position;
                        while self.bytes.number(1)? != usize::from(b'\n') {}
                        names.push(std::str::from_utf8(
                            &self.bytes.data[start..self.bytes.position - 1],
                        )?);
                    }
                    let name = names.join(".");
                    ensure!(
                        [
                            "collections.OrderedDict",
                            "torch.FloatStorage",
                            "torch._utils._rebuild_tensor_v2"
                        ]
                        .contains(&name.as_str()),
                        "unsupported pickle GLOBAL {name}"
                    );
                    Some(V::Global(name))
                }
                b't' => Some(V::Tuple(self.marked()?)),
                0x85 => Some(V::Tuple(vec![self.pop()?])),
                0x86 => {
                    let second = self.pop()?;
                    let first = self.pop()?;
                    Some(V::Tuple(vec![first, second]))
                }
                0x87 => {
                    let third = self.pop()?;
                    let second = self.pop()?;
                    let first = self.pop()?;
                    Some(V::Tuple(vec![first, second, third]))
                }
                b'R' => Some(self.reduce()?),
                b'Q' => Some(self.persistent()?),
                b'q' | b'r' => {
                    let index = self.bytes.number(if opcode == b'q' { 1 } else { 4 })?;
                    ensure!(
                        index < 8192
                            && self
                                .memo
                                .insert(
                                    index,
                                    self.stack
                                        .last()
                                        .context("pickle memo stack empty")?
                                        .clone()
                                )
                                .is_none(),
                        "pickle memo index overwrite/overflow"
                    );
                    None
                }
                b'h' | b'j' => {
                    let index = self.bytes.number(if opcode == b'h' { 1 } else { 4 })?;
                    let value = self
                        .memo
                        .get(&index)
                        .context("pickle memo reference missing")?;
                    ensure!(
                        !matches!(
                            **value,
                            V::Dictionary(_) | V::Tensor(_) | V::Storage(..) | V::Mark
                        ),
                        "mutable pickle alias unsupported"
                    );
                    self.stack.push(value.clone());
                    None
                }
                b'u' | b's' => {
                    self.set_items(opcode == b'u')?;
                    None
                }
                b'b' => {
                    let state = self.pop()?;
                    let state = state.dictionary()?;
                    ensure!(state.len() == 1, "unsupported pickle BUILD state");
                    let metadata = state
                        .get("_metadata")
                        .context("state metadata missing")?
                        .dictionary()?;
                    for value in metadata.values() {
                        let value = value.dictionary()?;
                        ensure!(
                            value.len() == 1
                                && value
                                    .get("version")
                                    .context("module version missing")?
                                    .integer()?
                                    == 1,
                            "unsupported module metadata"
                        );
                    }
                    self.stack
                        .last()
                        .context("BUILD target missing")?
                        .dictionary()?;
                    None
                }
                b'.' => {
                    ensure!(
                        self.stack.len() == 1 && self.bytes.position == self.bytes.data.len(),
                        "pickle trailing stack/data"
                    );
                    let result = self.pop()?;
                    let mut tensors = BTreeMap::new();
                    let mut storage_ranges = BTreeSet::new();
                    for (name, value) in result.dictionary()? {
                        let V::Tensor(tensor) = &**value else {
                            bail!("non-tensor state entry {name}")
                        };
                        ensure!(
                            storage_ranges.insert((tensor.range.start, tensor.range.end)),
                            "aliased tensor storage"
                        );
                        tensors.insert(name.clone(), tensor.clone());
                    }
                    ensure!(!tensors.is_empty(), "empty state dictionary");
                    return Ok(tensors);
                }
                _ => bail!(
                    "unsupported pickle opcode 0x{opcode:02x} at {}",
                    self.bytes.position - 1
                ),
            };
            if let Some(value) = value {
                self.stack.push(Rc::new(value));
            }
        }
    }
}

fn stored_zip_tensors(data: &[u8]) -> Result<BTreeMap<String, StoredTensor>> {
    let entries = stored_zip_entries(data)?;
    let pickle_names: Vec<_> = entries
        .keys()
        .filter(|s| s.ends_with("/data.pkl"))
        .collect();
    ensure!(pickle_names.len() == 1, "ZIP requires one data.pkl");
    let name = pickle_names[0];
    let prefix = name
        .strip_suffix("data.pkl")
        .context("pickle path suffix")?;
    for (suffix, expected) in [
        ("byteorder", b"little".as_slice()),
        ("version", b"3\n".as_slice()),
        (".format_version", b"1".as_slice()),
        (".storage_alignment", b"64".as_slice()),
    ] {
        let range = match entries.get(&format!("{prefix}{suffix}")) {
            Some(range) => range.clone(),
            None => {
                ensure!(
                    suffix == ".format_version" || suffix == ".storage_alignment",
                    "ZIP metadata missing"
                );
                continue;
            }
        };
        ensure!(&data[range] == expected, "unsupported ZIP {suffix}");
    }
    let range = entries[name].clone();
    ensure!(
        range.len() <= 16 * 1024 * 1024,
        "pickle metadata exceeds 16 MiB"
    );
    let tensors = CheckpointPickle {
        bytes: CheckpointBytes {
            data: &data[range],
            position: 0,
        },
        stack: Vec::new(),
        memo: BTreeMap::new(),
        entries: &entries,
        prefix,
    }
    .parse()?;
    let allowed: BTreeSet<_> = [
        "data.pkl",
        "byteorder",
        "version",
        ".format_version",
        ".storage_alignment",
        ".data/serialization_id",
    ]
    .map(|s| format!("{prefix}{s}"))
    .into_iter()
    .collect();
    ensure!(
        entries
            .iter()
            .all(|(name, range)| allowed.contains(name)
                || tensors.values().any(|t| t.range == *range)),
        "unconsumed ZIP entries"
    );
    let id = entries
        .get(&format!("{prefix}.data/serialization_id"))
        .with_context(|| format!("stored-ZIP member {prefix}.data/serialization_id missing"))?;
    let id = &data[id.clone()];
    ensure!(
        !id.is_empty() && id.iter().all(u8::is_ascii_digit),
        "invalid serialization ID"
    );
    Ok(tensors)
}

#[cfg(test)]
pub(crate) fn tar_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();
    let header = |name: &str, size: usize, typeflag: u8, out: &mut Vec<u8>| {
        out.extend(std::iter::repeat_n(0u8, 512));
        let start = out.len() - 512;
        out[start..start + name.len()].copy_from_slice(name.as_bytes());
        let octal = format!("{size:011o}\0");
        out[start + 124..start + 136].copy_from_slice(octal.as_bytes());
        out[start + 156] = typeflag;
        out[start + 257..262].copy_from_slice(b"ustar");
    };
    for (name, value) in members {
        header(name, value.len(), b'0', &mut data);
        data.extend_from_slice(value);
        data.extend(std::iter::repeat_n(
            0u8,
            value.len().div_ceil(512) * 512 - value.len(),
        ));
    }
    data.extend(std::iter::repeat_n(0u8, 1024));
    data
}

#[cfg(test)]
mod stored_zip_tests {
    use super::*;

    fn archive(pickle: &[u8], method: u16) -> Vec<u8> {
        let payload = [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat();
        zip_members(
            &[
                ("model/data.pkl", pickle),
                ("model/data/0", payload.as_slice()),
                ("model/byteorder", b"little".as_slice()),
                ("model/version", b"3\n".as_slice()),
                ("model/.data/serialization_id", b"12345678".as_slice()),
            ],
            method,
        )
    }

    fn zip_members(members: &[(&str, &[u8])], method: u16) -> Vec<u8> {
        let mut data = Vec::new();
        let mut directory = Vec::new();
        for &(name, value) in members {
            let offset = data.len() as u32;
            let mut crc = !0u32;
            for &b in value {
                crc ^= u32::from(b);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
                }
            }
            let crc = !crc;
            data.extend_from_slice(&0x04034b50u32.to_le_bytes());
            data.extend_from_slice(&[0; 4]);
            data.extend_from_slice(&method.to_le_bytes());
            data.extend_from_slice(&[0; 4]);
            data.extend_from_slice(&crc.to_le_bytes());
            for _ in 0..2 {
                data.extend_from_slice(&(value.len() as u32).to_le_bytes());
            }
            data.extend_from_slice(&(name.len() as u16).to_le_bytes());
            data.extend_from_slice(&[0; 2]);
            data.extend_from_slice(name.as_bytes());
            data.extend_from_slice(value);
            directory.extend_from_slice(&0x02014b50u32.to_le_bytes());
            directory.extend_from_slice(&[0; 6]);
            directory.extend_from_slice(&method.to_le_bytes());
            directory.extend_from_slice(&[0; 4]);
            directory.extend_from_slice(&crc.to_le_bytes());
            for _ in 0..2 {
                directory.extend_from_slice(&(value.len() as u32).to_le_bytes());
            }
            directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
            directory.extend_from_slice(&[0; 12]);
            directory.extend_from_slice(&offset.to_le_bytes());
            directory.extend_from_slice(name.as_bytes());
        }
        let start = data.len() as u32;
        data.extend_from_slice(&directory);
        data.extend_from_slice(&0x06054b50u32.to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        for _ in 0..2 {
            data.extend_from_slice(&(members.len() as u16).to_le_bytes());
        }
        data.extend_from_slice(&(directory.len() as u32).to_le_bytes());
        data.extend_from_slice(&start.to_le_bytes());
        data.extend_from_slice(&[0; 2]);
        data
    }

    fn pickle(offset: u8) -> Vec<u8> {
        let mut data = b"\x80\x02ccollections\nOrderedDict\n)R(".to_vec();
        fn string(data: &mut Vec<u8>, value: &str) {
            data.push(b'X');
            data.extend_from_slice(&(value.len() as u32).to_le_bytes());
            data.extend_from_slice(value.as_bytes());
        }
        string(&mut data, "weight");
        data.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
        string(&mut data, "storage");
        data.extend_from_slice(b"ctorch\nFloatStorage\n");
        string(&mut data, "0");
        string(&mut data, "cuda:0");
        data.extend_from_slice(&[b'K', 2, b't', b'Q', b'K', offset]);
        data.extend_from_slice(b"K\x02\x85K\x01\x85\x89ccollections\nOrderedDict\n)RtRu.");
        data
    }

    #[test]
    fn stored_zip_without_serialization_id_names_the_member() {
        let payload = [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat();
        let data = zip_members(
            &[
                ("model/data.pkl", pickle(0).as_slice()),
                ("model/data/0", payload.as_slice()),
                ("model/byteorder", b"little".as_slice()),
                ("model/version", b"3\n".as_slice()),
            ],
            0,
        );
        let err = match stored_zip_tensors(&data) {
            Ok(_) => panic!("a stored ZIP without a serialization ID must error"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains(".data/serialization_id missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[ignore = "CPU published NAF checkpoint; set FF_MODEL to its directory"]
    fn stored_zip_tensor_layout_rejects_invalid_metadata() -> Result<()> {
        let data = archive(&pickle(0), 0);
        let tensors = stored_zip_tensors(&data)?;
        ensure!(
            tensors.len() == 1 && tensors["weight"].shape == [2],
            "valid stored tensor layout"
        );
        let tensor = &tensors["weight"];
        ensure!(
            data[tensor.range.clone()] == [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat(),
            "valid stored tensor bytes"
        );
        ensure!(
            stored_zip_tensors(&archive(&pickle(1), 0)).is_err(),
            "wrong storage offset accepted"
        );
        let mut unknown = pickle(0);
        unknown[2] = 0xff;
        ensure!(
            stored_zip_tensors(&archive(&unknown, 0)).is_err(),
            "unknown opcode accepted"
        );
        ensure!(
            stored_zip_tensors(&archive(&pickle(0), 8)).is_err(),
            "compressed member accepted"
        );
        let root =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let checkpoint = MappedCheckpoint::open_component(&root, "naf_release.pth")?;
        ensure!(checkpoint.index.weight_map.len() == 37, "NAF key count");
        let mut elements = 0;
        for name in checkpoint.index.weight_map.keys() {
            let tensor = checkpoint.tensor(name)?;
            let values = checkpoint.f32_tensor(name, tensor.shape())?;
            ensure!(values.iter().all(|v| v.is_finite()), "nonfinite NAF {name}");
            elements += values.len();
        }
        ensure!(elements == 662544, "NAF storage element count");
        println!(
            "PASS stored_zip_tensor_layout valid=1 wrong_offset=rejected unknown_opcode=rejected compressed_member=rejected"
        );
        Ok(())
    }

    fn temp_nemo(bytes: &[u8]) -> Result<std::path::PathBuf> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ff-nemo-{}-{}.nemo",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    #[test]
    fn nemo_checkpoint_reads_the_tar_stored_zip_member() -> Result<()> {
        let nemo = temp_nemo(&tar_archive(&[(
            "model_weights.ckpt",
            &archive(&pickle(0), 0),
        )]))?;
        let checkpoint = MappedCheckpoint::open_nemo(&nemo)?;
        std::fs::remove_file(&nemo)?;
        ensure!(
            checkpoint.index.weight_map.len() == 1
                && checkpoint.index.weight_map.contains_key("weight"),
            "nemo tensor names"
        );
        let tensor = checkpoint.tensor("weight")?;
        ensure!(
            tensor.shape() == [2] && tensor.data().len() == 8,
            "nemo tensor view"
        );
        let values = checkpoint.f32_tensor("weight", tensor.shape())?;
        ensure!(values == [1.25, -2.5], "nemo tensor bytes");
        println!("PASS nemo_checkpoint member=1 tensors=1 bytes_exact=true");
        Ok(())
    }

    #[test]
    fn nemo_checkpoint_rejects_tar_and_payload_defects() -> Result<()> {
        let valid = archive(&pickle(0), 0);
        let missing = tar_archive(&[("model_config.yaml", b"key: value\n")]);
        let path = temp_nemo(&missing)?;
        let error = MappedCheckpoint::open_nemo(&path)
            .err()
            .context("missing member accepted")?;
        ensure!(
            error
                .to_string()
                .contains("tar member missing: model_weights.ckpt"),
            "missing member error"
        );
        std::fs::remove_file(&path)?;

        let mut nonregular = tar_archive(&[("model_weights.ckpt", &valid)]);
        let typeflag = nonregular.len() - 1024 - valid.len().div_ceil(512) * 512 - 512 + 156;
        nonregular[typeflag] = b'2';
        let path = temp_nemo(&nonregular)?;
        let error = MappedCheckpoint::open_nemo(&path)
            .err()
            .context("non-regular member accepted")?;
        ensure!(
            error.to_string().contains("not a regular file"),
            "non-regular member error"
        );
        std::fs::remove_file(&path)?;

        let mut truncated = tar_archive(&[("model_weights.ckpt", &valid)]);
        let size_field = truncated.len() - 1024 - valid.len().div_ceil(512) * 512 - 512 + 124;
        truncated[size_field] = b'7';
        let path = temp_nemo(&truncated)?;
        let error = MappedCheckpoint::open_nemo(&path)
            .err()
            .context("truncated member accepted")?;
        ensure!(
            error.to_string().contains("truncated"),
            "truncated member error"
        );
        std::fs::remove_file(&path)?;

        let mut corrupt = tar_archive(&[("model_weights.ckpt", &valid)]);
        let pickle_payload = 512 + 40 + valid.len() / 2;
        corrupt[pickle_payload] ^= 0xff;
        let path = temp_nemo(&corrupt)?;
        let error = MappedCheckpoint::open_nemo(&path)
            .err()
            .context("payload checksum accepted")?;
        ensure!(
            error.to_string().contains("checksum mismatch"),
            "payload checksum error"
        );
        std::fs::remove_file(&path)?;
        println!(
            "PASS nemo_checkpoint_rejects missing_member=true non_regular=true truncated=true payload_checksum=true"
        );
        Ok(())
    }

    #[test]
    #[ignore = "published parakeet_realtime_eou_120m-v1 archive; set FF_EOU_NEMO and FF_EOU_MANIFEST"]
    fn nemo_checkpoint_matches_the_published_manifest() -> Result<()> {
        let nemo = std::env::var("FF_EOU_NEMO").context("FF_EOU_NEMO missing")?;
        let manifest_path = std::env::var("FF_EOU_MANIFEST").context("FF_EOU_MANIFEST missing")?;
        let checkpoint = MappedCheckpoint::open_nemo(std::path::Path::new(&nemo))?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let tensors = manifest["tensors"]
            .as_object()
            .context("manifest tensors missing")?;
        ensure!(
            checkpoint.index.weight_map.len() == tensors.len(),
            "nemo tensor count {} against manifest {}",
            checkpoint.index.weight_map.len(),
            tensors.len()
        );
        let mut params = 0usize;
        for (name, entry) in tensors {
            let tensor = checkpoint.tensor(name)?;
            let shape = entry["shape"]
                .as_array()
                .context("manifest shape")?
                .iter()
                .map(|v| Ok(v.as_u64().context("manifest dim")? as usize))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                tensor.shape() == shape,
                "nemo tensor {name} shape {:?} against manifest {shape:?}",
                tensor.shape()
            );
            let bytes = entry["bytes"].as_u64().context("manifest bytes")? as usize;
            ensure!(tensor.data().len() == bytes, "nemo tensor {name} bytes");
            params += shape.iter().product::<usize>();
        }
        let expected = manifest["state_params"]
            .as_u64()
            .context("manifest params")? as usize;
        ensure!(
            params == expected,
            "nemo params {params} against {expected}"
        );
        let members = manifest["tar_members"]
            .as_object()
            .context("manifest members")?;
        ensure!(members.len() == 5, "manifest member count");
        let map = unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&nemo)?)? };
        for (name, size) in members {
            let range = tar_member(&map, name)?;
            ensure!(
                range.len() == size.as_u64().context("member size")? as usize,
                "member {name} size"
            );
        }
        println!(
            "PASS nemo_published_manifest tensors={} params={} members=5",
            tensors.len(),
            params
        );
        Ok(())
    }
}
#[cfg(test)]
mod gguf_tests {
    use super::*;
    use crate::gguf::GgmlType;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    struct SyntheticTensor {
        name: String,
        ggml_type: GgmlType,
        dims: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn put_u32(buf: &mut Vec<u8>, value: u32) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn put_str(buf: &mut Vec<u8>, value: &str) {
        put_u64(buf, value.len() as u64);
        buf.extend_from_slice(value.as_bytes());
    }

    fn ggml_type_id(ggml_type: GgmlType) -> u32 {
        match ggml_type {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Bf16 => 30,
            other => panic!("unsupported synthetic type {other:?}"),
        }
    }

    fn tensor_bytes(ggml_type: GgmlType, dims: &[u64]) -> Vec<u8> {
        let elements: u64 = dims.iter().product();
        let len = match ggml_type {
            GgmlType::F32 => elements * 4,
            GgmlType::F16 | GgmlType::Bf16 => elements * 2,
            GgmlType::Q4_0 => elements / 32 * 18,
            other => panic!("unsupported synthetic type {other:?}"),
        };
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn write_gguf(path: &Path, tensors: &[SyntheticTensor]) -> Result<()> {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        put_u32(&mut head, 3);
        put_u64(&mut head, tensors.len() as u64);
        put_u64(&mut head, 2);
        put_str(&mut head, "general.architecture");
        put_u32(&mut head, 8);
        put_str(&mut head, "minimax_h3");
        put_str(&mut head, "general.alignment");
        put_u32(&mut head, 4);
        put_u32(&mut head, 32);
        let mut infos = Vec::new();
        let mut offset = 0u64;
        for tensor in tensors {
            put_str(&mut infos, &tensor.name);
            put_u32(&mut infos, tensor.dims.len() as u32);
            for &dim in &tensor.dims {
                put_u64(&mut infos, dim);
            }
            put_u32(&mut infos, ggml_type_id(tensor.ggml_type));
            put_u64(&mut infos, offset);
            offset += tensor.bytes.len() as u64;
            offset = offset.div_ceil(32) * 32;
        }
        head.extend_from_slice(&infos);
        head.resize(head.len().div_ceil(32) * 32, 0);
        let mut file = head;
        for tensor in tensors {
            file.extend_from_slice(&tensor.bytes);
            let padded = tensor.bytes.len().div_ceil(32) * 32;
            file.resize(file.len() + padded - tensor.bytes.len(), 0);
        }
        std::fs::write(path, file)?;
        Ok(())
    }

    struct TempGguf(PathBuf);

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn temp_gguf(tensors: &[SyntheticTensor]) -> Result<TempGguf> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "flyingfish-gguf-manifest-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        write_gguf(&path, tensors)?;
        Ok(TempGguf(path))
    }

    fn spec() -> crate::minimax_h3::H3Spec {
        crate::minimax_h3::H3Spec {
            hidden: 32,
            heads: 1,
            head_dim: 32,
            layers: 1,
            refiner_layers: 1,
            ffn: 64,
            in_channels: 4,
            audio_in_channels: 4,
            patch: [1, 2, 2],
            text_dim: 16,
            freq_dim: 8,
            time_hidden: 16,
            time_dim: 8,
            rope_freq_dim: 4,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    fn synthetic_tensors(spec: &crate::minimax_h3::H3Spec) -> Vec<SyntheticTensor> {
        let hidden = spec.hidden as u64;
        let width = (spec.heads * spec.head_dim) as u64;
        let video = spec.video_row_width() as u64;
        let audio = spec.audio_in_channels as u64;
        let text = spec.text_dim as u64;
        let freq = spec.freq_dim as u64;
        let time_hidden = spec.time_hidden as u64;
        let time_dim = spec.time_dim as u64;
        let ffn = spec.ffn as u64;
        let modulation = (6 * spec.hidden * crate::minimax_h3::MODALITIES as usize) as u64;
        let mut out = Vec::new();
        let mut push = |name: String, ggml_type: GgmlType, dims: Vec<u64>| {
            let bytes = tensor_bytes(ggml_type, &dims);
            out.push(SyntheticTensor {
                name,
                ggml_type,
                dims,
                bytes,
            });
        };
        push(
            "video_patch_proj.weight".into(),
            GgmlType::F32,
            vec![video, hidden],
        );
        push("video_patch_proj.bias".into(), GgmlType::F32, vec![hidden]);
        push(
            "audio_patch_proj.weight".into(),
            GgmlType::F32,
            vec![audio, hidden],
        );
        push("audio_patch_proj.bias".into(), GgmlType::F32, vec![hidden]);
        push(
            "condition_proj.weight".into(),
            GgmlType::Bf16,
            vec![text, hidden],
        );
        push("condition_proj.bias".into(), GgmlType::Bf16, vec![hidden]);
        push(
            "time_embedder.proj_in.weight".into(),
            GgmlType::F32,
            vec![freq, time_hidden],
        );
        push(
            "time_embedder.proj_in.bias".into(),
            GgmlType::F32,
            vec![time_hidden],
        );
        push(
            "time_embedder.proj_out.weight".into(),
            GgmlType::F32,
            vec![time_hidden, time_dim],
        );
        push(
            "time_embedder.proj_out.bias".into(),
            GgmlType::F32,
            vec![time_dim],
        );
        push(
            "final_layer.video_out.weight".into(),
            GgmlType::F32,
            vec![hidden, video],
        );
        push(
            "final_layer.video_out.bias".into(),
            GgmlType::F32,
            vec![video],
        );
        push(
            "final_layer.audio_out.weight".into(),
            GgmlType::F32,
            vec![hidden, audio],
        );
        push(
            "final_layer.audio_out.bias".into(),
            GgmlType::F32,
            vec![audio],
        );
        push(
            "final_layer.norm.weight".into(),
            GgmlType::Bf16,
            vec![hidden],
        );
        push(
            "final_layer.adaln_proj.linear.weight".into(),
            GgmlType::Bf16,
            vec![time_dim, 2 * hidden],
        );
        push(
            "final_layer.adaln_proj.linear.bias".into(),
            GgmlType::Bf16,
            vec![2 * hidden],
        );
        push(
            "token_refiner.final_norm.weight".into(),
            GgmlType::Bf16,
            vec![hidden],
        );
        for (prefix, layers, adaln) in [
            ("blocks.", spec.layers, true),
            ("token_refiner.blocks.", spec.refiner_layers, false),
        ] {
            for index in 0..layers {
                let block = format!("{prefix}{index}");
                push(
                    format!("{block}.attn.qkv_proj.weight"),
                    GgmlType::Q4_0,
                    vec![hidden, 3 * width],
                );
                push(
                    format!("{block}.attn.out_proj.weight"),
                    GgmlType::Bf16,
                    vec![width, hidden],
                );
                push(
                    format!("{block}.mlp.fc1.weight"),
                    GgmlType::Q4_0,
                    vec![hidden, 2 * ffn],
                );
                push(
                    format!("{block}.mlp.fc2.weight"),
                    GgmlType::Bf16,
                    vec![ffn, hidden],
                );
                push(
                    format!("{block}.norm1.weight"),
                    GgmlType::Bf16,
                    vec![hidden],
                );
                push(
                    format!("{block}.norm2.weight"),
                    GgmlType::Bf16,
                    vec![hidden],
                );
                push(
                    format!("{block}.attn.q_norm.weight"),
                    GgmlType::Bf16,
                    vec![spec.head_dim as u64],
                );
                push(
                    format!("{block}.attn.k_norm.weight"),
                    GgmlType::Bf16,
                    vec![spec.head_dim as u64],
                );
                if adaln {
                    push(
                        format!("{block}.adaln_proj.linear.weight"),
                        GgmlType::Bf16,
                        vec![time_dim, modulation],
                    );
                    push(
                        format!("{block}.adaln_proj.linear.bias"),
                        GgmlType::Bf16,
                        vec![modulation],
                    );
                }
            }
        }
        push(
            "rope.inv_freq".into(),
            GgmlType::F32,
            vec![spec.rope_freq_dim as u64],
        );
        out
    }

    #[test]
    fn minimax_h3_gguf_manifest_maps_names_shapes_and_quants() -> Result<()> {
        let spec = spec();
        let file = temp_gguf(&synthetic_tensors(&spec))?;
        let gguf = crate::gguf::GgufFile::open(&file.0)?;
        let expected = minimax_h3_expected(&spec)?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        ensure!(
            manifest.entries.len() == expected.len(),
            "manifest has {} entries, expected {}",
            manifest.entries.len(),
            expected.len()
        );
        let entry = &manifest.entries["proj_in.weight"];
        ensure!(
            entry.shape == [spec.hidden, spec.video_row_width()]
                && entry.quant == Some(GgmlType::F32)
                && entry.data_offsets == [0, 32 * 16 * 4]
                && entry.shard == file.0.file_name().unwrap().to_str().unwrap(),
            "proj_in mapping: {entry:?}"
        );
        let q = &manifest.entries["transformer_blocks.0.attn.to_q.weight"];
        let k = &manifest.entries["transformer_blocks.0.attn.to_k.weight"];
        let v = &manifest.entries["transformer_blocks.0.attn.to_v.weight"];
        for entry in [q, k, v] {
            ensure!(
                entry.shape == [32, 32]
                    && entry.quant == Some(GgmlType::Q4_0)
                    && entry.bytes == 32 * 32 * 18 / 32
                    && entry.data_offsets == [0, entry.bytes as u64],
                "fused qkv slice mapping: {entry:?}"
            );
        }
        ensure!(
            k.data_start - q.data_start == q.bytes as u64
                && v.data_start - k.data_start == q.bytes as u64,
            "fused qkv column ranges are not contiguous"
        );
        let fc1 = &manifest.entries["transformer_blocks.0.ff.net.0.proj.weight"];
        ensure!(
            fc1.shape == [2 * spec.ffn, spec.hidden]
                && fc1.quant == Some(GgmlType::Q4_0)
                && fc1.bytes == 32 * 128 * 18 / 32,
            "fused fc1 mapping: {fc1:?}"
        );
        ensure!(
            manifest
                .entries
                .contains_key("token_refiner.refiner_blocks.0.attn.to_k.weight")
                && manifest.entries.contains_key("norm_out.linear.weight")
                && manifest.entries["norm_out.linear.weight"].shape
                    == [2 * spec.hidden, spec.time_dim]
                && !manifest.entries.contains_key("rope.inv_freq")
                && !manifest.entries["transformer_blocks.0.adaln_proj.linear.weight"].load,
            "fixed/refiner mapping mismatch"
        );
        let resident: usize = manifest
            .entries
            .values()
            .filter(|entry| entry.load)
            .map(|entry| entry.bytes)
            .sum();
        ensure!(
            manifest.resident_bytes == resident,
            "resident bytes {} != loaded entries {resident}",
            manifest.resident_bytes
        );
        Ok(())
    }

    #[test]
    fn minimax_h3_gguf_manifest_rejects_unknown_tensor() -> Result<()> {
        let spec = spec();
        let mut tensors = synthetic_tensors(&spec);
        tensors.push(SyntheticTensor {
            name: "unknown.tensor".into(),
            ggml_type: GgmlType::F32,
            dims: vec![4],
            bytes: tensor_bytes(GgmlType::F32, &[4]),
        });
        let file = temp_gguf(&tensors)?;
        let gguf = crate::gguf::GgufFile::open(&file.0)?;
        ensure!(
            MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf).is_err(),
            "unknown GGUF tensor was accepted"
        );
        Ok(())
    }

    #[test]
    fn minimax_h3_gguf_manifest_rejects_a_half_swapped_non_q4_fc1() -> Result<()> {
        let spec = spec();
        let mut tensors = synthetic_tensors(&spec);
        for tensor in &mut tensors {
            if tensor.name == "blocks.0.mlp.fc1.weight" {
                tensor.ggml_type = GgmlType::F16;
                tensor.bytes = tensor_bytes(GgmlType::F16, &tensor.dims);
            }
        }
        let file = temp_gguf(&tensors)?;
        let gguf = crate::gguf::GgufFile::open(&file.0)?;
        ensure!(
            MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf).is_err(),
            "an F16 fused fc1 with swap_halves was accepted"
        );
        Ok(())
    }

    #[test]
    fn minimax_h3_gguf_manifest_rejects_a_pass_through_dtype_mismatch() -> Result<()> {
        let spec = spec();
        let mut tensors = synthetic_tensors(&spec);
        for tensor in &mut tensors {
            if tensor.name == "blocks.0.attn.out_proj.weight" {
                tensor.ggml_type = GgmlType::F32;
                tensor.bytes = tensor_bytes(GgmlType::F32, &tensor.dims);
            }
        }
        let file = temp_gguf(&tensors)?;
        let gguf = crate::gguf::GgufFile::open(&file.0)?;
        ensure!(
            MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf).is_err(),
            "an F32 source for the BF16 attn.out_proj was accepted"
        );
        Ok(())
    }

    #[test]
    #[ignore = "FF_MODEL, FF_H3_GGUF"]
    fn minimax_h3_gguf_manifest_matches_the_published_target() -> Result<()> {
        let model = std::env::var("FF_MODEL").context("FF_MODEL missing")?;
        let gguf_path = std::env::var("FF_H3_GGUF").context("FF_H3_GGUF missing")?;
        let spec = crate::minimax_h3::H3Spec::read(std::path::Path::new(&model))?;
        let gguf = crate::gguf::GgufFile::open(std::path::Path::new(&gguf_path))?;
        let manifest = MappedCheckpoint::minimax_h3_gguf_manifest(&spec, &gguf)?;
        ensure!(
            manifest.entries.len() == 638,
            "manifest has {} entries, expected 638",
            manifest.entries.len()
        );
        let proj_in = &manifest.entries["proj_in.weight"];
        ensure!(
            proj_in.quant == Some(GgmlType::Q4_0)
                && proj_in.shape == [spec.hidden, spec.video_row_width()]
                && !proj_in.swap_halves,
            "proj_in mapping: {proj_in:?}"
        );
        let fc1 = &manifest.entries["transformer_blocks.0.ff.net.0.proj.weight"];
        ensure!(
            fc1.swap_halves && fc1.quant == Some(GgmlType::Q4_0),
            "fused fc1 mapping: {fc1:?}"
        );
        for (name, shape) in [
            (
                "transformer_blocks.0.attn.to_q.weight",
                vec![spec.heads * spec.head_dim, spec.hidden],
            ),
            (
                "transformer_blocks.0.ff.net.2.weight",
                vec![spec.hidden, spec.ffn],
            ),
        ] {
            let entry = &manifest.entries[name];
            ensure!(
                entry.quant == Some(GgmlType::Q4_0) && entry.shape == shape,
                "{name} mapping: {entry:?}"
            );
        }
        let norm = &manifest.entries["transformer_blocks.0.norm1.weight"];
        ensure!(
            norm.quant == Some(GgmlType::F16) && norm.dtype == Dtype::BF16,
            "norm mapping: {norm:?}"
        );
        let adaln = &manifest.entries["transformer_blocks.0.adaln_proj.linear.weight"];
        ensure!(
            adaln.quant == Some(GgmlType::Q4_0) && !adaln.load,
            "packed adaln mapping: {adaln:?}"
        );
        let resident: usize = manifest
            .entries
            .values()
            .filter(|entry| entry.load)
            .map(|entry| entry.bytes)
            .sum();
        ensure!(
            manifest.resident_bytes == resident,
            "resident bytes {} != loaded entries {resident}",
            manifest.resident_bytes
        );
        Ok(())
    }
}

#[cfg(test)]
mod image_manifest_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const F32_NAMES: [&str; 18] = [
        "proj_in.bias",
        "proj_in.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_2.bias",
        "time_embedder.linear_2.weight",
        "pdd_proj_out.0.bias",
        "pdd_proj_out.0.weight",
        "pdd_proj_out.1.bias",
        "pdd_proj_out.1.weight",
        "pdd_proj_out.2.bias",
        "pdd_proj_out.2.weight",
        "pdd_proj_out.3.bias",
        "pdd_proj_out.3.weight",
        "pdd_proj_out.4.bias",
        "pdd_proj_out.4.weight",
        "pdd_proj_out.5.bias",
        "pdd_proj_out.5.weight",
    ];

    fn published() -> Result<crate::minimax_h3::ImageSpec> {
        crate::minimax_h3::ImageSpec::parse(&serde_json::from_str(include_str!(
            "../tests/data/minimax-h3-image-transformer-config.json"
        ))?)
    }

    fn tiny() -> crate::minimax_h3::ImageSpec {
        crate::minimax_h3::ImageSpec {
            hidden: 32,
            heads: 1,
            head_dim: 32,
            layers: 1,
            refiner_layers: 1,
            ffn: 64,
            in_channels: 4,
            patch: [1, 2, 2],
            text_dim: 16,
            freq_dim: 8,
            time_hidden: 16,
            time_dim: 8,
            rope_freq_dim: 4,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            pdd_heads: 6,
        }
    }

    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_checkpoint(
        spec: &crate::minimax_h3::ImageSpec,
        drop_name: Option<&str>,
        extra: Option<&str>,
    ) -> Result<Scratch> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "flyingfish-h3-image-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)?;
        let expected = minimax_h3_image_expected(spec)?;
        let mut owned = Vec::new();
        for (name, (dtype, shape, _)) in &expected {
            if drop_name == Some(name.as_str()) {
                continue;
            }
            owned.push((
                name.clone(),
                vec![0u8; tensor_bytes(*dtype, shape)?],
                *dtype,
                shape.clone(),
            ));
        }
        if let Some(name) = extra {
            owned.push((name.to_owned(), vec![0u8; 4], Dtype::F32, vec![1]));
        }
        let views = owned
            .iter()
            .map(|(name, data, dtype, shape)| {
                Ok((name.as_str(), TensorView::new(*dtype, shape.clone(), data)?))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        safetensors::tensor::serialize_to_file(
            views,
            None,
            &dir.join("diffusion_pytorch_model.safetensors"),
        )?;
        Ok(Scratch(dir))
    }

    #[test]
    fn minimax_h3_image_expected_names_match_the_published_set() -> Result<()> {
        let spec = published()?;
        let expected = minimax_h3_image_expected(&spec)?;
        let f32_names = expected
            .iter()
            .filter(|(_, (dtype, _, _))| *dtype == Dtype::F32)
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        let bf16 = expected
            .values()
            .filter(|(dtype, _, _)| *dtype == Dtype::BF16)
            .count();
        let dormant = expected
            .values()
            .filter(|(_, _, scope)| *scope == TensorScope::Dormant)
            .count();
        ensure!(
            expected.len() == 644
                && bf16 == 626
                && dormant == 100
                && f32_names == F32_NAMES.into_iter().collect::<BTreeSet<_>>()
                && !expected
                    .keys()
                    .any(|name| name.contains("audio") || name.starts_with("proj_out.")),
            "published image tensor set"
        );
        ensure!(
            expected["proj_in.weight"].1 == [5376, 96]
                && expected["pdd_proj_out.0.weight"].1 == [96, 5376]
                && expected["pdd_proj_out.5.weight"].1 == [96, 5376]
                && expected["transformer_blocks.0.adaln_proj.linear.weight"].1 == [96768, 2688]
                && expected["transformer_blocks.0.adaln_proj.linear.weight"].2
                    == TensorScope::Dormant
                && !expected
                    .contains_key("token_refiner.refiner_blocks.0.adaln_proj.linear.weight"),
            "published image shapes"
        );
        Ok(())
    }

    #[test]
    fn minimax_h3_image_manifest_rejects_a_missing_head_and_an_audio_key() -> Result<()> {
        let spec = tiny();
        let complete = write_checkpoint(&spec, None, None)?;
        let manifest = MappedCheckpoint::open_component(&complete.0, "diffusion_pytorch_model")?
            .minimax_h3_image_manifest(&spec)?;
        ensure!(
            manifest.entries.len() == 46
                && manifest.entries["pdd_proj_out.5.weight"].dtype == Dtype::F32
                && manifest.entries["pdd_proj_out.5.weight"].load
                && manifest.entries["transformer_blocks.0.adaln_proj.linear.weight"].scope
                    == TensorScope::Dormant
                && !manifest.entries["transformer_blocks.0.adaln_proj.linear.weight"].load,
            "tiny image manifest"
        );
        let missing = write_checkpoint(&spec, Some("pdd_proj_out.5.weight"), None)?;
        ensure!(
            MappedCheckpoint::open_component(&missing.0, "diffusion_pytorch_model")?
                .minimax_h3_image_manifest(&spec)
                .is_err(),
            "accepted a checkpoint missing pdd_proj_out.5"
        );
        let audio = write_checkpoint(&spec, None, Some("audio_proj_in.weight"))?;
        ensure!(
            MappedCheckpoint::open_component(&audio.0, "diffusion_pytorch_model")?
                .minimax_h3_image_manifest(&spec)
                .is_err(),
            "accepted an audio projection"
        );
        Ok(())
    }
}
