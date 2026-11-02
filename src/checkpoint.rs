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
    if !index.weight_map.keys().any(|name| name.starts_with("mtp.")) {
        return Ok(false);
    }
    Ok(crate::model_spec::ModelSpec::read(dir)?
        .architecture
        .mtp_decode())
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
            #[cfg(any(feature = "cuda", feature = "rocm"))]
            files,
        })
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
            Architecture::Qwen35Text => "Q38",
            Architecture::MiniCpmV47 => "MiniCPM",
            Architecture::Flint => "Flint",
            Architecture::Clover => "Clover",
            Architecture::Llama
            | Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => {
                bail!("checkpoint manifest requires a Qwen3.5-family spec")
            }
            Architecture::DeepseekV41 => {
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
            Architecture::Flint | Architecture::Clover => {}
            Architecture::Llama
            | Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => unreachable!(),
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
        let Some(a) = &spec.audio else {
            return Ok(());
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
        let global_prefix = match spec.architecture {
            Architecture::Flint | Architecture::Clover => "model",
            Architecture::Qwen35Text | Architecture::MiniCpmV47 | Architecture::Llama => {
                "model.language_model"
            }
            Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::DeepseekV41
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2 => {
                unreachable!("architecture has a dedicated manifest path")
            }
        };
        for name in [
            format!("{global_prefix}.embed_tokens.weight"),
            "lm_head.weight".to_owned(),
        ] {
            put(
                expected,
                name,
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
                    let q_out = match spec.architecture {
                        Architecture::Flint => q_width,
                        Architecture::Qwen35Text
                        | Architecture::Clover
                        | Architecture::MiniCpmV47
                        | Architecture::Llama => 2 * q_width,
                        Architecture::Glm5Next
                        | Architecture::LoopVl
                        | Architecture::Gemma4
                        | Architecture::EmbeddingGemma2 => {
                            unreachable!("architecture has a dedicated manifest path")
                        }
                        Architecture::DeepseekV41 => {
                            unreachable!("DeepSeek-V4.1 has a dedicated manifest path")
                        }
                    };
                    projection(
                        expected,
                        format!("{p}.self_attn.q_proj"),
                        q_out,
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

    fn glm5_next_entries(
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
                Some(format!(
                    "{}.{}",
                    name.strip_suffix(".weight")
                        .context("invalid FP8 tensor suffix")?,
                    scale_suffix.context("FP8 tensor without scale layout")?
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

/// Layer name prefixes of the published checkpoints.
const LAYER_PREFIXES: [&str; 4] = [
    "model.layers.",
    "model.language_model.layers.",
    "language_model.layers.",
    "layers.",
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
    #[cfg(any(feature = "cuda", feature = "rocm", test))]
    pub(crate) fn native_byte_storage(&self) -> bool {
        matches!(self.dtype, Dtype::F8_E4M3 | Dtype::F32)
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
        self.role.starts_with("mlp.experts.")
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

#[derive(Debug, serde::Serialize)]
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

    #[test]
    fn clover_manifest_counts_all_side_weights() -> Result<()> {
        ensure!(
            serde_json::from_value::<crate::model_spec::Architecture>(serde_json::json!(
                "DiffusionGemma"
            ))
            .is_err(),
            "unvalidated architecture accepted in model geometry"
        );
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
            let expected_vision_bytes = if spec.hidden == 1536 {
                337_089_408
            } else {
                338_662_272
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
                    == if spec.hidden == 1536 {
                        614_368_768
                    } else {
                        617_514_496
                    }
                    && audio_manifest.entries.values().filter(|e| e.load).count() == 752
                    && audio_manifest
                        .entries
                        .values()
                        .all(|e| e.load == (e.scope == TensorScope::Audio)),
                "audio manifest residency differs from published shapes"
            );
            let mut wrong_audio = spec.clone();
            wrong_audio
                .audio
                .as_mut()
                .context("audio spec")?
                .conv_kernel = 3;
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
        Ok(())
    }
}
