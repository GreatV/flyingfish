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
            Architecture::Llama | Architecture::Glm5Next | Architecture::LoopVl => {
                bail!("checkpoint manifest requires a Qwen3.5-family spec")
            }
            Architecture::DeepseekV41 => {
                unreachable!("DeepSeek-V4.1 has a dedicated manifest path")
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
            Architecture::Flint => {}
            Architecture::Llama | Architecture::Glm5Next | Architecture::LoopVl => unreachable!(),
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
            Architecture::Flint => "model",
            Architecture::Qwen35Text | Architecture::MiniCpmV47 | Architecture::Llama => {
                "model.language_model"
            }
            Architecture::Glm5Next | Architecture::LoopVl | Architecture::DeepseekV41 => {
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
                        | Architecture::MiniCpmV47
                        | Architecture::Llama => 2 * q_width,
                        Architecture::Glm5Next | Architecture::LoopVl => {
                            unreachable!("GLM has a dedicated manifest path")
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
                    put(expected, format!("{p}.{name}"), Dtype::F32, vec![24], scope);
                }
                for name in ["hc_attn_scale", "hc_ffn_scale"] {
                    put(expected, format!("{p}.{name}"), Dtype::F32, vec![3], scope);
                }
                for name in ["hc_attn_fn", "hc_ffn_fn"] {
                    put(
                        expected,
                        format!("{p}.{name}"),
                        Dtype::BF16,
                        vec![24, 4 * hidden],
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
        let width = (g.value_heads * g.value_head_dim) as usize;
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
            vec![
                (glm.qk_nope_head_dim + glm.v_head_dim) as usize * spec.q_heads as usize,
                kv_lora,
            ],
            scope,
        );
        for (role, n, k) in [
            ("q_a_proj", q_lora, hidden),
            ("q_b_proj", spec.q_width as usize, q_lora),
            ("kv_a_proj_with_mqa", kv_lora, hidden),
            ("o_proj", hidden, spec.q_width as usize),
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
const LAYER_PREFIXES: [&str; 3] = ["model.layers.", "model.language_model.layers.", "layers."];

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
}
