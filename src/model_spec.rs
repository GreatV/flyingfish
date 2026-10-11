//! Normalized model geometry. Device execution remains gated by backend capability.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Qwen35Text,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Mixer {
    FullAttention,
    Gdn,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerPlan {
    pub mixer: Mixer,
    pub state_index: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSpec {
    pub abi_version: u32,
    pub architecture: Architecture,
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub q_width: u32,
    pub max_position: u32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub eos: Vec<u32>,
    pub layers: Vec<LayerPlan>,
    #[serde(default)]
    pub vision: Option<VisionConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub channels: u32,
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub output: u32,
    pub position_side: u32,
    pub image_token: u32,
    pub video_token: u32,
    pub start_token: u32,
    pub end_token: u32,
}

impl VisionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.depth,
                self.hidden,
                self.heads,
                self.intermediate,
                self.channels,
                self.patch,
                self.temporal_patch,
                self.merge,
                self.output,
                self.position_side
            ] == [27, 1152, 16, 4304, 3, 16, 2, 2, 5120, 48],
            "unsupported Q38 vision geometry"
        );
        ensure!(
            [
                self.image_token,
                self.video_token,
                self.start_token,
                self.end_token
            ] == [248056, 248057, 248053, 248054],
            "unsupported Q38 image token IDs"
        );
        Ok(())
    }

    pub fn patch_elements(&self) -> Result<usize> {
        self.validate()?;
        Ok((self.channels * self.temporal_patch * self.patch * self.patch) as usize)
    }

    fn from_root(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config") else {
            return Ok(None);
        };
        equal(v, "model_type", "qwen3_5".into())?;
        equal(v, "hidden_act", "gelu_pytorch_tanh".into())?;
        equal(v, "deepstack_visual_indexes", serde_json::json!([]))?;
        ensure!(
            integer(v, "num_position_embeddings")? == 2304,
            "unsupported vision position table"
        );
        let spec = Self {
            depth: integer(v, "depth")?,
            hidden: integer(v, "hidden_size")?,
            heads: integer(v, "num_heads")?,
            intermediate: integer(v, "intermediate_size")?,
            channels: integer(v, "in_channels")?,
            patch: integer(v, "patch_size")?,
            temporal_patch: integer(v, "temporal_patch_size")?,
            merge: integer(v, "spatial_merge_size")?,
            output: integer(v, "out_hidden_size")?,
            position_side: 48,
            image_token: integer(root, "image_token_id")?,
            video_token: integer(root, "video_token_id")?,
            start_token: integer(root, "vision_start_token_id")?,
            end_token: integer(root, "vision_end_token_id")?,
        };
        spec.validate()?;
        Ok(Some(spec))
    }
}
fn integer(v: &Value, key: &str) -> Result<u32> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing/invalid {key}"))?;
    u32::try_from(n).with_context(|| format!("{key} overflows u32"))
}
fn equal(v: &Value, key: &str, expected: Value) -> Result<()> {
    ensure!(
        v.get(key) == Some(&expected),
        "unsupported {key}: {:?}",
        v.get(key)
    );
    Ok(())
}
fn eos(v: &Value, vocab: u32) -> Result<Vec<u32>> {
    let values = match v {
        Value::Array(a) => a.clone(),
        Value::Number(_) => vec![v.clone()],
        _ => bail!("EOS must be an integer or array"),
    };
    let mut out = Vec::new();
    for value in values {
        let n = value.as_u64().context("EOS must be unsigned integer")?;
        ensure!(n < u64::from(vocab), "EOS outside vocabulary");
        out.push(n as u32);
    }
    out.sort_unstable();
    out.dedup();
    ensure!(!out.is_empty(), "empty EOS set");
    Ok(out)
}
impl ModelSpec {
    /// Compatibility metadata for the public model/CLI surface. Qwen execution
    /// uses ModelSpec; this does not relax Config::read's Llama validation.
    pub fn config(&self) -> crate::config::Config {
        crate::config::Config {
            model_type: match self.architecture {
                Architecture::Llama => "llama",
                Architecture::Qwen35Text => "qwen3_5_text",
            }
            .into(),
            hidden_size: self.hidden as usize,
            intermediate_size: self.intermediate as usize,
            num_hidden_layers: self.layers.len(),
            num_attention_heads: self.q_heads as usize,
            num_key_value_heads: self.kv_heads as usize,
            head_dim: self.head_dim as usize,
            vocab_size: self.vocab as usize,
            max_position_embeddings: self.max_position as usize,
            rms_norm_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: self.eos.clone(),
            attention_bias: false,
            mlp_bias: false,
        }
    }
    pub fn read(dir: &Path) -> Result<Self> {
        let root: Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
        if root.get("model_type").and_then(Value::as_str) == Some("llama") {
            let c = crate::config::Config::read(dir)?;
            return Self::from_llama(&c);
        }
        let path = dir.join("generation_config.json");
        let generation = if path.exists() {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            Value::Null
        };
        Self::qwen(&root, &generation)
    }
    pub fn from_llama(c: &crate::config::Config) -> Result<Self> {
        let u = |n: usize| u32::try_from(n).context("Llama dimension overflows u32");
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Llama,
            hidden: u(c.hidden_size)?,
            intermediate: u(c.intermediate_size)?,
            vocab: u(c.vocab_size)?,
            q_heads: u(c.num_attention_heads)?,
            kv_heads: u(c.num_key_value_heads)?,
            head_dim: u(c.head_dim)?,
            q_width: u(c.hidden_size)?,
            max_position: u(c.max_position_embeddings)?,
            rms_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            rotary_dim: u(c.head_dim)?,
            eos: c.eos_token_id.clone(),
            vision: None,
            layers: (0..u(c.num_hidden_layers)?)
                .map(|state_index| LayerPlan {
                    mixer: Mixer::FullAttention,
                    state_index,
                })
                .collect(),
        })
    }
    pub fn qwen(root: &Value, generation: &Value) -> Result<Self> {
        equal(root, "model_type", "qwen3_5".into())?;
        equal(
            root,
            "architectures",
            serde_json::json!(["Qwen3_5ForConditionalGeneration"]),
        )?;
        equal(root, "tie_word_embeddings", false.into())?;
        let t = root.get("text_config").context("missing text_config")?;
        for (key, val) in [
            ("model_type", "qwen3_5_text"),
            ("dtype", "bfloat16"),
            ("hidden_act", "silu"),
            ("mamba_ssm_dtype", "float32"),
            ("output_gate_type", "swish"),
        ] {
            equal(t, key, val.into())?;
        }
        for (key, val) in [
            ("hidden_size", 5120),
            ("intermediate_size", 17408),
            ("vocab_size", 248320),
            ("num_hidden_layers", 64),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
            ("full_attention_interval", 4),
            ("linear_conv_kernel_dim", 4),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 48),
            ("max_position_embeddings", 262144),
        ] {
            ensure!(integer(t, key)? == val, "unsupported {key}");
        }
        equal(t, "attention_bias", false.into())?;
        equal(t, "tie_word_embeddings", false.into())?;
        equal(t, "attn_output_gate", true.into())?;
        if let Some(bias) = t.get("mlp_bias") {
            ensure!(bias == &Value::Bool(false), "MLP bias unsupported");
        }
        let eps = t
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .context("missing rms_norm_eps")?;
        ensure!(eps == 1e-6, "unsupported norm epsilon");
        let rope = t
            .get("rope_parameters")
            .context("missing rope_parameters")?;
        equal(rope, "rope_type", "default".into())?;
        equal(rope, "mrope_interleaved", true.into())?;
        equal(rope, "mrope_section", serde_json::json!([11, 11, 10]))?;
        ensure!(
            integer(rope, "rope_theta")? == 10_000_000,
            "unsupported rope theta"
        );
        for value in [
            t.get("partial_rotary_factor"),
            rope.get("partial_rotary_factor"),
        ] {
            ensure!(
                value.and_then(Value::as_f64) == Some(0.25),
                "unsupported partial RoPE"
            );
        }
        let quant = root
            .get("quantization_config")
            .context("missing quantization_config")?;
        for (key, val) in [
            ("quant_method", "fp8"),
            ("fmt", "e4m3"),
            ("activation_scheme", "dynamic"),
        ] {
            equal(quant, key, val.into())?;
        }
        equal(quant, "weight_block_size", serde_json::json!([128, 128]))?;
        let types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("missing layer_types")?;
        ensure!(types.len() == 64, "wrong layer count");
        let mut counts = [0, 0];
        let mut layers = Vec::with_capacity(64);
        for (i, kind) in types.iter().enumerate() {
            let (mixer, slot, expected) = if i % 4 == 3 {
                (Mixer::FullAttention, 0, "full_attention")
            } else {
                (Mixer::Gdn, 1, "linear_attention")
            };
            ensure!(
                kind.as_str() == Some(expected),
                "unsupported mixer at layer {i}"
            );
            layers.push(LayerPlan {
                mixer,
                state_index: counts[slot],
            });
            counts[slot] += 1;
        }
        let eos_value = generation
            .get("eos_token_id")
            .filter(|x| !x.is_null())
            .or_else(|| t.get("eos_token_id"))
            .context("missing EOS")?;
        Ok(Self {
            abi_version: 1,
            architecture: Architecture::Qwen35Text,
            hidden: 5120,
            intermediate: 17408,
            vocab: 248320,
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            q_width: 6144,
            max_position: 262144,
            rms_eps: eps as f32,
            rope_theta: 1e7,
            rotary_dim: 64,
            eos: eos(eos_value, 248320)?,
            layers,
            vision: VisionConfig::from_root(root)?,
        })
    }
}
