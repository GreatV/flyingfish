use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, de::Error as _};
use std::{fs, path::Path};

pub const GLM5_NEXT_ARCHITECTURE: &str = "Glm5NextForConditionalGeneration";
pub const GLM5_NEXT_MODEL_TYPE: &str = "glm5_next";
pub const GLM5_NEXT_TEXT_MODEL_TYPE: &str = "glm5_next_text";
pub const FP8_WEIGHT_BLOCK_SIZE: [usize; 2] = [128, 128];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    LinearAttention,
    DeepseekSparseAttention,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MlpKind {
    Dense,
    Sparse,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IndexerKind {
    Full,
    Shared,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GlmQuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub fmt: String,
    pub weight_block_size: [usize; 2],
}

impl GlmQuantizationConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.quant_method == "fp8",
            "unsupported GLM quantization method {:?}; expected fp8",
            self.quant_method
        );
        anyhow::ensure!(
            self.activation_scheme == "dynamic",
            "unsupported GLM activation scheme {:?}; expected dynamic",
            self.activation_scheme
        );
        anyhow::ensure!(
            self.fmt == "e4m3",
            "unsupported GLM FP8 format {:?}; expected e4m3",
            self.fmt
        );
        anyhow::ensure!(
            self.weight_block_size == FP8_WEIGHT_BLOCK_SIZE,
            "unsupported GLM FP8 weight block {:?}; expected {:?}",
            self.weight_block_size,
            FP8_WEIGHT_BLOCK_SIZE
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmTextConfig {
    pub dtype: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub first_k_dense_replace: usize,
    pub n_shared_experts: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f64,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub scoring_func: String,
    pub topk_method: String,
    pub moe_router_dtype: String,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub layer_types: Vec<AttentionKind>,
    pub mlp_layer_types: Vec<MlpKind>,
    pub indexer_types: Vec<IndexerKind>,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub linear_num_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_lower_bound: Option<f64>,
    pub hc_mult: usize,
    pub hc_eps: f64,
    pub hc_sinkhorn_iters: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub max_position_embeddings: usize,
    pub num_nextn_predict_layers: usize,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub pad_token_id: u32,
    pub eos_token_ids: Vec<u32>,
}

impl GlmTextConfig {
    pub fn validate(&self) -> Result<()> {
        ensure_non_zero("vocab_size", self.vocab_size)?;
        ensure_non_zero("hidden_size", self.hidden_size)?;
        ensure_non_zero("intermediate_size", self.intermediate_size)?;
        ensure_non_zero("moe_intermediate_size", self.moe_intermediate_size)?;
        ensure_non_zero("num_hidden_layers", self.num_hidden_layers)?;
        ensure_non_zero("num_attention_heads", self.num_attention_heads)?;
        ensure_non_zero("num_key_value_heads", self.num_key_value_heads)?;
        ensure_non_zero("n_shared_experts", self.n_shared_experts)?;
        ensure_non_zero("n_routed_experts", self.n_routed_experts)?;
        ensure_non_zero("num_experts_per_tok", self.num_experts_per_tok)?;
        ensure_non_zero("n_group", self.n_group)?;
        ensure_non_zero("topk_group", self.topk_group)?;
        ensure_non_zero("q_lora_rank", self.q_lora_rank)?;
        ensure_non_zero("kv_lora_rank", self.kv_lora_rank)?;
        ensure_non_zero("qk_nope_head_dim", self.qk_nope_head_dim)?;
        ensure_non_zero("v_head_dim", self.v_head_dim)?;
        ensure_non_zero("index_topk", self.index_topk)?;
        ensure_non_zero("index_kpool", self.index_kpool)?;
        ensure_non_zero("index_n_heads", self.index_n_heads)?;
        ensure_non_zero("index_head_dim", self.index_head_dim)?;
        ensure_non_zero("linear_num_heads", self.linear_num_heads)?;
        ensure_non_zero("linear_head_dim", self.linear_head_dim)?;
        ensure_non_zero("linear_conv_kernel_dim", self.linear_conv_kernel_dim)?;
        ensure_non_zero("hc_mult", self.hc_mult)?;
        ensure_non_zero("hc_sinkhorn_iters", self.hc_sinkhorn_iters)?;
        ensure_non_zero("max_position_embeddings", self.max_position_embeddings)?;

        anyhow::ensure!(
            self.dtype == "bfloat16",
            "unsupported GLM text dtype {:?}; expected bfloat16",
            self.dtype
        );
        anyhow::ensure!(
            self.num_attention_heads == self.num_key_value_heads,
            "num_attention_heads ({}) must equal num_key_value_heads ({})",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        anyhow::ensure!(
            self.qk_rope_head_dim == 0,
            "GLM-5.3-Flash DSA is NoPE; qk_rope_head_dim must be zero"
        );
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .context("qk head dimension overflow")?;
        self.linear_num_heads
            .checked_mul(self.linear_head_dim)
            .context("linear-attention projection dimension overflow")?;
        self.hc_mult
            .checked_mul(self.hidden_size)
            .context("mHC flattened hidden dimension overflow")?;
        self.n_routed_experts
            .checked_mul(self.moe_intermediate_size)
            .context("MoE expert dimension overflow")?;

        anyhow::ensure!(
            self.layer_types.len() == self.num_hidden_layers,
            "layer_types has {} entries, expected {}",
            self.layer_types.len(),
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.mlp_layer_types.len() == self.num_hidden_layers,
            "mlp_layer_types has {} entries, expected {}",
            self.mlp_layer_types.len(),
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.indexer_types.len() == self.num_hidden_layers,
            "indexer_types has {} entries, expected {}",
            self.indexer_types.len(),
            self.num_hidden_layers
        );
        anyhow::ensure!(
            self.layer_types.contains(&AttentionKind::LinearAttention),
            "layer_types must contain at least one linear_attention layer"
        );
        anyhow::ensure!(
            self.layer_types
                .contains(&AttentionKind::DeepseekSparseAttention),
            "layer_types must contain at least one deepseek_sparse_attention layer"
        );

        anyhow::ensure!(
            self.first_k_dense_replace <= self.num_hidden_layers,
            "first_k_dense_replace exceeds num_hidden_layers"
        );
        for (layer, kind) in self.mlp_layer_types.iter().copied().enumerate() {
            let expected = if layer < self.first_k_dense_replace {
                MlpKind::Dense
            } else {
                MlpKind::Sparse
            };
            anyhow::ensure!(
                kind == expected,
                "mlp_layer_types[{layer}] is {kind:?}, expected {expected:?} from first_k_dense_replace={} ",
                self.first_k_dense_replace
            );
        }

        for layer in 0..self.num_hidden_layers {
            if self.layer_types[layer] == AttentionKind::DeepseekSparseAttention
                && self.indexer_types[layer] == IndexerKind::Shared
            {
                anyhow::ensure!(
                    layer > 0
                        && self.layer_types[layer - 1] == AttentionKind::DeepseekSparseAttention,
                    "shared DSA indexer at layer {layer} has no immediately preceding DSA selection"
                );
            }
        }

        anyhow::ensure!(
            self.index_topk.is_multiple_of(self.index_kpool),
            "index_topk ({}) must be divisible by index_kpool ({})",
            self.index_topk,
            self.index_kpool
        );
        anyhow::ensure!(
            self.n_routed_experts.is_multiple_of(self.n_group),
            "n_routed_experts ({}) must be divisible by n_group ({})",
            self.n_routed_experts,
            self.n_group
        );
        anyhow::ensure!(
            self.topk_group <= self.n_group,
            "topk_group ({}) exceeds n_group ({})",
            self.topk_group,
            self.n_group
        );
        anyhow::ensure!(
            self.n_group == 1 && self.topk_group == 1,
            "the GLM-5.3-Flash adapter currently supports exactly one router group"
        );
        let experts_per_group = self.n_routed_experts / self.n_group;
        anyhow::ensure!(
            experts_per_group >= 2,
            "each router group must contain at least two experts"
        );
        anyhow::ensure!(
            self.num_experts_per_tok <= experts_per_group * self.topk_group,
            "num_experts_per_tok ({}) exceeds the selected router groups' capacity ({})",
            self.num_experts_per_tok,
            experts_per_group * self.topk_group
        );
        anyhow::ensure!(
            self.norm_topk_prob,
            "only normalized top-k routing is supported"
        );
        anyhow::ensure!(
            self.scoring_func == "sigmoid",
            "unsupported MoE scoring function {:?}; expected sigmoid",
            self.scoring_func
        );
        anyhow::ensure!(
            self.topk_method == "noaux_tc",
            "unsupported MoE top-k method {:?}; expected noaux_tc",
            self.topk_method
        );
        anyhow::ensure!(
            self.moe_router_dtype == "float32",
            "unsupported MoE router dtype {:?}; expected float32",
            self.moe_router_dtype
        );
        anyhow::ensure!(
            self.hidden_act == "silu",
            "unsupported activation {:?}; expected silu",
            self.hidden_act
        );
        anyhow::ensure!(
            self.index_kpool_always_select_tail,
            "index_kpool_always_select_tail=false is incompatible with the bounded full-causal DSA profile"
        );
        anyhow::ensure!(
            !self.attention_bias,
            "attention_bias=true is not supported by the GLM-5.3-Flash checkpoint layout"
        );
        anyhow::ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings=true is not supported; the checkpoint has a separate lm_head"
        );
        anyhow::ensure!(
            self.use_cache,
            "use_cache=false is incompatible with the autoregressive KDA/MLA cache path"
        );
        anyhow::ensure!(
            (self.pad_token_id as usize) < self.vocab_size,
            "pad_token_id {} is outside vocab_size {}",
            self.pad_token_id,
            self.vocab_size
        );
        anyhow::ensure!(
            !self.eos_token_ids.is_empty(),
            "eos_token_id must not be empty"
        );
        for &token in &self.eos_token_ids {
            anyhow::ensure!(
                (token as usize) < self.vocab_size,
                "eos_token_id {token} is outside vocab_size {}",
                self.vocab_size
            );
        }

        ensure_finite_non_negative("attention_dropout", self.attention_dropout)?;
        anyhow::ensure!(
            self.attention_dropout <= 1.0,
            "attention_dropout must not exceed 1"
        );
        ensure_finite_positive("routed_scaling_factor", self.routed_scaling_factor)?;
        ensure_finite_positive("swiglu_limit", self.swiglu_limit)?;
        ensure_finite_positive("rms_norm_eps", self.rms_norm_eps)?;
        ensure_finite_positive("hc_eps", self.hc_eps)?;
        let lower_bound = self
            .linear_lower_bound
            .context("linear_lower_bound is required by the supported KDA profile")?;
        anyhow::ensure!(
            lower_bound.is_finite() && lower_bound < 0.0,
            "linear_lower_bound must be finite and negative"
        );
        Ok(())
    }

    pub fn linear_qkv_dim(&self) -> Result<usize> {
        self.linear_num_heads
            .checked_mul(self.linear_head_dim)
            .context("linear-attention projection dimension overflow")
    }

    pub fn mla_qk_head_dim(&self) -> Result<usize> {
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .context("MLA Q/K head dimension overflow")
    }

    pub fn linear_attention_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| (*kind == AttentionKind::LinearAttention).then_some(layer))
    }

    pub fn sparse_attention_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| {
                (*kind == AttentionKind::DeepseekSparseAttention).then_some(layer)
            })
    }

    #[cfg(test)]
    pub(crate) fn tiny() -> Self {
        Self {
            dtype: "bfloat16".to_owned(),
            vocab_size: 128,
            hidden_size: 16,
            intermediate_size: 32,
            moe_intermediate_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            first_k_dense_replace: 1,
            n_shared_experts: 1,
            n_routed_experts: 4,
            num_experts_per_tok: 2,
            routed_scaling_factor: 2.5,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            scoring_func: "sigmoid".to_owned(),
            topk_method: "noaux_tc".to_owned(),
            moe_router_dtype: "float32".to_owned(),
            q_lora_rank: 8,
            kv_lora_rank: 4,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 0,
            v_head_dim: 4,
            layer_types: vec![
                AttentionKind::LinearAttention,
                AttentionKind::LinearAttention,
                AttentionKind::LinearAttention,
                AttentionKind::DeepseekSparseAttention,
            ],
            mlp_layer_types: vec![
                MlpKind::Dense,
                MlpKind::Sparse,
                MlpKind::Sparse,
                MlpKind::Sparse,
            ],
            indexer_types: vec![IndexerKind::Full; 4],
            index_topk: 8,
            index_kpool: 2,
            index_kpool_always_select_tail: true,
            index_n_heads: 2,
            index_head_dim: 4,
            linear_num_heads: 2,
            linear_head_dim: 4,
            linear_conv_kernel_dim: 4,
            linear_lower_bound: Some(-5.0),
            hc_mult: 2,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 3,
            hidden_act: "silu".to_owned(),
            swiglu_limit: 10.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            attention_dropout: 0.0,
            max_position_embeddings: 64,
            num_nextn_predict_layers: 0,
            use_cache: true,
            tie_word_embeddings: false,
            pad_token_id: 0,
            eos_token_ids: vec![2],
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawGlmTextConfig {
    model_type: String,
    #[serde(alias = "torch_dtype")]
    dtype: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    first_k_dense_replace: usize,
    n_shared_experts: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    routed_scaling_factor: f64,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    scoring_func: String,
    topk_method: String,
    moe_router_dtype: String,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    head_dim: usize,
    v_head_dim: usize,
    mhc: bool,
    mla_use_nope: bool,
    #[serde(default)]
    layer_types: Option<Vec<AttentionKind>>,
    mlp_layer_types: Vec<MlpKind>,
    indexer_types: Vec<IndexerKind>,
    index_topk: usize,
    index_kpool: usize,
    index_kpool_always_select_tail: bool,
    index_n_heads: usize,
    index_head_dim: usize,
    #[serde(default)]
    linear_num_heads: Option<usize>,
    #[serde(default)]
    linear_head_dim: Option<usize>,
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    linear_lower_bound: Option<f64>,
    #[serde(default)]
    linear_attn_config: Option<LegacyLinearAttentionConfig>,
    hc_mult: usize,
    hc_eps: f64,
    hc_sinkhorn_iters: usize,
    hidden_act: String,
    swiglu_limit: f64,
    rms_norm_eps: f64,
    attention_bias: bool,
    attention_dropout: f64,
    max_position_embeddings: usize,
    num_nextn_predict_layers: usize,
    use_cache: bool,
    tie_word_embeddings: bool,
    pad_token_id: u32,
    eos_token_id: TokenIds,
}

#[derive(Debug, Deserialize)]
struct LegacyLinearAttentionConfig {
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    short_conv_kernel_size: Option<usize>,
    #[serde(default)]
    gate_lower_bound: Option<f64>,
    #[serde(default)]
    kda_layers: Option<Vec<usize>>,
    #[serde(default)]
    full_attn_layers: Option<Vec<usize>>,
}

impl<'de> Deserialize<'de> for GlmTextConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawGlmTextConfig::deserialize(deserializer)?;
        normalize_text_config(raw).map_err(D::Error::custom)
    }
}

fn normalize_text_config(raw: RawGlmTextConfig) -> std::result::Result<GlmTextConfig, String> {
    if raw.model_type != GLM5_NEXT_TEXT_MODEL_TYPE {
        return Err(format!(
            "text_config.model_type is {:?}, expected {GLM5_NEXT_TEXT_MODEL_TYPE}",
            raw.model_type
        ));
    }
    if !raw.mhc {
        return Err("text_config.mhc=false is incompatible with the mHC execution path".to_owned());
    }
    if !raw.mla_use_nope {
        return Err(
            "text_config.mla_use_nope=false is incompatible with the NoPE MLA execution path"
                .to_owned(),
        );
    }
    if raw.head_dim != 0 {
        return Err(format!(
            "text_config.head_dim must be zero for the specialized KDA/MLA heads, found {}",
            raw.head_dim
        ));
    }
    let expected_qk_head_dim = raw
        .qk_nope_head_dim
        .checked_add(raw.qk_rope_head_dim)
        .ok_or_else(|| "qk head dimension overflow".to_owned())?;
    if raw.qk_head_dim != expected_qk_head_dim {
        return Err(format!(
            "text_config.qk_head_dim is {}, expected qk_nope_head_dim + qk_rope_head_dim = {expected_qk_head_dim}",
            raw.qk_head_dim
        ));
    }
    let legacy = raw.linear_attn_config.as_ref();
    let linear_num_heads = reconcile(
        "linear_num_heads",
        raw.linear_num_heads,
        legacy.and_then(|value| value.num_heads),
    )?;
    let linear_head_dim = reconcile(
        "linear_head_dim",
        raw.linear_head_dim,
        legacy.and_then(|value| value.head_dim),
    )?;
    let linear_conv_kernel_dim = reconcile(
        "linear_conv_kernel_dim",
        raw.linear_conv_kernel_dim,
        legacy.and_then(|value| value.short_conv_kernel_size),
    )?;
    let linear_lower_bound = Some(reconcile_f64(
        "linear_lower_bound",
        raw.linear_lower_bound,
        legacy.and_then(|value| value.gate_lower_bound),
    )?);

    let layer_types = normalize_layer_schedule(
        raw.num_hidden_layers,
        raw.layer_types,
        legacy.and_then(|value| value.kda_layers.as_deref()),
        legacy.and_then(|value| value.full_attn_layers.as_deref()),
    )?;
    let mlp_layer_types = raw.mlp_layer_types;
    let indexer_types = raw.indexer_types;

    Ok(GlmTextConfig {
        dtype: raw.dtype,
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        moe_intermediate_size: raw.moe_intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        first_k_dense_replace: raw.first_k_dense_replace,
        n_shared_experts: raw.n_shared_experts,
        n_routed_experts: raw.n_routed_experts,
        num_experts_per_tok: raw.num_experts_per_tok,
        routed_scaling_factor: raw.routed_scaling_factor,
        n_group: raw.n_group,
        topk_group: raw.topk_group,
        norm_topk_prob: raw.norm_topk_prob,
        scoring_func: raw.scoring_func,
        topk_method: raw.topk_method,
        moe_router_dtype: raw.moe_router_dtype,
        q_lora_rank: raw.q_lora_rank,
        kv_lora_rank: raw.kv_lora_rank,
        qk_nope_head_dim: raw.qk_nope_head_dim,
        qk_rope_head_dim: raw.qk_rope_head_dim,
        v_head_dim: raw.v_head_dim,
        layer_types,
        mlp_layer_types,
        indexer_types,
        index_topk: raw.index_topk,
        index_kpool: raw.index_kpool,
        index_kpool_always_select_tail: raw.index_kpool_always_select_tail,
        index_n_heads: raw.index_n_heads,
        index_head_dim: raw.index_head_dim,
        linear_num_heads,
        linear_head_dim,
        linear_conv_kernel_dim,
        linear_lower_bound,
        hc_mult: raw.hc_mult,
        hc_eps: raw.hc_eps,
        hc_sinkhorn_iters: raw.hc_sinkhorn_iters,
        hidden_act: raw.hidden_act,
        swiglu_limit: raw.swiglu_limit,
        rms_norm_eps: raw.rms_norm_eps,
        attention_bias: raw.attention_bias,
        attention_dropout: raw.attention_dropout,
        max_position_embeddings: raw.max_position_embeddings,
        num_nextn_predict_layers: raw.num_nextn_predict_layers,
        use_cache: raw.use_cache,
        tie_word_embeddings: raw.tie_word_embeddings,
        pad_token_id: raw.pad_token_id,
        eos_token_ids: raw.eos_token_id.into_vec(),
    })
}

fn reconcile<T: Copy + PartialEq + std::fmt::Debug>(
    name: &str,
    canonical: Option<T>,
    legacy: Option<T>,
) -> std::result::Result<T, String> {
    if let (Some(canonical), Some(legacy)) = (canonical, legacy)
        && canonical != legacy
    {
        return Err(format!(
            "conflicting {name}: canonical value {canonical:?}, linear_attn_config value {legacy:?}"
        ));
    }
    canonical
        .or(legacy)
        .ok_or_else(|| format!("missing {name}; provide it directly or in linear_attn_config"))
}

fn reconcile_f64(
    name: &str,
    canonical: Option<f64>,
    legacy: Option<f64>,
) -> std::result::Result<f64, String> {
    if let (Some(canonical), Some(legacy)) = (canonical, legacy)
        && canonical.to_bits() != legacy.to_bits()
    {
        return Err(format!(
            "conflicting {name}: canonical value {canonical:?}, linear_attn_config value {legacy:?}"
        ));
    }
    canonical
        .or(legacy)
        .ok_or_else(|| format!("missing {name}; provide it directly or in linear_attn_config"))
}

fn normalize_layer_schedule(
    num_layers: usize,
    explicit: Option<Vec<AttentionKind>>,
    legacy_kda: Option<&[usize]>,
    legacy_full: Option<&[usize]>,
) -> std::result::Result<Vec<AttentionKind>, String> {
    let schedule = match explicit {
        Some(schedule) => schedule,
        None => schedule_from_legacy_lists(num_layers, legacy_kda, legacy_full)?,
    };
    if schedule.len() != num_layers {
        return Err(format!(
            "layer_types has {} entries, expected {num_layers}",
            schedule.len()
        ));
    }
    verify_legacy_layers(
        "kda_layers",
        legacy_kda,
        num_layers,
        &schedule,
        AttentionKind::LinearAttention,
    )?;
    verify_legacy_layers(
        "full_attn_layers",
        legacy_full,
        num_layers,
        &schedule,
        AttentionKind::DeepseekSparseAttention,
    )?;
    Ok(schedule)
}

fn schedule_from_legacy_lists(
    num_layers: usize,
    legacy_kda: Option<&[usize]>,
    legacy_full: Option<&[usize]>,
) -> std::result::Result<Vec<AttentionKind>, String> {
    let kda = legacy_kda.ok_or_else(|| {
        "missing layer_types and linear_attn_config.kda_layers; attention schedule is ambiguous"
            .to_owned()
    })?;
    let full = legacy_full.ok_or_else(|| {
        "missing layer_types and linear_attn_config.full_attn_layers; attention schedule is ambiguous"
            .to_owned()
    })?;
    let mut schedule = vec![None; num_layers];
    for (kind, layers) in [
        (AttentionKind::LinearAttention, kda),
        (AttentionKind::DeepseekSparseAttention, full),
    ] {
        for &layer in layers {
            if layer >= num_layers {
                return Err(format!(
                    "legacy attention schedule contains out-of-range layer {layer}"
                ));
            }
            if schedule[layer].replace(kind).is_some() {
                return Err(format!("legacy attention schedule repeats layer {layer}"));
            }
        }
    }
    schedule
        .into_iter()
        .enumerate()
        .map(|(layer, kind)| {
            kind.ok_or_else(|| format!("legacy attention schedule does not classify layer {layer}"))
        })
        .collect()
}

fn verify_legacy_layers(
    name: &str,
    declared: Option<&[usize]>,
    num_layers: usize,
    schedule: &[AttentionKind],
    expected_kind: AttentionKind,
) -> std::result::Result<(), String> {
    let Some(declared) = declared else {
        return Ok(());
    };
    let mut seen = vec![false; num_layers];
    for &layer in declared {
        if layer >= num_layers {
            return Err(format!("{name} contains out-of-range layer {layer}"));
        }
        if std::mem::replace(&mut seen[layer], true) {
            return Err(format!("{name} contains duplicate layer {layer}"));
        }
    }
    let actual: Vec<_> = schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, kind)| (*kind == expected_kind).then_some(layer))
        .collect();
    if declared != actual {
        return Err(format!(
            "{name} {declared:?} disagrees with layer_types {actual:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GlmConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: GlmTextConfig,
    pub quantization_config: GlmQuantizationConfig,
    pub tie_word_embeddings: bool,
}

impl GlmConfig {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_file(model_dir.as_ref().join("config.json"))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read GLM config {}", path.display()))?;
        let config = serde_json::from_slice::<Self>(&bytes)
            .with_context(|| format!("invalid GLM config {}", path.display()))?;
        config
            .validate()
            .with_context(|| format!("unsupported GLM config {}", path.display()))?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.architectures
                .iter()
                .any(|value| value == GLM5_NEXT_ARCHITECTURE),
            "architectures does not contain {GLM5_NEXT_ARCHITECTURE}"
        );
        anyhow::ensure!(
            self.model_type == GLM5_NEXT_MODEL_TYPE,
            "model_type is {:?}, expected {GLM5_NEXT_MODEL_TYPE}",
            self.model_type
        );
        anyhow::ensure!(
            !self.tie_word_embeddings,
            "root tie_word_embeddings=true is not supported"
        );
        self.text_config.validate()?;
        self.quantization_config.validate()?;
        anyhow::ensure!(
            self.tie_word_embeddings == self.text_config.tie_word_embeddings,
            "root and text_config disagree about tie_word_embeddings"
        );
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn tiny() -> Self {
        Self {
            architectures: vec![GLM5_NEXT_ARCHITECTURE.to_owned()],
            model_type: GLM5_NEXT_MODEL_TYPE.to_owned(),
            text_config: GlmTextConfig::tiny(),
            quantization_config: GlmQuantizationConfig {
                quant_method: "fp8".to_owned(),
                activation_scheme: "dynamic".to_owned(),
                fmt: "e4m3".to_owned(),
                weight_block_size: FP8_WEIGHT_BLOCK_SIZE,
            },
            tie_word_embeddings: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GlmGenerationConfig {
    #[serde(rename = "eos_token_id", deserialize_with = "deserialize_token_ids")]
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
    pub temperature: f64,
    pub top_p: f64,
}

impl GlmGenerationConfig {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_file(model_dir.as_ref().join("generation_config.json"))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read GLM generation config {}", path.display()))?;
        let config = serde_json::from_slice::<Self>(&bytes)
            .with_context(|| format!("invalid GLM generation config {}", path.display()))?;
        config
            .validate()
            .with_context(|| format!("unsupported GLM generation config {}", path.display()))?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.eos_token_ids.is_empty(),
            "eos_token_id must not be empty"
        );
        ensure_finite_non_negative("temperature", self.temperature)?;
        anyhow::ensure!(
            self.temperature == 0.0 || self.temperature.recip().is_finite(),
            "positive temperature is too small to invert without overflow"
        );
        anyhow::ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be finite and in (0, 1]"
        );
        Ok(())
    }

    pub fn validate_for_text(&self, text: &GlmTextConfig) -> Result<()> {
        self.validate()?;
        anyhow::ensure!(
            (self.pad_token_id as usize) < text.vocab_size,
            "generation pad_token_id {} is outside vocab_size {}",
            self.pad_token_id,
            text.vocab_size
        );
        for &token in &self.eos_token_ids {
            anyhow::ensure!(
                (token as usize) < text.vocab_size,
                "generation eos_token_id {token} is outside vocab_size {}",
                text.vocab_size
            );
        }
        anyhow::ensure!(
            self.pad_token_id == text.pad_token_id,
            "generation and text configs disagree about pad_token_id"
        );
        anyhow::ensure!(
            self.eos_token_ids == text.eos_token_ids,
            "generation and text configs disagree about eos_token_id"
        );
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    TokenIds::deserialize(deserializer).map(TokenIds::into_vec)
}

fn ensure_non_zero(name: &str, value: usize) -> Result<()> {
    anyhow::ensure!(value > 0, "{name} must be non-zero");
    Ok(())
}

fn ensure_finite_positive(name: &str, value: f64) -> Result<()> {
    anyhow::ensure!(
        value.is_finite() && value > 0.0,
        "{name} must be finite and positive"
    );
    Ok(())
}

fn ensure_finite_non_negative(name: &str, value: f64) -> Result<()> {
    anyhow::ensure!(
        value.is_finite() && value >= 0.0,
        "{name} must be finite and non-negative"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn small_text_json() -> serde_json::Value {
        serde_json::from_str(
            r#"{
                "model_type":"glm5_next_text","dtype":"bfloat16",
                "vocab_size":128,"hidden_size":16,
                "intermediate_size":32,"moe_intermediate_size":8,
                "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":2,
                "first_k_dense_replace":1,"n_shared_experts":1,"n_routed_experts":4,
                "num_experts_per_tok":2,"routed_scaling_factor":2.5,
                "n_group":1,"topk_group":1,"norm_topk_prob":true,
                "scoring_func":"sigmoid","topk_method":"noaux_tc",
                "moe_router_dtype":"float32","q_lora_rank":8,"kv_lora_rank":4,
                "qk_nope_head_dim":4,"qk_rope_head_dim":0,"qk_head_dim":4,
                "head_dim":0,"v_head_dim":4,"mhc":true,"mla_use_nope":true,
                "layer_types":["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention"],
                "mlp_layer_types":["dense","sparse","sparse","sparse"],
                "indexer_types":["full","full","full","full"],
                "index_topk":8,"index_kpool":2,"index_kpool_always_select_tail":true,
                "index_n_heads":2,"index_head_dim":4,
                "linear_attn_config":{"num_heads":2,"head_dim":4,
                    "short_conv_kernel_size":4,"gate_lower_bound":-5.0,
                    "kda_layers":[0,1,2],"full_attn_layers":[3]},
                "hc_mult":2,"hc_eps":0.000001,"hc_sinkhorn_iters":3,
                "hidden_act":"silu","swiglu_limit":10.0,"rms_norm_eps":0.00001,
                "attention_bias":false,"attention_dropout":0.0,
                "max_position_embeddings":64,"num_nextn_predict_layers":0,
                "use_cache":true,"tie_word_embeddings":false,
                "pad_token_id":0,"eos_token_id":[2]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_legacy_nested_linear_attention_fields() {
        let config: GlmTextConfig = serde_json::from_value(small_text_json()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.linear_num_heads, 2);
        assert_eq!(config.linear_head_dim, 4);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.linear_lower_bound, Some(-5.0));
        assert_eq!(
            config.linear_attention_layers().collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(config.sparse_attention_layers().collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn canonical_linear_fields_may_agree_with_legacy_fields() {
        let mut value = small_text_json();
        value["linear_num_heads"] = json!(2);
        value["linear_head_dim"] = json!(4);
        value["linear_conv_kernel_dim"] = json!(4);
        value["linear_lower_bound"] = json!(-5.0);
        let config: GlmTextConfig = serde_json::from_value(value).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn conflicting_canonical_and_legacy_fields_fail_closed() {
        let mut value = small_text_json();
        value["linear_head_dim"] = json!(8);
        let error = serde_json::from_value::<GlmTextConfig>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting linear_head_dim"));
    }

    #[test]
    fn legacy_layer_lists_must_match_the_explicit_schedule() {
        let mut value = small_text_json();
        value["linear_attn_config"]["full_attn_layers"] = json!([2]);
        let error = serde_json::from_value::<GlmTextConfig>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("full_attn_layers"));
    }

    #[test]
    fn complete_legacy_layer_lists_are_an_explicit_supported_schedule() {
        let mut value = small_text_json();
        value.as_object_mut().unwrap().remove("layer_types");
        let config: GlmTextConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            config.layer_types,
            [
                AttentionKind::LinearAttention,
                AttentionKind::LinearAttention,
                AttentionKind::LinearAttention,
                AttentionKind::DeepseekSparseAttention,
            ]
        );
    }

    #[test]
    fn incomplete_legacy_layer_lists_do_not_synthesize_a_schedule() {
        let mut value = small_text_json();
        value.as_object_mut().unwrap().remove("layer_types");
        value["linear_attn_config"]
            .as_object_mut()
            .unwrap()
            .remove("full_attn_layers");
        let error = serde_json::from_value::<GlmTextConfig>(value)
            .unwrap_err()
            .to_string();
        assert!(error.contains("attention schedule is ambiguous"));
    }

    #[test]
    fn execution_schedules_and_linear_parameters_are_required() {
        for field in [
            "mlp_layer_types",
            "indexer_types",
            "num_nextn_predict_layers",
        ] {
            let mut value = small_text_json();
            value.as_object_mut().unwrap().remove(field);
            let error = serde_json::from_value::<GlmTextConfig>(value)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(field),
                "unexpected error for {field}: {error}"
            );
        }

        for (field, error_field) in [
            ("num_heads", "linear_num_heads"),
            ("head_dim", "linear_head_dim"),
            ("short_conv_kernel_size", "linear_conv_kernel_dim"),
            ("gate_lower_bound", "linear_lower_bound"),
        ] {
            let mut value = small_text_json();
            value["linear_attn_config"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            let error = serde_json::from_value::<GlmTextConfig>(value)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(error_field),
                "unexpected error for {field}: {error}"
            );
        }
    }

    #[test]
    fn execution_architecture_flags_fail_closed() {
        for (field, replacement, expected) in [
            ("model_type", json!("other"), "text_config.model_type"),
            ("mhc", json!(false), "mhc=false"),
            ("mla_use_nope", json!(false), "mla_use_nope=false"),
            ("head_dim", json!(1), "head_dim must be zero"),
            ("qk_head_dim", json!(5), "expected qk_nope_head_dim"),
        ] {
            let mut value = small_text_json();
            value[field] = replacement;
            let error = serde_json::from_value::<GlmTextConfig>(value)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "unexpected error for {field}: {error}"
            );
        }

        for field in [
            "model_type",
            "mhc",
            "mla_use_nope",
            "head_dim",
            "qk_head_dim",
        ] {
            let mut value = small_text_json();
            value.as_object_mut().unwrap().remove(field);
            let error = serde_json::from_value::<GlmTextConfig>(value)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(field),
                "unexpected error for {field}: {error}"
            );
        }
    }

    #[test]
    fn validates_the_official_45_layer_schedule_shape() {
        let mut config = GlmTextConfig::tiny();
        config.num_hidden_layers = 45;
        config.first_k_dense_replace = 3;
        config.layer_types = (0..45)
            .map(|layer| {
                if layer % 4 == 3 {
                    AttentionKind::DeepseekSparseAttention
                } else {
                    AttentionKind::LinearAttention
                }
            })
            .collect();
        config.mlp_layer_types = (0..45)
            .map(|layer| {
                if layer < 3 {
                    MlpKind::Dense
                } else {
                    MlpKind::Sparse
                }
            })
            .collect();
        config.indexer_types = vec![IndexerKind::Full; 45];
        config.validate().unwrap();
        assert_eq!(config.linear_attention_layers().count(), 34);
        assert_eq!(config.sparse_attention_layers().count(), 11);
    }

    #[test]
    fn root_validation_requires_the_checkpoint_fp8_block() {
        let mut config = GlmConfig::tiny();
        config.quantization_config.weight_block_size = [64, 128];
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("weight block"));
    }

    #[test]
    fn validation_rejects_disabling_the_required_decoder_cache() {
        let mut config: GlmTextConfig = serde_json::from_value(small_text_json()).unwrap();
        config.use_cache = false;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("use_cache=false"));
    }

    #[test]
    fn generation_config_accepts_one_or_many_eos_ids() {
        let one: GlmGenerationConfig = serde_json::from_value(json!({
            "eos_token_id": 2, "pad_token_id": 0,
            "temperature": 1.0, "top_p": 0.95
        }))
        .unwrap();
        assert_eq!(one.eos_token_ids, [2]);
        one.validate().unwrap();

        let many: GlmGenerationConfig = serde_json::from_value(json!({
            "eos_token_id": [2, 3], "pad_token_id": 0,
            "temperature": 1.0, "top_p": 0.95
        }))
        .unwrap();
        assert_eq!(many.eos_token_ids, [2, 3]);
        many.validate().unwrap();
    }

    #[test]
    fn generation_sampling_fields_are_required_and_safely_invertible() {
        for field in ["temperature", "top_p"] {
            let mut value = json!({
                "eos_token_id": 2, "pad_token_id": 0,
                "temperature": 1.0, "top_p": 0.95
            });
            value.as_object_mut().unwrap().remove(field);
            let error = serde_json::from_value::<GlmGenerationConfig>(value)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(field),
                "unexpected error for {field}: {error}"
            );
        }

        let config: GlmGenerationConfig = serde_json::from_value(json!({
            "eos_token_id": 2, "pad_token_id": 0,
            "temperature": f64::from_bits(1), "top_p": 0.95
        }))
        .unwrap();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("too small to invert"));
    }
}
