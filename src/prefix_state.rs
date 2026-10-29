//! Pure identity, coverage, and receipts for exact Qwen3.8 prefix snapshots; CPU-only.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::{
    model_spec::{Architecture, Mixer, ModelSpec},
    op_contract::GdnSpec,
};

pub const SCHEMA: &str = "qwen38-prefix-state-e1-v1";
pub const DEFAULT_BLOCK_TOKENS: usize = 256;

pub fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    encode_hex(&h.finalize())
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 15)] as char);
    }
    out
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixIdentity {
    pub schema: String,
    pub model_id: String,
    pub model_revision: String,
    pub tokenizer: String,
    pub model_spec_sha256: String,
    pub hashed_weight_bytes: usize,
    pub attention_kv_dtype: String,
    pub gdn_dtype: String,
    pub layer_layout: String,
    pub rope: String,
    pub tensor_semantics: PrefixTensorSemantics,
}

pub fn validate_transfer_bytes(bytes: usize) -> Result<()> {
    ensure!(bytes > 0, "prefix transfer bytes must be nonzero");
    ensure!(
        bytes.is_multiple_of(16),
        "prefix transfer bytes must be 16-byte aligned"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixLogitsCache {
    valid: bool,
}

impl PrefixLogitsCache {
    pub fn mark_valid(&mut self) {
        self.valid = true;
    }

    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    pub fn read(&self) -> Result<()> {
        ensure!(
            self.valid,
            "decode logits are invalid after prefix restore; run one step or prefill first"
        );
        Ok(())
    }
}

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open identity file {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read identity file {}", path.display()))?;
        if count == 0 {
            break;
        }
        h.update(&buffer[..count]);
    }
    Ok(encode_hex(&h.finalize()))
}

impl PrefixIdentity {
    pub fn new(
        model_id: impl Into<String>,
        model_revision: impl Into<String>,
        tokenizer: impl Into<String>,
        rope: impl Into<String>,
        spec: &ModelSpec,
    ) -> Result<Self> {
        ensure!(
            spec.architecture == Architecture::Qwen35Text,
            "prefix snapshots require Qwen3.8"
        );
        Ok(Self {
            schema: SCHEMA.into(),
            model_id: model_id.into(),
            model_revision: model_revision.into(),
            tokenizer: tokenizer.into(),
            model_spec_sha256: sha256(&canonical(spec)?),
            hashed_weight_bytes: 0,
            attention_kv_dtype: "BF16".into(),
            gdn_dtype: "F32".into(),
            layer_layout: "layer-major natural engine layout".into(),
            rope: rope.into(),
            tensor_semantics: PrefixTensorSemantics::default(),
        })
    }

    /// Builds identity from the SHA-256 computed while the upload bytes were read; never re-reads weight files.
    pub fn from_prefix_cache(
        dir: &Path,
        rope: &str,
        spec: &ModelSpec,
        upload_content_sha256: &str,
        upload_hashed_bytes: usize,
    ) -> Result<Self> {
        let config = dir.join("config.json");
        let index = dir.join("model.safetensors.index.json");
        let tokenizer_json = dir.join("tokenizer.json");
        let tokenizer_config = dir.join("tokenizer_config.json");
        let config_sha = file_sha256(&config)?;
        let index_sha = if index.is_file() {
            file_sha256(&index)?
        } else {
            "single-file-model.safetensors".to_owned()
        };
        for path in [&tokenizer_json, &tokenizer_config] {
            ensure!(
                path.is_file(),
                "prefix-cache identity requires {}",
                path.display()
            );
        }
        let mut tokenizer = Sha256::new();
        for path in [tokenizer_json, tokenizer_config] {
            let digest = file_sha256(&path)?;
            tokenizer.update(
                path.file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .unwrap_or(""),
            );
            tokenizer.update([0]);
            tokenizer.update(digest.as_bytes());
            tokenizer.update([0]);
        }
        let mut revision = Sha256::new();
        for (name, digest) in [
            ("config.json", config_sha.clone()),
            ("model.safetensors.index.json", index_sha),
            ("uploaded-tensor-bytes", upload_content_sha256.to_owned()),
        ] {
            revision.update(name.as_bytes());
            revision.update([0]);
            revision.update(digest.as_bytes());
            revision.update([0]);
        }
        let identity = Self::new(
            format!("config-sha256:{config_sha}"),
            format!("uploaded-bytes-sha256:{:x}", revision.finalize()),
            format!("tokenizer-sha256:{:x}", tokenizer.finalize()),
            rope,
            spec,
        )?;
        ensure!(
            upload_hashed_bytes > 0 && identity.hashed_weight_bytes == 0,
            "invalid prefix-cache upload byte accounting"
        );
        Ok(Self {
            hashed_weight_bytes: upload_hashed_bytes,
            ..identity
        })
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.model_id.is_empty(), "prefix model id is empty");
        ensure!(
            !self.model_revision.is_empty(),
            "prefix model revision is empty"
        );
        ensure!(!self.tokenizer.is_empty(), "prefix tokenizer is empty");
        ensure!(!self.rope.is_empty(), "prefix rope context is empty");
        ensure!(self.schema == SCHEMA, "prefix schema mismatch");
        ensure!(
            crate::kv::KvFormat::from_tag(&self.attention_kv_dtype).is_ok()
                && self.gdn_dtype == "F32",
            "prefix dtype identity mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixTensorSemantics {
    pub attention_k: String,
    pub attention_v: String,
    pub gdn_s: String,
    pub gdn_history: String,
}

impl Default for PrefixTensorSemantics {
    fn default() -> Self {
        Self {
            attention_k: "post-q/k RMSNorm + RoPE".into(),
            attention_v: "attention output projection input state".into(),
            gdn_s: "recurrent state at exact prefix boundary".into(),
            gdn_history: "causal-conv history oldest-to-newest at exact prefix boundary".into(),
        }
    }
}

impl PrefixTensorSemantics {
    fn validated(&self) -> Result<()> {
        ensure!(
            self.attention_k == "post-q/k RMSNorm + RoPE"
                && self.attention_v == "attention output projection input state"
                && self.gdn_s == "recurrent state at exact prefix boundary"
                && self.gdn_history
                    == "causal-conv history oldest-to-newest at exact prefix boundary",
            "prefix tensor semantics mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixRecordKind {
    AttentionKvBlock,
    GdnS,
    GdnHistory,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixRecordKey {
    pub identity: PrefixIdentity,
    pub kind: PrefixRecordKind,
    pub layer: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub valid_tokens: usize,
    pub axes: Vec<String>,
    pub payload_axes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixRecordReceipt {
    pub key: PrefixRecordKey,
    pub key_sha256: String,
    pub bytes: usize,
    pub sha256: String,
    pub file: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixBoundary {
    pub prefix_tokens: usize,
    pub last_token: u32,
    pub last_input_kind: PrefixLastInputKind,
    pub pending_next_token: u32,
    pub token_ids_sha256: String,
    pub rope_context_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixLastInputKind {
    PrefillChunk,
    DecodeToken,
}

fn last_input_tag(kind: PrefixLastInputKind) -> &'static str {
    match kind {
        PrefixLastInputKind::PrefillChunk => "prefill_chunk",
        PrefixLastInputKind::DecodeToken => "decode_token",
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PrefixManifest {
    pub identity: PrefixIdentity,
    pub capacity: usize,
    pub block_tokens: usize,
    pub boundary: PrefixBoundary,
    pub attention_layer_count: usize,
    pub gdn_layer_count: usize,
    pub attention_merkle_root: String,
    pub gdn_boundary_root: String,
    pub state_root: String,
    pub total_state_bytes: usize,
    pub records: Vec<PrefixRecordReceipt>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PrefixSaveReceipt {
    pub prefix_tokens: usize,
    pub state_root: String,
    pub total_state_bytes: usize,
    pub transfer_bytes: usize,
    pub identity_load_ms: f64,
    pub identity_hashed_bytes: usize,
    pub wall_ms: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PrefixRestoreReceipt {
    pub prefix_tokens: usize,
    pub state_root: String,
    pub total_state_bytes: usize,
    pub transfer_bytes: usize,
    pub identity_load_ms: f64,
    pub identity_hashed_bytes: usize,
    pub wall_ms: f64,
}

#[derive(Clone, Debug)]
pub struct PrefixGeometry {
    pub kv_format: crate::kv::KvFormat,
    pub capacity: usize,
    pub prefix_tokens: usize,
    pub block_tokens: usize,
    pub attention_layers: Vec<usize>,
    pub gdn_layers: Vec<usize>,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub gdn_s_bytes: usize,
    pub gdn_history_bytes: usize,
    expected: BTreeMap<PrefixDescriptor, PrefixRecordKey>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct PrefixDescriptor {
    kind: PrefixRecordKind,
    layer: usize,
    block: usize,
}

impl PrefixGeometry {
    pub fn new(
        spec: &ModelSpec,
        capacity: usize,
        prefix_tokens: usize,
        block_tokens: usize,
        identity: &PrefixIdentity,
    ) -> Result<Self> {
        identity.validate()?;
        identity.tensor_semantics.validated()?;
        ensure!(
            spec.architecture == Architecture::Qwen35Text,
            "prefix geometry requires Qwen3.8"
        );
        ensure!(
            capacity > 0 && capacity <= spec.max_position as usize,
            "invalid prefix capacity"
        );
        ensure!(
            prefix_tokens > 0 && prefix_tokens < capacity,
            "prefix must leave a slot for the pending decode token"
        );
        ensure!(block_tokens > 0, "prefix block must be nonempty");
        let kv_format = crate::kv::KvFormat::from_tag(&identity.attention_kv_dtype)?;
        ensure!(
            kv_format == crate::kv::KvFormat::Bf16 || block_tokens.is_multiple_of(128),
            "compact KV prefix block must be a multiple of128"
        );
        let gdn = GdnSpec::from_model(spec)?;
        let s_bytes = gdn.state_bytes()?;
        let history_bytes = gdn.history_bytes()?;
        let mut attention = Vec::new();
        let mut gdn_layers = Vec::new();
        let mut attention_indices = BTreeSet::new();
        let mut gdn_indices = BTreeSet::new();
        for (physical, layer) in spec.layers.iter().enumerate() {
            match layer.mixer {
                Mixer::FullAttention => {
                    ensure!(
                        attention_indices.insert(layer.state_index as usize),
                        "duplicate full-attention state index"
                    );
                    attention.push(physical);
                }
                Mixer::Gdn => {
                    ensure!(
                        gdn_indices.insert(layer.state_index as usize),
                        "duplicate GDN state index"
                    );
                    gdn_layers.push(physical);
                }
            }
        }
        ensure!(!attention.is_empty(), "prefix has no attention layers");
        ensure!(!gdn_layers.is_empty(), "prefix has no GDN layers");
        ensure!(
            attention_indices == (0..attention.len()).collect::<BTreeSet<_>>(),
            "non-contiguous attention state mapping"
        );
        ensure!(
            gdn_indices == (0..gdn_layers.len()).collect::<BTreeSet<_>>(),
            "non-contiguous GDN state mapping"
        );
        let kv_heads = spec.kv_heads as usize;
        let head_dim = spec.head_dim as usize;
        let mut expected = BTreeMap::new();
        for &layer in &attention {
            for block in 0..prefix_tokens.div_ceil(block_tokens) {
                let start = block * block_tokens;
                let end = prefix_tokens.min(start + block_tokens);
                let key = PrefixRecordKey {
                    identity: identity.clone(),
                    kind: PrefixRecordKind::AttentionKvBlock,
                    layer,
                    token_start: start,
                    token_end: end,
                    valid_tokens: end - start,
                    axes: vec!["kv_head".into(), "token_in_block".into(), "channel".into()],
                    payload_axes: if kv_format == crate::kv::KvFormat::Bf16 {
                        vec!["K".into(), "V".into()]
                    } else {
                        [
                            "K_u8",
                            "V_u8",
                            "K_stats_f32",
                            "V_stats_f32",
                            "K_live_bf16",
                            "V_live_bf16",
                        ]
                        .map(str::to_owned)
                        .to_vec()
                    },
                };
                expected.insert(
                    PrefixDescriptor {
                        kind: PrefixRecordKind::AttentionKvBlock,
                        layer,
                        block,
                    },
                    key,
                );
            }
        }
        for &layer in &gdn_layers {
            let s = PrefixRecordKey {
                identity: identity.clone(),
                kind: PrefixRecordKind::GdnS,
                layer,
                token_start: prefix_tokens,
                token_end: prefix_tokens,
                valid_tokens: 0,
                axes: vec!["head".into(), "key".into(), "value".into()],
                payload_axes: vec!["F32".into()],
            };
            let history = PrefixRecordKey {
                identity: identity.clone(),
                kind: PrefixRecordKind::GdnHistory,
                layer,
                token_start: prefix_tokens,
                token_end: prefix_tokens,
                valid_tokens: 0,
                axes: vec!["channel".into(), "oldest_to_newest".into()],
                payload_axes: vec!["F32".into()],
            };
            expected.insert(
                PrefixDescriptor {
                    kind: PrefixRecordKind::GdnS,
                    layer,
                    block: 0,
                },
                s,
            );
            expected.insert(
                PrefixDescriptor {
                    kind: PrefixRecordKind::GdnHistory,
                    layer,
                    block: 0,
                },
                history,
            );
        }
        Ok(Self {
            kv_format,
            capacity,
            prefix_tokens,
            block_tokens,
            attention_layers: attention,
            gdn_layers,
            kv_heads,
            head_dim,
            gdn_s_bytes: usize::try_from(s_bytes)?,
            gdn_history_bytes: usize::try_from(history_bytes)?,
            expected,
        })
    }

    pub fn attention_block_bytes(&self, valid_tokens: usize) -> Result<usize> {
        self.attention_block_bytes_at(0, valid_tokens)
    }
    pub fn attention_block_bytes_at(&self, start: usize, valid_tokens: usize) -> Result<usize> {
        ensure!(valid_tokens > 0 && valid_tokens <= self.block_tokens);
        self.kv_format.block_bytes(
            start,
            valid_tokens,
            self.prefix_tokens,
            self.kv_heads,
            self.head_dim,
        )
    }

    fn descriptor_bytes(&self, descriptor: PrefixDescriptor) -> Result<usize> {
        match descriptor.kind {
            PrefixRecordKind::AttentionKvBlock => {
                let valid = self
                    .prefix_tokens
                    .checked_sub(descriptor.block * self.block_tokens)
                    .context("bad attention block")?
                    .min(self.block_tokens);
                self.attention_block_bytes_at(descriptor.block * self.block_tokens, valid)
            }
            PrefixRecordKind::GdnS => Ok(self.gdn_s_bytes),
            PrefixRecordKind::GdnHistory => Ok(self.gdn_history_bytes),
        }
    }

    fn expected_keys(&self) -> Result<BTreeMap<String, usize>> {
        self.expected
            .iter()
            .map(|(descriptor, key)| {
                Ok((
                    sha256(&canonical(key)?),
                    self.descriptor_bytes(*descriptor)?,
                ))
            })
            .collect()
    }

    pub fn total_state_bytes(&self) -> Result<usize> {
        self.expected.keys().try_fold(0usize, |sum, descriptor| {
            sum.checked_add(self.descriptor_bytes(*descriptor)?)
                .context("prefix state byte overflow")
        })
    }

    pub fn verify_manifest(&self, manifest: &PrefixManifest) -> Result<()> {
        ensure!(
            manifest.capacity == self.capacity
                && manifest.block_tokens == self.block_tokens
                && manifest.boundary.prefix_tokens == self.prefix_tokens
                && manifest.attention_layer_count == self.attention_layers.len()
                && manifest.gdn_layer_count == self.gdn_layers.len(),
            "prefix manifest geometry mismatch"
        );
        let expected = self.expected_keys()?;
        let got: BTreeSet<_> = manifest
            .records
            .iter()
            .map(|r| r.key_sha256.clone())
            .collect();
        ensure!(
            got == expected.keys().cloned().collect::<BTreeSet<_>>(),
            "prefix manifest coverage mismatch"
        );
        for receipt in &manifest.records {
            let expected_bytes = expected
                .get(&receipt.key_sha256)
                .context("prefix manifest contains an unknown record")?;
            ensure!(
                receipt.bytes == *expected_bytes,
                "prefix receipt byte extent mismatch for {}",
                receipt.key_sha256
            );
        }
        ensure!(
            manifest.total_state_bytes == self.total_state_bytes()?,
            "prefix manifest byte total mismatch"
        );
        Ok(())
    }
}

fn record_path(kind: PrefixRecordKind, layer: usize, block: usize) -> String {
    match kind {
        PrefixRecordKind::AttentionKvBlock => {
            format!("attention/L{layer:02}/B{block:05}.bin")
        }
        PrefixRecordKind::GdnS => format!("gdn/L{layer:02}/S.f32"),
        PrefixRecordKind::GdnHistory => format!("gdn/L{layer:02}/history.f32"),
    }
}

#[derive(Debug)]
pub struct PrefixBuilder {
    pub geometry: PrefixGeometry,
    boundary: PrefixBoundary,
    records: BTreeMap<String, (PrefixRecordReceipt, Vec<u8>)>,
}

impl PrefixBuilder {
    pub fn new(geometry: PrefixGeometry, boundary: PrefixBoundary) -> Result<Self> {
        geometry.identity_check()?;
        ensure!(
            boundary.prefix_tokens == geometry.prefix_tokens,
            "boundary/geometry prefix mismatch"
        );
        ensure!(boundary.last_token < u32::MAX, "invalid last token");
        Ok(Self {
            geometry,
            boundary,
            records: BTreeMap::new(),
        })
    }

    pub fn add(
        &mut self,
        kind: PrefixRecordKind,
        layer: usize,
        block: usize,
        bytes: Vec<u8>,
    ) -> Result<()> {
        let descriptor = PrefixDescriptor { kind, layer, block };
        let key = self
            .geometry
            .expected
            .get(&descriptor)
            .cloned()
            .with_context(|| format!("unexpected prefix record {descriptor:?}"))?;
        let expected_bytes = self.geometry.descriptor_bytes(descriptor)?;
        ensure!(
            bytes.len() == expected_bytes,
            "prefix record byte count mismatch: got {}, expected {expected_bytes}",
            bytes.len()
        );
        let key_json = canonical(&key)?;
        let key_sha = sha256(&key_json);
        ensure!(
            self.records
                .insert(
                    key_sha.clone(),
                    (
                        PrefixRecordReceipt {
                            key,
                            key_sha256: key_sha,
                            bytes: bytes.len(),
                            sha256: sha256(&bytes),
                            file: record_path(kind, layer, block),
                        },
                        bytes,
                    ),
                )
                .is_none(),
            "duplicate prefix record"
        );
        Ok(())
    }

    pub fn finish(self) -> Result<PrefixSnapshot> {
        let expected = self.geometry.expected_keys()?;
        let got: BTreeSet<_> = self.records.keys().cloned().collect();
        ensure!(
            got == expected.keys().cloned().collect::<BTreeSet<_>>(),
            "prefix coverage mismatch: got {} records, expected {}",
            got.len(),
            expected.len()
        );
        for (key_sha, receipt_bytes) in &self.records {
            ensure!(
                receipt_bytes.0.bytes == expected[key_sha]
                    && receipt_bytes.1.len() == expected[key_sha],
                "prefix record extent mismatch for {key_sha}"
            );
        }
        let mut records = Vec::with_capacity(self.records.len());
        let mut payloads = Vec::with_capacity(self.records.len());
        let mut attention_hashes = Vec::new();
        let mut gdn_hashes = Vec::new();
        let mut total = 0usize;
        for (_, (receipt, payload)) in self.records {
            ensure!(
                receipt.sha256 == sha256(&payload),
                "prefix payload SHA mismatch"
            );
            total += payload.len();
            let bundle = format!("{}{}", receipt.key_sha256, receipt.sha256);
            match receipt.key.kind {
                PrefixRecordKind::AttentionKvBlock => {
                    attention_hashes.push(sha256(bundle.as_bytes()))
                }
                PrefixRecordKind::GdnS | PrefixRecordKind::GdnHistory => {
                    gdn_hashes.push(sha256(bundle.as_bytes()))
                }
            }
            records.push(receipt);
            payloads.push(payload);
        }
        let attention_root = merkle(&attention_hashes);
        let gdn_root = merkle(&gdn_hashes);
        let state_root = sha256(
            format!(
                "capacity={};block={};prefix={};token_ids={};last_input={};rope={};pending={};root={}",
                self.geometry.capacity,
                self.geometry.block_tokens,
                self.boundary.prefix_tokens,
                self.boundary.token_ids_sha256,
                last_input_tag(self.boundary.last_input_kind),
                self.boundary.rope_context_sha256,
                self.boundary.pending_next_token,
                merkle(&[
                    sha256(attention_root.as_bytes()),
                    sha256(gdn_root.as_bytes())
                ])
            )
            .as_bytes(),
        );
        let manifest = PrefixManifest {
            identity: self
                .geometry
                .expected
                .values()
                .next()
                .context("empty prefix geometry")?
                .identity
                .clone(),
            capacity: self.geometry.capacity,
            block_tokens: self.geometry.block_tokens,
            boundary: self.boundary,
            attention_layer_count: self.geometry.attention_layers.len(),
            gdn_layer_count: self.geometry.gdn_layers.len(),
            attention_merkle_root: attention_root,
            gdn_boundary_root: gdn_root,
            state_root,
            total_state_bytes: total,
            records,
        };
        Ok(PrefixSnapshot { manifest, payloads })
    }
}

fn merkle(leaves: &[String]) -> String {
    let mut level = leaves.to_vec();
    if level.is_empty() {
        return sha256(b"empty");
    }
    while level.len() > 1 {
        if level.len() & 1 == 1 {
            let last = level.last().cloned().expect("nonempty");
            level.push(last);
        }
        level = level
            .chunks(2)
            .map(|pair| sha256(format!("{}{}", pair[0], pair[1]).as_bytes()))
            .collect();
    }
    level[0].clone()
}

#[derive(Clone, Debug)]
pub struct PrefixSnapshot {
    pub manifest: PrefixManifest,
    payloads: Vec<Vec<u8>>,
}

impl PrefixSnapshot {
    fn validate_manifest_only(&self) -> Result<()> {
        self.manifest.identity.validate()?;
        self.manifest.identity.tensor_semantics.validated()?;
        ensure!(!self.manifest.records.is_empty(), "empty prefix snapshot");
        ensure!(
            self.manifest.block_tokens > 0,
            "prefix snapshot block must be nonempty"
        );
        ensure!(
            self.manifest.boundary.prefix_tokens > 0
                && self.manifest.boundary.prefix_tokens < self.manifest.capacity,
            "prefix snapshot must leave a slot for the pending decode token"
        );
        let mut seen = BTreeSet::new();
        let mut total = 0usize;
        for receipt in &self.manifest.records {
            ensure!(
                seen.insert(receipt.key_sha256.clone()),
                "duplicate prefix receipt"
            );
            ensure!(
                receipt.key.identity == self.manifest.identity,
                "prefix record identity differs from manifest"
            );
            let block = if receipt.key.kind == PrefixRecordKind::AttentionKvBlock {
                receipt.key.token_start / self.manifest.block_tokens
            } else {
                0
            };
            ensure!(
                receipt.file == record_path(receipt.key.kind, receipt.key.layer, block),
                "prefix receipt path mismatch"
            );
            ensure!(
                receipt.key_sha256 == sha256(&canonical(&receipt.key)?),
                "prefix key SHA mismatch"
            );
            total = total
                .checked_add(receipt.bytes)
                .context("prefix receipt byte total overflow")?;
        }
        ensure!(
            total == self.manifest.total_state_bytes,
            "prefix total byte mismatch"
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_manifest_only()?;
        ensure!(
            self.manifest.records.len() == self.payloads.len(),
            "prefix receipt/payload count mismatch"
        );
        let mut attention = Vec::new();
        let mut gdn = Vec::new();
        let mut total = 0usize;
        for (receipt, payload) in self.manifest.records.iter().zip(&self.payloads) {
            ensure!(
                receipt.bytes == payload.len(),
                "prefix payload size mismatch"
            );
            ensure!(
                receipt.sha256 == sha256(payload),
                "prefix payload SHA mismatch"
            );
            let bundle = format!("{}{}", receipt.key_sha256, receipt.sha256);
            match receipt.key.kind {
                PrefixRecordKind::AttentionKvBlock => attention.push(sha256(bundle.as_bytes())),
                PrefixRecordKind::GdnS | PrefixRecordKind::GdnHistory => {
                    gdn.push(sha256(bundle.as_bytes()))
                }
            }
            total += payload.len();
        }
        ensure!(
            total == self.manifest.total_state_bytes,
            "prefix total byte mismatch"
        );
        ensure!(
            self.manifest.attention_merkle_root == merkle(&attention),
            "attention Merkle root mismatch"
        );
        ensure!(
            self.manifest.gdn_boundary_root == merkle(&gdn),
            "GDN boundary root mismatch"
        );
        let state_root = sha256(
            format!(
                "capacity={};block={};prefix={};token_ids={};last_input={};rope={};pending={};root={}",
                self.manifest.capacity,
                self.manifest.block_tokens,
                self.manifest.boundary.prefix_tokens,
                self.manifest.boundary.token_ids_sha256,
                last_input_tag(self.manifest.boundary.last_input_kind),
                self.manifest.boundary.rope_context_sha256,
                self.manifest.boundary.pending_next_token,
                merkle(&[
                    sha256(self.manifest.attention_merkle_root.as_bytes()),
                    sha256(self.manifest.gdn_boundary_root.as_bytes())
                ])
            )
            .as_bytes(),
        );
        ensure!(
            self.manifest.state_root == state_root,
            "prefix state root mismatch"
        );
        Ok(())
    }

    pub fn verify_prefix(&self, prefix_ids: &[u32]) -> Result<()> {
        self.validate()?;
        ensure!(
            prefix_ids.len() == self.manifest.boundary.prefix_tokens,
            "prefix token count mismatch"
        );
        let mut encoded = Vec::with_capacity(prefix_ids.len() * 4);
        for token in prefix_ids {
            encoded.extend_from_slice(&token.to_le_bytes());
        }
        ensure!(
            self.manifest.boundary.token_ids_sha256 == sha256(&encoded),
            "prefix token identity mismatch"
        );
        ensure!(
            prefix_ids.last().copied() == Some(self.manifest.boundary.last_token),
            "prefix last-token mismatch"
        );
        Ok(())
    }

    pub fn verify_prefix_and_pending(
        &self,
        prefix_ids: &[u32],
        pending_next_token: u32,
    ) -> Result<()> {
        self.verify_prefix(prefix_ids)?;
        ensure!(
            self.manifest.boundary.pending_next_token == pending_next_token,
            "prefix pending next-token mismatch"
        );
        Ok(())
    }

    pub fn verify_prefix_and_pending_within_vocab(
        &self,
        prefix_ids: &[u32],
        pending_next_token: u32,
        vocab: usize,
    ) -> Result<()> {
        self.verify_prefix_and_pending(prefix_ids, pending_next_token)?;
        ensure!(vocab > 0, "empty vocabulary");
        ensure!(
            prefix_ids.iter().all(|token| (*token as usize) < vocab),
            "prefix token outside vocabulary"
        );
        ensure!(
            (pending_next_token as usize) < vocab,
            "pending prefix token outside vocabulary"
        );
        Ok(())
    }

    pub fn records(&self) -> impl Iterator<Item = (&PrefixRecordReceipt, &[u8])> {
        self.manifest
            .records
            .iter()
            .zip(self.payloads.iter())
            .map(|(receipt, payload)| (receipt, payload.as_slice()))
    }

    pub fn write_package(&self, root: &Path) -> Result<()> {
        self.validate()?;
        ensure!(!root.exists(), "prefix package root already exists");
        std::fs::create_dir_all(root)?;
        let root = root.canonicalize()?;
        for (receipt, payload) in self.records() {
            let relative = Path::new(&receipt.file);
            ensure!(
                relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "unsafe prefix package path {}",
                receipt.file
            );
            let path = root.join(relative);
            let parent = path.parent().context("prefix package path has no parent")?;
            std::fs::create_dir_all(parent)?;
            std::fs::write(&path, payload)?;
        }
        let manifest = serde_json::to_vec_pretty(&self.manifest)?;
        std::fs::write(root.join("MANIFEST.json"), manifest)?;
        Ok(())
    }

    pub fn read_package(root: &Path) -> Result<Self> {
        let root = root.canonicalize().context("prefix package root missing")?;
        let manifest: PrefixManifest =
            serde_json::from_slice(&std::fs::read(root.join("MANIFEST.json"))?)?;
        let manifest_only = Self {
            manifest: manifest.clone(),
            payloads: Vec::new(),
        };
        manifest_only.validate_manifest_only()?;
        let mut payloads = Vec::with_capacity(manifest.records.len());
        for receipt in &manifest.records {
            let relative = Path::new(&receipt.file);
            ensure!(
                relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "unsafe prefix package path {}",
                receipt.file
            );
            let path = root.join(relative);
            let canonical = path
                .canonicalize()
                .with_context(|| format!("prefix package payload missing: {}", path.display()))?;
            ensure!(
                canonical.starts_with(&root),
                "prefix package path escapes root: {}",
                receipt.file
            );
            ensure!(
                std::fs::metadata(&canonical)
                    .with_context(|| {
                        format!("stat prefix package payload: {}", canonical.display())
                    })?
                    .len()
                    == u64::try_from(receipt.bytes)
                        .context("prefix receipt byte count overflow")?,
                "prefix package payload length mismatch: {}",
                receipt.file
            );
            payloads.push(std::fs::read(&canonical)?);
        }
        let snapshot = Self { manifest, payloads };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn manifest_path(root: &Path) -> PathBuf {
        root.join("MANIFEST.json")
    }
}

pub fn token_ids_sha256(prefix_ids: &[u32]) -> Result<String> {
    ensure!(!prefix_ids.is_empty(), "prefix token ids are empty");
    let mut bytes = Vec::with_capacity(prefix_ids.len() * 4);
    for token in prefix_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(sha256(&bytes))
}

impl PrefixGeometry {
    fn identity_check(&self) -> Result<()> {
        if let Some(key) = self.expected.values().next() {
            key.identity.validate()?;
            key.identity.tensor_semantics.validated()?;
        } else {
            bail!("prefix geometry has no records");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_spec::{GdnConfig, LayerPlan};

    fn spec() -> ModelSpec {
        let mut layers = Vec::new();
        let mut attention_index = 0;
        let mut gdn_index = 0;
        for i in 0..4 {
            let attention = i % 2 == 1;
            layers.push(LayerPlan {
                mixer: if attention {
                    Mixer::FullAttention
                } else {
                    Mixer::Gdn
                },
                rope: attention,
                sliding_window: None,
                ffn: crate::model_spec::FfnKind::Dense,
                state_index: if attention {
                    let x = attention_index;
                    attention_index += 1;
                    x
                } else {
                    let x = gdn_index;
                    gdn_index += 1;
                    x
                },
            });
        }
        ModelSpec {
            abi_version: 1,
            architecture: Architecture::Qwen35Text,
            expert_format: None,
            hidden: 5120,
            intermediate: 17408,
            vocab: 248320,
            q_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            q_width: 6144,
            max_position: 1024,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            rotary_dim: 128,
            linear_format: crate::model_spec::LinearFormat::fp8_block(
                [128, 128],
                safetensors::Dtype::BF16,
            )
            .unwrap(),
            eos: vec![1],
            layers,
            moe: None,
            gdn: Some(GdnConfig {
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
            }),
            glm5_next: None,
            deepseek_v41: None,
            vision: None,
        }
    }

    fn identity(spec: &ModelSpec) -> PrefixIdentity {
        let mut identity =
            PrefixIdentity::new("test-model", "rev1", "test-tokenizer", "rope-v1", spec).unwrap();
        identity.hashed_weight_bytes = 1;
        identity
    }

    #[test]
    fn compact_prefix_covers_typed_statistics_and_live_bf16() -> Result<()> {
        let mut spec = spec();
        spec.max_position = 4096;
        let format = crate::kv::KvFormat::Affine8;
        let mut identity = identity(&spec);
        identity.attention_kv_dtype = format.tag().into();
        let g = PrefixGeometry::new(&spec, 4096, 2305, 256, &identity)?;
        let mut builder = PrefixBuilder::new(g, boundary(&vec![7; 2305]))?;
        for layer in [1usize, 3] {
            for block in 0..2305usize.div_ceil(256) {
                let start = block * 256;
                let valid = (2305 - start).min(256);
                let n = builder.geometry.attention_block_bytes_at(start, valid)?;
                builder.add(
                    PrefixRecordKind::AttentionKvBlock,
                    layer,
                    block,
                    vec![19; n],
                )?;
            }
            builder.add(
                PrefixRecordKind::GdnS,
                layer - 1,
                0,
                vec![7; builder.geometry.gdn_s_bytes],
            )?;
            builder.add(
                PrefixRecordKind::GdnHistory,
                layer - 1,
                0,
                vec![11; builder.geometry.gdn_history_bytes],
            )?;
        }
        let snapshot = builder.finish()?;
        snapshot.verify_prefix(&vec![7; 2305])?;
        let g = PrefixGeometry::new(&spec, 4096, 2305, 256, &identity)?;
        g.verify_manifest(&snapshot.manifest)?;
        assert_eq!(g.attention_block_bytes_at(0, 256)?, 811008);
        assert!(PrefixGeometry::new(&spec, 4096, 2305, 127, &identity).is_err());
        let mut foreign = identity.clone();
        foreign.attention_kv_dtype = crate::kv::KvFormat::Bf16.tag().into();
        assert!(
            PrefixGeometry::new(&spec, 4096, 2305, 256, &foreign)?
                .verify_manifest(&snapshot.manifest)
                .is_err()
        );
        Ok(())
    }

    fn boundary(ids: &[u32]) -> PrefixBoundary {
        PrefixBoundary {
            prefix_tokens: ids.len(),
            last_token: *ids.last().unwrap(),
            last_input_kind: PrefixLastInputKind::PrefillChunk,
            pending_next_token: 77,
            token_ids_sha256: token_ids_sha256(ids).unwrap(),
            rope_context_sha256: sha256(b"text-rope"),
        }
    }

    fn geometry(spec: &ModelSpec, prefix: usize, block: usize) -> PrefixGeometry {
        PrefixGeometry::new(spec, 128, prefix, block, &identity(spec)).unwrap()
    }

    fn full_package() -> PrefixSnapshot {
        let spec = spec();
        let ids = [7u32, 8, 9, 10];
        let geometry = geometry(&spec, ids.len(), 2);
        let mut builder = PrefixBuilder::new(geometry, boundary(&ids)).unwrap();
        for layer in [1usize, 3] {
            for block in 0..2 {
                let valid = ids.len().min((block + 1) * 2) - block * 2;
                let bytes = vec![
                    u8::try_from(block + layer).unwrap();
                    builder.geometry.attention_block_bytes(valid).unwrap()
                ];
                builder
                    .add(PrefixRecordKind::AttentionKvBlock, layer, block, bytes)
                    .unwrap();
            }
        }
        for layer in [0usize, 2] {
            builder
                .add(
                    PrefixRecordKind::GdnS,
                    layer,
                    0,
                    vec![7; builder.geometry.gdn_s_bytes],
                )
                .unwrap();
            builder
                .add(
                    PrefixRecordKind::GdnHistory,
                    layer,
                    0,
                    vec![8; builder.geometry.gdn_history_bytes],
                )
                .unwrap();
        }
        builder.finish().unwrap()
    }

    #[test]
    fn decode_committed_boundary_is_representable_and_verified() -> Result<()> {
        let spec = spec();
        let ids = [7u32, 8, 9, 10];
        let mut boundary = boundary(&ids);
        boundary.last_input_kind = PrefixLastInputKind::DecodeToken;
        let geometry = geometry(&spec, ids.len(), 2);
        let mut builder = PrefixBuilder::new(geometry, boundary)?;
        for layer in [1usize, 3] {
            for block in 0..2 {
                let valid = ids.len().min((block + 1) * 2) - block * 2;
                builder.add(
                    PrefixRecordKind::AttentionKvBlock,
                    layer,
                    block,
                    vec![9; builder.geometry.attention_block_bytes(valid)?],
                )?;
            }
        }
        for layer in [0usize, 2] {
            builder.add(
                PrefixRecordKind::GdnS,
                layer,
                0,
                vec![9; builder.geometry.gdn_s_bytes],
            )?;
            builder.add(
                PrefixRecordKind::GdnHistory,
                layer,
                0,
                vec![9; builder.geometry.gdn_history_bytes],
            )?;
        }
        let package = builder.finish()?;
        assert_eq!(
            package.manifest.boundary.last_input_kind,
            PrefixLastInputKind::DecodeToken
        );
        package.verify_prefix_and_pending_within_vocab(&ids, 77, 100)?;
        Ok(())
    }

    #[test]
    fn geometry_counts_partial_blocks_and_exact_bytes() {
        let spec = spec();
        let geometry = geometry(&spec, 5, 2);
        assert_eq!(geometry.attention_layers, vec![1, 3]);
        assert_eq!(geometry.gdn_layers, vec![0, 2]);
        assert_eq!(
            geometry.attention_block_bytes(2).unwrap(),
            2 * 4 * 2 * 256 * 2
        );
        assert_eq!(
            geometry.total_state_bytes().unwrap(),
            2 * (2 * geometry.attention_block_bytes(2).unwrap()
                + geometry.attention_block_bytes(1).unwrap())
                + 2 * (geometry.gdn_s_bytes + geometry.gdn_history_bytes)
        );
    }

    #[test]
    fn geometry_rejects_empty_block_and_full_capacity() {
        let spec = spec();
        assert!(PrefixGeometry::new(&spec, 128, 4, 0, &identity(&spec)).is_err());
        assert!(PrefixGeometry::new(&spec, 128, 128, 2, &identity(&spec)).is_err());
        let mut zero_block = full_package();
        zero_block.manifest.block_tokens = 0;
        assert!(zero_block.validate().is_err());
    }

    #[test]
    fn prefix_cache_identity_uses_exact_upload_digest_and_single_file_layout() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "flyingfish-prefix-cache-identity-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("config.json"), b"config")?;
        std::fs::write(root.join("model.safetensors"), b"weights")?;
        std::fs::write(root.join("tokenizer.json"), b"tokenizer")?;
        std::fs::write(root.join("tokenizer_config.json"), b"tokenizer-config")?;
        let spec = spec();
        let first = PrefixIdentity::from_prefix_cache(&root, "rope", &spec, "digest-a", 3)?;
        first.validate()?;
        let second = PrefixIdentity::from_prefix_cache(&root, "rope", &spec, "digest-b", 3)?;
        assert_ne!(first, second);
        assert_eq!(second.hashed_weight_bytes, 3);
        assert!(!root.join("model.safetensors.index.json").exists());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn transfer_bytes_must_be_nonzero_and_aligned() {
        assert!(validate_transfer_bytes(16).is_ok());
        assert!(validate_transfer_bytes(0).is_err());
        assert!(validate_transfer_bytes(17).is_err());
    }

    #[test]
    fn restored_logits_are_invalid_until_next_forward() {
        let mut cache = PrefixLogitsCache::default();
        assert!(cache.read().is_err());
        cache.mark_valid();
        assert!(cache.read().is_ok());
        cache.invalidate();
        let error = cache.read().unwrap_err().to_string();
        assert!(error.contains("decode logits are invalid"), "{error}");
    }

    #[test]
    fn complete_package_and_prefix_identity_pass() {
        let package = full_package();
        package.validate().unwrap();
        package.verify_prefix(&[7, 8, 9, 10]).unwrap();
        package
            .verify_prefix_and_pending(&[7, 8, 9, 10], 77)
            .unwrap();
        assert!(
            package
                .verify_prefix_and_pending(&[7, 8, 9, 10], 78)
                .is_err()
        );
        assert!(package.verify_prefix(&[7, 8, 9, 11]).is_err());
        assert!(package.verify_prefix(&[7, 8, 10, 10]).is_err());
        let root = std::env::temp_dir().join(format!(
            "flyingfish-prefix-state-{}-{}",
            std::process::id(),
            line!()
        ));
        package.write_package(&root).unwrap();
        let loaded = PrefixSnapshot::read_package(&root).unwrap();
        assert_eq!(loaded.manifest, package.manifest);
        loaded.verify_prefix(&[7, 8, 9, 10]).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn same_shape_foreign_identity_and_wrong_extent_fail_before_restore() -> Result<()> {
        let spec = spec();
        let package = full_package();
        let mut foreign_spec = spec.clone();
        foreign_spec.hidden += 1;
        let foreign = identity(&foreign_spec);
        assert_ne!(package.manifest.identity, foreign);
        assert!(
            PrefixGeometry::new(
                &spec,
                package.manifest.capacity,
                package.manifest.boundary.prefix_tokens,
                package.manifest.block_tokens,
                &foreign,
            )?
            .verify_manifest(&package.manifest)
            .is_err()
        );

        let mut wrong_extent = package.clone();
        wrong_extent.manifest.records[0].bytes += 1;
        let geometry = geometry(
            &spec,
            wrong_extent.manifest.boundary.prefix_tokens,
            wrong_extent.manifest.block_tokens,
        );
        let err = geometry
            .verify_manifest(&wrong_extent.manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains("byte extent"), "{err}");
        Ok(())
    }

    #[test]
    fn boundary_rejects_wrong_last_id_pending_or_stale_sha() {
        let package = full_package();
        let mut wrong_last = package.clone();
        wrong_last.manifest.boundary.last_token = 11;
        assert!(wrong_last.verify_prefix(&[7, 8, 9, 10]).is_err());

        let mut wrong_pending = package.clone();
        wrong_pending.manifest.boundary.pending_next_token = 78;
        assert!(wrong_pending.validate().is_err());
        assert!(
            wrong_pending
                .verify_prefix_and_pending(&[7, 8, 9, 10], 78)
                .is_err()
        );

        let mut stale_sha = package.clone();
        stale_sha.manifest.boundary.token_ids_sha256 = sha256(b"stale-prefix");
        assert!(stale_sha.verify_prefix(&[7, 8, 9, 10]).is_err());
    }

    #[test]
    fn out_of_vocabulary_prefix_or_pending_is_rejected_before_restore() {
        let package = full_package();
        assert!(
            package
                .verify_prefix_and_pending_within_vocab(&[7, 8, 9, 10], 77, 100)
                .is_ok()
        );
        assert!(
            package
                .verify_prefix_and_pending_within_vocab(&[7, 8, 9, 100], 77, 100)
                .is_err()
        );
        assert!(
            package
                .verify_prefix_and_pending_within_vocab(&[7, 8, 9, 10], 100, 100)
                .is_err()
        );
    }

    #[test]
    fn package_write_and_read_validate_before_io_effects() -> Result<()> {
        let package = full_package();
        let mut invalid = package.clone();
        invalid.payloads[0][0] ^= 1;
        let absent = std::env::temp_dir().join(format!(
            "flyingfish-prefix-invalid-write-{}-{}",
            std::process::id(),
            line!()
        ));
        assert!(invalid.write_package(&absent).is_err());
        assert!(!absent.exists());

        let root = std::env::temp_dir().join(format!(
            "flyingfish-prefix-length-read-{}-{}",
            std::process::id(),
            line!()
        ));
        package.write_package(&root)?;
        let payload = root.join(&package.manifest.records[0].file);
        let mut bytes = std::fs::read(&payload)?;
        bytes.pop();
        std::fs::write(&payload, bytes)?;
        let err = PrefixSnapshot::read_package(&root).unwrap_err().to_string();
        assert!(err.contains("length mismatch"), "{err}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn record_identity_must_match_manifest_identity() {
        let mut package = full_package();
        package.manifest.records[0]
            .key
            .identity
            .model_id
            .push_str("-foreign");
        assert!(package.validate().is_err());
    }

    #[test]
    fn missing_gdn_or_corrupt_payload_fails_before_restore() {
        let spec = spec();
        let ids = [7u32, 8, 9, 10];
        let geometry = geometry(&spec, ids.len(), 2);
        let mut builder = PrefixBuilder::new(geometry, boundary(&ids)).unwrap();
        for layer in [1usize, 3] {
            for block in 0..2 {
                let valid = ids.len().min((block + 1) * 2) - block * 2;
                builder
                    .add(
                        PrefixRecordKind::AttentionKvBlock,
                        layer,
                        block,
                        vec![1; builder.geometry.attention_block_bytes(valid).unwrap()],
                    )
                    .unwrap();
            }
        }
        for layer in [0usize, 2] {
            builder
                .add(
                    PrefixRecordKind::GdnS,
                    layer,
                    0,
                    vec![1; builder.geometry.gdn_s_bytes],
                )
                .unwrap();
        }
        let err = builder.finish().unwrap_err().to_string();
        assert!(err.contains("coverage mismatch"), "{err}");

        let mut package = full_package();
        package.payloads[0][0] ^= 1;
        assert!(package.validate().is_err());
    }
}
