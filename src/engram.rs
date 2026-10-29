//! Engram n-gram hashing and native table-row service.
use crate::checkpoint::{TensorEntry, TensorManifest};
use crate::experts::disk::{AlignedBytes, SpanReader};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const ENGRAM_HEAD_DIM: usize = 256;
pub const ENGRAM_SCALES: usize = 8;
pub const ENGRAM_NATIVE_ROW_BYTES: usize = ENGRAM_HEAD_DIM + ENGRAM_SCALES;

const HASH_SCHEMA: u32 = 1;
const DEAD_TOKEN: i64 = -1;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngramHashSetup {
    pub schema: u32,
    pub layer_ids: Vec<u32>,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_vocab: u32,
    pub compressed_vocab: u32,
    pub pad_id: u32,
    pub token_map: Vec<u32>,
    pub multipliers: Vec<Vec<i64>>,
    pub primes: Vec<Vec<Vec<u64>>>,
    pub samples: Vec<EngramHashSample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EngramHashSample {
    pub input_ids: Vec<u32>,
    pub start_position: u32,
    pub text_mask: Option<Vec<bool>>,
    pub hashes: Vec<Vec<u64>>,
}

impl EngramHashSetup {
    pub fn read(path: &Path, config_sha256: &str, tokenizer_sha256: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read engram hash setup {}", path.display()))?;
        let setup: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode engram hash setup {}", path.display()))?;
        setup.validate(config_sha256, tokenizer_sha256)?;
        Ok(setup)
    }

    /// SHA-256 fingerprints of the loaded checkpoint's config and tokenizer files.
    pub fn model_fingerprints(model: &Path) -> Result<(String, String)> {
        let digest = |name: &str| -> Result<String> {
            let bytes = std::fs::read(model.join(name))
                .with_context(|| format!("read model {name} for Engram fingerprints"))?;
            let mut hash = sha2::Sha256::new();
            hash.update(bytes);
            Ok(format!("{:x}", hash.finalize()))
        };
        Ok((digest("config.json")?, digest("tokenizer.json")?))
    }

    /// Validates the record schema and its fingerprints against the loaded model files.
    pub fn validate_model(&self, model: &Path) -> Result<()> {
        let (config_sha256, tokenizer_sha256) = Self::model_fingerprints(model)?;
        self.validate(&config_sha256, &tokenizer_sha256)
    }

    pub fn validate(&self, config_sha256: &str, tokenizer_sha256: &str) -> Result<()> {
        ensure!(
            self.schema == HASH_SCHEMA,
            "engram hash schema {} differs from {HASH_SCHEMA}",
            self.schema
        );
        ensure!(
            config_sha256.len() == 64
                && tokenizer_sha256.len() == 64
                && self.config_sha256 == config_sha256
                && self.tokenizer_sha256 == tokenizer_sha256,
            "engram hash setup config/tokenizer fingerprint differs"
        );
        let layers = self.primes.len();
        ensure!(
            layers > 0
                && self.multipliers.len() == layers
                && self.layer_ids.len() == layers
                && self.layer_ids.iter().collect::<BTreeSet<_>>().len() == layers,
            "engram hash setup layer geometry differs"
        );
        let ngram = self.multipliers[0].len();
        let heads = self.primes[0]
            .first()
            .context("empty engram prime row")?
            .len();
        ensure!(
            ngram >= 2 && heads > 0,
            "engram hash ngram={ngram} heads={heads} is outside the model form"
        );
        ensure!(
            self.tokenizer_vocab > self.compressed_vocab
                && self.compressed_vocab > 1
                && self.pad_id < self.compressed_vocab,
            "engram tokenizer/compressed vocab or pad id is invalid"
        );
        ensure!(
            self.token_map.len() == self.tokenizer_vocab as usize,
            "engram token map holds {} of {} tokenizer ids",
            self.token_map.len(),
            self.tokenizer_vocab
        );
        for id in &self.token_map {
            ensure!(
                *id < self.compressed_vocab,
                "engram token map id {id} is out of bounds"
            );
        }
        let mut unique_primes = BTreeSet::new();
        for layer in &self.primes {
            ensure!(
                layer.len() + 1 == ngram,
                "engram prime n-gram count differs"
            );
            for size in layer {
                ensure!(size.len() == heads, "engram prime head count differs");
                for prime in size {
                    ensure!(
                        is_prime(*prime),
                        "engram bucket modulus {prime} is not prime"
                    );
                    ensure!(
                        unique_primes.insert(*prime),
                        "engram prime {prime} is reused"
                    );
                }
            }
        }
        let product_bound = i64::MAX / i64::from(self.compressed_vocab);
        for layer in &self.multipliers {
            ensure!(
                layer.len() == ngram,
                "engram multiplier n-gram count differs"
            );
            for multiplier in layer {
                ensure!(
                    *multiplier != 0 && multiplier.unsigned_abs() <= product_bound as u64,
                    "engram multiplier {multiplier} can overflow i64"
                );
            }
        }
        let width = (ngram - 1) * heads;
        for sample in &self.samples {
            ensure!(
                sample.hashes.len() == layers
                    && sample
                        .hashes
                        .iter()
                        .all(|rows| rows.len() == sample.input_ids.len() * width)
                    && sample
                        .text_mask
                        .as_ref()
                        .is_none_or(|mask| mask.len() == sample.input_ids.len()),
                "engram hash sample extent differs"
            );
        }
        Ok(())
    }

    pub fn layer_rows(&self) -> Result<Vec<u32>> {
        self.primes
            .iter()
            .map(|sizes| {
                sizes
                    .iter()
                    .flatten()
                    .try_fold(0u64, |sum, prime| {
                        sum.checked_add(*prime)
                            .context("engram table rows overflow")
                    })
                    .and_then(|rows| u32::try_from(rows).context("engram table rows exceed u32"))
            })
            .collect()
    }

    pub fn ngram(&self) -> usize {
        self.multipliers.first().map_or(0, Vec::len)
    }

    pub fn heads(&self) -> usize {
        self.primes
            .first()
            .and_then(|row| row.first())
            .map_or(0, Vec::len)
    }

    pub fn hash_width(&self) -> usize {
        (self.ngram() - 1) * self.heads()
    }

    fn offsets(&self) -> Vec<Vec<u64>> {
        // One disjoint range per (n-gram size, head), in the definition's flattened order.
        self.primes
            .iter()
            .map(|sizes| {
                let mut offsets = Vec::new();
                let mut sum = 0u64;
                for primes in sizes {
                    for prime in primes {
                        offsets.push(sum);
                        sum += prime;
                    }
                }
                offsets
            })
            .collect()
    }

    pub fn validate_samples(&self, model: &Path) -> Result<()> {
        let capacity = self
            .samples
            .iter()
            .map(|sample| sample.start_position as usize + sample.input_ids.len())
            .max()
            .unwrap_or(0);
        let mut hasher = NgramHistory::new_for_model(self.clone(), capacity, model)?;
        for (index, sample) in self.samples.iter().enumerate() {
            let observed = hasher.prepare(
                &sample.input_ids,
                sample.start_position,
                sample.text_mask.as_deref(),
            )?;
            ensure!(
                observed == sample.hashes,
                "engram hash sample {index} differs"
            );
            hasher.reset();
        }
        Ok(())
    }
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value == 2 {
        return true;
    }
    if value.is_multiple_of(2) {
        return false;
    }
    let mut divisor = 3u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

pub struct NgramHistory {
    compressed: Vec<i64>,
    initialized: Vec<bool>,
    setup: EngramHashSetup,
    offsets: Vec<Vec<u64>>,
}

impl NgramHistory {
    pub fn new(setup: EngramHashSetup, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "engram history capacity is zero");
        let offsets = setup.offsets();
        Ok(Self {
            compressed: vec![0; capacity],
            initialized: vec![false; capacity],
            setup,
            offsets,
        })
    }
    pub fn new_for_model(setup: EngramHashSetup, capacity: usize, model: &Path) -> Result<Self> {
        setup.validate_model(model)?;
        Self::new(setup, capacity)
    }

    pub fn reset(&mut self) {
        self.compressed.fill(0);
        self.initialized.fill(false);
    }

    pub fn prepare(
        &mut self,
        input_ids: &[u32],
        start_position: u32,
        text_mask: Option<&[bool]>,
    ) -> Result<Vec<Vec<u64>>> {
        ensure!(!input_ids.is_empty(), "engram request has no tokens");
        ensure!(
            text_mask.is_none_or(|mask| mask.len() == input_ids.len()),
            "engram text mask extent differs"
        );
        let start = usize::try_from(start_position).context("engram position exceeds usize")?;
        let end = start
            .checked_add(input_ids.len())
            .context("engram request extent overflows")?;
        ensure!(
            end <= self.compressed.len(),
            "engram request {}..{end} exceeds capacity {}",
            start,
            self.compressed.len()
        );
        let vocab = self.setup.token_map.len();
        for (offset, id) in input_ids.iter().enumerate() {
            ensure!(
                usize::try_from(*id).is_ok_and(|id| id < vocab),
                "engram token id {id} at position {} is outside the token map",
                start + offset
            );
            let compressed = if text_mask.is_some_and(|mask| !mask[offset]) {
                DEAD_TOKEN
            } else {
                i64::from(self.setup.token_map[usize::try_from(*id)?])
            };
            self.compressed[start + offset] = compressed;
            self.initialized[start + offset] = true;
        }

        let layers = self.setup.primes.len();
        let ngram = self.setup.ngram();
        let heads = self.setup.heads();
        let width = self.setup.hash_width();
        let mut result = vec![Vec::with_capacity(input_ids.len() * width); layers];
        for position in start..end {
            let mut tokens = [0u32; 16];
            ensure!(
                ngram <= tokens.len(),
                "engram ngram {ngram} exceeds history bound"
            );
            let mut blocked = false;
            for (shift, token) in tokens.iter_mut().enumerate().take(ngram) {
                if shift > position {
                    blocked = true;
                } else {
                    let source_position = position - shift;
                    ensure!(
                        self.initialized[source_position],
                        "engram history position {source_position} is missing"
                    );
                    if self.compressed[source_position] == DEAD_TOKEN {
                        blocked = true;
                    }
                }
                *token = if blocked {
                    self.setup.pad_id
                } else {
                    self.compressed[position - shift] as u32
                };
            }
            for (layer, result) in result.iter_mut().enumerate() {
                let mut rolling = i64::from(tokens[0]) * self.setup.multipliers[layer][0];
                for (ngram_size, token) in tokens.iter().enumerate().take(ngram).skip(1) {
                    let product = i64::from(*token)
                        .checked_mul(self.setup.multipliers[layer][ngram_size])
                        .context("engram hash product overflows")?;
                    rolling ^= product;
                    let unsigned =
                        u64::try_from(rolling).context("engram rolling hash is negative")?;
                    for head in 0..heads {
                        let bucket = unsigned % self.setup.primes[layer][ngram_size - 1][head];
                        let hash = self.offsets[layer][(ngram_size - 1) * heads + head]
                            .checked_add(bucket)
                            .context("engram hash offset overflows")?;
                        result.push(hash);
                    }
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeTensorLayout {
    shard: PathBuf,
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTableLayout {
    pub layer_id: u32,
    pub layer_index: usize,
    pub rows: u64,
    pub head_dim: usize,
    pub scales: usize,
    weight: NativeTensorLayout,
    scale: NativeTensorLayout,
}

impl NativeTableLayout {
    pub fn from_manifest(
        manifest: &TensorManifest,
        layer_ids: &[u32],
        layer_index: usize,
        head_dim: usize,
        scales: usize,
    ) -> Result<Self> {
        let layer_id = *layer_ids
            .get(layer_index)
            .context("engram layer id is missing")?;
        let prefix = format!("layers.{layer_id}.engram.embed");
        let weight = manifest
            .entries
            .get(&format!("{prefix}.weight"))
            .with_context(|| format!("manifest tensor {prefix}.weight is missing"))?;
        let scale = manifest
            .entries
            .get(&format!("{prefix}.scale"))
            .with_context(|| format!("manifest tensor {prefix}.scale is missing"))?;
        let rows = Self::rows(weight, scale, head_dim, scales)?;
        Self::validate_tensor(weight, Dtype::F8_E4M3, &[rows as usize, head_dim])?;
        Self::validate_tensor(scale, Dtype::F8_E8M0, &[rows as usize, scales])?;
        Ok(Self {
            layer_id,
            layer_index,
            rows,
            head_dim,
            scales,
            weight: NativeTensorLayout {
                shard: PathBuf::from(&weight.shard),
                offset: weight
                    .data_start
                    .checked_add(weight.data_offsets[0])
                    .context("engram weight disk offset overflows")?,
                bytes: weight.bytes as u64,
            },
            scale: NativeTensorLayout {
                shard: PathBuf::from(&scale.shard),
                offset: scale
                    .data_start
                    .checked_add(scale.data_offsets[0])
                    .context("engram scale disk offset overflows")?,
                bytes: scale.bytes as u64,
            },
        })
    }

    fn rows(
        weight: &TensorEntry,
        scale: &TensorEntry,
        head_dim: usize,
        scales: usize,
    ) -> Result<u64> {
        let weight_rows = weight
            .shape
            .first()
            .copied()
            .context("engram weight shape has no rows")?;
        let scale_rows = scale
            .shape
            .first()
            .copied()
            .context("engram scale shape has no rows")?;
        ensure!(
            weight.shape.len() == 2
                && scale.shape.len() == 2
                && weight.shape[1] == head_dim
                && scale.shape[1] == scales
                && weight_rows == scale_rows,
            "engram table shapes differ: weight={:?} scale={:?}",
            weight.shape,
            scale.shape
        );
        u64::try_from(weight_rows).context("engram table rows exceed u64")
    }

    fn validate_tensor(entry: &TensorEntry, dtype: Dtype, shape: &[usize]) -> Result<()> {
        ensure!(
            entry.dtype == dtype && entry.shape == shape,
            "engram tensor dtype/shape differs: expected {dtype:?} {shape:?}, got {:?} {:?}",
            entry.dtype,
            entry.shape
        );
        let extent = entry.data_offsets[1]
            .checked_sub(entry.data_offsets[0])
            .context("engram tensor data extent underflows")?;
        ensure!(
            extent == entry.bytes as u64,
            "engram tensor extent {extent} differs from {} bytes",
            entry.bytes
        );
        Ok(())
    }

    fn native_offset(&self, row: u64) -> Result<(u64, u64)> {
        ensure!(row < self.rows, "engram row {row} is outside {}", self.rows);
        let weight = self
            .weight
            .offset
            .checked_add(
                row.checked_mul(self.head_dim as u64)
                    .context("engram weight row offset overflows")?,
            )
            .context("engram weight offset overflows")?;
        let scale = self
            .scale
            .offset
            .checked_add(
                row.checked_mul(self.scales as u64)
                    .context("engram scale row offset overflows")?,
            )
            .context("engram scale offset overflows")?;
        Ok((weight, scale))
    }
}

/// Largest single direct read; longer merged spans are split.
const DIRECT_SPAN_MAX: u64 = 64 << 10;

impl NativeTableLayout {
    /// Native records for `rows`: 256 weight bytes then 8 scale bytes, read in merged spans.
    fn read_rows(
        &self,
        reader: &SpanReader,
        rows: &[u64],
        buffer: &mut AlignedBytes,
    ) -> Result<(Vec<Vec<u8>>, usize)> {
        let layout = self;
        let mut output: Vec<Vec<u8>> = rows
            .iter()
            .map(|_| vec![0; ENGRAM_NATIVE_ROW_BYTES])
            .collect();
        let mut weight = Vec::with_capacity(rows.len());
        let mut scale = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let (weight_offset, scale_offset) = layout.native_offset(*row)?;
            weight.push((weight_offset, index));
            scale.push((scale_offset, index));
        }
        let mut spans = 0;
        for (source, requests, item, destination) in [
            (&layout.weight, weight, layout.head_dim as u64, 0usize),
            (&layout.scale, scale, layout.scales as u64, ENGRAM_HEAD_DIM),
        ] {
            let pairs: Vec<_> = requests.iter().map(|&(offset, _)| (offset, item)).collect();
            let source_end = source
                .offset
                .checked_add(source.bytes)
                .context("engram source overflow")?;
            for &(offset, len) in &pairs {
                ensure!(
                    offset >= source.offset
                        && offset.checked_add(len).is_some_and(|end| end <= source_end),
                    "engram request outside native tensor"
                );
            }
            spans += reader.read_spans(&pairs, DIRECT_SPAN_MAX, buffer, |index, bytes| {
                let row = requests[index].1;
                let end = destination
                    .checked_add(bytes.len())
                    .context("engram destination overflow")?;
                ensure!(end <= output[row].len(), "engram record extent differs");
                output[row][destination..end].copy_from_slice(bytes);
                Ok(())
            })?;
        }
        Ok((output, spans))
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
struct CacheKey {
    layer: usize,
    row: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CacheOrder {
    frequency: u64,
    last_use: u64,
    key: CacheKey,
}

struct CacheEntry {
    native: Vec<u8>,
    frequency: u64,
    last_use: u64,
}

pub struct NativeLfu {
    capacity: usize,
    clock: u64,
    entries: HashMap<CacheKey, CacheEntry>,
    order: BTreeSet<CacheOrder>,
}

impl NativeLfu {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            entries: HashMap::new(),
            order: BTreeSet::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn native_bytes(&self) -> usize {
        self.entries.len() * ENGRAM_NATIVE_ROW_BYTES
    }

    fn touch(&mut self, key: CacheKey) -> bool {
        let Some(entry) = self.entries.get_mut(&key) else {
            return false;
        };
        self.clock += 1;
        self.order.remove(&CacheOrder {
            frequency: entry.frequency,
            last_use: entry.last_use,
            key,
        });
        entry.frequency = entry.frequency.saturating_add(1);
        entry.last_use = self.clock;
        self.order.insert(CacheOrder {
            frequency: entry.frequency,
            last_use: entry.last_use,
            key,
        });
        true
    }

    fn get(&mut self, key: CacheKey) -> Option<&[u8]> {
        if !self.touch(key) {
            return None;
        }
        self.entries.get(&key).map(|entry| entry.native.as_slice())
    }

    fn insert(&mut self, key: CacheKey, native: Vec<u8>) -> Result<()> {
        ensure!(
            native.len() == ENGRAM_NATIVE_ROW_BYTES,
            "engram native row holds {} bytes",
            native.len()
        );
        if self.entries.contains_key(&key) {
            self.touch(key);
            return Ok(());
        }
        while self.entries.len() >= self.capacity && self.capacity > 0 {
            let victim = *self
                .order
                .iter()
                .next()
                .expect("nonempty LFU has an order entry");
            self.order.remove(&victim);
            self.entries.remove(&victim.key);
        }
        if self.capacity == 0 {
            return Ok(());
        }
        self.clock += 1;
        let entry = CacheEntry {
            native,
            frequency: 1,
            last_use: self.clock,
        };
        let order = CacheOrder {
            frequency: entry.frequency,
            last_use: entry.last_use,
            key,
        };
        self.order.insert(order);
        self.entries.insert(key, entry);
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct PrepareReport {
    pub generation: u64,
    pub start_position: u32,
    pub rows: usize,
    pub requested_lookup_rows: usize,
    pub unique_table_rows: usize,
    pub cache_hits: usize,
    pub misses: usize,
    pub direct_row_reads: usize,
    pub direct_spans: usize,
}

pub struct EngramTicket {
    generation: u64,
    layer_ids: Vec<u32>,
    start_position: u32,
    rows: usize,
    text_mask: Vec<bool>,
    hashes: Vec<Vec<u64>>,
    records: Vec<Vec<u8>>,
    active_generation: Arc<AtomicU64>,
}

pub struct EngramNativeRows<'a> {
    pub generation: u64,
    pub layer_id: u32,
    pub start_position: u32,
    pub rows: usize,
    pub width: usize,
    pub text_mask: &'a [bool],
    pub hashes: &'a [u64],
    pub records: &'a [u8],
}

impl EngramTicket {
    pub fn take(&self, layer_id: u32) -> Result<EngramNativeRows<'_>> {
        let current = self.active_generation.load(Ordering::SeqCst);
        ensure!(
            current == self.generation,
            "engram ticket generation {current} supersedes request generation {}",
            self.generation
        );
        let layer = self
            .layer_ids
            .iter()
            .position(|id| *id == layer_id)
            .with_context(|| format!("engram layer {layer_id} is not in the ticket"))?;
        let width = self.hashes[layer].len() / self.rows;
        Ok(EngramNativeRows {
            generation: self.generation,
            layer_id,
            start_position: self.start_position,
            rows: self.rows,
            width,
            text_mask: &self.text_mask,
            hashes: &self.hashes[layer],
            records: &self.records[layer],
        })
    }
}

impl EngramNativeRows<'_> {
    pub fn record(&self, row: usize, hash_column: usize) -> Result<&[u8]> {
        ensure!(
            row < self.rows,
            "engram output row {row} is outside {}",
            self.rows
        );
        ensure!(
            hash_column < self.width,
            "engram hash column {hash_column} is outside {}",
            self.width
        );
        let start = (row * self.width + hash_column) * ENGRAM_NATIVE_ROW_BYTES;
        Ok(&self.records[start..start + ENGRAM_NATIVE_ROW_BYTES])
    }
}

pub struct EngramService {
    setup: EngramHashSetup,
    history: NgramHistory,
    tables: Vec<NativeTableLayout>,
    direct: Vec<SpanReader>,
    buffers: Vec<AlignedBytes>,
    cache: NativeLfu,
    generation: Arc<AtomicU64>,
}

impl EngramService {
    pub fn open(
        model_dir: &Path,
        manifest: &TensorManifest,
        setup: EngramHashSetup,
        capacity_rows: usize,
        history_capacity: usize,
    ) -> Result<Self> {
        setup.validate_model(model_dir)?;
        ensure!(capacity_rows > 0, "engram LFU capacity is zero");
        let layer_ids = setup.layer_ids.clone();
        let layer_rows = setup.layer_rows()?;
        let head_dim = setup
            .heads()
            .checked_mul(32)
            .context("engram head extent overflows")?;
        ensure!(
            head_dim == ENGRAM_HEAD_DIM,
            "published engram head dimension is required"
        );
        let mut tables = Vec::with_capacity(layer_ids.len());
        let mut direct = Vec::with_capacity(layer_ids.len());
        let mut buffers = Vec::with_capacity(layer_ids.len());
        for layer_index in 0..layer_ids.len() {
            let table = NativeTableLayout::from_manifest(
                manifest,
                &layer_ids,
                layer_index,
                head_dim,
                setup.heads(),
            )?;
            ensure!(
                table.rows == u64::from(layer_rows[layer_index]),
                "engram layer {} manifest rows {} differ from hash buckets {}",
                layer_ids[layer_index],
                table.rows,
                layer_rows[layer_index]
            );
            ensure!(
                table.weight.shard == table.scale.shard,
                "engram layer {} weight and scale shards differ",
                layer_ids[layer_index]
            );
            let reader = SpanReader::open(&model_dir.join(&table.weight.shard), true)?;
            let alignment =
                usize::try_from(reader.alignment).context("engram alignment exceeds usize")?;
            let mut buffer = AlignedBytes::new(
                usize::try_from(DIRECT_SPAN_MAX).context("engram span exceeds usize")?
                    + 2 * alignment,
                alignment,
            )?;
            let (probe, _) = table.read_rows(&reader, &[0], &mut buffer)?;
            ensure!(
                probe.len() == 1 && probe[0].len() == ENGRAM_NATIVE_ROW_BYTES,
                "engram row-0 probe differs"
            );
            tables.push(table);
            direct.push(reader);
            buffers.push(buffer);
        }
        let history = NgramHistory::new(setup, history_capacity)?;
        Ok(Self {
            setup: history.setup.clone(),
            history,
            tables,
            direct,
            buffers,
            cache: NativeLfu::new(capacity_rows),
            generation: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn reset(&mut self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.history.reset();
    }

    pub fn prepare(
        &mut self,
        input_ids: &[u32],
        start_position: u32,
        text_mask: Option<&[bool]>,
    ) -> Result<(EngramTicket, PrepareReport)> {
        let generation = self.generation.load(Ordering::SeqCst);
        let hashes = self.history.prepare(input_ids, start_position, text_mask)?;
        let rows = input_ids.len();
        let width = self.setup.hash_width();
        let mut records = Vec::with_capacity(self.tables.len());
        for layer in &hashes {
            ensure!(layer.len() == rows * width, "engram hash extent differs");
            records.push(vec![0; rows * width * ENGRAM_NATIVE_ROW_BYTES]);
        }
        let mut misses = vec![Vec::new(); self.tables.len()];
        let mut requested_native = HashMap::new();
        let mut requested = 0;
        let mut hits = 0;
        for (layer, row_hashes) in hashes.iter().enumerate() {
            let distinct: BTreeSet<u64> = row_hashes.iter().copied().collect();
            requested += distinct.len();
            for row in distinct {
                let key = CacheKey { layer, row };
                if let Some(native) = self.cache.get(key) {
                    requested_native.insert(key, native.to_vec());
                    hits += 1;
                } else {
                    misses[layer].push(row);
                }
            }
        }
        let miss_count: usize = misses.iter().map(Vec::len).sum();
        let mut spans = 0;
        for (layer, layer_misses) in misses.iter_mut().enumerate() {
            if layer_misses.is_empty() {
                continue;
            }
            let layer_rows = std::mem::take(layer_misses);
            let (native, reads) = self.tables[layer].read_rows(
                &self.direct[layer],
                &layer_rows,
                &mut self.buffers[layer],
            )?;
            spans += reads;
            for (row, bytes) in layer_rows.iter().zip(native) {
                let key = CacheKey { layer, row: *row };
                requested_native.insert(key, bytes.clone());
                self.cache.insert(key, bytes)?;
            }
        }
        for (layer, row_hashes) in hashes.iter().enumerate() {
            for (logical_index, row) in row_hashes.iter().enumerate() {
                let native = requested_native
                    .get(&CacheKey { layer, row: *row })
                    .context("engram requested native row missing")?;
                copy_record(&mut records[layer], logical_index, native)?;
            }
        }
        let mask = text_mask.map_or_else(|| vec![true; rows], <[bool]>::to_vec);
        let ticket = EngramTicket {
            generation,
            layer_ids: self.tables.iter().map(|table| table.layer_id).collect(),
            start_position,
            rows,
            text_mask: mask,
            hashes,
            records,
            active_generation: Arc::clone(&self.generation),
        };
        let report = PrepareReport {
            generation,
            start_position,
            rows,
            requested_lookup_rows: rows * width * self.tables.len(),
            unique_table_rows: requested,
            cache_hits: hits,
            misses: miss_count,
            direct_row_reads: miss_count,
            direct_spans: spans,
        };
        Ok((ticket, report))
    }
}

fn copy_record(output: &mut [u8], index: usize, native: &[u8]) -> Result<()> {
    let start = index
        .checked_mul(ENGRAM_NATIVE_ROW_BYTES)
        .context("engram output offset overflows")?;
    let end = start
        .checked_add(ENGRAM_NATIVE_ROW_BYTES)
        .context("engram output extent overflows")?;
    ensure!(
        end <= output.len() && native.len() == ENGRAM_NATIVE_ROW_BYTES,
        "engram output record extent differs"
    );
    output[start..end].copy_from_slice(native);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorManifest, TensorScope};
    use crate::experts::disk::plan_spans;
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    fn hash_setup() -> EngramHashSetup {
        EngramHashSetup {
            schema: 1,
            layer_ids: vec![1, 14],
            config_sha256: "a".repeat(64),
            tokenizer_sha256: "b".repeat(64),
            tokenizer_vocab: 11,
            compressed_vocab: 10,
            pad_id: 2,
            token_map: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
            multipliers: vec![vec![1, 3, 5, 7], vec![9, 11, 13, 15]],
            primes: vec![
                vec![vec![11, 13], vec![17, 19], vec![23, 29]],
                vec![vec![31, 37], vec![41, 43], vec![47, 53]],
            ],
            samples: Vec::new(),
        }
    }

    fn primes_from(start: u64, count: usize) -> Vec<u64> {
        let mut values = Vec::with_capacity(count);
        let mut candidate = (start + 1) | 1;
        while values.len() < count {
            if is_prime(candidate) {
                values.push(candidate);
            }
            candidate += 2;
        }
        values
    }

    fn wide_hash_setup() -> EngramHashSetup {
        let all = primes_from(15_999_999, 48);
        let mut setup = hash_setup();
        setup.tokenizer_vocab = 101;
        setup.compressed_vocab = 100;
        setup.pad_id = 2;
        setup.token_map = (0..100).chain([0]).collect();
        setup.multipliers = vec![vec![1, 3, 5, 7], vec![9, 11, 13, 15]];
        setup.primes = vec![
            (0..3)
                .map(|index| all[index * 8..(index + 1) * 8].to_vec())
                .collect(),
            (3..6)
                .map(|index| all[index * 8..(index + 1) * 8].to_vec())
                .collect(),
        ];
        setup
    }

    #[test]
    fn hash_history_blocks_boundary_and_image_span() -> Result<()> {
        let mut history = NgramHistory::new(hash_setup(), 8)?;
        let ids = [1u32, 2, 3, 4, 5, 6];
        let mask = [true, false, true, true, false, true];
        let hashes = history.prepare(&ids, 0, Some(&mask))?;
        let width = 6;
        let boundary = hashes[0][0..width].to_vec();
        let blocked_image = hashes[0][width..2 * width].to_vec();
        ensure!(
            hashes[0][4 * width..5 * width] == blocked_image,
            "image positions hash differently: {blocked_image:?} != {:?}",
            hashes[0][4 * width..5 * width].to_vec()
        );
        ensure!(
            blocked_image != boundary,
            "an image position hashes like a sequence boundary"
        );
        ensure!(hashes[0] != hashes[1], "engram layers did not separate");
        Ok(())
    }

    #[test]
    fn hash_history_continues_across_prepare_boundaries() -> Result<()> {
        let shared = hash_setup();
        let mut split = NgramHistory::new(shared.clone(), 4)?;
        let first = split.prepare(&[1, 2, 3], 0, None)?;
        let second = split.prepare(&[4], 3, None)?;
        let mut whole = NgramHistory::new(shared, 4)?;
        let reference = whole.prepare(&[1, 2, 3, 4], 0, None)?;
        ensure!(
            first[0][..12].to_vec() == reference[0][..12].to_vec()
                && first[1][..12].to_vec() == reference[1][..12].to_vec(),
            "split prefill prefix differs from whole prefill"
        );
        ensure!(
            reference[1][18..24] == second[1],
            "layer-14 split continuation differs from whole prefill: {:?} != {:?}",
            second[1].to_vec(),
            reference[1][18..24].to_vec()
        );
        ensure!(
            reference[0][18..24] == second[0],
            "split continuation differs from whole prefill: {:?} != {:?}",
            second[0].to_vec(),
            reference[0][18..24].to_vec()
        );
        Ok(())
    }

    fn entry(dtype: Dtype, shape: Vec<usize>, offset: u64) -> TensorEntry {
        let bytes = shape.iter().product::<usize>();
        TensorEntry {
            dtype,
            shape,
            shard: "table.bin".into(),
            bytes,
            load: false,
            data_start: 8,
            data_offsets: [offset, offset + bytes as u64],
            scope: TensorScope::Text,
            layer: Some(1),
            role: "engram.embed.weight".into(),
            scale_name: None,
        }
    }

    #[test]
    fn manifest_layout_keeps_published_native_extents() -> Result<()> {
        let mut manifest = TensorManifest {
            entries: BTreeMap::new(),
            resident_bytes: 0,
        };
        manifest.entries.insert(
            "layers.1.engram.embed.weight".into(),
            entry(Dtype::F8_E4M3, vec![384_006_168, 256], 17),
        );
        manifest.entries.insert(
            "layers.1.engram.embed.scale".into(),
            entry(Dtype::F8_E8M0, vec![384_006_168, 8], 98_305_579_025),
        );
        let layout = NativeTableLayout::from_manifest(&manifest, &[1, 14], 0, 256, 8)?;
        ensure!(layout.rows == 384_006_168);
        ensure!(layout.weight.offset == 25 && layout.scale.offset == 98_305_579_033);
        Ok(())
    }

    #[test]
    fn native_lfu_is_bounded_and_deterministic() -> Result<()> {
        let mut lfu = NativeLfu::new(2);
        let a = CacheKey { layer: 0, row: 1 };
        let b = CacheKey { layer: 0, row: 2 };
        let c = CacheKey { layer: 1, row: 3 };
        lfu.insert(a, vec![1; ENGRAM_NATIVE_ROW_BYTES])?;
        lfu.insert(b, vec![2; ENGRAM_NATIVE_ROW_BYTES])?;
        ensure!(lfu.get(a).is_some());
        ensure!(lfu.get(a).is_some());
        lfu.insert(c, vec![3; ENGRAM_NATIVE_ROW_BYTES])?;
        ensure!(lfu.get(a).is_some() && lfu.get(b).is_none());
        ensure!(lfu.len() == 2 && lfu.native_bytes() == 528);
        Ok(())
    }

    #[test]
    #[ignore = "requires a filesystem supporting O_DIRECT"]
    fn direct_service_returns_ordered_native_records() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "flyingfish-engram-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        std::fs::create_dir_all(&root)?;
        let result = (|| {
            let setup = wide_hash_setup();
            let layer_rows = setup.layer_rows()?;
            let ids = [1u32, 2, 3, 4];
            let mut hasher = NgramHistory::new(setup.clone(), ids.len())?;
            let selected = hasher.prepare(&ids, 0, None)?;
            let selected: Vec<BTreeSet<u64>> = selected
                .iter()
                .map(|rows| rows.iter().copied().chain([0]).collect())
                .collect();
            let mut manifest = TensorManifest {
                entries: BTreeMap::new(),
                resident_bytes: 0,
            };
            let mut expected = BTreeMap::new();
            let mut files = Vec::new();
            for (layer_index, layer_id) in [1u32, 14].into_iter().enumerate() {
                let name = format!("engram-{layer_id}.bin");
                let path = root.join(&name);
                let file = File::create(&path)?;
                let alignment = SpanReader::open(&path, true)?.alignment;
                let rows = u64::from(layer_rows[layer_index]);
                let weight_base = alignment - 8;
                let scale_base =
                    (alignment + rows * ENGRAM_HEAD_DIM as u64).div_ceil(alignment) * alignment;
                let file_len = scale_base + rows * ENGRAM_SCALES as u64 + alignment;
                file.set_len(file_len)?;
                for row in selected[layer_index].iter().copied() {
                    let native = native_record(row, layer_index as u8);
                    file.write_all_at(
                        &native[..ENGRAM_HEAD_DIM],
                        alignment + row * ENGRAM_HEAD_DIM as u64,
                    )?;
                    file.write_all_at(
                        &native[ENGRAM_HEAD_DIM..],
                        scale_base + row * ENGRAM_SCALES as u64,
                    )?;
                    expected.insert((layer_index, u32::try_from(row)?), native);
                }
                file.sync_all()?;
                files.push(file);
                let mut weight = entry(
                    Dtype::F8_E4M3,
                    vec![rows as usize, ENGRAM_HEAD_DIM],
                    weight_base,
                );
                weight.shard = name.clone();
                let mut scale = entry(
                    Dtype::F8_E8M0,
                    vec![rows as usize, ENGRAM_SCALES],
                    scale_base - 8,
                );
                scale.shard = name;
                manifest
                    .entries
                    .insert(format!("layers.{layer_id}.engram.embed.weight"), weight);
                manifest
                    .entries
                    .insert(format!("layers.{layer_id}.engram.embed.scale"), scale);
            }
            drop(files);
            let capacity: usize = layer_rows.iter().map(|rows| *rows as usize).sum();
            let mut service = EngramService::open(&root, &manifest, setup, capacity, 4)?;
            let (ticket, first) = service.prepare(&ids, 0, None)?;
            ensure!(first.misses > 0, "first request did not read native rows");
            ensure!(
                first.direct_spans <= 2 * first.direct_row_reads,
                "merged spans exceed the per-row read bound: {first:?}"
            );
            for (layer_index, layer_id) in [(0usize, 1u32), (1, 14)] {
                let rows = ticket.take(layer_id)?;
                ensure!(rows.records.len() == ids.len() * rows.width * ENGRAM_NATIVE_ROW_BYTES);
                for row in 0..ids.len() {
                    for column in 0..rows.width {
                        let hash = rows.hashes[row * rows.width + column];
                        let native = expected
                            .get(&(layer_index, hash as u32))
                            .with_context(|| format!("expected engram row {hash} is missing"))?;
                        let actual = rows.record(row, column)?.to_vec();
                        let difference = actual
                            .iter()
                            .zip(native)
                            .position(|(left, right)| left != right);
                        let offsets = service.tables[layer_index].native_offset(hash)?;
                        let mut disk_scale = [0u8; ENGRAM_SCALES];
                        File::open(root.join(format!("engram-{layer_id}.bin")))?
                            .read_exact_at(&mut disk_scale, offsets.1)?;
                        ensure!(
                            actual == native.as_slice(),
                            "native engram record layer={layer_id} row={row} column={column} hash={hash} offsets={offsets:?} differs at {difference:?}: actual={:?} disk_scale={disk_scale:?} expected={:?}",
                            &actual[248.min(actual.len())..],
                            &native[248.min(native.len())..]
                        );
                    }
                }
            }
            service.reset();
            ensure!(
                ticket.take(1).is_err(),
                "reset retained an old engram ticket"
            );
            let (_, second) = service.prepare(&ids, 0, None)?;
            ensure!(
                second.misses == 0
                    && second.cache_hits == second.unique_table_rows
                    && second.direct_row_reads == 0,
                "native LFU did not serve the repeated request: {second:?}"
            );
            service.cache = NativeLfu::new(1);
            service.reset();
            let (evicted, _) = service.prepare(&ids, 0, None)?;
            for (layer_index, layer_id) in [1u32, 14].into_iter().enumerate() {
                let rows = evicted.take(layer_id)?;
                for (index, &hash) in rows.hashes.iter().enumerate() {
                    ensure!(
                        rows.record(index / rows.width, index % rows.width)?
                            == expected[&(layer_index, hash as u32)],
                        "LFU eviction changed prepared native rows"
                    );
                }
            }
            ensure!(service.cache.len() == 1, "LFU exceeded one-row capacity");
            Ok(())
        })();
        std::fs::remove_dir_all(&root)?;
        result
    }

    fn native_record(row: u64, layer: u8) -> Vec<u8> {
        (0..ENGRAM_NATIVE_ROW_BYTES)
            .map(|index| ((row % 251) as u32 + u32::from(layer) + index as u32 % 251) & 0xff)
            .map(|value| value as u8)
            .collect()
    }

    fn fingerprint_model_dir(setup: &mut EngramHashSetup) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "flyingfish-engram-model-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        std::fs::create_dir_all(&dir).expect("create fingerprint model");
        std::fs::write(dir.join("config.json"), b"engram config").expect("write config");
        std::fs::write(dir.join("tokenizer.json"), b"engram tokenizer").expect("write tokenizer");
        let (config_sha256, tokenizer_sha256) =
            EngramHashSetup::model_fingerprints(&dir).expect("fingerprints");
        setup.config_sha256 = config_sha256;
        setup.tokenizer_sha256 = tokenizer_sha256;
        dir
    }

    fn sampled_setup() -> EngramHashSetup {
        let mut setup = hash_setup();
        let mask = [true, false, true, true];
        let mut hasher = NgramHistory::new(setup.clone(), 8).expect("history");
        let hashes = hasher
            .prepare(&[3u32, 5, 7, 9], 0, Some(&mask))
            .expect("sample hashes");
        setup.samples = vec![EngramHashSample {
            input_ids: vec![3, 5, 7, 9],
            start_position: 0,
            text_mask: Some(mask.to_vec()),
            hashes,
        }];
        setup
    }

    #[test]
    fn hash_samples_catch_setup_mutants() -> Result<()> {
        let mut setup = sampled_setup();
        let model = fingerprint_model_dir(&mut setup);
        setup.validate_model(&model)?;
        setup.validate_samples(&model)?;
        let mut mapping = setup.clone();
        mapping.token_map[3] = 9;
        mapping.validate(&mapping.config_sha256, &mapping.tokenizer_sha256)?;
        assert!(
            mapping.validate_samples(&model).is_err(),
            "an altered compressed-token map escaped the recorded samples"
        );
        let mut multiplier = setup.clone();
        multiplier.multipliers[1][2] = 15;
        multiplier.validate(&multiplier.config_sha256, &multiplier.tokenizer_sha256)?;
        assert!(
            multiplier.validate_samples(&model).is_err(),
            "an altered layer multiplier escaped the recorded samples"
        );
        let mut primes = setup.clone();
        primes.primes[0][1][0] = 59;
        primes.validate(&primes.config_sha256, &primes.tokenizer_sha256)?;
        assert!(
            primes.validate_samples(&model).is_err(),
            "an altered prime escaped the recorded samples"
        );
        let mut offset = setup.clone();
        offset.primes[0][1][0] = 61;
        offset.validate(&offset.config_sha256, &offset.tokenizer_sha256)?;
        assert!(
            offset.validate_samples(&model).is_err(),
            "an altered bucket extent escaped the recorded samples"
        );
        Ok(())
    }

    #[test]
    fn hash_offsets_give_each_size_and_head_a_disjoint_range() -> Result<()> {
        let setup = hash_setup();
        let history = NgramHistory::new(setup.clone(), 8)?;
        let rows = setup.layer_rows()?;
        let heads = setup.heads();
        for (layer, offsets) in history.offsets.iter().enumerate() {
            assert_eq!(offsets.len(), (setup.ngram() - 1) * heads);
            let mut expected = 0u64;
            for (index, offset) in offsets.iter().enumerate() {
                assert_eq!(*offset, expected, "layer {layer} index {index}");
                expected += setup.primes[layer][index / heads][index % heads];
            }
            assert_eq!(expected, u64::from(rows[layer]));
        }
        Ok(())
    }

    #[test]
    fn hash_depends_on_prior_tokens_and_the_mask() -> Result<()> {
        let setup = sampled_setup();
        let mut hasher = NgramHistory::new(setup, 8)?;
        let base = hasher.prepare(&[3u32, 5, 7, 9], 0, None)?;
        hasher.reset();
        let prior = hasher.prepare(&[4u32, 5, 7, 9], 0, None)?;
        assert_ne!(base, prior, "a prior token did not change the hashes");
        hasher.reset();
        let masked = hasher.prepare(&[3u32, 5, 7, 9], 0, Some(&[true, false, true, true]))?;
        assert_ne!(base, masked, "an image span did not change the hashes");
        hasher.reset();
        let repeat = hasher.prepare(&[3u32, 5, 7, 9], 0, None)?;
        assert_eq!(base, repeat, "identical inputs produced different hashes");
        assert_eq!(
            base[0].len(),
            4 * 6,
            "hash width differs from (ngram-1)*heads"
        );
        Ok(())
    }

    #[test]
    #[ignore = "reads the generated engram-hash-v1 record; set F3_HASH"]
    fn hash_record_reproduces_the_publisher_samples() -> Result<()> {
        let path = PathBuf::from(std::env::var("F3_HASH").context("F3_HASH is not set")?);
        let model = PathBuf::from(std::env::var("F3_MODEL").context("F3_MODEL is not set")?);
        let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let setup = EngramHashSetup::read(
            &path,
            document["config_sha256"].as_str().context("config sha")?,
            document["tokenizer_sha256"]
                .as_str()
                .context("tokenizer sha")?,
        )?;
        setup.validate_model(&model)?;
        setup.validate_samples(&model)?;
        let rows = setup.layer_rows()?;
        for sample in &setup.samples {
            for (layer, hashes) in sample.hashes.iter().enumerate() {
                ensure!(
                    hashes.iter().all(|hash| *hash < u64::from(rows[layer])),
                    "sample hash outside the layer {layer} table of {} rows",
                    rows[layer]
                );
            }
        }
        let mut mutant = setup.clone();
        mutant.samples[0].hashes[0][0] ^= 1;
        ensure!(
            mutant.validate_samples(&model).is_err(),
            "a mutated sample escaped validate_samples"
        );
        Ok(())
    }

    #[test]
    fn model_fingerprints_reject_a_wrong_tokenizer() -> Result<()> {
        let mut setup = sampled_setup();
        let model = fingerprint_model_dir(&mut setup);
        setup.validate_model(&model)?;
        let path = model.join("tokenizer.json");
        let mut bytes = std::fs::read(&path)?;
        bytes.push(b' ');
        std::fs::write(&path, bytes)?;
        let error = setup
            .validate_model(&model)
            .expect_err("a changed tokenizer matched the Engram record");
        assert!(
            error.to_string().contains("config/tokenizer fingerprint"),
            "wrong tokenizer error was {error}"
        );
        let _ = std::fs::remove_dir_all(&model);
        Ok(())
    }

    #[test]
    #[ignore = "reads the official shards; set F3_MODEL and F3_FIXTURE"]
    fn official_shards_return_the_recorded_native_bytes() -> Result<()> {
        let model = PathBuf::from(std::env::var("F3_MODEL").context("F3_MODEL is not set")?);
        let fixture = PathBuf::from(std::env::var("F3_FIXTURE").context("F3_FIXTURE is not set")?);
        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
        for layer_id in [1u32, 14] {
            let entry = &document["rows"][layer_id.to_string()];
            let rows = entry["table_rows"].as_u64().context("fixture table rows")?;
            let ids: Vec<u64> = entry["row_ids"]
                .as_array()
                .context("fixture row ids")?
                .iter()
                .map(|value| value.as_u64().context("fixture row id"))
                .collect::<Result<Vec<_>>>()?;
            let shard = entry["shard"].as_str().context("fixture shard")?;
            let layout = NativeTableLayout {
                layer_id,
                layer_index: 0,
                rows,
                head_dim: ENGRAM_HEAD_DIM,
                scales: ENGRAM_SCALES,
                weight: NativeTensorLayout {
                    shard: PathBuf::from(shard),
                    offset: entry["weight_offset"].as_u64().context("weight offset")?,
                    bytes: rows * ENGRAM_HEAD_DIM as u64,
                },
                scale: NativeTensorLayout {
                    shard: PathBuf::from(shard),
                    offset: entry["scale_offset"].as_u64().context("scale offset")?,
                    bytes: rows * ENGRAM_SCALES as u64,
                },
            };
            let reader = SpanReader::open(&model.join(shard), true)?;
            let alignment = usize::try_from(reader.alignment).context("alignment")?;
            let mut buffer =
                AlignedBytes::new(DIRECT_SPAN_MAX as usize + 2 * alignment, alignment)?;
            let (native, spans) = layout.read_rows(&reader, &ids, &mut buffer)?;
            let recorded = std::fs::read(fixture.join(format!("native-layer{layer_id}.bin")))?;
            ensure!(
                recorded.len() == ids.len() * ENGRAM_NATIVE_ROW_BYTES,
                "fixture extent differs"
            );
            for (index, bytes) in native.iter().enumerate() {
                ensure!(
                    bytes.as_slice()
                        == &recorded[index * ENGRAM_NATIVE_ROW_BYTES
                            ..(index + 1) * ENGRAM_NATIVE_ROW_BYTES],
                    "official row {} of layer {layer_id} differs from the recorded native bytes",
                    ids[index]
                );
            }
            ensure!(spans <= 2 * ids.len(), "spans exceed the per-row bound");
            eprintln!(
                "layer {layer_id} rows={} spans={spans} alignment={alignment}",
                ids.len()
            );
        }
        Ok(())
    }

    #[test]
    fn native_lfu_evicts_the_oldest_of_equal_frequency() -> Result<()> {
        let mut lfu = NativeLfu::new(2);
        let a = CacheKey { layer: 0, row: 1 };
        let b = CacheKey { layer: 0, row: 2 };
        let c = CacheKey { layer: 0, row: 3 };
        lfu.insert(a, vec![1; ENGRAM_NATIVE_ROW_BYTES])?;
        lfu.insert(b, vec![2; ENGRAM_NATIVE_ROW_BYTES])?;
        lfu.insert(c, vec![3; ENGRAM_NATIVE_ROW_BYTES])?;
        ensure!(
            lfu.get(a).is_none() && lfu.get(b).is_some() && lfu.get(c).is_some(),
            "the LFU did not evict the oldest entry of equal frequency"
        );
        Ok(())
    }

    #[test]
    fn ticket_records_survive_cache_eviction() -> Result<()> {
        let setup = sampled_setup();
        let mut hasher = NgramHistory::new(setup, 8)?;
        let hashes = hasher.prepare(&[3u32, 5, 7, 9], 0, None)?;
        let width = 6;
        let mut records = vec![vec![0u8; 4 * width * ENGRAM_NATIVE_ROW_BYTES]; 2];
        let mut lfu = NativeLfu::new(1);
        for (layer, row_hashes) in hashes.iter().enumerate() {
            for (index, row) in row_hashes.iter().enumerate() {
                let key = CacheKey { layer, row: *row };
                if !lfu.entries.contains_key(&key) {
                    lfu.insert(key, vec![layer as u8 + 1; ENGRAM_NATIVE_ROW_BYTES])?;
                }
                copy_record(&mut records[layer], index, lfu.get(key).context("row")?)?;
            }
        }
        let before = records.clone();
        for row in 0..8u64 {
            lfu.insert(
                CacheKey {
                    layer: 0,
                    row: 1000 + row,
                },
                vec![9; ENGRAM_NATIVE_ROW_BYTES],
            )?;
        }
        ensure!(lfu.len() == 1, "LFU capacity differs after eviction");
        ensure!(
            records == before,
            "ticket records changed when the cache evicted a row"
        );
        Ok(())
    }

    #[test]
    fn span_plan_merges_one_aligned_block_and_clamps_the_tail() -> Result<()> {
        let adjacent = [(4096u64, 256u64), (4352, 256), (8192, 256)];
        let spans = plan_spans(&adjacent, 4096, 1 << 20, DIRECT_SPAN_MAX)?;
        assert_eq!(spans.len(), 2);
        assert_eq!((spans[0].begin, spans[0].end), (4096, 8192));
        assert_eq!((spans[0].first, spans[0].last), (0, 2));
        assert_eq!((spans[1].begin, spans[1].end), (8192, 12288));
        assert_eq!((spans[1].first, spans[1].last), (2, 3));
        let dense: Vec<(u64, u64)> = (0..16).map(|index| (index * 256, 256)).collect();
        assert_eq!(plan_spans(&dense, 4096, 1 << 16, DIRECT_SPAN_MAX)?.len(), 1);
        let tail = plan_spans(&[(0u64, 256u64)], 4096, 300, DIRECT_SPAN_MAX)?;
        assert_eq!((tail[0].begin, tail[0].end), (0, 300));
        assert!(plan_spans(&[(0u64, 256u64)], 4096, 200, DIRECT_SPAN_MAX).is_err());
        assert!(plan_spans(&[(0u64, 256u64)], 1000, 1 << 16, DIRECT_SPAN_MAX).is_err());
        Ok(())
    }
}
