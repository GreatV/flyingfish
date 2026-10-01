//! Mmap-backed tensor access for Qwen3.8-27B checkpoints: the requant-ed
//! int4 layout written by `src/bin/requant.rs`, or the raw 16-bit
//! (bf16/f16) upstream weights.

use anyhow::{Context, Result, bail, ensure};
pub use ff_core::quant::QuantFormat;
use ff_core::weights::{CachePolicy, ModelWeights, TensorBytes, WeightSource};
use ff_edge0::int4::{GROUP_SIZE, GroupQuant, bf16_bytes_to_u16, bf16_to_f32};
use safetensors::Dtype;
use std::path::Path;

pub struct Qwen35Weights {
    weights: ModelWeights,
    format: QuantFormat,
}

/// The projection set the requantizer quantizes (and the format probe
/// classifies): text-side 2-D `.weight` tensors, vision tower excluded.
fn classifiable(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name.ends_with(".weight")
        && shape[1].is_multiple_of(64)
        && !name.starts_with("model.visual.")
}

impl Qwen35Weights {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let weights = ModelWeights::open(
            model_dir,
            WeightSource::Mmap,
            CachePolicy::unbounded_units(),
        )?;
        let format = detect_format(&weights)?;
        Ok(Self { weights, format })
    }

    pub fn format(&self) -> QuantFormat {
        self.format
    }

    fn dtype(&self, name: &str) -> Result<Dtype> {
        Ok(self.weights.raw_tensor_metadata(name)?.dtype)
    }

    /// Zero-copy tensor view straight off the mmap (file-backed pages are
    /// reclaimable; a materialized f32 copy is not — the f32-caching
    /// version of this path OOM-killed a session).
    fn view(&self, name: &str) -> Result<(Vec<usize>, TensorBytes)> {
        let shape = self.weights.raw_tensor_metadata(name)?.shape;
        Ok((shape, self.weights.tensor_bytes(name)?))
    }

    /// bf16 or f16 bytes widened element-wise, dispatched on the tensor's
    /// own dtype so small tensors work in every checkpoint format.
    fn widen(&self, name: &str, data: &[u8]) -> Result<Vec<f32>> {
        match self.dtype(name)? {
            Dtype::BF16 => Ok(bf16_to_f32(data)),
            Dtype::F16 => Ok(data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            other => bail!("{name}: cannot widen dtype {other:?} to f32"),
        }
    }

    /// Streaming 16-bit matvec over the mmap: per-element widening, threaded
    /// over row chunks, no f32 matrix residency.
    pub fn matvec16(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let key = format!("{name}.weight");
        let (shape, data) = self.view(&key)?;
        ensure!(shape.len() == 2, "{key} is not 2-D: {shape:?}");
        let (rows, cols) = (shape[0], shape[1]);
        ensure!(x.len() == cols, "{name}: x {} != in_dim {cols}", x.len());
        match self.dtype(&key)? {
            Dtype::BF16 => Ok(matvec_rows(&data, x, rows, cols, |b| {
                f32::from_bits(u32::from(b) << 16)
            })),
            Dtype::F16 => Ok(matvec_rows(&data, x, rows, cols, |b| {
                half::f16::from_bits(b).to_f32()
            })),
            other => bail!("{key}: matvec16 needs a 16-bit dtype, found {other:?}"),
        }
    }

    /// One 16-bit row widened (embed gather).
    pub fn row16(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let key = format!("{name}.weight");
        let (shape, data) = self.view(&key)?;
        ensure!(shape.len() == 2 && row < shape[0], "{name}: row {row}");
        let cols = shape[1];
        self.widen(&key, &data[row * cols * 2..(row + 1) * cols * 2])
    }

    /// 16-bit small tensor (norms, conv, A_log, dt_bias) widened to f32.
    pub fn f32_named(&self, name: &str) -> Result<Vec<f32>> {
        let (_shape, data) = self.view(name)?;
        self.widen(name, &data)
    }

    /// 16-bit tensor of any rank (the vision tower's conv/2-D weights),
    /// widened to f32. Shape preserved.
    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, data) = self.view(name)?;
        Ok((shape, self.widen(name, &data)?))
    }

    /// `{name}.weight` (U32 packed) + `.scales`/`.biases` (bf16) -> GroupQuant.
    pub fn quant_projection(&self, name: &str) -> Result<GroupQuant> {
        let (shape, packed) = self.view(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2, "{name}.weight is not 2-D: {shape:?}");
        let out_dim = shape[0];
        let (_s_shape, scales) = self.view(&format!("{name}.scales"))?;
        let (_b_shape, biases) = self.view(&format!("{name}.biases"))?;
        let scales = bf16_to_f32(&scales);
        let biases = bf16_to_f32(&biases);
        let packed: Vec<u32> = packed
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let groups = scales.len() / out_dim;
        let in_dim = groups * GROUP_SIZE;
        ensure!(
            shape[1] * 8 == in_dim,
            "{name}: weight width {} words implies in={} but scales imply {in_dim}",
            shape[1],
            shape[1] * 8
        );
        GroupQuant::new(packed, scales, biases, out_dim, in_dim, 4)
    }

    /// On-device bytes `{name}` occupies after upload. Int4 projections:
    /// packed weight bytes as stored, scales and biases widened bf16-to-f32.
    /// 16-bit projections (rank-2 `.weight`, no scales): raw dtype-native
    /// bytes. Small tensors widen to f32, doubling their stored bytes.
    /// Suffix-less names (A_log, dt_bias) resolve bare.
    pub fn tensor_device_bytes(&self, name: &str) -> Result<u64> {
        let suffixed = format!("{name}.weight");
        let key = if self.weights.contains(&suffixed) {
            &suffixed
        } else {
            name
        };
        let (shape, weight) = self.view(key)?;
        if self.weights.contains(&format!("{name}.scales")) {
            let (_s, scales) = self.view(&format!("{name}.scales"))?;
            let (_s, biases) = self.view(&format!("{name}.biases"))?;
            Ok(weight.len() as u64 + 2 * (scales.len() + biases.len()) as u64)
        } else if key.ends_with(".weight") && shape.len() == 2 {
            Ok(weight.len() as u64)
        } else {
            Ok(2 * weight.len() as u64)
        }
    }

    /// (out_dim, in_dim): int4 from the packed width and the scales, 16-bit
    /// straight from the tensor shape.
    pub fn projection_shape(&self, name: &str) -> Result<(usize, usize)> {
        let (shape, _packed) = self.view(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2, "{name}.weight is not 2-D: {shape:?}");
        let out_dim = shape[0];
        ensure!(out_dim > 0, "{name}.weight has zero rows");
        if !self.weights.contains(&format!("{name}.scales")) {
            return Ok((out_dim, shape[1]));
        }
        let (_s, scales) = self.view(&format!("{name}.scales"))?;
        let groups = scales.len() / 2 / out_dim;
        Ok((out_dim, groups * GROUP_SIZE))
    }

    /// One projection kept on the host for slot upload: the packed
    /// weights as a view when the mmap bytes are u32-aligned, plus the
    /// scales and biases widened to f32 once.
    pub fn host_projection(&self, name: &str) -> Result<HostProjection> {
        let (shape, packed_bytes) = self.view(&format!("{name}.weight"))?;
        ensure!(shape.len() == 2, "{name}.weight is not 2-D: {shape:?}");
        let out_dim = shape[0];
        let (_s, scales) = self.view(&format!("{name}.scales"))?;
        let (_b, biases) = self.view(&format!("{name}.biases"))?;
        let scales = bf16_bytes_to_u16(&scales);
        let biases = bf16_bytes_to_u16(&biases);
        ensure!(out_dim > 0, "{name}.weight has zero rows");
        ensure!(
            scales.len().is_multiple_of(out_dim),
            "{name}.scales has {} rows, not a multiple of out_dim {out_dim}",
            scales.len()
        );
        ensure!(
            biases.len() == scales.len(),
            "{name}.biases has {} rows but .scales has {}",
            biases.len(),
            scales.len()
        );
        let groups = scales.len() / out_dim;
        let in_dim = groups * GROUP_SIZE;
        ensure!(
            shape[1] * 8 == in_dim,
            "{name}: weight width {} words implies in={} but scales imply {in_dim}",
            shape[1],
            shape[1] * 8
        );
        ensure!(
            packed_bytes.len() % std::mem::size_of::<u32>() == 0,
            "{name}.weight byte length is not a whole u32 count"
        );
        let packed = if (packed_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>())
        {
            PackedView::Shared(packed_bytes)
        } else {
            PackedView::Owned(
                packed_bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        };
        Ok(HostProjection {
            packed,
            scales,
            biases,
            out_dim,
            in_dim,
        })
    }

    /// One 16-bit projection kept on the host for slot upload: the raw
    /// dtype-native bytes as a zero-copy view over the mmap.
    pub fn host_proj16(&self, name: &str) -> Result<HostProj16> {
        let key = format!("{name}.weight");
        let (shape, bytes) = self.view(&key)?;
        ensure!(shape.len() == 2, "{key} is not 2-D: {shape:?}");
        ensure!(
            matches!(self.dtype(&key)?, Dtype::BF16 | Dtype::F16),
            "{key}: host_proj16 needs a 16-bit dtype"
        );
        Ok(HostProj16 {
            bytes,
            out_dim: shape[0],
            in_dim: shape[1],
        })
    }
}

/// One 16-bit projection's weights on the host, ready for slot upload.
#[derive(Clone)]
pub struct HostProj16 {
    bytes: TensorBytes,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl HostProj16 {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// One row widened to f32 (the host-side embedding gather).
    pub fn row_f32(&self, format: QuantFormat, row: usize) -> Result<Vec<f32>> {
        let cols = self.in_dim;
        let bytes = &self.bytes()[row * cols * 2..(row + 1) * cols * 2];
        match format {
            QuantFormat::Bf16 => Ok(bf16_to_f32(bytes)),
            QuantFormat::F16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            QuantFormat::GroupAffine { .. } => bail!("row_f32 on an int4 projection"),
            QuantFormat::BlockFp8 { .. } => bail!("row_f32 on an FP8-block projection"),
        }
    }
}

/// The detectable format of every classifiable projection, or an error
/// naming the mixed or unknown dtypes.
/// The int4 projection format at the kernel's group width.
pub const INT4: QuantFormat = QuantFormat::GroupAffine { group: GROUP_SIZE };

fn detect_format(weights: &ModelWeights) -> Result<QuantFormat> {
    let mut format = None;
    let mut mixed = Vec::new();
    for name in weights.tensor_names() {
        let meta = weights.raw_tensor_metadata(name)?;
        if !classifiable(name, &meta.shape) {
            continue;
        }
        let kind = match meta.dtype {
            Dtype::U32 => QuantFormat::GroupAffine { group: GROUP_SIZE },
            Dtype::BF16 => QuantFormat::Bf16,
            Dtype::F16 => QuantFormat::F16,
            other => bail!("{name}: unsupported projection dtype {other:?}"),
        };
        if kind == (QuantFormat::GroupAffine { group: GROUP_SIZE }) {
            let stem = name.strip_suffix(".weight").expect("ends_with checked");
            ensure!(
                weights.contains(&format!("{stem}.scales"))
                    && weights.contains(&format!("{stem}.biases")),
                "{name}: int4 projection without scales/biases"
            );
        }
        match format {
            None => format = Some(kind),
            Some(seen) if seen == kind => {}
            Some(seen) => mixed.push(format!("{name} is {kind:?}, earlier projections {seen:?}")),
        }
    }
    ensure!(
        mixed.is_empty(),
        "mixed weight formats: {}",
        mixed.join("; ")
    );
    format.context("no classifiable projection tensors in checkpoint")
}

/// Row-parallel 16-bit matvec; `decode` widens one stored element.
fn matvec_rows(
    data: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
    decode: impl Fn(u16) -> f32 + Sync,
) -> Vec<f32> {
    let mut y = vec![0f32; rows];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(16);
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, y_chunk) in y.chunks_mut(chunk).enumerate() {
            let (data, x, decode) = (data, x, &decode);
            s.spawn(move || {
                let r0 = t * chunk;
                for (o, slot) in y_chunk.iter_mut().enumerate() {
                    let row = &data[(r0 + o) * cols * 2..(r0 + o + 1) * cols * 2];
                    let mut acc = 0f32;
                    for (c, xv) in x.iter().enumerate() {
                        let w = decode(u16::from(row[c * 2]) | (u16::from(row[c * 2 + 1]) << 8));
                        acc += w * xv;
                    }
                    *slot = acc;
                }
            });
        }
    });
    y
}

/// Packed int4 weights: a u32 view over the mmap when aligned, an owned
/// copy otherwise.
pub enum PackedView {
    /// Holds a whole number of u32-aligned words.
    Shared(TensorBytes),
    Owned(Vec<u32>),
}

impl PackedView {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::Shared(bytes) => unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4)
            },
            Self::Owned(words) => words,
        }
    }
}

/// One projection's weights on the host, ready for slot upload.
pub struct HostProjection {
    pub packed: PackedView,
    /// bf16 bits as stored in the checkpoint; the device widens in registers.
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub out_dim: usize,
    pub in_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn words(count: usize) -> Vec<u8> {
        (0..count as u32)
            .flat_map(|w| w.wrapping_mul(2654435761).to_le_bytes())
            .collect()
    }

    fn halves(count: usize, one: u16) -> Vec<u8> {
        (0..count as u16)
            .flat_map(|v| (one + v).to_le_bytes())
            .collect()
    }

    fn write(dir: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
            })
            .collect();
        let shard = "model-00001-of-00001.safetensors";
        safetensors::serialize_to_file(views, None, &dir.join(shard)).unwrap();
        let map: serde_json::Map<String, serde_json::Value> = tensors
            .iter()
            .map(|(name, ..)| (name.to_string(), shard.into()))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 0},
                "weight_map": map,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn int4_checkpoint_reads_back_the_bytes_of_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let packed = words(3 * 64);
        let scales = halves(3 * 8, 0x3f80);
        let biases = halves(3 * 8, 0x3e80);
        write(
            dir.path(),
            &[
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    Dtype::U32,
                    vec![3, 64],
                    packed.clone(),
                ),
                (
                    "model.layers.0.mlp.gate_proj.scales",
                    Dtype::BF16,
                    vec![3, 8],
                    scales.clone(),
                ),
                (
                    "model.layers.0.mlp.gate_proj.biases",
                    Dtype::BF16,
                    vec![3, 8],
                    biases.clone(),
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    Dtype::BF16,
                    vec![4],
                    halves(4, 0x3f80),
                ),
            ],
        );
        let weights = Qwen35Weights::open(dir.path()).unwrap();
        assert_eq!(
            weights.format(),
            QuantFormat::GroupAffine { group: GROUP_SIZE }
        );
        let name = "model.layers.0.mlp.gate_proj";
        let quant = weights.quant_projection(name).unwrap();
        assert_eq!(
            quant.packed,
            packed
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        );
        assert_eq!(quant.scales, bf16_to_f32(&scales));
        assert_eq!(quant.biases, bf16_to_f32(&biases));
        assert_eq!(weights.projection_shape(name).unwrap(), (3, 512));
        assert_eq!(
            weights.tensor_device_bytes(name).unwrap(),
            (packed.len() + 2 * (scales.len() + biases.len())) as u64
        );
        let host = weights.host_projection(name).unwrap();
        assert_eq!(host.packed.as_slice(), quant.packed);
        assert_eq!(host.scales, bf16_bytes_to_u16(&scales));
        let norm = weights
            .f32_named("model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(norm, bf16_to_f32(&halves(4, 0x3f80)));
    }

    #[test]
    fn sixteen_bit_checkpoint_streams_the_rows_of_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let rows = halves(2 * 64, 0x3c00);
        write(
            dir.path(),
            &[(
                "model.embed_tokens.weight",
                Dtype::F16,
                vec![2, 64],
                rows.clone(),
            )],
        );
        let weights = Qwen35Weights::open(dir.path()).unwrap();
        assert_eq!(weights.format(), QuantFormat::F16);
        let host = weights.host_proj16("model.embed_tokens").unwrap();
        assert_eq!(host.bytes(), rows.as_slice());
        let row = weights.row16("model.embed_tokens", 1).unwrap();
        assert_eq!(row, host.row_f32(QuantFormat::F16, 1).unwrap());
        let y = weights.matvec16("model.embed_tokens", &[1.0; 64]).unwrap();
        assert_eq!(y.len(), 2);
        assert_eq!(y[1], row.iter().sum::<f32>());
    }

    #[test]
    fn a_missing_tensor_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            &[(
                "model.embed_tokens.weight",
                Dtype::BF16,
                vec![2, 64],
                halves(128, 0x3f80),
            )],
        );
        let weights = Qwen35Weights::open(dir.path()).unwrap();
        let error = weights.f32_named("absent").unwrap_err();
        assert!(format!("{error:#}").contains("absent"));
    }
}
