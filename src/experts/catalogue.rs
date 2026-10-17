//! Native routed-expert byte ranges from the checkpoint manifest.
use crate::checkpoint::TensorManifest;
use crate::model_spec::ModelSpec;
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

/// Byte range of one native expert slice inside a shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRange {
    pub shard: Arc<str>,
    pub offset: u64,
    pub len: u64,
}

/// One expert's gate-up and down native slices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertRange {
    pub gate_up: NativeRange,
    pub down: NativeRange,
}

/// Routed-expert geometry of the text tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertGeometry {
    pub layers: u32,
    pub experts: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub top_k: u32,
}

impl ExpertGeometry {
    pub fn from_model(model: &ModelSpec) -> Result<Self> {
        let moe = model.moe.as_ref().context("model has no routed experts")?;
        let layers = u32::try_from(model.layers.len()).context("layer count exceeds u32")?;
        ensure!(
            layers > 0 && moe.experts > 0 && moe.experts_per_tok > 0,
            "expert geometry has a zero extent"
        );
        ensure!(
            moe.experts_per_tok <= moe.experts,
            "top_k {} exceeds the {} experts",
            moe.experts_per_tok,
            moe.experts
        );
        ensure!(
            model.hidden > 0 && moe.expert_intermediate > 0,
            "expert geometry has a zero dimension"
        );
        Ok(Self {
            layers,
            experts: moe.experts,
            hidden: model.hidden,
            intermediate: moe.expert_intermediate,
            top_k: moe.experts_per_tok,
        })
    }

    /// Bytes of one expert's fused `[2I, H]` gate-up slice.
    pub fn gate_up_bytes(&self) -> Result<u64> {
        2u64.checked_mul(u64::from(self.intermediate))
            .and_then(|n| n.checked_mul(u64::from(self.hidden)))
            .and_then(|n| n.checked_mul(2))
            .context("gate-up expert slice overflow")
    }

    /// Bytes of one expert's `[H, I]` down slice.
    pub fn down_bytes(&self) -> Result<u64> {
        u64::from(self.hidden)
            .checked_mul(u64::from(self.intermediate))
            .and_then(|n| n.checked_mul(2))
            .context("down expert slice overflow")
    }

    /// Bytes of one expert's native slices (gate-up plus down).
    pub fn slot_bytes(&self) -> Result<u64> {
        self.gate_up_bytes()?
            .checked_add(self.down_bytes()?)
            .context("expert slot overflow")
    }
}

/// Native ranges of every routable expert, in manifest order.
#[derive(Debug)]
pub struct ExpertCatalogue {
    geometry: ExpertGeometry,
    ranges: Vec<Vec<ExpertRange>>,
}

fn layer_entry<'a>(
    manifest: &'a TensorManifest,
    layer: u32,
    role: &str,
) -> Result<&'a crate::checkpoint::TensorEntry> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.{role}");
    manifest
        .entries
        .get(&name)
        .with_context(|| format!("expert tensor missing: {name}"))
}

fn slice_ranges(
    entry: &crate::checkpoint::TensorEntry,
    experts: u32,
    shape: &[usize],
    slice_bytes: u64,
) -> Result<Vec<NativeRange>> {
    ensure!(
        entry.dtype == safetensors::Dtype::BF16,
        "expert tensor {} is {:?}, expected BF16",
        entry.role,
        entry.dtype
    );
    ensure!(
        entry.shape == shape,
        "expert tensor {} has shape {:?}, expected {shape:?}",
        entry.role,
        entry.shape
    );
    let tensor_len = entry.data_offsets[1]
        .checked_sub(entry.data_offsets[0])
        .context("expert tensor offset underflow")?;
    let expected = u64::from(experts)
        .checked_mul(slice_bytes)
        .context("expert slice extent overflow")?;
    ensure!(
        tensor_len == expected,
        "expert tensor {} holds {tensor_len} bytes, expected {expected}",
        entry.role
    );
    let start = entry
        .data_start
        .checked_add(entry.data_offsets[0])
        .context("expert tensor start overflow")?;
    let shard: Arc<str> = Arc::from(entry.shard.as_str());
    let mut ranges = Vec::with_capacity(experts as usize);
    for expert in 0..experts {
        let offset = u64::from(expert)
            .checked_mul(slice_bytes)
            .and_then(|n| n.checked_add(start))
            .context("expert slice offset overflow")?;
        let end = offset
            .checked_add(slice_bytes)
            .context("expert slice end overflow")?;
        ensure!(
            end <= entry.data_start + entry.data_offsets[1],
            "expert {expert} of {} exceeds the tensor",
            entry.role
        );
        ranges.push(NativeRange {
            shard: shard.clone(),
            offset,
            len: slice_bytes,
        });
    }
    ensure!(
        ranges.last().is_some_and(
            |range| range.offset + range.len == entry.data_start + entry.data_offsets[1]
        ),
        "expert slices of {} do not end at the tensor end",
        entry.role
    );
    Ok(ranges)
}

impl ExpertCatalogue {
    pub fn from_manifest(manifest: &TensorManifest, geometry: ExpertGeometry) -> Result<Self> {
        let gate_up_bytes = geometry.gate_up_bytes()?;
        let down_bytes = geometry.down_bytes()?;
        let gate_shape = [
            geometry.experts as usize,
            2 * geometry.intermediate as usize,
            geometry.hidden as usize,
        ];
        let down_shape = [
            geometry.experts as usize,
            geometry.hidden as usize,
            geometry.intermediate as usize,
        ];
        let mut ranges = Vec::with_capacity(geometry.layers as usize);
        for layer in 0..geometry.layers {
            let gate = layer_entry(manifest, layer, "gate_up_proj")?;
            let down = layer_entry(manifest, layer, "down_proj")?;
            let gate_ranges = slice_ranges(gate, geometry.experts, &gate_shape, gate_up_bytes)?;
            let down_ranges = slice_ranges(down, geometry.experts, &down_shape, down_bytes)?;
            ranges.push(
                gate_ranges
                    .into_iter()
                    .zip(down_ranges)
                    .map(|(gate_up, down)| ExpertRange { gate_up, down })
                    .collect(),
            );
        }
        Ok(Self { geometry, ranges })
    }

    pub fn geometry(&self) -> ExpertGeometry {
        self.geometry
    }

    pub fn range(&self, layer: u32, expert: u32) -> Result<&ExpertRange> {
        ensure!(
            layer < self.geometry.layers,
            "expert layer {layer} outside {}",
            self.geometry.layers
        );
        ensure!(
            expert < self.geometry.experts,
            "expert {expert} outside {}",
            self.geometry.experts
        );
        Ok(&self.ranges[layer as usize][expert as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorManifest};
    use std::collections::BTreeMap;

    fn entry(role: &str, shape: Vec<usize>, shard: &str, start: u64, len: u64) -> TensorEntry {
        TensorEntry {
            dtype: safetensors::Dtype::BF16,
            shape,
            shard: shard.into(),
            bytes: len as usize,
            load: true,
            data_start: 64,
            data_offsets: [start, start + len],
            scope: crate::checkpoint::TensorScope::Text,
            layer: None,
            role: role.into(),
            scale_name: None,
        }
    }

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            layers: 2,
            experts: 4,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn manifest() -> TensorManifest {
        let gate = (4 * 8 * 8 * 2) as u64;
        let down = (4 * 8 * 4 * 2) as u64;
        let mut entries = BTreeMap::new();
        for layer in 0..2u32 {
            let gate_start = u64::from(layer) * (gate + down);
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                entry(
                    "mlp.experts.gate_up_proj",
                    vec![4, 8, 8],
                    "shard-0.safetensors",
                    gate_start,
                    gate,
                ),
            );
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                entry(
                    "mlp.experts.down_proj",
                    vec![4, 8, 4],
                    "shard-0.safetensors",
                    gate_start + gate,
                    down,
                ),
            );
        }
        TensorManifest {
            entries,
            resident_bytes: 0,
        }
    }

    #[test]
    fn slice_offsets_cover_each_expert_exactly() -> Result<()> {
        let catalogue = ExpertCatalogue::from_manifest(&manifest(), geometry())?;
        let first = catalogue.range(0, 0)?;
        assert_eq!(first.gate_up.offset, 64);
        assert_eq!(first.gate_up.len, 2 * 4 * 8 * 2);
        assert_eq!(first.down.offset, 64 + 4 * (2 * 4 * 8 * 2));
        let last_gate = catalogue.range(1, 3)?.gate_up.clone();
        assert_eq!(last_gate.offset + last_gate.len, 64 + 768 + 512);
        let layer = catalogue.range(1, 1)?;
        assert_eq!(layer.gate_up.offset, 64 + 768 + 128);
        assert_eq!(layer.gate_up.shard.as_ref(), "shard-0.safetensors");
        assert_eq!(catalogue.geometry(), geometry());
        assert!(catalogue.range(2, 0).is_err());
        assert!(catalogue.range(0, 4).is_err());
        Ok(())
    }

    #[test]
    fn manifest_mismatches_name_the_tensor() -> Result<()> {
        let mut bad = manifest();
        let name = "model.language_model.layers.1.mlp.experts.down_proj";
        bad.entries.get_mut(name).unwrap().shape = vec![4, 4, 8];
        let error = ExpertCatalogue::from_manifest(&bad, geometry())
            .unwrap_err()
            .to_string();
        assert!(error.contains("mlp.experts.down_proj"), "{error}");
        let mut bad = manifest();
        bad.entries
            .remove("model.language_model.layers.0.mlp.experts.gate_up_proj");
        let error = ExpertCatalogue::from_manifest(&bad, geometry())
            .unwrap_err()
            .to_string();
        assert!(error.contains("gate_up_proj"), "{error}");
        let mut bad = manifest();
        let name = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        bad.entries.get_mut(name).unwrap().dtype = safetensors::Dtype::F32;
        let error = ExpertCatalogue::from_manifest(&bad, geometry())
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected BF16"), "{error}");
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL"]
    fn catalogue_matches_the_published_checkpoint() -> Result<()> {
        use std::os::unix::fs::FileExt;
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let model = ModelSpec::read(&dir)?;
        let geometry = ExpertGeometry::from_model(&model)?;
        assert_eq!(
            (
                geometry.layers,
                geometry.experts,
                geometry.hidden,
                geometry.intermediate,
                geometry.top_k
            ),
            (40, 256, 2048, 512, 8)
        );
        assert_eq!(geometry.slot_bytes()?, 6_291_456);
        let mapped = crate::checkpoint::MappedCheckpoint::open(&dir)?;
        let manifest = mapped.manifest(&model, crate::checkpoint::LoadPolicy::TextOnly)?;
        let catalogue = ExpertCatalogue::from_manifest(&manifest, geometry)?;
        for (layer, expert) in [(0u32, 0u32), (0, 255), (39, 0), (39, 255)] {
            let range = catalogue.range(layer, expert)?;
            let file = std::fs::File::open(dir.join(range.gate_up.shard.as_ref()))?;
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let tensors = safetensors::SafeTensors::deserialize(&map)?;
            let name = format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj");
            let tensor = tensors.tensor(&name)?;
            let data = tensor.data();
            let slice = expert as usize * range.gate_up.len as usize;
            let mut head = [0u8; 64];
            file.read_exact_at(&mut head, range.gate_up.offset)?;
            assert_eq!(
                &head[..],
                &data[slice..slice + 64],
                "layer {layer} expert {expert}"
            );
            let entry = manifest.entries.get(&name).context("gate-up entry")?;
            if expert == 255 {
                assert_eq!(
                    range.gate_up.offset + range.gate_up.len,
                    entry.data_start + entry.data_offsets[1]
                );
            }
        }
        Ok(())
    }
}
