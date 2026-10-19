//! Native routed-expert byte ranges from the checkpoint manifest.
use crate::checkpoint::TensorManifest;
use crate::model_spec::{ExpertFormat, ModelSpec};
use anyhow::{Context, Result, ensure};
use std::ops::Range;
use std::sync::Arc;

/// Byte range of one native expert slice inside a shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRange {
    pub shard: Arc<str>,
    pub offset: u64,
    pub len: u64,
}

/// One expert's native slices in one of the published checkpoint layouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertRange {
    /// Fused BF16 `[E, 2I, H]` gate-up and `[E, H, I]` down slices.
    Bf16 {
        gate_up: NativeRange,
        down: NativeRange,
    },
    /// Per-expert 2-D E4M3 weights with BF16 per-output-channel scales.
    Fp8 {
        gate: NativeRange,
        up: NativeRange,
        down: NativeRange,
        gate_scale: NativeRange,
        up_scale: NativeRange,
        down_scale: NativeRange,
    },
}

/// Byte ranges of the six FP8 slices inside one expert slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp8Slot {
    pub gate: Range<u64>,
    pub up: Range<u64>,
    pub down: Range<u64>,
    pub gate_scale: Range<u64>,
    pub up_scale: Range<u64>,
    pub down_scale: Range<u64>,
}

/// Per-expert native slot layout of the checkpoint format; every subsystem sizes from this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertLayout {
    Bf16Fused { slot_bytes: u64, gate_up_bytes: u64 },
    Fp8W8a16(Fp8Slot),
}

impl ExpertLayout {
    pub fn new(format: ExpertFormat, geometry: &ExpertGeometry) -> Result<Self> {
        ensure!(
            geometry.layers > 0
                && geometry.experts > 0
                && geometry.hidden > 0
                && geometry.intermediate > 0
                && geometry.top_k > 0
                && geometry.top_k <= geometry.experts,
            "invalid native expert geometry {geometry:?}"
        );
        let intermediate = u64::from(geometry.intermediate);
        let hidden = u64::from(geometry.hidden);
        match format {
            ExpertFormat::Bf16 => {
                let gate_up = 2u64
                    .checked_mul(intermediate)
                    .and_then(|n| n.checked_mul(hidden))
                    .and_then(|n| n.checked_mul(2))
                    .context("gate-up expert slice overflow")?;
                let down = hidden
                    .checked_mul(intermediate)
                    .and_then(|n| n.checked_mul(2))
                    .context("down expert slice overflow")?;
                let slot_bytes = gate_up.checked_add(down).context("expert slot overflow")?;
                ensure!(
                    slot_bytes.is_multiple_of(16) && gate_up.is_multiple_of(16),
                    "native slot alignment: slot_bytes={slot_bytes} gate_up_bytes={gate_up}"
                );
                Ok(Self::Bf16Fused {
                    slot_bytes,
                    gate_up_bytes: gate_up,
                })
            }
            ExpertFormat::Fp8W8a16 => {
                let weight = intermediate
                    .checked_mul(hidden)
                    .context("FP8 expert weight extent overflow")?;
                let gate = 0..weight;
                let up = weight..weight.checked_mul(2).context("FP8 up extent overflow")?;
                let down = up.end..weight.checked_mul(3).context("FP8 down extent overflow")?;
                let gate_scale = down.end
                    ..down
                        .end
                        .checked_add(
                            intermediate
                                .checked_mul(2)
                                .context("scale extent overflow")?,
                        )
                        .context("FP8 gate scale extent overflow")?;
                let up_scale = gate_scale.end
                    ..gate_scale
                        .end
                        .checked_add(
                            intermediate
                                .checked_mul(2)
                                .context("scale extent overflow")?,
                        )
                        .context("FP8 up scale extent overflow")?;
                let down_scale = up_scale.end
                    ..up_scale
                        .end
                        .checked_add(hidden.checked_mul(2).context("scale extent overflow")?)
                        .context("FP8 down scale extent overflow")?;
                ensure!(
                    down_scale.end.is_multiple_of(16),
                    "FP8 slot alignment: slot_bytes={}",
                    down_scale.end
                );
                Ok(Self::Fp8W8a16(Fp8Slot {
                    gate,
                    up,
                    down,
                    gate_scale,
                    up_scale,
                    down_scale,
                }))
            }
        }
    }

    pub fn slot_bytes(&self) -> u64 {
        match self {
            Self::Bf16Fused { slot_bytes, .. } => *slot_bytes,
            Self::Fp8W8a16(slot) => slot.down_scale.end,
        }
    }

    pub fn down_offset(&self) -> u64 {
        match self {
            Self::Bf16Fused { gate_up_bytes, .. } => *gate_up_bytes,
            Self::Fp8W8a16(_) => 0,
        }
    }

    pub fn fp8_slot(&self) -> Option<&Fp8Slot> {
        match self {
            Self::Bf16Fused { .. } => None,
            Self::Fp8W8a16(slot) => Some(slot),
        }
    }
}

/// Routed-expert geometry of the text tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
}

/// Native ranges of every routable expert, in manifest order.
#[derive(Debug)]
pub struct ExpertCatalogue {
    geometry: ExpertGeometry,
    layout: ExpertLayout,
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

fn expert_entry<'a>(
    manifest: &'a TensorManifest,
    layer: u32,
    expert: u32,
    role: &str,
) -> Result<&'a crate::checkpoint::TensorEntry> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{role}");
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

/// One whole per-expert tensor as a native range, validated against its manifest dtype and shape.
fn native_entry(
    entry: &crate::checkpoint::TensorEntry,
    dtype: safetensors::Dtype,
    shape: &[usize],
) -> Result<NativeRange> {
    ensure!(
        entry.dtype == dtype,
        "expert tensor {} is {:?}, expected {dtype:?}",
        entry.role,
        entry.dtype
    );
    ensure!(
        entry.shape == shape,
        "expert tensor {} has shape {:?}, expected {shape:?}",
        entry.role,
        entry.shape
    );
    let len = entry.data_offsets[1]
        .checked_sub(entry.data_offsets[0])
        .context("expert tensor offset underflow")?;
    let elements = shape
        .iter()
        .try_fold(1u64, |n, &dim| n.checked_mul(dim as u64))
        .context("expert tensor extent overflow")?;
    let expected = elements
        .checked_mul(match dtype {
            safetensors::Dtype::BF16 => 2,
            safetensors::Dtype::F8_E4M3 => 1,
            other => anyhow::bail!("unsupported expert scale dtype {other:?}"),
        })
        .context("expert tensor byte extent overflow")?;
    ensure!(
        len == expected,
        "expert tensor {} holds {len} bytes, expected {expected}",
        entry.role
    );
    Ok(NativeRange {
        shard: Arc::from(entry.shard.as_str()),
        offset: entry
            .data_start
            .checked_add(entry.data_offsets[0])
            .context("expert tensor start overflow")?,
        len,
    })
}

impl ExpertCatalogue {
    pub fn from_manifest(
        manifest: &TensorManifest,
        geometry: ExpertGeometry,
        format: ExpertFormat,
    ) -> Result<Self> {
        let layout = ExpertLayout::new(format, &geometry)?;
        let mut ranges = Vec::with_capacity(geometry.layers as usize);
        for layer in 0..geometry.layers {
            match layout {
                ExpertLayout::Bf16Fused { gate_up_bytes, .. } => {
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
                    let gate = layer_entry(manifest, layer, "gate_up_proj")?;
                    let down = layer_entry(manifest, layer, "down_proj")?;
                    let gate_ranges =
                        slice_ranges(gate, geometry.experts, &gate_shape, gate_up_bytes)?;
                    let down_ranges = slice_ranges(
                        down,
                        geometry.experts,
                        &down_shape,
                        layout.slot_bytes() - gate_up_bytes,
                    )?;
                    ranges.push(
                        gate_ranges
                            .into_iter()
                            .zip(down_ranges)
                            .map(|(gate_up, down)| ExpertRange::Bf16 { gate_up, down })
                            .collect(),
                    );
                }
                ExpertLayout::Fp8W8a16(_) => {
                    let hidden = geometry.hidden as usize;
                    let intermediate = geometry.intermediate as usize;
                    let scales = geometry.experts as usize;
                    let mut layer_ranges = Vec::with_capacity(scales);
                    for expert in 0..geometry.experts {
                        let weight = |role: &str, shape: &[usize]| {
                            native_entry(
                                expert_entry(manifest, layer, expert, role)?,
                                safetensors::Dtype::F8_E4M3,
                                shape,
                            )
                        };
                        let scale = |role: &str, rows: usize| {
                            native_entry(
                                expert_entry(manifest, layer, expert, role)?,
                                safetensors::Dtype::BF16,
                                &[rows, 1],
                            )
                        };
                        layer_ranges.push(ExpertRange::Fp8 {
                            gate: weight("gate_proj.weight", &[intermediate, hidden])?,
                            up: weight("up_proj.weight", &[intermediate, hidden])?,
                            down: weight("down_proj.weight", &[hidden, intermediate])?,
                            gate_scale: scale("gate_proj.weight_scale", intermediate)?,
                            up_scale: scale("up_proj.weight_scale", intermediate)?,
                            down_scale: scale("down_proj.weight_scale", hidden)?,
                        });
                    }
                    ranges.push(layer_ranges);
                }
            }
        }
        Ok(Self {
            geometry,
            layout,
            ranges,
        })
    }

    pub fn geometry(&self) -> ExpertGeometry {
        self.geometry
    }

    pub fn layout(&self) -> &ExpertLayout {
        &self.layout
    }

    pub fn slot_bytes(&self) -> u64 {
        self.layout.slot_bytes()
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
    use anyhow::bail;
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

    fn typed(
        role: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        shard: &str,
        start: u64,
        len: u64,
    ) -> TensorEntry {
        TensorEntry {
            dtype,
            ..entry(role, shape, shard, start, len)
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

    /// FP8 manifest: per-expert 2-D weights and BF16 scales with distinct offsets per role.
    fn fp8_manifest() -> TensorManifest {
        let hidden = 8u64;
        let intermediate = 4u64;
        let mut entries = BTreeMap::new();
        let mut next = 0u64;
        for layer in 0..2u32 {
            for expert in 0..4u32 {
                let name = |role: &str| {
                    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{role}")
                };
                for (role, rows, cols, dtype) in [
                    (
                        "gate_proj.weight",
                        intermediate,
                        hidden,
                        safetensors::Dtype::F8_E4M3,
                    ),
                    (
                        "up_proj.weight",
                        intermediate,
                        hidden,
                        safetensors::Dtype::F8_E4M3,
                    ),
                    (
                        "down_proj.weight",
                        hidden,
                        intermediate,
                        safetensors::Dtype::F8_E4M3,
                    ),
                ] {
                    let len = rows * cols;
                    entries.insert(
                        name(role),
                        typed(
                            role,
                            dtype,
                            vec![rows as usize, cols as usize],
                            "fp8-0.safetensors",
                            next,
                            len,
                        ),
                    );
                    next += len;
                }
                for (role, rows) in [
                    ("gate_proj.weight_scale", intermediate),
                    ("up_proj.weight_scale", intermediate),
                    ("down_proj.weight_scale", hidden),
                ] {
                    let len = rows * 2;
                    entries.insert(
                        name(role),
                        typed(
                            role,
                            safetensors::Dtype::BF16,
                            vec![rows as usize, 1],
                            "fp8-0.safetensors",
                            next,
                            len,
                        ),
                    );
                    next += len;
                }
            }
        }
        TensorManifest {
            entries,
            resident_bytes: 0,
        }
    }

    #[test]
    fn slice_offsets_cover_each_expert_exactly() -> Result<()> {
        for bad in [
            ExpertGeometry {
                layers: 0,
                ..geometry()
            },
            ExpertGeometry {
                experts: 0,
                ..geometry()
            },
            ExpertGeometry {
                hidden: 0,
                ..geometry()
            },
            ExpertGeometry {
                intermediate: 0,
                ..geometry()
            },
            ExpertGeometry {
                top_k: 0,
                ..geometry()
            },
        ] {
            for format in [ExpertFormat::Bf16, ExpertFormat::Fp8W8a16] {
                assert!(ExpertLayout::new(format, &bad).is_err(), "{bad:?}");
            }
        }
        let catalogue =
            ExpertCatalogue::from_manifest(&manifest(), geometry(), ExpertFormat::Bf16)?;
        let first = match catalogue.range(0, 0)? {
            ExpertRange::Bf16 { gate_up, down } => (gate_up, down),
            _ => bail!("BF16 catalogue produced an FP8 range"),
        };
        assert_eq!(first.0.offset, 64);
        assert_eq!(first.0.len, 2 * 4 * 8 * 2);
        assert_eq!(first.1.offset, 64 + 4 * (2 * 4 * 8 * 2));
        let last_gate = match catalogue.range(1, 3)? {
            ExpertRange::Bf16 { gate_up, .. } => gate_up.clone(),
            _ => bail!("BF16 range expected"),
        };
        assert_eq!(last_gate.offset + last_gate.len, 64 + 768 + 512);
        let layer = match catalogue.range(1, 1)? {
            ExpertRange::Bf16 { gate_up, .. } => gate_up.clone(),
            _ => bail!("BF16 range expected"),
        };
        assert_eq!(layer.offset, 64 + 768 + 128);
        assert_eq!(layer.shard.as_ref(), "shard-0.safetensors");
        assert_eq!(catalogue.geometry(), geometry());
        assert_eq!(catalogue.slot_bytes(), 192);
        assert!(catalogue.range(2, 0).is_err());
        assert!(catalogue.range(0, 4).is_err());
        Ok(())
    }

    #[test]
    fn fp8_ranges_use_the_per_expert_layout() -> Result<()> {
        let catalogue =
            ExpertCatalogue::from_manifest(&fp8_manifest(), geometry(), ExpertFormat::Fp8W8a16)?;
        assert_eq!(catalogue.slot_bytes(), 128);
        let layout = match catalogue.layout() {
            ExpertLayout::Fp8W8a16(slot) => slot,
            _ => bail!("FP8 layout expected"),
        };
        assert_eq!((layout.gate.start, layout.gate.end), (0, 32));
        assert_eq!((layout.up.start, layout.up.end), (32, 64));
        assert_eq!((layout.down.start, layout.down.end), (64, 96));
        assert_eq!((layout.gate_scale.start, layout.gate_scale.end), (96, 104));
        assert_eq!((layout.up_scale.start, layout.up_scale.end), (104, 112));
        assert_eq!((layout.down_scale.start, layout.down_scale.end), (112, 128));
        let range = catalogue.range(0, 1)?;
        let ExpertRange::Fp8 {
            gate,
            up,
            down,
            gate_scale,
            up_scale,
            down_scale,
        } = range
        else {
            bail!("FP8 range expected");
        };
        let expert = 128u64;
        assert_eq!(gate.offset, 64 + expert);
        assert_eq!(up.offset, 64 + expert + 32);
        assert_eq!(down.offset, 64 + expert + 64);
        assert_eq!(gate_scale.offset, 64 + expert + 96);
        assert_eq!(up_scale.offset, 64 + expert + 104);
        assert_eq!(down_scale.offset, 64 + expert + 112);
        assert_eq!(down_scale.len, 16);
        Ok(())
    }

    #[test]
    fn manifest_mismatches_name_the_tensor() -> Result<()> {
        let mut bad = manifest();
        let name = "model.language_model.layers.1.mlp.experts.down_proj";
        bad.entries.get_mut(name).unwrap().shape = vec![4, 4, 8];
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Bf16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mlp.experts.down_proj"), "{error}");
        let mut bad = manifest();
        bad.entries
            .remove("model.language_model.layers.0.mlp.experts.gate_up_proj");
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Bf16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("gate_up_proj"), "{error}");
        let mut bad = manifest();
        let name = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        bad.entries.get_mut(name).unwrap().dtype = safetensors::Dtype::F32;
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Bf16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected BF16"), "{error}");
        let mut bad = fp8_manifest();
        let name = "model.language_model.layers.0.mlp.experts.2.gate_proj.weight_scale";
        bad.entries.get_mut(name).unwrap().dtype = safetensors::Dtype::F32;
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Fp8W8a16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("weight_scale"), "{error}");
        let mut bad = fp8_manifest();
        bad.entries
            .remove("model.language_model.layers.1.mlp.experts.3.down_proj.weight");
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Fp8W8a16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("down_proj.weight"), "{error}");
        let mut bad = manifest();
        bad.entries
            .remove("model.language_model.layers.0.mlp.experts.gate_up_proj");
        let error = ExpertCatalogue::from_manifest(&bad, geometry(), ExpertFormat::Fp8W8a16)
            .unwrap_err()
            .to_string();
        assert!(error.contains("gate_proj.weight"), "{error}");
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL"]
    fn catalogue_matches_the_published_checkpoint() -> Result<()> {
        use std::os::unix::fs::FileExt;
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let model = ModelSpec::read(&dir)?;
        ensure!(
            model.expert_format == Some(ExpertFormat::Bf16),
            "FF_MODEL is not the BF16 checkpoint"
        );
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
        let mapped = crate::checkpoint::MappedCheckpoint::open(&dir)?;
        let manifest = mapped.manifest(&model, crate::checkpoint::LoadPolicy::TextOnly)?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            model.expert_format.context("MoE expert_format missing")?,
        )?;
        assert_eq!(catalogue.slot_bytes(), 6_291_456);
        for (layer, expert) in [(0u32, 0u32), (0, 255), (39, 0), (39, 255)] {
            let range = catalogue.range(layer, expert)?;
            let gate_up = match range {
                ExpertRange::Bf16 { gate_up, .. } => gate_up,
                _ => bail!("BF16 range expected"),
            };
            let file = std::fs::File::open(dir.join(gate_up.shard.as_ref()))?;
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let tensors = safetensors::SafeTensors::deserialize(&map)?;
            let name = format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj");
            let tensor = tensors.tensor(&name)?;
            let data = tensor.data();
            let slice = expert as usize * gate_up.len as usize;
            let mut head = [0u8; 64];
            file.read_exact_at(&mut head, gate_up.offset)?;
            assert_eq!(
                &head[..],
                &data[slice..slice + 64],
                "layer {layer} expert {expert}"
            );
            let entry = manifest.entries.get(&name).context("gate-up entry")?;
            if expert == 255 {
                assert_eq!(
                    gate_up.offset + gate_up.len,
                    entry.data_start + entry.data_offsets[1]
                );
            }
        }
        Ok(())
    }

    /// Expert-only manifest of the published FP8 checkpoint, built from the shard headers.
    fn fp8_expert_manifest(
        dir: &std::path::Path,
        geometry: &ExpertGeometry,
    ) -> Result<TensorManifest> {
        use crate::checkpoint::{Index, TensorEntry};
        let index: Index =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let mut maps: BTreeMap<String, (memmap2::Mmap, u64)> = BTreeMap::new();
        for shard in index.weight_map.values() {
            if maps.contains_key(shard) {
                continue;
            }
            let file = std::fs::File::open(dir.join(shard))?;
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let hlen = u64::from_le_bytes(map[..8].try_into()?);
            maps.insert(shard.clone(), (map, 8 + hlen));
        }
        let mut views = BTreeMap::new();
        for (shard, (map, _)) in &maps {
            views.insert(shard.as_str(), safetensors::SafeTensors::deserialize(map)?);
        }
        let mut entries = BTreeMap::new();
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                for role in [
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                    "gate_proj.weight_scale",
                    "up_proj.weight_scale",
                    "down_proj.weight_scale",
                ] {
                    let name =
                        format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{role}");
                    let shard = index
                        .weight_map
                        .get(&name)
                        .with_context(|| format!("index entry missing: {name}"))?;
                    let (map, data_start) = &maps[shard];
                    let tensor = views[shard.as_str()].tensor(&name)?;
                    let file_offset = (tensor.data().as_ptr() as usize)
                        .checked_sub(map.as_ptr() as usize)
                        .context("tensor outside mapping")?
                        as u64;
                    let start = file_offset
                        .checked_sub(*data_start)
                        .context("tensor overlaps header")?;
                    let end = start + tensor.data().len() as u64;
                    entries.insert(
                        name,
                        TensorEntry {
                            dtype: tensor.dtype(),
                            shape: tensor.shape().to_vec(),
                            shard: shard.clone(),
                            bytes: tensor.data().len(),
                            load: true,
                            data_start: *data_start,
                            data_offsets: [start, end],
                            scope: crate::checkpoint::TensorScope::Text,
                            layer: Some(layer),
                            role: role.into(),
                            scale_name: None,
                        },
                    );
                }
            }
        }
        Ok(TensorManifest {
            entries,
            resident_bytes: 0,
        })
    }

    #[test]
    #[ignore = "CPU-only real checkpoint; set FF_MODEL"]
    fn fp8_catalogue_matches_the_published_checkpoint() -> Result<()> {
        use std::os::unix::fs::FileExt;
        let dir =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let model = ModelSpec::read(&dir)?;
        ensure!(
            model.expert_format == Some(ExpertFormat::Fp8W8a16),
            "FF_MODEL is not the FP8 checkpoint"
        );
        let geometry = ExpertGeometry::from_model(&model)?;
        let manifest = fp8_expert_manifest(&dir, &geometry)?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            model.expert_format.context("MoE expert_format missing")?,
        )?;
        assert_eq!(catalogue.slot_bytes(), 3_151_872);
        assert_eq!(
            catalogue.layout(),
            &ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?
        );
        for (layer, expert) in [(0u32, 0u32), (0, 255), (39, 0), (39, 255)] {
            let range = catalogue.range(layer, expert)?;
            let ExpertRange::Fp8 {
                gate,
                up,
                down,
                gate_scale,
                up_scale,
                down_scale,
            } = range
            else {
                bail!("FP8 range expected");
            };
            for (role, native) in [
                ("gate_proj.weight", gate),
                ("up_proj.weight", up),
                ("down_proj.weight", down),
                ("gate_proj.weight_scale", gate_scale),
                ("up_proj.weight_scale", up_scale),
                ("down_proj.weight_scale", down_scale),
            ] {
                let name =
                    format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{role}");
                let entry = manifest.entries.get(&name).context("expert entry")?;
                assert_eq!(
                    native.offset,
                    entry.data_start + entry.data_offsets[0],
                    "{name} offset"
                );
                assert_eq!(
                    native.offset + native.len,
                    entry.data_start + entry.data_offsets[1],
                    "{name} end"
                );
                assert_eq!(native.shard.as_ref(), entry.shard.as_str(), "{name} shard");
                let file = std::fs::File::open(dir.join(native.shard.as_ref()))?;
                let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&map)?;
                let tensor = tensors.tensor(&name)?;
                let data = tensor.data();
                let head = &data[..64.min(data.len())];
                let mut read = vec![0u8; head.len()];
                file.read_exact_at(&mut read, native.offset)?;
                assert_eq!(read, head, "{name} head");
            }
        }
        Ok(())
    }
}
