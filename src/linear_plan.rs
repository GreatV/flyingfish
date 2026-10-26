//! Checked single-layer materialization layout, shared by planner and Linear.
use crate::{checkpoint::TensorManifest, model_spec::LinearFormat};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use std::collections::BTreeMap;

/// Activation tile width of gemm_skinny_bf16.
pub const MAX_SKINNY_ROWS: usize = 16;

#[derive(Debug)]
pub struct DequantSlot {
    pub offset_bytes: u64,
    pub n: u32,
    pub k: u32,
    pub scale_name: String,
}
#[derive(Debug)]
pub struct LayerDequantPlan {
    pub layer: u32,
    pub prefix: String,
    pub bytes: u64,
    pub slots: BTreeMap<String, DequantSlot>,
}
impl LayerDequantPlan {
    pub fn from_manifest(
        manifest: &TensorManifest,
        layer: u32,
        format: LinearFormat,
    ) -> Result<Self> {
        let language = format!("model.language_model.layers.{layer}.");
        let model = format!("model.layers.{layer}.");
        let language_count = manifest
            .entries
            .keys()
            .filter(|name| name.starts_with(&language))
            .count();
        let model_count = manifest
            .entries
            .keys()
            .filter(|name| name.starts_with(&model))
            .count();
        ensure!(
            language_count > 0 || model_count > 0,
            "missing linear layer {layer}"
        );
        ensure!(
            language_count == 0 || model_count == 0,
            "layer {layer} has both model and language_model tensors"
        );
        let prefix = if language_count > 0 { language } else { model };
        Self::build(manifest, layer, prefix, format)
    }

    pub fn mtp(manifest: &TensorManifest) -> Result<Self> {
        Self::build(
            manifest,
            0,
            "mtp.layers.0.".into(),
            LinearFormat::fp8_block([128, 128], Dtype::BF16)?,
        )
    }

    fn build(
        manifest: &TensorManifest,
        layer: u32,
        prefix: String,
        format: LinearFormat,
    ) -> Result<Self> {
        let mut bytes = 0u64;
        let mut slots = BTreeMap::new();
        for (name, e) in &manifest.entries {
            if !e.load
                || e.is_routed_expert()
                || e.dtype != Dtype::F8_E4M3
                || !name.starts_with(&prefix)
            {
                continue;
            }
            ensure!(e.shape.len() == 2, "linear weight is not a matrix");
            let n = u32::try_from(e.shape[0])?;
            let k = u32::try_from(e.shape[1])?;
            let layout = format.fp8_layout().context("FP8 scale layout missing")?;
            let scale_name = format!(
                "{}.{}",
                name.strip_suffix(".weight").context("FP8 weight name")?,
                layout.scale_suffix()
            );
            let scale = manifest
                .entries
                .get(&scale_name)
                .context("FP8 weight missing scale")?;
            let scale_dtype = layout.scale_dtype();
            if let Some(block) = layout.block_scale() {
                block.validate_scale(n, k, scale.dtype, &scale.shape, scale.bytes)?;
            } else {
                ensure!(
                    scale.dtype == scale_dtype && scale.shape == layout.scale_shape(n, k),
                    "FP8 scale layout mismatch"
                );
            }
            ensure!(
                e.scale_name.as_ref() == Some(&scale_name),
                "FP8 scale name mismatch"
            );
            let size = u64::from(n)
                .checked_mul(u64::from(k))
                .and_then(|v| v.checked_mul(2))
                .context("dequant size overflow")?;
            let offset_bytes = bytes
                .checked_add(255)
                .context("dequant alignment overflow")?
                & !255;
            bytes = offset_bytes
                .checked_add(size)
                .context("layer arena overflow")?;
            slots.insert(
                name.clone(),
                DequantSlot {
                    offset_bytes,
                    n,
                    k,
                    scale_name,
                },
            );
        }
        ensure!(
            manifest.entries.iter().any(|(name, entry)| {
                entry.load
                    && !entry.is_routed_expert()
                    && name.starts_with(&prefix)
                    && name.ends_with(".weight")
                    && entry.shape.len() == 2
                    && matches!(entry.dtype, Dtype::BF16 | Dtype::F8_E4M3)
            }),
            "missing linear layer {layer}"
        );
        let plan_shape = match format {
            LinearFormat::Fp8E4m3Block { .. } => !slots.is_empty(),
            LinearFormat::Bf16 => slots.is_empty(),
            LinearFormat::Fp8E4m3Channel => true,
        };
        ensure!(
            plan_shape,
            "linear layer={layer} has {} FP8 slots for linear_format={format:?}",
            slots.len()
        );
        bytes = bytes.checked_add(255).context("arena size overflow")? & !255;
        Ok(Self {
            layer,
            prefix,
            bytes,
            slots,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorScope};

    #[test]
    fn layer_materialization_checks_native_storage_and_scale_layouts() -> Result<()> {
        let entry = TensorEntry {
            dtype: Dtype::BF16,
            shape: vec![8192, 2048],
            shard: "native.safetensors".into(),
            bytes: 8192 * 2048 * 2,
            load: true,
            data_start: 0,
            data_offsets: [0, 8192 * 2048 * 2],
            scope: TensorScope::Text,
            layer: Some(0),
            role: "linear_attn.in_proj_qkv.weight".into(),
            scale_name: None,
        };
        let mut manifest = TensorManifest {
            entries: [(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into(),
                entry,
            )]
            .into(),
            resident_bytes: 8192 * 2048 * 2,
        };
        let plan = LayerDequantPlan::from_manifest(&manifest, 0, LinearFormat::Bf16)?;
        let error = LayerDequantPlan::from_manifest(
            &manifest,
            0,
            LinearFormat::fp8_block([128, 128], Dtype::BF16)?,
        )
        .unwrap_err();
        assert!(error.to_string().contains("0 FP8 slots"), "{error}");
        assert_eq!(plan.bytes, 0);
        assert!(plan.slots.is_empty());
        assert!(LayerDequantPlan::from_manifest(&manifest, 1, LinearFormat::Bf16).is_err());
        let original = manifest.entries.keys().next().unwrap().clone();
        let mut ignored = manifest.entries.remove(&original).unwrap();
        let role = ignored.role.clone();
        ignored.role = "self_attn.q_proj.weight".into();
        let name = "model.language_model.layers.0.self_attn.q_proj.weight".to_owned();
        manifest.entries.insert(name.clone(), ignored);
        let plan = LayerDequantPlan::from_manifest(&manifest, 0, LinearFormat::Fp8E4m3Channel)?;
        assert_eq!(plan.bytes, 0);
        let mut ignored = manifest.entries.remove(&name).unwrap();
        ignored.role = role;
        manifest.entries.insert(original, ignored);
        let name = manifest.entries.keys().next().unwrap().clone();
        for format in [
            LinearFormat::fp8_block([128, 128], Dtype::BF16)?,
            LinearFormat::Fp8E4m3Channel,
        ] {
            let layout = format.fp8_layout().unwrap();
            let scale_name = format!(
                "{}.{}",
                name.strip_suffix(".weight").unwrap(),
                layout.scale_suffix()
            );
            let weight = manifest.entries.get_mut(&name).unwrap();
            weight.dtype = Dtype::F8_E4M3;
            weight.bytes = 8192 * 2048;
            weight.data_offsets = [0, weight.bytes as u64];
            weight.scale_name = Some(scale_name.clone());
            let shape = layout.scale_shape(8192, 2048);
            let bytes = shape.iter().product::<usize>() * 2;
            manifest.entries.insert(
                scale_name.clone(),
                TensorEntry {
                    dtype: Dtype::BF16,
                    shape: shape.to_vec(),
                    shard: "native.safetensors".into(),
                    bytes,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope: TensorScope::Text,
                    layer: Some(0),
                    role: "linear_attn.in_proj_qkv.scale".into(),
                    scale_name: None,
                },
            );
            let plan = LayerDequantPlan::from_manifest(&manifest, 0, format)?;
            assert_eq!(plan.bytes, 8192 * 2048 * 2);
            assert_eq!(plan.slots[&name].offset_bytes, 0);
            let scale = manifest.entries.get_mut(&scale_name).unwrap();
            scale.shape.swap(0, 1);
            assert!(LayerDequantPlan::from_manifest(&manifest, 0, format).is_err());
            manifest.entries.remove(&scale_name);
        }
        Ok(())
    }
}
