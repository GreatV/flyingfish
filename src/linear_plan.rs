//! Checked single-layer materialization layout, shared by planner and Linear.
use crate::{checkpoint::TensorManifest, model_spec::LinearFormat};
use anyhow::{Context, Result, bail, ensure};
use safetensors::Dtype;
use std::collections::BTreeMap;

pub const fn direct_fp8_rows(rows: usize) -> bool {
    rows >= 1 && rows <= 5
}

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
        let prefix = format!("model.language_model.layers.{layer}.");
        Self::build(manifest, layer, prefix, format)
    }

    pub fn mtp(manifest: &TensorManifest) -> Result<Self> {
        Self::build(
            manifest,
            0,
            "mtp.layers.0.".into(),
            LinearFormat::Fp8E4m3Block128,
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
            if !e.load || e.dtype != Dtype::F8_E4M3 || !name.starts_with(&prefix) {
                continue;
            }
            ensure!(e.shape.len() == 2, "linear weight is not a matrix");
            let n = u32::try_from(e.shape[0])?;
            let k = u32::try_from(e.shape[1])?;
            let scale_name = format!(
                "{}.weight_scale_inv",
                name.strip_suffix(".weight").context("FP8 weight name")?
            );
            let scale = manifest
                .entries
                .get(&scale_name)
                .context("FP8 weight missing scale")?;
            ensure!(
                scale.dtype == Dtype::BF16
                    && scale.shape == [n.div_ceil(128) as usize, k.div_ceil(128) as usize],
                "FP8 scale layout mismatch"
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
                    && name.starts_with(&prefix)
                    && name.ends_with(".weight")
                    && entry.shape.len() == 2
                    && matches!(entry.dtype, Dtype::BF16 | Dtype::F8_E4M3)
            }),
            "missing linear layer {layer}"
        );
        let plan_shape = match format {
            LinearFormat::Fp8E4m3Block128 => !slots.is_empty(),
            LinearFormat::Bf16 => slots.is_empty(),
            other => bail!("linear_format={other:?} has no dequant plan"),
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
    fn native_bf16_layer_requires_no_materialization_storage() -> Result<()> {
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
        let manifest = TensorManifest {
            entries: [(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into(),
                entry,
            )]
            .into(),
            resident_bytes: 8192 * 2048 * 2,
        };
        let plan = LayerDequantPlan::from_manifest(&manifest, 0, LinearFormat::Bf16)?;
        let error = LayerDequantPlan::from_manifest(&manifest, 0, LinearFormat::Fp8E4m3Block128)
            .unwrap_err();
        assert!(error.to_string().contains("0 FP8 slots"), "{error}");
        assert_eq!(plan.bytes, 0);
        assert!(plan.slots.is_empty());
        assert!(LayerDequantPlan::from_manifest(&manifest, 1, LinearFormat::Bf16).is_err());
        Ok(())
    }
}
