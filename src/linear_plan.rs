//! Checked single-layer materialization layout, shared by planner and Linear.
use crate::checkpoint::TensorManifest;
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use std::collections::BTreeMap;
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
    pub bytes: u64,
    pub slots: BTreeMap<String, DequantSlot>,
}
impl LayerDequantPlan {
    pub fn from_manifest(manifest: &TensorManifest, layer: u32) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.");
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
        ensure!(!slots.is_empty(), "missing FP8 layer {layer}");
        bytes = bytes.checked_add(255).context("arena size overflow")? & !255;
        Ok(Self {
            layer,
            bytes,
            slots,
        })
    }
}
