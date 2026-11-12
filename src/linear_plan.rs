//! Checked single-layer materialization layout, shared by planner and Linear.
use crate::{checkpoint::TensorManifest, model_spec::LinearFormat};
use anyhow::{Context, Result, ensure};
use safetensors::Dtype;
use std::collections::BTreeMap;

/// Activation tile width of gemm_skinny_bf16.
pub const MAX_SKINNY_ROWS: usize = 16;

/// Dense BF16-input FP32-accumulate m16n8k16 Tensor Core FLOPs per SM clock.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn tc_flops_per_sm_clock(sm: u32) -> Result<u32> {
    match sm {
        80 => Ok(2048),
        86 | 87 | 89 => Ok(512),
        90 => Ok(2048),
        100 => Ok(4096),
        120 => Ok(512),
        other => anyhow::bail!(
            "NVFP4 fused prefill unsupported compute capability sm_{other}: no BF16 Tensor Core rate defined"
        ),
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct NvFp4MMaxPlan {
    pub sm: u32,
    pub sms: usize,
    pub clock_khz: i32,
    pub memory_bytes_per_second: f64,
    pub tc_flops_per_sm_clock: u32,
    pub peak_tc_flops_per_second: f64,
    pub ridge_flops_per_byte: f64,
    pub m_max: usize,
}

#[cfg(any(feature = "cuda", test))]
impl NvFp4MMaxPlan {
    pub(crate) fn derive(
        sm: u32,
        sms: usize,
        clock_khz: i32,
        memory_bytes_per_second: f64,
    ) -> Result<Self> {
        let rate = tc_flops_per_sm_clock(sm)?;
        ensure!(
            sms > 0
                && clock_khz > 0
                && memory_bytes_per_second.is_finite()
                && memory_bytes_per_second > 0.0,
            "invalid device topology for NVFP4 m_max: sms={sms} clock_khz={clock_khz} bw={memory_bytes_per_second}"
        );
        let peak_tc_flops_per_second =
            (sms as f64) * f64::from(rate) * (f64::from(clock_khz) * 1000.0);
        let ridge_flops_per_byte = peak_tc_flops_per_second / memory_bytes_per_second;
        ensure!(
            ridge_flops_per_byte.is_finite() && ridge_flops_per_byte < 4096.0 / 7.0,
            "NVFP4 ridge_flops_per_byte={ridge_flops_per_byte} exceeds asymptotic byte-savings bound"
        );
        let mut m = MAX_SKINNY_ROWS + 1;
        loop {
            let bm_fused = if m <= 64 { 64usize } else { 128usize };
            let lt_passes = m.div_ceil(256) as f64;
            let fused_passes = m.div_ceil(bm_fused) as f64;
            let delta_bytes_per_weight = 2.5625 + 2.0 * lt_passes - 0.5625 * fused_passes;
            if delta_bytes_per_weight <= 0.0
                || (2.0 * m as f64) / delta_bytes_per_weight > ridge_flops_per_byte
            {
                break;
            }
            m = m.checked_add(1).context("NVFP4 m_max overflow")?;
        }
        Ok(Self {
            sm,
            sms,
            clock_khz,
            memory_bytes_per_second,
            tc_flops_per_sm_clock: rate,
            peak_tc_flops_per_second,
            ridge_flops_per_byte,
            m_max: m,
        })
    }
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
    pub fn from_model(
        manifest: &TensorManifest,
        layer: u32,
        spec: &crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        let plan = Self::from_manifest(manifest, layer, spec.linear_format)?;
        if layer == 45 && spec.architecture == crate::model_spec::Architecture::Glm5Next {
            return Ok(plan);
        }
        let layout = spec
            .layers
            .get(layer as usize)
            .context("linear layer outside model specification")?;
        if spec.architecture == crate::model_spec::Architecture::Glm5Next
            && layout.mixer == crate::model_spec::Mixer::FullAttention
        {
            manifest.validate_dsa_layer(spec, layer, &plan.prefix)?;
        }
        Ok(plan)
    }

    pub fn from_manifest(
        manifest: &TensorManifest,
        layer: u32,
        format: LinearFormat,
    ) -> Result<Self> {
        let language = format!("model.language_model.layers.{layer}.");
        let model = format!("model.layers.{layer}.");
        let direct = format!("language_model.layers.{layer}.");
        let count = |prefix: &str| {
            manifest
                .entries
                .keys()
                .filter(|name| name.starts_with(prefix))
                .count()
        };
        let language_count = count(&language);
        let model_count = count(&model);
        let direct_count = count(&direct);
        ensure!(
            language_count > 0 || model_count > 0 || direct_count > 0,
            "missing linear layer {layer}"
        );
        ensure!(
            [language_count, model_count, direct_count]
                .into_iter()
                .filter(|&count| count > 0)
                .count()
                == 1,
            "layer {layer} has ambiguous tensor namespaces"
        );
        let prefix = if language_count > 0 {
            language
        } else if model_count > 0 {
            model
        } else {
            direct
        };
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
        let mut native = 0usize;
        let mut push_slot = |name: String, n: u32, k: u32, scale_name: String| -> Result<()> {
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
                name,
                DequantSlot {
                    offset_bytes,
                    n,
                    k,
                    scale_name,
                },
            );
            Ok(())
        };
        if format == LinearFormat::NvFp4 {
            for (name, e) in &manifest.entries {
                if !e.load || e.is_routed_expert() || !name.starts_with(&prefix) {
                    continue;
                }
                ensure!(
                    !(e.dtype == Dtype::F8_E4M3 && name.ends_with(".weight")),
                    "FP8 weight in NVFP4 layer: {name}"
                );
                if e.dtype != Dtype::U8 {
                    continue;
                }
                ensure!(e.shape.len() == 2, "linear weight is not a matrix");
                let n = u32::try_from(e.shape[0])?;
                let packed_k = u32::try_from(e.shape[1])?;
                let k = packed_k.checked_mul(2).context("NVFP4 K extent overflow")?;
                ensure!(
                    n > 0
                        && k > 0
                        && k.is_multiple_of(16)
                        && Some(e.bytes) == e.shape[0].checked_mul(e.shape[1]),
                    "invalid NVFP4 weight layout: {name}"
                );
                let stem = name.strip_suffix(".weight").context("NVFP4 weight name")?;
                let scale_name = format!("{stem}.weight_scale");
                ensure!(
                    e.scale_name.as_ref() == Some(&scale_name),
                    "NVFP4 scale name mismatch"
                );
                let scale = manifest
                    .entries
                    .get(&scale_name)
                    .context("NVFP4 weight missing scale")?;
                let groups = (k / 16) as usize;
                ensure!(
                    scale.load
                        && scale.dtype == Dtype::F8_E4M3
                        && scale.shape == [n as usize, groups]
                        && Some(scale.bytes) == (n as usize).checked_mul(groups),
                    "NVFP4 scale layout mismatch: {scale_name}"
                );
                for aux in ["weight_scale_2", "input_scale"] {
                    let scalar = manifest
                        .entries
                        .get(&format!("{stem}.{aux}"))
                        .with_context(|| format!("NVFP4 weight missing {aux}"))?;
                    ensure!(
                        scalar.load
                            && scalar.dtype == Dtype::F32
                            && scalar.shape.is_empty()
                            && scalar.bytes == 4,
                        "NVFP4 {aux} layout mismatch: {stem}.{aux}"
                    );
                }
                push_slot(name.clone(), n, k, scale_name)?;
            }
        } else {
            for (name, e) in &manifest.entries {
                if !e.load || e.is_routed_expert() || !name.starts_with(&prefix) {
                    continue;
                }
                ensure!(
                    !(e.dtype == Dtype::U8 && e.is_linear_projection()),
                    "NVFP4 weight in {format:?} layer: {name}"
                );
                if e.dtype != Dtype::F8_E4M3 {
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
                if scale_dtype != Dtype::BF16 {
                    native += 1;
                    continue;
                }
                push_slot(name.clone(), n, k, scale_name)?;
            }
        }
        ensure!(
            manifest.entries.iter().any(|(name, entry)| {
                entry.is_linear_projection()
                    && name.starts_with(&prefix)
                    && matches!(entry.dtype, Dtype::BF16 | Dtype::F8_E4M3 | Dtype::U8)
            }),
            "missing linear layer {layer}"
        );
        let plan_shape = match format {
            LinearFormat::Fp8E4m3Block { .. } => native > 0 || !slots.is_empty(),
            LinearFormat::NvFp4 => !slots.is_empty(),
            LinearFormat::Bf16 => slots.is_empty(),
            LinearFormat::Fp8E4m3Channel => true,
        };
        let kind = if format == LinearFormat::NvFp4 {
            "NVFP4"
        } else {
            "FP8"
        };
        ensure!(
            plan_shape,
            "linear layer={layer} has {} {kind} slots for linear_format={format:?}",
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

    pub fn validate_contiguous_pair(
        &self,
        manifest: &TensorManifest,
        first: &str,
        second: &str,
    ) -> Result<([u32; 2], u64)> {
        let first_name = format!("{}{first}", self.prefix);
        let second_name = format!("{}{second}", self.prefix);
        let entry_a = manifest
            .entries
            .get(&first_name)
            .with_context(|| format!("layer {} missing manifest entry {first_name}", self.layer))?;
        let entry_b = manifest.entries.get(&second_name).with_context(|| {
            format!("layer {} missing manifest entry {second_name}", self.layer)
        })?;
        ensure!(
            entry_a.load
                && entry_b.load
                && entry_a.shard == entry_b.shard
                && entry_a.data_start == entry_b.data_start
                && entry_a.data_offsets[1] == entry_b.data_offsets[0]
                && entry_a.dtype == Dtype::BF16
                && entry_b.dtype == Dtype::BF16
                && entry_a.shape.len() == 2
                && entry_b.shape.len() == 2
                && entry_a.shape[1] == entry_b.shape[1],
            "layer {} projections {first_name} and {second_name} are not contiguous BF16 matrices with matching K in manifest",
            self.layer
        );
        let rows_a = u32::try_from(entry_a.shape[0])?;
        let rows_b = u32::try_from(entry_b.shape[0])?;
        let k = u32::try_from(entry_a.shape[1])?;
        let rows = rows_a
            .checked_add(rows_b)
            .context("fused projection output rows overflow")?;
        let bytes = (entry_a.bytes as u64)
            .checked_add(entry_b.bytes as u64)
            .context("fused projection byte span overflow")?;
        Ok(([rows, k], bytes))
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
            quant: None,
            swap_halves: false,
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
            LinearFormat::fp8_block([128, 128], Dtype::F32)?,
            LinearFormat::fp8_block([32, 32], Dtype::F8_E8M0)?,
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
            let bytes = shape.iter().product::<usize>() * (layout.scale_dtype().bitsize() / 8);
            manifest.entries.insert(
                scale_name.clone(),
                TensorEntry {
                    dtype: layout.scale_dtype(),
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
                    quant: None,
                    swap_halves: false,
                },
            );
            let plan = LayerDequantPlan::from_manifest(&manifest, 0, format)?;
            if layout.scale_dtype() == Dtype::BF16 {
                assert_eq!(plan.bytes, 8192 * 2048 * 2);
                assert_eq!(plan.slots[&name].offset_bytes, 0);
            } else {
                assert_eq!(plan.bytes, 0);
                assert!(plan.slots.is_empty());
            }
            let scale = manifest.entries.get_mut(&scale_name).unwrap();
            scale.shape.swap(0, 1);
            assert!(LayerDequantPlan::from_manifest(&manifest, 0, format).is_err());
            manifest.entries.remove(&scale_name);
        }
        let prefix = "model.language_model.layers.3.";
        let gate_bytes = 6144 * 1536 * 2;
        let q_bytes = 2048 * 1536 * 2;
        let v_bytes = 256 * 1536 * 2;
        let mut gemma_manifest = TensorManifest {
            entries: [
                (
                    format!("{prefix}mlp.gate_proj.weight"),
                    TensorEntry {
                        dtype: Dtype::BF16,
                        shape: vec![6144, 1536],
                        shard: "model.safetensors".into(),
                        bytes: gate_bytes,
                        load: true,
                        data_start: 1024,
                        data_offsets: [0, gate_bytes as u64],
                        scope: TensorScope::Text,
                        layer: Some(3),
                        role: "mlp.gate_proj.weight".into(),
                        scale_name: None,
                        quant: None,
                        swap_halves: false,
                    },
                ),
                (
                    format!("{prefix}mlp.up_proj.weight"),
                    TensorEntry {
                        dtype: Dtype::BF16,
                        shape: vec![6144, 1536],
                        shard: "model.safetensors".into(),
                        bytes: gate_bytes,
                        load: true,
                        data_start: 1024,
                        data_offsets: [gate_bytes as u64, (2 * gate_bytes) as u64],
                        scope: TensorScope::Text,
                        layer: Some(3),
                        role: "mlp.up_proj.weight".into(),
                        scale_name: None,
                        quant: None,
                        swap_halves: false,
                    },
                ),
                (
                    format!("{prefix}self_attn.q_proj.weight"),
                    TensorEntry {
                        dtype: Dtype::BF16,
                        shape: vec![2048, 1536],
                        shard: "model.safetensors".into(),
                        bytes: q_bytes,
                        load: true,
                        data_start: 1024,
                        data_offsets: [(2 * gate_bytes) as u64, (2 * gate_bytes + q_bytes) as u64],
                        scope: TensorScope::Text,
                        layer: Some(3),
                        role: "self_attn.q_proj.weight".into(),
                        scale_name: None,
                        quant: None,
                        swap_halves: false,
                    },
                ),
                (
                    format!("{prefix}self_attn.v_proj.weight"),
                    TensorEntry {
                        dtype: Dtype::BF16,
                        shape: vec![256, 1536],
                        shard: "model.safetensors".into(),
                        bytes: v_bytes,
                        load: true,
                        data_start: 1024,
                        data_offsets: [
                            (2 * gate_bytes + q_bytes) as u64,
                            (2 * gate_bytes + q_bytes + v_bytes) as u64,
                        ],
                        scope: TensorScope::Text,
                        layer: Some(3),
                        role: "self_attn.v_proj.weight".into(),
                        scale_name: None,
                        quant: None,
                        swap_halves: false,
                    },
                ),
            ]
            .into(),
            resident_bytes: 2 * gate_bytes + q_bytes + v_bytes,
        };
        let gemma_plan = LayerDequantPlan::from_manifest(&gemma_manifest, 3, LinearFormat::Bf16)?;
        assert_eq!(
            gemma_plan.validate_contiguous_pair(
                &gemma_manifest,
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight"
            )?,
            ([12288, 1536], (2 * gate_bytes) as u64)
        );
        assert_eq!(
            gemma_plan.validate_contiguous_pair(
                &gemma_manifest,
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight"
            )?,
            ([2304, 1536], (q_bytes + v_bytes) as u64)
        );
        gemma_manifest
            .entries
            .get_mut(&format!("{prefix}mlp.up_proj.weight"))
            .unwrap()
            .data_offsets[0] += 512;
        let gap_err = gemma_plan
            .validate_contiguous_pair(
                &gemma_manifest,
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            )
            .unwrap_err()
            .to_string();
        assert!(
            gap_err.contains("layer 3")
                && gap_err.contains("model.language_model.layers.3.mlp.gate_proj.weight")
                && gap_err.contains("model.language_model.layers.3.mlp.up_proj.weight"),
            "{gap_err}"
        );
        Ok(())
    }

    #[test]
    fn nvfp4_m_max_derivation_and_closed_capability_table() -> Result<()> {
        for (sm, want) in [
            (80, 2048),
            (86, 512),
            (87, 512),
            (89, 512),
            (90, 2048),
            (100, 4096),
            (120, 512),
        ] {
            assert_eq!(tc_flops_per_sm_clock(sm)?, want);
        }
        for unknown in [0, 70, 75, 88, 91, 101, 121] {
            let err = tc_flops_per_sm_clock(unknown).unwrap_err().to_string();
            assert!(err.contains(&format!("sm_{unknown}")), "{err}");
        }
        let rtx4090 = NvFp4MMaxPlan::derive(89, 128, 2_520_000, 1_008_096_000_000.0)?;
        assert_eq!(rtx4090.m_max, 385);
        let rtx4090_oc = NvFp4MMaxPlan::derive(89, 128, 2_640_000, 1_008_096_000_000.0)?;
        assert_eq!(rtx4090_oc.m_max, 385);
        let a4000 = NvFp4MMaxPlan::derive(86, 48, 1_560_000, 448_064_000_000.0)?;
        assert_eq!(a4000.m_max, 148);
        assert!(NvFp4MMaxPlan::derive(88, 128, 2_520_000, 1.0e12).is_err());
        assert!(NvFp4MMaxPlan::derive(89, 0, 2_520_000, 1.0e12).is_err());
        assert!(NvFp4MMaxPlan::derive(89, 128, 0, 1.0e12).is_err());
        assert!(NvFp4MMaxPlan::derive(89, 128, 2_520_000, 0.0).is_err());
        assert!(NvFp4MMaxPlan::derive(100, 256, 3_000_000, 1.0e12).is_err());
        Ok(())
    }
}
