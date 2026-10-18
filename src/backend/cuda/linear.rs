//! Typed Linear views; all BF16 plans execute through the shared Blas handle.
use super::{
    contracts::{RowDomain, Scratch, Tensor, TensorMut},
    weights::WeightCatalog,
};
use crate::fp8::Fp8ScaleLayout;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::sync::Arc;
/// Explicit visible bias boundary; the caller's numerical contract selects it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum LinearBiasMode {
    BeforeOutputRound,
    AfterOutputRound,
}
#[derive(Clone, Copy)]
pub(crate) enum LinearWeightRef {
    Bf16(Tensor<bf16, 2>),
    Fp8 {
        codes: Tensor<u8, 2>,
        scale: Tensor<bf16, 2>,
        layout: Fp8ScaleLayout,
    },
}
pub(crate) struct Fp8Ops {
    dequant: [CudaFunction; 2],
    small: [[CudaFunction; 5]; 2],
    fake_quant: CudaFunction,
}
impl Fp8Ops {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let m = super::cubin::module(ctx, "fp8_linear")?;
        Ok(Self {
            fake_quant: m.load_function("e4m3_fake_quant_bf16")?,
            dequant: [
                m.load_function("fp8_dequant_bf16")?,
                m.load_function("fp8_dequant_bf16_per_channel")?,
            ],
            small: [
                [
                    m.load_function("fp8_gemv_bf16")?,
                    m.load_function("fp8_skinny_bf16_m2")?,
                    m.load_function("fp8_skinny_bf16_m3")?,
                    m.load_function("fp8_skinny_bf16_m4")?,
                    m.load_function("fp8_skinny_bf16_m5")?,
                ],
                [
                    m.load_function("fp8_gemv_bf16_per_channel")?,
                    m.load_function("fp8_skinny_bf16_m2_per_channel")?,
                    m.load_function("fp8_skinny_bf16_m3_per_channel")?,
                    m.load_function("fp8_skinny_bf16_m4_per_channel")?,
                    m.load_function("fp8_skinny_bf16_m5_per_channel")?,
                ],
            ],
        })
    }
    pub fn fake_quant(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut TensorMut<bf16, 2>,
        codes: Option<&TensorMut<u8, 2>>,
        scales: Option<&TensorMut<bf16, 1>>,
        rows: Option<RowDomain>,
    ) -> Result<()> {
        x.0.validate()?;
        ensure!(
            x.0.shape.iter().all(|&n| n <= i32::MAX as u32)
                && x.0.stride == [u64::from(x.0.shape[1]), 1],
            "invalid E4M3 activation layout"
        );
        if let Some(out) = codes {
            out.0.validate()?;
            ensure!(
                out.0.shape == x.0.shape && out.0.stride == x.0.stride,
                "E4M3 code output mismatch"
            );
            disjoint(&x.0, &out.0)?;
            if let Some(scale) = scales {
                disjoint(&out.0, &scale.0)?;
            }
        }
        if let Some(out) = scales {
            out.0.validate()?;
            ensure!(
                out.0.shape == [x.0.shape[0]] && out.0.stride == [1],
                "E4M3 scale output mismatch"
            );
            disjoint(&x.0, &out.0)?;
        }
        let codes_ptr = codes.map_or(0, |out| out.0.ptr);
        let scales_ptr = scales.map_or(0, |out| out.0.ptr);
        if let Some(rows) = rows {
            rows.validate(x.0.shape[0])?;
        }
        let (meta, row_base, row_capacity) = rows.map_or((0, 0, x.0.shape[0]), |rows| {
            (rows.meta.ptr, rows.row_base, rows.row_capacity)
        });
        unsafe {
            stream
                .launch_builder(&self.fake_quant)
                .arg(&x.0.ptr)
                .arg(&codes_ptr)
                .arg(&scales_ptr)
                .arg(&(x.0.shape[0] as i32))
                .arg(&(x.0.shape[1] as i32))
                .arg(&meta)
                .arg(&row_base)
                .arg(&row_capacity)
                .launch(super::ops::grid(row_capacity as usize, 256))?;
        }
        Ok(())
    }
    fn check(codes: &Tensor<u8, 2>, scale: &Tensor<bf16, 2>, layout: Fp8ScaleLayout) -> Result<()> {
        codes.validate()?;
        scale.validate()?;
        ensure!(
            codes.shape.iter().all(|&v| v <= i32::MAX as u32)
                && codes.stride == [u64::from(codes.shape[1]), 1],
            "invalid FP8 matrix"
        );
        let shape = layout
            .scale_shape(codes.shape[0], codes.shape[1])
            .map(|v| v as u32);
        ensure!(
            scale.shape == shape && scale.stride == [u64::from(shape[1]), 1],
            "invalid FP8 scale"
        );
        Ok(())
    }
    pub fn materialize(
        &self,
        stream: &Arc<CudaStream>,
        codes: Tensor<u8, 2>,
        scale: Tensor<bf16, 2>,
        layout: Fp8ScaleLayout,
        dst: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        Self::check(&codes, &scale, layout)?;
        dst.0.validate()?;
        ensure!(
            dst.0.shape == codes.shape && dst.0.stride == codes.stride,
            "dequant output mismatch"
        );
        disjoint(&codes, &dst.0)?;
        disjoint(&scale, &dst.0)?;
        let count = u64::from(codes.shape[0]) * u64::from(codes.shape[1]);
        unsafe {
            stream
                .launch_builder(&self.dequant[layout as usize])
                .arg(&codes.ptr)
                .arg(&scale.ptr)
                .arg(&dst.0.ptr)
                .arg(&(codes.shape[0] as i32))
                .arg(&(codes.shape[1] as i32))
                .launch(super::ops::grid(
                    count.div_ceil(256).min(65535) as usize,
                    256,
                ))?;
        }
        Ok(())
    }
    pub fn small_gemm(
        &self,
        stream: &Arc<CudaStream>,
        codes: Tensor<u8, 2>,
        scale: Tensor<bf16, 2>,
        layout: Fp8ScaleLayout,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        Self::check(&codes, &scale, layout)?;
        validate_call(codes.shape, x, &y.0)?;
        ensure!(
            crate::linear_plan::direct_fp8_rows(x.shape[0] as usize),
            "direct FP8 linear requires M1..5"
        );
        disjoint(&codes, &y.0)?;
        disjoint(&scale, &y.0)?;
        unsafe {
            stream
                .launch_builder(&self.small[layout as usize][x.shape[0] as usize - 1])
                .arg(&codes.ptr)
                .arg(&scale.ptr)
                .arg(&x.ptr)
                .arg(&y.0.ptr)
                .arg(&(codes.shape[0] as i32))
                .arg(&(codes.shape[1] as i32))
                .launch(super::ops::grid(codes.shape[0].div_ceil(4) as usize, 128))?;
        }
        Ok(())
    }
}
pub(crate) fn disjoint<T: Copy, U: Copy, const R: usize, const S: usize>(
    a: &Tensor<T, R>,
    b: &Tensor<U, S>,
) -> Result<()> {
    a.validate()?;
    b.validate()?;
    ensure!(
        a.ptr + a.span_bytes <= b.ptr || b.ptr + b.span_bytes <= a.ptr,
        "linear buffers overlap"
    );
    Ok(())
}
pub(crate) fn validate_call(
    weight: [u32; 2],
    x: Tensor<bf16, 2>,
    y: &Tensor<bf16, 2>,
) -> Result<()> {
    x.validate()?;
    y.validate()?;
    ensure!(
        weight[0] > 0 && weight[1] > 0 && weight.iter().all(|&v| v <= i32::MAX as u32),
        "invalid linear dimensions"
    );
    ensure!(
        x.shape[1] == weight[1] && y.shape == [x.shape[0], weight[0]],
        "linear activation shape mismatch"
    );
    ensure!(
        x.stride == [u64::from(weight[1]), 1] && y.stride == [u64::from(weight[0]), 1],
        "linear requires row-major activations"
    );
    disjoint(&x, y)
}
impl WeightCatalog {
    pub fn linear_ref(&self, name: &str, stream: &Arc<CudaStream>) -> Result<LinearWeightRef> {
        let entry = self
            .manifest
            .entries
            .get(name)
            .context("linear tensor missing from manifest")?;
        ensure!(entry.shape.len() == 2, "linear tensor must be a matrix");
        let shape = [
            u32::try_from(entry.shape[0])?,
            u32::try_from(entry.shape[1])?,
        ];
        let stride = [u64::from(shape[1]), 1];
        match self
            .tensors
            .get(name)
            .context("linear tensor not uploaded")?
        {
            super::weights::StoredTensor::Bf16(w) => {
                let (p, _r) = w.device_ptr(stream);
                Ok(LinearWeightRef::Bf16(unsafe {
                    Tensor::new(p, shape, stride, entry.bytes as u64)?
                }))
            }
            super::weights::StoredTensor::Fp8(w) => {
                let scale_name = entry
                    .scale_name
                    .as_ref()
                    .context("FP8 scale descriptor missing")?;
                let layout = if scale_name.ends_with(".weight_scale_inv") {
                    Fp8ScaleLayout::Block128
                } else {
                    ensure!(
                        scale_name.ends_with(".weight_scale"),
                        "invalid FP8 scale descriptor {scale_name}"
                    );
                    Fp8ScaleLayout::PerChannel
                };
                let scale = self.bf16(scale_name)?;
                let sh = layout.scale_shape(shape[0], shape[1]).map(|v| v as u32);
                let (wp, _rw) = w.device_ptr(stream);
                let (sp, _rs) = scale.device_ptr(stream);
                Ok(LinearWeightRef::Fp8 {
                    layout,
                    codes: unsafe { Tensor::new(wp, shape, stride, entry.bytes as u64)? },
                    scale: unsafe {
                        Tensor::new(sp, sh, [u64::from(sh[1]), 1], (scale.len() * 2) as u64)?
                    },
                })
            }
        }
    }
}
impl crate::linear_plan::LayerDequantPlan {
    pub(crate) fn weight(&self, name: &str, arena: &Scratch) -> Result<LinearWeightRef> {
        arena.validate(self.bytes)?;
        let slot = self
            .slots
            .get(name)
            .context("missing materialized weight")?;
        Ok(LinearWeightRef::Bf16(unsafe {
            Tensor::new(
                arena.ptr + slot.offset_bytes,
                [slot.n, slot.k],
                [u64::from(slot.k), 1],
                u64::from(slot.n) * u64::from(slot.k) * 2,
            )?
        }))
    }
}

struct BoundProjection {
    resident: LinearWeightRef,
    prefill: Tensor<bf16, 2>,
}
/// Setup-bound views for a layer. Engine retains catalog/arena storage; run does
/// not construct tensor names, allocate scratch or invoke a Lt heuristic.
pub(crate) struct LayerLinearPlan {
    projections: std::collections::BTreeMap<String, BoundProjection>,
    materialize: Vec<(LinearWeightRef, TensorMut<bf16, 2>)>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BoundFfn {
    Dense,
    Moe {
        router: Tensor<bf16, 2>,
        shared: [Tensor<bf16, 2>; 3],
        shared_gate: Tensor<bf16, 1>,
    },
}

impl BoundFfn {
    pub fn bind(spec: &crate::model_spec::ModelSpec, linear: &mut LayerLinearPlan) -> Result<Self> {
        let Some(moe) = spec.moe else {
            return Ok(Self::Dense);
        };
        let h = spec.hidden;
        let i = moe.shared_intermediate;
        let router = linear.take_bf16("mlp.gate.weight", [moe.experts, h])?;
        let shared = [
            linear.take_bf16("mlp.shared_expert.gate_proj.weight", [i, h])?,
            linear.take_bf16("mlp.shared_expert.up_proj.weight", [i, h])?,
            linear.take_bf16("mlp.shared_expert.down_proj.weight", [h, i])?,
        ];
        let gate = linear.take_bf16("mlp.shared_expert_gate.weight", [1, h])?;
        let shared_gate = unsafe { Tensor::new(gate.ptr, [h], [1], gate.span_bytes)? };
        Ok(Self::Moe {
            router,
            shared,
            shared_gate,
        })
    }

    pub fn prepare(&self, blas: &super::blas::Blas, rows: u32) -> Result<()> {
        match self {
            Self::Dense => Ok(()),
            Self::Moe {
                shared,
                shared_gate,
                ..
            } => {
                shared_gate.validate()?;
                for weight in shared {
                    blas.prepare_fixed(rows, weight.shape[0], weight.shape[1])?;
                }
                Ok(())
            }
        }
    }
}

impl LayerLinearPlan {
    fn take_bf16(&mut self, role: &str, expected: [u32; 2]) -> Result<Tensor<bf16, 2>> {
        let projection = self
            .projections
            .remove(role)
            .with_context(|| format!("FFN projection not bound: role={role}"))?;
        let LinearWeightRef::Bf16(weight) = projection.resident else {
            anyhow::bail!(
                "FFN projection format mismatch: role={role} format=Fp8E4m3Block128 expected=Bf16"
            );
        };
        ensure!(
            weight.shape == expected && weight.stride == [u64::from(expected[1]), 1],
            "FFN projection layout mismatch: role={role} shape={:?} stride={:?} expected={expected:?}",
            weight.shape,
            weight.stride
        );
        Ok(weight)
    }

    pub fn bind(
        plan: &crate::linear_plan::LayerDequantPlan,
        catalog: &WeightCatalog,
        arena: Option<&Scratch>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        if plan.bytes > 0 {
            arena
                .context("FP8 layer has no materialization arena")?
                .validate(plan.bytes)?;
        }
        let prefix = &plan.prefix;
        let mut projections = std::collections::BTreeMap::new();
        let mut materialize = Vec::new();
        for (name, entry) in &catalog.manifest.entries {
            if !entry.load
                || entry.is_routed_expert()
                || !name.starts_with(prefix)
                || !name.ends_with(".weight")
                || entry.shape.len() != 2
            {
                continue;
            }
            let resident = catalog.linear_ref(name, stream)?;
            let prefill = match &resident {
                LinearWeightRef::Bf16(w) => *w,
                LinearWeightRef::Fp8 { .. } => {
                    let LinearWeightRef::Bf16(w) = plan.weight(
                        name,
                        arena.context("FP8 layer has no materialization arena")?,
                    )?
                    else {
                        unreachable!()
                    };
                    materialize.push((resident, TensorMut(w)));
                    w
                }
            };
            projections.insert(
                name[prefix.len()..].to_owned(),
                BoundProjection { resident, prefill },
            );
        }
        ensure!(
            materialize.len() == plan.slots.len(),
            "layer linear plan does not cover materialization slots"
        );
        Ok(Self {
            projections,
            materialize,
        })
    }
    pub fn prepare(&self, blas: &super::blas::Blas, rows: u32) -> Result<()> {
        for p in self.projections.values() {
            blas.prepare_fixed(rows, p.prefill.shape[0], p.prefill.shape[1])?;
        }
        Ok(())
    }
    pub fn materialize(
        &mut self,
        blas: &super::blas::Blas,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        for (weight, dst) in &mut self.materialize {
            let LinearWeightRef::Fp8 {
                codes,
                scale,
                layout,
            } = *weight
            else {
                anyhow::bail!("materialization source requires an FP8 weight");
            };
            blas.fp8_ops()?
                .materialize(stream, codes, scale, layout, dst)?;
        }
        Ok(())
    }
    pub fn run(
        &self,
        blas: &super::blas::Blas,
        role: &str,
        phase: crate::op_contract::ExecutionPhase,
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        let p = self
            .projections
            .get(role)
            .with_context(|| format!("layer projection not bound: {role}"))?;
        let weight = if phase == crate::op_contract::ExecutionPhase::Prefill
            && !crate::linear_plan::direct_fp8_rows(x.shape[0] as usize)
        {
            LinearWeightRef::Bf16(p.prefill)
        } else {
            p.resident
        };
        blas.linear_typed(phase, weight, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{
        Device, blas::Blas, closure::Closure, ops::Ops, weights::StoredTensor,
    };
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::op_contract::ExecutionPhase;
    use std::{path::PathBuf, rc::Rc};

    #[test]
    #[ignore = "GPU window, official FP8 model, T13 and projection reference required"]
    fn channel_projection_gpu() -> Result<()> {
        use super::super::q38_tests::view;
        use crate::fp8::{effective_weight, linear_reference};
        use cudarc::driver::sys;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = PathBuf::from(
            std::env::var_os("FF_Q38_PROBE_DIR").context("FF_Q38_PROBE_DIR missing")?,
        );
        let reference_dir = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("FF_MINICPM_REFERENCE_DIR missing")?,
        );
        let model =
            PathBuf::from(std::env::var_os("FF_FP8_MODEL").context("FF_FP8_MODEL missing")?);
        let spec = crate::model_spec::ModelSpec::read(&model)?;
        ensure!(
            spec.hidden == 2048
                && spec.q_width == 4096
                && spec.linear_format == crate::model_spec::LinearFormat::Fp8E4m3Channel,
            "invalid FP8 model geometry"
        );
        let checkpoint = crate::checkpoint::MappedCheckpoint::open(&model)?;
        let mut manifest = checkpoint.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let views = checkpoint.views()?;
        let tap_bytes = std::fs::read(reference_dir.join("fixtures/text_32_mixer.safetensors"))?;
        let taps = safetensors::SafeTensors::deserialize(&tap_bytes)?;
        let reference_bytes = std::fs::read(root.join("projection-reference.safetensors"))?;
        let references = safetensors::SafeTensors::deserialize(&reference_bytes)?;
        let device = Device::new(0)?;
        let stream = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &root,
            Rc::new(Closure::new(false, "channel FP8 projection gate".into())),
        )?;
        blas.enable_fp8(&device.ctx)?;
        let mut defects = 0;
        let mut cases = 0;
        // Twice the sm89 BF16 rows1..5,8,32 shape/FP32-path floors.
        for (op, n, k, input_name, limit) in [
            ("q", 8192u32, 2048u32, "layer_03_mixer_in_norm", 0.0625f32),
            ("k", 512, 2048, "layer_03_mixer_in_norm", 0.0625),
            ("v", 512, 2048, "layer_03_mixer_in_norm", 0.03125),
            ("o", 2048, 4096, "layer_03_o_proj_in", 0.0078125),
        ] {
            let role = format!("self_attn.{op}_proj.weight");
            let name = format!("model.language_model.layers.3.{role}");
            let scale_name = format!("{name}_scale");
            let entry = manifest
                .entries
                .remove(&name)
                .context("projection manifest entry missing")?;
            let scale_entry = manifest
                .entries
                .remove(&scale_name)
                .context("projection scale missing")?;
            let code = views[&entry.shard].tensor(&name)?;
            let scale = views[&scale_entry.shard].tensor(&scale_name)?;
            ensure!(
                code.dtype() == safetensors::Dtype::F8_E4M3
                    && code.shape() == [n as usize, k as usize]
                    && scale.dtype() == safetensors::Dtype::BF16
                    && scale.shape() == [n as usize, 1],
                "invalid published {op} layout"
            );
            let scales = scale
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let host_weight = code
                .data()
                .iter()
                .enumerate()
                .map(|(i, &c)| effective_weight(c, scales[i / k as usize]))
                .collect::<Vec<_>>();
            ensure!(
                host_weight.iter().all(|v| v.is_finite()),
                "nonfinite published {op} weight"
            );
            let input = taps.tensor(input_name)?;
            ensure!(
                input.dtype() == safetensors::Dtype::BF16 && input.shape() == [32, k as usize],
                "invalid T13 projection input"
            );
            let inputs = input
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let host_fake = inputs
                .chunks_exact(k as usize)
                .map(crate::fp8::fake_quant_reference)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(|(_, _, f)| f)
                .collect::<Vec<_>>();
            let scalar = linear_reference(&host_weight, &host_fake[..5 * k as usize], k as usize)?;
            let codes = stream.clone_htod(code.data())?;
            let scales = stream.clone_htod(&scales)?;
            let host_materialized = stream.clone_htod(&host_weight)?;
            let dequant = stream.alloc_zeros::<u8>(host_weight.len() * 2)?;
            let bytes = entry.bytes + scale_entry.bytes;
            let entries = [(name.clone(), entry), (scale_name.clone(), scale_entry)].into();
            let small_manifest = TensorManifest {
                entries,
                resident_bytes: bytes,
            };
            let dequant_plan = crate::linear_plan::LayerDequantPlan::from_manifest(
                &small_manifest,
                3,
                spec.linear_format,
            )?;
            let catalog = WeightCatalog {
                tensors: [
                    (name.clone(), StoredTensor::Fp8(codes)),
                    (scale_name.clone(), StoredTensor::Bf16(scales)),
                ]
                .into(),
                manifest: small_manifest,
                bytes,
                upload_content_sha256: None,
                upload_hashed_bytes: None,
                load_wall_ms: 0.0,
            };
            use cudarc::driver::DevicePtr;
            let scratch = Scratch {
                ptr: dequant.device_ptr(stream).0,
                bytes: dequant.len() as u64,
            };
            let mut plan = LayerLinearPlan::bind(&dequant_plan, &catalog, Some(&scratch), stream)?;
            plan.materialize(&blas, stream)?;
            stream.synchronize()?;
            let materialized = stream.clone_dtoh(&dequant)?;
            let weight_bad = materialized
                .iter()
                .copied()
                .ne(host_weight.iter().flat_map(|v| v.to_le_bytes()));
            println!(
                "FP8_MATERIALIZE op={op} stored_weight=[{n},{k}] mismatches_present={weight_bad} {}",
                if weight_bad { "DEFECT" } else { "PASS" }
            );
            defects += usize::from(weight_bad);
            for m in [1u32, 2, 3, 4, 5, 8, 32] {
                plan.prepare(&blas, m)?;
                let expected = if m <= 5 {
                    scalar[..m as usize * n as usize].to_vec()
                } else {
                    let x = stream.clone_htod(&host_fake[..m as usize * k as usize])?;
                    let y = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                    blas.linear_typed(
                        ExecutionPhase::Prefill,
                        LinearWeightRef::Bf16(view(
                            &host_materialized,
                            stream,
                            [n, k],
                            [u64::from(k), 1],
                        )?),
                        view(&x, stream, [m, k], [u64::from(k), 1])?,
                        &mut TensorMut(view(&y, stream, [m, n], [u64::from(n), 1])?),
                    )?;
                    stream.clone_dtoh(&y)?
                };
                let torch = references.tensor(&format!("{op}_bf16_rows{m}"))?;
                ensure!(
                    torch.dtype() == safetensors::Dtype::BF16
                        && torch.shape() == [m as usize, n as usize],
                    "invalid torch projection reference"
                );
                let torch = torch
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                for graph in [false, true] {
                    let x = stream.clone_htod(&inputs[..m as usize * k as usize])?;
                    let y = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                    let x_view = view(&x, stream, [m, k], [u64::from(k), 1])?;
                    let y_view = view(&y, stream, [m, n], [u64::from(n), 1])?;
                    device.finish_upload()?;
                    let execute = || -> Result<()> {
                        blas.fp8_ops()?.fake_quant(
                            stream,
                            &mut TensorMut(x_view),
                            None,
                            None,
                            None,
                        )?;
                        plan.run(
                            &blas,
                            &role,
                            if m == 1 {
                                ExecutionPhase::Decode
                            } else {
                                ExecutionPhase::Prefill
                            },
                            x_view,
                            &mut TensorMut(y_view),
                        )
                    };
                    if graph {
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let run = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        run?;
                        captured?.context("empty FP8 projection graph")?.launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&y)?;
                    ensure!(
                        got.iter().all(|v| v.is_finite()),
                        "nonfinite projection output"
                    );
                    let bit_bad = got
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let max_abs = got
                        .iter()
                        .zip(&torch)
                        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                        .fold(0.0f32, f32::max);
                    let mut bad = bit_bad > 0 || max_abs > limit;
                    if op == "o" {
                        let tap = taps.tensor("layer_03_mixer_out")?;
                        ensure!(
                            tap.dtype() == safetensors::Dtype::BF16
                                && tap.shape() == [32, n as usize],
                            "invalid T13 o output tap"
                        );
                        let max_t13 = got
                            .iter()
                            .zip(tap.data().as_chunks::<2>().0.iter())
                            .map(|(a, b)| (a.to_f32() - bf16::from_le_bytes(*b).to_f32()).abs())
                            .fold(0.0f32, f32::max);
                        bad |= max_t13 > limit;
                        println!(
                            "FP8_T13_O rows={m} graph={graph} max_abs={max_t13} limit={limit}"
                        );
                    }
                    println!(
                        "FP8_PROJECTION op={op} stored_weight=[{n},{k}] scale=[{n},1] rows={m} graph={graph} bit_mismatches={bit_bad} torch_max_abs={max_abs} limit={limit} {}",
                        if bad { "DEFECT" } else { "PASS" }
                    );
                    defects += usize::from(bad);
                    cases += 1;
                }
            }
        }
        println!("FP8_PROJECTION_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "FP8 projection DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    #[ignore = "GPU window and FF_MINICPM_REFERENCE_DIR required"]
    fn token_quantization_gpu() -> Result<()> {
        use super::super::q38_tests::view;
        use crate::fp8::fake_quant_reference;
        use cudarc::driver::sys;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let root = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("FF_MINICPM_REFERENCE_DIR missing")?,
        );
        let bytes = std::fs::read(root.join("fixtures/text_32_mixer.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut inputs = Vec::new();
        for (name, prefix, width) in [
            ("layer_03_mixer_in_norm", "layer_03_attn_in", 2048usize),
            ("layer_03_o_proj_in", "layer_03_o_proj_in", 4096usize),
        ] {
            let input = tensors.tensor(name)?;
            let scales = tensors.tensor(&format!("{prefix}_scale"))?;
            let codes = tensors.tensor(&format!("{prefix}_fp8_q"))?;
            let fake = tensors.tensor(&format!("{prefix}_fake_quant"))?;
            ensure!(
                input.dtype() == safetensors::Dtype::BF16
                    && input.shape() == [32, width]
                    && scales.dtype() == safetensors::Dtype::BF16
                    && scales.shape() == [32, 1]
                    && codes.dtype() == safetensors::Dtype::F8_E4M3
                    && codes.shape() == input.shape()
                    && fake.dtype() == safetensors::Dtype::BF16
                    && fake.shape() == input.shape(),
                "invalid T13 activation fixture {name}"
            );
            let values = input
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_le_bytes([v[0], v[1]]))
                .collect::<Vec<_>>();
            for (row, x) in values.chunks_exact(width).enumerate() {
                let (s, c, f) = fake_quant_reference(x)?;
                ensure!(
                    s.to_le_bytes() == scales.data()[row * 2..row * 2 + 2]
                        && c == codes.data()[row * width..(row + 1) * width]
                        && f.iter()
                            .flat_map(|v| v.to_le_bytes())
                            .eq(fake.data()[row * width * 2..(row + 1) * width * 2]
                                .iter()
                                .copied()),
                    "T13 scalar oracle invalid source={name} row={row}"
                );
            }
            for rows in [1, 2, 3, 4, 5, 8, 32] {
                inputs.push((name, rows, width, values[..rows * width].to_vec()));
            }
        }
        let width = 257usize;
        let tiny = bf16::from_f32(bf16::EPSILON.to_f32() * 0.5);
        let values = (0..9 * width)
            .map(|i| match i / width {
                0 => bf16::ZERO,
                1 => bf16::NEG_ZERO,
                2 => {
                    if i % 2 == 0 {
                        tiny
                    } else {
                        -tiny
                    }
                }
                3 => bf16::from_bits((i % 127 + 1) as u16),
                _ => bf16::from_f32(((i * 37 % 101) as f32 - 50.0) / 32.0),
            })
            .collect::<Vec<_>>();
        let (right, _, _) = fake_quant_reference(&values[2 * width..3 * width])?;
        let wrong = bf16::from_f32(tiny.to_f32().max(bf16::EPSILON.to_f32()) / 448.0);
        ensure!(
            right != wrong,
            "max(amax,eps) control is not discriminating"
        );
        println!("PASS E4M3_EPS_CONTROL right={right:?} max_amax_eps={wrong:?}");
        inputs.push(("zero_tiny_tail", 9, width, values));
        let device = Device::new(0)?;
        let stream = &device.stream;
        let op = Fp8Ops::new(&device.ctx)?;
        let mut defects = 0;
        let mut cases = 0;
        for (name, rows, width, values) in inputs {
            let mut want_codes = Vec::new();
            let mut want_scales = Vec::new();
            let mut want_fake = Vec::new();
            for x in values.chunks_exact(width) {
                let (s, c, f) = fake_quant_reference(x)?;
                want_scales.push(s);
                want_codes.extend(c);
                want_fake.extend(f);
            }
            let codes = stream.alloc_zeros::<u8>(values.len())?;
            let scales = stream.alloc_zeros::<bf16>(rows)?;
            let code_view = TensorMut(view(
                &codes,
                stream,
                [rows as u32, width as u32],
                [width as u64, 1],
            )?);
            let scale_view = TensorMut(view(&scales, stream, [rows as u32], [1])?);
            for graph in [false, true] {
                for capture_outputs in [false, true] {
                    let x = stream.clone_htod(&values)?;
                    let x_view = view(&x, stream, [rows as u32, width as u32], [width as u64, 1])?;
                    device.finish_upload()?;
                    let execute = || {
                        op.fake_quant(
                            stream,
                            &mut TensorMut(x_view),
                            capture_outputs.then_some(&code_view),
                            capture_outputs.then_some(&scale_view),
                            None,
                        )
                    };
                    if graph {
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let run = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        run?;
                        captured?
                            .context("empty activation quantization graph")?
                            .launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&x)?;
                    ensure!(
                        got.iter().all(|v| v.is_finite()),
                        "nonfinite fake-quant output"
                    );
                    let fake_bad = got
                        .iter()
                        .zip(&want_fake)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let (code_bad, scale_bad) = if capture_outputs {
                        (
                            stream
                                .clone_dtoh(&codes)?
                                .iter()
                                .zip(&want_codes)
                                .filter(|(a, b)| a != b)
                                .count(),
                            stream
                                .clone_dtoh(&scales)?
                                .iter()
                                .zip(&want_scales)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count(),
                        )
                    } else {
                        (0, 0)
                    };
                    let bad = fake_bad + code_bad + scale_bad;
                    println!(
                        "E4M3_QUANT source={name} stored_input=[{rows},{width}] graph={graph} capture_outputs={capture_outputs} scale_bad={scale_bad} code_bad={code_bad} fake_bad={fake_bad} {}",
                        if bad == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(bad != 0);
                    cases += 1;
                }
            }
        }
        println!("E4M3_QUANT_SUMMARY cases={cases} defects={defects}");
        ensure!(
            defects == 0,
            "E4M3 activation quantization DEFECT cases={defects}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "GPU window required"]
    fn token_quantization_live_rows_gpu() -> Result<()> {
        use super::super::{contracts::SeqRef, q38_tests::view};
        use crate::fp8::fake_quant_reference;
        use cudarc::driver::{DevicePtr, sys};
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let stream = &device.stream;
        let op = Fp8Ops::new(&device.ctx)?;
        let (rows, width) = (32usize, 257usize);
        let code_sentinel = 0xa5u8;
        let scale_sentinel = bf16::from_bits(0x7fc5);
        let mut defects = 0;
        let mut cases = 0;
        for (row_base, row_capacity) in [(0u32, 32u32), (3, 17)] {
            let values = (0..rows * width)
                .map(|i| {
                    let row = i / width;
                    if row < row_base as usize || row >= 20 {
                        scale_sentinel
                    } else {
                        bf16::from_f32((((row * 37 + (i % width) * 11) % 103) as f32 - 51.0) / 32.0)
                    }
                })
                .collect::<Vec<_>>();
            ensure!(
                values[20 * width..].iter().all(|v| v.is_nan()),
                "live-row inactive-NaN fixture invalid"
            );
            for graph in [false, true] {
                let mut x = stream.clone_htod(&values)?;
                let mut codes = stream.clone_htod(&vec![code_sentinel; values.len()])?;
                let mut scales = stream.clone_htod(&vec![scale_sentinel; rows])?;
                let mut metadata = stream.clone_htod(&[3i32, 1, 4, 64, 1, 0, 0, 0])?;
                let domain = RowDomain {
                    meta: SeqRef {
                        ptr: metadata.device_ptr(stream).0,
                    },
                    row_base,
                    row_capacity,
                };
                let x_view = view(&x, stream, [rows as u32, width as u32], [width as u64, 1])?;
                let code_view = TensorMut(view(
                    &codes,
                    stream,
                    [rows as u32, width as u32],
                    [width as u64, 1],
                )?);
                let scale_view = TensorMut(view(&scales, stream, [rows as u32], [1])?);
                device.finish_upload()?;
                let execute = || {
                    op.fake_quant(
                        stream,
                        &mut TensorMut(x_view),
                        Some(&code_view),
                        Some(&scale_view),
                        Some(domain),
                    )
                };
                let captured = if graph {
                    stream.begin_capture(
                        sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    let run = execute();
                    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                    run?;
                    Some(captured?.context("empty live-row quantization graph")?)
                } else {
                    None
                };
                for valid in [1i32, 17, 20, 5] {
                    stream.memcpy_htod(&values, &mut x)?;
                    stream.memcpy_htod(&vec![code_sentinel; values.len()], &mut codes)?;
                    stream.memcpy_htod(&vec![scale_sentinel; rows], &mut scales)?;
                    stream.memcpy_htod(&[3, valid, 3 + valid, 64, 1, 0, 0, 0], &mut metadata)?;
                    if let Some(captured) = &captured {
                        captured.launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&x)?;
                    let got_codes = stream.clone_dtoh(&codes)?;
                    let got_scales = stream.clone_dtoh(&scales)?;
                    let mut bad = 0;
                    for (row, got_scale) in got_scales.iter().enumerate() {
                        let range = row * width..(row + 1) * width;
                        let active = row >= row_base as usize
                            && row < (row_base + row_capacity) as usize
                            && row < valid as usize;
                        if active {
                            let (s, c, f) = fake_quant_reference(&values[range.clone()])?;
                            ensure!(
                                got[range.clone()].iter().all(|v| v.is_finite()),
                                "nonfinite active quantization output"
                            );
                            bad += usize::from(got_scale.to_bits() != s.to_bits());
                            bad += got_codes[range.clone()]
                                .iter()
                                .zip(&c)
                                .filter(|(a, b)| a != b)
                                .count();
                            bad += got[range]
                                .iter()
                                .zip(&f)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        } else {
                            bad += usize::from(got_scale.to_bits() != scale_sentinel.to_bits());
                            bad += got_codes[range.clone()]
                                .iter()
                                .filter(|&&v| v != code_sentinel)
                                .count();
                            bad += got[range.clone()]
                                .iter()
                                .zip(&values[range])
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                        }
                    }
                    println!(
                        "E4M3_LIVE_ROWS stored_input=[{rows},{width}] row_base={row_base} row_capacity={row_capacity} valid={valid} graph={graph} bit_mismatches={bad} {}",
                        if bad == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(bad != 0);
                    cases += 1;
                }
            }
        }
        println!("E4M3_LIVE_ROWS_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "live-row quantization DEFECT cases={defects}");
        Ok(())
    }

    #[test]
    fn moe_ffn_binding_uses_native_router_and_shared_layouts() -> Result<()> {
        let spec = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let shapes = [
            ("mlp.gate.weight", [256, 2048]),
            ("mlp.shared_expert.gate_proj.weight", [512, 2048]),
            ("mlp.shared_expert.up_proj.weight", [512, 2048]),
            ("mlp.shared_expert.down_proj.weight", [2048, 512]),
            ("mlp.shared_expert_gate.weight", [1, 2048]),
        ];
        for wrong in std::iter::once(None).chain((0..shapes.len()).map(Some)) {
            let mut linear = LayerLinearPlan {
                projections: std::collections::BTreeMap::new(),
                materialize: Vec::new(),
            };
            let mut ptr = 0x1000u64;
            for (index, (role, mut shape)) in shapes.iter().copied().enumerate() {
                if wrong == Some(index) {
                    shape[1] /= 2;
                }
                let bytes = u64::from(shape[0]) * u64::from(shape[1]) * 2;
                let weight = unsafe { Tensor::new(ptr, shape, [u64::from(shape[1]), 1], bytes)? };
                linear.projections.insert(
                    role.into(),
                    BoundProjection {
                        resident: LinearWeightRef::Bf16(weight),
                        prefill: weight,
                    },
                );
                ptr += bytes;
            }
            let bound = BoundFfn::bind(&spec, &mut linear);
            if let Some(index) = wrong {
                ensure!(
                    bound.is_err(),
                    "accepted wrong FFN layout: projection={index}"
                );
            } else {
                let BoundFfn::Moe {
                    router,
                    shared,
                    shared_gate,
                } = bound?
                else {
                    anyhow::bail!("MoE layer selected the dense FFN plan");
                };
                assert_eq!(router.shape, [256, 2048]);
                assert_eq!(
                    shared.map(|weight| weight.shape),
                    [[512, 2048], [512, 2048], [2048, 512]]
                );
                assert_eq!(shared_gate.shape, [2048]);
                assert_eq!(shared[0].ptr, router.ptr + router.span_bytes);
                assert_eq!(shared[1].ptr, shared[0].ptr + shared[0].span_bytes);
                assert_eq!(shared[2].ptr, shared[1].ptr + shared[1].span_bytes);
                assert_eq!(shared_gate.ptr, shared[2].ptr + shared[2].span_bytes);
                assert!(linear.projections.is_empty());
                assert!(linear.materialize.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "coordinator-granted GPU window and FF_Q38_PROBE_DIR required"]
    fn mixer_geometry_linear_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let stream = &device.stream;
        let root = PathBuf::from(
            std::env::var_os("FF_Q38_PROBE_DIR").context("FF_Q38_PROBE_DIR is required")?,
        );
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            &device,
            &ops,
            &root,
            Rc::new(Closure::new(false, "native BF16 mixer gate".into())),
        )?;
        let mut defects = 0;
        let mut cases = 0;
        for (n, k) in [(8192u32, 2048u32), (32, 2048), (129, 257)] {
            let role = "linear_attn.in_proj_qkv.weight";
            let name = format!("model.language_model.layers.0.{role}");
            let values: Vec<_> = (0..n as usize * k as usize)
                .map(|i| {
                    bf16::from_f32(
                        (((i / k as usize) * 37 + i % k as usize * 11) % 53) as f32 / 32.0
                            - 26.0 / 32.0,
                    )
                })
                .collect();
            let weight = stream.clone_htod(&values)?;
            let bytes = values.len() * 2;
            let manifest = TensorManifest {
                entries: [(
                    name.clone(),
                    TensorEntry {
                        dtype: safetensors::Dtype::BF16,
                        shape: vec![n as usize, k as usize],
                        shard: "synthetic.safetensors".into(),
                        bytes,
                        load: true,
                        data_start: 0,
                        data_offsets: [0, bytes as u64],
                        scope: TensorScope::Text,
                        layer: Some(0),
                        role: role.into(),
                        scale_name: None,
                    },
                )]
                .into(),
                resident_bytes: bytes,
            };
            let catalog = WeightCatalog {
                tensors: [(name.clone(), StoredTensor::Bf16(weight))].into(),
                manifest,
                bytes,
                upload_content_sha256: None,
                upload_hashed_bytes: None,
                load_wall_ms: 0.0,
            };
            let dequant = crate::linear_plan::LayerDequantPlan::from_manifest(
                &catalog.manifest,
                0,
                crate::model_spec::LinearFormat::Bf16,
            )?;
            ensure!(
                dequant.bytes == 0 && dequant.slots.is_empty(),
                "native BF16 fixture allocated materialization storage"
            );
            let plan = LayerLinearPlan::bind(&dequant, &catalog, None, stream)?;
            for m in [1u32, 2, 5, 8, 4096] {
                plan.prepare(&blas, m)?;
                let mut inputs = vec![bf16::ZERO; m as usize * k as usize];
                for row in 0..m as usize {
                    inputs[row * k as usize + (row * 17 + 13) % k as usize] = bf16::ONE;
                }
                let input_store = stream.clone_htod(&inputs)?;
                let output = stream.alloc_zeros::<bf16>(m as usize * n as usize)?;
                use cudarc::driver::DevicePtr;
                let input = unsafe {
                    Tensor::new(
                        input_store.device_ptr(stream).0,
                        [m, k],
                        [u64::from(k), 1],
                        (input_store.len() * 2) as u64,
                    )?
                };
                let target = TensorMut(unsafe {
                    Tensor::new(
                        output.device_ptr(stream).0,
                        [m, n],
                        [u64::from(n), 1],
                        (output.len() * 2) as u64,
                    )?
                });
                device.finish_upload()?;
                for (phase, graph) in [
                    (ExecutionPhase::Prefill, false),
                    (ExecutionPhase::Prefill, true),
                ]
                .into_iter()
                .chain((m == 1).then_some((ExecutionPhase::Decode, false)))
                {
                    let execute =
                        || plan.run(&blas, role, phase, input, &mut TensorMut(target.read()));
                    if graph {
                        use cudarc::driver::sys;
                        stream.begin_capture(
                            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        let run = execute();
                        let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
                        run?;
                        captured?
                            .context("empty native BF16 geometry graph")?
                            .launch()?;
                    } else {
                        execute()?;
                    }
                    stream.synchronize()?;
                    let got = stream.clone_dtoh(&output)?;
                    ensure!(
                        got.iter().all(|v| v.is_finite()),
                        "nonfinite native BF16 linear output"
                    );
                    let mismatches = got
                        .iter()
                        .enumerate()
                        .filter(|(i, v)| {
                            let row = *i / n as usize;
                            let column = (row * 17 + 13) % k as usize;
                            v.to_bits() != values[(*i % n as usize) * k as usize + column].to_bits()
                        })
                        .count();
                    println!(
                        "BF16_LINEAR stored_weight=[{n},{k}] rows={m} phase={phase:?} graph={graph} dequant_bytes=0 mismatches={mismatches} {}",
                        if mismatches == 0 { "PASS" } else { "DEFECT" }
                    );
                    defects += usize::from(mismatches != 0);
                    cases += 1;
                }
            }
        }
        println!("BF16_LINEAR_SUMMARY cases={cases} defects={defects}");
        ensure!(defects == 0, "BF16_LINEAR DEFECT cases={defects}");
        Ok(())
    }
}
