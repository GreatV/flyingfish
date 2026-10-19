//! Typed Linear views; all BF16 plans execute through the shared Blas handle.
use super::{
    contracts::{Scratch, Tensor, TensorMut},
    weights::WeightCatalog,
};
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
    BlockFp8 {
        codes: Tensor<u8, 2>,
        scale: Tensor<bf16, 2>,
    },
}
pub(crate) struct Fp8Ops {
    dequant: CudaFunction,
    small: [CudaFunction; 5],
}
impl Fp8Ops {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let m = super::cubin::module(ctx, "fp8_linear")?;
        Ok(Self {
            dequant: m.load_function("fp8_dequant_bf16")?,
            small: [
                m.load_function("fp8_gemv_bf16")?,
                m.load_function("fp8_skinny_bf16_m2")?,
                m.load_function("fp8_skinny_bf16_m3")?,
                m.load_function("fp8_skinny_bf16_m4")?,
                m.load_function("fp8_skinny_bf16_m5")?,
            ],
        })
    }
    fn check(codes: &Tensor<u8, 2>, scale: &Tensor<bf16, 2>) -> Result<()> {
        codes.validate()?;
        scale.validate()?;
        ensure!(
            codes.shape.iter().all(|&v| v <= i32::MAX as u32)
                && codes.stride == [u64::from(codes.shape[1]), 1],
            "invalid FP8 matrix"
        );
        let shape = codes.shape.map(|v| v.div_ceil(128));
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
        dst: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        Self::check(&codes, &scale)?;
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
                .launch_builder(&self.dequant)
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
        x: Tensor<bf16, 2>,
        y: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        Self::check(&codes, &scale)?;
        validate_call(codes.shape, x, &y.0)?;
        ensure!(
            crate::linear_plan::direct_fp8_rows(x.shape[0] as usize),
            "direct FP8 linear requires M1..5"
        );
        disjoint(&codes, &y.0)?;
        disjoint(&scale, &y.0)?;
        unsafe {
            stream
                .launch_builder(&self.small[x.shape[0] as usize - 1])
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
                let scale_name = format!(
                    "{}.weight_scale_inv",
                    name.strip_suffix(".weight").context("FP8 weight name")?
                );
                let scale = self.bf16(&scale_name)?;
                let sh = shape.map(|n| n.div_ceil(128));
                let (wp, _rw) = w.device_ptr(stream);
                let (sp, _rs) = scale.device_ptr(stream);
                Ok(LinearWeightRef::BlockFp8 {
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
    materialize: Vec<(Tensor<u8, 2>, Tensor<bf16, 2>, TensorMut<bf16, 2>)>,
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
        ensure!(
            spec.linear_format == crate::model_spec::LinearFormat::Bf16,
            "MoE FFN requires BF16 weights: linear_format={:?}",
            spec.linear_format
        );
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
                || !name.starts_with(prefix)
                || !name.ends_with(".weight")
                || entry.shape.len() != 2
            {
                continue;
            }
            let resident = catalog.linear_ref(name, stream)?;
            let prefill = match &resident {
                LinearWeightRef::Bf16(w) => *w,
                LinearWeightRef::BlockFp8 { codes, scale } => {
                    let LinearWeightRef::Bf16(w) = plan.weight(
                        name,
                        arena.context("FP8 layer has no materialization arena")?,
                    )?
                    else {
                        unreachable!()
                    };
                    materialize.push((*codes, *scale, TensorMut(w)));
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
        for (codes, scale, dst) in &mut self.materialize {
            blas.fp8_ops()?.materialize(stream, *codes, *scale, dst)?;
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
