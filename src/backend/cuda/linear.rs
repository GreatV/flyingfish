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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
impl LayerLinearPlan {
    pub fn bind(
        plan: &crate::linear_plan::LayerDequantPlan,
        catalog: &WeightCatalog,
        arena: &Scratch,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        arena.validate(plan.bytes)?;
        let prefix = format!("model.language_model.layers.{}.", plan.layer);
        let mut projections = std::collections::BTreeMap::new();
        let mut materialize = Vec::new();
        for (name, entry) in &catalog.manifest.entries {
            if !entry.load
                || !name.starts_with(&prefix)
                || !name.ends_with(".weight")
                || entry.shape.len() != 2
            {
                continue;
            }
            let resident = catalog.linear_ref(name, stream)?;
            let prefill = match &resident {
                LinearWeightRef::Bf16(w) => *w,
                LinearWeightRef::BlockFp8 { codes, scale } => {
                    let LinearWeightRef::Bf16(w) = plan.weight(name, arena)? else {
                        unreachable!()
                    };
                    materialize.push((*codes, *scale, TensorMut(w)));
                    w
                }
            };
            projections.insert(entry.role.clone(), BoundProjection { resident, prefill });
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
    pub fn materialize(&mut self, ops: &Fp8Ops, stream: &Arc<CudaStream>) -> Result<()> {
        for (codes, scale, dst) in &mut self.materialize {
            ops.materialize(stream, *codes, *scale, dst)?;
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
