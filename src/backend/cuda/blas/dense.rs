//! Setup selection for packed BF16 TN projections with FP32 accumulation.
use super::{Blas, LtAlgorithm, Plan, algo_config, heuristic_algos, sys};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeSet, mem::MaybeUninit};

type Algo = sys::cublasLtMatmulAlgo_t;
type Cap = sys::cublasLtMatmulAlgoCapAttributes_t;
type Attr = sys::cublasLtMatmulAlgoConfigAttributes_t;

fn capability<T: Default>(algo: &Algo, key: Cap) -> Result<T> {
    let mut value = T::default();
    let mut bytes = 0;
    unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(
            algo,
            key,
            (&raw mut value).cast(),
            size_of::<T>(),
            &mut bytes,
        )
        .result()?;
    }
    ensure!(
        bytes == size_of::<T>(),
        "Lt capability {key:?} returned {bytes} bytes"
    );
    Ok(value)
}

fn options(algo: &Algo, key: Cap) -> Result<Vec<u32>> {
    let mut bytes = 0;
    unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(algo, key, std::ptr::null_mut(), 0, &mut bytes)
            .result()?;
    }
    ensure!(
        bytes.is_multiple_of(size_of::<u32>()),
        "Lt option list {key:?} length {bytes}"
    );
    if bytes == 0 {
        return Ok(vec![0]);
    }
    let mut values = vec![0u32; bytes / size_of::<u32>()];
    let mut written = 0;
    unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(
            algo,
            key,
            values.as_mut_ptr().cast(),
            bytes,
            &mut written,
        )
        .result()?;
    }
    ensure!(written == bytes, "Lt option list {key:?} changed length");
    Ok(values)
}

fn configure<T>(algo: &mut Algo, key: Attr, value: T) -> Result<()> {
    unsafe {
        sys::cublasLtMatmulAlgoConfigSetAttribute(
            algo,
            key,
            (&raw const value).cast(),
            size_of::<T>(),
        )
        .result()?;
    }
    Ok(())
}

impl Blas {
    fn init_dense_algo(&self, id: i32) -> Result<Algo> {
        let dt = self.half_dtype;
        let mut algo = MaybeUninit::uninit();
        unsafe {
            sys::cublasLtMatmulAlgoInit(
                self.handle,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
                dt,
                dt,
                dt,
                dt,
                id,
                algo.as_mut_ptr(),
            )
            .result()?;
            Ok(algo.assume_init())
        }
    }

    fn restore_dense_algo(&self, id: LtAlgorithm) -> Result<Algo> {
        let mut algo = self.init_dense_algo(id.id)?;
        for (attr, value) in [
            (Attr::CUBLASLT_ALGO_CONFIG_TILE_ID, id.tile),
            (Attr::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, id.split_k),
            (Attr::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, id.reduction),
            (Attr::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, id.swizzle),
            (Attr::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, id.custom),
            (Attr::CUBLASLT_ALGO_CONFIG_STAGES_ID, id.stages),
        ] {
            configure(&mut algo, attr, value)?;
        }
        configure(
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
            id.inner,
        )?;
        configure(
            &mut algo,
            Attr::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            id.cluster,
        )?;
        ensure!(
            algo_config(&algo)? == id,
            "Lt algorithm configuration failed to round trip: {id:?}"
        );
        Ok(algo)
    }

    fn dense_algorithms(&self, plan: &Plan, desc: &Descriptor) -> Result<Vec<Algo>> {
        let budget = self.workspace_bytes();
        let incumbent = plan.algo.context("dense Lt incumbent missing")?;
        let mut search = Search {
            blas: self,
            plan,
            budget,
            alignment: desc.alignment,
            seen: BTreeSet::new(),
            legal: Vec::new(),
        };
        search.consider(incumbent, false)?;
        ensure!(
            !search.legal.is_empty(),
            "dense Lt incumbent fails its numerical or resource contract"
        );
        // The deployed descriptor can have been prepared with a smaller workspace.
        let pref = super::lt::create_matmul_pref()?;
        let result = (|| {
            let mask =
                sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
            unsafe {
                super::lt::set_matmul_pref_attribute(pref,sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,(&raw const budget).cast(),size_of::<usize>())?;
                super::lt::set_matmul_pref_attribute(pref,sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,(&raw const mask).cast(),size_of::<u32>())?;
                heuristic_algos(self.handle, plan.desc, plan.a, plan.b, plan.c, pref, 32)
            }
        })();
        unsafe {
            super::lt::destroy_matmul_pref(pref)?;
        }
        for heuristic in result? {
            search.consider(heuristic.algo, false)?;
            if search.legal.len() == 8 {
                break;
            }
        }
        if search.legal.len() == 1 {
            search.enumerate()?;
        }
        if desc.ffn_tail.is_some() {
            search.rectangular(incumbent)?;
        }
        Ok(search.legal)
    }
}

struct Search<'a> {
    blas: &'a Blas,
    plan: &'a Plan,
    budget: usize,
    alignment: [u32; 4],
    seen: BTreeSet<LtAlgorithm>,
    legal: Vec<Algo>,
}

impl Search<'_> {
    fn consider(&mut self, algo: Algo, trial_tile: bool) -> Result<()> {
        let id = algo_config(&algo)?;
        if !self.seen.insert(id) {
            return Ok(());
        }
        let _supported_reductions: u32 =
            capability(&algo, Cap::CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK)?;
        if id.split_k > 1
            || id.reduction != sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as i32
        {
            return Ok(());
        }
        let flags: u64 = capability(&algo, Cap::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)?;
        // cuBLASLt numerical implementation flags: BF16 input (bit 17), F32 accumulator (bit 9).
        if flags & (1 << 17) == 0 || flags & (1 << 9) == 0 {
            return Ok(());
        }
        for (key, alignment) in [
            Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
            Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
            Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
            Cap::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
        ]
        .into_iter()
        .zip(self.alignment)
        {
            if capability::<u32>(&algo, key)? > alignment {
                return Ok(());
            }
        }
        let mut check = MaybeUninit::uninit();
        let p = self.plan;
        let status = unsafe {
            sys::cublasLtMatmulAlgoCheck(
                self.blas.handle,
                p.desc,
                p.a,
                p.b,
                p.c,
                p.c,
                &algo,
                check.as_mut_ptr(),
            )
        };
        use sys::cublasStatus_t::{
            CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_NOT_SUPPORTED,
        };
        let rejects = |s| {
            matches!(s, CUBLAS_STATUS_NOT_SUPPORTED | CUBLAS_STATUS_ARCH_MISMATCH)
                || (trial_tile && s == CUBLAS_STATUS_INVALID_VALUE)
        };
        if rejects(status) {
            return Ok(());
        }
        status.result()?;
        let check = unsafe { check.assume_init() };
        if rejects(check.state) {
            return Ok(());
        }
        check.state.result()?;
        if check.workspaceSize <= self.budget {
            self.legal.push(algo);
        }
        Ok(())
    }

    fn rectangular(&mut self, incumbent: Algo) -> Result<()> {
        use sys::cublasLtMatmulTile_t::{CUBLASLT_MATMUL_TILE_64x128, CUBLASLT_MATMUL_TILE_128x64};
        let mut tiles = options(&incumbent, Cap::CUBLASLT_ALGO_CAP_TILE_IDS)?;
        tiles.retain(|&t| {
            t == CUBLASLT_MATMUL_TILE_64x128 as u32 || t == CUBLASLT_MATMUL_TILE_128x64 as u32
        });
        tiles.sort_unstable();
        let current = u32::try_from(algo_config(&incumbent)?.stages)?;
        let mut stages = options(&incumbent, Cap::CUBLASLT_ALGO_CAP_STAGES_IDS)?;
        stages.sort_unstable_by_key(|&stage| (stage.abs_diff(current), stage));
        let limit = self.legal.len() + 4;
        for stage in stages {
            for &tile in &tiles {
                let mut algo = incumbent;
                let config = configure(&mut algo, Attr::CUBLASLT_ALGO_CONFIG_TILE_ID, tile)
                    .and_then(|()| {
                        configure(&mut algo, Attr::CUBLASLT_ALGO_CONFIG_STAGES_ID, stage)
                    });
                if let Err(error) = config {
                    use sys::cublasStatus_t::{
                        CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_NOT_SUPPORTED,
                    };
                    if error
                        .downcast_ref::<cudarc::cublaslt::result::CublasError>()
                        .is_some_and(|e| {
                            matches!(
                                e.0,
                                CUBLAS_STATUS_INVALID_VALUE | CUBLAS_STATUS_NOT_SUPPORTED
                            )
                        })
                    {
                        continue;
                    }
                    return Err(error);
                }
                self.consider(algo, true)?;
                if self.legal.len() == limit {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn enumerate(&mut self) -> Result<()> {
        let dt = self.blas.half_dtype;
        let mut ids = [0i32; 256];
        let mut count = 0;
        unsafe {
            sys::cublasLtMatmulAlgoGetIds(
                self.blas.handle,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
                dt,
                dt,
                dt,
                dt,
                ids.len() as i32,
                ids.as_mut_ptr(),
                &mut count,
            )
            .result()?;
        }
        let mut variants = Vec::new();
        for &id in ids
            .get(..usize::try_from(count)?)
            .context("Lt algorithm count")?
        {
            let algo = self.blas.init_dense_algo(id)?;
            let tiles = options(&algo, Cap::CUBLASLT_ALGO_CAP_TILE_IDS)?;
            let stages = options(&algo, Cap::CUBLASLT_ALGO_CAP_STAGES_IDS)?;
            variants.push((algo, tiles, stages));
        }
        let mut tried = 0;
        let max = variants
            .iter()
            .map(|(_, tiles, stages)| tiles.len() * stages.len())
            .max()
            .unwrap_or(0);
        for i in 0..=max {
            for (base, tiles, stages) in &variants {
                let mut algo = *base;
                if i > 0 {
                    let Some(stage) = stages.get((i - 1) / tiles.len()) else {
                        continue;
                    };
                    configure(
                        &mut algo,
                        Attr::CUBLASLT_ALGO_CONFIG_TILE_ID,
                        tiles[(i - 1) % tiles.len()],
                    )?;
                    configure(&mut algo, Attr::CUBLASLT_ALGO_CONFIG_STAGES_ID, *stage)?;
                }
                if self.seen.contains(&algo_config(&algo)?) {
                    continue;
                }
                self.consider(algo, false)?;
                tried += 1;
                if tried == 128 || self.legal.len() == 8 {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

use crate::backend::cuda::contracts::Tensor;
use crate::backend::setup::{CacheKey, LinearShape};
use half::bf16;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Descriptor {
    key: CacheKey,
    sm: u32,
    shape: LinearShape,
    alignment: [u32; 4],
    workspace: usize,
    ffn_tail: Option<usize>,
    incumbent: LtAlgorithm,
    family: bool,
}

#[derive(Serialize, Deserialize)]
struct Selection {
    descriptor: Descriptor,
    incumbent: LtAlgorithm,
    algorithm: LtAlgorithm,
    required_workspace: usize,
    medians_ms: Vec<(LtAlgorithm, f32)>,
    challenger_margin: Option<f64>,
    fill_seconds: f64,
}

struct Operands {
    weight: Tensor<bf16, 2>,
    input: Tensor<bf16, 2>,
    output: Tensor<bf16, 2>,
}

impl Operands {
    fn alignments(&self) -> [u32; 4] {
        [
            self.weight.ptr,
            self.input.ptr,
            self.output.ptr,
            self.output.ptr,
        ]
        .map(|p| 1u32 << p.trailing_zeros().min(8))
    }

    fn launch(&self, blas: &Blas, plan: &Plan, algo: &Algo) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let workspace = blas.workspace.borrow();
        let (scratch, _guard) = workspace.device_ptr(&blas.stream);
        let (alpha, beta) = (1f32, 0f32);
        unsafe {
            super::lt::matmul(
                blas.handle,
                plan.desc,
                (&raw const alpha).cast(),
                (&raw const beta).cast(),
                self.weight.ptr as _,
                plan.a,
                self.input.ptr as _,
                plan.b,
                self.output.ptr as _,
                plan.c,
                self.output.ptr as _,
                plan.c,
                algo,
                scratch as _,
                workspace.len(),
                blas.stream.cu_stream() as _,
            )?;
        }
        Ok(())
    }
}

fn measured_winner(medians: &[(LtAlgorithm, f32)]) -> Result<(usize, Option<f64>)> {
    ensure!(
        !medians.is_empty() && medians.iter().all(|(_, t)| t.is_finite() && *t > 0.0),
        "dense Lt invalid medians"
    );
    let Some((i, &(_, fastest))) = medians
        .iter()
        .enumerate()
        .skip(1)
        .min_by(|a, b| a.1.1.total_cmp(&b.1.1).then_with(|| a.1.0.cmp(&b.1.0)))
    else {
        return Ok((0, None));
    };
    let baseline = f64::from(medians[0].1);
    let improvement = (baseline - f64::from(fastest)) / baseline;
    Ok((if improvement >= 0.03 { i } else { 0 }, Some(improvement)))
}

impl Blas {
    fn time_dense(
        &self,
        plan: &Plan,
        operands: &Operands,
        legal: &[Algo],
    ) -> Result<Vec<(LtAlgorithm, f32)>> {
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let begin = self.stream.context().new_event(Some(flags))?;
        let end = self.stream.context().new_event(Some(flags))?;
        let mut samples = vec![Vec::with_capacity(11); legal.len()];
        for round in 0..14 {
            for i in 0..legal.len() {
                let index = if round % 2 == 0 {
                    i
                } else {
                    legal.len() - 1 - i
                };
                if round >= 3 {
                    begin.record(&self.stream)?;
                }
                operands.launch(self, plan, &legal[index])?;
                if round >= 3 {
                    end.record(&self.stream)?;
                    end.synchronize()?;
                    samples[index].push(begin.elapsed_ms(&end)?);
                }
            }
        }
        samples
            .into_iter()
            .zip(legal)
            .map(|(mut times, algo)| {
                ensure!(
                    times.len() == 11 && times.iter().all(|t| t.is_finite() && *t > 0.0),
                    "dense Lt invalid launch times"
                );
                times.sort_unstable_by(f32::total_cmp);
                Ok((algo_config(algo)?, times[5]))
            })
            .collect()
    }
}

fn repeat_matches(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    scan: &cudarc::driver::CudaFunction,
    flags: &mut cudarc::driver::CudaSlice<u32>,
    pair: [u64; 2],
    len: usize,
    sms: usize,
) -> Result<bool> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let blocks = u32::try_from(len.div_ceil(256).min(8 * sms))?;
    ensure!(blocks > 0, "dense Lt repeat scan has no elements or SMs");
    stream.memset_zeros(flags)?;
    unsafe {
        stream
            .launch_builder(scan)
            .arg(&pair[0])
            .arg(&pair[1])
            .arg(&u64::try_from(len)?)
            .arg(&mut *flags)
            .launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let mut bits = [0u32];
    stream.memcpy_dtoh(flags, &mut bits)?;
    stream.synchronize()?;
    let [bits] = bits;
    ensure!(bits & 2 == 0, "nonfinite dense Lt setup output");
    ensure!(bits & 4 != 0, "dense Lt setup output has no signal");
    Ok(bits & 1 == 0)
}

struct Agreement {
    different: bool,
    largest: f32,
    peak: f32,
    residual: f64,
    energy: f64,
}

impl Agreement {
    fn read(a: u64, b: u64, len: usize) -> Result<Self> {
        let mut result = Self {
            different: false,
            largest: 0.0,
            peak: 0.0,
            residual: 0.0,
            energy: 0.0,
        };
        let mut lhs = vec![bf16::ZERO; len.min(1 << 20)];
        let mut rhs = lhs.clone();
        for at in (0..len).step_by(lhs.len()) {
            let count = lhs.len().min(len - at);
            let offset = u64::try_from(at * 2)?;
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(&mut lhs[..count], a + offset)?;
                cudarc::driver::result::memcpy_dtoh_sync(&mut rhs[..count], b + offset)?;
            }
            for (&reference, &trial) in lhs[..count].iter().zip(&rhs[..count]) {
                let (x, y) = (reference.to_f32(), trial.to_f32());
                ensure!(
                    x.is_finite() && y.is_finite(),
                    "nonfinite dense Lt setup output"
                );
                result.different |= reference.to_bits() != trial.to_bits();
                let d = (x - y).abs();
                result.largest = result.largest.max(d);
                result.peak = result.peak.max(x.abs());
                result.residual += f64::from(d).powi(2);
                result.energy += f64::from(x).powi(2);
            }
        }
        ensure!(result.energy > 0.0, "dense Lt setup output has no signal");
        Ok(result)
    }

    fn accepts(&self, exact: bool) -> bool {
        !self.different
            || (!exact
                && self.largest <= 2f32.powi(self.peak.log2().floor() as i32 - 7)
                && self.residual <= self.energy * (1.0 / 256.0f64).powi(2))
    }
}

impl Blas {
    fn row_order_matches(
        &self,
        ops: &Operands,
        desc: &Descriptor,
        algo: Algo,
        saved: u64,
    ) -> Result<bool> {
        let Some(remainder) = desc.ffn_tail else {
            return Ok(true);
        };
        let shape = desc.shape;
        let rows = remainder.max(1).min(shape.rows);
        if rows == shape.rows {
            return Ok(true);
        }
        let smaller = Plan::build(
            self.handle,
            (rows, shape.output, shape.input, shape.input),
            desc.workspace,
            None,
            super::PlanOperands {
                ab: self.half_dtype,
                c: self.half_dtype,
                device_alpha: false,
            },
            super::PlanAlgo::Fixed(algo),
        );
        let p = match smaller {
            Ok(plan) => plan,
            Err(error) if Self::is_not_supported(&error) => return Ok(false),
            Err(error) => return Err(error),
        };
        if p.workspace > desc.workspace {
            return Ok(false);
        }
        let start = shape.rows - rows;
        let mut tail = Operands {
            weight: ops.weight,
            input: ops.input,
            output: ops.output,
        };
        tail.input.ptr += u64::try_from(start * shape.input * 2)?;
        tail.launch(self, &p, &algo)?;
        self.stream.synchronize()?;
        let compare = Agreement::read(
            saved + u64::try_from(start * shape.output * 2)?,
            ops.output.ptr,
            rows * shape.output,
        )?;
        Ok(!compare.different)
    }

    fn fill_dense(
        &self,
        plan: &Plan,
        desc: Descriptor,
        ops: &Operands,
        copies: [u64; 2],
    ) -> Result<Selection> {
        let start = std::time::Instant::now();
        let candidates = self.dense_algorithms(plan, &desc)?;
        let mut legal = Vec::new();
        let len = desc
            .shape
            .rows
            .checked_mul(desc.shape.output)
            .context("dense setup output size")?;
        let copy = |dest| -> Result<()> {
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dest,
                    ops.output.ptr,
                    len * 2,
                    self.stream.cu_stream(),
                )?;
            }
            Ok(())
        };
        let scan = super::cubin::module(self.stream.context(), "q38_common")?
            .load_function("bf16_repeat_flags")?;
        let mut flags = self.stream.alloc_zeros::<u32>(1)?;
        for (i, algo) in candidates.into_iter().enumerate() {
            ops.launch(self, plan, &algo)?;
            if i == 0 {
                copy(copies[0])?;
            }
            copy(copies[1])?;
            ops.launch(self, plan, &algo)?;
            ensure!(
                repeat_matches(
                    &self.stream,
                    &scan,
                    &mut flags,
                    [copies[1], ops.output.ptr],
                    len,
                    self.skinny_device.0,
                )?,
                "unsplit dense Lt same-plan repeat differs: {:?}",
                algo_config(&algo)?
            );
            let exact = desc.ffn_tail.is_some_and(|rows| rows > 0);
            if i == 0
                || (Agreement::read(copies[0], ops.output.ptr, len)?.accepts(exact)
                    && self.row_order_matches(ops, &desc, algo, copies[1])?)
            {
                legal.push(algo);
            }
        }
        let medians_ms = self.time_dense(plan, ops, &legal)?;
        let (index, challenger_margin) = measured_winner(&medians_ms)?;
        let algorithm = algo_config(&legal[index])?;
        let fixed = Plan::build(
            self.handle,
            (
                desc.shape.rows,
                desc.shape.output,
                desc.shape.input,
                desc.shape.input,
            ),
            desc.workspace,
            None,
            super::PlanOperands {
                ab: self.half_dtype,
                c: self.half_dtype,
                device_alpha: false,
            },
            super::PlanAlgo::Fixed(legal[index]),
        )?;
        Ok(Selection {
            descriptor: desc,
            incumbent: medians_ms[0].0,
            algorithm,
            required_workspace: fixed.workspace,
            medians_ms,
            challenger_margin,
            fill_seconds: start.elapsed().as_secs_f64(),
        })
    }
}

impl Descriptor {
    fn selection(
        &self,
        runtime: &std::path::Path,
        measure: impl FnOnce() -> Result<Selection>,
    ) -> Result<(Selection, bool)> {
        std::fs::create_dir_all(runtime)?;
        // dense-v1 uses packed BF16 TN, FP32 compute/scale and the unbiased default epilogue.
        let path = runtime.join(format!(
            "dense-v1-{}.json",
            crate::backend::cuda::closure::hash(&serde_json::to_vec(self)?)
        ));
        let read = || -> Result<(Selection, bool)> {
            let entry: Selection = serde_json::from_slice(&std::fs::read(&path)?)?;
            entry.validate(self)?;
            Ok((entry, false))
        };
        if path.try_exists()? {
            return read();
        }
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path.with_extension("lock"))?;
        lock.lock()?;
        if path.try_exists()? {
            return read();
        }
        let result = measure()?;
        result.validate(self)?;
        crate::backend::cuda::closure::write_cache(&path, &result)?;
        Ok((result, true))
    }
}

impl Selection {
    fn validate(&self, desc: &Descriptor) -> Result<()> {
        ensure!(
            &self.descriptor == desc,
            "dense Lt cache descriptor differs"
        );
        let limit = if desc.ffn_tail.is_some() { 12 } else { 8 };
        ensure!(
            self.medians_ms.len() <= limit,
            "dense Lt candidate count exceeds {limit}"
        );
        let (winner, margin) = measured_winner(&self.medians_ms)?;
        ensure!(
            self.fill_seconds.is_finite()
                && self.fill_seconds >= 0.0
                && self.medians_ms[0].0 == self.incumbent
                && self.medians_ms[winner].0 == self.algorithm
                && self
                    .medians_ms
                    .iter()
                    .map(|v| v.0)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.medians_ms.len()
                && match (margin, self.challenger_margin) {
                    (None, None) => true,
                    (Some(a), Some(b)) => (a - b).abs() <= 4.0 * f64::EPSILON,
                    _ => false,
                },
            "dense Lt cache winner or measured margin differs"
        );
        ensure!(
            self.required_workspace <= desc.workspace
                && self.algorithm.split_k <= 1
                && self.algorithm.reduction == 0,
            "dense Lt cache requires unsupported workspace or reduction"
        );
        Ok(())
    }
}

impl Blas {
    pub(crate) fn bind_dense_projection(
        &self,
        matrices: [Tensor<bf16, 2>; 3],
        copies: [Tensor<bf16, 2>; 2],
        source: &cudarc::driver::CudaSlice<bf16>,
        ffn_tail: Option<usize>,
        family: bool,
    ) -> Result<()> {
        use crate::backend::cuda::contracts::no_overlap;
        let setup_start = std::time::Instant::now();
        use cudarc::driver::DevicePtr;
        let [weight, input, output] = matrices;
        for t in matrices.into_iter().chain(copies) {
            t.validate()?;
            ensure!(
                t.stride == [u64::from(t.shape[1]), 1],
                "dense Lt qualification requires packed matrices"
            );
        }
        let shape = LinearShape {
            rows: input.shape[0] as usize,
            input: input.shape[1] as usize,
            output: weight.shape[0] as usize,
        };
        let adopt_gate = ffn_tail.is_some_and(|tail| shape.rows != tail);
        ensure!(
            weight.shape[1] == input.shape[1]
                && output.shape == [input.shape[0], weight.shape[0]]
                && copies.iter().all(|c| c.shape == output.shape),
            "dense Lt qualification geometry"
        );
        let operands = Operands {
            weight,
            input,
            output,
        };
        let protected = [weight, input, output, copies[0], copies[1]];
        for (i, a) in protected.iter().enumerate() {
            for b in &protected[i + 1..] {
                no_overlap(a, b, "setup matrix", "setup matrix")?;
            }
        }
        ensure!(
            self.stream.capture_status()?
                == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "dense Lt calibration inside stream capture"
        );
        self.closure.require_setup()?;
        let key = (shape.rows, shape.output, shape.input, 0, family);
        let algo = {
            let mut plans = self.plans.borrow_mut();
            let plan = plans
                .get_mut(&key)
                .context("dense Lt projection lacks an incumbent")?;
            let incumbent = plan.identity()?;
            let desc = Descriptor {
                key: self.calibration.key.clone(),
                sm: super::cubin::capability(self.stream.context())?,
                shape,
                alignment: operands.alignments(),
                workspace: self.workspace_bytes(),
                ffn_tail,
                incumbent,
                family,
            };
            let (choice, filled) = desc.selection(self.calibration.runtime(), || {
                let bytes = input
                    .span_bytes
                    .min(u64::from(input.shape[0]) * u64::from(input.shape[1]) * 2)
                    as usize;
                let (src, _guard) = source.device_ptr(&self.stream);
                let source_bytes = std::mem::size_of::<bf16>() * source.len();
                ensure!(source_bytes > 0, "dense setup activation is empty");
                for at in (0..bytes).step_by(source_bytes) {
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            input.ptr + at as u64,
                            src,
                            source_bytes.min(bytes - at),
                            self.stream.cu_stream(),
                        )?;
                    }
                }
                self.fill_dense(plan, desc.clone(), &operands, copies.map(|t| t.ptr))
            })?;
            let algo = self.restore_dense_algo(choice.algorithm)?;
            let bound = Plan::build(
                self.handle,
                (shape.rows, shape.output, shape.input, shape.input),
                desc.workspace,
                None,
                super::PlanOperands {
                    ab: self.half_dtype,
                    c: self.half_dtype,
                    device_alpha: false,
                },
                super::PlanAlgo::Fixed(algo),
            )?;
            ensure!(
                bound.workspace == choice.required_workspace,
                "dense Lt cache workspace recheck differs"
            );
            if crate::backend::diagnostics_enabled() {
                eprintln!(
                    "{}",
                    serde_json::json!({"dense_lt":{"filled":filled,"setup_seconds":setup_start.elapsed().as_secs_f64(),"selection":choice}})
                );
            }
            if !family {
                let original = std::mem::replace(plan, bound);
                #[cfg(test)]
                self.dense_parents
                    .borrow_mut()
                    .insert((shape, false), original);
                #[cfg(not(test))]
                drop(original);
            }
            algo
        };
        if adopt_gate {
            self.adopt_qualified_gate_family(shape.output, shape.input, algo)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn exchange_dense_parents<T>(
        parents: &mut std::collections::BTreeMap<(LinearShape, bool), T>,
        plans: &mut std::collections::BTreeMap<(usize, usize, usize, u32, bool), T>,
    ) -> Result<usize> {
        for ((shape, family), old) in parents.iter_mut() {
            let current = plans
                .get_mut(&(shape.rows, shape.output, shape.input, 0, *family))
                .context("dense test binding missing")?;
            std::mem::swap(current, old);
        }
        Ok(parents.len())
    }

    #[cfg(test)]
    pub(crate) fn dense_parent_count(&self, shape: LinearShape, family: bool) -> usize {
        usize::from(self.dense_parents.borrow().contains_key(&(shape, family)))
    }

    #[cfg(test)]
    pub(crate) fn swap_dense_parents(&self) -> Result<usize> {
        self.stream.synchronize()?;
        Self::exchange_dense_parents(
            &mut self.dense_parents.borrow_mut(),
            &mut self.plans.borrow_mut(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "CUDA window for production dense Lt repeat extents"]
    fn dense_repeat_matches_host_words() -> Result<()> {
        use crate::backend::cuda::Device;
        use cudarc::driver::DevicePtr;
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window required"
        );
        let device = Device::new(0)?;
        let stream = &device.stream;
        let scan = crate::backend::cuda::cubin::module(&device.ctx, "q38_common")?
            .load_function("bf16_repeat_flags")?;
        let mut flags = stream.alloc_zeros::<u32>(1)?;
        for len in [39_741 * 7_168, 3_057 * 28_672] {
            let mut a = stream.alloc_zeros::<bf16>(len)?;
            let mut b = stream.alloc_zeros::<bf16>(len)?;
            for (first, lhs, rhs, expected) in [
                (0, 0, 0, Err("dense Lt setup output has no signal")),
                (0, 0x8000, 1, Err("dense Lt setup output has no signal")),
                (0, 1, 1, Ok(true)),
                (0, 0x3f80, 0x3f81, Ok(false)),
                (1, 0, 0x8000, Ok(false)),
                (0, 0x7f80, 0x3f80, Err("nonfinite dense Lt setup output")),
                (0, 0, 0x7fc1, Err("nonfinite dense Lt setup output")),
            ] {
                stream.memcpy_htod(&[bf16::from_bits(first)], &mut a.slice_mut(..1))?;
                stream.memcpy_htod(&[bf16::from_bits(first)], &mut b.slice_mut(..1))?;
                stream.memcpy_htod(&[bf16::from_bits(lhs)], &mut a.slice_mut(len - 1..))?;
                stream.memcpy_htod(&[bf16::from_bits(rhs)], &mut b.slice_mut(len - 1..))?;
                let (ap, _a) = a.device_ptr(stream);
                let (bp, _b) = b.device_ptr(stream);
                let actual =
                    repeat_matches(stream, &scan, &mut flags, [ap, bp], len, device.info.sms);
                let host = Agreement::read(ap, bp, len).map(|value| !value.different);
                assert_eq!(
                    host.map_err(|e| e.to_string()),
                    expected.map_err(String::from)
                );
                assert_eq!(
                    actual.map_err(|e| e.to_string()),
                    expected.map_err(String::from),
                    "{len} words, tail {lhs:04x}/{rhs:04x}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn swap_restores_the_pre_qualification_family_plan() -> Result<()> {
        let shape = LinearShape {
            rows: 1024,
            output: 28_672,
            input: 5_376,
        };
        let key = |family| (shape.rows, shape.output, shape.input, 0, family);
        let qualified = 21u8;
        let previous = 6u8;
        let mut plans =
            std::collections::BTreeMap::from([(key(false), qualified), (key(true), qualified)]);
        let mut parents = std::collections::BTreeMap::from([
            ((shape, false), previous),
            ((shape, true), previous),
        ]);
        let bound = Blas::exchange_dense_parents(&mut parents, &mut plans)?;
        assert_eq!(bound, 2);
        assert_eq!(plans[&key(true)], previous);
        assert_eq!(plans[&key(false)], previous);
        Blas::exchange_dense_parents(&mut parents, &mut plans)?;
        assert_eq!(plans[&key(true)], qualified);
        Ok(())
    }

    #[test]
    fn dense_cache_rejects_an_unmeasured_or_noise_level_switch() -> Result<()> {
        let parent = LtAlgorithm {
            id: 1,
            tile: 0,
            split_k: 1,
            reduction: 0,
            swizzle: 0,
            custom: 0,
            stages: 0,
            inner: 0,
            cluster: 0,
        };
        let alt = LtAlgorithm { id: 2, ..parent };
        let descriptor = Descriptor {
            key: CacheKey {
                device_uuid: "gpu".into(),
                driver_version: "driver".into(),
                binary_version: "binary".into(),
                cublaslt_version: 1,
            },
            sm: 89,
            shape: LinearShape {
                rows: 67,
                input: 128,
                output: 256,
            },
            alignment: [256; 4],
            workspace: 0,
            ffn_tail: None,
            incumbent: parent,
            family: false,
        };
        let mut cached = Selection {
            descriptor: descriptor.clone(),
            incumbent: parent,
            algorithm: alt,
            required_workspace: 0,
            medians_ms: vec![(parent, 100.0), (alt, 97.0)],
            challenger_margin: Some(0.03),
            fill_seconds: 1.0,
        };
        cached.validate(&descriptor)?;
        cached.algorithm = LtAlgorithm { id: 7, ..alt };
        assert!(cached.validate(&descriptor).is_err());
        cached.medians_ms[1].1 = 98.0;
        cached.challenger_margin = Some(0.02);
        cached.algorithm = alt;
        assert!(cached.validate(&descriptor).is_err());
        cached.algorithm = parent;
        cached.validate(&descriptor)?;
        cached.challenger_margin = Some(0.04);
        assert!(cached.validate(&descriptor).is_err());
        cached.challenger_margin = Some(0.02);
        let dir = std::env::temp_dir().join(format!("ff-dense-lt-reorder-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        let (first, filled) = descriptor.selection(&dir, || Ok(cached))?;
        assert!(filled);
        let reordered = [alt, parent];
        let (warm, filled) = descriptor.selection(&dir, || {
            anyhow::bail!(
                "reordered rank zero {:?} triggered a new fill",
                reordered[0]
            )
        })?;
        assert!(!filled);
        assert_eq!(warm.incumbent, first.incumbent);
        assert_eq!(warm.algorithm, first.algorithm);
        assert_ne!(warm.algorithm, reordered[0]);
        let mut expanded = warm;
        for id in 3..=12 {
            expanded
                .medians_ms
                .push((LtAlgorithm { id, ..parent }, 100.0));
        }
        assert!(expanded.validate(&descriptor).is_err());
        expanded.descriptor.ffn_tail = Some(0);
        expanded.validate(&expanded.descriptor)?;
        expanded
            .medians_ms
            .push((LtAlgorithm { id: 13, ..parent }, 100.0));
        assert!(expanded.validate(&expanded.descriptor).is_err());
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn dense_cache_entries_differ_by_incumbent() -> Result<()> {
        let parent = LtAlgorithm {
            id: 21,
            tile: 15,
            split_k: 1,
            reduction: 0,
            swizzle: 0,
            custom: 0,
            stages: 12,
            inner: 0,
            cluster: 0,
        };
        let alt = LtAlgorithm { id: 5, ..parent };
        let describe = |incumbent: LtAlgorithm, family: bool| Descriptor {
            key: CacheKey {
                device_uuid: "gpu".into(),
                driver_version: "driver".into(),
                binary_version: "binary".into(),
                cublaslt_version: 1,
            },
            sm: 89,
            shape: LinearShape {
                rows: 2278,
                input: 5376,
                output: 28672,
            },
            alignment: [256; 4],
            workspace: 0,
            ffn_tail: Some(2274),
            incumbent,
            family,
        };
        let entry = |descriptor: Descriptor| {
            let incumbent = descriptor.incumbent;
            Selection {
                descriptor,
                incumbent,
                algorithm: alt,
                required_workspace: 0,
                medians_ms: vec![(incumbent, 100.0), (alt, 97.0)],
                challenger_margin: Some(0.03),
                fill_seconds: 1.0,
            }
        };
        let family = describe(parent, true);
        let other = describe(LtAlgorithm { id: 6, ..parent }, true);
        let ordinary = describe(parent, false);
        let dir =
            std::env::temp_dir().join(format!("ff-dense-lt-incumbent-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        let result = (|| -> Result<()> {
            let (_first, filled) = family.selection(&dir, || Ok(entry(family.clone())))?;
            assert!(filled);
            let (_second, filled) = other.selection(&dir, || Ok(entry(other.clone())))?;
            assert!(filled);
            let (_third, filled) = ordinary.selection(&dir, || Ok(entry(ordinary.clone())))?;
            assert!(filled);
            let files = std::fs::read_dir(&dir)?
                .filter(|entry| {
                    entry.as_ref().is_ok_and(|found| {
                        found.path().extension().is_some_and(|ext| ext == "json")
                    })
                })
                .count();
            assert_eq!(files, 3);
            Ok(())
        })();
        std::fs::remove_dir_all(&dir)?;
        result
    }
}
