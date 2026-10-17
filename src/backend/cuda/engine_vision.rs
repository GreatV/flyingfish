//! Vision weights, patch/position preparation, and image-feature splice for Engine.
use crate::{
    backend::cuda::{
        Device,
        blas::Blas,
        contracts::{Scratch, Tensor, TensorMut},
        cubin,
        linear::{LinearBiasMode, disjoint},
        ops::{CommonOpsRef, GeluMode, grid},
        vision::{
            PatchGrid, PatchPacking, VisionAttnBindings, VisionAttnOps, VisionAttnPlan,
            VisionMergerBindings, VisionMergerOps, VisionMergerPlan, VisionSpec,
        },
        weights::{StoredTensor, WeightCatalog},
    },
    model_spec::ModelSpec,
    vision_input::{MultimodalInput, PositionPlan, SpliceRange, VisionPositions},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::{mem::size_of, sync::Arc};

const HIDDEN: u32 = 1152;
const PATCH: u32 = 1536;
const OUTPUT: u32 = 5120;

fn dense<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
    buffer: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    let mut stride = [1u64; R];
    let mut count = 1u64;
    for i in (0..R).rev() {
        stride[i] = count;
        count = count
            .checked_mul(u64::from(shape[i]))
            .context("vision view extent overflow")?;
    }
    ensure!(
        count == buffer.len() as u64,
        "vision view/allocation size mismatch"
    );
    let (ptr, _read) = buffer.device_ptr(stream);
    unsafe {
        Tensor::new(
            ptr,
            shape,
            stride,
            count
                .checked_mul(size_of::<T>() as u64)
                .context("vision view byte overflow")?,
        )
    }
}

fn matrix<T: Copy>(view: &Tensor<T, 2>, width: u32) -> Result<()> {
    view.validate()?;
    ensure!(
        view.shape[1] == width && view.stride == [u64::from(width), 1],
        "vision IO requires dense rows of width {width}"
    );
    Ok(())
}

fn splice_bounds(
    features: &Tensor<bf16, 2>,
    output: &Tensor<bf16, 2>,
    range: &SpliceRange,
) -> Result<u32> {
    matrix(features, OUTPUT)?;
    matrix(output, OUTPUT)?;
    disjoint(features, output)?;
    ensure!(
        range.feature_rows.start < range.feature_rows.end
            && range.chunk_rows.start < range.chunk_rows.end,
        "empty/reversed image splice"
    );
    ensure!(
        range.feature_rows.len() == range.chunk_rows.len(),
        "image splice row count mismatch"
    );
    ensure!(
        range.feature_rows.end <= features.shape[0] as usize
            && range.chunk_rows.end <= output.shape[0] as usize,
        "image splice outside allocation"
    );
    u32::try_from(range.feature_rows.len()).context("image splice exceeds device range")
}

pub(super) struct VisionIo {
    cast: CudaFunction,
    position_add: CudaFunction,
    splice: CudaFunction,
    pack: CudaFunction,
    residual: CudaFunction,
}

pub(super) struct PositionInputs {
    pub table: Tensor<bf16, 2>,
    pub indices: Tensor<i32, 2>,
    pub weights: Tensor<f32, 2>,
    pub patches: Tensor<bf16, 2>,
}

impl VisionIo {
    fn new(device: &Device) -> Result<Self> {
        let module = cubin::module(&device.ctx, "q38_vision_io")?;
        Ok(Self {
            cast: module.load_function("q38_vision_cast_patches")?,
            position_add: module.load_function("q38_vision_position_add")?,
            splice: module.load_function("q38_vision_splice")?,
            pack: module.load_function("q38_vision_pack")?,
            residual: module.load_function("q38_vision_residual")?,
        })
    }

    fn cast_patches(
        &self,
        stream: &Arc<CudaStream>,
        input: Tensor<f32, 2>,
        output: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        matrix(&input, PATCH)?;
        matrix(&output.0, PATCH)?;
        ensure!(input.shape == output.0.shape, "patch cast shape mismatch");
        disjoint(&input, &output.0)?;
        let count = u64::from(input.shape[0]) * u64::from(PATCH);
        unsafe {
            stream
                .launch_builder(&self.cast)
                .arg(&input.ptr)
                .arg(&output.0.ptr)
                .arg(&count)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    fn position_add(
        &self,
        stream: &Arc<CudaStream>,
        input: PositionInputs,
        output: &mut TensorMut<bf16, 2>,
        interpolated: &mut TensorMut<f32, 2>,
    ) -> Result<()> {
        let PositionInputs {
            table,
            indices,
            weights,
            patches,
        } = input;
        matrix(&table, HIDDEN)?;
        ensure!(
            table.shape[0] == 2304,
            "vision position table must be 48x48"
        );
        matrix(&patches, HIDDEN)?;
        let rows = patches.shape[0];
        matrix(&indices, rows)?;
        matrix(&weights, rows)?;
        ensure!(
            indices.shape[0] == 4 && weights.shape[0] == 4,
            "bilinear tables need four corners"
        );
        matrix(&output.0, HIDDEN)?;
        matrix(&interpolated.0, HIDDEN)?;
        ensure!(
            patches.shape == output.0.shape && patches.shape == interpolated.0.shape,
            "vision position output shape mismatch"
        );
        disjoint(&patches, &output.0)?;
        disjoint(&patches, &interpolated.0)?;
        disjoint(&output.0, &interpolated.0)?;
        disjoint(&table, &output.0)?;
        disjoint(&indices, &output.0)?;
        disjoint(&weights, &output.0)?;
        disjoint(&table, &interpolated.0)?;
        disjoint(&indices, &interpolated.0)?;
        disjoint(&weights, &interpolated.0)?;
        let count = u64::from(rows) * u64::from(HIDDEN);
        unsafe {
            stream
                .launch_builder(&self.position_add)
                .arg(&table.ptr)
                .arg(&indices.ptr)
                .arg(&weights.ptr)
                .arg(&patches.ptr)
                .arg(&output.0.ptr)
                .arg(&interpolated.0.ptr)
                .arg(&rows)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    fn splice(
        &self,
        stream: &Arc<CudaStream>,
        features: Tensor<bf16, 2>,
        output: &mut TensorMut<bf16, 2>,
        range: &SpliceRange,
    ) -> Result<()> {
        let rows = splice_bounds(&features, &output.0, range)?;
        let source = u32::try_from(range.feature_rows.start)?;
        let destination = u32::try_from(range.chunk_rows.start)?;
        let count = u64::from(rows) * u64::from(OUTPUT);
        unsafe {
            stream
                .launch_builder(&self.splice)
                .arg(&features.ptr)
                .arg(&output.0.ptr)
                .arg(&features.shape[0])
                .arg(&output.0.shape[0])
                .arg(&source)
                .arg(&destination)
                .arg(&rows)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }
}

fn take(catalog: &mut WeightCatalog, name: &str, shape: &[usize]) -> Result<CudaSlice<bf16>> {
    let entry = catalog
        .manifest
        .entries
        .get(name)
        .context("vision manifest entry missing")?;
    ensure!(
        entry.load
            && entry.scope == crate::checkpoint::TensorScope::Vision
            && entry.dtype == safetensors::Dtype::BF16
            && entry.shape == shape,
        "invalid vision tensor {name}"
    );
    match catalog.tensors.remove(name) {
        Some(StoredTensor::Bf16(buffer)) => Ok(buffer),
        _ => anyhow::bail!("vision weight not uploaded as BF16: {name}"),
    }
}

struct AffineWeights {
    weight: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
}
impl AffineWeights {
    fn take(catalog: &mut WeightCatalog, name: &str) -> Result<Self> {
        Ok(Self {
            weight: take(catalog, &format!("{name}.weight"), &[HIDDEN as usize])?,
            bias: take(catalog, &format!("{name}.bias"), &[HIDDEN as usize])?,
        })
    }
}

struct BiasedWeight {
    weight: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    shape: [u32; 2],
}
impl BiasedWeight {
    fn take(
        catalog: &mut WeightCatalog,
        name: &str,
        shape: [u32; 2],
        stored_shape: &[usize],
    ) -> Result<Self> {
        Ok(Self {
            weight: take(catalog, &format!("{name}.weight"), stored_shape)?,
            bias: take(catalog, &format!("{name}.bias"), &[shape[0] as usize])?,
            shape,
        })
    }
    fn prepare(&self, stream: &Arc<CudaStream>, blas: &mut Blas, rows: u32) -> Result<()> {
        blas.prepare_biased(
            rows,
            self.shape[0],
            self.shape[1],
            dense(&self.bias, stream, [self.shape[0]])?,
            LinearBiasMode::BeforeOutputRound,
        )
    }
    fn run(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        input: Tensor<bf16, 2>,
        output: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        blas.linear_biased(
            dense(&self.weight, stream, self.shape)?,
            dense(&self.bias, stream, [self.shape[0]])?,
            input,
            output,
            LinearBiasMode::BeforeOutputRound,
        )
    }
}

struct BlockWeights {
    norm1: AffineWeights,
    qkv: BiasedWeight,
    projection: BiasedWeight,
    norm2: AffineWeights,
    fc1: BiasedWeight,
    fc2: BiasedWeight,
}
struct MergerWeights {
    norm: AffineWeights,
    fc1: BiasedWeight,
    fc2: BiasedWeight,
}
struct VisionWeights {
    patch: BiasedWeight,
    position: CudaSlice<bf16>,
    blocks: Vec<BlockWeights>,
    merger: MergerWeights,
}
impl VisionWeights {
    fn take(catalog: &mut WeightCatalog) -> Result<Self> {
        let before = catalog.tensors.len();
        let patch = BiasedWeight::take(
            catalog,
            "model.visual.patch_embed.proj",
            [HIDDEN, PATCH],
            &[1152, 3, 2, 16, 16],
        )?;
        let position = take(catalog, "model.visual.pos_embed.weight", &[2304, 1152])?;
        let mut blocks = Vec::with_capacity(27);
        for i in 0..27 {
            let p = format!("model.visual.blocks.{i}");
            blocks.push(BlockWeights {
                norm1: AffineWeights::take(catalog, &format!("{p}.norm1"))?,
                qkv: BiasedWeight::take(
                    catalog,
                    &format!("{p}.attn.qkv"),
                    [3456, 1152],
                    &[3456, 1152],
                )?,
                projection: BiasedWeight::take(
                    catalog,
                    &format!("{p}.attn.proj"),
                    [1152, 1152],
                    &[1152, 1152],
                )?,
                norm2: AffineWeights::take(catalog, &format!("{p}.norm2"))?,
                fc1: BiasedWeight::take(
                    catalog,
                    &format!("{p}.mlp.linear_fc1"),
                    [4304, 1152],
                    &[4304, 1152],
                )?,
                fc2: BiasedWeight::take(
                    catalog,
                    &format!("{p}.mlp.linear_fc2"),
                    [1152, 4304],
                    &[1152, 4304],
                )?,
            });
        }
        let merger = MergerWeights {
            norm: AffineWeights::take(catalog, "model.visual.merger.norm")?,
            fc1: BiasedWeight::take(
                catalog,
                "model.visual.merger.linear_fc1",
                [4608, 4608],
                &[4608, 4608],
            )?,
            fc2: BiasedWeight::take(
                catalog,
                "model.visual.merger.linear_fc2",
                [5120, 4608],
                &[5120, 4608],
            )?,
        };
        ensure!(
            before.checked_sub(catalog.tensors.len()) == Some(333),
            "incomplete vision weight binding"
        );
        Ok(Self {
            patch,
            position,
            blocks,
            merger,
        })
    }
}

pub(super) struct VisionRuntime {
    model: ModelSpec,
    weights: VisionWeights,
    io: Arc<VisionIo>,
    common: CommonOpsRef,
}

pub(super) struct VisionPrefix {
    positions: PositionPlan,
    pixels: CudaSlice<f32>,
    packed: CudaSlice<bf16>,
    patch_output: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    interpolated: CudaSlice<f32>,
    indices: CudaSlice<i32>,
    coefficients: CudaSlice<f32>,
}

pub(super) struct VisionRequest {
    prefix: VisionPrefix,
    packing: PatchPacking,
    norm: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    core: CudaSlice<bf16>,
    delta: CudaSlice<bf16>,
    mlp: CudaSlice<bf16>,
    features: CudaSlice<bf16>,
    scratch: CudaSlice<u8>,
    attention: VisionAttnPlan,
    merger: VisionMergerPlan,
}

fn tower_bytes(packing: &PatchPacking) -> Result<(usize, usize)> {
    let rows = packing.real_rows() as usize;
    let spec = VisionSpec::q38();
    let attention = VisionAttnOps::workspace(&spec, packing)?;
    let merger = VisionMergerOps::workspace(&spec, packing.real_rows())?;
    ensure!(
        attention.alignment == 256 && merger.alignment == 256,
        "vision scratch alignment mismatch"
    );
    let scratch = usize::try_from(attention.bytes.max(merger.bytes))?;
    let payload = [
        prefix_bytes(rows)?,
        rows.checked_mul((5 * HIDDEN as usize + 3456 + 4304) * 2)
            .context("tower work overflow")?,
        (packing.padded_rows() as usize)
            .checked_mul(HIDDEN as usize * 2)
            .context("padded V overflow")?,
        (rows / 4)
            .checked_mul(OUTPUT as usize * 2)
            .context("merger output overflow")?,
        packing
            .images()
            .checked_mul(16)
            .and_then(|n| n.checked_add(8))
            .context("packing tables overflow")?,
        scratch,
    ]
    .into_iter()
    .try_fold(0usize, |sum, n| {
        sum.checked_add(n).context("vision allocation overflow")
    })?;
    Ok((payload, scratch))
}

fn record(
    trace: &mut Option<&mut crate::trace::Trace>,
    stream: &Arc<CudaStream>,
    name: &str,
    buffer: &CudaSlice<bf16>,
    width: u32,
) -> Result<()> {
    if let Some(t) = trace.as_deref_mut() {
        t.add(
            name.into(),
            vec![buffer.len() / width as usize, width as usize],
            stream
                .clone_dtoh(buffer)?
                .into_iter()
                .map(bf16::to_f32)
                .collect(),
        )?;
    }
    Ok(())
}

fn prefix_bytes(rows: usize) -> Result<usize> {
    ensure!(rows > 0, "empty vision prefix");
    rows.checked_mul(PATCH as usize * 6 + HIDDEN as usize * 8 + 4 * 8)
        .context("vision prefix byte overflow")
}

impl VisionRuntime {
    pub(super) fn request_bytes(
        model: &ModelSpec,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<usize> {
        input.validate(model, capacity)?;
        let packing = PatchPacking::build(
            &input
                .grids
                .iter()
                .map(|g| PatchGrid { h: g.h, w: g.w })
                .collect::<Vec<_>>(),
        )?;
        Ok(tower_bytes(&packing)?.0)
    }
    pub(super) fn prepare(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<VisionRequest> {
        let positions = input.validate(&self.model, capacity)?;
        let packing = PatchPacking::build(
            &input
                .grids
                .iter()
                .map(|g| PatchGrid { h: g.h, w: g.w })
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            packing.real_rows() as usize == positions.patch_rows(),
            "patch packing/input mismatch"
        );
        let (bytes, scratch_bytes) = tower_bytes(&packing)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16).is_some_and(|n| bytes <= n),
            "vision tower budget exceeded: required={bytes} free={free}"
        );
        let rows = packing.real_rows();
        let padded = packing.padded_rows();
        let s = &device.stream;
        let prefix = self.prepare_prefix(device, blas, input, capacity)?;
        let norm = s.alloc_zeros(rows as usize * HIDDEN as usize)?;
        let qkv = s.alloc_zeros(rows as usize * 3456)?;
        let q = s.alloc_zeros(rows as usize * HIDDEN as usize)?;
        let k = s.alloc_zeros(rows as usize * HIDDEN as usize)?;
        let v = s.alloc_zeros(padded as usize * HIDDEN as usize)?;
        let core = s.alloc_zeros(rows as usize * HIDDEN as usize)?;
        let delta = s.alloc_zeros(rows as usize * HIDDEN as usize)?;
        let mlp = s.alloc_zeros(rows as usize * 4304)?;
        let features = s.alloc_zeros(rows as usize / 4 * OUTPUT as usize)?;
        let scratch = s.alloc_zeros(scratch_bytes)?;
        for block in &self.weights.blocks {
            for weight in [&block.qkv, &block.projection, &block.fc1, &block.fc2] {
                weight.prepare(s, blas, rows)?;
            }
        }
        let spec = VisionSpec::q38();
        let attention = VisionAttnOps::prepare(
            device,
            &spec,
            &packing,
            VisionAttnBindings {
                q_in: dense(&q, s, [16, rows, 72])?,
                k_in: dense(&k, s, [16, rows, 72])?,
                v: dense(&v, s, [16, padded, 72])?,
                o: TensorMut(dense(&core, s, [rows, 16, 72])?),
            },
        )?;
        let m = &self.weights.merger;
        let merger = VisionMergerOps::prepare(
            &spec,
            VisionMergerBindings {
                x: dense(&prefix.hidden, s, [rows, HIDDEN])?,
                norm_w: dense(&m.norm.weight, s, [HIDDEN])?,
                norm_b: dense(&m.norm.bias, s, [HIDDEN])?,
                fc1_w: dense(&m.fc1.weight, s, m.fc1.shape)?,
                fc1_b: dense(&m.fc1.bias, s, [4608])?,
                fc2_w: dense(&m.fc2.weight, s, m.fc2.shape)?,
                fc2_b: dense(&m.fc2.bias, s, [OUTPUT])?,
                out: TensorMut(dense(&features, s, [rows / 4, OUTPUT])?),
            },
            self.common.clone(),
            blas,
        )?;
        s.synchronize()?;
        eprintln!(
            "{}",
            serde_json::json!({"vision_memory_plan": {
                "rows":rows,"padded_rows":padded,"images":packing.images(),
                "payload_bytes":bytes,"operator_scratch_bytes":scratch_bytes,"free_bytes":free,"reserve_bytes":total/16
            }})
        );
        Ok(VisionRequest {
            prefix,
            packing,
            norm,
            qkv,
            q,
            k,
            v,
            core,
            delta,
            mlp,
            features,
            scratch,
            attention,
            merger,
        })
    }

    pub(super) fn encode(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        mut request: VisionRequest,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<EncodedImages> {
        let launched = (|| -> Result<()> {
            self.run_prefix(stream, blas, &mut request.prefix)?;
            let r = &request;
            let rows = r.packing.real_rows();
            let padded = r.packing.padded_rows();
            for (name, b, width) in [
                ("pixel_values_bf16", &r.prefix.packed, PATCH),
                ("after_patch_embed", &r.prefix.patch_output, HIDDEN),
                ("after_pos_add", &r.prefix.hidden, HIDDEN),
            ] {
                record(&mut trace, stream, name, b, width)?;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.add(
                    "pos_embeds".into(),
                    vec![rows as usize, HIDDEN as usize],
                    stream.clone_dtoh(&r.prefix.interpolated)?,
                )?;
            }
            let hidden = dense(&r.prefix.hidden, stream, [rows, 1, HIDDEN])?;
            let norm = dense(&r.norm, stream, [rows, 1, HIDDEN])?;
            let (scratch_ptr, _hold) = r.scratch.device_ptr(stream);
            let scratch = || Scratch {
                ptr: scratch_ptr,
                bytes: r.scratch.len() as u64,
            };
            for (i, block) in self.weights.blocks.iter().enumerate() {
                if i == 26 {
                    record(&mut trace, stream, "b25_out", &r.prefix.hidden, HIDDEN)?;
                }
                self.common.norm.layer_norm(
                    stream,
                    hidden,
                    dense(&block.norm1.weight, stream, [HIDDEN])?,
                    dense(&block.norm1.bias, stream, [HIDDEN])?,
                    &mut TensorMut(norm),
                    1e-6,
                )?;
                if i == 0 || i == 26 {
                    record(&mut trace, stream, &format!("b{i}.norm1"), &r.norm, HIDDEN)?;
                }
                block.qkv.run(
                    stream,
                    blas,
                    dense(&r.norm, stream, [rows, HIDDEN])?,
                    &mut TensorMut(dense(&r.qkv, stream, [rows, 3456])?),
                )?;
                let input = dense(&r.qkv, stream, [rows, 3456])?;
                let q = dense(&r.q, stream, [16, rows, 72])?;
                let k = dense(&r.k, stream, [16, rows, 72])?;
                let v = dense(&r.v, stream, [16, padded, 72])?;
                for image in 0..r.packing.images() {
                    let first = r.packing.cu_seqlens()[image] as u32;
                    let n = r.packing.cu_seqlens()[image + 1] as u32 - first;
                    let key_first = r.packing.k_bases()[image] as u32;
                    unsafe {
                        stream
                            .launch_builder(&self.io.pack)
                            .arg(&input.ptr)
                            .arg(&q.ptr)
                            .arg(&k.ptr)
                            .arg(&v.ptr)
                            .arg(&rows)
                            .arg(&padded)
                            .arg(&first)
                            .arg(&key_first)
                            .arg(&n)
                            .launch(grid(
                                (u64::from(n) * 1152).div_ceil(256).min(65535) as usize,
                                256,
                            ))?;
                    }
                }
                VisionAttnOps::run(&r.attention, stream, scratch())?;
                block.projection.run(
                    stream,
                    blas,
                    dense(&r.core, stream, [rows, HIDDEN])?,
                    &mut TensorMut(dense(&r.delta, stream, [rows, HIDDEN])?),
                )?;
                self.add_residual(stream, &r.prefix.hidden, &r.delta)?;
                self.common.norm.layer_norm(
                    stream,
                    hidden,
                    dense(&block.norm2.weight, stream, [HIDDEN])?,
                    dense(&block.norm2.bias, stream, [HIDDEN])?,
                    &mut TensorMut(norm),
                    1e-6,
                )?;
                if i == 0 || i == 26 {
                    record(&mut trace, stream, &format!("b{i}.norm2"), &r.norm, HIDDEN)?;
                }
                block.fc1.run(
                    stream,
                    blas,
                    dense(&r.norm, stream, [rows, HIDDEN])?,
                    &mut TensorMut(dense(&r.mlp, stream, [rows, 4304])?),
                )?;
                let mlp = dense(&r.mlp, stream, [rows, 1, 4304])?;
                self.common
                    .gate
                    .gelu(stream, mlp, &mut TensorMut(mlp), GeluMode::Tanh)?;
                block.fc2.run(
                    stream,
                    blas,
                    dense(&r.mlp, stream, [rows, 4304])?,
                    &mut TensorMut(dense(&r.delta, stream, [rows, HIDDEN])?),
                )?;
                self.add_residual(stream, &r.prefix.hidden, &r.delta)?;
                if i == 0 || i == 26 {
                    record(
                        &mut trace,
                        stream,
                        &format!("block{i}_out"),
                        &r.prefix.hidden,
                        HIDDEN,
                    )?;
                }
            }
            VisionMergerOps::run(&r.merger, stream, blas, scratch())?;
            record(&mut trace, stream, "merger_out", &r.features, OUTPUT)?;
            Ok(())
        })();
        // All request allocations outlive asynchronous work, including the error path.
        let synchronized = stream.synchronize();
        launched?;
        synchronized?;
        EncodedImages::new(request.prefix.positions, request.features, self.io.clone())
    }

    fn add_residual(
        &self,
        stream: &Arc<CudaStream>,
        hidden: &CudaSlice<bf16>,
        delta: &CudaSlice<bf16>,
    ) -> Result<()> {
        ensure!(
            hidden.len() == delta.len(),
            "vision residual shape mismatch"
        );
        let (h, _h) = hidden.device_ptr(stream);
        let (d, _d) = delta.device_ptr(stream);
        let count = hidden.len() as u64;
        unsafe {
            stream
                .launch_builder(&self.io.residual)
                .arg(&h)
                .arg(&d)
                .arg(&count)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    pub(super) fn from_catalog(
        model: &ModelSpec,
        catalog: &mut WeightCatalog,
        device: &Device,
        common: CommonOpsRef,
    ) -> Result<Self> {
        model
            .vision
            .as_ref()
            .context("vision configuration missing")?
            .validate()?;
        let io = Arc::new(VisionIo::new(device)?);
        let weights = VisionWeights::take(catalog)?;
        Ok(Self {
            model: model.clone(),
            weights,
            io,
            common,
        })
    }

    pub(super) fn prepare_prefix(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<VisionPrefix> {
        let positions = input.validate(&self.model, capacity)?;
        let rows = positions.patch_rows();
        let bytes = prefix_bytes(rows)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|available| bytes <= available),
            "vision prefix does not fit memory budget: required={bytes} free={free}"
        );
        let row_count = u32::try_from(rows)?;
        let tables = VisionPositions::new(
            self.model
                .vision
                .as_ref()
                .context("vision configuration missing")?,
            input.grids,
            rows,
        )?;
        let s = &device.stream;
        let indices: Vec<_> = tables.bilinear_indices.into_iter().flatten().collect();
        let coefficients: Vec<_> = tables.bilinear_weights.into_iter().flatten().collect();
        self.weights.patch.prepare(s, blas, row_count)?;
        let prefix = VisionPrefix {
            positions,
            pixels: s.clone_htod(input.pixel_values)?,
            packed: s.alloc_zeros(rows * PATCH as usize)?,
            patch_output: s.alloc_zeros(rows * HIDDEN as usize)?,
            hidden: s.alloc_zeros(rows * HIDDEN as usize)?,
            interpolated: s.alloc_zeros(rows * HIDDEN as usize)?,
            indices: s.clone_htod(&indices)?,
            coefficients: s.clone_htod(&coefficients)?,
        };
        s.synchronize()?;
        Ok(prefix)
    }

    /// Computes only patch embedding and learned-position addition, not a full tower.
    pub(super) fn run_prefix(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        request: &mut VisionPrefix,
    ) -> Result<()> {
        let rows = u32::try_from(request.positions.patch_rows())?;
        let packed = dense(&request.packed, stream, [rows, PATCH])?;
        let patch_output = dense(&request.patch_output, stream, [rows, HIDDEN])?;
        let launched = (|| -> Result<()> {
            self.io.cast_patches(
                stream,
                dense(&request.pixels, stream, [rows, PATCH])?,
                &mut TensorMut(packed),
            )?;
            self.weights
                .patch
                .run(stream, blas, packed, &mut TensorMut(patch_output))?;
            self.io.position_add(
                stream,
                PositionInputs {
                    table: dense(&self.weights.position, stream, [2304, HIDDEN])?,
                    indices: dense(&request.indices, stream, [4, rows])?,
                    weights: dense(&request.coefficients, stream, [4, rows])?,
                    patches: patch_output,
                },
                &mut TensorMut(dense(&request.hidden, stream, [rows, HIDDEN])?),
                &mut TensorMut(dense(&request.interpolated, stream, [rows, HIDDEN])?),
            )?;
            Ok(())
        })();
        let synchronized = stream.synchronize();
        launched?;
        synchronized?;
        Ok(())
    }
}

/// Owns merger output until all prompt chunks have consumed their image rows.
pub(super) struct EncodedImages {
    positions: PositionPlan,
    features: CudaSlice<bf16>,
    io: Arc<VisionIo>,
}
impl EncodedImages {
    /// The caller must supply the completed merger output of this request.
    pub(super) fn new(
        positions: PositionPlan,
        features: CudaSlice<bf16>,
        io: Arc<VisionIo>,
    ) -> Result<Self> {
        let elements = positions
            .feature_rows()
            .checked_mul(OUTPUT as usize)
            .context("image feature size overflow")?;
        ensure!(
            elements > 0 && features.len() == elements,
            "merger feature/slot count mismatch"
        );
        Ok(Self {
            positions,
            features,
            io,
        })
    }
    pub(super) fn splice(
        &self,
        stream: &Arc<CudaStream>,
        first: usize,
        output: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        matrix(&output.0, OUTPUT)?;
        let end = first
            .checked_add(output.0.shape[0] as usize)
            .context("splice prompt extent overflow")?
            .min(self.positions.prompt_rows());
        if first >= end {
            return Ok(());
        }
        let features = dense(
            &self.features,
            stream,
            [u32::try_from(self.positions.feature_rows())?, OUTPUT],
        )?;
        let splices = self.positions.splices(first..end)?;
        for range in &splices {
            self.io.splice(stream, features, output, range)?;
        }
        if !splices.is_empty()
            && std::env::var("FLYINGFISH_SERVE_TRACE_IDS").is_ok_and(|value| value == "1")
        {
            eprintln!(
                "{}",
                serde_json::json!({"image_splice_hook": {
                    "images": self.positions.slots().len(), "splice_calls": splices.len(),
                    "feature_rows": self.positions.feature_rows(), "first": first, "end": end,
                }})
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tower_budget_includes_image_padding_and_merger_output() -> Result<()> {
        let packing = PatchPacking::build(&[PatchGrid { h: 2, w: 2 }, PatchGrid { h: 2, w: 4 }])?;
        assert_eq!((packing.real_rows(), packing.padded_rows()), (12, 128));
        let (bytes, scratch) = tower_bytes(&packing)?;
        let expected = 12 * 18464
            + 12 * (1152 * 5 + 3456 + 4304) * 2
            + 128 * 1152 * 2
            + 3 * 5120 * 2
            + 40
            + scratch;
        assert_eq!(bytes, expected);
        assert!(scratch >= VisionMergerOps::workspace(&VisionSpec::q38(), 12)?.bytes as usize);
        Ok(())
    }

    #[test]
    #[ignore = "requires local vision GPU window; pack and graph position checks"]
    fn vision_pack_positions_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let io = VisionIo::new(&device)?;
        let packing = PatchPacking::build(&[PatchGrid { h: 2, w: 2 }, PatchGrid { h: 2, w: 4 }])?;
        let rows = packing.real_rows();
        let padded = packing.padded_rows();
        let host: Vec<_> = (0..rows as usize * 3456)
            .map(|i| bf16::from_bits((0x2000 + i % 0x4000) as u16))
            .collect();
        let raw = s.clone_htod(&host)?;
        let q = s.alloc_zeros::<bf16>(rows as usize * 1152)?;
        let k = s.alloc_zeros::<bf16>(rows as usize * 1152)?;
        let sentinel = bf16::NEG_INFINITY;
        let v = s.clone_htod(&vec![sentinel; padded as usize * 1152])?;
        let input = dense(&raw, s, [rows, 3456])?;
        let qv = dense(&q, s, [16, rows, 72])?;
        let kv = dense(&k, s, [16, rows, 72])?;
        let vv = dense(&v, s, [16, padded, 72])?;
        for image in 0..packing.images() {
            let first = packing.cu_seqlens()[image] as u32;
            let n = packing.cu_seqlens()[image + 1] as u32 - first;
            let key_first = packing.k_bases()[image] as u32;
            unsafe {
                s.launch_builder(&io.pack)
                    .arg(&input.ptr)
                    .arg(&qv.ptr)
                    .arg(&kv.ptr)
                    .arg(&vv.ptr)
                    .arg(&rows)
                    .arg(&padded)
                    .arg(&first)
                    .arg(&key_first)
                    .arg(&n)
                    .launch(grid((n as usize * 1152).div_ceil(256), 256))?;
            }
        }
        let q = s.clone_dtoh(&q)?;
        let k = s.clone_dtoh(&k)?;
        let v = s.clone_dtoh(&v)?;
        let mut expected_v = vec![sentinel; v.len()];
        for image in 0..packing.images() {
            let first = packing.cu_seqlens()[image] as usize;
            let end = packing.cu_seqlens()[image + 1] as usize;
            for row in first..end {
                for h in 0..16usize {
                    for d in 0..72 {
                        let src = row * 3456 + h * 72 + d;
                        let dst = (h * rows as usize + row) * 72 + d;
                        assert_eq!(q[dst], host[src]);
                        assert_eq!(k[dst], host[src + 1152]);
                        let vr = packing.k_bases()[image] as usize + row - first;
                        expected_v[(h * padded as usize + vr) * 72 + d] = host[src + 2304];
                    }
                }
            }
        }
        assert_eq!(v, expected_v, "V packing changed an image or its padding");

        let gather =
            cubin::module(&device.ctx, "q38_vision_io")?.load_function("q38_text_positions")?;
        let image_axes: Vec<i32> = (0..12).flat_map(|p| [p, p / 2, p / 3]).collect();
        let mut table = s.clone_htod(&image_axes)?;
        let active = s.clone_htod(&[-1i32; 15])?;
        let mut meta = s.clone_htod(&[2i32, 3, 5, 12, 1, 0, 0, 0])?;
        let tp = {
            let (ptr, _guard) = table.device_ptr(s);
            ptr
        };
        let ap = {
            let (ptr, _guard) = active.device_ptr(s);
            ptr
        };
        let mp = {
            let (ptr, _guard) = meta.device_ptr(s);
            ptr
        };
        let launch = || -> Result<()> {
            unsafe {
                s.launch_builder(&gather)
                    .arg(&tp)
                    .arg(&ap)
                    .arg(&mp)
                    .arg(&5u32)
                    .launch(grid(1, 256))?;
            }
            Ok(())
        };
        launch()?;
        let got = s.clone_dtoh(&active)?;
        assert_eq!(&got[..9], &image_axes[6..15]);
        assert_eq!(&got[9..], &[-1; 6]);
        s.memcpy_htod(&[5i32, 1, 6, 12, 1, 0, 0, 0], &mut meta)?;
        s.synchronize()?;
        s.begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )?;
        launch()?;
        let graph = s
            .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
            .context("empty position graph")?;
        graph.launch()?;
        assert_eq!(&s.clone_dtoh(&active)?[..3], &image_axes[15..18]);
        let text: Vec<i32> = (0..12).flat_map(|p| [p; 3]).collect();
        s.memcpy_htod(&text, &mut table)?;
        s.memcpy_htod(&[6i32, 1, 7, 12, 1, 0, 0, 0], &mut meta)?;
        graph.launch()?;
        assert_eq!(&s.clone_dtoh(&active)?[..3], &[6; 3]);
        eprintln!(
            "VISION_PACK_AND_POSITIONS two images/padding/nonzero prefix/tail/graph/reset PASS"
        );
        Ok(())
    }
    #[test]
    fn prefix_budget_is_checked_and_counts_each_live_buffer() {
        assert_eq!(prefix_bytes(256).unwrap(), 256 * 18464);
        assert!(prefix_bytes(0).is_err());
        assert!(prefix_bytes(usize::MAX).is_err());
    }
    #[test]
    fn splice_guards_reject_mismatched_ranges_and_aliases() -> Result<()> {
        let features = unsafe {
            Tensor::new(
                0x10000,
                [4, OUTPUT],
                [u64::from(OUTPUT), 1],
                4 * u64::from(OUTPUT) * 2,
            )?
        };
        let output = unsafe {
            Tensor::new(
                0x100000,
                [6, OUTPUT],
                [u64::from(OUTPUT), 1],
                6 * u64::from(OUTPUT) * 2,
            )?
        };
        let good = SpliceRange {
            chunk_rows: 2..5,
            feature_rows: 1..4,
        };
        assert_eq!(splice_bounds(&features, &output, &good)?, 3);
        assert!(
            splice_bounds(
                &features,
                &output,
                &SpliceRange {
                    chunk_rows: 2..5,
                    feature_rows: 2..5
                }
            )
            .is_err()
        );
        assert!(
            splice_bounds(
                &features,
                &output,
                &SpliceRange {
                    chunk_rows: 2..6,
                    feature_rows: 1..4
                }
            )
            .is_err()
        );
        assert!(splice_bounds(&features, &features, &good).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires explicit local vision GPU window: FF_Q38_VISION_WINDOW=granted"]
    fn vision_io_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let io = Arc::new(VisionIo::new(&device)?);
        let model = ModelSpec::qwen(
            &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
            &serde_json::Value::Null,
        )?;
        let image_grids = [crate::vision_input::ImageGrid { t: 1, h: 4, w: 6 }];
        let rows = 24u32;
        let pattern = [
            -1.0f32,
            -0.25,
            -0.0,
            0.0,
            0.1,
            f32::from_bits(0x3f80_8000),
            1.0039064,
            f32::from_bits(0xbf80_8000),
        ];
        let pixels: Vec<_> = (0..rows as usize * PATCH as usize)
            .map(|i| pattern[i % pattern.len()])
            .collect();
        let input = s.clone_htod(&pixels)?;
        let packed = s.clone_htod(&vec![bf16::NAN; pixels.len()])?;
        io.cast_patches(
            s,
            dense(&input, s, [rows, PATCH])?,
            &mut TensorMut(dense(&packed, s, [rows, PATCH])?),
        )?;
        ensure!(
            s.clone_dtoh(&packed)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                == pixels
                    .iter()
                    .copied()
                    .map(|v| bf16::from_f32(v).to_bits())
                    .collect::<Vec<_>>(),
            "patch cast mismatch"
        );

        let tables =
            VisionPositions::new(model.vision.as_ref().unwrap(), &image_grids, rows as usize)?;
        let indices: Vec<_> = tables.bilinear_indices.iter().flatten().copied().collect();
        let coefficients: Vec<_> = tables.bilinear_weights.iter().flatten().copied().collect();
        let position_host: Vec<_> = (0..2304 * HIDDEN as usize)
            .map(|i| bf16::from_f32((i % 17) as f32 / 32.0 - 0.25))
            .collect();
        let patch_host: Vec<_> = (0..rows as usize * HIDDEN as usize)
            .map(|i| bf16::from_f32((i % 7) as f32 / 16.0 - 0.125))
            .collect();
        let position = s.clone_htod(&position_host)?;
        let patch = s.clone_htod(&patch_host)?;
        let index = s.clone_htod(&indices)?;
        let coefficient = s.clone_htod(&coefficients)?;
        let hidden = s.clone_htod(&vec![bf16::NAN; patch_host.len()])?;
        let interpolation = s.clone_htod(&vec![f32::NAN; patch_host.len()])?;
        io.position_add(
            s,
            PositionInputs {
                table: dense(&position, s, [2304, HIDDEN])?,
                indices: dense(&index, s, [4, rows])?,
                weights: dense(&coefficient, s, [4, rows])?,
                patches: dense(&patch, s, [rows, HIDDEN])?,
            },
            &mut TensorMut(dense(&hidden, s, [rows, HIDDEN])?),
            &mut TensorMut(dense(&interpolation, s, [rows, HIDDEN])?),
        )?;
        let got = s.clone_dtoh(&interpolation)?;
        let got_hidden = s.clone_dtoh(&hidden)?;
        for r in 0..rows as usize {
            for d in 0..HIDDEN as usize {
                let mut sum = 0f32;
                for k in 0..4 {
                    sum += position_host
                        [indices[k * rows as usize + r] as usize * HIDDEN as usize + d]
                        .to_f32()
                        * coefficients[k * rows as usize + r];
                }
                let i = r * HIDDEN as usize + d;
                ensure!(
                    got[i].to_bits() == sum.to_bits(),
                    "bilinear FP32 mismatch at {r},{d}"
                );
                let gold = bf16::from_f32(patch_host[i].to_f32() + bf16::from_f32(sum).to_f32());
                ensure!(
                    got_hidden[i].to_bits() == gold.to_bits(),
                    "position cast/add mismatch at {r},{d}"
                );
            }
        }

        let mut ids = vec![100, 248053];
        ids.extend([248056; 6]);
        ids.extend([248054, 200]);
        let mut modalities = vec![0; 10];
        modalities[2..8].fill(1);
        let plan = MultimodalInput {
            input_ids: &ids,
            modality_ids: &modalities,
            pixel_values: &pixels,
            grids: &image_grids,
        }
        .validate(&model, 32)?;
        let feature_host: Vec<_> = (0..6 * OUTPUT as usize)
            .map(|i| bf16::from_f32((i / OUTPUT as usize + 1) as f32 * 0.25))
            .collect();
        let encoded = EncodedImages::new(plan, s.clone_htod(&feature_host)?, io.clone())?;
        let sentinel = bf16::from_f32(-7.0);
        for first in [0usize, 5, 10] {
            let output = s.clone_htod(&vec![sentinel; 5 * OUTPUT as usize])?;
            encoded.splice(s, first, &mut TensorMut(dense(&output, s, [5, OUTPUT])?))?;
            let got = s.clone_dtoh(&output)?;
            for r in 0..5 {
                let absolute = first + r;
                let expected = if (2..8).contains(&absolute) {
                    bf16::from_f32((absolute - 1) as f32 * 0.25)
                } else {
                    sentinel
                };
                ensure!(
                    got[r * OUTPUT as usize..(r + 1) * OUTPUT as usize]
                        .iter()
                        .all(|v| *v == expected),
                    "splice corrupted image/text row {absolute}"
                );
            }
        }
        eprintln!(
            "VISION_IO cast/bilinear_fp32/cast_before_add/chunk_splice/text_preservation PASS"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_Q38_VISION_WINDOW=granted, FF_Q38_MODEL, FF_Q38_MERGER_INPUT_DIR and FF_Q38_VISION_OUTPUT_DIR"]
    fn vision_merger_stages_gpu() -> Result<()> {
        use crate::backend::cuda::{closure::Closure, ops::Ops};
        use std::{path::PathBuf, rc::Rc};
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let path = |name| -> Result<PathBuf> {
            Ok(PathBuf::from(
                std::env::var_os(name).with_context(|| format!("{name} is required"))?,
            ))
        };
        let model_dir = path("FF_Q38_MODEL")?;
        let input_dir = path("FF_Q38_MERGER_INPUT_DIR")?;
        let output = path("FF_Q38_VISION_OUTPUT_DIR")?;
        std::fs::create_dir_all(&output)?;
        let model = ModelSpec::read(&model_dir)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &output,
            Rc::new(Closure::new(false, "merger stages".into())),
        )?;
        let mut catalog = WeightCatalog::load_with_policy(
            &model_dir,
            &model,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        let weights = VisionWeights::take(&mut catalog)?;
        let m = &weights.merger;
        for case in ["img-report", "img-nonsquare", "img-marker", "img-tiny"] {
            let source =
                crate::trace::Trace::read(&input_dir.join(format!("{case}-tower.safetensors")))?;
            let host: Vec<_> = source
                .values("block26_out")?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            let rows = u32::try_from(host.len() / HIDDEN as usize)?;
            let tokens = rows / 4;
            let x = s.clone_htod(&host)?;
            let out = s.alloc_zeros::<bf16>((tokens * OUTPUT) as usize)?;
            let spec = VisionSpec::q38();
            let req = VisionMergerOps::workspace(&spec, rows)?;
            let scratch = s.alloc_zeros::<u8>(req.bytes as usize)?;
            let plan = VisionMergerOps::prepare(
                &spec,
                VisionMergerBindings {
                    x: dense(&x, s, [rows, HIDDEN])?,
                    norm_w: dense(&m.norm.weight, s, [HIDDEN])?,
                    norm_b: dense(&m.norm.bias, s, [HIDDEN])?,
                    fc1_w: dense(&m.fc1.weight, s, m.fc1.shape)?,
                    fc1_b: dense(&m.fc1.bias, s, [4608])?,
                    fc2_w: dense(&m.fc2.weight, s, m.fc2.shape)?,
                    fc2_b: dense(&m.fc2.bias, s, [OUTPUT])?,
                    out: TensorMut(dense(&out, s, [tokens, OUTPUT])?),
                },
                common.clone(),
                &mut blas,
            )?;
            let (ptr, _hold) = scratch.device_ptr(s);
            VisionMergerOps::run(
                &plan,
                s,
                &blas,
                Scratch {
                    ptr,
                    bytes: req.bytes,
                },
            )?;
            let actual = s.clone_dtoh(&out)?;
            let expected = source.values("merger_out")?;
            ensure!(
                actual.len() == expected.len()
                    && actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_f32().to_bits() == b.to_bits()),
                "isolated merger differs from tower: {case}"
            );
            let raw = s.clone_dtoh(&scratch)?;
            // The merger workspace contains two equally sized row buffers.
            let bytes = host.len() * size_of::<bf16>();
            ensure!(raw.len() == 2 * bytes, "merger workspace layout changed");
            let decode = |data: &[u8]| -> Vec<bf16> {
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                    .collect()
            };
            let ln = decode(&raw[..bytes]);
            let gelu = decode(&raw[bytes..]);
            // GELU overwrites fc1 in place; the replayed fc1 must equal the production scratch bits.
            let ln_device = s.clone_htod(&ln)?;
            let fc1_device = s.alloc_zeros::<bf16>(host.len())?;
            m.fc1.run(
                s,
                &blas,
                dense(&ln_device, s, [tokens, 4608])?,
                &mut TensorMut(dense(&fc1_device, s, [tokens, 4608])?),
            )?;
            let fc1 = s.clone_dtoh(&fc1_device)?;
            let activation = dense(&fc1_device, s, [tokens, 1, 4608])?;
            common
                .gate
                .gelu(s, activation, &mut TensorMut(activation), GeluMode::Exact)?;
            ensure!(
                s.clone_dtoh(&fc1_device)?
                    .iter()
                    .zip(&gelu)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "fc1 replay differs from production GELU: {case}"
            );
            let mut trace = crate::trace::Trace::default();
            trace.set_device(&device.info.name)?;
            for (name, width, values) in [
                ("merger_input", HIDDEN, host),
                ("merger_ln", HIDDEN, ln),
                ("merger_fc1", 4608, fc1),
                ("merger_gelu", 4608, gelu),
                ("merger_out", OUTPUT, actual),
            ] {
                trace.add(
                    name.into(),
                    vec![values.len() / width as usize, width as usize],
                    values.into_iter().map(bf16::to_f32).collect(),
                )?;
            }
            trace.save(&output.join(format!("{case}-merger.safetensors")))?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_Q38_VISION_WINDOW=granted and FF_Q38_VISION_OUTPUT_DIR"]
    fn vision_merger_bias_rounding_gpu() -> Result<()> {
        use crate::backend::cuda::{closure::Closure, ops::Ops};
        use std::{path::PathBuf, rc::Rc};
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let output = PathBuf::from(
            std::env::var_os("FF_Q38_VISION_OUTPUT_DIR")
                .context("vision output directory missing")?,
        );
        std::fs::create_dir_all(&output)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &output,
            Rc::new(Closure::new(false, "merger bias rounding".into())),
        )?;
        let input = 4608u32;
        let half_ulp = bf16::from_f32(1.0 / 256.0);
        // The dot product is a BF16 midpoint. Adding bias before conversion
        // gives 1 + 1/128; rounding the dot product first instead gives 1.
        let expected = bf16::from_bits(0x3f81);
        for width in [4608u32, 5120] {
            let mut weight = vec![bf16::ZERO; (width * input) as usize];
            for row in weight.chunks_exact_mut(input as usize) {
                row[0] = bf16::ONE;
                row[input as usize / 2] = half_ulp;
            }
            let weight = s.clone_htod(&weight)?;
            let bias = s.clone_htod(&vec![half_ulp; width as usize])?;
            for rows in [64u32, 1024] {
                let x = s.clone_htod(&vec![bf16::ONE; (rows * input) as usize])?;
                let y = s.clone_htod(&vec![bf16::NAN; (rows * width) as usize])?;
                let bias = dense(&bias, s, [width])?;
                blas.prepare_biased(rows, width, input, bias, LinearBiasMode::BeforeOutputRound)?;
                blas.linear_biased(
                    dense(&weight, s, [width, input])?,
                    bias,
                    dense(&x, s, [rows, input])?,
                    &mut TensorMut(dense(&y, s, [rows, width])?),
                    LinearBiasMode::BeforeOutputRound,
                )?;
                let got = s.clone_dtoh(&y)?;
                let wrong = got
                    .iter()
                    .filter(|v| v.to_bits() != expected.to_bits())
                    .count();
                ensure!(
                    wrong == 0,
                    "bias rounded after GEMM: M={rows} N={width}, {wrong} wrong elements"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local vision GPU window and model/fixture/output paths; full tower diagnostics"]
    fn vision_tower_gpu() -> Result<()> {
        use crate::backend::cuda::{closure::Closure, ops::Ops};
        use safetensors::{Dtype, SafeTensors};
        use std::{path::PathBuf, rc::Rc};
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let model_dir =
            PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("model directory missing")?);
        let fixtures = PathBuf::from(
            std::env::var_os("FF_Q38_FIXTURE_DIR").context("fixture directory missing")?,
        );
        let output = PathBuf::from(
            std::env::var_os("FF_Q38_VISION_OUTPUT_DIR")
                .context("vision output directory missing")?,
        );
        std::fs::create_dir_all(&output)?;
        let model = ModelSpec::read(&model_dir)?;
        let device = Device::new(0)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &output,
            Rc::new(Closure::new(false, "vision prefix diagnostic".into())),
        )?;
        let mut catalog = WeightCatalog::load_with_policy(
            &model_dir,
            &model,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        ensure!(
            catalog.bytes == 921_460_192,
            "vision-only weight ledger mismatch"
        );
        let runtime = VisionRuntime::from_catalog(&model, &mut catalog, &device, common)?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        for case in ["img-report", "img-nonsquare", "img-marker", "img-tiny"] {
            let file = std::fs::File::open(fixtures.join(format!("{case}.safetensors")))?;
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let source = SafeTensors::deserialize(&map)?;
            let integers = |key: &str| -> Result<Vec<i64>> {
                let t = source.tensor(key)?;
                ensure!(t.dtype() == Dtype::I64, "fixture integer dtype mismatch");
                Ok(t.data()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| i64::from_le_bytes(*v))
                    .collect())
            };
            let ids = integers("input_ids")?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let modalities = integers("mm_token_type_ids")?
                .into_iter()
                .map(u8::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            let grid = integers("image_grid_thw")?;
            ensure!(grid.len() == 3, "single image fixture expected");
            let grids = [crate::vision_input::ImageGrid {
                t: grid[0].try_into()?,
                h: grid[1].try_into()?,
                w: grid[2].try_into()?,
            }];
            let pixels = source.tensor("pixel_values_fp32")?;
            ensure!(pixels.dtype() == Dtype::F32, "fixture pixel dtype mismatch");
            let pixels: Vec<_> = pixels
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let input = MultimodalInput {
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &pixels,
                grids: &grids,
            };
            let request = runtime.prepare(&device, &mut blas, &input, ids.len())?;
            let rows = request.prefix.positions.patch_rows();
            let mut trace = crate::trace::Trace::default();
            trace.set_device(&device.info.name)?;
            let images = runtime.encode(s, &blas, request, Some(&mut trace))?;
            ensure!(
                images.features.len() == rows / 4 * OUTPUT as usize,
                "merger output missing"
            );
            trace.save(&output.join(format!("{case}-tower.safetensors")))?;
            eprintln!(
                "VISION_TOWER_DIAGNOSTIC {case} rows={rows} written; numerical verdict requires per-stage reference checks"
            );
        }
        Ok(())
    }
}
