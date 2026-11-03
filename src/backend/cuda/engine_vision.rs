//! Vision weights, patch/position preparation, and image-feature splice for Engine.
use crate::backend::gpu::{CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use crate::{
    backend::cuda::{
        Device,
        blas::Blas,
        contracts::{Scratch, Tensor, TensorMut, dense_layout},
        cubin,
        flash::{Flash, Query, Score, Shape},
        linear::{ClippedLinear, LinearBiasMode, LinearWeightRef, disjoint},
        ops::{CommonOpsRef, GeluMode, RmsMode, RmsOptions, flat, grid},
        vision::{
            PatchGrid, PatchPacking, VisionAttnBindings, VisionAttnOps, VisionAttnPlan,
            VisionMergerBindings, VisionMergerOps, VisionMergerPlan, VisionSpec,
        },
        weights::{StoredTensor, WeightCatalog},
    },
    model_spec::{Architecture, ModelSpec},
    vision_input::{ImageGrid, MultimodalInput, PositionPlan, SpliceRange, VisionPositions},
};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::{mem::size_of, sync::Arc};

#[cfg(test)]
const HIDDEN: u32 = 1152;
#[cfg(test)]
const PATCH: u32 = 1536;
#[cfg(test)]
const OUTPUT: u32 = 5120;

/// Tower geometry resolved from the validated model spec.
#[derive(Clone, Copy)]
struct VisionGeometry {
    spec: VisionSpec,
    patch: u32,
    mlp: u32,
    /// Patch rows per row after the insert merger: 1 without an insert, `vision.merge² / 4` with one.
    insert_scale: u32,
    /// Encoder layer after which the window-attention merger runs, if the model has one.
    insert_layer: Option<u32>,
    /// Encoder blocks whose outputs feed the deepstack mergers, if the model has them.
    deepstack: Option<[u32; 3]>,
}

impl VisionGeometry {
    fn from_model(model: &ModelSpec) -> Result<Self> {
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        vision.validate(model.architecture)?;
        let spec = match model.architecture {
            crate::model_spec::Architecture::Qwen35Text
            | crate::model_spec::Architecture::Qwen3Vl => VisionSpec::q38(),
            crate::model_spec::Architecture::MiniCpmV47 => {
                ensure!(
                    vision.heads > 0 && vision.hidden.is_multiple_of(vision.heads),
                    "vision heads do not divide the hidden width"
                );
                let merge = vision.merge.isqrt();
                ensure!(
                    merge * merge == vision.merge,
                    "vision merge {} is not a square",
                    vision.merge
                );
                VisionSpec {
                    hq: vision.heads,
                    d: vision.hidden / vision.heads,
                    hidden: vision.hidden,
                    merger_fc1: vision
                        .hidden
                        .checked_mul(merge)
                        .and_then(|width| width.checked_mul(merge))
                        .context("vision merger width overflow")?,
                    merger_fc2: vision.output,
                    merge,
                    merger_norm: crate::backend::cuda::vision::MergerNorm::Grouped,
                    rope: false,
                    ln_eps: crate::backend::cuda::vision::LN_EPS,
                }
            }
            crate::model_spec::Architecture::Llama
            | Architecture::Qwen3
            | crate::model_spec::Architecture::Flint
            | crate::model_spec::Architecture::Glm5Next
            | crate::model_spec::Architecture::LoopVl
            | crate::model_spec::Architecture::DeepseekV41
            | crate::model_spec::Architecture::Gemma4
            | crate::model_spec::Architecture::EmbeddingGemma2
            | crate::model_spec::Architecture::Clover
            | crate::model_spec::Architecture::DiffusionGemma => {
                bail!("vision configuration requires a multimodal architecture")
            }
        };
        spec.validate()?;
        let total = vision
            .merge
            .checked_mul(vision.merge)
            .context("merge overflow")?;
        let final_merge = spec.merge * spec.merge;
        ensure!(
            total.is_multiple_of(final_merge),
            "total downsample does not cover the final merge"
        );
        Ok(Self {
            spec,
            patch: u32::try_from(vision.patch_elements(model.architecture)?)?,
            mlp: vision.intermediate,
            insert_scale: total / final_merge,
            insert_layer: vision.minicpm.as_ref().map(|m| m.insert_layer),
            deepstack: match *vision.deepstack.as_slice() {
                [] => None,
                [a, b, c] => Some([a, b, c]),
                ref taps => bail!("unsupported deepstack taps: {taps:?}"),
            },
        })
    }

    fn hidden(self) -> u32 {
        self.spec.hidden
    }

    fn output(self) -> u32 {
        self.spec.merger_fc2
    }
}

fn dense<T: crate::backend::gpu::DeviceRepr + Copy, const R: usize>(
    buffer: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    let (stride, count) = dense_layout(shape)?;
    ensure!(
        count == buffer.len() as u64,
        "vision view/allocation size mismatch: view={count} allocated={}",
        buffer.len()
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

fn dense_at<T: crate::backend::gpu::DeviceRepr + Copy, const R: usize>(
    buffer: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    shape: [u32; R],
    offset: usize,
) -> Result<Tensor<T, R>> {
    let (stride, count) = dense_layout(shape)?;
    let end = (offset as u64)
        .checked_add(count)
        .context("vision view offset overflow")?;
    ensure!(
        end <= buffer.len() as u64,
        "vision view/allocation size mismatch: end={end} allocated={}",
        buffer.len()
    );
    let (ptr, _read) = buffer.device_ptr(stream);
    let bytes = count
        .checked_mul(size_of::<T>() as u64)
        .context("vision view byte overflow")?;
    let offset_bytes = u64::try_from(offset)?
        .checked_mul(size_of::<T>() as u64)
        .context("vision view offset overflow")?;
    unsafe { Tensor::new(ptr + offset_bytes, shape, stride, bytes) }
}

/// Stage view: exact-length `dense` when the extent covers the whole allocation, otherwise the
/// offset-0 prefix `dense_at`.
pub(super) fn stage_view<T: crate::backend::gpu::DeviceRepr + Copy, const R: usize>(
    buffer: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    let (_, count) = dense_layout(shape)?;
    if count == buffer.len() as u64 {
        dense(buffer, stream, shape)
    } else {
        dense_at(buffer, stream, shape, 0)
    }
}

fn matrix<T: Copy>(view: &Tensor<T, 2>, width: u32) -> Result<()> {
    view.validate()?;
    ensure!(
        view.shape[1] == width && view.stride == [u64::from(width), 1],
        "vision IO requires dense rows of width {width}, got shape {:?} stride {:?}",
        view.shape,
        view.stride
    );
    Ok(())
}

fn splice_bounds(
    features: &Tensor<bf16, 2>,
    output: &Tensor<bf16, 2>,
    range: &SpliceRange,
    width: u32,
) -> Result<u32> {
    matrix(features, width)?;
    matrix(output, width)?;
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

/// One image segment of the split-QKV pack: three token-major projections into head-major Q,
/// padded head-major K and V.
struct SegmentPack<'a> {
    q_in: &'a Tensor<bf16, 2>,
    k_in: &'a Tensor<bf16, 2>,
    v_in: &'a Tensor<bf16, 2>,
    q: &'a Tensor<bf16, 3>,
    /// Destination of the padded K rows.
    k: u64,
    v: &'a Tensor<bf16, 3>,
    rows: u32,
    padded: u32,
    first: u32,
    key_first: u32,
    n: u32,
}

pub(super) struct VisionIo {
    cast: CudaFunction,
    position_add: CudaFunction,
    splice: CudaFunction,
    width: u32,
    pack: CudaFunction,
    residual: CudaFunction,
    wide: WideVisionOps,
}

/// Model-independent versions of the vision IO ops (`kernels/vision_ops.cu`).
struct WideVisionOps {
    pack_split: CudaFunction,
    bucket_add: CudaFunction,
    gather: CudaFunction,
    mean: CudaFunction,
    patch_rows: CudaFunction,
}

pub(super) struct PositionInputs {
    pub table: Tensor<bf16, 2>,
    pub indices: Tensor<i32, 2>,
    pub weights: Tensor<f32, 2>,
    pub patches: Tensor<bf16, 2>,
    pub bias: Option<Tensor<bf16, 1>>,
    pub trace_patches: bool,
}

struct PatchPositionBindings<'a> {
    patch: &'a BiasedWeight,
    position: &'a CudaSlice<bf16>,
    indices: &'a CudaSlice<i32>,
    bilinear: Option<&'a BilinearTables>,
    packed: &'a CudaSlice<bf16>,
    patch_output: &'a CudaSlice<bf16>,
    hidden: &'a CudaSlice<bf16>,
    trace_patches: bool,
}

impl VisionIo {
    pub(super) fn new(device: &Device, width: u32) -> Result<Self> {
        let splice = match width {
            2048 => "vision_splice_2048",
            5120 => "q38_vision_splice",
            _ => "vision_splice_rows",
        };
        ensure!(width > 0, "vision splice output width is zero");
        let module = cubin::module(&device.ctx, "q38_vision_io")?;
        let wide = cubin::module(&device.ctx, "vision_ops")?;
        Ok(Self {
            cast: module.load_function("q38_vision_cast_patches")?,
            position_add: module.load_function("q38_vision_position_add")?,
            splice: module.load_function(splice)?,
            width,
            pack: module.load_function("q38_vision_pack")?,
            residual: module.load_function("q38_vision_residual")?,
            wide: WideVisionOps {
                pack_split: wide.load_function("vision_pack_split")?,
                bucket_add: wide.load_function("vision_bucket_position_add")?,
                gather: wide.load_function("vision_row_gather")?,
                mean: wide.load_function("vision_mean_rows")?,
                patch_rows: wide.load_function("vision_patch_rows")?,
            },
        })
    }

    fn cast_patches(
        &self,
        stream: &Arc<CudaStream>,
        input: Tensor<f32, 2>,
        output: &mut TensorMut<bf16, 2>,
        patch: u32,
    ) -> Result<()> {
        matrix(&input, patch)?;
        matrix(&output.0, patch)?;
        ensure!(input.shape == output.0.shape, "patch cast shape mismatch");
        disjoint(&input, &output.0)?;
        ensure!(patch == 1536, "Q38 patch cast width mismatch");
        let count = u64::from(input.shape[0]) * u64::from(patch);
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
        hidden: u32,
    ) -> Result<()> {
        let PositionInputs {
            table,
            indices,
            weights,
            patches,
            bias,
            trace_patches,
        } = input;
        matrix(&table, hidden)?;
        ensure!(
            table.shape[0] == 2304,
            "vision position table must be 48x48"
        );
        matrix(&patches, hidden)?;
        let rows = patches.shape[0];
        matrix(&indices, rows)?;
        matrix(&weights, rows)?;
        ensure!(
            indices.shape[0] == 4 && weights.shape[0] == 4,
            "bilinear tables need four corners"
        );
        matrix(&output.0, hidden)?;
        matrix(&interpolated.0, hidden)?;
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
        let bias_ptr = if let Some(bias) = bias {
            bias.validate()?;
            ensure!(
                bias.shape == [hidden] && bias.stride == [1],
                "position bias shape mismatch"
            );
            disjoint(&bias, &patches)?;
            disjoint(&bias, &output.0)?;
            disjoint(&bias, &interpolated.0)?;
            bias.ptr
        } else {
            ensure!(!trace_patches, "position patch trace requires bias");
            0
        };
        if trace_patches {
            disjoint(&table, &patches)?;
            disjoint(&indices, &patches)?;
            disjoint(&weights, &patches)?;
        }
        let trace_ptr = if trace_patches { patches.ptr } else { 0 };
        let count = u64::from(rows) * u64::from(hidden);
        unsafe {
            stream
                .launch_builder(&self.position_add)
                .arg(&table.ptr)
                .arg(&indices.ptr)
                .arg(&weights.ptr)
                .arg(&patches.ptr)
                .arg(&bias_ptr)
                .arg(&trace_ptr)
                .arg(&output.0.ptr)
                .arg(&interpolated.0.ptr)
                .arg(&rows)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    fn project_position(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        bindings: PatchPositionBindings<'_>,
    ) -> Result<()> {
        let PatchPositionBindings {
            patch,
            position,
            indices,
            bilinear,
            packed,
            patch_output,
            hidden,
            trace_patches,
        } = bindings;
        let width = patch.shape[0];
        let rows = u32::try_from(packed.len() / patch.shape[1] as usize)?;
        let output = dense(patch_output, stream, [rows, width])?;
        let table = dense(
            position,
            stream,
            [u32::try_from(position.len() / width as usize)?, width],
        )?;
        disjoint(&table, &output)?;
        let input = dense(packed, stream, [rows, patch.shape[1]])?;
        let bias = dense(&patch.bias, stream, [width])?;
        disjoint(&bias, &output)?;
        if cfg!(feature = "cuda") && bilinear.is_some() {
            blas.linear_typed(
                crate::op_contract::ExecutionPhase::Prefill,
                LinearWeightRef::Bf16(dense(&patch.weight, stream, patch.shape)?),
                input,
                &mut TensorMut(output),
            )?;
        } else {
            patch.run(stream, blas, input, &mut TensorMut(output))?;
        }
        let mut hidden = TensorMut(dense(hidden, stream, [rows, width])?);
        match bilinear {
            Some(bilinear) => self.position_add(
                stream,
                PositionInputs {
                    table,
                    indices: dense(indices, stream, [4, rows])?,
                    weights: dense(&bilinear.coefficients, stream, [4, rows])?,
                    patches: output,
                    bias: cfg!(feature = "cuda").then_some(bias),
                    trace_patches: cfg!(feature = "cuda") && trace_patches,
                },
                &mut hidden,
                &mut TensorMut(dense(&bilinear.values, stream, [rows, width])?),
                width,
            ),
            None => self.bucket_position_add(
                stream,
                table,
                dense(indices, stream, [rows])?,
                output,
                &mut hidden,
                width,
            ),
        }
    }

    fn splice(
        &self,
        stream: &Arc<CudaStream>,
        features: Tensor<bf16, 2>,
        output: &mut TensorMut<bf16, 2>,
        range: &SpliceRange,
    ) -> Result<()> {
        let width = self.width;
        let rows = splice_bounds(&features, &output.0, range, width)?;
        let source = u32::try_from(range.feature_rows.start)?;
        let destination = u32::try_from(range.chunk_rows.start)?;
        let count = u64::from(rows) * u64::from(width);
        unsafe {
            let mut launch = stream.launch_builder(&self.splice);
            launch
                .arg(&features.ptr)
                .arg(&output.0.ptr)
                .arg(&features.shape[0])
                .arg(&output.0.shape[0])
                .arg(&source)
                .arg(&destination)
                .arg(&rows);
            if !matches!(width, 2048 | 5120) {
                launch.arg(&width);
            }
            launch.launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    /// MiniCPM learned nearest-bucket position add: `table[indices[row]] + patch`.
    fn bucket_position_add(
        &self,
        stream: &Arc<CudaStream>,
        table: Tensor<bf16, 2>,
        indices: Tensor<i32, 1>,
        patches: Tensor<bf16, 2>,
        output: &mut TensorMut<bf16, 2>,
        hidden: u32,
    ) -> Result<()> {
        matrix(&table, hidden)?;
        matrix(&patches, hidden)?;
        matrix(&output.0, hidden)?;
        ensure!(
            indices.shape == [patches.shape[0]]
                && indices.stride == [1]
                && patches.shape == output.0.shape,
            "bucket position shape mismatch"
        );
        disjoint(&table, &output.0)?;
        disjoint(&patches, &output.0)?;
        disjoint(&indices, &output.0)?;
        let rows = patches.shape[0];
        let table_rows = table.shape[0];
        let count = u64::from(rows) * u64::from(hidden);
        unsafe {
            stream
                .launch_builder(&self.wide.bucket_add)
                .arg(&table.ptr)
                .arg(&indices.ptr)
                .arg(&patches.ptr)
                .arg(&output.0.ptr)
                .arg(&rows)
                .arg(&table_rows)
                .arg(&hidden)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    /// Three token-major projections -> head-major Q, padded K and V over one image segment.
    fn pack_split(
        &self,
        stream: &Arc<CudaStream>,
        pack: SegmentPack<'_>,
        spec: &VisionSpec,
    ) -> Result<()> {
        ensure!(
            pack.q_in.shape[0] == pack.k_in.shape[0]
                && pack.q_in.shape[0] == pack.v_in.shape[0]
                && pack.q_in.shape[1] == spec.hidden,
            "split projection shapes disagree"
        );
        let count = u64::from(pack.n) * u64::from(spec.hidden);
        unsafe {
            stream
                .launch_builder(&self.wide.pack_split)
                .arg(&pack.q_in.ptr)
                .arg(&pack.k_in.ptr)
                .arg(&pack.v_in.ptr)
                .arg(&pack.q.ptr)
                .arg(&pack.k)
                .arg(&pack.v.ptr)
                .arg(&pack.rows)
                .arg(&pack.padded)
                .arg(&pack.first)
                .arg(&pack.key_first)
                .arg(&pack.n)
                .arg(&spec.hidden)
                .arg(&spec.hq)
                .arg(&spec.d)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    /// NaViT pack -> per-patch rows with a BF16 cast: rows `[patch, channels*side*side]`.
    fn patch_rows(
        &self,
        stream: &Arc<CudaStream>,
        input: Tensor<f32, 2>,
        output: &mut TensorMut<bf16, 2>,
        rows: u32,
        channels: u32,
        side: u32,
    ) -> Result<()> {
        ensure!(
            channels > 0 && side > 0,
            "patch row pack needs positive channels/side: channels={channels} side={side}"
        );
        let width = channels
            .checked_mul(side)
            .and_then(|n| n.checked_mul(side))
            .context("patch row width overflow")?;
        matrix(&input, width)?;
        matrix(&output.0, width)?;
        ensure!(
            input.shape[0] == rows && output.0.shape[0] == rows && input.shape == output.0.shape,
            "patch row pack shape mismatch: rows={rows} input={:?} output={:?}",
            input.shape,
            output.0.shape
        );
        disjoint(&input, &output.0)?;
        let count = u64::from(rows) * u64::from(width);
        unsafe {
            stream
                .launch_builder(&self.wide.patch_rows)
                .arg(&input.ptr)
                .arg(&output.0.ptr)
                .arg(&rows)
                .arg(&channels)
                .arg(&side)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    /// Row gather over `width`-wide BF16 rows: `gathered[r] = input[indices[r]]`.
    fn row_gather(
        &self,
        stream: &Arc<CudaStream>,
        input: Tensor<bf16, 2>,
        indices: Tensor<i32, 1>,
        gathered: &mut TensorMut<bf16, 2>,
        rows: u32,
        width: u32,
    ) -> Result<()> {
        matrix(&input, width)?;
        matrix(&gathered.0, width)?;
        ensure!(
            input.shape[0] == rows
                && gathered.0.shape[0] == rows
                && input.shape[0] > 0
                && indices.shape == [rows]
                && indices.stride == [1],
            "window gather shape mismatch: rows={rows} input={:?} gathered={:?} indices={:?}",
            input.shape,
            gathered.0.shape,
            indices.shape
        );
        disjoint(&input, &gathered.0)?;
        disjoint(&indices, &gathered.0)?;
        let count = u64::from(rows) * u64::from(width);
        unsafe {
            stream
                .launch_builder(&self.wide.gather)
                .arg(&input.ptr)
                .arg(&indices.ptr)
                .arg(&gathered.0.ptr)
                .arg(&rows)
                .arg(&input.shape[0])
                .arg(&width)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    /// Mean of every consecutive `k`-row group over `width` BF16 columns.
    fn row_mean(
        &self,
        stream: &Arc<CudaStream>,
        input: Tensor<bf16, 2>,
        out: &mut TensorMut<bf16, 2>,
        rows: u32,
        width: u32,
        k: u32,
    ) -> Result<()> {
        ensure!(k > 0, "row mean needs a positive group size");
        matrix(&input, width)?;
        matrix(&out.0, width)?;
        ensure!(
            input.shape[0] == rows && out.0.shape[0] == rows / k && rows.is_multiple_of(k),
            "row mean shape mismatch: rows={rows} k={k} input={:?} output={:?}",
            input.shape,
            out.0.shape
        );
        disjoint(&input, &out.0)?;
        let count = u64::from(rows / k) * u64::from(width);
        unsafe {
            stream
                .launch_builder(&self.wide.mean)
                .arg(&input.ptr)
                .arg(&out.0.ptr)
                .arg(&rows)
                .arg(&width)
                .arg(&k)
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
    fn take(catalog: &mut WeightCatalog, name: &str, width: u32) -> Result<Self> {
        Ok(Self {
            weight: take(catalog, &format!("{name}.weight"), &[width as usize])?,
            bias: take(catalog, &format!("{name}.bias"), &[width as usize])?,
        })
    }
}

struct BiasedWeight {
    weight: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    shape: [u32; 2],
    bias_mode: LinearBiasMode,
}
impl BiasedWeight {
    fn take(
        catalog: &mut WeightCatalog,
        name: &str,
        shape: [u32; 2],
        stored_shape: &[usize],
        bias_mode: LinearBiasMode,
    ) -> Result<Self> {
        Ok(Self {
            weight: take(catalog, &format!("{name}.weight"), stored_shape)?,
            bias: take(catalog, &format!("{name}.bias"), &[shape[0] as usize])?,
            shape,
            bias_mode,
        })
    }
    fn patch(
        catalog: &mut WeightCatalog,
        name: &str,
        shape: [u32; 2],
        stored_shape: &[usize],
    ) -> Result<Self> {
        Self::take(
            catalog,
            name,
            shape,
            stored_shape,
            LinearBiasMode::AfterOutputRound,
        )
    }
    fn prepare(&self, stream: &Arc<CudaStream>, blas: &mut Blas, rows: u32) -> Result<()> {
        blas.prepare_biased(
            rows,
            self.shape[0],
            self.shape[1],
            dense(&self.bias, stream, [self.shape[0]])?,
            self.bias_mode,
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
            self.bias_mode,
        )
    }
}

enum QkvWeights {
    /// Fused `[3 * hidden, hidden]` projection of the shared Q38 block.
    Fused(BiasedWeight),
    /// Separate `[hidden, hidden]` Q, K and V projections.
    Split(Box<QkvSplit>),
}

struct QkvSplit {
    q: BiasedWeight,
    k: BiasedWeight,
    v: BiasedWeight,
}

struct BlockWeights {
    norm1: AffineWeights,
    qkv: QkvWeights,
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

/// MiniCPM's window-attention merger inserted after encoder layer `insert_layer`.
struct WindowMergerWeights {
    q: BiasedWeight,
    k: BiasedWeight,
    v: BiasedWeight,
    projection: BiasedWeight,
    norm1: AffineWeights,
    pre_norm: AffineWeights,
    fc1: BiasedWeight,
    fc2: BiasedWeight,
}

struct VisionWeights {
    patch: BiasedWeight,
    position: CudaSlice<bf16>,
    post_norm: Option<AffineWeights>,
    blocks: Vec<BlockWeights>,
    window: Option<WindowMergerWeights>,
    merger: MergerWeights,
    /// One merger per deepstack tap, LayerNorm over the merged `merger_fc1` row.
    deepstack: Vec<MergerWeights>,
}

impl VisionWeights {
    fn take(
        catalog: &mut WeightCatalog,
        geometry: VisionGeometry,
        vision: &crate::model_spec::VisionConfig,
        architecture: crate::model_spec::Architecture,
    ) -> Result<Self> {
        let before = catalog.tensors.len();
        let hidden = geometry.hidden();
        let mlp = geometry.mlp;
        let depth = vision.depth;
        let merge = geometry.spec.merge;
        let hidden_usize = hidden as usize;
        let mlp_usize = mlp as usize;
        let fc1_usize = geometry.spec.merger_fc1 as usize;
        let output_usize = geometry.output() as usize;
        let position_rows = (vision.position_side as usize)
            .checked_mul(vision.position_side as usize)
            .context("vision position table rows overflow")?;
        let patch_shape: Vec<usize> = if vision.temporal_patch > 1 {
            vec![
                hidden_usize,
                vision.channels as usize,
                vision.temporal_patch as usize,
                vision.patch as usize,
                vision.patch as usize,
            ]
        } else {
            vec![
                hidden_usize,
                vision.channels as usize,
                vision.patch as usize,
                vision.patch as usize,
            ]
        };
        let window_width = (merge * merge) as usize * mlp_usize;
        let (patch, position, post_norm, blocks, window, merger, deepstack, expected) =
            match architecture {
                Architecture::Qwen35Text | Architecture::Qwen3Vl => {
                    let patch = BiasedWeight::patch(
                        catalog,
                        "model.visual.patch_embed.proj",
                        [hidden, geometry.patch],
                        &patch_shape,
                    )?;
                    let position = take(
                        catalog,
                        "model.visual.pos_embed.weight",
                        &[position_rows, hidden_usize],
                    )?;
                    let mut blocks = Vec::with_capacity(depth as usize);
                    for i in 0..depth {
                        let p = format!("model.visual.blocks.{i}");
                        blocks.push(BlockWeights {
                            norm1: AffineWeights::take(catalog, &format!("{p}.norm1"), hidden)?,
                            qkv: QkvWeights::Fused(BiasedWeight::take(
                                catalog,
                                &format!("{p}.attn.qkv"),
                                [3 * hidden, hidden],
                                &[3 * hidden_usize, hidden_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?),
                            projection: BiasedWeight::take(
                                catalog,
                                &format!("{p}.attn.proj"),
                                [hidden, hidden],
                                &[hidden_usize, hidden_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?,
                            norm2: AffineWeights::take(catalog, &format!("{p}.norm2"), hidden)?,
                            fc1: BiasedWeight::take(
                                catalog,
                                &format!("{p}.mlp.linear_fc1"),
                                [mlp, hidden],
                                &[mlp_usize, hidden_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?,
                            fc2: BiasedWeight::take(
                                catalog,
                                &format!("{p}.mlp.linear_fc2"),
                                [hidden, mlp],
                                &[hidden_usize, mlp_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?,
                        });
                    }
                    let merger = MergerWeights {
                        norm: AffineWeights::take(catalog, "model.visual.merger.norm", hidden)?,
                        fc1: BiasedWeight::take(
                            catalog,
                            "model.visual.merger.linear_fc1",
                            [geometry.spec.merger_fc1, geometry.spec.merger_fc1],
                            &[fc1_usize, fc1_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                        fc2: BiasedWeight::take(
                            catalog,
                            "model.visual.merger.linear_fc2",
                            [geometry.output(), geometry.spec.merger_fc1],
                            &[output_usize, fc1_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                    };
                    let deepstack = (0..vision.deepstack.len())
                        .map(|k| {
                            let p = format!("model.visual.deepstack_merger_list.{k}");
                            Ok(MergerWeights {
                                norm: AffineWeights::take(
                                    catalog,
                                    &format!("{p}.norm"),
                                    geometry.spec.merger_fc1,
                                )?,
                                fc1: BiasedWeight::take(
                                    catalog,
                                    &format!("{p}.linear_fc1"),
                                    [geometry.spec.merger_fc1, geometry.spec.merger_fc1],
                                    &[fc1_usize, fc1_usize],
                                    LinearBiasMode::BeforeOutputRound,
                                )?,
                                fc2: BiasedWeight::take(
                                    catalog,
                                    &format!("{p}.linear_fc2"),
                                    [geometry.output(), geometry.spec.merger_fc1],
                                    &[output_usize, fc1_usize],
                                    LinearBiasMode::BeforeOutputRound,
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let expected = 2 + 1 + 6 + depth as usize * 12 + 6 * deepstack.len();
                    (
                        patch, position, None, blocks, None, merger, deepstack, expected,
                    )
                }
                Architecture::MiniCpmV47 => {
                    let patch = BiasedWeight::patch(
                        catalog,
                        "model.vision_tower.embeddings.patch_embedding",
                        [hidden, geometry.patch],
                        &patch_shape,
                    )?;
                    let position = take(
                        catalog,
                        "model.vision_tower.embeddings.position_embedding.weight",
                        &[position_rows, hidden_usize],
                    )?;
                    let post_norm = Some(AffineWeights::take(
                        catalog,
                        "model.vision_tower.post_layernorm",
                        hidden,
                    )?);
                    let projection = |catalog: &mut WeightCatalog,
                                      prefix: &str,
                                      name: &str|
                     -> Result<BiasedWeight> {
                        BiasedWeight::take(
                            catalog,
                            &format!("{prefix}.self_attn.{name}_proj"),
                            [hidden, hidden],
                            &[hidden_usize, hidden_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )
                    };
                    let mut blocks = Vec::with_capacity(depth as usize);
                    for i in 0..depth {
                        let p = format!("model.vision_tower.encoder.layers.{i}");
                        blocks.push(BlockWeights {
                            norm1: AffineWeights::take(
                                catalog,
                                &format!("{p}.layer_norm1"),
                                hidden,
                            )?,
                            qkv: QkvWeights::Split(Box::new(QkvSplit {
                                q: projection(catalog, &p, "q")?,
                                k: projection(catalog, &p, "k")?,
                                v: projection(catalog, &p, "v")?,
                            })),
                            projection: projection(catalog, &p, "out")?,
                            norm2: AffineWeights::take(
                                catalog,
                                &format!("{p}.layer_norm2"),
                                hidden,
                            )?,
                            fc1: BiasedWeight::take(
                                catalog,
                                &format!("{p}.mlp.fc1"),
                                [mlp, hidden],
                                &[mlp_usize, hidden_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?,
                            fc2: BiasedWeight::take(
                                catalog,
                                &format!("{p}.mlp.fc2"),
                                [hidden, mlp],
                                &[hidden_usize, mlp_usize],
                                LinearBiasMode::BeforeOutputRound,
                            )?,
                        });
                    }
                    let p = "model.vision_tower.vit_merger";
                    let window = WindowMergerWeights {
                        q: projection(catalog, p, "q")?,
                        k: projection(catalog, p, "k")?,
                        v: projection(catalog, p, "v")?,
                        projection: projection(catalog, p, "out")?,
                        norm1: AffineWeights::take(catalog, &format!("{p}.layer_norm1"), hidden)?,
                        pre_norm: AffineWeights::take(
                            catalog,
                            &format!("{p}.pre_norm"),
                            geometry.spec.merger_fc1,
                        )?,
                        fc1: BiasedWeight::take(
                            catalog,
                            &format!("{p}.linear_1"),
                            [merge * merge * mlp, geometry.spec.merger_fc1],
                            &[window_width, fc1_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                        fc2: BiasedWeight::take(
                            catalog,
                            &format!("{p}.linear_2"),
                            [hidden, merge * merge * mlp],
                            &[hidden_usize, window_width],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                    };
                    let m = "model.merger.mlp.0";
                    let merger = MergerWeights {
                        norm: AffineWeights::take(
                            catalog,
                            &format!("{m}.pre_norm"),
                            geometry.spec.merger_fc1,
                        )?,
                        fc1: BiasedWeight::take(
                            catalog,
                            &format!("{m}.linear_1"),
                            [geometry.spec.merger_fc1, geometry.spec.merger_fc1],
                            &[fc1_usize, fc1_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                        fc2: BiasedWeight::take(
                            catalog,
                            &format!("{m}.linear_2"),
                            [geometry.output(), geometry.spec.merger_fc1],
                            &[output_usize, fc1_usize],
                            LinearBiasMode::BeforeOutputRound,
                        )?,
                    };
                    let expected = 2 + 1 + 2 + depth as usize * 16 + 16 + 6;
                    (
                        patch,
                        position,
                        post_norm,
                        blocks,
                        Some(window),
                        merger,
                        Vec::new(),
                        expected,
                    )
                }
                Architecture::Llama
                | Architecture::Qwen3
                | Architecture::Flint
                | Architecture::Glm5Next
                | Architecture::LoopVl
                | Architecture::Gemma4
                | Architecture::EmbeddingGemma2
                | Architecture::Clover
                | Architecture::DiffusionGemma => {
                    bail!("vision configuration requires a multimodal architecture")
                }
                Architecture::DeepseekV41 => {
                    bail!("DeepSeek-V4.1 vision execution is not bound yet")
                }
            };
        ensure!(
            before.checked_sub(catalog.tensors.len()) == Some(expected),
            "incomplete vision weight binding"
        );
        Ok(Self {
            patch,
            position,
            post_norm,
            blocks,
            window,
            merger,
            deepstack,
        })
    }
}

pub(super) enum VisionRuntime {
    LayerNorm(Box<LayerNormTower>),
    Axial(Box<AxialTower>),
}

pub(super) enum VisionRequest {
    LayerNorm(Box<LayerNormRequest>),
    Axial(Box<AxialRequest>),
}

impl VisionRuntime {
    pub(super) fn request_bytes(
        model: &ModelSpec,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<usize> {
        if model.vision.as_ref().is_some_and(|v| v.axial.is_some()) {
            AxialTower::request_bytes(model, input, capacity)
        } else {
            LayerNormTower::request_bytes(model, input, capacity)
        }
    }

    pub(super) fn from_catalog(
        model: &ModelSpec,
        catalog: &mut WeightCatalog,
        device: &Device,
        common: CommonOpsRef,
    ) -> Result<Self> {
        if model.vision.as_ref().is_some_and(|v| v.axial.is_some()) {
            Ok(Self::Axial(Box::new(AxialTower::from_catalog(
                model, catalog, device, common,
            )?)))
        } else {
            Ok(Self::LayerNorm(Box::new(LayerNormTower::from_catalog(
                model, catalog, device, common,
            )?)))
        }
    }

    pub(super) fn prepare(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        capacity: usize,
        reserve: usize,
    ) -> Result<VisionRequest> {
        match self {
            Self::LayerNorm(tower) => Ok(VisionRequest::LayerNorm(Box::new(
                tower.prepare(device, blas, input, capacity, reserve)?,
            ))),
            Self::Axial(tower) => Ok(VisionRequest::Axial(Box::new(
                tower.prepare(device, blas, input, capacity, reserve)?,
            ))),
        }
    }

    pub(super) fn encode(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        request: VisionRequest,
        trace: Option<&mut crate::trace::Trace>,
    ) -> Result<EncodedImages> {
        match (self, request) {
            (Self::LayerNorm(tower), VisionRequest::LayerNorm(request)) => {
                tower.encode(stream, blas, *request, trace)
            }
            (Self::Axial(tower), VisionRequest::Axial(request)) => {
                tower.encode(stream, blas, *request, trace)
            }
            _ => bail!("vision request belongs to a different tower"),
        }
    }
}

pub(super) struct LayerNormTower {
    model: ModelSpec,
    geometry: VisionGeometry,
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
    /// Q38: four corner indices `[4, rows]`; MiniCPM: one learned-table id per row.
    indices: CudaSlice<i32>,
    bilinear: Option<BilinearTables>,
}

/// Q38 bilinear position tables and their interpolation trace buffer.
struct BilinearTables {
    coefficients: CudaSlice<f32>,
    values: CudaSlice<f32>,
}

pub(super) struct LayerNormRequest {
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
    down: Option<DownTower>,
    merger: VisionMergerPlan,
    /// One plan and `[features, output]` buffer per deepstack tap.
    deepstack: Vec<(VisionMergerPlan, CudaSlice<bf16>)>,
}

/// Stages that exist only when an insert merger divides the tower in two row extents.
pub(super) struct DownTower {
    packing: PatchPacking,
    window_packing: PatchPacking,
    attention: VisionAttnPlan,
    window: VisionAttnPlan,
    /// `[tower_rows, hidden]` tower output after the insert.
    hidden: CudaSlice<bf16>,
    /// `[rows, hidden]` patch rows in window order, carrying the window-attention residual.
    gathered: CudaSlice<bf16>,
    /// `[tower_rows, hidden]` mean of each window's four rows.
    mean: CudaSlice<bf16>,
    /// `[rows]` source patch row of each window-ordered row.
    window_indices: CudaSlice<i32>,
    /// `[tower_rows]` source tower row of each merger-group row (4 per output row).
    merge_indices: CudaSlice<i32>,
    /// `[tower_rows, hidden]` post-layernorm rows gathered into merger groups.
    grouped: CudaSlice<bf16>,
}

/// Attention packings of one request: the patch tower, and the post-insert and window stages when
/// the model has an insert merger.
struct VisionPackings {
    tower: PatchPacking,
    down: Option<PatchPacking>,
    window: Option<PatchPacking>,
}

impl VisionPackings {
    fn new(geometry: &VisionGeometry, grids: &[ImageGrid]) -> Result<Self> {
        let tower = PatchPacking::build(
            &grids
                .iter()
                .map(|g| PatchGrid { h: g.h, w: g.w })
                .collect::<Vec<_>>(),
        )?;
        if geometry.insert_layer.is_none() {
            return Ok(Self {
                tower,
                down: None,
                window: None,
            });
        }
        let merge = geometry.spec.merge;
        ensure!(
            geometry.insert_scale == merge * merge,
            "unsupported insert downsample: {}",
            geometry.insert_scale
        );
        let down = PatchPacking::build(
            &grids
                .iter()
                .map(|g| PatchGrid {
                    h: g.h / merge,
                    w: g.w / merge,
                })
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            down.real_rows() * merge * merge == tower.real_rows(),
            "insert merger row count mismatch"
        );
        let mut windows = Vec::new();
        for g in grids {
            ensure!(
                g.h.is_multiple_of(merge * merge) && g.w.is_multiple_of(merge * merge),
                "vision patch grid {}x{} is not divisible by {}",
                g.h,
                g.w,
                merge * merge
            );
            for _ in 0..(g.h / merge) * (g.w / merge) {
                windows.push(PatchGrid { h: merge, w: merge });
            }
        }
        let window = PatchPacking::build(&windows)?;
        ensure!(
            window.real_rows() == tower.real_rows(),
            "window packing row count mismatch"
        );
        Ok(Self {
            tower,
            down: Some(down),
            window: Some(window),
        })
    }

    fn v_rows(&self) -> u32 {
        let mut rows = self.tower.padded_rows();
        for packing in [self.down.as_ref(), self.window.as_ref()]
            .into_iter()
            .flatten()
        {
            rows = rows.max(packing.padded_rows());
        }
        rows
    }
}

fn tower_bytes(geometry: &VisionGeometry, packings: &VisionPackings) -> Result<(usize, usize)> {
    let packing = &packings.tower;
    let rows = packing.real_rows() as usize;
    let spec = geometry.spec;
    let hidden = geometry.hidden() as usize;
    let output = geometry.output() as usize;
    let merge = spec.merge as usize;
    let mut attention = 0u64;
    for candidate in [
        Some(packing),
        packings.down.as_ref(),
        packings.window.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let workspace = VisionAttnOps::workspace(&spec, candidate)?;
        ensure!(
            workspace.alignment == 256,
            "vision attention scratch alignment mismatch"
        );
        attention = attention.max(workspace.bytes);
    }
    let tower_rows = rows / geometry.insert_scale as usize;
    let merger = VisionMergerOps::workspace(&spec, u32::try_from(tower_rows)?)?;
    ensure!(
        merger.alignment == 256,
        "vision merger scratch alignment mismatch"
    );
    let scratch = usize::try_from(attention.max(merger.bytes))?;
    let output_rows = tower_rows / (merge * merge);
    let extra_rows = if packings.down.is_some() {
        3 * tower_rows
    } else {
        0
    };
    let gathered = usize::from(packings.down.is_some());
    let payload = [
        prefix_bytes(rows, geometry)?,
        rows.checked_mul(((5 + gathered) * hidden + 3 * hidden + geometry.mlp as usize) * 2)
            .context("tower work overflow")?,
        (packings.v_rows() as usize)
            .checked_mul(hidden * 2)
            .context("padded V overflow")?,
        extra_rows
            .checked_mul(hidden * 2)
            .context("insert buffer overflow")?,
        output_rows
            .checked_mul(output * 2 * (1 + geometry.deepstack.map_or(0, |taps| taps.len())))
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
    rows: u32,
) -> Result<()> {
    if let Some(t) = trace.as_deref_mut() {
        let mut values = stream.clone_dtoh(buffer)?;
        let count = rows as usize * width as usize;
        ensure!(
            count <= values.len() && buffer.len().is_multiple_of(width as usize),
            "trace extent exceeds its buffer"
        );
        values.truncate(count);
        t.add(
            name.into(),
            vec![rows as usize, width as usize],
            values.into_iter().map(bf16::to_f32).collect(),
        )?;
    }
    Ok(())
}

fn prefix_bytes(rows: usize, geometry: &VisionGeometry) -> Result<usize> {
    ensure!(rows > 0, "empty vision prefix");
    let projection_bytes = if cfg!(feature = "rocm") { 4 } else { 0 };
    rows.checked_mul(
        geometry.patch as usize * 6 + geometry.hidden() as usize * (8 + projection_bytes) + 4 * 8,
    )
    .context("vision prefix byte overflow")
}

/// One stage's split-QKV pack: the projection planes, the head-major destinations and the packing.
struct SplitStage<'a> {
    qkv: &'a CudaSlice<bf16>,
    packing: &'a PatchPacking,
    rows: u32,
    padded: u32,
    q: &'a Tensor<bf16, 3>,
    k_out: u64,
    v: &'a Tensor<bf16, 3>,
}

/// One tower pass over a fixed row extent: the patch rows before an insert merger, the merged rows after.
struct BlockStage<'a> {
    rows: u32,
    padded: u32,
    packing: &'a PatchPacking,
    attention: &'a VisionAttnPlan,
    /// Residual stream the blocks read and write.
    hidden: &'a CudaSlice<bf16>,
}

struct BlockBuffers<'a> {
    norm: &'a CudaSlice<bf16>,
    qkv: &'a CudaSlice<bf16>,
    q: &'a CudaSlice<bf16>,
    k: &'a CudaSlice<bf16>,
    v: &'a CudaSlice<bf16>,
    core: &'a CudaSlice<bf16>,
    delta: &'a CudaSlice<bf16>,
    mlp: &'a CudaSlice<bf16>,
}

struct BlockRun<'a> {
    stage: &'a BlockStage<'a>,
    buffers: &'a BlockBuffers<'a>,
    scratch: Scratch,
    block: &'a BlockWeights,
    index: usize,
}

impl LayerNormTower {
    /// Moves every deepstack tap one encoder block later.
    #[cfg(test)]
    pub(super) fn shift_deepstack_taps(&mut self) -> Result<()> {
        let taps = self
            .geometry
            .deepstack
            .as_mut()
            .context("tower has no deepstack taps")?;
        for tap in taps {
            *tap += 1;
        }
        Ok(())
    }
    pub(super) fn request_bytes(
        model: &ModelSpec,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<usize> {
        input.validate(model, capacity)?;
        let geometry = VisionGeometry::from_model(model)?;
        let packings = VisionPackings::new(&geometry, input.grids)?;
        Ok(tower_bytes(&geometry, &packings)?.0)
    }
    pub(super) fn prepare(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        capacity: usize,
        reserve: usize,
    ) -> Result<LayerNormRequest> {
        let positions = input.validate(&self.model, capacity)?;
        let packings = VisionPackings::new(&self.geometry, input.grids)?;
        let packing = packings.tower.clone();
        ensure!(
            packing.real_rows() as usize == positions.patch_rows(),
            "patch packing/input mismatch"
        );
        let (bytes, scratch_bytes) = tower_bytes(&self.geometry, &packings)?;
        let (free, _) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(reserve).is_some_and(|n| bytes <= n),
            "vision tower budget exceeded: required={bytes} free={free}"
        );
        let rows = packing.real_rows();
        let padded = packing.padded_rows();
        let geometry = self.geometry;
        let spec = geometry.spec;
        let hidden = geometry.hidden();
        let output = geometry.output();
        let merge = spec.merge;
        let tower_rows = rows / geometry.insert_scale;
        let s = &device.stream;
        let prefix = self.prepare_prefix(device, blas, input, capacity)?;
        let vrows = packings.v_rows();
        let norm = s.alloc_zeros(rows as usize * hidden as usize)?;
        let qkv = s.alloc_zeros(rows as usize * 3 * hidden as usize)?;
        let q = s.alloc_zeros(rows as usize * hidden as usize)?;
        let k = s.alloc_zeros(rows as usize * hidden as usize)?;
        let v = s.alloc_zeros(vrows as usize * hidden as usize)?;
        let core = s.alloc_zeros(rows as usize * hidden as usize)?;
        let delta = s.alloc_zeros(rows as usize * hidden as usize)?;
        let mlp = s.alloc_zeros(rows as usize * geometry.mlp as usize)?;
        let features =
            s.alloc_zeros(tower_rows as usize / (merge * merge) as usize * output as usize)?;
        let scratch = s.alloc_zeros(scratch_bytes)?;
        for (index, block) in self.weights.blocks.iter().enumerate() {
            let stage_rows = match self.geometry.insert_layer {
                Some(insert) if index as u32 > insert => tower_rows,
                _ => rows,
            };
            match &block.qkv {
                QkvWeights::Fused(weight) => weight.prepare(s, blas, stage_rows)?,
                QkvWeights::Split(split) => {
                    for weight in [&split.q, &split.k, &split.v] {
                        weight.prepare(s, blas, stage_rows)?;
                    }
                }
            }
            for weight in [&block.projection, &block.fc1, &block.fc2] {
                weight.prepare(s, blas, stage_rows)?;
            }
        }
        let attention = VisionAttnOps::prepare(
            device,
            &spec,
            &packings.tower,
            VisionAttnBindings {
                q_in: dense(&q, s, [spec.hq, rows, spec.d])?,
                k_in: dense(&k, s, [spec.hq, rows, spec.d])?,
                v: stage_view(&v, s, [spec.hq, packing.padded_rows(), spec.d])?,
                o: TensorMut(dense(&core, s, [rows, spec.hq, spec.d])?),
            },
        )?;
        let down = match (
            packings.down.as_ref(),
            packings.window.as_ref(),
            self.weights.window.as_ref(),
        ) {
            (Some(down_packing), Some(window_packing), Some(window_weights)) => {
                let gathered = s.alloc_zeros(rows as usize * hidden as usize)?;
                let down_hidden = s.alloc_zeros(tower_rows as usize * hidden as usize)?;
                let mean = s.alloc_zeros(tower_rows as usize * hidden as usize)?;
                let down_attention = VisionAttnOps::prepare(
                    device,
                    &spec,
                    down_packing,
                    VisionAttnBindings {
                        q_in: stage_view(&q, s, [spec.hq, tower_rows, spec.d])?,
                        k_in: stage_view(&k, s, [spec.hq, tower_rows, spec.d])?,
                        v: stage_view(&v, s, [spec.hq, down_packing.padded_rows(), spec.d])?,
                        o: TensorMut(stage_view(&core, s, [tower_rows, spec.hq, spec.d])?),
                    },
                )?;
                let window_attention = VisionAttnOps::prepare(
                    device,
                    &spec,
                    window_packing,
                    VisionAttnBindings {
                        q_in: dense(&q, s, [spec.hq, rows, spec.d])?,
                        k_in: dense(&k, s, [spec.hq, rows, spec.d])?,
                        v: stage_view(&v, s, [spec.hq, window_packing.padded_rows(), spec.d])?,
                        o: TensorMut(dense(&core, s, [rows, spec.hq, spec.d])?),
                    },
                )?;
                for weight in [
                    &window_weights.q,
                    &window_weights.k,
                    &window_weights.v,
                    &window_weights.projection,
                ] {
                    weight.prepare(s, blas, rows)?;
                }
                for weight in [&window_weights.fc1, &window_weights.fc2] {
                    weight.prepare(s, blas, tower_rows)?;
                }
                let gather = crate::vision_input::window_gather(input.grids, merge)?;
                ensure!(
                    gather.indices.len() == rows as usize
                        && gather.cu_seqlens == window_packing.cu_seqlens().to_vec(),
                    "window gather and packing disagree"
                );
                let window_indices: Vec<i32> = gather
                    .indices
                    .iter()
                    .map(|&row| i32::try_from(row).context("window index exceeds i32"))
                    .collect::<Result<_>>()?;
                let window_indices = s.clone_htod(&window_indices)?;
                let row_grids: Vec<(u32, u32)> = input
                    .grids
                    .iter()
                    .map(|g| (g.h / merge, g.w / merge))
                    .collect();
                let groups = crate::vision_input::merge_groups(&row_grids, merge)?;
                ensure!(
                    groups.len() == tower_rows as usize,
                    "merger group map extent mismatch"
                );
                let merge_indices: Vec<i32> = groups
                    .iter()
                    .map(|&row| i32::try_from(row).context("merger group row exceeds i32"))
                    .collect::<Result<_>>()?;
                let merge_indices = s.clone_htod(&merge_indices)?;
                let grouped = s.alloc_zeros(tower_rows as usize * hidden as usize)?;
                Some(DownTower {
                    packing: down_packing.clone(),
                    window_packing: window_packing.clone(),
                    attention: down_attention,
                    window: window_attention,
                    hidden: down_hidden,
                    gathered,
                    mean,
                    window_indices,
                    merge_indices,
                    grouped,
                })
            }
            (None, None, None) => None,
            _ => bail!("insert merger packings and weights disagree"),
        };
        let m = &self.weights.merger;
        let merger_input = match down.as_ref() {
            Some(tower) => &tower.grouped,
            None => &prefix.hidden,
        };
        let norm_width = match spec.merger_norm {
            crate::backend::cuda::vision::MergerNorm::PerPatch => hidden,
            crate::backend::cuda::vision::MergerNorm::Grouped => spec.merger_fc1,
        };
        let merger = VisionMergerOps::prepare(
            &spec,
            VisionMergerBindings {
                x: stage_view(merger_input, s, [tower_rows, hidden])?,
                norm_w: dense(&m.norm.weight, s, [norm_width])?,
                norm_b: dense(&m.norm.bias, s, [norm_width])?,
                fc1_w: dense(&m.fc1.weight, s, m.fc1.shape)?,
                fc1_b: dense(&m.fc1.bias, s, [spec.merger_fc1])?,
                fc2_w: dense(&m.fc2.weight, s, m.fc2.shape)?,
                fc2_b: dense(&m.fc2.bias, s, [output])?,
                out: TensorMut(dense(&features, s, [tower_rows / (merge * merge), output])?),
            },
            self.common.clone(),
            blas,
        )?;
        ensure!(
            down.is_none() || self.weights.deepstack.is_empty(),
            "deepstack taps with an insert merger are not bound"
        );
        let grouped = crate::backend::cuda::vision::VisionSpec {
            merger_norm: crate::backend::cuda::vision::MergerNorm::Grouped,
            ..spec
        };
        let deepstack = self
            .weights
            .deepstack
            .iter()
            .map(|m| {
                let out = s.alloc_zeros(
                    tower_rows as usize / (merge * merge) as usize * output as usize,
                )?;
                let plan = VisionMergerOps::prepare(
                    &grouped,
                    VisionMergerBindings {
                        x: stage_view(&prefix.hidden, s, [tower_rows, hidden])?,
                        norm_w: dense(&m.norm.weight, s, [spec.merger_fc1])?,
                        norm_b: dense(&m.norm.bias, s, [spec.merger_fc1])?,
                        fc1_w: dense(&m.fc1.weight, s, m.fc1.shape)?,
                        fc1_b: dense(&m.fc1.bias, s, [spec.merger_fc1])?,
                        fc2_w: dense(&m.fc2.weight, s, m.fc2.shape)?,
                        fc2_b: dense(&m.fc2.bias, s, [output])?,
                        out: TensorMut(dense(&out, s, [tower_rows / (merge * merge), output])?),
                    },
                    self.common.clone(),
                    blas,
                )?;
                Ok((plan, out))
            })
            .collect::<Result<Vec<_>>>()?;
        s.synchronize()?;
        eprintln!(
            "{}",
            serde_json::json!({"vision_memory_plan": {
                "rows":rows,"padded_rows":padded,"images":packing.images(),
                "payload_bytes":bytes,"operator_scratch_bytes":scratch_bytes,"free_bytes":free,"reserve_bytes":reserve
            }})
        );
        Ok(LayerNormRequest {
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
            down,
            merger,
            deepstack,
        })
    }

    pub(super) fn encode(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        mut request: LayerNormRequest,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<EncodedImages> {
        let launched = (|| -> Result<()> {
            self.run_prefix(stream, blas, &mut request.prefix, trace.is_some())?;
            let geometry = self.geometry;
            let spec = geometry.spec;
            let width = geometry.hidden();
            let rows = request.packing.real_rows();
            let tower_rows = rows / geometry.insert_scale;
            for (name, buffer, record_width, record_rows) in [
                (
                    "pixel_values_bf16",
                    &request.prefix.packed,
                    geometry.patch,
                    rows,
                ),
                (
                    "after_patch_embed",
                    &request.prefix.patch_output,
                    width,
                    rows,
                ),
                ("after_pos_add", &request.prefix.hidden, width, rows),
            ] {
                record(&mut trace, stream, name, buffer, record_width, record_rows)?;
            }
            if let (Some(t), Some(bilinear)) =
                (trace.as_deref_mut(), request.prefix.bilinear.as_ref())
            {
                t.add(
                    "pos_embeds".into(),
                    vec![rows as usize, width as usize],
                    stream.clone_dtoh(&bilinear.values)?,
                )?;
            }
            let (scratch_ptr, _hold) = request.scratch.device_ptr(stream);
            let scratch = || Scratch {
                ptr: scratch_ptr,
                bytes: request.scratch.len() as u64,
            };
            let buffers = BlockBuffers {
                norm: &request.norm,
                qkv: &request.qkv,
                q: &request.q,
                k: &request.k,
                v: &request.v,
                core: &request.core,
                delta: &request.delta,
                mlp: &request.mlp,
            };
            let mut stage = BlockStage {
                rows,
                padded: request.packing.padded_rows(),
                packing: &request.packing,
                attention: &request.attention,
                hidden: &request.prefix.hidden,
            };
            for (index, block) in self.weights.blocks.iter().enumerate() {
                if let (Some(tower), Some(insert)) = (request.down.as_ref(), geometry.insert_layer)
                    && index == insert as usize + 1
                {
                    stage = BlockStage {
                        rows: tower_rows,
                        padded: tower.packing.padded_rows(),
                        packing: &tower.packing,
                        attention: &tower.attention,
                        hidden: &tower.hidden,
                    };
                }
                self.run_block(
                    stream,
                    blas,
                    BlockRun {
                        stage: &stage,
                        buffers: &buffers,
                        scratch: scratch(),
                        block,
                        index,
                    },
                    &mut trace,
                )?;
                if let Some(k) = geometry
                    .deepstack
                    .and_then(|taps| taps.iter().position(|&tap| tap as usize == index))
                {
                    let (plan, out) = request
                        .deepstack
                        .get(k)
                        .context("deepstack merger missing")?;
                    VisionMergerOps::run(plan, stream, blas, scratch())?;
                    record(
                        &mut trace,
                        stream,
                        &format!("deepstack_{k}_out"),
                        out,
                        geometry.output(),
                        rows / (spec.merge * spec.merge),
                    )?;
                }
                if Some(index as u32) == geometry.insert_layer {
                    let insert = u32::try_from(index)?;
                    record(
                        &mut trace,
                        stream,
                        &format!("vision_layer_{insert:02}_out"),
                        stage.hidden,
                        width,
                        rows,
                    )?;
                    self.run_insert(stream, blas, &request, scratch())?;
                    record(
                        &mut trace,
                        stream,
                        "vit_merger_out",
                        &request
                            .down
                            .as_ref()
                            .context("insert stage missing")?
                            .hidden,
                        width,
                        tower_rows,
                    )?;
                }
            }
            if request.down.is_some() {
                let post_norm = self
                    .weights
                    .post_norm
                    .as_ref()
                    .context("post tower norm missing")?;
                let tower = request.down.as_ref().context("insert stage missing")?;
                self.common.norm.layer_norm(
                    stream,
                    dense(&tower.hidden, stream, [tower_rows, 1, width])?,
                    dense(&post_norm.weight, stream, [width])?,
                    dense(&post_norm.bias, stream, [width])?,
                    &mut TensorMut(dense_at(&request.norm, stream, [tower_rows, 1, width], 0)?),
                    spec.ln_eps,
                )?;
                record(
                    &mut trace,
                    stream,
                    "vision_tower_out",
                    &request.norm,
                    width,
                    tower_rows,
                )?;
                self.io.row_gather(
                    stream,
                    dense_at(&request.norm, stream, [tower_rows, width], 0)?,
                    dense(&tower.merge_indices, stream, [tower_rows])?,
                    &mut TensorMut(dense(&tower.grouped, stream, [tower_rows, width])?),
                    tower_rows,
                    width,
                )?;
            }
            VisionMergerOps::run(&request.merger, stream, blas, scratch())?;
            record(
                &mut trace,
                stream,
                "merger_out",
                &request.features,
                geometry.output(),
                tower_rows / (spec.merge * spec.merge),
            )?;
            Ok(())
        })();
        // All request allocations outlive asynchronous work, including the error path.
        let synchronized = stream.synchronize();
        launched?;
        synchronized?;
        let encoded =
            EncodedImages::new(request.prefix.positions, request.features, self.io.clone())?;
        let encoded =
            encoded.with_deepstack(request.deepstack.into_iter().map(|(_, out)| out).collect())?;
        Ok(encoded)
    }

    fn run_block(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        run: BlockRun<'_>,
        trace: &mut Option<&mut crate::trace::Trace>,
    ) -> Result<()> {
        let stage = run.stage;
        let buffers = run.buffers;
        let block = run.block;
        let index = run.index;
        let geometry = self.geometry;
        let spec = geometry.spec;
        let width = geometry.hidden();
        let rows = stage.rows;
        let last = self.weights.blocks.len().saturating_sub(1);
        if index == last {
            record(trace, stream, "b25_out", stage.hidden, width, rows)?;
        }
        self.common.norm.layer_norm(
            stream,
            dense(stage.hidden, stream, [rows, 1, width])?,
            dense(&block.norm1.weight, stream, [width])?,
            dense(&block.norm1.bias, stream, [width])?,
            &mut TensorMut(stage_view(buffers.norm, stream, [rows, 1, width])?),
            spec.ln_eps,
        )?;
        if index == 0 || index == last {
            record(
                trace,
                stream,
                &format!("b{index}.norm1"),
                buffers.norm,
                width,
                rows,
            )?;
        }
        {
            let normed = stage_view(buffers.norm, stream, [rows, width])?;
            let q = stage_view(buffers.q, stream, [spec.hq, rows, spec.d])?;
            let k = stage_view(buffers.k, stream, [spec.hq, rows, spec.d])?;
            let v = stage_view(buffers.v, stream, [spec.hq, stage.padded, spec.d])?;
            match &block.qkv {
                QkvWeights::Fused(weight) => {
                    weight.run(
                        stream,
                        blas,
                        normed,
                        &mut TensorMut(stage_view(buffers.qkv, stream, [rows, 3 * width])?),
                    )?;
                    let input = stage_view(buffers.qkv, stream, [rows, 3 * width])?;
                    for image in 0..stage.packing.images() {
                        let first = stage.packing.cu_seqlens()[image] as u32;
                        let n = stage.packing.cu_seqlens()[image + 1] as u32 - first;
                        let key_first = stage.packing.k_bases()[image] as u32;
                        unsafe {
                            stream
                                .launch_builder(&self.io.pack)
                                .arg(&input.ptr)
                                .arg(&q.ptr)
                                .arg(&k.ptr)
                                .arg(&v.ptr)
                                .arg(&rows)
                                .arg(&stage.padded)
                                .arg(&first)
                                .arg(&key_first)
                                .arg(&n)
                                .launch(grid(
                                    (u64::from(n) * u64::from(width)).div_ceil(256).min(65535)
                                        as usize,
                                    256,
                                ))?;
                        }
                    }
                }
                QkvWeights::Split(split) => {
                    let plane = rows as usize * width as usize;
                    let q_out = dense_at(buffers.qkv, stream, [rows, width], 0)?;
                    split.q.run(stream, blas, normed, &mut TensorMut(q_out))?;
                    let k_out = dense_at(buffers.qkv, stream, [rows, width], plane)?;
                    split.k.run(stream, blas, normed, &mut TensorMut(k_out))?;
                    let v_out = dense_at(buffers.qkv, stream, [rows, width], 2 * plane)?;
                    split.v.run(stream, blas, normed, &mut TensorMut(v_out))?;
                    self.pack_split_stage(
                        stream,
                        SplitStage {
                            qkv: buffers.qkv,
                            packing: stage.packing,
                            rows,
                            padded: stage.padded,
                            q: &q,
                            k_out: stage.attention.padded_key_ptr(&run.scratch),
                            v: &v,
                        },
                        &spec,
                    )?;
                }
            }
        }
        VisionAttnOps::run(stage.attention, stream, run.scratch)?;
        block.projection.run(
            stream,
            blas,
            stage_view(buffers.core, stream, [rows, width])?,
            &mut TensorMut(stage_view(buffers.delta, stream, [rows, width])?),
        )?;
        self.add_residual(
            stream,
            stage.hidden,
            buffers.delta,
            u64::from(rows) * u64::from(width),
        )?;
        self.common.norm.layer_norm(
            stream,
            dense(stage.hidden, stream, [rows, 1, width])?,
            dense(&block.norm2.weight, stream, [width])?,
            dense(&block.norm2.bias, stream, [width])?,
            &mut TensorMut(stage_view(buffers.norm, stream, [rows, 1, width])?),
            spec.ln_eps,
        )?;
        if index == 0 || index == last {
            record(
                trace,
                stream,
                &format!("b{index}.norm2"),
                buffers.norm,
                width,
                rows,
            )?;
        }
        block.fc1.run(
            stream,
            blas,
            stage_view(buffers.norm, stream, [rows, width])?,
            &mut TensorMut(stage_view(buffers.mlp, stream, [rows, geometry.mlp])?),
        )?;
        let mlp = stage_view(buffers.mlp, stream, [rows, 1, geometry.mlp])?;
        self.common
            .gate
            .gelu(stream, mlp, &mut TensorMut(mlp), GeluMode::Tanh)?;
        block.fc2.run(
            stream,
            blas,
            stage_view(buffers.mlp, stream, [rows, geometry.mlp])?,
            &mut TensorMut(stage_view(buffers.delta, stream, [rows, width])?),
        )?;
        self.add_residual(
            stream,
            stage.hidden,
            buffers.delta,
            u64::from(rows) * u64::from(width),
        )?;
        if index == 0 || index == last {
            record(
                trace,
                stream,
                &format!("block{index}_out"),
                stage.hidden,
                width,
                rows,
            )?;
        }
        Ok(())
    }

    /// Packs every image segment of one stage: three token-major planes into head-major Q,
    /// padded K and V.
    fn pack_split_stage(
        &self,
        stream: &Arc<CudaStream>,
        stage: SplitStage<'_>,
        spec: &VisionSpec,
    ) -> Result<()> {
        let width = spec.hidden;
        let plane = stage.rows as usize * width as usize;
        for segment in 0..stage.packing.images() {
            let first = stage.packing.cu_seqlens()[segment] as u32;
            let n = stage.packing.cu_seqlens()[segment + 1] as u32 - first;
            let key_first = stage.packing.k_bases()[segment] as u32;
            self.io.pack_split(
                stream,
                SegmentPack {
                    q_in: &dense_at(stage.qkv, stream, [stage.rows, width], 0)?,
                    k_in: &dense_at(stage.qkv, stream, [stage.rows, width], plane)?,
                    v_in: &dense_at(stage.qkv, stream, [stage.rows, width], 2 * plane)?,
                    q: stage.q,
                    k: stage.k_out,
                    v: stage.v,
                    rows: stage.rows,
                    padded: stage.padded,
                    first,
                    key_first,
                    n,
                },
                spec,
            )?;
        }
        Ok(())
    }

    /// Window-attention merger: gather patch rows into 2×2 windows, attend inside each window,
    /// flatten each window to one `4 · hidden` row, and add the window mean back.
    fn run_insert(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        request: &LayerNormRequest,
        scratch: Scratch,
    ) -> Result<()> {
        let tower = request.down.as_ref().context("insert stage missing")?;
        let window = self
            .weights
            .window
            .as_ref()
            .context("insert weights missing")?;
        let geometry = self.geometry;
        let spec = geometry.spec;
        let merge = spec.merge;
        let width = geometry.hidden();
        let rows = request.packing.real_rows();
        let tower_rows = rows / geometry.insert_scale;
        self.io.row_gather(
            stream,
            dense(&request.prefix.hidden, stream, [rows, width])?,
            dense(&tower.window_indices, stream, [rows])?,
            &mut TensorMut(dense(&tower.gathered, stream, [rows, width])?),
            rows,
            width,
        )?;
        self.common.norm.layer_norm(
            stream,
            dense(&tower.gathered, stream, [rows, 1, width])?,
            dense(&window.norm1.weight, stream, [width])?,
            dense(&window.norm1.bias, stream, [width])?,
            &mut TensorMut(dense(&request.norm, stream, [rows, 1, width])?),
            spec.ln_eps,
        )?;
        let normed = dense(&request.norm, stream, [rows, width])?;
        let plane = rows as usize * width as usize;
        let q_out = dense_at(&request.qkv, stream, [rows, width], 0)?;
        window.q.run(stream, blas, normed, &mut TensorMut(q_out))?;
        let k_out = dense_at(&request.qkv, stream, [rows, width], plane)?;
        window.k.run(stream, blas, normed, &mut TensorMut(k_out))?;
        let v_out = dense_at(&request.qkv, stream, [rows, width], 2 * plane)?;
        window.v.run(stream, blas, normed, &mut TensorMut(v_out))?;
        let q = dense(&request.q, stream, [spec.hq, rows, spec.d])?;
        let v = stage_view(
            &request.v,
            stream,
            [spec.hq, tower.window_packing.padded_rows(), spec.d],
        )?;
        self.pack_split_stage(
            stream,
            SplitStage {
                qkv: &request.qkv,
                packing: &tower.window_packing,
                rows,
                padded: tower.window_packing.padded_rows(),
                q: &q,
                k_out: tower.window.padded_key_ptr(&scratch),
                v: &v,
            },
            &spec,
        )?;
        VisionAttnOps::run(&tower.window, stream, scratch)?;
        window.projection.run(
            stream,
            blas,
            dense(&request.core, stream, [rows, width])?,
            &mut TensorMut(dense(&request.delta, stream, [rows, width])?),
        )?;
        self.add_residual(
            stream,
            &tower.gathered,
            &request.delta,
            u64::from(rows) * u64::from(width),
        )?;
        self.io.row_mean(
            stream,
            dense(&tower.gathered, stream, [rows, width])?,
            &mut TensorMut(dense(&tower.mean, stream, [tower_rows, width])?),
            rows,
            width,
            merge * merge,
        )?;
        self.common.norm.layer_norm(
            stream,
            dense(
                &tower.gathered,
                stream,
                [tower_rows, 1, merge * merge * width],
            )?,
            dense(&window.pre_norm.weight, stream, [merge * merge * width])?,
            dense(&window.pre_norm.bias, stream, [merge * merge * width])?,
            &mut TensorMut(dense(
                &request.norm,
                stream,
                [tower_rows, 1, merge * merge * width],
            )?),
            spec.ln_eps,
        )?;
        window.fc1.run(
            stream,
            blas,
            dense(&request.norm, stream, [tower_rows, merge * merge * width])?,
            &mut TensorMut(dense(
                &request.mlp,
                stream,
                [tower_rows, geometry.mlp * merge * merge],
            )?),
        )?;
        let mlp = dense(
            &request.mlp,
            stream,
            [tower_rows, 1, geometry.mlp * merge * merge],
        )?;
        self.common
            .gate
            .gelu(stream, mlp, &mut TensorMut(mlp), GeluMode::Tanh)?;
        window.fc2.run(
            stream,
            blas,
            dense(
                &request.mlp,
                stream,
                [tower_rows, geometry.mlp * merge * merge],
            )?,
            &mut TensorMut(dense(&tower.hidden, stream, [tower_rows, width])?),
        )?;
        self.add_residual(
            stream,
            &tower.hidden,
            &tower.mean,
            u64::from(tower_rows) * u64::from(width),
        )?;
        Ok(())
    }

    fn add_residual(
        &self,
        stream: &Arc<CudaStream>,
        hidden: &CudaSlice<bf16>,
        delta: &CudaSlice<bf16>,
        count: u64,
    ) -> Result<()> {
        ensure!(
            count > 0 && count <= hidden.len() as u64 && count <= delta.len() as u64,
            "vision residual shape mismatch"
        );
        let (h, _h) = hidden.device_ptr(stream);
        let (d, _d) = delta.device_ptr(stream);
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
        let geometry = VisionGeometry::from_model(model)?;
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let io = Arc::new(VisionIo::new(device, geometry.output())?);
        let weights = VisionWeights::take(catalog, geometry, vision, model.architecture)?;
        Ok(Self {
            model: model.clone(),
            geometry,
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
        let geometry = self.geometry;
        let bytes = prefix_bytes(rows, &geometry)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|available| bytes <= available),
            "vision prefix does not fit memory budget: required={bytes} free={free}"
        );
        let row_count = u32::try_from(rows)?;
        let vision = self
            .model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let s = &device.stream;
        let (indices, bilinear) = match self.model.architecture {
            Architecture::Qwen35Text | Architecture::Qwen3Vl => {
                let tables =
                    VisionPositions::new(vision, self.model.architecture, input.grids, rows)?;
                let indices: Vec<_> = tables.bilinear_indices.into_iter().flatten().collect();
                let coefficients: Vec<_> = tables.bilinear_weights.into_iter().flatten().collect();
                (
                    s.clone_htod(&indices)?,
                    Some(BilinearTables {
                        coefficients: s.clone_htod(&coefficients)?,
                        values: s.alloc_zeros(rows * geometry.hidden() as usize)?,
                    }),
                )
            }
            Architecture::MiniCpmV47 => {
                let ids =
                    crate::vision_input::nearest_position_ids(input.grids, vision.position_side)?;
                (s.clone_htod(&ids)?, None)
            }
            Architecture::Llama
            | Architecture::Qwen3
            | Architecture::Flint
            | Architecture::Glm5Next
            | Architecture::LoopVl
            | Architecture::Gemma4
            | Architecture::EmbeddingGemma2
            | Architecture::Clover
            | Architecture::DiffusionGemma => {
                bail!("vision configuration requires a multimodal architecture")
            }
            Architecture::DeepseekV41 => {
                bail!("DeepSeek-V4.1 vision execution is not bound yet")
            }
        };
        let hidden = geometry.hidden() as usize;
        if cfg!(feature = "cuda") && bilinear.is_some() {
            blas.prepare_request_fixed(row_count, geometry.hidden(), geometry.patch)?;
        } else {
            self.weights.patch.prepare(s, blas, row_count)?;
        }
        let prefix = VisionPrefix {
            positions,
            pixels: s.clone_htod(input.pixel_values)?,
            packed: s.alloc_zeros(rows * geometry.patch as usize)?,
            patch_output: s.alloc_zeros(rows * hidden)?,
            hidden: s.alloc_zeros(rows * hidden)?,
            indices,
            bilinear,
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
        trace_patches: bool,
    ) -> Result<()> {
        let rows = u32::try_from(request.positions.patch_rows())?;
        let geometry = self.geometry;
        let packed = dense(&request.packed, stream, [rows, geometry.patch])?;
        let launched = (|| -> Result<()> {
            match self.model.architecture {
                Architecture::Qwen35Text | Architecture::Qwen3Vl => {
                    self.io.cast_patches(
                        stream,
                        dense(&request.pixels, stream, [rows, geometry.patch])?,
                        &mut TensorMut(packed),
                        geometry.patch,
                    )?;
                }
                Architecture::MiniCpmV47 => {
                    let vision = self
                        .model
                        .vision
                        .as_ref()
                        .context("vision configuration missing")?;
                    self.io.patch_rows(
                        stream,
                        dense(&request.pixels, stream, [rows, geometry.patch])?,
                        &mut TensorMut(packed),
                        rows,
                        vision.channels,
                        vision.patch,
                    )?;
                }
                Architecture::Llama
                | Architecture::Qwen3
                | Architecture::Flint
                | Architecture::Glm5Next
                | Architecture::LoopVl
                | Architecture::Gemma4
                | Architecture::EmbeddingGemma2
                | Architecture::Clover
                | Architecture::DiffusionGemma => {
                    bail!("vision configuration requires a multimodal architecture")
                }
                Architecture::DeepseekV41 => {
                    bail!("DeepSeek-V4.1 vision execution is not bound yet")
                }
            }
            self.io.project_position(
                stream,
                blas,
                PatchPositionBindings {
                    patch: &self.weights.patch,
                    position: &self.weights.position,
                    indices: &request.indices,
                    bilinear: request.bilinear.as_ref(),
                    packed: &request.packed,
                    patch_output: &request.patch_output,
                    hidden: &request.hidden,
                    trace_patches,
                },
            )?;
            Ok(())
        })();
        let synchronized = stream.synchronize();
        launched?;
        synchronized?;
        Ok(())
    }
}

struct AxialBlock {
    input_norm: CudaSlice<bf16>,
    post_attention_norm: CudaSlice<bf16>,
    pre_ffn_norm: CudaSlice<bf16>,
    post_ffn_norm: CudaSlice<bf16>,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    q: ClippedLinear,
    k: ClippedLinear,
    v: ClippedLinear,
    output: ClippedLinear,
    gate: ClippedLinear,
    up: ClippedLinear,
    down: ClippedLinear,
}

struct AxialOps {
    affine: CudaFunction,
    position: CudaFunction,
    clamp: CudaFunction,
    pool: CudaFunction,
    rope: CudaFunction,
    branch: CudaFunction,
    activation: CudaFunction,
}

pub(super) struct AxialTower {
    model: ModelSpec,
    patch: ClippedLinear,
    position: CudaSlice<bf16>,
    projection: ClippedLinear,
    standardization: Option<[CudaSlice<bf16>; 2]>,
    blocks: Vec<AxialBlock>,
    ones: CudaSlice<bf16>,
    common: CommonOpsRef,
    ops: AxialOps,
    io: Arc<VisionIo>,
}

struct AxialImage {
    grid: ImageGrid,
    pixels: CudaSlice<f32>,
    angles: CudaSlice<bf16>,
    meta: CudaSlice<i32>,
    pool_meta: CudaSlice<i32>,
    feature_first: usize,
}

pub(super) struct AxialRequest {
    positions: PositionPlan,
    images: Vec<AxialImage>,
    patches: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    v_norm: CudaSlice<bf16>,
    core: CudaSlice<bf16>,
    delta: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    activation: CudaSlice<bf16>,
    clipped: CudaSlice<bf16>,
    pooled: CudaSlice<bf16>,
    pool_norm: CudaSlice<bf16>,
    features: CudaSlice<bf16>,
    attention: Flash,
}

impl AxialTower {
    fn from_catalog(
        model: &ModelSpec,
        catalog: &mut WeightCatalog,
        device: &Device,
        common: CommonOpsRef,
    ) -> Result<Self> {
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        vision.validate(model.architecture)?;
        let axial = vision
            .axial
            .as_ref()
            .context("axial configuration missing")?;
        let h = vision.hidden;
        let i = vision.intermediate;
        let d = axial.head_dim;
        let tower = if model.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
            "vision_tower"
        } else {
            "model.vision_tower"
        };
        let embedder = if model.architecture == crate::model_spec::Architecture::EmbeddingGemma2 {
            "embed_vision.embedding_projection.weight"
        } else {
            "model.embed_vision.embedding_projection.weight"
        };
        let patch = ClippedLinear::take(
            catalog,
            &format!("{tower}.patch_embedder.input_proj.weight"),
            [
                h,
                u32::try_from(vision.patch_elements(model.architecture)?)?,
            ],
            None,
        )?;
        let position = take(
            catalog,
            &format!("{tower}.patch_embedder.position_embedding_table"),
            &[2, vision.position_side as usize, h as usize],
        )?;
        let projection = ClippedLinear::take(catalog, embedder, [vision.output, h], None)?;
        let standardization = if axial.standardize {
            Some([
                take(catalog, &format!("{tower}.std_bias"), &[h as usize])?,
                take(catalog, &format!("{tower}.std_scale"), &[h as usize])?,
            ])
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(vision.depth as usize);
        for layer in 0..vision.depth {
            let prefix = format!("{tower}.encoder.layers.{layer}");
            let mut norm = |name: &str, width: u32| {
                take(
                    catalog,
                    &format!("{prefix}.{name}.weight"),
                    &[width as usize],
                )
            };
            let input_norm = norm("input_layernorm", h)?;
            let post_attention_norm = norm("post_attention_layernorm", h)?;
            let pre_ffn_norm = norm("pre_feedforward_layernorm", h)?;
            let post_ffn_norm = norm("post_feedforward_layernorm", h)?;
            let q_norm = norm("self_attn.q_norm", d)?;
            let k_norm = norm("self_attn.k_norm", d)?;
            let mut linear = |name: &str, shape: [u32; 2]| {
                let name = format!("{prefix}.{name}");
                ClippedLinear::take(
                    catalog,
                    &format!("{name}.linear.weight"),
                    shape,
                    axial.clipped_linears.then_some(name.as_str()),
                )
            };
            blocks.push(AxialBlock {
                input_norm,
                post_attention_norm,
                pre_ffn_norm,
                post_ffn_norm,
                q_norm,
                k_norm,
                q: linear("self_attn.q_proj", [h, h])?,
                k: linear("self_attn.k_proj", [h, h])?,
                v: linear("self_attn.v_proj", [h, h])?,
                output: linear("self_attn.o_proj", [h, h])?,
                gate: linear("mlp.gate_proj", [i, h])?,
                up: linear("mlp.up_proj", [i, h])?,
                down: linear("mlp.down_proj", [h, i])?,
            });
        }
        let module = cubin::module(&device.ctx, "vision_ops")?;
        let ops = AxialOps {
            affine: module.load_function("vision_affine_cast_bf16")?,
            position: module.load_function("vision_axis_position_add_bf16")?,
            clamp: module.load_function("vision_clamp_bf16")?,
            pool: module.load_function("vision_spatial_mean_scale_bf16")?,
            rope: cubin::module(&device.ctx, "vision_rope")?
                .load_function("vision_axial_rope_prepare_bf16")?,
            branch: cubin::module(&device.ctx, "q38_common")?.load_function("rms_branch_add")?,
            activation: cubin::module(&device.ctx, "act")?.load_function("silu_mul_split")?,
        };
        Ok(Self {
            model: model.clone(),
            patch,
            position,
            projection,
            standardization,
            blocks,
            ones: device.stream.clone_htod(&vec![bf16::ONE; h as usize])?,
            common,
            ops,
            io: Arc::new(VisionIo::new(device, vision.output)?),
        })
    }

    fn request_bytes(
        model: &ModelSpec,
        input: &MultimodalInput<'_>,
        capacity: usize,
    ) -> Result<usize> {
        let plan = input.validate(model, capacity)?;
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let axial = vision
            .axial
            .as_ref()
            .context("axial configuration missing")?;
        let max_rows = input
            .grids
            .iter()
            .map(|g| u128::from(g.h) * u128::from(g.w))
            .max()
            .context("empty image request")?;
        let patch = vision.patch_elements(model.architecture)? as u128;
        let h = u128::from(vision.hidden);
        let i = u128::from(vision.intermediate);
        let soft = max_rows / u128::from(vision.merge).pow(2);
        let workspace = 2 * max_rows * (patch + 10 * h + 3 * i + i.max(h).max(patch))
            + 4 * soft * h
            + 4 * max_rows * u128::from(vision.heads);
        let inputs = plan.patch_rows() as u128 * (4 * patch + 2 * u128::from(axial.head_dim))
            + plan.slots().len() as u128 * 64;
        let features = 2 * plan.feature_rows() as u128 * u128::from(vision.output);
        usize::try_from(
            workspace
                .checked_add(inputs)
                .and_then(|n| n.checked_add(features))
                .context("axial workspace overflow")?,
        )
        .context("axial workspace exceeds host indexing")
    }

    fn prepare(
        &self,
        device: &Device,
        blas: &mut Blas,
        input: &MultimodalInput<'_>,
        capacity: usize,
        reserve: usize,
    ) -> Result<AxialRequest> {
        let positions = input.validate(&self.model, capacity)?;
        let vision = self
            .model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let axial = vision
            .axial
            .as_ref()
            .context("axial configuration missing")?;
        let bytes = Self::request_bytes(&self.model, input, capacity)?;
        let (free, _) = device.ctx.mem_get_info()?;
        ensure!(
            free.checked_sub(reserve).is_some_and(|n| bytes <= n),
            "axial vision workspace exceeds memory budget: required={bytes} free={free}"
        );
        let rows = positions
            .slots()
            .iter()
            .map(|s| s.patches.len())
            .max()
            .context("empty image request")?;
        let h = vision.hidden as usize;
        let i = vision.intermediate as usize;
        let patch = self.patch.shape[1] as usize;
        let soft = rows / (vision.merge * vision.merge) as usize;
        let s = &device.stream;
        let attention = Flash::new_with_score(
            s.clone(),
            rows,
            vision.heads as usize,
            vision.heads as usize,
            axial.head_dim as usize,
            Score {
                scale: 1.0,
                window_left: None,
            },
            rows,
        )?;
        let frequencies = (0..axial.head_dim / 4)
            .map(|j| {
                1.0 / axial
                    .rope_theta
                    .powf(j as f32 / (axial.head_dim / 4) as f32)
            })
            .collect::<Vec<_>>();
        let mut images = Vec::with_capacity(input.grids.len());
        for slot in positions.slots() {
            let n = u32::try_from(slot.patches.len())?;
            let pooled = u32::try_from(slot.features.len())?;
            self.patch.prepare(blas, n)?;
            self.projection.prepare(blas, pooled)?;
            for block in &self.blocks {
                for projection in [
                    &block.q,
                    &block.k,
                    &block.v,
                    &block.output,
                    &block.gate,
                    &block.up,
                    &block.down,
                ] {
                    projection.prepare(blas, n)?;
                }
            }
            let mut angles = Vec::with_capacity(n as usize * axial.head_dim as usize);
            for y in 0..slot.grid.h {
                for x in 0..slot.grid.w {
                    for position in [x, y] {
                        for frequency in &frequencies {
                            let angle = position as f32 * frequency;
                            angles.push(bf16::from_f32(angle.cos()));
                            angles.push(bf16::from_f32(angle.sin()));
                        }
                    }
                }
            }
            images.push(AxialImage {
                grid: slot.grid,
                pixels: s.clone_htod(
                    &input.pixel_values[slot.patches.start * patch..slot.patches.end * patch],
                )?,
                angles: s.clone_htod(&angles)?,
                meta: s.clone_htod(&[0, n as i32, n as i32, n as i32, 0, 0, 0, 0])?,
                pool_meta: s.clone_htod(&[
                    0,
                    pooled as i32,
                    pooled as i32,
                    pooled as i32,
                    0,
                    0,
                    0,
                    0,
                ])?,
                feature_first: slot.features.start,
            });
        }
        let features = s.alloc_zeros(positions.feature_rows() * vision.output as usize)?;
        Ok(AxialRequest {
            positions,
            images,
            patches: s.alloc_zeros(rows * patch)?,
            hidden: s.alloc_zeros(rows * h)?,
            norm: s.alloc_zeros(rows * h)?,
            q: s.alloc_zeros(rows * h)?,
            k: s.alloc_zeros(rows * h)?,
            v: s.alloc_zeros(rows * h)?,
            q_norm: s.alloc_zeros(rows * h)?,
            k_norm: s.alloc_zeros(rows * h)?,
            v_norm: s.alloc_zeros(rows * h)?,
            core: s.alloc_zeros(rows * h)?,
            delta: s.alloc_zeros(rows * h)?,
            gate: s.alloc_zeros(rows * i)?,
            up: s.alloc_zeros(rows * i)?,
            activation: s.alloc_zeros(rows * i)?,
            clipped: s.alloc_zeros(rows * i.max(h).max(patch))?,
            pooled: s.alloc_zeros(soft * h)?,
            pool_norm: s.alloc_zeros(soft * h)?,
            features,
            attention,
        })
    }

    fn norm(
        &self,
        stream: &Arc<CudaStream>,
        domain: crate::backend::cuda::contracts::RowDomain,
        input: Tensor<bf16, 3>,
        weight: &CudaSlice<bf16>,
        output: Tensor<bf16, 3>,
        eps: f32,
    ) -> Result<()> {
        self.common.norm.rms_with(
            stream,
            domain,
            input,
            dense_at(weight, stream, [input.shape[2]], 0)?,
            &mut TensorMut(output),
            RmsOptions {
                mode: RmsMode::Unrounded,
                eps,
            },
        )
    }

    fn encode(
        &self,
        stream: &Arc<CudaStream>,
        blas: &Blas,
        mut request: AxialRequest,
        mut trace: Option<&mut crate::trace::Trace>,
    ) -> Result<EncodedImages> {
        use crate::backend::cuda::contracts::{RowDomain, SeqRef};
        let vision = self
            .model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let axial = vision
            .axial
            .as_ref()
            .context("axial configuration missing")?;
        let h = vision.hidden;
        let d = axial.head_dim;
        let heads = vision.heads;
        let run = (|| -> Result<()> {
            for (image_index, image) in request.images.iter().enumerate() {
                let rows = image.grid.h * image.grid.w;
                let soft = rows / (vision.merge * vision.merge);
                let domain = RowDomain {
                    meta: SeqRef {
                        ptr: image.meta.device_ptr(stream).0,
                    },
                    row_base: 0,
                    row_capacity: rows,
                };
                let linear =
                    |weight: &ClippedLinear, input: &CudaSlice<bf16>, output: &CudaSlice<bf16>| {
                        weight.run(
                            stream,
                            blas,
                            &self.ops.clamp,
                            stage_view(input, stream, [rows, weight.shape[1]])?,
                            TensorMut(stage_view(
                                &request.clipped,
                                stream,
                                [rows, weight.shape[1]],
                            )?),
                            &mut TensorMut(stage_view(output, stream, [rows, weight.shape[0]])?),
                        )
                    };
                let count = u64::from(rows) * u64::from(self.patch.shape[1]);
                unsafe {
                    stream
                        .launch_builder(&self.ops.affine)
                        .arg(&image.pixels)
                        .arg(&request.patches)
                        .arg(&count)
                        .arg(&-0.5f32)
                        .arg(&2.0f32)
                        .launch(flat(count as usize))?;
                }
                linear(&self.patch, &request.patches, &request.delta)?;
                record(
                    &mut trace,
                    stream,
                    &format!("image.{image_index}.patch"),
                    &request.delta,
                    h,
                    rows,
                )?;
                unsafe {
                    stream
                        .launch_builder(&self.ops.position)
                        .arg(&request.delta)
                        .arg(&self.position)
                        .arg(&request.hidden)
                        .arg(&image.grid.h)
                        .arg(&image.grid.w)
                        .arg(&h)
                        .arg(&vision.position_side)
                        .arg(&(vision.position_side as u64 * h as u64))
                        .arg(&(h as u64))
                        .launch(flat(rows as usize * h as usize))?;
                }
                record(
                    &mut trace,
                    stream,
                    &format!("image.{image_index}.position"),
                    &request.hidden,
                    h,
                    rows,
                )?;
                self.norm(
                    stream,
                    domain,
                    stage_view(&request.hidden, stream, [rows, 1, h])?,
                    &self.blocks[0].input_norm,
                    stage_view(&request.norm, stream, [rows, 1, h])?,
                    axial.rms_eps,
                )?;
                for (layer, block) in self.blocks.iter().enumerate() {
                    linear(&block.q, &request.norm, &request.q)?;
                    linear(&block.k, &request.norm, &request.k)?;
                    linear(&block.v, &request.norm, &request.v)?;
                    for (input, weight, output) in [
                        (&request.q, &block.q_norm, &request.q_norm),
                        (&request.k, &block.k_norm, &request.k_norm),
                        (&request.v, &self.ones, &request.v_norm),
                    ] {
                        self.norm(
                            stream,
                            domain,
                            stage_view(input, stream, [rows, heads, d])?,
                            weight,
                            stage_view(output, stream, [rows, heads, d])?,
                            axial.rms_eps,
                        )?;
                    }
                    unsafe {
                        stream
                            .launch_builder(&self.ops.rope)
                            .arg(&request.q_norm)
                            .arg(&request.k_norm)
                            .arg(&request.v_norm)
                            .arg(&image.angles)
                            .arg(&request.q)
                            .arg(&request.k)
                            .arg(&request.v)
                            .arg(&rows)
                            .arg(&heads)
                            .arg(&d)
                            .launch(flat(rows as usize * h as usize))?;
                    }
                    request.attention.run_masked(Query {
                        q: &request.q,
                        q_row_stride: h as usize,
                        k: &request.k,
                        v: &request.v,
                        out: &mut request.core,
                        shape: Shape {
                            rows: rows as usize,
                            start: 0,
                            capacity: rows as usize,
                            q_heads: heads as usize,
                            kv_heads: heads as usize,
                            head_dim: d as usize,
                            causal: false,
                            window_left: None,
                        },
                        keys: rows as usize,
                        prefix: None,
                    })?;
                    linear(&block.output, &request.core, &request.delta)?;
                    unsafe {
                        stream
                            .launch_builder(&self.ops.branch)
                            .arg(&request.hidden)
                            .arg(&request.delta)
                            .arg(&block.post_attention_norm)
                            .arg(&0u64)
                            .arg(&block.pre_ffn_norm)
                            .arg(&request.norm)
                            .arg(&h)
                            .arg(&axial.rms_eps)
                            .launch(grid(rows as usize, 256))?;
                    }
                    linear(&block.gate, &request.norm, &request.gate)?;
                    linear(&block.up, &request.norm, &request.up)?;
                    let count = i32::try_from(u64::from(rows) * u64::from(vision.intermediate))?;
                    let cfg = crate::backend::cuda::ops::silu_mul_split_grid(
                        usize::try_from(count)?,
                        request.gate.device_ptr(stream).0,
                        request.up.device_ptr(stream).0,
                        request.activation.device_ptr(stream).0,
                    )?;
                    unsafe {
                        stream
                            .launch_builder(&self.ops.activation)
                            .arg(&request.gate)
                            .arg(&request.up)
                            .arg(&request.activation)
                            .arg(&count)
                            .arg(&2i32)
                            .launch(cfg)?;
                    }
                    linear(&block.down, &request.activation, &request.delta)?;
                    let next = self
                        .blocks
                        .get(layer + 1)
                        .map_or(0, |b| b.input_norm.device_ptr(stream).0);
                    unsafe {
                        stream
                            .launch_builder(&self.ops.branch)
                            .arg(&request.hidden)
                            .arg(&request.delta)
                            .arg(&block.post_ffn_norm)
                            .arg(&0u64)
                            .arg(&next)
                            .arg(&request.norm)
                            .arg(&h)
                            .arg(&axial.rms_eps)
                            .launch(grid(rows as usize, 256))?;
                    }
                    record(
                        &mut trace,
                        stream,
                        &format!("image.{image_index}.layer.{layer}"),
                        &request.hidden,
                        h,
                        rows,
                    )?;
                }
                let (bias, scale) = self.standardization.as_ref().map_or((0, 0), |values| {
                    (
                        values[0].device_ptr(stream).0,
                        values[1].device_ptr(stream).0,
                    )
                });
                let multiplier = f64::from(h).sqrt() as f32;
                unsafe {
                    stream
                        .launch_builder(&self.ops.pool)
                        .arg(&request.hidden)
                        .arg(&request.pooled)
                        .arg(&bias)
                        .arg(&scale)
                        .arg(&image.grid.h)
                        .arg(&image.grid.w)
                        .arg(&h)
                        .arg(&vision.merge)
                        .arg(&multiplier)
                        .launch(flat(soft as usize * h as usize))?;
                }
                record(
                    &mut trace,
                    stream,
                    &format!("image.{image_index}.pool"),
                    &request.pooled,
                    h,
                    soft,
                )?;
                let domain = RowDomain {
                    meta: SeqRef {
                        ptr: image.pool_meta.device_ptr(stream).0,
                    },
                    row_base: 0,
                    row_capacity: soft,
                };
                self.norm(
                    stream,
                    domain,
                    stage_view(&request.pooled, stream, [soft, 1, h])?,
                    &self.ones,
                    stage_view(&request.pool_norm, stream, [soft, 1, h])?,
                    axial.rms_eps,
                )?;
                self.projection.run(
                    stream,
                    blas,
                    &self.ops.clamp,
                    stage_view(&request.pool_norm, stream, [soft, h])?,
                    TensorMut(stage_view(&request.clipped, stream, [soft, h])?),
                    &mut TensorMut(dense_at(
                        &request.features,
                        stream,
                        [soft, vision.output],
                        image.feature_first * vision.output as usize,
                    )?),
                )?;
            }
            record(
                &mut trace,
                stream,
                "image_features",
                &request.features,
                vision.output,
                u32::try_from(request.positions.feature_rows())?,
            )?;
            Ok(())
        })();
        let synchronized = stream.synchronize();
        run?;
        synchronized?;
        EncodedImages::new(request.positions, request.features, self.io.clone())
    }
}

/// Owns merger output until all prompt chunks have consumed their image rows.
pub(super) struct EncodedImages {
    pub(super) positions: PositionPlan,
    pub(super) features: CudaSlice<bf16>,
    /// Deepstack merger outputs in tap order, each shaped like `features`; empty without deepstack.
    pub(super) deepstack: Vec<CudaSlice<bf16>>,
    io: Arc<VisionIo>,
}
impl EncodedImages {
    /// The caller must supply the completed merger output of this request.
    pub(super) fn new(
        positions: PositionPlan,
        features: CudaSlice<bf16>,
        io: Arc<VisionIo>,
    ) -> Result<Self> {
        let width = io.width;
        let elements = positions
            .feature_rows()
            .checked_mul(width as usize)
            .context("image feature size overflow")?;
        ensure!(
            width > 0 && features.len() == elements,
            "merger feature/slot count mismatch"
        );
        Ok(Self {
            positions,
            features,
            deepstack: Vec::new(),
            io,
        })
    }
    /// Attaches the completed deepstack merger outputs of this request, each shaped like `features`.
    pub(super) fn with_deepstack(mut self, deepstack: Vec<CudaSlice<bf16>>) -> Result<Self> {
        ensure!(
            deepstack.iter().all(|f| f.len() == self.features.len()),
            "deepstack feature count mismatch"
        );
        self.deepstack = deepstack;
        Ok(self)
    }
    pub(super) fn splice(
        &self,
        stream: &Arc<CudaStream>,
        first: usize,
        output: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        matrix(&output.0, self.io.width)?;
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
            [u32::try_from(self.positions.feature_rows())?, self.io.width],
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

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DinoV3Config {
    pub hidden_act: String,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub key_bias: bool,
    pub layer_norm_eps: f32,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: u32,
    pub num_channels: u32,
    pub num_hidden_layers: usize,
    pub num_register_tokens: usize,
    pub patch_size: u32,
    pub proj_bias: bool,
    pub query_bias: bool,
    pub rope_theta: f32,
    pub use_gated_mlp: bool,
    pub value_bias: bool,
}

impl DinoV3Config {
    pub fn from_dir(dir: &std::path::Path) -> Result<Self> {
        let cfg: Self = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        ensure!(
            cfg.model_type == "dinov3_vit"
                && cfg.hidden_act == "silu"
                && cfg.use_gated_mlp
                && cfg.query_bias
                && !cfg.key_bias
                && cfg.value_bias
                && cfg.proj_bias
                && cfg.mlp_bias
                && cfg.hidden_size > 0
                && cfg.num_attention_heads > 0
                && cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
                && (cfg.hidden_size / cfg.num_attention_heads).is_multiple_of(4)
                && cfg.patch_size > 0
                && cfg.num_channels == 3
                && cfg.num_hidden_layers > 0
                && cfg.layer_norm_eps > 0.0,
            "unsupported DINOv3 configuration"
        );
        Ok(cfg)
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    pub fn patch_dim(&self) -> u32 {
        self.num_channels * self.patch_size * self.patch_size
    }

    pub fn prefix_rows(&self) -> usize {
        1 + self.num_register_tokens
    }

    pub fn patch_rows(&self, image_side: u32) -> Result<u32> {
        ensure!(
            image_side > 0 && image_side.is_multiple_of(self.patch_size),
            "image_side {image_side} not divisible by patch_size {}",
            self.patch_size
        );
        let grid = image_side / self.patch_size;
        grid.checked_mul(grid).context("DINOv3 patch_rows overflow")
    }

    pub fn total_rows(&self, image_side: u32) -> Result<u32> {
        u32::try_from(self.prefix_rows())?
            .checked_add(self.patch_rows(image_side)?)
            .context("DINOv3 total_rows overflow")
    }

    pub fn expected_loaded_elements(&self) -> usize {
        let h = self.hidden_size as usize;
        let inter = self.intermediate_size as usize;
        let patch_dim = self.patch_dim() as usize;
        let prefix = self.prefix_rows() * h;
        let patch_embed = h * patch_dim + h;
        let per_layer = 2 * (2 * h) + 4 * (h * h) + 3 * h + 2 * h + 3 * (inter * h) + 2 * inter + h;
        let final_norm = 2 * h;
        prefix + patch_embed + self.num_hidden_layers * per_layer + final_norm
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Flux2VaeConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub act_fn: String,
    pub batch_norm_eps: f32,
    pub block_out_channels: Vec<u32>,
    pub down_block_types: Vec<String>,
    pub in_channels: u32,
    pub latent_channels: u32,
    pub layers_per_block: usize,
    pub mid_block_add_attention: bool,
    pub norm_num_groups: u32,
    pub patch_size: [u32; 2],
    pub sample_size: u32,
    pub use_quant_conv: bool,
}

impl Flux2VaeConfig {
    pub fn from_dir(dir: &std::path::Path) -> Result<Self> {
        let path = if dir.join("config.json").exists()
            && (dir.join("diffusion_pytorch_model.safetensors").exists()
                || !dir.join("vae").join("config.json").exists())
        {
            dir.join("config.json")
        } else {
            dir.join("vae").join("config.json")
        };
        let cfg: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let n = cfg.block_out_channels.len();
        ensure!(
            cfg.class_name == "AutoencoderKLFlux2"
                && cfg.act_fn == "silu"
                && n > 0
                && cfg.down_block_types.len() == n
                && cfg
                    .down_block_types
                    .iter()
                    .all(|t| t == "DownEncoderBlock2D")
                && cfg.layers_per_block == 2
                && cfg.mid_block_add_attention
                && cfg.use_quant_conv
                && cfg.in_channels == 3
                && cfg.latent_channels > 0
                && cfg.norm_num_groups > 0
                && cfg
                    .block_out_channels
                    .iter()
                    .all(|&c| c > 0 && c.is_multiple_of(cfg.norm_num_groups))
                && cfg.patch_size == [2, 2]
                && cfg.sample_size > 0
                && (cfg.sample_size >> (n - 1)) > 0
                && (cfg.sample_size >> (n - 1)).is_multiple_of(cfg.patch_size[0]),
            "unsupported FLUX.2 VAE configuration"
        );
        Ok(cfg)
    }

    pub fn safetensors_path(dir: &std::path::Path) -> std::path::PathBuf {
        if dir.join("diffusion_pytorch_model.safetensors").exists() {
            dir.join("diffusion_pytorch_model.safetensors")
        } else {
            dir.join("vae").join("diffusion_pytorch_model.safetensors")
        }
    }

    pub fn latent_side(&self) -> u32 {
        self.sample_size >> (self.block_out_channels.len() - 1)
    }

    pub fn latent_spatial(&self) -> u32 {
        let s = self.latent_side();
        s * s
    }

    pub fn mid_channels(&self) -> u32 {
        *self.block_out_channels.last().expect("validated non-empty")
    }

    pub fn quant_channels(&self) -> u32 {
        2 * self.latent_channels
    }

    pub fn bn_channels(&self) -> u32 {
        self.latent_channels * self.patch_size[0] * self.patch_size[1]
    }

    pub fn patch_rows(&self) -> u32 {
        let ls = self.latent_side();
        (ls / self.patch_size[0]) * (ls / self.patch_size[1])
    }

    pub fn max_feature_elements(&self) -> usize {
        self.block_out_channels
            .iter()
            .enumerate()
            .map(|(s, &c)| {
                let hw = (self.sample_size >> s) as usize;
                hw * hw * (c as usize)
            })
            .max()
            .unwrap_or(0)
    }

    pub fn expected_loaded_elements(&self) -> usize {
        let mut total = (self.block_out_channels[0] as usize) * (self.in_channels as usize) * 9
            + (self.block_out_channels[0] as usize);
        let resnet_elems = |c_in: u32, c_out: u32| -> usize {
            let ci = c_in as usize;
            let co = c_out as usize;
            let mut e = 2 * ci + co * ci * 9 + co + 2 * co + co * co * 9 + co;
            if ci != co {
                e += co * ci + co;
            }
            e
        };
        for (s, &c1) in self.block_out_channels.iter().enumerate() {
            let c0 = if s == 0 {
                self.block_out_channels[0]
            } else {
                self.block_out_channels[s - 1]
            };
            total += resnet_elems(c0, c1) + resnet_elems(c1, c1);
            if s + 1 < self.block_out_channels.len() {
                let c = c1 as usize;
                total += c * c * 9 + c;
            }
        }
        let mc = self.mid_channels();
        let mcu = mc as usize;
        total += 2 * resnet_elems(mc, mc);
        total += 2 * mcu + 4 * (mcu * mcu + mcu);
        let qc = self.quant_channels() as usize;
        total += 2 * mcu + qc * mcu * 9 + qc;
        total += qc * qc + qc;
        total += 2 * (self.bn_channels() as usize);
        total
    }
}

struct SafetensorsLoader<'a> {
    st: safetensors::SafeTensors<'a>,
    remaining: std::collections::BTreeSet<String>,
    loaded_elements: usize,
}

impl<'a> SafetensorsLoader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(bytes)?;
        let remaining = st.names().into_iter().map(str::to_owned).collect();
        Ok(Self {
            st,
            remaining,
            loaded_elements: 0,
        })
    }

    fn take_bf16(
        &mut self,
        s: &Arc<CudaStream>,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<CudaSlice<bf16>> {
        ensure!(
            self.remaining.remove(name),
            "tensor {name} missing or loaded twice"
        );
        let view = self.st.tensor(name)?;
        ensure!(
            view.shape() == expected_shape,
            "tensor {name} shape {:?} != expected {expected_shape:?}",
            view.shape()
        );
        let host: Vec<bf16> = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| bf16::from_f32(f32::from_le_bytes(*b)))
                .collect(),
            dt => bail!("unsupported dtype {dt:?} for {name}"),
        };
        self.loaded_elements += host.len();
        Ok(s.clone_htod(&host)?)
    }

    fn remove_unused(&mut self, name: &str, expected_shape: &[usize]) -> Result<()> {
        ensure!(
            self.remaining.remove(name),
            "expected unused tensor {name} missing"
        );
        let view = self.st.tensor(name)?;
        ensure!(
            view.shape() == expected_shape,
            "unused tensor {name} shape {:?} != expected {expected_shape:?}",
            view.shape()
        );
        Ok(())
    }
}

struct UnbiasedWeight {
    weight: CudaSlice<bf16>,
    shape: [u32; 2],
}

impl UnbiasedWeight {
    fn load(
        loader: &mut SafetensorsLoader<'_>,
        s: &Arc<CudaStream>,
        name: &str,
        shape: [u32; 2],
    ) -> Result<Self> {
        Ok(Self {
            weight: loader.take_bf16(
                s,
                &format!("{name}.weight"),
                &[shape[0] as usize, shape[1] as usize],
            )?,
            shape,
        })
    }

    fn prepare(&self, blas: &Blas, rows: u32) -> Result<()> {
        blas.prepare_request_fixed(rows, self.shape[0], self.shape[1])
    }

    fn run(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        input: Tensor<bf16, 2>,
        output: &mut TensorMut<bf16, 2>,
    ) -> Result<()> {
        blas.linear_typed(
            crate::op_contract::ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(dense(&self.weight, s, self.shape)?),
            input,
            output,
        )
    }
}

fn load_biased_weight(
    loader: &mut SafetensorsLoader<'_>,
    s: &Arc<CudaStream>,
    name: &str,
    shape: [u32; 2],
    stored_shape: &[usize],
    bias_mode: LinearBiasMode,
) -> Result<BiasedWeight> {
    Ok(BiasedWeight {
        weight: loader.take_bf16(s, &format!("{name}.weight"), stored_shape)?,
        bias: loader.take_bf16(s, &format!("{name}.bias"), &[shape[0] as usize])?,
        shape,
        bias_mode,
    })
}

fn load_affine_weights(
    loader: &mut SafetensorsLoader<'_>,
    s: &Arc<CudaStream>,
    name: &str,
    width: u32,
) -> Result<AffineWeights> {
    Ok(AffineWeights {
        weight: loader.take_bf16(s, &format!("{name}.weight"), &[width as usize])?,
        bias: loader.take_bf16(s, &format!("{name}.bias"), &[width as usize])?,
    })
}

struct DinoV3Layer {
    norm1: AffineWeights,
    q: BiasedWeight,
    k: UnbiasedWeight,
    v: BiasedWeight,
    o: BiasedWeight,
    scale1: CudaSlice<bf16>,
    norm2: AffineWeights,
    gate: BiasedWeight,
    up: BiasedWeight,
    down: BiasedWeight,
    scale2: CudaSlice<bf16>,
}

pub struct DinoV3Workspace {
    pixel_packed: CudaSlice<f32>,
    patches: CudaSlice<bf16>,
    x_a: CudaSlice<bf16>,
    x_b: CudaSlice<bf16>,
    x_c: CudaSlice<bf16>,
    k_cache: CudaSlice<bf16>,
    v_cache: CudaSlice<bf16>,
    mlp_a: CudaSlice<bf16>,
    mlp_b: CudaSlice<bf16>,
    mlp_c: CudaSlice<bf16>,
    rope_table: CudaSlice<bf16>,
    feature1: CudaSlice<f32>,
    flash_d64: Flash,
}

pub struct DinoV3Encoder {
    cfg: DinoV3Config,
    image_side: u32,
    prefix_tokens: CudaSlice<bf16>,
    patch_embed: BiasedWeight,
    layers: Vec<DinoV3Layer>,
    final_norm: AffineWeights,
    common: CommonOpsRef,
    ops: crate::backend::cuda::ops::Ops,
    io: VisionIo,
    channel_scale_residual: CudaFunction,
    layer_norm_f32: CudaFunction,
    upcast_f32: CudaFunction,
    silu_mul_split: CudaFunction,
}

pub struct DinoV3Output {
    pub layer0: Option<Vec<f32>>,
    pub layer15: Option<Vec<f32>>,
    pub feature1: Vec<f32>,
}

impl DinoV3Encoder {
    pub fn load(
        device: &Device,
        blas: &mut Blas,
        common: CommonOpsRef,
        cfg: DinoV3Config,
        image_side: u32,
        safetensors_bytes: &[u8],
    ) -> Result<(Self, DinoV3Workspace)> {
        let s = &device.stream;
        let h = cfg.hidden_size;
        let hu = h as usize;
        let inter = cfg.intermediate_size;
        let inter_u = inter as usize;
        let patch = cfg.patch_size;
        let patch_u = patch as usize;
        let patch_dim = cfg.patch_dim();
        let patch_rows = cfg.patch_rows(image_side)?;
        let rows = cfg.total_rows(image_side)?;
        let rows_u = rows as usize;
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let patches_side = image_side / patch;

        let mut loader = SafetensorsLoader::new(safetensors_bytes)?;
        let cls = loader.take_bf16(s, "embeddings.cls_token", &[1, 1, hu])?;
        let reg = loader.take_bf16(
            s,
            "embeddings.register_tokens",
            &[1, cfg.num_register_tokens, hu],
        )?;
        loader.remove_unused("embeddings.mask_token", &[1, 1, hu])?;
        let mut prefix_host = s.clone_dtoh(&cls)?;
        prefix_host.extend(s.clone_dtoh(&reg)?);
        let prefix_tokens = s.clone_htod(&prefix_host)?;
        let patch_embed = load_biased_weight(
            &mut loader,
            s,
            "embeddings.patch_embeddings",
            [h, patch_dim],
            &[hu, cfg.num_channels as usize, patch_u, patch_u],
            LinearBiasMode::AfterOutputRound,
        )?;
        patch_embed.prepare(s, blas, patch_rows)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layer.{i}");
            let norm1 = load_affine_weights(&mut loader, s, &format!("{p}.norm1"), h)?;
            let q = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.attention.q_proj"),
                [h, h],
                &[hu, hu],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let k = UnbiasedWeight::load(&mut loader, s, &format!("{p}.attention.k_proj"), [h, h])?;
            let v = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.attention.v_proj"),
                [h, h],
                &[hu, hu],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let o = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.attention.o_proj"),
                [h, h],
                &[hu, hu],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let scale1 = loader.take_bf16(s, &format!("{p}.layer_scale1.lambda1"), &[hu])?;
            let norm2 = load_affine_weights(&mut loader, s, &format!("{p}.norm2"), h)?;
            let gate = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.mlp.gate_proj"),
                [inter, h],
                &[inter_u, hu],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let up = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.mlp.up_proj"),
                [inter, h],
                &[inter_u, hu],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let down = load_biased_weight(
                &mut loader,
                s,
                &format!("{p}.mlp.down_proj"),
                [h, inter],
                &[hu, inter_u],
                LinearBiasMode::BeforeOutputRound,
            )?;
            let scale2 = loader.take_bf16(s, &format!("{p}.layer_scale2.lambda1"), &[hu])?;
            for w in [&q, &v, &o, &gate, &up, &down] {
                w.prepare(s, blas, rows)?;
            }
            k.prepare(blas, rows)?;
            layers.push(DinoV3Layer {
                norm1,
                q,
                k,
                v,
                o,
                scale1,
                norm2,
                gate,
                up,
                down,
                scale2,
            });
        }
        let final_norm = load_affine_weights(&mut loader, s, "norm", h)?;
        ensure!(
            loader.remaining.is_empty(),
            "unconsumed DINOv3 safetensors keys: {:?}",
            loader.remaining
        );
        let expected_elems = cfg.expected_loaded_elements();
        ensure!(
            loader.loaded_elements == expected_elems,
            "DINOv3 loaded element count {} != {expected_elems}",
            loader.loaded_elements
        );
        let rope_host = crate::vision_input::dinov3_rope_table_bf16(
            cfg.prefix_rows(),
            patches_side,
            patches_side,
            head_dim,
            cfg.rope_theta,
        )?;
        let patch_elems = (patch_rows as usize) * (patch_dim as usize);
        let ws = DinoV3Workspace {
            pixel_packed: s.alloc_zeros(patch_elems)?,
            patches: s.alloc_zeros(patch_elems)?,
            x_a: s.alloc_zeros(rows_u * hu)?,
            x_b: s.alloc_zeros(rows_u * hu)?,
            x_c: s.alloc_zeros(rows_u * hu)?,
            k_cache: s.alloc_zeros(rows_u * hu)?,
            v_cache: s.alloc_zeros(rows_u * hu)?,
            mlp_a: s.alloc_zeros(rows_u * inter_u)?,
            mlp_b: s.alloc_zeros(rows_u * inter_u)?,
            mlp_c: s.alloc_zeros(rows_u * inter_u)?,
            rope_table: s.clone_htod(&rope_host)?,
            feature1: s.alloc_zeros(rows_u * hu)?,
            flash_d64: Flash::new_with_score(
                s.clone(),
                rows_u,
                heads as usize,
                heads as usize,
                head_dim as usize,
                Score::from_tokens((head_dim as f32).sqrt().recip(), None)?,
                rows_u,
            )?,
        };
        let q38_mod = cubin::module(&device.ctx, "q38_common")?;
        let vis_mod = cubin::module(&device.ctx, "vision_ops")?;
        let act_mod = cubin::module(&device.ctx, "act")?;
        Ok((
            Self {
                cfg,
                image_side,
                prefix_tokens,
                patch_embed,
                layers,
                final_norm,
                common,
                ops: crate::backend::cuda::ops::Ops::new(&device.ctx)?,
                io: VisionIo::new(device, h)?,
                channel_scale_residual: q38_mod.load_function("channel_scale_residual_bf16")?,
                layer_norm_f32: q38_mod.load_function("q38_layer_norm_f32")?,
                upcast_f32: vis_mod.load_function("vision_upcast_f32")?,
                silu_mul_split: act_mod.load_function("silu_mul_split")?,
            },
            ws,
        ))
    }

    fn layer_norm(
        &self,
        s: &Arc<CudaStream>,
        norm: &AffineWeights,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        rows: u32,
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        self.common.norm.layer_norm(
            s,
            dense(input, s, [rows, 1, h])?,
            dense(&norm.weight, s, [h])?,
            dense(&norm.bias, s, [h])?,
            &mut TensorMut(dense(output, s, [rows, 1, h])?),
            self.cfg.layer_norm_eps,
        )
    }

    fn scale_residual(
        &self,
        s: &Arc<CudaStream>,
        residual: &CudaSlice<bf16>,
        branch: &CudaSlice<bf16>,
        scale: &CudaSlice<bf16>,
        rows: u32,
    ) -> Result<()> {
        let h = self.cfg.hidden_size;
        let count = u64::from(rows) * u64::from(h);
        unsafe {
            s.launch_builder(&self.channel_scale_residual)
                .arg(residual)
                .arg(branch)
                .arg(scale)
                .arg(residual)
                .arg(&rows)
                .arg(&h)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    pub fn encode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        ws: &mut DinoV3Workspace,
        chw_f32: &[f32],
        rope_override: Option<&CudaSlice<bf16>>,
        capture_layers: bool,
    ) -> Result<DinoV3Output> {
        let side = self.image_side as usize;
        let patch = self.cfg.patch_size;
        let patch_dim = self.cfg.patch_dim();
        let patch_rows = self.cfg.patch_rows(self.image_side)?;
        let prefix_rows = self.cfg.prefix_rows();
        let rows = self.cfg.total_rows(self.image_side)?;
        let h = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim();
        let inter = self.cfg.intermediate_size;
        let packed_host = crate::vision_input::dinov3_pack_normalized_chw(
            chw_f32,
            side,
            side,
            patch as usize,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )?;
        s.memcpy_htod(&packed_host, &mut ws.pixel_packed)?;
        self.io.patch_rows(
            s,
            dense(&ws.pixel_packed, s, [patch_rows, patch_dim])?,
            &mut TensorMut(dense(&ws.patches, s, [patch_rows, patch_dim])?),
            patch_rows,
            self.cfg.num_channels,
            patch,
        )?;
        s.memcpy_dtod(
            &self.prefix_tokens,
            &mut ws.x_a.slice_mut(0..prefix_rows * h as usize),
        )?;
        self.patch_embed.run(
            s,
            blas,
            dense(&ws.patches, s, [patch_rows, patch_dim])?,
            &mut TensorMut(dense_at(
                &ws.x_a,
                s,
                [patch_rows, h],
                prefix_rows * h as usize,
            )?),
        )?;
        let rope_slice = rope_override.unwrap_or(&ws.rope_table);
        let mut layer0 = None;
        let mut layer15 = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            self.layer_norm(s, &layer.norm1, &ws.x_a, &ws.x_b, rows)?;
            layer.q.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(dense(&ws.x_c, s, [rows, h])?),
            )?;
            layer.k.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(stage_view(&ws.mlp_a, s, [rows, h])?),
            )?;
            layer.v.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(stage_view(&ws.mlp_b, s, [rows, h])?),
            )?;
            self.ops.rotary_prefix_kv(
                s,
                crate::backend::cuda::ops::RotaryPrefixKv {
                    q: TensorMut(dense(&ws.x_c, s, [rows, heads, head_dim])?),
                    key: stage_view(&ws.mlp_a, s, [rows, heads, head_dim])?,
                    value: stage_view(&ws.mlp_b, s, [rows, heads, head_dim])?,
                    k: TensorMut(dense(&ws.k_cache, s, [heads, rows, head_dim])?),
                    v: TensorMut(dense(&ws.v_cache, s, [heads, rows, head_dim])?),
                    table: dense(rope_slice, s, [rows, head_dim])?,
                    start: 0,
                },
            )?;
            ws.flash_d64.run(
                &ws.x_c,
                h as usize,
                &ws.k_cache,
                &ws.v_cache,
                &mut ws.x_b,
                Shape {
                    rows: rows as usize,
                    start: 0,
                    capacity: rows as usize,
                    q_heads: heads as usize,
                    kv_heads: heads as usize,
                    head_dim: head_dim as usize,
                    causal: false,
                    window_left: None,
                },
            )?;
            layer.o.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(dense(&ws.x_c, s, [rows, h])?),
            )?;
            self.scale_residual(s, &ws.x_a, &ws.x_c, &layer.scale1, rows)?;
            self.layer_norm(s, &layer.norm2, &ws.x_a, &ws.x_b, rows)?;
            layer.gate.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(dense(&ws.mlp_a, s, [rows, inter])?),
            )?;
            layer.up.run(
                s,
                blas,
                dense(&ws.x_b, s, [rows, h])?,
                &mut TensorMut(dense(&ws.mlp_b, s, [rows, inter])?),
            )?;
            let mlp_elems = (rows * inter) as usize;
            let cfg = crate::backend::cuda::ops::silu_mul_split_grid(
                mlp_elems,
                ws.mlp_a.device_ptr(s).0,
                ws.mlp_b.device_ptr(s).0,
                ws.mlp_c.device_ptr(s).0,
            )?;
            let mlp_count = i32::try_from(mlp_elems)?;
            unsafe {
                s.launch_builder(&self.silu_mul_split)
                    .arg(&ws.mlp_a)
                    .arg(&ws.mlp_b)
                    .arg(&ws.mlp_c)
                    .arg(&mlp_count)
                    .arg(&0i32)
                    .launch(cfg)?;
            }
            layer.down.run(
                s,
                blas,
                dense(&ws.mlp_c, s, [rows, inter])?,
                &mut TensorMut(dense(&ws.x_c, s, [rows, h])?),
            )?;
            self.scale_residual(s, &ws.x_a, &ws.x_c, &layer.scale2, rows)?;
            if capture_layers && idx == 0 {
                layer0 = Some(
                    s.clone_dtoh(&ws.x_a)?
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect(),
                );
            } else if capture_layers && idx == 15 {
                layer15 = Some(
                    s.clone_dtoh(&ws.x_a)?
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect(),
                );
            }
        }
        self.layer_norm(s, &self.final_norm, &ws.x_a, &ws.x_b, rows)?;
        let total_elems = rows * h;
        unsafe {
            s.launch_builder(&self.upcast_f32)
                .arg(&ws.x_b)
                .arg(&ws.feature1)
                .arg(&total_elems)
                .launch(grid((total_elems as usize).div_ceil(256).min(65535), 256))?;
            s.launch_builder(&self.layer_norm_f32)
                .arg(&ws.feature1)
                .arg(&0u64)
                .arg(&0u64)
                .arg(&ws.feature1)
                .arg(&rows)
                .arg(&1u32)
                .arg(&h)
                .arg(&u64::from(h))
                .arg(&u64::from(h))
                .arg(&1u64)
                .arg(&u64::from(h))
                .arg(&u64::from(h))
                .arg(&1u64)
                .arg(&self.cfg.layer_norm_eps)
                .launch(grid(rows as usize, 256))?;
        }
        Ok(DinoV3Output {
            layer0,
            layer15,
            feature1: s.clone_dtoh(&ws.feature1)?,
        })
    }
}

#[derive(Clone, Copy)]
struct Conv2dGeom {
    h: u32,
    w: u32,
    c_in: u32,
    c_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    pad: [u32; 4],
}

impl Conv2dGeom {
    fn out_hw(self) -> (u32, u32) {
        (
            (self.h + self.pad[0] + self.pad[1] - self.kh) / self.sh + 1,
            (self.w + self.pad[2] + self.pad[3] - self.kw) / self.sw + 1,
        )
    }

    fn strip_h(self, strip_pixels: u32) -> u32 {
        let (out_h, out_w) = self.out_hw();
        (strip_pixels / out_w).min(out_h)
    }
}

struct VaeResnetBlock {
    norm1: AffineWeights,
    conv1: BiasedWeight,
    norm2: AffineWeights,
    conv2: BiasedWeight,
    shortcut: Option<BiasedWeight>,
    h: u32,
    w: u32,
    c_in: u32,
    c_out: u32,
}

struct VaeDownBlock {
    resnets: [VaeResnetBlock; 2],
    downsampler: Option<(BiasedWeight, u32, u32, u32)>,
}

struct VaeMidAttn {
    group_norm: AffineWeights,
    to_q: BiasedWeight,
    to_k: BiasedWeight,
    to_v: BiasedWeight,
    to_out: BiasedWeight,
}

pub struct Flux2VaeWorkspace {
    pub strip_pixels: u32,
    input_chw: CudaSlice<f32>,
    buf_0: CudaSlice<bf16>,
    buf_1: CudaSlice<bf16>,
    buf_2: CudaSlice<bf16>,
    im2col: CudaSlice<bf16>,
    mid_v: CudaSlice<bf16>,
    noise_nchw: CudaSlice<bf16>,
    moments_nchw: CudaSlice<bf16>,
    feature2: CudaSlice<f32>,
    flash_d512: Flash,
}

pub struct Flux2VaeEncoder {
    cfg: Flux2VaeConfig,
    prefix_rows: usize,
    conv_in: BiasedWeight,
    down_blocks: Vec<VaeDownBlock>,
    mid_resnet0: VaeResnetBlock,
    mid_attn: VaeMidAttn,
    mid_resnet1: VaeResnetBlock,
    conv_norm_out: AffineWeights,
    conv_out: BiasedWeight,
    quant_conv: BiasedWeight,
    bn_mean: CudaSlice<bf16>,
    bn_var: CudaSlice<bf16>,
    ops: crate::backend::cuda::ops::Ops,
    chw_scale_shift: CudaFunction,
    im2col_fn: CudaFunction,
    group_norm_fn: CudaFunction,
    act_fn: CudaFunction,
    diag_gauss_bn_fn: CudaFunction,
}

pub struct Flux2VaeOutput {
    pub moments_f32: Vec<f32>,
    pub feature2_f32: Vec<f32>,
}

impl Flux2VaeEncoder {
    fn load_resnet(
        loader: &mut SafetensorsLoader<'_>,
        s: &Arc<CudaStream>,
        blas: &mut Blas,
        prefix: &str,
        hw_c: (u32, u32, u32, u32),
        strip_pixels: u32,
    ) -> Result<VaeResnetBlock> {
        let (h, w, c_in, c_out) = hw_c;
        let norm1 = load_affine_weights(loader, s, &format!("{prefix}.norm1"), c_in)?;
        let conv1 = load_biased_weight(
            loader,
            s,
            &format!("{prefix}.conv1"),
            [c_out, c_in * 9],
            &[c_out as usize, c_in as usize, 3, 3],
            LinearBiasMode::AfterOutputRound,
        )?;
        let norm2 = load_affine_weights(loader, s, &format!("{prefix}.norm2"), c_out)?;
        let conv2 = load_biased_weight(
            loader,
            s,
            &format!("{prefix}.conv2"),
            [c_out, c_out * 9],
            &[c_out as usize, c_out as usize, 3, 3],
            LinearBiasMode::AfterOutputRound,
        )?;
        let strip_rows = ((strip_pixels / w).min(h)) * w;
        conv1.prepare(s, blas, strip_rows)?;
        conv2.prepare(s, blas, strip_rows)?;
        let shortcut = if c_in != c_out {
            let sc = load_biased_weight(
                loader,
                s,
                &format!("{prefix}.conv_shortcut"),
                [c_out, c_in],
                &[c_out as usize, c_in as usize, 1, 1],
                LinearBiasMode::AfterOutputRound,
            )?;
            sc.prepare(s, blas, h * w)?;
            Some(sc)
        } else {
            None
        };
        Ok(VaeResnetBlock {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
            h,
            w,
            c_in,
            c_out,
        })
    }

    pub fn load(
        device: &Device,
        blas: &mut Blas,
        cfg: Flux2VaeConfig,
        prefix_rows: usize,
        safetensors_bytes: &[u8],
        reserve_bytes: usize,
    ) -> Result<(Self, Flux2VaeWorkspace)> {
        let s = &device.stream;
        let mut loader = SafetensorsLoader::new(safetensors_bytes)?;
        let names: Vec<String> = loader.remaining.iter().cloned().collect();
        let mut removed_decoder = 0usize;
        for name in names {
            if name.starts_with("decoder.") || name.starts_with("post_quant_conv.") {
                let shape = loader.st.tensor(&name)?.shape().to_vec();
                loader.remove_unused(&name, &shape)?;
                removed_decoder += 1;
            }
        }
        ensure!(
            removed_decoder == 140,
            "unexpected FLUX.2 VAE decoder key count {removed_decoder}"
        );
        let sample_size = cfg.sample_size;
        let in_ch = cfg.in_channels;
        let latent_side = cfg.latent_side();
        let latent_spatial = cfg.latent_spatial();
        let mid_ch = cfg.mid_channels();
        let quant_ch = cfg.quant_channels();
        let bn_ch = cfg.bn_channels();
        let feature2_rows = prefix_rows
            .checked_add(cfg.patch_rows() as usize)
            .context("feature2 row count overflow")?;
        let input_chw_elems = (in_ch as usize) * (sample_size as usize) * (sample_size as usize);
        let max_map = cfg.max_feature_elements();
        let mid_v_elems = (latent_spatial as usize) * (mid_ch as usize);
        let noise_elems = (cfg.latent_channels as usize) * (latent_spatial as usize);
        let moments_elems = (quant_ch as usize) * (latent_spatial as usize);
        let feature2_elems = feature2_rows * (bn_ch as usize);
        let base_ws = input_chw_elems * size_of::<f32>()
            + 3 * max_map * size_of::<bf16>()
            + mid_v_elems * size_of::<bf16>()
            + noise_elems * size_of::<bf16>()
            + moments_elems * size_of::<bf16>()
            + feature2_elems * size_of::<f32>();
        let (free, _) = device.ctx.mem_get_info()?;
        let avail = free.saturating_sub(reserve_bytes);
        let avail_im2col = avail
            .checked_sub(base_ws)
            .context("FLUX.2 VAE fixed workspace exceeds available device memory")?;
        let max_c_stage = cfg
            .block_out_channels
            .iter()
            .position(|&c| c == mid_ch)
            .context("mid_channels missing from block_out_channels")?;
        let peak_max_c_side = sample_size >> max_c_stage;
        let peak_max_c_pixels = (peak_max_c_side * peak_max_c_side) as usize;
        let col_bytes = (mid_ch as usize) * 9 * size_of::<bf16>();
        let max_fit_pixels = (avail_im2col / col_bytes).min(peak_max_c_pixels);
        ensure!(
            max_fit_pixels >= sample_size as usize,
            "FLUX.2 VAE im2col workspace does not fit a single row strip ({avail_im2col} B available)"
        );
        let strip_pixels = 1u32 << (max_fit_pixels as u32).ilog2();

        let c0_init = cfg.block_out_channels[0];
        let conv_in = load_biased_weight(
            &mut loader,
            s,
            "encoder.conv_in",
            [c0_init, in_ch * 9],
            &[c0_init as usize, in_ch as usize, 3, 3],
            LinearBiasMode::AfterOutputRound,
        )?;
        conv_in.prepare(
            s,
            blas,
            ((strip_pixels / sample_size).min(sample_size)) * sample_size,
        )?;
        let num_stages = cfg.block_out_channels.len();
        let mut down_blocks = Vec::with_capacity(num_stages);
        for (idx, &c1) in cfg.block_out_channels.iter().enumerate() {
            let hw = sample_size >> idx;
            let c0 = if idx == 0 {
                c0_init
            } else {
                cfg.block_out_channels[idx - 1]
            };
            let has_down = idx + 1 < num_stages;
            let p = format!("encoder.down_blocks.{idx}");
            let r0 = Self::load_resnet(
                &mut loader,
                s,
                blas,
                &format!("{p}.resnets.0"),
                (hw, hw, c0, c1),
                strip_pixels,
            )?;
            let r1 = Self::load_resnet(
                &mut loader,
                s,
                blas,
                &format!("{p}.resnets.1"),
                (hw, hw, c1, c1),
                strip_pixels,
            )?;
            let downsampler = if has_down {
                let conv = load_biased_weight(
                    &mut loader,
                    s,
                    &format!("{p}.downsamplers.0.conv"),
                    [c1, c1 * 9],
                    &[c1 as usize, c1 as usize, 3, 3],
                    LinearBiasMode::AfterOutputRound,
                )?;
                let out_hw = hw / 2;
                let strip_rows = ((strip_pixels / out_hw).min(out_hw)) * out_hw;
                conv.prepare(s, blas, strip_rows)?;
                Some((conv, hw, hw, c1))
            } else {
                None
            };
            down_blocks.push(VaeDownBlock {
                resnets: [r0, r1],
                downsampler,
            });
        }
        let mid_resnet0 = Self::load_resnet(
            &mut loader,
            s,
            blas,
            "encoder.mid_block.resnets.0",
            (latent_side, latent_side, mid_ch, mid_ch),
            strip_pixels,
        )?;
        let mcu = mid_ch as usize;
        let mid_attn = VaeMidAttn {
            group_norm: load_affine_weights(
                &mut loader,
                s,
                "encoder.mid_block.attentions.0.group_norm",
                mid_ch,
            )?,
            to_q: load_biased_weight(
                &mut loader,
                s,
                "encoder.mid_block.attentions.0.to_q",
                [mid_ch, mid_ch],
                &[mcu, mcu],
                LinearBiasMode::BeforeOutputRound,
            )?,
            to_k: load_biased_weight(
                &mut loader,
                s,
                "encoder.mid_block.attentions.0.to_k",
                [mid_ch, mid_ch],
                &[mcu, mcu],
                LinearBiasMode::BeforeOutputRound,
            )?,
            to_v: load_biased_weight(
                &mut loader,
                s,
                "encoder.mid_block.attentions.0.to_v",
                [mid_ch, mid_ch],
                &[mcu, mcu],
                LinearBiasMode::BeforeOutputRound,
            )?,
            to_out: load_biased_weight(
                &mut loader,
                s,
                "encoder.mid_block.attentions.0.to_out.0",
                [mid_ch, mid_ch],
                &[mcu, mcu],
                LinearBiasMode::BeforeOutputRound,
            )?,
        };
        for w in [
            &mid_attn.to_q,
            &mid_attn.to_k,
            &mid_attn.to_v,
            &mid_attn.to_out,
        ] {
            w.prepare(s, blas, latent_spatial)?;
        }
        let mid_resnet1 = Self::load_resnet(
            &mut loader,
            s,
            blas,
            "encoder.mid_block.resnets.1",
            (latent_side, latent_side, mid_ch, mid_ch),
            strip_pixels,
        )?;
        let conv_norm_out = load_affine_weights(&mut loader, s, "encoder.conv_norm_out", mid_ch)?;
        let qcu = quant_ch as usize;
        let conv_out = load_biased_weight(
            &mut loader,
            s,
            "encoder.conv_out",
            [quant_ch, mid_ch * 9],
            &[qcu, mcu, 3, 3],
            LinearBiasMode::AfterOutputRound,
        )?;
        conv_out.prepare(
            s,
            blas,
            ((strip_pixels / latent_side).min(latent_side)) * latent_side,
        )?;
        let quant_conv = load_biased_weight(
            &mut loader,
            s,
            "quant_conv",
            [quant_ch, quant_ch],
            &[qcu, qcu, 1, 1],
            LinearBiasMode::AfterOutputRound,
        )?;
        quant_conv.prepare(s, blas, latent_spatial)?;
        let bn_mean = loader.take_bf16(s, "bn.running_mean", &[bn_ch as usize])?;
        let bn_var = loader.take_bf16(s, "bn.running_var", &[bn_ch as usize])?;
        loader.remove_unused("bn.num_batches_tracked", &[])?;
        ensure!(
            loader.remaining.is_empty(),
            "unconsumed FLUX.2 VAE encoder keys: {:?}",
            loader.remaining
        );
        let expected_elems = cfg.expected_loaded_elements();
        ensure!(
            loader.loaded_elements == expected_elems,
            "FLUX.2 VAE loaded element count {} != {expected_elems}",
            loader.loaded_elements
        );
        let ws = Flux2VaeWorkspace {
            strip_pixels,
            input_chw: s.alloc_zeros(input_chw_elems)?,
            buf_0: s.alloc_zeros(max_map)?,
            buf_1: s.alloc_zeros(max_map)?,
            buf_2: s.alloc_zeros(max_map)?,
            im2col: s.alloc_zeros((strip_pixels as usize) * mcu * 9)?,
            mid_v: s.alloc_zeros(mid_v_elems)?,
            noise_nchw: s.alloc_zeros(noise_elems)?,
            moments_nchw: s.alloc_zeros(moments_elems)?,
            feature2: s.alloc_zeros(feature2_elems)?,
            flash_d512: Flash::new_with_score(
                s.clone(),
                latent_spatial as usize,
                1,
                1,
                mcu,
                Score::from_tokens((mid_ch as f32).sqrt().recip(), None)?,
                latent_spatial as usize,
            )?,
        };
        let vis_mod = cubin::module(&device.ctx, "vision_ops")?;
        let q38_mod = cubin::module(&device.ctx, "q38_common")?;
        Ok((
            Self {
                cfg,
                prefix_rows,
                conv_in,
                down_blocks,
                mid_resnet0,
                mid_attn,
                mid_resnet1,
                conv_norm_out,
                conv_out,
                quant_conv,
                bn_mean,
                bn_var,
                ops: crate::backend::cuda::ops::Ops::new(&device.ctx)?,
                chw_scale_shift: vis_mod.load_function("chw_to_nhwc_scale_shift_bf16")?,
                im2col_fn: vis_mod.load_function("masked_strided_patch_bf16")?,
                group_norm_fn: q38_mod.load_function("group_norm_nhwc_bf16")?,
                act_fn: vis_mod.load_function("activation_bf16")?,
                diag_gauss_bn_fn: vis_mod.load_function("diagonal_gaussian_patch_bn_f32")?,
            },
            ws,
        ))
    }

    fn group_norm(
        &self,
        s: &Arc<CudaStream>,
        norm: &AffineWeights,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        hw_c: (u32, u32, u32),
    ) -> Result<()> {
        let (h, w, c) = hw_c;
        let spatial = h * w;
        let groups = self.cfg.norm_num_groups;
        unsafe {
            s.launch_builder(&self.group_norm_fn)
                .arg(input)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(output)
                .arg(&1u32)
                .arg(&spatial)
                .arg(&c)
                .arg(&groups)
                .arg(&1e-6f32)
                .launch(grid(groups as usize, 256))?;
        }
        Ok(())
    }

    fn silu_in_place(
        &self,
        s: &Arc<CudaStream>,
        buf: &CudaSlice<bf16>,
        rows: u32,
        width: u32,
    ) -> Result<()> {
        let count = u64::from(rows) * u64::from(width);
        unsafe {
            s.launch_builder(&self.act_fn)
                .arg(buf)
                .arg(buf)
                .arg(&rows)
                .arg(&width)
                .arg(&1i32)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        Ok(())
    }

    fn add_in_place(
        &self,
        s: &Arc<CudaStream>,
        dst: &CudaSlice<bf16>,
        src: &CudaSlice<bf16>,
        count: u32,
    ) -> Result<()> {
        let n = i32::try_from(count)?;
        unsafe {
            s.launch_builder(&self.ops.residual)
                .arg(dst)
                .arg(src)
                .arg(&n)
                .launch(flat((count as usize).div_ceil(8)))?;
        }
        Ok(())
    }

    fn conv2d(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        conv: &BiasedWeight,
        buffers: (&CudaSlice<bf16>, &CudaSlice<bf16>, &CudaSlice<bf16>),
        geom: Conv2dGeom,
        strip_pixels: u32,
    ) -> Result<()> {
        let (input, im2col, output) = buffers;
        if geom.kh == 1 && geom.kw == 1 && geom.sh == 1 && geom.sw == 1 && geom.pad == [0; 4] {
            let rows = geom.h * geom.w;
            return conv.run(
                s,
                blas,
                stage_view(input, s, [rows, geom.c_in])?,
                &mut TensorMut(stage_view(output, s, [rows, geom.c_out])?),
            );
        }
        let (out_h, out_w) = geom.out_hw();
        let strip_h = geom.strip_h(strip_pixels);
        let col_width = geom.c_in * geom.kh * geom.kw;
        let mut y0 = 0u32;
        while y0 < out_h {
            let yc = strip_h.min(out_h - y0);
            let strip_rows = yc * out_w;
            let count = u64::from(strip_rows) * u64::from(col_width);
            unsafe {
                s.launch_builder(&self.im2col_fn)
                    .arg(input)
                    .arg(&0u64)
                    .arg(im2col)
                    .arg(&geom.h)
                    .arg(&geom.w)
                    .arg(&geom.c_in)
                    .arg(&geom.kh)
                    .arg(&geom.kw)
                    .arg(&geom.sh)
                    .arg(&geom.sw)
                    .arg(&geom.pad[0])
                    .arg(&geom.pad[1])
                    .arg(&geom.pad[2])
                    .arg(&geom.pad[3])
                    .arg(&y0)
                    .arg(&yc)
                    .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
            }
            conv.run(
                s,
                blas,
                stage_view(im2col, s, [strip_rows, col_width])?,
                &mut TensorMut(dense_at(
                    output,
                    s,
                    [strip_rows, geom.c_out],
                    (y0 * out_w * geom.c_out) as usize,
                )?),
            )?;
            y0 += yc;
        }
        Ok(())
    }

    fn run_resnet(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        block: &VaeResnetBlock,
        ws: &mut Flux2VaeWorkspace,
    ) -> Result<()> {
        let rows = block.h * block.w;
        let sp = ws.strip_pixels;
        self.group_norm(
            s,
            &block.norm1,
            &ws.buf_0,
            &ws.buf_1,
            (block.h, block.w, block.c_in),
        )?;
        self.silu_in_place(s, &ws.buf_1, rows, block.c_in)?;
        self.conv2d(
            s,
            blas,
            &block.conv1,
            (&ws.buf_1, &ws.im2col, &ws.buf_2),
            Conv2dGeom {
                h: block.h,
                w: block.w,
                c_in: block.c_in,
                c_out: block.c_out,
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                pad: [1, 1, 1, 1],
            },
            sp,
        )?;
        self.group_norm(
            s,
            &block.norm2,
            &ws.buf_2,
            &ws.buf_1,
            (block.h, block.w, block.c_out),
        )?;
        self.silu_in_place(s, &ws.buf_1, rows, block.c_out)?;
        self.conv2d(
            s,
            blas,
            &block.conv2,
            (&ws.buf_1, &ws.im2col, &ws.buf_2),
            Conv2dGeom {
                h: block.h,
                w: block.w,
                c_in: block.c_out,
                c_out: block.c_out,
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                pad: [1, 1, 1, 1],
            },
            sp,
        )?;
        if let Some(sc) = &block.shortcut {
            self.conv2d(
                s,
                blas,
                sc,
                (&ws.buf_0, &ws.im2col, &ws.buf_1),
                Conv2dGeom {
                    h: block.h,
                    w: block.w,
                    c_in: block.c_in,
                    c_out: block.c_out,
                    kh: 1,
                    kw: 1,
                    sh: 1,
                    sw: 1,
                    pad: [0, 0, 0, 0],
                },
                sp,
            )?;
            self.add_in_place(s, &ws.buf_1, &ws.buf_2, rows * block.c_out)?;
            std::mem::swap(&mut ws.buf_0, &mut ws.buf_1);
        } else {
            self.add_in_place(s, &ws.buf_0, &ws.buf_2, rows * block.c_out)?;
        }
        Ok(())
    }

    pub fn encode(
        &self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        ws: &mut Flux2VaeWorkspace,
        chw_f32: &[f32],
        noise_bf16: &[bf16],
        downsample_pad: [u32; 4],
    ) -> Result<Flux2VaeOutput> {
        let sample_size = self.cfg.sample_size;
        let in_ch = self.cfg.in_channels;
        let c0_init = self.cfg.block_out_channels[0];
        let latent_side = self.cfg.latent_side();
        let latent_spatial = self.cfg.latent_spatial();
        let mid_ch = self.cfg.mid_channels();
        let quant_ch = self.cfg.quant_channels();
        let latent_ch = self.cfg.latent_channels;
        let bn_ch = self.cfg.bn_channels() as usize;

        s.memcpy_htod(chw_f32, &mut ws.input_chw)?;
        s.memcpy_htod(noise_bf16, &mut ws.noise_nchw)?;
        let in_elems = u64::from(in_ch) * u64::from(sample_size) * u64::from(sample_size);
        unsafe {
            s.launch_builder(&self.chw_scale_shift)
                .arg(&ws.input_chw)
                .arg(&ws.buf_1)
                .arg(&in_ch)
                .arg(&sample_size)
                .arg(&sample_size)
                .arg(&2.0f32)
                .arg(&-1.0f32)
                .launch(grid(in_elems.div_ceil(256).min(65535) as usize, 256))?;
        }
        let sp = ws.strip_pixels;
        self.conv2d(
            s,
            blas,
            &self.conv_in,
            (&ws.buf_1, &ws.im2col, &ws.buf_0),
            Conv2dGeom {
                h: sample_size,
                w: sample_size,
                c_in: in_ch,
                c_out: c0_init,
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                pad: [1, 1, 1, 1],
            },
            sp,
        )?;
        for stage in &self.down_blocks {
            for r in &stage.resnets {
                self.run_resnet(s, blas, r, ws)?;
            }
            if let Some((conv, h, w, c)) = &stage.downsampler {
                self.conv2d(
                    s,
                    blas,
                    conv,
                    (&ws.buf_0, &ws.im2col, &ws.buf_1),
                    Conv2dGeom {
                        h: *h,
                        w: *w,
                        c_in: *c,
                        c_out: *c,
                        kh: 3,
                        kw: 3,
                        sh: 2,
                        sw: 2,
                        pad: downsample_pad,
                    },
                    sp,
                )?;
                std::mem::swap(&mut ws.buf_0, &mut ws.buf_1);
            }
        }
        self.run_resnet(s, blas, &self.mid_resnet0, ws)?;
        self.group_norm(
            s,
            &self.mid_attn.group_norm,
            &ws.buf_0,
            &ws.buf_1,
            (latent_side, latent_side, mid_ch),
        )?;
        self.mid_attn.to_q.run(
            s,
            blas,
            stage_view(&ws.buf_1, s, [latent_spatial, mid_ch])?,
            &mut TensorMut(stage_view(&ws.buf_2, s, [latent_spatial, mid_ch])?),
        )?;
        self.mid_attn.to_k.run(
            s,
            blas,
            stage_view(&ws.buf_1, s, [latent_spatial, mid_ch])?,
            &mut TensorMut(stage_view(&ws.im2col, s, [latent_spatial, mid_ch])?),
        )?;
        self.mid_attn.to_v.run(
            s,
            blas,
            stage_view(&ws.buf_1, s, [latent_spatial, mid_ch])?,
            &mut TensorMut(dense(&ws.mid_v, s, [latent_spatial, mid_ch])?),
        )?;
        ws.flash_d512.run(
            &ws.buf_2,
            mid_ch as usize,
            &ws.im2col,
            &ws.mid_v,
            &mut ws.buf_1,
            Shape {
                rows: latent_spatial as usize,
                start: 0,
                capacity: latent_spatial as usize,
                q_heads: 1,
                kv_heads: 1,
                head_dim: mid_ch as usize,
                causal: false,
                window_left: None,
            },
        )?;
        self.mid_attn.to_out.run(
            s,
            blas,
            stage_view(&ws.buf_1, s, [latent_spatial, mid_ch])?,
            &mut TensorMut(stage_view(&ws.buf_2, s, [latent_spatial, mid_ch])?),
        )?;
        self.add_in_place(s, &ws.buf_0, &ws.buf_2, latent_spatial * mid_ch)?;
        self.run_resnet(s, blas, &self.mid_resnet1, ws)?;
        self.group_norm(
            s,
            &self.conv_norm_out,
            &ws.buf_0,
            &ws.buf_1,
            (latent_side, latent_side, mid_ch),
        )?;
        self.silu_in_place(s, &ws.buf_1, latent_spatial, mid_ch)?;
        self.conv2d(
            s,
            blas,
            &self.conv_out,
            (&ws.buf_1, &ws.im2col, &ws.buf_0),
            Conv2dGeom {
                h: latent_side,
                w: latent_side,
                c_in: mid_ch,
                c_out: quant_ch,
                kh: 3,
                kw: 3,
                sh: 1,
                sw: 1,
                pad: [1, 1, 1, 1],
            },
            sp,
        )?;
        self.conv2d(
            s,
            blas,
            &self.quant_conv,
            (&ws.buf_0, &ws.im2col, &ws.buf_1),
            Conv2dGeom {
                h: latent_side,
                w: latent_side,
                c_in: quant_ch,
                c_out: quant_ch,
                kh: 1,
                kw: 1,
                sh: 1,
                sw: 1,
                pad: [0, 0, 0, 0],
            },
            sp,
        )?;
        let prefix_elems = self.prefix_rows * bn_ch;
        s.memcpy_htod(
            &vec![0.0f32; prefix_elems],
            &mut ws.feature2.slice_mut(0..prefix_elems),
        )?;
        let (feat2_ptr, _feat2_guard) = ws.feature2.device_ptr(s);
        let feat2_patch_ptr = feat2_ptr + (prefix_elems * size_of::<f32>()) as u64;
        let count = u64::from(latent_ch) * u64::from(latent_spatial);
        unsafe {
            s.launch_builder(&self.diag_gauss_bn_fn)
                .arg(&ws.buf_1)
                .arg(&ws.noise_nchw)
                .arg(&self.bn_mean)
                .arg(&self.bn_var)
                .arg(&ws.moments_nchw)
                .arg(&feat2_patch_ptr)
                .arg(&latent_ch)
                .arg(&latent_side)
                .arg(&latent_side)
                .arg(&self.cfg.batch_norm_eps)
                .launch(grid(count.div_ceil(256).min(65535) as usize, 256))?;
        }
        let moments_f32 = s
            .clone_dtoh(&ws.moments_nchw)?
            .into_iter()
            .map(bf16::to_f32)
            .collect();
        let feature2_f32 = s.clone_dtoh(&ws.feature2)?;
        Ok(Flux2VaeOutput {
            moments_f32,
            feature2_f32,
        })
    }
}

pub struct IdolVisionEncoderRuntime {
    pub dinov3: DinoV3Encoder,
    pub dinov3_ws: DinoV3Workspace,
    pub vae: Flux2VaeEncoder,
    pub vae_ws: Flux2VaeWorkspace,
}

impl IdolVisionEncoderRuntime {
    pub fn load(
        device: &Device,
        blas: &mut Blas,
        common: CommonOpsRef,
        dinov3_dir: &std::path::Path,
        vae_dir: &std::path::Path,
        reserve_bytes: usize,
    ) -> Result<Self> {
        let dinov3_cfg = DinoV3Config::from_dir(dinov3_dir)?;
        let vae_cfg = Flux2VaeConfig::from_dir(vae_dir)?;
        let prefix_rows = dinov3_cfg.prefix_rows();
        ensure!(
            dinov3_cfg.patch_rows(vae_cfg.sample_size)? == vae_cfg.patch_rows(),
            "DINOv3 patch_rows != FLUX.2 VAE patch_rows"
        );
        let dinov3_file = std::fs::File::open(dinov3_dir.join("model.safetensors"))?;
        let dinov3_mmap = unsafe { memmap2::MmapOptions::new().map(&dinov3_file)? };
        let (dinov3, dinov3_ws) = DinoV3Encoder::load(
            device,
            blas,
            common,
            dinov3_cfg,
            vae_cfg.sample_size,
            &dinov3_mmap,
        )?;
        let vae_file = std::fs::File::open(Flux2VaeConfig::safetensors_path(vae_dir))?;
        let vae_mmap = unsafe { memmap2::MmapOptions::new().map(&vae_file)? };
        let (vae, vae_ws) =
            Flux2VaeEncoder::load(device, blas, vae_cfg, prefix_rows, &vae_mmap, reserve_bytes)?;
        Ok(Self {
            dinov3,
            dinov3_ws,
            vae,
            vae_ws,
        })
    }

    pub fn encode_image(
        &mut self,
        s: &Arc<CudaStream>,
        blas: &Blas,
        chw_f32: &[f32],
        vae_noise_bf16: &[bf16],
        capture_dinov3_layers: bool,
    ) -> Result<(DinoV3Output, Flux2VaeOutput)> {
        let d_out = self.dinov3.encode(
            s,
            blas,
            &mut self.dinov3_ws,
            chw_f32,
            None,
            capture_dinov3_layers,
        )?;
        let v_out = self.vae.encode(
            s,
            blas,
            &mut self.vae_ws,
            chw_f32,
            vae_noise_bf16,
            [0, 1, 0, 1],
        )?;
        Ok((d_out, v_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_blas(
        device: &Device,
        output: &std::path::Path,
        closure: std::rc::Rc<crate::backend::cuda::closure::Closure>,
    ) -> Result<Blas> {
        #[cfg(feature = "cuda")]
        {
            let ops = crate::backend::cuda::ops::Ops::new(&device.ctx)?;
            Blas::new(device, &ops, output, closure)
        }
        #[cfg(feature = "rocm")]
        {
            let _ = (output, closure);
            Blas::new(device.stream.clone())
        }
    }

    fn q38_geometry() -> VisionGeometry {
        VisionGeometry {
            spec: VisionSpec::q38(),
            patch: PATCH,
            mlp: 4304,
            insert_scale: 1,
            insert_layer: None,
            deepstack: None,
        }
    }
    #[test]
    fn tower_budget_includes_image_padding_and_merger_output() -> Result<()> {
        let packing = PatchPacking::build(&[PatchGrid { h: 2, w: 2 }, PatchGrid { h: 2, w: 4 }])?;
        assert_eq!((packing.real_rows(), packing.padded_rows()), (12, 128));
        let packings = VisionPackings {
            tower: packing.clone(),
            down: None,
            window: None,
        };
        let (bytes, scratch) = tower_bytes(&q38_geometry(), &packings)?;
        let row_bytes = if cfg!(feature = "rocm") { 23072 } else { 18464 };
        let expected = 12 * row_bytes
            + 12 * (1152 * 5 + 3456 + 4304) * 2
            + 128 * 1152 * 2
            + 3 * 5120 * 2
            + 40
            + scratch;
        assert_eq!(bytes, expected);
        assert!(scratch >= VisionMergerOps::workspace(&VisionSpec::q38(), 12)?.bytes as usize);
        let deepstack = VisionGeometry {
            deepstack: Some([8, 16, 24]),
            ..q38_geometry()
        };
        assert_eq!(
            tower_bytes(&deepstack, &packings)?,
            (expected + 3 * 3 * 5120 * 2, scratch)
        );
        let root =
            serde_json::from_str(include_str!("../../../tests/data/gemma4-e2b-config.json"))?;
        let model = ModelSpec::gemma4(&root, &serde_json::json!({"eos_token_id":1}))?;
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let ids = [
            vec![vision.start_token],
            vec![vision.image_token; 280],
            vec![vision.end_token],
        ]
        .concat()
        .repeat(2);
        let modalities = [vec![0], vec![1; 280], vec![0]].concat().repeat(2);
        let pixels = vec![0.25; 2 * 2520 * 768];
        let grids = [ImageGrid { t: 1, h: 42, w: 60 }; 2];
        let input = MultimodalInput {
            audio: &[],
            input_ids: &ids,
            modality_ids: &modalities,
            pixel_values: &pixels,
            grids: &grids,
        };
        let single = MultimodalInput {
            audio: &[],
            input_ids: &ids[..282],
            modality_ids: &modalities[..282],
            pixel_values: &pixels[..2520 * 768],
            grids: &grids[..1],
        };
        let single_bytes = VisionRuntime::request_bytes(&model, &single, 1024)?;
        let pair_bytes = VisionRuntime::request_bytes(&model, &input, 1024)?;
        assert_eq!(
            pair_bytes - single_bytes,
            2520 * (4 * 768 + 2 * 64) + 280 * 1536 * 2 + 64
        );
        assert!(single_bytes >= 2520 * 768 * 4 + 280 * 1536 * 2 + 2520 * 768 * 2);
        assert!(VisionRuntime::request_bytes(&model, &input, 512).is_err());
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
        let io = VisionIo::new(&device, OUTPUT)?;
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
        let graph = crate::backend::cuda::moe::testing::capture(s, launch)?;
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
        let row_bytes = if cfg!(feature = "rocm") { 23072 } else { 18464 };
        assert_eq!(prefix_bytes(256, &q38_geometry()).unwrap(), 256 * row_bytes);
        assert!(prefix_bytes(0, &q38_geometry()).is_err());
        assert!(prefix_bytes(usize::MAX, &q38_geometry()).is_err());
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
        assert_eq!(splice_bounds(&features, &output, &good, OUTPUT)?, 3);
        assert!(
            splice_bounds(
                &features,
                &output,
                &SpliceRange {
                    chunk_rows: 2..5,
                    feature_rows: 2..5
                },
                OUTPUT
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
                },
                OUTPUT
            )
            .is_err()
        );
        assert!(splice_bounds(&features, &features, &good, OUTPUT).is_err());
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
        let io = Arc::new(VisionIo::new(&device, OUTPUT)?);
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
            PATCH,
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

        let tables = VisionPositions::new(
            model.vision.as_ref().unwrap(),
            model.architecture,
            &image_grids,
            rows as usize,
        )?;
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
                bias: None,
                trace_patches: false,
            },
            &mut TensorMut(dense(&hidden, s, [rows, HIDDEN])?),
            &mut TensorMut(dense(&interpolation, s, [rows, HIDDEN])?),
            HIDDEN,
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
        for width in [1536, 2048, 2560, OUTPUT] {
            let io = Arc::new(VisionIo::new(&device, width)?);
            let plan = MultimodalInput {
                audio: &[],
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &pixels,
                grids: &image_grids,
            }
            .validate(&model, 32)?;
            let feature_host: Vec<_> = (0..6 * width as usize)
                .map(|i| bf16::from_bits(0x3e00 + ((i * 37) % 509) as u16))
                .collect();
            let mut encoded = EncodedImages::new(plan, s.clone_htod(&feature_host)?, io)?;
            let sentinel = bf16::from_f32(-7.0);
            for first in [0usize, 5, 10] {
                let output = s.clone_htod(&vec![sentinel; 5 * width as usize])?;
                encoded.splice(s, first, &mut TensorMut(dense(&output, s, [5, width])?))?;
                let got = s.clone_dtoh(&output)?;
                for row in 0..5 {
                    let absolute = first + row;
                    let begin = row * width as usize;
                    let expected = if (2..8).contains(&absolute) {
                        &feature_host[(absolute - 2) * width as usize..][..width as usize]
                    } else {
                        &vec![sentinel; width as usize]
                    };
                    ensure!(
                        got[begin..begin + width as usize] == *expected,
                        "splice width={width} corrupted image/text row={absolute}"
                    );
                }
            }
            let mut shifted = feature_host.clone();
            shifted.rotate_left(width as usize);
            s.memcpy_htod(&shifted, &mut encoded.features)?;
            let output = s.clone_htod(&vec![sentinel; 10 * width as usize])?;
            encoded.splice(s, 0, &mut TensorMut(dense(&output, s, [10, width])?))?;
            let got = s.clone_dtoh(&output)?;
            let mismatches = got[2 * width as usize..8 * width as usize]
                .iter()
                .zip(&feature_host)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                mismatches > 0,
                "splice wrong-source-row control width={width} was not detected"
            );
            eprintln!(
                "VISION_SPLICE width={width} chunks=3 bitwise=true wrong_source_row_mismatches={mismatches} PASS"
            );
        }
        eprintln!(
            "VISION_IO cast/bilinear_fp32/cast_before_add/chunk_splice/text_preservation PASS"
        );
        Ok(())
    }

    struct MergedInput {
        ids: Vec<u32>,
        modalities: Vec<u8>,
        /// Per-image packs `[channels, side, patches*side]`, merged per (c, ky) stripe before use.
        pixel_packs: Vec<(usize, Vec<f32>)>,
        grids: Vec<ImageGrid>,
    }

    /// Interleaves NaViT packs along the patch axis: for every (channel, ky) row, each pack's columns
    /// follow in order, matching the official processor's multi-image layout.
    fn merge_pixel_packs(packs: &[(usize, Vec<f32>)], channels: usize, side: usize) -> Vec<f32> {
        let total: usize = packs.iter().map(|(patches, _)| *patches).sum();
        let mut merged = vec![0f32; channels * side * total * side];
        for c in 0..channels {
            for ky in 0..side {
                let row = c * side + ky;
                let mut column = 0usize;
                for (patches, pixels) in packs {
                    let source = row * patches * side;
                    let span = patches * side;
                    merged[row * total * side + column..][..span]
                        .copy_from_slice(&pixels[source..source + span]);
                    column += span;
                }
            }
        }
        merged
    }

    fn tensor_values<T, const N: usize>(
        source: &safetensors::SafeTensors<'_>,
        name: &str,
        dtype: safetensors::Dtype,
        convert: impl Fn([u8; N]) -> T,
    ) -> Result<Vec<T>> {
        let tensor = source.tensor(name)?;
        ensure!(tensor.dtype() == dtype, "{name} must be {dtype:?}");
        Ok(tensor
            .data()
            .as_chunks::<N>()
            .0
            .iter()
            .map(|v| convert(*v))
            .collect())
    }

    /// Metrics of one candidate/reference tap: max and mean absolute difference, p99, and cosine.
    fn tap_metrics(candidate: &[f32], reference: &[f32]) -> Result<serde_json::Value> {
        ensure!(
            candidate.len() == reference.len() && !candidate.is_empty(),
            "tap extent mismatch"
        );
        ensure!(
            candidate.iter().all(|v| v.is_finite()) && reference.iter().all(|v| v.is_finite()),
            "non-finite tap value"
        );
        let mut worst = 0f32;
        let mut sum = 0f64;
        let mut dot = 0f64;
        let mut norm_c = 0f64;
        let mut norm_r = 0f64;
        let mut diffs = Vec::with_capacity(candidate.len());
        for (&c, &r) in candidate.iter().zip(reference) {
            let diff = (c - r).abs();
            worst = worst.max(diff);
            sum += f64::from(diff);
            diffs.push(diff);
            dot += f64::from(c) * f64::from(r);
            norm_c += f64::from(c) * f64::from(c);
            norm_r += f64::from(r) * f64::from(r);
        }
        diffs.sort_by(f32::total_cmp);
        let p99 = diffs[(diffs.len() - 1) * 99 / 100];
        Ok(serde_json::json!({
            "elements": candidate.len(),
            "max_abs": worst,
            "mean_abs": sum / candidate.len() as f64,
            "p99_abs": p99,
            "cosine": dot / (norm_c.sqrt() * norm_r.sqrt()),
        }))
    }

    /// Gate of the MiniCPM tower against the T0 fixtures.
    #[test]
    #[ignore = "requires FF_MINICPM_VISION_WINDOW=granted, FF_MODEL, FF_MINICPM_REFERENCE_DIR, \
                FF_MINICPM_VISION_OUTPUT_DIR, FF_MINICPM_TOLERANCES and FF_MINICPM_VISION_REFERENCE"]
    fn minicpm_vision_tower_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
        use std::{collections::BTreeMap, path::PathBuf, rc::Rc};
        ensure!(
            std::env::var("FF_MINICPM_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let model_dir =
            PathBuf::from(std::env::var_os("FF_MODEL").context("model directory missing")?);
        let fixture_reference = PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("reference fixture directory missing")?,
        );
        let output = PathBuf::from(
            std::env::var_os("FF_MINICPM_VISION_OUTPUT_DIR")
                .context("vision output directory missing")?,
        );
        #[cfg(feature = "cuda")]
        let tolerance_path = PathBuf::from(
            std::env::var_os("FF_MINICPM_TOLERANCES").context("tolerance file missing")?,
        );
        let reference_dir = PathBuf::from(
            std::env::var_os("FF_MINICPM_VISION_REFERENCE")
                .context("reference directory missing")?,
        );
        #[cfg(feature = "cuda")]
        let tolerances: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&tolerance_path)?)?;
        #[cfg(feature = "rocm")]
        let tolerances = serde_json::from_str::<serde_json::Value>(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/minicpm_reference.json"
        )))?["vision_tower"]
            .clone();
        std::fs::create_dir_all(&output)?;
        let model = ModelSpec::read(&model_dir)?;
        ensure!(
            model.architecture == Architecture::MiniCpmV47,
            "MiniCPM model directory required"
        );
        let device = Device::new(0)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let mut blas = test_blas(
            &device,
            &output,
            Rc::new(Closure::new(false, "minicpm vision gate".into())),
        )?;
        let mut catalog = WeightCatalog::load_with_policy(
            &model_dir,
            &model,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        ensure!(
            catalog.bytes == 1_105_972_064,
            "vision-only weight ledger mismatch"
        );
        let runtime = LayerNormTower::from_catalog(&model, &mut catalog, &device, common)?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        let mut standalone: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut verdict: BTreeMap<String, (f64, f64, bool)> = BTreeMap::new();
        let mut merged_input: Option<MergedInput> = None;
        let mut merged_expected: Vec<(&str, usize)> = Vec::new();
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        for (case, pixels_key, sizes_key, video) in [
            ("image_hockey_16x", "pixel_values", "target_sizes", false),
            ("image_boat_16x", "pixel_values", "target_sizes", false),
            (
                "video_football_16x",
                "pixel_values_videos",
                "target_sizes_videos",
                true,
            ),
            (
                "video_control_same_grid_16x",
                "pixel_values_videos",
                "target_sizes_videos",
                true,
            ),
            (
                "video_control_clip2_16x",
                "pixel_values_videos",
                "target_sizes_videos",
                true,
            ),
        ] {
            let path = fixture_reference.join(format!("{case}.safetensors"));
            let file =
                std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
            let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
            let source = safetensors::SafeTensors::deserialize(&map)?;
            let sizes = tensor_values(
                &source,
                sizes_key,
                safetensors::Dtype::I32,
                i32::from_le_bytes,
            )?;
            ensure!(sizes.len().is_multiple_of(2), "target_sizes extent");
            let grids: Vec<ImageGrid> = sizes
                .chunks(2)
                .map(|pair| -> Result<ImageGrid> {
                    Ok(ImageGrid {
                        t: 1,
                        h: u32::try_from(pair[0])?,
                        w: u32::try_from(pair[1])?,
                    })
                })
                .collect::<Result<_>>()?;
            let ids: Vec<u32> = tensor_values(
                &source,
                "input_ids",
                safetensors::Dtype::I64,
                i64::from_le_bytes,
            )?
            .into_iter()
            .map(|id| Ok(u32::try_from(id)?))
            .collect::<Result<_>>()?;
            let modalities: Vec<u8> = tensor_values(
                &source,
                "mm_token_type_ids",
                safetensors::Dtype::I64,
                i64::from_le_bytes,
            )?
            .into_iter()
            .map(|m| Ok(u8::try_from(m)?))
            .collect::<Result<_>>()?;
            let pixel_tensor = source.tensor(pixels_key)?;
            ensure!(
                pixel_tensor.dtype() == safetensors::Dtype::F32,
                "fixture pixel dtype"
            );
            let pixels: Vec<f32> = pixel_tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let input = MultimodalInput {
                audio: &[],
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &pixels,
                grids: &grids,
            };
            let request = runtime.prepare(
                &device,
                &mut blas,
                &input,
                ids.len(),
                device.ctx.mem_get_info()?.1 / 16,
            )?;
            let feature_rows = request.prefix.positions.feature_rows();
            let mut trace = crate::trace::Trace::default();
            trace.set_device(&device.info.name)?;
            let images = runtime.encode(s, &blas, request, Some(&mut trace))?;
            ensure!(
                images.features.len()
                    == feature_rows * model.vision.as_ref().unwrap().output as usize,
                "merger output extent"
            );
            trace.save(&output.join(format!("{case}-tower.safetensors")))?;
            let ref_path = reference_dir.join(format!("ref-{case}.safetensors"));
            let ref_file = std::fs::File::open(&ref_path)
                .with_context(|| format!("open {}", ref_path.display()))?;
            let ref_map = unsafe { memmap2::MmapOptions::new().map(&ref_file)? };
            let reference = safetensors::SafeTensors::deserialize(&ref_map)?;
            let insert_tap = format!(
                "vision_layer_{:02}_out",
                model
                    .vision
                    .as_ref()
                    .and_then(|v| v.minicpm.as_ref())
                    .context("MiniCPM vision configuration missing")?
                    .insert_layer
            );
            let mut taps = vec![
                "vision_patch_embeddings",
                insert_tap.as_str(),
                "vit_merger_out",
                "vision_tower_out",
                "merger_out",
            ];
            if video {
                taps.push("vit_merger_in");
            }
            for tap in taps {
                let trace_tap = if tap == "vision_patch_embeddings" {
                    "after_pos_add"
                } else if tap == "vit_merger_in" {
                    insert_tap.as_str()
                } else {
                    tap
                };
                let candidate = trace.values(trace_tap)?;
                let reference_values =
                    tensor_values(&reference, tap, safetensors::Dtype::BF16, |v| {
                        bf16::from_bits(u16::from_le_bytes(v)).to_f32()
                    })?;
                let metrics = tap_metrics(&candidate, &reference_values)?;
                let k = tolerances[case][tap]["k"]
                    .as_f64()
                    .with_context(|| format!("missing k {case}/{tap}"))?;
                let rms = tolerances[case][tap]["rms"]
                    .as_f64()
                    .with_context(|| format!("missing rms {case}/{tap}"))?;
                let ratio =
                    crate::verify::half_ulp_ratio(&candidate, &reference_values, rms as f32);
                let fixture_values = tensor_values(&source, tap, safetensors::Dtype::BF16, |v| {
                    bf16::from_bits(u16::from_le_bytes(v)).to_f32()
                })?;
                let fixture_metrics = tap_metrics(&candidate, &fixture_values)?;
                let pass = ratio <= k;
                eprintln!(
                    "{}",
                    serde_json::json!({"minicpm_vision_gate": {"case": case, "tap": tap,
                        "max_abs": metrics["max_abs"], "max_ulp_ratio": ratio, "k": k, "rms": rms,
                        "cosine": metrics["cosine"],
                        "max_abs_over_rms": metrics["max_abs"].as_f64().context("max_abs")? / rms,
                        "pass": pass, "fixture_secondary": fixture_metrics}})
                );
                verdict.insert(
                    format!("{case}/{tap}"),
                    (metrics["max_abs"].as_f64().context("max_abs")?, ratio, pass),
                );
                standalone.insert(format!("{case}/{tap}"), candidate);
            }
            let patches: usize = grids.iter().map(|g| (g.h * g.w) as usize).sum();
            ensure!(
                pixels.len()
                    == vision.channels as usize
                        * vision.patch as usize
                        * patches
                        * vision.patch as usize,
                "fixture pixel extent"
            );
            if !video {
                match merged_input.as_mut() {
                    Some(merged) => {
                        merged.ids.extend_from_slice(&ids);
                        merged.modalities.extend_from_slice(&modalities);
                        merged.pixel_packs.push((patches, pixels));
                        merged.grids.extend_from_slice(&grids);
                    }
                    None => {
                        merged_input = Some(MergedInput {
                            ids,
                            modalities,
                            pixel_packs: vec![(patches, pixels)],
                            grids,
                        });
                    }
                }
                merged_expected.push((case, feature_rows));
            }
        }
        // Two-image reach control: each packed merger block is scored against its standalone run at
        // `merger_out_packed`; one-row-shifted and swapped-order reads must exceed that bound and the
        // two blocks must not be byte-identical.
        if let Some(merged) = merged_input {
            ensure!(
                merged_expected.len() == 2,
                "two-image reach control needs exactly two image cases, got {}",
                merged_expected.len()
            );
            let vision_config = model
                .vision
                .as_ref()
                .context("vision configuration missing")?;
            let merged_pixels = merge_pixel_packs(
                &merged.pixel_packs,
                vision_config.channels as usize,
                vision_config.patch as usize,
            );
            let input = MultimodalInput {
                audio: &[],
                input_ids: &merged.ids,
                modality_ids: &merged.modalities,
                pixel_values: &merged_pixels,
                grids: &merged.grids,
            };
            let request = runtime.prepare(
                &device,
                &mut blas,
                &input,
                merged.ids.len(),
                device.ctx.mem_get_info()?.1 / 16,
            )?;
            let mut trace = crate::trace::Trace::default();
            trace.set_device(&device.info.name)?;
            let _images = runtime.encode(s, &blas, request, Some(&mut trace))?;
            trace.save(&output.join("merged-tower.safetensors"))?;
            let merged = trace.values("merger_out")?;
            let output_width = model.vision.as_ref().unwrap().output as usize;
            let first = merged_expected[0].1 * output_width;
            let second = merged_expected[1].1 * output_width;
            ensure!(
                merged.len() == first + second,
                "merged merger output extent"
            );
            let mut packed_bounds = Vec::new();
            for (index, (case, rows)) in merged_expected.iter().enumerate() {
                let start = if index == 0 { 0 } else { first };
                let extent = rows * output_width;
                let block = &merged[start..start + extent];
                let reference = standalone
                    .get(&format!("{case}/merger_out"))
                    .with_context(|| format!("standalone merger missing for {case}"))?;
                let metrics = tap_metrics(block, reference)?;
                let bound = &tolerances[case]["merger_out_packed"];
                let k = bound["k"]
                    .as_f64()
                    .with_context(|| format!("missing k {case}/merger_out_packed"))?;
                let rms = bound["rms"]
                    .as_f64()
                    .with_context(|| format!("missing rms {case}/merger_out_packed"))?;
                let ratio = crate::verify::half_ulp_ratio(block, reference, rms as f32);
                let shifted = if index == 0 {
                    start + output_width
                } else {
                    start - output_width
                };
                let shifted_ratio = crate::verify::half_ulp_ratio(
                    &merged[shifted..shifted + extent],
                    reference,
                    rms as f32,
                );
                let pass = ratio <= k;
                eprintln!(
                    "{}",
                    serde_json::json!({"minicpm_vision_control": {"case": case,
                        "tap": "merger_out_packed", "max_ulp_ratio": ratio, "k": k, "rms": rms,
                        "shifted_ratio": shifted_ratio, "shifted_margin": shifted_ratio / k,
                        "pass": pass, "metrics": metrics}})
                );
                ensure!(
                    shifted_ratio > k,
                    "{case} one-row-shifted block is within the packed bound: {shifted_ratio} <= {k}"
                );
                verdict.insert(
                    format!("{case}/merger_out_packed"),
                    (metrics["max_abs"].as_f64().context("max_abs")?, ratio, pass),
                );
                packed_bounds.push((k, rms));
            }
            let (case, rows) = merged_expected[0];
            let extent = rows * output_width;
            ensure!(
                extent <= second,
                "swapped-order control needs the second block to cover the first"
            );
            let (k, rms) = packed_bounds[0];
            let swapped_ratio = crate::verify::half_ulp_ratio(
                &merged[first..first + extent],
                standalone
                    .get(&format!("{case}/merger_out"))
                    .with_context(|| format!("standalone merger missing for {case}"))?,
                rms as f32,
            );
            eprintln!(
                "{}",
                serde_json::json!({"minicpm_vision_control": {"case": case,
                    "tap": "merger_out_packed_swapped", "swapped_ratio": swapped_ratio, "k": k,
                    "swapped_margin": swapped_ratio / k}})
            );
            ensure!(
                swapped_ratio > k,
                "{case} swapped-order block is within the packed bound: {swapped_ratio} <= {k}"
            );
            ensure!(
                merged[..first] != merged[first..],
                "the two images produced identical merger blocks"
            );
        }
        let tap_reports: Vec<_> = verdict
            .iter()
            .map(|(key, (max_abs, ratio, pass))| {
                serde_json::json!({"tap": key, "max_abs": max_abs, "max_ulp_ratio": ratio, "pass": pass})
            })
            .collect();
        let defects: Vec<&String> = verdict
            .iter()
            .filter(|(_, (_, _, pass))| !pass)
            .map(|(key, _)| key)
            .collect();
        eprintln!(
            "{}",
            serde_json::json!({"minicpm_vision_verdict": {"taps": tap_reports, "defects": defects}})
        );
        ensure!(
            defects.is_empty(),
            "vision gate exceeds the frozen bound at {defects:?}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_Q38_VISION_WINDOW=granted, FF_Q38_MODEL, FF_Q38_MERGER_INPUT_DIR and FF_Q38_VISION_OUTPUT_DIR"]
    fn vision_merger_stages_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
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
        let mut blas = test_blas(
            &device,
            &output,
            Rc::new(Closure::new(false, "merger stages".into())),
        )?;
        let mut catalog = WeightCatalog::load_with_policy(
            &model_dir,
            &model,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        let geometry = VisionGeometry::from_model(&model)?;
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let weights = VisionWeights::take(&mut catalog, geometry, vision, model.architecture)?;
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

    fn patch_position_bits(
        device: &Device,
        blas: &mut Blas,
        patch: &BiasedWeight,
        host: &[bf16],
        expected_patch: &[bf16],
        cell: (u32, bool, bool),
    ) -> Result<usize> {
        let (rows, bilinear, zero_position) = cell;
        let s = &device.stream;
        let width = patch.shape[0];
        let table_rows = if bilinear { 2304 } else { 4900 };
        let position_host: Vec<_> = (0..table_rows * width)
            .map(|i| {
                if zero_position {
                    bf16::ZERO
                } else {
                    bf16::from_f32((i % 17) as f32 / 32.0 - 0.25)
                }
            })
            .collect();
        let position = s.clone_htod(&position_host)?;
        let (indices, coefficients) = if bilinear {
            let (h, w) = match rows {
                1 => (2, 2),
                37 => (2, 20),
                1120 => (28, 40),
                7104 => (96, 74),
                _ => bail!("patch position fixture row extent unsupported"),
            };
            let model = ModelSpec::qwen(
                &serde_json::from_str(include_str!("../../../tests/data/q38-config.json"))?,
                &serde_json::Value::Null,
            )?;
            let tables = VisionPositions::new(
                model.vision.as_ref().context("vision fixture missing")?,
                model.architecture,
                &[ImageGrid { t: 1, h, w }],
                (h * w) as usize,
            )?;
            let indices = tables
                .bilinear_indices
                .iter()
                .flat_map(|corner| corner[..rows as usize].iter().copied())
                .collect::<Vec<_>>();
            let coefficients = tables
                .bilinear_weights
                .iter()
                .flat_map(|corner| corner[..rows as usize].iter().copied())
                .collect::<Vec<_>>();
            (indices, coefficients)
        } else {
            (
                (0..rows)
                    .map(|row| ((row * 13) % table_rows) as i32)
                    .collect(),
                Vec::new(),
            )
        };
        let expected: Vec<_> = expected_patch
            .iter()
            .enumerate()
            .map(|(i, patch)| {
                let (row, column) = (i / width as usize, i % width as usize);
                let pos = if bilinear {
                    let mut sum = 0.0f32;
                    for corner in 0..4 {
                        let offset = corner * rows as usize + row;
                        sum += position_host[indices[offset] as usize * width as usize + column]
                            .to_f32()
                            * coefficients[offset];
                    }
                    bf16::from_f32(sum)
                } else {
                    position_host[indices[row] as usize * width as usize + column]
                };
                bf16::from_f32(patch.to_f32() + pos.to_f32())
            })
            .collect();
        let x = s.clone_htod(host)?;
        let y = s.alloc_zeros::<bf16>((rows * width) as usize)?;
        let hidden = s.alloc_zeros::<bf16>((rows * width) as usize)?;
        let indices = s.clone_htod(&indices)?;
        let tables = if bilinear {
            #[cfg(feature = "cuda")]
            blas.prepare_fixed(rows, width, patch.shape[1])?;
            #[cfg(feature = "rocm")]
            patch.prepare(s, blas, rows)?;
            Some(BilinearTables {
                coefficients: s.clone_htod(&coefficients)?,
                values: s.alloc_zeros((rows * width) as usize)?,
            })
        } else {
            patch.prepare(s, blas, rows)?;
            None
        };
        let io = VisionIo::new(device, OUTPUT)?;
        device.finish_upload()?;
        let execute = |trace_patches| {
            io.project_position(
                s,
                blas,
                PatchPositionBindings {
                    patch,
                    position: &position,
                    indices: &indices,
                    bilinear: tables.as_ref(),
                    packed: &x,
                    patch_output: &y,
                    hidden: &hidden,
                    trace_patches,
                },
            )
        };
        let mut defects = 0;
        for graph in [false, true] {
            for trace in [false, true] {
                let captured = if graph {
                    Some(crate::backend::cuda::moe::testing::capture(s, || {
                        execute(trace)
                    })?)
                } else {
                    None
                };
                for _ in 0..2 {
                    if let Some(graph) = &captured {
                        graph.launch()?;
                    } else {
                        execute(trace)?;
                    }
                    let actual = s.clone_dtoh(&hidden)?;
                    ensure!(
                        actual.iter().all(|v| v.is_finite()),
                        "nonfinite positioned patch"
                    );
                    let wrong = actual
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let pre_position = if trace || !bilinear {
                        s.clone_dtoh(&y)?
                            .iter()
                            .zip(expected_patch)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count()
                    } else {
                        0
                    };
                    defects += wrong + pre_position;
                    println!(
                        "{} VISION_PATCH_POSITION rows={rows} bilinear={bilinear} zero_position={zero_position} graph={graph} trace={trace} bit_mismatches={wrong} pre_position_mismatches={pre_position}",
                        if wrong + pre_position == 0 {
                            "PASS"
                        } else {
                            "DEFECT"
                        }
                    );
                }
            }
        }
        Ok(defects)
    }

    #[test]
    #[ignore = "requires FF_Q38_VISION_WINDOW=granted and FF_Q38_VISION_OUTPUT_DIR"]
    fn vision_merger_bias_rounding_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
        use crate::backend::gpu::DevicePtr;
        use std::{path::PathBuf, rc::Rc, time::Instant};
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
        let closure = Rc::new(Closure::new(false, "merger bias rounding".into()));
        let mut blas = test_blas(&device, &output, closure.clone())?;
        for rows in [1u32, 37, 1120, 7104] {
            blas.prepare_fixed(rows, HIDDEN, PATCH)?;
        }
        closure.freeze_fixed()?;
        closure.ready(Instant::now())?;
        let mut patch_defects = 0;
        let mut position_defects = 0;
        {
            use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
            use std::collections::BTreeMap;

            for (prefix, stored_shape) in [
                (
                    "model.vision_tower.embeddings.patch_embedding",
                    vec![1152usize, 3, 14, 14],
                ),
                (
                    "model.visual.patch_embed.proj",
                    vec![1152usize, 3, 2, 16, 16],
                ),
            ] {
                let width = stored_shape[0] as u32;
                let input = stored_shape[1..].iter().product::<usize>() as u32;
                let half_ulp = bf16::from_f32(bf16::EPSILON.to_f32() / 2.0);
                let mut weight = vec![bf16::ZERO; (width * input) as usize];
                let mut bias = Vec::with_capacity(width as usize);
                for (column, values) in weight.chunks_exact_mut(input as usize).enumerate() {
                    let sign = if column < width as usize / 2 {
                        1.0
                    } else {
                        -1.0
                    };
                    values[0] = bf16::from_f32(sign);
                    values[1] = bf16::from_f32(sign * half_ulp.to_f32());
                    bias.push(values[1]);
                }
                let mut tensors = BTreeMap::new();
                let mut entries = BTreeMap::new();
                let mut bytes = 0;
                for (suffix, shape, values) in [
                    ("weight", stored_shape.to_vec(), weight),
                    ("bias", vec![width as usize], bias),
                ] {
                    let name = format!("{prefix}.{suffix}");
                    let size = values.len() * size_of::<bf16>();
                    tensors.insert(name.clone(), StoredTensor::Bf16(s.clone_htod(&values)?));
                    entries.insert(
                        name,
                        TensorEntry {
                            dtype: safetensors::Dtype::BF16,
                            shape,
                            shard: "patch-fixture.safetensors".into(),
                            bytes: size,
                            load: true,
                            data_start: 0,
                            data_offsets: [0, size as u64],
                            scope: TensorScope::Vision,
                            layer: None,
                            role: suffix.into(),
                            scale_name: None,
                        },
                    );
                    bytes += size;
                }
                let mut catalog = WeightCatalog {
                    #[cfg(feature = "cuda")]
                    dense: None,
                    linear_format: crate::model_spec::LinearFormat::Bf16,
                    tensors,
                    manifest: TensorManifest {
                        entries,
                        resident_bytes: bytes,
                    },
                    bytes,
                    #[cfg(feature = "cuda")]
                    upload_content_sha256: None,
                    upload_hashed_bytes: None,
                    load_wall_ms: 0.0,
                };
                let patch =
                    BiasedWeight::patch(&mut catalog, prefix, [width, input], &stored_shape)?;
                let bias_pointer = patch.bias.device_ptr(s).0;
                for rows in [1u32, 37, 1120, 7104] {
                    let mut host = vec![bf16::ZERO; (rows * input) as usize];
                    for row in host.chunks_exact_mut(input as usize) {
                        row[..2].fill(bf16::ONE);
                    }
                    let x = s.clone_htod(&host)?;
                    let y = s.alloc_zeros::<bf16>((rows * width) as usize)?;
                    if rows == 1 {
                        let zero = s.alloc_zeros::<bf16>(width as usize)?;
                        blas.prepare_biased(
                            rows,
                            width,
                            input,
                            dense(&zero, s, [width])?,
                            LinearBiasMode::AfterOutputRound,
                        )?;
                        blas.linear_biased(
                            dense(&patch.weight, s, [width, input])?,
                            dense(&zero, s, [width])?,
                            dense(&x, s, [rows, input])?,
                            &mut TensorMut(dense(&y, s, [rows, width])?),
                            LinearBiasMode::AfterOutputRound,
                        )?;
                        let actual = s.clone_dtoh(&y)?;
                        let wrong = actual
                            .iter()
                            .enumerate()
                            .filter(|(i, value)| {
                                value.to_bits()
                                    != bf16::from_f32(if *i < width as usize / 2 {
                                        1.0
                                    } else {
                                        -1.0
                                    })
                                    .to_bits()
                            })
                            .count();
                        patch_defects += usize::from(wrong != 0);
                        println!(
                            "{} VISION_PATCH_ZERO_BIAS input={input} bit_mismatches={wrong} positive_bits={} negative_bits={}",
                            if wrong == 0 { "PASS" } else { "DEFECT" },
                            actual[0].to_bits(),
                            actual[width as usize / 2].to_bits()
                        );
                    }
                    if rows == 1 {
                        ensure!(
                            patch
                                .run(
                                    s,
                                    &blas,
                                    dense(&x, s, [rows, input])?,
                                    &mut TensorMut(dense(&y, s, [rows, width])?),
                                )
                                .is_err(),
                            "unprepared patch projection executed after Ready"
                        );
                    }
                    for mode in [
                        LinearBiasMode::AfterOutputRound,
                        LinearBiasMode::BeforeOutputRound,
                        LinearBiasMode::AfterOutputRound,
                    ] {
                        match mode {
                            LinearBiasMode::AfterOutputRound => {
                                patch.prepare(s, &mut blas, rows)?
                            }
                            LinearBiasMode::BeforeOutputRound => blas.prepare_biased(
                                rows,
                                width,
                                input,
                                dense(&patch.bias, s, [width])?,
                                mode,
                            )?,
                        }
                        match mode {
                            LinearBiasMode::AfterOutputRound => patch.run(
                                s,
                                &blas,
                                dense(&x, s, [rows, input])?,
                                &mut TensorMut(dense(&y, s, [rows, width])?),
                            )?,
                            LinearBiasMode::BeforeOutputRound => blas.linear_biased(
                                dense(&patch.weight, s, [width, input])?,
                                dense(&patch.bias, s, [width])?,
                                dense(&x, s, [rows, input])?,
                                &mut TensorMut(dense(&y, s, [rows, width])?),
                                mode,
                            )?,
                        }
                        let magnitude = match mode {
                            LinearBiasMode::AfterOutputRound => 1.0,
                            LinearBiasMode::BeforeOutputRound => 1.0 + 2.0 * half_ulp.to_f32(),
                        };
                        let actual = s.clone_dtoh(&y)?;
                        let wrong = actual
                            .iter()
                            .enumerate()
                            .filter(|(index, value)| {
                                let sign = if index % (width as usize) < width as usize / 2 {
                                    1.0
                                } else {
                                    -1.0
                                };
                                value.to_bits() != bf16::from_f32(sign * magnitude).to_bits()
                            })
                            .count();
                        ensure!(
                            patch.bias.device_ptr(s).0 == bias_pointer,
                            "patch preparation moved the bias pointer"
                        );
                        patch_defects += usize::from(wrong != 0);
                        println!(
                            "{} vision_patch_bias_rounding projection={prefix} stored_shape={stored_shape:?} rows={rows} width={width} input={input} mode={mode:?} bit_mismatches={wrong} positive_bits={} negative_bits={}",
                            if wrong == 0 { "PASS" } else { "DEFECT" },
                            actual[0].to_bits(),
                            actual[width as usize / 2].to_bits()
                        );
                    }
                    let expected: Vec<_> = (0..rows as usize * width as usize)
                        .map(|i| {
                            bf16::from_f32(if i % (width as usize) < width as usize / 2 {
                                1.0
                            } else {
                                -1.0
                            })
                        })
                        .collect();
                    position_defects += patch_position_bits(
                        &device,
                        &mut blas,
                        &patch,
                        &host,
                        &expected,
                        (rows, input == PATCH, true),
                    )?;
                }
            }
        }
        println!(
            "VISION_PATCH_BIAS_SUMMARY layouts=2 row_shapes=4 request_modes=3 defects={patch_defects}"
        );
        {
            let (input, width) = (1536u32, 1152u32);
            let activation = |phase: usize, k: usize| ((k * 5 + phase * 7) % 23) as i32 - 11;
            let coefficient = |phase: usize, k: usize| ((k * 11 + phase * 3) % 31) as i32 - 15;
            ensure!(
                input * 11 * 15 < 1 << 24,
                "full-K dot exceeds FP32 exact integers"
            );
            let mut expected = vec![0u16; 23 * 31];
            let mut truncated_differences = 0;
            for rp in 0..23 {
                for cp in 0..31 {
                    let dot: i32 = (0..input as usize)
                        .map(|k| activation(rp, k) * coefficient(cp, k))
                        .sum();
                    let short: i32 = (0..2).map(|k| activation(rp, k) * coefficient(cp, k)).sum();
                    let bias = (cp as f32 - 15.0) / 8.0;
                    let rounded = |value| {
                        bf16::from_f32(bf16::from_f32(value as f32).to_f32() + bias).to_bits()
                    };
                    expected[rp * 31 + cp] = rounded(dot);
                    truncated_differences += usize::from(rounded(dot) != rounded(short));
                }
            }
            ensure!(
                truncated_differences > 0,
                "truncated-K patch control escaped"
            );
            println!(
                "PASS VISION_PATCH_FULL_K_CONTROL truncated_k=2 differing_phases={truncated_differences}"
            );
            let weights: Vec<_> =
                (0..width as usize * input as usize)
                    .map(|i| {
                        bf16::from_f32(
                            coefficient((i / input as usize) % 31, i % input as usize) as f32
                        )
                    })
                    .collect();
            let weight = s.clone_htod(&weights)?;
            let bias = s.clone_htod(
                &(0..width as usize)
                    .map(|c| bf16::from_f32(((c % 31) as f32 - 15.0) / 8.0))
                    .collect::<Vec<_>>(),
            )?;
            let patch = BiasedWeight {
                weight,
                bias,
                shape: [width, input],
                bias_mode: LinearBiasMode::AfterOutputRound,
            };
            for rows in [1u32, 37, 1120, 7104] {
                let host: Vec<_> = (0..rows as usize * input as usize)
                    .map(|i| {
                        bf16::from_f32(
                            activation((i / input as usize) % 23, i % input as usize) as f32
                        )
                    })
                    .collect();
                let expected: Vec<_> = (0..rows as usize * width as usize)
                    .map(|i| {
                        bf16::from_bits(
                            expected[(i / width as usize % 23) * 31 + i % width as usize % 31],
                        )
                    })
                    .collect();
                let wrong = patch_position_bits(
                    &device,
                    &mut blas,
                    &patch,
                    &host,
                    &expected,
                    (rows, true, false),
                )?;
                println!(
                    "{} VISION_PATCH_FULL_K rows={rows} input={input} width={width} periods=23,31 bit_mismatches={wrong}",
                    if wrong == 0 { "PASS" } else { "DEFECT" }
                );
                patch_defects += usize::from(wrong != 0);
            }
        }
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
        closure.assert_ready()?;
        println!("VISION_PATCH_POSITION_SUMMARY bit_mismatches={position_defects}");
        ensure!(
            patch_defects == 0 && position_defects == 0,
            "VISION_PATCH_BIAS DEFECT cases={patch_defects} positioned={position_defects}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires exclusive sm89 vision window and the native Qwen checkpoint"]
    fn vision_patch_operation_timing_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
        use std::{path::PathBuf, rc::Rc, time::Instant};
        ensure!(
            std::env::var("FF_Q38_VISION_WINDOW").as_deref() == Ok("granted"),
            "vision GPU window missing"
        );
        let directory =
            PathBuf::from(std::env::var_os("FF_Q38_MODEL").context("model directory missing")?);
        let output = PathBuf::from(
            std::env::var_os("FF_Q38_VISION_OUTPUT_DIR")
                .context("vision output directory missing")?,
        );
        std::fs::create_dir_all(&output)?;
        let model = ModelSpec::read(&directory)?;
        ensure!(
            model.architecture == Architecture::Qwen35Text,
            "patch timing requires the Qwen checkpoint"
        );
        let geometry = VisionGeometry::from_model(&model)?;
        let vision = model
            .vision
            .as_ref()
            .context("vision configuration missing")?;
        let stored_shape = [
            vision.hidden as usize,
            vision.channels as usize,
            vision.temporal_patch as usize,
            vision.patch as usize,
            vision.patch as usize,
        ];
        ensure!(
            stored_shape == [1152, 3, 2, 16, 16] && geometry.patch == 1536,
            "patch timing checkpoint geometry mismatch"
        );
        let device = Device::new(0)?;
        ensure!(
            device.info.arch == crate::backend::setup::DeviceArchitecture::Cuda(89),
            "operator timing requires sm89"
        );
        let stream = &device.stream;
        let closure = Rc::new(Closure::new(false, "patch operation timing".into()));
        let blas = test_blas(&device, &output, closure.clone())?;
        let mut catalog = WeightCatalog::load_with_policy(
            &directory,
            &model,
            &device,
            crate::checkpoint::LoadPolicy::VisionOnly,
        )?;
        let patch = BiasedWeight::patch(
            &mut catalog,
            "model.visual.patch_embed.proj",
            [vision.hidden, geometry.patch],
            &stored_shape,
        )?;
        let position = take(
            &mut catalog,
            "model.visual.pos_embed.weight",
            &[2304, vision.hidden as usize],
        )?;
        let io = VisionIo::new(&device, geometry.output())?;
        drop(catalog);
        for rows in [1u32, 37, 1120, 7104] {
            blas.prepare_fixed(rows, vision.hidden, geometry.patch)?;
        }
        closure.freeze_fixed()?;
        closure.ready(Instant::now())?;
        device.finish_upload()?;
        for (rows, h, w) in [(1u32, 2, 2), (37, 2, 20), (1120, 28, 40), (7104, 96, 74)] {
            let input: Vec<_> = (0..rows as usize * geometry.patch as usize)
                .map(|index| bf16::from_f32(((index * 17 % 251) as f32 - 125.0) / 128.0))
                .collect();
            let input = stream.clone_htod(&input)?;
            let output = stream.alloc_zeros::<bf16>(rows as usize * vision.hidden as usize)?;
            let tables = VisionPositions::new(
                vision,
                model.architecture,
                &[ImageGrid { t: 1, h, w }],
                (h * w) as usize,
            )?;
            let indices: Vec<_> = tables
                .bilinear_indices
                .iter()
                .flat_map(|corner| corner[..rows as usize].iter().copied())
                .collect();
            let coefficients: Vec<_> = tables
                .bilinear_weights
                .iter()
                .flat_map(|corner| corner[..rows as usize].iter().copied())
                .collect();
            let indices = stream.clone_htod(&indices)?;
            let bilinear = BilinearTables {
                coefficients: stream.clone_htod(&coefficients)?,
                values: stream.alloc_zeros(rows as usize * vision.hidden as usize)?,
            };
            let hidden = stream.alloc_zeros::<bf16>(rows as usize * vision.hidden as usize)?;
            let execute = || {
                io.project_position(
                    stream,
                    &blas,
                    PatchPositionBindings {
                        patch: &patch,
                        position: &position,
                        indices: &indices,
                        bilinear: Some(&bilinear),
                        packed: &input,
                        patch_output: &output,
                        hidden: &hidden,
                        trace_patches: false,
                    },
                )
            };
            execute()?;
            let reference = stream.clone_dtoh(&hidden)?;
            ensure!(
                reference.iter().all(|value| value.is_finite()),
                "nonfinite native patch output"
            );
            for graph in [false, true] {
                let captured = if graph {
                    stream.begin_capture(
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                    )?;
                    let result = execute();
                    let captured =
                        stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                    result?;
                    Some(captured?.context("empty patch operation graph")?)
                } else {
                    None
                };
                let mean_ms = crate::backend::cuda::time_prepared_operation(
                    stream,
                    || Ok(()),
                    || {
                        if let Some(graph) = &captured {
                            graph.launch()?;
                            Ok(())
                        } else {
                            execute()
                        }
                    },
                )?;
                ensure!(
                    stream
                        .clone_dtoh(&hidden)?
                        .iter()
                        .zip(&reference)
                        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                    "patch operation replay changed rows={rows} graph={graph}"
                );
                println!(
                    "T20_OP_SAMPLE {}",
                    serde_json::json!({"operation":"patch","mode":"projection_bias_position",
                        "rows":rows,"valid":rows,"graph":graph,"warm":16,"calls":32,
                        "mean_ms":mean_ms,"native_weights":true,"bias_mode":format!("{:?}",patch.bias_mode)})
                );
            }
        }
        closure.assert_ready()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires local vision GPU window and model/fixture/output paths; full tower diagnostics"]
    fn vision_tower_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
        use safetensors::{Dtype, SafeTensors};
        use std::{path::PathBuf, rc::Rc, time::Instant};
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
        let closure = Rc::new(Closure::new(false, "vision prefix diagnostic".into()));
        let mut blas = test_blas(&device, &output, closure.clone())?;
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
        let runtime = LayerNormTower::from_catalog(&model, &mut catalog, &device, common)?;
        ensure!(
            catalog.tensors.is_empty(),
            "vision binding left unloaded owners"
        );
        closure.freeze_fixed()?;
        closure.ready(Instant::now())?;
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
                audio: &[],
                input_ids: &ids,
                modality_ids: &modalities,
                pixel_values: &pixels,
                grids: &grids,
            };
            let request = runtime.prepare(
                &device,
                &mut blas,
                &input,
                ids.len(),
                device.ctx.mem_get_info()?.1 / 16,
            )?;
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
        closure.assert_ready()?;
        eprintln!("VISION_REQUEST_BINDINGS PASS cases=4 after_ready=true");
        Ok(())
    }

    fn decode_rgb_png_chw_f32(path: &std::path::Path, side: usize) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        let decoded = crate::serve::image::decode_image(&bytes, None)?;
        ensure!(
            decoded.width as usize == side
                && decoded.height as usize == side
                && decoded.pixels.len() == 3 * side * side,
            "unexpected PNG dimensions {}x{}",
            decoded.width,
            decoded.height
        );
        let hw = side * side;
        let mut chw = vec![0.0f32; 3 * hw];
        for idx in 0..hw {
            for c in 0..3 {
                chw[c * hw + idx] = f32::from(decoded.pixels[idx * 3 + c]) / 255.0f32;
            }
        }
        Ok(chw)
    }

    fn max_abs_and_rmse(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
        ensure!(
            actual.len() == expected.len() && !actual.is_empty(),
            "metric length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        let mut max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (&a, &e) in actual.iter().zip(expected) {
            ensure!(a.is_finite() && e.is_finite(), "non-finite tensor value");
            let d = (f64::from(a) - f64::from(e)).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_sq += d * d;
        }
        Ok((max_abs, (sum_sq / actual.len() as f64).sqrt()))
    }

    fn read_f32_tensor(st: &safetensors::SafeTensors<'_>, key: &str) -> Result<Vec<f32>> {
        let v = st.tensor(key)?;
        ensure!(
            v.dtype() == safetensors::Dtype::F32,
            "expected F32 for {key}, got {:?}",
            v.dtype()
        );
        Ok(v.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn read_bf16_tensor(st: &safetensors::SafeTensors<'_>, key: &str) -> Result<Vec<bf16>> {
        let v = st.tensor(key)?;
        ensure!(
            v.dtype() == safetensors::Dtype::BF16,
            "expected BF16 for {key}, got {:?}",
            v.dtype()
        );
        Ok(v.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect())
    }

    #[test]
    #[ignore = "requires local GPU window and FF_DINOV3_MODEL, FF_FLUX2_VAE_MODEL, FF_IDOL_FIXTURE_DIR, FF_IDOL_THRESHOLDS"]
    fn dinov3_and_flux2_vae_match_definition_gpu() -> Result<()> {
        use crate::backend::cuda::closure::Closure;
        use std::{path::PathBuf, rc::Rc, time::Instant};

        let dinov3_dir =
            PathBuf::from(std::env::var_os("FF_DINOV3_MODEL").context("FF_DINOV3_MODEL missing")?);
        let vae_dir = PathBuf::from(
            std::env::var_os("FF_FLUX2_VAE_MODEL").context("FF_FLUX2_VAE_MODEL missing")?,
        );
        let fixture_dir = PathBuf::from(
            std::env::var_os("FF_IDOL_FIXTURE_DIR").context("FF_IDOL_FIXTURE_DIR missing")?,
        );
        let thresholds_path = PathBuf::from(
            std::env::var_os("FF_IDOL_THRESHOLDS").context("FF_IDOL_THRESHOLDS missing")?,
        );
        let ref_png_f32_path = PathBuf::from(
            std::env::var_os("FF_IDOL_REF_PNG_F32").context("FF_IDOL_REF_PNG_F32 missing")?,
        );
        let runtime_dir = std::env::temp_dir().join(format!("ff-idol-m2-{}", std::process::id()));
        std::fs::create_dir_all(&runtime_dir)?;

        let thresholds: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&thresholds_path)?)?;
        let tensor_tau = |name: &str| -> Result<(bool, bool, f64, f64)> {
            let entry = &thresholds["tensors"][name];
            Ok((
                entry["gating_max_abs"]
                    .as_bool()
                    .context("gating_max_abs missing")?,
                entry["gating_rmse"]
                    .as_bool()
                    .context("gating_rmse missing")?,
                entry["tau_max_abs"]
                    .as_f64()
                    .context("tau_max_abs missing")?,
                entry["tau_rmse"].as_f64().context("tau_rmse missing")?,
            ))
        };

        let device = Device::new(0)?;
        let s = &device.stream;
        let common = CommonOpsRef::new(&device.ctx)?;
        let closure = Rc::new(Closure::new(false, "I1-M2 DINOv3 + FLUX.2 VAE".into()));
        let mut blas = test_blas(&device, &runtime_dir, closure.clone())?;
        let reserve = device.ctx.mem_get_info()?.1 / 16;
        let mut runtime = IdolVisionEncoderRuntime::load(
            &device,
            &mut blas,
            common,
            &dinov3_dir,
            &vae_dir,
            reserve,
        )?;
        closure.freeze_fixed()?;
        closure.ready(Instant::now())?;

        let sample_size = runtime.vae.cfg.sample_size as usize;
        let ref_png_file = std::fs::File::open(&ref_png_f32_path)?;
        let ref_png_mmap = unsafe { memmap2::MmapOptions::new().map(&ref_png_file)? };
        let ref_png_st = safetensors::SafeTensors::deserialize(&ref_png_mmap)?;
        let mut worst: std::collections::BTreeMap<&str, (f64, f64)> =
            std::collections::BTreeMap::new();
        for case in ["armchair", "avocado"] {
            let chw =
                decode_rgb_png_chw_f32(&fixture_dir.join(format!("{case}_1024.png")), sample_size)?;
            let ref_file =
                std::fs::File::open(fixture_dir.join(case).join("reference.safetensors"))?;
            let ref_mmap = unsafe { memmap2::MmapOptions::new().map(&ref_file)? };
            let ref_st = safetensors::SafeTensors::deserialize(&ref_mmap)?;
            let ref_px = read_f32_tensor(&ref_png_st, &format!("{case}_pixel_values_f32"))?;
            ensure!(
                chw.len() == ref_px.len(),
                "pixel_values_f32 length mismatch: {} vs {}",
                chw.len(),
                ref_px.len()
            );
            let diff_f32_bits = chw
                .iter()
                .zip(&ref_px)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!("I1_M2_PNG_ENGINE_DECODE case={case} diff_f32_bits={diff_f32_bits}");
            ensure!(
                diff_f32_bits == 0,
                "engine decode_image F32 mismatch for {case}: {diff_f32_bits} elements differ"
            );
            let vae_eps = read_bf16_tensor(&ref_st, "vae_eps")?;

            let (d_out, v_out) = runtime.encode_image(s, &blas, &chw, &vae_eps, true)?;

            for (name, actual) in [
                ("dinov3_layer0", d_out.layer0.as_ref().unwrap()),
                ("dinov3_layer15", d_out.layer15.as_ref().unwrap()),
                ("feature1", &d_out.feature1),
                ("vae_moments", &v_out.moments_f32),
                ("feature2", &v_out.feature2_f32),
            ] {
                let expected = read_f32_tensor(&ref_st, name)?;
                let (ma, rmse) = max_abs_and_rmse(actual, &expected)?;
                let (gate_ma, gate_rmse, tau_ma, tau_rmse) = tensor_tau(name)?;
                println!(
                    "I1_M2_METRIC case={case} tensor={name} max_abs={ma:.8} tau_max_abs={tau_ma:.8} gating_max_abs={gate_ma} rmse={rmse:.8} tau_rmse={tau_rmse:.8} gating_rmse={gate_rmse}"
                );
                if gate_ma {
                    ensure!(
                        ma <= tau_ma,
                        "DEFECT {case} {name} max_abs {ma} > tau {tau_ma}"
                    );
                }
                if gate_rmse {
                    ensure!(
                        rmse <= tau_rmse,
                        "DEFECT {case} {name} rmse {rmse} > tau {tau_rmse}"
                    );
                }
                let entry = worst.entry(name).or_insert((0.0, 0.0));
                entry.0 = entry.0.max(ma);
                entry.1 = entry.1.max(rmse);
            }

            if case == "armchair" {
                let d_cfg = &runtime.dinov3.cfg;
                let patches_side = runtime.dinov3.image_side / d_cfg.patch_size;
                let head_dim = d_cfg.head_dim() as usize;
                let prefix_rows = d_cfg.prefix_rows();
                let mut mutant_rope = crate::vision_input::dinov3_rope_table_bf16(
                    prefix_rows,
                    patches_side,
                    patches_side,
                    d_cfg.head_dim(),
                    d_cfg.rope_theta,
                )?;
                mutant_rope.copy_within(prefix_rows * head_dim..2 * prefix_rows * head_dim, 0);
                let mutant_rope_gpu = s.clone_htod(&mutant_rope)?;
                let neg_d = runtime.dinov3.encode(
                    s,
                    &blas,
                    &mut runtime.dinov3_ws,
                    &chw,
                    Some(&mutant_rope_gpu),
                    false,
                )?;
                let ref_f1 = read_f32_tensor(&ref_st, "feature1")?;
                let (neg_f1_ma, neg_f1_rmse) = max_abs_and_rmse(&neg_d.feature1, &ref_f1)?;
                let (_, _, tau_f1_ma, tau_f1_rmse) = tensor_tau("feature1")?;
                println!(
                    "I1_M2_NEG control=neg_g1a_rotate_prefix tensor=feature1 max_abs={neg_f1_ma:.8} ratio_max_abs={:.4} rmse={neg_f1_rmse:.8} ratio_rmse={:.4}",
                    neg_f1_ma / tau_f1_ma,
                    neg_f1_rmse / tau_f1_rmse
                );
                ensure!(
                    neg_f1_rmse >= 10.0 * tau_f1_rmse,
                    "neg_g1a_rotate_prefix feature1 rmse {neg_f1_rmse} < 10 * tau {tau_f1_rmse}"
                );

                let neg_v = runtime.vae.encode(
                    s,
                    &blas,
                    &mut runtime.vae_ws,
                    &chw,
                    &vae_eps,
                    [1, 0, 1, 0],
                )?;
                for (name, actual) in [
                    ("vae_moments", &neg_v.moments_f32),
                    ("feature2", &neg_v.feature2_f32),
                ] {
                    let expected = read_f32_tensor(&ref_st, name)?;
                    let (neg_ma, neg_rmse) = max_abs_and_rmse(actual, &expected)?;
                    let (_, _, tau_ma, tau_rmse) = tensor_tau(name)?;
                    println!(
                        "I1_M2_NEG control=neg_g1b_wrong_asym_pad tensor={name} max_abs={neg_ma:.8} ratio_max_abs={:.4} rmse={neg_rmse:.8} ratio_rmse={:.4}",
                        neg_ma / tau_ma,
                        neg_rmse / tau_rmse
                    );
                    ensure!(
                        neg_ma >= 10.0 * tau_ma && neg_rmse >= 10.0 * tau_rmse,
                        "neg_g1b_wrong_asym_pad {name} did not exceed 10*tau: max_abs={neg_ma} rmse={neg_rmse}"
                    );
                }
            }
        }
        closure.assert_ready()?;
        let _ = std::fs::remove_dir_all(&runtime_dir);
        println!(
            "I1_M2_GATE PASS strip_pixels={} feature1_worst_rmse={:.8} vae_moments_worst_max_abs={:.8} vae_moments_worst_rmse={:.8} feature2_worst_max_abs={:.8} feature2_worst_rmse={:.8}",
            runtime.vae_ws.strip_pixels,
            worst["feature1"].1,
            worst["vae_moments"].0,
            worst["vae_moments"].1,
            worst["feature2"].0,
            worst["feature2"].1
        );
        Ok(())
    }
}
