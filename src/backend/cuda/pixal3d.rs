//! Dense structure and sparse shape-prefix decoding on published layouts.
use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{Tensor, TensorMut, view},
    cubin,
    ops::{Ops, flat, grid},
    sparse::SparseConv,
};
use crate::{
    backend::gpu::{CudaFunction, CudaSlice, DevicePtr, DeviceRepr, PushKernelArg},
    pixal3d::{DecoderCheckpoint, DecoderKind},
    sparse::SparseSites,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use safetensors::Dtype;
use std::{collections::BTreeMap, path::Path, rc::Rc};

fn ptr<T: DeviceRepr>(x: &CudaSlice<T>, d: &Device) -> u64 {
    x.device_ptr(&d.stream).0
}
fn matrix<T: DeviceRepr + Copy>(
    x: &CudaSlice<T>,
    d: &Device,
    rows: usize,
    width: usize,
) -> Result<Tensor<T, 2>> {
    view(
        x,
        &d.stream,
        0,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}
fn raw_matrix<T: Copy>(address: u64, rows: usize, width: usize) -> Result<Tensor<T, 2>> {
    let bytes = rows
        .checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<T>()))
        .context("decoder matrix overflow")?;
    unsafe {
        Tensor::new(
            address,
            [u32::try_from(rows)?, u32::try_from(width)?],
            [width as u64, 1],
            bytes as u64,
        )
    }
}
struct Weights {
    half: BTreeMap<String, CudaSlice<f16>>,
    float: BTreeMap<String, CudaSlice<f32>>,
}
impl Weights {
    fn load(d: &Device, checkpoint: &DecoderCheckpoint) -> Result<Self> {
        let mut half = BTreeMap::new();
        let mut float = BTreeMap::new();
        for (name, (_, _, f32_runtime)) in checkpoint.kind.inventory() {
            let t = checkpoint.mapped.tensor(&name)?;
            if f32_runtime {
                let values: Vec<f32> = if t.dtype() == Dtype::F32 {
                    t.data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect()
                } else {
                    t.data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f16::from_le_bytes(*b).to_f32())
                        .collect()
                };
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite decoder weight {name}"
                );
                float.insert(name, d.stream.clone_htod(&values)?);
            } else {
                let values: Vec<f16> = t
                    .data()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16::from_le_bytes(*b))
                    .collect();
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "nonfinite decoder weight {name}"
                );
                half.insert(name, d.stream.clone_htod(&values)?);
            }
        }
        Ok(Self { half, float })
    }
    fn affine(&self, d: &Device, name: Option<&str>) -> Result<[u64; 2]> {
        match name {
            Some(name) => Ok([
                ptr(
                    self.float
                        .get(&format!("{name}.weight"))
                        .context("norm weight missing")?,
                    d,
                ),
                ptr(
                    self.float
                        .get(&format!("{name}.bias"))
                        .context("norm bias missing")?,
                    d,
                ),
            ]),
            None => Ok([0, 0]),
        }
    }
}
struct Kernels {
    ln16: CudaFunction,
    ln32: CudaFunction,
    act16: CudaFunction,
    act32: CudaFunction,
    cast: CudaFunction,
    upcast: CudaFunction,
    bias16: CudaFunction,
    bias32: CudaFunction,
    residual: CudaFunction,
    shuffle: CudaFunction,
    c2s: CudaFunction,
    conv16: CudaFunction,
    conv32: CudaFunction,
}
impl Kernels {
    fn load(d: &Device) -> Result<Self> {
        let q = cubin::module(&d.ctx, "q38_common")?;
        let v = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            ln16: q.load_function("q38_layer_norm_f16")?,
            ln32: q.load_function("q38_layer_norm_f32")?,
            act16: v.load_function("activation_f16")?,
            act32: v.load_function("activation_f32")?,
            cast: v.load_function("vision_downcast_f32_to_f16")?,
            upcast: v.load_function("vision_upcast_f16_to_f32")?,
            bias16: q.load_function("q38_bias_add_f16")?,
            bias32: v.load_function("vision_bias_f32")?,
            residual: q.load_function("channel_scale_residual_f16")?,
            shuffle: v.load_function("pixel_shuffle_3d_f16")?,
            c2s: v.load_function("channel_to_spatial_f16")?,
            conv16: v.load_function("conv3d_im2col_f16")?,
            conv32: v.load_function("conv3d_im2col_f32")?,
        })
    }
    fn norm(
        &self,
        d: &Device,
        io: [u64; 2],
        shape: [usize; 2],
        affine: [u64; 2],
        f32_io: bool,
        eps: f32,
    ) -> Result<()> {
        let [rows, width] = shape;
        unsafe {
            d.stream
                .launch_builder(if f32_io { &self.ln32 } else { &self.ln16 })
                .arg(&io[0])
                .arg(&affine[0])
                .arg(&affine[1])
                .arg(&io[1])
                .arg(&(rows as u32))
                .arg(&1u32)
                .arg(&(width as u32))
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&(width as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&eps)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }
    fn act(&self, d: &Device, x: u64, rows: usize, width: usize, f32_io: bool) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(if f32_io { &self.act32 } else { &self.act16 })
                .arg(&x)
                .arg(&x)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .arg(&1i32)
                .launch(flat(rows * width))?;
        }
        Ok(())
    }
    fn cast(&self, d: &Device, io: [u64; 2], count: usize, up: bool) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(if up { &self.upcast } else { &self.cast })
                .arg(&io[0])
                .arg(&io[1])
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
    fn bias(
        &self,
        d: &Device,
        y: u64,
        bias: u64,
        shape: [usize; 2],
        overwrite: bool,
        f32_io: bool,
    ) -> Result<()> {
        let [rows, width] = shape;
        unsafe {
            if f32_io {
                d.stream
                    .launch_builder(&self.bias32)
                    .arg(&y)
                    .arg(&bias)
                    .arg(&y)
                    .arg(&(rows as u32))
                    .arg(&(width as u32))
                    .launch(flat(rows * width))?;
            } else {
                d.stream
                    .launch_builder(&self.bias16)
                    .arg(&y)
                    .arg(&bias)
                    .arg(&((rows * width) as u64))
                    .arg(&(width as u32))
                    .arg(&(overwrite as i32))
                    .launch(flat(rows * width))?;
            }
        }
        Ok(())
    }
    fn add(
        &self,
        d: &Device,
        x: &CudaSlice<f16>,
        y: &CudaSlice<f16>,
        ones: &CudaSlice<f16>,
        shape: [usize; 2],
    ) -> Result<()> {
        let [rows, width] = shape;
        unsafe {
            d.stream
                .launch_builder(&self.residual)
                .arg(x)
                .arg(y)
                .arg(ones)
                .arg(x)
                .arg(&(rows as u32))
                .arg(&(width as u32))
                .launch(flat(rows * width))?;
        }
        Ok(())
    }
}

pub struct PixalDecoder {
    device: Device,
    blas: Blas,
    weights: Weights,
    k: Kernels,
    kind: DecoderKind,
    workspace: usize,
    ones: CudaSlice<f16>,
    columns: Option<CudaSlice<u8>>,
}
pub struct SparseDecoderOutput {
    pub coordinates: Vec<[i32; 4]>,
    pub features: Vec<f32>,
    pub channels: usize,
}
pub struct SubdivisionGuide<'a> {
    pub coordinates: &'a [[i32; 4]],
    pub logits: &'a [f32],
}
impl PixalDecoder {
    pub fn open(
        dir: &Path,
        kind: DecoderKind,
        runtime: &Path,
        device_budget: usize,
    ) -> Result<Self> {
        let checkpoint = DecoderCheckpoint::open(&dir.join("ckpts"), kind)?;
        let device = Device::new(0)?;
        let (free, _) = device.ctx.mem_get_info()?;
        let available = device_budget
            .min(free)
            .checked_sub(checkpoint.runtime_bytes + 512 * 1024 * 1024)
            .context("decoder weights and reserve exceed device budget")?;
        ensure!(
            available >= 512 * 1024 * 1024,
            "decoder workspace cannot fit a stage"
        );
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(false, "Pixal decoder".into())),
        )?;
        let k = Kernels::load(&device)?;
        let weights = Weights::load(&device, &checkpoint)?;
        let ones = device.stream.clone_htod(&vec![f16::ONE; 1024])?;
        let columns = if matches!(kind, DecoderKind::Structure) {
            Some(
                device
                    .stream
                    .alloc_zeros::<u8>((available / 2).min(512 * 1024 * 1024))?,
            )
        } else {
            None
        };
        println!(
            "PIXAL_DECODER_SETUP kind={kind:?} weight_bytes={} workspace_budget={available} device_budget={device_budget} eager=true",
            checkpoint.runtime_bytes
        );
        Ok(Self {
            device,
            blas,
            weights,
            k,
            kind,
            workspace: available,
            ones,
            columns,
        })
    }
    fn linear16(
        &self,
        name: &str,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<f16>,
        shape: [usize; 3],
    ) -> Result<()> {
        let [rows, ci, co] = shape;
        let d = &self.device;
        let w = self
            .weights
            .half
            .get(&format!("{name}.weight"))
            .context("linear16 weight missing")?;
        let b = self
            .weights
            .half
            .get(&format!("{name}.bias"))
            .context("linear16 bias missing")?;
        self.k
            .bias(d, ptr(output, d), ptr(b, d), [rows, co], true, false)?;
        self.blas.linear_f16(
            matrix(w, d, co, ci)?,
            matrix(input, d, rows, ci)?,
            &mut TensorMut(matrix(output, d, rows, co)?),
            true,
        )
    }
    fn linear32(
        &self,
        name: &str,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        shape: [usize; 3],
    ) -> Result<()> {
        let [rows, ci, co] = shape;
        let d = &self.device;
        let w = self
            .weights
            .float
            .get(&format!("{name}.weight"))
            .context("linear32 weight missing")?;
        let b = self
            .weights
            .float
            .get(&format!("{name}.bias"))
            .context("linear32 bias missing")?;
        self.blas.linear_f32(
            matrix(w, d, co, ci)?,
            matrix(input, d, rows, ci)?,
            &mut TensorMut(matrix(output, d, rows, co)?),
        )?;
        self.k
            .bias(d, ptr(output, d), ptr(b, d), [rows, co], false, true)
    }
    fn sparse_conv(
        &self,
        conv: &SparseConv,
        name: &str,
        input: &CudaSlice<f16>,
        output: &mut CudaSlice<f16>,
        ci: usize,
    ) -> Result<()> {
        let d = &self.device;
        let w = self
            .weights
            .half
            .get(&format!("{name}.weight"))
            .context("sparse conv weight missing")?;
        let b = self
            .weights
            .half
            .get(&format!("{name}.bias"))
            .context("sparse conv bias missing")?;
        conv.run(
            &self.blas,
            input,
            matrix(w, d, b.len(), 27 * ci)?,
            view(b, &d.stream, 0, [b.len() as u32], [1])?,
            output,
        )
    }
    fn norm16(
        &self,
        name: Option<&str>,
        input: &CudaSlice<f16>,
        output: &CudaSlice<f16>,
        rows: usize,
        c: usize,
        eps: f32,
    ) -> Result<()> {
        self.k.norm(
            &self.device,
            [ptr(input, &self.device), ptr(output, &self.device)],
            [rows, c],
            self.weights.affine(&self.device, name)?,
            false,
            eps,
        )
    }
    fn next_blocks(
        &self,
        stage: usize,
        conv: &SparseConv,
        x: CudaSlice<f16>,
        c: usize,
        blocks: usize,
    ) -> Result<CudaSlice<f16>> {
        let d = &self.device;
        let n = x.len() / c;
        let features = n.checked_mul(c).context("sparse features overflow")?;
        let live = features
            .checked_mul(16)
            .context("sparse workspace overflow")?;
        ensure!(
            live.checked_add(512 * 1024 * 1024)
                .is_some_and(|b| b <= self.workspace),
            "sparse stage activations and gather exceed budget"
        );
        let mut a = d.stream.alloc_zeros::<f16>(features)?;
        let b = d.stream.alloc_zeros::<f16>(features)?;
        let mut wide = d.stream.alloc_zeros::<f16>(features * 4)?;
        for block in 0..blocks {
            let name = format!("blocks.{stage}.{block}");
            self.sparse_conv(conv, &format!("{name}.conv"), &x, &mut a, c)?;
            self.norm16(Some(&format!("{name}.norm")), &a, &b, n, c, 1e-6)?;
            self.linear16(&format!("{name}.mlp.0"), &b, &mut wide, [n, c, 4 * c])?;
            self.k.act(d, ptr(&wide, d), n, 4 * c, false)?;
            self.linear16(&format!("{name}.mlp.2"), &wide, &mut a, [n, 4 * c, c])?;
            self.k.add(d, &x, &a, &self.ones, [n, c])?;
        }
        Ok(x)
    }
    fn c2s(
        &self,
        x: &CudaSlice<f16>,
        pairs: &CudaSlice<u32>,
        y: &CudaSlice<f16>,
        shape: [usize; 4],
        add: bool,
    ) -> Result<()> {
        let [parents, rows, ci, co] = shape;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.k.c2s)
                .arg(x)
                .arg(pairs)
                .arg(y)
                .arg(&(parents as u32))
                .arg(&(rows as u32))
                .arg(&(ci as u32))
                .arg(&(co as u32))
                .arg(&(add as i32))
                .launch(flat(rows * co))?;
        }
        Ok(())
    }
    fn up_stage(
        &self,
        stage: usize,
        sites: SparseSites,
        x: CudaSlice<f16>,
        conv: SparseConv,
        guide: SubdivisionGuide<'_>,
    ) -> Result<(SparseSites, CudaSlice<f16>, Vec<f32>, SparseConv)> {
        let c = 1024 >> stage;
        let co = c / 2;
        let n = sites.coordinates().len();
        let blocks = [4, 16, 8, 4][stage];
        ensure!(
            guide.coordinates == sites.coordinates()
                && guide.logits.len() == n * 8
                && guide.logits.iter().all(|x| x.is_finite()),
            "subdivision guide does not match stage coordinates"
        );
        let active: Vec<[bool; 8]> = guide
            .logits
            .as_chunks::<8>()
            .0
            .iter()
            .map(|row| row.map(|x| x > 0.0))
            .collect();
        let capacity = self.workspace / (co * 16 + 108 + 16);
        let (children, source) = sites.subdivide(&active, capacity)?;
        let child_n = children.coordinates().len();
        let name = format!("blocks.{stage}.{blocks}");
        let d = &self.device;
        let live = n
            .checked_mul(c * 2 * 7 + 16)
            .and_then(|b| b.checked_add(child_n * co * 2 * 3 + child_n * 8))
            .context("C2S workspace overflow")?;
        let budget = self
            .workspace
            .checked_sub(live)
            .context("C2S live activations exceed budget")?
            .min(512 * 1024 * 1024);
        let mut subdivision = d.stream.alloc_zeros::<f16>(n * 8)?;
        self.linear16(
            &format!("{name}.to_subdiv"),
            &x,
            &mut subdivision,
            [n, c, 8],
        )?;
        let predicted = d
            .stream
            .clone_dtoh(&subdivision)?
            .into_iter()
            .map(f16::to_f32)
            .collect();
        drop(subdivision);
        let normalized = d.stream.alloc_zeros::<f16>(n * c)?;
        self.norm16(Some(&format!("{name}.norm1")), &x, &normalized, n, c, 1e-6)?;
        self.k.act(d, ptr(&normalized, d), n, c, false)?;
        let mut expanded = d.stream.alloc_zeros::<f16>(n * 8 * co)?;
        self.sparse_conv(
            &conv,
            &format!("{name}.conv1"),
            &normalized,
            &mut expanded,
            c,
        )?;
        drop(conv);
        drop(normalized);
        drop(sites);
        let pairs = d.stream.clone_htod(source.as_flattened())?;
        let child = d.stream.alloc_zeros::<f16>(child_n * co)?;
        self.c2s(&expanded, &pairs, &child, [n, child_n, co, co], false)?;
        drop(expanded);
        let normalized = d.stream.alloc_zeros::<f16>(child_n * co)?;
        self.norm16(None, &child, &normalized, child_n, co, 1e-6)?;
        drop(child);
        self.k.act(d, ptr(&normalized, d), child_n, co, false)?;
        let mut output = d.stream.alloc_zeros::<f16>(child_n * co)?;
        let conv = SparseConv::new(d, &children, co, budget)?;
        self.sparse_conv(
            &conv,
            &format!("{name}.conv2"),
            &normalized,
            &mut output,
            co,
        )?;
        self.c2s(&x, &pairs, &output, [n, child_n, c / 8, co], true)?;
        Ok((children, output, predicted, conv))
    }
    pub fn shape_prefix<F>(
        &self,
        coordinates: Vec<[i32; 4]>,
        latent: &[f32],
        guides: &[SubdivisionGuide<'_>],
        mut observe: F,
    ) -> Result<SparseDecoderOutput>
    where
        F: FnMut(usize, &SparseDecoderOutput, &[f32]) -> Result<()>,
    {
        ensure!(
            matches!(self.kind, DecoderKind::Shape),
            "shape decoder required"
        );
        ensure!(
            guides.len() == 4,
            "four recorded subdivision guides required"
        );
        let mut sites = SparseSites::new(coordinates)?;
        let n = sites.coordinates().len();
        ensure!(
            latent.len() == n * 32 && latent.iter().all(|x| x.is_finite()),
            "invalid denormalized sparse latent"
        );
        ensure!(
            n.checked_mul(32 * 4 + 1024 * 6)
                .is_some_and(|b| b < self.workspace),
            "sparse input exceeds workspace"
        );
        let d = &self.device;
        let input = d.stream.clone_htod(latent)?;
        let mut float = d.stream.alloc_zeros::<f32>(n * 1024)?;
        self.linear32("from_latent", &input, &mut float, [n, 32, 1024])?;
        drop(input);
        let mut x = d.stream.alloc_zeros::<f16>(n * 1024)?;
        self.k
            .cast(d, [ptr(&float, d), ptr(&x, d)], n * 1024, false)?;
        drop(float);
        let mut conv = SparseConv::new(d, &sites, 1024, 512 * 1024 * 1024)?;
        for (stage, guide) in guides.iter().enumerate() {
            x = self.next_blocks(stage, &conv, x, 1024 >> stage, [4, 16, 8, 4][stage])?;
            let (next, features, predicted, next_conv) = self.up_stage(
                stage,
                sites,
                x,
                conv,
                SubdivisionGuide {
                    coordinates: guide.coordinates,
                    logits: guide.logits,
                },
            )?;
            let host = SparseDecoderOutput {
                coordinates: next.coordinates().to_vec(),
                features: d
                    .stream
                    .clone_dtoh(&features)?
                    .into_iter()
                    .map(f16::to_f32)
                    .collect(),
                channels: 512 >> stage,
            };
            ensure!(
                host.features.iter().all(|x| x.is_finite())
                    && predicted.iter().all(|x| x.is_finite()),
                "nonfinite sparse stage output"
            );
            observe(stage, &host, &predicted)?;
            sites = next;
            x = features;
            conv = next_conv;
        }
        Ok(SparseDecoderOutput {
            coordinates: sites.coordinates().to_vec(),
            features: d
                .stream
                .clone_dtoh(&x)?
                .into_iter()
                .map(f16::to_f32)
                .collect(),
            channels: 64,
        })
    }
    fn dense_conv(&self, name: &str, io: [u64; 2], shape: [usize; 3], half: bool) -> Result<()> {
        let [side, ci, co] = shape;
        let n = side.pow(3);
        let width = 27 * ci;
        let d = &self.device;
        let columns = self.columns.as_ref().context("dense columns unavailable")?;
        let bytes = if half { 2 } else { 4 };
        let tile = (columns.len() / (width * bytes)).min(n);
        ensure!(tile > 0, "dense conv cannot fit one row");
        let (weight, bias) = if half {
            (
                ptr(
                    self.weights
                        .half
                        .get(&format!("{name}.weight"))
                        .context("dense F16 weight missing")?,
                    d,
                ),
                ptr(
                    self.weights
                        .half
                        .get(&format!("{name}.bias"))
                        .context("dense F16 bias missing")?,
                    d,
                ),
            )
        } else {
            (
                ptr(
                    self.weights
                        .float
                        .get(&format!("{name}.weight"))
                        .context("dense F32 weight missing")?,
                    d,
                ),
                ptr(
                    self.weights
                        .float
                        .get(&format!("{name}.bias"))
                        .context("dense F32 bias missing")?,
                    d,
                ),
            )
        };
        for first in (0..n).step_by(tile) {
            let rows = tile.min(n - first);
            let y = io[1] + (first * co * bytes) as u64;
            unsafe {
                d.stream
                    .launch_builder(if half { &self.k.conv16 } else { &self.k.conv32 })
                    .arg(&io[0])
                    .arg(columns)
                    .arg(&(ci as u32))
                    .arg(&(side as u32))
                    .arg(&(side as u32))
                    .arg(&(side as u32))
                    .arg(&3u32)
                    .arg(&(first as u64))
                    .arg(&(rows as u32))
                    .arg(&0u32)
                    .launch(flat(rows * width))?;
            }
            if half {
                self.blas.linear_f16(
                    raw_matrix::<f16>(weight, co, width)?,
                    raw_matrix(ptr(columns, d), rows, width)?,
                    &mut TensorMut(raw_matrix(y, rows, co)?),
                    false,
                )?;
            } else {
                self.blas.linear_f32(
                    raw_matrix::<f32>(weight, co, width)?,
                    raw_matrix(ptr(columns, d), rows, width)?,
                    &mut TensorMut(raw_matrix(y, rows, co)?),
                )?;
            }
            self.k.bias(d, y, bias, [rows, co], false, !half)?;
        }
        Ok(())
    }
    fn dense_residual(&self, name: &str, x: &CudaSlice<f16>, side: usize, c: usize) -> Result<()> {
        let d = &self.device;
        let rows = side.pow(3);
        let a = d.stream.alloc_zeros::<f16>(rows * c)?;
        let b = d.stream.alloc_zeros::<f16>(rows * c)?;
        self.norm16(Some(&format!("{name}.norm1")), x, &a, rows, c, 1e-5)?;
        self.k.act(d, ptr(&a, d), rows, c, false)?;
        self.dense_conv(
            &format!("{name}.conv1"),
            [ptr(&a, d), ptr(&b, d)],
            [side, c, c],
            true,
        )?;
        self.norm16(Some(&format!("{name}.norm2")), &b, &a, rows, c, 1e-5)?;
        self.k.act(d, ptr(&a, d), rows, c, false)?;
        self.dense_conv(
            &format!("{name}.conv2"),
            [ptr(&a, d), ptr(&b, d)],
            [side, c, c],
            true,
        )?;
        self.k.add(d, x, &b, &self.ones, [rows, c])
    }
    pub fn structure(&self, latent_chw: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            matches!(self.kind, DecoderKind::Structure),
            "structure decoder required"
        );
        ensure!(
            latent_chw.len() == 8 * 16usize.pow(3) && latent_chw.iter().all(|x| x.is_finite()),
            "structure latent must be finite [8,16,16,16]"
        );
        ensure!(
            self.workspace >= 512 * 1024 * 1024,
            "dense decoder workspace below live bound"
        );
        let rows = 16usize.pow(3);
        let input: Vec<f32> = (0..rows * 8)
            .map(|i| latent_chw[(i % 8) * rows + i / 8])
            .collect();
        let d = &self.device;
        let input = d.stream.clone_htod(&input)?;
        let float = d.stream.alloc_zeros::<f32>(rows * 512)?;
        self.dense_conv(
            "input_layer",
            [ptr(&input, d), ptr(&float, d)],
            [16, 8, 512],
            false,
        )?;
        drop(input);
        let mut x = d.stream.alloc_zeros::<f16>(rows * 512)?;
        self.k
            .cast(d, [ptr(&float, d), ptr(&x, d)], rows * 512, false)?;
        drop(float);
        for block in 0..2 {
            self.dense_residual(&format!("middle_block.{block}"), &x, 16, 512)?;
        }
        let mut side = 16usize;
        let mut c = 512;
        for block in 0..8 {
            if block == 2 || block == 5 {
                let co = c / 4;
                let expanded = d.stream.alloc_zeros::<f16>(side.pow(3) * co * 8)?;
                self.dense_conv(
                    &format!("blocks.{block}.conv"),
                    [ptr(&x, d), ptr(&expanded, d)],
                    [side, c, co * 8],
                    true,
                )?;
                drop(x);
                let next = d.stream.alloc_zeros::<f16>((side * 2).pow(3) * co)?;
                unsafe {
                    d.stream
                        .launch_builder(&self.k.shuffle)
                        .arg(&expanded)
                        .arg(&next)
                        .arg(&(side as u32))
                        .arg(&(co as u32))
                        .launch(flat(next.len()))?;
                }
                x = next;
                side *= 2;
                c = co;
            } else {
                self.dense_residual(&format!("blocks.{block}"), &x, side, c)?;
            }
        }
        let rows = side.pow(3);
        let float = d.stream.alloc_zeros::<f32>(rows * c)?;
        self.k
            .cast(d, [ptr(&x, d), ptr(&float, d)], rows * c, true)?;
        drop(x);
        let normalized = d.stream.alloc_zeros::<f32>(rows * c)?;
        self.k.norm(
            d,
            [ptr(&float, d), ptr(&normalized, d)],
            [rows, c],
            self.weights.affine(d, Some("out_layer.0"))?,
            true,
            1e-5,
        )?;
        drop(float);
        self.k.act(d, ptr(&normalized, d), rows, c, true)?;
        let logits = d.stream.alloc_zeros::<f32>(rows)?;
        self.dense_conv(
            "out_layer.2",
            [ptr(&normalized, d), ptr(&logits, d)],
            [side, c, 1],
            false,
        )?;
        let logits = d.stream.clone_dtoh(&logits)?;
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite structure logits"
        );
        Ok(logits)
    }
    pub fn structure_sites(logits: &[f32], side: usize, target: usize) -> Result<SparseSites> {
        ensure!(
            side > 0
                && target > 0
                && side.is_multiple_of(target)
                && side.checked_pow(3) == Some(logits.len())
                && logits.iter().all(|x| x.is_finite()),
            "invalid occupancy grid"
        );
        let factor = side / target;
        let mut points = Vec::new();
        for x in 0..target {
            for y in 0..target {
                for z in 0..target {
                    let mut occupied = false;
                    for dx in 0..factor {
                        for dy in 0..factor {
                            for dz in 0..factor {
                                occupied |= logits[((x * factor + dx) * side + y * factor + dy)
                                    * side
                                    + z * factor
                                    + dz]
                                    > 0.0;
                            }
                        }
                    }
                    if occupied {
                        points.push([0, i32::try_from(x)?, i32::try_from(y)?, i32::try_from(z)?]);
                    }
                }
            }
        }
        SparseSites::new(points)
    }
}

#[cfg(test)]
#[path = "pixal3d_tests.rs"]
mod tests;
