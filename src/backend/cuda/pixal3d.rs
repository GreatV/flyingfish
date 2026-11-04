//! Pixal3D flow execution on recorded conditioning and fixed coordinates.
#[cfg(test)]
#[path = "pixal3d_tests.rs"]
mod tests;
use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{Tensor, TensorMut, view},
    cubin,
    flash::{F32Attention, Flash, Shape},
    linear::LinearBiasMode,
    ops::{Ops, flat, grid},
};
use crate::{
    backend::gpu::{CudaFunction, CudaSlice, DevicePtr, DeviceRepr, PushKernelArg},
    pixal3d::{FlowCheckpoint, FlowSpec},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{collections::BTreeMap, mem::size_of, path::Path, rc::Rc};

fn ptr<T: DeviceRepr>(x: &CudaSlice<T>, d: &Device) -> u64 {
    x.device_ptr(&d.stream).0
}
fn dense<T: DeviceRepr + Copy, const R: usize>(
    x: &CudaSlice<T>,
    d: &Device,
    shape: [u32; R],
) -> Result<Tensor<T, R>> {
    let (stride, _) = super::contracts::dense_layout(shape)?;
    view(x, &d.stream, 0, shape, stride)
}
fn matrix<T: Copy>(address: u64, rows: usize, width: usize) -> Result<Tensor<T, 2>> {
    let bytes = rows
        .checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<T>()))
        .context("flow matrix size overflow")?;
    unsafe {
        Tensor::new(
            address,
            [u32::try_from(rows)?, u32::try_from(width)?],
            [width as u64, 1],
            bytes as u64,
        )
    }
}

#[derive(Clone, Copy)]
struct Linear<T> {
    weight: Tensor<T, 2>,
    bias: Tensor<T, 1>,
}
impl<T: DeviceRepr + Copy> Linear<T> {
    fn bind(
        weights: &BTreeMap<String, CudaSlice<T>>,
        d: &Device,
        name: &str,
        shape: [usize; 2],
    ) -> Result<Self> {
        let weight = weights
            .get(&format!("{name}.weight"))
            .with_context(|| format!("linear weight missing: {name}"))?;
        let bias = weights
            .get(&format!("{name}.bias"))
            .with_context(|| format!("linear bias missing: {name}"))?;
        Ok(Self {
            weight: dense(weight, d, [shape[0] as u32, shape[1] as u32])?,
            bias: dense(bias, d, [shape[0] as u32])?,
        })
    }
}
impl Linear<bf16> {
    fn prepare(self, blas: &mut Blas, rows: usize) -> Result<()> {
        blas.prepare_biased(
            rows as u32,
            self.weight.shape[0],
            self.weight.shape[1],
            self.bias,
            LinearBiasMode::BeforeOutputRound,
        )
    }
    fn run(self, blas: &Blas, rows: usize, input: u64, output: u64) -> Result<()> {
        blas.linear_biased(
            self.weight,
            self.bias,
            matrix(input, rows, self.weight.shape[1] as usize)?,
            &mut TensorMut(matrix(output, rows, self.weight.shape[0] as usize)?),
            LinearBiasMode::BeforeOutputRound,
        )
    }
}

struct Block {
    qkv: Linear<bf16>,
    self_out: Linear<bf16>,
    cross_q: Linear<bf16>,
    cross_kv: Linear<bf16>,
    cross_out: Linear<bf16>,
    projection: Linear<bf16>,
    fc1: Linear<bf16>,
    fc2: Linear<bf16>,
    norm_weight: u64,
    norm_bias: u64,
    modulation: u64,
    gamma: [u64; 4],
}
struct Kernels {
    heads: usize,
    head_dim: usize,
    norm: CudaFunction,
    norm_f32: CudaFunction,
    l2: CudaFunction,
    unpack: CudaFunction,
    rope: CudaFunction,
    rope_hm: CudaFunction,
    phases: CudaFunction,
    modulate: CudaFunction,
    gated: CudaFunction,
    add_mod: CudaFunction,
    residual: CudaFunction,
    cast: CudaFunction,
    upcast: CudaFunction,
    bias: CudaFunction,
    silu: CudaFunction,
    sinusoid: CudaFunction,
    gelu: CudaFunction,
    transpose: CudaFunction,
}
impl Kernels {
    fn load(d: &Device, spec: &FlowSpec) -> Result<Self> {
        let q = cubin::module(&d.ctx, "q38_common")?;
        let n = cubin::module(&d.ctx, "norm")?;
        let r = cubin::module(&d.ctx, "rope")?;
        let m = cubin::module(&d.ctx, "modulate")?;
        let v = cubin::module(&d.ctx, "vision_ops")?;
        Ok(Self {
            heads: spec.num_heads,
            head_dim: spec.model_channels / spec.num_heads,
            norm: q.load_function("q38_layer_norm_bf16_affine_f32")?,
            norm_f32: q.load_function("q38_layer_norm_f32")?,
            l2: n.load_function("l2_norm_scale_bf16")?,
            unpack: v.load_function("qkv_unpack_hm_bf16")?,
            rope: r.load_function("rope_pair_tail_rn")?,
            rope_hm: r.load_function("rope_pair_tail_hm")?,
            phases: r.load_function("rope_pair_rows_3d_direct")?,
            modulate: m.load_function("row_modulate")?,
            gated: m.load_function("row_gated_add")?,
            add_mod: m.load_function("add_row_bf16_f32")?,
            residual: n.load_function("residual")?,
            cast: v.load_function("vision_affine_cast_bf16")?,
            upcast: v.load_function("vision_upcast_f32")?,
            bias: v.load_function("vision_bias_f32")?,
            silu: v.load_function("vision_add_row_silu_f32")?,
            sinusoid: v.load_function("sinusoidal_embed_f32")?,
            gelu: q.load_function("q38_gelu")?,
            transpose: q.load_function("transpose_f32")?,
        })
    }
    fn norm(
        &self,
        d: &Device,
        input: u64,
        output: u64,
        shape: [usize; 2],
        affine: [u64; 2],
        arithmetic: (f32, bool),
    ) -> Result<()> {
        let [rows, width] = shape;
        let (eps, f32_io) = arithmetic;
        unsafe {
            d.stream
                .launch_builder(if f32_io { &self.norm_f32 } else { &self.norm })
                .arg(&input)
                .arg(&affine[0])
                .arg(&affine[1])
                .arg(&output)
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
    fn cast(&self, d: &Device, input: u64, output: u64, count: usize) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.cast)
                .arg(&input)
                .arg(&output)
                .arg(&(count as u64))
                .arg(&0f32)
                .arg(&1f32)
                .launch(flat(count))?;
        }
        Ok(())
    }
    fn upcast(&self, d: &Device, input: u64, output: u64, count: usize) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.upcast)
                .arg(&input)
                .arg(&output)
                .arg(&u32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
    fn l2(
        &self,
        d: &Device,
        input: u64,
        output: u64,
        gamma: u64,
        rows: usize,
        head_major: bool,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.l2)
                .arg(&input)
                .arg(&gamma)
                .arg(&output)
                .arg(&(rows as i32))
                .arg(&(self.heads as i32))
                .arg(&(self.head_dim as i32))
                .arg(&(self.head_dim as f32).sqrt())
                .arg(&1e-12f32)
                .arg(&(if head_major { rows as i32 } else { 0 }))
                .launch(grid(rows * self.heads, 32))?;
        }
        Ok(())
    }
    fn unpack(
        &self,
        d: &Device,
        input: u64,
        outputs: [u64; 3],
        rows: usize,
        parts: i32,
    ) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.unpack)
                .arg(&input)
                .arg(&outputs[0])
                .arg(&outputs[1])
                .arg(&outputs[2])
                .arg(&1u32)
                .arg(&(rows as u32))
                .arg(&(self.heads as u32))
                .arg(&(self.head_dim as u32))
                .arg(&parts)
                .launch(flat(rows * self.heads * self.head_dim))?;
        }
        Ok(())
    }
    fn residual(&self, d: &Device, x: u64, y: u64, count: usize) -> Result<()> {
        unsafe {
            d.stream
                .launch_builder(&self.residual)
                .arg(&x)
                .arg(&y)
                .arg(&i32::try_from(count)?)
                .launch(flat(count))?;
        }
        Ok(())
    }
}

struct Work {
    row_index: CudaSlice<u32>,
    hidden: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    kn: CudaSlice<bf16>,
    attention: CudaSlice<bf16>,
    ff: CudaSlice<bf16>,
    cross_kv: CudaSlice<bf16>,
    cross_k: CudaSlice<bf16>,
    cross_kn: CudaSlice<bf16>,
    cross_v: CudaSlice<bf16>,
    cross_q32: CudaSlice<f32>,
    cross_k32: CudaSlice<f32>,
    cross_v32: CudaSlice<f32>,
    cross_out32: CudaSlice<f32>,
    outer: CudaSlice<f32>,
    input_rows: CudaSlice<f32>,
    output_rows: CudaSlice<f32>,
    block_mod: CudaSlice<bf16>,
}
impl Work {
    fn new(d: &Device, n: usize, spec: &FlowSpec) -> Result<Self> {
        let s = &d.stream;
        let h = spec.model_channels;
        Ok(Self {
            row_index: s.alloc_zeros(n)?,
            hidden: s.alloc_zeros(n * h)?,
            norm: s.alloc_zeros(n * h)?,
            projected: s.alloc_zeros(n * h)?,
            qkv: s.alloc_zeros(n * 3 * h)?,
            q: s.alloc_zeros(n * h)?,
            k: s.alloc_zeros(n * h)?,
            v: s.alloc_zeros(n * h)?,
            kn: s.alloc_zeros(n * h)?,
            attention: s.alloc_zeros(n * h)?,
            ff: s.alloc_zeros(n * spec.feed_forward_width())?,
            cross_kv: s.alloc_zeros(10 * h)?,
            cross_k: s.alloc_zeros(5 * h)?,
            cross_kn: s.alloc_zeros(5 * h)?,
            cross_v: s.alloc_zeros(5 * h)?,
            cross_q32: s.alloc_zeros(n * h)?,
            cross_k32: s.alloc_zeros(5 * h)?,
            cross_v32: s.alloc_zeros(5 * h)?,
            cross_out32: s.alloc_zeros(n * h)?,
            outer: s.alloc_zeros(n * h)?,
            input_rows: s.alloc_zeros(n * spec.in_channels)?,
            output_rows: s.alloc_zeros(n * spec.in_channels)?,
            block_mod: s.alloc_zeros(6 * h)?,
        })
    }
}

fn validate_conditioning_origin(origin: &Rc<()>, flow: &Rc<()>) -> Result<()> {
    ensure!(
        Rc::ptr_eq(origin, flow),
        "conditioning originates from another flow"
    );
    Ok(())
}

pub struct Conditioning {
    origin: Rc<()>,
    global: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
}
pub struct PixalFlow {
    origin: Rc<()>,
    device: Device,
    spec: FlowSpec,
    rows: usize,
    coordinates: Vec<[i32; 3]>,
    steps: usize,
    _bf16: BTreeMap<String, CudaSlice<bf16>>,
    _f32: BTreeMap<String, CudaSlice<f32>>,
    blocks: Vec<Block>,
    input: Linear<f32>,
    output: Linear<f32>,
    time: [Linear<f32>; 3],
    kernels: Kernels,
    blas: Blas,
    flash: Flash,
    cross: F32Attention,
    work: Work,
    phases: CudaSlice<f32>,
    mods: CudaSlice<bf16>,
    time_count: usize,
}

impl PixalFlow {
    pub fn open(
        dir: &Path,
        stem: &str,
        runtime: &Path,
        coordinates: &[[i32; 3]],
        budget: usize,
    ) -> Result<Self> {
        let checkpoint = FlowCheckpoint::open(dir, stem)?;
        let spec = checkpoint.spec.clone();
        let pipeline = dir
            .parent()
            .context("checkpoint directory lacks model root")?
            .join(if stem.ends_with("_mv") {
                "pipeline_mv.json"
            } else {
                "pipeline.json"
            });
        let pipeline: serde_json::Value = serde_json::from_slice(&std::fs::read(pipeline)?)?;
        let sampler = &pipeline["args"][if spec.sparse() {
            "shape_slat_sampler"
        } else {
            "sparse_structure_sampler"
        }];
        let parameters = &sampler["params"];
        let steps = usize::try_from(
            parameters["steps"]
                .as_u64()
                .context("sampler steps missing")?,
        )?;
        ensure!(
            sampler["name"] == "FlowEulerGuidanceIntervalSampler"
                && sampler["args"]["sigma_min"] == 1e-5
                && steps == 12
                && parameters["guidance_strength"] == 7.5
                && parameters["guidance_interval"] == serde_json::json!([0.6, 1.0])
                && parameters["guidance_rescale"] == if spec.sparse() { 0.5 } else { 0.7 }
                && parameters["rescale_t"] == if spec.sparse() { 3.0 } else { 5.0 },
            "unsupported published flow sampler contract"
        );
        let rows = coordinates.len();
        ensure!(
            rows > 0 && rows <= i32::MAX as usize / (3 * spec.model_channels),
            "flow row extent overflow"
        );
        ensure!(
            coordinates.iter().flatten().all(|&v| v >= 0),
            "negative flow coordinates"
        );
        ensure!(
            coordinates
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == rows,
            "duplicate flow coordinates"
        );
        if !spec.sparse() {
            ensure!(
                rows == 4096
                    && coordinates
                        .iter()
                        .enumerate()
                        .all(|(i, p)| *p
                            == [(i / 256) as i32, ((i / 16) % 16) as i32, (i % 16) as i32]),
                "SS coordinates must be the published 16^3 grid order"
            );
        }
        let device = Device::new(0)?;
        let (free, total) = device.ctx.mem_get_info()?;
        ensure!(
            budget > 0 && budget <= total,
            "invalid admitted flow device budget"
        );
        let kernels = Kernels::load(&device, &spec)?;
        let ops = Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "Pixal3D flow".into())),
        )?;
        let mut bf = BTreeMap::new();
        let mut fp = BTreeMap::new();
        let tensors = checkpoint.mapped.component_tensors()?;
        for (name, (_, use_bf16)) in spec.inventory() {
            let bytes = tensors[&name].data;
            if use_bf16 {
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| bf16::from_f32(f32::from_le_bytes(*b)))
                    .collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite weight: {name}"
                );
                bf.insert(name, device.stream.clone_htod(&values)?);
            } else {
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "non-finite weight: {name}"
                );
                fp.insert(name, device.stream.clone_htod(&values)?);
            }
        }
        let mut blocks = Vec::new();
        let h = spec.model_channels;
        for i in 0..spec.num_blocks {
            let p = format!("blocks.{i}");
            let linear = |name: &str, output, input| {
                Linear::bind(&bf, &device, &format!("{p}.{name}"), [output, input])
            };
            let weight = |name: &str| ptr(&fp[&format!("{p}.{name}")], &device);
            let b = Block {
                qkv: linear("self_attn.to_qkv", 3 * h, h)?,
                self_out: linear("self_attn.to_out", h, h)?,
                cross_q: linear("cross_attn.cross_attn_block.to_q", h, h)?,
                cross_kv: linear(
                    "cross_attn.cross_attn_block.to_kv",
                    2 * h,
                    spec.cond_channels,
                )?,
                cross_out: linear("cross_attn.cross_attn_block.to_out", h, h)?,
                projection: linear("cross_attn.proj_linear", h, spec.projection_width())?,
                fc1: linear("mlp.mlp.0", spec.feed_forward_width(), h)?,
                fc2: linear("mlp.mlp.2", h, spec.feed_forward_width())?,
                norm_weight: weight("norm2.weight"),
                norm_bias: weight("norm2.bias"),
                modulation: weight("modulation"),
                gamma: [
                    weight("self_attn.q_rms_norm.gamma"),
                    weight("self_attn.k_rms_norm.gamma"),
                    weight("cross_attn.cross_attn_block.q_rms_norm.gamma"),
                    weight("cross_attn.cross_attn_block.k_rms_norm.gamma"),
                ],
            };
            for l in [
                b.qkv,
                b.self_out,
                b.cross_q,
                b.cross_out,
                b.projection,
                b.fc1,
                b.fc2,
            ] {
                l.prepare(&mut blas, rows)?;
            }
            b.cross_kv.prepare(&mut blas, 5)?;
            blocks.push(b);
        }
        let input = Linear::bind(&fp, &device, "input_layer", [h, spec.in_channels])?;
        let output = Linear::bind(&fp, &device, "out_layer", [spec.out_channels, h])?;
        let time = [
            Linear::bind(
                &fp,
                &device,
                "t_embedder.mlp.0",
                [h, tensors["t_embedder.mlp.0.weight"].shape[1]],
            )?,
            Linear::bind(&fp, &device, "t_embedder.mlp.2", [h, h])?,
            Linear::bind(&fp, &device, "adaLN_modulation.1", [6 * h, h])?,
        ];
        let heads = spec.num_heads;
        let dim = h / heads;
        let phases = if spec.sparse() {
            let values: Vec<_> = coordinates
                .iter()
                .flat_map(|p| p.iter().map(|&x| x as f32))
                .collect();
            let points = device.stream.clone_htod(&values)?;
            let freqs: Vec<f32> = (0..dim / 2)
                .map(|i| {
                    if i >= 3 * (dim / 6) {
                        0.0
                    } else {
                        1.0 / 10000f32.powf((i % (dim / 6)) as f32 / (dim / 6) as f32)
                    }
                })
                .collect();
            let freqs = device.stream.clone_htod(&freqs)?;
            let phase = device.stream.alloc_zeros(rows * dim)?;
            unsafe {
                device
                    .stream
                    .launch_builder(&kernels.phases)
                    .arg(&points)
                    .arg(&freqs)
                    .arg(&(rows as i32))
                    .arg(&1i32)
                    .arg(&(dim as i32))
                    .arg(&phase)
                    .launch(flat(rows * dim / 2))?;
            }
            phase
        } else {
            fp.remove("rope_phases").context("SS phases missing")?
        };
        let work = Work::new(&device, rows, &spec)?;
        let flash = Flash::new(device.stream.clone(), rows, heads, heads, dim)?;
        let mods = device.stream.alloc_zeros(steps * 6 * h)?;
        let condition_bytes = 4usize
            .checked_mul(5 * spec.cond_channels + rows * spec.projection_width())
            .context("flow conditioning bytes overflow")?;
        let sampler_bytes = rows
            .checked_mul(spec.in_channels)
            .and_then(|n| n.checked_mul(8 * size_of::<f32>()))
            .and_then(|n| n.checked_add(8))
            .context("flow sampler bytes overflow")?;
        let time_width = time[0].weight.shape[1] as usize;
        let time_bytes = size_of::<f32>() * (steps * (time_width + 8 * h + 1) + h + time_width / 2);
        let live_bytes = condition_bytes
            .checked_add(sampler_bytes)
            .and_then(|n| n.checked_add(time_bytes))
            .context("flow live bytes overflow")?;
        let cross_budget = device.workspace_budget(free, budget, live_bytes)?;
        let cross = F32Attention::new(
            device.stream.clone(),
            [heads as u32, rows as u32, 5, dim as u32, dim as u32],
            cross_budget,
        )?;
        device.workspace_budget(free, budget, live_bytes)?;
        Ok(Self {
            origin: Rc::new(()),
            device,
            spec,
            rows,
            coordinates: coordinates.to_vec(),
            steps,
            _bf16: bf,
            _f32: fp,
            blocks,
            input,
            output,
            time,
            kernels,
            blas,
            flash,
            cross,
            work,
            phases,
            mods,
            time_count: 0,
        })
    }
    fn linear_f32(
        &self,
        linear: Linear<f32>,
        rows: usize,
        input: u64,
        output: u64,
        silu: bool,
    ) -> Result<()> {
        self.blas.linear_f32(
            linear.weight,
            matrix(input, rows, linear.weight.shape[1] as usize)?,
            &mut TensorMut(matrix(output, rows, linear.weight.shape[0] as usize)?),
        )?;
        let width = linear.weight.shape[0];
        unsafe {
            let mut launch = self.device.stream.launch_builder(if silu {
                &self.kernels.silu
            } else {
                &self.kernels.bias
            });
            launch.arg(&output).arg(&linear.bias.ptr).arg(&output);
            launch
                .arg(
                    &(if silu {
                        rows as u32 * width
                    } else {
                        rows as u32
                    }),
                )
                .arg(&width)
                .launch(flat(rows * width as usize))?;
        }
        Ok(())
    }
    fn set_times(&mut self, times: &[f32]) -> Result<()> {
        ensure!(
            !times.is_empty() && times.len() <= self.steps && times.iter().all(|x| x.is_finite()),
            "invalid flow time setup"
        );
        let d = &self.device;
        let n = times.len();
        let h = self.spec.model_channels;
        let half = self.time[0].weight.shape[1] as usize / 2;
        let t = d.stream.clone_htod(times)?;
        let freq: Vec<_> = (0..half)
            .map(|i| (-10000f64.ln() as f32 * i as f32 / half as f32).exp())
            .collect();
        let freq = d.stream.clone_htod(&freq)?;
        let embed = d.stream.alloc_zeros::<f32>(n * half * 2)?;
        let first = d.stream.alloc_zeros::<f32>(n * h)?;
        let second = d.stream.alloc_zeros::<f32>(n * h)?;
        let all = d.stream.alloc_zeros::<f32>(n * 6 * h)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.sinusoid)
                .arg(&t)
                .arg(&freq)
                .arg(&embed)
                .arg(&(n as u32))
                .arg(&(half as u32))
                .arg(&0i32)
                .launch(flat(n * half))?;
        }
        self.linear_f32(self.time[0], n, ptr(&embed, d), ptr(&first, d), true)?;
        self.linear_f32(self.time[1], n, ptr(&first, d), ptr(&second, d), false)?;
        let zero = d.stream.alloc_zeros::<f32>(h)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.silu)
                .arg(&second)
                .arg(&zero)
                .arg(&second)
                .arg(&((n * h) as u32))
                .arg(&(h as u32))
                .launch(flat(n * h))?;
        }
        self.linear_f32(self.time[2], n, ptr(&second, d), ptr(&all, d), false)?;
        self.kernels
            .cast(d, ptr(&all, d), ptr(&self.mods, d), n * 6 * h)?;
        self.time_count = n;
        Ok(())
    }
    pub fn conditioning(
        &self,
        global: &[f32],
        projected: &[f32],
        coordinates: &[[i32; 3]],
    ) -> Result<Conditioning> {
        ensure!(
            coordinates == self.coordinates,
            "conditioning coordinate rows differ from flow setup"
        );
        ensure!(
            global.len() == 5 * self.spec.cond_channels
                && projected.len() == self.rows * self.spec.projection_width(),
            "flow conditioning shape mismatch"
        );
        ensure!(
            global.iter().chain(projected).all(|x| x.is_finite()),
            "non-finite conditioning"
        );
        let cast = |values: &[f32]| -> Result<_> {
            let values: Vec<_> = values.iter().map(|&v| bf16::from_f32(v)).collect();
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "conditioning overflows BF16"
            );
            Ok(self.device.stream.clone_htod(&values)?)
        };
        Ok(Conditioning {
            origin: self.origin.clone(),
            global: cast(global)?,
            projected: cast(projected)?,
        })
    }
    fn block(&mut self, index: usize, modulation: u64, conditioning: &Conditioning) -> Result<()> {
        let d = &self.device;
        let n = self.rows;
        let h = self.spec.model_channels;
        let heads = self.spec.num_heads;
        let dim = h / heads;
        let ff = self.spec.feed_forward_width();
        let b = &self.blocks[index];
        let k = &self.kernels;
        let hidden = ptr(&self.work.hidden, d);
        let norm = ptr(&self.work.norm, d);
        let projected = ptr(&self.work.projected, d);
        let q = ptr(&self.work.q, d);
        let key = ptr(&self.work.k, d);
        let value = ptr(&self.work.v, d);
        let kn = ptr(&self.work.kn, d);
        let attention = ptr(&self.work.attention, d);
        let bm = ptr(&self.work.block_mod, d);
        unsafe {
            d.stream
                .launch_builder(&k.add_mod)
                .arg(&modulation)
                .arg(&b.modulation)
                .arg(&bm)
                .arg(&((6 * h) as u64))
                .arg(&((6 * h) as u32))
                .launch(flat(6 * h))?;
        }
        k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
        self.modulate(norm, bm, 0)?;
        b.qkv.run(&self.blas, n, norm, ptr(&self.work.qkv, d))?;
        k.unpack(d, ptr(&self.work.qkv, d), [q, key, value], n, 3)?;
        k.l2(d, q, q, b.gamma[0], n, false)?;
        k.l2(d, key, kn, b.gamma[1], n, true)?;
        for (kernel, data) in [(&k.rope, q), (&k.rope_hm, kn)] {
            unsafe {
                d.stream
                    .launch_builder(kernel)
                    .arg(&data)
                    .arg(&self.phases)
                    .arg(&(n as i32))
                    .arg(&(heads as i32))
                    .arg(&(dim as i32))
                    .arg(&(dim as i32))
                    .launch(flat(n * heads * dim / 2))?;
            }
        }
        self.flash.run(
            &self.work.q,
            h,
            &self.work.kn,
            &self.work.v,
            &mut self.work.attention,
            Shape {
                rows: n,
                start: 0,
                capacity: n,
                q_heads: heads,
                kv_heads: heads,
                head_dim: dim,
                causal: false,
                window_left: None,
            },
        )?;
        b.self_out.run(&self.blas, n, attention, projected)?;
        self.gated_add(projected, bm, 2)?;
        k.norm(
            d,
            hidden,
            norm,
            [n, h],
            [b.norm_weight, b.norm_bias],
            (1e-6, false),
        )?;
        b.cross_q.run(&self.blas, n, norm, q)?;
        b.cross_kv.run(
            &self.blas,
            5,
            ptr(&conditioning.global, d),
            ptr(&self.work.cross_kv, d),
        )?;
        k.unpack(
            d,
            ptr(&self.work.cross_kv, d),
            [0, ptr(&self.work.cross_k, d), ptr(&self.work.cross_v, d)],
            5,
            2,
        )?;
        k.l2(d, q, q, b.gamma[2], n, false)?;
        k.l2(
            d,
            ptr(&self.work.cross_k, d),
            ptr(&self.work.cross_kn, d),
            b.gamma[3],
            5,
            true,
        )?;
        k.upcast(d, q, ptr(&self.work.cross_q32, d), n * h)?;
        k.upcast(
            d,
            ptr(&self.work.cross_kn, d),
            ptr(&self.work.cross_k32, d),
            5 * h,
        )?;
        k.upcast(
            d,
            ptr(&self.work.cross_v, d),
            ptr(&self.work.cross_v32, d),
            5 * h,
        )?;
        let q32 = view(
            &self.work.cross_q32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        let k32 = dense(&self.work.cross_k32, d, [heads as u32, 5, dim as u32])?;
        let v32 = dense(&self.work.cross_v32, d, [heads as u32, 5, dim as u32])?;
        let out32 = view(
            &self.work.cross_out32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        self.cross
            .run(&self.blas, q32, k32, v32, &mut TensorMut(out32))?;
        k.cast(d, ptr(&self.work.cross_out32, d), attention, n * h)?;
        b.cross_out.run(&self.blas, n, attention, projected)?;
        b.projection
            .run(&self.blas, n, ptr(&conditioning.projected, d), norm)?;
        k.residual(d, norm, projected, n * h)?;
        k.residual(d, hidden, norm, n * h)?;
        k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
        self.modulate(norm, bm, 3)?;
        b.fc1.run(&self.blas, n, norm, ptr(&self.work.ff, d))?;
        unsafe {
            d.stream
                .launch_builder(&k.gelu)
                .arg(&self.work.ff)
                .arg(&self.work.ff)
                .arg(&(n as u32))
                .arg(&1u32)
                .arg(&(ff as u32))
                .arg(&(ff as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&(ff as u64))
                .arg(&0u64)
                .arg(&1u64)
                .arg(&1i32)
                .launch(grid((n * ff).div_ceil(256).min(65535), 256))?;
        }
        b.fc2.run(&self.blas, n, ptr(&self.work.ff, d), projected)?;
        self.gated_add(projected, bm, 5)?;
        Ok(())
    }
    fn modulate(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.spec.model_channels;
        let shift = modulation + (part * h * 2) as u64;
        let scale = shift + (h * 2) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.modulate)
                .arg(&values)
                .arg(&shift)
                .arg(&scale)
                .arg(&self.work.row_index)
                .arg(&values)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    fn gated_add(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.spec.model_channels;
        let gate = modulation + (part * h * 2) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.gated)
                .arg(&self.work.hidden)
                .arg(&values)
                .arg(&gate)
                .arg(&self.work.row_index)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(crate::backend::gpu::LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
    fn forward(
        &mut self,
        input: &CudaSlice<f32>,
        time: usize,
        conditioning: &Conditioning,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.forward_observe(input, time, conditioning, output, |_, _| Ok(()))
    }
    fn forward_observe(
        &mut self,
        input: &CudaSlice<f32>,
        time: usize,
        conditioning: &Conditioning,
        output: &mut CudaSlice<f32>,
        mut observe: impl FnMut(&Self, usize) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            time < self.time_count
                && input.len() == self.rows * self.spec.in_channels
                && output.len() == input.len(),
            "flow input/time extent differs from setup"
        );
        validate_conditioning_origin(&conditioning.origin, &self.origin)?;
        ensure!(
            conditioning.global.len() == 5 * self.spec.cond_channels
                && conditioning.projected.len() == self.rows * self.spec.projection_width(),
            "conditioning belongs to different flow geometry"
        );
        let n = self.rows;
        let h = self.spec.model_channels;
        let input_ptr = if self.spec.sparse() {
            ptr(input, &self.device)
        } else {
            unsafe {
                self.device
                    .stream
                    .launch_builder(&self.kernels.transpose)
                    .arg(input)
                    .arg(&self.work.input_rows)
                    .arg(&(self.spec.in_channels as i64))
                    .arg(&(n as i64))
                    .launch(flat(input.len()))?;
            }
            ptr(&self.work.input_rows, &self.device)
        };
        self.linear_f32(
            self.input,
            n,
            input_ptr,
            ptr(&self.work.outer, &self.device),
            false,
        )?;
        self.kernels.cast(
            &self.device,
            ptr(&self.work.outer, &self.device),
            ptr(&self.work.hidden, &self.device),
            n * h,
        )?;
        let modulation = ptr(&self.mods, &self.device) + (time * 6 * h * 2) as u64;
        for i in 0..self.blocks.len() {
            self.block(i, modulation, conditioning)?;
            observe(self, i)?;
        }
        self.kernels.upcast(
            &self.device,
            ptr(&self.work.hidden, &self.device),
            ptr(&self.work.outer, &self.device),
            n * h,
        )?;
        self.kernels.norm(
            &self.device,
            ptr(&self.work.outer, &self.device),
            ptr(&self.work.outer, &self.device),
            [n, h],
            [0, 0],
            (1e-5, true),
        )?;
        let out = if self.spec.sparse() {
            ptr(output, &self.device)
        } else {
            ptr(&self.work.output_rows, &self.device)
        };
        self.linear_f32(
            self.output,
            n,
            ptr(&self.work.outer, &self.device),
            out,
            false,
        )?;
        if !self.spec.sparse() {
            unsafe {
                self.device
                    .stream
                    .launch_builder(&self.kernels.transpose)
                    .arg(&out)
                    .arg(&*output)
                    .arg(&(n as i64))
                    .arg(&(self.spec.out_channels as i64))
                    .launch(flat(output.len()))?;
            }
        }
        Ok(())
    }
}

struct Sampler {
    state: CudaSlice<f32>,
    positive: CudaSlice<f32>,
    negative: CudaSlice<f32>,
    velocity: CudaSlice<f32>,
    x0: CudaSlice<f32>,
    positive_x0: CudaSlice<f32>,
    std: CudaSlice<f32>,
    guided: CudaFunction,
    convert: CudaFunction,
    std_kernel: CudaFunction,
    rescale: CudaFunction,
    euler: CudaFunction,
}
impl Sampler {
    fn new(d: &Device, noise: &[f32]) -> Result<Self> {
        ensure!(
            noise.len() > 1 && noise.iter().all(|x| x.is_finite()),
            "invalid flow noise"
        );
        let common = cubin::module(&d.ctx, "q38_common")?;
        let flow = cubin::module(&d.ctx, "flow")?;
        Ok(Self {
            state: d.stream.clone_htod(noise)?,
            positive: d.stream.alloc_zeros(noise.len())?,
            negative: d.stream.alloc_zeros(noise.len())?,
            velocity: d.stream.alloc_zeros(noise.len())?,
            x0: d.stream.alloc_zeros(noise.len())?,
            positive_x0: d.stream.alloc_zeros(noise.len())?,
            std: d.stream.alloc_zeros(2)?,
            guided: common.load_function("guided_euler_f32")?,
            convert: flow.load_function("flow_x0_f32")?,
            std_kernel: cubin::module(&d.ctx, "norm")?.load_function("sample_std_f32")?,
            rescale: flow.load_function("flow_rescale_f32")?,
            euler: flow.load_function("euler_blend_f32")?,
        })
    }
    fn convert(&self, d: &Device, input: u64, output: u64, t: f64, inverse: bool) -> Result<()> {
        let alpha = (1.0f64 - 1e-5) as f32;
        let beta = (1e-5 + (1.0 - 1e-5) * t) as f32;
        unsafe {
            d.stream
                .launch_builder(&self.convert)
                .arg(&self.state)
                .arg(&input)
                .arg(&output)
                .arg(&(self.state.len() as u64))
                .arg(&alpha)
                .arg(&beta)
                .arg(&(inverse as i32))
                .launch(flat(self.state.len()))?;
        }
        Ok(())
    }
    fn step(&mut self, d: &Device, t: f64, next: f64, sparse: bool) -> Result<()> {
        ensure!(
            t.is_finite() && next.is_finite() && next >= 0.0 && next < t && t <= 1.0,
            "invalid Euler time pair"
        );
        let count = self.state.len();
        if (0.6..=1.0).contains(&t) {
            unsafe {
                d.stream
                    .launch_builder(&self.guided)
                    .arg(&0u64)
                    .arg(&self.positive)
                    .arg(&self.negative)
                    .arg(&7.5f32)
                    .arg(&0f32)
                    .arg(&(count as i64))
                    .arg(&1i32)
                    .arg(&self.velocity)
                    .launch(flat(count))?;
            }
            self.convert(
                d,
                ptr(&self.positive, d),
                ptr(&self.positive_x0, d),
                t,
                false,
            )?;
            self.convert(d, ptr(&self.velocity, d), ptr(&self.x0, d), t, false)?;
            for (i, input) in [&self.positive_x0, &self.x0].into_iter().enumerate() {
                let output = ptr(&self.std, d) + (i * 4) as u64;
                unsafe {
                    d.stream
                        .launch_builder(&self.std_kernel)
                        .arg(input)
                        .arg(&output)
                        .arg(&(count as u64))
                        .arg(&(sparse as i32))
                        .launch(grid(1, 256))?;
                }
            }
            let std_pos = ptr(&self.std, d);
            let std_cfg = std_pos + 4;
            unsafe {
                d.stream
                    .launch_builder(&self.rescale)
                    .arg(&self.x0)
                    .arg(&std_pos)
                    .arg(&std_cfg)
                    .arg(&(count as u64))
                    .arg(&(if sparse { 0.5f32 } else { 0.7f32 }))
                    .launch(flat(count))?;
            }
            self.convert(d, ptr(&self.x0, d), ptr(&self.velocity, d), t, true)?;
        } else {
            d.stream.memcpy_dtod(&self.positive, &mut self.velocity)?;
        }
        unsafe {
            d.stream
                .launch_builder(&self.euler)
                .arg(&self.state)
                .arg(&self.velocity)
                .arg(&(count as u64))
                .arg(&(-(t - next) as f32))
                .arg(&0f32)
                .arg(&1f32)
                .launch(flat(count))?;
        }
        Ok(())
    }
}

fn schedule(sparse: bool, steps: usize) -> Vec<f64> {
    let scale = if sparse { 3.0 } else { 5.0 };
    (0..=steps)
        .map(|i| {
            let t = if i == steps {
                0.0
            } else {
                1.0 + i as f64 * (-1.0 / steps as f64)
            };
            scale * t / (1.0 + (scale - 1.0) * t)
        })
        .collect()
}
impl PixalFlow {
    pub fn sample(
        &mut self,
        noise: &[f32],
        positive: &Conditioning,
        negative: &Conditioning,
    ) -> Result<Vec<f32>> {
        validate_conditioning_origin(&positive.origin, &self.origin)?;
        validate_conditioning_origin(&negative.origin, &self.origin)?;
        ensure!(
            noise.len() == self.rows * self.spec.in_channels,
            "noise extent differs from flow geometry"
        );
        let times = schedule(self.spec.sparse(), self.steps);
        let model_times: Vec<_> = times[..self.steps]
            .iter()
            .map(|t| (1000.0 * t) as f32)
            .collect();
        self.set_times(&model_times)?;
        let mut sampler = Sampler::new(&self.device, noise)?;
        for step in 0..self.steps {
            self.forward(&sampler.state, step, positive, &mut sampler.positive)?;
            if (0.6..=1.0).contains(&times[step]) {
                self.forward(&sampler.state, step, negative, &mut sampler.negative)?;
            }
            sampler.step(
                &self.device,
                times[step],
                times[step + 1],
                self.spec.sparse(),
            )?;
        }
        Ok(self.device.stream.clone_dtoh(&sampler.state)?)
    }
}

/// Published per-channel latent denormalization, F32 multiply then F32 add.
#[cfg(test)]
fn denormalize_shape(
    d: &Device,
    values: &mut CudaSlice<f32>,
    mean: &[f32],
    std: &[f32],
    channels: usize,
) -> Result<()> {
    ensure!(
        channels > 0
            && mean.len() == channels
            && std.len() == channels
            && !values.is_empty()
            && values.len().is_multiple_of(channels),
        "shape latent normalization extent mismatch"
    );
    ensure!(
        mean.iter().all(|x| x.is_finite()) && std.iter().all(|x| x.is_finite() && *x > 0.0),
        "invalid shape normalization coefficients"
    );
    let mean = d.stream.clone_htod(mean)?;
    let std = d.stream.clone_htod(std)?;
    let kernel = cubin::module(&d.ctx, "q38_common")?.load_function("channel_affine_f32")?;
    unsafe {
        d.stream
            .launch_builder(&kernel)
            .arg(&ptr(values, d))
            .arg(&std)
            .arg(&mean)
            .arg(&ptr(values, d))
            .arg(&u32::try_from(values.len() / channels)?)
            .arg(&u32::try_from(channels)?)
            .launch(flat(values.len()))?;
    }
    Ok(())
}
