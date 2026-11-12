use super::Device;
use super::blas::Blas;
use super::contracts::{TensorMut, view};
use super::flash::{F32Attention, F32Mask};
use super::ops::{flat, grid};
use super::pixal3d::{Block, Kernels, Linear, dense, ptr, run_linear_f32};
use crate::backend::gpu::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::pixal3d::FlowSpec;
use anyhow::{Context as _, Result, ensure};
use half::bf16;

/// Per-row rotary phases for the dense [dz, dy, dx] grid, stored interleaved
/// cos/sin [rows, dim]. Frequency-axis 0 is the slowest-varying grid
/// coordinate: the model builds rope_phases from meshgrid coords whose first
/// column is the slowest axis (sparse_structure_flow.py:108-113).
fn rope_freqs(freq_dim: usize) -> Vec<f32> {
    (0..freq_dim)
        .map(|i| (-10000f64.ln() as f32 * i as f32 / freq_dim as f32).exp())
        .collect()
}

/// Per-site rotary phases for sparse integer [z, y, x] coordinates, stored
/// interleaved cos/sin [sites, dim]. Frequency-axis 0 is the slowest grid
/// coordinate, matching the model's per-token phase buffer.
fn rope_phase_sites(coords: &[[i32; 3]], dim: usize, freq_dim: usize) -> Vec<f32> {
    let freqs = rope_freqs(freq_dim);
    let mut phase_values = vec![0f32; coords.len() * dim];
    for (row, c) in coords.iter().enumerate() {
        let site = [c[0] as f32, c[1] as f32, c[2] as f32];
        for pair in 0..dim / 2 {
            let base = row * dim + 2 * pair;
            if pair < 3 * freq_dim {
                let angle = site[pair / freq_dim] * freqs[pair % freq_dim];
                phase_values[base] = angle.cos();
                phase_values[base + 1] = angle.sin();
            } else {
                phase_values[base] = 1.0;
                phase_values[base + 1] = 0.0;
            }
        }
    }
    phase_values
}

pub struct TrellisSsFlow {
    device: Device,
    kernels: Kernels,
    _weights: super::pixal3d::FlowWeights<bf16>,
    _float_weights: super::pixal3d::FlowWeights<f32>,
    blocks: Vec<Block<bf16>>,
    input: Linear<f32>,
    output: Linear<f32>,
    time: [Linear<f32>; 3],
    flash: F32Attention,
    self_q32: CudaSlice<f32>,
    self_k32: CudaSlice<f32>,
    self_v32: CudaSlice<f32>,
    self_out32: CudaSlice<f32>,
    cross: F32Attention,
    cross_q32: CudaSlice<f32>,
    cross_k32: CudaSlice<f32>,
    cross_v32: CudaSlice<f32>,
    cross_out32: CudaSlice<f32>,
    rows: usize,
    context_rows: usize,
    heads: usize,
    dim: usize,
    width: usize,
    cond_width: usize,
    in_channels: usize,
    feed_forward: usize,
    phases: cudarc::driver::CudaSlice<f32>,
    context: cudarc::driver::CudaSlice<bf16>,
    hidden: cudarc::driver::CudaSlice<bf16>,
    norm: cudarc::driver::CudaSlice<bf16>,
    projected: cudarc::driver::CudaSlice<bf16>,
    qkv: cudarc::driver::CudaSlice<bf16>,
    q: cudarc::driver::CudaSlice<bf16>,
    k: cudarc::driver::CudaSlice<bf16>,
    v: cudarc::driver::CudaSlice<bf16>,
    kn: cudarc::driver::CudaSlice<bf16>,
    attention: cudarc::driver::CudaSlice<bf16>,
    block_mod: cudarc::driver::CudaSlice<bf16>,
    cross_kv: cudarc::driver::CudaSlice<bf16>,
    cross_k: cudarc::driver::CudaSlice<bf16>,
    cross_v: cudarc::driver::CudaSlice<bf16>,
    cross_kn: cudarc::driver::CudaSlice<bf16>,
    ff: cudarc::driver::CudaSlice<bf16>,
    input_x: CudaSlice<f32>,
    time_embed: cudarc::driver::CudaSlice<f32>,
    mods: cudarc::driver::CudaSlice<bf16>,
    time_out: cudarc::driver::CudaSlice<f32>,
    identity: CudaSlice<u32>,
    outer: CudaSlice<f32>,
    velocity_out: CudaSlice<f32>,
    capture: Vec<CudaSlice<bf16>>,
}

impl TrellisSsFlow {
    /// Loads a flow on the dense [16, 16, 16] sparse-structure grid.
    pub fn load(
        device: &Device,
        blas: &mut Blas,
        checkpoint: &std::path::Path,
        context_rows: usize,
        limit: usize,
    ) -> Result<Self> {
        let mut sites = Vec::with_capacity(4096);
        for z in 0..16i32 {
            for y in 0..16i32 {
                for x in 0..16i32 {
                    sites.push([z, y, x]);
                }
            }
        }
        Self::load_flow(device, blas, checkpoint, &sites, context_rows, limit)
    }

    /// Loads a published TRELLIS.2 flow over sparse [z, y, x] sites. The
    /// sibling config must declare the published DIT family; windowed
    /// attention fields are rejected (the definition compiles full
    /// attention only for these checkpoints).
    pub fn load_flow(
        device: &Device,
        blas: &mut Blas,
        checkpoint: &std::path::Path,
        sites: &[[i32; 3]],
        context_rows: usize,
        limit: usize,
    ) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(checkpoint.with_extension("json"))?)?;
        let args = &config["args"];
        ensure!(
            config["name"] == "SparseStructureFlowModel" || config["name"] == "SLatFlowModel",
            "unsupported flow model family"
        );
        ensure!(
            args.get("attn_mode").is_none()
                && args.get("window_size").is_none()
                && args.get("shift_window").is_none(),
            "windowed attention config not supported"
        );
        let read = |name: &str| -> Result<usize> {
            usize::try_from(args[name].as_u64().context("flow config field missing")?)
                .context("flow config field overflow")
        };
        let spec = FlowSpec {
            resolution: read("resolution")?,
            in_channels: read("in_channels")?,
            out_channels: read("out_channels")?,
            model_channels: 1536,
            cond_channels: 1024,
            num_blocks: 30,
            num_heads: 12,
            mlp_ratio: 5.3334,
            pe_mode: "rope".into(),
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
            image_attn_mode: "full".into(),
            dtype: "bfloat16".into(),
            proj_in_channels: None,
        };
        ensure!(
            args["model_channels"] == 1536
                && args["cond_channels"] == 1024
                && args["num_blocks"] == 30
                && args["num_heads"] == 12
                && args["mlp_ratio"] == 5.3334
                && args["pe_mode"] == "rope"
                && args["share_mod"] == true
                && args["qk_rms_norm"] == true
                && args["qk_rms_norm_cross"] == true
                && args["dtype"] == "bfloat16",
            "flow config differs from the published DIT family"
        );
        ensure!(
            !sites.is_empty()
                && sites
                    .iter()
                    .all(|c| c.iter().all(|&v| v >= 0 && (v as usize) < spec.resolution)),
            "flow sites outside the config resolution"
        );
        device.finish_upload()?;
        let (free_before, total) = device.ctx.mem_get_info()?;
        ensure!(
            limit > 0 && limit <= total,
            "invalid TRELLIS.2 SS flow device limit"
        );
        let mapped = crate::checkpoint::MappedCheckpoint::open_component(
            checkpoint
                .parent()
                .context("SS checkpoint parent missing")?,
            checkpoint
                .file_stem()
                .and_then(|s| s.to_str())
                .context("SS checkpoint stem missing")?,
        )?;
        let mut bf = super::pixal3d::FlowWeights::<bf16>::new();
        let mut fp = super::pixal3d::FlowWeights::<f32>::new();
        for (name, tensor) in mapped.component_tensors()? {
            let raw = tensor.data;
            match tensor.dtype.as_str() {
                "BF16" => {
                    let values: Vec<_> = raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
                        .collect();
                    bf.insert(name.clone(), device.stream.clone_htod(&values)?);
                }
                "F32" => {
                    let values: Vec<_> = raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    fp.insert(name.clone(), device.stream.clone_htod(&values)?);
                }
                other => anyhow::bail!("unexpected SS flow tensor dtype {other}: {name}"),
            }
        }
        let upcast = |bf: &super::pixal3d::FlowWeights<bf16>,
                      fp: &mut super::pixal3d::FlowWeights<f32>,
                      name: &str|
         -> Result<()> {
            ensure!(bf.contains_key(name), "SS tensor missing: {name}");
            let host = device.stream.clone_dtoh(&bf[name])?;
            let values: Vec<f32> = host.iter().map(|b| f32::from(*b)).collect();
            fp.insert(name.to_string(), device.stream.clone_htod(&values)?);
            Ok(())
        };
        for name in [
            "input_layer",
            "out_layer",
            "t_embedder.mlp.0",
            "t_embedder.mlp.2",
            "adaLN_modulation.1",
        ] {
            upcast(&bf, &mut fp, &format!("{name}.weight"))?;
            upcast(&bf, &mut fp, &format!("{name}.bias"))?;
        }
        for index in 0..spec.num_blocks {
            for name in [
                "norm2.weight",
                "norm2.bias",
                "modulation",
                "self_attn.q_rms_norm.gamma",
                "self_attn.k_rms_norm.gamma",
                "cross_attn.q_rms_norm.gamma",
                "cross_attn.k_rms_norm.gamma",
            ] {
                upcast(&bf, &mut fp, &format!("blocks.{index}.{name}"))?;
            }
        }
        let h = spec.model_channels;
        let rows = sites.len();
        let mut blocks = Vec::with_capacity(spec.num_blocks);
        for index in 0..spec.num_blocks {
            let p = format!("blocks.{index}");
            let linear = |fp: &super::pixal3d::FlowWeights<bf16>, name: &str, output, input| {
                Linear::bind(fp, device, &format!("{p}.{name}"), [output, input])
            };
            let weight = |fp: &super::pixal3d::FlowWeights<f32>, name: &str| {
                ptr(&fp[&format!("{p}.{name}")], device)
            };
            let block = Block {
                qkv: linear(&bf, "self_attn.to_qkv", 3 * h, h)?,
                self_out: linear(&bf, "self_attn.to_out", h, h)?,
                cross_q: linear(&bf, "cross_attn.to_q", h, h)?,
                cross_kv: linear(&bf, "cross_attn.to_kv", 2 * h, spec.cond_channels)?,
                cross_out: linear(&bf, "cross_attn.to_out", h, h)?,
                projection: None,
                fc1: linear(&bf, "mlp.mlp.0", spec.feed_forward_width(), h)?,
                fc2: linear(&bf, "mlp.mlp.2", h, spec.feed_forward_width())?,
                norm_weight: weight(&fp, "norm2.weight"),
                norm_bias: weight(&fp, "norm2.bias"),
                modulation: weight(&fp, "modulation"),
                gamma: [
                    weight(&fp, "self_attn.q_rms_norm.gamma"),
                    weight(&fp, "self_attn.k_rms_norm.gamma"),
                    weight(&fp, "cross_attn.q_rms_norm.gamma"),
                    weight(&fp, "cross_attn.k_rms_norm.gamma"),
                ],
            };
            for l in [
                &block.qkv,
                &block.self_out,
                &block.cross_q,
                &block.cross_out,
                &block.fc1,
                &block.fc2,
            ] {
                blas.prepare_biased_reference(
                    rows as u32,
                    l.weight.shape[0],
                    l.weight.shape[1],
                    l.bias,
                    super::linear::LinearBiasMode::BeforeOutputRound,
                )?;
            }
            blas.prepare_biased_reference(
                context_rows as u32,
                block.cross_kv.weight.shape[0],
                block.cross_kv.weight.shape[1],
                block.cross_kv.bias,
                super::linear::LinearBiasMode::BeforeOutputRound,
            )?;
            blocks.push(block);
        }
        let input = Linear::bind(&fp, device, "input_layer", [h, spec.in_channels])?;
        let output = Linear::bind(&fp, device, "out_layer", [spec.out_channels, h])?;
        let time = [
            Linear::bind(&fp, device, "t_embedder.mlp.0", [h, 256])?,
            Linear::bind(&fp, device, "t_embedder.mlp.2", [h, h])?,
            Linear::bind(&fp, device, "adaLN_modulation.1", [6 * h, h])?,
        ];
        let kernels = Kernels::load::<bf16>(device, &spec)?;
        let heads = spec.num_heads;
        let dim = h / heads;
        let freq_dim = dim / 2 / 3;
        let phases = device
            .stream
            .clone_htod(&rope_phase_sites(sites, dim, freq_dim))?;
        // The F32 attention cubin modules load here, before the budget
        // split, so their bytes are counted in the measured allocation.
        let _attention_modules = (
            super::cubin::module(&device.ctx, "sample")?,
            super::cubin::module(&device.ctx, "vision_ops")?,
        );
        let feed_forward = spec.feed_forward_width();
        // Live set in bytes: F32 seven rows-wide arrays, the staged input,
        // two context-wide, the time buffers and the velocity output; BF16
        // fourteen rows-wide arrays including the QKV and feed-forward tails,
        // five context-wide arrays, the conditioning context and the two
        // modulation vectors; the u32 identity row.
        let live = 4
            * (rows * h * 7
                + rows * spec.in_channels
                + context_rows * h * 2
                + 6 * h
                + 256
                + rows * spec.out_channels)
            + 2 * (rows * (h * 14 + feed_forward)
                + context_rows * (h * 5 + spec.cond_channels)
                + 12 * h)
            + 4 * rows;
        let remaining = device.workspace_budget(free_before, limit, live)?;
        let self_geometry = [
            heads as u32,
            rows as u32,
            rows as u32,
            dim as u32,
            dim as u32,
        ];
        let cross_geometry = [
            heads as u32,
            rows as u32,
            context_rows as u32,
            dim as u32,
            dim as u32,
        ];
        // Both attention workspaces divide the budget remaining after the
        // live arrays, half each, capped at the full 128-query tile; the
        // self-attention workspace takes the cross workspace's surplus.
        let cross_workspace = F32Attention::required_bytes(cross_geometry)?.min(remaining / 2);
        let self_workspace =
            F32Attention::required_bytes(self_geometry)?.min(remaining - cross_workspace);
        let cross = F32Attention::new(device.stream.clone(), cross_geometry, cross_workspace)?;
        let alloc = |n: usize| -> Result<cudarc::driver::CudaSlice<bf16>> {
            Ok(device.stream.alloc_zeros(n)?)
        };
        let flash = F32Attention::new(device.stream.clone(), self_geometry, self_workspace)?;
        println!(
            "TRELLIS2_SS_PLACEMENT self_tile={} cross_tile={}",
            flash.tile(),
            cross.tile()
        );
        let self_ = Self {
            device: device.clone(),
            kernels,
            _weights: bf,
            _float_weights: fp,
            blocks,
            input,
            output,
            time,
            flash,
            self_q32: device.stream.alloc_zeros(rows * h)?,
            self_k32: device.stream.alloc_zeros(rows * h)?,
            self_v32: device.stream.alloc_zeros(rows * h)?,
            self_out32: device.stream.alloc_zeros(rows * h)?,
            cross,
            cross_q32: device.stream.alloc_zeros(rows * h)?,
            cross_k32: device.stream.alloc_zeros(context_rows * h)?,
            cross_v32: device.stream.alloc_zeros(context_rows * h)?,
            cross_out32: device.stream.alloc_zeros(rows * h)?,
            rows,
            context_rows,
            heads,
            dim,
            width: h,
            cond_width: spec.cond_channels,
            in_channels: spec.in_channels,
            feed_forward,
            phases,
            context: device
                .stream
                .alloc_zeros(context_rows * spec.cond_channels)?,
            hidden: alloc(rows * h)?,
            norm: alloc(rows * h)?,
            projected: alloc(rows * h)?,
            qkv: alloc(rows * 3 * h)?,
            q: alloc(rows * h)?,
            k: alloc(rows * h)?,
            v: alloc(rows * h)?,
            kn: alloc(rows * h)?,
            attention: alloc(rows * h)?,
            block_mod: alloc(6 * h)?,
            cross_kv: alloc(context_rows * 2 * h)?,
            cross_k: alloc(context_rows * h)?,
            cross_v: alloc(context_rows * h)?,
            cross_kn: alloc(context_rows * h)?,
            ff: alloc(rows * feed_forward)?,
            input_x: device.stream.alloc_zeros(rows * spec.in_channels)?,
            time_embed: device.stream.alloc_zeros(256)?,
            mods: alloc(6 * h)?,
            time_out: device.stream.alloc_zeros(6 * h)?,
            identity: {
                let values: Vec<u32> = vec![0u32; rows];
                device.stream.clone_htod(&values)?
            },
            outer: device.stream.alloc_zeros(rows * h)?,
            velocity_out: device.stream.alloc_zeros(rows * spec.out_channels)?,
            capture: (0..3)
                .map(|_| alloc(rows * h))
                .collect::<Result<Vec<_>>>()?,
        };
        let (free_after, _) = device.ctx.mem_get_info()?;
        let used = free_before - free_after;
        ensure!(
            used <= limit,
            "SS flow used {used} device bytes under a {limit}-byte whole-request limit"
        );
        Ok(self_)
    }
}

impl TrellisSsFlow {
    fn compute_time(&mut self, blas: &Blas, t: f32) -> Result<()> {
        let d = &self.device;
        let h = self.width;
        let half = 128usize;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-10000f64.ln() as f32 * i as f32 / half as f32).exp())
            .collect();
        let freqs = d.stream.clone_htod(&freqs)?;
        let times = [t];
        let times = d.stream.clone_htod(&times)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.sinusoid)
                .arg(&times)
                .arg(&freqs)
                .arg(&self.time_embed)
                .arg(&1u32)
                .arg(&(half as u32))
                .arg(&0i32)
                .launch(flat(half * 2))?;
        }
        let first = d.stream.alloc_zeros::<f32>(h)?;
        let second = d.stream.alloc_zeros::<f32>(h)?;
        run_linear_f32(
            d,
            blas,
            &self.kernels,
            self.time[0],
            1,
            [ptr(&self.time_embed, d), ptr(&first, d)],
            true,
        )?;
        run_linear_f32(
            d,
            blas,
            &self.kernels,
            self.time[1],
            1,
            [ptr(&first, d), ptr(&second, d)],
            false,
        )?;
        let zero = d.stream.alloc_zeros::<f32>(h)?;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.silu)
                .arg(&second)
                .arg(&zero)
                .arg(&second)
                .arg(&(h as u32))
                .arg(&(h as u32))
                .launch(flat(h))?;
        }
        run_linear_f32(
            d,
            blas,
            &self.kernels,
            self.time[2],
            1,
            [ptr(&second, d), ptr(&self.time_out, d)],
            false,
        )?;
        self.kernels
            .cast(d, ptr(&self.time_out, d), ptr(&self.mods, d), 6 * h)?;
        Ok(())
    }

    fn modulate(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.width;
        let shift = modulation + (part * h * std::mem::size_of::<bf16>()) as u64;
        let scale = shift + (h * std::mem::size_of::<bf16>()) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.modulate)
                .arg(&values)
                .arg(&shift)
                .arg(&scale)
                .arg(&self.identity)
                .arg(&values)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn gated_add(&self, values: u64, modulation: u64, part: usize) -> Result<()> {
        let h = self.width;
        let gate = modulation + (part * h * std::mem::size_of::<bf16>()) as u64;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.gated)
                .arg(&self.hidden)
                .arg(&values)
                .arg(&gate)
                .arg(&self.identity)
                .arg(&(self.rows as i32))
                .arg(&(h as i32))
                .arg(&((6 * h) as u64))
                .arg(&1u32)
                .launch(LaunchConfig {
                    grid_dim: (self.rows as u32, h.div_ceil(256) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn block(&mut self, blas: &Blas, index: usize) -> Result<()> {
        let d = &self.device;
        let n = self.rows;
        let h = self.width;
        let heads = self.heads;
        let dim = self.dim;
        let ff = self.feed_forward;
        let b = &self.blocks[index];
        let k = &self.kernels;
        let hidden = ptr(&self.hidden, d);
        let norm = ptr(&self.norm, d);
        let projected = ptr(&self.projected, d);
        let q = ptr(&self.q, d);
        let key = ptr(&self.k, d);
        let value = ptr(&self.v, d);
        let kn = ptr(&self.kn, d);
        let attention = ptr(&self.attention, d);
        let bm = ptr(&self.block_mod, d);
        unsafe {
            d.stream
                .launch_builder(&k.add_mod)
                .arg(&ptr(&self.mods, d))
                .arg(&b.modulation)
                .arg(&bm)
                .arg(&((6 * h) as u64))
                .arg(&((6 * h) as u32))
                .launch(flat(6 * h))?;
        }
        k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
        self.modulate(norm, bm, 0)?;
        b.qkv.run(blas, n, norm, ptr(&self.qkv, d))?;
        k.unpack(d, ptr(&self.qkv, d), [q, key, value], n, 3)?;
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
        k.upcast(d, q, ptr(&self.self_q32, d), n * h)?;
        k.upcast(d, kn, ptr(&self.self_k32, d), n * h)?;
        k.upcast(d, value, ptr(&self.self_v32, d), n * h)?;
        let q32 = view(
            &self.self_q32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        let k32 = dense(&self.self_k32, d, [heads as u32, n as u32, dim as u32])?;
        let v32 = dense(&self.self_v32, d, [heads as u32, n as u32, dim as u32])?;
        let s_out32 = view(
            &self.self_out32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        self.flash.run(
            blas,
            q32,
            k32,
            v32,
            &mut TensorMut(s_out32),
            F32Mask {
                causal: false,
                valid_keys: n as u32,
                query_base: 0,
                window: 0,
            },
        )?;
        k.cast(d, ptr(&self.self_out32, d), attention, n * h)?;
        b.self_out.run(blas, n, attention, projected)?;
        self.gated_add(projected, bm, 2)?;
        k.norm(
            d,
            hidden,
            norm,
            [n, h],
            [b.norm_weight, b.norm_bias],
            (1e-6, false),
        )?;
        b.cross_q.run(blas, n, norm, q)?;
        b.cross_kv.run(
            blas,
            self.context_rows,
            ptr(&self.context, d),
            ptr(&self.cross_kv, d),
        )?;
        k.unpack(
            d,
            ptr(&self.cross_kv, d),
            [0, ptr(&self.cross_k, d), ptr(&self.cross_v, d)],
            self.context_rows,
            2,
        )?;
        k.l2(d, q, q, b.gamma[2], n, false)?;
        k.l2(
            d,
            ptr(&self.cross_k, d),
            ptr(&self.cross_kn, d),
            b.gamma[3],
            self.context_rows,
            true,
        )?;
        k.upcast(d, q, ptr(&self.cross_q32, d), n * h)?;
        k.upcast(
            d,
            ptr(&self.cross_kn, d),
            ptr(&self.cross_k32, d),
            self.context_rows * h,
        )?;
        k.upcast(
            d,
            ptr(&self.cross_v, d),
            ptr(&self.cross_v32, d),
            self.context_rows * h,
        )?;
        let q32 = view(
            &self.cross_q32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        let k32 = dense(
            &self.cross_k32,
            d,
            [heads as u32, self.context_rows as u32, dim as u32],
        )?;
        let v32 = dense(
            &self.cross_v32,
            d,
            [heads as u32, self.context_rows as u32, dim as u32],
        )?;
        let out32 = view(
            &self.cross_out32,
            &d.stream,
            0,
            [heads as u32, n as u32, dim as u32],
            [dim as u64, h as u64, 1],
        )?;
        self.cross.run(
            blas,
            q32,
            k32,
            v32,
            &mut TensorMut(out32),
            F32Mask {
                causal: false,
                valid_keys: self.context_rows as u32,
                query_base: 0,
                window: 0,
            },
        )?;
        k.cast(d, ptr(&self.cross_out32, d), attention, n * h)?;
        b.cross_out.run(blas, n, attention, projected)?;
        k.residual(d, hidden, projected, n * h)?;
        k.norm(d, hidden, norm, [n, h], [0, 0], (1e-6, false))?;
        self.modulate(norm, bm, 3)?;
        b.fc1.run(blas, n, norm, ptr(&self.ff, d))?;
        unsafe {
            d.stream
                .launch_builder(&k.gelu)
                .arg(&self.ff)
                .arg(&self.ff)
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
        b.fc2.run(blas, n, ptr(&self.ff, d), projected)?;
        self.gated_add(projected, bm, 5)?;
        Ok(())
    }

    /// Host staging: x rows, context, and the time modulation (definition
    /// sparse_structure_flow.py:224-241); no device compute, safe outside graphs.
    pub fn stage(&mut self, blas: &Blas, x_f32: &[f32], t: f32, cond: &[bf16]) -> Result<()> {
        ensure!(
            x_f32.len() == self.rows * self.in_channels
                && cond.len() == self.context_rows * self.cond_width,
            "flow input extent mismatch"
        );
        let d = &self.device;
        let n = self.rows;
        let h = self.width;
        d.stream
            .memcpy_htod(x_f32, &mut self.input_x)
            .map_err(anyhow::Error::from)?;
        d.stream
            .memcpy_htod(cond, &mut self.context)
            .map_err(anyhow::Error::from)?;
        run_linear_f32(
            d,
            blas,
            &self.kernels,
            self.input,
            n,
            [ptr(&self.input_x, d), ptr(&self.self_q32, d)],
            false,
        )?;
        self.kernels
            .cast(d, ptr(&self.self_q32, d), ptr(&self.hidden, d), n * h)?;
        self.compute_time(blas, t)?;
        Ok(())
    }

    pub fn compute(&mut self, blas: &Blas, captured_blocks: &[usize]) -> Result<()> {
        ensure_captured_blocks(captured_blocks, self.blocks.len(), self.capture.len())?;
        let mut slot = 0usize;
        for index in 0..self.blocks.len() {
            self.block(blas, index)?;
            if captured_blocks.contains(&index) {
                self.device
                    .stream
                    .memcpy_dtod(&self.hidden, &mut self.capture[slot])
                    .map_err(anyhow::Error::from)?;
                slot += 1;
            }
        }
        ensure!(slot == captured_blocks.len(), "capture slot mismatch");
        Ok(())
    }

    /// Host readback: unit-normed F32 rows through the out layer (velocity).
    pub fn read_velocity(&mut self, blas: &Blas) -> Result<Vec<f32>> {
        let d = &self.device;
        let n = self.rows;
        let h = self.width;
        self.kernels
            .upcast(d, ptr(&self.hidden, d), ptr(&self.outer, d), n * h)?;
        self.kernels.norm(
            d,
            ptr(&self.outer, d),
            ptr(&self.outer, d),
            [n, h],
            [0, 0],
            (1e-5, true),
        )?;
        run_linear_f32(
            d,
            blas,
            &self.kernels,
            self.output,
            n,
            [ptr(&self.outer, d), ptr(&self.velocity_out, d)],
            false,
        )?;
        Ok(self.device.stream.clone_dtoh(&self.velocity_out)?.to_vec())
    }

    pub fn read_capture(&self, captured_blocks: &[usize]) -> Result<Vec<Vec<f32>>> {
        ensure_captured_blocks(captured_blocks, self.blocks.len(), self.capture.len())?;
        let mut out = Vec::with_capacity(captured_blocks.len());
        for (slot, _block) in captured_blocks.iter().enumerate() {
            let host = self.device.stream.clone_dtoh(&self.capture[slot])?;
            let values: Vec<f32> = host.iter().map(|b| f32::from(*b)).collect();
            out.push(values);
        }
        Ok(out)
    }

    pub fn velocity(
        &mut self,
        blas: &Blas,
        x_f32: &[f32],
        t: f32,
        cond: &[bf16],
        captured_blocks: &[usize],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.stage(blas, x_f32, t, cond)?;
        self.compute(blas, captured_blocks)?;
        let velocity = self.read_velocity(blas)?;
        let blocks = self.read_capture(captured_blocks)?;
        Ok((velocity, blocks))
    }

    /// Euler interval-CFG sampling on the published 12-step schedule:
    /// sigma_min 1e-5, rescale_t 5.0, guidance 7.5 with 0.7 rescale inside
    /// [0.6, 1.0], single-branch outside; the unguided branch conditions on
    /// a zero context.
    pub fn sample(
        &mut self,
        blas: &Blas,
        noise: &[f32],
        cond: &[bf16],
        record_states: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        ensure!(
            noise.len() == self.rows * self.in_channels
                && cond.len() == self.context_rows * self.cond_width,
            "flow sample extent mismatch"
        );
        let negative = vec![bf16::from_f32(0.0); cond.len()];
        let times = super::pixal3d::schedule(false, 12);
        let mut sampler = super::pixal3d::Sampler::new(&self.device, noise, true)?;
        let mut states = Vec::new();
        for step in 0..12 {
            let t = times[step];
            let state: Vec<f32> = self.device.stream.clone_dtoh(&sampler.state)?.to_vec();
            let (positive, _) = self.velocity(blas, &state, (1000.0 * t) as f32, cond, &[])?;
            self.device
                .stream
                .memcpy_htod(&positive, &mut sampler.positive)
                .map_err(anyhow::Error::from)?;
            if (0.6..=1.0).contains(&t) {
                let (unguided, _) =
                    self.velocity(blas, &state, (1000.0 * t) as f32, &negative, &[])?;
                let branch = sampler
                    .negative
                    .as_mut()
                    .context("CFG branch buffer missing")?;
                self.device
                    .stream
                    .memcpy_htod(&unguided, branch)
                    .map_err(anyhow::Error::from)?;
            }
            sampler.step(&self.device, t, times[step + 1], false)?;
            if record_states {
                states.push(self.device.stream.clone_dtoh(&sampler.state)?.to_vec());
            }
        }
        let final_state: Vec<f32> = self.device.stream.clone_dtoh(&sampler.state)?.to_vec();
        Ok((final_state, states))
    }
}

fn ensure_captured_blocks(blocks: &[usize], total: usize, slots: usize) -> Result<()> {
    ensure!(
        blocks.len() <= slots,
        "capture has {slots} slots; {blocks:?} requested"
    );
    for pair in blocks.windows(2) {
        ensure!(
            pair[0] < pair[1],
            "captured blocks must be strictly ascending: {blocks:?}"
        );
    }
    if let Some(&last) = blocks.last() {
        ensure!(last < total, "captured block {last} beyond {total} blocks");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_phase_axes_are_slowest_first() {
        // Coprime extents so (z, y, x) = (1, 2, 3) has every coordinate
        // distinct: any axis permutation changes at least one angle.
        let extents = [3usize, 4, 5];
        let dim = 128usize;
        let freq_dim = dim / 2 / 3;
        let mut sites = Vec::new();
        for z in 0..extents[0] {
            for y in 0..extents[1] {
                for x in 0..extents[2] {
                    sites.push([z as i32, y as i32, x as i32]);
                }
            }
        }
        let phases = rope_phase_sites(&sites, dim, freq_dim);
        let row = (extents[1] + 2) * extents[2] + 3;
        // Expected phases built from the model definition: freqs[i] =
        // 10000^(-i/21), pair p uses axis p/21 (slowest first) and freq p%21.
        let expected = |pair: usize, coords: [f32; 3]| -> (f32, f32) {
            if pair < 3 * freq_dim {
                let freq = 10000f32.powf(-((pair % freq_dim) as f32) / freq_dim as f32);
                let angle = coords[pair / freq_dim] * freq;
                (angle.cos(), angle.sin())
            } else {
                (1.0, 0.0)
            }
        };
        let coords = [1.0f32, 2.0, 3.0];
        for pair in [
            0usize,
            1,
            freq_dim - 1,
            freq_dim,
            2 * freq_dim,
            3 * freq_dim - 1,
        ] {
            let base = row * dim + 2 * pair;
            let (cos, sin) = expected(pair, coords);
            assert!(
                (phases[base] - cos).abs() < 1e-6 && (phases[base + 1] - sin).abs() < 1e-6,
                "pair {pair}: got ({}, {}), expected ({cos}, {sin})",
                phases[base],
                phases[base + 1]
            );
        }
        let base = row * dim + 2 * 3 * freq_dim;
        assert_eq!(phases[base], 1.0);
        assert_eq!(phases[base + 1], 0.0);
    }

    #[test]
    fn published_checkpoint_metadata_header_loads() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("trellis2-metadata-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let mut header = std::collections::BTreeMap::new();
        header.insert(
            "__metadata__".to_string(),
            serde_json::json!({"format": "pt"}),
        );
        header.insert(
            "w".to_string(),
            serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}),
        );
        header.insert(
            "scale".to_string(),
            serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}),
        );
        let text = serde_json::to_string(&header)?;
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(text.as_bytes());
        file.extend_from_slice(&[1u8, 0, 2, 0]);
        file.extend_from_slice(&0.5f32.to_le_bytes());
        let path = dir.join("metadata_head.safetensors");
        std::fs::write(&path, &file)?;
        let mapped = crate::checkpoint::MappedCheckpoint::open_component(&dir, "metadata_head")?;
        let tensors = mapped.component_tensors()?;
        ensure!(tensors.len() == 2, "metadata entry leaked as a tensor");
        ensure!(tensors["w"].dtype == "BF16" && tensors["scale"].dtype == "F32");
        ensure!(tensors["scale"].data == 0.5f32.to_le_bytes());
        std::fs::remove_file(&path)?;
        std::fs::remove_dir(&dir)?;
        Ok(())
    }

    #[test]
    fn captured_blocks_must_be_strictly_ascending() {
        assert!(ensure_captured_blocks(&[0, 15, 29], 30, 3).is_ok());
        assert!(ensure_captured_blocks(&[], 30, 3).is_ok());
        assert!(ensure_captured_blocks(&[29, 0], 30, 3).is_err());
        assert!(ensure_captured_blocks(&[0, 0], 30, 3).is_err());
        assert!(ensure_captured_blocks(&[30], 30, 3).is_err());
        assert!(ensure_captured_blocks(&[0, 1, 2, 3], 30, 3).is_err());
    }

    fn read_tensor(path: &std::path::Path, key: &str) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        #[derive(serde::Deserialize)]
        struct Header {
            dtype: String,
            data_offsets: [usize; 2],
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
        let header: std::collections::BTreeMap<String, Header> =
            serde_json::from_slice(&bytes[8..8 + header_len])?;
        let entry = header
            .get(key)
            .with_context(|| format!("tensor {key} missing"))?;
        let start = entry.data_offsets[0];
        let end = entry.data_offsets[1];
        let data = &bytes[8 + header_len + start..8 + header_len + end];
        Ok(match entry.dtype.as_str() {
            "F32" => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            "BF16" => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(bf16::from_bits(u16::from_le_bytes(*b))))
                .collect(),
            other => anyhow::bail!("unexpected gate tensor dtype {other}"),
        })
    }

    #[test]
    #[ignore = "TRELLIS.2 SS flow cells: FF_TRELLIS2_SS_CHECKPOINT FF_TRELLIS2_SS_FIXTURES FF_TRELLIS2_SS_OUTPUT, optional FF_TRELLIS2_SS_LIMIT_BYTES"]
    fn ss_flow_cells_gpu() -> Result<()> {
        let checkpoint: std::path::PathBuf = std::env::var_os("FF_TRELLIS2_SS_CHECKPOINT")
            .map(Into::into)
            .context("FF_TRELLIS2_SS_CHECKPOINT missing")?;
        let root: std::path::PathBuf = std::env::var_os("FF_TRELLIS2_SS_FIXTURES")
            .map(Into::into)
            .context("FF_TRELLIS2_SS_FIXTURES missing")?;
        let out: std::path::PathBuf = std::env::var_os("FF_TRELLIS2_SS_OUTPUT")
            .map(Into::into)
            .context("FF_TRELLIS2_SS_OUTPUT missing")?;
        std::fs::create_dir_all(&out)?;
        let floor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("ss-floor-r3.json"))?)?;
        let cells = floor["cells"]
            .as_array()
            .context("SS floor cells missing")?;
        let tau = |id: &str| -> Result<(f64, f64)> {
            let cell = cells
                .iter()
                .find(|c| c["id"] == id)
                .with_context(|| format!("cell {id} missing"))?;
            Ok((
                cell["tau"]["max_abs"].as_f64().context("tau max_abs")?,
                cell["tau"]["rmse"].as_f64().context("tau rmse")?,
            ))
        };
        let metric = |a: &[f32], b: &[f32]| -> Result<(f64, f64)> {
            ensure!(a.len() == b.len() && !a.is_empty(), "cell extent mismatch");
            let mut maximum = 0.0f64;
            let mut square = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                let d = f64::from(*x) - f64::from(*y);
                maximum = maximum.max(d.abs());
                square += d * d;
            }
            Ok((maximum, (square / a.len() as f64).sqrt()))
        };
        let score = |native: &[f32],
                     reference: &[f32],
                     channels: usize,
                     id: &str,
                     defects: &mut usize|
         -> Result<()> {
            let (tau_max, tau_rmse) = tau(id)?;
            let (max_abs, rmse) = metric(native, reference)?;
            let width = native.len() / channels;
            let rotated: Vec<f32> = (0..native.len())
                .map(|index| {
                    let channel = index / width;
                    native[index - channel * width + ((channel + 1) % channels) * width]
                })
                .collect();
            let (control_max, control_rmse) = metric(native, &rotated)?;
            let pass = max_abs <= tau_max && rmse <= tau_rmse;
            let control_ok = control_max > 10.0 * tau_max && control_rmse > 10.0 * tau_rmse;
            println!(
                "TRELLIS2_SS_CELL id={id} max_abs={max_abs:.15e} rmse={rmse:.15e} tau_max={tau_max:.15e} tau_rmse={tau_rmse:.15e} control_max={control_max:.15e} control_rmse={control_rmse:.15e} result={}",
                if pass && control_ok { "PASS" } else { "DEFECT" }
            );
            *defects += usize::from(!pass) + usize::from(!control_ok);
            Ok(())
        };
        let device = Device::new(0)?;
        let ops = super::super::ops::Ops::new(&device.ctx)?;
        let mut blas = Blas::new(
            &device,
            &ops,
            &out,
            std::rc::Rc::new(super::super::closure::Closure::new(
                true,
                "ss flow cells".into(),
            )),
        )?;
        let (free, total) = device.ctx.mem_get_info()?;
        let limit = match std::env::var("FF_TRELLIS2_SS_LIMIT_BYTES") {
            Ok(value) => value
                .trim()
                .parse::<usize>()
                .with_context(|| format!("invalid FF_TRELLIS2_SS_LIMIT_BYTES: {value}"))?,
            Err(_) => (total - total / 8).saturating_sub(total - free),
        };
        let mut defects = 0usize;
        let mut scored = 0usize;
        let mut bit_failures = 0usize;
        for fixture in ["cottage", "robot_crab"] {
            let cond: Vec<bf16> = read_tensor(
                &root.join(format!(
                    "conditioning-r0a/{fixture}-512-features.safetensors"
                )),
                "features",
            )?
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect();
            // CFG runs pos (cond) then neg (zero context) per step; the final
            // step is unguided, so calls 2..=20 even carry the zero context.
            let neg_cond = vec![bf16::from_f32(0.0); cond.len()];
            let mut flow = TrellisSsFlow::load(&device, &mut blas, &checkpoint, 1029, limit)?;
            let capture_dir = root.join(format!("ss-{fixture}-r0a"));
            for call in 1..=22 {
                let input = read_tensor(
                    &capture_dir.join(format!("flow-call{call}-input.safetensors")),
                    "x",
                )?;
                let t = read_tensor(
                    &capture_dir.join(format!("flow-call{call}-input.safetensors")),
                    "t",
                )?[0];
                ensure!(input.len() == 8 * 4096, "SS input extent");
                let mut rows = vec![0f32; 4096 * 8];
                for (index, value) in input.iter().enumerate() {
                    rows[(index % 4096) * 8 + index / 4096] = *value;
                }
                let blocks: &[usize] = if [1, 11, 22].contains(&call) {
                    &[0, 15, 29]
                } else {
                    &[]
                };
                eprintln!(
                    "STAGE lens: x={} cond={} rows={} width={} ctx={}",
                    rows.len(),
                    cond.len(),
                    flow.rows,
                    flow.width,
                    flow.context_rows
                );
                let call_cond = if call % 2 == 0 && call < 22 {
                    &neg_cond
                } else {
                    &cond
                };
                let free_before_stage = if call == 2 {
                    Some(device.ctx.mem_get_info()?.0)
                } else {
                    None
                };
                flow.stage(&blas, &rows, t, call_cond)?;
                flow.compute(&blas, blocks)?;
                let velocity = flow.read_velocity(&blas)?;
                let captured = flow.read_capture(blocks)?;
                if let Some(before) = free_before_stage {
                    let after = device.ctx.mem_get_info()?.0;
                    ensure!(
                        after == before,
                        "per-call device allocation: {before} -> {after}"
                    );
                    println!("TRELLIS2_SS_STAGE_ALLOC free_stable=true bytes={after}");
                }
                let reference = read_tensor(
                    &capture_dir.join(format!("flow-call{call}-velocity.safetensors")),
                    "velocity",
                )?;
                let mut velocity_reference = vec![0f32; reference.len()];
                for (index, value) in reference.iter().enumerate() {
                    velocity_reference[(index % 4096) * 8 + index / 4096] = *value;
                }
                score(
                    &velocity,
                    &velocity_reference,
                    8,
                    &format!("ss/{fixture}/flow-call{call}-velocity/velocity"),
                    &mut defects,
                )?;
                scored += 1;
                for (slot, block) in blocks.iter().enumerate() {
                    let stored = read_tensor(
                        &capture_dir.join(format!("flow-call{call}-block{block}.safetensors")),
                        "output",
                    )?;
                    score(
                        &captured[slot],
                        &stored,
                        1536,
                        &format!("ss/{fixture}/flow-call{call}-block{block}/output"),
                        &mut defects,
                    )?;
                    scored += 1;
                }
                let stream = &device.stream;
                stream.synchronize()?;
                flow.stage(&blas, &rows, t, call_cond)?;
                stream.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                let launch = flow.compute(&blas, blocks);
                let graph = stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
                launch?;
                let graph = graph.context("empty SS graph")?;
                graph.as_ref().context("SS graph missing")?.launch()?;
                stream.synchronize()?;
                let replay_velocity = flow.read_velocity(&blas)?;
                let identical = velocity
                    .iter()
                    .zip(&replay_velocity)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                println!(
                    "TRELLIS2_SS_GRAPH_EAGER_IDENTICAL fixture={fixture} call={call} bits={identical}"
                );
                bit_failures += usize::from(!identical);
            }
        }
        ensure!(
            defects == 0 && bit_failures == 0,
            "TRELLIS.2 SS flow cells gate: {defects} defects, {bit_failures} graph mismatches"
        );
        println!(
            "PASS TRELLIS2_SS_FLOW cells={scored} fixtures=2 calls=22 graph_eager_bit_identical=true"
        );
        Ok(())
    }

    #[ignore = "TRELLIS.2 SS sampler and decode: FF_TRELLIS2_SS_CHECKPOINT FF_TRELLIS2_SS_FIXTURES FF_TRELLIS2_SS_OUTPUT FF_TRELLIS_IMAGE_MODEL FLYINGFISH_RUNTIME_DIR, optional FF_TRELLIS2_SS_LIMIT_BYTES"]
    #[test]
    fn ss_sample_decode_cells_gpu() -> Result<()> {
        let checkpoint: std::path::PathBuf = std::env::var_os("FF_TRELLIS2_SS_CHECKPOINT")
            .map(Into::into)
            .context("FF_TRELLIS2_SS_CHECKPOINT missing")?;
        ensure!(
            checkpoint.is_file(),
            "SS flow checkpoint not found: {}",
            checkpoint.display()
        );
        let root: std::path::PathBuf = std::env::var_os("FF_TRELLIS2_SS_FIXTURES")
            .map(Into::into)
            .context("FF_TRELLIS2_SS_FIXTURES missing")?;
        let image_model: std::path::PathBuf = std::env::var_os("FF_TRELLIS_IMAGE_MODEL")
            .map(Into::into)
            .context("FF_TRELLIS_IMAGE_MODEL missing")?;
        let runtime = std::path::PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        );
        let sampler_floor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("sampler-floor-r1/floor.json"))?)?;
        let sampler_cells = sampler_floor["cells"]
            .as_array()
            .context("sampler floor cells missing")?;
        let ss_floor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("ss-floor-r3.json"))?)?;
        let ss_cells = ss_floor["cells"]
            .as_array()
            .context("SS floor cells missing")?;
        let tau_of = |cells: &[serde_json::Value], id: &str| -> Result<(f64, f64)> {
            let cell = cells
                .iter()
                .find(|c| c["id"] == id)
                .with_context(|| format!("cell {id} missing"))?;
            Ok((
                cell["tau"]["max_abs"].as_f64().context("tau max_abs")?,
                cell["tau"]["rmse"].as_f64().context("tau rmse")?,
            ))
        };
        let metric = |a: &[f32], b: &[f32]| -> Result<(f64, f64)> {
            ensure!(a.len() == b.len() && !a.is_empty(), "cell extent mismatch");
            let mut maximum = 0.0f64;
            let mut square = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                let d = f64::from(*x) - f64::from(*y);
                maximum = maximum.max(d.abs());
                square += d * d;
            }
            Ok((maximum, (square / a.len() as f64).sqrt()))
        };
        let device = Device::new(0)?;
        let (free, total) = device.ctx.mem_get_info()?;
        let limit = match std::env::var("FF_TRELLIS2_SS_LIMIT_BYTES") {
            Ok(value) => value
                .trim()
                .parse::<usize>()
                .with_context(|| format!("invalid FF_TRELLIS2_SS_LIMIT_BYTES: {value}"))?,
            Err(_) => (total - total / 8).saturating_sub(total - free),
        };
        let mut defects = 0usize;
        let mut scored = 0usize;
        let rows = 4096usize;
        for fixture in ["cottage", "robot_crab"] {
            let record = root.join(format!("ss-{fixture}-r0a"));
            let mut noise = vec![0f32; rows * 8];
            {
                let stored = read_tensor(&record.join("sampler-input.safetensors"), "noise")?;
                for (index, value) in stored.iter().enumerate() {
                    noise[(index % rows) * 8 + index / rows] = *value;
                }
            }
            let stored_neg: Vec<bf16> =
                read_tensor(&record.join("sampler-input.safetensors"), "neg_cond")?
                    .iter()
                    .map(|v| bf16::from_f32(*v))
                    .collect();
            ensure!(
                stored_neg.iter().all(|v| v.to_bits() == 0),
                "stored unguided context is not all-zero bits"
            );
            // The frozen state cells price the Euler blend given the
            // reference model outputs (the captured per-call velocities),
            // not the native model difference, which the flow cells bound.
            let mut velocities = Vec::new();
            for call in 1..=22 {
                let stored = read_tensor(
                    &record.join(format!("flow-call{call}-velocity.safetensors")),
                    "velocity",
                )?;
                let mut row_major = vec![0f32; rows * 8];
                for (index, value) in stored.iter().enumerate() {
                    row_major[(index % rows) * 8 + index / rows] = *value;
                }
                velocities.push(row_major);
            }
            let times = super::super::pixal3d::schedule(false, 12);
            let mut sampler = super::super::pixal3d::Sampler::new(&device, &noise, true)?;
            let mut call = 0usize;
            let mut previous = noise.clone();
            for step in 0..12 {
                let t = times[step];
                device
                    .stream
                    .memcpy_htod(&velocities[call], &mut sampler.positive)
                    .map_err(anyhow::Error::from)?;
                call += 1;
                if (0.6..=1.0).contains(&t) {
                    let branch = sampler
                        .negative
                        .as_mut()
                        .context("CFG branch buffer missing")?;
                    device
                        .stream
                        .memcpy_htod(&velocities[call], branch)
                        .map_err(anyhow::Error::from)?;
                    call += 1;
                }
                sampler.step(&device, t, times[step + 1], false)?;
                let state: Vec<f32> = device.stream.clone_dtoh(&sampler.state)?.to_vec();
                let stored = read_tensor(
                    &record.join("sampler-output.safetensors"),
                    &format!("state{}", step + 1),
                )?;
                let mut reference = vec![0f32; rows * 8];
                for (index, value) in stored.iter().enumerate() {
                    reference[(index % rows) * 8 + index / rows] = *value;
                }
                let id = format!("sampler/ss-{fixture}-r0a/state{}", step + 1);
                let (tau_max, tau_rmse) = tau_of(sampler_cells, &id)?;
                let (max_abs, rmse) = metric(&state, &reference)?;
                let flipped: Vec<f32> = state
                    .iter()
                    .zip(previous.iter())
                    .map(|(s, p)| 2.0 * p - s)
                    .collect();
                let (control_max, control_rmse) = metric(&state, &flipped)?;
                let pass = max_abs <= tau_max && rmse <= tau_rmse;
                let control_ok = control_max > 10.0 * tau_max && control_rmse > 10.0 * tau_rmse;
                println!(
                    "TRELLIS2_SS_SAMPLE id={id} max_abs={max_abs:.15e} rmse={rmse:.15e} tau_max={tau_max:.15e} tau_rmse={tau_rmse:.15e} control_max={control_max:.15e} control_rmse={control_rmse:.15e} result={}",
                    if pass && control_ok { "PASS" } else { "DEFECT" }
                );
                scored += 1;
                defects += usize::from(!pass) + usize::from(!control_ok);
                previous = state;
            }
            ensure!(call == 22, "captured velocity call count mismatch: {call}");
            let decoder = super::super::pixal3d::PixalDecoder::open(
                &device,
                &image_model,
                crate::pixal3d::DecoderKind::Structure,
                &runtime,
                limit,
            )?;
            let latent = read_tensor(&record.join("decoder-final.safetensors"), "latent")?;
            let logits = decoder.structure(&latent)?;
            let reference = read_tensor(&record.join("decoder-final.safetensors"), "logits")?;
            for id in [
                format!("ss/{fixture}/decoder-final/logits"),
                format!("ss/{fixture}/decoder-out_layer.2/output"),
            ] {
                let (tau_max, tau_rmse) = tau_of(ss_cells, &id)?;
                let (max_abs, rmse) = metric(&logits, &reference)?;
                let flipped: Vec<f32> = logits.iter().map(|v| -v).collect();
                let (control_max, control_rmse) = metric(&logits, &flipped)?;
                let pass = max_abs <= tau_max && rmse <= tau_rmse;
                let control_ok = control_max > 10.0 * tau_max && control_rmse > 10.0 * tau_rmse;
                println!(
                    "TRELLIS2_SS_DECODE id={id} max_abs={max_abs:.15e} rmse={rmse:.15e} tau_max={tau_max:.15e} tau_rmse={tau_rmse:.15e} control_max={control_max:.15e} control_rmse={control_rmse:.15e} result={}",
                    if pass && control_ok { "PASS" } else { "DEFECT" }
                );
                scored += 1;
                defects += usize::from(!pass) + usize::from(!control_ok);
            }
        }
        ensure!(
            defects == 0,
            "TRELLIS.2 SS sampler/decode gate: {defects} defects over {scored} cells"
        );
        println!("PASS TRELLIS2_SS_SAMPLE_DECODE cells={scored} fixtures=2 states=12");
        Ok(())
    }
}
