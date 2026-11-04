use super::{
    Device,
    blas::Blas,
    closure::Closure,
    contracts::{Tensor, TensorMut, view},
    copy::{dtod_raw, dtoh_async, pinned_zeros},
    cubin,
    flash::{Flash, Shape},
    linear::LinearWeightRef,
    ops::Ops,
    weights::{WeightCatalog, checkpoint_mapped},
};
use crate::{
    backend::gpu::{
        CudaFunction, CudaGraph, CudaSlice, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    },
    music3::ArCheckpoint,
    op_contract::ExecutionPhase,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{collections::BTreeMap, path::Path, rc::Rc};

struct Layer {
    norm: [u64; 2],
    q_norm: u64,
    k_norm: u64,
    matrices: [Tensor<bf16, 2>; 7],
    cache: Vec<(CudaSlice<bf16>, CudaSlice<bf16>)>,
}

struct Work {
    hidden: CudaSlice<bf16>,
    normalized: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    attention: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    activated: CudaSlice<bf16>,
    branch_q: CudaSlice<bf16>,
    branch_out: CudaSlice<bf16>,
    last: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    ids: CudaSlice<u32>,
}

struct Kernels {
    norm: CudaFunction,
    residual: CudaFunction,
    silu: CudaFunction,
    embed: CudaFunction,
    rotary: CudaFunction,
    code_embed: CudaFunction,
    feedback: CudaFunction,
    policy: CudaFunction,
    draw: CudaFunction,
    rotary_device: CudaFunction,
    advance: CudaFunction,
}

struct Policy {
    probabilities: CudaSlice<f32>,
    scratch: CudaSlice<f32>,
    thresholds: CudaSlice<f32>,
    keep: CudaSlice<u8>,
    codes: CudaSlice<u32>,
    features: CudaSlice<bf16>,
    host_features: PinnedHostSlice<bf16>,
}

impl Policy {
    fn new(d: &Device) -> Result<Self> {
        Ok(Self {
            probabilities: d.stream.alloc_zeros(200000)?,
            scratch: d.stream.alloc_zeros(2 * 16385)?,
            thresholds: d.stream.alloc_zeros(2)?,
            keep: d.stream.alloc_zeros(200000)?,
            codes: d.stream.alloc_zeros(8)?,
            features: d.stream.alloc_zeros(32768)?,
            host_features: pinned_zeros(&d.ctx, 32768)?,
        })
    }
}

struct Depth {
    _weights: BTreeMap<String, CudaSlice<bf16>>,
    projection: Tensor<bf16, 2>,
    heads: Vec<Tensor<bf16, 2>>,
    embeddings: u64,
    norm: u64,
    positions: u64,
    layers: Vec<Layer>,
    sequence: CudaSlice<bf16>,
    outputs: CudaSlice<bf16>,
    last: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    flash: Flash,
    position: usize,
}

impl Depth {
    fn load(ar: &ArCheckpoint, d: &Device) -> Result<Self> {
        let (weights, _) = checkpoint_mapped(
            &ar.depth_weights,
            d,
            Some(&ar.depth_manifest),
            false,
            |loader| {
                ar.depth_manifest
                    .entries
                    .iter()
                    .map(|(name, entry)| {
                        Ok((
                            name.clone(),
                            loader.load(&[(name.clone(), entry.shape.clone())])?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            },
        )?;
        let mut layers = Vec::new();
        for i in 0..4 {
            let p = format!("layers.{i}");
            let m = |name: &str, rows, cols| {
                matrix(&weights[&format!("{p}.{name}.weight")], d, rows, cols)
            };
            let mut cache = Vec::new();
            for _ in 0..2 {
                cache.push((
                    d.stream.alloc_zeros(16 * 4096)?,
                    d.stream.alloc_zeros(16 * 4096)?,
                ));
            }
            layers.push(Layer {
                norm: [
                    ptr(&weights[&format!("{p}.input_layernorm.weight")], d),
                    ptr(&weights[&format!("{p}.post_attention_layernorm.weight")], d),
                ],
                q_norm: 0,
                k_norm: 0,
                matrices: [
                    m("attn.to_q", 4096, 4096)?,
                    m("attn.to_k", 4096, 4096)?,
                    m("attn.to_v", 4096, 4096)?,
                    m("attn.to_out", 4096, 4096)?,
                    m("gate_proj", 6144, 4096)?,
                    m("up_proj", 6144, 4096)?,
                    m("down_proj", 4096, 6144)?,
                ],
                cache,
            });
        }
        Ok(Self {
            projection: matrix(&weights["projection.weight"], d, 4096, 4096)?,
            heads: (0..7)
                .map(|i| matrix(&weights[&format!("audio_heads.{i}.weight")], d, 1024, 4096))
                .collect::<Result<_>>()?,
            embeddings: ptr(&weights["audio_embeddings.weight"], d),
            norm: ptr(&weights["norm.weight"], d),
            positions: ptr(&weights["pos_embedding.weight"], d),
            _weights: weights,
            layers,
            sequence: d.stream.alloc_zeros(2 * 8 * 4096)?,
            outputs: d.stream.alloc_zeros(2 * 8 * 4096)?,
            last: d.stream.alloc_zeros(2 * 4096)?,
            logits: d.stream.alloc_zeros(2 * 1024)?,
            flash: Flash::new(d.stream.clone(), 8, 16, 16, 256)?,
            position: 0,
        })
    }
}

pub struct Runtime {
    device: Device,
    weights: WeightCatalog,
    layers: Vec<Layer>,
    blas: Blas,
    kernels: Kernels,
    work: Work,
    flash: Flash,
    table: CudaSlice<bf16>,
    chunk: usize,
    capacity: usize,
    position: usize,
    depth: Depth,
    policy: Policy,
    frames: usize,
    cursor: CudaSlice<i32>,
    graphs: Option<Graphs>,
    embeddings: u64,
    norm_weight: u64,
    head_weight: Tensor<bf16, 2>,
    initial_free: usize,
    admitted_bytes: usize,
}

struct Graphs {
    global: Vec<CudaGraph>,
    depth: Vec<CudaGraph>,
}

fn ptr<T: crate::backend::gpu::DeviceRepr>(value: &CudaSlice<T>, d: &Device) -> u64 {
    value.device_ptr(&d.stream).0
}

fn matrix(
    value: &CudaSlice<bf16>,
    d: &Device,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        value,
        &d.stream,
        0,
        [rows as u32, width as u32],
        [width as u64, 1],
    )
}

impl Work {
    fn new(d: &Device, chunk: usize) -> Result<Self> {
        let n = chunk * 2;
        let s = &d.stream;
        Ok(Self {
            hidden: s.alloc_zeros(n * 4096)?,
            normalized: s.alloc_zeros(n * 4096)?,
            q: s.alloc_zeros(n * 4096)?,
            k: s.alloc_zeros(n * 4096)?,
            v: s.alloc_zeros(n * 4096)?,
            q_norm: s.alloc_zeros(n * 4096)?,
            k_norm: s.alloc_zeros(n * 1024)?,
            attention: s.alloc_zeros(n * 4096)?,
            projected: s.alloc_zeros(n * 4096)?,
            gate: s.alloc_zeros(n * 12288)?,
            up: s.alloc_zeros(n * 12288)?,
            activated: s.alloc_zeros(n * 12288)?,
            branch_q: s.alloc_zeros(chunk * 4096)?,
            branch_out: s.alloc_zeros(chunk * 4096)?,
            last: s.alloc_zeros(2 * 4096)?,
            logits: s.alloc_zeros(2 * 200000)?,
            ids: s.alloc_zeros(n)?,
        })
    }
}

impl Runtime {
    pub fn load(
        model: &Path,
        runtime: &Path,
        prompt: usize,
        frames: usize,
        device_limit: usize,
    ) -> Result<Self> {
        let ar = ArCheckpoint::open(model)?;
        ensure!(
            (1..=5000).contains(&prompt) && (1..=9000).contains(&frames),
            "AR request capacity"
        );
        let capacity = prompt + frames;
        let device = Device::new(0)?;
        let (free, total) = device.ctx.mem_get_info()?;
        let initial_free = free;
        ensure!(
            device_limit > 0 && device_limit <= total,
            "AR admitted device budget"
        );
        let external_reserve = free.saturating_sub(device_limit);
        let free = free.min(device_limit);
        let host_bytes = frames * 32768 * 4 + 65536;
        let host_available = usize::try_from(super::calibrate::host_memory(Path::new("/"))?)?;
        ensure!(
            host_available >= host_bytes + (4 << 30),
            "AR feature handoff and host reserve exceed available memory"
        );
        let reserve = ar.depth_manifest.resident_bytes + capacity * 294912 + total / 16;
        ensure!(
            free > reserve + (512 << 20),
            "AR fixed state exceeds free device budget"
        );
        let chunk = ((free - reserve) / (16 << 20)).clamp(8, 32);
        let work = Work::new(&device, chunk)?;
        let policy = Policy::new(&device)?;
        let ops = Ops::new(&device.ctx)?;
        let blas = Blas::new(
            &device,
            &ops,
            runtime,
            Rc::new(Closure::new(true, "Music3 AR".into())),
        )?;
        for rows in 1..=chunk {
            for (output, input) in [
                (4096, 4096),
                (1024, 4096),
                (12288, 4096),
                (4096, 12288),
                (200000, 4096),
                (6144, 4096),
                (4096, 6144),
            ] {
                blas.prepare_request_fixed((rows * 2) as u32, output, input)?;
            }
        }
        let mut caches = Vec::new();
        for _ in 0..36 {
            let mut branches = Vec::new();
            for _ in 0..2 {
                branches.push((
                    device.stream.alloc_zeros(capacity * 1024)?,
                    device.stream.alloc_zeros(capacity * 1024)?,
                ));
            }
            caches.push(branches);
        }
        let table = (0..capacity)
            .flat_map(|position| {
                (0..128).map(move |i| {
                    let frequency = 1.0f32 / 1e6f32.powf((2 * (i % 64)) as f32 / 128.0);
                    let angle = position as f32 * frequency;
                    bf16::from_f32(if i < 64 { angle.cos() } else { angle.sin() })
                })
            })
            .collect::<Vec<_>>();
        let table = device.stream.clone_htod(&table)?;
        let cursor = device.stream.alloc_zeros(1)?;
        let mut flash = Flash::new(device.stream.clone(), chunk, 32, 8, 128)?;
        flash.bind_position(&cursor, capacity)?;
        let depth = Depth::load(&ar, &device)?;
        let weights = WeightCatalog::load_dense(
            &model.join("language_model"),
            &ar.language,
            &device,
            external_reserve,
            0,
            false,
            host_bytes,
        )?;
        let mut layers = Vec::new();
        for (index, cache) in caches.into_iter().enumerate() {
            let p = format!("model.layers.{index}");
            let m = |name: &str| weights.matrix(&device.stream, &format!("{p}.{name}.weight"));
            let v = |name: &str| -> Result<u64> {
                Ok(ptr(weights.bf16(&format!("{p}.{name}.weight"))?, &device))
            };
            layers.push(Layer {
                norm: [v("input_layernorm")?, v("post_attention_layernorm")?],
                q_norm: v("self_attn.q_norm")?,
                k_norm: v("self_attn.k_norm")?,
                matrices: [
                    m("self_attn.q_proj")?,
                    m("self_attn.k_proj")?,
                    m("self_attn.v_proj")?,
                    m("self_attn.o_proj")?,
                    m("mlp.gate_proj")?,
                    m("mlp.up_proj")?,
                    m("mlp.down_proj")?,
                ],
                cache,
            });
        }
        let norm = cubin::module(&device.ctx, "norm")?;
        let kernels = Kernels {
            norm: norm.load_function("rms_norm_bf16")?,
            residual: norm.load_function("residual")?,
            silu: cubin::module(&device.ctx, "act")?.load_function("silu_mul_split")?,
            embed: ops.embed,
            rotary: cubin::module(&device.ctx, "rope")?.load_function("rotary_prefix_kv")?,
            code_embed: cubin::module(&device.ctx, "embed")?
                .load_function("embed_code_broadcast")?,
            feedback: cubin::module(&device.ctx, "embed")?
                .load_function("codebook_embedding_sum")?,
            policy: cubin::module(&device.ctx, "sample")?.load_function("guided_topk_bf16")?,
            draw: cubin::module(&device.ctx, "sample")?.load_function("categorical_sample_rows")?,
            rotary_device: cubin::module(&device.ctx, "rope")?
                .load_function("rotary_prefix_kv_device")?,
            advance: cubin::module(&device.ctx, "sample")?.load_function("advance")?,
        };
        device.finish_upload()?;
        eprintln!(
            "MUSIC3_AR_SETUP chunk={chunk} capacity={capacity} paired_rows=2 weights={} depth_bytes={} sms={}",
            weights.bytes, ar.depth_manifest.resident_bytes, device.info.sms
        );
        Ok(Self {
            embeddings: ptr(weights.bf16("model.embed_tokens.weight")?, &device),
            norm_weight: ptr(weights.bf16("model.norm.weight")?, &device),
            head_weight: weights.matrix(&device.stream, "lm_head.weight")?,
            device,
            weights,
            layers,
            blas,
            kernels,
            work,
            flash,
            table,
            chunk,
            capacity,
            position: 0,
            depth,
            policy,
            frames,
            cursor,
            graphs: None,
            initial_free,
            admitted_bytes: free,
        })
    }

    fn capture(&mut self, mut program: impl FnMut(&mut Self) -> Result<()>) -> Result<CudaGraph> {
        use cudarc::driver::sys;
        let s = self.device.stream.clone();
        s.memcpy_htod(&[0i32], &mut self.cursor)?;
        program(self)?;
        s.synchronize()?;
        s.memcpy_htod(&[0i32], &mut self.cursor)?;
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let result = program(self);
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        result?;
        let graph = captured?.context("Music3 graph capture was empty")?;
        graph.upload()?;
        s.synchronize()?;
        Ok(graph)
    }

    fn prepare_graphs(&mut self) -> Result<()> {
        if self.graphs.is_some() {
            return Ok(());
        }
        self.device
            .stream
            .memcpy_htod(&[151675, 0, 0, 0, 0, 0, 0, 0], &mut self.policy.codes)?;
        let saved = self.position;
        self.position = 0;
        let global = if self.weights.dense.is_none() {
            vec![self.capture(|ar| ar.decode_program())?]
        } else {
            let mut graphs = Vec::new();
            for index in 0..self.layers.len() {
                self.weights
                    .dense
                    .as_mut()
                    .context("streamed graph plan")?
                    .begin(index, &self.device.stream)?;
                graphs.push(self.capture(|ar| {
                    if index == 0 {
                        ar.embed_feedback()?;
                    }
                    ar.layer_compute(index, 1)
                })?);
                self.weights
                    .dense
                    .as_mut()
                    .context("streamed graph plan")?
                    .end(index, &self.device.stream)?;
            }
            graphs.push(self.capture(|ar| ar.decode_finish())?);
            graphs
        };
        let mut depth = Vec::new();
        for book in 1..8 {
            depth.push(self.capture(|ar| {
                ar.depth.position = if book == 1 { 0 } else { book };
                ar.depth_program(book)
            })?);
        }
        self.position = saved;
        self.device
            .stream
            .memcpy_htod(&[saved as i32], &mut self.cursor)?;
        eprintln!(
            "MUSIC3_AR_GRAPHS global={} depth={} streamed={} cursor_device=true",
            global.len(),
            depth.len(),
            self.weights.dense.is_some()
        );
        self.graphs = Some(Graphs { global, depth });
        let used = self
            .initial_free
            .saturating_sub(self.device.ctx.mem_get_info()?.0);
        ensure!(
            used <= self.admitted_bytes,
            "AR weights/state/graphs exceed admitted device bytes: used={used} admitted={}",
            self.admitted_bytes
        );
        eprintln!(
            "MUSIC3_AR_CAPACITY device_bytes={used} admitted_bytes={} capacity={} feature_host_bytes={} pinned_feature_tile_bytes=65536",
            self.admitted_bytes,
            self.capacity,
            self.frames * 131072
        );
        Ok(())
    }

    fn decode_finish(&mut self) -> Result<()> {
        self.finish_global(1)?;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.advance)
                .arg(&ptr(&self.cursor, &self.device))
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn decode_program(&mut self) -> Result<()> {
        self.embed_feedback()?;
        for index in 0..self.layers.len() {
            self.layer(index, 1)?;
        }
        self.decode_finish()
    }

    fn decode(&mut self, graphs: bool) -> Result<()> {
        ensure!(self.position < self.capacity, "AR decode cache capacity");
        if graphs {
            let global = &self
                .graphs
                .as_ref()
                .context("AR graphs not prepared")?
                .global;
            if let Some(streaming) = &mut self.weights.dense {
                for (index, graph) in global[..self.layers.len()].iter().enumerate() {
                    streaming.begin(index, &self.device.stream)?;
                    graph.launch()?;
                    streaming.end(index, &self.device.stream)?;
                }
                global.last().context("AR graph finish")?.launch()?;
            } else {
                global[0].launch()?;
            }
        } else {
            self.decode_program()?;
        }
        self.position += 1;
        Ok(())
    }

    fn distribution(&self, semantic: bool) -> Result<()> {
        let d = &self.device;
        let (logits, vocab, begin, count, extra, conditional) = if semantic {
            (&self.work.logits, 200000, 151675, 16384, 151670, 50)
        } else {
            (&self.depth.logits, 1024, 0, 1024, -1, 0)
        };
        unsafe {
            d.stream
                .launch_builder(&self.kernels.policy)
                .arg(&ptr(logits, d))
                .arg(&ptr(&self.policy.probabilities, d))
                .arg(&ptr(&self.policy.scratch, d))
                .arg(&ptr(&self.policy.thresholds, d))
                .arg(&ptr(&self.policy.keep, d))
                .arg(&vocab)
                .arg(&begin)
                .arg(&count)
                .arg(&extra)
                .arg(&conditional)
                .arg(&50i32)
                .arg(&1.5f32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn draw(&self, codebook: usize, seed: u64, draw: u64) -> Result<()> {
        ensure!(codebook < 8, "AR draw codebook");
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.draw)
                .arg(&ptr(&self.policy.probabilities, &self.device))
                .arg(&(ptr(&self.policy.codes, &self.device) + (codebook * 4) as u64))
                .arg(&1i32)
                .arg(&(if codebook == 0 { 200000i64 } else { 1024i64 }))
                .arg(&seed)
                .arg(&draw)
                .arg(&256i32)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn code_embedding(&self, codebook: usize) -> Result<()> {
        ensure!(codebook < 7, "AR projected codebook");
        let d = &self.device;
        let (table, offset, vocab) = if codebook == 0 {
            (self.embeddings, 0, 200000)
        } else {
            (self.depth.embeddings, ((codebook - 1) * 1024) as i32, 1024)
        };
        unsafe {
            d.stream
                .launch_builder(&self.kernels.code_embed)
                .arg(&table)
                .arg(&(ptr(&self.policy.codes, d) + (codebook * 4) as u64))
                .arg(&ptr(&self.work.hidden, d))
                .arg(&offset)
                .arg(&vocab)
                .arg(&4096i32)
                .arg(&2i32)
                .launch(LaunchConfig::for_num_elems(8192))?;
        }
        self.project_depth(&self.work.hidden, codebook + 1)
    }

    fn frame_depth(
        &mut self,
        seed: Option<(u64, u64)>,
        graphs: bool,
        mut observe: impl FnMut(&Self, usize) -> Result<()>,
    ) -> Result<()> {
        self.depth.position = 0;
        for codebook in 1..8 {
            if graphs {
                self.graphs
                    .as_ref()
                    .context("AR depth graphs not prepared")?
                    .depth[codebook - 1]
                    .launch()?;
                self.depth.position = codebook + 1;
            } else {
                self.depth_program(codebook)?;
            }
            observe(self, codebook)?;
            if let Some((seed, base)) = seed {
                self.draw(codebook, seed, base + codebook as u64)?;
            }
        }
        Ok(())
    }

    fn depth_program(&mut self, codebook: usize) -> Result<()> {
        if codebook == 1 {
            self.project_depth(&self.work.last, 0)?;
            self.feature_part(&self.work.last, 0)?;
        }
        self.code_embedding(codebook - 1)?;
        self.depth_step(codebook, true)?;
        self.distribution(false)?;
        self.feature_part(&self.depth.last, codebook)
    }

    fn feature_part(&self, source: &CudaSlice<bf16>, part: usize) -> Result<()> {
        unsafe {
            dtod_raw(
                ptr(&self.policy.features, &self.device) + (part * 8192) as u64,
                ptr(source, &self.device),
                8192,
                &self.device.stream,
            )?;
        }
        Ok(())
    }

    fn embed_feedback(&self) -> Result<()> {
        let d = &self.device;
        unsafe {
            d.stream
                .launch_builder(&self.kernels.feedback)
                .arg(&self.embeddings)
                .arg(&self.depth.embeddings)
                .arg(&ptr(&self.policy.codes, d))
                .arg(&ptr(&self.work.hidden, d))
                .arg(&151675i32)
                .arg(&16384i32)
                .arg(&7i32)
                .arg(&1024i32)
                .arg(&4096i32)
                .arg(&2i32)
                .arg(&(8.0f32.sqrt().recip()))
                .launch(LaunchConfig::for_num_elems(8192))?;
        }
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        recorded: Option<&[u32]>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.prepare_graphs()?;
        self.run(prompt, recorded, seed, true, |_, _, _| Ok(()))
    }

    fn run(
        &mut self,
        prompt: &[u32],
        recorded: Option<&[u32]>,
        seed: u64,
        graphs: bool,
        mut observe: impl FnMut(&Self, usize, usize) -> Result<()>,
    ) -> Result<Vec<f32>> {
        if let Some(codes) = recorded {
            ensure!(
                (codes.len() == (self.frames + 1) * 8
                    || (codes.len() % 8 == 1
                        && codes.len() <= self.frames * 8 + 1
                        && codes.last() == Some(&151670)))
                    && codes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .all(|row| (151675..168059).contains(&row[0])
                            && row[1..].iter().all(|&id| id < 1024)),
                "AR recorded draw extent/range"
            );
        }
        self.prefill(prompt)?;
        let mut features = Vec::with_capacity(self.frames * 32768);
        for frame in 0..=self.frames {
            self.head()?;
            self.distribution(true)?;
            observe(self, frame, 0)?;
            if let Some(codes) = recorded {
                let end = (frame * 8 + 8).min(codes.len());
                ensure!(frame * 8 < end, "AR recorded draw exhaustion");
                self.device.stream.memcpy_htod(
                    &codes[frame * 8..end],
                    &mut self.policy.codes.slice_mut(..end - frame * 8),
                )?;
            } else {
                self.draw(0, seed, (frame * 8) as u64)?;
            }
            let semantic = self
                .device
                .stream
                .clone_dtoh(&self.policy.codes.slice(..1))?[0];
            if semantic == 151670 {
                break;
            }
            ensure!(
                (151675..168059).contains(&semantic),
                "AR sampled semantic code range"
            );
            self.frame_depth(
                recorded.is_none().then_some((seed, (frame * 8) as u64)),
                graphs,
                |ar, book| observe(ar, frame, book),
            )?;
            if frame > 0 {
                unsafe {
                    dtoh_async(
                        self.policy.host_features.as_mut_slice()?,
                        ptr(&self.policy.features, &self.device),
                        &self.device.stream,
                    )?;
                }
                self.device.stream.synchronize()?;
                features.extend(
                    self.policy
                        .host_features
                        .as_slice()?
                        .iter()
                        .copied()
                        .map(f32::from),
                );
                if frame == self.frames {
                    break;
                }
            }
            self.decode(graphs)?;
        }
        ensure!(!features.is_empty(), "AR generated zero audio frames");
        Ok(features)
    }

    fn linear(
        &self,
        weight: Tensor<bf16, 2>,
        input: &CudaSlice<bf16>,
        output: &CudaSlice<bf16>,
        rows: usize,
    ) -> Result<()> {
        self.blas.linear_typed(
            ExecutionPhase::Prefill,
            LinearWeightRef::Bf16(weight),
            matrix(input, &self.device, rows, weight.shape[1] as usize)?,
            &mut TensorMut(matrix(
                output,
                &self.device,
                rows,
                weight.shape[0] as usize,
            )?),
        )
    }

    fn norm(
        &self,
        input: &CudaSlice<bf16>,
        weight: u64,
        output: &CudaSlice<bf16>,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.norm)
                .arg(&ptr(input, &self.device))
                .arg(&weight)
                .arg(&ptr(output, &self.device))
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&1e-6f32)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn add(&self, input: &CudaSlice<bf16>, other: &CudaSlice<bf16>, count: usize) -> Result<()> {
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.residual)
                .arg(&ptr(input, &self.device))
                .arg(&ptr(other, &self.device))
                .arg(&(count as i32))
                .launch(LaunchConfig::for_num_elems(count as u32))?;
        }
        Ok(())
    }

    fn layer(&mut self, index: usize, rows: usize) -> Result<()> {
        if let Some(streaming) = &mut self.weights.dense {
            streaming.begin(index, &self.device.stream)?;
        }
        self.layer_compute(index, rows)?;
        if let Some(streaming) = &mut self.weights.dense {
            streaming.end(index, &self.device.stream)?;
        }
        Ok(())
    }

    fn layer_compute(&mut self, index: usize, rows: usize) -> Result<()> {
        let layer = &self.layers[index];
        let n = rows * 2;
        self.norm(
            &self.work.hidden,
            layer.norm[0],
            &self.work.normalized,
            n,
            4096,
        )?;
        for (weight, out) in
            layer.matrices[..3]
                .iter()
                .zip([&self.work.q, &self.work.k, &self.work.v])
        {
            self.linear(*weight, &self.work.normalized, out, n)?;
        }
        self.norm(&self.work.q, layer.q_norm, &self.work.q_norm, n * 32, 128)?;
        self.norm(&self.work.k, layer.k_norm, &self.work.k_norm, n * 8, 128)?;
        for branch in 0..2 {
            let (k, v) = &layer.cache[branch];
            let d = &self.device;
            let q = ptr(&self.work.q_norm, d) + (branch * rows * 4096 * 2) as u64;
            let key = ptr(&self.work.k_norm, d) + (branch * rows * 1024 * 2) as u64;
            let value = ptr(&self.work.v, d) + (branch * rows * 1024 * 2) as u64;
            unsafe {
                d.stream
                    .launch_builder(&self.kernels.rotary_device)
                    .arg(&q)
                    .arg(&key)
                    .arg(&value)
                    .arg(&ptr(k, d))
                    .arg(&ptr(v, d))
                    .arg(&ptr(&self.table, d))
                    .arg(&ptr(&self.cursor, d))
                    .arg(&(rows as i32))
                    .arg(&32i32)
                    .arg(&8i32)
                    .arg(&128i32)
                    .arg(&128i32)
                    .arg(&(self.capacity as i32))
                    .launch(LaunchConfig::for_num_elems((rows * 40 * 128) as u32))?;
                dtod_raw(ptr(&self.work.branch_q, d), q, rows * 4096 * 2, &d.stream)?;
            }
            self.flash.run(
                &self.work.branch_q,
                4096,
                k,
                v,
                &mut self.work.branch_out,
                Shape {
                    rows,
                    start: self.position,
                    capacity: self.capacity,
                    q_heads: 32,
                    kv_heads: 8,
                    head_dim: 128,
                    causal: true,
                    window_left: None,
                },
            )?;
            unsafe {
                dtod_raw(
                    ptr(&self.work.attention, d) + (branch * rows * 4096 * 2) as u64,
                    ptr(&self.work.branch_out, d),
                    rows * 4096 * 2,
                    &d.stream,
                )?;
            }
        }
        self.finish_layer(layer, n)
    }

    fn finish_layer(&self, layer: &Layer, n: usize) -> Result<()> {
        let width = layer.matrices[4].shape[0] as usize;
        self.linear(
            layer.matrices[3],
            &self.work.attention,
            &self.work.projected,
            n,
        )?;
        self.add(&self.work.hidden, &self.work.projected, n * 4096)?;
        self.norm(
            &self.work.hidden,
            layer.norm[1],
            &self.work.normalized,
            n,
            4096,
        )?;
        self.linear(layer.matrices[4], &self.work.normalized, &self.work.gate, n)?;
        self.linear(layer.matrices[5], &self.work.normalized, &self.work.up, n)?;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.silu)
                .arg(&ptr(&self.work.gate, &self.device))
                .arg(&ptr(&self.work.up, &self.device))
                .arg(&ptr(&self.work.activated, &self.device))
                .arg(&((n * width) as i32))
                .arg(&0i32)
                .launch(LaunchConfig::for_num_elems((n * width) as u32))?;
        }
        self.linear(
            layer.matrices[6],
            &self.work.activated,
            &self.work.projected,
            n,
        )?;
        self.add(&self.work.hidden, &self.work.projected, n * 4096)?;
        Ok(())
    }

    fn depth_layer(&mut self, index: usize, rows: usize, start: usize) -> Result<()> {
        let layer = &self.depth.layers[index];
        let n = 2 * rows;
        self.norm(
            &self.work.hidden,
            layer.norm[0],
            &self.work.normalized,
            n,
            4096,
        )?;
        for (weight, out) in
            layer.matrices[..3]
                .iter()
                .zip([&self.work.q, &self.work.k, &self.work.v])
        {
            self.linear(*weight, &self.work.normalized, out, n)?;
        }
        for branch in 0..2 {
            let (k, v) = &layer.cache[branch];
            let d = &self.device;
            let offset = (branch * rows * 4096 * 2) as u64;
            let q = ptr(&self.work.q, d) + offset;
            let key = ptr(&self.work.k, d) + offset;
            let value = ptr(&self.work.v, d) + offset;
            unsafe {
                d.stream
                    .launch_builder(&self.kernels.rotary)
                    .arg(&q)
                    .arg(&key)
                    .arg(&value)
                    .arg(&ptr(k, d))
                    .arg(&ptr(v, d))
                    .arg(&0u64)
                    .arg(&(start as i32))
                    .arg(&(rows as i32))
                    .arg(&16i32)
                    .arg(&16i32)
                    .arg(&256i32)
                    .arg(&0i32)
                    .arg(&16i32)
                    .launch(LaunchConfig::for_num_elems((rows * 32 * 256) as u32))?;
                dtod_raw(ptr(&self.work.branch_q, d), q, rows * 4096 * 2, &d.stream)?;
            }
            self.depth.flash.run(
                &self.work.branch_q,
                4096,
                k,
                v,
                &mut self.work.branch_out,
                Shape {
                    rows,
                    start,
                    capacity: 16,
                    q_heads: 16,
                    kv_heads: 16,
                    head_dim: 256,
                    causal: true,
                    window_left: None,
                },
            )?;
            unsafe {
                dtod_raw(
                    ptr(&self.work.attention, d) + offset,
                    ptr(&self.work.branch_out, d),
                    rows * 4096 * 2,
                    &d.stream,
                )?;
            }
        }
        self.finish_layer(layer, n)
    }

    fn project_depth(&self, input: &CudaSlice<bf16>, position: usize) -> Result<()> {
        ensure!(position < 8, "depth projection position");
        self.linear(self.depth.projection, input, &self.work.projected, 2)?;
        for branch in 0..2 {
            unsafe {
                dtod_raw(
                    ptr(&self.depth.sequence, &self.device)
                        + ((branch * 8 + position) * 4096 * 2) as u64,
                    ptr(&self.work.projected, &self.device) + (branch * 4096 * 2) as u64,
                    4096 * 2,
                    &self.device.stream,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn depth_embed(&mut self, code: u32, table: u64) -> Result<()> {
        self.device
            .stream
            .memcpy_htod(&[code, code], &mut self.work.ids.slice_mut(..2))?;
        unsafe {
            self.device
                .stream
                .launch_builder(&self.kernels.embed)
                .arg(&table)
                .arg(&ptr(&self.work.ids, &self.device))
                .arg(&ptr(&self.work.hidden, &self.device))
                .arg(&2i32)
                .arg(&4096i32)
                .launch(LaunchConfig::for_num_elems(8192))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn depth_begin(&mut self, semantic: u32) -> Result<()> {
        ensure!(semantic < 16384, "depth semantic code out of range");
        self.depth.position = 0;
        self.project_depth(&self.work.last, 0)?;
        self.depth_embed(semantic + 151675, self.embeddings)?;
        self.project_depth(&self.work.hidden, 1)
    }

    #[cfg(test)]
    fn depth_append(&mut self, codebook: usize, code: u32) -> Result<()> {
        ensure!(
            (1..7).contains(&codebook) && code < 1024,
            "depth codebook/offset range"
        );
        self.depth_embed(code + ((codebook - 1) * 1024) as u32, self.depth.embeddings)?;
        self.project_depth(&self.work.hidden, codebook + 1)
    }

    fn depth_step(&mut self, codebook: usize, cached: bool) -> Result<()> {
        ensure!(
            (1..=7).contains(&codebook)
                && self.depth.position == if codebook == 1 { 0 } else { codebook },
            "depth cache order/reset"
        );
        let start = if cached { self.depth.position } else { 0 };
        let end = codebook + 1;
        let rows = end - start;
        for branch in 0..2 {
            let destination =
                ptr(&self.work.hidden, &self.device) + (branch * rows * 4096 * 2) as u64;
            unsafe {
                dtod_raw(
                    destination,
                    ptr(&self.depth.sequence, &self.device)
                        + ((branch * 8 + start) * 4096 * 2) as u64,
                    rows * 4096 * 2,
                    &self.device.stream,
                )?;
                self.device
                    .stream
                    .launch_builder(&self.kernels.residual)
                    .arg(&destination)
                    .arg(&(self.depth.positions + (start * 4096 * 2) as u64))
                    .arg(&((rows * 4096) as i32))
                    .launch(LaunchConfig::for_num_elems((rows * 4096) as u32))?;
            }
        }
        for layer in 0..4 {
            self.depth_layer(layer, rows, start)?;
        }
        self.norm(
            &self.work.hidden,
            self.depth.norm,
            &self.work.normalized,
            2 * rows,
            4096,
        )?;
        for branch in 0..2 {
            unsafe {
                dtod_raw(
                    ptr(&self.depth.outputs, &self.device)
                        + ((branch * 8 + start) * 4096 * 2) as u64,
                    ptr(&self.work.normalized, &self.device) + (branch * rows * 4096 * 2) as u64,
                    rows * 4096 * 2,
                    &self.device.stream,
                )?;
                dtod_raw(
                    ptr(&self.depth.last, &self.device) + (branch * 4096 * 2) as u64,
                    ptr(&self.work.normalized, &self.device)
                        + ((branch * rows + rows - 1) * 4096 * 2) as u64,
                    4096 * 2,
                    &self.device.stream,
                )?;
            }
        }
        self.linear(
            self.depth.heads[codebook - 1],
            &self.depth.last,
            &self.depth.logits,
            2,
        )?;
        self.depth.position = end;
        Ok(())
    }

    fn forward(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.chunk && self.position + rows <= self.capacity,
            "AR global cache capacity"
        );
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.cursor)?;
        for index in 0..self.layers.len() {
            self.layer(index, rows)?;
        }
        self.finish_global(rows)?;
        self.position += rows;
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.cursor)?;
        Ok(())
    }

    fn finish_global(&self, rows: usize) -> Result<()> {
        self.norm(
            &self.work.hidden,
            self.norm_weight,
            &self.work.normalized,
            rows * 2,
            4096,
        )?;
        for branch in 0..2 {
            unsafe {
                dtod_raw(
                    ptr(&self.work.last, &self.device) + (branch * 4096 * 2) as u64,
                    ptr(&self.work.normalized, &self.device)
                        + ((branch * rows + rows - 1) * 4096 * 2) as u64,
                    4096 * 2,
                    &self.device.stream,
                )?;
            }
        }
        Ok(())
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        ensure!(
            self.position == 0 && ids.len().is_multiple_of(2),
            "AR paired prompt/reset contract"
        );
        let prompt = ids.len() / 2;
        ensure!(
            prompt > 0
                && prompt <= 5000
                && prompt < self.capacity
                && ids.iter().all(|&id| id < 200000),
            "AR prompt extent/token"
        );
        for start in (0..prompt).step_by(self.chunk) {
            let rows = self.chunk.min(prompt - start);
            let mut pair = Vec::with_capacity(2 * rows);
            for branch in 0..2 {
                pair.extend_from_slice(
                    &ids[branch * prompt + start..branch * prompt + start + rows],
                );
            }
            self.device
                .stream
                .memcpy_htod(&pair, &mut self.work.ids.slice_mut(..pair.len()))?;
            unsafe {
                self.device
                    .stream
                    .launch_builder(&self.kernels.embed)
                    .arg(&self.embeddings)
                    .arg(&ptr(&self.work.ids, &self.device))
                    .arg(&ptr(&self.work.hidden, &self.device))
                    .arg(&((rows * 2) as i32))
                    .arg(&4096i32)
                    .launch(LaunchConfig::for_num_elems((rows * 2 * 4096) as u32))?;
            }
            self.forward(rows)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn feedback(&mut self, values: &[bf16]) -> Result<()> {
        ensure!(values.len() == 8192, "AR paired feedback shape");
        self.device
            .stream
            .memcpy_htod(values, &mut self.work.hidden.slice_mut(..8192))?;
        self.forward(1)
    }

    fn head(&self) -> Result<()> {
        self.linear(self.head_weight, &self.work.last, &self.work.logits, 2)
    }

    #[cfg(test)]
    fn logits(&self) -> Result<Vec<f32>> {
        self.head()?;
        Ok(self
            .device
            .stream
            .clone_dtoh(&self.work.logits)?
            .into_iter()
            .map(f32::from)
            .collect())
    }

    #[cfg(test)]
    fn hidden(&self) -> Result<Vec<f32>> {
        Ok(self
            .device
            .stream
            .clone_dtoh(&self.work.last)?
            .into_iter()
            .map(f32::from)
            .collect())
    }

    #[cfg(test)]
    fn kv_tail(&self, layer: usize, keys: bool) -> Result<Vec<f32>> {
        ensure!(
            layer < self.layers.len() && self.position > 0,
            "AR KV snapshot index"
        );
        let mut values = Vec::with_capacity(2048);
        for (k, v) in &self.layers[layer].cache {
            let buffer = if keys { k } else { v };
            for head in 0..8 {
                let offset = (head * self.capacity + self.position - 1) * 128;
                values.extend(
                    self.device
                        .stream
                        .clone_dtoh(&buffer.slice(offset..offset + 128))?
                        .into_iter()
                        .map(f32::from),
                );
            }
        }
        Ok(values)
    }
}

/// One production synthesis: the AR stage runs the prompt, its condition
/// features hand off to the synthesis stage, the AR runtime is released
/// before the synthesis weights load, and the flow + vocoder emit stereo
/// PCM. The AR runtime is device-0 resident; `device` selects the
/// synthesis device and must be 0 until the AR runtime is multi-device.
pub fn generate(
    model: &Path,
    request: &crate::music3::Music3Request,
    (device, runtime): (usize, &Path),
) -> Result<crate::music3::Music3Output> {
    use super::engine_music3_syn::{Music3Syn, Workspace};

    let spec = crate::music3::SynSpec::read(model)?;
    let length = spec.latent_length(request.frames);
    let channels = spec.in_channels as usize;
    ensure!(!request.prompt_ids.is_empty(), "empty AR prompt");
    ensure!(
        request.initial.len() == channels * length,
        "initial latent extent: got {} elements, in_channels * latent_length = {}",
        request.initial.len(),
        channels * length
    );
    ensure!(device == 0, "AR runtime is device-0 resident");
    let handoff_width = spec.cond_layers as usize * spec.cond_hidden as usize;
    let total: usize = spec.upsampling.iter().map(|r| *r as usize).product();

    let device_ctx = Device::new(0)?;
    let (free, _t) = device_ctx.ctx.mem_get_info()?;
    let mut ar = Runtime::load(
        model,
        runtime,
        request.prompt_ids.len(),
        request.frames,
        free,
    )?;
    let features = ar.generate(&request.prompt_ids, None, request.seed)?;
    ensure!(
        features.len() == request.frames * handoff_width,
        "AR feature handoff: got {} elements, frames * cond_layers * cond_hidden = {}",
        features.len(),
        request.frames * handoff_width
    );
    drop(ar);

    let (post_release, _t) = device_ctx.ctx.mem_get_info()?;
    let syn_bytes = syn_resident_bytes(model)?;
    let reserve = 1usize << 30;
    ensure!(
        post_release > syn_bytes + reserve,
        "synthesis resident need {} + {} reserve exceeds free {} after AR release",
        syn_bytes,
        reserve,
        post_release
    );
    let engine = Music3Syn::load(model, &device_ctx, length)?;
    let mut ws: Workspace = Music3Syn::workspace(&engine.spec, &device_ctx, length)?;
    let s = &device_ctx.stream;
    let d_features = s.clone_htod(&features)?;
    let d_initial = s.clone_htod(&request.initial)?;
    let mut audio = unsafe { s.alloc::<f32>(2 * length * total)? };
    engine.synthesize(&mut ws, &d_features, request.frames, &d_initial, &mut audio)?;
    let pcm = s.clone_dtoh(&audio)?;
    Ok(crate::music3::Music3Output { pcm })
}

/// Sum of the three synthesis component safetensors: the F32 resident
/// weight estimate for the release assert.
fn syn_resident_bytes(model: &Path) -> Result<usize> {
    let mut bytes = 0usize;
    for component in ["transformer", "condition_encoder", "vocoder"] {
        for entry in std::fs::read_dir(model.join(component))
            .with_context(|| format!("synthesis component dir {component}"))?
        {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors") {
                bytes += entry
                    .metadata()
                    .with_context(|| format!("component file {name}"))?
                    .len() as usize;
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use std::collections::BTreeMap;

    fn read(root: &Path, file: &str, key: &str) -> Result<Vec<f32>> {
        let data = std::fs::read(root.join(file))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let tensor = tensors.tensor(key)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32,
            "Runner FAIL reference dtype {key}"
        );
        Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    }

    #[derive(Default)]
    struct Distance {
        max: f64,
        squared: f64,
        count: usize,
    }
    fn compare(
        stats: &mut BTreeMap<String, Distance>,
        key: &str,
        a: &[f32],
        b: &[f32],
    ) -> Result<()> {
        ensure!(
            a.len() == b.len() && a.iter().chain(b).all(|x| x.is_finite()),
            "Runner FAIL shape/nonfinite {key}"
        );
        let cell = stats.entry(key.into()).or_default();
        for (&a, &b) in a.iter().zip(b) {
            let delta = f64::from(a) - f64::from(b);
            cell.max = cell.max.max(delta.abs());
            cell.squared += delta * delta;
            cell.count += 1;
        }
        Ok(())
    }

    fn branch_control(ar: &mut Runtime, feedback: &[bf16]) -> Result<()> {
        let position = ar.position;
        let initial = ar.device.stream.clone_dtoh(&ar.work.last)?;
        ar.feedback(feedback)?;
        let baseline = ar.hidden()?;
        ar.position = position;
        let saved = ar.device.stream.clone_dtoh(&ar.layers[0].cache[0].1)?;
        let mut poison = saved.clone();
        for head in 0..8 {
            for token in 0..position {
                for d in 0..128 {
                    let i = (head * ar.capacity + token) * 128 + d;
                    poison[i] = bf16::from_f32(f32::from(poison[i]) + 64.0);
                }
            }
        }
        ar.device
            .stream
            .memcpy_htod(&poison, &mut ar.layers[0].cache[0].1)?;
        ar.feedback(feedback)?;
        let changed = ar.hidden()?;
        let isolated = baseline[4096..]
            .iter()
            .zip(&changed[4096..])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let detected = baseline[..4096]
            .iter()
            .zip(&changed[..4096])
            .any(|(a, b)| a.to_bits() != b.to_bits());
        ar.device
            .stream
            .memcpy_htod(&saved, &mut ar.layers[0].cache[0].1)?;
        ar.device.stream.memcpy_htod(&initial, &mut ar.work.last)?;
        ar.position = position;
        ensure!(
            isolated && detected,
            "DEFECT paired CFG alias control isolated={isolated} detected={detected}"
        );
        println!("PASS music3_cfg_cache_control branch1_bits=0 branch0_perturbation_detected=true");
        Ok(())
    }

    #[test]
    #[ignore = "requires a GPU lease, FF_MODEL and FF_MUSIC3_REFERENCE oracle directory"]
    fn music3_ar_reference() -> Result<()> {
        let model =
            std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
        let root = std::path::PathBuf::from(
            std::env::var_os("FF_MUSIC3_REFERENCE").context("FF_MUSIC3_REFERENCE missing")?,
        );
        let freeze_name = std::env::var("FF_MUSIC3_FREEZE").context("FF_MUSIC3_FREEZE missing")?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(freeze_name).join("FROZEN.json"))?)?;
        let frames = usize::try_from(
            frozen["frames"]
                .as_u64()
                .context("Runner FAIL frozen frame extent")?,
        )?;
        ensure!(
            frozen["status"] == "FROZEN" && [25, 50, 1500].contains(&frames),
            "Runner FAIL supplementary freeze scope"
        );
        let oracle = root.join(frozen["oracle"].as_str().context("oracle path")?);
        let bf16 = root.join(
            frozen["references"][0]
                .as_str()
                .context("Runner FAIL first BF16 reference")?,
        );
        let data = std::fs::read(oracle.join("prompt.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;
        let ids = tensors.tensor("ar_prompt_ids")?;
        ensure!(
            ids.dtype() == safetensors::Dtype::I64 && ids.shape().len() == 2 && ids.shape()[0] == 2,
            "Runner FAIL prompt layout"
        );
        let ids = ids
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|x| u32::try_from(i64::from_le_bytes(*x)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let scratch = std::env::temp_dir().join(format!("ff-music3-{}", std::process::id()));
        std::fs::create_dir(&scratch)?;
        let device_limit = std::env::var("FF_MUSIC3_DEVICE_BYTES")
            .context("FF_MUSIC3_DEVICE_BYTES missing")?
            .parse()?;
        let mut ar = Runtime::load(&model, &scratch, ids.len() / 2, frames, device_limit)?;
        let placement =
            std::env::var("FF_MUSIC3_PLACEMENT").context("FF_MUSIC3_PLACEMENT missing")?;
        ensure!(
            match placement.as_str() {
                "resident" => ar.weights.dense.is_none(),
                "streamed" => ar.weights.dense.is_some(),
                _ => false,
            },
            "Runner FAIL forced placement {placement}"
        );
        let mut defects = if frames == 25 {
            isolated_reference(&mut ar, &ids, &bf16, &oracle, &frozen)?
        } else {
            0
        };
        defects += contract_reference(&mut ar, &ids, &bf16, &oracle, &frozen)?;
        drop(ar);
        std::fs::remove_dir_all(scratch)?;
        ensure!(defects == 0, "DEFECT Music3 global cells={defects}");
        println!("PASS music3_ar_reference frames={frames} paired_cfg=2 contract_heads=8");
        Ok(())
    }

    fn rotary_position_control(ar: &Runtime) -> Result<usize> {
        let d = &ar.device;
        let capacity = 14000usize;
        let pattern = |count: usize, stride: usize| {
            (0..count)
                .map(|i| bf16::from_f32(((i * stride % 251) as f32 - 125.0) / 128.0))
                .collect::<Vec<_>>()
        };
        let input = pattern(4096, 17);
        let keys = pattern(1024, 19);
        let values = pattern(1024, 23);
        let table = pattern(capacity * 128, 13);
        let rotate = |input: &[bf16], position: usize| {
            input
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let half = i % 128 >= 64;
                    let b = input[if half { i - 64 } else { i + 64 }];
                    let c = table[position * 128 + i % 64];
                    let s = table[position * 128 + i % 64 + 64];
                    let ac = bf16::from_f32(f32::from(*a) * f32::from(c));
                    let bs = bf16::from_f32(f32::from(b) * f32::from(s));
                    bf16::from_f32(if half {
                        f32::from(ac) + f32::from(bs)
                    } else {
                        f32::from(ac) - f32::from(bs)
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut q = d.stream.clone_htod(&input)?;
        let key = d.stream.clone_htod(&keys)?;
        let value = d.stream.clone_htod(&values)?;
        let table_device = d.stream.clone_htod(&table)?;
        let k = d.stream.alloc_zeros::<bf16>(8 * capacity * 128)?;
        let v = d.stream.alloc_zeros::<bf16>(8 * capacity * 128)?;
        let mut cursor = d.stream.alloc_zeros::<i32>(1)?;
        let mut expected_k = vec![bf16::ZERO; 8 * capacity * 128];
        let mut expected_v = expected_k.clone();
        let mut changed = 0;
        let mut detected = false;
        for (position, clamp) in [
            (10239usize, false),
            (10240, false),
            (10241, false),
            (13999, false),
            (10240, true),
        ] {
            let start = if clamp { 10239 } else { position as i32 };
            d.stream.memcpy_htod(&input, &mut q)?;
            d.stream.memcpy_htod(&[start], &mut cursor)?;
            unsafe {
                d.stream
                    .launch_builder(&ar.kernels.rotary_device)
                    .arg(&ptr(&q, d))
                    .arg(&ptr(&key, d))
                    .arg(&ptr(&value, d))
                    .arg(&ptr(&k, d))
                    .arg(&ptr(&v, d))
                    .arg(&ptr(&table_device, d))
                    .arg(&ptr(&cursor, d))
                    .arg(&1i32)
                    .arg(&32i32)
                    .arg(&8i32)
                    .arg(&128i32)
                    .arg(&128i32)
                    .arg(&(capacity as i32))
                    .launch(LaunchConfig::for_num_elems(40 * 128))?;
            }
            let delta = d
                .stream
                .clone_dtoh(&q)?
                .iter()
                .zip(rotate(&input, position))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if clamp {
                detected = delta > 0;
            } else {
                changed += delta;
                for (i, rotated) in rotate(&keys, position).into_iter().enumerate() {
                    let offset = (i / 128 * capacity + position) * 128 + i % 128;
                    expected_k[offset] = rotated;
                    expected_v[offset] = values[i];
                }
            }
        }
        for (actual, expected) in [(&k, expected_k), (&v, expected_v)] {
            changed += d
                .stream
                .clone_dtoh(actual)?
                .iter()
                .zip(expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        let pass = changed == 0 && detected;
        println!(
            "{} music3_position_control positions=10239,10240,10241,13999 capacity=14000 changed_values={changed} clamped_cursor_detected={detected}",
            if pass { "PASS" } else { "DEFECT" }
        );
        Ok(usize::from(!pass))
    }

    fn isolated_reference(
        ar: &mut Runtime,
        ids: &[u32],
        bf16: &Path,
        oracle: &Path,
        frozen: &serde_json::Value,
    ) -> Result<usize> {
        let position_defects = rotary_position_control(ar)?;
        ar.prefill(ids)?;
        let feedback = read(bf16, "frame-00000.safetensors", "ar_feedback")?
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        branch_control(ar, &feedback)?;
        let mut stats = BTreeMap::new();
        for step in -1..=25 {
            let file = format!("frame-{step:05}.safetensors");
            if step >= 0 {
                compare(
                    &mut stats,
                    "ar_lm_head_logits",
                    &ar.logits()?,
                    &read(oracle, &file, "ar_lm_head_logits")?,
                )?;
                compare(
                    &mut stats,
                    "ar_global_last_hidden",
                    &ar.hidden()?,
                    &read(oracle, &file, "ar_global_last_hidden")?,
                )?;
            }
            if step < 25 {
                if step >= 0 {
                    let feedback = read(bf16, &file, "ar_feedback")?
                        .into_iter()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>();
                    ar.feedback(&feedback)?;
                }
                for layer in [0, 35] {
                    for keys in [true, false] {
                        let key = format!(
                            "ar_next_kv_{layer}_{}",
                            if keys { "keys" } else { "values" }
                        );
                        compare(
                            &mut stats,
                            &key,
                            &ar.kv_tail(layer, keys)?,
                            &read(oracle, &file, &key)?,
                        )?;
                    }
                }
            }
        }
        let mut defects = position_defects;
        for (key, cell) in stats {
            let bound = &frozen["cells"][&key]["bound"];
            let mut max = bound["max"].as_f64().context("Runner FAIL max bound")?;
            let mut rmse = bound["rmse"].as_f64().context("Runner FAIL RMSE bound")?;
            if key == "ar_lm_head_logits" {
                max = max.min(
                    frozen["legacy_gates_unchanged"]["lm_head_logits"]["bound"]["max"]
                        .as_f64()
                        .context("legacy max")?,
                );
                rmse = rmse.min(
                    frozen["legacy_gates_unchanged"]["lm_head_logits"]["bound"]["rmse"]
                        .as_f64()
                        .context("legacy RMSE")?,
                );
            }
            let distance = (cell.squared / cell.count as f64).sqrt();
            let pass = cell.max <= max && distance <= rmse;
            defects += usize::from(!pass);
            println!(
                "{} music3_global_cell key={key} max={} rmse={distance} bound_max={max} bound_rmse={rmse} count={}",
                if pass { "PASS" } else { "DEFECT" },
                cell.max,
                cell.count
            );
        }
        defects += depth_reference(ar, bf16, oracle, frozen)?;
        Ok(defects)
    }

    fn read_device(
        ar: &Runtime,
        value: &CudaSlice<bf16>,
        offset: usize,
        count: usize,
    ) -> Result<Vec<f32>> {
        Ok(ar
            .device
            .stream
            .clone_dtoh(&value.slice(offset..offset + count))?
            .into_iter()
            .map(f32::from)
            .collect())
    }

    fn policy_snapshot(ar: &Runtime, output: &Path, frame: usize, book: usize) -> Result<()> {
        use safetensors::{Dtype, tensor::TensorView};
        let (logits, vocab) = if book == 0 {
            (&ar.work.logits, 200000)
        } else {
            (&ar.depth.logits, 1024)
        };
        let tensors = [
            (
                "logits",
                vec![2, vocab],
                read_device(ar, logits, 0, 2 * vocab)?,
            ),
            (
                "probabilities",
                vec![vocab],
                ar.device
                    .stream
                    .clone_dtoh(&ar.policy.probabilities.slice(..vocab))?,
            ),
            (
                "thresholds",
                vec![2],
                ar.device.stream.clone_dtoh(&ar.policy.thresholds)?,
            ),
        ];
        let data = tensors
            .iter()
            .map(|(_, _, values)| {
                values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let keep = ar
            .device
            .stream
            .clone_dtoh(&ar.policy.keep.slice(..vocab))?;
        let mut views = tensors
            .iter()
            .zip(&data)
            .map(|((name, shape, _), bytes)| {
                Ok((*name, TensorView::new(Dtype::F32, shape.clone(), bytes)?))
            })
            .collect::<Result<Vec<_>>>()?;
        views.push(("keep", TensorView::new(Dtype::U8, vec![vocab], &keep)?));
        safetensors::serialize_to_file(
            views,
            None,
            &output.join(format!("frame-{frame:05}-head-{book}.safetensors")),
        )?;
        Ok(())
    }

    fn contract_reference(
        ar: &mut Runtime,
        ids: &[u32],
        bf16_root: &Path,
        oracle: &Path,
        frozen: &serde_json::Value,
    ) -> Result<usize> {
        let policy_output = std::path::PathBuf::from(
            std::env::var_os("FF_MUSIC3_POLICY_OUTPUT")
                .context("FF_MUSIC3_POLICY_OUTPUT missing")?,
        );
        let root = bf16_root
            .parent()
            .and_then(Path::parent)
            .context("oracle root")?;
        let policy: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join("frozen-r2-ar-policy/FROZEN.json"),
        )?)?;
        ensure!(
            policy["status"] == "FROZEN",
            "Runner FAIL policy bounds not frozen before candidate"
        );
        std::fs::create_dir(&policy_output)?;
        let range = super::super::profile::Range::new(ar.frames == 25, c"music3_ar_eager");
        let legacy = if ar.frames == 25 {
            [
                "lm_head_logits",
                "rvq_depth_decoder_heads",
                "condition_input",
            ]
            .into_iter()
            .map(|key| {
                Ok((
                    key,
                    read(
                        &root.join("captures/forced-f32A"),
                        "capture.safetensors",
                        key,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?
        } else {
            BTreeMap::new()
        };
        let mut legacy_hidden_offset = 0;
        if ar.frames == 25 {
            policy_controls(ar, &policy_output)?;
        } else {
            ar.prepare_graphs()?;
        }
        let codes: Vec<u32> =
            serde_json::from_slice(&std::fs::read(bf16_root.join("CODES.json"))?)?;
        ar.position = 0;
        let mut stats = BTreeMap::new();
        ensure!(
            codes.len() == (ar.frames + 1) * 8,
            "Runner FAIL recorded frame census"
        );
        for (frame, codes) in codes.as_chunks::<8>().0.iter().take(ar.frames).enumerate() {
            ar.device.stream.memcpy_htod(codes, &mut ar.policy.codes)?;
            ar.embed_feedback()?;
            compare(
                &mut stats,
                "ar_feedback",
                &read_device(ar, &ar.work.hidden, 0, 8192)?,
                &read(
                    oracle,
                    &format!("frame-{frame:05}.safetensors"),
                    "ar_feedback",
                )?,
            )?;
        }
        let features = ar.run(ids, Some(&codes), 0, ar.frames != 25, |ar, frame, book| {
            let file = format!("frame-{frame:05}.safetensors");
            let key = if book == 0 {
                "ar_lm_head_logits".to_owned()
            } else {
                format!("ar_depth_head_{}", book - 1)
            };
            let value = if book == 0 {
                read_device(ar, &ar.work.logits, 0, 400000)?
            } else {
                read_device(ar, &ar.depth.logits, 0, 2048)?
            };
            compare(&mut stats, &key, &value, &read(oracle, &file, &key)?)?;
            if ar.frames == 25 {
                if book == 0 {
                    compare(
                        &mut stats,
                        "legacy_lm_head_logits",
                        &value,
                        &legacy["lm_head_logits"][frame * 400000..(frame + 1) * 400000],
                    )?;
                } else if frame * 7 + book <= 27 {
                    let mut hidden = Vec::new();
                    for branch in 0..2 {
                        hidden.extend(read_device(
                            ar,
                            &ar.depth.outputs,
                            branch * 8 * 4096,
                            (book + 1) * 4096,
                        )?);
                    }
                    compare(
                        &mut stats,
                        "legacy_rvq_depth_decoder_heads",
                        &hidden,
                        &legacy["rvq_depth_decoder_heads"]
                            [legacy_hidden_offset..legacy_hidden_offset + hidden.len()],
                    )?;
                    legacy_hidden_offset += hidden.len();
                }
                policy_snapshot(ar, &policy_output, frame, book)?;
            }
            Ok(())
        })?;
        drop(range);
        ensure!(
            features.len() == ar.frames * 32768 && ar.position == ids.len() / 2 + ar.frames,
            "DEFECT AR warmup/final feedback census"
        );
        compare(
            &mut stats,
            "condition_input",
            &features,
            &read(
                oracle,
                &format!("frame-{:05}.safetensors", ar.frames),
                "condition_input",
            )?,
        )?;
        if ar.frames == 25 {
            ensure!(
                legacy_hidden_offset == legacy["rvq_depth_decoder_heads"].len(),
                "Runner FAIL legacy depth call/layout mapping"
            );
            compare(
                &mut stats,
                "legacy_condition_input",
                &features,
                &legacy["condition_input"],
            )?;
        }
        let mut defects = 0;
        for (key, cell) in stats {
            let bound = if let Some(key) = key.strip_prefix("legacy_") {
                &frozen["legacy_gates_unchanged"][key]["bound"]
            } else {
                &frozen["cells"][&key]["bound"]
            };
            let max = bound["max"].as_f64().context("contract max bound")?;
            let rmse = bound["rmse"].as_f64().context("contract RMSE bound")?;
            let distance = (cell.squared / cell.count as f64).sqrt();
            let pass = cell.max <= max && distance <= rmse;
            defects += usize::from(!pass);
            println!(
                "{} music3_contract_cell key={key} max={} rmse={distance} bound_max={max} bound_rmse={rmse}",
                if pass { "PASS" } else { "DEFECT" },
                cell.max
            );
        }
        if ar.frames == 25 {
            let eager_kv = cache_bits(ar)?;
            ar.prepare_graphs()?;
            ar.position = 0;
            let mut changed = 0;
            let graph_features = ar.run(ids, Some(&codes), 0, true, |ar, frame, book| {
                let file = format!("frame-{frame:05}-head-{book}.safetensors");
                let logits = if book == 0 {
                    read_device(ar, &ar.work.logits, 0, 400000)?
                } else {
                    read_device(ar, &ar.depth.logits, 0, 2048)?
                };
                let expected = read(&policy_output, &file, "logits")?;
                changed += logits
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let vocab = logits.len() / 2;
                let probabilities = ar
                    .device
                    .stream
                    .clone_dtoh(&ar.policy.probabilities.slice(..vocab))?;
                ensure!(
                    logits.iter().chain(&probabilities).all(|x| x.is_finite()),
                    "Runner FAIL graph nonfinite frame={frame} book={book}"
                );
                let expected = read(&policy_output, &file, "probabilities")?;
                changed += probabilities
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let thresholds = ar.device.stream.clone_dtoh(&ar.policy.thresholds)?;
                let expected = read(&policy_output, &file, "thresholds")?;
                changed += thresholds
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let keep = ar
                    .device
                    .stream
                    .clone_dtoh(&ar.policy.keep.slice(..vocab))?;
                let bytes = std::fs::read(policy_output.join(&file))?;
                let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
                let expected = tensors.tensor("keep")?;
                ensure!(
                    expected.shape() == [vocab],
                    "Runner FAIL policy keep layout"
                );
                changed += keep
                    .iter()
                    .zip(expected.data())
                    .filter(|(a, b)| a != b)
                    .count();
                Ok(())
            })?;
            ensure!(
                features.len() == graph_features.len(),
                "Runner FAIL graph/eager frame extent"
            );
            changed += features
                .iter()
                .zip(&graph_features)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            changed += eager_kv
                .iter()
                .zip(cache_bits(ar)?)
                .filter(|(a, b)| **a != *b)
                .count();
            changed +=
                usize::from(ar.device.stream.clone_dtoh(&ar.cursor)?[0] as usize != ar.capacity);
            defects += usize::from(changed != 0);
            println!(
                "{} music3_ar_graph_identity changed_values={changed} quantities=features_logits_probabilities_thresholds_masks_KV_cursor streamed={}",
                if changed == 0 { "PASS" } else { "DEFECT" },
                ar.weights.dense.is_some()
            );
            ar.position = 0;
            let stopped = ar.generate(ids, Some(&[151670]), 0);
            let end = matches!(stopped, Err(ref error) if error.to_string() == "AR generated zero audio frames")
                && ar.position == ids.len() / 2
                && ar.depth.position == 8;
            defects += usize::from(!end);
            println!(
                "{} music3_ar_contract draws=208 semantic_calls=26 residual_calls=182 emitted_frames=25 feedback_calls=25 END_before_offset={end}",
                if end { "PASS" } else { "DEFECT" }
            );
        } else {
            println!(
                "PASS music3_ar_production frames={} draws={} graph=true feedback_calls={}",
                ar.frames,
                codes.len(),
                ar.frames
            );
        }
        Ok(defects)
    }

    fn cache_bits(ar: &Runtime) -> Result<Vec<u16>> {
        let mut bits = Vec::new();
        for layer in ar.layers.iter().chain(&ar.depth.layers) {
            for (k, v) in &layer.cache {
                for buffer in [k, v] {
                    bits.extend(
                        ar.device
                            .stream
                            .clone_dtoh(buffer)?
                            .into_iter()
                            .map(bf16::to_bits),
                    );
                }
            }
        }
        Ok(bits)
    }

    fn policy_controls(ar: &mut Runtime, output: &Path) -> Result<()> {
        let mut tie = vec![bf16::ZERO; 2048];
        tie[..61].fill(bf16::from_f32(2.0));
        tie[1024..1085].fill(bf16::from_f32(2.0));
        ar.device.stream.memcpy_htod(&tie, &mut ar.depth.logits)?;
        ar.distribution(false)?;
        let keep = ar.device.stream.clone_dtoh(&ar.policy.keep.slice(..1024))?;
        ensure!(
            keep.iter().filter(|&&x| x != 0).count() == 61,
            "DEFECT top-k cutoff ties"
        );
        policy_snapshot(ar, output, 26, 1)?;
        let mut semantic = vec![bf16::ZERO; 400000];
        semantic[151675..151736].fill(bf16::from_f32(2.0));
        semantic[351675..351736].fill(bf16::from_f32(2.0));
        semantic[151670] = bf16::from_f32(10.0);
        semantic[351776] = bf16::from_f32(-100.0);
        ar.device
            .stream
            .memcpy_htod(&semantic, &mut ar.work.logits)?;
        ar.distribution(true)?;
        let keep = ar.device.stream.clone_dtoh(&ar.policy.keep)?;
        ensure!(
            keep[151670] == 1
                && keep[151776] == 0
                && keep.iter().filter(|&&x| x != 0).count() == 62,
            "DEFECT conditional threshold/EOS mask"
        );
        policy_snapshot(ar, output, 26, 0)?;
        ar.device
            .stream
            .memset_zeros(&mut ar.policy.probabilities)?;
        ar.device
            .stream
            .memcpy_htod(&[1.0f32], &mut ar.policy.probabilities.slice_mut(17..18))?;
        ar.draw(1, 0, 38471471)?;
        let id = ar.device.stream.clone_dtoh(&ar.policy.codes.slice(1..2))?[0];
        let uniform = 4294967276.0f64 / 4294967296.0;
        ensure!(
            uniform as f32 == 1.0 && id == 17,
            "DEFECT categorical upper-tail control token={id}"
        );
        println!(
            "PASS music3_policy_controls ties=61 semantic_ties=62 conditional_wrong_guided_margin=48 categorical_token={id} upper_cdf_margin={}",
            1.0 - uniform
        );
        Ok(())
    }

    fn depth_snapshot(ar: &Runtime, codebook: usize) -> Result<BTreeMap<String, Vec<f32>>> {
        let mut values = BTreeMap::new();
        values.insert(
            format!("ar_depth_head_{}", codebook - 1),
            read_device(ar, &ar.depth.logits, 0, 2048)?,
        );
        let mut hidden = Vec::new();
        for branch in 0..2 {
            hidden.extend(read_device(
                ar,
                &ar.depth.outputs,
                branch * 8 * 4096,
                (codebook + 1) * 4096,
            )?);
        }
        values.insert(format!("ar_depth_hidden_{codebook}"), hidden);
        for (index, layer) in ar.depth.layers.iter().enumerate() {
            for keys in [true, false] {
                let mut flat = Vec::new();
                for (k, v) in &layer.cache {
                    let buffer = if keys { k } else { v };
                    for head in 0..16 {
                        flat.extend(read_device(ar, buffer, (head * 16 + codebook) * 256, 256)?);
                    }
                }
                values.insert(
                    format!(
                        "ar_depth_kv_{index}_{codebook}_{}",
                        if keys { "keys" } else { "values" }
                    ),
                    flat,
                );
            }
        }
        Ok(values)
    }

    fn depth_reference(
        ar: &mut Runtime,
        bf16_root: &Path,
        oracle: &Path,
        frozen: &serde_json::Value,
    ) -> Result<usize> {
        let schedule: Vec<u32> =
            serde_json::from_slice(&std::fs::read(bf16_root.join("CODES.json"))?)?;
        ensure!(schedule.len() == 208, "Runner FAIL depth schedule extent");
        let mut defects = 0;
        ar.depth_begin(schedule[0] - 151675)?;
        ar.depth_append(1, schedule[1])?;
        let correct = read_device(ar, &ar.depth.sequence, 2 * 4096, 4096)?;
        ar.depth_append(2, schedule[1])?;
        let wrong = read_device(ar, &ar.depth.sequence, 3 * 4096, 4096)?;
        let offset_error = correct
            .iter()
            .zip(&wrong)
            .map(|(&a, &b)| f64::from((a - b).abs()))
            .fold(0.0f64, f64::max);
        let offset_bound = frozen["cells"]["ar_projection_2"]["bound"]["max"]
            .as_f64()
            .context("Runner FAIL projection bound")?;
        let detected = offset_error > offset_bound;
        defects += usize::from(!detected);
        println!(
            "{} music3_depth_offset_control max={offset_error} bound={offset_bound}",
            if detected { "PASS" } else { "DEFECT" }
        );
        for cached in [false, true] {
            let mut stats = BTreeMap::new();
            for frame in 0..=25 {
                let file = format!("frame-{frame:05}.safetensors");
                let hidden = read(bf16_root, &file, "ar_global_last_hidden")?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect::<Vec<_>>();
                ar.device.stream.memcpy_htod(&hidden, &mut ar.work.last)?;
                ar.depth_begin(schedule[frame * 8] - 151675)?;
                for codebook in 1..8 {
                    ar.depth_step(codebook, cached)?;
                    for (key, value) in depth_snapshot(ar, codebook)? {
                        compare(&mut stats, &key, &value, &read(oracle, &file, &key)?)?;
                    }
                    if codebook < 7 {
                        ar.depth_append(codebook, schedule[frame * 8 + codebook])?;
                    }
                }
                for position in 0..8 {
                    let key = format!("ar_projection_{position}");
                    let mut values = Vec::new();
                    for branch in 0..2 {
                        values.extend(read_device(
                            ar,
                            &ar.depth.sequence,
                            (branch * 8 + position) * 4096,
                            4096,
                        )?);
                    }
                    compare(&mut stats, &key, &values, &read(oracle, &file, &key)?)?;
                }
            }
            for (key, cell) in stats {
                let max = frozen["cells"][&key]["bound"]["max"]
                    .as_f64()
                    .context("Runner FAIL depth max bound")?;
                let rmse = frozen["cells"][&key]["bound"]["rmse"]
                    .as_f64()
                    .context("Runner FAIL depth RMSE bound")?;
                let distance = (cell.squared / cell.count as f64).sqrt();
                let pass = cell.max <= max && distance <= rmse;
                defects += usize::from(!pass);
                println!(
                    "{} music3_depth_cell cached={cached} key={key} max={} rmse={distance} bound_max={max} bound_rmse={rmse}",
                    if pass { "PASS" } else { "DEFECT" },
                    cell.max
                );
            }
        }
        Ok(defects)
    }
}
