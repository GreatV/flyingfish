use super::*;
use crate::backend::cuda::{
    contracts::{RowDomain, SeqRef, Tensor, TensorMut, view},
    cubin,
    decode::GqaDecodeInput,
    linear::LinearWeightRef,
    ops::{AddRmsNorm, CommonOpsRef},
};
use crate::backend::cuda::{
    linear::LinearBiasMode,
    ops::{RmsMode, RmsOptions},
    weights::LoopvlVisionWeights,
};
use cudarc::driver::{CudaFunction, DevicePtr};

pub(super) struct Runtime {
    initial: CudaSlice<bf16>,
    high: CudaSlice<bf16>,
    low: CudaSlice<bf16>,
    rope: CudaSlice<bf16>,
    prefix: usize,
    meta: CudaSlice<i32>,
    embed: CudaFunction,
    rotary: CudaFunction,
    common: CommonOpsRef,
    program: Vec<(bool, usize)>,
    vision: VisionRuntime,
    vision_attention: Flash,
    vision_weights: LoopvlVisionWeights,
    media: Option<Media>,
    #[cfg(test)]
    control: u8,
}

struct Media {
    work: VisionWorkspace,
    anchor: CudaSlice<bf16>,
    slot: crate::vision_input::ImageSlot,
}

impl Runtime {
    pub(super) fn reset(&mut self, s: &Arc<CudaStream>, capacity: usize) -> Result<()> {
        self.media = None;
        self.prefix = 0;
        s.memcpy_htod(
            &yarn_table(&(0..capacity).map(|p| [p, p]).collect::<Vec<_>>()),
            &mut self.rope,
        )?;
        Ok(())
    }
}

fn matrix(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(
        b,
        s,
        0,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}

fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>, width: usize) -> Result<Tensor<bf16, 1>> {
    view(b, s, 0, [u32::try_from(width)?], [1])
}

fn linear(
    blas: &Blas,
    s: &Arc<CudaStream>,
    w: &CudaSlice<bf16>,
    x: Tensor<bf16, 2>,
    y: Tensor<bf16, 2>,
) -> Result<()> {
    let phase = if x.shape[0] == 1 {
        crate::op_contract::ExecutionPhase::Decode
    } else {
        crate::op_contract::ExecutionPhase::Prefill
    };
    blas.linear_typed(
        phase,
        LinearWeightRef::Bf16(matrix(w, s, y.shape[1] as usize, x.shape[1] as usize)?),
        x,
        &mut TensorMut(y),
    )
}

fn yarn_table(positions: &[[usize; 2]]) -> Vec<bf16> {
    let correction = |rotations: f32| {
        128.0 * (4096.0 / (rotations * 2.0 * std::f32::consts::PI)).ln() / (2.0 * 10000.0f32.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min(127.0);
    let inv: Vec<_> = (0..64)
        .map(|d| {
            let extrapolation = 1.0 / 10000.0f32.powf((2 * d) as f32 / 128.0);
            let ramp = ((d as f32 - low) / (high - low)).clamp(0.0, 1.0);
            bf16::from_f32(extrapolation / 2.0 * ramp + extrapolation * (1.0 - ramp)).to_f32()
        })
        .collect();
    let scale = (0.1 * 2.0f64.ln() + 1.0) as f32;
    positions
        .iter()
        .flat_map(|axes| {
            let inv = &inv;
            (0..256).map(move |d| {
                let p = axes[d / 128];
                let angle = p as f32 * inv[d % 64];
                bf16::from_f32(if d % 128 < 64 {
                    angle.cos() * scale
                } else {
                    angle.sin() * scale
                })
            })
        })
        .collect()
}

struct VisionWorkspace {
    x: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    gu: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    positions: CudaSlice<i32>,
    start: CudaSlice<i32>,
    meta: CudaSlice<i32>,
    f32_a: CudaSlice<f32>,
    f32_b: CudaSlice<f32>,
    f32_c: CudaSlice<f32>,
    projected_f32: CudaSlice<f32>,
    projected_bf16: CudaSlice<bf16>,
    instruction: CudaSlice<bf16>,
    gates: CudaSlice<bf16>,
    gate_query: CudaSlice<f32>,
}

impl VisionWorkspace {
    fn new(s: &Arc<CudaStream>, rows: usize) -> Result<Self> {
        Ok(Self {
            x: s.alloc_zeros(rows * 1024)?,
            n: s.alloc_zeros(rows * 1024)?,
            qkv: s.alloc_zeros(rows * 4096)?,
            attn: s.alloc_zeros(rows * 2048)?,
            out: s.alloc_zeros(rows * 1024)?,
            gu: s.alloc_zeros(rows * 6144)?,
            act: s.alloc_zeros(rows * 3072)?,
            k: s.alloc_zeros(rows * 1024)?,
            v: s.alloc_zeros(rows * 1024)?,
            positions: s.alloc_zeros(rows * 2)?,
            start: s.alloc_zeros(1)?,
            meta: s.clone_htod(&[0i32, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?,
            f32_a: s.alloc_zeros(rows * 1536)?,
            f32_b: s.alloc_zeros(rows * 1536)?,
            f32_c: s.alloc_zeros(rows * 1536)?,
            projected_f32: s.alloc_zeros(rows * 1536)?,
            projected_bf16: s.alloc_zeros(rows * 1536)?,
            instruction: s.alloc_zeros(1536)?,
            gates: s.alloc_zeros(rows)?,
            gate_query: s.alloc_zeros(256)?,
        })
    }
}

struct VisionEncoderCall<'a> {
    weights: &'a LoopvlVisionWeights,
    pixels: &'a CudaSlice<bf16>,
    rows: usize,
    grid_h: usize,
    grid_w: usize,
}

struct ReinjectCall<'a> {
    weights: &'a LoopvlVisionWeights,
    states: &'a mut CudaSlice<bf16>,
    anchor: &'a CudaSlice<bf16>,
    visual_begin: usize,
    visual_rows: usize,
    instruction_begin: usize,
    instruction_end: usize,
    cycle: usize,
}

struct VisionRuntime {
    #[cfg(test)]
    reinjection_calls: std::cell::Cell<usize>,
    #[cfg(test)]
    encoder_calls: std::cell::Cell<usize>,
    #[cfg(test)]
    projector_calls: std::cell::Cell<usize>,
    rope_2d: CudaFunction,
    kv_write: CudaFunction,
    upcast: CudaFunction,
    bias: CudaFunction,
    bias_gelu: CudaFunction,
    rms_scale: CudaFunction,
    add_row_silu: CudaFunction,
    gate_round: CudaFunction,
    masked_mean: CudaFunction,
    anchor: CudaFunction,
    common: CommonOpsRef,
    ops: Ops,
}

fn f32_weight(
    bytes: &CudaSlice<u8>,
    s: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<f32, 2>> {
    ensure!(
        bytes.len() == rows * width * 4,
        "FP32 weight byte extent mismatch"
    );
    unsafe {
        Tensor::new(
            bytes.device_ptr(s).0,
            [rows as u32, width as u32],
            [width as u64, 1],
            bytes.len() as u64,
        )
    }
}

fn matrix32(
    b: &CudaSlice<f32>,
    s: &Arc<CudaStream>,
    rows: usize,
    width: usize,
) -> Result<Tensor<f32, 2>> {
    view(
        b,
        s,
        0,
        [u32::try_from(rows)?, u32::try_from(width)?],
        [width as u64, 1],
    )
}

impl VisionRuntime {
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let rope = cubin::module(ctx, "rope")?;
        let vision = cubin::module(ctx, "vision_ops")?;
        Ok(Self {
            #[cfg(test)]
            reinjection_calls: std::cell::Cell::new(0),
            #[cfg(test)]
            encoder_calls: std::cell::Cell::new(0),
            #[cfg(test)]
            projector_calls: std::cell::Cell::new(0),
            rope_2d: rope.load_function("rope_2d_qk")?,
            kv_write: rope.load_function("kv_write")?,
            upcast: vision.load_function("vision_upcast_f32")?,
            bias: vision.load_function("vision_bias_f32")?,
            bias_gelu: vision.load_function("vision_bias_gelu_f32")?,
            rms_scale: vision.load_function("vision_rms_scale_f32")?,
            add_row_silu: vision.load_function("vision_add_row_silu_f32")?,
            gate_round: vision.load_function("vision_gate_round_bf16")?,
            masked_mean: vision.load_function("vision_masked_mean_bf16")?,
            anchor: vision.load_function("vision_anchor_reinject_bf16")?,
            common: CommonOpsRef::new(ctx)?,
            ops: Ops::new_with_swiglu_rounding(ctx, crate::model_spec::SwigluRounding::Twice)?,
        })
    }

    fn run_encoder(
        &self,
        blas: &mut Blas,
        attention: &mut Flash,
        s: &Arc<CudaStream>,
        work: &mut VisionWorkspace,
        call: VisionEncoderCall<'_>,
    ) -> Result<()> {
        #[cfg(test)]
        self.encoder_calls.set(self.encoder_calls.get() + 1);
        let VisionEncoderCall {
            weights,
            pixels,
            rows,
            grid_h,
            grid_w,
        } = call;
        ensure!(
            rows > 0 && rows == grid_h * grid_w && pixels.len() == rows * 588,
            "PenguinVL pixel/grid shape mismatch"
        );
        for (output, input) in [
            (1024usize, 588usize),
            (4096, 1024),
            (1024, 2048),
            (6144, 1024),
            (1024, 3072),
        ] {
            blas.prepare_request_fixed(rows as u32, output as u32, input as u32)?;
        }
        blas.prepare_biased(
            rows as u32,
            1024,
            588,
            vector(&weights.patch_bias, s, 1024)?,
            LinearBiasMode::BeforeOutputRound,
        )?;
        blas.linear_biased(
            matrix(&weights.patch_weight, s, 1024, 588)?,
            vector(&weights.patch_bias, s, 1024)?,
            matrix(pixels, s, rows, 588)?,
            &mut TensorMut(matrix(&work.x, s, rows, 1024)?),
            LinearBiasMode::BeforeOutputRound,
        )?;
        let positions: Vec<i32> = (0..grid_h)
            .flat_map(|r| (0..grid_w).map(move |c| (r, c)))
            .flat_map(|(r, c)| [r as i32, c as i32])
            .collect();
        let axis0: Vec<i32> = positions.iter().step_by(2).copied().collect();
        let axis1: Vec<i32> = positions.iter().skip(1).step_by(2).copied().collect();
        s.memcpy_htod(&axis0, &mut work.positions.slice_mut(..rows))?;
        s.memcpy_htod(&axis1, &mut work.positions.slice_mut(rows..rows * 2))?;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: work.meta.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: rows as u32,
        };
        let rms = |weight: &CudaSlice<bf16>, input: &CudaSlice<bf16>, output: &CudaSlice<bf16>| {
            self.common.norm.rms_with(
                s,
                domain,
                view(input, s, 0, [rows as u32, 1, 1024], [1024, 1024, 1])?,
                vector(weight, s, 1024)?,
                &mut TensorMut(view(output, s, 0, [rows as u32, 1, 1024], [1024, 1024, 1])?),
                RmsOptions {
                    mode: RmsMode::Standard,
                    eps: 1e-6,
                },
            )
        };
        rms(&weights.layers[0].input_norm, &work.x, &work.n)?;
        for (index, layer) in weights.layers.iter().enumerate() {
            linear(
                blas,
                s,
                &layer.qkv,
                matrix(&work.n, s, rows, 1024)?,
                matrix(&work.qkv, s, rows, 4096)?,
            )?;
            for (offset, heads, weight) in
                [(0usize, 16usize, &layer.q_norm), (2048, 8, &layer.k_norm)]
            {
                let qk = view(
                    &work.qkv,
                    s,
                    offset,
                    [rows as u32, heads as u32, 128],
                    [4096, 128, 1],
                )?;
                self.common.norm.rms_with(
                    s,
                    domain,
                    qk,
                    vector(weight, s, 128)?,
                    &mut TensorMut(qk),
                    RmsOptions {
                        mode: RmsMode::Standard,
                        eps: 1e-6,
                    },
                )?;
            }
            unsafe {
                s.launch_builder(&self.rope_2d)
                    .arg(&mut work.qkv)
                    .arg(&work.positions)
                    .arg(&(rows as i32))
                    .arg(&16i32)
                    .arg(&8i32)
                    .arg(&128i32)
                    .arg(&4096i32)
                    .arg(&1_000_000.0f32)
                    .launch(flat(rows * 24 * 64))?;
                s.launch_builder(&self.kv_write)
                    .arg(&work.qkv)
                    .arg(&mut work.k)
                    .arg(&mut work.v)
                    .arg(&work.start)
                    .arg(&(rows as i32))
                    .arg(&16i32)
                    .arg(&8i32)
                    .arg(&128i32)
                    .arg(&(rows as i32))
                    .launch(flat(rows * 1024))?;
            }
            attention.run(
                &work.qkv,
                4096,
                &work.k,
                &work.v,
                &mut work.attn,
                super::super::flash::Shape {
                    rows,
                    start: 0,
                    q_heads: 16,
                    kv_heads: 8,
                    capacity: rows,
                    head_dim: 128,
                    causal: false,
                    window_left: None,
                },
            )?;
            linear(
                blas,
                s,
                &layer.o,
                matrix(&work.attn, s, rows, 2048)?,
                matrix(&work.out, s, rows, 1024)?,
            )?;
            self.ops.add_rmsnorm(
                s,
                AddRmsNorm {
                    input: matrix(&work.out, s, rows, 1024)?,
                    residual: matrix(&work.x, s, rows, 1024)?,
                    weight: Some(vector(&layer.post_norm, s, 1024)?),
                    output: TensorMut(matrix(&work.n, s, rows, 1024)?),
                    sum: TensorMut(matrix(&work.x, s, rows, 1024)?),
                    eps: 1e-6,
                },
            )?;
            linear(
                blas,
                s,
                &layer.gu,
                matrix(&work.n, s, rows, 1024)?,
                matrix(&work.gu, s, rows, 6144)?,
            )?;
            unsafe {
                s.launch_builder(&self.ops.act)
                    .arg(&work.gu)
                    .arg(&mut work.act)
                    .arg(&(rows as i32))
                    .arg(&3072i32)
                    .arg(&0i32)
                    .launch(flat(rows * 3072))?;
            }
            linear(
                blas,
                s,
                &layer.down,
                matrix(&work.act, s, rows, 3072)?,
                matrix(&work.out, s, rows, 1024)?,
            )?;
            let next = weights
                .layers
                .get(index + 1)
                .map(|next| &next.input_norm)
                .unwrap_or(&weights.norm);
            self.ops.add_rmsnorm(
                s,
                AddRmsNorm {
                    input: matrix(&work.out, s, rows, 1024)?,
                    residual: matrix(&work.x, s, rows, 1024)?,
                    weight: Some(vector(next, s, 1024)?),
                    output: TensorMut(matrix(&work.n, s, rows, 1024)?),
                    sum: TensorMut(matrix(&work.x, s, rows, 1024)?),
                    eps: 1e-6,
                },
            )?;
        }
        Ok(())
    }

    fn run_projector(
        &self,
        blas: &Blas,
        weights: &LoopvlVisionWeights,
        s: &Arc<CudaStream>,
        work: &mut VisionWorkspace,
        rows: usize,
    ) -> Result<()> {
        #[cfg(test)]
        self.projector_calls.set(self.projector_calls.get() + 1);
        ensure!(rows > 0, "vision projector requires rows");
        unsafe {
            s.launch_builder(&self.upcast)
                .arg(&work.n)
                .arg(&mut work.f32_a)
                .arg(&((rows * 1024) as i32))
                .launch(flat(rows * 1024))?;
        }
        blas.linear_f32(
            f32_weight(&weights.fc1_weight, s, 1536, 1024)?,
            matrix32(&work.f32_a, s, rows, 1024)?,
            &mut TensorMut(matrix32(&work.f32_b, s, rows, 1536)?),
        )?;
        unsafe {
            let f32_b = work.f32_b.device_ptr(s).0;
            s.launch_builder(&self.bias_gelu)
                .arg(&f32_b)
                .arg(&f32_weight(&weights.fc1_bias, s, 1, 1536)?.ptr)
                .arg(&mut work.f32_b)
                .arg(&(rows as i32))
                .arg(&1536i32)
                .launch(flat(rows * 1536))?;
        }
        blas.linear_f32(
            f32_weight(&weights.fc2_weight, s, 1536, 1536)?,
            matrix32(&work.f32_b, s, rows, 1536)?,
            &mut TensorMut(matrix32(&work.f32_c, s, rows, 1536)?),
        )?;
        unsafe {
            let f32_c = work.f32_c.device_ptr(s).0;
            s.launch_builder(&self.bias)
                .arg(&f32_c)
                .arg(&f32_weight(&weights.fc2_bias, s, 1, 1536)?.ptr)
                .arg(&mut work.f32_c)
                .arg(&(rows as i32))
                .arg(&1536i32)
                .launch(flat(rows * 1536))?;
            s.launch_builder(&self.rms_scale)
                .arg(&work.f32_c)
                .arg(&f32_weight(&weights.visual_scale, s, 1, 1)?.ptr)
                .arg(&mut work.projected_f32)
                .arg(&mut work.projected_bf16)
                .arg(&1536i32)
                .arg(&1e-6f32)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }

    fn run_reinject(
        &self,
        blas: &Blas,
        s: &Arc<CudaStream>,
        work: &mut VisionWorkspace,
        call: ReinjectCall<'_>,
    ) -> Result<()> {
        let ReinjectCall {
            weights,
            states,
            anchor,
            visual_begin,
            visual_rows,
            instruction_begin,
            instruction_end,
            cycle,
        } = call;
        ensure!(
            visual_rows > 0
                && visual_begin + visual_rows <= states.len() / 1536
                && anchor.len() == visual_rows * 1536
                && instruction_begin < instruction_end
                && instruction_end <= states.len() / 1536,
            "reinjection mask extent"
        );
        #[cfg(test)]
        self.reinjection_calls.set(self.reinjection_calls.get() + 1);
        unsafe {
            s.launch_builder(&self.masked_mean)
                .arg(&*states)
                .arg(&mut work.instruction)
                .arg(&(instruction_begin as i32))
                .arg(&(instruction_end as i32))
                .arg(&1536i32)
                .launch(flat(1536))?;
            s.launch_builder(&self.upcast)
                .arg(&states.slice(visual_begin * 1536..(visual_begin + visual_rows) * 1536))
                .arg(&mut work.f32_a)
                .arg(&((visual_rows * 1536) as i32))
                .launch(flat(visual_rows * 1536))?;
            s.launch_builder(&self.upcast)
                .arg(&work.instruction)
                .arg(&mut work.f32_c)
                .arg(&1536i32)
                .launch(flat(1536))?;
        }
        blas.linear_f32(
            f32_weight(&weights.gate_visual, s, 256, 1536)?,
            matrix32(&work.f32_a, s, visual_rows, 1536)?,
            &mut TensorMut(matrix32(&work.f32_b, s, visual_rows, 256)?),
        )?;
        blas.linear_f32(
            f32_weight(&weights.gate_query, s, 256, 1536)?,
            matrix32(&work.f32_c, s, 1, 1536)?,
            &mut TensorMut(matrix32(&work.gate_query, s, 1, 256)?),
        )?;
        unsafe {
            let gate_b = work.f32_b.device_ptr(s).0;
            s.launch_builder(&self.add_row_silu)
                .arg(&gate_b)
                .arg(&work.gate_query)
                .arg(&gate_b)
                .arg(&((visual_rows * 256) as i32))
                .arg(&256i32)
                .launch(flat(visual_rows * 256))?;
        }
        blas.linear_f32(
            f32_weight(&weights.gate_out, s, 1, 256)?,
            matrix32(&work.f32_b, s, visual_rows, 256)?,
            &mut TensorMut(matrix32(&work.f32_c, s, visual_rows, 1)?),
        )?;
        unsafe {
            let score = work.f32_c.device_ptr(s).0;
            let state_ptr = states.device_ptr(s).0 + (visual_begin * 1536 * 2) as u64;
            s.launch_builder(&self.bias)
                .arg(&score)
                .arg(&f32_weight(&weights.gate_bias, s, 1, 1)?.ptr)
                .arg(&score)
                .arg(&(visual_rows as i32))
                .arg(&1i32)
                .launch(flat(visual_rows))?;
            s.launch_builder(&self.gate_round)
                .arg(&score)
                .arg(&mut work.gates)
                .arg(&(visual_rows as i32))
                .launch(flat(visual_rows))?;
            s.launch_builder(&self.anchor)
                .arg(&state_ptr)
                .arg(anchor)
                .arg(&work.gates)
                .arg(&f32_weight(&weights.raw_alpha, s, 1, 2)?.ptr)
                .arg(&(visual_rows as i32))
                .arg(&1536i32)
                .arg(&(cycle as i32))
                .launch(flat(visual_rows * 1536))?;
        }
        Ok(())
    }
}

impl Engine {
    pub(super) fn load_recurrent(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        let started = Instant::now();
        ensure!(
            (2..=model_spec.max_position as usize).contains(&capacity),
            "LoopVL KV capacity outside 2..={}",
            model_spec.max_position
        );
        ensure!(
            settings.draft_model.is_none() && !settings.prefix_cache,
            "LoopVL L1a does not support draft or prefix-cache execution"
        );
        ensure!(
            settings.kv_format == crate::kv::KvFormat::Bf16,
            "LoopVL requires BF16 KV"
        );
        let rows = capacity - 1;
        ensure!(
            requested_chunk.is_none_or(|n| n == rows),
            "LoopVL bidirectional prefix requires a whole-prompt workspace of {rows} rows"
        );
        let device = Device::new(ordinal)?;
        let config = model_spec.config();
        let manifest = crate::checkpoint::MappedCheckpoint::open(dir)?
            .manifest(&model_spec, crate::checkpoint::LoadPolicy::TextWithVision)?;
        let (free, total) = device.ctx.mem_get_info()?;
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model_spec)?.row_bytes()?
            + 4 * config.hidden_size
            + config.num_attention_heads * 4;
        let fixed = manifest.resident_bytes
            + rows * row_bytes
            + capacity * 256 * 2
            + 2048 * 16 * 4
            + 8 * 1024 * 1024;
        let per_token = model_spec.layers.len() * 2 * config.kv_dim() * 2;
        let available = free
            .checked_sub(total / 16 + fixed)
            .context("LoopVL resident weights/workspace exceed device memory budget")?;
        let admitted = (available / per_token).min(model_spec.max_position as usize);
        ensure!(
            capacity <= admitted,
            "LoopVL KV_CAPACITY_LIMIT={admitted}; requested={capacity}"
        );
        let (weights, initial) = Weights::load_loopvl(dir, &model_spec, &device)?;
        let vision_weights = LoopvlVisionWeights::load(dir, &model_spec, &device)?;
        let s = &device.stream;
        let state = super::super::state::StateStore::new(&model_spec, capacity, s)?;
        let checkpoint = super::super::state::StateCheckpoint::new(&state, &model_spec, 1, s)?;
        let ops = Ops::new_with_swiglu_rounding(&device.ctx, model_spec.swiglu_rounding())?;
        let closure = Rc::new(super::super::closure::Closure::new(
            false,
            "LoopVL fixed setup".into(),
        ));
        let runtime = settings.runtime_dir.clone().unwrap_or_default();
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        blas.bind_invariant(&device.ctx, 1)?;
        for n in [1] {
            for (output, input) in [
                (4 * config.hidden_size, config.hidden_size),
                (config.hidden_size, config.hidden_size),
                (2 * config.intermediate_size, config.hidden_size),
                (config.hidden_size, config.intermediate_size),
            ] {
                blas.prepare_fixed(n as u32, output as u32, input as u32)?;
            }
        }
        blas.prepare_fixed(1, config.vocab_size as u32, config.hidden_size as u32)?;
        let blocks = device.info.sms.div_ceil(config.num_attention_heads).max(1);
        let chunk = capacity.div_ceil(blocks).div_ceil(32) * 32;
        let threads = device.info.max_threads.min(256) / 32 * 32;
        let plan = crate::backend::setup::AttentionPlan {
            chunk,
            qpack: 1,
            threads,
            merge_threads: 128,
        };
        let decode = Attention::new(&device.ctx, s, &config, capacity, plan)?;
        let recurrent = Runtime {
            initial,
            high: s.alloc_zeros(rows * config.hidden_size)?,
            low: s.alloc_zeros(rows * config.hidden_size)?,
            rope: s.clone_htod(&yarn_table(
                &(0..capacity).map(|p| [p, p]).collect::<Vec<_>>(),
            ))?,
            prefix: 0,
            // Elementwise metadata spans the workspace; attention uses the request prefix.
            meta: s.clone_htod(&[0i32, rows as i32, rows as i32, capacity as i32, 1, 0, 0, 0])?,
            embed: cubin::module(&device.ctx, "embed")?.load_function("embed_scale_broadcast")?,
            rotary: cubin::module(&device.ctx, "rope")?.load_function("rope_table_kv")?,
            common: CommonOpsRef::new(&device.ctx)?,
            program: (0..8).map(|call| (call % 4 == 3, call * 16)).collect(),
            vision: VisionRuntime::new(&device.ctx)?,
            vision_attention: Flash::new(s.clone(), 2048, 16, 8, 128)?,
            vision_weights,
            media: None,
            #[cfg(test)]
            control: 0,
        };
        let work = Work::new(s, &model_spec, rows)?;
        let flash = Flash::new(s.clone(), rows, 12, 12, 128)?;
        let mut engine = Self {
            #[cfg(test)]
            capture_fail_after_layer: None,
            #[cfg(test)]
            capture_fail_stage: None,
            #[cfg(test)]
            flint_window_capture: None,
            #[cfg(test)]
            moe_test_ffn: None,
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            qwen_device_prefix: false,
            forward_rows: (1..=rows).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: settings.spec_budget,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            runtime,
            spec: None,
            hidden_capture: None,
            qwen: None,
            recurrent: Some(recurrent),
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            flint_frozen_decode_length: false,
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            weights,
            state,
            capture_checkpoint: Some(checkpoint),
            work,
            ops,
            blas,
            flash: Some(flash),
            decode: Some(decode),
            flint_logits: None,
            device,
            config,
            model_spec,
            capacity,
            chunk: rows,
            workspace_rows: rows,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        engine.closure.freeze_fixed()?;
        engine.reset()?;
        engine.capture()?;
        engine.closure.expect_captures(1)?;
        engine.reset()?;
        engine.closure.ready(started)?;
        Ok(engine)
    }

    pub(super) fn prefill_recurrent_media(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        trace: Option<&mut Trace>,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned && self.position == 0,
            "LoopVL image prefill requires reset"
        );
        let plan = input.validate(&self.model_spec, self.capacity)?;
        ensure!(
            input.input_ids.len() <= self.chunk,
            "LoopVL image prompt exceeds prefix workspace"
        );
        let slot = plan.slots()[0].clone();
        let rows = slot.tokens.len();
        let (free, total) = self.device.ctx.mem_get_info()?;
        // Workspace, pixels and anchor: 73,066 B/patch plus 4,132 B.
        ensure!(
            free.checked_sub(total / 16)
                .is_some_and(|n| rows * 73_066 + 4_132 <= n),
            "LoopVL vision request exceeds device memory budget"
        );
        let s = &self.device.stream;
        let mut work = VisionWorkspace::new(s, rows)?;
        let pixels = input
            .pixel_values
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        ensure!(
            pixels.iter().all(|p| p.is_finite()),
            "LoopVL pixels overflow BF16"
        );
        let pixels = s.clone_htod(&pixels)?;
        let runtime = self
            .recurrent
            .as_mut()
            .context("recurrent runtime missing")?;
        let result = (|| -> Result<()> {
            runtime.vision.run_encoder(
                &mut self.blas,
                &mut runtime.vision_attention,
                s,
                &mut work,
                VisionEncoderCall {
                    weights: &runtime.vision_weights,
                    pixels: &pixels,
                    rows,
                    grid_h: slot.grid.h as usize,
                    grid_w: slot.grid.w as usize,
                },
            )?;
            runtime.vision.run_projector(
                &self.blas,
                &runtime.vision_weights,
                s,
                &mut work,
                rows,
            )?;
            let axes = plan.packed_axes()?;
            let positions = axes
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| [p[0] as usize, p[1] as usize])
                .collect::<Vec<_>>();
            #[cfg(test)]
            let positions = if runtime.control == 4 {
                (0..self.capacity).map(|p| [p, p]).collect()
            } else if runtime.control == 7 {
                positions.into_iter().map(|p| [p[1], p[0]]).collect()
            } else {
                positions
            };
            s.memcpy_htod(&yarn_table(&positions), &mut runtime.rope)?;
            runtime.media = Some(Media {
                work,
                anchor: s.alloc_zeros(rows * 1536)?,
                slot,
            });
            Ok(())
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        self.prefill_recurrent(
            input.input_ids,
            input.input_ids.len(),
            trace,
            input.input_ids.len(),
        )
    }

    pub(super) fn prefill_recurrent(
        &mut self,
        ids: &[u32],
        prefix: usize,
        trace: Option<&mut Trace>,
        keep: usize,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned && self.position == 0,
            "LoopVL prefix requires reset before prefill"
        );
        ensure!(
            !ids.is_empty() && ids.len() <= self.chunk,
            "LoopVL prefix workspace={} exceeded: rows={}",
            self.chunk,
            ids.len()
        );
        ensure!(
            (1..=ids.len()).contains(&prefix),
            "LoopVL prefix boundary outside input"
        );
        ensure!(
            ids.iter().all(|&v| v < self.model_spec.vocab),
            "LoopVL input id outside vocabulary"
        );
        for (output, input) in [
            (4 * self.config.hidden_size, self.config.hidden_size),
            (self.config.hidden_size, self.config.hidden_size),
            (2 * self.config.intermediate_size, self.config.hidden_size),
            (self.config.hidden_size, self.config.intermediate_size),
        ] {
            self.blas
                .prepare_request_fixed(ids.len() as u32, output as u32, input as u32)?;
        }
        let s = &self.device.stream;
        s.memcpy_htod(ids, &mut self.work.ids.slice_mut(..ids.len()))?;
        self.recurrent
            .as_mut()
            .context("recurrent runtime missing")?
            .prefix = prefix;
        if let Err(e) = self.run_recurrent(
            ids.len(),
            false,
            trace.map(|trace| Target {
                trace,
                prefix: "prefill",
                skip: ids.len() - keep,
            }),
        ) {
            self.poisoned = true;
            return Err(e);
        }
        self.position = ids.len();
        let s = &self.device.stream;
        s.memcpy_htod(&[self.position as i32], &mut self.work.position)?;
        s.memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
        self.committed_prefix_ids = ids.to_vec();
        let token = self.token()?;
        self.pending_token = Some(token);
        Ok(token)
    }

    pub(super) fn run_recurrent(
        &mut self,
        rows: usize,
        token: bool,
        mut trace: Option<Target<'_>>,
    ) -> Result<()> {
        let r = self
            .recurrent
            .as_mut()
            .context("recurrent runtime missing")?;
        let s = &self.device.stream;
        let w = &mut self.work;
        let h = self.config.hidden_size;
        let f = self.config.intermediate_size;
        let qh = self.config.num_attention_heads;
        let kh = self.config.num_key_value_heads;
        let dim = self.config.head_dim;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: r.meta.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: rows as u32,
        };
        unsafe {
            let (visual, visual_begin, visual_rows) = if !token {
                r.media.as_ref().map_or((&self.weights.embed, 0, 0), |m| {
                    (
                        &m.work.projected_bf16,
                        m.slot.tokens.start as i32,
                        m.slot.tokens.len() as i32,
                    )
                })
            } else {
                (&self.weights.embed, 0, 0)
            };
            s.launch_builder(&r.embed)
                .arg(&self.weights.embed)
                .arg(if token { &w.token } else { &w.ids })
                .arg(&r.initial)
                .arg(&mut r.high)
                .arg(&mut r.low)
                .arg(&(rows as i32))
                .arg(&(h as i32))
                .arg(&(39.191835884530846f64 as f32))
                .arg(visual)
                .arg(&visual_begin)
                .arg(&visual_rows)
                .launch(flat(rows * h))?;
        }
        if !token && let Some(media) = &mut r.media {
            s.memcpy_dtod(
                &r.high
                    .slice(media.slot.tokens.start * h..media.slot.tokens.end * h),
                &mut media.anchor,
            )?;
        }
        for &(high, offset) in &r.program {
            #[cfg(test)]
            let skip_reinjection = r.control == 5 && offset == 0;
            #[cfg(not(test))]
            let skip_reinjection = false;
            if !token
                && !skip_reinjection
                && offset % 64 == 0
                && let Some(media) = &mut r.media
            {
                r.vision.run_reinject(
                    &self.blas,
                    s,
                    &mut media.work,
                    ReinjectCall {
                        weights: &r.vision_weights,
                        states: &mut r.high,
                        anchor: &media.anchor,
                        visual_begin: media.slot.tokens.start,
                        visual_rows: media.slot.tokens.len(),
                        instruction_begin: media.slot.tokens.end,
                        instruction_end: r.prefix,
                        cycle: offset / 64,
                    },
                )?;
            }
            let (target, other) = if high {
                (&mut r.high, &r.low)
            } else {
                (&mut r.low, &r.high)
            };
            self.ops.add_rmsnorm(
                s,
                AddRmsNorm {
                    input: matrix(other, s, rows, h)?,
                    residual: matrix(target, s, rows, h)?,
                    weight: None,
                    output: TensorMut(matrix(&w.n, s, rows, h)?),
                    sum: TensorMut(matrix(&w.x, s, rows, h)?),
                    eps: self.config.rms_norm_eps,
                },
            )?;
            for layer in 0..16 {
                let weight = &self.weights.layers[usize::from(high) * 16 + layer];
                linear(
                    &self.blas,
                    s,
                    &weight.qkv,
                    matrix(&w.n, s, rows, h)?,
                    matrix(&w.qkv, s, rows, 4 * h)?,
                )?;
                let slot = offset + layer;
                #[cfg(test)]
                let slot = if r.control == 6 { slot % 32 } else { slot };
                let kv = &mut self.state.kv[slot];
                unsafe {
                    s.launch_builder(&r.rotary)
                        .arg(&mut w.qkv)
                        .arg(&mut kv.k)
                        .arg(&mut kv.v)
                        .arg(&r.rope)
                        .arg(&w.position)
                        .arg(&(rows as i32))
                        .arg(&(qh as i32))
                        .arg(&(kh as i32))
                        .arg(&(dim as i32))
                        .arg(&(self.capacity as i32))
                        .arg(&((4 * h) as i32))
                        .launch(flat(rows * (qh + kh) * (dim / 2)))?;
                }
                if token {
                    self.decode
                        .as_mut()
                        .context("decode attention missing")?
                        .run_with(
                            s,
                            GqaDecodeInput {
                                q: &w.qkv,
                                q_offset: 0,
                                k: &kv.k,
                                v: &kv.v,
                                output: &mut w.attn,
                                output_offset: 0,
                                length: &w.length,
                                sinks: None,
                                window: None,
                            },
                        )?;
                } else {
                    self.flash
                        .as_mut()
                        .context("native prefix attention missing")?
                        .run_masked(super::super::flash::Query {
                            q: &w.qkv,
                            q_row_stride: 4 * h,
                            k: &kv.k,
                            v: &kv.v,
                            out: &mut w.attn,
                            keys: rows,
                            prefix: Some(r.prefix),
                            shape: super::super::flash::Shape {
                                rows,
                                start: 0,
                                q_heads: qh,
                                kv_heads: kh,
                                capacity: self.capacity,
                                head_dim: dim,
                                causal: false,
                                window_left: None,
                            },
                        })?;
                }
                let attn = view(
                    &w.attn,
                    s,
                    0,
                    [rows as u32, 1, h as u32],
                    [h as u64, h as u64, 1],
                )?;
                let gate = view(
                    &w.qkv,
                    s,
                    3 * h,
                    [rows as u32, 1, h as u32],
                    [(4 * h) as u64, h as u64, 1],
                )?;
                let gated = view(
                    &w.n,
                    s,
                    0,
                    [rows as u32, 1, h as u32],
                    [h as u64, h as u64, 1],
                )?;
                #[cfg(test)]
                let bypass = r.control == 3;
                #[cfg(not(test))]
                let bypass = false;
                if bypass {
                    s.memcpy_dtod(&w.attn.slice(..rows * h), &mut w.n.slice_mut(..rows * h))?;
                } else {
                    r.common
                        .gate
                        .sigmoid_mul(s, domain, attn, gate, &mut TensorMut(gated))?;
                }
                linear(
                    &self.blas,
                    s,
                    &weight.o,
                    matrix(&w.n, s, rows, h)?,
                    matrix(&w.out, s, rows, h)?,
                )?;
                self.ops.add_rmsnorm(
                    s,
                    AddRmsNorm {
                        input: matrix(&w.out, s, rows, h)?,
                        residual: matrix(&w.x, s, rows, h)?,
                        weight: None,
                        output: TensorMut(matrix(&w.n, s, rows, h)?),
                        sum: TensorMut(matrix(&w.x, s, rows, h)?),
                        eps: self.config.rms_norm_eps,
                    },
                )?;
                linear(
                    &self.blas,
                    s,
                    &weight.gu,
                    matrix(&w.n, s, rows, h)?,
                    matrix(&w.gu, s, rows, 2 * f)?,
                )?;
                unsafe {
                    s.launch_builder(&self.ops.act)
                        .arg(&w.gu)
                        .arg(&mut w.act)
                        .arg(&(rows as i32))
                        .arg(&(f as i32))
                        .arg(&0i32)
                        .launch(flat(rows * f))?;
                }
                linear(
                    &self.blas,
                    s,
                    &weight.down,
                    matrix(&w.act, s, rows, f)?,
                    matrix(&w.out, s, rows, h)?,
                )?;
                self.ops.add_rmsnorm(
                    s,
                    AddRmsNorm {
                        input: matrix(&w.out, s, rows, h)?,
                        residual: matrix(&w.x, s, rows, h)?,
                        weight: None,
                        output: TensorMut(matrix(
                            if layer == 15 { target } else { &w.n },
                            s,
                            rows,
                            h,
                        )?),
                        sum: TensorMut(matrix(&w.x, s, rows, h)?),
                        eps: self.config.rms_norm_eps,
                    },
                )?;
                #[cfg(test)]
                if r.control == 2 && high && layer == 15 {
                    s.memcpy_dtod(&w.x.slice(..rows * h), &mut target.slice_mut(..rows * h))?;
                }
            }
            if trace.is_some() {
                trace_value(
                    &mut trace,
                    s,
                    &format!("stack.{}", offset / 16),
                    target,
                    rows,
                    h,
                )?;
            }
        }
        linear(
            &self.blas,
            s,
            &self.weights.head,
            view(&r.high, s, (rows - 1) * h, [1, h as u32], [h as u64, 1])?,
            matrix(&w.logits, s, 1, self.config.vocab_size)?,
        )?;
        unsafe {
            s.launch_builder(&self.ops.argmax)
                .arg(&w.logits)
                .arg(&mut w.token)
                .arg(&(self.config.vocab_size as i32))
                .launch(grid(1, 1024))?;
        }
        self.logits_cache.mark_valid();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, SafeTensors};

    #[test]
    #[ignore = "authorized GPU window; shared decode head geometry"]
    fn recurrent_decode_shapes_gpu() -> Result<()> {
        let d = Device::new(0)?;
        let s = &d.stream;
        let root = serde_json::from_str(include_str!("../../../tests/data/loopvl-config.json"))?;
        let mut c = crate::model_spec::ModelSpec::loopvl(&root, &serde_json::Value::Null)?.config();
        let mut defects = 0;
        for (qh, kh) in [(12, 12), (16, 2), (32, 4)] {
            c.num_attention_heads = qh;
            c.num_key_value_heads = kh;
            let n = 257;
            let q: Vec<_> = (0..qh * 128)
                .map(|i| bf16::from_f32(((i * 17 % 263) as f32 - 131.0) / 64.0))
                .collect();
            let k: Vec<_> = (0..kh * n * 128)
                .map(|i| bf16::from_f32(((i * 29 % 269) as f32 - 134.0) / 128.0))
                .collect();
            let v: Vec<_> = (0..kh * n * 128)
                .map(|i| bf16::from_f32(((i * 31 % 271) as f32 - 135.0) / 128.0))
                .collect();
            let qd = s.clone_htod(&q)?;
            let kd = s.clone_htod(&k)?;
            let vd = s.clone_htod(&v)?;
            let length = s.clone_htod(&[n as i32])?;
            let mut out = s.alloc_zeros::<bf16>(qh * 128)?;
            let plan = crate::backend::setup::AttentionPlan {
                chunk: 64,
                qpack: 1,
                threads: 128,
                merge_threads: 128,
            };
            let mut attention = Attention::new(&d.ctx, s, &c, n, plan)?;
            attention.run(s, &qd, &kd, &vd, &mut out, &length)?;
            let actual: Vec<_> = s.clone_dtoh(&out)?.into_iter().map(bf16::to_f32).collect();
            let mut reference = vec![0.0f32; qh * 128];
            for head in 0..qh {
                let kv = head / (qh / kh);
                let mut scores: Vec<_> = (0..n)
                    .map(|p| {
                        (0..128)
                            .map(|i| {
                                f64::from(q[head * 128 + i].to_f32())
                                    * f64::from(k[(kv * n + p) * 128 + i].to_f32())
                            })
                            .sum::<f64>()
                            / 128.0f64.sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                }
                let total = scores.iter().sum::<f64>();
                for i in 0..128 {
                    reference[head * 128 + i] = (0..n)
                        .map(|p| scores[p] * f64::from(v[(kv * n + p) * 128 + i].to_f32()) / total)
                        .sum::<f64>() as f32;
                }
            }
            defects += usize::from(!compare(
                &format!("decode_geometry {qh}:{kh}:128"),
                &actual,
                &reference,
                [0.003, 0.002, 0.001],
            )?);
            let shifted: Vec<_> = reference
                .as_chunks::<128>()
                .0
                .iter()
                .flat_map(|h| h.iter().cycle().skip(1).take(128).copied())
                .collect();
            let detected = !compare(
                "decode_dimension_control",
                &actual,
                &shifted,
                [0.003, 0.002, 0.001],
            )?;
            defects += usize::from(!detected);
        }
        ensure!(
            defects == 0,
            "DEFECT shared decode geometry failures={defects}"
        );
        println!("PASS recurrent_decode_shapes_gpu shapes=12:12:128,16:2:128,32:4:128");
        Ok(())
    }

    fn floats(source: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
        let t = source.tensor(name)?;
        ensure!(t.dtype() == Dtype::F32, "Runner FAIL {name} dtype");
        Ok(t.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect())
    }
    fn ids(source: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
        let t = source.tensor(name)?;
        ensure!(t.dtype() == Dtype::I64, "Runner FAIL {name} dtype");
        t.data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i64::from_le_bytes(*v))?))
            .collect()
    }
    fn compare(label: &str, actual: &[f32], expected: &[f32], limits: [f32; 3]) -> Result<bool> {
        ensure!(
            !actual.is_empty() && actual.len() == expected.len(),
            "Runner FAIL {label} shape"
        );
        ensure!(
            actual.iter().chain(expected).all(|v| v.is_finite()),
            "Runner FAIL {label} nonfinite"
        );
        let mut errors: Vec<_> = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let rms = (errors.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / errors.len() as f64)
            .sqrt() as f32;
        errors.sort_by(f32::total_cmp);
        let max = errors[errors.len() - 1];
        let p99 = errors[(errors.len() - 1) * 99 / 100];
        let pass = max <= limits[0] && p99 <= limits[1] && rms <= limits[2];
        println!(
            "{} {label} max_abs={max} p99_abs={p99} rms={rms} limits={limits:?}",
            if pass { "PASS" } else { "DEFECT" }
        );
        Ok(pass)
    }

    fn reinject_reference(
        initial: &[bf16],
        visual_begin: usize,
        visual_rows: usize,
        gates: &[bf16],
        alpha: f32,
    ) -> Vec<bf16> {
        let mut output = initial.to_vec();
        let alpha = bf16::from_f32(alpha);
        for (row, gate) in gates.iter().enumerate().take(visual_rows) {
            let gate =
                bf16::from_f32((f64::from(alpha.to_f32()) * f64::from(gate.to_f32())) as f32);
            for column in 0..1536 {
                let index = (visual_begin + row) * 1536 + column;
                let delta = bf16::from_f32(
                    (f64::from(gate.to_f32()) * f64::from(initial[index].to_f32())) as f32,
                );
                output[index] = bf16::from_f32(
                    (f64::from(initial[index].to_f32()) + f64::from(delta.to_f32())) as f32,
                );
            }
        }
        output
    }

    #[test]
    #[ignore = "authorized GPU window; FF_MODEL and FF_LOOPVL_FIXTURES"]
    fn loopvl_vision_gpu() -> Result<()> {
        let dir =
            PathBuf::from(std::env::var_os("FF_MODEL").context("Runner FAIL FF_MODEL missing")?);
        let fixtures = PathBuf::from(
            std::env::var_os("FF_LOOPVL_FIXTURES")
                .context("Runner FAIL FF_LOOPVL_FIXTURES missing")?,
        );
        let spec = crate::model_spec::ModelSpec::read(&dir)?;
        let mapped = crate::checkpoint::MappedCheckpoint::open(&dir)?;
        let checkpoint_views = mapped.views()?;
        let d = Device::new(0)?;
        let s = &d.stream;
        let mut weights = LoopvlVisionWeights::load(&dir, &spec, &d)?;
        ensure!(
            weights.gate_visual.len() == 256 * 1536 * 4
                && weights.gate_query.len() == 256 * 1536 * 4
                && weights.gate_out.len() == 256 * 4
                && weights.gate_bias.len() == 4
                && weights.raw_alpha.len() == 8,
            "Runner FAIL reinjection weight extents"
        );
        let runtime = VisionRuntime::new(&d.ctx)?;
        let closure = Rc::new(crate::backend::cuda::closure::Closure::new(
            false,
            "LoopVL vision test".into(),
        ));
        let mut blas = Blas::new(&d, &runtime.ops, Path::new(""), closure)?;
        blas.bind_invariant(&d.ctx, 1)?;
        let mut defects = 0;
        let mut outputs = Vec::new();
        for case in ["image_hockey", "image_boat", "control_same_grid_boat"] {
            let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
            let source = SafeTensors::deserialize(&bytes)?;
            let pixels = floats(&source, "pixel_values")?;
            let grid = ids(&source, "grid_sizes")?;
            let input_ids = ids(&source, "input_ids")?;
            ensure!(grid.len() == 3 && grid[0] == 1, "Runner FAIL {case} grid");
            let (grid_h, grid_w) = (grid[1] as usize, grid[2] as usize);
            let rows = grid_h * grid_w;
            ensure!(
                pixels.len() == rows * 588,
                "Runner FAIL {case} pixel extent"
            );
            let pixels: Vec<bf16> = pixels.into_iter().map(bf16::from_f32).collect();
            let pixels = s.clone_htod(&pixels)?;
            let mut work = VisionWorkspace::new(s, rows)?;
            let mut attention = Flash::new(s.clone(), rows, 16, 8, 128)?;
            let encoder_before = runtime.encoder_calls.get();
            runtime.run_encoder(
                &mut blas,
                &mut attention,
                s,
                &mut work,
                VisionEncoderCall {
                    weights: &weights,
                    pixels: &pixels,
                    rows,
                    grid_h,
                    grid_w,
                },
            )?;
            let projector_before = runtime.projector_calls.get();
            runtime.run_projector(&blas, &weights, s, &mut work, rows)?;
            ensure!(
                runtime.encoder_calls.get() == encoder_before + 1
                    && runtime.projector_calls.get() == projector_before + 1,
                "DEFECT {case} media call count"
            );
            let raw: Vec<f32> = s
                .clone_dtoh(&work.n)?
                .into_iter()
                .map(bf16::to_f32)
                .collect();
            let raw_reference = floats(&source, "vision_raw_features")?;
            defects += usize::from(!compare(
                &format!("{case} encoder"),
                &raw,
                &raw_reference,
                [f32::MAX, 0.08, 0.035],
            )?);
            let projected = s.clone_dtoh(&work.projected_f32)?;
            let projected_reference = floats(&source, "visual_projected")?;
            defects += usize::from(!compare(
                &format!("{case} encoder_projector"),
                &projected,
                &projected_reference,
                [f32::MAX, 0.08, 0.035],
            )?);
            let projected_bf16 = s.clone_dtoh(&work.projected_bf16)?;
            let raw_bf16: Vec<bf16> = raw_reference.iter().copied().map(bf16::from_f32).collect();
            s.memcpy_htod(&raw_bf16, &mut work.n)?;
            runtime.run_projector(&blas, &weights, s, &mut work, rows)?;
            let isolated = s.clone_dtoh(&work.projected_f32)?;
            defects += usize::from(!compare(
                &format!("{case} isolated_projector"),
                &isolated,
                &projected_reference,
                [1e-4, f32::MAX, 1e-5],
            )?);
            if case == "image_hockey" {
                let projected = s.clone_dtoh(&work.projected_bf16)?;
                let embedding = checkpoint_views["model.safetensors"]
                    .tensor("language_model.model.embed_tokens.weight")?;
                let mut initial = vec![bf16::from_f32(0.0); input_ids.len() * 1536];
                let embedding_scale = 39.191835884530846f64 as f32;
                for position in 0..input_ids.len() {
                    for column in 0..1536 {
                        let index = position * 1536 + column;
                        initial[index] = if (6..6 + rows).contains(&position) {
                            bf16::from_f32(
                                projected[(position - 6) * 1536 + column].to_f32()
                                    * embedding_scale,
                            )
                        } else {
                            let offset = (input_ids[position] as usize * 1536 + column) * 2;
                            bf16::from_f32(
                                bf16::from_bits(u16::from_le_bytes(
                                    embedding.data()[offset..offset + 2].try_into()?,
                                ))
                                .to_f32()
                                    * embedding_scale,
                            )
                        };
                    }
                }
                let anchor = initial[6 * 1536..(6 + rows) * 1536].to_vec();
                let anchor_dev = s.clone_htod(&anchor)?;
                let mut device_states = s.clone_htod(&initial)?;
                runtime.run_reinject(
                    &blas,
                    s,
                    &mut work,
                    ReinjectCall {
                        weights: &weights,
                        states: &mut device_states,
                        anchor: &anchor_dev,
                        visual_begin: 6,
                        visual_rows: rows,
                        instruction_begin: 6 + rows,
                        instruction_end: input_ids.len(),
                        cycle: 0,
                    },
                )?;
                let gates = s.clone_dtoh(&work.gates)?;
                let gates_f32: Vec<f32> = gates.iter().map(|v| v.to_f32()).collect();
                defects += usize::from(!compare(
                    "cycle0_gate",
                    &gates_f32,
                    &floats(&source, "gate_h0")?,
                    [0.08, f32::MAX, 0.005],
                )?);
                let alphas = floats(&source, "alpha_h0_h1")?;
                let actual = s.clone_dtoh(&device_states)?;
                let expected = reinject_reference(&initial, 6, rows, &gates, alphas[0]);
                let bit_mismatches = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                defects += usize::from(bit_mismatches != 0);
                println!(
                    "{} cycle0_reinjection bit_mismatches={bit_mismatches}",
                    if bit_mismatches == 0 {
                        "PASS"
                    } else {
                        "DEFECT"
                    }
                );
                let saved_alpha = s.clone_dtoh(&weights.raw_alpha)?;
                s.memcpy_htod(&[0u8; 8], &mut weights.raw_alpha)?;
                ensure!(
                    s.clone_dtoh(&weights.raw_alpha)? == [0u8; 8],
                    "Runner FAIL zero alpha upload"
                );
                let mut zero_states = s.clone_htod(&initial)?;
                runtime.run_reinject(
                    &blas,
                    s,
                    &mut work,
                    ReinjectCall {
                        weights: &weights,
                        states: &mut zero_states,
                        anchor: &anchor_dev,
                        visual_begin: 6,
                        visual_rows: rows,
                        instruction_begin: 6 + rows,
                        instruction_end: input_ids.len(),
                        cycle: 0,
                    },
                )?;
                let zero = s.clone_dtoh(&zero_states)?;
                let zero_gates = s.clone_dtoh(&work.gates)?;
                let expected_zero = reinject_reference(&initial, 6, rows, &zero_gates, 0.25);
                let bit_mismatches = zero
                    .iter()
                    .zip(&expected_zero)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                defects += usize::from(bit_mismatches != 0);
                println!(
                    "{} control5_zero_alpha_replica bit_mismatches={bit_mismatches}",
                    if bit_mismatches == 0 {
                        "PASS"
                    } else {
                        "DEFECT"
                    }
                );
                let zero_f32: Vec<f32> = zero.iter().map(|v| v.to_f32()).collect();
                let expected_f32: Vec<f32> = expected.iter().map(|v| v.to_f32()).collect();
                let control_passes_good_bound = compare(
                    "control5_zero_alpha_vs_reinjected",
                    &zero_f32,
                    &expected_f32,
                    [f32::MAX, 0.20, 0.06],
                )?;
                defects += usize::from(control_passes_good_bound);
                let skipped_f32: Vec<f32> = initial.iter().map(|v| v.to_f32()).collect();
                let skip_passes_good_bound = compare(
                    "control5_skip_h0_vs_reinjected",
                    &skipped_f32,
                    &expected_f32,
                    [f32::MAX, 0.20, 0.06],
                )?;
                defects += usize::from(skip_passes_good_bound);
                println!("PASS control5 zero_alpha_and_skip_h0 detected=true");
                s.memcpy_htod(&saved_alpha, &mut weights.raw_alpha)?;
                let mut cycle1_states = s.clone_htod(&initial)?;
                runtime.run_reinject(
                    &blas,
                    s,
                    &mut work,
                    ReinjectCall {
                        weights: &weights,
                        states: &mut cycle1_states,
                        anchor: &anchor_dev,
                        visual_begin: 6,
                        visual_rows: rows,
                        instruction_begin: 6 + rows,
                        instruction_end: input_ids.len(),
                        cycle: 1,
                    },
                )?;
                let cycle1 = s.clone_dtoh(&cycle1_states)?;
                let expected1 = reinject_reference(&initial, 6, rows, &gates, alphas[1]);
                let bit_mismatches = cycle1
                    .iter()
                    .zip(&expected1)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                defects += usize::from(bit_mismatches != 0);
                println!(
                    "{} cycle1_reinjection bit_mismatches={bit_mismatches}",
                    if bit_mismatches == 0 {
                        "PASS"
                    } else {
                        "DEFECT"
                    }
                );
                ensure!(
                    runtime.reinjection_calls.get() == 3,
                    "DEFECT reinjection did not execute h0/h1 and control"
                );
            }
            println!(
                "PASS {case} media_calls encoder=1 projector=1 isolated_projector=1 rows={rows}"
            );
            outputs.push((case, input_ids, grid, projected_bf16));
        }
        let hockey = &outputs[0];
        let control = &outputs[2];
        ensure!(
            hockey.1 == control.1 && hockey.2 == control.2,
            "Runner FAIL same-grid control inputs differ"
        );
        let media_difference = hockey
            .3
            .iter()
            .zip(&control.3)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        defects += usize::from(media_difference == 0);
        println!(
            "{} media_same_grid projected_differing={media_difference}",
            if media_difference > 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        ensure!(defects == 0, "DEFECT loopvl_vision_gpu failures={defects}");
        println!("PASS loopvl_vision_gpu fixtures=3 controls=media_reachability");
        Ok(())
    }

    #[test]
    #[ignore = "authorized GPU window; FF_MODEL and FF_LOOPVL_FIXTURES"]
    fn loopvl_text_gpu() -> Result<()> {
        let dir =
            PathBuf::from(std::env::var_os("FF_MODEL").context("Runner FAIL FF_MODEL missing")?);
        let fixtures = PathBuf::from(
            std::env::var_os("FF_LOOPVL_FIXTURES")
                .context("Runner FAIL FF_LOOPVL_FIXTURES missing")?,
        );
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            runtime_dir: None,
        };
        let mut engine = Engine::load(&dir, 0, 8192, None, settings)?;
        let mut defects = 0;
        for (case, rows) in [("text_short", 21), ("text_medium", 42), ("text_cap", 64)] {
            let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
            let source = SafeTensors::deserialize(&bytes)?;
            let prompt = ids(&source, "input_ids")?;
            let tokens = ids(&source, "oracle_greedy_ids")?;
            let predictions = ids(&source, "oracle_argmax_ids")?;
            let reference = floats(&source, "oracle_step_logits")?;
            let margins = floats(&source, "oracle_top2_margins")?;
            ensure!(
                prompt.len() == rows
                    && tokens.len() == 32
                    && predictions.len() == 32
                    && margins.len() == 32
                    && source.tensor("oracle_step_logits")?.shape() == [32, 65536],
                "Runner FAIL fixture {case} extent"
            );
            let mut eager = Vec::new();
            let mut eager_ids = Vec::new();
            let mut eager_free = Vec::new();
            for graph in [false, true] {
                let mut trajectory = Vec::with_capacity(reference.len());
                engine.reset()?;
                let mut trace = Trace::default();
                engine.prefill(&prompt, Some(&mut trace))?;
                ensure!(
                    engine
                        .recurrent
                        .as_ref()
                        .context("recurrent runtime")?
                        .vision
                        .encoder_calls
                        .get()
                        == 0,
                    "DEFECT text-only request called the vision encoder"
                );
                let prefill = engine.decode_logits()?;
                defects += usize::from(!compare(
                    &format!("{case} prefill graph={graph}"),
                    &prefill,
                    &reference[..65536],
                    [0.50, 0.15, 0.06],
                )?);
                let low = floats(&source, "l_last_row_cycles")?;
                let high = floats(&source, "h_last_row_cycles")?;
                for call in 0..8 {
                    let tap = trace.values(&format!("prefill.stack.{call}"))?;
                    let oracle = if call % 4 == 3 {
                        &high[(call / 4) * 1536..(call / 4 + 1) * 1536]
                    } else {
                        let i = call - call / 4;
                        &low[i * 1536..(i + 1) * 1536]
                    };
                    defects += usize::from(!compare(
                        &format!("{case} stack={call}"),
                        &tap[(rows - 1) * 1536..],
                        oracle,
                        [f32::MAX, 0.20, 0.06],
                    )?);
                }
                for step in 0..32 {
                    if step > 0 {
                        engine.set_token(tokens[step - 1])?;
                        engine.decode(graph, None)?;
                    }
                    let actual = engine.decode_logits()?;
                    trajectory.extend_from_slice(&actual);
                    let label = format!("{case} step={step} graph={graph}");
                    defects += usize::from(!compare(
                        &label,
                        &actual,
                        &reference[step * 65536..(step + 1) * 65536],
                        [0.60, f32::MAX, f32::MAX],
                    )?);
                    let prediction = engine.token()?;
                    let margin = margins[step];
                    let pass = margin < 0.35 || prediction == predictions[step];
                    defects += usize::from(!pass);
                    println!(
                        "{} {label} id={prediction} oracle={} margin={margin}",
                        if pass { "PASS" } else { "DEFECT" },
                        predictions[step]
                    );
                    if graph {
                        let bits = actual
                            .iter()
                            .zip(&eager[step])
                            .filter(|(a, b): &(&f32, &f32)| a.to_bits() != b.to_bits())
                            .count();
                        println!(
                            "{} {label} graph_eager_bit_mismatches={bits}",
                            if bits == 0 { "PASS" } else { "DEFECT" }
                        );
                        defects += usize::from(bits != 0);
                        defects += usize::from(prediction != eager_ids[step]);
                    } else {
                        eager.push(actual);
                        eager_ids.push(prediction);
                    }
                }
                defects += usize::from(!compare(
                    &format!("{case} trajectory=[32,65536] graph={graph}"),
                    &trajectory,
                    &reference,
                    [0.60, 0.20, 0.06],
                )?);
                engine.reset()?;
                let mut generated = vec![engine.prefill(&prompt, None)?];
                for _ in 1..32 {
                    generated.push(engine.decode(graph, None)?);
                }
                println!(
                    "TRAJECTORY {case} graph={graph} ids={generated:?} text={:?}",
                    crate::tokenizer::decode(&dir, &generated)?
                );
                if graph {
                    let identical = generated == eager_free;
                    defects += usize::from(!identical);
                    println!(
                        "{} {case} free_running_graph_eager_ids_equal={identical}",
                        if identical { "PASS" } else { "DEFECT" }
                    );
                } else {
                    eager_free = generated;
                }
            }
            if case == "text_short" {
                defects += controls(&mut engine, &prompt, &tokens, &reference)?;
            }
        }
        let position = engine.position;
        ensure!(
            engine.rewind(0, 6).is_err() && engine.position == position,
            "DEFECT rewind accepted a truncated bidirectional prefix"
        );
        engine.reset()?;
        ensure!(
            engine.prefill(&vec![6; engine.capacity], None).is_err(),
            "DEFECT prefix workspace accepted capacity rows"
        );
        ensure!(
            engine.position == 0,
            "DEFECT rejected prefix mutated position"
        );
        engine.set_token(6)?;
        engine
            .device
            .stream
            .memcpy_htod(&[8191i32], &mut engine.work.position)?;
        engine
            .device
            .stream
            .memcpy_htod(&[8192i32], &mut engine.work.length)?;
        engine.position = 8191;
        engine.decode(false, None)?;
        engine.check_logits()?;
        ensure!(
            engine.decode(false, None).is_err(),
            "DEFECT decode exceeded KV capacity"
        );
        engine.synchronize()?;
        ensure!(
            defects == 0,
            "DEFECT loopvl_text_gpu comparisons_failed={defects}"
        );
        println!(
            "PASS loopvl_text_gpu fixtures=3 teacher_forced_steps=32 modes=eager,graph native_prefix=true controls=1,2,3,6"
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn loopvl_image_gpu() -> Result<()> {
        let dir =
            PathBuf::from(std::env::var_os("FF_MODEL").context("Runner FAIL FF_MODEL missing")?);
        let fixtures = PathBuf::from(
            std::env::var_os("FF_LOOPVL_FIXTURES")
                .context("Runner FAIL FF_LOOPVL_FIXTURES missing")?,
        );
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            draft_model: None,
            prefix_cache: false,
            kv_format: crate::kv::KvFormat::Bf16,
            runtime_dir: None,
        };
        let mut engine = Engine::load(&dir, 0, 8192, None, settings)?;
        let dump = std::env::var_os("FF_Q38_PROBE_DIR").map(PathBuf::from);
        if let Some(out) = &dump {
            std::fs::create_dir_all(out)?;
            ensure!(
                !out.canonicalize()?.starts_with(dir.canonicalize()?),
                "probe output is inside model directory"
            );
        }
        let mut defects = 0;
        let mut media_outputs = Vec::new();
        for (case, rows, steps) in [
            ("image_hockey", 580, 29),
            ("image_boat", 2069, 22),
            ("control_same_grid_boat", 580, 32),
        ] {
            let bytes = std::fs::read(fixtures.join(format!("{case}.safetensors")))?;
            let source = SafeTensors::deserialize(&bytes)?;
            let prompt = ids(&source, "input_ids")?;
            let tokens = ids(&source, "oracle_greedy_ids")?;
            let predictions = ids(&source, "oracle_argmax_ids")?;
            let reference = floats(&source, "oracle_step_logits")?;
            let margins = floats(&source, "oracle_top2_margins")?;
            let pixels = floats(&source, "pixel_values")?;
            let grid = ids(&source, "grid_sizes")?;
            let lengths = ids(&source, "header_and_instruction_lengths")?;
            ensure!(
                prompt.len() == rows
                    && tokens.len() == steps
                    && predictions.len() == steps
                    && margins.len() == steps
                    && source.tensor("oracle_step_logits")?.shape() == [steps, 65536]
                    && grid.len() == 3
                    && lengths.len() == 2,
                "Runner FAIL {case} fixture extent"
            );
            let grids = [crate::vision_input::ImageGrid {
                t: grid[0],
                h: grid[1],
                w: grid[2],
            }];
            let begin = lengths[0] as usize;
            let count = grid[1] as usize * grid[2] as usize;
            ensure!(
                begin + count + lengths[1] as usize == rows,
                "Runner FAIL {case} image span"
            );
            let mut modality = vec![0; rows];
            modality[begin..begin + count].fill(1);
            let input = crate::vision_input::MultimodalInput {
                input_ids: &prompt,
                modality_ids: &modality,
                pixel_values: &pixels,
                grids: &grids,
            };
            let mut eager = Vec::new();
            let mut eager_ids = Vec::new();
            let mut eager_free = Vec::new();
            for graph in [false, true] {
                engine.reset()?;
                let runtime = &engine
                    .recurrent
                    .as_ref()
                    .context("recurrent runtime")?
                    .vision;
                let calls = (
                    runtime.encoder_calls.get(),
                    runtime.projector_calls.get(),
                    runtime.reinjection_calls.get(),
                );
                let mut trace = Trace::default();
                engine.prefill_multimodal(&input, Some(&mut trace))?;
                let runtime = &engine
                    .recurrent
                    .as_ref()
                    .context("recurrent runtime")?
                    .vision;
                let reached = runtime.encoder_calls.get() == calls.0 + 1
                    && runtime.projector_calls.get() == calls.1 + 1
                    && runtime.reinjection_calls.get() == calls.2 + 2;
                defects += usize::from(!reached);
                println!(
                    "{} {case} graph={graph} encoder_projector_reinjection_delta=1,1,2 media_reached={reached}",
                    if reached { "PASS" } else { "DEFECT" }
                );
                let prefill = engine.decode_logits()?;
                defects += usize::from(!compare(
                    &format!("{case} prefill graph={graph}"),
                    &prefill,
                    &reference[..65536],
                    [0.50, 0.15, 0.06],
                )?);
                let low = floats(&source, "l_last_row_cycles")?;
                let high = floats(&source, "h_last_row_cycles")?;
                for call in 0..8 {
                    let tap = trace.values(&format!("prefill.stack.{call}"))?;
                    let oracle = if call % 4 == 3 {
                        &high[(call / 4) * 1536..(call / 4 + 1) * 1536]
                    } else {
                        let i = call - call / 4;
                        &low[i * 1536..(i + 1) * 1536]
                    };
                    defects += usize::from(!compare(
                        &format!("{case} stack={call}"),
                        &tap[(rows - 1) * 1536..],
                        oracle,
                        [f32::MAX, 0.20, 0.06],
                    )?);
                }
                let mut trajectory = Vec::with_capacity(reference.len());
                for step in 0..steps {
                    let mut step_trace = Trace::default();
                    if step > 0 {
                        engine.set_token(tokens[step - 1])?;
                        let capture =
                            (!graph && dump.is_some()).then_some((&mut step_trace, "decode"));
                        engine.decode(graph, capture)?;
                    }
                    let actual = engine.decode_logits()?;
                    if !graph && let Some(out) = &dump {
                        step_trace.set_device(&engine.device.info.name)?;
                        step_trace.add("logits".into(), vec![65536], actual.clone())?;
                        step_trace.save(&out.join(format!("{case}-step{step}.safetensors")))?;
                    }
                    trajectory.extend_from_slice(&actual);
                    defects += usize::from(!compare(
                        &format!("{case} step={step} graph={graph}"),
                        &actual,
                        &reference[step * 65536..(step + 1) * 65536],
                        [0.60, f32::MAX, f32::MAX],
                    )?);
                    let prediction = engine.token()?;
                    let pass = margins[step] < 0.35 || prediction == predictions[step];
                    defects += usize::from(!pass);
                    println!(
                        "{} {case} step={step} graph={graph} id={prediction} oracle={} margin={}",
                        if pass { "PASS" } else { "DEFECT" },
                        predictions[step],
                        margins[step]
                    );
                    if graph {
                        let bits = actual
                            .iter()
                            .zip(&eager[step])
                            .filter(|(a, b): &(&f32, &f32)| a.to_bits() != b.to_bits())
                            .count();
                        defects += usize::from(bits != 0 || prediction != eager_ids[step]);
                        println!(
                            "{} {case} step={step} graph_eager_bit_mismatches={bits}",
                            if bits == 0 { "PASS" } else { "DEFECT" }
                        );
                    } else {
                        eager.push(actual);
                        eager_ids.push(prediction);
                    }
                }
                defects += usize::from(!compare(
                    &format!("{case} trajectory=[{steps},65536] graph={graph}"),
                    &trajectory,
                    &reference,
                    [0.60, 0.20, 0.06],
                )?);
                engine.reset()?;
                let mut generated = vec![engine.prefill_multimodal(&input, None)?];
                for _ in 1..steps {
                    generated.push(engine.decode(graph, None)?);
                }
                let text = crate::tokenizer::decode(&dir, &generated)?;
                println!("TRAJECTORY {case} graph={graph} ids={generated:?} text={text:?}");
                if graph {
                    defects += usize::from(generated != eager_free);
                    println!(
                        "{} {case} free_running_graph_eager_ids_equal={}",
                        if generated == eager_free {
                            "PASS"
                        } else {
                            "DEFECT"
                        },
                        generated == eager_free
                    );
                } else {
                    eager_free = generated.clone();
                    media_outputs.push((prefill, generated, text));
                }
            }
            if case == "image_hockey" {
                for control in [4, 7, 5, 8] {
                    engine.reset()?;
                    let runtime = engine.recurrent.as_mut().context("recurrent runtime")?;
                    runtime.control = control;
                    let alpha = engine
                        .device
                        .stream
                        .clone_dtoh(&runtime.vision_weights.raw_alpha)?;
                    if control == 8 {
                        engine
                            .device
                            .stream
                            .memcpy_htod(&[0u8; 8], &mut runtime.vision_weights.raw_alpha)?;
                    }
                    engine.prefill_multimodal(&input, None)?;
                    let wrong = engine.decode_logits()?;
                    let detected = !compare(
                        &format!("control{control}_image_candidate"),
                        &wrong,
                        &reference[..65536],
                        [0.50, 0.15, 0.06],
                    )?;
                    defects += usize::from(!detected);
                    println!(
                        "{} control{control}_image_logits_detected={detected}",
                        if detected { "PASS" } else { "DEFECT" }
                    );
                    let runtime = engine.recurrent.as_mut().context("recurrent runtime")?;
                    runtime.control = 0;
                    engine
                        .device
                        .stream
                        .memcpy_htod(&alpha, &mut runtime.vision_weights.raw_alpha)?;
                }
            }
        }
        let shift = media_outputs[0]
            .0
            .iter()
            .zip(&media_outputs[2].0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let distinct = media_outputs
            .iter()
            .enumerate()
            .all(|(i, a)| media_outputs[..i].iter().all(|b| a.1 != b.1 && a.2 != b.2));
        defects += usize::from(shift < 4.0 || !distinct);
        println!(
            "{} same_grid_media_logit_shift={shift} min=4 distinct_ids_and_text={distinct}",
            if shift >= 4.0 && distinct {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        ensure!(defects == 0, "DEFECT loopvl_image_gpu failures={defects}");
        println!(
            "PASS loopvl_image_gpu fixtures=3 teacher_forced_steps=29,22,32 graph_eager_bits=0 controls=4,5 media_reached=true"
        );
        Ok(())
    }

    fn controls(
        engine: &mut Engine,
        prompt: &[u32],
        tokens: &[u32],
        oracle: &[f32],
    ) -> Result<usize> {
        let mut defects = 0;
        engine.reset()?;
        let mut base = Trace::default();
        engine.prefill(prompt, Some(&mut base))?;
        let baseline = base.values("prefill.stack.0")?;
        let mut changed = prompt.to_vec();
        let n = changed.len();
        changed[n - 1] = (changed[n - 1] + 1) % 65536;
        engine.reset()?;
        let mut trace = Trace::default();
        engine.prefill(&changed, Some(&mut trace))?;
        let mutated = trace.values("prefill.stack.0")?;
        let delta = baseline[..1536]
            .iter()
            .zip(&mutated[..1536])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        defects += usize::from(delta == 0);
        println!(
            "{} control1_bidirectional earlier_row_changed={delta}",
            if delta > 0 { "PASS" } else { "DEFECT" }
        );
        let mut both = prompt.to_vec();
        both.extend_from_slice(&tokens[..2]);
        engine.reset()?;
        let mut base = Trace::default();
        engine.prefill_recurrent(&both, prompt.len(), Some(&mut base), both.len())?;
        let pos = prompt.len() * 1536;
        let first = base.values("prefill.stack.7")?;
        let last = both.len() - 1;
        both[last] = (both[last] + 1) % 65536;
        engine.reset()?;
        let mut trace = Trace::default();
        engine.prefill_recurrent(&both, prompt.len(), Some(&mut trace), both.len())?;
        let second = trace.values("prefill.stack.7")?;
        let delta = first[pos..pos + 1536]
            .iter()
            .zip(&second[pos..pos + 1536])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        defects += usize::from(delta != 0);
        println!(
            "{} control1_causal earlier_response_bit_changes={delta}",
            if delta == 0 { "PASS" } else { "DEFECT" }
        );
        for control in [2, 3, 6] {
            engine.reset()?;
            engine
                .recurrent
                .as_mut()
                .context("runtime missing")?
                .control = control;
            engine.prefill(prompt, None)?;
            if control == 6 {
                engine.set_token(tokens[0])?;
                engine.decode(false, None)?;
            }
            let actual = engine.decode_logits()?;
            let step = usize::from(control == 6);
            let limits = if control == 2 {
                [f32::MAX, f32::MAX, 0.60]
            } else {
                [0.60, 0.20, 0.06]
            };
            let detected = !compare(
                &format!("control{control}_candidate"),
                &actual,
                &oracle[step * 65536..(step + 1) * 65536],
                limits,
            )?;
            println!(
                "{} control{control}_detected={detected}",
                if detected { "PASS" } else { "DEFECT" }
            );
            defects += usize::from(!detected);
            engine
                .recurrent
                .as_mut()
                .context("runtime missing")?
                .control = 0;
        }
        Ok(defects)
    }
}
