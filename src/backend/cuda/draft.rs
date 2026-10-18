use super::{
    Device,
    attention_d256::{AttentionBindings, FullAttentionOps, FullAttentionPlan, PositionBindings},
    blas::Blas,
    contracts::{Scratch, SeqRef, TensorMut, view},
    copy::{Rect, columns},
    flash::Shape,
    ops::{CommonOpsRef, Ops, RmsMode, flat, grid},
    verification::Verification,
    weights::{Layer, checkpoint},
};
use crate::{config::Config, dspark::DraftConfig, trace::Trace};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg};
use half::bf16;
use std::{path::Path, sync::Arc};

pub struct Compute<'a> {
    pub ops: &'a Ops,
    pub blas: &'a mut Blas,
}

enum Output {
    Tokens,
    Device,
    Base,
}

enum Attention {
    Cached(Verification),
    Block(BlockAttention),
}

struct BlockAttention {
    layers: Vec<FullAttentionPlan>,
    metadata: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    mask: CudaSlice<u64>,
    scratch: CudaSlice<u8>,
    prepare: cudarc::driver::CudaFunction,
}

impl BlockAttention {
    fn new(draft: &Draft, device: &Device) -> Result<Self> {
        use crate::op_contract::{AttentionLimits, FullAttentionSpec};
        let c = &draft.config;
        let s = &device.stream;
        let rows = u32::try_from(c.block_size)?;
        let groups = c.num_key_value_heads
            * ((c.num_attention_heads / c.num_key_value_heads) * c.block_size).div_ceil(64);
        let partitions = device.info.sms.div_ceil(groups).max(1);
        let key_chunk = draft.capacity.div_ceil(partitions).next_multiple_of(32);
        let limits = AttentionLimits {
            max_rows: rows,
            capacity: u32::try_from(draft.capacity)?,
            key_chunk: u32::try_from(key_chunk)?,
            causal: 1,
        };
        let spec = FullAttentionSpec {
            hq: u32::try_from(c.num_attention_heads)?,
            hkv: u32::try_from(c.num_key_value_heads)?,
            d: u32::try_from(c.head_dim)?,
            rotary_dim: u32::try_from(c.head_dim)?,
            scale: (c.head_dim as f32).sqrt().recip(),
            rms_eps: c.rms_norm_eps,
            rope_theta: c.rope_parameters.rope_theta,
        };
        let request = FullAttentionOps::workspace(&spec, limits)?;
        let live = if rows == 64 {
            u64::MAX
        } else {
            (1u64 << rows) - 1
        };
        let mask = s.clone_htod(&std::array::from_fn::<_, 64, _>(|row| {
            if row < c.block_size { live } else { 0 }
        }))?;
        let positions = s.alloc_zeros::<i32>(c.block_size * 3)?;
        let metadata = s.alloc_zeros::<i32>(8)?;
        let scratch = s.alloc_zeros::<u8>(usize::try_from(request.bytes)?)?;
        let common = CommonOpsRef::new(&device.ctx)?;
        let width = c.num_attention_heads * c.head_dim;
        let kv_width = c.kv_dim();
        let head = c.head_dim as u32;
        let q_shape = [rows, spec.hq, head];
        let q_stride = [width as u64, u64::from(head), 1];
        let kv_shape = [rows, spec.hkv, head];
        let kv_stride = [c.qkv_dim() as u64, u64::from(head), 1];
        let cache_shape = [spec.hkv, limits.capacity, head];
        let cache_stride = [(draft.capacity * c.head_dim) as u64, u64::from(head), 1];
        let mut layers = Vec::new();
        for (layer, cache) in draft.weights.layers.iter().zip(&draft.kv) {
            let bindings = AttentionBindings {
                compact: None,
                q_gate: view(
                    &draft.qkv,
                    s,
                    0,
                    q_shape,
                    [c.qkv_dim() as u64, u64::from(head), 1],
                )?,
                k: view(&draft.qkv, s, width, kv_shape, kv_stride)?,
                v: view(&draft.qkv, s, width + kv_width, kv_shape, kv_stride)?,
                q: TensorMut(view(&draft.attn, s, 0, q_shape, q_stride)?),
                gate: None,
                norm_mode: RmsMode::Standard,
                core_out: TensorMut(view(&draft.attn, s, 0, q_shape, q_stride)?),
                k_cache: TensorMut(view(&cache.k, s, 0, cache_shape, cache_stride)?),
                v_cache: TensorMut(view(&cache.v, s, 0, cache_shape, cache_stride)?),
                q_norm: view(&layer.q_norm, s, 0, [head], [1])?,
                k_norm: view(&layer.k_norm, s, 0, [head], [1])?,
                causal_mask: view(&mask, s, 0, [64], [1])?,
                positions: PositionBindings::Axes(view(&positions, s, 0, [rows, 3], [3, 1])?),
            };
            layers.push(FullAttentionOps::prepare_reusing(
                &device.ctx,
                &spec,
                limits,
                bindings,
                common.clone(),
                layers.first(),
            )?);
        }
        Ok(Self {
            layers,
            metadata,
            positions,
            mask,
            scratch,
            prepare: super::cubin::module(&device.ctx, "rope")?
                .load_function("sequence_positions")?,
        })
    }
}

struct LayerWeights {
    block: Layer,
    q_norm: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
}

struct DraftWeights {
    layers: Vec<LayerWeights>,
    fc: CudaSlice<bf16>,
    hidden_norm: CudaSlice<bf16>,
    norm: CudaSlice<bf16>,
    w1: CudaSlice<bf16>,
    w2: CudaSlice<bf16>,
    confidence: CudaSlice<bf16>,
    confidence_bias: CudaSlice<bf16>,
    bytes: usize,
}

impl DraftWeights {
    fn load(path: &Path, c: &DraftConfig, d: &Device) -> Result<Self> {
        checkpoint(path, d, |l| {
            let h = c.hidden_size;
            let f = c.intermediate_size;
            let kv = c.num_key_value_heads * c.head_dim;
            let fc = l.load(&[("fc.weight".into(), vec![h, c.target_layer_ids.len() * h])])?;
            let hidden_norm = l.load(&[("hidden_norm.weight".into(), vec![h])])?;
            let mut layers = Vec::new();
            for i in 0..c.num_hidden_layers {
                let prefix = format!("layers.{i}");
                let mut one = |name: &str, shape: Vec<usize>| {
                    l.load(&[(format!("{prefix}.{name}.weight"), shape)])
                };
                let input_norm = one("input_layernorm", vec![h])?;
                let post_norm = one("post_attention_layernorm", vec![h])?;
                let o = one("self_attn.o_proj", vec![h, h])?;
                let down = one("mlp.down_proj", vec![h, f])?;
                let q_norm = one("self_attn.q_norm", vec![c.head_dim])?;
                let k_norm = one("self_attn.k_norm", vec![c.head_dim])?;
                let qkv = l.load(&[
                    (format!("{prefix}.self_attn.q_proj.weight"), vec![h, h]),
                    (format!("{prefix}.self_attn.k_proj.weight"), vec![kv, h]),
                    (format!("{prefix}.self_attn.v_proj.weight"), vec![kv, h]),
                ])?;
                let gu = l.load(&[
                    (format!("{prefix}.mlp.gate_proj.weight"), vec![f, h]),
                    (format!("{prefix}.mlp.up_proj.weight"), vec![f, h]),
                ])?;
                layers.push(LayerWeights {
                    block: Layer {
                        input_norm,
                        qkv,
                        o,
                        post_norm,
                        gu,
                        down,
                    },
                    q_norm,
                    k_norm,
                });
            }
            let norm = l.load(&[("norm.weight".into(), vec![h])])?;
            let w1 = l.load(&[(
                "markov_head.markov_w1.weight".into(),
                vec![c.vocab_size, c.markov_rank],
            )])?;
            let w2 = l.load(&[(
                "markov_head.markov_w2.weight".into(),
                vec![c.vocab_size, c.markov_rank],
            )])?;
            let confidence = l.load(&[(
                "confidence_head.proj.weight".into(),
                vec![1, h + c.markov_rank],
            )])?;
            let confidence_bias = l.load(&[("confidence_head.proj.bias".into(), vec![1])])?;
            Ok(Self {
                layers,
                fc,
                hidden_norm,
                norm,
                w1,
                w2,
                confidence,
                confidence_bias,
                bytes: l.bytes,
            })
        })
    }
}

struct Kv {
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
}

pub struct Draft {
    markov: Option<super::markov::Markov>,
    score_rounding: super::markov::ScoreRounding,
    pub config: DraftConfig,
    weights: DraftWeights,
    kv: Vec<Kv>,
    capacity: usize,
    rows: usize,
    stream: Arc<CudaStream>,
    attention: Option<Attention>,
    #[cfg(test)]
    position_shift: i32,
    qknorm_rope_kv: cudarc::driver::CudaFunction,
    norm: cudarc::driver::CudaFunction,
    add_norm: cudarc::driver::CudaFunction,
    context_rope: cudarc::driver::CudaFunction,
    ids: CudaSlice<u32>,
    mask_ids: Vec<u32>,
    position: CudaSlice<i32>,
    x: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    gu: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    k_raw: CudaSlice<bf16>,
    k_norm: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    projected: CudaSlice<bf16>,
    context: CudaSlice<bf16>,
    packed: CudaSlice<bf16>,
    base: CudaSlice<bf16>,
    scores: CudaSlice<bf16>,
    combined: Option<CudaSlice<f32>>,
    confidence_values: CudaSlice<f32>,
    forced_tokens: CudaSlice<u32>,
    sum_argmax: cudarc::driver::CudaFunction,
    concat_dot_sigmoid: cudarc::driver::CudaFunction,
    prev: CudaSlice<bf16>,
    token: CudaSlice<u32>,
    tokens: CudaSlice<u32>,
}

fn norm(
    s: &Arc<CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    shape: (usize, usize),
    eps: f32,
) -> Result<()> {
    let (rows, dim) = shape;
    unsafe {
        s.launch_builder(kernel)
            .arg(input)
            .arg(weight)
            .arg(output)
            .arg(&(rows as i32))
            .arg(&(dim as i32))
            .arg(&eps)
            .launch(grid(rows, 256))?;
    }
    Ok(())
}

impl Draft {
    pub(super) fn calibrate_injection(
        &self,
        blas: &mut Blas,
        input: &cudarc::driver::CudaView<'_, bf16>,
        shape: crate::backend::setup::LinearShape,
    ) -> Result<()> {
        if shape.input == self.config.capture_width() && shape.output == self.config.hidden_size {
            blas.calibrate_shape(&self.weights.fc, input, shape)
        } else if shape.input == self.config.hidden_size
            && shape.output == self.config.num_key_value_heads * self.config.head_dim
        {
            let h = self.config.hidden_size;
            let kv = shape.output;
            let weight = self.weights.layers[0].block.qkv.slice(h * h..(h + kv) * h);
            blas.calibrate_shape(&weight, input, shape)
        } else {
            ensure!(
                shape.rows == 1
                    && shape.input == self.config.markov_rank
                    && shape.output == self.config.vocab_size,
                "unknown draft calibration shape: {shape:?}"
            );
            blas.calibrate_shape(&self.weights.w2, input, shape)
        }
    }
    pub(super) fn reset_setup(&mut self) -> Result<()> {
        let block_rows = self.config.block_size;
        self.mask_ids.fill(self.config.mask_token_id);
        let s = &self.stream;
        let dim = self.config.head_dim;
        for kv in &mut self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = head * self.capacity * dim;
                s.memset_zeros(&mut kv.k.slice_mut(lo..lo + (block_rows + 1) * dim))?;
                s.memset_zeros(&mut kv.v.slice_mut(lo..lo + (block_rows + 1) * dim))?;
            }
        }
        for value in [
            &mut self.x,
            &mut self.n,
            &mut self.qkv,
            &mut self.attn,
            &mut self.out,
            &mut self.gu,
            &mut self.act,
            &mut self.k_raw,
            &mut self.k_norm,
            &mut self.v_raw,
            &mut self.projected,
            &mut self.context,
            &mut self.packed,
            &mut self.base,
            &mut self.scores,
            &mut self.prev,
        ] {
            s.memset_zeros(value)?;
        }
        for value in [&mut self.token, &mut self.tokens] {
            s.memset_zeros(value)?;
        }
        s.memset_zeros(&mut self.position)?;
        if let Some(combined) = &mut self.combined {
            s.memset_zeros(combined)?;
        }
        s.memset_zeros(&mut self.confidence_values)?;
        s.memset_zeros(&mut self.forced_tokens)?;
        s.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        Ok(())
    }
    pub fn new(
        path: &Path,
        d: &Device,
        c: &Config,
        capacity: usize,
        rows: usize,
        blas: &mut Blas,
        score_rounding: super::markov::ScoreRounding,
    ) -> Result<Self> {
        let config = DraftConfig::read(path, c)?;
        ensure!(
            rows > config.block_size && capacity >= rows,
            "DSpark workspace must fit block_size+1 rows within capacity"
        );
        let weights = DraftWeights::load(path, &config, d)?;
        let s = &d.stream;
        let h = config.hidden_size;
        let f = config.intermediate_size;
        let kv = config.kv_dim();
        let qkv = config.qkv_dim();
        let vocab = config.vocab_size;
        let rank = config.markov_rank;
        let block_rows = config.block_size;
        let mut caches = Vec::new();
        for _ in 0..config.num_hidden_layers {
            caches.push(Kv {
                k: s.alloc_zeros(capacity * kv)?,
                v: s.alloc_zeros(capacity * kv)?,
            });
        }
        for (n, o, i) in [
            (block_rows, qkv, h),
            (block_rows, h, h),
            (block_rows, 2 * f, h),
            (block_rows, h, f),
            (block_rows, vocab, h),
            (rows, h, config.capture_width()),
            (rows, kv, h),
        ] {
            blas.prepare(n, o, i)?;
        }
        eprintln!(
            "draft_weights_bytes={} draft_kv_bytes={}",
            weights.bytes,
            config.num_hidden_layers * capacity * kv * 4
        );
        let mask_ids = vec![config.mask_token_id; block_rows];
        let qknorm_rope_kv =
            super::cubin::module(&d.ctx, "rope")?.load_function("draft_qknorm_rope_kv")?;
        let norm_module = super::cubin::module(&d.ctx, "norm")?;
        let rope_module = super::cubin::module(&d.ctx, "rope")?;
        let (norm_name, add_norm_name, rope_name) = if config.head_dim == 256 {
            ("rms_norm_bf16", "add_rmsnorm_bf16", "rope")
        } else {
            ("rms_norm", "add_rmsnorm", "rope_float")
        };
        Ok(Self {
            markov: None,
            score_rounding,
            config,
            weights,
            kv: caches,
            capacity,
            rows,
            stream: s.clone(),
            attention: None,
            #[cfg(test)]
            position_shift: 0,
            qknorm_rope_kv,
            norm: norm_module.load_function(norm_name)?,
            add_norm: norm_module.load_function(add_norm_name)?,
            context_rope: rope_module.load_function(rope_name)?,
            ids: s.clone_htod(&mask_ids)?,
            mask_ids,
            position: s.alloc_zeros(1)?,
            x: s.alloc_zeros(block_rows * h)?,
            n: s.alloc_zeros(block_rows * h)?,
            qkv: s.alloc_zeros(block_rows * qkv)?,
            attn: s.alloc_zeros(block_rows * h)?,
            out: s.alloc_zeros(block_rows * h)?,
            gu: s.alloc_zeros(block_rows * 2 * f)?,
            act: s.alloc_zeros(block_rows * f)?,
            k_raw: s.alloc_zeros(rows * kv)?,
            k_norm: s.alloc_zeros(rows * kv)?,
            v_raw: s.alloc_zeros(rows * kv)?,
            projected: s.alloc_zeros(rows * h)?,
            context: s.alloc_zeros(rows * h)?,
            packed: s.alloc_zeros(rows * 2 * kv)?,
            base: s.alloc_zeros(block_rows * vocab)?,
            scores: s.alloc_zeros(block_rows * vocab)?,
            combined: if score_rounding == super::markov::ScoreRounding::F32 {
                Some(s.alloc_zeros(block_rows * vocab)?)
            } else {
                None
            },
            confidence_values: s.alloc_zeros(block_rows)?,
            forced_tokens: s.alloc_zeros(block_rows)?,
            sum_argmax: super::cubin::module(&d.ctx, "markov")?.load_function("sum_argmax")?,
            concat_dot_sigmoid: super::cubin::module(&d.ctx, "markov")?
                .load_function("concat_dot_sigmoid")?,
            prev: s.alloc_zeros(rank)?,
            token: s.alloc_zeros(1)?,
            tokens: s.alloc_zeros(block_rows)?,
        })
    }

    pub(super) fn grow(&mut self, rows: usize) -> Result<()> {
        ensure!(
            rows >= self.rows && rows <= self.capacity,
            "draft workspace rows outside cache bound"
        );
        let c = &self.config;
        let s = &self.stream;
        let k_raw = s.alloc_zeros(rows * c.kv_dim())?;
        let k_norm = s.alloc_zeros(rows * c.kv_dim())?;
        let v_raw = s.alloc_zeros(rows * c.kv_dim())?;
        let projected = s.alloc_zeros(rows * c.hidden_size)?;
        let context = s.alloc_zeros(rows * c.hidden_size)?;
        let packed = s.alloc_zeros(rows * 2 * c.kv_dim())?;
        self.k_raw = k_raw;
        self.k_norm = k_norm;
        self.v_raw = v_raw;
        self.projected = projected;
        self.context = context;
        self.packed = packed;
        self.rows = rows;
        Ok(())
    }

    pub fn inject(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.inject_block(input, rows, start, compute, trace)
    }

    pub fn inject_device(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        position: &CudaSlice<i32>,
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.inject_block(input, rows, start, compute, None)
    }

    fn inject_block(
        &mut self,
        input: &CudaSlice<bf16>,
        rows: usize,
        start: usize,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        let Compute { ops, blas } = compute;
        let c = &self.config;
        ensure!(
            rows > 0 && rows <= self.rows && start + rows <= self.capacity,
            "draft injection exceeds workspace/cache"
        );
        let s = &self.stream;
        let h = c.hidden_size;
        let kv = c.kv_dim();
        blas.prepare(rows, h, c.capture_width())?;
        blas.prepare(rows, kv, h)?;
        blas.linear(
            &self.weights.fc,
            &input.slice(..rows * c.capture_width()),
            &mut self.projected.slice_mut(..rows * h),
            rows,
            h,
            c.capture_width(),
        )?;
        norm(
            s,
            &self.norm,
            &self.projected,
            &self.weights.hidden_norm,
            &mut self.context,
            (rows, h),
            self.config.rms_norm_eps,
        )?;
        for (i, layer) in self.weights.layers.iter().enumerate() {
            let k_weight = layer.block.qkv.slice(h * h..(h + kv) * h);
            let v_weight = layer.block.qkv.slice((h + kv) * h..(h + 2 * kv) * h);
            blas.linear(
                &k_weight,
                &self.context.slice(..rows * h),
                &mut self.k_raw.slice_mut(..rows * kv),
                rows,
                kv,
                h,
            )?;
            blas.linear(
                &v_weight,
                &self.context.slice(..rows * h),
                &mut self.v_raw.slice_mut(..rows * kv),
                rows,
                kv,
                h,
            )?;
            norm(
                s,
                &self.norm,
                &self.k_raw,
                &layer.k_norm,
                &mut self.k_norm,
                (rows * c.num_key_value_heads, c.head_dim),
                self.config.rms_norm_eps,
            )?;
            unsafe {
                s.launch_builder(&self.context_rope)
                    .arg(&mut self.k_norm)
                    .arg(&self.position)
                    .arg(&(rows as i32))
                    .arg(&(c.num_key_value_heads as i32))
                    .arg(&0i32)
                    .arg(&(c.head_dim as i32))
                    .arg(&c.rope_parameters.rope_theta)
                    .launch(flat(rows * kv / 2))?;
            }
            columns(
                s,
                &self.k_norm,
                &mut self.packed,
                Rect {
                    rows,
                    width: kv,
                    src_pitch: kv,
                    src_col: 0,
                    dst_pitch: 2 * kv,
                    dst_col: 0,
                },
            )?;
            columns(
                s,
                &self.v_raw,
                &mut self.packed,
                Rect {
                    rows,
                    width: kv,
                    src_pitch: kv,
                    src_col: 0,
                    dst_pitch: 2 * kv,
                    dst_col: kv,
                },
            )?;
            let cache = &mut self.kv[i];
            unsafe {
                s.launch_builder(&ops.kv_write)
                    .arg(&self.packed)
                    .arg(&mut cache.k)
                    .arg(&mut cache.v)
                    .arg(&self.position)
                    .arg(&(rows as i32))
                    .arg(&0i32)
                    .arg(&(c.num_key_value_heads as i32))
                    .arg(&(c.head_dim as i32))
                    .arg(&(self.capacity as i32))
                    .launch(flat(rows * kv))?;
            }
        }
        if let Some((trace, prefix)) = trace {
            add_trace(
                s,
                trace,
                format!("{prefix}.c_projected"),
                &self.projected,
                rows,
                h,
            )?;
            add_trace(
                s,
                trace,
                format!("{prefix}.c_committed"),
                &self.context,
                rows,
                h,
            )?;
        }
        Ok(())
    }
}

fn add_trace(
    s: &Arc<CudaStream>,
    trace: &mut Trace,
    name: String,
    data: &CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    let values = s.clone_dtoh(&data.slice(..rows * width))?;
    trace.add(
        name,
        vec![rows, width],
        values.into_iter().map(bf16::to_f32).collect(),
    )
}

impl Draft {
    pub fn setup_attention(
        &mut self,
        d: &Device,
        ops: &Ops,
        c: &Config,
        dir: &Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        if self.attention.is_some() {
            return Ok(());
        }
        if self.config.head_dim == 256 {
            self.attention = Some(Attention::Block(BlockAttention::new(self, d)?));
            return Ok(());
        }
        let block_rows = self.config.block_size;
        let mut geometry = c.clone();
        geometry.num_attention_heads = self.config.num_attention_heads;
        geometry.num_key_value_heads = self.config.num_key_value_heads;
        geometry.head_dim = self.config.head_dim;
        geometry.rope_theta = self.config.rope_parameters.rope_theta;
        geometry.rms_norm_eps = self.config.rms_norm_eps;
        let c = &geometry;
        let measured = super::multi_calibrate::load(
            d,
            ops,
            c,
            dir,
            crate::backend::setup::MultiShape {
                rows: block_rows,
                causal: false,
                capacity: self.capacity,
            },
            closure,
        )?;
        let (plan, selection) = measured.choose(self.capacity)?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"role":"draft","rows":block_rows,"causal":false,"attention":plan,"selection":selection,"calibration":measured}})
        );
        self.attention = Some(Attention::Cached(Verification::with_plan(
            &d.ctx,
            &d.stream,
            c,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows: block_rows,
                causal: false,
            },
            plan,
        )?));
        Ok(())
    }
    pub fn propose(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        self.mask_ids[0] = anchor;
        self.stream.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.block(start, target, compute, trace, forced, Output::Tokens)
    }

    pub fn propose_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream
            .memcpy_dtod(anchor, &mut self.ids.slice_mut(..1))?;
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Device)?;
        Ok(())
    }

    pub fn forward_base(
        &mut self,
        anchor: u32,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.mask_ids[0] = anchor;
        self.stream.memcpy_htod(&self.mask_ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Base)?;
        Ok(())
    }

    pub fn forward_base_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
    ) -> Result<()> {
        self.stream
            .memcpy_dtod(anchor, &mut self.ids.slice_mut(..1))?;
        self.stream.memcpy_dtod(position, &mut self.position)?;
        self.block(start, target, compute, None, None, Output::Base)?;
        Ok(())
    }

    pub fn setup_markov(
        &mut self,
        d: &Device,
        ops: &super::ops::Ops,
        dir: &std::path::Path,
        closure: &super::closure::Closure,
    ) -> Result<()> {
        if self.markov.is_none() {
            self.markov = Some(super::markov::Markov::new(
                d,
                ops,
                dir,
                closure,
                self.markov_shape(),
            )?);
        }
        Ok(())
    }

    pub(super) fn markov_shape(&self) -> super::markov::MarkovShape {
        super::markov::MarkovShape {
            rows: self.config.block_size,
            vocab: self.config.vocab_size,
            rank: self.config.markov_rank,
            rounding: self.score_rounding,
        }
    }

    pub fn distributions_batch(
        &mut self,
        requests: &[(u8, u32)],
    ) -> Result<Vec<crate::backend::Top4>> {
        self.markov
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Markov module is not initialized"))?
            .distributions_batch(&self.base, &self.weights.w1, &self.weights.w2, requests)
    }

    pub fn proposals(&self) -> Result<Vec<u32>> {
        Ok(self.stream.clone_dtoh(&self.tokens)?)
    }

    pub fn copy_proposals(&self, ids: &mut cudarc::driver::CudaViewMut<'_, u32>) -> Result<()> {
        self.stream.memcpy_dtod(&self.tokens, ids)?;
        Ok(())
    }

    pub fn state_bits(&self, start: usize, rows: usize, out: &mut Vec<u8>) -> Result<()> {
        ensure!(
            rows > 0 && start + rows <= self.capacity,
            "draft bit comparison range exceeds KV"
        );
        let s = &self.stream;
        if let Some(Attention::Block(attention)) = &self.attention {
            for values in [&attention.metadata, &attention.positions] {
                out.extend(s.clone_dtoh(values)?.iter().flat_map(|v| v.to_le_bytes()));
            }
            out.extend(
                s.clone_dtoh(&attention.mask)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes()),
            );
        }
        for values in [&self.n, &self.base, &self.scores] {
            super::copy::bits(s, values, out)?;
        }
        if let Some(combined) = &self.combined {
            out.extend(
                s.clone_dtoh(combined)?
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes()),
            );
        }
        out.extend(
            s.clone_dtoh(&self.confidence_values)?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes()),
        );
        let d = self.config.head_dim;
        for kv in &self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + start) * d;
                let hi = lo + rows * d;
                super::copy::bits(s, &kv.k.slice(lo..hi), out)?;
                super::copy::bits(s, &kv.v.slice(lo..hi), out)?;
            }
        }
        Ok(())
    }

    fn block(
        &mut self,
        start: usize,
        target: (&CudaSlice<bf16>, &CudaSlice<bf16>),
        compute: Compute<'_>,
        mut trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
        output: Output,
    ) -> Result<Vec<u32>> {
        if let Some(tokens) = forced {
            ensure!(
                tokens.len() == self.config.block_size
                    && tokens
                        .iter()
                        .all(|&v| (v as usize) < self.config.vocab_size),
                "invalid teacher-forced Markov tokens"
            );
        }
        let Compute { ops, blas } = compute;
        let c = &self.config;
        let block_rows = c.block_size;
        ensure!(
            start + block_rows <= self.capacity,
            "draft block exceeds KV capacity"
        );
        let s = &self.stream;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let qkv = c.qkv_dim();
        let eps = self.config.rms_norm_eps;
        let add_norm_kernel = &self.add_norm;
        let add_norm = |res: &CudaSlice<bf16>,
                        x: &mut CudaSlice<bf16>,
                        weight: &CudaSlice<bf16>,
                        output: &mut CudaSlice<bf16>|
         -> Result<()> {
            let (xp, _x) = x.device_ptr_mut(s);
            unsafe {
                s.launch_builder(add_norm_kernel)
                    .arg(res)
                    .arg(&xp)
                    .arg(weight)
                    .arg(output)
                    .arg(&xp)
                    .arg(&(block_rows as i32))
                    .arg(&(h as i32))
                    .arg(&eps)
                    .launch(grid(block_rows, 256))?;
            }
            Ok(())
        };
        unsafe {
            s.launch_builder(&ops.embed)
                .arg(target.0)
                .arg(&self.ids)
                .arg(&mut self.x)
                .arg(&(block_rows as i32))
                .arg(&(h as i32))
                .launch(flat(block_rows * h))?;
        }
        if let Some(Attention::Block(attention)) = &mut self.attention {
            let shift = {
                #[cfg(test)]
                {
                    self.position_shift
                }
                #[cfg(not(test))]
                {
                    0i32
                }
            };
            unsafe {
                s.launch_builder(&attention.prepare)
                    .arg(&self.position)
                    .arg(&mut attention.metadata)
                    .arg(&mut attention.positions)
                    .arg(&(block_rows as i32))
                    .arg(&(self.capacity as i32))
                    .arg(&shift)
                    .launch(grid(1, 256))?;
            }
        }
        for (i, layer) in self.weights.layers.iter().enumerate() {
            if i == 0 {
                norm(
                    s,
                    &self.norm,
                    &self.x,
                    &layer.block.input_norm,
                    &mut self.n,
                    (block_rows, h),
                    self.config.rms_norm_eps,
                )?;
            }
            blas.linear(
                &layer.block.qkv,
                &self.n.slice(..block_rows * h),
                &mut self.qkv.slice_mut(..block_rows * qkv),
                block_rows,
                qkv,
                h,
            )?;
            match self
                .attention
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("draft attention has not been initialized"))?
            {
                Attention::Cached(attention) => {
                    let cache = &mut self.kv[i];
                    unsafe {
                        s.launch_builder(&self.qknorm_rope_kv)
                            .arg(&mut self.qkv)
                            .arg(&cache.k)
                            .arg(&cache.v)
                            .arg(&layer.q_norm)
                            .arg(&layer.k_norm)
                            .arg(&self.position)
                            .arg(&(block_rows as i32))
                            .arg(&(c.num_attention_heads as i32))
                            .arg(&(c.num_key_value_heads as i32))
                            .arg(&(c.head_dim as i32))
                            .arg(&(self.capacity as i32))
                            .arg(&c.rope_parameters.rope_theta)
                            .arg(&self.config.rms_norm_eps)
                            .launch(grid(
                                block_rows * (c.num_attention_heads + 2 * c.num_key_value_heads),
                                256,
                            ))?;
                    }
                    attention.run(
                        &self.qkv,
                        &self.kv[i].k,
                        &self.kv[i].v,
                        &mut self.attn,
                        &self.position,
                        Shape {
                            rows: block_rows,
                            start,
                            q_heads: c.num_attention_heads,
                            kv_heads: c.num_key_value_heads,
                            capacity: self.capacity,
                            head_dim: c.head_dim,
                            causal: false,
                        },
                    )?;
                }
                Attention::Block(attention) => {
                    let (metadata, _meta_hold) = attention.metadata.device_ptr(s);
                    let (scratch, _scratch_hold) = attention.scratch.device_ptr(s);
                    FullAttentionOps::run(
                        &attention.layers[i],
                        s,
                        SeqRef { ptr: metadata },
                        Scratch {
                            ptr: scratch,
                            bytes: attention.scratch.len() as u64,
                        },
                    )?;
                }
            }
            blas.linear(
                &layer.block.o,
                &self.attn.slice(..block_rows * h),
                &mut self.out.slice_mut(..block_rows * h),
                block_rows,
                h,
                h,
            )?;
            add_norm(&self.out, &mut self.x, &layer.block.post_norm, &mut self.n)?;
            blas.linear(
                &layer.block.gu,
                &self.n.slice(..block_rows * h),
                &mut self.gu.slice_mut(..block_rows * 2 * f),
                block_rows,
                2 * f,
                h,
            )?;
            unsafe {
                s.launch_builder(&ops.act)
                    .arg(&self.gu)
                    .arg(&mut self.act)
                    .arg(&(block_rows as i32))
                    .arg(&(f as i32))
                    .launch(flat(block_rows * f))?;
            }
            blas.linear(
                &layer.block.down,
                &self.act.slice(..block_rows * f),
                &mut self.out.slice_mut(..block_rows * h),
                block_rows,
                h,
                f,
            )?;
            let weight = if i + 1 < self.weights.layers.len() {
                &self.weights.layers[i + 1].block.input_norm
            } else {
                &self.weights.norm
            };
            add_norm(&self.out, &mut self.x, weight, &mut self.n)?;
            if let Some((trace, prefix)) = trace.as_mut() {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.layer.{i}"),
                    &self.x,
                    block_rows,
                    h,
                )?;
            }
        }
        blas.linear(
            target.1,
            &self.n.slice(..block_rows * h),
            &mut self.base.slice_mut(..block_rows * c.vocab_size),
            block_rows,
            c.vocab_size,
            h,
        )?;
        if matches!(output, Output::Base) {
            return Ok(Vec::new());
        }
        s.memcpy_dtod(&self.ids.slice(..1), &mut self.token)?;
        if let Some(tokens) = forced {
            s.memcpy_htod(tokens, &mut self.forced_tokens)?;
        }
        for row in 0..block_rows {
            unsafe {
                s.launch_builder(&ops.embed)
                    .arg(&self.weights.w1)
                    .arg(&self.token)
                    .arg(&mut self.prev)
                    .arg(&1i32)
                    .arg(&(self.config.markov_rank as i32))
                    .launch(flat(self.config.markov_rank))?;
            }
            blas.linear(
                &self.weights.w2,
                &self.prev.slice(..self.config.markov_rank),
                &mut self
                    .scores
                    .slice_mut(row * c.vocab_size..(row + 1) * c.vocab_size),
                1,
                c.vocab_size,
                self.config.markov_rank,
            )?;
            let (bias, _bias_hold) = self.scores.device_ptr_mut(s);
            let (combined, _combined_hold) = if let Some(values) = &mut self.combined {
                let (ptr, hold) = values.device_ptr_mut(s);
                (Some(ptr), Some(hold))
            } else {
                (None, None)
            };
            let bias_row = bias + (row * c.vocab_size * 2) as u64;
            let score_row = combined.map_or(bias_row, |ptr| ptr + (row * c.vocab_size * 4) as u64);
            unsafe {
                s.launch_builder(&self.sum_argmax)
                    .arg(
                        &self
                            .base
                            .slice(row * c.vocab_size..(row + 1) * c.vocab_size),
                    )
                    .arg(&bias_row)
                    .arg(&score_row)
                    .arg(&mut self.token)
                    .arg(&(c.vocab_size as i32))
                    .arg(&(self.score_rounding as i32))
                    .launch(grid(1, 1024))?;
            }
            s.memcpy_dtod(&self.token, &mut self.tokens.slice_mut(row..row + 1))?;
            if forced.is_some() {
                s.memcpy_dtod(&self.forced_tokens.slice(row..row + 1), &mut self.token)?;
            }
        }
        unsafe {
            s.launch_builder(&self.concat_dot_sigmoid)
                .arg(&self.n)
                .arg(&self.weights.w1)
                .arg(&self.ids)
                .arg(if forced.is_some() {
                    &self.forced_tokens
                } else {
                    &self.tokens
                })
                .arg(&self.weights.confidence)
                .arg(&self.weights.confidence_bias)
                .arg(&mut self.confidence_values)
                .arg(&(block_rows as i32))
                .arg(&(h as i32))
                .arg(&(c.markov_rank as i32))
                .arg(&(c.vocab_size as i32))
                .launch(grid(block_rows, 256))?;
        }
        let proposals = if matches!(output, Output::Tokens) {
            s.clone_dtoh(&self.tokens)?
        } else {
            Vec::new()
        };
        if let Some((trace, prefix)) = trace.as_mut() {
            add_trace(s, trace, format!("{prefix}.H"), &self.n, block_rows, h)?;
            add_trace(
                s,
                trace,
                format!("{prefix}.B"),
                &self.base,
                block_rows,
                c.vocab_size,
            )?;
            if let Some(combined) = &self.combined {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.M"),
                    &self.scores,
                    block_rows,
                    c.vocab_size,
                )?;
                trace.add(
                    format!("{prefix}.s"),
                    vec![block_rows, c.vocab_size],
                    s.clone_dtoh(combined)?,
                )?;
            } else {
                add_trace(
                    s,
                    trace,
                    format!("{prefix}.s"),
                    &self.scores,
                    block_rows,
                    c.vocab_size,
                )?;
            }
            trace.add(
                format!("{prefix}.conf"),
                vec![block_rows],
                s.clone_dtoh(&self.confidence_values)?,
            )?;
            trace.add(
                format!("{prefix}.proposed"),
                vec![block_rows],
                proposals.iter().map(|&v| v as f32).collect(),
            )?;
        }
        Ok(proposals)
    }

    pub fn snapshot_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        let c = &self.config;
        ensure!(
            start + rows <= self.capacity,
            "draft KV snapshot exceeds capacity"
        );
        for (i, kv) in self.kv.iter().enumerate() {
            for (name, values) in [("k", &kv.k), ("v", &kv.v)] {
                let mut output = vec![0f32; rows * c.kv_dim()];
                for head in 0..c.num_key_value_heads {
                    let values = self.stream.clone_dtoh(&values.slice(
                        (head * self.capacity + start) * c.head_dim
                            ..(head * self.capacity + start + rows) * c.head_dim,
                    ))?;
                    for row in 0..rows {
                        for dim in 0..c.head_dim {
                            output[row * c.kv_dim() + head * c.head_dim + dim] =
                                values[row * c.head_dim + dim].to_f32();
                        }
                    }
                }
                trace.add(
                    format!("kv_{name}_{i}"),
                    vec![rows, c.num_key_value_heads, c.head_dim],
                    output,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use safetensors::{Dtype, SafeTensors};
    use std::{collections::BTreeSet, path::PathBuf, rc::Rc};

    fn values(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(tensor.shape() == shape, "reference {name} shape mismatch");
        let result: Vec<_> = match tensor.dtype() {
            Dtype::BF16 => tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect(),
            Dtype::F32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            _ => anyhow::bail!("reference {name} is not BF16/FP32"),
        };
        ensure!(
            result.iter().all(|v| v.is_finite()),
            "nonfinite reference {name}"
        );
        Ok(result)
    }

    fn ids(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<u32>> {
        let tensor = tensors.tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::I64 && tensor.shape() == shape,
            "reference {name} id layout mismatch"
        );
        tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i64::from_le_bytes(*v))?))
            .collect()
    }

    fn native(
        device: &Device,
        path: &Path,
        name: &str,
        shape: &[usize],
    ) -> Result<CudaSlice<bf16>> {
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let views = mapped.views()?;
        let shard = mapped
            .index
            .weight_map
            .get(name)
            .context("native tensor absent")?;
        let tensor = views[shard].tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::BF16 && tensor.shape() == shape,
            "native {name} layout mismatch"
        );
        let data: Vec<_> = tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        ensure!(
            data.iter().all(|v| v.to_f32().is_finite()),
            "nonfinite native {name}"
        );
        Ok(device.stream.clone_htod(&data)?)
    }

    fn coefficient(config: &DraftConfig, context: usize, layers: usize, extra: u32) -> f64 {
        let h = config.hidden_size;
        let d = config.head_dim;
        let keys = context + config.block_size;
        let mut reductions = vec![config.capture_width(), h];
        for _ in 0..layers {
            reductions.extend([h, h, d, d, keys, keys, h, h, h, config.intermediate_size]);
        }
        // Per layer: input norm2, QKV1, Q/K norm2, RoPE3, probability1, PV1, O1, add1,
        // post norm2, gate/up1, SiLU1, product1, down1, add1.
        let rounds = 3 + 19 * layers as u32 + extra;
        let chain = super::super::bf16_chain_bound(rounds, &reductions);
        let norm = super::super::bf16_chain_bound(2 * layers as u32, &[h, d]);
        let silu = super::super::bf16_chain_bound(2 * layers as u32, &[h]) / std::f64::consts::E;
        let softmax = super::super::gamma(keys, 2f64.powi(-24)) * 0.5;
        chain + norm + silu + softmax
    }

    fn compare(
        name: &str,
        actual: &[f32],
        expected: (&[f32], &[f32]),
        coefficient: f64,
        width: usize,
    ) -> Result<usize> {
        let (reference, floor) = expected;
        ensure!(
            actual.len() == reference.len()
                && floor.len() == reference.len()
                && width > 0
                && reference.len().is_multiple_of(width),
            "{name} extent mismatch"
        );
        ensure!(
            actual.iter().all(|v| v.is_finite())
                && floor.iter().all(|v| v.is_finite() && *v >= 0.0),
            "nonfinite {name}"
        );
        let mut failures = 0;
        let mut maximum = 0f64;
        let mut error = 0f64;
        for ((actual, reference), floor) in actual
            .chunks_exact(width)
            .zip(reference.chunks_exact(width))
            .zip(floor.chunks_exact(width))
        {
            let rms = (reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64)
                .sqrt();
            for ((a, b), f) in actual.iter().zip(reference).zip(floor) {
                let delta = (f64::from(*a) - f64::from(*b)).abs();
                let bound = coefficient * f64::from(*b).abs().max(rms) + f64::from(*f);
                failures += usize::from(delta > bound);
                maximum = maximum.max(if bound == 0.0 {
                    if delta == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    delta / bound
                });
                error = error.max(delta);
            }
        }
        println!(
            "T19_TENSOR name={name} max_abs={error} max_bound_ratio={maximum} coefficient={coefficient} violations={failures}"
        );
        Ok(failures)
    }

    fn calibrate(draft: &Draft, head: &CudaSlice<bf16>, blas: &mut Blas) -> Result<()> {
        use crate::backend::setup::LinearShape;
        let c = &draft.config;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let shapes = [
            (
                &draft.weights.layers[0].block.qkv,
                c.block_size,
                c.qkv_dim(),
                h,
            ),
            (&draft.weights.layers[0].block.o, c.block_size, h, h),
            (&draft.weights.layers[0].block.gu, c.block_size, 2 * f, h),
            (&draft.weights.layers[0].block.down, c.block_size, h, f),
            (head, c.block_size, c.vocab_size, h),
            (&draft.weights.w2, 1, c.vocab_size, c.markov_rank),
        ];
        let pairs: BTreeSet<_> = shapes
            .iter()
            .map(|(_, _, output, input)| (*output, *input))
            .chain([(h, c.capture_width()), (c.kv_dim(), h)])
            .collect();
        blas.declare(draft.rows, &pairs);
        let count = shapes
            .iter()
            .map(|(_, r, _, k)| r * k)
            .max()
            .context("no draft shapes")?;
        let input: Vec<_> = (0..count)
            .map(|i| bf16::from_f32(((i % 257) as f32 - 128.0) / 128.0))
            .collect();
        let input = draft.stream.clone_htod(&input)?;
        for (weight, rows, output, input_width) in shapes {
            let shape = LinearShape {
                rows,
                output,
                input: input_width,
            };
            if super::super::linear_calibrate::supports(shape) {
                blas.calibrate_shape(weight, &input.slice(..rows * input_width), shape)?;
            }
        }
        Ok(())
    }

    fn confidence_bounds(
        c: &DraftConfig,
        path: &Path,
        reference: (&[f32], &[f32]),
        parents: &[u32],
        propagated: f64,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (hidden, floor) = reference;
        ensure!(
            hidden.len() == c.block_size * c.hidden_size
                && floor.len() == c.block_size
                && parents.len() == c.block_size,
            "confidence reference extent mismatch"
        );
        ensure!(
            parents.iter().all(|p| (*p as usize) < c.vocab_size),
            "confidence parent outside vocabulary"
        );
        let mapped = crate::checkpoint::MappedCheckpoint::open(path)?;
        let views = mapped.views()?;
        let table = &views[mapped
            .index
            .weight_map
            .get("confidence_head.proj.weight")
            .context("confidence shard absent")?];
        let weight = values(
            table,
            "confidence_head.proj.weight",
            &[1, c.hidden_size + c.markov_rank],
        )?;
        let bias = values(table, "confidence_head.proj.bias", &[1])?[0].abs();
        let embedding_table = &views[mapped
            .index
            .weight_map
            .get("markov_head.markov_w1.weight")
            .context("embedding shard absent")?];
        let embedding_tensor = embedding_table.tensor("markov_head.markov_w1.weight")?;
        ensure!(
            embedding_tensor.dtype() == Dtype::BF16
                && embedding_tensor.shape() == [c.vocab_size, c.markov_rank],
            "confidence embedding layout mismatch"
        );
        let local = super::super::bf16_chain_bound(1, &[c.hidden_size + c.markov_rank]);
        let mut complete = Vec::new();
        let mut isolated = Vec::new();
        for (row, &parent) in parents.iter().enumerate() {
            let begin = parent as usize * c.markov_rank;
            let embedding: Vec<_> = embedding_tensor.data()[begin * 2..(begin + c.markov_rank) * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect();
            let h_product: f64 = hidden[row * c.hidden_size..(row + 1) * c.hidden_size]
                .iter()
                .zip(&weight[..c.hidden_size])
                .map(|(v, w)| f64::from(*v).abs() * f64::from(*w).abs())
                .sum();
            let m_product: f64 = embedding
                .iter()
                .zip(&weight[c.hidden_size..])
                .map(|(v, w)| f64::from(*v).abs() * f64::from(*w).abs())
                .sum();
            let budget =
                0.25 * local * (h_product + m_product + f64::from(bias)) + f64::from(floor[row]);
            isolated.push(budget as f32);
            complete.push((budget + 0.25 * propagated * h_product) as f32);
        }
        Ok((complete, isolated))
    }

    #[test]
    #[ignore = "requires a GPU window and native target/draft, T14 fixtures and frozen reference floors"]
    fn fixture_forward_gpu() -> Result<()> {
        ensure!(
            std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
            "GPU window not acknowledged"
        );
        let target = PathBuf::from(std::env::var("FF_ARGMAX_MODEL")?);
        let draft_path = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
        let reference = PathBuf::from(std::env::var("FF_MARKOV_FIXTURES")?);
        let floor_manifest = PathBuf::from(std::env::var("FF_MINICPM_TOLERANCES")?);
        let floor_root = floor_manifest
            .parent()
            .context("floor manifest parent missing")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&floor_manifest)?)?;
        ensure!(
            manifest["candidate_executed"] == false,
            "reference floors were not frozen before the candidate"
        );
        let config = crate::model_spec::ModelSpec::read(&target)?.config();
        let c = DraftConfig::read(&draft_path, &config)?;
        ensure!(
            c.block_size == 15 && c.hidden_size == 2048 && c.head_dim == 256,
            "T14 native draft geometry required"
        );
        let names = [
            "math-block-0",
            "math-block-1",
            "code-block-0",
            "code-block-1",
            "general-block-0",
            "general-block-1",
        ];
        let mut context_max = 0;
        let mut confidence_limits = std::collections::BTreeMap::new();
        for name in names {
            let data = std::fs::read(reference.join(format!("{name}.safetensors")))?;
            let tensors = SafeTensors::deserialize(&data)?;
            let shape = tensors.tensor("target_taps")?.shape().to_vec();
            ensure!(
                shape.len() == 4
                    && shape[0] == 1
                    && shape[2] == c.target_layer_ids.len()
                    && shape[3] == c.hidden_size,
                "T14 tap layout mismatch"
            );
            context_max = context_max.max(shape[1]);
            for depth in 0..=c.num_hidden_layers {
                println!(
                    "T19_FROZEN_BOUND fixture={name} layers={depth} coefficient={} context={} block={}",
                    coefficient(&c, shape[1], depth, 0),
                    shape[1],
                    c.block_size
                );
            }
            let floor_data = std::fs::read(floor_root.join(format!("{name}-floor.safetensors")))?;
            let floor = SafeTensors::deserialize(&floor_data)?;
            let b = c.block_size;
            let h = c.hidden_size;
            let block_ids = ids(&tensors, "draft_input_ids", &[1, b])?;
            let proposals = ids(&tensors, "proposals_tensor", &[b])?;
            let parents: Vec<_> = std::iter::once(block_ids[0])
                .chain(proposals.into_iter().take(b - 1))
                .collect();
            let limits = confidence_bounds(
                &c,
                &draft_path,
                (
                    &values(&tensors, "draft_norm_hidden", &[1, b, h])?,
                    &values(&floor, "confidence_c_k", &[b])?,
                ),
                &parents,
                coefficient(&c, shape[1], c.num_hidden_layers, 2),
            )?;
            println!(
                "T19_CONFIDENCE_BOUND fixture={name} sigmoid_lipschitz=0.25 full_max={} isolated_max={}",
                limits.0.iter().copied().fold(0.0f32, f32::max),
                limits.1.iter().copied().fold(0.0f32, f32::max)
            );
            confidence_limits.insert(name, limits);
        }
        let capacity = context_max
            .checked_add(c.block_size)
            .context("draft fixture capacity overflow")?
            .next_multiple_of(32);
        let rows = context_max.max(c.block_size + 1);
        let runtime = PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?);
        std::fs::create_dir_all(&runtime)?;
        let device = Device::new(0)?;
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::closure::Closure::new(
            true,
            "T19 frozen native fixture calibration".into(),
        ));
        let mut blas = Blas::new(&device, &ops, &runtime, closure.clone())?;
        let mut draft = Draft::new(
            &draft_path,
            &device,
            &config,
            capacity,
            rows,
            &mut blas,
            super::super::markov::ScoreRounding::F32,
        )?;
        let embedding = native(
            &device,
            &target,
            "model.language_model.embed_tokens.weight",
            &[c.vocab_size, c.hidden_size],
        )?;
        let head = native(
            &device,
            &target,
            "lm_head.weight",
            &[c.vocab_size, c.hidden_size],
        )?;
        calibrate(&draft, &head, &mut blas)?;
        draft.setup_attention(&device, &ops, &config, &runtime, &closure)?;
        draft.setup_markov(&device, &ops, &runtime, &closure)?;
        let s = &device.stream;
        let mut defects = 0;
        let mut rejected_control = 0;
        let mut graph_replays = 0;
        let mut native_requests = 0;
        for name in names {
            let data = std::fs::read(reference.join(format!("{name}.safetensors")))?;
            let tensors = SafeTensors::deserialize(&data)?;
            let floor_data = std::fs::read(floor_root.join(format!("{name}-floor.safetensors")))?;
            let floors = SafeTensors::deserialize(&floor_data)?;
            let context = tensors.tensor("target_taps")?.shape()[1];
            let h = c.hidden_size;
            let b = c.block_size;
            let v = c.vocab_size;
            let block_ids = ids(&tensors, "draft_input_ids", &[1, b])?;
            let positions = ids(&tensors, "draft_position_ids", &[1, b])?;
            let ctx_positions = ids(&tensors, "ctx_position_ids", &[1, context])?;
            ensure!(
                positions
                    .iter()
                    .enumerate()
                    .all(|(i, p)| *p as usize == context + i)
                    && ctx_positions
                        .iter()
                        .enumerate()
                        .all(|(i, p)| *p as usize == i),
                "T14 frozen absolute positions changed"
            );
            ensure!(
                block_ids[1..].iter().all(|id| *id == c.mask_token_id),
                "T14 mask layout changed"
            );
            let expected_ids = ids(&tensors, "proposals_tensor", &[b])?;
            let expected_hidden = values(&tensors, "draft_norm_hidden", &[1, b, h])?;
            let expected_confidence = values(&tensors, "confidence_c_k", &[b])?;
            let final_coefficient = coefficient(&c, context, c.num_hidden_layers, 2);
            let (confidence_budget, isolated_budget) = confidence_limits
                .get(name)
                .context("confidence budget absent")?;
            let taps: Vec<_> = values(&tensors, "target_taps", &[1, context, 5, h])?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            let taps = s.clone_htod(&taps)?;
            draft.position_shift = 0;
            draft.reset_setup()?;
            let poison = vec![bf16::from_f32(-17.0); capacity * c.kv_dim()];
            for cache in &mut draft.kv {
                s.memcpy_htod(&poison, &mut cache.k)?;
                s.memcpy_htod(&poison, &mut cache.v)?;
            }
            let mut trace = Trace::default();
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut trace, "context")),
            )?;
            let proposals = draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut trace, "draft")),
                Some(&expected_ids),
            )?;
            let mut fixture_defects = usize::from(proposals != expected_ids);
            for (point, key, coef) in [
                (
                    "context.c_projected",
                    "h_ctx_pre_norm",
                    super::super::bf16_chain_bound(1, &[c.capture_width()]),
                ),
                (
                    "context.c_committed",
                    "h_ctx",
                    coefficient(&c, context, 0, 0),
                ),
            ] {
                fixture_defects += compare(
                    &format!("{name}.{point}"),
                    &trace.values(point)?,
                    (
                        &values(&tensors, key, &[1, context, h])?,
                        &values(&floors, key, &[1, context, h])?,
                    ),
                    coef,
                    h,
                )?;
            }
            for layer in 0..c.num_hidden_layers {
                let key = format!("draft_layer_{layer}");
                fixture_defects += compare(
                    &format!("{name}.{key}"),
                    &trace.values(&format!("draft.layer.{layer}"))?,
                    (
                        &values(&tensors, &key, &[1, b, h])?,
                        &values(&floors, &key, &[1, b, h])?,
                    ),
                    coefficient(&c, context, layer + 1, 0),
                    h,
                )?;
            }
            fixture_defects += compare(
                &format!("{name}.final_norm"),
                &trace.values("draft.H")?,
                (
                    &expected_hidden,
                    &values(&floors, "draft_norm_hidden", &[1, b, h])?,
                ),
                final_coefficient,
                h,
            )?;
            for row in [0, b - 1] {
                for (point, key, coef) in [
                    (
                        "draft.B",
                        format!("u_logits_k{row}"),
                        coefficient(&c, context, c.num_hidden_layers, 3),
                    ),
                    (
                        "draft.M",
                        format!("b_logits_k{row}"),
                        super::super::bf16_chain_bound(1, &[c.markov_rank]),
                    ),
                    (
                        "draft.s",
                        format!("combined_logits_k{row}"),
                        coefficient(&c, context, c.num_hidden_layers, 3)
                            + super::super::bf16_chain_bound(1, &[c.markov_rank])
                            + super::super::gamma(1, 2f64.powi(-24)),
                    ),
                ] {
                    let actual = trace.values(point)?;
                    fixture_defects += compare(
                        &format!("{name}.{key}"),
                        &actual[row * v..(row + 1) * v],
                        (
                            &values(&tensors, &key, &[v])?,
                            &values(&floors, &key, &[v])?,
                        ),
                        coef,
                        v,
                    )?;
                }
            }
            let confidence = trace.values("draft.conf")?;
            ensure!(
                confidence.iter().all(|p| (0.0..=1.0).contains(p)),
                "nonfinite/out-of-range confidence"
            );
            fixture_defects += compare(
                &format!("{name}.confidence"),
                &confidence,
                (&expected_confidence, confidence_budget),
                0.0,
                b,
            )?;
            let supplied: Vec<_> = expected_hidden
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect();
            s.memcpy_htod(&supplied, &mut draft.n)?;
            unsafe {
                s.launch_builder(&draft.concat_dot_sigmoid)
                    .arg(&draft.n)
                    .arg(&draft.weights.w1)
                    .arg(&draft.ids)
                    .arg(&draft.forced_tokens)
                    .arg(&draft.weights.confidence)
                    .arg(&draft.weights.confidence_bias)
                    .arg(&mut draft.confidence_values)
                    .arg(&(b as i32))
                    .arg(&(h as i32))
                    .arg(&(c.markov_rank as i32))
                    .arg(&(v as i32))
                    .launch(grid(b, 256))?;
            }
            fixture_defects += compare(
                &format!("{name}.confidence_isolated"),
                &s.clone_dtoh(&draft.confidence_values)?,
                (&expected_confidence, isolated_budget),
                0.0,
                b,
            )?;
            let top = values(&tensors, "combined_top32_vals", &[b, 32])?;
            let top_ids = ids(&tensors, "combined_top32_ids", &[b, 32])?;
            let scores = trace.values("draft.s")?;
            let mut actual_top = Vec::with_capacity(b * 32);
            for row in scores.chunks_exact(v) {
                let mut sorted = row.to_vec();
                sorted.select_nth_unstable_by(31, |a, b| b.total_cmp(a));
                sorted[..32].sort_unstable_by(|a, b| b.total_cmp(a));
                actual_top.extend_from_slice(&sorted[..32]);
            }
            fixture_defects += compare(
                &format!("{name}.combined_top32_vals"),
                &actual_top,
                (&top, &values(&floors, "combined_top32_vals", &[b, 32])?),
                coefficient(&c, context, c.num_hidden_layers, 3)
                    + super::super::bf16_chain_bound(1, &[c.markov_rank])
                    + super::super::gamma(1, 2f64.powi(-24)),
                32,
            )?;
            for (row, (actual, expected)) in proposals.iter().zip(&expected_ids).enumerate() {
                if actual != expected {
                    let reference_actual = top_ids[row * 32..(row + 1) * 32]
                        .iter()
                        .position(|id| id == actual)
                        .map(|index| top[row * 32 + index]);
                    println!(
                        "T19_PROPOSAL_DEFECT fixture={name} row={row} expected={expected} actual={actual} candidate_expected={} candidate_actual={} reference_expected={} reference_actual={reference_actual:?} reference_margin={}",
                        scores[row * v + *expected as usize],
                        scores[row * v + *actual as usize],
                        top[row * 32],
                        top[row * 32] - top[row * 32 + 1],
                    );
                }
            }
            let margin = (0..b)
                .map(|r| top[r * 32] - top[r * 32 + 1])
                .fold(f32::INFINITY, f32::min);
            println!(
                "T19_TEACHER fixture={name} proposal_mismatches={} min_top12_margin={margin} defects={fixture_defects}",
                proposals
                    .iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count()
            );
            defects += fixture_defects;

            draft.reset_setup()?;
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
            )?;
            let free = draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
                None,
            )?;
            defects += usize::from(free != expected_ids);
            let mut expected_bits = Vec::new();
            draft.state_bits(context, b, &mut expected_bits)?;
            let anchor = s.clone_htod(&[block_ids[0]])?;
            let prefix = s.clone_htod(&[context as i32])?;
            s.begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )?;
            let launched = draft.propose_device(
                &anchor,
                &prefix,
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
            );
            let captured = s.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = captured?.context("empty draft graph")?;
            for _ in 0..2 {
                graph.launch()?;
                s.synchronize()?;
                let mut actual = Vec::new();
                draft.state_bits(context, b, &mut actual)?;
                defects += usize::from(actual != expected_bits);
                graph_replays += 1;
            }
            drop(graph);
            let requests: Vec<_> = (0..b)
                .map(|row| {
                    (
                        row as u8,
                        if row == 0 {
                            block_ids[0]
                        } else {
                            expected_ids[row - 1]
                        },
                    )
                })
                .collect();
            let first = draft.distributions_batch(&requests)?;
            let expanded: Vec<_> = requests.iter().copied().cycle().take(64).collect();
            let many = draft.distributions_batch(&expanded)?;
            for (i, actual) in many.iter().enumerate() {
                defects += usize::from(!actual.bits_equal(&first[i % b]));
            }
            for chunk in [1, 8] {
                let mut start = 0;
                for requests in requests.chunks(chunk) {
                    let actual = draft.distributions_batch(requests)?;
                    for (index, value) in actual.iter().enumerate() {
                        defects += usize::from(!value.bits_equal(&first[start + index]));
                    }
                    native_requests += actual.len();
                    start += actual.len();
                }
            }
            for (actual, expected) in first.iter().zip(&expected_ids) {
                defects += usize::from(actual.tokens[0] != *expected);
            }
            native_requests += requests.len() + expanded.len();
            if name == names[0] {
                let maximum = s
                    .clone_dtoh(&draft.weights.w2)?
                    .iter()
                    .map(|v| v.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let parent = block_ids[0] as usize;
                let embedding = s.clone_dtoh(
                    &draft
                        .weights
                        .w1
                        .slice(parent * c.markov_rank..(parent + 1) * c.markov_rank),
                )?;
                let magnitude: f32 =
                    embedding.iter().map(|v| v.to_f32().abs()).sum::<f32>() * maximum;
                let mut base = vec![bf16::ZERO; v];
                base[v - 1] = bf16::from_f32(2.0 * magnitude + 1.0);
                s.memcpy_htod(&base, &mut draft.base.slice_mut((b - 1) * v..b * v))?;
                let tail = draft.distributions_batch(&[((b - 1) as u8, block_ids[0])])?;
                defects += usize::from(tail[0].tokens[0] as usize != v - 1);
                native_requests += 1;
                println!(
                    "T19_VOCAB_TAIL vocab={v} row={} expected={} actual={} defects={defects}",
                    b - 1,
                    v - 1,
                    tail[0].tokens[0]
                );
            }

            draft.position_shift = 257;
            draft.reset_setup()?;
            draft.inject(
                &taps,
                context,
                0,
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                None,
            )?;
            let mut wrong = Trace::default();
            draft.propose(
                block_ids[0],
                context,
                (&embedding, &head),
                Compute {
                    ops: &ops,
                    blas: &mut blas,
                },
                Some((&mut wrong, "draft")),
                Some(&expected_ids),
            )?;
            let mut rejected = 0;
            for layer in 0..c.num_hidden_layers {
                let key = format!("draft_layer_{layer}");
                rejected += compare(
                    &format!("{name}.wrong_position.{key}"),
                    &wrong.values(&format!("draft.layer.{layer}"))?,
                    (
                        &values(&tensors, &key, &[1, b, h])?,
                        &values(&floors, &key, &[1, b, h])?,
                    ),
                    coefficient(&c, context, layer + 1, 0),
                    h,
                )?;
            }
            for row in [0, b - 1] {
                let key = format!("combined_logits_k{row}");
                let actual = wrong.values("draft.s")?;
                rejected += compare(
                    &format!("{name}.wrong_position.{key}"),
                    &actual[row * v..(row + 1) * v],
                    (
                        &values(&tensors, &key, &[v])?,
                        &values(&floors, &key, &[v])?,
                    ),
                    coefficient(&c, context, c.num_hidden_layers, 3)
                        + super::super::bf16_chain_bound(1, &[c.markov_rank]),
                    v,
                )?;
            }
            rejected_control += usize::from(rejected > 0);
            println!(
                "T19_FIXTURE fixture={name} free_proposal_mismatches={} wrong_position_violations={rejected} defects={defects}",
                free.iter()
                    .zip(&expected_ids)
                    .filter(|(a, b)| a != b)
                    .count()
            );
        }
        defects += usize::from(rejected_control == 0);
        println!(
            "T19_DRAFT fixtures=6 graph_replays={graph_replays} native_markov_requests={native_requests} wrong_position_rejected={rejected_control} defects={defects} {}",
            if defects == 0 { "PASS" } else { "DEFECT" }
        );
        ensure!(defects == 0, "T19 draft has {defects} finite defects");
        Ok(())
    }
}
