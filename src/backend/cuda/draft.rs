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
        let limits = AttentionLimits {
            max_rows: rows,
            capacity: u32::try_from(draft.capacity)?,
            key_chunk: 32,
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
    ops: &Ops,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    shape: (usize, usize),
    eps: f32,
) -> Result<()> {
    let (rows, dim) = shape;
    unsafe {
        s.launch_builder(&ops.norm)
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
            ops,
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
                ops,
                &self.k_raw,
                &layer.k_norm,
                &mut self.k_norm,
                (rows * c.num_key_value_heads, c.head_dim),
                self.config.rms_norm_eps,
            )?;
            unsafe {
                s.launch_builder(&ops.rope_float)
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
        let add_norm = |ops: &Ops,
                        res: &CudaSlice<bf16>,
                        x: &mut CudaSlice<bf16>,
                        weight: &CudaSlice<bf16>,
                        output: &mut CudaSlice<bf16>|
         -> Result<()> {
            let (xp, _x) = x.device_ptr_mut(s);
            unsafe {
                s.launch_builder(&ops.add_norm)
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
                    ops,
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
            add_norm(
                ops,
                &self.out,
                &mut self.x,
                &layer.block.post_norm,
                &mut self.n,
            )?;
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
            add_norm(ops, &self.out, &mut self.x, weight, &mut self.n)?;
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
