use super::{
    Device,
    blas::Blas,
    copy::{Rect, columns},
    flash::Shape,
    ops::{Ops, flat, grid},
    verification::Verification,
    weights::{Layer, Weights, checkpoint},
};
use crate::{config::Config, dspark::DraftConfig, trace::Trace};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, PushKernelArg};
use half::bf16;
use std::{path::Path, sync::Arc};

pub struct Compute<'a> {
    pub ops: &'a Ops,
    pub blas: &'a mut Blas,
    pub config: &'a Config,
}

enum Output {
    Tokens,
    Device,
    Base,
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
    confidence: Vec<f32>,
    confidence_bias: f32,
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
            let confidence = d
                .upload
                .clone_dtoh(&confidence)?
                .into_iter()
                .map(bf16::to_f32)
                .collect();
            let confidence_bias = d.upload.clone_dtoh(&confidence_bias)?[0].to_f32();
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
    pub config: DraftConfig,
    weights: DraftWeights,
    kv: Vec<Kv>,
    capacity: usize,
    rows: usize,
    stream: Arc<CudaStream>,
    attention: Option<Verification>,
    qknorm_rope_kv: cudarc::driver::CudaFunction,
    ids: CudaSlice<u32>,
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
        let s = &self.stream;
        let dim = self.config.head_dim;
        for kv in &mut self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = head * self.capacity * dim;
                s.memset_zeros(&mut kv.k.slice_mut(lo..lo + 8 * dim))?;
                s.memset_zeros(&mut kv.v.slice_mut(lo..lo + 8 * dim))?;
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
        s.memcpy_htod(&[self.config.mask_token_id; 7], &mut self.ids)?;
        Ok(())
    }
    pub fn new(
        path: &Path,
        d: &Device,
        c: &Config,
        capacity: usize,
        rows: usize,
        blas: &mut Blas,
    ) -> Result<Self> {
        let config = DraftConfig::read(path, c)?;
        ensure!(rows >= 8, "DSpark workspace needs at least eight rows");
        let weights = DraftWeights::load(path, &config, d)?;
        let s = &d.stream;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let kv = c.kv_dim();
        let mut caches = Vec::new();
        for _ in 0..config.num_hidden_layers {
            caches.push(Kv {
                k: s.alloc_zeros(capacity * kv)?,
                v: s.alloc_zeros(capacity * kv)?,
            });
        }
        for (n, o, i) in [
            (7, c.qkv_dim(), h),
            (7, h, h),
            (7, 2 * f, h),
            (7, h, f),
            (7, c.vocab_size, h),
            (rows, h, 5 * h),
            (rows, kv, h),
        ] {
            blas.prepare(n, o, i)?;
        }
        eprintln!(
            "draft_weights_bytes={} draft_kv_bytes={}",
            weights.bytes,
            config.num_hidden_layers * capacity * kv * 4
        );
        let mask = config.mask_token_id;
        let qknorm_rope_kv =
            super::cubin::module(&d.ctx, "rope")?.load_function("draft_qknorm_rope_kv")?;
        Ok(Self {
            markov: None,
            config,
            weights,
            kv: caches,
            capacity,
            rows,
            stream: s.clone(),
            attention: None,
            qknorm_rope_kv,
            ids: s.clone_htod(&[mask; 7])?,
            position: s.alloc_zeros(1)?,
            x: s.alloc_zeros(7 * h)?,
            n: s.alloc_zeros(7 * h)?,
            qkv: s.alloc_zeros(7 * c.qkv_dim())?,
            attn: s.alloc_zeros(7 * h)?,
            out: s.alloc_zeros(7 * h)?,
            gu: s.alloc_zeros(7 * 2 * f)?,
            act: s.alloc_zeros(7 * f)?,
            k_raw: s.alloc_zeros(rows * kv)?,
            k_norm: s.alloc_zeros(rows * kv)?,
            v_raw: s.alloc_zeros(rows * kv)?,
            projected: s.alloc_zeros(rows * h)?,
            context: s.alloc_zeros(rows * h)?,
            packed: s.alloc_zeros(rows * 2 * kv)?,
            base: s.alloc_zeros(7 * c.vocab_size)?,
            scores: s.alloc_zeros(7 * c.vocab_size)?,
            prev: s.alloc_zeros(config_rank(c))?,
            token: s.alloc_zeros(1)?,
            tokens: s.alloc_zeros(7)?,
        })
    }

    pub(super) fn grow(&mut self, c: &Config, rows: usize) -> Result<()> {
        ensure!(rows >= self.rows, "draft workspace rows cannot shrink");
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
        let Compute {
            ops,
            blas,
            config: c,
        } = compute;
        ensure!(
            rows > 0 && rows <= self.rows && start + rows <= self.capacity,
            "draft injection exceeds workspace/cache"
        );
        let s = &self.stream;
        let h = c.hidden_size;
        let kv = c.kv_dim();
        blas.prepare(rows, h, 5 * h)?;
        blas.prepare(rows, kv, h)?;
        blas.linear(
            &self.weights.fc,
            &input.slice(..rows * 5 * h),
            &mut self.projected.slice_mut(..rows * h),
            rows,
            h,
            5 * h,
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
                    .arg(&c.rope_theta)
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
                format!("{prefix}.c_committed"),
                &self.context,
                rows,
                h,
            )?;
        }
        Ok(())
    }
}

fn config_rank(_c: &Config) -> usize {
    256
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
        let measured = super::multi_calibrate::load(
            d,
            ops,
            c,
            dir,
            crate::backend::setup::MultiShape {
                rows: 7,
                causal: false,
                capacity: self.capacity,
            },
            closure,
        )?;
        let (plan, selection) = measured.choose(self.capacity)?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"role":"draft","rows":7,"causal":false,"attention":plan,"selection":selection,"calibration":measured}})
        );
        self.attention = Some(Verification::with_plan(
            &d.ctx,
            &d.stream,
            c,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows: 7,
                causal: false,
            },
            plan,
        )?);
        Ok(())
    }
    pub fn propose(
        &mut self,
        anchor: u32,
        start: usize,
        target: &Weights,
        compute: Compute<'_>,
        trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        let mut ids = [self.config.mask_token_id; 7];
        ids[0] = anchor;
        self.stream.memcpy_htod(&ids, &mut self.ids)?;
        self.stream
            .memcpy_htod(&[start as i32], &mut self.position)?;
        self.block(start, target, compute, trace, forced, Output::Tokens)
    }

    pub fn propose_device(
        &mut self,
        anchor: &CudaSlice<u32>,
        position: &CudaSlice<i32>,
        start: usize,
        target: &Weights,
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
        target: &Weights,
        compute: Compute<'_>,
    ) -> Result<()> {
        let mut ids = [self.config.mask_token_id; 7];
        ids[0] = anchor;
        self.stream.memcpy_htod(&ids, &mut self.ids)?;
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
        target: &Weights,
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
            self.markov = Some(super::markov::Markov::new(d, ops, dir, closure)?);
        }
        Ok(())
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
        for values in [&self.n, &self.base, &self.scores] {
            super::copy::bits(s, values, out)?;
        }
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
        target: &Weights,
        compute: Compute<'_>,
        mut trace: Option<(&mut Trace, &str)>,
        forced: Option<&[u32]>,
        output: Output,
    ) -> Result<Vec<u32>> {
        if let Some(tokens) = forced {
            ensure!(
                tokens.len() == 7
                    && tokens
                        .iter()
                        .all(|&v| (v as usize) < self.config.vocab_size),
                "invalid teacher-forced Markov tokens"
            );
        }
        let Compute {
            ops,
            blas,
            config: c,
        } = compute;
        ensure!(
            start + 7 <= self.capacity,
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
                    .arg(&7i32)
                    .arg(&(h as i32))
                    .arg(&eps)
                    .launch(grid(7, 256))?;
            }
            Ok(())
        };
        unsafe {
            s.launch_builder(&ops.embed)
                .arg(&target.embed)
                .arg(&self.ids)
                .arg(&mut self.x)
                .arg(&7i32)
                .arg(&(h as i32))
                .launch(flat(7 * h))?;
        }
        for (i, layer) in self.weights.layers.iter().enumerate() {
            if i == 0 {
                norm(
                    s,
                    ops,
                    &self.x,
                    &layer.block.input_norm,
                    &mut self.n,
                    (7, h),
                    self.config.rms_norm_eps,
                )?;
            }
            blas.linear(
                &layer.block.qkv,
                &self.n.slice(..7 * h),
                &mut self.qkv.slice_mut(..7 * qkv),
                7,
                qkv,
                h,
            )?;
            let cache = &mut self.kv[i];
            unsafe {
                s.launch_builder(&self.qknorm_rope_kv)
                    .arg(&mut self.qkv)
                    .arg(&cache.k)
                    .arg(&cache.v)
                    .arg(&layer.q_norm)
                    .arg(&layer.k_norm)
                    .arg(&self.position)
                    .arg(&7i32)
                    .arg(&(c.num_attention_heads as i32))
                    .arg(&(c.num_key_value_heads as i32))
                    .arg(&(c.head_dim as i32))
                    .arg(&(self.capacity as i32))
                    .arg(&c.rope_theta)
                    .arg(&self.config.rms_norm_eps)
                    .launch(grid(
                        7 * (c.num_attention_heads + 2 * c.num_key_value_heads),
                        256,
                    ))?;
            }
            self.attention
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("draft attention has not been initialized"))?
                .run(
                    &self.qkv,
                    &self.kv[i].k,
                    &self.kv[i].v,
                    &mut self.attn,
                    &self.position,
                    Shape {
                        rows: 7,
                        start,
                        q_heads: c.num_attention_heads,
                        kv_heads: c.num_key_value_heads,
                        capacity: self.capacity,
                        causal: false,
                    },
                )?;
            blas.linear(
                &layer.block.o,
                &self.attn.slice(..7 * h),
                &mut self.out.slice_mut(..7 * h),
                7,
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
                &self.n.slice(..7 * h),
                &mut self.gu.slice_mut(..7 * 2 * f),
                7,
                2 * f,
                h,
            )?;
            unsafe {
                s.launch_builder(&ops.act)
                    .arg(&self.gu)
                    .arg(&mut self.act)
                    .arg(&7i32)
                    .arg(&(f as i32))
                    .launch(flat(7 * f))?;
            }
            blas.linear(
                &layer.block.down,
                &self.act.slice(..7 * f),
                &mut self.out.slice_mut(..7 * h),
                7,
                h,
                f,
            )?;
            let weight = if i + 1 < self.weights.layers.len() {
                &self.weights.layers[i + 1].block.input_norm
            } else {
                &self.weights.norm
            };
            add_norm(ops, &self.out, &mut self.x, weight, &mut self.n)?;
        }
        blas.linear(
            &target.head,
            &self.n.slice(..7 * h),
            &mut self.base.slice_mut(..7 * c.vocab_size),
            7,
            c.vocab_size,
            h,
        )?;
        if matches!(output, Output::Base) {
            return Ok(Vec::new());
        }
        s.memcpy_dtod(&self.ids.slice(..1), &mut self.token)?;
        let mut confidence = Vec::new();
        for row in 0..7 {
            unsafe {
                s.launch_builder(&ops.embed)
                    .arg(&self.weights.w1)
                    .arg(&self.token)
                    .arg(&mut self.prev)
                    .arg(&1i32)
                    .arg(&(self.config.markov_rank as i32))
                    .launch(flat(self.config.markov_rank))?;
            }
            if trace.is_some() {
                let hrow = s.clone_dtoh(&self.n.slice(row * h..(row + 1) * h))?;
                let prev = s.clone_dtoh(&self.prev)?;
                let sum = hrow
                    .into_iter()
                    .chain(prev)
                    .zip(&self.weights.confidence)
                    .map(|(v, w)| v.to_f32() * w)
                    .sum::<f32>()
                    + self.weights.confidence_bias;
                let value = bf16::from_f32(sum).to_f32();
                confidence.push(bf16::from_f32(1.0 / (1.0 + (-value).exp())).to_f32());
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
            unsafe {
                s.launch_builder(&ops.residual)
                    .arg(
                        &mut self
                            .scores
                            .slice_mut(row * c.vocab_size..(row + 1) * c.vocab_size),
                    )
                    .arg(
                        &self
                            .base
                            .slice(row * c.vocab_size..(row + 1) * c.vocab_size),
                    )
                    .arg(&(c.vocab_size as i32))
                    .launch(flat(c.vocab_size))?;
                s.launch_builder(&ops.argmax)
                    .arg(
                        &self
                            .scores
                            .slice(row * c.vocab_size..(row + 1) * c.vocab_size),
                    )
                    .arg(&mut self.token)
                    .arg(&(c.vocab_size as i32))
                    .launch(grid(1, 1024))?;
            }
            s.memcpy_dtod(&self.token, &mut self.tokens.slice_mut(row..row + 1))?;
            if let Some(tokens) = forced {
                s.memcpy_htod(&[tokens[row]], &mut self.token)?;
            }
        }
        let proposals = if matches!(output, Output::Tokens) {
            s.clone_dtoh(&self.tokens)?
        } else {
            Vec::new()
        };
        if let Some((trace, prefix)) = trace.as_mut() {
            add_trace(s, trace, format!("{prefix}.H"), &self.n, 7, h)?;
            add_trace(s, trace, format!("{prefix}.B"), &self.base, 7, c.vocab_size)?;
            add_trace(
                s,
                trace,
                format!("{prefix}.s"),
                &self.scores,
                7,
                c.vocab_size,
            )?;
            trace.add(format!("{prefix}.conf"), vec![7], confidence)?;
            trace.add(
                format!("{prefix}.proposed"),
                vec![7],
                proposals.iter().map(|&v| v as f32).collect(),
            )?;
        }
        Ok(proposals)
    }

    pub fn snapshot_kv(
        &self,
        trace: &mut Trace,
        start: usize,
        rows: usize,
        c: &Config,
    ) -> Result<()> {
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
