use super::*;
use crate::backend::cuda::{
    contracts::{RowDomain, SeqRef, Tensor, TensorMut, view},
    cubin,
    decode::GqaDecodeInput,
    linear::LinearWeightRef,
    ops::{AddRmsNorm, CommonOpsRef},
};
use cudarc::driver::{CudaFunction, DevicePtr};

pub(super) const ROWWISE_PREFIX_LIMIT: usize = 64;

pub(super) struct Runtime {
    initial: CudaSlice<bf16>,
    high: CudaSlice<bf16>,
    low: CudaSlice<bf16>,
    rope: CudaSlice<bf16>,
    lengths: CudaSlice<i32>,
    meta: CudaSlice<i32>,
    embed: CudaFunction,
    rotary: CudaFunction,
    common: CommonOpsRef,
    program: Vec<(bool, usize)>,
    #[cfg(test)]
    control: u8,
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

fn yarn_table(capacity: usize) -> Vec<bf16> {
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
    (0..capacity)
        .flat_map(|p| {
            let inv = &inv;
            (0..128).map(move |d| {
                let angle = p as f32 * inv[d % 64];
                bf16::from_f32(if d < 64 {
                    angle.cos() * scale
                } else {
                    angle.sin() * scale
                })
            })
        })
        .collect()
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
        let rows = ROWWISE_PREFIX_LIMIT.min(capacity - 1);
        ensure!(
            requested_chunk.is_none_or(|n| n == rows),
            "LoopVL bidirectional prefix cannot be chunked; ROWWISE_PREFIX_LIMIT={rows}"
        );
        let device = Device::new(ordinal)?;
        let config = model_spec.config();
        let manifest = crate::checkpoint::MappedCheckpoint::open(dir)?
            .manifest(&model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let (free, total) = device.ctx.mem_get_info()?;
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model_spec)?.row_bytes()?
            + 4 * config.hidden_size;
        let fixed =
            manifest.resident_bytes + rows * row_bytes + capacity * 128 * 2 + 8 * 1024 * 1024;
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
        for n in 1..=rows {
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
            rope: s.clone_htod(&yarn_table(capacity))?,
            lengths: s.alloc_zeros(rows)?,
            // Elementwise gate metadata spans the workspace; attention uses lengths.
            meta: s.clone_htod(&[0i32, rows as i32, rows as i32, capacity as i32, 1, 0, 0, 0])?,
            embed: cubin::module(&device.ctx, "embed")?.load_function("embed_scale_broadcast")?,
            rotary: cubin::module(&device.ctx, "rope")?.load_function("rope_table_kv")?,
            common: CommonOpsRef::new(&device.ctx)?,
            program: (0..8).map(|call| (call % 4 == 3, call * 16)).collect(),
            #[cfg(test)]
            control: 0,
        };
        let work = Work::new(s, &model_spec, rows)?;
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
            flash: None,
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
            "LoopVL ROWWISE_PREFIX_LIMIT={} exceeded: rows={}",
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
        let s = &self.device.stream;
        s.memcpy_htod(ids, &mut self.work.ids.slice_mut(..ids.len()))?;
        let lengths: Vec<_> = (0..ids.len())
            .map(|r| {
                if r < prefix {
                    prefix as i32
                } else {
                    (r + 1) as i32
                }
            })
            .collect();
        s.memcpy_htod(
            &lengths,
            &mut self
                .recurrent
                .as_mut()
                .context("recurrent runtime missing")?
                .lengths
                .slice_mut(..ids.len()),
        )?;
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
            s.launch_builder(&r.embed)
                .arg(&self.weights.embed)
                .arg(if token { &w.token } else { &w.ids })
                .arg(&r.initial)
                .arg(&mut r.high)
                .arg(&mut r.low)
                .arg(&(rows as i32))
                .arg(&(h as i32))
                .arg(&(39.191835884530846f64 as f32))
                .launch(flat(rows * h))?;
        }
        for &(high, offset) in &r.program {
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
                for row in 0..rows {
                    let length = if token {
                        w.length.slice(..)
                    } else {
                        r.lengths.slice(row..row + 1)
                    };
                    self.decode
                        .as_mut()
                        .context("decode attention missing")?
                        .run_with(
                            s,
                            GqaDecodeInput {
                                q: &w.qkv,
                                q_offset: row * 4 * h,
                                k: &kv.k,
                                v: &kv.v,
                                output: &mut w.attn,
                                output_offset: row * h,
                                length: &length,
                                sinks: None,
                                window: None,
                            },
                        )?;
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
            engine
                .prefill(&[6; ROWWISE_PREFIX_LIMIT + 1], None)
                .is_err(),
            "DEFECT ROWWISE_PREFIX_LIMIT accepted excess row"
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
            "PASS loopvl_text_gpu fixtures=3 teacher_forced_steps=32 modes=eager,graph prefill_limit=64 controls=1,2,3,6"
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
