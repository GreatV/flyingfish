//! Clover masked-block canvas execution.
use super::*;
use crate::backend::cuda::cubin;
use crate::backend::gpu::PushKernelArg;
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::collections::VecDeque;

pub(crate) struct Runtime {
    state: Box<CloverState>,
    prefix: usize,
    pending: VecDeque<u32>,
    canvas: usize,
    limit: Option<usize>,
    produced: usize,
    terminal: bool,
}

struct CloverState {
    forward: crate::backend::cuda::block::BlockForward,
    sampler: crate::backend::cuda::sample::BlockSampler,
    logits: CudaSlice<f32>,
    upcast: crate::backend::gpu::CudaFunction,
    #[cfg(test)]
    capture: Option<CloverCapture>,
    #[cfg(test)]
    policy: Option<VecDeque<CloverPolicyStep>>,
}

#[cfg(test)]
struct CloverPolicyStep {
    picks: Vec<i32>,
    confidence: Vec<f32>,
    block: Vec<i32>,
    pending: Vec<u8>,
}

#[cfg(test)]
struct CloverCapture {
    root: std::path::PathBuf,
    request: usize,
    calls: Vec<serde_json::Value>,
    committed: Vec<std::collections::BTreeMap<String, Vec<u32>>>,
    control: bool,
    control_attempted: bool,
    defects: Vec<String>,
}

#[cfg(test)]
impl CloverCapture {
    fn directory(&self) -> std::path::PathBuf {
        self.root.join(format!("request{}", self.request - 1))
    }

    fn save<'a>(
        &self,
        name: &str,
        tensors: impl IntoIterator<Item = (&'a str, Vec<usize>, Vec<u8>, safetensors::Dtype)>,
    ) -> Result<()> {
        use safetensors::tensor::TensorView;
        let values: Vec<_> = tensors.into_iter().collect();
        let views = values
            .iter()
            .map(|(key, shape, bytes, dtype)| {
                Ok((*key, TensorView::new(*dtype, shape.clone(), bytes)?))
            })
            .collect::<Result<Vec<_>>>()?;
        safetensors::serialize_to_file(views, None, &self.directory().join(name))?;
        Ok(())
    }

    fn metadata(&self) -> Result<()> {
        std::fs::write(
            self.directory().join("capture.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"keys":813,"calls":self.calls,"preview_commit_control":self.control,"defects":self.defects}),
            )?,
        )?;
        Ok(())
    }
}

impl Engine {
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn validate_canvas_reference(
        &mut self,
        reference: &std::path::Path,
    ) -> Result<usize> {
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("FROZEN.json"))?)?;
        let data = std::fs::read(reference.join("inputs.safetensors"))?;
        let inputs = safetensors::SafeTensors::deserialize(&data)?;
        let integers = |name: &str| -> Result<Vec<i32>> {
            let value = inputs.tensor(name)?;
            ensure!(
                value.dtype() == safetensors::Dtype::I32,
                "Runner FAIL canonical input dtype"
            );
            Ok(value
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v))
                .collect())
        };
        let mut defects = 0;
        for case in frozen["cases"]
            .as_array()
            .context("Runner FAIL canonical cases")?
        {
            self.reset()?;
            self.generation_request(case["maximum"].as_u64().unwrap() as usize, false, false)?;
            let rows = case["prompt"].as_array().unwrap().len() % 32;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            state
                .forward
                .prepare_causal_rows(&self.device, if rows == 0 { 32 } else { rows as u32 })?;
            for call in frozen["calls"]
                .as_array()
                .context("Runner FAIL canonical calls")?
                .iter()
                .filter(|call| call["case"] == case["tag"])
            {
                let key = call["key"].as_str().unwrap();
                let tokens = integers(&format!("{key}.tokens"))?;
                if call["preview"] == false {
                    let tokens: Vec<_> = tokens.into_iter().map(|id| id as u32).collect();
                    for chunk in tokens.chunks(32) {
                        self.diffusion_encoder_chunk(chunk)?;
                    }
                } else {
                    let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
                    runtime.produced = call["length"].as_u64().unwrap() as usize;
                    let state = &mut runtime.state;
                    self.device
                        .stream
                        .memcpy_htod(&tokens, &mut state.sampler.block)?;
                    self.canvas_forward()?;
                    self.capture_clover("raw", None)?;
                }
            }
            let runtime = self.diffusion.as_ref().context("canvas runtime missing")?;
            let state = &runtime.state;
            let capture = state
                .capture
                .as_ref()
                .context("Runner FAIL canonical capture missing")?;
            defects += capture.defects.len() + usize::from(!capture.control);
            std::fs::write(
                capture.directory().join("result.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                "mode":"canonical","case":case["tag"],"prompt":case["prompt"],"position":self.position}))?,
            )?;
            println!(
                "CLOVER_GENERATION_CANONICAL case={} calls={} cache_defects={} control={}",
                case["tag"],
                capture.calls.len(),
                capture.defects.len(),
                capture.control
            );
        }
        defects += self.validate_clover_policy()?;
        let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
        let saved = (
            runtime.prefix,
            runtime.terminal,
            std::mem::take(&mut runtime.pending),
        );
        runtime.prefix = self.capacity;
        runtime.terminal = false;
        runtime.pending.extend([17, 19]);
        let first = self.decode_diffusion(false);
        let second = self.decode_diffusion(false);
        let exhausted = self.decode_diffusion(false);
        let valid =
            first.is_ok_and(|id| id == 17) && second.is_ok_and(|id| id == 19) && exhausted.is_err();
        let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
        (runtime.prefix, runtime.terminal, runtime.pending) = saved;
        defects += usize::from(!valid);
        println!(
            "CLOVER_CANVAS_QUEUE_BOUNDARY {} returned=2 exhausted_capacity_rejected={}",
            if valid { "PASS" } else { "DEFECT" },
            exhausted.is_err()
        );
        Ok(defects)
    }

    #[cfg(all(test, feature = "cuda"))]
    fn validate_clover_policy(&mut self) -> Result<usize> {
        use crate::backend::cuda::sample::tests::{
            expected_fill, histogram, integer_values, restore,
        };
        let fixture = crate::backend::cuda::moe::reference::CloverMoeFixture::open_named(
            "sampling.safetensors",
        )?;
        ensure!(
            fixture.table["bounds"]
                == serde_json::json!({"logit_bits":0,"ids":0,"pending":0,"counts":0,"returned":0}),
            "Runner FAIL fixed policy bounds"
        );
        let mut defects = 0;
        let raw = fixture.values("raw")?;
        for case in fixture.table["penalty_cases"]
            .as_array()
            .context("Runner FAIL penalty cases")?
        {
            self.reset()?;
            let tag = case["tag"].as_str().unwrap();
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            restore(&mut state.sampler, &fixture, case, &serde_json::Value::Null)?;
            self.device.stream.memcpy_htod(&raw, &mut state.logits)?;
            self.canvas_sample()?;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            let mut expected = raw.clone();
            for (index, value) in integer_values(&fixture, &format!("{tag}.patch_ids"))?
                .into_iter()
                .zip(fixture.values(&format!("{tag}.patch_values"))?)
            {
                expected[index as usize] = value;
            }
            let (block, pending) = expected_fill(&fixture, tag)?;
            let s = &self.device.stream;
            let valid = s
                .clone_dtoh(&state.logits)?
                .iter()
                .map(|v| v.to_bits())
                .eq(expected.iter().map(|v| v.to_bits()))
                && s.clone_dtoh(&state.sampler.picks)?
                    == integer_values(&fixture, &format!("{tag}.picks"))?
                && s.clone_dtoh(&state.sampler.block)? == block
                && s.clone_dtoh(&state.sampler.pending)? == pending
                && s.clone_dtoh(&state.sampler.counts)?
                    == histogram(
                        &fixture,
                        &format!("{tag}.counts"),
                        self.model_spec.vocab as usize,
                    )?;
            defects += usize::from(!valid);
            println!(
                "CLOVER_DRIVER_PENALTY {} tag={tag} logit_bits_exact={valid}",
                if valid { "PASS" } else { "DEFECT" }
            );
        }
        for case in fixture.table["policy_cases"]
            .as_array()
            .context("Runner FAIL policy cases")?
        {
            self.reset()?;
            let tag = case["tag"].as_str().unwrap();
            let maximum = case["maximum"].as_u64().unwrap() as usize;
            self.generation_request(maximum, false, false)?;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            state.forward.prepare_causal_rows(&self.device, 3)?;
            let literal_mask =
                self.device
                    .stream
                    .clone_dtoh(&state.sampler.block.slice(..1))?[0] as u32;
            self.diffusion_encoder_chunk(&[literal_mask, 17, 17])?;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            let capture = state.capture.take();
            let mut steps = VecDeque::new();
            for event in case["events"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|event| event["kind"] == "fill")
            {
                let key = event["key"].as_str().unwrap();
                steps.push_back(CloverPolicyStep {
                    picks: integer_values(&fixture, &format!("{key}.picks"))?,
                    confidence: fixture.values(&format!("{key}.confidence"))?,
                    block: integer_values(&fixture, &format!("{key}.block"))?,
                    pending: integer_values(&fixture, &format!("{key}.pending"))?
                        .into_iter()
                        .map(|v| v as u8)
                        .collect(),
                });
            }
            state.policy = Some(steps);
            let mut returned = Vec::new();
            let result: Result<()> = (|| {
                loop {
                    self.canvas_step()?;
                    let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
                    returned.extend(runtime.pending.drain(..));
                    if runtime.terminal {
                        break;
                    }
                }
                Ok(())
            })();
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let state = &mut runtime.state;
            state.capture = capture;
            let exhausted = state.policy.take().unwrap().is_empty();
            let last = case["events"].as_array().unwrap().last().unwrap();
            let counts = histogram(
                &fixture,
                &format!("{}.counts", last["key"].as_str().unwrap()),
                self.model_spec.vocab as usize,
            )?;
            let expected: Vec<_> = integer_values(&fixture, &format!("{tag}.returned"))?
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let valid = result.is_ok()
                && exhausted
                && returned == expected
                && self.device.stream.clone_dtoh(&state.sampler.counts)? == counts
                && self.position == 3 + 32 * (maximum.div_ceil(32) - 1);
            defects += usize::from(!valid);
            println!(
                "CLOVER_DRIVER_POLICY {} tag={tag} returned={} position={} error={:?}",
                if valid { "PASS" } else { "DEFECT" },
                returned.len(),
                self.position,
                result.err()
            );
        }
        Ok(defects)
    }

    #[cfg(test)]
    fn capture_clover(&mut self, phase: &str, host: Option<&[u32]>) -> Result<()> {
        use safetensors::Dtype;
        let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
        let state = &mut runtime.state;
        let Some(capture) = &mut state.capture else {
            return Ok(());
        };
        let s = &self.device.stream;
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let integers = |values: &[i32]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        match phase {
            "causal" => {
                let tokens: Vec<i32> = match host {
                    Some(ids) => ids.iter().map(|&v| v as i32).collect(),
                    None => s.clone_dtoh(&state.sampler.block)?,
                };
                let rows = tokens.len();
                let image = state.forward.committed_image(&self.device)?;
                capture.committed = image
                    .iter()
                    .map(|cache| {
                        cache
                            .iter()
                            .map(|(name, values)| {
                                (name.clone(), values.iter().map(|v| v.to_bits()).collect())
                            })
                            .collect()
                    })
                    .collect();
                let mut tensors = vec![(
                    "tokens".to_owned(),
                    vec![1, rows],
                    integers(&tokens),
                    Dtype::I32,
                )];
                for (index, cache) in image.iter().enumerate() {
                    for (name, values) in cache {
                        ensure!(
                            values.iter().all(|v| v.is_finite()),
                            "Runner FAIL nonfinite Clover cache"
                        );
                        let (name, shape) = match name.as_str() {
                            "state_ssm" => ("ssm", vec![1, 32, 128, 128]),
                            "state_conv" => ("conv", vec![1, 8192, 3]),
                            "state_key" => ("key", vec![1, 2, runtime.prefix, 256]),
                            "state_value" => ("value", vec![1, 2, runtime.prefix, 256]),
                            _ => anyhow::bail!("Runner FAIL unknown Clover cache field"),
                        };
                        tensors.push((
                            format!("cache.layer{index}.{name}"),
                            shape,
                            bytes(values),
                            Dtype::F32,
                        ));
                    }
                }
                let key = format!("call{:04}", capture.calls.len());
                capture.save(
                    &format!("{key}.safetensors"),
                    tensors.iter().map(|(n, shape, data, dtype)| {
                        (n.as_str(), shape.clone(), data.clone(), *dtype)
                    }),
                )?;
                capture
                    .calls
                    .push(serde_json::json!({"key":key,"preview":false,
                    "prefix":runtime.prefix - rows,"rows":rows,"tier_counts":state.forward.counts}));
            }
            "raw" => {
                let image = state.forward.committed_image(&self.device)?;
                let image: Vec<std::collections::BTreeMap<String, Vec<u32>>> = image
                    .iter()
                    .map(|cache| {
                        cache
                            .iter()
                            .map(|(name, values)| {
                                (name.clone(), values.iter().map(|v| v.to_bits()).collect())
                            })
                            .collect()
                    })
                    .collect();
                let unchanged = image == capture.committed;
                if !unchanged {
                    capture
                        .defects
                        .push("preview changed committed state".into());
                }
                if !capture.control_attempted {
                    capture.control_attempted = true;
                    capture.control = state.forward.preview_commit_control(&self.device)?;
                    if !capture.control {
                        capture
                            .defects
                            .push("preview commit control escaped".into());
                    }
                }
                let key = format!("call{:04}", capture.calls.len());
                let tokens = s.clone_dtoh(&state.sampler.block)?;
                let logits = s.clone_dtoh(&state.forward.logits)?;
                let upcast = s.clone_dtoh(&state.logits)?;
                ensure!(
                    logits.iter().all(|v| v.to_f32().is_finite())
                        && upcast.iter().all(|v| v.is_finite()),
                    "Runner FAIL nonfinite Clover logits"
                );
                let upcast_exact = upcast
                    .iter()
                    .zip(&logits)
                    .all(|(a, b)| a.to_bits() == b.to_f32().to_bits());
                if !upcast_exact {
                    capture.defects.push("BF16 upcast differs".into());
                }
                capture.save(
                    &format!("{key}.safetensors"),
                    [
                        ("tokens", vec![1, 32], integers(&tokens), Dtype::I32),
                        (
                            "logits",
                            vec![32, self.model_spec.vocab as usize],
                            logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
                            Dtype::BF16,
                        ),
                    ],
                )?;
                capture.calls.push(
                    serde_json::json!({"key":key,"preview":true,"prefix":runtime.prefix,"rows":32,
                    "length":runtime.produced,"committed_unchanged":unchanged,"tier_counts":state.forward.counts,"upcast_exact":upcast_exact}),
                );
            }
            "fill" => {
                let call = capture
                    .calls
                    .last()
                    .context("Runner FAIL fill without preview")?;
                let key = call["key"].as_str().context("Runner FAIL capture key")?;
                let picks = s.clone_dtoh(&state.sampler.picks)?;
                let probabilities = s.clone_dtoh(&state.sampler.probabilities)?;
                let confidence: Vec<_> = picks
                    .iter()
                    .enumerate()
                    .map(|(row, &id)| {
                        probabilities[row * self.model_spec.vocab as usize + id as usize]
                    })
                    .collect();
                let logits = s.clone_dtoh(&state.logits)?;
                let raw = s.clone_dtoh(&state.forward.logits)?;
                let mut indices = Vec::new();
                let mut values = Vec::new();
                for (index, (&value, original)) in logits.iter().zip(raw).enumerate() {
                    if value.to_bits() != original.to_f32().to_bits() {
                        indices.push(i32::try_from(index)?);
                        values.push(value);
                    }
                }
                capture.save(
                    &format!("{key}.policy.safetensors"),
                    [
                        (
                            "penalty_indices",
                            vec![indices.len()],
                            integers(&indices),
                            Dtype::I32,
                        ),
                        (
                            "penalty_values",
                            vec![values.len()],
                            bytes(&values),
                            Dtype::F32,
                        ),
                        ("picks", vec![32], integers(&picks), Dtype::I32),
                        ("confidence", vec![32], bytes(&confidence), Dtype::F32),
                    ],
                )?;
                let block = s.clone_dtoh(&state.sampler.block)?;
                let pending: Vec<_> = s
                    .clone_dtoh(&state.sampler.pending)?
                    .into_iter()
                    .map(i32::from)
                    .collect();
                let counts = s.clone_dtoh(&state.sampler.counts)?;
                let control = s.clone_dtoh(&state.sampler.state)?;
                let count_bytes = counts.iter().flat_map(|v| v.to_le_bytes()).collect();
                capture.save(
                    &format!("{key}.fill.safetensors"),
                    [
                        ("block", vec![1, 32], integers(&block), Dtype::I32),
                        ("pending", vec![1, 32], integers(&pending), Dtype::I32),
                        ("counts", vec![counts.len()], count_bytes, Dtype::U32),
                    ],
                )?;
                capture.calls.last_mut().unwrap()["state"] = serde_json::json!(control);
            }
            "commit" => {
                let control = s.clone_dtoh(&state.sampler.state)?;
                let returned = s.clone_dtoh(&state.sampler.block.slice(..control[1] as usize))?;
                let call = capture
                    .calls
                    .last_mut()
                    .context("Runner FAIL commit without preview")?;
                call["commit"] = serde_json::json!({"state":control,"returned":returned,
                    "produced":runtime.produced,"terminal":runtime.terminal});
            }
            _ => anyhow::bail!("Runner FAIL unknown Clover capture phase"),
        }
        capture.metadata()
    }

    pub(crate) fn generation_request(
        &mut self,
        count: usize,
        ignore_eos: bool,
        graph: bool,
    ) -> Result<()> {
        if let Some(runtime) = &mut self.diffusion {
            ensure!(
                count > 0 && count <= self.model_spec.max_position as usize,
                "Clover response limit outside model capacity"
            );
            ensure!(
                !ignore_eos,
                "Clover generation requires the published EOS policy"
            );
            ensure!(!graph, "Clover canvas graph execution is not bound");
            ensure!(
                self.position == 0 && runtime.limit.is_none(),
                "Clover generation requires reset"
            );
            runtime.limit = Some(count);
            runtime.produced = 0;
            runtime.terminal = false;
            #[cfg(test)]
            if let Some(capture) = &mut runtime.state.capture {
                capture.request += 1;
                capture.calls.clear();
                capture.committed.clear();
                capture.control = false;
                capture.control_attempted = false;
                capture.defects.clear();
                std::fs::create_dir(capture.directory())?;
            }
        }
        Ok(())
    }

    pub(super) fn load_clover(
        dir: &std::path::Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        let started = std::time::Instant::now();
        ensure!(
            requested_chunk.is_none_or(|n| n == 32),
            "Clover causal chunk must be 32"
        );
        ensure!(
            capacity >= 32 && capacity <= model_spec.max_position as usize,
            "Clover capacity outside model bounds"
        );
        ensure!(
            settings.draft_model.is_none() && !settings.prefix_cache,
            "Clover draft and prefix-cache execution are not bound"
        );
        ensure!(
            settings.kv_format == crate::kv::KvFormat::Bf16,
            "Clover KV storage requires BF16"
        );
        let device = Device::new(ordinal)?;
        let (free, _) = device.ctx.mem_get_info()?;
        let work = super::Work::new(&device.stream, &model_spec, 1)?;
        let state = super::super::state::StateStore::empty(capacity)?;
        let budget = settings
            .device_budget_gib
            .map(|n| n << 30)
            .unwrap_or(free as u64)
            .min(free as u64)
            .checked_sub(work.bytes() as u64 + (4 << 20))
            .context("Clover base workspace exceeds device budget")?;
        let policy = crate::model_spec::BlockGeneration::read(dir)?;
        let forward = crate::backend::cuda::block::BlockForward::load(
            &device,
            dir,
            settings
                .runtime_dir
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("")),
            u32::try_from(capacity)?,
            Some(budget),
        )?;
        let sampler = crate::backend::cuda::sample::BlockSampler::prepare(&device, policy)?;
        let logits = device.stream.alloc_zeros(32 * model_spec.vocab as usize)?;
        let upcast =
            cubin::module(&device.ctx, "vision_ops")?.load_function("vision_upcast_f32")?;
        let runtime = Runtime {
            state: Box::new(CloverState {
                forward,
                sampler,
                logits,
                upcast,
                #[cfg(test)]
                capture: std::env::var_os("FF_CLOVER_CAPTURE").map(|root| CloverCapture {
                    root: root.into(),
                    request: 0,
                    calls: Vec::new(),
                    committed: Vec::new(),
                    control: false,
                    control_attempted: false,
                    defects: Vec::new(),
                }),
                #[cfg(test)]
                policy: None,
            }),
            prefix: 0,
            pending: VecDeque::new(),
            canvas: 32,
            limit: None,
            produced: 0,
            terminal: false,
        };
        Self::from_canvas(
            device,
            model_spec,
            settings,
            (capacity, started),
            runtime,
            state,
            work,
        )
    }

    fn from_canvas(
        device: Device,
        model_spec: crate::model_spec::ModelSpec,
        settings: Settings,
        setup: (usize, std::time::Instant),
        runtime: Runtime,
        state: super::super::state::StateStore,
        work: super::Work,
    ) -> Result<Self> {
        let (capacity, started) = setup;
        let canvas = runtime.canvas;
        let config = model_spec.config();
        let s = &device.stream;
        let ops = crate::backend::cuda::ops::Ops::new_with_swiglu_rounding(
            &device.ctx,
            model_spec.swiglu_rounding(),
        )?;
        let closure = std::rc::Rc::new(super::super::closure::Closure::new(
            false,
            "Clover canvas setup".into(),
        ));
        let runtime_dir = settings.runtime_dir.clone().unwrap_or_default();
        #[cfg(feature = "cuda")]
        let mut blas = Blas::new(&device, &ops, &runtime_dir, closure.clone())?;
        #[cfg(feature = "rocm")]
        let mut blas = Blas::new(s.clone())?;
        blas.bind_invariant(&device.ctx, 1)?;
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
            forward_rows: (1..=canvas).collect(),
            graphs: Vec::new(),
            closure,
            graph: None,
            poisoned: false,
            spec_budget: settings.spec_budget,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime_dir,
            spec: None,
            hidden_capture: None,
            qwen: None,
            deepseek_mixer: None,
            deepseek_ffn: None,
            deepseek_engram: None,
            recurrent: None,
            diffusion: Some(runtime),
            hc: None,
            glm_dsa: None,
            moe: None,
            moe_slots: None,
            moe_prefill: None,
            flint_frozen_decode_length: false,
            #[cfg(feature = "cuda")]
            mtp: None,
            #[cfg(feature = "cuda")]
            mtp_state: None,
            #[cfg(feature = "cuda")]
            mtp_epoch: 0,
            images: None,
            weights: crate::backend::cuda::weights::Weights {
                embed: None,
                layers: Vec::new(),
                norm: None,
                head: s.alloc_zeros::<bf16>(0)?,
                bytes: 0,
            },
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash: None,
            decode: None,
            logits_f32: None,
            device,
            config,
            model_spec,
            capacity,
            chunk: canvas,
            workspace_rows: canvas,
            trace_layer: 0,
            position: 0,
            #[cfg(feature = "cuda")]
            prefix_identity: None,
            #[cfg(feature = "cuda")]
            prefix_identity_load_ms: 0.0,
            logits_cache: Default::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        engine.closure.freeze_fixed()?;
        engine.reset()?;
        engine.closure.ready(started)?;
        Ok(engine)
    }

    pub(super) fn prefill_diffusion(
        &mut self,
        ids: &[u32],
        trace: Option<&mut crate::trace::Trace>,
    ) -> Result<u32> {
        ensure!(trace.is_none(), "Clover prefill trace is not bound");
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned && self.position == 0,
            "prefill requires reset"
        );
        ensure!(
            !ids.is_empty() && ids.len() + self.diffusion_canvas()? <= self.capacity,
            "prompt plus one canvas exceeds KV capacity {}",
            self.capacity
        );
        ensure!(
            ids.iter().all(|&x| x < self.model_spec.vocab),
            "input token id outside vocabulary"
        );
        let canvas = self.diffusion_canvas()?;
        if let Some(runtime) = &mut self.diffusion {
            let state = &mut runtime.state;
            let count = runtime
                .limit
                .context("Clover prefill requires a generation request")?;
            let extent = ids
                .len()
                .checked_add(
                    count
                        .div_ceil(32)
                        .checked_mul(32)
                        .context("Clover request canvas extent overflow")?,
                )
                .context("Clover request extent overflow")?;
            ensure!(
                extent <= self.capacity && extent <= self.model_spec.max_position as usize,
                "Clover prompt plus response canvases exceeds capacity"
            );
            let tail = ids.len() % 32;
            state
                .forward
                .prepare_causal_rows(&self.device, if tail == 0 { 32 } else { tail as u32 })?;
        }
        for chunk in ids.chunks(canvas) {
            self.diffusion_encoder_chunk(chunk)?;
        }
        self.canvas_step()?;
        self.diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .pending
            .pop_front()
            .context("canvas commit produced no tokens")
    }

    fn diffusion_encoder_chunk(&mut self, ids: &[u32]) -> Result<()> {
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        let rows = ids.len();
        ensure!(rows <= runtime.canvas, "encoder chunk exceeds the canvas");
        let state = &mut runtime.state;
        #[cfg(all(test, feature = "cuda"))]
        let _range = crate::backend::cuda::profile::Range::new(
            state.capture.is_some(),
            c"clover_causal_state_only",
        );
        state.forward.run(&self.device, ids, false)?;
        runtime.prefix += rows;
        self.position = runtime.prefix;
        #[cfg(test)]
        self.capture_clover("causal", Some(ids))?;
        Ok(())
    }

    fn diffusion_canvas(&self) -> Result<usize> {
        Ok(self
            .diffusion
            .as_ref()
            .context("diffusion runtime missing")?
            .canvas)
    }

    fn canvas_forward(&mut self) -> Result<()> {
        let state = &mut self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .state;
        #[cfg(test)]
        if state.policy.is_some() {
            return Ok(());
        }
        #[cfg(all(test, feature = "cuda"))]
        let _range = crate::backend::cuda::profile::Range::new(
            state.capture.is_some(),
            c"clover_preview_head",
        );
        state.forward.preview(&self.device, &state.sampler.block)?;
        let count = u32::try_from(state.logits.len())?;
        unsafe {
            self.device
                .stream
                .launch_builder(&state.upcast)
                .arg(&state.forward.logits)
                .arg(&mut state.logits)
                .arg(&count)
                .launch(flat(count as usize))?;
        }
        Ok(())
    }

    fn canvas_step(&mut self) -> Result<()> {
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        ensure!(
            runtime.prefix + runtime.canvas <= self.capacity,
            "canvas exceeds KV capacity"
        );
        ensure!(
            !runtime.terminal && runtime.limit.is_some(),
            "Clover generation request is not active"
        );
        runtime.state.sampler.begin_block()?;
        let mut unresolved = 32;
        for _ in 0..32 {
            self.canvas_forward()?;
            #[cfg(test)]
            self.capture_clover("raw", None)?;
            self.canvas_sample()?;
            #[cfg(test)]
            self.capture_clover("fill", None)?;
            let runtime = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?;
            let pending = self
                .device
                .stream
                .clone_dtoh(&runtime.state.sampler.state.slice(3..4))?[0];
            ensure!(
                pending < unresolved,
                "Clover denoise step did not resolve a row"
            );
            unresolved = pending;
            if pending == 0 {
                break;
            }
        }
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        let state = &mut runtime.state;
        ensure!(unresolved == 0, "Clover block unresolved after 32 steps");
        let remaining = runtime
            .limit
            .context("Clover request limit missing")?
            .checked_sub(runtime.produced)
            .context("Clover response exceeded request limit")?;
        state.sampler.commit(u32::try_from(remaining.min(32))?)?;
        let s = &self.device.stream;
        let control = s.clone_dtoh(&state.sampler.state)?;
        let returned = control[1] as usize;
        ensure!(
            returned > 0 && returned <= remaining.min(32),
            "Clover returned extent outside request"
        );
        let tokens: Vec<u32> = s
            .clone_dtoh(&state.sampler.block.slice(..returned))?
            .into_iter()
            .map(|token| token as u32)
            .collect();
        runtime.produced += returned;
        ensure!(
            control[0] as usize == runtime.produced,
            "Clover response count disagrees with driver"
        );
        runtime.terminal = control[2] != 0 || runtime.produced == runtime.limit.unwrap();
        #[cfg(test)]
        self.capture_clover("commit", None)?;
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        let state = &mut runtime.state;
        if !runtime.terminal {
            #[cfg(all(test, feature = "cuda"))]
            let _range = crate::backend::cuda::profile::Range::new(
                state.capture.is_some(),
                c"clover_causal_state_only",
            );
            state
                .forward
                .causal_device(&self.device, &state.sampler.block)?;
            runtime.prefix += 32;
            self.position = runtime.prefix;
            #[cfg(test)]
            self.capture_clover("causal", None)?;
        }
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        runtime.pending.extend(tokens);
        Ok(())
    }

    fn canvas_sample(&mut self) -> Result<()> {
        let state = &mut self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .state;
        #[cfg(test)]
        if let Some(steps) = &mut state.policy {
            let step = steps
                .pop_front()
                .context("Runner FAIL policy fixture exhausted")?;
            let s = &self.device.stream;
            s.memcpy_htod(&step.picks, &mut state.sampler.picks)?;
            for (row, (&pick, &confidence)) in step.picks.iter().zip(&step.confidence).enumerate() {
                let offset = row * self.model_spec.vocab as usize + pick as usize;
                s.memcpy_htod(
                    &[confidence],
                    &mut state.sampler.probabilities.slice_mut(offset..offset + 1),
                )?;
            }
            state.sampler.fill()?;
            ensure!(
                s.clone_dtoh(&state.sampler.block)? == step.block
                    && s.clone_dtoh(&state.sampler.pending)? == step.pending,
                "Clover DEFECT shared-driver policy step differs"
            );
            return Ok(());
        }
        state.sampler.step(&mut state.logits)
    }

    pub(super) fn decode_diffusion(&mut self, graph: bool) -> Result<u32> {
        ensure!(!graph, "Clover canvas graph execution is not bound");
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "canvas state is poisoned; reset is required"
        );
        let empty = {
            let r = self
                .diffusion
                .as_ref()
                .context("diffusion runtime missing")?;
            if r.pending.is_empty() {
                ensure!(!r.terminal, "Clover generation request is complete");
                ensure!(
                    r.prefix + r.canvas <= self.capacity,
                    "canvas exceeds KV capacity"
                );
            }
            r.pending.is_empty()
        };
        if empty {
            self.canvas_step()?;
        }
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        r.pending
            .pop_front()
            .context("canvas commit produced no tokens")
    }
}

impl Runtime {
    pub(super) fn reset(&mut self, device: &Device) -> Result<()> {
        device.stream.synchronize()?;
        self.prefix = 0;
        self.pending.clear();
        self.limit = None;
        self.produced = 0;
        self.terminal = false;
        self.state.forward.reset(device)?;
        self.state.sampler.reset()?;
        #[cfg(test)]
        {
            self.state.policy = None;
        }
        Ok(())
    }
}
