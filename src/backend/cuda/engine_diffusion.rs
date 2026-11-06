//! Clover and DiffusionGemma canvas execution.
use super::*;
use crate::backend::cuda::copy::RingCopy;
use crate::backend::cuda::moe::combine::MoeCombineRouted;
use crate::backend::cuda::{
    contracts::{RowDomain, Scratch, SeqRef, Tensor, TensorMut, view},
    cubin,
    moe::{PrefillOps, RouteOps, prefill::MoePrefillContribute},
    ops::{CommonOpsRef, RmsMode, RmsOptions},
    weights::DiffusionWeights,
};
use crate::backend::gpu::{CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use anyhow::{Context, Result, ensure};

/// Tier-table marks: values below HOST_MARK are GPU slot indices.
pub(crate) const HOST_MARK: u32 = 1 << 30;
/// Values at or above DISK_MARK index the disk-source records.
pub(crate) const DISK_MARK: u32 = 1 << 31;

/// Host-tier staging slots: experts of one staged wave live here between
/// the H2D copies and the wave's grouped GEMMs (16 x 11.9 MB = 190 MB).
pub(crate) const STAGING_SLOTS: usize = 16;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use half::bf16;
use std::collections::VecDeque;
use std::sync::Arc;

/// Canvas-step state machine: the prompt pass, then canvases of `canvas`
/// committed tokens each.
pub(crate) struct Runtime {
    program: Program,
    prefix: usize,
    pending: VecDeque<u32>,
    canvas: usize,
    limit: Option<usize>,
    produced: usize,
    terminal: bool,
}

enum Program {
    Gemma(Box<GemmaState>),
    Clover(Box<CloverState>),
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

pub(crate) struct GemmaState {
    weights: DiffusionWeights,
    rope_tables: [CudaSlice<bf16>; 2],
    flash_sliding_canvas: Flash,
    flash_sliding_commit: Flash,
    flash_full: Flash,
    gemm: CudaBlas,
    route: RouteOps,
    moe: PrefillOps,
    rope_sliding: cudarc::driver::CudaFunction,
    rope_shared: cudarc::driver::CudaFunction,
    softcap_f32: cudarc::driver::CudaFunction,
    residual_scale: cudarc::driver::CudaFunction,
    softmax: cudarc::driver::CudaFunction,
    accept_scatter: cudarc::driver::CudaFunction,
    scaled_embed: cudarc::driver::CudaFunction,
    temperature: cudarc::driver::CudaFunction,
    entropy: cudarc::driver::CudaFunction,
    accept: cudarc::driver::CudaFunction,
    philox: cudarc::driver::CudaFunction,
    categorical: cudarc::driver::CudaFunction,
    stop: cudarc::driver::CudaFunction,
    argmax: cudarc::driver::CudaFunction,
    common: CommonOpsRef,
    unit: CudaSlice<bf16>,
    ones: CudaSlice<u8>,
    meta: CudaSlice<i32>,
    // 256-row canvas workspace.
    x: CudaSlice<bf16>,
    x2: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    qkv: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    norm_sum: CudaSlice<bf16>,
    dense_gu: CudaSlice<bf16>,
    dense_act: CudaSlice<bf16>,
    dense_out: CudaSlice<bf16>,
    x_dense: CudaSlice<bf16>,
    x_experts: CudaSlice<bf16>,
    dense_normed: CudaSlice<bf16>,
    routed_normed: CudaSlice<bf16>,
    summed: CudaSlice<bf16>,
    prescale_out: CudaSlice<bf16>,
    route_logits: CudaSlice<bf16>,
    route_ids: CudaSlice<i32>,
    route_weights: CudaSlice<f32>,
    routed: CudaSlice<bf16>,
    cohort_scratch: CudaSlice<u8>,
    /// Disk-tier source mapping, owned for the Runtime's lifetime. Declared
    /// before every field that holds pointers into its mappings.
    mapped: Option<crate::checkpoint::MappedCheckpoint>,
    expert_gpu_slots: CudaSlice<bf16>,
    expert_host: Vec<Vec<bf16>>,
    expert_disk: Vec<(*const u8, usize, *const u8, usize)>,
    expert_tier: Vec<u32>,
    slot_generation: CudaSlice<u64>,
    route_active: CudaSlice<u32>,
    route_gen: CudaSlice<u64>,
    staging_gu: CudaSlice<bf16>,
    staging_dn: CudaSlice<bf16>,
    wave_slots_gu: CudaSlice<u64>,
    wave_slots_dn: CudaSlice<u64>,
    wave_valid: CudaSlice<u32>,
    wave_active: CudaSlice<u32>,
    shared_dummy_gate: CudaSlice<bf16>,
    shared_dummy_down: CudaSlice<bf16>,
    contributions: CudaSlice<bf16>,
    route_done: CudaSlice<u32>,
    head_bf16: CudaSlice<bf16>,
    pub(crate) logits: CudaSlice<f32>,
    processed: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    pub(crate) soft: CudaSlice<bf16>,
    sc_pre: CudaSlice<bf16>,
    sc_emb: CudaSlice<bf16>,
    sc_act: CudaSlice<bf16>,
    sc_down: CudaSlice<bf16>,
    sc_summed: CudaSlice<bf16>,
    sc_post: CudaSlice<bf16>,
    embed_x: CudaSlice<bf16>,
    pub(crate) entropies: CudaSlice<f32>,
    pub(crate) accept_mask: CudaSlice<u8>,
    pub(crate) argmax_cur: CudaSlice<i32>,
    argmax_prev: CudaSlice<i32>,
    picks: CudaSlice<i32>,
    renoise: CudaSlice<i32>,
    canvas_ids: CudaSlice<u32>,
    commit_ids: CudaSlice<u32>,
    stop_flag: CudaSlice<u8>,
    stage_k: CudaSlice<bf16>,
    stage_v: CudaSlice<bf16>,
    zero: CudaSlice<bf16>,
    position: CudaSlice<i32>,
    decoder_scalars_host: Vec<f32>,
    encoder_scalars_host: Vec<f32>,
    denoise_steps: u32,
    t_min: f32,
    t_max: f32,
    entropy_bound: f32,
    confidence: f32,
    pad: u32,
    /// When false the emitted canvas pads after the first EOS; when true
    /// (generation requested ignore_eos) post-EOS tokens are kept.
    ignore_eos: bool,
    /// Tokens requested by generation_request; prefill validates the full
    /// multi-canvas extent against capacity.
    requested: usize,
    #[cfg(test)]
    pub(crate) tap_capture: Option<Vec<CudaSlice<bf16>>>,
    #[cfg(test)]
    pub(crate) route_taps: Vec<(usize, CudaSlice<i32>, CudaSlice<f32>)>,
    seed: u64,
    draws: u64,
}

fn matrix(
    b: &CudaSlice<bf16>,
    s: &Arc<CudaStream>,
    offset: usize,
    rows: usize,
    width: usize,
) -> Result<Tensor<bf16, 2>> {
    view(b, s, offset, [rows as u32, width as u32], [width as u64, 1])
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
            let Program::Clover(state) = &mut runtime.program else {
                anyhow::bail!("Runner FAIL canonical payload");
            };
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
                    let Program::Clover(state) = &mut runtime.program else {
                        unreachable!();
                    };
                    self.device
                        .stream
                        .memcpy_htod(&tokens, &mut state.sampler.block)?;
                    self.canvas_forward()?;
                    self.capture_clover("raw", None)?;
                }
            }
            let runtime = self.diffusion.as_ref().context("canvas runtime missing")?;
            let Program::Clover(state) = &runtime.program else {
                unreachable!();
            };
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
            let Program::Clover(state) = &mut runtime.program else {
                unreachable!();
            };
            restore(&mut state.sampler, &fixture, case, &serde_json::Value::Null)?;
            self.device.stream.memcpy_htod(&raw, &mut state.logits)?;
            self.canvas_sample(1)?;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let Program::Clover(state) = &mut runtime.program else {
                unreachable!();
            };
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
            let Program::Clover(state) = &mut runtime.program else {
                unreachable!();
            };
            state.forward.prepare_causal_rows(&self.device, 3)?;
            let literal_mask =
                self.device
                    .stream
                    .clone_dtoh(&state.sampler.block.slice(..1))?[0] as u32;
            self.diffusion_encoder_chunk(&[literal_mask, 17, 17])?;
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let Program::Clover(state) = &mut runtime.program else {
                unreachable!();
            };
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
                    self.canvas_step_forced(None, None)?;
                    let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
                    returned.extend(runtime.pending.drain(..));
                    if runtime.terminal {
                        break;
                    }
                }
                Ok(())
            })();
            let runtime = self.diffusion.as_mut().context("canvas runtime missing")?;
            let Program::Clover(state) = &mut runtime.program else {
                unreachable!();
            };
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
        let Program::Clover(state) = &mut runtime.program else {
            return Ok(());
        };
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
        if self.diffusion.is_some() {
            ensure!(!graph, "canvas graph execution is not bound");
        }
        if let Some(runtime) = &mut self.diffusion
            && let Program::Clover(_) = &runtime.program
        {
            ensure!(
                count > 0 && count <= self.model_spec.max_position as usize,
                "Clover response limit outside model capacity"
            );
            ensure!(
                !ignore_eos,
                "Clover generation requires the published EOS policy"
            );
            ensure!(
                self.position == 0 && runtime.limit.is_none(),
                "Clover generation requires reset"
            );
            runtime.limit = Some(count);
            runtime.produced = 0;
            runtime.terminal = false;
            #[cfg(test)]
            if let Program::Clover(state) = &mut runtime.program
                && let Some(capture) = &mut state.capture
            {
                capture.request += 1;
                capture.calls.clear();
                capture.committed.clear();
                capture.control = false;
                capture.control_attempted = false;
                capture.defects.clear();
                std::fs::create_dir(capture.directory())?;
            }
        }
        if let Some(runtime) = &mut self.diffusion
            && let Program::Gemma(state) = &mut runtime.program
        {
            state.ignore_eos = ignore_eos;
            state.requested = count;
            // Necessary condition at request time (the prompt length is only
            // known at prefill, which carries the full extent check).
            let canvas = self.diffusion_canvas()?;
            let canvases = count.div_ceil(canvas);
            let span = canvases
                .checked_mul(canvas)
                .context("canvas extent overflow")?;
            ensure!(
                span <= self.capacity,
                "requested canvases exceed KV capacity {} ({} x {} canvas)",
                self.capacity,
                canvases,
                canvas
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn generation_request_for_test(
        &mut self,
        count: usize,
        ignore_eos: bool,
    ) -> Result<()> {
        self.generation_request(count, ignore_eos, false)
    }

    #[cfg(test)]
    pub(crate) fn diffusion_requested_for_test(&self) -> Option<usize> {
        self.diffusion.as_ref().and_then(|runtime| {
            if let Program::Gemma(state) = &runtime.program {
                Some(state.requested)
            } else {
                None
            }
        })
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
            program: Program::Clover(Box::new(CloverState {
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
            })),
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
            "Canvas setup".into(),
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
            catalog: None,
            dense_graphs: Vec::new(),
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
            #[cfg(test)]
            diffusion_single_step: false,
            #[cfg(test)]
            diffusion_forced_draws: None,
        };
        engine.closure.freeze_fixed()?;
        engine.reset()?;
        engine.closure.ready(started)?;
        Ok(engine)
    }

    pub(super) fn load_diffusion(
        dir: &std::path::Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        model_spec: crate::model_spec::ModelSpec,
    ) -> Result<Self> {
        if model_spec.architecture == crate::model_spec::Architecture::Clover {
            return Self::load_clover(
                dir,
                ordinal,
                capacity,
                requested_chunk,
                settings,
                model_spec,
            );
        }
        let started = std::time::Instant::now();
        let canvas = model_spec
            .canvas_length
            .context("DiffusionGemma canvas length missing")? as usize;
        ensure!(
            requested_chunk.is_none_or(|n| n == canvas),
            "DiffusionGemma commit chunk is the canvas length {canvas}"
        );
        ensure!(
            canvas < capacity && capacity <= model_spec.max_position as usize,
            "DiffusionGemma capacity must cover one canvas beyond the prompt"
        );
        ensure!(
            settings.draft_model.is_none() && !settings.prefix_cache,
            "DiffusionGemma does not support draft or prefix-cache execution"
        );
        let diffusion_cfg = model_spec
            .diffusion
            .clone()
            .context("DiffusionGemma sampler schedule missing")?;
        let device = Device::new(ordinal)?;
        let s = &device.stream;
        let (free, _total) = device.ctx.mem_get_info()?;
        let mapped = crate::checkpoint::MappedCheckpoint::open(dir)?;
        let manifest = mapped.manifest(&model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
        let hidden = model_spec.hidden as usize;
        let vocab = model_spec.vocab as usize;
        let moe = model_spec
            .moe
            .as_ref()
            .context("DiffusionGemma MoE missing")?;
        let expert_inter = moe.expert_intermediate as usize;
        let weights = DiffusionWeights::load(dir, &model_spec, &device)?;
        let ring = model_spec
            .layers
            .iter()
            .filter_map(|l| l.sliding_window)
            .map(|window| window as usize + canvas - 1)
            .max()
            .context("sliding window missing")?;
        // Per-layer projection widths: sliding packs q|k|v, full packs q|k.
        let (mut qkv_width, mut q_width, mut kv_width) = (0usize, 0usize, 0usize);
        for layer in &model_spec.layers {
            let geometry = layer.geometry.context("layer geometry missing")?;
            let kvh = geometry.kv_heads.unwrap_or(model_spec.kv_heads) as usize;
            qkv_width = qkv_width.max(
                model_spec.q_heads as usize * geometry.head_dim as usize
                    + kvh
                        * geometry.head_dim as usize
                        * (1 + usize::from(layer.sliding_window.is_some())),
            );
            q_width = q_width.max(model_spec.q_heads as usize * geometry.head_dim as usize);
            kv_width = kv_width.max(kvh * geometry.head_dim as usize);
        }
        let mut tables = Vec::new();
        for full in [false, true] {
            let geometry = model_spec
                .layers
                .iter()
                .find_map(|l| l.geometry.filter(|g| (g.head_dim == 512) == full))
                .context("per-layer rope geometry missing")?;
            tables.push(s.clone_htod(&geometry.rope.table(geometry.head_dim, capacity)?)?);
        }
        let rope_tables: [CudaSlice<bf16>; 2] = tables
            .try_into()
            .map_err(|_| anyhow::anyhow!("two rope tables required"))?;
        let flash_sliding_canvas = Flash::new_with_score(
            s.clone(),
            canvas,
            model_spec.q_heads as usize,
            model_spec.kv_heads as usize,
            256,
            crate::backend::cuda::flash::Score::from_tokens(1.0, None)?,
            ring,
        )?;
        let flash_sliding_commit = Flash::new_with_score(
            s.clone(),
            canvas,
            model_spec.q_heads as usize,
            model_spec.kv_heads as usize,
            256,
            // from_tokens takes window TOKENS (minus one inside); the commit window
            // in left-inclusive tokens is the sliding_window itself.
            crate::backend::cuda::flash::Score::from_tokens(1.0, Some(ring - canvas + 1))?,
            ring + canvas,
        )?;
        let flash_full = Flash::new_with_score(
            s.clone(),
            canvas,
            model_spec.q_heads as usize,
            2,
            512,
            crate::backend::cuda::flash::Score::from_tokens(1.0, None)?,
            capacity,
        )?;
        let rope = cubin::module(&device.ctx, "rope")?;
        let route_ops = RouteOps::new(&device.ctx, s, moe.experts, canvas)?;
        let scratch_bytes =
            crate::backend::cuda::moe::prefill::prefill_scratch_bytes_mixed_with_boundary(
                canvas,
                canvas,
                moe.experts_per_tok as usize,
                moe.experts as usize,
                hidden,
                (expert_inter, expert_inter),
                (
                    None,
                    crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
                ),
            )?;
        let moe_scratch = s.alloc_zeros::<u8>(scratch_bytes as usize)?;
        let moe_ops = crate::backend::cuda::moe::PrefillOps::new_with_boundaries(
            &device.ctx,
            s,
            crate::model_spec::ExpertFormat::Bf16,
            crate::model_spec::SwigluRounding::GeluTwice,
            crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
        )?;

        let gemm = CudaBlas::new(s.clone())?;
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *gemm.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()?;
        }
        let common = CommonOpsRef::new(&device.ctx)?;
        let act = cubin::module(&device.ctx, "act")?;
        let sample = cubin::module(&device.ctx, "sample")?;
        let embed = cubin::module(&device.ctx, "embed")?;

        let ones = vec![1u8; canvas];
        let prev = vec![-1i32; canvas];
        let intermediate = model_spec.intermediate as usize;
        let read_scalar = |slice: &CudaSlice<bf16>| -> Result<f32> {
            let host = s.clone_dtoh(&slice.slice(..1))?;
            Ok(host[0].to_f32())
        };
        let mut decoder_scalars_host = Vec::with_capacity(model_spec.layers.len());
        let mut encoder_scalars_host = Vec::with_capacity(model_spec.layers.len());
        for i in 0..model_spec.layers.len() {
            decoder_scalars_host.push(read_scalar(&weights.layers[i].layer_scalar)?);
            encoder_scalars_host.push(read_scalar(&weights.encoder_scalars[i])?);
        }
        let mut runtime = GemmaState {
            weights,
            rope_tables,
            flash_sliding_canvas,
            flash_sliding_commit,
            flash_full,
            gemm,
            route: route_ops,
            moe: moe_ops,
            rope_sliding: rope.load_function("qknorm_rope_table_kv")?,
            rope_shared: rope.load_function("qknorm_rope_table_shared_kv")?,
            softcap_f32: act.load_function("softcap_cast_f32")?,
            residual_scale: act.load_function("residual_scale")?,
            softmax: sample.load_function("softmax_rows_dual")?,
            accept_scatter: sample.load_function("canvas_accept_scatter")?,
            scaled_embed: embed.load_function("embed_scale_broadcast")?,
            temperature: sample.load_function("temperature_scale_f32")?,
            entropy: sample.load_function("row_entropy_f32")?,
            accept: sample.load_function("entropy_bound_accept")?,
            philox: sample.load_function("philox_uniform_ids")?,
            categorical: sample.load_function("categorical_sample_rows")?,
            stop: sample.load_function("canvas_stability_stop")?,
            argmax: sample.load_function("argmax_rows_f32")?,
            common,
            unit: s.clone_htod(&vec![bf16::ONE; hidden])?,
            ones: s.clone_htod(&ones)?,
            meta: s.clone_htod(&[
                0i32,
                canvas as i32,
                canvas as i32,
                canvas as i32,
                1,
                0,
                0,
                0,
            ])?,
            x: s.alloc_zeros(canvas * hidden)?,
            x2: s.alloc_zeros(canvas * hidden)?,
            n: s.alloc_zeros(canvas * hidden)?,
            qkv: s.alloc_zeros(canvas * qkv_width)?,
            attn: s.alloc_zeros(canvas * q_width)?,
            o: s.alloc_zeros(canvas * hidden)?,
            norm_sum: s.alloc_zeros(canvas * hidden)?,
            dense_gu: s.alloc_zeros(canvas * 2 * intermediate)?,
            dense_act: s.alloc_zeros(canvas * intermediate)?,
            dense_out: s.alloc_zeros(canvas * hidden)?,
            x_dense: s.alloc_zeros(canvas * hidden)?,
            x_experts: s.alloc_zeros(canvas * hidden)?,
            dense_normed: s.alloc_zeros(canvas * hidden)?,
            routed_normed: s.alloc_zeros(canvas * hidden)?,
            summed: s.alloc_zeros(canvas * hidden)?,
            prescale_out: s.alloc_zeros(canvas * hidden)?,
            route_logits: s.alloc_zeros(canvas * moe.experts as usize)?,
            route_ids: s.alloc_zeros(canvas * moe.experts_per_tok as usize)?,
            route_weights: s.alloc_zeros(canvas * moe.experts_per_tok as usize)?,
            routed: s.alloc_zeros(canvas * hidden)?,
            cohort_scratch: moe_scratch,
            mapped: None,
            expert_gpu_slots: s.alloc_zeros(0)?,
            expert_host: Vec::new(),
            expert_disk: Vec::new(),
            expert_tier: Vec::new(),
            slot_generation: s.clone_htod(&vec![0u64; moe.experts as usize])?,
            route_active: s.clone_htod(&vec![1u32; canvas * moe.experts_per_tok as usize])?,
            route_gen: s.clone_htod(&vec![0u64; canvas * moe.experts_per_tok as usize])?,
            staging_gu: s.alloc_zeros(0)?,
            staging_dn: s.alloc_zeros(0)?,
            wave_slots_gu: s.clone_htod(&vec![0u64; moe.experts as usize])?,
            wave_slots_dn: s.clone_htod(&vec![0u64; moe.experts as usize])?,
            wave_valid: s.clone_htod(&vec![0u32; moe.experts as usize])?,
            wave_active: s.alloc_zeros(canvas * moe.experts_per_tok as usize)?,
            shared_dummy_gate: s.alloc_zeros(expert_inter * hidden)?,
            shared_dummy_down: s.alloc_zeros(hidden * expert_inter)?,
            contributions: s.alloc_zeros(canvas * moe.experts_per_tok as usize * hidden)?,
            route_done: s.alloc_zeros(canvas * moe.experts_per_tok as usize)?,
            head_bf16: s.alloc_zeros(canvas * vocab)?,
            logits: s.alloc_zeros(canvas * vocab)?,
            processed: s.alloc_zeros(canvas * vocab)?,
            probs: s.alloc_zeros(canvas * vocab)?,
            soft: s.alloc_zeros(canvas * vocab)?,
            sc_pre: s.alloc_zeros(canvas * hidden)?,
            sc_emb: s.alloc_zeros(canvas * hidden)?,
            sc_act: s.alloc_zeros(canvas * intermediate)?,
            sc_down: s.alloc_zeros(canvas * hidden)?,
            sc_summed: s.alloc_zeros(canvas * hidden)?,
            sc_post: s.alloc_zeros(canvas * hidden)?,
            embed_x: s.alloc_zeros(canvas * hidden)?,
            entropies: s.alloc_zeros(canvas)?,
            accept_mask: s.clone_htod(&vec![0u8; canvas])?,
            argmax_cur: s.clone_htod(&prev)?,
            argmax_prev: s.clone_htod(&prev)?,
            picks: s.alloc_zeros(canvas)?,
            renoise: s.alloc_zeros(canvas)?,
            canvas_ids: s.alloc_zeros(canvas)?,
            commit_ids: s.alloc_zeros(canvas)?,
            stop_flag: s.alloc_zeros(1)?,
            stage_k: s.alloc_zeros(kv_width * (ring + canvas))?,
            stage_v: s.alloc_zeros(kv_width * (ring + canvas))?,
            zero: s.alloc_zeros(canvas * hidden)?,
            position: s.clone_htod(&[0i32])?,
            decoder_scalars_host,
            encoder_scalars_host,
            denoise_steps: diffusion_cfg.denoise_steps,
            t_min: diffusion_cfg.t_min,
            t_max: diffusion_cfg.t_max,
            entropy_bound: diffusion_cfg.entropy_bound,
            confidence: diffusion_cfg.confidence,
            pad: diffusion_cfg.pad,
            ignore_eos: false,
            requested: 0,
            #[cfg(test)]
            tap_capture: None,
            #[cfg(test)]
            route_taps: Vec::new(),
            seed: 0,
            draws: 0,
        };
        let _ = &runtime.rope_tables;
        let state = super::super::state::StateStore::new(&model_spec, capacity, s)?;
        let work = super::Work::new(s, &model_spec, canvas)?;
        let workspace_bytes = runtime
            .workspace_bytes()?
            .checked_add(state.bytes() as u64)
            .and_then(|n| n.checked_add(work.bytes() as u64))
            .and_then(|n| n.checked_add(48 << 20))
            .context("canvas fixed workspace bytes overflow")?;
        // Shared model-tier planner: per-expert GPU slots, a cgroup-bounded
        // host tier and the disk remainder, from the counted budget.
        // The declared budget constrains the derivation; unset senses free.
        let device_budget = settings
            .device_budget_gib
            .map(|gib| gib << 30)
            .unwrap_or(free as u64)
            .min(free as u64);
        let resources = crate::experts::plan::BlockResources {
            device_free_bytes: device_budget,
            host_available_bytes: crate::backend::cuda::calibrate::host_memory(
                std::path::Path::new("/"),
            )?,
            device_work_bytes: workspace_bytes,
            host_work_bytes: 256 << 20,
            compute_units: u32::try_from(device.info.sms)?,
            copy_engines: 2,
            rows: canvas as u32,
            staging_slots_max: STAGING_SLOTS as u32,
        };
        let residency =
            crate::experts::plan::BlockResidency::new(&model_spec, &manifest, resources)?;
        ensure!(
            residency.dense.host_bytes == 0 && residency.dense.disk_bytes == 0,
            "DiffusionGemma forward requires the derived dense-resident placement"
        );
        println!(
            "DIFFUSION_WORKSPACE fixed_bytes={workspace_bytes} staging_slots={} budget_bytes={device_budget}",
            residency.experts.prefill_staging_slots
        );
        let slots_per_layer = residency.experts.slots_per_layer.clone();
        let host_blocks = residency.experts.host_blocks;
        let slot_elems = 2 * expert_inter * hidden + hidden * expert_inter;
        let slots_total: u32 = slots_per_layer.iter().sum();
        let mut expert_gpu_slots = s.alloc_zeros::<bf16>(slots_total as usize * slot_elems)?;
        let (gpu_base, _) = expert_gpu_slots.device_ptr(s);
        let mut expert_slot_prefix = Vec::with_capacity(slots_per_layer.len() + 1);
        let mut running = 0u32;
        expert_slot_prefix.push(0);
        for slots in &slots_per_layer {
            running += slots;
            expert_slot_prefix.push(running);
        }
        let views = mapped.views()?;
        // One fused expert tensor spans every expert of the layer; each tier
        // source slices its own expert out of the mapping. gate_up holds
        // experts x [2*inter, hidden]; down holds experts x [hidden, inter].
        let gu_span = 2 * expert_inter * hidden * 2;
        let dn_span = hidden * expert_inter * 2;
        let expert_tensor = |name: &str| -> Result<(*const u8, usize)> {
            let entry = manifest
                .entries
                .get(name)
                .with_context(|| format!("expert tensor {name} missing from the manifest"))?;
            let shard = views
                .get(&entry.shard)
                .with_context(|| format!("expert shard {} missing", entry.shard))?;
            let data = shard.tensor(name)?.data();
            let (pointer, length) = (data.as_ptr(), data.len());
            ensure!(
                length == moe.experts as usize * gu_span
                    || length == moe.experts as usize * dn_span,
                "expert tensor byte count mismatch for {name}: {length}"
            );
            Ok((pointer, length))
        };
        let mut expert_host: Vec<Vec<bf16>> = Vec::new();
        let mut expert_disk = Vec::new();
        // Per-(layer, expert) tier: the GPU slot index, HOST_MARK plus the
        // arena index, or DISK_MARK plus the disk-record index.
        let mut expert_tier = Vec::with_capacity(moe.experts as usize * model_spec.layers.len());
        let mut host_used = 0u32;
        let mut slot_fill = 0u32;
        for (layer, slots) in slots_per_layer.iter().enumerate() {
            for e in 0..moe.experts as usize {
                let name_gu = format!("model.decoder.layers.{layer}.experts.gate_up_proj");
                let name_dn = format!("model.decoder.layers.{layer}.experts.down_proj");
                let (gu_ptr, _gu_len) = expert_tensor(&name_gu)?;
                let (dn_ptr, _dn_len) = expert_tensor(&name_dn)?;
                let gu_ptr = gu_ptr as usize + e * gu_span;
                let dn_ptr = dn_ptr as usize + e * dn_span;
                let gu_ptr = gu_ptr as *const u8;
                let dn_ptr = dn_ptr as *const u8;
                let (gu_len, dn_len) = (gu_span, dn_span);
                if (e as u32) < *slots {
                    expert_tier.push(slot_fill);
                    let offset = slot_fill as usize * slot_elems;
                    // Slot layout: gate_up bytes then down bytes.
                    unsafe {
                        let gu_slice =
                            std::slice::from_raw_parts(gu_ptr as *const bf16, gu_len / 2);
                        let dn_slice =
                            std::slice::from_raw_parts(dn_ptr as *const bf16, dn_len / 2);
                        s.memcpy_htod(
                            gu_slice,
                            &mut expert_gpu_slots.slice_mut(offset..offset + gu_len / 2),
                        )?;
                        s.memcpy_htod(
                            dn_slice,
                            &mut expert_gpu_slots
                                .slice_mut(offset + gu_len / 2..offset + slot_elems),
                        )?;
                    }
                    slot_fill += 1;
                } else if host_used < host_blocks {
                    expert_tier.push(HOST_MARK | host_used);
                    let mut block = Vec::with_capacity(slot_elems);
                    unsafe {
                        block.extend_from_slice(std::slice::from_raw_parts(
                            gu_ptr as *const bf16,
                            gu_len / 2,
                        ));
                        block.extend_from_slice(std::slice::from_raw_parts(
                            dn_ptr as *const bf16,
                            dn_len / 2,
                        ));
                    }
                    expert_host.push(block);
                    host_used += 1;
                } else {
                    expert_tier.push(DISK_MARK | (expert_disk.len() as u32));
                    expert_disk.push((gu_ptr, gu_len, dn_ptr, dn_len));
                }
            }
        }
        ensure!(
            slot_fill == slots_total
                && host_used == host_blocks
                && expert_disk.len()
                    == moe.experts as usize * model_spec.layers.len()
                        - slots_total as usize
                        - host_blocks as usize,
            "DiffusionGemma tier assignment lost experts: slots={slot_fill}/{slots_total} host={host_used}/{host_blocks} disk={}",
            expert_disk.len()
        );
        println!(
            "DIFFUSION_TIER_PLAN slots_total={slots_total} slots_per_layer_max={} host_blocks={host_blocks} disk_experts={} slot_bytes={} dense_resident={} dense_host={} dense_disk={}",
            slots_per_layer.iter().copied().max().unwrap_or(0),
            expert_disk.len(),
            residency.experts.slot_bytes(),
            residency.dense.resident_bytes,
            residency.dense.host_bytes,
            residency.dense.disk_bytes,
        );
        let _ = gpu_base;
        runtime.mapped = Some(mapped);
        runtime.expert_gpu_slots = expert_gpu_slots;
        runtime.expert_host = expert_host;
        runtime.expert_disk = expert_disk;
        runtime.expert_tier = expert_tier;
        let staging = residency.experts.prefill_staging_slots as usize;
        runtime.staging_gu = s.alloc_zeros(staging * 2 * expert_inter * hidden)?;
        runtime.staging_dn = s.alloc_zeros(staging * hidden * expert_inter)?;
        let runtime = Runtime {
            program: Program::Gemma(Box::new(runtime)),
            prefix: 0,
            pending: VecDeque::new(),
            canvas,
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
}

/// y[r, o] = x[r, :] @ w[o, :]^T over BF16 with w a [o, i] row-major slice.
fn gemm_t<W, X, Y>(
    gemm: &CudaBlas,
    w: &W,
    x: &X,
    y: &mut Y,
    r: usize,
    o: usize,
    i: usize,
) -> Result<()>
where
    W: crate::backend::gpu::DevicePtr<bf16>,
    X: crate::backend::gpu::DevicePtr<bf16>,
    Y: crate::backend::gpu::DevicePtrMut<bf16>,
{
    let config = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: o as i32,
        n: r as i32,
        k: i as i32,
        alpha: bf16::ONE,
        lda: i as i32,
        ldb: i as i32,
        beta: bf16::ZERO,
        ldc: o as i32,
    };
    unsafe { gemm.gemm(config, w, x, y)? };
    Ok(())
}

/// y[r, o] = x[r, :] @ w[r-facing, :] with w an [i, o] row-major slice
/// (the soft-embedding leg: probs @ embed).
fn gemm_n<W, X, Y>(
    gemm: &CudaBlas,
    w: &W,
    x: &X,
    y: &mut Y,
    r: usize,
    o: usize,
    i: usize,
) -> Result<()>
where
    W: crate::backend::gpu::DevicePtr<bf16>,
    X: crate::backend::gpu::DevicePtr<bf16>,
    Y: crate::backend::gpu::DevicePtrMut<bf16>,
{
    let config = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: o as i32,
        n: r as i32,
        k: i as i32,
        alpha: bf16::ONE,
        lda: o as i32,
        ldb: i as i32,
        beta: bf16::ZERO,
        ldc: o as i32,
    };
    unsafe { gemm.gemm(config, w, x, y)? };
    Ok(())
}

/// Shared context of the canvas norm calls: the op set, the row-metadata
/// device pointer and the model epsilon.
pub(super) struct NormCtx<'a> {
    common: &'a CommonOpsRef,
    meta: u64,
    eps: f32,
}

fn rms(
    s: &Arc<CudaStream>,
    ctx: &NormCtx<'_>,
    input: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    rows: usize,
    hidden: usize,
) -> Result<()> {
    let domain = RowDomain {
        meta: SeqRef { ptr: ctx.meta },
        row_base: 0,
        row_capacity: rows as u32,
    };
    ctx.common.norm.rms_with(
        s,
        domain,
        view(
            input,
            s,
            0,
            [rows as u32, 1, hidden as u32],
            [hidden as u64, hidden as u64, 1],
        )?,
        view(weight, s, 0, [hidden as u32], [1])?,
        &mut TensorMut(view(
            out,
            s,
            0,
            [rows as u32, 1, hidden as u32],
            [hidden as u64, hidden as u64, 1],
        )?),
        RmsOptions {
            mode: RmsMode::Unrounded,
            eps: ctx.eps,
        },
    )
}

fn gelu_split(
    s: &Arc<CudaStream>,
    gate: &crate::backend::cuda::ops::GateOps,
    gu: &CudaSlice<bf16>,
    act: &mut CudaSlice<bf16>,
    rows: usize,
    inter: usize,
) -> Result<()> {
    gate.gelu_tanh_mul(
        s,
        view(
            gu,
            s,
            0,
            [rows as u32, 1, inter as u32],
            [(2 * inter) as u64, inter as u64, 1],
        )?,
        view(
            gu,
            s,
            inter,
            [rows as u32, 1, inter as u32],
            [(2 * inter) as u64, inter as u64, 1],
        )?,
        &mut TensorMut(view(
            act,
            s,
            0,
            [rows as u32, 1, inter as u32],
            [inter as u64, inter as u64, 1],
        )?),
    )
}

/// Bidirectional canvas read extent over the sliding ring: the window
/// [prefix-1023, prefix+canvas) occupies every ring slot exactly once once
/// prefix >= ring - canvas; earlier canvases see fewer keys.
pub(crate) fn canvas_window(prefix: usize, canvas: usize, ring: usize) -> (usize, usize) {
    let keys = (prefix + canvas).min(ring);
    (keys, keys - canvas)
}

/// Staged windowed-causal pass: the history extent (the visible prefix — the
/// canvas denoise writes already clobber the ring's oldest slots), the
/// copy-back length and its source/destination slots.
pub(crate) fn commit_window(
    prefix: usize,
    rows: usize,
    ring: usize,
    canvas: usize,
) -> (usize, usize, usize, usize) {
    let history = prefix.min(ring - canvas);
    let end = history + rows;
    let keep = end.min(ring);
    (history, keep, end - keep, prefix + rows - keep)
}

/// One layer's routed execution over the three tiers: resident experts
/// contribute once over the GPU slot table; host and disk experts stream
/// through the staging slots in fixed waves. Done bits accumulate across
/// cohorts; the caller joins once.
pub(crate) struct TieredCall<'a> {
    pub ops: &'a crate::backend::cuda::moe::PrefillOps,
    pub stream: &'a Arc<CudaStream>,
    pub scratch: Scratch,
    pub gpu_slots: &'a CudaSlice<bf16>,
    pub tiers: &'a [u32],
    pub host_arena: &'a [Vec<bf16>],
    pub disk_sources: &'a [(*const u8, usize, *const u8, usize)],
    pub staging_gu: &'a mut CudaSlice<bf16>,
    pub staging_dn: &'a mut CudaSlice<bf16>,
    /// Planned staging slot count; waves are sized by it, not by a constant.
    pub staging_slots: usize,
    pub wave_slots_gu: &'a mut CudaSlice<u64>,
    pub wave_slots_dn: &'a mut CudaSlice<u64>,
    pub wave_valid: &'a mut CudaSlice<u32>,
    pub wave_active: &'a mut CudaSlice<u32>,
    pub x_experts: &'a CudaSlice<bf16>,
    pub ids: &'a CudaSlice<i32>,
    pub ids_host: &'a [i32],
    pub weights: &'a CudaSlice<f32>,
    pub generation: &'a CudaSlice<u64>,
    pub slot_generation: &'a CudaSlice<u64>,
    pub shared_gate: &'a CudaSlice<bf16>,
    pub shared_down: &'a CudaSlice<bf16>,
    pub contributions: &'a mut CudaSlice<bf16>,
    pub done: &'a mut CudaSlice<u32>,
    pub rows: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub expert_inter: usize,
}

pub(crate) fn contribute_tiered(call: TieredCall<'_>) -> Result<()> {
    let TieredCall {
        ops,
        stream,
        scratch,
        gpu_slots,
        tiers,
        host_arena,
        disk_sources,
        staging_gu,
        staging_dn,
        staging_slots,
        wave_slots_gu,
        wave_slots_dn,
        wave_valid,
        wave_active,
        x_experts,
        ids,
        ids_host,
        weights,
        generation,
        slot_generation,
        shared_gate,
        shared_down,
        contributions,
        done,
        rows,
        experts,
        top_k,
        hidden,
        expert_inter,
    } = call;
    ensure!(
        tiers.len() == experts && ids_host.len() == rows * top_k,
        "tiered extents mismatch rows={rows} experts={experts}"
    );
    let gu_stride = 2 * expert_inter * hidden;
    let dn_stride = hidden * expert_inter;
    let slot_elems = gu_stride + dn_stride;
    ensure!(
        gpu_slots.len() % slot_elems == 0,
        "gpu slot extent mismatch"
    );
    let (gpu_base, _) = gpu_slots.device_ptr(stream);
    let (stg_gu, _) = staging_gu.device_ptr(stream);
    let (stg_dn, _) = staging_dn.device_ptr(stream);
    let mut resident: Vec<usize> = Vec::new();
    let mut staged: Vec<usize> = Vec::new();
    let mut routed = vec![false; tiers.len()];
    for &id in ids_host {
        let id = usize::try_from(id).context("tiered route id negative")?;
        if id < routed.len() {
            routed[id] = true;
        }
    }
    for (expert, tier) in tiers.iter().enumerate() {
        if *tier < HOST_MARK {
            resident.push(expert);
        } else if routed[expert] {
            staged.push(expert);
        }
    }
    let mut run_cohort = |gu_table: &[u64],
                          dn_table: &[u64],
                          valid_table: &[u32],
                          active_table: &[u32]|
     -> Result<()> {
        if active_table.iter().all(|value| *value == 0) {
            return Ok(());
        }
        stream.memcpy_htod(gu_table, wave_slots_gu)?;
        stream.memcpy_htod(dn_table, wave_slots_dn)?;
        stream.memcpy_htod(valid_table, wave_valid)?;
        stream.memcpy_htod(active_table, wave_active)?;
        let weights_view = unsafe {
            crate::backend::cuda::contracts::Tensor::<f32, 2>::new(
                weights.device_ptr(stream).0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
                (rows * top_k) as u64 * 4,
            )?
        };
        let contrib = unsafe {
            crate::backend::cuda::contracts::Tensor::<bf16, 3>::new(
                contributions.device_ptr(stream).0,
                [rows as u32, top_k as u32, hidden as u32],
                [(top_k * hidden) as u64, hidden as u64, 1],
                (rows * top_k * hidden) as u64 * 2,
            )?
        };
        let id_view = view(
            ids,
            stream,
            0,
            [rows as u32, top_k as u32],
            [top_k as u64, 1],
        )?;
        let active_view = unsafe {
            crate::backend::cuda::contracts::Tensor::<u32, 2>::new(
                wave_active.device_ptr(stream).0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
                (rows * top_k) as u64 * 4,
            )?
        };
        let shared = view(
            shared_gate,
            stream,
            0,
            [expert_inter as u32, hidden as u32],
            [hidden as u64, 1],
        )?;
        let input = MoePrefillContribute {
            mask_from_slots: false,
            expert_inter: expert_inter as u32,
            wave: None,
            wait_copy: None,
            reader_event: None,
            cpu_arithmetic: &vec![false; experts],
            fp8_slot: None,
            x: view(
                x_experts,
                stream,
                0,
                [rows as u32, hidden as u32],
                [hidden as u64, 1],
            )?,
            ids: id_view,
            weights: weights_view,
            active: active_view,
            assignment_gen: Some(view(
                generation,
                stream,
                0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
            )?),
            slots_gate_up: view(wave_slots_gu, stream, 0, [experts as u32], [1])?,
            slots_down: view(wave_slots_dn, stream, 0, [experts as u32], [1])?,
            slot_valid: view(wave_valid, stream, 0, [experts as u32], [1])?,
            slot_generation: view(slot_generation, stream, 0, [experts as u32], [1])?,
            slots_gate_up_host: gu_table,
            slots_down_host: dn_table,
            shared_gate: shared,
            shared_up: shared,
            shared_down: view(
                shared_down,
                stream,
                0,
                [hidden as u32, expert_inter as u32],
                [expert_inter as u64, 1],
            )?,
            shared_expert_gate: None,
            contributions: contrib,
            done: TensorMut(view(
                done,
                stream,
                0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
            )?),
        };
        ops.prefill_contribute(stream, input, &scratch)
    };
    let mut gu_table = vec![0u64; experts];
    let mut dn_table = vec![0u64; experts];
    let mut valid_table = vec![0u32; experts];
    let mut active_table = vec![0u32; rows * top_k];
    if !resident.is_empty() {
        for &expert in &resident {
            let slot = tiers[expert] as usize;
            gu_table[expert] = gpu_base + (slot * slot_elems * 2) as u64;
            dn_table[expert] = gpu_base + (slot * slot_elems * 2 + gu_stride * 2) as u64;
            valid_table[expert] = 1;
        }
        for (index, &id) in ids_host.iter().enumerate() {
            let id = usize::try_from(id).context("tiered route id negative")?;
            if valid_table.get(id).is_some_and(|valid| *valid == 1) {
                active_table[index] = 1;
            }
        }
        run_cohort(&gu_table, &dn_table, &valid_table, &active_table)?;
    }
    for wave in staged.chunks(staging_slots.max(1)) {
        gu_table.iter_mut().for_each(|value| *value = 0);
        dn_table.iter_mut().for_each(|value| *value = 0);
        valid_table.iter_mut().for_each(|value| *value = 0);
        active_table.iter_mut().for_each(|value| *value = 0);
        for (slot, &expert) in wave.iter().enumerate() {
            let tier = tiers[expert];
            if tier < DISK_MARK {
                let block = host_arena
                    .get((tier ^ HOST_MARK) as usize)
                    .context("host arena index outside the tier")?;
                stream.memcpy_htod(
                    &block[..gu_stride],
                    &mut staging_gu.slice_mut(slot * gu_stride..slot * gu_stride + gu_stride),
                )?;
                stream.memcpy_htod(
                    &block[gu_stride..],
                    &mut staging_dn.slice_mut(slot * dn_stride..slot * dn_stride + dn_stride),
                )?;
            } else {
                let (gu_ptr, gu_len, dn_ptr, dn_len) = disk_sources[(tier ^ DISK_MARK) as usize];
                unsafe {
                    stream.memcpy_htod(
                        std::slice::from_raw_parts(gu_ptr as *const bf16, gu_len / 2),
                        &mut staging_gu.slice_mut(slot * gu_stride..slot * gu_stride + gu_stride),
                    )?;
                    stream.memcpy_htod(
                        std::slice::from_raw_parts(dn_ptr as *const bf16, dn_len / 2),
                        &mut staging_dn.slice_mut(slot * dn_stride..slot * dn_stride + dn_stride),
                    )?;
                }
            }
            gu_table[expert] = stg_gu + (slot * gu_stride * 2) as u64;
            dn_table[expert] = stg_dn + (slot * dn_stride * 2) as u64;
            valid_table[expert] = 1;
        }
        for (index, &id) in ids_host.iter().enumerate() {
            let id = usize::try_from(id).context("tiered route id negative")?;
            if valid_table.get(id).is_some_and(|valid| *valid == 1) {
                active_table[index] = 1;
            }
        }
        run_cohort(&gu_table, &dn_table, &valid_table, &active_table)?;
    }
    Ok(())
}

impl Runtime {
    fn gemma(&self) -> Result<&GemmaState> {
        match &self.program {
            Program::Gemma(state) => Ok(state),
            Program::Clover(_) => {
                anyhow::bail!("DiffusionGemma operation requires its canvas payload")
            }
        }
    }

    fn gemma_mut(&mut self) -> Result<&mut GemmaState> {
        match &mut self.program {
            Program::Gemma(state) => Ok(state),
            Program::Clover(_) => {
                anyhow::bail!("DiffusionGemma operation requires its canvas payload")
            }
        }
    }

    pub(super) fn reset(&mut self, device: &Device) -> Result<()> {
        device.stream.synchronize()?;
        self.prefix = 0;
        self.pending.clear();
        self.limit = None;
        self.produced = 0;
        self.terminal = false;
        match &mut self.program {
            Program::Gemma(state) => {
                let _ = &state.mapped;
                state.draws = 0;
            }
            Program::Clover(state) => {
                state.forward.reset(device)?;
                state.sampler.reset()?;
                #[cfg(test)]
                {
                    state.policy = None;
                }
            }
        }
        Ok(())
    }
}

impl GemmaState {
    fn workspace_bytes(&self) -> Result<u64> {
        let buffers = [
            2u64 * [
                self.unit.len() as u64,
                self.x.len() as u64,
                self.x2.len() as u64,
                self.n.len() as u64,
                self.qkv.len() as u64,
                self.attn.len() as u64,
                self.o.len() as u64,
                self.norm_sum.len() as u64,
                self.dense_gu.len() as u64,
                self.dense_act.len() as u64,
                self.dense_out.len() as u64,
                self.x_dense.len() as u64,
                self.x_experts.len() as u64,
                self.dense_normed.len() as u64,
                self.routed_normed.len() as u64,
                self.summed.len() as u64,
                self.prescale_out.len() as u64,
                self.route_logits.len() as u64,
                self.routed.len() as u64,
                self.shared_dummy_gate.len() as u64,
                self.shared_dummy_down.len() as u64,
                self.contributions.len() as u64,
                self.head_bf16.len() as u64,
                self.soft.len() as u64,
                self.sc_pre.len() as u64,
                self.sc_emb.len() as u64,
                self.sc_act.len() as u64,
                self.sc_down.len() as u64,
                self.sc_summed.len() as u64,
                self.sc_post.len() as u64,
                self.embed_x.len() as u64,
                self.stage_k.len() as u64,
                self.stage_v.len() as u64,
                self.zero.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
            4u64 * [
                self.route_weights.len() as u64,
                self.logits.len() as u64,
                self.processed.len() as u64,
                self.probs.len() as u64,
                self.entropies.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
            4u64 * [
                self.meta.len() as u64,
                self.route_ids.len() as u64,
                self.argmax_cur.len() as u64,
                self.argmax_prev.len() as u64,
                self.picks.len() as u64,
                self.renoise.len() as u64,
                self.position.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
            4u64 * [
                self.route_active.len() as u64,
                self.wave_valid.len() as u64,
                self.wave_active.len() as u64,
                self.route_done.len() as u64,
                self.canvas_ids.len() as u64,
                self.commit_ids.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
            8u64 * [
                self.slot_generation.len() as u64,
                self.route_gen.len() as u64,
                self.wave_slots_gu.len() as u64,
                self.wave_slots_dn.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
            [
                self.ones.len() as u64,
                self.cohort_scratch.len() as u64,
                self.accept_mask.len() as u64,
                self.stop_flag.len() as u64,
            ]
            .into_iter()
            .sum::<u64>(),
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| {
            sum.checked_add(bytes)
                .context("canvas buffer bytes overflow")
        })?;
        [
            self.route.workspace_bytes() as u64,
            self.rope_tables.iter().map(|v| v.len() as u64 * 2).sum(),
            self.flash_sliding_canvas.allocated_bytes() as u64,
            self.flash_sliding_commit.allocated_bytes() as u64,
            self.flash_full.allocated_bytes() as u64,
            ((super::super::moe::prefill::GEMM1_PTR_WORDS
                + super::super::moe::prefill::GEMM2_PTR_WORDS)
                * 8) as u64,
        ]
        .into_iter()
        .try_fold(buffers, |sum, bytes| {
            sum.checked_add(bytes)
                .context("canvas operation workspace bytes overflow")
        })
    }
}

impl Engine {
    /// Prompt prefill through the causal encoder program, then the first
    /// canvas; returns the first committed token.
    /// Gate entries: the encoder pass without a canvas, the runtime view,
    /// and the engine stream (trajectory gate plumbing).
    #[cfg(test)]
    pub(crate) fn diffusion_encoder_chunk_public(&mut self, ids: &[u32]) -> Result<()> {
        for chunk in ids.chunks(self.diffusion_canvas()?) {
            self.diffusion_encoder_chunk(chunk)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn diffusion_runtime_public(&mut self) -> (usize, &mut GemmaState) {
        let runtime = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")
            .unwrap();
        (runtime.canvas, runtime.gemma_mut().unwrap())
    }

    #[cfg(test)]
    pub(crate) fn stream_public(&self) -> Arc<cudarc::driver::CudaStream> {
        self.device.stream.clone()
    }

    #[cfg(test)]
    pub(crate) fn diffusion_layer_count(&self) -> usize {
        self.model_spec.layers.len()
    }

    pub(super) fn prefill_diffusion(
        &mut self,
        ids: &[u32],
        trace: Option<&mut crate::trace::Trace>,
    ) -> Result<u32> {
        ensure!(trace.is_none(), "DiffusionGemma prefill trace is not bound");
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
        if let Some(runtime) = &self.diffusion
            && let Program::Gemma(state) = &runtime.program
        {
            let count = state.requested;
            let canvas = self.diffusion_canvas()?;
            let canvases = count.div_ceil(canvas);
            let extent = ids
                .len()
                .checked_add(
                    canvases
                        .checked_mul(canvas)
                        .context("canvas extent overflow")?,
                )
                .context("request extent overflow")?;
            ensure!(
                extent <= self.capacity,
                "prompt plus requested canvases exceeds KV capacity {} ({} prompt + {} x {} canvas)",
                self.capacity,
                ids.len(),
                canvases,
                canvas
            );
        }
        ensure!(
            ids.iter().all(|&x| x < self.model_spec.vocab),
            "input token id outside vocabulary"
        );
        let canvas = self.diffusion_canvas()?;
        if let Some(runtime) = &mut self.diffusion
            && let Program::Clover(state) = &mut runtime.program
        {
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
        self.canvas_step_forced(None, None)?;
        self.diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .pending
            .pop_front()
            .context("canvas commit produced no tokens")
    }

    /// One committed chunk through the causal program at the current prefix.
    fn diffusion_encoder_chunk(&mut self, ids: &[u32]) -> Result<()> {
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        let s = &self.device.stream;
        let rows = ids.len();
        ensure!(rows <= r.canvas, "encoder chunk exceeds the canvas");
        let prefix = r.prefix;
        if let Program::Clover(state) = &mut r.program {
            #[cfg(all(test, feature = "cuda"))]
            let _range = crate::backend::cuda::profile::Range::new(
                state.capture.is_some(),
                c"clover_causal_state_only",
            );
            state.forward.run(&self.device, ids, false)?;
            r.prefix += rows;
            self.position = r.prefix;
            #[cfg(test)]
            self.capture_clover("causal", Some(ids))?;
            return Ok(());
        }
        let state = r.gemma_mut()?;
        s.memcpy_htod(ids, &mut state.commit_ids.slice_mut(..rows))?;
        s.memcpy_htod(&[prefix as i32], &mut state.position)?;
        self.diffusion_encoder_body(rows)
    }

    /// embed(ids) * sqrt(hidden) into the layer input.
    fn diffusion_embed_chunk(&mut self, rows: usize) -> Result<()> {
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .gemma_mut()?;
        let s = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        let scale = bf16::from_f32((hidden as f32).sqrt()).to_f32();
        unsafe {
            s.launch_builder(&r.scaled_embed)
                .arg(&r.weights.embed)
                .arg(&r.commit_ids)
                .arg(&r.unit)
                .arg(&mut r.embed_x)
                .arg(&mut r.n)
                .arg(&(rows as i32))
                .arg(&(hidden as i32))
                .arg(&scale)
                .arg(&r.weights.embed)
                .arg(&0i32)
                .arg(&0i32)
                .launch(flat(rows * hidden))?;
        }
        // The residual chain starts at the scaled embedding.
        s.memcpy_dtod(
            &r.embed_x.slice(..rows * hidden),
            &mut r.x.slice_mut(..rows * hidden),
        )?;
        Ok(())
    }

    /// One shared-weight layer. `encoder` selects the causal windowed form
    /// (prompt and commit passes); otherwise the bidirectional canvas form.
    fn diffusion_layer(&mut self, i: usize, rows: usize, encoder: bool) -> Result<()> {
        let prefix = self
            .diffusion
            .as_ref()
            .context("canvas runtime missing")?
            .prefix;
        let canvas = self
            .diffusion
            .as_ref()
            .context("canvas runtime missing")?
            .canvas;
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .gemma_mut()?;
        let s = &self.device.stream;
        let layer = &r.weights.layers[i];
        let plan = &self.model_spec.layers[i];
        let geometry = plan.geometry.context("layer geometry missing")?;
        let full = plan.sliding_window.is_none();
        let (hq, hkv, d) = (
            self.model_spec.q_heads as usize,
            geometry.kv_heads.unwrap_or(self.model_spec.kv_heads) as usize,
            geometry.head_dim as usize,
        );
        let kv_width = hkv * d;
        let q_width = hq * d;
        let qkv_width = q_width + kv_width * (1 + usize::from(!full));
        let hidden = self.model_spec.hidden as usize;
        let eps = self.model_spec.rms_eps;
        let (meta, _) = r.meta.device_ptr(s);
        let ctx = NormCtx {
            common: &r.common,
            meta,
            eps,
        };

        // Attention input norm, projection, K/V preparation.
        rms(s, &ctx, &r.x, &layer.input_norm, &mut r.n, rows, hidden)?;
        gemm_t(
            &r.gemm, &layer.qkv, &r.n, &mut r.qkv, rows, qkv_width, hidden,
        )?;
        let state_index = plan.state_index as usize;
        let ring = self.state.kv_ring(state_index)?;
        let sliding = !full;
        let canvas_now = !encoder;
        let (k_target, v_target, cache_capacity, cache_start, ring_tokens) = if sliding {
            let history = prefix.min(ring - canvas);
            if encoder && history > 0 {
                let kv = &self.state.kv[state_index];
                for (source, target) in [(&kv.k, &mut r.stage_k), (&kv.v, &mut r.stage_v)] {
                    RingCopy {
                        stream: s,
                        source,
                        target,
                        source_capacity: ring,
                        target_capacity: ring + canvas,
                        heads: hkv,
                        head_dim: d,
                    }
                    .run((prefix - history) % ring, 0, history, true, false)?;
                }
                (&r.stage_k, &r.stage_v, ring + canvas, history as i32, 0)
            } else {
                let kv = &self.state.kv[state_index];
                (&kv.k, &kv.v, ring, -1, ring as i32)
            }
        } else {
            let kv = &self.state.kv[state_index];
            (&kv.k, &kv.v, self.capacity, -1, 0)
        };
        {
            let table = &r.rope_tables[usize::from(full)];
            let function = if full {
                &r.rope_shared
            } else {
                &r.rope_sliding
            };
            let prep_heads = if full { hkv } else { 2 * hkv };
            unsafe {
                s.launch_builder(function)
                    .arg(&mut r.qkv)
                    .arg(k_target)
                    .arg(v_target)
                    .arg(&layer.q_norm)
                    .arg(&layer.k_norm)
                    .arg(table)
                    .arg(&r.position)
                    .arg(&cache_start)
                    .arg(&(rows as i32))
                    .arg(&(hq as i32))
                    .arg(&(hkv as i32))
                    .arg(&(d as i32))
                    .arg(&(cache_capacity as i32))
                    .arg(&ring_tokens)
                    .arg(&eps)
                    .arg(&(qkv_width as u64))
                    .arg(&(q_width as u64))
                    .arg(&(qkv_width as u64))
                    .arg(&((q_width + kv_width) as u64))
                    .arg(&(qkv_width as u64))
                    .launch(LaunchConfig {
                        grid_dim: ((rows * (hq + prep_heads)) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        let sliding_history = prefix.min(ring - canvas);
        let keys = if encoder {
            if sliding {
                sliding_history + rows
            } else {
                prefix + rows
            }
        } else if sliding {
            canvas_window(prefix, rows, ring).0
        } else {
            prefix + rows
        };
        {
            let start = if sliding && canvas_now {
                canvas_window(prefix, rows, ring).1
            } else if sliding {
                sliding_history
            } else {
                prefix
            };
            let shape = crate::backend::cuda::verification::Shape {
                rows,
                start,
                q_heads: hq,
                kv_heads: hkv,
                capacity: cache_capacity,
                head_dim: d,
                causal: encoder,
                window_left: (encoder && sliding).then_some(ring - canvas),
            };
            let query = crate::backend::cuda::flash::Query {
                q: &r.qkv,
                q_row_stride: qkv_width,
                k: k_target,
                v: v_target,
                out: &mut r.attn,
                shape,
                keys,
                prefix: None,
                image_blocks: None,
            };
            if sliding {
                let flash = if encoder {
                    &mut r.flash_sliding_commit
                } else {
                    &mut r.flash_sliding_canvas
                };
                flash.run_masked(query)?;
            } else {
                r.flash_full.run_masked(query)?;
            }
        }
        if encoder && sliding && prefix > 0 {
            // Commit the staged window tail back into the ring; the first
            // chunk (prefix 0) wrote the ring directly.
            let (_, keep, source_offset, absolute_start) =
                commit_window(prefix, rows, ring, canvas);
            let kv = &mut self.state.kv[state_index];
            for (source, target) in [(&r.stage_k, &mut kv.k), (&r.stage_v, &mut kv.v)] {
                RingCopy {
                    stream: s,
                    source,
                    target,
                    source_capacity: ring + canvas,
                    target_capacity: ring,
                    heads: hkv,
                    head_dim: d,
                }
                .run(source_offset, absolute_start % ring, keep, false, true)?;
            }
        }
        gemm_t(&r.gemm, &layer.o, &r.attn, &mut r.o, rows, hidden, q_width)?;
        // Residual + post-attention norm: x2 = x + post_attn(o) — the norm
        // consumes the attention output, then the residual adds (add_rmsnorm
        // norms the sum, a different model form).
        rms(
            s,
            &ctx,
            &r.o,
            &layer.post_attn,
            &mut r.norm_sum,
            rows,
            hidden,
        )?;
        unsafe {
            let (xptr, _) = r.x.device_ptr(s);
            let (nptr, _) = r.norm_sum.device_ptr(s);
            let (optr, _) = r.x2.device_ptr_mut(s);
            s.launch_builder(&r.residual_scale)
                .arg(&xptr)
                .arg(&nptr)
                .arg(&1.0f32)
                .arg(&optr)
                .arg(&((rows * hidden) as i64))
                .launch(flat(rows * hidden))?;
        }

        // Parallel FFN: dense half.
        let dense_inter = geometry.intermediate as usize;
        rms(s, &ctx, &r.x2, &layer.pre_ffn, &mut r.x_dense, rows, hidden)?;
        gemm_t(
            &r.gemm,
            &layer.dense_gu,
            &r.x_dense,
            &mut r.dense_gu,
            rows,
            2 * dense_inter,
            hidden,
        )?;
        gelu_split(
            s,
            &r.common.gate,
            &r.dense_gu,
            &mut r.dense_act,
            rows,
            dense_inter,
        )?;
        gemm_t(
            &r.gemm,
            &layer.dense_down,
            &r.dense_act,
            &mut r.dense_out,
            rows,
            hidden,
            dense_inter,
        )?;
        rms(
            s,
            &ctx,
            &r.dense_out,
            &layer.post1,
            &mut r.dense_normed,
            rows,
            hidden,
        )?;

        // Parallel FFN: routed half. The router reads the residual before the
        // FFN input norms.
        let experts = self.model_spec.moe.as_ref().context("MoE missing")?.experts as usize;
        let top_k = self
            .model_spec
            .moe
            .as_ref()
            .context("MoE missing")?
            .experts_per_tok as usize;
        let expert_inter = self
            .model_spec
            .moe
            .as_ref()
            .context("MoE missing")?
            .expert_intermediate as usize;
        r.route.diffusion_topk(
            s,
            crate::backend::cuda::moe::DiffusionRoute {
                x: &r.x2,
                scale: &layer.router_scale,
                proj: &layer.router_proj,
                per_expert: &layer.per_expert_scale,
                prescaled: &mut r.prescale_out,
                logits: &mut r.route_logits,
                ids: &mut r.route_ids,
                weights: &mut r.route_weights,
                rows,
                hidden,
                top_k,
                eps,
            },
        )?;
        #[cfg(test)]
        {
            r.route_taps.push((
                i,
                s.clone_dtod(&r.route_ids.slice(..rows * top_k))?,
                s.clone_dtod(&r.route_weights.slice(..rows * top_k))?,
            ));
        }
        rms(
            s,
            &ctx,
            &r.x2,
            &layer.pre_ffn2,
            &mut r.x_experts,
            rows,
            hidden,
        )?;
        // Routed experts through the shared cohort path over the three
        // tiers: resident experts contribute once over the slot table;
        // host-tier experts stream from the cgroup-bounded arena and disk
        // experts from the checkpoint mappings, both through the staging
        // slots in fixed waves (each cohort's count D2H synchronizes).
        s.memset_zeros(&mut r.route_done.slice_mut(..rows * top_k))?;
        s.memset_zeros(&mut r.routed.slice_mut(..rows * hidden))?;
        {
            let ids_host = s.clone_dtoh(&r.route_ids.slice(..rows * top_k))?;
            let scratch = Scratch {
                ptr: r.cohort_scratch.device_ptr(s).0,
                bytes: r.cohort_scratch.len() as u64,
            };
            let staging_slots = r.staging_gu.len() / (2 * expert_inter * hidden);
            contribute_tiered(TieredCall {
                ops: &r.moe,
                stream: s,
                scratch,
                gpu_slots: &r.expert_gpu_slots,
                tiers: &r.expert_tier[i * experts..(i + 1) * experts],
                host_arena: &r.expert_host,
                disk_sources: &r.expert_disk,
                staging_gu: &mut r.staging_gu,
                staging_dn: &mut r.staging_dn,
                staging_slots,
                wave_slots_gu: &mut r.wave_slots_gu,
                wave_slots_dn: &mut r.wave_slots_dn,
                wave_valid: &mut r.wave_valid,
                wave_active: &mut r.wave_active,
                x_experts: &r.x_experts,
                ids: &r.route_ids,
                ids_host: &ids_host,
                weights: &r.route_weights,
                generation: &r.route_gen,
                slot_generation: &r.slot_generation,
                shared_gate: &r.shared_dummy_gate,
                shared_down: &r.shared_dummy_down,
                contributions: &mut r.contributions,
                done: &mut r.route_done,
                rows,
                experts,
                top_k,
                hidden,
                expert_inter,
            })?;
        }
        {
            let ids = view(
                &r.route_ids,
                s,
                0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
            )?;
            let active = view(
                &r.route_active,
                s,
                0,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
            )?;
            let contributions = unsafe {
                crate::backend::cuda::contracts::Tensor::<bf16, 3>::new(
                    r.contributions.device_ptr(s).0,
                    [rows as u32, top_k as u32, hidden as u32],
                    [(top_k * hidden) as u64, hidden as u64, 1],
                    (rows * top_k * hidden) as u64 * 2,
                )?
            };
            let combine = MoeCombineRouted {
                ids,
                final_active: active,
                done: view(
                    &r.route_done,
                    s,
                    0,
                    [rows as u32, top_k as u32],
                    [top_k as u64, 1],
                )?,
                contributions,
                routed_out: TensorMut(matrix(&r.routed, s, 0, rows, hidden)?),
            };
            r.moe.combine_routed(s, &combine)?;
        }
        rms(
            s,
            &ctx,
            &r.routed,
            &layer.post2,
            &mut r.routed_normed,
            rows,
            hidden,
        )?;
        // Tail: sum, final post-norm, residual add and the per-tower scalar.
        unsafe {
            s.launch_builder(&r.residual_scale)
                .arg(&r.dense_normed)
                .arg(&r.routed_normed)
                .arg(&1.0f32)
                .arg(&mut r.summed)
                .arg(&((rows * hidden) as i64))
                .launch(flat(rows * hidden))?;
        }
        {
            rms(
                s,
                &ctx,
                &r.summed,
                &layer.post,
                &mut r.dense_out,
                rows,
                hidden,
            )?;
        }
        unsafe {
            let scalar = if encoder {
                r.encoder_scalars_host[i]
            } else {
                r.decoder_scalars_host[i]
            };
            let (hptr, _) = r.x2.device_ptr(s);
            let (sptr2, _) = r.dense_out.device_ptr(s);
            let (optr, _) = r.x2.device_ptr_mut(s);
            s.launch_builder(&r.residual_scale)
                .arg(&hptr)
                .arg(&sptr2)
                .arg(&scalar)
                .arg(&optr)
                .arg(&((rows * hidden) as i64))
                .launch(flat(rows * hidden))?;
        }
        std::mem::swap(&mut r.x, &mut r.x2);
        #[cfg(test)]
        if let Some(slot) = r.tap_capture.as_mut().and_then(|taps| taps.get_mut(i)) {
            *slot = s.clone_dtod(&r.x.slice(..rows * hidden))?;
        }
        Ok(())
    }
}

impl Engine {
    fn diffusion_canvas(&self) -> Result<usize> {
        Ok(self
            .diffusion
            .as_ref()
            .context("diffusion runtime missing")?
            .canvas)
    }

    fn diffusion_encoder_body(&mut self, rows: usize) -> Result<()> {
        self.diffusion_embed_chunk(rows)?;
        for i in 0..self.model_spec.layers.len() {
            self.diffusion_layer(i, rows, true)?;
        }
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?;
        r.prefix += rows;
        self.position = r.prefix;
        Ok(())
    }

    /// Canvas self-conditioning, layer stack, final norm and head.
    fn canvas_forward(&mut self) -> Result<()> {
        if let Program::Clover(state) = &mut self
            .diffusion
            .as_mut()
            .context("canvas runtime missing")?
            .program
        {
            #[cfg(test)]
            if state.policy.is_some() {
                return Ok(());
            }
            #[cfg(all(test, feature = "cuda"))]
            let _range = crate::backend::cuda::profile::Range::new(
                state.capture.is_some(),
                c"clover_preview_head",
            );
            self.logits_cache.invalidate();
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
            self.logits_cache.mark_valid();
            return Ok(());
        }
        self.logits_cache.invalidate();
        let (canvas, hidden, vocab, eps, dense_inter, cap) = {
            let r = self
                .diffusion
                .as_ref()
                .context("diffusion runtime missing")?;
            (
                r.canvas,
                self.model_spec.hidden as usize,
                self.model_spec.vocab as usize,
                self.model_spec.rms_eps,
                self.model_spec.intermediate as usize,
                self.model_spec.logit_softcap.context("softcap missing")?,
            )
        };
        {
            let r = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?
                .gemma_mut()?;
            let s = &self.device.stream;
            let (meta, _) = r.meta.device_ptr(s);
            let ctx = NormCtx {
                common: &r.common,
                meta,
                eps,
            };
            let scale = bf16::from_f32((hidden as f32).sqrt()).to_f32();

            // Self-conditioning: the soft embedding is the previous
            // probabilities' BF16 GEMM against the tied embedding, scaled.
            gemm_n(
                &r.gemm,
                &r.weights.embed,
                &r.soft,
                &mut r.sc_emb,
                canvas,
                hidden,
                vocab,
            )?;
            unsafe {
                let (sptr, _) = r.sc_emb.device_ptr(s);
                let (zptr, _) = r.zero.device_ptr(s);
                let (optr, _) = r.sc_emb.device_ptr_mut(s);
                s.launch_builder(&r.residual_scale)
                    .arg(&sptr)
                    .arg(&zptr)
                    .arg(&scale)
                    .arg(&optr)
                    .arg(&((canvas * hidden) as i64))
                    .launch(flat(canvas * hidden))?;
            }
            rms(
                s,
                &ctx,
                &r.sc_emb,
                &r.weights.sc_pre_norm,
                &mut r.sc_pre,
                canvas,
                hidden,
            )?;
            gemm_t(
                &r.gemm,
                &r.weights.sc_gu,
                &r.sc_pre,
                &mut r.dense_gu,
                canvas,
                2 * dense_inter,
                hidden,
            )?;
            gelu_split(
                s,
                &r.common.gate,
                &r.dense_gu,
                &mut r.sc_act,
                canvas,
                dense_inter,
            )?;
            gemm_t(
                &r.gemm,
                &r.weights.sc_down,
                &r.sc_act,
                &mut r.sc_down,
                canvas,
                hidden,
                dense_inter,
            )?;
            unsafe {
                s.launch_builder(&r.scaled_embed)
                    .arg(&r.weights.embed)
                    .arg(&r.canvas_ids)
                    .arg(&r.unit)
                    .arg(&mut r.embed_x)
                    .arg(&mut r.n)
                    .arg(&(canvas as i32))
                    .arg(&(hidden as i32))
                    .arg(&scale)
                    .arg(&r.weights.embed)
                    .arg(&0i32)
                    .arg(&0i32)
                    .launch(flat(canvas * hidden))?;
                s.launch_builder(&r.residual_scale)
                    .arg(&r.embed_x)
                    .arg(&r.sc_down)
                    .arg(&1.0f32)
                    .arg(&mut r.sc_summed)
                    .arg(&((canvas * hidden) as i64))
                    .launch(flat(canvas * hidden))?;
            }
            rms(
                s,
                &ctx,
                &r.sc_summed,
                &r.unit,
                &mut r.sc_post,
                canvas,
                hidden,
            )?;
            unsafe {
                // The self-conditioning module's output IS the canvas input
                // (post_norm(embed + sc_signal)); it does not add onto the
                // raw embedding (DiffusionGemmaSelfConditioning.forward).
                let (sptr, _) = r.sc_post.device_ptr(s);
                let (zptr, _) = r.zero.device_ptr(s);
                let (optr, _) = r.x.device_ptr_mut(s);
                s.launch_builder(&r.residual_scale)
                    .arg(&sptr)
                    .arg(&zptr)
                    .arg(&1.0f32)
                    .arg(&optr)
                    .arg(&((canvas * hidden) as i64))
                    .launch(flat(canvas * hidden))?;
            }
        }
        for i in 0..self.model_spec.layers.len() {
            self.diffusion_layer(i, canvas, false)?;
        }
        {
            let r = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?
                .gemma_mut()?;
            let s = &self.device.stream;
            let (meta, _) = r.meta.device_ptr(s);
            let ctx = NormCtx {
                common: &r.common,
                meta,
                eps,
            };
            rms(
                s,
                &ctx,
                &r.x,
                &r.weights.final_norm,
                &mut r.n,
                canvas,
                hidden,
            )?;
            gemm_t(
                &r.gemm,
                &r.weights.embed,
                &r.n,
                &mut r.head_bf16,
                canvas,
                vocab,
                hidden,
            )?;
            unsafe {
                s.launch_builder(&r.softcap_f32)
                    .arg(&r.head_bf16)
                    .arg(&mut r.logits)
                    .arg(&((canvas * vocab) as i64))
                    .arg(&cap)
                    .launch(flat(canvas * vocab))?;
            }
        }
        self.logits_cache.mark_valid();
        Ok(())
    }

    /// One full canvas: initialization, the denoise loop with the sampler
    /// sequence, and the 256-row encoder commit of the argmax canvas.
    /// Teacher-forced inputs (`canvas_ids`, `soft`) serve the trajectory
    /// gate's replay; both `None` is the production init.
    pub(crate) fn canvas_step_forced(
        &mut self,
        canvas_ids: Option<&[u32]>,
        soft: Option<&[bf16]>,
    ) -> Result<()> {
        let vocab = self.model_spec.vocab as usize;
        let steps = {
            let runtime = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?;
            ensure!(
                runtime.prefix + runtime.canvas <= self.capacity,
                "canvas exceeds KV capacity"
            );
            let prefix = runtime.prefix;
            let canvas = runtime.canvas;
            match &mut runtime.program {
                Program::Clover(state) => {
                    ensure!(
                        canvas_ids.is_none() && soft.is_none(),
                        "Clover canvas uses the masked-block initialization"
                    );
                    ensure!(
                        !runtime.terminal && runtime.limit.is_some(),
                        "Clover generation request is not active"
                    );
                    state.sampler.begin_block()?;
                    32
                }
                Program::Gemma(r) => {
                    let s = &self.device.stream;
                    s.memcpy_htod(&[prefix as i32], &mut r.position)?;
                    if let Some(ids) = canvas_ids {
                        ensure!(ids.len() == canvas, "forced canvas length differs");
                        s.memcpy_htod(ids, &mut r.canvas_ids)?;
                    } else {
                        unsafe {
                            s.launch_builder(&r.philox)
                                .arg(&r.seed)
                                .arg(&(r.draws * 2 + 1))
                                .arg(&mut r.renoise)
                                .arg(&(canvas as i64))
                                .arg(&(vocab as i32))
                                .launch(flat(canvas / 4))?;
                            s.launch_builder(&r.accept_scatter)
                                .arg(&r.ones)
                                .arg(&r.renoise)
                                .arg(&r.renoise)
                                .arg(&mut r.canvas_ids)
                                .arg(&(canvas as i32))
                                .launch(flat(canvas))?;
                        }
                    }
                    match soft {
                        Some(input) => {
                            ensure!(input.len() == canvas * vocab);
                            s.memcpy_htod(input, &mut r.soft)?;
                        }
                        None => s.memset_zeros(&mut r.soft)?,
                    }
                    s.memcpy_htod(&vec![-1i32; canvas], &mut r.argmax_prev)?;
                    r.draws += 1;
                    r.denoise_steps
                }
            }
        };
        #[cfg(test)]
        if self.diffusion_single_step {
            self.diffusion
                .as_ref()
                .context("diffusion runtime missing")?
                .gemma()?;
            self.canvas_forward()?;
            let draws = self.diffusion_forced_draws.take();
            self.canvas_sample_draws(steps, draws)?;
            return Ok(());
        }
        let mut unresolved = 32;
        for cur_step in (1..=steps).rev() {
            self.canvas_forward()?;
            #[cfg(test)]
            self.capture_clover("raw", None)?;
            self.canvas_sample(cur_step)?;
            #[cfg(test)]
            self.capture_clover("fill", None)?;
            let runtime = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?;
            let s = &self.device.stream;
            let stop = match &mut runtime.program {
                Program::Clover(state) => {
                    let pending = s.clone_dtoh(&state.sampler.state.slice(3..4))?[0];
                    ensure!(
                        pending < unresolved,
                        "Clover denoise step did not resolve a row"
                    );
                    unresolved = pending;
                    pending == 0
                }
                Program::Gemma(r) => {
                    let flag = s.clone_dtoh(&r.stop_flag)?;
                    s.memcpy_dtod(&r.argmax_cur.slice(..), &mut r.argmax_prev.slice_mut(..))?;
                    flag[0] == 1
                }
            };
            if stop {
                break;
            }
        }
        {
            let runtime = self
                .diffusion
                .as_mut()
                .context("diffusion runtime missing")?;
            if let Program::Clover(state) = &mut runtime.program {
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
                let Program::Clover(state) = &mut runtime.program else {
                    unreachable!();
                };
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
                return Ok(());
            }
        }
        let canvas = self.diffusion_canvas()?;
        let padded = {
            let r = self
                .diffusion
                .as_ref()
                .context("diffusion runtime missing")?
                .gemma()?;
            let committed = self.device.stream.clone_dtoh(&r.argmax_cur)?;
            let mut padded: Vec<u32> = committed.iter().map(|&v| v as u32).collect();
            if !r.ignore_eos
                && let Some(position) = padded.iter().position(|v| self.model_spec.eos.contains(v))
            {
                padded[position + 1..].fill(r.pad);
            }
            padded
        };
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .gemma_mut()?;
        self.device.stream.memcpy_htod(&padded, &mut r.commit_ids)?;
        self.diffusion_encoder_body(canvas)?;
        self.diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .pending
            .extend(padded);
        Ok(())
    }

    /// Eager sampler tail of one denoise step.
    fn canvas_sample(&mut self, cur_step: u32) -> Result<()> {
        if let Program::Clover(state) = &mut self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .program
        {
            #[cfg(test)]
            if let Some(steps) = &mut state.policy {
                let step = steps
                    .pop_front()
                    .context("Runner FAIL policy fixture exhausted")?;
                let s = &self.device.stream;
                s.memcpy_htod(&step.picks, &mut state.sampler.picks)?;
                for (row, (&pick, &confidence)) in
                    step.picks.iter().zip(&step.confidence).enumerate()
                {
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
            return state.sampler.step(&mut state.logits);
        }
        self.canvas_sample_draws(cur_step, None)
    }

    /// `draws` injects recorded picks/renoise (the trajectory gate's replay)
    /// instead of launching the categorical and Philox kernels.
    pub(crate) fn canvas_sample_draws(
        &mut self,
        cur_step: u32,
        draws: Option<(&[i32], &[i32])>,
    ) -> Result<()> {
        let canvas = self
            .diffusion
            .as_ref()
            .context("canvas runtime missing")?
            .canvas;
        let r = self
            .diffusion
            .as_mut()
            .context("diffusion runtime missing")?
            .gemma_mut()?;
        let s = &self.device.stream;
        let vocab = self.model_spec.vocab as usize;
        let steps = r.denoise_steps;
        let t = r.t_min + (r.t_max - r.t_min) * (cur_step as f32 / steps as f32);
        let counter = r.draws;
        unsafe {
            s.launch_builder(&r.temperature)
                .arg(&r.logits)
                .arg(&mut r.processed)
                .arg(&((canvas * vocab) as i64))
                .arg(&t)
                .launch(flat(canvas * vocab))?;
            s.launch_builder(&r.softmax)
                .arg(&r.processed)
                .arg(&mut r.probs)
                .arg(&mut r.soft)
                .arg(&(canvas as i32))
                .arg(&(vocab as i64))
                .launch(grid(canvas, 1024))?;
            s.launch_builder(&r.entropy)
                .arg(&r.processed)
                .arg(&mut r.entropies)
                .arg(&(canvas as i32))
                .arg(&(vocab as i64))
                .arg(&1024i32)
                .launch(grid(canvas, 1024))?;
            s.launch_builder(&r.accept)
                .arg(&r.entropies)
                .arg(&mut r.accept_mask)
                .arg(&(canvas as i32))
                .arg(&r.entropy_bound)
                .launch(grid(1, canvas as u32))?;
            s.launch_builder(&r.argmax)
                .arg(&r.processed)
                .arg(&mut r.argmax_cur)
                .arg(&(canvas as i32))
                .arg(&(vocab as i64))
                .arg(&1024i32)
                .launch(grid(canvas, 1024))?;
            if let Some((picks, renoise)) = draws {
                s.memcpy_htod(picks, &mut r.picks)?;
                s.memcpy_htod(renoise, &mut r.renoise)?;
            } else {
                s.launch_builder(&r.categorical)
                    .arg(&r.probs)
                    .arg(&mut r.picks)
                    .arg(&(canvas as i32))
                    .arg(&(vocab as i64))
                    .arg(&r.seed)
                    .arg(&(counter * 2))
                    .arg(&1024i32)
                    .launch(grid(canvas, 1024))?;
                s.launch_builder(&r.philox)
                    .arg(&r.seed)
                    .arg(&(counter * 2 + 1))
                    .arg(&mut r.renoise)
                    .arg(&(canvas as i64))
                    .arg(&(vocab as i32))
                    .launch(flat(canvas / 4))?;
            }
            s.launch_builder(&r.accept_scatter)
                .arg(&r.accept_mask)
                .arg(&r.picks)
                .arg(&r.renoise)
                .arg(&mut r.canvas_ids)
                .arg(&(canvas as i32))
                .launch(flat(canvas))?;
            s.launch_builder(&r.stop)
                .arg(&r.argmax_prev)
                .arg(&r.argmax_cur)
                .arg(&r.entropies)
                .arg(&mut r.stop_flag)
                .arg(&(canvas as i32))
                .arg(&r.confidence)
                .launch(grid(1, 256))?;
        }
        r.draws += 1;
        Ok(())
    }

    /// Canvas steps execute eagerly when the committed queue is empty.
    pub(super) fn canvas_logits(&self) -> Result<crate::backend::gpu::CudaView<'_, bf16>> {
        let runtime = self.diffusion.as_ref().context("canvas runtime missing")?;
        let width = self.model_spec.vocab as usize;
        let logits = match &runtime.program {
            Program::Clover(state) => &state.forward.logits,
            Program::Gemma(state) => &state.head_bf16,
        };
        Ok(logits.slice((runtime.canvas - 1) * width..runtime.canvas * width))
    }

    pub(super) fn canvas_logits_f32(&self) -> Result<Vec<f32>> {
        let runtime = self.diffusion.as_ref().context("canvas runtime missing")?;
        if let Program::Gemma(state) = &runtime.program {
            let width = self.model_spec.vocab as usize;
            return Ok(self.device.stream.clone_dtoh(
                &state
                    .logits
                    .slice((runtime.canvas - 1) * width..runtime.canvas * width),
            )?);
        }
        Ok(self
            .device
            .stream
            .clone_dtoh(&self.canvas_logits()?)?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    pub(super) fn decode_diffusion(&mut self, graph: bool) -> Result<u32> {
        ensure!(!graph, "canvas graph execution is not bound");
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
            self.canvas_step_forced(None, None)?;
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

#[cfg(all(test, feature = "cuda"))]
#[test]
#[ignore = "GPU window and published Clover checkpoint"]
fn clover_canvas_logits_readback_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );

    fn state(runtime: &Runtime) -> &CloverState {
        match &runtime.program {
            Program::Clover(state) => state,
            Program::Gemma(_) => panic!("expected Clover"),
        }
    }
    fn state_mut(runtime: &mut Runtime) -> &mut CloverState {
        match &mut runtime.program {
            Program::Clover(state) => state,
            Program::Gemma(_) => panic!("expected Clover"),
        }
    }
    let directory =
        std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let settings = Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        device_budget_gib: Some(14),
        runtime_dir: Some(std::path::PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        )),
    };
    let mut engine = Engine::load(&directory, 0, 97, None, settings)?;
    ensure!(engine.check_logits().is_err(), "reset logits accepted");
    engine.generation_request(64, false, false)?;
    engine.diffusion_encoder_chunk(&[17; 32])?;
    ensure!(engine.check_logits().is_err(), "state-only logits accepted");
    let s = engine.device.stream.clone();
    let width = engine.model_spec.vocab as usize;
    s.memcpy_htod(
        &vec![bf16::from_f32(f32::NAN); width],
        &mut engine.work.logits,
    )?;
    state_mut(engine.diffusion.as_mut().unwrap())
        .sampler
        .begin_block()?;
    engine.canvas_forward()?;
    engine.check_logits()?;
    let raw = &mut state_mut(engine.diffusion.as_mut().unwrap()).forward.logits;
    s.memcpy_htod(&[bf16::from_f32(-42.0)], &mut raw.slice_mut(17..18))?;
    s.memcpy_htod(
        &[bf16::from_f32(42.0)],
        &mut raw.slice_mut(31 * width + 17..31 * width + 18),
    )?;
    let expected = s.clone_dtoh(
        &state(engine.diffusion.as_ref().unwrap())
            .forward
            .logits
            .slice(31 * width..32 * width),
    )?;
    let actual = engine.decode_logits()?;
    ensure!(
        actual.len() == width
            && actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_f32().to_bits()),
        "canvas readback differs from raw final preview row"
    );
    let mut sample = engine.sample_buffer(1)?;
    engine.copy_logits(&mut sample, 0)?;
    ensure!(s.clone_dtoh(&sample.0)? == expected, "canvas copy differs");
    let runtime = state_mut(engine.diffusion.as_mut().unwrap());
    runtime.sampler.apply_history(&mut runtime.logits)?;
    let mask = 31 * width + 151669;
    ensure!(
        s.clone_dtoh(&runtime.logits.slice(mask..mask + 1))?[0] == f32::NEG_INFINITY,
        "mask infinity control absent"
    );
    engine.check_logits()?;
    engine.diffusion_encoder_chunk(&[17; 32])?;
    engine.check_logits()?;
    let runtime = engine.diffusion.as_mut().unwrap();
    runtime.pending.extend([17, 19]);
    runtime.terminal = true;
    ensure!(engine.decode(false, None)? == 17, "first queued token");
    ensure!(engine.decode(false, None)? == 19, "last queued token");
    engine.check_logits()?;
    s.memcpy_htod(
        &[bf16::from_f32(f32::NAN)],
        &mut state_mut(engine.diffusion.as_mut().unwrap())
            .forward
            .logits
            .slice_mut(31 * width + 17..31 * width + 18),
    )?;
    ensure!(engine.check_logits().is_err(), "raw NaN control accepted");
    engine.reset()?;
    ensure!(
        engine.decode_logits().is_err() && engine.copy_logits(&mut sample, 0).is_err(),
        "reset preserved stale canvas logits"
    );
    println!(
        "CLOVER_SERVING_LOGITS PASS raw_row_bits_exact=true copy_bits_exact=true stale_work_ignored=true masked_fp32_ignored=true state_only_and_queue_preserved=true raw_nan_rejected=true reset_invalid=true"
    );
    Ok(())
}
