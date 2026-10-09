use crate::backend;
use crate::model::Model;
use anyhow::{Result, ensure};
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(Clone, Copy, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Decode,
    Prefill,
}

#[derive(Serialize)]
struct Sample {
    run: usize,
    step: usize,
    kv_tokens: usize,
    ms: f64,
    kv_read_bytes: usize,
    weight_read_bytes: usize,
    gb_s: f64,
    roofline_pct: f64,
}

pub struct Settings {
    pub device: usize,
    pub capacity: usize,
    pub dram_gb_s: f64,
    pub gemm_tflops: f64,
    pub chunk: Option<usize>,
    pub kernels: backend::Settings,
    pub steps: usize,
    pub warmup: usize,
    pub runs: usize,
    pub graph: bool,
}

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = q * (sorted.len() - 1) as f64;
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo as f64)
}

fn event(model: &Model) -> Result<backend::Event> {
    model.event()
}

fn check(ids: &[u32], s: &Settings) -> Result<()> {
    ensure!(
        s.dram_gb_s.is_finite() && s.dram_gb_s > 0.0,
        "dram-gb-s must be finite and positive"
    );
    ensure!(
        s.gemm_tflops.is_finite() && s.gemm_tflops > 0.0,
        "gemm-tflops must be finite and positive"
    );
    ensure!(!ids.is_empty(), "benchmark input is empty");
    ensure!(
        s.steps > 0 && s.runs > 0,
        "benchmark steps and runs must be positive"
    );
    ensure!(
        ids.len()
            .checked_add(s.steps.max(s.warmup))
            .is_some_and(|n| n <= s.capacity),
        "benchmark exceeds KV capacity"
    );
    Ok(())
}

fn warm(model: &mut Model, ids: &[u32], s: &Settings) -> Result<u32> {
    model.reset()?;
    let first = model.prefill(ids, None)?;
    model.check_logits()?;
    if s.graph {
        model.capture()?;
    }
    for _ in 0..s.warmup {
        model.step(s.graph)?;
    }
    model.check_logits()?;
    model.rewind(ids.len(), first)?;
    Ok(first)
}

pub fn run(dir: &Path, ids: &[u32], s: &Settings) -> Result<()> {
    check(ids, s)?;
    ensure!(
        s.steps >= 200,
        "benchmark requires at least 200 measured steps per run"
    );
    let start = Instant::now();
    let mut model = Model::load(dir, s.device, s.capacity, s.chunk, s.kernels.clone())?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    warm(&mut model, ids, s)?;
    let pairs: Vec<_> = (0..s.steps)
        .map(|_| Ok((event(&model)?, event(&model)?)))
        .collect::<Result<_>>()?;
    let vocab = model.config.vocab_size;
    let mut sampled_logits = model.sample_buffer(s.steps)?;
    let mut samples = Vec::new();
    let mut prefill_ms = Vec::new();
    let mut prefill_gpu_ms = Vec::new();
    let mut run_wall_ms = Vec::new();
    for run in 0..s.runs {
        model.reset()?;
        let begin = event(&model)?;
        let end = event(&model)?;
        model.record(&begin)?;
        let start = Instant::now();
        let first = model.prefill(ids, None)?;
        model.record(&end)?;
        end.synchronize()?;
        prefill_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        prefill_gpu_ms.push(begin.elapsed_ms(&end)? as f64);
        if s.graph {
            model.capture()?;
        }
        for _ in 0..s.warmup {
            model.step(s.graph)?;
        }
        model.rewind(ids.len(), first)?;
        let start = Instant::now();
        for (step, (begin, end)) in pairs.iter().enumerate() {
            model.record(begin)?;
            model.step(s.graph)?;
            model.record(end)?;
            model.copy_logits(&mut sampled_logits, step)?;
        }
        model.synchronize()?;
        run_wall_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        let last_token = model.token()?;
        let checked_values = model.check_samples(&sampled_logits)?;
        for (step, (begin, end)) in pairs.iter().enumerate() {
            let ms = begin.elapsed_ms(end)? as f64;
            ensure!(ms.is_finite() && ms > 0.0, "invalid CUDA sample time");
            let kv_tokens = ids.len() + step + 1;
            let kv_read_bytes = model.kv_read_bytes(kv_tokens);
            let weight_read_bytes = model.decode_weight_bytes();
            let bytes = weight_read_bytes + kv_read_bytes;
            let sample = Sample {
                run,
                step,
                kv_tokens,
                ms,
                kv_read_bytes,
                weight_read_bytes,
                gb_s: bytes as f64 / (ms * 1e6),
                roofline_pct: bytes as f64 / (s.dram_gb_s * 1e9 * ms * 1e-3) * 100.0,
            };
            println!("{}", serde_json::json!({"decode_sample":sample}));
            samples.push(sample);
        }
        println!(
            "{}",
            serde_json::json!({"run":run,"prefill_wall_ms":prefill_ms[run],"prefill_gpu_ms":prefill_gpu_ms[run],"decode_run_wall_ms":run_wall_ms[run],"last_token":last_token,"finite_logits_checked":checked_values})
        );
    }
    let times: Vec<_> = samples.iter().map(|x| x.ms).collect();
    let roof: Vec<_> = samples.iter().map(|x| x.roofline_pct).collect();
    let median = percentile(&times, 0.5);
    let h = model.config.hidden_size as f64;
    let f = model.config.intermediate_size as f64;
    let qkv = model.config.qkv_dim() as f64;
    let t = ids.len() as f64;
    let linear_flops =
        2.0 * t * model.config.num_hidden_layers as f64 * (h * qkv + h * h + 3.0 * h * f)
            + 2.0 * h * model.config.vocab_size as f64;
    let attention_flops = 2.0 * h * t * (t + 1.0) * model.config.num_hidden_layers as f64;
    let prefill_ms_gpu = percentile(&prefill_gpu_ms, 0.5);
    let prefill_bandwidth_floor_ms =
        (model.decode_weight_bytes() + model.kv_read_bytes(ids.len())) as f64 / (s.dram_gb_s * 1e9)
            * 1000.0;
    let prefill_compute_floor_ms =
        (linear_flops + attention_flops) / (s.gemm_tflops * 1e12) * 1000.0;
    println!(
        "{}",
        serde_json::json!({"benchmark_summary":{
        "batch":1,"graph":s.graph,"prompt_tokens":ids.len(),"samples":samples.len(),"steps_per_run":s.steps,"warmup_steps":s.warmup,
        "prefill_chunk":model.chunk(),"prefill_attention":"FA2 BF16 d128 causal",
        "kernel_settings":s.kernels,"attention_chunk":model.attention_chunk(),"attention_plan":model.attention_plan(),"decode_linear":model.decode_linear(),
        "context_read_min":ids.len()+1,"context_read_max":ids.len()+s.steps,"decode_ms_median":median,
        "decode_ms_p10":percentile(&times,0.1),"decode_ms_p90":percentile(&times,0.9),"decode_tok_s":1000.0/median,
        "decode_roofline_pct_median":percentile(&roof,0.5),"dram_denominator_gb_s":s.dram_gb_s,
        "checkpoint_weight_bytes":model.weight_bytes(),"decode_weight_read_bytes":model.decode_weight_bytes(),
        "prefill_wall_ms_median":percentile(&prefill_ms,0.5),"prefill_gpu_ms_median":percentile(&prefill_gpu_ms,0.5),
        "prefill_tok_s":t*1000.0/percentile(&prefill_ms,0.5),"prefill_linear_flops":linear_flops,"prefill_causal_attention_flops":attention_flops,
        "prefill_ideal_minimum_read_bytes":model.decode_weight_bytes()+model.kv_read_bytes(ids.len()),
        "prefill_compute_denominator_tflops":s.gemm_tflops,"prefill_compute_denominator_source":"CLI measured calibration input",
        "prefill_compute_floor_ms":prefill_compute_floor_ms,"prefill_bandwidth_floor_ms":prefill_bandwidth_floor_ms,
        "prefill_roofline_pct":prefill_compute_floor_ms.max(prefill_bandwidth_floor_ms)/prefill_ms_gpu*100.0,
        "load_ms":load_ms,"ignore_eos":true,"validation_copy_bytes_per_step":vocab*2,
        "timing_scope":"CUDA events around graph step; D2D logit snapshots and final D2H validation outside sample intervals"}})
    );
    Ok(())
}

pub fn profile(dir: &Path, ids: &[u32], s: &Settings, phase: Phase) -> Result<()> {
    check(ids, s)?;
    let mut model = Model::load(dir, s.device, s.capacity, s.chunk, s.kernels.clone())?;
    warm(&mut model, ids, s)?;
    if matches!(phase, Phase::Prefill) {
        model.reset()?;
    }
    let begin = event(&model)?;
    let end = event(&model)?;
    model.synchronize()?;
    model.profiler_start()?;
    let result = (|| -> Result<()> {
        model.record(&begin)?;
        match phase {
            Phase::Decode => model.step(s.graph)?,
            Phase::Prefill => {
                model.prefill(ids, None)?;
            }
        }
        model.record(&end)?;
        end.synchronize()?;
        Ok(())
    })();
    let stop = model.profiler_stop();
    result?;
    stop?;
    model.check_logits()?;
    println!(
        "{}",
        serde_json::json!({"profile":{"phase":phase,"graph":s.graph,"prompt_tokens":ids.len(),"prefill_chunk":model.chunk(),"kernel_settings":s.kernels,"attention_chunk":model.attention_chunk(),"gpu_ms":begin.elapsed_ms(&end)?,"checkpoint_weight_bytes":model.weight_bytes(),"decode_weight_read_bytes":model.decode_weight_bytes(),"kv_read_bytes_at_first_step":model.kv_read_bytes(ids.len()+1)}})
    );
    Ok(())
}
