use crate::backend;
use crate::model::Model;
use crate::model_spec::{Architecture, ModelSpec};
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

use crate::verify::percentile;

fn event(model: &Model) -> Result<backend::Event> {
    model.event()
}

fn check(dir: &Path, ids: &[u32], s: &Settings) -> Result<()> {
    let architecture = ModelSpec::read(dir)?.architecture;
    ensure!(
        architecture == Architecture::Llama,
        "benchmark accounting is not implemented for {architecture:?}; only Llama models are supported"
    );
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
    check(dir, ids, s)?;
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
        "prefill_compute_denominator_tflops":s.gemm_tflops,"prefill_compute_denominator_source":"configured reference input",
        "prefill_compute_floor_ms":prefill_compute_floor_ms,"prefill_bandwidth_floor_ms":prefill_bandwidth_floor_ms,
        "prefill_roofline_pct":prefill_compute_floor_ms.max(prefill_bandwidth_floor_ms)/prefill_ms_gpu*100.0,
        "load_ms":load_ms,"ignore_eos":true,"validation_copy_bytes_per_step":vocab*2,
        "timing_scope":"CUDA events around graph step; D2D logit snapshots and final D2H validation outside sample intervals"}})
    );
    Ok(())
}

pub fn profile(dir: &Path, ids: &[u32], s: &Settings, phase: Phase) -> Result<()> {
    check(dir, ids, s)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_llama_accounting_is_rejected_before_loading_weights() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("ff-bench-accounting-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        std::fs::write(
            dir.join("config.json"),
            include_bytes!("../tests/data/q38-config.json"),
        )?;
        let settings = Settings {
            device: 0,
            capacity: 512,
            dram_gb_s: 1.0,
            gemm_tflops: 1.0,
            chunk: None,
            kernels: backend::Settings {
                tree_builder: backend::TreeBuilder::Waves,
                spec_budget: backend::SpecBudget::Chain,
                spec_graph: false,
                draft_model: None,
                runtime_dir: None,
                prefix_cache: false,
                kv_format: crate::kv::KvFormat::Bf16,
            },
            steps: 200,
            warmup: 0,
            runs: 1,
            graph: false,
        };
        let results = [
            run(&dir, &[0], &settings),
            profile(&dir, &[0], &settings, Phase::Decode),
            profile(&dir, &[0], &settings, Phase::Prefill),
        ];
        std::fs::remove_dir_all(&dir)?;
        for result in results {
            let error = result.unwrap_err().to_string();
            assert!(
                error.contains("accounting is not implemented for Qwen35Text"),
                "{error}"
            );
        }
        Ok(())
    }

    #[test]
    fn percentiles_interpolate_short_and_tail_distributions() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0, 4.0], 0.5), 2.5);
        assert_eq!(percentile(&[10.0], 0.1), 10.0);
        let x: Vec<_> = (0..201).map(|i| i as f64).collect();
        assert_eq!(percentile(&x, 0.1), 20.0);
        assert_eq!(percentile(&x, 0.9), 180.0);
    }
}
