//! Music3 synthesis engine gates. GPU gates are `#[ignore]`d and run under a
//! coordinator-granted window with `FF_MUSIC3_MODEL` naming the official
//! weights directory; the frozen cells come from
//! `~/ff-work/minimax/evidence/oracle/frozen-r3-syn` (input captures under
//! `evidence/oracle/captures`).

#[cfg(feature = "cuda")]
mod gates {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::Context;

    fn model_dir() -> anyhow::Result<std::path::PathBuf> {
        let dir = std::env::var("FF_MUSIC3_MODEL").context("FF_MUSIC3_MODEL not set")?;
        Ok(std::path::PathBuf::from(dir))
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL=<dir>"]
    fn music3_syn_real_load_gpu() -> anyhow::Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(&model_dir()?, &device)?;
        assert_eq!(
            engine.weights.flow.blocks.len(),
            engine.spec.flow_layers as usize
        );
        assert_eq!(
            engine.weights.vocoder.blocks.len(),
            engine.spec.upsampling.len()
        );
        println!(
            "MUSIC3_SYN_LOAD ok flow={}x{}x{} cond={} vocoder_stages={} math=CUBLAS_DEFAULT_MATH",
            engine.spec.flow_layers,
            engine.spec.flow_heads,
            engine.spec.flow_head_dim,
            engine.spec.cond_layers,
            engine.spec.upsampling.len()
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod velocity {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
    use safetensors::SafeTensors;

    fn env_path(name: &str) -> Result<std::path::PathBuf> {
        std::env::var(name)
            .with_context(|| format!("{name} not set"))
            .map(std::path::PathBuf::from)
    }

    fn frozen_bound(root: &serde_json::Value, cell: &str) -> Result<f64> {
        root.get("cells")
            .and_then(|c| c.get(cell))
            .and_then(|c| c.get("bound"))
            .and_then(serde_json::Value::as_f64)
            .with_context(|| format!("frozen cell {cell} missing"))
    }

    fn tensor_f32<'a>(view: &'a SafeTensors<'a>, name: &str) -> Result<Vec<f32>> {
        let t = view
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        ensure!(t.dtype() == safetensors::Dtype::F32, "{name} not F32");
        let bytes = t.data();
        ensure!(bytes.len() % 4 == 0, "{name} unaligned F32 bytes");
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE, FF_MUSIC3_FROZEN"]
    fn music3_syn_flow_velocity_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(&env_path("FF_MUSIC3_MODEL")?, &device)?;
        let capture = std::fs::read(env_path("FF_MUSIC3_CAPTURE")?)?;
        let view = SafeTensors::deserialize(&capture)?;
        let frozen: serde_json::Value =
            serde_json::from_slice(&std::fs::read(env_path("FF_MUSIC3_FROZEN")?)?)?;
        let bound_cond = frozen_bound(&frozen, "flow_velocity_cond")?;
        let bound_uncond = frozen_bound(&frozen, "flow_velocity_uncond")?;
        let bound_guided = frozen_bound(&frozen, "flow_velocity_guided")?;

        let length: usize = view
            .tensor("syn_x_t_000")?
            .shape()
            .last()
            .copied()
            .context("empty x_t shape")?;
        let tokens = length + 1;
        let rotary_dim = engine.spec.rotary_dim as usize;
        let half = rotary_dim / 2;
        let mut cos_table = Vec::with_capacity(tokens * rotary_dim);
        let mut sin_table = Vec::with_capacity(tokens * rotary_dim);
        for pos in 0..tokens {
            let mut freqs = Vec::with_capacity(half);
            for i in 0..half {
                let inv = 1.0f32 / 10000.0f32.powf((2 * i) as f32 / rotary_dim as f32);
                freqs.push(pos as f32 * inv);
            }
            for _ in 0..2 {
                for f in &freqs {
                    cos_table.push(f.cos());
                    sin_table.push(f.sin());
                }
            }
        }
        let timesteps = tensor_f32(&view, "syn_timesteps")?;
        let s = &engine.stream;
        let d_cos = s.clone_htod(&cos_table)?;
        let d_sin = s.clone_htod(&sin_table)?;

        let mut max_cond = 0.0f64;
        let mut max_uncond = 0.0f64;
        let mut engine_cond = Vec::new();
        let mut engine_uncond = Vec::new();
        let mut velocity = unsafe { s.alloc::<f32>(engine.spec.in_channels as usize * length)? };
        for (index, t) in (0..60).zip(&timesteps) {
            let x_t = tensor_f32(&view, &format!("syn_x_t_{index:03}"))?;
            let cond = tensor_f32(&view, &format!("syn_cond_{index:03}"))?;
            let d_x = s.clone_htod(&x_t)?;
            let d_cond = s.clone_htod(&cond)?;
            engine.flow_velocity(&d_x, *t, &d_cond, (&d_cos, &d_sin), length, &mut velocity)?;
            let got = s.clone_dtoh(&velocity)?;
            let want = tensor_f32(&view, &format!("syn_velocity_{index:03}"))?;
            let delta = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() as f64)
                .fold(0.0f64, f64::max);
            if index % 2 == 0 {
                engine_cond.push(got);
                max_cond = max_cond.max(delta);
            } else {
                engine_uncond.push(got);
                max_uncond = max_uncond.max(delta);
            }
        }
        println!(
            "MUSIC3_FLOW_VELOCITY cond={max_cond:.6e} (bound {bound_cond:.6e}) uncond={max_uncond:.6e} (bound {bound_uncond:.6e})"
        );
        ensure!(max_cond <= bound_cond, "cond cell exceeds frozen bound");
        ensure!(
            max_uncond <= bound_uncond,
            "uncond cell exceeds frozen bound"
        );

        let mut max_guided = 0.0f64;
        for step in 0..30 {
            let want = tensor_f32(&view, &format!("syn_velocity_guided_{step:03}"))?;
            let (c, u) = (&engine_cond[step], &engine_uncond[step]);
            let delta = c
                .iter()
                .zip(u)
                .zip(&want)
                .map(|((a, b), w)| (b + 1.7 * (a - b) - w).abs() as f64)
                .fold(0.0f64, f64::max);
            max_guided = max_guided.max(delta);
        }
        println!("MUSIC3_FLOW_GUIDED max_abs={max_guided:.6e} (bound {bound_guided:.6e})");
        ensure!(
            max_guided <= bound_guided,
            "guided cell exceeds frozen bound"
        );

        // Control: dropping the condition on a cond call must exceed the
        // frozen bound (the gate discriminates a silent condition loss).
        let x_t = tensor_f32(&view, "syn_x_t_010")?;
        let zeros = vec![0.0f32; length * engine.spec.condition_dim as usize];
        let d_x = s.clone_htod(&x_t)?;
        let d_zero = s.clone_htod(&zeros)?;
        engine.flow_velocity(
            &d_x,
            timesteps[10],
            &d_zero,
            (&d_cos, &d_sin),
            length,
            &mut velocity,
        )?;
        let got = s.clone_dtoh(&velocity)?;
        let want = tensor_f32(&view, "syn_velocity_010")?;
        let control = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        ensure!(
            control > bound_cond,
            "dropped-condition control did not exceed the bound"
        );
        println!("MUSIC3_FLOW_CONTROL dropped_condition={control:.6e} > bound");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod probe {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result};
    use safetensors::SafeTensors;

    fn tensor_f32<'a>(view: &'a SafeTensors<'a>, name: &str) -> Result<Vec<f32>> {
        let t = view
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        let bytes = t.data();
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() as f64)
            .fold(0.0, f64::max)
    }

    /// Per-stage bisection of the flow forward against definition taps for
    /// capture call 000. Prints every stage delta; the first gross delta
    /// names the broken stage.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE, FF_MUSIC3_PROBE"]
    fn music3_syn_flow_probe_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
        )?;
        let capture = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CAPTURE",
        )?))?;
        let view = SafeTensors::deserialize(&capture)?;
        let probe_file =
            std::fs::read(std::path::PathBuf::from(std::env::var("FF_MUSIC3_PROBE")?))?;
        let probe = SafeTensors::deserialize(&probe_file)?;

        let call: String = std::env::var("FF_MUSIC3_PROBE_CALL").unwrap_or_else(|_| "000".into());
        let length = 86usize;
        let tokens = length + 1;
        let rotary_dim = engine.spec.rotary_dim as usize;
        let half = rotary_dim / 2;
        let dim = engine.spec.flow_dim() as usize;
        let mut cos_table = Vec::with_capacity(tokens * rotary_dim);
        let mut sin_table = Vec::with_capacity(tokens * rotary_dim);
        for pos in 0..tokens {
            let freqs: Vec<f32> = (0..half)
                .map(|i| pos as f32 / 10000.0f32.powf((2 * i) as f32 / rotary_dim as f32))
                .collect();
            for _ in 0..2 {
                for f in &freqs {
                    cos_table.push(f.cos());
                    sin_table.push(f.sin());
                }
            }
        }
        let x_t = tensor_f32(&view, &format!("syn_x_t_{call}"))?;
        let cond = tensor_f32(&view, &format!("syn_cond_{call}"))?;
        let t = tensor_f32(&view, "syn_timesteps")?[call.parse::<usize>()?];
        let s = &engine.stream;
        let d_x = s.clone_htod(&x_t)?;
        let d_cond = s.clone_htod(&cond)?;
        let d_cos = s.clone_htod(&cos_table)?;
        let d_sin = s.clone_htod(&sin_table)?;
        let mut velocity = unsafe { s.alloc::<f32>(engine.spec.in_channels as usize * length)? };
        let mut taps: Vec<(&'static str, Vec<f32>)> = Vec::new();
        engine.flow_velocity_probed(
            &d_x,
            t,
            &d_cond,
            (&d_cos, &d_sin),
            length,
            &mut velocity,
            Some(&mut taps),
        )?;

        // preprocess: engine [length, concat] rows vs definition [concat, length]
        let want_pre = tensor_f32(&probe, "preprocess_out")?;
        for (name, got) in &taps {
            match *name {
                "preprocess" => {
                    let mut transposed = vec![0.0f32; got.len()];
                    for (tt, row) in got.chunks(2304).enumerate() {
                        for (c, v) in row.iter().enumerate() {
                            transposed[c * length + tt] = *v;
                        }
                    }
                    println!(
                        "PROBE preprocess max_abs={:.6e}",
                        max_abs(&transposed, &want_pre)
                    );
                }
                "fourier" => {
                    let want = tensor_f32(&probe, "time_proj_out")?;
                    println!("PROBE fourier max_abs={:.6e}", max_abs(got, &want));
                    let head: Vec<String> = got
                        .iter()
                        .zip(&want)
                        .take(6)
                        .map(|(a, b)| format!("{a:.4}/{b:.4}"))
                        .collect();
                    println!("PROBE fourier head engine/want: {}", head.join(" "));
                }
                "time_l1" => {
                    if probe.tensor("time_l1_out").is_ok() {
                        println!(
                            "PROBE time_l1(bias,silu) max_abs={:.6e}",
                            max_abs(got, &tensor_f32(&probe, "time_l1_out")?)
                        );
                    }
                }
                "temb" => println!(
                    "PROBE temb max_abs={:.6e}",
                    max_abs(got, &tensor_f32(&probe, "time_l2_out")?)
                ),
                "wide_after_proj_in" => {
                    let want = tensor_f32(&probe, "proj_in_out")?;
                    let body: Vec<f32> = got[dim..].to_vec();
                    println!("PROBE proj_in max_abs={:.6e}", max_abs(&body, &want));
                }
                "block0_norm1" => println!(
                    "PROBE block0_norm1 max_abs={:.6e}",
                    max_abs(got, &tensor_f32(&probe, "block0_norm1")?)
                ),
                "block0_attn" => println!(
                    "PROBE block0_attn max_abs={:.6e}",
                    max_abs(got, &tensor_f32(&probe, "block0_attn")?)
                ),
                "block0_out" => println!(
                    "PROBE block0_out max_abs={:.6e}",
                    max_abs(got, &tensor_f32(&probe, "block0_out")?)
                ),
                "proj_out" => println!(
                    "PROBE proj_out max_abs={:.6e}",
                    max_abs(got, &tensor_f32(&probe, "proj_out_out")?)
                ),
                _ => {}
            }
        }
        let got_velocity = s.clone_dtoh(&velocity)?;
        println!(
            "PROBE velocity max_abs={:.6e}",
            max_abs(&got_velocity, &tensor_f32(&probe, "velocity_cpu")?)
        );
        Ok(())
    }
}
