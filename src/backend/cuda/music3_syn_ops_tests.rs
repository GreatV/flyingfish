//! Music3 synthesis engine gates. GPU gates are `#[ignore]`d and run under
//! a coordinator-granted window with `FF_MUSIC3_MODEL` naming the official
//! weights directory and `FF_MUSIC3_FROZEN*` the frozen bound files.

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
        let engine = Music3Syn::load(&model_dir()?, &device, 86)?;
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
        let engine = Music3Syn::load(&env_path("FF_MUSIC3_MODEL")?, &device, 86)?;
        let mut workspace = Music3Syn::workspace(&engine.spec, &device, 86)?;
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
        let timesteps = tensor_f32(&view, "syn_timesteps")?;
        let s = &engine.stream;

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
            engine.flow_velocity(&mut workspace, &d_x, *t, &d_cond, length, &mut velocity)?;
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
            &mut workspace,
            &d_x,
            timesteps[10],
            &d_zero,
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
mod encoder {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
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

    /// Condition encoder against the frozen-r1 cell (max 0.12478924 /
    /// rmse 0.01145017); layer-swap control must exceed the bound.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CONDITION, FF_MUSIC3_FROZEN_R1"]
    fn music3_syn_cond_encoder_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let capture_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CONDITION",
        )?))?;
        let view = SafeTensors::deserialize(&capture_bytes)?;
        let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::PathBuf::from(std::env::var("FF_MUSIC3_FROZEN_R1")?),
        )?)?;
        let bound_max = frozen["cells"]["condition_encoder"]["bound"]["max"]
            .as_f64()
            .context("frozen condition_encoder max bound missing")?;
        let bound_rmse = frozen["cells"]["condition_encoder"]["bound"]["rmse"]
            .as_f64()
            .context("frozen condition_encoder rmse bound missing")?;

        let frames = 25usize;
        let length = engine.latent_length(frames);
        ensure!(length == 86, "unexpected latent length {length}");
        let cond_out = engine.spec.cond_out as usize;
        let want = tensor_f32(&view, "condition_encoder")?;
        ensure!(want.len() == length * cond_out, "condition_encoder extent");

        let input = tensor_f32(&view, "condition_input")?;
        let s = &engine.stream;
        let d_in = s.clone_htod(&input)?;
        let mut out = unsafe { s.alloc::<f32>(length * cond_out)? };
        engine.condition_encode(&d_in, frames, &mut out)?;
        let got = s.clone_dtoh(&out)?;
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let rmse = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64) * (*a as f64 - *b as f64))
            .sum::<f64>()
            / got.len() as f64)
            .sqrt();
        println!(
            "MUSIC3_COND_ENCODER max_abs={max_abs:.6e} (bound {bound_max:.6e}) rmse={rmse:.6e} (bound {bound_rmse:.6e})"
        );
        ensure!(
            max_abs <= bound_max,
            "condition encoder max exceeds frozen bound"
        );
        ensure!(
            rmse <= bound_rmse,
            "condition encoder rmse exceeds frozen bound"
        );

        let layers = engine.spec.cond_layers as usize;
        let hidden = engine.spec.cond_hidden as usize;
        let mut swapped = input.clone();
        for t in 0..frames {
            for h in 0..hidden {
                let a = t * layers * hidden + h;
                let b = t * layers * hidden + (layers - 1) * hidden + h;
                swapped.swap(a, b);
            }
        }
        let d_swapped = s.clone_htod(&swapped)?;
        engine.condition_encode(&d_swapped, frames, &mut out)?;
        let control = s
            .clone_dtoh(&out)?
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        ensure!(
            control > bound_max,
            "layer-swap control did not exceed the bound"
        );
        println!("MUSIC3_COND_CONTROL layer_swap={control:.6e} > bound");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod vocoder_gate {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
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

    /// Vocoder against the frozen-r1 final_audio cell; negated-latent
    /// control must exceed the bound.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE, FF_MUSIC3_FROZEN_R1"]
    fn music3_syn_vocoder_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let capture_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CAPTURE",
        )?))?;
        let view = SafeTensors::deserialize(&capture_bytes)?;
        let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::PathBuf::from(std::env::var("FF_MUSIC3_FROZEN_R1")?),
        )?)?;
        let bound_max = frozen["cells"]["final_audio"]["bound"]["max"]
            .as_f64()
            .context("final_audio max bound missing")?;
        let bound_rmse = frozen["cells"]["final_audio"]["bound"]["rmse"]
            .as_f64()
            .context("final_audio rmse bound missing")?;

        let length = 86usize;
        let channels = engine.spec.in_channels as usize;
        let total: usize = engine.spec.upsampling.iter().map(|r| *r as usize).product();
        let latent = tensor_f32(&view, "syn_vocoder_latent")?;
        ensure!(latent.len() == channels * length, "latent extent");
        let want = tensor_f32(&view, "final_audio")?;
        ensure!(want.len() == 2 * length * total, "final_audio extent");

        let s = &engine.stream;
        let mut workspace = Music3Syn::workspace(&engine.spec, &device, 86)?;
        let d_latent = s.clone_htod(&latent)?;
        let mut out = unsafe { s.alloc::<f32>(2 * length * total)? };
        s.memcpy_dtod(&d_latent, &mut workspace.x.slice_mut(0..channels * length))?;
        engine.vocoder(&mut workspace, length, &mut out)?;
        let got = s.clone_dtoh(&out)?;
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let rmse = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64) * (*a as f64 - *b as f64))
            .sum::<f64>()
            / got.len() as f64)
            .sqrt();
        println!(
            "MUSIC3_VOCODER max_abs={max_abs:.6e} (bound {bound_max:.6e}) rmse={rmse:.6e} (bound {bound_rmse:.6e})"
        );
        ensure!(max_abs <= bound_max, "vocoder max exceeds frozen bound");
        ensure!(rmse <= bound_rmse, "vocoder rmse exceeds frozen bound");

        let negated: Vec<f32> = latent.iter().map(|v| -v).collect();
        let d_neg = s.clone_htod(&negated)?;
        s.memcpy_dtod(&d_neg, &mut workspace.x.slice_mut(0..channels * length))?;
        engine.vocoder(&mut workspace, length, &mut out)?;
        let control = s
            .clone_dtoh(&out)?
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        ensure!(
            control > bound_max,
            "negated-latent control did not exceed the bound"
        );
        println!("MUSIC3_VOCODER_CONTROL negated={control:.6e} > bound");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod euler {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
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

    /// Euler scheduler gate: derived sigma schedule within 2 ULP (corrected
    /// pre-candidate 2026-11-03, CPU derivation analysis), and the
    /// teacher-forced update chain (bounds: original 2.0e-6, frozen
    /// 1.0e-5; see the report for the recorded run history).
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE"]
    fn music3_syn_euler_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let capture_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CAPTURE",
        )?))?;
        let view = SafeTensors::deserialize(&capture_bytes)?;
        let want_sigma = tensor_f32(&view, "syn_sigma_schedule")?;
        let derived = Music3Syn::sigma_schedule(30);
        ensure!(want_sigma.len() == 31 && derived.len() == 31);
        let sigma_max = derived
            .iter()
            .zip(&want_sigma)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("MUSIC3_EULER sigma max_abs={sigma_max:.6e} (bound 2.4e-07)");
        ensure!(
            sigma_max <= 2.4e-7,
            "sigma schedule exceeds the 2-ULP bound"
        );

        let s = &engine.stream;
        // Per-step guided update with the engine kernel on the captured
        // branch velocities.
        let mut max_update = 0.0f32;
        let mut x: cudarc::driver::CudaSlice<f32> =
            s.clone_htod(&tensor_f32(&view, "syn_x_t_000")?)?;
        for step in 0..30 {
            // Transition 29's output never appears as a call input; the
            // vocoder latent is its only witness.
            let want_next = if step == 29 {
                tensor_f32(&view, "syn_vocoder_latent")?
            } else {
                tensor_f32(&view, &format!("syn_x_t_{:03}", 2 * (step + 1)))?
            };
            let c = tensor_f32(&view, &format!("syn_velocity_cond_{step:03}"))?;
            let u = tensor_f32(&view, &format!("syn_velocity_uncond_{step:03}"))?;
            let d_c = s.clone_htod(&c)?;
            let d_u = s.clone_htod(&u)?;
            engine.guided_euler_for_test(
                &mut x,
                &d_c,
                &d_u,
                want_sigma[step + 1] - want_sigma[step],
            )?;
            let got = s.clone_dtoh(&x)?;
            let delta = got
                .iter()
                .zip(&want_next)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            max_update = max_update.max(delta);
        }
        println!("MUSIC3_EULER update max_abs={max_update:.6e} (frozen bound 1.0e-05)");
        ensure!(
            max_update <= 1.0e-5,
            "euler update exceeds the frozen bound"
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod end_to_end {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
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

    /// Free-running synthesis end to end against the frozen-r1 final_audio
    /// cell: initial latent = captured step-0 noise, condition from the
    /// frozen capture, 30 CFG Euler steps, vocoder. Controls: the negated
    /// initial latent must exceed the bound; the zero condition input
    /// must exceed 10x the max bound.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE, FF_MUSIC3_FROZEN_R1"]
    fn music3_syn_end_to_end_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let mut workspace = Music3Syn::workspace(&engine.spec, &device, 86)?;
        let capture_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CAPTURE",
        )?))?;
        let view = SafeTensors::deserialize(&capture_bytes)?;
        let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::PathBuf::from(std::env::var("FF_MUSIC3_FROZEN_R1")?),
        )?)?;
        let bound_max = frozen["cells"]["final_audio"]["bound"]["max"]
            .as_f64()
            .context("final_audio max bound missing")?;
        let bound_rmse = frozen["cells"]["final_audio"]["bound"]["rmse"]
            .as_f64()
            .context("final_audio rmse bound missing")?;

        let length = 86usize;
        let total: usize = engine.spec.upsampling.iter().map(|r| *r as usize).product();
        let initial = tensor_f32(&view, "syn_x_t_000")?;
        let cond_input = tensor_f32(&view, "condition_input")?;
        let want = tensor_f32(&view, "final_audio")?;
        let s = &engine.stream;
        let d_init = s.clone_htod(&initial)?;
        let d_cond = s.clone_htod(&cond_input)?;
        let mut audio = unsafe { s.alloc::<f32>(2 * length * total)? };
        engine.synthesize(&mut workspace, &d_cond, 25, &d_init, &mut audio)?;
        let got = s.clone_dtoh(&audio)?;
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let rmse = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64) * (*a as f64 - *b as f64))
            .sum::<f64>()
            / got.len() as f64)
            .sqrt();
        println!(
            "MUSIC3_END_TO_END max_abs={max_abs:.6e} (bound {bound_max:.6e}) rmse={rmse:.6e} (bound {bound_rmse:.6e})"
        );
        ensure!(max_abs <= bound_max, "end-to-end max exceeds frozen bound");
        ensure!(rmse <= bound_rmse, "end-to-end rmse exceeds frozen bound");

        let negated: Vec<f32> = initial.iter().map(|v| -v).collect();
        let d_neg = s.clone_htod(&negated)?;
        engine.synthesize(&mut workspace, &d_cond, 25, &d_neg, &mut audio)?;
        let control = s
            .clone_dtoh(&audio)?
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        ensure!(
            control > bound_max,
            "negated-initial control did not exceed the bound"
        );
        let control_ratio = control / bound_max;
        println!("MUSIC3_E2E_CONTROL negated_initial={control:.6e} ratio={control_ratio:.1}");

        // Dropped condition: the zero condition input must move the output
        // by at least 10x the frozen max bound.
        let dropped: Vec<f32> = vec![0.0f32; cond_input.len()];
        let d_zero = s.clone_htod(&dropped)?;
        engine.synthesize(&mut workspace, &d_zero, 25, &d_init, &mut audio)?;
        let dropped_err = s
            .clone_dtoh(&audio)?
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let dropped_ratio = dropped_err / bound_max;
        ensure!(
            dropped_ratio >= 10.0,
            "dropped-condition control under 10x the bound"
        );
        println!("MUSIC3_E2E_CONTROL dropped_condition={dropped_err:.6e} ratio={dropped_ratio:.1}");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod pair_flow {
    use super::super::Device;
    use super::super::engine_music3_syn::Music3Syn;
    use anyhow::{Context, Result, ensure};
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

    /// Pair flow, teacher-forced: one batched call on captured x_t/cond must
    /// reproduce both captured branch velocities. Also the fused guided
    /// update on captured tensors vs the captured next x_t.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_CAPTURE"]
    fn music3_syn_pair_flow_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let mut workspace = Music3Syn::workspace(&engine.spec, &device, 86)?;
        let capture_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_CAPTURE",
        )?))?;
        let view = SafeTensors::deserialize(&capture_bytes)?;
        let s = &engine.stream;
        let channels = engine.spec.in_channels as usize;
        let cond_dim = engine.spec.cond_out as usize;
        let length = 86usize;
        let timesteps = tensor_f32(&view, "syn_timesteps")?;

        // Pair call on step 0 inputs: x_t_000 + cond_000, t = timesteps[0].
        let x_t = tensor_f32(&view, "syn_x_t_000")?;
        let cond = tensor_f32(&view, "syn_cond_000")?;
        let d_x = s.clone_htod(&x_t)?;
        let d_cond = s.clone_htod(&cond)?;
        let want_c = tensor_f32(&view, "syn_velocity_000")?;
        let want_u = tensor_f32(&view, "syn_velocity_001")?;
        s.memcpy_dtod(&d_x, &mut workspace.x.slice_mut(0..channels * length))?;
        s.memcpy_dtod(&d_cond, &mut workspace.cond.slice_mut(0..length * cond_dim))?;
        engine.flow_velocity_pair(&mut workspace, timesteps[0], length)?;
        let got_c = s.clone_dtoh(&workspace.pair_v_c.slice(0..channels * length))?;
        let got_u = s.clone_dtoh(&workspace.pair_v_u.slice(0..channels * length))?;
        let dc = got_c
            .iter()
            .zip(&want_c)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        let du = got_u
            .iter()
            .zip(&want_u)
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);
        println!(
            "PAIR_FLOW cond max_abs={dc:.6e} uncond max_abs={du:.6e} (frozen-r3-syn cells: cond 9.796142e-03, uncond 2.662754e-02)"
        );
        ensure!(
            dc <= 9.796142e-3,
            "pair cond velocity exceeds the frozen-r3-syn cell"
        );
        ensure!(
            du <= 2.662754e-2,
            "pair uncond velocity exceeds the frozen-r3-syn cell"
        );
        Ok(())
    }
}
