//! Music3 synthesis engine gates. GPU gates are `#[ignore]`d and run under
//! a coordinator-granted window with `FF_MUSIC3_MODEL` naming the official
//! weights directory; frozen cells live in the ff-work evidence tree.

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
        let d_latent = s.clone_htod(&latent)?;
        let mut out = unsafe { s.alloc::<f32>(2 * length * total)? };
        engine.vocoder(&d_latent, length, &mut out)?;
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
        engine.vocoder(&d_neg, length, &mut out)?;
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
        println!(
            "MUSIC3_EULER sigma max_abs={sigma_max:.6e} (bound 2.4e-07, corrected pre-candidate 2026-11-03)"
        );
        ensure!(
            sigma_max <= 2.4e-7,
            "sigma schedule exceeds the corrected 2-ULP bound"
        );

        let s = &engine.stream;
        let channels = engine.spec.in_channels as usize;
        let length = 86usize;
        let mut max_update = 0.0f32;
        let mut x: cudarc::driver::CudaSlice<f32> =
            s.clone_htod(&tensor_f32(&view, "syn_x_t_000")?)?;
        let mut velocity: cudarc::driver::CudaSlice<f32> =
            s.clone_htod(&vec![0.0f32; channels * length])?;
        for step in 0..29 {
            let want_next = tensor_f32(&view, &format!("syn_x_t_{:03}", 2 * (step + 1)))?;
            let c = tensor_f32(&view, &format!("syn_velocity_cond_{step:03}"))?;
            let u = tensor_f32(&view, &format!("syn_velocity_uncond_{step:03}"))?;
            let guided: Vec<f32> = c
                .iter()
                .zip(&u)
                .map(|(a, b)| b + 1.7f32 * (a - b))
                .collect();
            let d_guided = s.clone_htod(&guided)?;
            s.memcpy_dtod(&d_guided, &mut velocity)?;
            engine.euler_step(
                &mut x,
                &velocity,
                want_sigma[step + 1],
                want_sigma[step + 2],
            )?;
            let got = s.clone_dtoh(&x)?;
            let delta = got
                .iter()
                .zip(&want_next)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            max_update = max_update.max(delta);
        }
        println!(
            "MUSIC3_EULER update max_abs={max_update:.6e} (frozen bound 1.0e-05; original bound 2.0e-06)"
        );
        ensure!(
            max_update <= 1.0e-5,
            "euler update exceeds the new frozen bound"
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
    /// frozen capture, 30 CFG Euler steps, vocoder. Negative control: the
    /// negated initial latent must exceed the bound.
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
        println!("MUSIC3_E2E_CONTROL negated_initial={control:.6e} > bound");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod vocoder_disc {
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

    /// Vocoder stage discriminator: dec_in_proj, conv_in, block0 convT and
    /// unit1 against the definition taps (engine rows [T, C] transposed vs
    /// definition [2, C, T]). Splits the im2col conv from convT.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_MUSIC3_MODEL, FF_MUSIC3_VOCODER_PROBE"]
    fn music3_syn_vocoder_disc_gpu() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let device = Device::new(0)?;
        let engine = Music3Syn::load(
            &std::path::PathBuf::from(std::env::var("FF_MUSIC3_MODEL")?),
            &device,
            86,
        )?;
        let probe_bytes = std::fs::read(std::path::PathBuf::from(std::env::var(
            "FF_MUSIC3_VOCODER_PROBE",
        )?))?;
        let probe = SafeTensors::deserialize(&probe_bytes)?;
        let want_dec = tensor_f32(&probe, "dec_in_out")?;
        let want_conv_in = tensor_f32(&probe, "conv_in_out")?;
        let want_convt = tensor_f32(&probe, "block0_convt")?;
        let want_u1 = tensor_f32(&probe, "block0_u1")?;

        let s = &engine.stream;
        let latent = tensor_f32(&probe, "latent")?;
        let length = 86usize;
        let streams = engine.spec.vocoder_stream_channels() as usize;
        let input_dim = engine.spec.decoder_input as usize;
        let hidden = engine.spec.decoder_hidden as usize;
        ensure!(latent.len() == 2 * streams * length, "latent extent");

        // dec_in_proj per stream: x = latent half transposed; GEMM + bias.
        let mut buf: Vec<f32> = vec![0.0; 2 * length * input_dim];
        for half in 0..2 {
            let half_latent = &latent[half * streams * length..(half + 1) * streams * length];
            let d_half = s.clone_htod(half_latent)?;
            let mut x = unsafe { s.alloc::<f32>(length * streams)? };
            engine.transpose_for_disc(&d_half, &mut x, streams, length)?;
            let mut out = unsafe { s.alloc::<f32>(length * input_dim)? };
            {
                use super::super::engine_music3_syn::LinearDisc;
                engine.linear_for_disc(LinearDisc {
                    weight: &engine.weights.vocoder.dec_in_proj_w,
                    out_rows: input_dim,
                    in_cols: streams,
                    x: &x,
                    rows: length,
                    out: &mut out,
                    bias: &engine.weights.vocoder.dec_in_proj_b,
                })?;
            }
            let got = s.clone_dtoh(&out)?;
            let want = &want_dec[half * input_dim * length..];
            let mut worst = 0.0f64;
            for t in 0..length {
                for c in 0..input_dim {
                    worst = worst
                        .max((got[t * input_dim + c] as f64 - want[c * length + t] as f64).abs());
                }
            }
            println!("DISC_V s{half} dec_in max_abs={worst:.6e}");
            buf[half * length * input_dim..(half + 1) * length * input_dim].copy_from_slice(&got);
        }

        // conv_in per stream on the published folded weight.
        let d_buf = s.clone_htod(&buf)?;
        let want_conv = [want_conv_in.as_slice(), want_conv_in.as_slice()];
        let _ = &want_conv;
        for half in 0..2 {
            let src = d_buf.slice(half * length * input_dim..(half + 1) * length * input_dim);
            let mut wide = unsafe { s.alloc::<f32>(length * hidden)? };
            engine.conv1d_for_disc(
                &engine.vocoder_folds_for_disc().conv_in_w,
                &src,
                length,
                input_dim,
                &mut wide,
            )?;
            let got = s.clone_dtoh(&wide)?;
            let want = &want_conv_in[half * hidden * length..];
            let mut worst = 0.0f64;
            for t in 0..length {
                for c in 0..hidden {
                    worst =
                        worst.max((got[t * hidden + c] as f64 - want[c * length + t] as f64).abs());
                }
            }
            println!("DISC_V s{half} conv_in max_abs={worst:.6e}");
        }
        // Stage 2: block0 convT + unit1 on stream 0 (conv_in output as the
        // snake input, then the transposed conv, then unit1).
        let d_buf0 = s.clone_htod(&buf[0..length * input_dim].to_vec())?;
        let mut wide = unsafe { s.alloc::<f32>(length * hidden)? };
        {
            let folds = engine.vocoder_folds_for_disc();
            engine.conv1d_for_disc(
                &folds.conv_in_w,
                &d_buf0.slice(0..length * input_dim),
                length,
                input_dim,
                &mut wide,
            )?;
        }
        // snake1 on the block input.
        let count = length * hidden;
        let mut snaked = unsafe { s.alloc::<f32>(count)? };
        engine.snake_rows_for_disc(
            &wide,
            &engine.weights.vocoder.blocks[0].snake1_alpha,
            &mut snaked,
            count,
            hidden,
        )?;
        // convT: [length, hidden] -> [length*8, 768].
        let out_dim = hidden / 2;
        let stride = engine.spec.upsampling[0] as usize;
        let mut up = unsafe { s.alloc::<f32>(length * stride * out_dim)? };
        {
            use super::super::engine_music3_syn::ConvTransposeDisc;
            engine.conv_transpose_for_disc(ConvTransposeDisc {
                x: &snaked,
                w: &engine.vocoder_folds_for_disc().blocks[0].conv_t_w,
                bias: &engine.weights.vocoder.blocks[0].conv_t_b,
                out: &mut up,
                rows: length,
                in_dim: hidden,
                out_dim,
                stride,
            })?;
        }
        let got_convt = s.clone_dtoh(&up)?;
        let rows = length * stride;
        let mut worst = 0.0f64;
        for t in 0..rows {
            for c in 0..out_dim {
                worst = worst.max(
                    (got_convt[t * out_dim + c] as f64 - want_convt[c * rows + t] as f64).abs(),
                );
            }
        }
        println!("DISC_V s0 convt0 max_abs={worst:.6e}");

        // Stage 3: unit1 (snake, dilated k=7 c1, snake, k=1 c2, residual)
        // on the convT output, vs block0_u1.
        let dim = out_dim;
        let unit = &engine.vocoder_folds_for_disc().blocks[0].units[0];
        let w = &engine.weights.vocoder.blocks[0].units[0];
        let y = up;
        // snake1 on the unit input.
        let mut sn1 = unsafe { s.alloc::<f32>(rows * dim)? };
        engine.snake_rows_for_disc(&y, &w.snake1_alpha, &mut sn1, rows * dim, dim)?;
        // c1: k=7 dilation 1.
        let mut conv = unsafe { s.alloc::<f32>(rows * dim)? };
        {
            use super::super::engine_music3_syn::Conv1dSite;
            engine.conv1d_dilated_for_disc(Conv1dSite {
                w: &unit.c1_w,
                x: &sn1,
                rows,
                in_dim: dim,
                out: &mut conv,
                bias: &w.c1_b,
                dilation: 1,
            })?;
        }
        // snake2.
        let mut sn2 = unsafe { s.alloc::<f32>(rows * dim)? };
        engine.snake_rows_for_disc(&conv, &w.snake2_alpha, &mut sn2, rows * dim, dim)?;
        // c2: k=1 GEMM + bias.
        let mut gated = unsafe { s.alloc::<f32>(rows * dim)? };
        {
            use super::super::engine_music3_syn::LinearDiscBytes;
            engine.linear_bytes_bias_for_disc(LinearDiscBytes {
                weight: &unit.c2_w,
                out_rows: dim,
                in_cols: dim,
                x: &sn2,
                rows,
                out: &mut gated,
                bias: &w.c2_b,
            })?;
        }
        // residual.
        let mut after = unsafe { s.alloc::<f32>(rows * dim)? };
        s.memcpy_dtod(&y, &mut after)?;
        use crate::backend::gpu::{DevicePtr, PushKernelArg};
        let a_ptr = after.device_ptr(s).0;
        let g_ptr = gated.device_ptr(s).0;
        unsafe {
            s.launch_builder(engine.add_inplace_for_disc())
                .arg(&a_ptr)
                .arg(&g_ptr)
                .arg(&((rows * dim) as i64))
                .launch(super::super::ops::flat(rows * dim))?;
        }
        let got_u1 = s.clone_dtoh(&after)?;
        let mut worst_u1 = 0.0f64;
        for t in 0..rows {
            for c in 0..dim {
                worst_u1 =
                    worst_u1.max((got_u1[t * dim + c] as f64 - want_u1[c * rows + t] as f64).abs());
            }
        }
        println!("DISC_V s0 unit1 max_abs={worst_u1:.6e}");

        // Units 2 and 3 chained on the unit1 result, then block0_out.
        {
            let folds = engine.vocoder_folds_for_disc();
            let wts = &engine.weights.vocoder.blocks[0];
            let mut y = after;
            for (u, dilation) in [(1usize, 3usize), (2, 9)] {
                let unit = &folds.blocks[0].units[u];
                let w = &wts.units[u];
                let mut sn1 = unsafe { s.alloc::<f32>(rows * dim)? };
                engine.snake_rows_for_disc(&y, &w.snake1_alpha, &mut sn1, rows * dim, dim)?;
                let mut conv = unsafe { s.alloc::<f32>(rows * dim)? };
                {
                    use super::super::engine_music3_syn::Conv1dSite;
                    engine.conv1d_dilated_for_disc(Conv1dSite {
                        w: &unit.c1_w,
                        x: &sn1,
                        rows,
                        in_dim: dim,
                        out: &mut conv,
                        bias: &w.c1_b,
                        dilation,
                    })?;
                }
                let mut sn2 = unsafe { s.alloc::<f32>(rows * dim)? };
                engine.snake_rows_for_disc(&conv, &w.snake2_alpha, &mut sn2, rows * dim, dim)?;
                let mut gated = unsafe { s.alloc::<f32>(rows * dim)? };
                {
                    use super::super::engine_music3_syn::LinearDiscBytes;
                    engine.linear_bytes_bias_for_disc(LinearDiscBytes {
                        weight: &unit.c2_w,
                        out_rows: dim,
                        in_cols: dim,
                        x: &sn2,
                        rows,
                        out: &mut gated,
                        bias: &w.c2_b,
                    })?;
                }
                let mut next = unsafe { s.alloc::<f32>(rows * dim)? };
                s.memcpy_dtod(&y, &mut next)?;
                use crate::backend::gpu::DevicePtr;
                let n_ptr = next.device_ptr(s).0;
                let g_ptr = gated.device_ptr(s).0;
                unsafe {
                    use crate::backend::gpu::PushKernelArg;
                    s.launch_builder(engine.add_inplace_for_disc())
                        .arg(&n_ptr)
                        .arg(&g_ptr)
                        .arg(&((rows * dim) as i64))
                        .launch(super::super::ops::flat(rows * dim))?;
                }
                y = next;
            }
            let got_b0 = s.clone_dtoh(&y)?;
            let want_b0 = tensor_f32(&probe, "block0_out")?;
            let w0 = &want_b0[0..rows * dim];
            let mut worst_b0 = 0.0f64;
            for t in 0..rows {
                for c in 0..dim {
                    worst_b0 =
                        worst_b0.max((got_b0[t * dim + c] as f64 - w0[c * rows + t] as f64).abs());
                }
            }
            println!("DISC_V s0 block0_units123 max_abs={worst_b0:.6e}");
        }

        // Full-chain walk: engine vocoder, every block tap vs definition.
        let latent_all = s.clone_htod(&latent)?;
        let total: usize = engine.spec.upsampling.iter().map(|r| *r as usize).product();
        let mut audio = unsafe { s.alloc::<f32>(2 * length * total)? };
        engine.vocoder_taps.borrow_mut().clear();
        engine.vocoder(&latent_all, length, &mut audio)?;
        let stage_dims = [768usize, 384, 192, 96];
        let stage_rows = [86usize * 8, 86 * 64, 86 * 256, 86 * 512];
        let want_blocks: Vec<Vec<f32>> = (0..4)
            .map(|i| tensor_f32(&probe, &format!("block{i}_out")))
            .collect::<Result<_>>()?;
        let taps = engine.vocoder_taps.borrow();
        let block_taps: Vec<&Vec<f32>> = taps
            .iter()
            .filter(|(n, _)| *n == "block")
            .map(|(_, d)| d)
            .collect();
        for (i, got) in block_taps.iter().enumerate() {
            let half = i / 4;
            let block_index = i % 4;
            let rows = stage_rows[block_index];
            let dim = stage_dims[block_index];
            let g = got.as_slice();
            let w = &want_blocks[block_index][half * dim * rows..half * dim * rows + rows * dim];
            let mut worst = 0.0f64;
            for t in 0..rows {
                for c in 0..dim {
                    worst = worst.max((g[t * dim + c] as f64 - w[c * rows + t] as f64).abs());
                }
            }
            println!("DISC_V s{half} block{block_index} max_abs={worst:.6e}");
        }
        drop(taps);
        let want_final = tensor_f32(&probe, "final")?;
        let got_audio = s.clone_dtoh(&audio)?;
        let mut worst_f = 0.0f64;
        for (a, b) in got_audio.iter().zip(&want_final) {
            worst_f = worst_f.max((a - b).abs() as f64);
        }
        println!("DISC_V final max_abs={worst_f:.6e}");
        Ok(())
    }
}
