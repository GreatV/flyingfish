//! clef-flash text gates against the frozen reference arms: the engine
//! loads with the published geometry, mixer-boundary taps and text logits
//! cells must land inside the frozen bounds, and an in-memory conv-weight
//! mutant must exceed them.

#[cfg(feature = "cuda")]
mod gates {
    use crate::backend::Settings;
    use crate::backend::cuda::engine::Engine;
    use crate::trace::Trace;
    use anyhow::{Context, Result, ensure};
    use safetensors::SafeTensors;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn tensor_f32<'a>(view: &'a SafeTensors<'a>, name: &str) -> Result<Vec<f32>> {
        let t = view
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        let bytes = t.data();
        match t.dtype() {
            safetensors::Dtype::F32 => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            // BF16 -> F32 by appending 16 zero mantissa bits.
            safetensors::Dtype::BF16 => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from_le_bytes([0, 0, c[0], c[1]])))
                .collect()),
            other => anyhow::bail!("unsupported tap dtype {other:?}"),
        }
    }

    #[test]
    fn reference_tap_loader_widens_bf16_bits_exactly() -> Result<()> {
        let data = [0x80u8, 0x3F, 0x00, 0xC0, 0x00, 0x3F];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![3], &data)?;
        let bytes = safetensors::tensor::serialize([("tap", view)], None)?;
        let parsed = SafeTensors::deserialize(&bytes)?;
        assert_eq!(tensor_f32(&parsed, "tap")?, [1.0f32, -2.0, 0.5]);
        Ok(())
    }

    fn load_frozen() -> Result<BTreeMap<String, (f64, f64)>> {
        let path = std::env::var("FF_CLEF_FROZEN").context("FF_CLEF_FROZEN")?;
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(PathBuf::from(path))?)?;
        let mut cells = BTreeMap::new();
        for (name, cell) in v["cells"].as_object().context("freeze cells")? {
            let bound = &cell["bound"];
            cells.insert(
                name.clone(),
                (
                    bound["max_abs"].as_f64().context("bound max_abs")?,
                    bound["rmse"].as_f64().context("bound rmse")?,
                ),
            );
        }
        Ok(cells)
    }

    fn load_ref_taps() -> Result<BTreeMap<String, Vec<f32>>> {
        let path = std::env::var("FF_CLEF_REF_TAPS").context("FF_CLEF_REF_TAPS")?;
        let bytes = std::fs::read(PathBuf::from(path))?;
        let view = SafeTensors::deserialize(&bytes)?;
        let mut out = BTreeMap::new();
        for name in view.names() {
            // The ids tensor is I32 and is loaded separately; skip it here.
            if view.tensor(name)?.dtype() == safetensors::Dtype::I32 {
                continue;
            }
            out.insert(name.to_owned(), tensor_f32(&view, name)?);
        }
        Ok(out)
    }

    /// The exact torch-generated fixture ids (the reference capture's
    /// inputs), exported once by the reference environment as I32.
    fn fixture_ids(length: usize) -> Result<Vec<u32>> {
        let path = std::env::var("FF_CLEF_FIXTURE_IDS").context("FF_CLEF_FIXTURE_IDS")?;
        let bytes = std::fs::read(PathBuf::from(path))?;
        let view = SafeTensors::deserialize(&bytes)?;
        let t = view.tensor(&format!("ids{length}"))?;
        ensure!(t.dtype() == safetensors::Dtype::I32, "fixture id dtype");
        ensure!(t.shape()[0] == length, "fixture id extent");
        Ok(t.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn max_rmse(got: &[f32], want: &[f32]) -> Result<(f64, f64)> {
        ensure!(
            got.len() == want.len(),
            "cell extent differs: got {} want {}",
            got.len(),
            want.len()
        );
        let mut max = 0.0f64;
        let mut sum = 0.0f64;
        for (a, b) in got.iter().zip(want) {
            let d = (*a as f64 - *b as f64).abs();
            max = max.max(d);
            sum += d * d;
        }
        Ok((max, (sum / got.len() as f64).sqrt()))
    }

    fn settings(runtime: PathBuf) -> Settings {
        Settings {
            tree_builder: crate::backend::TreeBuilder::Waves,
            spec_budget: crate::backend::SpecBudget::Chain,
            spec_graph: true,
            draft_model: None,
            runtime_dir: Some(runtime),
            prefix_cache: false,
            device_budget_gib: None,
            kv_format: crate::kv::KvFormat::Bf16,
        }
    }

    /// The engine loads clef-flash with the published geometry and a
    /// clean loader census.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_FROZEN, FF_CLEF_REF_TAPS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_loads_with_published_geometry() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let runtime = std::env::temp_dir().join(format!("clef-flash-load-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        let engine = Engine::load(&dir, 0, 4096, None, settings(runtime))?;
        let config = engine.config();
        ensure!(
            config.hidden_size == 4096 && config.head_dim == 256,
            "unexpected clef geometry"
        );
        ensure!(config.num_hidden_layers == 32, "unexpected layer count");
        println!("CLEF_LOAD ok layers=32 hidden=4096 head=256");
        Ok(())
    }

    fn layer_cell(
        trace: &mut Trace,
        refs: &BTreeMap<String, Vec<f32>>,
        frozen: &BTreeMap<String, (f64, f64)>,
        layer: usize,
        point: &str,
        length: usize,
    ) -> Result<()> {
        let name = format!("layer.{layer}.debug.{point}");
        let got = trace.values(&name)?;
        let want = refs
            .get(&format!("h{length}_l{layer}_{point}"))
            .with_context(|| format!("reference tap h{length}_l{layer}_{point} missing"))?;
        let (max, rmse) = max_rmse(&got, want)?;
        let bound = frozen
            .get(&format!("h{length}_l{layer}_{point}"))
            .with_context(|| format!("frozen cell h{length}_l{layer}_{point} missing"))?;
        println!(
            "CLEF_LAYER l{layer}_{point} max={max:.6e} (bound {:.6e}) rmse={rmse:.6e} (bound {:.6e})",
            bound.0, bound.1
        );
        ensure!(
            max <= bound.0 && rmse <= bound.1,
            "layer {layer} {point} exceeds the frozen cell"
        );
        Ok(())
    }

    /// Clean taps at layers {2,3}, then the in-memory conv-overwrite
    /// mutant at the layer-2 GDN block, beyond 10x its bound.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_FROZEN, FF_CLEF_REF_TAPS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_layer_taps_and_controls() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let frozen = load_frozen()?;
        let refs = load_ref_taps()?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-flash-taps-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;

        // Clean taps against the frozen bounds.
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime.clone()))?;
        let mut trace = Trace::all_layers();
        engine.prefill(&ids, Some(&mut trace))?;
        for (layer, point) in [
            (2usize, "mixer_out"),
            (2, "o_proj"),
            (3, "mixer_out"),
            (3, "o_proj"),
        ] {
            layer_cell(&mut trace, &refs, &frozen, layer, point, 512)?;
        }
        let clean_l2 = trace.values("layer.2.debug.mixer_out")?;
        let clean_logits = engine.decode_logits()?;
        drop(engine);

        // Control: the layer-2 in-projection is zeroed and the conv weights
        // are overwritten with the next GDN layer's BEFORE plan binding. The
        // control engine gets its own runtime dir and its own trace file.
        let runtime_control =
            std::env::temp_dir().join(format!("clef-flash-taps-control-{}", std::process::id()));
        unsafe {
            std::env::set_var("FF_CLEF_TEST_PLANT", "conv:4:2");
        }
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime_control.clone()))?;
        unsafe {
            std::env::remove_var("FF_CLEF_TEST_PLANT");
        }
        engine.trace_layer = 2;
        let mut trace = Trace::default();
        engine.prefill(&ids, Some(&mut trace))?;
        let control_logits = engine.decode_logits()?;
        let logits_same = control_logits.len() == clean_logits.len()
            && control_logits
                .iter()
                .zip(clean_logits.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let logits_diff = control_logits
            .iter()
            .zip(clean_logits.iter())
            .position(|(a, b)| a.to_bits() != b.to_bits());
        println!(
            "CLEF_CONTROL_LOGITS equal={logits_same} len={} first_diff={logits_diff:?}",
            control_logits.len()
        );
        let (conv_nonzero, conv_absmax) = engine.control_readback(2, "conv1d.weight")?;
        println!("CLEF_CONTROL_READBACK conv nonzero={conv_nonzero} absmax={conv_absmax:.6e}");
        let got = trace.values("prefill.0.debug.mixer_out")?;
        let out_path = runtime_control.join("control_mixer_out.f32");
        let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&out_path, bytes)?;
        println!(
            "CLEF_CONTROL_OUT path={} mtime={:?} len={}",
            out_path.display(),
            std::fs::metadata(&out_path)?.modified().ok(),
            got.len()
        );
        let same = got.len() == clean_l2.len()
            && got
                .iter()
                .zip(clean_l2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let first_diff = got
            .iter()
            .zip(clean_l2.iter())
            .position(|(a, b)| a.to_bits() != b.to_bits());
        println!(
            "CLEF_CONTROL_COMPARE equal={same} len={} clean_len={} first_diff={first_diff:?}",
            got.len(),
            clean_l2.len()
        );
        let want = refs.get("h512_l2_mixer_out").context("ref tap")?;
        let (max, rmse) = max_rmse(&got, want)?;
        let bound = frozen.get("h512_l2_mixer_out").context("cell")?;
        println!(
            "CLEF_CONTROL conv_overwrite max={max:.6e} (>= {:.6e}) rmse={rmse:.6e} (>= {:.6e})",
            10.0 * bound.0,
            10.0 * bound.1
        );
        ensure!(
            max >= 10.0 * bound.0 && rmse >= 10.0 * bound.1,
            "conv-overwrite control under 10x"
        );
        drop(engine);

        Ok(())
    }

    type DecodeRefs = (BTreeMap<String, Vec<f32>>, BTreeMap<String, (f64, f64)>);

    fn load_decode_refs() -> Result<DecodeRefs> {
        let path = PathBuf::from(
            std::env::var("FF_CLEF_DECODE_REF").context("FF_CLEF_DECODE_REF missing")?,
        );
        let bytes = std::fs::read(&path)?;
        let view = SafeTensors::deserialize(&bytes)?;
        let mut refs = BTreeMap::new();
        for name in view.names() {
            if view.tensor(name)?.dtype() == safetensors::Dtype::I32 {
                continue;
            }
            refs.insert(name.to_owned(), tensor_f32(&view, name)?);
        }
        let tau_path = PathBuf::from(
            std::env::var("FF_CLEF_DECODE_TAUS").context("FF_CLEF_DECODE_TAUS missing")?,
        );
        let taus: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&tau_path)
                .with_context(|| format!("decode tau file {} missing", tau_path.display()))?,
        )?;
        let mut bounds = BTreeMap::new();
        for (name, cell) in taus.as_object().context("tau cells")? {
            bounds.insert(
                name.clone(),
                (
                    cell["tau"].as_f64().context("tau max")?,
                    cell["tau_rmse"].as_f64().context("tau rmse")?,
                ),
            );
        }
        Ok((refs, bounds))
    }

    fn decode_cell(
        refs: &BTreeMap<String, Vec<f32>>,

        bounds: &BTreeMap<String, (f64, f64)>,
        cell: &str,
        got: &[f32],
    ) -> Result<()> {
        let want = refs
            .get(cell)
            .with_context(|| format!("decode reference {cell} missing"))?;
        ensure!(
            got.len() == want.len(),
            "decode extent {cell}: got {} want {}",
            got.len(),
            want.len()
        );
        let (max, rmse) = max_rmse(got, want)?;
        let bound = bounds
            .get(cell)
            .with_context(|| format!("frozen decode bound {cell} missing"))?;
        println!(
            "CLEF_DECODE {cell} max={max:.6e} (bound {:.6e}) rmse={rmse:.6e} (bound {:.6e})",
            bound.0, bound.1
        );
        ensure!(
            max <= bound.0 && rmse <= bound.1,
            "decode cell {cell} exceeds the frozen bound"
        );
        Ok(())
    }

    fn emitted_ids(refs: &BTreeMap<String, Vec<f32>>) -> Result<Vec<u32>> {
        (0..8)
            .map(|step| {
                let logits = refs
                    .get(&format!("decode_logits_s{step}"))
                    .context("decode logits cell missing")?;
                let argmax = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .context("empty logits")?;
                Ok(argmax.0 as u32)
            })
            .collect()
    }

    /// Decode cells D1+D2: teacher-forced K=8 steps after the 512-token
    /// prefill. Per-step logits and the post-step GDN state/conv history
    /// are gated against the frozen decode references.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_DECODE_REF, FF_CLEF_DECODE_TAUS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_decode_cells() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let (refs, bounds) = load_decode_refs()?;
        let emitted = emitted_ids(&refs)?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-decode-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime.clone()))?;
        engine.prefill(&ids, None)?;
        for (step, token) in emitted.iter().enumerate() {
            let cell = format!("decode_logits_s{step}");
            let logits = engine.decode_logits()?;
            decode_cell(&refs, &bounds, &cell, &logits)?;
            let (state_step, _) = engine.decode_gdn_state(0)?;
            let head = |h: usize| state_step[h * 128 * 128..(h + 1) * 128 * 128].to_vec();
            let absmax = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            println!(
                "CLEF_D2_STEP {step} absmax={:.6e} h4={:.6e} h21={:.6e} h2={:.6e}",
                absmax(&state_step),
                absmax(&head(4)),
                absmax(&head(21)),
                absmax(&head(2)),
            );
            if let Some(step_ref) = refs.get(&format!("step{step}_l0_recurrent_states")) {
                let ref_head4 = &step_ref[4 * 128 * 128..(4 + 1) * 128 * 128];
                let (max, rmse) = max_rmse(&head(4), ref_head4)?;
                println!("CLEF_D2_STEPREF {step} h4_vs_ref max={max:.6e} rmse={rmse:.6e}");
            }
            if step + 1 < 8 {
                engine.set_token(*token)?;
                engine.decode(false, None)?;
            }
        }
        for layer in [0usize, 1, 2] {
            let (state, history) = engine.decode_gdn_state(layer)?;
            if layer == 0 {
                // Localization: per-head breakdown of the recurrent state
                // against the reference [hv, dk, dv] layout.
                let want = refs.get("l0_recurrent_states").context("ref missing")?;
                let hv = 32usize;
                for h in 0..hv {
                    let got_h = &state[h * 128 * 128..(h + 1) * 128 * 128];
                    let want_h = &want[h * 128 * 128..(h + 1) * 128 * 128];
                    let (max, rmse) = max_rmse(got_h, want_h)?;
                    println!("CLEF_D2_DEBUG l0 head={h} max={max:.6e} rmse={rmse:.6e}");
                }
                let transposed_max = (0..128usize)
                    .filter_map(|d| {
                        max_rmse(
                            &state[d..].iter().step_by(128).copied().collect::<Vec<_>>(),
                            &want[d..].iter().step_by(128).copied().collect::<Vec<_>>(),
                        )
                        .ok()
                        .map(|(m, _)| m)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                println!("CLEF_D2_DEBUG l0 transposed_probe max={transposed_max:.6e}");
            }
            let want_conv = refs
                .get(&format!("l{layer}_conv_states_history"))
                .with_context(|| format!("l{layer}_conv_states_history missing"))?;
            ensure!(
                history.len() == want_conv.len(),
                "history extent layer {layer}: got {} want {}",
                history.len(),
                want_conv.len()
            );
            let (max, rmse) = max_rmse(&history, want_conv)?;
            let bound = bounds
                .get(&format!("l{layer}_conv_states_history"))
                .context("history bound missing")?;
            println!(
                "CLEF_DECODE l{layer}_conv_states_history max={max:.6e} (bound {:.6e}) rmse={rmse:.6e} (bound {:.6e})",
                bound.0, bound.1
            );
            ensure!(
                max <= bound.0 && rmse <= bound.1,
                "decode cell l{layer}_conv_states_history exceeds the frozen bound"
            );
            decode_cell(
                &refs,
                &bounds,
                &format!("l{layer}_recurrent_states"),
                &state,
            )?;
            // The engine's history [conv_dim, 3] is the last three
            // columns of the reference window [conv_dim, 4].
            let want = refs
                .get(&format!("l{layer}_conv_states_history"))
                .with_context(|| format!("l{layer}_conv_states_history missing"))?;
            ensure!(
                history.len() == want.len(),
                "history extent layer {layer}: got {} want {}",
                history.len(),
                want.len()
            );
            let (max, rmse) = max_rmse(&history, want)?;
            let bound = bounds
                .get(&format!("l{layer}_conv_states_history"))
                .context("history bound missing")?;
            println!(
                "CLEF_DECODE l{layer}_conv_states_history max={max:.6e} (bound {:.6e}) rmse={rmse:.6e} (bound {:.6e})",
                bound.0, bound.1
            );
            ensure!(
                max <= bound.0 && rmse <= bound.1,
                "decode cell l{layer}_conv_states_history exceeds the frozen bound"
            );
            decode_cell(
                &refs,
                &bounds,
                &format!("l{layer}_recurrent_states"),
                &state,
            )?;
        }
        Ok(())
    }

    /// Decode D3: eager and CUDA-graph decode over teacher-forced steps
    /// must be bit-identical, and the decode reset must preserve the
    /// identity across all four mode pairs.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_DECODE_REF, FF_CLEF_DECODE_TAUS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_decode_identity() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let (refs, _) = load_decode_refs()?;
        let emitted = emitted_ids(&refs)?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-dident-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        fn probe(
            engine: &mut Engine,
            emitted: &[u32],
            ids: &[u32],
            graph: bool,
        ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
            engine.reset()?;
            engine.prefill(ids, None)?;
            let mut tokens = Vec::new();
            let mut logits = Vec::new();
            for token in emitted {
                let step_logits = engine.decode_logits()?;
                logits.extend(step_logits);
                tokens.push(*token);
                engine.set_token(*token)?;
                engine.decode(graph, None)?;
            }
            let (state, _) = engine.decode_gdn_state(1)?;
            Ok((tokens, logits, state))
        }
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime))?;
        let mut defects = 0;
        for (first_mode, second_mode) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let first = probe(&mut engine, &emitted, &ids, first_mode)?;
            let second = probe(&mut engine, &emitted, &ids, second_mode)?;
            let same = first.0 == second.0
                && first.1.len() == second.1.len()
                && first
                    .1
                    .iter()
                    .zip(second.1.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && first.2 == second.2;
            println!(
                "CLEF_DECODE_IDENTITY first={first_mode} second={second_mode} identical={same}"
            );
            defects += usize::from(!same);
        }
        ensure!(
            defects == 0,
            "DEFECT clef_flash_decode_identity pairs={defects}"
        );
        Ok(())
    }

    /// Decode D4: the conv-overwrite mutant applied after the prefill
    /// moves the first decode-step logits far beyond the frozen bound.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_DECODE_REF, FF_CLEF_DECODE_TAUS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_decode_control() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let (refs, bounds) = load_decode_refs()?;
        let emitted = emitted_ids(&refs)?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-dcontrol-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        // The accepted control form: the mutant is applied at load time
        // (before plan binding), so the whole context carries it.
        unsafe {
            std::env::set_var("FF_CLEF_TEST_PLANT", "conv:4:2");
        }
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime.clone()))?;
        unsafe {
            std::env::remove_var("FF_CLEF_TEST_PLANT");
        }
        engine.prefill(&ids, None)?;
        engine.set_token(emitted[0])?;
        engine.decode(false, None)?;
        let step1 = engine.decode_logits()?;
        let want = refs
            .get("decode_logits_s1")
            .context("decode reference s1 missing")?;
        let (max, rmse) = max_rmse(&step1, want)?;
        let bound = bounds.get("decode_logits_s1").context("tau s1 missing")?;
        println!(
            "CLEF_DECODE_CONTROL decode_conv_overwrite max={max:.6e} (>= {:.6e}) rmse={rmse:.6e} (>= {:.6e})",
            10.0 * bound.0,
            10.0 * bound.1
        );
        ensure!(
            max >= 10.0 * bound.0 && rmse >= 10.0 * bound.1,
            "decode conv control under 10x"
        );
        Ok(())
    }

    /// Text logits cells at 8 spread positions on both fixtures.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_FROZEN, FF_CLEF_REF_TAPS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_logits_cells() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let frozen = load_frozen()?;
        let refs = load_ref_taps()?;
        let runtime =
            std::env::temp_dir().join(format!("clef-flash-logits-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        for length in [512usize, 2048] {
            let ids = fixture_ids(length)?;
            let step = length / 8;
            for i in 0..8 {
                // Row i * length / 8, the reference capture's position scheme.
                let pos = (i * step).min(length - 1);
                let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime.clone()))?;
                engine.prefill(&ids[..=pos], None)?;
                let got = engine.decode_logits()?;
                let want = refs
                    .get(&format!("logits{length}_p{i}"))
                    .with_context(|| format!("logits cell L{length}_p{i} missing"))?;
                ensure!(
                    got.len() == want.len(),
                    "logits extent: got {} want {}",
                    got.len(),
                    want.len()
                );
                let (max, rmse) = max_rmse(&got, want)?;
                let bound = frozen
                    .get(&format!("logits{length}_p{i}"))
                    .with_context(|| format!("frozen cell logits{length}_p{i} missing"))?;
                println!(
                    "CLEF_LOGITS L{length}_p{i} max={max:.6e} (bound {:.6e}) rmse={rmse:.6e} (bound {:.6e})",
                    bound.0, bound.1
                );
                ensure!(
                    max <= bound.0 && rmse <= bound.1,
                    "logits cell L{length}_p{i} exceeds the frozen cell"
                );
            }
        }
        Ok(())
    }

    /// Engine layer-0 sub-stage trace: saves every recorded layer-0 point
    /// (qkv_projected absent in the GDN path; mixer_out, o_proj, norm1/2,
    /// post_attention, ffn, gdn_s, gdn_history) to the run dir for the
    /// CPU localization table.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_N2_TRACE_OUT"]
    fn clef_flash_layer0_trace_dump() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let out_dir = PathBuf::from(std::env::var("FF_N2_TRACE_OUT").context("FF_N2_TRACE_OUT")?);
        std::fs::create_dir_all(&out_dir)?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-tr0-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime))?;
        engine.trace_layer = 0;
        let mut trace = Trace::all_layers();
        engine.prefill(&ids, Some(&mut trace))?;
        for name in trace.names() {
            if !name.starts_with("layer.") {
                continue;
            }
            let values = trace.values(name)?;
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let fname = name.replace('/', "_");
            std::fs::write(out_dir.join(format!("{fname}.f32")), bytes)?;
            println!("CLEF_TRACE0_DUMP {name} elements={}", values.len());
        }
        println!("CLEF_TRACE0_DUMP out={}", out_dir.display());
        Ok(())
    }
    /// Engine layer-0 sub-stage trace against the reference localization
    /// taps: the in-proj projection, the conv history, the gated output
    /// and the block output, each compared with max/rmse. Localization
    /// only — these tensors never enter a freeze.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_REF_TAPS, FF_CLEF_FIXTURE_IDS"]
    fn clef_flash_layer0_substage_trace() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let refs = load_ref_taps()?;
        let ids = fixture_ids(512)?;
        let runtime = std::env::temp_dir().join(format!("clef-tr0-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 4096, None, settings(runtime.clone()))?;
        let mut engine = Engine::load(&dir, 0, 4096, None, settings(runtime))?;
        engine.trace_layer = 0;
        let mut trace = Trace::all_layers();
        engine.prefill(&ids, Some(&mut trace))?;
        let names = trace.names();
        println!("CLEF_TRACE0 points={}", names.len());
        for n in &names {
            println!("CLEF_TRACE0 has {n}");
        }

        // Conv-history check: the engine's history is the exact widening
        // of the pre-conv input's last three rows; compare against the
        // reference in_proj output's rows 509..512 laid out [channel, 3].
        if let Ok(hist) = trace.values("layer.0.debug.gdn_history.e2e") {
            // The reference splits concatenate channel-wise into the
            // 8192-wide pre-conv input; its last three rows, laid out
            // [channel, 3], are the conv history.
            let q = refs.get("sub_in_proj_q").context("sub_in_proj_q missing")?;
            let k = refs.get("sub_in_proj_k").context("sub_in_proj_k missing")?;
            let v = refs.get("sub_in_proj_v").context("sub_in_proj_v missing")?;
            let mut tail = Vec::with_capacity(8192 * 3);
            for c in 0..8192usize {
                let (split, col, stride) = if c < 2048 {
                    (q, c, 2048usize)
                } else if c < 4096 {
                    (k, c - 2048, 2048usize)
                } else {
                    (v, c - 4096, 4096usize)
                };
                for j in 0..3 {
                    tail.push(split[(509 + j) * stride + col]);
                }
            }
            println!(
                "CLEF_TRACE0 gdn_history extent={} tail={}",
                hist.len(),
                tail.len()
            );
            if tail.len() == hist.len() {
                let (max, rmse) = max_rmse(&hist, &tail)?;
                println!(
                    "CLEF_TRACE0 gdn_history vs ref_inproj_tail: max={max:.6e} rmse={rmse:.6e}"
                );
            }
        }
        if let Ok(s_state) = trace.values("layer.0.debug.gdn_s.e2e") {
            let absmax = s_state.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
            let finite = s_state.iter().all(|v| v.is_finite());
            println!(
                "CLEF_TRACE0 gdn_s absmax={absmax:.6e} finite={finite} extent={}",
                s_state.len()
            );
        }
        for (point, ref_name) in [
            ("layer.0.debug.mixer_out", "sub_norm_out"),
            ("layer.0.debug.o_proj", "sub_block_out"),
        ] {
            let got = match trace.values(point) {
                Ok(v) => v,
                Err(e) => {
                    println!("CLEF_TRACE0 {point} absent: {e}");
                    continue;
                }
            };
            let want = match refs.get(ref_name) {
                Some(w) => w,
                None => {
                    println!("CLEF_TRACE0 {point}: reference {ref_name} absent");
                    continue;
                }
            };
            if got.len() != want.len() {
                println!(
                    "CLEF_TRACE0 {point} vs {ref_name}: EXTENT got {} want {}",
                    got.len(),
                    want.len()
                );
                continue;
            }
            let (max, rmse) = max_rmse(&got, want)?;
            println!(
                "CLEF_TRACE0 {point} vs {ref_name}: max={max:.6e} rmse={rmse:.6e} extent={}",
                got.len()
            );
        }
        Ok(())
    }
}

/// Applies FF_CLEF_TEST_PLANT entries to the loaded catalog before plan
/// binding: "zero:<layer>:<needle>" zeroes the first matching tensor and
/// "conv:<src>:<dst>" copies the conv weights of the source layer into
/// the destination layer's buffer.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn apply_test_plants(
    catalog: &mut crate::backend::cuda::weights::WeightCatalog,
    device: &crate::backend::cuda::Device,
) -> anyhow::Result<()> {
    use anyhow::Context;
    use half::bf16;
    let Ok(plants) = std::env::var("FF_CLEF_TEST_PLANT") else {
        return Ok(());
    };
    let s = &device.stream;
    device.finish_upload()?;
    for plant in plants.split(',') {
        let parts: Vec<&str> = plant.split(':').collect();
        match (parts[0], parts.len()) {
            ("zero", 3) => {
                let dst = catalog
                    .tensors
                    .keys()
                    .find(|k| k.contains(&format!("layers.{}.", parts[1])) && k.contains(parts[2]))
                    .cloned()
                    .with_context(|| format!("plant zero target {plant}"))?;
                let zeros = vec![bf16::from_f32(0.0); catalog.bf16(&dst)?.len()];
                {
                    let slice = catalog.bf16_mut(&dst)?;
                    s.memcpy_htod(&zeros, slice)?;
                }
                let readback = s.clone_dtoh(catalog.bf16(&dst)?)?;
                let nonzero = readback.iter().filter(|v| v.to_bits() != 0).count();
                println!(
                    "CLEF_LOAD_PLANT_ZERO {dst} len={} nonzero_after={nonzero}",
                    zeros.len()
                );
            }
            ("conv", 3) => {
                let pick = |layer: &str| -> anyhow::Result<String> {
                    catalog
                        .tensors
                        .keys()
                        .find(|k| {
                            k.contains(&format!("layers.{layer}.")) && k.contains("conv1d.weight")
                        })
                        .cloned()
                        .with_context(|| format!("plant conv layer {layer}"))
                };
                let src = pick(parts[1])?;
                let dst = pick(parts[2])?;
                let src_host = s.clone_dtoh(catalog.bf16(&src)?)?;
                {
                    let dst_ref = catalog.bf16_mut(&dst)?;
                    s.memcpy_htod(&src_host, dst_ref)?;
                }
                let readback = s.clone_dtoh(catalog.bf16(&dst)?)?;
                let absmax = readback.iter().fold(0.0f32, |m, v| m.max(v.to_f32().abs()));
                println!("CLEF_LOAD_PLANT_CONV {src} -> {dst} absmax_after={absmax:.6e}");
            }
            _ => anyhow::bail!("unrecognized FF_CLEF_TEST_PLANT entry {plant}"),
        }
    }
    Ok(())
}
