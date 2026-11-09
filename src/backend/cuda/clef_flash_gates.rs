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
    use half::bf16;
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
    fn max_rmse_rejects_non_finite_cells() -> Result<()> {
        let clean = max_rmse(&[1.0, 2.0], &[1.0, 2.5])?;
        ensure!(clean.0 > 0.0, "finite comparison must produce a delta");
        ensure!(
            max_rmse(&[f32::NAN, 1.0], &[0.0, 1.0]).is_err(),
            "a NaN input must error, not score"
        );
        ensure!(
            max_rmse(&[1.0, 1.0], &[f32::NAN, 1.0]).is_err(),
            "a NaN reference must error, not score"
        );
        Ok(())
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
            let (a, b) = (*a as f64, *b as f64);
            ensure!(
                a.is_finite() && b.is_finite(),
                "non-finite cell value (runner FAIL)"
            );
            let d = (a - b).abs();
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
            engine.set_token(*token)?;
            engine.decode(false, None)?;
        }
        for layer in [0usize, 1, 2] {
            let (state, history) = engine.decode_gdn_state(layer)?;
            decode_cell(
                &refs,
                &bounds,
                &format!("l{layer}_recurrent_states"),
                &state,
            )?;
            // The engine's history [conv_dim, 3] is the last three
            // columns of the reference window [conv_dim, 4].
            let wide = refs
                .get(&format!("l{layer}_conv_states"))
                .with_context(|| format!("l{layer}_conv_states missing"))?;
            ensure!(wide.len() % 4 == 0, "conv window width mismatch");
            let want: Vec<f32> = (0..wide.len() / 4)
                .flat_map(|c| wide[c * 4 + 1..c * 4 + 4].iter().copied())
                .collect();
            ensure!(
                history.len() == want.len(),
                "history extent layer {layer}: got {} want {}",
                history.len(),
                want.len()
            );
            let (max, rmse) = max_rmse(&history, &want)?;
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
    fn joint_fixture() -> Result<crate::backend::cuda::joint_schema::JointFixture> {
        let path = std::env::var("FF_CLEF_JOINT_FIXTURE").context("FF_CLEF_JOINT_FIXTURE")?;
        crate::backend::cuda::joint_schema::JointFixture::parse(&std::fs::read_to_string(path)?)
    }

    fn joint_taus() -> Result<BTreeMap<String, serde_json::Value>> {
        let path = std::env::var("FF_CLEF_JOINT_TAUS").context("FF_CLEF_JOINT_TAUS")?;
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    fn joint_tau(taus: &BTreeMap<String, serde_json::Value>, cell: &str) -> Result<(f64, f64)> {
        let v = taus
            .get(cell)
            .with_context(|| format!("tau cell {cell} missing"))?;
        Ok((
            v["tau_max"].as_f64().context("tau_max")?,
            v["tau_rmse"].as_f64().context("tau_rmse")?,
        ))
    }

    fn joint_cell(
        failures: &mut Vec<String>,
        label: &str,
        got: &[f32],
        want: &[f32],
        tau: (f64, f64),
        gated: bool,
    ) -> Result<()> {
        let (max, rmse) = max_rmse(got, want)?;
        println!(
            "CLEF_JOINT {label} max={max:.6e} (tau {:.6e}) rmse={rmse:.6e} (tau {:.6e}) gated={gated}",
            tau.0, tau.1
        );
        if gated && (max > tau.0 || rmse > tau.1) {
            failures.push(label.to_owned());
        }
        Ok(())
    }

    fn joint_bf16_file(dir: &std::path::Path, file: &str, name: &str) -> Result<Vec<bf16>> {
        let bytes = std::fs::read(dir.join(format!("{file}.safetensors")))?;
        let view = SafeTensors::deserialize(&bytes)?;
        let t = view
            .tensor(name)
            .with_context(|| format!("tensor {name}"))?;
        ensure!(
            t.dtype() == safetensors::Dtype::BF16,
            "tensor {name} not BF16"
        );
        Ok(t.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    fn joint_f32_file(dir: &std::path::Path, file: &str, name: &str) -> Result<Vec<f32>> {
        let bytes = std::fs::read(dir.join(format!("{file}.safetensors")))?;
        let view = SafeTensors::deserialize(&bytes)?;
        tensor_f32(&view, name)
    }

    fn joint_bf16_from_f32_file(
        dir: &std::path::Path,
        file: &str,
        name: &str,
    ) -> Result<Vec<bf16>> {
        Ok(joint_f32_file(dir, file, name)?
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect())
    }

    fn control_ratio(
        failures: &mut Vec<String>,
        label: &str,
        delta: (f64, f64),
        tau: (f64, f64),
        rmse_only: bool,
    ) -> Result<()> {
        println!(
            "CLEF_JOINT_CONTROL {label} max={:.3e} (tau {:.3e}, {:.1}x) rmse={:.3e} (tau {:.3e}, {:.1}x)",
            delta.0,
            tau.0,
            delta.0 / tau.0,
            delta.1,
            tau.1,
            delta.1 / tau.1
        );
        if rmse_only {
            if delta.1 < 10.0 * tau.1 {
                failures.push(format!("{label} below 10x tau_rmse"));
            }
        } else if delta.0 < 10.0 * tau.0 || delta.1 < 10.0 * tau.1 {
            failures.push(format!("{label} below 10x tau"));
        }
        Ok(())
    }

    /// The multi-chunk capture discriminator: the captured hidden rows with a
    /// forced prefill chunk (at least two chunks, derived from the fixture)
    /// must match the single-chunk capture inside the hidden_norm floor.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_JOINT_FIXTURE, FF_CLEF_JOINT_TAUS"]
    fn clef_flash_joint_multi_chunk_capture() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let taus = joint_taus()?;
        let fixture = joint_fixture()?;
        let seq = fixture.ids.len();
        let forced = seq.div_ceil(2).next_multiple_of(32).min(seq - 1).max(32);
        println!("CLEF_JOINT_MULTI_CHUNK seq={seq} forced_chunk={forced}");

        let runtime = std::env::temp_dir().join(format!("clef-joint-mc-a-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 8192, None, settings(runtime.clone()))?;
        let mut engine = Engine::load(&dir, 0, 8192, None, settings(runtime))?;
        let mut plan = engine.joint_plan(&fixture)?;
        if std::env::var_os("FF_CLEF_JOINT_FORCED_CHUNK").is_some() {
            let forced = fixture
                .ids
                .len()
                .div_ceil(2)
                .next_multiple_of(32)
                .min(fixture.ids.len() - 1)
                .max(32);
            println!("CLEF_JOINT_FORCED_CHUNK {forced}");
            engine.force_prefill_chunk(forced)?;
        }
        engine.joint_capture(&mut plan, &fixture)?;
        let single: Vec<f32> = engine
            .joint_hidden_rows(&plan)?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        drop(engine);

        let runtime_b =
            std::env::temp_dir().join(format!("clef-joint-mc-b-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 8192, None, settings(runtime_b.clone()))?;
        let mut engine_b = Engine::load(&dir, 0, 8192, None, settings(runtime_b))?;
        let mut plan_b = engine_b.joint_plan(&fixture)?;
        engine_b.force_prefill_chunk(forced)?;
        engine_b.joint_capture(&mut plan_b, &fixture)?;
        let multi: Vec<f32> = engine_b
            .joint_hidden_rows(&plan_b)?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let (max, rmse) = max_rmse(&multi, &single)?;
        let tau = joint_tau(&taus, "hidden_norm")?;
        println!(
            "CLEF_JOINT_MULTI_CHUNK max={max:.6e} (tau {:.6e}) rmse={rmse:.6e} (tau {:.6e})",
            tau.0, tau.1
        );
        ensure!(
            max <= tau.0 && rmse <= tau.1,
            "multi-chunk capture diverges from the single-chunk capture"
        );
        Ok(())
    }

    /// The joint-head cells against joint-ref-taus-v3: gated end-to-end stages
    /// (projections, fields, decoder0, per-question logits), forced common-input
    /// stages on the saved reference rows (hidden_norm, routing0/1, decoder1-3),
    /// the disclosed end-to-end routing/decoder/hidden_norm measurements, and the
    /// ruled neighbour-substitution controls per family.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL, FF_CLEF_JOINT_TAUS, FF_CLEF_JOINT_REF, FF_CLEF_JOINT_FIXTURE, FF_CLEF_JOINT_FORCED"]
    fn clef_flash_joint_cells() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir = PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let forced_dir =
            PathBuf::from(std::env::var("FF_CLEF_JOINT_FORCED").context("FF_CLEF_JOINT_FORCED")?);
        let taus = joint_taus()?;
        let fixture = joint_fixture()?;
        let ref_path =
            PathBuf::from(std::env::var("FF_CLEF_JOINT_REF").context("FF_CLEF_JOINT_REF")?);
        let ref_bytes = std::fs::read(ref_path)?;
        let refs = SafeTensors::deserialize(&ref_bytes)?;
        let ref_f32 = |name: &str| tensor_f32(&refs, name);

        let runtime = std::env::temp_dir().join(format!("clef-joint-{}", std::process::id()));
        Engine::calibrate(&dir, 0, 8192, None, settings(runtime.clone()))?;
        let mut engine = Engine::load(&dir, 0, 8192, None, settings(runtime))?;
        let mut plan = engine.joint_plan(&fixture)?;
        engine.joint_capture(&mut plan, &fixture)?;
        let stages = engine.joint_forward(&mut plan, &fixture)?;
        let mut failures: Vec<String> = Vec::new();

        joint_cell(
            &mut failures,
            "proj_memory_projection",
            &stages.memory_row0,
            &ref_f32("proj_memory_projection")?,
            joint_tau(&taus, "proj_memory_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "proj_question_projection",
            &stages.question_proj_row0,
            &ref_f32("proj_question_projection")?,
            joint_tau(&taus, "proj_question_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "proj_option_question_projection",
            &stages.option_question_last_row0,
            &ref_f32("proj_option_question_projection")?,
            joint_tau(&taus, "proj_option_question_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "proj_option_context_projection",
            &stages.option_context_last_row0,
            &ref_f32("proj_option_context_projection")?,
            joint_tau(&taus, "proj_option_context_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "proj_option_lexical_projection",
            &stages.option_lexical_last_row0,
            &ref_f32("proj_option_lexical_projection")?,
            joint_tau(&taus, "proj_option_lexical_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "proj_global_projection",
            &stages.global,
            &ref_f32("proj_global_projection")?,
            joint_tau(&taus, "proj_global_projection")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "fields",
            &stages.fields_row0,
            &ref_f32("fields")?,
            joint_tau(&taus, "fields")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "decoder0",
            &stages.decoder[0],
            &ref_f32("decoder0")?,
            joint_tau(&taus, "decoder0")?,
            true,
        )?;
        joint_cell(
            &mut failures,
            "routing0",
            &stages.routing[0],
            &ref_f32("routing0")?,
            joint_tau(&taus, "routing0")?,
            false,
        )?;
        joint_cell(
            &mut failures,
            "routing1",
            &stages.routing[1],
            &ref_f32("routing1")?,
            joint_tau(&taus, "routing1")?,
            false,
        )?;
        for layer in 1..4 {
            joint_cell(
                &mut failures,
                &format!("decoder{layer}"),
                &stages.decoder[layer],
                &ref_f32(&format!("decoder{layer}"))?,
                joint_tau(&taus, &format!("decoder{layer}"))?,
                false,
            )?;
        }
        joint_cell(
            &mut failures,
            "hidden_norm",
            &stages.hidden_norm_full,
            &ref_f32("hidden_norm")?,
            joint_tau(&taus, "hidden_norm")?,
            false,
        )?;
        for (index, question) in stages.logits.iter().enumerate() {
            joint_cell(
                &mut failures,
                &format!("logits_q{index}"),
                question,
                &ref_f32(&format!("logits_q{index}"))?,
                joint_tau(&taus, &format!("logits_q{index}"))?,
                true,
            )?;
        }

        // the loaded weights and restored after each planted run.

        // Corrected q2 floor and forced logits-head cells (v4 freeze).
        let head_taus_path =
            std::env::var("FF_CLEF_JOINT_HEAD_TAUS").context("FF_CLEF_JOINT_HEAD_TAUS")?;
        let head_taus: BTreeMap<String, serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(head_taus_path)?)?;
        let arm_dir =
            PathBuf::from(std::env::var("FF_CLEF_JOINT_ARM").context("FF_CLEF_JOINT_ARM")?);
        let epilogue_bytes = std::fs::read(arm_dir.join("epilogue_inputs.safetensors"))?;
        let epilogue = SafeTensors::deserialize(&epilogue_bytes)?;
        let row = |name: &str| -> Result<Vec<bf16>> {
            let t = epilogue.tensor(name)?;
            ensure!(t.dtype() == safetensors::Dtype::BF16, "{name} not BF16");
            Ok(t.data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        };
        let logits_arm_bytes = std::fs::read(arm_dir.join("logits_bf16.safetensors"))?;
        let logits_arm = SafeTensors::deserialize(&logits_arm_bytes)?;
        let forced_logits = engine.joint_forced_logits(
            &mut plan,
            &fixture,
            crate::backend::cuda::joint_schema::ForcedLogits {
                fields_all: &row("fields_all")?,
                routed: &row("routed")?,
                lexical: &row("lexical")?,
                qvecs: &row("qvecs")?,
                global: &row("global")?,
            },
        )?;
        for (index, question) in forced_logits.iter().enumerate() {
            joint_cell(
                &mut failures,
                &format!("logits_head_forced_q{index}"),
                question,
                &tensor_f32(&logits_arm, &format!("logits_q{index}"))?,
                joint_tau(&head_taus, &format!("logits_head_forced_q{index}"))?,
                true,
            )?;
        }
        joint_cell(
            &mut failures,
            "logits_q2_varied_floor",
            &stages.logits[2],
            &tensor_f32(&logits_arm, "logits_q2")?,
            joint_tau(&head_taus, "logits_q2_varied_floor")?,
            true,
        )?;

        // logits family: residual_scorer.0 <- option_question_projection.
        {
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.scorer0_w.clone();
                w.scorer0_w = w.option_question_proj.clone();
                saved
            };
            let planted = engine.joint_forward(&mut plan, &fixture)?;
            {
                let w = engine.joint_plant_handles()?;
                w.scorer0_w = saved;
            }
            for (index, question) in stages.logits.iter().enumerate() {
                let delta = max_rmse(&planted.logits[index], question)?;
                control_ratio(
                    &mut failures,
                    &format!("logits_q{index}"),
                    delta,
                    joint_tau(&taus, &format!("logits_q{index}"))?,
                    false,
                )?;
            }
        }

        // fields family: question_projection <- global_projection.
        {
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.question_proj.clone();
                w.question_proj = w.global_proj.clone();
                saved
            };
            let planted = engine.joint_forward(&mut plan, &fixture)?;
            {
                let w = engine.joint_plant_handles()?;
                w.question_proj = saved;
            }
            control_ratio(
                &mut failures,
                "fields",
                max_rmse(&planted.fields_row0, &stages.fields_row0)?,
                joint_tau(&taus, "fields")?,
                false,
            )?;
        }

        // decoder0 family (end-to-end cell): self_attn.in_proj <- layer 1.
        {
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.decoder[0].self_in_proj.clone();
                w.decoder[0].self_in_proj = w.decoder[1].self_in_proj.clone();
                saved
            };
            let planted = engine.joint_forward(&mut plan, &fixture)?;
            {
                let w = engine.joint_plant_handles()?;
                w.decoder[0].self_in_proj = saved;
            }
            control_ratio(
                &mut failures,
                "decoder0",
                max_rmse(&planted.decoder[0], &stages.decoder[0])?,
                joint_tau(&taus, "decoder0")?,
                false,
            )?;
        }

        // projmem family: memory_projection <- question_projection.
        {
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.memory_proj.clone();
                w.memory_proj = w.question_proj.clone();
                saved
            };
            let planted = engine.joint_forward(&mut plan, &fixture)?;
            {
                let w = engine.joint_plant_handles()?;
                w.memory_proj = saved;
            }
            control_ratio(
                &mut failures,
                "projmem",
                max_rmse(&planted.memory_row0, &stages.memory_row0)?,
                joint_tau(&taus, "proj_memory_projection")?,
                false,
            )?;
        }

        // Forced common-input cells on the saved reference rows.
        let hidden_in = joint_bf16_file(&forced_dir, "hidden_input", "rows")?;
        let hidden_ref = joint_f32_file(&forced_dir, "hidden_norm_out", "rows")?;
        let clean_hidden = engine.joint_hidden_norm(&mut plan, &hidden_in)?;
        joint_cell(
            &mut failures,
            "hidden_norm_forced",
            &clean_hidden,
            &hidden_ref,
            joint_tau(&taus, "hidden_norm_forced")?,
            true,
        )?;

        let queries = joint_bf16_from_f32_file(&forced_dir, "stage_inputs", "routing_queries")?;
        let memory = joint_bf16_from_f32_file(&forced_dir, "stage_inputs", "memory")?;
        let clean_routing = engine.joint_forced_routing(&mut plan, &queries, &memory)?;
        for (index, cell) in [(0usize, "routing0_forced"), (1usize, "routing1_forced")] {
            joint_cell(
                &mut failures,
                cell,
                &clean_routing[index],
                &joint_f32_file(&forced_dir, "forced_out_cuda", cell)?,
                joint_tau(&taus, cell)?,
                true,
            )?;
        }
        let d1_in = joint_bf16_from_f32_file(&forced_dir, "stage_inputs", "decoder1_in")?;
        let mut clean_chain: Vec<Vec<f32>> = Vec::with_capacity(3);
        let mut fields_now = engine.joint_forced_decoder(&mut plan, 1, &d1_in, &memory)?;
        clean_chain.push(fields_now.clone());
        joint_cell(
            &mut failures,
            "decoder1_forced",
            &fields_now,
            &joint_f32_file(&forced_dir, "forced_out_cuda", "decoder1_forced")?,
            joint_tau(&taus, "decoder1_forced")?,
            true,
        )?;
        for layer in 2..4 {
            let rows: Vec<bf16> = fields_now.iter().map(|v| bf16::from_f32(*v)).collect();
            fields_now = engine.joint_forced_decoder(&mut plan, layer, &rows, &memory)?;
            clean_chain.push(fields_now.clone());
            joint_cell(
                &mut failures,
                &format!("decoder{layer}_forced"),
                &fields_now,
                &joint_f32_file(
                    &forced_dir,
                    "forced_out_cuda",
                    &format!("decoder{layer}_forced"),
                )?,
                joint_tau(&taus, &format!("decoder{layer}_forced"))?,
                true,
            )?;
        }

        // hidden family (forced cell): hidden_norm <- the backbone final norm.
        {
            let norm_rows = engine.joint_backbone_norm_rows()?;
            let saved = {
                let planted = engine.joint_upload_bf16(&norm_rows)?;
                let w = engine.joint_plant_handles()?;
                std::mem::replace(&mut w.hidden_norm_w, planted)
            };
            let planted = engine.joint_hidden_norm(&mut plan, &hidden_in)?;
            {
                let w = engine.joint_plant_handles()?;
                w.hidden_norm_w = saved;
            }
            control_ratio(
                &mut failures,
                "hidden",
                max_rmse(&planted, &clean_hidden)?,
                joint_tau(&taus, "hidden_norm_forced")?,
                false,
            )?;
        }

        // routing families (forced cells): the two attention in_proj tensors swap.
        for (index, name) in [(0usize, "routing0"), (1usize, "routing1")] {
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.routing[index].attn_in_proj.clone();
                w.routing[index].attn_in_proj = w.routing[1 - index].attn_in_proj.clone();
                saved
            };
            let planted = engine.joint_forced_routing(&mut plan, &queries, &memory)?;
            {
                let w = engine.joint_plant_handles()?;
                w.routing[index].attn_in_proj = saved;
            }
            control_ratio(
                &mut failures,
                name,
                max_rmse(&planted[index], &clean_routing[index])?,
                joint_tau(&taus, &format!("{name}_forced"))?,
                true,
            )?;
        }

        // decoder1-3 families (forced cells): self_attn.in_proj <- the next layer
        // (cyclic). decoder3 is the recorded failed control: printed, not gated.
        for layer in 1..4 {
            let source = (layer + 1) % 4;
            let saved = {
                let w = engine.joint_plant_handles()?;
                let saved = w.decoder[layer].self_in_proj.clone();
                w.decoder[layer].self_in_proj = w.decoder[source].self_in_proj.clone();
                saved
            };
            let input_rows: Vec<bf16> = if layer == 1 {
                d1_in.clone()
            } else {
                clean_chain[layer - 2]
                    .iter()
                    .map(|v| bf16::from_f32(*v))
                    .collect()
            };
            let planted = engine.joint_forced_decoder(&mut plan, layer, &input_rows, &memory)?;
            {
                let w = engine.joint_plant_handles()?;
                w.decoder[layer].self_in_proj = saved;
            }
            let delta = max_rmse(&planted, &clean_chain[layer - 1])?;
            let tau = joint_tau(&taus, &format!("decoder{layer}_forced"))?;
            println!(
                "CLEF_JOINT_CONTROL decoder{layer} max={:.3e} ({:.1}x) rmse={:.3e} ({:.1}x)",
                delta.0,
                delta.0 / tau.0,
                delta.1,
                delta.1 / tau.1
            );
            if layer != 3 && delta.1 < 10.0 * tau.1 {
                failures.push(format!("decoder{layer} control below 10x tau_rmse"));
            }
        }

        ensure!(failures.is_empty(), "joint gate failures: {failures:?}");
        println!("CLEF_JOINT_CELLS ok");
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
