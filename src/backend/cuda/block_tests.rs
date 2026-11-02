use super::*;
use crate::backend::cuda::moe::prefill::{
    MoePrefillContribute, MoePrefillFinish, PrefillOps, prefill_scratch_bytes_mixed_with_boundary,
};
use crate::backend::cuda::moe::reference::CloverMoeFixture;
use crate::backend::cuda::moe::route::{BlockRouteBindings, BlockRoutePlan, RouteOps};
use crate::model_spec::{ExpertFormat, RoutedWeightBoundary, SwigluRounding};
use anyhow::Context;
use std::collections::BTreeMap;

#[test]
#[ignore = "GPU window; frozen Clover generation references and published checkpoint"]
fn clover_generation_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let reference = std::path::PathBuf::from(
        std::env::var_os("FF_CLOVER_REFERENCE").context("FF_CLOVER_REFERENCE missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("FROZEN.json"))?)?;
    ensure!(
        frozen["revision"] == "4b0402b65e523d0183a14c431955438714cb16ff"
            && frozen["keys"] == 813
            && frozen["candidate_observations"] == 0
            && frozen["r2_changed"] == false
            && frozen["structural_tolerance"] == 0,
        "Runner FAIL generation freeze identity"
    );
    let cases = frozen["cases"]
        .as_array()
        .context("Runner FAIL generation cases")?;
    ensure!(cases.len() == 2, "Runner FAIL generation case coverage");
    let output = std::path::PathBuf::from(
        std::env::var_os("FF_CLOVER_CAPTURE").context("FF_CLOVER_CAPTURE missing")?,
    );
    std::fs::create_dir(&output)?;
    let directory =
        std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let runtime = std::env::temp_dir().join(format!(
        "flyingfish-clover-generation-{}",
        std::process::id()
    ));
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        runtime_dir: Some(runtime.clone()),
        prefix_cache: false,
        device_budget_gib: Some(14),
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let mut model = crate::model::Model::load(&directory, 0, 97, None, settings)?;
    ensure!(
        !model.graph_decode(),
        "Clover DEFECT canvas graph capability"
    );
    let mut results = Vec::new();
    let mut defects = 0usize;
    cudarc::driver::profiler_start()?;
    for (request, index) in [0usize, 1, 0].into_iter().enumerate() {
        let case = &cases[index];
        model.reset()?;
        ensure!(
            model.position() == 0,
            "Clover DEFECT request reset position"
        );
        let prompt: Vec<u32> = case["prompt"]
            .as_array()
            .context("Runner FAIL prompt IDs")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .context("Runner FAIL prompt ID")
            })
            .collect::<Result<_>>()?;
        let maximum = case["maximum"]
            .as_u64()
            .context("Runner FAIL response maximum")? as usize;
        let mut callback = Vec::new();
        let result = crate::model::generate_plain(
            &mut model,
            &prompt,
            maximum,
            false,
            false,
            None,
            &mut |id| {
                callback.push(id);
                true
            },
        );
        match result {
            Ok(result) => {
                defects += usize::from(
                    result.ids != callback || result.ids.is_empty() || result.ids.len() > maximum,
                );
                defects += usize::from(
                    result.ids.len() < maximum
                        && !model
                            .config
                            .eos_token_id
                            .contains(result.ids.last().unwrap()),
                );
                defects += usize::from(model.decode(false, None).is_ok());
                let path = output.join(format!("request{request}"));
                let captured: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(path.join("capture.json"))?)?;
                let calls = captured["calls"]
                    .as_array()
                    .context("Runner FAIL native calls")?;
                let commits: Vec<_> = calls.iter().filter_map(|call| call.get("commit")).collect();
                let returned: Vec<u32> = commits
                    .iter()
                    .flat_map(|commit| {
                        commit["returned"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|id| id.as_u64().unwrap() as u32)
                    })
                    .collect();
                let causal: Vec<_> = calls
                    .iter()
                    .filter(|call| call["preview"] == false)
                    .collect();
                let last = causal
                    .last()
                    .context("Runner FAIL absent native causal pass")?;
                let committed_position =
                    last["prefix"].as_u64().unwrap() + last["rows"].as_u64().unwrap();
                let structural = returned == result.ids
                    && model.position() == committed_position as usize
                    && calls.last().is_some_and(|call| {
                        call["preview"] == true && call["commit"]["terminal"] == true
                    })
                    && captured["preview_commit_control"] == true
                    && captured["defects"].as_array().is_some_and(Vec::is_empty);
                defects += usize::from(!structural);
                std::fs::write(
                    path.join("result.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                    "mode":"native","case":case["tag"],"prompt":prompt,"maximum":maximum,"response":result.ids,
                    "position":model.position(),"structural":structural}))?,
                )?;
                println!(
                    "CLOVER_GENERATION_NATIVE {} request={request} case={} returned={} position={} previews={} causal_passes={} terminal_replay=false control={}",
                    if structural { "PASS" } else { "DEFECT" },
                    case["tag"],
                    result.ids.len(),
                    model.position(),
                    calls.iter().filter(|call| call["preview"] == true).count(),
                    causal.len(),
                    captured["preview_commit_control"]
                );
                results.push(result.ids);
            }
            Err(error) => {
                let text = format!("{error:#}");
                ensure!(
                    text.starts_with("Clover "),
                    "Runner FAIL generation execution: {text}"
                );
                defects += 1;
                println!("CLOVER_GENERATION_NATIVE DEFECT request={request} error={text}");
                results.push(Vec::new());
            }
        }
    }
    defects += usize::from(results[0] != results[2]);
    defects += model.validate_canvas_reference(&reference)?;
    cudarc::driver::profiler_stop()?;
    drop(model);
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    println!(
        "CLOVER_GENERATION_DRIVER {} requests=3 cases=2 defects={defects} numerical_witness=pending_oracle r2_changed=false",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "Clover generation driver defects={defects}");
    Ok(())
}

struct LayerWeights {
    values: BTreeMap<String, CudaSlice<bf16>>,
    shapes: BTreeMap<String, Vec<usize>>,
}

impl LayerWeights {
    fn load(device: &Device, fixture: &CloverMoeFixture, layer: usize) -> Result<Self> {
        let prefix = format!("layer{layer}.weight.");
        let archive = fixture.archive()?;
        let mut values = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        for name in archive
            .names()
            .iter()
            .filter(|name| name.starts_with(&prefix))
        {
            let key = name.strip_prefix(&prefix).unwrap().to_owned();
            shapes.insert(key.clone(), archive.tensor(name)?.shape().to_vec());
            values.insert(key, device.stream.clone_htod(&fixture.bf16(name)?)?);
        }
        Ok(Self { values, shapes })
    }

    fn matrix(&self, s: &Arc<CudaStream>, name: &str) -> Result<Tensor<bf16, 2>> {
        let shape = self
            .shapes
            .get(name)
            .context("Runner FAIL missing weight")?;
        ensure!(shape.len() >= 2, "Runner FAIL matrix weight rank");
        dense(
            &self.values[name],
            s,
            shape[0] as u32,
            shape[1..].iter().product::<usize>() as u32,
        )
    }

    fn vector(&self, s: &Arc<CudaStream>, name: &str) -> Result<Tensor<bf16, 1>> {
        let b = self
            .values
            .get(name)
            .context("Runner FAIL missing vector")?;
        view(b, s, 0, [b.len() as u32], [1])
    }
}

struct LayerMoe {
    ops: PrefillOps,
    router: RouteOps,
    plans: [BlockRoutePlan; 2],
    logits: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    done: CudaSlice<u32>,
    active: CudaSlice<u32>,
    generations: CudaSlice<u64>,
    gu: CudaSlice<u64>,
    dn: CudaSlice<u64>,
    live: CudaSlice<u32>,
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    contributions: CudaSlice<bf16>,
    scratch: CudaSlice<u8>,
    _route_scratch: CudaSlice<u8>,
    _valid: CudaSlice<i32>,
    _allowed: CudaSlice<u32>,
    _experts: Vec<CudaSlice<bf16>>,
}

impl LayerMoe {
    fn new(device: &Device, fixture: &CloverMoeFixture, case: &serde_json::Value) -> Result<Self> {
        let s = &device.stream;
        let layer = case["layer"].as_u64().unwrap();
        let mut gu_host = vec![0u64; 256];
        let mut dn_host = gu_host.clone();
        let mut live = vec![0u32; 256];
        let mut experts = Vec::new();
        for expert in case["available_experts"].as_array().unwrap() {
            let e = expert.as_u64().unwrap() as usize;
            let gu =
                s.clone_htod(&fixture.bf16(&format!("layer{layer}.expert{e}.gate_up_proj"))?)?;
            let dn = s.clone_htod(&fixture.bf16(&format!("layer{layer}.expert{e}.down_proj"))?)?;
            ensure!(
                gu.len() == 1024 * 2048 && dn.len() == 2048 * 512,
                "Runner FAIL expert shape"
            );
            gu_host[e] = gu.device_ptr(s).0;
            dn_host[e] = dn.device_ptr(s).0;
            live[e] = 1;
            experts.extend([gu, dn]);
        }
        let logits = s.alloc_zeros(32 * 256)?;
        let ids = s.alloc_zeros(32 * 8)?;
        let weights = s.alloc_zeros(32 * 8)?;
        let allowed = s.alloc_zeros(256)?;
        let valid = s.clone_htod(&[1i32; 32])?;
        let route_scratch = s.alloc_zeros::<u8>(32 * 256 * 4)?;
        let bind = |top_p| {
            BlockRoutePlan::prepare(
                &device.ctx,
                BlockRouteBindings {
                    logits: dense(&logits, s, 32, 256)?,
                    ids: TensorMut(dense(&ids, s, 32, 8)?),
                    weights: TensorMut(dense(&weights, s, 32, 8)?),
                    valid: view(&valid, s, 0, [32], [1])?,
                    allowed: TensorMut(view(&allowed, s, 0, [256], [1])?),
                },
                Scratch {
                    ptr: route_scratch.device_ptr(s).0,
                    bytes: route_scratch.len() as u64,
                },
                top_p,
            )
        };
        let plans = [bind(Some(0.9))?, bind(None)?];
        let bytes = prefill_scratch_bytes_mixed_with_boundary(
            32,
            32,
            8,
            256,
            2048,
            (512, 512),
            (None, RoutedWeightBoundary::BeforeContributionBf16),
        )?;
        Ok(Self {
            ops: PrefillOps::new_with_boundaries(
                &device.ctx,
                s,
                ExpertFormat::Bf16,
                SwigluRounding::Once,
                RoutedWeightBoundary::BeforeContributionBf16,
            )?,
            router: RouteOps::new(&device.ctx, s, 256, 32)?,
            plans,
            logits,
            ids,
            weights,
            done: s.alloc_zeros(256)?,
            active: s.clone_htod(&[1u32; 256])?,
            generations: s.clone_htod(&[1u64; 256])?,
            gu: s.clone_htod(&gu_host)?,
            dn: s.clone_htod(&dn_host)?,
            live: s.clone_htod(&live)?,
            gu_host,
            dn_host,
            contributions: s.alloc_zeros(256 * 2048)?,
            scratch: s.alloc_zeros(bytes as usize)?,
            _route_scratch: route_scratch,
            _valid: valid,
            _allowed: allowed,
            _experts: experts,
        })
    }

    fn run(
        &mut self,
        device: &Device,
        w: &LayerWeights,
        x: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>,
        preview: bool,
    ) -> Result<()> {
        let s = &device.stream;
        let x = dense(x, s, 32, 2048)?;
        self.router.logits_bf16(
            x,
            w.matrix(s, "mlp.gate.weight")?,
            &mut TensorMut(dense(&self.logits, s, 32, 256)?),
        )?;
        self.plans[usize::from(!preview)].run(s)?;
        let ids = s.clone_dtoh(&self.ids)?;
        ensure!(
            ids.iter()
                .all(|&i| (0..256).contains(&i) && self.gu_host[i as usize] != 0),
            "Runner FAIL native route selected an expert absent from the reference fixture"
        );
        s.memset_zeros(&mut self.done)?;
        let scratch = Scratch {
            ptr: self.scratch.device_ptr(s).0,
            bytes: self.scratch.len() as u64,
        };
        let ids = dense(&self.ids, s, 32, 8)?;
        let active = dense(&self.active, s, 32, 8)?;
        let contribution = heads(&self.contributions, s, 32, 8, 2048)?;
        let shared_gate = w.matrix(s, "mlp.shared_expert.gate_proj.weight")?;
        let shared_up = w.matrix(s, "mlp.shared_expert.up_proj.weight")?;
        let shared_down = w.matrix(s, "mlp.shared_expert.down_proj.weight")?;
        let shared_expert_gate = Some(w.vector(s, "mlp.shared_expert_gate.weight")?);
        self.ops.prefill_contribute(
            s,
            MoePrefillContribute {
                mask_from_slots: false,
                expert_inter: 512,
                wave: Some((0, 0, 256)),
                wait_copy: None,
                reader_event: None,
                cpu_arithmetic: &[true; 256],
                fp8_slot: None,
                x,
                ids,
                weights: dense(&self.weights, s, 32, 8)?,
                active,
                assignment_gen: Some(dense(&self.generations, s, 32, 8)?),
                slots_gate_up: view(&self.gu, s, 0, [256], [1])?,
                slots_down: view(&self.dn, s, 0, [256], [1])?,
                slot_valid: view(&self.live, s, 0, [256], [1])?,
                slot_generation: view(&self.generations, s, 0, [256], [1])?,
                slots_gate_up_host: &self.gu_host,
                slots_down_host: &self.dn_host,
                shared_gate,
                shared_up,
                shared_down,
                shared_expert_gate,
                contributions: contribution,
                done: TensorMut(dense(&self.done, s, 32, 8)?),
            },
            &scratch,
        )?;
        self.ops.prefill_finish(
            s,
            MoePrefillFinish {
                expert_inter: 512,
                x,
                ids,
                final_active: active,
                contributions: contribution,
                done: dense(&self.done, s, 32, 8)?,
                experts: 256,
                shared_gate,
                shared_up,
                shared_down,
                shared_expert_gate,
                y: TensorMut(dense(y, s, 32, 2048)?),
            },
            &scratch,
        )
    }
}

fn layer_block(
    device: &Device,
    blas: &Blas,
    w: &LayerWeights,
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
    table: &CudaSlice<bf16>,
) -> Result<PreNormBlock> {
    let s = &device.stream;
    let tag = case["tag"].as_str().unwrap();
    let mixer = if w.values.contains_key("linear_attn.in_proj_qkv.weight") {
        let config =
            serde_json::from_str(include_str!("../../../tests/fixtures/clover_config.json"))?;
        let spec = crate::model_spec::ModelSpec::clover(
            &config,
            &serde_json::json!({"eos_token_id":248044}),
        )?;
        let state = s.clone_htod(&transpose_state(
            &fixture.values(&format!("{tag}.cache.ssm"))?,
        ))?;
        let history = s.clone_htod(&fixture.values(&format!("{tag}.cache.conv"))?)?;
        let m = |suffix: &str| w.matrix(s, &format!("linear_attn.{suffix}"));
        let v = |suffix: &str| w.vector(s, &format!("linear_attn.{suffix}"));
        BlockMixer::Gdn(Box::new(GdnBlockMixer::prepare(
            device,
            blas,
            32,
            GdnBlockWeights {
                spec: GdnSpec::from_model(&spec)?,
                qkv: m("in_proj_qkv.weight")?,
                a: m("in_proj_a.weight")?,
                b: m("in_proj_b.weight")?,
                z: m("in_proj_z.weight")?,
                output: m("out_proj.weight")?,
                conv: m("conv1d.weight")?,
                right: m("conv_right_weight")?,
                decay: [v("A_log")?, v("dt_bias")?],
                softmax: [
                    v("bd_softmax_logit_scale")?,
                    v("bd_softmax_history_bias")?,
                    v("bd_softmax_history_c")?,
                ],
                norm: v("norm.weight")?,
            },
            state,
            history,
        )?))
    } else {
        let prefix = case["prefix"].as_u64().unwrap() as usize;
        let cache = |name| -> Result<_> {
            let input = fixture.bf16(&format!("{tag}.cache.{name}"))?;
            let mut full = vec![bf16::from_f32(-97.0); 2 * 96 * 256];
            for h in 0..2 {
                full[h * 96 * 256..h * 96 * 256 + prefix * 256]
                    .copy_from_slice(&input[h * prefix * 256..(h + 1) * prefix * 256]);
            }
            Ok(s.clone_htod(&full)?)
        };
        let m = |suffix: &str| w.matrix(s, &format!("self_attn.{suffix}.weight"));
        BlockMixer::Attention(Box::new(AttentionBlockMixer::prepare(
            device,
            blas,
            32,
            AttentionBlockWeights {
                q_gate: m("q_proj")?,
                k: m("k_proj")?,
                v: m("v_proj")?,
                output: m("o_proj")?,
                q_norm: w.vector(s, "self_attn.q_norm.weight")?,
                k_norm: w.vector(s, "self_attn.k_norm.weight")?,
                table: dense(table, s, 96, 64)?,
                heads: 16,
                kv_heads: 2,
                dim: 256,
            },
            cache("key")?,
            cache("value")?,
        )?))
    };
    PreNormBlock::prepare(
        device,
        32,
        [
            w.vector(s, "input_layernorm.weight")?,
            w.vector(s, "post_attention_layernorm.weight")?,
        ],
        mixer,
        1e-6,
        case["prefix"].as_u64().unwrap() as u32,
    )
}

fn score(
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
    name: &str,
    values: &[f32],
) -> Result<bool> {
    let tag = case["tag"].as_str().unwrap();
    let reference = fixture.values(&format!("{tag}.fp.{name}"))?;
    ensure!(
        values.len() == reference.len() && values.iter().chain(&reference).all(|v| v.is_finite()),
        "Runner FAIL layer shape/nonfinite {tag} {name}"
    );
    let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for (&value, &expected) in values.iter().zip(&reference) {
        let delta = f64::from(value) - f64::from(expected);
        maximum = maximum.max(delta.abs());
        squared += delta * delta;
        norm += f64::from(expected).powi(2);
    }
    let relative = (squared / norm.max(1e-60)).sqrt();
    let bound = &case["bounds"][name];
    let pass = maximum
        <= bound["max_abs"]
            .as_f64()
            .context("Runner FAIL frozen max bound")?
        && relative
            <= bound["relative_l2"]
                .as_f64()
                .context("Runner FAIL frozen L2 bound")?;
    println!(
        "CLOVER_LAYER {} tag={tag} metric={name} max_abs={maximum:.9e} relative_l2={relative:.9e} bound={bound}",
        if pass { "PASS" } else { "DEFECT" }
    );
    if !pass {
        let narrow = fixture.values(&format!("{tag}.bf.{name}"))?;
        ensure!(
            narrow.len() == values.len(),
            "Runner FAIL narrow reference shape"
        );
        let index = values
            .iter()
            .zip(&reference)
            .enumerate()
            .max_by(|(_, (a, b)), (_, (c, d))| {
                (f64::from(**a) - f64::from(**b))
                    .abs()
                    .total_cmp(&(f64::from(**c) - f64::from(**d)).abs())
            })
            .context("Runner FAIL empty reference")?
            .0;
        println!(
            "CLOVER_LAYER_EXTREMUM tag={tag} metric={name} index={index} candidate={} bf16_reference={} fp32_reference={}",
            values[index], narrow[index], reference[index]
        );
    }
    Ok(pass)
}

fn full_layer(layer: usize) -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("layers.safetensors")?;
    ensure!(
        fixture.table["cross_device"].is_string(),
        "Runner FAIL layer thresholds lack cross-device qualification"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let ops = Ops::new_with_swiglu_rounding(&device.ctx, SwigluRounding::Once)?;
    if layer == 3 {
        use cudarc::driver::sys::CUfunction_attribute as F;
        let kernel = crate::backend::cuda::cubin::module(&device.ctx, "rope")?
            .load_function("rotary_prefix_kv")?;
        println!(
            "BLOCK_RESOURCE kernel=rotary_prefix_kv registers={} local_bytes={} shared_bytes={} resident_blocks={}",
            kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            kernel.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
            kernel.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)?
        );
    }
    let runtime =
        std::env::temp_dir().join(format!("flyingfish-clover-layer-{}", std::process::id()));
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "clover_layer_gate".into(),
    ));
    let blas = Blas::new(&device, &ops, &runtime, closure)?;
    let weights = LayerWeights::load(&device, &fixture, layer)?;
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/clover_config.json"))?;
    let theta = config["rope_parameters"]["rope_theta"].as_f64().unwrap() as f32;
    let table = s.clone_htod(
        &crate::model_spec::RopeSpec {
            theta,
            pairs: 32,
            pair_stride: 32,
            denominator: 64,
        }
        .table(64, 96)?,
    )?;
    let cases: Vec<_> = fixture.table["layer_cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["layer"] == layer)
        .collect();
    ensure!(cases.len() == 4, "Runner FAIL two-block layer cases");
    let mut block = layer_block(&device, &blas, &weights, &fixture, cases[0], &table)?;
    let mut moe = LayerMoe::new(&device, &fixture, cases[0])?;
    let mut defects = 0;
    for case in &cases {
        let tag = case["tag"].as_str().unwrap();
        let preview = case["preview"].as_bool().unwrap();
        let prefix = case["prefix"].as_u64().unwrap() as usize;
        ensure!(
            block.prefix as usize == prefix,
            "Runner FAIL native block prefix"
        );
        let before = cache_values(&block, s, prefix)?;
        let mut x = s.clone_htod(&fixture.bf16(&format!("{tag}.x"))?)?;
        block.run(
            &device,
            &blas,
            &ops,
            &mut x,
            (32, preview),
            |input, output| moe.run(&device, &weights, input, output, preview),
        )?;
        for (name, buffer) in block
            .boundaries()
            .into_iter()
            .chain([("output", &x), ("route_logits", &moe.logits)])
        {
            let values: Vec<_> = s.clone_dtoh(buffer)?.iter().map(|v| v.to_f32()).collect();
            defects += usize::from(!score(&fixture, case, name, &values)?);
        }
        let after = cache_values(&block, s, prefix + 32)?;
        for (name, values) in &after {
            defects += usize::from(!score(&fixture, case, name, values)?);
        }
        if preview {
            let untouched =
                cache_values(&block, s, prefix)? == before && block.prefix as usize == prefix;
            defects += usize::from(!untouched);
            println!("CLOVER_LAYER_CACHE tag={tag} preview_committed_unchanged={untouched}");
        } else {
            defects += usize::from(block.prefix as usize != prefix + 32);
        }
        let ids = s.clone_dtoh(&moe.ids)?;
        let native_weights = s.clone_dtoh(&moe.weights)?;
        let expected = fixture.values(&format!("{tag}.bf.ids"))?;
        let changed = ids
            .iter()
            .zip(expected)
            .filter(|(a, b)| **a != *b as i32)
            .count();
        let unique: std::collections::BTreeSet<_> = ids.iter().copied().collect();
        println!(
            "CLOVER_LAYER_ROUTES tag={tag} ids_changed={changed} U={} zero_fillers={} reference_top8_gap={} reference_ties={}",
            unique.len(),
            native_weights.iter().filter(|&&w| w == 0.0).count(),
            case["route_gap_min"],
            case["route_ties"]
        );
    }
    let mut dropped = layer_block(&device, &blas, &weights, &fixture, cases[0], &table)?;
    dropped.prefix = 39;
    if let BlockMixer::Attention(m) = &mut dropped.mixer {
        for cache in [&mut m.k, &mut m.v] {
            for h in 0..2 {
                s.memset_zeros(&mut cache.slice_mut((h * 96 + 7) * 256..(h * 96 + 39) * 256))?;
            }
        }
    }
    let tag = cases[2]["tag"].as_str().unwrap();
    let mut x = s.clone_htod(&fixture.bf16(&format!("{tag}.x"))?)?;
    dropped.run(&device, &blas, &ops, &mut x, (32, true), |_, output| {
        Ok(s.memset_zeros(output)?)
    })?;
    println!("CLOVER_LAYER_CONTROL_BEGIN layer={layer} mutation=drop_first_commit");
    let control_state = cache_values(&dropped, s, 71)?;
    let mut control = false;
    for (name, values) in control_state {
        control |= !score(&fixture, cases[2], &name, &values)?;
    }
    println!("CLOVER_LAYER_CONTROL_END layer={layer} detected={control}");
    defects += usize::from(!control);
    println!(
        "CLOVER_LAYER {} layer={layer} cases=4 carried_blocks=2 cache_drop_control={control} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    ensure!(defects == 0, "Clover full-layer defects={defects}");
    Ok(())
}

#[test]
#[ignore = "GPU window; frozen full-layer Clover reference required"]
fn clover_gdn_full_layer_gpu() -> Result<()> {
    full_layer(0)
}

#[test]
#[ignore = "GPU window; frozen full-layer Clover reference required"]
fn clover_attention_full_layer_gpu() -> Result<()> {
    full_layer(3)
}

fn report_full_routes(
    moe: &BlockMoe,
    s: &Arc<CudaStream>,
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
    origin: &str,
) -> Result<()> {
    let tag = case["tag"].as_str().context("Runner FAIL route tag")?;
    let row = 12usize;
    let logits: Vec<_> = s
        .clone_dtoh(&moe.logits)?
        .into_iter()
        .map(|v| v.to_f32())
        .collect();
    let ids = s.clone_dtoh(&moe.ids)?;
    let weights = s.clone_dtoh(&moe.weights)?;
    let bf_logits = fixture.values(&format!("{tag}.bf.route_logits"))?;
    let fp_logits = fixture.values(&format!("{tag}.fp.route_logits"))?;
    let bf_ids = fixture.values(&format!("{tag}.bf.ids"))?;
    let fp_ids = fixture.values(&format!("{tag}.fp.ids"))?;
    let bf_weights = fixture.values(&format!("{tag}.bf.weights"))?;
    let fp_weights = fixture.values(&format!("{tag}.fp.weights"))?;
    ensure!(
        logits.len() == 32 * 256
            && bf_logits.len() == logits.len()
            && fp_logits.len() == logits.len()
            && [
                ids.len(),
                weights.len(),
                bf_ids.len(),
                fp_ids.len(),
                bf_weights.len(),
                fp_weights.len()
            ]
            .iter()
            .all(|&n| n == 32 * 8),
        "Runner FAIL route diagnostic shape"
    );
    let native = &ids[row * 8..(row + 1) * 8];
    let bf: Vec<_> = bf_ids[row * 8..(row + 1) * 8]
        .iter()
        .map(|&v| v as i32)
        .collect();
    let fp: Vec<_> = fp_ids[row * 8..(row + 1) * 8]
        .iter()
        .map(|&v| v as i32)
        .collect();
    let sorted = |ids: &[i32]| {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids
    };
    let z = &logits[row * 256..(row + 1) * 256];
    let bz = &bf_logits[row * 256..(row + 1) * 256];
    let fz = &fp_logits[row * 256..(row + 1) * 256];
    let ranking = |z: &[f32]| {
        let mut ids: Vec<_> = (0..256).collect();
        ids.sort_by(|&a, &b| z[b].total_cmp(&z[a]).then(a.cmp(&b)));
        (
            ids[..9].iter().map(|&i| (i, z[i])).collect::<Vec<_>>(),
            z[ids[7]] - z[ids[8]],
        )
    };
    let maximum = native
        .iter()
        .map(|&i| f64::from(z[i as usize]))
        .fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<_> = native
        .iter()
        .map(|&i| (f64::from(z[i as usize]) - maximum).exp())
        .collect();
    let denominator: f64 = probabilities.iter().sum();
    let reconstructed: Vec<_> = probabilities.iter().map(|p| p / denominator).collect();
    let native_weights = &weights[row * 8..(row + 1) * 8];
    let arithmetic_error = reconstructed
        .iter()
        .zip(native_weights)
        .map(|(&a, &b)| (a - f64::from(b)).abs())
        .fold(0.0f64, f64::max);
    let aligned: Vec<_> = native
        .iter()
        .enumerate()
        .map(|(slot, &id)| {
            let at = id as usize;
            let reference = bf
                .iter()
                .position(|&v| v == id)
                .map(|slot| bf_weights[row * 8 + slot]);
            (id, native_weights[slot], reference, z[at], bz[at], fz[at])
        })
        .collect();
    println!(
        "CLOVER_FULL_ROUTE_ROW {}",
        serde_json::json!({"tag":tag,"origin":origin,"row":row,
        "native_ids":native,"bf_ids":bf,"fp_ids":fp,"same_set_bf":sorted(native)==sorted(&bf),"same_set_fp":sorted(native)==sorted(&fp),
        "native_weights":native_weights,"bf_weights":&bf_weights[row*8..(row+1)*8],"fp_weights":&fp_weights[row*8..(row+1)*8],
        "native_top9_gap":ranking(z),"bf_top9_gap":ranking(bz),"fp_top9_gap":ranking(fz),
        "native_selected_softmax_f64":reconstructed,"weight_arithmetic_max_abs":arithmetic_error,
        "aligned_id_weight_bfweight_logit_bflogit_fplogit":aligned})
    );
    Ok(())
}

fn restore_full_cache(
    block: &mut PreNormBlock,
    s: &Arc<CudaStream>,
    values: &BTreeMap<String, Vec<f32>>,
) -> Result<()> {
    let read = |name: &str| values.get(name).context("Runner FAIL cache tensor missing");
    match &mut block.mixer {
        BlockMixer::Gdn(m) => {
            s.memcpy_htod(&transpose_state(read("state_ssm")?), &mut m.state)?;
            s.memcpy_htod(read("state_conv")?, &mut m.history)?;
        }
        BlockMixer::Attention(m) => {
            let heads = m.kv_heads as usize;
            let dim = m.dim as usize;
            let capacity = m.k.len() / (heads * dim);
            for (name, buffer) in [("state_key", &mut m.k), ("state_value", &mut m.v)] {
                let source = read(name)?;
                ensure!(
                    source.len().is_multiple_of(heads * dim),
                    "Runner FAIL cache geometry"
                );
                let width = source.len() / heads;
                ensure!(width <= capacity * dim, "Runner FAIL cache capacity");
                for head in 0..heads {
                    let begin = head * capacity * dim;
                    let words: Vec<_> = source[head * width..(head + 1) * width]
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect();
                    s.memcpy_htod(&words, &mut buffer.slice_mut(begin..begin + width))?;
                }
            }
        }
    }
    Ok(())
}

fn track_full(
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
    name: &str,
    actual: &[f32],
    scored: bool,
) -> Result<bool> {
    let tag = case["tag"].as_str().context("Runner FAIL tracking tag")?;
    for arm in ["bf", "fp"] {
        let reference = fixture.values(&format!("{tag}.{arm}.{name}"))?;
        ensure!(
            actual.len() == reference.len(),
            "Runner FAIL tracking shape"
        );
        let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in actual.iter().zip(&reference) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "Runner FAIL tracking nonfinite"
            );
            let delta = f64::from(a) - f64::from(b);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(b).powi(2);
        }
        println!(
            "CLOVER_FULL_TRACK tag={tag} metric={name} reference={arm} max_abs={maximum:.9e} relative_l2={:.9e}",
            (squared / norm.max(1e-60)).sqrt()
        );
    }
    if scored && fixture.table["acceptance"] == "frozen_union_r2" {
        score(fixture, case, name, actual)
    } else {
        Ok(true)
    }
}

// 2026-10-31 17:33:15Z: native logits witness exact oracle IDs; weight pairs retain shared IDs.
fn paired_weight_ids(actual: &[i32], reference: &[i32], logits: &[f32]) -> Option<Vec<(i32, i32)>> {
    if actual.is_empty()
        || actual.len() != reference.len()
        || actual
            .iter()
            .chain(reference)
            .any(|&id| id < 0 || !logits.get(id as usize).is_some_and(|v| v.is_finite()))
    {
        return None;
    }
    let mut actual = actual.to_vec();
    let mut reference = reference.to_vec();
    actual.sort_unstable();
    reference.sort_unstable();
    if actual
        .windows(2)
        .chain(reference.windows(2))
        .any(|v| v[0] == v[1])
    {
        return None;
    }
    let mut pairs: Vec<_> = actual
        .iter()
        .filter(|id| reference.contains(id))
        .map(|&id| (id, id))
        .collect();
    let mut native_only: Vec<_> = actual
        .iter()
        .filter(|id| !reference.contains(id))
        .copied()
        .collect();
    let mut reference_only: Vec<_> = reference
        .iter()
        .filter(|id| !actual.contains(id))
        .copied()
        .collect();
    let order = |a: &i32, b: &i32| {
        logits[*a as usize]
            .total_cmp(&logits[*b as usize])
            .then(a.cmp(b))
    };
    native_only.sort_by(order);
    reference_only.sort_by(order);
    for (a, b) in native_only.into_iter().zip(reference_only) {
        pairs.push((a, b));
    }
    Some(pairs)
}

#[test]
fn clover_route_weight_pairs_preserve_shared_ids() {
    let logits = [0.0, 1.0, 1.0, 0.25, 0.0, -0.25];
    assert_eq!(
        paired_weight_ids(&[3, 1, 2], &[2, 3, 1], &logits),
        Some(vec![(1, 1), (2, 2), (3, 3)])
    );
    assert_eq!(
        paired_weight_ids(&[1, 2, 4], &[1, 2, 3], &logits),
        Some(vec![(1, 1), (2, 2), (4, 3)])
    );
    assert_eq!(
        paired_weight_ids(&[1, 4, 5], &[1, 2, 3], &logits),
        Some(vec![(1, 1), (5, 3), (4, 2)])
    );
    assert_eq!(paired_weight_ids(&[1, 1, 3], &[1, 2, 3], &logits), None);
    assert_eq!(paired_weight_ids(&[-1, 2, 3], &[1, 2, 3], &logits), None);
}

// 2026-10-31: accumulated routes are diagnostics; matched-input weights use paired expert identities.
fn score_full_routes(
    moe: &BlockMoe,
    s: &Arc<CudaStream>,
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
) -> Result<bool> {
    let tag = case["tag"]
        .as_str()
        .context("Runner FAIL route score tag")?;
    let ids = s.clone_dtoh(&moe.ids)?;
    let weights = s.clone_dtoh(&moe.weights)?;
    let reference_ids = fixture.values(&format!("{tag}.bf.ids"))?;
    let reference_weights = fixture.values(&format!("{tag}.bf.weights"))?;
    let logits = fixture.values(&format!("{tag}.bf.route_logits"))?;
    let native_logits = s.clone_dtoh(&moe.logits)?;
    let allowed = s.clone_dtoh(&moe._allowed)?;
    ensure!(
        ids.len() == 32 * 8
            && weights.len() == ids.len()
            && reference_ids.len() == ids.len()
            && reference_weights.len() == ids.len()
            && logits.len() == 32 * 256
            && native_logits.len() == logits.len()
            && native_logits.iter().all(|v| v.is_finite())
            && logits.iter().all(|v| v.is_finite())
            && allowed.len() == 256
            && allowed.iter().all(|&v| v <= 1),
        "Runner FAIL matched route shape"
    );
    let margin_bound = case["bounds"]["route_logits"]["max_abs"]
        .as_f64()
        .context("Runner FAIL router bound")?;
    let bound = &case["bounds"]["weights"];
    println!(
        "CLOVER_FULL_ROUTE_DATA {}",
        serde_json::json!({"revision":fixture.table["revision"],"schema_keys":813,
            "tag":tag,"preview":case["preview"],"native_ids":ids,"native_weights":weights,
            "native_allowed":allowed,"native_logits":native_logits.iter().map(|v|v.to_f32()).collect::<Vec<_>>(),
            "reference_ids":reference_ids,"reference_weights":reference_weights,
            "reference_logits":logits,"bounds":case["bounds"],"acceptance":fixture.table["acceptance"]})
    );
    let logit_error = native_logits
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a.to_f64() - f64::from(*b)).abs())
        .fold(0.0f64, f64::max);
    let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    let (mut id_bad, mut tie_exempt, mut changed) = (0, 0, 0);
    for row in 0..32 {
        let mut ranking = logits[row * 256..(row + 1) * 256].to_vec();
        ranking.sort_by(|a, b| b.total_cmp(a));
        let gap = f64::from(ranking[7]) - f64::from(ranking[8]);
        let exempt = gap < margin_bound;
        tie_exempt += usize::from(exempt);
        let mut actual = [0.0f64; 256];
        let mut expected = [0.0f64; 256];
        let mut selected = Vec::with_capacity(8);
        let mut reference_selected = Vec::with_capacity(8);
        for slot in row * 8..(row + 1) * 8 {
            ensure!(
                (0..256).contains(&ids[slot])
                    && (0.0..256.0).contains(&reference_ids[slot])
                    && weights[slot].is_finite()
                    && reference_weights[slot].is_finite(),
                "Runner FAIL matched route values"
            );
            let differs = ids[slot] != reference_ids[slot] as i32;
            changed += usize::from(differs);
            selected.push(ids[slot]);
            reference_selected.push(reference_ids[slot] as i32);
            actual[ids[slot] as usize] = f64::from(weights[slot]);
            expected[reference_ids[slot] as usize] = f64::from(reference_weights[slot]);
        }
        let row_logits = &logits[row * 256..(row + 1) * 256];
        let Some(pairs) = paired_weight_ids(&selected, &reference_selected, row_logits) else {
            id_bad += 1;
            println!(
                "CLOVER_FULL_ROUTE_SET_DEFECT tag={tag} row={row} native={selected:?} reference={reference_selected:?} margin={gap} native_weights={:?} reference_weights={:?}",
                &weights[row * 8..(row + 1) * 8],
                &reference_weights[row * 8..(row + 1) * 8]
            );
            continue;
        };
        for (native, reference) in pairs {
            let (a, b) = (actual[native as usize], expected[reference as usize]);
            if native != reference {
                let difference = f64::from(row_logits[reference as usize])
                    - f64::from(row_logits[native as usize]);
                println!(
                    "CLOVER_FULL_ROUTE_PAIR tag={tag} row={row} native_only={native} reference_only={reference} native_weight={a:.9e} reference_weight={b:.9e} reference_logit_difference={difference:.9e} id_acceptance=oracle_witness"
                );
            }
            maximum = maximum.max((a - b).abs());
            squared += (a - b).powi(2);
            norm += b * b;
        }
    }
    let relative = (squared / norm.max(1e-60)).sqrt();
    let pass = id_bad == 0
        && logit_error <= margin_bound
        && maximum
            <= bound["max_abs"]
                .as_f64()
                .context("Runner FAIL route max bound")?
        && relative
            <= bound["relative_l2"]
                .as_f64()
                .context("Runner FAIL route L2 bound")?;
    let frozen = fixture.table["acceptance"] == "frozen_union_r2";
    let verdict = if !frozen {
        "TRACK"
    } else if pass {
        "PASS"
    } else {
        "DEFECT"
    };
    println!(
        "CLOVER_FULL_MATCHED_ROUTE_WEIGHTS {verdict} tag={tag} invalid_id_rows={id_bad} changed_slots={changed} raw_margin_rows={tie_exempt} logit_max_abs={logit_error:.9e} logit_bound={margin_bound} weights_id_aligned=true substitutions_paired_by_reference_logits=true max_abs={maximum:.9e} relative_l2={relative:.9e} bound={bound} id_acceptance=pending_oracle_witness"
    );
    Ok(!frozen || pass)
}

fn localize_full_layer(
    model: &mut BlockForward,
    device: &Device,
    fixture: &CloverMoeFixture,
    case: &serde_json::Value,
) -> Result<bool> {
    let layer = case["layer"].as_u64().context("Runner FAIL layer index")? as usize;
    let phase = case["phase"].as_str().context("Runner FAIL phase")?;
    let prefix = case["prefix"].as_u64().context("Runner FAIL prefix")? as u32;
    let preview = case["preview"].as_bool().context("Runner FAIL preview")?;
    let tag = case["tag"].as_str().context("Runner FAIL case tag")?;
    let s = &device.stream;
    let block = &mut model.layers[layer];
    let saved = cache_values(block, s, (prefix + 32) as usize)?;
    let saved_prefix = block.prefix;
    let reference_cache = saved
        .iter()
        .map(|(name, values)| {
            let data = if prefix == 0 {
                vec![0.0; values.len()]
            } else {
                fixture.values(&format!("prefill.layer{layer}.bf.{name}"))?
            };
            Ok((name.clone(), data))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    restore_full_cache(block, s, &reference_cache)?;
    block.prefix = prefix;
    let input_name = if layer == 0 {
        format!("{phase}.input")
    } else {
        format!("{phase}.layer{}.bf.output", layer - 1)
    };
    let mut input = s.clone_htod(&fixture.bf16(&input_name)?)?;
    block.run(
        device,
        &model.blas,
        &model.ops,
        &mut input,
        (32, preview),
        |x, y| {
            model.moe.run(
                device,
                &model.moe_weights[layer],
                &mut model.experts,
                layer as u32,
                (x, y),
                (32, preview),
            )
        },
    )?;
    report_full_routes(&model.moe, s, fixture, case, "reference_input")?;
    let route_pass = score_full_routes(&model.moe, s, fixture, case)?;
    let mut values = cache_values(block, s, (prefix + 32) as usize)?;
    values.insert(
        "output".into(),
        s.clone_dtoh(&input)?
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
    );
    values.insert(
        "route_logits".into(),
        s.clone_dtoh(&model.moe.logits)?
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
    );
    values.insert("weights".into(), s.clone_dtoh(&model.moe.weights)?);
    for (name, actual) in values {
        let reference = fixture.values(&format!("{tag}.bf.{name}"))?;
        ensure!(
            actual.len() == reference.len(),
            "Runner FAIL localization shape"
        );
        let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in actual.iter().zip(&reference) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "Runner FAIL localization nonfinite"
            );
            let delta = f64::from(a) - f64::from(b);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(b).powi(2);
        }
        println!(
            "CLOVER_FULL_LOCALIZE tag={tag} metric={name} input=reference_bf16 cache=reference_bf16 max_abs={maximum:.9e} relative_l2={:.9e} diagnostic_only=true",
            (squared / norm.max(1e-60)).sqrt()
        );
    }
    restore_full_cache(block, s, &saved)?;
    block.prefix = saved_prefix;
    ensure!(
        cache_values(block, s, (prefix + 32) as usize)? == saved,
        "Runner FAIL localization cache restoration"
    );
    Ok(route_pass)
}

fn boundary_distance(
    fixture: &CloverMoeFixture,
    key: &str,
    form: &str,
    values: &[f32],
    bound: &serde_json::Value,
) -> Result<bool> {
    let mut within = false;
    for precision in ["bf", "fp"] {
        let reference = fixture.values(&format!("{key}.{precision}"))?;
        ensure!(
            values.len() == reference.len()
                && values.iter().chain(&reference).all(|v| v.is_finite()),
            "Runner FAIL boundary shape/nonfinite {key}"
        );
        let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in values.iter().zip(&reference) {
            let delta = f64::from(a) - f64::from(b);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(b).powi(2);
        }
        let relative = (squared / norm.max(1e-60)).sqrt();
        let fits = maximum <= bound["max_abs"].as_f64().context("boundary max bound")?
            && relative <= bound["relative_l2"].as_f64().context("boundary L2 bound")?;
        if precision == "fp" {
            within = fits;
        }
        println!(
            "CLOVER_BOUNDARY_TRACK key={key} form={form} reference={precision} max_abs={maximum:.9e} relative_l2={relative:.9e} within_reference_envelope={fits} bound={bound}"
        );
    }
    Ok(within)
}

#[test]
#[ignore = "GPU window; matched FP32 layer2 commit operands required"]
fn clover_generation_commit_operands_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("layers.safetensors")?;
    ensure!(
        fixture.table["scope"] == "matched_fp32_commit_operands"
            && fixture.table["diagnostic_only"] == true
            && fixture.table["acceptance_rescore"] == false
            && fixture.table["r2_changed"] == false,
        "Runner FAIL commit operand reference identity"
    );
    let cases = fixture.table["layer_cases"]
        .as_array()
        .context("Runner FAIL commit operand cases")?;
    ensure!(cases.len() == 1, "Runner FAIL commit operand coverage");
    let case = &cases[0];
    ensure!(
        case["layer"] == 2
            && case["prefix"] == 33
            && case["rows"] == 32
            && case["preview"] == false,
        "Runner FAIL commit operand geometry"
    );
    ensure!(
        case["bounds"]
            == serde_json::json!({
                "state_conv":{"max_abs":0.125,"relative_l2":0.008403462879568346},
                "state_ssm":{"max_abs":0.048273324966430664,"relative_l2":0.011617168958272568}
            }),
        "Runner FAIL commit operand frozen limits changed"
    );
    let tag = case["tag"].as_str().context("Runner FAIL commit tag")?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let ops = Ops::new_with_swiglu_rounding(&device.ctx, SwigluRounding::Once)?;
    let runtime = std::env::temp_dir().join(format!(
        "flyingfish-clover-commit-operands-{}",
        std::process::id()
    ));
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "clover_commit_operands".into(),
    ));
    let blas = Blas::new(&device, &ops, &runtime, closure)?;
    let weights = LayerWeights::load(&device, &fixture, 2)?;
    let table = s.alloc_zeros::<bf16>(96 * 64)?;
    let distance = |values: &[f32], reference: &[f32]| -> Result<serde_json::Value> {
        ensure!(
            !values.is_empty()
                && values.len() == reference.len()
                && values.iter().chain(reference).all(|v| v.is_finite()),
            "Runner FAIL commit output shape/nonfinite"
        );
        let (mut maximum, mut squared, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        for (&value, &expected) in values.iter().zip(reference) {
            let delta = f64::from(value) - f64::from(expected);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(expected).powi(2);
        }
        Ok(serde_json::json!({"max_abs":maximum,"relative_l2":(squared/norm.max(1e-60)).sqrt()}))
    };
    let mut defects = 0;
    let mut controls = 0;
    for form in ["native", "zero_input_norm"] {
        let mut block = layer_block(&device, &blas, &weights, &fixture, case, &table)?;
        let x = s.clone_htod(&fixture.bf16(&format!("{tag}.x"))?)?;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: block.metadata.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: 32,
        };
        block.common.norm.rms_with(
            s,
            domain,
            heads(&x, s, 32, 1, 2048)?,
            block.weights[0],
            &mut TensorMut(heads(&block.normalized, s, 32, 1, 2048)?),
            RmsOptions {
                mode: RmsMode::ZeroCentered,
                eps: block.eps,
            },
        )?;
        if form == "zero_input_norm" {
            s.memset_zeros(&mut block.normalized)?;
        }
        let context = BlockContext {
            stream: s,
            blas: &blas,
            common: &block.common,
            ops: &ops,
            domain,
            eps: block.eps,
        };
        let BlockMixer::Gdn(mixer) = &mut block.mixer else {
            anyhow::bail!("Runner FAIL commit operand mixer");
        };
        mixer.run(&context, dense(&block.normalized, s, 32, 2048)?, false)?;
        let mut outside = false;
        for (name, values) in cache_values(&block, s, 65)? {
            let bf = fixture.values(&format!("{tag}.bf.{name}"))?;
            let fp = fixture.values(&format!("{tag}.fp.{name}"))?;
            let native_vs_fp = distance(&values, &fp)?;
            let bound = &case["bounds"][&name];
            let pass = native_vs_fp["max_abs"]
                .as_f64()
                .context("Runner FAIL commit max error")?
                <= bound["max_abs"]
                    .as_f64()
                    .context("Runner FAIL commit max limit")?
                && native_vs_fp["relative_l2"]
                    .as_f64()
                    .context("Runner FAIL commit L2 error")?
                    <= bound["relative_l2"]
                        .as_f64()
                        .context("Runner FAIL commit L2 limit")?;
            outside |= !pass;
            if form == "native" {
                defects += usize::from(!pass);
            }
            println!(
                "CLOVER_COMMIT_OPERAND_{} {}",
                if pass { "PASS" } else { "DEFECT" },
                serde_json::json!({"metric":name,"form":form,"bf_vs_fp":distance(&bf,&fp)?,
                    "native_vs_bf":distance(&values,&bf)?,"native_vs_fp":native_vs_fp,
                    "limits":bound,"r2_changed":false})
            );
        }
        if form == "zero_input_norm" {
            controls += usize::from(outside);
        }
    }
    ensure!(controls == 1, "Runner FAIL commit operand control escaped");
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    println!(
        "CLOVER_COMMIT_OPERAND_RESULT {} layer=2 rows=32 prefix=33 defects={defects} controls=1/1 bounds=unchanged",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(
        defects == 0,
        "Clover matched-input commit defects={defects}"
    );
    Ok(())
}

#[test]
#[ignore = "GPU window; matched-input Clover boundary reference required"]
fn clover_full_boundaries_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("layers.safetensors")?;
    ensure!(
        fixture.table["allow_bf16_reduced_precision_reduction"] == false,
        "Runner FAIL strict boundary reference required"
    );
    let device = Device::new(0)?;
    let s = &device.stream;
    let ops = Ops::new_with_swiglu_rounding(&device.ctx, SwigluRounding::Once)?;
    let runtime = std::env::temp_dir().join(format!(
        "flyingfish-clover-boundaries-{}",
        std::process::id()
    ));
    let closure = std::rc::Rc::new(crate::backend::cuda::closure::Closure::new(
        false,
        "clover_boundaries".into(),
    ));
    let blas = Blas::new(&device, &ops, &runtime, closure)?;
    let router = RouteOps::new(&device.ctx, s, 256, 32)?;
    let table = s.alloc_zeros::<bf16>(96 * 64)?;
    let cases = fixture.table["layer_cases"]
        .as_array()
        .context("boundary cases")?;
    ensure!(cases.len() == 2, "Runner FAIL boundary scope");
    let mut controls = 0;
    for case in cases {
        let layer = case["layer"].as_u64().context("boundary layer")? as usize;
        ensure!(matches!(layer, 20 | 36), "Runner FAIL boundary layer scope");
        let tag = case["tag"].as_str().context("boundary tag")?;
        let weights = LayerWeights::load(&device, &fixture, layer)?;
        for form in [
            "native",
            "input_norm",
            "mixer",
            "residual",
            "moe_input",
            "zero_input_norm",
        ] {
            let mut block = layer_block(&device, &blas, &weights, &fixture, case, &table)?;
            let x = s.clone_htod(&fixture.bf16(&format!("{tag}.x"))?)?;
            let domain = RowDomain {
                meta: SeqRef {
                    ptr: block.metadata.device_ptr(s).0,
                },
                row_base: 0,
                row_capacity: 32,
            };
            block.common.norm.rms_with(
                s,
                domain,
                heads(&x, s, 32, 1, 2048)?,
                block.weights[0],
                &mut TensorMut(heads(&block.normalized, s, 32, 1, 2048)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: block.eps,
                },
            )?;
            if form == "input_norm" {
                s.memcpy_htod(
                    &fixture.bf16(&format!("{tag}.input_norm.bf"))?,
                    &mut block.normalized,
                )?;
            } else if form == "zero_input_norm" {
                s.memset_zeros(&mut block.normalized)?;
            }
            let context = BlockContext {
                stream: s,
                blas: &blas,
                common: &block.common,
                ops: &ops,
                domain,
                eps: block.eps,
            };
            let BlockMixer::Gdn(mixer) = &mut block.mixer else {
                anyhow::bail!("Runner FAIL GDN boundary mixer");
            };
            mixer.run(&context, dense(&block.normalized, s, 32, 2048)?, false)?;
            if form == "mixer" {
                s.memcpy_htod(
                    &fixture.bf16(&format!("{tag}.mixer.bf"))?,
                    &mut mixer.output.output,
                )?;
            }
            block.common.norm.rms_add_with(
                s,
                domain,
                ResidualRmsInput {
                    x: &mut TensorMut(heads(&x, s, 32, 1, 2048)?),
                    residual: heads(&mixer.output.output, s, 32, 1, 2048)?,
                    weight: block.weights[1],
                },
                &mut TensorMut(heads(&block.moe_input, s, 32, 1, 2048)?),
                RmsOptions {
                    mode: RmsMode::ZeroCentered,
                    eps: block.eps,
                },
            )?;
            let mut residual = x;
            if form == "residual" {
                s.memcpy_htod(&fixture.bf16(&format!("{tag}.residual.bf"))?, &mut residual)?;
                block.common.norm.rms_with(
                    s,
                    domain,
                    heads(&residual, s, 32, 1, 2048)?,
                    block.weights[1],
                    &mut TensorMut(heads(&block.moe_input, s, 32, 1, 2048)?),
                    RmsOptions {
                        mode: RmsMode::ZeroCentered,
                        eps: block.eps,
                    },
                )?;
            } else if form == "moe_input" {
                s.memcpy_htod(
                    &fixture.bf16(&format!("{tag}.moe_input.bf"))?,
                    &mut block.moe_input,
                )?;
            }
            let logits = s.alloc_zeros::<bf16>(32 * 256)?;
            router.logits_bf16(
                dense(&block.moe_input, s, 32, 2048)?,
                weights.matrix(s, "mlp.gate.weight")?,
                &mut TensorMut(dense(&logits, s, 32, 256)?),
            )?;
            for (name, buffer) in [
                ("input_norm", &block.normalized),
                ("mixer", &mixer.output.output),
                ("residual", &residual),
                ("moe_input", &block.moe_input),
                ("route_logits", &logits),
            ] {
                let values: Vec<_> = s.clone_dtoh(buffer)?.iter().map(|v| v.to_f32()).collect();
                let within = boundary_distance(
                    &fixture,
                    &format!("{tag}.{name}"),
                    form,
                    &values,
                    &case["bounds"][name],
                )?;
                if form == "zero_input_norm" && name == "input_norm" {
                    controls += usize::from(!within);
                }
            }
            for (name, values) in [
                ("state_ssm", transpose_state(&s.clone_dtoh(&mixer.state)?)),
                ("state_conv", s.clone_dtoh(&mixer.history)?),
            ] {
                boundary_distance(
                    &fixture,
                    &format!("{tag}.{name}"),
                    form,
                    &values,
                    &case["bounds"][name],
                )?;
            }
        }
    }
    s.synchronize()?;
    drop(blas);
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    ensure!(controls == 2, "Runner FAIL boundary controls");
    println!("CLOVER_BOUNDARY_TRACK PASS layers=2 forms=6 controls=2/2 full_model_rescore=false");
    Ok(())
}

#[test]
#[ignore = "GPU window; matched-QKV Clover SDPA reference required"]
fn clover_attention_operands_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("layers.safetensors")?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let cases = fixture.table["attention_cases"]
        .as_array()
        .context("attention cases")?;
    ensure!(cases.len() == 2, "Runner FAIL attention diagnostic scope");
    let (mut defects, mut controls) = (0, 0);
    for case in cases {
        let tag = case["tag"].as_str().context("attention tag")?;
        let q = s.clone_htod(&fixture.bf16(&format!("{tag}.q"))?)?;
        let k = s.clone_htod(&fixture.bf16(&format!("{tag}.k"))?)?;
        let mut v = s.clone_htod(&fixture.bf16(&format!("{tag}.v"))?)?;
        ensure!(
            q.len() == 32 * 16 * 256 && k.len() == 2 * 32 * 256 && v.len() == k.len(),
            "Runner FAIL QKV layout"
        );
        let mut output = s.alloc_zeros::<bf16>(q.len())?;
        let mut flash = Flash::new_with_score(
            s.clone(),
            32,
            16,
            2,
            256,
            Score::from_tokens(0.0625, None)?,
            32,
        )?;
        for form in ["native", "zero_value"] {
            if form == "zero_value" {
                s.memset_zeros(&mut v)?;
            }
            flash.run_masked(Query {
                q: &q,
                q_row_stride: 16 * 256,
                k: &k,
                v: &v,
                out: &mut output,
                shape: Shape {
                    rows: 32,
                    start: 0,
                    capacity: 32,
                    q_heads: 16,
                    kv_heads: 2,
                    head_dim: 256,
                    causal: true,
                    window_left: None,
                },
                keys: 32,
                prefix: None,
            })?;
            let values: Vec<_> = s.clone_dtoh(&output)?.iter().map(|v| v.to_f32()).collect();
            let within = boundary_distance(
                &fixture,
                &format!("{tag}.output"),
                form,
                &values,
                &case["bounds"],
            )?;
            if form == "native" {
                defects += usize::from(!within);
            } else {
                controls += usize::from(!within);
            }
        }
    }
    ensure!(controls == 2, "Runner FAIL attention controls");
    println!(
        "CLOVER_ATTENTION_OPERANDS {} cases=2 controls=2/2 defects={defects} full_model_rescore=false",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    Ok(())
}

#[test]
#[ignore = "GPU window; actual native Clover operands and published checkpoint"]
fn clover_native_operands_gpu() -> Result<()> {
    use safetensors::{Dtype, tensor::TensorView};
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("FF_CLOVER_CAPTURE").context("FF_CLOVER_CAPTURE missing")?,
    );
    std::fs::create_dir(&output)?;
    let fixture = CloverMoeFixture::open_named("full.safetensors")?;
    ensure!(
        fixture.table["acceptance"] == "frozen_union_r2"
            && fixture.table["allow_bf16_reduced_precision_reduction"] == false,
        "Runner FAIL native operand reference identity"
    );
    let directory =
        std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let runtime =
        std::env::temp_dir().join(format!("flyingfish-clover-operands-{}", std::process::id()));
    let device = Device::new(0)?;
    let s = &device.stream;
    let mut model = BlockForward::load(&device, &directory, &runtime, 4096, Some(14 << 30))?;
    for layer in 0..=20 {
        ensure!(
            model.layers[layer].prefix == 0
                && cache_values(&model.layers[layer], s, 0)?
                    .values()
                    .flatten()
                    .all(|&v| v == 0.0),
            "Runner FAIL nonzero prefill cache"
        );
    }
    let tokens: Vec<u32> = fixture
        .values("prefill.tokens")?
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let mut tensors = BTreeMap::new();
    let ids = s.clone_htod(&tokens)?;
    let mut initial = s.alloc_zeros::<bf16>(32 * 2048)?;
    unsafe {
        s.launch_builder(&model.ops.embed)
            .arg(model.weights.bf16("model.embed_tokens.weight")?)
            .arg(&ids)
            .arg(&mut initial)
            .arg(&32i32)
            .arg(&2048i32)
            .launch(flat(32 * 2048))?;
    }
    tensors.insert(
        "layer0.input".into(),
        (
            Dtype::BF16,
            vec![1, 32, 2048],
            s.clone_dtoh(&initial)?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    );
    model.forward(&device, &tokens, false, |index, block, moe, hidden| {
        if index < 20 {
            let values = s.clone_dtoh(hidden)?;
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "Runner FAIL nonfinite input"
            );
            tensors.insert(
                format!("layer{}.input", index + 1),
                (
                    Dtype::BF16,
                    vec![1, 32, 2048],
                    values
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            );
        }
        if index <= 20 {
            ensure!(block.prefix == 32, "Runner FAIL captured prefix");
            let mixer_output = match &block.mixer {
                BlockMixer::Gdn(mixer) => &mixer.output.output,
                BlockMixer::Attention(mixer) => &mixer.output.output,
            };
            for (name, buffer, width) in [
                ("output", hidden, 2048),
                ("input_norm", &block.normalized, 2048),
                ("mixer", mixer_output, 2048),
                ("moe_input", &block.moe_input, 2048),
                ("moe", &block.moe_output, 2048),
                ("route_logits", &moe.logits, 256),
            ] {
                let values = s.clone_dtoh(buffer)?;
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "Runner FAIL nonfinite {name}"
                );
                tensors.insert(
                    format!("layer{index}.{name}"),
                    (
                        Dtype::BF16,
                        vec![1, 32, width],
                        values
                            .iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                            .collect(),
                    ),
                );
            }
            let case = fixture.table["full_cases"]
                .as_array()
                .context("full cases")?
                .iter()
                .find(|case| case["tag"] == format!("prefill.layer{index}"))
                .context("captured case missing")?;
            let hidden: Vec<_> = s.clone_dtoh(hidden)?.iter().map(|v| v.to_f32()).collect();
            track_full(&fixture, case, "output", &hidden, false)?;
            for (name, state) in cache_values(block, s, 32)? {
                ensure!(
                    state.iter().all(|v| v.is_finite()),
                    "Runner FAIL nonfinite state"
                );
                let archive = fixture.archive()?;
                let reference = archive.tensor(&format!("prefill.layer{index}.bf.{name}"))?;
                ensure!(
                    reference.dtype() != Dtype::BF16
                        || state.iter().all(|&v| bf16::from_f32(v).to_f32() == v),
                    "Runner FAIL captured BF16 state changed precision"
                );
                let bytes = match reference.dtype() {
                    Dtype::F32 => state.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    Dtype::BF16 => state
                        .iter()
                        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                        .collect(),
                    _ => anyhow::bail!("Runner FAIL captured state dtype"),
                };
                tensors.insert(
                    format!("layer{index}.{name}"),
                    (reference.dtype(), reference.shape().to_vec(), bytes),
                );
                track_full(&fixture, case, &name, &state, false)?;
            }
        }
        Ok(())
    })?;
    ensure!(tensors.len() == 189, "Runner FAIL native operand coverage");
    let views = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            Ok((
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, None, &output.join("native.safetensors"))?;
    std::fs::write(
        output.join("capture.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "revision":fixture.table["revision"],"schema_keys":813,"scope":"actual_native_operands",
            "layers":(0..=20).collect::<Vec<_>>(),"rows":32,"prefix":0,"phase":"prefill","tensors":189,
            "r2_changed":false,"acceptance_rescore":false
        }))?,
    )?;
    drop(model);
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    println!(
        "CLOVER_NATIVE_CAPTURE PASS layers=21 tensors=189 prefix=0 diagnostic_only=true r2_changed=false"
    );
    Ok(())
}

#[test]
#[ignore = "GPU window; published weights and frozen full-model reference required"]
fn clover_full_model_tiers_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("full.safetensors")?;
    ensure!(
        fixture.table["consumed_keys"] == 813,
        "Runner FAIL incomplete model reference"
    );
    ensure!(
        fixture.table["allow_bf16_reduced_precision_reduction"] == false
            && matches!(
                fixture.table["acceptance"].as_str(),
                Some("reference_only" | "frozen_union_r2")
            ),
        "Runner FAIL full reference accumulation/acceptance contract"
    );
    let cases = fixture.table["full_cases"]
        .as_array()
        .context("Runner FAIL full cases missing")?;
    let heads = fixture.table["head_cases"]
        .as_array()
        .context("Runner FAIL head cases missing")?;
    ensure!(
        cases.len() == 120 && heads.len() == 3,
        "Runner FAIL full phase coverage"
    );
    let directory =
        std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let runtime =
        std::env::temp_dir().join(format!("flyingfish-clover-full-{}", std::process::id()));
    let device = Device::new(0)?;
    let s = &device.stream;
    let mut model = BlockForward::load(&device, &directory, &runtime, 4096, Some(14 << 30))?;
    model.experts.verify_tiers(s)?;
    let mut defects = 0usize;
    let mut cache_control = false;
    for (phase, prefix, preview) in [
        ("prefill", 0usize, false),
        ("preview", 32, true),
        ("commit", 32, false),
    ] {
        let tokens: Vec<u32> = fixture
            .values(&format!("{phase}.tokens"))?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let before = model
            .layers
            .iter()
            .map(|block| cache_values(block, s, prefix))
            .collect::<Result<Vec<_>>>()?;
        let mut totals = [0u64; 5];
        let mut localize = Vec::new();
        cudarc::driver::profiler_start()?;
        model.forward(&device, &tokens, preview, |index, block, moe, output| {
            let tag = format!("{phase}.layer{index}");
            let case = cases.iter().find(|case| case["tag"] == tag).context("Runner FAIL missing layer phase")?;
            if phase == "commit" && matches!(index, 35 | 36) {
                report_full_routes(moe, s, &fixture, case, "accumulated")?;
            }
            for (name, buffer) in [("output", output), ("route_logits", &moe.logits)] {
                let values: Vec<_> = s.clone_dtoh(buffer)?.into_iter().map(|v| v.to_f32()).collect();
                defects += usize::from(!track_full(&fixture, case, name, &values, true)?);
            }
            let route_weights = s.clone_dtoh(&moe.weights)?;
            ensure!(route_weights.iter().all(|v| v.is_finite()), "Runner FAIL nonfinite accumulated weights");
            track_full(&fixture, case, "weights", &route_weights, false)?;
            let ids = s.clone_dtoh(&moe.ids)?;
            let expected = fixture.values(&format!("{tag}.bf.ids"))?;
            let changed = ids.iter().zip(expected).filter(|(a, b)| **a != *b as i32).count();
            let complete = s.clone_dtoh(&moe.done)?.iter().all(|&v| v == 1);
            defects += usize::from(!complete);
            println!("CLOVER_FULL_ROUTES tag={tag} ids_changed={changed} reference_top8_gap={} reference_ties={} complete={complete} U={} resident={} host_h2d_bytes={} disk_h2d_bytes={} cohorts={}", case["route_gap_min"], case["route_ties"], moe.counts[0], moe.counts[1], moe.counts[2], moe.counts[3], moe.counts[4]);
            for (name, values) in cache_values(block, s, prefix + 32)? {
                defects += usize::from(!track_full(&fixture, case, &name, &values, true)?);
            }
            let expected_prefix = if preview { prefix } else { prefix + 32 };
            let cache_ok = block.prefix as usize == expected_prefix
                && (!preview || cache_values(block, s, prefix)? == before[index]);
            defects += usize::from(!cache_ok);
            println!("CLOVER_FULL_CACHE tag={tag} prefix={} preview={preview} exact_committed_boundary={cache_ok}", block.prefix);
            for (total, count) in totals.iter_mut().zip(moe.counts) {
                *total += count;
            }
            localize.push(case);
            Ok(())
        })?;
        s.synchronize()?;
        cudarc::driver::profiler_stop()?;
        let case = heads
            .iter()
            .find(|case| case["phase"] == phase)
            .context("Runner FAIL missing head phase")?;
        let logits: Vec<_> = s
            .clone_dtoh(&model.logits)?
            .into_iter()
            .map(|v| v.to_f32())
            .collect();
        defects += usize::from(!track_full(&fixture, case, "output", &logits, true)?);
        println!(
            "CLOVER_FULL_COUNTS phase={phase} layers=40 U_sum={} resident_sum={} host_h2d_bytes={} disk_h2d_bytes={} cohorts={} route_d2h_bytes={}",
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            totals[4],
            40 * 32 * 8 * 4
        );
        defects += usize::from(totals[1..4].contains(&0));
        if phase == "prefill" {
            let BlockMixer::Gdn(mixer) = &mut model.layers[0].mixer else {
                anyhow::bail!("Runner FAIL first mixer is not GDN");
            };
            let saved = s.clone_dtoh(&mixer.state)?;
            s.memset_zeros(&mut mixer.state)?;
            let case = cases
                .iter()
                .find(|case| case["tag"] == "prefill.layer0")
                .unwrap();
            let changed = cache_values(&model.layers[0], s, 32)?;
            println!("CLOVER_FULL_CONTROL_BEGIN mutation=drop_committed_ssm");
            cache_control = !score(&fixture, case, "state_ssm", &changed["state_ssm"])?;
            println!("CLOVER_FULL_CONTROL_END detected={cache_control}");
            let BlockMixer::Gdn(mixer) = &mut model.layers[0].mixer else {
                unreachable!()
            };
            s.memcpy_htod(&saved, &mut mixer.state)?;
            ensure!(
                s.clone_dtoh(&mixer.state)? == saved,
                "Runner FAIL control restoration"
            );
        }
        for case in localize {
            defects += usize::from(!localize_full_layer(&mut model, &device, &fixture, case)?);
        }
    }
    s.synchronize()?;
    drop(model);
    if runtime.exists() {
        std::fs::remove_dir_all(runtime)?;
    }
    defects += usize::from(!cache_control);
    println!(
        "CLOVER_FULL_NATIVE_RESULT {}",
        serde_json::json!({"layers":40,"phases":3,"native_defects":defects,
            "cache_control":cache_control,"acceptance":fixture.table["acceptance"],
            "route_ids":"pending_oracle_witness"})
    );
    println!(
        "CLOVER_FULL {} layers=40 phases=3 forced_storage_tiers=3 cache_control={cache_control} defects={defects}",
        if defects != 0 {
            "DEFECT"
        } else if fixture.table["acceptance"] == "reference_only" {
            "TRACK"
        } else {
            "PENDING_ROUTE_ORACLE"
        }
    );
    ensure!(defects == 0, "Clover full-model defects={defects}");
    Ok(())
}

#[test]
#[ignore = "GPU setup window; real Clover checkpoint headers required"]
fn clover_model_residency_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let path = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = crate::model_spec::ModelSpec::read(&path)?;
    let checkpoint = crate::checkpoint::MappedCheckpoint::open(&path)?;
    let manifest = checkpoint.manifest(&spec, crate::checkpoint::LoadPolicy::TextOnly)?;
    let device = Device::new(0)?;
    for capacity in [4096, spec.max_position] {
        let plan = PreNormBlock::model_residency(&device, &spec, &manifest, capacity, None)?;
        let (free, _) = device.ctx.mem_get_info()?;
        let host = crate::backend::cuda::calibrate::host_memory(std::path::Path::new("/"))?;
        let slot = plan.experts.slot_bytes();
        let dense_device = plan.dense.resident_bytes as u64 + 2 * plan.dense.slot_bytes as u64;
        let device_bytes =
            plan.device_fixed_bytes + dense_device + u64::from(plan.experts.slots_total()) * slot;
        let host_bytes = plan.host_fixed_bytes
            + plan.dense.host_bytes as u64
            + u64::from(plan.experts.host_blocks + plan.experts.spare_blocks) * slot;
        let unique = plan.global_dense_bytes
            + plan.dense.resident_bytes as u64
            + plan.dense.host_bytes as u64
            + plan.dense.disk_bytes as u64
            + u64::from(plan.experts.slots_total() + plan.experts.host_blocks) * slot
            + plan.experts.disk_bytes;
        ensure!(
            device_bytes <= free as u64
                && host_bytes <= host
                && unique == manifest.resident_bytes as u64,
            "Runner FAIL model residency capacity/conservation"
        );
        ensure!(
            plan.experts.span_rows == 32 && plan.experts.wave_rows == 32,
            "block residency reduced full-block routing rows"
        );
        println!(
            "CLOVER_RESIDENCY PASS capacity={capacity} device_bytes={device_bytes} host_bytes={host_bytes} unique_bytes={unique} full_route_rows=32"
        );
    }
    Ok(())
}
