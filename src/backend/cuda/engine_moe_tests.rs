use super::*;
use crate::backend::cuda::moe::reference::{MoeDims, MoeFixture, fixture};
use crate::backend::cuda::weights::ExpertStorage;
use crate::experts::plan::Capacity;

fn hybrid_spec() -> Result<crate::model_spec::ModelSpec> {
    use crate::model_spec::{LayerPlan, Mixer};
    let mut spec = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?,
        &serde_json::json!({"eos_token_id": [248044, 248046]}),
    )?;
    spec.hidden = 128;
    spec.intermediate = 64;
    spec.vocab = 32;
    spec.max_position = 32;
    spec.eos = vec![31];
    spec.vision = None;
    spec.moe = Some(crate::model_spec::MoeConfig {
        experts: 16,
        experts_per_tok: 4,
        expert_intermediate: 64,
        shared_intermediate: 64,
    });
    spec.layers = vec![
        LayerPlan {
            mixer: Mixer::Gdn,
            state_index: 0,
        },
        LayerPlan {
            mixer: Mixer::FullAttention,
            state_index: 0,
        },
    ];
    Ok(spec)
}

fn hybrid_manifest(
    spec: &crate::model_spec::ModelSpec,
) -> Result<crate::checkpoint::TensorManifest> {
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::model_spec::Mixer;
    ensure!(
        spec.architecture == crate::model_spec::Architecture::MiniCpmV47,
        "hybrid fixture requires MiniCpmV47 architecture, received {:?}",
        spec.architecture
    );
    ensure!(
        spec.layers
            .iter()
            .map(|layer| layer.mixer)
            .eq([Mixer::Gdn, Mixer::FullAttention]),
        "hybrid fixture requires one GDN and one full-attention layer"
    );
    let g = crate::op_contract::GdnSpec::from_model(spec)?;
    let a = crate::op_contract::FullAttentionSpec::from_model(spec)?;
    let h = spec.hidden as usize;
    let m = spec.moe.context("hybrid fixture MoE missing")?;
    let i = m.shared_intermediate as usize;
    let mut manifest = TensorManifest {
        entries: Default::default(),
        resident_bytes: 0,
    };
    for (layer, plan) in spec.layers.iter().enumerate() {
        let mut shapes = vec![
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("mlp.gate.weight", vec![m.experts as usize, h]),
            ("mlp.shared_expert.gate_proj.weight", vec![i, h]),
            ("mlp.shared_expert.up_proj.weight", vec![i, h]),
            ("mlp.shared_expert.down_proj.weight", vec![h, i]),
            ("mlp.shared_expert_gate.weight", vec![1, h]),
        ];
        shapes.extend(match plan.mixer {
            Mixer::Gdn => vec![
                (
                    "linear_attn.in_proj_qkv.weight",
                    vec![g.conv_dim()? as usize, h],
                ),
                (
                    "linear_attn.in_proj_z.weight",
                    vec![(g.hv * g.dv) as usize, h],
                ),
                ("linear_attn.in_proj_a.weight", vec![g.hv as usize, h]),
                ("linear_attn.in_proj_b.weight", vec![g.hv as usize, h]),
                (
                    "linear_attn.out_proj.weight",
                    vec![h, (g.hv * g.dv) as usize],
                ),
                (
                    "linear_attn.conv1d.weight",
                    vec![g.conv_dim()? as usize, 1, g.conv_width as usize],
                ),
                ("linear_attn.A_log", vec![g.hv as usize]),
                ("linear_attn.dt_bias", vec![g.hv as usize]),
                ("linear_attn.norm.weight", vec![g.dv as usize]),
            ],
            Mixer::FullAttention => vec![
                (
                    "self_attn.q_proj.weight",
                    vec![(2 * a.hq * a.d) as usize, h],
                ),
                ("self_attn.k_proj.weight", vec![(a.hkv * a.d) as usize, h]),
                ("self_attn.v_proj.weight", vec![(a.hkv * a.d) as usize, h]),
                ("self_attn.o_proj.weight", vec![h, (a.hq * a.d) as usize]),
                ("self_attn.q_norm.weight", vec![a.d as usize]),
                ("self_attn.k_norm.weight", vec![a.d as usize]),
            ],
        });
        for (role, shape) in shapes {
            let bytes = shape.iter().product::<usize>() * size_of::<bf16>();
            manifest.resident_bytes += bytes;
            manifest.entries.insert(
                format!("model.language_model.layers.{layer}.{role}"),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape,
                    shard: "synthetic.safetensors".into(),
                    bytes,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope: TensorScope::Text,
                    layer: Some(layer as u32),
                    role: role.into(),
                    scale_name: None,
                },
            );
        }
    }
    Ok(manifest)
}

impl Engine {
    pub(crate) fn load_moe_test(fixture: &MoeFixture, runtime: &Path) -> Result<Self> {
        use crate::backend::cuda::{
            linear::LayerLinearPlan,
            weights::{StoredTensor, WeightCatalog},
        };
        let d = fixture.dims;
        ensure!(
            d.hidden == 128 && d.inter == 64 && d.experts == 16 && d.top_k == 4,
            "MoE hook expects H128/I64/E16/K4, received {d:?}"
        );
        let model_spec = hybrid_spec()?;
        let config = model_spec.config();
        let manifest = hybrid_manifest(&model_spec)?;
        let geometry = ExpertGeometry::from_model(&model_spec)?;
        let device = Device::new(0)?;
        let stream = &device.stream;
        let count = (geometry.layers * geometry.experts) as usize;
        let bytes = geometry.slot_bytes()?;
        let capacity = Capacity {
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: bytes * count as u64,
            host_expert_bytes: bytes * count as u64,
            slots_per_layer: vec![16; geometry.layers as usize],
            host_blocks: count as u32,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
        };
        let storage = ExpertStorage::new(stream, &device.upload, geometry, &capacity)?;
        stream.synchronize()?;
        let mut gu = Vec::with_capacity(count);
        let mut dn = Vec::with_capacity(count);
        let gu_len = 2 * d.inter * d.hidden;
        let dn_len = d.hidden * d.inter;
        for layer in 0..geometry.layers {
            for expert in 0..d.experts {
                let (gate_up, down) = storage.slot(layer, expert as u32)?;
                // SAFETY: setup synchronization completed the owned stream-ordered allocation.
                unsafe {
                    result::memcpy_htod_sync(
                        gate_up,
                        &fixture.gate_up[expert * gu_len..(expert + 1) * gu_len],
                    )?;
                    result::memcpy_htod_sync(
                        down,
                        &fixture.down[expert * dn_len..(expert + 1) * dn_len],
                    )?;
                }
                let mut claim = storage
                    .host()
                    .context("test host arena missing")?
                    .claim(layer * geometry.experts + expert as u32)?;
                claim.slice()?[..gu_len]
                    .copy_from_slice(&fixture.gate_up[expert * gu_len..(expert + 1) * gu_len]);
                claim.slice()?[gu_len..]
                    .copy_from_slice(&fixture.down[expert * dn_len..(expert + 1) * dn_len]);
                gu.push(gate_up);
                dn.push(down);
            }
        }
        let mut tensors = std::collections::BTreeMap::new();
        for (name, entry) in &manifest.entries {
            let values = match entry.role.as_str() {
                "mlp.gate.weight" => fixture.router_w.clone(),
                "mlp.shared_expert.gate_proj.weight" => fixture.shared_gate.clone(),
                "mlp.shared_expert.up_proj.weight" => fixture.shared_up.clone(),
                "mlp.shared_expert.down_proj.weight" => fixture.shared_down.clone(),
                "mlp.shared_expert_gate.weight" => fixture.shared_expert_gate.clone(),
                role if role.ends_with("norm.weight") => vec![bf16::ONE; entry.bytes / 2],
                "linear_attn.A_log" | "linear_attn.dt_bias" => vec![bf16::ZERO; entry.bytes / 2],
                _ => (0..entry.bytes / 2)
                    .map(|index| {
                        let width = *entry.shape.last().unwrap();
                        bf16::from_f32(
                            (((index / width * 37 + index % width * 11 + 7) % 53) as f32 - 26.0)
                                / 4096.0,
                        )
                    })
                    .collect(),
            };
            ensure!(
                values.len() * 2 == entry.bytes,
                "hybrid fixture tensor extent: {name}"
            );
            tensors.insert(
                name.clone(),
                StoredTensor::Bf16(stream.clone_htod(&values)?),
            );
        }
        let bytes = manifest.resident_bytes;
        let catalog = WeightCatalog {
            tensors,
            manifest,
            bytes,
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: 0.0,
        };
        let plan = crate::linear_plan::LayerDequantPlan::from_manifest(
            &catalog.manifest,
            0,
            model_spec.linear_format,
        )?;
        let mut linear = LayerLinearPlan::bind(&plan, &catalog, None, stream)?;
        let ffn = BoundFfn::bind(&model_spec, &mut linear)?;
        let slots = Slots {
            storage,
            gate_up: stream.clone_htod(&gu)?,
            down: stream.clone_htod(&dn)?,
            valid: stream.clone_htod(&vec![1u32; count])?,
            generation: stream.clone_htod(&vec![1u64; count])?,
            published: vec![1; count],
        };
        let weights = Weights {
            embed: stream.alloc_zeros(32 * d.hidden)?,
            norm: stream.clone_htod(&vec![bf16::ONE; d.hidden])?,
            head: stream.alloc_zeros(32 * d.hidden)?,
            bytes: 0,
            layers: Vec::new(),
        };
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::super::closure::Closure::new(
            true,
            "MoE layer gate".into(),
        ));
        let blas = Blas::new(&device, &ops, runtime, closure.clone())?;
        let state = super::super::super::state::StateStore::new(&model_spec, 32, stream)?;
        let mut work = Work::new(stream, &model_spec, 8)?;
        stream.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = Some(Flash::new(stream.clone(), 8, model_spec.q_heads as usize)?);
        let moe = Some(Decode::new(&device, geometry, 8, 2)?);
        let mut engine = Self {
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            qwen: None,
            moe,
            moe_slots: Some(slots),
            moe_test_ffn: Some(ffn),
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            qwen_device_prefix: false,
            forward_rows: (1..=8).collect(),
            graphs: Vec::new(),
            closure,
            poisoned: false,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            tree_builder: TreeBuilder::Serial,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime.to_owned(),
            spec: None,
            graph: None,
            weights,
            state,
            capture_checkpoint: None,
            work,
            ops,
            blas,
            flash,
            decode: None,
            device,
            config,
            model_spec,
            capacity: 32,
            chunk: 8,
            workspace_rows: 8,
            trace_layer: 0,
            position: 0,
            prefix_identity: None,
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        let scratch = engine.device.stream.alloc_zeros(qwen::Runtime::workspace(
            &engine.model_spec,
            8,
            32,
            crate::gdn_plan::GdnPolicy::for_device(engine.device.info.arch, engine.device.info.sms),
        )?)?;
        engine.qwen = Some(qwen::Runtime::new(
            &engine, runtime, catalog, None, scratch, None, 8,
        )?);
        for rows in 1..=8 {
            engine.prepare_qwen(rows)?;
            engine.prepare_prefill_rows(rows)?;
        }
        Ok(engine)
    }

    pub(crate) fn run_layer_from_reference(
        &mut self,
        layer: usize,
        hidden_in: &[bf16],
        pos_axes: [&[i32]; 3],
        trace: &mut Trace,
    ) -> Result<()> {
        let hidden = self.model_spec.hidden as usize;
        ensure!(
            layer < self.model_spec.layers.len(),
            "layer injection layer={layer} outside layers={}",
            self.model_spec.layers.len()
        );
        ensure!(
            hidden_in.len().is_multiple_of(hidden),
            "layer injection hidden length={} width={hidden}",
            hidden_in.len()
        );
        let rows = hidden_in.len() / hidden;
        ensure!(
            rows > 0 && rows <= self.workspace_rows && pos_axes.iter().all(|a| a.len() == rows),
            "layer injection rows={rows} workspace={} axes={:?}",
            self.workspace_rows,
            pos_axes.map(<[i32]>::len)
        );
        self.inject_moe_test_positions(pos_axes)?;
        let stream = &self.device.stream;
        stream.memcpy_htod(hidden_in, &mut self.work.x.slice_mut(..hidden_in.len()))?;
        if self.qwen.is_none() {
            unsafe {
                stream
                    .launch_builder(&self.ops.norm)
                    .arg(&self.work.x)
                    .arg(&self.weights.layers[layer].input_norm)
                    .arg(&mut self.work.n)
                    .arg(&(rows as i32))
                    .arg(&(hidden as i32))
                    .arg(&self.config.rms_norm_eps)
                    .launch(grid(rows, 256))?;
            }
        }
        self.run_layer(
            layer,
            crate::backend::Step {
                rows,
                input: crate::backend::Input::Ids,
                mask: crate::backend::Mask::Causal,
                head: Head::None,
            },
            &mut Some(Target {
                trace,
                prefix: "layer",
                skip: 0,
            }),
        )
    }

    pub(crate) fn record_test_stats(&self, trace: &mut Trace) -> Result<()> {
        let stats = self
            .moe
            .as_ref()
            .context("MoE statistics executor missing")?
            .stats;
        for (name, count) in [
            "route_calls",
            "gpu_resident_hits",
            "cpu_worker_misses",
            "cpu_pool_calls",
            "combine_calls",
        ]
        .into_iter()
        .zip(stats)
        {
            trace.add(format!("stats.{name}"), vec![1], vec![count as f32])?;
        }
        Ok(())
    }
}

fn step(rows: usize) -> crate::backend::Step<'static> {
    crate::backend::Step {
        rows,
        input: crate::backend::Input::Ids,
        mask: crate::backend::Mask::Causal,
        head: Head::None,
    }
}

fn direct(engine: &Engine, input: &[bf16], rows: usize) -> Result<Trace> {
    let (h, k, i) = (128, 4, 64);
    let stream = &engine.device.stream;
    let ctx = &engine.device.ctx;
    let BoundFfn::Moe {
        router,
        shared,
        shared_gate,
    } = engine.moe_test_ffn.context("test FFN absent")?
    else {
        anyhow::bail!("dense test FFN");
    };
    let x = stream.clone_htod(input)?;
    let ids = stream.alloc_zeros::<i32>(rows * k)?;
    let weights = stream.alloc_zeros::<bf16>(rows * k)?;
    let active = stream.clone_htod(&vec![1u32; rows * k])?;
    let generation = stream.clone_htod(&vec![1u64; rows * k])?;
    let contributions = stream.alloc_zeros::<bf16>(rows * k * h)?;
    let shared_out = stream.alloc_zeros::<bf16>(rows * h)?;
    let done = stream.alloc_zeros::<u32>(rows * k)?;
    let y = stream.alloc_zeros::<bf16>(rows * h)?;
    let scratch = stream.alloc_zeros::<u8>(decode_scratch_bytes(rows, k, i) as usize)?;
    let slots = engine
        .moe_slots
        .as_ref()
        .context("test slots absent")?
        .bindings(0, stream)?;
    let xv = qwen::matrix(&x, stream, 0, rows, h)?;
    let idv = qwen::view(&ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
    let weightv = qwen::matrix(&weights, stream, 0, rows, k)?;
    RouteOps::new(ctx, stream)?.route(
        stream,
        xv,
        router,
        &mut TensorMut(idv),
        &mut TensorMut(weightv),
    )?;
    DecodeOps::new(ctx)?
        .decode(
            stream,
            &MoeDecode {
                x: xv,
                ids: idv,
                active: qwen::view(&active, stream, 0, [rows as u32, k as u32], [k as u64, 1])?,
                weights: weightv,
                slots_gate_up: slots.gate_up,
                slots_down: slots.down,
                slot_valid: slots.valid,
                slot_generation: slots.generation,
                assignment_gen: qwen::view(
                    &generation,
                    stream,
                    0,
                    [rows as u32, k as u32],
                    [k as u64, 1],
                )?,
                shared_gate: shared[0],
                shared_up: shared[1],
                shared_down: shared[2],
                shared_expert_gate: shared_gate,
                contributions: qwen::view(
                    &contributions,
                    stream,
                    0,
                    [rows as u32, k as u32, h as u32],
                    [(k * h) as u64, h as u64, 1],
                )?,
                shared_out: &mut TensorMut(qwen::matrix(&shared_out, stream, 0, rows, h)?),
                done: &mut TensorMut(qwen::view(
                    &done,
                    stream,
                    0,
                    [rows as u32, k as u32],
                    [k as u64, 1],
                )?),
            },
            &mut TensorMut(qwen::matrix(&y, stream, 0, rows, h)?),
            &qwen::scratch(&scratch, stream),
        )?
        .synchronize()?;
    let mut trace = Trace::default();
    trace.add(
        "layer.moe.ids".into(),
        vec![rows, k],
        stream
            .clone_dtoh(&ids)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    )?;
    trace.add(
        "layer.moe.done".into(),
        vec![rows, k],
        stream
            .clone_dtoh(&done)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    )?;
    for (name, data, shape) in [
        ("layer.moe.weights", weights, vec![rows, k]),
        ("layer.moe.contributions", contributions, vec![rows, k, h]),
        ("layer.moe.shared", shared_out, vec![rows, h]),
        ("layer.ffn", y, vec![rows, h]),
    ] {
        trace.add(
            name.into(),
            shape,
            stream
                .clone_dtoh(&data)?
                .into_iter()
                .map(bf16::to_f32)
                .collect(),
        )?;
    }
    Ok(trace)
}

fn compare(a: &Trace, b: &Trace) -> Result<usize> {
    let mut mismatch = 0;
    for name in [
        "layer.moe.ids",
        "layer.moe.weights",
        "layer.moe.contributions",
        "layer.moe.shared",
        "layer.moe.done",
        "layer.ffn",
    ] {
        let (a, b) = (a.values(name)?, b.values(name)?);
        ensure!(
            a.len() == b.len(),
            "reference/candidate trace extent differs for {name}"
        );
        mismatch += a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    Ok(mismatch)
}

fn cpu_control(
    engine: &mut Engine,
    input: &[bf16],
    rows: usize,
    reference: &Trace,
) -> Result<usize> {
    use crate::cpu::expert_ffn::{Contribution, Isa, Scratch, contribution};
    let ffn = engine.moe_test_ffn.context("CPU control FFN missing")?;
    let slots = engine
        .moe_slots
        .as_ref()
        .context("CPU control slots missing")?;
    let host = slots
        .storage
        .host()
        .context("CPU control host missing")?
        .clone();
    let shape = Shape::new(128, 64)?;
    let mut scratch = Scratch::new(&shape);
    let local = ExpertWeights {
        gate_up: 0..16384,
        gate_up_stride: 128,
        down: 16384..24576,
        down_stride: 64,
    };
    let experts = (0..16)
        .map(|expert| {
            let range = host.block_range(expert)?;
            Ok(ExpertWeights {
                gate_up: range.start..range.start + 16384,
                gate_up_stride: 128,
                down: range.start + 16384..range.end,
                down_stride: 64,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ids = reference.values("layer.moe.ids")?;
    let weights = reference.values("layer.moe.weights")?;
    let mut contributions = reference
        .values("layer.moe.contributions")?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let mut active = vec![1u32; rows * 4];
    let mut assignments = Vec::new();
    let mut miss_rows = Vec::new();
    for row in (0..rows).step_by(2) {
        let choice = row * 4 + (row * 3 + 1) % 4;
        let expert = ids[choice] as u32;
        let lease = host.lease(&[expert])?;
        contribution(
            Isa::detect(),
            Contribution {
                shape: &shape,
                arena: lease.read(expert)?,
                weights: &local,
                hidden: &input[row * 128..(row + 1) * 128],
                routing: bf16::from_f32(weights[choice]),
            },
            &mut contributions[choice * 128..(choice + 1) * 128],
            &mut scratch,
        )?;
        active[choice] = 0;
        assignments.push(Assignment {
            row: miss_rows.len() as u32,
            choice: choice as u32,
            expert,
            weight: bf16::from_f32(weights[choice]),
        });
        miss_rows.push(row as u32);
    }
    let decode = engine.moe.as_mut().context("CPU control decode missing")?;
    let stream = &engine.device.stream;
    let x = qwen::matrix(&engine.work.n, stream, 0, rows, 128)?;
    decode.route(&engine.device, ffn, x)?;
    let duplicate_route = decode.route(&engine.device, ffn, x).is_err();
    let alternate = stream.clone_htod(input)?;
    let mut probe_out = TensorMut(qwen::matrix(&engine.work.out, stream, 0, rows, 128)?);
    let dispatch = || Dispatch {
        active: &active,
        generation: &[1u64; 32][..rows * 4],
        host: Some(host.clone()),
        experts: &experts,
        assignments: &assignments,
        miss_rows: &miss_rows,
    };
    let different_input = decode
        .finish(
            &engine.device,
            ffn,
            slots.bindings(0, stream)?,
            dispatch(),
            (
                qwen::matrix(&alternate, stream, 0, rows, 128)?,
                &mut probe_out,
            ),
        )
        .is_err();
    decode
        .finish(
            &engine.device,
            ffn,
            slots.bindings(0, stream)?,
            Dispatch {
                active: &active,
                generation: &vec![1; rows * 4],
                host: Some(host.clone()),
                experts: &experts,
                assignments: &assignments,
                miss_rows: &miss_rows,
            },
            (
                x,
                &mut TensorMut(qwen::matrix(&engine.work.out, stream, 0, rows, 128)?),
            ),
        )?
        .synchronize()?;
    let duplicate_finish = decode
        .finish(
            &engine.device,
            ffn,
            slots.bindings(0, stream)?,
            dispatch(),
            (x, &mut probe_out),
        )
        .is_err();
    ensure!(
        duplicate_route && different_input && duplicate_finish,
        "DEFECT MoE phase controls duplicate_route={duplicate_route} different_input={different_input} duplicate_finish={duplicate_finish}"
    );
    let mut actual = Trace::default();
    decode.trace(stream, &mut actual, "layer")?;
    let observed = actual.values("layer.moe.contributions")?;
    let mismatches = observed
        .iter()
        .zip(&contributions)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    let done_mismatches = actual
        .values("layer.moe.done")?
        .iter()
        .filter(|&&d| d != 1.0)
        .count();
    let expected_contributions = stream.clone_htod(&contributions)?;
    let expected_out = stream.alloc_zeros::<bf16>(rows * 128)?;
    let expected_scale = stream.alloc_zeros::<bf16>(rows)?;
    let BoundFfn::Moe { shared_gate, .. } = ffn else {
        anyhow::bail!("dense CPU control FFN");
    };
    let scale = qwen::view(&expected_scale, stream, 0, [rows as u32, 1], [1, 1])?;
    let combine = CombineOps::new(&engine.device.ctx)?;
    combine.shared_scale(stream, x, shared_gate, &mut TensorMut(scale))?;
    combine
        .combine(
            stream,
            &MoeCombine {
                ids: qwen::view(&decode.ids, stream, 0, [rows as u32, 4], [4, 1])?,
                final_active: qwen::view(
                    &decode.final_active,
                    stream,
                    0,
                    [rows as u32, 4],
                    [4, 1],
                )?,
                done: qwen::view(&decode.done, stream, 0, [rows as u32, 4], [4, 1])?,
                contributions: qwen::view(
                    &expected_contributions,
                    stream,
                    0,
                    [rows as u32, 4, 128],
                    [512, 128, 1],
                )?,
                shared_out: qwen::matrix(&decode.shared, stream, 0, rows, 128)?,
                x,
                shared_expert_gate: shared_gate,
                scale: TensorMut(scale),
                y: TensorMut(qwen::matrix(&expected_out, stream, 0, rows, 128)?),
            },
        )?
        .synchronize()?;
    let expected = stream.clone_dtoh(&expected_out)?;
    let observed = stream.clone_dtoh(&engine.work.out.slice(..rows * 128))?;
    let output_mismatches = expected
        .iter()
        .zip(&observed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let defects = mismatches + done_mismatches + output_mismatches;
    println!(
        "{} eager_moe_cpu_control rows={rows} cpu_choices={} compacted_rows={} contribution_mismatches={mismatches} done_mismatches={done_mismatches} output_mismatches={output_mismatches}",
        if defects == 0 { "PASS" } else { "DEFECT" },
        assignments.len(),
        miss_rows.len()
    );
    Ok(usize::from(defects != 0))
}

fn gate(row_cases: &[usize], variants: &[u32], cpu: bool) -> Result<()> {
    let runtime =
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("MoE gate runtime directory missing")?;
    let runtime = Path::new(&runtime);
    std::fs::create_dir_all(runtime)?;
    let mut defects = 0;
    for &variant in variants {
        let fixture = fixture(MoeDims::new(8, 16, 4, 128, 64), variant);
        let mut engine = Engine::load_moe_test(&fixture, runtime)?;
        for &rows in row_cases {
            let input = &fixture.x[..rows * 128];
            let reference = direct(&engine, input, rows)?;
            ensure!(
                compare(&reference, &direct(&engine, input, rows)?)? == 0,
                "Runner FAIL same-form reference repeat differs rows={rows} variant={variant}"
            );
            let mut trace = Trace::default();
            engine
                .device
                .stream
                .memcpy_htod(input, &mut engine.work.n.slice_mut(..input.len()))?;
            {
                let _range = super::super::super::profile::Range::new(true, c"engine_moe_ffn");
                engine.run_ffn(
                    0,
                    step(rows),
                    &mut Some(Target {
                        trace: &mut trace,
                        prefix: "layer",
                        skip: 0,
                    }),
                )?;
            }
            let mismatches = compare(&reference, &trace)?;
            defects += usize::from(mismatches != 0);
            println!(
                "{} eager_moe_all_resident rows={rows} variant={variant} bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
            if cpu {
                defects += cpu_control(&mut engine, input, rows, &reference)?;
            }
        }
        let mut stats = Trace::default();
        engine.record_test_stats(&mut stats)?;
        let calls = row_cases.len() * if cpu { 2 } else { 1 };
        let cpu_choices = if cpu {
            row_cases.iter().map(|r| r.div_ceil(2)).sum::<usize>()
        } else {
            0
        };
        let gpu_choices =
            row_cases.iter().sum::<usize>() * 4 * if cpu { 2 } else { 1 } - cpu_choices;
        for (name, expected) in [
            ("route_calls", calls),
            ("combine_calls", calls),
            ("gpu_resident_hits", gpu_choices),
            ("cpu_worker_misses", cpu_choices),
            ("cpu_pool_calls", if cpu { row_cases.len() } else { 0 }),
        ] {
            ensure!(
                stats.values(&format!("stats.{name}"))? == [expected as f32],
                "DEFECT stats.{name} expected={expected}"
            );
        }
        let axes = [0, 1];
        let _range = super::super::super::profile::Range::new(true, c"engine_moe_hybrid");
        for layer in 0..2 {
            let mut trace = Trace::default();
            engine.trace_layer = layer;
            engine.run_layer_from_reference(layer, &fixture.x[..256], [&axes; 3], &mut trace)?;
            ensure!(
                trace.values("layer.ffn")?.len() == 256,
                "hybrid layer injection did not execute FFN layer={layer}"
            );
            ensure!(
                trace
                    .values("layer.debug.mixer_out")?
                    .iter()
                    .all(|value| value.is_finite()),
                "Runner FAIL hybrid layer mixer output nonfinite layer={layer}"
            );
            println!(
                "PASS eager_moe_hybrid_injection layer={layer} mixer={:?} rows=2 q_heads=16 kv_heads=2 head_dim=256 rotary_dim=64 hybrid_bound=true",
                engine.model_spec.layers[layer].mixer
            );
        }
    }
    ensure!(defects == 0, "eager MoE candidate defects={defects}");
    println!(
        "PASS eager_moe_gate all_resident=true cpu_control={cpu} variants={} row_cases={} graphs=false",
        variants.len(),
        row_cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "authorized GPU window"]
fn eager_moe_smoke_gpu() -> Result<()> {
    gate(&[2], &[3], false)
}

#[test]
#[ignore = "authorized GPU window"]
fn eager_moe_decode_gpu() -> Result<()> {
    gate(&[1, 2, 5, 8, 2], &[3, 7], true)
}

#[test]
fn dispatch_covers_original_choices_once_after_row_compaction() -> Result<()> {
    let mut active = [1u32; 20];
    active[1] = 0;
    active[9] = 0;
    let weights = (0..20)
        .map(|i| bf16::from_f32((i + 1) as f32 / 32.0))
        .collect::<Vec<_>>();
    let generations = [7u64; 20];
    let host = Some(HostBlocks::new(1, 32)?);
    let experts = [ExpertWeights {
        gate_up: 0..16,
        gate_up_stride: 4,
        down: 16..32,
        down_stride: 4,
    }];
    let valid = [
        Assignment {
            row: 0,
            choice: 1,
            expert: 0,
            weight: weights[1],
        },
        Assignment {
            row: 1,
            choice: 9,
            expert: 0,
            weight: weights[9],
        },
    ];
    let check = |active: &[u32], assignments: &[Assignment], rows: &[u32]| {
        validate_dispatch(
            5,
            4,
            &weights,
            &Dispatch {
                active,
                generation: &generations,
                host: host.clone(),
                experts: &experts,
                assignments,
                miss_rows: rows,
            },
        )
    };
    check(&active, &valid, &[0, 2])?;
    assert!(check(&active, &valid[..1], &[0, 2]).is_err());
    assert!(check(&active, &[valid[0], valid[0], valid[1]], &[0, 2]).is_err());
    let mut bad = valid;
    bad[1].row = 0;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    bad = valid;
    bad[1].choice = 8;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    bad = valid;
    bad[1].expert = 1;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    assert!(check(&active, &valid, &[0, 5]).is_err());
    assert!(check(&active, &valid, &[2, 0]).is_err());
    active[1] = 1;
    assert!(check(&active, &valid, &[0, 2]).is_err());
    println!(
        "PASS eager_moe_dispatch_controls exact_cover=true compacted_rows=true original_choice=true bounds=true"
    );
    Ok(())
}

#[test]
fn injected_layers_bind_hybrid_mixer_geometry() -> Result<()> {
    use crate::model_spec::{Architecture, Mixer};
    let spec = hybrid_spec()?;
    let manifest = hybrid_manifest(&spec)?;
    assert_eq!(spec.architecture, Architecture::MiniCpmV47);
    assert_eq!(
        (spec.q_heads, spec.kv_heads, spec.head_dim, spec.rotary_dim),
        (16, 2, 256, 64)
    );
    let g = crate::op_contract::GdnSpec::from_model(&spec)?;
    assert_eq!(
        (g.hk, g.hv, g.dk, g.dv, g.conv_width),
        (16, 32, 128, 128, 4)
    );
    for (layer, mixer) in [Mixer::Gdn, Mixer::FullAttention].into_iter().enumerate() {
        assert!(matches!(
            crate::model::layer_program(&spec, layer)?[1],
            crate::model::LayerStage::Mixer(bound) if bound == mixer
        ));
        let plan = crate::linear_plan::LayerDequantPlan::from_manifest(
            &manifest,
            layer as u32,
            spec.linear_format,
        )?;
        assert_eq!(plan.bytes, 0);
        assert!(plan.slots.is_empty());
    }
    for (name, shape) in [
        ("0.linear_attn.in_proj_qkv.weight", [8192, 128]),
        ("0.linear_attn.out_proj.weight", [128, 4096]),
        ("1.self_attn.q_proj.weight", [8192, 128]),
        ("1.self_attn.k_proj.weight", [512, 128]),
        ("1.self_attn.o_proj.weight", [128, 4096]),
    ] {
        assert_eq!(
            manifest.entries[&format!("model.language_model.layers.{name}")].shape,
            shape
        );
    }
    assert!(qwen::Runtime::workspace(&spec, 8, 32, crate::gdn_plan::GdnPolicy::LEGACY)? > 0);
    let mut old = spec.clone();
    old.architecture = Architecture::Llama;
    assert!(hybrid_manifest(&old).is_err());
    old = spec.clone();
    old.head_dim = 128;
    old.q_width = old.q_heads * old.head_dim;
    assert!(hybrid_manifest(&old).is_err());
    println!(
        "PASS eager_moe_hybrid_preflight architecture=MiniCpmV47 layers=Gdn,FullAttention q16_kv2_d256_rotary64=true gdn_hk16_hv32=true legacy_rejected=true d128_rejected=true GPU_executed=false"
    );
    Ok(())
}

#[test]
fn injected_layer_checks_rope_preconditions_before_launch() -> Result<()> {
    use crate::backend::cuda::flash::Shape;

    let shape = Shape {
        rows: 2,
        start: 0,
        q_heads: 16,
        kv_heads: 2,
        capacity: 32,
        head_dim: 128,
        causal: true,
    };
    super::super::super::ops::check_rope_kv(shape, 10000.0)?;
    for invalid in [
        Shape {
            q_heads: 1,
            kv_heads: 1,
            ..shape
        },
        Shape {
            head_dim: 256,
            ..shape
        },
        Shape { rows: 0, ..shape },
        Shape { start: 31, ..shape },
        Shape {
            capacity: 0,
            ..shape
        },
    ] {
        assert!(super::super::super::ops::check_rope_kv(invalid, 10000.0).is_err());
    }
    for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(super::super::super::ops::check_rope_kv(shape, theta).is_err());
    }
    println!(
        "PASS eager_moe_rope_preflight q_heads=16 kv_heads=2 head_dim=128 unsupported_heads_rejected=true dimension_rejected=true theta_rejected=true extent_rejected=true"
    );
    Ok(())
}
