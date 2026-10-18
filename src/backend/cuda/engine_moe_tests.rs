use super::*;
use crate::backend::cuda::moe::reference::{MoeDims, MoeFixture, fixture};
use crate::experts::plan::Capacity;

#[test]
fn expert_host_profile_obeys_the_current_cgroup_limit() -> Result<()> {
    let (available, locked) = super::host_memory()?;
    ensure!(
        available > 0 && locked <= available,
        "invalid expert host memory profile"
    );
    let groups = std::fs::read_to_string("/proc/self/cgroup")?;
    let group = groups
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup missing")?;
    let limit = Path::new("/sys/fs/cgroup")
        .join(group.trim_start_matches('/'))
        .join("memory.max");
    if limit.try_exists()? {
        let text = std::fs::read_to_string(limit)?;
        if text.trim() != "max" {
            ensure!(
                available <= text.trim().parse::<u64>()?,
                "expert arena exceeds cgroup limit"
            );
        }
    }
    Ok(())
}

fn native_fixture(
    fixture: &MoeFixture,
    directory: &Path,
) -> Result<Arc<crate::experts::catalogue::ExpertCatalogue>> {
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let d = fixture.dims;
    let mut bytes = Vec::new();
    let mut manifest = TensorManifest {
        entries: Default::default(),
        resident_bytes: 0,
    };
    for layer in 0..geometry.layers {
        for (role, values, shape) in [
            (
                "gate_up_proj",
                &fixture.gate_up,
                vec![d.experts, 2 * d.inter, d.hidden],
            ),
            (
                "down_proj",
                &fixture.down,
                vec![d.experts, d.hidden, d.inter],
            ),
        ] {
            let start = bytes.len() as u64;
            bytes.extend(values.iter().flat_map(|value| {
                let value = if layer == 0 {
                    *value
                } else {
                    bf16::from_f32(-0.75 * value.to_f32())
                };
                value.to_bits().to_le_bytes()
            }));
            let end = bytes.len() as u64;
            manifest.entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.{role}"),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape,
                    shard: "expert-fixture.bin".into(),
                    bytes: (end - start) as usize,
                    load: false,
                    data_start: 0,
                    data_offsets: [start, end],
                    scope: TensorScope::Text,
                    layer: Some(layer),
                    role: format!("mlp.experts.{role}"),
                    scale_name: None,
                },
            );
        }
    }
    std::fs::create_dir_all(directory)?;
    let path = directory.join("expert-fixture.bin");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    std::io::Write::write_all(&mut file, &bytes)?;
    Ok(Arc::new(
        crate::experts::catalogue::ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Bf16,
        )?,
    ))
}

pub(super) fn hybrid_spec() -> Result<crate::model_spec::ModelSpec> {
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
    pub(crate) fn load_moe_test(
        fixture: &MoeFixture,
        runtime: &Path,
        workspace_rows: usize,
    ) -> Result<Self> {
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let count = geometry.layers * geometry.experts;
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?;
        let capacity = Capacity {
            layout: layout.clone(),
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: layout.slot_bytes() * u64::from(count),
            host_expert_bytes: 0,
            slots_per_layer: vec![geometry.experts; geometry.layers as usize],
            host_blocks: 0,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: false,
        };
        Self::load_moe_placed(
            fixture,
            runtime,
            &capacity,
            &vec![(0..geometry.experts).collect::<Vec<_>>(); geometry.layers as usize],
            crate::experts::plan::Admission {
                budget_bytes_per_token: 0,
            },
            2,
            workspace_rows,
        )
    }

    pub(in crate::backend::cuda::engine) fn load_moe_placed(
        fixture: &MoeFixture,
        runtime: &Path,
        capacity: &Capacity,
        initial: &[Vec<u32>],
        admission: crate::experts::plan::Admission,
        engines: u32,
        workspace_rows: usize,
    ) -> Result<Self> {
        use crate::backend::cuda::{
            linear::LayerLinearPlan,
            weights::{StoredTensor, WeightCatalog},
        };
        let d = fixture.dims;
        ensure!(
            d.hidden == 128 && d.inter == 64 && d.experts == 16 && d.top_k == 4,
            "MoE hook expects H128/I64/E16/K4, received {d:?}"
        );
        ensure!(
            (8..=32).contains(&workspace_rows),
            "hybrid fixture workspace_rows={workspace_rows} outside8..32"
        );
        let model_spec = hybrid_spec()?;
        let config = model_spec.config();
        let manifest = hybrid_manifest(&model_spec)?;
        let geometry = ExpertGeometry::from_model(&model_spec)?;
        let service = tier_service(capacity.worker_count)?;
        let cpu_bounds = if capacity.prefill_staging_slots > 0 {
            let bound = residency::cpu_bound(
                geometry,
                capacity.slot_bytes(),
                capacity.span_rows,
                &service,
                1_000_000_000.0,
            )?;
            (bound.0.max(8), bound.1.max(8 * geometry.top_k as usize))
        } else {
            (8, 8 * geometry.top_k as usize)
        };
        let device = Device::new(0)?;
        let stream = &device.stream;
        let catalogue = native_fixture(fixture, runtime)?;
        let mut slots = Slots::load(
            &device,
            geometry,
            residency::Load {
                capacity,
                admission,
                initial,
                catalogue,
                directory: runtime,
                engines,
                max_rows: cpu_bounds.0 as u32,
            },
        )?;
        std::fs::remove_file(runtime.join("expert-fixture.bin"))?;
        let mut frequency = vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize];
        for (layer, experts) in initial.iter().enumerate() {
            for &expert in experts {
                frequency[layer][expert as usize] = 1.0;
            }
        }
        slots.begin_request(&frequency)?;
        device.upload.synchronize()?;
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
        let weights = Weights {
            embed: stream.clone_htod(
                &(0..32 * d.hidden)
                    .map(|index| {
                        let (token, column) = (index / d.hidden, index % d.hidden);
                        if column < 8 {
                            if token % 8 == column {
                                bf16::ONE
                            } else {
                                bf16::ZERO
                            }
                        } else {
                            bf16::from_f32((((token * 37 + column * 11) % 53) as f32 - 26.0) / 64.0)
                        }
                    })
                    .collect::<Vec<_>>(),
            )?,
            norm: stream.clone_htod(&vec![bf16::ONE; d.hidden])?,
            head: stream.clone_htod(
                &(0..32 * d.hidden)
                    .map(|index| {
                        bf16::from_f32(
                            (((index / d.hidden * 17 + index % d.hidden * 7) % 61) as f32 - 30.0)
                                / 128.0,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?,
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
        let capture_checkpoint = Some(super::super::super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            stream,
        )?);
        let mut work = Work::new(stream, &model_spec, workspace_rows)?;
        stream.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = Some(Flash::new(
            stream.clone(),
            workspace_rows,
            model_spec.q_heads as usize,
        )?);
        let moe = Some(Decode::new(
            &device,
            geometry,
            8,
            capacity.worker_count,
            cpu_bounds,
        )?);
        let moe_prefill = if capacity.prefill_staging_slots > 0 {
            let reserves = crate::experts::plan::Reserves {
                device_fixed_bytes: 0,
                host_fixed_bytes: 0,
                residual_row_bytes: u64::from(geometry.hidden) * 2,
                normalized_row_bytes: u64::from(geometry.hidden) * 2,
                contribution_row_bytes: u64::from(geometry.top_k) * u64::from(geometry.hidden) * 2,
                done_row_bytes: u64::from(geometry.top_k) * 4,
                route_row_bytes: u64::from(geometry.top_k) * 18,
                prefill_staging_slots: capacity.prefill_staging_slots,
                disk_ring_slots: 0,
            };
            Some(Prefill::new(
                &device,
                &mut slots,
                capacity,
                reserves,
                service,
                1_000_000_000.0,
                (workspace_rows, 8),
            )?)
        } else {
            None
        };
        let mut engine = Self {
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            qwen: None,
            moe,
            moe_slots: Some(slots),
            moe_prefill,
            moe_test_ffn: Some(ffn),
            mtp: None,
            mtp_state: None,
            mtp_epoch: 0,
            images: None,
            qwen_device_prefix: false,
            forward_rows: (1..=workspace_rows).collect(),
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
            capture_checkpoint,
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
            workspace_rows,
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
            workspace_rows,
            32,
            crate::gdn_plan::GdnPolicy::for_device(engine.device.info.arch, engine.device.info.sms),
        )?)?;
        engine.qwen = Some(qwen::Runtime::new(
            &engine, runtime, catalog, None, scratch, None, 8,
        )?);
        for rows in 1..=workspace_rows {
            engine.prepare_qwen(rows)?;
            engine.prepare_prefill_rows(rows)?;
        }
        engine.device.finish_upload()?;
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
        self.moe_slots
            .as_mut()
            .context("injected layer has no residency")?
            .begin_token()?;
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
        )?;
        self.moe_slots
            .as_mut()
            .context("injected layer lost residency")?
            .finish_token()
    }

    pub(crate) fn record_test_stats(&self, trace: &mut Trace) -> Result<()> {
        trace.add(
            "stats.vision_encode_calls".into(),
            vec![1],
            vec![self.qwen.as_ref().map_or(0, |q| q.vision_encode_calls) as f32],
        )?;
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
        self.moe_slots
            .as_ref()
            .context("MoE residency statistics missing")?
            .trace(trace)?;
        Ok(())
    }
}

fn graph_state(engine: &Engine) -> Result<Vec<u32>> {
    let stream = &engine.device.stream;
    let mut values = vec![engine.position as u32];
    for kv in &engine.state.kv {
        for data in [&kv.k, &kv.v] {
            values.extend(
                stream
                    .clone_dtoh(data)?
                    .into_iter()
                    .map(|v| u32::from(v.to_bits())),
            );
        }
    }
    for state in &engine.state.gdn {
        for data in [&state.s, &state.history] {
            values.extend(stream.clone_dtoh(data)?.into_iter().map(f32::to_bits));
        }
    }
    if let Some(sequence) = &engine.state.sequence {
        values.extend(stream.clone_dtoh(sequence)?.into_iter().map(|v| v as u32));
    }
    values.extend(stream.clone_dtoh(&engine.work.token)?);
    values.extend(
        stream
            .clone_dtoh(&engine.work.position)?
            .into_iter()
            .map(|v| v as u32),
    );
    values.extend(
        stream
            .clone_dtoh(&engine.work.length)?
            .into_iter()
            .map(|v| v as u32),
    );
    Ok(values)
}

#[test]
#[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
fn moe_graphs_match_eager_and_preserve_capture_state_gpu() -> Result<()> {
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("graph gate runtime missing")?,
    );
    let fixture = tier_fixture(3);
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let frequency = vec![vec![1.0; 16]; 2];
    let mut defects = 0;
    let mut control = false;
    for (resident, pinned, engines, cap) in [
        (true, false, 1, 0),
        (false, false, 1, 2),
        (false, true, 2, 2),
        (false, false, 2, 0),
    ] {
        let mut capacity = tier_capacity(geometry, pinned)?;
        if cap == 0 {
            capacity.spare_blocks = 1;
            capacity.host_expert_bytes = 9 * capacity.slot_bytes();
        }
        let mut pair = (0..2)
            .map(|arm| {
                let directory =
                    runtime.join(format!("graphs-{resident}-{pinned}-{engines}-{cap}-{arm}"));
                if resident {
                    Engine::load_moe_test(&fixture, &directory, 8)
                } else {
                    Engine::load_moe_placed(
                        &fixture,
                        &directory,
                        &capacity,
                        &[vec![0, 1, 2, 3], vec![0, 1, 2, 3]],
                        crate::experts::plan::Admission {
                            budget_bytes_per_token: cap * capacity.slot_bytes(),
                        },
                        engines,
                        8,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidate = pair.pop().context("graph candidate missing")?;
        let mut reference = pair.pop().context("graph reference missing")?;
        for engine in [&mut reference, &mut candidate] {
            engine.reset()?;
            engine
                .moe_slots
                .as_mut()
                .context("graph slots missing")?
                .begin_request(&frequency)?;
            engine.closure.freeze_fixed()?;
        }
        candidate.declare_graphs()?;
        let before = graph_state(&candidate)?;
        candidate.capture()?;
        let after = graph_state(&candidate)?;
        let capture_mismatches = before.iter().zip(&after).filter(|(a, b)| a != b).count()
            + before.len().abs_diff(after.len());
        defects += usize::from(capture_mismatches != 0);
        let expected_graphs = if resident { 1 } else { 6 };
        ensure!(
            candidate.graphs.len() == expected_graphs,
            "Runner FAIL graph declaration count"
        );
        candidate.closure.expect_captures(expected_graphs)?;
        candidate.check_graphs()?;
        reference.closure.ready(Instant::now())?;
        candidate.closure.ready(Instant::now())?;
        println!(
            "{} moe_graph_capture resident={resident} pinned={pinned} engines={engines} admission_cap={cap} graphs={expected_graphs} state_bit_mismatches={capture_mismatches}",
            if capture_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        for request in 0..2 {
            if request > 0 {
                for engine in [&mut reference, &mut candidate] {
                    engine.reset()?;
                    engine
                        .moe_slots
                        .as_mut()
                        .context("graph reset slots missing")?
                        .begin_request(&frequency)?;
                }
            }
            for step in 0..6 {
                let token = (3 + step * 7 + request * 11) % 32;
                reference.set_token(token)?;
                candidate.set_token(token)?;
                {
                    let _range =
                        super::super::super::profile::Range::new(true, c"moe_graph_reference");
                    reference.decode(false, None)?;
                }
                {
                    let _range =
                        super::super::super::profile::Range::new(true, c"moe_graph_candidate");
                    candidate.decode(true, None)?;
                }
                let a = reference.decode_logits()?;
                let b = candidate.decode_logits()?;
                ensure!(
                    a.iter().chain(&b).all(|v| v.is_finite()),
                    "Runner FAIL graph comparison is nonfinite"
                );
                let logits = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let a = graph_state(&reference)?;
                let b = graph_state(&candidate)?;
                let state =
                    a.iter().zip(&b).filter(|(a, b)| a != b).count() + a.len().abs_diff(b.len());
                let mut a = Trace::default();
                let mut b = Trace::default();
                reference.record_test_stats(&mut a)?;
                candidate.record_test_stats(&mut b)?;
                let mut placement = 0;
                for key in ["residency.tier", "residency.generation"] {
                    let (a, b) = (a.values(key)?, b.values(key)?);
                    placement += a
                        .iter()
                        .zip(b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
                let mismatch = logits + state + placement;
                defects += usize::from(mismatch != 0);
                println!(
                    "{} moe_graph_decode resident={resident} pinned={pinned} engines={engines} admission_cap={cap} request={request} step={step} logits_bit_mismatches={logits} state_bit_mismatches={state} placement_mismatches={placement} graph_launches={expected_graphs}",
                    if mismatch == 0 { "PASS" } else { "DEFECT" }
                );
            }
        }
        if !resident && !control {
            for engine in [&mut reference, &mut candidate] {
                engine.reset()?;
                engine
                    .moe_slots
                    .as_mut()
                    .context("graph control slots missing")?
                    .begin_request(&frequency)?;
                engine.set_token(3)?;
            }
            reference.decode(false, None)?;
            {
                let segments = &mut candidate
                    .moe
                    .as_mut()
                    .context("control executor missing")?
                    .segments;
                let (first, tail) = segments.split_at_mut(1);
                std::mem::swap(&mut first[0].join, &mut tail[0].join);
            }
            candidate.decode(true, None)?;
            let a = reference.decode_logits()?;
            let b = candidate.decode_logits()?;
            ensure!(
                a.iter().chain(&b).all(|v| v.is_finite()),
                "Runner FAIL wrong-join control is nonfinite"
            );
            let mismatches = a
                .iter()
                .zip(&b)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                mismatches > 0,
                "Runner FAIL wrong-join control was not detected"
            );
            println!("PASS moe_graph_control wrong_join_logits_bit_mismatches={mismatches}");
            control = true;
        }
    }
    ensure!(control, "Runner FAIL graph control did not execute");
    ensure!(defects == 0, "DEFECT MoE graph cells={defects}");
    println!("PASS moe_graph_gate cases=4 requests=2 steps=6 bitwise=true control=wrong_join");
    Ok(())
}

fn step(rows: usize) -> crate::backend::Step<'static> {
    crate::backend::Step {
        rows,
        input: crate::backend::Input::Ids,
        mask: crate::backend::Mask::Causal,
        head: Head::None,
    }
}

pub(super) fn tier_fixture(variant: u32) -> MoeFixture {
    let mut values = fixture(MoeDims::new(8, 16, 4, 128, 64), variant);
    values.router_w.fill(bf16::ZERO);
    for row in 0..8 {
        values.x[row * 128..row * 128 + 8].fill(bf16::ZERO);
        values.x[row * 128 + row] = bf16::ONE;
        let experts = routed_experts(row);
        for expert in 0..16 {
            let score = experts
                .iter()
                .position(|&id| id == expert)
                .map_or(-1.0 - expert as f32 / 32.0, |rank| 3.0 - rank as f32 / 2.0);
            values.router_w[expert as usize * 128 + row] = bf16::from_f32(score);
        }
    }
    values
}

fn routed_experts(row: usize) -> [u32; 4] {
    let offset = row as u32;
    [
        offset % 4,
        4 + (offset + 2) % 4,
        8 + (offset + 1) % 4,
        12 + (offset + 3) % 4,
    ]
}

pub(super) fn tier_service(workers: u32) -> Result<crate::experts::plan::ServiceTable> {
    crate::experts::plan::ServiceTable::new(
        "fixture",
        [1, 2, 5, 8, 16, 32, 64]
            .into_iter()
            .flat_map(|rows| {
                [1, 2, 4, 8]
                    .into_iter()
                    .map(move |experts| crate::experts::plan::ServiceCell {
                        workers,
                        rows,
                        experts,
                        seconds: f64::from(rows * experts) * 0.00002,
                    })
            })
            .collect(),
    )
}

pub(super) fn tier_capacity(geometry: ExpertGeometry, pinned: bool) -> Result<Capacity> {
    let layout = crate::experts::catalogue::ExpertLayout::new(
        crate::model_spec::ExpertFormat::Bf16,
        &geometry,
    )?;
    let slot = layout.slot_bytes();
    Ok(Capacity {
        layout,
        span_rows: 8,
        wave_rows: 8,
        device_expert_bytes: slot * 8,
        host_expert_bytes: slot * 10,
        slots_per_layer: vec![4, 4],
        host_blocks: 8,
        spare_blocks: 2,
        disk_bytes: slot * 16,
        worker_count: 2,
        disk_workers: 2,
        prefill_staging_slots: 0,
        pinned,
    })
}

fn tier_schedule() -> [(&'static [u32], &'static [usize]); 3] {
    [
        (&[0], &[1, 1, 1, 1, 1, 1, 2, 5, 8, 2]),
        (&[0], &[8, 2, 1]),
        (&[4, 5, 7, 8], &[8, 1]),
    ]
}

struct TierOracle {
    policy: crate::experts::runtime::Residency,
    reader: crate::experts::disk::DiskReader,
    catalogue: Arc<crate::experts::catalogue::ExpertCatalogue>,
}

impl TierOracle {
    fn new(fixture: &MoeFixture, directory: &Path, capacity: &Capacity) -> Result<Self> {
        use crate::experts::{
            arena::SlotLayout,
            disk::DiskReader,
            runtime::{LoadInputs, Residency},
        };
        let catalogue = native_fixture(fixture, directory)?;
        let geometry = catalogue.geometry();
        let layout = SlotLayout::new(geometry, capacity)?;
        let host = if layout.host_blocks() > 0 {
            Some(HostBlocks::new(
                layout.host_blocks(),
                usize::try_from(capacity.slot_bytes() / 2)?,
            )?)
        } else {
            None
        };
        let reader = DiskReader::open(&catalogue, directory, 1)?;
        let (policy, _) = Residency::load(LoadInputs {
            geometry,
            capacity: capacity.clone(),
            admission: crate::experts::plan::Admission {
                budget_bytes_per_token: 2 * capacity.slot_bytes(),
            },
            max_rows: 8,
            initial: &vec![vec![0, 1, 2, 3]; geometry.layers as usize],
            host: host.as_ref(),
            catalogue: &catalogue,
            reader: &reader,
        })?;
        std::fs::remove_file(directory.join("expert-fixture.bin"))?;
        Ok(Self {
            policy,
            reader,
            catalogue,
        })
    }

    fn routes(rows: usize) -> Vec<crate::experts::runtime::Route> {
        (0..rows)
            .flat_map(|row| {
                routed_experts(row)
                    .into_iter()
                    .enumerate()
                    .map(move |(rank, expert)| crate::experts::runtime::Route {
                        row: row as u32,
                        choice: (row * 4 + rank) as u32,
                        expert,
                        weight: bf16::from_f32(0.25),
                    })
            })
            .collect()
    }

    fn cpu_values(
        &self,
        layer: u32,
        input: &[bf16],
        reference: &Trace,
        active: &[u32],
    ) -> Result<Vec<bf16>> {
        use crate::cpu::expert_ffn::{Contribution, Isa, Scratch, contribution};
        let shape = Shape::new(128, 64)?;
        let mut scratch = Scratch::new(&shape);
        let native = ExpertWeights {
            gate_up: 0..16384,
            gate_up_stride: 128,
            down: 16384..24576,
            down_stride: 64,
        };
        let mut output = reference
            .values("layer.moe.contributions")?
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let ids = reference.values("layer.moe.ids")?;
        let weights = reference.values("layer.moe.weights")?;
        for (choice, &active) in active.iter().enumerate() {
            if active == 1 {
                continue;
            }
            let slot = self
                .reader
                .read_expert(layer, ids[choice] as u32, &self.catalogue)?;
            let values = self.reader.values(slot)?;
            let row = choice / 4;
            contribution(
                Isa::detect(),
                Contribution {
                    shape: &shape,
                    arena: &values,
                    weights: &native,
                    hidden: &input[row * 128..(row + 1) * 128],
                    routing: bf16::from_f32(weights[choice]),
                },
                &mut output[choice * 128..(choice + 1) * 128],
                &mut scratch,
            )?;
        }
        Ok(output)
    }

    fn compare_state(&self, engine: &Engine) -> Result<usize> {
        use crate::experts::plan::Tier;
        let mut actual = Trace::default();
        engine.record_test_stats(&mut actual)?;
        let tiers = actual.values("residency.tier")?;
        let generations = actual.values("residency.generation")?;
        let mut mismatches = 0;
        for layer in 0..2 {
            for expert in 0..16 {
                let tier = self.policy.ledger().tier(layer, expert)?;
                let value = match tier {
                    Tier::Gpu => 0.0,
                    Tier::Host => 1.0,
                    Tier::Disk => 2.0,
                };
                let generation = if tier == Tier::Gpu {
                    self.policy.generation(layer, expert)?
                } else {
                    0
                };
                let index = layer as usize * 16 + expert as usize;
                mismatches +=
                    usize::from(tiers[index] != value || generations[index] != generation as f32);
            }
        }
        Ok(mismatches)
    }
}

#[test]
fn decode_residency_fixture_covers_tiers_requests_and_cap() -> Result<()> {
    use crate::experts::runtime::{MoveSource, Route};
    let directory = std::env::temp_dir().join(format!(
        "flyingfish-decode-residency-{}",
        std::process::id()
    ));
    let fixture = tier_fixture(3);
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let mut oracle = TierOracle::new(&fixture, &directory, &tier_capacity(geometry, false)?)?;
    for layer in 0..2 {
        for expert in 0..16 {
            let slot = oracle
                .reader
                .read_expert(layer, expert, &oracle.catalogue)?;
            let values = oracle.reader.values(slot)?;
            let expert = expert as usize;
            let expected = fixture.gate_up[expert * 16384..(expert + 1) * 16384]
                .iter()
                .chain(&fixture.down[expert * 8192..(expert + 1) * 8192])
                .map(|&value| {
                    if layer == 0 {
                        value
                    } else {
                        bf16::from_f32(value.to_f32() * -0.75)
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(values.values(), expected);
        }
    }
    let mut event = 1;
    let (
        mut host,
        mut disk,
        mut admissions,
        mut host_moves,
        mut disk_moves,
        mut chains,
        mut compact,
    ) = (0, 0, 0, 0, 0, 0, false);
    let mut writebacks = 0;
    for (target, rows) in tier_schedule() {
        let mut frequency = vec![vec![0.0; 16]; 2];
        for layer in &mut frequency {
            for &expert in target {
                layer[expert as usize] = 1.0;
            }
        }
        let moves = oracle.policy.begin_request(&frequency)?.to_vec();
        for movement in moves {
            writebacks += usize::from(movement.dest_block.is_some());
            host_moves += usize::from(matches!(movement.source, MoveSource::Host(_)));
            disk_moves += usize::from(matches!(movement.source, MoveSource::Disk));
            chains += usize::from(movement.chain_slot.is_some());
            oracle.policy.record_copy(&[movement.slot], event)?;
            event += 1;
        }
        for &rows in rows {
            oracle.policy.begin_token()?;
            let routes: Vec<Route> = TierOracle::routes(rows);
            for layer in 0..2 {
                let dispatch = oracle.policy.dispatch(layer, &routes)?;
                host += dispatch.host.len();
                disk += dispatch.disk.len();
                compact |= dispatch.miss_rows.first().is_some_and(|&row| row > 0);
            }
            let updates = oracle.policy.finish_token()?.to_vec();
            assert!(updates.len() <= 2);
            admissions += updates.len();
            for update in updates {
                writebacks += usize::from(update.dest_block.is_some());
                host_moves += usize::from(update.from_host);
                disk_moves += usize::from(!update.from_host);
                oracle.policy.record_copy(&[update.slot], event)?;
                event += 1;
            }
        }
        oracle.policy.end_request()?;
    }
    assert!(
        host > 0
            && disk > 0
            && admissions > 0
            && host_moves > 0
            && disk_moves > 0
            && chains > 0
            && compact
    );
    drop(oracle);
    std::fs::remove_dir(directory)?;
    println!(
        "PASS decode_residency_cpu_preflight host_choices={host} disk_choices={disk} admissions={admissions} host_moves={host_moves} disk_moves={disk_moves} writebacks={writebacks} chains={chains} compact_nonzero_base={compact} requests=3 cap=2 GPU_executed=false"
    );
    Ok(())
}

fn direct(engine: &Engine, layer: usize, input: &[bf16], rows: usize) -> Result<Trace> {
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
    let mut generation = stream.alloc_zeros::<u64>(rows * k)?;
    let contributions = stream.alloc_zeros::<bf16>(rows * k * h)?;
    let shared_out = stream.alloc_zeros::<bf16>(rows * h)?;
    let done = stream.alloc_zeros::<u32>(rows * k)?;
    let y = stream.alloc_zeros::<bf16>(rows * h)?;
    let scratch = stream.alloc_zeros::<u8>(decode_scratch_bytes(rows, k, i) as usize)?;
    let slots = engine
        .moe_slots
        .as_ref()
        .context("test slots absent")?
        .bindings(layer, stream)?;
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
    let routed = stream.clone_dtoh(&ids)?;
    let published = &engine
        .moe_slots
        .as_ref()
        .context("direct reference slots absent")?
        .published;
    let generations = routed
        .iter()
        .map(|&id| published[layer * 16 + id as usize])
        .collect::<Vec<_>>();
    stream.memcpy_htod(&generations, &mut generation)?;
    DecodeOps::new(ctx)?.decode(
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
    )?;
    stream.synchronize()?;
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
        ensure!(
            a.iter().chain(&b).all(|value| value.is_finite()),
            "Runner FAIL reference/candidate trace nonfinite for {name}"
        );
        mismatch += a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    Ok(mismatch)
}

fn combine_expected(engine: &Engine, contributions: &[bf16], rows: usize) -> Result<Vec<bf16>> {
    let stream = &engine.device.stream;
    let decode = engine.moe.as_ref().context("reference decode missing")?;
    let x = qwen::matrix(&engine.work.n, stream, 0, rows, 128)?;
    let ffn = engine.moe_test_ffn.context("reference FFN missing")?;
    let expected_contributions = stream.clone_htod(contributions)?;
    let expected_out = stream.alloc_zeros::<bf16>(rows * 128)?;
    let expected_scale = stream.alloc_zeros::<bf16>(rows)?;
    let BoundFfn::Moe { shared_gate, .. } = ffn else {
        anyhow::bail!("dense CPU control FFN");
    };
    let scale = qwen::view(&expected_scale, stream, 0, [rows as u32, 1], [1, 1])?;
    let combine = CombineOps::new(&engine.device.ctx)?;
    combine.shared_scale(stream, x, shared_gate, &mut TensorMut(scale))?;
    combine.combine(
        stream,
        &MoeCombine {
            ids: qwen::view(&decode.ids, stream, 0, [rows as u32, 4], [4, 1])?,
            final_active: qwen::view(&decode.final_active, stream, 0, [rows as u32, 4], [4, 1])?,
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
    )?;
    stream.synchronize()?;
    Ok(stream.clone_dtoh(&expected_out)?)
}

fn tier_cell(
    engine: &mut Engine,
    oracle: &mut TierOracle,
    reference: &Trace,
    layer: u32,
    input: &[bf16],
    rows: usize,
) -> Result<(usize, usize, usize)> {
    let ids = reference.values("layer.moe.ids")?;
    let weights = reference.values("layer.moe.weights")?;
    let routes = ids
        .iter()
        .zip(&weights)
        .enumerate()
        .map(
            |(choice, (&expert, &weight))| crate::experts::runtime::Route {
                row: choice as u32 / 4,
                choice: choice as u32,
                expert: expert as u32,
                weight: bf16::from_f32(weight),
            },
        )
        .collect::<Vec<_>>();
    ensure!(
        routes
            .iter()
            .all(|route| route.expert
                == routed_experts(route.row as usize)[route.choice as usize % 4]),
        "Runner FAIL tier reference ids differ from the frozen 0.5-margin logits"
    );
    let dispatch = oracle.policy.dispatch(layer, &routes)?.clone();
    let expected = oracle.cpu_values(layer, input, reference, &dispatch.active)?;
    engine
        .device
        .stream
        .memcpy_htod(input, &mut engine.work.n.slice_mut(..input.len()))?;
    let mut trace = Trace::default();
    engine.run_ffn(
        layer as usize,
        step(rows),
        &mut Some(Target {
            trace: &mut trace,
            prefix: "layer",
            skip: 0,
        }),
    )?;
    for (name, length) in [
        ("layer.moe.ids", rows * 4),
        ("layer.moe.weights", rows * 4),
        ("layer.moe.shared", rows * 128),
        ("layer.moe.contributions", rows * 4 * 128),
        ("layer.moe.done", rows * 4),
        ("layer.ffn", rows * 128),
    ] {
        let values = trace.values(name)?;
        let baseline = reference.values(name)?;
        ensure!(
            values.len() == length
                && baseline.len() == length
                && values
                    .iter()
                    .chain(&baseline)
                    .all(|value| value.is_finite()),
            "Runner FAIL tier trace {name} extent/nonfinite layer={layer} rows={rows} expected={length} actual={} reference={}",
            values.len(),
            baseline.len()
        );
    }
    let mut mismatches = 0;
    for name in ["layer.moe.ids", "layer.moe.weights", "layer.moe.shared"] {
        mismatches += trace
            .values(name)?
            .iter()
            .zip(reference.values(name)?)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    mismatches += trace
        .values("layer.moe.contributions")?
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    mismatches += trace
        .values("layer.moe.done")?
        .iter()
        .filter(|&&done| done != 1.0)
        .count();
    let expected_out = combine_expected(engine, &expected, rows)?;
    mismatches += trace
        .values("layer.ffn")?
        .iter()
        .zip(expected_out)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    println!(
        "{} decode_residency_cell layer={layer} rows={rows} gpu_choices={} host_choices={} disk_choices={} compacted_rows={:?} bit_mismatches={mismatches}",
        if mismatches == 0 { "PASS" } else { "DEFECT" },
        dispatch.active.iter().sum::<u32>(),
        dispatch.host.len(),
        dispatch.disk.len(),
        dispatch.miss_rows
    );
    Ok((
        usize::from(mismatches != 0),
        dispatch.host.len(),
        dispatch.disk.len(),
    ))
}

fn tier_gate(variants: &[u32], cases: &[(bool, u32)]) -> Result<()> {
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("tier gate runtime missing")?,
    );
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let mut defects = 0;
    for &variant in variants {
        let fixture = tier_fixture(variant);
        let reference =
            Engine::load_moe_test(&fixture, &runtime.join(format!("reference-{variant}")), 8)?;
        let mut references = std::collections::BTreeMap::new();
        for layer in 0..2 {
            for rows in [1, 2, 5, 8] {
                let input = &fixture.x[..rows * 128];
                let trace = direct(&reference, layer, input, rows)?;
                ensure!(
                    compare(&trace, &direct(&reference, layer, input, rows)?)? == 0,
                    "Runner FAIL tier same-form reference repeat differs layer={layer} rows={rows}"
                );
                references.insert((layer, rows), trace);
            }
        }
        for &(pinned, engines) in cases {
            let directory = runtime.join(format!("tier-{variant}-{pinned}-{engines}"));
            let capacity = tier_capacity(geometry, pinned)?;
            let mut oracle = TierOracle::new(&fixture, &directory.join("oracle"), &capacity)?;
            let mut engine = Engine::load_moe_placed(
                &fixture,
                &directory,
                &capacity,
                &[vec![0, 1, 2, 3], vec![0, 1, 2, 3]],
                crate::experts::plan::Admission {
                    budget_bytes_per_token: 2 * capacity.slot_bytes(),
                },
                engines,
                8,
            )?;
            engine
                .moe_slots
                .as_mut()
                .context("tier slots missing")?
                .end_request()?;
            let (mut host_choices, mut disk_choices, mut admissions) = (0, 0, 0);
            let mut event = 1;
            let _range = super::super::super::profile::Range::new(true, c"engine_moe_residency");
            for (request, (target, rows)) in tier_schedule().into_iter().enumerate() {
                let mut frequency = vec![vec![0.0; 16]; 2];
                for layer in &mut frequency {
                    for &expert in target {
                        layer[expert as usize] = 1.0;
                    }
                }
                let moves = oracle.policy.begin_request(&frequency)?.to_vec();
                for movement in &moves {
                    oracle.policy.record_copy(&[movement.slot], event)?;
                    event += 1;
                }
                {
                    let slots = engine
                        .moe_slots
                        .as_mut()
                        .context("tier request slots missing")?;
                    slots.delay_copies()?;
                    slots.begin_request(&frequency)?;
                }
                defects += oracle.compare_state(&engine)?;
                for &rows in rows {
                    oracle.policy.begin_token()?;
                    engine
                        .moe_slots
                        .as_mut()
                        .context("tier token slots missing")?
                        .begin_token()?;
                    for layer in 0..2 {
                        let (cell, host, disk) = tier_cell(
                            &mut engine,
                            &mut oracle,
                            &references[&(layer, rows)],
                            layer as u32,
                            &fixture.x[..rows * 128],
                            rows,
                        )?;
                        defects += cell;
                        host_choices += host;
                        disk_choices += disk;
                    }
                    let updates = oracle.policy.finish_token()?.to_vec();
                    ensure!(
                        updates.len() <= 2,
                        "Runner FAIL CPU oracle exceeds frozen admission cap"
                    );
                    admissions += updates.len();
                    for update in &updates {
                        oracle.policy.record_copy(&[update.slot], event)?;
                        event += 1;
                    }
                    let slots = engine
                        .moe_slots
                        .as_mut()
                        .context("tier admission slots missing")?;
                    slots.delay_copies()?;
                    slots.finish_token()?;
                    let mismatch = oracle.compare_state(&engine)?;
                    defects += mismatch;
                    println!(
                        "{} decode_residency_admission variant={variant} pinned={pinned} engines={engines} request={request} rows={rows} moves={} cap=2 state_mismatches={mismatch}",
                        if mismatch == 0 { "PASS" } else { "DEFECT" },
                        updates.len()
                    );
                }
                oracle.policy.end_request()?;
                engine
                    .moe_slots
                    .as_mut()
                    .context("tier request end slots missing")?
                    .end_request()?;
            }
            let mut stats = Trace::default();
            engine.record_test_stats(&mut stats)?;
            let pending = stats.values("stats.copy_wait_pending")?[0];
            ensure!(
                pending > 0.0,
                "Runner FAIL delayed copy control did not execute an outstanding first-read wait"
            );
            println!(
                "PASS decode_residency_coverage variant={variant} pinned={pinned} engines={engines} host_choices={host_choices} disk_choices={disk_choices} admissions={admissions} pending_first_reads={pending} requests=3 end_request=true"
            );
        }
    }
    ensure!(defects == 0, "DEFECT decode residency cells={defects}");
    println!(
        "PASS decode_residency_gate variants={} cases={} host=true disk=true admissions=true writeback=true requests=3 graphs=false",
        variants.len(),
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "authorized GPU window"]
fn decode_residency_smoke_gpu() -> Result<()> {
    tier_gate(&[3], &[(false, 1)])
}

#[test]
#[ignore = "authorized GPU window"]
fn decode_residency_gpu() -> Result<()> {
    tier_gate(&[3, 7], &[(false, 1), (false, 2), (true, 1), (true, 2)])
}

fn cpu_control(
    engine: &mut Engine,
    fixture: &MoeFixture,
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
    let host = HostBlocks::new(16, 24576)?;
    for expert in 0..16 {
        let mut claim = host.claim(expert)?;
        let values = claim.slice()?;
        let expert = expert as usize;
        values[..16384].copy_from_slice(&fixture.gate_up[expert * 16384..(expert + 1) * 16384]);
        values[16384..].copy_from_slice(&fixture.down[expert * 8192..(expert + 1) * 8192]);
    }
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
    let generations = ids
        .iter()
        .map(|&expert| slots.published[expert as usize])
        .collect::<Vec<_>>();
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
        generation: &generations,
        host: Some(host.clone()),
        experts: &experts,
        assignments: &assignments,
        miss_rows: &miss_rows,
        host_dma: None,
        disk: &[],
        disk_cpu: None,
        layer: 0,
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
    decode.finish(
        &engine.device,
        ffn,
        slots.bindings(0, stream)?,
        Dispatch {
            active: &active,
            generation: &generations,
            host: Some(host.clone()),
            experts: &experts,
            assignments: &assignments,
            miss_rows: &miss_rows,
            host_dma: None,
            disk: &[],
            disk_cpu: None,
            layer: 0,
        },
        (
            x,
            &mut TensorMut(qwen::matrix(&engine.work.out, stream, 0, rows, 128)?),
        ),
    )?;
    stream.synchronize()?;
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
    let expected = combine_expected(engine, &contributions, rows)?;
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
        let mut engine = Engine::load_moe_test(&fixture, runtime, 8)?;
        for &rows in row_cases {
            let input = &fixture.x[..rows * 128];
            let reference = direct(&engine, 0, input, rows)?;
            ensure!(
                compare(&reference, &direct(&engine, 0, input, rows)?)? == 0,
                "Runner FAIL same-form reference repeat differs rows={rows} variant={variant}"
            );
            let mut trace = Trace::default();
            engine
                .device
                .stream
                .memcpy_htod(input, &mut engine.work.n.slice_mut(..input.len()))?;
            {
                engine
                    .moe_slots
                    .as_mut()
                    .context("gate slots absent")?
                    .begin_token()?;
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
            engine
                .moe_slots
                .as_mut()
                .context("gate slots lost")?
                .finish_token()?;
            let mismatches = compare(&reference, &trace)?;
            defects += usize::from(mismatches != 0);
            println!(
                "{} eager_moe_all_resident rows={rows} variant={variant} bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
            if cpu {
                defects += cpu_control(&mut engine, &fixture, input, rows, &reference)?;
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
        engine
            .moe_slots
            .as_mut()
            .context("gate slots lost at request end")?
            .end_request()?;
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
                host_dma: None,
                disk: &[],
                disk_cpu: None,
                layer: 0,
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
