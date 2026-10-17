use super::*;
use crate::backend::cuda::moe::reference::{MoeDims, MoeFixture, fixture};
use crate::backend::cuda::weights::{ExpertStorage, Layer};
use crate::experts::plan::Capacity;
use std::collections::BTreeSet;

fn layer_coverage(hidden: usize) -> (usize, BTreeSet<(usize, usize)>) {
    (8, BTreeSet::from([(3 * hidden, hidden), (hidden, hidden)]))
}

fn check_layer_coverage(
    coverage: Option<&(usize, BTreeSet<(usize, usize)>)>,
    hidden: usize,
) -> Result<()> {
    for rows in 1..=8 {
        for output in [3 * hidden, hidden] {
            super::super::super::blas::check_coverage(
                crate::backend::setup::LinearShape {
                    rows,
                    output,
                    input: hidden,
                },
                coverage,
            )?;
        }
    }
    Ok(())
}

impl Engine {
    pub(crate) fn load_moe_test(fixture: &MoeFixture, runtime: &Path) -> Result<Self> {
        let d = fixture.dims;
        ensure!(
            d.hidden == 128 && d.inter == 64 && d.experts == 16 && d.top_k == 4,
            "small MoE hook expects H128/I64/E16/K4, received {d:?}"
        );
        let config = Config {
            model_type: "llama".into(),
            hidden_size: d.hidden,
            intermediate_size: d.inter,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 128,
            vocab_size: 32,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling: None,
            hidden_act: "silu".into(),
            tie_word_embeddings: false,
            torch_dtype: "bfloat16".into(),
            eos_token_id: vec![31],
            attention_bias: false,
            mlp_bias: false,
        };
        let mut model_spec = crate::model_spec::ModelSpec::from_llama(&config)?;
        model_spec.moe = Some(crate::model_spec::MoeConfig {
            experts: 16,
            experts_per_tok: 4,
            expert_intermediate: 64,
            shared_intermediate: 64,
        });
        let geometry = ExpertGeometry::from_model(&model_spec)?;
        let coverage = layer_coverage(config.hidden_size);
        check_layer_coverage(Some(&coverage), config.hidden_size)?;
        let device = Device::new(0)?;
        let stream = &device.stream;
        let bytes = geometry.slot_bytes()?;
        let capacity = Capacity {
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: bytes * 16,
            host_expert_bytes: bytes * 16,
            slots_per_layer: vec![16],
            host_blocks: 16,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
        };
        let storage = ExpertStorage::new(stream, &device.upload, geometry, &capacity)?;
        let mut gu = Vec::new();
        let mut dn = Vec::new();
        let gu_len = 2 * d.inter * d.hidden;
        let dn_len = d.hidden * d.inter;
        for expert in 0..d.experts {
            let (gate_up, down) = storage.slot(0, expert as u32)?;
            // SAFETY: SlotLayout owns the exact native slices through every launch and copy.
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
                .claim(expert as u32)?;
            claim.slice()?[..gu_len]
                .copy_from_slice(&fixture.gate_up[expert * gu_len..(expert + 1) * gu_len]);
            claim.slice()?[gu_len..]
                .copy_from_slice(&fixture.down[expert * dn_len..(expert + 1) * dn_len]);
            gu.push(gate_up);
            dn.push(down);
        }
        let weights = [
            &fixture.router_w,
            &fixture.shared_gate,
            &fixture.shared_up,
            &fixture.shared_down,
            &fixture.shared_expert_gate,
        ]
        .into_iter()
        .map(|w| stream.clone_htod(w))
        .collect::<std::result::Result<Vec<_>, _>>()?;
        let ffn = BoundFfn::Moe {
            router: qwen::matrix(&weights[0], stream, 0, d.experts, d.hidden)?,
            shared: [
                qwen::matrix(&weights[1], stream, 0, d.inter, d.hidden)?,
                qwen::matrix(&weights[2], stream, 0, d.inter, d.hidden)?,
                qwen::matrix(&weights[3], stream, 0, d.hidden, d.inter)?,
            ],
            shared_gate: qwen::view(&weights[4], stream, 0, [d.hidden as u32], [1])?,
        };
        let slots = Slots {
            storage,
            gate_up: stream.clone_htod(&gu)?,
            down: stream.clone_htod(&dn)?,
            valid: stream.clone_htod(&[1u32; 16])?,
            generation: stream.clone_htod(&[1u64; 16])?,
            published: vec![1; 16],
            _weights: weights,
        };
        let weights = Weights {
            embed: stream.alloc_zeros(32 * d.hidden)?,
            norm: stream.clone_htod(&vec![bf16::ONE; d.hidden])?,
            head: stream.alloc_zeros(32 * d.hidden)?,
            bytes: 0,
            layers: vec![Layer {
                input_norm: stream.clone_htod(&vec![bf16::ONE; d.hidden])?,
                qkv: stream.alloc_zeros(3 * d.hidden * d.hidden)?,
                o: stream.alloc_zeros(d.hidden * d.hidden)?,
                post_norm: stream.clone_htod(&vec![bf16::ONE; d.hidden])?,
                gu: stream.alloc_zeros(2 * d.inter * d.hidden)?,
                down: stream.alloc_zeros(d.hidden * d.inter)?,
            }],
        };
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::super::closure::Closure::new(
            true,
            "MoE layer gate".into(),
        ));
        let mut blas = Blas::new(&device, &ops, runtime, closure.clone())?;
        blas.declare(coverage.0, &coverage.1);
        let state = super::super::super::state::StateStore::new(&model_spec, 32, stream)?;
        let mut work = Work::new(stream, &model_spec, 8)?;
        stream.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = Some(Flash::new(stream.clone(), 8, 1)?);
        let moe = Some(Decode::new(&device, geometry, 8, 2)?);
        Ok(Self {
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
        })
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
        let mut trace = Trace::default();
        engine.run_layer_from_reference(0, &fixture.x[..256], [&axes; 3], &mut trace)?;
        ensure!(
            trace.values("layer.ffn")?.len() == 256,
            "layer injection did not execute FFN"
        );
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
fn injected_layer_uses_declared_linear_coverage() -> Result<()> {
    let coverage = layer_coverage(128);
    check_layer_coverage(Some(&coverage), 128)?;
    assert!(check_layer_coverage(None, 128).is_err());
    let mut without_qkv = coverage.clone();
    without_qkv.1.remove(&(384, 128));
    assert!(check_layer_coverage(Some(&without_qkv), 128).is_err());
    let mut too_few_rows = coverage;
    too_few_rows.0 = 1;
    assert!(check_layer_coverage(Some(&too_few_rows), 128).is_err());
    println!(
        "PASS eager_moe_layer_coverage rows=1..8 qkv=[384,128] output=[128,128] undeclared_rejected=true missing_qkv_rejected=true row_bound_rejected=true"
    );
    Ok(())
}
