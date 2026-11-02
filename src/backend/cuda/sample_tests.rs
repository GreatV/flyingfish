use super::*;
use crate::backend::cuda::moe::reference::CloverMoeFixture;
use anyhow::Context;
use serde_json::Value;

pub(crate) fn integer_values(fixture: &CloverMoeFixture, name: &str) -> Result<Vec<i32>> {
    Ok(fixture.values(name)?.iter().map(|&v| v as i32).collect())
}

pub(crate) fn histogram(fixture: &CloverMoeFixture, tag: &str, vocab: usize) -> Result<Vec<u32>> {
    let mut values = vec![0; vocab];
    let ids = integer_values(fixture, &format!("{tag}.ids"))?;
    let counts = integer_values(fixture, &format!("{tag}.values"))?;
    ensure!(ids.len() == counts.len(), "Runner FAIL histogram extent");
    for (id, count) in ids.into_iter().zip(counts) {
        ensure!(
            id >= 0 && (id as usize) < vocab && count >= 0,
            "Runner FAIL histogram bounds"
        );
        values[id as usize] = count as u32;
    }
    Ok(values)
}

fn score<T: PartialEq>(tag: &str, metric: &str, actual: &[T], expected: &[T]) -> Result<usize> {
    ensure!(
        actual.len() == expected.len(),
        "Runner FAIL {tag} {metric} extent"
    );
    let mismatches = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    println!(
        "CLOVER_SAMPLE {} tag={tag} metric={metric} mismatches={mismatches} bound=0",
        if mismatches == 0 { "PASS" } else { "DEFECT" }
    );
    Ok(usize::from(mismatches != 0))
}

pub(crate) fn restore(
    sampler: &mut BlockSampler,
    fixture: &CloverMoeFixture,
    case: &Value,
    control: &Value,
) -> Result<()> {
    let tag = case["tag"].as_str().context("Runner FAIL penalty tag")?;
    let mut counts = histogram(
        fixture,
        &format!("{tag}.counts"),
        sampler.policy.vocab as usize,
    )?;
    if let Some(id) = control["count_id"].as_u64() {
        counts[id as usize] += 1;
    }
    if control["clear_close"] == true {
        counts[sampler.policy.close_token as usize] = 0;
    }
    let mut block = integer_values(fixture, &format!("{tag}.block"))?;
    if control["visible"] == false {
        for token in &mut block {
            if *token == sampler.policy.close_token as i32 {
                *token = sampler.policy.mask as i32;
            }
        }
    }
    let pending: Vec<_> = integer_values(fixture, &format!("{tag}.pending"))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    let state = [
        case["length"]
            .as_u64()
            .context("Runner FAIL response length")? as u32,
        0,
        0,
        pending.iter().map(|&v| u32::from(v)).sum(),
    ];
    sampler.stream.memcpy_htod(&counts, &mut sampler.counts)?;
    sampler.stream.memcpy_htod(&block, &mut sampler.block)?;
    sampler.stream.memcpy_htod(&pending, &mut sampler.pending)?;
    sampler.stream.memcpy_htod(&state, &mut sampler.state)?;
    Ok(())
}

pub(crate) fn expected_fill(fixture: &CloverMoeFixture, tag: &str) -> Result<(Vec<i32>, Vec<u8>)> {
    let confidence = fixture.values(&format!("{tag}.confidence"))?;
    let picks = integer_values(fixture, &format!("{tag}.picks"))?;
    let mut block = integer_values(fixture, &format!("{tag}.block"))?;
    let mut pending: Vec<_> = integer_values(fixture, &format!("{tag}.pending"))?
        .iter()
        .map(|&v| v as u8)
        .collect();
    let live: Vec<_> = (0..pending.len()).filter(|&r| pending[r] != 0).collect();
    let mut selected: Vec<_> = live
        .iter()
        .copied()
        .filter(|&r| confidence[r] > 0.95)
        .collect();
    if selected.is_empty() {
        let mut best = *live
            .first()
            .context("Runner FAIL no pending reference row")?;
        for &r in &live {
            if confidence[r] > confidence[best] {
                best = r;
            }
        }
        selected.push(best);
    }
    for r in selected {
        block[r] = picks[r];
        pending[r] = 0;
    }
    Ok((block, pending))
}

fn penalty_cells(device: &Device, fixture: &CloverMoeFixture) -> Result<(usize, usize)> {
    let root = serde_json::from_str(include_str!("../../../tests/fixtures/clover_config.json"))?;
    let policy = BlockGeneration::clover(&root, &fixture.table["generation"])?;
    let raw = fixture.values("raw")?;
    let s = &device.stream;
    let mut sampler = BlockSampler::prepare(device, policy.clone())?;
    let mut logits = s.clone_htod(&raw)?;
    let (mut defects, mut controls) = (0, 0);
    for case in fixture.table["penalty_cases"]
        .as_array()
        .context("Runner FAIL penalty cases")?
    {
        let tag = case["tag"].as_str().unwrap();
        let mut expected = raw.clone();
        for (index, value) in integer_values(fixture, &format!("{tag}.patch_ids"))?
            .into_iter()
            .zip(fixture.values(&format!("{tag}.patch_values"))?)
        {
            ensure!(
                index >= 0 && (index as usize) < expected.len(),
                "Runner FAIL patch index"
            );
            expected[index as usize] = value;
        }
        let expected_bits: Vec<_> = expected.iter().map(|v| v.to_bits()).collect();
        let picks = integer_values(fixture, &format!("{tag}.picks"))?;
        let (block, pending) = expected_fill(fixture, tag)?;
        let counts = histogram(fixture, &format!("{tag}.counts"), policy.vocab as usize)?;
        for graph_mode in [false, true] {
            restore(&mut sampler, fixture, case, &Value::Null)?;
            s.memcpy_htod(&raw, &mut logits)?;
            if graph_mode {
                s.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                sampler.step(&mut logits)?;
                let graph = s
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("Runner FAIL empty sampler graph")?;
                graph.launch()?;
            } else {
                sampler.step(&mut logits)?;
            }
            let actual: Vec<_> = s.clone_dtoh(&logits)?.iter().map(|v| v.to_bits()).collect();
            let name = format!("{tag}.graph{graph_mode}");
            defects += score(&name, "logit_bits", &actual, &expected_bits)?;
            defects += score(&name, "picks", &s.clone_dtoh(&sampler.picks)?, &picks)?;
            defects += score(&name, "block", &s.clone_dtoh(&sampler.block)?, &block)?;
            defects += score(&name, "pending", &s.clone_dtoh(&sampler.pending)?, &pending)?;
            defects += score(
                &name,
                "history_unchanged",
                &s.clone_dtoh(&sampler.counts)?,
                &counts,
            )?;
            let probabilities = s.clone_dtoh(&sampler.probabilities)?;
            let reference = fixture.values(&format!("{tag}.confidence"))?;
            let maximum = picks
                .iter()
                .enumerate()
                .map(|(r, &id)| {
                    (probabilities[r * policy.vocab as usize + id as usize] - reference[r]).abs()
                })
                .fold(0.0f32, f32::max);
            println!(
                "CLOVER_SAMPLE_MARGIN tag={name} confidence_max_abs={maximum} reference_margin={} top2_margin={}",
                case["confidence_margin"], case["top2_margin"]
            );
        }
        for control in case["controls"].as_array().unwrap() {
            let mut generation = fixture.table["generation"].clone();
            if let Some(field) = control["field"].as_str() {
                generation[field] = control["value"].clone();
            }
            let mut altered = BlockGeneration::clover(&root, &generation)?;
            if let Some(mask) = control["mask"].as_u64() {
                altered.mask = mask as u32;
            }
            let mut mutant = BlockSampler::prepare(device, altered)?;
            restore(&mut mutant, fixture, case, control)?;
            s.memcpy_htod(&raw, &mut logits)?;
            mutant.apply_history(&mut logits)?;
            let mismatches = s
                .clone_dtoh(&logits)?
                .iter()
                .zip(&expected_bits)
                .filter(|(a, b)| a.to_bits() != **b)
                .count();
            println!(
                "CLOVER_SAMPLE_CONTROL tag={tag} name={} mismatches={mismatches} detected={}",
                control["name"],
                mismatches > 0
            );
            defects += usize::from(mismatches == 0);
            controls += 1;
        }
    }
    Ok((defects, controls))
}

fn policy_cells(device: &Device, fixture: &CloverMoeFixture) -> Result<(usize, usize)> {
    let root = serde_json::from_str(include_str!("../../../tests/fixtures/clover_config.json"))?;
    let policy = BlockGeneration::clover(&root, &fixture.table["generation"])?;
    let mut sampler = BlockSampler::prepare(device, policy.clone())?;
    let s = &device.stream;
    let (mut defects, mut controls) = (0, 0);
    for case in fixture.table["policy_cases"]
        .as_array()
        .context("Runner FAIL policy cases")?
    {
        let tag = case["tag"].as_str().unwrap();
        sampler.reset()?;
        let mut committed = vec![0u32; policy.vocab as usize];
        let mut returned = Vec::new();
        let mut begin = false;
        for event in case["events"].as_array().unwrap() {
            let key = event["key"].as_str().unwrap();
            if event["kind"] == "fill" {
                if begin {
                    sampler.begin_block()?;
                    begin = false;
                }
                let picks = integer_values(fixture, &format!("{key}.picks"))?;
                let confidence = fixture.values(&format!("{key}.confidence"))?;
                s.memcpy_htod(&picks, &mut sampler.picks)?;
                for (r, (&pick, &value)) in picks.iter().zip(&confidence).enumerate() {
                    let index = r * policy.vocab as usize + pick as usize;
                    s.memcpy_htod(
                        &[value],
                        &mut sampler.probabilities.slice_mut(index..index + 1),
                    )?;
                }
                sampler.fill()?;
                let expected = integer_values(fixture, &format!("{key}.block"))?;
                let pending: Vec<_> = integer_values(fixture, &format!("{key}.pending"))?
                    .iter()
                    .map(|&v| v as u8)
                    .collect();
                let state = [
                    event["length"].as_u64().unwrap() as u32,
                    0,
                    0,
                    pending.iter().map(|&v| u32::from(v)).sum(),
                ];
                defects += score(key, "block", &s.clone_dtoh(&sampler.block)?, &expected)?;
                defects += score(key, "pending", &s.clone_dtoh(&sampler.pending)?, &pending)?;
                defects += score(key, "state", &s.clone_dtoh(&sampler.state)?, &state)?;
                defects += score(
                    key,
                    "no_early_counts",
                    &s.clone_dtoh(&sampler.counts)?,
                    &committed,
                )?;
                if tag == "early_eos" && key.ends_with("step1") {
                    let premature = expected.iter().any(|&id| policy.eos.contains(&(id as u32)))
                        && state[3] > 0;
                    println!("CLOVER_SAMPLE_CONTROL name=premature_eos detected={premature}");
                    defects += usize::from(!premature);
                    controls += 1;
                }
                if tag == "force_tie" && key.ends_with("step1") {
                    let detected = pending[1] == 0 && pending[2] == 1 && pending[3] == 1;
                    println!(
                        "CLOVER_SAMPLE_CONTROL name=no_progress_or_last_tie detected={detected}"
                    );
                    defects += usize::from(!detected);
                    controls += 1;
                }
                if tag == "strict" && key.ends_with("step1") {
                    let detected = pending[0] == 1 && pending[1] == 0 && pending[2] == 1;
                    println!("CLOVER_SAMPLE_CONTROL name=threshold_ge detected={detected}");
                    defects += usize::from(!detected);
                    controls += 1;
                }
            } else {
                ensure!(event["kind"] == "commit", "Runner FAIL policy event kind");
                sampler.commit(event["remaining"].as_u64().unwrap() as u32)?;
                committed = histogram(fixture, &format!("{key}.counts"), policy.vocab as usize)?;
                let state = [
                    event["length"].as_u64().unwrap() as u32,
                    event["returned"].as_u64().unwrap() as u32,
                    u32::from(event["stopped"] == true),
                    0,
                ];
                defects += score(key, "state", &s.clone_dtoh(&sampler.state)?, &state)?;
                defects += score(key, "counts", &s.clone_dtoh(&sampler.counts)?, &committed)?;
                returned.extend_from_slice(&s.clone_dtoh(&sampler.block)?[..state[1] as usize]);
                if tag == "limit1" {
                    let detected = committed.iter().sum::<u32>() == 1 && state[1] != policy.rows;
                    println!("CLOVER_SAMPLE_CONTROL name=count_entire_preview detected={detected}");
                    defects += usize::from(!detected);
                    controls += 1;
                }
                begin = true;
            }
        }
        defects += score(
            tag,
            "returned",
            &returned,
            &integer_values(fixture, &format!("{tag}.returned"))?,
        )?;
    }
    Ok((defects, controls))
}

#[test]
#[ignore = "GPU window; frozen publisher sampling fixtures required"]
fn clover_publisher_sampling_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let fixture = CloverMoeFixture::open_named("sampling.safetensors")?;
    ensure!(
        fixture.table["bounds"]
            == serde_json::json!({"logit_bits":0,"ids":0,"pending":0,"counts":0,"returned":0}),
        "Runner FAIL exact sampling bounds changed"
    );
    let device = Device::new(0)?;
    device.ctx.synchronize()?;
    // All sampler fixture allocations and launches use device.stream.
    unsafe { device.ctx.disable_event_tracking() };
    use cudarc::driver::sys::CUfunction_attribute as F;
    let module = super::super::cubin::module(&device.ctx, "sample")?;
    for (name, threads) in [
        ("token_history_bias", 256),
        ("confidence_threshold_fill", 32),
        ("response_count_commit", 32),
    ] {
        let function = module.load_function(name)?;
        println!(
            "SAMPLE_RESOURCE kernel={name} registers={} local_bytes={} shared_bytes={} resident_blocks={}",
            function.get_attribute(F::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            function.get_attribute(F::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            function.get_attribute(F::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
            function.occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)?
        );
    }
    let (a, ca) = penalty_cells(&device, &fixture)?;
    let (b, cb) = policy_cells(&device, &fixture)?;
    let defects = a + b;
    println!(
        "CLOVER_SAMPLE {} penalty_cases=10 eager_graph=2 policy_cases=7 controls={} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" },
        ca + cb
    );
    ensure!(defects == 0, "Clover publisher sampling defects={defects}");
    Ok(())
}
