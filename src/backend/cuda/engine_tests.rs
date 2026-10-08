use super::*;

#[test]
#[ignore = "requires a reserved GPU window and FF_ARGMAX_MODEL/FF_ARGMAX_DRAFT/FF_ARGMAX_INPUTS/FLYINGFISH_RUNTIME_DIR"]
fn chain_argmax_matches_single_on_real_logits() -> Result<()> {
    let target =
        PathBuf::from(std::env::var("FF_ARGMAX_MODEL").context("FF_ARGMAX_MODEL missing")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT").context("FF_ARGMAX_DRAFT missing")?);
    let inputs =
        PathBuf::from(std::env::var("FF_ARGMAX_INPUTS").context("FF_ARGMAX_INPUTS missing")?);
    let runtime = PathBuf::from(
        std::env::var("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
    );
    println!(
        "argmax_target={} draft={}",
        target.display(),
        draft.display()
    );
    let mut compared = 0;
    let mut rounds = 0;
    for prompt in [
        "p1", "p2", "p3", "p4", "math1", "math2", "math3", "code1", "code2", "code3", "gen1",
        "gen2", "gen3",
    ] {
        #[derive(serde::Deserialize)]
        struct Input {
            input_ids: Vec<u32>,
        }
        let ids: Input =
            serde_json::from_slice(&std::fs::read(inputs.join(format!("{prompt}.json")))?)?;
        let settings = Settings {
            tree_builder: TreeBuilder::Waves,
            spec_budget: SpecBudget::Chain,
            spec_graph: true,
            linear_choices: Vec::new(),
            runtime_dir: Some(runtime.clone()),
        };
        let mut model = Engine::load(
            &target,
            0,
            (ids.input_ids.len() + 512).max(4096),
            None,
            settings,
        )?;
        model.enable_draft(&draft)?;
        let first = model.prefill(&ids.input_ids, None)?;
        let mut actual_ids = vec![first];
        let mut old_ids = vec![first];
        let s = model.device.stream.clone();
        let mut old = s.alloc_zeros::<u32>(8)?;
        let mut generated = 1;
        let first_round = rounds;
        while generated < 256 {
            let round = model.spec_round(256 - generated, None, "")?;
            let spec = model.spec.as_ref().context("missing draft state")?;
            let vocab = model.config.vocab_size;
            ensure!(
                spec.active_rows == 8,
                "chain verification must have eight rows"
            );
            for row in 0..8 {
                unsafe {
                    s.launch_builder(&model.ops.argmax)
                        .arg(&spec.logits.slice(row * vocab..(row + 1) * vocab))
                        .arg(&mut old.slice_mut(row..row + 1))
                        .arg(&(vocab as i32))
                        .launch(grid(1, 1024))?;
                }
            }
            let expected = s.clone_dtoh(&old)?;
            let actual = s.clone_dtoh(&spec.predictions)?;
            ensure!(
                actual == expected,
                "chain argmax differs prompt={prompt} round={rounds}: actual={actual:?} expected={expected:?}"
            );
            let accepted = crate::dspark::accept(&round.proposals, &expected)?;
            let mut old_output = round.proposals[..accepted.proposals].to_vec();
            old_output.push(accepted.bonus);
            old_output.truncate(256 - generated);
            ensure!(
                round.accepted == accepted.proposals && round.output == old_output,
                "chain acceptance/output differs from single-row sampler"
            );
            actual_ids.extend(&round.output);
            old_ids.extend(old_output);
            compared += actual.len();
            generated += round.committed;
            rounds += 1;
        }
        ensure!(
            generated == 256 && actual_ids == old_ids,
            "chain output count mismatch"
        );
        println!(
            "{}",
            serde_json::json!({"chain_argmax_equivalence":{"prompt":prompt,"output_tokens":generated,"rounds":rounds-first_round,"prediction_bits_equal":true,"output_ids_equal":true,"actual_ids":actual_ids,"single_sampler_ids":old_ids,"scope":"same actual BF16 logits; batched production predictions vs eight original single-row launches; reconstruct original sampler acceptance and committed outputs"}})
        );
    }
    println!(
        "{}",
        serde_json::json!({"chain_argmax_equivalence_summary":{"prompts":13,"rounds":rounds,"predictions_compared":compared,"pass":true}})
    );
    Ok(())
}
