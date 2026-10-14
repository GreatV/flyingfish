use anyhow::{Context, Result, ensure};
use flyingfish::{
    backend::{Settings, SpecBudget, TreeBuilder},
    model::Model,
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

fn settings(runtime: &Path) -> Settings {
    Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        runtime_dir: Some(runtime.to_owned()),
        prefix_cache: false,
    }
}
fn save(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn row_receipt(i: usize, prediction: u32, forced: u32, maximum: f32, elapsed: f64) -> Value {
    json!({"i":i,"prediction":prediction,"forced_history_token":forced,
        "absolute_logit_row":16319+i,"maximum_logit":maximum,"elapsed_s":elapsed})
}
fn run(root: &Path, mode: &str, model: &Path) -> Result<()> {
    let runtime = root.join("runtime");
    if mode == "calibrate" {
        Model::calibrate(model, 0, 16640, Some(2048), settings(&runtime))?;
        return Ok(());
    }
    let start = Instant::now();
    let mut engine = Model::load(model, 0, 16640, Some(2048), settings(&runtime))?;
    engine.check_ready()?;
    let load_s = start.elapsed().as_secs_f64();
    let paths: Vec<Value> = serde_json::from_slice(&fs::read(root.join("PATHS.json"))?)?;
    ensure!(paths.len() == 5, "five frozen histories required");
    let mut completed = Vec::new();
    for path in paths {
        let name = path["id"].as_str().context("id")?;
        if mode == "smoke" && !completed.is_empty() {
            break;
        }
        let input: Value = serde_json::from_slice(&fs::read(
            root.join(path["input_file"].as_str().context("input")?),
        )?)?;
        let mut ids: Vec<u32> = serde_json::from_value(input["input_ids"].clone())?;
        let forced: Vec<u32> = serde_json::from_value(path["tokens"].clone())?;
        ensure!(ids.len() == 16320 && forced.len() == 64, "frozen extent");
        let rows = if mode == "smoke" {
            ids.truncate(32);
            2
        } else {
            64
        };
        engine.reset()?;
        let wall = Instant::now();
        engine.prefill(&ids, None)?;
        let output = root.join("native");
        fs::create_dir_all(&output)?;
        let mut logits_file = if mode == "formal" {
            Some(BufWriter::new(File::create(
                output.join(format!("{name}.f32le")),
            )?))
        } else {
            None
        };
        let mut positions = Vec::new();
        for (i, &history_token) in forced.iter().take(rows).enumerate() {
            let prediction = engine.token()?;
            let logits = engine.decode_logits()?;
            ensure!(
                logits.len() == 248320 && logits.iter().all(|x| x.is_finite()),
                "nonfinite/invalid native logits"
            );
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            ensure!(
                logits[prediction as usize] == maximum,
                "public prediction/logits alignment"
            );
            if let Some(file) = logits_file.as_mut() {
                let mut bytes = Vec::with_capacity(logits.len() * 4);
                for v in logits {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                file.write_all(&bytes)?;
            }
            positions.push(row_receipt(
                i,
                prediction,
                history_token,
                maximum,
                wall.elapsed().as_secs_f64(),
            ));
            if i + 1 < rows {
                // Read row zero before first capture; capture restores persistent state.
                if i == 0 {
                    engine.capture()?;
                }
                engine.set_token(history_token)?;
                engine.step(true)?;
            }
        }
        if let Some(file) = logits_file.as_mut() {
            file.flush()?;
        }
        engine.synchronize()?;
        let record = json!({"path":path,"positions":positions,"elapsed_s":wall.elapsed().as_secs_f64(),
            "logits_shape":[rows,248320],"stored_dtype":"FP32 little-endian; public native BF16 widened",
            "input_tokens":ids.len(),"decode_graph":true,"teacher_forcing":"row i predicts after prompt + frozen tokens[..i]"});
        if mode == "formal" {
            save(&output.join(format!("{name}.json")), &record)?;
        }
        completed.push(record);
    }
    let result = if mode == "smoke" {
        json!({"status":"PASS","mode":"smoke","real_short_requests":1,"rows":2,"output_discarded":true})
    } else {
        json!({"status":"PASS","mode":"formal","paths":completed,"load_s":load_s,
            "capacity":16640,"chunk":engine.chunk(),"decode_linear":engine.decode_linear(),
            "device":engine.device_info(),"mtp":"plain teacher-forced public API; no speculative generation"})
    };
    save(
        &root.join(if mode == "smoke" {
            "SMOKE.json"
        } else {
            "NATIVE_RESULT.json"
        }),
        &result,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    ensure!(a.len() == 3, "MODE ROOT MODEL");
    let root = PathBuf::from(&a[1]);
    fs::create_dir_all(&root)?;
    if a[0] == "receipt-smoke" {
        // Exercise every native JSON writer before START, with representative nested rows.
        let row = row_receipt(23, 74455, 198, 13.5625, 0.123);
        for (name, value) in [
            (
                "CPU_PATH.json",
                json!({"path":{"id":"path-0","tokens":[198],"records":[{"engine":"vllm"}]},"positions":[row]}),
            ),
            (
                "CPU_SMOKE.json",
                json!({"status":"PASS","output_discarded":true}),
            ),
            (
                "CPU_NATIVE_RESULT.json",
                json!({"status":"PASS","paths":[],"load_s":0.0,"capacity":16640,"chunk":2048}),
            ),
            (
                "CPU_NATIVE_FAIL.json",
                json!({"status":"FAIL","error":"representative resource/finite failure"}),
            ),
        ] {
            let p = root.join(name);
            save(&p, &value)?;
            ensure!(
                serde_json::from_slice::<Value>(&fs::read(&p)?)? == value,
                "receipt roundtrip"
            );
            fs::remove_file(p)?;
        }
        println!("CPU_NATIVE_RECEIPTS_READY");
        return Ok(());
    }
    ensure!(
        std::env::var("FF_NATIVE16K_WINDOW").as_deref() == Ok("granted"),
        "window required"
    );
    ensure!(
        ["calibrate", "smoke", "formal"].contains(&a[0].as_str()),
        "mode"
    );
    if let Err(error) = run(&root, &a[0], Path::new(&a[2])) {
        save(
            &root.join("NATIVE_FAIL.json"),
            &json!({"status":"FAIL","mode":a[0],"error":format!("{error:#}")}),
        )?;
        return Err(error);
    }
    Ok(())
}
