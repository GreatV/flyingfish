use super::*;
use cudarc::driver::LaunchConfig;
use std::{io::Write, process::Command};

#[test]
#[ignore = "authorized GPU window; each illegal launch runs in a separate process"]
fn q38_batch_meta_launch_contract_gpu() -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
        "batch metadata gate requires an authorized GPU window"
    );
    const TEST: &str =
        "backend::cuda::engine::mtp_launch_tests::q38_batch_meta_launch_contract_gpu";
    const CASES: [&str; 7] = [
        "valid", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
    ];
    let Ok(case) = std::env::var("FF_Q38_BATCH_META_LAUNCH_CASE") else {
        for case in CASES {
            let output = Command::new(std::env::current_exe()?)
                .args([
                    TEST,
                    "--exact",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("FF_Q38_BATCH_META_LAUNCH_CASE", case)
                .output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            ensure!(
                output.status.success() && stdout.contains(&format!("BATCH_META_GATE {case} PASS")),
                "batch metadata case {case} failed: {stdout} {}",
                String::from_utf8_lossy(&output.stderr)
            );
            print!("{stdout}");
        }
        return Ok(());
    };
    ensure!(
        CASES.contains(&case.as_str()),
        "unknown batch metadata gate case"
    );
    let device = Device::new(0)?;
    let stream = &device.stream;
    let function =
        super::super::cubin::module(&device.ctx, "q38_common")?.load_function("q38_batch_meta")?;
    let position = stream.clone_htod(&[37i32])?;
    let mut meta = stream.alloc_zeros::<i32>(8)?;
    let mut launch = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    match case.as_str() {
        "grid_x" => launch.grid_dim.0 = 2,
        "grid_y" => launch.grid_dim.1 = 2,
        "grid_z" => launch.grid_dim.2 = 2,
        "block_x" => launch.block_dim.0 = 2,
        "block_y" => launch.block_dim.1 = 2,
        "block_z" => launch.block_dim.2 = 2,
        "valid" => {}
        _ => unreachable!(),
    }
    for rows in 1..=5i32 {
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&position)
                .arg(&mut meta)
                .arg(&rows)
                .arg(&128i32)
                .launch(launch)?;
        }
        let sync = stream.synchronize();
        if case != "valid" {
            let error = sync.expect_err("illegal launch was not trapped");
            ensure!(
                matches!(
                    error.0,
                    sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
                        | sys::CUresult::CUDA_ERROR_LAUNCH_FAILED
                ),
                "illegal launch returned an unrelated CUDA error: {error:?}"
            );
            println!("BATCH_META_GATE {case} PASS {error:?}");
            std::io::stdout().flush()?;
            // A trapped CUDA context cannot be used again, including by destructors.
            std::process::exit(0);
        }
        sync?;
        ensure!(
            stream.clone_dtoh(&meta)? == [37, rows, 37 + rows, 128, 1, 0, 0, 0],
            "valid batch metadata mismatch"
        );
    }
    println!("BATCH_META_GATE valid PASS rows=1,2,3,4,5");
    Ok(())
}
