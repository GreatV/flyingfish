//! CLI gate for the to-3d Pixal3D class: the binary's output equals the
//! library entry's on the same inputs and seed, and a different seed must
//! change the mesh.
#[cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::process::Command;

#[cfg(feature = "cuda")]
fn env_path(name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(
        std::env::var_os(name).with_context(|| format!("{name} required"))?,
    ))
}

#[cfg(feature = "cuda")]
fn run_binary(args: &[&str]) -> Result<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_flyingfish"))
        .args(args)
        .output()
        .context("spawn flyingfish")?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    ensure!(
        output.status.success(),
        "flyingfish {:?} failed: {}{}",
        args,
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(stdout)
}

#[cfg(feature = "cuda")]
fn read_mesh(path: &Path) -> Result<flyingfish::media::MeshRead> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_owned();
    flyingfish::media::read_mesh(&std::fs::read(path)?, &extension)
}

#[cfg(feature = "cuda")]
fn assert_mesh_equal(
    label: &str,
    actual: &flyingfish::media::MeshRead,
    expected: &flyingfish::sparse::SparseMesh,
) -> Result<()> {
    ensure!(
        actual.positions == expected.vertices,
        "{label}: {} written positions differ from the library entry's {}",
        actual.positions.len(),
        expected.vertices.len()
    );
    ensure!(
        actual.triangles == expected.faces,
        "{label}: {} written faces differ from the library entry's {}",
        actual.triangles.len(),
        expected.faces.len()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU + published models; set FF_PIXAL_MODEL, FF_DINOV3_MODEL, FF_NAF_MODEL, FF_PIXAL_FIXTURES (dir with cottage.png and cottage-camera.json)"]
fn to3d_pixal_cli_equals_the_library_entry() -> Result<()> {
    let model = env_path("FF_PIXAL_MODEL")?;
    let dino = env_path("FF_DINOV3_MODEL")?;
    let naf = env_path("FF_NAF_MODEL")?;
    let fixtures = env_path("FF_PIXAL_FIXTURES")?;
    let image = fixtures.join("cottage.png");
    let camera_json = fixtures.join("cottage-camera.json");
    let runtime = std::env::temp_dir().join(format!("ff-to3d-pixal-{}", std::process::id()));
    std::fs::create_dir_all(&runtime)?;
    let out = runtime.join("cottage-seed42.ply");
    let stdout = run_binary(&[
        "--model",
        model.to_str().context("model path")?,
        "--runtime-dir",
        runtime.to_str().context("runtime path")?,
        "to-3d",
        image.to_str().context("image path")?,
        "--dino",
        dino.to_str().context("dino path")?,
        "--naf",
        naf.to_str().context("naf path")?,
        "--camera",
        camera_json.to_str().context("camera path")?,
        "--resolution",
        "1024",
        "--seed",
        "42",
        "-o",
        out.to_str().context("output path")?,
    ])?;
    ensure!(
        stdout.contains("resolution 1024"),
        "CLI printed {stdout:?} without the requested resolution"
    );
    let cli_mesh = read_mesh(&out)?;

    // Library arm: the same construction the command performs.
    let (rgba, width, height) =
        flyingfish::serve::image::decode_image_rgba(&std::fs::read(&image)?, None)?;
    let camera: serde_json::Value = serde_json::from_slice(&std::fs::read(&camera_json)?)?;
    let distance = camera["distance"].as_f64().context("distance")? as f32;
    let views = [flyingfish::backend::PixalView {
        rgba: &rgba,
        dimensions: [height, width],
        camera: flyingfish::backend::ProjectionCamera {
            world_to_camera: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, distance,
            ],
            horizontal_fov: camera["camera_angle_x"].as_f64().context("fov")? as f32,
            mesh_scale: camera["mesh_scale"].as_f64().unwrap_or(1.0) as f32,
        },
    }];
    let device = flyingfish::backend::BackendDevice::new(0)?;
    let output = flyingfish::backend::PixalSeededRequest {
        model: &model,
        dino: &dino,
        naf: &naf,
        runtime: &runtime,
        views: &views,
        multiview: false,
        requested_resolution: 1024,
    }
    .run(42, &device)?;
    ensure!(
        output.resolution == 1024,
        "library resolution {} vs requested 1024",
        output.resolution
    );
    assert_mesh_equal("pixal seed 42", &cli_mesh, &output.mesh)?;

    // Control: a different seed must change the mesh.
    let control_out = runtime.join("cottage-seed43.ply");
    run_binary(&[
        "--model",
        model.to_str().context("model path")?,
        "--runtime-dir",
        runtime.to_str().context("runtime path")?,
        "to-3d",
        image.to_str().context("image path")?,
        "--dino",
        dino.to_str().context("dino path")?,
        "--naf",
        naf.to_str().context("naf path")?,
        "--camera",
        camera_json.to_str().context("camera path")?,
        "--resolution",
        "1024",
        "--seed",
        "43",
        "-o",
        control_out.to_str().context("output path")?,
    ])?;
    let control_mesh = read_mesh(&control_out)?;
    ensure!(
        control_mesh.positions != cli_mesh.positions
            || control_mesh.triangles != cli_mesh.triangles,
        "seed 43 reproduced the seed 42 mesh"
    );
    std::fs::remove_dir_all(&runtime).ok();
    println!(
        "PASS to3d_pixal_cli vertices={} faces={} seed_control=different",
        cli_mesh.positions.len() / 3,
        cli_mesh.triangles.len() / 3
    );
    Ok(())
}
