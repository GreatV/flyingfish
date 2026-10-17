//! GDN chunkwise kernel gate runner over a SHA-verified `--package`: numerics against the FP64 oracle, NULL-band ABBA timing, bit-identity and end-to-end engine timing; `--stub` runs without a CUDA device.
//! `--route p2` runs the chunkwise-P2 split (one 32-column strip, `--s` ignored) and writes the same dump file names as `--route p1`.
use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SCHEMA_MANIFEST: &str = "gdn-p1-frozen-inputs/1";
const SCHEMA_BASIS: &str = "gdn-p1-gate-basis/3";
const SCHEMA_GATE: &str = "gdn-p1-gate-report/1";
const SCHEMA_TIMING: &str = "gdn-p1-timing-report/1";
#[cfg(feature = "cuda")]
const SCHEMA_IDENTITY: &str = "gdn-p1-identity-report/1";

/// Chunk size the P1 core is compiled with (P1_C).
const KERNEL_C: u32 = 64;
/// Paired ABBA blocks per timing configuration.
const TIMING_BLOCKS: u32 = 12;
/// Minimum identity launches per row.
const IDENTITY_MIN_RUNS: u32 = 20;
/// ULP floor margin below the model's frac_le_1ulp.
const FRAC_FLOOR_MARGIN: f64 = 0.02;
/// Decode steps the e2e phase runs and discards before it resets.
const E2E_WARMUP_STEPS: usize = 4;

#[cfg(feature = "cuda")]
const P1_MODULE: &str = "gdn/gdn_chunkwise_p1";
#[cfg(feature = "cuda")]
const P2_MODULE: &str = "gdn/gdn_cw_p2";
/// `gdn_prep_fp32`, the P1 reference arm's widen.
#[cfg(feature = "cuda")]
const PREP_MODULE: &str = "gdn/gdn_prep";

#[derive(Parser)]
#[command(about = "GDN chunkwise P1 primitive gate runner", long_about = None)]
struct Cli {
    #[command(subcommand)]
    phase: Phase,
}

impl Cli {
    /// The route the parsed phase will run under.
    fn phase_route(&self) -> Route {
        match &self.phase {
            Phase::Numerics(a) => a.route,
            Phase::Timing(a) => a.route,
            Phase::Identity(a) => a.route,
            Phase::E2e(_) => Route::P1,
        }
    }

    /// Argument checks that need no device; runs before any CUDA query.
    fn validate(&self) -> Result<()> {
        match &self.phase {
            Phase::Numerics(a) => {
                ensure!(a.c == KERNEL_C, "--c {} is not the built C{KERNEL_C}", a.c);
                for s in route_splits(a.route, &a.s) {
                    ensure!(
                        [4u32, 8, 16].contains(&s),
                        "--s {s} is not one of 4/8/16 (protocol section A)"
                    );
                }
                Ok(())
            }
            Phase::Timing(a) => {
                ensure!(a.c == KERNEL_C, "--c {} is not the built C{KERNEL_C}", a.c);
                ensure!(
                    a.blocks == TIMING_BLOCKS,
                    "--blocks {} is not the registered {TIMING_BLOCKS}",
                    a.blocks
                );
                for s in route_splits(a.route, &a.s) {
                    ensure!(
                        [4u32, 8, 16].contains(&s),
                        "--s {s} is not one of 4/8/16 (protocol section A)"
                    );
                }
                Ok(())
            }
            Phase::Identity(a) => {
                ensure!(a.c == KERNEL_C, "--c {} is not the built C{KERNEL_C}", a.c);
                ensure!(
                    a.runs >= IDENTITY_MIN_RUNS,
                    "--runs {} is below the identity contract's {IDENTITY_MIN_RUNS}",
                    a.runs
                );
                for s in route_splits(a.route, &a.s) {
                    ensure!(
                        [4u32, 8, 16].contains(&s),
                        "--s {s} is not one of 4/8/16 (protocol section A)"
                    );
                }
                Ok(())
            }
            Phase::E2e(a) => {
                ensure!(a.prefill_reps > 0, "--prefill-reps must be positive");
                ensure!(a.decode_steps > 0, "--decode-steps must be positive");
                Ok(())
            }
        }
    }
}

/// Parses and validates `argv` (program name first) before any device query.
fn parse_and_validate<I, T>(argv: I) -> Result<Cli>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::try_parse_from(argv).map_err(|e| anyhow::anyhow!("{e}"))?;
    cli.validate()?;
    Ok(cli)
}

#[derive(Subcommand)]
enum Phase {
    /// Run the kernel per fixture and emit the primitive numerics gate report.
    Numerics(NumericsArgs),
    /// Freeze the NULL band, then measure P1 against the main core.
    Timing(TimingArgs),
    /// Launch one P1 entry >=N times from identical device inputs and byte-compare out and state.
    Identity(IdentityArgs),
    /// End-to-end engine timing: CUDA events around prefill and decode only.
    E2e(E2eArgs),
}

#[derive(Parser)]
struct IdentityArgs {
    /// Frozen P1 package directory (MANIFEST.json, inputs/, truth/, gate_basis.json).
    #[arg(long)]
    package: PathBuf,
    /// Run directory for the identity report.
    #[arg(long)]
    run_dir: PathBuf,
    /// Chunk size; the built P1 core is C64 only.
    #[arg(long, default_value_t = KERNEL_C)]
    c: u32,
    /// Value splits.
    #[arg(long, value_delimiter = ',', default_value = "8,4,16")]
    s: Vec<u32>,
    /// Which core to launch: the fused P1 core, or the chunkwise-P2 split.
    #[arg(long, value_enum, default_value = "p1")]
    route: Route,
    /// Launches per (fixture, split), each from a fresh S0; at least 20.
    #[arg(long, default_value_t = 20)]
    runs: u32,
    /// Only these fixtures (default: every fixture in the manifest).
    #[arg(long, value_delimiter = ',')]
    case: Vec<String>,
}

#[derive(Parser)]
struct E2eArgs {
    /// Model directory.
    #[arg(long)]
    model: PathBuf,
    /// Runtime directory for this process.
    #[arg(long)]
    runtime_dir: PathBuf,
    /// JSON array of token ids to prefill.
    #[arg(long)]
    tokens_file: PathBuf,
    /// Output record path.
    #[arg(long)]
    output: PathBuf,
    /// Arm label written into the record.
    #[arg(long)]
    label: String,
    #[arg(long, default_value_t = 20000)]
    capacity: usize,
    /// Measured prefill repetitions, each from a reset.
    #[arg(long, default_value_t = 4)]
    prefill_reps: usize,
    /// Decode steps measured after each prefill repetition.
    #[arg(long, default_value_t = 16)]
    decode_steps: usize,
    /// Prefill chunk size in rows; None uses the planner's admitted chunk.
    #[arg(long)]
    chunk: Option<usize>,
}

#[cfg(feature = "cuda")]
fn run_e2e(a: &E2eArgs) -> Result<()> {
    use flyingfish::model::Model;
    use std::time::Instant;
    let ids = read_e2e_ids(a)?;
    let settings = flyingfish::backend::Settings {
        tree_builder: flyingfish::backend::TreeBuilder::Waves,
        spec_budget: flyingfish::backend::SpecBudget::Chain,
        draft_model: None,
        spec_graph: true,
        runtime_dir: Some(a.runtime_dir.clone()),
        prefix_cache: false,
        kv_format: flyingfish::kv::KvFormat::Bf16,
    };
    let started = Instant::now();
    let mut model = Model::load(&a.model, 0, a.capacity, a.chunk, settings)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    // One discarded prefill+decode, then reset.
    model.reset()?;
    let _ = model.prefill(&ids, None)?;
    for _ in 0..E2E_WARMUP_STEPS {
        model.step(false)?;
    }
    model.reset()?;
    let mut prefill_gpu_ms = Vec::new();
    let mut decode_gpu_ms = Vec::new();
    let mut peak_vram_bytes = 0u64;
    for _ in 0..a.prefill_reps {
        model.reset()?;
        let begin = model.event()?;
        let end = model.event()?;
        model.record(&begin)?;
        model.prefill(&ids, None)?;
        model.record(&end)?;
        end.synchronize()?;
        prefill_gpu_ms.push(begin.elapsed_ms(&end)? as f64);
        peak_vram_bytes = model.live_device_bytes()?;
        let dbegin = model.event()?;
        let dend = model.event()?;
        model.record(&dbegin)?;
        for _ in 0..a.decode_steps {
            model.step(false)?;
        }
        model.record(&dend)?;
        dend.synchronize()?;
        decode_gpu_ms.push(dbegin.elapsed_ms(&dend)? as f64 / a.decode_steps as f64);
    }
    let out = serde_json::json!({
        "label": a.label, "tokens": ids.len(), "load_ms": load_ms,
        "prefill_gpu_ms": prefill_gpu_ms, "decode_step_gpu_ms": decode_gpu_ms,
        "decode_steps": a.decode_steps, "capacity": a.capacity, "chunk": a.chunk,
        "admitted_chunk": model.chunk(), "peak_vram_bytes": peak_vram_bytes,
    });
    std::fs::write(&a.output, serde_json::to_vec_pretty(&out)?)?;
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

/// The e2e phase's token file and the capacity it must fit, checked on both legs.
fn read_e2e_ids(a: &E2eArgs) -> Result<Vec<u32>> {
    ensure!(a.prefill_reps > 0 && a.decode_steps > 0, "positive reps");
    let ids: Vec<u32> =
        serde_json::from_slice(&std::fs::read(&a.tokens_file)?).context("tokens file")?;
    let need = e2e_positions(ids.len(), a.decode_steps)?;
    ensure!(
        !ids.is_empty() && need <= a.capacity,
        "{need} positions ({} tokens + {} steps, the warm-up's {} included) against \
         capacity {}",
        ids.len(),
        a.decode_steps,
        E2E_WARMUP_STEPS,
        a.capacity
    );
    Ok(ids)
}

#[cfg(not(feature = "cuda"))]
fn run_e2e(a: &E2eArgs) -> Result<()> {
    read_e2e_ids(a)?;
    anyhow::bail!("the e2e phase needs the crate's cuda feature")
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum Clock {
    /// CUDA events around the launch.
    Cuda,
    /// Deterministic per-launch durations; no CUDA device is touched.
    Stub,
}

/// Core the phase launches: the fused P1 core per value split, or the chunkwise-P2 split (stage A, stage H) over a workspace slot.
/// P2 has one 32-column strip, reported under the `s4` tag; `--s` is ignored under p2.
#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
enum Route {
    /// The fused `gdn_chunkwise_p1_core_s{4,8,16}` entry points.
    P1,
    /// The chunkwise-P2 split: `gdn_cw_p2_stage_{a,h}` over one workspace slot.
    P2,
}

/// Value-split tag the P2 split reports under: its 32-column stage-H strip (`P2_VW`) equals P1's s4.
const P2_SPLIT_TAG: u32 = 4;

/// Value-split tags a route runs: every `--s` under p1, the single P2 strip under p2.
fn route_splits(route: Route, s: &[u32]) -> Vec<u32> {
    match route {
        Route::P1 => s.to_vec(),
        Route::P2 => vec![P2_SPLIT_TAG],
    }
}

/// Workspace bytes of one case: 0 for p1; for p2 the P2 policy's workspace minus the fused layout, checked against `cw_p2_bytes`.
#[cfg(feature = "cuda")]
fn route_workspace_bytes(route: Route, rows: u32) -> Result<u64> {
    if route == Route::P1 {
        return Ok(0);
    }
    let spec = flyingfish::op_contract::GdnSpec::q38();
    let armed = flyingfish::gdn_plan::GdnPolicy::P2.workspace(&spec, rows)?;
    let fused = flyingfish::gdn_plan::GdnWorkspace::new(&spec, rows)?;
    let slot = armed
        .bytes
        .checked_sub(fused.bytes)
        .context("the P2 policy's workspace is smaller than the fused layout it extends")?;
    ensure!(
        slot == flyingfish::gdn_plan::cw_p2_bytes(48, rows),
        "the P2 slot region is {slot} B for {rows} rows; the planner derives {}",
        flyingfish::gdn_plan::cw_p2_bytes(48, rows)
    );
    Ok(slot)
}

/// Compute capability and SM count via `cuDeviceGetAttribute`.
#[cfg(feature = "cuda")]
fn query_topology(ctx: &std::sync::Arc<cudarc::driver::CudaContext>) -> Result<DeviceInfo> {
    use cudarc::driver::sys::CUdevice_attribute as Attribute;
    let major = ctx
        .attribute(Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .context("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)")?;
    let minor = ctx
        .attribute(Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .context("cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)")?;
    let sm_count = ctx
        .attribute(Attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .context("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)")?;
    ensure!(
        major > 0 && (0..=9).contains(&minor) && sm_count > 0,
        "CUDA driver returned an unusable topology: compute capability {major}.{minor}, \
         {sm_count} SMs"
    );
    Ok(DeviceInfo {
        name: ctx.name().unwrap_or_else(|_| "<unnamed>".to_string()),
        compute_capability: (major * 10 + minor) as u32,
        sm_count: sm_count as u32,
    })
}

/// Route policy for the queried topology; a failed query or, under p2, an unqualified device is an error.
#[cfg(any(feature = "cuda", test))]
fn require_qualified_topology(
    route: Route,
    topology: Result<DeviceInfo>,
) -> Result<flyingfish::gdn_plan::GdnPolicy> {
    let device = topology.with_context(|| {
        format!(
            "--route {} needs the device's compute capability and SM count from \
             cuDeviceGetAttribute, and that query failed. The topology guard is not \
             bypassed: re-run on a device whose driver answers the attribute query.",
            if route == Route::P1 { "p1" } else { "p2" }
        )
    })?;
    let policy = match route {
        Route::P1 => flyingfish::gdn_plan::GdnPolicy::for_device(
            device.compute_capability,
            device.sm_count as usize,
        ),
        Route::P2 => flyingfish::gdn_plan::GdnPolicy::P2,
    };
    if route == Route::P2 {
        ensure!(
            qualified_topology(Some(&device)),
            "--route p2 needs the qualified topology (arch 89, 128 SMs); this device is \
             sm{} with {} SMs. Re-run on the gate box or with --route p1.",
            device.compute_capability,
            device.sm_count
        );
        ensure!(
            policy.cw_p2_for_step(flyingfish::gdn_plan::MIN_VERIFY_ROWS + 1),
            "--route p2 runs under the P2 policy, which must carry the chunkwise-P2 split"
        );
    }
    Ok(policy)
}

/// The same guard, reading the topology from the device the phase is about to launch on.
#[cfg(feature = "cuda")]
fn route_policy(
    route: Route,
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
) -> Result<flyingfish::gdn_plan::GdnPolicy> {
    require_qualified_topology(route, query_topology(ctx))
}

#[derive(Parser)]
struct NumericsArgs {
    /// Frozen P1 package directory (MANIFEST.json, inputs/, truth/, gate_basis.json).
    #[arg(long)]
    package: PathBuf,
    /// Run directory for the raw outputs and the gate report.
    #[arg(long)]
    run_dir: PathBuf,
    /// Chunk size; the built core is C64 only.
    #[arg(long, default_value_t = KERNEL_C)]
    c: u32,
    /// Value splits; ignored under `--route p2`.
    #[arg(long, value_delimiter = ',', default_value = "8,4,16")]
    s: Vec<u32>,
    /// Which core to launch: the fused P1 core, or the chunkwise-P2 split.
    #[arg(long, value_enum, default_value = "p1")]
    route: Route,
    /// Clock to substitute for the kernel launch.
    #[arg(long, value_enum, default_value = "stub")]
    clock: Clock,
    /// Record the box3-shaped environment receipt instead of the local one.
    #[arg(long)]
    simulate_box3: bool,
    /// Only these fixtures (default: every fixture in the manifest).
    #[arg(long, value_delimiter = ',')]
    case: Vec<String>,
}

#[derive(Parser)]
struct TimingArgs {
    /// Frozen package directory; optional only with `--rows`.
    #[arg(long, required_unless_present = "rows")]
    package: Option<PathBuf>,
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long, default_value_t = KERNEL_C)]
    c: u32,
    #[arg(long, value_delimiter = ',', default_value = "8,4,16")]
    s: Vec<u32>,
    /// Candidate core; the baseline is the main core.
    #[arg(long, value_enum, default_value = "p1")]
    route: Route,
    /// Paired ABBA blocks per configuration.
    #[arg(long, default_value_t = TIMING_BLOCKS)]
    blocks: u32,
    #[arg(long, value_enum, default_value = "stub")]
    clock: Clock,
    /// Fixture for the timing legs (one configuration's worth of work).
    #[arg(long, conflicts_with = "rows")]
    case: Option<String>,
    /// Synthesise a deterministic single-step case of N live rows instead of reading a fixture.
    #[arg(long, conflicts_with = "case")]
    rows: Option<usize>,
    #[arg(long)]
    simulate_box3: bool,
}

/// splitmix64 over a per-(array, index) key, centred and scaled to the frozen fixtures' range.
fn synth(rows: usize, array: u64, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut key =
        0x9E37_79B9_7F4A_7C15u64 ^ (array.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ rows as u64);
    for _ in 0..n {
        key = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = key;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // (0,1) scaled to about +/-1.
        let unit = ((z >> 40) as f32 / 8_388_608.0) - 1.0;
        // beta = sigmoid >= 0, g = -exp(a_log)*softplus <= 0 (q38_gdn_gates); q/k/v/S0 stay signed unit.
        let value = match array {
            4 => 1.0 / (1.0 + (-unit).exp()),
            5 => -(-unit).exp().ln_1p(),
            _ => unit,
        };
        out.push(value);
    }
    out
}

/// A single-step case of `rows` live rows, shaped like a manifest fixture.
fn synth_case(rows: usize) -> (Fixture, CaseArrays) {
    const HEADS: usize = 48;
    const DK: usize = 128;
    const DV: usize = 128;
    let name = format!("synth{rows}");
    let rec = |shape: Vec<usize>| ArrayRec {
        path: String::new(),
        sha256: String::new(),
        dtype: "f32".into(),
        shape,
    };
    let arrays = CaseArrays {
        q: synth(rows, 1, rows * HEADS * DK),
        k: synth(rows, 2, rows * HEADS * DK),
        v: synth(rows, 3, rows * HEADS * DV),
        beta: synth(rows, 4, rows * HEADS),
        g: synth(rows, 5, rows * HEADS),
        s0: synth(rows, 6, HEADS * DK * DV),
    };
    let mut map = BTreeMap::new();
    map.insert("q".to_string(), rec(vec![rows, HEADS, DK]));
    map.insert("k".to_string(), rec(vec![rows, HEADS, DK]));
    map.insert("v".to_string(), rec(vec![rows, HEADS, DV]));
    map.insert("beta".to_string(), rec(vec![rows, HEADS]));
    map.insert("g".to_string(), rec(vec![rows, HEADS]));
    map.insert("S0".to_string(), rec(vec![HEADS, DK, DV]));
    (
        Fixture {
            name,
            kind: "synthesised single step".into(),
            m: rows,
            prefix: 0,
            seed: Some(rows as i64),
            arrays: map,
            note: format!("{rows} synthesised rows; splitmix64 keyed by rows and array"),
        },
        arrays,
    )
}

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    generated_utc: String,
    purpose: String,
    residual_record_spec: ResidualSpec,
    files: Vec<ManifestFile>,
    fixtures: Vec<Fixture>,
    /// The FP64 oracle's per-fixture out/state.
    #[serde(default)]
    truth: Vec<TruthFile>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ResidualSpec {
    what: String,
    norms: Vec<String>,
    #[serde(rename = "NOT")]
    not: String,
}

#[derive(Deserialize, Clone)]
struct ManifestFile {
    fixture: String,
    array: String,
    path: String,
    sha256: String,
    bytes: u64,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    kind: String,
    /// Live rows this fixture contributes to the core; the manifest calls it M.
    #[serde(rename = "M")]
    m: usize,
    prefix: usize,
    #[serde(rename = "seed")]
    seed: Option<i64>,
    arrays: BTreeMap<String, ArrayRec>,
    note: String,
}

#[derive(Deserialize, Clone)]
struct ArrayRec {
    path: String,
    sha256: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Deserialize, Clone)]
struct TruthFile {
    fixture: String,
    /// "out" or "state".
    which: String,
    path: String,
    sha256: String,
    bytes: u64,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Basis {
    schema: String,
    band: f64,
    ulp_floor: f64,
    rows: Vec<BasisRow>,
}

#[derive(Deserialize, Serialize, Clone)]
struct BasisRow {
    case: String,
    #[serde(rename = "C")]
    c: u32,
    s: u32,
    out_max_abs: f64,
    out_rel_rms: f64,
    state_rel_rms: f64,
    frac_le_1ulp: f64,
    gate_out_max_abs: f64,
    gate_out_rel_rms: f64,
    gate_state_rel_rms: f64,
    frac_floor: f64,
    finite: bool,
    input_sha256: String,
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

/// The order of the six arrays in the frozen package and in `input_sha256`.
const ARRAYS: [&str; 6] = ["q", "k", "v", "beta", "g", "S0"];

struct Package {
    root: PathBuf,
    manifest: Manifest,
    basis: BTreeMap<(String, u32, u32), BasisRow>,
    basis_doc: Basis,
    truth: BTreeMap<(String, String), TruthFile>,
    verified: usize,
    truth_verified: usize,
}

impl Package {
    fn truth_of(&self, fixture: &str, which: &str) -> Result<&TruthFile> {
        self.truth
            .get(&(fixture.to_string(), which.to_string()))
            .with_context(|| format!("manifest has no {which} truth for fixture {fixture}"))
    }
}

/// Verifies every manifest file before any array is read.
fn open_package(root: &Path) -> Result<Package> {
    let manifest_path = root.join("MANIFEST.json");
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    ensure!(
        manifest.schema == SCHEMA_MANIFEST,
        "manifest schema {} is not {SCHEMA_MANIFEST}",
        manifest.schema
    );
    ensure!(!manifest.fixtures.is_empty(), "manifest lists no fixtures");

    let mut verified = 0usize;
    for f in &manifest.files {
        ensure!(
            f.dtype == "float32",
            "{} is {}, not float32",
            f.path,
            f.dtype
        );
        let path = root.join(&f.path);
        let got = sha256_file(&path)?;
        ensure!(
            got == f.sha256,
            "SHA mismatch on {}: manifest {}, file {got}",
            f.path,
            f.sha256
        );
        let bytes = std::fs::metadata(&path)?.len();
        ensure!(
            bytes == f.bytes,
            "{} is {bytes} B, manifest says {} B",
            f.path,
            f.bytes
        );
        let expect: u64 = f.shape.iter().map(|d| *d as u64).product();
        ensure!(
            expect * 4 == f.bytes,
            "{} shape {:?} does not match {} B",
            f.path,
            f.shape,
            f.bytes
        );
        verified += 1;
    }
    // The per-fixture records must agree with the file list.
    for fx in &manifest.fixtures {
        for a in ARRAYS {
            let rec = fx
                .arrays
                .get(a)
                .with_context(|| format!("fixture {} has no {a}", fx.name))?;
            let listed = manifest
                .files
                .iter()
                .find(|f| f.fixture == fx.name && f.array == a)
                .with_context(|| format!("{} {a} is not in the file list", fx.name))?;
            ensure!(
                listed.sha256 == rec.sha256
                    && listed.path == rec.path
                    && listed.dtype == rec.dtype
                    && listed.shape == rec.shape,
                "fixture {} {a} disagrees with the file list on path, sha256, dtype or shape",
                fx.name
            );
        }
        ensure!(fx.m >= 1, "fixture {} has no live row", fx.name);
        check_fixture_shapes(fx)?;
    }

    for fx in &manifest.fixtures {
        ensure!(
            !fx.name.is_empty()
                && !fx.name.contains('/')
                && !fx.name.contains('\\')
                && fx.name != "."
                && fx.name != ".."
                && Path::new(&fx.name)
                    .file_name()
                    .is_some_and(|n| n == fx.name.as_str()),
            "fixture name {:?} is not a plain basename",
            fx.name
        );
    }
    let mut fixture_names = BTreeMap::new();
    for fx in &manifest.fixtures {
        ensure!(
            fixture_names.insert(fx.name.clone(), ()).is_none(),
            "manifest lists fixture {} twice; the second would overwrite the first",
            fx.name
        );
    }

    // The FP64 truth, verified on the same terms as the inputs, for every fixture.
    let mut truth = BTreeMap::new();
    let mut truth_verified = 0usize;
    for t in &manifest.truth {
        ensure!(
            t.dtype == "float64",
            "{} is {}, not float64",
            t.path,
            t.dtype
        );
        ensure!(
            t.which == "out" || t.which == "state",
            "{} names which={}, which is neither out nor state",
            t.path,
            t.which
        );
        let path = root.join(&t.path);
        let got = sha256_file(&path)?;
        ensure!(
            got == t.sha256,
            "SHA mismatch on {}: manifest {}, file {got}",
            t.path,
            t.sha256
        );
        let bytes = std::fs::metadata(&path)?.len();
        ensure!(
            bytes == t.bytes,
            "{} is {bytes} B, the manifest says {} B",
            t.path,
            t.bytes
        );
        let expect: u64 = t.shape.iter().map(|d| *d as u64).product();
        ensure!(
            expect * 8 == t.bytes,
            "{} shape {:?} does not match {} B",
            t.path,
            t.shape,
            t.bytes
        );
        let key = (t.fixture.clone(), t.which.clone());
        ensure!(
            truth.insert(key.clone(), t.clone()).is_none(),
            "manifest has two {} truth records for fixture {}; the second would overwrite the first",
            t.which,
            t.fixture
        );
        truth_verified += 1;
    }
    for fx in &manifest.fixtures {
        for w in ["out", "state"] {
            let key = (fx.name.clone(), w.to_string());
            let t = truth
                .get(&key)
                .with_context(|| format!("manifest has no {w} truth for fixture {}", fx.name))?;
            // A truth record's path must be its fixture's own out/state file.
            ensure!(
                t.path == format!("truth/{}.{}.f64", fx.name, w),
                "truth record for {} claims path {}, not truth/{}.{}.f64",
                fx.name,
                t.path,
                fx.name,
                w
            );
            let want: Vec<usize> = if w == "out" {
                vec![fx.m, HEADS, DK]
            } else {
                vec![HEADS, DK, DV]
            };
            ensure!(
                t.shape == want,
                "truth {} for {} is {:?}, the fixture is {:?}",
                t.path,
                fx.name,
                t.shape,
                want
            );
        }
    }

    let basis_path = root.join("gate_basis.json");
    let text = std::fs::read_to_string(&basis_path)
        .with_context(|| format!("read {}", basis_path.display()))?;
    let basis_doc: Basis =
        serde_json::from_str(&text).with_context(|| format!("parse {}", basis_path.display()))?;
    ensure!(
        basis_doc.schema == SCHEMA_BASIS,
        "gate basis schema {} is not {SCHEMA_BASIS}",
        basis_doc.schema
    );
    ensure!(
        (basis_doc.band - GATE_BAND).abs() <= 1e-12
            && (basis_doc.ulp_floor - FRAC_FLOOR_MARGIN).abs() <= 1e-12,
        "gate basis declares band {} and ulp_floor {}, not the registered {GATE_BAND} and {FRAC_FLOOR_MARGIN}",
        basis_doc.band,
        basis_doc.ulp_floor
    );
    let mut basis = BTreeMap::new();
    for r in &basis_doc.rows {
        let key = (r.case.clone(), r.c, r.s);
        ensure!(
            basis.insert(key.clone(), r.clone()).is_none(),
            "gate_basis.json has two rows for {} C{} s{}; the second would overwrite the first",
            r.case,
            r.c,
            r.s
        );
    }
    Ok(Package {
        root: root.to_path_buf(),
        manifest,
        basis,
        basis_doc,
        truth,
        verified,
        truth_verified,
    })
}

fn read_f32(root: &Path, rec: &ArrayRec) -> Result<Vec<f32>> {
    ensure!(
        rec.dtype == "float32",
        "{} is {}, not float32",
        rec.path,
        rec.dtype
    );
    let path = root.join(&rec.path);
    let got = sha256_file(&path)?;
    ensure!(
        got == rec.sha256,
        "SHA mismatch on {}: manifest {}, file {got}",
        rec.path,
        rec.sha256
    );
    let bytes = std::fs::read(&path)?;
    let n: usize = rec.shape.iter().product();
    ensure!(
        bytes.len() == n * 4,
        "{} is {} B, shape {:?} needs {}",
        rec.path,
        bytes.len(),
        rec.shape,
        n * 4
    );
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The FP64 oracle's array for one fixture, verified against its manifest record on every call.
fn read_truth_f64(root: &Path, rec: &TruthFile) -> Result<Vec<f64>> {
    ensure!(
        rec.dtype == "float64",
        "{} is {}, not float64",
        rec.path,
        rec.dtype
    );
    let path = root.join(&rec.path);
    let got = sha256_file(&path)?;
    ensure!(
        got == rec.sha256,
        "SHA mismatch on {}: manifest {}, file {got}",
        rec.path,
        rec.sha256
    );
    let bytes = std::fs::read(&path)?;
    let n: usize = rec.shape.iter().product();
    ensure!(
        bytes.len() == n * 8,
        "{} is {} B, shape {:?} needs {}",
        rec.path,
        bytes.len(),
        rec.shape,
        n * 8
    );
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Local BF16 ULP at |x| (gdn_phase2_model_w.bf16_ulp): 2^(floor(log2|x|) - 7), and the 2^-133 subnormal grid below 2^-126.
fn bf16_ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 {
        return 2f64.powi(-133);
    }
    if a < 2f64.powi(-126) {
        return 2f64.powi(-133);
    }
    2f64.powi(a.log2().floor() as i32 - 7)
}

struct Metrics {
    max_abs: f64,
    rel_rms: f64,
    frac_le_1ulp: f64,
    finite: bool,
}

fn metrics(got: &[f32], reference: &[f64]) -> Result<Metrics> {
    ensure!(
        got.len() == reference.len(),
        "length {} against reference {}",
        got.len(),
        reference.len()
    );
    let mut max_abs = 0f64;
    let mut sum = 0f64;
    let mut sum_ref = 0f64;
    let mut within = 0usize;
    let mut finite = true;
    for (g, r) in got.iter().zip(reference) {
        let g = *g as f64;
        let r = *r;
        finite &= g.is_finite() && r.is_finite();
        let d = (g - r).abs();
        max_abs = max_abs.max(d);
        sum += d * d;
        sum_ref += r * r;
        let u = bf16_ulp(r);
        if d / u <= 1.0 {
            within += 1;
        }
    }
    let n = got.len() as f64;
    let rms = (sum / n).sqrt();
    let ref_rms = (sum_ref / n).sqrt();
    Ok(Metrics {
        max_abs,
        rel_rms: if ref_rms > 0.0 {
            rms / ref_rms
        } else {
            f64::NAN
        },
        frac_le_1ulp: within as f64 / n,
        finite,
    })
}

/// RNE to BF16 and back.
fn bf16_round(x: f64) -> f64 {
    let bits = (x as f32).to_bits();
    // round-to-nearest-even on the truncated upper 16 bits
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fffu32 + lsb;
    let up = (bits + rounding) & 0xffff0000;
    f32::from_bits(up) as f64
}

/// The documented stub field: sin(n * 0.7) and sin(n * 0.7 + 1.3) interleaved, scaled
/// by 1/4, BF16-rounded; O(1) magnitude and non-constant.
fn stub_field(n: usize, phase: f64) -> f64 {
    let x = n as f64;
    bf16_round((x * 0.7 + phase).sin() * 0.25)
}

#[derive(Serialize)]
struct GateRow {
    case: String,
    c: u32,
    s: u32,
    chunks: u32,
    multi_chunk: bool,
    model_out_max_abs: f64,
    model_out_rel_rms: f64,
    model_state_rel_rms: f64,
    model_frac_le_1ulp: f64,
    got_out_max_abs: f64,
    got_out_rel_rms: f64,
    got_state_rel_rms: f64,
    got_frac_le_1ulp: f64,
    /// None where the ratio is infinite, i.e. a zero baseline and a non-zero candidate.
    ratio_out_max_abs: Option<f64>,
    ratio_out_rel_rms: Option<f64>,
    ratio_state_rel_rms: Option<f64>,
    frac_margin: f64,
    out_finite: bool,
    state_finite: bool,
    lengths_match: bool,
    heads_present: u32,
    verdict: &'static str,
    failures: Vec<String>,
}

/// One (fixture, split) row of the identity leg: what every launch produced.
#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct IdentityRow {
    case: String,
    #[serde(rename = "C")]
    c: u32,
    s: u32,
    /// The entry point actually launched.
    entry: String,
    arm: &'static str,
    runs: u32,
    /// The digest every run must match, and how many distinct digests appeared.
    out_sha256: String,
    state_sha256: String,
    distinct_out_sha256: usize,
    distinct_state_sha256: usize,
    /// The first launch whose out or state digest left the common value.
    first_divergent_run: Option<u32>,
    run_out_sha256: Vec<String>,
    run_state_sha256: Vec<String>,
    verdict: &'static str,
}

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct IdentityReport {
    schema: &'static str,
    generated_utc: String,
    clock: &'static str,
    package: String,
    manifest_generated_utc: String,
    files_sha256_verified: usize,
    /// What runs between two measured launches (see PERTURBATION).
    perturbation: &'static str,
    environment: EnvReceipt,
    rows: Vec<IdentityRow>,
    verdict: &'static str,
}

#[derive(Serialize)]
struct GateReport {
    schema: &'static str,
    generated_utc: String,
    clock: String,
    package: String,
    manifest_generated_utc: String,
    manifest_purpose: String,
    files_sha256_verified: usize,
    truth_files_sha256_verified: usize,
    reference: &'static str,
    band: f64,
    ulp_floor: f64,
    chunk_sizes_evaluated: Vec<u32>,
    splits_evaluated: Vec<u32>,
    rows: Vec<GateRow>,
    fixtures: Vec<FixtureReceipt>,
    multi_chunk_condition: MultiChunk,
    residual: ResidualStatus,
    environment: EnvReceipt,
    inputs_reused_from_manifest: bool,
    /// (file name, sha256) of the derived BF16 sources both arms read, written into the run directory.
    derived_bf16_sources: Vec<(String, String)>,
}

/// What each fixture in the package is, carried into the receipt.
#[derive(Serialize)]
struct FixtureReceipt {
    name: String,
    kind: String,
    rows: usize,
    prefix: usize,
    live_rows: usize,
    seed: Option<i64>,
    /// SHA-verified input bytes this fixture contributed (f32 = 4 B per element).
    verified_input_bytes: usize,
    note: String,
}

#[derive(Serialize)]
struct MultiChunk {
    rule: &'static str,
    satisfied: bool,
    detail: Vec<MultiChunkRow>,
}

#[derive(Serialize)]
struct MultiChunkRow {
    c: u32,
    multi_chunk_fixtures: Vec<String>,
    passing: Vec<String>,
    satisfied: bool,
}

#[derive(Serialize)]
struct ResidualStatus {
    required_by: &'static str,
    what: String,
    norms: Vec<String>,
    not: String,
    /// Protocol treatment of this record; reported, never a verdict.
    gating: &'static str,
    emitted: bool,
    reason: String,
}

#[derive(Serialize)]
struct EnvReceipt {
    label: String,
    simulated_box3: bool,
    rust: &'static str,
    /// The live device when one was touched, otherwise the no-device profile.
    target_profile: String,
    nvcc_observed: &'static str,
    cuda_visible_devices: String,
    device_touched: bool,
    device: Option<DeviceInfo>,
    /// Deviations of a STUB leg from a device-touching run, which carries `device`.
    differs_from_box3: Vec<String>,
}

/// What the CUDA driver reports about the device this run used; a stub leg carries None.
#[derive(Serialize, Clone)]
struct DeviceInfo {
    name: String,
    compute_capability: u32,
    sm_count: u32,
}

/// True when the engine's policy selects the chunkwise-P2 split on this device.
fn qualified_topology(device: Option<&DeviceInfo>) -> bool {
    device.is_some_and(|d| {
        flyingfish::gdn_plan::GdnPolicy::for_device(d.compute_capability, d.sm_count as usize)
            .cw_p2_for_step(flyingfish::gdn_plan::MIN_VERIFY_ROWS + 1)
    })
}

fn now_utc() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = t / 86_400;
    let secs = t % 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// Howard Hinnant's civil_from_days, the standard days->(y,m,d) conversion.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn env_receipt(
    simulate_box3: bool,
    device: Option<DeviceInfo>,
    device_touched: bool,
) -> EnvReceipt {
    let stub = !device_touched;
    EnvReceipt {
        label: match (&device, simulate_box3) {
            (Some(_), _) => "measured on this host",
            (None, true) => "box3-shaped stub, no CUDA device present",
            (None, false) => "local stub, no CUDA device present",
        }
        .to_string(),
        simulated_box3: simulate_box3,
        rust: "see the build log (build.rs records nvcc; rustc is the pinned toolchain)",
        target_profile: match &device {
            Some(d) => format!("sm{} / {}", d.compute_capability, d.name),
            None if simulate_box3 => "sm89 / CUDA 13.0 (asserted, not observed)".to_string(),
            None => "no device; the stub profile of this host".to_string(),
        },
        nvcc_observed: "recorded by build.rs as a cargo warning; see the build log",
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
            .unwrap_or_else(|_| "<unset>".into()),
        device_touched,
        device: device.clone(),
        differs_from_box3: if !stub {
            match device.as_ref() {
                Some(d)
                    if flyingfish::gdn_plan::GdnPolicy::for_device(
                        d.compute_capability,
                        d.sm_count as usize,
                    )
                    .prefill_p1_split()
                        == 0 =>
                {
                    vec![format!(
                        "the launch ran on sm{} with {} SMs, not the qualified topology \
                         (arch 89, 128 SMs), so a gate verdict here is INVALID",
                        d.compute_capability, d.sm_count
                    )]
                }
                _ => vec![],
            }
        } else if simulate_box3 {
            vec![
                "no CUDA device is present here, so the kernel launch and the CUDA-event clock are the stub on both box3 and here".to_string(),
            ]
        } else {
            vec![
                "the kernel launch is the documented stub; on box3 it is the real gdn_chunkwise_p1_core_s* launch".to_string(),
                "the clock is the documented stub; on box3 it is CUDA events around the launch".to_string(),
            ]
        },
    }
}

#[cfg(feature = "cuda")]
fn split_function(s: u32) -> &'static str {
    match s {
        4 => "gdn_chunkwise_p1_core_s4",
        8 => "gdn_chunkwise_p1_core_s8",
        16 => "gdn_chunkwise_p1_core_s16",
        _ => panic!("split {s} is not 4/8/16"),
    }
}

/// The P1 core's dynamic shared-memory bytes at this split.
#[cfg(feature = "cuda")]
fn split_smem(s: u32) -> Result<u32> {
    Ok(match s {
        4 => flyingfish::gdn_plan::cw_p1::S4_BYTES,
        8 => flyingfish::gdn_plan::cw_p1::S8_BYTES,
        16 => flyingfish::gdn_plan::cw_p1::S16_BYTES,
        _ => anyhow::bail!("split {s} is not 4/8/16"),
    })
}

const HEADS: usize = 48;
const DK: usize = 128;
const DV: usize = 128;

/// Checks every fixture against the kernel ABI shapes (q/k/v [M,48,D], beta/g [M,48], S0 [48,128,128]) before any array read.
fn check_fixture_shapes(fx: &Fixture) -> Result<(usize, usize, usize)> {
    ensure!(fx.m >= 1, "fixture {} has no live row", fx.name);
    let want = |array: &str, shape: &[usize]| -> Result<()> {
        let rec = fx
            .arrays
            .get(array)
            .with_context(|| format!("fixture {} has no {array} array", fx.name))?;
        ensure!(
            rec.shape == shape,
            "fixture {} {array} is {:?}, the kernel ABI wants {:?}",
            fx.name,
            rec.shape,
            shape
        );
        Ok(())
    };
    for array in ["q", "k", "v"] {
        want(array, &[fx.m, HEADS, DK])?;
    }
    for array in ["beta", "g"] {
        want(array, &[fx.m, HEADS])?;
    }
    want("S0", &[HEADS, DK, DV])?;
    Ok((HEADS, DK, DV))
}

fn run_numerics(args: &NumericsArgs) -> Result<()> {
    ensure!(
        args.c == KERNEL_C,
        "the built P1 core is C{KERNEL_C} only; C{} is not built and is not run \
         (protocol §A fixes C64 for every candidate)",
        args.c
    );
    let splits = route_splits(args.route, &args.s);
    for s in &splits {
        ensure!(
            [4u32, 8, 16].contains(s),
            "split {s} is not one of 4/8/16 (protocol §A)"
        );
    }
    let pkg = open_package(&args.package)?;
    std::fs::create_dir_all(&args.run_dir)
        .with_context(|| format!("create {}", args.run_dir.display()))?;
    let fixtures = select_fixtures(&pkg, &args.case)?;

    // The reference arrays for the configurations actually run, and the basis rows.
    let mut rows = Vec::new();
    // The receipts are only produced on the CUDA leg, where the sources are derived.
    #[cfg(feature = "cuda")]
    let mut source_receipts: Vec<(String, String)> = Vec::new();
    #[cfg(not(feature = "cuda"))]
    let source_receipts: Vec<(String, String)> = Vec::new();
    let mut bytes_by_fixture: BTreeMap<String, usize> = BTreeMap::new();
    #[cfg(feature = "cuda")]
    let mut gpu: Option<Gpu> = None;
    // On the no-CUDA leg there is nothing to open; the stub clock is the only one.
    #[cfg(not(feature = "cuda"))]
    let gpu: Option<Gpu> = None;
    if args.clock == Clock::Cuda {
        #[cfg(feature = "cuda")]
        {
            let opened = Gpu::open(args.route)?;
            route_policy(args.route, &opened.ctx)?;
            gpu = Some(opened);
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("--clock cuda needs the crate's cuda feature");
    }

    for fx in &fixtures {
        let m = fx.m;
        let (heads, dk, dv) = check_fixture_shapes(fx)?;
        let arrays = CaseArrays::new(&pkg.root, fx)?;
        // The derived BF16 sources, written into the run directory.
        #[cfg(feature = "cuda")]
        {
            let (q_src, k_src, v_src) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
            for receipt in write_sources(
                &args.run_dir.join("sources"),
                &fx.name,
                &q_src,
                &k_src,
                &v_src,
            )? {
                source_receipts.push(receipt);
            }
        }
        let input_bytes = 4
            * (arrays.q.len()
                + arrays.k.len()
                + arrays.v.len()
                + arrays.beta.len()
                + arrays.g.len()
                + arrays.s0.len());
        bytes_by_fixture.insert(fx.name.clone(), input_bytes);

        // The FP64 truth does not depend on C or s, so it is read once per fixture.
        let truth_out = read_truth_f64(&pkg.root, pkg.truth_of(&fx.name, "out")?)?;
        let truth_state = read_truth_f64(&pkg.root, pkg.truth_of(&fx.name, "state")?)?;

        for &s in &splits {
            let basis = pkg
                .basis
                .get(&(fx.name.clone(), args.c, s))
                .with_context(|| {
                    format!(
                        "gate_basis.json has no row for {} C{} s{s}",
                        fx.name, args.c
                    )
                })?
                .clone();
            let input_sha = ARRAYS
                .iter()
                .map(|a| fx.arrays[*a].sha256.clone())
                .collect::<Vec<_>>()
                .join(";");
            ensure!(
                input_sha == basis.input_sha256,
                "gate basis row {} C{} s{s} names different input bytes; \
                 the basis and the package are not the same run",
                fx.name,
                args.c
            );

            let (out, state) = match (gpu.as_ref(), args.route) {
                #[cfg(feature = "cuda")]
                (Some(gpu), Route::P1) => gpu.run_p1(fx, &arrays, s)?,
                #[cfg(feature = "cuda")]
                (Some(gpu), Route::P2) => gpu.run_p2(fx, &arrays)?,
                (_, _) => (
                    stub_field_array(m * heads * dk, 0.0),
                    stub_field_array(heads * dk * dv, 1.3),
                ),
            };
            ensure!(
                out.len() == m * heads * dv,
                "{} C{} s{s}: output is {} elements, expected {}",
                fx.name,
                args.c,
                out.len(),
                m * heads * dv
            );
            ensure!(
                state.len() == heads * dk * dv,
                "{} C{} s{s}: state is {} elements, expected {}",
                fx.name,
                args.c,
                state.len(),
                heads * dk * dv
            );
            write_f32(
                &args
                    .run_dir
                    .join(format!("{}.C{}.s{}.out.f32", fx.name, args.c, s)),
                &out,
            )?;
            write_f32(
                &args
                    .run_dir
                    .join(format!("{}.C{}.s{}.state.f32", fx.name, args.c, s)),
                &state,
            )?;

            let out_metrics = metrics(&out, &truth_out)?;
            let state_metrics = metrics(&state, &truth_state)?;
            rows.push(gate_row(
                fx,
                args.c,
                s,
                &basis,
                &out_metrics,
                &state_metrics,
                heads as u32,
            )?);
            let verdict = rows.last().map(|r| r.verdict).unwrap_or("?");
            eprintln!("  {:16} C{} s{s:<2} {verdict}", fx.name, args.c);
        }
    }
    let residual = residual_status(&pkg.manifest.residual_record_spec);
    let device_touched = gpu.is_some();
    let device = device_of(gpu.as_ref());
    if device_touched && !qualified_topology(device.as_ref()) {
        for r in rows.iter_mut() {
            r.verdict = "INVALID";
            r.failures.push(format!(
                "device {:?} is not the qualified topology (arch 89, 128 SMs), so no row \
                 carries a verdict",
                device.as_ref().map(|d| (d.compute_capability, d.sm_count))
            ));
        }
    }
    if args.clock == Clock::Stub {
        for r in rows.iter_mut() {
            r.verdict = "STUB";
            if r.failures.is_empty() {
                r.failures.push(
                    "the stub clock substitutes documented fixed durations; no row carries a verdict".into(),
                );
            }
        }
    }

    let multi = multi_chunk_condition(&rows, args.c);
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        *counts.entry(r.case.as_str()).or_default() += 1;
    }
    let receipts: Vec<FixtureReceipt> = fixtures
        .iter()
        .map(|f| FixtureReceipt {
            name: f.name.clone(),
            kind: f.kind.clone(),
            rows: counts.get(f.name.as_str()).copied().unwrap_or(0),
            prefix: f.prefix,
            live_rows: f.m,
            seed: f.seed,
            verified_input_bytes: bytes_by_fixture
                .get(f.name.as_str())
                .copied()
                .unwrap_or_default(),
            note: f.note.clone(),
        })
        .collect();
    let report = GateReport {
        schema: SCHEMA_GATE,
        generated_utc: now_utc(),
        clock: format!("{:?}", args.clock).to_lowercase(),
        package: args.package.display().to_string(),
        manifest_generated_utc: pkg.manifest.generated_utc.clone(),
        manifest_purpose: pkg.manifest.purpose.clone(),
        files_sha256_verified: pkg.verified,
        truth_files_sha256_verified: pkg.truth_verified,
        reference: "truth/<fixture>.{out,state}.f64 — the FP64 recurrent oracle, the same \
                    array gate_basis.json measured the CPU P1 model against",
        band: pkg.basis_doc.band,
        ulp_floor: pkg.basis_doc.ulp_floor,
        chunk_sizes_evaluated: vec![args.c],
        splits_evaluated: splits.clone(),
        residual,
        environment: env_receipt(args.simulate_box3, device.clone(), device_touched),
        inputs_reused_from_manifest: true,
        derived_bf16_sources: source_receipts,
        rows,
        fixtures: receipts,
        multi_chunk_condition: multi,
    };
    let path = args.run_dir.join("GATE.json");
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(&path, text.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    eprintln!("[gate] {}", path.display());
    let pass = report.rows.iter().filter(|r| r.verdict == "PASS").count();
    eprintln!(
        "[gate] route {:?}: both arms read the derived BF16 sources in sources/ \
         (p1 = sources -> gdn_prep_fp32 -> gdn_chunkwise_p1_core_s*, p2 = sources -> the \
         folded stages); the 1.25x band is read for each arm on those same inputs",
        args.route
    );
    eprintln!(
        "[gate] {}/{} rows PASS, multi-chunk condition {}",
        pass,
        report.rows.len(),
        if report.multi_chunk_condition.satisfied {
            "satisfied"
        } else {
            "NOT satisfied"
        }
    );
    // The report is emitted for every verdict; the exit code is non-zero only for plumbing failures.
    Ok(())
}

fn stub_field_array(n: usize, phase: f64) -> Vec<f32> {
    (0..n).map(|i| stub_field(i, phase) as f32).collect()
}

fn write_f32(path: &Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &bytes).with_context(|| format!("write {}", path.display()))
}

/// A zero baseline with a non-zero candidate has no ratio and is reported as null.
fn gate_ratio(got: f64, model: f64) -> Option<f64> {
    let r = if model == 0.0 {
        if got == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        got / model
    };
    r.is_finite().then_some(r)
}

fn residual_status(spec: &ResidualSpec) -> ResidualStatus {
    ResidualStatus {
        required_by: "4d 2026-10-13T07:06:44Z item 1",
        what: spec.what.clone(),
        norms: spec.norms.clone(),
        not: spec.not.clone(),
        gating: "recorded, not gating: Codex-D 07:28 item 1 and 4d's amendment item 1 keep \
                 the a-priori nu reported and non-gating (K2_REPORT 7564-7572)",
        emitted: false,
        reason: "the per-chunk records were computed by the CPU model run that wrote the \
                 package (ffa/rustfa/gdn_p1_frozen_basis.json, schema gdn-p1-gate-basis/3, \
                 key residual_records) and were dropped when the package copy was projected, \
                 so the frozen gate_basis.json carries no record to emit. Recomputing them \
                 here would need a host FP64 reimplementation of the reference A and T, \
                 which measures the model and not the candidate. Ruling requested."
            .to_string(),
    }
}

fn gate_row(
    fx: &Fixture,
    c: u32,
    s: u32,
    basis: &BasisRow,
    mo: &Metrics,
    state_m: &Metrics,
    heads: u32,
) -> Result<GateRow> {
    let mut failures = Vec::new();
    let chunks = fx.m.div_ceil(c as usize) as u32;
    let multi = chunks > 1;
    // A quantity whose basis is zero keeps ZERO tolerance; no epsilon is added.
    let r_max = gate_ratio(mo.max_abs, basis.out_max_abs);
    let r_rms = gate_ratio(mo.rel_rms, basis.out_rel_rms);
    let r_state = gate_ratio(state_m.rel_rms, basis.state_rel_rms);
    // The stored gate_* columns must equal the pre-registered 1.25x products.
    ensure!(
        (basis.gate_out_max_abs - basis.out_max_abs * GATE_BAND).abs()
            <= 1e-12 * basis.gate_out_max_abs.max(1e-300)
            && (basis.gate_out_rel_rms - basis.out_rel_rms * GATE_BAND).abs()
                <= 1e-12 * basis.gate_out_rel_rms.max(1e-300)
            && (basis.gate_state_rel_rms - basis.state_rel_rms * GATE_BAND).abs()
                <= 1e-12 * basis.gate_state_rel_rms.max(1e-300),
        "gate basis row {} C{} s{}: the stored gate columns are not {GATE_BAND}x the model columns",
        fx.name,
        c,
        s
    );
    ensure!(
        basis.finite,
        "gate basis row {} C{} s{}: the registered baseline is not finite, so no candidate \
         can be measured against it",
        fx.name,
        c,
        s
    );
    for (name, value) in [
        ("out_max_abs", basis.out_max_abs),
        ("out_rel_rms", basis.out_rel_rms),
        ("state_rel_rms", basis.state_rel_rms),
    ] {
        ensure!(
            value.is_finite() && value >= 0.0,
            "gate basis row {} C{} s{}: {name} {value} is negative or non-finite; a \
             negative baseline makes every candidate ratio negative and the > band \
             checks vacuous",
            fx.name,
            c,
            s
        );
    }
    ensure!(
        (0.0..=1.0).contains(&basis.frac_le_1ulp),
        "gate basis row {} C{} s{}: frac_le_1ulp {} is outside [0, 1]",
        fx.name,
        c,
        s,
        basis.frac_le_1ulp
    );
    ensure!(
        (basis.frac_floor - (basis.frac_le_1ulp - FRAC_FLOOR_MARGIN)).abs()
            <= 1e-12 * basis.frac_le_1ulp.abs().max(1e-300),
        "gate basis row {} C{} s{}: frac_floor {} is not frac_le_1ulp {} - {FRAC_FLOOR_MARGIN}",
        fx.name,
        c,
        s,
        basis.frac_floor,
        basis.frac_le_1ulp
    );
    if !mo.finite {
        failures.push("non-finite output value".into());
    }
    if !state_m.finite {
        failures.push("non-finite state value".into());
    }
    if heads == 0 {
        failures.push("no head".into());
    }
    if basis.out_max_abs == 0.0 && mo.max_abs != 0.0 {
        failures.push("zero-baseline out_max_abs exceeded with zero tolerance".into());
    }
    if basis.state_rel_rms == 0.0 && state_m.rel_rms != 0.0 {
        failures.push("zero-baseline state_rel_rms exceeded with zero tolerance".into());
    }
    for (name, r, limit) in [
        ("out_max_abs", r_max, GATE_BAND),
        ("out_rel_rms", r_rms, GATE_BAND),
        ("state_rel_rms", r_state, GATE_BAND),
    ] {
        match r {
            Some(v) if v > limit => {
                failures.push(format!("{name} ratio {v:.4} over {limit}"));
            }
            Some(_) => {}
            None => failures.push(format!(
                "{name} has no ratio: the baseline is zero and the candidate is not"
            )),
        }
    }
    if mo.frac_le_1ulp < basis.frac_floor {
        failures.push(format!(
            "frac_le_1ulp {:.4} below floor {:.4}",
            mo.frac_le_1ulp, basis.frac_floor
        ));
    }
    Ok(GateRow {
        case: fx.name.clone(),
        c,
        s,
        chunks,
        multi_chunk: multi,
        model_out_max_abs: basis.out_max_abs,
        model_out_rel_rms: basis.out_rel_rms,
        model_state_rel_rms: basis.state_rel_rms,
        model_frac_le_1ulp: basis.frac_le_1ulp,
        got_out_max_abs: mo.max_abs,
        got_out_rel_rms: mo.rel_rms,
        got_state_rel_rms: state_m.rel_rms,
        got_frac_le_1ulp: mo.frac_le_1ulp,
        ratio_out_max_abs: r_max,
        ratio_out_rel_rms: r_rms,
        ratio_state_rel_rms: r_state,
        frac_margin: mo.frac_le_1ulp - basis.frac_floor,
        out_finite: mo.finite,
        state_finite: state_m.finite,
        lengths_match: true,
        heads_present: heads,
        verdict: if failures.is_empty() { "PASS" } else { "FAIL" },
        failures,
    })
}

/// The fixtures a `--case` selection names; every name must match, so a typo cannot quietly shrink the run.
fn select_fixtures<'a>(pkg: &'a Package, case: &[String]) -> Result<Vec<&'a Fixture>> {
    let chosen: Vec<&Fixture> = pkg
        .manifest
        .fixtures
        .iter()
        .filter(|f| case.is_empty() || case.iter().any(|c| c == &f.name))
        .collect();
    for name in case {
        ensure!(
            pkg.manifest.fixtures.iter().any(|f| &f.name == name),
            "--case {name} is not a fixture in the manifest"
        );
    }
    ensure!(!chosen.is_empty(), "no fixture selected");
    Ok(chosen)
}

/// The multi-chunk condition: every C evaluated must have at least one passing fixture of more than one chunk.
fn multi_chunk_condition(rows: &[GateRow], c: u32) -> MultiChunk {
    let mut multi: Vec<String> = Vec::new();
    let mut passing: Vec<String> = Vec::new();
    for r in rows {
        if r.c == c && r.multi_chunk {
            if !multi.contains(&r.case) {
                multi.push(r.case.clone());
            }
            if r.verdict == "PASS" && !passing.contains(&r.case) {
                passing.push(r.case.clone());
            }
        }
    }
    let satisfied = !passing.is_empty();
    MultiChunk {
        rule: "every chunk size C must have at least one multi-chunk fixture (ceil(M/C) > 1) passing",
        satisfied,
        detail: vec![MultiChunkRow {
            c,
            multi_chunk_fixtures: multi,
            passing,
            satisfied,
        }],
    }
}

#[derive(Serialize, Clone)]
struct BlockStat {
    block: u32,
    log_ratio: f64,
    ratio: f64,
    a_ms: f64,
    b_ms: f64,
    /// Shortest individual leg duration in this block, in ms.
    min_leg_ms: f64,
}

#[derive(Serialize)]
struct LegSeries {
    label: &'static str,
    blocks: Vec<BlockStat>,
    geometric_mean_ratio: f64,
    ci_low: f64,
    ci_high: f64,
    mean_log_ratio: f64,
    stderr_log_ratio: f64,
    t_critical: f64,
    max_direction_drift: f64,
    n_legs: usize,
}

#[derive(Serialize, Clone)]
struct NullBand {
    label: &'static str,
    computed_before_any_p1_comparison: bool,
    blocks: u32,
    sd_log_ratio: f64,
    floor: f64,
    quantisation_term: f64,
    band: f64,
    valid: bool,
    invalid_reason: Option<String>,
    band_written_to: String,
}

#[derive(Serialize)]
struct TimingReport {
    schema: &'static str,
    generated_utc: String,
    clock: String,
    case: String,
    c: u32,
    splits: Vec<u32>,
    blocks: u32,
    configurations: Vec<TimingConfig>,
    environment: EnvReceipt,
    device_touched: bool,
}

#[derive(Serialize)]
struct TimingConfig {
    c: u32,
    s: u32,
    null_band: f64,
    null_valid: bool,
    comparisons: Vec<LegSeries>,
    verdict: &'static str,
    verdict_reason: String,
}

const GATE_BAND: f64 = 1.25;

fn run_timing(args: &TimingArgs) -> Result<()> {
    if let Some(rows) = args.rows {
        ensure!(rows >= 1, "--rows must be at least 1, not {rows}");
    }
    ensure!(
        args.c == KERNEL_C,
        "the built P1 core is C{KERNEL_C} only; C{} is not built and is not run \
         (protocol §A fixes C64 for every candidate)",
        args.c
    );
    ensure!(
        args.blocks == TIMING_BLOCKS,
        "the timing protocol fixes {TIMING_BLOCKS} paired ABBA blocks per configuration \
         (protocol §F); {} blocks are not a gate report",
        args.blocks
    );
    let splits = route_splits(args.route, &args.s);
    for s in &splits {
        ensure!(
            [4u32, 8, 16].contains(s),
            "split {s} is not one of 4/8/16 (protocol §A)"
        );
    }
    let synth_owned = args.rows.map(synth_case);
    let pkg = match (&args.package, &synth_owned) {
        (Some(root), _) => Some(open_package(root)?),
        (None, None) => anyhow::bail!("timing needs --package unless --rows synthesises the case"),
        (None, Some(_)) => None,
    };
    let fx: &Fixture = match synth_owned.as_ref() {
        Some((fx, _)) => fx,
        None => {
            let pkg = pkg.as_ref().expect("package present without --rows");
            ensure!(
                !pkg.manifest.fixtures.is_empty(),
                "manifest lists no fixtures"
            );
            match &args.case {
                Some(name) => pkg
                    .manifest
                    .fixtures
                    .iter()
                    .find(|f| f.name == *name)
                    .with_context(|| format!("fixture {name} is not in the manifest"))?,
                None => &pkg.manifest.fixtures[0],
            }
        }
    };
    std::fs::create_dir_all(&args.run_dir)
        .with_context(|| format!("create {}", args.run_dir.display()))?;

    #[cfg(feature = "cuda")]
    let mut gpu: Option<Gpu> = None;
    // On the no-CUDA leg there is nothing to open; the stub clock is the only one.
    #[cfg(not(feature = "cuda"))]
    let gpu: Option<Gpu> = None;
    if args.clock == Clock::Cuda {
        #[cfg(feature = "cuda")]
        {
            let opened = Gpu::open(args.route)?;
            route_policy(args.route, &opened.ctx)?;
            gpu = Some(opened);
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("--clock cuda needs the crate's cuda feature");
    }

    // Inputs load once; every leg restores S from S0.
    let arrays = match synth_owned.as_ref() {
        Some((_, a)) => CaseArrays {
            q: a.q.clone(),
            k: a.k.clone(),
            v: a.v.clone(),
            beta: a.beta.clone(),
            g: a.g.clone(),
            s0: a.s0.clone(),
        },
        None => CaseArrays::new(
            &pkg.as_ref().expect("package present without --rows").root,
            fx,
        )?,
    };

    let device_touched = gpu.is_some();
    let device = device_of(gpu.as_ref());
    // Side A is the baseline, side B the candidate; the NULL leg runs side A against itself.
    let (baseline, candidate) = match args.route {
        Route::P1 => (Which::Main, Which::P1),
        Route::P2 => (Which::Main, Which::P2),
    };
    let comparison_label: &'static str = match args.route {
        Route::P1 => "P1 core vs the main GDN core",
        Route::P2 => "chunkwise-P2 split (stages A, H) vs the main GDN core",
    };
    // Pass 1: every split's NULL band is written before any candidate leg runs.
    let mut bands: BTreeMap<u32, NullBand> = BTreeMap::new();
    for &s in &splits {
        let null_blocks = abba(args.clock, args.blocks, baseline, baseline, |which, leg| {
            let _ = leg;
            time_leg(gpu.as_ref(), fx, &arrays, which, s)
        });
        let band_path = args.run_dir.join(format!(
            "NULL_BAND.C{}.s{s}.{}.json",
            args.c,
            baseline.slug()
        ));
        let null = freeze_null_band(&null_blocks, &band_path)?;
        std::fs::write(&band_path, serde_json::to_string_pretty(&null)?.as_bytes())
            .with_context(|| format!("write {}", band_path.display()))?;
        bands.insert(s, null);
    }

    // Pass 2: the candidate legs, each against the band already written to disk.
    let mut configs = Vec::new();
    for &s in &splits {
        let null = bands
            .get(&s)
            .cloned()
            .with_context(|| format!("no null band was frozen for split {s}"))?;
        let p1_blocks = abba(
            args.clock,
            args.blocks,
            baseline,
            candidate,
            |which, leg| {
                let _ = leg;
                time_leg(gpu.as_ref(), fx, &arrays, which, s)
            },
        );
        let series = summarise(&p1_blocks, args.blocks as usize * 4, comparison_label);
        let unfinites = p1_blocks
            .iter()
            .filter(|b| !(b.log_ratio.is_finite() && b.a_ms.is_finite() && b.b_ms.is_finite()))
            .count();
        let (verdict, reason) = if !null.valid {
            (
                "INVALID",
                format!(
                    "the null band is not usable ({}) so the comparison is reported, not read",
                    null.invalid_reason.clone().unwrap_or_default()
                ),
            )
        } else if unfinites > 0 || !(series.ci_low.is_finite() && series.ci_high.is_finite()) {
            (
                "INVALID",
                format!(
                    "{unfinites} candidate block(s) produced no finite duration and the CI \
                     bounds are {}/{}, so the comparison is invalid rather than negative",
                    series.ci_low, series.ci_high
                ),
            )
        } else if args.clock == Clock::Stub {
            (
                "STUB",
                "the stub clock carries no measurement; this configuration is reported, not read"
                    .to_string(),
            )
        } else if device_touched && !qualified_topology(device_of(gpu.as_ref()).as_ref()) {
            (
                "INVALID",
"the device is not the qualified topology (arch 89, 128 SMs), so this comparison is reported, not read".to_string(),
            )
        } else {
            timing_verdict(&series, null.band)
        };
        configs.push(TimingConfig {
            c: args.c,
            s,
            null_band: null.band,
            null_valid: null.valid,
            comparisons: vec![series],
            verdict,
            verdict_reason: reason,
        });
    }

    let report = TimingReport {
        schema: SCHEMA_TIMING,
        generated_utc: now_utc(),
        clock: format!("{:?}", args.clock).to_lowercase(),
        case: fx.name.clone(),
        c: args.c,
        splits: splits.clone(),
        blocks: args.blocks,
        configurations: configs,
        environment: env_receipt(args.simulate_box3, device.clone(), device_touched),
        device_touched,
    };
    let path = args.run_dir.join("TIMING.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?.as_bytes())
        .with_context(|| format!("write {}", path.display()))?;
    eprintln!("[timing] {}", path.display());
    Ok(())
}

/// `log_ratio` is log(t_P1/t_main), so a faster candidate is a ratio below 1 and clears when the whole CI sits under 1 - band; a straddle does not clear.
fn timing_verdict(series: &LegSeries, band: f64) -> (&'static str, String) {
    if series.ci_high < 1.0 - band {
        (
            "PASS",
            format!(
                "the CI upper bound {:.4} clears the band {:.4} below 1",
                series.ci_high, band
            ),
        )
    } else {
        (
            "NOT_CLEARED",
            format!(
                "the CI [{:.4}, {:.4}] does not clear the band {:.4} below 1",
                series.ci_low, series.ci_high, band
            ),
        )
    }
}

/// One A/B pair inside a block: A is always the main core, B is the candidate.
#[derive(Copy, Clone, PartialEq)]
enum Which {
    Main,
    P1,
    P2,
}

impl Which {
    /// Filename-safe arm name.
    fn slug(self) -> &'static str {
        match self {
            Which::Main => "chunked",
            Which::P1 => "p1",
            Which::P2 => "p2",
        }
    }
}

fn time_leg(
    gpu: Option<&Gpu>,
    fx: &Fixture,
    arrays: &CaseArrays,
    which: Which,
    s: u32,
) -> Result<f64> {
    let _ = (fx, arrays);
    match gpu {
        None => Ok(stub_duration(which, s)),
        #[cfg(feature = "cuda")]
        Some(gpu) => gpu.time(which, fx, arrays, s),
        #[cfg(not(feature = "cuda"))]
        Some(_) => anyhow::bail!("the CUDA clock needs the crate's cuda feature"),
    }
}

/// Stub durations in milliseconds: fixed bases per core and a fixed 1% quantisation per leg.
fn stub_duration(which: Which, s: u32) -> f64 {
    let base = match which {
        Which::Main => 4.07,
        Which::P1 => 4.07 / (s as f64).sqrt(),
        // Two launches per chunk-row tile; same 32-column strip cost as the fused s4 core.
        Which::P2 => 4.07 / (P2_SPLIT_TAG as f64).sqrt(),
    };
    base * (1.0 + 0.01 * ((s % 4) as f64 - 1.5) / 1.5)
}

/// Four legs per block in A B B A order; the per-block statistic is the mean of the two paired log ratios, (leg0 A, leg1 B) and (leg2 B, leg3 A), as protocol §F registers it, and `a_ms`/`b_ms` are the legs' arithmetic means.
fn abba<F>(clock: Clock, blocks: u32, a: Which, b: Which, mut leg: F) -> Vec<BlockStat>
where
    F: FnMut(Which, usize) -> Result<f64>,
{
    let mut out = Vec::new();
    for blk in 0..blocks {
        let n = blk * 4;
        let order = [(a, n), (b, n + 1), (b, n + 2), (a, n + 3)];
        // Sides are assigned by leg position.
        let side = [true, false, false, true];
        let mut ta = [f64::NAN; 2];
        let mut tb = [f64::NAN; 2];
        let mut filled = [0usize; 2];
        for (i, (w, n)) in order.into_iter().enumerate() {
            // A failed leg is NaN, not a dropped sample.
            let t = leg(w, n as usize).unwrap_or(f64::NAN);
            if side[i] {
                ta[filled[0]] = t;
                filled[0] += 1;
            } else {
                tb[filled[1]] = t;
                filled[1] += 1;
            }
        }
        let paired = |a: f64, b: f64| {
            if a > 0.0 && b > 0.0 {
                (b / a).ln()
            } else {
                f64::NAN
            }
        };
        let lr = (paired(ta[0], tb[0]) + paired(ta[1], tb[1])) / 2.0;
        let a = (ta[0] + ta[1]) / 2.0;
        let bmean = (tb[0] + tb[1]) / 2.0;
        let min_leg_ms = [ta[0], ta[1], tb[0], tb[1]]
            .into_iter()
            .filter(|t| t.is_finite())
            .fold(f64::INFINITY, f64::min);
        out.push(BlockStat {
            block: blk,
            log_ratio: lr,
            ratio: if lr.is_nan() { f64::NAN } else { lr.exp() },
            a_ms: a,
            b_ms: bmean,
            min_leg_ms,
        });
    }
    let _ = clock;
    out
}

fn freeze_null_band(blocks: &[BlockStat], band_path: &Path) -> Result<NullBand> {
    let n = blocks.len() as f64;
    let lr: Vec<f64> = blocks.iter().map(|b| b.log_ratio).collect();
    let finite = lr.iter().all(|x| x.is_finite());
    let mean = if finite {
        lr.iter().sum::<f64>() / n
    } else {
        f64::NAN
    };
    let var = if finite {
        lr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        f64::NAN
    };
    let sd = var.sqrt();
    // event quantisation / batch time: the shortest individual leg in the block set
    let min_t = blocks
        .iter()
        .map(|b| b.min_leg_ms)
        .filter(|t| t.is_finite())
        .fold(f64::INFINITY, f64::min);
    let quant = if min_t.is_finite() && min_t > 0.0 {
        0.5 / 1000.0 / min_t // 0.5 us of event quantisation over the shortest leg
    } else {
        f64::INFINITY
    };
    let floor: f64 = 0.03;
    let spread = if sd.is_finite() {
        (3.0 * sd).exp() - 1.0
    } else {
        f64::INFINITY
    };
    let band = floor.max(spread).max(quant);
    let drift = blocks
        .iter()
        .map(|b| b.log_ratio.abs())
        .fold(0.0f64, |m, x| if x > m { x } else { m });
    let mut invalid = None;
    if !finite {
        invalid = Some("a null leg produced no finite duration".into());
    } else if band > 0.05 {
        invalid = Some(format!("null band {band:.4} exceeds 5%"));
    } else if drift > 0.05 {
        invalid = Some(format!("null direction drift {drift:.4} exceeds 5%"));
    }
    Ok(NullBand {
        label: "main core vs main core, the same ABBA block structure",
        computed_before_any_p1_comparison: true,
        blocks: blocks.len() as u32,
        sd_log_ratio: sd,
        floor,
        quantisation_term: quant,
        band,
        valid: invalid.is_none(),
        invalid_reason: invalid,
        band_written_to: band_path.display().to_string(),
    })
}

fn summarise(blocks: &[BlockStat], n_legs: usize, label: &'static str) -> LegSeries {
    let lr: Vec<f64> = blocks.iter().map(|b| b.log_ratio).collect();
    let n = lr.len() as f64;
    let mean = lr.iter().sum::<f64>() / n;
    let var = lr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    // Bonferroni over the nine (C, s) configurations, two-sided.
    let alpha = 0.05 / 9.0;
    let t = t_critical(alpha / 2.0, n - 1.0);
    let drift = lr.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    LegSeries {
        label,
        blocks: blocks.to_vec(),
        geometric_mean_ratio: mean.exp(),
        ci_low: (mean - t * se).exp(),
        ci_high: (mean + t * se).exp(),
        mean_log_ratio: mean,
        stderr_log_ratio: se,
        t_critical: t,
        max_direction_drift: drift,
        n_legs,
    }
}

/// Two-sided t critical value: the Student t CDF inverted through the regularised
/// incomplete beta by bisection (Numerical Recipes' betai/betacf).
fn t_critical(tail: f64, df: f64) -> f64 {
    let mut lo = 0.0f64;
    let mut hi = 200.0f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, df) < 1.0 - tail {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn student_t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let p = 0.5 * regularized_beta(x, df / 2.0, 0.5);
    if t > 0.0 { 1.0 - p } else { p }
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (lbeta + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(x, a, b) / a
    } else {
        1.0 - (front * betacf(1.0 - x, b, a) / b)
    }
}

/// Continued fraction for the incomplete beta (modified Lentz).
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-16 {
            break;
        }
    }
    h
}

/// Lanczos approximation of ln Gamma (g = 7, n = 9 coefficients).
fn ln_gamma(z: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut a = COEF[0];
    for (i, c) in COEF.iter().enumerate().skip(1) {
        a += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

/// Load one kernel module from this build's own cubin bytes.
#[cfg(feature = "cuda")]
fn load_module(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    name: &str,
) -> Result<std::sync::Arc<cudarc::driver::CudaModule>> {
    use cudarc::nvrtc::Ptx;
    let sm = compute_capability(ctx)?;
    let bytes = flyingfish::backend::kernel_cubin(name, sm)
        .with_context(|| format!("no cubin for {name} at sm{sm}; set CUDA_ARCH_LIST"))?;
    Ok(ctx.load_module(Ptx::from_binary(bytes.to_vec()))?)
}

#[cfg(feature = "cuda")]
fn compute_capability(ctx: &cudarc::driver::CudaContext) -> Result<u32> {
    use cudarc::driver::sys::CUdevice_attribute as A;
    let major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    ensure!(
        major > 0 && (0..=9).contains(&minor),
        "invalid CUDA compute capability {major}.{minor}"
    );
    Ok((major * 10 + minor) as u32)
}

/// Placeholder on the no-CUDA leg, where the stub clock is the only clock.
#[cfg(not(feature = "cuda"))]
struct Gpu;

#[cfg(feature = "cuda")]
fn device_of(gpu: Option<&Gpu>) -> Option<DeviceInfo> {
    gpu.map(|g| g.device_info.clone())
}

#[cfg(not(feature = "cuda"))]
fn device_of(_gpu: Option<&Gpu>) -> Option<DeviceInfo> {
    None
}

/// The device-side arguments of one case's chunkwise-P2 launches.
#[cfg(feature = "cuda")]
/// 1/sqrt(DK), the Q scale `gdn_prep_fp32` and the folded stage apply; the fixture's Q carries it.
#[cfg(feature = "cuda")]
fn p2_qscale() -> f32 {
    1.0f32 / 128.0f32.sqrt()
}

/// BF16 sources both arms read, folded from the fixture's post-prep FP32 arrays (48 value heads) back to 16 key heads:
///
///   q_src[m][kh][t] = bf16_rn(q_fixture[m][3*kh][t] / 1/sqrt(128))
///   k_src[m][kh][t] = bf16_rn(k_fixture[m][3*kh][t])   -- exact: the fixture's k is BF16
///   v_src[m][h][t]  = bf16_rn(v_fixture[m][h][t])      -- exact: the fixture's v is BF16
/// The three GQA replicas of q and k must be bit-equal.
#[cfg(feature = "cuda")]
fn derive_bf16_sources(
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> anyhow::Result<(Vec<half::bf16>, Vec<half::bf16>, Vec<half::bf16>)> {
    const VH: usize = 48;
    const KH: usize = 16;
    const DK: usize = 128;
    anyhow::ensure!(
        q.len() == k.len() && k.len() == v.len() && q.len().is_multiple_of(VH * DK),
        "fixture q/k/v lengths disagree or are not [rows, {VH}, {DK}]: {} / {} / {}",
        q.len(),
        k.len(),
        v.len()
    );
    let rows = q.len() / (VH * DK);
    let mut q_src = Vec::with_capacity(rows * KH * DK);
    let mut k_src = Vec::with_capacity(rows * KH * DK);
    let mut v_src = Vec::with_capacity(q.len());
    let qscale = p2_qscale();
    for m in 0..rows {
        for kh in 0..KH {
            for t in 0..DK {
                let base = (m * VH + 3 * kh) * DK + t;
                let (r1, r2) = (base + DK, base + 2 * DK);
                anyhow::ensure!(
                    q[base].to_bits() == q[r1].to_bits() && q[base].to_bits() == q[r2].to_bits(),
                    "fixture q GQA replicas disagree at row {m} key head {kh} col {t}"
                );
                anyhow::ensure!(
                    k[base].to_bits() == k[r1].to_bits() && k[base].to_bits() == k[r2].to_bits(),
                    "fixture k GQA replicas disagree at row {m} key head {kh} col {t}"
                );
                q_src.push(half::bf16::from_f32(q[base] / qscale));
                k_src.push(half::bf16::from_f32(k[base]));
            }
        }
        for x in &v[m * VH * DK..(m + 1) * VH * DK] {
            v_src.push(half::bf16::from_f32(*x));
        }
    }
    Ok((q_src, k_src, v_src))
}

/// The derived sources as raw bf16 bytes with their SHA-256.
#[cfg(feature = "cuda")]
fn write_sources(
    dir: &Path,
    name: &str,
    q_src: &[half::bf16],
    k_src: &[half::bf16],
    v_src: &[half::bf16],
) -> Result<Vec<(String, String)>> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let mut out = Vec::new();
    for (tag, values) in [("q", q_src), ("k", k_src), ("v", v_src)] {
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        let path = dir.join(format!("{name}.{tag}_src.bf16"));
        std::fs::write(&path, &raw).with_context(|| format!("write {}", path.display()))?;
        out.push((format!("{name}.{tag}_src.bf16"), sha256_bytes(&raw)));
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
struct P2Case<'a> {
    // q and k are BF16 16 KEY heads; v is a [M, 48, 128] contiguous BF16 block.
    dq: &'a cudarc::driver::CudaSlice<half::bf16>,
    dk: &'a cudarc::driver::CudaSlice<half::bf16>,
    dvv: &'a cudarc::driver::CudaSlice<half::bf16>,
    dbeta: &'a cudarc::driver::CudaSlice<f32>,
    dg: &'a cudarc::driver::CudaSlice<f32>,
    ds: &'a mut cudarc::driver::CudaSlice<f32>,
    dout: &'a mut cudarc::driver::CudaSlice<half::bf16>,
    meta: &'a cudarc::driver::CudaSlice<i32>,
    ws: &'a cudarc::driver::CudaSlice<u8>,
    m: usize,
    heads: u32,
}

#[cfg(feature = "cuda")]
struct Gpu {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    p1: [cudarc::driver::CudaFunction; 3],
    /// gdn_prep_fp32, the widen the P1 reference arm runs over the derived BF16 sources.
    prep: cudarc::driver::CudaFunction,
    /// The two chunkwise-P2 stages, loaded only under `--route p2`.
    p2: Option<[cudarc::driver::CudaFunction; 2]>,
    main: cudarc::driver::CudaFunction,
    device_info: DeviceInfo,
}

#[cfg(feature = "cuda")]
impl Gpu {
    /// Thread blocks of the P1 core and the main core, and the main core's dynamic shared memory in bytes.
    const P1_THREADS: u32 = 256;
    const MAIN_THREADS: u32 = 512;
    const MAIN_SMEM: u32 = 62_336;
    /// The production prefill core, the timing baseline (gdn_chunked.cu).
    const MAIN_NAME: &str = "gdn_chunked_core_v1";

    fn open(route: Route) -> Result<Self> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        use cudarc::driver::{CudaContext, LaunchConfig, sys};
        let ctx = CudaContext::new(0).context("create CUDA context")?;
        let arch = compute_capability(&ctx)?;
        let stream = ctx.new_stream().context("create CUDA stream")?;
        let p1_mod = load_module(&ctx, P1_MODULE)?;
        let p1 = [
            p1_mod.load_function(split_function(4))?,
            p1_mod.load_function(split_function(8))?,
            p1_mod.load_function(split_function(16))?,
        ];
        let main_mod = load_module(&ctx, "gdn/gdn_chunked")?;
        let main = main_mod.load_function(Self::MAIN_NAME)?;
        let prep = load_module(&ctx, PREP_MODULE)?.load_function("gdn_prep_fp32")?;
        for (i, bytes) in [split_smem(4)?, split_smem(8)?, split_smem(16)?]
            .iter()
            .enumerate()
        {
            if *bytes > 48 * 1024 {
                p1[i].set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    *bytes as i32,
                )?;
            }
        }
        if Self::MAIN_SMEM > 48 * 1024 {
            main.set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                Self::MAIN_SMEM as i32,
            )?;
        }
        // Touch LaunchConfig so the cfg type is used on this leg.
        let _ = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // The split's two entry points load under `--route p2` only.
        let p2 = if route == Route::P2 {
            let p2_mod = load_module(&ctx, P2_MODULE)?;
            Some([
                p2_mod.load_function("gdn_cw_p2_stage_a")?,
                p2_mod.load_function("gdn_cw_p2_stage_h")?,
            ])
        } else {
            None
        };
        eprintln!(
            "[cuda] sm{arch}, {P1_MODULE}{} and gdn/gdn_chunked loaded",
            if p2.is_some() {
                format!(", {P2_MODULE}")
            } else {
                String::new()
            }
        );
        let sm_count = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let name = ctx.name().unwrap_or_else(|_| "<unnamed>".to_string());
        let device_info = DeviceInfo {
            name,
            compute_capability: arch,
            sm_count,
        };
        eprintln!("[cuda] device {}", device_info.name);
        Ok(Gpu {
            ctx,
            stream,
            p1,
            prep,
            p2,
            main,
            device_info,
        })
    }

    /// gdn_prep_fp32 over the derived BF16 sources, returning the three widened FP32 buffers; grid (DV/128, HEADS, rows), one thread per (row, value head, element).
    fn widen_p1(
        &self,
        fx: &Fixture,
        dq_src: &cudarc::driver::CudaSlice<half::bf16>,
        dk_src: &cudarc::driver::CudaSlice<half::bf16>,
        dv_src: &cudarc::driver::CudaSlice<half::bf16>,
        heads: usize,
        dv: usize,
    ) -> Result<(
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<f32>,
    )> {
        use cudarc::driver::{LaunchConfig, PushKernelArg as _};
        let m = fx.m;
        let meta = self.meta(fx)?;
        let cfg = LaunchConfig {
            grid_dim: (1, heads as u32, u32::try_from(m).context("live rows")?),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut dq = self.stream.alloc_zeros::<f32>(m * heads * dv)?;
        let mut dk = self.stream.alloc_zeros::<f32>(m * heads * dv)?;
        let mut dvv = self.stream.alloc_zeros::<f32>(m * heads * dv)?;
        unsafe {
            self.stream
                .launch_builder(&self.prep)
                .arg(dq_src)
                .arg(dk_src)
                .arg(dv_src)
                .arg(&mut dq)
                .arg(&mut dk)
                .arg(&mut dvv)
                .arg(&(heads as u32 * dv as u32))
                .arg(&meta)
                .launch(cfg)?;
        }
        Ok((dq, dk, dvv))
    }

    /// The P1 reference arm: derived BF16 sources -> gdn_prep_fp32 -> fused P1 core.
    fn run_p1(&self, fx: &Fixture, arrays: &CaseArrays, s: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        use cudarc::driver::{LaunchConfig, PushKernelArg as _};
        let m = fx.m;
        let heads = fx.arrays["q"].shape[1];
        let dv = fx.arrays["v"].shape[2];
        let (q_src, k_src, v_src) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
        let dq_src = self.stream.clone_htod(&q_src)?;
        let dk_src = self.stream.clone_htod(&k_src)?;
        let dv_src = self.stream.clone_htod(&v_src)?;
        let (dq, dk, dvv) = self.widen_p1(fx, &dq_src, &dk_src, &dv_src, heads, dv)?;
        let dbeta = self.stream.clone_htod(&arrays.beta)?;
        let dg = self.stream.clone_htod(&arrays.g)?;
        let mut ds = self.stream.clone_htod(&arrays.s0)?;
        let dout = self.stream.alloc_zeros::<half::bf16>(m * heads * dv)?;
        let meta = self.meta(fx)?;
        let idx = match s {
            4 => 0,
            8 => 1,
            16 => 2,
            _ => anyhow::bail!("split {s} is not 4/8/16"),
        };
        let cfg = LaunchConfig {
            grid_dim: (s, heads as u32, 1),
            block_dim: (Self::P1_THREADS, 1, 1),
            shared_mem_bytes: split_smem(s)?,
        };
        unsafe {
            self.stream
                .launch_builder(&self.p1[idx])
                .arg(&dq)
                .arg(&dk)
                .arg(&dvv)
                .arg(&dbeta)
                .arg(&dg)
                .arg(&mut ds)
                .arg(&dout)
                .arg(&meta)
                // 0: the kernel reads meta->valid_rows.
                .arg(&0i32)
                .launch(cfg)?;
        }
        let out: Vec<half::bf16> = self.stream.clone_dtoh(&dout)?;
        let state: Vec<f32> = self.stream.clone_dtoh(&ds)?;
        let widened: Vec<f32> = out.iter().map(|b| b.to_f32()).collect();
        Ok((widened, state))
    }

    /// The two chunkwise-P2 launches for one case in binder order on `stream`: stage A fills the slot, stage H walks the chunks over one 32-column strip and writes `out`.
    #[cfg(feature = "cuda")]
    fn launch_p2(&self, stream: &cudarc::driver::CudaStream, case: &mut P2Case<'_>) -> Result<()> {
        use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg as _};
        let stages = self
            .p2
            .as_ref()
            .context("--route p2 needs the two gdn_cw_p2 stage entry points")?;
        let heads = case.heads;
        let nchunk = u32::try_from(case.m)
            .context("live rows")?
            .div_ceil(flyingfish::gdn_plan::CW_P2_CHUNK)
            .max(1);
        let smem = flyingfish::gdn_plan::cw_p2::SMEM_BYTES as u32;
        let per_chunk = LaunchConfig {
            grid_dim: (nchunk, heads, 1),
            block_dim: (Self::P1_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        let per_value_block = LaunchConfig {
            grid_dim: (
                u32::try_from(DV).context("DV")? / flyingfish::gdn_plan::CW_P2_VALUE_BLOCK,
                heads,
                1,
            ),
            block_dim: (Self::P1_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        let (slot, _hold) = case.ws.device_ptr(&self.stream);
        let stage = flyingfish::gdn_plan::P2StageArgs {
            q: case.dq.device_ptr(&self.stream).0,
            k: case.dk.device_ptr(&self.stream).0,
            v: case.dvv.device_ptr(&self.stream).0,
            beta: case.dbeta.device_ptr(&self.stream).0,
            g: case.dg.device_ptr(&self.stream).0,
            ws: slot,
            meta: case.meta.device_ptr(&self.stream).0,
            valid_rows: 0,
            // v is a contiguous [M, 48, 128] block; in production it is a conv_out column window with row stride conv_dim.
            v_row_stride: DV as u32 * 48,
        };
        unsafe {
            stream
                .launch_builder(&stages[0])
                .arg(&stage.q)
                .arg(&stage.k)
                .arg(&stage.v)
                .arg(&stage.beta)
                .arg(&stage.g)
                .arg(&stage.ws)
                .arg(&stage.meta)
                // 0: the kernel reads meta->valid_rows.
                .arg(&stage.valid_rows)
                .arg(&stage.v_row_stride)
                .launch(per_chunk)?;
            stream
                .launch_builder(&stages[1])
                .arg(&stage.q)
                .arg(&stage.k)
                .arg(&stage.v)
                .arg(&stage.beta)
                .arg(&stage.g)
                .arg(&*case.ds)
                .arg(&*case.dout)
                .arg(&stage.ws)
                .arg(&stage.meta)
                .arg(&stage.valid_rows)
                .launch(per_value_block)?;
        }
        Ok(())
    }

    /// The slot a case's stages run in, sized by the binder's planner.
    #[cfg(feature = "cuda")]
    fn p2_workspace(&self, m: usize) -> Result<cudarc::driver::CudaSlice<u8>> {
        let bytes = route_workspace_bytes(Route::P2, u32::try_from(m).context("live rows")?)?;
        ensure!(bytes > 0, "the P2 workspace for {m} rows is empty");
        Ok(self.stream.alloc_zeros::<u8>(bytes as usize)?)
    }

    /// Allocate, upload, launch the P2 stages, download; every leg starts from S0.
    #[cfg(feature = "cuda")]
    fn run_p2(&self, fx: &Fixture, arrays: &CaseArrays) -> Result<(Vec<f32>, Vec<f32>)> {
        let m = fx.m;
        let heads = fx.arrays["q"].shape[1] as u32;
        let dv = fx.arrays["v"].shape[2];
        let (q_bf16, k_bf16, v_bf16) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
        let dq = self.stream.clone_htod(&q_bf16)?;
        let dk = self.stream.clone_htod(&k_bf16)?;
        let dvv = self.stream.clone_htod(&v_bf16)?;
        let dbeta = self.stream.clone_htod(&arrays.beta)?;
        let dg = self.stream.clone_htod(&arrays.g)?;
        let mut ds = self.stream.clone_htod(&arrays.s0)?;
        let mut dout = self
            .stream
            .alloc_zeros::<half::bf16>(m * heads as usize * dv)?;
        let meta = self.meta(fx)?;
        let ws = self.p2_workspace(m)?;
        let mut case = P2Case {
            dq: &dq,
            dk: &dk,
            dvv: &dvv,
            dbeta: &dbeta,
            dg: &dg,
            ds: &mut ds,
            dout: &mut dout,
            meta: &meta,
            ws: &ws,
            m,
            heads,
        };
        self.launch_p2(&self.stream, &mut case)?;
        let out: Vec<half::bf16> = self.stream.clone_dtoh(&dout)?;
        let state: Vec<f32> = self.stream.clone_dtoh(&ds)?;
        let widened: Vec<f32> = out.iter().map(|b| b.to_f32()).collect();
        Ok((widened, state))
    }

    /// Launch the P2 stages `runs` times from identical inputs, digesting out and state after each; every run re-uploads S0 and zeroes out and the slot.
    #[cfg(feature = "cuda")]
    fn identity_p2(
        &self,
        fx: &Fixture,
        arrays: &CaseArrays,
        runs: u32,
    ) -> Result<(&'static str, Vec<String>, Vec<String>)> {
        let m = fx.m;
        let heads = fx.arrays["q"].shape[1] as u32;
        let dv = fx.arrays["v"].shape[2];
        let (q_bf16, k_bf16, v_bf16) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
        let dq = self.stream.clone_htod(&q_bf16)?;
        let dk = self.stream.clone_htod(&k_bf16)?;
        let dvv = self.stream.clone_htod(&v_bf16)?;
        let dbeta = self.stream.clone_htod(&arrays.beta)?;
        let dg = self.stream.clone_htod(&arrays.g)?;
        let meta = self.meta(fx)?;
        let entry = "gdn_cw_p2_stage_a+h";
        // The perturbation runs on a second stream; the measured launch is enqueued while it runs.
        let perturb_stream = self.ctx.new_stream()?;
        let mut ds_filler = self.stream.clone_htod(&arrays.s0)?;
        let mut dout_filler = self
            .stream
            .alloc_zeros::<half::bf16>(m * heads as usize * dv)?;
        let ws_filler = self.p2_workspace(m)?;
        let mut out_digests = Vec::with_capacity(runs as usize);
        let mut state_digests = Vec::with_capacity(runs as usize);
        for run in 1..=runs {
            // Fresh inputs for this run, uploaded and drained before anything is enqueued.
            let mut ds = self.stream.clone_htod(&arrays.s0)?;
            let mut dout = self
                .stream
                .alloc_zeros::<half::bf16>(m * heads as usize * dv)?;
            let ws = self.p2_workspace(m)?;
            self.stream.synchronize()?;
            for _ in 0..2 {
                let mut filler = P2Case {
                    dq: &dq,
                    dk: &dk,
                    dvv: &dvv,
                    dbeta: &dbeta,
                    dg: &dg,
                    ds: &mut ds_filler,
                    dout: &mut dout_filler,
                    meta: &meta,
                    ws: &ws_filler,
                    m,
                    heads,
                };
                self.launch_p2(&perturb_stream, &mut filler)?;
            }
            let mut case = P2Case {
                dq: &dq,
                dk: &dk,
                dvv: &dvv,
                dbeta: &dbeta,
                dg: &dg,
                ds: &mut ds,
                dout: &mut dout,
                meta: &meta,
                ws: &ws,
                m,
                heads,
            };
            self.launch_p2(&self.stream, &mut case)?;
            self.stream.synchronize()?;
            perturb_stream.synchronize()?;
            let out: Vec<half::bf16> = self.stream.clone_dtoh(&dout)?;
            let state: Vec<f32> = self.stream.clone_dtoh(&ds)?;
            let out_bytes: Vec<u8> = out.iter().flat_map(|b| b.to_ne_bytes()).collect();
            let state_bytes: Vec<u8> = state.iter().flat_map(|f| f.to_ne_bytes()).collect();
            out_digests.push(sha256_bytes(&out_bytes));
            state_digests.push(sha256_bytes(&state_bytes));
            eprintln!(
                "    run {run}/{runs} out {} state {}",
                &out_digests[out_digests.len() - 1][..16],
                &state_digests[state_digests.len() - 1][..16]
            );
        }
        Ok((entry, out_digests, state_digests))
    }

    /// Launch the P1 core `runs` times from IDENTICAL device inputs, digesting out and state after every launch; each run re-uploads S0 and zeroes the out buffer.
    fn identity_p1(
        &self,
        fx: &Fixture,
        arrays: &CaseArrays,
        s: u32,
        runs: u32,
    ) -> Result<(&'static str, Vec<String>, Vec<String>)> {
        use cudarc::driver::{LaunchConfig, PushKernelArg as _};
        let m = fx.m;
        let heads = fx.arrays["q"].shape[1];
        let dv = fx.arrays["v"].shape[2];
        let index = match s {
            4 => 0,
            8 => 1,
            16 => 2,
            _ => anyhow::bail!("split {s} is not 4/8/16"),
        };
        let entry = split_function(s);
        let fun = &self.p1[index];
        // The derived sources and prep, hoisted out of the measured launch.
        let (q_src, k_src, v_src) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
        let dq_src = self.stream.clone_htod(&q_src)?;
        let dk_src = self.stream.clone_htod(&k_src)?;
        let dv_src = self.stream.clone_htod(&v_src)?;
        let (dq, dk, dvv) = self.widen_p1(fx, &dq_src, &dk_src, &dv_src, heads, dv)?;
        let dbeta = self.stream.clone_htod(&arrays.beta)?;
        let dg = self.stream.clone_htod(&arrays.g)?;
        let meta = self.meta(fx)?;
        let cfg = LaunchConfig {
            grid_dim: (s, heads as u32, 1),
            block_dim: (Self::P1_THREADS, 1, 1),
            shared_mem_bytes: split_smem(s)?,
        };
        let launch = |stream: &cudarc::driver::CudaStream,
                      ds: &mut cudarc::driver::CudaSlice<f32>,
                      dout: &cudarc::driver::CudaSlice<half::bf16>|
         -> Result<()> {
            unsafe {
                stream
                    .launch_builder(fun)
                    .arg(&dq)
                    .arg(&dk)
                    .arg(&dvv)
                    .arg(&dbeta)
                    .arg(&dg)
                    .arg(ds)
                    .arg(dout)
                    .arg(&meta)
                    .arg(&0i32)
                    .launch(cfg)?;
            }
            Ok(())
        };
        // The perturbation's own buffers on a second stream; the measured launch is enqueued while the filler is still running.
        let perturb_stream = self.ctx.new_stream()?;
        let mut ds_filler = self.stream.clone_htod(&arrays.s0)?;
        let dout_filler = self.stream.alloc_zeros::<half::bf16>(m * heads * dv)?;
        let mut out_digests = Vec::with_capacity(runs as usize);
        let mut state_digests = Vec::with_capacity(runs as usize);
        for run in 1..=runs {
            // Fresh inputs for this run, uploaded and drained before anything is enqueued.
            let mut ds = self.stream.clone_htod(&arrays.s0)?;
            let mut dout = self.stream.alloc_zeros::<half::bf16>(m * heads * dv)?;
            self.stream.synchronize()?;
            launch(&perturb_stream, &mut ds_filler, &dout_filler)?;
            launch(&perturb_stream, &mut ds_filler, &dout_filler)?;
            launch(&self.stream, &mut ds, &mut dout)?;
            self.stream.synchronize()?;
            perturb_stream.synchronize()?;
            let out: Vec<half::bf16> = self.stream.clone_dtoh(&dout)?;
            let state: Vec<f32> = self.stream.clone_dtoh(&ds)?;
            let out_bytes: Vec<u8> = out.iter().flat_map(|b| b.to_ne_bytes()).collect();
            let state_bytes: Vec<u8> = state.iter().flat_map(|f| f.to_ne_bytes()).collect();
            out_digests.push(sha256_bytes(&out_bytes));
            state_digests.push(sha256_bytes(&state_bytes));
            eprintln!(
                "    run {run}/{runs} out {} state {}",
                &out_digests[out_digests.len() - 1][..16],
                &state_digests[state_digests.len() - 1][..16]
            );
        }
        Ok((entry, out_digests, state_digests))
    }

    fn meta(&self, fx: &Fixture) -> Result<cudarc::driver::CudaSlice<i32>> {
        // The kernel declares `struct GdnMeta { int prefix, valid_rows, key_count,
        // capacity, causal, r20, r24, r28; }`; eight i32 in that order.
        let key_count = fx.prefix as i32 + fx.m as i32;
        Ok(self.stream.clone_htod(&[
            fx.prefix as i32,
            fx.m as i32,
            key_count,
            key_count,
            1,
            0,
            0,
            0,
        ])?)
    }

    fn time(&self, which: Which, fx: &Fixture, arrays: &CaseArrays, s: u32) -> Result<f64> {
        use cudarc::driver::{LaunchConfig, PushKernelArg as _, sys};
        let m = fx.m;
        let heads = fx.arrays["q"].shape[1];
        let dv = fx.arrays["v"].shape[2];
        // Shared locals are FP32 as the fixture holds them; the P2 arm derives its BF16 sources in its match arm.
        let dq = self.stream.clone_htod(&arrays.q)?;
        let dk = self.stream.clone_htod(&arrays.k)?;
        let dvv = self.stream.clone_htod(&arrays.v)?;
        let dbeta = self.stream.clone_htod(&arrays.beta)?;
        let dg = self.stream.clone_htod(&arrays.g)?;
        let mut ds = self.stream.clone_htod(&arrays.s0)?;
        let mut dout = self.stream.alloc_zeros::<half::bf16>(m * heads * dv)?;
        let meta = self.meta(fx)?;
        let ws = if which == Which::P2 {
            Some(self.p2_workspace(m)?)
        } else {
            None
        };
        // The P2 arm's BF16 sources are uploaded before the event pair.
        let p2_sources = if which == Which::P2 {
            let (q_bf16, k_bf16, v_bf16) = derive_bf16_sources(&arrays.q, &arrays.k, &arrays.v)?;
            Some((
                self.stream.clone_htod(&q_bf16)?,
                self.stream.clone_htod(&k_bf16)?,
                self.stream.clone_htod(&v_bf16)?,
            ))
        } else {
            None
        };
        self.stream.synchronize()?;
        // Only the launch is enqueued between the events; the stream is synchronised before reading them.
        let start = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&self.stream)?;
        unsafe {
            match which {
                Which::Main => {
                    self.stream
                        .launch_builder(&self.main)
                        .arg(&dq)
                        .arg(&dk)
                        .arg(&dvv)
                        .arg(&dbeta)
                        .arg(&dg)
                        .arg(&mut ds)
                        .arg(&dout)
                        .arg(&meta)
                        .arg(&0i32)
                        .launch(LaunchConfig {
                            grid_dim: (1, heads as u32, 1),
                            block_dim: (Self::MAIN_THREADS, 1, 1),
                            shared_mem_bytes: Self::MAIN_SMEM,
                        })?;
                }
                Which::P2 => {
                    let ws = ws.as_ref().context("the P2 arm allocates its workspace")?;
                    let (dq, dk, dvv) = p2_sources
                        .as_ref()
                        .context("P2 timing without its derived sources")?;
                    let mut case = P2Case {
                        dq,
                        dk,
                        dvv,
                        dbeta: &dbeta,
                        dg: &dg,
                        ds: &mut ds,
                        dout: &mut dout,
                        meta: &meta,
                        ws,
                        m,
                        heads: heads as u32,
                    };
                    self.launch_p2(&self.stream, &mut case)?;
                }
                Which::P1 => {
                    let idx = match s {
                        4 => 0,
                        8 => 1,
                        16 => 2,
                        _ => anyhow::bail!("split {s} is not 4/8/16"),
                    };
                    self.stream
                        .launch_builder(&self.p1[idx])
                        .arg(&dq)
                        .arg(&dk)
                        .arg(&dvv)
                        .arg(&dbeta)
                        .arg(&dg)
                        .arg(&mut ds)
                        .arg(&dout)
                        .arg(&meta)
                        .arg(&0i32)
                        .launch(LaunchConfig {
                            grid_dim: (s, heads as u32, 1),
                            block_dim: (Self::P1_THREADS, 1, 1),
                            shared_mem_bytes: split_smem(s)?,
                        })?;
                }
            }
        }
        let end = self
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.record(&self.stream)?;
        self.stream.synchronize()?;
        let ms = start.elapsed_ms(&end)? as f64;
        ensure!(ms.is_finite() && ms > 0.0, "event pair gave {ms} ms");
        Ok(ms)
    }
}

#[cfg(feature = "cuda")]
fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// The frozen per-fixture input arrays as one argument.
struct CaseArrays {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    beta: Vec<f32>,
    g: Vec<f32>,
    s0: Vec<f32>,
}

impl CaseArrays {
    fn new(root: &Path, fx: &Fixture) -> Result<Self> {
        Ok(Self {
            q: read_f32(root, &fx.arrays["q"])?,
            k: read_f32(root, &fx.arrays["k"])?,
            v: read_f32(root, &fx.arrays["v"])?,
            beta: read_f32(root, &fx.arrays["beta"])?,
            g: read_f32(root, &fx.arrays["g"])?,
            s0: read_f32(root, &fx.arrays["S0"])?,
        })
    }
}

fn check_identity_args(args: &IdentityArgs) -> Result<()> {
    ensure!(
        args.c == KERNEL_C,
        "the built P1 core is C{KERNEL_C} only; C{} is not built and is not run \
         (protocol §A fixes C64 for every candidate)",
        args.c
    );
    ensure!(
        args.runs >= IDENTITY_MIN_RUNS,
        "the identity gate contract requires at least {IDENTITY_MIN_RUNS} launches per \
         row; {} is a rehearsal and cannot carry a verdict",
        args.runs
    );
    for s in &route_splits(args.route, &args.s) {
        ensure!(
            [4u32, 8, 16].contains(s),
            "split {s} is not one of 4/8/16 (protocol §A)"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_identity(args: &IdentityArgs) -> Result<()> {
    check_identity_args(args)?;
    let pkg = open_package(&args.package)?;
    std::fs::create_dir_all(&args.run_dir)
        .with_context(|| format!("create {}", args.run_dir.display()))?;
    let fixtures = select_fixtures(&pkg, &args.case)?;
    let gpu = Gpu::open(args.route)?;
    let device = device_of(Some(&gpu));
    route_policy(args.route, &gpu.ctx)?;
    let splits = route_splits(args.route, &args.s);

    let mut rows = Vec::new();
    for fx in &fixtures {
        check_fixture_shapes(fx)?;
        let arrays = CaseArrays::new(&pkg.root, fx)?;
        for &s in &splits {
            let (entry, out_digests, state_digests) = match args.route {
                Route::P1 => gpu.identity_p1(fx, &arrays, s, args.runs)?,
                Route::P2 => gpu.identity_p2(fx, &arrays, args.runs)?,
            };
            let distinct_out = out_digests
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let distinct_state = state_digests
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let first_divergent = out_digests
                .iter()
                .position(|d| d != &out_digests[0])
                .or_else(|| state_digests.iter().position(|d| d != &state_digests[0]))
                .map(|i| u32::try_from(i).unwrap_or(u32::MAX) + 1);
            let identical = distinct_out == 1 && distinct_state == 1;
            let qualified = qualified_topology(device.as_ref());
            eprintln!(
                "  {:16} C{} s{s:<2} {} out={} state={} {}",
                fx.name,
                args.c,
                entry,
                distinct_out,
                distinct_state,
                if identical { "IDENTICAL" } else { "DIVERGED" }
            );
            rows.push(IdentityRow {
                case: fx.name.clone(),
                c: args.c,
                s,
                entry: entry.to_string(),
                arm: "production",
                runs: args.runs,
                out_sha256: out_digests[0].clone(),
                state_sha256: state_digests[0].clone(),
                distinct_out_sha256: distinct_out,
                distinct_state_sha256: distinct_state,
                first_divergent_run: first_divergent,
                run_out_sha256: out_digests,
                run_state_sha256: state_digests,
                verdict: if !qualified {
                    "INVALID"
                } else if identical {
                    "PASS"
                } else {
                    "FAIL"
                },
            });
        }
    }
    let verdict = if !qualified_topology(device.as_ref()) {
        "INVALID"
    } else if rows.iter().all(|r| r.verdict == "PASS") {
        "PASS"
    } else {
        "FAIL"
    };
    let report = IdentityReport {
        schema: SCHEMA_IDENTITY,
        generated_utc: now_utc(),
        clock: "cuda",
        package: args.package.display().to_string(),
        manifest_generated_utc: pkg.manifest.generated_utc.clone(),
        files_sha256_verified: pkg.verified,
        perturbation: PERTURBATION,
        environment: env_receipt(false, device, true),
        rows,
        verdict,
    };
    let path = args.run_dir.join("IDENTITY.json");
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(&path, text.as_bytes()).with_context(|| format!("write {}", path.display()))?;
    eprintln!(
        "[identity] {} — {} rows, verdict {verdict}",
        path.display(),
        report.rows.len()
    );
    // The report is written for every verdict; the exit code is non-zero only for plumbing failures.
    Ok(())
}

/// Runs between two measured launches on throwaway buffers.
#[cfg(feature = "cuda")]
const PERTURBATION: &str = "two full-length launches of the same entry on a second stream over throwaway S and out buffers, enqueued before the measured launch on the main stream and not joined until after it, then synchronize(stream) + synchronize(second stream), before every measured launch is read";

#[cfg(not(feature = "cuda"))]
fn run_identity(args: &IdentityArgs) -> Result<()> {
    check_identity_args(args)?;
    anyhow::bail!("the identity phase needs the crate's cuda feature")
}

fn main() -> Result<()> {
    // Arguments are validated before any device or topology query.
    let cli = parse_and_validate(std::env::args_os())?;
    // The route is logged before anything else.
    eprintln!("[route] {:?}", cli.phase_route());
    match cli.phase {
        Phase::Numerics(a) => run_numerics(&a),
        Phase::Timing(a) => run_timing(&a),
        Phase::Identity(a) => run_identity(&a),
        Phase::E2e(a) => run_e2e(&a),
    }
}

/// Positions the e2e phase reserves: tokens + max(decode steps, the fixed warm-up).
fn e2e_positions(tokens: usize, decode_steps: usize) -> Result<usize> {
    tokens
        .checked_add(decode_steps.max(E2E_WARMUP_STEPS))
        .context("token count + warm-up/decode steps overflows usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `blocks` ABBA blocks of constant per-arm durations with a fixed +/-0.4% jitter, so the CI has width.
    fn series_for(blocks: u32, a_ms: f64, b_ms: f64) -> LegSeries {
        let blocks = abba(
            Clock::Stub,
            blocks,
            Which::Main,
            Which::P1,
            move |w, leg| {
                let base = match w {
                    Which::Main => a_ms,
                    Which::P1 | Which::P2 => b_ms,
                };
                Ok(base * (1.0 + if leg % 2 == 0 { -0.004 } else { 0.004 }))
            },
        );
        summarise(&blocks, blocks.len() * 4, "P1 core vs the main GDN core")
    }

    #[test]
    fn a_p1_ten_percent_faster_clears_and_a_p1_ten_percent_slower_does_not() {
        let faster = timing_verdict(&series_for(TIMING_BLOCKS, 1.0, 0.9), 0.03);
        assert_eq!(faster.0, "PASS", "a 10% faster P1 must clear: {faster:?}");
        let slower = timing_verdict(&series_for(TIMING_BLOCKS, 1.0, 1.1), 0.03);
        assert_eq!(
            slower.0, "NOT_CLEARED",
            "a 10% slower P1 must not clear: {slower:?}"
        );
    }

    /// `--route` defaults to p1, accepts p2 and rejects anything else.
    #[test]
    fn the_route_flag_defaults_to_p1_and_takes_only_p1_or_p2() {
        let numerics = |extra: &[&str]| -> Result<Route> {
            let mut argv = vec![
                "gdn_p1_runner",
                "numerics",
                "--package",
                "p",
                "--run-dir",
                "d",
            ];
            argv.extend_from_slice(extra);
            Ok(Cli::try_parse_from(argv)?.phase_route())
        };
        assert_eq!(numerics(&[]).unwrap(), Route::P1, "numerics defaults to p1");
        assert_eq!(numerics(&["--route", "p1"]).unwrap(), Route::P1);
        assert_eq!(numerics(&["--route", "p2"]).unwrap(), Route::P2);
        let bad = numerics(&["--route", "p3"]).expect_err("an unknown route must be rejected");
        assert!(
            bad.to_string().contains("p3"),
            "the error must name the route it refused: {bad}"
        );

        let identity = |extra: &[&str]| -> Result<Route> {
            let mut argv = vec![
                "gdn_p1_runner",
                "identity",
                "--package",
                "p",
                "--run-dir",
                "d",
            ];
            argv.extend_from_slice(extra);
            Ok(Cli::try_parse_from(argv)?.phase_route())
        };
        assert_eq!(identity(&[]).unwrap(), Route::P1, "identity defaults to p1");
        assert_eq!(identity(&["--route", "p2"]).unwrap(), Route::P2);
        assert!(identity(&["--route", "fused"]).is_err());

        let timing = |extra: &[&str]| -> Result<Route> {
            let mut argv = vec!["gdn_p1_runner", "timing", "--run-dir", "d", "--rows", "64"];
            argv.extend_from_slice(extra);
            Ok(Cli::try_parse_from(argv)?.phase_route())
        };
        assert_eq!(timing(&[]).unwrap(), Route::P1, "timing defaults to p1");
        assert_eq!(timing(&["--route", "p2"]).unwrap(), Route::P2);
        assert!(
            timing(&["--route", "P2"]).is_err(),
            "the value is lower-case only"
        );
    }

    /// Under p2 every `--s` collapses to the s4 tag.
    #[test]
    fn the_p2_route_runs_one_configuration_however_s_is_given() {
        assert_eq!(
            route_splits(Route::P2, &[8, 4, 16]),
            vec![P2_SPLIT_TAG],
            "p2 ignores --s"
        );
        assert_eq!(route_splits(Route::P2, &[]), vec![P2_SPLIT_TAG]);
        assert_eq!(route_splits(Route::P1, &[8, 4, 16]), vec![8, 4, 16]);
        assert_eq!(route_splits(Route::P1, &[]), Vec::<u32>::new());
        assert_eq!(P2_SPLIT_TAG, 4, "the P2 strip is P1's s4");
    }

    /// Route p2 sizes its slot from the planner and refuses an unqualified device.
    #[test]
    fn the_p2_route_needs_the_qualified_topology_and_the_planner_sized_slot() {
        let qualified = DeviceInfo {
            name: "gate".into(),
            compute_capability: 89,
            sm_count: 128,
        };
        let other = DeviceInfo {
            name: "other".into(),
            compute_capability: 86,
            sm_count: 48,
        };
        // On the qualified topology the P2 policy carries the split.
        assert!(
            require_qualified_topology(Route::P2, Ok(qualified.clone()))
                .unwrap()
                .cw_p2_for_step(flyingfish::gdn_plan::MIN_VERIFY_ROWS + 1)
        );
        let err = require_qualified_topology(Route::P2, Ok(other.clone()))
            .expect_err("an unqualified device must not run the split");
        assert!(
            err.to_string().contains("qualified topology"),
            "the error must name the condition: {err}"
        );
        // Route p1 carries the device policy and selects no P1 split.
        assert_eq!(
            require_qualified_topology(Route::P1, Ok(qualified.clone()))
                .unwrap()
                .prefill_p1_split(),
            0
        );
        assert_eq!(
            require_qualified_topology(Route::P1, Ok(other.clone()))
                .unwrap()
                .prefill_p1_split(),
            0
        );
    }

    /// A failed attribute query is a refusal on both routes and names the query.
    #[test]
    fn a_failed_attribute_query_refuses_the_topology_guard() {
        let failed = || {
            Err(anyhow::anyhow!(
                "cuDeviceGetAttribute returned CUDA_ERROR_800"
            ))
        };
        for route in [Route::P1, Route::P2] {
            let err = require_qualified_topology(route, failed())
                .expect_err("a failed attribute query must refuse");
            let text = err.to_string();
            assert!(
                text.contains("cuDeviceGetAttribute"),
                "the refusal must name the query that failed: {text}"
            );
            assert!(
                !text.contains("no CUDA-capable device"),
                "a failed attribute query is not a missing device: {text}"
            );
        }
        // the chain still names the driver's own error underneath
        let err = require_qualified_topology(Route::P2, failed()).unwrap_err();
        assert!(
            format!("{err:#}").contains("CUDA_ERROR_800"),
            "the driver's error must survive into the chain: {err:#}"
        );
    }

    /// Each control drops one required argument and must fail on that argument, never on a device.
    #[test]
    fn a_missing_required_argument_fails_in_parsing_not_on_a_device() {
        let run = |argv: &[&str]| parse_and_validate(argv.to_vec()).map(|_| ());
        // numerics without --package, identity without --run-dir, e2e without --model
        for (phase, drop_arg, name) in [
            (
                vec!["gdn_p1_runner", "numerics", "--run-dir", "d"],
                "--package",
                "package",
            ),
            (
                vec![
                    "gdn_p1_runner",
                    "identity",
                    "--package",
                    "p",
                    "--runs",
                    "20",
                ],
                "--run-dir",
                "run-dir",
            ),
            (
                vec![
                    "gdn_p1_runner",
                    "e2e",
                    "--run-dir",
                    "d",
                    "--output",
                    "o",
                    "--label",
                    "l",
                    "--tokens-file",
                    "t",
                ],
                "--model",
                "model",
            ),
        ] {
            let err = run(&phase).expect_err("a missing required argument must fail");
            assert!(
                err.to_string().contains(name),
                "{:?} without {drop_arg} must name {name}: {err}",
                phase[1]
            );
        }
    }

    /// Value checks run before any device query.
    #[test]
    fn an_out_of_contract_value_fails_in_validation_not_on_a_device() {
        let err = parse_and_validate(vec![
            "gdn_p1_runner",
            "identity",
            "--package",
            "p",
            "--run-dir",
            "d",
            "--runs",
            "3",
        ])
        .map(|_| ())
        .expect_err("3 runs is a rehearsal and cannot carry a verdict");
        assert!(
            err.to_string().contains("runs"),
            "the error must name the argument: {err}"
        );
        let err = parse_and_validate(vec![
            "gdn_p1_runner",
            "numerics",
            "--package",
            "p",
            "--run-dir",
            "d",
            "--c",
            "32",
        ])
        .map(|_| ())
        .expect_err("the built core is C64 only");
        assert!(err.to_string().contains("C64"), "{err}");
    }

    /// The P2 slot is the planner's number: one (chunk, head) slot per 64 rows.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_p2_slot_is_the_planners_number_per_chunk_and_head() {
        assert_eq!(route_workspace_bytes(Route::P1, 2048).unwrap(), 0);
        for rows in [1u32, 64, 2048, 8192] {
            let slot = route_workspace_bytes(Route::P2, rows).unwrap();
            assert_eq!(
                slot,
                flyingfish::gdn_plan::cw_p2_bytes(48, rows),
                "rows={rows}"
            );
            assert_eq!(
                slot,
                u64::from(rows.div_ceil(64)) * 48 * flyingfish::gdn_plan::cw_p2::SLOT_BYTES,
                "rows={rows}"
            );
        }
    }

    #[test]
    fn the_e2e_reservation_counts_the_fixed_warm_up() {
        // 16 tokens with 2 measured steps: the 4 warm-up steps are the binding term.
        assert_eq!(e2e_positions(16, 2).unwrap(), 20);
        assert_eq!(e2e_positions(16, 8).unwrap(), 24);
        assert!(e2e_positions(16, 2).unwrap() > 16 + 2);
        assert!(e2e_positions(1, usize::MAX).is_err());
    }

    #[test]
    fn the_synthesised_beta_and_g_sit_inside_their_production_domains() {
        for rows in [1usize, 64, 1021] {
            let (_, arrays) = synth_case(rows);
            assert!(
                arrays.beta.iter().all(|b| (0.0..=1.0).contains(b)),
                "beta left (0,1) at {rows} rows"
            );
            assert!(
                arrays.g.iter().all(|v| *v <= 0.0),
                "g went positive at {rows} rows"
            );
            assert!(
                arrays.g.iter().any(|v| *v < 0.0),
                "g is all zero at {rows} rows"
            );
            assert!(
                arrays.q.iter().any(|v| *v < 0.0) && arrays.q.iter().any(|v| *v > 0.0),
                "q lost its signed range at {rows} rows"
            );
        }
    }

    #[test]
    fn the_quantisation_floor_uses_the_shortest_individual_leg() {
        // One block whose per-arm MEANS are equal (2.5) but whose legs include a 1.0:
        // the floor's min must be 1.0, not the mean 2.5.
        let blocks = abba(Clock::Stub, 1, Which::Main, Which::P1, |w, leg| {
            Ok(match (w, leg) {
                (Which::Main, 0) => 1.0,
                (Which::Main, _) => 4.0,
                (Which::P1, 1) => 2.0,
                (Which::P1 | Which::P2, _) => 3.0,
            })
        });
        assert_eq!(blocks[0].a_ms, 2.5);
        assert_eq!(blocks[0].b_ms, 2.5);
        assert!((blocks[0].min_leg_ms - 1.0).abs() < 1e-12);
        // freeze_null_band's quantisation term must derive from that 1.0, not 2.5.
        let dir = std::env::temp_dir().join(format!("qfloor-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let band = freeze_null_band(&blocks, &dir.join("NULL.json")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let want = 0.5 / 1000.0 / 1.0;
        assert!(
            (band.quantisation_term - want).abs() < 1e-15,
            "quantisation {} is not over the shortest leg ({}), it used something else",
            band.quantisation_term,
            want
        );
    }

    #[test]
    fn a_zero_baseline_with_a_non_zero_candidate_has_no_json_ratio() {
        assert_eq!(gate_ratio(0.0, 0.0), Some(0.0));
        assert_eq!(gate_ratio(1.5, 1.0), Some(1.5));
        assert_eq!(gate_ratio(0.5, 1.0), Some(0.5));
        let r = gate_ratio(3.0, 0.0);
        assert_eq!(
            r, None,
            "a zero baseline and a non-zero candidate has no ratio"
        );
        // serde_json writes a non-finite f64 as null.
        let row = serde_json::json!({ "ratio_state_rel_rms": r });
        assert_eq!(
            serde_json::to_string(&row).unwrap(),
            "{\"ratio_state_rel_rms\":null}"
        );
        let bare = serde_json::json!({ "ratio_state_rel_rms": f64::INFINITY });
        assert_eq!(
            serde_json::to_string(&bare).unwrap(),
            "{\"ratio_state_rel_rms\":null}"
        );
    }

    #[test]
    fn the_block_statistic_is_the_mean_of_the_paired_log_ratios() {
        // One block, A B B A: main 1.0 then 4.0, P1 2.0 then 1.0; the paired log ratios are log(2/1) and log(1/4), and log(mean B / mean A) is a different number.
        let blocks = abba(Clock::Stub, 1, Which::Main, Which::P1, |w, leg| {
            Ok(match (w, leg) {
                (Which::Main, 0) => 1.0,
                (Which::Main, _) => 4.0,
                (Which::P1 | Which::P2, 1) => 2.0,
                (Which::P1 | Which::P2, _) => 1.0,
            })
        });
        let paired = (2.0f64 / 1.0).ln() + (1.0f64 / 4.0).ln();
        let mean_of_ratios = (1.5f64 / 2.5).ln();
        assert!(
            (blocks[0].log_ratio - paired / 2.0).abs() < 1e-12,
            "block statistic {} is not the mean of the paired log ratios {}",
            blocks[0].log_ratio,
            paired / 2.0
        );
        assert!(
            (mean_of_ratios - paired / 2.0).abs() > 1e-3,
            "the test case does not separate the two statistics"
        );
    }

    fn test_fixture() -> Fixture {
        Fixture {
            name: "c1".into(),
            kind: "margin".into(),
            note: String::new(),
            m: 1,
            prefix: 0,
            seed: None,
            arrays: BTreeMap::new(),
        }
    }

    fn basis_row() -> BasisRow {
        BasisRow {
            case: "c1".into(),
            c: 64,
            s: 8,
            out_max_abs: 1e-4,
            out_rel_rms: 1e-3,
            state_rel_rms: 1e-4,
            frac_le_1ulp: 0.65,
            gate_out_max_abs: 1.25e-4,
            gate_out_rel_rms: 1.25e-3,
            gate_state_rel_rms: 1.25e-4,
            frac_floor: 0.63,
            finite: true,
            input_sha256: String::new(),
        }
    }

    #[test]
    fn a_negative_or_out_of_range_basis_metric_is_rejected() {
        let fx = test_fixture();
        let m0 = Metrics {
            max_abs: 0.0,
            rel_rms: 0.0,
            frac_le_1ulp: 1.0,
            finite: true,
        };
        for (field, bad) in [
            ("out_max_abs", -1e-4),
            ("out_rel_rms", -1e-3),
            ("state_rel_rms", -1e-4),
        ] {
            let mut b = basis_row();
            match field {
                "out_max_abs" => {
                    b.out_max_abs = bad;
                    b.gate_out_max_abs = bad * GATE_BAND;
                }
                "out_rel_rms" => {
                    b.out_rel_rms = bad;
                    b.gate_out_rel_rms = bad * GATE_BAND;
                }
                _ => {
                    b.state_rel_rms = bad;
                    b.gate_state_rel_rms = bad * GATE_BAND;
                }
            }
            let m0 = Metrics {
                max_abs: 0.0,
                rel_rms: 0.0,
                frac_le_1ulp: 1.0,
                finite: true,
            };
            let err = gate_row(&fx, 64, 8, &b, &m0, &m0, 48)
                .map(|_| ())
                .expect_err("a negative basis metric must be rejected");
            assert!(
                err.to_string().contains(field),
                "the error for {field} must name the field: {err}"
            );
        }
        for bad in [-0.01f64, 1.4, f64::NAN] {
            let mut b = basis_row();
            b.frac_le_1ulp = bad;
            assert!(
                gate_row(&fx, 64, 8, &b, &m0, &m0, 48).is_err(),
                "frac_le_1ulp {bad} must be rejected"
            );
        }
        assert!(
            gate_row(&fx, 64, 8, &basis_row(), &m0, &m0, 48).is_ok(),
            "the well-formed row must still pass"
        );
    }
}
