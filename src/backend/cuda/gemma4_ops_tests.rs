//! Gemma 4 per-operation GPU gates against fixtures written by the model definition.
//! Fixture directory: FF_GEMMA4_FIXTURES.
use super::{Device, cubin, ops::flat};
use crate::backend::gpu::{LaunchConfig, PushKernelArg};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::SafeTensors;
use std::collections::BTreeMap;

/// Frozen before any candidate run (gemma/reports/agy.md, PLAN G4-S2a §4).
const MAX_ULP: u32 = 1;
const BIT_EQUAL_MIN: f64 = 0.999;
const MUTANT_ULP: u32 = 4;
const MUTANT_FRACTION: f64 = 0.01;

type Fixture = BTreeMap<String, (Vec<usize>, Vec<bf16>)>;

#[test]
#[ignore = "requires a GPU"]
fn shared_kv_preparation_matches_independent_raw_projection_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let module = cubin::module(&device.ctx, "rope")?;
    let independent = module.load_function("qknorm_rope_table_kv")?;
    let shared = module.load_function("qknorm_rope_table_shared_kv")?;
    let mut cases = 0;
    for (hq, hkv) in [(16usize, 1usize), (16, 2), (32, 4)] {
        for d in [256usize, 512] {
            for rows in [1usize, 17, 1032] {
                let capacity = rows + 35;
                let q_width = hq * d;
                let kv_width = hkv * d;
                let raw: Vec<_> = (0..rows * (q_width + kv_width))
                    .map(|i| bf16::from_f32(((i * 37 % 1009) as f32 - 504.0) / 128.0))
                    .collect();
                let mut copied = raw.clone();
                copied.extend_from_slice(&raw[rows * q_width..]);
                let norms: Vec<_> = (0..d)
                    .map(|i| bf16::from_f32(0.25 + (i * 17 % 101) as f32 / 32.0))
                    .collect();
                let norms = s.clone_htod(&norms)?;
                let rope = crate::model_spec::RopeSpec {
                    theta: if d == 512 { 1_000_000.0 } else { 10_000.0 },
                    pairs: if d == 512 { 64 } else { 128 },
                    pair_stride: d as u32 / 2,
                    denominator: d as u32,
                };
                let table = s.clone_htod(&rope.table(d as u32, capacity)?)?;
                let start = s.clone_htod(&[13i32])?;
                let run = |shared_input: bool| -> Result<_> {
                    let mut qkv = s.clone_htod(if shared_input { &raw } else { &copied })?;
                    let poison = vec![bf16::from_f32(-71.0); hkv * capacity * d];
                    let mut k = s.clone_htod(&poison)?;
                    let mut v = s.clone_htod(&poison)?;
                    unsafe {
                        s.launch_builder(if shared_input { &shared } else { &independent })
                            .arg(&mut qkv)
                            .arg(&mut k)
                            .arg(&mut v)
                            .arg(&norms)
                            .arg(&norms)
                            .arg(&table)
                            .arg(&start)
                            .arg(&-1i32)
                            .arg(&(rows as i32))
                            .arg(&(hq as i32))
                            .arg(&(hkv as i32))
                            .arg(&(d as i32))
                            .arg(&(capacity as i32))
                            .arg(&0i32)
                            .arg(&1e-6f32)
                            .arg(&(q_width as u64))
                            .arg(&((rows * q_width) as u64))
                            .arg(&(kv_width as u64))
                            .arg(&((rows * (q_width + kv_width)) as u64))
                            .arg(&(kv_width as u64))
                            .launch(LaunchConfig {
                                grid_dim: (
                                    (rows * (hq + if shared_input { hkv } else { 2 * hkv })) as u32,
                                    1,
                                    1,
                                ),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    Ok((
                        s.clone_dtoh(&qkv.slice(..raw.len()))?,
                        s.clone_dtoh(&k)?,
                        s.clone_dtoh(&v)?,
                    ))
                };
                let expected = run(false)?;
                let actual = run(true)?;
                let repeat = run(true)?;
                ensure!(
                    actual == expected && repeat == actual,
                    "shared KV differs rows={rows} hq={hq} hkv={hkv} d={d}"
                );
                ensure!(
                    actual.1 != actual.2,
                    "normalized/rotated K-as-V control undetected"
                );
                for head in 0..hkv {
                    for position in (0..13).chain(13 + rows..capacity) {
                        let begin = (head * capacity + position) * d;
                        ensure!(
                            actual.1[begin..begin + d]
                                .iter()
                                .chain(&actual.2[begin..begin + d])
                                .all(|x| *x == bf16::from_f32(-71.0)),
                            "cache guard overwritten"
                        );
                    }
                }
                println!(
                    "SHARED_KV_PREP rows={rows} hq={hq} hkv={hkv} d={d} bit_mismatches=0 repeat_mismatches=0 wrong_value_control=detected guards=PASS"
                );
                cases += 1;
            }
        }
    }
    println!("PASS shared_kv_preparation cells={cases} defects=0");
    Ok(())
}

#[test]
#[ignore = "requires a GPU"]
fn per_layer_kv_state_and_checkpoint_preserve_heads_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    for (text, bytes) in [
        (
            include_str!("../../../tests/data/gemma4-12b-config.json"),
            369_098_752usize,
        ),
        (
            include_str!("../../../tests/data/gemma4-31b-config.json"),
            1_006_632_960,
        ),
    ] {
        let spec = crate::model_spec::ModelSpec::gemma4(
            &serde_json::from_str(text)?,
            &serde_json::Value::Null,
        )?;
        let mut state = super::state::StateStore::new(&spec, 2048, s)?;
        ensure!(
            state.bytes() == bytes + 32,
            "per-layer state byte accounting"
        );
        state.set_sequence(s, 1022, 8, 8)?;
        let mut token = s.clone_htod(&[17u32])?;
        for (index, kv) in state.kv.iter_mut().enumerate() {
            let g = spec.layers[index].geometry.context("layer geometry")?;
            let heads = g.kv_heads.unwrap_or(spec.kv_heads) as usize;
            let dim = g.head_dim as usize;
            let capacity = spec.layers[index]
                .sliding_window
                .map_or(2048, |w| w as usize);
            ensure!(
                kv.k.len() == heads * capacity * dim && kv.v.len() == kv.k.len(),
                "layer {index} allocation"
            );
            for head in 0..heads {
                for row in 0..8 {
                    let offset = (head * capacity + (1022 + row) % capacity) * dim;
                    let values: Vec<_> = (0..dim)
                        .map(|d| {
                            bf16::from_f32(
                                ((index * 17 + head * 41 + row * 29 + d * 7) % 1013) as f32 / 64.0,
                            )
                        })
                        .collect();
                    s.memcpy_htod(&values, &mut kv.k.slice_mut(offset..offset + dim))?;
                    s.memcpy_htod(&values, &mut kv.v.slice_mut(offset..offset + dim))?;
                }
            }
        }
        let mut checkpoint = super::state::StateCheckpoint::new(&state, &spec, 8, s)?;
        checkpoint.save(&state, &token, 1022, s, None)?;
        let mut wrong = spec.clone();
        for layer in &mut wrong.layers {
            layer.geometry.as_mut().context("layer geometry")?.kv_heads = None;
        }
        let mut wrong_checkpoint = super::state::StateCheckpoint::new(&state, &wrong, 8, s)?;
        ensure!(
            wrong_checkpoint
                .save(&state, &token, 1022, s, None)
                .is_err(),
            "model-wide KV-head control undetected"
        );
        for kv in &mut state.kv {
            s.memset_zeros(&mut kv.k)?;
            s.memset_zeros(&mut kv.v)?;
        }
        checkpoint.restore(&mut state, &mut token, s)?;
        for (index, kv) in state.kv.iter().enumerate() {
            let g = spec.layers[index].geometry.context("layer geometry")?;
            let heads = g.kv_heads.unwrap_or(spec.kv_heads) as usize;
            let dim = g.head_dim as usize;
            let capacity = spec.layers[index]
                .sliding_window
                .map_or(2048, |w| w as usize);
            for head in 0..heads {
                for row in 0..8 {
                    let offset = (head * capacity + (1022 + row) % capacity) * dim;
                    let expected: Vec<_> = (0..dim)
                        .map(|d| {
                            bf16::from_f32(
                                ((index * 17 + head * 41 + row * 29 + d * 7) % 1013) as f32 / 64.0,
                            )
                        })
                        .collect();
                    ensure!(
                        s.clone_dtoh(&kv.k.slice(offset..offset + dim))? == expected
                            && s.clone_dtoh(&kv.v.slice(offset..offset + dim))? == expected,
                        "snapshot head/row mismatch"
                    );
                }
            }
        }
        println!(
            "PASS per_layer_kv_state hidden={} cache_bytes={bytes} checkpoint_bits=0 wrong_heads_control=detected",
            spec.hidden
        );
    }
    Ok(())
}

fn load_fixture(path: &std::path::Path) -> Result<Fixture> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    tensors
        .iter()
        .map(|(name, view)| {
            ensure!(
                view.dtype() == safetensors::Dtype::BF16,
                "{name} is not BF16"
            );
            let values = view
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect();
            Ok((name.to_owned(), (view.shape().to_vec(), values)))
        })
        .collect()
}

fn fixture(file: &str) -> Result<Fixture> {
    let dir = std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?;
    load_fixture(&std::path::Path::new(&dir).join(file))
}

fn bf16_ulp(v: f32) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        2f32.powi(v.abs().log2().floor() as i32 - 7)
    }
}

fn ordered(x: bf16) -> i32 {
    let bits = i32::from(x.to_bits());
    if bits & 0x8000 != 0 {
        -(bits & 0x7fff)
    } else {
        bits
    }
}

struct Stats {
    elements: usize,
    max_ulp: u32,
    bit_equal: usize,
    strict_equal: usize,
    beyond_mutant: usize,
    exempt: usize,
    worst: Option<(usize, f32, f32)>,
}

impl Stats {
    fn new(candidate: &[bf16], reference: &[bf16]) -> Result<Self> {
        Self::with_floor(candidate, reference, None)
    }

    /// An element within `floor[i]` of the reference counts as matching.
    fn with_floor(candidate: &[bf16], reference: &[bf16], floor: Option<&[f32]>) -> Result<Self> {
        ensure!(
            candidate.len() == reference.len() && !candidate.is_empty(),
            "RUNNER_FAIL candidate and reference extents differ"
        );
        ensure!(
            candidate
                .iter()
                .chain(reference)
                .all(|x| x.to_f32().is_finite()),
            "RUNNER_FAIL non-finite value"
        );
        let mut stats = Self {
            elements: candidate.len(),
            max_ulp: 0,
            bit_equal: 0,
            strict_equal: 0,
            beyond_mutant: 0,
            exempt: 0,
            worst: None,
        };
        for (i, (c, r)) in candidate.iter().zip(reference).enumerate() {
            let ulp = ordered(*c).abs_diff(ordered(*r));
            let within = floor.is_some_and(|f| (c.to_f32() - r.to_f32()).abs() <= f[i]);
            stats.exempt += usize::from(within && ulp > 0);
            if !within {
                if ulp > stats.max_ulp {
                    stats.worst = Some((i, c.to_f32(), r.to_f32()));
                }
                stats.max_ulp = stats.max_ulp.max(ulp);
                stats.beyond_mutant += usize::from(ulp > MUTANT_ULP);
            }
            stats.bit_equal += usize::from(ulp == 0 || within);
            stats.strict_equal += usize::from(ulp == 0);
        }
        Ok(stats)
    }

    fn passes(&self) -> bool {
        self.max_ulp <= MAX_ULP && self.bit_equal as f64 / self.elements as f64 >= BIT_EQUAL_MIN
    }

    fn is_detected_mutant(&self) -> bool {
        self.beyond_mutant as f64 / self.elements as f64 >= MUTANT_FRACTION
    }

    fn line(&self, name: &str) -> String {
        format!(
            "{name} elements={} max_ulp={} worst={:?} bit_equal={:.6} strict_equal={} strict_fraction={:.6} exempt_within_floor={} exempt_fraction={:.6} beyond_{}ulp={:.6}",
            self.elements,
            self.max_ulp,
            self.worst,
            self.bit_equal as f64 / self.elements as f64,
            self.strict_equal,
            self.strict_equal as f64 / self.elements as f64,
            self.exempt,
            self.exempt as f64 / self.elements as f64,
            MUTANT_ULP,
            self.beyond_mutant as f64 / self.elements as f64
        )
    }
}

fn tensor<'a>(fixture: &'a Fixture, name: &str) -> Result<&'a [bf16]> {
    Ok(&fixture
        .get(name)
        .with_context(|| format!("fixture tensor {name} missing"))?
        .1)
}

/// Reference values keep their stored dtype. F32 tensors are the FP32 oracle and are
/// never rounded; BF16 tensors widen to F32 exactly.
enum ReferenceValues {
    Bf16(Vec<bf16>),
    F32(Vec<f32>),
}

impl ReferenceValues {
    fn as_bf16(&self) -> Option<&[bf16]> {
        match self {
            Self::Bf16(values) => Some(values),
            Self::F32(_) => None,
        }
    }

    fn as_f32(&self) -> Vec<f32> {
        match self {
            Self::Bf16(values) => values.iter().map(|value| value.to_f32()).collect(),
            Self::F32(values) => values.clone(),
        }
    }
}

type ReferenceFixture = BTreeMap<String, (Vec<usize>, ReferenceValues)>;

fn audio_output_matches(
    device: &Device,
    output: &crate::backend::gpu::CudaSlice<bf16>,
    expected: &[f32],
    bound: &serde_json::Value,
) -> Result<bool> {
    let host = device.stream.clone_dtoh(output)?;
    ensure!(
        host.len() == expected.len() + 32
            && host[expected.len()..].iter().all(|v| v.to_bits() == 0x7fc1),
        "RUNNER_FAIL audio output guard overwritten"
    );
    vision_distance(
        &host[..expected.len()]
            .iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        expected,
        bound,
    )
}

#[test]
#[ignore = "requires frozen cross-device audio operator fixtures and a GPU window"]
fn audio_operators_match_definition_gpu() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("fixtures missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("AUDIO_FROZEN.json"))?)?;
    ensure!(
        frozen["status"] == "FROZEN" && frozen["candidate_observations"] == 0,
        "RUNNER_FAIL audio reference freeze missing"
    );
    let device = Device::new(0)?;
    let stream = &device.stream;
    let pointwise = cubin::module(&device.ctx, "vision_ops")?;
    let patch = pointwise.load_function("masked_strided_patch_bf16")?;
    let activation = pointwise.load_function("activation_bf16")?;
    let conv = pointwise.load_function("causal_depthwise_conv_bf16")?;
    let branch =
        cubin::module(&device.ctx, "q38_common")?.load_function("rms_scaled_branch_add")?;
    let attention =
        cubin::module(&device.ctx, "attention")?.load_function("relative_window_attention_d128")?;
    let mut cells = 0;
    let mut defects = 0;
    for (name, cell) in frozen["cells"].as_object().context("audio cells")? {
        let fixture = load_reference(&directory.join(format!("{name}.safetensors")))?;
        if name == "audio-relative-table" {
            let actual = super::engine::audio_runtime::relative_positions(13, 1024)?
                .into_iter()
                .map(bf16::to_f32)
                .collect::<Vec<_>>();
            let expected = reference(&fixture, "positions")?.as_f32();
            let pass = vision_distance(&actual, &expected, &cell["bounds"]["positions"])?;
            let wrong = (0..13)
                .flat_map(|row| {
                    (0..1024).map(move |d| {
                        let exponent = (d % 512) as f32 * -(10000_f64.ln() / 511.0) as f32;
                        let phase = (12 - row) as f32 * exponent.exp();
                        bf16::from_f32(if d < 512 { phase.sin() } else { phase.cos() }).to_f32()
                    })
                })
                .collect::<Vec<_>>();
            let detected = !vision_distance(&wrong, &expected, &cell["bounds"]["positions"])?;
            cells += 2;
            defects += usize::from(!pass) + usize::from(!detected);
            println!(
                "{} audio_relative_table elements=13312 old_fp32_control={detected}",
                if pass && detected { "PASS" } else { "DEFECT" }
            );
            continue;
        }
        let upload =
            |name: &str| -> Result<_> { Ok(stream.clone_htod(reference_bf16(&fixture, name)?)?) };
        let output = |count: usize| -> Result<_> {
            Ok(stream.clone_htod(&vec![bf16::from_bits(0x7fc1); count + 32])?)
        };
        let compare =
            |key: &str, out: &crate::backend::gpu::CudaSlice<bf16>, mutant: bool| -> Result<bool> {
                let pass = audio_output_matches(
                    &device,
                    out,
                    &reference(&fixture, key)?.as_f32(),
                    &cell["bounds"][key],
                )?;
                println!(
                    "AUDIO_OP {name} {key} mutant={mutant} {}",
                    if pass != mutant { "PASS" } else { "DEFECT" }
                );
                Ok(pass != mutant)
            };
        let mask_host = reference_bf16(&fixture, "mask")?
            .iter()
            .map(|v| u8::from(v.to_f32() != 0.0))
            .collect::<Vec<_>>();
        let mask = stream.clone_htod(&mask_host)?;
        let x = upload("x")?;
        if name.starts_with("audio-patch-") {
            let [time, frequency, channels] = <[usize; 3]>::try_from(fixture["x"].0.clone())
                .map_err(|_| anyhow::anyhow!("patch shape"))?;
            let count = time.div_ceil(2) * frequency.div_ceil(2) * channels * 9;
            let out = output(count)?;
            let run = |mask: &crate::backend::gpu::CudaSlice<u8>| -> Result<()> {
                unsafe {
                    stream
                        .launch_builder(&patch)
                        .arg(&x)
                        .arg(mask)
                        .arg(&out)
                        .arg(&(time as u32))
                        .arg(&(frequency as u32))
                        .arg(&(channels as u32))
                        .launch(flat(count))?;
                }
                Ok(())
            };
            run(&mask)?;
            defects += usize::from(!compare("patches", &out, false)?);
            cells += 1;
            run(&stream.clone_htod(&vec![1u8; time])?)?;
            defects += usize::from(!compare("patches", &out, true)?);
            cells += 1;
            continue;
        }
        let rows = u32::try_from(cell["rows"].as_u64().context("audio rows")?)?;
        let count = rows as usize * 1024;
        for (key, input_key, width, n, mode) in [
            ("relu", "x", 1024u32, rows, 0i32),
            ("silu", "x", 1024, rows, 1),
            ("glu", "gate", 1024, rows, 2),
            ("softplus", "scale", 128, 1, 3),
        ] {
            let input = upload(input_key)?;
            let out = output(n as usize * width as usize)?;
            unsafe {
                stream
                    .launch_builder(&activation)
                    .arg(&input)
                    .arg(&out)
                    .arg(&n)
                    .arg(&width)
                    .arg(&mode)
                    .launch(flat(n as usize * width as usize))?;
            }
            defects += usize::from(!compare(key, &out, false)?);
            cells += 1;
            if key == "glu" {
                let mut swapped = reference_bf16(&fixture, "gate")?.to_vec();
                for row in swapped.as_chunks_mut::<2048>().0 {
                    row.rotate_left(1024);
                }
                let input = stream.clone_htod(&swapped)?;
                unsafe {
                    stream
                        .launch_builder(&activation)
                        .arg(&input)
                        .arg(&out)
                        .arg(&n)
                        .arg(&width)
                        .arg(&mode)
                        .launch(flat(n as usize * width as usize))?;
                }
                defects += usize::from(!compare(key, &out, true)?);
                cells += 1;
            }
        }
        let weight = upload("conv_weight")?;
        let out = output(count)?;
        let run_conv = |input: &crate::backend::gpu::CudaSlice<bf16>| -> Result<()> {
            unsafe {
                stream
                    .launch_builder(&conv)
                    .arg(input)
                    .arg(&weight)
                    .arg(&out)
                    .arg(&rows)
                    .arg(&1024u32)
                    .arg(&5u32)
                    .launch(flat(count))?;
            }
            Ok(())
        };
        run_conv(&x)?;
        defects += usize::from(!compare("conv", &out, false)?);
        cells += 1;
        if rows > 1 {
            let mut wrong = reference_bf16(&fixture, "x")?.to_vec();
            wrong.rotate_right(1024);
            run_conv(&stream.clone_htod(&wrong)?)?;
            defects += usize::from(!compare("conv", &out, true)?);
            cells += 1;
        }
        let b = upload("branch")?;
        let w = upload("weight")?;
        for mutant in [false, true] {
            let mut host = reference_bf16(&fixture, "x")?
                .iter()
                .map(|v| {
                    if mutant {
                        bf16::from_f32(v.to_f32() * 0.5)
                    } else {
                        *v
                    }
                })
                .collect::<Vec<_>>();
            host.extend([bf16::from_bits(0x7fc1); 32]);
            let out = stream.clone_htod(&host)?;
            unsafe {
                stream
                    .launch_builder(&branch)
                    .arg(&out)
                    .arg(&b)
                    .arg(&w)
                    .arg(&1024u32)
                    .arg(&1e-6f32)
                    .arg(&1e10f32)
                    .arg(&0.5f32)
                    .launch(super::ops::grid(rows as usize, 256))?;
            }
            defects += usize::from(!compare("branch_out", &out, mutant)?);
            cells += 1;
        }
        let q = upload("q")?;
        let k = upload("k")?;
        let v = upload("v")?;
        let relative = upload("relative")?;
        let scale = upload("scale")?;
        let softplus = stream.alloc_zeros::<bf16>(128)?;
        unsafe {
            stream
                .launch_builder(&activation)
                .arg(&scale)
                .arg(&softplus)
                .arg(&1u32)
                .arg(&128u32)
                .arg(&3i32)
                .launch(flat(128))?;
        }
        let out = output(count)?;
        let q_scale = (128f64.powf(-0.5) / 2f64.ln()) as f32;
        let k_scale = ((1.0 + std::f64::consts::E).ln() / 2f64.ln()) as f32;
        for left in [13u32, 14] {
            if left == 14 && rows < 13 {
                continue;
            }
            unsafe {
                stream
                    .launch_builder(&attention)
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&relative)
                    .arg(&softplus)
                    .arg(&mask)
                    .arg(&out)
                    .arg(&rows)
                    .arg(&8u32)
                    .arg(&12u32)
                    .arg(&left)
                    .arg(&q_scale)
                    .arg(&k_scale)
                    .arg(&50.0f32)
                    .arg(&-1e9f32)
                    .launch(LaunchConfig {
                        grid_dim: (rows, 8, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            defects += usize::from(!compare("attention", &out, left == 14)?);
            cells += 1;
        }
    }
    println!(
        "AUDIO_OPS {} cells={cells} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "audio operator defects={defects}");
    Ok(())
}

fn load_reference(path: &std::path::Path) -> Result<ReferenceFixture> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    tensors
        .iter()
        .map(|(name, view)| {
            let values = match view.dtype() {
                safetensors::Dtype::BF16 => ReferenceValues::Bf16(
                    view.data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| bf16::from_le_bytes(*b))
                        .collect(),
                ),
                safetensors::Dtype::F32 => ReferenceValues::F32(
                    view.data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                ),
                dtype => anyhow::bail!("reference tensor {name} has unsupported dtype {dtype:?}"),
            };
            Ok((name.to_owned(), (view.shape().to_vec(), values)))
        })
        .collect()
}

fn reference_fixture(file: &str) -> Result<ReferenceFixture> {
    let dir = std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?;
    load_reference(&std::path::Path::new(&dir).join(file))
}

fn reference<'a>(fixture: &'a ReferenceFixture, name: &str) -> Result<&'a ReferenceValues> {
    Ok(&fixture
        .get(name)
        .with_context(|| format!("reference tensor {name} missing"))?
        .1)
}

fn reference_bf16<'a>(fixture: &'a ReferenceFixture, name: &str) -> Result<&'a [bf16]> {
    reference(fixture, name)?
        .as_bf16()
        .with_context(|| format!("reference tensor {name} is not BF16"))
}

struct VisionFixture<'a> {
    device: &'a Device,
    values: ReferenceFixture,
    height: u32,
    width: u32,
    defects: usize,
}

impl VisionFixture<'_> {
    fn upload(&self, name: &str) -> Result<crate::backend::gpu::CudaSlice<bf16>> {
        Ok(self
            .device
            .stream
            .clone_htod(reference_bf16(&self.values, name)?)?)
    }

    fn compare(
        &mut self,
        name: &str,
        output: &crate::backend::gpu::CudaSlice<bf16>,
        control: bool,
    ) -> Result<()> {
        let actual = self.device.stream.clone_dtoh(output)?;
        let expected = reference_bf16(&self.values, name)?;
        ensure!(
            actual.len() == expected.len() && actual.iter().all(|v| v.is_finite()),
            "RUNNER_FAIL vision output shape or finiteness: {name}"
        );
        let differences = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let pass = if control {
            differences > 0
        } else {
            differences == 0
        };
        self.defects += usize::from(!pass);
        println!(
            "{} vision_op {name} grid={}x{} control={control} bit_mismatches={differences}",
            if pass { "PASS" } else { "DEFECT" },
            self.height,
            self.width
        );
        Ok(())
    }

    fn pointwise(&mut self) -> Result<()> {
        let s = &self.device.stream;
        let module = cubin::module(&self.device.ctx, "vision_ops")?;
        let cast = module.load_function("vision_affine_cast_bf16")?;
        let clamp = module.load_function("vision_clamp_bf16")?;
        let position = module.load_function("vision_axis_position_add_bf16")?;
        let pixels = s.clone_htod(&reference(&self.values, "pixels")?.as_f32())?;
        let scaled = self.upload("scaled")?;
        let table = self.upload("table")?;
        let mut wrong_table = reference_bf16(&self.values, "table")?.to_vec();
        let half = wrong_table.len() / 2;
        wrong_table.rotate_left(half);
        let wrong_table = s.clone_htod(&wrong_table)?;
        let hidden = u32::try_from(self.values["scaled"].0[1])?;
        let positions = u32::try_from(self.values["table"].0[1])?;
        let count = u64::from(self.height) * u64::from(self.width) * u64::from(hidden);
        ensure!(
            pixels.len() == count as usize
                && scaled.len() == pixels.len()
                && table.len() == 2 * positions as usize * hidden as usize,
            "RUNNER_FAIL pointwise fixture extents"
        );
        let mut output = s.alloc_zeros::<bf16>(count as usize)?;
        for control in [false, true] {
            let (shift, scale) = if control {
                (0.0f32, 1.0f32)
            } else {
                (-0.5, 2.0)
            };
            unsafe {
                s.launch_builder(&cast)
                    .arg(&pixels)
                    .arg(&mut output)
                    .arg(&count)
                    .arg(&shift)
                    .arg(&scale)
                    .launch(flat(count as usize))?;
            }
            self.compare("scaled", &output, control)?;
            let lower = if control {
                s.clone_htod(&[bf16::from_f32(-2.0)])?
            } else {
                self.upload("lower")?
            };
            let upper = if control {
                s.clone_htod(&[bf16::from_f32(2.0)])?
            } else {
                self.upload("upper")?
            };
            unsafe {
                s.launch_builder(&clamp)
                    .arg(&scaled)
                    .arg(&mut output)
                    .arg(&count)
                    .arg(&lower)
                    .arg(&upper)
                    .launch(flat(count as usize))?;
            }
            self.compare("clamped", &output, control)?;
            unsafe {
                s.launch_builder(&position)
                    .arg(&scaled)
                    .arg(if control { &wrong_table } else { &table })
                    .arg(&mut output)
                    .arg(&self.height)
                    .arg(&self.width)
                    .arg(&hidden)
                    .arg(&positions)
                    .launch(flat(count as usize))?;
            }
            self.compare("positioned", &output, control)?;
        }
        Ok(())
    }

    fn pool(&mut self) -> Result<()> {
        let s = &self.device.stream;
        let module = cubin::module(&self.device.ctx, "vision_ops")?;
        let pool = module.load_function("vision_spatial_mean_scale_bf16")?;
        let input = self.upload("pool_input")?;
        let bias = self.upload("bias")?;
        let multiplier = self.upload("multiplier")?;
        let hidden = u32::try_from(self.values["pool_input"].0[1])?;
        let count = u64::from(self.height) * u64::from(self.width) * u64::from(hidden) / 9;
        let mut output = s.alloc_zeros::<bf16>(count as usize)?;
        ensure!(
            input.len() == 9 * count as usize
                && bias.len() == hidden as usize
                && multiplier.len() == bias.len(),
            "RUNNER_FAIL pooling fixture extents"
        );
        let scale = f64::from(hidden).sqrt() as f32;
        let original = reference_bf16(&self.values, "pool_input")?;
        let mut transposed = vec![bf16::ZERO; original.len()];
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let source = (y * self.width as usize + x) * hidden as usize;
                let target = (x * self.height as usize + y) * hidden as usize;
                transposed[target..target + hidden as usize]
                    .copy_from_slice(&original[source..source + hidden as usize]);
            }
        }
        let transposed = s.clone_htod(&transposed)?;
        for control in [false, true] {
            let (h, w) = if control {
                (self.width, self.height)
            } else {
                (self.height, self.width)
            };
            let input = if control { &transposed } else { &input };
            unsafe {
                s.launch_builder(&pool)
                    .arg(input)
                    .arg(&mut output)
                    .arg(&0u64)
                    .arg(&0u64)
                    .arg(&h)
                    .arg(&w)
                    .arg(&hidden)
                    .arg(&3u32)
                    .arg(&scale)
                    .launch(flat(count as usize))?;
            }
            self.compare("pooled", &output, control)?;
            unsafe {
                s.launch_builder(&pool)
                    .arg(input)
                    .arg(&mut output)
                    .arg(&bias)
                    .arg(&multiplier)
                    .arg(&h)
                    .arg(&w)
                    .arg(&hidden)
                    .arg(&3u32)
                    .arg(&scale)
                    .launch(flat(count as usize))?;
            }
            self.compare("standardized", &output, control)?;
        }
        Ok(())
    }

    fn rope(&mut self) -> Result<()> {
        let s = &self.device.stream;
        let module = cubin::module(&self.device.ctx, "vision_rope")?;
        let rope = module.load_function("vision_axial_rope_prepare_bf16")?;
        let q = self.upload("q")?;
        let k = self.upload("k")?;
        let v = self.upload("v")?;
        let rows = self.height * self.width;
        let heads = u32::try_from(self.values["q"].0[1])?;
        let dim = u32::try_from(self.values["q"].0[2])?;
        let mut q_out = s.alloc_zeros::<bf16>(q.len())?;
        let mut k_out = s.alloc_zeros::<bf16>(k.len())?;
        let mut v_out = s.alloc_zeros::<bf16>(v.len())?;
        ensure!(
            q.len() == rows as usize * heads as usize * dim as usize
                && q.len() == k.len()
                && q.len() == v.len(),
            "RUNNER_FAIL rotary fixture extents"
        );
        for control in [false, true] {
            let mut table = reference_bf16(&self.values, "angles")?.to_vec();
            ensure!(
                table.len() == rows as usize * dim as usize,
                "RUNNER_FAIL rotary table extent"
            );
            if control {
                for row in table.chunks_exact_mut(dim as usize) {
                    row.rotate_left(dim as usize / 2);
                }
            }
            let table = s.clone_htod(&table)?;
            unsafe {
                s.launch_builder(&rope)
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&table)
                    .arg(&mut q_out)
                    .arg(&mut k_out)
                    .arg(&mut v_out)
                    .arg(&rows)
                    .arg(&heads)
                    .arg(&dim)
                    .launch(flat(q.len()))?;
            }
            self.compare("rotated_q", &q_out, control)?;
            self.compare("rotated_k", &k_out, control)?;
            self.compare("transposed_v", &v_out, false)?;
        }
        Ok(())
    }
}

#[test]
#[ignore = "requires a GPU and frozen FF_GEMMA4_FIXTURES vision operator references"]
fn axial_vision_pointwise_pool_rope_gpu() -> Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("VISION_FROZEN.json"))?)?;
    ensure!(
        frozen["status"] == "FROZEN"
            && frozen["max_bit_mismatches"] == 0
            && frozen["reference_devices"] == serde_json::json!(["cpu", "sm89", "sm86"]),
        "RUNNER_FAIL vision operator reference freeze missing"
    );
    let device = Device::new(0)?;
    let mut defects = 0;
    for (height, width) in [(9, 15), (48, 48), (42, 60)] {
        let name = format!("vision-ops-{height}x{width}");
        ensure!(
            frozen["cells"][&name]["cross_device_mismatches"] == 0,
            "RUNNER_FAIL vision reference cross-device noise differs"
        );
        let mut cell = VisionFixture {
            device: &device,
            values: load_reference(&directory.join(format!("{name}.safetensors")))?,
            height,
            width,
            defects: 0,
        };
        cell.pointwise()?;
        cell.pool()?;
        cell.rope()?;
        defects += cell.defects;
    }
    println!(
        "{} axial_vision_pointwise_pool_rope_gpu cells=3 defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "DEFECT axial vision operations");
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a GPU and frozen FF_GEMMA4_FIXTURES attention references"]
fn vision_bidirectional_attention_gpu() -> Result<()> {
    use super::flash::{Flash, Query, Score, Shape};
    let directory = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?,
    );
    let limits: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("D64_FROZEN.json"))?)?;
    ensure!(
        limits["status"] == "FROZEN"
            && limits["reference_devices"] == serde_json::json!(["sm89", "sm86"]),
        "RUNNER_FAIL attention reference freeze missing"
    );
    let device = Device::new(0)?;
    // SAFETY: all allocations, copies and launches use this test's single stream.
    unsafe {
        device.ctx.disable_event_tracking();
    }
    let stream = &device.stream;
    let mut defects = 0;
    for rows in [1usize, 63, 64, 65, 135, 2304, 2520] {
        let name = format!("vision-attention-{rows}");
        let fixture = load_reference(&directory.join(format!("{name}.safetensors")))?;
        let bound = &limits["cells"][&name];
        let max_limit = bound["output_max"].as_f64().context("max limit")?;
        let rmse_limit = bound["output_rmse"].as_f64().context("RMSE limit")?;
        ensure!(
            max_limit.is_finite() && max_limit > 0.0 && rmse_limit.is_finite() && rmse_limit > 0.0,
            "RUNNER_FAIL attention limits"
        );
        let q = stream.clone_htod(reference_bf16(&fixture, "q")?)?;
        let k = stream.clone_htod(reference_bf16(&fixture, "k")?)?;
        let v = stream.clone_htod(reference_bf16(&fixture, "v")?)?;
        let heads = fixture["k"].0[0];
        let capacity = fixture["k"].0[1];
        let pitch = fixture["q"].0[1];
        let keys = usize::try_from(bound["keys"].as_u64().context("key extent")?)?;
        let expected = reference(&fixture, "oracle")?.as_f32();
        let count = rows * heads * 64;
        ensure!(
            fixture["q"].0 == [rows, pitch]
                && fixture["k"].0 == [heads, capacity, 64]
                && fixture["v"].0 == fixture["k"].0
                && fixture["oracle"].0 == [rows, heads, 64]
                && expected.len() == count
                && expected.iter().all(|v| v.is_finite()),
            "RUNNER_FAIL attention fixture extents/finiteness"
        );
        let mut out = stream.clone_htod(&vec![bf16::from_bits(0x7fc1); count + 32])?;
        let run = |op: &mut Flash, out: &mut crate::backend::gpu::CudaSlice<bf16>| {
            op.run_masked(Query {
                q: &q,
                q_row_stride: pitch,
                k: &k,
                v: &v,
                out,
                shape: Shape {
                    rows,
                    start: 0,
                    capacity,
                    q_heads: heads,
                    kv_heads: heads,
                    head_dim: 64,
                    causal: false,
                    window_left: None,
                },
                keys,
                prefix: None,
            })
        };
        for control in [false, true] {
            let mut op = Flash::new_with_score(
                stream.clone(),
                rows,
                heads,
                heads,
                64,
                Score {
                    scale: if control { 0.125 } else { 1.0 },
                    window_left: None,
                },
                capacity,
            )?;
            run(&mut op, &mut out)?;
            let actual = stream.clone_dtoh(&out)?;
            ensure!(
                actual[..count].iter().all(|v| v.is_finite()),
                "RUNNER_FAIL attention nonfinite"
            );
            let mut maximum = 0.0f64;
            let mut squared = 0.0;
            for (actual, expected) in actual[..count].iter().zip(&expected) {
                let error = actual.to_f64() - f64::from(*expected);
                maximum = maximum.max(error.abs());
                squared += error * error;
            }
            let rmse = (squared / count as f64).sqrt();
            let within = maximum <= max_limit && rmse <= rmse_limit;
            let guard = actual[count..].iter().all(|v| v.to_bits() == 0x7fc1);
            let pass = within != control && guard;
            defects += usize::from(!pass);
            println!(
                "{} vision_attention rows={rows} control={control} max={maximum:e} rmse={rmse:e} limits={max_limit:e}/{rmse_limit:e} guard={guard}",
                if pass { "PASS" } else { "DEFECT" }
            );
            if !control {
                stream.begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )?;
                run(&mut op, &mut out)?;
                let graph = stream
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))?
                    .context("attention graph missing")?;
                graph.launch()?;
                let captured = stream.clone_dtoh(&out)?;
                let bits = captured
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                defects += usize::from(bits != 0);
                println!(
                    "{} vision_attention rows={rows} graph_eager_bits={bits}",
                    if bits == 0 { "PASS" } else { "DEFECT" }
                );
            }
        }
    }
    println!(
        "{} vision_bidirectional_attention_gpu cells=7 defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "DEFECT vision bidirectional attention");
    Ok(())
}

#[test]
#[ignore = "requires a GPU and FF_GEMMA4_FIXTURES"]
fn gelu_tanh_gated_activation_matches_the_definition_gpu() -> Result<()> {
    use super::contracts::{TensorMut, view};
    use super::ops::{CommonOpsRef, GeluMode};
    let device = Device::new(0)?;
    let s = &device.stream;
    let module = cubin::module(&device.ctx, "act")?;
    let fused = module.load_function("silu_mul")?;
    let split = module.load_function("silu_mul_split")?;
    let data = fixture("act.safetensors")?;
    let (mut cases, mut defects, mut undetected) = (0, 0, 0);
    for rows in [1usize, 17, 64] {
        for dim in [256usize, 6144, 10240, 12288] {
            let key = format!("r{rows}.d{dim}");
            let gate = tensor(&data, &format!("{key}.gate"))?;
            let up = tensor(&data, &format!("{key}.up"))?;
            let (gate_host, up_host) = (gate, up);
            let reference = tensor(&data, &format!("{key}.out"))?;
            let packed: Vec<bf16> = (0..rows)
                .flat_map(|r| {
                    gate[r * dim..(r + 1) * dim]
                        .iter()
                        .chain(&up[r * dim..(r + 1) * dim])
                        .copied()
                })
                .collect();
            let packed = s.clone_htod(&packed)?;
            let (gate, up) = (s.clone_htod(gate)?, s.clone_htod(up)?);
            let run = |mode: i32| -> Result<(Vec<bf16>, Vec<bf16>)> {
                let mut y = s.alloc_zeros::<bf16>(rows * dim)?;
                let mut z = s.alloc_zeros::<bf16>(rows * dim)?;
                unsafe {
                    s.launch_builder(&fused)
                        .arg(&packed)
                        .arg(&mut y)
                        .arg(&(rows as i32))
                        .arg(&(dim as i32))
                        .arg(&mode)
                        .launch(flat(rows * dim))?;
                    s.launch_builder(&split)
                        .arg(&gate)
                        .arg(&up)
                        .arg(&mut z)
                        .arg(&((rows * dim) as i32))
                        .arg(&mode)
                        .launch(flat(rows * dim))?;
                }
                Ok((s.clone_dtoh(&y)?, s.clone_dtoh(&z)?))
            };
            let floor: Vec<f32> = gate_host
                .iter()
                .zip(up_host)
                .map(|(g, u)| g.to_f32().abs() * u.to_f32().abs() * 2f32.powi(-23))
                .collect();
            let (fused_out, split_out) = run(2)?;
            for (kernel, out) in [("silu_mul", &fused_out), ("silu_mul_split", &split_out)] {
                let stats = Stats::with_floor(out, reference, Some(&floor))?;
                println!(
                    "GEMMA4_GELU_TANH {}",
                    stats.line(&format!("{kernel}.{key}"))
                );
                if !stats.passes() {
                    println!("GEMMA4_GELU_TANH DEFECT {kernel}.{key}");
                    defects += 1;
                }
            }
            let (silu_fused, silu_split) = run(0)?;
            for (kernel, out) in [("silu_mul", &silu_fused), ("silu_mul_split", &silu_split)] {
                let stats = Stats::with_floor(out, reference, Some(&floor))?;
                println!(
                    "GEMMA4_GELU_TANH_MUTANT silu {}",
                    stats.line(&format!("{kernel}.{key}"))
                );
                if !(stats.is_detected_mutant() && !stats.passes()) {
                    println!("GEMMA4_GELU_TANH_MUTANT NOT_DETECTED {kernel}.{key}");
                    undetected += 1;
                }
            }
            cases += 1;
        }
    }
    println!("GEMMA4_GELU_TANH cases={cases} defects={defects} undetected_mutants={undetected}");
    ensure!(defects == 0 && undetected == 0, "GEMMA4_GELU_TANH FAIL");
    let common = CommonOpsRef::new(&device.ctx)?;
    device.finish_upload()?;
    let (mut product_cases, mut product_defects, mut controls) = (0, 0, 0);
    for rows in [1u32, 17, 64, 1024] {
        for heads in [1u32, 2] {
            let pitch = if heads == 1 { 256 } else { 263 };
            let stride = [u64::from(heads) * pitch, pitch, 1];
            let count = rows as usize * stride[0] as usize + 32;
            let xh: Vec<_> = (0..count)
                .map(|i| bf16::from_f32(((i * 37 % 1009) as f32 - 504.0) / 128.0))
                .collect();
            let uh: Vec<_> = (0..count)
                .map(|i| bf16::from_f32(((i * 41 % 1013) as f32 - 506.0) / 256.0))
                .collect();
            let x = s.clone_htod(&xh)?;
            let up = s.clone_htod(&uh)?;
            let sentinel = vec![bf16::from_f32(-32.0); count];
            let raw = s.clone_htod(&sentinel)?;
            let xv = view(&x, s, 16, [rows, heads, 256], stride)?;
            let uv = view(&up, s, 16, [rows, heads, 256], stride)?;
            common.gate.gelu(
                s,
                xv,
                &mut TensorMut(view(&raw, s, 16, [rows, heads, 256], stride)?),
                GeluMode::Tanh,
            )?;
            let raw = s.clone_dtoh(&raw)?;
            let mut expected = sentinel.clone();
            let mut single_round_mismatches = 0;
            for row in 0..rows as usize {
                for head in 0..heads as usize {
                    for d in 0..256 {
                        let i = 16 + row * stride[0] as usize + head * pitch as usize + d;
                        expected[i] = bf16::from_f32(raw[i].to_f32() * uh[i].to_f32());
                        let v = xh[i].to_f32();
                        let cubic = (v * v) * v;
                        let inner = 0.797_884_6_f32 * (v + 0.044_715_f32 * cubic);
                        let gelu = (0.5 * v) * (1.0 + inner.tanh());
                        let wrong = bf16::from_f32(gelu * uh[i].to_f32());
                        single_round_mismatches += usize::from(wrong != expected[i]);
                    }
                }
            }
            controls += usize::from(single_round_mismatches > 0);
            for graph in [false, true] {
                let out = s.clone_htod(&sentinel)?;
                let ov = view(&out, s, 16, [rows, heads, 256], stride)?;
                if graph {
                    s.synchronize()?;
                    #[cfg(feature = "cuda")]
                    s.begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                    #[cfg(feature = "rocm")]
                    let capture = s.begin_capture()?;
                    let result = common.gate.gelu_tanh_mul(s, xv, uv, &mut TensorMut(ov));
                    #[cfg(feature = "cuda")]
                    let captured = s
                        .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))
                        .map_err(anyhow::Error::from)
                        .and_then(|g| g.context("gated GELU graph is empty"));
                    #[cfg(feature = "rocm")]
                    let captured = capture.finish();
                    result?;
                    let captured = captured?;
                    captured.launch()?;
                    captured.launch()?;
                } else {
                    common.gate.gelu_tanh_mul(s, xv, uv, &mut TensorMut(ov))?;
                }
                let out = s.clone_dtoh(&out)?;
                let mismatches = out
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "GELU_TANH_MUL rows={rows} heads={heads} graph={graph} bit_mismatches={mismatches} single_round_mismatches={single_round_mismatches}"
                );
                product_cases += 1;
                product_defects += usize::from(mismatches != 0);
            }
        }
    }
    println!(
        "GELU_TANH_MUL_RESULT cases={product_cases} defects={product_defects} single_round_controls={controls}"
    );
    ensure!(
        product_cases == 16 && product_defects == 0 && controls == 8,
        "GELU_TANH_MUL FAIL"
    );
    Ok(())
}

fn model_tag(spec: &crate::model_spec::ModelSpec) -> Result<&'static str> {
    match spec.hidden {
        1536 => Ok("e2b"),
        2560 => Ok("e4b"),
        hidden => anyhow::bail!("unexpected Gemma 4 hidden size {hidden}"),
    }
}

fn embedding_table(dir: &std::path::Path) -> Result<Vec<bf16>> {
    let checkpoint = crate::checkpoint::MappedCheckpoint::open(dir)?;
    let name = "model.language_model.embed_tokens.weight";
    let views = checkpoint.views()?;
    let view = views
        .values()
        .find_map(|shard| shard.tensor(name).ok())
        .context("embedding table missing")?;
    Ok(view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect())
}

#[test]
#[ignore = "requires a GPU, FF_GEMMA4_FIXTURES and a Gemma 4 E2B/E4B directory in FF_MODEL"]
fn softcap_tied_head_and_embedding_scale_match_the_definition_gpu() -> Result<()> {
    #[cfg(feature = "cuda")]
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys as bs};
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    let tag = model_tag(&spec)?;
    let (hidden, vocab) = (spec.hidden as usize, spec.vocab as usize);
    let cap = spec
        .logit_softcap
        .context("softcap missing from the spec")?;
    let device = Device::new(0)?;
    let s = &device.stream;
    let data = fixture("head.safetensors")?;
    let table = embedding_table(&dir)?;
    ensure!(table.len() == vocab * hidden, "embedding table extent");
    let table = s.clone_htod(&table)?;
    let (mut defects, mut undetected) = (0, 0);

    let ids: Vec<u32> = [0u32, 1, 2, 262143]
        .into_iter()
        .chain((0..60u32).map(|i| (i * 4099 + 7) % 262144))
        .collect();
    let ids_device = s.clone_htod(&ids)?;
    let embed = cubin::module(&device.ctx, "embed")?.load_function("embed_scale_broadcast")?;
    let reference = tensor(&data, &format!("embed.{tag}.out"))?;
    let initial = s.alloc_zeros::<bf16>(hidden)?;
    let scale_for = |scale: f32| -> Result<Vec<bf16>> {
        let mut embedded = s.alloc_zeros::<bf16>(ids.len() * hidden)?;
        let mut broadcast = s.alloc_zeros::<bf16>(ids.len() * hidden)?;
        unsafe {
            s.launch_builder(&embed)
                .arg(&table)
                .arg(&ids_device)
                .arg(&initial)
                .arg(&mut embedded)
                .arg(&mut broadcast)
                .arg(&(ids.len() as i32))
                .arg(&(hidden as i32))
                .arg(&scale)
                .arg(&0u64)
                .arg(&0i32)
                .arg(&0i32)
                .launch(flat(ids.len() * hidden))?;
        }
        s.clone_dtoh(&embedded)
            .context("Gemma device readback failed")
    };
    let scale = bf16::from_f32((hidden as f32).sqrt()).to_f32();
    let stats = Stats::new(&scale_for(scale)?, reference)?;
    println!("GEMMA4_EMBED_SCALE {}", stats.line(tag));
    defects += usize::from(!stats.passes());
    let stats = Stats::new(&scale_for(1.0)?, reference)?;
    println!("GEMMA4_EMBED_SCALE_MUTANT unscaled {}", stats.line(tag));
    undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));

    let h_host = tensor(&data, &format!("head.{tag}.h"))?;
    let raw = tensor(&data, &format!("head.{tag}.raw"))?;
    let capped = tensor(&data, &format!("head.{tag}.capped"))?;
    let rows = h_host.len() / hidden;
    let h = s.clone_htod(h_host)?;
    #[cfg(feature = "cuda")]
    let blas = CudaBlas::new(s.clone())?;
    #[cfg(feature = "cuda")]
    let product = |alpha: bf16| -> Result<Vec<bf16>> {
        let mut y = s.alloc_zeros::<bf16>(rows * vocab)?;
        let config = GemmConfig {
            transa: bs::cublasOperation_t::CUBLAS_OP_T,
            transb: bs::cublasOperation_t::CUBLAS_OP_N,
            m: vocab as i32,
            n: rows as i32,
            k: hidden as i32,
            alpha,
            lda: hidden as i32,
            ldb: hidden as i32,
            beta: bf16::ZERO,
            ldc: vocab as i32,
        };
        unsafe {
            blas.gemm(config, &table, &h, &mut y)?;
        }
        s.clone_dtoh(&y).context("Gemma device readback failed")
    };
    #[cfg(feature = "rocm")]
    let blas = crate::backend::gpu::blas::Blas::new(s.clone())?;
    #[cfg(feature = "rocm")]
    blas.prepare_choice(
        crate::backend::setup::LinearShape {
            rows,
            output: vocab,
            input: hidden,
        },
        None,
        None,
    )?;
    #[cfg(feature = "rocm")]
    let product = |alpha: bf16| -> Result<Vec<bf16>> {
        let mut y = s.alloc_zeros::<bf16>(rows * vocab)?;
        blas.linear(&table, &h, &mut y, rows, vocab, hidden)?;
        let mut values = s.clone_dtoh(&y)?;
        if alpha != bf16::ONE {
            for value in &mut values {
                *value = bf16::from_f32(value.to_f32() * alpha.to_f32());
            }
        }
        Ok(values)
    };
    let reduction = |out: &[bf16], reference: &[bf16]| -> (f64, f64) {
        let (mut dot, mut a, mut b, mut worst, mut peak) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (x, y) in out.iter().zip(reference) {
            let (x, y) = (f64::from(x.to_f32()), f64::from(y.to_f32()));
            dot += x * y;
            a += x * x;
            b += y * y;
            worst = worst.max((x - y).abs());
            peak = peak.max(y.abs());
        }
        (dot / (a.sqrt() * b.sqrt()), worst / peak)
    };
    let tied = product(bf16::ONE)?;
    let (cosine, relative) = reduction(&tied, raw);
    println!(
        "GEMMA4_TIED_HEAD {tag} rows={rows} cosine={cosine:.7} max_abs_over_peak={relative:.6}"
    );
    defects += usize::from(!(cosine >= 0.9999 && relative <= 1.0 / 64.0));
    let scaled = product(bf16::from_f32(scale))?;
    let (cosine, relative) = reduction(&scaled, raw);
    println!(
        "GEMMA4_TIED_HEAD_MUTANT embed_scaled {tag} cosine={cosine:.7} max_abs_over_peak={relative:.6}"
    );
    undetected += usize::from(cosine >= 0.9999 && relative <= 1.0 / 64.0);

    let softcap = cubin::module(&device.ctx, "act")?.load_function("softcap")?;
    let run_cap = |input: &[bf16], cap: f32| -> Result<Vec<bf16>> {
        let mut x = s.clone_htod(input)?;
        unsafe {
            s.launch_builder(&softcap)
                .arg(&mut x)
                .arg(&(input.len() as i64))
                .arg(&cap)
                .launch(flat(input.len()))?;
        }
        s.clone_dtoh(&x).context("Gemma device readback failed")
    };
    let stats = Stats::new(&run_cap(raw, cap)?, capped)?;
    println!("GEMMA4_SOFTCAP {}", stats.line(tag));
    defects += usize::from(!stats.passes());
    let stats = Stats::new(raw, capped)?;
    println!("GEMMA4_SOFTCAP_MUTANT tanh_absent {}", stats.line(tag));
    undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
    println!("GEMMA4_HEAD {tag} defects={defects} undetected_mutants={undetected}");
    ensure!(defects == 0 && undetected == 0, "GEMMA4_HEAD FAIL");
    Ok(())
}

fn rms_f32(x: &[f32], weight: &[bf16]) -> Vec<bf16> {
    let mean = x.iter().map(|v| f64::from(v * v)).sum::<f64>() as f32 / x.len() as f32;
    let inv = 1.0 / (mean + 1e-6).sqrt();
    x.iter()
        .zip(weight)
        .map(|(v, w)| bf16::from_f32(v * inv * w.to_f32()))
        .collect()
}

#[test]
#[ignore = "requires a GPU and FF_GEMMA4_FIXTURES"]
fn unrounded_norm_and_branch_boundary_match_the_definition_gpu() -> Result<()> {
    use super::contracts::{RowDomain, SeqRef, Tensor, TensorMut};
    use super::ops::{CommonOpsRef, RmsMode, RmsOptions};
    use crate::backend::gpu::DevicePtr;
    use crate::backend::gpu::{CudaSlice, CudaStream};
    use std::sync::Arc;
    fn heads(
        b: &CudaSlice<bf16>,
        s: &Arc<CudaStream>,
        rows: usize,
        h: u32,
        d: u32,
    ) -> Result<Tensor<bf16, 3>> {
        let bytes = (b.len() * 2) as u64;
        // SAFETY: the allocation outlives every launch in this test and the extent is the whole buffer.
        unsafe {
            Tensor::new(
                b.device_ptr(s).0,
                [u32::try_from(rows)?, h, d],
                [u64::from(h) * u64::from(d), u64::from(d), 1],
                bytes,
            )
        }
    }
    fn vector(b: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> Result<Tensor<bf16, 1>> {
        // SAFETY: as above.
        unsafe {
            Tensor::new(
                b.device_ptr(s).0,
                [u32::try_from(b.len())?],
                [1],
                (b.len() * 2) as u64,
            )
        }
    }
    let device = Device::new(0)?;
    let s = &device.stream;
    let common = CommonOpsRef::new(&device.ctx)?;
    let data = fixture("norm.safetensors")?;
    let (mut cases, mut defects, mut undetected) = (0, 0, 0);
    let options = RmsOptions {
        mode: RmsMode::Unrounded,
        eps: 1e-6,
    };

    for rows in [1usize, 17, 64] {
        let meta = s.clone_htod(&[0i32, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: meta.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: rows as u32,
        };
        for (h, d) in [(1usize, 256usize), (1, 512), (2, 256), (2, 512)] {
            let key = format!("ns.r{rows}.h{h}.d{d}");
            let x = s.clone_htod(tensor(&data, &format!("{key}.x"))?)?;
            let reference = tensor(&data, &format!("{key}.out"))?;
            let run = |weight: &[bf16]| -> Result<Vec<bf16>> {
                let weight = s.clone_htod(weight)?;
                let y = s.alloc_zeros::<bf16>(rows * h * d)?;
                common.norm.rms_with(
                    s,
                    domain,
                    heads(&x, s, rows, h as u32, d as u32)?,
                    vector(&weight, s)?,
                    &mut TensorMut(heads(&y, s, rows, h as u32, d as u32)?),
                    options,
                )?;
                s.clone_dtoh(&y).context("Gemma device readback failed")
            };
            let stats = Stats::new(&run(&vec![bf16::ONE; d])?, reference)?;
            println!("GEMMA4_V_NORM {}", stats.line(&key));
            defects += usize::from(!stats.passes());
            let scaled = tensor(&data, &format!("kw{d}"))?;
            let stats = Stats::new(&run(scaled)?, reference)?;
            println!("GEMMA4_V_NORM_MUTANT scaled {}", stats.line(&key));
            undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
            cases += 1;
        }
    }

    let module = cubin::module(&device.ctx, "q38_common")?;
    let branch = module.load_function("rms_branch_add")?;
    for tag in ["e2b", "e4b"] {
        let weight = |name: &str| tensor(&data, &format!("br.{tag}.{name}"));
        let hidden = weight("w_post_attn")?.len();
        let scalar = s.clone_htod(weight("scalar")?)?;
        for rows in [1usize, 17, 64] {
            let key = format!("br.{tag}.r{rows}");
            let (xh, bh) = (
                tensor(&data, &format!("{key}.x"))?,
                tensor(&data, &format!("{key}.b"))?,
            );
            let b = s.clone_htod(bh)?;
            for (label, w_branch, w_norm, with_scalar, out_key, n_key) in [
                (
                    "post_attention",
                    "w_post_attn",
                    "w_pre_ff",
                    false,
                    "out_a",
                    "n_a",
                ),
                ("layer_end", "w_post_ff", "w_next", true, "out_b", "n_b"),
            ] {
                let w_norm_host = weight(w_norm)?;
                let (w_branch, w_norm) = (
                    s.clone_htod(weight(w_branch)?)?,
                    s.clone_htod(weight(w_norm)?)?,
                );
                let expected_x = tensor(&data, &format!("{key}.{out_key}"))?;
                let expected_n = tensor(&data, &format!("{key}.{n_key}"))?;
                let run = |use_scalar: bool| -> Result<(Vec<bf16>, Vec<bf16>)> {
                    let mut x = s.clone_htod(xh)?;
                    let mut n = s.alloc_zeros::<bf16>(rows * hidden)?;
                    let null = 0u64;
                    unsafe {
                        let mut launch = s.launch_builder(&branch);
                        launch.arg(&mut x).arg(&b).arg(&w_branch);
                        if use_scalar {
                            launch.arg(&scalar);
                        } else {
                            launch.arg(&null);
                        }
                        launch
                            .arg(&w_norm)
                            .arg(&mut n)
                            .arg(&(hidden as u32))
                            .arg(&1e-6f32)
                            .launch(super::ops::grid(rows, 256))?;
                    }
                    Ok((s.clone_dtoh(&x)?, s.clone_dtoh(&n)?))
                };
                let (got_x, got_n) = run(with_scalar)?;
                let scale = if with_scalar {
                    weight("scalar")?[0].to_f32().abs()
                } else {
                    1.0
                };
                let floor_x: Vec<f32> = xh
                    .iter()
                    .map(|v| 2.0 * scale * bf16_ulp(v.to_f32()))
                    .collect();
                let floor_n: Vec<f32> = expected_x
                    .chunks(hidden)
                    .zip(floor_x.chunks(hidden))
                    .zip(expected_n.chunks(hidden))
                    .flat_map(|((row, fx), n_ref)| {
                        let mean = row.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>()
                            / hidden as f32;
                        let inv = 1.0 / (mean + 1e-6).sqrt();
                        row.iter()
                            .enumerate()
                            .map(|(i, _)| {
                                inv * w_norm_host[i].to_f32().abs() * fx[i]
                                    + bf16_ulp(n_ref[i].to_f32())
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                for (name, got, reference, floor) in [
                    ("x", &got_x, expected_x, &floor_x),
                    ("n", &got_n, expected_n, &floor_n),
                ] {
                    let stats = Stats::with_floor(got, reference, Some(floor))?;
                    println!(
                        "GEMMA4_BRANCH_NORM {}",
                        stats.line(&format!("{key}.{label}.{name}"))
                    );
                    defects += usize::from(!stats.passes());
                }
                if with_scalar {
                    let (got_x, _) = run(false)?;
                    let stats = Stats::new(&got_x, expected_x)?;
                    println!(
                        "GEMMA4_BRANCH_NORM_MUTANT scalar_omitted {}",
                        stats.line(&format!("{key}.{label}.x"))
                    );
                    undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
                }
                let summed: Vec<bf16> = xh
                    .iter()
                    .zip(bh)
                    .map(|(a, b)| bf16::from_f32(a.to_f32() + b.to_f32()))
                    .collect();
                let w = weight(if with_scalar {
                    "w_post_ff"
                } else {
                    "w_post_attn"
                })?;
                let llama: Vec<bf16> = summed
                    .chunks(hidden)
                    .flat_map(|row| rms_f32(&row.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), w))
                    .collect();
                let stats = Stats::new(&llama, expected_x)?;
                println!(
                    "GEMMA4_BRANCH_NORM_MUTANT norm_after_add {}",
                    stats.line(&format!("{key}.{label}.x"))
                );
                undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
            }
            cases += 1;
        }
    }
    for hidden in [1536usize, 2560] {
        for rows in [1usize, 17, 64, 1024] {
            let values = |period: usize, count: usize| {
                (0..count)
                    .map(|i| bf16::from_f32(((i * 37 + 11) % period) as f32 / 256.0 - 1.0))
                    .collect::<Vec<_>>()
            };
            let mut x = s.clone_htod(&values(1009, rows * hidden))?;
            let b = s.clone_htod(&values(1013, rows * hidden))?;
            let wb = s.clone_htod(&values(509, hidden))?;
            let wn = s.clone_htod(&values(503, hidden))?;
            let scalar = s.clone_htod(&[bf16::from_f32(0.875)])?;
            let mut reused = s.alloc_zeros::<bf16>(rows * hidden)?;
            unsafe {
                s.launch_builder(&branch)
                    .arg(&mut x)
                    .arg(&b)
                    .arg(&wb)
                    .arg(&scalar)
                    .arg(&wn)
                    .arg(&mut reused)
                    .arg(&(hidden as u32))
                    .arg(&1e-6f32)
                    .launch(super::ops::grid(rows, 256))?;
            }
            let meta = s.clone_htod(&[0i32, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
            let domain = RowDomain {
                meta: SeqRef {
                    ptr: meta.device_ptr(s).0,
                },
                row_base: 0,
                row_capacity: rows as u32,
            };
            let normalize = |weight: &CudaSlice<bf16>| -> Result<Vec<bf16>> {
                let output = s.alloc_zeros::<bf16>(rows * hidden)?;
                common.norm.rms_with(
                    s,
                    domain,
                    heads(&x, s, rows, 1, hidden as u32)?,
                    vector(weight, s)?,
                    &mut TensorMut(heads(&output, s, rows, 1, hidden as u32)?),
                    options,
                )?;
                s.clone_dtoh(&output).context("norm reuse readback failed")
            };
            let actual = s.clone_dtoh(&reused)?;
            let expected = normalize(&wn)?;
            let wrong = normalize(&wb)?;
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let control = actual
                .iter()
                .zip(&wrong)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(mismatches == 0 && control > 0, "norm reuse/control differs");
            println!(
                "NORM_REUSE PASS rows={rows} hidden={hidden} bit_mismatches={mismatches} wrong_weight_mismatches={control}"
            );
        }
    }
    println!("GEMMA4_NORM cases={cases} defects={defects} undetected_mutants={undetected}");
    ensure!(defects == 0 && undetected == 0, "GEMMA4_NORM FAIL");
    Ok(())
}

fn rope_positions() -> Vec<u32> {
    [0u32, 1, 511, 512, 513, 1023, 1024, 131071]
        .into_iter()
        .chain((0..56u32).map(|i| (i * 997 + 13) % 131072))
        .collect()
}

#[test]
#[ignore = "requires a GPU and FF_GEMMA4_FIXTURES"]
fn proportional_and_default_rope_tables_match_the_definition_gpu() -> Result<()> {
    use crate::model_spec::RopeSpec;
    const CAPACITY: usize = 131072;
    let device = Device::new(0)?;
    let s = &device.stream;
    let rope = cubin::module(&device.ctx, "rope")?.load_function("rope_table_kv")?;
    let data = fixture("rope.safetensors")?;
    let positions = rope_positions();
    let positions_device =
        s.clone_htod(&positions.iter().map(|&p| p as i32).collect::<Vec<_>>())?;
    let dummy = s.alloc_zeros::<bf16>(16)?;
    let sliding = RopeSpec {
        theta: 10000.0,
        pairs: 128,
        pair_stride: 128,
        denominator: 256,
    };
    let full = RopeSpec {
        theta: 1000000.0,
        pairs: 64,
        pair_stride: 256,
        denominator: 512,
    };
    let (mut cases, mut defects, mut undetected) = (0, 0, 0);
    for (tag, d, spec) in [("sliding", 256usize, sliding), ("full", 512usize, full)] {
        let half = d / 2;
        let table = spec.table(d as u32, CAPACITY)?;
        // |angle error| <= p * inv_i * 2^-24 (inverse frequency) + p * inv_i * 2^-24 (product), both F32 roundings.
        let cs_floor: Vec<f32> = positions
            .iter()
            .flat_map(|&p| {
                (0..half).map(move |i| {
                    if (i as u32) < spec.pairs {
                        let inv = 1.0 / spec.theta.powf((2 * i) as f32 / spec.denominator as f32);
                        p as f32 * inv * 2f32.powi(-23)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        for (name, reference) in [("cos", 0usize), ("sin", half)] {
            let want = tensor(&data, &format!("rope.{tag}.{name}"))?;
            let got: Vec<bf16> = positions
                .iter()
                .flat_map(|&p| {
                    table[p as usize * d + reference..p as usize * d + reference + half]
                        .iter()
                        .copied()
                })
                .collect();
            let want_half: Vec<bf16> = want
                .chunks(d)
                .flat_map(|row| row[..half].iter().copied())
                .collect();
            let stats = Stats::with_floor(&got, &want_half, Some(&cs_floor))?;
            println!("GEMMA4_ROPE_TABLE {}", stats.line(&format!("{tag}.{name}")));
            defects += usize::from(!stats.passes());
        }
        let upload = |table: &[bf16]| {
            let paired: Vec<_> = table
                .chunks_exact(d)
                .flat_map(|row| row.iter().chain(row).copied())
                .collect();
            s.clone_htod(&paired)
        };
        let table_device = upload(&table)?;
        let run = |table: &crate::backend::gpu::CudaSlice<bf16>,
                   x: &[bf16],
                   heads: usize|
         -> Result<Vec<bf16>> {
            let mut x = s.clone_htod(x)?;
            for row in 0..positions.len() {
                let mut view = x.slice_mut(row * heads * d..(row + 1) * heads * d);
                let start = positions_device.slice(row..row + 1);
                unsafe {
                    s.launch_builder(&rope)
                        .arg(&mut view)
                        .arg(&dummy)
                        .arg(&dummy)
                        .arg(table)
                        .arg(&start)
                        .arg(&1i32)
                        .arg(&(heads as i32))
                        .arg(&0i32)
                        .arg(&(d as i32))
                        .arg(&(CAPACITY as i32))
                        .arg(&((heads * d) as i32))
                        .launch(flat(heads * half))?;
                }
            }
            s.clone_dtoh(&x).context("Gemma device readback failed")
        };
        let mutants: Vec<(&str, crate::backend::gpu::CudaSlice<bf16>)> = if tag == "full" {
            [
                ("rotate_all_256", RopeSpec { pairs: 256, ..spec }),
                (
                    "denominator_256",
                    RopeSpec {
                        denominator: 256,
                        ..spec
                    },
                ),
                (
                    "denominator_128",
                    RopeSpec {
                        denominator: 128,
                        ..spec
                    },
                ),
            ]
            .into_iter()
            .map(|(name, mutant)| Ok((name, upload(&mutant.table(d as u32, CAPACITY)?)?)))
            .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        for heads in [1usize, 2, 8] {
            let key = format!("rope.{tag}.h{heads}");
            let x = tensor(&data, &format!("{key}.x"))?;
            let reference = tensor(&data, &format!("{key}.out"))?;
            let cos_ref = tensor(&data, &format!("rope.{tag}.cos"))?;
            let sin_ref = tensor(&data, &format!("rope.{tag}.sin"))?;
            // F_out = a*(F_cs+U(c)) + b*(F_cs+U(s)) + U(a|c|) + U(b|s|) + U(|out_ref|).
            let out_floor: Vec<f32> = (0..positions.len() * heads * d)
                .map(|e| {
                    let (row, lane) = (e / (heads * d), e % d);
                    let i = lane % half;
                    let base = e - lane;
                    let (a, b) = (
                        x[base + i].to_f32().abs(),
                        x[base + half + i].to_f32().abs(),
                    );
                    let (c, sn) = (cos_ref[row * d + i].to_f32(), sin_ref[row * d + i].to_f32());
                    let (fc, fs) = (
                        cs_floor[row * half + i] + bf16_ulp(c),
                        cs_floor[row * half + i] + bf16_ulp(sn),
                    );
                    let (cos_operand, sin_operand) = if lane < half { (a, b) } else { (b, a) };
                    cos_operand * fc
                        + sin_operand * fs
                        + bf16_ulp(cos_operand * c.abs())
                        + bf16_ulp(sin_operand * sn.abs())
                        + bf16_ulp(reference[e].to_f32())
                })
                .collect();
            let stats =
                Stats::with_floor(&run(&table_device, x, heads)?, reference, Some(&out_floor))?;
            println!("GEMMA4_ROPE {}", stats.line(&key));
            defects += usize::from(!stats.passes());
            for (name, mutant) in &mutants {
                let stats = Stats::new(&run(mutant, x, heads)?, reference)?;
                println!("GEMMA4_ROPE_MUTANT {name} {}", stats.line(&key));
                undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
            }
            if tag == "full" {
                let mut stride = x.to_vec();
                for (r, &p) in positions.iter().enumerate() {
                    for h in 0..heads {
                        let base = (r * heads + h) * d;
                        for i in 0..64usize {
                            let row = p as usize * d;
                            let (c, sn) = (table[row + i].to_f32(), table[row + half + i].to_f32());
                            let (a, b) = (x[base + i].to_f32(), x[base + 64 + i].to_f32());
                            stride[base + i] = bf16::from_f32(a * c - b * sn);
                            stride[base + 64 + i] = bf16::from_f32(b * c + a * sn);
                        }
                    }
                }
                let stats = Stats::new(&stride, reference)?;
                println!("GEMMA4_ROPE_MUTANT pair_stride_64 {}", stats.line(&key));
                undetected += usize::from(!(stats.is_detected_mutant() && !stats.passes()));
            }
            cases += 1;
        }
    }
    println!("GEMMA4_ROPE cases={cases} defects={defects} undetected_mutants={undetected}");
    ensure!(defects == 0 && undetected == 0, "GEMMA4_ROPE FAIL");
    Ok(())
}

#[test]
#[ignore = "requires CUDA, D512 model-definition fixtures and the published model"]
fn d512_table_kv_preparation_matches_definition_gpu() -> Result<()> {
    let device = Device::new(0)?;
    let s = &device.stream;
    let data = fixture("reference.safetensors")?;
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    let geometry = spec.layers[4]
        .geometry
        .context("D512 layer geometry missing")?;
    let (rows, d, capacity) = (14usize, 512usize, 640usize);
    let q = tensor(&data, "stage.a.q_proj")?;
    let k = tensor(&data, "stage.a.k_proj")?;
    let v = tensor(&data, "stage.a.v_proj")?;
    let q_rope = tensor(&data, "stage.a.q_rope")?;
    let q_width = spec.q_heads as usize * d;
    ensure!(
        q.len() == rows * q_width
            && q_rope.len() == rows * q_width
            && k.len() == rows * d
            && v.len() == rows * d,
        "D512 preparation fixture extents differ"
    );
    let first_head = |values: &[bf16]| -> Vec<bf16> {
        values
            .chunks_exact(q_width)
            .flat_map(|row| row[..d].iter().copied())
            .collect()
    };
    let mut qkv = Vec::with_capacity(rows * 3 * d);
    qkv.extend_from_slice(&first_head(q));
    qkv.extend_from_slice(k);
    qkv.extend_from_slice(v);
    let mut qkv = s.clone_htod(&qkv)?;
    let mut cache_k = s.alloc_zeros::<bf16>(capacity * d)?;
    let mut cache_v = s.alloc_zeros::<bf16>(capacity * d)?;
    let q_norm = s.clone_htod(&checkpoint_tensor(
        &dir,
        "model.language_model.layers.4.self_attn.q_norm.weight",
    )?)?;
    let k_norm = s.clone_htod(&checkpoint_tensor(
        &dir,
        "model.language_model.layers.4.self_attn.k_norm.weight",
    )?)?;
    let table = s.clone_htod(&geometry.rope.table(d as u32, capacity)?)?;
    let start = s.clone_htod(&[0i32])?;
    let function = cubin::module(&device.ctx, "rope")?.load_function("qknorm_rope_table_kv")?;
    unsafe {
        s.launch_builder(&function)
            .arg(&mut qkv)
            .arg(&mut cache_k)
            .arg(&mut cache_v)
            .arg(&q_norm)
            .arg(&k_norm)
            .arg(&table)
            .arg(&start)
            .arg(&-1i32)
            .arg(&(rows as i32))
            .arg(&1i32)
            .arg(&1i32)
            .arg(&(d as i32))
            .arg(&(capacity as i32))
            .arg(&0i32)
            .arg(&1e-6f32)
            .arg(&(d as u64))
            .arg(&((rows * d) as u64))
            .arg(&(d as u64))
            .arg(&((rows * 2 * d) as u64))
            .arg(&(d as u64))
            .launch(LaunchConfig {
                grid_dim: (3 * rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let got_q = s.clone_dtoh(&qkv.slice(..rows * d))?;
    let got_k = s.clone_dtoh(&cache_k.slice(..rows * d))?;
    let got_v = s.clone_dtoh(&cache_v.slice(..rows * d))?;
    let q_rope = first_head(q_rope);
    let expected = [
        ("q", got_q, &q_rope[..rows * d]),
        ("k", got_k, tensor(&data, "stage.a.k_rope")?),
        ("v", got_v, tensor(&data, "stage.a.v_norm")?),
    ];
    let mut defects = 0;
    for (name, actual, expected) in expected {
        let stats = Stats::new(&actual, expected)?;
        println!("GEMMA4_D512_PREP {name} {}", stats.line(name));
        defects += usize::from(!stats.passes());
    }
    ensure!(defects == 0, "GEMMA4_D512_PREP defects={defects}");
    Ok(())
}

fn oracle_layer_distance(
    trace: &crate::trace::Trace,
    oracle: &ReferenceFixture,
    reference_layers: &ReferenceFixture,
    family: &str,
    mode: &str,
    row: Option<usize>,
    layer: usize,
) -> Result<bool> {
    let name = if mode == "decode" {
        format!("decode_layers.{family}")
    } else {
        format!("layers.{family}")
    };
    let oracle_values = reference(oracle, &name)?.as_f32();
    let reference_values = reference(reference_layers, &name)?.as_f32();
    let candidate_values = trace.values(&format!(".{layer}"))?;
    let hidden = oracle[name.as_str()].0[2];
    let fixture_rows = oracle[name.as_str()].0[1];
    let candidate_rows = candidate_values.len() / hidden;
    ensure!(
        candidate_rows == 1 || candidate_rows == fixture_rows,
        "oracle layer candidate row extent mismatch"
    );
    if let Some(row) = row {
        ensure!(row < fixture_rows, "oracle layer row exceeds cell extent");
    }
    let layer_rows = fixture_rows * hidden;
    let oracle_values = &oracle_values[layer * layer_rows..(layer + 1) * layer_rows];
    let reference_values = &reference_values[layer * layer_rows..(layer + 1) * layer_rows];
    let (candidate_values, oracle_values, reference_values) = if let Some(row) = row {
        let candidate_row = if candidate_rows == 1 { 0 } else { row };
        (
            &candidate_values[candidate_row * hidden..(candidate_row + 1) * hidden],
            &oracle_values[row * hidden..(row + 1) * hidden],
            &reference_values[row * hidden..(row + 1) * hidden],
        )
    } else {
        (candidate_values.as_slice(), oracle_values, reference_values)
    };
    ensure!(
        candidate_values.len() == oracle_values.len()
            && candidate_values.len() == reference_values.len(),
        "oracle layer extent mismatch at layer {layer}"
    );
    let (mut candidate_max, mut reference_max) = (0.0f64, 0.0f64);
    let (mut candidate_sq, mut reference_sq) = (0.0f64, 0.0f64);
    for ((candidate, oracle), reference) in candidate_values
        .iter()
        .zip(oracle_values)
        .zip(reference_values)
    {
        let candidate_delta = f64::from(*candidate - *oracle);
        let reference_delta = f64::from(*reference - *oracle);
        candidate_max = candidate_max.max(candidate_delta.abs());
        reference_max = reference_max.max(reference_delta.abs());
        candidate_sq += candidate_delta * candidate_delta;
        reference_sq += reference_delta * reference_delta;
    }
    let candidate_rmse = (candidate_sq / candidate_values.len() as f64).sqrt();
    let reference_rmse = (reference_sq / reference_values.len() as f64).sqrt();
    let beyond = candidate_max > 2.0 * reference_max || candidate_rmse > 2.0 * reference_rmse;
    let row = row.map_or_else(|| "all".to_owned(), |row| row.to_string());
    println!(
        "GEMMA4_ORACLE_LAYER row={row} rows={} layer={layer} candidate_max={candidate_max} candidate_rmse={candidate_rmse} reference_max={reference_max} reference_rmse={reference_rmse} beyond_2x={beyond}",
        candidate_values.len() / hidden
    );
    Ok(beyond)
}

fn oracle_store_distance(
    channel: &str,
    candidate_values: &[f32],
    reference_values: &[f32],
    oracle_values: &[f32],
    row: usize,
    rows: usize,
    store: &str,
) -> Result<bool> {
    ensure!(
        reference_values.len() == oracle_values.len()
            && reference_values.len().is_multiple_of(rows)
            && row < rows,
        "oracle store extent mismatch at {store}"
    );
    let width = reference_values.len() / rows;
    ensure!(
        candidate_values.len() == reference_values.len() || candidate_values.len() == width,
        "oracle store candidate extent mismatch at {store}"
    );
    let candidate_values = if candidate_values.len() == width {
        candidate_values
    } else {
        &candidate_values[row * width..(row + 1) * width]
    };
    let reference_values = &reference_values[row * width..(row + 1) * width];
    let oracle_values = &oracle_values[row * width..(row + 1) * width];
    let (mut candidate_max, mut reference_max) = (0.0f64, 0.0f64);
    let (mut candidate_sq, mut reference_sq) = (0.0f64, 0.0f64);
    let mut reference_bit_equal = 0usize;
    let mut oracle_rounded_bit_equal = 0usize;
    let mut max_index = 0usize;
    let (mut candidate_at_max, mut reference_at_max, mut oracle_at_max) = (0.0f64, 0.0f64, 0.0f64);
    for (index, ((candidate, reference), oracle)) in candidate_values
        .iter()
        .zip(reference_values)
        .zip(oracle_values)
        .enumerate()
    {
        let candidate_delta = f64::from(*candidate - *oracle);
        let reference_delta = f64::from(*reference - *oracle);
        if candidate_delta.abs() > candidate_max {
            max_index = index;
            candidate_at_max = f64::from(*candidate);
            reference_at_max = f64::from(*reference);
            oracle_at_max = f64::from(*oracle);
        }
        candidate_max = candidate_max.max(candidate_delta.abs());
        reference_max = reference_max.max(reference_delta.abs());
        candidate_sq += candidate_delta * candidate_delta;
        reference_sq += reference_delta * reference_delta;
        reference_bit_equal += usize::from(candidate.to_bits() == reference.to_bits());
        oracle_rounded_bit_equal +=
            usize::from(candidate.to_bits() == bf16::from_f32(*oracle).to_f32().to_bits());
    }
    let candidate_rmse = (candidate_sq / width as f64).sqrt();
    let reference_rmse = (reference_sq / width as f64).sqrt();
    let beyond = candidate_max > 2.0 * reference_max || candidate_rmse > 2.0 * reference_rmse;
    let interval = 2.0 * (reference_at_max - oracle_at_max).abs();
    let interval_low = oracle_at_max - interval;
    let interval_high = oracle_at_max + interval;
    println!(
        "GEMMA4_ORACLE_STORE channel={channel} row={row} store={store} width={width} candidate_max={candidate_max} candidate_rmse={candidate_rmse} reference_max={reference_max} reference_rmse={reference_rmse} reference_bit_equal={reference_bit_equal} oracle_rounded_bit_equal={oracle_rounded_bit_equal} max_index={max_index} candidate_value={candidate_at_max} reference_value={reference_at_max} oracle_value={oracle_at_max} interval_low={interval_low} interval_high={interval_high} beyond_2x={beyond}"
    );
    Ok(beyond)
}

fn k_projection_rule_b(
    candidate: &[bf16],
    input: &[bf16],
    reference: &[bf16],
    weight: &[bf16],
    row: usize,
) -> Result<()> {
    let hidden = 1536usize;
    let width = 256usize;
    ensure!(
        candidate.len() == reference.len()
            && candidate.len().is_multiple_of(width)
            && input.len() == candidate.len() / width * hidden
            && weight.len() == width * hidden
            && row < candidate.len() / width,
        "K projection rule-B extent"
    );
    let gamma = hidden as f64 * 2f64.powi(-24);
    let mut differing = 0usize;
    let mut outside = [0usize; 2];
    let mut max_excess = 0.0f64;
    let mut max_excess_index = 0usize;
    for column in 0..width {
        let index = row * width + column;
        if candidate[index].to_bits() == reference[index].to_bits() {
            continue;
        }
        differing += 1;
        let mut sum = 0.0f64;
        let mut absolute = 0.0f64;
        for k in 0..hidden {
            let product = input[row * hidden + k].to_f64() * weight[column * hidden + k].to_f64();
            sum += product;
            absolute += product.abs();
        }
        let interval = gamma * absolute;
        let low = bf16::from_f64(sum - interval).to_f64();
        let high = bf16::from_f64(sum + interval).to_f64();
        let candidate_value = candidate[index].to_f64();
        let reference_value = reference[index].to_f64();
        let candidate_inside = candidate_value >= low && candidate_value <= high;
        let reference_inside = reference_value >= low && reference_value <= high;
        outside[0] += usize::from(!candidate_inside);
        outside[1] += usize::from(!reference_inside);
        let excess = (low - candidate_value)
            .max(candidate_value - high)
            .max(low - reference_value)
            .max(reference_value - high)
            .max(0.0);
        if excess > max_excess {
            max_excess = excess;
            max_excess_index = column;
        }
        println!(
            "GEMMA4_K_RULE_B row={row} index={column} candidate={candidate_value} reference={reference_value} fp64_sum={sum} fp64_bf16={} interval_low={low} interval_high={high} candidate_inside={candidate_inside} reference_inside={reference_inside} excess={excess}",
            bf16::from_f64(sum).to_f64()
        );
    }
    println!(
        "GEMMA4_K_RULE_B_SUMMARY row={row} differing={differing} candidate_outside={} reference_outside={} max_excess={max_excess} max_excess_index={max_excess_index}",
        outside[0], outside[1]
    );
    Ok(())
}

fn oracle_store_set(
    channel: &str,
    prefix: &str,
    stores: &[(&'static str, Vec<f32>)],
    oracle: &ReferenceFixture,
    reference_layers: &ReferenceFixture,
    selected_rows: &[usize],
    rows: usize,
) -> Result<Option<&'static str>> {
    let mut first_beyond = None;
    for (store, candidate) in stores {
        let stage_name = format!("{prefix}.{store}");
        let reference_values = reference(reference_layers, &stage_name)?.as_f32();
        let oracle_values = reference(oracle, &stage_name)?.as_f32();
        for row in selected_rows {
            if oracle_store_distance(
                channel,
                candidate,
                &reference_values,
                &oracle_values,
                *row,
                rows,
                store,
            )? && first_beyond.is_none()
            {
                first_beyond = Some(*store);
            }
        }
    }
    Ok(first_beyond)
}

fn row_nll(values: impl Iterator<Item = f32> + Clone, target: f32) -> f64 {
    let peak = values.clone().fold(f32::NEG_INFINITY, f32::max);
    let sum = values.map(|value| (value - peak).exp() as f64).sum::<f64>();
    -(target - peak) as f64 + sum.ln()
}

fn oracle_token_metrics(
    channel: &str,
    actual: &[bf16],
    reference: &[bf16],
    oracle: &[f32],
    target: Option<usize>,
    row: usize,
) -> Result<[f64; 3]> {
    ensure!(
        actual.len() == reference.len() && actual.len() == oracle.len(),
        "oracle token logit extent mismatch"
    );
    let (mut candidate_max, mut reference_max) = (0.0f64, 0.0f64);
    let (mut candidate_sq, mut reference_sq) = (0.0f64, 0.0f64);
    for ((candidate, reference), oracle) in actual.iter().zip(reference).zip(oracle) {
        let candidate_delta = f64::from(candidate.to_f32() - *oracle);
        let reference_delta = f64::from(reference.to_f32() - *oracle);
        candidate_max = candidate_max.max(candidate_delta.abs());
        reference_max = reference_max.max(reference_delta.abs());
        candidate_sq += candidate_delta * candidate_delta;
        reference_sq += reference_delta * reference_delta;
    }
    let candidate_rmse = (candidate_sq / actual.len() as f64).sqrt();
    let reference_rmse = (reference_sq / actual.len() as f64).sqrt();
    let mut best = [f32::NEG_INFINITY; 2];
    for &score in oracle {
        if score > best[0] {
            best[1] = best[0];
            best[0] = score;
        } else if score > best[1] {
            best[1] = score;
        }
    }
    let oracle_margin = best[0] - best[1];
    let losses = if let Some(target) = target {
        [
            row_nll(
                actual.iter().map(|value| value.to_f32()),
                actual[target].to_f32(),
            ),
            row_nll(
                reference.iter().map(|value| value.to_f32()),
                reference[target].to_f32(),
            ),
            row_nll(oracle.iter().copied(), oracle[target]),
        ]
    } else {
        [0.0; 3]
    };
    let candidate_rank = target.map(|target| {
        1 + actual
            .iter()
            .filter(|value| value.to_f32() > actual[target].to_f32())
            .count()
    });
    let reference_rank = target.map(|target| {
        1 + reference
            .iter()
            .filter(|value| value.to_f32() > reference[target].to_f32())
            .count()
    });
    let oracle_rank = target.map(|target| {
        1 + oracle
            .iter()
            .filter(|value| **value > oracle[target])
            .count()
    });
    println!(
        "GEMMA4_ORACLE_TOKEN channel={channel} row={row} target={target:?} candidate_nll={} reference_nll={} oracle_nll={} candidate_delta={} reference_delta={} candidate_max={candidate_max} candidate_rmse={candidate_rmse} reference_max={reference_max} reference_rmse={reference_rmse} oracle_margin={oracle_margin} candidate_rank={candidate_rank:?} reference_rank={reference_rank:?} oracle_rank={oracle_rank:?}",
        losses[0],
        losses[1],
        losses[2],
        losses[0] - losses[2],
        losses[1] - losses[2]
    );
    Ok(losses)
}

fn oracle_nll_summary(
    actual: &[bf16],
    reference_values: &[bf16],
    oracle_values: &[f32],
    sequence: &[u32],
    bounds: &serde_json::Value,
    losses: [f64; 3],
) -> Result<()> {
    let count = sequence.len().saturating_sub(1).max(1) as f64;
    let candidate_sum = (losses[0] / count - losses[2] / count).abs();
    let reference_sum = (losses[1] / count - losses[2] / count).abs();
    let (_, candidate_line, candidate_cell) =
        model_metrics(actual, oracle_values, sequence, bounds)?;
    let (_, reference_line, reference_cell) =
        model_metrics(reference_values, oracle_values, sequence, bounds)?;
    ensure!(
        candidate_sum.to_bits() == candidate_cell.to_bits()
            && reference_sum.to_bits() == reference_cell.to_bits(),
        "per-row NLL contributions do not reconcile with the cell metric"
    );
    println!(
        "GEMMA4_ORACLE_NLL_SUM candidate_sum={candidate_sum} reference_sum={reference_sum} candidate_cell={candidate_cell} reference_cell={reference_cell} reconciled=true"
    );
    println!("GEMMA4_ORACLE_NLL_CELL candidate {candidate_line}");
    println!("GEMMA4_ORACLE_NLL_CELL reference {reference_line}");
    Ok(())
}

fn oracle_first_layer_rule_a(
    engine: &mut super::engine::Engine,
    oracle: &ReferenceFixture,
    reference_layers: &ReferenceFixture,
    family: &str,
    sequence: &[u32],
    selected_rows: &[usize],
) -> Result<()> {
    let rows = sequence.len();
    let prefix = format!("stage.{family}");
    let input = reference_bf16(reference_layers, &format!("{prefix}.embed"))?;
    let mut trace = crate::trace::Trace::default();
    engine.gemma_first_layer_probe(sequence, input, &mut trace)?;
    let (cache_k, cache_v) = engine.gemma_kv_rows(0, rows)?;
    let cache_k: Vec<f32> = cache_k.into_iter().map(|value| value.to_f32()).collect();
    let cache_v: Vec<f32> = cache_v.into_iter().map(|value| value.to_f32()).collect();
    let raw = trace.values("layer_rule_a.debug.qkv_projected")?;
    let q_width = reference(oracle, &format!("{prefix}.q_proj"))?
        .as_f32()
        .len()
        / rows;
    let kv_width = reference(oracle, &format!("{prefix}.k_proj"))?
        .as_f32()
        .len()
        / rows;
    ensure!(
        raw.len() == rows * (q_width + 2 * kv_width),
        "first-layer raw QKV extent mismatch"
    );
    let input: Vec<f32> = input.iter().map(|value| value.to_f32()).collect();
    let stores = vec![
        ("embed", input),
        ("input_layernorm", trace.values("layer_rule_a.debug.norm1")?),
        ("q_proj", raw[..rows * q_width].to_vec()),
        (
            "k_proj",
            raw[rows * q_width..rows * (q_width + kv_width)].to_vec(),
        ),
        ("v_proj", raw[rows * (q_width + kv_width)..].to_vec()),
        ("q_rope", trace.values("layer_rule_a.debug.qkv_prepared")?),
        ("k_rope", cache_k),
        ("v_norm", cache_v),
        (
            "attention_out",
            trace.values("layer_rule_a.debug.mixer_out")?,
        ),
        ("self_attn", trace.values("layer_rule_a.debug.o_proj")?),
        (
            "pre_feedforward_layernorm",
            trace.values("layer_rule_a.debug.norm2")?,
        ),
        (
            "mlp_gate_proj",
            trace.values("layer_rule_a.debug.mlp_gate")?,
        ),
        ("mlp_up_proj", trace.values("layer_rule_a.debug.mlp_up")?),
        (
            "mlp_activation",
            trace.values("layer_rule_a.debug.mlp_activation")?,
        ),
        (
            "mlp_product",
            trace.values("layer_rule_a.debug.mlp_product")?,
        ),
        ("mlp", trace.values("layer_rule_a.ffn")?),
        (
            "per_layer_input_gate",
            trace.values("layer_rule_a.debug.ple_gate")?,
        ),
        ("act_fn", trace.values("layer_rule_a.debug.ple_act")?),
        (
            "per_layer_projection",
            trace.values("layer_rule_a.debug.per_layer_input")?,
        ),
    ];
    let layer_name = format!("layers.{family}");
    let layer_values = trace.values("layer_rule_a.layer.0")?;
    let mut first_beyond = oracle_store_set(
        "first_layer_rule_a",
        &prefix,
        &stores,
        oracle,
        reference_layers,
        selected_rows,
        rows,
    )?;
    let reference_values = reference(reference_layers, &layer_name)?.as_f32();
    let oracle_values = reference(oracle, &layer_name)?.as_f32();
    let hidden = layer_values.len() / rows;
    for row in selected_rows {
        if oracle_store_distance(
            "first_layer_rule_a",
            &layer_values,
            &reference_values[..rows * hidden],
            &oracle_values[..rows * hidden],
            *row,
            rows,
            "layer_output",
        )? && first_beyond.is_none()
        {
            first_beyond = Some("layer_output");
        }
    }
    let widen = |values: Vec<bf16>| {
        values
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()
    };
    let norm1 = reference_bf16(reference_layers, &format!("{prefix}.input_layernorm"))?;
    let k_projection = engine.gemma_linear_probe("self_attn.k_proj.weight", norm1)?;
    #[cfg(feature = "cuda")]
    {
        let plans = engine.gemma_linear_plan_receipt()?;
        for output in [256usize, 2048] {
            let selected: Vec<_> = plans["lt"]
                .as_array()
                .context("Lt plan list missing")?
                .iter()
                .filter(|plan| {
                    plan["rows"] == rows && plan["output"] == output && plan["input"] == 1536
                })
                .collect();
            println!(
                "GEMMA4_LINEAR_PLAN rows={rows} output={output} input=1536 {}",
                serde_json::to_string(&selected)?
            );
        }
    }
    let model_dir = std::path::PathBuf::from(
        std::env::var_os("FF_MODEL").context("FF_MODEL missing for K rule B")?,
    );
    let k_weight = checkpoint_tensor(
        &model_dir,
        "model.language_model.layers.0.self_attn.k_proj.weight",
    )?;
    let reference_k = reference_bf16(reference_layers, &format!("{prefix}.k_proj"))?;
    for row in selected_rows {
        k_projection_rule_b(&k_projection, norm1, reference_k, &k_weight, *row)?;
    }
    let attention_stores = [
        (
            "q_proj",
            widen(engine.gemma_linear_probe("self_attn.q_proj.weight", norm1)?),
        ),
        ("k_proj", widen(k_projection)),
        (
            "v_proj",
            widen(engine.gemma_linear_probe("self_attn.v_proj.weight", norm1)?),
        ),
    ];
    let mut first_attention_beyond = None;
    for (store, candidate) in attention_stores {
        let stage_name = format!("{prefix}.{store}");
        let reference_values = reference(reference_layers, &stage_name)?.as_f32();
        let oracle_values = reference(oracle, &stage_name)?.as_f32();
        for row in selected_rows {
            if oracle_store_distance(
                "attention_operand_rule_a",
                &candidate,
                &reference_values,
                &oracle_values,
                *row,
                rows,
                store,
            )? && first_attention_beyond.is_none()
            {
                first_attention_beyond = Some(store);
            }
        }
    }
    let pre_ffn = reference_bf16(
        reference_layers,
        &format!("{prefix}.pre_feedforward_layernorm"),
    )?;
    let reference_gate = reference_bf16(reference_layers, &format!("{prefix}.mlp_gate_proj"))?;
    let reference_up = reference_bf16(reference_layers, &format!("{prefix}.mlp_up_proj"))?;
    let reference_product = reference_bf16(reference_layers, &format!("{prefix}.mlp_product"))?;
    let gate = engine.gemma_linear_probe("mlp.gate_proj.weight", pre_ffn)?;
    let up = engine.gemma_linear_probe("mlp.up_proj.weight", pre_ffn)?;
    let (activation, product) = engine.gemma_mlp_activation_probe(reference_gate, reference_up)?;
    let down = engine.gemma_linear_probe("mlp.down_proj.weight", reference_product)?;
    let operand_stores = [
        ("mlp_gate_proj", widen(gate)),
        ("mlp_up_proj", widen(up)),
        ("mlp_activation", widen(activation)),
        ("mlp_product", widen(product)),
        ("mlp", widen(down)),
    ];
    let mut first_operand_beyond = None;
    for (store, candidate) in operand_stores {
        let stage_name = format!("{prefix}.{store}");
        let reference_values = reference(reference_layers, &stage_name)?.as_f32();
        let oracle_values = reference(oracle, &stage_name)?.as_f32();
        for row in selected_rows {
            if oracle_store_distance(
                "mlp_operand_rule_a",
                &candidate,
                &reference_values,
                &oracle_values,
                *row,
                rows,
                store,
            )? && first_operand_beyond.is_none()
            {
                first_operand_beyond = Some(store);
            }
        }
    }
    println!(
        "GEMMA4_ORACLE_FIRST_LAYER first_beyond={first_beyond:?} attention_operand_first_beyond={first_attention_beyond:?} mlp_operand_first_beyond={first_operand_beyond:?}"
    );
    Ok(())
}

fn localize_oracle_cell(
    engine: &mut super::engine::Engine,
    oracle: &ReferenceFixture,
    reference_layers: &ReferenceFixture,
    ids: &serde_json::Value,
    cell: &str,
    bounds: &serde_json::Value,
) -> Result<()> {
    let (family, mode) = cell
        .split_once('.')
        .context("oracle cell must be family.mode")?;
    ensure!(
        matches!(family, "a" | "b" | "wrap"),
        "unknown oracle family {family}"
    );
    let sequence: Vec<u32> = if family == "wrap" {
        serde_json::from_value(ids["wrap"].clone())?
    } else {
        let index = usize::from(family == "b");
        serde_json::from_value(ids["short"][index].clone())?
    };
    let layers = oracle["layers.a"].0[0];
    let selected_rows: Option<Vec<usize>> = std::env::var_os("FF_ORACLE_ROWS")
        .map(|rows| {
            rows.to_string_lossy()
                .split(',')
                .map(|row| row.parse::<usize>().map_err(anyhow::Error::from))
                .collect()
        })
        .transpose()?;
    if let Some(rows) = &selected_rows {
        ensure!(
            !rows.is_empty() && rows.iter().all(|row| *row < sequence.len()),
            "oracle row selection is empty or exceeds the cell"
        );
    }
    match mode {
        "prefill" => {
            let mut trace = crate::trace::Trace::all_layers();
            engine.gemma_prefill_trace(&sequence, &mut trace)?;
            let mut beyond = false;
            if let Some(rows) = &selected_rows {
                for layer in 0..layers {
                    for row in rows {
                        beyond |= oracle_layer_distance(
                            &trace,
                            oracle,
                            reference_layers,
                            family,
                            "prefill",
                            Some(*row),
                            layer,
                        )?;
                    }
                }
            } else {
                for layer in 0..layers {
                    beyond |= oracle_layer_distance(
                        &trace,
                        oracle,
                        reference_layers,
                        family,
                        "prefill",
                        None,
                        layer,
                    )?;
                }
            }
            let name = format!("{family}.prefill");
            let oracle_values = reference(oracle, &name)?.as_f32();
            let reference_values = reference_bf16(reference_layers, &name)?;
            let vocab = oracle_values.len() / sequence.len();
            let mut actual_values = Vec::with_capacity(oracle_values.len());
            let mut losses = [0.0; 3];
            for row in 0..sequence.len() {
                let actual = engine.gemma_all_logits(row)?;
                let contribution = oracle_token_metrics(
                    "body",
                    &actual,
                    &reference_values[row * vocab..(row + 1) * vocab],
                    &oracle_values[row * vocab..(row + 1) * vocab],
                    sequence.get(row + 1).map(|target| *target as usize),
                    row,
                )?;
                actual_values.extend(actual);
                for (sum, value) in losses.iter_mut().zip(contribution) {
                    *sum += value;
                }
            }
            if !beyond {
                oracle_nll_summary(
                    &actual_values,
                    reference_values,
                    &oracle_values,
                    &sequence,
                    bounds,
                    losses,
                )?;
            }
            if let Some(rows) = &selected_rows {
                oracle_first_layer_rule_a(
                    engine,
                    oracle,
                    reference_layers,
                    family,
                    &sequence,
                    rows,
                )?;
                for (channel, hidden_fixture) in [
                    ("head_oracle", oracle),
                    ("head_bf16_reference", reference_layers),
                ] {
                    let layer_name = format!("layers.{family}");
                    let layer_values = reference(hidden_fixture, &layer_name)?.as_f32();
                    let hidden = hidden_fixture[layer_name.as_str()].0[2];
                    let final_base = (layers - 1) * sequence.len() * hidden;
                    let mut final_input = Vec::with_capacity(rows.len() * hidden);
                    for row in rows {
                        let base = final_base + row * hidden;
                        final_input.extend(
                            layer_values[base..base + hidden]
                                .iter()
                                .map(|value| bf16::from_f32(*value)),
                        );
                    }
                    println!(
                        "GEMMA4_ORACLE_HEAD_INPUT channel={channel} rows={rows:?} storage=bf16"
                    );
                    let head_values = engine.gemma_final_probe(&final_input)?;
                    for (probe_row, row) in rows.iter().enumerate() {
                        oracle_token_metrics(
                            channel,
                            &head_values[probe_row * vocab..(probe_row + 1) * vocab],
                            &reference_values[*row * vocab..(*row + 1) * vocab],
                            &oracle_values[*row * vocab..(*row + 1) * vocab],
                            sequence.get(*row + 1).map(|target| *target as usize),
                            *row,
                        )?;
                    }
                }
            }
        }
        "decode" => {
            let name = format!("{family}.decode");
            let oracle_values = reference(oracle, &name)?.as_f32();
            let reference_values = reference_bf16(reference_layers, &name)?;
            let vocab = oracle_values.len() / sequence.len();
            let mut actual_values = Vec::with_capacity(oracle_values.len());
            let mut losses = [0.0; 3];
            let mut beyond = false;
            for (row, &id) in sequence.iter().enumerate() {
                let mut trace = crate::trace::Trace::all_layers();
                engine.gemma_decode_trace(id, &mut trace)?;
                for layer in 0..layers {
                    beyond |= oracle_layer_distance(
                        &trace,
                        oracle,
                        reference_layers,
                        family,
                        "decode",
                        Some(row),
                        layer,
                    )?;
                }
                let actual: Vec<bf16> = engine
                    .decode_logits()?
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect();
                let contribution = oracle_token_metrics(
                    "body",
                    &actual,
                    &reference_values[row * vocab..(row + 1) * vocab],
                    &oracle_values[row * vocab..(row + 1) * vocab],
                    sequence.get(row + 1).map(|target| *target as usize),
                    row,
                )?;
                actual_values.extend(actual);
                for (sum, value) in losses.iter_mut().zip(contribution) {
                    *sum += value;
                }
            }
            if !beyond {
                oracle_nll_summary(
                    &actual_values,
                    reference_values,
                    &oracle_values,
                    &sequence,
                    bounds,
                    losses,
                )?;
            }
        }
        _ => anyhow::bail!("unknown oracle cell mode {mode}"),
    }
    Ok(())
}

fn model_metrics(
    actual: &[bf16],
    expected: &[f32],
    ids: &[u32],
    bounds: &serde_json::Value,
) -> Result<(bool, String, f64)> {
    ensure!(
        actual.len() == expected.len(),
        "model logit extent mismatch"
    );
    ensure!(!ids.is_empty(), "model metric requires rows");
    ensure!(
        actual.iter().all(|v| v.to_f32().is_finite()) && expected.iter().all(|v| v.is_finite()),
        "Runner FAIL non-finite model logits"
    );
    let vocab = expected.len() / ids.len();
    ensure!(
        expected.len() == ids.len() * vocab,
        "model logit rows mismatch"
    );
    let mut maximum = 0.0f64;
    let mut squared = 0.0f64;
    let mut losses = [0.0f64; 2];
    let mut qualified = 0usize;
    let mut id_defects = 0usize;
    let margin_cut = bounds["margin_cut"].as_f64().context("margin cut")?;
    for row in 0..ids.len() {
        let a = &actual[row * vocab..(row + 1) * vocab];
        let b = &expected[row * vocab..(row + 1) * vocab];
        let mut reference_best = 0usize;
        for (a, b) in a.iter().zip(b) {
            let delta = f64::from(a.to_f32() - *b);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
        }
        let mut best = [f32::NEG_INFINITY; 2];
        let mut candidate = 0usize;
        let mut candidate_best = f32::NEG_INFINITY;
        for (i, (av, bv)) in a.iter().zip(b).enumerate() {
            let score = *bv;
            if score > best[0] {
                best[1] = best[0];
                best[0] = score;
                reference_best = i;
            } else if score > best[1] {
                best[1] = score;
            }
            if av.to_f32() > candidate_best {
                candidate_best = av.to_f32();
                candidate = i;
            }
        }
        if f64::from(best[0] - best[1]) > margin_cut {
            qualified += 1;
            id_defects += usize::from(candidate != reference_best);
        }
        if row + 1 < ids.len() {
            let target = ids[row + 1] as usize;
            losses[0] += row_nll(a.iter().map(|value| value.to_f32()), a[target].to_f32());
            losses[1] += row_nll(b.iter().copied(), b[target]);
        }
    }
    let count = ids.len().saturating_sub(1).max(1) as f64;
    let nll_delta = (losses[0] / count - losses[1] / count).abs();
    let rmse = (squared / actual.len() as f64).sqrt();
    let pass = maximum <= bounds["output_max"].as_f64().context("output max")?
        && rmse <= bounds["output_rmse"].as_f64().context("output rmse")?
        && nll_delta <= bounds["nll_max"].as_f64().context("NLL max")?
        && id_defects == 0;
    Ok((
        pass,
        format!(
            "output_max={maximum} output_rmse={rmse} nll_delta={nll_delta} qualified_ids={qualified} id_defects={id_defects}"
        ),
        nll_delta,
    ))
}

fn write_f32_fixture(path: &std::path::Path, values: &[f32]) -> Result<()> {
    let mut body = Vec::with_capacity(values.len() * 4);
    for value in values {
        body.extend_from_slice(&value.to_le_bytes());
    }
    let header = format!(
        "{{\"a.prefill\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,{}]}}}}",
        body.len()
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&body);
    std::fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn f32_reference_distances_differ_from_bf16_rounding() -> Result<()> {
    let path = std::env::temp_dir().join(format!(
        "gemma4-f32-reference-{}.safetensors",
        std::process::id()
    ));
    // 1 + 2^-10 sits between the BF16 neighbours 1.0 and 1.0078125.
    let values = [1.0f32 + 2f32.powi(-10), 2.0, 3.0, 4.0];
    write_f32_fixture(&path, &values)?;
    let bf16_loader = load_fixture(&path);
    let reference = load_reference(&path)?;
    std::fs::remove_file(&path)?;
    ensure!(
        bf16_loader.is_err(),
        "BF16 loader must reject an F32 reference tensor"
    );
    let oracle = reference["a.prefill"].1.as_f32();
    ensure!(
        oracle == values,
        "F32 reference values changed in the loader"
    );
    let actual = [
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
    ];
    let ids = [0u32, 1];
    let bounds = serde_json::json!({
        "output_max": 0.0,
        "output_rmse": 0.0,
        "nll_max": 1e9,
        "margin_cut": 0.0
    });
    let (pass, line, _) = model_metrics(&actual, &oracle, &ids, &bounds)?;
    ensure!(
        !pass && line.starts_with("output_max=0.0009765625"),
        "F32 oracle distance was not scored: {line}"
    );
    let rounded: Vec<f32> = oracle
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let (pass, line, _) = model_metrics(&actual, &rounded, &ids, &bounds)?;
    ensure!(
        pass && line.starts_with("output_max=0 "),
        "BF16 rounding did not hide the sub-ulp distance: {line}"
    );
    Ok(())
}

fn native_path(
    engine: &mut super::engine::Engine,
    ids: &[u32],
    prefill: usize,
) -> Result<Vec<bf16>> {
    let mut out = Vec::new();
    if prefill > 0 {
        engine.gemma_prefill_all(&ids[..prefill])?;
        for row in 0..prefill {
            out.extend(engine.gemma_all_logits(row)?);
        }
    }
    for token in &ids[prefill..] {
        engine.set_token(*token)?;
        engine.decode(false, None)?;
        out.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
    }
    Ok(out)
}

fn vision_distance(actual: &[f32], expected: &[f32], bounds: &serde_json::Value) -> Result<bool> {
    ensure!(
        actual.len() == expected.len()
            && !actual.is_empty()
            && actual.iter().chain(expected).all(|v| v.is_finite()),
        "Runner FAIL vision comparison extent or finiteness"
    );
    let mut maximum = 0.0f64;
    let mut squared = 0.0f64;
    for (a, b) in actual.iter().zip(expected) {
        let delta = f64::from(*a) - f64::from(*b);
        maximum = maximum.max(delta.abs());
        squared += delta * delta;
    }
    let rmse = (squared / actual.len() as f64).sqrt();
    let pass = maximum <= bounds["output_max"].as_f64().context("vision max")?
        && rmse <= bounds["output_rmse"].as_f64().context("vision RMSE")?;
    println!("vision_distance max={maximum} rmse={rmse} pass={pass}");
    Ok(pass)
}

#[test]
#[ignore = "requires Gemma image references, frozen limits, published weights and a GPU window"]
fn image_model_definition_and_graph_gpu() -> Result<()> {
    use crate::vision_input::{ImageGrid, MultimodalInput};
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("FROZEN.json"))?)?;
    ensure!(frozen["status"] == "FROZEN", "vision limits are not frozen");
    let review: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("MEDIA_REVIEW.json"))?)?;
    ensure!(review["accepted"] == true, "media review missing");
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("inputs/INPUT.json"))?)?;
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(std::path::PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime directory missing")?,
        )),
    };
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    let image_token = spec
        .vision
        .as_ref()
        .context("vision config missing")?
        .image_token;
    let mut engine = super::engine::Engine::load(&dir, 0, 896, Some(896), settings)?;
    let mut defects = 0usize;
    let mut street_features = None;
    let mut street_input = None;
    for name in ["street", "winter", "wide", "long"] {
        let cell = &metadata[name];
        let bounds = &frozen["cells"][name];
        let bytes = std::fs::read(fixtures.join(format!("inputs/{name}-input.safetensors")))?;
        let data = SafeTensors::deserialize(&bytes)?;
        let ids_view = data.tensor("ids")?;
        let ids: Vec<_> = ids_view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        let modality_view = data.tensor("modalities")?;
        let pixels_view = data.tensor("pixels")?;
        ensure!(
            ids_view.dtype() == safetensors::Dtype::I32
                && modality_view.dtype() == safetensors::Dtype::U8
                && pixels_view.dtype() == safetensors::Dtype::F32,
            "image input dtype mismatch"
        );
        let pixels: Vec<_> = pixels_view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let grids = [ImageGrid {
            t: 1,
            h: u32::try_from(cell["grid_xy"][1].as_u64().context("grid h")?)?,
            w: u32::try_from(cell["grid_xy"][0].as_u64().context("grid w")?)?,
        }];
        let input = MultimodalInput {
            audio: &[],
            input_ids: &ids,
            modality_ids: modality_view.data(),
            pixel_values: &pixels,
            grids: &grids,
        };
        let identities: Vec<_> = ids
            .iter()
            .map(|&id| if id == image_token { 0 } else { id })
            .collect();
        let oracle = load_reference(&fixtures.join(format!("oracle/{name}-packed.safetensors")))?;
        let first = usize::try_from(cell["score_first"].as_u64().context("score first")?)?;
        engine.reset()?;
        let mut trace = crate::trace::Trace::default();
        engine.gemma_vision_trace(&input, &identities, &mut trace)?;
        let features = trace.values("image_features")?;
        let feature_pass = vision_distance(
            &features,
            &oracle["features"].1.as_f32(),
            &bounds["features"],
        )?;
        defects += usize::from(!feature_pass);
        let combined = trace.values(".debug.ple_identity")?;
        let ple = combined
            .get(..ids.len() * 256)
            .context("layer-major PLE trace is shorter than layer zero")?;
        let ple_pass = vision_distance(ple, &oracle["ple"].1.as_f32(), &bounds["ple"])?;
        defects += usize::from(!ple_pass);
        let mut actual = Vec::new();
        for row in first..ids.len() {
            actual.extend(engine.gemma_all_logits(row)?);
        }
        let (pass, metrics, _) = model_metrics(
            &actual,
            &oracle["logits"].1.as_f32(),
            &ids[first..],
            &bounds["logits"],
        )?;
        defects += usize::from(!pass);
        println!(
            "{} image_model {name} features={feature_pass} ple={ple_pass} {metrics}",
            if pass && feature_pass && ple_pass {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        if name == "street" {
            street_features = Some(features.clone());
            street_input = Some((
                ids.clone(),
                modality_view.data().to_vec(),
                pixels.clone(),
                grids[0],
            ));
            engine.reset()?;
            let mut wrong = crate::trace::Trace::default();
            engine.gemma_vision_trace(&input, &ids, &mut wrong)?;
            let wrong_combined = wrong.values(".debug.ple_identity")?;
            let wrong_ple = wrong_combined
                .get(..ids.len() * 256)
                .context("control PLE trace is shorter than layer zero")?;
            let detected = !vision_distance(wrong_ple, &oracle["ple"].1.as_f32(), &bounds["ple"])?;
            defects += usize::from(!detected);
            let text_differences = ids
                .iter()
                .enumerate()
                .filter(|(_, id)| **id != image_token)
                .filter(|(row, _)| ple[row * 256..][..256] != wrong_ple[row * 256..][..256])
                .count();
            defects += usize::from(text_differences != 0);
            println!(
                "image_identity_control detected={detected} text_row_differences={text_differences}"
            );
        }
        if name == "winter" {
            let detected = !vision_distance(
                street_features
                    .as_ref()
                    .context("street features missing")?,
                &oracle["features"].1.as_f32(),
                &bounds["features"],
            )?;
            defects += usize::from(!detected);
            println!("two_media_control detected={detected}");
        }
        if name == "wide" {
            let previous = street_input.as_ref().context("street input missing")?;
            let mut mixed_ids = previous.0.clone();
            mixed_ids.extend_from_slice(&ids);
            let mut mixed_modalities = previous.1.clone();
            mixed_modalities.extend_from_slice(modality_view.data());
            let mut mixed_pixels = previous.2.clone();
            mixed_pixels.extend_from_slice(&pixels);
            let mixed_grids = [previous.3, grids[0]];
            let mixed_input = MultimodalInput {
                audio: &[],
                input_ids: &mixed_ids,
                modality_ids: &mixed_modalities,
                pixel_values: &mixed_pixels,
                grids: &mixed_grids,
            };
            let mixed_identities: Vec<_> = mixed_ids
                .iter()
                .map(|&id| if id == image_token { 0 } else { id })
                .collect();
            engine.reset()?;
            let mut mixed_trace = crate::trace::Trace::default();
            engine.gemma_vision_trace(&mixed_input, &mixed_identities, &mut mixed_trace)?;
            let actual = mixed_trace.values("image_features")?;
            let mut expected = street_features
                .as_ref()
                .context("street features missing")?
                .clone();
            expected.extend_from_slice(&features);
            ensure!(
                actual.len() == expected.len() && actual.iter().all(|v| v.is_finite()),
                "Runner FAIL mixed-image feature extent or finiteness"
            );
            let differences = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            defects += usize::from(differences != 0);
            println!(
                "mixed_image_batch_invariance images=2 grids=48x48,42x60 bit_mismatches={differences}"
            );
        }
        if name == "long" {
            continue;
        }
        let prompt = first + 1;
        let prompt_input = MultimodalInput {
            audio: &[],
            input_ids: &ids[..prompt],
            modality_ids: &modality_view.data()[..prompt],
            pixel_values: &pixels,
            grids: &grids,
        };
        let mut eager = Vec::new();
        for graph in [false, true] {
            engine.reset()?;
            engine.prefill_multimodal(&prompt_input, None)?;
            let mut output: Vec<_> = engine
                .decode_logits()?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            if graph {
                engine.capture()?;
            }
            for &token in &ids[prompt..] {
                engine.set_token(token)?;
                engine.decode(graph, None)?;
                output.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
            }
            let (pass, metrics, _) = model_metrics(
                &output,
                &oracle["logits"].1.as_f32(),
                &ids[first..],
                &bounds["logits"],
            )?;
            defects += usize::from(!pass);
            println!(
                "{} image_decode {name} graph={graph} {metrics}",
                if pass { "PASS" } else { "DEFECT" }
            );
            if graph {
                let differences = output.iter().zip(&eager).filter(|(a, b)| a != b).count();
                defects += usize::from(differences != 0);
                println!("image_graph {name} bit_mismatches={differences}");
            } else {
                eager = output;
            }
        }
    }
    println!(
        "{} image_model_definition_and_graph_gpu cells=4 defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "Gemma image candidate defects={defects}");
    Ok(())
}

#[test]
#[ignore = "requires accepted audio media review, frozen model references, published weights and a GPU window"]
fn audio_model_definition_and_graph_gpu() -> Result<()> {
    use crate::vision_input::{AudioClip, MultimodalInput};
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("model missing")?);
    let fixtures = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("fixtures missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("FROZEN.json"))?)?;
    let review: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("MEDIA_REVIEW.json"))?)?;
    ensure!(
        frozen["status"] == "FROZEN"
            && frozen["candidate_observations"] == 0
            && review["accepted"] == true,
        "audio freeze or human review missing"
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("inputs/INPUT.json"))?)?;
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(std::path::PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("runtime missing")?,
        )),
    };
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    let audio_token = spec.audio.as_ref().context("audio config missing")?.token;
    let mut engine = super::engine::Engine::load(&dir, 0, 1536, Some(1536), settings.clone())?;
    let mut defects = 0;
    let mut singles = Vec::new();
    let mut stew = None;
    for name in ["stew", "wizard", "pair", "long"] {
        let cell = &metadata[name];
        let bounds = &frozen["cells"][name];
        let bytes = std::fs::read(fixtures.join(format!("inputs/{name}-input.safetensors")))?;
        let data = SafeTensors::deserialize(&bytes)?;
        let ids_view = data.tensor("ids")?;
        let modalities = data.tensor("modalities")?;
        let features = data.tensor("audio_features")?;
        let mask = data.tensor("audio_mask")?;
        ensure!(
            ids_view.dtype() == safetensors::Dtype::I32
                && modalities.dtype() == safetensors::Dtype::U8
                && features.dtype() == safetensors::Dtype::F32
                && mask.dtype() == safetensors::Dtype::BOOL,
            "audio fixture dtype mismatch"
        );
        let ids = ids_view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let values = features
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let valid = mask.data().iter().map(|v| *v != 0).collect::<Vec<_>>();
        let frames = features.shape()[1];
        let clips = values
            .chunks_exact(frames * 128)
            .zip(valid.chunks_exact(frames))
            .map(|(features, mask)| AudioClip { features, mask })
            .collect::<Vec<_>>();
        let input = MultimodalInput {
            input_ids: &ids,
            modality_ids: modalities.data(),
            pixel_values: &[],
            grids: &[],
            audio: &clips,
        };
        let identities = ids
            .iter()
            .map(|&id| if id == audio_token { 0 } else { id })
            .collect::<Vec<_>>();
        let oracle = load_reference(&fixtures.join(format!("oracle/{name}-packed.safetensors")))?;
        let first = cell["score_first"].as_u64().context("score first")? as usize;
        engine.reset()?;
        let mut trace = crate::trace::Trace::default();
        engine.gemma_vision_trace(&input, &identities, &mut trace)?;
        for (stage, width, frequency, stride) in [
            ("subsample0", 128usize, 64, 2),
            ("subsample1", 32, 32, 4),
            ("layer0", 1024, 1, 4),
            ("tower", 1536, 1, 4),
            ("features", spec.hidden as usize, 1, 4),
        ] {
            let mut actual = Vec::new();
            for (index, clip) in clips.iter().enumerate() {
                let raw = trace.values(&format!("audio.{index}.{stage}"))?;
                let row_width = width * frequency;
                ensure!(
                    raw.len().is_multiple_of(row_width)
                        && raw.len() <= clip.mask.len().div_ceil(stride) * row_width,
                    "audio stage extent mismatch"
                );
                for (row, &keep) in clip.mask.iter().step_by(stride).enumerate() {
                    if keep {
                        actual.extend_from_slice(
                            raw.get(row * row_width..(row + 1) * row_width)
                                .context("audio stage omitted a valid row")?,
                        );
                    }
                }
            }
            let pass = vision_distance(&actual, &oracle[stage].1.as_f32(), &bounds[stage])?;
            defects += usize::from(!pass);
            println!(
                "{} audio_stage {name} {stage}",
                if pass { "PASS" } else { "DEFECT" }
            );
            if stage == "features" {
                if name == "stew" || name == "wizard" {
                    singles.push(actual.clone());
                }
                if name == "pair" {
                    let expected = singles.iter().flatten().copied().collect::<Vec<_>>();
                    let mismatch = actual
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    defects += usize::from(actual.len() != expected.len() || mismatch != 0);
                    println!("audio_clip_isolation bit_mismatches={mismatch}");
                }
            }
        }
        let combined = trace.values(".debug.ple_identity")?;
        let ple = combined.get(..ids.len() * 256).context("PLE extent")?;
        let ple_pass = vision_distance(ple, &oracle["ple"].1.as_f32(), &bounds["ple"])?;
        defects += usize::from(!ple_pass);
        let mut actual = Vec::new();
        for row in first..ids.len() {
            actual.extend(engine.gemma_all_logits(row)?);
        }
        trace.add(
            "scored_logits".into(),
            vec![ids.len() - first, actual.len() / (ids.len() - first)],
            actual.clone(),
        )?;
        let diagnostics = settings.runtime_dir.as_ref().context("runtime missing")?;
        std::fs::create_dir_all(diagnostics)?;
        trace.save(&diagnostics.join(format!("{name}-native.safetensors")))?;
        let (pass, metrics, _) = model_metrics(
            &actual,
            &oracle["logits"].1.as_f32(),
            &ids[first..],
            &bounds["logits"],
        )?;
        defects += usize::from(!pass);
        println!(
            "{} audio_model {name} ple={ple_pass} {metrics}",
            if pass && ple_pass { "PASS" } else { "DEFECT" }
        );
        if name == "stew" {
            engine.reset()?;
            let mut wrong = crate::trace::Trace::default();
            engine.gemma_vision_trace(&input, &ids, &mut wrong)?;
            let wrong = wrong.values(".debug.ple_identity")?;
            let wrong = &wrong[..ids.len() * 256];
            let detected = !vision_distance(wrong, &oracle["ple"].1.as_f32(), &bounds["ple"])?;
            let text_changed = ids
                .iter()
                .enumerate()
                .filter(|(_, id)| **id != audio_token)
                .filter(|(row, _)| ple[row * 256..][..256] != wrong[row * 256..][..256])
                .count();
            defects += usize::from(!detected || text_changed != 0);
            println!("audio_identity_control detected={detected} text_rows_changed={text_changed}");
            stew = Some((
                ids.clone(),
                modalities.data().to_vec(),
                values.clone(),
                valid.clone(),
            ));
            let image_bytes = std::fs::read(fixtures.join("inputs/image-input.safetensors"))?;
            let image = SafeTensors::deserialize(&image_bytes)?;
            let vision = spec.vision.as_ref().context("vision config")?;
            let image_ids = image
                .tensor("ids")?
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| u32::from_le_bytes(*v))
                .collect::<Vec<_>>();
            let image_end = image_ids
                .iter()
                .position(|&id| id == vision.end_token)
                .context("image marker")?
                + 1;
            let image_modalities = image.tensor("modalities")?;
            let pixels = image
                .tensor("pixels")?
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect::<Vec<_>>();
            let positions = image.tensor("positions")?;
            let xy = positions
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v))
                .collect::<Vec<_>>();
            let grids = [crate::vision_input::ImageGrid {
                t: 1,
                h: (xy.iter().skip(1).step_by(2).max().context("image height")? + 1) as u32,
                w: (xy.iter().step_by(2).max().context("image width")? + 1) as u32,
            }];
            let image_input = MultimodalInput {
                input_ids: &image_ids[..image_end],
                modality_ids: &image_modalities.data()[..image_end],
                pixel_values: &pixels,
                grids: &grids,
                audio: &[],
            };
            let image_identities = image_input
                .input_ids
                .iter()
                .map(|&id| if id == vision.image_token { 0 } else { id })
                .collect::<Vec<_>>();
            engine.reset()?;
            let mut image_trace = crate::trace::Trace::default();
            engine.gemma_vision_trace(&image_input, &image_identities, &mut image_trace)?;
            let mut mixed_ids = ids[..first + 1].to_vec();
            mixed_ids.extend_from_slice(image_input.input_ids);
            let mut mixed_modalities = modalities.data()[..first + 1].to_vec();
            mixed_modalities.extend_from_slice(image_input.modality_ids);
            let mixed = MultimodalInput {
                input_ids: &mixed_ids,
                modality_ids: &mixed_modalities,
                pixel_values: &pixels,
                grids: &grids,
                audio: &clips,
            };
            let mixed_identities = mixed_ids
                .iter()
                .map(|&id| {
                    if id == vision.image_token || id == audio_token {
                        0
                    } else {
                        id
                    }
                })
                .collect::<Vec<_>>();
            engine.reset()?;
            let mut mixed_trace = crate::trace::Trace::default();
            engine.gemma_vision_trace(&mixed, &mixed_identities, &mut mixed_trace)?;
            let mut expected = singles[0].clone();
            expected.extend(image_trace.values("media_features")?);
            let actual = mixed_trace.values("media_features")?;
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            defects += usize::from(actual.len() != expected.len() || mismatches != 0);
            println!(
                "mixed_image_audio_splice bit_mismatches={mismatches} encoder_calls={:?}",
                engine.gemma_encoder_calls()?
            );
        }
        if name == "long" {
            continue;
        }
        let prompt = first + 1;
        let prompt_input = MultimodalInput {
            input_ids: &ids[..prompt],
            modality_ids: &modalities.data()[..prompt],
            pixel_values: &[],
            grids: &[],
            audio: &clips,
        };
        let mut eager = Vec::new();
        let mut eager_kv = None;
        for graph in [false, true] {
            engine.reset()?;
            engine.prefill_multimodal(&prompt_input, None)?;
            let mut output = engine
                .decode_logits()?
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>();
            if graph {
                engine.capture()?;
            }
            for &token in &ids[prompt..] {
                engine.set_token(token)?;
                engine.decode(graph, None)?;
                output.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
            }
            let calls = engine.gemma_encoder_calls()?;
            let kv = (
                engine.gemma_kv_rows(0, ids.len())?,
                engine.gemma_kv_rows(4, ids.len())?,
            );
            defects += usize::from(calls != (0, clips.len()));
            let (pass, metrics, _) = model_metrics(
                &output,
                &oracle["logits"].1.as_f32(),
                &ids[first..],
                &bounds["logits"],
            )?;
            defects += usize::from(!pass);
            println!(
                "{} audio_decode {name} graph={graph} encoder_calls={calls:?} {metrics}",
                if pass { "PASS" } else { "DEFECT" }
            );
            if graph {
                let differences = output.iter().zip(&eager).filter(|(a, b)| a != b).count();
                defects += usize::from(differences != 0);
                println!("audio_graph {name} bit_mismatches={differences}");
                let state_equal = eager_kv.as_ref() == Some(&kv);
                defects += usize::from(!state_equal);
                println!("audio_graph_state {name} kv_equal={state_equal}");
            } else {
                eager = output;
                eager_kv = Some(kv);
            }
        }
    }
    ensure!(singles.len() == 2, "two audio controls missing");
    let changed = singles[0]
        .iter()
        .zip(&singles[1])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    defects += usize::from(changed == 0);
    println!("audio_two_media changed_prefix_elements={changed}");
    drop(engine);
    let (ids, modalities, values, valid) = stew.context("stew chunk fixture")?;
    let clips = [AudioClip {
        features: &values,
        mask: &valid,
    }];
    let first = metadata["stew"]["score_first"]
        .as_u64()
        .context("score first")? as usize;
    let input = MultimodalInput {
        input_ids: &ids[..first + 1],
        modality_ids: &modalities[..first + 1],
        pixel_values: &[],
        grids: &[],
        audio: &clips,
    };
    let mut engine = super::engine::Engine::load(&dir, 0, 1536, Some(128), settings)?;
    engine.prefill_multimodal(&input, None)?;
    let mut output = engine
        .decode_logits()?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    for &token in &ids[first + 1..] {
        engine.set_token(token)?;
        engine.decode(false, None)?;
        output.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
    }
    let oracle = load_reference(&fixtures.join("oracle/stew-packed.safetensors"))?;
    let (pass, metrics, _) = model_metrics(
        &output,
        &oracle["logits"].1.as_f32(),
        &ids[first..],
        &frozen["cells"]["stew"]["logits"],
    )?;
    let calls = engine.gemma_encoder_calls()?;
    defects += usize::from(!pass || calls != (0, 1));
    println!(
        "{} audio_chunk_splice chunk=128 encoder_calls={calls:?} {metrics}",
        if pass { "PASS" } else { "DEFECT" }
    );
    println!(
        "{} audio_model_definition_and_graph_gpu cells=4 defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "audio candidate defects={defects}");
    Ok(())
}

fn checkpoint_tensor(dir: &std::path::Path, name: &str) -> Result<Vec<bf16>> {
    let mapped = crate::checkpoint::MappedCheckpoint::open(dir)?;
    let shard = mapped
        .index
        .weight_map
        .get(name)
        .context("checkpoint tensor is absent")?;
    let tensors = mapped.views()?;
    let view = tensors[shard].tensor(name)?;
    let mut bytes = vec![0u8; view.data().len()];
    mapped.read_mapped_range(shard, view.data(), &mut bytes)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16,
        "{name} is not BF16"
    );
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect())
}

#[cfg(feature = "cuda")]
fn no_ple_exit_control(
    engine: &mut super::engine::Engine,
    dir: &std::path::Path,
    hidden: usize,
) -> Result<usize> {
    let mut defects = 0;
    let post = checkpoint_tensor(
        dir,
        "model.language_model.layers.0.post_feedforward_layernorm.weight",
    )?;
    let next = checkpoint_tensor(dir, "model.language_model.layers.1.input_layernorm.weight")?;
    let scale = checkpoint_tensor(dir, "model.language_model.layers.0.layer_scalar")?[0].to_f32();
    ensure!(
        scale != 1.0,
        "layer-exit scalar control is not discriminating"
    );
    for rows in [1, 17] {
        engine.reset()?;
        let x: Vec<_> = (0..rows * hidden)
            .map(|i| bf16::from_f32(((i * 37 % 1009) as f32 - 504.0) / 256.0))
            .collect();
        let branch: Vec<_> = (0..rows * hidden)
            .map(|i| bf16::from_f32(((i * 71 % 997) as f32 - 498.0) / 128.0))
            .collect();
        let normalized: Vec<_> = branch
            .chunks(hidden)
            .flat_map(|row| rms_f32(&row.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), &post))
            .collect();
        let unscaled: Vec<_> = x
            .iter()
            .zip(&normalized)
            .map(|(x, b)| bf16::from_f32(x.to_f32() + b.to_f32()))
            .collect();
        let expected_x: Vec<_> = unscaled
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() * scale))
            .collect();
        let expected_n: Vec<_> = expected_x
            .chunks(hidden)
            .flat_map(|row| rms_f32(&row.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), &next))
            .collect();
        let floor_x: Vec<_> = x
            .iter()
            .map(|v| 2.0 * scale.abs() * bf16_ulp(v.to_f32()))
            .collect();
        let floor_n: Vec<_> = expected_x
            .chunks(hidden)
            .zip(floor_x.chunks(hidden))
            .zip(expected_n.chunks(hidden))
            .flat_map(|((row, floor), norm)| {
                let mean = row.iter().map(|v| v.to_f32() * v.to_f32()).sum::<f32>() / hidden as f32;
                let inv = 1.0 / (mean + 1e-6).sqrt();
                floor
                    .iter()
                    .zip(&next)
                    .zip(norm)
                    .map(|((error, weight), value)| {
                        inv * weight.to_f32().abs() * error + bf16_ulp(value.to_f32())
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let (actual_x, actual_n) = engine.gemma_exit_probe(&x, &branch)?;
        for (tag, actual, expected, floor) in [
            ("x", &actual_x, &expected_x, &floor_x),
            ("n", &actual_n, &expected_n, &floor_n),
        ] {
            let stats = Stats::with_floor(actual, expected, Some(floor))?;
            println!(
                "GEMMA4_NO_PLE_EXIT rows={rows} hidden={hidden} scalar={scale} {}",
                stats.line(tag)
            );
            defects += usize::from(!stats.passes());
        }
        let control = Stats::new(&expected_x, &unscaled)?;
        ensure!(
            control.is_detected_mutant(),
            "omitted scalar control is not discriminating"
        );
        println!("PASS no_ple_scalar_control rows={rows} hidden={hidden} omitted_scalar=detected");
    }
    engine.reset()?;
    Ok(defects)
}

#[cfg(feature = "cuda")]
fn large_model_definition(
    dir: &std::path::Path,
    settings: crate::backend::Settings,
    spec: &crate::model_spec::ModelSpec,
    data: &ReferenceFixture,
    ids: &serde_json::Value,
    frozen: &serde_json::Value,
) -> Result<()> {
    let window = spec
        .layers
        .iter()
        .find_map(|layer| layer.sliding_window)
        .context("large Gemma window missing")? as usize;
    let capacity = window + 128;
    let mut engine = super::engine::Engine::load(dir, 0, capacity, Some(window + 8), settings)?;
    let plan = engine
        .dense_placement()
        .context("large-model gate did not execute dense streaming")?;
    println!("GEMMA4_DENSE_PLACEMENT {}", serde_json::to_string(plan)?);
    ensure!(
        plan.host_bytes > 0,
        "large-model host tier was not executed"
    );
    if spec.hidden == 5376 {
        ensure!(plan.disk_bytes > 0, "31B gate must force the disk tier");
    }
    let mut defects = no_ple_exit_control(&mut engine, dir, spec.hidden as usize)?;
    let mut wrap_eager = Vec::new();
    let mut wrap_state = None;
    let global = spec
        .layers
        .iter()
        .position(|layer| layer.sliding_window.is_none())
        .context("global layer missing")?;
    for tag in ["a", "b", "wrap", "long"] {
        let sequence: Vec<u32> = serde_json::from_value(if matches!(tag, "a" | "b") {
            ids["short"][usize::from(tag == "b")].clone()
        } else {
            ids[tag].clone()
        })?;
        for mode in ["prefill", "decode"] {
            engine.reset()?;
            let name = format!("{tag}.{mode}");
            let rows = data[&name].0[0];
            let actual = if tag == "long" && mode == "decode" {
                let mut values = Vec::new();
                for (piece, chunk) in sequence.chunks(window / 2).enumerate() {
                    engine.gemma_prefill_all(chunk)?;
                    let start = piece * (window / 2);
                    for row in 0..chunk.len() {
                        if start + row >= sequence.len() - rows {
                            values.extend(engine.gemma_all_logits(row)?);
                        }
                    }
                }
                values
            } else {
                let prefill = if mode == "prefill" {
                    sequence.len()
                } else if tag == "wrap" {
                    window
                } else {
                    0
                };
                let mut values = native_path(&mut engine, &sequence, prefill)?;
                drop(values.drain(..values.len() - rows * spec.vocab as usize));
                values
            };
            let (pass, line, _) = model_metrics(
                &actual,
                &reference(data, &name)?.as_f32(),
                &sequence[sequence.len() - rows..],
                &frozen["families"][tag],
            )?;
            println!(
                "GEMMA4_MODEL {tag} {mode} {} {line}",
                if pass { "PASS" } else { "DEFECT" }
            );
            defects += usize::from(!pass);
            if tag == "wrap" && mode == "decode" {
                wrap_eager = actual;
                wrap_state = Some((
                    engine.gemma_kv_rows(0, window)?,
                    engine.gemma_kv_rows(global, sequence.len())?,
                    engine.gemma_hidden_state()?,
                ));
            }
        }
    }
    let sequence: Vec<u32> = serde_json::from_value(ids["wrap"].clone())?;
    engine.reset()?;
    engine.gemma_prefill_all(&sequence[..window])?;
    let before = (
        engine.gemma_kv_rows(0, window)?,
        engine.gemma_kv_rows(global, window)?,
    );
    engine.capture()?;
    let restored = before
        == (
            engine.gemma_kv_rows(0, window)?,
            engine.gemma_kv_rows(global, window)?,
        );
    defects += usize::from(!restored);
    let mut graph = Vec::new();
    for &token in &sequence[window..] {
        engine.set_token(token)?;
        engine.decode(true, None)?;
        graph.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
    }
    let differing = graph
        .iter()
        .zip(&wrap_eager)
        .filter(|(a, b)| a != b)
        .count();
    let final_state = (
        engine.gemma_kv_rows(0, window)?,
        engine.gemma_kv_rows(global, sequence.len())?,
        engine.gemma_hidden_state()?,
    );
    let state_equal = wrap_state.as_ref() == Some(&final_state);
    defects += usize::from(differing != 0 || !state_equal);
    let (pass, line, _) = model_metrics(
        &graph,
        &reference(data, "wrap.decode")?.as_f32(),
        &sequence[window..],
        &frozen["families"]["wrap"],
    )?;
    defects += usize::from(!pass);
    println!(
        "GEMMA4_MODEL wrap graph {} {line}",
        if pass { "PASS" } else { "DEFECT" }
    );
    println!("GEMMA4_GRAPH_EAGER bits={differing} restored={restored} state_equal={state_equal}");
    println!(
        "GEMMA4_MODEL {} cells=9 defects={defects} window={window} long_targets=512 streamed_graph=true",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(defects == 0, "large Gemma candidate defects={defects}");
    Ok(())
}

#[test]
#[ignore = "requires Gemma 4 model-definition fixtures, FROZEN.json, model weights and a GPU window"]
fn model_definition_prefill_decode_wrap_and_graph_gpu() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let fixtures = std::path::PathBuf::from(
        std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?,
    );
    let frozen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("FROZEN.json"))?)?;
    ensure!(
        frozen["status"] == "FROZEN",
        "model thresholds are not frozen"
    );
    let ids: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("ids.json"))?)?;
    let data = reference_fixture("reference.safetensors")?;
    let settings = crate::backend::Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: true,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(std::path::PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("FLYINGFISH_RUNTIME_DIR missing")?,
        )),
    };
    let runtime = settings
        .runtime_dir
        .clone()
        .context("runtime directory missing")?;
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    if spec.per_layer_input.is_none() {
        #[cfg(feature = "cuda")]
        return large_model_definition(&dir, settings, &spec, &data, &ids, &frozen);
        #[cfg(feature = "rocm")]
        anyhow::bail!("large Gemma streamed text gate requires CUDA");
    }
    let device = Device::new(0)?;
    let catalog = super::weights::WeightCatalog::load(&dir, &spec, &device)?;
    for full in [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.self_attn.k_norm.weight",
        "model.language_model.layers.4.self_attn.k_norm.weight",
    ] {
        let actual = catalog.bf16(full)?;
        let actual = device.stream.clone_dtoh(actual)?;
        let expected = checkpoint_tensor(&dir, full)?;
        ensure!(actual == expected, "published tensor mismatch: {full}");
    }
    drop(catalog);
    let admission_stream = device.stream.clone();
    let mut engine = super::engine::Engine::load(&dir, 0, 640, Some(640), settings.clone())?;
    let (admission_rows, admission_capacity) = (65_504usize, 131_072usize);
    let per_layer_inputs = admission_rows * spec.layers.len() * 256 * 2;
    let rope_tables = admission_capacity * (256 + 512) * 2;
    let flash = spec.layers.iter().try_fold(0usize, |total, layer| {
        let geometry = layer.geometry.context("Gemma geometry missing")?;
        super::flash::Flash::workspace(
            &admission_stream,
            admission_rows,
            spec.q_heads as usize,
            geometry.head_dim as usize,
            admission_capacity,
        )
        .map(|bytes| total + bytes)
    })?;
    let staging = (512 + admission_rows) * spec.kv_heads as usize * spec.head_dim as usize * 4;
    let ring = spec
        .layers
        .iter()
        .filter_map(|l| l.sliding_window)
        .max()
        .context("Gemma sliding window missing")? as usize;
    let decode = spec.q_heads as usize * ring.div_ceil(32) * (spec.head_dim as usize + 2) * 4;
    let expected = per_layer_inputs + rope_tables + flash + staging + decode;
    let admitted = engine.gemma_workspace(admission_rows, admission_capacity)?;
    println!(
        "GEMMA4_WORKSPACE admitted={admitted} expected={expected} per_layer_inputs={per_layer_inputs} rope_tables={rope_tables} flash={flash} staging={staging} decode={decode}"
    );
    ensure!(admitted == expected, "Gemma workspace admission mismatch");
    let scratch = engine.gemma_scratch_len();
    println!("GEMMA4_SCRATCH bytes={scratch}");
    ensure!(
        scratch == decode,
        "Gemma shared decode scratch extent differs"
    );
    let global_layer = spec
        .layers
        .iter()
        .position(|layer| layer.geometry.is_some_and(|g| g.head_dim == 512))
        .context("no D512 layer in Gemma model")?;
    let full_table = spec.layers[global_layer]
        .geometry
        .context("D512 layer geometry missing")?
        .rope
        .table(512, 14)?;
    let expected_table: Vec<bf16> = [0usize, 1, 13]
        .into_iter()
        .flat_map(|position| full_table[position * 512..(position + 1) * 512].to_vec())
        .collect();
    let actual_table = engine.gemma_rope_table(true, &[0, 1, 13])?;
    let mismatches = actual_table
        .iter()
        .zip(&expected_table)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "GEMMA4_D512_TABLE mismatches={mismatches} actual={:?} expected={:?} first={:?}",
        &actual_table[..16],
        &expected_table[..16],
        actual_table
            .iter()
            .zip(&expected_table)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(8)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect::<Vec<_>>()
    );
    ensure!(mismatches == 0, "D512 RoPE table content mismatch");
    if reference(&data, "stage.a.input_layernorm")?
        .as_bf16()
        .is_none()
    {
        println!("GEMMA4_PROBES skipped=f32_reference");
    } else {
        let probe_input = reference_bf16(&data, "stage.a.input_layernorm")?;
        let projected = engine.gemma_linear_probe("self_attn.q_proj.weight", probe_input)?;
        let expected = reference_bf16(&data, "stage.a.q_proj")?;
        let delta: Vec<f32> = projected
            .iter()
            .zip(expected)
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .collect();
        let maximum = delta.iter().copied().fold(0.0f32, f32::max);
        let rmse = (delta.iter().map(|v| v * v).sum::<f32>() / delta.len() as f32).sqrt();
        println!("GEMMA4_LINEAR_PROBE output_max={maximum} output_rmse={rmse}");
        let first_ids: Vec<u32> = serde_json::from_value(ids["short"][0].clone())?;
        let mut trace = crate::trace::Trace::all_layers();
        engine.gemma_prefill_trace(&first_ids, &mut trace)?;
        let native_input: Vec<bf16> = trace
            .values(".0.debug.norm1")?
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let native_projected =
            engine.gemma_linear_probe("self_attn.q_proj.weight", &native_input)?;
        let raw = trace.values(".0.debug.qkv_projected")?;
        let rows = first_ids.len();
        let q_width = native_projected.len() / rows;
        let v_shape = &data["stage.a.v_norm"].0;
        let (hkv, d) = (v_shape[1], v_shape[2]);
        let kv_width = hkv * d;
        let delta: Vec<f32> = native_projected
            .iter()
            .zip(&raw)
            .map(|(a, b)| (a.to_f32() - b).abs())
            .collect();
        let maximum = delta.iter().copied().fold(0.0f32, f32::max);
        let rmse = (delta.iter().map(|v| v * v).sum::<f32>() / delta.len() as f32).sqrt();
        println!("GEMMA4_LINEAR_NATIVE_PROBE output_max={maximum} output_rmse={rmse}");
        for (role, offset) in [
            ("self_attn.k_proj.weight", rows * q_width),
            ("self_attn.v_proj.weight", rows * (q_width + kv_width)),
        ] {
            let projected = engine.gemma_linear_probe(role, &native_input)?;
            let delta: Vec<f32> = projected
                .iter()
                .zip(&raw[offset..offset + rows * kv_width])
                .map(|(a, b)| (a.to_f32() - b).abs())
                .collect();
            let maximum = delta.iter().copied().fold(0.0f32, f32::max);
            let rmse = (delta.iter().map(|v| v * v).sum::<f32>() / delta.len() as f32).sqrt();
            println!("GEMMA4_LINEAR_NATIVE_PROBE {role} output_max={maximum} output_rmse={rmse}");
        }
        let (native_k, native_v) = engine.gemma_kv_rows(0, first_ids.len())?;
        let reference_v = reference_bf16(&data, "stage.a.v_norm")?;
        trace.add(
            "debug.native_k".into(),
            v_shape.clone(),
            native_k.iter().map(|v| v.to_f32()).collect(),
        )?;
        trace.add(
            "debug.native_v".into(),
            v_shape.clone(),
            native_v.iter().map(|v| v.to_f32()).collect(),
        )?;
        let (native_k4, native_v4) = engine.gemma_kv_rows(global_layer, first_ids.len())?;
        trace.add(
            "debug.native_k4".into(),
            vec![first_ids.len(), spec.kv_heads as usize, 512],
            native_k4.iter().map(|v| v.to_f32()).collect(),
        )?;
        trace.add(
            "debug.native_v4".into(),
            vec![first_ids.len(), spec.kv_heads as usize, 512],
            native_v4.iter().map(|v| v.to_f32()).collect(),
        )?;
        trace.set_device(&engine.device_info().name)?;
        trace.save(&runtime.join("native-trace.safetensors"))?;
        let mut raw_v = Vec::with_capacity(reference_v.len());
        for row in 0..rows {
            for head in 0..hkv {
                let base = rows * (q_width + kv_width) + row * kv_width + head * d;
                raw_v.extend(raw[base..base + d].iter().map(|v| bf16::from_f32(*v)));
            }
        }
        let unit_weight = vec![bf16::ONE; d];
        let expected_v: Vec<bf16> = raw_v
            .chunks(d)
            .flat_map(|row| {
                rms_f32(
                    &row.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &unit_weight,
                )
            })
            .collect();
        let raw_stats = Stats::new(&raw_v, &expected_v)?;
        println!("GEMMA4_KV_SCATTER_CONTROL raw_v {}", raw_stats.line("v"));
        ensure!(
            raw_stats.is_detected_mutant(),
            "GEMMA4_KV_SCATTER raw-V control is not discriminating"
        );
        let stats = Stats::new(&native_v, &expected_v)?;
        println!("GEMMA4_KV_SCATTER {}", stats.line("v"));
        ensure!(stats.passes(), "GEMMA4_KV_SCATTER v DEFECT");
        for (name, actual, expected) in [
            ("v", &native_v, reference_v),
            ("k", &native_k, reference_bf16(&data, "stage.a.k_rope")?),
        ] {
            let stats = Stats::new(actual, expected)?;
            println!("GEMMA4_KV_REFERENCE {}", stats.line(name));
        }
    }
    let mut defects = 0usize;
    let mut graph_defects = 0usize;
    let mut graph_state_defects = 0usize;
    let mut first = false;
    for (tag, prefill) in [("a", 0usize), ("b", 0), ("wrap", 512), ("long", 0)] {
        let sequence: Vec<u32> = if matches!(tag, "wrap" | "long") {
            serde_json::from_value(ids[tag].clone())?
        } else {
            serde_json::from_value(ids["short"][if tag == "a" { 0 } else { 1 }].clone())?
        };
        let bounds = &frozen["families"][tag];
        for (mode, cached) in [("prefill", false), ("decode", true)] {
            if !first {
                engine.reset()?;
            }
            first = false;
            let actual = if cached {
                native_path(&mut engine, &sequence, prefill)?
            } else {
                native_path(&mut engine, &sequence, sequence.len())?
            };
            let tensor_name = format!("{tag}.{mode}");
            let expected_rows = data[tensor_name.as_str()].0[0];
            let expected = reference(&data, &tensor_name)?.as_f32();
            let target_ids = &sequence[sequence.len() - expected_rows..];
            let actual = actual[actual.len() - expected.len()..].to_vec();
            let (pass, line, _) = model_metrics(&actual, &expected, target_ids, bounds)?;
            println!("GEMMA4_MODEL {tag} {mode} {line}");
            defects += usize::from(!pass);
        }
        if tag == "wrap" {
            engine.reset()?;
            engine.gemma_prefill_all(&sequence[..512])?;
            let eager_prefill_kv = engine.gemma_kv_rows(0, 512)?;
            let mut eager = Vec::new();
            let mut eager_decode0_kv = None;
            let mut eager_decode0_hidden = None;
            for (row, &token) in sequence[512..].iter().enumerate() {
                if row == 0 {
                    println!("GEMMA4_POSITION eager {:?}", engine.gemma_position()?);
                }
                engine.set_token(token)?;
                engine.decode(false, None)?;
                eager.extend(engine.decode_logits()?.into_iter().map(bf16::from_f32));
                if row == 0 {
                    eager_decode0_kv = Some(engine.gemma_kv_rows(0, 512)?);
                    eager_decode0_hidden = Some(engine.gemma_hidden_state()?);
                }
            }
            engine.reset()?;
            engine.gemma_prefill_all(&sequence[..512])?;
            let prefill_equal = engine.gemma_kv_rows(0, 512)? == eager_prefill_kv;
            graph_state_defects += usize::from(!prefill_equal);
            println!("GEMMA4_GRAPH_KV prefill equal={prefill_equal}");
            engine.capture()?;
            let restore_equal = engine.gemma_kv_rows(0, 512)? == eager_prefill_kv;
            graph_state_defects += usize::from(!restore_equal);
            println!("GEMMA4_GRAPH_KV capture_restore equal={restore_equal}");
            let mut graph_logits = Vec::new();
            for (row, &token) in sequence[512..].iter().enumerate() {
                if row == 0 {
                    println!("GEMMA4_POSITION graph {:?}", engine.gemma_position()?);
                }
                engine.set_token(token)?;
                engine.decode(true, None)?;
                if row == 0 {
                    let actual_kv = engine.gemma_kv_rows(0, 512)?;
                    let decode_equal =
                        &actual_kv == eager_decode0_kv.as_ref().context("eager KV row missing")?;
                    graph_state_defects += usize::from(!decode_equal);
                    println!("GEMMA4_GRAPH_KV decode0 equal={decode_equal}");
                    let (actual_x, actual_n) = engine.gemma_hidden_state()?;
                    let (expected_x, expected_n) = eager_decode0_hidden
                        .as_ref()
                        .context("eager hidden row missing")?;
                    let x_equal = &actual_x == expected_x;
                    let n_equal = &actual_n == expected_n;
                    graph_state_defects += usize::from(!x_equal || !n_equal);
                    println!("GEMMA4_GRAPH_HIDDEN decode0 x={x_equal} n={n_equal}");
                }
                let actual = engine.decode_logits()?;
                let eager = &eager[row * actual.len()..(row + 1) * actual.len()];
                let graph_equal = actual
                    .iter()
                    .map(|v| bf16::from_f32(*v))
                    .collect::<Vec<_>>()
                    == eager;
                graph_defects += usize::from(!graph_equal);
                println!(
                    "GEMMA4_GRAPH_EAGER wrap row={row} {}",
                    if graph_equal { "PASS" } else { "DEFECT" }
                );
                graph_logits.extend(actual.iter().map(|v| bf16::from_f32(*v)));
            }
            let (pass, line, _) = model_metrics(
                &graph_logits,
                &reference(&data, "wrap.decode")?.as_f32(),
                &sequence[512..],
                bounds,
            )?;
            println!("GEMMA4_MODEL wrap graph {line}");
            defects += usize::from(!pass);
        }
    }
    if let Some(cell) = std::env::var_os("FF_ORACLE_CELL") {
        let cell = cell.to_string_lossy().into_owned();
        let family = cell.split_once('.').map(|(family, _)| family).unwrap_or("");
        let reference_dir = std::env::var_os("FF_GEMMA4_REFERENCE_FIXTURES")
            .context("FF_GEMMA4_REFERENCE_FIXTURES missing for oracle localization")?;
        let reference =
            load_reference(&std::path::Path::new(&reference_dir).join("reference.safetensors"))?;
        engine.reset()?;
        localize_oracle_cell(
            &mut engine,
            &data,
            &reference,
            &ids,
            &cell,
            &frozen["families"][family],
        )?;
    }
    let wrap: Vec<u32> = serde_json::from_value(ids["wrap"].clone())?;
    drop(engine);
    let mut chunked = super::engine::Engine::load(&dir, 0, 640, Some(8), settings)?;
    let mut actual = Vec::new();
    for chunk in wrap.chunks(8) {
        chunked.gemma_prefill_all(chunk)?;
        for row in 0..chunk.len() {
            actual.extend(chunked.gemma_all_logits(row)?);
        }
    }
    let expected = reference(&data, "wrap.prefill")?.as_f32();
    let actual = actual[actual.len() - expected.len()..].to_vec();
    let (pass, line, _) = model_metrics(
        &actual,
        &expected,
        &wrap[wrap.len() - 8..],
        &frozen["families"]["wrap"],
    )?;
    println!("GEMMA4_CHUNKED_WRAP chunk=8 {line}");
    defects += usize::from(!pass);
    ensure!(defects == 0, "GEMMA4_MODEL defects={defects}");
    ensure!(
        graph_defects == 0 && graph_state_defects == 0,
        "GEMMA4_GRAPH_EAGER defects={graph_defects} state_defects={graph_state_defects}"
    );
    println!("GEMMA4_MODEL PASS prefill decode wrap graph");
    Ok(())
}

fn gemma4_upload_matches(
    loaded: &std::collections::BTreeSet<String>,
    text: &std::collections::BTreeSet<String>,
) -> Result<()> {
    ensure!(
        loaded == text,
        "uploaded set differs from the Text scope: extra={:?} missing={:?}",
        loaded.difference(text).take(3).collect::<Vec<_>>(),
        text.difference(loaded).take(3).collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn ensure_no_other_compute_app() -> Result<()> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()?;
    ensure!(output.status.success(), "RUNNER_FAIL nvidia-smi");
    let own = std::process::id();
    let others: Vec<u32> = std::str::from_utf8(&output.stdout)?
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .filter(|&pid| pid != own)
        .collect();
    ensure!(
        others.is_empty(),
        "RUNNER_FAIL co-tenant compute app: {others:?}"
    );
    Ok(())
}

#[test]
#[ignore = "requires an otherwise idle GPU and a Gemma 4 E2B/E4B directory in FF_MODEL"]
fn gemma4_catalog_uploads_exactly_the_text_set_gpu() -> Result<()> {
    use super::weights::{StoredTensor, WeightCatalog};
    use crate::checkpoint::TensorScope;
    use std::collections::BTreeSet;
    let dir = std::path::PathBuf::from(std::env::var_os("FF_MODEL").context("FF_MODEL missing")?);
    let spec = crate::model_spec::ModelSpec::read(&dir)?;
    let (tensors, resident) = match spec.hidden {
        1536 => (540usize, 9_257_138_758usize),
        2560 => (665, 14_926_026_836),
        hidden => anyhow::bail!("unexpected Gemma 4 hidden size {hidden}"),
    };
    #[cfg(feature = "cuda")]
    ensure_no_other_compute_app()?;
    let device = crate::backend::cuda::Device::new(0)?;
    let (free_before, _) = device.ctx.mem_get_info()?;
    let catalog = WeightCatalog::load(&dir, &spec, &device)?;
    device.stream.synchronize()?;
    let (free_after, _) = device.ctx.mem_get_info()?;
    let text: BTreeSet<String> = catalog
        .manifest
        .entries
        .iter()
        .filter(|(_, entry)| entry.scope == TensorScope::Text)
        .map(|(name, _)| name.clone())
        .collect();
    let loaded: BTreeSet<String> = catalog.tensors.keys().cloned().collect();
    gemma4_upload_matches(&loaded, &text)?;
    let dormant = catalog
        .manifest
        .entries
        .iter()
        .find(|(_, entry)| entry.scope == TensorScope::Dormant)
        .map(|(name, _)| name.clone())
        .context("manifest has no Dormant tensor")?;
    let mut mutant = loaded.clone();
    mutant.insert(dormant);
    ensure!(
        gemma4_upload_matches(&mutant, &text).is_err(),
        "a Dormant upload was not detected"
    );
    let embed = catalog.bf16("model.language_model.embed_tokens.weight")?;
    let per_layer = catalog.bf16("model.language_model.embed_tokens_per_layer.weight")?;
    let layers = spec.layers.len();
    ensure!(
        !catalog.tensors.contains_key("lm_head.weight")
            && embed.len() == spec.vocab as usize * spec.hidden as usize
            && per_layer.len() == spec.vocab as usize * layers * 256
            && loaded.len() == tensors
            && catalog.bytes == resident,
        "Gemma 4 upload: tensors={} bytes={} embed={} per_layer={}",
        loaded.len(),
        catalog.bytes,
        embed.len(),
        per_layer.len()
    );
    let (bf16_bytes, native_bytes) =
        catalog
            .tensors
            .values()
            .fold(
                (0usize, 0usize),
                |(bf16_bytes, native_bytes), tensor| match tensor {
                    StoredTensor::Bf16(values) => (bf16_bytes + values.len() * 2, native_bytes),
                    StoredTensor::Bytes(values) => (bf16_bytes, native_bytes + values.len()),
                    #[cfg(feature = "cuda")]
                    StoredTensor::Slot { .. } => {
                        panic!("resident upload gate contains streamed weights")
                    }
                },
            );
    ensure!(
        bf16_bytes + native_bytes == resident && catalog.bytes == resident,
        "allocator ledger mismatch: bf16={bf16_bytes} native={native_bytes} expected={resident}"
    );
    println!(
        "GEMMA4_ALLOCATOR PASS tensors={} bf16_bytes={} native_bytes={} total={}",
        loaded.len(),
        bf16_bytes,
        native_bytes,
        bf16_bytes + native_bytes
    );
    #[cfg(feature = "cuda")]
    ensure_no_other_compute_app()?;
    let used = free_before
        .checked_sub(free_after)
        .context("device free memory grew during the load")?;
    ensure!(
        used >= resident && used <= resident + (512 << 20),
        "device memory delta {used} outside [{resident}, {resident} + 512 MiB]"
    );
    println!(
        "GEMMA4_LOADER PASS tensors={} bytes={} device_delta={used}",
        loaded.len(),
        catalog.bytes
    );
    Ok(())
}
