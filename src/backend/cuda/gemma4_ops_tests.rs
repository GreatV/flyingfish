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

#[test]
#[ignore = "requires a GPU and FF_GEMMA4_FIXTURES"]
fn gelu_tanh_gated_activation_matches_the_definition_gpu() -> Result<()> {
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
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
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
    token: usize,
    layer: usize,
) -> Result<bool> {
    let name = format!("layers.{family}");
    let oracle_values = reference(oracle, &name)?.as_f32();
    let reference_values = reference(reference_layers, &name)?.as_f32();
    let candidate_values = trace.values(&format!(".{layer}"))?;
    ensure!(
        candidate_values.len() == oracle_values.len()
            && candidate_values.len() == reference_values.len(),
        "oracle layer extent mismatch at layer {layer}"
    );
    let (mut candidate_max, mut reference_max) = (0.0f64, 0.0f64);
    let (mut candidate_sq, mut reference_sq) = (0.0f64, 0.0f64);
    for ((candidate, oracle), reference) in candidate_values
        .iter()
        .zip(&oracle_values)
        .zip(&reference_values)
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
    println!(
        "GEMMA4_ORACLE_LAYER token={token} layer={layer} candidate_max={candidate_max} candidate_rmse={candidate_rmse} reference_max={reference_max} reference_rmse={reference_rmse} beyond_2x={beyond}"
    );
    Ok(beyond)
}

fn oracle_token_nll(
    actual: &[bf16],
    expected: &[f32],
    ids: &[u32],
    bounds: &serde_json::Value,
    token: usize,
) -> Result<()> {
    let (_, line) = model_metrics(actual, expected, ids, bounds)?;
    println!("GEMMA4_ORACLE_TOKEN token={token} {line}");
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
    match mode {
        "prefill" => {
            let mut trace = crate::trace::Trace::all_layers();
            engine.gemma_prefill_trace(&sequence, &mut trace)?;
            let mut beyond = false;
            for layer in 0..layers {
                beyond |= oracle_layer_distance(&trace, oracle, reference, family, 0, layer)?;
            }
            if !beyond {
                let expected = reference(oracle, &format!("{family}.prefill"))?.as_f32();
                for (row, &id) in sequence.iter().enumerate() {
                    let actual = engine.gemma_all_logits(row)?;
                    oracle_token_nll(
                        &actual,
                        &expected[row * actual.len()..(row + 1) * actual.len()],
                        &[id],
                        bounds,
                        row,
                    )?;
                }
            }
        }
        "decode" => {
            for (row, &id) in sequence.iter().enumerate() {
                let mut trace = crate::trace::Trace::all_layers();
                engine.gemma_decode_trace(id, &mut trace)?;
                let mut beyond = false;
                for layer in 0..layers {
                    beyond |= oracle_layer_distance(&trace, oracle, reference, family, row, layer)?;
                }
                if !beyond {
                    let expected = reference(oracle, &format!("{family}.decode"))?.as_f32();
                    let actual: Vec<bf16> = engine
                        .decode_logits()?
                        .into_iter()
                        .map(bf16::from_f32)
                        .collect();
                    oracle_token_nll(
                        &actual,
                        &expected[row * actual.len()..(row + 1) * actual.len()],
                        &[id],
                        bounds,
                        row,
                    )?;
                }
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
) -> Result<(bool, String)> {
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
            let candidate_peak = a
                .iter()
                .map(|v| v.to_f32())
                .fold(f32::NEG_INFINITY, f32::max);
            let candidate_sum = a
                .iter()
                .map(|v| (v.to_f32() - candidate_peak).exp() as f64)
                .sum::<f64>();
            losses[0] += -(a[target].to_f32() - candidate_peak) as f64 + candidate_sum.ln();
            let reference_peak = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let reference_sum = b
                .iter()
                .map(|v| (v - reference_peak).exp() as f64)
                .sum::<f64>();
            losses[1] += -(b[target] - reference_peak) as f64 + reference_sum.ln();
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
    let (pass, line) = model_metrics(&actual, &oracle, &ids, &bounds)?;
    ensure!(
        !pass && line.starts_with("output_max=0.0009765625"),
        "F32 oracle distance was not scored: {line}"
    );
    let rounded: Vec<f32> = oracle
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let (pass, line) = model_metrics(&actual, &rounded, &ids, &bounds)?;
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

fn checkpoint_tensor(dir: &std::path::Path, name: &str) -> Result<Vec<bf16>> {
    let bytes = std::fs::read(dir.join("model.safetensors"))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let view = tensors.tensor(name)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16,
        "{name} is not BF16"
    );
    Ok(view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect())
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
    let mut engine = super::engine::Engine::load(&dir, 0, 640, Some(640), settings)?;
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
    for (tag, prefill) in [("a", 0usize), ("b", 0), ("wrap", 512)] {
        let sequence: Vec<u32> = if tag == "wrap" {
            serde_json::from_value(ids["wrap"].clone())?
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
            let expected = reference(&data, &format!("{tag}.{mode}"))?.as_f32();
            let expected_rows = if cached {
                sequence.len() - prefill
            } else if tag == "wrap" {
                8
            } else {
                sequence.len()
            };
            let target_ids = &sequence[sequence.len() - expected_rows..];
            let actual = actual[actual.len() - expected.len()..].to_vec();
            let (pass, line) = model_metrics(&actual, &expected, target_ids, bounds)?;
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
            let (pass, line) = model_metrics(
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
