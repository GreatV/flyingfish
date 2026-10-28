//! Gemma 4 per-operation GPU gates against fixtures written by the model definition.
//! Fixture directory: FF_GEMMA4_FIXTURES.
use super::{Device, cubin, ops::flat};
use crate::backend::gpu::PushKernelArg;
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

fn fixture(file: &str) -> Result<Fixture> {
    let dir = std::env::var_os("FF_GEMMA4_FIXTURES").context("FF_GEMMA4_FIXTURES missing")?;
    let bytes = std::fs::read(std::path::Path::new(&dir).join(file))?;
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

#[test]
#[ignore = "requires CUDA and FF_GEMMA4_FIXTURES"]
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
#[ignore = "requires CUDA, FF_GEMMA4_FIXTURES and a Gemma 4 E2B/E4B directory in FF_MODEL"]
fn softcap_tied_head_and_embedding_scale_match_the_definition_gpu() -> Result<()> {
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
                .launch(flat(ids.len() * hidden))?;
        }
        Ok(s.clone_dtoh(&embedded)?)
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
    let blas = CudaBlas::new(s.clone())?;
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
        Ok(s.clone_dtoh(&y)?)
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
        Ok(s.clone_dtoh(&x)?)
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
#[ignore = "requires CUDA and FF_GEMMA4_FIXTURES"]
fn unrounded_norm_and_branch_boundary_match_the_definition_gpu() -> Result<()> {
    use super::contracts::{RowDomain, SeqRef, Tensor, TensorMut};
    use super::ops::{CommonOpsRef, RmsMode, RmsOptions};
    use crate::backend::gpu::{CudaSlice, CudaStream};
    use cudarc::driver::DevicePtr;
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
                Ok(s.clone_dtoh(&y)?)
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
#[ignore = "requires CUDA and FF_GEMMA4_FIXTURES"]
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
        let table_device = s.clone_htod(&table)?;
        let run = |table: &cudarc::driver::CudaSlice<bf16>,
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
            Ok(s.clone_dtoh(&x)?)
        };
        let mutants: Vec<(&str, cudarc::driver::CudaSlice<bf16>)> = if tag == "full" {
            [
                ("rotate_all_512", RopeSpec { pairs: 256, ..spec }),
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
            .map(|(name, mutant)| Ok((name, s.clone_htod(&mutant.table(d as u32, CAPACITY)?)?)))
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

fn model_metrics(
    actual: &[bf16],
    expected: &[bf16],
    ids: &[u32],
    bounds: &serde_json::Value,
) -> Result<(bool, String)> {
    ensure!(
        actual.len() == expected.len(),
        "model logit extent mismatch"
    );
    ensure!(!ids.is_empty(), "model metric requires rows");
    ensure!(
        actual
            .iter()
            .chain(expected)
            .all(|v| v.to_f32().is_finite()),
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
            let delta = f64::from(a.to_f32() - b.to_f32());
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
        }
        let mut best = [f32::NEG_INFINITY; 2];
        let mut candidate = 0usize;
        let mut candidate_best = f32::NEG_INFINITY;
        for (i, (av, bv)) in a.iter().zip(b).enumerate() {
            let score = bv.to_f32();
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
            for (side, values) in [0, 1].map(|side| (side, if side == 0 { a } else { b })) {
                let peak = values
                    .iter()
                    .map(|v| v.to_f32())
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum = values
                    .iter()
                    .map(|v| (v.to_f32() - peak).exp() as f64)
                    .sum::<f64>();
                losses[side] += -(values[target].to_f32() - peak) as f64 + sum.ln();
            }
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
        engine.gemma_decode_all(*token)?;
        out.extend(engine.gemma_all_logits(0)?);
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
    let data = fixture("reference.safetensors")?;
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
    for name in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ] {
        let full = format!("model.language_model.layers.0.{name}");
        let actual = catalog.bf16(&full)?;
        let actual = device.stream.clone_dtoh(actual)?;
        let expected = checkpoint_tensor(&dir, &full)?;
        ensure!(actual == expected, "published tensor mismatch: {full}");
    }
    drop(catalog);
    let mut engine = super::engine::Engine::load(&dir, 0, 640, Some(640), settings)?;
    let probe_input = tensor(&data, "stage.a.input_layernorm")?;
    let projected = engine.gemma_linear_probe("self_attn.q_proj.weight", probe_input)?;
    let expected = tensor(&data, "stage.a.q_proj")?;
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
    trace.set_device(&engine.device_info().name)?;
    trace.save(&runtime.join("native-trace.safetensors"))?;
    let mut defects = 0usize;
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
            let expected = tensor(&data, &format!("{tag}.{mode}"))?;
            let expected_rows = if cached {
                sequence.len() - prefill
            } else if tag == "wrap" {
                8
            } else {
                sequence.len()
            };
            let target_ids = &sequence[sequence.len() - expected_rows..];
            let actual = actual[actual.len() - expected.len()..].to_vec();
            let (pass, line) = model_metrics(&actual, expected, target_ids, bounds)?;
            println!("GEMMA4_MODEL {tag} {mode} {line}");
            defects += usize::from(!pass);
        }
        if tag == "wrap" {
            engine.reset()?;
            let eager = native_path(&mut engine, &sequence, 512)?;
            engine.reset()?;
            engine.gemma_prefill_all(&sequence[..512])?;
            engine.capture()?;
            for (row, _) in sequence[512..].iter().enumerate() {
                engine.decode(true, None)?;
                let actual = engine.decode_logits()?;
                let expected =
                    &tensor(&data, "wrap.decode")?[row * actual.len()..(row + 1) * actual.len()];
                let eager = &eager[row * actual.len()..(row + 1) * actual.len()];
                ensure!(
                    actual
                        .iter()
                        .map(|v| bf16::from_f32(*v))
                        .collect::<Vec<_>>()
                        == eager,
                    "graph/eager logits differ at wrap row {row}"
                );
                let actual: Vec<bf16> = actual.iter().map(|v| bf16::from_f32(*v)).collect();
                let (pass, line) =
                    model_metrics(&actual, expected, &sequence[512..=512 + row], bounds)?;
                println!("GEMMA4_MODEL wrap graph row={row} {line}");
                defects += usize::from(!pass);
            }
        }
    }
    ensure!(defects == 0, "GEMMA4_MODEL defects={defects}");
    println!("GEMMA4_MODEL PASS prefill decode wrap graph");
    Ok(())
}
