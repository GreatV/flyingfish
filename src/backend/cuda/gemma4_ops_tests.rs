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
    beyond_mutant: usize,
}

impl Stats {
    fn new(candidate: &[bf16], reference: &[bf16]) -> Result<Self> {
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
            beyond_mutant: 0,
        };
        for (c, r) in candidate.iter().zip(reference) {
            let ulp = ordered(*c).abs_diff(ordered(*r));
            stats.max_ulp = stats.max_ulp.max(ulp);
            stats.bit_equal += usize::from(ulp == 0);
            stats.beyond_mutant += usize::from(ulp > MUTANT_ULP);
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
            "{name} elements={} max_ulp={} bit_equal={:.6} beyond_{}ulp={:.6}",
            self.elements,
            self.max_ulp,
            self.bit_equal as f64 / self.elements as f64,
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
            let (fused_out, split_out) = run(2)?;
            for (kernel, out) in [("silu_mul", &fused_out), ("silu_mul_split", &split_out)] {
                let stats = Stats::new(out, reference)?;
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
                let stats = Stats::new(out, reference)?;
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
