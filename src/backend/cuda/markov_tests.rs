use super::{Markov, check_requests};
use crate::backend::cuda::{Device, ops::Ops};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, PushKernelArg};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{hint::black_box, path::Path, time::Instant};

#[path = "markov_reference.rs"]
mod reference;

const BATCH_COUNTS: [usize; 16] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 129];

#[derive(Default, serde::Serialize)]
struct Differences {
    requests: usize,
    tokens: usize,
    logp: usize,
    lse: usize,
    max_logp_abs: f64,
    max_lse_abs: f64,
    actual_mass_max: f64,
}

impl Differences {
    fn track(
        &mut self,
        actual: &crate::backend::Top4,
        expected: &crate::backend::Top4,
        context: &str,
        request: (u8, u32),
    ) {
        self.tokens += usize::from(actual.tokens != expected.tokens);
        self.logp += usize::from(actual.logp.map(f32::to_bits) != expected.logp.map(f32::to_bits));
        self.lse += usize::from(actual.lse.to_bits() != expected.lse.to_bits());
        self.actual_mass_max = self
            .actual_mass_max
            .max(actual.logp.iter().map(|&v| f64::from(v).exp()).sum());
        for (&a, &b) in actual.logp.iter().zip(&expected.logp) {
            self.max_logp_abs = self.max_logp_abs.max((f64::from(a) - f64::from(b)).abs());
        }
        self.max_lse_abs = self
            .max_lse_abs
            .max((f64::from(actual.lse) - f64::from(expected.lse)).abs());
        if !actual.bits_equal(expected) {
            self.requests += 1;
            if self.requests <= 8 {
                println!(
                    "{}",
                    serde_json::json!({"markov_bit_difference":{"context":context,"request":request,
                    "actual":{"tokens":actual.tokens,"logp":actual.logp,"lse":actual.lse,"logp_bits":actual.logp.map(f32::to_bits),"lse_bits":actual.lse.to_bits()},
                    "expected":{"tokens":expected.tokens,"logp":expected.logp,"lse":expected.lse,"logp_bits":expected.logp.map(f32::to_bits),"lse_bits":expected.lse.to_bits()}}})
                );
            }
        }
    }
}

#[test]
fn bit_diagnostics_keep_float_and_token_differences_separate() {
    let expected = crate::backend::Top4 {
        tokens: [0, 1, 2, 3],
        logp: [-4.0, -5.0, -6.0, -7.0],
        lse: 1.0,
    };
    let mut actual = expected.clone();
    actual.logp[0] = f32::from_bits(actual.logp[0].to_bits() + 1);
    let mut stats = Differences::default();
    stats.track(&actual, &expected, "one-ulp test", (0, 0));
    assert_eq!(
        (stats.requests, stats.tokens, stats.logp, stats.lse),
        (1, 0, 1, 0)
    );
    assert_eq!(
        stats.max_logp_abs,
        (f64::from(actual.logp[0]) - f64::from(expected.logp[0])).abs()
    );
    actual = expected.clone();
    actual.tokens[0] = 4;
    actual.lse = f32::from_bits(actual.lse.to_bits() + 1);
    stats.track(&actual, &expected, "token and lse test", (0, 0));
    assert_eq!(
        (stats.requests, stats.tokens, stats.logp, stats.lse),
        (2, 1, 1, 1)
    );
}

fn chain_requests(path: &Path) -> Result<Vec<(u8, u32)>> {
    let trace = crate::trace::Trace::read(path)?;
    let block = trace.values("draft_block_ids")?;
    let proposals = trace.values("proposals")?;
    ensure!(
        block.len() == 7 && proposals.len() == 7,
        "fixture chain extent mismatch"
    );
    Ok((0..7)
        .map(|row| {
            (
                row as u8,
                if row == 0 {
                    block[0] as u32
                } else {
                    proposals[row - 1] as u32
                },
            )
        })
        .collect())
}

fn tensor(d: &Device, path: &Path, name: &str, shape: &[usize]) -> Result<CudaSlice<bf16>> {
    let file = std::fs::File::open(path)?;
    let map = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&map)?;
    let t = tensors.tensor(name)?;
    ensure!(t.shape() == shape, "test tensor {name}: wrong shape");
    let values = tensor_values(t)?;
    let output = d.upload.clone_htod(&values)?;
    d.finish_upload()?;
    Ok(output)
}

fn tensor_values(t: safetensors::tensor::TensorView<'_>) -> Result<Vec<bf16>> {
    Ok(match t.dtype() {
        Dtype::BF16 => t
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect::<Vec<_>>(),
        Dtype::F32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| bf16::from_f32(f32::from_le_bytes(*v)))
            .collect(),
        dtype => anyhow::bail!("test tensor has unsupported dtype {dtype:?}"),
    })
}

#[test]
fn markov_tensor_bytes_and_rounding() -> Result<()> {
    use safetensors::tensor::TensorView;
    let bits = [0u16, 0x8000, 1, 0x3f80, 0x7fc1];
    let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = TensorView::new(Dtype::BF16, vec![bits.len()], &bytes)?;
    assert_eq!(
        tensor_values(tensor)?
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
    let values = [0u32, 0x80000000, 0x3f808000, 0x3f818000, 0xbf808000].map(f32::from_bits);
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = TensorView::new(Dtype::F32, vec![values.len()], &bytes)?;
    assert_eq!(
        tensor_values(tensor)?
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [0, 0x8000, 0x3f80, 0x3f82, 0xbf80]
    );
    Ok(())
}

fn weights(d: &Device) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
    let path = std::env::var("FF_MARKOV_DRAFT").context("FF_MARKOV_DRAFT missing")?;
    println!("markov_draft={path}");
    let path = Path::new(&path).join("model.safetensors");
    let w1 = tensor(d, &path, "markov_head.markov_w1.weight", &[130560, 256])?;
    let w2 = tensor(d, &path, "markov_head.markov_w2.weight", &[130560, 256])?;
    Ok((w1, w2))
}

fn requests(p: usize, seed: usize) -> Vec<(u8, u32)> {
    (0..p)
        .map(|i| {
            (
                ((i + seed) % 7) as u8,
                ((i * 19837 + seed * 97) % 130560) as u32,
            )
        })
        .collect()
}

#[test]
fn request_bounds() -> Result<()> {
    check_requests(&[])?;
    check_requests(&[(0, 0), (6, 130559)])?;
    assert!(check_requests(&[(7, 0)]).is_err());
    assert!(check_requests(&[(0, 130560)]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires a reserved GPU window and FF_MARKOV_DRAFT/FF_MARKOV_FIXTURES"]
fn batch_matches_single() -> Result<()> {
    let d = Device::new(0)?;
    let (w1, w2) = weights(&d)?;
    let fixtures = std::env::var("FF_MARKOV_FIXTURES").context("FF_MARKOV_FIXTURES missing")?;
    let ops = Ops::new(&d.ctx)?;
    let runtime = std::env::temp_dir().join(format!("ff-markov-cal-{}", std::process::id()));
    std::fs::create_dir_all(&runtime)?;
    let calibration = crate::backend::cuda::closure::Closure::new(
        true,
        "explicit Markov test calibration".into(),
    );
    let mut markov = Markov::new(&d, &ops, &runtime, &calibration)?;
    let mut compared = 0;
    let mut differences = Differences::default();
    for (seed, file) in ["r0008", "r0042", "r0089", "r0140"].iter().enumerate() {
        let base = tensor(
            &d,
            &Path::new(&fixtures).join(format!("{file}.safetensors")),
            "B",
            &[7, 130560],
        )?;
        assert!(markov.distributions_batch(&base, &w1, &w2, &[])?.is_empty());
        let req = chain_requests(&Path::new(&fixtures).join(format!("{file}.safetensors")))?;
        let batch = markov.distributions_batch(&base, &w1, &w2, &req)?;
        for (index, request) in req.iter().enumerate() {
            let single = markov.distributions_batch(&base, &w1, &w2, &[*request])?;
            differences.track(
                &batch[index],
                &single[0],
                &format!("{file} chain row{index}"),
                *request,
            );
            compared += 1;
        }
        for p in BATCH_COUNTS {
            let before = differences.requests;
            let mut req = requests(p, seed);
            req.reverse();
            if p > 1 {
                req[p - 1] = req[0];
            }
            let batch = markov.distributions_batch(&base, &w1, &w2, &req)?;
            ensure!(batch.len() == req.len(), "batch lost requests");
            for (i, request) in req.iter().enumerate() {
                let single = markov.distributions_batch(&base, &w1, &w2, &[*request])?;
                differences.track(
                    &batch[i],
                    &single[0],
                    &format!("{file} P{p} index{i}"),
                    *request,
                );
                compared += 1;
            }
            println!(
                "{}",
                serde_json::json!({"markov_batch_equivalence":{"fixture":file,"P":p,"pass":differences.requests==before,"bits":"tokens/u32, logp/f32, lse/f32"}})
            );
        }
        let mut invalid = requests(65, seed);
        invalid[64] = (7, 0);
        assert!(
            markov
                .distributions_batch(&base, &w1, &w2, &invalid)
                .is_err()
        );
    }
    println!(
        "{}",
        serde_json::json!({"markov_batch_equivalence_summary":{"requests_compared":compared,"pass":differences.requests==0,"differences":differences}})
    );
    ensure!(
        differences.requests == 0,
        "Markov batch/single bit differences: {} requests",
        differences.requests
    );
    Ok(())
}

#[test]
#[ignore = "requires a reserved GPU window and FF_MARKOV_DRAFT/FF_MARKOV_FIXTURES/FF_MARKOV_V1_DIR"]
fn top4_matches_reference() -> Result<()> {
    let d = Device::new(0)?;
    let (w1, w2) = weights(&d)?;
    let fixtures = std::env::var("FF_MARKOV_FIXTURES").context("FF_MARKOV_FIXTURES missing")?;
    let ops = Ops::new(&d.ctx)?;
    let runtime = std::env::temp_dir().join(format!("ff-markov-cal-{}", std::process::id()));
    std::fs::create_dir_all(&runtime)?;
    let calibration = crate::backend::cuda::closure::Closure::new(
        true,
        "explicit Markov test calibration".into(),
    );
    let mut markov = Markov::new(&d, &ops, &runtime, &calibration)?;
    let mut reference = reference::Reference::new(&d)?;
    let mut compared = 0;
    let mut differences = Differences::default();
    for (seed, file) in ["r0008", "r0042", "r0089", "r0140"].iter().enumerate() {
        let base = tensor(
            &d,
            &Path::new(&fixtures).join(format!("{file}.safetensors")),
            "B",
            &[7, 130560],
        )?;
        let req = chain_requests(&Path::new(&fixtures).join(format!("{file}.safetensors")))?;
        let got = markov.distributions_batch(&base, &w1, &w2, &req)?;
        let expected = reference.batch(&base, &w1, &w2, &req)?;
        for (index, (actual, old)) in got.iter().zip(&expected).enumerate() {
            differences.track(actual, old, &format!("{file} chain row{index}"), req[index]);
            compared += 1;
        }
        for p in BATCH_COUNTS {
            let before = differences.requests;
            let before_tokens = differences.tokens;
            let mut req = requests(p, seed);
            req.reverse();
            if p > 1 {
                req[p - 1] = req[0];
            }
            let got = markov.distributions_batch(&base, &w1, &w2, &req)?;
            let expected = reference.batch(&base, &w1, &w2, &req)?;
            ensure!(
                got.len() == req.len() && expected.len() == req.len(),
                "reference batch lost requests"
            );
            for (index, (actual, old)) in got.iter().zip(&expected).enumerate() {
                differences.track(
                    actual,
                    old,
                    &format!("{file} P{p} index{index}"),
                    req[index],
                );
                compared += 1;
            }
            println!(
                "{}",
                serde_json::json!({"markov_reference_equivalence":{"fixture":file,"P":p,"token_bit_gate":differences.tokens==before_tokens,"float_bit_differences_diagnostic":differences.requests-before}})
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"markov_reference_summary":{"requests_compared":compared,"pass":differences.tokens==0 && differences.actual_mass_max<=1.000001,"differences":differences,"rule":"top4 token bits equal; FP32 bit equality is diagnostic; FP64 accuracy checked separately"}})
    );
    ensure!(
        differences.tokens == 0 && differences.actual_mass_max <= 1.000001,
        "Markov v1/version token differences={} probability mass={}",
        differences.tokens,
        differences.actual_mass_max
    );
    Ok(())
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let index = (values.len() - 1) as f64 * q;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    values[low] + (values[high] - values[low]) * (index - low as f64)
}

#[test]
#[ignore = "requires a reserved timing GPU window and FF_MARKOV_DRAFT/FF_MARKOV_FIXTURES"]
fn batch_wall() -> Result<()> {
    let d = Device::new(0)?;
    let (w1, w2) = weights(&d)?;
    let fixtures = std::env::var("FF_MARKOV_FIXTURES").context("FF_MARKOV_FIXTURES missing")?;
    let base = tensor(
        &d,
        &Path::new(&fixtures).join("r0008.safetensors"),
        "B",
        &[7, 130560],
    )?;
    let ops = Ops::new(&d.ctx)?;
    let runtime = std::env::temp_dir().join(format!("ff-markov-cal-{}", std::process::id()));
    std::fs::create_dir_all(&runtime)?;
    let calibration = crate::backend::cuda::closure::Closure::new(
        true,
        "explicit Markov test calibration".into(),
    );
    let mut markov = Markov::new(&d, &ops, &runtime, &calibration)?;
    let count = d.info.l2_bytes.max(1024 * 1024);
    let mut flush = d.stream.alloc_zeros::<bf16>(count)?;
    let zeros = d.stream.alloc_zeros::<bf16>(count)?;
    let tile = (0..65536)
        .map(|i| bf16::from_f32(((i * 7919 % 65521) as f32 - 32760.0) / 32760.0))
        .collect::<Vec<_>>();
    for offset in (0..count).step_by(tile.len()) {
        let n = tile.len().min(count - offset);
        d.stream
            .memcpy_htod(&tile[..n], &mut flush.slice_mut(offset..offset + n))?;
    }
    d.stream.synchronize()?;
    for p in [1, 8, 16, 32, 64] {
        let req = requests(p, 0);
        for _ in 0..10 {
            black_box(markov.distributions_batch(&base, &w1, &w2, &req)?);
        }
        for cold in [false, true] {
            let mut times = Vec::new();
            for _ in 0..100 {
                if cold {
                    unsafe {
                        d.stream
                            .launch_builder(&ops.residual)
                            .arg(&mut flush)
                            .arg(&zeros)
                            .arg(&(count as i32))
                            .launch(super::super::ops::grid(count.div_ceil(2048), 256))?;
                    }
                    d.stream.synchronize()?;
                }
                let begin = Instant::now();
                black_box(markov.distributions_batch(&base, &w1, &w2, &req)?);
                times.push(begin.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"markov_batch_wall":{"P":p,"cold_L2":cold,"samples":times.len(),
                "median_ms":quantile(&times,0.5),"p10_ms":quantile(&times,0.1),"p90_ms":quantile(&times,0.9),
                "method":"host wall: upload+phase1+phase2+one D2H+one sync+Top4 validation; cold flush and its sync excluded"}})
            );
        }
    }
    Ok(())
}
