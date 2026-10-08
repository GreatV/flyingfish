use super::{Markov, check_requests};
use crate::backend::cuda::{Device, ops::Ops};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, PushKernelArg};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{hint::black_box, path::Path, time::Instant};

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
    let mut markov = Markov::new(&d)?;
    let mut compared = 0;
    for (seed, file) in ["r0008", "r0042", "r0089", "r0140"].iter().enumerate() {
        let base = tensor(
            &d,
            &Path::new(&fixtures).join(format!("{file}.safetensors")),
            "B",
            &[7, 130560],
        )?;
        assert!(markov.distributions_batch(&base, &w1, &w2, &[])?.is_empty());
        for p in [1, 2, 7, 8, 9, 16, 32, 63, 64, 65, 128, 129] {
            let mut req = requests(p, seed);
            req.reverse();
            if p > 1 {
                req[p - 1] = req[0];
            }
            let batch = markov.distributions_batch(&base, &w1, &w2, &req)?;
            ensure!(batch.len() == req.len(), "batch lost requests");
            for (i, request) in req.iter().enumerate() {
                let single = markov.distributions_batch(&base, &w1, &w2, &[*request])?;
                ensure!(
                    batch[i].bits_equal(&single[0]),
                    "batch mismatch fixture={file} P={p} index={i} request={request:?} batch={:?} single={:?}",
                    batch[i],
                    single[0]
                );
                compared += 1;
            }
            println!(
                "{}",
                serde_json::json!({"markov_batch_equivalence":{"fixture":file,"P":p,"pass":true,"bits":"tokens/u32, logp/f32, lse/f32"}})
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
        serde_json::json!({"markov_batch_equivalence_summary":{"requests_compared":compared,"pass":true}})
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
    let mut markov = Markov::new(&d)?;
    let ops = Ops::new(&d.ctx)?;
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
