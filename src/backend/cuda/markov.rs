use super::{Device, cubin};
use crate::backend::Top4;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub struct Markov {
    stream: Arc<CudaStream>,
    first: Vec<CudaFunction>,
    second: CudaFunction,
    tokens: CudaSlice<u32>,
    rows: CudaSlice<i32>,
    partial_top: CudaSlice<f32>,
    partial_lse: CudaSlice<f32>,
    output: CudaSlice<u32>,
}

impl Markov {
    pub fn new(d: &Device) -> Result<Self> {
        let module = cubin::module(&d.ctx, "markov")?;
        let s = &d.stream;
        ensure!(
            d.info.shared_bytes >= 32768,
            "Markov v2 kernels require 32768 bytes static shared memory"
        );
        Ok(Self {
            stream: s.clone(),
            first: [1, 8, 64]
                .iter()
                .map(|p| module.load_function(&format!("markov2_top4_phase1_p{p}")))
                .collect::<std::result::Result<_, _>>()?,
            second: module.load_function("markov2_top4_phase2")?,
            tokens: s.alloc_zeros(64)?,
            rows: s.alloc_zeros(64)?,
            partial_top: s.alloc_zeros(510 * 64 * 8)?,
            partial_lse: s.alloc_zeros(510 * 64 * 2)?,
            output: s.alloc_zeros(64 * 9)?,
        })
    }

    pub fn distributions_batch(
        &mut self,
        base: &CudaSlice<bf16>,
        w1: &CudaSlice<bf16>,
        w2: &CudaSlice<bf16>,
        requests: &[(u8, u32)],
    ) -> Result<Vec<Top4>> {
        check_requests(requests)?;
        let mut result = Vec::with_capacity(requests.len());
        for chunk in requests.chunks(64) {
            result.extend(self.run(base, w1, w2, chunk)?);
        }
        Ok(result)
    }

    pub fn run(
        &mut self,
        base: &CudaSlice<bf16>,
        w1: &CudaSlice<bf16>,
        w2: &CudaSlice<bf16>,
        requests: &[(u8, u32)],
    ) -> Result<Vec<Top4>> {
        let p = requests.len();
        ensure!(p > 0 && p <= 64, "Markov batch must have1..64 requests");
        check_requests(requests)?;
        ensure!(
            base.len() == 7 * 130560 && w1.len() == 130560 * 256 && w2.len() == 130560 * 256,
            "Markov native weight/base layout mismatch"
        );
        let (index, pt) = if p == 1 {
            (0, 1)
        } else if p <= 8 {
            (1, 8)
        } else {
            (2, 64)
        };
        let tokens: Vec<_> = requests.iter().map(|r| r.1).collect();
        let rows: Vec<_> = requests.iter().map(|r| r.0 as i32).collect();
        let s = &self.stream;
        s.memcpy_htod(&tokens, &mut self.tokens.slice_mut(..p))?;
        s.memcpy_htod(&rows, &mut self.rows.slice_mut(..p))?;
        unsafe {
            s.launch_builder(&self.first[index])
                .arg(base)
                .arg(w1)
                .arg(w2)
                .arg(&self.tokens)
                .arg(&self.rows)
                .arg(&mut self.partial_top)
                .arg(&mut self.partial_lse)
                .arg(&130560i32)
                .arg(&256i32)
                .arg(&(p as i32))
                .launch(LaunchConfig {
                    grid_dim: (510, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            let mut output = self.output.slice_mut(..p * 9);
            let (mut top, mut floats) = output.split_at_mut(p * 4);
            let (mut logp, mut lse) = floats.split_at_mut(p * 4);
            s.launch_builder(&self.second)
                .arg(&self.partial_top)
                .arg(&self.partial_lse)
                .arg(&mut top)
                .arg(&mut logp)
                .arg(&mut lse)
                .arg(&130560i32)
                .arg(&(p as i32))
                .arg(&pt)
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let mut output = vec![0u32; p * 9];
        s.memcpy_dtoh(&self.output.slice(..p * 9), &mut output)?;
        s.synchronize()?;
        let top4 = (0..p)
            .map(|i| Top4 {
                tokens: std::array::from_fn(|j| output[i * 4 + j]),
                logp: std::array::from_fn(|j| f32::from_bits(output[p * 4 + i * 4 + j])),
                lse: f32::from_bits(output[p * 8 + i]),
            })
            .collect::<Vec<_>>();
        for (i, top) in top4.iter().enumerate() {
            drop(top.distribution().with_context(|| {
                format!(
                    "Markov normalization request {:?}: top={:?} logp={:?} lse={} mass={}",
                    requests[i],
                    top.tokens,
                    top.logp,
                    top.lse,
                    top.logp.iter().map(|&v| (v as f64).exp()).sum::<f64>()
                )
            })?);
        }
        Ok(top4)
    }
}

fn check_requests(requests: &[(u8, u32)]) -> Result<()> {
    ensure!(
        requests
            .iter()
            .all(|&(row, token)| row < 7 && token < 130560),
        "Markov request B row/token is invalid"
    );
    Ok(())
}

#[cfg(test)]
#[path = "markov_tests.rs"]
mod tests;
