use crate::backend::{
    Top4,
    cuda::{Device, cubin},
};
use anyhow::{Context, Result, ensure};
use cudarc::{
    driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use half::bf16;
use std::{path::Path, sync::Arc};

pub struct Reference {
    stream: Arc<CudaStream>,
    first: [CudaFunction; 3],
    second: CudaFunction,
    tokens: CudaSlice<u32>,
    rows: CudaSlice<i32>,
    top: CudaSlice<f32>,
    lse: CudaSlice<f32>,
    output: CudaSlice<u32>,
}

impl Reference {
    pub fn new(d: &Device) -> Result<Self> {
        let directory = std::env::var("FF_MARKOV_V1_DIR").context("FF_MARKOV_V1_DIR missing")?;
        let sm = cubin::capability(&d.ctx)?;
        let path = Path::new(&directory).join(format!("markov_sm{sm}.cubin"));
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read v1 reference cubin {}", path.display()))?;
        ensure!(
            bytes.starts_with(b"\x7fELF"),
            "v1 reference must be a native cubin"
        );
        println!("markov_v1_cubin={}", path.display());
        let module = d.ctx.load_module(Ptx::from_binary(bytes))?;
        let s = &d.stream;
        Ok(Self {
            stream: s.clone(),
            first: [
                module.load_function("markov_top4_phase1_p1")?,
                module.load_function("markov_top4_phase1_p8")?,
                module.load_function("markov_top4_phase1_p64")?,
            ],
            second: module.load_function("markov_top4_phase2")?,
            tokens: s.alloc_zeros(64)?,
            rows: s.alloc_zeros(64)?,
            top: s.alloc_zeros(16320 * 64 * 8)?,
            lse: s.alloc_zeros(16320 * 64 * 2)?,
            output: s.alloc_zeros(64 * 9)?,
        })
    }

    pub fn batch(
        &mut self,
        base: &CudaSlice<bf16>,
        w1: &CudaSlice<bf16>,
        w2: &CudaSlice<bf16>,
        requests: &[(u8, u32)],
    ) -> Result<Vec<Top4>> {
        let mut values = Vec::with_capacity(requests.len());
        for chunk in requests.chunks(64) {
            values.extend(self.run(base, w1, w2, chunk)?);
        }
        Ok(values)
    }

    fn run(
        &mut self,
        base: &CudaSlice<bf16>,
        w1: &CudaSlice<bf16>,
        w2: &CudaSlice<bf16>,
        requests: &[(u8, u32)],
    ) -> Result<Vec<Top4>> {
        let p = requests.len();
        ensure!(
            (1..=64).contains(&p),
            "v1 reference request count must be in1..64"
        );
        ensure!(
            base.len() == 7 * 130560 && w1.len() == 130560 * 256 && w2.len() == 130560 * 256,
            "v1 reference weight/base layout mismatch"
        );
        let (index, pt) = if p == 1 {
            (0, 1)
        } else if p <= 8 {
            (1, 8)
        } else {
            (2, 64)
        };
        let tokens: Vec<_> = requests.iter().map(|r| r.1).collect();
        let rows: Vec<_> = requests.iter().map(|r| i32::from(r.0)).collect();
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
                .arg(&mut self.top)
                .arg(&mut self.lse)
                .arg(&130560i32)
                .arg(&256i32)
                .arg(&(p as i32))
                .launch(LaunchConfig {
                    grid_dim: (16320, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 616 * pt,
                })?;
            let mut output = self.output.slice_mut(..p * 9);
            let (mut top, mut floats) = output.split_at_mut(p * 4);
            let (mut logp, mut lse) = floats.split_at_mut(p * 4);
            s.launch_builder(&self.second)
                .arg(&self.top)
                .arg(&self.lse)
                .arg(&mut top)
                .arg(&mut logp)
                .arg(&mut lse)
                .arg(&130560i32)
                .arg(&(p as i32))
                .arg(&(pt as i32))
                .launch(LaunchConfig {
                    grid_dim: (p as u32, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 40960,
                })?;
        }
        let mut values = vec![0u32; p * 9];
        s.memcpy_dtoh(&self.output.slice(..p * 9), &mut values)?;
        s.synchronize()?;
        Ok((0..p)
            .map(|i| Top4 {
                tokens: std::array::from_fn(|j| values[i * 4 + j]),
                logp: std::array::from_fn(|j| f32::from_bits(values[p * 4 + i * 4 + j])),
                lse: f32::from_bits(values[p * 8 + i]),
            })
            .collect())
    }
}
