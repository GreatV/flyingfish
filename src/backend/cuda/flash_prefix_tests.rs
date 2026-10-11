use super::*;
use anyhow::Context;
use cudarc::driver::{CudaContext, sys};

struct Probe {
    flash: Flash,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    count: CudaSlice<i32>,
    shape: Shape,
}
impl Probe {
    fn new(dim: usize, rows: usize) -> Result<Self> {
        ensure!(
            std::env::var("FF_Q38_FULL_MODEL_WINDOW").as_deref() == Ok("granted"),
            "FA2 prefix gate requires an authorized GPU window"
        );
        let ctx = CudaContext::new(0)?;
        let s = ctx.new_stream()?;
        let shape = Shape {
            rows,
            start: 0,
            q_heads: 6,
            kv_heads: 2,
            capacity: 320,
            head_dim: dim,
            causal: true,
        };
        let data = |n, seed| {
            (0..n)
                .map(|i| bf16::from_f32((((i * 37 + seed) % 257) as f32 - 128.) / 97.))
                .collect::<Vec<_>>()
        };
        let q = s.clone_htod(&data(rows * 6 * dim, 5))?;
        let k = s.clone_htod(&data(320 * 2 * dim, 19))?;
        let v = s.clone_htod(&data(320 * 2 * dim, 43))?;
        let out = s.alloc_zeros(rows * 6 * dim)?;
        let count = s.clone_htod(&[rows as i32])?;
        let flash = Flash::new(s.clone(), rows, 6)?;
        s.synchronize()?;
        unsafe {
            ctx.disable_event_tracking();
        }
        Ok(Self {
            flash,
            q,
            k,
            v,
            out,
            count,
            shape,
        })
    }
    fn set(&mut self, start: usize) -> Result<()> {
        self.shape.start = start;
        Ok(self
            .flash
            .stream
            .memcpy_htod(&[(start + self.shape.rows) as i32], &mut self.count)?)
    }
    fn launch(&mut self, device: bool) -> Result<()> {
        let launch = if device {
            Launch::Device {
                shape: self.shape,
                key_count: self.count.slice(..1),
            }
        } else {
            self.shape.into()
        };
        self.flash.run(
            &self.q,
            6 * self.shape.head_dim,
            &self.k,
            &self.v,
            &mut self.out,
            launch,
        )
    }
    fn bits(&self) -> Result<(Vec<u16>, Vec<u32>)> {
        let s = &self.flash.stream;
        let out = s.clone_dtoh(&self.out)?;
        let lse = s.clone_dtoh(&self.flash.lse)?;
        ensure!(
            out.iter().all(|x| x.is_finite()) && lse.iter().all(|x| x.is_finite()),
            "nonfinite FA2 result"
        );
        Ok((
            out.into_iter().map(bf16::to_bits).collect(),
            lse.into_iter().map(f32::to_bits).collect(),
        ))
    }
}
fn receipt(name: &str, cases: usize) -> Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var_os("FF_Q38_MTP_OUTPUT").context("FF_Q38_MTP_OUTPUT missing")?,
    );
    std::fs::create_dir_all(&root)?;
    std::fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec_pretty(
            &serde_json::json!({"gate":name,"cases":cases,"result":"PASS","comparison":"output BF16 and LSE f32 bits"}),
        )?,
    )?;
    Ok(())
}
#[test]
#[ignore = "authorized GPU window required"]
fn fa2_device_prefix_bits_gpu() -> Result<()> {
    let mut cases = 0;
    for dim in [128, 256] {
        for rows in 1..=4 {
            let mut p = Probe::new(dim, rows)?;
            for start in [0, 1, 63, 64, 127, 255] {
                p.set(start)?;
                p.launch(false)?;
                let expected = p.bits()?;
                p.launch(true)?;
                ensure!(
                    p.bits()? == expected,
                    "FA2 output/LSE mismatch d{dim} rows{rows} start{start}"
                );
                cases += 1;
            }
        }
    }
    receipt("fa2_device_prefix_bits", cases)
}
#[test]
#[ignore = "authorized GPU window required"]
fn fa2_device_prefix_graph_gpu() -> Result<()> {
    for dim in [128, 256] {
        for rows in 1..=4 {
            let mut p = Probe::new(dim, rows)?;
            p.set(7)?;
            p.launch(true)?;
            p.flash.stream.synchronize()?;
            p.flash
                .stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let run = p.launch(true);
            let captured = p.flash.stream.end_capture(sys::CUgraphInstantiate_flags(0));
            run?;
            let graph = captured?.context("empty FA2 graph")?;
            // One graph, same count allocation, changed only device count.
            for start in [7, 129, 7] {
                p.set(start)?;
                p.launch(false)?;
                let expected = p.bits()?;
                graph.launch()?;
                ensure!(
                    p.bits()? == expected,
                    "FA2 graph froze prefix d{dim} rows{rows} start{start}"
                );
            }
        }
    }
    receipt("fa2_device_prefix_graph", 24)
}
#[test]
#[ignore = "authorized GPU window required"]
fn fa2_device_prefix_stale_control_gpu() -> Result<()> {
    for dim in [128, 256] {
        let mut p = Probe::new(dim, 4)?;
        p.set(7)?;
        p.shape.start = 129; // Deliberately leave count at the previous prefix.
        p.launch(false)?;
        let expected = p.bits()?;
        p.launch(true)?;
        let stale = p.bits()?;
        ensure!(
            stale.0 != expected.0 && stale.1 != expected.1,
            "stale-prefix control escaped detection d{dim}"
        );
        p.set(129)?;
        p.launch(true)?;
        ensure!(p.bits()? == expected, "corrected prefix failed d{dim}");
    }
    receipt("fa2_device_prefix_stale_control", 2)
}
