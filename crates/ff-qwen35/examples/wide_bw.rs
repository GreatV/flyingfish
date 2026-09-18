//! Cold-truth bandwidth for the qwen wide shapes, per shape family, with
//! L2 flush between iterations (the edge0 lesson: warm benches read 3x
//! high). Everything on the current path runs the in-4096-chunk primitive
//! (158125d), so the "current" numbers here are that primitive at each
//! target's chunk shape; "full-width" targets are what wide_gemv aims at.
//!
//! Shapes (synthetic tensors — bandwidth-bound kernels don't care about
//! values): down [5120,17408] = 4.25 chunks of [5120,4096]; gate/up
//! [2x17408,5120] = chunks of [17408,4096]; in-proj group ~[16K,5120];
//! lm_head [248320,5120] anchor.

use ff_edge0::gpu::GpuContext;
use ff_edge0::int4::GroupQuant;
use std::time::Instant;

fn synth(rows: usize, in_dim: usize) -> GroupQuant {
    let words_per_row = in_dim / 8;
    let packed: Vec<u32> = (0..rows * words_per_row)
        .map(|i| (i as u32).wrapping_mul(2654435761))
        .collect();
    let groups = in_dim / 64;
    let sb: Vec<f32> = (0..rows * groups)
        .map(|i| (i as f32 * 0.37).sin() * 0.05)
        .collect();
    GroupQuant::new(packed, sb.clone(), sb, rows, in_dim, 4).unwrap()
}

fn main() -> anyhow::Result<()> {
    let ctx = GpuContext::new()?;
    let flush_buf = ctx.upload_f32(&vec![0.5f32; 64 * 1024 * 1024])?;
    let launch_flush = || ctx.flush_l2(&flush_buf);

    let bench = |tag: &str, rows: usize, in_dim: usize, chunks: usize| {
        let q = synth(rows, in_dim);
        let gpu = ctx.upload(&q, None)?;
        let x: Vec<f32> = vec![0.25; in_dim];
        // Warm + flush-alone calibration.
        for _ in 0..2 {
            launch_flush()?;
            gpu.matvec_sync(&ctx, &x)?;
        }
        let iters = 30usize;
        let t = Instant::now();
        for _ in 0..iters {
            launch_flush()?;
            gpu.matvec_sync(&ctx, &x)?;
        }
        ctx.stream.synchronize()?;
        let total = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t2 = Instant::now();
        for _ in 0..iters {
            launch_flush()?;
        }
        ctx.stream.synchronize()?;
        let flush_us = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let us = (total - flush_us) / chunks as f64;
        // Device bytes per chunk: packed + f32 scales/biases.
        let bytes = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);
        println!(
            "{tag:34} [{rows:>6},{in_dim}] x{chunks} chunks: {us:7.1} us/chunk, {:_>7.0} GB/s",
            bytes / us / 1e3
        );
        anyhow::Ok(())
    };

    // Current-path chunk shapes (in-4096 primitive), cold.
    bench("down chunk (current)", 5120, 4096, 1)?;
    bench("gate/up chunk (current)", 17408, 4096, 1)?;
    bench("in-proj group chunk (current)", 16384, 4096, 1)?;
    bench("lm_head chunk anchor (current)", 248320, 4096, 1)?;

    // Split-K full-width down [5120, 17408]: correctness vs CPU matvec,
    // then cold rate at split 2/4.
    {
        use ff_qwen35::wide::WideKernels;
        let wk = WideKernels::load(&ctx)?;
        let rows = 5120usize;
        let in_dim = 17408usize;
        let q = synth(rows, in_dim);
        let gpu = ctx.upload(&q, None)?;
        let (packed, scales, biases) = gpu.tensors();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.031).sin() * 0.5)
            .collect();
        let dx = ctx.upload_f32(&x)?;
        let y = ctx.upload_f32(&vec![0f32; rows])?;
        let cpu = q.matvec(&x, None);
        let bytes = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);
        // gate/up shape through the same kernel, split=1 full width.
        {
            let rows = 17408usize;
            let in_dim = 5120usize;
            let q = synth(rows, in_dim);
            let gpu = ctx.upload(&q, None)?;
            let (packed, scales, biases) = gpu.tensors();
            let x: Vec<f32> = (0..in_dim)
                .map(|i| ((i as f32) * 0.029).cos() * 0.5)
                .collect();
            let dx = ctx.upload_f32(&x)?;
            let y = ctx.upload_f32(&vec![0f32; rows])?;
            let cpu = q.matvec(&x, None);
            let scratch = ctx.upload_f32(&vec![0f32; rows])?;
            wk.down(
                &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows, in_dim,
                1, 1,
            )?;
            let got = ctx.dtoh(&y)?;
            let rel = cpu
                .iter()
                .zip(&got)
                .map(|(&a2, &b2)| (a2 - b2).abs() / a2.abs().max(1.0))
                .fold(0.0f32, f32::max);
            let bytes = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);
            for _ in 0..2 {
                launch_flush()?;
                wk.down(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows,
                    in_dim, 1, 1,
                )?;
            }
            let iters = 30usize;
            let t = Instant::now();
            for _ in 0..iters {
                launch_flush()?;
                wk.down(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows,
                    in_dim, 1, 1,
                )?;
            }
            ctx.stream.synchronize()?;
            let total = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t2 = Instant::now();
            for _ in 0..iters {
                launch_flush()?;
            }
            ctx.stream.synchronize()?;
            let us = total - t2.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "gate/up full-width splitk=1    [{rows},{in_dim}]: {us:7.1} us, {:_>7.0} GB/s  (rel {rel:.2e})",
                bytes / us / 1e3
            );
        }
        for split in [2usize, 4, 8] {
            let scratch = ctx.upload_f32(&vec![0f32; split * rows])?;
            wk.down(
                &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows, in_dim,
                split, 1,
            )?;
            let got = ctx.dtoh(&y)?;
            let rel = cpu
                .iter()
                .zip(&got)
                .map(|(&a, &b)| (a - b).abs() / a.abs().max(1.0))
                .fold(0.0f32, f32::max);
            for _ in 0..2 {
                launch_flush()?;
                wk.down(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows,
                    in_dim, split, 1,
                )?;
            }
            let iters = 30usize;
            let t = Instant::now();
            for _ in 0..iters {
                launch_flush()?;
                wk.down(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scratch, &scratch, rows,
                    in_dim, split, 1,
                )?;
            }
            ctx.stream.synchronize()?;
            let total = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t2 = Instant::now();
            for _ in 0..iters {
                launch_flush()?;
            }
            ctx.stream.synchronize()?;
            let flush_us = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let us = total - flush_us;
            println!(
                "down full-width splitk={split}      [{rows},{in_dim}]: {us:7.1} us, {:_>7.0} GB/s  (rel {rel:.2e})",
                bytes / us / 1e3
            );
        }
    }

    // RPB tail probe: back-to-back (NO flush — the in-pipeline regime)
    // splitk rates at the two RPB candidates. Cold-isolated said 16 > 8;
    // the pipeline may disagree if the tail matters.
    {
        use ff_qwen35::wide::WideKernels;
        let wk = WideKernels::load(&ctx)?;
        let rows = 5120usize;
        let in_dim = 17408usize;
        let q = synth(rows, in_dim);
        let gpu = ctx.upload(&q, None)?;
        let (packed, scales, biases) = gpu.tensors();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.033).sin() * 0.5)
            .collect();
        let dx = ctx.upload_f32(&x)?;
        let y = ctx.upload_f32(&vec![0f32; rows])?;
        let scr = ctx.upload_f32(&vec![0f32; 4 * rows])?;
        let iters = 200usize;
        for _ in 0..20 {
            wk.down(
                &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scr, &scr, rows, in_dim, 4, 1,
            )?;
        }
        ctx.stream.synchronize()?;
        let t = Instant::now();
        for _ in 0..iters {
            wk.down(
                &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scr, &scr, rows, in_dim, 4, 1,
            )?;
        }
        ctx.stream.synchronize()?;
        let bytes = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);
        println!(
            "back-to-back splitk (RPB as compiled): {:.1} us, {:.0} GB/s",
            t.elapsed().as_secs_f64() * 1e6 / iters as f64,
            bytes / (t.elapsed().as_secs_f64() / iters as f64) / 1e3
        );
    }

    // uint4-widened splitk A/B: correctness + cold + warm regimes.
    {
        use ff_qwen35::wide::WideKernels;
        let wk = WideKernels::load(&ctx)?;
        let rows = 5120usize;
        let in_dim = 17408usize;
        let q = synth(rows, in_dim);
        let gpu = ctx.upload(&q, None)?;
        let (packed, scales, biases) = gpu.tensors();
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.035).sin() * 0.5)
            .collect();
        let dx = ctx.upload_f32(&x)?;
        let y = ctx.upload_f32(&vec![0f32; rows])?;
        let scr = ctx.upload_f32(&vec![0f32; 4 * rows])?;
        let cpu = q.matvec(&x, None);
        let bytes = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);

        wk.down_v4(
            &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scr, &scr, rows, in_dim, 4, 1,
        )?;
        let got = ctx.dtoh(&y)?;
        let rel = cpu
            .iter()
            .zip(&got)
            .map(|(&c, &g)| (c - g).abs() / c.abs().max(1.0))
            .fold(0.0f32, f32::max);
        println!("v4 correctness rel (split4): {rel:.2e}");
        // split=1 isolate: single slice, no cross-slice combine.
        {
            let q1 = synth(5120, 5120);
            let g1 = ctx.upload(&q1, None)?;
            let (p1, s1c, b1) = g1.tensors();
            let x1: Vec<f32> = (0..5120)
                .map(|i| ((i as f32) * 0.021).cos() * 0.5)
                .collect();
            let dx1 = ctx.upload_f32(&x1)?;
            let y1 = ctx.upload_f32(&vec![0f32; 5120])?;
            let scr1 = ctx.upload_f32(&vec![0f32; 5120])?;
            wk.down_v4(
                &ctx, p1, s1c, b1, &dx1, &dx1, &y1, &y1, &scr1, &scr1, 5120, 5120, 1, 1,
            )?;
            let got1 = ctx.dtoh(&y1)?;
            let cpu1 = q1.matvec(&x1, None);

            let rel1 = got1
                .iter()
                .zip(&cpu1)
                .map(|(&c, &g)| (c - g).abs() / c.abs().max(1.0))
                .fold(0.0f32, f32::max);
            println!("v4 split=1 rel: {rel1:.2e}");
        }

        for tag in ["cold", "warm"] {
            for _ in 0..(if tag == "cold" { 3 } else { 20 }) {
                if tag == "cold" {
                    launch_flush()?;
                }
                wk.down_v4(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scr, &scr, rows, in_dim, 4, 1,
                )?;
            }
            ctx.stream.synchronize()?;
            let iters = 30usize;
            let t = Instant::now();
            for _ in 0..iters {
                if tag == "cold" {
                    launch_flush()?;
                }
                wk.down_v4(
                    &ctx, packed, scales, biases, &dx, &dx, &y, &y, &scr, &scr, rows, in_dim, 4, 1,
                )?;
            }
            ctx.stream.synchronize()?;
            let mut us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            if tag == "cold" {
                let t2 = Instant::now();
                for _ in 0..iters {
                    launch_flush()?;
                }
                ctx.stream.synchronize()?;
                us -= t2.elapsed().as_secs_f64() * 1e6 / iters as f64;
            }
            println!("v4 splitk {tag}: {us:7.1} us, {:.0} GB/s", bytes / us / 1e3);
        }
    }

    // Column-B free-ness: cols=2 vs two cols=1 launches; bit-identity of
    // column A across col modes (determinism guarantee, asserted == 0 bits).
    {
        use ff_qwen35::wide::WideKernels;
        let wk = WideKernels::load(&ctx)?;
        let rows = 5120usize;
        let in_dim = 17408usize;
        let q = synth(rows, in_dim);
        let gpu = ctx.upload(&q, None)?;
        let (packed, scales, biases) = gpu.tensors();
        let xa: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.031).sin() * 0.5)
            .collect();
        let xb: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.043).cos() * 0.5)
            .collect();
        let dxa = ctx.upload_f32(&xa)?;
        let dxb = ctx.upload_f32(&xb)?;
        let ya = ctx.upload_f32(&vec![0f32; rows])?;
        let yb = ctx.upload_f32(&vec![0f32; rows])?;
        let scr = ctx.upload_f32(&vec![0f32; 4 * rows])?;
        let scr_b = ctx.upload_f32(&vec![0f32; 4 * rows])?;
        let cpu_a = q.matvec(&xa, None);
        let cpu_b = q.matvec(&xb, None);
        let split = 4usize;

        wk.down(
            &ctx, packed, scales, biases, &dxa, &dxb, &ya, &yb, &scr, &scr_b, rows, in_dim, split,
            2,
        )?;
        let ga = ctx.dtoh(&ya)?;
        let gb = ctx.dtoh(&yb)?;
        let rel_a = cpu_a
            .iter()
            .zip(&ga)
            .map(|(&c, &g)| (c - g).abs() / c.abs().max(1.0))
            .fold(0.0f32, f32::max);
        let rel_b = cpu_b
            .iter()
            .zip(&gb)
            .map(|(&c, &g)| (c - g).abs() / c.abs().max(1.0))
            .fold(0.0f32, f32::max);
        let y_only = ctx.upload_f32(&vec![0f32; rows])?;
        wk.down(
            &ctx, packed, scales, biases, &dxa, &dxa, &y_only, &y_only, &scr, &scr, rows, in_dim,
            split, 1,
        )?;
        let go = ctx.dtoh(&y_only)?;
        let identical = ga.iter().zip(&go).all(|(p, r)| p.to_bits() == r.to_bits());
        println!("cols2: relA {rel_a:.2e} relB {rel_b:.2e}  A(cols1)==A(cols2) bits: {identical}");

        for _ in 0..2 {
            launch_flush()?;
            wk.down(
                &ctx, packed, scales, biases, &dxa, &dxb, &ya, &yb, &scr, &scr_b, rows, in_dim,
                split, 2,
            )?;
        }
        ctx.stream.synchronize()?;
        let iters = 30usize;
        let t = Instant::now();
        for _ in 0..iters {
            launch_flush()?;
            wk.down(
                &ctx, packed, scales, biases, &dxa, &dxb, &ya, &yb, &scr, &scr_b, rows, in_dim,
                split, 2,
            )?;
        }
        ctx.stream.synchronize()?;
        let tot2 = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t2 = Instant::now();
        for _ in 0..iters {
            launch_flush()?;
            wk.down(
                &ctx, packed, scales, biases, &dxa, &dxa, &ya, &ya, &scr, &scr, rows, in_dim,
                split, 1,
            )?;
            wk.down(
                &ctx, packed, scales, biases, &dxb, &dxb, &yb, &yb, &scr, &scr, rows, in_dim,
                split, 1,
            )?;
        }
        ctx.stream.synchronize()?;
        let tot1x2 = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t3 = Instant::now();
        for _ in 0..iters {
            launch_flush()?;
        }
        ctx.stream.synchronize()?;
        let fl = t3.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let bytes1 = rows as f64 * (in_dim as f64 / 2.0 + in_dim as f64 / 64.0 * 8.0);
        println!(
            "cols=2: {:.1} us ({:.0} GB/s eff) vs 2x cols=1: {:.1} us — col-2 overhead {:+.0}%",
            tot2 - fl,
            2.0 * bytes1 / (tot2 - fl) / 1e3,
            tot1x2 - 2.0 * fl,
            ((tot2 - fl) / (tot1x2 - 2.0 * fl) - 1.0) * 100.0
        );
    }

    Ok(())
}
