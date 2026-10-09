use super::{Device, EmbedOut, Kv, State};
use crate::config::Config;
use crate::tree::Tree;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, sys};
use half::bf16;
use std::path::PathBuf;

/// A graph captured with a narrow tree replays a wider tree bit for bit against eager.
#[test]
#[ignore = "requires a reserved GPU window and FF_SMALL_OPS_MODEL (a model directory)"]
fn tree_prepare_embed_norm_replay_serves_wider_tree() -> Result<()> {
    let model =
        PathBuf::from(std::env::var("FF_SMALL_OPS_MODEL").context("FF_SMALL_OPS_MODEL missing")?);
    let config = Config::read(&model)?;
    let d = Device::new(0)?;
    let capacity = 4096;
    let budget = 16;
    let hidden = config.hidden_size;
    let qkv_dim = config.qkv_dim();
    let kv_dim = config.kv_dim();
    let s = &d.stream;

    let mut kv: Vec<Kv> = Vec::with_capacity(config.num_hidden_layers);
    for _ in 0..config.num_hidden_layers {
        kv.push(Kv {
            k: s.alloc_zeros(capacity * kv_dim)?,
            v: s.alloc_zeros(capacity * kv_dim)?,
        });
    }
    let mut state = State::new(&d, &config, &kv, capacity, budget)?;
    // Same prologue the engine runs before every step (engine.rs:501): drain
    // the upload stream and stop event tracking, or the capture below sees
    // uncaptured work from another stream.
    d.finish_upload()?;

    // Deterministic non-zero inputs, and a sentinel that the segment must
    // overwrite so that "written" and "untouched" are both observable.
    let pattern = |n: usize, seed: u16| -> Vec<bf16> {
        (0..n)
            .map(|i| bf16::from_bits(0x3f00 | (((i as u16).wrapping_mul(seed)) & 0x3f) ^ 0x15))
            .collect()
    };
    let sentinel = |n: usize| -> Vec<bf16> { vec![bf16::from_bits(0x3f80); n] };
    let d_table = s.clone_htod(&pattern(64 * hidden, 7))?;
    let d_weight = s.clone_htod(&pattern(hidden, 3))?;
    let mut x = s.alloc_zeros(budget * hidden)?;
    let mut y = s.alloc_zeros(budget * hidden)?;
    let mut ids = s.alloc_zeros::<u32>(budget)?;
    let mut qkv = s.alloc_zeros(budget * qkv_dim)?;
    let mut prefix = s.alloc_zeros::<i32>(1)?;

    let reset = |x: &mut CudaSlice<bf16>,
                 y: &mut CudaSlice<bf16>,
                 ids: &mut CudaSlice<u32>,
                 qkv: &mut CudaSlice<bf16>,
                 kv: &mut [Kv]|
     -> Result<()> {
        s.memcpy_htod(&sentinel(budget * hidden), x)?;
        s.memcpy_htod(&sentinel(budget * hidden), y)?;
        // qkv gets a pattern, not the sentinel: the V heads are copied through
        // unchanged, so a sentinel there would be indistinguishable from "not
        // written".
        s.memcpy_htod(&pattern(budget * qkv_dim, 5), qkv)?;
        s.memcpy_htod(&vec![u32::MAX; budget], ids)?;
        for layer in kv.iter_mut() {
            s.memcpy_htod(&sentinel(capacity * kv_dim), &mut layer.k)?;
            s.memcpy_htod(&sentinel(capacity * kv_dim), &mut layer.v)?;
        }
        s.synchronize()?;
        Ok(())
    };

    let prepare = |state: &State,
                   prefix: &mut CudaSlice<i32>,
                   ids: &mut CudaSlice<u32>,
                   x: &mut CudaSlice<bf16>,
                   y: &mut CudaSlice<bf16>,
                   qkv: &mut CudaSlice<bf16>,
                   kv: &mut [Kv]|
     -> Result<()> {
        state.prepare_embed_norm(prefix, &d_table, &d_weight, EmbedOut { ids, x, y }, &config)?;
        let layer = &mut kv[0];
        state.rope_kv(qkv, &mut layer.k, &mut layer.v)
    };

    let wide = |v: &[bf16]| -> Vec<u32> { v.iter().map(|x| x.to_bits() as u32).collect() };
    let wide_u32 = |v: &[u32]| v.to_vec();
    let wide_i32 = |v: &[i32]| v.iter().map(|x| *x as u32).collect();

    let small_tokens: Vec<u32> = vec![11, 22, 33];
    let small_parents: Vec<i32> = vec![-1, 0, 0];
    let big_tokens: Vec<u32> = (0..budget as u32).map(|i| 7 + i * 3).collect();
    let big_parents: Vec<i32> = vec![-1, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Capture the segment with the narrow tree loaded: the launch geometry is
    // frozen here and the replay has to live with it.
    let small = Tree::edges(&small_tokens, &small_parents, 100, capacity)?;
    state.load(&small)?;
    s.memcpy_htod(&[small.prefix() as i32], &mut prefix)?;
    s.synchronize()?;
    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    // Warm the segment eagerly first, exactly as the engine does before it
    // captures a verify graph: the capture must not carry first-launch work.
    prepare(
        &state,
        &mut prefix,
        &mut ids,
        &mut x,
        &mut y,
        &mut qkv,
        &mut kv,
    )?;
    s.synchronize()?;
    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    s.synchronize()?;
    s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = prepare(
        &state,
        &mut prefix,
        &mut ids,
        &mut x,
        &mut y,
        &mut qkv,
        &mut kv,
    );
    let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
    launched?;
    let graph = captured?.context("empty tree segment graph")?;
    graph.upload()?;
    s.synchronize()?;

    // Same width: the captured graph must equal the eager result.
    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    graph.launch()?;
    s.synchronize().context("same-width graph replay")?;
    let narrow = (wide(&s.clone_dtoh(&x)?), wide(&s.clone_dtoh(&y)?));
    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    prepare(
        &state,
        &mut prefix,
        &mut ids,
        &mut x,
        &mut y,
        &mut qkv,
        &mut kv,
    )?;
    s.synchronize()?;
    ensure!(
        narrow.0 == wide(&s.clone_dtoh(&x)?) && narrow.1 == wide(&s.clone_dtoh(&y)?),
        "same-width graph replay diverges from eager"
    );

    // The gate: widen the tree in the same buffers and replay the captured
    // graph. A grid sized from the captured row count traps right here.
    let big = Tree::edges(&big_tokens, &big_parents, 200, capacity)?;
    state.load(&big)?;
    s.memcpy_htod(&[big.prefix() as i32], &mut prefix)?;
    s.synchronize()?;
    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    graph.launch().context("wider-tree graph launch")?;
    s.synchronize().context("wider-tree graph replay")?;
    let replayed = [
        wide(&s.clone_dtoh(&x)?),
        wide(&s.clone_dtoh(&y)?),
        wide_u32(&s.clone_dtoh(&ids)?),
        wide_i32(&s.clone_dtoh(state.positions())?),
        wide_i32(&s.clone_dtoh(state.slots())?),
        wide_i32(&s.clone_dtoh(state.prefix())?),
        wide(&s.clone_dtoh(&qkv)?),
        wide(&s.clone_dtoh(&kv[0].k)?),
        wide(&s.clone_dtoh(&kv[0].v)?),
    ];

    reset(&mut x, &mut y, &mut ids, &mut qkv, &mut kv)?;
    prepare(
        &state,
        &mut prefix,
        &mut ids,
        &mut x,
        &mut y,
        &mut qkv,
        &mut kv,
    )?;
    s.synchronize()?;
    let eager = [
        wide(&s.clone_dtoh(&x)?),
        wide(&s.clone_dtoh(&y)?),
        wide_u32(&s.clone_dtoh(&ids)?),
        wide_i32(&s.clone_dtoh(state.positions())?),
        wide_i32(&s.clone_dtoh(state.slots())?),
        wide_i32(&s.clone_dtoh(state.prefix())?),
        wide(&s.clone_dtoh(&qkv)?),
        wide(&s.clone_dtoh(&kv[0].k)?),
        wide(&s.clone_dtoh(&kv[0].v)?),
    ];

    // The wider tree must really have written its rows, otherwise the equality
    // below could hold on untouched sentinel buffers. Cache slots are indexed
    // (kv_head * capacity + position) * head_dim + d, so each kv head has its
    // own written window.
    let sentinel_word = u32::from(bf16::from_bits(0x3f80).to_bits());
    let rewritten = |got: &[u32], from: usize, len: usize| -> usize {
        got[from..from + len]
            .iter()
            .filter(|v| **v != sentinel_word)
            .count()
    };
    let window = budget * config.head_dim;
    let head0 = big.prefix() * config.head_dim;
    let head1 = (capacity + big.prefix()) * config.head_dim;
    let rewritten_rows = [
        (
            "x",
            rewritten(&replayed[0], 0, budget * hidden),
            budget * hidden,
        ),
        (
            "y",
            rewritten(&replayed[1], 0, budget * hidden),
            budget * hidden,
        ),
        (
            "qkv",
            rewritten(&replayed[6], 0, budget * qkv_dim),
            budget * qkv_dim,
        ),
        (
            "k",
            rewritten(&replayed[7], head0, window) + rewritten(&replayed[7], head1, window),
            2 * window,
        ),
        (
            "v",
            rewritten(&replayed[8], head0, window) + rewritten(&replayed[8], head1, window),
            2 * window,
        ),
    ];
    for (name, count, elements) in rewritten_rows {
        println!(
            "{}",
            serde_json::json!({"tree_segment_replay": {
                "phase": "rewritten",
                "output": name,
                "non_sentinel_elements": count,
                "of": elements
            }})
        );
        ensure!(count > 0, "wider-tree replay left {name} untouched");
    }
    let ids_changed = replayed[2]
        .iter()
        .zip(&big_tokens)
        .filter(|(a, b)| **a != **b)
        .count();
    let slots: Vec<i32> = (0..budget)
        .map(|r| big.prefix() as i32 + r as i32)
        .collect();
    ensure!(
        replayed[4] == wide_i32(&slots) && replayed[5] == wide_i32(&[big.prefix() as i32]),
        "capture slots or snapshot do not follow the wider tree prefix"
    );
    ensure!(
        ids_changed == 0 && replayed[2] == wide_u32(&big_tokens),
        "capture ids do not follow the wider tree tokens"
    );

    let names = [
        "x",
        "y",
        "ids",
        "positions",
        "slots",
        "snapshot",
        "qkv",
        "k",
        "v",
    ];
    let mut diffs = 0usize;
    for ((name, got), want) in names.iter().zip(replayed.iter()).zip(eager.iter()) {
        let differing = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        diffs += differing;
        println!(
            "{}",
            serde_json::json!({"tree_segment_replay": {
                "output": name,
                "replay_vs_eager_differing": differing,
                "pass": differing == 0
            }})
        );
    }
    println!(
        "{}",
        serde_json::json!({"tree_segment_replay": {
            "captured_rows": small.nodes().len(),
            "replayed_rows": big.nodes().len(),
            "budget": budget,
            "pass": diffs == 0
        }})
    );
    ensure!(diffs == 0, "wider-tree graph replay diverged from eager");
    Ok(())
}
