//! CPU gates for the MoE prefill op (scratch sizing, problem grouping) and
//! the frozen GPU gate. The GPU test is ignored by default and runs only
//! under an acknowledged window with --include-ignored --test-threads=1.
use super::combine::{CombineOps, MoeCombine};
use super::prefill::{
    GEMM1_PTR_WORDS, GEMM2_PTR_WORDS, MAX_EXPERTS, MoePrefill, MoePrefillContribute,
    MoePrefillFinish, PrefillOps, d2h_active_total_word, d2h_window_words, group_runs,
    prefill_layout, prefill_scratch_bytes, prefill_scratch_bytes_mixed, problem_order,
};
use super::reference::{self, Combine, MoeDims, MoeFixture};
use super::testing;
use crate::backend::cuda::contracts::{Scratch, TensorMut};
use crate::backend::cuda::cubin;
use crate::backend::cuda::ops::grid;
use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::sync::Arc;

fn up256(n: u64) -> u64 {
    n.div_ceil(256) * 256
}

#[test]
fn scratch_sizing_matches_region_math() {
    for (rows, contribute_rows, top_k, experts, hidden, inter) in [
        (2usize, 2usize, 2usize, 4usize, 8usize, 4usize),
        (37, 37, 8, 256, 2048, 512),
        (256, 16352, 8, 256, 2048, 512),
        (4096, 4096, 8, 256, 2048, 512),
    ] {
        let assignments = rows * top_k;
        let contribute_assignments = contribute_rows * top_k;
        let regions: [(usize, u64); 13] = [
            (experts, 4),
            (1, 4),
            (experts + 1, 4),
            (experts, 4),
            (contribute_assignments, 4),
            (assignments * hidden, 2),
            (assignments * 2 * inter, 2),
            (assignments * inter, 2),
            (assignments * hidden, 2),
            (rows * 2 * inter, 2),
            (rows * inter, 2),
            (rows * hidden, 2),
            (rows, 2),
        ];
        let expected: u64 = regions
            .iter()
            .map(|&(n, size)| up256((n as u64) * size))
            .sum();
        let got = prefill_scratch_bytes(rows, contribute_rows, top_k, experts, hidden, inter, None)
            .unwrap();
        assert_eq!(
            got, expected,
            "rows={rows} contribute_rows={contribute_rows} top_k={top_k}"
        );
        assert_eq!(got % 256, 0);
        assert!(
            got > regions.iter().map(|&(n, s)| (n as u64) * s).sum::<u64>(),
            "scratch must pad at least one region"
        );
    }
    // Production MiniCPM prefill scratch: 24,976,384 B at 2144 rows and 25,431,040 B at 16352 rows (vs 33,312,768 B on main).
    assert_eq!(
        prefill_scratch_bytes(256, 2144, 8, 256, 2048, 512, None).unwrap(),
        24_976_384
    );
    assert_eq!(
        prefill_scratch_bytes(256, 16352, 8, 256, 2048, 512, None).unwrap(),
        25_431_040
    );
    assert!(prefill_scratch_bytes(256, 16352, 8, 256, 2048, 512, None).unwrap() <= 33_312_768);
    assert_eq!(
        prefill_scratch_bytes(4096, 4096, 8, 256, 2048, 512, None).unwrap(),
        398_601_728
    );
    assert!(
        prefill_scratch_bytes(4096, 4096, 8, 256, 2048, 512, None).unwrap()
            > prefill_scratch_bytes(512, 512, 8, 256, 2048, 512, None).unwrap()
    );
    use crate::{
        experts::catalogue::{ExpertGeometry, ExpertLayout},
        model_spec::ExpertFormat,
    };
    let geometry = ExpertGeometry {
        first_layer: 0,
        layers: 40,
        experts: 256,
        hidden: 2048,
        intermediate: 512,
        top_k: 8,
    };
    let native = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry).unwrap();
    let wave = ExpertLayout::new(ExpertFormat::Bf16, &geometry).unwrap();
    assert_eq!(native.slot_bytes(), 3_151_872);
    assert_eq!(wave.slot_bytes(), 6_291_456);
    for rows in [1usize, 2, 37, 4096] {
        let baseline = prefill_scratch_bytes(rows, rows, 8, 256, 2048, 512, None).unwrap();
        let got = prefill_scratch_bytes(rows, rows, 8, 256, 2048, 512, native.fp8_slot()).unwrap();
        assert_eq!(got, baseline);
    }
    for (hidden, intermediate, supported) in [
        (64, 128, false),
        (192, 128, false),
        (128, 96, false),
        (128, 64, true),
        (384, 192, true),
    ] {
        let geometry = ExpertGeometry {
            hidden,
            intermediate,
            ..geometry
        };
        let native = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry).unwrap();
        let scratch = prefill_scratch_bytes(
            37,
            37,
            8,
            256,
            hidden as usize,
            intermediate as usize,
            native.fp8_slot(),
        );
        assert_eq!(scratch.is_ok(), supported, "native geometry {geometry:?}");
    }
    assert!(
        prefill_scratch_bytes(usize::MAX, usize::MAX, 2, 256, 2048, 512, native.fp8_slot())
            .is_err()
    );
    assert!(prefill_scratch_bytes(1, 1, 1, 256, usize::MAX, 512, native.fp8_slot()).is_err());
    let mut wrong = native.fp8_slot().unwrap().clone();
    wrong.up.start += 16;
    assert!(prefill_scratch_bytes(37, 37, 8, 256, 2048, 512, Some(&wrong)).is_err());
}

#[test]
#[ignore = "GPU window and FF_MINICPM_REFERENCE_DIR required"]
fn fp8_prefill_gpu() -> Result<()> {
    use crate::{
        experts::catalogue::{ExpertGeometry, ExpertLayout},
        model_spec::ExpertFormat,
    };
    use anyhow::{Context, ensure};
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window required"
    );
    let root = std::path::PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("FF_MINICPM_REFERENCE_DIR missing")?,
    );
    let bytes = std::fs::read(root.join("expert_l0e0_dequant.safetensors"))?;
    let anchor = safetensors::SafeTensors::deserialize(&bytes)?;
    let model = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?,
        &serde_json::json!({"eos_token_id":[248044,248046]}),
    )?;
    let production = ExpertGeometry::from_model(&model)?;
    let (experts, top_k, hidden, inter) = (
        16usize,
        production.top_k as usize,
        production.hidden as usize,
        production.intermediate as usize,
    );
    let geometry = ExpertGeometry {
        experts: experts as u32,
        ..production
    };
    let native_layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
    let slot = native_layout
        .fp8_slot()
        .context("native FP8 descriptor missing")?;
    let mut fx = reference::fixture(MoeDims::new(257, experts, top_k, hidden, inter), 47);
    let mut native = vec![0u8; native_layout.slot_bytes() as usize * experts];
    for (op, n, k, code_range, scale_range) in [
        ("gate", inter, hidden, &slot.gate, &slot.gate_scale),
        ("up", inter, hidden, &slot.up, &slot.up_scale),
        ("down", hidden, inter, &slot.down, &slot.down_scale),
    ] {
        let code = anchor.tensor(&format!("{op}_proj.weight_fp8"))?;
        let scale = anchor.tensor(&format!("{op}_proj.weight_scale"))?;
        let want = anchor.tensor(op)?;
        ensure!(
            code.dtype() == safetensors::Dtype::F8_E4M3
                && code.shape() == [n, k]
                && scale.dtype() == safetensors::Dtype::BF16
                && scale.shape() == [n, 1]
                && want.dtype() == safetensors::Dtype::BF16
                && want.shape() == [n, k],
            "invalid real expert anchor {op}"
        );
        for e in 0..experts {
            let base = e * native_layout.slot_bytes() as usize;
            for row in 0..n {
                for col in 0..k {
                    native[base + code_range.start as usize + row * k + col] =
                        code.data()[row * k + (col + e * 37) % k];
                }
            }
            for (row, value) in scale.data().as_chunks::<2>().0.iter().enumerate() {
                let scale = bf16::from_f32(
                    bf16::from_le_bytes(*value).to_f32() * (1.0 + ((e * 3) % 7) as f32 / 16.0),
                );
                let at = base + scale_range.start as usize + row * 2;
                native[at..at + 2].copy_from_slice(&scale.to_le_bytes());
                for col in 0..k {
                    let code = native[base + code_range.start as usize + row * k + col];
                    let value = crate::fp8::effective_weight(code, scale);
                    ensure!(value.is_finite(), "nonfinite effective expert weight");
                    if e == 0 {
                        let at = (row * k + col) * 2;
                        ensure!(
                            value.to_le_bytes() == want.data()[at..at + 2],
                            "real expert scalar anchor mismatch"
                        );
                    }
                    if op == "down" {
                        fx.down[(e * hidden + row) * inter + col] = value;
                    } else {
                        let offset = usize::from(op == "up") * inter;
                        fx.gate_up[(e * 2 * inter + offset + row) * hidden + col] = value;
                    }
                }
            }
        }
    }
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s, ExpertFormat::Fp8W8a16)?;
    let native_buf = s.clone_htod(&native)?;
    let native_base = native_buf.device_ptr(s).0;
    let addresses = (0..experts)
        .map(|e| native_base + e as u64 * native_layout.slot_bytes())
        .collect::<Vec<_>>();
    let fp8_slots = device_slots(
        s,
        addresses.clone(),
        addresses,
        vec![1; experts],
        vec![1; experts],
    )?;
    let gu = s.clone_htod(&fx.gate_up)?;
    let dn = s.clone_htod(&fx.down)?;
    let bf16_slots = device_slots(
        s,
        (0..experts)
            .map(|e| gu.device_ptr(s).0 + (e * 2 * inter * hidden * 2) as u64)
            .collect(),
        (0..experts)
            .map(|e| dn.device_ptr(s).0 + (e * hidden * inter * 2) as u64)
            .collect(),
        vec![1; experts],
        vec![1; experts],
    )?;
    let shared = SharedBufs {
        sg: s.clone_htod(&fx.shared_gate)?,
        su: s.clone_htod(&fx.shared_up)?,
        sd: s.clone_htod(&fx.shared_down)?,
        seg: s.clone_htod(&fx.shared_expert_gate)?,
    };
    let mut defects = 0;
    for (rows, arithmetic) in [1usize, 17, 37, 129, 257]
        .into_iter()
        .flat_map(|rows| (0..3).map(move |arithmetic| (rows, arithmetic)))
    {
        let cpu_arithmetic = (0..experts)
            .map(|e| arithmetic == 2 || (arithmetic == 1 && e % 3 == 1))
            .collect::<Vec<_>>();
        let layout = prefill_layout(rows, rows, top_k, experts, hidden, inter, Some(slot))?;
        let scratch_bytes =
            prefill_scratch_bytes(rows, rows, top_k, experts, hidden, inter, Some(slot))?;
        let scratch_buf = s.alloc_zeros::<u8>(scratch_bytes as usize)?;
        let scratch = Scratch {
            ptr: scratch_buf.device_ptr(s).0,
            bytes: scratch_bytes,
        };
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let bufs = case_bufs(
            s,
            &fx,
            rows,
            &ids,
            &vec![1; rows * top_k],
            &weights,
            &vec![1; rows * top_k],
        )?;
        let reference = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        let mut reference_input =
            contribute_input(s, &fx, rows, &bufs, &bf16_slots, &shared, &reference)?;
        reference_input.cpu_arithmetic = &cpu_arithmetic;
        ops.prefill_contribute(s, reference_input, &scratch)?;
        let expected = s.clone_dtoh(&reference.contributions)?;
        let canonical = || -> Result<Vec<u8>> {
            let assignments = rows * top_k;
            let positions = s.clone_dtoh(
                &scratch_buf
                    .slice(layout.pos_of as usize..layout.pos_of as usize + assignments * 4),
            )?;
            let gate_up = s.clone_dtoh(
                &scratch_buf
                    .slice(layout.h1 as usize..layout.h1 as usize + assignments * 2 * inter * 2),
            )?;
            let down = s.clone_dtoh(&scratch_buf.slice(
                layout.y_perm as usize..layout.y_perm as usize + assignments * hidden * 2,
            ))?;
            let mut ordered = Vec::with_capacity(gate_up.len() + down.len());
            for bytes in positions.as_chunks::<4>().0 {
                let position = u32::from_le_bytes(*bytes) as usize;
                ensure!(position < assignments, "prefill gate segment position");
                ordered
                    .extend_from_slice(&gate_up[position * 4 * inter..(position + 1) * 4 * inter]);
                ordered
                    .extend_from_slice(&down[position * 2 * hidden..(position + 1) * 2 * hidden]);
            }
            Ok(ordered)
        };
        let expected_gemms = canonical()?;
        let candidate = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        let mut input = contribute_input(s, &fx, rows, &bufs, &fp8_slots, &shared, &candidate)?;
        input.fp8_slot = Some(slot);
        input.cpu_arithmetic = &cpu_arithmetic;
        d.finish_upload()?;
        ops.prefill_contribute(s, input, &scratch)?;
        let actual = s.clone_dtoh(&candidate.contributions)?;
        ensure!(
            actual.iter().all(|v| v.is_finite()),
            "nonfinite FP8 prefill contribution"
        );
        let output_bad = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let gemm_bad = canonical()?
            .iter()
            .zip(&expected_gemms)
            .filter(|(a, b)| a != b)
            .count();
        let done_bad = s
            .clone_dtoh(&candidate.done)?
            .iter()
            .filter(|&&v| v != 1)
            .count();
        let bad = output_bad + gemm_bad + done_bad;
        println!(
            "FP8_PREFILL rows={rows} arithmetic={arithmetic} expert_subset={experts} top_k={top_k} native_gate=[{inter},{hidden}] native_up=[{inter},{hidden}] native_down=[{hidden},{inter}] native_slot_bytes={} expert_materialization_bytes=0 gemm_byte_mismatches={gemm_bad} contribution_bit_mismatches={output_bad} done_mismatches={done_bad} {}",
            native_layout.slot_bytes(),
            if bad == 0 { "PASS" } else { "DEFECT" }
        );
        defects += usize::from(bad != 0);
        for control in ["code", "scale"]
            .into_iter()
            .filter(|_| rows == 17 && arithmetic == 1)
        {
            let mut swapped = native.clone();
            let ranges = if control == "code" {
                [&slot.gate, &slot.up, &slot.down]
            } else {
                [&slot.gate_scale, &slot.up_scale, &slot.down_scale]
            };
            for range in ranges {
                for i in range.clone() {
                    swapped.swap(i as usize, native_layout.slot_bytes() as usize + i as usize);
                }
            }
            ensure!(
                swapped != native,
                "distinct expert fixture control is not discriminating"
            );
            let swapped_buf = s.clone_htod(&swapped)?;
            let base = swapped_buf.device_ptr(s).0;
            let addresses = (0..experts)
                .map(|e| base + e as u64 * native_layout.slot_bytes())
                .collect::<Vec<_>>();
            let swapped_slots = device_slots(
                s,
                addresses.clone(),
                addresses,
                vec![1; experts],
                vec![1; experts],
            )?;
            let mutant = CohortBufs {
                contributions: s.alloc_zeros(rows * top_k * hidden)?,
                done: s.alloc_zeros(rows * top_k)?,
            };
            let mut input =
                contribute_input(s, &fx, rows, &bufs, &swapped_slots, &shared, &mutant)?;
            input.fp8_slot = Some(slot);
            input.cpu_arithmetic = &cpu_arithmetic;
            ops.prefill_contribute(s, input, &scratch)?;
            let changed = s
                .clone_dtoh(&mutant.contributions)?
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(changed > 0, "FP8 expert swap control undetected");
            println!(
                "PASS FP8_PREFILL_SWAP control={control} rows={rows} arithmetic={arithmetic} contribution_bit_mismatches={changed}"
            );
        }
    }
    println!(
        "FP8_PREFILL_SUMMARY cases=15 defects={defects} exact_bf16_grouped={}",
        defects == 0
    );
    ensure!(defects == 0, "FP8 prefill DEFECT cases={defects}");
    Ok(())
}

/// Stage capacities: the D2H window holds counts[E], the 256-byte-aligned
/// active total and offsets[E+1] for every experts count up to MAX_EXPERTS,
/// and the pointer blocks hold one A/B/C pointer per active expert plus the
/// shared entries.
#[test]
fn stage_capacity_math() {
    for experts in [1usize, 2, 8, 255, 256, 257, 1023, MAX_EXPERTS] {
        let padded_counts = up256(experts as u64 * 4) as usize;
        assert_eq!(d2h_active_total_word(experts), padded_counts / 4);
        assert_eq!(
            d2h_window_words(experts),
            padded_counts / 4 + up256(4) as usize / 4 + experts + 1,
            "experts={experts}"
        );
    }
    for experts in 1..=MAX_EXPERTS {
        assert!(
            d2h_window_words(experts) <= d2h_window_words(MAX_EXPERTS),
            "D2H window exceeds the stage capacity at experts={experts}"
        );
    }
    assert_eq!(d2h_window_words(MAX_EXPERTS), 2113);
    assert_eq!(GEMM1_PTR_WORDS, 4 * (MAX_EXPERTS + 2));
    assert_eq!(GEMM2_PTR_WORDS, 4 * (MAX_EXPERTS + 1));
}

#[test]
fn grouping_orders_and_runs() {
    let mut order = Vec::new();
    let mut runs = Vec::new();
    let counts = [0i32, 3, 0, 1, 3, 2];
    problem_order(&counts, &[false; 6], &mut order);
    assert_eq!(order, vec![3, 5, 1, 4]);
    group_runs(&order, &counts, &mut runs);
    assert_eq!(runs, vec![(1, 1), (2, 1), (3, 2)]);
    problem_order(&[0, 0, 0], &[false; 3], &mut order);
    assert!(order.is_empty());
    group_runs(&order, &[0, 0, 0], &mut runs);
    assert!(runs.is_empty());
    let flat = [2, 2, 2, 2];
    problem_order(&flat, &[false; 4], &mut order);
    assert_eq!(order, vec![0, 1, 2, 3]);
    group_runs(&order, &flat, &mut runs);
    assert_eq!(runs, vec![(2, 4)]);
    problem_order(
        &counts,
        &[false, true, false, true, false, false],
        &mut order,
    );
    assert_eq!(order, vec![5, 4, 3, 1]);
    group_runs(&order[..2], &counts, &mut runs);
    assert_eq!(runs, vec![(2, 1), (3, 1)]);
}

/// The eviction-arm fixture: hot-expert weights scaled by 1.25 (bf16 round);
/// the hot expert occupies choice 0, the other choices are distinct experts
/// (router outputs are distinct per row).
fn eviction_fixture(fx: &MoeFixture, rows: usize) -> (MoeFixture, Vec<i32>, Vec<bf16>) {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let mut dims = fx.dims;
    dims.rows = rows;
    let mut fx2 = reference::fixture(dims, 0);
    let gu_base = HOT_EXPERT * 2 * i * h;
    for (dst, src) in fx2.gate_up[gu_base..gu_base + 2 * i * h]
        .iter_mut()
        .zip(fx.gate_up[gu_base..gu_base + 2 * i * h].iter())
    {
        *dst = bf16::from_f32(src.to_f32() * 1.25);
    }
    let dn_base = HOT_EXPERT * h * i;
    for (dst, src) in fx2.down[dn_base..dn_base + h * i]
        .iter_mut()
        .zip(fx.down[dn_base..dn_base + h * i].iter())
    {
        *dst = bf16::from_f32(src.to_f32() * 1.25);
    }
    let top_k = fx.dims.top_k;
    let mut ids = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        ids.push(HOT_EXPERT as i32);
        for c in 0..top_k - 1 {
            let mut e = c;
            if e == HOT_EXPERT {
                e = top_k;
            }
            ids.push(e as i32);
        }
    }
    let weights: Vec<bf16> = (0..rows * top_k)
        .map(|a| bf16::from_f32(0.1 + 0.0125 * (a % 5) as f32))
        .collect();
    (fx2, ids, weights)
}

fn planted_noise(
    fx: &MoeFixture,
    ids: &[i32],
    weights: &[bf16],
    active: &[u32],
    rows: &[usize],
) -> reference::CombineNoise {
    let k = fx.dims.top_k;
    let mut noise = reference::CombineNoise {
        ascending_vs_descending: 0.0,
        ascending_vs_fp32: 0.0,
        max_abs: 0.0,
    };
    for &row in rows {
        let ids = &ids[row * k..(row + 1) * k];
        let weights = &weights[row * k..(row + 1) * k];
        let mut keep: Vec<usize> = (0..k)
            .filter(|&choice| active[row * k + choice] == 1)
            .collect();
        let zero = vec![bf16::ZERO; k];
        let weights = if keep.is_empty() {
            keep.extend(0..k);
            &zero
        } else {
            weights
        };
        let row = reference::row_noise(fx, row, ids, weights, &keep);
        noise.ascending_vs_descending = noise
            .ascending_vs_descending
            .max(row.ascending_vs_descending);
        noise.ascending_vs_fp32 = noise.ascending_vs_fp32.max(row.ascending_vs_fp32);
        noise.max_abs = noise.max_abs.max(row.max_abs);
    }
    noise
}

#[test]
fn eviction_noise_cell_is_pinned() {
    let fx = reference::fixture(MoeDims::minicpm47(6), 0);
    let (fx2, ids, weights) = eviction_fixture(&fx, 6);
    let active = all_active(6, fx.dims.top_k);
    let noise = planted_noise(&fx2, &ids, &weights, &active, &[0, 1, 2, 3, 4]);
    assert_eq!(
        noise
            .ascending_vs_descending
            .max(noise.ascending_vs_fp32)
            .to_bits(),
        0.001953125f32.to_bits()
    );
}

const SKEW_SUBSET: [usize; 16] = [
    3, 17, 42, 77, 100, 137, 150, 171, 190, 201, 220, 231, 240, 247, 250, 255,
];

/// Hot-expert id of the skew and eviction cases.
const HOT_EXPERT: usize = 7;

fn planted_uniform(rows: usize, experts: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        for kk in 0..top_k {
            ids[r * top_k + kk] = ((r * top_k + kk) % experts) as i32;
            weights[r * top_k + kk] = bf16::from_f32(0.1 + 0.0125 * ((r + 3 * kk) % 5) as f32);
        }
    }
    (ids, weights)
}

fn planted_skewed(rows: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        for kk in 0..top_k {
            ids[r * top_k + kk] = SKEW_SUBSET[(r + kk) % 16] as i32;
            weights[r * top_k + kk] = bf16::from_f32(0.1 + 0.0125 * ((r + 5 * kk) % 5) as f32);
        }
    }
    (ids, weights)
}

/// One expert takes ~1/3 of all assignments: the hot expert occupies choice 0
/// of every row (distinct top-k caps a single expert at rows assignments);
/// other choices are distinct experts.
fn planted_hot(rows: usize, experts: usize, top_k: usize) -> (Vec<i32>, Vec<bf16>) {
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    for r in 0..rows {
        ids[r * top_k] = HOT_EXPERT as i32;
        for c in 1..top_k {
            let mut e = (r + c - 1) % (experts - 1);
            if e == HOT_EXPERT {
                e = experts - 1;
            }
            ids[r * top_k + c] = e as i32;
        }
        for c in 0..top_k {
            weights[r * top_k + c] = bf16::from_f32(0.1 + 0.0125 * ((r + 3 * c) % 5) as f32);
        }
    }
    (ids, weights)
}

/// 16 rows at an odd stride, coprime with the power-of-two counts under test.
fn sample_rows(rows: usize) -> Vec<usize> {
    (0..16).map(|j| (403 * j) % rows).collect()
}

fn all_active(rows: usize, top_k: usize) -> Vec<u32> {
    vec![1u32; rows * top_k]
}

struct CaseBufs {
    x: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    active: CudaSlice<u32>,
    weights: CudaSlice<bf16>,
    gens: CudaSlice<u64>,
    contributions: CudaSlice<bf16>,
    done: CudaSlice<u32>,
    y: CudaSlice<bf16>,
}

fn case_bufs(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
) -> Result<CaseBufs> {
    let (top_k, hidden) = (fx.dims.top_k, fx.dims.hidden);
    Ok(CaseBufs {
        x: s.clone_htod(&fx.x[..rows * hidden])?,
        ids: s.clone_htod(ids)?,
        active: s.clone_htod(active)?,
        weights: s.clone_htod(weights)?,
        gens: s.clone_htod(gens)?,
        contributions: s.alloc_zeros(rows * top_k * hidden)?,
        done: s.alloc_zeros(rows * top_k)?,
        y: s.alloc_zeros(rows * hidden)?,
    })
}

struct SharedBufs {
    sg: CudaSlice<bf16>,
    su: CudaSlice<bf16>,
    sd: CudaSlice<bf16>,
    seg: CudaSlice<bf16>,
}

/// Host slot tables plus the device tables (pointer arrays, valid mask,
/// generation) they were published with.
struct DeviceSlots {
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    gu_dev: CudaSlice<u64>,
    dn_dev: CudaSlice<u64>,
    valid: CudaSlice<u32>,
    generation: CudaSlice<u64>,
}

fn device_slots(
    s: &Arc<CudaStream>,
    gu_host: Vec<u64>,
    dn_host: Vec<u64>,
    valid: Vec<u32>,
    generation: Vec<u64>,
) -> Result<DeviceSlots> {
    Ok(DeviceSlots {
        gu_dev: s.clone_htod(&gu_host)?,
        dn_dev: s.clone_htod(&dn_host)?,
        valid: s.clone_htod(&valid)?,
        generation: s.clone_htod(&generation)?,
        gu_host,
        dn_host,
    })
}

fn moe_input<'a>(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    bufs: &CaseBufs,
    slots: &'a DeviceSlots,
    shared: &SharedBufs,
    y: TensorMut<bf16, 2>,
) -> Result<MoePrefill<'a>> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    Ok(MoePrefill {
        expert_inter: inter as u32,
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&slots.gu_dev, s, [experts as u32], [1])?,
        slots_down: testing::view(&slots.dn_dev, s, [experts as u32], [1])?,
        slot_valid: testing::view(&slots.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&slots.generation, s, [experts as u32], [1])?,
        assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
        slots_gate_up_host: &slots.gu_host,
        slots_down_host: &slots.dn_host,
        shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
        contributions: testing::view(
            &bufs.contributions,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        done: testing::mut_view(
            &bufs.done,
            s,
            [rows as u32, top_k as u32],
            [top_k as u64, 1],
        )?,
        y,
    })
}

/// Caller-owned contribute targets: the contributions/done pair a cohort
/// writes into.
struct CohortBufs {
    contributions: CudaSlice<bf16>,
    done: CudaSlice<u32>,
}

/// A contribute input over caller-owned contributions/done buffers.
fn contribute_input<'a>(
    s: &Arc<CudaStream>,
    fx: &MoeFixture,
    rows: usize,
    bufs: &CaseBufs,
    slots: &'a DeviceSlots,
    shared: &SharedBufs,
    cohort: &CohortBufs,
) -> Result<MoePrefillContribute<'a>> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    Ok(MoePrefillContribute {
        expert_inter: inter as u32,
        cpu_arithmetic: &[false; MAX_EXPERTS][..experts],
        fp8_slot: None,
        wave: None,
        wait_copy: None,
        reader_event: None,
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&bufs.weights, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&slots.gu_dev, s, [experts as u32], [1])?,
        slots_down: testing::view(&slots.dn_dev, s, [experts as u32], [1])?,
        slot_valid: testing::view(&slots.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&slots.generation, s, [experts as u32], [1])?,
        assignment_gen: testing::dense(&bufs.gens, s, rows as u32, top_k as u32)?,
        slots_gate_up_host: &slots.gu_host,
        slots_down_host: &slots.dn_host,
        shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
        contributions: testing::view(
            &cohort.contributions,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        done: testing::mut_view(
            &cohort.done,
            s,
            [rows as u32, top_k as u32],
            [top_k as u64, 1],
        )?,
    })
}

/// Raw device-address copies for the mutant/join cells.
fn raw_dtoh<T>(s: &Arc<CudaStream>, src: u64, dst: &mut [T]) -> Result<()> {
    s.context().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().cast(),
            src,
            std::mem::size_of_val(dst),
            s.cu_stream(),
        )
        .result()?;
    }
    s.synchronize()?;
    Ok(())
}

fn raw_htod<T>(s: &Arc<CudaStream>, dst: u64, src: &[T]) -> Result<()> {
    s.context().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            dst,
            src.as_ptr().cast(),
            std::mem::size_of_val(src),
            s.cu_stream(),
        )
        .result()?;
    }
    s.synchronize()?;
    Ok(())
}

struct Arm<'a> {
    label: &'a str,
    rows: usize,
    ids: &'a [i32],
    active: &'a [u32],
    weights: &'a [bf16],
    check: &'a [usize],
    gens: &'a [u64],
    slots: &'a DeviceSlots,
}

struct GpuCase<'a> {
    s: &'a Arc<CudaStream>,
    ops: &'a PrefillOps,
    fx: &'a MoeFixture,
    scratch: &'a Scratch,
    shared: &'a SharedBufs,
}

impl GpuCase<'_> {
    fn bufs(
        &self,
        rows: usize,
        ids: &[i32],
        active: &[u32],
        weights: &[bf16],
        gens: &[u64],
    ) -> Result<CaseBufs> {
        case_bufs(self.s, self.fx, rows, ids, active, weights, gens)
    }

    /// Runs one arm, compares the active choices of every checked row against
    /// the reference and records tolerance misses into `defects`.
    fn run(&self, arm: Arm<'_>, defects: &mut Vec<String>) -> Result<()> {
        let Arm {
            label,
            rows,
            ids,
            active,
            weights,
            check,
            gens,
            slots,
        } = arm;
        let tol = planted_noise(self.fx, ids, weights, active, check).tolerance();
        let s = self.s;
        let (top_k, hidden) = (self.fx.dims.top_k, self.fx.dims.hidden);
        let case_bufs = self.bufs(rows, ids, active, weights, gens)?;
        let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
        let input = moe_input(s, self.fx, rows, &case_bufs, slots, self.shared, y_view)?;
        self.ops.prefill(s, input, self.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&case_bufs.y)?;
        let mut max_diff = 0.0f32;
        for &row in check {
            let keep_row: Vec<usize> = (0..top_k)
                .filter(|&c| active[row * top_k + c] == 1)
                .collect();
            let ids_row = &ids[row * top_k..(row + 1) * top_k];
            let weights_row = &weights[row * top_k..(row + 1) * top_k];
            // Zero router weights make every contribution an exact zero, so
            // the expected row of an all-inactive arm is the gated shared
            // expert alone.
            let expected = if keep_row.is_empty() {
                let zeros = vec![bf16::ZERO; top_k];
                let all: Vec<usize> = (0..top_k).collect();
                reference::moe_row_ids(self.fx, row, ids_row, &zeros, &all, Combine::Fp32)
            } else {
                reference::moe_row_ids(self.fx, row, ids_row, weights_row, &keep_row, Combine::Fp32)
            };
            for c in 0..hidden {
                let got = y[row * hidden + c];
                anyhow::ensure!(
                    got.to_f32().is_finite(),
                    "moe_prefill {label} row {row} column {c} is non-finite"
                );
                max_diff = max_diff.max((got.to_f32() - expected[c].to_f32()).abs());
            }
        }
        eprintln!(
            "moe_prefill {label} rows={rows} checked={} max_diff={max_diff:.6} tol={:.6}",
            check.len(),
            tol
        );
        if max_diff > tol {
            defects.push(format!(
                "{label} rows={rows} max_diff {max_diff:.6} > tol {:.6}",
                tol
            ));
        }
        Ok(())
    }
}

/// Shared context of the K2/K4 join cells.
struct GateCtx<'a> {
    s: &'a Arc<CudaStream>,
    ops: &'a PrefillOps,
    combine: &'a CombineOps,
    fx: &'a MoeFixture,
    slots: &'a DeviceSlots,
    shared: &'a SharedBufs,
    scratch: &'a Scratch,
}

/// Real shared-FFN bits for the join cells: one wrapper run on this input,
/// then the shared_out region read back from scratch.
fn capture_shared_out(
    ctx: &GateCtx<'_>,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
) -> Result<CudaSlice<bf16>> {
    let fx = ctx.fx;
    let (top_k, hidden, experts, inter) = (
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.experts,
        fx.dims.inter,
    );
    let case_bufs = case_bufs(ctx.s, fx, rows, ids, active, weights, gens)?;
    let y: CudaSlice<bf16> = ctx.s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, ctx.s, rows as u32, hidden as u32)?;
    let input = moe_input(ctx.s, fx, rows, &case_bufs, ctx.slots, ctx.shared, y_view)?;
    ctx.ops.prefill(ctx.s, input, ctx.scratch)?;
    ctx.s.synchronize()?;
    let layout = prefill_layout(rows, rows, top_k, experts, hidden, inter, None)?;
    let mut host = vec![bf16::ZERO; rows * hidden];
    raw_dtoh(ctx.s, ctx.scratch.ptr + layout.shared_out, &mut host)?;
    Ok(ctx.s.clone_htod(&host)?)
}

/// One wrapper-style join over caller-owned buffers: sigmoid gate once per
/// row, then the combine with `bufs.active` as the final union mask;
/// returns the y bits.
fn join_combine(
    ctx: &GateCtx<'_>,
    rows: usize,
    bufs: &CaseBufs,
    shared_out: &CudaSlice<bf16>,
    contributions: &CudaSlice<bf16>,
    done: &CudaSlice<u32>,
) -> Result<Vec<u16>> {
    let (s, fx, shared) = (ctx.s, ctx.fx, ctx.shared);
    let (top_k, hidden) = (fx.dims.top_k, fx.dims.hidden);
    let scale: CudaSlice<bf16> = s.alloc_zeros(rows)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let mut scale_view = testing::dense_mut(&scale, s, rows as u32, 1)?;
    ctx.combine.shared_scale(
        s,
        testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        testing::view(&shared.seg, s, [hidden as u32], [1])?,
        &mut scale_view,
    )?;
    let input = MoeCombine {
        ids: testing::dense(&bufs.ids, s, rows as u32, top_k as u32)?,
        final_active: testing::dense(&bufs.active, s, rows as u32, top_k as u32)?,
        done: testing::view(done, s, [rows as u32, top_k as u32], [top_k as u64, 1])?,
        contributions: testing::view(
            contributions,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        shared_out: testing::dense(shared_out, s, rows as u32, hidden as u32)?,
        x: testing::dense(&bufs.x, s, rows as u32, hidden as u32)?,
        shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
        scale: Some(scale_view),
        y: testing::dense_mut(&y, s, rows as u32, hidden as u32)?,
    };
    ctx.combine.combine(s, &input)?;
    s.synchronize()?;
    Ok(s.clone_dtoh(&y)?.iter().map(|v| v.to_bits()).collect())
}

/// Contribute `cohorts` sequentially into one caller-owned
/// contributions/done pair, then one join; returns the y bits.
fn join_bits(
    ctx: &GateCtx<'_>,
    rows: usize,
    ids: &[i32],
    weights: &[bf16],
    gens: &[u64],
    shared_out: &CudaSlice<bf16>,
    cohorts: &[Vec<u32>],
) -> Result<Vec<u16>> {
    let (s, fx) = (ctx.s, ctx.fx);
    let (top_k, hidden) = (fx.dims.top_k, fx.dims.hidden);
    let cohort = CohortBufs {
        contributions: s.alloc_zeros(rows * top_k * hidden)?,
        done: s.alloc_zeros(rows * top_k)?,
    };
    for active in cohorts {
        let bufs = case_bufs(s, fx, rows, ids, active, weights, gens)?;
        let input = contribute_input(s, fx, rows, &bufs, ctx.slots, ctx.shared, &cohort)?;
        ctx.ops.prefill_contribute(s, input, ctx.scratch)?;
        s.synchronize()?;
    }
    let union = case_bufs(s, fx, rows, ids, &all_active(rows, top_k), weights, gens)?;
    join_combine(
        ctx,
        rows,
        &union,
        shared_out,
        &cohort.contributions,
        &cohort.done,
    )
}

/// Runs one control input and returns its max diff against the reference on
/// the original ids/weights.
fn control_diff(
    case: &GpuCase<'_>,
    rows: usize,
    ids: &[i32],
    active: &[u32],
    weights: &[bf16],
    gens: &[u64],
    slots: &DeviceSlots,
) -> Result<f32> {
    let s = case.s;
    let (top_k, hidden) = (case.fx.dims.top_k, case.fx.dims.hidden);
    let case_bufs = case.bufs(rows, ids, active, weights, gens)?;
    let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
    let input = moe_input(s, case.fx, rows, &case_bufs, slots, case.shared, y_view)?;
    case.ops.prefill(s, input, case.scratch)?;
    s.synchronize()?;
    let y = s.clone_dtoh(&case_bufs.y)?;
    let all: Vec<usize> = (0..top_k).collect();
    let mut max_diff = 0.0f32;
    for row in 0..rows {
        let expected = reference::moe_row_ids(
            case.fx,
            row,
            &ids[row * top_k..(row + 1) * top_k],
            &weights[row * top_k..(row + 1) * top_k],
            &all,
            Combine::Fp32,
        );
        for c in 0..hidden {
            max_diff = max_diff.max((y[row * hidden + c].to_f32() - expected[c].to_f32()).abs());
        }
    }
    Ok(max_diff)
}

#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn moe_prefill_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    match std::env::var("FF_MOE_CHILD").as_deref() {
        Ok("stale") => return stale_generation_child(),
        Ok("dup") => return duplicate_ids_child(),
        Ok("oobpos") => return pos_of_oob_child(),
        Ok("k5c") => return k5c_seam_child(),
        _ => {}
    }
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let mut defects: Vec<String> = Vec::new();
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    // The 16384-row arm slices the same fixture, so its weights are the ones
    // every smaller-rows arm already scores against.
    let fx = reference::fixture(MoeDims::minicpm47(16384), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let gu_buf: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn_buf: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _g1) = gu_buf.device_ptr(s);
    let (dn_ptr, _g2) = dn_buf.device_ptr(s);
    let gu_stride = (2 * inter * hidden * 2) as u64;
    let dn_stride = (hidden * inter * 2) as u64;
    let main_gu: Vec<u64> = (0..experts)
        .map(|e| gu_ptr + e as u64 * gu_stride)
        .collect();
    let main_dn: Vec<u64> = (0..experts)
        .map(|e| dn_ptr + e as u64 * dn_stride)
        .collect();
    let main_slots = device_slots(
        s,
        main_gu.clone(),
        main_dn.clone(),
        vec![1u32; experts],
        vec![1u64; experts],
    )?;
    let scratch_bytes = prefill_scratch_bytes(16384, 16384, top_k, experts, hidden, inter, None)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    let case = GpuCase {
        s,
        ops: &ops,
        fx: &fx,
        scratch: &scratch,
        shared: &shared,
    };

    // The per-expert GEMM2 shape (n = m_e per expert, restored from the
    // per-assignment n=1 problems) is confirmed on device by every arm
    // below: they score the full path against the unchanged oracle under
    // the frozen tolerance.
    for rows in [1usize, 37, 512, 4096] {
        // Full-row reference compares run only at rows <= 37; above that the
        // debug-build host reference (about 0.1 GFLOP per row) exceeds the
        // GPU window, so checks sample rows and every arm still runs.
        let check: Vec<usize> = if rows <= 37 {
            (0..rows).collect()
        } else {
            sample_rows(rows)
        };
        let active = all_active(rows, top_k);
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "uniform",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let (ids, weights) = planted_skewed(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "skewed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        if rows == 4096 {
            let (ids, weights) = planted_hot(rows, experts, top_k);
            let gens = vec![1u64; rows * top_k];
            case.run(
                Arm {
                    label: "hot",
                    rows,
                    ids: &ids,
                    active: &active,
                    weights: &weights,
                    check: &check,
                    gens: &gens,
                    slots: &main_slots,
                },
                &mut defects,
            )?;
        }
        // Fixture-driven end-to-end: route_row ids/weights for every row at
        // rows <= 512, and for the coprime-stride sample at 4096.
        let sample = sample_rows(rows);
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            if rows <= 512 || sample.contains(&r) {
                let routed = reference::route_row(&fx, r);
                ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
                weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
            } else {
                for kk in 0..top_k {
                    ids[r * top_k + kk] = ((r * top_k + kk) % experts) as i32;
                    weights[r * top_k + kk] = bf16::from_f32(0.1);
                }
            }
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "routed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // rows=16384: 131072 assignments drive the permute/scatter grids, the
    // count loop and the largest per-expert GEMM2 segments; 16-row coprime
    // sample checks like the 512/4096 arms.
    {
        let rows = 16384usize;
        let check = sample_rows(rows);
        let active = all_active(rows, top_k);
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "uniform",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let (ids, weights) = planted_skewed(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "skewed",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // Flint geometry: routed experts use I=64 while the always-on shared
    // expert uses I=128. The old single-width path read past each routed
    // down-projection row.
    {
        let mut mixed_dims = MoeDims::new(2, 16, 4, 128, 64);
        mixed_dims.shared_inter = 128;
        let mixed = reference::fixture(mixed_dims, 11);
        let (mi, msi, mh, mk, me) = (
            mixed.dims.inter,
            mixed.dims.shared_inter,
            mixed.dims.hidden,
            mixed.dims.top_k,
            mixed.dims.experts,
        );
        let x_m = testing::upload(&d.upload, mixed.x.clone())?;
        let gu_m = testing::upload(&d.upload, mixed.gate_up.clone())?;
        let dn_m = testing::upload(&d.upload, mixed.down.clone())?;
        let sg_m = testing::upload(&d.upload, mixed.shared_gate.clone())?;
        let su_m = testing::upload(&d.upload, mixed.shared_up.clone())?;
        let sd_m = testing::upload(&d.upload, mixed.shared_down.clone())?;
        let seg_m = testing::upload(&d.upload, mixed.shared_expert_gate.clone())?;
        d.finish_upload()?;
        let routed0 = reference::route_row(&mixed, 0);
        let routed1 = reference::route_row(&mixed, 1);
        let mut ids_host_m = routed0.ids.clone();
        ids_host_m.extend_from_slice(&routed1.ids);
        let mut weights_host_m = routed0.weights.clone();
        weights_host_m.extend_from_slice(&routed1.weights);
        let active_m = vec![1u32; 2 * mk];
        let generations_m = vec![1u64; me];
        let valid_m = vec![1u32; me];
        let gu_base_m = gu_m.device_ptr(s).0;
        let dn_base_m = dn_m.device_ptr(s).0;
        let gu_stride_m = (2 * mi * mh * 2) as u64;
        let dn_stride_m = (mh * mi * 2) as u64;
        let gu_table_m: Vec<u64> = (0..me)
            .map(|expert| gu_base_m + expert as u64 * gu_stride_m)
            .collect();
        let dn_table_m: Vec<u64> = (0..me)
            .map(|expert| dn_base_m + expert as u64 * dn_stride_m)
            .collect();
        let gu_table_dev = testing::upload(s, gu_table_m.clone())?;
        let dn_table_dev = testing::upload(s, dn_table_m.clone())?;
        let ids_m = testing::upload(s, ids_host_m.clone())?;
        let weights_m = testing::upload(s, weights_host_m.clone())?;
        let valid_dev_m = testing::upload(s, valid_m)?;
        let generations_dev_m = testing::upload(s, generations_m)?;
        let assignment_gen_m = testing::upload(s, vec![1u64; 2 * mk])?;
        let active_dev_m = testing::upload(s, active_m)?;
        let contributions_buf_m = testing::zeros::<bf16>(s, 2 * mk * mh)?;
        let done_buf_m = testing::zeros::<u32>(s, 2 * mk)?;
        let contributions_m = testing::view(
            &contributions_buf_m,
            s,
            [2, mk as u32, mh as u32],
            [(mk * mh) as u64, mh as u64, 1],
        )?;
        let done_m = testing::mut_view(&done_buf_m, s, [2, mk as u32], [mk as u64, 1])?;
        let scratch_bytes_m = prefill_scratch_bytes_mixed(2, 2, mk, me, mh, (mi, msi), None)?;
        let scratch_m = testing::zeros::<u8>(s, scratch_bytes_m as usize)?;
        let scratch_view_m = Scratch {
            ptr: scratch_m.device_ptr(s).0,
            bytes: scratch_bytes_m,
        };
        let y_m = testing::zeros::<bf16>(s, 2 * mh)?;
        let input_m = MoePrefill {
            expert_inter: mi as u32,
            x: testing::dense(&x_m, s, 2, mh as u32)?,
            ids: testing::dense(&ids_m, s, 2, mk as u32)?,
            active: testing::dense(&active_dev_m, s, 2, mk as u32)?,
            weights: testing::dense(&weights_m, s, 2, mk as u32)?,
            slots_gate_up: testing::view(&gu_table_dev, s, [me as u32], [1])?,
            slots_down: testing::view(&dn_table_dev, s, [me as u32], [1])?,
            slot_valid: testing::view(&valid_dev_m, s, [me as u32], [1])?,
            slot_generation: testing::view(&generations_dev_m, s, [me as u32], [1])?,
            assignment_gen: testing::dense(&assignment_gen_m, s, 2, mk as u32)?,
            slots_gate_up_host: &gu_table_m,
            slots_down_host: &dn_table_m,
            shared_gate: testing::dense(&sg_m, s, msi as u32, mh as u32)?,
            shared_up: testing::dense(&su_m, s, msi as u32, mh as u32)?,
            shared_down: testing::dense(&sd_m, s, mh as u32, msi as u32)?,
            shared_expert_gate: Some(testing::view(&seg_m, s, [mh as u32], [1])?),
            contributions: contributions_m,
            done: done_m,
            y: testing::dense_mut(&y_m, s, 2, mh as u32)?,
        };
        ops.prefill(s, input_m, &scratch_view_m)?;
        s.synchronize()?;
        let got_m = s.clone_dtoh(&y_m)?;
        let want_m = reference::moe_row_ids(
            &mixed,
            0,
            &routed0.ids,
            &routed0.weights,
            &(0..mk).collect::<Vec<_>>(),
            reference::Combine::Fp32,
        );
        let diff_m = got_m[..mh]
            .iter()
            .zip(&want_m)
            .map(|(got, want)| (got.to_f32() - want.to_f32()).abs())
            .fold(0.0f32, f32::max);
        let tol_m = reference::tolerance(&mixed, &[0]);
        if diff_m > tol_m {
            defects.push(format!(
                "mixed prefill expert_inter={mi} shared_inter={msi} max_diff={diff_m:e} tol={tol_m:e}"
            ));
        } else {
            println!(
                "MOE_PREFILL_MIXED expert_inter={mi} shared_inter={msi} max_diff={diff_m:e} tol={tol_m:e} PASS"
            );
        }
    }

    // Partial-active cohort at rows=37: every 5th assignment inactive; the
    // reference computes exactly the kept choices of each row.
    {
        let rows = 37usize;
        let check: Vec<usize> = (0..rows).collect();
        let mut active = all_active(rows, top_k);
        for (a, v) in active.iter_mut().enumerate() {
            if a % 5 == 0 {
                *v = 0;
            }
        }
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "partial-active",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // All-inactive cohort at rows=2: active_total == 0 skips the routed
    // problems and the combine writes the gated shared expert only.
    {
        let rows = 2usize;
        let check: Vec<usize> = (0..rows).collect();
        let active = vec![0u32; rows * top_k];
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        case.run(
            Arm {
                label: "all-inactive",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
    }

    // Negative controls at rows=37 (uniform arm covers every expert, so
    // experts 3 and 17 receive rows).
    {
        let rows = 37usize;
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let tol =
            planted_noise(&fx, &ids, &weights, &active, &(0..rows).collect::<Vec<_>>()).tolerance();
        let mut swapped_gu = main_gu.clone();
        let mut swapped_dn = main_dn.clone();
        swapped_gu.swap(3, 17);
        swapped_dn.swap(3, 17);
        let swapped = device_slots(
            s,
            swapped_gu,
            swapped_dn,
            vec![1u32; experts],
            vec![1u64; experts],
        )?;
        let diff = control_diff(&case, rows, &ids, &active, &weights, &gens, &swapped)?;
        anyhow::ensure!(
            diff > tol,
            "swapped slot-table control was not detected: {diff:.6} <= {tol:.6}"
        );

        // Gate/up split off-by-one: every expert block of gate_up rolled by
        // one row.
        let mut rolled = fx.gate_up.clone();
        for e in 0..experts {
            let base = e * 2 * inter * hidden;
            for j in 0..2 * inter {
                let src = base + ((j + 1) % (2 * inter)) * hidden;
                let dst = base + j * hidden;
                rolled.copy_within(src..src + hidden, dst);
            }
        }
        let rolled_buf: CudaSlice<bf16> = s.clone_htod(rolled.as_slice())?;
        let (rolled_ptr, _gr) = rolled_buf.device_ptr(s);
        let rolled_slots = device_slots(
            s,
            (0..experts)
                .map(|e| rolled_ptr + e as u64 * gu_stride)
                .collect(),
            main_dn.clone(),
            vec![1u32; experts],
            vec![1u64; experts],
        )?;
        let diff = control_diff(&case, rows, &ids, &active, &weights, &gens, &rolled_slots)?;
        anyhow::ensure!(
            diff > tol,
            "rolled gate/up split control was not detected: {diff:.6} <= {tol:.6}"
        );
    }

    // Eviction-reuse control: all assignments route to the hot expert, whose
    // weights move to a new device buffer scaled by 1.25 (bf16 round) with
    // the slot generation bumped to 2; the relaunch must read the replaced
    // weights. Each checked row has a measured combine-order spread and BF16 ulp bound.
    {
        let rows = 37usize;
        let (fx2, ids, weights) = eviction_fixture(&fx, rows);
        let check: Vec<usize> = (0..rows).collect();
        let active = all_active(rows, top_k);
        let tol_evict = planted_noise(&fx2, &ids, &weights, &active, &check).tolerance();
        let gens = vec![1u64; rows * top_k];
        let ev_case = GpuCase {
            s: case.s,
            ops: case.ops,
            fx: case.fx,
            scratch: case.scratch,
            shared: case.shared,
        };
        ev_case.run(
            Arm {
                label: "eviction-before",
                rows,
                ids: &ids,
                active: &active,
                weights: &weights,
                check: &check,
                gens: &gens,
                slots: &main_slots,
            },
            &mut defects,
        )?;
        let scale_block = |src: &[bf16], e: usize, rows_n: usize, width: usize| -> Vec<bf16> {
            let mut out = src[e * rows_n * width..(e + 1) * rows_n * width].to_vec();
            for v in &mut out {
                *v = bf16::from_f32(v.to_f32() * 1.25);
            }
            out
        };
        let hot_gu = scale_block(&fx.gate_up, HOT_EXPERT, 2 * inter, hidden);
        let hot_dn = scale_block(&fx.down, HOT_EXPERT, hidden, inter);
        let hot_gu_buf: CudaSlice<bf16> = s.clone_htod(&hot_gu)?;
        let hot_dn_buf: CudaSlice<bf16> = s.clone_htod(&hot_dn)?;
        let (hot_gu_ptr, _h1) = hot_gu_buf.device_ptr(s);
        let (hot_dn_ptr, _h2) = hot_dn_buf.device_ptr(s);
        let mut ev_gu = main_gu.clone();
        let mut ev_dn = main_dn.clone();
        ev_gu[HOT_EXPERT] = hot_gu_ptr;
        ev_dn[HOT_EXPERT] = hot_dn_ptr;
        // Only the replaced slot advances to generation 2; the snapshot each
        // assignment carries comes from the table observed at routing time.
        let mut ev_gen = vec![1u64; experts];
        ev_gen[HOT_EXPERT] = 2;
        let mut snapshot = vec![1u64; rows * top_k];
        for r in 0..rows {
            snapshot[r * top_k] = 2;
        }
        let evicted = device_slots(s, ev_gu, ev_dn, vec![1u32; experts], ev_gen)?;
        let case_bufs = case.bufs(rows, &ids, &active, &weights, &snapshot)?;
        let y_view = testing::dense_mut(&case_bufs.y, s, rows as u32, hidden as u32)?;
        let input = moe_input(s, case.fx, rows, &case_bufs, &evicted, case.shared, y_view)?;
        case.ops.prefill(s, input, case.scratch)?;
        s.synchronize()?;
        let y = s.clone_dtoh(&case_bufs.y)?;
        let all: Vec<usize> = (0..top_k).collect();
        let (mut new_diff, mut old_diff) = (0.0f32, 0.0f32);
        for row in 0..rows {
            let ids_row = &ids[row * top_k..(row + 1) * top_k];
            let weights_row = &weights[row * top_k..(row + 1) * top_k];
            let expected_new =
                reference::moe_row_ids(&fx2, row, ids_row, weights_row, &all, Combine::Fp32);
            let expected_old =
                reference::moe_row_ids(&fx, row, ids_row, weights_row, &all, Combine::Fp32);
            for c in 0..hidden {
                let got = y[row * hidden + c].to_f32();
                new_diff = new_diff.max((got - expected_new[c].to_f32()).abs());
                old_diff = old_diff.max((got - expected_old[c].to_f32()).abs());
            }
        }
        eprintln!(
            "moe_prefill eviction-reuse rows={rows} expert={HOT_EXPERT} generation=2 new-weight max_diff={new_diff:.6} old-weight max_diff={old_diff:.6} tol={tol_evict:.6}"
        );
        if new_diff > tol_evict {
            defects.push(format!(
                "eviction-reuse new-weight {new_diff:.6} > tol {tol_evict:.6}"
            ));
        }
        anyhow::ensure!(
            old_diff > tol_evict,
            "eviction-reuse matched the old weights: new {new_diff:.6} old {old_diff:.6} tol {tol_evict:.6}"
        );
    }

    // K2 duplicate-pos_of mutant: after a clean all-active contribute, the
    // real moe_scatter is re-launched over a planted pos_of where choice 1
    // of row 0 points at choice 0's Y_perm row; the joined output must move
    // beyond the frozen tolerance (two active assignments reading one
    // Y_perm row).
    {
        let rows = 37usize;
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let active = all_active(rows, top_k);
        let gens = vec![1u64; rows * top_k];
        let tol =
            planted_noise(&fx, &ids, &weights, &active, &(0..rows).collect::<Vec<_>>()).tolerance();
        let combine_ops = CombineOps::new(&d.ctx)?;
        let ctx = GateCtx {
            s,
            ops: &ops,
            combine: &combine_ops,
            fx: &fx,
            slots: &main_slots,
            shared: &shared,
            scratch: &scratch,
        };
        let shared_out = capture_shared_out(&ctx, rows, &ids, &active, &weights, &gens)?;
        let cohort = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        let bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &gens)?;
        let input = contribute_input(s, &fx, rows, &bufs, &main_slots, &shared, &cohort)?;
        ops.prefill_contribute(s, input, &scratch)?;
        s.synchronize()?;
        let layout = prefill_layout(rows, rows, top_k, experts, hidden, inter, None)?;
        let mut pos_of = vec![0i32; rows * top_k];
        raw_dtoh(s, scratch.ptr + layout.pos_of, &mut pos_of)?;
        let planted = (pos_of[0], pos_of[1]);
        pos_of[1] = pos_of[0];
        raw_htod(s, scratch.ptr + layout.pos_of + 4, &pos_of[1..2])?;
        let module = cubin::module(&d.ctx, "moe_prefill")?;
        let scatter = module.load_function("moe_scatter")?;
        let (contributions_ptr, _gc) = cohort.contributions.device_ptr(s);
        let (weights_ptr, _gw) = bufs.weights.device_ptr(s);
        let (active_ptr, _ga) = bufs.active.device_ptr(s);
        unsafe {
            s.launch_builder(&scatter)
                .arg(&(scratch.ptr + layout.y_perm))
                .arg(&weights_ptr)
                .arg(&(scratch.ptr + layout.pos_of))
                .arg(&active_ptr)
                .arg(&contributions_ptr)
                .arg(&((rows * top_k) as i64))
                .arg(&(hidden as i64))
                .launch(grid(rows * top_k, 256))?;
        }
        s.synchronize()?;
        let bits = join_combine(
            &ctx,
            rows,
            &bufs,
            &shared_out,
            &cohort.contributions,
            &cohort.done,
        )?;
        let mut row0_diff = 0.0f32;
        let mut max_diff = 0.0f32;
        for row in 0..rows {
            let expected = reference::moe_row_ids(
                &fx,
                row,
                &ids[row * top_k..(row + 1) * top_k],
                &weights[row * top_k..(row + 1) * top_k],
                &(0..top_k).collect::<Vec<_>>(),
                Combine::Fp32,
            );
            for c in 0..hidden {
                let got = bf16::from_bits(bits[row * hidden + c]).to_f32();
                let diff = (got - expected[c].to_f32()).abs();
                if row == 0 {
                    row0_diff = row0_diff.max(diff);
                }
                max_diff = max_diff.max(diff);
            }
        }
        eprintln!(
            "moe_prefill K2 duplicate-pos_of rows={rows} planted pos_of[1]={} (was {}) row0 max_diff={row0_diff:.6} max_diff={max_diff:.6} tol={:.6}",
            planted.0, planted.1, tol
        );
        anyhow::ensure!(
            row0_diff > tol && max_diff > tol,
            "duplicate pos_of was not detected: row0 {row0_diff:.6} max {max_diff:.6} <= tol {tol:.6}"
        );
    }

    // K4 two-cohort join, two partitions of the routed rows=37 input into
    // ONE caller-owned contributions/done pair plus one wrapper-style join
    // with the union mask. The FAIL predicate is the production cohort
    // form: split by expert id parity (every expert's rows wholly in one
    // cohort). The choice-parity split is reported only (first differing
    // word if any).
    {
        let rows = 37usize;
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        let combine_ops = CombineOps::new(&d.ctx)?;
        let ctx = GateCtx {
            s,
            ops: &ops,
            combine: &combine_ops,
            fx: &fx,
            slots: &main_slots,
            shared: &shared,
            scratch: &scratch,
        };
        let shared_out =
            capture_shared_out(&ctx, rows, &ids, &all_active(rows, top_k), &weights, &gens)?;
        let single = join_bits(
            &ctx,
            rows,
            &ids,
            &weights,
            &gens,
            &shared_out,
            &[all_active(rows, top_k)],
        )?;
        let mut by_expert = (vec![0u32; rows * top_k], vec![0u32; rows * top_k]);
        for (a, &id) in ids.iter().enumerate() {
            if id % 2 == 0 {
                by_expert.0[a] = 1;
            } else {
                by_expert.1[a] = 1;
            }
        }
        let expert_split = join_bits(
            &ctx,
            rows,
            &ids,
            &weights,
            &gens,
            &shared_out,
            &[by_expert.0, by_expert.1],
        )?;
        let expert_bad = single.iter().zip(&expert_split).position(|(a, b)| a != b);
        eprintln!(
            "moe_prefill K4 two-cohort join rows={rows} cohorts=expert-parity words={} first_differing_word={:?} (FAIL predicate)",
            single.len(),
            expert_bad
        );
        anyhow::ensure!(
            expert_bad.is_none(),
            "expert-parity two-cohort join is not bitwise equal to the single cohort: first differing word {expert_bad:?} of {}",
            single.len()
        );
        let mut even = vec![0u32; rows * top_k];
        let mut odd = vec![0u32; rows * top_k];
        for a in 0..rows * top_k {
            if a % 2 == 0 {
                even[a] = 1;
            } else {
                odd[a] = 1;
            }
        }
        let split = join_bits(&ctx, rows, &ids, &weights, &gens, &shared_out, &[even, odd])?;
        let first_bad = single.iter().zip(&split).position(|(a, b)| a != b);
        eprintln!(
            "moe_prefill K4 two-cohort join rows={rows} cohorts=choice-parity words={} first_differing_word={:?} (reported result, not a FAIL predicate)",
            single.len(),
            first_bad
        );
    }

    // K5c seam cell (child-isolated so a trap cannot cascade): two
    // contribute cohorts plus a CPU-completed choice set merged on host into
    // the caller-owned buffers, then prefill_finish — bitwise equal to the
    // wrapper.
    {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "moe_prefill_gpu",
                "--ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("FF_Q38_GPU_WINDOW", "granted")
            .env("FF_MOE_CHILD", "k5c")
            .status()?;
        anyhow::ensure!(status.success(), "K5c seam child failed: {status:?}");
    }

    anyhow::ensure!(defects.is_empty(), "recorded defects: {defects:?}");
    Ok(())
}

/// True when the error chain reports the device-side trap's launch failure.
fn trap_launched(err: &anyhow::Error) -> bool {
    const MARKERS: [&str; 4] = [
        "illegal instruction",
        "launch failed",
        "unspecified launch failure",
        "assert",
    ];
    err.chain().any(|cause| {
        let text = cause.to_string().to_lowercase();
        MARKERS.iter().any(|marker| text.contains(marker))
    })
}

/// Child shared setup: fixture-sized slot tables, shared expert buffers and
/// scratch. Returns the weight buffers so their device storage outlives the
/// launch.
struct ChildSetup {
    slots: DeviceSlots,
    shared: SharedBufs,
    scratch: Scratch,
    _weights: (CudaSlice<bf16>, CudaSlice<bf16>),
    _scratch_buf: CudaSlice<u8>,
}

fn child_setup(s: &Arc<CudaStream>, fx: &MoeFixture, generation: Vec<u64>) -> Result<ChildSetup> {
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let gu_ptr = {
        let (ptr, guard) = gu.device_ptr(s);
        drop(guard);
        ptr
    };
    let dn_ptr = {
        let (ptr, guard) = dn.device_ptr(s);
        drop(guard);
        ptr
    };
    let slots = device_slots(
        s,
        (0..experts)
            .map(|e| gu_ptr + e as u64 * (2 * inter * hidden * 2) as u64)
            .collect(),
        (0..experts)
            .map(|e| dn_ptr + e as u64 * (hidden * inter * 2) as u64)
            .collect(),
        vec![1u32; experts],
        generation,
    )?;
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    let scratch_bytes = prefill_scratch_bytes(
        fx.dims.rows,
        fx.dims.rows,
        top_k,
        experts,
        hidden,
        inter,
        None,
    )?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let scratch_ptr = {
        let (ptr, guard) = scratch_buf.device_ptr(s);
        drop(guard);
        ptr
    };
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    Ok(ChildSetup {
        slots,
        shared,
        scratch,
        _weights: (gu, dn),
        _scratch_buf: scratch_buf,
    })
}

/// A small fixture whose replaced slot (generation 2) mismatches assignment
/// 0's routing-time snapshot (generation 1); the permute guard must trap.
/// Exits 3 only on the trap's launch failure; every other outcome fails the
/// test (harness exit 101).
fn stale_generation_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let fx = reference::fixture(MoeDims::new(8, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let mut table_gen = vec![1u64; experts];
    table_gen[ids[0] as usize] = 2;
    let mut snapshot = vec![1u64; rows * top_k];
    for r in 0..rows {
        for c in 0..top_k {
            snapshot[r * top_k + c] = table_gen[ids[r * top_k + c] as usize];
        }
    }
    snapshot[0] = 1;
    let setup = child_setup(s, &fx, table_gen)?;
    let active = all_active(rows, top_k);
    let case_bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &snapshot)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = moe_input(
        s,
        &fx,
        rows,
        &case_bufs,
        &setup.slots,
        &setup.shared,
        y_view,
    )?;
    match ops.prefill(s, input, &setup.scratch) {
        Ok(_event) => anyhow::bail!("stale generation did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("prefill failed without the trap: {err:#}"),
    }
}

/// Duplicate expert ids within a row's active choices must trip the permute
/// guard. Exits 3 only on the trap's launch failure; every other outcome
/// fails the test (harness exit 101).
fn duplicate_ids_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let fx = reference::fixture(MoeDims::new(2, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let mut dup = ids.clone();
    dup[1] = dup[0];
    let active = all_active(rows, top_k);
    let gens = vec![1u64; rows * top_k];
    let setup = child_setup(s, &fx, vec![1u64; experts])?;
    let case_bufs = case_bufs(s, &fx, rows, &dup, &active, &weights, &gens)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = moe_input(
        s,
        &fx,
        rows,
        &case_bufs,
        &setup.slots,
        &setup.shared,
        y_view,
    )?;
    match ops.prefill(s, input, &setup.scratch) {
        Ok(_event) => anyhow::bail!("duplicate ids did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("prefill failed without the trap: {err:#}"),
    }
}

/// pos_of[0] planted to `assignments` (out of range) before a direct
/// moe_scatter relaunch over the cohort's scratch; the scatter's position
/// guard must trap. Exits 3 only on the trap's launch failure; every other
/// outcome fails the test (harness exit 101).
fn pos_of_oob_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let fx = reference::fixture(MoeDims::new(8, 16, 4, 32, 16), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let active = all_active(rows, top_k);
    let gens = vec![1u64; rows * top_k];
    let setup = child_setup(s, &fx, vec![1u64; experts])?;
    let cohort = CohortBufs {
        contributions: s.alloc_zeros(rows * top_k * hidden)?,
        done: s.alloc_zeros(rows * top_k)?,
    };
    let bufs = case_bufs(s, &fx, rows, &ids, &active, &weights, &gens)?;
    let input = contribute_input(s, &fx, rows, &bufs, &setup.slots, &setup.shared, &cohort)?;
    ops.prefill_contribute(s, input, &setup.scratch)?;
    s.synchronize()?;
    let layout = prefill_layout(rows, rows, top_k, experts, hidden, inter, None)?;
    let planted = [(rows * top_k) as i32];
    raw_htod(s, setup.scratch.ptr + layout.pos_of, &planted)?;
    let module = cubin::module(&d.ctx, "moe_prefill")?;
    let scatter = module.load_function("moe_scatter")?;
    let (contributions_ptr, _gc) = cohort.contributions.device_ptr(s);
    let (weights_ptr, _gw) = bufs.weights.device_ptr(s);
    let (active_ptr, _ga) = bufs.active.device_ptr(s);
    let launched = unsafe {
        s.launch_builder(&scatter)
            .arg(&(setup.scratch.ptr + layout.y_perm))
            .arg(&weights_ptr)
            .arg(&(setup.scratch.ptr + layout.pos_of))
            .arg(&active_ptr)
            .arg(&contributions_ptr)
            .arg(&((rows * top_k) as i64))
            .arg(&(hidden as i64))
            .launch(grid(rows * top_k, 256))
    };
    let outcome = launched
        .and_then(|_| s.synchronize())
        .map_err(anyhow::Error::from);
    eprintln!(
        "moe_prefill K2 out-of-range pos_of rows={rows} planted pos_of[0]={} assignments={}",
        planted[0],
        rows * top_k
    );
    match outcome {
        Ok(()) => anyhow::bail!("out-of-range pos_of did not trap"),
        Err(err) if trap_launched(&err) => std::process::exit(3),
        Err(err) => anyhow::bail!("scatter relaunch failed without the trap: {err:#}"),
    }
}

// Same isolation as the gate children: the parent makes no CUDA call.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_stale_generation_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "stale")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "stale-generation child exit {:?} != 3 (the permute trap was not proven)",
        status.code()
    );
    Ok(())
}

// Parent for the duplicate-ids child; no CUDA call in the parent.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_duplicate_ids_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "dup")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "duplicate-ids child exit {:?} != 3 (the permute trap was not proven)",
        status.code()
    );
    Ok(())
}

// Parent for the out-of-range pos_of child; no CUDA call in the parent.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_pos_of_bounds_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "moe_prefill_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "oobpos")
        .status()?;
    anyhow::ensure!(
        status.code() == Some(3),
        "out-of-range pos_of child exit {:?} != 3 (the scatter trap was not proven)",
        status.code()
    );
    Ok(())
}

#[test]
fn stage_accounting_is_pinned() {
    assert_eq!(super::prefill::stage_host_pinned_bytes(), 74_084);
    assert_eq!(super::prefill::stage_device_bytes(), 65_632);
}

/// Small frozen case for the prefill racecheck: rows=64, E=16, K=4, H=128,
/// I=64 exercises count/scan/permute/GEMMs/silu/scatter/shared_scale/
/// combine once each within a sanitizer budget.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_racecheck_small_gpu() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let fx = reference::fixture(MoeDims::new(64, 16, 4, 128, 64), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = gu.device_ptr(s);
    let (dn_ptr, _b) = dn.device_ptr(s);
    let table = testing::slots(
        s,
        experts,
        gu_ptr,
        (2 * inter * hidden * 2) as u64,
        dn_ptr,
        (hidden * inter * 2) as u64,
    )?;
    let sg: CudaSlice<bf16> = s.clone_htod(fx.shared_gate.as_slice())?;
    let su: CudaSlice<bf16> = s.clone_htod(fx.shared_up.as_slice())?;
    let sd: CudaSlice<bf16> = s.clone_htod(fx.shared_down.as_slice())?;
    let seg: CudaSlice<bf16> = s.clone_htod(fx.shared_expert_gate.as_slice())?;
    let scratch_bytes = prefill_scratch_bytes(64, 64, top_k, experts, hidden, inter, None)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _c) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let active = all_active(rows, top_k);
    let gens = vec![1u64; rows * top_k];
    let x: CudaSlice<bf16> = s.clone_htod(fx.x.as_slice())?;
    let ids_buf: CudaSlice<i32> = s.clone_htod(&ids)?;
    let active_buf: CudaSlice<u32> = s.clone_htod(&active)?;
    let weights_buf: CudaSlice<bf16> = s.clone_htod(&weights)?;
    let gens_buf: CudaSlice<u64> = s.clone_htod(&gens)?;
    let contributions_buf: CudaSlice<bf16> = s.alloc_zeros(rows * top_k * hidden)?;
    let done_buf: CudaSlice<u32> = s.alloc_zeros(rows * top_k)?;
    let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
    let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
    let input = MoePrefill {
        expert_inter: inter as u32,
        x: testing::dense(&x, s, rows as u32, hidden as u32)?,
        ids: testing::dense(&ids_buf, s, rows as u32, top_k as u32)?,
        active: testing::dense(&active_buf, s, rows as u32, top_k as u32)?,
        weights: testing::dense(&weights_buf, s, rows as u32, top_k as u32)?,
        assignment_gen: testing::dense(&gens_buf, s, rows as u32, top_k as u32)?,
        slots_gate_up: testing::view(&table.gate_up, s, [experts as u32], [1])?,
        slots_down: testing::view(&table.down, s, [experts as u32], [1])?,
        slot_valid: testing::view(&table.valid, s, [experts as u32], [1])?,
        slot_generation: testing::view(&table.generation, s, [experts as u32], [1])?,
        slots_gate_up_host: &gu_host_vecs(gu_ptr, experts, 2 * inter * hidden * 2).0,
        slots_down_host: &gu_host_vecs(dn_ptr, experts, hidden * inter * 2).1,
        shared_gate: testing::dense(&sg, s, inter as u32, hidden as u32)?,
        shared_up: testing::dense(&su, s, inter as u32, hidden as u32)?,
        shared_down: testing::dense(&sd, s, hidden as u32, inter as u32)?,
        shared_expert_gate: Some(testing::view(&seg, s, [hidden as u32], [1])?),
        contributions: testing::view(
            &contributions_buf,
            s,
            [rows as u32, top_k as u32, hidden as u32],
            [(top_k * hidden) as u64, hidden as u64, 1],
        )?,
        done: testing::mut_view(&done_buf, s, [rows as u32, top_k as u32], [top_k as u64, 1])?,
        y: y_view,
    };
    ops.prefill(s, input, &scratch)?;
    s.synchronize()?;
    Ok(())
}

fn gu_host_vecs(base: u64, experts: usize, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let v: Vec<u64> = (0..experts as u64)
        .map(|e| base + e * stride as u64)
        .collect();
    (v.clone(), v)
}

// K5c mutant: a finish whose done bits were reset must fail the
// exactly-once combine check. Child runs contribute, zeroes one done bit,
// then prefill_finish; parent requires exit 3.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_reset_done_gpu() -> Result<()> {
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    if std::env::var("FF_MOE_CHILD").as_deref() == Ok("resetdone") {
        let d = crate::backend::cuda::Device::new(0)?;
        let s = &d.stream;
        let fx = reference::fixture(MoeDims::minicpm47(8), 0);
        let (experts, top_k, hidden, inter) = (
            fx.dims.experts,
            fx.dims.top_k,
            fx.dims.hidden,
            fx.dims.inter,
        );
        let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
        let rows = fx.dims.rows;
        let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
        let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
        let (gu_ptr, _a) = gu.device_ptr(s);
        let (dn_ptr, _b) = dn.device_ptr(s);
        let gu_host: Vec<u64> = (0..experts as u64)
            .map(|e| gu_ptr + e * (2 * inter * hidden * 2) as u64)
            .collect();
        let dn_host: Vec<u64> = (0..experts as u64)
            .map(|e| dn_ptr + e * (hidden * inter * 2) as u64)
            .collect();
        let slots = device_slots(
            s,
            gu_host,
            dn_host,
            vec![1u32; experts],
            vec![1u64; experts],
        )?;
        let scratch_bytes = prefill_scratch_bytes(rows, rows, top_k, experts, hidden, inter, None)?;
        let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
        let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
        let scratch = Scratch {
            ptr: scratch_ptr,
            bytes: scratch_bytes,
        };
        let shared = SharedBufs {
            sg: s.clone_htod(fx.shared_gate.as_slice())?,
            su: s.clone_htod(fx.shared_up.as_slice())?,
            sd: s.clone_htod(fx.shared_down.as_slice())?,
            seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
        };
        let (ids, weights) = planted_uniform(rows, experts, top_k);
        let gens = vec![1u64; rows * top_k];
        let mut cohort = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        let bufs = case_bufs(
            s,
            &fx,
            rows,
            &ids,
            &all_active(rows, top_k),
            &weights,
            &gens,
        )?;
        let input = contribute_input(s, &fx, rows, &bufs, &slots, &shared, &cohort)?;
        ops.prefill_contribute(s, input, &scratch)?;
        s.synchronize()?;
        let mut host_d: Vec<u32> = s.clone_dtoh(&cohort.done)?;
        host_d[0] = 0;
        s.memcpy_htod(&host_d, &mut cohort.done)?;
        let y: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
        let union = case_bufs(
            s,
            &fx,
            rows,
            &ids,
            &all_active(rows, top_k),
            &weights,
            &gens,
        )?;
        let fin = MoePrefillFinish {
            expert_inter: fx.dims.inter as u32,
            x: testing::dense(&union.x, s, rows as u32, hidden as u32)?,
            ids: testing::dense(&union.ids, s, rows as u32, top_k as u32)?,
            final_active: testing::dense(&union.active, s, rows as u32, top_k as u32)?,
            contributions: testing::view(
                &cohort.contributions,
                s,
                [rows as u32, top_k as u32, hidden as u32],
                [(top_k * hidden) as u64, hidden as u64, 1],
            )?,
            done: testing::view(
                &cohort.done,
                s,
                [rows as u32, top_k as u32],
                [top_k as u64, 1],
            )?,
            experts: experts as u32,
            shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
            shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
            shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
            shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
            y: testing::dense_mut(&y, s, rows as u32, hidden as u32)?,
        };
        let outcome = ops
            .prefill_finish(s, fin, &scratch)
            .and_then(|_| s.synchronize().map_err(anyhow::Error::from));
        eprintln!("moe_prefill reset-done child outcome: {outcome:?}");
        let trapped = match &outcome {
            Err(err) => {
                let text = format!("{err:#}");
                text.contains("illegal")
                    || text.contains("assert")
                    || text.contains("unspecified launch failure")
                    || text.contains("launch failed")
            }
            Ok(_) => false,
        };
        if trapped {
            std::process::exit(3);
        }
        anyhow::bail!("reset done did not fail the combine discriminantly: {outcome:?}")
    }
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "prefill_reset_done_gpu",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FF_Q38_GPU_WINDOW", "granted")
        .env("FF_MOE_CHILD", "resetdone")
        .status()?;
    anyhow::ensure!(!status.success(), "reset-done child exited successfully");
    anyhow::ensure!(
        status.code() == Some(3),
        "reset-done child must discriminate (exit 3), got {status:?}"
    );
    eprintln!(
        "moe_prefill reset-done child exit={:?} DETECTED",
        status.code()
    );
    Ok(())
}

/// K5c seam cell body, run via FF_MOE_CHILD=k5c.
fn k5c_seam_child() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let fx = reference::fixture(MoeDims::minicpm47(37), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let main_gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let main_dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = main_gu.device_ptr(s);
    let (dn_ptr, _b) = main_dn.device_ptr(s);
    let main_slots = device_slots(
        s,
        (0..experts as u64)
            .map(|e| gu_ptr + e * (2 * inter * hidden * 2) as u64)
            .collect(),
        (0..experts as u64)
            .map(|e| dn_ptr + e * (hidden * inter * 2) as u64)
            .collect(),
        vec![1u32; experts],
        vec![1u64; experts],
    )?;
    let scratch_bytes = prefill_scratch_bytes(37, 37, top_k, experts, hidden, inter, None)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    // Two contribute cohorts plus a CPU-completed choice set merged on host
    // into the same caller-owned buffers, then prefill_finish ==
    // the single wrapper run, bitwise.
    {
        let rows = 37usize;
        let mut ids = vec![0i32; rows * top_k];
        let mut weights = vec![bf16::ZERO; rows * top_k];
        for r in 0..rows {
            let routed = reference::route_row(&fx, r);
            ids[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.ids);
            weights[r * top_k..(r + 1) * top_k].copy_from_slice(&routed.weights);
        }
        let gens = vec![1u64; rows * top_k];
        let y_wrap: CudaSlice<bf16> = {
            let bufs = case_bufs(
                s,
                &fx,
                rows,
                &ids,
                &all_active(rows, top_k),
                &weights,
                &gens,
            )?;
            let y = s.alloc_zeros(rows * hidden)?;
            let y_view = testing::dense_mut(&y, s, rows as u32, hidden as u32)?;
            let input = moe_input(s, &fx, rows, &bufs, &main_slots, &shared, y_view)?;
            ops.prefill(s, input, &scratch)?;
            s.synchronize()?;
            y
        };
        let mask = |pred: &dyn Fn(usize) -> bool| -> Vec<u32> {
            (0..rows * top_k)
                .map(|a| if pred(a) { 1 } else { 0 })
                .collect()
        };
        let cpu_set = mask(&|a| a % 3 == 0);
        let even = mask(&|a| a % 3 != 0 && ids[a] % 2 == 0);
        let odd = mask(&|a| a % 3 != 0 && ids[a] % 2 == 1);
        let mut cohort = CohortBufs {
            contributions: s.alloc_zeros(rows * top_k * hidden)?,
            done: s.alloc_zeros(rows * top_k)?,
        };
        for active in [&even, &odd] {
            let bufs = case_bufs(s, &fx, rows, &ids, active, &weights, &gens)?;
            let input = contribute_input(s, &fx, rows, &bufs, &main_slots, &shared, &cohort)?;
            ops.prefill_contribute(s, input, &scratch)?;
            s.synchronize()?;
        }
        // CPU completion stand-in: a third contribute restricted to the CPU
        // set, its bits round-tripped through host memory into the caller
        // buffers.
        {
            let tmp = CohortBufs {
                contributions: s.alloc_zeros(rows * top_k * hidden)?,
                done: s.alloc_zeros(rows * top_k)?,
            };
            let bufs = case_bufs(s, &fx, rows, &ids, &cpu_set, &weights, &gens)?;
            let input = contribute_input(s, &fx, rows, &bufs, &main_slots, &shared, &tmp)?;
            ops.prefill_contribute(s, input, &scratch)?;
            s.synchronize()?;
            let tmp_c: Vec<bf16> = s.clone_dtoh(&tmp.contributions)?;
            let tmp_d: Vec<u32> = s.clone_dtoh(&tmp.done)?;
            let mut host_c: Vec<bf16> = s.clone_dtoh(&cohort.contributions)?;
            let mut host_d: Vec<u32> = s.clone_dtoh(&cohort.done)?;
            for (a, &m) in cpu_set.iter().enumerate() {
                if m == 1 {
                    host_d[a] = tmp_d[a];
                    host_c[a * hidden..(a + 1) * hidden]
                        .copy_from_slice(&tmp_c[a * hidden..(a + 1) * hidden]);
                }
            }
            s.memcpy_htod(&host_c, &mut cohort.contributions)?;
            s.memcpy_htod(&host_d, &mut cohort.done)?;
            s.synchronize()?;
        }
        let y_fin: CudaSlice<bf16> = s.alloc_zeros(rows * hidden)?;
        let fin = {
            let union = case_bufs(
                s,
                &fx,
                rows,
                &ids,
                &all_active(rows, top_k),
                &weights,
                &gens,
            )?;
            let y_view = testing::dense_mut(&y_fin, s, rows as u32, hidden as u32)?;
            MoePrefillFinish {
                expert_inter: fx.dims.inter as u32,
                x: testing::dense(&union.x, s, rows as u32, hidden as u32)?,
                ids: testing::dense(&union.ids, s, rows as u32, top_k as u32)?,
                final_active: testing::dense(&union.active, s, rows as u32, top_k as u32)?,
                contributions: testing::view(
                    &cohort.contributions,
                    s,
                    [rows as u32, top_k as u32, hidden as u32],
                    [(top_k * hidden) as u64, hidden as u64, 1],
                )?,
                done: testing::view(
                    &cohort.done,
                    s,
                    [rows as u32, top_k as u32],
                    [top_k as u64, 1],
                )?,
                experts: fx.dims.experts as u32,
                shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
                shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
                shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
                shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
                y: y_view,
            }
        };
        ops.prefill_finish(s, fin, &scratch)?;
        s.synchronize()?;
        let wrap_bits: Vec<u16> = s.clone_dtoh(&y_wrap)?.iter().map(|v| v.to_bits()).collect();
        let fin_bits: Vec<u16> = s.clone_dtoh(&y_fin)?.iter().map(|v| v.to_bits()).collect();
        let first_bad = wrap_bits.iter().zip(&fin_bits).position(|(a, b)| a != b);
        eprintln!(
            "moe_prefill K5c seam rows={rows} cohorts=2+cpu words={} first_differing_word={:?}",
            wrap_bits.len(),
            first_bad
        );
        anyhow::ensure!(
            first_bad.is_none(),
            "K5c seam join is not bitwise equal to the wrapper: first differing word {first_bad:?} of {}",
            wrap_bits.len()
        );
    }
    Ok(())
}

#[test]
fn k5c_partition_is_disjoint_and_covers_all() {
    let top_k = 8usize;
    let rows = 37usize;
    let ids: Vec<i32> = (0..rows * top_k)
        .map(|a| ((a * 7 + 3) % 251) as i32)
        .collect();
    let cpu: Vec<bool> = (0..rows * top_k).map(|a| a % 3 == 0).collect();
    let even: Vec<bool> = (0..rows * top_k)
        .map(|a| a % 3 != 0 && ids[a] % 2 == 0)
        .collect();
    let odd: Vec<bool> = (0..rows * top_k)
        .map(|a| a % 3 != 0 && ids[a] % 2 == 1)
        .collect();
    for a in 0..rows * top_k {
        let n = cpu[a] as u8 + even[a] as u8 + odd[a] as u8;
        assert_eq!(n, 1, "assignment {a} covered {n} times");
    }
    // The host merge writes exactly the cpu rows: done word a and
    // contribution words [a*hidden, (a+1)*hidden).
    let hidden = 2048usize;
    let mut done = vec![0u32; rows * top_k];
    let mut contributions = vec![bf16::ZERO; rows * top_k * hidden];
    let tmp_d = vec![1u32; rows * top_k];
    let tmp_c = vec![bf16::ONE; rows * top_k * hidden];
    for (a, &m) in cpu.iter().enumerate() {
        if m {
            done[a] = tmp_d[a];
            contributions[a * hidden..(a + 1) * hidden]
                .copy_from_slice(&tmp_c[a * hidden..(a + 1) * hidden]);
        }
    }
    assert_eq!(
        done.iter().filter(|&&v| v == 1).count(),
        cpu.iter().filter(|&v| *v).count()
    );
    assert!(
        contributions
            .chunks(hidden)
            .enumerate()
            .all(|(a, c)| (c[0] == bf16::ONE) == cpu[a])
    );
}

/// H1 discriminator: three prefill calls queued back-to-back with no host
/// sync between them (the stage mirror is rewritten while earlier copies may
/// still be queued); the third output must equal three independent runs.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_stage_backtoback_gpu() -> Result<()> {
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let fx = reference::fixture(MoeDims::minicpm47(8), 0);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    let ops = PrefillOps::new(&d.ctx, s, crate::model_spec::ExpertFormat::Bf16)?;
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = gu.device_ptr(s);
    let (dn_ptr, _b) = dn.device_ptr(s);
    let slots = device_slots(
        s,
        (0..experts as u64)
            .map(|e| gu_ptr + e * (2 * inter * hidden * 2) as u64)
            .collect(),
        (0..experts as u64)
            .map(|e| dn_ptr + e * (hidden * inter * 2) as u64)
            .collect(),
        vec![1u32; experts],
        vec![1u64; experts],
    )?;
    let scratch_bytes = prefill_scratch_bytes(
        fx.dims.rows,
        fx.dims.rows,
        top_k,
        experts,
        hidden,
        inter,
        None,
    )?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    let rows = fx.dims.rows;
    let (ids, weights) = planted_uniform(rows, experts, top_k);
    let gens = vec![1u64; rows * top_k];
    let run_once = |y: &CudaSlice<bf16>| -> Result<()> {
        let bufs = case_bufs(
            s,
            &fx,
            rows,
            &ids,
            &all_active(rows, top_k),
            &weights,
            &gens,
        )?;
        let y_view = testing::dense_mut(y, s, rows as u32, hidden as u32)?;
        let input = moe_input(s, &fx, rows, &bufs, &slots, &shared, y_view)?;
        ops.prefill(s, input, &scratch)?;
        Ok(())
    };
    let mut queued: Vec<CudaSlice<bf16>> = Vec::new();
    for _ in 0..3 {
        let y = s.alloc_zeros(rows * hidden)?;
        run_once(&y)?;
        queued.push(y);
    }
    s.synchronize()?;
    let mut all_equal = true;
    for y in &queued {
        let a: Vec<u16> = s.clone_dtoh(y)?.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u16> = s
            .clone_dtoh(&queued[0])?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        if a != b {
            all_equal = false;
        }
    }
    anyhow::ensure!(
        all_equal,
        "back-to-back prefill outputs diverged: the stage mirror was rewritten before a queued copy completed"
    );
    eprintln!("moe_prefill H1 back-to-back 3x outputs bitwise-identical PASS");
    Ok(())
}

/// Discriminating GPU test for T25.2 Step 2A:
/// Runs one span (rows=769, coprime with 256 and 8) with forced 256-row waves
/// and with packed waves from SpanPlan::pack_waves, with coprime activations
/// and experts at both <= 16 rows (lane16) and > 16 rows (grouped GEMM).
/// Asserts 0 differing bits in contributions, then verifies that an off-by-one
/// wave base shift (and a host/device count mismatch) fail discriminantly.
#[test]
#[ignore = "requires GPU window, FF_Q38_GPU_WINDOW=granted"]
fn prefill_packed_waves_gpu() -> Result<()> {
    use crate::backend::cuda::contracts::view;
    use crate::experts::{
        catalogue::{ExpertGeometry, ExpertLayout},
        plan::{ServiceCell, ServiceTable},
        span::{SpanPlan, SpanRates},
    };
    use crate::model_spec::ExpertFormat;
    anyhow::ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let d = crate::backend::cuda::Device::new(0)?;
    let s = &d.stream;
    let rows = 769usize;
    let wave_rows = 256usize;
    let mut fx = reference::fixture(MoeDims::minicpm47(rows), 19);
    let (experts, top_k, hidden, inter) = (
        fx.dims.experts,
        fx.dims.top_k,
        fx.dims.hidden,
        fx.dims.inter,
    );
    // Confirm consecutive rows of fx.x carry distinct signal.
    for r in 0..rows - 1 {
        anyhow::ensure!(
            fx.x[r * hidden..(r + 1) * hidden] != fx.x[(r + 1) * hidden..(r + 2) * hidden],
            "fixture rows {r} and {} are identical",
            r + 1
        );
    }
    let geometry = ExpertGeometry {
        first_layer: 0,
        layers: 1,
        experts: experts as u32,
        hidden: hidden as u32,
        intermediate: inter as u32,
        top_k: top_k as u32,
    };
    let slot_bytes = ExpertLayout::new(ExpertFormat::Bf16, &geometry)?.slot_bytes();
    let native_layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
    let fp8_slot = native_layout
        .fp8_slot()
        .ok_or_else(|| anyhow::anyhow!("native FP8 descriptor missing"))?;
    let mut native = vec![0u8; native_layout.slot_bytes() as usize * experts];
    let mut k_swapped_native = vec![0u8; native_layout.slot_bytes() as usize * experts];
    for (op_idx, (op, n, k, code_range, scale_range)) in [
        ("gate", inter, hidden, &fp8_slot.gate, &fp8_slot.gate_scale),
        ("up", inter, hidden, &fp8_slot.up, &fp8_slot.up_scale),
        ("down", hidden, inter, &fp8_slot.down, &fp8_slot.down_scale),
    ]
    .into_iter()
    .enumerate()
    {
        for e in 0..20 {
            let base = e * native_layout.slot_bytes() as usize;
            for row in 0..n {
                let scale =
                    bf16::from_f32(0.015625 * (1 + ((e * 5 + op_idx * 3 + row * 11) % 13)) as f32);
                let scale_at = base + scale_range.start as usize + row * 2;
                native[scale_at..scale_at + 2].copy_from_slice(&scale.to_le_bytes());
                k_swapped_native[scale_at..scale_at + 2].copy_from_slice(&scale.to_le_bytes());
                for col in 0..k {
                    let half = (col >> 5) & 1;
                    let mag =
                        1u8 + ((e * 17 + op_idx * 13 + row * 7 + col * 11 + half * 23) % 72) as u8;
                    let sign = if (e + row + col + half) % 3 == 0 {
                        0x80u8
                    } else {
                        0u8
                    };
                    let code = sign | mag;
                    native[base + code_range.start as usize + row * k + col] = code;
                    let value = crate::fp8::effective_weight(code, scale);
                    if op == "down" {
                        fx.down[(e * hidden + row) * inter + col] = value;
                    } else {
                        let offset = usize::from(op == "up") * inter;
                        fx.gate_up[(e * 2 * inter + offset + row) * hidden + col] = value;
                    }
                }
                for col in 0..k {
                    k_swapped_native[base + code_range.start as usize + row * k + col] =
                        native[base + code_range.start as usize + row * k + (col ^ 32)];
                }
            }
        }
    }
    anyhow::ensure!(
        k_swapped_native != native,
        "stage k-half swap control must perturb FP8 weight codes"
    );

    let mut cells = Vec::new();
    for workers in [1u32, 2] {
        for r in [1u32, 16, 256, 1024] {
            for exp in [1u32, 2] {
                cells.push(ServiceCell {
                    workers,
                    rows: r,
                    experts: exp,
                    seconds: f64::from(r) * 0.01 * f64::from(exp) / f64::from(workers),
                });
            }
        }
    }
    let service = ServiceTable::new("packed_wave_test", cells)?;
    let rates = SpanRates {
        cpu_row_limit: 16,
        service: &service,
        h2d_bytes_per_second: 1e10,
    };

    // Routing construction across 769 rows * 8 choices = 6152 assignments:
    // - Resident experts (0..15), 4 choices per row (choices 0..4):
    //   - Wave 0 covers rows 0..512 (512 * 4 = 2048 active assignments):
    //     experts 10 and 11 get 11 rows each (<= 16 -> lane16); experts 9, 12, 13, 14 take the rest.
    //   - Wave 1 covers rows 512..769 (257 * 4 = 1028 active assignments):
    //     experts 0..=8 get exact per-expert acts [256, 65, 64, 33, 32, 31, 17, 16, 1] on moe_gemm_w8a16.
    // - Staged experts (16..20, 2 cohorts of 2 experts), 4 choices per row (choices 4..8).
    let mut ids = vec![0i32; rows * top_k];
    let mut weights = vec![bf16::ZERO; rows * top_k];
    let mut counts = vec![0u32; experts];
    const WAVE0_RESIDENT: [i32; 4] = [9, 12, 13, 14];
    for r in 0..rows {
        for c in 0..top_k {
            let expert = if c < 4 {
                if r < 512 {
                    if r < 11 && c == 0 {
                        10
                    } else if r < 11 && c == 1 {
                        11
                    } else {
                        WAVE0_RESIDENT[(r + c) & 3]
                    }
                } else {
                    let u = r - 512;
                    match c {
                        0 => {
                            if u < 256 {
                                0
                            } else {
                                8
                            }
                        }
                        1 => {
                            if u < 65 {
                                1
                            } else if u < 129 {
                                2
                            } else if u < 162 {
                                3
                            } else if u < 194 {
                                4
                            } else if u < 225 {
                                5
                            } else if u < 242 {
                                6
                            } else {
                                7
                            }
                        }
                        2 => {
                            if u == 0 {
                                7
                            } else {
                                9
                            }
                        }
                        _ => 12,
                    }
                }
            } else {
                16 + ((r * 5 + (c - 4)) % 4) as i32
            };
            ids[r * top_k + c] = expert;
            counts[expert as usize] += 1;
            weights[r * top_k + c] =
                bf16::from_f32(0.0625 + 0.015625 * ((r * 13 + c * 7) % 11) as f32);
        }
    }
    anyhow::ensure!(
        counts[0..=8] == [256, 65, 64, 33, 32, 31, 17, 16, 1]
            && counts[10] == 11
            && counts[11] == 11
            && counts[16] > 16,
        "expected boundary expert counts [256,65,64,33,32,31,17,16,1] and lane16=11: counts={:?}",
        &counts[..20]
    );
    let mut resident = vec![false; experts];
    for flag in &mut resident[..15] {
        *flag = true;
    }
    let mut plan = SpanPlan::new(&geometry, slot_bytes, 2, 2)?;
    plan.update(&counts, &resident, rows as u32, wave_rows as u32, &rates)?;
    plan.pack_waves(&counts, &ids)?;
    anyhow::ensure!(
        plan.cohort_waves(0)? == [(0, 512, 2048), (512, 257, 1028)]
            && plan.cohort_waves(1)?.len() == 1,
        "expected cohort 0 waves [(0,512,2048),(512,257,1028)] and cohort 1 len 1: c0={:?} c1={:?}",
        plan.cohort_waves(0)?,
        plan.cohort_waves(1)?
    );

    let mut cpu_arithmetic = vec![false; experts];
    cpu_arithmetic[10] = true;
    cpu_arithmetic[11] = true;
    anyhow::ensure!(
        cpu_arithmetic[10] && !cpu_arithmetic[0] && !cpu_arithmetic[8] && !cpu_arithmetic[16],
        "expected both lane16 (true) and tensor-core GEMM (false) in cpu_arithmetic"
    );

    let ops = PrefillOps::new(&d.ctx, s, ExpertFormat::Bf16)?;
    let fp8_ops = PrefillOps::new(&d.ctx, s, ExpertFormat::Fp8W8a16)?;
    let total_cohorts = 1 + plan.cohort_count() as usize;
    let cohort_sets: Vec<Vec<u32>> = (0..total_cohorts)
        .map(|cohort| {
            if cohort == 0 {
                plan.resident().to_vec()
            } else {
                plan.cohorts().nth(cohort - 1).unwrap().to_vec()
            }
        })
        .collect();
    let multi_cohort_slots = |gu_addrs: &[u64], dn_addrs: &[u64]| -> Result<DeviceSlots> {
        let mut gu_all = vec![0u64; total_cohorts * experts];
        let mut dn_all = vec![0u64; total_cohorts * experts];
        let valid_all = vec![1u32; total_cohorts * experts];
        let gen_all = vec![1u64; total_cohorts * experts];
        for (cohort, set) in cohort_sets.iter().enumerate() {
            for e in 0..experts {
                let idx = cohort * experts + e;
                let src = if set.contains(&(e as u32)) {
                    e
                } else {
                    (e + cohort + 1) % experts
                };
                gu_all[idx] = gu_addrs[src];
                dn_all[idx] = dn_addrs[src];
            }
        }
        device_slots(s, gu_all, dn_all, valid_all, gen_all)
    };
    let gu: CudaSlice<bf16> = s.clone_htod(fx.gate_up.as_slice())?;
    let dn: CudaSlice<bf16> = s.clone_htod(fx.down.as_slice())?;
    let (gu_ptr, _a) = gu.device_ptr(s);
    let (dn_ptr, _b) = dn.device_ptr(s);
    let bf16_gu_addrs: Vec<u64> = (0..experts as u64)
        .map(|e| gu_ptr + e * (2 * inter * hidden * 2) as u64)
        .collect();
    let bf16_dn_addrs: Vec<u64> = (0..experts as u64)
        .map(|e| dn_ptr + e * (hidden * inter * 2) as u64)
        .collect();
    let slots = multi_cohort_slots(&bf16_gu_addrs, &bf16_dn_addrs)?;
    let native_buf: CudaSlice<u8> = s.clone_htod(&native)?;
    let (native_base, _nb) = native_buf.device_ptr(s);
    let native_addrs: Vec<u64> = (0..experts as u64)
        .map(|e| native_base + e * native_layout.slot_bytes())
        .collect();
    let fp8_slots = multi_cohort_slots(&native_addrs, &native_addrs)?;
    let k_swapped_buf: CudaSlice<u8> = s.clone_htod(&k_swapped_native)?;
    let (k_swapped_base, _ksb) = k_swapped_buf.device_ptr(s);
    let k_swapped_addrs: Vec<u64> = (0..experts as u64)
        .map(|e| k_swapped_base + e * native_layout.slot_bytes())
        .collect();
    let k_swapped_slots = multi_cohort_slots(&k_swapped_addrs, &k_swapped_addrs)?;
    let shared = SharedBufs {
        sg: s.clone_htod(fx.shared_gate.as_slice())?,
        su: s.clone_htod(fx.shared_up.as_slice())?,
        sd: s.clone_htod(fx.shared_down.as_slice())?,
        seg: s.clone_htod(fx.shared_expert_gate.as_slice())?,
    };
    // Split layout: rows = wave_rows (256), contribute_rows = rows (769).
    let scratch_bytes =
        prefill_scratch_bytes(wave_rows, rows, top_k, experts, hidden, inter, None)?;
    let scratch_buf: CudaSlice<u8> = s.alloc_zeros(scratch_bytes as usize)?;
    let (scratch_ptr, _gs) = scratch_buf.device_ptr(s);
    let scratch = Scratch {
        ptr: scratch_ptr,
        bytes: scratch_bytes,
    };

    let x_dev: CudaSlice<bf16> = s.clone_htod(&fx.x[..rows * hidden])?;
    let ids_dev: CudaSlice<i32> = s.clone_htod(&ids)?;
    let weights_dev: CudaSlice<bf16> = s.clone_htod(&weights)?;
    let gens_dev: CudaSlice<u64> = s.clone_htod(&vec![1u64; rows * top_k])?;
    let cohort_masks: Vec<CudaSlice<u32>> = cohort_sets
        .iter()
        .map(|set| {
            let mask: Vec<u32> = ids
                .iter()
                .map(|&e| u32::from(set.contains(&(e as u32))))
                .collect();
            Ok(s.clone_htod(&mask)?)
        })
        .collect::<Result<_>>()?;
    let copy_event = d.ctx.new_event(None)?;
    let reader_event = d.ctx.new_event(None)?;

    let run_waves = |run_ops: &PrefillOps,
                     run_slots: &DeviceSlots,
                     run_fp8_slot: Option<&crate::experts::catalogue::Fp8Slot>,
                     packed: bool,
                     shift_first_packed_wave: bool,
                     corrupt_cohort_offset: bool|
     -> Result<(Vec<u16>, Vec<u32>)> {
        let contributions: CudaSlice<bf16> = s.alloc_zeros(rows * top_k * hidden)?;
        let done: CudaSlice<u32> = s.alloc_zeros(rows * top_k)?;
        for (cohort, active_dev) in cohort_masks.iter().enumerate() {
            if cohort > 0 {
                copy_event.record(s)?;
            }
            let table_offset = if corrupt_cohort_offset && cohort == 1 {
                cohort * experts + experts
            } else {
                cohort * experts
            };
            let waves: Vec<(u32, u32, Option<u32>)> = if packed {
                plan.cohort_waves(cohort)?
                    .iter()
                    .map(|&(base, w_rows, active)| (base, w_rows, Some(active)))
                    .collect()
            } else {
                (0..rows as u32)
                    .step_by(wave_rows)
                    .map(|base| (base, (wave_rows as u32).min(rows as u32 - base), None))
                    .collect()
            };
            for (w_idx, (base, w_rows, wave_active)) in waves.into_iter().enumerate() {
                let x_base = if shift_first_packed_wave && cohort == 0 && w_idx == 1 {
                    base - 1
                } else {
                    base
                };
                let b = base as usize;
                run_ops.prefill_contribute(
                    s,
                    MoePrefillContribute {
                        expert_inter: inter as u32,
                        cpu_arithmetic: &cpu_arithmetic,
                        fp8_slot: run_fp8_slot,
                        wave: wave_active.map(|a| (cohort, base, a)),
                        wait_copy: (cohort > 0 && w_idx == 0).then_some(&copy_event),
                        reader_event: Some(&reader_event),
                        x: view(
                            &x_dev,
                            s,
                            x_base as usize * hidden,
                            [w_rows, hidden as u32],
                            [hidden as u64, 1],
                        )?,
                        ids: view(
                            &ids_dev,
                            s,
                            b * top_k,
                            [w_rows, top_k as u32],
                            [top_k as u64, 1],
                        )?,
                        weights: view(
                            &weights_dev,
                            s,
                            b * top_k,
                            [w_rows, top_k as u32],
                            [top_k as u64, 1],
                        )?,
                        active: view(
                            active_dev,
                            s,
                            b * top_k,
                            [w_rows, top_k as u32],
                            [top_k as u64, 1],
                        )?,
                        assignment_gen: view(
                            &gens_dev,
                            s,
                            b * top_k,
                            [w_rows, top_k as u32],
                            [top_k as u64, 1],
                        )?,
                        slots_gate_up: view(
                            &run_slots.gu_dev,
                            s,
                            table_offset,
                            [experts as u32],
                            [1],
                        )?,
                        slots_down: view(
                            &run_slots.dn_dev,
                            s,
                            table_offset,
                            [experts as u32],
                            [1],
                        )?,
                        slot_valid: view(&run_slots.valid, s, table_offset, [experts as u32], [1])?,
                        slot_generation: view(
                            &run_slots.generation,
                            s,
                            table_offset,
                            [experts as u32],
                            [1],
                        )?,
                        slots_gate_up_host: &run_slots.gu_host
                            [table_offset..table_offset + experts],
                        slots_down_host: &run_slots.dn_host[table_offset..table_offset + experts],
                        shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
                        shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
                        shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
                        shared_expert_gate: Some(testing::view(
                            &shared.seg,
                            s,
                            [hidden as u32],
                            [1],
                        )?),
                        contributions: view(
                            &contributions,
                            s,
                            b * top_k * hidden,
                            [w_rows, top_k as u32, hidden as u32],
                            [(top_k * hidden) as u64, hidden as u64, 1],
                        )?,
                        done: TensorMut(view(
                            &done,
                            s,
                            b * top_k,
                            [w_rows, top_k as u32],
                            [top_k as u64, 1],
                        )?),
                    },
                    &scratch,
                )?;
            }
        }
        s.synchronize()?;
        let bits = s
            .clone_dtoh(&contributions)?
            .into_iter()
            .map(|v| v.to_bits())
            .collect();
        let done_host = s.clone_dtoh(&done)?;
        Ok((bits, done_host))
    };

    let (fixed_bits, fixed_done) = run_waves(&ops, &slots, None, false, false, false)?;
    let (packed_bits, packed_done) = run_waves(&ops, &slots, None, true, false, false)?;
    anyhow::ensure!(
        fixed_done.iter().all(|&d| d == 1) && packed_done.iter().all(|&d| d == 1),
        "incomplete done bits in fixed or packed wave run"
    );
    let diff_bits = fixed_bits
        .iter()
        .zip(&packed_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "prefill_packed_waves_gpu bf16 rows={rows} cohorts={total_cohorts} c0_waves={} c1_waves={} diff_bits={diff_bits}",
        plan.cohort_waves(0)?.len(),
        plan.cohort_waves(1)?.len()
    );
    anyhow::ensure!(
        diff_bits == 0,
        "packed waves diverged from forced 256-row waves: diff_bits={diff_bits}"
    );

    let (fp8_fixed_bits, fp8_fixed_done) =
        run_waves(&fp8_ops, &fp8_slots, Some(fp8_slot), false, false, false)?;
    let (fp8_packed_bits, fp8_packed_done) =
        run_waves(&fp8_ops, &fp8_slots, Some(fp8_slot), true, false, false)?;
    anyhow::ensure!(
        fp8_fixed_done.iter().all(|&d| d == 1) && fp8_packed_done.iter().all(|&d| d == 1),
        "incomplete done bits in FP8 fixed or packed wave run"
    );
    let fp8_wave_diff = fp8_fixed_bits
        .iter()
        .zip(&fp8_packed_bits)
        .filter(|(a, b)| a != b)
        .count();
    let fp8_vs_bf16_diff = fp8_packed_bits
        .iter()
        .zip(&packed_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "prefill_packed_waves_gpu fp8 rows={rows} acts={:?} fp8_wave_diff_bits={fp8_wave_diff} fp8_vs_bf16_diff_bits={fp8_vs_bf16_diff}",
        &counts[0..=8]
    );
    anyhow::ensure!(
        fp8_wave_diff == 0 && fp8_vs_bf16_diff == 0,
        "FP8 packed waves diverged: fp8_wave_diff={fp8_wave_diff} fp8_vs_bf16_diff={fp8_vs_bf16_diff}"
    );

    // Control 0 (Step 2B): swap the two k-halves (col ^ 32) of each 64-K stage in FP8 codes; must FAIL (>0 differing bits).
    let (k_swapped_bits, _) = run_waves(
        &fp8_ops,
        &k_swapped_slots,
        Some(fp8_slot),
        true,
        false,
        false,
    )?;
    let k_swap_diff = fp8_packed_bits
        .iter()
        .zip(&k_swapped_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "prefill_packed_waves_gpu fp8 stage k-half swap control k_swap_diff_bits={k_swap_diff} DETECTED"
    );
    anyhow::ensure!(
        k_swap_diff > 0,
        "FP8 stage k-half swap control was not detected"
    );

    // Control 1: plant an off-by-one wave base (shifted by 1 row); must FAIL (>0 differing bits).
    let (shifted_bits, _) = run_waves(&ops, &slots, None, true, true, false)?;
    let shifted_diff = fixed_bits
        .iter()
        .zip(&shifted_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "prefill_packed_waves_gpu off-by-one base control shifted_diff_bits={shifted_diff} DETECTED"
    );
    anyhow::ensure!(
        shifted_diff > 0,
        "off-by-one wave base control was not detected"
    );

    // Control 3 (Step 2C): corrupted cohort slice offset (cohort * experts + experts); must FAIL (>0 differing bits).
    let (corrupted_cohort_bits, _) = run_waves(&ops, &slots, None, true, false, true)?;
    let cohort_offset_diff = packed_bits
        .iter()
        .zip(&corrupted_cohort_bits)
        .filter(|(a, b)| a != b)
        .count();
    let (fp8_corrupted_cohort_bits, _) =
        run_waves(&fp8_ops, &fp8_slots, Some(fp8_slot), true, false, true)?;
    let fp8_cohort_offset_diff = fp8_packed_bits
        .iter()
        .zip(&fp8_corrupted_cohort_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "prefill_packed_waves_gpu cohort offset control bf16_diff_bits={cohort_offset_diff} fp8_diff_bits={fp8_cohort_offset_diff} DETECTED"
    );
    anyhow::ensure!(
        cohort_offset_diff > 0 && fp8_cohort_offset_diff > 0,
        "corrupted cohort slice offset control was not detected: bf16={cohort_offset_diff} fp8={fp8_cohort_offset_diff}"
    );

    // Control 2: mismatch host-packed active count against device count at :899; must return Err.
    let &(base, w_rows, active) = &plan.cohort_waves(0)?[0];
    let bad_c: CudaSlice<bf16> = s.alloc_zeros(w_rows as usize * top_k * hidden)?;
    let bad_d: CudaSlice<u32> = s.alloc_zeros(w_rows as usize * top_k)?;
    let mismatch = ops.prefill_contribute(
        s,
        MoePrefillContribute {
            expert_inter: inter as u32,
            cpu_arithmetic: &cpu_arithmetic,
            fp8_slot: None,
            wave: Some((0, base, active - 1)),
            wait_copy: None,
            reader_event: None,
            x: view(
                &x_dev,
                s,
                base as usize * hidden,
                [w_rows, hidden as u32],
                [hidden as u64, 1],
            )?,
            ids: view(
                &ids_dev,
                s,
                base as usize * top_k,
                [w_rows, top_k as u32],
                [top_k as u64, 1],
            )?,
            weights: view(
                &weights_dev,
                s,
                base as usize * top_k,
                [w_rows, top_k as u32],
                [top_k as u64, 1],
            )?,
            active: view(
                &cohort_masks[0],
                s,
                base as usize * top_k,
                [w_rows, top_k as u32],
                [top_k as u64, 1],
            )?,
            assignment_gen: view(
                &gens_dev,
                s,
                base as usize * top_k,
                [w_rows, top_k as u32],
                [top_k as u64, 1],
            )?,
            slots_gate_up: view(&slots.gu_dev, s, 0, [experts as u32], [1])?,
            slots_down: view(&slots.dn_dev, s, 0, [experts as u32], [1])?,
            slot_valid: view(&slots.valid, s, 0, [experts as u32], [1])?,
            slot_generation: view(&slots.generation, s, 0, [experts as u32], [1])?,
            slots_gate_up_host: &slots.gu_host[..experts],
            slots_down_host: &slots.dn_host[..experts],
            shared_gate: testing::dense(&shared.sg, s, inter as u32, hidden as u32)?,
            shared_up: testing::dense(&shared.su, s, inter as u32, hidden as u32)?,
            shared_down: testing::dense(&shared.sd, s, hidden as u32, inter as u32)?,
            shared_expert_gate: Some(testing::view(&shared.seg, s, [hidden as u32], [1])?),
            contributions: view(
                &bad_c,
                s,
                0,
                [w_rows, top_k as u32, hidden as u32],
                [(top_k * hidden) as u64, hidden as u64, 1],
            )?,
            done: TensorMut(view(
                &bad_d,
                s,
                0,
                [w_rows, top_k as u32],
                [top_k as u64, 1],
            )?),
        },
        &scratch,
    );
    anyhow::ensure!(
        mismatch.is_err(),
        "mismatched wave active count control was not detected"
    );
    Ok(())
}
