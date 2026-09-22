//! Groupwise affine int4/int8 quantized weights: decoding and fused matvec.
//!
//! Format (byte-level pinned): payload words
//! are little-endian U32 packing 8x unsigned int4 (low nibble first; the
//! MLX `gather_qmm` convention) or 4x unsigned int8; scales and biases are
//! bf16 `[out, in/group_size]`, independently rounded by the encoder. The
//! exact reconstruction is `w = s*q + b` in f32 — the biases tensor is
//! authoritative (the integer zero-point is NOT stored and its recovery is
//! approximate), which is why the kernel never materializes `q - z`.
//!
//! The bias term factors per group: for a matvec,
//! `y[o] = sum_g s[o,g]*dot_g(o) + b[o,g]*sum_g(x)`, so group sums of the
//! input are computed once per call and the packed payload is touched
//! exactly once per output element.

use anyhow::{Result, ensure};

pub const GROUP_SIZE: usize = 64;

/// 8-accumulator scalar fallback for the group dot.
fn dot_scalar(u: &[f32], v: &[f32]) -> f32 {
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) =
        (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    let mut m = 0;
    while m < GROUP_SIZE {
        a0 += u[m] * v[m];
        a1 += u[m + 1] * v[m + 1];
        a2 += u[m + 2] * v[m + 2];
        a3 += u[m + 3] * v[m + 3];
        a4 += u[m + 4] * v[m + 4];
        a5 += u[m + 5] * v[m + 5];
        a6 += u[m + 6] * v[m + 6];
        a7 += u[m + 7] * v[m + 7];
        m += 8;
    }
    ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::needless_range_loop)] // lane -> __m256 register mapping
unsafe fn dot_avx2(u: &[f32], v: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // Eight independent f32x8 FMA chains cover the 64-wide group. The
    // lane-indexed __m256 array is the algorithm — iterating it would
    // obscure the register mapping.
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut m = 0;
    while m < GROUP_SIZE {
        for lane in 0..8 {
            unsafe {
                let uu = _mm256_loadu_ps(u.as_ptr().add(m + lane * 8));
                let vv = _mm256_loadu_ps(v.as_ptr().add(m + lane * 8));
                acc[lane] = _mm256_fmadd_ps(uu, vv, acc[lane]);
            }
        }
        m += 64;
    }
    let mut sum = _mm256_setzero_ps();
    for lane in 0..8 {
        sum = _mm256_add_ps(sum, acc[lane]);
    }
    let mut out = [0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), sum) };
    out.iter().sum::<f32>()
}

fn group_dot(u: &[f32], v: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_avx2(u, v) };
        }
    }
    dot_scalar(u, v)
}

/// Fused int4 dequant-and-dot over a full row: one `u32` word holds 8
/// consecutive columns (low nibble first), so two words dequant straight
/// into two f32x8 lanes via byte-level nibble extraction — no intermediate
/// unpack buffer. Accumulates per group so scales/biases apply per 64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_fused_avx2(
    words: &[u32],
    x: &[f32],
    scales: &[f32],
    biases: &[f32],
    group_sums: &[f32],
    groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let nibble_mask = _mm_set1_epi8(0x0F_i8);
    let mut total = 0f32;
    for group in 0..groups {
        let column = group * GROUP_SIZE;
        let mut acc = _mm256_setzero_ps();
        let mut c = column;
        // 64 columns per group = 8 words, processed two at a time.
        let word_base = group * 8;
        for pair in 0..4 {
            unsafe {
                let pair_words =
                    _mm_loadl_epi64(words.as_ptr().add(word_base + pair * 2) as *const __m128i);
                let even = _mm_and_si128(pair_words, nibble_mask);
                let odd = _mm_and_si128(_mm_srli_epi16(pair_words, 4), nibble_mask);
                let ev = _mm256_cvtepu8_epi32(even);
                let od = _mm256_cvtepu8_epi32(odd);
                let ev_f = _mm256_cvtepi32_ps(ev);
                let od_f = _mm256_cvtepi32_ps(od);
                let lo = _mm256_unpacklo_ps(ev_f, od_f);
                let hi = _mm256_unpackhi_ps(ev_f, od_f);
                let first = _mm256_permute2f128_ps(lo, hi, 0x20);
                let second = _mm256_permute2f128_ps(lo, hi, 0x31);
                let xv1 = _mm256_loadu_ps(x.as_ptr().add(c));
                let xv2 = _mm256_loadu_ps(x.as_ptr().add(c + 8));
                acc = _mm256_fmadd_ps(first, xv1, acc);
                acc = _mm256_fmadd_ps(second, xv2, acc);
            }
            c += 16;
        }
        let mut lanes = [0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let dot: f32 = lanes.iter().sum();
        total += scales[group] * dot + biases[group] * group_sums[group];
    }
    total
}

/// AVX-512 variant: two words (16 nibbles) become one __m512 directly —
/// byte-interleave of even/odd nibbles yields sequential columns with no
/// 128-bit lane permute. Applied to instruction-bound buckets only; the
/// lm_head is bandwidth-side after the AVX2 kernel, where wider lanes
/// buy nothing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
unsafe fn row_dot_fused_avx512(
    words: &[u32],
    x: &[f32],
    scales: &[f32],
    biases: &[f32],
    group_sums: &[f32],
    groups: usize,
) -> f32 {
    use std::arch::x86_64::*;
    let nibble_mask = _mm_set1_epi8(0x0F_i8);
    let mut total = 0f32;
    for group in 0..groups {
        let mut acc = _mm512_setzero_ps();
        let word_base = group * 8;
        let mut c = group * GROUP_SIZE;
        for pair in 0..4 {
            unsafe {
                let pair_words =
                    _mm_loadl_epi64(words.as_ptr().add(word_base + pair * 2) as *const __m128i);
                let even = _mm_and_si128(pair_words, nibble_mask);
                let odd = _mm_and_si128(_mm_srli_epi16(pair_words, 4), nibble_mask);
                let inter = _mm_unpacklo_epi8(even, odd);
                let nibbles = _mm512_cvtepu8_epi32(inter);
                let f = _mm512_cvtepi32_ps(nibbles);
                let xv = _mm512_loadu_ps(x.as_ptr().add(c));
                acc = _mm512_fmadd_ps(f, xv, acc);
            }
            c += 16;
        }
        let mut lanes = [0f32; 16];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        let dot: f32 = lanes.iter().sum();
        total += scales[group] * dot + biases[group] * group_sums[group];
    }
    total
}

#[cfg(target_arch = "x86_64")]
fn row_dot_fused(
    words: &[u32],
    x: &[f32],
    scales: &[f32],
    biases: &[f32],
    group_sums: &[f32],
    groups: usize,
) -> Option<f32> {
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
    {
        return Some(unsafe { row_dot_fused_avx512(words, x, scales, biases, group_sums, groups) });
    }
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return Some(unsafe { row_dot_fused_avx2(words, x, scales, biases, group_sums, groups) });
    }
    None
}

#[cfg(not(target_arch = "x86_64"))]
fn row_dot_fused(_: &[u32], _: &[f32], _: &[f32], _: &[f32], _: &[f32], _: usize) -> Option<f32> {
    None
}

/// A quantized projection resident in host memory.
#[derive(Debug)]
pub struct GroupQuant {
    /// `[out, in/bits_per_word]` packed payload, row-major.
    pub packed: Vec<u32>,
    /// `[out, in/GROUP_SIZE]`, converted from bf16 to f32 at load.
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub bits: u32,
}

/// bf16 (little-endian u16 payload) -> f32.
pub fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|pair| f32::from_bits((u32::from(pair[0]) | (u32::from(pair[1]) << 8)) << 16))
        .collect()
}

impl GroupQuant {
    pub fn new(
        packed: Vec<u32>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        bits: u32,
    ) -> Result<Self> {
        let per_word = match bits {
            4 => 8,
            8 => 4,
            other => anyhow::bail!("unsupported Edge0 quantization width {other}"),
        };
        let groups = in_dim / GROUP_SIZE;
        ensure!(
            packed.len() == out_dim * in_dim / per_word,
            "packed payload has {} words; expected {} for [{out_dim}, {in_dim}] at {bits} bits",
            packed.len(),
            out_dim * in_dim / per_word
        );
        ensure!(
            scales.len() == out_dim * groups && biases.len() == out_dim * groups,
            "scales/biases length disagrees with [{out_dim}, {groups}]"
        );
        Ok(Self {
            packed,
            scales,
            biases,
            out_dim,
            in_dim,
            bits,
        })
    }

    #[inline]
    fn element(&self, row: usize, column: usize) -> u32 {
        let per_word = if self.bits == 4 { 8 } else { 4 };
        let shift = if self.bits == 4 { 4 } else { 8 };
        let word = self.packed[row * (self.in_dim / per_word) + column / per_word];
        (word >> (shift * (column % per_word))) & ((1 << shift) - 1)
    }

    /// Dequantize one full row (used by tests and small paths).
    pub fn row(&self, row: usize) -> Vec<f32> {
        let groups = self.in_dim / GROUP_SIZE;
        (0..self.in_dim)
            .map(|column| {
                let group = column / GROUP_SIZE;
                self.scales[row * groups + group] * self.element(row, column) as f32
                    + self.biases[row * groups + group]
            })
            .collect()
    }

    /// Fused quantized matvec with an optional rank-`r` LoRA pair.
    ///
    /// `lora` is `(a, b)` with `a` `[r, in]` and `b` `[out, r]`, applied as
    /// `y += b * (a * x)` — the adapters stay unmerged by design (merging
    /// would force 2.6 GiB of full-precision residency; see the design
    /// doc's LoRA decision).
    pub fn matvec(&self, x: &[f32], lora: Option<(&[f32], &[f32], usize)>) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.in_dim,
            "matvec input has {} elements, projection expects {}",
            x.len(),
            self.in_dim
        );
        if let Some((a, b, rank)) = lora {
            assert_eq!(
                a.len(),
                rank * self.in_dim,
                "lora A has {} elements, expected [rank {rank}, in {}]",
                a.len(),
                self.in_dim
            );
            assert_eq!(
                b.len(),
                self.out_dim * rank,
                "lora B has {} elements, expected [out {}, rank {rank}]",
                b.len(),
                self.out_dim
            );
        }
        let groups = self.in_dim / GROUP_SIZE;
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_threads = std::env::var("EDGE0_MATVEC_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or_else(|| (self.out_dim / 512).clamp(1, cores));
        let per_word = if self.bits == 4 { 8usize } else { 4 };
        let shift = if self.bits == 4 { 4u32 } else { 8 };
        let mask: u32 = (1 << shift) - 1;
        let words_per_row = self.in_dim / per_word;
        // Group sums of the input, hoisted out of the output loop.
        let mut group_sums = vec![0f32; groups];
        for (column, &value) in x.iter().enumerate() {
            group_sums[column / GROUP_SIZE] += value;
        }
        let mut y = vec![0f32; self.out_dim];
        // Expert slices and gates stay inline: spawn overhead beats
        // parallelism at a few hundred rows.
        let force_inline = std::env::var_os("EDGE0_MATVEC_INLINE").is_some();
        if self.out_dim <= 512 || force_inline {
            #[allow(clippy::needless_range_loop)] // row drives 4 slice views
            for row in 0..self.out_dim {
                if self.bits == 4
                    && let Some(value) = row_dot_fused(
                        &self.packed[row * words_per_row..(row + 1) * words_per_row],
                        x,
                        &self.scales[row * groups..(row + 1) * groups],
                        &self.biases[row * groups..(row + 1) * groups],
                        &group_sums,
                        groups,
                    )
                {
                    y[row] = value;
                    continue;
                }
                let mut unpacked = vec![0f32; self.in_dim];
                y[row] = self.row_dot(row, x, &group_sums, groups, &mut unpacked);
            }
            if let Some((a, b, rank)) = lora {
                let delta = lora_delta(a, b, rank, x, self.in_dim, self.out_dim);
                for (slot, d) in y.iter_mut().zip(delta) {
                    *slot += d;
                }
            }
            return y;
        }
        let chunk = (self.out_dim / num_threads).max(1);
        std::thread::scope(|scope| {
            let slices: Vec<&mut [f32]> = y.chunks_mut(chunk).collect();
            for (index, slice) in slices.into_iter().enumerate() {
                let q = &self;
                let sums = &group_sums[..];
                scope.spawn(move || {
                    let row_start = index * chunk;
                    let mut unpacked = vec![0f32; q.in_dim];
                    for (offset, slot) in slice.iter_mut().enumerate() {
                        let row = row_start + offset;
                        let row_words = &q.packed[row * words_per_row..(row + 1) * words_per_row];
                        if q.bits == 4
                            && let Some(value) = row_dot_fused(
                                row_words,
                                x,
                                &q.scales[row * groups..(row + 1) * groups],
                                &q.biases[row * groups..(row + 1) * groups],
                                sums,
                                groups,
                            )
                        {
                            *slot = value;
                            continue;
                        }
                        for (word_index, &word) in row_words.iter().enumerate() {
                            let base = word_index * per_word;
                            for j in 0..per_word {
                                unpacked[base + j] = ((word >> (shift * j as u32)) & mask) as f32;
                            }
                        }
                        let mut total = 0f32;
                        #[allow(clippy::needless_range_loop)] // 3-array group indexing
                        for group in 0..groups {
                            let start = group * GROUP_SIZE;
                            let dot = group_dot(
                                &unpacked[start..start + GROUP_SIZE],
                                &x[start..start + GROUP_SIZE],
                            );
                            total += q.scales[row * groups + group] * dot
                                + q.biases[row * groups + group] * sums[group];
                        }
                        *slot = total;
                    }
                });
            }
        });
        if let Some((a, b, rank)) = lora {
            let delta = lora_delta(a, b, rank, x, self.in_dim, self.out_dim);
            for (slot, d) in y.iter_mut().zip(delta) {
                *slot += d;
            }
        }
        y
    }

    fn row_dot(
        &self,
        row: usize,
        x: &[f32],
        group_sums: &[f32],
        groups: usize,
        unpacked: &mut [f32],
    ) -> f32 {
        let per_word = if self.bits == 4 { 8usize } else { 4 };
        let shift = if self.bits == 4 { 4u32 } else { 8 };
        let mask: u32 = (1 << shift) - 1;
        let words_per_row = self.in_dim / per_word;
        for word_index in 0..words_per_row {
            let word = self.packed[row * words_per_row + word_index];
            let base = word_index * per_word;
            for j in 0..per_word {
                unpacked[base + j] = ((word >> (shift * j as u32)) & mask) as f32;
            }
        }
        let mut total = 0f32;
        #[allow(clippy::needless_range_loop)] // 3-array group indexing
        for group in 0..groups {
            let start = group * GROUP_SIZE;
            let dot = group_dot(
                &unpacked[start..start + GROUP_SIZE],
                &x[start..start + GROUP_SIZE],
            );
            total += self.scales[row * groups + group] * dot
                + self.biases[row * groups + group] * group_sums[group];
        }
        total
    }
}

pub(crate) fn lora_delta(
    a: &[f32],
    b: &[f32],
    rank: usize,
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut inner = vec![0f32; rank];
    for (k, slot) in inner.iter_mut().enumerate() {
        let mut acc = 0f32;
        for column in 0..in_dim {
            acc += a[k * in_dim + column] * x[column];
        }
        *slot = acc;
    }
    let mut out = vec![0f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0f32;
        for (k, &d) in inner.iter().enumerate() {
            acc += b[row * rank + k] * d;
        }
        out[row] = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/edge0_test_vectors.json")).unwrap()
    }

    fn quant_from_fixture(vectors: &serde_json::Value) -> GroupQuant {
        let packed: Vec<u32> = vectors["packed_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let scales: Vec<f32> = vectors["scales_f32"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let biases: Vec<f32> = vectors["biases_f32"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let in_dim = vectors["in_dim"].as_u64().unwrap() as usize;
        let groups = in_dim / GROUP_SIZE;
        GroupQuant::new(
            packed,
            scales[..groups].to_vec(),
            biases[..groups].to_vec(),
            1,
            in_dim,
            4,
        )
        .unwrap()
    }

    /// Format gate: 1 ULP because the Python reference computes `s*q+b`
    /// in f64 (one rounding) while Rust rounds twice.
    #[test]
    fn dequant_row_matches_the_python_reference_within_one_ulp() {
        let vectors = fixture();
        let quant = quant_from_fixture(&vectors);
        let reference: Vec<f64> = vectors["values_low_nibble_first"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let row = quant.row(0);
        assert_eq!(row.len(), reference.len());
        for (index, (&got, &want)) in row.iter().zip(&reference).enumerate() {
            let want = want as f32;
            let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulps <= 1,
                "element {index}: got {got:e} (bits {:x}), want {want:e} (bits {:x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    /// Guard that the discriminator has teeth: high-first must disagree.
    #[test]
    fn the_wrong_nibble_order_actually_disagrees() {
        let vectors = fixture();
        let quant = quant_from_fixture(&vectors);
        let reference: Vec<f64> = vectors["values_high_nibble_first"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let row = quant.row(0);
        let mismatches = row
            .iter()
            .zip(&reference)
            .filter(|(got, want)| (**got - **want as f32).abs() > f32::EPSILON)
            .count();
        assert!(
            mismatches > 100,
            "expected widespread disagreement, got {mismatches}"
        );
    }

    /// The one tolerance-free assertion: z=0 groups carry bias exactly 0.0.
    #[test]
    fn z0_groups_have_exactly_zero_bias() {
        let vectors = fixture();
        let zeros: Vec<f64> = vectors["z0_biases_sample"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert!(!zeros.is_empty());
        assert!(zeros.iter().all(|&b| b == 0.0));
    }

    /// Accumulation gate: bound = 4*sqrt(n)*eps from the n-term dot model
    /// (two accumulation orders), NOT 1 ULP — that bound is layer-1 only.
    #[test]
    fn matvec_matches_serial_dot_within_derived_tolerance() {
        let vectors = fixture();
        let quant = quant_from_fixture(&vectors);
        let in_dim = quant.in_dim;
        let x: Vec<f32> = (0..in_dim)
            .map(|i| (i as f32 * 0.037).sin() * 0.5)
            .collect();
        let y = quant.matvec(&x, None)[0];
        let row = quant.row(0);
        let mut serial = 0f32;
        for (w, &v) in row.iter().zip(&x) {
            serial += w * v;
        }
        let n = in_dim as f32;
        let bound = 4.0 * n.sqrt() * f32::EPSILON * serial.abs().max(1.0);
        assert!(
            (y - serial).abs() <= bound,
            "matvec {y:e} vs serial {serial:e}, bound {bound:e}"
        );
    }
}
