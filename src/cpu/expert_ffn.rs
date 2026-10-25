//! Host-side routed-expert FFN for decode misses; model-independent.
use crate::experts::arena::{BlockLease, HostBlocks};
use crate::experts::catalogue::{ExpertGeometry, ExpertLayout, Fp8Slot};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use std::ops::Range;
use std::sync::{
    Arc, Condvar, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU16, AtomicU64, AtomicUsize, Ordering},
};
use std::thread::{self, JoinHandle};

/// Partial-sum lanes shared by every arm; the scalar arm defines this order.
const LANES: usize = 16;

/// Spin rounds a parked worker and the calling thread run before they block.
const SPIN_ROUNDS: u32 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: usize,
    pub intermediate: usize,
}

impl Shape {
    pub fn new(hidden: usize, intermediate: usize) -> Result<Self> {
        ensure!(hidden > 0 && intermediate > 0, "zero expert dimension");
        ensure!(
            intermediate.is_multiple_of(8) && hidden.is_multiple_of(4),
            "expert dimensions are not multiple-of-lane shaped"
        );
        Ok(Self {
            hidden,
            intermediate,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512Bf16,
}

impl Isa {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::is_x86_feature_detected as has;
            if has!("avx512bf16") && has!("avx512f") {
                return Self::Avx512Bf16;
            }
            if has!("avx2") && has!("fma") {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2+fma",
            Self::Avx512Bf16 => "avx512bf16",
        }
    }

    fn available(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                use std::arch::is_x86_feature_detected as has;
                has!("avx2") && has!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512Bf16 => {
                use std::arch::is_x86_feature_detected as has;
                has!("avx512bf16") && has!("avx512f")
            }
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

/// BF16 arena-word ranges: fused matrices for BF16 or the full native FP8 slot in both ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertWeights {
    pub gate_up: Range<usize>,
    pub gate_up_stride: usize,
    pub down: Range<usize>,
    pub down_stride: usize,
}

impl ExpertWeights {
    pub fn from_slot(shape: Shape, range: Range<usize>, layout: &ExpertLayout) -> Result<Self> {
        ensure!(
            range.end >= range.start
                && range.len().checked_mul(2) == Some(usize::try_from(layout.slot_bytes())?),
            "expert slot extent differs from layout"
        );
        let (gate_up, down) = match layout {
            ExpertLayout::Bf16Fused { gate_up_bytes, .. } => {
                let down = range.start + usize::try_from(gate_up_bytes / 2)?;
                (range.start..down, down..range.end)
            }
            ExpertLayout::Bf16Separate { matrix_bytes, .. } => {
                let down = range.start + usize::try_from(*matrix_bytes)?;
                (range.start..down, down..range.end)
            }
            ExpertLayout::Fp8W8a16(_) => (range.clone(), range),
        };
        Ok(Self {
            gate_up,
            gate_up_stride: shape.hidden,
            down,
            down_stride: shape.intermediate,
        })
    }

    fn validate(&self, shape: &Shape, arena: usize, fp8: Option<&Fp8Slot>) -> Result<()> {
        let (hidden, intermediate) = (shape.hidden, shape.intermediate);
        if let Some(slot) = fp8 {
            ensure!(
                self.gate_up.start <= self.gate_up.end
                    && self.gate_up == self.down
                    && self.gate_up.end <= arena
                    && self.gate_up.len().checked_mul(2)
                        == Some(usize::try_from(slot.down_scale.end)?)
                    && self.gate_up_stride == hidden
                    && self.down_stride == intermediate,
                "FP8 expert slot differs from the pool layout"
            );
            return Ok(());
        }
        ensure!(
            self.gate_up_stride >= hidden && self.down_stride >= intermediate,
            "expert stride below the row width"
        );
        let gate_up_end = self
            .gate_up
            .start
            .checked_add((2 * intermediate - 1) * self.gate_up_stride + hidden)
            .context("gate-up range overflow")?;
        ensure!(
            gate_up_end <= self.gate_up.end && self.gate_up.end <= arena,
            "gate-up range exceeds the arena"
        );
        let down_end = self
            .down
            .start
            .checked_add((hidden - 1) * self.down_stride + intermediate)
            .context("down range overflow")?;
        ensure!(
            down_end <= self.down.end && self.down.end <= arena,
            "down range exceeds the arena"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assignment {
    pub row: u32,
    pub choice: u32,
    pub expert: u32,
    pub weight: bf16,
}

pub struct Scratch {
    hidden: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gate_bf: Vec<bf16>,
    up_bf: Vec<bf16>,
    product: Vec<f32>,
    activated: Vec<bf16>,
    activated_f: Vec<f32>,
    down: Vec<f32>,
    down_bf: Vec<bf16>,
    weighted: Vec<f32>,
}

impl Scratch {
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn stages(&self) -> [&[bf16]; 4] {
        [&self.gate_bf, &self.up_bf, &self.activated, &self.down_bf]
    }

    pub fn new(shape: &Shape) -> Self {
        Self {
            hidden: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            gate_bf: vec![bf16::ZERO; shape.intermediate],
            up_bf: vec![bf16::ZERO; shape.intermediate],
            product: vec![0.0; shape.intermediate],
            activated: vec![bf16::ZERO; shape.intermediate],
            activated_f: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
            down_bf: vec![bf16::ZERO; shape.hidden],
            weighted: vec![0.0; shape.hidden],
        }
    }

    fn check(&self, shape: &Shape) -> Result<()> {
        ensure!(
            self.hidden.len() == shape.hidden
                && self.gate.len() == shape.intermediate
                && self.up.len() == shape.intermediate
                && self.gate_bf.len() == shape.intermediate
                && self.up_bf.len() == shape.intermediate
                && self.product.len() == shape.intermediate
                && self.activated.len() == shape.intermediate
                && self.activated_f.len() == shape.intermediate
                && self.down.len() == shape.hidden
                && self.down_bf.len() == shape.hidden
                && self.weighted.len() == shape.hidden,
            "scratch dimensions differ from the expert shape"
        );
        Ok(())
    }
}

type DotFn = unsafe fn(&[f32], &[bf16]) -> f32;
type Fp8DotFn = unsafe fn(&[f32], &[u8], bf16) -> f32;
type RoundFn = unsafe fn(&[f32], &mut [bf16]);

struct Arm {
    dot: DotFn,
    dot_e4m3: Fp8DotFn,
    round: RoundFn,
}

const SCALAR_ARM: Arm = Arm {
    dot: scalar_dot,
    dot_e4m3: scalar_dot_e4m3,
    round: scalar_round,
};

fn arm_for(isa: Isa) -> Result<&'static Arm> {
    match isa {
        Isa::Scalar => Ok(&SCALAR_ARM),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => {
            ensure!(isa.available(), "AVX2+FMA not available");
            Ok(&AVX2_ARM)
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512Bf16 => {
            ensure!(isa.available(), "AVX512_BF16 not available");
            Ok(&AVX512_ARM)
        }
        #[cfg(not(target_arch = "x86_64"))]
        _ => bail!("unsupported ISA arm {}", isa.name()),
    }
}

/// One routed-expert contribution input.
pub struct Contribution<'a> {
    pub shape: &'a Shape,
    pub arena: &'a [bf16],
    pub weights: &'a ExpertWeights,
    pub hidden: &'a [bf16],
    pub routing: bf16,
    pub rounding: crate::model_spec::SwigluRounding,
    pub routed_weight: crate::model_spec::RoutedWeightBoundary,
}

/// One `(row, choice)` contribution with the model-definition BF16 boundaries.
pub fn contribution(
    isa: Isa,
    input: Contribution<'_>,
    out: &mut [bf16],
    scratch: &mut Scratch,
) -> Result<()> {
    ensure!(
        input.hidden.len() == input.shape.hidden && out.len() == input.shape.hidden,
        "hidden/out width differs from the expert shape"
    );
    input
        .weights
        .validate(input.shape, input.arena.len(), None)?;
    scratch.check(input.shape)?;
    compute(arm_for(isa)?, input, out, scratch);
    Ok(())
}

fn compute(arm: &Arm, input: Contribution<'_>, out: &mut [bf16], scratch: &mut Scratch) {
    let Contribution {
        shape,
        arena,
        weights,
        hidden,
        routing,
        rounding,
        routed_weight,
    } = input;
    let (hidden_width, intermediate) = (shape.hidden, shape.intermediate);
    for (value, source) in scratch.hidden.iter_mut().zip(hidden) {
        *value = f32::from(*source);
    }
    let gate_up = &arena[weights.gate_up.clone()];
    for j in 0..intermediate {
        let gate = &gate_up[j * weights.gate_up_stride..][..hidden_width];
        let up = &gate_up[(intermediate + j) * weights.gate_up_stride..][..hidden_width];
        // SAFETY: the arm was selected by `Isa::detect` or checked by `Isa::available`.
        scratch.gate[j] = unsafe { (arm.dot)(&scratch.hidden, gate) };
        scratch.up[j] = unsafe { (arm.dot)(&scratch.hidden, up) };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.gate, &mut scratch.gate_bf) };
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.up, &mut scratch.up_bf) };
    for j in 0..intermediate {
        let gate = f32::from(scratch.gate_bf[j]);
        let up = f32::from(scratch.up_bf[j]);
        let activation = gate * (1.0 / (1.0 + (-gate).exp()));
        scratch.product[j] = match rounding {
            crate::model_spec::SwigluRounding::Twice => f32::from(bf16::from_f32(activation)) * up,
            crate::model_spec::SwigluRounding::Once => activation * up,
        };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.product, &mut scratch.activated) };
    for (value, source) in scratch.activated_f.iter_mut().zip(&scratch.activated) {
        *value = f32::from(*source);
    }
    let down = &arena[weights.down.clone()];
    for k in 0..hidden_width {
        let row = &down[k * weights.down_stride..][..intermediate];
        // SAFETY: as above.
        scratch.down[k] = unsafe { (arm.dot)(&scratch.activated_f, row) };
    }
    let weight = f32::from(routing);
    match routed_weight {
        crate::model_spec::RoutedWeightBoundary::AfterDownBf16 => {
            // SAFETY: as above.
            unsafe { (arm.round)(&scratch.down, &mut scratch.down_bf) };
            for (value, source) in scratch.weighted.iter_mut().zip(&scratch.down_bf) {
                *value = f32::from(*source) * weight;
            }
        }
        crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16 => {
            for (value, source) in scratch.weighted.iter_mut().zip(&scratch.down) {
                *value = *source * weight;
            }
        }
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.weighted, out) };
}

/// One routed-expert contribution over one native FP8 slot: E4M3 weights scaled per output
/// channel, the BF16 arm's rounding boundaries and accumulation order.
pub struct Fp8Contribution<'a> {
    pub shape: &'a Shape,
    pub slot: &'a [u8],
    pub slot_layout: &'a crate::experts::catalogue::Fp8Slot,
    pub hidden: &'a [bf16],
    pub routing: bf16,
    pub rounding: crate::model_spec::SwigluRounding,
    pub routed_weight: crate::model_spec::RoutedWeightBoundary,
}

struct Fp8Rows<'a> {
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
    gate_scale: &'a [u8],
    up_scale: &'a [u8],
    down_scale: &'a [u8],
    rounding: crate::model_spec::SwigluRounding,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
}

fn fp8_slice<'a>(slot: &'a [u8], range: &Range<u64>, bytes: usize, role: &str) -> Result<&'a [u8]> {
    let start = usize::try_from(range.start).context("FP8 slot offset exceeds usize")?;
    let end = usize::try_from(range.end).context("FP8 slot end exceeds usize")?;
    ensure!(
        end <= slot.len(),
        "FP8 {role} range {start}..{end} exceeds the {}-byte slot",
        slot.len()
    );
    ensure!(
        end - start == bytes,
        "FP8 {role} range holds {} bytes, expected {bytes}",
        end - start
    );
    Ok(&slot[start..end])
}

fn row_scale(bytes: &[u8], row: usize) -> bf16 {
    let at = 2 * row;
    bf16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

pub fn fp8_contribution(
    isa: Isa,
    input: Fp8Contribution<'_>,
    out: &mut [bf16],
    scratch: &mut Scratch,
) -> Result<()> {
    let (hidden, intermediate) = (input.shape.hidden, input.shape.intermediate);
    ensure!(
        input.hidden.len() == hidden && out.len() == hidden,
        "hidden/out width differs from the expert shape"
    );
    let layout = input.slot_layout;
    let rows = Fp8Rows {
        gate: fp8_slice(input.slot, &layout.gate, intermediate * hidden, "gate")?,
        up: fp8_slice(input.slot, &layout.up, intermediate * hidden, "up")?,
        down: fp8_slice(input.slot, &layout.down, hidden * intermediate, "down")?,
        gate_scale: fp8_slice(
            input.slot,
            &layout.gate_scale,
            intermediate * 2,
            "gate scale",
        )?,
        up_scale: fp8_slice(input.slot, &layout.up_scale, intermediate * 2, "up scale")?,
        down_scale: fp8_slice(input.slot, &layout.down_scale, hidden * 2, "down scale")?,
        rounding: input.rounding,
        routed_weight: input.routed_weight,
    };
    scratch.check(input.shape)?;
    compute_fp8(
        arm_for(isa)?,
        input.shape,
        rows,
        input.hidden,
        input.routing,
        out,
        scratch,
    );
    Ok(())
}

fn compute_fp8(
    arm: &Arm,
    shape: &Shape,
    rows: Fp8Rows<'_>,
    hidden: &[bf16],
    routing: bf16,
    out: &mut [bf16],
    scratch: &mut Scratch,
) {
    let (hidden_width, intermediate) = (shape.hidden, shape.intermediate);
    for (value, source) in scratch.hidden.iter_mut().zip(hidden) {
        *value = f32::from(*source);
    }
    for j in 0..intermediate {
        let gate = &rows.gate[j * hidden_width..][..hidden_width];
        let up = &rows.up[j * hidden_width..][..hidden_width];
        // SAFETY: the arm was selected by `Isa::detect` or checked by `Isa::available`.
        scratch.gate[j] =
            unsafe { (arm.dot_e4m3)(&scratch.hidden, gate, row_scale(rows.gate_scale, j)) };
        scratch.up[j] = unsafe { (arm.dot_e4m3)(&scratch.hidden, up, row_scale(rows.up_scale, j)) };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.gate, &mut scratch.gate_bf) };
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.up, &mut scratch.up_bf) };
    for j in 0..intermediate {
        let gate = f32::from(scratch.gate_bf[j]);
        let up = f32::from(scratch.up_bf[j]);
        let activation = gate * (1.0 / (1.0 + (-gate).exp()));
        scratch.product[j] = match rows.rounding {
            crate::model_spec::SwigluRounding::Twice => f32::from(bf16::from_f32(activation)) * up,
            crate::model_spec::SwigluRounding::Once => activation * up,
        };
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.product, &mut scratch.activated) };
    for (value, source) in scratch.activated_f.iter_mut().zip(&scratch.activated) {
        *value = f32::from(*source);
    }
    for k in 0..hidden_width {
        let row = &rows.down[k * intermediate..][..intermediate];
        // SAFETY: as above.
        scratch.down[k] =
            unsafe { (arm.dot_e4m3)(&scratch.activated_f, row, row_scale(rows.down_scale, k)) };
    }
    let weight = f32::from(routing);
    match rows.routed_weight {
        crate::model_spec::RoutedWeightBoundary::AfterDownBf16 => {
            // SAFETY: as above.
            unsafe { (arm.round)(&scratch.down, &mut scratch.down_bf) };
            for (value, source) in scratch.weighted.iter_mut().zip(&scratch.down_bf) {
                *value = f32::from(*source) * weight;
            }
        }
        crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16 => {
            for (value, source) in scratch.weighted.iter_mut().zip(&scratch.down) {
                *value = *source * weight;
            }
        }
    }
    // SAFETY: as above.
    unsafe { (arm.round)(&scratch.weighted, out) };
}

fn pairwise(lanes: &[f32; LANES]) -> f32 {
    let low = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let high = ((lanes[8] + lanes[9]) + (lanes[10] + lanes[11]))
        + ((lanes[12] + lanes[13]) + (lanes[14] + lanes[15]));
    low + high
}

fn scalar_dot(x: &[f32], weights: &[bf16]) -> f32 {
    assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
    let mut lanes = [0f32; LANES];
    let mut k = 0;
    while k + LANES <= x.len() {
        for lane in 0..LANES {
            lanes[lane] = x[k + lane].mul_add(f32::from(weights[k + lane]), lanes[lane]);
        }
        k += LANES;
    }
    while k < x.len() {
        let lane = k % LANES;
        lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
        k += 1;
    }
    pairwise(&lanes)
}

fn scalar_dot_e4m3(x: &[f32], codes: &[u8], scale: bf16) -> f32 {
    assert_eq!(x.len(), codes.len(), "dot operand width mismatch");
    let mut lanes = [0f32; LANES];
    let mut k = 0;
    while k + LANES <= x.len() {
        for lane in 0..LANES {
            let weight = f32::from(crate::fp8::effective_weight(codes[k + lane], scale));
            lanes[lane] = x[k + lane].mul_add(weight, lanes[lane]);
        }
        k += LANES;
    }
    while k < x.len() {
        let lane = k % LANES;
        let weight = f32::from(crate::fp8::effective_weight(codes[k], scale));
        lanes[lane] = x[k].mul_add(weight, lanes[lane]);
        k += 1;
    }
    pairwise(&lanes)
}

fn scalar_round(values: &[f32], out: &mut [bf16]) {
    assert_eq!(values.len(), out.len(), "round operand width mismatch");
    for (value, target) in values.iter().zip(out.iter_mut()) {
        *target = bf16::from_f32(*value);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Arm, LANES, bf16, pairwise};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(x: &[f32], weights: &[bf16]) -> f32 {
        assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
        let mut low = _mm256_setzero_ps();
        let mut high = _mm256_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xl = _mm256_loadu_ps(x.as_ptr().add(k));
                let xh = _mm256_loadu_ps(x.as_ptr().add(k + 8));
                low = _mm256_fmadd_ps(xl, widen8(weights.as_ptr().add(k)), low);
                high = _mm256_fmadd_ps(xh, widen8(weights.as_ptr().add(k + 8)), high);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe {
            _mm256_storeu_ps(lanes.as_mut_ptr(), low);
            _mm256_storeu_ps(lanes.as_mut_ptr().add(8), high);
        }
        while k < x.len() {
            let lane = k % LANES;
            lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn widen8(values: *const bf16) -> __m256 {
        unsafe {
            let packed = _mm_loadu_si128(values as *const __m128i);
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16))
        }
    }

    /// E4M3 byte to f32 as the scalar decode defines it: bias 7, no infinity,
    /// magnitude 0x7f is NaN, exponent 0 decodes to mantissa / 512.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn decode8(codes: *const u8) -> __m256 {
        unsafe {
            let packed = _mm_loadl_epi64(codes as *const __m128i);
            let value = _mm256_cvtepu8_epi32(packed);
            let sign = _mm256_slli_epi32(_mm256_and_si256(value, _mm256_set1_epi32(0x80)), 24);
            let magnitude = _mm256_and_si256(value, _mm256_set1_epi32(0x7f));
            let exponent = _mm256_srli_epi32(magnitude, 3);
            let mantissa = _mm256_and_si256(magnitude, _mm256_set1_epi32(7));
            let normal = _mm256_castsi256_ps(_mm256_or_si256(
                sign,
                _mm256_or_si256(
                    _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(120)), 23),
                    _mm256_slli_epi32(mantissa, 20),
                ),
            ));
            let subnormal = _mm256_or_ps(
                _mm256_castsi256_ps(sign),
                _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_set1_ps(1.0 / 512.0)),
            );
            let nan = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_set1_epi32(0x7fc0_0000)));
            let subnormal_mask =
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
            let nan_mask =
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x7f)));
            _mm256_blendv_ps(
                _mm256_blendv_ps(normal, subnormal, subnormal_mask),
                nan,
                nan_mask,
            )
        }
    }

    /// Rounds f32 lanes to BF16 (RNE) and widens back; matches `bf16::from_f32` for finite values.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn round_bf16x8(values: __m256) -> __m256 {
        let bits = _mm256_castps_si256(values);
        let lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        let bias = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), lsb);
        let rounded = _mm256_add_epi32(bits, bias);
        _mm256_castsi256_ps(_mm256_and_si256(rounded, _mm256_set1_epi32(-65536)))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_e4m3(x: &[f32], codes: &[u8], scale: bf16) -> f32 {
        assert_eq!(x.len(), codes.len(), "dot operand width mismatch");
        let factor = _mm256_set1_ps(scale.to_f32());
        let mut low = _mm256_setzero_ps();
        let mut high = _mm256_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xl = _mm256_loadu_ps(x.as_ptr().add(k));
                let xh = _mm256_loadu_ps(x.as_ptr().add(k + 8));
                let wl = round_bf16x8(_mm256_mul_ps(decode8(codes.as_ptr().add(k)), factor));
                let wh = round_bf16x8(_mm256_mul_ps(decode8(codes.as_ptr().add(k + 8)), factor));
                low = _mm256_fmadd_ps(xl, wl, low);
                high = _mm256_fmadd_ps(xh, wh, high);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe {
            _mm256_storeu_ps(lanes.as_mut_ptr(), low);
            _mm256_storeu_ps(lanes.as_mut_ptr().add(8), high);
        }
        while k < x.len() {
            let lane = k % LANES;
            let weight = f32::from(crate::fp8::effective_weight(codes[k], scale));
            lanes[lane] = x[k].mul_add(weight, lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn dot_512(x: &[f32], weights: &[bf16]) -> f32 {
        assert_eq!(x.len(), weights.len(), "dot operand width mismatch");
        let mut acc = _mm512_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xv = _mm512_loadu_ps(x.as_ptr().add(k));
                acc = _mm512_fmadd_ps(xv, widen16(weights.as_ptr().add(k)), acc);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        while k < x.len() {
            let lane = k % LANES;
            lanes[lane] = x[k].mul_add(f32::from(weights[k]), lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn widen16(values: *const bf16) -> __m512 {
        unsafe {
            let packed = _mm256_loadu_si256(values as *const __m256i);
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16))
        }
    }

    /// E4M3 byte to f32 as the scalar decode defines it: bias 7, no infinity,
    /// magnitude 0x7f is NaN, exponent 0 decodes to mantissa / 512.
    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn decode16(codes: *const u8) -> __m512 {
        unsafe {
            let packed = _mm_loadu_si128(codes as *const __m128i);
            let value = _mm512_cvtepu8_epi32(packed);
            let sign = _mm512_slli_epi32(_mm512_and_si512(value, _mm512_set1_epi32(0x80)), 24);
            let magnitude = _mm512_and_si512(value, _mm512_set1_epi32(0x7f));
            let exponent = _mm512_srli_epi32(magnitude, 3);
            let mantissa = _mm512_and_si512(magnitude, _mm512_set1_epi32(7));
            let normal = _mm512_castsi512_ps(_mm512_or_si512(
                sign,
                _mm512_or_si512(
                    _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
                    _mm512_slli_epi32(mantissa, 20),
                ),
            ));
            let subnormal = _mm512_or_ps(
                _mm512_castsi512_ps(sign),
                _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(1.0 / 512.0)),
            );
            let nan = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x7fc0_0000)));
            let subnormal_mask = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
            let nan_mask = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x7f));
            let decoded = _mm512_mask_blend_ps(subnormal_mask, normal, subnormal);
            _mm512_mask_blend_ps(nan_mask, decoded, nan)
        }
    }

    /// Rounds f32 lanes to BF16 (RNE) and widens back; matches `bf16::from_f32` for finite values.
    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn round_bf16x16(values: __m512) -> __m512 {
        let bits = _mm512_castps_si512(values);
        let lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
        let bias = _mm512_add_epi32(_mm512_set1_epi32(0x7fff), lsb);
        let rounded = _mm512_add_epi32(bits, bias);
        _mm512_castsi512_ps(_mm512_and_si512(rounded, _mm512_set1_epi32(-65536)))
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn dot_e4m3_512(x: &[f32], codes: &[u8], scale: bf16) -> f32 {
        assert_eq!(x.len(), codes.len(), "dot operand width mismatch");
        let factor = _mm512_set1_ps(scale.to_f32());
        let mut acc = _mm512_setzero_ps();
        let mut k = 0;
        while k + LANES <= x.len() {
            unsafe {
                let xv = _mm512_loadu_ps(x.as_ptr().add(k));
                let weights = round_bf16x16(_mm512_mul_ps(decode16(codes.as_ptr().add(k)), factor));
                acc = _mm512_fmadd_ps(xv, weights, acc);
            }
            k += LANES;
        }
        let mut lanes = [0f32; LANES];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
        while k < x.len() {
            let lane = k % LANES;
            let weight = f32::from(crate::fp8::effective_weight(codes[k], scale));
            lanes[lane] = x[k].mul_add(weight, lanes[lane]);
            k += 1;
        }
        pairwise(&lanes)
    }

    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) unsafe fn round_512(values: &[f32], out: &mut [bf16]) {
        assert_eq!(values.len(), out.len(), "round operand width mismatch");
        let mut k = 0;
        while k + 32 <= values.len() {
            let low = unsafe { _mm512_loadu_ps(values.as_ptr().add(k)) };
            let high = unsafe { _mm512_loadu_ps(values.as_ptr().add(k + 16)) };
            let packed = _mm512_cvtne2ps_pbh(high, low);
            let raw: __m512i = unsafe { std::mem::transmute(packed) };
            unsafe { _mm512_storeu_si512(out.as_mut_ptr().add(k) as *mut __m512i, raw) };
            k += 32;
        }
        for (value, target) in values[k..].iter().zip(out[k..].iter_mut()) {
            *target = bf16::from_f32(*value);
        }
    }

    pub(super) const AVX2: Arm = Arm {
        dot,
        dot_e4m3,
        round: super::scalar_round,
    };

    pub(super) const AVX512: Arm = Arm {
        dot: dot_512,
        dot_e4m3: dot_e4m3_512,
        round: round_512,
    };
}

#[cfg(target_arch = "x86_64")]
use x86::{AVX2 as AVX2_ARM, AVX512 as AVX512_ARM};

/// A fixed worker pool over native expert slices with per-job block read leases.
pub struct ExpertPool {
    shape: Shape,
    workers: usize,
    max_rows: usize,
    max_assignments: usize,
    isa: Isa,
    fp8_slot: Option<Fp8Slot>,
    shared: Arc<PoolShared>,
    handles: Vec<JoinHandle<()>>,
}

pub struct Job<'a> {
    pub arena: Arc<HostBlocks>,
    pub hidden: &'a [bf16],
    pub rows: usize,
    pub experts: &'a [ExpertWeights],
    pub assignments: &'a [Assignment],
    pub out: &'a mut [bf16],
}

/// A published pool job; dropping it waits for completion and releases the lease.
/// `ExpertPool::finish` is the form that reports a worker error.
pub struct PendingJob<'a> {
    pool: &'a ExpertPool,
    out: Option<&'a mut [bf16]>,
    armed: bool,
}

impl Drop for PendingJob<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        if let Some(out) = self.out.take() {
            let _ = self.pool.complete(out);
        }
    }
}

struct PoolShared {
    job: RwLock<JobSlot>,
    park: Mutex<()>,
    wake: Condvar,
    error: Mutex<Option<String>>,
    workers: usize,
    swiglu_rounding: crate::model_spec::SwigluRounding,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
    generation: AtomicU64,
    pending: AtomicBool,
    shutdown: AtomicBool,
    phase1: AtomicUsize,
    finished: AtomicUsize,
    failed: AtomicBool,
    activated: Box<[AtomicU16]>,
    out: Box<[AtomicU16]>,
}

struct JobSlot {
    rows: usize,
    arena: Option<Arc<BlockLease>>,
    hidden: Vec<bf16>,
    experts: Vec<ExpertWeights>,
    assignments: Vec<Assignment>,
}

impl JobSlot {
    fn new(shape: &Shape, max_rows: usize, max_assignments: usize) -> Self {
        Self {
            rows: 0,
            arena: None,
            hidden: vec![bf16::ZERO; max_rows * shape.hidden],
            experts: Vec::with_capacity(max_assignments),
            assignments: Vec::with_capacity(max_assignments),
        }
    }
}

struct Local {
    hidden_bf: Vec<bf16>,
    hidden_f: Vec<f32>,
    hidden_rows: Vec<bool>,
    experts: Vec<ExpertWeights>,
    items: Vec<Assignment>,
    activated_bf: Vec<bf16>,
    act_f: Vec<f32>,
    block: Vec<bf16>,
}

impl Local {
    fn new(shape: &Shape, workers: usize, max_rows: usize, max_assignments: usize) -> Self {
        let block = (max_assignments * shape.hidden.max(shape.intermediate)).div_ceil(workers);
        Self {
            hidden_bf: vec![bf16::ZERO; max_rows * shape.hidden],
            hidden_f: vec![0.0; max_rows * shape.hidden],
            hidden_rows: vec![false; max_rows],
            experts: Vec::with_capacity(max_assignments),
            items: Vec::with_capacity(max_assignments),
            activated_bf: vec![bf16::ZERO; max_assignments * shape.intermediate],
            act_f: vec![0.0; shape.intermediate],
            block: vec![bf16::ZERO; block],
        }
    }
}

/// Host block of every assignment expert; each expert's slices must stay inside one block.
fn assignment_blocks(
    arena: &HostBlocks,
    experts: &[ExpertWeights],
    assignments: &[Assignment],
) -> Result<Vec<u32>> {
    let per_block = arena.block_elements();
    let mut blocks = Vec::with_capacity(assignments.len());
    for (index, assignment) in assignments.iter().enumerate() {
        let weights = experts
            .get(assignment.expert as usize)
            .with_context(|| format!("assignment {index} expert {}", assignment.expert))?;
        let block = weights.gate_up.start / per_block;
        let end = (block + 1) * per_block;
        ensure!(
            weights.gate_up.end <= end
                && weights.down.start >= block * per_block
                && weights.down.end <= end,
            "expert {} weights span host blocks",
            assignment.expert
        );
        blocks.push(u32::try_from(block).context("host block index exceeds u32")?);
    }
    Ok(blocks)
}

fn block(index: usize, workers: usize, total: usize) -> std::ops::Range<usize> {
    let chunk = total.div_ceil(workers);
    let start = (index * chunk).min(total);
    let end = ((index + 1) * chunk).min(total);
    start..end
}

impl ExpertPool {
    pub fn new(
        workers: usize,
        shape: Shape,
        max_rows: usize,
        max_assignments: usize,
        fp8_slot: Option<Fp8Slot>,
    ) -> Result<Self> {
        Self::new_with_swiglu_rounding(
            workers,
            shape,
            max_rows,
            max_assignments,
            fp8_slot,
            crate::model_spec::SwigluRounding::Twice,
        )
    }

    pub fn new_with_swiglu_rounding(
        workers: usize,
        shape: Shape,
        max_rows: usize,
        max_assignments: usize,
        fp8_slot: Option<Fp8Slot>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
    ) -> Result<Self> {
        Self::new_with_boundaries(
            workers,
            shape,
            max_rows,
            max_assignments,
            fp8_slot,
            swiglu_rounding,
            crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        )
    }

    pub fn new_with_boundaries(
        workers: usize,
        shape: Shape,
        max_rows: usize,
        max_assignments: usize,
        fp8_slot: Option<Fp8Slot>,
        swiglu_rounding: crate::model_spec::SwigluRounding,
        routed_weight: crate::model_spec::RoutedWeightBoundary,
    ) -> Result<Self> {
        ensure!(
            workers > 0 && workers <= 256,
            "invalid expert worker count {workers}"
        );
        ensure!(max_rows > 0, "zero expert row capacity");
        ensure!(
            max_assignments > 0 && max_assignments <= 1 << 16,
            "invalid expert assignment capacity {max_assignments}"
        );
        let isa = Isa::detect();
        if let Some(slot) = &fp8_slot {
            let geometry = ExpertGeometry {
                first_layer: 0,
                layers: 1,
                experts: 1,
                hidden: u32::try_from(shape.hidden)?,
                intermediate: u32::try_from(shape.intermediate)?,
                top_k: 1,
            };
            ensure!(
                ExpertLayout::new(crate::model_spec::ExpertFormat::Fp8W8a16, &geometry)?
                    == ExpertLayout::Fp8W8a16(slot.clone()),
                "FP8 pool slot differs from the expert shape"
            );
        }
        let shared = Arc::new(PoolShared {
            job: RwLock::new(JobSlot::new(&shape, max_rows, max_assignments)),
            park: Mutex::new(()),
            wake: Condvar::new(),
            error: Mutex::new(None),
            workers,
            swiglu_rounding,
            routed_weight,
            generation: AtomicU64::new(0),
            pending: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            phase1: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            activated: (0..max_assignments * shape.intermediate)
                .map(|_| AtomicU16::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            out: (0..max_assignments * shape.hidden)
                .map(|_| AtomicU16::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        });
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let worker_shared = shared.clone();
            let worker_fp8 = fp8_slot.clone();
            let spawned = thread::Builder::new()
                .name(format!("expert-ffn-{index}"))
                .spawn(move || {
                    worker(
                        index,
                        worker_shared,
                        shape,
                        isa,
                        max_rows,
                        max_assignments,
                        worker_fp8,
                    )
                });
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    {
                        let _guard = shared.park.lock();
                        shared.shutdown.store(true, Ordering::Release);
                        shared.wake.notify_all();
                    }
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(error).context("spawn expert worker");
                }
            }
        }
        Ok(Self {
            shape,
            workers,
            max_rows,
            max_assignments,
            isa,
            fp8_slot,
            shared,
            handles,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// Publishes a job; the guard holds the output and the lease until finish or drop.
    pub fn submit<'a>(&'a self, job: Job<'a>) -> Result<PendingJob<'a>> {
        ensure!(
            !self.shared.shutdown.load(Ordering::Acquire),
            "expert pool worker synchronization failed"
        );
        let Job {
            arena,
            hidden,
            rows,
            experts,
            assignments,
            out,
        } = job;
        ensure!(
            !self.shared.pending.swap(true, Ordering::AcqRel),
            "expert pool job is already pending"
        );
        if let Err(error) = self.publish(&arena, hidden, rows, experts, assignments, out.len()) {
            self.shared.pending.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(PendingJob {
            pool: self,
            out: Some(out),
            armed: true,
        })
    }

    fn publish(
        &self,
        arena: &Arc<HostBlocks>,
        hidden: &[bf16],
        rows: usize,
        experts: &[ExpertWeights],
        assignments: &[Assignment],
        out_len: usize,
    ) -> Result<()> {
        self.validate(
            arena.elements(),
            hidden.len(),
            rows,
            experts,
            assignments,
            out_len,
        )?;
        let blocks = assignment_blocks(arena, experts, assignments)?;
        let lease = Arc::new(arena.lease(&blocks)?);
        let mut error = self
            .shared
            .error
            .lock()
            .map_err(|_| anyhow::anyhow!("expert pool error mutex poisoned"))?;
        let _guard = self
            .shared
            .park
            .lock()
            .map_err(|_| anyhow::anyhow!("expert pool park mutex poisoned"))?;
        {
            let mut slot = self
                .shared
                .job
                .write()
                .map_err(|_| anyhow::anyhow!("expert pool job lock poisoned"))?;
            slot.rows = rows;
            slot.arena = Some(lease);
            slot.hidden[..hidden.len()].copy_from_slice(hidden);
            slot.experts.clear();
            slot.experts.extend_from_slice(experts);
            slot.assignments.clear();
            slot.assignments.extend_from_slice(assignments);
        }
        *error = None;
        self.shared.phase1.store(0, Ordering::Release);
        self.shared.finished.store(0, Ordering::Release);
        self.shared.failed.store(false, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Waits for the pending job and copies its output; on error the output is left unmodified.
    pub fn finish(&self, mut pending: PendingJob<'_>) -> Result<()> {
        debug_assert!(std::ptr::eq(self, pending.pool));
        pending.armed = false;
        let out = pending.out.take().context("pending job has no output")?;
        self.complete(out)
    }

    pub fn run(&self, job: Job<'_>) -> Result<()> {
        let pending = self.submit(job)?;
        self.finish(pending)
    }

    fn complete(&self, out: &mut [bf16]) -> Result<()> {
        let mut spins = 0u32;
        loop {
            ensure!(
                !self.shared.shutdown.load(Ordering::Acquire),
                "expert pool worker synchronization failed"
            );
            if self.shared.finished.load(Ordering::Acquire) >= self.workers {
                break;
            }
            if spins < SPIN_ROUNDS {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let mut guard = self
                .shared
                .park
                .lock()
                .map_err(|_| anyhow::anyhow!("expert pool park mutex poisoned"))?;
            while self.shared.finished.load(Ordering::Acquire) < self.workers
                && !self.shared.shutdown.load(Ordering::Acquire)
            {
                guard =
                    self.shared.wake.wait(guard).map_err(|_| {
                        anyhow::anyhow!("expert pool completion wait mutex poisoned")
                    })?;
            }
        }
        ensure!(
            !self.shared.shutdown.load(Ordering::Acquire),
            "expert pool worker synchronization failed"
        );
        if let Some(error) = self
            .shared
            .error
            .lock()
            .map_err(|_| anyhow::anyhow!("expert pool error mutex poisoned"))?
            .take()
        {
            self.release()?;
            bail!("{error}");
        }
        let len = out.len();
        for (target, source) in out.iter_mut().zip(&self.shared.out[..len]) {
            *target = bf16::from_bits(source.load(Ordering::Acquire));
        }
        self.release()?;
        Ok(())
    }

    fn release(&self) -> Result<()> {
        {
            let mut slot = self
                .shared
                .job
                .write()
                .map_err(|_| anyhow::anyhow!("expert pool job lock poisoned at release"))?;
            slot.arena = None;
        }
        self.shared.pending.store(false, Ordering::Release);
        Ok(())
    }

    fn validate(
        &self,
        arena_len: usize,
        hidden_len: usize,
        rows: usize,
        experts: &[ExpertWeights],
        assignments: &[Assignment],
        out_len: usize,
    ) -> Result<()> {
        let hidden = self.shape.hidden;
        ensure!(
            rows > 0 && rows <= self.max_rows,
            "rows {rows} exceed the pool's {}",
            self.max_rows
        );
        ensure!(
            hidden_len == rows * hidden,
            "hidden extent differs from rows * hidden"
        );
        ensure!(
            assignments.len() <= self.max_assignments,
            "assignments exceed the pool's {}",
            self.max_assignments
        );
        ensure!(
            experts.len() <= self.max_assignments,
            "experts exceed the pool's {}",
            self.max_assignments
        );
        ensure!(
            out_len.is_multiple_of(hidden) && out_len >= hidden,
            "out extent is not a whole number of rows"
        );
        let out_rows = out_len / hidden;
        ensure!(
            out_rows <= self.max_assignments,
            "out extent exceeds the pool's {} slots",
            self.max_assignments
        );
        for (index, weights) in experts.iter().enumerate() {
            weights
                .validate(&self.shape, arena_len, self.fp8_slot.as_ref())
                .with_context(|| format!("expert {index} weights"))?;
        }
        for (index, assignment) in assignments.iter().enumerate() {
            ensure!(
                (assignment.row as usize) < rows,
                "assignment {index} row {} outside {rows}",
                assignment.row
            );
            ensure!(
                (assignment.expert as usize) < experts.len(),
                "assignment {index} expert {} outside {}",
                assignment.expert,
                experts.len()
            );
            ensure!(
                (assignment.choice as usize) < out_rows,
                "assignment {index} choice {} outside {out_rows}",
                assignment.choice
            );
            for other in &assignments[..index] {
                ensure!(
                    other.choice != assignment.choice,
                    "duplicate choice slot {}",
                    assignment.choice
                );
            }
        }
        Ok(())
    }
}

impl Drop for ExpertPool {
    fn drop(&mut self) {
        {
            let guard = self.shared.park.lock();
            if guard.is_err() {
                eprintln!("expert pool park mutex poisoned during drop");
            }
            self.shared.shutdown.store(true, Ordering::Release);
            self.shared.wake.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker(
    index: usize,
    shared: Arc<PoolShared>,
    shape: Shape,
    isa: Isa,
    max_rows: usize,
    max_assignments: usize,
    fp8_slot: Option<Fp8Slot>,
) {
    let mut local = Local::new(&shape, shared.workers, max_rows, max_assignments);
    let phases = if fp8_slot.is_some() {
        phases::<true>
    } else {
        phases::<false>
    };
    let result = (|| -> Result<()> {
        let mut seen = 0u64;
        loop {
            let mut spins = 0u32;
            loop {
                if shared.shutdown.load(Ordering::Acquire) {
                    return Ok(());
                }
                let current = shared.generation.load(Ordering::Acquire);
                if current != seen {
                    seen = current;
                    break;
                }
                if spins < SPIN_ROUNDS {
                    spins += 1;
                    std::hint::spin_loop();
                    continue;
                }
                let mut guard = shared
                    .park
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert worker park mutex poisoned"))?;
                while shared.generation.load(Ordering::Acquire) == seen
                    && !shared.shutdown.load(Ordering::Acquire)
                {
                    guard = shared
                        .wake
                        .wait(guard)
                        .map_err(|_| anyhow::anyhow!("expert worker wake mutex poisoned"))?;
                }
                drop(guard);
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                phases(
                    index,
                    &shared,
                    &shape,
                    isa,
                    &mut local,
                    fp8_slot.as_ref(),
                    shared.swiglu_rounding,
                )
            }));
            let message = match outcome {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error.to_string()),
                Err(_) => Some("expert worker panicked".to_string()),
            };
            if let Some(message) = message {
                let mut error = shared
                    .error
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert worker error mutex poisoned"))?;
                if error.is_none() {
                    *error = Some(message);
                }
                shared.failed.store(true, Ordering::Release);
            }
            let done = shared.finished.fetch_add(1, Ordering::AcqRel) + 1;
            if done == shared.workers {
                let _guard = shared
                    .park
                    .lock()
                    .map_err(|_| anyhow::anyhow!("expert worker completion mutex poisoned"))?;
                shared.wake.notify_all();
            }
        }
    })();
    if let Err(failure) = result {
        shared.failed.store(true, Ordering::Release);
        match shared.error.lock() {
            Ok(mut error) => {
                if error.is_none() {
                    *error = Some(format!("{failure:#}"));
                }
            }
            Err(_) => eprintln!("expert worker error mutex poisoned: {failure:#}"),
        }
        let _guard = shared.park.lock();
        shared.shutdown.store(true, Ordering::Release);
        shared.wake.notify_all();
    }
}

pub(crate) fn native_bytes(values: &[bf16]) -> &[u8] {
    // BF16 storage contains initialized bytes and the caller holds the block read lease.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn phases<const FP8: bool>(
    index: usize,
    shared: &PoolShared,
    shape: &Shape,
    isa: Isa,
    local: &mut Local,
    fp8_slot: Option<&Fp8Slot>,
    swiglu_rounding: crate::model_spec::SwigluRounding,
) -> Result<()> {
    let arm = arm_for(isa)?;
    let (hidden, intermediate) = (shape.hidden, shape.intermediate);
    let workers = shared.workers;
    let (lease, items_len, range1, rows) = {
        let slot = shared
            .job
            .read()
            .map_err(|_| anyhow::anyhow!("expert worker job lock poisoned"))?;
        local.experts.clear();
        local.experts.extend_from_slice(&slot.experts);
        local.items.clear();
        local.items.extend_from_slice(&slot.assignments);
        let items_len = slot.assignments.len();
        let range1 = block(index, workers, items_len * intermediate);
        local.hidden_rows[..slot.rows].fill(false);
        if range1.start < range1.end {
            let first = range1.start / intermediate;
            let last = (range1.end - 1) / intermediate;
            for item in &slot.assignments[first..=last] {
                let row = item.row as usize;
                if !local.hidden_rows[row] {
                    local.hidden_bf[row * hidden..][..hidden]
                        .copy_from_slice(&slot.hidden[row * hidden..][..hidden]);
                    local.hidden_rows[row] = true;
                }
            }
        }
        let lease = slot.arena.clone().context("job lease missing")?;
        (lease, items_len, range1, slot.rows)
    };
    for row in 0..rows {
        if local.hidden_rows[row] {
            for (value, source) in local.hidden_f[row * hidden..][..hidden]
                .iter_mut()
                .zip(&local.hidden_bf[row * hidden..][..hidden])
            {
                *value = f32::from(*source);
            }
        }
    }
    for unit in range1.clone() {
        let item = &local.items[unit / intermediate];
        let j = unit % intermediate;
        let expert = &local.experts[item.expert as usize];
        let block_elements = lease.block_elements();
        let block = expert.gate_up.start / block_elements;
        let data = lease.read(u32::try_from(block)?)?;
        let gate_up = &data[expert.gate_up.start % block_elements..]
            [..expert.gate_up.end - expert.gate_up.start];
        let x = &local.hidden_f[item.row as usize * hidden..][..hidden];
        let (gate, up) = if FP8 {
            let slot = fp8_slot.context("FP8 worker slot missing")?;
            let bytes = native_bytes(gate_up);
            let gate = &bytes[slot.gate.start as usize + j * hidden..][..hidden];
            let up = &bytes[slot.up.start as usize + j * hidden..][..hidden];
            let gate_scale = row_scale(&bytes[slot.gate_scale.start as usize..], j);
            let up_scale = row_scale(&bytes[slot.up_scale.start as usize..], j);
            unsafe {
                (
                    (arm.dot_e4m3)(x, gate, gate_scale),
                    (arm.dot_e4m3)(x, up, up_scale),
                )
            }
        } else {
            unsafe {
                (
                    (arm.dot)(x, &gate_up[j * expert.gate_up_stride..][..hidden]),
                    (arm.dot)(
                        x,
                        &gate_up[(intermediate + j) * expert.gate_up_stride..][..hidden],
                    ),
                )
            }
        };
        let gate = bf16::from_f32(gate);
        let up = bf16::from_f32(up);
        let gate = f32::from(gate);
        let up = f32::from(up);
        let activation = gate * (1.0 / (1.0 + (-gate).exp()));
        local.block[unit - range1.start] = bf16::from_f32(match swiglu_rounding {
            crate::model_spec::SwigluRounding::Twice => f32::from(bf16::from_f32(activation)) * up,
            crate::model_spec::SwigluRounding::Once => activation * up,
        });
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    for (offset, value) in local.block[..range1.len()].iter().enumerate() {
        shared.activated[range1.start + offset].store(value.to_bits(), Ordering::Relaxed);
    }
    shared.phase1.fetch_add(1, Ordering::Release);
    let mut spins = 0u32;
    while shared.phase1.load(Ordering::Acquire) < workers && !shared.failed.load(Ordering::Acquire)
    {
        if spins < 4096 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    let total2 = items_len * hidden;
    let range2 = block(index, workers, total2);
    if range2.start < range2.end {
        let first_item = range2.start / hidden;
        let last_item = (range2.end - 1) / hidden;
        let start = first_item * intermediate;
        let end = (last_item + 1) * intermediate;
        local.activated_bf.clear();
        for slot in &shared.activated[start..end] {
            local
                .activated_bf
                .push(bf16::from_bits(slot.load(Ordering::Relaxed)));
        }
        let mut cached = usize::MAX;
        for unit in range2.clone() {
            let item_index = unit / hidden;
            if cached != item_index {
                let start = (item_index - first_item) * intermediate;
                for (value, source) in local
                    .act_f
                    .iter_mut()
                    .zip(&local.activated_bf[start..start + intermediate])
                {
                    *value = f32::from(*source);
                }
                cached = item_index;
            }
            let item = &local.items[item_index];
            let expert = &local.experts[item.expert as usize];
            let block = expert.down.start / lease.block_elements();
            let data = lease.read(u32::try_from(block)?)?;
            let down = &data[expert.down.start % lease.block_elements()..]
                [..expert.down.end - expert.down.start];
            let k = unit % hidden;
            let value = if FP8 {
                let slot = fp8_slot.context("FP8 worker slot missing")?;
                let bytes = native_bytes(down);
                let row = &bytes[slot.down.start as usize + k * intermediate..][..intermediate];
                let scale = row_scale(&bytes[slot.down_scale.start as usize..], k);
                unsafe { (arm.dot_e4m3)(&local.act_f, row, scale) }
            } else {
                unsafe {
                    (arm.dot)(
                        &local.act_f,
                        &down[k * expert.down_stride..][..intermediate],
                    )
                }
            };
            let weight = f32::from(item.weight);
            let weighted = match shared.routed_weight {
                crate::model_spec::RoutedWeightBoundary::AfterDownBf16 => {
                    f32::from(bf16::from_f32(value)) * weight
                }
                crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16 => value * weight,
            };
            local.block[unit - range2.start] = bf16::from_f32(weighted);
        }
    }
    if shared.failed.load(Ordering::Acquire) {
        return Ok(());
    }
    for (offset, unit) in range2.clone().enumerate() {
        let item = &local.items[unit / hidden];
        let k = unit % hidden;
        shared.out[item.choice as usize * hidden + k]
            .store(local.block[offset].to_bits(), Ordering::Relaxed);
    }
    Ok(())
}

pub fn measure_service_table(
    shape: Shape,
    ring_experts: u32,
    workers: &[u32],
    rows: &[u32],
    missed: &[u32],
    format: crate::model_spec::ExpertFormat,
) -> Result<Vec<crate::experts::plan::ServiceCell>> {
    Shape::new(shape.hidden, shape.intermediate)?;
    let rows_max = *rows.iter().max().context("empty service row lattice")? as usize;
    let missed_max = *missed.iter().max().context("empty service miss lattice")? as usize;
    ensure!(
        !workers.is_empty()
            && workers.iter().all(|&n| n > 0 && n <= 256)
            && rows.iter().all(|&n| n > 0)
            && missed.iter().all(|&n| n > 0),
        "invalid expert service lattice: workers={workers:?} rows={rows:?} missed={missed:?}"
    );
    ensure!(
        ring_experts as usize >= missed_max,
        "expert service ring {ring_experts} cannot hold {missed_max} misses"
    );
    let geometry = ExpertGeometry {
        first_layer: 0,
        layers: 1,
        experts: ring_experts,
        hidden: u32::try_from(shape.hidden)?,
        intermediate: u32::try_from(shape.intermediate)?,
        top_k: u32::try_from(missed_max)?,
    };
    let layout = ExpertLayout::new(format, &geometry)?;
    let per = usize::try_from(layout.slot_bytes() / 2)?;
    let max_assignments = rows_max
        .checked_mul(missed_max)
        .context("service assignment extent overflow")?;
    ensure!(
        max_assignments <= 1 << 16,
        "service assignments {max_assignments} exceed pool capacity {}",
        1 << 16
    );
    let arena = HostBlocks::new(ring_experts, per)?;
    let mut values = vec![bf16::ZERO; per];
    match &layout {
        ExpertLayout::Bf16Fused { .. } | ExpertLayout::Bf16Separate { .. } => {
            for (index, value) in values.iter_mut().enumerate() {
                *value = bf16::from_f32((((index % 251) as f32) - 125.0) / 128.0);
            }
        }
        ExpertLayout::Fp8W8a16(slot) => {
            for (word, value) in values[..slot.gate_scale.start as usize / 2]
                .iter_mut()
                .enumerate()
            {
                let code =
                    |index| (((index * 37 + 11) % 64) as u8) | ((((index / 7) & 1) as u8) << 7);
                *value = bf16::from_le_bytes([code(2 * word), code(2 * word + 1)]);
            }
            for (row, value) in values[slot.gate_scale.start as usize / 2..]
                .iter_mut()
                .enumerate()
            {
                *value = bf16::from_f32((1.0 + (row % 19) as f32 / 32.0) / 8.0);
            }
        }
    }
    for expert in 0..ring_experts {
        arena.claim(expert)?.slice()?.copy_from_slice(&values);
    }
    let hidden: Vec<bf16> = (0..rows_max * shape.hidden)
        .map(|k| bf16::from_f32(((k as f64 * 0.6180339887498949 + 0.9).sin() * 0.75) as f32))
        .collect();
    let mut cells = Vec::new();
    let mut cursor = 0usize;
    for &workers in workers {
        let pool = ExpertPool::new(
            workers as usize,
            shape,
            rows_max,
            max_assignments,
            layout.fp8_slot().cloned(),
        )?;
        for &rows in rows {
            let row_started = std::time::Instant::now();
            for &missed in missed {
                let assignments: Vec<Assignment> = (0..rows * missed)
                    .map(|slot| Assignment {
                        row: slot / missed,
                        choice: slot,
                        expert: slot % missed,
                        weight: bf16::from_f32(0.125 * ((slot % missed) as f32 + 1.0)),
                    })
                    .collect();
                let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                let mut samples = [0.0; 5];
                for trial in 0..6 {
                    cursor = (cursor + missed as usize) % ring_experts as usize;
                    let selected: Vec<ExpertWeights> = (0..missed as usize)
                        .map(|choice| {
                            let start = ((cursor + choice) % ring_experts as usize) * per;
                            ExpertWeights::from_slot(shape, start..start + per, &layout)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let started = std::time::Instant::now();
                    pool.run(Job {
                        arena: arena.clone(),
                        hidden: &hidden[..rows as usize * shape.hidden],
                        rows: rows as usize,
                        experts: &selected,
                        assignments: &assignments,
                        out: &mut out,
                    })?;
                    let seconds = started.elapsed().as_secs_f64();
                    ensure!(
                        out.iter().all(|x| x.to_f32().is_finite()),
                        "non-finite CPU service output: workers={workers} rows={rows} missed={missed}"
                    );
                    if trial > 0 {
                        samples[trial - 1] = seconds;
                    }
                }
                samples.sort_by(f64::total_cmp);
                cells.push(crate::experts::plan::ServiceCell {
                    workers,
                    rows,
                    experts: missed,
                    seconds: samples[2],
                });
            }
            eprintln!(
                "{}",
                serde_json::json!({"expert_service_setup":{"workers":workers,"rows":rows,"cells":missed.len(),"wall_seconds":row_started.elapsed().as_secs_f64()}})
            );
        }
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 2048;
    const I: usize = 512;

    fn production_shape() -> Shape {
        Shape::new(H, I).unwrap()
    }

    #[test]
    fn separate_bf16_slots_cover_two_gate_up_matrices_and_one_down_matrix() -> Result<()> {
        let shape = production_shape();
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: 2,
            hidden: H as u32,
            intermediate: I as u32,
            top_k: 2,
        };
        let layout = ExpertLayout::new(crate::model_spec::ExpertFormat::Bf16Separate, &geometry)?;
        let words = usize::try_from(layout.slot_bytes() / 2)?;
        let weights = ExpertWeights::from_slot(shape, 0..words, &layout)?;
        weights.validate(&shape, words, None)?;
        let matrix_words = shape.intermediate * shape.hidden;
        assert_eq!(weights.gate_up, 0..2 * matrix_words);
        assert_eq!(weights.down, 2 * matrix_words..3 * matrix_words);
        Ok(())
    }

    fn pattern(seed: usize, phase: f64) -> bf16 {
        bf16::from_f32(((seed as f64 * 0.6180339887498949 + phase).sin() * 0.75) as f32)
    }

    fn per_expert(shape: &Shape) -> usize {
        2 * shape.intermediate * shape.hidden + shape.hidden * shape.intermediate
    }

    fn build_arena(shape: &Shape, experts: usize) -> Arc<HostBlocks> {
        let per = per_expert(shape);
        let arena = HostBlocks::new(u32::try_from(experts).unwrap(), per).unwrap();
        let mut values = vec![bf16::ZERO; per];
        for expert in 0..experts {
            for j in 0..2 * shape.intermediate {
                for k in 0..shape.hidden {
                    values[j * shape.hidden + k] =
                        pattern(expert * 7919 + j * shape.hidden + k, 0.11);
                }
            }
            let down = 2 * shape.intermediate * shape.hidden;
            for j in 0..shape.hidden {
                for k in 0..shape.intermediate {
                    values[down + j * shape.intermediate + k] =
                        pattern(expert * 104729 + j * shape.intermediate + k, 1.7);
                }
            }
            write_block(&arena, u32::try_from(expert).unwrap(), &values);
        }
        arena
    }

    fn write_block(arena: &Arc<HostBlocks>, block: u32, values: &[bf16]) {
        let mut claim = arena.claim(block).unwrap();
        claim.slice().unwrap().copy_from_slice(values);
    }

    fn arena_flat(arena: &Arc<HostBlocks>) -> Vec<bf16> {
        let blocks: Vec<u32> = (0..arena.blocks()).collect();
        let lease = arena.lease(&blocks).unwrap();
        let mut flat = Vec::with_capacity(arena.elements());
        for block in blocks {
            flat.extend_from_slice(lease.read(block).unwrap());
        }
        flat
    }

    fn weights(shape: &Shape, expert: usize) -> ExpertWeights {
        let per = per_expert(shape);
        let base = expert * per;
        let gate_up_len = 2 * shape.intermediate * shape.hidden;
        ExpertWeights {
            gate_up: base..base + gate_up_len,
            gate_up_stride: shape.hidden,
            down: base + gate_up_len..base + per,
            down_stride: shape.intermediate,
        }
    }

    fn hidden_row(shape: &Shape, row: usize) -> Vec<bf16> {
        (0..shape.hidden)
            .map(|k| pattern(row * 4099 + k, 2.3))
            .collect()
    }

    /// FP64 accumulation with the model's BF16 rounding points (f64 → f32 → bf16).
    fn reference(
        shape: &Shape,
        arena: &[bf16],
        weights: &ExpertWeights,
        hidden: &[bf16],
        routing: bf16,
    ) -> Vec<bf16> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let gate_up = &arena[weights.gate_up.clone()];
        let mut gate = vec![0.0f64; i];
        let mut up = vec![0.0f64; i];
        for j in 0..i {
            for k in 0..h {
                gate[j] +=
                    f64::from(hidden[k]) * f64::from(gate_up[j * weights.gate_up_stride + k]);
                up[j] +=
                    f64::from(hidden[k]) * f64::from(gate_up[(i + j) * weights.gate_up_stride + k]);
            }
        }
        let mut activated = vec![0.0f64; i];
        for j in 0..i {
            let gate = f64::from(bf16::from_f32(gate[j] as f32));
            let up = f64::from(bf16::from_f32(up[j] as f32));
            let silu = f64::from(bf16::from_f32(
                (gate * (1.0 / (1.0 + (-gate).exp()))) as f32,
            ));
            activated[j] = f64::from(bf16::from_f32((silu * up) as f32));
        }
        let down = &arena[weights.down.clone()];
        let mut out = Vec::with_capacity(h);
        for k in 0..h {
            let mut acc = 0.0f64;
            for j in 0..i {
                acc += activated[j] * f64::from(down[k * weights.down_stride + j]);
            }
            let linear = bf16::from_f32(acc as f32);
            out.push(bf16::from_f32(f32::from(linear) * f32::from(routing)));
        }
        out
    }

    fn order_key(value: bf16) -> u32 {
        let bits = u32::from(value.to_bits());
        if bits & 0x8000 != 0 {
            (!bits) & 0xffff
        } else {
            bits | 0x8000
        }
    }

    #[test]
    fn native_expert_silu_boundary_is_per_model() -> Result<()> {
        let shape = production_shape();
        let (h, i) = (shape.hidden, shape.intermediate);
        let mut flat = vec![bf16::ZERO; per_expert(&shape)];
        let gate = -0.015625f32;
        let up = 1.5;
        flat[0] = bf16::from_f32(gate);
        flat[i * h] = bf16::from_f32(up);
        for row in 0..h {
            flat[2 * i * h + row * i] = bf16::ONE;
        }
        let mut hidden = vec![bf16::ZERO; h];
        hidden[0] = bf16::ONE;
        let silu = gate / (1.0 + (-gate).exp());
        let twice = bf16::from_f32(f32::from(bf16::from_f32(silu)) * up);
        let once = bf16::from_f32(silu * up);
        assert_ne!(twice.to_bits(), once.to_bits());
        let expected = twice;
        let expert = weights(&shape, 0);
        let mut scratch = Scratch::new(&shape);
        let mut output = vec![bf16::ZERO; h];
        let mut mismatches = 0;
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            contribution(
                isa,
                Contribution {
                    shape: &shape,
                    arena: &flat,
                    weights: &expert,
                    hidden: &hidden,
                    routing: bf16::ONE,

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut output,
                &mut scratch,
            )?;
            let count = output
                .iter()
                .filter(|v| v.to_bits() != expected.to_bits())
                .count();
            println!(
                "expert_silu_boundary arm={} bit_mismatches={count}",
                isa.name()
            );
            mismatches += count;
        }
        let arena = HostBlocks::new(1, flat.len())?;
        arena.claim(0)?.slice()?.copy_from_slice(&flat);
        let pool = ExpertPool::new(2, shape, 1, 1, None)?;
        let assignments = [Assignment {
            row: 0,
            choice: 0,
            expert: 0,
            weight: bf16::ONE,
        }];
        let experts = [expert.clone()];
        let pending = pool.submit(Job {
            arena,
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &assignments,
            out: &mut output,
        })?;
        pool.finish(pending)?;
        let count = output
            .iter()
            .filter(|v| v.to_bits() != expected.to_bits())
            .count();
        println!("expert_silu_boundary arm=pool bit_mismatches={count}");
        assert_eq!(mismatches + count, 0);
        let (mut native, slot) = fp8_fixture(&shape);
        native.fill(0);
        for bytes in native[slot.gate_scale.start as usize..]
            .as_chunks_mut::<2>()
            .0
        {
            bytes.copy_from_slice(&bf16::ONE.to_le_bytes());
        }
        native[0] = 0x88;
        native[slot.up.start as usize] = 0x3c;
        for row in 0..h {
            native[slot.down.start as usize + row * i] = 0x38;
        }
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            fp8_contribution(
                isa,
                Fp8Contribution {
                    shape: &shape,
                    slot: &native,
                    slot_layout: &slot,
                    hidden: &hidden,
                    routing: bf16::ONE,

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut output,
                &mut scratch,
            )?;
            assert!(output.iter().all(|v| v.to_bits() == expected.to_bits()));
            println!(
                "expert_silu_boundary arm=fp8_{} bit_mismatches=0",
                isa.name()
            );
        }
        let arena = HostBlocks::new(1, native.len() / 2)?;
        {
            let mut claim = arena.claim(0)?;
            for (value, bytes) in claim.slice()?.iter_mut().zip(native.as_chunks::<2>().0) {
                *value = bf16::from_le_bytes(*bytes);
            }
        }
        let layout = ExpertLayout::Fp8W8a16(slot.clone());
        let experts = [ExpertWeights::from_slot(
            shape,
            arena.block_range(0)?,
            &layout,
        )?];
        let pool = ExpertPool::new(2, shape, 1, 1, Some(slot.clone()))?;
        pool.run(Job {
            arena,
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &assignments,
            out: &mut output,
        })?;
        assert!(output.iter().all(|v| v.to_bits() == expected.to_bits()));
        println!("expert_silu_boundary arm=fp8_pool bit_mismatches=0");
        let expected = once;
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            contribution(
                isa,
                Contribution {
                    shape: &shape,
                    arena: &flat,
                    weights: &expert,
                    hidden: &hidden,
                    routing: bf16::ONE,
                    rounding: crate::model_spec::SwigluRounding::Once,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
                },
                &mut output,
                &mut scratch,
            )?;
            assert!(output.iter().all(|v| v.to_bits() == expected.to_bits()));
            println!(
                "expert_silu_boundary_once arm={} bit_mismatches=0",
                isa.name()
            );
        }
        let arena = HostBlocks::new(1, flat.len())?;
        arena.claim(0)?.slice()?.copy_from_slice(&flat);
        let once_experts = [expert.clone()];
        let pool = ExpertPool::new_with_swiglu_rounding(
            2,
            shape,
            1,
            1,
            None,
            crate::model_spec::SwigluRounding::Once,
        )?;
        pool.run(Job {
            arena,
            hidden: &hidden,
            rows: 1,
            experts: &once_experts,
            assignments: &assignments,
            out: &mut output,
        })?;
        assert!(output.iter().all(|v| v.to_bits() == expected.to_bits()));
        println!("expert_silu_boundary_once arm=pool bit_mismatches=0");
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            fp8_contribution(
                isa,
                Fp8Contribution {
                    shape: &shape,
                    slot: &native,
                    slot_layout: &slot,
                    hidden: &hidden,
                    routing: bf16::ONE,
                    rounding: crate::model_spec::SwigluRounding::Once,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
                },
                &mut output,
                &mut scratch,
            )?;
            assert!(output.iter().all(|v| v.to_bits() == expected.to_bits()));
            println!(
                "expert_silu_boundary_once arm=fp8_{} bit_mismatches=0",
                isa.name()
            );
        }
        flat[h] = bf16::from_f32(-3.0);
        flat[(i + 1) * h] = bf16::from_f32(0.7);
        for row in 0..h {
            flat[2 * i * h + row * i + 1] = bf16::ONE;
        }
        let routing = bf16::from_f32(0.1);
        let mut after = vec![bf16::ZERO; h];
        let mut before = vec![bf16::ZERO; h];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &flat,
                weights: &expert,
                hidden: &hidden,
                routing,
                rounding: crate::model_spec::SwigluRounding::Twice,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
            &mut after,
            &mut scratch,
        )?;
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &flat,
                weights: &expert,
                hidden: &hidden,
                routing,
                rounding: crate::model_spec::SwigluRounding::Twice,
                routed_weight: crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
            },
            &mut before,
            &mut scratch,
        )?;
        assert_ne!(after, before);
        println!("expert_routed_weight_boundary bit_mismatches>0");
        Ok(())
    }

    #[test]
    fn scalar_matches_the_fp64_reference_at_the_bf16_boundaries() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let mut scratch = Scratch::new(&shape);
        let mut elements = 0usize;
        let mut differing = 0usize;
        let mut worst = 0u32;
        for (expert, routing) in [(0usize, 0.3125f32), (1, -0.5), (2, 0.0625), (3, 1.0)] {
            let hidden = hidden_row(&shape, expert);
            let routing = bf16::from_f32(routing);
            let mut out = vec![bf16::ZERO; shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &weights(&shape, expert),
                    hidden: &hidden,
                    routing,
                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut out,
                &mut scratch,
            )?;
            let expected = reference(
                &shape,
                &arena_flat(&arena),
                &weights(&shape, expert),
                &hidden,
                routing,
            );
            for (actual, expected) in out.iter().zip(&expected) {
                let distance = order_key(*actual).abs_diff(order_key(*expected));
                elements += 1;
                differing += usize::from(distance != 0);
                worst = worst.max(distance);
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "scalar_vs_fp64": {
                    "elements": elements,
                    "differing": differing,
                    "max_bf16_steps": worst,
                }
            })
        );
        assert_eq!(worst, 0, "FP32 and FP64 rounded contributions differ");
        Ok(())
    }

    fn fp8_fixture(shape: &Shape) -> (Vec<u8>, crate::experts::catalogue::Fp8Slot) {
        let geometry = crate::experts::catalogue::ExpertGeometry {
            first_layer: 0,
            layers: 1,
            experts: 1,
            hidden: shape.hidden as u32,
            intermediate: shape.intermediate as u32,
            top_k: 1,
        };
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Fp8W8a16,
            &geometry,
        )
        .unwrap();
        let crate::experts::catalogue::ExpertLayout::Fp8W8a16(slot_layout) = layout else {
            panic!("FP8 layout expected");
        };
        let mut slot = vec![0u8; slot_layout.down_scale.end as usize];
        let codes = 3 * shape.hidden * shape.intermediate;
        for (index, byte) in slot.iter_mut().take(codes).enumerate() {
            *byte = (((index * 37 + 11) % 64) as u8) | ((((index / 7) & 1) as u8) << 7);
        }
        for (range, rows) in [
            (&slot_layout.gate_scale, shape.intermediate),
            (&slot_layout.up_scale, shape.intermediate),
            (&slot_layout.down_scale, shape.hidden),
        ] {
            for row in 0..rows {
                let value = bf16::from_f32(0.5 + (row % 9) as f32 * 0.125);
                let at = range.start as usize + 2 * row;
                slot[at..at + 2].copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        (slot, slot_layout)
    }

    #[test]
    fn fp8_effective_weight_rounds_to_bf16_before_the_dot() -> Result<()> {
        let scale = bf16::from_f32(1.0078125);
        let product = crate::fp8::decode_e4m3(0x3f) * scale.to_f32();
        let rounded = f32::from(crate::fp8::effective_weight(0x3f, scale));
        assert_eq!(product, 1.889_648_4);
        assert_eq!(rounded, 1.890_625);
        assert_ne!(product, rounded);
        let x = [1.0f32];
        let codes = [0x3f];
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                println!("arm {} unavailable", isa.name());
                continue;
            }
            // SAFETY: the arm was selected by `Isa::detect` or checked by `Isa::available`.
            let value = unsafe { (arm_for(isa)?.dot_e4m3)(&x, &codes, scale) };
            assert_eq!(value, rounded, "{} feeds the unrounded weight", isa.name());
        }
        Ok(())
    }

    /// FP64 accumulation over the dequantized weights with the model's BF16 rounding points.
    fn fp8_oracle(
        shape: &Shape,
        slot: &[u8],
        layout: &crate::experts::catalogue::Fp8Slot,
        hidden: &[bf16],
        routing: bf16,
    ) -> Vec<bf16> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let scale = |range: &std::ops::Range<u64>, row: usize| -> bf16 {
            let at = range.start as usize + 2 * row;
            bf16::from_bits(u16::from_le_bytes([slot[at], slot[at + 1]]))
        };
        let mut gate_w = vec![0.0f32; i * h];
        let mut up_w = vec![0.0f32; i * h];
        for j in 0..i {
            let gate_scale = scale(&layout.gate_scale, j);
            let up_scale = scale(&layout.up_scale, j);
            for k in 0..h {
                gate_w[j * h + k] = f32::from(crate::fp8::effective_weight(
                    slot[layout.gate.start as usize + j * h + k],
                    gate_scale,
                ));
                up_w[j * h + k] = f32::from(crate::fp8::effective_weight(
                    slot[layout.up.start as usize + j * h + k],
                    up_scale,
                ));
            }
        }
        let mut down_w = vec![0.0f32; h * i];
        for k in 0..h {
            let down_scale = scale(&layout.down_scale, k);
            for j in 0..i {
                down_w[k * i + j] = f32::from(crate::fp8::effective_weight(
                    slot[layout.down.start as usize + k * i + j],
                    down_scale,
                ));
            }
        }
        let mut gate = vec![0.0f64; i];
        let mut up = vec![0.0f64; i];
        for j in 0..i {
            for k in 0..h {
                gate[j] += f64::from(hidden[k]) * f64::from(gate_w[j * h + k]);
                up[j] += f64::from(hidden[k]) * f64::from(up_w[j * h + k]);
            }
        }
        let mut activated = vec![0.0f64; i];
        for j in 0..i {
            let gate = f64::from(bf16::from_f32(gate[j] as f32));
            let up = f64::from(bf16::from_f32(up[j] as f32));
            let silu = f64::from(bf16::from_f32(
                (gate * (1.0 / (1.0 + (-gate).exp()))) as f32,
            ));
            activated[j] = f64::from(bf16::from_f32((silu * up) as f32));
        }
        let mut out = Vec::with_capacity(h);
        for k in 0..h {
            let mut acc = 0.0f64;
            for j in 0..i {
                acc += activated[j] * f64::from(down_w[k * i + j]);
            }
            let linear = bf16::from_f32(acc as f32);
            out.push(bf16::from_f32(f32::from(linear) * f32::from(routing)));
        }
        out
    }

    #[test]
    fn fp8_scalar_matches_the_oracle_at_the_bf16_boundaries() -> Result<()> {
        let shape = production_shape();
        let (slot, layout) = fp8_fixture(&shape);
        let mut scratch = Scratch::new(&shape);
        let mut elements = 0usize;
        let mut differing = 0usize;
        let mut worst = 0u32;
        for (row, routing) in [(0usize, 0.3125f32), (1, -0.5), (2, 0.0625), (3, 1.0)] {
            let hidden = hidden_row(&shape, row);
            let routing = bf16::from_f32(routing);
            let mut out = vec![bf16::ZERO; shape.hidden];
            fp8_contribution(
                Isa::Scalar,
                Fp8Contribution {
                    shape: &shape,
                    slot: &slot,
                    slot_layout: &layout,
                    hidden: &hidden,
                    routing,
                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut out,
                &mut scratch,
            )?;
            let expected = fp8_oracle(&shape, &slot, &layout, &hidden, routing);
            for (actual, expected) in out.iter().zip(&expected) {
                let distance = order_key(*actual).abs_diff(order_key(*expected));
                elements += 1;
                differing += usize::from(distance != 0);
                worst = worst.max(distance);
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "fp8_scalar_vs_oracle": {
                    "elements": elements,
                    "differing": differing,
                    "max_bf16_steps": worst,
                }
            })
        );
        assert_eq!(worst, 0, "FP32 and FP64 rounded contributions differ");
        Ok(())
    }

    #[test]
    fn fp8_pool_matches_native_contributions_and_holds_block_leases() -> Result<()> {
        let shape = production_shape();
        let (codes, slot) = fp8_fixture(&shape);
        let layout = ExpertLayout::Fp8W8a16(slot.clone());
        let per = codes.len() / 2;
        let arena = HostBlocks::new(4, per)?;
        for block in [1, 3] {
            let mut claim = arena.claim(block)?;
            for (dst, bytes) in claim.slice()?.iter_mut().zip(codes.as_chunks::<2>().0) {
                *dst = bf16::from_le_bytes(*bytes);
            }
        }
        let experts = [1, 3]
            .map(|block| ExpertWeights::from_slot(shape, arena.block_range(block)?, &layout))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let hidden = [hidden_row(&shape, 0), hidden_row(&shape, 1)].concat();
        let assignments = [(1, 3, 1, 0.3125), (0, 0, 0, -0.5), (1, 5, 0, 0.125)].map(
            |(row, choice, expert, weight)| Assignment {
                row,
                choice,
                expert,
                weight: bf16::from_f32(weight),
            },
        );
        let mut expected = vec![bf16::ZERO; 6 * shape.hidden];
        let mut scratch = Scratch::new(&shape);
        for item in &assignments {
            fp8_contribution(
                Isa::Scalar,
                Fp8Contribution {
                    shape: &shape,
                    slot: &codes,
                    slot_layout: &slot,
                    hidden: &hidden[item.row as usize * shape.hidden..][..shape.hidden],
                    routing: item.weight,

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut expected[item.choice as usize * shape.hidden..][..shape.hidden],
                &mut scratch,
            )?;
        }
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for workers in [1, 4] {
            let pool = ExpertPool::new(workers, shape, 2, 6, Some(slot.clone()))?;
            let mut out = vec![bf16::ZERO; expected.len()];
            let pending = pool.submit(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 2,
                experts: &experts,
                assignments: &assignments,
                out: &mut out,
            })?;
            assert!(arena.claim(1).is_err() && arena.claim(3).is_err());
            assert!(arena.claim(0).is_ok());
            pool.finish(pending)?;
            assert_eq!(bits(&out), bits(&expected));
            assert!(arena.claim(1).is_ok() && arena.claim(3).is_ok());
            let mut malformed = experts.clone();
            malformed[0].gate_up.end += 1;
            assert!(
                pool.run(Job {
                    arena: arena.clone(),
                    hidden: &hidden,
                    rows: 2,
                    experts: &malformed,
                    assignments: &assignments,
                    out: &mut out,
                })
                .is_err()
            );
            assert_eq!(bits(&out), bits(&expected));
            println!(
                "PASS FP8_POOL stored_gate=[512,2048] stored_down=[2048,512] rows=2 choices=3 slots=6 workers={workers} isa={} bit_mismatches=0 leases=true bounds=true",
                pool.isa().name()
            );
        }
        {
            let mut claim = arena.claim(3)?;
            let bytes = claim.slice()?;
            bytes[slot.down_scale.start as usize / 2..slot.down_scale.end as usize / 2]
                .rotate_left(1);
        }
        let pool = ExpertPool::new(4, shape, 2, 6, Some(slot))?;
        let mut wrong = vec![bf16::ZERO; expected.len()];
        pool.run(Job {
            arena,
            hidden: &hidden,
            rows: 2,
            experts: &experts,
            assignments: &assignments,
            out: &mut wrong,
        })?;
        let mismatches = bits(&wrong)
            .iter()
            .zip(bits(&expected))
            .filter(|(a, b)| **a != *b)
            .count();
        assert!(mismatches > 0);
        println!("PASS FP8_POOL_SCALE_ROW_CONTROL bit_mismatches={mismatches}");
        Ok(())
    }

    #[test]
    fn fp8_detected_arms_match_the_scalar_arm_bitwise() -> Result<()> {
        let shape = production_shape();
        let (slot, layout) = fp8_fixture(&shape);
        let mut scratch = Scratch::new(&shape);
        for (row, routing) in [(0usize, 0.3125f32), (1, -0.5)] {
            let hidden = hidden_row(&shape, row);
            let routing = bf16::from_f32(routing);
            let mut scalar = vec![bf16::ZERO; shape.hidden];
            fp8_contribution(
                Isa::Scalar,
                Fp8Contribution {
                    shape: &shape,
                    slot: &slot,
                    slot_layout: &layout,
                    hidden: &hidden,
                    routing,
                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut scalar,
                &mut scratch,
            )?;
            for isa in [Isa::Avx2, Isa::Avx512Bf16] {
                if !isa.available() {
                    println!("arm {} unavailable", isa.name());
                    continue;
                }
                let mut out = vec![bf16::ZERO; shape.hidden];
                fp8_contribution(
                    isa,
                    Fp8Contribution {
                        shape: &shape,
                        slot: &slot,
                        slot_layout: &layout,
                        hidden: &hidden,
                        routing,
                        rounding: crate::model_spec::SwigluRounding::Twice,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut out,
                    &mut scratch,
                )?;
                assert_eq!(out, scalar, "{} differs from the scalar arm", isa.name());
                println!("fp8 arm {} bitwise", isa.name());
            }
            let mut mutated = slot.clone();
            let down_scale = layout.down_scale.start as usize;
            mutated[down_scale + 1] ^= 0x80;
            let mut mutant = vec![bf16::ZERO; shape.hidden];
            fp8_contribution(
                Isa::Scalar,
                Fp8Contribution {
                    shape: &shape,
                    slot: &mutated,
                    slot_layout: &layout,
                    hidden: &hidden,
                    routing,
                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut mutant,
                &mut scratch,
            )?;
            assert_ne!(
                mutant, scalar,
                "a flipped down-scale sign left the output unchanged"
            );
        }
        Ok(())
    }

    #[test]
    fn detected_arms_match_the_scalar_arm_bitwise() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let mut scratch = Scratch::new(&shape);
        for (expert, routing) in [(0usize, 0.3125f32), (1, -0.5), (2, 0.0625), (3, 1.0)] {
            let hidden = hidden_row(&shape, expert);
            let weights0 = weights(&shape, expert);
            let routing = bf16::from_f32(routing);
            let mut scalar = vec![bf16::ZERO; shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &weights0,
                    hidden: &hidden,
                    routing,
                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut scalar,
                &mut scratch,
            )?;
            for isa in [Isa::Avx2, Isa::Avx512Bf16] {
                if !isa.available() {
                    println!("arm {} unavailable", isa.name());
                    continue;
                }
                let mut out = vec![bf16::ZERO; shape.hidden];
                contribution(
                    isa,
                    Contribution {
                        shape: &shape,
                        arena: &arena_flat(&arena),
                        weights: &weights0,
                        hidden: &hidden,
                        routing,
                        rounding: crate::model_spec::SwigluRounding::Twice,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut out,
                    &mut scratch,
                )?;
                assert!(
                    out.iter()
                        .zip(&scalar)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} differs from the scalar arm",
                    isa.name()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn gate_up_split_and_expert_pointer_controls_change_the_result() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 2);
        let hidden = hidden_row(&shape, 0);
        let routing = bf16::from_f32(0.25);
        let mut scratch = Scratch::new(&shape);
        let mut correct = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &arena_flat(&arena),
                weights: &weights(&shape, 0),
                hidden: &hidden,
                routing,
                rounding: crate::model_spec::SwigluRounding::Twice,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
            &mut correct,
            &mut scratch,
        )?;
        let mut wrong_expert = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &arena_flat(&arena),
                weights: &weights(&shape, 1),
                hidden: &hidden,
                routing,
                rounding: crate::model_spec::SwigluRounding::Twice,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
            &mut wrong_expert,
            &mut scratch,
        )?;
        assert!(
            correct != wrong_expert,
            "wrong expert pointer produced identical output"
        );
        let (h, i) = (shape.hidden, shape.intermediate);
        let flat = arena_flat(&arena);
        let mut shifted: Vec<bf16> = flat.clone();
        let source = &flat[..2 * i * h];
        for j in 0..i - 1 {
            shifted[(i + j) * h..][..h].copy_from_slice(&source[(i + j + 1) * h..][..h]);
        }
        shifted[(2 * i - 1) * h..][..h].copy_from_slice(&source[i * h..][..h]);
        let shifted_weights = ExpertWeights {
            gate_up: 0..2 * i * h,
            gate_up_stride: h,
            down: 2 * i * h..per_expert(&shape),
            down_stride: i,
        };
        let mut split = vec![bf16::ZERO; shape.hidden];
        contribution(
            Isa::Scalar,
            Contribution {
                shape: &shape,
                arena: &shifted,
                weights: &shifted_weights,
                hidden: &hidden,
                routing,
                rounding: crate::model_spec::SwigluRounding::Twice,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
            &mut split,
            &mut scratch,
        )?;
        assert!(
            correct != split,
            "gate/up split shifted by one row produced identical output"
        );
        Ok(())
    }

    #[test]
    fn pool_matches_single_threaded_contributions() -> Result<()> {
        for shape in [Shape::new(128, 64)?, Shape::new(64, 128)?] {
            let arena = build_arena(&shape, 8);
            let mut scratch = Scratch::new(&shape);
            let rows = 8usize;
            let hidden: Vec<bf16> = (0..rows * shape.hidden).map(|k| pattern(k, 0.7)).collect();
            for experts_used in [1usize, 2, 5, 8] {
                let mut assignments = Vec::new();
                for row in 0..rows {
                    for choice in 0..8 {
                        assignments.push(Assignment {
                            row: row as u32,
                            choice: (row * 8 + choice) as u32,
                            expert: ((row * 3 + choice) % experts_used) as u32,
                            weight: bf16::from_f32(0.125 * (choice as f32 + 1.0)),
                        });
                    }
                }
                let experts: Vec<_> = (0..experts_used).map(|e| weights(&shape, e)).collect();
                let mut expected = vec![bf16::ZERO; assignments.len() * shape.hidden];
                for assignment in &assignments {
                    let row = &hidden[assignment.row as usize * shape.hidden..][..shape.hidden];
                    let out =
                        &mut expected[assignment.choice as usize * shape.hidden..][..shape.hidden];
                    contribution(
                        Isa::Scalar,
                        Contribution {
                            shape: &shape,
                            arena: &arena_flat(&arena),
                            weights: &experts[assignment.expert as usize],
                            hidden: row,
                            routing: assignment.weight,

                            rounding: crate::model_spec::SwigluRounding::Twice,
                            routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                        },
                        out,
                        &mut scratch,
                    )?;
                }
                for workers in [1usize, 2, 4, 8] {
                    let pool = ExpertPool::new(workers, shape, rows, assignments.len(), None)?;
                    let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                    pool.run(Job {
                        arena: arena.clone(),
                        hidden: &hidden,
                        rows,
                        experts: &experts,
                        assignments: &assignments,
                        out: &mut out,
                    })?;
                    assert_eq!(out, expected, "workers {workers}, experts {experts_used}");
                    // Assignment order must not move results: `out` is indexed by choice slot.
                    let mut shuffled = assignments.clone();
                    shuffled.reverse();
                    let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
                    pool.run(Job {
                        arena: arena.clone(),
                        hidden: &hidden,
                        rows,
                        experts: &experts,
                        assignments: &shuffled,
                        out: &mut out,
                    })?;
                    assert_eq!(
                        out, expected,
                        "reversed assignment order, workers {workers}, experts {experts_used}"
                    );
                }
            }
        }
        Ok(())
    }

    struct PoolFixture {
        shape: Shape,
        arena: Arc<HostBlocks>,
        hidden: Vec<bf16>,
        experts: Vec<ExpertWeights>,
        assignments: Vec<Assignment>,
    }

    fn pool_fixture() -> Result<PoolFixture> {
        let shape = Shape::new(128, 64)?;
        let arena = build_arena(&shape, 4);
        let rows = 2usize;
        let hidden: Vec<bf16> = (0..rows).flat_map(|row| hidden_row(&shape, row)).collect();
        let assignments: Vec<Assignment> = (0..rows)
            .flat_map(|row| {
                (0..2).map(move |choice| Assignment {
                    row: row as u32,
                    choice: (row * 2 + choice) as u32,
                    expert: ((row + choice) % 3) as u32,
                    weight: bf16::from_f32(0.5 + choice as f32 * 0.25),
                })
            })
            .collect();
        let experts: Vec<ExpertWeights> = (0..3).map(|expert| weights(&shape, expert)).collect();
        Ok(PoolFixture {
            shape,
            arena,
            hidden,
            experts,
            assignments,
        })
    }

    fn expected_output(fixture: &PoolFixture) -> Result<Vec<bf16>> {
        let shape = &fixture.shape;
        let mut scratch = Scratch::new(shape);
        let flat = arena_flat(&fixture.arena);
        let mut expected = vec![bf16::ZERO; fixture.assignments.len() * shape.hidden];
        for assignment in &fixture.assignments {
            let row = &fixture.hidden[assignment.row as usize * shape.hidden..][..shape.hidden];
            let out = &mut expected[assignment.choice as usize * shape.hidden..][..shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape,
                    arena: &flat,
                    weights: &fixture.experts[assignment.expert as usize],
                    hidden: row,
                    routing: assignment.weight,

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                out,
                &mut scratch,
            )?;
        }
        Ok(expected)
    }

    fn fixture_job<'a>(fixture: &'a PoolFixture, out: &'a mut [bf16]) -> Job<'a> {
        Job {
            arena: fixture.arena.clone(),
            hidden: &fixture.hidden,
            rows: 2,
            experts: &fixture.experts,
            assignments: &fixture.assignments,
            out,
        }
    }

    #[test]
    fn poisoned_job_lock_rejects_work_without_publishing_output() -> Result<()> {
        let fixture = pool_fixture()?;
        let pool = ExpertPool::new(2, fixture.shape, 2, fixture.assignments.len(), None)?;
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = pool.shared.job.write().unwrap();
            panic!("expert job poison control");
        }));
        assert!(poisoned.is_err());
        let mut out = vec![bf16::ONE; fixture.assignments.len() * fixture.shape.hidden];
        let before = out.clone();
        assert!(pool.run(fixture_job(&fixture, &mut out)).is_err());
        assert_eq!(out, before);
        Ok(())
    }

    #[test]
    fn pending_guard_holds_the_lease_until_finish() -> Result<()> {
        let fixture = pool_fixture()?;
        let pool = ExpertPool::new(2, fixture.shape, 2, fixture.assignments.len(), None)?;
        let mut out = vec![bf16::ZERO; fixture.assignments.len() * fixture.shape.hidden];
        let block =
            u32::try_from(fixture.experts[0].gate_up.start / fixture.arena.block_elements())?;
        let pending = pool.submit(fixture_job(&fixture, &mut out))?;
        assert!(
            fixture.arena.claim(block).is_err(),
            "a pending guard must keep its lease"
        );
        pool.finish(pending)?;
        assert!(
            fixture.arena.claim(block).is_ok(),
            "finish must release the lease"
        );
        Ok(())
    }

    #[test]
    fn overlapping_submit_errors_without_touching_its_output() -> Result<()> {
        let fixture = pool_fixture()?;
        let pool = ExpertPool::new(2, fixture.shape, 2, fixture.assignments.len(), None)?;
        let mut first = vec![bf16::ZERO; fixture.assignments.len() * fixture.shape.hidden];
        let pending = pool.submit(fixture_job(&fixture, &mut first))?;
        let mut second =
            vec![bf16::from_f32(-3.0); fixture.assignments.len() * fixture.shape.hidden];
        let error = pool
            .submit(fixture_job(&fixture, &mut second))
            .err()
            .context("overlapping submit must fail")?
            .to_string();
        assert!(error.contains("already pending"), "{error}");
        assert!(
            second.iter().all(|value| *value == bf16::from_f32(-3.0)),
            "an errored submit must leave its output unmodified"
        );
        pool.finish(pending)?;
        Ok(())
    }

    #[test]
    fn dropping_the_guard_waits_and_releases() -> Result<()> {
        let fixture = pool_fixture()?;
        let expected = expected_output(&fixture)?;
        let pool = ExpertPool::new(2, fixture.shape, 2, fixture.assignments.len(), None)?;
        let mut out = vec![bf16::ZERO; fixture.assignments.len() * fixture.shape.hidden];
        {
            let _pending = pool.submit(fixture_job(&fixture, &mut out))?;
        }
        assert_eq!(out, expected, "drop must wait for completion");
        let block =
            u32::try_from(fixture.experts[0].gate_up.start / fixture.arena.block_elements())?;
        assert!(
            fixture.arena.claim(block).is_ok(),
            "drop must release the lease"
        );
        Ok(())
    }

    #[test]
    fn run_and_submit_finish_are_bitwise_identical() -> Result<()> {
        let fixture = pool_fixture()?;
        let expected = expected_output(&fixture)?;
        let pool = ExpertPool::new(2, fixture.shape, 2, fixture.assignments.len(), None)?;
        let mut out_run = vec![bf16::ZERO; fixture.assignments.len() * fixture.shape.hidden];
        pool.run(fixture_job(&fixture, &mut out_run))?;
        let mut out_seam = vec![bf16::ZERO; fixture.assignments.len() * fixture.shape.hidden];
        let pending = pool.submit(fixture_job(&fixture, &mut out_seam))?;
        pool.finish(pending)?;
        let bits = |values: &[bf16]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&out_run),
            bits(&out_seam),
            "run and submit+finish differ"
        );
        assert_eq!(out_run, expected);
        Ok(())
    }

    #[test]
    fn pool_production_shape_matches_scalar() -> Result<()> {
        let shape = production_shape();
        let arena = build_arena(&shape, 4);
        let hidden = hidden_row(&shape, 0);
        let experts: Vec<_> = (0..4).map(|e| weights(&shape, e)).collect();
        let assignments: Vec<_> = (0..8)
            .map(|choice| Assignment {
                row: 0,
                choice,
                expert: choice % 4,
                weight: bf16::from_f32(0.125),
            })
            .collect();
        let mut scratch = Scratch::new(&shape);
        let mut expected = vec![bf16::ZERO; assignments.len() * shape.hidden];
        for assignment in &assignments {
            let out = &mut expected[assignment.choice as usize * shape.hidden..][..shape.hidden];
            contribution(
                Isa::Scalar,
                Contribution {
                    shape: &shape,
                    arena: &arena_flat(&arena),
                    weights: &experts[assignment.expert as usize],
                    hidden: &hidden,
                    routing: assignment.weight,

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                out,
                &mut scratch,
            )?;
        }
        let pool = ExpertPool::new(4, shape, 1, assignments.len(), None)?;
        let mut out = vec![bf16::ZERO; assignments.len() * shape.hidden];
        pool.run(Job {
            arena,
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &assignments,
            out: &mut out,
        })?;
        assert_eq!(out, expected);
        Ok(())
    }

    #[test]
    fn pool_rejects_invalid_jobs() -> Result<()> {
        let shape = Shape::new(128, 64)?;
        let arena = build_arena(&shape, 1);
        let experts = [weights(&shape, 0)];
        let hidden = vec![bf16::ZERO; shape.hidden];
        let pool = ExpertPool::new(1, shape, 1, 2, None)?;
        let good = Assignment {
            row: 0,
            choice: 0,
            expert: 0,
            weight: bf16::from_f32(0.5),
        };
        let mut out = vec![bf16::ZERO; shape.hidden];
        pool.run(Job {
            arena: arena.clone(),
            hidden: &hidden,
            rows: 1,
            experts: &experts,
            assignments: &[good],
            out: &mut out,
        })?;
        assert!(ExpertPool::new(0, shape, 1, 1, None).is_err());
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 2,
                experts: &experts,
                assignments: &[good],
                out: &mut out,
            })
            .is_err()
        );
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden[..shape.hidden - 1],
                rows: 1,
                experts: &experts,
                assignments: &[good],
                out: &mut out,
            })
            .is_err()
        );
        let bad_row = Assignment { row: 1, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[bad_row],
                out: &mut out,
            })
            .is_err()
        );
        let bad_expert = Assignment { expert: 1, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[bad_expert],
                out: &mut out,
            })
            .is_err()
        );
        let duplicate = Assignment { choice: 0, ..good };
        assert!(
            pool.run(Job {
                arena: arena.clone(),
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[good, duplicate],
                out: &mut out,
            })
            .is_err()
        );
        let bad_choice = Assignment { choice: 1, ..good };
        assert!(
            pool.run(Job {
                arena,
                hidden: &hidden,
                rows: 1,
                experts: &experts,
                assignments: &[good, bad_choice],
                out: &mut out,
            })
            .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires FF_MINICPM_REFERENCE_DIR/torch-expert.safetensors"]
    fn model_definition_matches_torch_cpu() -> Result<()> {
        let path = std::path::PathBuf::from(
            std::env::var_os("FF_MINICPM_REFERENCE_DIR")
                .context("FF_MINICPM_REFERENCE_DIR missing")?,
        )
        .join("torch-expert.safetensors");
        let bytes = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let read = |name: &str| -> Result<Vec<bf16>> {
            let tensor = tensors.tensor(name)?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::BF16,
                "{name} must be BF16"
            );
            Ok(tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
                .collect())
        };
        let shape = production_shape();
        let hidden = read("x")?;
        let gate_up = read("gate_up")?;
        let down = read("down")?;
        let weight = read("weight")?;
        let expected = read("contribution")?;
        ensure!(
            hidden.len() == shape.hidden
                && gate_up.len() == 2 * shape.intermediate * shape.hidden
                && down.len() == shape.hidden * shape.intermediate
                && weight.len() == 1
                && expected.len() == shape.hidden,
            "torch fixture shape"
        );
        let mut values = gate_up;
        values.extend_from_slice(&down);
        let arena: Arc<[bf16]> = Arc::from(values.into_boxed_slice());
        let weights = ExpertWeights {
            gate_up: 0..2 * shape.intermediate * shape.hidden,
            gate_up_stride: shape.hidden,
            down: 2 * shape.intermediate * shape.hidden..per_expert(&shape),
            down_stride: shape.intermediate,
        };
        let reference64 = reference(&shape, &arena, &weights, &hidden, weight[0]);
        let reference_noise = expected
            .iter()
            .zip(&reference64)
            .map(|(&a, &b)| order_key(a).abs_diff(order_key(b)))
            .max()
            .context("empty torch contribution reference")?;
        let mut scratch = Scratch::new(&shape);
        for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512Bf16] {
            if !isa.available() {
                continue;
            }
            let mut out = vec![bf16::ZERO; shape.hidden];
            contribution(
                isa,
                Contribution {
                    shape: &shape,
                    arena: &arena,
                    weights: &weights,
                    hidden: &hidden,
                    routing: weight[0],

                    rounding: crate::model_spec::SwigluRounding::Twice,
                    routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                },
                &mut out,
                &mut scratch,
            )?;
            let mut different = 0usize;
            let mut worst = 0u32;
            for (actual, reference) in out.iter().zip(&expected) {
                let distance = order_key(*actual).abs_diff(order_key(*reference));
                different += usize::from(distance != 0);
                worst = worst.max(distance);
            }
            println!(
                "{}",
                serde_json::json!({
                    "torch_parity": {
                        "isa": isa.name(),
                        "elements": out.len(),
                        "different": different,
                        "max_bf16_steps": worst,
                    }
                })
            );
            assert!(
                worst <= reference_noise,
                "{} differs from the torch contribution beyond the reference noise",
                isa.name()
            );
        }
        Ok(())
    }

    /// 12.5x the 64 MiB LLC of the Ryzen 7950X; 128 x 6,291,456 B = 805 MB.
    const RING_EXPERTS: usize = 128;

    #[test]
    #[ignore = "descriptive CPU timing table"]
    fn service_time_table() -> Result<()> {
        let shape = production_shape();
        let per = per_expert(&shape);
        println!(
            "{}",
            serde_json::json!({
                "table": "service_time",
                "isa": Isa::detect().name(),
                "ring_experts": RING_EXPERTS,
                "ring_bytes": RING_EXPERTS * per * 2,
                "expert_bytes": per * 2,
            })
        );
        let cells = measure_service_table(
            shape,
            RING_EXPERTS as u32,
            &[1, 2, 4, 8, 16, 24],
            &[1, 2, 5, 8],
            &[1, 2, 4, 8],
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        for cell in cells {
            let cold_bytes = f64::from(cell.experts) * per as f64 * 2.0;
            let flops = f64::from(cell.rows * cell.experts)
                * 6.0
                * shape.hidden as f64
                * shape.intermediate as f64;
            println!(
                "{}",
                serde_json::json!({
                    "workers": cell.workers,
                    "rows": cell.rows,
                    "missed": cell.experts,
                    "seconds": cell.seconds,
                    "GFLOPs": flops / cell.seconds / 1e9,
                    "cold_GB_s": cold_bytes / cell.seconds / 1e9,
                })
            );
        }
        Ok(())
    }
}
