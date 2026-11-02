//! MoE model-definition oracle: BF16 linear/SiLU/product/contribution boundaries.
//! Router softmax and weight renormalization use FP32; exact ties select lower expert ids.
//! The grouped_mm form sums routed contributions in FP32, rounds to BF16, then adds gated BF16 shared output.
use anyhow::{Result, ensure};
use half::bf16;
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScoredId {
    pub id: i32,
    pub score: f32,
}

impl PartialEq for ScoredId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}
impl Eq for ScoredId {}
impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .reverse()
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A max-heap whose top is the current worst selected item.
pub(crate) struct BoundedTopK {
    k: usize,
    heap: BinaryHeap<ScoredId>,
}

impl BoundedTopK {
    pub fn new(k: usize) -> Result<Self> {
        ensure!(k > 0, "top-k bound must be positive");
        Ok(Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        })
    }

    pub fn push(&mut self, item: ScoredId) -> Result<()> {
        ensure!(item.id >= 0, "top-k id must be nonnegative");
        ensure!(!item.score.is_nan(), "top-k score must not be NaN");
        if self.heap.len() < self.k {
            self.heap.push(item);
            return Ok(());
        }
        let worst = self.heap.peek().expect("nonempty top-k heap");
        let replace = item.score.total_cmp(&worst.score) == Ordering::Greater
            || (item.score.total_cmp(&worst.score) == Ordering::Equal && item.id < worst.id);
        if replace {
            self.heap.pop();
            self.heap.push(item);
        }
        Ok(())
    }

    pub fn finish(self) -> Vec<ScoredId> {
        let mut items = self.heap.into_vec();
        items.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        items
    }
}

fn sqrt_softplus(value: f32) -> f32 {
    let softplus = value.max(0.0) + (-value.abs()).exp().ln_1p();
    softplus.sqrt()
}

/// Select by score+bias, then normalize the unbiased score with the published guard and scale.
pub(crate) fn noaux_tc_route(
    logits: &[f32],
    correction_bias: &[f32],
    top_k: usize,
    score: crate::model_spec::RouterScore,
    routed_scaling: f32,
) -> Result<(Vec<i32>, Vec<f32>)> {
    ensure!(
        !logits.is_empty() && logits.len() == correction_bias.len(),
        "router logits/bias extents differ"
    );
    ensure!(top_k > 0 && top_k <= logits.len(), "invalid router top-k");
    ensure!(
        routed_scaling.is_finite() && routed_scaling > 0.0,
        "invalid routed scaling"
    );
    ensure!(
        score != crate::model_spec::RouterScore::Softmax,
        "noaux_tc requires a pointwise score"
    );
    let mut selected = BoundedTopK::new(top_k)?;
    let mut raw = Vec::with_capacity(logits.len());
    for (id, (&logit, &bias)) in logits.iter().zip(correction_bias).enumerate() {
        ensure!(
            logit.is_finite() && bias.is_finite(),
            "nonfinite router input"
        );
        let value = match score {
            crate::model_spec::RouterScore::Sigmoid => sigmoid(logit),
            crate::model_spec::RouterScore::SqrtSoftplus => sqrt_softplus(logit),
            crate::model_spec::RouterScore::Softmax => unreachable!(),
        };
        ensure!(value.is_finite(), "nonfinite router score");
        raw.push(value);
        selected.push(ScoredId {
            id: i32::try_from(id)?,
            score: value + bias,
        })?;
    }
    let selected = selected.finish();
    let mut weights = Vec::with_capacity(top_k);
    let mut sum = 0.0f32;
    for item in &selected {
        let value = raw[item.id as usize];
        weights.push(value);
        sum += value;
    }
    let denominator = sum + 1e-20;
    for value in &mut weights {
        *value = (*value / denominator) * routed_scaling;
    }
    Ok((selected.iter().map(|item| item.id).collect(), weights))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeDims {
    pub rows: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub inter: usize,
    pub shared_inter: usize,
}

impl MoeDims {
    pub fn new(rows: usize, experts: usize, top_k: usize, hidden: usize, inter: usize) -> Self {
        Self {
            rows,
            experts,
            top_k,
            hidden,
            inter,
            shared_inter: inter,
        }
    }

    /// MiniCPM-V-4.7-35B-A3B text tower MoE: E=256, K=8, H=2048, I=512.
    pub fn minicpm47(rows: usize) -> Self {
        Self::new(rows, 256, 8, 2048, 512)
    }
}

/// Synthetic host-side weights. Layouts (row-major):
/// x `[rows, hidden]`, router_w `[experts, hidden]`,
/// gate_up `[experts, 2*inter, hidden]` with gate rows `[0, inter)` and up
/// rows `[inter, 2*inter)`, down `[experts, hidden, inter]`,
/// shared_gate/shared_up `[shared_inter, hidden]`,
/// shared_down `[hidden, shared_inter]`, shared_expert_gate `[hidden]`.
pub(crate) struct MoeFixture {
    pub dims: MoeDims,
    pub x: Vec<bf16>,
    pub router_w: Vec<bf16>,
    pub gate_up: Vec<bf16>,
    pub down: Vec<bf16>,
    pub shared_gate: Vec<bf16>,
    pub shared_up: Vec<bf16>,
    pub shared_down: Vec<bf16>,
    pub shared_expert_gate: Vec<bf16>,
    /// Row carrying two experts with bitwise-equal router rows.
    pub exact_tie_row: usize,
    pub exact_tie_experts: (usize, usize),
    /// Row with no planted structure; near_tie_experts are the rank k-1 and
    /// rank k experts of its routing and near_tie_margin their FP32
    /// probability gap (the top-k cut margin). BF16 logit quantisation makes
    /// one logit ulp the smallest representable race.
    pub near_tie_row: usize,
    pub near_tie_experts: (usize, usize),
    pub near_tie_margin: f32,
}

impl MoeFixture {
    pub fn gate_row(&self, e: usize, j: usize) -> &[bf16] {
        let (h, i) = (self.dims.hidden, self.dims.inter);
        &self.gate_up[(e * 2 * i + j) * h..][..h]
    }
    pub fn up_row(&self, e: usize, j: usize) -> &[bf16] {
        let (h, i) = (self.dims.hidden, self.dims.inter);
        &self.gate_up[(e * 2 * i + i + j) * h..][..h]
    }
    pub fn down_row(&self, e: usize, d: usize) -> &[bf16] {
        let (h, i) = (self.dims.hidden, self.dims.inter);
        &self.down[(e * h + d) * i..][..i]
    }
}

// Element values are pure functions of (salt, indices) through a splitmix64
// finalizer. The per-index multipliers are odd, so no dimension aliases at
// the power-of-two extents 2048/512/256/8.
fn mix(salt: u64, a: usize, b: usize, c: usize) -> u64 {
    let mut z = salt
        .wrapping_add((a as u64).wrapping_mul(0x9E37_79B1))
        .wrapping_add((b as u64).wrapping_mul(0x85EB_CA77))
        .wrapping_add((c as u64).wrapping_mul(0xC2B2_AE3D));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(salt: u64, a: usize, b: usize, c: usize) -> f32 {
    ((mix(salt, a, b, c) >> 40) as f32) / 8_388_608.0 - 1.0
}

fn seed(variant: u32, slot: u64) -> u64 {
    u64::from(variant).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ slot.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

pub(crate) fn fixture(dims: MoeDims, variant: u32) -> MoeFixture {
    let MoeDims {
        rows,
        experts,
        top_k,
        hidden: h,
        inter: i,
        shared_inter: si,
    } = dims;
    assert!(
        rows >= 2 && experts >= 6 && top_k >= 4 && experts > top_k && h >= 8 && i >= 1 && si >= 1
    );
    let wscale = 2.0 / (h as f32).sqrt();
    let dscale = 1.0 / (i as f32).sqrt();
    let shared_dscale = 1.0 / (si as f32).sqrt();
    let mut x = Vec::with_capacity(rows * h);
    for r in 0..rows {
        for c in 0..h {
            x.push(bf16::from_f32(unit(seed(variant, 1), r, c, 0)));
        }
    }
    let mut router_w = Vec::with_capacity(experts * h);
    for e in 0..experts {
        for c in 0..h {
            router_w.push(bf16::from_f32(unit(seed(variant, 2), e, c, 0) * wscale));
        }
    }
    let mut gate_up = Vec::with_capacity(experts * 2 * i * h);
    for e in 0..experts {
        for j in 0..2 * i {
            for c in 0..h {
                gate_up.push(bf16::from_f32(unit(seed(variant, 3), e, j, c) * wscale));
            }
        }
    }
    let mut down = Vec::with_capacity(experts * h * i);
    for e in 0..experts {
        for d in 0..h {
            for j in 0..i {
                down.push(bf16::from_f32(unit(seed(variant, 4), e, d, j) * dscale));
            }
        }
    }
    let mut shared_gate = Vec::with_capacity(si * h);
    let mut shared_up = Vec::with_capacity(si * h);
    for j in 0..si {
        for c in 0..h {
            shared_gate.push(bf16::from_f32(unit(seed(variant, 5), 0, j, c) * wscale));
            shared_up.push(bf16::from_f32(unit(seed(variant, 6), 0, j, c) * wscale));
        }
    }
    let mut shared_down = Vec::with_capacity(h * si);
    for d in 0..h {
        for j in 0..si {
            shared_down.push(bf16::from_f32(
                unit(seed(variant, 7), 0, d, j) * shared_dscale,
            ));
        }
    }
    let mut shared_expert_gate = Vec::with_capacity(h);
    for c in 0..h {
        shared_expert_gate.push(bf16::from_f32(unit(seed(variant, 8), 0, c, 0) * wscale));
    }

    // The exact-tie pair: both router rows equal bf16(0.5 * x[exact_row]),
    // which dominates every random row, so the pair sits inside the top-k
    // with bitwise-equal BF16 logits.
    let exact = (1, experts - 1);
    let exact_row = 0usize;
    for c in 0..h {
        let tied = bf16::from_f32(x[exact_row * h + c].to_f32() * 0.5);
        router_w[exact.0 * h + c] = tied;
        router_w[exact.1 * h + c] = tied;
    }
    let near_row = 1usize;
    let probs = softmax_probs(&router_logits(
        &router_w,
        &x[near_row * h..(near_row + 1) * h],
        experts,
    ));
    let order = ranking(&probs);
    let near = (order[top_k - 1], order[top_k]);
    let near_margin = (probs[near.0] - probs[near.1]).abs();
    assert!(
        near_margin > 0.0,
        "fixture variant {variant} ties at the top-k cut of row {near_row}"
    );

    MoeFixture {
        dims,
        x,
        router_w,
        gate_up,
        down,
        shared_gate,
        shared_up,
        shared_down,
        shared_expert_gate,
        exact_tie_row: exact_row,
        exact_tie_experts: exact,
        near_tie_row: near_row,
        near_tie_experts: near,
        near_tie_margin: near_margin,
    }
}

fn router_logits(router: &[bf16], xr: &[bf16], experts: usize) -> Vec<bf16> {
    let h = xr.len();
    (0..experts)
        .map(|e| {
            let dot: f32 = xr
                .iter()
                .zip(&router[e * h..(e + 1) * h])
                .map(|(&a, &b)| a.to_f32() * b.to_f32())
                .sum();
            bf16::from_f32(dot)
        })
        .collect()
}

fn softmax_probs(logits: &[bf16]) -> Vec<f32> {
    let max = logits
        .iter()
        .map(|v| v.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l.to_f32() - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

/// Expert ids ordered probability-descending with exact ties by lower id.
fn ranking(probs: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .expect("non-finite router probability")
            .then(a.cmp(&b))
    });
    order
}

pub(crate) struct RouteRow {
    /// `[top_k]` expert ids: probability descending, exact ties by lower id.
    pub ids: Vec<i32>,
    /// `[top_k]` renormalised FP32 weights cast to BF16.
    pub weights: Vec<bf16>,
    /// `[experts]` FP32 softmax of the BF16-rounded logits.
    pub probs: Vec<f32>,
    /// probs[rank k-1] - probs[rank k]; 0 marks an exact tie at the cut.
    pub margin: f32,
}

pub(crate) fn route_row(fx: &MoeFixture, row: usize) -> RouteRow {
    let (e, k, h) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    assert!(row < fx.dims.rows);
    route_logits(
        &router_logits(&fx.router_w, &fx.x[row * h..(row + 1) * h], e),
        k,
    )
}

pub(crate) fn sigmoid_route_row(fx: &MoeFixture, row: usize) -> RouteRow {
    let (e, k, h) = (fx.dims.experts, fx.dims.top_k, fx.dims.hidden);
    assert!(row < fx.dims.rows);
    let logits = router_logits(&fx.router_w, &fx.x[row * h..(row + 1) * h], e);
    let logits: Vec<f32> = logits.iter().map(|value| value.to_f32()).collect();
    let (ids, weights) = noaux_tc_route(
        &logits,
        &vec![0.0; e],
        k,
        crate::model_spec::RouterScore::Sigmoid,
        1.0,
    )
    .expect("valid sigmoid route");
    let probs: Vec<f32> = logits.iter().map(|&value| sigmoid(value)).collect();
    let order = ranking(&probs);
    let margin = probs[order[k - 1]] - probs[order[k]];
    RouteRow {
        ids,
        weights: weights.into_iter().map(bf16::from_f32).collect(),
        probs,
        margin,
    }
}

pub(crate) fn route_logits(logits: &[bf16], k: usize) -> RouteRow {
    let probs = softmax_probs(logits);
    let order = ranking(&probs);
    let top = &order[..k];
    let ids = top.iter().map(|&id| id as i32).collect();
    let tsum: f32 = top.iter().map(|&id| probs[id]).sum();
    let weights = top
        .iter()
        .map(|&id| bf16::from_f32(probs[id] / tsum))
        .collect();
    let margin = probs[order[k - 1]] - probs[order[k]];
    RouteRow {
        ids,
        weights,
        probs,
        margin,
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn dot(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.to_f32() * y.to_f32())
        .sum()
}

fn expert_out(
    fx: &MoeFixture,
    e: usize,
    xr: &[bf16],
    weight: bf16,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
) -> Vec<bf16> {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let mut prod = Vec::with_capacity(i);
    for j in 0..i {
        let g = bf16::from_f32(dot(xr, fx.gate_row(e, j)));
        let u = bf16::from_f32(dot(xr, fx.up_row(e, j)));
        let s = bf16::from_f32(silu(g.to_f32()));
        prod.push(bf16::from_f32(s.to_f32() * u.to_f32()));
    }
    (0..h)
        .map(|d| {
            let lin = dot(&prod, fx.down_row(e, d));
            let weighted = match routed_weight {
                crate::model_spec::RoutedWeightBoundary::AfterDownBf16 => {
                    bf16::from_f32(lin).to_f32() * weight.to_f32()
                }
                crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16 => {
                    lin * weight.to_f32()
                }
                crate::model_spec::RoutedWeightBoundary::BeforeDownQuantization => {
                    unreachable!("quantized-down boundary in the BF16 reference")
                }
            };
            bf16::from_f32(weighted)
        })
        .collect()
}

fn shared_out(fx: &MoeFixture, xr: &[bf16]) -> Vec<bf16> {
    let (h, si) = (fx.dims.hidden, fx.dims.shared_inter);
    let mut prod = Vec::with_capacity(si);
    for j in 0..si {
        let g = bf16::from_f32(dot(xr, &fx.shared_gate[j * h..(j + 1) * h]));
        let u = bf16::from_f32(dot(xr, &fx.shared_up[j * h..(j + 1) * h]));
        let s = bf16::from_f32(silu(g.to_f32()));
        prod.push(bf16::from_f32(s.to_f32() * u.to_f32()));
    }
    (0..h)
        .map(|d| bf16::from_f32(dot(&prod, &fx.shared_down[d * si..(d + 1) * si])))
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Combine {
    AscendingBf16,
    DescendingBf16,
    Fp32,
}

/// Routed contributions selected by `keep` are reduced per `combine`, rounded
/// to BF16, then added to the BF16 sigmoid-gated shared expert in BF16.
/// Fp32 sums use ascending expert id order.
pub(crate) fn moe_row_ids(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
    combine: Combine,
) -> Vec<bf16> {
    moe_row_ids_with_boundary(
        fx,
        row,
        ids,
        weights,
        keep,
        combine,
        crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
    )
}

pub(crate) fn moe_row_ids_with_boundary(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
    combine: Combine,
    routed_weight: crate::model_spec::RoutedWeightBoundary,
) -> Vec<bf16> {
    let h = fx.dims.hidden;
    assert!(row < fx.dims.rows);
    assert!(!keep.is_empty());
    assert_eq!(ids.len(), weights.len());
    let mut slots: Vec<(usize, bf16)> = keep
        .iter()
        .map(|&c| {
            assert!(c < ids.len(), "choice index {c} outside ids/weights extent");
            (ids[c] as usize, weights[c])
        })
        .collect();
    slots.sort_by_key(|&(id, _)| id);
    let xr = &fx.x[row * h..(row + 1) * h];
    let contributions: Vec<Vec<bf16>> = slots
        .iter()
        .map(|&(e, w)| expert_out(fx, e, xr, w, routed_weight))
        .collect();
    let shared = shared_out(fx, xr);
    let scale = bf16::from_f32(sigmoid(
        bf16::from_f32(dot(xr, &fx.shared_expert_gate)).to_f32(),
    ));
    let gated: Vec<bf16> = (0..h)
        .map(|d| bf16::from_f32(scale.to_f32() * shared[d].to_f32()))
        .collect();
    let mut y = vec![bf16::from_f32(0.0); h];
    let mut yf = vec![0.0f32; h];
    match combine {
        Combine::AscendingBf16 | Combine::DescendingBf16 => {
            let order: Box<dyn Iterator<Item = &Vec<bf16>>> = match combine {
                Combine::AscendingBf16 => Box::new(contributions.iter()),
                Combine::DescendingBf16 => Box::new(contributions.iter().rev()),
                Combine::Fp32 => unreachable!(),
            };
            for c in order {
                for d in 0..h {
                    y[d] = bf16::from_f32(y[d].to_f32() + c[d].to_f32());
                }
            }
            for d in 0..h {
                y[d] = bf16::from_f32(y[d].to_f32() + gated[d].to_f32());
            }
        }
        Combine::Fp32 => {
            for c in &contributions {
                for d in 0..h {
                    yf[d] += c[d].to_f32();
                }
            }
            for d in 0..h {
                let routed = bf16::from_f32(yf[d]);
                y[d] = bf16::from_f32(routed.to_f32() + gated[d].to_f32());
            }
        }
    }
    y
}

pub(crate) fn moe_row_with(
    fx: &MoeFixture,
    row: usize,
    combine: Combine,
    choices: &[usize],
) -> Vec<bf16> {
    let routed = route_row(fx, row);
    moe_row_ids(fx, row, &routed.ids, &routed.weights, choices, combine)
}

pub(crate) fn moe_row(fx: &MoeFixture, row: usize) -> Vec<bf16> {
    moe_row_with(fx, row, Combine::AscendingBf16, &choice_all(fx))
}

/// The row's shared expert plus only the routing choices in `choices`
/// (top-k choice indices), accumulated ascending expert id among the chosen.
pub(crate) fn moe_row_choices(fx: &MoeFixture, row: usize, choices: &[usize]) -> Vec<bf16> {
    moe_row_with(fx, row, Combine::AscendingBf16, choices)
}

fn choice_all(fx: &MoeFixture) -> Vec<usize> {
    (0..fx.dims.top_k).collect()
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CombineNoise {
    pub ascending_vs_descending: f32,
    pub ascending_vs_fp32: f32,
    pub max_abs: f32,
}

impl CombineNoise {
    pub fn tolerance(self) -> f32 {
        self.ascending_vs_descending.max(self.ascending_vs_fp32) + bf16_ulp(self.max_abs)
    }
}

pub(crate) fn bf16_ulp(value: f32) -> f32 {
    let value = bf16::from_f32(value.abs());
    bf16::from_bits(value.to_bits() + 1).to_f32() - value.to_f32()
}

pub(crate) fn row_noise(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
) -> CombineNoise {
    let asc = moe_row_ids(fx, row, ids, weights, keep, Combine::AscendingBf16);
    let desc = moe_row_ids(fx, row, ids, weights, keep, Combine::DescendingBf16);
    let f32c = moe_row_ids(fx, row, ids, weights, keep, Combine::Fp32);
    let mut noise = CombineNoise {
        ascending_vs_descending: 0.0,
        ascending_vs_fp32: 0.0,
        max_abs: 0.0,
    };
    for ((asc, desc), fp32) in asc.iter().zip(&desc).zip(&f32c) {
        let (asc, desc, fp32) = (asc.to_f32(), desc.to_f32(), fp32.to_f32());
        noise.ascending_vs_descending = noise.ascending_vs_descending.max((asc - desc).abs());
        noise.ascending_vs_fp32 = noise.ascending_vs_fp32.max((asc - fp32).abs());
        noise.max_abs = noise.max_abs.max(asc.abs()).max(desc.abs()).max(fp32.abs());
    }
    noise
}

pub(crate) fn combine_noise(fx: &MoeFixture, rows: &[usize]) -> CombineNoise {
    let mut noise = CombineNoise {
        ascending_vs_descending: 0.0,
        ascending_vs_fp32: 0.0,
        max_abs: 0.0,
    };
    for &row in rows {
        assert!(row < fx.dims.rows);
        let all = choice_all(fx);
        let route = route_row(fx, row);
        let measured = row_noise(fx, row, &route.ids, &route.weights, &all);
        noise.ascending_vs_descending = noise
            .ascending_vs_descending
            .max(measured.ascending_vs_descending);
        noise.ascending_vs_fp32 = noise.ascending_vs_fp32.max(measured.ascending_vs_fp32);
        noise.max_abs = noise.max_abs.max(measured.max_abs);
    }
    noise
}

/// Measured combine-order spread plus one BF16 ulp at the largest reference output.
pub(crate) fn tolerance(fx: &MoeFixture, rows: &[usize]) -> f32 {
    combine_noise(fx, rows).tolerance()
}

pub(crate) struct CloverMoeFixture {
    pub table: serde_json::Value,
    bytes: Vec<u8>,
}

impl CloverMoeFixture {
    pub fn open() -> Result<Self> {
        Self::open_named("moe.safetensors")
    }

    pub fn open_named(filename: &str) -> Result<Self> {
        use anyhow::Context;
        let root = std::path::PathBuf::from(
            std::env::var_os("FF_CLOVER_REFERENCE").context("FF_CLOVER_REFERENCE missing")?,
        );
        let table: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("thresholds.json"))?)?;
        ensure!(
            table["revision"] == "4b0402b65e523d0183a14c431955438714cb16ff"
                && table["schema_keys"] == 813,
            "Runner FAIL Clover reference identity"
        );
        Ok(Self {
            table,
            bytes: std::fs::read(root.join(filename))?,
        })
    }

    pub fn archive(&self) -> Result<safetensors::SafeTensors<'_>> {
        Ok(safetensors::SafeTensors::deserialize(&self.bytes)?)
    }

    pub fn values(&self, name: &str) -> Result<Vec<f32>> {
        use safetensors::Dtype;
        let archive = safetensors::SafeTensors::deserialize(&self.bytes)?;
        let tensor = archive.tensor(name)?;
        let values = match tensor.dtype() {
            Dtype::BF16 => tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_le_bytes(*v).to_f32())
                .collect(),
            Dtype::F32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            Dtype::I32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| i32::from_le_bytes(*v) as f32)
                .collect(),
            dtype => anyhow::bail!("Runner FAIL fixture dtype {name}: {dtype:?}"),
        };
        Ok(values)
    }

    pub fn bf16(&self, name: &str) -> Result<Vec<bf16>> {
        Ok(self.values(name)?.into_iter().map(bf16::from_f32).collect())
    }

    pub fn score(
        &self,
        tag: &str,
        metric: &str,
        values: &[bf16],
        bound: &serde_json::Value,
    ) -> Result<bool> {
        let reference = self.values(&format!("{tag}.fp.{metric}"))?;
        ensure!(
            values.len() == reference.len()
                && values.iter().all(|v| v.is_finite())
                && reference.iter().all(|v| v.is_finite()),
            "Runner FAIL MoE shape/nonfinite {tag} {metric}"
        );
        let mut maximum = 0.0f64;
        let mut squared = 0.0f64;
        let mut norm = 0.0f64;
        for (value, expected) in values.iter().zip(reference) {
            let delta = value.to_f64() - f64::from(expected);
            maximum = maximum.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(expected).powi(2);
        }
        let relative = (squared / norm.max(1e-60)).sqrt();
        let pass = maximum <= bound["max_abs"].as_f64().unwrap()
            && relative <= bound["relative_l2"].as_f64().unwrap();
        println!(
            "CLOVER_MOE {} tag={tag} metric={metric} max_abs={maximum:.9e} relative_l2={relative:.9e} bound={bound}",
            if pass { "PASS" } else { "DEFECT" }
        );
        Ok(pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(rows: usize) -> MoeDims {
        MoeDims::new(rows, 16, 4, 128, 64)
    }

    #[test]
    fn fixture_pins_exact_and_near_ties_inside_top_k() {
        let fx = fixture(small(2), 0);
        let exact = route_row(&fx, fx.exact_tie_row);
        let (a, b) = fx.exact_tie_experts;
        assert_eq!(exact.probs[a].to_bits(), exact.probs[b].to_bits());
        let pa = exact.ids.iter().position(|&id| id as usize == a).unwrap();
        let pb = exact.ids.iter().position(|&id| id as usize == b).unwrap();
        assert!(pa < pb, "exact tie must order by lower expert id");
        let near = route_row(&fx, fx.near_tie_row);
        let (c, d) = fx.near_tie_experts;
        assert_eq!(
            (near.probs[c] - near.probs[d]).abs().to_bits(),
            fx.near_tie_margin.to_bits()
        );
        assert!(near.margin > 0.0, "top-k cut must stay unambiguous");
    }

    #[test]
    fn choice_subset_drops_only_named_contributions() {
        let fx = fixture(small(2), 5);
        let full = moe_row(&fx, 1);
        let routed = route_row(&fx, 1);
        let drop = routed
            .ids
            .iter()
            .enumerate()
            .max_by_key(|&(_c, &id)| id)
            .unwrap()
            .0;
        let keep: Vec<usize> = (0..fx.dims.top_k).filter(|&c| c != drop).collect();
        let sub = moe_row_choices(&fx, 1, &keep);
        let maxdiff = (0..fx.dims.hidden)
            .map(|d| (full[d].to_f32() - sub[d].to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(maxdiff > 0.0, "dropping a contribution must change the row");
    }

    #[test]
    fn combine_order_noise_is_pinned() {
        let fx = fixture(MoeDims::minicpm47(8), 0);
        let noise = combine_noise(&fx, &[0, 1]);
        assert_eq!(
            noise.ascending_vs_descending.to_bits(),
            0.001953125f32.to_bits()
        );
        assert_eq!(noise.ascending_vs_fp32.to_bits(), 0.0009765625f32.to_bits());
    }

    #[derive(serde::Deserialize)]
    struct TopKFixture {
        scores: Vec<f32>,
        k: usize,
        ids: Vec<i32>,
    }
    #[derive(serde::Deserialize)]
    struct RouterFixture {
        mode: String,
        logits: Vec<f32>,
        bias: Vec<f32>,
        top_k: usize,
        scale: f32,
        ids: Vec<i32>,
        weights: Vec<f32>,
    }
    #[derive(serde::Deserialize)]
    struct SharedFixture {
        topk: TopKFixture,
        routers: Vec<RouterFixture>,
    }

    fn shared_fixture() -> SharedFixture {
        serde_json::from_str(include_str!("../../../../tests/data/shared-ops.json")).unwrap()
    }

    #[test]
    fn bounded_topk_matches_model_definition_order() -> Result<()> {
        let fixture = shared_fixture().topk;
        let mut selected = BoundedTopK::new(fixture.k)?;
        for (id, score) in fixture.scores.iter().copied().enumerate() {
            selected.push(ScoredId {
                id: i32::try_from(id)?,
                score,
            })?;
        }
        let selected = selected.finish();
        let ids: Vec<_> = selected.iter().map(|item| item.id).collect();
        assert_eq!(ids, fixture.ids);
        for pair in selected.windows(2) {
            let [a, b] = pair else { unreachable!() };
            assert!(
                a.score > b.score || (a.score == b.score && a.id < b.id),
                "selection order is not score-descending/id-ascending: {a:?} {b:?}"
            );
        }
        let mut tied = BoundedTopK::new(2)?;
        for id in [2, 1, 0] {
            tied.push(ScoredId { id, score: 1.0 })?;
        }
        let ids: Vec<_> = tied.finish().into_iter().map(|item| item.id).collect();
        assert_eq!(ids, [0, 1]);
        Ok(())
    }

    #[test]
    fn noaux_tc_modes_match_model_definition_weights() -> Result<()> {
        for fixture in shared_fixture().routers {
            let mode = match fixture.mode.as_str() {
                "sigmoid" => crate::model_spec::RouterScore::Sigmoid,
                "sqrtsoftplus" => crate::model_spec::RouterScore::SqrtSoftplus,
                other => anyhow::bail!("unknown score mode {other}"),
            };
            let (ids, weights) = noaux_tc_route(
                &fixture.logits,
                &fixture.bias,
                fixture.top_k,
                mode,
                fixture.scale,
            )?;
            assert_eq!(ids, fixture.ids, "{}", fixture.mode);
            for (index, (&actual, &expected)) in weights.iter().zip(&fixture.weights).enumerate() {
                let tolerance = 2e-6 * expected.abs().max(1.0);
                ensure!(
                    (actual - expected).abs() <= tolerance,
                    "{} weight {index}: actual={actual} expected={expected} tolerance={tolerance}",
                    fixture.mode
                );
            }
            ensure!(
                weights.iter().all(|weight| weight.is_finite()),
                "nonfinite route weight"
            );
            let sum: f32 = weights.iter().sum();
            ensure!(
                (sum - fixture.scale).abs() < 1e-4,
                "{} normalized scale {sum} differs from {}",
                fixture.mode,
                fixture.scale
            );
        }
        let (ids, weights) = noaux_tc_route(
            &[0.1, -0.2],
            &[0.0, 0.0],
            2,
            crate::model_spec::RouterScore::Sigmoid,
            1.0,
        )?;
        assert_eq!(ids.len(), 2);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        Ok(())
    }
}
