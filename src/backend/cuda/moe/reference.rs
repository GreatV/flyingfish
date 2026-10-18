//! Host fixture and numerics reference for the MoE operator gates.
//! Semantics follow the model definition, transformers 5.19 qwen3_5_moe
//! (Qwen3_5MoeTopKRouter, Qwen3_5MoeExperts, Qwen3_5MoeSparseMoeBlock):
//! router logits are FP32 accumulations rounded to BF16 (F.linear boundary),
//! softmax runs in FP32 over the BF16 logits, top-k is descending with exact
//! ties broken by the lower expert id, the selected weights are renormalised
//! in FP32 and cast to BF16. Expert gate/up outputs, SiLU, the gate·up
//! product, the down output and each weighted contribution round at BF16
//! boundaries; contributions accumulate in ascending expert id with BF16
//! rounding per addition (index_add_); sigmoid(x·shared_expert_gate) scales
//! the shared expert and the final sums round to BF16.
use half::bf16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeDims {
    pub rows: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub inter: usize,
}

impl MoeDims {
    pub fn new(rows: usize, experts: usize, top_k: usize, hidden: usize, inter: usize) -> Self {
        Self {
            rows,
            experts,
            top_k,
            hidden,
            inter,
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
/// shared_gate/shared_up `[inter, hidden]`, shared_down `[hidden, inter]`,
/// shared_expert_gate `[hidden]`.
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
    } = dims;
    assert!(rows >= 2 && experts >= 6 && top_k >= 4 && experts > top_k && h >= 8 && i >= 1);
    let wscale = 2.0 / (h as f32).sqrt();
    let dscale = 1.0 / (i as f32).sqrt();
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
    let mut shared_gate = Vec::with_capacity(i * h);
    let mut shared_up = Vec::with_capacity(i * h);
    for j in 0..i {
        for c in 0..h {
            shared_gate.push(bf16::from_f32(unit(seed(variant, 5), 0, j, c) * wscale));
            shared_up.push(bf16::from_f32(unit(seed(variant, 6), 0, j, c) * wscale));
        }
    }
    let mut shared_down = Vec::with_capacity(h * i);
    for d in 0..h {
        for j in 0..i {
            shared_down.push(bf16::from_f32(unit(seed(variant, 7), 0, d, j) * dscale));
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
    let probs = softmax_probs(&router_w, &x[near_row * h..(near_row + 1) * h], experts);
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

/// FP32 softmax over the BF16-rounded FP32 dot products of one row.
fn softmax_probs(router: &[bf16], xr: &[bf16], experts: usize) -> Vec<f32> {
    let h = xr.len();
    let logits: Vec<f32> = (0..experts)
        .map(|e| {
            let dot: f32 = xr
                .iter()
                .zip(&router[e * h..(e + 1) * h])
                .map(|(&a, &b)| a.to_f32() * b.to_f32())
                .sum();
            bf16::from_f32(dot).to_f32()
        })
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
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
    let probs = softmax_probs(&fx.router_w, &fx.x[row * h..(row + 1) * h], e);
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

fn expert_out(fx: &MoeFixture, e: usize, xr: &[bf16], weight: bf16) -> Vec<bf16> {
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
            let lin = bf16::from_f32(dot(&prod, fx.down_row(e, d)));
            bf16::from_f32(lin.to_f32() * weight.to_f32())
        })
        .collect()
}

fn shared_out(fx: &MoeFixture, xr: &[bf16]) -> Vec<bf16> {
    let (h, i) = (fx.dims.hidden, fx.dims.inter);
    let mut prod = Vec::with_capacity(i);
    for j in 0..i {
        let g = bf16::from_f32(dot(xr, &fx.shared_gate[j * h..(j + 1) * h]));
        let u = bf16::from_f32(dot(xr, &fx.shared_up[j * h..(j + 1) * h]));
        let s = bf16::from_f32(silu(g.to_f32()));
        prod.push(bf16::from_f32(s.to_f32() * u.to_f32()));
    }
    (0..h)
        .map(|d| bf16::from_f32(dot(&prod, &fx.shared_down[d * i..(d + 1) * i])))
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Combine {
    AscendingBf16,
    DescendingBf16,
    Fp32,
}

/// The row's shared expert (sigmoid-gated) plus the routed contributions
/// selected by `keep` (indices into `ids`/`weights`), accumulated per
/// `combine` with contributions ordered ascending expert id.
pub(crate) fn moe_row_ids(
    fx: &MoeFixture,
    row: usize,
    ids: &[i32],
    weights: &[bf16],
    keep: &[usize],
    combine: Combine,
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
        .map(|&(e, w)| expert_out(fx, e, xr, w))
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
                y[d] = bf16::from_f32(yf[d] + gated[d].to_f32());
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
        assert_eq!(noise.ascending_vs_fp32.to_bits(), 0.001953125f32.to_bits());
    }
}
