use anyhow::{Result, ensure};
use half::bf16;

pub(crate) const TRACE_LEN: usize = 886720;
const QK: usize = 0;
const SCORE: usize = 40960;
const P: usize = 81920;
const PB: usize = 122880;
const KV: usize = 163840;
const NUM: usize = 491520;
const DEN: usize = 819200;
const MAX: usize = 819840;
const FACTOR: usize = 820480;
const FINAL_NUM: usize = 821120;
const FINAL_DEN: usize = 853888;
const DIV: usize = 853952;
const U: f64 = 1.0 / 16777216.0;

fn gamma(n: usize) -> f64 {
    let x = n as f64 * U;
    x / (1.0 - x)
}
fn rn(x: f64) -> f64 {
    f64::from(x as f32)
}
fn bf(x: f64) -> f64 {
    f64::from(bf16::from_f32(x as f32).to_f32())
}

#[test]
fn certified_fp32_store_keeps_both_midpoint_sides() {
    let interval = Interval {
        lo: -0.20947273075580597,
        hi: -0.20947256684303284,
    }
    .bf16();
    assert_eq!(interval.lo, -0.2099609375);
    assert_eq!(interval.hi, -0.208984375);
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn pair(a: f64, b: f64) -> Self {
        Self {
            lo: a.min(b),
            hi: a.max(b),
        }
    }
    fn point(a: f64) -> Self {
        Self { lo: a, hi: a }
    }
    fn mul(self, b: Self) -> Self {
        let v = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self {
            lo: v.iter().copied().fold(f64::INFINITY, f64::min),
            hi: v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
    fn add(self, b: Self) -> Self {
        Self {
            lo: self.lo + b.lo,
            hi: self.hi + b.hi,
        }
    }
    fn max_abs(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
    fn widen(self, error: f64) -> Self {
        Self {
            lo: self.lo - error,
            hi: self.hi + error,
        }
    }
    fn rounded(self) -> Self {
        Self {
            lo: rn(self.lo),
            hi: rn(self.hi),
        }
    }
    fn bf16(self) -> Self {
        Self {
            lo: bf(self.lo),
            hi: bf(self.hi),
        }
    }
    fn contains(self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }
}

pub(crate) struct Inputs<'a> {
    pub layer: usize,
    pub position: usize,
    pub query: [&'a [u16]; 2],
    pub sink: &'a [f32],
    pub candidate: &'a [f32],
    pub publisher: &'a [f32],
    pub output: &'a [bf16],
    pub expected: &'a [u16],
}

struct Store {
    name: &'static str,
    block: usize,
    head: usize,
    column: usize,
    n: usize,
    index: usize,
    interval: Interval,
    exact: [f64; 2],
    sum_abs: [f64; 2],
    focus: bool,
}

pub(crate) struct Certificate {
    pub output_differences: usize,
    pub store_defects: usize,
    pub operand_defects: usize,
}

struct Proof<'a> {
    inputs: Inputs<'a>,
    result: Certificate,
}
impl Proof<'_> {
    fn pair(&self, index: usize) -> Interval {
        Interval::pair(self.a(index), self.b(index))
    }
    fn a(&self, index: usize) -> f64 {
        f64::from(self.inputs.candidate[index])
    }
    fn b(&self, index: usize) -> f64 {
        f64::from(self.inputs.publisher[index])
    }
    fn check(&mut self, store: Store) {
        self.values(&store, self.a(store.index), self.b(store.index));
    }
    fn values(&mut self, s: &Store, a: f64, b: f64) {
        let valid = s.interval.contains(a) && s.interval.contains(b);
        self.result.store_defects += usize::from(!valid);
        if !valid || (s.focus && a != b) {
            let source = match s.name {
                "qk" => "identical BF16 Q/KV; exact dot +/- gamma512 sum_abs",
                "score" => "certified QK interval times stored FP32 scale, RN32",
                "maximum" => "maximum over certified score intervals",
                "factor" => "exp of certified max difference, RN32; factor<=1",
                "probability32" => "exp of certified score-minus-max interval, RN32",
                "probability_bf16" => "BF16-rounded closure of certified probability interval",
                "numerator" => {
                    "certified P_BF16/KV/carry/factor; rescale U plus gamma_n(sum_abs+carry_abs)"
                }
                "denominator" => {
                    "certified P32/carry/factor; gamma_n sum_abs, rescale U, addition U"
                }
                "final_numerator" => "last certified numerator store",
                "final_denominator" => {
                    "recorded denominator plus sink exp from certified max; addition U"
                }
                "division" => "certified numerator/positive-denominator intervals, RN32",
                "output_bf16" => {
                    "BF16-rounded closure of certified division; no empirical tolerance"
                }
                _ => unreachable!(),
            };
            println!(
                "ATTENTION_STORE layer={} position={} store={} block={} head={} column={} n={} gamma={:e} lower={:e} upper={:e} candidate={a:e} publisher={b:e} exact_candidate={:e} exact_publisher={:e} sum_abs_candidate={:e} sum_abs_publisher={:e} both_inside={valid} source={source:?}",
                self.inputs.layer,
                self.inputs.position,
                s.name,
                s.block,
                s.head,
                s.column,
                s.n,
                gamma(s.n),
                s.interval.lo,
                s.interval.hi,
                s.exact[0],
                s.exact[1],
                s.sum_abs[0],
                s.sum_abs[1]
            );
        }
    }

    fn head(&mut self, head: usize) {
        let bad: Vec<_> = (0..512)
            .map(|d| {
                self.inputs.output[head * 512 + d].to_bits() != self.inputs.expected[head * 512 + d]
            })
            .collect();
        let focus = bad.iter().any(|&v| v);
        let mut running = Interval::point(f64::NEG_INFINITY);
        let mut num = vec![Interval::point(0.0); 512];
        let mut num_exact = vec![[0.0; 2]; 512];
        let mut num_sums = vec![[0.0; 2]; 512];
        let mut maximum_exact = [f64::NEG_INFINITY; 2];
        let mut last_n = 0;
        for block in 0..10 {
            let mut scores = [Interval::point(f64::NEG_INFINITY); 64];
            let mut n = 0;
            for (k, score) in scores.iter_mut().enumerate() {
                let i = (block * 64 + head) * 64 + k;
                let mut exact = [0.0; 2];
                let mut absolute = [0.0; 2];
                if self.b(QK + i).is_finite() {
                    n += 1;
                    for side in 0..2 {
                        for d in 0..512 {
                            let q = f64::from(
                                bf16::from_bits(self.inputs.query[side][head * 512 + d]).to_f32(),
                            );
                            let v = self.b(KV + (block * 64 + k) * 512 + d);
                            exact[side] += q * v;
                            absolute[side] += (q * v).abs();
                        }
                    }
                } else {
                    exact = [f64::NEG_INFINITY; 2];
                }
                let dot = Interval {
                    lo: (exact[0] - gamma(512) * absolute[0])
                        .min(exact[1] - gamma(512) * absolute[1]),
                    hi: (exact[0] + gamma(512) * absolute[0])
                        .max(exact[1] + gamma(512) * absolute[1]),
                };
                self.check(Store {
                    name: "qk",
                    block,
                    head,
                    column: k,
                    n: 512,
                    index: QK + i,
                    interval: dot.rounded(),
                    exact,
                    sum_abs: absolute,
                    focus,
                });
                maximum_exact[0] = maximum_exact[0].max(self.a(SCORE + i));
                maximum_exact[1] = maximum_exact[1].max(self.b(SCORE + i));
                let scale = f64::from((512.0f32).sqrt().recip());
                *score = Interval {
                    lo: rn(dot.lo * scale),
                    hi: rn(dot.hi * scale),
                };
                self.check(Store {
                    name: "score",
                    block,
                    head,
                    column: k,
                    n: 512,
                    index: SCORE + i,
                    interval: *score,
                    exact: [self.a(QK + i) * scale, self.b(QK + i) * scale],
                    sum_abs: [0.0; 2],
                    focus,
                });
            }
            last_n = n;
            let previous_max = running;
            for score in scores {
                running.lo = running.lo.max(score.lo);
                running.hi = running.hi.max(score.hi);
            }
            let i = block * 64 + head;
            self.check(Store {
                name: "maximum",
                block,
                head,
                column: 0,
                n: 0,
                index: MAX + i,
                interval: running,
                exact: maximum_exact,
                sum_abs: [0.0; 2],
                focus,
            });
            let factor = if block == 0 {
                Interval::point(0.0)
            } else {
                Interval {
                    lo: rn(rn(previous_max.lo - running.hi).exp()),
                    hi: rn(rn(previous_max.hi - running.lo).exp()).min(1.0),
                }
            };
            let factor_exact = if block == 0 {
                [0.0; 2]
            } else {
                [
                    rn(self.a(MAX + i - 64) - self.a(MAX + i)).exp(),
                    rn(self.b(MAX + i - 64) - self.b(MAX + i)).exp(),
                ]
            };
            self.check(Store {
                name: "factor",
                block,
                head,
                column: 0,
                n: 0,
                index: FACTOR + i,
                interval: factor,
                exact: factor_exact,
                sum_abs: [0.0; 2],
                focus,
            });
            for (k, score) in scores.iter().enumerate() {
                let j = (block * 64 + head) * 64 + k;
                let probability = Interval {
                    lo: rn(rn(score.lo - running.hi).exp()),
                    hi: rn(rn(score.hi - running.lo).exp()),
                };
                let exact = [
                    rn(self.a(SCORE + j) - self.a(MAX + i)).exp(),
                    rn(self.b(SCORE + j) - self.b(MAX + i)).exp(),
                ];
                for (name, offset, interval) in [
                    ("probability32", P, probability),
                    ("probability_bf16", PB, probability.bf16()),
                ] {
                    self.check(Store {
                        name,
                        block,
                        head,
                        column: k,
                        n: 512,
                        index: offset + j,
                        interval,
                        exact,
                        sum_abs: [0.0; 2],
                        focus,
                    });
                }
            }
            for d in 0..512 {
                let j = (block * 64 + head) * 512 + d;
                let previous = if block == 0 {
                    [0.0; 2]
                } else {
                    [self.a(NUM + j - 64 * 512), self.b(NUM + j - 64 * 512)]
                };
                let carry = Interval::pair(previous[0], previous[1]).mul(self.pair(FACTOR + i));
                let carry = carry.widen(U * carry.max_abs());
                let mut dot = Interval::point(0.0);
                let mut absolute = 0.0;
                let mut exact = [
                    previous[0] * self.a(FACTOR + i),
                    previous[1] * self.b(FACTOR + i),
                ];
                let mut sums = [exact[0].abs(), exact[1].abs()];
                for k in 0..64 {
                    let v = self.b(KV + (block * 64 + k) * 512 + d);
                    let j = (block * 64 + head) * 64 + k;
                    let probability = self.pair(PB + j);
                    dot = dot.add(probability.mul(Interval::point(v)));
                    absolute += probability.max_abs() * v.abs();
                    let terms = [self.a(PB + j) * v, self.b(PB + j) * v];
                    for side in 0..2 {
                        exact[side] += terms[side];
                        sums[side] += terms[side].abs();
                    }
                }
                num[d] = carry
                    .add(dot)
                    .widen(gamma(n) * (absolute + carry.max_abs()))
                    .rounded();
                num_exact[d] = exact;
                num_sums[d] = sums;
                self.check(Store {
                    name: "numerator",
                    block,
                    head,
                    column: d,
                    n,
                    index: NUM + j,
                    interval: num[d],
                    exact,
                    sum_abs: sums,
                    focus: bad[d],
                });
            }
            let previous = if block == 0 {
                [0.0; 2]
            } else {
                [self.a(DEN + i - 64), self.b(DEN + i - 64)]
            };
            let carry = Interval::pair(previous[0], previous[1]).mul(self.pair(FACTOR + i));
            let carry = carry.widen(U * carry.max_abs());
            let mut sum = Interval::point(0.0);
            let mut absolute = 0.0;
            let mut exact = [
                previous[0] * self.a(FACTOR + i),
                previous[1] * self.b(FACTOR + i),
            ];
            let mut sums = [exact[0].abs(), exact[1].abs()];
            for k in 0..64 {
                let j = (block * 64 + head) * 64 + k;
                let p = self.pair(P + j);
                sum = sum.add(p);
                absolute += p.max_abs();
                let terms = [self.a(P + j), self.b(P + j)];
                for side in 0..2 {
                    exact[side] += terms[side];
                    sums[side] += terms[side].abs();
                }
            }
            let den = carry.add(sum.widen(gamma(n) * absolute));
            let den = den.widen(U * den.max_abs()).rounded();
            self.check(Store {
                name: "denominator",
                block,
                head,
                column: 0,
                n,
                index: DEN + i,
                interval: den,
                exact,
                sum_abs: sums,
                focus,
            });
        }
        let sink = f64::from(self.inputs.sink[head]);
        let sink_interval = Interval {
            lo: rn(rn(sink - running.hi).exp()),
            hi: rn(rn(sink - running.lo).exp()),
        };
        let den = self.pair(DEN + 9 * 64 + head).add(sink_interval);
        let den = den.widen(U * den.max_abs()).rounded();
        let exact = [
            self.a(DEN + 9 * 64 + head) + rn(sink - self.a(MAX + 9 * 64 + head)).exp(),
            self.b(DEN + 9 * 64 + head) + rn(sink - self.b(MAX + 9 * 64 + head)).exp(),
        ];
        self.check(Store {
            name: "final_denominator",
            block: 9,
            head,
            column: 0,
            n: 1,
            index: FINAL_DEN + head,
            interval: den,
            exact,
            sum_abs: [0.0; 2],
            focus,
        });
        if den.lo <= 0.0 || !den.hi.is_finite() {
            self.result.store_defects += 1;
            println!(
                "ATTENTION_STORE_INVALID layer={} position={} head={head} denominator_interval_positive=false",
                self.inputs.layer, self.inputs.position
            );
            return;
        }
        for d in 0..512 {
            let i = head * 512 + d;
            self.check(Store {
                name: "final_numerator",
                block: 9,
                head,
                column: d,
                n: last_n,
                index: FINAL_NUM + i,
                interval: num[d],
                exact: num_exact[d],
                sum_abs: num_sums[d],
                focus: bad[d],
            });
            let division = num[d]
                .mul(Interval {
                    lo: 1.0 / den.hi,
                    hi: 1.0 / den.lo,
                })
                .rounded();
            let exact = [
                self.a(FINAL_NUM + i) / self.a(FINAL_DEN + head),
                self.b(FINAL_NUM + i) / self.b(FINAL_DEN + head),
            ];
            self.check(Store {
                name: "division",
                block: 9,
                head,
                column: d,
                n: 1,
                index: DIV + i,
                interval: division,
                exact,
                sum_abs: [0.0; 2],
                focus: bad[d],
            });
            self.values(
                &Store {
                    name: "output_bf16",
                    block: 9,
                    head,
                    column: d,
                    n: 1,
                    index: 0,
                    interval: division.bf16(),
                    exact,
                    sum_abs: [0.0; 2],
                    focus: bad[d],
                },
                f64::from(self.inputs.output[i].to_f32()),
                f64::from(bf16::from_bits(self.inputs.expected[i]).to_f32()),
            );
        }
    }
}

pub(crate) fn prove(inputs: Inputs<'_>) -> Result<Certificate> {
    ensure!(
        inputs.query.iter().all(|q| q.len() == 64 * 512)
            && inputs.sink.len() == 64
            && inputs.output.len() == 64 * 512
            && inputs.expected.len() == 64 * 512
            && inputs.candidate.len() == TRACE_LEN
            && inputs.publisher.len() == TRACE_LEN,
        "attention certificate extent mismatch"
    );
    let output_differences = inputs
        .output
        .iter()
        .zip(inputs.expected)
        .filter(|(a, b)| a.to_bits() != **b)
        .count();
    let operand_defects = inputs.candidate[KV..NUM]
        .iter()
        .zip(&inputs.publisher[KV..NUM])
        .filter(|(a, b)| a != b)
        .count();
    let mut proof = Proof {
        inputs,
        result: Certificate {
            output_differences,
            store_defects: 0,
            operand_defects,
        },
    };
    if operand_defects == 0 {
        for head in 0..64 {
            proof.head(head);
        }
    }
    println!(
        "ATTENTION_CERTIFICATE layer={} position={} output_differences={} operand_defects={} store_defects={}",
        proof.inputs.layer,
        proof.inputs.position,
        proof.result.output_differences,
        proof.result.operand_defects,
        proof.result.store_defects
    );
    Ok(proof.result)
}
