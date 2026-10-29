//! GLM all-layer isolated-boundary candidate gate over the shared mHC runtime
//! and native MoE executor.

use crate::backend::cuda::Device;
use crate::backend::cuda::contracts::view;
use crate::backend::cuda::hc;
use crate::backend::cuda::linear::{Fp8Ops, LinearWeightRef, NativeProjection};
use crate::backend::cuda::weights::Loader;
use crate::backend::gpu::CudaSlice;
use crate::backend::gpu::{CudaStream, DevicePtr};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

/// Frozen bands: four times the measured model-definition/FP64 boundary floor.
pub struct FrozenThresholds {
    bands: BTreeMap<String, f64>,
}

fn load_projection(
    loader: &mut Loader<'_>,
    consumed: &mut BTreeSet<String>,
    base: &str,
    rows: usize,
    columns: usize,
) -> Result<(CudaSlice<u8>, CudaSlice<u8>)> {
    let codes = loader.load_bytes(&format!("{base}.weight"), &[rows, columns], Dtype::F8_E4M3)?;
    let scales = loader.load_bytes(
        &format!("{base}.weight_scale_inv"),
        &[rows.div_ceil(128), columns.div_ceil(128)],
        Dtype::F32,
    )?;
    mark(consumed, format!("{base}.weight"))?;
    mark(consumed, format!("{base}.weight_scale_inv"))?;
    Ok((codes, scales))
}

fn mark(consumed: &mut BTreeSet<String>, name: String) -> Result<()> {
    ensure!(
        consumed.insert(name.clone()),
        "tensor {name} consumed twice"
    );
    Ok(())
}

fn expert_codes(projections: [(CudaSlice<u8>, CudaSlice<u8>); 3]) -> ExpertCodes {
    let [gate, up, down] = projections;
    ExpertCodes {
        codes: [gate.0, up.0, down.0],
        scales: [gate.1, up.1, down.1],
    }
}

impl FrozenThresholds {
    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading frozen thresholds from {}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).context("parsing thresholds.json")?;
        ensure!(
            value.get("frozen").and_then(|v| v.as_bool()) == Some(true),
            "thresholds.json frozen=false; refusing to gate against an unfrozen record"
        );
        let bands_value = value
            .get("bands")
            .and_then(|v| v.as_object())
            .context("thresholds.json has no bands object")?;
        let mut bands = BTreeMap::new();
        for (key, value) in bands_value {
            let band = value
                .as_f64()
                .with_context(|| format!("band {key} is not numeric"))?;
            ensure!(band.is_finite() && band >= 0.0, "invalid band {key}={band}");
            bands.insert(key.clone(), band);
        }
        Ok(Self { bands })
    }

    fn band(&self, layer: usize, boundary: &str) -> Result<f64> {
        let key = format!("layer{layer}.{boundary}");
        self.bands.get(&key).copied().with_context(|| {
            format!(
                "boundary {key} is not in the frozen bands ({:?})",
                self.bands.keys().collect::<Vec<_>>()
            )
        })
    }
}

/// Rule A is the isolated-boundary max; Rule B counts edge-adjacent values
/// without changing the threshold.
struct BoundaryCheck {
    max_abs: f64,
    near_tie_fraction: f64,
    band: f64,
    defect: bool,
}

fn check_boundary(values: &[f64], band: f64) -> BoundaryCheck {
    let max_abs = values.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let half_edge = band / 512.0;
    let near = values
        .iter()
        .filter(|&&v| (v.abs() - band).abs() <= half_edge)
        .count();
    BoundaryCheck {
        max_abs,
        near_tie_fraction: if values.is_empty() {
            0.0
        } else {
            near as f64 / values.len() as f64
        },
        band,
        defect: max_abs > band,
    }
}

fn assert_boundary(
    thresholds: &FrozenThresholds,
    layer: usize,
    boundary: &str,
    actual: &[f64],
    expected: &[f64],
) -> Result<BoundaryCheck> {
    ensure!(
        actual.len() == expected.len(),
        "{boundary} extent mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    let differences: Vec<f64> = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| f64::from(a as f32) - e)
        .collect();
    ensure!(
        differences.iter().all(|v| v.is_finite()),
        "{boundary} contains a nonfinite candidate value"
    );
    let band = thresholds.band(layer, boundary)?;
    let result = check_boundary(&differences, band);
    Ok(result)
}

fn fixture_tensors(bytes: &'_ [u8]) -> Result<safetensors::SafeTensors<'_>> {
    SafeTensors::deserialize(bytes).context("deserialize activation fixture")
}

fn fixture_f32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f64>> {
    let tensor = st.tensor(name).with_context(|| format!("fixture {name}"))?;
    ensure!(tensor.dtype() == Dtype::F32, "{name} is not F32");
    Ok(tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f64::from(f32::from_le_bytes(*v)))
        .collect())
}

fn fixture_bf16(st: &SafeTensors<'_>, name: &str) -> Result<Vec<bf16>> {
    let tensor = st.tensor(name).with_context(|| format!("fixture {name}"))?;
    ensure!(tensor.dtype() == Dtype::BF16, "{name} is not BF16");
    Ok(tensor
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
        .collect())
}

fn bf16_values(values: &[bf16]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v.to_f32())).collect()
}

#[derive(Clone, Copy)]
struct CollapseExtent {
    rows: usize,
    copies: usize,
    hidden: usize,
}

/// Fp32 FMA reduction and its exact FP64 companion for the four-copy mHC
/// collapse. The FP64 row supplies Rule B's accumulator on recorded operands.
fn recorded_pre_collapse(
    stream: &[bf16],
    pre: &[f64],
    extent: CollapseExtent,
) -> Result<(Vec<bf16>, Vec<f64>, Vec<f64>)> {
    let CollapseExtent {
        rows,
        copies,
        hidden,
    } = extent;
    ensure!(
        stream.len() == rows * copies * hidden && pre.len() == rows * copies,
        "recorded-pre collapse operand extent mismatch"
    );
    let mut rounded = Vec::with_capacity(rows * hidden);
    let mut exact = Vec::with_capacity(rows * hidden);
    let mut operand_abs_sum = Vec::with_capacity(rows * hidden);
    for row in 0..rows {
        for column in 0..hidden {
            let mut fused = 0.0f32;
            let mut accumulator = 0.0f64;
            let mut absolute = 0.0f64;
            for copy in 0..copies {
                let coefficient = pre[row * copies + copy];
                let operand = f64::from(stream[(row * copies + copy) * hidden + column].to_f32());
                fused = (coefficient as f32).mul_add(operand as f32, fused);
                let product = coefficient * operand;
                accumulator += product;
                absolute += product.abs();
            }
            rounded.push(bf16::from_f32(fused));
            exact.push(accumulator);
            operand_abs_sum.push(absolute);
        }
    }
    Ok((rounded, exact, operand_abs_sum))
}

fn recorded_pre_boundary(
    thresholds: &FrozenThresholds,
    layer: usize,
    boundary: &str,
    operands: (&[bf16], &[f64], &[bf16]),
    extent: CollapseExtent,
) -> Result<(BoundaryCheck, usize, usize, Vec<String>)> {
    let (stream, pre, expected) = operands;
    let (candidate, exact, operand_abs_sum) = recorded_pre_collapse(stream, pre, extent)?;
    ensure!(
        candidate.len() == expected.len(),
        "{boundary} recorded-pre extent mismatch: replay={} record={}",
        candidate.len(),
        expected.len()
    );
    let differences: Vec<f64> = candidate
        .iter()
        .zip(expected)
        .map(|(&actual, &record)| f64::from(actual.to_f32()) - f64::from(record.to_f32()))
        .collect();
    ensure!(
        differences.iter().all(|value| value.is_finite()),
        "{boundary} contains a nonfinite recorded-pre difference"
    );
    let band = thresholds.band(layer, boundary)?;
    let mut result = check_boundary(&differences, band);
    let unit_roundoff = 2.0f64.powi(-24);
    let terms = extent.copies as f64;
    let gamma_n = terms * unit_roundoff / (1.0 - terms * unit_roundoff);
    let mut exempt = 0usize;
    let mut defects = 0usize;
    let mut proofs = Vec::new();
    for index in 0..candidate.len() {
        if differences[index].abs() <= band {
            continue;
        }
        let reduction_bound = gamma_n * operand_abs_sum[index];
        let exact = exact[index];
        let raw_lower = exact - reduction_bound;
        let raw_upper = exact + reduction_bound;
        let lower_store = f64::from(bf16::from_f32(raw_lower as f32).to_f32());
        let upper_store = f64::from(bf16::from_f32(raw_upper as f32).to_f32());
        let lower = raw_lower.min(lower_store);
        let upper = raw_upper.max(upper_store);
        let candidate_value = f64::from(candidate[index].to_f32());
        let publisher_value = f64::from(expected[index].to_f32());
        let exempted = (candidate_value >= lower && candidate_value <= upper)
            && (publisher_value >= lower && publisher_value <= upper);
        if exempted {
            exempt += 1;
        } else {
            defects += 1;
        }
        proofs.push(format!(
            "index={index}:exact={exact:e}:reduction_bound={reduction_bound:e}:interval=[{lower:e},{upper:e}]:candidate={candidate_value:e}:publisher={publisher_value:e}:abs={:e}:exempt={exempted}",
            differences[index],
        ));
    }
    result.defect = defects > 0;
    Ok((result, exempt, defects, proofs))
}

#[derive(Clone, Copy)]
struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    fn new(center: f64, radius: f64) -> Self {
        Self {
            lower: center - radius,
            upper: center + radius,
        }
    }

    fn scale_add(self, factor: f64, addend: f64) -> Self {
        let (lower_product, upper_product) = if factor >= 0.0 {
            (self.lower * factor, self.upper * factor)
        } else {
            (self.upper * factor, self.lower * factor)
        };
        Self {
            lower: lower_product + addend,
            upper: upper_product + addend,
        }
    }
}

#[derive(Clone)]
struct TaylorForm {
    center: f64,
    gradient: Vec<f64>,
    nonlinear: f64,
    rounding: f64,
}

fn gamma(terms: usize) -> f64 {
    let unit_roundoff = 2.0f64.powi(-24);
    let terms = terms as f64;
    terms * unit_roundoff / (1.0 - terms * unit_roundoff)
}

impl TaylorForm {
    fn bound(&self) -> f64 {
        self.gradient
            .iter()
            .fold(self.nonlinear + self.rounding, |a, value| a + value.abs())
    }

    fn add(self, other: &Self) -> Self {
        Self {
            center: self.center + other.center,
            gradient: self
                .gradient
                .iter()
                .zip(&other.gradient)
                .map(|(a, b)| a + b)
                .collect(),
            nonlinear: self.nonlinear + other.nonlinear,
            rounding: self.rounding + other.rounding,
        }
    }

    fn scale_round(self, factor: f64) -> Self {
        let before = self.center.abs() + self.bound();
        let rounding = gamma(1) * before * factor.abs();
        Self {
            center: self.center * factor,
            gradient: self.gradient.iter().map(|value| value * factor).collect(),
            nonlinear: self.nonlinear * factor.abs(),
            rounding: self.rounding * factor.abs() + rounding,
        }
    }

    fn add_round(self, value: f64) -> Self {
        let rounding = gamma(1) * (self.center.abs() + self.bound() + value.abs());
        Self {
            center: self.center + value,
            gradient: self.gradient,
            nonlinear: self.nonlinear,
            rounding: self.rounding + rounding,
        }
    }

    fn exp_round(self) -> Self {
        let delta = self.bound();
        let center = self.center.exp();
        let nonlinear = 0.5 * (self.center + delta).exp() * delta * delta;
        let upper = center + center.abs() * delta + nonlinear;
        let rounding = gamma(1) * upper;
        Self {
            center,
            gradient: self.gradient.iter().map(|value| value * center).collect(),
            nonlinear,
            rounding,
        }
    }

    fn subtract_round(self, other: &Self) -> Self {
        let mut result = self.add(&Self {
            center: -other.center,
            gradient: other.gradient.iter().map(|value| -value).collect(),
            nonlinear: other.nonlinear,
            rounding: other.rounding,
        });
        result.rounding += gamma(1) * result.center.abs() + gamma(1) * result.bound();
        result
    }

    fn sum_round(forms: &[Self]) -> Result<Self> {
        let mut sum = forms.first().context("mHC sum has no operands")?.clone();
        for form in &forms[1..] {
            sum = sum.add(form);
        }
        sum.rounding +=
            gamma(forms.len() - 1) * sum.center.abs() + gamma(forms.len() - 1) * sum.bound();
        Ok(sum)
    }

    fn divide_round(self, denominator: &Self) -> Result<Self> {
        let numerator_delta = self.bound();
        let denominator_delta = denominator.bound();
        let denominator_lower = denominator.center - denominator_delta;
        ensure!(
            denominator.center.is_finite()
                && denominator_lower.is_finite()
                && denominator_lower > 0.0,
            "mHC Sinkhorn denominator interval includes zero"
        );
        let denominator_square = denominator.center * denominator.center;
        let center = self.center / denominator.center;
        let gradient = self
            .gradient
            .iter()
            .zip(&denominator.gradient)
            .map(|(numerator, denominator_value)| {
                (numerator * denominator.center - self.center * denominator_value)
                    / denominator_square
            })
            .collect::<Vec<_>>();
        let numerator_upper = self.center.abs() + numerator_delta;
        let cross = numerator_delta * denominator_delta / (denominator_lower * denominator_lower);
        let second = numerator_upper * denominator_delta * denominator_delta
            / (denominator_lower * denominator_lower * denominator_lower);
        let nonlinear = cross + second;
        let quotient_upper = numerator_upper / denominator_lower;
        let rounding = gamma(1) * quotient_upper;
        Ok(Self {
            center,
            gradient,
            nonlinear,
            rounding,
        })
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        value.exp() / (1.0 + value.exp())
    }
}

fn max_sigmoid_prime(lower: f64, upper: f64) -> f64 {
    if lower <= 0.0 && upper >= 0.0 {
        0.25
    } else {
        let edge = if lower > 0.0 { lower } else { upper };
        sigmoid(edge) * (1.0 - sigmoid(edge))
    }
}

fn f32_floor(value: f64) -> f64 {
    let rounded = value as f32;
    if f64::from(rounded) <= value {
        f64::from(rounded)
    } else {
        f64::from(f32::from_bits(rounded.to_bits() - 1))
    }
}

fn f32_ceiling(value: f64) -> f64 {
    let rounded = value as f32;
    if f64::from(rounded) >= value {
        f64::from(rounded)
    } else {
        f64::from(f32::from_bits(rounded.to_bits() + 1))
    }
}

fn outward_f32_interval(lower: f64, upper: f64) -> (f64, f64) {
    (f32_floor(lower), f32_ceiling(upper))
}

struct HcHostWeights {
    fn_w: Vec<bf16>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

impl HcHostWeights {
    fn read(stream: &Arc<CudaStream>, weights: &hc::HcSublayerWeights) -> Result<Self> {
        let fn_bytes = stream.clone_dtoh(&weights.fn_w)?;
        let base_bytes = stream.clone_dtoh(&weights.base)?;
        let scale_bytes = stream.clone_dtoh(&weights.scale)?;
        let fn_w = fn_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|value| bf16::from_bits(u16::from_le_bytes(*value)))
            .collect();
        let base = base_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect();
        let scale = scale_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect();
        Ok(Self { fn_w, base, scale })
    }
}

struct CoefficientRow {
    post_exact: Vec<f64>,
    post_interval: Vec<(f64, f64)>,
    comb_exact: Vec<f64>,
    comb_interval: Vec<CoefficientInterval>,
}

#[derive(Clone, Copy, Debug)]
struct CoefficientInterval {
    lower: f64,
    upper: f64,
    jacobian: f64,
    nonlinear: f64,
    rounding: f64,
}

impl CoefficientRow {
    fn new(
        stream: &[bf16],
        weights: &HcHostWeights,
        params: &crate::backend::cuda::mhc::HcParams,
    ) -> Result<Self> {
        let mult = params.mult as usize;
        let hidden = params.hidden as usize;
        let width = mult * hidden;
        let rows = (2 + mult) * mult;
        ensure!(
            stream.len() >= width
                && weights.fn_w.len() == rows * width
                && weights.base.len() == rows
                && weights.scale.len() == 3,
            "mHC coefficient operand extent mismatch"
        );
        let mut square_sum = 0.0f64;
        for &value in &stream[..width] {
            let value = f64::from(value.to_f32());
            square_sum += value * value;
        }
        let inverse_root = 1.0 / (square_sum / width as f64 + f64::from(params.norm_eps)).sqrt();
        let unit_roundoff = 2.0f64.powi(-24);
        let gamma_n = width as f64 * unit_roundoff / (1.0 - width as f64 * unit_roundoff);
        let mut mix = Vec::with_capacity(rows);
        let mut mix_interval = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut exact = 0.0f64;
            let mut absolute = 0.0f64;
            for (&weight, &value) in weights.fn_w[row * width..(row + 1) * width]
                .iter()
                .zip(&stream[..width])
            {
                let weight = f64::from(weight.to_f32());
                let value = f64::from(value.to_f32());
                exact += weight * value;
                absolute += (weight * value).abs();
            }
            let center = exact * inverse_root;
            let radius = gamma_n * absolute * inverse_root;
            mix.push(center);
            mix_interval.push(Interval::new(center, radius));
        }
        ensure!(
            mix.iter().all(|value| value.is_finite())
                && mix_interval
                    .iter()
                    .all(|value| value.lower.is_finite() && value.upper.is_finite()),
            "nonfinite exact mHC coefficient operand"
        );
        ensure!(
            mix.iter()
                .zip(&mix_interval)
                .all(|(&exact, interval)| interval.lower <= exact && exact <= interval.upper),
            "mHC projection interval excludes its exact value"
        );
        let post_exact = (0..mult)
            .map(|index| {
                2.0 * sigmoid(
                    mix[mult + index] * f64::from(weights.scale[1])
                        + f64::from(weights.base[mult + index]),
                )
            })
            .collect::<Vec<_>>();
        let post_interval = (0..mult)
            .map(|index| {
                let center = mix[mult + index];
                let interval = mix_interval[mult + index];
                let scaled = interval.scale_add(
                    f64::from(weights.scale[1]),
                    f64::from(weights.base[mult + index]),
                );
                let radius = 2.0
                    * f64::from(weights.scale[1]).abs()
                    * (center - interval.lower)
                    * max_sigmoid_prime(scaled.lower, scaled.upper);
                let exact = post_exact[index];
                outward_f32_interval(exact - radius, exact + radius)
            })
            .collect::<Vec<_>>();
        let mut z_forms = Vec::with_capacity(mult * mult);
        for index in 0..mult * mult {
            let mix_index = 2 * mult + index;
            let radius = mix_interval[mix_index].upper - mix[mix_index];
            let mut gradient = vec![0.0f64; mult * mult];
            gradient[index] = radius;
            let dot = TaylorForm {
                center: mix[mix_index],
                gradient,
                nonlinear: 0.0,
                rounding: 0.0,
            };
            let z = dot
                .scale_round(f64::from(weights.scale[2]))
                .add_round(f64::from(weights.base[mix_index]));
            z_forms.push(z);
        }
        let mut matrix = Vec::with_capacity(mult * mult);
        for row_start in (0..mult * mult).step_by(mult) {
            let row = &z_forms[row_start..row_start + mult];
            let max_index = (0..mult)
                .max_by(|a, b| row[*a].center.total_cmp(&row[*b].center))
                .context("mHC softmax row has no maximum")?;
            let max_form = &row[max_index];
            let separated = row.iter().enumerate().all(|(index, form)| {
                index == max_index
                    || max_form.center - max_form.bound() > form.center + form.bound()
            });
            let exponentials = if separated {
                row.iter()
                    .map(|form| form.clone().subtract_round(max_form).exp_round())
                    .collect::<Vec<_>>()
            } else {
                row.iter()
                    .map(|form| form.clone().exp_round())
                    .collect::<Vec<_>>()
            };
            let denominator = TaylorForm::sum_round(&exponentials)?;
            for numerator in exponentials {
                matrix.push(
                    numerator
                        .divide_round(&denominator)?
                        .add_round(f64::from(params.eps)),
                );
            }
        }
        fn normalize_rows_taylor(matrix: &mut [TaylorForm], mult: usize, eps: f64) -> Result<()> {
            for row in matrix.chunks_mut(mult) {
                let denominator = TaylorForm::sum_round(row)?.add_round(eps);
                for form in row.iter_mut() {
                    *form = form.clone().divide_round(&denominator)?;
                }
            }
            Ok(())
        }
        fn normalize_columns_taylor(
            matrix: &mut [TaylorForm],
            mult: usize,
            eps: f64,
        ) -> Result<()> {
            for column in 0..mult {
                let values = (0..mult)
                    .map(|row| matrix[row * mult + column].clone())
                    .collect::<Vec<_>>();
                let denominator = TaylorForm::sum_round(&values)?.add_round(eps);
                for row in 0..mult {
                    let index = row * mult + column;
                    matrix[index] = matrix[index].clone().divide_round(&denominator)?;
                }
            }
            Ok(())
        }
        fn require_finite(matrix: &[TaylorForm]) -> Result<()> {
            for form in matrix {
                let radius = form.bound();
                ensure!(
                    form.center.is_finite()
                        && radius.is_finite()
                        && form.center - radius >= 0.0
                        && form.center + radius > 0.0,
                    "nonfinite mHC Sinkhorn coefficient interval"
                );
                ensure!(
                    form.center - radius <= form.center && form.center <= form.center + radius,
                    "mHC Sinkhorn interval excludes its exact value"
                );
            }
            Ok(())
        }
        require_finite(&matrix)?;
        normalize_columns_taylor(&mut matrix, mult, f64::from(params.eps))?;
        require_finite(&matrix)?;
        for _ in 1..params.iters as usize {
            normalize_rows_taylor(&mut matrix, mult, f64::from(params.eps))?;
            normalize_columns_taylor(&mut matrix, mult, f64::from(params.eps))?;
            require_finite(&matrix)?;
        }
        let comb_exact = matrix.iter().map(|form| form.center).collect::<Vec<_>>();
        let comb_interval = matrix
            .iter()
            .map(|form| {
                let jacobian = form.gradient.iter().map(|value| value.abs()).sum::<f64>();
                let radius = jacobian + form.nonlinear + form.rounding;
                let (lower, upper) =
                    outward_f32_interval(form.center - radius, form.center + radius);
                CoefficientInterval {
                    lower,
                    upper,
                    jacobian,
                    nonlinear: form.nonlinear,
                    rounding: form.rounding,
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            post_exact,
            post_interval,
            comb_exact,
            comb_interval,
        })
    }
}
struct CoefficientBoundary<'a> {
    thresholds: &'a FrozenThresholds,
    layer: usize,
    boundary: &'a str,
    candidate: &'a [f64],
    publisher: &'a [f64],
    stream: &'a [bf16],
    weights: &'a HcHostWeights,
    params: crate::backend::cuda::mhc::HcParams,
}

fn coefficient_boundary(
    check: CoefficientBoundary<'_>,
) -> Result<(BoundaryCheck, usize, usize, Vec<String>)> {
    let CoefficientBoundary {
        thresholds,
        layer,
        boundary,
        candidate,
        publisher,
        stream,
        weights,
        params,
    } = check;
    ensure!(
        candidate.len() == publisher.len(),
        "{boundary} extent mismatch"
    );
    let differences = candidate
        .iter()
        .zip(publisher)
        .map(|(&actual, &expected)| actual - expected)
        .collect::<Vec<_>>();
    ensure!(
        differences.iter().all(|value| value.is_finite()),
        "{boundary} contains a nonfinite coefficient"
    );
    let band = thresholds.band(layer, boundary)?;
    let mut result = check_boundary(&differences, band);
    let mult = params.mult as usize;
    let width = if boundary.ends_with(".0") {
        mult
    } else {
        mult * mult
    };
    let beyond = differences
        .iter()
        .enumerate()
        .filter(|(_, value)| value.abs() > band)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut rows = BTreeMap::new();
    let mut proof_rows = BTreeSet::new();
    for index in &beyond {
        let row = index / width;
        proof_rows.insert(row);
    }
    for row in proof_rows {
        let start = row * mult * params.hidden as usize;
        rows.insert(
            row,
            CoefficientRow::new(&stream[start..], weights, &params)?,
        );
    }
    let mut exempt = 0usize;
    let mut defects = 0usize;
    let mut proofs = Vec::new();
    for index in beyond {
        let row = index / width;
        let column = index % width;
        let proof = &rows[&row];
        let (exact, interval, method) = if boundary.ends_with(".0") {
            let exact = proof.post_exact[column];
            let (lower, upper) = proof.post_interval[column];
            (
                exact,
                CoefficientInterval {
                    lower,
                    upper,
                    jacobian: (upper - lower) / 2.0,
                    nonlinear: 0.0,
                    rounding: 0.0,
                },
                "sigmoid_derivative",
            )
        } else {
            (
                proof.comb_exact[column],
                proof.comb_interval[column],
                "jacobian_taylor_remainder_fp32_rounding",
            )
        };
        ensure!(
            exact.is_finite()
                && interval.lower.is_finite()
                && interval.upper.is_finite()
                && interval.lower > 0.0
                && interval.lower <= exact
                && exact <= interval.upper,
            "nonfinite or inward mHC coefficient interval at index {index}"
        );
        let candidate_value = candidate[index];
        let publisher_value = publisher[index];
        let exempted = (candidate_value >= interval.lower && candidate_value <= interval.upper)
            && (publisher_value >= interval.lower && publisher_value <= interval.upper);
        if exempted {
            exempt += 1;
        } else {
            defects += 1;
        }
        proofs.push(format!(
            "index={index}:method={method}:exact={exact:e}:jacobian={:e}:nonlinear={:e}:rounding={:e}:interval=[{:e},{:e}]:candidate={candidate_value:e}:publisher={publisher_value:e}:abs={:e}:exempt={exempted}",
            interval.jacobian,
            interval.nonlinear,
            interval.rounding,
            interval.lower,
            interval.upper,
            differences[index],
        ));
    }
    result.defect = defects > 0;
    Ok((result, exempt, defects, proofs))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateEnv {
    Model,
    FixtureDir,
    Thresholds,
}

fn env(name: GateEnv) -> Result<PathBuf> {
    let key = match name {
        GateEnv::Model => "FF_GLM53_MODEL",
        GateEnv::FixtureDir => "FF_GLM53_LAYER_FIXTURE",
        GateEnv::Thresholds => "FF_GLM53_THRESHOLDS",
    };
    let value = std::env::var_os(key)
        .with_context(|| format!("{key} is required by the isolated-boundary gate"))?;
    Ok(PathBuf::from(value))
}

struct ExpertCodes {
    codes: [CudaSlice<u8>; 3],
    scales: [CudaSlice<u8>; 3],
}

struct GlmGateWeights {
    hc: [hc::HcSublayerWeights; 2],
    norms: [CudaSlice<bf16>; 2],
    ffn: GlmGateFfnWeights,
}

enum GlmGateFfnWeights {
    Dense(ExpertCodes),
    Moe {
        router: CudaSlice<bf16>,
        bias: CudaSlice<u8>,
        experts: Vec<ExpertCodes>,
        shared: ExpertCodes,
    },
}

fn load_layer_weights(
    mapped: &crate::checkpoint::MappedCheckpoint,
    device: &Device,
    spec: &crate::model_spec::ModelSpec,
    layer: usize,
) -> Result<GlmGateWeights> {
    let manifest = mapped.manifest(spec, crate::checkpoint::LoadPolicy::TextOnly)?;
    let dense_layers = spec
        .glm5_next
        .as_ref()
        .context("GLM configuration missing")?
        .dense_layers as usize;
    let hidden = spec.hidden as usize;
    let prefix = format!("model.language_model.layers.{layer}");
    let mut consumed = BTreeSet::new();
    fn mark(consumed: &mut BTreeSet<String>, name: String) -> Result<()> {
        ensure!(
            consumed.insert(name.clone()),
            "tensor {name} consumed twice"
        );
        Ok(())
    }
    let weights = crate::backend::cuda::weights::checkpoint_mapped(
        mapped,
        device,
        Some(&manifest),
        false,
        |loader| {
            let hc = [
                hc::HcSublayerWeights {
                    fn_w: loader.load_bytes(
                        &format!("{prefix}.hc_attn_fn"),
                        &[24, 4 * hidden],
                        Dtype::BF16,
                    )?,
                    base: loader.load_bytes(
                        &format!("{prefix}.hc_attn_base"),
                        &[24],
                        Dtype::F32,
                    )?,
                    scale: loader.load_bytes(
                        &format!("{prefix}.hc_attn_scale"),
                        &[3],
                        Dtype::F32,
                    )?,
                },
                hc::HcSublayerWeights {
                    fn_w: loader.load_bytes(
                        &format!("{prefix}.hc_ffn_fn"),
                        &[24, 4 * hidden],
                        Dtype::BF16,
                    )?,
                    base: loader.load_bytes(&format!("{prefix}.hc_ffn_base"), &[24], Dtype::F32)?,
                    scale: loader.load_bytes(
                        &format!("{prefix}.hc_ffn_scale"),
                        &[3],
                        Dtype::F32,
                    )?,
                },
            ];
            for name in [
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            ] {
                mark(&mut consumed, format!("{prefix}.{name}"))?;
            }
            let norms = [
                loader.load(&[(format!("{prefix}.input_layernorm.weight"), vec![hidden])])?,
                loader.load(&[(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![hidden],
                )])?,
            ];
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                mark(&mut consumed, format!("{prefix}.{name}"))?;
            }
            let ffn = if layer < dense_layers {
                let intermediate = spec.intermediate as usize;
                GlmGateFfnWeights::Dense(expert_codes([
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.gate_proj"),
                        intermediate,
                        hidden,
                    )?,
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.up_proj"),
                        intermediate,
                        hidden,
                    )?,
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.down_proj"),
                        hidden,
                        intermediate,
                    )?,
                ]))
            } else {
                let moe = spec.moe.as_ref().context("GLM MoE geometry missing")?;
                let expert_count = moe.experts as usize;
                let intermediate = moe.expert_intermediate as usize;
                let router = loader.load(&[(
                    format!("{prefix}.mlp.gate.weight"),
                    vec![expert_count, hidden],
                )])?;
                mark(&mut consumed, format!("{prefix}.mlp.gate.weight"))?;
                let bias = loader.load_bytes(
                    &format!("{prefix}.mlp.gate.e_score_correction_bias"),
                    &[expert_count],
                    Dtype::F32,
                )?;
                mark(
                    &mut consumed,
                    format!("{prefix}.mlp.gate.e_score_correction_bias"),
                )?;
                let shared_intermediate = moe.shared_intermediate as usize;
                let shared = [
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.shared_experts.gate_proj"),
                        shared_intermediate,
                        hidden,
                    )?,
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.shared_experts.up_proj"),
                        shared_intermediate,
                        hidden,
                    )?,
                    load_projection(
                        loader,
                        &mut consumed,
                        &format!("{prefix}.mlp.shared_experts.down_proj"),
                        hidden,
                        shared_intermediate,
                    )?,
                ];
                let mut experts = Vec::with_capacity(expert_count);
                for expert in 0..expert_count {
                    experts.push([
                        load_projection(
                            loader,
                            &mut consumed,
                            &format!("{prefix}.mlp.experts.{expert}.gate_proj"),
                            intermediate,
                            hidden,
                        )?,
                        load_projection(
                            loader,
                            &mut consumed,
                            &format!("{prefix}.mlp.experts.{expert}.up_proj"),
                            intermediate,
                            hidden,
                        )?,
                        load_projection(
                            loader,
                            &mut consumed,
                            &format!("{prefix}.mlp.experts.{expert}.down_proj"),
                            hidden,
                            intermediate,
                        )?,
                    ]);
                }
                GlmGateFfnWeights::Moe {
                    router,
                    bias,
                    experts: experts.into_iter().map(expert_codes).collect::<Vec<_>>(),
                    shared: expert_codes(shared),
                }
            };
            for (name, entry) in &manifest.entries {
                if entry.load && !consumed.contains(name) {
                    loader.defer(name)?;
                }
            }
            Ok(GlmGateWeights { hc, norms, ffn })
        },
    )?
    .0;
    Ok(weights)
}

fn pointer_set(stream: &Arc<CudaStream>, tensor: &CudaSlice<u8>) -> u64 {
    tensor.device_ptr(stream).0
}

unsafe fn f32_view_u8(
    bytes: &CudaSlice<u8>,
    s: &Arc<CudaStream>,
    shape: [u32; 2],
) -> Result<crate::backend::cuda::contracts::Tensor<f32, 2>> {
    let (stride, count) = crate::backend::cuda::contracts::dense_layout(shape)?;
    let span = count * std::mem::size_of::<f32>() as u64;
    ensure!(
        bytes.len() == span as usize,
        "F32 byte allocation extent mismatch"
    );
    let (ptr, _) = bytes.device_ptr(s);
    // SAFETY: bytes is an owned allocation of exactly `count` F32 values on this context.
    // SAFETY: bytes is an owned allocation of exactly `count` F32 values on this context.
    unsafe { crate::backend::cuda::contracts::Tensor::new(ptr, shape, stride, span) }
}

struct DenseFfn<'a> {
    stream: &'a Arc<CudaStream>,
    fp8: &'a Fp8Ops,
    common: &'a crate::backend::cuda::ops::CommonOpsRef,
    weights: &'a ExpertCodes,
    hidden: usize,
    intermediate: usize,
    layout: crate::fp8::Fp8ScaleLayout,
}

impl DenseFfn<'_> {
    fn projection(&self, index: usize) -> Result<LinearWeightRef> {
        let (output, input) = if index == 2 {
            (self.hidden, self.intermediate)
        } else {
            (self.intermediate, self.hidden)
        };
        Ok(LinearWeightRef::Fp8 {
            codes: view(
                &self.weights.codes[index],
                self.stream,
                0,
                [output as u32, input as u32],
                [input as u64, 1],
            )?,
            scale: crate::backend::cuda::linear::Fp8ScaleRef::F32(unsafe {
                f32_view_u8(
                    &self.weights.scales[index],
                    self.stream,
                    [output.div_ceil(128) as u32, input.div_ceil(128) as u32],
                )?
            }),
            layout: self.layout,
        })
    }

    fn run_chunk(
        &self,
        chunk_rows: usize,
        x: crate::backend::cuda::contracts::Tensor<bf16, 2>,
        output_offset: usize,
        output: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let gate = NativeProjection::new(
            self.fp8,
            self.stream,
            self.projection(0)?,
            chunk_rows as u32,
        )?;
        let up = NativeProjection::new(
            self.fp8,
            self.stream,
            self.projection(1)?,
            chunk_rows as u32,
        )?;
        let down = NativeProjection::new(
            self.fp8,
            self.stream,
            self.projection(2)?,
            chunk_rows as u32,
        )?;
        let gate_out = self.stream.alloc_zeros(chunk_rows * self.intermediate)?;
        let up_out = self.stream.alloc_zeros(chunk_rows * self.intermediate)?;
        let hidden = self.stream.alloc_zeros(chunk_rows * self.intermediate)?;
        let meta = self.stream.clone_htod(&[
            0,
            chunk_rows as i32,
            chunk_rows as i32,
            chunk_rows as i32,
            1,
            0,
            0,
            0,
        ])?;
        let rows = crate::backend::cuda::contracts::RowDomain {
            meta: crate::backend::cuda::contracts::SeqRef {
                ptr: meta.device_ptr(self.stream).0,
            },
            row_base: 0,
            row_capacity: chunk_rows as u32,
        };
        let matrix = |buffer, offset, width| {
            view(
                buffer,
                self.stream,
                offset,
                [chunk_rows as u32, width as u32],
                [width as u64, 1],
            )
        };
        crate::backend::cuda::moe::w8a8::MoeSharedW8A8::run_block(
            self.stream,
            crate::backend::cuda::moe::w8a8::W8A8BlockCall {
                projections: &[gate, up, down],
                fp8: self.fp8,
                common: self.common,
                rows,
                x,
                gate: matrix(&gate_out, 0, self.intermediate)?,
                up: matrix(&up_out, 0, self.intermediate)?,
                activation: matrix(&hidden, 0, self.intermediate)?,
                out: matrix(output, output_offset, self.hidden)?,
            },
        )
    }
}

fn fixture_path(dir: &std::path::Path, layer: usize) -> PathBuf {
    dir.join(format!("layer{layer}-activations.safetensors"))
}

#[test]
fn recorded_pre_boundary_rejects_negative_beyond_band() -> Result<()> {
    let thresholds = FrozenThresholds {
        bands: BTreeMap::from([("layer0.attn_hc.2".to_owned(), 0.5)]),
    };
    let stream = [bf16::from_f32(1.0)];
    let pre = [1.0];
    let expected = [bf16::from_f32(2.0)];
    let (result, exempt, defects, proofs) = recorded_pre_boundary(
        &thresholds,
        0,
        "attn_hc.2",
        (&stream, &pre, &expected),
        CollapseExtent {
            rows: 1,
            copies: 1,
            hidden: 1,
        },
    )?;
    assert_eq!(exempt, 0);
    assert_eq!(defects, 1);
    assert_eq!(proofs.len(), 1);
    assert!(result.defect);
    Ok(())
}

#[test]
fn recorded_pre_boundary_exempts_cancellation_interval() -> Result<()> {
    let thresholds = FrozenThresholds {
        bands: BTreeMap::from([("layer0.ffn_hc.2".to_owned(), 0.0001)]),
    };
    let stream = [bf16::from_f32(1.0); 4];
    let pre = [1000.0, -1000.0, 1000.0, -1000.0];
    let expected = [bf16::from_f32(0.0005)];
    let (result, exempt, defects, proofs) = recorded_pre_boundary(
        &thresholds,
        0,
        "ffn_hc.2",
        (&stream, &pre, &expected),
        CollapseExtent {
            rows: 1,
            copies: 4,
            hidden: 1,
        },
    )?;
    assert_eq!(exempt, 1);
    assert_eq!(defects, 0);
    assert_eq!(proofs.len(), 1);
    assert!(!result.defect);
    assert!(proofs[0].contains("interval="));
    Ok(())
}

#[test]
fn coefficient_boundary_exempts_projection_interval() -> Result<()> {
    let params = crate::backend::cuda::mhc::HcParams {
        mult: 2,
        hidden: 1,
        iters: 1,
        eps: 1e-6,
        norm_eps: 1e-6,
    };
    let weights = HcHostWeights {
        fn_w: (0..8)
            .flat_map(|row| {
                [if (2..4).contains(&row) {
                    bf16::ONE
                } else {
                    bf16::ZERO
                }; 2]
            })
            .collect(),
        base: vec![0.0; 8],
        scale: vec![1.0, 1.0, 1.0],
    };
    let stream = [bf16::ONE, bf16::ONE];
    let exact = CoefficientRow::new(&stream, &weights, &params)?;
    let candidate = exact.post_exact.clone();
    let publisher = vec![exact.post_exact[0] + 1e-8, exact.post_exact[1]];
    let thresholds = FrozenThresholds {
        bands: BTreeMap::from([("layer0.attn_hc.0".to_owned(), 0.0)]),
    };
    let (result, exempt, defects, proofs) = coefficient_boundary(CoefficientBoundary {
        thresholds: &thresholds,
        layer: 0,
        boundary: "attn_hc.0",
        candidate: &candidate,
        publisher: &publisher,
        stream: &stream,
        weights: &weights,
        params,
    })?;
    assert_eq!(exempt, 1);
    assert_eq!(defects, 0);
    assert_eq!(proofs.len(), 1);
    assert!(!result.defect);
    assert!(proofs[0].contains("exact="));
    assert!(proofs[0].contains("interval="));
    Ok(())
}

#[test]
fn coefficient_row_comb_interval_contains_exact() -> Result<()> {
    let params = crate::backend::cuda::mhc::HcParams {
        mult: 4,
        hidden: 1,
        iters: 20,
        eps: 1e-6,
        norm_eps: 1e-5,
    };
    let stream = [bf16::ONE, -bf16::ONE, bf16::ONE, -bf16::ONE];
    let mut value = 0x12345678u32;
    let mut next = || {
        value = value.wrapping_mul(1664525).wrapping_add(1013904223);
        bf16::from_f32(((value >> 24) % 33) as f32 / 32.0 - 0.5)
    };
    let weights = HcHostWeights {
        fn_w: (0..24 * 4).map(|_| next()).collect(),
        base: (0..24).map(|_| next().to_f32()).collect(),
        scale: vec![1.0, 1.0, 1.0],
    };
    let row = CoefficientRow::new(&stream, &weights, &params)?;
    for (index, (&exact, interval)) in row.comb_exact.iter().zip(&row.comb_interval).enumerate() {
        assert!(
            interval.lower.is_finite()
                && interval.upper.is_finite()
                && interval.lower > 0.0
                && interval.upper < f64::INFINITY
                && interval.lower <= exact
                && exact <= interval.upper,
            "comb index {index}: exact={exact:e}, interval={interval:?}"
        );
    }
    Ok(())
}

#[test]
fn coefficient_boundary_rejects_out_of_interval_comb_control() -> Result<()> {
    let params = crate::backend::cuda::mhc::HcParams {
        mult: 4,
        hidden: 1,
        iters: 20,
        eps: 1e-6,
        norm_eps: 1e-5,
    };
    let stream = [bf16::ONE, -bf16::ONE, bf16::ONE, -bf16::ONE];
    let mut value = 0x12345678u32;
    let mut next = || {
        value = value.wrapping_mul(1664525).wrapping_add(1013904223);
        bf16::from_f32(((value >> 24) % 33) as f32 / 32.0 - 0.5)
    };
    let weights = HcHostWeights {
        fn_w: (0..24 * 4).map(|_| next()).collect(),
        base: (0..24).map(|_| next().to_f32()).collect(),
        scale: vec![1.0, 1.0, 1.0],
    };
    let exact_row = CoefficientRow::new(&stream, &weights, &params)?;
    let mut candidate = exact_row.comb_exact.clone();
    candidate[0] = exact_row.comb_interval[0].upper + 1.0;
    let publisher = exact_row.comb_exact.clone();
    let thresholds = FrozenThresholds {
        bands: BTreeMap::from([("layer0.attn_hc.1".to_owned(), 0.0)]),
    };
    let (result, exempt, defects, proofs) = coefficient_boundary(CoefficientBoundary {
        thresholds: &thresholds,
        layer: 0,
        boundary: "attn_hc.1",
        candidate: &candidate,
        publisher: &publisher,
        stream: &stream,
        weights: &weights,
        params,
    })?;
    assert_eq!(exempt, 0);
    assert_eq!(defects, 1);
    assert_eq!(proofs.len(), 1);
    assert!(result.defect);
    assert!(proofs[0].contains("method=jacobian_taylor_remainder_fp32_rounding"));
    assert!(proofs[0].contains("exempt=false"));
    Ok(())
}

#[test]
#[ignore = "requires a coordinator-granted GPU window; FF_GLM53_GPU_WINDOW=granted, FF_GLM53_MODEL, FF_GLM53_LAYER_FIXTURE and FF_GLM53_THRESHOLDS"]
fn glm_all_layers_match_model_definition_boundaries() -> Result<()> {
    use crate::backend::cuda::contracts::{RowDomain, SeqRef, view};
    use crate::backend::cuda::hc::HcRuntime;
    use crate::backend::cuda::hc::glm_binding;
    use crate::backend::cuda::moe::decode::{NativeMoe, NativeMoeGeometry, NativeMoeWeights};
    use crate::backend::cuda::ops::{CommonOpsRef, RmsMode, RmsOptions};
    use crate::model::HcSublayer;

    ensure!(
        std::env::var("FF_GLM53_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let model_dir = env(GateEnv::Model)?;
    let fixture_dir = env(GateEnv::FixtureDir)?;
    let thresholds_path = env(GateEnv::Thresholds)?;
    let thresholds = FrozenThresholds::read(&thresholds_path)?;
    let spec = crate::model_spec::ModelSpec::read(&model_dir)?;
    ensure!(
        spec.architecture == crate::model_spec::Architecture::Glm5Next,
        "gate model is not GLM-5.3"
    );
    let glm = spec
        .glm5_next
        .as_ref()
        .context("GLM configuration missing")?;
    let layer_count = spec.layers.len();
    let device = Device::new(0)?;
    let s = &device.stream;
    let rows = 85usize;
    let hidden = spec.hidden as usize;
    let mult = glm.hc_mult as usize;
    let fixture_bytes = (0..layer_count)
        .map(|layer| {
            std::fs::read(fixture_path(&fixture_dir, layer))
                .with_context(|| format!("read layer{layer} fixture"))
        })
        .collect::<Result<Vec<_>>>()?;
    let common = CommonOpsRef::new_with_swiglu_rounding(
        &device.ctx,
        spec.swiglu_rounding(),
        spec.swiglu_limit(),
    )?;
    let fp8 = Fp8Ops::new(&device.ctx)?;
    let mapped = crate::checkpoint::MappedCheckpoint::open(&model_dir)?;
    let mut boundary_reports = Vec::new();
    for (layer, data) in fixture_bytes.iter().enumerate() {
        let st = fixture_tensors(data)?;
        let tensor_name = |name: &str| format!("layer{layer}.{name}");
        let input = fixture_bf16(&st, &tensor_name("input"))?;
        ensure!(
            input.len() == rows * mult * hidden,
            "layer{layer} fixture input extent"
        );
        let weights = load_layer_weights(&mapped, &device, &spec, layer)?;
        let GlmGateWeights {
            hc: hc_weights,
            norms,
            ffn,
        } = weights;
        let host_weights = [
            HcHostWeights::read(s, &hc_weights[0])?,
            HcHostWeights::read(s, &hc_weights[1])?,
        ];
        let mut hc = HcRuntime::new(&device.ctx, s, glm_binding(&spec)?, 1, rows)?;
        let hc_params = hc.binding.params;
        let [attn_hc, ffn_hc] = hc_weights;
        hc.bind_layer(0, attn_hc, ffn_hc)?;
        s.memcpy_htod(&input, &mut hc.stream)?;
        let compare_mix = |runtime: &HcRuntime,
                           boundary: &str,
                           expected_pre: &[f64],
                           expected_comb: &[f64],
                           coefficient_stream: &[bf16],
                           coefficient_weights: &HcHostWeights|
         -> Result<Vec<(String, BoundaryCheck)>> {
            let comb = s.clone_dtoh(&runtime.mix_comb)?;
            let boundaries: Vec<(String, Vec<f64>, Vec<f64>)> = vec![
                (
                    format!("{boundary}.0"),
                    s.clone_dtoh(&runtime.mix_post)?
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect(),
                    expected_pre.to_vec(),
                ),
                (
                    format!("{boundary}.1"),
                    comb.iter().map(|&v| f64::from(v)).collect(),
                    expected_comb.to_vec(),
                ),
            ];
            let mut results = Vec::new();
            for (name, actual, expected) in boundaries {
                let mut result = assert_boundary(&thresholds, layer, &name, &actual, &expected)?;
                if result.defect {
                    let (rule_b, exempt, defects, proofs) =
                        coefficient_boundary(CoefficientBoundary {
                            thresholds: &thresholds,
                            layer,
                            boundary: &name,
                            candidate: &actual,
                            publisher: &expected,
                            stream: coefficient_stream,
                            weights: coefficient_weights,
                            params: hc_params,
                        })?;
                    println!(
                        "GLM_COEFFICIENT_RULE_B layer={layer} boundary={name} beyond={} exempt={exempt} defect={defects}",
                        proofs.len()
                    );
                    for proof in proofs {
                        println!(
                            "GLM_COEFFICIENT_RULE_B_PROOF layer={layer} boundary={name} {proof}"
                        );
                    }
                    result = rule_b;
                }
                results.push((name, result));
            }
            Ok(results)
        };
        let mut branch = s.alloc_zeros(rows * hidden)?;
        hc.hc_pre(
            s,
            0,
            HcSublayer::Attention,
            crate::model::HcPreSource::Own,
            rows,
            &mut branch,
        )?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=attn_hc_pre done");
        let attn_results = compare_mix(
            &hc,
            "attn_hc",
            &fixture_f32(&st, &tensor_name("attn_hc.0"))?,
            &fixture_f32(&st, &tensor_name("attn_hc.1"))?,
            &input,
            &host_weights[0],
        )?;
        boundary_reports.extend(
            attn_results
                .into_iter()
                .map(|(name, result)| (layer, name, result)),
        );
        let expected_collapsed = fixture_bf16(&st, &tensor_name("attn_hc.2"))?;
        let (isolated_result, exempt, defects, proofs) = recorded_pre_boundary(
            &thresholds,
            layer,
            "attn_hc.2",
            (
                &input,
                &fixture_f32(&st, &tensor_name("attn_hc.pre"))?,
                &expected_collapsed,
            ),
            CollapseExtent {
                rows,
                copies: mult,
                hidden,
            },
        )?;
        boundary_reports.push((layer, "attn_hc.2".to_owned(), isolated_result));
        println!(
            "GLM_RECORDED_PRE_RULE_B layer={layer} boundary=attn_hc.2 beyond={} exempt={exempt} defect={defects}",
            proofs.len()
        );
        for proof in proofs {
            println!("GLM_RECORDED_PRE_PROOF layer={layer} boundary=attn_hc.2 {proof}");
        }
        let attention = fixture_bf16(&st, &tensor_name("self_attn.0"))?;
        s.memcpy_htod(&attention, &mut branch)?;
        let recorded_post: Vec<f32> = fixture_f32(&st, &tensor_name("attn_hc.0"))?
            .iter()
            .map(|&value| value as f32)
            .collect();
        let recorded_comb: Vec<f32> = fixture_f32(&st, &tensor_name("attn_hc.1"))?
            .iter()
            .map(|&value| value as f32)
            .collect();
        s.memcpy_htod(&recorded_post, &mut hc.mix_post)?;
        s.memcpy_htod(&recorded_comb, &mut hc.mix_comb)?;
        hc.hc_post(s, 0, HcSublayer::Attention, rows, &branch)?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=attn_hc_post done");
        let attention_stream = s.clone_dtoh(&hc.stream)?;
        let mut ffn_input = s.alloc_zeros(rows * hidden)?;
        hc.hc_pre(
            s,
            0,
            HcSublayer::Ffn,
            crate::model::HcPreSource::Own,
            rows,
            &mut ffn_input,
        )?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=ffn_hc_pre done");
        let ffn_results = compare_mix(
            &hc,
            "ffn_hc",
            &fixture_f32(&st, &tensor_name("ffn_hc.0"))?,
            &fixture_f32(&st, &tensor_name("ffn_hc.1"))?,
            &attention_stream,
            &host_weights[1],
        )?;
        boundary_reports.extend(
            ffn_results
                .into_iter()
                .map(|(name, result)| (layer, name, result)),
        );
        let expected_ffn_collapsed = fixture_bf16(&st, &tensor_name("ffn_hc.2"))?;
        let (isolated_result, exempt, defects, proofs) = recorded_pre_boundary(
            &thresholds,
            layer,
            "ffn_hc.2",
            (
                &attention_stream,
                &fixture_f32(&st, &tensor_name("ffn_hc.pre"))?,
                &expected_ffn_collapsed,
            ),
            CollapseExtent {
                rows,
                copies: mult,
                hidden,
            },
        )?;
        boundary_reports.push((layer, "ffn_hc.2".to_owned(), isolated_result));
        println!(
            "GLM_RECORDED_PRE_RULE_B layer={layer} boundary=ffn_hc.2 beyond={} exempt={exempt} defect={defects}",
            proofs.len()
        );
        for proof in proofs {
            println!("GLM_RECORDED_PRE_PROOF layer={layer} boundary=ffn_hc.2 {proof}");
        }
        let meta = s.clone_htod(&[0, rows as i32, rows as i32, rows as i32, 1, 0, 0, 0])?;
        let domain = RowDomain {
            meta: SeqRef {
                ptr: meta.device_ptr(s).0,
            },
            row_base: 0,
            row_capacity: rows as u32,
        };
        let norm = |source: &CudaSlice<bf16>,
                    weight: &CudaSlice<bf16>,
                    output: &mut CudaSlice<bf16>|
         -> Result<()> {
            common.norm.rms_with(
                s,
                domain,
                view(
                    source,
                    s,
                    0,
                    [rows as u32, 1, hidden as u32],
                    [hidden as u64, hidden as u64, 1],
                )?,
                view(weight, s, 0, [hidden as u32], [1])?,
                &mut crate::backend::cuda::contracts::TensorMut(view(
                    output,
                    s,
                    0,
                    [rows as u32, 1, hidden as u32],
                    [hidden as u64, hidden as u64, 1],
                )?),
                RmsOptions {
                    mode: RmsMode::Standard,
                    eps: spec.rms_eps,
                },
            )
        };
        let mut normed_input = s.alloc_zeros(rows * hidden)?;
        let attn_hc_reference = s.clone_htod(&fixture_bf16(&st, &tensor_name("attn_hc.2"))?)?;
        norm(&attn_hc_reference, &norms[0], &mut normed_input)?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=input_norm done");
        let actual_input_norm = s.clone_dtoh(&normed_input)?;
        boundary_reports.push((
            layer,
            "input_layernorm.0".to_owned(),
            assert_boundary(
                &thresholds,
                layer,
                "input_layernorm.0",
                &bf16_values(&actual_input_norm),
                &bf16_values(&fixture_bf16(&st, &tensor_name("input_layernorm.0"))?),
            )?,
        ));
        let mut normed_ffn = s.alloc_zeros(rows * hidden)?;
        let ffn_hc_reference = s.clone_htod(&fixture_bf16(&st, &tensor_name("ffn_hc.2"))?)?;
        norm(&ffn_hc_reference, &norms[1], &mut normed_ffn)?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=post_norm done");
        let actual_post_norm = s.clone_dtoh(&normed_ffn)?;
        boundary_reports.push((
            layer,
            "post_attention_layernorm.0".to_owned(),
            assert_boundary(
                &thresholds,
                layer,
                "post_attention_layernorm.0",
                &bf16_values(&actual_post_norm),
                &bf16_values(&fixture_bf16(
                    &st,
                    &tensor_name("post_attention_layernorm.0"),
                )?),
            )?,
        ));
        let mut moe_output = s.alloc_zeros(rows * hidden)?;
        let post_norm_reference = s.clone_htod(&fixture_bf16(
            &st,
            &tensor_name("post_attention_layernorm.0"),
        )?)?;
        let mut moe_offset = 0usize;
        match &ffn {
            GlmGateFfnWeights::Dense(projections) => {
                let dense = DenseFfn {
                    stream: s,
                    fp8: &fp8,
                    common: &common,
                    weights: projections,
                    hidden,
                    intermediate: spec.intermediate as usize,
                    layout: spec
                        .linear_format
                        .fp8_layout()
                        .context("GLM native FP8 layout missing")?,
                };
                for chunk_rows in [16usize, 16, 16, 16, 16, 5] {
                    let x = view(
                        &post_norm_reference,
                        s,
                        moe_offset,
                        [chunk_rows as u32, hidden as u32],
                        [hidden as u64, 1],
                    )?;
                    dense.run_chunk(chunk_rows, x, moe_offset, &mut moe_output)?;
                    moe_offset += chunk_rows * hidden;
                    s.synchronize()?;
                    println!(
                        "GLM_STAGE layer={layer} stage=dense_ffn_chunk rows={chunk_rows} done"
                    );
                }
            }
            GlmGateFfnWeights::Moe {
                router,
                bias,
                experts,
                shared,
            } => {
                let moe = spec.moe.as_ref().context("GLM MoE geometry missing")?;
                let expert_count = moe.experts as usize;
                let run_moe_chunk = |chunk_rows: usize,
                                     x: crate::backend::cuda::contracts::Tensor<bf16, 2>,
                                     output_offset: usize,
                                     output: &mut CudaSlice<bf16>|
                 -> Result<()> {
                    let mut slots = vec![0u64; expert_count * 6];
                    let valid = s.clone_htod(&vec![1u32; expert_count])?;
                    for (id, expert) in experts.iter().enumerate() {
                        for projection in 0..3 {
                            slots[id * 6 + projection * 2] =
                                pointer_set(s, &expert.codes[projection]);
                            slots[id * 6 + projection * 2 + 1] =
                                pointer_set(s, &expert.scales[projection]);
                        }
                    }
                    let slots_gpu = s.clone_htod(&slots)?;
                    let bias_bytes = s.clone_dtoh(bias)?;
                    let bias_values: Vec<f32> = bias_bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|value| f32::from_le_bytes(*value))
                        .collect();
                    let mut bias_tiled = Vec::<f32>::with_capacity(chunk_rows * expert_count);
                    for _ in 0..chunk_rows {
                        bias_tiled.extend_from_slice(&bias_values);
                    }
                    let bias_gpu: CudaSlice<f32> = s.clone_htod(&bias_tiled)?;
                    let intermediate = moe.shared_intermediate as usize;
                    let shared_codes = [
                        view(
                            &shared.codes[0],
                            s,
                            0,
                            [intermediate as u32, hidden as u32],
                            [hidden as u64, 1],
                        )?,
                        view(
                            &shared.codes[1],
                            s,
                            0,
                            [intermediate as u32, hidden as u32],
                            [hidden as u64, 1],
                        )?,
                        view(
                            &shared.codes[2],
                            s,
                            0,
                            [hidden as u32, intermediate as u32],
                            [intermediate as u64, 1],
                        )?,
                    ];
                    let shared_scales = [
                        unsafe {
                            f32_view_u8(
                                &shared.scales[0],
                                s,
                                [(intermediate / 128) as u32, (hidden / 128) as u32],
                            )?
                        },
                        unsafe {
                            f32_view_u8(
                                &shared.scales[1],
                                s,
                                [(intermediate / 128) as u32, (hidden / 128) as u32],
                            )?
                        },
                        unsafe {
                            f32_view_u8(
                                &shared.scales[2],
                                s,
                                [(hidden / 128) as u32, (intermediate / 128) as u32],
                            )?
                        },
                    ];
                    let ffn = unsafe {
                        NativeMoe::new(
                            &device.ctx,
                            s,
                            NativeMoeGeometry {
                                hidden: spec.hidden,
                                intermediate: moe.expert_intermediate,
                                experts: moe.experts,
                                top_k: moe.experts_per_tok,
                                scaling: spec
                                    .glm5_next
                                    .as_ref()
                                    .context("GLM configuration missing")?
                                    .routed_scaling_factor,
                                limit: spec
                                    .glm5_next
                                    .as_ref()
                                    .context("GLM configuration missing")?
                                    .swiglu_limit,
                            },
                            NativeMoeWeights {
                                router: view(
                                    router,
                                    s,
                                    0,
                                    [expert_count as u32, hidden as u32],
                                    [hidden as u64, 1],
                                )?,
                                bias: crate::backend::cuda::contracts::Tensor::new(
                                    bias_gpu.device_ptr(s).0,
                                    [chunk_rows as u32, expert_count as u32],
                                    [expert_count as u64, 1],
                                    (chunk_rows * expert_count * std::mem::size_of::<f32>()) as u64,
                                )?,
                                slots: view(&slots_gpu, s, 0, [expert_count as u32, 6], [6, 1])?,
                                valid: view(&valid, s, 0, [expert_count as u32], [1])?,
                                shared: std::array::from_fn(|index| LinearWeightRef::Fp8 {
                                    codes: shared_codes[index],
                                    scale: crate::backend::cuda::linear::Fp8ScaleRef::F32(
                                        shared_scales[index],
                                    ),
                                    layout: spec
                                        .linear_format
                                        .fp8_layout()
                                        .expect("checked native FP8 layout"),
                                }),
                            },
                            chunk_rows as u32,
                        )?
                    };
                    ffn.route(s, x)?;
                    ffn.run(
                        s,
                        x,
                        &crate::backend::cuda::contracts::TensorMut(view(
                            output,
                            s,
                            output_offset,
                            [chunk_rows as u32, hidden as u32],
                            [hidden as u64, 1],
                        )?),
                    )
                };
                for chunk_rows in [16usize, 16, 16, 16, 16, 5] {
                    let x = view(
                        &post_norm_reference,
                        s,
                        moe_offset,
                        [chunk_rows as u32, hidden as u32],
                        [hidden as u64, 1],
                    )?;
                    run_moe_chunk(chunk_rows, x, moe_offset, &mut moe_output)?;
                    moe_offset += chunk_rows * hidden;
                    s.synchronize()?;
                    println!("GLM_STAGE layer={layer} stage=moe_chunk rows={chunk_rows} done");
                }
            }
        }
        ensure!(moe_offset == rows * hidden, "FFN chunk coverage mismatch");
        let actual_ffn = s.clone_dtoh(&moe_output)?;
        boundary_reports.push((
            layer,
            "mlp.0".to_owned(),
            assert_boundary(
                &thresholds,
                layer,
                "mlp.0",
                &bf16_values(&actual_ffn),
                &bf16_values(&fixture_bf16(&st, &tensor_name("mlp.0"))?),
            )?,
        ));
        let ffn_reference = s.clone_htod(&fixture_bf16(&st, &tensor_name("mlp.0"))?)?;
        hc.hc_post(s, 0, HcSublayer::Ffn, rows, &ffn_reference)?;
        s.synchronize()?;
        println!("GLM_STAGE layer={layer} stage=ffn_hc_post done");
        let actual_output = s.clone_dtoh(&hc.stream)?;
        boundary_reports.push((
            layer,
            "output".to_owned(),
            assert_boundary(
                &thresholds,
                layer,
                "output",
                &bf16_values(&actual_output),
                &bf16_values(&fixture_bf16(&st, &tensor_name("output"))?),
            )?,
        ));
        drop(hc);
        s.synchronize()?;
    }
    let required = [
        "output",
        "attn_hc.0",
        "attn_hc.1",
        "attn_hc.2",
        "input_layernorm.0",
        "ffn_hc.0",
        "ffn_hc.1",
        "ffn_hc.2",
        "post_attention_layernorm.0",
        "mlp.0",
    ];
    for layer in 0..layer_count {
        for boundary in required {
            ensure!(
                boundary_reports
                    .iter()
                    .any(|(report_layer, name, _)| *report_layer == layer && name == boundary),
                "layer{layer} boundary {boundary} did not execute"
            );
        }
    }
    let executed: Vec<String> = boundary_reports
        .iter()
        .map(|(layer, name, result)| {
            format!(
                "layer{layer}.{name}:max={:e}:band={:e}:near_tie={:.3e}:defect={}",
                result.max_abs, result.band, result.near_tie_fraction, result.defect
            )
        })
        .collect();
    let defects = boundary_reports
        .iter()
        .filter(|(_, _, result)| result.defect)
        .count();
    println!(
        "RESULT GLM_ALL_LAYER_BOUNDARIES layers=[0,{}] boundaries={} defects={defects} rule_a=frozen_bands_full_operand rule_b=fp32_reduction_interval_per_element results={}",
        layer_count - 1,
        executed.len(),
        serde_json::json!(executed)
    );
    ensure!(
        defects == 0,
        "GLM all-layer candidate DEFECT count={defects}"
    );
    Ok(())
}
