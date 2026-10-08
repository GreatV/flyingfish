use crate::{model::Model, trace::Trace};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

const STEPS: usize = 64;
const COSINE_MIN: f64 = 0.999;
const LOGITS_MAX: f64 = 1.5;
const LOGITS_MEAN: f64 = 0.1;
const JUMP_MAX: f64 = 3.0;
const OUTLIER_RATE: f64 = 0.005;
const OUTLIER_ALPHA: f64 = 0.01;

pub struct Rules {
    hidden: Vec<LayerLimit>,
    source: String,
    logits_mean: Vec<f64>,
    logits_max: Vec<f64>,
    logits_source: String,
    systematic: Option<SystemRule>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Gates {
    ratio_median_max: f64,
    ratio_p90_max: f64,
    outlier_binomial: Binomial,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Binomial {
    p: f64,
    alpha: f64,
}

struct SystemRule {
    gates: Gates,
    mean: Vec<f64>,
    max: Vec<f64>,
}

fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = q * (sorted.len() - 1) as f64;
    let lo = sorted[index.floor() as usize];
    let hi = sorted[index.ceil() as usize];
    if lo == hi {
        lo
    } else {
        lo + (hi - lo) * (index - index.floor())
    }
}

fn logits_steps(value: &serde_json::Value) -> Result<usize> {
    Ok(value
        .as_array()
        .context("logit limits must be arrays")?
        .len())
}

fn outlier_p_value(steps: usize, count: usize, rate: f64) -> Result<f64> {
    ensure!(
        steps > 0 && steps <= i32::MAX as usize && count <= steps,
        "invalid binomial step/count range"
    );
    if count == 0 {
        return Ok(1.0);
    }
    ensure!(
        rate.is_finite() && rate > 0.0 && rate < 1.0,
        "invalid binomial expected rate"
    );
    let q = 1.0 - rate;
    let mut probability = q.powi(steps as i32);
    let mut cumulative = probability;
    for k in 1..count {
        probability *= (steps - k + 1) as f64 / k as f64 * rate / q;
        cumulative += probability;
    }
    Ok((1.0 - cumulative).clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct LayerLimit {
    rel: f64,
    cos_min: f64,
}

impl LayerLimit {
    fn accepts(&self, m: &Metrics) -> bool {
        m.rel_rmse <= self.rel && m.cosine >= self.cos_min
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HiddenLimit {
    Legacy(f64),
    Current(LayerLimit),
}

impl HiddenLimit {
    fn value(&self) -> LayerLimit {
        match self {
            Self::Legacy(rel) => LayerLimit {
                rel: *rel,
                cos_min: COSINE_MIN,
            },
            Self::Current(v) => *v,
        }
    }
}

impl Rules {
    pub fn check_step(
        &self,
        actual: &[f32],
        reference: &GreedyReference,
        step: usize,
        token: u32,
    ) -> Result<Metrics> {
        ensure!(
            step < reference.tokens.len() && step < self.logits_mean.len(),
            "logit step outside reference/rules"
        );
        let vocab = reference.logits.len() / reference.tokens.len();
        let m = metrics(actual, &reference.logits[step * vocab..(step + 1) * vocab])?;
        ensure!(
            top_two(actual)?.0 == token,
            "verification argmax disagrees with logits at step {step}"
        );
        let exact = token == reference.tokens[step];
        let near = !exact && (reference.margins[step] as f64) < self.logits_max[step];
        let warn = m.mean_abs > self.logits_mean[step] || m.max_abs > self.logits_max[step];
        println!(
            "{}",
            serde_json::json!({"step":step,"logits":m,"actual_token":token,
            "expected_token":reference.tokens[step],"reference_margin":reference.margins[step],
            "mean_abs_limit":self.logits_mean[step],"max_abs_limit":self.logits_max[step],
            "exact_match":exact,"near_tie_exception":near,"warn":warn,"pass":exact||near})
        );
        ensure!(
            exact || near,
            "verification token differs outside near-tie at step {step}"
        );
        Ok(m)
    }
    pub fn read(
        path: Option<&Path>,
        logit_path: Option<&Path>,
        fixture: &Path,
        layers: usize,
    ) -> Result<Self> {
        let (hidden, source) = if let Some(path) = path {
            let table: BTreeMap<String, BTreeMap<usize, HiddenLimit>> =
                serde_json::from_slice(&std::fs::read(path)?)?;
            let stem = fixture
                .file_stem()
                .and_then(|s| s.to_str())
                .context("invalid fixture stem")?;
            let values = table
                .get(stem)
                .with_context(|| format!("hidden limit table missing {stem}"))?;
            ensure!(
                values.keys().copied().eq(0..layers + 1),
                "hidden limit table {stem} must contain each layer 0..={layers}"
            );
            (
                values.values().map(HiddenLimit::value).collect(),
                path.display().to_string(),
            )
        } else {
            (
                vec![
                    LayerLimit {
                        rel: 0.05,
                        cos_min: COSINE_MIN
                    };
                    layers + 1
                ],
                "uniform provisional hidden limits".into(),
            )
        };
        ensure!(
            hidden.len() == layers + 1,
            "hidden limit count must be {}",
            layers + 1
        );
        ensure!(
            hidden.iter().all(|v| v.rel.is_finite()
                && v.rel >= 0.0
                && v.cos_min.is_finite()
                && (-1.0..=1.0).contains(&v.cos_min)),
            "invalid per-layer hidden limits"
        );
        let (logits_mean, logits_max, logits_source, systematic) = if let Some(path) = logit_path {
            let table: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let stem = fixture
                .file_stem()
                .and_then(|s| s.to_str())
                .context("invalid fixture stem")?;
            let item = table
                .get(stem)
                .with_context(|| format!("logit limit table missing {stem}"))?;
            let mean: Vec<f64> = serde_json::from_value(
                item.get("mean_abs")
                    .context("mean_abs limits missing")?
                    .clone(),
            )?;
            let max: Vec<f64> = serde_json::from_value(
                item.get("max_abs")
                    .context("max_abs limits missing")?
                    .clone(),
            )?;
            let systematic = {
                let gates = table
                    .get("gates")
                    .context("systematic logit gates missing")?;
                let gates: Gates = serde_json::from_value(gates.clone())?;
                ensure!(
                    gates.ratio_median_max.is_finite()
                        && gates.ratio_median_max >= 0.0
                        && gates.ratio_p90_max.is_finite()
                        && gates.ratio_p90_max >= 0.0,
                    "invalid systematic logit gates"
                );
                ensure!(
                    gates.outlier_binomial.p.is_finite()
                        && gates.outlier_binomial.p > 0.0
                        && gates.outlier_binomial.p < 1.0
                        && gates.outlier_binomial.alpha.is_finite()
                        && gates.outlier_binomial.alpha > 0.0
                        && gates.outlier_binomial.alpha < 1.0,
                    "invalid outlier binomial p/alpha"
                );
                let mean: Vec<f64> = serde_json::from_value(
                    item.get("probe_mean_abs")
                        .context("probe_mean_abs missing for systematic gates")?
                        .clone(),
                )?;
                let max: Vec<f64> = serde_json::from_value(
                    item.get("probe_max_abs")
                        .context("probe_max_abs missing for systematic gates")?
                        .clone(),
                )?;
                ensure!(
                    mean.len() == logits_steps(&item["mean_abs"])?
                        && max.len() == mean.len()
                        && mean.iter().chain(&max).all(|v| v.is_finite() && *v >= 0.0),
                    "systematic probes must match the prompt step count and be finite/nonnegative"
                );
                Some(SystemRule { gates, mean, max })
            };
            (mean, max, path.display().to_string(), systematic)
        } else {
            (
                vec![LOGITS_MEAN; STEPS],
                vec![LOGITS_MAX; STEPS],
                "uniform provisional logit limits".into(),
                None,
            )
        };
        ensure!(
            !logits_mean.is_empty() && logits_max.len() == logits_mean.len(),
            "logit limit arrays must cover the same nonempty step range"
        );
        ensure!(
            logits_mean.iter().all(|v| v.is_finite() && *v >= 0.0)
                && logits_max.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid logit limits"
        );
        Ok(Self {
            hidden,
            source,
            logits_mean,
            logits_max,
            logits_source,
            systematic,
        })
    }

    pub fn systematic(&self, values: &[Metrics]) -> Result<bool> {
        ensure!(
            values.len() == self.logits_mean.len(),
            "systematic logits check must cover every prompt step"
        );
        let mut passed = true;
        let binomial = self.systematic.as_ref().map_or(
            Binomial {
                p: OUTLIER_RATE,
                alpha: OUTLIER_ALPHA,
            },
            |r| r.gates.outlier_binomial,
        );
        for (kind, engine, limits) in [
            (
                "mean_abs",
                values.iter().map(|m| m.mean_abs).collect::<Vec<_>>(),
                self.logits_mean.clone(),
            ),
            (
                "max_abs",
                values.iter().map(|m| m.max_abs).collect::<Vec<_>>(),
                self.logits_max.clone(),
            ),
        ] {
            let count = engine
                .iter()
                .zip(&limits)
                .filter(|(a, limit)| a > limit)
                .count();
            let probability = outlier_p_value(values.len(), count, binomial.p)?;
            let pass = probability >= binomial.alpha;
            passed &= pass;
            println!(
                "{}",
                serde_json::json!({"logit_outlier_rate":kind,"exceed_count":count,
                "steps":values.len(),"exceed_fraction":count as f64/values.len() as f64,
                "rule":"v3.4 binomial count","expected_rate":binomial.p,
                "alpha":binomial.alpha,"p_value":probability,"pass":pass})
            );
        }
        let Some(rule) = &self.systematic else {
            return Ok(passed);
        };
        for (kind, probe, engine) in [
            (
                "mean_abs",
                &rule.mean,
                values.iter().map(|m| m.mean_abs).collect::<Vec<_>>(),
            ),
            (
                "max_abs",
                &rule.max,
                values.iter().map(|m| m.max_abs).collect::<Vec<_>>(),
            ),
        ] {
            let ratios: Vec<_> = engine
                .iter()
                .zip(probe)
                .map(|(a, p)| if *p > 0.0 { *a / *p } else { f64::INFINITY })
                .collect();
            let median = percentile(&ratios, 0.5);
            let p90 = percentile(&ratios, 0.9);
            let pass = median <= rule.gates.ratio_median_max && p90 <= rule.gates.ratio_p90_max;
            passed &= pass;
            println!(
                "{}",
                serde_json::json!({"systematic_logits":kind,"ratio_median":median,
                "ratio_p90":p90,"limits":rule.gates,"pass":pass})
            );
        }
        Ok(passed)
    }

    fn print(&self) {
        println!(
            "{}",
            serde_json::json!({"rules_source":self.source,"hidden_limits":self.hidden,"logits_source":self.logits_source,"logits_max_abs":self.logits_max,"logits_mean_abs":self.logits_mean,"hidden_jump_diagnostic":JUMP_MAX})
        );
    }
}

#[derive(Serialize, Deserialize)]
pub struct Metrics {
    max_abs: f64,
    mean_abs: f64,
    rel_rmse: f64,
    cosine: f64,
    exact: bool,
}

fn metrics(actual: &[f32], expected: &[f32]) -> Result<Metrics> {
    ensure!(
        !actual.is_empty() && actual.len() == expected.len(),
        "comparison shape mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut max_abs = 0f64;
    let (mut sum_abs, mut square_error, mut square_ref, mut square_actual, mut dot) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut exact = true;
    for (&a, &r) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && r.is_finite(), "nonfinite comparison value");
        let e = (a - r).abs();
        max_abs = max_abs.max(e as f64);
        sum_abs += e as f64;
        square_error += (e as f64).powi(2);
        square_ref += (r as f64).powi(2);
        square_actual += (a as f64).powi(2);
        dot += a as f64 * r as f64;
        exact &= a.to_bits() == r.to_bits();
    }
    let rel_rmse = if square_ref == 0.0 {
        if square_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (square_error / square_ref).sqrt()
    };
    let cosine = if square_actual == 0.0 || square_ref == 0.0 {
        if exact { 1.0 } else { 0.0 }
    } else {
        dot / (square_actual * square_ref).sqrt()
    };
    Ok(Metrics {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        rel_rmse,
        cosine,
        exact,
    })
}

fn floats(t: TensorView<'_>) -> Result<Vec<f32>> {
    Ok(match t.dtype() {
        Dtype::F32 => t
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        Dtype::BF16 => t
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
            .collect(),
        dtype => anyhow::bail!("reference tensor has unsupported dtype {dtype:?}"),
    })
}

fn ids(t: TensorView<'_>) -> Result<Vec<u32>> {
    ensure!(t.dtype() == Dtype::I64, "reference token ids must be I64");
    t.data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u32::try_from(i64::from_le_bytes(*v)).map_err(Into::into))
        .collect()
}

fn top_two(values: &[f32]) -> Result<(u32, f32)> {
    ensure!(values.len() >= 2, "logits vocabulary is too small");
    let (mut best, mut second, mut id) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0u32);
    for (i, &v) in values.iter().enumerate() {
        ensure!(v.is_finite(), "nonfinite logits");
        if v > best {
            second = best;
            best = v;
            id = i as u32;
        } else if v > second {
            second = v;
        }
    }
    Ok((id, best - second))
}

struct Reference {
    input: Vec<u32>,
    greedy: Vec<u32>,
    margins: Vec<f32>,
    logits: Vec<f32>,
}

pub struct GreedyReference {
    pub input: Vec<u32>,
    pub tokens: Vec<u32>,
    pub margins: Vec<f32>,
    logits: Vec<f32>,
}

impl GreedyReference {
    pub fn read(path: &Path, vocab: usize, steps: usize) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let reference = Reference::read_steps(&tensors, vocab, steps)?;
        Ok(Self {
            input: reference.input,
            tokens: reference.greedy,
            margins: reference.margins,
            logits: reference.logits,
        })
    }
}

impl Reference {
    fn read(t: &SafeTensors<'_>, vocab: usize) -> Result<Self> {
        Self::read_steps(t, vocab, STEPS)
    }

    fn read_steps(t: &SafeTensors<'_>, vocab: usize, steps: usize) -> Result<Self> {
        let input_tensor = t.tensor("input_ids")?;
        ensure!(
            input_tensor.shape().len() == 1,
            "input_ids must have shape [seq]"
        );
        let input = ids(input_tensor)?;
        ensure!(
            !input.is_empty() && input.iter().all(|&v| (v as usize) < vocab),
            "invalid reference input ids"
        );
        let greedy_tensor = t.tensor("greedy_ids")?;
        ensure!(
            greedy_tensor.shape() == [steps],
            "greedy_ids must have {steps} steps"
        );
        let greedy = ids(greedy_tensor)?;
        let margin_tensor = t.tensor("margins")?;
        ensure!(
            margin_tensor.dtype() == Dtype::F32 && margin_tensor.shape() == [steps],
            "margins must be FP32 [{steps}]"
        );
        let margins = floats(margin_tensor)?;
        ensure!(
            margins.iter().all(|v| v.is_finite() && *v >= 0.0),
            "margins must be finite and nonnegative"
        );
        let logits_tensor = t.tensor("logits_per_step")?;
        ensure!(
            logits_tensor.dtype() == Dtype::F32 && logits_tensor.shape() == [steps, vocab],
            "logits_per_step must be FP32 [{steps}, vocab]"
        );
        let logits = floats(logits_tensor)?;
        for step in 0..steps {
            let (token, margin) = top_two(&logits[step * vocab..(step + 1) * vocab])?;
            ensure!(
                token == greedy[step],
                "reference greedy_ids[{step}] disagrees with logits argmax"
            );
            ensure!(
                margin.to_bits() == margins[step].to_bits(),
                "reference margins[{step}] disagrees with top1-top2"
            );
        }
        Ok(Self {
            input,
            greedy,
            margins,
            logits,
        })
    }
}

fn hidden_compare(
    reference: &SafeTensors<'_>,
    trace: &Trace,
    rows: usize,
    width: usize,
    rules: &Rules,
) -> Result<bool> {
    let parts = trace.prefill_parts()?;
    let mut passed = true;
    let mut previous = 0f64;
    for layer in 0..rules.hidden.len() {
        let tensor = reference.tensor(&format!("hidden_states.{layer}"))?;
        let shape = tensor.shape();
        ensure!(
            tensor.dtype() == Dtype::F32
                && shape.len() == 2
                && shape[1] == width
                && shape[0] > 0
                && shape[0] <= rows,
            "hidden_states.{layer} must be FP32 [full sequence or tail, hidden]"
        );
        let expected_rows = shape[0];
        let expected = floats(tensor)?;
        let suffix = if layer == 0 {
            "embed".into()
        } else if layer + 1 == rules.hidden.len() {
            "norm".into()
        } else {
            format!("layer.{}", layer - 1)
        };
        let mut actual = Vec::with_capacity(rows * width);
        for part in 0..parts {
            actual.extend(trace.values(&format!("prefill.{part}.{suffix}"))?);
        }
        ensure!(
            actual.len() == rows * width || actual.len() == expected_rows * width,
            "unexpected captured hidden length at layer {layer}"
        );
        let actual = &actual[actual.len() - expected_rows * width..];
        let m = metrics(actual, &expected).with_context(|| format!("hidden_states.{layer}"))?;
        let jump = if layer > 1 && previous > 0.0 {
            Some(m.rel_rmse / previous)
        } else {
            None
        };
        let pass = if layer == 0 {
            m.exact
        } else {
            rules.hidden[layer].accepts(&m)
        };
        println!(
            "{}",
            serde_json::json!({"tensor":format!("hidden_states.{layer}"),"metrics":m,"rel_rmse_limit":rules.hidden[layer],"jump_ratio":jump,"jump_flag":jump.is_some_and(|v|v>JUMP_MAX),"pass":pass})
        );
        passed &= pass;
        previous = m.rel_rmse;
    }
    Ok(passed)
}

fn logits_compare(
    actual: &[f32],
    reference: &Reference,
    step: usize,
    vocab: usize,
    token: u32,
    rules: &Rules,
) -> Result<(bool, bool, Metrics)> {
    let m = metrics(actual, &reference.logits[step * vocab..(step + 1) * vocab])?;
    ensure!(
        top_two(actual)?.0 == token,
        "device argmax disagrees with logits at step {step}"
    );
    let exact = token == reference.greedy[step];
    let near_tie = !exact && (reference.margins[step] as f64) < rules.logits_max[step];
    let pass = exact || near_tie;
    let warn = m.max_abs > rules.logits_max[step] || m.mean_abs > rules.logits_mean[step];
    println!(
        "{}",
        serde_json::json!({"step":step,"logits":m,"mean_abs_limit":rules.logits_mean[step],"max_abs_limit":rules.logits_max[step],"actual_token":token,"expected_token":reference.greedy[step],"reference_margin":reference.margins[step],"exact_match":exact,"near_tie_exception":near_tie,"warn":warn,"pass":pass})
    );
    Ok((pass, near_tie, m))
}

pub fn compare_prefill(
    fixture: &Path,
    dump: &Path,
    vocab: usize,
    hidden: usize,
    rules: &Rules,
) -> Result<()> {
    let bytes = std::fs::read(fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let reference = Reference::read(&tensors, vocab)?;
    let trace = Trace::read(dump)?;
    rules.print();
    let hidden_pass = hidden_compare(&tensors, &trace, reference.input.len(), hidden, rules)?;
    let last = trace.prefill_parts()? - 1;
    let logits = trace.values(&format!("prefill.{last}.logits"))?;
    ensure!(
        logits.len() >= vocab && logits.len().is_multiple_of(vocab),
        "invalid prefill logits shape"
    );
    let last_logits = &logits[logits.len() - vocab..];
    let (logits_pass, _, _) = logits_compare(
        last_logits,
        &reference,
        0,
        vocab,
        top_two(last_logits)?.0,
        rules,
    )?;
    println!(
        "{}",
        serde_json::json!({"fixture":fixture,"dump":dump,"scope":"prefill_only","hidden_pass":hidden_pass,"logits_pass":logits_pass,"pass":hidden_pass && logits_pass})
    );
    ensure!(hidden_pass && logits_pass, "prefill comparison failed");
    Ok(())
}

pub fn run(model: &mut Model, fixture: &Path, rules: &Rules, graph: bool) -> Result<()> {
    let bytes = std::fs::read(fixture)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let vocab = model.config.vocab_size;
    let reference = Reference::read(&tensors, vocab)?;
    rules.print();
    let mut trace = Trace::default();
    let tail = tensors.tensor("hidden_states.0")?.shape()[0];
    let first = model.prefill_tail(&reference.input, &mut trace, tail)?;
    let mut passed = hidden_compare(
        &tensors,
        &trace,
        reference.input.len(),
        model.config.hidden_size,
        rules,
    )?;
    let last = trace.prefill_parts()? - 1;
    let first_logits = trace.values(&format!("prefill.{last}.logits"))?;
    let (pass, near, metrics) = logits_compare(
        &first_logits[first_logits.len() - vocab..],
        &reference,
        0,
        vocab,
        first,
        rules,
    )?;
    passed &= pass;
    let mut logit_metrics = vec![metrics];
    let mut near_ties = usize::from(near);
    if graph {
        model.capture()?;
    }
    for step in 1..STEPS {
        model.set_token(reference.greedy[step - 1])?;
        let token = model.decode(graph, None)?;
        let actual = model.decode_logits()?;
        let (pass, near, metrics) = logits_compare(&actual, &reference, step, vocab, token, rules)?;
        logit_metrics.push(metrics);
        passed &= pass;
        near_ties += usize::from(near);
    }
    passed &= rules.systematic(&logit_metrics)?;
    println!(
        "{}",
        serde_json::json!({"fixture":fixture,"teacher_forced":true,"steps":STEPS,"graph":graph,"near_tie_exceptions":near_ties,"pass":passed})
    );
    ensure!(passed, "fixture comparison failed");
    Ok(())
}

pub fn compare_saved(fixture: &Path, log: &Path, rules: &Rules) -> Result<()> {
    let text = std::fs::read_to_string(log)?;
    let mut hidden = BTreeMap::new();
    let mut logits = BTreeMap::new();
    let mut source = None;
    for line in text.lines().filter(|x| x.starts_with('{')) {
        let entry: serde_json::Value = serde_json::from_str(line)?;
        if let Some(name) = entry
            .get("tensor")
            .and_then(|v| v.as_str())
            .and_then(|x| x.strip_prefix("hidden_states."))
        {
            let index: usize = name.parse()?;
            let m: Metrics = serde_json::from_value(
                entry
                    .get("metrics")
                    .context("saved hidden metrics missing")?
                    .clone(),
            )?;
            ensure!(
                hidden.insert(index, m).is_none(),
                "duplicate saved hidden layer {index}"
            );
        } else if let Some(step) = entry.get("step").and_then(|v| v.as_u64()) {
            ensure!(
                logits.insert(step as usize, entry).is_none(),
                "duplicate saved logit step {step}"
            );
        } else if entry.get("teacher_forced").is_some() {
            source = Some(entry);
        }
    }
    ensure!(
        hidden.keys().copied().eq(0..rules.hidden.len()),
        "saved log must cover all hidden layers"
    );
    ensure!(
        logits.keys().copied().eq(0..STEPS),
        "saved log must cover all 64 logit steps"
    );
    let source = source.context("saved source summary missing")?;
    ensure!(
        source.get("teacher_forced").and_then(|v| v.as_bool()) == Some(true),
        "saved log must be teacher-forced"
    );
    let source_fixture = source
        .get("fixture")
        .and_then(|v| v.as_str())
        .context("source fixture path missing")?;
    ensure!(
        Path::new(source_fixture).canonicalize()? == fixture.canonicalize()?,
        "saved log fixture identity mismatch"
    );
    rules.print();
    let mut passed = true;
    let mut exact_tokens = 0usize;
    let mut near_ties = 0usize;
    for (layer, m) in &hidden {
        let pass = if *layer == 0 {
            m.exact
        } else {
            rules.hidden[*layer].accepts(m)
        };
        passed &= pass;
        println!(
            "{}",
            serde_json::json!({"saved_layer":layer,"metrics":m,"limit":rules.hidden[*layer],"pass":pass})
        );
    }
    let mut logit_metrics = Vec::with_capacity(STEPS);
    for (step, entry) in &logits {
        let m: Metrics = serde_json::from_value(
            entry
                .get("logits")
                .context("saved logit metrics missing")?
                .clone(),
        )?;
        let a = entry
            .get("actual_token")
            .and_then(|v| v.as_u64())
            .context("saved actual token missing")?;
        let e = entry
            .get("expected_token")
            .and_then(|v| v.as_u64())
            .context("saved expected token missing")?;
        let margin = entry
            .get("reference_margin")
            .and_then(|v| v.as_f64())
            .context("saved margin missing")?;
        let exact = a == e;
        let near = !exact && margin < rules.logits_max[*step];
        let pass = exact || near;
        let warn = m.max_abs > rules.logits_max[*step] || m.mean_abs > rules.logits_mean[*step];
        passed &= pass;
        exact_tokens += usize::from(exact);
        near_ties += usize::from(near);
        println!(
            "{}",
            serde_json::json!({"saved_step":step,"metrics":m,"mean_limit":rules.logits_mean[*step],"max_limit":rules.logits_max[*step],"exact_token":exact,"near_tie":near,"warn":warn,"pass":pass})
        );
        logit_metrics.push(m);
    }
    passed &= rules.systematic(&logit_metrics)?;
    println!(
        "{}",
        serde_json::json!({"fixture":fixture,"saved_log":log,"scope":"cpu_recheck_saved_metrics","hidden_count":hidden.len(),"steps":logits.len(),"exact_token_matches":exact_tokens,"near_tie_exceptions":near_ties,"pass":passed})
    );
    ensure!(passed, "saved metric comparison failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_byte_decoding_preserves_bits_and_id_bounds() -> Result<()> {
        let bits = [
            0u32, 0x80000000, 1, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc12345,
        ];
        let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![bits.len()], &bytes)?;
        assert_eq!(
            floats(tensor)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        let bits = [0u16, 0x8000, 1, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1];
        let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::BF16, vec![bits.len()], &bytes)?;
        assert_eq!(
            floats(tensor)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bits.map(|v| u32::from(v) << 16)
        );
        let values = [0i64, 130559, i64::from(u32::MAX)];
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::I64, vec![values.len()], &bytes)?;
        assert_eq!(ids(tensor)?, [0, 130559, u32::MAX]);
        for value in [-1i64, i64::from(u32::MAX) + 1] {
            let bytes = value.to_le_bytes();
            assert!(ids(TensorView::new(Dtype::I64, vec![1], &bytes)?).is_err());
        }
        Ok(())
    }

    #[test]
    fn systematic_gates_detect_small_but_persistent_errors() -> Result<()> {
        let mut rules = Rules::read(None, None, Path::new("p1.safetensors"), 42)?;
        rules.systematic = Some(SystemRule {
            gates: Gates {
                ratio_median_max: 1.5,
                ratio_p90_max: 2.5,
                outlier_binomial: Binomial {
                    p: OUTLIER_RATE,
                    alpha: OUTLIER_ALPHA,
                },
            },
            mean: vec![0.01; STEPS],
            max: vec![0.2; STEPS],
        });
        let make = |ratio: f64| Metrics {
            max_abs: 0.2 * ratio,
            mean_abs: 0.01 * ratio,
            rel_rmse: 0.0,
            cosine: 1.0,
            exact: false,
        };
        let mut values: Vec<_> = (0..STEPS).map(|_| make(1.25)).collect();
        assert!(rules.systematic(&values)?);
        values[0] = make(5.0);
        assert!(rules.systematic(&values)?);
        for item in &mut values[..8] {
            *item = make(3.0);
        }
        assert!(!rules.systematic(&values)?);
        let values: Vec<_> = (0..STEPS).map(|_| make(2.0)).collect();
        assert!(!rules.systematic(&values)?);
        Ok(())
    }

    #[test]
    fn per_layer_cosine_schema_controls_acceptance() -> Result<()> {
        let legacy: HiddenLimit = serde_json::from_str("0.2")?;
        let current: HiddenLimit = serde_json::from_str(r#"{"rel":0.2,"cos_min":0.99}"#)?;
        let m = metrics(&[1.0, 0.1], &[1.0, 0.0])?;
        assert!(!legacy.value().accepts(&m));
        assert!(current.value().accepts(&m));
        assert!(serde_json::from_str::<HiddenLimit>(r#"{"rel":0.2}"#).is_err());
        Ok(())
    }

    #[test]
    fn outlier_scale_and_localized_error() -> Result<()> {
        let reference = [10000.0, 1.0, -1.0];
        let got = [10098.0, 1.0, -1.0];
        let m = metrics(&got, &reference)?;
        assert_eq!(m.max_abs, 98.0);
        assert!(m.rel_rmse < 0.05 && m.cosine >= COSINE_MIN);
        let m = metrics(&[0.0, 10.0, -1.0], &reference)?;
        assert!(m.rel_rmse > 0.05 || m.cosine < COSINE_MIN);
        Ok(())
    }

    #[test]
    fn near_tie_and_single_step_warnings_do_not_excuse_wrong_tokens() -> Result<()> {
        let rules = Rules::read(None, None, Path::new("p1.safetensors"), 42)?;
        let r = Reference {
            input: vec![0],
            greedy: vec![0],
            margins: vec![0.05],
            logits: vec![1.0, 0.95],
        };
        assert!(logits_compare(&[0.96, 0.99], &r, 0, 2, 1, &rules)?.0);
        assert!(logits_compare(&[1.0, 3.0], &r, 0, 2, 1, &rules)?.0);
        let mut r = Reference {
            input: vec![0],
            greedy: vec![0],
            margins: vec![1.5],
            logits: vec![2.0, 0.5],
        };
        r.logits.resize(64, 0.0);
        let mut got = vec![0.0; 64];
        got[..2].copy_from_slice(&[1.2, 1.3]);
        assert!(!logits_compare(&got, &r, 0, 64, 1, &rules)?.0);
        Ok(())
    }

    #[test]
    fn outlier_counts_are_independent_per_metric_and_use_binomial_significance() -> Result<()> {
        let mut rules = Rules::read(None, None, Path::new("p1.safetensors"), 42)?;
        rules.logits_mean = vec![0.1; 256];
        rules.logits_max = vec![1.5; 256];
        let mut values: Vec<_> = (0..256)
            .map(|_| Metrics {
                max_abs: 1.0,
                mean_abs: 0.05,
                rel_rmse: 0.0,
                cosine: 1.0,
                exact: false,
            })
            .collect();
        for value in &mut values[..4] {
            value.mean_abs = 0.11;
        }
        for value in &mut values[4..8] {
            value.max_abs = 1.6;
        }
        assert!(rules.systematic(&values)?);
        values[8].mean_abs = 0.11;
        assert!(!rules.systematic(&values)?);
        values[8].mean_abs = 0.05;
        values[8].max_abs = 1.6;
        assert!(!rules.systematic(&values)?);
        Ok(())
    }

    #[test]
    fn binomial_count_cutoffs_match_the_approved_64_and_256_step_cases() -> Result<()> {
        assert!(outlier_p_value(64, 2, OUTLIER_RATE)? > OUTLIER_ALPHA);
        assert!(outlier_p_value(64, 3, OUTLIER_RATE)? < OUTLIER_ALPHA);
        assert!(outlier_p_value(256, 4, OUTLIER_RATE)? > OUTLIER_ALPHA);
        assert!(outlier_p_value(256, 5, OUTLIER_RATE)? < OUTLIER_ALPHA);
        assert_eq!(outlier_p_value(256, 0, OUTLIER_RATE)?, 1.0);
        assert!(outlier_p_value(64, 65, OUTLIER_RATE).is_err());
        Ok(())
    }

    #[test]
    fn ties_select_smallest_id_and_positive_margin() -> Result<()> {
        assert_eq!(top_two(&[-2.0, -2.0, -3.0])?, (0, 0.0));
        assert_eq!(top_two(&[-3.0, -1.0, -2.0])?, (1, 1.0));
        Ok(())
    }

    #[test]
    fn variable_length_reference_validates_ids_and_margins() -> Result<()> {
        let input = 0i64.to_le_bytes();
        let logits: Vec<u8> = [1f32, 2.0, 2.0, -1.0, 0.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let make = |tokens: [i64; 2], margins: [f32; 2]| -> Result<Vec<u8>> {
            let tokens: Vec<u8> = tokens.into_iter().flat_map(i64::to_le_bytes).collect();
            let margins: Vec<u8> = margins.into_iter().flat_map(f32::to_le_bytes).collect();
            Ok(safetensors::tensor::serialize(
                [
                    ("input_ids", TensorView::new(Dtype::I64, vec![1], &input)?),
                    ("greedy_ids", TensorView::new(Dtype::I64, vec![2], &tokens)?),
                    ("margins", TensorView::new(Dtype::F32, vec![2], &margins)?),
                    (
                        "logits_per_step",
                        TensorView::new(Dtype::F32, vec![2, 3], &logits)?,
                    ),
                ],
                None,
            )?)
        };
        let data = make([1, 2], [0.0, 3.0])?;
        let tensors = SafeTensors::deserialize(&data)?;
        assert_eq!(Reference::read_steps(&tensors, 3, 2)?.greedy, [1, 2]);
        assert!(Reference::read_steps(&tensors, 3, 64).is_err());
        let data = make([2, 2], [0.0, 3.0])?;
        assert!(Reference::read_steps(&SafeTensors::deserialize(&data)?, 3, 2).is_err());
        let data = make([1, 2], [1.0, 3.0])?;
        assert!(Reference::read_steps(&SafeTensors::deserialize(&data)?, 3, 2).is_err());
        Ok(())
    }
}
