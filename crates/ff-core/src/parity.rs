use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, safetensors::Load};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    hash::BuildHasher,
    path::Path,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityTolerance {
    pub atol: f64,
    pub rtol: f64,
}

impl ParityTolerance {
    pub fn new(atol: f64, rtol: f64) -> Result<Self> {
        let tolerance = Self { atol, rtol };
        tolerance.validate()?;
        Ok(tolerance)
    }

    fn validate(self) -> Result<()> {
        anyhow::ensure!(
            self.atol.is_finite() && self.atol >= 0.0,
            "parity atol must be finite and non-negative"
        );
        anyhow::ensure!(
            self.rtol.is_finite() && self.rtol >= 0.0,
            "parity rtol must be finite and non-negative"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorParityStatus {
    Passed,
    ValueMismatch,
    ShapeMismatch,
    DtypeMismatch,
    Missing,
    Unexpected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorParityReport {
    pub name: String,
    pub status: TensorParityStatus,
    pub passed: bool,
    #[serde(deserialize_with = "crate::required_option")]
    pub reference_shape: Option<Vec<usize>>,
    #[serde(deserialize_with = "crate::required_option")]
    pub actual_shape: Option<Vec<usize>>,
    #[serde(deserialize_with = "crate::required_option")]
    pub reference_dtype: Option<String>,
    #[serde(deserialize_with = "crate::required_option")]
    pub actual_dtype: Option<String>,
    pub compared_elements: u64,
    pub mismatched_elements: u64,
    pub non_finite_elements: u64,
    pub non_finite_mismatches: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub max_abs: Option<f64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub max_rel: Option<f64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub rmse: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityReport {
    pub passed: bool,
    pub unexpected_allowed: bool,
    pub tolerance: ParityTolerance,
    pub reference_tensor_count: usize,
    pub actual_tensor_count: usize,
    pub matched_tensor_count: usize,
    pub compared_tensor_count: usize,
    pub failed_tensor_count: usize,
    pub compared_elements: u64,
    pub mismatched_elements: u64,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub tensors: Vec<TensorParityReport>,
}

impl ParityReport {
    pub fn to_pretty_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn allow_unexpected(mut self) -> Self {
        self.unexpected_allowed = true;
        let accepted = self
            .tensors
            .iter()
            .filter(|tensor| tensor.status == TensorParityStatus::Unexpected)
            .count();
        self.failed_tensor_count = self.failed_tensor_count.saturating_sub(accepted);
        self.passed = self.failed_tensor_count == 0;
        self
    }
}

pub trait TensorMap {
    fn tensor_names(&self) -> Vec<&str>;
    fn tensor(&self, name: &str) -> Option<&Tensor>;
}

impl<S: BuildHasher> TensorMap for HashMap<String, Tensor, S> {
    fn tensor_names(&self) -> Vec<&str> {
        self.keys().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.get(name)
    }
}

impl TensorMap for BTreeMap<String, Tensor> {
    fn tensor_names(&self) -> Vec<&str> {
        self.keys().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.get(name)
    }
}

pub fn compare_safetensors(
    reference_path: impl AsRef<Path>,
    actual_path: impl AsRef<Path>,
    tolerance: ParityTolerance,
) -> Result<ParityReport> {
    tolerance.validate()?;
    let reference_path = reference_path.as_ref();
    let actual_path = actual_path.as_ref();
    let reference_file = File::open(reference_path).with_context(|| {
        format!(
            "failed to open reference safetensors {}",
            reference_path.display()
        )
    })?;
    let actual_file = File::open(actual_path).with_context(|| {
        format!(
            "failed to open actual safetensors {}",
            actual_path.display()
        )
    })?;
    let reference_mmap = unsafe { MmapOptions::new().map(&reference_file) }.with_context(|| {
        format!(
            "failed to mmap reference safetensors {}",
            reference_path.display()
        )
    })?;
    let actual_mmap = unsafe { MmapOptions::new().map(&actual_file) }.with_context(|| {
        format!(
            "failed to mmap actual safetensors {}",
            actual_path.display()
        )
    })?;
    let reference = SafeTensors::deserialize(&reference_mmap)
        .with_context(|| format!("invalid reference safetensors {}", reference_path.display()))?;
    let actual = SafeTensors::deserialize(&actual_mmap)
        .with_context(|| format!("invalid actual safetensors {}", actual_path.display()))?;

    compare_safetensor_views(&reference, &actual, tolerance)
}

pub fn compare_tensor_maps<R: TensorMap + ?Sized, A: TensorMap + ?Sized>(
    reference: &R,
    actual: &A,
    tolerance: ParityTolerance,
) -> Result<ParityReport> {
    tolerance.validate()?;
    let reference_names = names(reference.tensor_names());
    let actual_names = names(actual.tensor_names());
    compare_name_sets(
        &reference_names,
        &actual_names,
        tolerance,
        |name| {
            let reference = reference.tensor(name).with_context(|| {
                format!("reference tensor disappeared during comparison: {name}")
            })?;
            let actual = actual
                .tensor(name)
                .with_context(|| format!("actual tensor disappeared during comparison: {name}"))?;
            compare_candle_pair(name, reference, actual, tolerance)
        },
        |name, side| {
            let tensor = match side {
                Side::Reference => reference.tensor(name),
                Side::Actual => actual.tensor(name),
            }
            .with_context(|| format!("tensor disappeared during comparison: {name}"))?;
            Ok(TensorDescription {
                shape: tensor.dims().to_vec(),
                dtype: tensor.dtype().as_str().to_owned(),
            })
        },
    )
}

pub fn compare_safetensors_to_tensor_map<A: TensorMap + ?Sized>(
    reference_path: impl AsRef<Path>,
    actual: &A,
    tolerance: ParityTolerance,
) -> Result<ParityReport> {
    tolerance.validate()?;
    let reference_path = reference_path.as_ref();
    let reference_file = File::open(reference_path).with_context(|| {
        format!(
            "failed to open reference safetensors {}",
            reference_path.display()
        )
    })?;
    let reference_mmap = unsafe { MmapOptions::new().map(&reference_file) }.with_context(|| {
        format!(
            "failed to mmap reference safetensors {}",
            reference_path.display()
        )
    })?;
    let reference = SafeTensors::deserialize(&reference_mmap)
        .with_context(|| format!("invalid reference safetensors {}", reference_path.display()))?;
    let reference_names = names(reference.names());
    let actual_names = names(actual.tensor_names());

    compare_name_sets(
        &reference_names,
        &actual_names,
        tolerance,
        |name| {
            let reference_view = reference.tensor(name).with_context(|| {
                format!("reference tensor disappeared during comparison: {name}")
            })?;
            let actual_tensor = actual
                .tensor(name)
                .with_context(|| format!("actual tensor disappeared during comparison: {name}"))?;
            compare_view_and_candle(name, &reference_view, actual_tensor, tolerance)
        },
        |name, side| match side {
            Side::Reference => {
                let view = reference.tensor(name).with_context(|| {
                    format!("reference tensor disappeared during comparison: {name}")
                })?;
                Ok(describe_view(&view))
            }
            Side::Actual => {
                let tensor = actual.tensor(name).with_context(|| {
                    format!("actual tensor disappeared during comparison: {name}")
                })?;
                Ok(TensorDescription {
                    shape: tensor.dims().to_vec(),
                    dtype: tensor.dtype().as_str().to_owned(),
                })
            }
        },
    )
}

fn compare_safetensor_views(
    reference: &SafeTensors<'_>,
    actual: &SafeTensors<'_>,
    tolerance: ParityTolerance,
) -> Result<ParityReport> {
    let reference_names = names(reference.names());
    let actual_names = names(actual.names());
    compare_name_sets(
        &reference_names,
        &actual_names,
        tolerance,
        |name| {
            let reference = reference.tensor(name).with_context(|| {
                format!("reference tensor disappeared during comparison: {name}")
            })?;
            let actual = actual
                .tensor(name)
                .with_context(|| format!("actual tensor disappeared during comparison: {name}"))?;
            compare_view_pair(name, &reference, &actual, tolerance)
        },
        |name, side| {
            let view = match side {
                Side::Reference => reference.tensor(name),
                Side::Actual => actual.tensor(name),
            }
            .with_context(|| format!("tensor disappeared during comparison: {name}"))?;
            Ok(describe_view(&view))
        },
    )
}

#[derive(Clone, Copy)]
enum Side {
    Reference,
    Actual,
}

struct TensorDescription {
    shape: Vec<usize>,
    dtype: String,
}

fn names<'a>(values: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    values.into_iter().map(str::to_owned).collect()
}

fn compare_name_sets(
    reference_names: &BTreeSet<String>,
    actual_names: &BTreeSet<String>,
    tolerance: ParityTolerance,
    mut compare_pair: impl FnMut(&str) -> Result<TensorParityReport>,
    mut describe: impl FnMut(&str, Side) -> Result<TensorDescription>,
) -> Result<ParityReport> {
    let missing_tensors = reference_names
        .difference(actual_names)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_tensors = actual_names
        .difference(reference_names)
        .cloned()
        .collect::<Vec<_>>();
    let matched_tensor_count = reference_names.intersection(actual_names).count();
    let mut tensors = Vec::with_capacity(reference_names.union(actual_names).count());

    for name in reference_names.union(actual_names) {
        if !actual_names.contains(name) {
            let reference = describe(name, Side::Reference)?;
            tensors.push(TensorParityReport::unpaired(
                name,
                TensorParityStatus::Missing,
                Some(reference),
                None,
            ));
        } else if !reference_names.contains(name) {
            let actual = describe(name, Side::Actual)?;
            tensors.push(TensorParityReport::unpaired(
                name,
                TensorParityStatus::Unexpected,
                None,
                Some(actual),
            ));
        } else {
            tensors.push(compare_pair(name)?);
        }
    }

    let compared_tensor_count = tensors
        .iter()
        .filter(|tensor| {
            matches!(
                tensor.status,
                TensorParityStatus::Passed | TensorParityStatus::ValueMismatch
            )
        })
        .count();
    let failed_tensor_count = tensors.iter().filter(|tensor| !tensor.passed).count();
    let compared_elements = checked_sum(
        tensors.iter().map(|tensor| tensor.compared_elements),
        "parity compared-element count overflow",
    )?;
    let mismatched_elements = checked_sum(
        tensors.iter().map(|tensor| tensor.mismatched_elements),
        "parity mismatch count overflow",
    )?;

    Ok(ParityReport {
        passed: failed_tensor_count == 0,
        unexpected_allowed: false,
        tolerance,
        reference_tensor_count: reference_names.len(),
        actual_tensor_count: actual_names.len(),
        matched_tensor_count,
        compared_tensor_count,
        failed_tensor_count,
        compared_elements,
        mismatched_elements,
        missing_tensors,
        unexpected_tensors,
        tensors,
    })
}

fn checked_sum(mut values: impl Iterator<Item = u64>, message: &'static str) -> Result<u64> {
    values.try_fold(0u64, |sum, value| sum.checked_add(value).context(message))
}

impl TensorParityReport {
    fn unpaired(
        name: &str,
        status: TensorParityStatus,
        reference: Option<TensorDescription>,
        actual: Option<TensorDescription>,
    ) -> Self {
        debug_assert!(matches!(
            status,
            TensorParityStatus::Missing | TensorParityStatus::Unexpected
        ));
        Self {
            name: name.to_owned(),
            status,
            passed: false,
            reference_shape: reference.as_ref().map(|value| value.shape.clone()),
            actual_shape: actual.as_ref().map(|value| value.shape.clone()),
            reference_dtype: reference.map(|value| value.dtype),
            actual_dtype: actual.map(|value| value.dtype),
            compared_elements: 0,
            mismatched_elements: 0,
            non_finite_elements: 0,
            non_finite_mismatches: 0,
            max_abs: None,
            max_rel: None,
            rmse: None,
        }
    }

    fn shape_mismatch(name: &str, reference: TensorDescription, actual: TensorDescription) -> Self {
        Self {
            name: name.to_owned(),
            status: TensorParityStatus::ShapeMismatch,
            passed: false,
            reference_shape: Some(reference.shape),
            actual_shape: Some(actual.shape),
            reference_dtype: Some(reference.dtype),
            actual_dtype: Some(actual.dtype),
            compared_elements: 0,
            mismatched_elements: 0,
            non_finite_elements: 0,
            non_finite_mismatches: 0,
            max_abs: None,
            max_rel: None,
            rmse: None,
        }
    }

    fn dtype_mismatch(name: &str, reference: TensorDescription, actual: TensorDescription) -> Self {
        Self {
            name: name.to_owned(),
            status: TensorParityStatus::DtypeMismatch,
            passed: false,
            reference_shape: Some(reference.shape),
            actual_shape: Some(actual.shape),
            reference_dtype: Some(reference.dtype),
            actual_dtype: Some(actual.dtype),
            compared_elements: 0,
            mismatched_elements: 0,
            non_finite_elements: 0,
            non_finite_mismatches: 0,
            max_abs: None,
            max_rel: None,
            rmse: None,
        }
    }
}

fn describe_view(view: &TensorView<'_>) -> TensorDescription {
    TensorDescription {
        shape: view.shape().to_vec(),
        dtype: safetensors_dtype_name(view.dtype()),
    }
}

fn safetensors_dtype_name(dtype: Dtype) -> String {
    match dtype {
        Dtype::F4 => "f4".to_owned(),
        Dtype::F6_E2M3 => "f6e2m3".to_owned(),
        Dtype::F6_E3M2 => "f6e3m2".to_owned(),
        Dtype::U8 => "u8".to_owned(),
        Dtype::F8_E4M3 => "f8e4m3".to_owned(),
        Dtype::F8_E8M0 => "f8e8m0".to_owned(),
        Dtype::I16 => "i16".to_owned(),
        Dtype::F16 => "f16".to_owned(),
        Dtype::BF16 => "bf16".to_owned(),
        Dtype::I32 => "i32".to_owned(),
        Dtype::U32 => "u32".to_owned(),
        Dtype::F32 => "f32".to_owned(),
        Dtype::F64 => "f64".to_owned(),
        Dtype::I64 => "i64".to_owned(),
        unsupported => format!("{unsupported:?}").to_ascii_lowercase(),
    }
}

fn compare_view_pair(
    name: &str,
    reference: &TensorView<'_>,
    actual: &TensorView<'_>,
    tolerance: ParityTolerance,
) -> Result<TensorParityReport> {
    let reference_description = describe_view(reference);
    let actual_description = describe_view(actual);
    if reference_description.shape != actual_description.shape {
        return Ok(TensorParityReport::shape_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }
    if reference_description.dtype != actual_description.dtype {
        return Ok(TensorParityReport::dtype_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }

    let reference_values = view_values(reference)
        .with_context(|| format!("failed to read reference tensor {name}"))?;
    let actual_values =
        view_values(actual).with_context(|| format!("failed to read actual tensor {name}"))?;
    compare_values(
        name,
        reference_description,
        actual_description,
        &reference_values,
        &actual_values,
        tolerance,
    )
}

fn compare_candle_pair(
    name: &str,
    reference: &Tensor,
    actual: &Tensor,
    tolerance: ParityTolerance,
) -> Result<TensorParityReport> {
    let reference_description = TensorDescription {
        shape: reference.dims().to_vec(),
        dtype: reference.dtype().as_str().to_owned(),
    };
    let actual_description = TensorDescription {
        shape: actual.dims().to_vec(),
        dtype: actual.dtype().as_str().to_owned(),
    };
    if reference_description.shape != actual_description.shape {
        return Ok(TensorParityReport::shape_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }
    if reference_description.dtype != actual_description.dtype {
        return Ok(TensorParityReport::dtype_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }

    let reference_values = candle_values(reference)
        .with_context(|| format!("failed to read reference tensor {name}"))?;
    let actual_values =
        candle_values(actual).with_context(|| format!("failed to read actual tensor {name}"))?;
    compare_values(
        name,
        reference_description,
        actual_description,
        &reference_values,
        &actual_values,
        tolerance,
    )
}

fn compare_view_and_candle(
    name: &str,
    reference: &TensorView<'_>,
    actual: &Tensor,
    tolerance: ParityTolerance,
) -> Result<TensorParityReport> {
    let reference_description = describe_view(reference);
    let actual_description = TensorDescription {
        shape: actual.dims().to_vec(),
        dtype: actual.dtype().as_str().to_owned(),
    };
    if reference_description.shape != actual_description.shape {
        return Ok(TensorParityReport::shape_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }
    if reference_description.dtype != actual_description.dtype {
        return Ok(TensorParityReport::dtype_mismatch(
            name,
            reference_description,
            actual_description,
        ));
    }

    let reference_values = view_values(reference)
        .with_context(|| format!("failed to read reference tensor {name}"))?;
    let actual_values =
        candle_values(actual).with_context(|| format!("failed to read actual tensor {name}"))?;
    compare_values(
        name,
        reference_description,
        actual_description,
        &reference_values,
        &actual_values,
        tolerance,
    )
}

fn view_values(view: &TensorView<'_>) -> Result<Vec<f64>> {
    let tensor = view.load(&Device::Cpu)?;
    candle_values(&tensor)
}

fn candle_values(tensor: &Tensor) -> Result<Vec<f64>> {
    tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_dtype(DType::F64)?
        .to_vec1::<f64>()
        .map_err(Into::into)
}

fn compare_values(
    name: &str,
    reference_description: TensorDescription,
    actual_description: TensorDescription,
    reference: &[f64],
    actual: &[f64],
    tolerance: ParityTolerance,
) -> Result<TensorParityReport> {
    anyhow::ensure!(
        reference.len() == actual.len(),
        "same-shape tensors yielded different element counts for {name}"
    );
    let compared_elements = u64::try_from(reference.len())
        .with_context(|| format!("tensor {name} element count exceeds u64"))?;
    let mut mismatched_elements = 0u64;
    let mut non_finite_elements = 0u64;
    let mut non_finite_mismatches = 0u64;
    let mut finite_elements = 0u64;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut absolute_error_unbounded = false;
    let mut relative_error_unbounded = false;
    let mut squared_errors = SquaredErrorAccumulator::default();

    for (&reference, &actual) in reference.iter().zip(actual) {
        if !reference.is_finite() || !actual.is_finite() {
            non_finite_elements += 1;
            let passes = reference == actual && !reference.is_nan();
            if !passes {
                mismatched_elements += 1;
                non_finite_mismatches += 1;
            }
            continue;
        }

        finite_elements += 1;
        let error = (actual - reference).abs();
        let passes = if error.is_finite() {
            error <= tolerance.atol + tolerance.rtol * reference.abs()
        } else {
            finite_pair_passes_scaled(reference, actual, tolerance)
        };
        if !passes {
            mismatched_elements += 1;
        }

        if error.is_finite() {
            max_abs = max_abs.max(error);
            squared_errors.push(error);
        } else {
            absolute_error_unbounded = true;
        }

        if reference == 0.0 {
            if error != 0.0 {
                relative_error_unbounded = true;
            }
        } else {
            let relative = relative_error(reference, actual);
            if relative.is_finite() {
                max_rel = max_rel.max(relative);
            } else {
                relative_error_unbounded = true;
            }
        }
    }

    let passed = mismatched_elements == 0;
    Ok(TensorParityReport {
        name: name.to_owned(),
        status: if passed {
            TensorParityStatus::Passed
        } else {
            TensorParityStatus::ValueMismatch
        },
        passed,
        reference_shape: Some(reference_description.shape),
        actual_shape: Some(actual_description.shape),
        reference_dtype: Some(reference_description.dtype),
        actual_dtype: Some(actual_description.dtype),
        compared_elements,
        mismatched_elements,
        non_finite_elements,
        non_finite_mismatches,
        max_abs: (finite_elements > 0 && !absolute_error_unbounded).then_some(max_abs),
        max_rel: (finite_elements > 0 && !relative_error_unbounded).then_some(max_rel),
        rmse: (finite_elements > 0 && !absolute_error_unbounded)
            .then(|| squared_errors.rmse(finite_elements)),
    })
}

fn finite_pair_passes_scaled(reference: f64, actual: f64, tolerance: ParityTolerance) -> bool {
    let scale = reference
        .abs()
        .max(actual.abs())
        .max(tolerance.atol)
        .max(f64::MIN_POSITIVE);
    (actual / scale - reference / scale).abs()
        <= tolerance.atol / scale + tolerance.rtol * (reference.abs() / scale)
}

fn relative_error(reference: f64, actual: f64) -> f64 {
    let reference_abs = reference.abs();
    let actual_abs = actual.abs();
    if reference.is_sign_negative() == actual.is_sign_negative() {
        (actual_abs / reference_abs - 1.0).abs()
    } else {
        actual_abs / reference_abs + 1.0
    }
}

#[derive(Default)]
struct SquaredErrorAccumulator {
    scale: f64,
    scaled_sum: f64,
}

impl SquaredErrorAccumulator {
    fn push(&mut self, value: f64) {
        if value == 0.0 {
            return;
        }
        if self.scale < value {
            self.scaled_sum = 1.0 + self.scaled_sum * (self.scale / value).powi(2);
            self.scale = value;
        } else {
            self.scaled_sum += (value / self.scale).powi(2);
        }
    }

    fn rmse(&self, count: u64) -> f64 {
        if self.scale == 0.0 {
            0.0
        } else {
            self.scale * (self.scaled_sum / count as f64).sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::safetensors;
    use serde_json::Value;
    use tempfile::tempdir;

    fn tensor_map(
        entries: impl IntoIterator<Item = (&'static str, Tensor)>,
    ) -> BTreeMap<String, Tensor> {
        entries
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect()
    }

    #[test]
    fn applies_absolute_and_relative_tolerance() {
        let device = Device::Cpu;
        let reference = tensor_map([(
            "values",
            Tensor::from_slice(&[0.0f32, 10.0, -10.0], 3, &device).unwrap(),
        )]);
        let actual = tensor_map([(
            "values",
            Tensor::from_slice(&[0.005f32, 10.05, -10.2], 3, &device).unwrap(),
        )]);
        let report = compare_tensor_maps(
            &reference,
            &actual,
            ParityTolerance::new(0.01, 0.01).unwrap(),
        )
        .unwrap();

        assert!(!report.passed);
        assert_eq!(report.compared_elements, 3);
        assert_eq!(report.mismatched_elements, 1);
        let tensor = &report.tensors[0];
        assert_eq!(tensor.status, TensorParityStatus::ValueMismatch);
        assert_eq!(tensor.mismatched_elements, 1);
        assert!((tensor.max_abs.unwrap() - 0.2).abs() < 1e-5);
        assert!(
            tensor.max_rel.is_none(),
            "zero reference makes max_rel unbounded"
        );
        assert!((tensor.rmse.unwrap() - 0.119_058).abs() < 1e-5);
    }

    #[test]
    fn reports_missing_unexpected_and_shape_mismatches_in_sorted_order() {
        let device = Device::Cpu;
        let reference = tensor_map([
            ("missing", Tensor::zeros(1, DType::F32, &device).unwrap()),
            ("shape", Tensor::zeros(2, DType::F32, &device).unwrap()),
        ]);
        let actual = tensor_map([
            ("shape", Tensor::zeros(3, DType::F32, &device).unwrap()),
            ("unexpected", Tensor::zeros(1, DType::F32, &device).unwrap()),
        ]);
        let report = compare_tensor_maps(&reference, &actual, ParityTolerance::default()).unwrap();

        assert!(!report.passed);
        assert_eq!(report.missing_tensors, ["missing"]);
        assert_eq!(report.unexpected_tensors, ["unexpected"]);
        assert_eq!(report.matched_tensor_count, 1);
        assert_eq!(report.compared_tensor_count, 0);
        assert_eq!(report.failed_tensor_count, 3);
        assert_eq!(
            report
                .tensors
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            ["missing", "shape", "unexpected"]
        );
        assert_eq!(report.tensors[1].status, TensorParityStatus::ShapeMismatch);
        assert_eq!(report.tensors[1].reference_shape.as_deref(), Some(&[2][..]));
        assert_eq!(report.tensors[1].actual_shape.as_deref(), Some(&[3][..]));
    }

    #[test]
    fn safetensors_dtype_mismatch_fails_before_value_conversion() {
        let directory = tempdir().unwrap();
        let reference_path = directory.path().join("reference.safetensors");
        let actual_path = directory.path().join("actual.safetensors");
        safetensors::save(
            &HashMap::from([(
                "value".to_owned(),
                Tensor::new(&[1.0f32], &Device::Cpu).unwrap(),
            )]),
            &reference_path,
        )
        .unwrap();
        safetensors::save(
            &HashMap::from([(
                "value".to_owned(),
                Tensor::new(&[1.0f64], &Device::Cpu).unwrap(),
            )]),
            &actual_path,
        )
        .unwrap();

        let report =
            compare_safetensors(&reference_path, &actual_path, ParityTolerance::default()).unwrap();

        assert!(!report.passed);
        assert_eq!(report.compared_tensor_count, 0);
        assert_eq!(report.compared_elements, 0);
        assert_eq!(report.mismatched_elements, 0);
        assert_eq!(report.failed_tensor_count, 1);
        assert_eq!(report.tensors[0].status, TensorParityStatus::DtypeMismatch);
        assert_eq!(report.tensors[0].reference_dtype.as_deref(), Some("f32"));
        assert_eq!(report.tensors[0].actual_dtype.as_deref(), Some("f64"));
    }

    #[test]
    fn safetensors_and_candle_fp8_dtype_names_use_one_canonical_spelling() {
        assert_eq!(
            safetensors_dtype_name(Dtype::F8_E4M3),
            DType::F8E4M3.as_str()
        );
        assert_eq!(
            safetensors_dtype_name(Dtype::F8_E8M0),
            DType::F8E8M0.as_str()
        );
        assert_eq!(
            safetensors_dtype_name(Dtype::F6_E2M3),
            DType::F6E2M3.as_str()
        );
        assert_eq!(
            safetensors_dtype_name(Dtype::F6_E3M2),
            DType::F6E3M2.as_str()
        );
    }

    #[test]
    fn can_accept_extra_actual_metadata_tensors() {
        let device = Device::Cpu;
        let reference = tensor_map([("value", Tensor::zeros(1, DType::F32, &device).unwrap())]);
        let actual = tensor_map([
            ("value", Tensor::zeros(1, DType::F32, &device).unwrap()),
            ("metadata", Tensor::zeros(1, DType::U32, &device).unwrap()),
        ]);
        let strict = compare_tensor_maps(&reference, &actual, ParityTolerance::default()).unwrap();
        assert!(!strict.passed);
        let allowed = strict.allow_unexpected();
        assert!(allowed.passed);
        assert!(allowed.unexpected_allowed);
        assert_eq!(allowed.failed_tensor_count, 0);
        assert_eq!(allowed.unexpected_tensors, ["metadata"]);
    }

    #[test]
    fn nan_fails_but_equal_signed_infinities_pass() {
        let device = Device::Cpu;
        let reference = tensor_map([(
            "non_finite",
            Tensor::from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 3, &device).unwrap(),
        )]);
        let actual = tensor_map([(
            "non_finite",
            Tensor::from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 3, &device).unwrap(),
        )]);
        let report = compare_tensor_maps(&reference, &actual, ParityTolerance::default()).unwrap();
        let tensor = &report.tensors[0];

        assert!(!report.passed);
        assert_eq!(tensor.non_finite_elements, 3);
        assert_eq!(tensor.non_finite_mismatches, 1);
        assert_eq!(tensor.mismatched_elements, 1);
        assert_eq!(tensor.max_abs, None);
        assert_eq!(tensor.max_rel, None);
        assert_eq!(tensor.rmse, None);
    }

    #[test]
    fn serializes_unbounded_metrics_as_valid_json_nulls() {
        let device = Device::Cpu;
        let reference = tensor_map([("zero", Tensor::from_slice(&[0.0f64], 1, &device).unwrap())]);
        let actual = tensor_map([("zero", Tensor::from_slice(&[1.0f64], 1, &device).unwrap())]);
        let report = compare_tensor_maps(&reference, &actual, ParityTolerance::default()).unwrap();
        let encoded = report.to_pretty_json().unwrap();
        let decoded: Value = serde_json::from_str(&encoded).unwrap();

        assert!(decoded["tensors"][0]["max_rel"].is_null());
        assert_eq!(decoded["tensors"][0]["max_abs"], 1.0);
        assert_eq!(decoded["passed"], false);
    }

    #[test]
    fn compares_safetensors_files_and_cross_compares_a_tensor_map() {
        let directory = tempdir().unwrap();
        let reference_path = directory.path().join("reference.safetensors");
        let actual_path = directory.path().join("actual.safetensors");
        let device = Device::Cpu;
        let reference = tensor_map([
            ("a", Tensor::from_slice(&[1.0f32, 2.0], 2, &device).unwrap()),
            ("b", Tensor::from_slice(&[3.0f32], 1, &device).unwrap()),
        ]);
        let actual = tensor_map([
            (
                "a",
                Tensor::from_slice(&[1.0f32, 2.001], 2, &device).unwrap(),
            ),
            ("b", Tensor::from_slice(&[3.0f32], 1, &device).unwrap()),
        ]);
        let reference_file = reference
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect::<HashMap<_, _>>();
        let actual_file = actual
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect::<HashMap<_, _>>();
        safetensors::save(&reference_file, &reference_path).unwrap();
        safetensors::save(&actual_file, &actual_path).unwrap();
        let tolerance = ParityTolerance::new(0.002, 0.0).unwrap();

        let file_report = compare_safetensors(&reference_path, &actual_path, tolerance).unwrap();
        let map_report =
            compare_safetensors_to_tensor_map(&reference_path, &actual, tolerance).unwrap();

        assert!(file_report.passed);
        assert!(map_report.passed);
        assert_eq!(file_report.compared_tensor_count, 2);
        assert_eq!(map_report.compared_elements, 3);
    }

    #[test]
    fn rejects_invalid_tolerances() {
        assert!(ParityTolerance::new(-1.0, 0.0).is_err());
        assert!(ParityTolerance::new(0.0, f64::NAN).is_err());
        let device = Device::Cpu;
        let values = tensor_map([("x", Tensor::zeros(1, DType::F32, &device).unwrap())]);
        assert!(
            compare_tensor_maps(
                &values,
                &values,
                ParityTolerance {
                    atol: f64::INFINITY,
                    rtol: 0.0,
                },
            )
            .is_err()
        );
    }
}
