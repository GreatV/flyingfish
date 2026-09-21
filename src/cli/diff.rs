use anyhow::Result;
use flyingfish::runtime::parity::{ParityTolerance, compare_safetensors};
use std::path::PathBuf;

pub(super) fn run_compare_tensors(
    reference: PathBuf,
    actual: PathBuf,
    atol: f64,
    rtol: f64,
    allow_unexpected: bool,
) -> Result<()> {
    let mut report = compare_safetensors(&reference, &actual, ParityTolerance::new(atol, rtol)?)?;
    if allow_unexpected {
        report = report.allow_unexpected();
    }
    println!("{}", report.to_pretty_json()?);
    anyhow::ensure!(
        report.passed,
        "tensor parity failed: {} tensors and {} elements outside tolerance",
        report.failed_tensor_count,
        report.mismatched_elements
    );
    Ok(())
}
