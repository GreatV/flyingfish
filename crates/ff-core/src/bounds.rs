use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBudget {
    #[serde(deserialize_with = "crate::required_option")]
    pub max_host_bytes: Option<u64>,
    #[serde(deserialize_with = "crate::required_option")]
    pub max_device_bytes: Option<u64>,
}

impl ResourceBudget {
    pub fn check_peaks(self, peak_host_bytes: u64, peak_device_bytes: u64) -> BudgetReport {
        let mut violations = Vec::with_capacity(2);
        if let Some(limit_bytes) = self.max_host_bytes
            && peak_host_bytes > limit_bytes
        {
            violations.push(BudgetViolation::new(
                ResourceDomain::Host,
                peak_host_bytes,
                limit_bytes,
            ));
        }
        if let Some(limit_bytes) = self.max_device_bytes
            && peak_device_bytes > limit_bytes
        {
            violations.push(BudgetViolation::new(
                ResourceDomain::Device,
                peak_device_bytes,
                limit_bytes,
            ));
        }
        BudgetReport {
            within_budget: violations.is_empty(),
            violations,
        }
    }

    pub fn validate_peaks(self, peak_host_bytes: u64, peak_device_bytes: u64) -> Result<()> {
        let report = self.check_peaks(peak_host_bytes, peak_device_bytes);
        if report.within_budget {
            Ok(())
        } else {
            bail!(report)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceDomain {
    Host,
    Device,
}

impl fmt::Display for ResourceDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => formatter.write_str("host"),
            Self::Device => formatter.write_str("device"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetViolation {
    pub domain: ResourceDomain,
    pub estimated_bytes: u64,
    pub limit_bytes: u64,
    pub excess_bytes: u64,
}

impl BudgetViolation {
    fn new(domain: ResourceDomain, estimated_bytes: u64, limit_bytes: u64) -> Self {
        Self {
            domain,
            estimated_bytes,
            limit_bytes,
            excess_bytes: estimated_bytes - limit_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetReport {
    pub within_budget: bool,
    pub violations: Vec<BudgetViolation>,
}

impl fmt::Display for BudgetReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.within_budget {
            return formatter.write_str("resource estimate is within budget");
        }
        formatter.write_str("resource budget exceeded: ")?;
        for (index, violation) in self.violations.iter().enumerate() {
            if index > 0 {
                formatter.write_str(", ")?;
            }
            write!(
                formatter,
                "{} needs {} but limit is {} ({} over)",
                violation.domain,
                format_bytes(violation.estimated_bytes),
                format_bytes(violation.limit_bytes),
                format_bytes(violation.excess_bytes)
            )?;
        }
        Ok(())
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1_024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1_024. && unit + 1 < UNITS.len() {
        value /= 1_024.;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_peaks_without_t2va_geometry() {
        let budget = ResourceBudget {
            max_host_bytes: Some(100),
            max_device_bytes: Some(50),
        };
        assert!(budget.check_peaks(100, 50).within_budget);
        let report = budget.check_peaks(101, 10);
        assert!(!report.within_budget);
        assert_eq!(report.violations[0].domain, ResourceDomain::Host);
        assert_eq!(report.violations[0].excess_bytes, 1);
    }
}
