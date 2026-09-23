use anyhow::{Context, Result, bail};
use candle_core::Device;

pub(crate) fn parse_device(value: &str) -> Result<Device> {
    if value == "cpu" {
        return Ok(Device::Cpu);
    }
    if value == "auto" {
        return Device::cuda_if_available(0).context("failed to initialize automatic device");
    }
    if let Some(list) = value.strip_prefix("cuda:") {
        let ordinal = list
            .split(',')
            .next()
            .context("cuda device list is empty")?
            .parse::<usize>()
            .context("invalid CUDA device ordinal")?;
        return Device::new_cuda(ordinal).context("failed to initialize CUDA device");
    }
    if let Some(ordinal) = value.strip_prefix("metal:") {
        let ordinal = ordinal
            .parse::<usize>()
            .context("invalid Metal device ordinal")?;
        return Device::new_metal(ordinal).context("failed to initialize Metal device");
    }
    bail!("unknown device {value:?}; use cpu, auto, cuda:N, or metal:N")
}

/// The ordinals behind a `cuda:N[,M...]` device list, deduplicated in order.
/// `cpu` and `auto` carry no ordinals; a list is the multi-device generation
/// request.
/// Like [`parse_device`], but a comma-separated `cuda:N[,M...]` list is
/// rejected: for commands that run exactly one device pipeline, silently
/// dropping the trailing devices would run less than the operator asked for.
pub(crate) fn parse_device_single(value: &str) -> Result<Device> {
    anyhow::ensure!(
        parse_device_ordinals(value)
            .map(|ordinals| ordinals.len() <= 1)
            .unwrap_or(true),
        "comma-separated CUDA device lists are only accepted by commands that run one \
         pipeline per device"
    );
    parse_device(value)
}

pub(crate) fn parse_device_ordinals(value: &str) -> Result<Vec<usize>> {
    if value == "cpu" || value == "auto" {
        return Ok(Vec::new());
    }
    // metal: and unknown backends carry no ordinals; parse_device owns them.
    let Some(list) = value.strip_prefix("cuda:") else {
        return Ok(Vec::new());
    };
    let mut ordinals = Vec::new();
    for token in list.split(',') {
        let ordinal = token
            .trim()
            .parse::<usize>()
            .with_context(|| format!("invalid CUDA device ordinal {token:?}"))?;
        if !ordinals.contains(&ordinal) {
            ordinals.push(ordinal);
        }
    }
    anyhow::ensure!(!ordinals.is_empty(), "cuda device list is empty");
    Ok(ordinals)
}
