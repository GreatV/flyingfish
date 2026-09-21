use anyhow::{Context, Result, bail};
use candle_core::Device;

pub(crate) fn parse_device(value: &str) -> Result<Device> {
    if value == "cpu" {
        return Ok(Device::Cpu);
    }
    if value == "auto" {
        return Device::cuda_if_available(0).context("failed to initialize automatic device");
    }
    if let Some(ordinal) = value.strip_prefix("cuda:") {
        let ordinal = ordinal
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
