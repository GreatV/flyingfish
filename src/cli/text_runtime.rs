use anyhow::{Context, Result, bail};
use candle_core::Device;

/// The int4 text adapters drive CUDA through cudarc contexts.
pub(crate) enum TextDevice {
    Cpu,
    Cuda(Vec<usize>),
}

pub(crate) fn resolve_text_device(value: &str) -> Result<(TextDevice, bool)> {
    let selected = match value {
        "cpu" => TextDevice::Cpu,
        "auto" => {
            if Device::cuda_if_available(0)
                .is_ok_and(|device| device.is_cuda() && ptx_floor_supported(&device))
            {
                TextDevice::Cuda(vec![0])
            } else {
                TextDevice::Cpu
            }
        }
        _ if value.starts_with("cuda:") => {
            let tail = &value["cuda:".len()..];
            let ordinals: Result<Vec<usize>> = tail
                .split(',')
                .map(|segment| {
                    let trimmed = segment.trim();
                    anyhow::ensure!(
                        !trimmed.is_empty(),
                        "empty ordinal in device list {value:?}"
                    );
                    trimmed.parse::<usize>().map_err(|_| {
                        anyhow::anyhow!(
                            "malformed ordinal {trimmed:?} in device list {value:?}; \
                             use cpu, auto, or cuda:N[,M...]"
                        )
                    })
                })
                .collect();
            TextDevice::Cuda(ordinals?)
        }
        _ => bail!("unknown device {value:?}; use cpu, auto, or cuda:N[,M...]"),
    };
    #[cfg(not(feature = "cuda"))]
    if matches!(selected, TextDevice::Cuda(_)) {
        bail!("CUDA decoding requires a binary built with --features cuda");
    }
    Ok((selected, value == "auto"))
}

/// The int4 text adapters' kernels ship as compute_80 PTX; older GPUs cannot
/// JIT it, so `auto` must not select them.
#[cfg(feature = "cuda")]
fn ptx_floor_supported(device: &Device) -> bool {
    device
        .as_cuda_device()
        .ok()
        .and_then(|cuda| cuda.cuda_stream().context().compute_capability().ok())
        .is_some_and(|(major, _)| major >= 8)
}

#[cfg(not(feature = "cuda"))]
fn ptx_floor_supported(_: &Device) -> bool {
    false
}

pub(crate) fn greedy_token(logits: &[f32]) -> Result<u32> {
    anyhow::ensure!(
        logits.iter().all(|value| value.is_finite()),
        "model produced non-finite logits"
    );
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
        .map(|(index, _)| index as u32)
        .context("model produced no logits")
}
