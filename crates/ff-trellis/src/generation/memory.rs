//! Stage-local workspace estimates for streamed TRELLIS models.
use super::*;
use crate::config::{AttentionMode, SLatCoderArgs, SLatFlowArgs};
use ff_core::weights::ModelWeights;

fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |n, &v| {
        n.checked_mul(v).context("TRELLIS workspace size overflow")
    })
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |n, &v| {
        n.checked_add(v).context("TRELLIS workspace sum overflow")
    })
}

fn weight_load(weights: &ModelWeights, device: &Device) -> Result<u64> {
    let largest = weights.tensor_names().try_fold(0, |largest, name| {
        Ok::<_, anyhow::Error>(largest.max(weights.produced_bytes(name, device)?))
    })?;
    product(&[3, largest])
}

fn transformer(
    rows: u64,
    keys: u64,
    width: u64,
    heads: u64,
    ratio: f64,
    chunk: usize,
) -> Result<u64> {
    anyhow::ensure!(ratio.is_finite() && ratio > 0., "invalid TRELLIS MLP ratio");
    let intermediate = (width as f64 * ratio).ceil();
    anyhow::ensure!(intermediate < u64::MAX as f64, "TRELLIS MLP size overflow");
    sum(&[
        product(&[16, rows, width, 4])?,
        product(&[4, keys, width, 4])?,
        product(&[2, rows, intermediate as u64, 4])?,
        product(&[3, heads, rows.min(chunk as u64), keys, 4])?,
    ])
}

pub(super) fn flow(
    args: &SLatFlowArgs,
    grid: &Grid,
    condition_rows: usize,
    weights: &ModelWeights,
    device: &Device,
    options: &GenerationOptions,
) -> Result<u64> {
    let mut coords = grid.coords.clone();
    let mut levels = vec![coords.len() as u64];
    for _ in &args.io_block_channels {
        coords = coords
            .iter()
            .map(|c| [c[0], c[1] / 2, c[2] / 2, c[3] / 2])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        levels.push(coords.len() as u64);
    }
    let rows = *levels.last().unwrap();
    let width = args.model_channels as u64;
    let io_width = args.io_block_channels.iter().copied().max().unwrap_or(0) as u64;
    let io_rows = grid.coords.len() as u64;
    sum(&[
        transformer(
            rows,
            rows.max(condition_rows as u64),
            width,
            args.num_heads as u64,
            args.mlp_ratio,
            options.query_chunk,
        )?,
        product(&[sum(&levels)?, 32, 4])?,
        product(&[3, rows, width, 4])?,
        product(&[
            sum(&levels)?,
            io_width.max(width),
            args.num_io_res_blocks as u64,
            4,
        ])?,
        product(&[12, io_rows, io_width, 4])?,
        if io_width == 0 {
            0
        } else {
            product(&[
                2,
                io_rows.min(options.voxel_chunk as u64),
                27,
                io_width.max(width),
                4,
            ])?
        },
        product(&[
            8,
            io_rows,
            args.in_channels.max(args.out_channels) as u64,
            4,
        ])?,
        weight_load(weights, device)?,
    ])
}

pub(super) fn gaussian(
    args: &SLatCoderArgs,
    grid: &Grid,
    weights: &ModelWeights,
    device: &Device,
    query_chunk: usize,
) -> Result<u64> {
    let rows = grid.coords.len() as u64;
    let keys = match args.attn_mode {
        AttentionMode::Full => rows,
        AttentionMode::Swin => {
            [0, args.window_size / 2]
                .into_iter()
                .try_fold(0, |largest, offset| {
                    Ok::<_, anyhow::Error>(
                        largest.max(
                            grid.partitions(Some((args.window_size, offset)))?
                                .iter()
                                .map(|group| group.len() as u64)
                                .max()
                                .unwrap_or(0),
                        ),
                    )
                })?
        }
    };
    let output = *weights
        .raw_tensor_metadata("out_layer.weight")?
        .shape
        .first()
        .context("Gaussian output projection has no output dimension")? as u64;
    sum(&[
        transformer(
            rows,
            keys,
            args.model_channels as u64,
            args.num_heads as u64,
            args.mlp_ratio,
            query_chunk,
        )?,
        product(&[3, rows, args.model_channels as u64, 4])?,
        product(&[2, rows, output, 4])?,
        product(&[4, rows, 4])?,
        weight_load(weights, device)?,
    ])
}

pub(crate) fn flow2(
    config: &serde_json::Value,
    rows: usize,
    condition_rows: usize,
    weights: &ModelWeights,
    device: &Device,
    chunk: usize,
) -> Result<u64> {
    let n = |key: &str| {
        config[key]
            .as_u64()
            .with_context(|| format!("missing flow {key}"))
    };
    let width = n("model_channels")?;
    let ratio = config["mlp_ratio"]
        .as_f64()
        .context("missing flow MLP ratio")?;
    sum(&[
        transformer(
            rows as u64,
            rows.max(condition_rows) as u64,
            width,
            n("num_heads")?,
            ratio,
            chunk,
        )?,
        product(&[4, rows as u64, width, 4])?,
        product(&[
            12,
            rows as u64,
            n("in_channels")?.max(n("out_channels")?),
            4,
        ])?,
        product(&[2, rows as u64, 4])?,
        weight_load(weights, device)?,
    ])
}

pub(crate) fn vae_blocks(
    weights: &ModelWeights,
    device: &Device,
    stage: usize,
    rows: usize,
    width: usize,
    chunk: usize,
) -> Result<u64> {
    let prefix = format!("blocks.{stage}.");
    let mut intermediate = width as u64;
    for name in weights
        .tensor_names()
        .filter(|name| name.starts_with(&prefix) && name.ends_with(".mlp.0.weight"))
    {
        intermediate = intermediate.max(
            *weights
                .raw_tensor_metadata(name)?
                .shape
                .first()
                .context("VAE MLP has no output dimension")? as u64,
        );
    }
    let output = *weights
        .raw_tensor_metadata("output_layer.weight")?
        .shape
        .first()
        .context("VAE output projection has no output dimension")? as u64;
    sum(&[
        product(&[2, rows as u64, output, 4])?,
        product(&[12, rows as u64, width as u64, 4])?,
        product(&[3, rows as u64, intermediate, 4])?,
        product(&[2, rows.min(chunk) as u64, 27, width as u64, 4])?,
        product(&[rows as u64, 27, 4])?,
        weight_load(weights, device)?,
    ])
}

pub(crate) fn vae_subdivision(
    weights: &ModelWeights,
    device: &Device,
    parent_rows: usize,
    child_rows: usize,
    input: usize,
    output: usize,
    chunk: usize,
) -> Result<u64> {
    sum(&[
        product(&[8, parent_rows as u64, input as u64, 4])?,
        product(&[2, parent_rows as u64, 8, output as u64, 4])?,
        product(&[10, child_rows as u64, output as u64, 4])?,
        product(&[
            2,
            parent_rows.max(child_rows).min(chunk) as u64,
            27,
            input.max(output) as u64,
            4,
        ])?,
        product(&[sum(&[parent_rows as u64, child_rows as u64])?, 28, 4])?,
        weight_load(weights, device)?,
    ])
}

pub(crate) fn prepare(
    cache: &DeviceCache,
    device: &Device,
    stage: &str,
    tensor_bytes: u64,
) -> Result<()> {
    if !device.is_cuda() || !cache.policy().is_enabled() {
        return Ok(());
    }
    device.synchronize()?;
    let free = ff_core::probe::ResourceSnapshot::capture(Some(device))
        .device_free_memory_bytes
        .unwrap_or(0);
    let reserve = sum(&[tensor_bytes, 1 << 30])?;
    let capacity = cache
        .stats()
        .resident_bytes
        .saturating_add(free)
        .saturating_sub(reserve);
    cache.set_capacity_bytes(capacity);
    eprintln!(
        "TRELLIS {stage}: {} MiB tensor/workspace reserve, {} MiB cache ceiling",
        reserve / (1 << 20),
        cache.stats().max_bytes / (1 << 20)
    );
    Ok(())
}
