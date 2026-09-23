use super::WeightCacheArgs;
use super::device_parse::parse_device_single;
use anyhow::Result;
use flyingfish::runtime::weights::ModelWeights;
use std::path::PathBuf;

pub(super) fn run_inspect(
    checkpoint: PathBuf,
    weights: WeightCacheArgs,
    verify: bool,
) -> Result<()> {
    let weights = ModelWeights::open(&checkpoint, weights.weight_source, weights.cache_policy()?)?;
    let inventory = weights.inventory();
    println!("checkpoint: {}", checkpoint.display());
    println!("index: {}", weights.index_path().display());
    println!("tensors: {}", inventory.tensors);
    println!("shards: {}", inventory.shards);
    if let Some(bytes) = inventory.indexed_bytes {
        println!(
            "weight bytes: {bytes} ({:.2} GiB)",
            bytes as f64 / 1024f64.powi(3)
        );
    }
    if verify {
        let report = weights.verify()?;
        println!(
            "verified: {} tensors across {} shards",
            report.checked_tensors, report.checked_shards
        );
    }
    Ok(())
}

pub(super) fn run_tensor(
    checkpoint: PathBuf,
    name: String,
    device: String,
    weights: WeightCacheArgs,
) -> Result<()> {
    let weights = ModelWeights::open(checkpoint, weights.weight_source, weights.cache_policy()?)?;
    let metadata = weights.metadata(&name)?;
    println!(
        "{}: dtype={}, shape={:?}, bytes={}, shard={}",
        metadata.name,
        metadata.dtype,
        metadata.shape,
        metadata.bytes,
        metadata.shard.display()
    );
    let device = parse_device_single(&device)?;
    let tensor = weights.load(&name, &device)?;
    println!("materialized on {:?}: {:?}", device, tensor.shape());
    Ok(())
}
