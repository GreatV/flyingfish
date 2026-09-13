use super::*;

pub(super) fn run_inspect(command: Command) -> Result<()> {
    let Command::Inspect {
        checkpoint,
        weights: weight_args,
        verify,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for inspect");
    };
    let weights = ModelWeights::open(
        &checkpoint,
        weight_args.weight_source,
        weight_args.cache_policy()?,
    )?;
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

pub(super) fn run_tensor(command: Command) -> Result<()> {
    let Command::Tensor {
        checkpoint,
        name,
        device,
        weights: weight_args,
    } = command
    else {
        bail!("internal CLI dispatch mismatch for tensor");
    };
    let weights = ModelWeights::open(
        checkpoint,
        weight_args.weight_source,
        weight_args.cache_policy()?,
    )?;
    let metadata = weights.metadata(&name)?;
    println!(
        "{}: dtype={}, shape={:?}, bytes={}, shard={}",
        metadata.name,
        metadata.dtype,
        metadata.shape,
        metadata.bytes,
        metadata.shard.display()
    );
    let device = parse_device(&device)?;
    let tensor = weights.load(&name, &device)?;
    println!("materialized on {:?}: {:?}", device, tensor.shape());
    Ok(())
}
