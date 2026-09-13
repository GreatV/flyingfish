use super::*;
use candle_core::Device;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize_to_file},
};
use std::fs;

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let data = vec![0_u8; 4096];
    for name in ["a", "b"] {
        serialize_to_file(
            [(name, TensorView::new(Dtype::U8, vec![4096], &data).unwrap())],
            None,
            &root.path().join(format!("{name}.safetensors")),
        )
        .unwrap();
    }
    fs::write(root.path().join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":8192},"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#).unwrap();
    root
}

#[test]
fn reconfiguration_retains_metadata_and_is_forbidden_after_payload_access() {
    let root = fixture();
    let mut weights =
        ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    weights.cache_inventory().unwrap();
    let headers = weights.cache_stats().header_parses;
    weights
        .configure_unloaded_cache(
            WeightSource::Memory,
            CachePolicy::unbounded_units()
                .with_max_bytes(8192)
                .with_granularity(CacheGranularity::Tensor),
        )
        .unwrap();
    assert_eq!(weights.cache_stats().header_parses, headers);
    weights.load("a", &Device::Cpu).unwrap();
    assert!(
        weights
            .configure_unloaded_cache(WeightSource::Mmap, CachePolicy::new(1))
            .is_err()
    );
}

#[test]
fn payload_bound_evicts_while_actual_file_total_retains_all_shards() {
    let root = fixture();
    let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    let inventory = weights.cache_inventory().unwrap();
    let payload = inventory.total_bytes(CacheGranularity::Tensor).unwrap();
    let files = inventory.total_bytes(CacheGranularity::Shard).unwrap();
    assert!(files > payload);
    let stats = weights.cache_stats();
    assert_eq!(
        (
            stats.misses,
            stats.resident_bytes,
            stats.memory_source_reads
        ),
        (0, 0, 0)
    );
    for (bound, expected) in [(payload, (0, 4, 3)), (files, (2, 2, 0))] {
        let policy = CachePolicy::unbounded_units().with_max_bytes(bound);
        let weights = ModelWeights::open(root.path(), WeightSource::Memory, policy).unwrap();
        let estimate = estimate_cache_residency(
            &inventory,
            WeightSource::Memory,
            policy,
            CacheLoadLifetimes::SERIAL,
        )
        .unwrap();
        for name in ["a", "b", "a", "b"] {
            weights.load(name, &Device::Cpu).unwrap();
        }
        let stats = weights.cache_stats();
        assert_eq!((stats.hits, stats.misses, stats.evictions), expected);
        assert!(estimate.owned_weight_bytes >= stats.resident_bytes);
        assert_eq!(estimate.complete_set_fits, bound == files);
        if bound == files {
            assert_eq!(estimate.peak_storage_bytes, files);
        }
    }
}

#[test]
fn count_limit_remains_binding_and_tensor_units_exclude_headers() {
    let root = fixture();
    let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    let inv = weights.cache_inventory().unwrap();
    let bytes = inv.total_bytes(CacheGranularity::Shard).unwrap();
    let policy = CachePolicy::new(1).with_max_bytes(bytes);
    let estimate = estimate_cache_residency(
        &inv,
        WeightSource::Memory,
        policy,
        CacheLoadLifetimes::SERIAL,
    )
    .unwrap();
    assert!(!estimate.complete_set_fits);
    assert!(estimate.loading_bytes > 0);
    let weights = ModelWeights::open(root.path(), WeightSource::Memory, policy).unwrap();
    for name in ["a", "b", "a", "b"] {
        weights.load(name, &Device::Cpu).unwrap();
    }
    assert_eq!(weights.cache_stats().hits, 0);
    let policy = CachePolicy::unbounded_units()
        .with_max_bytes(8192)
        .with_granularity(CacheGranularity::Tensor);
    let estimate = estimate_cache_residency(
        &inv,
        WeightSource::Memory,
        policy,
        CacheLoadLifetimes::SERIAL,
    )
    .unwrap();
    assert!(estimate.complete_set_fits);
    assert_eq!(estimate.owned_weight_bytes, 8192);
    let weights = ModelWeights::open(root.path(), WeightSource::Memory, policy).unwrap();
    for name in ["a", "b", "a", "b"] {
        weights.load(name, &Device::Cpu).unwrap();
    }
    assert_eq!(weights.cache_stats().hits, 2);
}

#[test]
fn overlap_borrowed_storage_and_oversized_units_are_charged() {
    let root = fixture();
    let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    let inv = weights.cache_inventory().unwrap();
    let unit = inv.largest_unit_bytes(CacheGranularity::Shard);
    let tiny = CachePolicy::new(1).with_max_bytes(1);
    let serial =
        estimate_cache_residency(&inv, WeightSource::Mmap, tiny, CacheLoadLifetimes::SERIAL)
            .unwrap();
    assert_eq!(serial.peak_storage_bytes, 2 * unit);
    let complete = CachePolicy::unbounded_units();
    let parallel = estimate_cache_residency(
        &inv,
        WeightSource::Memory,
        complete,
        CacheLoadLifetimes {
            maximum_concurrent_loads: 2,
            externally_held_bytes: 123,
            additional_storage_bytes: 456,
        },
    )
    .unwrap();
    assert!(parallel.complete_set_fits);
    assert_eq!(parallel.loading_bytes, unit);
    assert_eq!(
        parallel.peak_storage_bytes,
        inv.total_bytes(CacheGranularity::Shard).unwrap() + unit + 579
    );
    assert!(
        estimate_cache_residency(
            &inv,
            WeightSource::Memory,
            complete,
            CacheLoadLifetimes {
                maximum_concurrent_loads: u64::MAX,
                ..CacheLoadLifetimes::SERIAL
            }
        )
        .is_err()
    );
    assert!(
        estimate_cache_residency(
            &inv,
            WeightSource::Memory,
            complete,
            CacheLoadLifetimes {
                maximum_concurrent_loads: 0,
                ..CacheLoadLifetimes::SERIAL
            }
        )
        .is_err()
    );
}

#[test]
fn subset_inventory_deduplicates_names_and_index_totals_are_verified() {
    let root = fixture();
    let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    let one = weights.cache_inventory_for(["a", "a"]).unwrap();
    assert_eq!(one.shards.len(), 1);
    assert_eq!(one.shards[0].selected_tensor_count, 1);
    assert!(one.shards[0].file_bytes > one.shards[0].selected_tensor_bytes);
    assert!(weights.cache_inventory_for(["missing"]).is_err());
    assert!(weights.cache_inventory_for([]).is_err());
    let index = root.path().join("model.safetensors.index.json");
    fs::write(
        &index,
        fs::read_to_string(&index).unwrap().replace("8192", "8191"),
    )
    .unwrap();
    let weights = ModelWeights::open(root.path(), WeightSource::Mmap, CachePolicy::new(1)).unwrap();
    assert!(
        weights
            .cache_inventory()
            .unwrap_err()
            .to_string()
            .contains("disagrees")
    );
}
