use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    pub arch: u32,
    pub sms: usize,
    pub l2_bytes: usize,
    pub warp: usize,
    pub max_threads: u32,
    pub shared_bytes: usize,
    pub sm_shared_bytes: usize,
    pub cooperative: bool,
    pub optin_shared_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub device_uuid: String,
    pub driver_version: String,
    pub binary_version: String,
    pub cublaslt_version: usize,
}

impl CacheKey {
    pub fn tag(&self) -> String {
        format!(
            "{}-{}-{}-lt{}",
            self.device_uuid, self.driver_version, self.binary_version, self.cublaslt_version
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum Selection {
    Calibrated { key: CacheKey, cached: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPlan {
    pub chunk: usize,
    pub qpack: usize,
    pub threads: u32,
    pub merge_threads: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiImpl {
    V1,
    Tcmqa,
    TcmqaW,
    TcmqaWs,
    TcmqaShort,
    TcmqaWShort,
}

impl MultiImpl {
    pub fn phase1(self) -> Self {
        match self {
            Self::TcmqaShort => Self::Tcmqa,
            Self::TcmqaWShort => Self::TcmqaW,
            other => other,
        }
    }

    pub fn short_merge(self) -> bool {
        matches!(self, Self::TcmqaShort | Self::TcmqaWShort)
    }

    /// Query rows covered by one phase-1 tile (blockIdx.z).
    pub fn phase1_tile_rows(self) -> usize {
        match self {
            MultiImpl::V1 => 64,
            MultiImpl::Tcmqa | MultiImpl::TcmqaShort => 64,
            MultiImpl::TcmqaW | MultiImpl::TcmqaWShort | MultiImpl::TcmqaWs => 128,
        }
    }

    /// Dynamic shared memory the phase-1 kernel is launched with.
    pub fn phase1_smem_bytes(self) -> usize {
        match self {
            MultiImpl::V1 => 0,
            MultiImpl::Tcmqa | MultiImpl::TcmqaShort => 35328,
            MultiImpl::TcmqaW | MultiImpl::TcmqaWShort => 52736,
            MultiImpl::TcmqaWs => 50176,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiPlan {
    pub implementation: MultiImpl,
    pub launch: AttentionPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiShape {
    pub capacity: usize,
    pub rows: usize,
    pub causal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LinearShape {
    pub rows: usize,
    pub output: usize,
    pub input: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearImpl {
    Cublas,
    CublasLt(LtAlgorithm),
    Skinny,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LtAlgorithm {
    pub id: i32,
    pub tile: i32,
    pub split_k: i32,
    pub reduction: i32,
    pub swizzle: i32,
    pub custom: i32,
    pub stages: i32,
    pub inner: u16,
    pub cluster: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearChoice {
    pub shape: LinearShape,
    pub implementation: LinearImpl,
    pub key: CacheKey,
    pub median_us: f64,
}

#[derive(Debug, Serialize)]
pub struct Setup {
    pub device: DeviceInfo,
    pub selection: Selection,
    pub attention: AttentionPlan,
}

impl Setup {
    pub fn select(
        info: DeviceInfo,
        c: &Config,
        capacity: usize,
        choose: impl FnOnce(&DeviceInfo, usize) -> Result<(AttentionPlan, Selection)>,
    ) -> Result<Self> {
        let (attention, selection) = choose(&info, capacity)?;
        let chunk = attention.chunk;
        ensure!(
            capacity > 0 && capacity < i32::MAX as usize && c.head_dim > 0,
            "invalid backend capacity or head dimension"
        );
        ensure!(info.warp == 32, "CUDA attention requires a 32-thread warp");
        ensure!(
            [1, 2, 4, 8].contains(&attention.qpack),
            "attention pack must be 1, 2, 4 or 8"
        );
        ensure!(
            attention.threads >= 32
                && attention.threads <= info.max_threads
                && attention.threads.is_multiple_of(32),
            "attention threads exceed device limits or are not a multiple of 32"
        );
        ensure!(
            attention.merge_threads >= 128
                && attention.merge_threads <= info.max_threads
                && attention.merge_threads.is_power_of_two(),
            "attention merge threads exceed device limits or are not a power of two"
        );
        ensure!(
            chunk > 0 && chunk <= i32::MAX as usize - capacity,
            "attention chunk and capacity must fit int"
        );
        ensure!(
            capacity <= i32::MAX as usize / c.head_dim,
            "attention KV stride exceeds int range"
        );
        let chunks = capacity.div_ceil(chunk);
        let first = attention.qpack * (attention.threads as usize / info.warp) * 130 * 4;
        let second = (2 * chunks + attention.merge_threads as usize) * 4;
        ensure!(
            first.max(second) <= info.shared_bytes.min(48 * 1024),
            "attention shared memory exceeds device/kernel limit"
        );
        Ok(Self {
            device: info,
            selection,
            attention,
        })
    }
}

pub const ATTENTION_CAPACITIES: [usize; 5] = [2048, 16384, 49152, 98304, 131072];

pub const ATTENTION_LENGTHS: [usize; 5] = [1024, 8192, 32768, 65536, 130752];

pub fn attention_bucket(capacity: usize) -> Result<usize> {
    ensure!(
        capacity > 0 && capacity <= ATTENTION_CAPACITIES[ATTENTION_CAPACITIES.len() - 1],
        "attention capacity {capacity} is outside supported buckets"
    );
    Ok(ATTENTION_CAPACITIES.partition_point(|&upper| upper < capacity))
}

pub fn attention_lengths(capacity: usize, rows: usize) -> Result<[usize; 4]> {
    let bucket = attention_bucket(capacity)?;
    let maximum = capacity
        .checked_sub(rows)
        .context("attention query rows exceed capacity")?;
    let base = ATTENTION_LENGTHS[bucket].min(maximum);
    Ok([0, 1, 128, 256].map(|growth| (base + growth).min(maximum)))
}

pub fn attention_chunks(bucket: usize) -> &'static [usize] {
    if bucket < 3 {
        &[128, 256, 512, 1024]
    } else {
        &[256, 512, 1024, 2048, 4096]
    }
}

#[cfg(test)]
mod long_tests {
    use super::*;

    #[test]
    fn long_buckets_cover_capacity_and_growth() -> Result<()> {
        let mut lower = 1;
        for (bucket, &upper) in ATTENTION_CAPACITIES.iter().enumerate() {
            assert!(lower <= upper);
            assert_eq!(attention_bucket(lower)?, bucket);
            assert_eq!(attention_bucket(upper)?, bucket);
            lower = upper + 1;
        }
        for (capacity, expected) in [
            (2048, 0),
            (2049, 1),
            (16384, 1),
            (16385, 2),
            (33024, 2),
            (66048, 3),
            (131072, 4),
        ] {
            assert_eq!(attention_bucket(capacity)?, expected);
        }
        assert!(attention_bucket(0).is_err());
        assert!(attention_bucket(131073).is_err());
        for capacity in [
            1, 7, 64, 257, 1023, 1024, 1280, 1312, 2048, 2049, 16384, 16385, 49152, 49153, 98304,
            98305, 131072,
        ] {
            for rows in [0, 7, 8, 16, 64] {
                if capacity < rows {
                    assert!(attention_lengths(capacity, rows).is_err());
                    continue;
                }
                let lengths = attention_lengths(capacity, rows)?;
                assert!(lengths.iter().all(|&length| length + rows <= capacity));
                assert_eq!(
                    lengths[0],
                    ATTENTION_LENGTHS[attention_bucket(capacity)?].min(capacity - rows)
                );
                assert!(lengths.windows(2).all(|pair| pair[0] <= pair[1]));
            }
        }
        for length in ATTENTION_LENGTHS {
            assert!(length + 256 + 64 <= 131072);
        }
        for capacity in [66048usize, 131072] {
            assert!(capacity * 128 < i32::MAX as usize);
            for &chunk in attention_chunks(attention_bucket(capacity)?) {
                assert!((2 * capacity.div_ceil(chunk) + 256) * 4 <= 49152);
            }
        }
        Ok(())
    }

    #[test]
    fn merge_threads_must_be_a_power_of_two() -> Result<()> {
        let c: Config = serde_json::from_value(serde_json::json!({
            "model_type": "llama", "hidden_size": 2048, "intermediate_size": 8192,
            "num_hidden_layers": 1, "num_attention_heads": 16, "num_key_value_heads": 2,
            "head_dim": 128, "vocab_size": 1024, "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "rope_scaling": null,
            "hidden_act": "silu", "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "eos_token_id": [1]
        }))?;
        let info = DeviceInfo {
            name: String::new(),
            arch: 89,
            sms: 128,
            l2_bytes: 72 << 20,
            warp: 32,
            max_threads: 1024,
            shared_bytes: 48 << 10,
            sm_shared_bytes: 100 << 10,
            cooperative: true,
            optin_shared_bytes: 99 << 10,
        };
        for (merge_threads, valid) in [
            (128, true),
            (256, true),
            (512, true),
            (1024, true),
            (384, false),
            (640, false),
            (768, false),
            (896, false),
        ] {
            let plan = AttentionPlan {
                chunk: 512,
                qpack: 2,
                threads: 128,
                merge_threads,
            };
            let selected = Setup::select(info.clone(), &c, 2048, |_, _| {
                Ok((
                    plan,
                    Selection::Calibrated {
                        key: CacheKey {
                            device_uuid: String::new(),
                            driver_version: String::new(),
                            binary_version: String::new(),
                            cublaslt_version: 130201,
                        },
                        cached: true,
                    },
                ))
            });
            assert_eq!(selected.is_ok(), valid, "merge_threads={merge_threads}");
        }
        Ok(())
    }
}
