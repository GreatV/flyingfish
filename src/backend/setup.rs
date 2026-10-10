use crate::{config::Config, prefill::Plan};
use anyhow::{Result, ensure};
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
            MultiImpl::TcmqaW | MultiImpl::TcmqaWShort => 128,
        }
    }

    /// Dynamic shared memory the phase-1 kernel is launched with.
    pub fn phase1_smem_bytes(self) -> usize {
        match self {
            MultiImpl::V1 => 0,
            MultiImpl::Tcmqa | MultiImpl::TcmqaShort => 35328,
            MultiImpl::TcmqaW | MultiImpl::TcmqaWShort => 52736,
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
    CublasLt(u32),
    Skinny,
    Candidate(String),
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
                && attention.merge_threads.is_multiple_of(128),
            "attention merge threads exceed device limits or are not a multiple of 128"
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
    pub fn prefill(
        &self,
        c: &Config,
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
    ) -> Result<Plan> {
        Plan::new(c, capacity, requested, free, total, alignment, false)
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
}
