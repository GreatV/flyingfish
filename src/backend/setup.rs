use crate::{config::Config, prefill::Plan};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceInfo {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LinearImpl {
    Cublas,
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
        Plan::new(c, capacity, requested, free, total, alignment)
    }
}
