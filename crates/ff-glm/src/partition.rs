//! Serializable layer ownership and per-device execution contracts.
use crate::{
    admission::GlmLayerScope,
    execution_policy::{GlmExecutionBackend, GlmExecutionPolicy},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmRankPolicy {
    pub ordinal: usize,
    pub scope: GlmLayerScope,
    /// Static residency and expert-cache bounds apply only to this rank's scope.
    pub execution: GlmExecutionPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlmPartitionPolicy {
    pub schema_version: u32,
    /// Decode, then prefill transport and evidence for each adjacent-rank boundary.
    pub transports: Vec<[(GlmPartitionTransport, String); 2]>,
    pub ranks: Vec<GlmRankPolicy>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum GlmPartitionTransport {
    #[value(name = "peer")]
    SynchronizedCudaDeviceCopyV1,
    #[value(name = "host")]
    HostStagedCopyV1,
}

impl GlmPartitionPolicy {
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn select_transport(
        pair: [usize; 2],
        capability: [Option<bool>; 2],
        unified: bool,
        failures: [Option<String>; 2],
        timings: Option<(usize, [u64; 3], [u64; 3])>,
        requested: Option<GlmPartitionTransport>,
    ) -> Result<(GlmPartitionTransport, String)> {
        use GlmPartitionTransport::{
            HostStagedCopyV1 as Host, SynchronizedCudaDeviceCopyV1 as Peer,
        };
        let failure = if unified {
            Some(format!(
                "{}→{} unified host/device memory",
                pair[0], pair[1]
            ))
        } else {
            (0..2).find_map(|direction| {
                let from = pair[direction];
                let to = pair[1 - direction];
                failures[direction]
                    .as_ref()
                    .map(|reason| format!("{from}→{to} {reason}"))
            })
        };
        let topology = format!(
            "direct-peer capability {}→{} {:?}, {}→{} {:?}",
            pair[0], pair[1], capability[0], pair[1], pair[0], capability[1]
        );
        let override_note = if requested.is_some() { " override" } else { "" };
        if let Some(reason) = failure {
            ensure!(
                requested != Some(Peer),
                "cannot force peer {}→{}: {reason}",
                pair[0],
                pair[1]
            );
            return Ok((Host, format!("host{override_note}: {reason}; {topology}")));
        }
        let (bytes, peer, host) = timings.ok_or_else(|| {
            anyhow::anyhow!("{}→{} transport timings are missing", pair[0], pair[1])
        })?;
        ensure!(
            bytes > 0
                && [peer, host]
                    .iter()
                    .all(|v| v[0] > 0 && v[0] <= v[1] && v[1] <= v[2]),
            "{}→{} transport timings are invalid",
            pair[0],
            pair[1]
        );
        let faster = peer[2] < host[0];
        let selected = requested.unwrap_or(if faster { Peer } else { Host });
        let name = if selected == Peer { "peer" } else { "host" };
        let overlap = peer[0] <= host[2] && host[0] <= peer[2];
        let times = format!(
            "{bytes} B: peer median {} ns [{}..{}] vs host median {} ns [{}..{}]{}",
            peer[1],
            peer[0],
            peer[2],
            host[1],
            host[0],
            host[2],
            if overlap {
                "; overlapping spreads, tie"
            } else {
                ""
            }
        );
        Ok((
            selected,
            format!(
                "{name}{override_note}: {}→{}; {times}; {topology}",
                pair[0], pair[1]
            ),
        ))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 2 && self.ranks.len() >= 2,
            "invalid GLM partition policy schema/rank count"
        );
        ensure!(
            self.transports.len() == self.ranks.len() - 1
                && self
                    .transports
                    .iter()
                    .flatten()
                    .all(|(_, reason)| !reason.is_empty()),
            "GLM boundary transport evidence is incomplete"
        );
        let total = self.ranks[0].scope.total_layers;
        let mut seen = std::collections::BTreeSet::new();
        for (rank, p) in self.ranks.iter().enumerate() {
            ensure!(
                seen.insert(p.ordinal),
                "duplicate CUDA ordinal in GLM partition"
            );
            ensure!(
                p.scope == GlmLayerScope::for_rank(total, rank, self.ranks.len())?,
                "noncanonical GLM layer assignment"
            );
            p.execution.validate()?;
            ensure!(
                p.execution.backend == GlmExecutionBackend::Cuda,
                "GLM partitions require CUDA"
            );
            ensure!(
                p.execution
                    .expert_cache
                    .readmission_interval_tokens
                    .is_none(),
                "partition cache bounds must be fixed"
            );
            let mut math = p.execution.clone();
            let reference = &self.ranks[0].execution;
            math.weights = reference.weights.clone();
            math.resident_static = reference.resident_static;
            math.expert_cache = reference.expert_cache.clone();
            ensure!(math == *reference, "GLM rank numerical contracts disagree");
        }
        Ok(())
    }
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_policy::ExpertCacheOptions;
    #[test]
    fn partition_policy_binds_devices_scopes_and_consistent_numerics() {
        use crate::expert_cache_manager::ExpertCacheLayout;
        use candle_core::Device;
        use ff_core::weights::{CachePolicy, WeightSource};
        let mut execution = GlmExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            true,
            8,
            ExpertCacheOptions {
                layout: ExpertCacheLayout::PerLayerSplit,
                maximum_bound_bytes: 1024,
                minimum_bound_bytes: 1024,
                adaptive: false,
            },
        )
        .unwrap();
        execution.backend = GlmExecutionBackend::Cuda;
        let policy = GlmPartitionPolicy {
            schema_version: 2,
            transports: vec![std::array::from_fn(|_| {
                (
                    GlmPartitionTransport::SynchronizedCudaDeviceCopyV1,
                    "validated fixture".into(),
                )
            })],
            ranks: (0..2)
                .map(|rank| GlmRankPolicy {
                    ordinal: rank,
                    scope: GlmLayerScope::for_rank(4, rank, 2).unwrap(),
                    execution: execution.clone(),
                })
                .collect(),
        };
        let bytes = policy.canonical_json().unwrap();
        let restored: GlmPartitionPolicy = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, policy);
        let mut altered = policy.clone();
        altered.ranks[1].ordinal = 0;
        assert!(altered.validate().is_err());
        let mut altered = policy.clone();
        altered.ranks[1].scope.start = 1;
        assert!(altered.validate().is_err());
        let mut altered = policy.clone();
        altered.ranks[1].execution.cpu_fp8_dequantization = false;
        assert!(
            altered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("numerical")
        );
        let mut altered = policy.clone();
        altered.ranks[1].ordinal = 2;
        altered.validate().unwrap();
        assert_ne!(
            altered.canonical_json().unwrap(),
            policy.canonical_json().unwrap()
        );
        let mut altered = policy.clone();
        altered.ranks[1].execution.expert_cache.maximum_bound_bytes = 2048;
        altered.ranks[1].execution.expert_cache.minimum_bound_bytes = 2048;
        assert_ne!(
            altered.canonical_json().unwrap(),
            policy.canonical_json().unwrap()
        );
    }

    #[test]
    fn transport_selection_requires_bidirectional_validity_and_faster_copies() {
        use GlmPartitionTransport::{
            HostStagedCopyV1 as Host, SynchronizedCudaDeviceCopyV1 as Peer,
        };
        let times = Some((32, [400, 410, 420], [620, 630, 640]));
        let choose = |caps,
                      unified,
                      failures,
                      times: Option<(usize, [u64; 3], [u64; 3])>,
                      requested| {
            GlmPartitionPolicy::select_transport([2, 5], caps, unified, failures, times, requested)
        };
        let missing = choose([Some(false), Some(true)], false, [None, None], times, None).unwrap();
        assert_eq!(missing.0, Peer);
        assert!(missing.1.contains("2→5 Some(false)"));
        assert_eq!(
            choose([None, None], false, [None, None], times, None)
                .unwrap()
                .0,
            Peer
        );
        assert_eq!(
            choose(
                [Some(false), Some(false)],
                false,
                [None, None],
                times,
                Some(Peer)
            )
            .unwrap()
            .0,
            Peer
        );
        let failed = choose(
            [Some(true), Some(true)],
            false,
            [None, Some("validation failed".into())],
            None,
            None,
        )
        .unwrap();
        assert_eq!(failed.0, Host);
        assert!(failed.1.contains("5→2 validation failed"));
        assert_eq!(
            choose([Some(true), Some(true)], true, [None, None], None, None)
                .unwrap()
                .0,
            Host
        );
        assert_eq!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, None],
                Some((32, [690, 700, 710], [620, 630, 640])),
                None
            )
            .unwrap()
            .0,
            Host
        );
        assert_eq!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, None],
                Some((32, [620, 630, 640], [620, 630, 640])),
                None
            )
            .unwrap()
            .0,
            Host
        );
        assert_eq!(
            choose([Some(true), Some(true)], false, [None, None], times, None)
                .unwrap()
                .0,
            Peer
        );
        assert_eq!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, None],
                Some((32, [400, 410, 640], [620, 630, 650])),
                None
            )
            .unwrap()
            .0,
            Host
        );
        assert_eq!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, None],
                times,
                Some(Host)
            )
            .unwrap()
            .0,
            Host
        );
        assert_eq!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, None],
                Some((32, [690, 700, 710], [620, 630, 640])),
                Some(Peer)
            )
            .unwrap()
            .0,
            Peer
        );
        assert!(
            choose(
                [Some(true), Some(true)],
                false,
                [None, Some("validation failed".into())],
                None,
                Some(Peer)
            )
            .unwrap_err()
            .to_string()
            .contains("5→2 validation failed")
        );
        assert!(
            choose(
                [Some(true), Some(true)],
                true,
                [None, None],
                times,
                Some(Peer)
            )
            .is_err()
        );
        assert!(choose([Some(true), Some(true)], false, [None, None], None, None).is_err());
    }
}
