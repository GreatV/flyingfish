//! Hardware-adaptive TPU weight placement planner derived from sensed HBM and host RAM.

use super::TpuDevice;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::fs;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Sensed TPU device and host memory capabilities at setup.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TpuTopology {
    pub chip_kind: String,
    pub hbm_limit_bytes: u64,
    pub hbm_baseline_used_bytes: u64,
    pub host_available_bytes: u64,
    pub has_pinned_host: bool,
    pub h2d_bytes_per_s: Option<f64>,
}

impl TpuTopology {
    /// Senses HBM stats and addressable memory spaces from `dev` and host `MemAvailable` from `/proc/meminfo`.
    pub fn detect(dev: &TpuDevice, h2d_bytes_per_s: Option<f64>) -> Result<Self> {
        let stats = dev.memory_stats()?;
        let raw_limit = stats
            .bytes_limit
            .context("TPU device did not report bytes_limit")?;
        ensure!(
            raw_limit > 0,
            "TPU device reported non-positive bytes_limit={raw_limit}"
        );
        let hbm_limit_bytes = u64::try_from(raw_limit).context("bytes_limit overflows u64")?;
        let hbm_baseline_used_bytes =
            u64::try_from(stats.bytes_in_use.max(0)).context("bytes_in_use overflows u64")?;
        let host_available_bytes = read_host_available_bytes()?;
        Ok(Self {
            chip_kind: dev.attributes.chip_kind.clone(),
            hbm_limit_bytes,
            hbm_baseline_used_bytes,
            host_available_bytes,
            has_pinned_host: dev.pinned_host_memory().is_some(),
            h2d_bytes_per_s,
        })
    }

    /// Usable HBM above the baseline occupancy after reserving `max(512 MiB, hbm_limit / 16)` for XLA fragmentation.
    pub fn usable_hbm_bytes(&self) -> u64 {
        let reserve = (self.hbm_limit_bytes / 16).max(512 * MIB);
        self.hbm_limit_bytes
            .saturating_sub(self.hbm_baseline_used_bytes)
            .saturating_sub(reserve)
    }

    /// Usable host RAM for weight staging after reserving `min(host / 4, max(2 GiB, host / 8))` for OS and process.
    pub fn usable_host_bytes(&self) -> u64 {
        let quarter = self.host_available_bytes / 4;
        let eighth = self.host_available_bytes / 8;
        let reserve = eighth.max(2 * GIB).min(quarter);
        self.host_available_bytes.saturating_sub(reserve)
    }

    /// Smallest `host_available_bytes` whose `usable_host_bytes()` is at least `target_usable_bytes`.
    pub fn host_cap_for_usable_bytes(target_usable_bytes: u64) -> u64 {
        let usable = |h: u64| {
            let reserve = (h / 8).max(2 * GIB).min(h / 4);
            h.saturating_sub(reserve)
        };
        let mut h = if target_usable_bytes <= 6 * GIB {
            target_usable_bytes.saturating_mul(4).div_ceil(3)
        } else if target_usable_bytes <= 14 * GIB {
            target_usable_bytes.saturating_add(2 * GIB)
        } else {
            target_usable_bytes.saturating_mul(8).div_ceil(7)
        };
        while usable(h) < target_usable_bytes {
            h = h.saturating_add(1);
        }
        while h > 0 && usable(h - 1) >= target_usable_bytes {
            h -= 1;
        }
        h
    }
}

/// Active weight and execution workspace footprint of a transformer stack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelFootprint {
    /// Persistent non-block HBM tensors (AdaLN tables, rotary tables, residual activations, heads).
    pub persistent_hbm_bytes: u64,
    /// Peak compiled HBM of one block executable (`CompiledMemoryStats.peak_memory_in_bytes`).
    pub compiled_block_peak_bytes: u64,
    /// Argument bytes included inside `compiled_block_peak_bytes` that belong to one block's weights.
    pub compiled_block_weight_arg_bytes: u64,
    /// Active streamed weight bytes per transformer block in layer order.
    pub block_weight_bytes: Vec<u64>,
}

/// Storage tier assigned to one transformer block for the denoising loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum BlockPlacement {
    HbmResident,
    PinnedHostStreamed,
    PageableHostStreamed,
    DiskStreamed,
}

/// Setup-derived per-block placement plan and accounted memory budgets.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TpuPlacementPlan {
    pub chip_kind: String,
    pub usable_hbm_bytes: u64,
    pub usable_host_bytes: u64,
    pub exec_workspace_bytes: u64,
    pub prefetch_depth: usize,
    pub staging_hbm_bytes: u64,
    pub hbm_weight_budget_bytes: u64,
    pub hbm_resident_bytes: u64,
    pub host_staged_bytes: u64,
    pub disk_streamed_bytes: u64,
    pub blocks: Vec<BlockPlacement>,
    pub estimated_stream_s_per_step: Option<f64>,
}

impl TpuPlacementPlan {
    /// Derives the fastest placement that fits `topology` without exceeding usable HBM or host RAM.
    pub fn derive(topology: &TpuTopology, footprint: &ModelFootprint) -> Result<Self> {
        ensure!(
            !footprint.block_weight_bytes.is_empty(),
            "model footprint has zero transformer blocks"
        );
        let usable_hbm = topology.usable_hbm_bytes();
        let usable_host = topology.usable_host_bytes();
        let max_block = footprint
            .block_weight_bytes
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let total_block_bytes: u64 = footprint.block_weight_bytes.iter().copied().sum();

        // Non-weight workspace of the compiled block executable plus persistent tensors.
        let compiled_non_weight = footprint
            .compiled_block_peak_bytes
            .saturating_sub(footprint.compiled_block_weight_arg_bytes);
        let exec_workspace = footprint
            .persistent_hbm_bytes
            .checked_add(compiled_non_weight)
            .context("exec_workspace_bytes overflow")?;

        let min_required_hbm = exec_workspace
            .checked_add(max_block)
            .context("minimum HBM requirement overflow")?;
        ensure!(
            min_required_hbm <= usable_hbm,
            "single block ({max_block} B) plus workspace ({exec_workspace} B) exceeds usable HBM ({usable_hbm} B) on {}",
            topology.chip_kind
        );

        // If every block fits resident in HBM alongside workspace, zero streaming slots are needed.
        let all_resident_hbm = exec_workspace
            .checked_add(total_block_bytes)
            .context("all-resident HBM sum overflow")?;
        let (prefetch_depth, staging_hbm) = if all_resident_hbm <= usable_hbm {
            (0usize, 0u64)
        } else {
            let double_buf = max_block
                .checked_mul(2)
                .context("double-buffer staging overflow")?;
            if exec_workspace.saturating_add(double_buf) <= usable_hbm {
                (2usize, double_buf)
            } else {
                (1usize, max_block)
            }
        };

        let hbm_weight_budget = usable_hbm
            .saturating_sub(exec_workspace)
            .saturating_sub(staging_hbm);
        let host_tier = if topology.has_pinned_host {
            BlockPlacement::PinnedHostStreamed
        } else {
            BlockPlacement::PageableHostStreamed
        };

        let mut blocks = Vec::with_capacity(footprint.block_weight_bytes.len());
        let mut hbm_resident_bytes = 0u64;
        let mut host_staged_bytes = 0u64;
        let mut disk_streamed_bytes = 0u64;

        for &bytes in &footprint.block_weight_bytes {
            if hbm_resident_bytes.saturating_add(bytes) <= hbm_weight_budget {
                hbm_resident_bytes += bytes;
                blocks.push(BlockPlacement::HbmResident);
            } else if host_staged_bytes.saturating_add(bytes) <= usable_host {
                host_staged_bytes += bytes;
                blocks.push(host_tier);
            } else {
                disk_streamed_bytes += bytes;
                blocks.push(BlockPlacement::DiskStreamed);
            }
        }

        let streamed_bytes = host_staged_bytes.saturating_add(disk_streamed_bytes);
        let estimated_stream_s_per_step = topology
            .h2d_bytes_per_s
            .filter(|bw| bw.is_finite() && *bw > 0.0)
            .map(|bw| streamed_bytes as f64 / bw);

        Ok(Self {
            chip_kind: topology.chip_kind.clone(),
            usable_hbm_bytes: usable_hbm,
            usable_host_bytes: usable_host,
            exec_workspace_bytes: exec_workspace,
            prefetch_depth,
            staging_hbm_bytes: staging_hbm,
            hbm_weight_budget_bytes: hbm_weight_budget,
            hbm_resident_bytes,
            host_staged_bytes,
            disk_streamed_bytes,
            blocks,
            estimated_stream_s_per_step,
        })
    }

    pub fn count(&self, placement: BlockPlacement) -> usize {
        self.blocks.iter().filter(|&&p| p == placement).count()
    }

    pub fn hbm_peak_accounted_bytes(&self) -> u64 {
        self.exec_workspace_bytes + self.staging_hbm_bytes + self.hbm_resident_bytes
    }
}

/// Reads `MemAvailable` in bytes from `/proc/meminfo`.
pub fn read_host_available_bytes() -> Result<u64> {
    let text = fs::read_to_string("/proc/meminfo").context("read /proc/meminfo")?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kib = rest
                .split_whitespace()
                .next()
                .context("missing MemAvailable value")?
                .parse::<u64>()
                .context("parse MemAvailable KiB")?;
            return kib
                .checked_mul(1024)
                .context("MemAvailable bytes overflow u64");
        }
    }
    anyhow::bail!("MemAvailable not found in /proc/meminfo")
}

#[cfg(test)]
mod tests {
    use super::*;

    const H3_IMAGE_LAYERS: usize = 50;
    const H3_IMAGE_ACTIVE_BLOCK_BYTES: u64 = 770_725_376;
    const V5E1_RECORDED_HBM_LIMIT: u64 = 16_909_336_064;
    const V5E1_RECORDED_HOST_AVAIL: u64 = 42_181_884 * 1024;
    const V5E1_RECORDED_PINNED_H2D_BPS: f64 = 13.18 * (GIB as f64);
    const V6E1_HBM_LIMIT: u64 = 32 * GIB;

    fn h3_image_footprint() -> ModelFootprint {
        // Persistent non-block HBM (~160 MiB: AdaLN tables, rotary, residual, norm_out/pdd)
        // plus head-chunked block executable peak (~1.4 GiB including 771 MB block weight args).
        ModelFootprint {
            persistent_hbm_bytes: 160 * MIB,
            compiled_block_peak_bytes: H3_IMAGE_ACTIVE_BLOCK_BYTES + 640 * MIB,
            compiled_block_weight_arg_bytes: H3_IMAGE_ACTIVE_BLOCK_BYTES,
            block_weight_bytes: vec![H3_IMAGE_ACTIVE_BLOCK_BYTES; H3_IMAGE_LAYERS],
        }
    }

    #[test]
    fn v5e1_plan_from_recorded_metrics_uses_hbm_and_pinned_host_without_disk_spill() -> Result<()> {
        let topo = TpuTopology {
            chip_kind: "TPU v5 lite".into(),
            hbm_limit_bytes: V5E1_RECORDED_HBM_LIMIT,
            hbm_baseline_used_bytes: 0,
            host_available_bytes: V5E1_RECORDED_HOST_AVAIL,
            has_pinned_host: true,
            h2d_bytes_per_s: Some(V5E1_RECORDED_PINNED_H2D_BPS),
        };
        let plan = TpuPlacementPlan::derive(&topo, &h3_image_footprint())?;
        assert_eq!(plan.blocks.len(), H3_IMAGE_LAYERS);
        assert_eq!(plan.prefetch_depth, 2);
        let resident = plan.count(BlockPlacement::HbmResident);
        let pinned = plan.count(BlockPlacement::PinnedHostStreamed);
        let disk = plan.count(BlockPlacement::DiskStreamed);
        assert!(
            (15..=18).contains(&resident),
            "unexpected v5e1 resident={resident}"
        );
        assert_eq!(resident + pinned, H3_IMAGE_LAYERS);
        assert_eq!(disk, 0);
        assert!(plan.hbm_peak_accounted_bytes() <= plan.usable_hbm_bytes);
        assert!(plan.host_staged_bytes <= plan.usable_host_bytes);
        // Adding one more resident block would cross hbm_weight_budget_bytes.
        assert!(
            plan.hbm_resident_bytes + H3_IMAGE_ACTIVE_BLOCK_BYTES > plan.hbm_weight_budget_bytes
        );
        assert!(plan.estimated_stream_s_per_step.is_some());
        Ok(())
    }

    #[test]
    fn v6e1_plan_keeps_majority_of_h3_image_blocks_resident() -> Result<()> {
        let topo = TpuTopology {
            chip_kind: "TPU v6e".into(),
            hbm_limit_bytes: V6E1_HBM_LIMIT,
            hbm_baseline_used_bytes: 0,
            host_available_bytes: V5E1_RECORDED_HOST_AVAIL,
            has_pinned_host: true,
            h2d_bytes_per_s: Some(V5E1_RECORDED_PINNED_H2D_BPS),
        };
        let plan = TpuPlacementPlan::derive(&topo, &h3_image_footprint())?;
        let resident = plan.count(BlockPlacement::HbmResident);
        let pinned = plan.count(BlockPlacement::PinnedHostStreamed);
        let disk = plan.count(BlockPlacement::DiskStreamed);
        assert!(
            (36..=42).contains(&resident),
            "unexpected v6e1 resident={resident}"
        );
        assert!(pinned > 0 && resident + pinned == H3_IMAGE_LAYERS);
        assert_eq!(disk, 0);
        assert_eq!(plan.prefetch_depth, 2);
        assert!(plan.hbm_peak_accounted_bytes() <= plan.usable_hbm_bytes);
        Ok(())
    }

    #[test]
    fn constrained_host_ram_spills_overflow_blocks_to_disk() -> Result<()> {
        let topo = TpuTopology {
            chip_kind: "TPU v5 lite".into(),
            hbm_limit_bytes: V5E1_RECORDED_HBM_LIMIT,
            hbm_baseline_used_bytes: 0,
            host_available_bytes: 16 * GIB,
            has_pinned_host: true,
            h2d_bytes_per_s: Some(V5E1_RECORDED_PINNED_H2D_BPS),
        };
        let plan = TpuPlacementPlan::derive(&topo, &h3_image_footprint())?;
        let resident = plan.count(BlockPlacement::HbmResident);
        let pinned = plan.count(BlockPlacement::PinnedHostStreamed);
        let disk = plan.count(BlockPlacement::DiskStreamed);
        assert!(resident > 0, "expected resident blocks");
        assert!(pinned > 0, "expected pinned_host blocks");
        assert!(disk > 0, "expected disk-spilled blocks");
        assert_eq!(resident + pinned + disk, H3_IMAGE_LAYERS);
        assert!(plan.host_staged_bytes <= plan.usable_host_bytes);
        assert!(plan.host_staged_bytes + H3_IMAGE_ACTIVE_BLOCK_BYTES > plan.usable_host_bytes);
        // Placement must be monotonic: HbmResident -> PinnedHostStreamed -> DiskStreamed.
        assert!(
            plan.blocks[..resident]
                .iter()
                .all(|&p| p == BlockPlacement::HbmResident)
        );
        assert!(
            plan.blocks[resident..resident + pinned]
                .iter()
                .all(|&p| p == BlockPlacement::PinnedHostStreamed)
        );
        assert!(
            plan.blocks[resident + pinned..]
                .iter()
                .all(|&p| p == BlockPlacement::DiskStreamed)
        );
        Ok(())
    }

    #[test]
    fn all_resident_model_needs_zero_staging_slots_and_impossible_hbm_errors() -> Result<()> {
        let small = ModelFootprint {
            persistent_hbm_bytes: 160 * MIB,
            compiled_block_peak_bytes: H3_IMAGE_ACTIVE_BLOCK_BYTES + 640 * MIB,
            compiled_block_weight_arg_bytes: H3_IMAGE_ACTIVE_BLOCK_BYTES,
            block_weight_bytes: vec![H3_IMAGE_ACTIVE_BLOCK_BYTES; 8],
        };
        let topo = TpuTopology {
            chip_kind: "TPU v5 lite".into(),
            hbm_limit_bytes: V5E1_RECORDED_HBM_LIMIT,
            hbm_baseline_used_bytes: 0,
            host_available_bytes: V5E1_RECORDED_HOST_AVAIL,
            has_pinned_host: true,
            h2d_bytes_per_s: None,
        };
        let plan = TpuPlacementPlan::derive(&topo, &small)?;
        assert_eq!(plan.prefetch_depth, 0);
        assert_eq!(plan.staging_hbm_bytes, 0);
        assert_eq!(plan.count(BlockPlacement::HbmResident), 8);

        let tiny_hbm = TpuTopology {
            hbm_limit_bytes: 1024 * MIB,
            ..topo
        };
        assert!(TpuPlacementPlan::derive(&tiny_hbm, &small).is_err());
        Ok(())
    }

    #[test]
    fn host_cap_for_usable_bytes_forces_exact_disk_spill_on_v5e1_and_v6e1() -> Result<()> {
        let cases = [
            (
                "TPU v5 lite",
                V5E1_RECORDED_HBM_LIMIT,
                V5E1_RECORDED_HOST_AVAIL,
            ),
            ("TPU v6e", V6E1_HBM_LIMIT, 169 * GIB),
        ];
        for (chip_kind, hbm_limit_bytes, host_available_bytes) in cases {
            let topo = TpuTopology {
                chip_kind: chip_kind.into(),
                hbm_limit_bytes,
                hbm_baseline_used_bytes: 0,
                host_available_bytes,
                has_pinned_host: true,
                h2d_bytes_per_s: Some(V5E1_RECORDED_PINNED_H2D_BPS),
            };
            let base = TpuPlacementPlan::derive(&topo, &h3_image_footprint())?;
            let host_blocks = base.count(BlockPlacement::PinnedHostStreamed)
                + base.count(BlockPlacement::PageableHostStreamed);
            assert!(host_blocks >= 3, "{chip_kind}: host_blocks={host_blocks}");
            let target_usable = (host_blocks - 3) as u64 * H3_IMAGE_ACTIVE_BLOCK_BYTES;
            let cap = TpuTopology::host_cap_for_usable_bytes(target_usable);
            let forced_topo = TpuTopology {
                host_available_bytes: topo.host_available_bytes.min(cap),
                ..topo
            };
            let forced = TpuPlacementPlan::derive(&forced_topo, &h3_image_footprint())?;
            assert_eq!(
                forced.count(BlockPlacement::DiskStreamed),
                3,
                "{chip_kind}: expected 3 DiskStreamed blocks, got {}",
                forced.count(BlockPlacement::DiskStreamed)
            );
            assert_eq!(
                forced.count(BlockPlacement::HbmResident),
                base.count(BlockPlacement::HbmResident)
            );
        }
        Ok(())
    }
}
