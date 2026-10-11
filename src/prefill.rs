use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Plan {
    pub chunk: usize,
    pub requested_chunk: Option<usize>,
    pub free_bytes: usize,
    pub reserve_bytes: usize,
    pub row_bytes: usize,
    pub row_limit: usize,
    pub alignment: usize,
}

impl Plan {
    pub fn new(
        c: &Config,
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
        draft: bool,
    ) -> Result<Self> {
        let row_bytes = (4 * c.hidden_size + c.qkv_dim() + 3 * c.intermediate_size) * 2
            + 4
            + c.num_attention_heads * 4
            + if draft {
                (7 * c.hidden_size + 5 * c.kv_dim()) * 2
            } else {
                0
            };
        Self::budget(capacity, requested, free, total, alignment, row_bytes)
    }

    fn budget(
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
        row_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            alignment > 0 && row_bytes > 0,
            "invalid prefill alignment or row size"
        );
        ensure!(free <= total, "free VRAM exceeds total VRAM");
        let reserve_bytes = total / 16;
        let available = free.checked_sub(reserve_bytes).ok_or_else(|| {
            anyhow::anyhow!(
                "prefill VRAM budget: required reserve={reserve_bytes} available={free}"
            )
        })?;
        let row_limit = capacity.min(available / row_bytes);
        ensure!(
            row_limit > 0,
            "prefill workspace cannot fit one row: required={row_bytes} available={available}"
        );
        let chunk = if let Some(chunk) = requested {
            ensure!(
                chunk > 0 && chunk <= row_limit,
                "requested prefill chunk {chunk} exceeds row budget {row_limit}: required={} available={available}",
                chunk
                    .checked_mul(row_bytes)
                    .context("prefill byte count overflow")?
            );
            chunk
        } else if row_limit < alignment {
            row_limit
        } else {
            let tiles = row_limit / alignment;
            alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
        };
        Ok(Self {
            chunk,
            requested_chunk: requested,
            free_bytes: free,
            reserve_bytes,
            row_bytes,
            row_limit,
            alignment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Plan;

    #[test]
    fn budget_bounds_and_override() {
        let plan = Plan::budget(33024, None, 160000, 160000, 64, 4).unwrap();
        assert_eq!(plan.chunk, 32768);
        assert!(plan.chunk * plan.row_bytes + plan.reserve_bytes <= plan.free_bytes);
        assert!(Plan::budget(33024, Some(33025), 160000, 160000, 64, 4).is_err());
        assert_eq!(
            Plan::budget(33024, Some(33024), 160000, 160000, 64, 4)
                .unwrap()
                .chunk,
            33024
        );
        let exact = Plan::budget(16, Some(16), 72, 128, 1, 4).unwrap();
        assert_eq!(exact.row_limit, 16);
        assert!(Plan::budget(16, Some(16), 71, 128, 1, 4).is_err());
    }

    #[test]
    fn memory_limit_and_small_sequence() {
        assert_eq!(
            Plan::budget(32768, None, 20000, 160000, 64, 4)
                .unwrap()
                .chunk,
            2048
        );
        assert_eq!(
            Plan::budget(17, None, 160000, 160000, 64, 4).unwrap().chunk,
            17
        );
        assert!(Plan::budget(32768, None, 10000, 160000, 64, 4).is_err());
    }
}

/// Shared budget input for models with persistent mixer state and per-operator scratch.
/// `free_after_resident` is sampled after weights, KV/state and layer arena allocation;
/// they are recorded in the ledger but are not subtracted a second time.
#[derive(Debug, Serialize)]
pub struct WorkspaceBudget {
    pub plan: Plan,
    pub operator_workspace_bytes: usize,
    pub planned_transient_bytes: usize,
    pub accounting: &'static str,
}
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceRequest {
    pub capacity: usize,
    pub requested: Option<usize>,
    pub free_after_resident: usize,
    pub total: usize,
    pub alignment: usize,
    pub row_bytes: usize,
    pub fixed_transient_bytes: usize,
}
impl WorkspaceBudget {
    /// `workspace(rows)` must be monotone and include every simultaneously live
    /// operator temporary, with alignment. Sequential operators contribute their max.
    pub fn new(
        request: WorkspaceRequest,
        workspace: impl Fn(usize) -> Result<usize>,
    ) -> Result<Self> {
        let WorkspaceRequest {
            capacity,
            requested,
            free_after_resident,
            total,
            alignment,
            row_bytes,
            fixed_transient_bytes,
        } = request;
        ensure!(
            capacity > 0 && alignment > 0 && row_bytes > 0 && free_after_resident <= total,
            "invalid workspace budget"
        );
        let reserve = total / 16;
        let available = free_after_resident
            .checked_sub(reserve)
            .context("VRAM reserve exceeds free memory")?;
        let bytes = |rows: usize| -> Result<usize> {
            let operator_bytes = workspace(rows)?;
            rows.checked_mul(row_bytes)
                .and_then(|v| v.checked_add(fixed_transient_bytes))
                .and_then(|v| v.checked_add(operator_bytes))
                .context("workspace byte count overflow")
        };
        ensure!(bytes(1)? <= available, "workspace cannot fit one row");
        let (mut low, mut high) = (1usize, capacity);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if bytes(mid)? <= available {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let row_limit = low;
        let chunk = if let Some(n) = requested {
            ensure!(
                n > 0 && n <= row_limit,
                "requested chunk {n} exceeds exact workspace budget: capacity={capacity} row_limit={row_limit} available_bytes={available} row_bytes={row_bytes} fixed_transient_bytes={fixed_transient_bytes}"
            );
            n
        } else if row_limit < alignment {
            row_limit
        } else {
            let tiles = row_limit / alignment;
            alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
        };
        let scratch = workspace(chunk)?;
        Ok(Self {
            plan: Plan {
                chunk,
                requested_chunk: requested,
                free_bytes: free_after_resident,
                reserve_bytes: reserve,
                row_bytes,
                row_limit,
                alignment,
            },
            operator_workspace_bytes: scratch,
            planned_transient_bytes: bytes(chunk)?,
            accounting: "free_after_resident",
        })
    }
}
