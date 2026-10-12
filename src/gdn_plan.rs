//! Pure GDN plan geometry, dispatch and shared-scratch admission accounting.
use crate::op_contract::{ExecutionPhase, GdnSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
const HV: u32 = 48;
const HK: u32 = 16;
const DK: u32 = 128;
const DV: u32 = 128;

pub mod shared {
    include!(concat!(env!("OUT_DIR"), "/gdn_b_shared.rs"));
}

/// Immutable device qualification; no runtime GPU query or tuning in a forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GdnPolicy {
    scan: bool,
}
impl GdnPolicy {
    pub const LEGACY: Self = Self { scan: false };
    pub const fn for_device(arch: u32, sms: usize) -> Self {
        Self {
            scan: arch == 89 && sms == 128,
        }
    }
    pub const fn identity(self) -> &'static str {
        if self.scan {
            "gdn-plan-v3-c32-vs1-fp32-bscan-g4-w32"
        } else {
            GdnSpec::PLAN_ID
        }
    }
    /// Prepared upper bounds, capped at the physical projection/output extent.
    pub fn buckets(self, physical_rows: u32) -> Vec<u32> {
        if !self.scan || physical_rows < 512 {
            return Vec::new();
        }
        let mut result = Vec::new();
        for bound in [512, 1024, 2048] {
            let bound = bound.min(physical_rows);
            if result.last() != Some(&bound) {
                result.push(bound);
            }
        }
        result
    }
    /// Structural Step bound only. The device live count is never read here.
    pub fn select(
        self,
        physical_rows: u32,
        step_rows: u32,
        phase: ExecutionPhase,
    ) -> Result<Option<u32>> {
        ensure!(
            step_rows > 0 && step_rows <= physical_rows,
            "GDN step exceeds physical rows"
        );
        if !self.scan || phase == ExecutionPhase::Decode || !(512..=2048).contains(&step_rows) {
            return Ok(None);
        }
        Ok(Some(step_rows.next_power_of_two().min(physical_rows)))
    }
    /// All prebound operators reuse the same arena. Large legacy plans do not
    /// pay scan scratch on top: the maximum of complete plan layouts is enough.
    pub fn workspace(self, spec: &GdnSpec, physical_rows: u32) -> Result<WorkspaceReq> {
        let mut bytes = GdnWorkspace::q38(spec, physical_rows)?.bytes;
        for bound in self.buckets(physical_rows) {
            bytes = bytes.max(GdnWorkspace::with_scan(spec, bound)?.bytes);
        }
        Ok(WorkspaceReq {
            bytes,
            alignment: 256,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScanWorkspace {
    pub t: Region,
    pub maps: Region,
    pub entries: Region,
    pub leaves: u32,
}
/// One named region of the single Engine slice. The binder defines its layout
/// and allocates no workspace of its own (ABI §6).
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: u64,
    pub bytes: u64,
}

/// The GDN workspace. Every intermediate the binder produces is listed here with
/// its own offset, so the Engine never has to guess and never hard-codes a total.
///
/// `workspace()` is the ONLY source of the Engine's byte accounting. The Engine
/// binds legacy prefill/decode and qualified scan buckets, and shares ONE scratch
/// sized to the largest complete layout, as it does for HD256. Nothing here depends on `prepare` having run.
#[derive(Clone, Copy, Debug)]
pub struct GdnWorkspace {
    /// `[R, conv_dim]` BF16 — the conv result, then SiLU'd IN PLACE. ABI §5: the
    /// conv output cast may not be elided; the common `silu` reads and writes BF16
    /// over the same bytes, so there is no separate `act` region.
    pub conv_out: Region,
    /// `[R, 16, 128]` BF16 — Q after `l2_heads`, KEY heads, before the widen.
    pub q_bf16: Region,
    /// `[R, 16, 128]` BF16 — K after `l2_heads`, KEY heads.
    pub k_bf16: Region,
    // There is deliberately no `v_bf16` region, and its absence is deliberate rather than an
    // oversight: V is never transformed (ABI section 5 norms Q and K only), so it is not an
    // intermediate. `gdn_prep_fp32` reads it in place from the V columns of `conv_out`,
    // whose row stride is conv_dim, which is why that kernel takes an explicit
    // `v_row_stride`. Declaring the region would have been a dead 20,480 B/row.
    /// `[R, 48]` BF16 — `beta = BF16(sigmoid(b))`, as emitted by `gdn_gates`.
    pub beta: Region,
    /// `[R, 48]` FP32 — the exact, value-preserving widen of `beta`. Both cores
    /// declare `const float* beta`, so this widen is required, not optional.
    pub beta_fp32: Region,
    /// `[R, 48]` FP32 — `g = -exp(A_log) * softplus(a + dt_bias)`.
    pub g: Region,
    /// `[R, 48, 128]` FP32 — Q widened, GQA-expanded and scaled by `1/sqrt(DK)`.
    pub q_fp32: Region,
    /// `[R, 48, 128]` FP32 — K widened and expanded to VALUE heads.
    pub k_fp32: Region,
    /// `[R, 48, 128]` FP32 — V widened.
    pub v_fp32: Region,
    pub bytes: u64,
    pub alignment: u64,
    pub scan: Option<ScanWorkspace>,
}

pub(crate) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .context("GDN workspace multiplication overflow")
}
pub(crate) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("GDN workspace addition overflow")
}
fn align(n: u64) -> Result<u64> {
    Ok(add(n, 255)? & !255)
}

impl GdnWorkspace {
    /// CPU-only checked sizing: reads no GPU state and allocates nothing.
    ///
    /// `max_rows` is a WORKSPACE capacity, not a device row count. `prefix` and
    /// `capacity` are per-call and arrive in `GdnMeta`; there is deliberately no
    /// `capacity >= prefix + max_rows` check here, because `prepare` has no
    /// `meta` and the real device-side condition is
    /// `capacity >= prefix + valid_rows` with `valid_rows <= max_rows`.
    pub fn q38(spec: &GdnSpec, max_rows: u32) -> Result<Self> {
        spec.validate()?;
        ensure!(max_rows > 0, "GDN max_rows must be positive");
        let conv_dim = u64::from(spec.conv_dim()?);
        let rows = u64::from(max_rows);
        let hd = u64::from(HV);
        let dv = u64::from(DV);
        let hk = u64::from(HK);
        let dk = u64::from(DK);

        // Every region is [rows, heads, dim]; the per-row byte count is heads*dim*elem.
        // `act` is absent: the common silu runs in place over `conv_out`.
        let hd_dv = mul(hd, dv)?;
        let hk_dk = mul(hk, dk)?;
        let sizes = [
            mul(rows, mul(conv_dim, 2)?)?, // conv_out  BF16 [R, conv_dim]
            mul(rows, mul(hk_dk, 2)?)?,    // q_bf16     BF16 [R, 16, 128] KEY heads
            mul(rows, mul(hk_dk, 2)?)?,    // k_bf16     BF16 [R, 16, 128] KEY heads
            mul(rows, mul(hd, 2)?)?,       // beta       BF16 [R, 48]
            mul(rows, mul(hd, 4)?)?,       // beta_fp32  FP32 [R, 48]
            mul(rows, mul(hd, 4)?)?,       // g          FP32 [R, 48]
            mul(rows, mul(hd_dv, 4)?)?,    // q_fp32     FP32 [R, 48, 128]
            mul(rows, mul(hd_dv, 4)?)?,    // k_fp32     FP32 [R, 48, 128]
            mul(rows, mul(hd_dv, 4)?)?,    // v_fp32     FP32 [R, 48, 128]
        ];
        let mut off = 0u64;
        let mut r = [Region {
            offset: 0,
            bytes: 0,
        }; 9];
        for (i, s) in sizes.iter().enumerate() {
            let padded = align(*s)?;
            r[i] = Region {
                offset: off,
                bytes: padded,
            };
            off = align(add(off, padded)?)?;
        }
        Ok(Self {
            conv_out: r[0],
            q_bf16: r[1],
            k_bf16: r[2],
            beta: r[3],
            beta_fp32: r[4],
            g: r[5],
            q_fp32: r[6],
            k_fp32: r[7],
            v_fp32: r[8],
            scan: None,
            bytes: off,
            alignment: 256,
        })
    }
}

impl GdnWorkspace {
    pub(crate) fn with_scan(spec: &GdnSpec, bound: u32) -> Result<Self> {
        ensure!(
            (512..=2048).contains(&bound),
            "GDN scan bound must be 512..2048"
        );
        let mut ws = Self::q38(spec, bound)?;
        let leaves = bound.div_ceil(128).next_power_of_two();
        let mut region = |floats: u64| -> Result<Region> {
            let bytes = align(mul(floats, 4)?)?;
            let r = Region {
                offset: ws.bytes,
                bytes,
            };
            ws.bytes = add(ws.bytes, bytes)?;
            Ok(r)
        };
        let t = region(mul(mul(48, u64::from(leaves) * 4)?, 32 * 32)?)?;
        let maps = region(mul(mul(48, u64::from(leaves) * 2 - 2)?, 2 * 128 * 128)?)?;
        let entries = region(mul(mul(48, u64::from(leaves))?, 128 * 128)?)?;
        ws.scan = Some(ScanWorkspace {
            t,
            maps,
            entries,
            leaves,
        });
        Ok(ws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qualified_buckets_and_structural_dispatch() -> Result<()> {
        let scan = GdnPolicy::for_device(89, 128);
        for (arch, sms) in [(86, 48), (89, 80), (90, 128)] {
            let legacy = GdnPolicy::for_device(arch, sms);
            assert_eq!(legacy, GdnPolicy::LEGACY);
            assert!(legacy.buckets(2048).is_empty());
        }
        assert_eq!(scan.buckets(1021), vec![512, 1021]);
        assert_eq!(scan.buckets(4096), vec![512, 1024, 2048]);
        for (rows, want) in [
            (1, None),
            (5, None),
            (511, None),
            (512, Some(512)),
            (513, Some(1024)),
            (1021, Some(1024)),
            (1024, Some(1024)),
            (1025, Some(2048)),
            (2048, Some(2048)),
            (2049, None),
        ] {
            assert_eq!(scan.select(4096, rows, ExecutionPhase::Prefill)?, want);
        }
        assert_eq!(scan.select(1021, 513, ExecutionPhase::Prefill)?, Some(1021));
        assert_eq!(scan.select(2048, 1, ExecutionPhase::Decode)?, None);
        assert!(scan.select(1024, 1025, ExecutionPhase::Prefill).is_err());
        assert!(scan.select(1024, 0, ExecutionPhase::Prefill).is_err());
        assert_ne!(scan.identity(), GdnPolicy::LEGACY.identity());
        Ok(())
    }
    #[test]
    fn exact_shared_arena_and_monotone_admission() -> Result<()> {
        let spec = GdnSpec::q38();
        let policy = GdnPolicy::for_device(89, 128);
        for (rows, extra) in [(512, 51), (1024, 114), (2048, 240)] {
            let old = GdnWorkspace::q38(&spec, rows)?;
            let new = GdnWorkspace::with_scan(&spec, rows)?;
            assert_eq!(new.bytes - old.bytes, extra * 1024 * 1024);
            let scan = new.scan.unwrap();
            let regions = [scan.t, scan.maps, scan.entries];
            assert_eq!(regions[0].offset, old.bytes);
            for r in regions {
                assert_eq!(r.offset % 256, 0);
                assert_eq!(r.bytes % 256, 0);
                assert!(r.offset + r.bytes <= new.bytes);
            }
            assert_eq!(scan.t.offset + scan.t.bytes, scan.maps.offset);
            assert_eq!(scan.maps.offset + scan.maps.bytes, scan.entries.offset);
            assert_eq!(policy.workspace(&spec, rows)?.bytes, new.bytes);
            assert_eq!(GdnPolicy::LEGACY.workspace(&spec, rows)?.bytes, old.bytes);
        }
        let large = GdnWorkspace::q38(&spec, 8192)?.bytes;
        assert_eq!(policy.workspace(&spec, 8192)?.bytes, large);
        let mut last = 0;
        for rows in 1..=8192 {
            let bytes = policy.workspace(&spec, rows)?.bytes;
            assert!(bytes >= last, "non-monotone at {rows}");
            last = bytes;
        }
        assert!(policy.workspace(&spec, 0).is_err());
        assert_eq!(
            [
                shared::P1_BYTES,
                shared::LEAF_BYTES,
                shared::UP_BYTES,
                shared::DOWN_BYTES,
                shared::P3_BYTES
            ],
            [21248, 43264, 12288, 8192, 62336]
        );
        Ok(())
    }
}
