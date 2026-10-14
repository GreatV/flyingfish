//! Pure GDN plan geometry, dispatch and shared-scratch admission accounting.
use crate::op_contract::{GdnSpec, WorkspaceReq};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
const HV: u32 = 48;
const HK: u32 = 16;
const DK: u32 = 128;
const DV: u32 = 128;

/// Largest prefill step the MTP-verify small plans cover. Steps of more rows take the
/// chunkwise-P1 core on a qualified device.
pub const MIN_VERIFY_ROWS: u32 = 4;

/// Chunkwise P1 core shared-memory bytes per value split, from
/// `kernels/gdn/gdn_cw_p1_layout.h`. The kernel static_asserts its arena against the
/// same constants, so the launcher and the kernel derive the size from one definition.
pub mod cw_p1 {
    include!(concat!(env!("OUT_DIR"), "/gdn_cw_p1_shared.rs"));
}

/// Immutable device qualification; no runtime GPU query or tuning in a forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GdnPolicy {
    /// Protocol §A value split of the production chunkwise-P1 prefill core, 0 = the
    /// shipped recursive core. Derived once here from the device qualification: the
    /// core's gate was run on sm89 with 128 SMs (box3), and every other device keeps
    /// the shipped core until it has its own gate.
    prefill_p1_split: u32,
}
impl GdnPolicy {
    pub const LEGACY: Self = Self {
        prefill_p1_split: 0,
    };
    pub const fn for_device(arch: u32, sms: usize) -> Self {
        Self {
            prefill_p1_split: if arch == 89 && sms == 128 { 4 } else { 0 },
        }
    }
    /// The device's prepared chunkwise-P1 value split; `p1_split_for_step` is the
    /// per-step decision.
    pub const fn prefill_p1_split(self) -> u32 {
        self.prefill_p1_split
    }
    /// The split a prefill step of `step_rows` rows carries. Every prefill step of
    /// more than MIN_VERIFY_ROWS rows takes the chunkwise-P1 core on a qualified
    /// device; 1..=MIN_VERIFY_ROWS keep split 0 because MTP verify's contract is the
    /// shipped recursive core.
    pub const fn p1_split_for_step(self, step_rows: u32) -> u32 {
        if step_rows > MIN_VERIFY_ROWS {
            self.prefill_p1_split
        } else {
            0
        }
    }
    /// The plan id: the P1-named route where prefill_p1_split is prepared.
    pub const fn identity(self) -> &'static str {
        if self.prefill_p1_split != 0 {
            "gdn-plan-v3-c32-vs1-fp32-p1s4-w32"
        } else {
            GdnSpec::PLAN_ID
        }
    }
    /// All prebound operators reuse the same arena: the maximum of complete plan
    /// layouts is enough, and with the scan buckets gone it is the plain arena.
    pub fn workspace(self, spec: &GdnSpec, physical_rows: u32) -> Result<WorkspaceReq> {
        Ok(WorkspaceReq {
            bytes: GdnWorkspace::q38(spec, physical_rows)?.bytes,
            alignment: 256,
        })
    }
}

/// One named region of the single Engine slice. The binder defines its layout
/// and allocates no workspace of its own.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: u64,
    pub bytes: u64,
}

/// The GDN workspace. Every intermediate the binder produces is listed here with
/// its own offset, so the Engine never has to guess and never hard-codes a total.
///
/// `workspace()` is the ONLY source of the Engine's byte accounting. The Engine
/// binds prefill and decode plans, and shares ONE scratch sized to the largest complete
/// layout, as it does for HD256. Nothing here depends on `prepare` having run.
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
            bytes: off,
            alignment: 256,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_prefill_step_above_the_verify_floor_takes_p1() {
        let qualified = GdnPolicy::for_device(89, 128);
        for rows in [1u32, 2, 3, 4] {
            assert_eq!(
                qualified.p1_split_for_step(rows),
                0,
                "{rows} keeps the verify core"
            );
        }
        for rows in [5u32, 63, 64, 512, 1024, 2048, 2049, 4096, 16384] {
            assert_eq!(qualified.p1_split_for_step(rows), 4, "{rows} must take P1");
        }
        for (arch, sms) in [(86, 48), (89, 80), (90, 128)] {
            let legacy = GdnPolicy::for_device(arch, sms);
            assert_eq!(legacy, GdnPolicy::LEGACY);
            assert_eq!(legacy.prefill_p1_split(), 0);
            assert_eq!(legacy.p1_split_for_step(64), 0);
            assert_eq!(legacy.p1_split_for_step(16384), 0);
        }
    }
    #[test]
    fn exact_shared_arena_and_monotone_admission() -> Result<()> {
        let spec = GdnSpec::q38();
        let policy = GdnPolicy::for_device(89, 128);
        for rows in [512u32, 1024, 2048] {
            let ws = GdnWorkspace::q38(&spec, rows)?;
            assert_eq!(policy.workspace(&spec, rows)?.bytes, ws.bytes);
            assert_eq!(GdnPolicy::LEGACY.workspace(&spec, rows)?.bytes, ws.bytes);
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
            [cw_p1::S4_BYTES, cw_p1::S8_BYTES, cw_p1::S16_BYTES],
            [44800, 40704, 38656]
        );
        Ok(())
    }
}
