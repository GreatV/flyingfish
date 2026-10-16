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

/// Chunkwise P1 core shared-memory bytes per split, from `kernels/gdn/gdn_cw_p1_layout.h`.
pub mod cw_p1 {
    include!(concat!(env!("OUT_DIR"), "/gdn_cw_p1_shared.rs"));
}

/// Chunkwise-P2 split arena and workspace-slot bytes, from
/// `kernels/gdn/gdn_cw_p2_layout.h`.
pub mod cw_p2 {
    include!(concat!(env!("OUT_DIR"), "/gdn_cw_p2_shared.rs"));
}

/// Chunk rows C of the chunkwise cores, from `P2_C` in `kernels/gdn/gdn_cw_p2.cu`.
pub const CW_P2_CHUNK: u32 = 64;
/// Value columns per stage-H CTA and per stage-O tile, P1's s4 strip width.
pub const CW_P2_VALUE_BLOCK: u32 = 32;
/// The chunkwise-P2 workspace slot of one (chunk, head), in kernel byte order, with the
/// writing stage: `W` bf16[64,128], `U` fp32[64,128], `qk` bf16[64,72] (row stride
/// `P1_T_LD`, so stage H reads it with `p1_a_sbf` in place), then the FP32[64] vectors
/// `Gc`, `Be`, `Sc`.
pub const CW_P2_SLOT: [(&str, u64, char); 6] = [
    ("W", 64 * 128 * 2, 'A'),
    ("U", 64 * 128 * 4, 'A'),
    ("qk", 64 * 72 * 2, 'A'),
    ("Gc", 64 * 4, 'A'),
    ("Be", 64 * 4, 'A'),
    ("Sc", 64 * 4, 'A'),
];

/// Field offsets of the slot, each 256 B aligned, in the kernel's order.
pub const fn cw_p2_slot_layout() -> [(u64, u64, char); 6] {
    let mut layout = [(0u64, 0u64, '?'); 6];
    let mut off = 0u64;
    let mut i = 0;
    while i < CW_P2_SLOT.len() {
        off = (off + 255) & !255;
        let (_, bytes, writer) = CW_P2_SLOT[i];
        layout[i] = (off, bytes, writer);
        off += bytes;
        i += 1;
    }
    layout
}

/// Bytes of one slot: the six fields above.
pub const CW_P2_SLOT_BYTES: u64 = 64 * 128 * 2 + 64 * 128 * 4 + 64 * 72 * 2 + 3 * 64 * 4;
const _: () = assert!(
    CW_P2_SLOT_BYTES == cw_p2::SLOT_BYTES,
    "the chunkwise-P2 slot fields and gdn_cw_p2_layout.h disagree"
);

/// The chunkwise-P2 workspace for `max_rows`: one slot per (chunk, head), C = 64 rows to
/// a chunk and 48 value heads, so ceil(rows/64) * 48 slots of 141,056 B -- 2204 B per row
/// per head, 105,792 B per row across the 48 heads, with the tail chunk rounded up whole.
pub const fn cw_p2_bytes(max_rows: u32) -> u64 {
    let chunks = if max_rows == 0 {
        0
    } else {
        (max_rows - 1) / CW_P2_CHUNK + 1
    };
    chunks as u64 * 48 * CW_P2_SLOT_BYTES
}

/// The chunkwise-P2 stage ABI: field order IS the argument order of all three
/// entries, as declared in `kernels/gdn/gdn_cw_p2.cu`.
#[derive(Copy, Clone)]
pub struct P2StageArgs {
    /// BF16 [M, 16, DK] key heads; the stages expand to the 48 value heads.
    pub q: u64,
    /// BF16 [M, 16, DK] key heads.
    pub k: u64,
    /// BF16 [M, 48, 128] rows, `v_row_stride` elements apart.
    pub v: u64,
    /// FP32 [M, 48]: gdn_beta_widen's output.
    pub beta: u64,
    /// FP32 [M, 48]: gdn_gates' output.
    pub g: u64,
    /// The workspace slot base, or the state for stage H.
    pub ws: u64,
    /// The sequence metadata (eight i32).
    pub meta: u64,
    /// 0, the sentinel that makes the kernel read `meta->valid_rows`.
    pub valid_rows: i32,
    /// Stage A only: the element stride between V's rows.
    pub v_row_stride: u32,
}

/// Immutable device qualification; no runtime GPU query or tuning in a forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GdnPolicy {
    /// Protocol §A value split of the production chunkwise-P1 prefill core, 0 = the
    /// shipped recursive core; qualified for arch 89 with 128 SMs, 0 elsewhere.
    prefill_p1_split: u32,
    /// The chunkwise-P2 split (three kernels over a workspace slot) replaces the fused
    /// P1 core; prepared for arch 89 with 128 SMs.
    cw_p2: bool,
}
impl GdnPolicy {
    pub const LEGACY: Self = Self {
        prefill_p1_split: 0,
        cw_p2: false,
    };
    /// The chunkwise-P2 split.
    pub const P2: Self = Self {
        prefill_p1_split: 0,
        cw_p2: true,
    };
    pub const fn for_device(arch: u32, sms: usize) -> Self {
        Self {
            prefill_p1_split: 0,
            cw_p2: arch == 89 && sms == 128,
        }
    }
    /// The device's prepared chunkwise-P1 value split; `p1_split_for_step` is the
    /// per-step decision.
    pub const fn prefill_p1_split(self) -> u32 {
        self.prefill_p1_split
    }
    /// The split a prefill step of `step_rows` rows carries: steps above
    /// MIN_VERIFY_ROWS take the chunkwise core, 1..=MIN_VERIFY_ROWS keep split 0.
    pub const fn p1_split_for_step(self, step_rows: u32) -> u32 {
        if step_rows > MIN_VERIFY_ROWS {
            self.prefill_p1_split
        } else {
            0
        }
    }
    /// Whether a prefill step of `step_rows` rows carries the chunkwise-P2 split, on the
    /// same verify floor as `p1_split_for_step`.
    pub const fn cw_p2_for_step(self, step_rows: u32) -> bool {
        self.cw_p2 && step_rows > MIN_VERIFY_ROWS
    }
    /// The plan id.
    pub const fn identity(self) -> &'static str {
        if self.cw_p2 {
            "gdn-plan-v3-c32-vs1-fp32-p2-w32"
        } else {
            GdnSpec::PLAN_ID
        }
    }
    /// All prebound operators reuse the arena sized by the maximum complete plan layout.
    pub fn workspace(self, spec: &GdnSpec, physical_rows: u32) -> Result<WorkspaceReq> {
        let bytes = GdnWorkspace::q38(spec, physical_rows)?.bytes;
        Ok(WorkspaceReq {
            bytes: if self.cw_p2 {
                add(bytes, cw_p2_bytes(physical_rows))?
            } else {
                bytes
            },
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

    /// The stage ABI declaration and the argument counts must agree with each other: the
    /// struct carries ten fields, A pushes all ten, and H and O push nine (no V stride).
    #[test]
    fn every_prefill_step_above_the_verify_floor_takes_the_chunkwise_core() {
        let qualified = GdnPolicy::for_device(89, 128);
        assert_eq!(qualified, GdnPolicy::P2);
        for rows in [1u32, 2, 3, 4] {
            assert_eq!(
                qualified.p1_split_for_step(rows),
                0,
                "{rows} keeps the verify core"
            );
            assert!(
                !qualified.cw_p2_for_step(rows),
                "{rows} keeps the verify core"
            );
        }
        for rows in [5u32, 63, 64, 512, 1021, 2048, 2049, 4096, 16384] {
            assert!(qualified.cw_p2_for_step(rows), "{rows} must take P2");
        }
        for (arch, sms) in [(86, 48), (89, 80), (90, 128)] {
            let legacy = GdnPolicy::for_device(arch, sms);
            assert_eq!(legacy, GdnPolicy::LEGACY);
            assert_eq!(legacy.prefill_p1_split(), 0);
            assert!(!legacy.cw_p2_for_step(64));
            assert!(!legacy.cw_p2_for_step(16384));
        }
    }
    #[test]
    fn exact_shared_arena_and_monotone_admission() -> Result<()> {
        let spec = GdnSpec::q38();
        let policy = GdnPolicy::for_device(89, 128);
        let legacy_ws = GdnPolicy::LEGACY.workspace(&spec, 2048)?;
        assert!(policy.workspace(&spec, 2048)?.bytes > legacy_ws.bytes);
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
