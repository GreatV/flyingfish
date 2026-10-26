//! Pure GDN plan geometry, dispatch and shared-scratch admission accounting.
use crate::{
    backend::setup::DeviceInfo,
    op_contract::{GdnSpec, WorkspaceReq},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

/// Largest prefill step the MTP-verify small plans cover.
pub const MIN_VERIFY_ROWS: u32 = 4;

/// Chunkwise P1 core shared-memory bytes per split, from `kernels/gdn/gdn_cw_p1_layout.h`.
pub mod cw_p1 {
    include!(concat!(env!("OUT_DIR"), "/gdn_cw_p1_shared.rs"));
}

/// Chunkwise-P2 split arena and workspace-slot bytes, from `kernels/gdn/gdn_cw_p2_layout.h`.
pub mod cw_p2 {
    include!(concat!(env!("OUT_DIR"), "/gdn_cw_p2_shared.rs"));
}

/// Chunk rows C of the chunkwise cores, from `P2_C` in `kernels/gdn/gdn_cw_p2.cu`.
pub const CW_P2_CHUNK: u32 = 64;
/// Value columns per stage-H CTA and per stage-O tile, P1's s4 strip width.
pub const CW_P2_VALUE_BLOCK: u32 = 32;
/// One (chunk, head) chunkwise-P2 slot in kernel byte order: (field, bytes, writing stage); `qk` row stride is `P1_T_LD`.
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
/// Group-count bound for 128-wide FP32 V and 64-row slot chunks.
pub const CW_P2_MAX_GROUPS: usize =
    (2 * CW_P2_SLOT_BYTES).div_ceil(CW_P2_CHUNK as u64 * 128 * 4) as usize;
const _: () = assert!(
    CW_P2_SLOT_BYTES == cw_p2::SLOT_BYTES,
    "the chunkwise-P2 slot fields and gdn_cw_p2_layout.h disagree"
);

/// The chunkwise-P2 workspace for `max_rows`: one slot per (chunk, value head), ceil(rows/64) * value_heads slots.
pub const fn cw_p2_bytes(value_heads: u32, max_rows: u32) -> u64 {
    let chunks = if max_rows == 0 {
        0
    } else {
        (max_rows - 1) / CW_P2_CHUNK + 1
    };
    chunks as u64 * value_heads as u64 * CW_P2_SLOT_BYTES
}

/// The chunkwise-P2 stage ABI: field order is the argument order in `kernels/gdn/gdn_cw_p2.cu`.
#[derive(Copy, Clone)]
pub struct P2StageArgs {
    /// Scaled FP32 [M, hv, 128].
    pub q: u64,
    /// FP32 [M, hv, 128].
    pub k: u64,
    /// BF16 [M, hv, 128] rows, `v_row_stride` elements apart.
    pub v: u64,
    /// FP32 [M, hv]: gdn_beta_widen's output.
    pub beta: u64,
    /// FP32 [M, hv]: gdn_gates' output.
    pub g: u64,
    /// The workspace slot base, or the state for stage H.
    pub ws: u64,
    /// The sequence metadata (eight i32).
    pub meta: u64,
    /// Physical row bound of the prebound plan.
    pub max_rows: i32,
    /// First row of this 64-row-aligned group.
    pub row_base: i32,
    /// Physical rows in this group, including its partial final chunk.
    pub group_rows: i32,
    /// Stage A only: the element stride between V's rows.
    pub v_row_stride: u32,
}

/// Immutable device qualification; no runtime GPU query or tuning in a forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GdnPolicy {
    cw_p2: bool,
}
impl GdnPolicy {
    pub const P2: Self = Self { cw_p2: true };
    pub const fn for_device(info: &DeviceInfo) -> Self {
        Self {
            cw_p2: info.arch >= 80
                && info.sms > 0
                && info.warp == 32
                && info.max_threads >= 256
                && info.optin_shared_bytes as u64 >= cw_p2::SMEM_BYTES,
        }
    }
    /// Steps above MIN_VERIFY_ROWS use the resource-qualified chunkwise core.
    pub const fn cw_p2_for_step(self, step_rows: u32) -> bool {
        self.cw_p2 && step_rows > MIN_VERIFY_ROWS
    }
    pub const fn identity(self) -> &'static str {
        if self.cw_p2 {
            "gdn-plan-v5-c32-vs1-l2fp32-p2-compact-w32"
        } else {
            GdnSpec::PLAN_ID
        }
    }
    /// All prebound operators reuse the arena sized by the maximum complete plan layout.
    pub fn workspace(self, spec: &GdnSpec, physical_rows: u32) -> Result<WorkspaceReq> {
        let p2 = self.cw_p2_for_step(physical_rows);
        Ok(WorkspaceReq {
            bytes: GdnWorkspace::new(spec, physical_rows, p2)?.bytes,
            alignment: 256,
        })
    }
}

/// One named region of the single Engine slice; the binder allocates no workspace of its own.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: u64,
    pub bytes: u64,
}

/// Every GDN binder intermediate with its own offset; the only source of the GDN workspace bytes.
#[derive(Clone, Copy, Debug)]
pub struct GdnWorkspace {
    /// `[R, conv_dim]` BF16 — the conv result, SiLU'd in place; there is no `act` region.
    pub conv_out: Region,
    /// `[R, hv]` BF16 — `beta = BF16(sigmoid(b))`, as emitted by `gdn_gates`.
    pub beta: Region,
    /// `[R, hv]` FP32 — the exact widen of `beta`; both cores read `const float* beta`.
    pub beta_fp32: Region,
    /// `[R, hv]` FP32 — `g = -exp(A_log) * softplus(a + dt_bias)`.
    pub g: Region,
    /// `[R, qk_heads, 128]` FP32 — Q normalized and scaled by `1/sqrt(DK)`.
    pub q_fp32: Region,
    /// `[R, qk_heads, 128]` FP32 — normalized K.
    pub k_fp32: Region,
    /// `[R, hv, 128]` FP32 — V widened.
    pub v_fp32: Region,
    /// `[R, hk, 128]` BF16 for unscaled P2 A operands; empty for expanded sources.
    pub q_bf16: Region,
    pub k_bf16: Region,
    /// P2 slots alias V after prep, or occupy one additive chunk for short plans.
    pub p2: Region,
    /// Maximum rows per P2 slot group; a multiple of 64, or zero without P2.
    pub p2_group_rows: u32,
    /// Physical head count of normalized Q/K, hk for compact P2 and hv otherwise.
    pub qk_heads: u32,
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
    /// CPU-only checked sizing; `max_rows` is a workspace capacity, not a device row count.
    pub fn new(spec: &GdnSpec, max_rows: u32, p2: bool) -> Result<Self> {
        spec.validate()?;
        ensure!(max_rows > 0, "GDN max_rows must be positive");
        let conv_dim = u64::from(spec.conv_dim()?);
        let rows = u64::from(max_rows);
        let hd = u64::from(spec.hv);
        let dv = u64::from(spec.dv);

        let hd_dv = mul(hd, dv)?;
        let compact = p2 && spec.hv == 48 && max_rows > MIN_VERIFY_ROWS;
        let qk_heads = if compact { spec.hk } else { spec.hv };
        let qk_width = mul(u64::from(qk_heads), u64::from(spec.dk))?;
        let rounded = if compact {
            mul(rows, mul(qk_width, 2)?)?
        } else {
            0
        };
        let sizes = [
            mul(rows, mul(conv_dim, 2)?)?, // conv_out  BF16 [R, conv_dim]
            mul(rows, mul(hd, 2)?)?,       // beta       BF16 [R, hv]
            mul(rows, mul(hd, 4)?)?,       // beta_fp32  FP32 [R, hv]
            mul(rows, mul(hd, 4)?)?,       // g          FP32 [R, hv]
            mul(rows, mul(qk_width, 4)?)?, // q_fp32
            mul(rows, mul(qk_width, 4)?)?, // k_fp32
            mul(rows, mul(hd_dv, 4)?)?,    // v_fp32     FP32 [R, hv, 128]
            rounded,
            rounded,
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
        let (p2_region, p2_group_rows) = if p2 {
            let chunk_bytes = cw_p2_bytes(spec.hv, CW_P2_CHUNK);
            let alias_chunks =
                (r[6].bytes / chunk_bytes).min(u64::from(max_rows.div_ceil(CW_P2_CHUNK)));
            let chunks = alias_chunks.max(1);
            (
                Region {
                    offset: if alias_chunks > 0 { r[6].offset } else { off },
                    bytes: mul(chunks, chunk_bytes)?,
                },
                u32::try_from(mul(chunks, u64::from(CW_P2_CHUNK))?)?,
            )
        } else {
            (
                Region {
                    offset: off,
                    bytes: 0,
                },
                0,
            )
        };
        let mut bytes = off.max(add(p2_region.offset, p2_region.bytes)?);
        if p2 && p2_region.offset < off {
            let first_alias_rows = cw_p2_bytes(spec.hv, CW_P2_CHUNK).div_ceil(mul(hd_dv, 4)?);
            let short_rows = u32::try_from(first_alias_rows - 1)?;
            bytes = bytes.max(Self::new(spec, short_rows, true)?.bytes);
        }
        Ok(Self {
            conv_out: r[0],
            beta: r[1],
            beta_fp32: r[2],
            g: r[3],
            q_fp32: r[4],
            k_fp32: r[5],
            v_fp32: r[6],
            q_bf16: r[7],
            k_bf16: r[8],
            p2: p2_region,
            p2_group_rows,
            qk_heads,
            bytes,
            alignment: 256,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_prefill_step_above_the_verify_floor_takes_the_chunkwise_core() {
        let mut info = DeviceInfo {
            name: String::new(),
            arch: 86,
            sms: 48,
            l2_bytes: 0,
            warp: 32,
            max_threads: 1024,
            shared_bytes: 48 * 1024,
            sm_shared_bytes: 100 * 1024,
            cooperative: false,
            optin_shared_bytes: 99 * 1024,
            memory_bytes_per_second: 448_064_000_000.0,
        };
        for (arch, sms) in [(86, 48), (89, 128), (89, 80), (90, 128)] {
            info.arch = arch;
            info.sms = sms;
            let policy = GdnPolicy::for_device(&info);
            assert_eq!(policy, GdnPolicy::P2);
            for rows in 1..=MIN_VERIFY_ROWS {
                assert!(!policy.cw_p2_for_step(rows));
            }
            for rows in [5, 63, 64, 512, 1021, 2048, 2049, 4096, 16384] {
                assert!(policy.cw_p2_for_step(rows));
            }
        }
        for (threads, shared, eligible) in [
            (256, cw_p2::SMEM_BYTES, true),
            (255, cw_p2::SMEM_BYTES, false),
            (256, cw_p2::SMEM_BYTES - 1, false),
        ] {
            info.max_threads = threads;
            info.optin_shared_bytes = shared as usize;
            assert_eq!(GdnPolicy::for_device(&info).cw_p2_for_step(4096), eligible);
        }
    }
    #[test]
    fn exact_shared_arena_and_monotone_admission() -> Result<()> {
        let spec = GdnSpec::q38();
        let policy = GdnPolicy::P2;
        let direct_ws = GdnWorkspace::new(&spec, 2048, false)?;
        assert!(policy.workspace(&spec, 2048)?.bytes < direct_ws.bytes);
        for hv in [32, 48] {
            let spec = GdnSpec { hv, ..spec };
            let mut last = 0;
            for rows in 1..=65_535 {
                let p2 = policy.cw_p2_for_step(rows);
                let ws = GdnWorkspace::new(&spec, rows, p2)?;
                let bytes = policy.workspace(&spec, rows)?.bytes;
                assert!(bytes >= last, "non-monotone at hv={hv} rows={rows}");
                last = bytes;
                let base = ws.k_bf16.offset + ws.k_bf16.bytes;
                let original = base + if p2 { cw_p2_bytes(hv, rows) } else { 0 };
                assert!(bytes <= original, "grew at hv={hv} rows={rows}");
                assert!(ws.p2.offset.is_multiple_of(256));
                assert!(ws.p2.offset + ws.p2.bytes <= bytes);
                if !p2 {
                    assert_eq!(ws.p2.bytes, 0);
                    assert_eq!(ws.p2_group_rows, 0);
                    continue;
                }
                assert!(ws.p2_group_rows.is_multiple_of(CW_P2_CHUNK));
                assert!(ws.p2_group_rows > 0);
                assert!(rows.div_ceil(ws.p2_group_rows) as usize <= CW_P2_MAX_GROUPS);
                assert_eq!(ws.p2.bytes, cw_p2_bytes(hv, ws.p2_group_rows));
                for live in [
                    ws.conv_out,
                    ws.beta,
                    ws.beta_fp32,
                    ws.g,
                    ws.q_fp32,
                    ws.k_fp32,
                    ws.q_bf16,
                    ws.k_bf16,
                ] {
                    assert!(
                        live.offset + live.bytes <= ws.p2.offset
                            || ws.p2.offset + ws.p2.bytes <= live.offset
                    );
                }
                if rows >= 116 {
                    assert_eq!(ws.p2.offset, ws.v_fp32.offset);
                    assert!(ws.p2.bytes <= ws.v_fp32.bytes);
                }
                if rows == 4096 {
                    assert_eq!(bytes, base);
                    assert_eq!(ws.p2_group_rows, 2240);
                }
            }
        }
        assert!(policy.workspace(&spec, 0).is_err());
        assert_eq!(
            [cw_p1::S4_BYTES, cw_p1::S8_BYTES, cw_p1::S16_BYTES],
            [44800, 40704, 38656]
        );
        Ok(())
    }

    #[test]
    fn value_head_geometry_bounds_every_workspace_region() -> Result<()> {
        for hv in [32, 48] {
            let spec = GdnSpec {
                hv,
                ..GdnSpec::q38()
            };
            let cd = (2 * spec.hk + hv) * 128;
            assert_eq!(spec.conv_dim()?, cd);
            assert_eq!(spec.state_bytes()?, u64::from(hv) * 128 * 128 * 4);
            assert_eq!(spec.history_bytes()?, u64::from(cd) * 3 * 4);
            for rows in [1, 2, 5, 8, 37, 512, 4096] {
                let ws = GdnWorkspace::new(&spec, rows, false)?;
                let padded = |n: u64| (n + 255) & !255;
                let rows = u64::from(rows);
                assert_eq!(ws.conv_out.bytes, padded(rows * u64::from(cd) * 2));
                assert_eq!(ws.beta.bytes, padded(rows * u64::from(hv) * 2));
                assert_eq!(ws.g.bytes, padded(rows * u64::from(hv) * 4));
                for region in [ws.q_fp32, ws.k_fp32, ws.v_fp32] {
                    assert_eq!(region.bytes, padded(rows * u64::from(hv) * 128 * 4));
                    assert!(region.offset + region.bytes <= ws.bytes);
                }
                assert_eq!(ws.v_fp32.offset + ws.v_fp32.bytes, ws.bytes);
                let compact = GdnWorkspace::new(&spec, rows as u32, true)?;
                let active = hv == 48 && rows > u64::from(MIN_VERIFY_ROWS);
                assert_eq!(compact.qk_heads, if active { spec.hk } else { hv });
                let width = rows * u64::from(compact.qk_heads) * 128;
                assert_eq!(compact.q_fp32.bytes, padded(width * 4));
                assert_eq!(compact.k_fp32.bytes, padded(width * 4));
                assert_eq!(
                    compact.q_bf16.bytes,
                    if active { padded(width * 2) } else { 0 }
                );
                assert_eq!(compact.k_bf16.bytes, compact.q_bf16.bytes);
                let mut end = 0;
                for region in [
                    compact.conv_out,
                    compact.beta,
                    compact.beta_fp32,
                    compact.g,
                    compact.q_fp32,
                    compact.k_fp32,
                    compact.v_fp32,
                    compact.q_bf16,
                    compact.k_bf16,
                ] {
                    assert!(region.offset >= end && region.offset.is_multiple_of(256));
                    end = region.offset + region.bytes;
                }
                assert!(end <= compact.bytes);
            }
        }
        Ok(())
    }
}
