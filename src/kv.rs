use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

include!(concat!(env!("OUT_DIR"), "/kv_geometry.rs"));
pub const RING_TOKENS: usize = SINK_TOKENS + RECENT_TOKENS + GROUP_TOKENS - 1 + TENTATIVE_TOKENS;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvFormat {
    Bf16,
    Affine8,
}

impl KvFormat {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Affine8 => "UINT8_KC128_VT_F32_S64_R2048_MUL_ADD_BF16_RNE_V1",
        }
    }
    pub fn from_tag(tag: &str) -> Result<Self> {
        [Self::Bf16, Self::Affine8]
            .into_iter()
            .find(|f| f.tag() == tag)
            .context("unknown KV format identity")
    }
    pub fn layout(self, capacity: usize, heads: usize, dim: usize) -> Result<KvLayout> {
        ensure!(capacity > 0 && heads > 0 && dim > 0, "empty KV layout");
        ensure!(
            self == Self::Bf16 || dim == 256,
            "8-bit KV requires dimension256"
        );
        let mul = |a: usize, b| a.checked_mul(b).context("KV size overflow");
        let ring = if self == Self::Bf16 {
            capacity
        } else {
            capacity.min(RING_TOKENS)
        };
        let codes = if self == Self::Bf16 {
            0
        } else {
            mul(mul(heads, capacity)?, dim)?
        };
        let k_stats = match self {
            Self::Bf16 => 0,
            Self::Affine8 => mul(mul(mul(heads, capacity.div_ceil(GROUP_TOKENS))?, dim)?, 2)?,
        };
        let v_stats = match self {
            Self::Bf16 => 0,
            Self::Affine8 => mul(mul(heads, capacity)?, 2)?,
        };
        let ring_elements = mul(mul(heads, ring)?, dim)?;
        let bytes = mul(ring_elements, 4)?
            .checked_add(mul(codes, 2)?)
            .and_then(|n| {
                k_stats
                    .checked_add(v_stats)
                    .and_then(|v| v.checked_mul(4))
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| n.checked_add(if self == Self::Bf16 { 0 } else { 16 }))
            .context("KV byte count overflow")?;
        Ok(KvLayout {
            capacity,
            ring,
            ring_elements,
            codes,
            k_stats,
            v_stats,
            bytes,
        })
    }
    pub fn block_bytes(
        self,
        start: usize,
        tokens: usize,
        prefix: usize,
        heads: usize,
        dim: usize,
    ) -> Result<usize> {
        let end = start.checked_add(tokens).context("KV block overflow")?;
        ensure!(tokens > 0 && end <= prefix, "KV block outside prefix");
        if self == Self::Bf16 {
            return heads
                .checked_mul(tokens)
                .and_then(|n| n.checked_mul(dim))
                .and_then(|n| n.checked_mul(4))
                .context("KV block bytes overflow");
        }
        ensure!(
            start.is_multiple_of(GROUP_TOKENS),
            "compact KV block must start on128-token group"
        );
        let live = (start..end)
            .filter(|&p| p < SINK_TOKENS || p >= sealed_until(prefix))
            .count();
        let encoded = end.min(sealed_until(prefix)).saturating_sub(start);
        let stats = encoded / GROUP_TOKENS * dim * 2 + encoded * 2;
        heads
            .checked_mul(encoded * dim * 2 + stats * 4 + live * dim * 4)
            .context("compact prefix bytes overflow")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    pub capacity: usize,
    pub ring: usize,
    pub ring_elements: usize,
    pub codes: usize,
    pub k_stats: usize,
    pub v_stats: usize,
    pub bytes: usize,
}

pub fn sealed_until(committed: usize) -> usize {
    committed.saturating_sub(RECENT_TOKENS) / GROUP_TOKENS * GROUP_TOKENS
}

pub fn ring_slot(position: usize, ring: usize) -> Result<usize> {
    ensure!(
        ring > 0 && (position < ring || ring > SINK_TOKENS),
        "KV ring cannot address position"
    );
    Ok(if position < SINK_TOKENS {
        position
    } else {
        SINK_TOKENS + (position - SINK_TOKENS) % (ring - SINK_TOKENS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_payload_excludes_unsealed_codes_and_partial_statistics() -> Result<()> {
        let format = KvFormat::Affine8;
        assert_eq!(
            format.block_bytes(2048, 129, 2177, 4, 256)?,
            4 * 129 * 256 * 4
        );
        assert_eq!(
            format.block_bytes(128, 128, 2177, 4, 256)?,
            4 * 128 * 256 * 4
        );
        assert_eq!(format.block_bytes(0, 127, 127, 4, 256)?, 4 * 127 * 256 * 4);
        Ok(())
    }
    #[test]
    fn typed_bytes_include_scales_ring_and_cursor() -> Result<()> {
        let a = KvFormat::Affine8.layout(262144, 4, 256)?;
        assert_eq!(
            (a.codes * 2 + (a.k_stats + a.v_stats) * 4) * 16,
            8_992_587_776
        );
        assert_eq!(a.ring_elements * 4 * 16, 146_997_248);
        assert_eq!(a.bytes * 16, 9_139_585_280);
        assert_eq!(
            KvFormat::Bf16.layout(262144, 4, 256)?.bytes * 16,
            17_179_869_184
        );
        assert!(KvFormat::Affine8.layout(usize::MAX, 4, 256).is_err());
        Ok(())
    }
    #[test]
    fn committed_complete_blocks_and_four_tentative_slots() -> Result<()> {
        let ring = KvFormat::Affine8.layout(262144, 4, 256)?.ring;
        for n in 0..16384 {
            let start = sealed_until(n);
            assert!(start.is_multiple_of(128) && start <= n.saturating_sub(2048));
            let slots = (start.max(64)..n + 4)
                .map(|p| ring_slot(p, ring))
                .collect::<Result<std::collections::BTreeSet<_>>>()?;
            assert_eq!(slots.len(), n + 4 - start.max(64).min(n + 4));
        }
        assert_eq!([2175, 2176, 2177].map(sealed_until), [0, 128, 128]);
        assert_eq!([2047, 2048, 2049].map(sealed_until), [0, 0, 0]);
        assert_eq!(ring_slot(63, ring)?, 63);
        assert!(ring_slot(64, 64).is_err());
        Ok(())
    }
}
