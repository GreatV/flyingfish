use anyhow::{Result, ensure};

/// Weight storage format of a checkpoint, as its config declares it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantFormat {
    GroupAffine {
        group: usize,
    },
    BlockFp8 {
        block: usize,
    },
    /// 16-bit dtype-native weights.
    Bf16,
    F16,
}

impl QuantFormat {
    /// Dtype-native 16-bit storage: the value is carried, not packed.
    pub fn is_16bit(self) -> bool {
        matches!(self, Self::Bf16 | Self::F16)
    }
}

impl QuantFormat {
    /// FP8 with `block`-square scale blocks; a non-square block is an error.
    pub fn block_fp8(block: [usize; 2]) -> Result<Self> {
        ensure!(
            block[0] == block[1] && block[0] > 0,
            "FP8 weight block {block:?} is not square"
        );
        Ok(Self::BlockFp8 { block: block[0] })
    }
}

#[cfg(test)]
mod tests {
    use super::QuantFormat;

    #[test]
    fn square_blocks_only() {
        assert_eq!(
            QuantFormat::block_fp8([128, 128]).unwrap(),
            QuantFormat::BlockFp8 { block: 128 }
        );
        assert!(QuantFormat::block_fp8([64, 128]).is_err());
        assert!(QuantFormat::block_fp8([0, 0]).is_err());
    }
}
