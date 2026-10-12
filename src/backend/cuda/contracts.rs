//! Non-owning operator views. Engine owns storage through completion of all launches.
use anyhow::{Context, Result, ensure};
use std::{marker::PhantomData, mem::size_of};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Tensor<T, const R: usize> {
    pub ptr: u64,
    pub shape: [u32; R],
    pub stride: [u64; R],
    pub span_bytes: u64,
    marker: PhantomData<T>,
}
#[derive(Debug)]
pub(crate) struct TensorMut<T, const R: usize>(pub Tensor<T, R>);
impl<T: Copy, const R: usize> Tensor<T, R> {
    /// # Safety
    /// Pointer must address storage on this context, alive until all uses complete.
    /// The caller must order writes/reads and retain the allocation for captured graphs.
    pub unsafe fn new(
        ptr: u64,
        shape: [u32; R],
        stride: [u64; R],
        span_bytes: u64,
    ) -> Result<Self> {
        let view = Self {
            ptr,
            shape,
            stride,
            span_bytes,
            marker: PhantomData,
        };
        view.validate()?;
        Ok(view)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.ptr != 0 && self.ptr.is_multiple_of(16),
            "unaligned/null tensor pointer"
        );
        ensure!(R > 0 && size_of::<T>() > 0, "empty tensor type");
        let mut last = 0u64;
        for (&n, &s) in self.shape.iter().zip(&self.stride) {
            ensure!(n > 0 && s > 0, "zero tensor dimension/stride");
            last = last
                .checked_add(
                    u64::from(n - 1)
                        .checked_mul(s)
                        .context("tensor stride overflow")?,
                )
                .context("tensor extent overflow")?;
        }
        let bytes = last
            .checked_add(1)
            .and_then(|v| v.checked_mul(size_of::<T>() as u64))
            .context("tensor bytes overflow")?;
        ensure!(bytes <= self.span_bytes, "tensor span too small");
        self.ptr
            .checked_add(self.span_bytes)
            .context("tensor address overflow")?;
        Ok(())
    }

    /// Narrow the logical leading dimension without relocating physical storage.
    pub fn prefix_rows(self, rows: u32) -> Result<Self> {
        self.validate()?;
        ensure!(
            rows > 0 && rows <= self.shape[0],
            "rows outside tensor view"
        );
        let mut view = self;
        view.shape[0] = rows;
        Ok(view)
    }
}
impl<T: Copy, const R: usize> TensorMut<T, R> {
    /// # Safety
    /// In addition to Tensor::new, writes must be exclusive, except explicitly allowed exact aliases.
    pub unsafe fn new(
        ptr: u64,
        shape: [u32; R],
        stride: [u64; R],
        span_bytes: u64,
    ) -> Result<Self> {
        Ok(Self(unsafe {
            Tensor::new(ptr, shape, stride, span_bytes)?
        }))
    }
    pub fn read(&self) -> Tensor<T, R> {
        self.0
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeqRef {
    pub ptr: u64,
}
impl SeqRef {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.ptr != 0 && self.ptr.is_multiple_of(16),
            "invalid sequence metadata pointer"
        );
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowDomain {
    pub meta: SeqRef,
    pub row_base: u32,
    pub row_capacity: u32,
}
impl RowDomain {
    pub fn validate(self, rows: u32) -> Result<()> {
        self.meta.validate()?;
        ensure!(
            self.row_capacity > 0
                && self
                    .row_base
                    .checked_add(self.row_capacity)
                    .is_some_and(|x| x <= rows),
            "row domain outside tensor"
        );
        Ok(())
    }
}
#[derive(Debug)]
pub(crate) struct Scratch {
    pub ptr: u64,
    pub bytes: u64,
}
impl Scratch {
    pub fn validate(&self, bytes: u64) -> Result<()> {
        ensure!(
            self.ptr != 0 && self.ptr.is_multiple_of(256) && self.bytes >= bytes,
            "scratch alignment/capacity mismatch"
        );
        self.ptr
            .checked_add(self.bytes)
            .context("scratch address overflow")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn prefix_rows_preserves_physical_projection_offsets() -> Result<()> {
        // Synthetic addresses are used only for CPU layout validation.
        let workspace = 2048u64;
        for (offset, width) in [(0, 12288), (12288, 1024), (13312, 1024), (48, 48)] {
            let full = Tensor::<bf16, 2> {
                ptr: 0x1000 + workspace * offset * 2,
                shape: [workspace as u32, width as u32],
                stride: [width, 1],
                span_bytes: workspace * width * 2,
                marker: PhantomData,
            };
            for rows in 1..=4 {
                let small = full.prefix_rows(rows)?;
                small.validate()?;
                assert_eq!(small.ptr, full.ptr);
                assert_eq!(small.shape, [rows, width as u32]);
                assert_eq!(small.stride, full.stride);
                assert_eq!(small.span_bytes, full.span_bytes);
            }
            assert!(full.prefix_rows(0).is_err());
            assert!(full.prefix_rows(workspace as u32 + 1).is_err());
            assert!(full.prefix_rows(2)?.prefix_rows(3).is_err());
        }
        Ok(())
    }
}
