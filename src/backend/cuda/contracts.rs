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
