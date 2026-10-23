//! Non-owning operator views. Engine owns storage through completion of all launches.
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;
use std::{marker::PhantomData, mem::size_of};

pub(crate) fn view<T: cudarc::driver::DeviceRepr + Copy, const R: usize>(
    b: &CudaSlice<T>,
    s: &Arc<CudaStream>,
    offset: usize,
    shape: [u32; R],
    stride: [u64; R],
) -> Result<Tensor<T, R>> {
    ensure!(offset < b.len(), "view offset outside allocation");
    let elements = shape
        .iter()
        .zip(&stride)
        .try_fold(1u64, |extent, (&n, &step)| {
            ensure!(n > 0, "zero view dimension");
            extent
                .checked_add(
                    u64::from(n - 1)
                        .checked_mul(step)
                        .context("view stride overflow")?,
                )
                .context("view extent overflow")
        })?;
    ensure!(
        elements <= (b.len() - offset) as u64,
        "view exceeds allocation"
    );
    let bytes = elements
        .checked_mul(size_of::<T>() as u64)
        .context("view byte overflow")?;
    let (ptr, _read) = b.device_ptr(s);
    unsafe { Tensor::new(ptr + (offset * size_of::<T>()) as u64, shape, stride, bytes) }
}

pub(crate) fn dense_layout<const R: usize>(shape: [u32; R]) -> Result<([u64; R], u64)> {
    ensure!(R > 0, "empty tensor rank");
    let mut stride = [1; R];
    let mut count = 1u64;
    for axis in (0..R).rev() {
        ensure!(shape[axis] > 0, "zero tensor dimension");
        stride[axis] = count;
        count = count
            .checked_mul(u64::from(shape[axis]))
            .context("dense extent overflow")?;
    }
    Ok((stride, count))
}

pub(crate) fn spans_disjoint(a_ptr: u64, a_bytes: u64, b_ptr: u64, b_bytes: u64) -> Result<bool> {
    let a_end = a_ptr
        .checked_add(a_bytes)
        .context("first span address overflow")?;
    let b_end = b_ptr
        .checked_add(b_bytes)
        .context("second span address overflow")?;
    Ok(a_end <= b_ptr || b_end <= a_ptr)
}

pub(crate) fn no_overlap<T: Copy, U: Copy, const R: usize, const S: usize>(
    a: &Tensor<T, R>,
    b: &Tensor<U, S>,
    aname: &str,
    bname: &str,
) -> Result<()> {
    ensure!(
        spans_disjoint(a.ptr, a.span_bytes, b.ptr, b.span_bytes)?,
        "tensors overlap: {aname} vs {bname} (0x{:x}+0x{:x}, 0x{:x}+0x{:x})",
        a.ptr,
        a.span_bytes,
        b.ptr,
        b.span_bytes
    );
    Ok(())
}

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
impl Tensor<u32, 1> {
    pub fn element_ptr(self, index: usize) -> Result<u64> {
        self.validate()?;
        ensure!(
            self.stride == [1] && index < self.shape[0] as usize,
            "u32 element outside contiguous tensor"
        );
        self.ptr
            .checked_add((index * size_of::<u32>()) as u64)
            .context("u32 element address overflow")
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
            "scratch alignment/capacity mismatch: ptr={:#x} bytes={} required={bytes}",
            self.ptr,
            self.bytes
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
    fn scalar_word_offsets_preserve_aligned_tensor_storage() -> Result<()> {
        for (words, index) in [(2, 1), (4, 1), (8, 7)] {
            let base =
                unsafe { Tensor::<u32, 1>::new(0x1000, [words], [1], u64::from(words) * 4)? };
            let ptr = base.element_ptr(index)?;
            assert_eq!(ptr, 0x1000 + index as u64 * 4);
            assert_eq!(ptr % 4, 0);
            assert!(unsafe { Tensor::<u32, 1>::new(ptr, [1], [1], 4) }.is_err());
            assert!(base.element_ptr(words as usize).is_err());
        }
        Ok(())
    }

    #[test]
    fn prefix_rows_preserves_physical_projection_offsets() -> Result<()> {
        assert_eq!(dense_layout([3, 5, 7])?, ([35, 7, 1], 105));
        assert!(dense_layout([3, 0]).is_err());
        assert!(dense_layout([u32::MAX; 3]).is_err());
        assert!(dense_layout([]).is_err());
        for (a, b, expected) in [
            ((0x1000, 32), (0x1020, 16), true),
            ((0x1020, 16), (0x1000, 32), true),
            ((0x1000, 32), (0x1010, 16), false),
            ((0x1000, 32), (0x1000, 32), false),
        ] {
            assert_eq!(spans_disjoint(a.0, a.1, b.0, b.1)?, expected);
        }
        assert!(spans_disjoint(u64::MAX - 7, 16, 0x1000, 32).is_err());
        assert!(spans_disjoint(0x1000, 32, u64::MAX - 7, 16).is_err());
        let strided = unsafe { Tensor::<bf16, 2>::new(0x1000, [3, 5], [8, 1], 42)? };
        assert_eq!(strided.prefix_rows(2)?.span_bytes, 42);
        assert!(unsafe { Tensor::<bf16, 2>::new(0x1000, [3, 5], [8, 1], 40) }.is_err());
        let (stride, count) = dense_layout([3, 5])?;
        for span in [count * 2, 64] {
            let dense = unsafe { Tensor::<bf16, 2>::new(0x1000, [3, 5], stride, span)? };
            assert_eq!(dense.prefix_rows(2)?.span_bytes, span);
            assert!(no_overlap(&dense, &dense, "input", "output").is_err());
            let next = unsafe { Tensor::<bf16, 2>::new(0x1040, [1, 5], [5, 1], 10)? };
            no_overlap(&dense, &next, "input", "output")?;
        }
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
