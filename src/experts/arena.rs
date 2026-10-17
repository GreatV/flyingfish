//! Host expert block arena with per-block read leases.
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Fixed-size host expert blocks. A block is written only while its lease count is zero, after
/// the caller has ordered the write behind that block's device copy event.
pub struct HostBlocks {
    data: UnsafeCell<Box<[bf16]>>,
    elements: usize,
    block_elements: usize,
    leases: Box<[AtomicU32]>,
}

// SAFETY: reads go through `read` under a lease and writes through `write` with no lease on the
// block; the data pointer is allocated once for the arena lifetime.
unsafe impl Send for HostBlocks {}
unsafe impl Sync for HostBlocks {}

/// Read lease over a set of host blocks; releasing it lets the owner rewrite those blocks.
pub struct BlockLease {
    arena: Arc<HostBlocks>,
    blocks: Vec<u32>,
}

impl Drop for BlockLease {
    fn drop(&mut self) {
        for &block in &self.blocks {
            self.arena.leases[block as usize].fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl HostBlocks {
    pub fn new(blocks: u32, block_elements: usize) -> Result<Arc<Self>> {
        ensure!(blocks > 0, "empty host block arena");
        ensure!(block_elements > 0, "zero host block size");
        let elements = usize::try_from(blocks)?
            .checked_mul(block_elements)
            .context("host arena extent overflow")?;
        Ok(Arc::new(Self {
            data: UnsafeCell::new(vec![bf16::ZERO; elements].into_boxed_slice()),
            elements,
            block_elements,
            leases: (0..blocks).map(|_| AtomicU32::new(0)).collect(),
        }))
    }

    pub fn blocks(&self) -> u32 {
        u32::try_from(self.leases.len()).expect("block count fits u32")
    }

    pub fn block_elements(&self) -> usize {
        self.block_elements
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn block_range(&self, block: u32) -> Result<Range<usize>> {
        ensure!(
            (block as usize) < self.leases.len(),
            "host block {block} outside {}",
            self.leases.len()
        );
        let start = block as usize * self.block_elements;
        Ok(start..start + self.block_elements)
    }

    /// Leases every listed block until the guard is dropped; duplicates are counted once.
    pub fn lease(self: &Arc<Self>, blocks: &[u32]) -> Result<BlockLease> {
        let mut unique = blocks.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for &block in &unique {
            ensure!(
                (block as usize) < self.leases.len(),
                "host block {block} outside {}",
                self.leases.len()
            );
            self.leases[block as usize].fetch_add(1, Ordering::AcqRel);
        }
        Ok(BlockLease {
            arena: self.clone(),
            blocks: unique,
        })
    }

    /// Reads a range of the arena; the caller must hold a lease covering every touched block.
    pub fn read(&self, range: Range<usize>) -> Result<&[bf16]> {
        ensure!(
            range.start <= range.end && range.end <= self.elements,
            "host read {range:?} outside {} elements",
            self.elements
        );
        // SAFETY: the lease excludes writers on every block the caller reads; see the type contract.
        let data: &[bf16] = unsafe { &*self.data.get() };
        Ok(&data[range])
    }

    /// Writes one whole block; requires zero outstanding leases on that block.
    pub fn write(&self, block: u32, values: &[bf16]) -> Result<()> {
        let range = self.block_range(block)?;
        ensure!(
            values.len() == range.len(),
            "host block {block} write has {} elements, expected {}",
            values.len(),
            range.len()
        );
        ensure!(
            self.leases[block as usize].load(Ordering::Acquire) == 0,
            "host block {block} is leased"
        );
        // SAFETY: no lease on this block; the caller has ordered the write behind the block's
        // device copy event.
        let data: &mut [bf16] = unsafe { &mut *self.data.get() };
        data[range].copy_from_slice(values);
        Ok(())
    }

    pub fn lease_count(&self, block: u32) -> Result<u32> {
        self.block_range(block)?;
        Ok(self.leases[block as usize].load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leased_block_rejects_writes_until_the_lease_drops() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        let values: Vec<bf16> = (0..4).map(|i| bf16::from_f32(i as f32)).collect();
        arena.write(0, &values)?;
        assert_eq!(arena.read(0..4)?, values.as_slice());
        let lease = arena.lease(&[0])?;
        assert_eq!(arena.lease_count(0)?, 1);
        let error = arena.write(0, &values).unwrap_err().to_string();
        assert!(error.contains("block 0 is leased"), "{error}");
        drop(lease);
        assert_eq!(arena.lease_count(0)?, 0);
        arena.write(0, &values)?;
        assert!(arena.write(3, &values).is_err());
        assert!(arena.write(1, &values[..3]).is_err());
        Ok(())
    }

    #[test]
    fn overlapping_leases_count_once_and_release_together() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        let a = arena.lease(&[1, 0, 1])?;
        assert_eq!(arena.lease_count(0)?, 1);
        assert_eq!(arena.lease_count(1)?, 1);
        let b = arena.lease(&[1])?;
        assert_eq!(arena.lease_count(1)?, 2);
        drop(a);
        assert_eq!(arena.lease_count(1)?, 1);
        drop(b);
        assert_eq!(arena.lease_count(1)?, 0);
        assert!(arena.read(0..9).is_err());
        Ok(())
    }
}
