//! Host expert block arena with per-block read leases and writer claims.
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Lease word value of an exclusive writer claim; any other value is a reader count.
const WRITER: u32 = u32::MAX;

/// Fixed-size host expert blocks. A block is written only under an exclusive writer claim, after
/// the caller has ordered the write behind that block's device copy event. The allocation is owned
/// as a raw pointer; references are formed per block only.
pub struct HostBlocks {
    base: *mut bf16,
    elements: usize,
    block_elements: usize,
    words: Box<[AtomicU32]>,
    claim: Mutex<()>,
}

// SAFETY: reads go through a lease on the block and writes through a writer claim on the block;
// the base pointer is allocated once for the arena lifetime.
unsafe impl Send for HostBlocks {}
unsafe impl Sync for HostBlocks {}

impl Drop for HostBlocks {
    fn drop(&mut self) {
        // SAFETY: the allocation came from Box::into_raw in `new` and is freed exactly once.
        drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.base, self.elements))
        });
    }
}

/// Read lease over a set of host blocks; releasing it lets the owner claim those blocks.
pub struct BlockLease {
    arena: Arc<HostBlocks>,
    blocks: Vec<u32>,
}

impl Drop for BlockLease {
    fn drop(&mut self) {
        for &block in &self.blocks {
            self.arena.words[block as usize].fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl BlockLease {
    pub fn block_elements(&self) -> usize {
        self.arena.block_elements
    }

    /// Whole leased block; the slice is tied to this guard.
    pub fn read(&self, block: u32) -> Result<&[bf16]> {
        ensure!(
            self.blocks.binary_search(&block).is_ok(),
            "host block {block} is not leased"
        );
        self.arena.block_slice(block)
    }
}

/// Exclusive writer claim over one host block; exposes only that block.
pub struct WriteClaim {
    arena: Arc<HostBlocks>,
    block: u32,
}

impl Drop for WriteClaim {
    fn drop(&mut self) {
        self.arena.words[self.block as usize].store(0, Ordering::Release);
    }
}

impl WriteClaim {
    /// Host slice of the claimed block.
    pub fn slice(&mut self) -> Result<&mut [bf16]> {
        let range = self.arena.block_range(self.block)?;
        // SAFETY: this guard holds the block's exclusive writer claim and the reference covers
        // only the claimed block.
        Ok(
            unsafe {
                std::slice::from_raw_parts_mut(self.arena.base.add(range.start), range.len())
            },
        )
    }

    /// Host address of the claimed block for a D2H copy.
    pub fn pointer(&self) -> Result<u64> {
        let range = self.arena.block_range(self.block)?;
        Ok(unsafe { self.arena.base.add(range.start) } as u64)
    }
}

impl HostBlocks {
    pub fn new(blocks: u32, block_elements: usize) -> Result<Arc<Self>> {
        ensure!(blocks > 0, "empty host block arena");
        ensure!(block_elements > 0, "zero host block size");
        let elements = usize::try_from(blocks)?
            .checked_mul(block_elements)
            .context("host arena extent overflow")?;
        let data: Box<[bf16]> = vec![bf16::ZERO; elements].into_boxed_slice();
        let base = Box::into_raw(data) as *mut bf16;
        Ok(Arc::new(Self {
            base,
            elements,
            block_elements,
            words: (0..blocks).map(|_| AtomicU32::new(0)).collect(),
            claim: Mutex::new(()),
        }))
    }

    pub fn blocks(&self) -> u32 {
        u32::try_from(self.words.len()).expect("block count fits u32")
    }

    pub fn block_elements(&self) -> usize {
        self.block_elements
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Base host address and byte length for the setup-time page-lock registration.
    pub fn host_span(&self) -> (u64, usize) {
        (
            self.base as u64,
            self.elements * std::mem::size_of::<bf16>(),
        )
    }

    pub fn block_range(&self, block: u32) -> Result<Range<usize>> {
        ensure!(
            (block as usize) < self.words.len(),
            "host block {block} outside {}",
            self.words.len()
        );
        let start = block as usize * self.block_elements;
        Ok(start..start + self.block_elements)
    }

    /// Leases every listed block until the guard is dropped; duplicates are counted once.
    pub fn lease(self: &Arc<Self>, blocks: &[u32]) -> Result<BlockLease> {
        for &block in blocks {
            self.block_range(block)?;
        }
        let mut unique = blocks.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let _guard = self.claim.lock().unwrap_or_else(|e| e.into_inner());
        for (index, &block) in unique.iter().enumerate() {
            let word = &self.words[block as usize];
            let current = word.load(Ordering::Acquire);
            if current == WRITER {
                for &held in &unique[..index] {
                    self.words[held as usize].fetch_sub(1, Ordering::AcqRel);
                }
                anyhow::bail!("host block {block} is writer-claimed");
            }
            word.store(current + 1, Ordering::Release);
        }
        Ok(BlockLease {
            arena: self.clone(),
            blocks: unique,
        })
    }

    /// Claims one block exclusively; fails while it is leased or claimed.
    pub fn claim(self: &Arc<Self>, block: u32) -> Result<WriteClaim> {
        self.block_range(block)?;
        let _guard = self.claim.lock().unwrap_or_else(|e| e.into_inner());
        self.words[block as usize]
            .compare_exchange(0, WRITER, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|current| {
                anyhow::anyhow!("host block {block} is busy (word {current}, writer {WRITER})")
            })?;
        Ok(WriteClaim {
            arena: self.clone(),
            block,
        })
    }

    pub fn lease_count(&self, block: u32) -> Result<u32> {
        self.block_range(block)?;
        Ok(self.words[block as usize].load(Ordering::Acquire))
    }

    fn block_slice(&self, block: u32) -> Result<&[bf16]> {
        let range = self.block_range(block)?;
        // SAFETY: the caller holds a lease on this block; the reference covers only this block.
        Ok(unsafe { std::slice::from_raw_parts(self.base.add(range.start), range.len()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read after the lease drops does not compile: `BlockLease::read` borrows the guard.
    #[test]
    fn a_claim_fails_while_the_block_is_leased() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        let lease = arena.lease(&[0])?;
        let Err(error) = arena.claim(0) else {
            panic!("claim succeeded while leased");
        };
        let error = error.to_string();
        assert!(error.contains("host block 0 is busy"), "{error}");
        assert_eq!(arena.lease_count(0)?, 1);
        drop(lease);
        let mut claim = arena.claim(0)?;
        let values: Vec<bf16> = (0..4).map(|i| bf16::from_f32(i as f32)).collect();
        claim.slice()?.copy_from_slice(&values);
        drop(claim);
        let lease = arena.lease(&[0])?;
        assert_eq!(lease.read(0)?, values.as_slice());
        Ok(())
    }

    #[test]
    fn a_lease_fails_while_the_block_is_claimed() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        let claim = arena.claim(1)?;
        let Err(error) = arena.lease(&[0, 1]) else {
            panic!("lease succeeded while claimed");
        };
        let error = error.to_string();
        assert!(error.contains("block 1 is writer-claimed"), "{error}");
        assert_eq!(arena.lease_count(0)?, 0);
        drop(claim);
        let lease = arena.lease(&[0, 1])?;
        assert_eq!(lease.read(1)?.len(), 4);
        assert!(lease.read(0).is_ok());
        Ok(())
    }

    #[test]
    fn invalid_indices_leave_every_count_unchanged() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        assert!(arena.lease(&[0, 5]).is_err());
        assert_eq!(arena.lease_count(0)?, 0);
        assert_eq!(arena.lease_count(1)?, 0);
        assert!(arena.claim(9).is_err());
        let lease = arena.lease(&[1, 0, 1])?;
        assert_eq!(arena.lease_count(0)?, 1);
        assert_eq!(arena.lease_count(1)?, 1);
        assert!(lease.read(0).is_ok());
        assert!(lease.read(1).is_ok());
        drop(lease);
        assert_eq!(arena.lease_count(0)?, 0);
        assert_eq!(arena.lease_count(1)?, 0);
        Ok(())
    }

    #[test]
    fn a_claim_exposes_only_its_block() -> Result<()> {
        let arena = HostBlocks::new(2, 4)?;
        let mut claim = arena.claim(0)?;
        assert_eq!(claim.slice()?.len(), 4);
        assert!(claim.pointer()? > 0);
        let second = arena.claim(1)?;
        assert!(arena.claim(0).is_err());
        drop(second);
        drop(claim);
        let other = arena.claim(1)?;
        drop(other);
        Ok(())
    }
}
