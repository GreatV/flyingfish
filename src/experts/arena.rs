//! Native device-slot layout, slot event lifetimes and host expert blocks with read leases
//! and writer claims.
use crate::experts::{catalogue::ExpertGeometry, plan::Capacity};
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
        let bytes = elements
            .checked_mul(std::mem::size_of::<bf16>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .context("host arena byte extent overflow")?;
        let mut data = Vec::new();
        data.try_reserve_exact(elements).with_context(|| {
            format!("allocate host expert arena: blocks={blocks} bytes={bytes}")
        })?;
        data.resize(elements, bf16::ZERO);
        let data: Box<[bf16]> = data.into_boxed_slice();
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
        let _guard = self
            .claim
            .lock()
            .map_err(|error| anyhow::anyhow!("host block claim mutex poisoned: {error}"))?;
        for (index, &block) in unique.iter().enumerate() {
            let word = &self.words[block as usize];
            if let Err(current) = word.try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1).filter(|&count| count < WRITER)
            }) {
                for &held in &unique[..index] {
                    self.words[held as usize].fetch_sub(1, Ordering::AcqRel);
                }
                if current == WRITER {
                    anyhow::bail!("host block {block} is writer-claimed");
                }
                anyhow::bail!("host block {block} reader count overflow: count={current}");
            }
        }
        Ok(BlockLease {
            arena: self.clone(),
            blocks: unique,
        })
    }

    /// Claims one block exclusively; fails while it is leased or claimed.
    pub fn claim(self: &Arc<Self>, block: u32) -> Result<WriteClaim> {
        self.block_range(block)?;
        let _guard = self
            .claim
            .lock()
            .map_err(|error| anyhow::anyhow!("host block claim mutex poisoned: {error}"))?;
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

    #[cfg(test)]
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

/// Native expert slots in rank-major, layer-minor order, followed by persistent staging.
pub struct SlotLayout {
    geometry: ExpertGeometry,
    slot_bytes: usize,
    slots: Vec<Vec<Range<usize>>>,
    resident_bytes: usize,
    staging_slots: u32,
    host_blocks: u32,
}

impl SlotLayout {
    pub fn new(geometry: ExpertGeometry, capacity: &Capacity) -> Result<Self> {
        ensure!(
            geometry.layers > 0
                && geometry.experts > 0
                && geometry.hidden > 0
                && geometry.intermediate > 0
                && geometry.top_k > 0
                && geometry.top_k <= geometry.experts,
            "invalid slot geometry {geometry:?}"
        );
        let slot_bytes = usize::try_from(capacity.slot_bytes())?;
        let gate_up_bytes = capacity.layout.gate_up_bytes().unwrap_or(0);
        ensure!(
            slot_bytes.is_multiple_of(16) && gate_up_bytes.is_multiple_of(16),
            "native slot alignment: slot_bytes={slot_bytes} gate_up_bytes={gate_up_bytes}"
        );
        ensure!(
            capacity.slots_per_layer.len() == geometry.layers as usize,
            "slot layout layer count {} differs from {}",
            capacity.slots_per_layer.len(),
            geometry.layers
        );
        let resident_count = capacity
            .slots_per_layer
            .iter()
            .try_fold(0u64, |n, &count| {
                n.checked_add(u64::from(count))
                    .context("resident slot count overflow")
            })?;
        let resident_extent = resident_count
            .checked_mul(slot_bytes as u64)
            .filter(|&n| n <= isize::MAX as u64)
            .context("resident expert arena overflow")?;
        ensure!(
            resident_extent <= capacity.device_expert_bytes,
            "resident expert arena bytes={resident_extent} exceed budget={}",
            capacity.device_expert_bytes
        );
        let mut slots = Vec::new();
        slots.try_reserve_exact(geometry.layers as usize)?;
        for (layer, &count) in capacity.slots_per_layer.iter().enumerate() {
            ensure!(
                count > 0 && count <= geometry.experts,
                "slot layout layer={layer} slots={count} experts={}",
                geometry.experts
            );
            let mut layer_slots = Vec::new();
            layer_slots.try_reserve_exact(count as usize)?;
            slots.push(layer_slots);
        }
        let mut resident_bytes = 0usize;
        for rank in 0..capacity.slots_per_layer.iter().copied().max().unwrap_or(0) {
            for (layer, &count) in capacity.slots_per_layer.iter().enumerate() {
                if rank < count {
                    let end = resident_bytes
                        .checked_add(slot_bytes)
                        .context("resident expert arena overflow")?;
                    slots[layer].push(resident_bytes..end);
                    resident_bytes = end;
                }
            }
        }
        let host_blocks = capacity
            .host_blocks
            .checked_add(capacity.spare_blocks)
            .context("host data plus spare block count overflow")?;
        let host_bytes = u64::from(host_blocks)
            .checked_mul(slot_bytes as u64)
            .context("host expert arena overflow")?;
        ensure!(
            host_bytes <= capacity.host_expert_bytes,
            "host expert arena bytes={host_bytes} exceed budget={}",
            capacity.host_expert_bytes
        );
        let layout = Self {
            geometry,
            slot_bytes,
            slots,
            resident_bytes,
            staging_slots: capacity.prefill_staging_slots,
            host_blocks,
        };
        layout.allocation_bytes()?;
        Ok(layout)
    }

    pub fn allocation_bytes(&self) -> Result<usize> {
        self.slot_bytes
            .checked_mul(self.staging_slots as usize)
            .and_then(|n| n.checked_add(self.resident_bytes))
            .filter(|&n| n <= isize::MAX as usize)
            .context("device expert plus staging arena extent overflow")
    }

    pub fn geometry(&self) -> ExpertGeometry {
        self.geometry
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn host_blocks(&self) -> u32 {
        self.host_blocks
    }

    pub fn resident_slot(&self, layer: u32, rank: u32) -> Result<Range<usize>> {
        self.slots
            .get(layer as usize)
            .and_then(|slots| slots.get(rank as usize))
            .cloned()
            .with_context(|| format!("native slot outside layout: layer={layer} rank={rank}"))
    }

    /// Existing ranks of one layer.
    pub fn slot_count(&self, layer: u32) -> Result<u32> {
        self.slots
            .get(layer as usize)
            .map(|slots| slots.len() as u32)
            .with_context(|| format!("slot layout layer {layer} outside {}", self.slots.len()))
    }

    pub fn staging_slot(&self, slot: u32) -> Result<Range<usize>> {
        ensure!(
            slot < self.staging_slots,
            "staging slot={slot} outside count={}",
            self.staging_slots
        );
        let bytes = self.slot_bytes;
        let start = self.resident_bytes + slot as usize * bytes;
        Ok(start..start + bytes)
    }

    /// Borrowed storage excludes persistent staging and retains the first slot of every layer.
    pub fn borrowed_tail(&self, slots: u32) -> Result<Range<usize>> {
        let bytes = self.slot_bytes;
        let count = self.resident_bytes / bytes;
        ensure!(
            slots as usize <= count - self.geometry.layers as usize,
            "borrowed slots={slots} exceed resident count={count} minus layers={}",
            self.geometry.layers
        );
        Ok(self.resident_bytes - slots as usize * bytes..self.resident_bytes)
    }
}

/// Copy and reader events of one slot; the streams wait on them, placement never does.
pub struct SlotEvents {
    slots: Vec<SlotLifetime>,
}

#[derive(Clone, Copy, Default)]
struct SlotLifetime {
    copy: Option<u64>,
    reader: Option<u64>,
}

impl SlotEvents {
    pub fn new(slots: usize) -> Self {
        Self {
            slots: vec![SlotLifetime::default(); slots],
        }
    }

    fn index(&self, slot: u32) -> Result<usize> {
        let index = usize::try_from(slot).context("slot id exceeds usize")?;
        ensure!(
            index < self.slots.len(),
            "slot {slot} outside {} positions",
            self.slots.len()
        );
        Ok(index)
    }

    /// Records one copy event covering every listed slot; it starts their lifetimes.
    pub fn record_copy(&mut self, slots: &[u32], event: u64) -> Result<()> {
        for &slot in slots {
            self.index(slot)?;
        }
        for &slot in slots {
            let index = self.index(slot)?;
            let lifetime = &mut self.slots[index];
            lifetime.copy = Some(event);
            lifetime.reader = None;
        }
        Ok(())
    }

    /// Records one reader event covering every listed slot; the last event on the reader stream
    /// wins.
    pub fn record_readers(&mut self, slots: &[u32], event: u64) -> Result<()> {
        for &slot in slots {
            let index = self.index(slot)?;
            ensure!(
                self.slots[index].copy.is_some(),
                "slot {slot} has no recorded copy to read"
            );
        }
        for &slot in slots {
            let index = self.index(slot)?;
            self.slots[index].reader = Some(event);
        }
        Ok(())
    }

    /// Copy event the reader stream waits on for the first read of a slot.
    pub fn copy_event(&self, slot: u32) -> Result<Option<u64>> {
        Ok(self.slots[self.index(slot)?].copy)
    }

    /// Returns `[last copy event, last reader event]` and clears the slot's lifetime.
    pub fn reclaim(&mut self, slot: u32) -> Result<[Option<u64>; 2]> {
        let index = self.index(slot)?;
        let lifetime = &mut self.slots[index];
        let events = [lifetime.copy, lifetime.reader];
        *lifetime = SlotLifetime::default();
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn slot_capacity(geometry: ExpertGeometry, slots_per_layer: Vec<u32>) -> Result<Capacity> {
        let layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?;
        let bytes = layout.slot_bytes();
        Ok(Capacity {
            layout,
            span_rows: 8,
            wave_rows: 2,
            device_expert_bytes: slots_per_layer.iter().map(|&n| u64::from(n)).sum::<u64>() * bytes,
            host_expert_bytes: 3 * bytes,
            slots_per_layer,
            host_blocks: 2,
            spare_blocks: 1,
            disk_bytes: 0,
            worker_count: 1,
            disk_workers: 0,
            prefill_staging_slots: 2,
            pinned: false,
        })
    }

    #[test]
    fn slot_layout_matches_ledger_tail_with_uneven_layers() -> Result<()> {
        let geometry = ExpertGeometry {
            layers: 3,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        };
        let capacity = slot_capacity(geometry, vec![3, 2, 2])?;
        let layout = SlotLayout::new(geometry, &capacity)?;
        let bytes = layout.slot_bytes();
        let order = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)];
        for (physical, &(layer, rank)) in order.iter().enumerate() {
            assert_eq!(
                layout.resident_slot(layer, rank)?,
                physical * bytes..(physical + 1) * bytes
            );
        }
        assert!(layout.resident_slot(1, 2).is_err());
        assert_eq!(layout.staging_slot(0)?, 7 * bytes..8 * bytes);
        assert_eq!(layout.staging_slot(1)?, 8 * bytes..9 * bytes);
        assert!(layout.staging_slot(2).is_err());
        assert_eq!(layout.allocation_bytes()?, 9 * bytes);
        assert_eq!(layout.borrowed_tail(0)?, 7 * bytes..7 * bytes);
        assert_eq!(layout.borrowed_tail(2)?, 5 * bytes..7 * bytes);
        assert_eq!(layout.borrowed_tail(4)?, 3 * bytes..7 * bytes);
        assert!(layout.borrowed_tail(5).is_err());
        let initial = vec![vec![4, 1, 7], vec![2, 6], vec![5, 0]];
        let mut ledger = crate::experts::plan::TierLedger::new(layout, 0, &initial)?;
        let borrowed = ledger.borrow_tail(2, 0)?;
        assert_eq!(borrowed.experts, vec![(0, 7), (2, 0)]);
        assert_eq!(borrowed.disk_bytes, 2 * bytes as u64);
        ledger.check()?;
        Ok(())
    }

    #[test]
    fn native_slot_layout_prices_spares_and_rejects_out_of_bounds_extents() -> Result<()> {
        let geometry = ExpertGeometry {
            layers: 40,
            experts: 256,
            hidden: 2048,
            intermediate: 512,
            top_k: 8,
        };
        let mut capacity = slot_capacity(geometry, crate::experts::plan::equal_slots(2701, 40)?)?;
        let bytes = capacity.slot_bytes() as usize;
        assert_eq!(bytes, 6_291_456);
        assert_eq!(capacity.layout.gate_up_bytes(), Some(4_194_304));
        assert_eq!(bytes as u64 - 4_194_304, 2_097_152);
        let layout = SlotLayout::new(geometry, &capacity)?;
        assert_eq!(layout.resident_slot(20, 67)?, 2700 * bytes..2701 * bytes);
        assert!(layout.resident_slot(21, 67).is_err());
        assert_eq!(layout.borrowed_tail(429)?, 2272 * bytes..2701 * bytes);
        assert_eq!(layout.staging_slot(0)?.start, 2701 * bytes);
        assert_eq!(layout.allocation_bytes()?, 2703 * bytes);
        capacity.device_expert_bytes -= 1;
        assert!(SlotLayout::new(geometry, &capacity).is_err());
        capacity.device_expert_bytes += 1;
        capacity.host_expert_bytes -= 1;
        assert!(SlotLayout::new(geometry, &capacity).is_err());
        capacity.host_expert_bytes += 1;
        capacity.spare_blocks = u32::MAX;
        assert!(SlotLayout::new(geometry, &capacity).is_err());
        assert!(HostBlocks::new(2, usize::MAX / 2).is_err());
        Ok(())
    }

    #[test]
    fn slot_events_keep_one_copy_and_one_reader_event() -> Result<()> {
        let mut events = SlotEvents::new(3);
        events.record_copy(&[0, 1, 2], 7)?;
        for slot in 0..3u32 {
            assert_eq!(events.copy_event(slot)?, Some(7));
        }
        for event in 8..1_008u64 {
            events.record_readers(&[0, 1, 2], event)?;
        }
        assert_eq!(events.reclaim(1)?, [Some(7), Some(1_007)]);
        assert_eq!(events.copy_event(1)?, None);
        assert_eq!(
            std::mem::size_of::<SlotLifetime>(),
            2 * std::mem::size_of::<Option<u64>>()
        );
        events.record_copy(&[0], 11)?;
        let error = events.record_readers(&[0, 9], 12).unwrap_err().to_string();
        assert!(error.contains("slot 9 outside"), "{error}");
        assert_eq!(
            events.reclaim(0)?,
            [Some(11), None],
            "a rejected batch leaves its valid slots unchanged"
        );
        let error = events.record_readers(&[1], 13).unwrap_err().to_string();
        assert!(error.contains("has no recorded copy to read"), "{error}");
        assert_eq!(events.reclaim(2)?, [Some(7), Some(1_007)]);
        Ok(())
    }

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
    fn concurrent_lease_release_preserves_the_reader_count() -> Result<()> {
        let arena = HostBlocks::new(1, 4)?;
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| -> Result<()> {
            let arena = &arena;
            let barrier = &barrier;
            let releasing = scope.spawn(move || -> Result<()> {
                for delay in 0..100_000 {
                    let lease = arena.lease(&[0])?;
                    barrier.wait();
                    for _ in 0..delay % 64 {
                        std::hint::spin_loop();
                    }
                    drop(lease);
                    barrier.wait();
                }
                Ok(())
            });
            for _ in 0..100_000 {
                barrier.wait();
                let lease = arena.lease(&[0])?;
                drop(lease);
                barrier.wait();
            }
            releasing
                .join()
                .map_err(|_| anyhow::anyhow!("lease-release thread panicked"))??;
            Ok(())
        })?;
        assert_eq!(arena.lease_count(0)?, 0);
        let claim = arena.claim(0)?;
        drop(claim);
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
