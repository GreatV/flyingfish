//! Decode residency runtime: host-tier load, request-boundary moves, dispatch, admission and
//! device-side slot events.
use crate::cpu::expert_ffn::Assignment;
use crate::experts::arena::{HostBlocks, SlotEvents, SlotLayout};
use crate::experts::catalogue::{ExpertCatalogue, ExpertGeometry};
use crate::experts::disk::DiskReader;
use crate::experts::plan::{
    Admission, Borrow, Capacity, F_DECAY, THETA_MIN, Tier, TierLedger, initial_residency,
    seed_scores,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

/// Block id of a host expert without a host block.
pub(crate) const NO_BLOCK: u32 = u32::MAX;

/// Setup sources of the host-tier load.
pub struct LoadInputs<'a> {
    pub geometry: ExpertGeometry,
    pub capacity: Capacity,
    pub admission: Admission,
    pub max_rows: u32,
    /// Initial GPU set from the planner's static prefill residency.
    pub initial: &'a [Vec<u32>],
    /// Host block arena; required exactly when the plan allocates host blocks.
    pub host: Option<&'a Arc<HostBlocks>>,
    pub catalogue: &'a ExpertCatalogue,
    pub reader: &'a DiskReader,
}

/// Host-tier load cost at setup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadReport {
    pub host_bytes: u64,
    pub seconds: f64,
}

/// One routed (row, choice, expert, weight) entry of a layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    pub row: u32,
    pub choice: u32,
    pub expert: u32,
    pub weight: bf16,
}

/// Per-layer dispatch: resident hits with their generation snapshots, host and disk misses.
/// Buffers are allocated once in `Residency::load` and refilled in place.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    /// 1 for a GPU-resident hit, 0 for a miss; index = route order.
    pub active: Vec<u32>,
    /// Published generation per route; meaningful only for hits.
    pub generation: Vec<u64>,
    /// Host misses; `row` indexes `miss_rows`, `expert` indexes `host_experts`.
    pub host: Vec<Assignment>,
    /// Distinct host experts in compacted order.
    pub host_experts: Vec<u32>,
    /// Disk misses; `row` indexes `miss_rows`, `expert` is the logical expert id.
    pub disk: Vec<Route>,
    /// Distinct rows with any miss, ascending.
    pub miss_rows: Vec<u32>,
}

/// Where a prompt-boundary move reads its candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveSource {
    Host(u32),
    Disk,
}

/// One expert installation of a request boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromptMove {
    pub layer: u32,
    pub expert: u32,
    /// Resident expert whose slot the candidate takes; `None` installs into an empty slot.
    pub victim: Option<u32>,
    pub source: MoveSource,
    /// D2H destination of the victim, written before the candidate copy; `None` drops it to disk.
    pub dest_block: Option<u32>,
    /// Event the D2H into `dest_block` waits on; `None` for a chained destination.
    pub dest_wait: Option<u64>,
    /// Slot of the preceding host swap whose H2D event a chained destination waits on.
    pub chain_slot: Option<u32>,
    /// Physical slot the candidate takes (SlotLayout id).
    pub slot: u32,
    pub generation: u64,
}

/// One admission published for the next token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Published {
    pub layer: u32,
    pub expert: u32,
    pub victim: u32,
    pub from_host: bool,
    /// Candidate's host block, the H2D source; `None` for a disk candidate.
    pub source_block: Option<u32>,
    /// D2H destination of the victim, written before the candidate copy; `None` drops it to disk.
    pub dest_block: Option<u32>,
    /// Event the D2H into `dest_block` waits on.
    pub dest_wait: Option<u64>,
    /// Physical slot the candidate takes (SlotLayout id).
    pub slot: u32,
    pub generation: u64,
}

/// A vacated host block and the copy event captured for its reuse.
#[derive(Clone, Copy, Debug)]
struct ReleasedBlock {
    block: u32,
    slot: u32,
    event: Option<u64>,
}

/// Free host block and the event its next D2H destination waits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeBlock {
    id: u32,
    wait: Option<u64>,
}

/// Destination kind of a planned move before block ids are popped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dest {
    Pool,
    Chain,
    Disk,
    None,
}

#[derive(Clone, Copy, Debug)]
struct PlannedMove {
    layer: u32,
    expert: u32,
    victim: Option<u32>,
    tier: Tier,
    source_block: u32,
    dest: Dest,
}

/// Decode residency state: one request boundary, decode tokens, admission per token.
pub struct Residency {
    geometry: ExpertGeometry,
    capacity: Capacity,
    pub admission: Admission,
    max_rows: u32,
    ledger: TierLedger,
    events: SlotEvents,
    host_block: Vec<Vec<u32>>,
    reserve: u32,
    free_blocks: VecDeque<FreeBlock>,
    /// Vacated candidate blocks; `record_copy` captures the event that gates their reuse.
    released: Vec<ReleasedBlock>,
    generation: Vec<Vec<u64>>,
    next_generation: u64,
    scores: Vec<Vec<f64>>,
    routed: Vec<Vec<bool>>,
    missed: Vec<Vec<u32>>,
    dispatches: Vec<Dispatch>,
    out: Vec<u32>,
    incoming: Vec<u32>,
    victims: Vec<u32>,
    candidates: Vec<u32>,
    pending: Vec<(f64, u32, u32, u32)>,
    block_needs: Vec<bool>,
    plan: Vec<PlannedMove>,
    published: Vec<Published>,
    moves: Vec<PromptMove>,
    dispatched: Vec<bool>,
    token_open: bool,
    request_open: bool,
}

fn compact_row(miss_rows: &[u32], row: u32) -> Result<u32> {
    u32::try_from(
        miss_rows
            .binary_search(&row)
            .map_err(|_| anyhow::anyhow!("miss row {row} is not compacted"))?,
    )
    .context("miss row index exceeds u32")
}

fn fresh_dispatch(max_routes: usize, experts: usize, max_rows: usize) -> Dispatch {
    Dispatch {
        active: Vec::with_capacity(max_routes),
        generation: Vec::with_capacity(max_routes),
        host: Vec::with_capacity(max_routes),
        host_experts: Vec::with_capacity(experts),
        disk: Vec::with_capacity(max_routes),
        miss_rows: Vec::with_capacity(max_rows),
    }
}

fn host_fill_order(
    initial: &[Vec<u32>],
    geometry: &ExpertGeometry,
    backed_layers: u32,
    peer_complete: bool,
    host_blocks: u32,
) -> Vec<(u32, u32)> {
    let routed = u64::from(geometry.layers) * u64::from(geometry.experts);
    let swappable = u64::from(backed_layers) * u64::from(geometry.experts);
    let full_routed = u64::from(host_blocks) == routed;
    let full_swappable = peer_complete && u64::from(host_blocks) == swappable;
    let mut per_layer: Vec<Vec<u32>> = Vec::with_capacity(geometry.layers as usize);
    for (layer, set) in initial.iter().enumerate() {
        let inclusive = full_routed || (full_swappable && (layer as u32) < backed_layers);
        per_layer.push(
            (0..geometry.experts)
                .filter(|expert| inclusive || !set.contains(expert))
                .collect(),
        );
    }
    let mut order = Vec::with_capacity(host_blocks as usize);
    let mut rank = 0usize;
    'ranks: loop {
        let mut placed = false;
        for (layer, experts) in per_layer.iter().enumerate() {
            let Some(&expert) = experts.get(rank) else {
                continue;
            };
            if order.len() == host_blocks as usize {
                break 'ranks;
            }
            order.push((layer as u32, expert));
            placed = true;
        }
        if !placed {
            break;
        }
        rank += 1;
    }
    order
}

impl Residency {
    #[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
    pub(crate) fn freeze_probe_admissions(&mut self) -> Result<()> {
        ensure!(!self.token_open, "placement probe has an open token");
        self.admission.budget_bytes_per_token = 0;
        Ok(())
    }

    /// Setup: installs the initial GPU set, fills every host data block from the native ranges
    /// through the disk reader, and reports the host-tier load cost. The request boundary only
    /// changes the GPU set.
    pub fn load(inputs: LoadInputs<'_>) -> Result<(Self, LoadReport)> {
        let LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows,
            initial,
            host,
            catalogue,
            reader,
        } = inputs;
        let started = Instant::now();
        ensure!(max_rows > 0, "zero dispatch row bound");
        ensure!(
            capacity.slots_per_layer.len() == geometry.layers as usize,
            "slot plan layer count mismatch"
        );
        ensure!(
            initial.len() == geometry.layers as usize,
            "initial set layer count mismatch"
        );
        let slot_bytes = capacity.slot_bytes();
        let cap = admission.hard_cap(slot_bytes)?;
        let routed = u64::from(geometry.layers) * u64::from(geometry.experts);
        let peer_complete = capacity.slots_per_layer[capacity.gpu0_layers as usize..]
            .iter()
            .all(|&slots| slots == geometry.experts);
        let swappable = u64::from(capacity.gpu0_layers) * u64::from(geometry.experts);
        let reserve = if u64::from(capacity.slots_total()) == routed
            || u64::from(capacity.host_blocks) == routed
            || (peer_complete && u64::from(capacity.host_blocks) == swappable)
        {
            0
        } else if capacity.host_blocks > 0 {
            cap.max(1)
        } else {
            cap
        };
        ensure!(
            capacity.spare_blocks == reserve,
            "plan reserves {} blocks for a reserve of {reserve}",
            capacity.spare_blocks
        );
        let host_blocks = capacity
            .host_blocks
            .checked_add(capacity.spare_blocks)
            .context("host block count overflow")?;
        ensure!(
            (host_blocks > 0) == host.is_some(),
            "host arena presence differs from the plan's {host_blocks} blocks"
        );
        if let Some(arena) = host {
            ensure!(
                arena.blocks() == host_blocks,
                "host arena holds {} blocks, plan allocates {host_blocks}",
                arena.blocks()
            );
            ensure!(
                arena.block_elements() == usize::try_from(slot_bytes / 2)?,
                "host block holds {} elements, a slot holds {}",
                arena.block_elements(),
                slot_bytes / 2
            );
        }
        let mut positions = 0u32;
        for &count in &capacity.slots_per_layer {
            positions = positions
                .checked_add(count)
                .context("slot position count overflow")?;
        }
        let layout = SlotLayout::new(geometry, &capacity)?;
        let mut ledger = TierLedger::new(layout, capacity.host_blocks, initial)?;
        let experts = geometry.experts as usize;
        let layers = geometry.layers as usize;
        let mut generation = vec![vec![0u64; experts]; layers];
        let mut next_generation = 0u64;
        for (layer, set) in initial.iter().enumerate() {
            for &expert in set {
                next_generation += 1;
                generation[layer][expert as usize] = next_generation;
            }
        }
        let mut host_block = vec![vec![NO_BLOCK; experts]; layers];
        let order = host_fill_order(
            initial,
            &geometry,
            capacity.gpu0_layers,
            peer_complete,
            capacity.host_blocks,
        );
        let mut host_bytes = 0u64;
        for (index, &(layer, expert)) in order.iter().enumerate() {
            let block = u32::try_from(index).context("host block id exceeds u32")?;
            let arena = host.context("host arena missing for the host fill")?;
            let slot = reader.read_expert(layer, expert, catalogue, catalogue.slot_bytes())?;
            let values = reader.values(slot)?;
            let mut claim = arena.claim(block)?;
            claim.slice()?.copy_from_slice(&values);
            drop(claim);
            drop(values);
            ledger.place_host(layer, expert)?;
            host_block[layer as usize][expert as usize] = block;
            host_bytes = host_bytes
                .checked_add(slot_bytes)
                .context("host load bytes overflow")?;
        }
        let free_blocks: VecDeque<FreeBlock> = (order.len() as u32..host_blocks)
            .map(|id| FreeBlock { id, wait: None })
            .collect();
        let max_routes = max_rows as usize * geometry.top_k as usize;
        let select_capacity = (positions as usize).max(cap as usize);
        let dispatches = (0..layers)
            .map(|_| fresh_dispatch(max_routes, experts, max_rows as usize))
            .collect();
        let residency = Self {
            geometry,
            capacity,
            admission,
            max_rows,
            ledger,
            events: SlotEvents::new(positions as usize),
            host_block,
            reserve,
            free_blocks,
            released: Vec::with_capacity(host_blocks as usize),
            generation,
            next_generation,
            scores: vec![vec![0.0; experts]; layers],
            routed: vec![vec![false; experts]; layers],
            missed: vec![Vec::with_capacity(experts); layers],
            dispatches,
            out: Vec::with_capacity(experts),
            incoming: Vec::with_capacity(experts),
            victims: Vec::with_capacity(experts),
            candidates: Vec::with_capacity(experts),
            pending: Vec::with_capacity(select_capacity),
            block_needs: Vec::with_capacity(select_capacity),
            plan: Vec::with_capacity(positions as usize),
            published: Vec::with_capacity(select_capacity),
            moves: Vec::with_capacity(positions as usize),
            dispatched: vec![false; layers],
            token_open: false,
            request_open: false,
        };
        let report = LoadReport {
            host_bytes,
            seconds: started.elapsed().as_secs_f64(),
        };
        Ok((residency, report))
    }

    /// Starts a request: installs the initial GPU set from the prefill frequency and emits the
    /// prompt-boundary moves. Host swaps chain through their vacated blocks; every block need is
    /// validated before the plan is applied.
    pub fn begin_request(&mut self, frequency: &[Vec<f64>]) -> Result<&[PromptMove]> {
        ensure!(!self.token_open, "request boundary inside an open token");
        ensure!(
            !self.request_open,
            "request boundary inside an open request"
        );
        self.validate_released()?;
        let initial = initial_residency(frequency, &self.capacity.slots_per_layer, &self.geometry)?;
        let scores = seed_scores(frequency, &self.geometry)?;
        self.plan.clear();
        let mut pool = u32::try_from(self.free_blocks.len() + self.released.len())
            .context("free host blocks exceed u32")?;
        let mut first_host = true;
        let backed = self.ledger.has_full_host_backing();
        for layer in 0..self.geometry.layers {
            let l = layer as usize;
            let target = &initial[l];
            self.out.clear();
            for expert in 0..self.geometry.experts {
                if self.ledger.tier(layer, expert)? == Tier::Gpu && !target.contains(&expert) {
                    self.out.push(expert);
                }
            }
            self.incoming.clear();
            for &expert in target {
                if self.ledger.tier(layer, expert)? != Tier::Gpu {
                    self.incoming.push(expert);
                }
            }
            self.incoming.sort_unstable();
            let mut paired = 0usize;
            for index in 0..self.incoming.len() {
                let expert = self.incoming[index];
                let tier = self.ledger.tier(layer, expert)?;
                let victim = self.out.get(paired).copied();
                let source_block = match tier {
                    Tier::Host => {
                        let block = self.host_block[l][expert as usize];
                        ensure!(
                            block != NO_BLOCK,
                            "host expert {expert} of layer {layer} has no block"
                        );
                        block
                    }
                    Tier::Disk => NO_BLOCK,
                    Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
                };
                let dest = match (victim, tier) {
                    (Some(_), Tier::Gpu) => unreachable!("incoming experts are not GPU resident"),
                    (Some(_), Tier::Host) => {
                        if backed {
                            Dest::None
                        } else if first_host {
                            ensure!(pool >= 1, "request needs a free host block, none");
                            pool -= 1;
                            first_host = false;
                            Dest::Pool
                        } else {
                            Dest::Chain
                        }
                    }
                    (Some(_), Tier::Disk) => {
                        if pool > self.reserve {
                            pool -= 1;
                            Dest::Pool
                        } else {
                            Dest::Disk
                        }
                    }
                    (None, _) => Dest::None,
                };
                if victim.is_some() {
                    paired += 1;
                }
                self.plan.push(PlannedMove {
                    layer,
                    expert,
                    victim,
                    tier,
                    source_block,
                    dest,
                });
            }
            ensure!(
                paired == self.out.len(),
                "layer {layer} leaves {} evicted experts without a replacement",
                self.out.len() - paired
            );
            let installs = self
                .plan
                .iter()
                .filter(|planned| planned.layer == layer && planned.victim.is_none())
                .count() as u32;
            let empty = self.ledger.empty_slots(layer)?;
            ensure!(
                installs <= empty,
                "layer {layer} needs {installs} install slots for {empty} empty slots"
            );
        }
        self.drain_released()?;
        self.moves.clear();
        let mut chain: Option<(u32, u32)> = None;
        for index in 0..self.plan.len() {
            let planned = self.plan[index];
            let layer = planned.layer;
            let l = layer as usize;
            let expert = planned.expert;
            let mut chain_slot = None;
            let (dest_block, dest_wait, slot) = match planned.victim {
                Some(victim) => {
                    let slot = self.ledger.slot(layer, victim)?;
                    let (dest_block, dest_wait) = match planned.dest {
                        Dest::Pool => {
                            let free = self.pool_block()?;
                            (Some(free.id), free.wait)
                        }
                        Dest::Chain => {
                            let (block, previous_slot) =
                                chain.context("chained swap without a predecessor")?;
                            chain_slot = Some(previous_slot);
                            (Some(block), None)
                        }
                        Dest::Disk | Dest::None => (None, None),
                    };
                    match planned.tier {
                        Tier::Host => {
                            self.ledger.admit_host(layer, expert, victim)?;
                            if !backed {
                                self.host_block[l][expert as usize] = NO_BLOCK;
                                self.host_block[l][victim as usize] =
                                    dest_block.context("host swap without a destination block")?;
                                chain = Some((planned.source_block, slot));
                            }
                        }
                        Tier::Disk => {
                            self.ledger
                                .admit_disk(layer, expert, victim, dest_block.is_some())?;
                            if let Some(block) = dest_block {
                                self.host_block[l][victim as usize] = block;
                            }
                        }
                        Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
                    }
                    (dest_block, dest_wait, slot)
                }
                None => {
                    let slot = self.ledger.install(layer, expert)?;
                    if planned.tier == Tier::Host && !backed {
                        self.host_block[l][expert as usize] = NO_BLOCK;
                        self.released.push(ReleasedBlock {
                            block: planned.source_block,
                            slot,
                            event: None,
                        });
                    }
                    (None, None, slot)
                }
            };
            self.next_generation += 1;
            self.generation[l][expert as usize] = self.next_generation;
            let source = match planned.tier {
                Tier::Host => MoveSource::Host(planned.source_block),
                Tier::Disk => MoveSource::Disk,
                Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
            };
            self.moves.push(PromptMove {
                layer,
                expert,
                victim: planned.victim,
                source,
                dest_block,
                dest_wait,
                chain_slot,
                slot,
                generation: self.next_generation,
            });
        }
        if let Some((block, slot)) = chain {
            self.released.push(ReleasedBlock {
                block,
                slot,
                event: None,
            });
        }
        self.scores = scores;
        self.ledger.check()?;
        self.request_open = true;
        Ok(&self.moves)
    }

    pub fn begin_token(&mut self) -> Result<()> {
        ensure!(self.request_open, "token outside a request");
        ensure!(!self.token_open, "token already open");
        self.validate_released()?;
        let free_after = self.free_host_blocks()
            + u32::try_from(self.released.len()).context("released block count exceeds u32")?;
        ensure!(
            free_after >= self.reserve,
            "free host blocks {free_after} below the {}-block reserve",
            self.reserve
        );
        self.drain_released()?;
        self.token_open = true;
        for layer in 0..self.geometry.layers as usize {
            self.routed[layer].fill(false);
            self.missed[layer].clear();
        }
        self.dispatched.fill(false);
        self.pending.clear();
        Ok(())
    }

    /// Splits one layer's routed list and records its admission candidates.
    pub fn dispatch(&mut self, layer: u32, routes: &[Route]) -> Result<&Dispatch> {
        ensure!(self.token_open, "dispatch outside a token");
        ensure!(
            layer < self.geometry.layers,
            "expert layer {layer} outside {}",
            self.geometry.layers
        );
        let (l, experts, top_k) = (layer as usize, self.geometry.experts, self.geometry.top_k);
        ensure!(
            !self.dispatched[l],
            "layer {layer} dispatched twice in one token"
        );
        ensure!(
            routes.len() <= self.dispatches[l].active.capacity(),
            "{} routes exceed the {} row bound",
            routes.len(),
            self.dispatches[l].active.capacity()
        );
        for (index, route) in routes.iter().enumerate() {
            ensure!(
                route.expert < experts,
                "route expert {} outside {experts}",
                route.expert
            );
            ensure!(
                route.row < self.max_rows,
                "route row {} outside the {} row bound",
                route.row,
                self.max_rows
            );
            ensure!(
                route.choice / top_k == route.row,
                "route choice {} does not belong to row {} under top-k {top_k}",
                route.choice,
                route.row
            );
            ensure!(
                !routes[..index]
                    .iter()
                    .any(|other| other.choice == route.choice),
                "duplicate (row, choice) {} in layer {layer}",
                route.choice
            );
        }
        for score in self.scores[l].iter_mut() {
            *score *= F_DECAY;
        }
        let dispatch = &mut self.dispatches[l];
        dispatch.active.clear();
        dispatch.generation.clear();
        dispatch.host.clear();
        dispatch.host_experts.clear();
        dispatch.disk.clear();
        dispatch.miss_rows.clear();
        for route in routes {
            let e = route.expert as usize;
            if !self.routed[l][e] {
                self.routed[l][e] = true;
                self.scores[l][e] += 1.0;
            }
        }
        for route in routes {
            let tier = self.ledger.tier(layer, route.expert)?;
            match tier {
                Tier::Gpu => {
                    dispatch.active.push(1);
                    dispatch
                        .generation
                        .push(self.generation[l][route.expert as usize]);
                }
                Tier::Host | Tier::Disk => {
                    dispatch.active.push(0);
                    dispatch.generation.push(0);
                    if !dispatch.miss_rows.contains(&route.row) {
                        dispatch.miss_rows.push(route.row);
                    }
                    if tier == Tier::Host {
                        dispatch.host_experts.push(route.expert);
                    }
                }
            }
        }
        dispatch.miss_rows.sort_unstable();
        dispatch.host_experts.sort_unstable();
        dispatch.host_experts.dedup();
        for route in routes {
            match self.ledger.tier(layer, route.expert)? {
                Tier::Gpu => {}
                Tier::Host => {
                    let row = compact_row(&dispatch.miss_rows, route.row)?;
                    let expert = u32::try_from(
                        dispatch
                            .host_experts
                            .binary_search(&route.expert)
                            .expect("host expert is listed"),
                    )
                    .context("host expert index exceeds u32")?;
                    dispatch.host.push(Assignment {
                        row,
                        choice: route.choice,
                        expert,
                        weight: route.weight,
                    });
                }
                Tier::Disk => {
                    let row = compact_row(&dispatch.miss_rows, route.row)?;
                    dispatch.disk.push(Route { row, ..*route });
                }
            }
        }
        for expert in dispatch
            .host_experts
            .iter()
            .chain(dispatch.disk.iter().map(|miss| &miss.expert))
        {
            if !self.missed[l].contains(expert) {
                self.missed[l].push(*expert);
            }
        }
        self.missed[l].sort_unstable();
        self.dispatched[l] = true;
        Ok(dispatch)
    }

    /// Resolves every non-GPU expert selected by the current dispatch before
    /// the layer's device launch. Victims are chosen only from experts not
    /// routed by this token.
    pub fn resolve(&mut self, layer: u32, experts: &[u32]) -> Result<Vec<Published>> {
        ensure!(self.token_open, "resolve outside a token");
        ensure!(
            layer < self.geometry.layers,
            "expert layer {layer} outside {}",
            self.geometry.layers
        );
        let l = layer as usize;
        ensure!(self.dispatched[l], "layer {layer} resolved before dispatch");
        let mut unique = experts.to_vec();
        unique.sort_unstable();
        unique.dedup();
        for &expert in &unique {
            ensure!(
                expert < self.geometry.experts,
                "resolve expert {expert} of layer {layer} outside {}",
                self.geometry.experts
            );
        }
        self.drain_released()?;
        self.victims.clear();
        for expert in 0..self.geometry.experts {
            if !self.routed[l][expert as usize] && self.ledger.tier(layer, expert)? == Tier::Gpu {
                self.victims.push(expert);
            }
        }
        self.victims.sort_by(|&a, &b| {
            self.scores[l][a as usize]
                .total_cmp(&self.scores[l][b as usize])
                .then(a.cmp(&b))
        });
        let backed = self.ledger.has_full_host_backing();
        let mut pool =
            u32::try_from(self.free_blocks.len()).context("free host blocks exceed u32")?;
        self.published.clear();
        for expert in unique {
            if self.ledger.tier(layer, expert)? == Tier::Gpu {
                continue;
            }
            let empty = self.ledger.empty_slot(layer)?;
            let victim = if empty.is_some() {
                anyhow::bail!(
                    "layer {layer} has an empty slot; synchronous tier resolution requires a victim"
                )
            } else {
                let victim = *self
                    .victims
                    .first()
                    .context("no unrouted GPU victim for a synchronous expert miss")?;
                self.victims.remove(0);
                Some(victim)
            };
            let source_block = match self.ledger.tier(layer, expert)? {
                Tier::Host => {
                    let block = self.host_block[l][expert as usize];
                    ensure!(
                        block != NO_BLOCK,
                        "host expert {expert} of layer {layer} has no block"
                    );
                    Some(block)
                }
                Tier::Disk | Tier::Gpu => None,
            };
            let (slot, dest_block, dest_wait) = match victim {
                Some(victim) => {
                    let slot = self.ledger.slot(layer, victim)?;
                    let (dest_block, dest_wait) = match self.ledger.tier(layer, expert)? {
                        Tier::Host if backed => (None, None),
                        Tier::Host => {
                            ensure!(pool >= 1, "synchronous host swap needs a free block");
                            pool -= 1;
                            let free = self.pool_block()?;
                            (Some(free.id), free.wait)
                        }
                        Tier::Disk => {
                            if pool > self.reserve {
                                pool -= 1;
                                let free = self.pool_block()?;
                                (Some(free.id), free.wait)
                            } else {
                                (None, None)
                            }
                        }
                        Tier::Gpu => unreachable!("resolved expert is not GPU resident"),
                    };
                    match self.ledger.tier(layer, expert)? {
                        Tier::Host => {
                            self.ledger.admit_host(layer, expert, victim)?;
                            if !backed {
                                self.host_block[l][expert as usize] = NO_BLOCK;
                                self.host_block[l][victim as usize] =
                                    dest_block.context("host swap has no destination block")?;
                                self.released.push(ReleasedBlock {
                                    block: source_block.context("host swap source missing")?,
                                    slot,
                                    event: None,
                                });
                            }
                        }
                        Tier::Disk => {
                            self.ledger
                                .admit_disk(layer, expert, victim, dest_block.is_some())?;
                            if let Some(block) = dest_block {
                                self.host_block[l][victim as usize] = block;
                            }
                        }
                        Tier::Gpu => unreachable!("resolved expert is not GPU resident"),
                    }
                    (slot, dest_block, dest_wait)
                }
                None => {
                    let slot = self.ledger.install(layer, expert)?;
                    if self.ledger.tier(layer, expert)? == Tier::Host && !backed {
                        let block = source_block.context("host miss without a source block")?;
                        self.host_block[l][expert as usize] = NO_BLOCK;
                        self.released.push(ReleasedBlock {
                            block,
                            slot,
                            event: None,
                        });
                    }
                    (slot, None, None)
                }
            };
            self.next_generation += 1;
            self.generation[l][expert as usize] = self.next_generation;
            self.published.push(Published {
                layer,
                expert,
                victim: victim.expect("synchronous tier victim is present"),
                from_host: source_block.is_some(),
                source_block,
                dest_block,
                dest_wait,
                slot,
                generation: self.next_generation,
            });
        }
        self.missed[l].retain(|expert| !self.published.iter().any(|p| p.expert == *expert));
        self.ledger.check()?;
        Ok(std::mem::take(&mut self.published))
    }

    /// Publishes at most the hard per-token cap after validating block requirements.
    pub fn finish_token(&mut self) -> Result<&[Published]> {
        ensure!(self.token_open, "no token open");
        let cap = self.admission.hard_cap(self.capacity.slot_bytes())?;
        self.pending.clear();
        for layer in 0..self.geometry.layers {
            let l = layer as usize;
            if self.missed[l].is_empty() {
                continue;
            }
            self.victims.clear();
            for expert in 0..self.geometry.experts {
                if !self.routed[l][expert as usize] && self.ledger.tier(layer, expert)? == Tier::Gpu
                {
                    self.victims.push(expert);
                }
            }
            self.victims.sort_unstable_by(|&a, &b| {
                self.scores[l][a as usize]
                    .total_cmp(&self.scores[l][b as usize])
                    .then(a.cmp(&b))
            });
            self.candidates.clear();
            self.candidates.extend_from_slice(&self.missed[l]);
            self.candidates.sort_unstable_by(|&a, &b| {
                self.scores[l][b as usize]
                    .total_cmp(&self.scores[l][a as usize])
                    .then(a.cmp(&b))
            });
            for (rank, &candidate) in self.candidates.iter().enumerate() {
                let Some(&victim) = self.victims.get(rank) else {
                    break;
                };
                let candidate_score = self.scores[l][candidate as usize];
                let delta = candidate_score - self.scores[l][victim as usize];
                if candidate_score >= THETA_MIN && delta > 0.0 {
                    self.pending.push((delta, layer, candidate, victim));
                }
            }
        }
        self.pending
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        self.pending.truncate(cap as usize);
        self.pending
            .sort_unstable_by(|a, b| a.1.cmp(&b.1).then(b.0.total_cmp(&a.0)).then(a.2.cmp(&b.2)));
        let backed = self.ledger.has_full_host_backing();
        let mut pool =
            u32::try_from(self.free_blocks.len()).context("free host blocks exceed u32")?;
        self.block_needs.clear();
        for index in 0..self.pending.len() {
            let (_, layer, candidate, _) = self.pending[index];
            if backed {
                self.block_needs.push(false);
            } else if self.ledger.tier(layer, candidate)? == Tier::Host {
                ensure!(
                    self.host_block[layer as usize][candidate as usize] != NO_BLOCK,
                    "host expert {candidate} of layer {layer} has no block"
                );
                ensure!(pool >= 1, "host write-back needs a free block, none");
                pool -= 1;
                self.block_needs.push(true);
            } else if pool > self.reserve {
                pool -= 1;
                self.block_needs.push(true);
            } else {
                self.block_needs.push(false);
            }
        }
        self.published.clear();
        for index in 0..self.pending.len() {
            let (_, layer, candidate, victim) = self.pending[index];
            let l = layer as usize;
            let slot = self.ledger.slot(layer, victim)?;
            let from_host = self.ledger.tier(layer, candidate)? == Tier::Host;
            let (source_block, dest_block, dest_wait) = if from_host {
                let block = self.host_block[l][candidate as usize];
                if backed {
                    self.ledger.admit_host(layer, candidate, victim)?;
                    (Some(block), None, None)
                } else {
                    let free = self.pool_block()?;
                    self.ledger.admit_host(layer, candidate, victim)?;
                    self.host_block[l][candidate as usize] = NO_BLOCK;
                    self.host_block[l][victim as usize] = free.id;
                    self.released.push(ReleasedBlock {
                        block,
                        slot,
                        event: None,
                    });
                    (Some(block), Some(free.id), free.wait)
                }
            } else {
                let dest = if self.block_needs[index] {
                    Some(self.pool_block()?)
                } else {
                    None
                };
                let dest_block = dest.map(|free| free.id);
                let dest_wait = dest.and_then(|free| free.wait);
                self.ledger
                    .admit_disk(layer, candidate, victim, dest_block.is_some())?;
                if let Some(block) = dest_block {
                    self.host_block[l][victim as usize] = block;
                }
                (None, dest_block, dest_wait)
            };
            self.next_generation += 1;
            self.generation[l][candidate as usize] = self.next_generation;
            self.published.push(Published {
                layer,
                expert: candidate,
                victim,
                from_host,
                source_block,
                dest_block,
                dest_wait,
                slot,
                generation: self.next_generation,
            });
        }
        self.token_open = false;
        debug_assert!(self.ledger.check().is_ok());
        Ok(&self.published)
    }

    /// Records one copy event covering every listed slot and captures it for any released block
    /// waiting on that slot.
    pub fn record_copy(&mut self, slots: &[u32], event: u64) -> Result<()> {
        self.events.record_copy(slots, event)?;
        for released in &mut self.released {
            if released.event.is_none() && slots.contains(&released.slot) {
                released.event = Some(event);
            }
        }
        Ok(())
    }

    /// Records one reader event covering every listed slot.
    pub fn record_readers(&mut self, slots: &[u32], event: u64) -> Result<()> {
        self.events.record_readers(slots, event)
    }

    /// Copy event the reader stream waits on for the first read of a slot.
    pub fn first_read(&self, slot: u32) -> Result<Option<u64>> {
        self.events.copy_event(slot)
    }

    /// `[last copy event, last reader event]` of a slot; the copy stream waits on the reader
    /// event before overwriting. Clears the slot's lifetime.
    pub fn reclaim(&mut self, slot: u32) -> Result<[Option<u64>; 2]> {
        self.events.reclaim(slot)
    }

    /// Borrows the resident tail for a prefill span; victims take a free block above the cap
    /// reserve or drop to disk.
    pub fn borrow(&mut self, slots: u32) -> Result<Borrow> {
        ensure!(
            !self.token_open && !self.request_open,
            "borrow runs only between requests"
        );
        self.validate_released()?;
        let available = u32::try_from(self.free_blocks.len() + self.released.len())
            .context("free host blocks exceed u32")?
            .saturating_sub(self.reserve);
        let mut borrow = self.ledger.borrow_tail(slots, available)?;
        self.drain_released()?;
        let write_backs = borrow.write_back.iter().filter(|&&copy| copy).count();
        ensure!(
            write_backs <= self.free_blocks.len(),
            "borrow assigned {write_backs} host copies for {} free blocks",
            self.free_blocks.len()
        );
        borrow.host_waits.resize(borrow.experts.len(), None);
        for (index, &(layer, expert)) in borrow.experts.iter().enumerate() {
            if borrow.write_back[index] {
                let free = self.pool_block()?;
                self.host_block[layer as usize][expert as usize] = free.id;
                borrow.host_waits[index] = free.wait;
            }
        }
        Ok(borrow)
    }

    /// Returns every borrowed position to admission.
    pub fn refill(&mut self) -> Result<Vec<u32>> {
        ensure!(
            !self.token_open && !self.request_open,
            "refill runs only between requests"
        );
        self.ledger.refill_tail()
    }

    /// Closes a request; borrowing and refilling run before the next boundary opens one.
    pub fn end_request(&mut self) -> Result<()> {
        ensure!(!self.token_open, "request ends inside an open token");
        ensure!(self.request_open, "no open request");
        self.request_open = false;
        Ok(())
    }

    fn validate_released(&self) -> Result<()> {
        for released in &self.released {
            ensure!(
                released.event.is_some(),
                "released block {} has no copy event on slot {}",
                released.block,
                released.slot
            );
        }
        Ok(())
    }

    fn drain_released(&mut self) -> Result<()> {
        self.validate_released()?;
        for released in &self.released {
            self.free_blocks.push_back(FreeBlock {
                id: released.block,
                wait: released.event,
            });
        }
        self.released.clear();
        Ok(())
    }

    fn pool_block(&mut self) -> Result<FreeBlock> {
        self.free_blocks
            .pop_front()
            .context("free host block missing from the pool")
    }

    pub fn ledger(&self) -> &TierLedger {
        &self.ledger
    }

    /// Published generation of an expert; 0 means not resident through this runtime.
    pub fn generation(&self, layer: u32, expert: u32) -> Result<u64> {
        ensure!(
            layer < self.geometry.layers && expert < self.geometry.experts,
            "expert {expert} of layer {layer} outside the geometry"
        );
        Ok(self.generation[layer as usize][expert as usize])
    }

    pub fn host_block(&self, layer: u32, expert: u32) -> Result<u32> {
        ensure!(
            layer < self.geometry.layers && expert < self.geometry.experts,
            "expert {expert} of layer {layer} outside the geometry"
        );
        Ok(self.host_block[layer as usize][expert as usize])
    }

    pub fn free_host_blocks(&self) -> u32 {
        u32::try_from(self.free_blocks.len()).expect("free host block count fits u32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::experts::plan::{PlanInputs, Reserves, ServiceCell, ServiceTable, capacity};
    use std::path::{Path, PathBuf};

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            first_layer: 0,
            layers: 2,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn three_layers() -> ExpertGeometry {
        ExpertGeometry {
            first_layer: 0,
            layers: 3,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn service() -> ServiceTable {
        ServiceTable::new(
            "test",
            vec![
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 0.001,
                },
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 2,
                    seconds: 0.002,
                },
            ],
        )
        .unwrap()
    }

    fn reserves() -> Reserves {
        Reserves {
            device_fixed_bytes: 1_000,
            host_fixed_bytes: 100,
            residual_row_bytes: 16,
            normalized_row_bytes: 16,
            contribution_row_bytes: 32,
            done_row_bytes: 8,
            route_row_bytes: 40,
            prefill_staging_slots: 0,
            disk_ring_slots: 1,
        }
    }

    fn frequency() -> Vec<Vec<f64>> {
        vec![vec![1.0; 8]; 2]
    }

    fn entry(role: &str, shape: Vec<usize>, start: u64, len: u64) -> TensorEntry {
        TensorEntry {
            dtype: safetensors::Dtype::BF16,
            shape,
            shard: "shard-0.safetensors".into(),
            bytes: len as usize,
            load: true,
            data_start: 0,
            data_offsets: [start, start + len],
            scope: TensorScope::Text,
            layer: None,
            role: role.into(),
            scale_name: None,
        }
    }

    fn manifest(geometry: ExpertGeometry) -> TensorManifest {
        let experts = u64::from(geometry.experts);
        let gate_up =
            experts * u64::from(geometry.intermediate) * u64::from(geometry.hidden) * 2 * 2;
        let down = experts * u64::from(geometry.hidden) * u64::from(geometry.intermediate) * 2;
        let mut entries = std::collections::BTreeMap::new();
        for layer in 0..geometry.layers {
            let base = u64::from(layer) * (gate_up + down);
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"),
                entry(
                    "mlp.experts.gate_up_proj",
                    vec![
                        geometry.experts as usize,
                        2 * geometry.intermediate as usize,
                        geometry.hidden as usize,
                    ],
                    base,
                    gate_up,
                ),
            );
            entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.down_proj"),
                entry(
                    "mlp.experts.down_proj",
                    vec![
                        geometry.experts as usize,
                        geometry.hidden as usize,
                        geometry.intermediate as usize,
                    ],
                    base + gate_up,
                    down,
                ),
            );
        }
        TensorManifest {
            entries,
            resident_bytes: 0,
        }
    }

    fn pattern(bytes: usize) -> Vec<u8> {
        (0..bytes)
            .map(|i| ((i * 37 + i / 251) % 256) as u8)
            .collect()
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(tag: &str) -> Result<TempDir> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "flyingfish-runtime-{}-{}-{}",
            std::process::id(),
            tag,
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)?;
        Ok(TempDir(dir))
    }

    fn staged(geometry: ExpertGeometry) -> Result<(TempDir, ExpertCatalogue)> {
        let dir = temp_dir("ranges")?;
        let manifest = manifest(geometry);
        let mut bytes = 0usize;
        for entry in manifest.entries.values() {
            bytes = bytes.max(entry.data_offsets[1] as usize);
        }
        std::fs::write(dir.path().join("shard-0.safetensors"), pattern(bytes))?;
        let catalogue = ExpertCatalogue::from_manifest(
            &manifest,
            geometry,
            crate::model_spec::ExpertFormat::Bf16,
        )?;
        Ok((dir, catalogue))
    }

    fn bf16_layout(geometry: &ExpertGeometry) -> Result<crate::experts::catalogue::ExpertLayout> {
        crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            geometry,
        )
    }

    fn slots(geometry: ExpertGeometry) -> Result<u32> {
        geometry
            .layers
            .checked_mul(2)
            .context("slot count overflow")
    }

    fn build_full(
        geometry: ExpertGeometry,
        host_blocks: u32,
        cap: u32,
    ) -> Result<(Residency, LoadReport, Arc<HostBlocks>, TempDir)> {
        let reserves = reserves();
        let service = service();
        let layout = bf16_layout(&geometry)?;
        let slot = layout.slot_bytes();
        let inputs = PlanInputs {
            geometry,
            layout,
            device_free_bytes: reserves.persistent_bytes(slot)?
                + u64::from(slots(geometry)?) * slot,
            host_available_bytes: reserves.host_bytes(slot)? + u64::from(host_blocks + cap) * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: u64::MAX,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, cap)?;
        assert_eq!(capacity.host_blocks, host_blocks);
        let admission = Admission {
            budget_bytes_per_token: slot * u64::from(cap),
        };
        let host = HostBlocks::new(
            host_blocks + capacity.spare_blocks,
            usize::try_from(slot / 2)?,
        )?;
        let (dir, catalogue) = staged(geometry)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let frequency = vec![vec![1.0; 8]; geometry.layers as usize];
        let initial = initial_residency(&frequency, &capacity.slots_per_layer, &geometry)?;
        let (residency, report) = Residency::load(LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows: 8,
            initial: &initial,
            host: Some(&host),
            catalogue: &catalogue,
            reader: &reader,
        })?;
        Ok((residency, report, host, dir))
    }

    fn build(host_blocks: u32, cap: u32) -> Result<Residency> {
        let (mut residency, _, _, _) = build_full(geometry(), host_blocks, cap)?;
        residency.begin_request(&frequency())?;
        Ok(residency)
    }

    fn route(row: u32, choice: u32, expert: u32) -> Route {
        Route {
            row,
            choice,
            expert,
            weight: bf16::from_f32(1.0),
        }
    }

    #[test]
    fn synchronous_host_swap_recycles_source_after_copy() -> Result<()> {
        let mut state = build(4, 1)?;
        for iteration in 0..8u64 {
            let expert = (0..8)
                .find(|&id| state.ledger().tier(0, id).is_ok_and(|t| t == Tier::Host))
                .context("host expert missing")?;
            state.begin_token()?;
            state.dispatch(0, &[route(0, 0, expert)])?;
            let moves = state.resolve(0, &[expert])?;
            assert_eq!(moves.len(), 1);
            assert_eq!(state.released.len(), 1);
            assert!(state.validate_released().is_err());
            state.record_copy(&[moves[0].slot], iteration + 1)?;
            state.finish_token()?;
        }
        Ok(())
    }

    fn set_scores(residency: &mut Residency, layer: usize, scores: [f64; 8]) {
        residency.scores[layer] = scores.to_vec();
    }

    fn host_swap(
        residency: &mut Residency,
        layer: usize,
        active: u32,
        candidate: u32,
        victim: u32,
        event: u64,
    ) -> Result<Published> {
        let mut scores = [0.0; 8];
        scores[active as usize] = 5.0;
        scores[candidate as usize] = 30.0;
        scores[victim as usize] = 1.0;
        set_scores(residency, layer, scores);
        residency.begin_token()?;
        residency.dispatch(layer as u32, &[route(0, 0, active), route(0, 1, candidate)])?;
        let published = residency.finish_token()?.to_vec();
        let entry = published[0];
        assert!(entry.from_host);
        assert_eq!((entry.expert, entry.victim), (candidate, victim));
        residency.record_copy(&[entry.slot], event)?;
        Ok(entry)
    }

    #[test]
    fn load_fills_the_host_tier_interleaved_from_the_native_ranges() -> Result<()> {
        let (residency, report, host, _dir) = build_full(geometry(), 3, 2)?;
        let slot = bf16_layout(&geometry())?.slot_bytes();
        assert_eq!(report.host_bytes, 3 * slot);
        assert!(report.seconds.is_finite() && report.seconds >= 0.0);
        for layer in 0..2u32 {
            for expert in 0..2u32 {
                assert_eq!(residency.ledger().tier(layer, expert)?, Tier::Gpu);
            }
        }
        let bytes = pattern(3_072);
        for (layer, expert, block) in [(0u32, 2u32, 0u32), (1, 2, 1), (0, 3, 2)] {
            assert_eq!(residency.ledger().tier(layer, expert)?, Tier::Host);
            assert_eq!(residency.host_block(layer, expert)?, block);
            let lease = host.lease(&[block])?;
            let values = lease.read(block)?;
            let base = layer as usize * 1_536;
            let gate = base + expert as usize * 128;
            let down = base + 1_024 + expert as usize * 64;
            for (index, value) in values.iter().enumerate() {
                let byte = if index < 64 {
                    gate + index * 2
                } else {
                    down + (index - 64) * 2
                };
                assert_eq!(
                    value.to_bits(),
                    u16::from_le_bytes([bytes[byte], bytes[byte + 1]]),
                    "layer {layer} expert {expert} element {index}"
                );
            }
            drop(lease);
        }
        assert_eq!(residency.ledger().tier(1, 3)?, Tier::Disk);
        assert_eq!(residency.free_host_blocks(), 2, "the cap reserve is free");
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn request_boundary_swaps_only_the_gpu_set() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.end_request()?;
        assert!(residency.begin_request(&frequency())?.is_empty());
        residency.end_request()?;
        let mut hot = frequency();
        hot[0][2] = 5.0;
        let moves = residency.begin_request(&hot)?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 2));
        assert_eq!(movement.victim, Some(1));
        assert_eq!(movement.source, MoveSource::Host(0));
        assert_eq!(movement.dest_block, Some(3));
        assert_eq!(movement.dest_wait, None);
        assert_eq!(movement.chain_slot, None);
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        assert_eq!(residency.host_block(0, 1)?, 3);
        assert_eq!(residency.host_block(0, 2)?, NO_BLOCK);
        residency.record_copy(&[movement.slot], 50)?;
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), 2);
        residency.finish_token()?.to_vec();
        residency.end_request()?;
        let moves = residency.begin_request(&frequency())?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 1));
        assert_eq!(movement.victim, Some(2));
        assert_eq!(movement.source, MoveSource::Host(3));
        assert_eq!(
            movement.dest_block,
            Some(4),
            "the oldest free block is reused"
        );
        assert_eq!(movement.dest_wait, None);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Gpu);
        assert_eq!(residency.host_block(0, 2)?, 4);
        assert_eq!(
            residency.free_blocks.front().map(|block| block.wait),
            Some(Some(50))
        );
        assert!(residency.generation(0, 1)? > residency.generation(0, 0)?);
        Ok(())
    }

    #[test]
    fn request_boundary_chains_host_swaps_through_vacated_blocks() -> Result<()> {
        let (mut residency, _, _, _) = build_full(three_layers(), 4, 1)?;
        let mut hot = vec![vec![1.0; 8]; 3];
        for layer in &mut hot {
            layer[2] = 5.0;
            layer[3] = 4.0;
        }
        residency.begin_request(&frequency_for(3))?;
        residency.end_request()?;
        let moves = residency.begin_request(&hot)?.to_vec();
        assert_eq!(moves.len(), 6);
        assert_eq!(
            residency.free_host_blocks(),
            0,
            "one block serves the whole chain"
        );
        let mut chained = 0;
        for movement in &moves {
            if let MoveSource::Host(_) = movement.source {
                chained += 1;
            }
        }
        assert_eq!(chained, 4);
        assert_eq!(moves[0].dest_block, Some(4));
        assert_eq!(moves[0].chain_slot, None);
        assert_eq!(moves[1].dest_block, Some(0));
        assert_eq!(moves[1].chain_slot, Some(moves[0].slot));
        assert_eq!(moves[2].dest_block, Some(3));
        assert_eq!(moves[2].chain_slot, Some(moves[1].slot));
        assert_eq!(moves[4].dest_block, Some(1));
        assert_eq!(moves[4].chain_slot, Some(moves[2].slot));
        for movement in &moves {
            residency.record_copy(&[movement.slot], 60 + u64::from(movement.slot))?;
        }
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), 1);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    fn frequency_for(layers: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0; 8]; layers]
    }

    #[test]
    fn mixed_admissions_keep_the_pool_at_the_cap_reserve() -> Result<()> {
        let mut residency = build(6, 2)?;
        let reserve = 2;
        let mut event = 100u64;
        let _ = host_swap(&mut residency, 0, 0, 2, 1, event)?;
        event += 1;
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), reserve);
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        set_scores(
            &mut residency,
            0,
            [1.0, 0.0, 2.0, 20.0, 0.0, 0.0, 30.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 6), route(0, 1, 3)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2, "mixed token");
        assert_eq!((published[0].expert, published[0].from_host), (6, false));
        assert!(
            published[0].dest_block.is_none(),
            "the reserve blocks the disk write-back"
        );
        assert_eq!((published[1].expert, published[1].from_host), (3, true));
        assert!(published[1].dest_block.is_some());
        for admitted in &published {
            residency.record_copy(&[admitted.slot], event)?;
            event += 1;
        }
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), reserve);
        residency.finish_token()?.to_vec();
        let _ = host_swap(&mut residency, 0, 3, 4, 6, event)?;
        event += 1;
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), reserve);
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        set_scores(
            &mut residency,
            0,
            [0.0, 0.0, 0.5, 1.0, 2.0, 0.0, 20.0, 30.0],
        );
        residency.dispatch(0, &[route(0, 0, 7), route(0, 1, 6)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2, "second mixed token");
        assert_eq!((published[0].expert, published[0].from_host), (7, false));
        assert!(published[0].dest_block.is_none());
        assert_eq!((published[1].expert, published[1].from_host), (6, true));
        assert!(published[1].dest_block.is_some());
        for admitted in &published {
            residency.record_copy(&[admitted.slot], event)?;
            event += 1;
        }
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), reserve);
        residency.finish_token()?.to_vec();
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn dispatch_partitions_every_choice_exactly_once() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        let routes = vec![
            route(0, 0, 2),
            route(0, 1, 1),
            route(1, 2, 6),
            route(1, 3, 7),
        ];
        {
            let dispatch = residency.dispatch(0, &routes)?;
            assert_eq!(dispatch.active, vec![0, 1, 0, 0]);
            assert_eq!(dispatch.generation, vec![0, 2, 0, 0]);
            assert_eq!(dispatch.miss_rows, vec![0, 1]);
            assert_eq!(dispatch.host_experts, vec![2]);
            assert_eq!(dispatch.host.len(), 1);
            assert_eq!(dispatch.host[0].row, 0);
            assert_eq!(dispatch.host[0].choice, 0);
            assert_eq!(dispatch.host[0].expert, 0);
            assert_eq!(dispatch.disk.len(), 2);
            assert_eq!(dispatch.disk[0].choice, 2);
            assert_eq!(dispatch.disk[0].expert, 6);
            assert_eq!(dispatch.disk[0].row, 1);
            assert_eq!(dispatch.disk[1].choice, 3);
            assert_eq!(dispatch.disk[1].expert, 7);
        }
        assert!(residency.dispatch(0, &[route(0, 0, 0)]).is_err());
        let published = residency.finish_token()?.to_vec();
        for entry in &published {
            residency.record_copy(&[entry.slot], 80)?;
        }
        residency.begin_token()?;
        assert!(residency.dispatch(0, &[route(0, 2, 0)]).is_err());
        assert!(residency.dispatch(0, &[route(8, 16, 0)]).is_err());
        assert!(residency.dispatch(0, &[route(0, 0, 9)]).is_err());
        let mut rejected = routes;
        rejected.push(route(0, 0, 2));
        assert!(residency.dispatch(0, &rejected).is_err());
        Ok(())
    }

    #[test]
    fn finish_token_admits_under_the_cap_and_bumps_generations() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]);
        set_scores(&mut residency, 1, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 7)])?;
        residency.dispatch(1, &[route(0, 0, 1), route(0, 1, 6)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2, "hard cap is 2 experts per token");
        assert_eq!(published[0].layer, 0);
        assert_eq!(published[1].layer, 1);
        for entry in &published {
            assert_eq!(
                residency.generation(entry.layer, entry.expert)?,
                entry.generation
            );
            assert_eq!(
                residency.ledger().tier(entry.layer, entry.expert)?,
                Tier::Gpu
            );
            assert_eq!(
                residency.ledger().slot(entry.layer, entry.expert)?,
                entry.slot
            );
            assert!(!entry.from_host);
            assert!(entry.source_block.is_none());
            assert!(
                entry.dest_block.is_none(),
                "the cap reserve blocks the write-back"
            );
        }
        residency.begin_token()?;
        let expert = published[0].expert;
        let hit = {
            let dispatch = residency.dispatch(0, &[route(0, 0, expert)])?;
            (dispatch.active.clone(), dispatch.generation.clone())
        };
        assert_eq!(hit, (vec![1], vec![published[0].generation]));
        Ok(())
    }

    #[test]
    fn per_layer_candidates_take_disjoint_victims_under_the_cap() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 2), route(0, 1, 3)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2);
        assert_eq!(published[0].expert, 2);
        assert_eq!(published[0].victim, 0);
        assert_eq!(published[1].expert, 3);
        assert_eq!(published[1].victim, 1);
        assert!(published[0].from_host && published[1].from_host);
        assert_eq!(residency.host_block(0, 0)?, 3);
        assert_eq!(residency.host_block(0, 1)?, 4);
        assert_eq!(residency.free_host_blocks(), 0);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn the_cap_binds_the_global_selection() -> Result<()> {
        let mut residency = build(3, 1)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 2), route(0, 1, 3)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].expert, 2);
        assert_eq!(published[0].victim, 0);
        assert_eq!(residency.ledger().tier(0, 3)?, Tier::Host);
        Ok(())
    }

    #[test]
    fn theta_and_delta_reject_a_candidate() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [0.0; 8]);
        residency.dispatch(0, &[route(0, 0, 2)])?;
        assert!(
            residency
                .finish_token()?
                .iter()
                .all(|entry| entry.layer != 0),
            "candidate 1.0 is below theta against a zero victim"
        );
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 6.05, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        let published = residency.finish_token()?.to_vec();
        assert!(
            published.iter().all(|entry| entry.layer != 0),
            "candidate 5.9 is below the non-active victim 5.93"
        );
        Ok(())
    }

    #[test]
    fn an_active_resident_is_never_the_victim() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 admission");
        assert_eq!(entry.expert, 2);
        assert_eq!(entry.victim, 1);
        assert!(entry.from_host);
        assert_eq!(residency.ledger().tier(0, 0)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        Ok(())
    }

    #[test]
    fn a_host_candidate_swaps_through_the_pool() -> Result<()> {
        let mut residency = build(3, 2)?;
        let source = residency.host_block(0, 2)?;
        let spare = *residency.free_blocks.iter().next().expect("free block");
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 host admission");
        assert!(entry.from_host);
        assert_eq!((entry.expert, entry.victim), (2, 1));
        assert_eq!(entry.source_block, Some(source));
        assert_eq!(entry.dest_block, Some(spare.id));
        assert_eq!(entry.dest_wait, spare.wait);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        assert_eq!(residency.host_block(0, 1)?, spare.id);
        assert_eq!(residency.host_block(0, 2)?, NO_BLOCK);
        assert_eq!(residency.free_host_blocks(), 1);
        residency.record_copy(&[entry.slot], 70)?;
        residency.begin_token()?;
        assert_eq!(residency.free_host_blocks(), 2);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn released_host_block_wait_keeps_its_source_copy_event() -> Result<()> {
        let mut residency = build(3, 2)?;
        let entry = host_swap(&mut residency, 0, 0, 2, 1, 90)?;
        let source = entry
            .source_block
            .context("host admission source missing")?;
        residency.record_copy(&[entry.slot], 91)?;
        residency.begin_token()?;
        let free = residency
            .free_blocks
            .iter()
            .find(|free| free.id == source)
            .context("admitted source block did not return to the free pool")?;
        assert_eq!(free.wait, Some(90));
        Ok(())
    }

    #[test]
    fn drain_uses_the_event_captured_at_record_copy() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.end_request()?;
        residency.begin_request(&frequency())?;
        let entry = host_swap(&mut residency, 0, 0, 2, 1, 90)?;
        residency.end_request()?;
        let borrow = residency.borrow(2)?;
        assert!(borrow.positions.contains(&entry.slot));
        residency.reclaim(entry.slot)?;
        assert_eq!(residency.refill()?.len(), 2);
        let moves = residency.begin_request(&frequency())?.to_vec();
        assert_eq!(moves.len(), 2, "the evicted rank-1 experts are installed");
        assert_eq!(residency.free_host_blocks(), 2);
        Ok(())
    }

    #[test]
    fn borrow_and_refill_run_only_between_requests() -> Result<()> {
        let mut residency = build(3, 2)?;
        let error = residency.borrow(2).unwrap_err().to_string();
        assert!(error.contains("between requests"), "{error}");
        let error = residency.refill().unwrap_err().to_string();
        assert!(error.contains("between requests"), "{error}");
        assert_eq!(residency.free_host_blocks(), 2);
        residency.begin_token()?;
        assert!(residency.end_request().is_err());
        residency.finish_token()?.to_vec();
        residency.end_request()?;
        residency.borrow(2)?;
        residency.refill()?;
        Ok(())
    }

    #[test]
    fn request_validation_leaves_the_pool_unchanged() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.end_request()?;
        residency.begin_request(&frequency())?;
        let _ = host_swap(&mut residency, 0, 0, 2, 1, 91)?;
        residency.end_request()?;
        let borrow = residency.borrow(2)?;
        assert_eq!(borrow.positions.len(), 2);
        let free_before = residency.free_host_blocks();
        let error = residency
            .begin_request(&frequency())
            .unwrap_err()
            .to_string();
        assert!(error.contains("install slots"), "{error}");
        assert_eq!(
            residency.free_host_blocks(),
            free_before,
            "a failed boundary drained the pool"
        );
        residency.refill()?;
        Ok(())
    }

    #[test]
    fn zero_cap_all_resident_loads_without_a_host_tier() -> Result<()> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let slot = bf16_layout(&geometry)?.slot_bytes();
        let routed = u64::from(geometry.layers)
            .checked_mul(u64::from(geometry.experts))
            .context("routed count overflow")?;
        let inputs = PlanInputs {
            geometry,
            layout: bf16_layout(&geometry)?,
            device_free_bytes: reserves.persistent_bytes(slot)?
                + routed.checked_mul(slot).context("device extent overflow")?,
            host_available_bytes: reserves.host_bytes(slot)?,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: u64::MAX,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, 0)?;
        assert_eq!(capacity.slots_per_layer, vec![8, 8]);
        assert_eq!(capacity.host_blocks, 0);
        assert_eq!(capacity.spare_blocks, 0, "no host tier, no reserve");
        let admission = Admission {
            budget_bytes_per_token: 0,
        };
        let (dir, catalogue) = staged(geometry)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let frequency = vec![vec![1.0; 8]; 2];
        let initial = initial_residency(&frequency, &capacity.slots_per_layer, &geometry)?;
        let (mut residency, _report) = Residency::load(LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows: 8,
            initial: &initial,
            host: None,
            catalogue: &catalogue,
            reader: &reader,
        })?;
        for layer in 0..2u32 {
            for expert in 0..8u32 {
                assert_eq!(residency.ledger().tier(layer, expert)?, Tier::Gpu);
            }
        }
        assert_eq!(residency.free_host_blocks(), 0);
        assert!(residency.begin_request(&frequency)?.is_empty());
        Ok(())
    }

    #[test]
    fn host_surplus_above_full_coverage_absorbs_borrow_victims() -> Result<()> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let layout = bf16_layout(&geometry)?;
        let slot = layout.slot_bytes();
        let cap = 1u32;
        let spare = cap;
        let host_blocks = 14u32;
        let inputs = PlanInputs {
            geometry,
            layout,
            device_free_bytes: reserves.persistent_bytes(slot)?
                + u64::from(slots(geometry)?) * slot,
            host_available_bytes: reserves.host_bytes(slot)?
                + u64::from(host_blocks + spare) * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: u64::MAX,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, cap)?;
        assert_eq!(
            capacity.host_blocks, host_blocks,
            "the planner sizes the non-GPU experts plus the borrow headroom"
        );
        let admission = Admission {
            budget_bytes_per_token: slot * u64::from(cap),
        };
        let host = HostBlocks::new(host_blocks + spare, usize::try_from(slot / 2)?)?;
        let (dir, catalogue) = staged(geometry)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let frequency = frequency();
        let initial = initial_residency(&frequency, &capacity.slots_per_layer, &geometry)?;
        let (mut residency, report) = Residency::load(LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows: 8,
            initial: &initial,
            host: Some(&host),
            catalogue: &catalogue,
            reader: &reader,
        })?;
        assert_eq!(
            report.host_bytes,
            12 * slot,
            "only the non-GPU experts are loaded"
        );
        assert_eq!(residency.free_host_blocks(), host_blocks - 12 + spare);
        residency.begin_request(&frequency)?;
        let admitted = host_swap(&mut residency, 0, 0, 2, 1, 90)?;
        let source = admitted.source_block.context("host source missing")?;
        residency.end_request()?;
        let free_before = residency.free_host_blocks();
        assert_eq!(free_before, host_blocks - 12 + spare - 1);
        assert_eq!(residency.released.len(), 1);
        assert_eq!(residency.released[0].block, source);
        assert_eq!(residency.released[0].event, Some(90));
        assert!(residency.borrow(u32::MAX).is_err());
        assert_eq!(residency.free_host_blocks(), free_before);
        assert_eq!(residency.released.len(), 1);
        assert_eq!(residency.ledger().disk_bytes()?, 0);
        let borrow = residency.borrow(2)?;
        assert_eq!((borrow.to_host, borrow.to_disk), (2, 0));
        assert!(borrow.host_waits.iter().all(Option::is_none));
        assert_eq!(
            residency.free_blocks.back().map(|block| block.wait),
            Some(Some(90))
        );
        assert!(residency.released.is_empty());
        assert_eq!(borrow.d2h_bytes, 2 * slot);
        assert_eq!(borrow.disk_bytes, 0);
        assert_eq!(residency.free_host_blocks(), spare);
        assert_eq!(residency.ledger().unique_bytes()?, 16 * slot);
        assert_eq!(residency.ledger().disk_bytes()?, 0);
        let returned = residency.refill()?;
        assert_eq!(returned.len(), 2);
        Ok(())
    }

    #[test]
    fn reserve_floor_covers_zero_cap_host_swaps() -> Result<()> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let slot = bf16_layout(&geometry)?.slot_bytes();
        let inputs = PlanInputs {
            geometry,
            layout: bf16_layout(&geometry)?,
            device_free_bytes: reserves.persistent_bytes(slot)? + 4 * slot,
            host_available_bytes: reserves.host_bytes(slot)? + 3 * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: u64::MAX,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, 0)?;
        assert_eq!(capacity.spare_blocks, 1, "the reserve floor");
        assert_eq!(capacity.host_blocks, 2);
        let admission = Admission {
            budget_bytes_per_token: 0,
        };
        let host = HostBlocks::new(3, usize::try_from(slot / 2)?)?;
        let (dir, catalogue) = staged(geometry)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let initial = initial_residency(&frequency(), &capacity.slots_per_layer, &geometry)?;
        let (mut residency, _report) = Residency::load(LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows: 8,
            initial: &initial,
            host: Some(&host),
            catalogue: &catalogue,
            reader: &reader,
        })?;
        assert_eq!(residency.free_host_blocks(), 1);
        residency.begin_request(&frequency())?;
        residency.end_request()?;
        let mut hot = frequency();
        hot[0][2] = 5.0;
        let moves = residency.begin_request(&hot)?.to_vec();
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].expert, 2);
        assert_eq!(moves[0].dest_block, Some(2));
        assert_eq!(residency.free_host_blocks(), 0);
        Ok(())
    }

    #[test]
    fn slot_events_are_bounded_and_reclaim_both_events() -> Result<()> {
        let mut residency = build(3, 2)?;
        let slot0 = residency.ledger().slot(0, 0)?;
        let slot1 = residency.ledger().slot(0, 1)?;
        residency.record_copy(&[slot0, slot1], 7)?;
        for event in 8..1_008u64 {
            residency.record_readers(&[slot0, slot1], event)?;
        }
        assert_eq!(residency.reclaim(slot0)?, [Some(7), Some(1_007)]);
        assert_eq!(residency.first_read(slot0)?, None);
        let error = residency
            .record_readers(&[slot0], 1_009)
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no recorded copy to read"), "{error}");
        assert_eq!(
            residency.reclaim(slot1)?,
            [Some(7), Some(1_007)],
            "a rejected batch leaves its valid slots unchanged"
        );
        residency.record_copy(&[slot0, slot1], 11)?;
        residency.record_readers(&[slot0, slot1], 12)?;
        assert_eq!(residency.reclaim(slot1)?, [Some(11), Some(12)]);
        Ok(())
    }

    #[test]
    fn borrow_returns_positions_and_respects_the_cap_reserve() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.end_request()?;
        assert_eq!(residency.free_host_blocks(), 2);
        let borrow = residency.borrow(2)?;
        assert_eq!(borrow.positions, vec![3, 2]);
        assert_eq!(borrow.experts, vec![(1, 1), (0, 1)]);
        assert_eq!((borrow.to_host, borrow.to_disk), (0, 2));
        assert!(borrow.host_waits.iter().all(Option::is_none));
        assert_eq!(residency.free_host_blocks(), 2);
        assert_eq!(residency.refill()?, vec![2, 3]);
        assert!(residency.ledger().check().is_ok());
        let (mut backed, _, host, _directory) = build_full(geometry(), 16, 2)?;
        let blocks = backed.host_block.clone();
        let bytes = backed.ledger().unique_bytes()?;
        let frequency = frequency();
        assert!(backed.begin_request(&frequency)?.is_empty());
        for token in 0..4 {
            backed.begin_token()?;
            for layer in 0..2 {
                set_scores(&mut backed, layer, [0.0; 8]);
                backed.dispatch(layer as u32, &[route(0, 0, 6 + token % 2)])?;
                backed.scores[layer][(6 + token % 2) as usize] = 30.0;
            }
            let moves = backed.finish_token()?.to_vec();
            assert!(moves.iter().all(|m| m.dest_block.is_none()));
            for movement in &moves {
                backed.record_copy(&[movement.slot], u64::from(token) + 1)?;
            }
            assert_eq!(backed.host_block, blocks);
            assert_eq!(backed.ledger().unique_bytes()?, bytes);
            assert_eq!(backed.ledger().disk_bytes()?, 0);
            assert_eq!(backed.free_host_blocks(), 0);
        }
        backed.end_request()?;
        let borrow = backed.borrow(2)?;
        assert_eq!((borrow.to_host, borrow.to_disk), (2, 0));
        assert_eq!(borrow.d2h_bytes, 0);
        assert!(borrow.write_back.iter().all(|&copy| !copy));
        assert_eq!(backed.host_block, blocks);
        backed.refill()?;
        let moves = backed.begin_request(&frequency)?.to_vec();
        assert!(moves.iter().all(|m| m.dest_block.is_none()));
        assert_eq!(backed.host_block, blocks);
        assert_eq!(backed.ledger().unique_bytes()?, bytes);
        let lease = host.lease(&(0..16).collect::<Vec<_>>())?;
        assert!(host.claim(0).is_err());
        assert_eq!(lease.read(0)?.len(), host.block_elements());

        let peer_geom = geometry();
        let peer_layout = bf16_layout(&peer_geom)?;
        let peer_slot = peer_layout.slot_bytes();
        let peer_cap = Capacity {
            layout: peer_layout,
            span_rows: 4,
            wave_rows: 2,
            device_expert_bytes: 3 * peer_slot,
            peer_expert_bytes: 8 * peer_slot,
            gpu0_layers: 1,
            host_expert_bytes: 8 * peer_slot,
            slots_per_layer: vec![3, 8],
            host_blocks: 8,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 1,
            pinned: true,
        };
        let peer_host = HostBlocks::new(8, usize::try_from(peer_slot / 2)?)?;
        let (peer_dir, peer_catalogue) = staged(peer_geom)?;
        let peer_reader = DiskReader::open(&peer_catalogue, peer_dir.path(), 1)?;
        let peer_initial = vec![vec![0, 1, 2], (0..8).collect()];
        let (mut peer_backed, _) = Residency::load(LoadInputs {
            geometry: peer_geom,
            capacity: peer_cap,
            admission: Admission {
                budget_bytes_per_token: peer_slot * 2,
            },
            max_rows: 8,
            initial: &peer_initial,
            host: Some(&peer_host),
            catalogue: &peer_catalogue,
            reader: &peer_reader,
        })?;
        assert!(peer_backed.ledger().has_full_host_backing());
        assert!(peer_backed.host_block[0].iter().all(|&b| b != NO_BLOCK));
        assert!(peer_backed.host_block[1].iter().all(|&b| b == NO_BLOCK));
        assert_eq!(peer_backed.ledger().disk_bytes()?, 0);
        let peer_blocks = peer_backed.host_block.clone();
        assert!(peer_backed.begin_request(&frequency)?.is_empty());
        peer_backed.begin_token()?;
        set_scores(&mut peer_backed, 0, [0.0; 8]);
        set_scores(&mut peer_backed, 1, [0.0; 8]);
        peer_backed.dispatch(0, &[route(0, 0, 7)])?;
        peer_backed.dispatch(1, &[route(0, 0, 7)])?;
        peer_backed.scores[0][7] = 30.0;
        let peer_moves = peer_backed.finish_token()?.to_vec();
        assert_eq!(peer_moves.len(), 1);
        assert!(peer_moves.iter().all(|m| m.dest_block.is_none()));
        for movement in &peer_moves {
            peer_backed.record_copy(&[movement.slot], 1)?;
        }
        assert_eq!(peer_backed.host_block, peer_blocks);
        peer_backed.end_request()?;
        let peer_borrow = peer_backed.borrow(2)?;
        assert_eq!(peer_borrow.positions, vec![10, 9]);
        assert_eq!(peer_borrow.d2h_bytes, 0);
        assert!(peer_borrow.write_back.iter().all(|&copy| !copy));
        peer_backed.refill()?;

        println!(
            "PASS residency_complete_backing requests=2 tokens=4 borrow=2 d2h_bytes=0 spare_blocks=0 unique_union=true read_lease=true peer_swappable=true"
        );
        Ok(())
    }
}
