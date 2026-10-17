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
use std::sync::Arc;
use std::time::Instant;

/// Block id of a host expert without a host block.
const NO_BLOCK: u32 = u32::MAX;

/// Setup sources of the host-tier load.
pub struct LoadInputs<'a> {
    pub geometry: ExpertGeometry,
    pub capacity: Capacity,
    pub admission: Admission,
    pub max_rows: u32,
    pub host: &'a Arc<HostBlocks>,
    pub catalogue: &'a ExpertCatalogue,
    pub reader: &'a DiskReader,
    pub prefill_frequency: &'a [Vec<f64>],
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
    /// Physical slot the candidate takes (SlotLayout id).
    pub slot: u32,
    pub generation: u64,
}

/// Destination kind of a planned move before block ids are popped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dest {
    Spare,
    Data,
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
    admission: Admission,
    max_rows: u32,
    ledger: TierLedger,
    events: SlotEvents,
    host_block: Vec<Vec<u32>>,
    /// Data block count: ids `0..host_blocks`; spares follow.
    host_blocks: u32,
    data_blocks: Vec<u32>,
    spares: Vec<u32>,
    released: Vec<u32>,
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
    frequency: &[Vec<f64>],
    initial: &[Vec<u32>],
    geometry: &ExpertGeometry,
    host_blocks: u32,
) -> Result<Vec<(u32, u32)>> {
    let mut order = Vec::with_capacity(host_blocks as usize);
    for layer in 0..geometry.layers {
        let l = layer as usize;
        let mut ranked: Vec<u32> = (0..geometry.experts)
            .filter(|expert| !initial[l].contains(expert))
            .collect();
        ranked.sort_unstable_by(|&a, &b| {
            frequency[l][b as usize]
                .total_cmp(&frequency[l][a as usize])
                .then(a.cmp(&b))
        });
        for expert in ranked {
            if order.len() == host_blocks as usize {
                break;
            }
            order.push((layer, expert));
        }
        if order.len() == host_blocks as usize {
            break;
        }
    }
    ensure!(
        order.len() == host_blocks as usize,
        "host tier {host_blocks} exceeds {} non-GPU experts",
        order.len()
    );
    Ok(order)
}

impl Residency {
    /// Setup: installs the initial GPU set, fills every host data block from the native ranges
    /// through the disk reader, and reports the host-tier load cost. The request boundary only
    /// changes the GPU set.
    pub fn load(inputs: LoadInputs<'_>) -> Result<(Self, LoadReport)> {
        let LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows,
            host,
            catalogue,
            reader,
            prefill_frequency,
        } = inputs;
        let started = Instant::now();
        ensure!(max_rows > 0, "zero dispatch row bound");
        ensure!(
            capacity.slots_per_layer.len() == geometry.layers as usize,
            "slot plan layer count mismatch"
        );
        let slot_bytes = geometry.slot_bytes()?;
        let cap = admission.hard_cap(slot_bytes)?;
        ensure!(
            capacity.spare_blocks == cap,
            "plan reserves {} spare blocks for an admission cap of {cap}",
            capacity.spare_blocks
        );
        let host_blocks = capacity
            .host_blocks
            .checked_add(capacity.spare_blocks)
            .context("host block count overflow")?;
        ensure!(host_blocks > 0, "zero host blocks");
        ensure!(
            host.blocks() == host_blocks,
            "host arena holds {} blocks, plan allocates {host_blocks}",
            host.blocks()
        );
        ensure!(
            host.block_elements() == usize::try_from(slot_bytes / 2)?,
            "host block holds {} elements, a slot holds {}",
            host.block_elements(),
            slot_bytes / 2
        );
        let mut positions = 0u32;
        for &count in &capacity.slots_per_layer {
            positions = positions
                .checked_add(count)
                .context("slot position count overflow")?;
        }
        let initial = initial_residency(prefill_frequency, &capacity.slots_per_layer, &geometry)?;
        let layout = SlotLayout::new(geometry, &capacity)?;
        let mut ledger = TierLedger::new(layout, host_blocks, &initial)?;
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
        let order = host_fill_order(prefill_frequency, &initial, &geometry, capacity.host_blocks)?;
        let mut host_bytes = 0u64;
        for (index, &(layer, expert)) in order.iter().enumerate() {
            let block = u32::try_from(index).context("host block id exceeds u32")?;
            let slot = reader.read_expert(layer, expert, catalogue)?;
            let values = reader.values(slot)?;
            let mut claim = host.claim(block)?;
            claim.slice()?.copy_from_slice(values);
            drop(claim);
            reader.release(slot)?;
            ledger.place_host(layer, expert)?;
            host_block[layer as usize][expert as usize] = block;
            host_bytes = host_bytes
                .checked_add(slot_bytes)
                .context("host load bytes overflow")?;
        }
        let data_blocks: Vec<u32> = (order.len() as u32..capacity.host_blocks).rev().collect();
        let spares: Vec<u32> = (capacity.host_blocks..host_blocks).rev().collect();
        let data_blocks_count = capacity.host_blocks;
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
            host_blocks: data_blocks_count,
            data_blocks,
            spares,
            released: Vec::new(),
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
    /// prompt-boundary moves. The host tier and generations persist across requests; every block
    /// need is validated before the plan is applied.
    pub fn begin_request(&mut self, frequency: &[Vec<f64>]) -> Result<&[PromptMove]> {
        ensure!(!self.token_open, "request boundary inside an open token");
        self.drain_released();
        let initial = initial_residency(frequency, &self.capacity.slots_per_layer, &self.geometry)?;
        let scores = seed_scores(frequency, &self.geometry)?;
        self.moves.clear();
        self.plan.clear();
        let mut spare_need = 0u32;
        let mut data_free =
            u32::try_from(self.data_blocks.len()).context("free data blocks exceed u32")?;
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
                        spare_need += 1;
                        Dest::Spare
                    }
                    (Some(_), Tier::Disk) => {
                        if data_free > 0 {
                            data_free -= 1;
                            Dest::Data
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
        ensure!(
            spare_need <= u32::try_from(self.spares.len()).context("free spares exceed u32")?,
            "request needs {spare_need} spare host blocks, {} free",
            self.spares.len()
        );
        for index in 0..self.plan.len() {
            let planned = self.plan[index];
            let layer = planned.layer;
            let l = layer as usize;
            let expert = planned.expert;
            let (dest_block, slot) = match planned.victim {
                Some(victim) => {
                    let slot = self.ledger.slot(layer, victim)?;
                    let dest = match planned.dest {
                        Dest::Spare => Some(self.spare_block()?),
                        Dest::Data => Some(self.data_block()?),
                        Dest::Disk | Dest::None => None,
                    };
                    match planned.tier {
                        Tier::Host => {
                            self.ledger.admit_host(layer, expert, victim)?;
                            self.host_block[l][expert as usize] = NO_BLOCK;
                            self.host_block[l][victim as usize] =
                                dest.context("host swap without a destination block")?;
                            self.released.push(planned.source_block);
                        }
                        Tier::Disk => {
                            self.ledger
                                .admit_disk(layer, expert, victim, dest.is_some())?;
                            if let Some(block) = dest {
                                self.host_block[l][victim as usize] = block;
                            }
                        }
                        Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
                    }
                    (dest, slot)
                }
                None => {
                    let slot = self.ledger.install(layer, expert)?;
                    if planned.tier == Tier::Host {
                        self.host_block[l][expert as usize] = NO_BLOCK;
                        self.released.push(planned.source_block);
                    }
                    (None, slot)
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
                slot,
                generation: self.next_generation,
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
        self.drain_released();
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

    /// Runs the global §3.1 selection for the token and publishes at most the hard step cap.
    /// Every pair's block needs are validated before the plan is applied.
    pub fn finish_token(&mut self) -> Result<&[Published]> {
        ensure!(self.token_open, "no token open");
        let cap = self.admission.hard_cap(self.geometry.slot_bytes()?)?;
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
        let mut spare_need = 0u32;
        let mut data_free =
            u32::try_from(self.data_blocks.len()).context("free data blocks exceed u32")?;
        for index in 0..self.pending.len() {
            let (_, layer, candidate, _) = self.pending[index];
            if self.ledger.tier(layer, candidate)? == Tier::Host {
                ensure!(
                    self.host_block[layer as usize][candidate as usize] != NO_BLOCK,
                    "host expert {candidate} of layer {layer} has no block"
                );
                spare_need += 1;
            } else {
                data_free = data_free.saturating_sub(1);
            }
        }
        ensure!(
            spare_need <= u32::try_from(self.spares.len()).context("free spares exceed u32")?,
            "host write-backs need {spare_need} spare blocks, {} free",
            self.spares.len()
        );
        self.published.clear();
        for index in 0..self.pending.len() {
            let (_, layer, candidate, victim) = self.pending[index];
            let l = layer as usize;
            let slot = self.ledger.slot(layer, victim)?;
            let from_host = self.ledger.tier(layer, candidate)? == Tier::Host;
            let (source_block, dest_block) = if from_host {
                let block = self.host_block[l][candidate as usize];
                let spare = self.spare_block()?;
                self.ledger.admit_host(layer, candidate, victim)?;
                self.host_block[l][candidate as usize] = NO_BLOCK;
                self.host_block[l][victim as usize] = spare;
                self.released.push(block);
                (Some(block), Some(spare))
            } else {
                let dest = if self.data_blocks.is_empty() {
                    None
                } else {
                    Some(self.data_block()?)
                };
                self.ledger
                    .admit_disk(layer, candidate, victim, dest.is_some())?;
                if let Some(block) = dest {
                    self.host_block[l][victim as usize] = block;
                }
                (None, dest)
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
                slot,
                generation: self.next_generation,
            });
        }
        self.token_open = false;
        debug_assert!(self.ledger.check().is_ok());
        Ok(&self.published)
    }

    /// Records one copy event covering every listed slot.
    pub fn record_copy(&mut self, slots: &[u32], event: u64) -> Result<()> {
        self.events.record_copy(slots, event)
    }

    /// Records one reader event covering every listed slot.
    pub fn record_readers(&mut self, slots: &[u32], event: u64) -> Result<()> {
        self.events.record_readers(slots, event)
    }

    /// Copy event the reader stream waits on for the first read of a slot.
    pub fn first_read(&self, slot: u32) -> Result<Option<u64>> {
        self.events.copy_event(slot)
    }

    /// Reader events the copy stream waits on before overwriting the slot; clears its lifetime.
    pub fn reclaim(&mut self, slot: u32) -> Result<Vec<u64>> {
        self.events.reclaim(slot)
    }

    /// Borrows the resident tail for a prefill span; victims take a free data block or drop to
    /// disk.
    pub fn borrow(&mut self, slots: u32) -> Result<Borrow> {
        let limit = u32::try_from(self.data_blocks.len()).context("free data blocks exceed u32")?;
        let borrow = self.ledger.borrow_tail(slots, limit)?;
        let to_host = borrow.to_host as usize;
        ensure!(
            to_host <= self.data_blocks.len(),
            "borrow assigned {to_host} host victims for {} free data blocks",
            self.data_blocks.len()
        );
        for &(layer, expert) in borrow.experts.iter().take(to_host) {
            let block = self.data_block()?;
            self.host_block[layer as usize][expert as usize] = block;
        }
        Ok(borrow)
    }

    /// Returns every borrowed position to admission.
    pub fn refill(&mut self) -> Result<Vec<u32>> {
        self.ledger.refill_tail()
    }

    fn drain_released(&mut self) {
        for block in self.released.drain(..) {
            if block < self.host_blocks {
                self.data_blocks.push(block);
            } else {
                self.spares.push(block);
            }
        }
    }

    fn spare_block(&mut self) -> Result<u32> {
        self.spares
            .pop()
            .context("spare host block missing from the pool")
    }

    fn data_block(&mut self) -> Result<u32> {
        self.data_blocks
            .pop()
            .context("data host block missing from the pool")
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

    pub fn free_data_blocks(&self) -> u32 {
        u32::try_from(self.data_blocks.len()).expect("free data block count fits u32")
    }

    pub fn free_spare_blocks(&self) -> u32 {
        u32::try_from(self.spares.len()).expect("free spare block count fits u32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::experts::catalogue::ExpertCatalogue;
    use crate::experts::plan::{PlanInputs, Reserves, ServiceCell, ServiceTable, capacity};
    use std::path::{Path, PathBuf};

    fn geometry() -> ExpertGeometry {
        ExpertGeometry {
            layers: 2,
            experts: 8,
            hidden: 8,
            intermediate: 4,
            top_k: 2,
        }
    }

    fn service() -> ServiceTable {
        ServiceTable::new(
            "test",
            vec![ServiceCell {
                workers: 1,
                rows: 1,
                experts: 1,
                seconds: 0.001,
            }],
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
        let catalogue = ExpertCatalogue::from_manifest(&manifest, geometry)?;
        Ok((dir, catalogue))
    }

    fn build_full(
        host_blocks: u32,
        cap: u32,
    ) -> Result<(Residency, LoadReport, Arc<HostBlocks>, TempDir)> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let slot = geometry.slot_bytes()?;
        let inputs = PlanInputs {
            geometry,
            device_free_bytes: reserves.persistent_bytes(&geometry)? + 4 * slot,
            host_available_bytes: reserves.host_bytes(&geometry)?
                + u64::from(host_blocks + cap) * slot,
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
        assert_eq!(capacity.slots_per_layer, vec![2, 2]);
        assert_eq!(capacity.host_blocks, host_blocks);
        let admission = Admission {
            budget_bytes_per_token: slot * u64::from(cap),
        };
        let host = HostBlocks::new(host_blocks + cap, usize::try_from(slot / 2)?)?;
        let (dir, catalogue) = staged(geometry)?;
        let reader = DiskReader::open(&catalogue, dir.path(), 1)?;
        let (residency, report) = Residency::load(LoadInputs {
            geometry,
            capacity,
            admission,
            max_rows: 8,
            host: &host,
            catalogue: &catalogue,
            reader: &reader,
            prefill_frequency: &frequency(),
        })?;
        Ok((residency, report, host, dir))
    }

    fn build(host_blocks: u32, cap: u32) -> Result<Residency> {
        let (mut residency, _, _, _) = build_full(host_blocks, cap)?;
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

    fn set_scores(residency: &mut Residency, layer: usize, scores: [f64; 8]) {
        residency.scores[layer] = scores.to_vec();
    }

    #[test]
    fn load_fills_the_host_tier_from_the_native_ranges() -> Result<()> {
        let (residency, report, host, _dir) = build_full(3, 2)?;
        let slot = geometry().slot_bytes()?;
        assert_eq!(report.host_bytes, 3 * slot);
        assert!(report.seconds.is_finite() && report.seconds >= 0.0);
        for layer in 0..2u32 {
            for expert in 0..2u32 {
                assert_eq!(residency.ledger().tier(layer, expert)?, Tier::Gpu);
            }
        }
        let bytes = pattern(3_072);
        for (layer, expert, block) in [(0u32, 2u32, 0u32), (0, 3, 1), (0, 4, 2)] {
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
        assert_eq!(residency.ledger().tier(0, 5)?, Tier::Disk);
        assert_eq!(residency.free_data_blocks(), 0);
        assert_eq!(residency.free_spare_blocks(), 2);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn request_boundary_swaps_only_the_gpu_set() -> Result<()> {
        let mut residency = build(3, 2)?;
        assert!(residency.begin_request(&frequency())?.is_empty());
        let mut hot = frequency();
        hot[0][2] = 5.0;
        let moves = residency.begin_request(&hot)?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 2));
        assert_eq!(movement.victim, Some(1));
        assert_eq!(movement.source, MoveSource::Host(0));
        assert_eq!(movement.dest_block, Some(3));
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        assert_eq!(residency.host_block(0, 1)?, 3);
        assert_eq!(residency.host_block(0, 2)?, NO_BLOCK);
        residency.begin_token()?;
        assert_eq!(residency.free_data_blocks(), 1);
        residency.finish_token()?.to_vec();
        let moves = residency.begin_request(&frequency())?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 1));
        assert_eq!(movement.victim, Some(2));
        assert_eq!(movement.source, MoveSource::Host(3));
        assert_eq!(movement.dest_block, Some(4));
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Gpu);
        assert_eq!(residency.host_block(0, 2)?, 4);
        assert!(residency.generation(0, 1)? > residency.generation(0, 0)?);
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
        residency.finish_token()?.to_vec();
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
            assert!(entry.dest_block.is_none(), "no free data block at load");
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
        assert_eq!(residency.free_spare_blocks(), 0);
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
    fn a_host_candidate_swaps_through_a_spare_block() -> Result<()> {
        let mut residency = build(3, 2)?;
        let source = residency.host_block(0, 2)?;
        assert_eq!(source, 0);
        let spare = *residency.spares.last().expect("free spare");
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
        assert_eq!(entry.dest_block, Some(spare));
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        assert_eq!(residency.host_block(0, 1)?, spare);
        assert_eq!(residency.host_block(0, 2)?, NO_BLOCK);
        assert_eq!(residency.free_spare_blocks(), 1);
        residency.begin_token()?;
        assert_eq!(
            residency.free_data_blocks(),
            1,
            "the candidate block returns"
        );
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn disk_admission_uses_a_data_block_and_never_a_spare() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        assert_eq!(
            (residency.free_data_blocks(), residency.free_spare_blocks()),
            (1, 1)
        );
        set_scores(
            &mut residency,
            0,
            [1.0, 30.0, 2.0, 20.0, 1.0, 30.0, 0.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 5), route(0, 1, 3)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2);
        assert_eq!((published[0].expert, published[0].victim), (5, 0));
        assert!(!published[0].from_host);
        let data = published[0].dest_block.expect("data block");
        assert!(data < 3, "disk write-back must use a data block");
        assert_eq!(residency.host_block(0, 0)?, data);
        assert_eq!((published[1].expert, published[1].victim), (3, 2));
        assert!(published[1].from_host);
        assert!(published[1].dest_block.expect("spare") >= 3);
        assert_eq!(residency.free_data_blocks(), 0);
        assert_eq!(residency.free_spare_blocks(), 0);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn host_write_backs_without_a_spare_fail_without_side_effects() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 0.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 3)])?;
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        assert_eq!(
            (residency.free_data_blocks(), residency.free_spare_blocks()),
            (2, 0)
        );
        set_scores(
            &mut residency,
            0,
            [1.0, 0.0, 10.0, 1.0, 0.0, 30.0, 0.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 5)])?;
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        assert_eq!(
            (residency.free_data_blocks(), residency.free_spare_blocks()),
            (1, 0)
        );
        let before = (
            residency.ledger().tier(0, 3)?,
            residency.ledger().tier(0, 6)?,
            residency.host_block(0, 4)?,
            residency.free_data_blocks(),
            residency.free_spare_blocks(),
            residency.published.len(),
        );
        set_scores(
            &mut residency,
            0,
            [0.0, 0.0, 1.0, 0.0, 20.0, 2.0, 30.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 6), route(0, 1, 4)])?;
        let error = residency.finish_token().unwrap_err().to_string();
        assert!(error.contains("spare blocks, 0 free"), "{error}");
        let after = (
            residency.ledger().tier(0, 3)?,
            residency.ledger().tier(0, 6)?,
            residency.host_block(0, 4)?,
            residency.free_data_blocks(),
            residency.free_spare_blocks(),
            residency.published.len(),
        );
        assert_eq!(
            before, after,
            "a failed selection leaves the state unchanged"
        );
        assert!(residency.begin_token().is_err(), "the token stays open");
        Ok(())
    }

    #[test]
    fn slot_events_move_to_device_side_waiting() -> Result<()> {
        let mut residency = build(3, 2)?;
        let slot0 = residency.ledger().slot(0, 0)?;
        let slot1 = residency.ledger().slot(0, 1)?;
        let slot2 = residency.ledger().slot(1, 0)?;
        residency.record_copy(&[slot0, slot1, slot2], 7)?;
        for slot in [slot0, slot1, slot2] {
            assert_eq!(residency.first_read(slot)?, Some(7));
        }
        residency.record_readers(&[slot0, slot1], 9)?;
        residency.record_readers(&[slot0], 10)?;
        assert_eq!(residency.reclaim(slot0)?, vec![9, 10]);
        assert_eq!(residency.reclaim(slot1)?, vec![9]);
        assert_eq!(residency.reclaim(slot2)?, Vec::<u64>::new());
        assert_eq!(residency.first_read(slot0)?, None);
        let error = residency
            .record_readers(&[slot0], 11)
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no recorded copy to read"), "{error}");
        residency.record_copy(&[slot0, slot1, slot2], 12)?;
        assert_eq!(residency.first_read(slot2)?, Some(12));
        assert!(residency.reclaim(99).is_err());
        Ok(())
    }

    #[test]
    fn borrow_assigns_data_blocks_and_refill_returns_positions() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(&mut residency, 0, [5.0, 7.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        residency.finish_token()?.to_vec();
        residency.begin_token()?;
        assert_eq!(residency.free_data_blocks(), 1);
        let borrow = residency.borrow(2)?;
        assert_eq!(borrow.experts, vec![(1, 1), (0, 2)]);
        assert_eq!((borrow.to_host, borrow.to_disk), (1, 1));
        assert_eq!(residency.host_block(1, 1)?, 0);
        assert_eq!(residency.ledger().tier(1, 1)?, Tier::Host);
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Disk);
        assert_eq!(residency.free_data_blocks(), 0);
        assert_eq!(residency.refill()?, vec![2, 3]);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }
}
