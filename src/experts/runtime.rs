//! Decode residency runtime: request-boundary moves, dispatch split, admission and slot events.
use crate::cpu::expert_ffn::Assignment;
use crate::experts::arena::SlotLayout;
use crate::experts::catalogue::ExpertGeometry;
use crate::experts::plan::{
    Admission, Capacity, F_DECAY, THETA_MIN, Tier, TierLedger, initial_residency, seed_scores,
};
use anyhow::{Context, Result, ensure};
use half::bf16;

/// Block id of a host expert without a host block.
const NO_BLOCK: u32 = u32::MAX;

/// One routed (row, choice, expert, weight) entry of a layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    pub row: u32,
    pub choice: u32,
    pub expert: u32,
    pub weight: bf16,
}

/// Per-layer dispatch: resident hits with their generation snapshots, host and disk misses.
/// Buffers are allocated once in `Residency::new` and refilled in place.
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

/// Copy and reader lifecycle of one physical slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Copying { event: u64 },
    Ready,
    Readers { event: u64 },
}

struct SlotEvents {
    states: Vec<SlotState>,
}

impl SlotEvents {
    fn new(slots: usize) -> Self {
        Self {
            states: vec![SlotState::Free; slots],
        }
    }

    fn index(&self, slot: u32) -> Result<usize> {
        let index = usize::try_from(slot).context("slot id exceeds usize")?;
        ensure!(
            index < self.states.len(),
            "slot {slot} outside {} positions",
            self.states.len()
        );
        Ok(index)
    }

    fn begin_copy(&mut self, slot: u32, event: u64) -> Result<()> {
        let index = self.index(slot)?;
        ensure!(
            self.states[index] == SlotState::Free,
            "slot {slot} is not free for its copy"
        );
        self.states[index] = SlotState::Copying { event };
        Ok(())
    }

    fn copy_done(&mut self, event: u64) -> Result<()> {
        for state in &mut self.states {
            if let SlotState::Copying { event: in_flight } = *state
                && in_flight == event
            {
                *state = SlotState::Ready;
                return Ok(());
            }
        }
        anyhow::bail!("no slot copy in flight with event {event}")
    }

    fn record_readers(&mut self, slot: u32, event: u64) -> Result<()> {
        let index = self.index(slot)?;
        ensure!(
            self.states[index] == SlotState::Ready,
            "slot {slot} is read before its copy completed"
        );
        self.states[index] = SlotState::Readers { event };
        Ok(())
    }

    fn readers_done(&mut self, event: u64) -> Result<()> {
        for state in &mut self.states {
            if let SlotState::Readers { event: in_flight } = *state
                && in_flight == event
            {
                *state = SlotState::Ready;
                return Ok(());
            }
        }
        anyhow::bail!("no slot readers in flight with event {event}")
    }

    fn reclaim(&mut self, slot: u32) -> Result<()> {
        let index = self.index(slot)?;
        match self.states[index] {
            SlotState::Free | SlotState::Ready => {
                self.states[index] = SlotState::Free;
                Ok(())
            }
            SlotState::Copying { event } => {
                anyhow::bail!("slot {slot} is still copying (event {event})")
            }
            SlotState::Readers { event } => {
                anyhow::bail!("slot {slot} still has readers (event {event})")
            }
        }
    }
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
    free_blocks: Vec<u32>,
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

impl Residency {
    /// Starts with an empty ledger; `begin_request` installs the request's initial GPU set.
    pub fn new(
        geometry: ExpertGeometry,
        capacity: Capacity,
        admission: Admission,
        max_rows: u32,
    ) -> Result<Self> {
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
        let mut positions = 0u32;
        for &count in &capacity.slots_per_layer {
            positions = positions
                .checked_add(count)
                .context("slot position count overflow")?;
        }
        let layout = SlotLayout::new(geometry, &capacity)?;
        let ledger = TierLedger::empty(layout, host_blocks)?;
        let experts = geometry.experts as usize;
        let layers = geometry.layers as usize;
        let max_routes = max_rows as usize * geometry.top_k as usize;
        let select_capacity = (positions as usize).max(cap as usize);
        let dispatches = (0..layers)
            .map(|_| fresh_dispatch(max_routes, experts, max_rows as usize))
            .collect();
        Ok(Self {
            geometry,
            capacity,
            admission,
            max_rows,
            ledger,
            events: SlotEvents::new(positions as usize),
            host_block: vec![vec![NO_BLOCK; experts]; layers],
            free_blocks: (0..host_blocks).rev().collect(),
            released: Vec::with_capacity(host_blocks as usize),
            generation: vec![vec![0u64; experts]; layers],
            next_generation: 0,
            scores: vec![vec![0.0; experts]; layers],
            routed: vec![vec![false; experts]; layers],
            missed: vec![Vec::with_capacity(experts); layers],
            dispatches,
            out: Vec::with_capacity(experts),
            incoming: Vec::with_capacity(experts),
            victims: Vec::with_capacity(experts),
            candidates: Vec::with_capacity(experts),
            pending: Vec::with_capacity(select_capacity),
            published: Vec::with_capacity(select_capacity),
            moves: Vec::with_capacity(positions as usize),
            dispatched: vec![false; layers],
            token_open: false,
            request_open: false,
        })
    }

    /// Starts a request: installs the initial GPU set from the prefill frequency, reseeds F and
    /// emits the prompt-boundary moves. The host tier and generations persist across requests.
    pub fn begin_request(&mut self, frequency: &[Vec<f64>]) -> Result<&[PromptMove]> {
        ensure!(!self.token_open, "request boundary inside an open token");
        self.drain_released();
        let initial = initial_residency(frequency, &self.capacity.slots_per_layer, &self.geometry)?;
        self.scores = seed_scores(frequency, &self.geometry)?;
        self.moves.clear();
        for layer in 0..self.geometry.layers {
            let l = layer as usize;
            let target = &initial[l];
            self.out.clear();
            for expert in 0..self.geometry.experts {
                if self.ledger.tier(layer, expert)? == Tier::Gpu
                    && target.binary_search(&expert).is_err()
                {
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
                let (source, dest_block, slot) = match victim {
                    Some(victim) => {
                        let slot = self.ledger.slot(layer, victim)?;
                        self.events.reclaim(slot)?;
                        let (source, dest_block) = match tier {
                            Tier::Host => {
                                let block = self.host_block[l][expert as usize];
                                ensure!(
                                    block != NO_BLOCK,
                                    "host expert {expert} of layer {layer} has no block"
                                );
                                ensure!(
                                    !self.free_blocks.is_empty(),
                                    "no free host block for the prompt write-back"
                                );
                                let spare = self.spare_block()?;
                                self.ledger.admit_host(layer, expert, victim)?;
                                self.host_block[l][expert as usize] = NO_BLOCK;
                                self.host_block[l][victim as usize] = spare;
                                self.released.push(block);
                                (MoveSource::Host(block), Some(spare))
                            }
                            Tier::Disk => {
                                let dest = if self.free_blocks.is_empty() {
                                    None
                                } else {
                                    Some(self.spare_block()?)
                                };
                                self.ledger
                                    .admit_disk(layer, expert, victim, dest.is_some())?;
                                if let Some(block) = dest {
                                    self.host_block[l][victim as usize] = block;
                                }
                                (MoveSource::Disk, dest)
                            }
                            Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
                        };
                        paired += 1;
                        (source, dest_block, slot)
                    }
                    None => {
                        let position = self.ledger.empty_slot(layer)?.with_context(|| {
                            format!("layer {layer} has no empty slot for expert {expert}")
                        })?;
                        self.events.reclaim(position)?;
                        let slot = self.ledger.install(layer, expert)?;
                        ensure!(slot == position, "install moved the candidate slot");
                        let source = match tier {
                            Tier::Host => {
                                let block = self.host_block[l][expert as usize];
                                ensure!(
                                    block != NO_BLOCK,
                                    "host expert {expert} of layer {layer} has no block"
                                );
                                self.host_block[l][expert as usize] = NO_BLOCK;
                                self.released.push(block);
                                MoveSource::Host(block)
                            }
                            Tier::Disk => MoveSource::Disk,
                            Tier::Gpu => unreachable!("incoming experts are not GPU resident"),
                        };
                        (source, None, slot)
                    }
                };
                self.next_generation += 1;
                self.generation[l][expert as usize] = self.next_generation;
                self.moves.push(PromptMove {
                    layer,
                    expert,
                    victim,
                    source,
                    dest_block,
                    slot,
                    generation: self.next_generation,
                });
            }
            ensure!(
                paired == self.out.len(),
                "layer {layer} leaves {} evicted experts without a replacement",
                self.out.len() - paired
            );
        }
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
    pub fn finish_token(&mut self) -> Result<&[Published]> {
        ensure!(self.token_open, "no token open");
        self.published.clear();
        let cap = self.admission.hard_cap(self.geometry.slot_bytes()?)?;
        if cap == 0 {
            self.token_open = false;
            return Ok(&self.published);
        }
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
        let mut needed = 0u32;
        for index in 0..self.pending.len() {
            let (_, layer, candidate, victim) = self.pending[index];
            let slot = self.ledger.slot(layer, victim)?;
            self.events.reclaim(slot)?;
            if self.ledger.tier(layer, candidate)? == Tier::Host {
                ensure!(
                    self.host_block[layer as usize][candidate as usize] != NO_BLOCK,
                    "host expert {candidate} of layer {layer} has no block"
                );
                needed += 1;
            }
        }
        ensure!(
            u32::try_from(self.free_blocks.len()).context("free block count exceeds u32")?
                >= needed,
            "host write-backs need {needed} free blocks, {} free",
            self.free_blocks.len()
        );
        for index in 0..self.pending.len() {
            let (_, layer, candidate, victim) = self.pending[index];
            let slot = self.ledger.slot(layer, victim)?;
            let from_host = self.ledger.tier(layer, candidate)? == Tier::Host;
            let (source_block, dest_block) = if from_host {
                let block = self.host_block[layer as usize][candidate as usize];
                let spare = self.spare_block()?;
                self.ledger.admit_host(layer, candidate, victim)?;
                self.host_block[layer as usize][candidate as usize] = NO_BLOCK;
                self.host_block[layer as usize][victim as usize] = spare;
                self.released.push(block);
                (Some(block), Some(spare))
            } else {
                let dest = if self.free_blocks.is_empty() {
                    None
                } else {
                    Some(self.spare_block()?)
                };
                self.ledger
                    .admit_disk(layer, candidate, victim, dest.is_some())?;
                if let Some(block) = dest {
                    self.host_block[layer as usize][victim as usize] = block;
                }
                (None, dest)
            };
            self.next_generation += 1;
            self.generation[layer as usize][candidate as usize] = self.next_generation;
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

    /// Records the copy event of a published slot before the engine reads it.
    pub fn record_copy(&mut self, slot: u32, event: u64) -> Result<()> {
        self.events.begin_copy(slot, event)
    }

    /// Records completion of the copy event started with `record_copy`.
    pub fn copy_done(&mut self, event: u64) -> Result<()> {
        self.events.copy_done(event)
    }

    /// Records that readers of a resident slot started; the slot cannot be overwritten until
    /// `readers_done`.
    pub fn record_readers(&mut self, slot: u32, event: u64) -> Result<()> {
        self.events.record_readers(slot, event)
    }

    /// Records completion of the readers started with `record_readers`.
    pub fn readers_done(&mut self, event: u64) -> Result<()> {
        self.events.readers_done(event)
    }

    fn drain_released(&mut self) {
        self.free_blocks.append(&mut self.released);
    }

    fn spare_block(&mut self) -> Result<u32> {
        self.free_blocks
            .pop()
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
        u32::try_from(self.free_blocks.len()).expect("free block count fits u32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::plan::{PlanInputs, Reserves, ServiceCell, ServiceTable, capacity};

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

    fn build(host_blocks: u32, cap: u32) -> Result<Residency> {
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
        let mut residency = Residency::new(geometry, capacity, admission, 8)?;
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
    fn request_boundary_installs_the_initial_set_and_tracks_moves() -> Result<()> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let slot = geometry.slot_bytes()?;
        let inputs = PlanInputs {
            geometry,
            device_free_bytes: reserves.persistent_bytes(&geometry)? + 4 * slot,
            host_available_bytes: reserves.host_bytes(&geometry)? + 14 * slot,
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
        let capacity = capacity(&inputs, 2)?;
        assert_eq!(capacity.host_blocks, 12);
        let admission = Admission {
            budget_bytes_per_token: 2 * slot,
        };
        let mut residency = Residency::new(geometry, capacity, admission, 8)?;
        assert_eq!(residency.free_host_blocks(), 14);
        let moves = residency.begin_request(&frequency())?.to_vec();
        assert_eq!(moves.len(), 4, "two slots per layer are installed");
        for (index, movement) in moves.iter().enumerate() {
            assert_eq!(movement.source, MoveSource::Disk);
            assert!(movement.victim.is_none());
            assert!(movement.dest_block.is_none());
            assert!(movement.generation > 0);
            if index > 0 {
                assert!(movement.generation > moves[index - 1].generation);
            }
            assert_eq!(
                residency.ledger().tier(movement.layer, movement.expert)?,
                Tier::Gpu
            );
            assert_eq!(
                residency.ledger().slot(movement.layer, movement.expert)?,
                movement.slot
            );
        }
        assert_eq!(residency.ledger().slot(0, 0)?, 0);
        assert_eq!(residency.ledger().slot(1, 0)?, 1);
        assert_eq!(residency.ledger().slot(0, 1)?, 2);
        assert_eq!(residency.ledger().slot(1, 1)?, 3);
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Disk);
        assert!(residency.ledger().check().is_ok());

        let mut hot = frequency();
        hot[0] = vec![1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let moves = residency.begin_request(&hot)?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 2));
        assert_eq!(movement.victim, Some(1));
        assert_eq!(movement.source, MoveSource::Disk);
        assert!(movement.dest_block.is_some());
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        assert_eq!(
            residency.host_block(0, 1)?,
            movement.dest_block.expect("victim block")
        );
        assert!(residency.generation(0, 2)? > residency.generation(0, 0)?);
        assert_eq!(residency.free_host_blocks(), 13);

        let moves = residency.begin_request(&frequency())?.to_vec();
        assert_eq!(moves.len(), 1);
        let movement = moves[0];
        assert_eq!((movement.layer, movement.expert), (0, 1));
        assert_eq!(movement.victim, Some(2));
        assert!(matches!(movement.source, MoveSource::Host(_)));
        assert!(movement.dest_block.is_some());
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 2)?, Tier::Host);
        assert_eq!(
            residency.host_block(0, 2)?,
            movement.dest_block.expect("victim block")
        );
        assert_eq!(residency.free_host_blocks(), 12);
        residency.begin_token()?;
        assert_eq!(
            residency.free_host_blocks(),
            13,
            "the candidate block returns to the pool at the next token"
        );
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn dispatch_partitions_every_choice_exactly_once() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 7)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].victim, 0);
        residency.begin_token()?;
        let routes = vec![
            route(0, 0, 0),
            route(0, 1, 1),
            route(1, 2, 6),
            route(1, 3, 7),
        ];
        {
            let dispatch = residency.dispatch(0, &routes)?;
            assert_eq!(dispatch.active, vec![0, 1, 0, 1]);
            assert_eq!(dispatch.generation, vec![0, 2, 0, 5]);
            assert_eq!(dispatch.miss_rows, vec![0, 1]);
            assert_eq!(dispatch.host_experts, vec![0]);
            assert_eq!(dispatch.host.len(), 1);
            assert_eq!(dispatch.host[0].row, 0);
            assert_eq!(dispatch.host[0].choice, 0);
            assert_eq!(dispatch.host[0].expert, 0);
            assert_eq!(dispatch.disk.len(), 1);
            assert_eq!(dispatch.disk[0].choice, 2);
            assert_eq!(dispatch.disk[0].expert, 6);
            assert_eq!(dispatch.disk[0].row, 1);
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
        assert!(residency.ledger().check().is_ok());
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
        for entry in &published {
            assert_eq!(
                residency.ledger().tier(entry.layer, entry.expert)?,
                Tier::Gpu
            );
        }
        assert_eq!(residency.ledger().tier(0, 0)?, Tier::Host);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
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
        assert_eq!(residency.ledger().tier(0, 3)?, Tier::Disk);
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
        assert_eq!(residency.ledger().tier(0, 0)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        Ok(())
    }

    #[test]
    fn a_host_candidate_swaps_through_a_spare_block() -> Result<()> {
        let mut residency = build(3, 2)?;
        residency.begin_token()?;
        set_scores(
            &mut residency,
            0,
            [30.0, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 2)])?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 admission");
        assert_eq!((entry.expert, entry.victim), (2, 1));
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Host);
        let victim_block = residency.host_block(0, 1)?;
        assert_eq!(entry.dest_block, Some(victim_block));
        residency.begin_token()?;
        set_scores(
            &mut residency,
            0,
            [5.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 1)])?;
        let spare = *residency
            .free_blocks
            .last()
            .context("free host block for the write-back")?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 host admission");
        assert!(entry.from_host);
        assert_eq!((entry.expert, entry.victim), (1, 0));
        assert_eq!(entry.source_block, Some(victim_block));
        assert_eq!(entry.dest_block, Some(spare));
        assert_eq!(residency.ledger().tier(0, 1)?, Tier::Gpu);
        assert_eq!(residency.host_block(0, 0)?, spare);
        assert_eq!(residency.host_block(0, 1)?, NO_BLOCK);
        assert!(residency.ledger().check().is_ok());
        Ok(())
    }

    #[test]
    fn slot_events_gate_reads_on_copies_and_overwrites_on_readers() -> Result<()> {
        let mut residency = build(3, 2)?;
        let slot = residency.ledger().slot(0, 0)?;
        let error = residency.record_readers(slot, 10).unwrap_err().to_string();
        assert!(error.contains("read before its copy completed"), "{error}");
        residency.record_copy(slot, 11)?;
        let error = residency.record_readers(slot, 12).unwrap_err().to_string();
        assert!(error.contains("read before its copy completed"), "{error}");
        let error = residency.copy_done(99).unwrap_err().to_string();
        assert!(error.contains("no slot copy in flight"), "{error}");
        residency.copy_done(11)?;
        residency.record_readers(slot, 12)?;
        residency.begin_token()?;
        set_scores(
            &mut residency,
            0,
            [1.0, 30.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        residency.dispatch(0, &[route(0, 0, 2)])?;
        let error = residency.finish_token().unwrap_err().to_string();
        assert!(error.contains("still has readers"), "{error}");
        residency.readers_done(12)?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 admission");
        assert_eq!((entry.expert, entry.victim), (2, 0));
        residency.record_copy(entry.slot, 13)?;
        let error = residency.copy_done(12).unwrap_err().to_string();
        assert!(error.contains("no slot copy in flight"), "{error}");
        residency.copy_done(13)?;
        Ok(())
    }
}
