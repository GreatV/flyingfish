//! Decode residency runtime: dispatch split, decayed-score admission and published generations.
use crate::cpu::expert_ffn::Assignment;
use crate::experts::arena::HostBlocks;
use crate::experts::catalogue::ExpertGeometry;
use crate::experts::plan::{
    Admission, Capacity, F_DECAY, THETA_MIN, Tier, TierLedger, seed_scores,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

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

/// One disk miss; the engine reads it from the native shard ranges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskMiss {
    pub row: u32,
    pub choice: u32,
    pub expert: u32,
    pub weight: bf16,
}

/// Per-layer dispatch: resident hits with their generation snapshots, host and disk misses.
/// Buffers are allocated once in `Residency::new` and refilled in place.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    pub layer: u32,
    /// 1 for a GPU-resident hit, 0 for a miss; index = route order.
    pub active: Vec<u32>,
    /// Published generation per route; meaningful only for hits.
    pub generation: Vec<u64>,
    /// Host misses; `row` indexes `miss_rows`, `expert` indexes `host_experts`.
    pub host: Vec<Assignment>,
    /// Distinct host experts in compacted order.
    pub host_experts: Vec<u32>,
    /// Disk misses; `row` indexes `miss_rows`, `expert` is the logical expert id.
    pub disk: Vec<DiskMiss>,
    /// Distinct rows with any miss, ascending.
    pub miss_rows: Vec<u32>,
}

/// One admission published for the next token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Published {
    pub layer: u32,
    pub expert: u32,
    pub victim: u32,
    /// Candidate was host-resident and its block moved to the victim.
    pub from_host: bool,
    /// Victim was written back into a spare host block.
    pub host_block_free: bool,
    /// Generation the candidate takes in the device table.
    pub generation: u64,
}

/// Decode residency state for one request.
pub struct Residency {
    geometry: ExpertGeometry,
    capacity: Capacity,
    admission: Admission,
    host: Arc<HostBlocks>,
    ledger: TierLedger,
    host_block: Vec<Vec<u32>>,
    free_blocks: Vec<u32>,
    generation: Vec<Vec<u64>>,
    next_generation: u64,
    scores: Vec<Vec<f64>>,
    routed: Vec<Vec<bool>>,
    missed: Vec<Vec<u32>>,
    dispatches: Vec<Dispatch>,
    pending: Vec<(f64, u32, u32, u32, bool)>,
    published: Vec<Published>,
    token_open: bool,
}

fn compact_row(miss_rows: &[u32], row: u32) -> Result<u32> {
    u32::try_from(
        miss_rows
            .binary_search(&row)
            .map_err(|_| anyhow::anyhow!("miss row {row} is not compacted"))?,
    )
    .context("miss row index exceeds u32")
}

fn fresh_dispatch(layer: u32, max_routes: usize, experts: usize, max_rows: usize) -> Dispatch {
    Dispatch {
        layer,
        active: Vec::with_capacity(max_routes),
        generation: Vec::with_capacity(max_routes),
        host: Vec::with_capacity(max_routes),
        host_experts: Vec::with_capacity(experts),
        disk: Vec::with_capacity(max_routes),
        miss_rows: Vec::with_capacity(max_rows),
    }
}

impl Residency {
    /// Starts a request: initial GPU set by prefill frequency, host tier empty, F seeded.
    pub fn new(
        geometry: ExpertGeometry,
        capacity: Capacity,
        admission: Admission,
        host: Arc<HostBlocks>,
        initial: &[Vec<u32>],
        prefill_frequency: &[Vec<f64>],
        max_rows: u32,
    ) -> Result<Self> {
        ensure!(max_rows > 0, "zero dispatch row bound");
        let ledger = TierLedger::new(
            geometry,
            capacity.slots_per_layer.clone(),
            capacity.host_blocks,
            initial,
        )?;
        let blocks = capacity
            .host_blocks
            .checked_add(capacity.spare_blocks)
            .context("host block count overflow")?;
        ensure!(
            host.blocks() == blocks,
            "host arena holds {} blocks, plan allocates {blocks}",
            host.blocks()
        );
        let scores = seed_scores(prefill_frequency, &geometry)?;
        let mut generation = vec![vec![0u64; geometry.experts as usize]; geometry.layers as usize];
        for (layer, set) in initial.iter().enumerate() {
            for &expert in set {
                generation[layer][expert as usize] = 1;
            }
        }
        let max_routes = max_rows as usize * geometry.top_k as usize;
        let dispatches = (0..geometry.layers)
            .map(|layer| {
                fresh_dispatch(
                    layer,
                    max_routes,
                    geometry.experts as usize,
                    max_rows as usize,
                )
            })
            .collect();
        Ok(Self {
            geometry,
            capacity,
            admission,
            host,
            ledger,
            host_block: vec![vec![NO_BLOCK; geometry.experts as usize]; geometry.layers as usize],
            free_blocks: (0..blocks).rev().collect(),
            generation,
            next_generation: 1,
            scores,
            routed: vec![vec![false; geometry.experts as usize]; geometry.layers as usize],
            missed: vec![Vec::with_capacity(geometry.experts as usize); geometry.layers as usize],
            dispatches,
            pending: Vec::with_capacity(max_routes * geometry.layers as usize),
            published: Vec::with_capacity(max_routes * geometry.layers as usize),
            token_open: false,
        })
    }

    pub fn begin_token(&mut self) -> Result<()> {
        ensure!(!self.token_open, "token already open");
        self.token_open = true;
        for layer in 0..self.geometry.layers as usize {
            self.routed[layer].fill(false);
            self.missed[layer].clear();
        }
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
        let (l, experts) = (layer as usize, self.geometry.experts);
        {
            let dispatch = &mut self.dispatches[l];
            ensure!(
                routes.len() <= dispatch.active.capacity(),
                "{} routes exceed the {} row bound",
                routes.len(),
                dispatch.active.capacity()
            );
        }
        for (index, route) in routes.iter().enumerate() {
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
            ensure!(
                route.expert < experts,
                "route expert {} outside {experts}",
                route.expert
            );
            let e = route.expert as usize;
            if !self.routed[l][e] {
                self.routed[l][e] = true;
                self.scores[l][e] += 1.0;
            }
        }
        for route in routes {
            match self.ledger.tier(layer, route.expert)? {
                Tier::Gpu => {
                    dispatch.active.push(1);
                    dispatch
                        .generation
                        .push(self.generation[l][route.expert as usize]);
                }
                Tier::Host => {
                    dispatch.active.push(0);
                    dispatch.generation.push(0);
                    if !dispatch.miss_rows.contains(&route.row) {
                        dispatch.miss_rows.push(route.row);
                    }
                    dispatch.host_experts.push(route.expert);
                }
                Tier::Disk => {
                    dispatch.active.push(0);
                    dispatch.generation.push(0);
                    if !dispatch.miss_rows.contains(&route.row) {
                        dispatch.miss_rows.push(route.row);
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
                    dispatch.disk.push(DiskMiss {
                        row,
                        choice: route.choice,
                        expert: route.expert,
                        weight: route.weight,
                    });
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
        Ok(dispatch)
    }

    /// Runs the global §3.1 selection for the token and publishes at most the hard step cap.
    pub fn finish_token(&mut self) -> Result<&[Published]> {
        ensure!(self.token_open, "no token open");
        self.token_open = false;
        self.published.clear();
        let slot_bytes = self.geometry.slot_bytes()?;
        let cap = self.admission.hard_cap(slot_bytes);
        if cap == 0 {
            return Ok(&self.published);
        }
        self.pending.clear();
        for layer in 0..self.geometry.layers {
            let l = layer as usize;
            if self.missed[l].is_empty() {
                continue;
            }
            let victim = (0..self.geometry.experts)
                .filter(|&expert| {
                    self.ledger
                        .tier(layer, expert)
                        .is_ok_and(|tier| tier == Tier::Gpu)
                        && !self.routed[l][expert as usize]
                })
                .min_by(|&a, &b| {
                    self.scores[l][a as usize]
                        .total_cmp(&self.scores[l][b as usize])
                        .then(a.cmp(&b))
                });
            let Some(victim) = victim else { continue };
            for &candidate in &self.missed[l] {
                let candidate_score = self.scores[l][candidate as usize];
                let delta = candidate_score - self.scores[l][victim as usize];
                if candidate_score >= THETA_MIN && delta > 0.0 {
                    let from_host = self.ledger.tier(layer, candidate)? == Tier::Host;
                    self.pending
                        .push((delta, layer, candidate, victim, from_host));
                }
            }
        }
        self.pending
            .sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for &(_, layer, candidate, victim, from_host) in &self.pending {
            if self.published.len() as u32 >= cap {
                break;
            }
            let host_block_free = self.ledger.free_host_block();
            if from_host {
                self.ledger.admit_host(layer, candidate, victim)?;
                let block = self.host_block[layer as usize][candidate as usize];
                ensure!(block != NO_BLOCK, "host expert {candidate} has no block");
                self.host_block[layer as usize][candidate as usize] = NO_BLOCK;
                self.host_block[layer as usize][victim as usize] = block;
            } else {
                self.ledger
                    .admit_disk(layer, candidate, victim, host_block_free)?;
                if host_block_free {
                    let block = self
                        .free_blocks
                        .pop()
                        .context("free host block missing from the pool")?;
                    self.host_block[layer as usize][victim as usize] = block;
                }
            }
            self.next_generation += 1;
            self.generation[layer as usize][candidate as usize] = self.next_generation;
            self.published.push(Published {
                layer,
                expert: candidate,
                victim,
                from_host,
                host_block_free,
                generation: self.next_generation,
            });
        }
        self.ledger.check()?;
        Ok(&self.published)
    }

    pub fn ledger(&self) -> &TierLedger {
        &self.ledger
    }

    pub fn capacity(&self) -> &Capacity {
        &self.capacity
    }

    pub fn host_arena(&self) -> &Arc<HostBlocks> {
        &self.host
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
    use crate::experts::plan::{
        Admission, PlanInputs, Reserves, ServiceCell, ServiceTable, capacity,
    };

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
        let mut layer0 = vec![1.0; 8];
        layer0[7] = 5.0;
        let mut layer1 = vec![1.0; 8];
        layer1[6] = 5.0;
        vec![layer0, layer1]
    }

    fn build(host_blocks: u32) -> Result<Residency> {
        let geometry = geometry();
        let reserves = reserves();
        let service = service();
        let slot = geometry.slot_bytes()?;
        let inputs = PlanInputs {
            geometry,
            device_free_bytes: reserves.persistent_bytes(&geometry)? + 4 * slot,
            host_available_bytes: reserves.host_bytes(&geometry)?
                + u64::from(host_blocks + 2) * slot,
            h2d_bytes_per_second: 1e9,
            d2h_bytes_per_second: 1e9,
            copy_engines: 2,
            disk_readers: 1,
            service: &service,
            reserves: &reserves,
            span_rows: 4,
            wave_rows: 2,
        };
        let capacity = capacity(&inputs, 2)?;
        assert_eq!(capacity.slots_per_layer, vec![2, 2]);
        assert_eq!(capacity.host_blocks, host_blocks);
        let blocks = capacity.host_blocks + capacity.spare_blocks;
        let host = HostBlocks::new(blocks, (geometry.slot_bytes()? / 2) as usize)?;
        let initial = vec![vec![0, 1], vec![0, 1]];
        Residency::new(
            geometry,
            capacity,
            Admission {
                budget_bytes_per_token: geometry.slot_bytes()? * 2,
            },
            host,
            &initial,
            &frequency(),
            8,
        )
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
    fn dispatch_partitions_every_choice_exactly_once() -> Result<()> {
        let mut residency = build(3)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 7)])?;
        residency.finish_token()?.to_vec();
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
            assert_eq!(dispatch.generation, vec![0, 1, 0, dispatch.generation[3]]);
            assert_eq!(dispatch.miss_rows, vec![0, 1]);
            assert_eq!(dispatch.host_experts, vec![0]);
            assert_eq!(dispatch.host.len(), 1);
            assert_eq!(dispatch.host[0].row, 0);
            assert_eq!(dispatch.host[0].expert, 0);
            assert_eq!(dispatch.disk.len(), 1);
            assert_eq!(dispatch.disk[0].expert, 6);
            assert_eq!(dispatch.disk[0].row, 1);
        }
        let mut rejected = routes;
        rejected.push(route(0, 0, 2));
        assert!(residency.dispatch(0, &rejected).is_err());
        Ok(())
    }

    #[test]
    fn finish_token_admits_under_the_cap_and_bumps_generations() -> Result<()> {
        let mut residency = build(3)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 7)])?;
        residency.dispatch(1, &[route(0, 0, 1), route(0, 1, 6)])?;
        let published = residency.finish_token()?.to_vec();
        assert_eq!(published.len(), 2, "hard cap is 2 experts per token");
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
    fn host_admission_moves_the_block_to_the_victim() -> Result<()> {
        let mut residency = build(3)?;
        residency.begin_token()?;
        residency.dispatch(0, &[route(0, 0, 0), route(0, 1, 7)])?;
        let published = residency.finish_token()?.to_vec();
        let entry = published
            .iter()
            .find(|entry| entry.layer == 0)
            .expect("layer 0 admission");
        assert!(!entry.from_host);
        assert!(entry.host_block_free);
        assert_eq!(residency.ledger().tier(0, entry.expert)?, Tier::Gpu);
        assert_eq!(residency.ledger().tier(0, entry.victim)?, Tier::Host);
        assert_ne!(residency.host_block(0, entry.victim)?, NO_BLOCK);
        assert_eq!(residency.free_host_blocks(), 4);
        Ok(())
    }
}
