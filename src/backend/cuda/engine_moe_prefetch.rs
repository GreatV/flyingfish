//! Bounded native disk reads with one owner of the registered ring.
use super::*;
use crate::experts::span::{Prefetch, Requested, Retired};
use std::sync::mpsc::sync_channel;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread::JoinHandle;

struct Read {
    reader: Reader,
    catalogue: Arc<ExpertCatalogue>,
    layer: u32,
    expert: u32,
    reclaimed: Vec<u32>,
}

struct Completed {
    reader: Reader,
    layer: u32,
    expert: u32,
    result: Result<u32>,
    reclaimed: Vec<u32>,
}

pub(super) struct ReadAhead {
    queue: Option<Prefetch>,
    depth: u32,
    input: Option<SyncSender<Read>>,
    output: Option<Receiver<Completed>>,
    worker: Option<JoinHandle<()>>,
    pending: bool,
    failed: bool,
    predictions: Vec<(u32, u32)>,
    forecast: Vec<(u32, u32)>,
    consumed: Vec<(u32, u32)>,
    taken: Vec<bool>,
    recycled: Vec<u32>,
    pub stats: [u64; 6],
}

impl ReadAhead {
    pub fn new(staging: u32, rings: u32, experts: u32) -> Result<Self> {
        ensure!(
            staging > 0 && experts > 0,
            "prefetch staging={staging} experts={experts}"
        );
        let depth = staging.min(rings);
        if rings == 0 {
            return Ok(Self {
                queue: None,
                depth,
                input: None,
                output: None,
                worker: None,
                pending: false,
                failed: false,
                predictions: Vec::new(),
                forecast: Vec::with_capacity(experts as usize),
                consumed: Vec::with_capacity(experts as usize),
                taken: Vec::new(),
                recycled: Vec::new(),
                stats: [0; 6],
            });
        }
        let queue = Some(Prefetch::new(depth, rings)?);
        let (input, requests) = sync_channel::<Read>(1);
        let (results, output) = sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("expert-read-ahead".into())
            .spawn(move || {
                let mut copied = vec![false; rings as usize];
                while let Ok(mut request) = requests.recv() {
                    for (previous, copy) in copied.iter_mut().zip(&request.reader.copied) {
                        *previous = copy.is_some();
                    }
                    let result = (|| {
                        request.reader.context.bind_to_thread()?;
                        request
                            .reader
                            .read(request.layer, request.expert, &request.catalogue)
                    })();
                    request.reclaimed.clear();
                    for (slot, &was_copied) in copied.iter().enumerate() {
                        if was_copied && request.reader.copied[slot].is_none() {
                            request.reclaimed.push(slot as u32);
                        }
                    }
                    if results
                        .send(Completed {
                            reader: request.reader,
                            layer: request.layer,
                            expert: request.expert,
                            result,
                            reclaimed: request.reclaimed,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .context("spawn native expert read-ahead worker")?;
        Ok(Self {
            queue,
            depth,
            input: Some(input),
            output: Some(output),
            worker: Some(worker),
            pending: false,
            failed: false,
            predictions: Vec::with_capacity(depth as usize),
            forecast: Vec::with_capacity(experts as usize),
            consumed: Vec::with_capacity(experts as usize),
            taken: vec![false; rings as usize],
            recycled: Vec::with_capacity(rings as usize),
            stats: [0; 6],
        })
    }

    fn release(&mut self, ring: u32) -> Result<()> {
        let taken = self
            .taken
            .get_mut(ring as usize)
            .context("prefetch reclaimed ring outside allocation")?;
        if *taken {
            ensure!(
                self.queue
                    .as_mut()
                    .context("prefetch ring has no queue")?
                    .release(ring),
                "prefetch taken ring={ring} has no queue owner"
            );
            *taken = false;
        }
        Ok(())
    }

    fn receive(&mut self, slots: &mut Slots, wait: bool) -> Result<bool> {
        if !self.pending {
            return Ok(false);
        }
        let completed = if wait {
            Some(
                self.output
                    .as_ref()
                    .context("prefetch pending read has no receiver")?
                    .recv()
                    .context("native read-ahead worker stopped")?,
            )
        } else {
            match self
                .output
                .as_ref()
                .context("prefetch pending read has no receiver")?
                .try_recv()
            {
                Ok(completed) => Some(completed),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    anyhow::bail!("native read-ahead worker disconnected")
                }
            }
        };
        let Some(completed) = completed else {
            return Ok(false);
        };
        ensure!(
            slots.disk.is_none(),
            "prefetch completion has a second disk reader owner"
        );
        slots.disk = Some(completed.reader);
        self.pending = false;
        let mut reclaimed = completed.reclaimed;
        for ring in reclaimed.drain(..) {
            self.release(ring)?;
        }
        self.recycled = reclaimed;
        let ring = match completed.result {
            Ok(ring) => ring,
            Err(error) => {
                self.failed = true;
                return Err(error).context("native expert prefetch read failed");
            }
        };
        self.queue
            .as_mut()
            .context("prefetch completion has no queue")?
            .complete(completed.layer, completed.expert, ring)?;
        self.stats[0] += 1;
        Ok(true)
    }

    fn reclaim(&mut self, slots: &mut Slots, wait: bool) -> Result<bool> {
        let Some(reader) = slots.disk.as_mut() else {
            return Ok(false);
        };
        let mut reclaimed = false;
        for ring in 0..reader.copied.len() {
            if let Some(copy) = &reader.copied[ring]
                && copy.event.try_is_complete()?
            {
                let copy = reader.copied[ring]
                    .take()
                    .context("prefetch completed copy disappeared")?;
                reader.disk.reclaim(ring as u32, copy.id)?;
                self.release(ring as u32)?;
                reclaimed = true;
            }
        }
        if !reclaimed && wait {
            let ring = reader
                .copied
                .iter()
                .enumerate()
                .filter_map(|(ring, copy)| copy.as_ref().map(|copy| (copy.id, ring)))
                .min()
                .map(|(_, ring)| ring);
            if let Some(ring) = ring {
                let copy = reader.copied[ring]
                    .take()
                    .context("prefetch busy ring has no event")?;
                copy.event.synchronize()?;
                reader.disk.reclaim(ring as u32, copy.id)?;
                self.release(ring as u32)?;
                reclaimed = true;
            }
        }
        Ok(reclaimed)
    }

    pub fn drive(&mut self, slots: &mut Slots) -> Result<()> {
        ensure!(
            !self.failed,
            "native read-ahead is invalid after an earlier failure"
        );
        self.receive(slots, false)?;
        if self.pending {
            return Ok(());
        }
        self.reclaim(slots, false)?;
        let next = self.queue.as_mut().and_then(Prefetch::begin_read);
        if let Some((layer, expert)) = next {
            let reader = slots
                .disk
                .take()
                .context("prefetch disk reader has no owner")?;
            ensure!(
                reader.copied.len() == self.taken.len(),
                "prefetch reader rings={} differ from setup rings={}",
                reader.copied.len(),
                self.taken.len()
            );
            let request = Read {
                reader,
                catalogue: slots.catalogue.clone(),
                layer,
                expert,
                reclaimed: std::mem::take(&mut self.recycled),
            };
            ensure!(
                request.reclaimed.is_empty() && request.reclaimed.capacity() >= self.taken.len(),
                "prefetch completion scratch len={} capacity={} rings={}",
                request.reclaimed.len(),
                request.reclaimed.capacity(),
                self.taken.len()
            );
            ensure!(
                Arc::ptr_eq(&request.reader.context, slots.transfers.h2d.context()),
                "prefetch reader and H2D stream contexts differ: layer={} expert={}",
                request.layer,
                request.expert
            );
            request.catalogue.range(request.layer, request.expert)?;
            self.input
                .as_ref()
                .context("prefetch input is closed")?
                .send(request)
                .map_err(|_| anyhow::anyhow!("native read-ahead submission failed"))?;
            self.pending = true;
        }
        Ok(())
    }

    fn discard(&mut self, slots: &mut Slots, ring: u32) -> Result<()> {
        slots
            .disk
            .as_ref()
            .context("prefetch discard has no reader owner")?
            .disk
            .discard(ring)?;
        self.stats[2] += slots.catalogue.slot_bytes();
        Ok(())
    }

    pub fn begin_layer(&mut self, slots: &mut Slots, layer: u32, plan: &SpanPlan) -> Result<()> {
        self.receive(slots, true)?;
        self.retire_forecast(layer, plan)?;
        self.consumed
            .retain(|&(completed_layer, _)| completed_layer >= layer);
        let mut index = 0;
        while index < self.predictions.len() {
            let (predicted_layer, expert) = self.predictions[index];
            if predicted_layer <= layer
                && !(predicted_layer == layer && plan.staged().contains(&expert))
            {
                if let Retired::Loaded(ring) = self
                    .queue
                    .as_mut()
                    .context("prefetch cancellation has no queue")?
                    .cancel(predicted_layer, expert)?
                {
                    self.discard(slots, ring)?;
                }
                self.predictions.remove(index);
            } else {
                index += 1;
            }
        }
        self.prime(slots, layer, plan)
    }

    pub fn prime(&mut self, slots: &mut Slots, layer: u32, plan: &SpanPlan) -> Result<()> {
        self.receive(slots, false)?;
        if !self.pending {
            self.reclaim(slots, false)?;
        }
        for &expert in plan.staged() {
            if slots.policy.ledger().tier(layer, expert)? != Tier::Disk
                || self.predictions.contains(&(layer, expert))
                || self.consumed.contains(&(layer, expert))
            {
                continue;
            }
            let queue = self
                .queue
                .as_mut()
                .context("disk prefetch has no setup ring")?;
            if queue.entries() >= self.depth {
                break;
            }
            let (requested, retired) = queue.request(layer, expert, false, false)?;
            ensure!(
                matches!(requested, Requested::Queued | Requested::Present) && retired.is_none(),
                "prefetch non-demand request={requested:?} retired={retired:?}"
            );
            self.predictions.push((layer, expert));
            self.stats[3] += 1;
        }
        self.drive(slots)
    }

    pub fn predict(&mut self, slots: &mut Slots, layer: u32, plan: &SpanPlan) -> Result<()> {
        ensure!(
            self.forecast.is_empty(),
            "prefetch forecast for layer={layer} overlaps an earlier layer"
        );
        for &expert in plan.staged() {
            if slots.policy.ledger().tier(layer, expert)? == Tier::Disk {
                ensure!(
                    self.forecast.len() < self.forecast.capacity(),
                    "prefetch forecast exceeds experts={}",
                    self.forecast.capacity()
                );
                self.forecast.push((layer, expert));
            }
        }
        self.prime(slots, layer, plan)
    }

    fn retire_forecast(&mut self, layer: u32, plan: &SpanPlan) -> Result<()> {
        ensure!(
            self.forecast
                .iter()
                .all(|&(predicted, _)| predicted >= layer),
            "prefetch forecast precedes actual layer={layer}"
        );
        self.stats[4] += self
            .forecast
            .iter()
            .filter(|&&(predicted, expert)| predicted == layer && !plan.staged().contains(&expert))
            .count() as u64;
        self.forecast.retain(|&(predicted, _)| predicted > layer);
        Ok(())
    }

    pub fn demand(&mut self, slots: &mut Slots, layer: u32, expert: u32) -> Result<u32> {
        ensure!(
            self.queue.is_some(),
            "disk demand layer={layer} expert={expert} has no setup ring"
        );
        self.receive(slots, true)?;
        let predicted = self.predictions.contains(&(layer, expert));
        ensure!(
            !self.consumed.contains(&(layer, expert)),
            "prefill disk expert layer={layer} expert={expert} was streamed twice"
        );
        if !predicted
            && self
                .queue
                .as_ref()
                .context("prefetch demand has no queue")?
                .entries()
                >= self.depth
            && !self.predictions.is_empty()
        {
            let (old_layer, old_expert) = self.predictions.remove(0);
            if let Retired::Loaded(ring) = self
                .queue
                .as_mut()
                .context("prefetch cancellation has no queue")?
                .cancel(old_layer, old_expert)?
            {
                self.discard(slots, ring)?;
            }
        }
        let (_, retired) = self
            .queue
            .as_mut()
            .context("prefetch demand has no queue")?
            .request(layer, expert, true, false)?;
        ensure!(
            retired.is_none(),
            "prefetch demand retired an untracked ring={retired:?}"
        );
        self.predictions
            .retain(|&request| request != (layer, expert));
        loop {
            if let Some(ring) = self
                .queue
                .as_mut()
                .context("prefetch demand has no queue")?
                .take(layer, expert)
            {
                ensure!(
                    !self.taken[ring as usize],
                    "prefetch ring={ring} was already taken"
                );
                self.taken[ring as usize] = true;
                ensure!(
                    self.consumed.len() < self.consumed.capacity(),
                    "prefetch consumed experts exceed capacity={}",
                    self.consumed.capacity()
                );
                self.consumed.push((layer, expert));
                self.stats[1] += u64::from(predicted);
                return Ok(ring);
            }
            self.drive(slots)?;
            if !self.pending {
                ensure!(
                    self.reclaim(slots, true)?,
                    "prefetch demand layer={layer} expert={expert} cannot make ring progress"
                );
                self.drive(slots)?;
            }
            ensure!(
                self.pending,
                "prefetch demand layer={layer} expert={expert} has no queued read"
            );
            self.receive(slots, true)?;
        }
    }

    pub fn finish(&mut self, slots: &mut Slots) -> Result<()> {
        self.receive(slots, true)?;
        while let Some((layer, expert)) = self.predictions.pop() {
            if let Retired::Loaded(ring) = self
                .queue
                .as_mut()
                .context("prefetch cancellation has no queue")?
                .cancel(layer, expert)?
            {
                self.discard(slots, ring)?;
            }
        }
        while self.reclaim(slots, true)? {}
        self.consumed.clear();
        self.forecast.clear();
        ensure!(
            self.queue.as_ref().is_none_or(|queue| queue.entries() == 0)
                && self.taken.iter().all(|taken| !taken)
                && (self.depth == 0 || slots.disk.is_some()),
            "prefetch finish entries={} taken={:?} reader_owned={}",
            self.queue.as_ref().map_or(0, Prefetch::entries),
            self.taken,
            slots.disk.is_some()
        );
        self.stats[5] += 1;
        Ok(())
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        self.input.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_cancellations_ignore_read_completion() -> Result<()> {
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 2,
            experts: 16,
            hidden: 128,
            intermediate: 64,
            top_k: 4,
        };
        let service = crate::experts::plan::ServiceTable::new(
            "fixture",
            [1, 64]
                .into_iter()
                .map(|rows| crate::experts::plan::ServiceCell {
                    workers: 2,
                    rows,
                    experts: 1,
                    seconds: f64::from(rows) * 0.00002,
                })
                .collect(),
        )?;
        let mut plan = SpanPlan::new(
            &geometry,
            crate::experts::catalogue::ExpertLayout::new(
                crate::model_spec::ExpertFormat::Bf16,
                &geometry,
            )?
            .slot_bytes(),
            2,
            2,
        )?;
        let mut counts = [0; 16];
        for expert in [0, 2, 5] {
            counts[expert] = 65;
        }
        counts[1] = 1;
        counts[7] = 1;
        let mut resident = [false; 16];
        resident[1] = true;
        plan.update(
            &counts,
            &resident,
            263,
            4,
            &SpanRates {
                service: &service,
                h2d_bytes_per_second: 1_000_000_000.0,
                cpu_row_limit: LANE16_ROWS,
            },
        )?;
        assert_eq!(plan.staged(), [0, 2, 5]);
        assert_eq!(plan.cpu(), [7]);
        for completed in [false, true] {
            let mut ahead = ReadAhead::new(2, 0, 16)?;
            assert!(ahead.worker.is_none() && ahead.queue.is_none() && ahead.depth == 0);
            ahead.forecast.extend([(1, 0), (1, 2), (1, 5), (1, 7)]);
            let mut queue = Prefetch::new(2, 2)?;
            queue.request(1, 7, false, false)?;
            if completed {
                assert_eq!(queue.begin_read(), Some((1, 7)));
                queue.complete(1, 7, 0)?;
                ahead.stats[2] = 49152;
            }
            ahead.queue = Some(queue);
            ahead.retire_forecast(1, &plan)?;
            assert_eq!(ahead.stats[4], 1);
            assert!(ahead.forecast.is_empty());
            ahead.retire_forecast(1, &plan)?;
            assert_eq!(ahead.stats[4], 1);
        }
        println!(
            "PASS prefill_forecast_cancellations queued=1 completed=1 duplicate_decisions=0 zero_ring_worker=false GPU_executed=false"
        );
        Ok(())
    }
}
