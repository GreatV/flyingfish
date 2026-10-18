//! Bounded native disk reads with one owner of the registered ring.
use super::*;
use crate::experts::span::{Prefetch, Requested, Retired};
#[cfg(test)]
use std::sync::mpsc::sync_channel;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::thread::JoinHandle;

struct Read {
    reader: Reader,
    catalogue: Arc<ExpertCatalogue>,
    layer: u32,
    expert: u32,
}

struct Completed {
    reader: Reader,
    layer: u32,
    expert: u32,
    result: Result<u32>,
    reclaimed: Vec<u32>,
}

pub(super) struct ReadAhead {
    queue: Prefetch,
    depth: u32,
    input: Option<SyncSender<Read>>,
    output: Receiver<Completed>,
    worker: Option<JoinHandle<()>>,
    pending: bool,
    failed: bool,
    predictions: Vec<(u32, u32)>,
    consumed: Vec<(u32, u32)>,
    taken: Vec<bool>,
    pub stats: [u64; 4],
}

impl ReadAhead {
    #[cfg(test)]
    pub fn new(staging: u32, rings: u32) -> Result<Self> {
        let depth = staging.min(rings);
        let queue = Prefetch::new(depth, rings)?;
        let (input, requests) = sync_channel::<Read>(1);
        let (results, output) = sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("expert-read-ahead".into())
            .spawn(move || {
                while let Ok(mut request) = requests.recv() {
                    let copied = request
                        .reader
                        .copied
                        .iter()
                        .map(Option::is_some)
                        .collect::<Vec<_>>();
                    let result = request
                        .reader
                        .context
                        .bind_to_thread()
                        .map_err(anyhow::Error::from)
                        .and_then(|()| {
                            request
                                .reader
                                .read(request.layer, request.expert, &request.catalogue)
                        });
                    let reclaimed = copied
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, &was_copied)| {
                            (was_copied && request.reader.copied[slot].is_none())
                                .then_some(slot as u32)
                        })
                        .collect();
                    if results
                        .send(Completed {
                            reader: request.reader,
                            layer: request.layer,
                            expert: request.expert,
                            result,
                            reclaimed,
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
            output,
            worker: Some(worker),
            pending: false,
            failed: false,
            predictions: Vec::with_capacity(depth as usize),
            consumed: Vec::new(),
            taken: vec![false; rings as usize],
            stats: [0; 4],
        })
    }

    fn release(&mut self, ring: u32) -> Result<()> {
        let taken = self
            .taken
            .get_mut(ring as usize)
            .context("prefetch reclaimed ring outside allocation")?;
        if *taken {
            ensure!(
                self.queue.release(ring),
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
                    .recv()
                    .context("native read-ahead worker stopped")?,
            )
        } else {
            match self.output.try_recv() {
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
        for ring in completed.reclaimed {
            self.release(ring)?;
        }
        let ring = match completed.result {
            Ok(ring) => ring,
            Err(error) => {
                self.failed = true;
                return Err(error).context("native expert prefetch read failed");
            }
        };
        self.queue
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
        if let Some((layer, expert)) = self.queue.begin_read() {
            let reader = slots
                .disk
                .take()
                .context("prefetch disk reader has no owner")?;
            let request = Read {
                reader,
                catalogue: slots.catalogue.clone(),
                layer,
                expert,
            };
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
        self.stats[2] += slots.catalogue.geometry().slot_bytes()?;
        Ok(())
    }

    pub fn begin_layer(&mut self, slots: &mut Slots, layer: u32, plan: &SpanPlan) -> Result<()> {
        self.receive(slots, true)?;
        self.consumed
            .retain(|&(completed_layer, _)| completed_layer >= layer);
        let mut index = 0;
        while index < self.predictions.len() {
            let (predicted_layer, expert) = self.predictions[index];
            if predicted_layer <= layer
                && !(predicted_layer == layer
                    && plan.staged.iter().any(|cohort| cohort.contains(&expert)))
            {
                if let Retired::Loaded(ring) = self.queue.cancel(predicted_layer, expert)? {
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
        for &expert in plan.staged.iter().flatten() {
            if self.queue.entries() >= self.depth {
                break;
            }
            if slots.policy.ledger().tier(layer, expert)? != Tier::Disk
                || self.predictions.contains(&(layer, expert))
                || self.consumed.contains(&(layer, expert))
            {
                continue;
            }
            let (requested, retired) = self.queue.request(layer, expert, false, false)?;
            ensure!(
                matches!(requested, Requested::Queued | Requested::Present) && retired.is_none(),
                "prefetch non-demand request={requested:?} retired={retired:?}"
            );
            self.predictions.push((layer, expert));
            self.stats[3] += 1;
        }
        self.drive(slots)
    }

    pub fn demand(&mut self, slots: &mut Slots, layer: u32, expert: u32) -> Result<u32> {
        self.receive(slots, true)?;
        let predicted = self.predictions.contains(&(layer, expert));
        ensure!(
            !self.consumed.contains(&(layer, expert)),
            "prefill disk expert layer={layer} expert={expert} was streamed twice"
        );
        if !predicted && self.queue.entries() >= self.depth && !self.predictions.is_empty() {
            let (old_layer, old_expert) = self.predictions.remove(0);
            if let Retired::Loaded(ring) = self.queue.cancel(old_layer, old_expert)? {
                self.discard(slots, ring)?;
            }
        }
        let (_, retired) = self.queue.request(layer, expert, true, false)?;
        ensure!(
            retired.is_none(),
            "prefetch demand retired an untracked ring={retired:?}"
        );
        self.predictions
            .retain(|&request| request != (layer, expert));
        loop {
            if let Some(ring) = self.queue.take(layer, expert) {
                ensure!(
                    !self.taken[ring as usize],
                    "prefetch ring={ring} was already taken"
                );
                self.taken[ring as usize] = true;
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
            if let Retired::Loaded(ring) = self.queue.cancel(layer, expert)? {
                self.discard(slots, ring)?;
            }
        }
        while self.reclaim(slots, true)? {}
        self.consumed.clear();
        ensure!(
            self.queue.entries() == 0
                && self.taken.iter().all(|taken| !taken)
                && slots.disk.is_some(),
            "prefetch finish entries={} taken={:?} reader_owned={}",
            self.queue.entries(),
            self.taken,
            slots.disk.is_some()
        );
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
