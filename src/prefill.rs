#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::config::Config;
use anyhow::{Context, Result, ensure};
use serde::Serialize;

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn prompt_direct_max_rows(
    format: &crate::model_spec::LinearFormat,
    #[cfg(feature = "cuda")] fp8_direct_rows: std::ops::RangeInclusive<usize>,
    #[cfg(feature = "cuda")] nvfp4_direct_max_rows: impl FnOnce() -> Result<usize>,
) -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        if *format == crate::model_spec::LinearFormat::NvFp4 {
            nvfp4_direct_max_rows()
        } else {
            Ok(*fp8_direct_rows.end())
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = format;
        Ok(crate::mtp::MAX_VERIFY_ROWS)
    }
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn dense_prompt_bytes(
    spec: &crate::model_spec::ModelSpec,
    capacity: usize,
    chunk: usize,
    kv: crate::kv::KvFormat,
    direct_max_rows: usize,
) -> Result<usize> {
    ensure!(
        capacity > 0 && chunk > 0 && chunk <= capacity,
        "invalid dense prefill extent"
    );
    if spec.moe.is_some()
        || !matches!(
            spec.linear_format,
            crate::model_spec::LinearFormat::Fp8E4m3Block { .. }
                | crate::model_spec::LinearFormat::NvFp4
        )
        || kv != crate::kv::KvFormat::Bf16
        || chunk <= direct_max_rows
        || capacity == chunk
    {
        return Ok(0);
    }
    capacity
        .checked_mul(spec.hidden as usize)
        .and_then(|n| n.checked_mul(2 * std::mem::size_of::<u16>()))
        .context("dense prefill prompt buffer extent overflow")
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn chunk_end(start: usize, expected: usize, rows: usize, total: usize) -> Result<usize> {
    ensure!(
        start == expected && rows > 0,
        "prefill chunks are out of order"
    );
    let end = start
        .checked_add(rows)
        .context("prefill chunk extent overflow")?;
    ensure!(end <= total, "prefill chunk exceeds the request");
    Ok(end)
}
/// Schedule visits of one component times the workload's request count. A one-shot request count is 1.
#[cfg(any(test, feature = "cuda", feature = "rocm"))]
pub(crate) fn placement_visits(per_request: usize, requests: usize) -> Result<usize> {
    ensure!(
        per_request > 0 && requests > 0,
        "dense placement visit count"
    );
    per_request
        .checked_mul(requests)
        .context("dense placement visit count overflow")
}

#[derive(Clone, Debug, Serialize)]
pub struct DensePlacement {
    pub slots: Vec<Option<usize>>,
    pub host: Vec<bool>,
    pub slot_bytes: usize,
    pub resident_bytes: usize,
    pub host_bytes: usize,
    pub disk_bytes: usize,
}

impl DensePlacement {
    pub fn device_budget(
        free: usize,
        total: usize,
        fixed: usize,
        transient: usize,
        media: usize,
    ) -> Result<usize> {
        free.checked_sub(total / 16)
            .and_then(|n| n.checked_sub(fixed))
            .and_then(|n| n.checked_sub(transient))
            .and_then(|n| n.checked_sub(media))
            .context("dense fixed weights, media and workspace exceed VRAM")
    }

    /// `host_layers` sizes the host/disk tier per layer; `None` makes it the device size, the native BF16 form.
    /// A quantized manifest passes the stored source bytes here and the BF16 tile size in `layers`.
    pub fn new(
        layers: &[usize],
        host_layers: Option<&[usize]>,
        device_bytes: usize,
        host_bytes: usize,
    ) -> Result<Self> {
        Self::with_allocations(layers, layers, host_layers, device_bytes, host_bytes, 0, 2)
    }

    /// Host bytes spent for `visits`. One visit spends none.
    #[cfg(any(test, feature = "cuda", feature = "rocm"))]
    pub(crate) fn host_budget_for_visits(host_bytes: usize, visits: usize) -> Result<usize> {
        ensure!(visits > 0, "dense placement visit count");
        Ok(if visits == 1 { 0 } else { host_bytes })
    }

    pub(crate) fn with_allocations(
        layers: &[usize],
        slot_layers: &[usize],
        host_layers: Option<&[usize]>,
        device_bytes: usize,
        host_bytes: usize,
        streamed_overhead: usize,
        slot_count: usize,
    ) -> Result<Self> {
        ensure!(slot_count == 2, "dense streaming slot count {slot_count}");
        ensure!(
            !layers.is_empty() && layers.iter().all(|&n| n > 0 && n >= streamed_overhead),
            "empty dense layer extent"
        );
        ensure!(
            slot_layers.len() == layers.len() && slot_layers.iter().all(|&n| n > 0),
            "dense slot layer extent differs from resident layers"
        );
        let host_layers = host_layers.unwrap_or(layers);
        ensure!(
            host_layers.len() == layers.len() && host_layers.iter().all(|&n| n > 0),
            "dense host layer extent differs from the device layers"
        );
        let total = layers.iter().try_fold(0usize, |sum, &n| {
            sum.checked_add(n).context("dense weight extent overflow")
        })?;
        let mut plan = Self {
            slots: vec![None; layers.len()],
            host: vec![false; layers.len()],
            slot_bytes: 0,
            resident_bytes: total,
            host_bytes: 0,
            disk_bytes: 0,
        };
        if total <= device_bytes {
            return Ok(plan);
        }
        let (slot, resident, kept) = select_slot(
            layers,
            slot_layers,
            device_bytes,
            streamed_overhead,
            slot_count,
        )?
        .with_context(|| format!("dense fixed tensors and {slot_count} layer slots exceed VRAM"))?;
        plan.slot_bytes = slot;
        plan.resident_bytes = kept;
        let mut order: Vec<_> = (0..layers.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(layers[i]));
        let mut index = 0;
        for (i, &resident) in resident.iter().enumerate() {
            if !resident {
                plan.slots[i] = Some(index % slot_count);
                index += 1;
            }
        }
        let mut remaining = host_bytes;
        let mut streamed_device = 0usize;
        let mut streamed_source = 0usize;
        for &i in &order {
            if plan.slots[i].is_some() {
                streamed_device += layers[i];
                streamed_source += host_layers[i];
                if host_layers[i] <= remaining {
                    plan.host[i] = true;
                    remaining -= host_layers[i];
                    plan.host_bytes += host_layers[i];
                } else {
                    plan.disk_bytes += host_layers[i];
                }
            }
        }
        ensure!(
            plan.resident_bytes + streamed_device == total,
            "dense device byte conservation"
        );
        ensure!(
            plan.host_bytes + plan.disk_bytes == streamed_source,
            "dense tier byte conservation"
        );
        Ok(plan)
    }
}

fn select_slot(
    layers: &[usize],
    slot_layers: &[usize],
    device_bytes: usize,
    streamed_overhead: usize,
    slot_count: usize,
) -> Result<Option<(usize, Vec<bool>, usize)>> {
    let mut sizes = slot_layers.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    let mut order: Vec<_> = (0..layers.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(layers[i]));
    let mut best = None;
    for slot in sizes {
        let slots = slot
            .checked_mul(slot_count)
            .context("dense slots overflow")?;
        let setup = layers
            .len()
            .checked_mul(streamed_overhead)
            .and_then(|n| n.checked_add(slots))
            .context("dense streaming setup overflow")?;
        let Some(mut remaining) = device_bytes.checked_sub(setup) else {
            continue;
        };
        let mut resident = vec![false; layers.len()];
        let mut required = 0usize;
        for &i in &order {
            if slot_layers[i] > slot {
                required += layers[i] - streamed_overhead;
                resident[i] = true;
            }
        }
        let Some(available) = remaining.checked_sub(required) else {
            continue;
        };
        remaining = available;
        for &i in &order {
            if !resident[i] && layers[i] - streamed_overhead <= remaining {
                resident[i] = true;
                remaining -= layers[i] - streamed_overhead;
            }
        }
        let kept = layers
            .iter()
            .zip(&resident)
            .filter_map(|(&bytes, &kept)| kept.then_some(bytes))
            .sum::<usize>();
        if best
            .as_ref()
            .is_none_or(|(_, _, previous)| kept > *previous)
        {
            best = Some((slot, resident, kept));
        }
    }
    Ok(best)
}

/// Payload bytes per second of a sysfs `max_link_speed` (`"16.0 GT/s"`) at `width` lanes.
/// At 8.0 GT/s and above the encoding is 128/130; below it is 8/10.
#[cfg(any(test, feature = "cuda"))]
pub(crate) fn pcie_payload_bytes_per_second(max_link_speed: &str, width: usize) -> Result<f64> {
    let head = max_link_speed
        .split_whitespace()
        .next()
        .context("empty PCIe link speed")?;
    let gt_per_second: f64 = head.parse().context("PCIe link speed")?;
    ensure!(
        gt_per_second.is_finite() && gt_per_second > 0.0 && width > 0,
        "invalid PCIe link"
    );
    let encoding = if gt_per_second >= 8.0 {
        128.0 / 130.0
    } else {
        8.0 / 10.0
    };
    let payload = gt_per_second * 1.0e9 * (width as f64) / 8.0 * encoding;
    ensure!(payload.is_finite() && payload > 0.0, "PCIe payload rate");
    Ok(payload)
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub chunk: usize,
    pub requested_chunk: Option<usize>,
    pub free_bytes: usize,
    pub reserve_bytes: usize,
    pub row_bytes: usize,
    pub row_limit: usize,
    pub alignment: usize,
}

impl Plan {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub(crate) fn classic_row_bytes(c: &Config, draft_row_bytes: usize) -> usize {
        (4 * c.hidden_size + c.qkv_dim() + 3 * c.intermediate_size) * 2
            + 4
            + c.num_attention_heads * 4
            + draft_row_bytes
    }

    pub fn new(request: WorkspaceRequest) -> Result<Self> {
        Self::budget(
            request.capacity,
            request.requested,
            request.free_after_resident,
            request.total,
            request.alignment,
            request.row_bytes,
            request.fixed_transient_bytes,
        )
    }

    fn budget(
        capacity: usize,
        requested: Option<usize>,
        free: usize,
        total: usize,
        alignment: usize,
        row_bytes: usize,
        fixed_transient_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            alignment > 0 && row_bytes > 0,
            "invalid prefill alignment or row size"
        );
        ensure!(free <= total, "free VRAM exceeds total VRAM");
        let reserve_bytes = total / 16;
        let available = free
            .checked_sub(reserve_bytes)
            .and_then(|n| n.checked_sub(fixed_transient_bytes))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "prefill fixed reservations exceed VRAM: fixed={fixed_transient_bytes} reserve={reserve_bytes} available={free}"
                )
            })?;
        let row_limit = capacity.min(available / row_bytes);
        ensure!(
            row_limit > 0,
            "prefill workspace cannot fit one row: required={row_bytes} available={available} fixed_bytes={fixed_transient_bytes}"
        );
        let chunk = if let Some(chunk) = requested {
            ensure!(
                chunk > 0 && chunk <= row_limit,
                "requested prefill chunk {chunk} exceeds row budget {row_limit}: required={} available={available}",
                chunk
                    .checked_mul(row_bytes)
                    .context("prefill byte count overflow")?
            );
            chunk
        } else if row_limit < alignment {
            row_limit
        } else {
            let tiles = row_limit / alignment;
            alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
        };
        Ok(Self {
            chunk,
            requested_chunk: requested,
            free_bytes: free,
            reserve_bytes,
            row_bytes,
            row_limit,
            alignment,
        })
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
#[derive(Clone, Copy)]
pub(crate) struct DenseStage<'a> {
    pub device: &'a [usize],
    pub source: &'a [usize],
    pub visits: usize,
    pub load_scratch: usize,
    pub device_staging: usize,
}

/// Host bytes a disk supply holds while `disk_layers` of a plan stay on disk: one slot of
/// `slot_bytes` and one `scratch` buffer per worker, `workers` at most and never more than
/// the disk layers, which is the slot count the supply spawns.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn supply_reserve(
    (slot_bytes, scratch, disk_layers, workers): (usize, usize, usize, usize),
) -> Result<usize> {
    slot_bytes
        .checked_add(scratch)
        .and_then(|per_worker| workers.min(disk_layers).checked_mul(per_worker))
        .context("dense supply reserve overflow")
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl DenseStage<'_> {
    pub(crate) fn minimum_device_bytes(self, page: usize) -> Result<usize> {
        self.choices(usize::MAX, page)?
            .into_iter()
            .map(|(bytes, _)| bytes)
            .min()
            .context("resident stage has no legal device tier")
    }

    fn choices(self, limit: usize, page: usize) -> Result<Vec<(usize, DensePlacement)>> {
        ensure!(
            !self.device.is_empty()
                && self.source.len() == self.device.len()
                && self.visits > 0
                && self.device.iter().chain(self.source).all(|&n| n > 0),
            "invalid resident stage extent"
        );
        let mut order: Vec<_> = (0..self.device.len()).collect();
        order.sort_by_key(|&index| std::cmp::Reverse(self.device[index]));
        let mut out = Vec::new();
        let mut resident_bytes = 0usize;
        let mut resident_device = 0usize;
        let charge = |bytes: usize| {
            bytes
                .div_ceil(page)
                .checked_mul(page)
                .context("resident stage device allocation overflow")
        };
        for first in 0..=order.len() {
            let streamed = &order[first..];
            let slot_bytes = streamed.first().map_or(0, |&index| self.device[index]);
            let slots = charge(slot_bytes)?
                .checked_add(if streamed.is_empty() { 0 } else { page })
                .and_then(|n| n.checked_mul(2))
                .context("resident upload and mapping slots overflow")?;
            let staging = if streamed.is_empty() {
                0
            } else {
                charge(self.device_staging)?
            };
            let device = slots
                .checked_add(staging)
                .and_then(|bytes| bytes.checked_add(resident_device))
                .context("resident stage device extent overflow")?;
            if device <= limit {
                let mut slots = vec![None; order.len()];
                let mut slot = 0;
                let mut disk_bytes = 0usize;
                for (index, bytes) in self.source.iter().enumerate() {
                    if streamed.contains(&index) {
                        slots[index] = Some(slot % 2);
                        slot += 1;
                        disk_bytes = disk_bytes
                            .checked_add(*bytes)
                            .context("resident disk extent overflow")?;
                    }
                }
                out.push((
                    device,
                    DensePlacement {
                        host: vec![false; slots.len()],
                        slots,
                        slot_bytes,
                        resident_bytes,
                        host_bytes: 0,
                        disk_bytes,
                    },
                ));
            }
            if let Some(&index) = order.get(first) {
                resident_bytes = resident_bytes
                    .checked_add(self.device[index])
                    .context("resident stage weight extent overflow")?;
                resident_device = resident_device
                    .checked_add(charge(self.device[index])?)
                    .context("resident stage device allocation overflow")?;
            }
        }
        Ok(out)
    }

    fn price(self, plan: &DensePlacement, (pcie, disk): (f64, f64)) -> f64 {
        let mut read_at = 0.0f64;
        let mut upload_at = 0.0f64;
        let mut reusable = [0.0f64; 2];
        let mut next = 0;
        for (layer, slot) in plan.slots.iter().enumerate() {
            if slot.is_none() {
                continue;
            }
            let bytes = self.source[layer] as f64;
            if plan.host[layer] {
                upload_at += bytes / pcie;
            } else {
                read_at = read_at.max(reusable[next]) + bytes / disk;
                upload_at = upload_at.max(read_at) + bytes / pcie;
                reusable[next] = upload_at;
                next ^= 1;
            }
        }
        upload_at * self.visits as f64
    }

    fn pin(self, plan: &mut DensePlacement, layer: usize) {
        let bytes = self.source[layer];
        plan.host[layer] = true;
        plan.host_bytes += bytes;
        plan.disk_bytes -= bytes;
    }

    fn host_options(
        self,
        plan: &DensePlacement,
        rates: (f64, f64),
        ring_workers: usize,
    ) -> Result<HostOptions> {
        let disk_layers = plan.slots.iter().flatten().count();
        let ring_slot = plan.slot_bytes.max(self.device_staging);
        let reserve = |pinned: usize| {
            supply_reserve((
                ring_slot,
                self.load_scratch,
                disk_layers - pinned,
                ring_workers,
            ))
        };
        let full = reserve(0)?;
        let mut plan = plan.clone();
        let mut choices = vec![HostChoice {
            held: full,
            reserve: full,
            layers: 0,
            seconds: self.price(&plan, rates),
        }];
        let mut layers = 0;
        for layer in 0..plan.slots.len() {
            if plan.slots[layer].is_some() {
                self.pin(&mut plan, layer);
                layers += 1;
                let reserve = reserve(layers)?;
                choices.push(HostChoice {
                    held: plan
                        .host_bytes
                        .checked_add(reserve)
                        .context("resident host tier extent overflow")?,
                    reserve,
                    layers,
                    seconds: self.price(&plan, rates),
                });
            }
        }
        // A choice dominated on both seconds and held bytes by another leaves the set.
        let mut order: Vec<usize> = (0..choices.len()).collect();
        order.sort_by(|&a, &b| {
            choices[a]
                .seconds
                .total_cmp(&choices[b].seconds)
                .then_with(|| choices[a].held.cmp(&choices[b].held))
        });
        let mut keep = vec![true; choices.len()];
        let (mut below, mut run) = (usize::MAX, usize::MAX);
        let mut previous: Option<usize> = None;
        for &index in &order {
            if previous.is_some_and(|last| choices[last].seconds != choices[index].seconds) {
                below = below.min(run);
                run = usize::MAX;
            }
            keep[index] = choices[index].held < below && choices[index].held < run;
            run = run.min(choices[index].held);
            previous = Some(index);
        }
        let choices: Vec<_> = choices
            .into_iter()
            .zip(keep)
            .filter_map(|(choice, keep)| keep.then_some(choice))
            .collect();
        Ok(HostOptions {
            tail: choices.partition_point(|choice| choice.reserve == full),
            least: choices.iter().map(|choice| choice.held).min().unwrap_or(0),
            choices,
        })
    }

    fn pin_layers(self, plan: &DensePlacement, count: usize) -> DensePlacement {
        let mut plan = plan.clone();
        let mut remaining = count;
        for layer in 0..plan.slots.len() {
            if remaining > 0 && plan.slots[layer].is_some() {
                self.pin(&mut plan, layer);
                remaining -= 1;
            }
        }
        plan
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
struct HostChoice {
    /// Host bytes the choice holds: its pinned tiers and its supply.
    held: usize,
    reserve: usize,
    layers: usize,
    seconds: f64,
}

/// The host choices of one stage beside one base placement, in recorded order: seconds fall
/// with every later choice, and no choice holds both no more seconds and no more host bytes
/// than another. The leading run holds every ring worker's reserve; the tail holds fewer once
/// pinning leaves the stage fewer disk layers than ring workers.
#[cfg(any(feature = "cuda", feature = "rocm"))]
struct HostOptions {
    choices: Vec<HostChoice>,
    tail: usize,
    least: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl HostOptions {
    /// The choices holding at most `room` host bytes, in recorded order.
    fn fitting(&self, room: usize) -> impl Iterator<Item = &HostChoice> + '_ {
        let (lead, tail) = self.choices.split_at(self.tail);
        lead.iter()
            .take_while(move |choice| choice.held <= room)
            .chain(tail.iter().filter(move |choice| choice.held <= room))
    }

    /// The last choice holding at most `room` host bytes with at most `mappings` host mappings.
    fn last_fitting(&self, room: usize, mappings: usize) -> Option<&HostChoice> {
        let fits = |choice: &HostChoice| choice.held <= room && choice.layers <= mappings;
        let lead = self.choices[..self.tail].partition_point(|choice| fits(choice));
        let mut last = lead.checked_sub(1);
        for index in self.tail..self.choices.len() {
            if fits(&self.choices[index]) {
                last = Some(index);
            }
        }
        last.map(|index| &self.choices[index])
    }
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct PricedDense {
    pub plans: [DensePlacement; 3],
    pub device_bytes: [usize; 3],
    pub decoder_resident_seconds: Option<f64>,
    pub decoder_streamed_seconds: Option<f64>,
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn price_dense_stages(
    stages: [DenseStage<'_>; 3],
    (device_limit, host_limit, device_page): (usize, usize, usize),
    rates: (f64, f64),
    ring_workers: usize,
) -> Result<PricedDense> {
    try_price_dense_stages(
        stages,
        (device_limit, host_limit, device_page),
        rates,
        0.0,
        ring_workers,
    )?
    .with_context(|| {
        format!(
            "resident slots and weights cannot fit device={device_limit} host={host_limit} bytes"
        )
    })
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn try_price_dense_stages(
    stages: [DenseStage<'_>; 3],
    (device_limit, host_limit, device_page): (usize, usize, usize),
    rates: (f64, f64),
    decoder_stream_layer_seconds: f64,
    ring_workers: usize,
) -> Result<Option<PricedDense>> {
    ensure!(
        [rates.0, rates.1]
            .into_iter()
            .all(|n| n.is_finite() && n > 0.0),
        "invalid resident placement transfer rates"
    );
    ensure!(ring_workers > 0, "dense ring worker count is zero");
    ensure!(device_page > 0, "invalid resident device page size");
    let choices = [
        stages[0].choices(device_limit, device_page)?,
        stages[1].choices(device_limit, device_page)?,
        stages[2].choices(device_limit, device_page)?,
    ];
    let decoder_dispatch: Vec<_> = choices[2]
        .iter()
        .map(|(_, plan)| {
            decoder_stream_layer_seconds
                * (stages[2].visits * plan.slots.iter().flatten().count()) as f64
        })
        .collect();
    let options = [
        choices[0]
            .iter()
            .map(|(_, plan)| stages[0].host_options(plan, rates, ring_workers))
            .collect::<Result<Vec<_>>>()?,
        choices[1]
            .iter()
            .map(|(_, plan)| stages[1].host_options(plan, rates, ring_workers))
            .collect::<Result<Vec<_>>>()?,
        choices[2]
            .iter()
            .map(|(_, plan)| stages[2].host_options(plan, rates, ring_workers))
            .collect::<Result<Vec<_>>>()?,
    ];
    let mut best = None;
    let mut decoder_resident_seconds: Option<f64> = None;
    let mut decoder_streamed_seconds: Option<f64> = None;
    for (i, (d0, c0)) in choices[0].iter().enumerate() {
        for (j, (d1, c1)) in choices[1].iter().enumerate() {
            for (k, (d2, c2)) in choices[2].iter().enumerate() {
                let Some(mapping_capacity) = d0
                    .checked_add(*d1)
                    .and_then(|n| n.checked_add(*d2))
                    .and_then(|n| device_limit.checked_sub(n))
                    .map(|room| room / device_page)
                else {
                    continue;
                };
                let (firsts, seconds, decoder) = (&options[0][i], &options[1][j], &options[2][k]);
                let Some(room) = host_limit
                    .checked_sub(seconds.least)
                    .and_then(|n| n.checked_sub(decoder.least))
                else {
                    continue;
                };
                for first in firsts.fitting(room) {
                    let Some(mappings) = mapping_capacity.checked_sub(first.layers) else {
                        break;
                    };
                    let room = host_limit - first.held - decoder.least;
                    for second in seconds.fitting(room) {
                        let Some(mappings) = mappings.checked_sub(second.layers) else {
                            break;
                        };
                        let room = host_limit - first.held - second.held;
                        let Some(last) = decoder.last_fitting(room, mappings) else {
                            continue;
                        };
                        let seconds =
                            first.seconds + second.seconds + last.seconds + decoder_dispatch[k];
                        let alternative = if c2.slot_bytes == 0 {
                            &mut decoder_resident_seconds
                        } else {
                            &mut decoder_streamed_seconds
                        };
                        if alternative.is_none_or(|other| seconds < other) {
                            *alternative = Some(seconds);
                        }
                        if best
                            .as_ref()
                            .is_none_or(|(previous, _, _)| seconds < *previous)
                        {
                            best = Some((
                                seconds,
                                [
                                    stages[0].pin_layers(c0, first.layers),
                                    stages[1].pin_layers(c1, second.layers),
                                    stages[2].pin_layers(c2, last.layers),
                                ],
                                [
                                    d0 + first.layers * device_page,
                                    d1 + second.layers * device_page,
                                    d2 + last.layers * device_page,
                                ],
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(best.map(|(_, plans, device_bytes)| PricedDense {
        plans,
        device_bytes,
        decoder_resident_seconds,
        decoder_streamed_seconds,
    }))
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    #[test]
    fn resident_stage_rates_select_different_decoder_tiers() -> anyhow::Result<()> {
        use super::{DenseStage, price_dense_stages};
        let conditioner = [1200; 4];
        let dit = [800; 5];
        let decoder = [500; 4];
        let stages = [
            DenseStage {
                device: &conditioner,
                source: &conditioner,
                visits: 1,
                load_scratch: 1000,
                device_staging: 0,
            },
            DenseStage {
                device: &dit,
                source: &dit,
                visits: 6,
                load_scratch: 600,
                device_staging: 0,
            },
            DenseStage {
                device: &decoder,
                source: &decoder,
                visits: 1,
                load_scratch: 400,
                device_staging: 0,
            },
        ];
        let slow_pcie = price_dense_stages(stages, (6032, 9000, 1), (1.0, 100.0), 2)?;
        assert!(slow_pcie.plans[2].slot_bytes > 0);
        assert!(slow_pcie.decoder_streamed_seconds < slow_pcie.decoder_resident_seconds);
        let slow_disk = price_dense_stages(stages, (6032, 9000, 1), (100.0, 1.0), 2)?;
        assert_eq!(slow_disk.plans[2].slot_bytes, 0);
        assert!(slow_disk.decoder_resident_seconds < slow_disk.decoder_streamed_seconds);
        assert!(price_dense_stages(stages, (1, 9000, 1), (1.0, 1.0), 2).is_err());
        let layer_bytes = [8; 4];
        let streamed = DenseStage {
            device: &layer_bytes,
            source: &layer_bytes,
            visits: 1,
            load_scratch: 0,
            device_staging: 0,
        };
        let full = price_dense_stages([streamed; 3], (72, 150, 4), (100.0, 1.0), 2)?;
        assert_eq!(full.device_bytes, [24; 3]);
        assert!(full.plans.iter().all(|plan| plan.host_bytes == 0));
        let mapped = price_dense_stages([streamed; 3], (76, 150, 4), (100.0, 1.0), 2)?;
        assert_eq!(mapped.device_bytes.iter().sum::<usize>(), 76);
        assert_eq!(
            mapped
                .plans
                .iter()
                .map(|plan| plan.host_bytes)
                .sum::<usize>(),
            8
        );
        let small = DenseStage {
            device: &[8],
            source: &[8],
            ..streamed
        };
        let casting = DenseStage {
            device: &[80; 6],
            source: &[160; 6],
            device_staging: 160,
            ..streamed
        };
        let stages = [small, small, casting];
        assert!(price_dense_stages(stages, (344, 319, 4), (100.0, 1.0), 2).is_err());
        let converted = price_dense_stages(stages, (344, 320, 4), (100.0, 1.0), 2)?;
        assert_eq!(converted.device_bytes, [8, 8, 328]);
        assert_eq!(converted.plans[2].slot_bytes, 80);
        assert_eq!(converted.plans[2].disk_bytes, 960);
        // A pin that leaves the modelled time unchanged and holds less host memory stays a
        // choice: the all-host plan fits at 3000 host bytes and nowhere below.
        let three = DenseStage {
            device: &[1000; 3],
            source: &[1000; 3],
            visits: 1,
            load_scratch: 100,
            device_staging: 0,
        };
        let stages = [three, small, small];
        let equal = price_dense_stages(stages, (2021, 3000, 1), (1.0, 100.0), 4)?;
        assert_eq!(equal.plans[0].host_bytes, 3000);
        assert_eq!(equal.plans[0].disk_bytes, 0);
        assert!(
            price_dense_stages(stages, (2021, 2999, 1), (1.0, 100.0), 4).is_err(),
            "a host limit below the all-host plan placed a stage"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ring_worker_reserve_prices_the_spawned_slots_at_production_shapes() -> anyhow::Result<()> {
        use super::{DensePlacement, DenseStage, PricedDense, price_dense_stages};
        use anyhow::ensure;
        // Cold text-to-image shapes: conditioner and DiT dense stages of 50 layers averaging
        // 505,054,494 B, decoder 393 tensors averaging 4,110,337 B with 1,615,562,080 B F32
        // staging, largest dense tensor 262,144,000 B, device budget 23,183,563,776 B; pinned
        // H2D 19.9 GB/s, disk 2.0 GB/s.
        let conditioner = vec![505_054_494; 50];
        let dit = vec![505_054_494; 50];
        let decoder = vec![4_110_337; 393];
        let stages = |dit_visits: usize| {
            [
                DenseStage {
                    device: &conditioner,
                    source: &conditioner,
                    visits: 1,
                    load_scratch: 262_144_000,
                    device_staging: 0,
                },
                DenseStage {
                    device: &dit,
                    source: &dit,
                    visits: dit_visits,
                    load_scratch: 262_144_000,
                    device_staging: 0,
                },
                DenseStage {
                    device: &decoder,
                    source: &decoder,
                    visits: 1,
                    load_scratch: 16_777_216,
                    device_staging: 1_615_562_080,
                },
            ]
        };
        let (device_limit, page) = (23_183_563_776usize, 2_097_152usize);
        let rates = (19.9e9, 2.0e9);
        let disk_layers = |plan: &DensePlacement| {
            plan.slots
                .iter()
                .zip(&plan.host)
                .filter(|(slot, host)| slot.is_some() && !**host)
                .count()
        };
        let host_layers = |plan: &DensePlacement| {
            plan.slots
                .iter()
                .zip(&plan.host)
                .filter(|(slot, host)| slot.is_some() && **host)
                .count()
        };
        // Host bytes the supplies of a plan hold: min(workers, the stage's disk layers) slots,
        // each with one largest-tensor bounce buffer.
        let charge = |priced: &PricedDense, workers: usize| {
            priced
                .plans
                .iter()
                .zip(stages(40))
                .map(|(plan, stage)| {
                    if plan.slot_bytes == 0 {
                        0
                    } else {
                        workers.min(disk_layers(plan))
                            * (plan.slot_bytes.max(stage.device_staging) + stage.load_scratch)
                    }
                })
                .sum::<usize>()
        };
        let host_bytes = |priced: &PricedDense| {
            priced
                .plans
                .iter()
                .map(|plan| plan.host_bytes)
                .sum::<usize>()
        };
        let shape = |priced: &PricedDense| {
            priced
                .plans
                .iter()
                .map(|plan| (host_layers(plan), disk_layers(plan)))
                .collect::<Vec<_>>()
        };
        let place = |host: usize, workers: usize| {
            price_dense_stages(stages(40), (device_limit, host, page), rates, workers)
        };
        // A pin that leaves the modelled time unchanged and holds less host memory stays a
        // choice. At 31,313,378,628 B of host budget every streamed layer is on the host tier,
        // the conditioner's 50 and the DiT's 12 beside 38 resident DiT layers, and no supply
        // spawns; one byte less leaves disk layers in the conditioner.
        let whole = place(31_313_378_628, 8)?;
        ensure!(
            shape(&whole) == [(50, 0), (12, 0), (0, 0)]
                && host_bytes(&whole) == 31_313_378_628
                && charge(&whole, 8) == 0
                && whole.plans[2].slot_bytes == 0,
            "the all-host plan at its own size: {:?} host={} charge={}",
            shape(&whole),
            host_bytes(&whole),
            charge(&whole, 8)
        );
        let below = place(31_313_378_627, 2)?;
        ensure!(
            disk_layers(&below.plans[0]) > 0 && host_bytes(&below) < 31_313_378_628,
            "one byte below the all-host plan keeps every streamed layer on the host tier: {:?}",
            shape(&below)
        );
        // 28 GiB of host budget leave the conditioner 6, 9 and 15 disk layers at 2, 4 and 8
        // workers; the DiT keeps 12 host layers and the decoder stays resident.
        for (workers, disk, host, reserve) in [
            (2usize, 6usize, 28_283_051_664usize, 1_534_396_988usize),
            (4, 9, 26_767_888_182, 3_068_793_976),
            (8, 15, 23_737_561_218, 6_137_587_952),
        ] {
            let priced = place(28usize << 30, workers)?;
            ensure!(
                shape(&priced) == [(50 - disk, disk), (12, 0), (0, 0)]
                    && charge(&priced, workers) == reserve
                    && host_bytes(&priced) == host,
                "workers {workers}: {:?} charge={} host={}",
                shape(&priced),
                charge(&priced, workers),
                host_bytes(&priced)
            );
            // The charge is what the plan's supply holds: a host budget of that charge and
            // the plan's own host tiers places the same tiers.
            let exact = place(reserve + host, workers)?;
            ensure!(
                shape(&exact) == shape(&priced) && host_bytes(&exact) == host,
                "workers {workers}: the plan's charge does not place its host tiers: {:?} host={}",
                shape(&exact),
                host_bytes(&exact)
            );
            // One scratch per streamed stage is below that charge and places fewer host tiers.
            let one_scratch = workers.min(disk) * 505_054_494 + 262_144_000;
            ensure!(
                one_scratch < reserve,
                "workers {workers}: one scratch per stage is not below the charge: {one_scratch} vs {reserve}"
            );
            let single = place(one_scratch + host, workers)?;
            ensure!(
                host_bytes(&single) < host,
                "workers {workers}: one scratch per stage places the full host tier: {} vs {host}",
                host_bytes(&single)
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn stage_with_fewer_disk_layers_than_workers_pays_the_spawned_slots() -> anyhow::Result<()> {
        use super::{DenseStage, price_dense_stages};
        use anyhow::ensure;
        // Layers of 1000 device bytes stream through two slots at four ring workers; a worker
        // holds one slot and one 100-byte bounce buffer, 1100 bytes. The device limit leaves
        // one mapping page per host tier.
        let price = |layers: usize, source: usize, host: usize| {
            let device = vec![1000; layers];
            let host_layers = vec![source; layers];
            let tiny = [1usize; 1];
            let stages = [
                DenseStage {
                    device: &device,
                    source: &host_layers,
                    visits: 1,
                    load_scratch: 100,
                    device_staging: 0,
                },
                DenseStage {
                    device: &tiny,
                    source: &tiny,
                    visits: 1,
                    load_scratch: 1,
                    device_staging: 0,
                },
                DenseStage {
                    device: &tiny,
                    source: &tiny,
                    visits: 1,
                    load_scratch: 1,
                    device_staging: 0,
                },
            ];
            price_dense_stages(stages, (2004 + layers, host, 1), (100.0, 1.0), 4)
                .map(|priced| (priced.plans[0].host_bytes, priced.plans[0].disk_bytes))
        };
        // Three disk layers at four workers: three slots, then two beside one host tier.
        ensure!(
            price(3, 2000, 3300)? == (0, 6000),
            "three disk layers do not pay exactly three slots: {:?}",
            price(3, 2000, 3300)
        );
        ensure!(
            price(3, 2000, 3299).is_err(),
            "three slots fit one byte short"
        );
        ensure!(
            price(3, 2000, 4200)? == (2000, 4000),
            "one host tier does not leave exactly two slots: {:?}",
            price(3, 2000, 4200)
        );
        ensure!(
            price(3, 2000, 4199)? == (0, 6000),
            "two slots beside a host tier fit one byte short: {:?}",
            price(3, 2000, 4199)
        );
        // A tier smaller than a worker's slot: pinning the last layers frees more than it holds.
        ensure!(
            price(3, 1000, 3050)? == (3000, 0),
            "the last host tiers are not reached behind a full reserve: {:?}",
            price(3, 1000, 3050)
        );
        ensure!(
            price(3, 1000, 2999).is_err(),
            "the last host tier fits one byte short"
        );
        // Five disk layers: every worker pays its slot.
        ensure!(
            price(5, 2000, 4400)? == (0, 10000),
            "four workers beside five disk layers do not pay four slots: {:?}",
            price(5, 2000, 4400)
        );
        ensure!(
            price(5, 2000, 4399).is_err(),
            "four slots fit one byte short"
        );
        Ok(())
    }

    use super::{
        Plan, WorkspaceBudget, WorkspaceRequest, chunk_end, dense_prompt_bytes,
        pcie_payload_bytes_per_second, prompt_direct_max_rows,
    };

    #[test]
    fn two_streaming_slots_reject_a_third() -> anyhow::Result<()> {
        use super::DensePlacement;
        let rate = pcie_payload_bytes_per_second("16.0 GT/s", 16)?;
        assert_eq!(rate, 16.0e9 * 16.0 / 8.0 * (128.0 / 130.0));
        assert!((rate - 31_507_692_307.692).abs() < 1e-3);
        assert_eq!(
            pcie_payload_bytes_per_second("8.0 GT/s", 16)?,
            8.0e9 * 16.0 / 8.0 * (128.0 / 130.0)
        );
        assert_eq!(
            pcie_payload_bytes_per_second("5.0 GT/s", 16)?,
            8_000_000_000.0
        );
        assert!(pcie_payload_bytes_per_second("fast", 16).is_err());
        let two = DensePlacement::with_allocations(&[100; 10], &[100; 10], None, 800, 0, 0, 2)?;
        assert_eq!(two.resident_bytes, 600);
        assert_eq!(two.resident_bytes + 2 * two.slot_bytes, 800);
        assert!(
            DensePlacement::with_allocations(&[100; 10], &[100; 10], None, 800, 0, 0, 3).is_err()
        );
        Ok(())
    }

    #[test]
    fn dense_media_reservation_preserves_device_capacity() -> anyhow::Result<()> {
        use super::DensePlacement;
        let total = 16usize << 30;
        let fixed = 2usize << 30;
        let transient = 1usize << 30;
        let media_weights = 512usize << 20;
        let media_workspace = 256usize << 20;
        let request_buffers = 128usize << 20;
        let media = media_weights + media_workspace + request_buffers;
        let layers = [256usize << 20; 48];
        let budget = DensePlacement::device_budget(total, total, fixed, transient, media)?;
        let plan = DensePlacement::new(&layers, None, budget, 4 << 30)?;
        assert!(plan.slot_bytes > 0);
        assert!(
            plan.resident_bytes + 2 * plan.slot_bytes + fixed + transient + media
                <= total - total / 16
        );
        let old_budget = DensePlacement::device_budget(total, total, fixed, transient, 0)?;
        let old = DensePlacement::new(&layers, None, old_budget, 4 << 30)?;
        assert_eq!(old.slot_bytes, 0);
        assert!(old.resident_bytes + fixed + transient + media > total - total / 16);
        assert!(DensePlacement::device_budget(total, total, fixed, transient, total).is_err());
        eprintln!(
            "PASS dense_media_capacity total={total} media_weights={media_weights} media_workspace={media_workspace} request_buffers={request_buffers} resident={} slots={} old_omitted_media_control=detected",
            plan.resident_bytes,
            2 * plan.slot_bytes
        );
        Ok(())
    }

    #[test]
    fn dense_tiers_preserve_capacity_and_slot_bounds() -> anyhow::Result<()> {
        use super::DensePlacement;
        for (layers, device, host) in [
            (vec![448_266_240; 48], 19 << 30, 4 << 30),
            (
                (0..60)
                    .map(|i| {
                        if i % 6 == 5 {
                            1_067_974_656
                        } else {
                            957_874_176
                        }
                    })
                    .collect(),
                18 << 30,
                32 << 30,
            ),
            (vec![13, 29, 17, 29, 13], 78, 17),
        ] {
            let plan = DensePlacement::new(&layers, None, device, host)?;
            assert!(plan.resident_bytes + 2 * plan.slot_bytes <= device);
            assert!(plan.host_bytes <= host);
            assert_eq!(
                plan.resident_bytes + plan.host_bytes + plan.disk_bytes,
                layers.iter().sum::<usize>()
            );
            let mut sequence = 0;
            for (i, slot) in plan.slots.iter().enumerate() {
                if let Some(slot) = slot {
                    assert_eq!(*slot, sequence % 2);
                    assert!(layers[i] <= plan.slot_bytes);
                    sequence += 1;
                } else {
                    assert!(!plan.host[i]);
                }
            }
            assert!(sequence > 0);
        }
        let resident = DensePlacement::new(&[13, 29, 17], None, 59, 0)?;
        assert_eq!(resident.slot_bytes, 0);
        assert!(DensePlacement::new(&[13, 29, 17], None, 25, 100).is_err());
        assert!(DensePlacement::new(&[usize::MAX, 1], None, usize::MAX, 0).is_err());
        assert!(DensePlacement::new(&[0], None, 1, 0).is_err());
        Ok(())
    }

    #[test]
    fn dense_host_tiers_preserve_streamed_source_and_device_bytes() -> anyhow::Result<()> {
        use super::DensePlacement;
        let layers = [10usize, 8, 6, 4];
        let host_layers = [10usize, 9, 7, 5];
        let device = 24usize;
        let host = 12usize;
        let plan = DensePlacement::new(&layers, Some(&host_layers), device, host)?;
        let total = layers.iter().sum::<usize>();
        let mut streamed_device = 0usize;
        let mut streamed_source = 0usize;
        for (i, slot) in plan.slots.iter().enumerate() {
            if slot.is_some() {
                streamed_device += layers[i];
                streamed_source += host_layers[i];
            }
        }
        assert!(plan.resident_bytes + 2 * plan.slot_bytes <= device);
        assert!(plan.resident_bytes > 0);
        assert_eq!(plan.resident_bytes + streamed_device, total);
        assert_eq!(plan.host_bytes + plan.disk_bytes, streamed_source);
        assert!(plan.host_bytes <= host);
        assert!(plan.disk_bytes > 0);
        for (i, &on_host) in plan.host.iter().enumerate() {
            if plan.slots[i].is_some() && !on_host {
                assert!(host_layers[i] > host - plan.host_bytes);
            }
        }
        assert!(plan.host[0]);
        assert!(!plan.host[1]);
        assert!(!plan.host[2]);
        assert_eq!(plan.host_bytes, 10);
        assert_eq!(plan.disk_bytes, 16);
        assert!(DensePlacement::new(&layers, Some(&host_layers[..3]), device, host).is_err());
        assert!(DensePlacement::new(&layers, Some(&[10, 9, 0, 5]), device, host).is_err());
        Ok(())
    }

    #[test]
    fn dense_visit_count_keeps_a_reused_layer_on_the_host_tier() -> anyhow::Result<()> {
        use super::{DensePlacement, placement_visits};
        let layers = vec![100usize; 50];
        let device = 200usize;
        let host = 25 * 100;
        let today = DensePlacement::new(&layers, None, device, host)?;
        assert_eq!(today.host.iter().filter(|on| **on).count(), 25);
        let streamed = today
            .slots
            .iter()
            .zip(&layers)
            .filter_map(|(slot, bytes)| slot.is_some().then_some(*bytes))
            .sum::<usize>();
        let once_budget = DensePlacement::host_budget_for_visits(host, placement_visits(1, 1)?)?;
        let once = DensePlacement::new(&layers, None, device, once_budget)?;
        assert_eq!(once_budget, 0);
        assert_eq!(once.host_bytes, 0);
        assert!(once.host.iter().all(|on| !on));
        assert_eq!(once.disk_bytes, streamed);
        assert_eq!(once.slots, today.slots);
        let multi_budget = DensePlacement::host_budget_for_visits(host, placement_visits(6, 1)?)?;
        let multi = DensePlacement::new(&layers, None, device, multi_budget)?;
        assert_eq!(multi.host, today.host);
        assert_eq!(multi.host_bytes, today.host_bytes);
        assert_eq!(multi.disk_bytes, today.disk_bytes);
        let resident_budget =
            DensePlacement::host_budget_for_visits(host, placement_visits(1, 4)?)?;
        let resident = DensePlacement::new(&layers, None, device, resident_budget)?;
        assert_eq!(resident.host, today.host);
        assert!(resident.host_bytes > 0);
        eprintln!(
            "PASS dense_visit_count once={} multi={} resident={}",
            once.host_bytes, multi.host_bytes, resident.host_bytes
        );
        Ok(())
    }

    #[test]
    fn dense_prefill_chunk_order_rejects_a_swapped_predecessor() -> anyhow::Result<()> {
        let mut expected = 0;
        for (start, rows) in [(0, 8192), (8192, 8128)] {
            expected = chunk_end(start, expected, rows, 16320)?;
        }
        assert_eq!(expected, 16320);
        assert!(chunk_end(8192, 0, 8128, 16320).is_err());
        assert!(chunk_end(0, 0, 16321, 16320).is_err());
        assert!(chunk_end(usize::MAX, usize::MAX, 1, usize::MAX).is_err());
        println!(
            "PASS dense_prefill_chunk_order_control rows=16320 swapped_start=8192 expected_start=0 rejected=true"
        );
        Ok(())
    }

    #[test]
    fn dense_prompt_buffers_preserve_exact_capacity_bounds() -> anyhow::Result<()> {
        use crate::{
            gdn_plan::GdnPolicy,
            kv::KvFormat,
            model_spec::{LinearFormat, ModelSpec},
            op_contract::GdnSpec,
        };
        let root = serde_json::from_str(include_str!("../tests/data/q38-config.json"))?;
        let mut model = ModelSpec::qwen(&root, &serde_json::json!({}))?;
        model.linear_format = LinearFormat::fp8_block([128, 128], safetensors::Dtype::BF16)?;
        assert_eq!(model.hidden, 5120);
        let gdn = GdnSpec::from_model(&model)?;
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model)?.row_bytes()?;
        let direct_fp8 = prompt_direct_max_rows(
            &model.linear_format,
            #[cfg(feature = "cuda")]
            crate::backend::cuda::blas::Blas::fp8_direct_rows(0),
            #[cfg(feature = "cuda")]
            || anyhow::bail!("FP8 format must not query NVFP4 direct bound"),
        )?;
        for chunk in 1..=crate::mtp::MAX_VERIFY_ROWS {
            assert_eq!(
                dense_prompt_bytes(&model, 1024, chunk, KvFormat::Bf16, direct_fp8)?,
                0
            );
        }
        for chunk in [8, 6, crate::linear_plan::MAX_SKINNY_ROWS] {
            assert_eq!(
                dense_prompt_bytes(&model, 1024, chunk, KvFormat::Bf16, direct_fp8)?,
                20_971_520
            );
        }
        for (capacity, expected) in [(1024, 20_971_520), (4096, 83_886_080), (16384, 335_544_320)] {
            let chunk = capacity / 2;
            let bytes = dense_prompt_bytes(&model, capacity, chunk, KvFormat::Bf16, direct_fp8)?;
            assert_eq!(bytes, expected);
            assert_eq!(
                dense_prompt_bytes(&model, capacity, capacity, KvFormat::Bf16, direct_fp8)?,
                0
            );
            assert_eq!(
                dense_prompt_bytes(&model, capacity, chunk, KvFormat::Affine8, direct_fp8)?,
                0
            );
            let workspace = |rows| {
                Ok(usize::try_from(
                    GdnPolicy::P2.workspace(&gdn, rows as u32)?.bytes,
                )?)
            };
            let fixed = bytes + 4096;
            let total = 48usize << 30;
            let live = fixed + row_bytes * chunk + workspace(chunk)?;
            let request = WorkspaceRequest {
                capacity,
                requested: Some(chunk),
                free_after_resident: total / 16 + live,
                total,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: fixed,
            };
            let budget = WorkspaceBudget::new(request, workspace, Ok)?;
            assert_eq!(budget.plan.chunk, chunk);
            assert_eq!(budget.plan.row_limit, chunk);
            assert_eq!(budget.planned_transient_bytes, live);
            assert!(
                WorkspaceBudget::new(
                    WorkspaceRequest {
                        free_after_resident: request.free_after_resident - 1,
                        ..request
                    },
                    workspace,
                    Ok
                )
                .is_err()
            );
            println!(
                "PASS dense_prompt_capacity capacity={capacity} chunk={chunk} layout=[2,{capacity},5120] prompt_bytes={bytes} planned_transient_bytes={live} boundary_minus_one_rejected=true"
            );
        }
        assert!(dense_prompt_bytes(&model, usize::MAX, 32, KvFormat::Bf16, direct_fp8).is_err());
        assert!(dense_prompt_bytes(&model, 32, 33, KvFormat::Bf16, direct_fp8).is_err());
        model.linear_format = LinearFormat::Bf16;
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 8192, KvFormat::Bf16, direct_fp8)?,
            0
        );
        model.linear_format = LinearFormat::NvFp4;
        #[cfg(feature = "cuda")]
        {
            let fp8_invariant = prompt_direct_max_rows(
                &LinearFormat::fp8_block([128, 128], safetensors::Dtype::BF16)?,
                crate::backend::cuda::blas::Blas::fp8_direct_rows(
                    crate::linear_plan::MAX_SKINNY_ROWS,
                ),
                || anyhow::bail!("FP8 format must not query NVFP4 direct bound"),
            )?;
            assert_eq!(fp8_invariant, crate::linear_plan::MAX_SKINNY_ROWS);
            let nvfp4_sm89 = prompt_direct_max_rows(
                &model.linear_format,
                crate::backend::cuda::blas::Blas::fp8_direct_rows(0),
                || Ok(384),
            )?;
            assert_eq!(nvfp4_sm89, 384);
        }
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 384, KvFormat::Bf16, 384)?,
            0
        );
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 512, KvFormat::Bf16, 384)?,
            335_544_320
        );
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 8192, KvFormat::Bf16, 384)?,
            335_544_320
        );
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 128, KvFormat::Bf16, 147)?,
            0
        );
        assert_eq!(
            dense_prompt_bytes(&model, 16384, 256, KvFormat::Bf16, 147)?,
            335_544_320
        );
        let root = serde_json::from_str(include_str!("../tests/data/minicpm-config.json"))?;
        let mut moe =
            ModelSpec::minicpm(&root, &serde_json::json!({"eos_token_id":[248044,248046]}))?;
        moe.linear_format = LinearFormat::Fp8E4m3Channel;
        for capacity in [1024, 4096, 16384] {
            assert_eq!(
                dense_prompt_bytes(&moe, capacity, capacity / 2, KvFormat::Bf16, direct_fp8)?,
                0
            );
        }
        Ok(())
    }

    #[test]
    fn budget_bounds_and_override() {
        let plan = Plan::budget(33024, None, 160000, 160000, 64, 4, 0).unwrap();
        assert_eq!(plan.chunk, 32768);
        assert!(plan.chunk * plan.row_bytes + plan.reserve_bytes <= plan.free_bytes);
        assert!(Plan::budget(33024, Some(33025), 160000, 160000, 64, 4, 0).is_err());
        assert_eq!(
            Plan::budget(33024, Some(33024), 160000, 160000, 64, 4, 0)
                .unwrap()
                .chunk,
            33024
        );
        let exact = Plan::budget(16, Some(16), 72, 128, 1, 4, 0).unwrap();
        assert_eq!(exact.row_limit, 16);
        assert!(Plan::budget(16, Some(16), 71, 128, 1, 4, 0).is_err());
    }

    #[test]
    fn memory_limit_and_small_sequence() {
        assert_eq!(
            Plan::budget(32768, None, 20000, 160000, 64, 4, 0)
                .unwrap()
                .chunk,
            2048
        );
        assert_eq!(
            Plan::budget(17, None, 160000, 160000, 64, 4, 0)
                .unwrap()
                .chunk,
            17
        );
        assert!(Plan::budget(32768, None, 10000, 160000, 64, 4, 0).is_err());
    }
    #[test]
    fn classic_fixed_budget_shrinks_automatic_chunk() -> anyhow::Result<()> {
        let capacity = 131072usize;
        let total = 48usize << 30;
        let hidden = 7168usize;
        let qkv = hidden + 2 * 1024;
        let intermediate = 22016usize;
        let row_bytes = (4 * hidden + qkv + 3 * intermediate) * 2 + 4 + 56 * 4;
        let embedding = 166912 * hidden * 2;
        let fixed_weights = 2 * embedding + 2 * hidden + 52 * 2 * 2 * hidden;
        let kv = capacity * 52 * 1024 * 4;
        let fixed = fixed_weights + kv;
        let old = Plan::budget(capacity, None, total, total, 128, row_bytes, 0)?;
        assert_eq!(old.chunk, capacity);
        assert!(old.chunk * row_bytes + fixed > total - total / 16);
        let plan = Plan::new(WorkspaceRequest {
            capacity,
            requested: None,
            free_after_resident: total,
            total,
            alignment: 128,
            row_bytes,
            fixed_transient_bytes: fixed,
        })?;
        assert_eq!(plan.chunk, 65536);
        assert!(plan.chunk * row_bytes + fixed <= total - total / 16);
        assert!(
            Plan::budget(
                capacity,
                Some(capacity),
                total,
                total,
                128,
                row_bytes,
                fixed
            )
            .is_err()
        );
        let below_minimum = total - total / 16 - row_bytes + 1;
        let error =
            Plan::budget(capacity, None, total, total, 128, row_bytes, below_minimum).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prefill workspace cannot fit one row")
        );
        println!(
            "PASS classic_fixed_budget capacity={capacity} total_gib=48 old_chunk={} chunk={} row_bytes={row_bytes} fixed_weights={fixed_weights} kv_bytes={kv} minimum_error_named=true",
            old.chunk, plan.chunk
        );
        Ok(())
    }
}

/// Shared budget input for models with persistent mixer state and per-operator scratch.
/// `free_after_resident` is sampled after weights, KV/state and layer arena allocation;
/// they are recorded in the ledger but are not subtracted a second time.
#[derive(Debug, Serialize)]
pub struct WorkspaceBudget {
    pub plan: Plan,
    pub operator_workspace_bytes: usize,
    pub planned_transient_bytes: usize,
    pub accounting: &'static str,
}
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceRequest {
    pub capacity: usize,
    pub requested: Option<usize>,
    pub free_after_resident: usize,
    pub total: usize,
    pub alignment: usize,
    pub row_bytes: usize,
    pub fixed_transient_bytes: usize,
}
impl WorkspaceBudget {
    /// `workspace(rows)` must be monotone and include every simultaneously live
    /// operator temporary, with alignment. Sequential operators contribute their max.
    /// `select` bounds the DEFAULT chunk only; an explicit `requested` chunk is
    /// admitted as given, up to the same row limit.
    pub fn new(
        request: WorkspaceRequest,
        workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        Self::with_reserve(request, request.total / 16, 1, workspace, select)
    }

    fn with_reserve(
        request: WorkspaceRequest,
        reserve: usize,
        minimum_rows: usize,
        workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        let WorkspaceRequest {
            capacity,
            requested,
            free_after_resident,
            total,
            alignment,
            row_bytes,
            fixed_transient_bytes,
        } = request;
        ensure!(
            capacity > 0 && alignment > 0 && row_bytes > 0 && free_after_resident <= total,
            "invalid workspace budget"
        );
        ensure!(
            minimum_rows > 0 && minimum_rows <= capacity,
            "workspace row floor {minimum_rows} outside capacity {capacity}"
        );
        let available = free_after_resident.checked_sub(reserve).with_context(|| {
            format!("VRAM reserve {reserve} exceeds free memory {free_after_resident}")
        })?;
        let bytes = |rows: usize| -> Result<usize> {
            let operator_bytes = workspace(rows)?;
            rows.max(minimum_rows)
                .checked_mul(row_bytes)
                .and_then(|v| v.checked_add(fixed_transient_bytes))
                .and_then(|v| v.checked_add(operator_bytes))
                .context("workspace byte count overflow")
        };
        let one_row = bytes(1)?;
        ensure!(
            one_row <= available,
            "workspace cannot fit one row: required={one_row} available={available} row_bytes={row_bytes} fixed_bytes={fixed_transient_bytes}"
        );
        let (mut low, mut high) = (1usize, capacity);
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if bytes(mid)? <= available {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let row_limit = low;
        let chunk = if let Some(n) = requested {
            ensure!(
                n > 0 && n <= row_limit,
                "requested chunk {n} exceeds exact workspace budget: capacity={capacity} row_limit={row_limit} available_bytes={available} row_bytes={row_bytes} fixed_transient_bytes={fixed_transient_bytes}"
            );
            n
        } else {
            let admitted = if row_limit < alignment {
                row_limit
            } else {
                let tiles = row_limit / alignment;
                alignment * (1usize << (usize::BITS - 1 - tiles.leading_zeros()))
            };
            let selected = select(admitted)?;
            ensure!(
                selected > 0
                    && selected <= admitted
                    && (selected == admitted || selected.is_multiple_of(alignment)),
                "default chunk selection must preserve admission and alignment"
            );
            selected
        };
        let scratch = workspace(chunk)?;
        Ok(Self {
            plan: Plan {
                chunk,
                requested_chunk: requested,
                free_bytes: free_after_resident,
                reserve_bytes: reserve,
                row_bytes,
                row_limit,
                alignment,
            },
            operator_workspace_bytes: scratch,
            planned_transient_bytes: bytes(chunk)?,
            accounting: "free_after_resident",
        })
    }
}

/// Candidate row domains and measured resources before expert-slot allocation.
#[derive(Clone, Copy, Debug)]
pub struct ExpertWorkspaceRequest {
    pub mixer: WorkspaceRequest,
    pub span_rows: u32,
    pub wave_rows: u32,
    pub wave_workspace_bytes: usize,
    pub device_headroom_bytes: usize,
    /// CPU/disk scratch, pinned staging, runtime state and measured host headroom.
    pub host_fixed_bytes: u64,
    pub prefill_staging_slots: u32,
    pub disk_ring_slots: u32,
}

#[derive(Debug)]
pub struct ExpertWorkspaceBudget {
    pub mixer: WorkspaceBudget,
    pub workspace_rows: usize,
    /// Whole tail slots borrowed at the candidate's span bound.
    pub borrowed_slots: u32,
    pub candidate: crate::experts::plan::SpanCandidate,
}

#[derive(Debug)]
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) struct ExpertSpanLayout {
    pub rows: u32,
    pub residual: std::ops::Range<usize>,
    pub normalized: std::ops::Range<usize>,
    pub contributions: std::ops::Range<usize>,
    pub done: std::ops::Range<usize>,
    pub ids: std::ops::Range<usize>,
    pub weights: std::ops::Range<usize>,
    pub active: std::ops::Range<usize>,
    pub bytes: usize,
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
impl ExpertSpanLayout {
    pub fn validate_wave(&self, wave_rows: u32, workspace_rows: usize) -> Result<()> {
        ensure!(
            wave_rows > 0 && wave_rows <= self.rows && wave_rows as usize <= workspace_rows,
            "expert wave_rows={wave_rows} exceed span_rows={} or workspace_rows={workspace_rows}",
            self.rows
        );
        Ok(())
    }

    pub fn new(
        reserves: &crate::experts::plan::Reserves,
        geometry: crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
        rows: u32,
    ) -> Result<Self> {
        ensure!(rows > 0, "zero expert span rows={rows}");
        let hidden = u64::from(geometry.hidden) * 2;
        let choices = u64::from(geometry.top_k);
        ensure!(
            hidden > 0
                && choices > 0
                && reserves.residual_row_bytes == hidden
                && reserves.normalized_row_bytes == hidden
                && reserves.contribution_row_bytes == hidden * choices
                && reserves.done_row_bytes == choices * 4
                && reserves.route_row_bytes == choices * (4 + 2 + 4),
            "expert span reserves disagree with geometry={geometry:?} rows={rows}"
        );
        let mut bytes = 0usize;
        let mut region = |row_bytes: u64| -> Result<std::ops::Range<usize>> {
            let first = bytes
                .checked_add(15)
                .context("expert span alignment overflow")?
                & !15;
            let count = usize::try_from(
                row_bytes
                    .checked_mul(u64::from(rows))
                    .context("expert span region overflow")?,
            )?;
            bytes = first
                .checked_add(count)
                .context("expert span extent overflow")?;
            Ok(first..bytes)
        };
        let residual = region(reserves.residual_row_bytes)?;
        let normalized = region(reserves.normalized_row_bytes)?;
        let contributions = region(reserves.contribution_row_bytes)?;
        let done = region(reserves.done_row_bytes)?;
        let ids = region(choices * 4)?;
        let weights = region(choices * 2)?;
        drop(region);
        if weights.len() % 16 != 0 {
            bytes = bytes
                .checked_add(15)
                .context("expert span weight padding overflow")?
                & !15;
        }
        let mut region = |row_bytes: u64| -> Result<std::ops::Range<usize>> {
            let first = bytes
                .checked_add(15)
                .context("expert span alignment overflow")?
                & !15;
            let count = usize::try_from(
                row_bytes
                    .checked_mul(u64::from(rows))
                    .context("expert span region overflow")?,
            )?;
            bytes = first
                .checked_add(count)
                .context("expert span extent overflow")?;
            Ok(first..bytes)
        };
        let active = region(choices * 4)?;
        let raw = reserves.span_bytes(rows)?;
        let slot = slot_bytes;
        ensure!(
            slot.is_multiple_of(16) && bytes as u64 <= raw.div_ceil(slot) * slot,
            "aligned expert span bytes={bytes} exceed borrowed bytes={} raw={raw}",
            raw.div_ceil(slot) * slot
        );
        Ok(Self {
            rows,
            residual,
            normalized,
            contributions,
            done,
            ids,
            weights,
            active,
            bytes,
        })
    }
}

impl ExpertWorkspaceBudget {
    pub fn new(
        request: ExpertWorkspaceRequest,
        geometry: crate::experts::catalogue::ExpertGeometry,
        slot_bytes: u64,
        mixer_workspace: impl Fn(usize) -> Result<usize>,
        select: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Self> {
        use crate::experts::plan::{Reserves, SpanCandidate, span_borrow_slots};
        let ExpertWorkspaceRequest {
            mut mixer,
            span_rows,
            wave_rows,
            wave_workspace_bytes,
            device_headroom_bytes,
            host_fixed_bytes,
            prefill_staging_slots,
            disk_ring_slots,
        } = request;
        ensure!(
            mixer.free_after_resident <= mixer.total,
            "expert free-after-resident bytes {} exceed total bytes {}",
            mixer.free_after_resident,
            mixer.total
        );
        ensure!(
            geometry.layers > 0
                && geometry.experts > 0
                && geometry.hidden > 0
                && geometry.intermediate > 0
                && geometry.top_k > 0
                && geometry.top_k <= geometry.experts,
            "invalid expert workspace geometry: {geometry:?}"
        );
        ensure!(
            span_rows > 0 && span_rows as usize <= mixer.capacity,
            "expert span rows {span_rows} outside capacity {}",
            mixer.capacity
        );
        ensure!(
            wave_rows > 0 && wave_rows <= span_rows,
            "expert wave rows {wave_rows} outside span {span_rows}"
        );
        let hidden_bytes = u64::from(geometry.hidden) * 2;
        let choices = u64::from(geometry.top_k);
        let mut reserves = Reserves {
            device_fixed_bytes: 0,
            host_fixed_bytes,
            residual_row_bytes: hidden_bytes,
            normalized_row_bytes: hidden_bytes,
            contribution_row_bytes: hidden_bytes
                .checked_mul(choices)
                .context("expert contribution row overflow")?,
            done_row_bytes: choices * size_of::<u32>() as u64,
            // ids i32, weights BF16 and final-active u32.
            route_row_bytes: choices
                * (size_of::<i32>() + size_of::<half::bf16>() + size_of::<u32>()) as u64,
            prefill_staging_slots,
            disk_ring_slots,
        };
        let staging_bytes = reserves.persistent_bytes(slot_bytes)?;
        let span_bytes = reserves.span_bytes(span_rows)?;
        let borrowed_slots = span_borrow_slots(&reserves, slot_bytes, span_rows, span_rows)?;
        let routed_slots = u64::from(geometry.layers) * u64::from(geometry.experts);
        ensure!(
            u64::from(borrowed_slots) + u64::from(geometry.layers) <= routed_slots,
            "span bytes {span_bytes} borrow {borrowed_slots} slots, leaving fewer than one per layer: routed_slots={routed_slots} layers={}",
            geometry.layers
        );
        let borrowed_bytes = u64::from(borrowed_slots)
            .checked_mul(slot_bytes)
            .context("borrowed expert slot extent overflow")?;
        // Decode retains one contribution/done row and two route snapshots.
        let decode_mailbox_bytes = choices
            .checked_mul(
                2 * (size_of::<i32>()
                    + size_of::<half::bf16>()
                    + size_of::<u32>()
                    + size_of::<u64>()) as u64,
            )
            .and_then(|n| n.checked_add(reserves.contribution_row_bytes))
            .and_then(|n| n.checked_add(reserves.done_row_bytes))
            .context("decode mailbox extent overflow")?;
        let minimum_experts = u64::from(geometry.layers)
            .checked_mul(slot_bytes)
            .context("minimum expert capacity overflow")?;
        let withheld = usize::try_from(
            borrowed_bytes
                .checked_add(staging_bytes)
                .and_then(|n| n.checked_add(minimum_experts))
                .and_then(|n| n.checked_add(decode_mailbox_bytes))
                .context("expert workspace reservation overflow")?,
        )?;
        mixer.free_after_resident = mixer.free_after_resident.checked_sub(withheld).with_context(|| {
            format!(
                "borrowed span slots {borrowed_bytes} plus staging {staging_bytes}, minimum slots {minimum_experts} and decode mailboxes {decode_mailbox_bytes} exceed free bytes {}",
                mixer.free_after_resident
            )
        })?;
        mixer.capacity = span_rows as usize;
        let mixer = WorkspaceBudget::with_reserve(
            mixer,
            device_headroom_bytes,
            wave_rows as usize,
            |rows| Ok(mixer_workspace(rows)?.max(wave_workspace_bytes)),
            select,
        )
        .with_context(|| {
            format!(
                "expert mixer/wave admission failed: span_rows={span_rows} wave_rows={wave_rows} borrowed_slots={borrowed_slots} minimum_expert_bytes={minimum_experts} headroom_bytes={device_headroom_bytes}"
            )
        })?;
        let workspace_rows = mixer.plan.chunk.max(wave_rows as usize);
        reserves.device_fixed_bytes = u64::try_from(
            mixer
                .planned_transient_bytes
                .checked_add(device_headroom_bytes)
                .context("expert fixed reserve overflow")?,
        )?
        .checked_add(decode_mailbox_bytes)
        .context("expert fixed decode reserve overflow")?;
        Ok(Self {
            mixer,
            workspace_rows,
            borrowed_slots,
            candidate: SpanCandidate {
                span_rows,
                wave_rows,
                reserves,
            },
        })
    }

    pub fn select(
        inputs: &crate::experts::plan::PlanInputs<'_>,
        prefill_rows: u32,
        candidates: &[Self],
    ) -> Result<(
        usize,
        crate::experts::plan::Capacity,
        crate::experts::plan::Admission,
    )> {
        Self::select_with_peer(inputs, None, prefill_rows, candidates)
    }

    pub fn select_with_peer(
        inputs: &crate::experts::plan::PlanInputs<'_>,
        peer: Option<crate::experts::plan::PeerInputs<'_>>,
        prefill_rows: u32,
        candidates: &[Self],
    ) -> Result<(
        usize,
        crate::experts::plan::Capacity,
        crate::experts::plan::Admission,
    )> {
        use crate::experts::plan::{Admission, admission_budget, select_span_with_peer};
        let budget_bytes_per_token = admission_budget(
            &inputs.geometry,
            inputs.service,
            inputs.h2d_bytes_per_second,
            inputs.d2h_bytes_per_second,
            inputs.copy_engines,
        )?;
        let mut admission = Admission {
            budget_bytes_per_token,
        };
        let span_candidates = candidates
            .iter()
            .map(|budget| budget.candidate.clone())
            .collect::<Vec<_>>();
        let (index, mut capacity) = select_span_with_peer(
            inputs,
            admission.hard_cap(inputs.layout.slot_bytes())?,
            prefill_rows,
            &span_candidates,
            peer,
        )?;
        if capacity.gpu0_layers < inputs.geometry.layers {
            admission.budget_bytes_per_token = admission_budget(
                &crate::experts::catalogue::ExpertGeometry {
                    layers: capacity.gpu0_layers,
                    ..inputs.geometry
                },
                inputs.service,
                inputs.h2d_bytes_per_second,
                inputs.d2h_bytes_per_second,
                inputs.copy_engines,
            )?;
            let (_, updated) = select_span_with_peer(
                inputs,
                admission.hard_cap(inputs.layout.slot_bytes())?,
                prefill_rows,
                &span_candidates[index..=index],
                peer,
            )?;
            capacity = updated;
        }
        Ok((index, capacity, admission))
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    fn request(capacity: usize, requested: Option<usize>, free: usize) -> WorkspaceRequest {
        WorkspaceRequest {
            capacity,
            requested,
            free_after_resident: free,
            total: 160_000,
            alignment: 32,
            row_bytes: 4,
            fixed_transient_bytes: 512,
        }
    }

    /// The default chunk is the memory-admitted row limit rounded DOWN to the
    /// alignment and then bounded by `select`; an explicit request is honoured up to
    /// the row limit WITHOUT `select`.
    #[test]
    fn default_chunk_is_bounded_by_select_and_requests_are_not() -> Result<()> {
        let cap = |admitted: usize| Ok(admitted.min(2048));
        let wide = WorkspaceBudget::new(request(262_144, None, 160_000), |r| Ok(8 * r), cap)?;
        assert_eq!(wide.plan.chunk, 2048, "the default is clamped");
        assert!(wide.plan.chunk <= wide.plan.row_limit);
        let exact = WorkspaceBudget::new(request(262_144, None, 160_000), |r| Ok(8 * r), Ok)?;
        assert!(
            exact.plan.chunk > 2048,
            "select may pass the admitted chunk through"
        );
        let explicit =
            WorkspaceBudget::new(request(262_144, Some(8192), 160_000), |r| Ok(8 * r), cap)?;
        assert_eq!(
            explicit.plan.chunk, 8192,
            "an explicit chunk is never clamped"
        );
        let over =
            WorkspaceBudget::new(request(262_144, Some(1 << 20), 160_000), |r| Ok(8 * r), cap);
        assert!(over.is_err(), "an explicit chunk still faces the row limit");
        Ok(())
    }

    #[test]
    fn default_chunk_is_the_admitted_limit_rounded_down() -> Result<()> {
        for capacity in [
            1, 17, 31, 32, 511, 512, 513, 1021, 1024, 2047, 2048, 2049, 4096, 8192,
        ] {
            for free in [11_000, 14_000, 25_000, 160_000] {
                let budget =
                    WorkspaceBudget::new(request(capacity, None, free), |r| Ok(8 * r), Ok)?;
                let chunk = budget.plan.chunk;
                assert!(chunk > 0 && chunk <= budget.plan.row_limit);
                assert!(chunk == budget.plan.row_limit || chunk.is_multiple_of(32));
                assert_eq!(budget.plan.requested_chunk, None);
                assert_eq!(budget.operator_workspace_bytes, 8 * chunk);
                assert_eq!(budget.planned_transient_bytes, 4 * chunk + 512 + 8 * chunk);
            }
        }
        Ok(())
    }

    #[test]
    fn every_budget_field_is_carried_and_requests_are_bounded() -> Result<()> {
        let budget = WorkspaceBudget::new(request(8192, Some(2048), 160_000), |r| Ok(8 * r), Ok)?;
        assert_eq!(budget.plan.chunk, 2048);
        assert_eq!(budget.plan.requested_chunk, Some(2048));
        assert_eq!(budget.plan.free_bytes, 160_000);
        assert_eq!(budget.plan.reserve_bytes, 160_000 / 16);
        assert_eq!(budget.plan.row_bytes, 4);
        assert_eq!(budget.plan.alignment, 32);
        assert_eq!(budget.accounting, "free_after_resident");
        assert!(
            WorkspaceBudget::new(request(8192, Some(1 << 20), 160_000), |r| Ok(8 * r), Ok).is_err()
        );
        assert!(WorkspaceBudget::new(request(8192, Some(0), 160_000), |r| Ok(8 * r), Ok).is_err());
        assert!(WorkspaceBudget::new(request(0, None, 160_000), |r| Ok(8 * r), Ok).is_err());
        assert!(WorkspaceBudget::new(request(8192, None, 1), |r| Ok(8 * r), Ok).is_err());
        Ok(())
    }

    #[test]
    fn expert_span_destinations_and_wave_floor_precede_cache_capacity() -> Result<()> {
        use crate::experts::{
            catalogue::ExpertGeometry,
            plan::{PlanInputs, ServiceCell, ServiceTable},
        };
        let model = crate::model_spec::ModelSpec::minicpm(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/minicpm-config.json"
            )))?,
            &serde_json::json!({"eos_token_id": [248044, 248046]}),
        )?;
        let geometry = ExpertGeometry::from_model(&model)?;
        let layout = crate::experts::catalogue::ExpertLayout::new(
            model.expert_format.context("MoE expert_format missing")?,
            &geometry,
        )?;
        let slot_bytes = layout.slot_bytes();
        let row_bytes = crate::work_layout::RowWorkSpec::new(&model)?.row_bytes()?;
        let request = ExpertWorkspaceRequest {
            mixer: WorkspaceRequest {
                capacity: 65_536,
                requested: Some(37),
                free_after_resident: 24 << 30,
                total: 24 << 30,
                alignment: 32,
                row_bytes,
                fixed_transient_bytes: 8192,
            },
            span_rows: 65_536,
            wave_rows: 71,
            wave_workspace_bytes: 71 * 937,
            device_headroom_bytes: 4096,
            host_fixed_bytes: 16384,
            prefill_staging_slots: 2,
            disk_ring_slots: 3,
        };
        let candidates = [97, 65_536]
            .map(|span_rows| {
                ExpertWorkspaceBudget::new(
                    ExpertWorkspaceRequest {
                        span_rows,
                        ..request
                    },
                    geometry,
                    slot_bytes,
                    |r| Ok(r * 257),
                    Ok,
                )
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let budget = &candidates[1];
        let reserves = &budget.candidate.reserves;
        assert_eq!(budget.mixer.plan.chunk, 37);
        assert_eq!(budget.workspace_rows, 71);
        assert_eq!(budget.mixer.operator_workspace_bytes, 71 * 937);
        assert_eq!(reserves.residual_row_bytes, 4096);
        assert_eq!(reserves.normalized_row_bytes, 4096);
        assert_eq!(reserves.contribution_row_bytes, 32768);
        assert_eq!(reserves.done_row_bytes, 32);
        assert_eq!(reserves.route_row_bytes, 80);
        assert_eq!(budget.borrowed_slots, 428);
        let fixed = 8192 + 71 * row_bytes + 71 * 937 + 4096 + 33088;
        assert_eq!(reserves.device_fixed_bytes, fixed as u64);
        assert_eq!(reserves.span_bytes(request.span_rows)?, 65_536 * 41072);
        for rows in [1, 97, request.span_rows] {
            let layout = ExpertSpanLayout::new(reserves, geometry, slot_bytes, rows)?;
            layout.validate_wave(1, 1)?;
            assert!(layout.validate_wave(1, 0).is_err());
            if rows >= 12 {
                assert!(layout.validate_wave(12, 8).is_err());
            }
            assert_eq!(layout.bytes, rows as usize * 41072);
            let regions = [
                &layout.residual,
                &layout.normalized,
                &layout.contributions,
                &layout.done,
                &layout.ids,
                &layout.weights,
                &layout.active,
            ];
            assert!(regions.iter().all(|r| r.start % 16 == 0));
            assert!(regions.windows(2).all(|r| r[0].end <= r[1].start));
            assert_eq!(
                layout.contributions.len(),
                layout.rows as usize * 8 * 2048 * 2
            );
            assert_eq!(layout.active.end, layout.bytes);
        }
        assert_eq!(
            reserves.persistent_bytes(slot_bytes)?,
            fixed as u64 + 2 * 6_291_456
        );
        assert_eq!(reserves.host_bytes(slot_bytes)?, 16384 + 3 * 6_291_456);
        let service = ServiceTable::new(
            "fixture",
            vec![
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 1.0 / 65536.0,
                },
                ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 8,
                    seconds: 0.001,
                },
            ],
        )?;
        let inputs = PlanInputs {
            geometry,
            layout: layout.clone(),
            device_free_bytes: request.mixer.free_after_resident as u64,
            host_available_bytes: 48 << 30,
            h2d_bytes_per_second: (16u64 << 30) as f64,
            d2h_bytes_per_second: (16u64 << 30) as f64,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: 0,
            service: &service,
            reserves,
            span_rows: request.span_rows,
            wave_rows: request.wave_rows,
        };
        let (index, capacity, admission) =
            ExpertWorkspaceBudget::select(&inputs, 65_536, &candidates)?;
        assert_eq!(index, 1);
        assert_eq!(admission.budget_bytes_per_token, 10 << 20);
        assert_eq!(capacity.spare_blocks, 1);
        assert_eq!(capacity.span_rows, 65_536);
        assert_eq!(capacity.wave_rows, 71);
        let slots: u64 = capacity.slots_per_layer.iter().map(|&n| u64::from(n)).sum();
        let remaining = inputs.device_free_bytes - reserves.persistent_bytes(slot_bytes)?;
        assert_eq!(slots, remaining / slot_bytes);
        assert!(slots - u64::from(budget.borrowed_slots) >= u64::from(geometry.layers));
        let short = &candidates[0];
        assert_eq!(short.borrowed_slots, 1);
        assert_eq!(
            short.candidate.reserves.device_fixed_bytes,
            reserves.device_fixed_bytes
        );
        let short_capacity = crate::experts::plan::capacity(
            &PlanInputs {
                span_rows: 97,
                reserves: &short.candidate.reserves,
                ..inputs.clone()
            },
            1,
        )?;
        assert_eq!(short_capacity.slots_per_layer, capacity.slots_per_layer);
        for changed in [
            PlanInputs {
                copy_engines: 1,
                ..inputs.clone()
            },
            PlanInputs {
                d2h_bytes_per_second: (8u64 << 30) as f64,
                ..inputs.clone()
            },
        ] {
            let (_, capacity, admission) =
                ExpertWorkspaceBudget::select(&changed, 65_536, &candidates)?;
            assert_eq!(admission.budget_bytes_per_token, 5 << 20);
            assert!(capacity.host_blocks > 0);
            assert_eq!(capacity.spare_blocks, 1);
        }
        assert!(
            ExpertWorkspaceBudget::select(
                &PlanInputs {
                    d2h_bytes_per_second: f64::NAN,
                    ..inputs
                },
                65_536,
                &candidates,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn expert_workspace_exact_boundary_preserves_minimum_slots_and_headroom() -> Result<()> {
        use crate::experts::catalogue::ExpertGeometry;
        let geometry = ExpertGeometry {
            first_layer: 0,
            layers: 2,
            experts: 8,
            hidden: 16,
            intermediate: 8,
            top_k: 2,
        };
        let mut request = ExpertWorkspaceRequest {
            mixer: WorkspaceRequest {
                capacity: 16,
                requested: Some(3),
                free_after_resident: 1 << 20,
                total: 1 << 20,
                alignment: 1,
                row_bytes: 128,
                fixed_transient_bytes: 1024,
            },
            span_rows: 16,
            wave_rows: 5,
            wave_workspace_bytes: 2048,
            device_headroom_bytes: 4096,
            host_fixed_bytes: 512,
            prefill_staging_slots: 2,
            disk_ring_slots: 1,
        };
        let slot_bytes = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &geometry,
        )?
        .slot_bytes();
        let budget = ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok)?;
        let exact = budget.candidate.reserves.persistent_bytes(slot_bytes)?
            + (u64::from(budget.borrowed_slots) + u64::from(geometry.layers)) * slot_bytes;
        request.mixer.free_after_resident = usize::try_from(exact)?;
        let admitted =
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok)?;
        assert_eq!(admitted.workspace_rows, 5);
        assert_eq!(admitted.mixer.operator_workspace_bytes, 2048);
        assert_eq!(
            admitted.candidate.reserves.persistent_bytes(slot_bytes)?
                + (u64::from(admitted.borrowed_slots) + u64::from(geometry.layers)) * slot_bytes,
            exact
        );
        request.mixer.free_after_resident -= 1;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.free_after_resident = 1 << 20;
        request.wave_rows = 17;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.wave_rows = 5;
        request.mixer.requested = Some(17);
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.requested = Some(3);
        request.mixer.free_after_resident = request.mixer.total + 1;
        assert!(
            ExpertWorkspaceBudget::new(request, geometry, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );
        request.mixer.free_after_resident = request.mixer.total;
        let oversized = ExpertGeometry {
            hidden: u32::MAX,
            experts: u32::MAX,
            top_k: u32::MAX,
            ..geometry
        };
        assert!(
            ExpertWorkspaceBudget::new(request, oversized, slot_bytes, |r| Ok(r * 13), Ok).is_err()
        );

        let peer_geom = crate::experts::catalogue::ExpertGeometry {
            first_layer: 0,
            layers: 4,
            experts: 8,
            hidden: 128,
            intermediate: 64,
            top_k: 2,
        };
        let peer_layout = crate::experts::catalogue::ExpertLayout::new(
            crate::model_spec::ExpertFormat::Bf16,
            &peer_geom,
        )?;
        let peer_slot = peer_layout.slot_bytes();
        let mut peer_req = request;
        peer_req.mixer.requested = None;
        peer_req.mixer.free_after_resident = usize::try_from(peer_slot * 32 + (1 << 20))?;
        peer_req.mixer.total = peer_req.mixer.free_after_resident;
        let peer_cand =
            ExpertWorkspaceBudget::new(peer_req, peer_geom, peer_slot, |r| Ok(r * 13), Ok)?;
        let table = crate::experts::plan::ServiceTable::new(
            "test",
            vec![
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 1,
                    seconds: 0.001,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 1,
                    experts: 2,
                    seconds: 0.002,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 16,
                    experts: 1,
                    seconds: 0.016,
                },
                crate::experts::plan::ServiceCell {
                    workers: 1,
                    rows: 16,
                    experts: 2,
                    seconds: 0.032,
                },
            ],
        )?;
        let plan_inputs = crate::experts::plan::PlanInputs {
            geometry: peer_geom,
            layout: peer_layout,
            device_free_bytes: peer_cand.candidate.reserves.persistent_bytes(peer_slot)?
                + u64::from(peer_cand.borrowed_slots) * peer_slot
                + 12 * peer_slot,
            host_available_bytes: peer_cand.candidate.reserves.host_bytes(peer_slot)?
                + 64 * peer_slot,
            h2d_bytes_per_second: (peer_slot as f64) * 1000.0,
            d2h_bytes_per_second: (peer_slot as f64) * 1000.0,
            copy_engines: 2,
            disk_readers: 1,
            host_registration_bytes: 64 * peer_slot,
            service: &table,
            reserves: &peer_cand.candidate.reserves,
            span_rows: peer_cand.candidate.span_rows,
            wave_rows: peer_cand.candidate.wave_rows,
        };
        let (_, single_cap, single_adm) =
            ExpertWorkspaceBudget::select(&plan_inputs, 16, std::slice::from_ref(&peer_cand))?;
        let xfer = [(8_192u64, 1e10, 1e10), (65_536u64, 1e10, 1e10)];
        let (_, split_cap, split_adm) = ExpertWorkspaceBudget::select_with_peer(
            &plan_inputs,
            Some(crate::experts::plan::PeerInputs {
                device_free_bytes: 16 * peer_slot,
                device_reserved_bytes: 0,
                decode_transfer: xfer[0],
                prefill_transfers: &xfer,
            }),
            16,
            std::slice::from_ref(&peer_cand),
        )?;
        assert_eq!(single_cap.gpu0_layers, 4);
        assert_eq!(split_cap.gpu0_layers, 2);
        assert_eq!(
            split_adm.budget_bytes_per_token * 2,
            single_adm.budget_bytes_per_token
        );
        Ok(())
    }
}
