//! Decode graph ownership and capture transactions over stable expert tables.
use super::*;

pub(super) struct Segments {
    pub route: CudaGraph,
    pub contribute: CudaGraph,
    pub join: CudaGraph,
}

fn record(
    engine: &mut Engine,
    launch: impl FnOnce(&mut Engine) -> Result<()>,
) -> Result<CudaGraph> {
    engine.closure.capture()?;
    let stream = engine.device.stream.clone();
    #[cfg(test)]
    ensure!(
        engine.capture_fail_stage != Some(CaptureFault::Begin),
        "injected capture begin failure"
    );
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("cuStreamBeginCapture")?;
    let launched = launch(engine);
    let captured = stream
        .end_capture(sys::CUgraphInstantiate_flags(0))
        .context("cuStreamEndCapture");
    launched?;
    #[cfg(test)]
    ensure!(
        engine.capture_fail_stage != Some(CaptureFault::End),
        "injected capture end failure"
    );
    let graph = captured?.context("empty MoE decode segment")?;
    #[cfg(test)]
    ensure!(
        engine.capture_fail_stage != Some(CaptureFault::Instantiate),
        "injected capture instantiate failure"
    );
    graph.upload().context("cuGraphUpload")?;
    stream.synchronize().context("cuStreamSynchronize")?;
    Ok(graph)
}

impl Decode {
    fn resident_mask(
        &self,
        device: &Device,
        layer: usize,
        slots: SlotBindings,
        rows: usize,
    ) -> Result<()> {
        if self
            .peer
            .as_ref()
            .is_some_and(|peer| layer >= peer.gpu0_layers as usize)
        {
            return Ok(());
        }
        let stream = &device.stream;
        let k = self.geometry.top_k;
        let ids = view(&self.ids, stream, 0, [rows as u32, k], [u64::from(k), 1])?;
        let active = view(&self.active, stream, 0, [rows as u32, k], [u64::from(k), 1])?;
        let generation = view(
            &self.generation,
            stream,
            0,
            [rows as u32, k],
            [u64::from(k), 1],
        )?;
        // SAFETY: every table covers the expert domain and each assignment buffer covers rows*top_k.
        unsafe {
            stream
                .launch_builder(&self.mask)
                .arg(&ids.ptr)
                .arg(&slots.valid.ptr)
                .arg(&slots.generation.ptr)
                .arg(&active.ptr)
                .arg(&generation.ptr)
                .arg(&((rows * k as usize) as i32))
                .arg(&(self.geometry.experts as i32))
                .launch(grid((rows * k as usize).div_ceil(256), 256))?;
        }
        Ok(())
    }

    fn warmup_completion(&mut self, device: &Device, rows: usize) -> Result<()> {
        let k = rows * self.geometry.top_k as usize;
        device
            .stream
            .memcpy_dtod(&self.final_active.slice(..k), &mut self.done.slice_mut(..k))?;
        Ok(())
    }
}

impl Engine {
    fn moe_route_segment(
        &mut self,
        layer: usize,
        decode: &Decode,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        use crate::backend::Operator as O;
        if layer == 0 {
            self.run_operator(O::Embed, 0, step, &mut None)?;
        }
        self.qwen_layer_begin(layer, step)?;
        self.run_operator(O::Norm, layer, step, &mut None)?;
        self.run_mixer(self.model_spec.layers[layer].mixer, layer, step, &mut None)?;
        self.run_operator(O::PostNorm, layer, step, &mut None)?;
        if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Dense {
            return Ok(());
        }
        let slot_layer = self.moe_slot_layer(layer)?;
        decode.route_device(
            &self.device,
            slot_layer,
            &self.moe_ffn_binding(layer)?,
            qwen::matrix(
                &self.work.n,
                &self.device.stream,
                0,
                step.rows,
                self.model_spec.hidden as usize,
            )?,
        )
    }

    fn dense_ffn_segment(&mut self, layer: usize, step: crate::backend::Step<'_>) -> Result<()> {
        use crate::backend::Operator as O;
        for op in [O::GateUp, O::Activate, O::Down] {
            self.run_operator(op, layer, step, &mut None)?;
        }
        Ok(())
    }

    pub(super) fn moe_slot_layer(&self, layer: usize) -> Result<usize> {
        let first = self
            .model_spec
            .layers
            .iter()
            .position(|plan| plan.ffn == crate::model_spec::FfnKind::Moe)
            .context("model has no MoE layer")?;
        ensure!(
            layer >= first,
            "dense model layer={layer} has no expert slot"
        );
        Ok(layer - first)
    }

    fn moe_join_segment(
        &mut self,
        layer: usize,
        decode: &Decode,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        use crate::backend::Operator as O;
        let stream = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Dense {
            self.run_operator(O::NextNorm, layer, step, &mut None)?;
            if layer + 1 == self.model_spec.layers.len() {
                self.run_operator(O::Head, 0, step, &mut None)?;
                if step.head == Head::Last {
                    self.qwen_advance()?;
                }
            }
            #[cfg(test)]
            self.check_capture_layer(layer)?;
            return Ok(());
        }
        let slot_layer = self.moe_slot_layer(layer)?;
        decode.combine_device(
            &self.device,
            slot_layer,
            self.moe_ffn_binding(layer)?,
            qwen::matrix(&self.work.n, stream, 0, step.rows, hidden)?,
            &mut TensorMut(qwen::matrix(&self.work.out, stream, 0, step.rows, hidden)?),
        )?;
        self.run_operator(O::NextNorm, layer, step, &mut None)?;
        if layer + 1 == self.model_spec.layers.len() {
            self.run_operator(O::Head, 0, step, &mut None)?;
            if step.head == Head::Last {
                self.qwen_advance()?;
            }
        }
        #[cfg(test)]
        self.check_capture_layer(layer)?;
        Ok(())
    }

    fn moe_device_decode(
        &mut self,
        decode: &mut Decode,
        slots: &Slots,
        resident: bool,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        if !resident {
            stream.memset_zeros(&mut decode.active)?;
            stream.memset_zeros(&mut decode.contributions)?;
        }
        for layer in 0..self.model_spec.layers.len() {
            self.moe_route_segment(layer, decode, step)?;
            if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Dense {
                self.dense_ffn_segment(layer, step)?;
                self.moe_join_segment(layer, decode, step)?;
                continue;
            }
            let slot_layer = self.moe_slot_layer(layer)?;
            let slot_stream = decode.slot_stream(&self.device, slot_layer);
            let bindings = slots.bindings(slot_layer, slot_stream)?;
            if resident {
                decode.resident_mask(&self.device, slot_layer, bindings, step.rows)?;
            }
            decode.contribute_device(
                &self.device,
                slot_layer,
                self.moe_ffn_binding(layer)?,
                bindings,
                qwen::matrix(
                    &self.work.n,
                    &stream,
                    0,
                    step.rows,
                    self.model_spec.hidden as usize,
                )?,
            )?;
            if !resident
                && decode
                    .peer
                    .as_ref()
                    .is_none_or(|peer| slot_layer < peer.gpu0_layers as usize)
            {
                decode.warmup_completion(&self.device, step.rows)?;
            }
            self.moe_join_segment(layer, decode, step)?;
        }
        Ok(())
    }

    fn moe_peer_tail_decode(
        &mut self,
        decode: &mut Decode,
        slots: &Slots,
        start: usize,
        step: crate::backend::Step<'_>,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        for layer in start..self.model_spec.layers.len() {
            self.moe_route_segment(layer, decode, step)?;
            if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Dense {
                self.dense_ffn_segment(layer, step)?;
                self.moe_join_segment(layer, decode, step)?;
                continue;
            }
            let slot_layer = self.moe_slot_layer(layer)?;
            ensure!(
                decode
                    .peer
                    .as_ref()
                    .is_some_and(|peer| slot_layer >= peer.gpu0_layers as usize),
                "peer tail layer={layer} slot_layer={slot_layer} is not on peer device"
            );
            let slot_stream = decode.slot_stream(&self.device, slot_layer);
            let bindings = slots.bindings(slot_layer, slot_stream)?;
            decode.contribute_device(
                &self.device,
                slot_layer,
                self.moe_ffn_binding(layer)?,
                bindings,
                qwen::matrix(
                    &self.work.n,
                    &stream,
                    0,
                    step.rows,
                    self.model_spec.hidden as usize,
                )?,
            )?;
            self.moe_join_segment(layer, decode, step)?;
        }
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn moe_peer_tail_start(&self) -> Result<Option<usize>> {
        if self
            .moe_slots
            .as_ref()
            .context("MoE slots missing")?
            .all_resident()
        {
            return Ok(None);
        }
        let Some(peer) = self
            .moe
            .as_ref()
            .context("MoE decode missing")?
            .peer
            .as_ref()
        else {
            return Ok(None);
        };
        let mut first = None;
        for layer in 0..self.model_spec.layers.len() {
            if self.model_spec.layers[layer].ffn != crate::model_spec::FfnKind::Moe {
                continue;
            }
            let slot_layer = self.moe_slot_layer(layer)?;
            if slot_layer >= peer.gpu0_layers as usize {
                first.get_or_insert(layer);
            } else {
                ensure!(
                    first.is_none(),
                    "non-contiguous peer tail: model layer={layer} slot_layer={slot_layer} < gpu0_layers={}",
                    peer.gpu0_layers
                );
            }
        }
        if let Some(start) = first {
            ensure!(
                start > 0,
                "peer tail cannot cover layer 0 under partial residency"
            );
        }
        Ok(first)
    }

    #[cfg(test)]
    pub(in crate::backend::cuda::engine) fn clear_moe_graphs(&mut self) {
        self.graph = None;
        self.graphs.clear();
        if let Some(spec) = self.spec.as_mut() {
            spec.moe_verify_graphs.clear();
        }
        if let Some(decode) = self.moe.as_mut() {
            decode.segments.clear();
            decode.verify_segments.clear();
            decode.peer_tail = None;
            decode.verify_peer_tails.clear();
        }
    }

    pub(in crate::backend::cuda::engine) fn capture_moe(&mut self) -> Result<()> {
        let verify = self.spec.is_some();
        let widths = match &self.spec {
            Some(spec) => crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows,
            None => 1..=1,
        };
        let max_rows = *widths.end();
        let resident = self
            .moe_slots
            .as_ref()
            .context("MoE capture has no physical slots")?
            .all_resident();
        if resident
            && if verify {
                self.has_moe_verify_graphs()
            } else {
                self.graph.is_some()
            }
        {
            return Ok(());
        }
        if !resident && self.has_moe_segments() {
            return Ok(());
        }
        ensure!(!self.poisoned, "MoE capture requires healthy state");
        ensure!(
            self.position
                .checked_add(max_rows)
                .is_some_and(|end| end <= self.capacity),
            "MoE capture exceeds KV capacity"
        );
        for rows in widths.clone() {
            self.prepare(rows)?;
        }
        let stream = self.device.stream.clone();
        stream.synchronize()?;
        let saved_position = stream.clone_dtoh(&self.work.position)?;
        let saved_length = stream.clone_dtoh(&self.work.length)?;
        let tail_start = self.moe_peer_tail_start()?;
        let seg_end = tail_start.unwrap_or(self.model_spec.layers.len());
        let mut checkpoint = if let Some(spec) = self.spec.as_mut() {
            spec.checkpoint
                .take()
                .context("MoE verify capture checkpoint missing")?
        } else {
            self.capture_checkpoint
                .take()
                .context("MoE capture checkpoint missing")?
        };
        let mut decode = self.moe.take().context("MoE capture executor missing")?;
        let slots = self.moe_slots.take().context("MoE capture slots missing")?;
        let previous_prefix = self.qwen_device_prefix;
        self.qwen_device_prefix = verify;
        let mut resident_graphs = std::collections::BTreeMap::new();
        let result = (|| -> Result<()> {
            stream.wait(&slots.refill)?;
            checkpoint.save_rows(
                &self.state,
                &self.work.token,
                self.position,
                max_rows,
                &stream,
                None,
            )?;
            for rows in widths {
                let step = crate::backend::Step {
                    rows,
                    input: if verify {
                        crate::backend::Input::Ids
                    } else {
                        crate::backend::Input::Token
                    },
                    mask: crate::backend::Mask::Causal,
                    head: if verify { Head::All } else { Head::Last },
                };
                self.state.set_sequence(
                    &stream,
                    self.position as u32,
                    rows as u32,
                    self.workspace_rows as u32,
                )?;
                self.moe_device_decode(&mut decode, &slots, resident, step)?;
                stream.synchronize()?;
                if let Some(peer) = &decode.peer {
                    peer.device.stream.synchronize()?;
                }
                checkpoint.restore(&mut self.state, &mut self.work.token, &stream)?;
                stream.memcpy_htod(&saved_position, &mut self.work.position)?;
                stream.memcpy_htod(&saved_length, &mut self.work.length)?;
                self.state.set_sequence(
                    &stream,
                    self.position as u32,
                    rows as u32,
                    self.workspace_rows as u32,
                )?;
                stream.synchronize()?;
                if resident {
                    let graph = record(self, |engine| {
                        engine.moe_device_decode(&mut decode, &slots, true, step)
                    })?;
                    graph.launch().context("cuGraphLaunch")?;
                    stream.synchronize().context("cuStreamSynchronize")?;
                    resident_graphs.insert(rows, graph);
                } else {
                    let mut segments = Vec::with_capacity(seg_end);
                    for layer in 0..seg_end {
                        let route = record(self, |engine| {
                            engine.moe_route_segment(layer, &decode, step)
                        })?;
                        let contribute = if self.model_spec.layers[layer].ffn
                            == crate::model_spec::FfnKind::Dense
                        {
                            record(self, |engine| engine.dense_ffn_segment(layer, step))?
                        } else {
                            record(self, |engine| {
                                let slot_layer = engine.moe_slot_layer(layer)?;
                                let slot_stream = decode.slot_stream(&engine.device, slot_layer);
                                let bindings = slots.bindings(slot_layer, slot_stream)?;
                                decode.contribute_device(
                                    &engine.device,
                                    slot_layer,
                                    engine.moe_ffn_binding(layer)?,
                                    bindings,
                                    qwen::matrix(
                                        &engine.work.n,
                                        &stream,
                                        0,
                                        rows,
                                        engine.model_spec.hidden as usize,
                                    )?,
                                )?;
                                Ok(())
                            })?
                        };
                        let join =
                            record(self, |engine| engine.moe_join_segment(layer, &decode, step))?;
                        segments.push(Segments {
                            route,
                            contribute,
                            join,
                        });
                    }
                    let peer_tail = if let Some(start) = tail_start {
                        Some((
                            start,
                            record(self, |engine| {
                                engine.moe_peer_tail_decode(&mut decode, &slots, start, step)
                            })?,
                        ))
                    } else {
                        None
                    };
                    for segment in &segments {
                        segment.route.launch()?;
                        segment.contribute.launch()?;
                        decode.warmup_completion(&self.device, rows)?;
                        segment.join.launch()?;
                    }
                    if let Some((_, tail)) = &peer_tail {
                        tail.launch().context("cuGraphLaunch MoePeerTail")?;
                    }
                    stream.synchronize()?;
                    if let Some(peer) = &decode.peer {
                        peer.device.stream.synchronize()?;
                    }
                    if verify {
                        decode.verify_segments.insert(rows, segments);
                        if let Some(tail) = peer_tail {
                            decode.verify_peer_tails.insert(rows, tail);
                        }
                    } else {
                        decode.segments = segments;
                        decode.peer_tail = peer_tail;
                    }
                }
                let logits = if let Some(spec) = &self.spec {
                    stream
                        .clone_dtoh(&spec.logits.slice(..rows * self.config.vocab_size))?
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect::<Vec<_>>()
                } else if let Some(logits) = &self.flint_logits {
                    stream.clone_dtoh(logits)?
                } else {
                    stream
                        .clone_dtoh(&self.work.logits.slice(..self.config.vocab_size))?
                        .into_iter()
                        .map(bf16::to_f32)
                        .collect::<Vec<_>>()
                };
                let nonfinite = logits.iter().filter(|value| !value.is_finite()).count();
                let mut finite_min = f32::INFINITY;
                let mut finite_max = f32::NEG_INFINITY;
                for value in logits.iter().filter(|value| value.is_finite()).copied() {
                    finite_min = finite_min.min(value);
                    finite_max = finite_max.max(value);
                }
                ensure!(
                    nonfinite == 0,
                    "nonfinite MoE capture logits count={nonfinite} finite_min={finite_min} finite_max={finite_max}"
                );
                checkpoint.restore(&mut self.state, &mut self.work.token, &stream)?;
                stream.memcpy_htod(&saved_position, &mut self.work.position)?;
                stream.memcpy_htod(&saved_length, &mut self.work.length)?;
                stream.synchronize()?;
            }
            Ok(())
        })();
        let restored = (|| -> Result<()> {
            checkpoint.restore(&mut self.state, &mut self.work.token, &stream)?;
            stream.memcpy_htod(&saved_position, &mut self.work.position)?;
            stream.memcpy_htod(&saved_length, &mut self.work.length)?;
            stream.synchronize()?;
            Ok(())
        })();
        if result.is_err() || restored.is_err() {
            decode.segments.clear();
            decode.verify_segments.clear();
            decode.peer_tail = None;
            decode.verify_peer_tails.clear();
        }
        if let Some(spec) = self.spec.as_mut() {
            spec.checkpoint = Some(checkpoint);
        } else {
            self.capture_checkpoint = Some(checkpoint);
        }
        self.qwen_device_prefix = previous_prefix;
        self.moe = Some(decode);
        self.moe_slots = Some(slots);
        if let Err(error) = restored {
            self.poisoned = true;
            return Err(error.context(format!(
                "MoE capture rollback failed; capture error={:?}",
                result.as_ref().err()
            )));
        }
        match result {
            Ok(()) => {
                if let Some(spec) = self.spec.as_mut() {
                    spec.moe_verify_graphs = resident_graphs;
                } else {
                    self.graph = resident_graphs.remove(&1);
                }
            }
            Err(error) => {
                self.poisoned = true;
                return Err(error.context("MoE capture transaction failed"));
            }
        }
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn has_moe_peer_tail(&self) -> bool {
        let Some(expected) = self.moe_peer_tail_start().ok().flatten() else {
            return false;
        };
        let Some(decode) = &self.moe else {
            return false;
        };
        if let Some(spec) = &self.spec {
            (crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows).all(|rows| {
                decode
                    .verify_peer_tails
                    .get(&rows)
                    .is_some_and(|(start, _)| *start == expected)
            })
        } else {
            decode
                .peer_tail
                .as_ref()
                .is_some_and(|(start, _)| *start == expected)
        }
    }

    pub(in crate::backend::cuda::engine) fn has_moe_segments(&self) -> bool {
        let layers = self.model_spec.layers.len();
        let Some(decode) = &self.moe else {
            return false;
        };
        let tail_start = self.moe_peer_tail_start().ok().flatten();
        let seg_end = tail_start.unwrap_or(layers);
        let segments_ok = if let Some(spec) = &self.spec {
            (crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows).all(|rows| {
                decode
                    .verify_segments
                    .get(&rows)
                    .is_some_and(|s| s.len() == seg_end && !s.is_empty())
            })
        } else {
            decode.segments.len() == seg_end && !decode.segments.is_empty()
        };
        segments_ok && (tail_start.is_none() || self.has_moe_peer_tail())
    }

    pub(in crate::backend::cuda::engine) fn has_moe_verify_graphs(&self) -> bool {
        let Some(spec) = &self.spec else {
            return false;
        };
        (crate::dspark::MIN_MOE_VERIFY_ROWS..=spec.max_verify_rows)
            .all(|rows| spec.moe_verify_graphs.contains_key(&rows))
    }

    pub(in crate::backend::cuda::engine) fn run_moe_graph(&mut self, rows: usize) -> Result<()> {
        let mut decode = self.moe.take().context("MoE graph executor missing")?;
        let mut slots = self.moe_slots.take().context("MoE graph slots missing")?;
        let mut token_open = false;
        let result = (|| -> Result<()> {
            if slots.all_resident() {
                ensure!(slots.request_open, "MoE resident graph has no request");
                self.device.stream.wait(&slots.refill)?;
                if let Some(spec) = &self.spec {
                    spec.moe_verify_graphs
                        .get(&rows)
                        .with_context(|| format!("MoE resident verify graph rows={rows} missing"))?
                        .launch()?;
                } else {
                    self.graph
                        .as_ref()
                        .context("MoE resident graph missing")?
                        .launch()?;
                }
                slots.record_all_readers(&self.device)?;
                let layers = self.model_spec.layers.len() as u64 * rows as u64;
                decode.stats[0] += layers;
                decode.stats[1] += layers
                    * u64::from(
                        self.model_spec
                            .moe
                            .as_ref()
                            .context("MoE geometry missing")?
                            .experts_per_tok,
                    );
                decode.stats[4] += layers;
            } else {
                let expected_seg_end = match &decode.peer {
                    Some(peer) => {
                        let mut first = None;
                        for layer in 0..self.model_spec.layers.len() {
                            if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Moe
                                && self.moe_slot_layer(layer)? >= peer.gpu0_layers as usize
                            {
                                first = Some(layer);
                                break;
                            }
                        }
                        first.unwrap_or(self.model_spec.layers.len())
                    }
                    None => self.model_spec.layers.len(),
                };
                let segments = if rows == 1 {
                    &decode.segments
                } else {
                    decode
                        .verify_segments
                        .get(&rows)
                        .with_context(|| format!("MoE verify graph width={rows} missing"))?
                };
                ensure!(
                    segments.len() == expected_seg_end,
                    "MoE graph layers={} expected={expected_seg_end}",
                    segments.len(),
                );
                slots.begin_token()?;
                token_open = true;
                for layer in 0..expected_seg_end {
                    let segment = decode.segment(rows, layer)?;
                    segment.route.launch()?;
                    if self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Dense {
                        segment.contribute.launch()?;
                        segment.join.launch()?;
                    } else {
                        let stream = &self.device.stream;
                        let hidden = self.model_spec.hidden as usize;
                        slots.run_graph(
                            &self.device,
                            &mut decode,
                            u32::try_from(self.moe_slot_layer(layer)?)?,
                            layer,
                            self.moe_ffn_binding(layer)?,
                            (
                                qwen::matrix(&self.work.n, stream, 0, rows, hidden)?,
                                &mut TensorMut(qwen::matrix(
                                    &self.work.out,
                                    stream,
                                    0,
                                    rows,
                                    hidden,
                                )?),
                            ),
                        )?;
                    }
                }
                if expected_seg_end < self.model_spec.layers.len() {
                    ensure!(slots.request_open, "MoE peer tail graph has no request");
                    self.device.stream.wait(&slots.refill)?;
                    let (start, tail) = if rows == 1 {
                        decode
                            .peer_tail
                            .as_ref()
                            .context("MoE peer tail graph missing")?
                    } else {
                        decode.verify_peer_tails.get(&rows).with_context(|| {
                            format!("MoE verify peer tail graph width={rows} missing")
                        })?
                    };
                    ensure!(
                        *start == expected_seg_end,
                        "MoE peer tail start={start} expected={expected_seg_end}"
                    );
                    tail.launch()?;
                    slots.record_peer_readers(&self.device)?;
                    let peer_moe_layers = (expected_seg_end..self.model_spec.layers.len())
                        .filter(|&layer| {
                            self.model_spec.layers[layer].ffn == crate::model_spec::FfnKind::Moe
                        })
                        .count() as u64;
                    let top_k = u64::from(
                        self.model_spec
                            .moe
                            .as_ref()
                            .context("MoE geometry missing")?
                            .experts_per_tok,
                    );
                    decode.stats[0] += peer_moe_layers;
                    decode.stats[1] += peer_moe_layers * rows as u64 * top_k;
                    decode.stats[4] += peer_moe_layers;
                }
            }
            if self.spec.is_none() {
                self.logits_cache.mark_valid();
            } else {
                self.logits_cache.invalidate();
            }
            Ok(())
        })();
        let completed = if token_open {
            slots.finish_token()
        } else {
            Ok(())
        };
        let result = result.and(completed);
        self.moe = Some(decode);
        self.moe_slots = Some(slots);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
