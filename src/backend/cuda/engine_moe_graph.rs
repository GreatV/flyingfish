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
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let launched = launch(engine);
    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
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
    graph.upload()?;
    stream.synchronize()?;
    Ok(graph)
}

impl Decode {
    fn resident_mask(&self, device: &Device, slots: SlotBindings) -> Result<()> {
        let stream = &device.stream;
        let k = self.geometry.top_k;
        let ids = qwen::view(&self.ids, stream, 0, [1, k], [u64::from(k), 1])?;
        let active = qwen::view(&self.active, stream, 0, [1, k], [u64::from(k), 1])?;
        let generation = qwen::view(&self.generation, stream, 0, [1, k], [u64::from(k), 1])?;
        // SAFETY: every table covers the bound expert domain and every assignment buffer covers k.
        unsafe {
            stream
                .launch_builder(&self.mask)
                .arg(&ids.ptr)
                .arg(&slots.valid.ptr)
                .arg(&slots.generation.ptr)
                .arg(&active.ptr)
                .arg(&generation.ptr)
                .arg(&(k as i32))
                .arg(&(self.geometry.experts as i32))
                .launch(grid((k as usize).div_ceil(256), 256))?;
        }
        Ok(())
    }

    fn warmup_completion(&mut self, device: &Device) -> Result<()> {
        let k = self.geometry.top_k as usize;
        device
            .stream
            .memcpy_dtod(&self.final_active.slice(..k), &mut self.done.slice_mut(..k))?;
        Ok(())
    }
}

impl Engine {
    fn moe_route_segment(&mut self, layer: usize, decode: &Decode) -> Result<()> {
        use crate::backend::Operator as O;
        let step = crate::backend::Step {
            rows: 1,
            input: crate::backend::Input::Token,
            mask: crate::backend::Mask::Causal,
            head: Head::Last,
        };
        if layer == 0 {
            self.run_operator(O::Embed, 0, step, &mut None)?;
        }
        self.qwen_layer_begin(layer, step)?;
        self.run_operator(O::Norm, layer, step, &mut None)?;
        self.run_mixer(self.model_spec.layers[layer].mixer, layer, step, &mut None)?;
        self.run_operator(O::PostNorm, layer, step, &mut None)?;
        decode.route_device(
            &self.device,
            self.moe_ffn_binding(layer)?,
            qwen::matrix(
                &self.work.n,
                &self.device.stream,
                0,
                1,
                self.model_spec.hidden as usize,
            )?,
        )
    }

    fn moe_join_segment(&mut self, layer: usize, decode: &Decode) -> Result<()> {
        use crate::backend::Operator as O;
        let stream = &self.device.stream;
        let hidden = self.model_spec.hidden as usize;
        decode.combine_device(
            &self.device,
            self.moe_ffn_binding(layer)?,
            qwen::matrix(&self.work.n, stream, 0, 1, hidden)?,
            &mut TensorMut(qwen::matrix(&self.work.out, stream, 0, 1, hidden)?),
        )?;
        let step = crate::backend::Step {
            rows: 1,
            input: crate::backend::Input::Token,
            mask: crate::backend::Mask::Causal,
            head: Head::Last,
        };
        self.run_operator(O::NextNorm, layer, step, &mut None)?;
        if layer + 1 == self.model_spec.layers.len() {
            self.run_operator(O::Head, 0, step, &mut None)?;
            self.qwen_advance()?;
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
        retain_events: bool,
    ) -> Result<()> {
        let stream = self.device.stream.clone();
        if !resident {
            stream.memset_zeros(&mut decode.active)?;
            stream.memset_zeros(&mut decode.contributions)?;
        }
        for layer in 0..self.model_spec.layers.len() {
            self.moe_route_segment(layer, decode)?;
            let bindings = slots.bindings(layer, &stream)?;
            if resident {
                decode.resident_mask(&self.device, bindings)?;
            }
            let reader = decode.contribute_device(
                &self.device,
                self.moe_ffn_binding(layer)?,
                bindings,
                qwen::matrix(&self.work.n, &stream, 0, 1, self.model_spec.hidden as usize)?,
            )?;
            if retain_events {
                decode.capture_events.push(reader);
            }
            if !resident {
                decode.warmup_completion(&self.device)?;
            }
            self.moe_join_segment(layer, decode)?;
        }
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn capture_moe(&mut self) -> Result<()> {
        ensure!(
            self.spec.is_none(),
            "MoE graphs cannot attach a draft runtime"
        );
        let resident = self
            .moe_slots
            .as_ref()
            .context("MoE capture has no physical slots")?
            .all_resident();
        if resident && self.graph.is_some() {
            return Ok(());
        }
        if !resident && self.has_moe_segments() {
            return Ok(());
        }
        ensure!(!self.poisoned, "MoE capture requires healthy state");
        ensure!(
            self.position < self.capacity,
            "MoE capture exceeds KV capacity"
        );
        self.prepare(1)?;
        let stream = self.device.stream.clone();
        stream.synchronize()?;
        let saved_position = stream.clone_dtoh(&self.work.position)?;
        let saved_length = stream.clone_dtoh(&self.work.length)?;
        let mut checkpoint = self
            .capture_checkpoint
            .take()
            .context("MoE capture checkpoint missing")?;
        let mut decode = self.moe.take().context("MoE capture executor missing")?;
        let slots = self.moe_slots.take().context("MoE capture slots missing")?;
        decode.capture_events.reserve(self.model_spec.layers.len());
        decode.segments.reserve(self.model_spec.layers.len());
        let result = (|| -> Result<Option<CudaGraph>> {
            stream.wait(&slots.refill)?;
            checkpoint.save(&self.state, &self.work.token, self.position, &stream, None)?;
            self.moe_device_decode(&mut decode, &slots, resident, false)?;
            stream.synchronize()?;
            checkpoint.restore(&mut self.state, &mut self.work.token, &stream)?;
            stream.memcpy_htod(&saved_position, &mut self.work.position)?;
            stream.memcpy_htod(&saved_length, &mut self.work.length)?;
            stream.synchronize()?;
            if resident {
                let graph = record(self, |engine| {
                    engine.moe_device_decode(&mut decode, &slots, true, true)
                })?;
                graph.launch()?;
                stream.synchronize()?;
                return Ok(Some(graph));
            }
            for layer in 0..self.model_spec.layers.len() {
                let route = record(self, |engine| engine.moe_route_segment(layer, &decode))?;
                let contribute = record(self, |engine| {
                    let reader = decode.contribute_device(
                        &engine.device,
                        engine.moe_ffn_binding(layer)?,
                        slots.bindings(layer, &stream)?,
                        qwen::matrix(
                            &engine.work.n,
                            &stream,
                            0,
                            1,
                            engine.model_spec.hidden as usize,
                        )?,
                    )?;
                    decode.capture_events.push(reader);
                    Ok(())
                })?;
                let join = record(self, |engine| engine.moe_join_segment(layer, &decode))?;
                decode.segments.push(Segments {
                    route,
                    contribute,
                    join,
                });
            }
            for layer in 0..decode.segments.len() {
                decode.segments[layer].route.launch()?;
                decode.segments[layer].contribute.launch()?;
                decode.warmup_completion(&self.device)?;
                decode.segments[layer].join.launch()?;
            }
            stream.synchronize()?;
            Ok(None)
        })();
        let result = result.and_then(|graph| {
            let logits = stream.clone_dtoh(&self.work.logits.slice(..self.config.vocab_size))?;
            ensure!(
                logits.iter().all(|value| value.to_f32().is_finite()),
                "nonfinite MoE capture logits"
            );
            Ok(graph)
        });
        let restored = (|| -> Result<()> {
            checkpoint.restore(&mut self.state, &mut self.work.token, &stream)?;
            stream.memcpy_htod(&saved_position, &mut self.work.position)?;
            stream.memcpy_htod(&saved_length, &mut self.work.length)?;
            stream.synchronize()?;
            Ok(())
        })();
        if result.is_err() || restored.is_err() {
            decode.segments.clear();
            decode.capture_events.clear();
        }
        self.capture_checkpoint = Some(checkpoint);
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
            Ok(graph) => self.graph = graph,
            Err(error) => {
                self.poisoned = true;
                return Err(error.context("MoE capture transaction failed"));
            }
        }
        Ok(())
    }

    pub(in crate::backend::cuda::engine) fn has_moe_segments(&self) -> bool {
        self.moe.as_ref().is_some_and(|decode| {
            decode.segments.len() == self.model_spec.layers.len() && !decode.segments.is_empty()
        })
    }

    pub(in crate::backend::cuda::engine) fn run_moe_graph(&mut self) -> Result<()> {
        let mut decode = self.moe.take().context("MoE graph executor missing")?;
        let mut slots = self.moe_slots.take().context("MoE graph slots missing")?;
        let result = (|| -> Result<()> {
            if slots.all_resident() {
                ensure!(slots.request_open, "MoE resident graph has no request");
                self.device.stream.wait(&slots.refill)?;
                self.graph
                    .as_ref()
                    .context("MoE resident graph missing")?
                    .launch()?;
                slots.record_all_readers(&self.device)?;
                let layers = self.model_spec.layers.len() as u64;
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
                ensure!(
                    decode.segments.len() == self.model_spec.layers.len(),
                    "MoE graph layers={} expected={}",
                    decode.segments.len(),
                    self.model_spec.layers.len()
                );
                slots.begin_token()?;
                for layer in 0..self.model_spec.layers.len() {
                    decode.segments[layer].route.launch()?;
                    let stream = &self.device.stream;
                    let hidden = self.model_spec.hidden as usize;
                    slots.run_graph(
                        &self.device,
                        &mut decode,
                        layer as u32,
                        self.moe_ffn_binding(layer)?,
                        (
                            qwen::matrix(&self.work.n, stream, 0, 1, hidden)?,
                            &mut TensorMut(qwen::matrix(&self.work.out, stream, 0, 1, hidden)?),
                        ),
                    )?;
                }
                slots.finish_token()?;
            }
            self.logits_cache.mark_valid();
            Ok(())
        })();
        self.moe = Some(decode);
        self.moe_slots = Some(slots);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
