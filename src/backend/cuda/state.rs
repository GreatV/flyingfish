//! Engine-owned persistent mixer state; operators receive non-owning layer views.
use super::{
    contracts::{SeqRef, Tensor},
    engine::Kv,
};
use crate::{
    model_spec::{Architecture, Mixer, ModelSpec},
    op_contract::{GdnSpec, SequenceMeta},
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, PushKernelArg};
use half::bf16;
use std::sync::Arc;

pub(crate) struct GdnState {
    pub s: CudaSlice<f32>,
    pub history: CudaSlice<f32>,
}
pub(crate) struct StateStore {
    pub kv: Vec<Kv>,
    pub gdn: Vec<GdnState>,
    pub sequence: Option<CudaSlice<i32>>,
    capacity: u32,
}
impl StateStore {
    pub fn new(spec: &ModelSpec, capacity: usize, stream: &Arc<CudaStream>) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= spec.max_position as usize && capacity <= i32::MAX as usize,
            "invalid state capacity"
        );
        let elements = capacity
            .checked_mul(spec.kv_heads as usize)
            .and_then(|n| n.checked_mul(spec.head_dim as usize))
            .context("KV state size overflow")?;
        let mut kv = Vec::new();
        let mut gdn = Vec::new();
        for layer in &spec.layers {
            match layer.mixer {
                Mixer::FullAttention => {
                    ensure!(
                        layer.state_index as usize == kv.len(),
                        "invalid full-state mapping"
                    );
                    kv.push(Kv {
                        k: stream.alloc_zeros::<bf16>(elements)?,
                        v: stream.alloc_zeros::<bf16>(elements)?,
                    });
                }
                Mixer::Gdn => {
                    ensure!(
                        spec.architecture == Architecture::Qwen35Text
                            && layer.state_index as usize == gdn.len(),
                        "invalid GDN state mapping"
                    );
                    let g = GdnSpec::q38();
                    gdn.push(GdnState {
                        s: stream.alloc_zeros::<f32>(usize::try_from(g.state_bytes()? / 4)?)?,
                        history: stream
                            .alloc_zeros::<f32>(usize::try_from(g.history_bytes()? / 4)?)?,
                    });
                }
            }
        }
        let sequence = if spec.architecture == Architecture::Qwen35Text {
            Some(stream.clone_htod(&[0i32, 1, 1, capacity as i32, 1, 0, 0, 0])?)
        } else {
            None
        };
        Ok(Self {
            kv,
            gdn,
            sequence,
            capacity: capacity as u32,
        })
    }
    pub fn seq_ref(&self, stream: &Arc<CudaStream>) -> Result<SeqRef> {
        let meta = self
            .sequence
            .as_ref()
            .context("model has no Q38 sequence metadata")?;
        let (ptr, _read) = meta.device_ptr(stream);
        Ok(SeqRef { ptr })
    }
    pub fn set_sequence(
        &mut self,
        stream: &Arc<CudaStream>,
        prefix: u32,
        rows: u32,
        max_rows: u32,
    ) -> Result<()> {
        let meta = SequenceMeta {
            prefix: i32::try_from(prefix)?,
            valid_rows: i32::try_from(rows)?,
            key_count: i32::try_from(
                prefix
                    .checked_add(rows)
                    .context("sequence count overflow")?,
            )?,
            capacity: self.capacity as i32,
            causal: 1,
            reserved: [0; 3],
        };
        meta.validate(self.capacity, max_rows)?;
        stream.memcpy_htod(
            &[
                meta.prefix,
                meta.valid_rows,
                meta.key_count,
                meta.capacity,
                meta.causal,
                0,
                0,
                0,
            ],
            self.sequence
                .as_mut()
                .context("missing sequence metadata")?,
        )?;
        Ok(())
    }
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        for state in &mut self.gdn {
            stream.memset_zeros(&mut state.s)?;
            stream.memset_zeros(&mut state.history)?;
        }
        if self.sequence.is_some() {
            self.set_sequence(stream, 0, 1, 1)?;
        }
        // Invalid KV remains inaccessible through sequence bounds; no KV zeroing is required.
        Ok(())
    }
    pub fn bytes(&self) -> usize {
        self.kv
            .iter()
            .map(|v| (v.k.len() + v.v.len()) * 2)
            .sum::<usize>()
            + self
                .gdn
                .iter()
                .map(|v| (v.s.len() + v.history.len()) * 4)
                .sum::<usize>()
            + self.sequence.as_ref().map_or(0, |m| m.len() * 4)
    }
}

/// Preallocated snapshot for capture/warmup. Only the KV slots that can be written
/// are copied; every recurrent state/history element is copied.
pub(crate) struct StateCheckpoint {
    kv: Vec<Kv>,
    gdn: Vec<GdnState>,
    sequence: Option<CudaSlice<i32>>,
    token: CudaSlice<u32>,
    prefix: usize,
    rows: usize,
    active_rows: usize,
    heads: usize,
    dim: usize,
    capacity: usize,
    valid: bool,
}
impl StateCheckpoint {
    pub fn new(
        store: &StateStore,
        spec: &ModelSpec,
        rows: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= store.capacity as usize,
            "invalid checkpoint window"
        );
        let (heads, dim) = (spec.kv_heads as usize, spec.head_dim as usize);
        let slots = heads
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(dim))
            .context("checkpoint KV size overflow")?;
        let mut kv = Vec::new();
        let mut gdn = Vec::new();
        for _ in &store.kv {
            kv.push(Kv {
                k: stream.alloc_zeros(slots)?,
                v: stream.alloc_zeros(slots)?,
            });
        }
        for state in &store.gdn {
            gdn.push(GdnState {
                s: stream.alloc_zeros(state.s.len())?,
                history: stream.alloc_zeros(state.history.len())?,
            });
        }
        Ok(Self {
            kv,
            gdn,
            sequence: store
                .sequence
                .as_ref()
                .map(|_| stream.alloc_zeros(8))
                .transpose()?,
            token: stream.alloc_zeros(1)?,
            prefix: 0,
            rows,
            active_rows: 0,
            heads,
            dim,
            capacity: store.capacity as usize,
            valid: false,
        })
    }
    fn check(&self, store: &StateStore) -> Result<()> {
        ensure!(
            self.kv.len() == store.kv.len()
                && self.gdn.len() == store.gdn.len()
                && self.sequence.is_some() == store.sequence.is_some()
                && self.capacity == store.capacity as usize,
            "checkpoint/state geometry mismatch"
        );
        for (a, b) in self.gdn.iter().zip(&store.gdn) {
            ensure!(
                a.s.len() == b.s.len() && a.history.len() == b.history.len(),
                "checkpoint GDN extent mismatch"
            );
        }
        let extent = self
            .heads
            .checked_mul(self.capacity)
            .and_then(|n| n.checked_mul(self.dim))
            .context("checkpoint extent overflow")?;
        ensure!(
            store
                .kv
                .iter()
                .all(|kv| kv.k.len() == extent && kv.v.len() == extent),
            "checkpoint KV extent mismatch"
        );
        Ok(())
    }
    pub fn save(
        &mut self,
        store: &StateStore,
        token: &CudaSlice<u32>,
        prefix: usize,
        stream: &Arc<CudaStream>,
        validation: Option<(&CudaFunction, Tensor<u32, 1>)>,
    ) -> Result<()> {
        self.valid = false;
        self.check(store)?;
        ensure!(
            token.len() == 1 && prefix < self.capacity,
            "checkpoint writes exceed capacity"
        );
        let active_rows = self.rows.min(self.capacity - prefix);
        if let Some(seq) = &store.sequence {
            ensure!(seq.len() == 8, "checkpoint sequence extent mismatch");
            if let Some((check, flag)) = validation {
                flag.validate()?;
                ensure!(flag.stride == [1], "checkpoint status extent mismatch");
                let flag_ptr = flag.element_ptr(flag.shape[0] as usize - 1)?;
                unsafe {
                    stream
                        .launch_builder(check)
                        .arg(seq)
                        .arg(&(prefix as i32))
                        .arg(&(self.capacity as i32))
                        .arg(&(self.rows as i32))
                        .arg(&flag_ptr)
                        .launch(super::ops::grid(1, 1))?;
                }
            } else {
                let host = stream.clone_dtoh(seq)?;
                ensure!(
                    host.len() == 8 && host[0] >= 0 && host[0] as usize == prefix,
                    "checkpoint prefix differs from device state"
                );
                SequenceMeta {
                    prefix: host[0],
                    valid_rows: host[1],
                    key_count: host[2],
                    capacity: host[3],
                    causal: host[4],
                    reserved: [host[5], host[6], host[7]],
                }
                .validate(store.capacity, self.rows as u32)?;
            }
        }
        for (src, dst) in store.gdn.iter().zip(&mut self.gdn) {
            stream.memcpy_dtod(&src.s, &mut dst.s)?;
            stream.memcpy_dtod(&src.history, &mut dst.history)?;
        }
        for (src, dst) in store.kv.iter().zip(&mut self.kv) {
            for (a, b) in [(&src.k, &mut dst.k), (&src.v, &mut dst.v)] {
                super::copy::columns(
                    stream,
                    a,
                    b,
                    super::copy::Rect {
                        rows: self.heads,
                        width: active_rows * self.dim,
                        src_pitch: self.capacity * self.dim,
                        src_col: prefix * self.dim,
                        dst_pitch: self.rows * self.dim,
                        dst_col: 0,
                    },
                )?;
            }
        }
        if let (Some(seq), Some(saved)) = (&store.sequence, &mut self.sequence) {
            stream.memcpy_dtod(seq, saved)?;
        }
        stream.memcpy_dtod(token, &mut self.token)?;
        self.prefix = prefix;
        self.active_rows = active_rows;
        self.valid = true;
        Ok(())
    }
    pub fn restore(
        &self,
        store: &mut StateStore,
        token: &mut CudaSlice<u32>,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        ensure!(self.valid, "checkpoint has no valid snapshot");
        self.check(store)?;
        ensure!(token.len() == 1, "checkpoint token extent mismatch");
        for (src, dst) in self.gdn.iter().zip(&mut store.gdn) {
            stream.memcpy_dtod(&src.s, &mut dst.s)?;
            stream.memcpy_dtod(&src.history, &mut dst.history)?;
        }
        for (src, dst) in self.kv.iter().zip(&mut store.kv) {
            for (a, b) in [(&src.k, &mut dst.k), (&src.v, &mut dst.v)] {
                super::copy::columns(
                    stream,
                    a,
                    b,
                    super::copy::Rect {
                        rows: self.heads,
                        width: self.active_rows * self.dim,
                        src_pitch: self.rows * self.dim,
                        src_col: 0,
                        dst_pitch: self.capacity * self.dim,
                        dst_col: self.prefix * self.dim,
                    },
                )?;
            }
        }
        if let (Some(saved), Some(seq)) = (&self.sequence, &mut store.sequence) {
            stream.memcpy_dtod(saved, seq)?;
        }
        stream.memcpy_dtod(&self.token, token)?;
        Ok(())
    }
    pub fn bytes(&self) -> usize {
        self.kv
            .iter()
            .map(|v| (v.k.len() + v.v.len()) * 2)
            .sum::<usize>()
            + self
                .gdn
                .iter()
                .map(|v| (v.s.len() + v.history.len()) * 4)
                .sum::<usize>()
            + 4
            + self.sequence.as_ref().map_or(0, |s| s.len() * 4)
    }
}
