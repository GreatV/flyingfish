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
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, PushKernelArg};
use half::bf16;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct CacheView {
    pub k_code: u64,
    pub v_code: u64,
    pub k_stats: u64,
    pub v_stats: u64,
    pub k_ring: u64,
    pub v_ring: u64,
    pub cursor: u64,
    pub capacity: i32,
    pub ring: i32,
}
unsafe impl DeviceRepr for CacheView {}
impl CacheView {
    pub fn validate(self, capacity: u32, rows: u32) -> Result<()> {
        ensure!(
            self.capacity == capacity as i32
                && self.ring == self.capacity.min(crate::kv::RING_TOKENS as i32)
                && self.ring > 0,
            "compact KV binding geometry"
        );
        ensure!(
            (1..=crate::mtp::MAX_VERIFY_ROWS as u32).contains(&rows),
            "compact KV binding row extent"
        );
        for ptr in [self.k_code, self.v_code, self.k_ring, self.v_ring] {
            ensure!(
                ptr != 0 && ptr.is_multiple_of(16),
                "compact KV code/ring alignment"
            );
        }
        for ptr in [self.k_stats, self.v_stats, self.cursor] {
            ensure!(
                ptr != 0 && ptr.is_multiple_of(4),
                "compact KV statistic/cursor alignment"
            );
        }
        Ok(())
    }
}

pub(crate) struct CompactKv {
    pub k_code: CudaSlice<u8>,
    pub v_code: CudaSlice<u8>,
    pub k_stats: CudaSlice<f32>,
    pub v_stats: CudaSlice<f32>,
    pub cursor: CudaSlice<i32>,
    pub layout: crate::kv::KvLayout,
    pack: CudaFunction,
    tail: CudaFunction,
    update: CudaFunction,
    dequant: CudaFunction,
}

impl CompactKv {
    fn new(layout: crate::kv::KvLayout, stream: &Arc<CudaStream>) -> Result<Self> {
        let module = super::cubin::module(stream.context(), "tcmqa_d256")?;
        Ok(Self {
            k_code: stream.alloc_zeros(layout.codes)?,
            v_code: stream.alloc_zeros(layout.codes)?,
            k_stats: stream.alloc_zeros(layout.k_stats)?,
            v_stats: stream.alloc_zeros(layout.v_stats)?,
            cursor: stream.alloc_zeros(4)?,
            layout,
            pack: module.load_function("kv_cache_pack")?,
            tail: module.load_function("kv_cache_tail")?,
            update: module.load_function("kv_cache_cursor")?,
            dequant: module.load_function("kv_cache_dequant")?,
        })
    }
    pub fn view(&self, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>, s: &Arc<CudaStream>) -> CacheView {
        CacheView {
            k_code: self.k_code.device_ptr(s).0,
            v_code: self.v_code.device_ptr(s).0,
            k_stats: self.k_stats.device_ptr(s).0,
            v_stats: self.v_stats.device_ptr(s).0,
            k_ring: k.device_ptr(s).0,
            v_ring: v.device_ptr(s).0,
            cursor: self.cursor.device_ptr(s).0,
            capacity: self.layout.capacity as i32,
            ring: self.layout.ring as i32,
        }
    }
    pub fn bytes(&self) -> usize {
        self.k_code.len()
            + self.v_code.len()
            + (self.k_stats.len() + self.v_stats.len() + self.cursor.len()) * 4
    }
    pub fn cursor_mode(
        &self,
        view: CacheView,
        meta: SeqRef,
        mode: i32,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        view.validate(self.layout.capacity as u32, 1)?;
        meta.validate()?;
        ensure!(
            (0..=3).contains(&mode),
            "compact KV cursor mode outside 0..3"
        );
        unsafe {
            s.launch_builder(&self.update)
                .arg(&view)
                .arg(&meta.ptr)
                .arg(&mode)
                .launch(super::ops::grid(1, 1))
                .context("compact KV cursor launch")?;
        }
        Ok(())
    }
    pub fn finish(
        &self,
        view: CacheView,
        meta: SeqRef,
        full: Option<(&CudaSlice<bf16>, &CudaSlice<bf16>, usize)>,
        force: bool,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        view.validate(self.layout.capacity as u32, 1)?;
        meta.validate()?;
        if let Some((k, v, rows)) = full {
            let elements = 4 * self.layout.capacity * 256;
            ensure!(
                k.len() == elements && v.len() == elements,
                "compact KV pack full K/V extent"
            );
            ensure!(
                rows > 0 && rows <= self.layout.capacity,
                "compact KV pack row extent"
            );
        }
        let (k, v, rows) = full.map_or((0, 0, 4), |(k, v, n)| {
            (k.device_ptr(s).0, v.device_ptr(s).0, n)
        });
        let force = i32::from(force);
        unsafe {
            s.launch_builder(&self.pack)
                .arg(&view)
                .arg(&k)
                .arg(&v)
                .arg(&meta.ptr)
                .arg(&force)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (rows.div_ceil(crate::kv::GROUP_TOKENS) as u32 + 1, 4, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("compact KV pack launch")?;
        }
        if full.is_some() {
            unsafe {
                s.launch_builder(&self.tail)
                    .arg(&view)
                    .arg(&k)
                    .arg(&v)
                    .arg(&meta.ptr)
                    .launch(super::ops::flat(4 * self.layout.ring * 256))
                    .context("compact KV live-tail launch")?;
            }
        }
        self.cursor_mode(
            view,
            meta,
            if full.is_some() {
                3
            } else if force != 0 {
                2
            } else {
                0
            },
            s,
        )
    }
    pub fn dequant_prefix(
        &self,
        view: CacheView,
        meta: SeqRef,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        prefix: usize,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        view.validate(self.layout.capacity as u32, 1)?;
        meta.validate()?;
        ensure!(
            prefix <= self.layout.capacity,
            "compact KV dequant prefix exceeds capacity"
        );
        ensure!(
            k.len() == 4 * self.layout.capacity * 256 && v.len() == k.len(),
            "compact KV dequant full K/V extent"
        );
        if prefix > 0 {
            unsafe {
                s.launch_builder(&self.dequant)
                    .arg(&view)
                    .arg(&k.device_ptr(s).0)
                    .arg(&v.device_ptr(s).0)
                    .arg(&meta.ptr)
                    .launch(super::ops::flat(4 * prefix * 256))
                    .context("compact KV prefix dequant launch")?;
            }
        }
        Ok(())
    }
}

struct PrefixTransfer<'a> {
    stream: &'a Arc<CudaStream>,
    ring: &'a mut cudarc::driver::PinnedHostSlice<u8>,
    restore: Option<&'a [u8]>,
    bytes: Vec<u8>,
    offset: usize,
}
impl PrefixTransfer<'_> {
    fn segment<T: DeviceRepr>(
        &mut self,
        tensor: &mut CudaSlice<T>,
        start: usize,
        elements: usize,
    ) -> Result<()> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<T>())
            .context("prefix segment overflow")?;
        ensure!(
            start
                .checked_add(elements)
                .is_some_and(|end| end <= tensor.len()),
            "prefix segment outside tensor"
        );
        if let Some(data) = self.restore {
            let data = data
                .get(self.offset..self.offset + bytes)
                .context("compact prefix payload extent")?;
            super::engine::htod_pinned_chunks(self.stream, tensor, start, data, self.ring)?;
        } else {
            self.bytes.extend(super::engine::dtoh_pinned_chunks(
                self.stream,
                tensor,
                start,
                bytes,
                self.ring,
            )?);
        }
        self.offset += bytes;
        Ok(())
    }
}
impl Kv {
    pub(super) fn compact_prefix_block(
        &mut self,
        span: std::ops::Range<usize>,
        prefix: usize,
        stream: &Arc<CudaStream>,
        ring: &mut cudarc::driver::PinnedHostSlice<u8>,
        restore: Option<&[u8]>,
    ) -> Result<Vec<u8>> {
        let c = self
            .compact
            .as_mut()
            .context("compact prefix format missing")?;
        let (start, valid) = (span.start, span.len());
        let encoded = span
            .end
            .min(crate::kv::sealed_until(prefix))
            .saturating_sub(start);
        let expected = crate::kv::KvFormat::Affine8.block_bytes(start, valid, prefix, 4, 256)?;
        if let Some(data) = restore {
            ensure!(data.len() == expected, "compact prefix payload byte count");
        }
        let mut transfer = PrefixTransfer {
            stream,
            ring,
            restore,
            bytes: Vec::with_capacity(if restore.is_some() { 0 } else { expected }),
            offset: 0,
        };
        for tensor in [&mut c.k_code, &mut c.v_code] {
            for h in 0..4 {
                transfer.segment(tensor, (h * c.layout.capacity + start) * 256, encoded * 256)?;
            }
        }
        let (kpitch, kstart, kn, vpitch, vstart, vn) = (
            c.layout.capacity.div_ceil(crate::kv::GROUP_TOKENS) * 256 * 2,
            start / crate::kv::GROUP_TOKENS * 256 * 2,
            encoded / crate::kv::GROUP_TOKENS * 256 * 2,
            c.layout.capacity * 2,
            start * 2,
            encoded * 2,
        );
        for h in 0..4 {
            transfer.segment(&mut c.k_stats, h * kpitch + kstart, kn)?;
        }
        for h in 0..4 {
            transfer.segment(&mut c.v_stats, h * vpitch + vstart, vn)?;
        }
        for tensor in [&mut self.k, &mut self.v] {
            for h in 0..4 {
                for pos in span
                    .clone()
                    .filter(|&p| p < crate::kv::SINK_TOKENS || p >= crate::kv::sealed_until(prefix))
                {
                    transfer.segment(
                        tensor,
                        (h * c.layout.ring + crate::kv::ring_slot(pos, c.layout.ring)?) * 256,
                        256,
                    )?;
                }
            }
        }
        ensure!(
            transfer.offset == expected,
            "compact prefix typed coverage mismatch"
        );
        Ok(transfer.bytes)
    }
}

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
        Self::new_with_format(spec, capacity, crate::kv::KvFormat::Bf16, stream)
    }
    pub fn new_with_format(
        spec: &ModelSpec,
        capacity: usize,
        format: crate::kv::KvFormat,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= spec.max_position as usize && capacity <= i32::MAX as usize,
            "invalid state capacity"
        );
        ensure!(
            format == crate::kv::KvFormat::Bf16 || (spec.kv_heads == 4 && spec.head_dim == 256),
            "compact cache requires four256-wide KV heads"
        );
        let layout = format.layout(capacity, spec.kv_heads as usize, spec.head_dim as usize)?;
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
                        k: stream.alloc_zeros::<bf16>(layout.ring_elements)?,
                        v: stream.alloc_zeros::<bf16>(layout.ring_elements)?,
                        compact: if format == crate::kv::KvFormat::Bf16 {
                            None
                        } else {
                            Some(CompactKv::new(layout, stream)?)
                        },
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
        for kv in &mut self.kv {
            if let Some(c) = &mut kv.compact {
                stream.memset_zeros(&mut c.cursor)?;
            }
        }
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
        self.kv.iter().map(Kv::bytes).sum::<usize>()
            + self
                .gdn
                .iter()
                .map(|v| (v.s.len() + v.history.len()) * 4)
                .sum::<usize>()
            + self.sequence.as_ref().map_or(0, |m| m.len() * 4)
    }
    pub fn compact_commit(&self, stream: &Arc<CudaStream>, force: bool) -> Result<()> {
        if self.kv.first().is_none_or(|kv| kv.compact.is_none()) {
            return Ok(());
        }
        let meta = self.seq_ref(stream)?;
        for kv in &self.kv {
            if let Some(c) = &kv.compact {
                c.finish(c.view(&kv.k, &kv.v, stream), meta, None, force, stream)?;
            }
        }
        Ok(())
    }
    pub fn compact_defer(&self, stream: &Arc<CudaStream>) -> Result<()> {
        for kv in &self.kv {
            if let Some(c) = &kv.compact {
                c.cursor_mode(
                    c.view(&kv.k, &kv.v, stream),
                    self.seq_ref(stream)?,
                    1,
                    stream,
                )?;
            }
        }
        Ok(())
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
                compact: None,
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
            store.kv.iter().all(|kv| {
                let n = kv
                    .compact
                    .as_ref()
                    .map_or(extent, |c| c.layout.ring_elements);
                kv.k.len() == n && kv.v.len() == n
            }),
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
                        .launch(super::ops::grid(1, 1))
                        .context("checkpoint sequence-check launch")?;
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
            let cap = src
                .compact
                .as_ref()
                .map_or(self.capacity, |c| c.layout.ring);
            for (a, b) in [(&src.k, &mut dst.k), (&src.v, &mut dst.v)] {
                if src.compact.is_none() {
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
                    continue;
                }
                for row in 0..active_rows {
                    let pos = crate::kv::ring_slot(prefix + row, cap)?;
                    super::copy::columns(
                        stream,
                        a,
                        b,
                        super::copy::Rect {
                            rows: self.heads,
                            width: self.dim,
                            src_pitch: cap * self.dim,
                            src_col: pos * self.dim,
                            dst_pitch: self.rows * self.dim,
                            dst_col: row * self.dim,
                        },
                    )?;
                }
            }
            if let Some(c) = &src.compact {
                c.cursor_mode(
                    c.view(&src.k, &src.v, stream),
                    store.seq_ref(stream)?,
                    1,
                    stream,
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
            let cap = dst
                .compact
                .as_ref()
                .map_or(self.capacity, |c| c.layout.ring);
            let compact = dst.compact.is_some();
            for (a, b) in [(&src.k, &mut dst.k), (&src.v, &mut dst.v)] {
                if !compact {
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
                    continue;
                }
                for row in 0..self.active_rows {
                    let pos = crate::kv::ring_slot(self.prefix + row, cap)?;
                    super::copy::columns(
                        stream,
                        a,
                        b,
                        super::copy::Rect {
                            rows: self.heads,
                            width: self.dim,
                            src_pitch: self.rows * self.dim,
                            src_col: row * self.dim,
                            dst_pitch: cap * self.dim,
                            dst_col: pos * self.dim,
                        },
                    )?;
                }
            }
        }
        if let (Some(saved), Some(seq)) = (&self.sequence, &mut store.sequence) {
            stream.memcpy_dtod(saved, seq)?;
        }
        for kv in &store.kv {
            if let Some(c) = &kv.compact {
                c.cursor_mode(
                    c.view(&kv.k, &kv.v, stream),
                    store.seq_ref(stream)?,
                    2,
                    stream,
                )?;
            }
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
