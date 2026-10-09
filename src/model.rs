use crate::{
    backend::{Backend, Event, Logits, Settings},
    config::Config,
    trace::Trace,
};
use anyhow::Result;
use std::path::Path;

pub struct Model {
    backend: Backend,
    pub config: Config,
}

impl Model {
    pub fn check_ready(&self) -> Result<()> {
        self.backend.check_ready()
    }
    pub fn device_info(&self) -> &crate::backend::setup::DeviceInfo {
        self.backend.device_info()
    }

    pub fn decode_linear(&self) -> &'static str {
        self.backend.decode_linear()
    }
    pub fn calibrate(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<()> {
        Backend::calibrate(dir, ordinal, capacity, chunk, settings)
    }
    pub fn load(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<Self> {
        let backend = Backend::load(dir, ordinal, capacity, chunk, settings)?;
        let config = backend.config().clone();
        Ok(Self { backend, config })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.backend.reset()
    }
    pub fn prefill(&mut self, ids: &[u32], mut trace: Option<&mut Trace>) -> Result<u32> {
        if let Some(trace) = trace.as_deref_mut() {
            trace.set_device(&self.device_info().name)?;
        }
        self.backend.prefill(ids, trace)
    }
    pub fn prefill_tail(&mut self, ids: &[u32], trace: &mut Trace, keep: usize) -> Result<u32> {
        trace.set_device(&self.device_info().name)?;
        self.backend.prefill_tail(ids, trace, keep)
    }
    pub fn decode(&mut self, graph: bool, mut trace: Option<(&mut Trace, &str)>) -> Result<u32> {
        if let Some((trace, _)) = trace.as_mut() {
            trace.set_device(&self.device_info().name)?;
        }
        self.backend.decode(graph, trace)
    }
    pub fn step(&mut self, graph: bool) -> Result<()> {
        self.backend.step(graph)
    }
    pub fn rewind(&mut self, position: usize, token: u32) -> Result<()> {
        self.backend.rewind(position, token)
    }
    pub fn weight_bytes(&self) -> usize {
        self.backend.weight_bytes()
    }
    pub fn decode_weight_bytes(&self) -> usize {
        self.backend.decode_weight_bytes()
    }
    pub fn kv_read_bytes(&self, tokens: usize) -> usize {
        self.backend.kv_read_bytes(tokens)
    }
    pub fn capture(&mut self) -> Result<()> {
        self.backend.capture()
    }
    pub fn token(&self) -> Result<u32> {
        self.backend.token()
    }
    pub fn set_token(&mut self, token: u32) -> Result<()> {
        self.backend.set_token(token)
    }
    pub fn decode_logits(&self) -> Result<Vec<f32>> {
        self.backend.decode_logits()
    }
    pub fn check_logits(&self) -> Result<()> {
        self.backend.check_logits()
    }
    pub fn verification_logits(&self, row: usize) -> Result<Vec<f32>> {
        self.backend.verification_logits(row)
    }
    pub fn profile_rounds(&mut self, enabled: bool) {
        self.backend.profile_rounds(enabled);
    }
    pub fn trace_layer(&mut self, layer: usize) -> Result<()> {
        self.backend.trace_layer(layer)
    }
    pub fn chunk(&self) -> usize {
        self.backend.chunk()
    }
    pub fn attention_chunk(&self) -> Option<usize> {
        self.backend.attention_chunk()
    }

    pub fn attention_plan(&self) -> Option<crate::backend::setup::AttentionPlan> {
        self.backend.attention_plan()
    }
    pub fn event(&self) -> Result<Event> {
        self.backend.event()
    }
    pub fn record(&self, event: &Event) -> Result<()> {
        self.backend.record(event)
    }
    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }
    pub fn sample_buffer(&self, count: usize) -> Result<Logits> {
        self.backend.sample_buffer(count)
    }
    pub fn copy_logits(&self, dst: &mut Logits, step: usize) -> Result<()> {
        self.backend.copy_logits(dst, step)
    }
    pub fn check_samples(&self, src: &Logits) -> Result<usize> {
        self.backend.check_samples(src)
    }
    pub fn profiler_start(&self) -> Result<()> {
        self.backend.profiler_start()
    }
    pub fn profiler_stop(&self) -> Result<()> {
        self.backend.profiler_stop()
    }
}

pub const LAYER: [crate::backend::Operator; 10] = [
    crate::backend::Operator::Norm,
    crate::backend::Operator::Qkv,
    crate::backend::Operator::RopeKv,
    crate::backend::Operator::Attention,
    crate::backend::Operator::Output,
    crate::backend::Operator::PostNorm,
    crate::backend::Operator::GateUp,
    crate::backend::Operator::Activate,
    crate::backend::Operator::Down,
    crate::backend::Operator::NextNorm,
];

impl Model {
    pub fn compare_tree_builders(&mut self, enabled: bool) -> Result<()> {
        self.backend.compare_tree_builders(enabled)
    }
    pub fn verify_tree(
        &mut self,
        tree: &crate::tree::Tree,
        mut trace: Option<crate::backend::Target<'_>>,
    ) -> Result<Vec<u32>> {
        if let Some(trace) = trace.as_mut() {
            trace.trace.set_device(&self.device_info().name)?;
        }
        self.backend.verify_tree(tree, trace)
    }
    pub fn tree_metadata(&self) -> Result<crate::backend::TreeMetadata> {
        self.backend.tree_metadata()
    }
    pub fn tree_hidden(&self) -> Result<Vec<f32>> {
        self.backend.tree_hidden()
    }
    pub fn poison_tree_padding(&mut self, tree: &crate::tree::Tree) -> Result<()> {
        self.backend.poison_tree_padding(tree)
    }
    pub fn commit_tree(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
    ) -> Result<()> {
        self.backend.commit_tree(tree, plan, eos)
    }
    pub fn commit_tree_fixture(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        predictions: &[u32],
    ) -> Result<()> {
        self.backend
            .commit_tree_fixture(tree, plan, eos, predictions)
    }
    pub fn tree_kv_path(&self, path: &[usize]) -> Result<Vec<f32>> {
        self.backend.tree_kv_path(path)
    }
    pub fn tree_injected_hidden(&self) -> Result<Vec<f32>> {
        self.backend.tree_injected_hidden()
    }
    pub fn spec_round(
        &mut self,
        limit: usize,
        mut trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        if let Some(trace) = trace.as_deref_mut() {
            trace.set_device(&self.device_info().name)?;
        }
        self.backend.spec_round(limit, trace, prefix)
    }
    pub fn set_spec_graph(&mut self, graph: bool) -> Result<()> {
        self.backend.set_spec_graph(graph)
    }
    pub fn set_spec_eos(&mut self, eos: &[u32]) -> Result<()> {
        self.backend.set_spec_eos(eos)
    }
    pub fn spec_state_bits(&self, start: usize, rows: usize) -> Result<Vec<u8>> {
        self.backend.spec_state_bits(start, rows)
    }
    pub fn inject_hidden(&mut self, hidden: &[f32], start: usize, rows: usize) -> Result<()> {
        self.backend.inject_hidden(hidden, start, rows)
    }
    pub fn draft_proposals(
        &mut self,
        anchor: u32,
        position: usize,
        trace: &mut Trace,
        prefix: &str,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        trace.set_device(&self.device_info().name)?;
        self.backend
            .draft_proposals(anchor, position, trace, prefix, forced)
    }
    pub fn draft_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        trace.set_device(&self.device_info().name)?;
        self.backend.draft_kv(trace, start, rows)
    }
    pub fn position(&self) -> usize {
        self.backend.position()
    }
}

impl Model {
    pub fn kv_snapshot(&self, start: usize, rows: usize) -> Result<Vec<f32>> {
        self.backend.kv_snapshot(start, rows)
    }
}

impl Model {
    pub fn verification_impl(&self) -> &'static str {
        self.backend.verification_impl()
    }
}
