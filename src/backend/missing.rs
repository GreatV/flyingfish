use crate::{
    backend::{Event, Logits, Settings},
    config::Config,
    trace::Trace,
};
use anyhow::{Result, bail};
use std::path::Path;

pub enum Backend {}
impl Backend {
    pub fn check_ready(&self) -> Result<()> {
        match *self {}
    }
    pub fn device_info(&self) -> &crate::backend::setup::DeviceInfo {
        match *self {}
    }

    pub fn profile_rounds(&mut self, enabled: bool) {
        let _ = enabled;
        match *self {}
    }
    pub fn verification_logits(&self, row: usize) -> Result<Vec<f32>> {
        let _ = row;
        match *self {}
    }
    pub fn decode_linear(&self) -> &'static str {
        match *self {}
    }
    pub fn calibrate(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<()> {
        Self::load(dir, ordinal, capacity, chunk, settings)?;
        Ok(())
    }
    pub fn load(
        _dir: &Path,
        _ordinal: usize,
        _capacity: usize,
        _chunk: Option<usize>,
        _settings: Settings,
    ) -> Result<Self> {
        bail!(
            "no inference backend available: CUDA backend is missing; rebuild with --features cuda"
        )
    }
    pub fn run_step(
        &mut self,
        step: crate::backend::Step<'_>,
        trace: Option<crate::backend::Target<'_>>,
    ) -> Result<()> {
        let _ = (step, trace);
        match *self {}
    }
    pub fn run_layer(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<crate::backend::Target<'_>>,
    ) -> Result<()> {
        let _ = (layer, step, trace);
        match *self {}
    }
    pub fn run_operator(
        &mut self,
        op: crate::backend::Operator,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<crate::backend::Target<'_>>,
    ) -> Result<()> {
        let _ = (op, layer, step, trace);
        match *self {}
    }
    pub fn config(&self) -> &Config {
        match *self {}
    }
    pub fn run_decode_step(&mut self, trace: Option<crate::backend::Target<'_>>) -> Result<()> {
        let _ = trace;
        match *self {}
    }
    pub fn reset(&mut self) -> Result<()> {
        match *self {}
    }
    pub fn prefill(&mut self, ids: &[u32], trace: Option<&mut Trace>) -> Result<u32> {
        let _ = (ids, trace);
        match *self {}
    }
    pub fn prefill_tail(&mut self, ids: &[u32], trace: &mut Trace, keep: usize) -> Result<u32> {
        let _ = (ids, trace, keep);
        match *self {}
    }
    pub fn decode(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<u32> {
        let _ = (graph, trace);
        match *self {}
    }
    pub fn step(&mut self, graph: bool) -> Result<()> {
        let _ = graph;
        match *self {}
    }
    pub fn rewind(&mut self, position: usize, token: u32) -> Result<()> {
        let _ = (position, token);
        match *self {}
    }
    pub fn weight_bytes(&self) -> usize {
        match *self {}
    }
    pub fn decode_weight_bytes(&self) -> usize {
        match *self {}
    }
    pub fn kv_read_bytes(&self, tokens: usize) -> usize {
        let _ = tokens;
        match *self {}
    }
    pub fn capture(&mut self) -> Result<()> {
        match *self {}
    }
    pub fn token(&self) -> Result<u32> {
        match *self {}
    }
    pub fn set_token(&mut self, token: u32) -> Result<()> {
        let _ = token;
        match *self {}
    }
    pub fn decode_logits(&self) -> Result<Vec<f32>> {
        match *self {}
    }
    pub fn check_logits(&self) -> Result<()> {
        match *self {}
    }
    pub fn trace_layer(&mut self, layer: usize) -> Result<()> {
        let _ = layer;
        match *self {}
    }
    pub fn chunk(&self) -> usize {
        match *self {}
    }
    pub fn attention_chunk(&self) -> Option<usize> {
        match *self {}
    }

    pub fn attention_plan(&self) -> Option<crate::backend::setup::AttentionPlan> {
        match *self {}
    }
    pub fn event(&self) -> Result<Event> {
        match *self {}
    }
    pub fn record(&self, event: &Event) -> Result<()> {
        let _ = event;
        match *self {}
    }
    pub fn synchronize(&self) -> Result<()> {
        match *self {}
    }
    pub fn sample_buffer(&self, count: usize) -> Result<Logits> {
        let _ = count;
        match *self {}
    }
    pub fn copy_logits(&self, dst: &mut Logits, step: usize) -> Result<()> {
        let _ = (dst, step);
        match *self {}
    }
    pub fn check_samples(&self, src: &Logits) -> Result<usize> {
        let _ = src;
        match *self {}
    }
    pub fn profiler_start(&self) -> Result<()> {
        match *self {}
    }
    pub fn profiler_stop(&self) -> Result<()> {
        match *self {}
    }
}

impl Backend {
    pub fn compare_tree_builders(&mut self, enabled: bool) -> Result<()> {
        let _ = enabled;
        match *self {}
    }
    pub fn verify_tree(
        &mut self,
        tree: &crate::tree::Tree,
        trace: Option<crate::backend::Target<'_>>,
    ) -> Result<Vec<u32>> {
        let _ = (tree, trace);
        match *self {}
    }
    pub fn tree_metadata(&self) -> Result<crate::backend::TreeMetadata> {
        match *self {}
    }
    pub fn tree_hidden(&self) -> Result<Vec<f32>> {
        match *self {}
    }
    pub fn poison_tree_padding(&mut self, tree: &crate::tree::Tree) -> Result<()> {
        let _ = tree;
        match *self {}
    }
    pub fn commit_tree(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
    ) -> Result<()> {
        let _ = (tree, plan, eos);
        match *self {}
    }
    pub fn commit_tree_fixture(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        predictions: &[u32],
    ) -> Result<()> {
        let _ = (tree, plan, eos, predictions);
        match *self {}
    }
    pub fn tree_kv_path(&self, path: &[usize]) -> Result<Vec<f32>> {
        let _ = path;
        match *self {}
    }
    pub fn tree_injected_hidden(&self) -> Result<Vec<f32>> {
        match *self {}
    }
    pub fn spec_round(
        &mut self,
        limit: usize,
        trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        let _ = (limit, trace, prefix);
        match *self {}
    }
    pub fn set_spec_graph(&mut self, graph: bool) -> Result<()> {
        let _ = graph;
        match *self {}
    }
    pub fn set_spec_eos(&mut self, eos: &[u32]) -> Result<()> {
        let _ = eos;
        match *self {}
    }
    pub fn spec_state_bits(&self, start: usize, rows: usize) -> Result<Vec<u8>> {
        let _ = (start, rows);
        match *self {}
    }
    pub fn inject_hidden(&mut self, hidden: &[f32], start: usize, rows: usize) -> Result<()> {
        let _ = (hidden, start, rows);
        match *self {}
    }
    pub fn draft_proposals(
        &mut self,
        anchor: u32,
        position: usize,
        trace: &mut Trace,
        prefix: &str,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        let _ = (anchor, position, trace, prefix, forced);
        match *self {}
    }
    pub fn draft_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        let _ = (trace, start, rows);
        match *self {}
    }
    pub fn position(&self) -> usize {
        match *self {}
    }
}

impl Backend {
    pub fn kv_snapshot(&self, start: usize, rows: usize) -> Result<Vec<f32>> {
        let _ = (start, rows);
        match *self {}
    }
}

impl Backend {
    pub fn verification_impl(&self) -> &'static str {
        match *self {}
    }
}
