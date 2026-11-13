use crate::{
    backend::{Backend, Event, Logits, Settings},
    config::Config,
    trace::Trace,
};
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;
use std::time::Instant;

pub fn check_generation_capacity(prompt: usize, count: usize, capacity: usize) -> Result<()> {
    ensure!(count > 0, "max-new-tokens must be positive");
    // Prefill emits the first token; graph capture needs one free KV row.
    let required = prompt
        .checked_add((count - 1).max(1))
        .context("prompt and generation required KV capacity overflows usize")?;
    ensure!(
        required <= capacity,
        "prompt and generation require KV capacity {required}, got {capacity}"
    );
    Ok(())
}

fn embedding_input_parts(ids: &[u32]) -> Result<(Vec<u8>, Vec<bool>)> {
    ensure!(!ids.is_empty(), "embedding input is empty");
    Ok((vec![0; ids.len()], vec![true; ids.len()]))
}

fn embedding_workspace(tokens: usize, max_position: u32) -> Result<usize> {
    ensure!(tokens > 0, "embedding input is empty");
    ensure!(
        tokens <= max_position as usize,
        "embedding input {tokens} exceeds model context {max_position}"
    );
    Ok(tokens.max(1024))
}

pub fn embedding_json(values: &[f32]) -> Result<serde_json::Value> {
    ensure!(
        values.len() == 768 && values.iter().all(|value| value.is_finite()),
        "embedding output must be 768 finite values"
    );
    Ok(serde_json::json!({"embedding": values, "dimension": values.len()}))
}

pub fn embed_text(directory: &Path, text: &str, options: crate::RunOptions) -> Result<Vec<f32>> {
    let ids = crate::tokenizer::encode(directory, text)?;
    let spec = crate::model_spec::ModelSpec::read(directory)?;
    let workspace = embedding_workspace(ids.len(), spec.max_position)?;
    let (modality_ids, mask) = embedding_input_parts(&ids)?;
    let settings = Settings {
        tree_builder: crate::backend::TreeBuilder::Waves,
        spec_budget: crate::backend::SpecBudget::Chain,
        spec_graph: false,
        draft_model: None,
        prefix_cache: false,
        kv_format: crate::kv::KvFormat::Bf16,
        runtime_dir: Some(options.runtime_dir),
        device_budget_gib: None,
    };
    let mut model = Model::load(
        directory,
        options.device,
        workspace,
        Some(workspace),
        settings,
    )?;
    let input = crate::vision_input::MultimodalInput {
        input_ids: &ids,
        modality_ids: &modality_ids,
        pixel_values: &[],
        grids: &[],
        audio: &[],
    };
    let values = model.embed(&input, &mask, None)?;
    ensure!(
        values.len() == 768 && values.iter().all(|value| value.is_finite()),
        "embedding output must be 768 finite values"
    );
    Ok(values)
}

pub struct GenerationOutput {
    pub ids: Vec<u32>,
    pub prefill_ms: f64,
    pub capture_ms: f64,
    pub decode_ms: f64,
}

pub fn generate_plain(
    model: &mut Model,
    ids: &[u32],
    count: usize,
    ignore_eos: bool,
    graph: bool,
    trace: Option<&mut Trace>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<GenerationOutput> {
    generate_input(
        model,
        GenerationInput::Text(ids),
        count,
        ignore_eos,
        graph,
        trace,
        on_token,
    )
}

pub fn generate_multimodal(
    model: &mut Model,
    input: &crate::vision_input::MultimodalInput<'_>,
    count: usize,
    ignore_eos: bool,
    graph: bool,
    trace: Option<&mut Trace>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<GenerationOutput> {
    generate_input(
        model,
        GenerationInput::Image(input),
        count,
        ignore_eos,
        graph,
        trace,
        on_token,
    )
}

enum GenerationInput<'a> {
    Text(&'a [u32]),
    Image(&'a crate::vision_input::MultimodalInput<'a>),
}

fn generate_input(
    model: &mut Model,
    input: GenerationInput<'_>,
    count: usize,
    ignore_eos: bool,
    graph: bool,
    mut trace: Option<&mut Trace>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<GenerationOutput> {
    ensure!(count > 0, "max-new-tokens must be positive");
    model.backend.generation_request(count, ignore_eos, graph)?;
    let start = Instant::now();
    let first = match input {
        GenerationInput::Text(ids) => model.prefill(ids, trace.as_deref_mut())?,
        GenerationInput::Image(input) => model.prefill_multimodal(input, trace.as_deref_mut())?,
    };
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut generated = vec![first];
    let mut running = on_token(first);
    let capture_start = Instant::now();
    if running && graph && count > 1 && (ignore_eos || !model.config.eos_token_id.contains(&first))
    {
        model.capture()?;
    }
    let capture_ms = if graph {
        capture_start.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let start = Instant::now();
    while running
        && generated.len() < count
        && (ignore_eos
            || !model
                .config
                .eos_token_id
                .contains(generated.last().context("generated token missing")?))
    {
        let prefix = format!("decode.{}", generated.len() - 1);
        let token = model.decode(graph, trace.as_deref_mut().map(|t| (t, prefix.as_str())))?;
        generated.push(token);
        running = on_token(token);
    }
    Ok(GenerationOutput {
        ids: generated,
        prefill_ms,
        capture_ms,
        decode_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

pub struct Model {
    backend: Backend,
    pub config: Config,
}

impl Model {
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn validate_canvas_reference(&mut self, reference: &Path) -> Result<usize> {
        self.backend.validate_canvas_reference(reference)
    }

    /// Generate one Q38 text request using its checkpoint's MTP component.
    pub fn generate_mtp(
        dir: &Path,
        ids: &[u32],
        options: &crate::spec::Options,
        depth: usize,
        policy: crate::mtp::VerifyPolicy,
        graph: bool,
    ) -> Result<crate::mtp::Output> {
        Backend::generate_mtp(dir, ids, options, depth, policy, graph)
    }

    pub fn load_mtp(
        dir: &Path,
        options: &crate::spec::Options,
        depth: usize,
        graph: bool,
    ) -> Result<Self> {
        let backend = Backend::load_mtp(dir, options, depth, graph)?;
        let config = backend.config().clone();
        Ok(Self { backend, config })
    }
    pub fn generate_mtp_reuse(
        &mut self,
        ids: &[u32],
        options: &crate::spec::Options,
        graph: bool,
        on_tokens: &mut dyn FnMut(&[u32], usize) -> bool,
    ) -> Result<crate::mtp::Output> {
        self.backend.generate_mtp_prepared(
            ids,
            options.count,
            options.ignore_eos,
            crate::mtp::VerifyPolicy::default(),
            graph,
            on_tokens,
        )
    }
    pub fn reset_mtp(&mut self) -> Result<()> {
        self.backend.reset_mtp()
    }

    pub fn prepare_resident_vision(&mut self) -> Result<()> {
        self.backend.prepare_resident_vision()
    }

    pub(crate) fn check_resident_vision_budget(
        dir: &Path,
        options: &crate::spec::Options,
    ) -> Result<()> {
        Backend::check_resident_vision_budget(dir, options)
    }

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
    ) -> Result<Option<Self>> {
        let backend = Backend::calibrate(dir, ordinal, capacity, chunk, settings)?;
        Ok(backend.map(|backend| Self {
            config: backend.config().clone(),
            backend,
        }))
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
    /// Prefill one complete processor-packed image/text prompt after reset.
    pub fn embed(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        mask: &[bool],
        trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>> {
        self.backend.embed(input, mask, trace)
    }

    pub fn prefill_multimodal(
        &mut self,
        input: &crate::vision_input::MultimodalInput<'_>,
        mut trace: Option<&mut Trace>,
    ) -> Result<u32> {
        if let Some(trace) = trace.as_deref_mut() {
            trace.set_device(&self.device_info().name)?;
        }
        self.backend.prefill_multimodal(input, trace)
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
    pub fn graph_decode(&self) -> bool {
        self.backend.graph_decode()
    }
    pub fn mtp_graph_mode(&self) -> bool {
        self.backend.mtp_graph_mode()
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

    pub fn live_device_bytes(&self) -> Result<u64> {
        self.backend.live_device_bytes()
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

/// Model-independent layer skeleton. A mixer replaces the four Llama attention steps; the
/// per-layer-input stage follows the FFN when the spec carries a per-layer input width.
#[derive(Clone, Copy, Debug)]
pub enum LayerStage {
    Common(crate::backend::Operator),
    Mixer(crate::model_spec::Mixer),
    Ffn,
    Hc(HcStage),
    PerLayerInput,
}

/// Manifold-constrained hyper-connection stage around a residual branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcStage {
    Pre,
    Post,
}

/// mHC sublayer instance owning one fn/base/scale weight triple.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcSublayer {
    Attention,
    Ffn,
}

/// Pre-mix an HcPre stage collapses the stream with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcPreSource {
    /// Pre-mix carried across layers: the previous layer's FFN pre; one-hot on
    /// copy 0 at the first layer of a forward.
    Carried,
    /// This layer's attention pre.
    AttentionOwn,
    /// The sublayer's own just-computed pre.
    Own,
}

/// Per-model pre-mix source policy of the two HcPre stages: (attention, ffn).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcPrePolicy {
    pub attention: HcPreSource,
    pub ffn: HcPreSource,
}

/// Pre-mix source policy of an mHC architecture: DeepSeek collapses the
/// attention stage with the carried pre-mix and the FFN stage with the
/// attention's own (model.py:978-980); GLM collapses both stages with the
/// sublayer's own pre (modeling_glm5_next.py:1314-1349), no carried state.
pub fn hc_pre_policy(architecture: crate::model_spec::Architecture) -> anyhow::Result<HcPrePolicy> {
    let policy = match architecture {
        crate::model_spec::Architecture::DeepseekV41 => HcPrePolicy {
            attention: HcPreSource::Carried,
            ffn: HcPreSource::AttentionOwn,
        },
        crate::model_spec::Architecture::Glm5Next => HcPrePolicy {
            attention: HcPreSource::Own,
            ffn: HcPreSource::Own,
        },
        other => anyhow::bail!("architecture {other:?} has no mHC pre-source policy"),
    };
    Ok(policy)
}

/// Resolved execution step of one Hc stage of `LayerProgram::Hc`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HcStagePlan {
    Pre {
        sublayer: HcSublayer,
        source: HcPreSource,
    },
    Post {
        sublayer: HcSublayer,
    },
}

/// Stage-index plan of the frozen nine-stage program
/// `[HcPre, Norm, Mixer, HcPost, HcPre, PostNorm, Ffn, HcPost, NextNorm]`
/// under one architecture's pre-source policy.
pub fn hc_stage_plan(stage: usize, kind: HcStage, policy: HcPrePolicy) -> Result<HcStagePlan> {
    let plan = match (stage, kind) {
        (0, HcStage::Pre) => HcStagePlan::Pre {
            sublayer: HcSublayer::Attention,
            source: policy.attention,
        },
        (3, HcStage::Post) => HcStagePlan::Post {
            sublayer: HcSublayer::Attention,
        },
        (4, HcStage::Pre) => HcStagePlan::Pre {
            sublayer: HcSublayer::Ffn,
            source: policy.ffn,
        },
        (7, HcStage::Post) => HcStagePlan::Post {
            sublayer: HcSublayer::Ffn,
        },
        _ => anyhow::bail!(
            "layer program position {stage} ({kind:?}) is not an mHC stage of the frozen nine-stage program"
        ),
    };
    Ok(plan)
}

/// Stage order of one transformer layer.
#[derive(Clone, Copy, Debug)]
pub enum LayerProgram {
    Standard([LayerStage; 5]),
    Hc([LayerStage; 9]),
    PerLayerInput([LayerStage; 6]),
}

impl std::ops::Deref for LayerProgram {
    type Target = [LayerStage];
    fn deref(&self) -> &[LayerStage] {
        match self {
            Self::Standard(stages) => &stages[..],
            Self::Hc(stages) => &stages[..],
            Self::PerLayerInput(stages) => &stages[..],
        }
    }
}

pub fn layer_program(
    spec: &crate::model_spec::ModelSpec,
    layer: usize,
) -> anyhow::Result<LayerProgram> {
    use crate::backend::Operator as O;
    use anyhow::Context;
    let mixer = spec
        .layers
        .get(layer)
        .context("layer outside model plan")?
        .mixer;
    if spec.per_layer_input.is_some() {
        return Ok(LayerProgram::PerLayerInput([
            LayerStage::Common(O::Norm),
            LayerStage::Mixer(mixer),
            LayerStage::Common(O::PostNorm),
            LayerStage::Ffn,
            LayerStage::PerLayerInput,
            LayerStage::Common(O::NextNorm),
        ]));
    }
    match spec.architecture {
        crate::model_spec::Architecture::Clover => {
            anyhow::bail!("Clover canvas execution is not bound")
        }
        crate::model_spec::Architecture::DeepseekV41
        | crate::model_spec::Architecture::Glm5Next => Ok(LayerProgram::Hc([
            LayerStage::Hc(HcStage::Pre),
            LayerStage::Common(O::Norm),
            LayerStage::Mixer(mixer),
            LayerStage::Hc(HcStage::Post),
            LayerStage::Hc(HcStage::Pre),
            LayerStage::Common(O::PostNorm),
            LayerStage::Ffn,
            LayerStage::Hc(HcStage::Post),
            LayerStage::Common(O::NextNorm),
        ])),
        crate::model_spec::Architecture::Qwen3Vl => {
            bail!("Qwen3-VL runs only as the MiniMax-H3 conditioner")
        }
        crate::model_spec::Architecture::DiffusionGemma => {
            bail!("DiffusionGemma layer program is not bound yet")
        }
        crate::model_spec::Architecture::Llama
        | crate::model_spec::Architecture::Qwen3
        | crate::model_spec::Architecture::Qwen35Text
        | crate::model_spec::Architecture::MiniCpmV47
        | crate::model_spec::Architecture::Flint
        | crate::model_spec::Architecture::LoopVl
        | crate::model_spec::Architecture::Gemma4
        | crate::model_spec::Architecture::EmbeddingGemma2 => Ok(LayerProgram::Standard([
            LayerStage::Common(O::Norm),
            LayerStage::Mixer(mixer),
            LayerStage::Common(O::PostNorm),
            LayerStage::Ffn,
            LayerStage::Common(O::NextNorm),
        ])),
    }
}

pub const LLAMA_MIXER: [crate::backend::Operator; 4] = [
    crate::backend::Operator::Qkv,
    crate::backend::Operator::RopeKv,
    crate::backend::Operator::Attention,
    crate::backend::Operator::Output,
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
    pub fn draft_config(&self) -> Result<&crate::dspark::DraftConfig> {
        self.backend.draft_config()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_input_and_json_have_the_published_shape() -> Result<()> {
        let ids = [11, 22, 33];
        let (modality, mask) = embedding_input_parts(&ids)?;
        assert_eq!(modality, [0, 0, 0]);
        assert_eq!(mask, [true, true, true]);
        let report = embedding_json(&[0.25; 768])?;
        assert_eq!(report["dimension"], 768);
        assert_eq!(
            report["embedding"].as_array().context("embedding")?.len(),
            768
        );
        assert_eq!(report["embedding"][0], 0.25);
        assert!(embedding_json(&[0.0; 767]).is_err());
        assert!(embedding_json(&[f32::NAN; 768]).is_err());
        assert_eq!(embedding_workspace(1, 8192)?, 1024);
        assert_eq!(embedding_workspace(1024, 8192)?, 1024);
        assert_eq!(embedding_workspace(1025, 8192)?, 1025);
        assert_eq!(embedding_workspace(8192, 8192)?, 8192);
        assert!(embedding_workspace(0, 8192).is_err());
        assert!(embedding_workspace(8193, 8192).is_err());
        Ok(())
    }
}
