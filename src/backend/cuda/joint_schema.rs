//! Native joint schema head weights for clef-flash: the published BF16
//! tensors of joint_head.safetensors as context views, dims derived from
//! joint_head_config.json. Forward stages compose shared ops.

use crate::backend::gpu::CudaSlice;
use crate::backend::gpu::CudaStream;
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

pub struct RoutingWeights {
    pub query_norm_w: Arc<CudaSlice<bf16>>,
    pub query_norm_b: Arc<CudaSlice<bf16>>,
    pub memory_norm_w: Arc<CudaSlice<bf16>>,
    pub memory_norm_b: Arc<CudaSlice<bf16>>,
    pub attn_in_proj: Arc<CudaSlice<bf16>>,
    pub attn_in_proj_b: Arc<CudaSlice<bf16>>,
    pub attn_out_proj: Arc<CudaSlice<bf16>>,
    pub attn_out_proj_b: Arc<CudaSlice<bf16>>,
    pub ff0_w: Arc<CudaSlice<bf16>>,
    pub ff0_b: Arc<CudaSlice<bf16>>,
    pub ff3_w: Arc<CudaSlice<bf16>>,
    pub ff3_b: Arc<CudaSlice<bf16>>,
    pub ff_norm_w: Arc<CudaSlice<bf16>>,
    pub ff_norm_b: Arc<CudaSlice<bf16>>,
}

pub struct DecoderWeights {
    pub self_in_proj: Arc<CudaSlice<bf16>>,
    pub self_in_proj_b: Arc<CudaSlice<bf16>>,
    pub self_out_proj: Arc<CudaSlice<bf16>>,
    pub self_out_proj_b: Arc<CudaSlice<bf16>>,
    pub cross_in_proj: Arc<CudaSlice<bf16>>,
    pub cross_in_proj_b: Arc<CudaSlice<bf16>>,
    pub cross_out_proj: Arc<CudaSlice<bf16>>,
    pub cross_out_proj_b: Arc<CudaSlice<bf16>>,
    pub norm1_w: Arc<CudaSlice<bf16>>,
    pub norm1_b: Arc<CudaSlice<bf16>>,
    pub norm2_w: Arc<CudaSlice<bf16>>,
    pub norm2_b: Arc<CudaSlice<bf16>>,
    pub norm3_w: Arc<CudaSlice<bf16>>,
    pub norm3_b: Arc<CudaSlice<bf16>>,
    pub linear1_w: Arc<CudaSlice<bf16>>,
    pub linear1_b: Arc<CudaSlice<bf16>>,
    pub linear2_w: Arc<CudaSlice<bf16>>,
    pub linear2_b: Arc<CudaSlice<bf16>>,
}

pub struct JointWeights {
    pub hidden_norm_w: Arc<CudaSlice<bf16>>,
    pub hidden_norm_b: Arc<CudaSlice<bf16>>,
    pub memory_proj: Arc<CudaSlice<bf16>>,
    pub question_proj: Arc<CudaSlice<bf16>>,
    pub option_question_proj: Arc<CudaSlice<bf16>>,
    pub global_proj: Arc<CudaSlice<bf16>>,
    pub option_context_proj: Arc<CudaSlice<bf16>>,
    pub option_lexical_proj: Arc<CudaSlice<bf16>>,
    pub type_embedding: Arc<CudaSlice<bf16>>,
    pub field_norm_w: Arc<CudaSlice<bf16>>,
    pub field_norm_b: Arc<CudaSlice<bf16>>,
    pub option_norm_w: Arc<CudaSlice<bf16>>,
    pub option_norm_b: Arc<CudaSlice<bf16>>,
    pub option_summary_norm_w: Arc<CudaSlice<bf16>>,
    pub option_summary_norm_b: Arc<CudaSlice<bf16>>,
    pub scorer0_w: Arc<CudaSlice<bf16>>,
    pub scorer0_b: Arc<CudaSlice<bf16>>,
    pub scorer3_w: Arc<CudaSlice<bf16>>,
    pub scorer3_b: Arc<CudaSlice<bf16>>,
    pub prior_logit_scale: f32,
    pub joint_logit_scale: f32,
    pub residual_gate: f32,
    pub routing: Vec<RoutingWeights>,
    pub decoder: Vec<DecoderWeights>,
}

/// Uploads one BF16 [rows, cols] tensor from the published file.
pub fn upload2(
    view: &safetensors::SafeTensors,
    name: &str,
    shape: [usize; 2],
    stream: &Arc<CudaStream>,
) -> Result<Arc<CudaSlice<bf16>>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == shape,
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(shape[0] * shape[1])?;
    stream.memcpy_htod(&host, &mut dev)?;
    Ok(Arc::new(dev))
}

/// Uploads one BF16 [n] tensor from the published file.
pub fn upload1(
    view: &safetensors::SafeTensors,
    name: &str,
    n: usize,
    stream: &Arc<CudaStream>,
) -> Result<Arc<CudaSlice<bf16>>> {
    let t = view.tensor(name)?;
    ensure!(
        t.dtype() == safetensors::Dtype::BF16 && t.shape() == [n],
        "joint tensor {name} dtype/shape mismatch: {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let host: Vec<bf16> = t
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut dev = stream.alloc_zeros::<bf16>(n)?;
    stream.memcpy_htod(&host, &mut dev)?;
    Ok(Arc::new(dev))
}

/// Reads one BF16 scalar from the published file.
pub fn scalar(view: &safetensors::SafeTensors, name: &str) -> Result<f32> {
    let t = view.tensor(name)?;
    let d = t.data();
    Ok(bf16::from_bits(u16::from_le_bytes([d[0], d[1]])).to_f32())
}

/// Loads the published joint-head weights as context views on `stream`.
pub fn load_joint_weights(
    model_dir: &std::path::Path,
    stream: &Arc<CudaStream>,
) -> Result<JointWeights> {
    let bytes = std::fs::read(model_dir.join("joint_head.safetensors"))
        .context("open joint_head.safetensors")?;
    let view = safetensors::SafeTensors::deserialize(&bytes)?;
    let routing = (0..2)
        .map(|i| {
            Ok(RoutingWeights {
                query_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.query_norm.weight"),
                    1024,
                    stream,
                )?,
                query_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.query_norm.bias"),
                    1024,
                    stream,
                )?,
                memory_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.memory_norm.weight"),
                    1024,
                    stream,
                )?,
                memory_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.memory_norm.bias"),
                    1024,
                    stream,
                )?,
                attn_in_proj: upload2(
                    &view,
                    &format!("evidence_layers.{i}.attention.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                attn_in_proj_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.attention.in_proj_bias"),
                    3072,
                    stream,
                )?,
                attn_out_proj: upload2(
                    &view,
                    &format!("evidence_layers.{i}.attention.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                attn_out_proj_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.attention.out_proj.bias"),
                    1024,
                    stream,
                )?,
                ff0_w: upload2(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.0.weight"),
                    [4096, 1024],
                    stream,
                )?,
                ff0_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.0.bias"),
                    4096,
                    stream,
                )?,
                ff3_w: upload2(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.3.weight"),
                    [1024, 4096],
                    stream,
                )?,
                ff3_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward.3.bias"),
                    1024,
                    stream,
                )?,
                ff_norm_w: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward_norm.weight"),
                    1024,
                    stream,
                )?,
                ff_norm_b: upload1(
                    &view,
                    &format!("evidence_layers.{i}.feedforward_norm.bias"),
                    1024,
                    stream,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let decoder = (0..4)
        .map(|i| {
            Ok(DecoderWeights {
                self_in_proj: upload2(
                    &view,
                    &format!("layers.{i}.self_attn.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                self_in_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.self_attn.in_proj_bias"),
                    3072,
                    stream,
                )?,
                self_out_proj: upload2(
                    &view,
                    &format!("layers.{i}.self_attn.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                self_out_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.self_attn.out_proj.bias"),
                    1024,
                    stream,
                )?,
                cross_in_proj: upload2(
                    &view,
                    &format!("layers.{i}.multihead_attn.in_proj_weight"),
                    [3072, 1024],
                    stream,
                )?,
                cross_in_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.multihead_attn.in_proj_bias"),
                    3072,
                    stream,
                )?,
                cross_out_proj: upload2(
                    &view,
                    &format!("layers.{i}.multihead_attn.out_proj.weight"),
                    [1024, 1024],
                    stream,
                )?,
                cross_out_proj_b: upload1(
                    &view,
                    &format!("layers.{i}.multihead_attn.out_proj.bias"),
                    1024,
                    stream,
                )?,
                norm1_w: upload1(&view, &format!("layers.{i}.norm1.weight"), 1024, stream)?,
                norm1_b: upload1(&view, &format!("layers.{i}.norm1.bias"), 1024, stream)?,
                norm2_w: upload1(&view, &format!("layers.{i}.norm2.weight"), 1024, stream)?,
                norm2_b: upload1(&view, &format!("layers.{i}.norm2.bias"), 1024, stream)?,
                norm3_w: upload1(&view, &format!("layers.{i}.norm3.weight"), 1024, stream)?,
                norm3_b: upload1(&view, &format!("layers.{i}.norm3.bias"), 1024, stream)?,
                linear1_w: upload2(
                    &view,
                    &format!("layers.{i}.linear1.weight"),
                    [4096, 1024],
                    stream,
                )?,
                linear1_b: upload1(&view, &format!("layers.{i}.linear1.bias"), 4096, stream)?,
                linear2_w: upload2(
                    &view,
                    &format!("layers.{i}.linear2.weight"),
                    [1024, 4096],
                    stream,
                )?,
                linear2_b: upload1(&view, &format!("layers.{i}.linear2.bias"), 1024, stream)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(JointWeights {
        hidden_norm_w: upload1(&view, "hidden_norm.weight", 4096, stream)?,
        hidden_norm_b: upload1(&view, "hidden_norm.bias", 4096, stream)?,
        memory_proj: upload2(&view, "memory_projection.weight", [1024, 4096], stream)?,
        question_proj: upload2(&view, "question_projection.weight", [1024, 4096], stream)?,
        option_question_proj: upload2(
            &view,
            "option_question_projection.weight",
            [1024, 4096],
            stream,
        )?,
        global_proj: upload2(&view, "global_projection.weight", [1024, 4096], stream)?,
        option_context_proj: upload2(
            &view,
            "option_context_projection.weight",
            [1024, 4096],
            stream,
        )?,
        option_lexical_proj: upload2(
            &view,
            "option_lexical_projection.weight",
            [1024, 4096],
            stream,
        )?,
        type_embedding: upload2(&view, "type_embedding.weight", [3, 1024], stream)?,
        field_norm_w: upload1(&view, "field_norm.weight", 1024, stream)?,
        field_norm_b: upload1(&view, "field_norm.bias", 1024, stream)?,
        option_norm_w: upload1(&view, "option_norm.weight", 1024, stream)?,
        option_norm_b: upload1(&view, "option_norm.bias", 1024, stream)?,
        option_summary_norm_w: upload1(&view, "option_summary_norm.weight", 1024, stream)?,
        option_summary_norm_b: upload1(&view, "option_summary_norm.bias", 1024, stream)?,
        scorer0_w: upload2(&view, "residual_scorer.0.weight", [1024, 4096], stream)?,
        scorer0_b: upload1(&view, "residual_scorer.0.bias", 1024, stream)?,
        scorer3_w: upload2(&view, "residual_scorer.3.weight", [1, 1024], stream)?,
        scorer3_b: upload1(&view, "residual_scorer.3.bias", 1, stream)?,
        prior_logit_scale: scalar(&view, "prior_logit_scale")?,
        joint_logit_scale: scalar(&view, "joint_logit_scale")?,
        residual_gate: scalar(&view, "residual_gate")?,
        routing,
        decoder,
    })
}

/// One question of a scored record: type id, question span and option spans
/// over the record token sequence.
pub struct JointQuestion {
    pub type_id: i32,
    pub qspan: [i32; 2],
    pub option_spans: Vec<[i32; 2]>,
}

/// A scored record fixture: token ids and the question/option spans.
pub struct JointFixture {
    pub ids: Vec<u32>,
    pub questions: Vec<JointQuestion>,
}

impl JointFixture {
    /// Parses the fixture JSON: `{"input_ids": [...], "spans": [{"type",
    /// "qspan", "option_spans"}, ...]}`.
    pub fn parse(text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)?;
        let ids = value["input_ids"]
            .as_array()
            .context("input_ids missing")?
            .iter()
            .map(|v| {
                u32::try_from(v.as_i64().context("input id not integer")?)
                    .context("input id outside u32")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut questions = Vec::new();
        for span in value["spans"].as_array().context("spans missing")? {
            let pair = |v: &serde_json::Value| -> Result<[i32; 2]> {
                let a = v.as_array().context("span not a pair")?;
                ensure!(a.len() == 2, "span pair extent");
                Ok([
                    i32::try_from(a[0].as_i64().context("span bound not integer")?)?,
                    i32::try_from(a[1].as_i64().context("span bound not integer")?)?,
                ])
            };
            let option_spans = span["option_spans"]
                .as_array()
                .context("option_spans missing")?
                .iter()
                .map(pair)
                .collect::<Result<Vec<_>>>()?;
            questions.push(JointQuestion {
                type_id: i32::try_from(span["type"].as_i64().context("type missing")?)?,
                qspan: pair(&span["qspan"])?,
                option_spans,
            });
        }
        ensure!(!ids.is_empty() && !questions.is_empty(), "empty fixture");
        for q in &questions {
            ensure!(
                q.type_id >= 0 && (q.type_id as usize) < 3 && !q.option_spans.is_empty(),
                "question type/options out of range"
            );
            for span in [q.qspan].into_iter().chain(q.option_spans.iter().copied()) {
                ensure!(
                    span[0] >= 0 && span[0] < span[1] && (span[1] as usize) <= ids.len(),
                    "span outside the token sequence"
                );
            }
        }
        Ok(Self { ids, questions })
    }

    /// Total option rows across all questions.
    pub fn total_options(&self) -> usize {
        self.questions.iter().map(|q| q.option_spans.len()).sum()
    }
}

/// The head-width, head-count, head-dimension and feed-forward constants the
/// loader's shape checks already pin.
const WIDTH: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const FF: usize = 4096;

/// Per-fixture bindings: the three attention problems and the stage scratch.
pub struct JointPlan {
    flash_routing: crate::backend::cuda::flash::Flash,
    flash_cross: crate::backend::cuda::flash::Flash,
    flash_self: crate::backend::cuda::flash::Flash,
    /// Captured post-backbone hidden rows `[seq, 4096]` (hidden_norm input).
    pub hidden_src: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    memory: CudaSlice<bf16>,
    memory_normed: CudaSlice<bf16>,
    kv_k: CudaSlice<bf16>,
    kv_v: CudaSlice<bf16>,
    k_hm: CudaSlice<bf16>,
    v_hm: CudaSlice<bf16>,
    dummy_q: CudaSlice<bf16>,
    q_proj: CudaSlice<bf16>,
    attn_out: CudaSlice<bf16>,
    proj_out: CudaSlice<bf16>,
    ff_hidden: CudaSlice<bf16>,
    small: CudaSlice<bf16>,
    small2: CudaSlice<bf16>,
    fields: CudaSlice<bf16>,
    routing: [CudaSlice<bf16>; 2],
    oq: CudaSlice<bf16>,
    global_row: CudaSlice<bf16>,
    features: CudaSlice<bf16>,
    scorer_h: CudaSlice<bf16>,
    scorer_out: CudaSlice<bf16>,
    qvecs: CudaSlice<bf16>,
    ctx: CudaSlice<bf16>,
    lex: CudaSlice<bf16>,
    gathered: CudaSlice<bf16>,
    qbegins: CudaSlice<i32>,
    qends: CudaSlice<i32>,
    begins: CudaSlice<i32>,
    ends: CudaSlice<i32>,
    lex_idx: CudaSlice<i32>,
    pub seq: usize,
    pub questions: usize,
    pub options: usize,
    gather_rows: usize,
}

impl JointPlan {
    /// Binds the attention problems and scratch for one fixture geometry.
    pub fn bind(
        stream: &Arc<CudaStream>,
        seq: usize,
        questions: usize,
        options: usize,
        gather_rows: usize,
    ) -> Result<Self> {
        ensure!(
            seq > 0 && questions > 0 && options >= questions && gather_rows > 0,
            "joint plan geometry: seq={seq} questions={questions} options={options} gather_rows={gather_rows}"
        );
        let score = crate::backend::cuda::flash::Score {
            scale: (HEAD_DIM as f32).sqrt().recip(),
            window_left: None,
        };
        let flash = |rows: usize, capacity: usize| {
            crate::backend::cuda::flash::Flash::new_with_score(
                stream.clone(),
                rows,
                HEADS,
                HEADS,
                HEAD_DIM,
                score,
                capacity,
            )
        };
        let alloc = |n: usize| stream.alloc_zeros::<bf16>(n);
        let big = seq * WIDTH;
        Ok(Self {
            flash_routing: flash(options, seq)?,
            flash_cross: flash(questions, seq)?,
            flash_self: flash(questions, questions)?,
            hidden_src: alloc(seq * 4096)?,
            normed: alloc(seq * 4096)?,
            memory: alloc(big)?,
            memory_normed: alloc(big)?,
            kv_k: alloc(big)?,
            kv_v: alloc(big)?,
            k_hm: alloc(HEADS * seq * HEAD_DIM)?,
            v_hm: alloc(HEADS * seq * HEAD_DIM)?,
            dummy_q: alloc(big)?,
            q_proj: alloc(options * WIDTH)?,
            attn_out: alloc(options * WIDTH)?,
            proj_out: alloc(options * WIDTH)?,
            ff_hidden: alloc(options * FF)?,
            small: alloc(options * WIDTH)?,
            small2: alloc(options * WIDTH)?,
            fields: alloc(questions * WIDTH)?,
            routing: [alloc(options * WIDTH)?, alloc(options * WIDTH)?],
            oq: alloc(questions * WIDTH)?,
            global_row: alloc(4096)?,
            features: alloc(options * 4 * WIDTH)?,
            scorer_h: alloc(options * WIDTH)?,
            scorer_out: alloc(options)?,
            qvecs: alloc(questions * 4096)?,
            ctx: alloc(options * 4096)?,
            lex: alloc(options * 4096)?,
            gathered: alloc(gather_rows * 4096)?,
            qbegins: stream.alloc_zeros::<i32>(questions)?,
            qends: stream.alloc_zeros::<i32>(questions)?,
            begins: stream.alloc_zeros::<i32>(options)?,
            ends: stream.alloc_zeros::<i32>(options)?,
            lex_idx: stream.alloc_zeros::<i32>(gather_rows)?,
            seq,
            questions,
            options,
            gather_rows,
        })
    }
}

/// The scored-stage outputs at the frozen cell extents.
pub struct JointStages {
    /// memory_projection row 0.
    pub memory_row0: Vec<f32>,
    /// question_projection row 0.
    pub question_proj_row0: Vec<f32>,
    /// option_question_projection row 0 of the last question's call.
    pub option_question_last_row0: Vec<f32>,
    /// option_context_projection row 0 of the last question's call.
    pub option_context_last_row0: Vec<f32>,
    /// option_lexical_projection row 0 of the last question's call.
    pub option_lexical_last_row0: Vec<f32>,
    /// global_projection row.
    pub global: Vec<f32>,
    /// routing layer outputs `[options * 1024]`.
    pub routing: [Vec<f32>; 2],
    /// decoder layer outputs `[questions * 1024]`.
    pub decoder: [Vec<f32>; 4],
    /// field_norm row 0.
    pub fields_row0: Vec<f32>,
    /// per-question option logits.
    pub logits: Vec<Vec<f32>>,
    /// hidden_norm over the full sequence (disclosed measurement).
    pub hidden_norm_full: Vec<f32>,
}

/// The shared op kernels the head launches directly.
pub(crate) struct JointOps {
    bias_add: cudarc::driver::CudaFunction,
    gelu: cudarc::driver::CudaFunction,
    pack_split: cudarc::driver::CudaFunction,
    gather: cudarc::driver::CudaFunction,
    mean_span: cudarc::driver::CudaFunction,
}

impl JointOps {
    pub(crate) fn new(ctx: &Arc<cudarc::driver::CudaContext>) -> Result<Self> {
        let common = crate::backend::cuda::cubin::module(ctx, "q38_common")?;
        let vision = crate::backend::cuda::cubin::module(ctx, "vision_ops")?;
        Ok(Self {
            bias_add: common.load_function("q38_bias_add")?,
            gelu: common.load_function("q38_gelu")?,
            pack_split: vision.load_function("vision_pack_split")?,
            gather: vision.load_function("vision_row_gather")?,
            mean_span: common.load_function("mean_span_rows")?,
        })
    }
}

/// Span-mean operands: the source rows, the device span bounds and the
/// destination, with the derived widths.
pub struct SpanMean<'a> {
    pub x: &'a CudaSlice<bf16>,
    pub begins: &'a CudaSlice<i32>,
    pub ends: &'a CudaSlice<i32>,
    pub out: &'a mut CudaSlice<bf16>,
    pub spans: usize,
    pub width: usize,
}

// When set, the attention projection stages dump their tensors (defect
// localization only).
#[cfg(test)]
thread_local! {
    static JOINT_MHA_DUMP: std::cell::RefCell<Option<std::path::PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Points the attention-stage dumps at a directory; None disables them.
#[cfg(test)]
pub fn set_joint_mha_dump(dir: Option<std::path::PathBuf>) {
    JOINT_MHA_DUMP.with(|d| *d.borrow_mut() = dir);
}

#[cfg(test)]
fn dump_mha(name: &str, x: &CudaSlice<bf16>, stream: &Arc<CudaStream>) -> Result<()> {
    let Some(dir) = JOINT_MHA_DUMP.with(|d| d.borrow().clone()) else {
        return Ok(());
    };
    let host = stream.clone_dtoh(x)?;
    let bytes: Vec<u8> = host
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    std::fs::write(std::path::Path::new(&dir).join(name), bytes)?;
    Ok(())
}

fn flat(n: usize) -> crate::backend::gpu::LaunchConfig {
    crate::backend::gpu::LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Affine LayerNorm over `[rows, width]` BF16 rows, eps 1e-5.
impl JointRun<'_> {
    fn layer_norm(
        &self,
        x: &CudaSlice<bf16>,
        y: &mut CudaSlice<bf16>,
        rows: usize,
        width: usize,
        weight: &Arc<CudaSlice<bf16>>,
        bias: &Arc<CudaSlice<bf16>>,
    ) -> Result<()> {
        let stream = self.stream;
        let common = self.common;
        use crate::backend::cuda::contracts::{TensorMut, view};
        let span = (rows * width) as u64;
        let xv = view(
            x,
            stream,
            0,
            [1, u32::try_from(rows)?, width as u32],
            [span, width as u64, 1],
        )?;
        let yv = view(
            y,
            stream,
            0,
            [1, u32::try_from(rows)?, width as u32],
            [span, width as u64, 1],
        )?;
        let wv = view(weight, stream, 0, [width as u32], [1])?;
        let bv = view(bias, stream, 0, [width as u32], [1])?;
        common
            .norm
            .layer_norm(stream, xv, wv, bv, &mut TensorMut(yv), 1e-5)
    }

    /// In-place affine LayerNorm; the shared kernel reads each row before it
    /// writes it.
    fn layer_norm_inplace(
        &self,
        buf: &mut CudaSlice<bf16>,
        rows: usize,
        width: usize,
        weight: &Arc<CudaSlice<bf16>>,
        bias: &Arc<CudaSlice<bf16>>,
    ) -> Result<()> {
        let stream = self.stream;
        let common = self.common;
        use crate::backend::cuda::contracts::{Tensor, TensorMut, view};
        let span = (rows * width) as u64;
        let xv = view(
            &*buf,
            stream,
            0,
            [1, u32::try_from(rows)?, width as u32],
            [span, width as u64, 1],
        )?;
        let wv = view(weight, stream, 0, [width as u32], [1])?;
        let bv = view(bias, stream, 0, [width as u32], [1])?;
        let yv = unsafe { Tensor::new(xv.ptr, xv.shape, xv.stride, xv.span_bytes)? };
        common
            .norm
            .layer_norm(stream, xv, wv, bv, &mut TensorMut(yv), 1e-5)
    }

    /// In-place bias add over a flat BF16 buffer: `y[i] += bias[i % width]`.
    fn bias_add(
        &self,
        y: &mut CudaSlice<bf16>,
        bias: &Arc<CudaSlice<bf16>>,
        bias_offset: usize,
        width: usize,
    ) -> Result<()> {
        let stream = self.stream;
        let ops = self.ops;
        use crate::backend::gpu::{DevicePtr, DevicePtrMut, PushKernelArg};
        ensure!(
            y.len().is_multiple_of(width) && bias.len() >= bias_offset + width,
            "bias add extent: y={} width={width}",
            y.len()
        );
        let count = y.len();
        let (y_ptr, _guard) = y.device_ptr_mut(stream);
        let b_ptr = bias.device_ptr(stream).0 + (bias_offset * 2) as u64;
        unsafe {
            stream
                .launch_builder(&ops.bias_add)
                .arg(&y_ptr)
                .arg(&b_ptr)
                .arg(&(count as u64))
                .arg(&(width as u32))
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// Erf GELU over a `[rows, width]` BF16 buffer, F32 compute, BF16 round.
    fn gelu(&self, x: &mut CudaSlice<bf16>, rows: usize, width: usize) -> Result<()> {
        let stream = self.stream;
        let ops = self.ops;
        use crate::backend::gpu::{DevicePtrMut, PushKernelArg};
        let count = rows * width;
        ensure!(x.len() >= count, "gelu extent");
        let stride = width as u64;
        let (x_ptr, _guard) = x.device_ptr_mut(stream);
        unsafe {
            stream
                .launch_builder(&ops.gelu)
                .arg(&x_ptr)
                .arg(&x_ptr)
                .arg(&(rows as u32))
                .arg(&1u32)
                .arg(&(width as u32))
                .arg(&stride)
                .arg(&stride)
                .arg(&1u64)
                .arg(&stride)
                .arg(&stride)
                .arg(&1u64)
                .arg(&0i32)
                .launch(flat(count))?;
        }
        Ok(())
    }

    /// Row gather over `width`-wide BF16 rows: `gathered[r] = src[indices[r]]`.
    fn row_gather(
        &self,
        src: &CudaSlice<bf16>,
        indices: &CudaSlice<i32>,
        gathered: &mut CudaSlice<bf16>,
        rows: usize,
        in_rows: usize,
        width: usize,
    ) -> Result<()> {
        let stream = self.stream;
        let ops = self.ops;
        use crate::backend::gpu::{DevicePtr, DevicePtrMut, PushKernelArg};
        ensure!(
            indices.len() == rows && gathered.len() == rows * width && src.len() == in_rows * width,
            "row gather extent"
        );
        unsafe {
            stream
                .launch_builder(&ops.gather)
                .arg(&src.device_ptr(stream).0)
                .arg(&indices.device_ptr(stream).0)
                .arg(&gathered.device_ptr_mut(stream).0)
                .arg(&(rows as u32))
                .arg(&(in_rows as u32))
                .arg(&(width as u32))
                .launch(flat(rows * width))?;
        }
        Ok(())
    }

    /// Per-span row means over `x` with device bounds `begins`/`ends`.
    fn span_means(
        &self,
        x: &CudaSlice<bf16>,
        begins: &CudaSlice<i32>,
        ends: &CudaSlice<i32>,
        out: &mut CudaSlice<bf16>,
        spans: usize,
        width: usize,
    ) -> Result<()> {
        mean_span_rows(
            self.stream,
            &self.ops.mean_span,
            SpanMean {
                x,
                begins,
                ends,
                out,
                spans,
                width,
            },
        )
    }
}

/// Elementwise BF16 add: F32 compute, one BF16 round per element.
fn add_round(a: &[bf16], b: &[bf16]) -> Result<Vec<bf16>> {
    ensure!(a.len() == b.len(), "add extent");
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32()))
        .collect())
}

/// F32 dot of BF16 vectors.
fn dot_f32(a: &[bf16], b: &[bf16]) -> Result<f32> {
    ensure!(a.len() == b.len(), "dot extent");
    Ok(a.iter().zip(b).map(|(x, y)| x.to_f32() * y.to_f32()).sum())
}

/// L2-normalizes one BF16 row: F32 norm and division, BF16 round per element.
fn normalize_row(row: &[bf16]) -> Vec<bf16> {
    let norm = row
        .iter()
        .map(|v| {
            let f = v.to_f32();
            f * f
        })
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    row.iter()
        .map(|v| bf16::from_f32(v.to_f32() / norm))
        .collect()
}

/// Softmax over BF16 scores: F32 max-subtracted exponential, BF16 weights.
fn softmax_bf16(scores: &[bf16]) -> Vec<bf16> {
    let m = scores
        .iter()
        .map(|v| v.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|v| (v.to_f32() - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| bf16::from_f32(e / sum)).collect()
}

/// The two attention operands of one projection.
struct AttnSides<'a> {
    queries: &'a CudaSlice<bf16>,
    rows: usize,
    keys_values: &'a CudaSlice<bf16>,
    keys: usize,
}

/// The attention projection weights of one routing or decoder sub-layer.
struct AttnWeights {
    in_proj: Arc<CudaSlice<bf16>>,
    in_bias: Arc<CudaSlice<bf16>>,
    out_w: Arc<CudaSlice<bf16>>,
    out_b: Arc<CudaSlice<bf16>>,
}

/// The mutable scratch lanes one attention projection writes.
struct AttnLane<'a> {
    q_proj: &'a mut CudaSlice<bf16>,
    kv_k: &'a mut CudaSlice<bf16>,
    kv_v: &'a mut CudaSlice<bf16>,
    dummy_q: &'a mut CudaSlice<bf16>,
    k_hm: &'a mut CudaSlice<bf16>,
    v_hm: &'a mut CudaSlice<bf16>,
    attn_out: &'a mut CudaSlice<bf16>,
    out: &'a mut CudaSlice<bf16>,
}

/// One bound head execution over the shared ops; the caller owns the plan.
pub struct JointRun<'a> {
    stream: &'a Arc<CudaStream>,
    blas: &'a crate::backend::cuda::blas::Blas,
    common: &'a crate::backend::cuda::ops::CommonOpsRef,
    ops: &'a JointOps,
    w: &'a JointWeights,
}

impl<'a> JointRun<'a> {
    pub(crate) fn new(
        stream: &'a Arc<CudaStream>,
        blas: &'a crate::backend::cuda::blas::Blas,
        common: &'a crate::backend::cuda::ops::CommonOpsRef,
        ops: &'a JointOps,
        w: &'a JointWeights,
    ) -> Self {
        Self {
            stream,
            blas,
            common,
            ops,
            w,
        }
    }
}

impl JointRun<'_> {
    fn read(&self, x: &CudaSlice<bf16>) -> Result<Vec<bf16>> {
        Ok(self.stream.clone_dtoh(x)?)
    }

    fn read_rows(&self, x: &CudaSlice<bf16>, elements: usize) -> Result<Vec<bf16>> {
        ensure!(elements <= x.len(), "read extent");
        Ok(self.stream.clone_dtoh(&x.slice(..elements))?)
    }

    fn upload(&self, host: &[bf16], dst: &mut CudaSlice<bf16>) -> Result<()> {
        ensure!(host.len() <= dst.len(), "upload extent");
        self.stream
            .memcpy_htod(host, &mut dst.slice_mut(..host.len()))?;
        Ok(())
    }

    fn upload_spans(
        &self,
        spans: &[[i32; 2]],
        begins: &mut CudaSlice<i32>,
        ends: &mut CudaSlice<i32>,
    ) -> Result<()> {
        ensure!(
            spans.len() == begins.len() && spans.len() == ends.len(),
            "span buffer extent"
        );
        let b: Vec<i32> = spans.iter().map(|s| s[0]).collect();
        let e: Vec<i32> = spans.iter().map(|s| s[1]).collect();
        self.stream
            .memcpy_htod(&b, &mut begins.slice_mut(..spans.len()))?;
        self.stream
            .memcpy_htod(&e, &mut ends.slice_mut(..spans.len()))?;
        Ok(())
    }

    /// Multi-head attention: in-projection of `queries` `[rows, WIDTH]` and
    /// `keys_values` `[keys, WIDTH]` through the weight row ranges of one
    /// in_proj tensor, the bidirectional D64 kernel, and the output
    /// projection into `lane.out` `[rows, WIDTH]`.
    fn mha(
        &self,
        weights: &AttnWeights,
        lane: AttnLane<'_>,
        sides: AttnSides<'_>,
        flash: &mut crate::backend::cuda::flash::Flash,
    ) -> Result<()> {
        use crate::backend::gpu::{DevicePtr, PushKernelArg};
        let AttnSides {
            queries,
            rows,
            keys_values,
            keys,
        } = sides;
        let AttnLane {
            q_proj,
            kv_k,
            kv_v,
            dummy_q,
            k_hm,
            v_hm,
            attn_out,
            out,
        } = lane;
        self.blas
            .gemm_bf16(&weights.in_proj, 0..WIDTH, queries, q_proj, rows, WIDTH)?;
        self.bias_add(q_proj, &weights.in_bias, 0, WIDTH)?;
        #[cfg(test)]
        dump_mha("mha_qproj.raw", q_proj, self.stream)?;
        self.blas.gemm_bf16(
            &weights.in_proj,
            WIDTH..2 * WIDTH,
            keys_values,
            kv_k,
            keys,
            WIDTH,
        )?;
        self.bias_add(kv_k, &weights.in_bias, WIDTH, WIDTH)?;
        #[cfg(test)]
        dump_mha("mha_kv_k.raw", kv_k, self.stream)?;
        self.blas.gemm_bf16(
            &weights.in_proj,
            2 * WIDTH..3 * WIDTH,
            keys_values,
            kv_v,
            keys,
            WIDTH,
        )?;
        self.bias_add(kv_v, &weights.in_bias, 2 * WIDTH, WIDTH)?;
        unsafe {
            self.stream
                .launch_builder(&self.ops.pack_split)
                .arg(&kv_k.device_ptr(self.stream).0)
                .arg(&kv_k.device_ptr(self.stream).0)
                .arg(&kv_v.device_ptr(self.stream).0)
                .arg(&dummy_q.device_ptr(self.stream).0)
                .arg(&k_hm.device_ptr(self.stream).0)
                .arg(&v_hm.device_ptr(self.stream).0)
                .arg(&(keys as u32))
                .arg(&(keys as u32))
                .arg(&0u32)
                .arg(&0u32)
                .arg(&(keys as u32))
                .arg(&(WIDTH as u32))
                .arg(&(HEADS as u32))
                .arg(&(HEAD_DIM as u32))
                .launch(flat(keys * WIDTH))?;
        }
        #[cfg(test)]
        dump_mha("mha_k_hm.raw", k_hm, self.stream)?;
        flash.run(
            q_proj,
            WIDTH,
            k_hm,
            v_hm,
            attn_out,
            crate::backend::cuda::flash::Shape {
                rows,
                start: 0,
                q_heads: HEADS,
                kv_heads: HEADS,
                capacity: keys,
                head_dim: HEAD_DIM,
                causal: false,
                window_left: None,
            },
        )?;
        #[cfg(test)]
        dump_mha("mha_attnout.raw", attn_out, self.stream)?;
        self.blas
            .gemm_bf16(&weights.out_w, 0..WIDTH, attn_out, out, rows, WIDTH)?;
        self.bias_add(out, &weights.out_b, 0, WIDTH)?;
        Ok(())
    }

    /// Dumps one stage tensor as raw BF16 bits when FF_CLEF_JOINT_DUMP names a
    /// directory.
    #[cfg(test)]
    fn dump_stage(&self, name: &str, x: &CudaSlice<bf16>) -> Result<()> {
        let Ok(dir) = std::env::var("FF_CLEF_JOINT_DUMP") else {
            return Ok(());
        };
        let host = self.stream.clone_dtoh(x)?;
        let path = std::path::Path::new(&dir).join(name);
        let bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// One evidence routing layer: `out = queries + FF(ff_norm(queries +
    /// attention(q_norm(queries), m_norm(memory))))`, reading
    /// `plan.routing[in_slot]` and writing `plan.routing[1 - in_slot]` over
    /// the raw projected memory in `plan.memory`.
    fn routing_layer(&self, plan: &mut JointPlan, layer: usize, in_slot: usize) -> Result<()> {
        let options = plan.options;
        let seq = plan.seq;
        ensure!(in_slot < 2, "routing slot");
        let w = &self.w.routing[layer];
        self.layer_norm(
            &plan.memory,
            &mut plan.memory_normed,
            seq,
            WIDTH,
            &w.memory_norm_w,
            &w.memory_norm_b,
        )?;
        let queries = self.read(&plan.routing[in_slot])?;
        self.layer_norm(
            &plan.routing[in_slot],
            &mut plan.small,
            options,
            WIDTH,
            &w.query_norm_w,
            &w.query_norm_b,
        )?;
        #[cfg(test)]
        {
            self.dump_stage(&format!("routing{layer}_qnorm.raw"), &plan.small)?;
        }
        {
            let weights = self.routing_attn(w);
            let lane = AttnLane {
                q_proj: &mut plan.q_proj,
                kv_k: &mut plan.kv_k,
                kv_v: &mut plan.kv_v,
                dummy_q: &mut plan.dummy_q,
                k_hm: &mut plan.k_hm,
                v_hm: &mut plan.v_hm,
                attn_out: &mut plan.attn_out,
                out: &mut plan.proj_out,
            };
            self.mha(
                &weights,
                lane,
                AttnSides {
                    queries: &plan.small,
                    rows: options,
                    keys_values: &plan.memory_normed,
                    keys: seq,
                },
                &mut plan.flash_routing,
            )?;
        }
        #[cfg(test)]
        {
            self.dump_stage(&format!("routing{layer}_attn.raw"), &plan.proj_out)?;
        }
        let attn = self.read(&plan.proj_out)?;
        let residual = add_round(&queries, &attn)?;
        self.upload(&residual, &mut plan.small)?;
        self.layer_norm_inplace(&mut plan.small, options, WIDTH, &w.ff_norm_w, &w.ff_norm_b)?;
        self.blas.gemm_bf16(
            &w.ff0_w,
            0..FF,
            &plan.small,
            &mut plan.ff_hidden,
            options,
            WIDTH,
        )?;
        self.bias_add(&mut plan.ff_hidden, &w.ff0_b, 0, FF)?;
        self.gelu(&mut plan.ff_hidden, options, FF)?;
        self.blas.gemm_bf16(
            &w.ff3_w,
            0..WIDTH,
            &plan.ff_hidden,
            &mut plan.small2,
            options,
            FF,
        )?;
        self.bias_add(&mut plan.small2, &w.ff3_b, 0, WIDTH)?;
        let ff = self.read(&plan.small2)?;
        let result = add_round(&residual, &ff)?;
        self.upload(&result, &mut plan.routing[1 - in_slot])?;
        Ok(())
    }

    /// One norm-first decoder layer over `plan.fields` `[questions, WIDTH]`,
    /// updating it in place against the raw projected memory in
    /// `plan.memory`.
    fn decoder_layer(&self, plan: &mut JointPlan, layer: usize) -> Result<()> {
        let nq = plan.questions;
        let seq = plan.seq;
        let w = &self.w.decoder[layer];
        ensure!(plan.fields.len() == nq * WIDTH, "decoder extent");
        {
            let weights = self.decoder_attn(w, false);
            let lane = AttnLane {
                q_proj: &mut plan.q_proj,
                kv_k: &mut plan.kv_k,
                kv_v: &mut plan.kv_v,
                dummy_q: &mut plan.dummy_q,
                k_hm: &mut plan.k_hm,
                v_hm: &mut plan.v_hm,
                attn_out: &mut plan.attn_out,
                out: &mut plan.proj_out,
            };
            self.mha(
                &weights,
                lane,
                AttnSides {
                    queries: &plan.fields,
                    rows: nq,
                    keys_values: &plan.fields,
                    keys: nq,
                },
                &mut plan.flash_self,
            )?;
        }
        let attn = self.read_rows(&plan.proj_out, nq * WIDTH)?;
        let input = self.read(&plan.fields)?;
        let t = add_round(&input, &attn)?;
        self.upload(&t, &mut plan.small)?;
        self.layer_norm(
            &plan.small,
            &mut plan.fields,
            nq,
            WIDTH,
            &w.norm1_w,
            &w.norm1_b,
        )?;
        {
            let weights = self.decoder_attn(w, true);
            let lane = AttnLane {
                q_proj: &mut plan.q_proj,
                kv_k: &mut plan.kv_k,
                kv_v: &mut plan.kv_v,
                dummy_q: &mut plan.dummy_q,
                k_hm: &mut plan.k_hm,
                v_hm: &mut plan.v_hm,
                attn_out: &mut plan.attn_out,
                out: &mut plan.proj_out,
            };
            self.mha(
                &weights,
                lane,
                AttnSides {
                    queries: &plan.fields,
                    rows: nq,
                    keys_values: &plan.memory,
                    keys: seq,
                },
                &mut plan.flash_cross,
            )?;
        }
        let attn = self.read_rows(&plan.proj_out, nq * WIDTH)?;
        let input = self.read(&plan.fields)?;
        let t = add_round(&input, &attn)?;
        self.upload(&t, &mut plan.small)?;
        self.layer_norm(
            &plan.small,
            &mut plan.fields,
            nq,
            WIDTH,
            &w.norm2_w,
            &w.norm2_b,
        )?;
        self.blas.gemm_bf16(
            &w.linear1_w,
            0..FF,
            &plan.fields,
            &mut plan.ff_hidden,
            nq,
            WIDTH,
        )?;
        self.bias_add(&mut plan.ff_hidden, &w.linear1_b, 0, FF)?;
        self.gelu(&mut plan.ff_hidden, nq, FF)?;
        self.blas.gemm_bf16(
            &w.linear2_w,
            0..WIDTH,
            &plan.ff_hidden,
            &mut plan.small2,
            nq,
            FF,
        )?;
        self.bias_add(&mut plan.small2, &w.linear2_b, 0, WIDTH)?;
        let ff = self.read_rows(&plan.small2, nq * WIDTH)?;
        let input = self.read(&plan.fields)?;
        let t = add_round(&input, &ff)?;
        self.upload(&t, &mut plan.small)?;
        self.layer_norm(
            &plan.small,
            &mut plan.fields,
            nq,
            WIDTH,
            &w.norm3_w,
            &w.norm3_b,
        )?;
        Ok(())
    }

    fn routing_attn(&self, w: &RoutingWeights) -> AttnWeights {
        AttnWeights {
            in_proj: w.attn_in_proj.clone(),
            in_bias: w.attn_in_proj_b.clone(),
            out_w: w.attn_out_proj.clone(),
            out_b: w.attn_out_proj_b.clone(),
        }
    }

    fn decoder_attn(&self, w: &DecoderWeights, cross: bool) -> AttnWeights {
        AttnWeights {
            in_proj: if cross {
                w.cross_in_proj.clone()
            } else {
                w.self_in_proj.clone()
            },
            in_bias: if cross {
                w.cross_in_proj_b.clone()
            } else {
                w.self_in_proj_b.clone()
            },
            out_w: if cross {
                w.cross_out_proj.clone()
            } else {
                w.self_out_proj.clone()
            },
            out_b: if cross {
                w.cross_out_proj_b.clone()
            } else {
                w.self_out_proj_b.clone()
            },
        }
    }

    /// The staged head forward over the captured hidden rows: every frozen
    /// cell extent plus the disclosed end-to-end stages. `head_rows` is the
    /// output embedding `[vocab, 4096]`.
    pub fn forward(
        &self,
        plan: &mut JointPlan,
        fixture: &JointFixture,
        head_rows: &CudaSlice<bf16>,
        vocab: usize,
    ) -> Result<JointStages> {
        let seq = plan.seq;
        let nq = plan.questions;
        let no = plan.options;
        let w = &self.w;
        self.layer_norm(
            &plan.hidden_src,
            &mut plan.normed,
            seq,
            4096,
            &w.hidden_norm_w,
            &w.hidden_norm_b,
        )?;
        let hidden_norm_full: Vec<f32> = self
            .read(&plan.normed)?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        self.blas.gemm_bf16(
            &w.memory_proj,
            0..WIDTH,
            &plan.normed,
            &mut plan.memory,
            seq,
            4096,
        )?;
        let memory_all = self.read(&plan.memory)?;
        self.stream.memcpy_dtod(
            &plan.normed.slice((seq - 1) * 4096..seq * 4096),
            &mut plan.global_row.slice_mut(..4096),
        )?;
        self.blas.gemm_bf16(
            &w.global_proj,
            0..WIDTH,
            &plan.global_row,
            &mut plan.small,
            1,
            4096,
        )?;
        let global: Vec<bf16> = self.read(&plan.small)?[..WIDTH].to_vec();

        let qspans: Vec<[i32; 2]> = fixture.questions.iter().map(|q| q.qspan).collect();
        let ospans: Vec<[i32; 2]> = fixture
            .questions
            .iter()
            .flat_map(|q| q.option_spans.iter().copied())
            .collect();
        self.upload_spans(&qspans, &mut plan.qbegins, &mut plan.qends)?;
        self.span_means(
            &plan.normed,
            &plan.qbegins,
            &plan.qends,
            &mut plan.qvecs,
            nq,
            4096,
        )?;
        self.upload_spans(&ospans, &mut plan.begins, &mut plan.ends)?;
        self.span_means(
            &plan.normed,
            &plan.begins,
            &plan.ends,
            &mut plan.ctx,
            no,
            4096,
        )?;
        let indices: Vec<i32> = ospans
            .iter()
            .flat_map(|s| {
                fixture.ids[s[0] as usize..s[1] as usize]
                    .iter()
                    .map(|&t| t as i32)
            })
            .collect();
        ensure!(indices.len() == plan.gather_rows, "lexical index extent");
        self.stream.memcpy_htod(&indices, &mut plan.lex_idx)?;
        self.row_gather(
            head_rows,
            &plan.lex_idx,
            &mut plan.gathered,
            plan.gather_rows,
            vocab,
            4096,
        )?;
        let mut bounds: Vec<[i32; 2]> = Vec::with_capacity(no);
        let mut offset = 0i32;
        for s in &ospans {
            bounds.push([offset, offset + (s[1] - s[0])]);
            offset += s[1] - s[0];
        }
        self.upload_spans(&bounds, &mut plan.begins, &mut plan.ends)?;
        self.span_means(
            &plan.gathered,
            &plan.begins,
            &plan.ends,
            &mut plan.lex,
            no,
            4096,
        )?;

        self.blas.gemm_bf16(
            &w.option_context_proj,
            0..WIDTH,
            &plan.ctx,
            &mut plan.small,
            no,
            4096,
        )?;
        let ctx_proj = self.read(&plan.small)?;
        self.blas.gemm_bf16(
            &w.option_lexical_proj,
            0..WIDTH,
            &plan.lex,
            &mut plan.small2,
            no,
            4096,
        )?;
        let lex_proj = self.read(&plan.small2)?;
        self.blas.gemm_bf16(
            &w.option_question_proj,
            0..WIDTH,
            &plan.qvecs,
            &mut plan.oq,
            nq,
            4096,
        )?;
        let oq_proj = self.read(&plan.oq)?;
        self.blas.gemm_bf16(
            &w.question_proj,
            0..WIDTH,
            &plan.qvecs,
            &mut plan.scorer_h,
            nq,
            4096,
        )?;
        let base_fields = self.read(&plan.scorer_h)?;

        let mut option_queries = Vec::with_capacity(no * WIDTH);
        let mut q_base = 0usize;
        for (qi, q) in fixture.questions.iter().enumerate() {
            let count = q.option_spans.len();
            for oi in 0..count {
                let i = q_base + oi;
                let t = add_round(
                    &ctx_proj[i * WIDTH..(i + 1) * WIDTH],
                    &lex_proj[i * WIDTH..(i + 1) * WIDTH],
                )?;
                let row = add_round(&t, &oq_proj[qi * WIDTH..(qi + 1) * WIDTH])?;
                option_queries.extend_from_slice(&row);
            }
            q_base += count;
        }
        self.upload(&option_queries, &mut plan.routing[0])?;

        self.routing_layer(plan, 0, 0)?;
        let routing0: Vec<f32> = self
            .read(&plan.routing[1])?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        self.routing_layer(plan, 1, 1)?;
        let routed = self.read(&plan.routing[0])?;
        let routing1: Vec<f32> = routed.iter().map(|v| v.to_f32()).collect();

        let mut summaries: Vec<bf16> = Vec::with_capacity(nq * WIDTH);
        let mut q_base = 0usize;
        for (qi, q) in fixture.questions.iter().enumerate() {
            let count = q.option_spans.len();
            let field = &base_fields[qi * WIDTH..(qi + 1) * WIDTH];
            let scores: Vec<bf16> = (q_base..q_base + count)
                .map(|i| {
                    let dot = bf16::from_f32(dot_f32(&routed[i * WIDTH..(i + 1) * WIDTH], field)?);
                    Ok(bf16::from_f32(dot.to_f32() / (WIDTH as f32).sqrt()))
                })
                .collect::<Result<Vec<_>>>()?;
            let weights = softmax_bf16(&scores);
            let mut acc = vec![0f32; WIDTH];
            for (wi, w) in weights.iter().enumerate() {
                let row = &routed[(q_base + wi) * WIDTH..(q_base + wi + 1) * WIDTH];
                for c in 0..WIDTH {
                    let product = bf16::from_f32(w.to_f32() * row[c].to_f32());
                    acc[c] += product.to_f32();
                }
            }
            summaries.extend(acc.iter().map(|v| bf16::from_f32(*v)));
            q_base += count;
        }
        self.upload(&summaries, &mut plan.small)?;
        self.layer_norm(
            &plan.small,
            &mut plan.small2,
            nq,
            WIDTH,
            &w.option_summary_norm_w,
            &w.option_summary_norm_b,
        )?;
        let normed_summaries = self.read(&plan.small2)?;
        let type_embedding = self.read(&w.type_embedding)?;
        let mut fields_host: Vec<bf16> = Vec::with_capacity(nq * WIDTH);
        for qi in 0..nq {
            let t = add_round(
                &base_fields[qi * WIDTH..(qi + 1) * WIDTH],
                &normed_summaries[qi * WIDTH..(qi + 1) * WIDTH],
            )?;
            let t = add_round(&t, &global)?;
            let row = add_round(
                &t,
                &type_embedding[fixture.questions[qi].type_id as usize * WIDTH
                    ..(fixture.questions[qi].type_id as usize + 1) * WIDTH],
            )?;
            fields_host.extend_from_slice(&row);
        }
        self.upload(&fields_host, &mut plan.fields)?;

        let mut decoder: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (layer, cell) in decoder.iter_mut().enumerate() {
            self.decoder_layer(plan, layer)?;
            *cell = self
                .read(&plan.fields)?
                .iter()
                .map(|v| v.to_f32())
                .collect();
        }
        self.layer_norm(
            &plan.fields,
            &mut plan.small,
            nq,
            WIDTH,
            &w.field_norm_w,
            &w.field_norm_b,
        )?;
        let fields_all = self.read(&plan.small)?;
        self.layer_norm(
            &plan.routing[0],
            &mut plan.small2,
            no,
            WIDTH,
            &w.option_norm_w,
            &w.option_norm_b,
        )?;
        let options_normed = self.read(&plan.small2)?;

        let qvecs = self.read(&plan.qvecs)?;
        let global_hidden: Vec<bf16> =
            self.read(&plan.normed)?[(seq - 1) * 4096..seq * 4096].to_vec();
        let lexical_all = self.read(&plan.lex)?;

        let mut features: Vec<bf16> = Vec::with_capacity(no * 4 * WIDTH);
        let mut priors: Vec<bf16> = Vec::with_capacity(no);
        let mut cosines: Vec<bf16> = Vec::with_capacity(no);
        let mut q_base = 0usize;
        for (qi, q) in fixture.questions.iter().enumerate() {
            let count = q.option_spans.len();
            let field = &fields_all[qi * WIDTH..(qi + 1) * WIDTH];
            let anchor = normalize_row(&add_round(
                &qvecs[qi * 4096..(qi + 1) * 4096],
                &global_hidden,
            )?);
            let field_norm = field
                .iter()
                .map(|v| {
                    let f = v.to_f32();
                    f * f
                })
                .sum::<f32>()
                .sqrt();
            for oi in 0..count {
                let i = q_base + oi;
                let option = &options_normed[i * WIDTH..(i + 1) * WIDTH];
                let lexical_anchor = normalize_row(&lexical_all[i * 4096..(i + 1) * 4096]);
                priors.push(bf16::from_f32(dot_f32(&lexical_anchor, &anchor)?));
                let option_norm = option
                    .iter()
                    .map(|v| {
                        let f = v.to_f32();
                        f * f
                    })
                    .sum::<f32>()
                    .sqrt();
                cosines.push(bf16::from_f32(
                    dot_f32(field, option)? / (field_norm * option_norm).max(1e-12),
                ));
                features.extend_from_slice(field);
                features.extend_from_slice(option);
                for c in 0..WIDTH {
                    features.push(bf16::from_f32(field[c].to_f32() * option[c].to_f32()));
                }
                for c in 0..WIDTH {
                    features.push(bf16::from_f32(
                        (field[c].to_f32() - option[c].to_f32()).abs(),
                    ));
                }
            }
            q_base += count;
        }
        self.upload(&features, &mut plan.features)?;
        self.blas.gemm_bf16(
            &w.scorer0_w,
            0..WIDTH,
            &plan.features,
            &mut plan.scorer_h,
            no,
            4 * WIDTH,
        )?;
        self.bias_add(&mut plan.scorer_h, &w.scorer0_b, 0, WIDTH)?;
        self.gelu(&mut plan.scorer_h, no, WIDTH)?;
        self.blas.gemm_bf16(
            &w.scorer3_w,
            0..1,
            &plan.scorer_h,
            &mut plan.scorer_out,
            no,
            WIDTH,
        )?;
        self.bias_add(&mut plan.scorer_out, &w.scorer3_b, 0, 1)?;
        let residual = self.read(&plan.scorer_out)?;

        let prior_scale = bf16::from_f32(w.prior_logit_scale.min((100.0f32).ln()).exp()).to_f32();
        let joint_scale = bf16::from_f32(w.joint_logit_scale.min((100.0f32).ln()).exp()).to_f32();
        let gate = bf16::from_f32(1.0 / (1.0 + (-w.residual_gate).exp())).to_f32();
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(nq);
        let mut i = 0usize;
        for q in &fixture.questions {
            let count = q.option_spans.len();
            let mut q_logits = Vec::with_capacity(count);
            for oi in 0..count {
                let prior = bf16::from_f32(prior_scale * priors[i + oi].to_f32());
                let joint = bf16::from_f32(joint_scale * cosines[i + oi].to_f32());
                let joint = bf16::from_f32(joint.to_f32() + residual[i + oi].to_f32());
                let gated = bf16::from_f32(gate * joint.to_f32());
                q_logits.push(bf16::from_f32(prior.to_f32() + gated.to_f32()).to_f32());
            }
            logits.push(q_logits);
            i += count;
        }

        let count_last = fixture.questions[nq - 1].option_spans.len();
        Ok(JointStages {
            memory_row0: memory_all[..WIDTH].iter().map(|v| v.to_f32()).collect(),
            question_proj_row0: base_fields[..WIDTH].iter().map(|v| v.to_f32()).collect(),
            option_question_last_row0: oq_proj[(nq - 1) * WIDTH..nq * WIDTH]
                .iter()
                .map(|v| v.to_f32())
                .collect(),
            option_context_last_row0: ctx_proj
                [(no - count_last) * WIDTH..(no - count_last + 1) * WIDTH]
                .iter()
                .map(|v| v.to_f32())
                .collect(),
            option_lexical_last_row0: lex_proj
                [(no - count_last) * WIDTH..(no - count_last + 1) * WIDTH]
                .iter()
                .map(|v| v.to_f32())
                .collect(),
            global: global.iter().map(|v| v.to_f32()).collect(),
            routing: [routing0, routing1],
            decoder,
            fields_row0: fields_all[..WIDTH].iter().map(|v| v.to_f32()).collect(),
            logits,
            hidden_norm_full,
        })
    }

    /// The hidden_norm stage on caller rows (forced common-input cell).
    pub fn hidden_norm(&self, plan: &mut JointPlan, rows: &[bf16]) -> Result<Vec<f32>> {
        ensure!(rows.len() == plan.seq * 4096, "hidden_norm extent");
        self.upload(rows, &mut plan.hidden_src)?;
        self.layer_norm(
            &plan.hidden_src,
            &mut plan.normed,
            plan.seq,
            4096,
            &self.w.hidden_norm_w,
            &self.w.hidden_norm_b,
        )?;
        Ok(self
            .read(&plan.normed)?
            .iter()
            .map(|v| v.to_f32())
            .collect())
    }

    /// The routing pair on caller-supplied rows (forced common-input cells):
    /// both layers over the raw projected memory.
    pub fn forced_routing(
        &self,
        plan: &mut JointPlan,
        queries: &[bf16],
        memory: &[bf16],
    ) -> Result<[Vec<f32>; 2]> {
        ensure!(
            queries.len() == plan.options * WIDTH && memory.len() == plan.seq * WIDTH,
            "forced routing extent"
        );
        self.upload(memory, &mut plan.memory)?;
        self.upload(queries, &mut plan.routing[0])?;
        self.routing_layer(plan, 0, 0)?;
        let r0: Vec<f32> = self
            .read(&plan.routing[1])?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        self.routing_layer(plan, 1, 1)?;
        let r1: Vec<f32> = self
            .read(&plan.routing[0])?
            .iter()
            .map(|v| v.to_f32())
            .collect();
        Ok([r0, r1])
    }

    /// One decoder layer on caller-supplied rows (forced common-input cells).
    pub fn forced_decoder(
        &self,
        plan: &mut JointPlan,
        layer: usize,
        fields: &[bf16],
        memory: &[bf16],
    ) -> Result<Vec<f32>> {
        ensure!(
            fields.len() == plan.questions * WIDTH && memory.len() == plan.seq * WIDTH,
            "forced decoder extent"
        );
        self.upload(memory, &mut plan.memory)?;
        self.upload(fields, &mut plan.fields)?;
        self.decoder_layer(plan, layer)?;
        Ok(self
            .read(&plan.fields)?
            .iter()
            .map(|v| v.to_f32())
            .collect())
    }
}

/// Per-span row means of a BF16 [rows, width] slice: F32 accumulation over
/// each [begin, end) row range, BF16 result rows. Spans are device i32 pairs.
pub fn mean_span_rows(
    stream: &Arc<CudaStream>,
    kernel: &cudarc::driver::CudaFunction,
    m: SpanMean<'_>,
) -> Result<()> {
    use crate::backend::gpu::{DevicePtr, LaunchConfig, PushKernelArg};
    let SpanMean {
        x,
        begins,
        ends,
        out,
        spans,
        width,
    } = m;
    ensure!(
        x.len() >= begins.len() * width
            && out.len() == spans * width
            && begins.len() == spans
            && ends.len() == spans,
        "mean_span_rows extent mismatch: x={} spans={spans} width={width} out={}",
        x.len(),
        out.len()
    );
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&x.device_ptr(stream).0)
            .arg(&begins.device_ptr(stream).0)
            .arg(&ends.device_ptr(stream).0)
            .arg(&out.device_ptr(stream).0)
            .arg(&(width as i32))
            .launch(LaunchConfig {
                grid_dim: (spans as u32, width.div_ceil(256) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    /// Loads the published joint-head weights and asserts the full census:
    /// every tensor's element count and the three scalar gates.
    #[test]
    #[ignore = "requires coordinator-granted GPU window; FF_CLEF_MODEL"]
    fn joint_weights_load_census() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dir =
            std::path::PathBuf::from(std::env::var("FF_CLEF_MODEL").context("FF_CLEF_MODEL")?);
        let dev = crate::backend::cuda::Device::new(0)?;
        let w = load_joint_weights(&dir, &dev.stream)?;
        let ln1 = |t: &CudaSlice<bf16>| t.len() == 1024;
        let ln4096 = |t: &CudaSlice<bf16>| t.len() == 4096;
        let p2 = |t: &CudaSlice<bf16>| t.len() == 1024 * 4096;
        ensure!(
            ln4096(&w.hidden_norm_w) && ln4096(&w.hidden_norm_b),
            "hidden_norm"
        );
        ensure!(p2(&w.memory_proj), "memory_proj");
        ensure!(
            p2(&w.question_proj) && p2(&w.option_question_proj),
            "q/oq proj"
        );
        ensure!(
            p2(&w.global_proj) && p2(&w.option_context_proj),
            "g/ctx proj"
        );
        ensure!(p2(&w.option_lexical_proj), "lex proj");
        ensure!(w.type_embedding.len() == 3 * 1024, "type_embedding");
        ensure!(ln1(&w.field_norm_w) && ln1(&w.field_norm_b), "field_norm");
        ensure!(
            ln1(&w.option_norm_w) && ln1(&w.option_norm_b),
            "option_norm"
        );
        ensure!(
            ln1(&w.option_summary_norm_w) && ln1(&w.option_summary_norm_b),
            "option_summary_norm"
        );
        ensure!(p2(&w.scorer0_w) && w.scorer0_b.len() == 1024, "scorer0");
        ensure!(
            w.scorer3_w.len() == 1024 && w.scorer3_b.len() == 1,
            "scorer3"
        );
        for r in &w.routing {
            ensure!(
                ln1(&r.query_norm_w) && ln1(&r.query_norm_b),
                "routing qnorm"
            );
            ensure!(
                ln1(&r.memory_norm_w) && ln1(&r.memory_norm_b),
                "routing mnorm"
            );
            ensure!(r.attn_in_proj.len() == 3072 * 1024, "routing in_proj");
            ensure!(r.attn_in_proj_b.len() == 3072, "routing in_proj bias");
            ensure!(r.attn_out_proj.len() == 1024 * 1024, "routing out_proj");
            ensure!(r.attn_out_proj_b.len() == 1024, "routing out bias");
            ensure!(
                r.ff0_w.len() == 4096 * 1024 && r.ff0_b.len() == 4096,
                "routing ff0"
            );
            ensure!(
                r.ff3_w.len() == 1024 * 4096 && r.ff3_b.len() == 1024,
                "routing ff3"
            );
            ensure!(ln1(&r.ff_norm_w) && ln1(&r.ff_norm_b), "routing ffnorm");
        }
        for d in &w.decoder {
            ensure!(d.self_in_proj.len() == 3072 * 1024, "dec self in_proj");
            ensure!(d.self_in_proj_b.len() == 3072, "dec self bias");
            ensure!(d.self_out_proj.len() == 1024 * 1024, "dec self out");
            ensure!(d.self_out_proj_b.len() == 1024, "dec self out bias");
            ensure!(d.cross_in_proj.len() == 3072 * 1024, "dec cross in");
            ensure!(d.cross_in_proj_b.len() == 3072, "dec cross bias");
            ensure!(d.cross_out_proj.len() == 1024 * 1024, "dec cross out");
            ensure!(d.cross_out_proj_b.len() == 1024, "dec cross out bias");
            ensure!(ln1(&d.norm1_w) && ln1(&d.norm1_b), "dec norm1");
            ensure!(ln1(&d.norm2_w) && ln1(&d.norm2_b), "dec norm2");
            ensure!(ln1(&d.norm3_w) && ln1(&d.norm3_b), "dec norm3");
            ensure!(d.linear1_w.len() == 4096 * 1024, "dec lin1");
            ensure!(d.linear1_b.len() == 4096, "dec lin1 bias");
            ensure!(d.linear2_w.len() == 1024 * 4096, "dec lin2");
            ensure!(d.linear2_b.len() == 1024, "dec lin2 bias");
        }
        ensure!(w.routing.len() == 2 && w.decoder.len() == 4, "layer counts");
        println!(
            "JOINT_CENSUS routing={} decoder={} scales p={:.4} j={:.4} g={:.4}",
            w.routing.len(),
            w.decoder.len(),
            w.prior_logit_scale,
            w.joint_logit_scale,
            w.residual_gate
        );
        Ok(())
    }

    /// mean_span_rows against a host reference on synthetic spans.
    #[test]
    #[ignore = "requires coordinator-granted GPU window"]
    fn mean_span_rows_matches_host() -> Result<()> {
        std::env::var("FF_Q38_GPU_WINDOW").context("GPU window not granted")?;
        let dev = crate::backend::cuda::Device::new(0)?;
        let stream = &dev.stream;
        let rows = 12usize;
        let width = 320usize;
        let host: Vec<bf16> = (0..rows * width)
            .map(|i| bf16::from_f32(((i * 37) % 199) as f32 / 199.0))
            .collect();
        let begins = vec![0i32, 5, 11];
        let ends = vec![5i32, 11, 12];
        let spans = begins.len();
        let mut x = stream.alloc_zeros::<bf16>(rows * width)?;
        stream.memcpy_htod(&host, &mut x)?;
        let mut b = stream.alloc_zeros::<i32>(spans)?;
        stream.memcpy_htod(&begins, &mut b)?;
        let mut e = stream.alloc_zeros::<i32>(spans)?;
        stream.memcpy_htod(&ends, &mut e)?;
        let mut out = stream.alloc_zeros::<bf16>(spans * width)?;
        let module = crate::backend::cuda::cubin::module(&dev.ctx, "q38_common")?;
        let kernel = module.load_function("mean_span_rows")?;
        mean_span_rows(
            stream,
            &kernel,
            SpanMean {
                x: &x,
                begins: &b,
                ends: &e,
                out: &mut out,
                spans,
                width,
            },
        )?;
        let got = stream.clone_dtoh(&out)?;
        for (s, (bs, en)) in begins.iter().zip(ends.iter()).enumerate() {
            for c in 0..width {
                let sum: f32 = (*bs as usize..*en as usize)
                    .map(|r| host[r * width + c].to_f32())
                    .sum();
                let want = bf16::from_f32(sum / (en - bs) as f32);
                ensure!(
                    got[s * width + c].to_bits() == want.to_bits(),
                    "span {s} col {c} mismatch"
                );
            }
        }
        println!("MEAN_SPAN_ROWS ok spans={spans} width={width}");
        let _ = (&mut x, &mut b, &mut e);
        Ok(())
    }
}
