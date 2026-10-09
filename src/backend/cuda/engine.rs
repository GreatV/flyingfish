use crate::{
    backend::cuda::weights::Weights,
    backend::cuda::{
        Device,
        blas::Blas,
        decode::Attention,
        flash::{Flash, Shape},
        ops::{Ops, flat, grid},
    },
    backend::{Event, Head, Logits, Settings, SpecBudget, Target, TreeBuilder},
    config::Config,
    trace::Trace,
};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, DevicePtrMut, PushKernelArg, sys};
use half::bf16;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

fn trace_value(
    trace: &mut Option<Target<'_>>,
    stream: &Arc<CudaStream>,
    name: &str,
    data: &CudaSlice<bf16>,
    rows: usize,
    width: usize,
) -> Result<()> {
    if let Some(t) = trace.as_mut() {
        ensure!(t.skip < rows, "trace selection is empty");
        let data = stream.clone_dtoh(&data.slice(t.skip * width..rows * width))?;
        t.trace.add(
            format!("{}.{name}", t.prefix),
            vec![1, rows - t.skip, width],
            data.into_iter().map(bf16::to_f32).collect(),
        )?;
    }
    Ok(())
}

pub(crate) struct Kv {
    pub(crate) k: CudaSlice<bf16>,
    pub(crate) v: CudaSlice<bf16>,
}

pub(crate) struct Work {
    pub(crate) ids: CudaSlice<u32>,
    pub(crate) position: CudaSlice<i32>,
    pub(crate) length: CudaSlice<i32>,
    pub(crate) x: CudaSlice<bf16>,
    pub(crate) n: CudaSlice<bf16>,
    pub(crate) qkv: CudaSlice<bf16>,
    pub(crate) attn: CudaSlice<bf16>,
    pub(crate) out: CudaSlice<bf16>,
    pub(crate) gu: CudaSlice<bf16>,
    pub(crate) act: CudaSlice<bf16>,
    pub(crate) logits: CudaSlice<bf16>,
    pub(crate) token: CudaSlice<u32>,
}

/// DSpark capture destination for one layer slot.
struct Capture<'a> {
    buffer: &'a mut CudaSlice<bf16>,
    slot: usize,
}

impl Work {
    fn grow(&mut self, stream: &Arc<CudaStream>, c: &Config, rows: usize) -> Result<()> {
        ensure!(rows >= self.ids.len(), "workspace rows cannot shrink");
        let ids = stream.alloc_zeros(rows)?;
        let x = stream.alloc_zeros(rows * c.hidden_size)?;
        let n = stream.alloc_zeros(rows * c.hidden_size)?;
        let qkv = stream.alloc_zeros(rows * c.qkv_dim())?;
        let attn = stream.alloc_zeros(rows * c.hidden_size)?;
        let out = stream.alloc_zeros(rows * c.hidden_size)?;
        let gu = stream.alloc_zeros(rows * 2 * c.intermediate_size)?;
        let act = stream.alloc_zeros(rows * c.intermediate_size)?;
        self.ids = ids;
        self.x = x;
        self.n = n;
        self.qkv = qkv;
        self.attn = attn;
        self.out = out;
        self.gu = gu;
        self.act = act;
        Ok(())
    }

    fn add_norm(
        &mut self,
        ops: &Ops,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        rows: usize,
        c: &Config,
    ) -> Result<()> {
        let (res, _res) = self.x.device_ptr_mut(stream);
        unsafe {
            stream
                .launch_builder(&ops.add_norm)
                .arg(&self.out)
                .arg(&res)
                .arg(weight)
                .arg(&mut self.n)
                .arg(&res)
                .arg(&(rows as i32))
                .arg(&(c.hidden_size as i32))
                .arg(&c.rms_norm_eps)
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }

    /// add_rmsnorm plus the DSpark capture mirror of the residual stream it
    /// just wrote: capture[row][slot * hidden + d] = x[row][d]. Replaces the
    /// separate 2D device-to-device column copy for capture layers.
    fn add_norm_capture(
        &mut self,
        ops: &Ops,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        capture: Capture<'_>,
        rows: usize,
        c: &Config,
    ) -> Result<()> {
        let Capture {
            buffer: capture,
            slot,
        } = capture;
        ensure!(
            capture.len() >= rows * 5 * c.hidden_size,
            "DSpark capture buffer is too small for the capture column copy"
        );
        let (res, _res) = self.x.device_ptr_mut(stream);
        let (cap, _cap) = capture.device_ptr_mut(stream);
        unsafe {
            stream
                .launch_builder(&ops.add_norm_capture)
                .arg(&self.out)
                .arg(&res)
                .arg(weight)
                .arg(&mut self.n)
                .arg(&res)
                .arg(&cap)
                .arg(&(rows as i32))
                .arg(&(c.hidden_size as i32))
                .arg(&c.rms_norm_eps)
                .arg(&((5 * c.hidden_size) as i32))
                .arg(&((slot * c.hidden_size) as i32))
                .launch(grid(rows, 256))?;
        }
        Ok(())
    }

    fn advance(&mut self, ops: &Ops, stream: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            stream
                .launch_builder(&ops.advance)
                .arg(&mut self.position)
                .launch(grid(1, 1))?;
            stream
                .launch_builder(&ops.advance)
                .arg(&mut self.length)
                .launch(grid(1, 1))?;
        }
        Ok(())
    }
}

pub(crate) struct Views<'a> {
    pub device: &'a Device,
    pub config: &'a Config,
    pub weights: &'a Weights,
    pub kv: &'a mut [Kv],
    pub work: &'a mut Work,
    pub capacity: usize,
    pub head_stride: usize,
    pub pos_stride: usize,
}

impl Views<'_> {
    fn check(&self, rows: usize) -> Result<()> {
        let c = self.config;
        ensure!(
            self.device.info.warp == 32,
            "CUDA workspace requires warp32"
        );
        ensure!(
            self.head_stride == self.capacity * c.head_dim && self.pos_stride == c.head_dim,
            "KV view strides do not match capacity/head dimension"
        );
        ensure!(
            self.weights.layers.len() == c.num_hidden_layers
                && self.weights.head.len() == c.vocab_size * c.hidden_size,
            "weight view does not match configured layers/head"
        );
        ensure!(
            self.kv.len() == c.num_hidden_layers
                && self
                    .kv
                    .iter()
                    .all(|kv| kv.k.len() == self.capacity * c.kv_dim()
                        && kv.v.len() == self.capacity * c.kv_dim()),
            "KV view does not match configured layers/capacity"
        );
        ensure!(
            rows > 0
                && self.work.ids.len() >= rows
                && self.work.x.len() >= rows * c.hidden_size
                && self.work.n.len() >= rows * c.hidden_size
                && self.work.qkv.len() >= rows * c.qkv_dim()
                && self.work.attn.len() >= rows * c.hidden_size
                && self.work.out.len() >= rows * c.hidden_size
                && self.work.gu.len() >= rows * 2 * c.intermediate_size
                && self.work.act.len() >= rows * c.intermediate_size,
            "workspace views cannot hold {rows} rows"
        );
        Ok(())
    }
}

struct Spec {
    eos: Vec<u32>,
    tree_verify_graph: Option<CudaGraph>,
    tree_commit_graph: Option<CudaGraph>,
    tree_draft_graph: Option<CudaGraph>,
    draft_graph: Option<CudaGraph>,
    verify_graph: Option<CudaGraph>,
    inject_graphs: Vec<Option<CudaGraph>>,
    tree: Option<super::tree::State>,
    tree_attention: Option<super::verification::Verification>,
    active_rows: usize,
    graph: bool,
    verification: Option<super::verification::Verification>,
    draft: super::draft::Draft,
    capture: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    predictions: CudaSlice<u32>,
    row_counts: CudaSlice<i32>,
}

#[derive(Clone, Copy)]
enum SpecPhase {
    Draft,
    Verify,
    Inject(usize),
}

impl SpecPhase {
    fn rows(self) -> usize {
        match self {
            Self::Draft => 7,
            Self::Verify => 8,
            Self::Inject(rows) => rows,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Verify => "verify",
            Self::Inject(_) => "inject",
        }
    }
}

impl Spec {
    fn graph_slot(&mut self, phase: SpecPhase) -> Result<&mut Option<CudaGraph>> {
        Ok(match phase {
            SpecPhase::Draft => &mut self.draft_graph,
            SpecPhase::Verify => &mut self.verify_graph,
            SpecPhase::Inject(rows) => {
                ensure!(rows > 0 && rows <= 8, "inject Graph M must be within1..8");
                &mut self.inject_graphs[rows - 1]
            }
        })
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Graph {
    Decode,
    Draft,
    Verify,
    TreeDraft,
    TreeVerify,
    TreeCommit,
    Inject(usize),
}

pub struct Engine {
    graphs: Vec<Graph>,
    closure: Rc<super::closure::Closure>,
    graph: Option<CudaGraph>,
    poisoned: bool,
    spec_budget: SpecBudget,
    spec_graph: bool,
    tree_builder: TreeBuilder,
    compare_builders: bool,
    profile_rounds: bool,
    runtime: PathBuf,
    spec: Option<Spec>,
    weights: Weights,
    kv: Vec<Kv>,
    work: Work,
    ops: Ops,
    blas: Blas,
    flash: Flash,
    decode: Option<Attention>,
    device: Device,
    config: Config,
    capacity: usize,
    chunk: usize,
    workspace_rows: usize,
    trace_layer: usize,
    position: usize,
}

impl Engine {
    pub fn load(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
    ) -> Result<Self> {
        Self::load_kind(dir, ordinal, capacity, requested_chunk, settings, false)
    }
    pub fn calibrate(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        chunk: Option<usize>,
        settings: Settings,
    ) -> Result<()> {
        Self::load_kind(dir, ordinal, capacity, chunk, settings, true)?;
        Ok(())
    }
    fn load_kind(
        dir: &Path,
        ordinal: usize,
        capacity: usize,
        requested_chunk: Option<usize>,
        settings: Settings,
        write: bool,
    ) -> Result<Self> {
        let started = Instant::now();
        let draft_path = settings.draft_model.clone();
        let quote = |p: &Path| format!("'{}'", p.display().to_string().replace('\'', "'\"'\"'"));
        let runtime_arg = settings
            .runtime_dir
            .as_deref()
            .context("CUDA setup requires --runtime-dir or FLYINGFISH_RUNTIME_DIR")?;
        let mut command = format!(
            "{} --model {} --device {ordinal} --capacity {capacity} --runtime-dir {}",
            quote(&std::env::current_exe()?),
            quote(dir),
            quote(runtime_arg)
        );
        if let Some(path) = &draft_path {
            command.push_str(&format!(
                " --draft-model {} --spec-budget {}",
                quote(path),
                match settings.spec_budget {
                    SpecBudget::Chain => "chain",
                    SpecBudget::Tree16 => "tree16",
                    SpecBudget::Tree32 => "tree32",
                    SpecBudget::Tree64 => "tree64",
                }
            ));
        }
        if let Some(chunk) = requested_chunk {
            command.push_str(&format!(" --chunk {chunk}"));
        }
        command.push_str(" calibrate");
        let closure = Rc::new(super::closure::Closure::new(write, command));
        let config = Config::read(dir)?;
        ensure!(
            capacity > 0
                && capacity <= config.max_position_embeddings
                && capacity < i32::MAX as usize,
            "invalid KV capacity {capacity}"
        );
        ensure!(
            !settings.spec_budget.is_tree() || capacity >= settings.spec_budget.rows(),
            "tree budget {} exceeds KV capacity {capacity}",
            settings.spec_budget.rows()
        );
        let runtime = super::calibrate::runtime_dir(
            settings.runtime_dir.as_deref().context(
                "CUDA attention calibration requires --runtime-dir or FLYINGFISH_RUNTIME_DIR",
            )?,
            dir,
            write,
        )?;
        let device = Device::new(ordinal)?;
        let weights = Weights::load(dir, &config, &device).context("load target weights")?;
        let s = &device.stream;
        let h = config.hidden_size;
        let f = config.intermediate_size;
        let mut kv = Vec::new();
        for _ in 0..config.num_hidden_layers {
            kv.push(Kv {
                k: s.alloc_zeros(capacity * config.kv_dim())?,
                v: s.alloc_zeros(capacity * config.kv_dim())?,
            });
        }
        let ops = Ops::new(&device.ctx).context("load CUDA operator modules")?;
        eprintln!("{}", serde_json::json!({"backend_device":device.info}));
        let blas = Blas::new(
            &device,
            &ops,
            &runtime,
            settings.linear_choices.clone(),
            closure.clone(),
        )?;
        let workspace_rows = settings.spec_budget.rows().min(capacity);
        let chunk = workspace_rows;
        let mut work = Work {
            ids: s.alloc_zeros(workspace_rows)?,
            position: s.alloc_zeros(1)?,
            length: s.alloc_zeros(1)?,
            x: s.alloc_zeros(workspace_rows * h)?,
            n: s.alloc_zeros(workspace_rows * h)?,
            qkv: s.alloc_zeros(workspace_rows * config.qkv_dim())?,
            attn: s.alloc_zeros(workspace_rows * h)?,
            out: s.alloc_zeros(workspace_rows * h)?,
            gu: s.alloc_zeros(workspace_rows * 2 * f)?,
            act: s.alloc_zeros(workspace_rows * f)?,
            logits: s.alloc_zeros(config.vocab_size)?,
            token: s.alloc_zeros(1)?,
        };
        s.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = Flash::new(s.clone(), chunk, config.num_attention_heads)?;
        s.synchronize()?;
        let mut engine = Self {
            graphs: Vec::new(),
            closure,
            poisoned: false,
            spec_budget: settings.spec_budget,
            spec_graph: settings.spec_graph,
            tree_builder: settings.tree_builder,
            compare_builders: false,
            profile_rounds: false,
            runtime,
            spec: None,
            graph: None,
            weights,
            kv,
            work,
            ops,
            blas,
            flash,
            decode: None,
            device,
            config,
            capacity,
            chunk,
            workspace_rows,
            trace_layer: 0,
            position: 0,
        };
        if let Some(path) = settings.draft_model {
            let path = path.canonicalize()?;
            ensure!(
                engine.spec_budget.rows() <= engine.capacity,
                "draft workspace needs {} KV rows but capacity is {}",
                engine.spec_budget.rows(),
                engine.capacity
            );
            let draft = super::draft::Draft::new(
                &path,
                &engine.device,
                &engine.config,
                engine.capacity,
                engine.workspace_rows,
                &mut engine.blas,
            )?;
            engine.prepare(8)?;
            engine
                .blas
                .prepare(8, engine.config.vocab_size, engine.config.hidden_size)?;
            let rows = engine.spec_budget.rows();
            if engine.spec_budget.is_tree() {
                engine.prepare(rows)?;
                engine
                    .blas
                    .prepare(rows, engine.config.vocab_size, engine.config.hidden_size)?;
            }
            let s = &engine.device.stream;
            let count_values: Vec<_> = (1..=rows).map(|n| n as i32).collect();
            let row_counts = s.clone_htod(&count_values)?;
            s.synchronize()?;
            engine.spec = Some(Spec {
                eos: Vec::new(),
                tree_verify_graph: None,
                tree_commit_graph: None,
                tree_draft_graph: None,
                tree: if engine.spec_budget.is_tree() {
                    Some(super::tree::State::new(
                        &engine.device,
                        &engine.config,
                        &engine.kv,
                        engine.capacity,
                        rows,
                    )?)
                } else {
                    None
                },
                tree_attention: None,
                active_rows: 0,
                graph: engine.spec_graph,
                draft_graph: None,
                verify_graph: None,
                inject_graphs: (0..8).map(|_| None).collect(),
                verification: None,
                draft,
                capture: s.alloc_zeros(engine.workspace_rows * 5 * engine.config.hidden_size)?,
                logits: s.alloc_zeros(rows * engine.config.vocab_size)?,
                predictions: s.alloc_zeros(rows)?,
                row_counts,
            });
        }
        engine.size_workspace(requested_chunk)?;
        engine.declare_graphs();
        if engine.spec.is_some() {
            eprintln!(
                "{}",
                serde_json::json!({"spec_execution":{"graph":engine.spec_graph,"budget":engine.spec_budget,"draft_rows":7,"verify_rows":engine.spec_budget.rows(),"workspace_rows":engine.workspace_rows,"inject_rows":"actual committed M1..8","acceptance":"host greedy prefix"}})
            );
        }
        engine.device.stream.synchronize()?;
        let (free, total) = engine.device.ctx.mem_get_info()?;
        let kv_bytes: usize = engine
            .kv
            .iter()
            .map(|kv| (kv.k.len() + kv.v.len()) * size_of::<bf16>())
            .sum();
        eprintln!(
            "weights_bytes={} kv_bytes={} device_used_bytes={} device_total_bytes={}",
            engine.weights.bytes,
            kv_bytes,
            total - free,
            total
        );
        engine.initialize(dir, draft_path.as_deref(), write, started)?;
        Ok(engine)
    }

    fn grow_rows(&mut self, rows: usize, chunk: usize) -> Result<()> {
        let old_rows = self.workspace_rows;
        let old_chunk = self.chunk;
        ensure!(rows >= old_rows, "workspace cannot shrink during load");
        if rows != old_rows {
            self.work.grow(&self.device.stream, &self.config, rows)?;
            if let Some(spec) = &mut self.spec {
                spec.draft.grow(&self.config, rows)?;
                spec.capture = self
                    .device
                    .stream
                    .alloc_zeros(rows * 5 * self.config.hidden_size)?;
            }
        }
        if chunk != old_chunk {
            self.flash = Flash::new(
                self.device.stream.clone(),
                chunk,
                self.config.num_attention_heads,
            )?;
        }
        self.workspace_rows = rows;
        self.chunk = chunk;
        Ok(())
    }

    fn size_workspace(&mut self, requested_chunk: Option<usize>) -> Result<()> {
        self.device.finish_upload()?;
        self.device.stream.synchronize()?;
        let (free, total) = self.device.ctx.mem_get_info()?;
        let plan = crate::prefill::Plan::new(
            &self.config,
            self.capacity,
            requested_chunk,
            free,
            total,
            Flash::tile()?,
            self.spec.is_some(),
        )?;
        let rows = self.spec_budget.workspace_rows(plan.chunk);
        let required = rows
            .checked_mul(plan.row_bytes)
            .context("workspace byte count overflow")?;
        let available = free.saturating_sub(plan.reserve_bytes);
        ensure!(
            rows <= plan.row_limit,
            "workspace {rows} rows exceeds row budget {}: required={required} available={available}",
            plan.row_limit
        );
        eprintln!(
            "{}",
            serde_json::json!({"prefill_memory_plan":plan,"draft_allocated":self.spec.is_some()})
        );
        self.poisoned = true;
        self.grow_rows(rows, plan.chunk).with_context(|| {
            format!("allocate prefill workspace: required={required} available={available}")
        })?;
        self.prepare(plan.chunk)?;
        self.poisoned = false;
        Ok(())
    }

    fn ensure_decode(&mut self) -> Result<()> {
        if self.decode.is_some() {
            return Ok(());
        }
        let calibrated = super::calibrate::load(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            self.capacity,
            &self.closure,
        )
        .context("attention setup calibration")?;
        let setup = crate::backend::setup::Setup::select(
            self.device.info.clone(),
            &self.config,
            self.capacity,
            |_info, capacity| calibrated.choose(capacity),
        )?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"rows":1,"device":setup.device,"selection":setup.selection,"attention":setup.attention,"calibration":calibrated}})
        );
        let decode = Attention::new(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            self.capacity,
            setup.attention,
        )?;
        self.decode = Some(decode);
        Ok(())
    }

    pub(crate) fn views(&mut self) -> Views<'_> {
        Views {
            device: &self.device,
            config: &self.config,
            weights: &self.weights,
            kv: &mut self.kv,
            work: &mut self.work,
            capacity: self.capacity,
            head_stride: self.capacity * self.config.head_dim,
            pos_stride: self.config.head_dim,
        }
    }

    pub fn device_info(&self) -> &crate::backend::setup::DeviceInfo {
        &self.device.info
    }

    fn declare_graphs(&mut self) {
        self.graphs = match &self.spec {
            None => vec![Graph::Decode],
            Some(spec) if !spec.graph => Vec::new(),
            Some(spec) => {
                let mut graphs = if self.spec_budget.is_tree() {
                    vec![Graph::TreeDraft, Graph::TreeVerify, Graph::TreeCommit]
                } else {
                    vec![Graph::Draft, Graph::Verify]
                };
                graphs.extend((1..=spec.inject_graphs.len()).map(Graph::Inject));
                graphs
            }
        };
        eprintln!(
            "{}",
            serde_json::json!({"backend_graph_declaration":self.graphs})
        );
    }

    fn has_graph(&self, graph: Graph) -> bool {
        if matches!(graph, Graph::Decode) {
            return self.graph.is_some();
        }
        let Some(spec) = &self.spec else {
            return false;
        };
        match graph {
            Graph::Decode => self.graph.is_some(),
            Graph::Draft => spec.draft_graph.is_some(),
            Graph::Verify => spec.verify_graph.is_some(),
            Graph::TreeDraft => spec.tree_draft_graph.is_some(),
            Graph::TreeVerify => spec.tree_verify_graph.is_some(),
            Graph::TreeCommit => spec.tree_commit_graph.is_some(),
            Graph::Inject(rows) => rows
                .checked_sub(1)
                .and_then(|r| spec.inject_graphs.get(r))
                .is_some_and(Option::is_some),
        }
    }
    fn check_graphs(&self) -> Result<()> {
        for &graph in &self.graphs {
            ensure!(
                self.has_graph(graph),
                "production Graph is missing: {graph:?}"
            );
        }
        Ok(())
    }
    pub fn check_ready(&self) -> Result<()> {
        self.closure.assert_ready()?;
        self.check_graphs()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn prepare(&mut self, rows: usize) -> Result<()> {
        let c = &self.config;
        for (o, i) in [
            (c.qkv_dim(), c.hidden_size),
            (c.hidden_size, c.hidden_size),
            (2 * c.intermediate_size, c.hidden_size),
            (c.hidden_size, c.intermediate_size),
        ] {
            self.blas.prepare(rows, o, i)?;
        }
        Ok(())
    }

    fn snapshot(
        &self,
        trace: &mut Trace,
        name: String,
        data: &CudaSlice<bf16>,
        skip: usize,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let values = self
            .device
            .stream
            .clone_dtoh(&data.slice(skip * width..rows * width))?;
        trace.add(
            name,
            vec![1, rows - skip, width],
            values.into_iter().map(bf16::to_f32).collect(),
        )
    }

    fn forward(
        &mut self,
        rows: usize,
        use_token: bool,
        head: bool,
        trace: Option<Target<'_>>,
    ) -> Result<()> {
        let step = crate::backend::Step {
            rows,
            input: if use_token {
                crate::backend::Input::Token
            } else {
                crate::backend::Input::Ids
            },
            head: if head { Head::Last } else { Head::None },
            mask: crate::backend::Mask::Causal,
        };
        self.run_step(step, trace)
    }

    pub fn run_step(
        &mut self,
        step: crate::backend::Step<'_>,
        mut trace: Option<Target<'_>>,
    ) -> Result<()> {
        step.check(self.workspace_rows)?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            self.prepare_tree(step)?;
        }
        if step.input == crate::backend::Input::Token {
            self.ensure_decode()?;
        } else if step.head == Head::All && matches!(step.mask, crate::backend::Mask::Causal) {
            ensure!(
                step.rows <= 8,
                "chain verification supports at most eight rows"
            );
            self.ensure_verification()?;
        }
        self.run_operator(crate::backend::Operator::Embed, 0, step, &mut trace)?;
        for i in 0..self.config.num_hidden_layers {
            self.run_layer(i, step, &mut trace)?;
        }
        self.run_operator(crate::backend::Operator::Head, 0, step, &mut trace)
    }

    /// Host-side validation for the tree step. The metadata kernel is no longer
    /// launched here: prepare_embed_norm (the Embed operator, which this always
    /// precedes) writes ids/positions/slots/snapshot in the same launch as the
    /// embedding gather and the layer-0 norm.
    fn prepare_tree(&mut self, step: crate::backend::Step<'_>) -> Result<()> {
        self.spec_budget
            .check_tree(step, self.position, self.capacity)?;
        let spec = self
            .spec
            .as_ref()
            .context("tree verification requires enabled draft state")?;
        ensure!(
            spec.predictions.len() >= step.rows,
            "tree predictions exceed allocated workspace"
        );
        let tree = spec.tree.as_ref().context("tree State is not enabled")?;
        ensure!(
            tree.rows() == step.rows && tree.budget() == self.spec_budget.rows(),
            "tree metadata rows/budget do not match forward"
        );
        self.views().check(step.rows)?;
        self.ensure_tree_attention()?;
        Ok(())
    }

    fn run_attention(&mut self, layer: usize, step: crate::backend::Step<'_>) -> Result<()> {
        if matches!(step.mask, crate::backend::Mask::Tree { .. }) {
            let spec = self
                .spec
                .as_mut()
                .context("tree attention requires draft State")?;
            let tree = spec.tree.as_ref().context("tree State is not enabled")?;
            let kv = &self.kv[layer];
            return spec
                .tree_attention
                .as_mut()
                .context("tree attention is not calibrated")?
                .run_tree(
                    super::verification::Query {
                        qkv: &self.work.qkv,
                        k: &kv.k,
                        v: &kv.v,
                        out: &mut self.work.attn,
                        prefix: tree.prefix(),
                        shape: Shape {
                            rows: tree.budget(),
                            start: self.position,
                            q_heads: self.config.num_attention_heads,
                            kv_heads: self.config.num_key_value_heads,
                            capacity: self.capacity,
                            causal: false,
                        },
                    },
                    tree.ancestors(),
                );
        }
        if step.input == crate::backend::Input::Token {
            self.ensure_decode()?;
        }
        let w = &mut self.work;
        let kv = &self.kv[layer];
        let c = &self.config;
        if step.input == crate::backend::Input::Token {
            self.decode
                .as_mut()
                .context("decode attention has not been initialized")?
                .run(
                    &self.device.stream,
                    &w.qkv,
                    &kv.k,
                    &kv.v,
                    &mut w.attn,
                    &w.length,
                )
        } else {
            let shape = Shape {
                rows: step.rows,
                start: self.position,
                q_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                capacity: self.capacity,
                causal: true,
            };
            if step.head == Head::All {
                self.spec
                    .as_mut()
                    .context("verification attention requires draft state")?
                    .verification
                    .as_mut()
                    .context("verification attention has not been initialized")?
                    .run(&w.qkv, &kv.k, &kv.v, &mut w.attn, &w.position, shape)
            } else {
                self.flash.run(&w.qkv, &kv.k, &kv.v, &mut w.attn, shape)
            }
        }
    }

    pub fn verification_impl(&self) -> &'static str {
        if self.spec_budget.is_tree() {
            return self
                .spec
                .as_ref()
                .and_then(|s| s.tree_attention.as_ref())
                .map(|s| s.implementation())
                .unwrap_or("tree attention has not been initialized");
        }
        self.spec
            .as_ref()
            .and_then(|s| s.verification.as_ref())
            .map(|s| s.implementation())
            .unwrap_or("verification has not been initialized")
    }
    pub fn profile_rounds(&mut self, enabled: bool) {
        self.profile_rounds = enabled;
    }

    pub fn verification_logits(&self, row: usize) -> Result<Vec<f32>> {
        let spec = self.spec.as_ref().context("draft state is not enabled")?;
        ensure!(
            row < spec.active_rows,
            "verification logit row {row} exceeds active rows {}",
            spec.active_rows
        );
        let n = self.config.vocab_size;
        Ok(self
            .device
            .stream
            .clone_dtoh(&spec.logits.slice(row * n..(row + 1) * n))?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    pub fn run_layer(
        &mut self,
        layer: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        step.check(self.workspace_rows)?;
        ensure!(
            layer < self.config.num_hidden_layers,
            "backend layer {layer} is outside model"
        );
        for op in crate::model::LAYER {
            self.run_operator(op, layer, step, trace)?;
        }
        Ok(())
    }

    pub fn run_decode_step(&mut self, trace: Option<Target<'_>>) -> Result<()> {
        self.ensure_decode()?;
        self.forward(1, true, true, trace)?;
        self.work.advance(&self.ops, &self.device.stream)
    }

    pub fn run_operator(
        &mut self,
        op: crate::backend::Operator,
        i: usize,
        step: crate::backend::Step<'_>,
        trace: &mut Option<Target<'_>>,
    ) -> Result<()> {
        step.check(self.workspace_rows)?;
        ensure!(
            i < self.config.num_hidden_layers,
            "backend operator layer {i} is outside model"
        );
        let is_tree = matches!(step.mask, crate::backend::Mask::Tree { .. });
        if is_tree && matches!(op, crate::backend::Operator::RopeKv) {
            let tree = self
                .spec
                .as_ref()
                .context("tree operator requires draft State")?
                .tree
                .as_ref()
                .context("tree State is not enabled")?;
            let kv = &mut self.kv[i];
            return tree.rope_kv(&mut self.work.qkv, &mut kv.k, &mut kv.v);
        }
        let rows = if is_tree {
            self.spec_budget.rows()
        } else {
            step.rows
        };
        let trace_rows = step.rows;
        let use_token = matches!(step.input, crate::backend::Input::Token);
        let head = step.head == Head::Last;
        let c = &self.config;
        let w = &mut self.work;
        let s = &self.device.stream;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let nr = rows as i32;
        let hd = h as i32;
        let fd = f as i32;
        let qh = c.num_attention_heads as i32;
        let kh = c.num_key_value_heads as i32;
        let dim = c.head_dim as i32;
        let cap = self.capacity as i32;
        let layer = &self.weights.layers[i];
        match op {
            crate::backend::Operator::Embed => {
                // Fused gather + layer-0 RMSNorm: the Embed operator is always
                // the step's first call and always immediately precedes layer 0,
                // whose first operator is Norm, so the two launches belong in
                // one. On the tree path prepare() joins the same launch.
                if is_tree {
                    let tree = self
                        .spec
                        .as_ref()
                        .context("tree embed requires draft State")?
                        .tree
                        .as_ref()
                        .context("tree State is not enabled")?;
                    tree.prepare_embed_norm(
                        &w.position,
                        &self.weights.embed,
                        &layer.input_norm,
                        super::tree::EmbedOut {
                            ids: &mut w.ids,
                            x: &mut w.x,
                            y: &mut w.n,
                        },
                        c,
                    )?;
                } else {
                    let ids = if use_token { &w.token } else { &w.ids };
                    unsafe {
                        s.launch_builder(&self.ops.embed_rmsnorm)
                            .arg(&self.weights.embed)
                            .arg(ids)
                            .arg(&mut w.x)
                            .arg(&layer.input_norm)
                            .arg(&mut w.n)
                            .arg(&nr)
                            .arg(&hd)
                            .arg(&c.rms_norm_eps)
                            .launch(grid(rows, 256))?;
                    }
                }
                trace_value(trace, s, "embed", &w.x, trace_rows, h)?;
            }
            crate::backend::Operator::Norm => {
                // The layer-0 norm is folded into Embed above; for every other
                // layer the Norm operator has no work.
            }
            crate::backend::Operator::Qkv => {
                self.blas
                    .linear(
                        &layer.qkv,
                        &w.n.slice(..rows * h),
                        &mut w.qkv.slice_mut(..rows * c.qkv_dim()),
                        rows,
                        c.qkv_dim(),
                        h,
                    )
                    .with_context(|| format!("layer {i} qkv"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.norm1", &w.n, trace_rows, h)?;
                    trace_value(trace, s, "debug.qkv", &w.qkv, trace_rows, c.qkv_dim())?;
                }
            }
            crate::backend::Operator::RopeKv => unsafe {
                // Fused rotation + KV write: one launch over rows x 20 heads x
                // (head_dim/2) element pairs, replacing (rope, kv_write).
                let kv = &mut self.kv[i];
                s.launch_builder(&self.ops.rope_kv)
                    .arg(&mut w.qkv)
                    .arg(&mut kv.k)
                    .arg(&mut kv.v)
                    .arg(&w.position)
                    .arg(&nr)
                    .arg(&qh)
                    .arg(&kh)
                    .arg(&dim)
                    .arg(&c.rope_theta)
                    .arg(&cap)
                    .launch(flat(
                        rows * (c.num_attention_heads + 2 * c.num_key_value_heads)
                            * (c.head_dim / 2),
                    ))?;
            },
            crate::backend::Operator::Attention => {
                self.run_attention(i, step)?;
            }
            crate::backend::Operator::Output => {
                self.blas
                    .linear(
                        &layer.o,
                        &w.attn.slice(..rows * h),
                        &mut w.out.slice_mut(..rows * h),
                        rows,
                        h,
                        h,
                    )
                    .with_context(|| format!("layer {i} output projection"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.rope", &w.qkv, trace_rows, c.qkv_dim())?;
                    trace_value(trace, s, "debug.attention", &w.attn, trace_rows, h)?;
                    trace_value(trace, s, "debug.o", &w.out, trace_rows, h)?;
                }
            }
            crate::backend::Operator::PostNorm => {
                w.add_norm(&self.ops, s, &layer.post_norm, rows, c)?;
            }
            crate::backend::Operator::GateUp => {
                self.blas
                    .linear(
                        &layer.gu,
                        &w.n.slice(..rows * h),
                        &mut w.gu.slice_mut(..rows * 2 * f),
                        rows,
                        2 * f,
                        h,
                    )
                    .with_context(|| format!("layer {i} gate/up"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.norm2", &w.n, trace_rows, h)?;
                    trace_value(trace, s, "debug.gu", &w.gu, trace_rows, 2 * f)?;
                }
            }
            crate::backend::Operator::Activate => unsafe {
                s.launch_builder(&self.ops.act)
                    .arg(&w.gu)
                    .arg(&mut w.act)
                    .arg(&nr)
                    .arg(&fd)
                    .launch(flat(rows * f))?;
            },
            crate::backend::Operator::Down => {
                self.blas
                    .linear(
                        &layer.down,
                        &w.act.slice(..rows * f),
                        &mut w.out.slice_mut(..rows * h),
                        rows,
                        h,
                        f,
                    )
                    .with_context(|| format!("layer {i} down"))?;
                if i == self.trace_layer {
                    trace_value(trace, s, "debug.act", &w.act, trace_rows, f)?;
                    trace_value(trace, s, "debug.down", &w.out, trace_rows, h)?;
                }
            }
            crate::backend::Operator::NextNorm => {
                let weight = if let Some(next) = self.weights.layers.get(i + 1) {
                    &next.input_norm
                } else {
                    &self.weights.norm
                };
                if let Some(spec) = self.spec.as_mut()
                    && let Some(slot) =
                        crate::dspark::capture_slot(&spec.draft.config.target_layer_ids, i)
                {
                    // Capture layer: the residual stream this writes is the
                    // capture source, so the copy is folded into the norm.
                    w.add_norm_capture(
                        &self.ops,
                        s,
                        weight,
                        Capture {
                            buffer: &mut spec.capture,
                            slot,
                        },
                        rows,
                        c,
                    )?;
                } else {
                    w.add_norm(&self.ops, s, weight, rows, c)?;
                }
                trace_value(trace, s, &format!("layer.{i}"), &w.x, trace_rows, h)?;
            }
            crate::backend::Operator::Head => {
                if step.head == Head::All {
                    let spec = self
                        .spec
                        .as_mut()
                        .context("all-row logits require enabled DSpark state")?;
                    ensure!(
                        rows <= spec.predictions.len(),
                        "verification rows exceed allocated predictions"
                    );
                    self.blas.linear(
                        &self.weights.head,
                        &w.n.slice(..rows * h),
                        &mut spec.logits.slice_mut(..rows * c.vocab_size),
                        rows,
                        c.vocab_size,
                        h,
                    )?;
                    let valid_rows = if is_tree {
                        spec.tree
                            .as_ref()
                            .context("tree State is not enabled")?
                            .valid_rows()
                            .slice(..1)
                    } else {
                        spec.row_counts.slice(rows - 1..rows)
                    };
                    unsafe {
                        s.launch_builder(&self.ops.argmax_rows)
                            .arg(&spec.logits)
                            .arg(&mut spec.predictions)
                            .arg(&(c.vocab_size as i32))
                            .arg(&valid_rows)
                            .launch(grid(rows, 1024))?;
                    }
                    spec.active_rows = step.rows;
                    return Ok(());
                }
                if !head && trace.is_none() {
                    return Ok(());
                }
                if let Some(t) = trace.as_mut() {
                    self.snapshot(
                        t.trace,
                        format!("{}.norm", t.prefix),
                        &self.work.n,
                        t.skip,
                        rows,
                        h,
                    )?;
                }
                if !head {
                    return Ok(());
                }
                let w = &mut self.work;
                let input = w.n.slice((rows - 1) * h..rows * h);
                self.blas
                    .linear(
                        &self.weights.head,
                        &input,
                        &mut w.logits.slice_mut(..c.vocab_size),
                        1,
                        c.vocab_size,
                        h,
                    )
                    .context("lm_head")?;
                let logits = w.logits.slice(..c.vocab_size);
                let vocab = c.vocab_size as i32;
                unsafe {
                    s.launch_builder(&self.ops.argmax)
                        .arg(&logits)
                        .arg(&mut w.token)
                        .arg(&vocab)
                        .launch(grid(1, 1024))?;
                }
                if let Some(t) = trace.as_mut() {
                    self.snapshot(
                        t.trace,
                        format!("{}.logits", t.prefix),
                        &self.work.logits,
                        0,
                        1,
                        c.vocab_size,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.device.stream.synchronize()?;
        self.poisoned = false;
        if let Some(spec) = self.spec.as_mut() {
            spec.active_rows = 0;
        }
        self.position = 0;
        self.device
            .stream
            .memcpy_htod(&[0i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[1i32], &mut self.work.length)?;
        Ok(())
    }

    pub fn prefill(&mut self, ids: &[u32], trace: Option<&mut Trace>) -> Result<u32> {
        self.prefill_inner(ids, trace, ids.len())
    }

    pub fn prefill_tail(&mut self, ids: &[u32], trace: &mut Trace, keep: usize) -> Result<u32> {
        ensure!(
            keep > 0 && keep <= ids.len(),
            "invalid hidden trace tail length {keep}"
        );
        self.prefill_inner(ids, Some(trace), keep)
    }

    fn prefill_inner(
        &mut self,
        ids: &[u32],
        mut trace: Option<&mut Trace>,
        keep: usize,
    ) -> Result<u32> {
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(!ids.is_empty(), "empty input token ids");
        ensure!(
            self.position + ids.len() <= self.capacity,
            "prefill exceeds KV capacity"
        );
        ensure!(
            ids.iter().all(|&x| (x as usize) < self.config.vocab_size),
            "input token id outside vocabulary"
        );
        let trace_start = self.position + ids.len() - keep;
        let end = self.position + ids.len();
        let mut part = 0usize;
        for chunk in ids.chunks(self.chunk) {
            let skip = trace_start.saturating_sub(self.position).min(chunk.len());
            let selected = trace.is_some() && skip < chunk.len();
            self.prepare(chunk.len())?;
            self.device
                .stream
                .memcpy_htod(chunk, &mut self.work.ids.slice_mut(..chunk.len()))?;
            self.device
                .stream
                .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
            let prefix = format!("prefill.{part}");
            self.forward(
                chunk.len(),
                false,
                self.position + chunk.len() == end,
                if selected {
                    trace.as_deref_mut().map(|t| Target {
                        trace: t,
                        prefix: prefix.as_str(),
                        skip,
                    })
                } else {
                    None
                },
            )?;
            if selected {
                part += 1;
            }
            if self.spec.is_some() {
                self.inject_capture(chunk.len(), self.position, None)?;
            }
            self.position += chunk.len();
        }
        self.device
            .stream
            .memcpy_htod(&[self.position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(self.position + 1) as i32], &mut self.work.length)?;
        self.token()
    }

    fn advance(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<()> {
        ensure!(
            !graph || self.spec.is_none(),
            "graph decode is not declared for draft-enabled engines"
        );
        self.closure.assert_ready()?;
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(self.position < self.capacity, "decode exceeds KV capacity");
        ensure!(
            !graph || trace.is_none(),
            "graph tracing is unsupported; use eager tracing"
        );
        if graph {
            self.graph
                .as_ref()
                .context("decode graph is not captured")?
                .launch()?;
        } else {
            self.run_decode_step(trace.map(|(trace, prefix)| Target {
                trace,
                prefix,
                skip: 0,
            }))?;
        }
        self.position += 1;
        Ok(())
    }

    pub fn decode(&mut self, graph: bool, trace: Option<(&mut Trace, &str)>) -> Result<u32> {
        self.advance(graph, trace)?;
        self.token()
    }

    pub fn step(&mut self, graph: bool) -> Result<()> {
        self.advance(graph, None)
    }

    pub fn rewind(&mut self, position: usize, token: u32) -> Result<()> {
        ensure!(
            position <= self.position,
            "rewind position exceeds current cache length"
        );
        self.device.stream.synchronize()?;
        self.device
            .stream
            .memcpy_htod(&[position as i32], &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&[(position + 1) as i32], &mut self.work.length)?;
        self.set_token(token)?;
        self.position = position;
        self.device.stream.synchronize()?;
        Ok(())
    }

    pub fn weight_bytes(&self) -> usize {
        self.weights.bytes
    }

    pub fn decode_weight_bytes(&self) -> usize {
        self.weights.bytes - self.weights.embed.len() * size_of::<bf16>()
            + self.config.hidden_size * size_of::<bf16>()
    }

    pub fn kv_read_bytes(&self, tokens: usize) -> usize {
        2 * self.config.num_hidden_layers * tokens * self.config.kv_dim() * size_of::<bf16>()
    }

    pub fn capture(&mut self) -> Result<()> {
        ensure!(
            self.spec.is_none(),
            "graph decode is not declared for draft-enabled engines"
        );
        if self.graph.is_some() {
            return Ok(());
        }
        self.closure.capture()?;
        self.ensure_decode()?;
        ensure!(
            self.position < self.capacity,
            "graph capture exceeds KV capacity"
        );
        let saved = self.token()?;
        let saved_position = self.device.stream.clone_dtoh(&self.work.position)?;
        let saved_length = self.device.stream.clone_dtoh(&self.work.length)?;
        self.run_decode_step(None)?;
        self.device.stream.synchronize()?;
        self.device
            .stream
            .memcpy_htod(&saved_position, &mut self.work.position)?;
        self.device
            .stream
            .memcpy_htod(&saved_length, &mut self.work.length)?;
        self.device
            .stream
            .memcpy_htod(&[saved], &mut self.work.token)?;
        self.device.stream.synchronize()?;
        self.device
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let result = (|| -> Result<()> {
            self.run_decode_step(None)?;
            Ok(())
        })();
        let graph = self
            .device
            .stream
            .end_capture(sys::CUgraphInstantiate_flags(0));
        result?;
        self.graph = Some(graph?.context("empty decode graph")?);
        self.graph.as_ref().context("missing graph")?.upload()?;
        self.device.stream.synchronize()?;
        Ok(())
    }

    pub fn token(&self) -> Result<u32> {
        Ok(self.device.stream.clone_dtoh(&self.work.token)?[0])
    }

    pub fn set_token(&mut self, token: u32) -> Result<()> {
        ensure!(
            (token as usize) < self.config.vocab_size,
            "token outside vocabulary"
        );
        self.device
            .stream
            .memcpy_htod(&[token], &mut self.work.token)?;
        Ok(())
    }

    pub fn decode_logits(&self) -> Result<Vec<f32>> {
        let values = self
            .device
            .stream
            .clone_dtoh(&self.work.logits.slice(..self.config.vocab_size))?;
        Ok(values.into_iter().map(bf16::to_f32).collect())
    }

    pub fn event(&self) -> Result<Event> {
        Ok(Event(
            self.device
                .ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
        ))
    }

    pub fn record(&self, event: &Event) -> Result<()> {
        Ok(event.0.record(&self.device.stream)?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.device.stream.synchronize()?)
    }

    pub fn sample_buffer(&self, count: usize) -> Result<Logits> {
        Ok(Logits(
            self.device
                .stream
                .alloc_zeros(count * self.config.vocab_size)?,
        ))
    }

    pub fn copy_logits(&self, dst: &mut Logits, step: usize) -> Result<()> {
        let width = self.config.vocab_size;
        ensure!(
            (step + 1)
                .checked_mul(width)
                .is_some_and(|n| n <= dst.0.len()),
            "logit sample destination size mismatch"
        );
        self.device.stream.memcpy_dtod(
            &self.work.logits.slice(..width),
            &mut dst.0.slice_mut(step * width..(step + 1) * width),
        )?;
        Ok(())
    }

    pub fn check_samples(&self, src: &Logits) -> Result<usize> {
        let values = self.device.stream.clone_dtoh(&src.0)?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite logits in benchmark samples"
        );
        Ok(values.len())
    }

    pub fn profiler_start(&self) -> Result<()> {
        Ok(cudarc::driver::profiler_start()?)
    }

    pub fn profiler_stop(&self) -> Result<()> {
        Ok(cudarc::driver::profiler_stop()?)
    }

    pub fn check_logits(&self) -> Result<()> {
        let values = self.decode_logits()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite model logits"
        );
        Ok(())
    }

    pub fn trace_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            layer < self.config.num_hidden_layers,
            "trace layer {layer} is outside model"
        );
        self.trace_layer = layer;
        Ok(())
    }

    pub fn decode_linear(&self) -> &'static str {
        self.blas.decode_impl()
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    pub fn attention_chunk(&self) -> Option<usize> {
        self.decode.as_ref().map(Attention::chunk)
    }

    pub fn attention_plan(&self) -> Option<crate::backend::setup::AttentionPlan> {
        self.decode.as_ref().map(Attention::plan)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Err(error) = self.device.stream.synchronize() {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr().lock(),
                "model stream synchronization failed during drop: {error}"
            );
        }
    }
}

impl Engine {
    fn ensure_verification(&mut self) -> Result<()> {
        let spec = self.spec.as_mut().context("draft state is not enabled")?;
        if spec.verification.is_some() {
            return Ok(());
        }
        let calibrated = super::multi_calibrate::load(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            crate::backend::setup::MultiShape {
                rows: 8,
                causal: true,
                capacity: self.capacity,
            },
            &self.closure,
        )
        .context("MQ verification setup calibration")?;
        let (attention, selection) = calibrated.choose(self.capacity)?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{
                "rows":8,"attention":attention,"selection":selection,"calibration":calibrated,
                "length":"device position counts prefix KV before the query block"
            }})
        );
        spec.verification = Some(super::verification::Verification::with_plan(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows: 8,
                causal: true,
            },
            attention,
        )?);
        Ok(())
    }

    fn ensure_tree_attention(&mut self) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("tree requires draft State")?
            .tree_attention
            .is_some()
        {
            return Ok(());
        }
        let rows = self.spec_budget.rows();
        let measured = super::multi_calibrate::load_tree(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            rows,
            self.capacity,
            &self.closure,
        )?;
        let (plan, selection) = measured.choose(self.capacity)?;
        let attention = super::verification::Verification::with_plan(
            &self.device.ctx,
            &self.device.stream,
            &self.config,
            crate::backend::setup::MultiShape {
                capacity: self.capacity,
                rows,
                causal: false,
            },
            plan,
        )?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup":{"role":"tree","rows":rows,"attention":plan,"selection":selection,"calibration":measured}})
        );
        let spec = self.spec.as_mut().context("tree requires draft State")?;
        spec.tree_attention = Some(attention);
        Ok(())
    }

    pub fn verify_tree(
        &mut self,
        tree: &crate::tree::Tree,
        trace: Option<Target<'_>>,
    ) -> Result<Vec<u32>> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(
            tree.prefix() == self.position,
            "tree prefix does not match committed KV"
        );
        self.spec
            .as_mut()
            .context("tree verification requires draft State")?
            .tree
            .as_mut()
            .context("tree verification requires --spec-budget tree16/tree32/tree64")?
            .load(tree)?;
        let parents: Vec<_> = tree.nodes().iter().map(|n| n.parent).collect();
        let positions: Vec<_> = tree.nodes().iter().map(|n| n.position).collect();
        let step = crate::backend::Step {
            rows: tree.nodes().len(),
            input: crate::backend::Input::Ids,
            head: Head::All,
            mask: crate::backend::Mask::Tree {
                parents: &parents,
                positions: &positions,
            },
        };
        let graph = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .graph;
        ensure!(
            !graph || trace.is_none(),
            "tree Graph tracing requires --spec-graph false"
        );
        if graph {
            self.ensure_tree_verify_graph(step)?;
            let spec = self.spec.as_mut().context("draft State is not enabled")?;
            spec.tree_verify_graph
                .as_ref()
                .context("tree verify Graph is not captured")?
                .launch()?;
            spec.active_rows = tree.nodes().len();
        } else {
            self.run_step(step, trace)?;
        }
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        Ok(self
            .device
            .stream
            .clone_dtoh(&spec.predictions.slice(..tree.nodes().len()))?)
    }

    fn ensure_tree_verify_graph(&mut self, step: crate::backend::Step<'_>) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree_verify_graph
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let s = self.device.stream.clone();
        let timer = std::time::Instant::now();
        let token = s.clone_dtoh(&self.work.token)?;
        let position = s.clone_dtoh(&self.work.position)?;
        let length = s.clone_dtoh(&self.work.length)?;
        self.run_step(step, None)?;
        s.synchronize()?;
        s.memcpy_htod(&token, &mut self.work.token)?;
        s.memcpy_htod(&position, &mut self.work.position)?;
        s.memcpy_htod(&length, &mut self.work.length)?;
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = self.run_step(step, None);
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        launched?;
        let graph = captured?.context("empty tree verify Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .tree_verify_graph = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"tree_graph_setup":{"phase":"verify","budget":self.spec_budget.rows(),"wall_ms":timer.elapsed().as_secs_f64()*1000.0}})
        );
        Ok(())
    }

    fn tree_gather_scatter(&mut self) -> Result<()> {
        let spec = self.spec.as_mut().context("draft State is not enabled")?;
        let state = spec.tree.as_mut().context("tree State is not enabled")?;
        state.gather_path(&self.kv, &spec.capture)?;
        state.scatter(&mut self.kv)
    }

    fn capture_tree_commit(&mut self) -> Result<()> {
        self.closure.capture()?;
        let s = self.device.stream.clone();
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = self.tree_gather_scatter();
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        launched?;
        let graph = captured?.context("empty tree commit Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .tree_commit_graph = Some(graph);
        Ok(())
    }

    fn tree_inject(&mut self, start: usize) -> Result<()> {
        let spec = self.spec.as_mut().context("draft State is not enabled")?;
        let state = spec.tree.as_ref().context("tree State is not enabled")?;
        spec.draft.inject_device(
            state.compact_hidden(),
            state.keep(),
            start,
            state.prefix(),
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
                config: &self.config,
            },
        )
    }

    fn capture_tree_inject(&mut self, start: usize, rows: usize) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .inject_graphs[rows - 1]
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let s = self.device.stream.clone();
        self.tree_inject(start)?;
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = self.tree_inject(start);
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        launched?;
        let graph = captured?.context("empty tree inject Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .inject_graphs[rows - 1] = Some(graph);
        Ok(())
    }

    pub fn tree_metadata(&self) -> Result<crate::backend::TreeMetadata> {
        let tree = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        let s = &self.device.stream;
        let n = tree.rows();
        Ok(crate::backend::TreeMetadata {
            ids: s.clone_dtoh(&tree.ids().slice(..n))?,
            ancestors: s.clone_dtoh(&tree.ancestors().slice(..n))?,
            positions: s.clone_dtoh(&tree.positions().slice(..n))?,
            slots: s.clone_dtoh(&tree.slots().slice(..n))?,
            valid_rows: s.clone_dtoh(tree.valid_rows())?[0],
            prefix: s.clone_dtoh(tree.prefix())?[0],
        })
    }

    pub fn tree_hidden(&self) -> Result<Vec<f32>> {
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        let rows = spec
            .tree
            .as_ref()
            .context("tree State is not enabled")?
            .rows();
        Ok(self
            .device
            .stream
            .clone_dtoh(&spec.capture.slice(..rows * 5 * self.config.hidden_size))?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    pub fn poison_tree_padding(&mut self, tree: &crate::tree::Tree) -> Result<()> {
        let n = tree.nodes().len();
        let budget = self.spec_budget.rows();
        ensure!(
            self.spec_budget.is_tree() && n < budget && tree.prefix() + budget <= self.capacity,
            "padding negative control requires actual N<tree budget within capacity"
        );
        let d = self.config.head_dim;
        let values = vec![bf16::NAN; (budget - n) * d];
        for kv in &mut self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + tree.prefix() + n) * d;
                self.device
                    .stream
                    .memcpy_htod(&values, &mut kv.k.slice_mut(lo..lo + values.len()))?;
                self.device
                    .stream
                    .memcpy_htod(&values, &mut kv.v.slice_mut(lo..lo + values.len()))?;
            }
        }
        self.device.stream.synchronize()?;
        Ok(())
    }

    pub fn commit_tree(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
    ) -> Result<()> {
        let spec = self.spec.as_ref().context("draft State is not enabled")?;
        let predictions = self
            .device
            .stream
            .clone_dtoh(&spec.predictions.slice(..tree.nodes().len()))?;
        self.commit_tree_checked(tree, plan, eos, &predictions)
    }

    pub fn commit_tree_fixture(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        reference_predictions: &[u32],
    ) -> Result<()> {
        ensure!(
            reference_predictions.len() == tree.nodes().len()
                && reference_predictions
                    .iter()
                    .all(|&v| (v as usize) < self.config.vocab_size),
            "teacher-forced reference predictions have invalid extent/token"
        );
        self.commit_tree_checked(tree, plan, eos, reference_predictions)
    }

    fn commit_tree_checked(
        &mut self,
        tree: &crate::tree::Tree,
        plan: &crate::tree::Commit,
        eos: &[u32],
        predictions: &[u32],
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        let mut cursor = crate::tree::Cursor {
            position: self.position,
            token: self.token()?,
        };
        tree.check_commit(plan, predictions, eos, &cursor)?;
        let result = (|| -> Result<()> {
            let spec = self.spec.as_mut().context("draft State is not enabled")?;
            spec.tree
                .as_mut()
                .context("tree State is not enabled")?
                .load_commit(plan)?;
            let graph = spec.graph;
            self.device.stream.synchronize()?;
            let s = self.device.stream.clone();
            let cached = graph
                && self
                    .spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .tree_commit_graph
                    .is_some();
            if cached {
                self.spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .tree_commit_graph
                    .as_ref()
                    .context("tree commit Graph missing")?
                    .launch()?;
            } else {
                self.tree_gather_scatter()?;
            }
            s.synchronize()?;
            if graph && !cached {
                self.capture_tree_commit()?;
            }
            if graph {
                self.capture_tree_inject(self.position, plan.rows.len())?;
            }
            if graph {
                self.spec
                    .as_ref()
                    .context("draft State is not enabled")?
                    .inject_graphs[plan.rows.len() - 1]
                    .as_ref()
                    .context("tree inject Graph missing")?
                    .launch()?;
            } else {
                self.tree_inject(self.position)?;
            }
            s.synchronize()?;
            let spec = self.spec.as_ref().context("draft State is not enabled")?;
            spec.tree
                .as_ref()
                .context("tree State is not enabled")?
                .select_logits(&spec.logits, &mut self.work.logits)?;
            s.memcpy_htod(&[plan.position as i32], &mut self.work.position)?;
            s.memcpy_htod(&[(plan.position + 1) as i32], &mut self.work.length)?;
            s.memcpy_htod(&[plan.next], &mut self.work.token)?;
            s.synchronize()?;
            tree.finish(plan, predictions, eos, &mut cursor)?;
            self.position = cursor.position;
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result.context("tree commit failed; reset/reload required")
    }

    pub fn tree_kv_path(&self, path: &[usize]) -> Result<Vec<f32>> {
        let state = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        ensure!(
            !path.is_empty()
                && path[0] == 0
                && path.iter().all(|&i| i < state.rows())
                && path.windows(2).all(|r| r[0] < r[1]),
            "tree KV path is invalid"
        );
        let prefix = self.device.stream.clone_dtoh(state.prefix())?[0];
        ensure!(
            prefix >= 0 && prefix as usize == self.position,
            "tree KV path must be read before commit"
        );
        let mut result = Vec::new();
        let d = self.config.head_dim;
        for layer in &self.kv {
            for data in [&layer.k, &layer.v] {
                for head in 0..self.config.num_key_value_heads {
                    for &row in path {
                        let lo = (head * self.capacity + self.position + row) * d;
                        result.extend(
                            self.device
                                .stream
                                .clone_dtoh(&data.slice(lo..lo + d))?
                                .into_iter()
                                .map(bf16::to_f32),
                        );
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn tree_injected_hidden(&self) -> Result<Vec<f32>> {
        let state = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .tree
            .as_ref()
            .context("tree State is not enabled")?;
        ensure!(
            state.keep() > 0,
            "tree injected hidden requires a committed path"
        );
        Ok(self
            .device
            .stream
            .clone_dtoh(
                &state
                    .compact_hidden()
                    .slice(..state.keep() * 5 * self.config.hidden_size),
            )?
            .into_iter()
            .map(bf16::to_f32)
            .collect())
    }

    fn inject_capture(
        &mut self,
        rows: usize,
        start: usize,
        trace: Option<(&mut Trace, &str)>,
    ) -> Result<()> {
        let spec = self.spec.as_mut().context("draft state is not enabled")?;
        spec.draft.inject(
            &spec.capture,
            rows,
            start,
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
                config: &self.config,
            },
            trace,
        )
    }

    fn run_spec_device(&mut self, phase: SpecPhase, start: usize) -> Result<()> {
        match phase {
            SpecPhase::Draft => self
                .spec
                .as_mut()
                .context("draft is not enabled")?
                .draft
                .propose_device(
                    &self.work.token,
                    &self.work.position,
                    start,
                    &self.weights,
                    super::draft::Compute {
                        ops: &self.ops,
                        blas: &mut self.blas,
                        config: &self.config,
                    },
                ),
            SpecPhase::Verify => {
                self.spec
                    .as_ref()
                    .context("draft is not enabled")?
                    .draft
                    .copy_proposals(&mut self.work.ids.slice_mut(1..8))?;
                self.device
                    .stream
                    .memcpy_dtod(&self.work.token, &mut self.work.ids.slice_mut(..1))?;
                self.run_step(
                    crate::backend::Step {
                        rows: 8,
                        input: crate::backend::Input::Ids,
                        head: Head::All,
                        mask: crate::backend::Mask::Causal,
                    },
                    None,
                )
            }
            SpecPhase::Inject(rows) => {
                let spec = self.spec.as_mut().context("draft is not enabled")?;
                spec.draft.inject_device(
                    &spec.capture,
                    rows,
                    start,
                    &self.work.position,
                    super::draft::Compute {
                        ops: &self.ops,
                        blas: &mut self.blas,
                        config: &self.config,
                    },
                )
            }
        }
    }

    fn ensure_spec_graph(&mut self, phase: SpecPhase, start: usize) -> Result<()> {
        if self
            .spec
            .as_mut()
            .context("draft is not enabled")?
            .graph_slot(phase)?
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let timer = std::time::Instant::now();
        let s = self.device.stream.clone();
        let token = s.clone_dtoh(&self.work.token)?;
        let position = s.clone_dtoh(&self.work.position)?;
        let length = s.clone_dtoh(&self.work.length)?;
        self.run_spec_device(phase, start)?;
        s.synchronize()?;
        s.memcpy_htod(&token, &mut self.work.token)?;
        s.memcpy_htod(&position, &mut self.work.position)?;
        s.memcpy_htod(&length, &mut self.work.length)?;
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = self.run_spec_device(phase, start);
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        launched?;
        let graph = captured?.context("empty spec Graph")?;
        graph.upload()?;
        s.synchronize()?;
        *self
            .spec
            .as_mut()
            .context("draft is not enabled")?
            .graph_slot(phase)? = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"spec_graph_setup":{"phase":phase.name(),"rows":phase.rows(),"capacity":self.capacity,"wall_ms":timer.elapsed().as_secs_f64()*1000.0,"prefix":"device position","warmup":"selected implementation; token/position/length restored"}})
        );
        Ok(())
    }

    fn launch_spec_graph(&mut self, phase: SpecPhase) -> Result<()> {
        let spec = self.spec.as_mut().context("draft is not enabled")?;
        spec.graph_slot(phase)?
            .as_ref()
            .context("spec Graph has not been captured")?
            .launch()?;
        if matches!(phase, SpecPhase::Verify) {
            spec.active_rows = phase.rows();
        }
        Ok(())
    }

    fn run_tree_draft_device(&mut self, start: usize) -> Result<()> {
        self.spec
            .as_mut()
            .context("draft is not enabled")?
            .draft
            .forward_base_device(
                &self.work.token,
                &self.work.position,
                start,
                &self.weights,
                super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                    config: &self.config,
                },
            )
    }

    fn ensure_tree_draft_graph(&mut self, start: usize) -> Result<()> {
        if self
            .spec
            .as_ref()
            .context("draft is not enabled")?
            .tree_draft_graph
            .is_some()
        {
            return Ok(());
        }
        self.closure.capture()?;
        let timer = std::time::Instant::now();
        let s = self.device.stream.clone();
        self.run_tree_draft_device(start)?;
        s.synchronize()?;
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let launched = self.run_tree_draft_device(start);
        let captured = s.end_capture(sys::CUgraphInstantiate_flags(0));
        launched?;
        let graph = captured?.context("empty tree draft Graph")?;
        graph.upload()?;
        s.synchronize()?;
        self.spec
            .as_mut()
            .context("draft is not enabled")?
            .tree_draft_graph = Some(graph);
        eprintln!(
            "{}",
            serde_json::json!({"tree_graph_setup":{"phase":"draft","rows":7,"capacity":self.capacity,"wall_ms":timer.elapsed().as_secs_f64()*1000.0}})
        );
        Ok(())
    }

    pub fn spec_round(
        &mut self,
        limit: usize,
        mut trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        self.closure.assert_ready()?;
        if self.spec_budget.is_tree() {
            return self.tree_round(limit, trace, prefix);
        }
        self.ensure_verification()?;
        self.spec
            .as_mut()
            .context("draft state is not enabled")?
            .draft
            .setup_attention(
                &self.device,
                &self.ops,
                &self.config,
                &self.runtime,
                &self.closure,
            )?;
        let _round_range = super::profile::Range::new(self.profile_rounds, c"dspark_round");
        ensure!(limit > 0, "spec round output limit must be positive");
        ensure!(
            self.position + 8 <= self.capacity,
            "spec verification exceeds KV capacity"
        );
        let graph = self.spec.as_ref().context("draft is not enabled")?.graph;
        ensure!(
            !graph || trace.is_none(),
            "spec Graph tracing requires --spec-graph false"
        );
        let start = self.position;
        if graph {
            self.ensure_spec_graph(SpecPhase::Draft, start)?;
            self.ensure_spec_graph(SpecPhase::Verify, start)?;
        }
        let anchor = if graph { None } else { Some(self.token()?) };
        let draft_range = super::profile::Range::new(self.profile_rounds, c"draft");
        let draft_start = std::time::Instant::now();
        let proposals = if graph {
            self.launch_spec_graph(SpecPhase::Draft)?;
            self.spec
                .as_ref()
                .context("draft is not enabled")?
                .draft
                .proposals()?
        } else {
            let anchor = anchor.context("eager draft anchor missing")?;
            let spec = self.spec.as_mut().context("draft is not enabled")?;
            spec.draft.propose(
                anchor,
                start,
                &self.weights,
                super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                    config: &self.config,
                },
                trace.as_deref_mut().map(|t| (t, prefix)),
                None,
            )?
        };
        self.device.stream.synchronize()?;
        let draft_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        drop(draft_range);
        if !graph {
            let mut ids = vec![anchor.context("eager verify anchor missing")?];
            ids.extend(&proposals);
            self.device
                .stream
                .memcpy_htod(&ids, &mut self.work.ids.slice_mut(..8))?;
            self.device
                .stream
                .memcpy_htod(&[start as i32], &mut self.work.position)?;
        }
        let verify_start = std::time::Instant::now();
        let verify_range = super::profile::Range::new(self.profile_rounds, c"verify");
        if graph {
            self.launch_spec_graph(SpecPhase::Verify)?;
        } else {
            self.run_step(
                crate::backend::Step {
                    rows: 8,
                    input: crate::backend::Input::Ids,
                    head: Head::All,
                    mask: crate::backend::Mask::Causal,
                },
                None,
            )?;
        }
        let predictions = self.device.stream.clone_dtoh(
            &self
                .spec
                .as_ref()
                .context("draft is not enabled")?
                .predictions,
        )?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        drop(verify_range);
        self.position += 8;
        let accepted = crate::dspark::accept(&proposals, &predictions)?;
        let mut output = proposals[..accepted.proposals].to_vec();
        output.push(accepted.bonus);
        output.truncate(limit);
        let committed = output.len();
        let bonus = *output.last().context("spec round emitted no token")?;
        if graph {
            self.ensure_spec_graph(SpecPhase::Inject(committed), start)?;
        }
        let inject_start = std::time::Instant::now();
        let _inject_range = super::profile::Range::new(self.profile_rounds, c"inject");
        if graph {
            self.launch_spec_graph(SpecPhase::Inject(committed))?;
        } else {
            self.inject_capture(committed, start, trace.as_deref_mut().map(|t| (t, prefix)))?;
        }
        let s = &self.device.stream;
        let spec = self.spec.as_ref().context("draft is not enabled")?;
        s.memcpy_dtod(
            &spec.logits.slice(
                (committed - 1) * self.config.vocab_size..committed * self.config.vocab_size,
            ),
            &mut self.work.logits,
        )?;
        self.rewind(start + committed, bonus)?;
        if let Some(t) = trace {
            t.add(
                format!("{prefix}.target_argmax"),
                vec![8],
                predictions.iter().map(|&v| v as f32).collect(),
            )?;
            t.add(
                format!("{prefix}.m"),
                vec![1],
                vec![accepted.proposals as f32],
            )?;
            t.add(
                format!("{prefix}.bonus"),
                vec![1],
                vec![accepted.bonus as f32],
            )?;
        }
        Ok(crate::dspark::Round {
            logits_rows: (0..committed).collect(),
            proposals,
            predictions,
            accepted: accepted.proposals,
            committed,
            output,
            position: self.position,
            draft_ms,
            verify_ms,
            inject_ms: inject_start.elapsed().as_secs_f64() * 1000.0,
            tree: None,
        })
    }

    pub fn set_spec_graph(&mut self, graph: bool) -> Result<()> {
        self.device.stream.synchronize()?;
        ensure!(
            !graph || !self.graphs.is_empty(),
            "spec Graphs were not prepared; load with --spec-graph true"
        );
        self.spec.as_mut().context("draft is not enabled")?.graph = graph;
        eprintln!("{}", serde_json::json!({"spec_execution":{"graph":graph}}));
        Ok(())
    }

    pub fn set_spec_eos(&mut self, eos: &[u32]) -> Result<()> {
        ensure!(
            eos.iter().all(|&v| (v as usize) < self.config.vocab_size),
            "spec EOS token is outside vocabulary"
        );
        self.spec
            .as_mut()
            .context("draft State is not enabled")?
            .eos = eos.to_vec();
        Ok(())
    }

    fn tree_round(
        &mut self,
        limit: usize,
        trace: Option<&mut Trace>,
        prefix: &str,
    ) -> Result<crate::dspark::Round> {
        ensure!(
            !self.poisoned,
            "tree transaction failed; reset/reload is required"
        );
        ensure!(limit > 0, "tree round output limit must be positive");
        let start = self.position;
        ensure!(
            start
                .checked_add(self.spec_budget.rows())
                .is_some_and(|end| end <= self.capacity),
            "tree scratch exceeds KV capacity"
        );
        let anchor = self.token()?;
        let draft_start = std::time::Instant::now();
        let graph = self
            .spec
            .as_ref()
            .context("tree round requires enabled draft State")?
            .graph;
        let spec = self
            .spec
            .as_mut()
            .context("tree round requires enabled draft State")?;
        spec.draft.setup_attention(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            &self.closure,
        )?;
        spec.draft
            .setup_markov(&self.device, &self.ops, &self.runtime, &self.closure)?;
        if graph {
            self.ensure_tree_draft_graph(start)?;
            self.spec
                .as_mut()
                .context("tree round requires enabled draft State")?
                .tree_draft_graph
                .as_ref()
                .context("tree draft Graph is not captured")?
                .launch()?;
        } else {
            let spec = self
                .spec
                .as_mut()
                .context("tree round requires enabled draft State")?;
            spec.draft.forward_base(
                anchor,
                start,
                &self.weights,
                super::draft::Compute {
                    ops: &self.ops,
                    blas: &mut self.blas,
                    config: &self.config,
                },
            )?;
        }
        self.device.stream.synchronize()?;
        let base_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        let build_start = std::time::Instant::now();
        let (tree, stats) = self.build_tree(self.tree_builder, anchor, start)?;
        let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
        if self.compare_builders {
            let other = match self.tree_builder {
                TreeBuilder::Serial => TreeBuilder::Waves,
                TreeBuilder::Waves => TreeBuilder::Serial,
            };
            let (reference, _) = self.build_tree(other, anchor, start)?;
            ensure!(
                tree.nodes().len() == reference.nodes().len(),
                "tree builders return different node counts"
            );
            for (row, (actual, expected)) in tree.nodes().iter().zip(reference.nodes()).enumerate()
            {
                ensure!(
                    actual == expected && actual.logp.to_bits() == expected.logp.to_bits(),
                    "tree builders differ at node {row}: actual={actual:?}, expected={expected:?}"
                );
            }
        }
        let draft_ms = draft_start.elapsed().as_secs_f64() * 1000.0;
        let verify_start = std::time::Instant::now();
        let predictions = self.verify_tree(
            &tree,
            trace.map(|trace| Target {
                trace,
                prefix,
                skip: 0,
            }),
        )?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let eos = self
            .spec
            .as_ref()
            .context("draft State is not enabled")?
            .eos
            .clone();
        let plan = tree.select(&predictions, limit, &eos)?;
        let commit_start = std::time::Instant::now();
        self.commit_tree(&tree, &plan, &eos)?;
        let commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
        Ok(crate::dspark::Round {
            logits_rows: plan.rows.clone(),
            proposals: tree.nodes()[1..].iter().map(|n| n.token).collect(),
            predictions,
            accepted: plan.matched,
            committed: plan.output.len(),
            output: plan.output,
            position: plan.position,
            draft_ms,
            verify_ms,
            inject_ms: commit_ms,
            tree: Some(crate::dspark::TreeTiming {
                builder: self.tree_builder,
                base_ms,
                build_ms,
                waves: stats.waves,
                requests: stats.requests,
                used_requests: stats.used_requests,
                batch_sizes: stats.batch_sizes,
                compared_serial: self.compare_builders,
            }),
        })
    }

    fn build_tree(
        &mut self,
        builder: TreeBuilder,
        anchor: u32,
        prefix: usize,
    ) -> Result<(crate::tree::Tree, crate::tree::WaveStats)> {
        let spec = self
            .spec
            .as_mut()
            .context("tree building requires draft state")?;
        match builder {
            TreeBuilder::Waves => crate::tree::Tree::best_first_waves(
                anchor,
                prefix,
                self.capacity,
                self.spec_budget.rows(),
                7,
                |requests| {
                    spec.draft
                        .distributions_batch(requests)?
                        .iter()
                        .map(crate::backend::Top4::distribution)
                        .collect()
                },
            ),
            TreeBuilder::Serial => {
                let mut cache = std::collections::HashMap::new();
                let mut count = 0;
                let tree = crate::tree::Tree::best_first(
                    anchor,
                    prefix,
                    self.capacity,
                    self.spec_budget.rows(),
                    7,
                    |row, token| {
                        if let Some(value) = cache.get(&(row, token)) {
                            return Ok(crate::tree::Distribution::clone(value));
                        }
                        let value = spec
                            .draft
                            .distributions_batch(&[(row, token)])?
                            .into_iter()
                            .next()
                            .context("empty serial Markov result")?
                            .distribution()?;
                        count += 1;
                        cache.insert((row, token), value.clone());
                        Ok(value)
                    },
                )?;
                Ok((
                    tree,
                    crate::tree::WaveStats {
                        waves: count,
                        requests: count,
                        batch_sizes: vec![1; count],
                        used_requests: count,
                    },
                ))
            }
        }
    }

    pub fn compare_tree_builders(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !enabled || self.spec_budget.is_tree(),
            "builder comparison requires a tree budget"
        );
        self.compare_builders = enabled;
        Ok(())
    }

    pub fn distributions_batch(
        &mut self,
        requests: &[(u8, u32)],
    ) -> Result<Vec<crate::backend::Top4>> {
        ensure!(
            self.spec_budget.is_tree(),
            "distributions_batch requires a tree spec budget"
        );
        self.closure.assert_ready()?;
        self.spec
            .as_mut()
            .context("draft state is not enabled")?
            .draft
            .distributions_batch(requests)
    }

    pub fn spec_state_bits(&self, start: usize, rows: usize) -> Result<Vec<u8>> {
        ensure!(
            rows > 0 && start + rows == self.position,
            "spec bit comparison must cover newly committed prefix"
        );
        let s = &self.device.stream;
        let mut out = Vec::new();
        super::copy::bits(s, &self.work.logits, &mut out)?;
        let spec = self.spec.as_ref().context("draft is not enabled")?;
        super::copy::bits(s, &spec.logits, &mut out)?;
        spec.draft.state_bits(start, rows, &mut out)?;
        let d = self.config.head_dim;
        for kv in &self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = (head * self.capacity + start) * d;
                let hi = lo + rows * d;
                super::copy::bits(s, &kv.k.slice(lo..hi), &mut out)?;
                super::copy::bits(s, &kv.v.slice(lo..hi), &mut out)?;
            }
        }
        for x in s
            .clone_dtoh(&self.work.position)?
            .into_iter()
            .chain(s.clone_dtoh(&self.work.length)?)
        {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for x in s.clone_dtoh(&self.work.token)? {
            out.extend_from_slice(&x.to_le_bytes());
        }
        Ok(out)
    }

    pub fn inject_hidden(&mut self, hidden: &[f32], start: usize, rows: usize) -> Result<()> {
        ensure!(
            hidden.len() == rows * 5 * self.config.hidden_size,
            "injected hidden shape mismatch"
        );
        let spec = self.spec.as_mut().context("draft is not enabled")?;
        ensure!(rows <= self.workspace_rows, "injection exceeds workspace");
        let values: Vec<bf16> = hidden.iter().copied().map(bf16::from_f32).collect();
        self.device
            .stream
            .memcpy_htod(&values, &mut spec.capture.slice_mut(..values.len()))?;
        self.inject_capture(rows, start, None)
    }

    pub fn draft_proposals(
        &mut self,
        anchor: u32,
        position: usize,
        trace: &mut Trace,
        prefix: &str,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        let spec = self.spec.as_mut().context("draft is not enabled")?;
        spec.draft.setup_attention(
            &self.device,
            &self.ops,
            &self.config,
            &self.runtime,
            &self.closure,
        )?;
        spec.draft.propose(
            anchor,
            position,
            &self.weights,
            super::draft::Compute {
                ops: &self.ops,
                blas: &mut self.blas,
                config: &self.config,
            },
            Some((trace, prefix)),
            forced,
        )
    }

    pub fn draft_kv(&self, trace: &mut Trace, start: usize, rows: usize) -> Result<()> {
        self.spec
            .as_ref()
            .context("draft is not enabled")?
            .draft
            .snapshot_kv(trace, start, rows, &self.config)
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

impl Engine {
    pub fn kv_snapshot(&self, start: usize, rows: usize) -> Result<Vec<f32>> {
        ensure!(
            start
                .checked_add(rows)
                .is_some_and(|end| end <= self.position),
            "KV snapshot exceeds valid prefix"
        );
        let c = &self.config;
        let mut output = Vec::new();
        for layer in &self.kv {
            for buffer in [&layer.k, &layer.v] {
                for head in 0..c.num_key_value_heads {
                    let values = self.device.stream.clone_dtoh(&buffer.slice(
                        (head * self.capacity + start) * c.head_dim
                            ..(head * self.capacity + start + rows) * c.head_dim,
                    ))?;
                    output.extend(values.into_iter().map(bf16::to_f32));
                }
            }
        }
        Ok(output)
    }
}

impl Engine {
    fn initialize(
        &mut self,
        model: &Path,
        draft: Option<&Path>,
        write: bool,
        started: Instant,
    ) -> Result<()> {
        let timer = Instant::now();
        let budget = self.spec.as_ref().map(|_| self.spec_budget);
        let profile = super::closure::Profile::new(
            &self.config,
            self.capacity,
            self.chunk,
            budget,
            model,
            draft,
            self.spec.as_ref().map(|s| &s.draft.config),
        )?;
        self.blas.declare(self.workspace_rows, &profile.pairs);
        let key = super::calibrate::key(&self.device)?;
        let bucket = crate::backend::setup::attention_bucket(self.capacity)?;
        let mut paths: Vec<_> = profile
            .linear
            .iter()
            .map(|&shape| self.blas.choice_path(shape))
            .collect();
        use super::closure::Attention as Kind;
        for &kind in &profile.attention {
            paths.push(match kind {
                Kind::Decode => super::calibrate::cache_path(&self.runtime, &key, bucket, None),
                Kind::Draft => {
                    super::multi_calibrate::cache_path(&self.runtime, &key, bucket, 7, false, false)
                }
                Kind::Verify | Kind::Tree => super::multi_calibrate::cache_path(
                    &self.runtime,
                    &key,
                    bucket,
                    self.spec_budget.rows(),
                    kind == Kind::Verify,
                    kind == Kind::Tree,
                ),
            });
        }
        if budget.is_some_and(SpecBudget::is_tree) {
            paths.push(super::markov_calibrate::cache_path(&self.runtime, &key));
        }
        paths.push(profile.path(&self.runtime, &key)?);
        self.closure.preflight(&paths)?;
        if write {
            let c = &self.config;
            for &shape in &profile.linear {
                let mut state = 1u32;
                let values: Vec<_> = (0..shape.rows * shape.input)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        bf16::from_f32((state >> 8) as f32 / 16777216.0 - 0.5)
                    })
                    .collect();
                let input = self.device.stream.clone_htod(&values)?;
                let layer = &self.weights.layers[0];
                let weight = if shape.output == c.vocab_size && shape.input == c.hidden_size {
                    Some(&self.weights.head)
                } else {
                    [
                        (&layer.qkv, c.qkv_dim(), c.hidden_size),
                        (&layer.o, c.hidden_size, c.hidden_size),
                        (&layer.gu, 2 * c.intermediate_size, c.hidden_size),
                        (&layer.down, c.hidden_size, c.intermediate_size),
                    ]
                    .into_iter()
                    .find(|(_, o, i)| *o == shape.output && *i == shape.input)
                    .map(|(w, _, _)| w)
                };
                if let Some(weight) = weight {
                    self.blas.calibrate_shape(weight, &input.slice(..), shape)?;
                } else {
                    self.spec
                        .as_ref()
                        .context("draft shape without draft")?
                        .draft
                        .calibrate_injection(&mut self.blas, &input.slice(..), shape)?;
                }
            }
        } else {
            self.blas.load_choices(&profile.linear)?;
        }
        for &kind in &profile.attention {
            match kind {
                Kind::Decode => self.ensure_decode()?,
                Kind::Draft => self
                    .spec
                    .as_mut()
                    .context("draft state missing")?
                    .draft
                    .setup_attention(
                        &self.device,
                        &self.ops,
                        &self.config,
                        &self.runtime,
                        &self.closure,
                    )?,
                Kind::Verify => self.ensure_verification()?,
                Kind::Tree => self.ensure_tree_attention()?,
            }
        }
        if budget.is_some_and(SpecBudget::is_tree) {
            self.spec
                .as_mut()
                .context("draft state missing")?
                .draft
                .setup_markov(&self.device, &self.ops, &self.runtime, &self.closure)?;
        }
        self.device.stream.synchronize()?;
        self.closure.finish(profile, &self.runtime, key)?;
        let choices_ms = timer.elapsed().as_secs_f64() * 1000.0;
        if write {
            return Ok(());
        }
        let capture = Instant::now();
        self.prepare_graphs()?;
        self.closure.expect_captures(self.graphs.len())?;
        self.check_graphs()?;
        let capture_ms = capture.elapsed().as_secs_f64() * 1000.0;
        let restore = Instant::now();
        self.restore_setup()?;
        eprintln!(
            "{}",
            serde_json::json!({"backend_setup_times":{"choices_ms":choices_ms,"graphs_ms":capture_ms,"restore_ms":restore.elapsed().as_secs_f64()*1000.0}})
        );
        self.closure.ready(started)
    }

    fn prepare_graphs(&mut self) -> Result<()> {
        self.set_token(0)?;
        if self.spec.is_none() {
            self.capture()?;
            self.graph
                .as_ref()
                .context("decode Graph missing")?
                .launch()?;
            self.device.stream.synchronize()?;
            self.check_logits()?;
            return Ok(());
        }
        if self.graphs.is_empty() {
            return Ok(());
        }
        let injections: Vec<_> = self
            .graphs
            .iter()
            .filter_map(|graph| match graph {
                Graph::Inject(rows) => Some(*rows),
                _ => None,
            })
            .collect();
        if !self.spec_budget.is_tree() {
            self.ensure_spec_graph(SpecPhase::Draft, 0)?;
            self.ensure_spec_graph(SpecPhase::Verify, 0)?;
            self.launch_spec_graph(SpecPhase::Draft)?;
            self.launch_spec_graph(SpecPhase::Verify)?;
            for rows in injections {
                self.ensure_spec_graph(SpecPhase::Inject(rows), 0)?;
                self.launch_spec_graph(SpecPhase::Inject(rows))?;
            }
        } else {
            self.ensure_tree_draft_graph(0)?;
            let count = self.spec_budget.rows();
            let tokens: Vec<_> = (0..count)
                .map(|r| if r < 8 { 0 } else { r as u32 })
                .collect();
            let parents: Vec<_> = (0..count)
                .map(|r| {
                    if r == 0 {
                        -1
                    } else if r < 8 {
                        (r - 1) as i32
                    } else {
                        0
                    }
                })
                .collect();
            let tree = crate::tree::Tree::edges(&tokens, &parents, 0, self.capacity)?;
            self.spec
                .as_mut()
                .context("draft state missing")?
                .tree
                .as_mut()
                .context("tree state missing")?
                .load(&tree)?;
            let positions: Vec<_> = tree.nodes().iter().map(|n| n.position).collect();
            let step = crate::backend::Step {
                rows: count,
                input: crate::backend::Input::Ids,
                head: Head::All,
                mask: crate::backend::Mask::Tree {
                    parents: &parents,
                    positions: &positions,
                },
            };
            self.ensure_tree_verify_graph(step)?;
            self.spec
                .as_ref()
                .context("draft state missing")?
                .tree_verify_graph
                .as_ref()
                .context("tree verify Graph missing")?
                .launch()?;
            let predictions = vec![0; count];
            for rows in injections {
                let plan = tree.select(&predictions, rows, &[])?;
                self.spec
                    .as_mut()
                    .context("draft state missing")?
                    .tree
                    .as_mut()
                    .context("tree state missing")?
                    .load_commit(&plan)?;
                self.tree_gather_scatter()?;
                if rows == 1 {
                    self.capture_tree_commit()?;
                }
                self.spec
                    .as_ref()
                    .context("draft state missing")?
                    .tree_commit_graph
                    .as_ref()
                    .context("tree commit Graph missing")?
                    .launch()?;
                self.capture_tree_inject(0, rows)?;
                self.spec
                    .as_ref()
                    .context("draft state missing")?
                    .inject_graphs[rows - 1]
                    .as_ref()
                    .context("tree inject Graph missing")?
                    .launch()?;
            }
            let spec = self.spec.as_mut().context("draft state missing")?;
            spec.tree_draft_graph
                .as_ref()
                .context("tree draft Graph missing")?
                .launch()?;
            for p in [1, 8, 64] {
                spec.draft.distributions_batch(&vec![(0, 0); p])?;
            }
        }
        self.device.stream.synchronize()?;
        Ok(())
    }

    fn restore_setup(&mut self) -> Result<()> {
        let s = self.device.stream.clone();
        s.synchronize()?;
        let rows = if self.spec.is_some() {
            self.spec_budget.rows()
        } else {
            1
        };
        for kv in &mut self.kv {
            for head in 0..self.config.num_key_value_heads {
                let lo = head * self.capacity * self.config.head_dim;
                let end = lo + rows * self.config.head_dim;
                s.memset_zeros(&mut kv.k.slice_mut(lo..end))?;
                s.memset_zeros(&mut kv.v.slice_mut(lo..end))?;
            }
        }
        for value in [
            &mut self.work.x,
            &mut self.work.n,
            &mut self.work.qkv,
            &mut self.work.attn,
            &mut self.work.out,
            &mut self.work.gu,
            &mut self.work.act,
            &mut self.work.logits,
        ] {
            s.memset_zeros(value)?;
        }
        s.memset_zeros(&mut self.work.ids)?;
        s.memset_zeros(&mut self.work.token)?;
        if let Some(spec) = &mut self.spec {
            spec.draft.reset_setup()?;
            for value in [&mut spec.capture, &mut spec.logits] {
                s.memset_zeros(value)?;
            }
            s.memset_zeros(&mut spec.predictions)?;
            spec.active_rows = 0;
            if let Some(tree) = &mut spec.tree {
                tree.reset_setup()?;
            }
        }
        self.reset()?;
        s.synchronize()?;
        ensure!(
            self.position == 0
                && self.token()? == 0
                && s.clone_dtoh(&self.work.position)? == [0]
                && s.clone_dtoh(&self.work.length)? == [1],
            "setup state was not restored"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
