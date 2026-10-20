pub mod detok;
pub mod http;
pub mod image;

use crate::model::Model;
use crate::spec::{self, Options as SpecOptions, Step};
use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::mpsc as tokio_mpsc;

pub struct Request {
    pub ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub images: Option<image::PreparedInput>,
}

pub enum EngineCommand {
    Generate {
        request: Request,
        events: tokio_mpsc::Sender<EngineEvent>,
    },
    Abort,
}

pub enum EngineEvent {
    Tokens {
        ids: Vec<u32>,
        accepted: usize,
    },
    Done {
        ids: Vec<u32>,
        rounds: usize,
        length_reached: bool,
    },
    Aborted {
        ids: Vec<u32>,
    },
    Failed(String),
}

/// Single-request admission state: Idle accepts, Running rejects with an
/// explicit error; the engine thread clears back to Idle when a run ends.
#[derive(PartialEq, Eq, Debug)]
pub enum Slot {
    Idle,
    Running,
    Unavailable,
}

impl Slot {
    fn admit(&mut self) -> Result<()> {
        ensure!(
            *self != Slot::Unavailable,
            "unavailable: engine recovery failed"
        );
        ensure!(*self == Slot::Idle, "busy: a request is already running");
        *self = Slot::Running;
        Ok(())
    }
}

pub struct Backend {
    commands: mpsc::Sender<EngineCommand>,
    slot: Arc<Mutex<Slot>>,
    options: SpecOptions,
    vocab_size: usize,
    draft_config: Option<crate::dspark::DraftConfig>,
    mtp: bool,
}

/// Owns admission until the engine has accepted the request.
pub(super) struct Admission<'a> {
    backend: &'a Backend,
    pending: bool,
}

impl Admission<'_> {
    fn send(mut self, request: Request, events: tokio_mpsc::Sender<EngineEvent>) -> Result<()> {
        self.backend
            .commands
            .send(EngineCommand::Generate { request, events })
            .context("engine thread has exited")?;
        self.pending = false;
        Ok(())
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        if self.pending {
            let mut slot = self
                .backend
                .slot
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *slot = Slot::Idle;
        }
    }
}

impl Backend {
    fn reserve(&self) -> Result<Admission<'_>> {
        self.slot
            .lock()
            .map_err(|_| anyhow::anyhow!("slot mutex poisoned"))?
            .admit()?;
        Ok(Admission {
            backend: self,
            pending: true,
        })
    }

    pub fn add_request(
        &self,
        request: Request,
        events: tokio_mpsc::Sender<EngineEvent>,
    ) -> Result<()> {
        self.reserve()?.send(request, events)
    }

    pub fn abort(&self) -> Result<()> {
        self.commands
            .send(EngineCommand::Abort)
            .context("engine thread has exited")
    }

    pub fn options(&self) -> SpecOptions {
        self.options.clone()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn load_model(target: &Path, options: &SpecOptions) -> Result<Model> {
    Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        options.backend.clone(),
    )
}

pub fn run(
    target: PathBuf,
    draft: Option<PathBuf>,
    options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
) -> Result<()> {
    run_inner(target, draft, options, tokenizer_dir, port, None)
}
pub fn run_mtp(
    target: PathBuf,
    options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
    depth: usize,
) -> Result<()> {
    run_inner(target, None, options, tokenizer_dir, port, Some(depth))
}
fn run_inner(
    target: PathBuf,
    draft: Option<PathBuf>,
    mut options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
    mtp: Option<usize>,
) -> Result<()> {
    if let Some(draft) = &draft {
        options.backend = options.backend.with_draft(draft);
    }
    let (command_tx, command_rx) = mpsc::channel::<EngineCommand>();
    let slot = Arc::new(Mutex::new(Slot::Idle));
    let (ready_tx, ready_rx) =
        mpsc::channel::<Result<(usize, Option<crate::dspark::DraftConfig>)>>();
    let engine_slot = slot.clone();
    let engine_options = options.clone();
    // The model holds non-Send CUDA handles; readiness carries the load result and vocab size before bind.
    let handle = thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let loaded = crate::model_spec::ModelSpec::read(&target).and_then(|spec| {
                let model = if let Some(depth) = mtp {
                    Model::check_resident_vision_budget(&target, &engine_options)
                        .and_then(|()| Model::load_mtp(&target, &engine_options, depth, true))
                        .and_then(|mut model| {
                            model.prepare_resident_vision()?;
                            Ok(model)
                        })
                } else {
                    load_model(&target, &engine_options)
                }?;
                let draft = engine_options
                    .backend
                    .draft_model
                    .as_ref()
                    .map(|_| model.draft_config().cloned())
                    .transpose()?;
                Ok((model, spec, draft))
            });
            let (model, model_spec, draft_config) = match loaded {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok((model.config.vocab_size, draft_config)));
            engine_loop(
                model,
                engine_options,
                model_spec,
                command_rx,
                engine_slot,
                mtp.is_some(),
            )
        })
        .context("spawn engine thread")?;
    let (vocab_size, draft_config) = ready_rx
        .recv()
        .context("engine thread exited before reporting readiness")??;
    let backend = Backend {
        commands: command_tx,
        slot,
        options,
        vocab_size,
        draft_config,
        mtp: mtp.is_some(),
    };
    http::serve(backend, tokenizer_dir, port, handle)
}

fn engine_loop(
    mut model: Model,
    options: SpecOptions,
    model_spec: crate::model_spec::ModelSpec,
    commands: mpsc::Receiver<EngineCommand>,
    slot: Arc<Mutex<Slot>>,
    mtp: bool,
) {
    while let Ok(command) = commands.recv() {
        match command {
            EngineCommand::Generate { request, events } => {
                generate_once(
                    &mut model,
                    &options,
                    &model_spec,
                    request,
                    events,
                    &commands,
                    mtp,
                );
                let recovered = if mtp { model.reset_mtp() } else { Ok(()) };
                let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
                *state = if recovered.is_ok() {
                    Slot::Idle
                } else {
                    Slot::Unavailable
                };
                if let Err(error) = recovered {
                    eprintln!("serve recovery failed: {error:#}");
                }
            }
            // A stale abort must not release a slot held by image preprocessing.
            EngineCommand::Abort => {}
        }
    }
}

fn generate_once(
    model: &mut Model,
    options: &SpecOptions,
    model_spec: &crate::model_spec::ModelSpec,
    request: Request,
    events: tokio_mpsc::Sender<EngineEvent>,
    commands: &mpsc::Receiver<EngineCommand>,
    mtp: bool,
) {
    let mtp = mtp && request.images.is_none();
    let route = if request.images.is_some() {
        "plain_image"
    } else if mtp {
        "mtp"
    } else {
        "plain"
    };
    let run_options = SpecOptions {
        count: request.max_new_tokens,
        ..options.clone()
    };
    let checked = if mtp {
        crate::mtp::check_request(request.ids.len(), run_options.count, run_options.capacity)
    } else if run_options.backend.draft_model.is_some() {
        model
            .draft_config()
            .and_then(|draft| check_request(&request.ids, &run_options, Some(draft)))
    } else {
        check_request(&request.ids, &run_options, None)
    };
    if let Err(error) = checked {
        let _ = events.blocking_send(EngineEvent::Failed(error.to_string()));
        return;
    }
    let input_dump = match trace_image_input(model_spec, &request) {
        Ok(path) => path,
        Err(error) => {
            let _ = events.blocking_send(EngineEvent::Failed(error.to_string()));
            return;
        }
    };
    let mut aborted = false;
    let mut on_event = |event| {
        if events.blocking_send(event).is_err() {
            aborted = true;
            return false;
        }
        match commands.try_recv() {
            Ok(EngineCommand::Abort) => {
                aborted = true;
                false
            }
            Ok(EngineCommand::Generate { .. }) => {
                aborted = true;
                false
            }
            Err(mpsc::TryRecvError::Empty) => true,
            Err(mpsc::TryRecvError::Disconnected) => {
                aborted = true;
                false
            }
        }
    };
    let mut mtp_rounds = 0usize;
    let result = if mtp {
        let mut first = true;
        model
            .generate_mtp_reuse(&request.ids, &run_options, true, &mut |ids, accepted| {
                if first {
                    first = false;
                } else {
                    mtp_rounds += 1;
                }
                on_event(EngineEvent::Tokens {
                    ids: ids.to_vec(),
                    accepted,
                })
            })
            .map(|output| spec::Output {
                ids: output.ids,
                rounds: Vec::new(),
                prefill_ms: output.prefill_ms,
                decode_ms: output.decode_ms,
            })
    } else if run_options.backend.draft_model.is_some() {
        spec::produce_events(
            model,
            &request.ids,
            request
                .images
                .as_ref()
                .map(|images| images.input(&request.ids))
                .as_ref(),
            &run_options,
            None,
            &mut |step| {
                on_event(match step {
                    Step::First(token) => EngineEvent::Tokens {
                        ids: vec![token],
                        accepted: 0,
                    },
                    Step::Round(round) => EngineEvent::Tokens {
                        ids: round.output.clone(),
                        accepted: round.accepted,
                    },
                })
            },
        )
    } else {
        produce_plain_events(
            model,
            &request.ids,
            request.images.as_ref(),
            &run_options,
            &mut |token| {
                on_event(EngineEvent::Tokens {
                    ids: vec![token],
                    accepted: 0,
                })
            },
        )
    };
    let result = result.and_then(|output| {
        model.check_ready()?;
        let mut record = serde_json::json!({"serve_request":{"prefill_ms":output.prefill_ms,"decode_ms":output.decode_ms,"initial_rounds":output.rounds.iter().take(2).collect::<Vec<_>>(),"mtp":mtp,"route":route,"mtp_rounds":mtp_rounds}});
        if std::env::var("FLYINGFISH_SERVE_TRACE_IDS").is_ok_and(|value| value == "1") {
            record["serve_request"]["generated_ids"] = serde_json::json!(output.ids);
            record["serve_request"]["aborted"] = serde_json::json!(aborted);
            record["serve_request"]["input_ids"] = serde_json::json!(request.ids);
            record["serve_request"]["image_input"] = serde_json::json!(input_dump);
        }
        eprintln!("{record}");
        Ok(output)
    });
    match result {
        Ok(output) => {
            if aborted {
                if !mtp {
                    let _ = model.reset();
                }
                let _ = events.blocking_send(EngineEvent::Aborted { ids: output.ids });
            } else {
                let eos_hit = !run_options.ignore_eos
                    && output
                        .ids
                        .last()
                        .is_some_and(|token| model.config.eos_token_id.contains(token));
                let length_reached = output.ids.len() >= run_options.count && !eos_hit;
                let _ = events.blocking_send(EngineEvent::Done {
                    ids: output.ids,
                    rounds: if mtp { mtp_rounds } else { output.rounds.len() },
                    length_reached,
                });
            }
        }
        Err(error) => {
            if !mtp {
                let _ = model.reset();
            }
            let _ = events.blocking_send(EngineEvent::Failed(error.to_string()));
        }
    }
}

fn trace_image_input(
    spec: &crate::model_spec::ModelSpec,
    request: &Request,
) -> Result<Option<std::path::PathBuf>> {
    if !std::env::var("FLYINGFISH_SERVE_TRACE_IDS").is_ok_and(|v| v == "1") {
        return Ok(None);
    }
    let (Some(images), Some(dir)) = (
        &request.images,
        std::env::var_os("FLYINGFISH_SERVE_TRACE_INPUT_DIR"),
    ) else {
        return Ok(None);
    };
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!(
        "image-input-{}-{sequence}.safetensors",
        std::process::id()
    ));
    images.save(spec, &request.ids, &path)?;
    Ok(Some(path))
}

fn check_request(
    ids: &[u32],
    options: &SpecOptions,
    draft: Option<&crate::dspark::DraftConfig>,
) -> Result<()> {
    ensure!(!ids.is_empty(), "prompt must not be empty");
    if options.backend.draft_model.is_some() {
        spec::check(
            ids,
            options,
            draft.context("loaded draft config is missing from request admission")?,
        )
    } else {
        crate::model::check_generation_capacity(ids.len(), options.count, options.capacity)
    }
}

fn produce_plain_events(
    model: &mut Model,
    ids: &[u32],
    images: Option<&image::PreparedInput>,
    options: &SpecOptions,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<spec::Output> {
    model.reset()?;
    let output = match images {
        Some(images) => crate::model::generate_multimodal(
            model,
            &images.input(ids),
            options.count,
            options.ignore_eos,
            true,
            None,
            on_token,
        )?,
        None => crate::model::generate_plain(
            model,
            ids,
            options.count,
            options.ignore_eos,
            true,
            None,
            on_token,
        )?,
    };
    model.synchronize()?;
    model.check_logits()?;
    model.check_ready()?;
    Ok(spec::Output {
        ids: output.ids,
        rounds: Vec::new(),
        decode_ms: output.decode_ms,
        prefill_ms: output.prefill_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_rejects_second_admission_then_recovers() {
        let mut slot = Slot::Idle;
        slot.admit().expect("first admission");
        let error = slot.admit().expect_err("second admission must fail");
        assert!(error.to_string().contains("busy"));
        slot = Slot::Idle;
        slot.admit().expect("admission after idle");
    }
    #[test]
    fn unavailable_engine_never_admits_a_new_request() {
        let mut slot = Slot::Unavailable;
        assert!(
            slot.admit()
                .expect_err("failed recovery stays unavailable")
                .to_string()
                .contains("unavailable")
        );
        assert_eq!(slot, Slot::Unavailable);
    }
}
