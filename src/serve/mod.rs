pub mod detok;
pub mod http;

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
}

impl Slot {
    fn admit(&mut self) -> Result<()> {
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
}

impl Backend {
    pub fn add_request(
        &self,
        request: Request,
        events: tokio_mpsc::Sender<EngineEvent>,
    ) -> Result<()> {
        self.slot.lock().expect("slot mutex poisoned").admit()?;
        match self
            .commands
            .send(EngineCommand::Generate { request, events })
        {
            Ok(()) => Ok(()),
            Err(error) => {
                *self.slot.lock().expect("slot mutex poisoned") = Slot::Idle;
                Err(anyhow::Error::from(error)).context("engine thread has exited")
            }
        }
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

fn load_model(target: &Path, draft: Option<&Path>, options: &SpecOptions) -> Result<Model> {
    let mut settings = options.backend.clone();
    if let Some(draft) = draft {
        settings = settings.with_draft(draft);
    }
    Model::load(
        target,
        options.device,
        options.capacity,
        options.chunk,
        settings,
    )
}

pub fn run(
    target: PathBuf,
    draft: Option<PathBuf>,
    options: SpecOptions,
    tokenizer_dir: PathBuf,
    port: u16,
) -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel::<EngineCommand>();
    let slot = Arc::new(Mutex::new(Slot::Idle));
    let (ready_tx, ready_rx) = mpsc::channel::<Result<usize>>();
    let engine_slot = slot.clone();
    let engine_options = options.clone();
    // Model is loaded inside the engine thread: it holds CUDA handles
    // that are not Send, and the engine thread is their only user. The
    // readiness channel carries the load result and the model vocabulary
    // size to `serve` before it binds, so a failed load fails startup.
    let handle = thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let model = match load_model(&target, draft.as_deref(), &engine_options) {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(model.config.vocab_size));
            engine_loop(model, engine_options, command_rx, engine_slot)
        })
        .context("spawn engine thread")?;
    let vocab_size = ready_rx
        .recv()
        .context("engine thread exited before reporting readiness")??;
    let backend = Backend {
        commands: command_tx,
        slot,
        options,
        vocab_size,
    };
    http::serve(backend, tokenizer_dir, port, handle)
}

fn engine_loop(
    mut model: Model,
    options: SpecOptions,
    commands: mpsc::Receiver<EngineCommand>,
    slot: Arc<Mutex<Slot>>,
) {
    while let Ok(command) = commands.recv() {
        match command {
            EngineCommand::Generate { request, events } => {
                generate_once(&mut model, &options, request, events, &commands);
            }
            EngineCommand::Abort => {}
        }
        *slot.lock().expect("slot mutex poisoned") = Slot::Idle;
    }
}

fn generate_once(
    model: &mut Model,
    options: &SpecOptions,
    request: Request,
    events: tokio_mpsc::Sender<EngineEvent>,
    commands: &mpsc::Receiver<EngineCommand>,
) {
    let run_options = SpecOptions {
        count: request.max_new_tokens,
        ..options.clone()
    };
    if let Err(error) = spec::check(&request.ids, &run_options) {
        let _ = events.blocking_send(EngineEvent::Failed(error.to_string()));
        return;
    }
    let mut aborted = false;
    let result = spec::produce_events(model, &request.ids, &run_options, None, &mut |step| {
        let event = match step {
            Step::First(token) => EngineEvent::Tokens {
                ids: vec![token],
                accepted: 0,
            },
            Step::Round(round) => EngineEvent::Tokens {
                ids: round.output.clone(),
                accepted: round.accepted,
            },
        };
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
    });
    match result {
        Ok(output) => {
            if aborted {
                let _ = model.reset();
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
                    rounds: output.rounds.len(),
                    length_reached,
                });
            }
        }
        Err(error) => {
            let _ = model.reset();
            let _ = events.blocking_send(EngineEvent::Failed(error.to_string()));
        }
    }
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
}
