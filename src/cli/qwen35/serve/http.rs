use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    num::NonZeroUsize,
    sync::{Arc, mpsc},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub prompt: String,
    #[serde(default = "default_tokens")]
    pub max_new_tokens: NonZeroUsize,
}

fn default_tokens() -> NonZeroUsize {
    NonZeroUsize::new(128).unwrap()
}

#[derive(Debug)]
pub(super) struct Error {
    status: StatusCode,
    message: String,
}

impl Error {
    pub fn invalid(message: impl ToString) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.to_string(),
        }
    }

    pub fn failed(message: impl ToString) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }

    fn unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "model is busy or unavailable".into(),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

pub(super) trait Generator {
    fn generate(&mut self, input: Input) -> std::result::Result<Value, Error>;
}

struct Job {
    input: Input,
    reply: oneshot::Sender<std::result::Result<Value, Error>>,
    _permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(super) struct Worker {
    sender: mpsc::SyncSender<Job>,
    slot: Arc<Semaphore>,
}

impl Worker {
    pub async fn start<G: Generator + 'static>(
        load: impl FnOnce() -> Result<G> + Send + 'static,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel::<Job>(1);
        let (ready, loaded) = oneshot::channel();
        let slot = Arc::new(Semaphore::new(1));
        let worker_slot = slot.clone();
        std::thread::Builder::new()
            .name("qwen-inference".into())
            .spawn(move || {
                struct Close(Arc<Semaphore>);
                impl Drop for Close {
                    fn drop(&mut self) {
                        self.0.close();
                    }
                }
                let _close = Close(worker_slot);
                let mut engine = match load() {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready.send(Err(error));
                        return;
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return;
                }
                while let Ok(job) = receiver.recv() {
                    if job.reply.is_closed() {
                        continue;
                    }
                    let result = engine.generate(job.input);
                    let failed = result
                        .as_ref()
                        .is_err_and(|error| error.status.is_server_error());
                    if failed {
                        _close.0.close();
                    }
                    drop(job._permit);
                    let _ = job.reply.send(result);
                    if failed {
                        break;
                    }
                }
            })
            .context("start inference worker")?;
        loaded
            .await
            .context("inference worker exited during startup")??;
        Ok(Self { sender, slot })
    }

    async fn generate(&self, input: Input) -> std::result::Result<Value, Error> {
        let permit = self
            .slot
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::unavailable())?;
        let (reply, result) = oneshot::channel();
        self.sender
            .try_send(Job {
                input,
                reply,
                _permit: permit,
            })
            .map_err(|_| Error::unavailable())?;
        result.await.map_err(|_| Error::unavailable())?
    }
}

pub(super) fn router(worker: Worker) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(worker)
}

async fn health(State(worker): State<Worker>) -> Response {
    if worker.slot.is_closed() {
        Error::unavailable().into_response()
    } else {
        Json(json!({"status": "ready", "busy": worker.slot.available_permits() == 0}))
            .into_response()
    }
}

async fn generate(
    State(worker): State<Worker>,
    input: std::result::Result<Json<Input>, JsonRejection>,
) -> std::result::Result<Json<Value>, Error> {
    let Json(input) = input.map_err(|error| Error {
        status: error.status(),
        message: error.body_text(),
    })?;
    worker.generate(input).await.map(Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    struct Mock {
        started: Option<oneshot::Sender<()>>,
        release: mpsc::Receiver<()>,
    }

    impl Generator for Mock {
        fn generate(&mut self, input: Input) -> std::result::Result<Value, Error> {
            match input.prompt.as_str() {
                "hold" => {
                    let _ = self.started.take().unwrap().send(());
                    self.release.recv().unwrap();
                }
                "bad" => return Err(Error::invalid("too long")),
                "fail" => return Err(Error::failed("GPU failed")),
                _ => {}
            }
            Ok(json!({"tokens": input.max_new_tokens.get()}))
        }
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn input(prompt: &str) -> Input {
        Input {
            prompt: prompt.into(),
            max_new_tokens: default_tokens(),
        }
    }

    #[test]
    fn cancellation_keeps_inference_exclusive() {
        runtime().block_on(async {
            let (started, running) = oneshot::channel();
            let (release, wait) = mpsc::channel();
            let worker = Worker::start(move || {
                Ok(Mock {
                    started: Some(started),
                    release: wait,
                })
            })
            .await
            .unwrap();
            let request_worker = worker.clone();
            let request = tokio::spawn(async move { request_worker.generate(input("hold")).await });
            running.await.unwrap();
            assert_eq!(
                worker.generate(input("next")).await.unwrap_err().status,
                StatusCode::SERVICE_UNAVAILABLE
            );
            request.abort();
            let _ = request.await;
            assert_eq!(health(State(worker.clone())).await.status(), StatusCode::OK);
            assert_eq!(
                worker.generate(input("next")).await.unwrap_err().status,
                StatusCode::SERVICE_UNAVAILABLE
            );
            release.send(()).unwrap();
            let permit =
                tokio::time::timeout(std::time::Duration::from_secs(5), worker.slot.acquire())
                    .await
                    .unwrap()
                    .unwrap();
            drop(permit);
            assert!(worker.generate(input("next")).await.is_ok());
        });
    }

    #[test]
    fn validation_preserves_worker_and_gpu_failure_closes_it() {
        runtime().block_on(async {
            let (_release, wait) = mpsc::channel();
            let worker = Worker::start(move || {
                Ok(Mock {
                    started: None,
                    release: wait,
                })
            })
            .await
            .unwrap();
            assert_eq!(
                worker.generate(input("bad")).await.unwrap_err().status,
                StatusCode::BAD_REQUEST
            );
            assert!(worker.generate(input("next")).await.is_ok());
            assert_eq!(
                worker.generate(input("fail")).await.unwrap_err().status,
                StatusCode::INTERNAL_SERVER_ERROR
            );
            assert_eq!(
                health(State(worker.clone())).await.status(),
                StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(
                worker.generate(input("next")).await.unwrap_err().status,
                StatusCode::SERVICE_UNAVAILABLE
            );
        });
    }

    #[test]
    fn http_rejects_invalid_and_oversized_payloads() {
        runtime().block_on(async {
            let (_release, wait) = mpsc::channel();
            let worker = Worker::start(move || {
                Ok(Mock {
                    started: None,
                    release: wait,
                })
            })
            .await
            .unwrap();
            let app = router(worker);
            for (body, status) in [
                (r#"{"prompt":"ok"}"#.to_owned(), StatusCode::OK),
                (
                    r#"{"prompt":"ok","max_new_tokens":0}"#.to_owned(),
                    StatusCode::UNPROCESSABLE_ENTITY,
                ),
                (
                    r#"{"prompt":"ok","stream":true}"#.to_owned(),
                    StatusCode::UNPROCESSABLE_ENTITY,
                ),
                ("{".to_owned(), StatusCode::BAD_REQUEST),
                (" ".repeat(1024 * 1024 + 1), StatusCode::PAYLOAD_TOO_LARGE),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::post("/generate")
                            .header("content-type", "application/json")
                            .body(Body::from(body))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
            }
        });
    }

    #[test]
    fn startup_failure_is_returned() {
        runtime().block_on(async {
            let result = Worker::start(|| -> Result<Mock> { anyhow::bail!("load failed") }).await;
            assert_eq!(result.err().unwrap().to_string(), "load failed");
        });
    }
}
