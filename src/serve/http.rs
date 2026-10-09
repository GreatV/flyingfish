use super::detok::Shared;
use super::{Backend, EngineEvent, Request};
use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream::{StreamExt, unfold};
use minijinja::{Environment, Value};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::sync::mpsc as tokio_mpsc;

struct AppState {
    backend: Backend,
    shared: Shared,
    env: Environment<'static>,
    model_name: String,
}

pub fn serve(
    backend: Backend,
    model_dir: std::path::PathBuf,
    port: u16,
    engine: JoinHandle<()>,
) -> Result<()> {
    let shared = Shared::load(&model_dir)?;
    let template_text = read_chat_template(&model_dir)?;
    let mut env = Environment::new();
    env.set_unknown_method_callback(minicp_string_methods);
    env.add_template_owned("chat", template_text)
        .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))
        .context("serve init")?;
    let model_name = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "flyingfish".into());
    let state = Arc::new(AppState {
        backend,
        shared,
        env,
        model_name,
    });
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
        .context("tokio runtime")?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::from_std(listener)?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
    })?;
    drop(runtime);
    engine
        .join()
        .map_err(|_| anyhow::anyhow!("engine thread panicked"))?;
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("install ctrl-c handler");
}

/// String methods the MiniCPM5-2B chat template calls that core minijinja
/// does not provide; anything else stays an unknown-method error.
fn minicp_string_methods(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, minijinja::Error> {
    let text = value.as_str().ok_or_else(|| {
        minijinja::Error::new(minijinja::ErrorKind::UnknownMethod, "not a string")
    })?;
    match (method, args.first().and_then(|a| a.as_str())) {
        ("startswith", Some(prefix)) => Ok(Value::from(text.starts_with(prefix))),
        ("endswith", Some(suffix)) => Ok(Value::from(text.ends_with(suffix))),
        _ => Err(minijinja::Error::new(
            minijinja::ErrorKind::UnknownMethod,
            format!("string has no method named {method}"),
        )),
    }
}

fn read_chat_template(model_dir: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(model_dir.join("chat_template.jinja"))
        .context("chat_template.jinja missing or unreadable in the model directory")
}

fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

fn busy_response() -> Response {
    (
        StatusCode::CONFLICT,
        Json(json!({"error": {"message": "busy: a request is already running", "type": "server_busy"}})),
    )
        .into_response()
}

fn error_response(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": message, "type": "server_error"}})),
    )
        .into_response()
}

#[derive(Deserialize)]
struct CommonParams {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<serde_json::Value>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct CompletionsBody {
    #[serde(default)]
    prompt: Option<serde_json::Value>,
    #[serde(flatten)]
    params: CommonParams,
}

#[derive(Deserialize)]
struct ChatBody {
    #[serde(default)]
    messages: Option<Vec<serde_json::Value>>,
    #[serde(flatten)]
    params: CommonParams,
}

fn check_greedy(params: &CommonParams) -> Result<(), Box<Response>> {
    if params.temperature.is_some_and(|t| t != 0.0) {
        return Err(Box::new(bad_request(
            "greedy only: temperature must be 0 or absent",
        )));
    }
    if params.top_p.is_some_and(|p| p != 1.0) {
        return Err(Box::new(bad_request(
            "greedy only: top_p must be 1 or absent",
        )));
    }
    if params.top_k.is_some() {
        return Err(Box::new(bad_request("greedy only: top_k is not supported")));
    }
    if params.n.is_some_and(|n| n != 1) {
        return Err(Box::new(bad_request("greedy only: n must be 1")));
    }
    if params.logit_bias.is_some() {
        return Err(Box::new(bad_request(
            "greedy only: logit_bias is not supported",
        )));
    }
    if params.presence_penalty.is_some_and(|p| p != 0.0) {
        return Err(Box::new(bad_request(
            "greedy only: presence_penalty must be 0 or absent",
        )));
    }
    if params.frequency_penalty.is_some_and(|p| p != 0.0) {
        return Err(Box::new(bad_request(
            "greedy only: frequency_penalty must be 0 or absent",
        )));
    }
    Ok(())
}

fn prompt_to_ids(
    state: &AppState,
    prompt: Option<serde_json::Value>,
) -> Result<Vec<u32>, Box<Response>> {
    match prompt {
        Some(serde_json::Value::String(text)) => state
            .shared
            .encode(&text)
            .map_err(|e| Box::new(bad_request(&format!("tokenize: {e}")))),
        Some(serde_json::Value::Array(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items {
                let id = item.as_u64().ok_or_else(|| {
                    Box::new(bad_request("prompt arrays must contain only token ids"))
                })?;
                ids.push(u32::try_from(id).map_err(|_| bad_request("prompt id exceeds u32"))?);
            }
            Ok(ids)
        }
        _ => Err(Box::new(bad_request(
            "prompt must be a string or a flat array of token ids",
        ))),
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn add_request(
    state: &AppState,
    ids: Vec<u32>,
    params: &CommonParams,
) -> Result<tokio_mpsc::Receiver<EngineEvent>, Box<Response>> {
    let (event_tx, event_rx) = tokio_mpsc::channel::<EngineEvent>(64);
    let request = Request {
        ids,
        max_new_tokens: params.max_tokens.unwrap_or(256),
    };
    state
        .backend
        .add_request(request, event_tx)
        .map_err(|error| {
            let message = error.to_string();
            if message.contains("busy") {
                Box::new(busy_response())
            } else {
                Box::new(error_response(message))
            }
        })?;
    Ok(event_rx)
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CompletionsBody>,
) -> Response {
    if let Err(response) = check_greedy(&body.params) {
        return *response;
    }
    let ids = match prompt_to_ids(&state, body.prompt) {
        Ok(ids) => ids,
        Err(response) => return *response,
    };
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, &body.params) {
        Ok(events) => events,
        Err(response) => return *response,
    };
    let id = format!("cmpl-{}", now_secs());
    respond(
        events,
        state,
        id,
        false,
        body.params.stream.unwrap_or(false),
        prompt_tokens,
    )
    .await
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatBody>,
) -> Response {
    if let Err(response) = check_greedy(&body.params) {
        return *response;
    }
    let messages = match body.messages {
        Some(messages) if !messages.is_empty() => messages,
        _ => return bad_request("messages must be a non-empty array"),
    };
    for message in &messages {
        if message.get("role").and_then(|r| r.as_str()).is_none()
            || message.get("content").and_then(|c| c.as_str()).is_none()
        {
            return bad_request("each message needs string role and content");
        }
    }
    let rendered = match state
        .env
        .get_template("chat")
        .and_then(|template| {
            template.render(minijinja::context! {
                messages => messages,
                add_generation_prompt => true,
            })
        }) {
        Ok(rendered) => rendered,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": {"message": format!("chat template render: {error}"), "type": "invalid_request_error"}})),
            )
                .into_response()
        }
    };
    let ids = match state.shared.encode(&rendered) {
        Ok(ids) => ids,
        Err(e) => return bad_request(&format!("tokenize: {e}")),
    };
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, &body.params) {
        Ok(events) => events,
        Err(response) => return *response,
    };
    let id = format!("chatcmpl-{}", now_secs());
    respond(
        events,
        state,
        id,
        true,
        body.params.stream.unwrap_or(false),
        prompt_tokens,
    )
    .await
}

async fn respond(
    mut events: tokio_mpsc::Receiver<EngineEvent>,
    state: Arc<AppState>,
    id: String,
    is_chat: bool,
    stream: bool,
    prompt_tokens: usize,
) -> Response {
    if stream {
        let model = state.model_name.clone();
        let mut detok = state.shared.detok();
        let sse = unfold(events, |mut events| async move {
            events.recv().await.map(|event| (event, events))
        })
        .map(move |event| sse_event(&event, &model, &id, is_chat, &mut detok));
        return Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    let mut detok = state.shared.detok();
    let mut finish_reason = "stop";
    while let Some(event) = events.recv().await {
        match event {
            EngineEvent::Tokens { ids, .. } => {
                if detok.push(&ids).is_err() {
                    return error_response("detokenize failed".into());
                }
            }
            EngineEvent::Done { length_reached, .. } => {
                finish_reason = if length_reached { "length" } else { "stop" };
                break;
            }
            EngineEvent::Aborted { .. } => {
                finish_reason = "aborted";
                break;
            }
            EngineEvent::Failed(message) => return error_response(message),
        }
    }
    let completion_tokens = detok.ids().len();
    let text = detok.text().to_string();
    let body = if is_chat {
        json!({
            "id": id,
            "object": "chat.completion",
            "created": now_secs(),
            "model": state.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
        })
    } else {
        json!({
            "id": id,
            "object": "text_completion",
            "created": now_secs(),
            "model": state.model_name,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
        })
    };
    Json(body).into_response()
}

fn sse_event(
    event: &EngineEvent,
    model: &str,
    id: &str,
    is_chat: bool,
    detok: &mut super::detok::Detok,
) -> Result<Event, axum::Error> {
    match event {
        EngineEvent::Tokens { ids, .. } => {
            let text = detok.push(ids).unwrap_or_default();
            sse_json(&chunk_payload(model, id, is_chat, &text, None))
        }
        EngineEvent::Done { .. } => sse_json(&chunk_payload(model, id, is_chat, "", Some("stop"))),
        EngineEvent::Aborted { .. } => {
            sse_json(&chunk_payload(model, id, is_chat, "", Some("aborted")))
        }
        EngineEvent::Failed(message) => sse_json(&json!({
            "error": {"message": message, "type": "server_error"}
        })),
    }
}

fn sse_json(payload: &serde_json::Value) -> Result<Event, axum::Error> {
    Ok(Event::default().data(payload.to_string()))
}

fn chunk_payload(
    model: &str,
    id: &str,
    is_chat: bool,
    text: &str,
    finish: Option<&str>,
) -> serde_json::Value {
    if is_chat {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]
        })
    } else {
        json!({
            "id": id,
            "object": "text_completion.chunk",
            "created": now_secs(),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}]
        })
    }
}
