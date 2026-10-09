use super::detok::Shared;
use super::{Backend, EngineEvent, Request};
use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::stream::unfold;
use minijinja::{Environment, Value};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use tokio::sync::mpsc as tokio_mpsc;

struct AppState {
    backend: Backend,
    shared: Shared,
    env: Environment<'static>,
    model_name: String,
    bos_token: String,
    request_counter: AtomicU64,
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
    env.set_unknown_method_callback(template_string_methods);
    env.add_template_owned("chat", template_text)
        .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))
        .context("serve init")?;
    let model_name = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "flyingfish".into());
    let bos_token = read_bos_token(&model_dir)?;
    let state = Arc::new(AppState {
        backend,
        shared,
        env,
        model_name,
        bos_token,
        request_counter: AtomicU64::new(0),
    });
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    listener
        .set_nonblocking(true)
        .context("listener nonblocking")?;
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

fn read_bos_token(model_dir: &std::path::Path) -> Result<String> {
    let tok_config: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(model_dir.join("tokenizer_config.json"))
            .with_context(|| format!("open {}", model_dir.display()))?,
    )
    .context("parse tokenizer_config.json")?;
    tok_config
        .get("bos_token")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .context("tokenizer_config.json has no bos_token")
}

/// String methods the MiniCPM5-2B chat template calls that core minijinja
/// does not provide: lstrip, rstrip, strip, split, replace, startswith,
/// endswith. Anything else stays an unknown-method error.
fn template_string_methods(
    _state: &minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, minijinja::Error> {
    let unknown = |name: &str| {
        minijinja::Error::new(
            minijinja::ErrorKind::UnknownMethod,
            format!("value has no method named {name}"),
        )
    };
    if let Some(text) = value.as_str() {
        let arg0 = args.first().and_then(|a| a.as_str());
        match method {
            "startswith" => {
                let prefix = arg0.ok_or_else(|| unknown("startswith"))?;
                return Ok(Value::from(text.starts_with(prefix)));
            }
            "endswith" => {
                let suffix = arg0.ok_or_else(|| unknown("endswith"))?;
                return Ok(Value::from(text.ends_with(suffix)));
            }
            "strip" => match arg0 {
                Some(chars) => return Ok(Value::from(text.trim_matches(|c| chars.contains(c)))),
                None => return Ok(Value::from(text.trim())),
            },
            "lstrip" => match arg0 {
                Some(chars) => {
                    return Ok(Value::from(text.trim_start_matches(|c| chars.contains(c))));
                }
                None => return Ok(Value::from(text.trim_start())),
            },
            "rstrip" => match arg0 {
                Some(chars) => {
                    return Ok(Value::from(text.trim_end_matches(|c| chars.contains(c))));
                }
                None => return Ok(Value::from(text.trim_end())),
            },
            "replace" => {
                let from = arg0.ok_or_else(|| unknown("replace"))?;
                let to = args.get(1).and_then(|a| a.as_str()).unwrap_or("");
                return Ok(Value::from(text.replace(from, to)));
            }
            "split" => {
                let sep = arg0.ok_or_else(|| unknown("split"))?;
                let parts: Vec<Value> = text.split(sep).map(Value::from).collect();
                return Ok(Value::from(parts));
            }
            _ => return Err(unknown(method)),
        }
    }
    Err(unknown(method))
}

/// Message roles the MiniCPM5-2B chat template renders.
const TEMPLATE_ROLES: [&str; 4] = ["system", "user", "assistant", "tool"];

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
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stop: Option<serde_json::Value>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    #[serde(flatten)]
    unrecognized: serde_json::Map<String, serde_json::Value>,
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
    if params.stop.is_some() {
        return Err(Box::new(bad_request("stop sequences are not supported")));
    }
    Ok(())
}

/// Rejects a model name other than the loaded model and any non-null
/// parameter the server does not recognize.
fn check_request(state: &AppState, params: &CommonParams) -> Result<(), Box<Response>> {
    if let Some(model) = &params.model
        && *model != state.model_name
    {
        return Err(Box::new(
            (
                StatusCode::NOT_FOUND,
                Json(json!({"error": {
                    "message": format!("model {model:?} is not served; the loaded model is {:?}", state.model_name),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }})),
            )
                .into_response(),
        ));
    }
    if let Some((name, _)) = params.unrecognized.iter().find(|(_, v)| !v.is_null()) {
        return Err(Box::new(bad_request(&format!(
            "unsupported parameter {name:?}"
        ))));
    }
    Ok(())
}

/// Resolve max_tokens / max_completion_tokens; both present and unequal is
/// an explicit 400.
fn resolve_max_tokens(params: &CommonParams) -> Result<usize, Box<Response>> {
    match (params.max_tokens, params.max_completion_tokens) {
        (Some(a), Some(b)) if a != b => Err(Box::new(bad_request(
            "max_tokens and max_completion_tokens must agree when both present",
        ))),
        (a, b) => Ok(a.or(b).unwrap_or(256)),
    }
}

fn check_validity(state: &AppState, ids: &[u32], max_tokens: usize) -> Result<(), Box<Response>> {
    let vocab_size = state.backend.vocab_size();
    if let Some(id) = ids.iter().find(|&&id| id as usize >= vocab_size) {
        return Err(Box::new(bad_request(&format!(
            "prompt token id {id} is outside the vocabulary of size {vocab_size}"
        ))));
    }
    let options = crate::spec::Options {
        count: max_tokens,
        ..state.backend.options()
    };
    crate::spec::check(ids, &options)
        .map_err(|error| Box::new(bad_request(&format!("request rejected: {error}"))))
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
                ids.push(
                    u32::try_from(id)
                        .map_err(|_| Box::new(bad_request("prompt id exceeds u32")))?,
                );
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

fn next_request_id(state: &AppState, prefix: &str) -> String {
    let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{seq}", now_secs())
}

fn add_request(
    state: &AppState,
    ids: Vec<u32>,
    max_tokens: usize,
) -> Result<tokio_mpsc::Receiver<EngineEvent>, Box<Response>> {
    let (event_tx, event_rx) = tokio_mpsc::channel::<EngineEvent>(64);
    let request = Request {
        ids,
        max_new_tokens: max_tokens,
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
    if let Err(response) = check_request(&state, &body.params) {
        return *response;
    }
    let ids = match prompt_to_ids(&state, body.prompt) {
        Ok(ids) => ids,
        Err(response) => return *response,
    };
    let max_tokens = match resolve_max_tokens(&body.params) {
        Ok(v) => v,
        Err(response) => return *response,
    };
    if let Err(response) = check_validity(&state, &ids, max_tokens) {
        return *response;
    }
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, max_tokens) {
        Ok(events) => events,
        Err(response) => return *response,
    };
    let id = next_request_id(&state, "cmpl");
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
    if let Err(response) = check_request(&state, &body.params) {
        return *response;
    }
    let messages = match body.messages {
        Some(messages) if !messages.is_empty() => messages,
        _ => return bad_request("messages must be a non-empty array"),
    };
    for message in &messages {
        let Some(role) = message.get("role").and_then(|r| r.as_str()) else {
            return bad_request("each message needs string role and content");
        };
        if message.get("content").and_then(|c| c.as_str()).is_none() {
            return bad_request("each message needs string role and content");
        }
        if !TEMPLATE_ROLES.contains(&role) {
            return bad_request(&format!(
                "role {role:?} is not rendered by the chat template; expected one of {TEMPLATE_ROLES:?}"
            ));
        }
    }
    let max_tokens = match resolve_max_tokens(&body.params) {
        Ok(v) => v,
        Err(response) => return *response,
    };
    let rendered = match state
        .env
        .get_template("chat")
        .and_then(|template| {
            template.render(minijinja::context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => state.bos_token,
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
    let ids = match state.shared.encode_chat(&rendered) {
        Ok(ids) => ids,
        Err(e) => return bad_request(&format!("tokenize: {e}")),
    };
    if let Err(response) = check_validity(&state, &ids, max_tokens) {
        return *response;
    }
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, max_tokens) {
        Ok(events) => events,
        Err(response) => return *response,
    };
    let id = next_request_id(&state, "chatcmpl");
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
        let detok = state.shared.detok();
        let sse = unfold(
            StreamState::Running(events, Box::new(detok)),
            move |stream_state| {
                let model = model.clone();
                let id = id.clone();
                async move {
                    match stream_state {
                        StreamState::Running(mut events, mut detok) => match events.recv().await {
                            Some(event) => {
                                let item = sse_event(&event, &model, &id, is_chat, &mut detok);
                                let next = match event {
                                    EngineEvent::Tokens { .. } => {
                                        StreamState::Running(events, detok)
                                    }
                                    EngineEvent::Done { .. } | EngineEvent::Aborted { .. } => {
                                        StreamState::Sentinel
                                    }
                                    EngineEvent::Failed(_) => StreamState::End,
                                };
                                Some((item, next))
                            }
                            None => Some((
                                sse_json(&json!({"error": {
                                    "message": ENGINE_EOF,
                                    "type": "server_error"
                                }})),
                                StreamState::End,
                            )),
                        },
                        StreamState::Sentinel => {
                            Some((Ok(Event::default().data("[DONE]")), StreamState::End))
                        }
                        StreamState::End => None,
                    }
                }
            },
        );
        return Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    let mut detok = state.shared.detok();
    let finish_reason = loop {
        match events.recv().await {
            Some(EngineEvent::Tokens { ids, .. }) => {
                if detok.push(&ids).is_err() {
                    return error_response("detokenize failed".into());
                }
            }
            Some(EngineEvent::Done { length_reached, .. }) => {
                break if length_reached { "length" } else { "stop" };
            }
            Some(EngineEvent::Aborted { .. }) => break "aborted",
            Some(EngineEvent::Failed(message)) => return error_response(message),
            None => return error_response(ENGINE_EOF.into()),
        }
    };
    if detok.finish().is_err() {
        return error_response("detokenize failed".into());
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

const ENGINE_EOF: &str = "engine stopped before completing the request";

enum StreamState {
    Running(tokio_mpsc::Receiver<EngineEvent>, Box<super::detok::Detok>),
    Sentinel,
    End,
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
        EngineEvent::Done { length_reached, .. } => {
            let text = detok.finish().unwrap_or_default();
            let finish = if *length_reached { "length" } else { "stop" };
            sse_json(&chunk_payload(model, id, is_chat, &text, Some(finish)))
        }
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
            "object": "text_completion",
            "created": now_secs(),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}]
        })
    }
}
