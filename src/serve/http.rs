use super::detok::Shared;
use super::{Backend, EngineEvent, Request};
use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, State};
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
    image_processor: Option<super::image::Processor>,
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
    env.add_function("raise_exception", template_exception);
    env.add_template_owned("chat", template_text)
        .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))
        .context("serve init")?;
    let model_name = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "flyingfish".into());
    let bos_token = read_bos_token(&model_dir)?;
    let image_processor = super::image::Processor::load(&model_dir, Some(&shared))?;
    let state = Arc::new(AppState {
        backend,
        shared,
        env,
        model_name,
        bos_token,
        request_counter: AtomicU64::new(0),
        image_processor,
    });
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(128 * 1024 * 1024))
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
    match tok_config.get("bos_token") {
        None | Some(serde_json::Value::Null) => Ok(String::new()),
        Some(serde_json::Value::String(token)) => Ok(token.clone()),
        _ => anyhow::bail!("tokenizer_config.json bos_token must be a string or null"),
    }
}

fn template_exception(message: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        message,
    ))
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
    #[serde(default)]
    chat_template_kwargs: ChatTemplateOptions,
    #[serde(flatten)]
    params: CommonParams,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTemplateOptions {
    enable_thinking: Option<bool>,
}

fn render_chat(
    env: &Environment<'_>,
    messages: &[serde_json::Value],
    bos_token: &str,
    options: &ChatTemplateOptions,
) -> Result<String, minijinja::Error> {
    let mut context = json!({
        "messages": messages,
        "add_generation_prompt": true,
        "bos_token": bos_token,
    });
    if let Some(thinking) = options.enable_thinking {
        context["enable_thinking"] = json!(thinking);
    }
    env.get_template("chat")?.render(context)
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

fn check_validity(
    state: &AppState,
    ids: &[u32],
    max_tokens: usize,
    image: bool,
) -> Result<(), Box<Response>> {
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
    (if state.backend.mtp && !image {
        crate::mtp::check_request(ids.len(), max_tokens, options.capacity)
    } else {
        super::check_request(ids, &options, state.backend.draft_config.as_ref())
    })
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
    images: Option<super::image::PreparedInput>,
    admission: Option<super::Admission<'_>>,
) -> Result<tokio_mpsc::Receiver<EngineEvent>, Box<Response>> {
    let (event_tx, event_rx) = tokio_mpsc::channel::<EngineEvent>(64);
    let request = Request {
        ids,
        max_new_tokens: max_tokens,
        images,
    };
    let result = match admission {
        Some(admission) => admission.send(request, event_tx),
        None => state.backend.add_request(request, event_tx),
    };
    result.map_err(admission_error)?;
    Ok(event_rx)
}

fn admission_error(error: anyhow::Error) -> Box<Response> {
    let message = error.to_string();
    if message.contains("busy") {
        Box::new(busy_response())
    } else {
        Box::new(error_response(message))
    }
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
    if let Err(response) = check_validity(&state, &ids, max_tokens, false) {
        return *response;
    }
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, max_tokens, None, None) {
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
    let mut messages = match body.messages {
        Some(messages) if !messages.is_empty() => messages,
        _ => return bad_request("messages must be a non-empty array"),
    };
    for message in &messages {
        let Some(role) = message.get("role").and_then(|r| r.as_str()) else {
            return bad_request("each message needs string role and content");
        };
        if !message
            .get("content")
            .is_some_and(|c| c.is_string() || c.is_array())
        {
            return bad_request("message content must be a string or content-parts array");
        }
        if !TEMPLATE_ROLES.contains(&role) {
            return bad_request(&format!(
                "role {role:?} is not rendered by the chat template; expected one of {TEMPLATE_ROLES:?}"
            ));
        }
    }
    let (images, videos) = match prepare_message_media(&state, &mut messages).await {
        Ok(media) => media,
        Err(error) => return bad_request(&format!("media input: {error}")),
    };
    let max_tokens = match resolve_max_tokens(&body.params) {
        Ok(v) => v,
        Err(response) => return *response,
    };
    let rendered = match render_chat(
        &state.env,
        &messages,
        &state.bos_token,
        &body.chat_template_kwargs,
    ) {
        Ok(rendered) => rendered,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": {"message": format!("chat template render: {error}"), "type": "invalid_request_error"}})),
            )
                .into_response()
        }
    };
    let mut ids = match state.shared.encode_chat(&rendered) {
        Ok(ids) => ids,
        Err(e) => return bad_request(&format!("tokenize: {e}")),
    };
    let mut admission = None;
    let images = if images.is_empty() && videos.is_empty() {
        None
    } else {
        let Some(processor) = &state.image_processor else {
            return bad_request("this model does not support image input");
        };
        let expanded_tokens = match processor.expanded_token_count(&ids, &images, &videos) {
            Ok(count) => count,
            Err(error) => return bad_request(&format!("media input: {error}")),
        };
        if let Err(error) = crate::model::check_generation_capacity(
            expanded_tokens,
            max_tokens,
            state.backend.options().capacity,
        ) {
            return bad_request(&format!("media input: {error}"));
        }
        admission = match state.backend.reserve() {
            Ok(reservation) => Some(reservation),
            Err(error) => return *admission_error(error),
        };
        let images = match images
            .into_iter()
            .map(|image| processor.prepare_planned(image))
            .collect::<Result<Vec<_>>>()
        {
            Ok(images) => images,
            Err(error) => return bad_request(&format!("image input: {error}")),
        };
        let mut prepared_videos = Vec::with_capacity(videos.len());
        for video in videos {
            let processor = processor.clone();
            match tokio::task::spawn_blocking(move || processor.prepare_planned_video(video)).await
            {
                Ok(Ok(video)) => prepared_videos.push(video),
                Ok(Err(error)) => return bad_request(&format!("video input: {error}")),
                Err(error) => {
                    return bad_request(&format!("video input: preparation task failed: {error}"));
                }
            }
        }
        let videos = prepared_videos;
        match processor.expand(&ids, images, videos, state.backend.options().capacity) {
            Ok((expanded, prepared)) => {
                ids = expanded;
                Some(prepared)
            }
            Err(error) => return bad_request(&format!("media input: {error}")),
        }
    };
    if let Err(response) = check_validity(&state, &ids, max_tokens, images.is_some()) {
        return *response;
    }
    let prompt_tokens = ids.len();
    let events = match add_request(&state, ids, max_tokens, images, admission) {
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

async fn prepare_message_media(
    state: &AppState,
    messages: &mut [serde_json::Value],
) -> Result<(
    Vec<super::image::PlannedImage>,
    Vec<super::image::PlannedVideo>,
)> {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    let mut image_tokens = 0usize;
    let mut video_tokens = 0usize;
    for message in messages {
        let is_system = message["role"] == "system";
        let Some(parts) = message.get_mut("content").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        for part in parts.iter_mut() {
            match part.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if part.get("text").and_then(|v| v.as_str()).is_none() {
                        anyhow::bail!("text content part needs a string text field");
                    }
                }
                Some("image_url") => {
                    anyhow::ensure!(!is_system, "system messages cannot contain images");
                    let processor = state
                        .image_processor
                        .as_ref()
                        .context("this model does not support image input")?;
                    let url = part
                        .get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(|v| v.as_str())
                        .context("image_url needs a string url")?;
                    let image = processor.plan_url(url)?;
                    image_tokens = image_tokens
                        .checked_add(processor.image_tokens(&image, images.len())?)
                        .context("image token count overflow")?;
                    anyhow::ensure!(
                        image_tokens <= state.backend.options().capacity,
                        "image tokens exceed request capacity before pixel preprocessing"
                    );
                    images.push(image);
                    *part = json!({"type":"image"});
                }
                Some("video_url") => {
                    anyhow::ensure!(!is_system, "system messages cannot contain videos");
                    anyhow::ensure!(
                        state.backend.options().backend.draft_model.is_none(),
                        "video input does not support a draft model"
                    );
                    let object = part
                        .get("video_url")
                        .and_then(|v| v.as_object())
                        .context("video_url needs an object")?;
                    for key in object.keys() {
                        anyhow::ensure!(
                            matches!(
                                key.as_str(),
                                "url" | "max_num_frames" | "stack_frames" | "downsample_mode"
                            ),
                            "unknown video_url field {key}"
                        );
                    }
                    let url = object
                        .get("url")
                        .and_then(|v| v.as_str())
                        .context("video_url needs a string url")?;
                    // Video processor class default and the model card pass 128
                    // (video_processing_minicpmv4_6.py:99, README.md:213/250).
                    let max_num_frames = match object.get("max_num_frames") {
                        None => 128,
                        Some(value) => usize::try_from(
                            value
                                .as_u64()
                                .context("max_num_frames must be a positive integer")?,
                        )
                        .context("max_num_frames exceeds usize")?,
                    };
                    let stack_frames = match object.get("stack_frames") {
                        None => 1,
                        Some(value) => u32::try_from(
                            value
                                .as_u64()
                                .context("stack_frames must be a positive integer")?,
                        )
                        .context("stack_frames exceeds u32")?,
                    };
                    anyhow::ensure!(
                        stack_frames == 1,
                        "stack_frames {stack_frames} is not supported; first support requires stack_frames = 1"
                    );
                    let downsample = match object.get("downsample_mode") {
                        None => "16x",
                        Some(value) => {
                            value.as_str().context("downsample_mode must be a string")?
                        }
                    };
                    anyhow::ensure!(
                        downsample == "16x",
                        "downsample_mode {downsample:?} is not supported; the model uses 16x"
                    );
                    let processor = state
                        .image_processor
                        .as_ref()
                        .context("this model does not support video input")?
                        .clone();
                    // max_slice_nums = 1: the model card's video call (README.md:214); the
                    // processor class default is 9 (video_processing_minicpmv4_6.py:92).
                    let options = super::image::VideoOptions {
                        max_num_frames,
                        stack_frames,
                        max_slice_nums: Some(1),
                    };
                    let url = url.to_owned();
                    let capacity = state.backend.options().capacity;
                    let planning = processor.clone();
                    let video = tokio::task::spawn_blocking(move || {
                        planning.plan_video(&url, options, capacity)
                    })
                    .await
                    .context("video planning task failed")??;
                    video_tokens = video_tokens
                        .checked_add(processor.video_tokens(&video)?)
                        .context("video token count overflow")?;
                    anyhow::ensure!(
                        image_tokens + video_tokens <= state.backend.options().capacity,
                        "video tokens exceed request capacity before pixel preprocessing"
                    );
                    videos.push(video);
                    *part = json!({"type":"video"});
                }
                _ => anyhow::bail!("content part type must be text, image_url or video_url"),
            }
        }
        if parts.iter().all(|p| p["type"] == "text") {
            let text = parts
                .iter()
                .map(|p| {
                    p["text"]
                        .as_str()
                        .context("text content part needs a string text field")
                })
                .collect::<Result<Vec<_>>>()?
                .concat();
            message["content"] = json!(text);
        }
    }
    Ok((images, videos))
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
        let detok = Box::new(state.shared.detok());
        let start = if is_chat {
            StreamState::Role(events, detok)
        } else {
            StreamState::Running(events, detok)
        };
        let sse = unfold(start, move |stream_state| {
            let model = model.clone();
            let id = id.clone();
            async move {
                match stream_state {
                    StreamState::Role(events, detok) => Some((
                        sse_json(&json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": now_secs(),
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
                        })),
                        StreamState::Running(events, detok),
                    )),
                    StreamState::Running(mut events, mut detok) => match events.recv().await {
                        Some(event) => {
                            let (item, detok_failed) =
                                match sse_event(&event, &model, &id, is_chat, &mut detok) {
                                    Ok(item) => (item, false),
                                    Err(message) => (sse_error(&message), true),
                                };
                            let next = if detok_failed || matches!(event, EngineEvent::Failed(_)) {
                                StreamState::End
                            } else {
                                match event {
                                    EngineEvent::Tokens { .. } => {
                                        StreamState::Running(events, detok)
                                    }
                                    EngineEvent::Done { .. } | EngineEvent::Aborted { .. } => {
                                        StreamState::Sentinel
                                    }
                                    EngineEvent::Failed(_) => StreamState::End,
                                }
                            };
                            Some((Ok(item), next))
                        }
                        None => Some((Ok(sse_error(ENGINE_EOF)), StreamState::End)),
                    },
                    StreamState::Sentinel => {
                        Some((Ok(Event::default().data("[DONE]")), StreamState::End))
                    }
                    StreamState::End => None,
                }
            }
        });
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
    Role(tokio_mpsc::Receiver<EngineEvent>, Box<super::detok::Detok>),
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
) -> Result<Event, String> {
    match event {
        EngineEvent::Tokens { ids, .. } => {
            let text = detok
                .push(ids)
                .map_err(|error| format!("detokenize failed: {error}"))?;
            sse_json(&chunk_payload(model, id, is_chat, &text, None))
                .map_err(|error| error.to_string())
        }
        EngineEvent::Done { length_reached, .. } => {
            let text = detok
                .finish()
                .map_err(|error| format!("detokenize failed: {error}"))?;
            let finish = if *length_reached { "length" } else { "stop" };
            sse_json(&chunk_payload(model, id, is_chat, &text, Some(finish)))
                .map_err(|error| error.to_string())
        }
        EngineEvent::Aborted { .. } => {
            sse_json(&chunk_payload(model, id, is_chat, "", Some("aborted")))
                .map_err(|error| error.to_string())
        }
        EngineEvent::Failed(message) => sse_json(&json!({
            "error": {"message": message, "type": "server_error"}
        }))
        .map_err(|error| error.to_string()),
    }
}

fn sse_error(message: &str) -> Event {
    Event::default().data(
        json!({
            "error": {"message": message, "type": "server_error"}
        })
        .to_string(),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Settings, SpecBudget, TreeBuilder};
    use crate::serve::{EngineCommand, Slot};
    use anyhow::ensure;
    use std::sync::{Mutex, mpsc};
    use std::time::Duration;

    fn state(
        shared: Shared,
        template: String,
        bos_token: String,
        capacity: usize,
        slot: Slot,
    ) -> (Arc<AppState>, mpsc::Receiver<EngineCommand>) {
        let (commands, receiver) = mpsc::channel();
        let mut env = Environment::new();
        env.set_unknown_method_callback(template_string_methods);
        env.add_function("raise_exception", template_exception);
        env.add_template_owned("chat", template).expect("template");
        let backend = Backend {
            commands,
            slot: Arc::new(Mutex::new(slot)),
            vocab_size: 300_000,
            draft_config: None,
            mtp: false,
            options: crate::spec::Options {
                device: 0,
                capacity,
                chunk: None,
                backend: Settings {
                    tree_builder: TreeBuilder::Waves,
                    spec_budget: SpecBudget::Chain,
                    spec_graph: true,
                    draft_model: None,
                    runtime_dir: None,
                    prefix_cache: false,
                    kv_format: crate::kv::KvFormat::Bf16,
                },
                count: 256,
                ignore_eos: false,
            },
        };
        (
            Arc::new(AppState {
                backend,
                shared,
                env,
                model_name: "test".into(),
                bos_token,
                request_counter: AtomicU64::new(0),
                image_processor: None,
            }),
            receiver,
        )
    }

    fn test_state(capacity: usize, slot: Slot) -> (Arc<AppState>, mpsc::Receiver<EngineCommand>) {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".into(), 0), ("a".into(), 1)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .expect("word-level model");
        state(
            Shared::new(tokenizers::Tokenizer::new(model)),
            "{{ messages[0].content }}".into(),
            String::new(),
            capacity,
            slot,
        )
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response bytes");
        serde_json::from_slice(&bytes).expect("JSON response")
    }

    #[tokio::test]
    async fn oversized_prompt_is_rejected_before_admission() {
        let (state, commands) = test_state(2, Slot::Idle);
        let request =
            serde_json::from_value(json!({"prompt": [1, 1], "max_tokens": 1})).expect("request");
        let response = completions(State(state.clone()), Json(request)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("KV capacity 3")
        );
        assert!(commands.try_recv().is_err());
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
    }

    #[tokio::test]
    async fn concurrent_request_gets_documented_busy_response() {
        let (state, commands) = test_state(16, Slot::Running);
        let request =
            serde_json::from_value(json!({"prompt": [1], "max_tokens": 1})).expect("request");
        let response = completions(State(state), Json(request)).await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(body_json(response).await["error"]["type"], "server_busy");
        assert!(commands.try_recv().is_err());
    }

    #[tokio::test]
    async fn dropping_stream_closes_engine_event_receiver() {
        let (state, commands) = test_state(16, Slot::Idle);
        let request =
            serde_json::from_value(json!({"prompt": [1], "max_tokens": 8, "stream": true}))
                .expect("request");
        let response = completions(State(state), Json(request)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let EngineCommand { events, .. } = commands.try_recv().expect("generate");
        assert!(!events.is_closed());
        drop(response);
        assert!(events.is_closed());
        assert!(
            events
                .send(EngineEvent::Tokens {
                    ids: vec![1],
                    accepted: 0
                })
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn stream_detokenizer_error_emits_error_and_stops() -> Result<()> {
        let (state, _commands) = test_state(16, Slot::Idle);
        let bad_id = 299_999u32;
        ensure!(
            state
                .shared
                .detok()
                .push(std::slice::from_ref(&bad_id))
                .is_err(),
            "out-of-vocabulary fixture must fail detokenization"
        );
        let (events, receiver) = tokio_mpsc::channel(1);
        events
            .send(EngineEvent::Tokens {
                ids: vec![bad_id],
                accepted: 0,
            })
            .await?;
        let response = respond(receiver, state, "test".into(), false, true, 1).await;
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await?;
        let text = std::str::from_utf8(&bytes)?;
        assert!(!text.contains("data: [DONE]"), "{text}");
        let payload = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(payload.len() == 1, "{payload:?}");
        ensure!(
            payload[0]["error"]["message"]
                .as_str()
                .is_some_and(|message| message.starts_with("detokenize failed")),
            "{payload:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn completion_respects_exact_plain_capacity_and_counts() {
        let (state, commands) = test_state(2, Slot::Idle);
        let worker = std::thread::spawn(move || {
            let EngineCommand { request, events } =
                commands.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(request.ids, [1]);
            assert_eq!(request.max_new_tokens, 1);
            events
                .blocking_send(EngineEvent::Tokens {
                    ids: vec![1],
                    accepted: 0,
                })
                .unwrap();
            events
                .blocking_send(EngineEvent::Done {
                    ids: vec![1],
                    rounds: 0,
                    length_reached: true,
                })
                .unwrap();
        });
        let request = serde_json::from_value(json!({"prompt": [1], "max_tokens": 1})).unwrap();
        let response = completions(State(state), Json(request)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["choices"][0]["text"], "a");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(
            body["usage"],
            json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        );
        worker.join().unwrap();
    }

    fn image_processor(tag: &str) -> Result<super::super::image::Processor> {
        let dir =
            std::env::temp_dir().join(format!("ff-image-header-{tag}-{}", std::process::id()));
        std::fs::create_dir(&dir)?;
        std::fs::write(
            dir.join("config.json"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/q38-config.json"
            )),
        )?;
        std::fs::write(
            dir.join("preprocessor_config.json"),
            serde_json::to_vec(&json!({
                "size":{"shortest_edge":65536,"longest_edge":16777216},
                "patch_size":16,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]
            }))?,
        )?;
        let processor = super::super::image::Processor::load(&dir, None)?.context("processor")?;
        std::fs::remove_dir_all(dir)?;
        Ok(processor)
    }

    fn image_request_state(
        slot: Slot,
        tag: &str,
    ) -> Result<(Arc<AppState>, mpsc::Receiver<EngineCommand>)> {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".into(), 0), ("<|image_pad|>".into(), 248056)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (mut state, commands) = state(
            Shared::new(tokenizers::Tokenizer::new(model)),
            "<|image_pad|>".into(),
            String::new(),
            256,
            slot,
        );
        Arc::get_mut(&mut state).unwrap().image_processor = Some(image_processor(tag)?);
        Ok((state, commands))
    }

    fn pixel_fixture(animated: bool) -> Result<String> {
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            if animated {
                encoder.set_animated(2, 0)?;
            }
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&[1, 2, 3])?;
            if animated {
                writer.write_image_data(&[3, 2, 1])?;
            }
        }
        // Header stays readable, but no frame can be decoded.
        let idat = png
            .windows(4)
            .position(|bytes| bytes == b"IDAT")
            .context("IDAT")?;
        png.truncate(idat + 4);
        Ok(format!("data:image/png;base64,{}", base64::encode(png)))
    }

    async fn image_response(state: Arc<AppState>, url: &str) -> Response {
        let request = serde_json::from_value(json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":url}}
        ]}],"max_tokens":1}))
        .unwrap();
        chat_completions(State(state), Json(request)).await
    }

    #[test]
    fn admission_guard_releases_on_drop_and_engine_disconnect() -> Result<()> {
        let (state, commands) = test_state(256, Slot::Idle);
        let admission = state.backend.reserve()?;
        assert!(state.backend.reserve().is_err());
        drop(admission);
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        let admission = state.backend.reserve()?;
        drop(commands);
        let (events, _receiver) = tokio_mpsc::channel(1);
        assert!(
            admission
                .send(
                    Request {
                        ids: vec![1],
                        max_new_tokens: 1,
                        images: None
                    },
                    events
                )
                .is_err()
        );
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        Ok(())
    }

    #[tokio::test]
    async fn busy_image_request_rejects_before_pixel_decode() -> Result<()> {
        let (state, commands) = image_request_state(Slot::Running, "busy")?;
        let url = pixel_fixture(false)?;
        let processor = state.image_processor.as_ref().unwrap();
        let planned = processor.plan_url(&url)?;
        assert_eq!(processor.image_tokens(&planned, 0)?, 64);
        assert!(processor.prepare_url(&url).is_err());
        let response = image_response(state.clone(), &url).await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(commands.try_recv().is_err());
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Running);
        Ok(())
    }

    #[tokio::test]
    async fn image_decode_failure_releases_admission() -> Result<()> {
        let (state, commands) = image_request_state(Slot::Idle, "decode-error")?;
        let response = image_response(state.clone(), &pixel_fixture(false)?).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let message = body_json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(message.contains("image input:"), "{message}");
        assert!(commands.try_recv().is_err());
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        // A second reservation proves the failed request did not retain ownership.
        drop(state.backend.reserve()?);
        Ok(())
    }

    #[tokio::test]
    async fn animated_png_rejects_before_frame_decode() -> Result<()> {
        let (state, commands) = image_request_state(Slot::Idle, "apng")?;
        let url = pixel_fixture(true)?;
        let error = super::super::image::load_image(&url)
            .err()
            .context("APNG decoded")?;
        assert_eq!(error.to_string(), "animated PNG not supported");
        let response = image_response(state.clone(), &url).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(
            body_json(response).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("animated PNG not supported")
        );
        assert!(commands.try_recv().is_err());
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        Ok(())
    }

    #[tokio::test]
    async fn multi_image_budget_rejects_before_any_pixel_decode() -> Result<()> {
        let processor = image_processor("budget")?;
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header()?.write_image_data(&[1, 2, 3])?;
        }
        let idat = png
            .windows(4)
            .position(|bytes| bytes == b"IDAT")
            .context("IDAT")?;
        png.truncate(idat + 4);
        let url = format!("data:image/png;base64,{}", base64::encode(png));
        let planned = processor.plan_url(&url)?;
        assert_eq!(processor.image_tokens(&planned, 0)?, 64);
        assert!(
            processor.prepare_url(&url).is_err(),
            "fixture must reject pixel decoding"
        );
        let (mut state, commands) = test_state(64, Slot::Idle);
        Arc::get_mut(&mut state).unwrap().image_processor = Some(processor);
        let request = serde_json::from_value(json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":url}},
                {"type":"image_url","image_url":{"url":url}}
            ]}],"max_tokens":1
        }))?;
        let response = chat_completions(State(state.clone()), Json(request)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(
            body_json(response).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("image tokens exceed request capacity before pixel preprocessing")
        );
        assert!(commands.try_recv().is_err());
        assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        Ok(())
    }

    #[tokio::test]
    async fn malformed_content_parts_are_rejected_before_admission() {
        for content in [
            json!([{"type":"text","text":2}]),
            json!([{"type":"video_url","video_url":{"url":"file:///a"}}]),
            json!([{"type":"image_url","image_url":{"url":"file:///a"}}]),
        ] {
            let (state, commands) = test_state(256, Slot::Idle);
            let request = serde_json::from_value(json!({
                "messages":[{"role":"user","content":content}], "max_tokens":1
            }))
            .unwrap();
            let response = chat_completions(State(state.clone()), Json(request)).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(commands.try_recv().is_err());
            assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        }
    }

    #[tokio::test]
    async fn video_parts_name_unsupported_processor_options() {
        for (part, needle) in [
            (
                json!({"type":"video_url","video_url":{"url":"/nonexistent.mp4","stack_frames":2}}),
                "stack_frames 2 is not supported",
            ),
            (
                json!({"type":"video_url","video_url":{"url":"/nonexistent.mp4","downsample_mode":"4x"}}),
                "downsample_mode \"4x\" is not supported",
            ),
        ] {
            let (state, commands) = test_state(256, Slot::Idle);
            let request = serde_json::from_value(json!({
                "messages":[{"role":"user","content":[part]}], "max_tokens":1
            }))
            .unwrap();
            let response = chat_completions(State(state.clone()), Json(request)).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = body_json(response).await;
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.contains(needle), "{needle}: {message}");
            assert!(commands.try_recv().is_err());
            assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
        }
    }

    #[tokio::test]
    #[ignore = "requires FF_IMAGE_CHAT_REFERENCE generated by the HF processor"]
    async fn image_chat_matches_hf_ids_and_grids() -> Result<()> {
        let reference = std::env::var("FF_IMAGE_CHAT_REFERENCE")?;
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(reference)?)?;
        let model = std::path::Path::new(reference["model"].as_str().context("model path")?);
        for (index, case) in reference["cases"]
            .as_array()
            .context("cases")?
            .iter()
            .enumerate()
        {
            let expected: Vec<u32> = serde_json::from_value(case["input_ids"].clone())?;
            let mut content = Vec::new();
            for file in case["files"].as_array().context("image files")? {
                let file = file.as_str().context("image file")?;
                let bytes = std::fs::read(file)?;
                let url = format!("data:image/png;base64,{}", base64::encode(bytes));
                content.push(json!({"type":"image_url","image_url":{"url":url}}));
            }
            content.push(json!({"type":"text","text":case["prompt"]}));
            let (mut state, commands) = state(
                Shared::load(model)?,
                read_chat_template(model)?,
                read_bos_token(model)?,
                expected.len() + 1,
                Slot::Idle,
            );
            Arc::get_mut(&mut state).unwrap().image_processor =
                super::super::image::Processor::load(model, None)?;
            if index == 0 {
                let rejected = serde_json::from_value(json!({
                    "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///etc/passwd"}}]}],
                    "max_tokens":1
                }))?;
                let response = chat_completions(State(state.clone()), Json(rejected)).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                assert!(
                    body_json(response).await["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("base64 data URL")
                );
                assert!(commands.try_recv().is_err());
                assert_eq!(*state.backend.slot.lock().unwrap(), Slot::Idle);
            }
            let request = serde_json::from_value(json!({
                "messages":[{"role":"user","content":content}], "max_tokens":1,
                "chat_template_kwargs":{"enable_thinking":false}, "stream":true
            }))?;
            let response = chat_completions(State(state), Json(request)).await;
            assert_eq!(response.status(), StatusCode::OK, "{}", case["name"]);
            let EngineCommand { request, events } = commands.try_recv()?;
            assert_eq!(request.ids, expected, "{} chat IDs", case["name"]);
            let prepared = request.images.context("image payload")?;
            let grids: Vec<_> = prepared.grids.iter().map(|g| [g.t, g.h, g.w]).collect();
            assert_eq!(
                serde_json::to_value(&grids)?,
                case["grids"],
                "{} grids",
                case["name"]
            );
            let features: usize = prepared
                .grids
                .iter()
                .map(|g| g.feature_rows(2).unwrap())
                .sum();
            assert_eq!(
                prepared.modality_ids.iter().filter(|&&v| v == 1).count(),
                features
            );
            assert_eq!(prepared.pixel_values.len(), features * 4 * 1536);
            let processor =
                super::super::image::Processor::load(model, None)?.context("processor")?;
            let mut direct_pixels = Vec::new();
            for file in case["files"].as_array().unwrap() {
                let image = super::super::image::decode_image(
                    &std::fs::read(file.as_str().unwrap())?,
                    None,
                )?;
                direct_pixels.extend(processor.prepare_rgb(&image)?.pixel_values);
            }
            assert!(
                prepared
                    .pixel_values
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(direct_pixels.iter().map(|v| v.to_bits())),
                "HTTP pixels differ from direct Rust preprocessing"
            );
            let path = std::env::temp_dir().join(format!(
                "flyingfish-image-roundtrip-{}-{index}.safetensors",
                std::process::id()
            ));
            let spec = crate::model_spec::ModelSpec::read(model)?;
            prepared.save(&spec, &request.ids, &path)?;
            let replay = super::super::image::PreparedInput::read(&spec, &path, &request.ids)?;
            assert!(super::super::image::PreparedInput::read(&spec, &path, &[0]).is_err());
            std::fs::remove_file(&path)?;
            assert_eq!(replay.grids, prepared.grids);
            assert_eq!(replay.modality_ids, prepared.modality_ids);
            assert!(
                replay
                    .pixel_values
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(prepared.pixel_values.iter().map(|v| v.to_bits())),
                "input trace changed pixel bits"
            );
            events
                .send(EngineEvent::Done {
                    ids: vec![],
                    rounds: 0,
                    length_reached: true,
                })
                .await?;
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await?;
            assert!(std::str::from_utf8(&bytes)?.contains("data: [DONE]"));
            println!(
                "{}: {} IDs, {} images exact",
                case["name"],
                expected.len(),
                grids.len()
            );
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires FF_CHAT_REFERENCE generated by the HF processor and its local model files"]
    async fn qwen_chat_matches_hf_processor_and_serves_both_response_modes() -> Result<()> {
        let reference = std::env::var("FF_CHAT_REFERENCE").context("FF_CHAT_REFERENCE")?;
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(reference)?)?;
        let model = std::path::Path::new(reference["model"].as_str().context("model path")?);
        let cases = reference["cases"].as_array().context("cases")?;
        ensure!(!cases.is_empty(), "reference must contain cases");
        for (index, case) in cases.iter().enumerate() {
            let shared = Shared::load(model)?;
            let output = shared.encode_chat("4")?;
            let expected: Vec<u32> = serde_json::from_value(case["input_ids"].clone())?;
            let (state, commands) = state(
                shared,
                read_chat_template(model)?,
                read_bos_token(model)?,
                4096,
                Slot::Idle,
            );
            let messages =
                serde_json::from_value::<Vec<serde_json::Value>>(case["messages"].clone())?;
            let options = serde_json::from_value(case["kwargs"].clone())?;
            let rendered = render_chat(&state.env, &messages, &state.bos_token, &options)?;
            assert_eq!(
                rendered,
                case["rendered"].as_str().unwrap(),
                "{} text",
                case["name"]
            );
            assert_eq!(
                state.shared.encode_chat(&rendered)?,
                expected,
                "{} ids",
                case["name"]
            );
            let worker = std::thread::spawn(move || {
                let EngineCommand { request, events } =
                    commands.recv_timeout(Duration::from_secs(5)).unwrap();
                events
                    .blocking_send(EngineEvent::Tokens {
                        ids: output.clone(),
                        accepted: 0,
                    })
                    .unwrap();
                events
                    .blocking_send(EngineEvent::Done {
                        ids: output,
                        rounds: 0,
                        length_reached: true,
                    })
                    .unwrap();
                request.ids
            });
            let stream = index % 2 == 0;
            let request = serde_json::from_value(json!({
                "messages": case["messages"], "chat_template_kwargs": case["kwargs"],
                "temperature": 0, "max_tokens": 1, "stream": stream,
            }))?;
            let response = chat_completions(State(state), Json(request)).await;
            assert_eq!(response.status(), StatusCode::OK, "{}", case["name"]);
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await?;
            if stream {
                let text = std::str::from_utf8(&bytes)?;
                assert!(text.contains("data: [DONE]"));
                let mut generated = String::new();
                for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                    if line == "[DONE]" {
                        continue;
                    }
                    let chunk: serde_json::Value = serde_json::from_str(line)?;
                    if let Some(content) = chunk["choices"][0]["delta"]["content"].as_str() {
                        generated.push_str(content);
                    }
                }
                assert_eq!(generated, "4");
            } else {
                let body: serde_json::Value = serde_json::from_slice(&bytes)?;
                assert_eq!(body["choices"][0]["message"]["content"], "4");
                assert_eq!(body["usage"]["prompt_tokens"], expected.len());
            }
            assert_eq!(
                worker.join().expect("mock worker"),
                expected,
                "{} HTTP ids",
                case["name"]
            );
        }
        Ok(())
    }
}
