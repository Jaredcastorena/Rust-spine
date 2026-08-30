use std::{
    io,
    net::IpAddr,
    sync::{Arc, RwLock},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use spine_runtime::{Message, MessageRole};
use tokio::sync::{mpsc::UnboundedSender, oneshot};

const HTML: &str = include_str!("../web/index.html");
const CSS: &str = include_str!("../web/style.css");
const JS: &str = include_str!("../web/app.js");

#[derive(Clone, Serialize)]
pub struct WebMessage {
    role: String,
    content: String,
}

#[derive(Clone, Serialize)]
pub struct WebTool {
    id: String,
    name: String,
    status: String,
    success: Option<bool>,
    input: Option<String>,
    output: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct WebSnapshot {
    phase: String,
    activity: String,
    completed_tools: u64,
    busy: bool,
    checkpoint_available: bool,
    heart: String,
    incognito: bool,
    grounding: bool,
    tool_count: usize,
    messages: Vec<WebMessage>,
    tools: Vec<WebTool>,
    notice: Option<String>,
}

#[derive(Clone)]
pub struct WebUi {
    state: Arc<RwLock<WebSnapshot>>,
}

impl WebUi {
    pub fn begin_turn(&self, text: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        if !state.busy {
            state.messages.push(WebMessage {
                role: "user".into(),
                content: text.into(),
            });
        }
        state.busy = true;
        state.phase = "running".into();
        state.activity = "Checking memory".into();
        state.completed_tools = 0;
        state.notice = None;
    }

    pub fn begin_resume(&self) {
        let mut state = self.state.write().expect("web state poisoned");
        state.busy = true;
        state.phase = "running".into();
        state.activity = "Resuming".into();
        state.notice = None;
    }

    pub fn activity(&self, text: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        state.activity = text.into();
    }

    pub fn tool_started(&self, id: &str, name: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        state.tools.push(WebTool {
            id: id.into(),
            name: name.into(),
            status: "running".into(),
            success: None,
            input: None,
            output: None,
        });
    }

    pub fn tool_completed(&self, id: &str, success: bool) {
        let mut state = self.state.write().expect("web state poisoned");
        if let Some(tool) = state.tools.iter_mut().rev().find(|tool| tool.id == id) {
            tool.status = if success { "completed" } else { "failed" }.into();
            tool.success = Some(success);
        }
        state.completed_tools = state.completed_tools.saturating_add(1);
        state.activity = "Thinking".into();
    }

    pub fn capture_messages(&self, messages: &[Message]) {
        let mut state = self.state.write().expect("web state poisoned");
        for message in messages {
            if message.role == MessageRole::Assistant {
                if let Some(reasoning) = &message.reasoning
                    && !reasoning.trim().is_empty()
                {
                    state.messages.push(WebMessage {
                        role: "reasoning".into(),
                        content: reasoning.clone(),
                    });
                }
                for call in &message.tool_calls {
                    if let Some(tool) = state.tools.iter_mut().rev().find(|tool| tool.id == call.id)
                    {
                        tool.input = Some(call.arguments.to_string());
                    }
                }
            } else if message.role == MessageRole::Tool
                && let Some(id) = &message.tool_call_id
                && let Some(tool) = state.tools.iter_mut().rev().find(|tool| &tool.id == id)
            {
                tool.output = Some(message.content.clone());
            }
        }
    }

    pub fn complete(&self, response: &str, stopped: bool, checkpoint: bool) {
        let mut state = self.state.write().expect("web state poisoned");
        if !response.trim().is_empty() {
            state.messages.push(WebMessage {
                role: "assistant".into(),
                content: response.into(),
            });
        }
        state.busy = false;
        state.phase = if stopped { "stopped" } else { "idle" }.into();
        state.activity = if stopped { "Stopped safely" } else { "Ready" }.into();
        state.checkpoint_available = checkpoint;
    }

    pub fn fail(&self, message: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        state.busy = false;
        state.phase = "error".into();
        state.activity = "Turn failed".into();
        state.notice = Some(message.into());
    }

    pub fn notice(&self, message: impl Into<String>) {
        self.state.write().expect("web state poisoned").notice = Some(message.into());
    }

    pub fn begin_onboarding_model_turn(&self) {
        let mut state = self.state.write().expect("web state poisoned");
        state.busy = true;
        state.phase = "onboarding".into();
        state.activity = "Getting acquainted".into();
        state.notice = None;
    }

    pub fn begin_onboarding_answer(&self, text: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        if !state.busy {
            state.messages.push(WebMessage {
                role: "user".into(),
                content: text.into(),
            });
        }
        state.busy = true;
        state.phase = "onboarding".into();
        state.activity = "Listening".into();
        state.notice = None;
    }

    pub fn finish_onboarding(&self, response: &str, complete: bool) {
        let mut state = self.state.write().expect("web state poisoned");
        if !response.trim().is_empty() {
            state.messages.push(WebMessage {
                role: "assistant".into(),
                content: response.into(),
            });
        }
        state.busy = false;
        state.phase = if complete { "idle" } else { "onboarding" }.into();
        state.activity = if complete {
            "Ready"
        } else {
            "Getting to know you"
        }
        .into();
        state.notice = if complete {
            None
        } else {
            Some("Reply naturally, or type /skip to start working now.".into())
        };
    }

    pub fn pause_onboarding(&self, message: &str) {
        let mut state = self.state.write().expect("web state poisoned");
        state.busy = false;
        state.phase = "idle".into();
        state.activity = "Ready".into();
        state.notice = Some(message.into());
    }
}

pub struct WebServer {
    ui: WebUi,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
    pub access_url: String,
}

impl WebServer {
    pub fn ui(&self) -> WebUi {
        self.ui.clone()
    }

    pub async fn shutdown(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let _ = self.task.await;
    }
}

#[derive(Clone)]
struct AppState {
    ui: WebUi,
    input: UnboundedSender<io::Result<String>>,
    token: String,
}

#[derive(Deserialize)]
struct TextRequest {
    message: String,
}

#[derive(Deserialize)]
struct ControlRequest {
    action: String,
}

pub struct WebBind<'a> {
    pub host: &'a str,
    pub port: u16,
    pub allow_remote: bool,
}

pub async fn start(
    bind: WebBind<'_>,
    input: UnboundedSender<io::Result<String>>,
    heart: String,
    incognito: bool,
    grounding: bool,
    tool_count: usize,
) -> Result<WebServer, Box<dyn std::error::Error>> {
    let ip: IpAddr = bind.host.parse()?;
    if !ip.is_loopback() && !bind.allow_remote {
        return Err("a non-loopback --web-host requires --allow-remote-web".into());
    }
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes)?;
    let token = hex::encode(bytes);
    let ui = WebUi {
        state: Arc::new(RwLock::new(WebSnapshot {
            phase: "idle".into(),
            activity: "Ready".into(),
            completed_tools: 0,
            busy: false,
            checkpoint_available: false,
            heart,
            incognito,
            grounding,
            tool_count,
            messages: Vec::new(),
            tools: Vec::new(),
            notice: None,
        })),
    };
    let app_state = AppState {
        ui: ui.clone(),
        input,
        token: token.clone(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/style.css", get(style))
        .route("/app.js", get(script))
        .route("/api/state", get(api_state))
        .route("/api/message", post(api_message))
        .route("/api/guidance", post(api_guidance))
        .route("/api/control", post(api_control))
        .with_state(app_state);
    let listener = tokio::net::TcpListener::bind((ip, bind.port)).await?;
    let address = listener.local_addr()?;
    let url = format!("http://{address}");
    let access_url = format!("{url}/#{token}");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });
    Ok(WebServer {
        ui,
        shutdown: Some(shutdown_tx),
        task,
        access_url,
    })
}

async fn index() -> Response {
    let mut response = Html(HTML).into_response();
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; object-src 'none'; form-action 'self'; frame-ancestors 'none'".parse().unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response.headers_mut().insert(
        header::HeaderName::from_static("referrer-policy"),
        "no-referrer".parse().unwrap(),
    );
    response.headers_mut().insert(
        header::HeaderName::from_static("x-content-type-options"),
        "nosniff".parse().unwrap(),
    );
    response
}

async fn style() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], CSS)
}
async fn script() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        JS,
    )
}

fn authorized(headers: &HeaderMap, state: &AppState) -> bool {
    headers
        .get("x-spine-token")
        .and_then(|value| value.to_str().ok())
        == Some(&state.token)
}

async fn api_state(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&headers, &state) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut response =
        Json(state.ui.state.read().expect("web state poisoned").clone()).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

async fn api_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<TextRequest>,
) -> Response {
    if !authorized(&headers, &state) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let message = body.message.trim();
    if message.is_empty() {
        return (StatusCode::BAD_REQUEST, "message is empty").into_response();
    }
    {
        let mut snapshot = state.ui.state.write().expect("web state poisoned");
        if snapshot.busy {
            return (StatusCode::CONFLICT, "a turn is already active").into_response();
        }
        let onboarding = snapshot.phase == "onboarding";
        snapshot.busy = true;
        snapshot.phase = if onboarding { "onboarding" } else { "queued" }.into();
        snapshot.activity = if onboarding { "Listening" } else { "Queued" }.into();
        snapshot.messages.push(WebMessage {
            role: "user".into(),
            content: message.into(),
        });
    }
    if state.input.send(Ok(message.into())).is_err() {
        state.ui.fail("partner input channel is closed");
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    StatusCode::ACCEPTED.into_response()
}

async fn api_guidance(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<TextRequest>,
) -> Response {
    if !authorized(&headers, &state) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let message = body.message.trim();
    if message.is_empty() {
        return (StatusCode::BAD_REQUEST, "guidance is empty").into_response();
    }
    if !state.ui.state.read().expect("web state poisoned").busy {
        return (StatusCode::CONFLICT, "no turn is active").into_response();
    }
    if state.input.send(Ok(message.into())).is_err() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    state
        .ui
        .notice("Guidance queued for the next tool boundary");
    StatusCode::ACCEPTED.into_response()
}

async fn api_control(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ControlRequest>,
) -> Response {
    if !authorized(&headers, &state) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let (busy, checkpoint) = {
        let snapshot = state.ui.state.read().expect("web state poisoned");
        (snapshot.busy, snapshot.checkpoint_available)
    };
    let command = match body.action.as_str() {
        "stop" if busy => "/stop",
        "interrupt" if busy => "/interrupt",
        "resume" if !busy && checkpoint => "/resume",
        "quit" => "/quit",
        "tasks" if !busy => "/tasks",
        "stop" | "interrupt" => {
            return (StatusCode::CONFLICT, "no turn is active").into_response();
        }
        "resume" => {
            return (StatusCode::CONFLICT, "no resumable checkpoint").into_response();
        }
        "tasks" => {
            return (StatusCode::CONFLICT, "tasks are available between turns").into_response();
        }
        _ => return (StatusCode::BAD_REQUEST, "unknown action").into_response(),
    };
    if body.action == "resume" {
        let mut snapshot = state.ui.state.write().expect("web state poisoned");
        snapshot.busy = true;
        snapshot.phase = "queued".into();
        snapshot.activity = "Resume queued".into();
    }
    if state.input.send(Ok(command.into())).is_err() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    StatusCode::ACCEPTED.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_remote_binding_without_an_explicit_opt_in() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(
            start(
                WebBind {
                    host: "0.0.0.0",
                    port: 0,
                    allow_remote: false,
                },
                tx,
                "test".into(),
                true,
                false,
                1,
            )
            .await
            .is_err()
        );

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start(
            WebBind {
                host: "0.0.0.0",
                port: 0,
                allow_remote: true,
            },
            tx,
            "test".into(),
            true,
            false,
            1,
        )
        .await
        .unwrap();
        assert!(server.access_url.starts_with("http://0.0.0.0:"));
        server.shutdown().await;
    }

    #[tokio::test]
    async fn loopback_server_starts_on_an_ephemeral_port() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start(
            WebBind {
                host: "127.0.0.1",
                port: 0,
                allow_remote: false,
            },
            tx,
            "test".into(),
            true,
            false,
            1,
        )
        .await
        .unwrap();
        assert!(server.access_url.starts_with("http://127.0.0.1:"));
        server.shutdown().await;
    }

    #[tokio::test]
    async fn onboarding_uses_the_same_browser_conversation_without_duplicate_answers() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start(
            WebBind {
                host: "127.0.0.1",
                port: 0,
                allow_remote: false,
            },
            tx,
            "test".into(),
            false,
            false,
            1,
        )
        .await
        .unwrap();
        let ui = server.ui();
        ui.begin_onboarding_model_turn();
        ui.finish_onboarding("How do you like to work?", false);
        ui.begin_onboarding_answer("Be direct.");
        ui.begin_onboarding_answer("Be direct.");

        let state = ui.state.read().unwrap().clone();
        assert_eq!(state.phase, "onboarding");
        assert!(state.busy);
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].role, "assistant");
        assert_eq!(state.messages[1].content, "Be direct.");
        server.shutdown().await;
    }

    #[tokio::test]
    async fn api_requires_page_token_and_forwards_messages() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start(
            WebBind {
                host: "127.0.0.1",
                port: 0,
                allow_remote: false,
            },
            tx,
            "test".into(),
            true,
            false,
            1,
        )
        .await
        .unwrap();
        let base_url = server
            .access_url
            .split('#')
            .next()
            .unwrap()
            .trim_end_matches('/');
        let client = reqwest::Client::new();
        let token = server.access_url.split('#').nth(1).unwrap();
        let html_response = client.get(base_url).send().await.unwrap();
        assert_eq!(
            html_response
                .headers()
                .get("referrer-policy")
                .and_then(|value| value.to_str().ok()),
            Some("no-referrer")
        );
        let html = html_response.text().await.unwrap();
        assert_eq!(token.len(), 64);
        assert!(!html.contains(token));
        assert!(html.contains("/style.css"));
        assert!(html.contains("/app.js"));
        let css = client
            .get(format!("{base_url}/style.css"))
            .send()
            .await
            .unwrap();
        assert_eq!(css.status(), StatusCode::OK);
        assert!(css.text().await.unwrap().contains("color-scheme:dark"));
        let script = client
            .get(format!("{base_url}/app.js"))
            .send()
            .await
            .unwrap();
        assert_eq!(script.status(), StatusCode::OK);
        assert!(script.text().await.unwrap().contains("x-spine-token"));
        assert_eq!(
            client
                .get(format!("{base_url}/api/state"))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            client
                .post(format!("{base_url}/api/message"))
                .header("x-spine-token", token)
                .json(&serde_json::json!({"message": "hello from browser"}))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::ACCEPTED
        );
        let state = client
            .get(format!("{base_url}/api/state"))
            .header("x-spine-token", token)
            .send()
            .await
            .unwrap();
        assert_eq!(
            state
                .headers()
                .get(header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("no-store")
        );
        assert_eq!(rx.recv().await.unwrap().unwrap(), "hello from browser");
        server.shutdown().await;
    }
}
