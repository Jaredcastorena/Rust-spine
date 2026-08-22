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

pub async fn start(
    host: &str,
    port: u16,
    input: UnboundedSender<io::Result<String>>,
    heart: String,
    incognito: bool,
    grounding: bool,
    tool_count: usize,
) -> Result<WebServer, Box<dyn std::error::Error>> {
    let ip: IpAddr = host.parse()?;
    if !ip.is_loopback() && !is_tailscale_ip(ip) {
        return Err(
            "--web-host must be loopback or a Tailscale IPv4 address in the managed address range".into(),
        );
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
    let listener = tokio::net::TcpListener::bind((ip, port)).await?;
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

fn is_tailscale_ip(ip: IpAddr) -> bool {
    let IpAddr::V4(ip) = ip else {
        return false;
    };
    let octets = ip.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

async fn index() -> Response {
    let mut response = Html(HTML).into_response();
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; frame-ancestors 'none'".parse().unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
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
    Json(state.ui.state.read().expect("web state poisoned").clone()).into_response()
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
        snapshot.busy = true;
        snapshot.phase = "queued".into();
        snapshot.activity = "Queued".into();
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
    async fn rejects_non_tailscale_remote_binding() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(
            start("0.0.0.0", 0, tx, "test".into(), true, false, 1)
                .await
                .is_err()
        );
    }

    #[test]
    fn accepts_only_loopback_and_tailscale_ranges() {
        assert!(is_tailscale_ip(IpAddr::from([100, 64, 0, 1])));
        assert!(is_tailscale_ip(IpAddr::from([100, 127, 255, 255])));
        assert!(!is_tailscale_ip(IpAddr::from([100, 128, 0, 1])));
        assert!(!is_tailscale_ip(IpAddr::from([192, 168, 1, 2])));
    }

    #[tokio::test]
    async fn loopback_server_starts_on_an_ephemeral_port() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start("127.0.0.1", 0, tx, "test".into(), true, false, 1)
            .await
            .unwrap();
        assert!(server.access_url.starts_with("http://127.0.0.1:"));
        server.shutdown().await;
    }

    #[tokio::test]
    async fn api_requires_page_token_and_forwards_messages() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let server = start("127.0.0.1", 0, tx, "test".into(), true, false, 1)
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
        let html = client
            .get(base_url)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(token.len(), 64);
        assert!(!html.contains(token));
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
        assert_eq!(rx.recv().await.unwrap().unwrap(), "hello from browser");
        server.shutdown().await;
    }
}
