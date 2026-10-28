use std::{
    collections::BTreeSet,
    error::Error as _,
    sync::{
        OnceLock,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tool_parser::parse_text_tool_calls;
use crate::{
    CompletionRequest, MessageRole, ModelProvider, ModelTurn, Result, RuntimeError, TokenUsage,
    ToolCall,
};

#[derive(Clone)]
pub struct LlamaCppConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub temperature: f32,
    pub max_tokens: i64,
    pub reasoning_effort: Option<String>,
    pub timeout: Duration,
    pub maximum_retries: u32,
    pub max_context_tokens: Option<usize>,
    pub context_safety_tokens: usize,
}

impl LlamaCppConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            model: None,
            temperature: 0.2,
            max_tokens: 4_096,
            reasoning_effort: None,
            timeout: Duration::from_secs(7_200),
            maximum_retries: 2,
            max_context_tokens: None,
            context_safety_tokens: 2_048,
        }
    }
}

pub struct LlamaCppProvider {
    config: LlamaCppConfig,
    client: reqwest::Client,
    detected_context_tokens: AtomicUsize,
    detected_model: OnceLock<String>,
    detected_tools: AtomicU8,
}

impl LlamaCppProvider {
    pub fn new(mut config: LlamaCppConfig) -> Result<Self> {
        config.base_url = config.base_url.trim_end_matches('/').to_owned();
        config.reasoning_effort = config
            .reasoning_effort
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        if config.base_url.is_empty()
            || !config.temperature.is_finite()
            || config.temperature < 0.0
            || config.max_tokens == 0
            || config.timeout.is_zero()
        {
            return Err(RuntimeError::InvalidConfig(
                "llama.cpp URL, temperature, or token limit is invalid".into(),
            ));
        }
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(api_key) = config.api_key.as_ref().filter(|value| !value.is_empty()) {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|error| RuntimeError::InvalidConfig(error.to_string()))?;
            headers.insert(AUTHORIZATION, value);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(15))
            .pool_max_idle_per_host(0)
            .tcp_keepalive(Duration::from_secs(30))
            .timeout(config.timeout)
            .build()
            .map_err(provider_error)?;
        Ok(Self {
            detected_context_tokens: AtomicUsize::new(config.max_context_tokens.unwrap_or(0)),
            detected_model: OnceLock::new(),
            detected_tools: AtomicU8::new(0),
            config,
            client,
        })
    }

    fn endpoint(&self, path: &str) -> String {
        if self.config.base_url.ends_with("/v1")
            && let Some(suffix) = path.strip_prefix("/v1")
        {
            return format!("{}{suffix}", self.config.base_url);
        }
        format!("{}{path}", self.config.base_url)
    }

    pub async fn health(&self) -> Result<()> {
        let health = self.client.get(self.endpoint("/health")).send().await;
        let health_ok = health
            .as_ref()
            .is_ok_and(|response| response.status().is_success());
        let (props_response, models_response) = tokio::join!(
            self.client.get(self.endpoint("/props")).send(),
            self.client.get(self.endpoint("/v1/models")).send(),
        );
        // A public health endpoint does not establish API access. Do not accept
        // rejected credentials just because the process itself is healthy.
        if models_response.as_ref().is_ok_and(|response| {
            matches!(
                response.status(),
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
            )
        }) {
            return Err(RuntimeError::Provider(
                "Model API rejected access; check the API key and permissions".into(),
            ));
        }
        let (props, props_error) = discovery_json(props_response).await;
        let (models, models_error) = discovery_json(models_response).await;
        let discovery_ok = props.is_some() || models.is_some();
        let selected = models
            .as_ref()
            .and_then(|v| v.get("data"))
            .and_then(Value::as_array)
            .and_then(|models| {
                self.config
                    .model
                    .as_deref()
                    .and_then(|id| {
                        models
                            .iter()
                            .find(|m| m.get("id").and_then(Value::as_str) == Some(id))
                    })
                    .or_else(|| models.first())
            });
        let supports_tools = selected
            .and_then(advertised_tools)
            .or_else(|| props.as_ref().and_then(advertised_tools));
        self.detected_tools.store(
            match supports_tools {
                Some(false) => 1,
                Some(true) => 2,
                None => 0,
            },
            Ordering::Relaxed,
        );
        if self.config.max_context_tokens.is_none()
            && let Some(tokens) = props
                .as_ref()
                .and_then(advertised_context_tokens)
                .or_else(|| models.as_ref().and_then(advertised_context_tokens))
        {
            self.detected_context_tokens
                .store(tokens, Ordering::Relaxed);
        }
        if self.config.model.is_none()
            && let Some(model) = models
                .as_ref()
                .and_then(advertised_model)
                .or_else(|| props.as_ref().and_then(advertised_model))
        {
            let _ = self.detected_model.set(model.to_owned());
        }
        if health_ok || discovery_ok {
            return Ok(());
        }
        if let Some(error) = models_error.or(props_error) {
            return Err(error);
        }
        match health {
            Ok(response) => Err(http_error(response).await),
            Err(error) => Err(provider_error(error)),
        }
    }

    pub fn context_tokens(&self) -> Option<usize> {
        let tokens = self.detected_context_tokens.load(Ordering::Relaxed);
        (tokens > 0).then_some(tokens)
    }

    pub fn model(&self) -> Option<&str> {
        self.config
            .model
            .as_deref()
            .filter(|model| !model.is_empty())
            .or_else(|| self.detected_model.get().map(String::as_str))
    }
}

fn advertised_tools(value: &Value) -> Option<bool> {
    value
        .get("tool_calls")
        .and_then(Value::as_bool)
        .or_else(|| {
            value
                .pointer("/capabilities/tool_calls")
                .and_then(Value::as_bool)
        })
        .or_else(|| {
            value
                .pointer("/capabilities/tools")
                .and_then(Value::as_bool)
        })
}

async fn discovery_json(
    response: std::result::Result<reqwest::Response, reqwest::Error>,
) -> (Option<Value>, Option<RuntimeError>) {
    match response {
        Ok(response) if response.status().is_success() => match response.json::<Value>().await {
            Ok(value) => (Some(value), None),
            Err(error) => (None, Some(provider_error(error))),
        },
        Ok(response) => (None, Some(http_error(response).await)),
        Err(error) => (None, Some(provider_error(error))),
    }
}

#[async_trait]
impl ModelProvider for LlamaCppProvider {
    async fn complete(&self, mut request: CompletionRequest) -> Result<ModelTurn> {
        let temperature = request.temperature.unwrap_or(self.config.temperature);
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(RuntimeError::InvalidConfig(
                "completion-request temperature must be finite and nonnegative".into(),
            ));
        }
        let allow_tool_calls = request.allow_tool_calls;
        let allowed_tool_names = request
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<BTreeSet<_>>();
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect();
        let maximum_context_tokens = self.context_tokens();
        let completion_tokens =
            bounded_completion_tokens(self.config.max_tokens, maximum_context_tokens);
        let native_tools = self.detected_tools.load(Ordering::Relaxed) != 1;
        let reserved_tool_tokens = if allow_tool_calls && native_tools {
            estimated_json_tokens(&tools)
        } else {
            0
        };
        if allow_tool_calls && !native_tools && !tools.is_empty() {
            let catalogue = format!(
                "\n[TEXT TOOL INTERFACE] Tools are available through text calls. To execute a tool, emit a JSON object outside any thoughts or examples: {{\"tool\":\"name\",\"args\":{{...}}}}. Call only tools needed now. Available functions and argument schemas:\n{}",
                serde_json::to_string(&tools)?
            );
            if let Some(system) = request
                .messages
                .iter_mut()
                .find(|m| m.role == MessageRole::System)
            {
                system.content.push_str(&catalogue);
            } else {
                request
                    .messages
                    .insert(0, crate::Message::new(MessageRole::System, catalogue));
            }
        }
        let prepared = prepare_messages(
            request.messages,
            maximum_context_tokens,
            self.config.context_safety_tokens,
            completion_tokens,
            reserved_tool_tokens,
        );
        let messages: Vec<Value> = prepared
            .iter()
            .map(|message| {
                let role = match message.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };
                let mut value = json!({"role": role, "content": message.content});
                if let Some(call_id) = &message.tool_call_id {
                    value["tool_call_id"] = json!(call_id);
                }
                if !message.tool_calls.is_empty() {
                    value["tool_calls"] = Value::Array(
                        message
                            .tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments.to_string(),
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                if let Some(reasoning) = &message.reasoning {
                    value["reasoning_content"] = json!(reasoning);
                }
                value
            })
            .collect();
        let mut payload = json!({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": completion_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if let Some(model) = self.model() {
            payload["model"] = json!(model);
        }
        if let Some(reasoning_effort) = &self.config.reasoning_effort {
            payload["reasoning_effort"] = json!(reasoning_effort);
        }
        if allow_tool_calls && native_tools && !tools.is_empty() {
            payload["tools"] = Value::Array(tools);
            payload["tool_choice"] = json!("auto");
        }

        static NEXT_REQUEST: AtomicUsize = AtomicUsize::new(1);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let conversation_id = format!(
            "spine-{timestamp}-{}",
            NEXT_REQUEST.fetch_add(1, Ordering::Relaxed)
        );
        let mut cancellation = CancellationGuard {
            client: self.client.clone(),
            url: self.endpoint("/v1/stream"),
            id: conversation_id.clone(),
            armed: true,
        };
        let mut attempt = 0_u32;
        let mut stream_fallback = false;
        let response = loop {
            let response = self
                .client
                .post(self.endpoint("/v1/chat/completions"))
                .header("X-Conversation-Id", &conversation_id)
                .json(&payload)
                .send()
                .await;
            match response {
                Ok(response) => {
                    if matches!(response.status().as_u16(), 400 | 422) && !stream_fallback {
                        let status = response.status();
                        let body = response.text().await.map_err(provider_error)?;
                        if body.to_ascii_lowercase().contains("stream") {
                            payload["stream"] = json!(false);
                            payload.as_object_mut().unwrap().remove("stream_options");
                            stream_fallback = true;
                            continue;
                        }
                        cancellation.armed = false;
                        let body: String = body.chars().take(2000).collect();
                        return Err(RuntimeError::Provider(format!("HTTP {status}: {body}")));
                    }
                    if response.status().is_success()
                        || !retryable_status(response.status())
                        || attempt >= self.config.maximum_retries
                    {
                        break response;
                    }
                }
                Err(error) if attempt >= self.config.maximum_retries => {
                    return Err(provider_error(error));
                }
                Err(_) => {}
            }
            let delay = Duration::from_millis(250_u64.saturating_mul(1_u64 << attempt.min(6)));
            tokio::time::sleep(delay).await;
            attempt = attempt.saturating_add(1);
        };
        if !response.status().is_success() {
            cancellation.armed = false;
            return Err(http_error(response).await);
        }
        let response: ChatResponse = if response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"))
        {
            serde_json::from_value(
                crate::streaming::read_response(response, request.progress.as_ref()).await?,
            )?
        } else {
            response.json().await.map_err(provider_error)?
        };
        cancellation.armed = false;
        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::Provider("provider returned no choices".into()))?
            .message;
        let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
        let mut seen_calls = BTreeSet::new();
        for (index, call) in message.tool_calls.into_iter().enumerate() {
            let arguments = match call.function.arguments {
                Value::String(arguments) if arguments.trim().is_empty() => json!({}),
                Value::String(arguments) => serde_json::from_str(&arguments).map_err(|error| {
                    RuntimeError::Provider(format!(
                        "provider returned invalid arguments for {:?}: {error}",
                        call.function.name
                    ))
                })?,
                Value::Null => json!({}),
                arguments => arguments,
            };
            let call = ToolCall {
                id: call.id.unwrap_or_else(|| format!("llama-call-{index}")),
                name: call.function.name,
                arguments,
            };
            seen_calls.insert((call.name.clone(), call.arguments.to_string()));
            tool_calls.push(call);
        }
        let (mut content, reasoning) = crate::streaming::separate_thoughts(
            &message.content.unwrap_or_default(),
            message.reasoning_content.as_deref(),
        );
        if allow_tool_calls {
            let (cleaned, fallback_calls) =
                parse_text_tool_calls(&content, &allowed_tool_names, tool_calls.len());
            tool_calls.extend(
                fallback_calls.into_iter().filter(|call| {
                    seen_calls.insert((call.name.clone(), call.arguments.to_string()))
                }),
            );
            content = cleaned;
        }
        Ok(ModelTurn {
            content,
            reasoning,
            tool_calls,
            usage: TokenUsage {
                prompt: response.usage.prompt_tokens,
                completion: response.usage.completion_tokens,
            },
        })
    }
}

fn provider_error(error: reqwest::Error) -> RuntimeError {
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connection"
    } else if error.is_body() {
        "response body"
    } else if error.is_decode() {
        "response decode"
    } else if error.is_request() {
        "request"
    } else {
        "transport"
    };
    let mut detail = error.to_string();
    let mut cause = error.source();
    while let Some(source) = cause {
        let source_text = source.to_string();
        if !source_text.is_empty() && !detail.contains(&source_text) {
            detail.push_str(": ");
            detail.push_str(&source_text);
        }
        cause = source.source();
    }
    RuntimeError::Provider(format!("provider {kind} error: {detail}"))
}

async fn http_error(response: reqwest::Response) -> RuntimeError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let body: String = body.chars().take(2_000).collect();
    RuntimeError::Provider(format!("OpenAI-compatible provider HTTP {status}: {body}"))
}

fn retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn advertised_context_tokens(models: &Value) -> Option<usize> {
    [
        "/default_generation_settings/n_ctx",
        "/data/0/meta/n_ctx",
        "/models/0/meta/n_ctx",
        "/data/0/context_length",
        "/models/0/context_length",
        "/n_ctx",
    ]
    .into_iter()
    .filter_map(|pointer| models.pointer(pointer))
    .find_map(|value| {
        value
            .as_u64()
            .and_then(|tokens| usize::try_from(tokens).ok())
            .or_else(|| value.as_str().and_then(|tokens| tokens.parse().ok()))
            .filter(|tokens| *tokens > 0)
    })
}

fn advertised_model(models: &Value) -> Option<&str> {
    [
        "/data/0/id",
        "/data/0/model",
        "/data/0/name",
        "/models/0/model",
        "/models/0/name",
        "/model_alias",
        "/model_path",
    ]
    .into_iter()
    .filter_map(|pointer| models.pointer(pointer).and_then(Value::as_str))
    .find(|model| !model.trim().is_empty())
}

fn bounded_completion_tokens(requested: i64, maximum_tokens: Option<usize>) -> i64 {
    if requested < 0 {
        return requested;
    }
    let Some(maximum_tokens) = maximum_tokens else {
        return requested;
    };
    let ceiling = (maximum_tokens / 4).max(256).min(i64::MAX as usize) as i64;
    requested.min(ceiling)
}

fn prepare_messages(
    mut messages: Vec<crate::Message>,
    maximum_tokens: Option<usize>,
    safety_tokens: usize,
    completion_tokens: i64,
    reserved_prompt_tokens: usize,
) -> Vec<crate::Message> {
    let Some(maximum_tokens) = maximum_tokens else {
        return messages;
    };
    let completion_tokens = completion_tokens.max(0) as usize;
    let safety_tokens = safety_tokens.min(maximum_tokens / 8);
    let prompt_budget = maximum_tokens
        .saturating_sub(safety_tokens)
        .saturating_sub(completion_tokens)
        .saturating_sub(reserved_prompt_tokens)
        .max(256);
    while estimated_tokens(&messages) > prompt_budget {
        let Some(latest_user) = messages
            .iter()
            .rposition(|message| message.role == MessageRole::User)
        else {
            break;
        };
        if latest_user <= 1 {
            break;
        }
        let end = messages[2..latest_user]
            .iter()
            .position(|message| message.role == MessageRole::User)
            .map_or(latest_user, |offset| offset + 2);
        messages.drain(1..end);
    }
    if estimated_tokens(&messages) > prompt_budget {
        // Tool-call metadata cannot be truncated without breaking the assistant/tool
        // protocol. Reserve it and message framing before distributing the remaining
        // character budget, with the latest user instruction receiving first priority.
        let framing_tokens = messages
            .iter()
            .map(|message| {
                message
                    .tool_calls
                    .iter()
                    .map(|call| call.name.len() + call.arguments.to_string().len())
                    .sum::<usize>()
                    .div_ceil(4)
                    .saturating_add(9)
            })
            .sum::<usize>();
        let maximum_chars = prompt_budget
            .saturating_sub(framing_tokens)
            .saturating_mul(4)
            .max(1);
        let latest_user = messages
            .iter()
            .rposition(|message| message.role == MessageRole::User);
        let mut priority = Vec::with_capacity(messages.len());
        if let Some(index) = latest_user {
            priority.push(index);
            priority.extend((index + 1..messages.len()).rev());
        }
        if !messages.is_empty() && latest_user != Some(0) {
            priority.push(0);
        }
        if let Some(index) = latest_user {
            priority.extend((1..index).rev());
        } else {
            priority.extend((1..messages.len()).rev());
        }
        let mut remaining = maximum_chars;
        for index in priority {
            let message = &mut messages[index];
            let count = message.content.chars().count();
            if count <= remaining {
                remaining -= count;
            } else {
                message.content = compact_text(&message.content, remaining);
                remaining = 0;
            }
        }
    }
    messages
}

fn compact_text(text: &str, maximum_chars: usize) -> String {
    if maximum_chars == 0 {
        return String::new();
    }
    if text.chars().count() <= maximum_chars {
        return text.to_owned();
    }
    const MARKER: &str = "\n[earlier content compacted]\n";
    let marker_chars = MARKER.chars().count();
    if maximum_chars <= marker_chars + 2 {
        return text.chars().take(maximum_chars).collect();
    }
    let content_chars = maximum_chars - marker_chars;
    let prefix_chars = content_chars.div_ceil(2);
    let suffix_chars = content_chars / 2;
    let prefix: String = text.chars().take(prefix_chars).collect();
    let suffix: String = text
        .chars()
        .rev()
        .take(suffix_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{prefix}{MARKER}{suffix}")
}

fn estimated_tokens(messages: &[crate::Message]) -> usize {
    messages
        .iter()
        .map(|message| {
            (message.content.chars().count()
                + message
                    .tool_calls
                    .iter()
                    .map(|call| call.name.len() + call.arguments.to_string().len())
                    .sum::<usize>())
            .div_ceil(4)
            .saturating_add(8)
        })
        .sum()
}

fn estimated_json_tokens(values: &[Value]) -> usize {
    if values.is_empty() {
        return 0;
    }
    serde_json::to_string(values)
        .map(|json| {
            json.chars()
                .count()
                .div_ceil(3)
                .saturating_add(values.len().saturating_mul(8))
        })
        .unwrap_or(0)
}

struct CancellationGuard {
    client: reqwest::Client,
    url: String,
    id: String,
    armed: bool,
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let client = self.client.clone();
        let url = self.url.clone();
        let id = self.id.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                // A scoped request ID prevents cancellation of another child/model request.
                for delay in [0, 250, 750] {
                    if delay > 0 {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                    if let Ok(response) = client
                        .delete(&url)
                        .query(&[("conv_id", &id)])
                        .timeout(Duration::from_secs(2))
                        .send()
                        .await
                        && response.status().is_success()
                    {
                        return;
                    }
                }
                eprintln!(
                    "[remote model cancellation was requested but not confirmed by the endpoint]"
                );
            });
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

#[derive(Deserialize)]
struct ResponseToolCall {
    id: Option<String>,
    function: ResponseFunction,
}

#[derive(Deserialize)]
struct ResponseFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Default, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::{Message, ToolCategory, ToolRisk, ToolSpec};
    use axum::{
        Json, Router,
        extract::State,
        routing::{get, post},
    };

    #[tokio::test]
    async fn streaming_thoughts_and_calls_are_ordered_and_scoped_cancel_reaches_server() {
        use axum::{
            body::Body,
            extract::Query,
            http::{HeaderMap, Response},
            routing::delete,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let ids = Arc::new(Mutex::new(Vec::<String>::new()));
        let post_ids = ids.clone();
        let delete_ids = ids.clone();
        let app = Router::new().route("/v1/chat/completions", post(move |headers: HeaderMap, Json(body): Json<Value>| {
            let ids=post_ids.clone();
            async move {
                assert_eq!(body["stream"],true);
                ids.lock().unwrap().push(headers["x-conversation-id"].to_str().unwrap().into());
                if body["messages"][0]["content"] == "hang" {
                    std::future::pending::<()>().await;
                }
                let events = [
                    json!({"choices":[{"delta":{"content":"<think>private 🦀 example {\"tool\":\"shell\"}</think>"}}]}),
                    json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"file_read","arguments":"{\"path\":"}}]}}]}),
                    json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"note.txt\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":8}})
                ];
                let wire=events.iter().map(|v|format!("data: {v}\r\n\r\n")).collect::<String>()+"data: [DONE]\r\n\r\n";
                Response::builder().header("content-type","text/event-stream").body(Body::from(wire)).unwrap()
            }
        })).route("/v1/stream",delete(move |Query(params):Query<std::collections::HashMap<String,String>>| {
            let ids=delete_ids.clone(); async move {ids.lock().unwrap().push(format!("DELETE:{}",params["conv_id"]));"ok"}
        }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider =
            LlamaCppProvider::new(LlamaCppConfig::new(format!("http://{address}"))).unwrap();
        let request = |text: &str| CompletionRequest {
            progress: None,
            messages: vec![Message::new(MessageRole::User, text)],
            tools: vec![],
            allow_tool_calls: true,
            temperature: None,
        };
        let turn = provider.complete(request("stream")).await.unwrap();
        assert!(turn.content.is_empty());
        assert_eq!(
            turn.reasoning.as_deref(),
            Some("private 🦀 example {\"tool\":\"shell\"}")
        );
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].arguments, json!({"path":"note.txt"}));
        assert_eq!(turn.usage.completion, 8);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(150),
                provider.complete(request("hang"))
            )
            .await
            .is_err()
        );
        for _ in 0..50 {
            if ids.lock().unwrap().len() == 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let ids = ids.lock().unwrap();
        assert_eq!(ids.len(), 3, "{ids:?}");
        assert_ne!(ids[0], ids[1]);
        assert_eq!(ids[2], format!("DELETE:{}", ids[1]));
        server.abort();
    }
    #[tokio::test]
    async fn distinct_native_calls_with_equal_arguments_survive_parsing() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route("/v1/chat/completions", post(|| async {
            Json(json!({"choices":[{"message":{"content":"", "tool_calls":[
                {"id":"first", "type":"function", "function":{"name":"file_read","arguments":"{\"path\":\"a\"}"}},
                {"id":"second", "type":"function", "function":{"name":"file_read","arguments":"{\"path\":\"a\"}"}}
            ]}}]}))
        }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider =
            LlamaCppProvider::new(LlamaCppConfig::new(format!("http://{address}"))).unwrap();
        let turn = provider
            .complete(CompletionRequest {
                progress: None,
                messages: vec![Message::new(MessageRole::User, "read twice")],
                tools: vec![],
                allow_tool_calls: true,
                temperature: None,
            })
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].id, "first");
        assert_eq!(turn.tool_calls[1].id, "second");
        server.abort();
    }

    #[tokio::test]
    async fn advertised_text_tools_and_stream_rejection_keep_tools_available() {
        use axum::http::StatusCode;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::<Value>::new()));
        let recorded = requests.clone();
        let app = Router::new()
            .route("/v1/models", get(|| async { Json(json!({"data":[{"id":"text-model","capabilities":{"tool_calls":false}}]})) }))
            .route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push(body.clone());
                    if body["stream"] == true {
                        return (StatusCode::BAD_REQUEST, Json(json!({"error":"stream unsupported"})));
                    }
                    (StatusCode::OK, Json(json!({"choices":[{"message":{"content":"<think>Example {\"tool\":\"file_read\",\"args\":{\"path\":\"wrong\"}}</think>{\"tool\":\"file_read\",\"args\":{\"path\":\"note.txt\"}}"}}]})))
                }
            }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider =
            LlamaCppProvider::new(LlamaCppConfig::new(format!("http://{address}"))).unwrap();
        provider.health().await.unwrap();
        let turn = provider
            .complete(CompletionRequest {
                progress: None,
                messages: vec![Message::new(MessageRole::User, "read")],
                tools: vec![ToolSpec {
                    name: "file_read".into(),
                    description: "Read a file".into(),
                    category: ToolCategory::Internal,
                    risk: ToolRisk::ReadOnly,
                    parameters: json!({"type":"object","properties":{"path":{"type":"string"}}}),
                }],
                allow_tool_calls: true,
                temperature: None,
            })
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].arguments["path"], "note.txt");
        assert!(turn.content.is_empty());
        assert!(turn.reasoning.unwrap().contains("wrong"));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|r| {
            r.get("tools").is_none()
                && r["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("TEXT TOOL INTERFACE")
        }));
        assert_eq!(requests[1]["stream"], false);
        assert!(requests[1].get("stream_options").is_none());
        server.abort();
    }

    async fn capture_completion_request(
        State(requests): State<Arc<Mutex<Vec<Value>>>>,
        Json(request): Json<Value>,
    ) -> Json<Value> {
        requests.lock().expect("request capture lock").push(request);
        Json(json!({
            "choices": [{"message": {"content": "done", "tool_calls": []}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}
        }))
    }

    async fn send_and_capture(request: CompletionRequest, reasoning_effort: Option<&str>) -> Value {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/chat/completions", post(capture_completion_request))
            .with_state(Arc::clone(&requests));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut config = LlamaCppConfig::new(format!("http://{address}"));
        config.reasoning_effort = reasoning_effort.map(str::to_owned);
        let provider = LlamaCppProvider::new(config).unwrap();
        provider.complete(request).await.unwrap();

        let request = requests
            .lock()
            .expect("request capture lock")
            .first()
            .cloned()
            .expect("captured completion request");
        server.abort();
        request
    }

    #[test]
    fn context_budget_keeps_system_and_latest_task() {
        let messages = vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "x".repeat(8_000)),
            Message::new(MessageRole::Assistant, "old answer"),
            Message::new(MessageRole::User, "latest task"),
        ];
        let prepared = prepare_messages(messages, Some(1_000), 100, 200, 0);
        assert_eq!(prepared.first().expect("system").role, MessageRole::System);
        assert_eq!(prepared.last().expect("latest").content, "latest task");
        assert!(estimated_tokens(&prepared) <= 700);
    }

    #[test]
    fn context_trimming_does_not_leave_leading_orphan_tool_results() {
        let messages = vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::Assistant, "x".repeat(8_000)),
            Message::tool("call", "old tool result"),
            Message::new(MessageRole::User, "latest task"),
        ];
        let prepared = prepare_messages(messages, Some(1_000), 100, 200, 0);
        assert_ne!(
            prepared.get(1).map(|message| message.role),
            Some(MessageRole::Tool)
        );
    }

    #[test]
    fn tool_round_compaction_never_drops_the_active_user_task() {
        let task = "write the file, read it, search for it, then run the shell check";
        let messages = vec![
            Message::new(MessageRole::System, "system guidance ".repeat(1_000)),
            Message::new(MessageRole::User, task),
            Message::assistant(
                "",
                None,
                vec![ToolCall {
                    id: "write".into(),
                    name: "file_write".into(),
                    arguments: json!({"path":"smoke/hello.txt","content":"smooth tools"}),
                }],
            ),
            Message::tool("write", "Written: smoke/hello.txt\nSize: 12 chars, 1 lines"),
        ];
        let prepared = prepare_messages(messages, Some(4_096), 512, 1_024, 2_000);
        assert!(
            prepared
                .iter()
                .any(|message| { message.role == MessageRole::User && message.content == task })
        );
        assert!(prepared.iter().any(|message| {
            message.role == MessageRole::Tool && message.content.contains("smoke/hello.txt")
        }));
        assert!(estimated_tokens(&prepared) <= 560);
    }

    #[test]
    fn compact_text_preserves_both_ends_and_unicode_boundaries() {
        let text = format!("start 🦀 {} finish", "middle ".repeat(20));
        let compacted = compact_text(&text, 42);
        assert!(compacted.starts_with("start"));
        assert!(compacted.ends_with("finish"));
        assert_eq!(compacted.chars().count(), 42);
    }

    #[test]
    fn advertised_context_window_is_read_from_llama_model_metadata() {
        let models = json!({
            "data": [{
                "id": "active-model.gguf",
                "meta": {"n_ctx": 4096, "n_ctx_train": 262144}
            }]
        });
        assert_eq!(advertised_context_tokens(&models), Some(4_096));
        assert_eq!(advertised_model(&models), Some("active-model.gguf"));
        assert_eq!(
            advertised_context_tokens(&json!({"n_ctx": "8192"})),
            Some(8_192)
        );
        assert_eq!(advertised_context_tokens(&json!({"data": []})), None);
        assert_eq!(
            advertised_model(&json!({"models": [{"name": "local"}]})),
            Some("local")
        );
        let props = json!({
            "model_alias": "active-alias",
            "default_generation_settings": {"n_ctx": 16_384}
        });
        assert_eq!(advertised_context_tokens(&props), Some(16_384));
        assert_eq!(advertised_model(&props), Some("active-alias"));
    }

    #[tokio::test]
    async fn health_discovers_the_active_model_and_runtime_context_from_server_metadata() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
            .route(
                "/props",
                get(|| async {
                    Json(json!({
                        "model_alias":"props-alias",
                        "default_generation_settings":{"n_ctx":16_384}
                    }))
                }),
            )
            .route(
                "/v1/models",
                get(|| async {
                    Json(json!({
                        "data":[{"id":"active-model.gguf","meta":{"n_ctx":4_096}}]
                    }))
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let provider =
            LlamaCppProvider::new(LlamaCppConfig::new(format!("http://{address}"))).unwrap();
        provider.health().await.unwrap();
        assert_eq!(provider.model(), Some("active-model.gguf"));
        assert_eq!(provider.context_tokens(), Some(16_384));
        server.abort();
    }

    #[tokio::test]
    async fn request_serializes_tool_arguments_as_json_strings_and_omits_default_reasoning() {
        let request = send_and_capture(
            CompletionRequest {
                progress: None,
                messages: vec![
                    Message::new(MessageRole::System, "system"),
                    Message::new(MessageRole::User, "read the file"),
                    Message::assistant(
                        "",
                        None,
                        vec![ToolCall {
                            id: "call-1".into(),
                            name: "file_read".into(),
                            arguments: json!({"path": "notes.txt", "line": 7}),
                        }],
                    ),
                    Message::tool("call-1", "contents"),
                ],
                tools: vec![ToolSpec {
                    name: "file_read".into(),
                    description: "Read a file".into(),
                    category: ToolCategory::Internal,
                    risk: ToolRisk::ReadOnly,
                    parameters: json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }),
                }],
                allow_tool_calls: true,
                temperature: None,
            },
            None,
        )
        .await;

        let arguments = request
            .pointer("/messages/2/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .expect("tool arguments use the OpenAI JSON-string wire format");
        assert_eq!(
            serde_json::from_str::<Value>(arguments).unwrap(),
            json!({"path": "notes.txt", "line": 7})
        );
        assert_eq!(
            request.pointer("/messages/2/role"),
            Some(&json!("assistant"))
        );
        assert_eq!(
            request.pointer("/messages/2/tool_calls/0/id"),
            Some(&json!("call-1"))
        );
        assert_eq!(
            request.pointer("/messages/2/tool_calls/0/type"),
            Some(&json!("function"))
        );
        assert_eq!(
            request.pointer("/messages/2/tool_calls/0/function/name"),
            Some(&json!("file_read"))
        );
        assert_eq!(request.pointer("/messages/3/role"), Some(&json!("tool")));
        assert_eq!(
            request.pointer("/messages/3/tool_call_id"),
            Some(&json!("call-1"))
        );
        assert_eq!(
            request.pointer("/messages/3/content"),
            Some(&json!("contents"))
        );
        assert!(request.get("reasoning_effort").is_none());
    }

    #[tokio::test]
    async fn request_includes_reasoning_effort_only_when_explicitly_configured() {
        let request = send_and_capture(
            CompletionRequest {
                progress: None,
                messages: vec![Message::new(MessageRole::User, "hello")],
                tools: Vec::new(),
                allow_tool_calls: false,
                temperature: None,
            },
            Some("  high  "),
        )
        .await;
        assert_eq!(request.get("reasoning_effort"), Some(&json!("high")));
    }

    #[test]
    fn reasoning_effort_is_opt_in_and_blank_values_are_omitted() {
        assert!(
            LlamaCppProvider::new(LlamaCppConfig::new("http://localhost:8080"))
                .unwrap()
                .config
                .reasoning_effort
                .is_none()
        );
        let mut config = LlamaCppConfig::new("http://localhost:8080");
        config.reasoning_effort = Some("  high  ".into());
        assert_eq!(
            LlamaCppProvider::new(config)
                .unwrap()
                .config
                .reasoning_effort
                .as_deref(),
            Some("high")
        );
        let mut config = LlamaCppConfig::new("http://localhost:8080");
        config.reasoning_effort = Some("  ".into());
        assert!(
            LlamaCppProvider::new(config)
                .unwrap()
                .config
                .reasoning_effort
                .is_none()
        );
    }

    #[tokio::test]
    async fn completion_request_temperature_overrides_provider_default() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(None::<Value>));
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(
                    |State(captured): State<Arc<Mutex<Option<Value>>>>,
                     Json(payload): Json<Value>| async move {
                        *captured.lock().unwrap() = Some(payload);
                        Json(json!({
                            "choices": [{"message": {
                                "content": "ok",
                                "reasoning_content": null,
                                "tool_calls": []
                            }}]
                        }))
                    },
                ),
            )
            .with_state(Arc::clone(&captured));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let provider =
            LlamaCppProvider::new(LlamaCppConfig::new(format!("http://{address}"))).unwrap();

        let turn = provider
            .complete(CompletionRequest {
                progress: None,
                messages: vec![Message::new(MessageRole::User, "hello")],
                tools: Vec::new(),
                allow_tool_calls: false,
                temperature: Some(0.45),
            })
            .await
            .unwrap();

        assert_eq!(turn.content, "ok");
        let temperature = captured.lock().unwrap().as_ref().unwrap()["temperature"]
            .as_f64()
            .expect("numeric temperature");
        assert!((temperature - 0.45).abs() < 1e-6);
        server.abort();
    }

    #[test]
    fn tool_schemas_are_reserved_inside_the_prompt_budget() {
        let messages = vec![
            Message::new(MessageRole::System, "s".repeat(8_000)),
            Message::new(MessageRole::User, "latest task"),
        ];
        let without_tools = prepare_messages(messages.clone(), Some(4_096), 2_048, 1_024, 0);
        let with_tools = prepare_messages(messages, Some(4_096), 2_048, 1_024, 2_000);
        assert!(estimated_tokens(&with_tools) < estimated_tokens(&without_tools));
        let estimated = estimated_tokens(&with_tools);
        assert!(estimated <= 560, "estimated {estimated} tokens");
    }

    #[test]
    fn completion_limit_scales_down_to_the_detected_context() {
        assert_eq!(bounded_completion_tokens(8_192, Some(4_096)), 1_024);
        assert_eq!(bounded_completion_tokens(512, Some(4_096)), 512);
        assert_eq!(bounded_completion_tokens(8_192, None), 8_192);
        assert_eq!(bounded_completion_tokens(-1, Some(4_096)), -1);
    }

    #[test]
    fn endpoint_accepts_server_roots_and_standard_v1_base_urls() {
        let root = LlamaCppProvider::new(LlamaCppConfig::new("http://localhost:8080")).unwrap();
        assert_eq!(
            root.endpoint("/v1/chat/completions"),
            "http://localhost:8080/v1/chat/completions"
        );
        let versioned =
            LlamaCppProvider::new(LlamaCppConfig::new("https://example.test/v1/")).unwrap();
        assert_eq!(
            versioned.endpoint("/v1/chat/completions"),
            "https://example.test/v1/chat/completions"
        );
        assert_eq!(
            versioned.endpoint("/v1/models"),
            "https://example.test/v1/models"
        );
    }
}
