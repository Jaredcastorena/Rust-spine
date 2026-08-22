use std::{error::Error as _, time::Duration};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};

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
            reasoning_effort: Some("low".into()),
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
}

impl LlamaCppProvider {
    pub fn new(mut config: LlamaCppConfig) -> Result<Self> {
        config.base_url = config.base_url.trim_end_matches('/').to_owned();
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
        Ok(Self { config, client })
    }

    pub async fn health(&self) -> Result<()> {
        let health = self
            .client
            .get(format!("{}/health", self.config.base_url))
            .send()
            .await;
        if health
            .as_ref()
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        let models = self
            .client
            .get(format!("{}/v1/models", self.config.base_url))
            .send()
            .await
            .map_err(provider_error)?;
        if models.status().is_success() {
            Ok(())
        } else {
            Err(http_error(models).await)
        }
    }
}

#[async_trait]
impl ModelProvider for LlamaCppProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
        let prepared = prepare_messages(
            request.messages,
            self.config.max_context_tokens,
            self.config.context_safety_tokens,
            self.config.max_tokens,
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
                                        "arguments": call.arguments,
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
        let mut payload = json!({
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": false,
        });
        if let Some(model) = &self.config.model {
            payload["model"] = json!(model);
        }
        if let Some(reasoning_effort) = &self.config.reasoning_effort {
            payload["reasoning_effort"] = json!(reasoning_effort);
        }
        if request.allow_tool_calls && !tools.is_empty() {
            payload["tools"] = Value::Array(tools);
            payload["tool_choice"] = json!("auto");
        }

        let mut attempt = 0_u32;
        let response = loop {
            let response = self
                .client
                .post(format!("{}/v1/chat/completions", self.config.base_url))
                .json(&payload)
                .send()
                .await;
            match response {
                Ok(response) => {
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
            return Err(http_error(response).await);
        }
        let response: ChatResponse = response.json().await.map_err(provider_error)?;
        let message = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::Provider("provider returned no choices".into()))?
            .message;
        let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
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
            tool_calls.push(ToolCall {
                id: call.id.unwrap_or_else(|| format!("llama-call-{index}")),
                name: call.function.name,
                arguments,
            });
        }
        Ok(ModelTurn {
            content: message.content.unwrap_or_default(),
            reasoning: message.reasoning_content,
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

fn prepare_messages(
    mut messages: Vec<crate::Message>,
    maximum_tokens: Option<usize>,
    safety_tokens: usize,
    completion_tokens: i64,
) -> Vec<crate::Message> {
    let Some(maximum_tokens) = maximum_tokens else {
        return messages;
    };
    let completion_tokens = completion_tokens.max(0) as usize;
    let prompt_budget = maximum_tokens
        .saturating_sub(safety_tokens)
        .saturating_sub(completion_tokens)
        .max(256);
    while messages.len() > 2 && estimated_tokens(&messages) > prompt_budget {
        messages.remove(1);
        while messages
            .get(1)
            .is_some_and(|message| message.role == MessageRole::Tool)
        {
            messages.remove(1);
        }
    }
    if estimated_tokens(&messages) > prompt_budget {
        let maximum_chars = prompt_budget.saturating_mul(4).max(1_024);
        let mut remaining = maximum_chars;
        for message in messages.iter_mut().rev() {
            let count = message.content.chars().count();
            if count <= remaining {
                remaining -= count;
            } else {
                let suffix: String = message
                    .content
                    .chars()
                    .rev()
                    .take(remaining)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                message.content = format!("[earlier content compacted]\n{suffix}");
                remaining = 0;
            }
        }
    }
    messages
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
    use crate::Message;

    #[test]
    fn context_budget_keeps_system_and_latest_task() {
        let messages = vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "x".repeat(8_000)),
            Message::new(MessageRole::Assistant, "old answer"),
            Message::new(MessageRole::User, "latest task"),
        ];
        let prepared = prepare_messages(messages, Some(1_000), 100, 200);
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
        let prepared = prepare_messages(messages, Some(1_000), 100, 200);
        assert_ne!(
            prepared.get(1).map(|message| message.role),
            Some(MessageRole::Tool)
        );
    }
}
