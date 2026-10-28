use crate::{ModelProgress, Result, RuntimeError};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Remove closed and truncated inline thoughts before tool parsing or display.
pub(crate) fn separate_thoughts(text: &str, supplied: Option<&str>) -> (String, Option<String>) {
    let lower = text.to_ascii_lowercase();
    let mut visible = String::new();
    let mut thoughts = Vec::new();
    if let Some(s) = supplied.filter(|s| !s.is_empty()) {
        thoughts.push(s.to_owned());
    }
    let mut at = 0;
    while let Some(relative) = lower[at..].find("<think>") {
        let start = at + relative;
        visible.push_str(&text[at..start]);
        let body = start + 7;
        if let Some(end) = lower[body..].find("</think>") {
            let thought = text[body..body + end].trim();
            if !thought.is_empty() {
                thoughts.push(thought.into());
            }
            at = body + end + 8;
        } else {
            let thought = text[body..].trim();
            if !thought.is_empty() {
                thoughts.push(thought.into());
            }
            at = text.len();
            break;
        }
    }
    visible.push_str(&text[at..]);
    (
        visible.trim().to_owned(),
        (!thoughts.is_empty()).then(|| thoughts.join("\n")),
    )
}

#[derive(Default)]
struct Accumulator {
    content: String,
    reasoning: String,
    calls: BTreeMap<u64, Value>,
    usage: Value,
    finished: bool,
}

impl Accumulator {
    fn event(&mut self, data: &str) -> Result<()> {
        if data.trim() == "[DONE]" {
            self.finished = true;
            return Ok(());
        }
        let value: Value = serde_json::from_str(data)?;
        if let Some(error) = value.get("error") {
            return Err(RuntimeError::Provider(format!("stream error: {error}")));
        }
        if value["usage"].is_object() {
            self.usage = value["usage"].clone();
        }
        for choice in value["choices"].as_array().into_iter().flatten() {
            if choice["index"].as_u64().unwrap_or(0) != 0 {
                continue;
            }
            if choice["finish_reason"].is_string() {
                self.finished = true;
            }
            let delta = &choice["delta"];
            if let Some(s) = delta["content"].as_str() {
                self.content.push_str(s);
            }
            if let Some(s) = delta["reasoning_content"]
                .as_str()
                .or_else(|| delta["reasoning"].as_str())
            {
                self.reasoning.push_str(s);
            }
            for fragment in delta["tool_calls"].as_array().into_iter().flatten() {
                let index = fragment["index"].as_u64().unwrap_or(0);
                let call = self.calls.entry(index).or_insert_with(
                    || json!({"id":null,"type":"function","function":{"name":"","arguments":""}}),
                );
                if let Some(id) = fragment["id"].as_str() {
                    call["id"] = json!(id);
                }
                for key in ["name", "arguments"] {
                    if let Some(part) = fragment["function"][key].as_str() {
                        let mut complete = call["function"][key]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned();
                        complete.push_str(part);
                        call["function"][key] = json!(complete);
                    }
                }
            }
        }
        Ok(())
    }
    fn progress(&self) -> ModelProgress {
        let (mut content, reasoning) = separate_thoughts(&self.content, Some(&self.reasoning));
        // A marker split across frames is held until its meaning is known.
        let lower = content.to_ascii_lowercase();
        for size in (1..7).rev() {
            if lower.ends_with(&"<think>"[..size]) {
                content.truncate(content.len() - size);
                break;
            }
        }
        ModelProgress { content, reasoning }
    }
    fn response(self) -> Result<Value> {
        if !self.finished {
            return Err(RuntimeError::Provider(
                "stream ended before a completion marker; partial tool calls were not executed"
                    .into(),
            ));
        }
        Ok(
            json!({"choices":[{"message":{"role":"assistant","content":self.content,
            "reasoning_content":self.reasoning,"tool_calls":self.calls.into_values().collect::<Vec<_>>()}}],
            "usage":if self.usage.is_object(){self.usage}else{json!({})}}),
        )
    }
}

pub(crate) async fn read_response(
    mut response: reqwest::Response,
    updates: Option<&tokio::sync::mpsc::UnboundedSender<ModelProgress>>,
) -> Result<Value> {
    let mut accumulator = Accumulator::default();
    let mut pending = Vec::new();
    let mut data = String::new();
    while let Some(bytes) = response
        .chunk()
        .await
        .map_err(|e| RuntimeError::Provider(e.to_string()))?
    {
        pending.extend_from_slice(&bytes);
        if pending.len() > 32 * 1024 * 1024 {
            return Err(RuntimeError::Provider("oversized SSE frame".into()));
        }
        while let Some(end) = pending.iter().position(|b| *b == b'\n') {
            let line = String::from_utf8(pending.drain(..=end).collect())
                .map_err(|e| RuntimeError::Provider(e.to_string()))?;
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !data.is_empty() {
                    accumulator.event(data.trim_end_matches('\n'))?;
                    if let Some(updates) = updates {
                        let _ = updates.send(accumulator.progress());
                    }
                    data.clear();
                }
            } else if let Some(value) = line.strip_prefix("data:") {
                data.push_str(value.strip_prefix(' ').unwrap_or(value));
                data.push('\n');
            }
        }
    }
    if !pending.is_empty() {
        let line = String::from_utf8(pending).map_err(|e| RuntimeError::Provider(e.to_string()))?;
        if let Some(value) = line.trim_end_matches('\r').strip_prefix("data:") {
            data.push_str(value.trim_start());
        }
    }
    if !data.is_empty() {
        accumulator.event(data.trim())?;
    }
    accumulator.response()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thought_examples_and_truncation_never_become_visible_calls() {
        let (answer, thought) = separate_thoughts(
            "<think>Example {\"tool\":\"shell\"}</think>Answer <THINK>unfinished",
            None,
        );
        assert_eq!(answer, "Answer");
        assert_eq!(
            thought.as_deref(),
            Some("Example {\"tool\":\"shell\"}\nunfinished")
        );
    }
    #[test]
    fn fragments_keep_tool_order_arguments_reasoning_and_usage() {
        let mut a = Accumulator::default();
        a.event(r#"{"choices":[{"delta":{"reasoning_content":"first ","tool_calls":[{"index":1,"id":"b","function":{"name":"shell","arguments":"{\"command\":"}},{"index":0,"id":"a","function":{"name":"file_read","arguments":"{}"}}]}}]}"#).unwrap();
        a.event(r#"{"choices":[{"delta":{"reasoning_content":"second","tool_calls":[{"index":1,"function":{"arguments":"\"pwd\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":17,"completion_tokens":9}}"#).unwrap();
        let v = a.response().unwrap();
        assert_eq!(
            v["choices"][0]["message"]["reasoning_content"],
            "first second"
        );
        assert_eq!(v["choices"][0]["message"]["tool_calls"][0]["id"], "a");
        assert_eq!(
            v["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"],
            r#"{"command":"pwd"}"#
        );
        assert_eq!(v["usage"]["completion_tokens"], 9);
        assert!(Accumulator::default().response().is_err());
    }
}
