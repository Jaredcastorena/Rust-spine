use std::{collections::BTreeSet, sync::OnceLock};

use regex::Regex;
use serde_json::{Value, json};

use crate::ToolCall;

pub(crate) fn parse_text_tool_calls(
    text: &str,
    allowed: &BTreeSet<String>,
    id_offset: usize,
) -> (String, Vec<ToolCall>) {
    if allowed.is_empty() || text.trim().is_empty() {
        return (text.to_owned(), Vec::new());
    }
    let mut candidates = fenced_candidates(text);
    candidates.extend(marked_candidates(text));
    candidates.extend(laguna_candidates(text));
    candidates.extend(bare_json_candidates(text));
    candidates.sort_by(|left, right| {
        left.remove_start
            .cmp(&right.remove_start)
            .then_with(|| right.remove_end.cmp(&left.remove_end))
            .then_with(|| left.call_start.cmp(&right.call_start))
    });

    let mut accepted = Vec::new();
    let mut removals = Vec::new();
    let mut seen = BTreeSet::new();
    let mut covered_until = 0;
    let mut current_removal = None;
    for candidate in candidates {
        let removal = (candidate.remove_start, candidate.remove_end);
        if current_removal != Some(removal) {
            if candidate.remove_start < covered_until {
                continue;
            }
            covered_until = candidate.remove_end;
            current_removal = Some(removal);
            removals.push(removal);
        }
        let key = (candidate.name.clone(), candidate.arguments.to_string());
        if seen.insert(key) {
            accepted.push(candidate);
        }
    }

    let mut cleaned = String::with_capacity(text.len());
    let mut cursor = 0;
    for (start, end) in removals {
        cleaned.push_str(&text[cursor..start]);
        cursor = end;
    }
    cleaned.push_str(&text[cursor..]);
    let calls = accepted
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| ToolCall {
            id: format!("text-call-{}", id_offset + index),
            name: candidate.name,
            arguments: candidate.arguments,
        })
        .collect();
    (cleaned.trim().to_owned(), calls)
}

struct Candidate {
    call_start: usize,
    remove_start: usize,
    remove_end: usize,
    name: String,
    arguments: Value,
}

fn fenced_candidates(text: &str) -> Vec<Candidate> {
    static FENCE: OnceLock<Regex> = OnceLock::new();
    static LEGACY_MARKER: OnceLock<Regex> = OnceLock::new();
    let fence = FENCE.get_or_init(|| {
        Regex::new(r"(?is)```([^\n`]*)\n(.*?)(?:```|$)").expect("static tool fence regex")
    });
    let legacy_marker = LEGACY_MARKER.get_or_init(|| {
        Regex::new(r"(?i)^tool_call[ \t\r]*\n").expect("static legacy tool marker regex")
    });
    fence
        .captures_iter(text)
        .filter_map(|captures| {
            let whole = captures.get(0)?;
            let language = captures.get(1)?.as_str().trim().to_ascii_lowercase();
            let body = captures.get(2)?;
            let (payload, payload_start) = if matches!(language.as_str(), "tool_call" | "json") {
                (body.as_str(), body.start())
            } else {
                let trimmed = body.as_str().trim_start();
                let marker = legacy_marker.find(trimmed)?;
                let payload = &trimmed[marker.end()..];
                let offset = body.as_str().len() - trimmed.len() + marker.end();
                (payload, body.start() + offset)
            };
            let candidates = json_call_candidates(payload, payload_start, None);
            (!candidates.is_empty()).then(|| {
                candidates
                    .into_iter()
                    .map(|mut candidate| {
                        candidate.remove_start = whole.start();
                        candidate.remove_end = whole.end();
                        candidate
                    })
                    .collect::<Vec<_>>()
            })
        })
        .flatten()
        .collect()
}

fn marked_candidates(text: &str) -> Vec<Candidate> {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    let marker =
        MARKER.get_or_init(|| Regex::new(r"(?i)tool_call\s*\n").expect("static tool marker regex"));
    marker
        .find_iter(text)
        .filter_map(|marker| {
            let mut candidates = json_call_candidates(&text[marker.end()..], marker.end(), Some(1));
            let mut candidate = candidates.pop()?;
            candidate.remove_start = marker.start();
            Some(candidate)
        })
        .collect()
}

fn laguna_candidates(text: &str) -> Vec<Candidate> {
    static CALL: OnceLock<Regex> = OnceLock::new();
    static ARGUMENT: OnceLock<Regex> = OnceLock::new();
    let call = CALL.get_or_init(|| {
        Regex::new(r"(?is)<tool_call>\s*([a-zA-Z_][a-zA-Z0-9_.-]*)(.*?)</tool_call>")
            .expect("static Laguna call regex")
    });
    let argument = ARGUMENT.get_or_init(|| {
        Regex::new(r"(?is)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>")
            .expect("static Laguna argument regex")
    });
    call.captures_iter(text)
        .filter_map(|captures| {
            let whole = captures.get(0)?;
            let name = captures.get(1)?.as_str().trim().to_owned();
            let body = captures.get(2).map_or("", |item| item.as_str());
            let mut arguments = serde_json::Map::new();
            for captures in argument.captures_iter(body) {
                let Some(key) = captures
                    .get(1)
                    .map(|item| decode_markup(item.as_str().trim()))
                else {
                    continue;
                };
                if key.is_empty() {
                    continue;
                }
                let raw = captures
                    .get(2)
                    .map_or_else(String::new, |item| decode_markup(item.as_str().trim()));
                let value = serde_json::from_str(&raw).unwrap_or_else(|_| json!(raw));
                arguments.insert(key, value);
            }
            Some(Candidate {
                call_start: whole.start(),
                remove_start: whole.start(),
                remove_end: whole.end(),
                name,
                arguments: Value::Object(arguments),
            })
        })
        .collect()
}

fn bare_json_candidates(text: &str) -> Vec<Candidate> {
    json_call_candidates(text, 0, None)
}

fn json_call_candidates(text: &str, absolute_start: usize, limit: Option<usize>) -> Vec<Candidate> {
    let bytes = text.as_bytes();
    let mut candidates = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let Some(relative) = bytes[start..].iter().position(|byte| *byte == b'{') else {
            break;
        };
        let object_start = start + relative;
        let Some(object_end) = json_object_end(bytes, object_start) else {
            start = object_start + 1;
            continue;
        };
        if let Some((name, arguments)) = parse_json_call(&text[object_start..object_end]) {
            candidates.push(Candidate {
                call_start: absolute_start + object_start,
                remove_start: absolute_start + object_start,
                remove_end: absolute_start + object_end,
                name,
                arguments,
            });
            if limit.is_some_and(|limit| candidates.len() >= limit) {
                break;
            }
            start = object_end;
        } else {
            start = object_start + 1;
        }
    }
    candidates
}

fn json_object_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0_u32;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'{' => depth = depth.saturating_add(1),
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_json_call(payload: &str) -> Option<(String, Value)> {
    let value: Value = serde_json::from_str(payload.trim()).ok()?;
    let function = value.get("function");
    let name = value
        .get("tool")
        .or_else(|| value.get("name"))
        .or_else(|| function.and_then(|item| item.get("name")))?
        .as_str()?
        .trim();
    if name.is_empty() {
        return None;
    }
    let raw_arguments = value
        .get("args")
        .or_else(|| value.get("arguments"))
        .or_else(|| function.and_then(|item| item.get("arguments")))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match raw_arguments {
        Value::String(raw) => serde_json::from_str(&raw).ok()?,
        Value::Null => json!({}),
        value => value,
    };
    arguments.is_object().then(|| (name.to_owned(), arguments))
}

fn decode_markup(value: &str) -> String {
    value
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed() -> BTreeSet<String> {
        ["file_read", "shell"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn parses_mixed_fenced_and_laguna_calls_without_losing_prose() {
        let text = "Before\n```tool_call\n{\"tool\":\"file_read\",\"args\":{\"path\":\"a\"}}\n```\nMiddle\n<tool_call>shell<arg_key>command</arg_key><arg_value>\"pwd\"</arg_value></tool_call>\nAfter";
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 0);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].arguments["command"], "pwd");
        assert!(cleaned.contains("Before"));
        assert!(cleaned.contains("Middle"));
        assert!(cleaned.contains("After"));
        assert!(!cleaned.contains("tool_call"));
    }

    #[test]
    fn parses_nested_bare_json_and_deduplicates_overlapping_fence_matches() {
        let text = "```json\n{\"tool\":\"shell\",\"args\":{\"command\":\"printf '{x}'\"}}\n```";
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 4);
        assert!(cleaned.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "text-call-4");
        assert_eq!(calls[0].arguments["command"], "printf '{x}'");
    }

    #[test]
    fn parses_and_strips_legacy_unfenced_and_truncated_forms() {
        let text = concat!(
            "Before\n```\ntool_call\n",
            "{\"tool\":\"file_read\",\"args\":{\"path\":\"a\"}}\n```\n",
            "Middle\ntool_call\n",
            "{\"tool\":\"shell\",\"args\":{\"command\":\"pwd\"}}\nAfter"
        );
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 0);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].name, "shell");
        assert_eq!(cleaned, "Before\n\nMiddle\n\nAfter");

        let truncated =
            "Checking\n```tool_call\n{\"tool\":\"shell\",\"args\":{\"command\":\"pwd\"}}";
        let (cleaned, calls) = parse_text_tool_calls(truncated, &allowed(), 0);
        assert_eq!(calls.len(), 1);
        assert_eq!(cleaned, "Checking");
    }

    #[test]
    fn parses_multiple_calls_inside_one_recognized_fence() {
        let text = concat!(
            "```json\n",
            "{\"tool\":\"file_read\",\"args\":{\"path\":\"a\"}}\n",
            "{\"tool\":\"shell\",\"args\":{\"command\":\"pwd\"}}\n",
            "```"
        );
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 0);
        assert_eq!(calls.len(), 2);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn unknown_tools_are_forwarded_for_explicit_harness_feedback() {
        let text = "A tool_call is not markup. {\"tool\":\"unknown\",\"args\":{}}";
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 0);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "unknown");
        assert_eq!(cleaned, "A tool_call is not markup.");
    }

    #[test]
    fn duplicate_calls_are_executed_once_and_all_serializations_are_removed() {
        let text = "before {\"tool\":\"shell\",\"args\":{\"command\":\"pwd\"}} between {\"tool\":\"shell\",\"args\":{\"command\":\"pwd\"}} after";
        let (cleaned, calls) = parse_text_tool_calls(text, &allowed(), 0);
        assert_eq!(calls.len(), 1);
        assert_eq!(cleaned, "before  between  after");
    }
}
