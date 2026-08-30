use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

const MAX_PLAN_STEPS: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PlanStepStatus {
    Pending,
    Active,
    Done,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub index: usize,
    pub text: String,
    pub status: PlanStepStatus,
    pub evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostPlan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
    pub cursor: usize,
}

impl HostPlan {
    pub fn new(goal: impl Into<String>, steps: Vec<String>) -> Option<Self> {
        if !(2..=MAX_PLAN_STEPS).contains(&steps.len()) {
            return None;
        }
        let mut plan = Self {
            goal: goal.into(),
            steps: steps
                .into_iter()
                .enumerate()
                .map(|(index, text)| PlanStep {
                    index: index + 1,
                    text,
                    status: PlanStepStatus::Pending,
                    evidence: String::new(),
                })
                .collect(),
            cursor: 0,
        };
        plan.activate();
        Some(plan)
    }

    pub fn current(&self) -> Option<&PlanStep> {
        self.steps.get(self.cursor)
    }

    pub fn done(&self) -> bool {
        self.steps.is_empty() || self.cursor >= self.steps.len()
    }

    pub fn mark_current_done(&mut self, evidence: impl Into<String>) -> bool {
        let Some(step) = self.steps.get_mut(self.cursor) else {
            return false;
        };
        step.status = PlanStepStatus::Done;
        step.evidence = evidence.into();
        self.cursor += 1;
        self.activate();
        true
    }

    pub fn progress(&self) -> String {
        let total = self.steps.len();
        let header = if self.done() {
            format!("STEPS: done ({total}/{total})")
        } else {
            format!(
                "STEPS: {}/{}  {}",
                self.cursor + 1,
                total,
                self.current().map_or("", |step| step.text.as_str())
            )
        };
        let steps = self.steps.iter().map(|step| {
            let mark = match step.status {
                PlanStepStatus::Pending => ' ',
                PlanStepStatus::Active => '>',
                PlanStepStatus::Done => 'x',
            };
            format!("  [{mark}] {}. {}", step.index, step.text)
        });
        std::iter::once(header)
            .chain(steps)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn activate(&mut self) {
        if let Some(step) = self.steps.get_mut(self.cursor) {
            step.status = PlanStepStatus::Active;
        }
    }
}

pub fn parse_plan_steps(text: &str) -> Vec<String> {
    static FENCE: OnceLock<Regex> = OnceLock::new();
    let fence = FENCE
        .get_or_init(|| Regex::new(r"(?is)```plan\s*\n(.*?)```").expect("static plan fence regex"));
    if let Some(captures) = fence.captures(text) {
        let body = captures.get(1).map_or("", |item| item.as_str()).trim();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
            let values = value
                .as_array()
                .or_else(|| value.get("steps").and_then(|steps| steps.as_array()));
            if let Some(values) = values {
                return clean_steps(
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .map(str::to_owned),
                );
            }
        }
        return numbered_steps(body, false);
    }
    numbered_steps(text, true)
}

pub fn promised_more_work(text: &str) -> bool {
    static PROMISE: OnceLock<Regex> = OnceLock::new();
    static WORD: OnceLock<Regex> = OnceLock::new();
    let promise = PROMISE.get_or_init(|| {
        Regex::new(
            r"(?ix)\b(?:let\s+me|i'll|i\s+will|i\s+am\s+going\s+to|i'm\s+going\s+to|gonna)\s+(?:\w+[\s,]+){0,3}(?:list|read|open|find|search|inspect|check|run|group|summarize|compare|locate|scan|collect|map|pull|look|test|verify|investigate|edit|update|change|fix|implement)\b|\b(?:i'm|i\s+am)\s+(?:now\s+)?(?:reading|opening|searching|inspecting|checking|running|testing|verifying|investigating|editing|updating|fixing|implementing)\b",
        )
        .expect("static work-promise regex")
    });
    let word = WORD.get_or_init(|| Regex::new(r"(?i)[a-z']+").expect("static word regex"));
    let tail = text
        .char_indices()
        .rev()
        .nth(599)
        .map_or(text, |(index, _)| &text[index..]);
    promise.find_iter(tail).any(|matched| {
        !word
            .find_iter(matched.as_str())
            .any(|item| matches!(item.as_str().to_ascii_lowercase().as_str(), "not" | "never"))
    })
}

fn numbered_steps(text: &str, require_action: bool) -> Vec<String> {
    static NUMBERED: OnceLock<Regex> = OnceLock::new();
    static ACTION: OnceLock<Regex> = OnceLock::new();
    let numbered = NUMBERED.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:\d+[.)-]|[-*])\s+(\S.+)$").expect("static numbered regex")
    });
    let action = ACTION.get_or_init(|| {
        Regex::new(r"(?i)\b(list|read|open|find|search|inspect|check|run|group|summarize|compare|locate|scan|collect|map|pull|look|test|verify|investigate|edit|update|change|fix|implement)\b")
            .expect("static plan-action regex")
    });
    clean_steps(numbered.captures_iter(text).filter_map(|captures| {
        let line = captures.get(1)?.as_str().trim();
        (line.chars().count() <= 160 && (!require_action || action.is_match(line)))
            .then(|| line.to_owned())
    }))
}

fn clean_steps(steps: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut cleaned = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for step in steps {
        let step = step.trim().to_owned();
        if !step.is_empty() && step.chars().count() <= 160 && seen.insert(step.to_lowercase()) {
            cleaned.push(step);
        }
    }
    if (2..=MAX_PLAN_STEPS).contains(&cleaned.len()) {
        cleaned
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_and_json_plans_but_not_numbered_essays() {
        assert_eq!(
            parse_plan_steps("```plan\n1. List projects\n2. Read tests\n```"),
            ["List projects", "Read tests"]
        );
        assert_eq!(
            parse_plan_steps(
                r#"```plan
{"steps":["Read foo","Read bar"]}
```"#
            ),
            ["Read foo", "Read bar"]
        );
        assert!(
            parse_plan_steps("1. kangaroo — confidence 0.8\n2. sims — confidence 0.4").is_empty()
        );
        assert!(parse_plan_steps("```plan\n1. Only one\n```").is_empty());
    }

    #[test]
    fn cursor_advances_only_when_the_host_marks_evidence() {
        let mut plan = HostPlan::new("inspect", vec!["List".into(), "Read".into()]).unwrap();
        assert_eq!(plan.current().unwrap().status, PlanStepStatus::Active);
        assert!(plan.mark_current_done("tools"));
        assert_eq!(plan.current().unwrap().text, "Read");
        assert!(plan.mark_current_done("writeup"));
        assert!(plan.done());
        assert!(plan.progress().contains("done (2/2)"));
    }

    #[test]
    fn promise_detection_requires_a_concrete_non_negated_closing_action() {
        assert!(promised_more_work(
            "Let me pull up its structure and inspect the tests."
        ));
        assert!(promised_more_work(
            "I'm going to inspect the failing test next."
        ));
        assert!(!promised_more_work("Let me be precise about that."));
        assert!(!promised_more_work(
            "I will not run that destructive command."
        ));
        assert!(!promised_more_work(
            "I will never inspect those private files."
        ));
        let completed = format!(
            "Let me inspect first. {}That is my complete assessment.",
            "The evidence supports the result. ".repeat(30)
        );
        assert!(!promised_more_work(&completed));
    }
}
