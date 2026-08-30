use std::path::Path;

use spine_heart::{ClaimExtractor, NliReport, NliVerifier};
use spine_models::{MiniLmNli, NliAssets};

const TERMINAL_CAVEAT: &str = "I could not verify every factual claim against the available evidence. Treat unsupported details as uncertain; I will not present them as established fact.";

pub struct GroundingGate {
    extractor: ClaimExtractor,
    verifier: NliVerifier<MiniLmNli>,
}

pub fn append_terminal_caveat(outcome: &mut spine_runtime::RunOutcome) -> bool {
    if outcome.response.contains(TERMINAL_CAVEAT) {
        return false;
    }
    if !outcome.response.trim().is_empty() {
        outcome.response.push_str("\n\n");
    }
    outcome.response.push_str(TERMINAL_CAVEAT);
    if let Some(message) = outcome
        .messages
        .iter_mut()
        .rev()
        .find(|message| message.role == spine_runtime::MessageRole::Assistant)
    {
        if !message.content.trim().is_empty() {
            message.content.push_str("\n\n");
        }
        message.content.push_str(TERMINAL_CAVEAT);
    } else {
        outcome.messages.push(spine_runtime::Message::new(
            spine_runtime::MessageRole::Assistant,
            TERMINAL_CAVEAT,
        ));
    }
    true
}

pub struct GroundingDecision {
    pub claim_count: usize,
    pub report: NliReport,
    pub needs_repair: bool,
}

impl GroundingGate {
    pub fn load(directory: impl AsRef<Path>) -> spine_heart::Result<Self> {
        Ok(Self {
            extractor: ClaimExtractor::new()?,
            verifier: NliVerifier::new(MiniLmNli::load(NliAssets::from_directory(directory))?, 3)?,
        })
    }

    pub fn verify(
        &self,
        response: &str,
        evidence: &[String],
    ) -> spine_heart::Result<GroundingDecision> {
        let claims = self.extractor.extract(response);
        let report = self.verifier.verify(&claims, evidence)?;
        let needs_repair =
            !claims.is_empty() && (report.coverage < 0.5 || report.contradiction >= 0.5);
        Ok(GroundingDecision {
            claim_count: claims.len(),
            report,
            needs_repair,
        })
    }
}

pub fn evidence_from_recall_and_messages(
    recalled: &str,
    messages: &[spine_runtime::Message],
) -> Vec<String> {
    let mut evidence = serde_json::from_str::<Vec<serde_json::Value>>(recalled)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            item.get("text")
                .and_then(|text| text.as_str())
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    evidence.extend(
        messages
            .iter()
            .filter(|message| message.role == spine_runtime::MessageRole::Tool)
            .map(|message| message.content.clone()),
    );
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_collection_ignores_malformed_recall() {
        let messages = vec![spine_runtime::Message::tool("one", "tool evidence")];
        assert_eq!(
            evidence_from_recall_and_messages("not-json", &messages),
            ["tool evidence"]
        );
    }

    #[test]
    fn terminal_caveat_updates_visible_and_persisted_answers_once() {
        let mut outcome = spine_runtime::RunOutcome {
            response: "draft".into(),
            stopped_gracefully: false,
            checkpoint: None,
            completed_tool_calls: 0,
            completed_tool_rounds: 0,
            usage: spine_runtime::TokenUsage::default(),
            messages: vec![spine_runtime::Message::new(
                spine_runtime::MessageRole::Assistant,
                "draft",
            )],
            host_plan: None,
        };
        assert!(append_terminal_caveat(&mut outcome));
        assert!(!append_terminal_caveat(&mut outcome));
        assert_eq!(outcome.response, outcome.messages[0].content);
        assert!(
            outcome
                .response
                .contains("could not verify every factual claim")
        );
    }
}
