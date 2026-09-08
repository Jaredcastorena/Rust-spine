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

impl GroundingDecision {
    /// Claim-free output has no factual training label, rather than zero coverage.
    pub fn risk_target(&self) -> Option<f32> {
        (self.claim_count > 0).then(|| {
            (0.6 * (1.0 - self.report.coverage) + 0.4 * self.report.contradiction).clamp(0.0, 1.0)
        })
    }
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
        coverage_threshold: f32,
    ) -> spine_heart::Result<GroundingDecision> {
        let claims = self.extractor.extract(response);
        let report = self.verifier.verify(&claims, evidence)?;
        let needs_repair = needs_repair(claims.len(), &report, coverage_threshold);
        Ok(GroundingDecision {
            claim_count: claims.len(),
            report,
            needs_repair,
        })
    }
}

fn needs_repair(claim_count: usize, report: &NliReport, coverage_threshold: f32) -> bool {
    let coverage_threshold = if coverage_threshold.is_finite() {
        coverage_threshold.clamp(0.0, 1.0)
    } else {
        0.5
    };
    claim_count > 0 && (report.coverage < coverage_threshold || report.contradiction >= 0.5)
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

    #[test]
    fn claim_free_output_does_not_train_the_risk_field() {
        let mut decision = GroundingDecision {
            claim_count: 0,
            report: NliReport::default(),
            needs_repair: false,
        };
        assert_eq!(decision.risk_target(), None);
        decision.claim_count = 1;
        decision.report.coverage = 0.0;
        assert_eq!(decision.risk_target(), Some(0.6));
        decision.report.coverage = 1.0;
        assert_eq!(decision.risk_target(), Some(0.0));
    }

    #[test]
    fn dynamic_coverage_threshold_controls_the_host_gate() {
        let report = NliReport {
            coverage: 0.7,
            contradiction: 0.1,
            ..NliReport::default()
        };
        assert!(!needs_repair(1, &report, 0.65));
        assert!(needs_repair(1, &report, 0.8));
        assert!(!needs_repair(0, &report, 1.0));
    }

    #[test]
    fn contradiction_and_malformed_threshold_remain_safe() {
        let contradiction = NliReport {
            coverage: 1.0,
            contradiction: 0.5,
            ..NliReport::default()
        };
        assert!(needs_repair(1, &contradiction, 0.0));
        let coverage = NliReport {
            coverage: 0.49,
            contradiction: 0.0,
            ..NliReport::default()
        };
        assert!(needs_repair(1, &coverage, f32::NAN));
    }
}
