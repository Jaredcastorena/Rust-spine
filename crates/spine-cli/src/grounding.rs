use std::path::Path;

use spine_heart::{ClaimExtractor, NliReport, NliVerifier};
use spine_models::{MiniLmNli, NliAssets};

pub struct GroundingGate {
    extractor: ClaimExtractor,
    verifier: NliVerifier<MiniLmNli>,
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
}
