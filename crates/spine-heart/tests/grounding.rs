use std::sync::{Arc, Mutex};

use spine_heart::{
    GroundingEvidence, GroundingMethod, GroundingVerdict as V, GroundingVerifier, HeartError,
    NliLabelOrder, NliModel, Result,
};

#[derive(Clone)]
struct Model {
    logits: Vec<[f32; 3]>,
    seen: Arc<Mutex<Vec<(String, String)>>>,
    bad_length: bool,
    fail: bool,
}

impl Model {
    fn new(logits: [f32; 3]) -> Self {
        Self {
            logits: vec![logits],
            seen: Arc::default(),
            bad_length: false,
            fail: false,
        }
    }
}

impl NliModel for Model {
    fn label_order(&self) -> NliLabelOrder {
        NliLabelOrder::CROSS_ENCODER_MINILM
    }
    fn predict_logits(&self, pairs: &[(String, String)]) -> Result<Vec<[f32; 3]>> {
        let mut seen = self.seen.lock().unwrap();
        let start = seen.len();
        seen.extend_from_slice(pairs);
        if self.fail {
            return Err(HeartError::Model("injected inference failure".into()));
        }
        if self.bad_length {
            return Ok(Vec::new());
        }
        Ok((0..pairs.len())
            .map(|i| self.logits[(start + i) % self.logits.len()])
            .collect())
    }
}

fn source(text: &str) -> GroundingEvidence {
    GroundingEvidence {
        id: "source-1".into(),
        text: text.into(),
    }
}

fn exact(response: &str, evidence: &str) -> spine_heart::GroundingReport {
    let mut model = Model::new([10.0, -10.0, -10.0]);
    model.fail = true; // Any unexpected semantic veto is a failure.
    GroundingVerifier::new(model)
        .unwrap()
        .verify(response, &[source(evidence)], 0.8)
        .unwrap()
}

#[test]
fn exact_strings_preserve_case_unicode_punctuation_and_leading_zeroes() {
    for value in [
        "CORRECT_LABEL",
        "CORRECT_Ω",
        "café-07",
        "Α7",
        "🧭-7",
        "0042",
        "rc-2.4",
        "/tmp/a.b",
        "space ; and punctuation!",
    ] {
        let response = format!("The recovery label is \"{value}\".");
        let report = exact(&response, &response);
        assert_eq!(report.claims.len(), 1, "{value}");
        assert_eq!(report.claims[0].verdict, V::Supported, "{value}");
        assert_eq!(report.claims[0].method, GroundingMethod::Exact);
        assert_eq!(report.evaluated_pairs, 0);
        assert_eq!(report.coverage(), Some(1.0));
        assert!(!report.needs_repair());
    }
    for (left, right) in [
        ("AbC9", "abC9"),
        ("0042", "42"),
        ("Α7", "A7"),
        ("rc-2.4", "rc_2.4"),
        ("café", "cafe"),
    ] {
        let report = exact(
            &format!("The access code is \"{left}\"."),
            &format!("The access code is \"{right}\"."),
        );
        assert_eq!(report.claims[0].verdict, V::Contradicted, "{left}/{right}");
    }
}

#[test]
fn exact_ascii_and_unicode_issue_grammar_with_unquoted_source() {
    for suffix in ["LABEL", "Ω"] {
        let correct = format!("The recovery label is CORRECT_{suffix}.");
        let good = exact(
            &format!("The recovery label is \"CORRECT_{suffix}\"."),
            &correct,
        );
        let bad = exact(
            &format!("The recovery label is \"WRONG_{suffix}\"."),
            &correct,
        );
        assert_eq!(good.claims[0].verdict, V::Supported);
        assert_eq!(bad.claims[0].verdict, V::Contradicted);
    }
}

#[test]
fn entity_attribute_and_tense_must_match() {
    let model = Model::new([0.0, 10.0, 0.0]);
    let seen = model.seen.clone();
    let verifier = GroundingVerifier::new(model).unwrap();
    for evidence in ["Noor's badge code is K4.", "Zoë's locker code is K4."] {
        let report = verifier
            .verify("Zoë's badge code is K4.", &[source(evidence)], 0.8)
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified);
    }
    assert!(seen.lock().unwrap().is_empty());
    let report = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0]))
        .unwrap()
        .verify("The code is K4.", &[source("The code was K4.")], 0.8)
        .unwrap();
    assert_eq!(report.claims[0].method, GroundingMethod::Semantic);
    assert_eq!(report.claims[0].verdict, V::Unverified);
}

#[test]
fn negation_and_conflicting_sources_are_not_flattened() {
    for (response, evidence, verdict) in [
        ("The code is not Q8.", "The code is Q7.", V::Supported),
        ("The code isn't Q7.", "The code is Q7.", V::Contradicted),
        ("The code is Q7.", "The code is not Q7.", V::Contradicted),
        ("The code is Q7.", "The code is not Q8.", V::Unverified),
        ("The code is not Q7.", "The code is not Q8.", V::Unverified),
        ("The code is not Q7.", "The code is not Q7.", V::Supported),
        (
            "The code is Q7.",
            "The code is Q7. The code is Q8.",
            V::Conflicting,
        ),
        (
            "The code is Q7.",
            "The code is Q7. The code is not Q7.",
            V::Conflicting,
        ),
    ] {
        let report = exact(response, evidence);
        assert_eq!(report.claims[0].verdict, verdict, "{response} / {evidence}");
    }
    let verifier = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0])).unwrap();
    let evidence = vec![
        GroundingEvidence {
            id: "one".into(),
            text: "The code is Q7.".into(),
        },
        GroundingEvidence {
            id: "two".into(),
            text: "The code is Q8.".into(),
        },
    ];
    let report = verifier.verify("The code is Q7.", &evidence, 0.8).unwrap();
    assert_eq!(report.claims[0].verdict, V::Conflicting);
    assert_eq!(report.claims[0].evidence_ids, ["one", "two"]);
}

#[test]
fn separate_clauses_cannot_hide_unverified_content() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0])).unwrap();
    for response in [
        "The code is Q7. Unicorns dance on Jupiter.",
        "The code is Q7 and unicorns are dancing on Jupiter.",
        "The code is Q7, and unicorns are dancing on Jupiter.",
        "The code is Q7; unicorns are dancing on Jupiter.",
        "- The code is Q7.\n- Unicorns dance on Jupiter.",
        "1. The code is Q7.\n2. Unicorns dance on Jupiter.",
    ] {
        let report = verifier
            .verify(response, &[source("The code is Q7.")], 0.8)
            .unwrap();
        assert_eq!(report.claim_count(), 2, "{response}: {:?}", report.claims);
        assert_eq!(report.claims[0].verdict, V::Supported, "{response}");
        assert_eq!(report.claims[1].verdict, V::Unverified, "{response}");
        assert_eq!(report.coverage(), Some(0.5));
        assert!(report.needs_repair());
    }
}

#[test]
fn no_recognized_attribute_still_gets_a_report_without_evidence() {
    let report = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0]))
        .unwrap()
        .verify("Whales can fly. 🧭", &[], 0.8)
        .unwrap();
    assert_eq!(report.claim_count(), 2);
    assert!(report.claims.iter().all(|c| c.verdict == V::Unverified));
    assert!(report.needs_repair());
    assert_eq!(report.risk_target(), Some(1.0));
}

#[test]
fn semantic_fallback_uses_original_premise_then_hypothesis() {
    let model = Model::new([0.0, 10.0, 0.0]);
    let seen = model.seen.clone();
    let verifier = GroundingVerifier::new(model).unwrap();
    let response = "Cedar uses Q7 as the badge code.";
    let premise = "Cedar's badge code is Q7.";
    let report = verifier.verify(response, &[source(premise)], 0.8).unwrap();
    assert_eq!(*seen.lock().unwrap(), [(premise.into(), response.into())]);
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(report.claims[0].method, GroundingMethod::Semantic);
    assert_eq!(report.evaluated_pairs, 1);
}

#[test]
fn general_numeric_and_semantic_claims_have_fallback() {
    let model = Model::new([0.0, 10.0, 0.0]);
    let seen = model.seen.clone();
    let report = GroundingVerifier::new(model)
        .unwrap()
        .verify(
            "The bridge spans 10 kilometers. Cedar won the competition.",
            &[source(
                "The bridge is ten kilometers long. Cedar took first place.",
            )],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claim_count(), 2);
    assert!(report.claims.iter().all(|c| c.verdict == V::Supported));
    let pairs = seen.lock().unwrap();
    assert_eq!(report.claims[0].method, GroundingMethod::Exact);
    assert!(pairs.contains(&(
        "Cedar took first place.".into(),
        "Cedar won the competition.".into()
    )));
}

#[test]
fn semantic_assignment_source_can_be_paraphrased_without_synthesizing_text() {
    let model = Model::new([0.0, 10.0, 0.0]);
    let seen = model.seen.clone();
    let source_text = "The recovery label assigned to this task reads Q7.";
    let response = "The recovery label is Q7.";
    let report = GroundingVerifier::new(model)
        .unwrap()
        .verify(response, &[source(source_text)], 0.8)
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(
        *seen.lock().unwrap(),
        [(source_text.into(), response.into())]
    );
}

#[test]
fn semantic_support_contradiction_and_unknown_are_per_statement() {
    let model = Model {
        logits: vec![[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]],
        ..Model::new([0.0; 3])
    };
    let report = GroundingVerifier::new(model)
        .unwrap()
        .verify(
            "Whales can fly.",
            &[source("One source."), source("Another source.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Conflicting);
    let model = Model {
        logits: vec![[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ..Model::new([0.0; 3])
    };
    let report = GroundingVerifier::new(model)
        .unwrap()
        .verify(
            "Whales can swim. Whales can fly.",
            &[source("Whales swim.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(report.claims[1].verdict, V::Unverified);
    assert!(report.needs_repair());
}

#[test]
fn acknowledgments_questions_commands_and_hypotheticals_are_explicit() {
    let mut model = Model::new([0.0; 3]);
    model.fail = true;
    let verifier = GroundingVerifier::new(model).unwrap();
    for response in [
        "Understood, thanks.",
        "Is the code Q8?",
        "Please check whether the code is Q8.",
        "If the code is Q8, ask for confirmation.",
        "I do not know the code.",
        "Example sentence only: \"The code is Q8.\" This is not a factual answer.",
    ] {
        let report = verifier
            .verify(response, &[source("The code is Q7.")], 0.8)
            .unwrap();
        assert_eq!(report.claim_count(), 0, "{response}");
        assert!(!report.claims.is_empty());
        assert!(report.claims.iter().all(|c| c.verdict == V::NonFactual));
        assert!(!report.needs_repair());
        assert_eq!(report.coverage(), None);
        assert_eq!(report.risk_target(), None);
    }
}

#[test]
fn uncertain_prefaces_and_command_reasons_do_not_hide_assertions() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0])).unwrap();
    for response in [
        "I think the code is Q8.",
        "Maybe the code is Q8.",
        "Use Q8 because the code is Q8.",
        "I don't know the code, but the status is active.",
    ] {
        let report = verifier.verify(response, &[], 0.8).unwrap();
        assert!(report.claim_count() > 0, "{response}");
        assert!(report.needs_repair());
    }
}

#[test]
fn nonfactual_sources_do_not_establish_facts() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for evidence in [
        "Is the code Q8?",
        "If the code is Q8, ask for confirmation.",
        "Please say the code is Q8.",
    ] {
        let report = verifier
            .verify("The code is Q8.", &[source(evidence)], 0.8)
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified, "{evidence}");
        assert_eq!(report.evaluated_pairs, 0);
    }
}

#[test]
fn inference_failure_nonfinite_scores_and_wrong_batch_length_are_errors() {
    for model in [
        Model {
            fail: true,
            ..Model::new([0.0; 3])
        },
        Model {
            bad_length: true,
            ..Model::new([0.0; 3])
        },
        Model::new([f32::NAN, 0.0, 0.0]),
        Model::new([0.0, f32::INFINITY, 0.0]),
        Model::new([0.0, 0.0, f32::NEG_INFINITY]),
    ] {
        assert!(
            GroundingVerifier::new(model)
                .unwrap()
                .verify("Birds fly.", &[source("Birds have wings.")], 0.8)
                .is_err()
        );
    }
}

#[test]
fn thresholds_are_validated_before_any_claim_free_shortcut() {
    let verifier = GroundingVerifier::new(Model::new([0.0; 3])).unwrap();
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 0.0, 0.49, 1.01] {
        assert!(verifier.verify("Thanks.", &[], threshold).is_err());
    }
}

#[test]
fn too_many_response_statements_are_explicitly_unverified() {
    let response = "The code is Q7. ".repeat(150);
    let report = exact(&response, "The code is Q7.");
    assert!(report.truncated);
    assert_eq!(report.claims.len(), 129);
    assert_eq!(report.claims.last().unwrap().verdict, V::Unverified);
    assert!(report.needs_repair());
}

#[test]
fn response_byte_limit_is_unicode_safe_and_never_confirms_a_fragment() {
    let response = format!("The code is Q7. {}", "Ω".repeat(40_000));
    let report = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0]))
        .unwrap()
        .verify(&response, &[source("The code is Q7.")], 0.8)
        .unwrap();
    assert!(report.truncated);
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(report.claims[1].verdict, V::Unverified);
    assert!(report.needs_repair());
}

#[test]
fn omitted_evidence_cannot_result_in_fully_verified_answer() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let mut evidence = (0..64)
        .map(|i| GroundingEvidence {
            id: i.to_string(),
            text: "The code is Q7.".into(),
        })
        .collect::<Vec<_>>();
    evidence.push(source("The code is Q8."));
    let report = verifier.verify("The code is Q7.", &evidence, 0.8).unwrap();
    assert!(report.truncated);
    assert_eq!(report.claims[0].verdict, V::Unverified);
    assert!(report.needs_repair());
    let report = verifier
        .verify("Birds fly.", &[source(&"Birds fly. ".repeat(2000))], 0.8)
        .unwrap();
    assert!(report.truncated);
    assert_eq!(report.claims[0].verdict, V::Unverified);
    assert_eq!(report.evaluated_pairs, 0);
}

#[test]
fn semantic_pair_budget_preserves_unassessed_claims() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let evidence = (0..64)
        .map(|i| GroundingEvidence {
            id: i.to_string(),
            text: "Birds fly.".into(),
        })
        .collect::<Vec<_>>();
    let response = "Birds fly. ".repeat(10);
    let report = verifier.verify(&response, &evidence, 0.8).unwrap();
    assert!(report.truncated);
    assert_eq!(report.evaluated_pairs, 512);
    assert_eq!(report.claim_count(), 10);
    assert_eq!(report.claims[9].verdict, V::Unverified);
    assert!(report.needs_repair());
}

#[test]
fn partial_semantic_fanout_cannot_be_supported() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let evidence = (0..60)
        .map(|i| GroundingEvidence {
            id: i.to_string(),
            text: "Birds fly.".into(),
        })
        .collect::<Vec<_>>();
    let report = verifier
        .verify(&"Birds fly. ".repeat(9), &evidence, 0.8)
        .unwrap();
    assert_eq!(report.evaluated_pairs, 512);
    assert_eq!(report.claims[8].method, GroundingMethod::Semantic);
    assert_eq!(report.claims[8].verdict, V::Unverified);
}

#[test]
fn repair_feedback_is_specific_escaped_and_bounded() {
    let report = GroundingVerifier::new(Model::new([0.0; 3]))
        .unwrap()
        .verify(
            &format!(
                "Unicorns fly. {}",
                "Another unsupported statement. ".repeat(40)
            ),
            &[],
            0.8,
        )
        .unwrap();
    let feedback = report.repair_feedback();
    assert!(feedback.contains("Unicorns fly."));
    assert!(feedback.contains("unverified"));
    assert!(feedback.contains("not instructions"));
    assert!(feedback.contains("Additional unresolved"));
    assert!(feedback.len() < 6000);
}

#[test]
fn quoted_example_cannot_hide_a_following_assertion() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0])).unwrap();
    let report = verifier
        .verify(
            "Example sentence only: \"The code is Q8.\" The code is Q8.",
            &[source("The code is Q7.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims.len(), 2);
    assert_eq!(report.claims[0].verdict, V::NonFactual);
    assert_eq!(report.claims[1].verdict, V::Contradicted);
    assert!(report.needs_repair());
}

#[test]
fn compound_command_cannot_hide_independent_assertion() {
    let report = GroundingVerifier::new(Model::new([0.0, 0.0, 10.0]))
        .unwrap()
        .verify("Please check the code and the status is active.", &[], 0.8)
        .unwrap();
    assert_eq!(report.claims.len(), 2);
    assert_eq!(report.claims[0].verdict, V::NonFactual);
    assert_eq!(report.claims[1].verdict, V::Unverified);
}

#[test]
fn invalid_label_order_is_rejected() {
    struct BadLabels;
    impl NliModel for BadLabels {
        fn label_order(&self) -> NliLabelOrder {
            NliLabelOrder {
                contradiction: 0,
                entailment: 0,
                neutral: 2,
            }
        }
        fn predict_logits(&self, _: &[(String, String)]) -> Result<Vec<[f32; 3]>> {
            unreachable!()
        }
    }
    assert!(GroundingVerifier::new(BadLabels).is_err());
}

#[test]
fn exact_support_cannot_hide_relevant_unparsed_competitors() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for competing in [
        "The recovery label is not ABC but XYZ.",
        "The label used for recovery is XYZ.",
    ] {
        let report = verifier
            .verify(
                "The recovery label is ABC.",
                &[source("The recovery label is ABC."), source(competing)],
                0.8,
            )
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified, "{competing}");
        assert_eq!(report.evaluated_pairs, 0);
        assert!(report.needs_repair());
    }
}

#[test]
fn question_and_conditional_clauses_cannot_hide_a_declaration() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for response in [
        "The code is Q8 and is the status active?",
        "If you need help, ask, but the code is Q8.",
    ] {
        let report = verifier
            .verify(response, &[source("The code is Q7.")], 0.8)
            .unwrap();
        assert!(
            report.claims.iter().any(|c| c.verdict == V::Contradicted),
            "{response}: {:?}",
            report.claims
        );
        assert!(report.needs_repair());
    }
    let report = verifier
        .verify(
            "If the code is Q8 and the status is active, ask for help.",
            &[],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claim_count(), 0);
    let report = verifier
        .verify("Is the code Q8 and is the status active?", &[], 0.8)
        .unwrap();
    assert_eq!(report.claim_count(), 0);
}

#[test]
fn command_like_nouns_remain_declarations() {
    for (response, evidence) in [
        ("Run status is failed.", "Run status is successful."),
        ("Output code is Q8.", "Output code is Q7."),
    ] {
        let report = exact(response, evidence);
        assert_eq!(report.claim_count(), 1);
        assert_eq!(report.claims[0].verdict, V::Contradicted);
    }
}

#[test]
fn entity_paths_preserve_case() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let report = verifier
        .verify(
            "File /tmp/A checksum is ABC.",
            &[source("File /tmp/a checksum is ABC.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Unverified);
    assert_eq!(report.evaluated_pairs, 0);
}

#[test]
fn upstream_truncation_marker_preserves_uncertainty() {
    let report = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0]))
        .unwrap()
        .verify(
            "The code is Q7.",
            &[
                source("The code is Q7."),
                source("The code is Q8.\n[truncated]"),
            ],
            0.8,
        )
        .unwrap();
    assert!(report.truncated);
    assert_eq!(report.claims[0].verdict, V::Unverified);
}

#[test]
fn quantitative_guards_override_overconfident_semantic_mismatches() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (response, evidence) in [
        (
            "The Alder tank can hold 680 litres.",
            "The maximum storage capacity of the Alder tank is 640 litres.",
        ),
        (
            "The archive keeps exactly two copies of each poster.",
            "The archive keeps exactly three copies of each poster.",
        ),
        (
            "The Lantern release is scheduled for May 18, 2027.",
            "The Lantern release ledger lists May 19, 2027.",
        ),
        (
            "The Lantern release is scheduled for May 18, 2027.",
            "The Lantern release ledger lists June 18, 2027.",
        ),
    ] {
        let report = verifier.verify(response, &[source(evidence)], 0.8).unwrap();
        assert_eq!(
            report.claims[0].verdict,
            V::Contradicted,
            "{response} / {evidence}"
        );
    }
}

#[test]
fn equivalent_quantities_do_not_inherit_spurious_nli_contradiction() {
    let verifier = GroundingVerifier::new(Model::new([10.0, 0.0, 0.0])).unwrap();
    for (response, evidence) in [
        ("The count is 1.", "The count is 1.0."),
        (
            "The marked Elm trail is 2,500 metres long.",
            "The total length of the marked Elm trail is 2.5 kilometres.",
        ),
        (
            "The marked Elm trail is 2.5 kilometres long.",
            "The marked Elm trail is 250000 centimetres long.",
        ),
    ] {
        let report = verifier.verify(response, &[source(evidence)], 0.8).unwrap();
        assert_eq!(
            report.claims[0].verdict,
            V::Supported,
            "{response} / {evidence}"
        );
        assert_eq!(report.evaluated_pairs, 0);
    }
}

#[test]
fn quoted_literal_guard_applies_outside_assignment_grammar() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let report = verifier.verify("The scanner expects the literal key \"gate–9\".", &[source("The scanner expects the literal key \"gate-9\". Only the ASCII hyphen-minus is accepted.")], 0.8).unwrap();
    assert_eq!(report.claims[0].verdict, V::Contradicted);
}

#[test]
fn distinct_location_relations_and_unresolved_pronouns_are_not_supported() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (response, evidence) in [
        ("Maren lives in Bergen.", "Maren works in Bergen."),
        (
            "Her office is on the third floor.",
            "Rowan and Ellis have offices. Rowan's office is on the third floor. Ellis's office is on the fourth floor.",
        ),
    ] {
        let report = verifier.verify(response, &[source(evidence)], 0.8).unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified);
        assert!(report.needs_repair());
    }
}

#[test]
fn sentence_level_source_conflicts_are_preserved_with_one_id() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    let response = "The archive keeps exactly two copies of each poster.";
    let evidence = format!("{response} The archive keeps exactly three copies of each poster.");
    let report = verifier
        .verify(response, &[source(&evidence)], 0.8)
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Conflicting);
    assert_eq!(report.claims[0].evidence_ids, ["source-1"]);
}

#[test]
fn additional_explicit_nonfactual_forms_are_classified_conservatively() {
    let verifier = GroundingVerifier::new(Model::new([10.0, 0.0, 0.0])).unwrap();
    for response in [
        "Thanks; understood. I appreciate the clarification.",
        "I cannot determine who owns the workshop from these records.",
        "I do not have enough information to give an opening date.",
        "In this invented example, the robot has four arms. Treat that as fiction for the exercise.",
        "Sample wording only: \"The library closes at midnight.\" This is an example sentence, not a statement of its hours.",
        "I would prefer the blue cover for this draft.",
        "Please do the following:\n- Find the latest register.\n- Check the room number.\n- Ask the coordinator to confirm it.",
    ] {
        let report = verifier.verify(response, &[], 0.8).unwrap();
        assert_eq!(report.claim_count(), 0, "{response}: {:?}", report.claims);
        assert!(!report.needs_repair());
    }
}

#[test]
fn quantitative_equivalence_cannot_merge_different_owners_or_dimensions() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (response, evidence) in [
        (
            "Alice's Elm trail is 2.5 kilometres long.",
            "Bob's Elm trail is 2500 metres long.",
        ),
        (
            "The Elm building is 12 metres high.",
            "The Elm building is 12 metres wide.",
        ),
    ] {
        let report = verifier.verify(response, &[source(evidence)], 0.8).unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified, "{response}");
    }
}

#[test]
fn numerical_paraphrases_without_a_scalar_rule_use_original_nli_pairs() {
    let model = Model::new([0.0, 10.0, 0.0]);
    let seen = model.seen.clone();
    let premise = "The group bought ten stamps.";
    let hypothesis = "The team purchased 10 stamps.";
    let report = GroundingVerifier::new(model)
        .unwrap()
        .verify(hypothesis, &[source(premise)], 0.8)
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(*seen.lock().unwrap(), [(premise.into(), hypothesis.into())]);
}

#[test]
fn numeric_comparison_is_exact_for_large_adjacent_and_tiny_values() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (claim, fact) in [
        ("10,000,000,000", "10,000,000,001"),
        ("9007199254740992", "9007199254740993"),
        ("0.00000000000001", "0.00000000000002"),
        (
            "0.00000000000000000000000000000000000001",
            "0.00000000000000000000000000000000000002",
        ),
    ] {
        let report = verifier
            .verify(
                &format!("The count is {claim}."),
                &[source(&format!("The count is {fact}."))],
                0.8,
            )
            .unwrap();
        assert_eq!(
            report.claims[0].verdict,
            V::Contradicted,
            "{claim} / {fact}"
        );
        assert_eq!(report.evaluated_pairs, 0);
    }
    for value in [
        "999999999999999999999999999999999999999999999999",
        "0.0000000000000000000000000000000000000001",
    ] {
        let text = format!("The count is {value}.");
        let report = verifier.verify(&text, &[source(&text)], 0.8).unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified);
        assert_eq!(report.evaluated_pairs, 0);
    }
}

#[test]
fn quoted_literals_and_paths_do_not_match_substrings() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (response, evidence) in [
        (
            "The scanner expects the literal key \"gate-9\".",
            "The scanner expects the literal key \"PREFIXgate-9SUFFIX\".",
        ),
        (
            "The file /tmp/A contains a certificate.",
            "The file /tmp/A2 contains a certificate.",
        ),
    ] {
        let report = verifier.verify(response, &[source(evidence)], 0.8).unwrap();
        assert_ne!(report.claims[0].verdict, V::Supported);
        assert_eq!(report.evaluated_pairs, 0);
    }
}

#[test]
fn explicit_attribute_and_time_distractors_do_not_veto_numeric_support() {
    let verifier = GroundingVerifier::new(Model::new([10.0, 0.0, 0.0])).unwrap();
    for (response, matching, distractor) in [
        (
            "The Foundry vessel has a capacity of 300 litres.",
            "The Foundry vessel capacity is 300 litres.",
            "The Foundry vessel maximum operating temperature is 350 degrees Celsius.",
        ),
        (
            "In 2022, the entry fee was 12 credits.",
            "In 2022, the entry fee was 12 credits.",
            "In 2024, the entry fee changed to 15 credits.",
        ),
    ] {
        let report = verifier
            .verify(response, &[source(matching), source(distractor)], 0.8)
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Supported, "{response}");
    }
}

#[test]
fn iso_and_named_calendar_dates_compare_without_nli_rounding() {
    let verifier = GroundingVerifier::new(Model::new([10.0, 0.0, 0.0])).unwrap();
    let report = verifier
        .verify(
            "The Meadow exhibition opens on August 14, 2026.",
            &[source("The Meadow exhibition opening date is 2026-08-14.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Supported);
    assert_eq!(report.evaluated_pairs, 0);
    let report = verifier
        .verify(
            "The Meadow exhibition opens on August 15, 2026.",
            &[source("The Meadow exhibition opening date is 2026-08-14.")],
            0.8,
        )
        .unwrap();
    assert_eq!(report.claims[0].verdict, V::Contradicted);
}

#[test]
fn claimed_numeric_subset_retains_the_original_semantic_pair() {
    for (hypothesis, premise) in [
        (
            "None of the six inspected crates were wet.",
            "The inspector examined six crates and recorded that not one of them was wet.",
        ),
        (
            "The Orchard desk opens on Mondays and closes at 18:00 that day.",
            "The Orchard desk is open every Monday from 09:00 to 18:00.",
        ),
    ] {
        let model = Model::new([0.0, 10.0, 0.0]);
        let seen = model.seen.clone();
        let report = GroundingVerifier::new(model)
            .unwrap()
            .verify(hypothesis, &[source(premise)], 0.8)
            .unwrap();
        assert_eq!(
            report.claims[0].verdict,
            V::Supported,
            "{hypothesis}: {:?}",
            report.claims
        );
        assert_eq!(*seen.lock().unwrap(), [(premise.into(), hypothesis.into())]);
    }
}

#[test]
fn a_uniqueness_constraint_does_not_create_an_unparsed_value_conflict() {
    let report = exact(
        "Cedar's badge code is not Q8.",
        "Cedar's badge code is Q7. Cedar has exactly one badge code.",
    );
    assert_eq!(report.claims[0].verdict, V::Supported);
    let report = exact(
        "Cedar's badge code is Q7.",
        "Cedar's badge code is Q7. Cedar has exactly one badge code Q8.",
    );
    assert_eq!(report.claims[0].verdict, V::Unverified);
}

#[test]
fn source_anaphora_preserves_context_and_does_not_score_antecedents_as_competitors() {
    for (hypothesis, premise) in [
        (
            "Talia manages the costume archive.",
            "Talia joined the theatre last year. She manages the costume archive.",
        ),
        (
            "The Cedar rover's serial number is \"RV-13\".",
            "Cedar is a rover. Its serial number is \"RV-13\".",
        ),
    ] {
        let model = Model::new([0.0, 10.0, 0.0]);
        let seen = model.seen.clone();
        let report = GroundingVerifier::new(model)
            .unwrap()
            .verify(hypothesis, &[source(premise)], 0.8)
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Supported, "{hypothesis}");
        assert_eq!(*seen.lock().unwrap(), [(premise.into(), hypothesis.into())]);
    }
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for premise in [
        "Talia and Nora joined the theatre. She manages the costume archive.",
        "Talia met Nora. She manages the costume archive.",
    ] {
        let report = verifier
            .verify(
                "Talia manages the costume archive.",
                &[source(premise)],
                0.8,
            )
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified);
    }
}

#[test]
fn unsupported_exponent_forms_cannot_escape_numeric_checks() {
    let verifier = GroundingVerifier::new(Model::new([0.0, 10.0, 0.0])).unwrap();
    for (claim, fact) in [("1e100", "1e101"), ("1.2e-10", "1.3e-10"), ("1e2", "100")] {
        let report = verifier
            .verify(
                &format!("The count is {claim}."),
                &[source(&format!("The count is {fact}."))],
                0.8,
            )
            .unwrap();
        assert_eq!(report.claims[0].verdict, V::Unverified);
        assert_eq!(report.evaluated_pairs, 0);
    }
}
