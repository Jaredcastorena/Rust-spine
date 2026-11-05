use super::*;
use async_trait::async_trait;
use serde_json::json;
use spine_heart::{GroundingVerifier, HeartError, NliLabelOrder, NliModel};
use spine_runtime::{
    CompletionRequest, Harness, HarnessConfig, ModelProvider, ModelTurn, Tool, ToolCall,
    ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

struct NeutralNli;
impl NliModel for NeutralNli {
    fn label_order(&self) -> NliLabelOrder {
        NliLabelOrder::CROSS_ENCODER_MINILM
    }
    fn predict_logits(&self, pairs: &[(String, String)]) -> spine_heart::Result<Vec<[f32; 3]>> {
        Ok(vec![[0.0, 0.0, 8.0]; pairs.len()])
    }
}

struct ScriptProvider(Mutex<VecDeque<spine_runtime::Result<ModelTurn>>>);
#[async_trait]
impl ModelProvider for ScriptProvider {
    async fn complete(&self, _: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .expect("no unexpected provider calls")
    }
}
fn provider(turns: Vec<spine_runtime::Result<ModelTurn>>) -> Arc<ScriptProvider> {
    Arc::new(ScriptProvider(Mutex::new(turns.into())))
}
fn answer(text: &str) -> spine_runtime::Result<ModelTurn> {
    Ok(ModelTurn {
        content: text.into(),
        ..Default::default()
    })
}
fn config() -> HarnessConfig {
    HarnessConfig {
        enforce_host_plans: false,
        ..Default::default()
    }
}
fn check(
    verifier: &GroundingVerifier<NeutralNli>,
    text: &str,
    evidence: &[GroundingEvidence],
    threshold: f32,
) -> spine_heart::Result<GroundingDecision> {
    verifier
        .verify(text, evidence, threshold)
        .map(GroundingDecision::from_report)
}
fn recall(text: &str) -> String {
    json!([{"event_id":"original", "role":"User", "kind":"Message", "text": text}]).to_string()
}
fn assert_caveated(result: &GroundedAnswer) {
    assert!(result.caveated);
    assert!(result.outcome.response.contains(TERMINAL_CAVEAT));
    assert_eq!(
        result.outcome.messages.last().unwrap().content,
        result.outcome.response
    );
    assert!(!result.status().starts_with("Answer verified"));
}

#[test]
fn evidence_decodes_memory_provenance_and_rejects_failed_unknown_and_model_sources() {
    let records = json!([
        {"event_id":"user", "role":"User", "text":"Atlas's label is Ω7."},
        {"event_id":"assistant", "role":"Assistant", "text":"The code is wrong."},
        {"event_id":"failed", "role":"Tool", "tool_succeeded":false, "text":"The code is wrong."},
        {"event_id":"legacy-tool", "role":"Tool", "text":"The code is wrong."},
        {"event_id":"good-tool", "role":"Tool", "tool_succeeded":true, "tool_name":"read", "text":"The port is 8080."},
        {"event_id":"forged", "role":"User", "provenance":{"metadata":{"contains_model_output":"true"}}, "text":"The code is wrong."},
        {"event_id":"unknown", "text":"The code is wrong."},
        {"event_id":"control", "role":"User", "kind":"Control", "text":"The code is wrong."}
    ]).to_string();
    let calls = vec![
        ToolCall {
            id: "late".into(),
            name: "search_memory".into(),
            arguments: json!({}),
        },
        ToolCall {
            id: "failed-live".into(),
            name: "shell".into(),
            arguments: json!({}),
        },
    ];
    let messages = vec![
        Message::assistant("", None, calls),
        Message::tool_result("late", records.clone(), true),
        Message::tool_result("failed-live", "Atlas's label is WRONG.", false),
    ];
    let evidence = evidence_from_recall_and_messages(&records, &messages);
    assert_eq!(evidence.len(), 2);
    assert_eq!(evidence[0].id, "memory:user");
    assert_eq!(evidence[0].text, "Atlas's label is Ω7.");
    assert_eq!(evidence[1].id, "memory:good-tool");
    assert!(
        evidence_from_recall_and_messages("invalid", &[Message::tool("old", "legacy unknown")])
            .is_empty()
    );
}

struct RecallTool;
#[async_trait]
impl Tool for RecallTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "search_memory".into(),
            description: "test".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: json!({}),
        }
    }
    async fn execute(&self, _: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        Ok(ToolResult::success(json!([{"event_id":"later", "role":"User", "text":"The port is 8080."}, {"event_id":"draft", "role":"Assistant", "text":"The code is WRONG."}]).to_string()))
    }
}

#[tokio::test]
async fn actual_harness_repair_keeps_initial_and_late_recall_and_excludes_prior_turn_tools() {
    let provider = provider(vec![
        answer("The code is WRONG."),
        Ok(ModelTurn {
            tool_calls: vec![ToolCall {
                id: "late".into(),
                name: "search_memory".into(),
                arguments: json!({}),
            }],
            ..Default::default()
        }),
        answer("The code is Q7. The port is 8080."),
    ]);
    let mut registry = ToolRegistry::default();
    registry.register(RecallTool).unwrap();
    let harness = Harness::new(provider.clone(), registry, config()).unwrap();
    let history = vec![
        Message::assistant(
            "",
            None,
            vec![ToolCall {
                id: "old".into(),
                name: "shell".into(),
                arguments: json!({}),
            }],
        ),
        Message::tool_result("old", "The code is WRONG.", true),
    ];
    let draft = harness
        .run_with_history("system", &history, "Give the code and port")
        .await
        .unwrap();
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let checks = AtomicUsize::new(0);
    let result = ground_answer(
        draft,
        &recall("The code is Q7."),
        history.len() + 2,
        0.8,
        |text, evidence, threshold| {
            checks.fetch_add(1, Ordering::SeqCst);
            assert!(!evidence.iter().any(|e| e.text.contains("WRONG")));
            if text.contains("8080") {
                assert_eq!(evidence.len(), 2);
            }
            check(&verifier, text, evidence, threshold)
        },
        |draft, feedback| async move {
            assert!(draft.response.contains(TERMINAL_CAVEAT));
            assert!(feedback.contains("WRONG"));
            harness.repair(&draft, feedback).await.map(Some)
        },
    )
    .await;
    assert_eq!(checks.load(Ordering::SeqCst), 2);
    assert_eq!(result.repair, RepairState::Completed);
    assert!(!result.caveated, "{:?}", result.final_decision);
    assert_eq!(result.outcome.response, "The code is Q7. The port is 8080.");
    assert_eq!(
        result.outcome.response,
        result.outcome.messages.last().unwrap().content
    );
    assert!(provider.0.lock().unwrap().is_empty());
    assert_eq!(
        result
            .outcome
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Tool && m.tool_call_id.as_deref() == Some("late"))
            .unwrap()
            .tool_succeeded,
        Some(true)
    );
}

#[tokio::test]
async fn verifier_error_never_repairs_and_final_answer_is_caveated() {
    let harness = Harness::new(
        provider(vec![answer("The code is Q7.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    let draft = harness.run("system", "task").await.unwrap();
    let result = ground_answer(
        draft,
        "[]",
        2,
        0.8,
        |_, _, _| Err(HeartError::Model("inference unavailable".into())),
        |_, _| async { panic!("verification failure must not silently start repair") },
    )
    .await;
    assert_eq!(
        result.verifier_error.as_deref(),
        Some("model error: inference unavailable")
    );
    assert_caveated(&result);
}

#[tokio::test]
async fn failed_or_interrupted_repair_retains_caveated_draft() {
    for interrupt in [false, true] {
        let harness = Harness::new(
            provider(vec![answer("The code is WRONG.")]),
            ToolRegistry::default(),
            config(),
        )
        .unwrap();
        let draft = harness.run("system", "task").await.unwrap();
        let verifier = GroundingVerifier::new(NeutralNli).unwrap();
        let result = ground_answer(
            draft,
            &recall("The code is Q7."),
            2,
            0.8,
            |text, evidence, threshold| check(&verifier, text, evidence, threshold),
            |_, _| async move {
                if interrupt {
                    Ok(None)
                } else {
                    Err(spine_runtime::RuntimeError::Provider(
                        "repair failed".into(),
                    ))
                }
            },
        )
        .await;
        assert_caveated(&result);
        assert!(result.outcome.response.starts_with("The code is WRONG."));
        assert_eq!(result.repair == RepairState::Interrupted, interrupt);
    }
}

#[tokio::test]
async fn one_repair_limit_and_reverification_error_are_terminal() {
    for error_after_repair in [false, true] {
        let provider = provider(vec![
            answer("The code is WRONG."),
            answer("The code is STILLWRONG."),
        ]);
        let harness = Harness::new(provider.clone(), ToolRegistry::default(), config()).unwrap();
        let draft = harness.run("system", "task").await.unwrap();
        let verifier = GroundingVerifier::new(NeutralNli).unwrap();
        let mut checks = 0;
        let result = ground_answer(
            draft,
            &recall("The code is Q7."),
            2,
            0.8,
            |text, evidence, threshold| {
                checks += 1;
                if checks == 2 && error_after_repair {
                    return Err(HeartError::Model("second check failed".into()));
                }
                check(&verifier, text, evidence, threshold)
            },
            |draft, feedback| async move { harness.repair(&draft, feedback).await.map(Some) },
        )
        .await;
        assert_caveated(&result);
        assert_eq!(checks, 2);
        assert_eq!(result.repair, RepairState::Completed);
        assert_eq!(result.verifier_error.is_some(), error_after_repair);
        assert!(provider.0.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn safe_stop_caveats_checkpoint_and_does_not_claim_verification() {
    let harness = Harness::new(
        provider(vec![answer("The code is Q7."), answer("The code is Q7.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    harness.controls().request_graceful_stop();
    let draft = harness.run("system", "task").await.unwrap();
    let result = ground_answer(
        draft,
        &recall("The code is Q7."),
        2,
        0.8,
        |_, _, _| panic!("stopped work is incomplete"),
        |_, _| async { panic!("stopped work requires resume") },
    )
    .await;
    assert_caveated(&result);
    assert_eq!(
        result.outcome.checkpoint.unwrap().messages,
        result.outcome.messages
    );
}

#[test]
fn legacy_message_decoding_preserves_unknown_status() {
    let message: Message = serde_json::from_value(
        json!({"role":"Tool", "content":"The code is Q7.", "tool_call_id":"old"}),
    )
    .unwrap();
    assert_eq!(message.tool_succeeded, None);
}

struct TinyEncoder(spine_heart::ModelManifest);
impl TinyEncoder {
    fn new() -> Self {
        Self(spine_heart::ModelManifest {
            schema: 1,
            model_name: "grounding-fixture".into(),
            artifact_hash: [51; 32],
            tokenizer_hash: [52; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        })
    }
}
impl spine_heart::SemanticEncoder for TinyEncoder {
    fn manifest(&self) -> &spine_heart::ModelManifest {
        &self.0
    }
    fn encode(&self, _: &str) -> spine_heart::Result<spine_heart::Embedding> {
        spine_heart::Embedding::normalized(vec![1.0, 2.0, 3.0], 3)
    }
}
fn journal_fixture() -> (
    tempfile::TempDir,
    Arc<spine_heart::SpineHeart>,
    TinyEncoder,
    spine_heart::AgentId,
    spine_heart::ThreadId,
) {
    use spine_heart::SemanticEncoder;
    let directory = tempfile::tempdir().unwrap();
    let heart = Arc::new(
        spine_heart::SpineHeart::create(
            spine_heart::HeartConfig::new(directory.path().join("journal.spine")),
            "test-passphrase",
        )
        .unwrap()
        .heart,
    );
    let encoder = TinyEncoder::new();
    heart
        .initialize_cognition(
            spine_heart::CognitiveConfig::new(1, encoder.manifest().clone(), 2).unwrap(),
        )
        .unwrap();
    (
        directory,
        heart,
        encoder,
        spine_heart::AgentId::new("test").unwrap(),
        spine_heart::ThreadId::new("grounding").unwrap(),
    )
}

#[tokio::test]
async fn finalized_caveat_survives_journal_recovery_without_raw_draft_or_duplicates() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let harness = Harness::new(
        provider(vec![answer("The code is WRONG.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap()
    .with_journal(Arc::new(
        crate::transcript::HeartJournal::new(heart.clone(), agent.clone(), thread.clone())
            .with_grounding(true),
    ));
    let draft = harness.run("system", "task").await.unwrap();
    assert!(
        heart.events_canonical().unwrap().is_empty(),
        "draft is not evidence before verification"
    );
    let result = ground_answer(
        draft,
        "[]",
        2,
        0.8,
        |_, _, _| Err(HeartError::Model("offline".into())),
        |_, _| async { panic!("no repair") },
    )
    .await;
    crate::transcript::finalize_grounding(&heart, &agent, &thread, &result.outcome, 2).unwrap();
    crate::transcript::recover_thread(&heart, &encoder, Some(&thread)).unwrap();
    let events = heart.events_canonical().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].body.interaction.content,
        spine_heart::Content::Inline(result.outcome.response)
    );
    crate::transcript::recover_thread(&heart, &encoder, Some(&thread)).unwrap();
    assert_eq!(heart.events_canonical().unwrap().len(), 1);
    assert!(heart.local_states("harness/").unwrap().is_empty());
}

#[tokio::test]
async fn crash_before_verification_recovers_explicitly_unverified_output() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let harness = Harness::new(
        provider(vec![answer("The code is WRONG.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap()
    .with_journal(Arc::new(
        crate::transcript::HeartJournal::new(heart.clone(), agent, thread.clone())
            .with_grounding(true),
    ));
    harness.run("system", "task").await.unwrap();
    // No finalization: simulate termination between the model and the gate.
    crate::transcript::recover_thread(&heart, &encoder, Some(&thread)).unwrap();
    let events = heart.events_canonical().unwrap();
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0].body.interaction.content, spine_heart::Content::Inline(text) if text.contains("Grounding was not finalized"))
    );
}

#[tokio::test]
async fn failed_repair_keeps_completed_tools_and_never_recovers_an_uncaveated_draft() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let provider = provider(vec![
        answer("The code is WRONG."),
        Ok(ModelTurn {
            content: "The code is STILLWRONG.".into(),
            tool_calls: vec![ToolCall {
                id: "late".into(),
                name: "search_memory".into(),
                arguments: json!({}),
            }],
            ..Default::default()
        }),
        Err(spine_runtime::RuntimeError::Provider(
            "lost repair connection".into(),
        )),
    ]);
    let mut registry = ToolRegistry::default();
    registry.register(RecallTool).unwrap();
    let harness = Harness::new(provider, registry, config())
        .unwrap()
        .with_journal(Arc::new(
            crate::transcript::HeartJournal::new(heart.clone(), agent.clone(), thread.clone())
                .with_grounding(true),
        ));
    let draft = harness.run("system", "task").await.unwrap();
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let mut result = ground_answer(
        draft,
        &recall("The code is Q7."),
        2,
        0.8,
        |text, evidence, threshold| check(&verifier, text, evidence, threshold),
        |draft, feedback| {
            let (heart, agent, thread, harness) = (&heart, &agent, &thread, &harness);
            async move {
                crate::transcript::finalize_grounding(heart, agent, thread, &draft, 2)?;
                harness.repair(&draft, feedback).await.map(Some)
            }
        },
    )
    .await;
    assert_caveated(&result);
    assert!(
        crate::transcript::reconcile_grounding_repair(
            &heart,
            &encoder,
            &agent,
            &thread,
            &mut result.outcome,
            2
        )
        .unwrap()
    );
    assert!(result.outcome.checkpoint.is_some());
    assert!(result.outcome.response.contains("8080"));
    let events = heart.events_canonical().unwrap();
    assert!(events.iter().any(|event| {
        event
            .body
            .interaction
            .tool
            .as_ref()
            .is_some_and(|tool| tool.succeeded == Some(true))
    }));
    for event in &events {
        if let spine_heart::Content::Inline(text) = &event.body.interaction.content
            && event.body.interaction.role == spine_heart::ParticipantRole::Assistant
            && event.body.interaction.kind == spine_heart::EventKind::Message
            && text.contains("WRONG")
        {
            assert!(
                text.contains(TERMINAL_CAVEAT) || text.contains("Grounding was not finalized"),
                "{text}"
            );
        }
    }
    let count = events.len();
    crate::transcript::recover_thread(&heart, &encoder, Some(&thread)).unwrap();
    assert_eq!(heart.events_canonical().unwrap().len(), count);
}

struct FailingTool;
#[async_trait]
impl Tool for FailingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read".into(),
            description: "test".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: json!({}),
        }
    }
    async fn execute(&self, _: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        Ok(ToolResult::failure("The code is Q7."))
    }
}

#[tokio::test]
async fn actual_failed_tool_result_never_becomes_evidence_and_success_status_persists() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let provider = provider(vec![
        Ok(ModelTurn {
            tool_calls: vec![ToolCall {
                id: "failed".into(),
                name: "read".into(),
                arguments: json!({}),
            }],
            ..Default::default()
        }),
        answer("The code is Q7."),
    ]);
    let mut registry = ToolRegistry::default();
    registry.register(FailingTool).unwrap();
    let harness = Harness::new(provider, registry, config()).unwrap();
    let draft = harness.run("system", "task").await.unwrap();
    assert!(evidence_from_recall_and_messages("[]", &draft.messages[2..]).is_empty());
    assert_eq!(
        draft
            .messages
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .unwrap()
            .tool_succeeded,
        Some(false)
    );
    crate::transcript::persist(&heart, &encoder, &agent, &thread, &draft.messages[2..]).unwrap();
    assert_eq!(
        heart
            .events_canonical()
            .unwrap()
            .iter()
            .find(|e| e.body.interaction.kind == spine_heart::EventKind::ToolResult)
            .unwrap()
            .body
            .interaction
            .tool
            .as_ref()
            .unwrap()
            .succeeded,
        Some(false)
    );
}

struct InterruptProvider(AtomicUsize);
#[async_trait]
impl ModelProvider for InterruptProvider {
    async fn complete(&self, _: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            answer("The code is WRONG.")
        } else {
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn operator_interrupt_cancels_actual_repair_and_preserves_visible_caveat() {
    let harness = Harness::new(
        Arc::new(InterruptProvider(AtomicUsize::new(0))),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    let draft = harness.run("system", "task").await.unwrap();
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    sender.send(Ok("/interrupt".into())).unwrap();
    let result = ground_answer(
        draft,
        &recall("The code is Q7."),
        2,
        0.8,
        |text, evidence, threshold| check(&verifier, text, evidence, threshold),
        |draft, feedback| async move {
            let repair = Box::pin(harness.repair(&draft, feedback))
                as std::pin::Pin<Box<dyn Future<Output = _>>>;
            crate::run_with_operator_controls(
                &harness,
                repair,
                &mut receiver,
                &crate::TerminalStatus::new(),
            )
            .await
            .outcome
        },
    )
    .await;
    assert_eq!(result.repair, RepairState::Interrupted);
    assert_caveated(&result);
}

#[tokio::test]
async fn supported_and_nonfactual_answers_complete_without_repair() {
    for text in ["The code is Q7.", "Thank you!"] {
        let harness = Harness::new(
            provider(vec![answer(text)]),
            ToolRegistry::default(),
            config(),
        )
        .unwrap();
        let draft = harness.run("system", "task").await.unwrap();
        let verifier = GroundingVerifier::new(NeutralNli).unwrap();
        let result = ground_answer(
            draft,
            &recall("The code is Q7."),
            2,
            0.8,
            |text, evidence, threshold| check(&verifier, text, evidence, threshold),
            |_, _| async { panic!("supported/nonfactual output needs no repair") },
        )
        .await;
        assert!(!result.caveated);
        assert_eq!(result.outcome.response, text);
        assert_eq!(result.repair, RepairState::NotNeeded);
        if text == "Thank you!" {
            assert_eq!(result.final_decision.unwrap().risk_target(), None);
        }
    }
}

#[test]
fn truncated_recall_remains_visible_to_verifier_as_unassessed_evidence() {
    let messages = vec![
        Message::assistant(
            "",
            None,
            vec![ToolCall {
                id: "cut".into(),
                name: "search_memory".into(),
                arguments: json!({}),
            }],
        ),
        Message::tool_result(
            "cut",
            "[{\"role\":\"User\",\"text\":\"The code is\n[truncated]",
            true,
        ),
    ];
    let evidence = evidence_from_recall_and_messages(&recall("The code is Q7."), &messages);
    assert_eq!(evidence.len(), 2);
    let report = GroundingVerifier::new(NeutralNli)
        .unwrap()
        .verify("The code is Q7.", &evidence, 0.8)
        .unwrap();
    assert!(report.truncated);
    assert!(report.needs_repair());
}

#[test]
fn ambiguous_or_future_tool_identity_and_model_authored_tasks_are_not_evidence() {
    let call = |name: &str| {
        Message::assistant(
            "",
            None,
            vec![ToolCall {
                id: "same".into(),
                name: name.into(),
                arguments: json!({}),
            }],
        )
    };
    let tool = Message::tool_result("same", "The code is Q7.", true);
    assert!(evidence_from_recall_and_messages("[]", &[tool.clone(), call("read")]).is_empty());
    assert!(
        evidence_from_recall_and_messages("[]", &[call("search_memory"), tool, call("read")])
            .is_empty()
    );
    let recalled=json!([{"role":"User","text":"The code is Q7.","provenance":{"provider":"spine-subagent-task"}}]).to_string();
    assert!(evidence_from_recall_and_messages(&recalled, &[]).is_empty());
}

#[tokio::test]
async fn quoted_caveat_does_not_replace_a_host_terminal_notice_and_idempotence_syncs_transcript() {
    let text = format!("The code is WRONG.\n\n```text\n{TERMINAL_CAVEAT}\n```");
    let harness = Harness::new(
        provider(vec![answer(&text)]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    let mut outcome = harness.run("system", "task").await.unwrap();
    assert!(append_terminal_caveat(&mut outcome));
    assert!(outcome.response.ends_with(TERMINAL_CAVEAT));
    assert_eq!(outcome.response.matches(TERMINAL_CAVEAT).count(), 2);
    outcome.messages.last_mut().unwrap().content = text;
    assert!(!append_terminal_caveat(&mut outcome));
    assert_eq!(outcome.messages.last().unwrap().content, outcome.response);
}

#[tokio::test]
async fn current_user_facts_support_answer_without_importing_prior_user_history() {
    let harness = Harness::new(
        provider(vec![answer("The label is Ω7. The port is 8080.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    let history = vec![
        Message::new(MessageRole::User, "The label is WRONG."),
        Message::new(MessageRole::Assistant, "The label is WRONG."),
    ];
    let draft = harness
        .run_with_history(
            "system",
            &history,
            "The label is Ω7. The port is 8080. Please repeat those details.",
        )
        .await
        .unwrap();
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let result = ground_answer(
        draft,
        "[]",
        history.len() + 1,
        0.8,
        |text, evidence, threshold| {
            assert_eq!(evidence.len(), 1);
            assert!(evidence[0].id.starts_with("user:"));
            assert!(!evidence[0].text.contains("WRONG"));
            check(&verifier, text, evidence, threshold)
        },
        |_, _| async { panic!("user supplied both facts") },
    )
    .await;
    assert!(!result.caveated, "{:?}", result.final_decision);
    assert_eq!(result.repair, RepairState::NotNeeded);
}

#[test]
fn operator_guidance_is_evidence_but_host_repair_and_plan_quotes_cannot_self_confirm() {
    let messages = vec![
        Message::new(
            MessageRole::User,
            "[HOST GROUNDING REPAIR] Rejected draft: The label is WRONG.",
        ),
        Message::new(
            MessageRole::User,
            "[HOST PLAN — CURRENT STEP]\nThe label is WRONG.",
        ),
        Message::new(
            MessageRole::User,
            "[RESUME FROM SAFE CHECKPOINT] The label is WRONG.",
        ),
        Message::new(
            MessageRole::User,
            "[OPERATOR INJECTION — AFTER TOOL BOUNDARY]\nThe label is Ω7.\nThis supplements the active task. Briefly answer or acknowledge it, then continue unless it requests a stop.",
        ),
    ];
    let evidence = evidence_from_recall_and_messages("[]", &messages);
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].text, "The label is Ω7.");
}

#[tokio::test]
async fn delegated_output_and_nested_memory_cannot_gain_trust_through_canonical_recall() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let mut messages = Vec::new();
    for (id, name, text) in [
        (
            "delegated",
            "check_results",
            "The code is WRONG.".to_owned(),
        ),
        ("read", "read", "The port is 8080.".to_owned()),
        (
            "nested",
            "search_memory",
            json!([{"event_id":"model-memory","role":"Assistant","text":"The code is WRONG."}])
                .to_string(),
        ),
    ] {
        messages.push(Message::assistant(
            "",
            None,
            vec![ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: json!({}),
            }],
        ));
        messages.push(Message::tool_result(id, text, true));
    }
    crate::transcript::persist(&heart, &encoder, &agent, &thread, &messages).unwrap();
    let events = heart.events_canonical().unwrap();
    let delegated = events
        .iter()
        .find(|event| {
            event.body.interaction.kind == spine_heart::EventKind::ToolResult
                && event
                    .body
                    .interaction
                    .tool
                    .as_ref()
                    .is_some_and(|tool| tool.tool_name == "check_results")
        })
        .unwrap();
    assert_eq!(
        delegated.body.interaction.provenance.metadata["contains_model_output"],
        "true"
    );
    let mut registry = ToolRegistry::default();
    crate::cognition_tools::register_cognition_tools(
        &mut registry,
        heart.clone(),
        Arc::new(TinyEncoder::new()),
        false,
    )
    .unwrap();
    let recalled = registry
        .get("search_memory")
        .unwrap()
        .execute(
            &ToolCall {
                id: "recall".into(),
                name: "search_memory".into(),
                arguments: json!({"query":"code port","top_k":16}),
            },
            &ToolContext {
                agent_id: Some(agent),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(recalled.success);
    assert!(
        recalled.output.contains("check_results"),
        "recall retains tool identity: {}",
        recalled.output
    );
    let evidence = evidence_from_recall_and_messages(&recalled.output, &[]);
    assert!(
        evidence
            .iter()
            .any(|entry| entry.text == "The port is 8080.")
    );
    assert!(
        !evidence.iter().any(|entry| entry.text.contains("WRONG")),
        "{evidence:?}"
    );
    // Legacy canonical records may lack the new metadata but retain tool names.
    let legacy=json!([{"role":"Tool","tool_name":"delegate","tool_succeeded":true,"text":"The code is WRONG."}]).to_string();
    assert!(evidence_from_recall_and_messages(&legacy, &[]).is_empty());
}

struct RepairActionProvider {
    turns: AtomicUsize,
    interrupt: bool,
}
#[async_trait]
impl ModelProvider for RepairActionProvider {
    async fn complete(&self, _: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        match self.turns.fetch_add(1, Ordering::SeqCst) {
            0 => answer("The code is WRONG."),
            1 => Ok(ModelTurn {
                tool_calls: vec![ToolCall {
                    id: "write-once".into(),
                    name: "write_once".into(),
                    arguments: json!({}),
                }],
                ..Default::default()
            }),
            _ if self.interrupt => std::future::pending().await,
            _ => Err(spine_runtime::RuntimeError::Provider(
                "connection lost after action".into(),
            )),
        }
    }
}
struct WriteOnce {
    executions: Arc<AtomicUsize>,
    done: Arc<tokio::sync::Notify>,
}
#[async_trait]
impl Tool for WriteOnce {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_once".into(),
            description: "fixture write".into(),
            category: ToolCategory::Action,
            risk: ToolRisk::Mutating,
            parameters: json!({}),
        }
    }
    async fn execute(&self, _: &ToolCall, _: &ToolContext) -> spine_runtime::Result<ToolResult> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        self.done.notify_one();
        Ok(ToolResult::success("Created ledger entry receipt-17."))
    }
}

#[tokio::test]
async fn production_reconciliation_exposes_completed_repair_actions_before_next_turn_without_replay()
 {
    for interrupt in [false, true] {
        let (_directory, heart, encoder, agent, thread) = journal_fixture();
        let executions = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(tokio::sync::Notify::new());
        let mut registry = ToolRegistry::default();
        registry
            .register(WriteOnce {
                executions: executions.clone(),
                done: done.clone(),
            })
            .unwrap();
        let harness = Harness::new(
            Arc::new(RepairActionProvider {
                turns: AtomicUsize::new(0),
                interrupt,
            }),
            registry,
            config(),
        )
        .unwrap()
        .with_journal(Arc::new(
            crate::transcript::HeartJournal::new(heart.clone(), agent.clone(), thread.clone())
                .with_grounding(true),
        ));
        let draft = harness
            .run("system", "create one ledger entry and give the code")
            .await
            .unwrap();
        let verifier = GroundingVerifier::new(NeutralNli).unwrap();
        let mut grounded = ground_answer(
            draft,
            &recall("The code is Q7."),
            2,
            0.8,
            |text, evidence, threshold| check(&verifier, text, evidence, threshold),
            |draft, feedback| {
                let (heart, agent, thread, harness) = (&heart, &agent, &thread, &harness);
                async move {
                    crate::transcript::finalize_grounding(heart, agent, thread, &draft, 2)?;
                    if interrupt {
                        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
                        tokio::spawn(async move {
                            done.notified().await;
                            sender.send(Ok("/interrupt".into())).unwrap();
                        });
                        let repair = Box::pin(harness.repair(&draft, feedback))
                            as std::pin::Pin<Box<dyn Future<Output = _>>>;
                        crate::run_with_operator_controls(
                            harness,
                            repair,
                            &mut receiver,
                            &crate::TerminalStatus::new(),
                        )
                        .await
                        .outcome
                    } else {
                        harness.repair(&draft, feedback).await.map(Some)
                    }
                }
            },
        )
        .await;
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert!(
            crate::transcript::reconcile_grounding_repair(
                &heart,
                &encoder,
                &agent,
                &thread,
                &mut grounded.outcome,
                2
            )
            .unwrap()
        );
        assert_eq!(grounded.outcome.completed_action_calls, 1);
        assert!(grounded.outcome.response.contains("receipt-17"));
        assert!(grounded.outcome.response.contains("No action was replayed"));
        assert!(grounded.outcome.response.contains("unverified"));
        assert!(grounded.outcome.stopped_gracefully);
        assert_eq!(
            grounded.outcome.checkpoint.as_ref().unwrap().messages,
            grounded.outcome.messages
        );
        let mut history = Vec::new();
        crate::finish_visible_turn(&grounded.outcome, &mut history, 4, 20000);
        assert!(history.last().unwrap().content.contains("receipt-17"));
        crate::context_state::remember(&heart, &agent, &thread, &grounded.outcome).unwrap();
        let count = heart.events_canonical().unwrap().len();
        assert!(
            !crate::transcript::reconcile_grounding_repair(
                &heart,
                &encoder,
                &agent,
                &thread,
                &mut grounded.outcome,
                2
            )
            .unwrap()
        );
        assert_eq!(heart.events_canonical().unwrap().len(), count);
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        let mut checkpoint = None;
        let mut breaker = crate::resilience::CircuitBreaker::default();
        crate::checkpoint_from_outcome(
            &heart,
            &encoder,
            &agent,
            &thread,
            &grounded.outcome,
            &mut checkpoint,
            &mut breaker,
        )
        .unwrap();
        assert_eq!(
            heart.events_canonical().unwrap().len(),
            count,
            "reuse recovered checkpoint identity"
        );
        assert!(checkpoint.is_some());
    }
}

#[tokio::test]
async fn current_canonical_triangle_sources_support_answer_but_retained_working_text_does_not() {
    use spine_heart::SemanticEncoder;
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let mut leaves = Vec::new();
    for (role, text) in [
        (spine_heart::ParticipantRole::User, "The code is Q7."),
        (
            spine_heart::ParticipantRole::Assistant,
            "The code is WRONG.",
        ),
    ] {
        let (commit, memory) = heart
            .commit_embedded(
                spine_heart::InteractionInput {
                    agent_id: agent.clone(),
                    thread_id: thread.clone(),
                    role,
                    kind: spine_heart::EventKind::Message,
                    content: spine_heart::Content::Inline(text.into()),
                    causal_parents: Vec::new(),
                    provenance: Default::default(),
                    tool: None,
                    attachments: Vec::new(),
                    outcome: None,
                },
                encoder.encode(text).unwrap(),
            )
            .unwrap();
        leaves.push(spine_heart::ContextLeaf {
            node_id: memory.node_id,
            chronology: commit.event.body.device_sequence,
        });
    }
    heart.compact_context(leaves, 6).unwrap();
    let harness = Harness::new(
        provider(vec![answer("The code is Q7.")]),
        ToolRegistry::default(),
        config(),
    )
    .unwrap();
    let draft = harness.run("system", "Give the code").await.unwrap();
    let mut retained = draft.clone();
    retained
        .messages
        .push(Message::tool("old", "The code is WRONG."));
    crate::context_state::remember(&heart, &agent, &thread, &retained).unwrap();
    let triangle = crate::context_state::rehydrate(
        &heart,
        &encoder,
        &thread,
        &encoder.encode("code").unwrap(),
    )
    .unwrap();
    assert!(triangle.contains("working_evidence"));
    let recalled = recall_with_rehydrated("[]", &triangle);
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let grounded = ground_answer(
        draft,
        &recalled,
        2,
        0.8,
        |text, evidence, threshold| {
            assert!(
                evidence.iter().any(|entry| entry.text == "The code is Q7."),
                "{evidence:?}"
            );
            assert!(!evidence.iter().any(|entry| entry.text.contains("WRONG")));
            check(&verifier, text, evidence, threshold)
        },
        |_, _| async { panic!("canonical current rehydration supplies the fact") },
    )
    .await;
    assert!(!grounded.caveated);
    assert_eq!(grounded.repair, RepairState::NotNeeded);
}

struct BeforeRepairOutput {
    calls: AtomicUsize,
    requested: Arc<tokio::sync::Notify>,
}
#[async_trait]
impl ModelProvider for BeforeRepairOutput {
    async fn complete(&self, _: CompletionRequest) -> spine_runtime::Result<ModelTurn> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            answer("The code is WRONG.")
        } else {
            self.requested.notify_one();
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn repair_cancelled_before_any_output_still_recovers_an_unverified_terminal_summary() {
    let (_directory, heart, encoder, agent, thread) = journal_fixture();
    let requested = Arc::new(tokio::sync::Notify::new());
    let harness = Harness::new(
        Arc::new(BeforeRepairOutput {
            calls: AtomicUsize::new(0),
            requested: requested.clone(),
        }),
        ToolRegistry::default(),
        config(),
    )
    .unwrap()
    .with_journal(Arc::new(
        crate::transcript::HeartJournal::new(heart.clone(), agent.clone(), thread.clone())
            .with_grounding(true),
    ));
    let draft = harness.run("system", "task").await.unwrap();
    let verifier = GroundingVerifier::new(NeutralNli).unwrap();
    let mut grounded = ground_answer(
        draft,
        &recall("The code is Q7."),
        2,
        0.8,
        |text, evidence, threshold| check(&verifier, text, evidence, threshold),
        |draft, feedback| async move {
            let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
            tokio::spawn(async move {
                requested.notified().await;
                sender.send(Ok("/interrupt".into())).unwrap();
            });
            let repair = Box::pin(harness.repair(&draft, feedback))
                as std::pin::Pin<Box<dyn Future<Output = _>>>;
            crate::run_with_operator_controls(
                &harness,
                repair,
                &mut receiver,
                &crate::TerminalStatus::new(),
            )
            .await
            .outcome
        },
    )
    .await;
    assert!(
        crate::transcript::reconcile_grounding_repair(
            &heart,
            &encoder,
            &agent,
            &thread,
            &mut grounded.outcome,
            2
        )
        .unwrap()
    );
    assert!(
        grounded
            .outcome
            .response
            .ends_with("factual details remain unverified.]")
    );
    assert_eq!(grounded.outcome.completed_tool_calls, 0);
}

#[tokio::test]
async fn duplicate_tool_ids_cannot_launder_model_output_through_persistence_and_recall() {
    for same_batch in [true, false] {
        let (_directory, heart, encoder, agent, thread) = journal_fixture();
        let delegated = ToolCall {
            id: "same".into(),
            name: "check_results".into(),
            arguments: json!({}),
        };
        let read = ToolCall {
            id: "same".into(),
            name: "file_read".into(),
            arguments: json!({}),
        };
        let mut messages = if same_batch {
            vec![
                Message::assistant("", None, vec![delegated, read]),
                Message::tool_result("same", "The code is WRONG.", true),
            ]
        } else {
            vec![
                Message::assistant("", None, vec![delegated]),
                Message::tool_result("same", "The code is WRONG.", true),
                Message::assistant("", None, vec![read]),
            ]
        };
        messages.push(Message::tool_result("same", "The code is Q7.", true));
        messages.push(Message::assistant(
            "",
            None,
            vec![ToolCall {
                id: "unique".into(),
                name: "file_read".into(),
                arguments: json!({}),
            }],
        ));
        messages.push(Message::tool_result("unique", "The port is 8080.", true));
        assert!(
            !evidence_from_recall_and_messages("[]", &messages)
                .iter()
                .any(|entry| entry.text.contains("code"))
        );
        crate::transcript::persist(&heart, &encoder, &agent, &thread, &messages).unwrap();
        let events = heart.events_canonical().unwrap();
        let ambiguous = events
            .iter()
            .filter_map(|event| {
                (event.body.interaction.kind == spine_heart::EventKind::ToolResult)
                    .then_some(event.body.interaction.tool.as_ref())
                    .flatten()
            })
            .filter(|tool| tool.operation_id == "same")
            .collect::<Vec<_>>();
        assert_eq!(ambiguous.len(), 2);
        assert!(ambiguous.iter().all(|tool| tool.tool_name == "unknown"));
        let mut registry = ToolRegistry::default();
        crate::cognition_tools::register_cognition_tools(
            &mut registry,
            heart.clone(),
            Arc::new(TinyEncoder::new()),
            false,
        )
        .unwrap();
        let recalled = registry
            .get("search_memory")
            .unwrap()
            .execute(
                &ToolCall {
                    id: "recall".into(),
                    name: "search_memory".into(),
                    arguments: json!({"query":"code port","top_k":16}),
                },
                &ToolContext {
                    agent_id: Some(agent),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let evidence = evidence_from_recall_and_messages(&recalled.output, &[]);
        assert!(
            evidence
                .iter()
                .any(|entry| entry.text == "The port is 8080.")
        );
        assert!(
            !evidence.iter().any(|entry| entry.text.contains("code")),
            "{evidence:?}"
        );
    }
}
