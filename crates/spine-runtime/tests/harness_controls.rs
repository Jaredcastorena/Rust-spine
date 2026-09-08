use std::{
    collections::VecDeque,
    num::{NonZeroU64, NonZeroUsize},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use spine_heart::{AgentId, ThreadId, ThymosConfig};
use spine_runtime::{
    CompletionRequest, Harness, HarnessCheckpoint, HarnessConfig, HarnessPolicy, HostPlan, Message,
    MessageRole, ModelProvider, ModelTurn, Result, SubagentHarnessFactory, Tool, ToolCall,
    ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};

struct ScriptedProvider {
    turns: Mutex<VecDeque<ModelTurn>>,
    requests: Mutex<Vec<CompletionRequest>>,
}

impl ScriptedProvider {
    fn new(turns: Vec<ModelTurn>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ModelProvider for ScriptedProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
        self.requests.lock().unwrap().push(request);
        Ok(self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted model turn"))
    }
}

struct Probe {
    executed: Arc<Mutex<Vec<String>>>,
    risk: ToolRisk,
}

#[async_trait]
impl Tool for Probe {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "probe".into(),
            description: "inspect one item".into(),
            category: ToolCategory::Internal,
            risk: self.risk,
            parameters: serde_json::json!({"type":"object"}),
        }
    }

    async fn execute(&self, call: &ToolCall, _context: &ToolContext) -> Result<ToolResult> {
        let item = call.arguments["item"].as_str().unwrap().to_owned();
        self.executed.lock().unwrap().push(item.clone());
        Ok(ToolResult::success(format!("checked {item}")))
    }
}

fn call(index: usize) -> ToolCall {
    ToolCall {
        id: format!("call-{index}"),
        name: "probe".into(),
        arguments: serde_json::json!({"item": index.to_string()}),
    }
}

fn tool_turn(calls: Vec<ToolCall>) -> ModelTurn {
    ModelTurn {
        content: "checking".into(),
        tool_calls: calls,
        ..ModelTurn::default()
    }
}

fn answer(text: &str) -> ModelTurn {
    ModelTurn {
        content: text.into(),
        ..ModelTurn::default()
    }
}

struct MetadataProbe;

#[async_trait]
impl Tool for MetadataProbe {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "metadata_probe".into(),
            description: "Inspect the host-owned tool context".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: serde_json::json!({"type":"object"}),
        }
    }

    async fn execute(&self, _: &ToolCall, context: &ToolContext) -> Result<ToolResult> {
        Ok(ToolResult::success(
            serde_json::to_string(&context.metadata).unwrap(),
        ))
    }
}

#[tokio::test]
async fn host_introspection_metadata_is_bounded_isolated_and_cannot_override_task() {
    let mut registry = ToolRegistry::default();
    registry.register(MetadataProbe).unwrap();
    let make_provider = || {
        Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![ToolCall {
                id: "metadata".into(),
                name: "metadata_probe".into(),
                arguments: serde_json::json!({}),
            }]),
            answer("Finished."),
        ]))
    };
    let mut harness =
        Harness::new(make_provider(), registry.clone(), HarnessConfig::default()).unwrap();
    harness
        .set_tool_metadata(
            [
                ("spine_trajectory".into(), "{\"surprise\":0.25}".into()),
                ("task".into(), "spoofed task".into()),
                (
                    "unrelated_private_value".into(),
                    "must not reach tools".into(),
                ),
            ]
            .into(),
        )
        .unwrap();
    assert!(
        harness
            .set_tool_metadata([("spine_trajectory".into(), "[]".into())].into())
            .is_err()
    );
    assert!(
        harness
            .set_tool_metadata([("spine_trajectory".into(), " ".repeat(8_193))].into())
            .is_err()
    );
    let outcome = harness.run("Use the probe.", "actual task").await.unwrap();
    let message = outcome
        .messages
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&message.content).unwrap();
    assert_eq!(metadata["task"], "actual task");
    assert_eq!(metadata["spine_trajectory"], "{\"surprise\":0.25}");
    assert!(metadata.get("unrelated_private_value").is_none());
    let separate = Harness::new(make_provider(), registry, HarnessConfig::default()).unwrap();
    let outcome = separate
        .run("Use the probe.", "independent task")
        .await
        .unwrap();
    let message = outcome
        .messages
        .iter()
        .find(|message| message.role == MessageRole::Tool)
        .unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&message.content).unwrap();
    assert_eq!(metadata, serde_json::json!({"task":"independent task"}));
}

fn registry(executed: Arc<Mutex<Vec<String>>>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry
        .register(Probe {
            executed,
            risk: ToolRisk::ReadOnly,
        })
        .unwrap();
    registry
}

struct ActionProbe {
    executed: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for ActionProbe {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "act".into(),
            description: "perform one external action".into(),
            category: ToolCategory::Action,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({"type":"object"}),
        }
    }

    async fn execute(&self, call: &ToolCall, _context: &ToolContext) -> Result<ToolResult> {
        let item = call.arguments["item"].as_str().unwrap().to_owned();
        self.executed.lock().unwrap().push(item.clone());
        Ok(ToolResult::success(format!("acted on {item}")))
    }
}

fn action_call(index: usize) -> ToolCall {
    ToolCall {
        id: format!("action-{index}"),
        name: "act".into(),
        arguments: serde_json::json!({"item": index.to_string()}),
    }
}

#[tokio::test]
async fn tool_rounds_are_unlimited_by_default() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let mut turns: Vec<_> = (0..5).map(|index| tool_turn(vec![call(index)])).collect();
    turns.push(answer("all five complete"));
    let provider = Arc::new(ScriptedProvider::new(turns));
    let harness = Harness::new(
        provider,
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();
    let result = harness.run("system", "inspect all").await.unwrap();
    assert_eq!(result.completed_tool_calls, 5);
    assert_eq!(result.completed_tool_rounds, 5);
    assert_eq!(result.response, "all five complete");
    assert_eq!(executed.lock().unwrap().len(), 5);
    assert_eq!(
        result
            .messages
            .iter()
            .map(|message| message.tool_calls.len())
            .sum::<usize>(),
        5
    );
}

#[tokio::test]
async fn positive_tool_ceiling_remains_configurable() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(1)]),
        tool_turn(vec![call(2)]),
        tool_turn(vec![call(3)]),
        answer("stopped at ceiling"),
    ]));
    let harness = Harness::new(
        provider,
        registry(Arc::clone(&executed)),
        HarnessConfig {
            max_tool_rounds: NonZeroU64::new(2),
            ..HarnessConfig::default()
        },
    )
    .unwrap();
    let result = harness.run("system", "inspect").await.unwrap();
    assert_eq!(&*executed.lock().unwrap(), &["1", "2"]);
    assert_eq!(result.response, "stopped at ceiling");
}

#[tokio::test]
async fn per_run_policy_can_expand_a_positive_tool_ceiling() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(1)]),
        tool_turn(vec![call(2)]),
        tool_turn(vec![call(3)]),
        tool_turn(vec![call(4)]),
        answer("stopped at expanded ceiling"),
    ]));
    let harness = Harness::new(
        provider,
        registry(Arc::clone(&executed)),
        HarnessConfig {
            max_tool_rounds: NonZeroU64::new(1),
            ..HarnessConfig::default()
        },
    )
    .unwrap();

    let result = harness
        .run_with_history_policy(
            "system",
            &[],
            "inspect",
            HarnessPolicy {
                max_tool_rounds: NonZeroU64::new(3),
                ..HarnessPolicy::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(&*executed.lock().unwrap(), &["1", "2", "3"]);
    assert_eq!(result.completed_tool_rounds, 3);
    assert_eq!(result.response, "stopped at expanded ceiling");
}

#[tokio::test]
async fn legacy_checkpoint_without_policy_inherits_configured_total_round_limit() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let initial = Harness::new(
        Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![call(1)]),
            answer("paused"),
        ])),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();
    initial.controls().request_graceful_stop();
    let saved = initial
        .run("system", "inspect")
        .await
        .unwrap()
        .checkpoint
        .unwrap();
    let mut legacy_json = serde_json::to_value(&saved).unwrap();
    legacy_json.as_object_mut().unwrap().remove("policy");
    let legacy: HarnessCheckpoint = serde_json::from_value(legacy_json).unwrap();
    let record = legacy
        .to_interaction(
            AgentId::new("main").unwrap(),
            ThreadId::new("interactive").unwrap(),
        )
        .unwrap();
    let restored = HarnessCheckpoint::from_interaction(&record).unwrap();
    assert!(
        restored.policy.is_none(),
        "persistence must not convert a missing legacy policy into explicit unlimited"
    );
    let resumed = Harness::new(
        Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![call(2)]),
            answer("at ceiling"),
        ])),
        registry(Arc::clone(&executed)),
        HarnessConfig {
            max_tool_rounds: NonZeroU64::new(1),
            ..HarnessConfig::default()
        },
    )
    .unwrap();
    let outcome = resumed.resume(restored).await.unwrap();
    assert_eq!(
        &*executed.lock().unwrap(),
        &["1"],
        "legacy resume must not execute past the configured total ceiling"
    );
    assert_eq!(outcome.completed_tool_rounds, 1);
    assert_eq!(outcome.policy.max_tool_rounds, NonZeroU64::new(1));
}

#[tokio::test]
async fn checkpoint_preserves_effective_tool_ceiling_across_new_harness_config() {
    for original_ceiling in [None, NonZeroU64::new(2)] {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![call(1)]),
            answer("paused"),
        ]));
        let harness = Harness::new(
            provider,
            registry(Arc::clone(&executed)),
            HarnessConfig {
                max_tool_rounds: original_ceiling,
                ..HarnessConfig::default()
            },
        )
        .unwrap();
        harness.controls().request_graceful_stop();
        let checkpoint = harness
            .run("system", "inspect")
            .await
            .unwrap()
            .checkpoint
            .unwrap();
        let record = checkpoint
            .to_interaction(
                AgentId::new("main").unwrap(),
                ThreadId::new("interactive").unwrap(),
            )
            .unwrap();
        let checkpoint = HarnessCheckpoint::from_interaction(&record).unwrap();
        assert_eq!(checkpoint.policy.unwrap().max_tool_rounds, original_ceiling);

        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![call(2)]),
            tool_turn(vec![call(3)]),
            answer("finished"),
        ]));
        let resumed = Harness::new(
            provider,
            registry(Arc::clone(&executed)),
            HarnessConfig {
                max_tool_rounds: NonZeroU64::new(1),
                ..HarnessConfig::default()
            },
        )
        .unwrap();
        let result = resumed.resume(checkpoint).await.unwrap();
        assert_eq!(
            result.completed_tool_calls,
            if original_ceiling.is_some() { 2 } else { 3 }
        );
    }
}

#[tokio::test]
async fn explicit_unlimited_policy_overrides_configured_ceiling() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(1)]),
        tool_turn(vec![call(2)]),
        answer("finished"),
    ]));
    let harness = Harness::new(
        provider,
        registry(Arc::clone(&executed)),
        HarnessConfig {
            max_tool_rounds: NonZeroU64::new(1),
            ..HarnessConfig::default()
        },
    )
    .unwrap();
    let result = harness
        .run_with_history_policy("system", &[], "inspect", HarnessPolicy::default())
        .await
        .unwrap();
    assert_eq!(result.completed_tool_calls, 2);
}

#[tokio::test]
async fn repair_keeps_action_budget_and_checkpoint_counters_from_the_draft() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![action_call(1)]),
        answer("draft"),
        tool_turn(vec![action_call(2)]),
        answer("paused repair"),
    ]));
    let mut registry = ToolRegistry::default();
    registry
        .register(ActionProbe {
            executed: Arc::clone(&executed),
        })
        .unwrap();
    let harness = Harness::new(provider.clone(), registry, HarnessConfig::default()).unwrap();
    let policy = HarnessPolicy {
        temperature: Some(0.45),
        max_action_calls: NonZeroUsize::new(1),
        max_tool_rounds: None,
    };
    let draft = harness
        .run_with_history_policy("system with recalled evidence", &[], "act", policy)
        .await
        .unwrap();
    harness.controls().request_graceful_stop();
    let repaired = harness.repair(&draft, "verify the draft").await.unwrap();
    assert_eq!(&*executed.lock().unwrap(), &["1"]);
    assert_eq!(repaired.completed_action_calls, 1);
    assert_eq!(repaired.completed_tool_calls, 1);
    assert_eq!(repaired.completed_tool_rounds, 2);
    assert_eq!(repaired.policy, policy);
    let checkpoint = repaired.checkpoint.as_ref().unwrap();
    assert_eq!(checkpoint.completed_action_calls, 1);
    assert_eq!(checkpoint.completed_tool_calls, 1);
    assert_eq!(checkpoint.completed_tool_rounds, 2);
    assert_eq!(checkpoint.policy, Some(policy));
    checkpoint.validate().unwrap();
    let count = {
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests[2].messages[0], draft.messages[0]);
        assert!(
            requests
                .iter()
                .all(|request| request.temperature == Some(0.45))
        );
        requests.len()
    };
    assert!(harness.repair(&repaired, "ignore the stop").await.is_err());
    assert_eq!(provider.requests.lock().unwrap().len(), count);
}

#[tokio::test]
async fn repair_retains_the_total_round_ceiling_and_explicit_unlimited_policy() {
    for ceiling in [NonZeroU64::new(1), None] {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_turn(vec![call(1)]),
            answer("draft"),
            tool_turn(vec![call(2)]),
            answer("repaired"),
        ]));
        let harness = Harness::new(
            provider,
            registry(Arc::clone(&executed)),
            HarnessConfig {
                max_tool_rounds: NonZeroU64::new(1),
                ..HarnessConfig::default()
            },
        )
        .unwrap();
        let policy = HarnessPolicy {
            max_tool_rounds: ceiling,
            ..HarnessPolicy::default()
        };
        let draft = harness
            .run_with_history_policy("system", &[], "inspect", policy)
            .await
            .unwrap();
        let repaired = harness.repair(&draft, "verify").await.unwrap();
        let expected = if ceiling.is_some() { 1 } else { 2 };
        assert_eq!(repaired.completed_tool_calls, expected);
        assert_eq!(repaired.completed_tool_rounds, expected);
        assert_eq!(executed.lock().unwrap().len(), expected as usize);
        assert_eq!(repaired.policy, policy);
        assert!(repaired.checkpoint.is_none());
    }
}

#[tokio::test]
async fn exhausted_checkpoint_action_budget_still_honors_stop_and_policy() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![action_call(1)]),
        answer("paused"),
        tool_turn(vec![action_call(2), action_call(3)]),
        answer("still paused"),
    ]));
    let mut registry = ToolRegistry::default();
    registry
        .register(ActionProbe {
            executed: Arc::clone(&executed),
        })
        .unwrap();
    let harness = Harness::new(provider.clone(), registry, HarnessConfig::default()).unwrap();
    let policy = HarnessPolicy {
        temperature: Some(0.45),
        max_action_calls: NonZeroUsize::new(1),
        max_tool_rounds: None,
    };
    harness.controls().request_graceful_stop();
    let checkpoint = harness
        .run_with_history_policy("system", &[], "act", policy)
        .await
        .unwrap()
        .checkpoint
        .unwrap();
    let checkpoint: HarnessCheckpoint =
        serde_json::from_str(&serde_json::to_string(&checkpoint).unwrap()).unwrap();
    harness.controls().request_graceful_stop();
    let resumed = harness.resume(checkpoint).await.unwrap();
    assert!(resumed.stopped_gracefully);
    assert_eq!(&*executed.lock().unwrap(), &["1"]);
    let checkpoint = resumed.checkpoint.unwrap();
    assert_eq!(checkpoint.completed_action_calls, 1);
    assert_eq!(checkpoint.policy, Some(policy));
    assert_eq!(checkpoint.completed_tool_rounds, 2);
    assert_eq!(checkpoint.completed_tool_calls, 1);
    checkpoint
        .validate()
        .expect("skipped action rounds remain restartable");
    let record = checkpoint
        .to_interaction(
            AgentId::new("main").unwrap(),
            ThreadId::new("interactive").unwrap(),
        )
        .unwrap();
    assert_eq!(
        HarnessCheckpoint::from_interaction(&record).unwrap(),
        checkpoint
    );
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|request| request.temperature == Some(0.45))
    );
}

#[tokio::test]
async fn per_run_policy_enforces_temperature_and_total_action_budget() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![action_call(1)]),
        tool_turn(vec![action_call(2)]),
        answer("safe action complete"),
    ]));
    let mut registry = ToolRegistry::default();
    registry
        .register(ActionProbe {
            executed: Arc::clone(&executed),
        })
        .unwrap();
    let harness = Harness::new(provider.clone(), registry, HarnessConfig::default()).unwrap();
    let policy = HarnessPolicy {
        temperature: Some(0.45),
        max_action_calls: NonZeroUsize::new(1),
        max_tool_rounds: None,
    };

    let result = harness
        .run_with_history_policy("system", &[], "act safely", policy)
        .await
        .unwrap();

    assert_eq!(&*executed.lock().unwrap(), &["1"]);
    assert_eq!(result.completed_tool_calls, 1);
    assert!(result.messages.iter().any(|message| {
        message.tool_call_id.as_deref() == Some("action-2")
            && message.content.contains("host action budget")
    }));
    assert!(
        provider
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|request| request.temperature == Some(0.45))
    );
}

#[tokio::test]
async fn guidance_is_injected_after_one_completed_call_and_stale_batch_is_skipped() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(1), call(2)]),
        tool_turn(vec![call(3)]),
        answer("adjusted and finished"),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();
    harness.controls().queue_guidance("skip item two");
    let result = harness.run("system", "inspect").await.unwrap();
    assert_eq!(&*executed.lock().unwrap(), &["1", "3"]);
    assert_eq!(result.response, "adjusted and finished");
    let requests = provider.requests.lock().unwrap();
    assert!(requests[1].messages.iter().any(|message| {
        message.role == MessageRole::User && message.content.contains("skip item two")
    }));
    assert!(requests[1].messages.iter().any(|message| {
        message.role == MessageRole::Tool
            && message.tool_call_id.as_deref() == Some("call-2")
            && message.content.contains("skipped")
    }));
}

#[tokio::test]
async fn a_tool_free_work_promise_is_not_accepted_as_the_final_answer() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        answer("That explains the first issue. I will inspect the failing test next."),
        tool_turn(vec![call(1)]),
        answer("The failing test is now inspected and the task is complete."),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();

    let result = harness.run("system", "inspect the failure").await.unwrap();

    assert_eq!(&*executed.lock().unwrap(), &["1"]);
    assert_eq!(result.completed_tool_calls, 1);
    let requests = provider.requests.lock().unwrap();
    assert!(requests[1].messages.iter().any(|message| {
        message.role == MessageRole::User && message.content.contains("NEED PLAN")
    }));
}

#[tokio::test]
async fn repeated_empty_model_turns_return_visible_retry_guidance() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        ModelTurn::default(),
        ModelTurn::default(),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(executed),
        HarnessConfig::default(),
    )
    .unwrap();

    let result = harness.run("system", "answer this").await.unwrap();

    assert!(result.response.contains("repeated empty responses"));
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].messages.iter().any(|message| {
        message.role == MessageRole::User && message.content.contains("HOST EMPTY TURN")
    }));
}

#[tokio::test]
async fn host_plan_advances_one_step_per_evidenced_tool_round() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let first = ModelTurn {
        content: "```plan\n1. Inspect the first item\n2. Inspect the second item\n```".into(),
        tool_calls: vec![call(1)],
        ..ModelTurn::default()
    };
    let provider = Arc::new(ScriptedProvider::new(vec![
        first,
        tool_turn(vec![call(2)]),
        answer("Both planned checks completed."),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();

    let result = harness.run("system", "inspect both items").await.unwrap();

    assert_eq!(&*executed.lock().unwrap(), &["1", "2"]);
    let plan = result.host_plan.expect("installed host plan");
    assert!(plan.done());
    assert_eq!(plan.cursor, 2);
    let requests = provider.requests.lock().unwrap();
    assert!(requests[1].messages.iter().any(|message| {
        message.role == MessageRole::User
            && message.content.contains("Do only step 2/2")
            && message.content.contains("Inspect the second item")
    }));
}

#[tokio::test]
async fn graceful_stop_waits_for_boundary_and_returns_resumable_checkpoint() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(1), call(2)]),
        answer("one complete; item two remains"),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();
    harness.controls().request_graceful_stop();
    let result = harness.run("system", "inspect").await.unwrap();
    assert_eq!(&*executed.lock().unwrap(), &["1"]);
    assert!(result.stopped_gracefully);
    let checkpoint = result.checkpoint.unwrap();
    assert_eq!(checkpoint.completed_tool_calls, 1);
    let requests = provider.requests.lock().unwrap();
    assert!(!requests[1].allow_tool_calls);
    assert!(requests[1].tools.is_empty());
}

#[tokio::test]
async fn resume_restores_and_prompts_the_open_host_plan_step() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(vec![
        tool_turn(vec![call(2)]),
        answer("The resumed plan is complete."),
    ]));
    let harness = Harness::new(
        provider.clone(),
        registry(Arc::clone(&executed)),
        HarnessConfig::default(),
    )
    .unwrap();
    let mut plan = HostPlan::new(
        "inspect",
        vec![
            "Inspect the first item".into(),
            "Inspect the second item".into(),
        ],
    )
    .unwrap();
    plan.mark_current_done("tools");
    let checkpoint = HarnessCheckpoint {
        schema: 1,
        harness_id: "old-harness".into(),
        messages: vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "inspect"),
            Message::assistant("", None, vec![call(1)]),
            Message::tool("call-1", "one complete"),
            Message::new(MessageRole::Assistant, "item two remains"),
        ],
        completed_tool_calls: 1,
        completed_tool_rounds: 1,
        completed_action_calls: 0,
        pending_task: "inspect".into(),
        host_plan: Some(plan),
        policy: Some(spine_runtime::HarnessPolicy::default()),
    };

    let result = harness.resume(checkpoint).await.unwrap();

    assert_eq!(&*executed.lock().unwrap(), &["2"]);
    assert!(result.host_plan.unwrap().done());
    let requests = provider.requests.lock().unwrap();
    assert!(requests[0].messages.iter().any(|message| {
        message.role == MessageRole::User
            && message.content.contains("Do only step 2/2")
            && message.content.contains("Inspect the second item")
    }));
}

#[test]
fn checkpoint_interactions_round_trip_only_after_strict_validation() {
    let checkpoint = HarnessCheckpoint {
        schema: 1,
        harness_id: "harness-test".into(),
        messages: vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "inspect"),
            Message::new(MessageRole::Assistant, "paused safely"),
        ],
        completed_tool_calls: 0,
        completed_tool_rounds: 0,
        pending_task: "inspect".into(),
        host_plan: None,
        completed_action_calls: 0,
        policy: Some(spine_runtime::HarnessPolicy::default()),
    };
    let mut interaction = checkpoint
        .to_interaction(
            AgentId::new("main").unwrap(),
            ThreadId::new("interactive").unwrap(),
        )
        .unwrap();

    assert_eq!(
        HarnessCheckpoint::from_interaction(&interaction).unwrap(),
        checkpoint
    );
    interaction
        .provenance
        .metadata
        .insert("harness_id".into(), "different".into());
    assert!(HarnessCheckpoint::from_interaction(&interaction).is_err());
}

#[test]
fn checkpoint_validation_rejects_unsafe_boundaries_and_impossible_counters() {
    let mut checkpoint = HarnessCheckpoint {
        schema: 1,
        harness_id: "harness-test".into(),
        messages: vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "inspect"),
            Message::new(MessageRole::Assistant, "paused safely"),
        ],
        completed_tool_calls: 0,
        completed_tool_rounds: 0,
        pending_task: "inspect".into(),
        host_plan: None,
        completed_action_calls: 0,
        policy: Some(spine_runtime::HarnessPolicy::default()),
    };
    checkpoint.messages[2] = Message::assistant("", None, vec![call(1)]);
    assert!(checkpoint.validate().is_err());

    checkpoint.messages[2] = Message::new(MessageRole::Assistant, "paused safely");
    checkpoint.completed_tool_calls = 1;
    assert!(checkpoint.validate().is_err());

    checkpoint.completed_tool_calls = 0;
    checkpoint.completed_action_calls = 1;
    assert!(checkpoint.validate().is_err());
    checkpoint.completed_action_calls = 0;
    checkpoint.policy.as_mut().unwrap().temperature = Some(f32::NAN);
    assert!(checkpoint.validate().is_err());
    checkpoint.policy.as_mut().unwrap().temperature = Some(-1.0);
    assert!(checkpoint.validate().is_err());
    checkpoint.policy.as_mut().unwrap().temperature = None;
    checkpoint.schema = 2;
    assert!(checkpoint.validate().is_err());

    checkpoint.schema = 1;
    let mut plan = HostPlan::new("inspect", vec!["one".into(), "two".into()]).unwrap();
    plan.mark_current_done("");
    checkpoint.host_plan = Some(plan);
    assert!(checkpoint.validate().is_err());
}

#[test]
fn temporary_subagents_get_independent_curated_harnesses_and_thymos() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(ScriptedProvider::new(Vec::new()));
    let factory = SubagentHarnessFactory::new(
        provider,
        registry(executed),
        ["probe"],
        HarnessConfig::default(),
        ThymosConfig::new(3, 2).unwrap(),
    );
    let first = factory
        .create(AgentId::new("temporary-one").unwrap())
        .unwrap();
    let second = factory
        .create(AgentId::new("temporary-two").unwrap())
        .unwrap();
    assert_ne!(first.harness.id(), second.harness.id());
    assert_eq!(first.harness.registry().len(), 1);
    assert_eq!(second.harness.registry().len(), 1);
    assert_ne!(first.thymos.tensor(), second.thymos.tensor());
    assert!(!first.persistent);
    assert!(!second.persistent);
    first.harness.controls().queue_guidance("first only");
    // Control planes are independently owned; the second remains usable and
    // receives no mutation from the first's queue.
    second.harness.controls().request_graceful_stop();
}

#[test]
fn rejected_duplicate_registration_preserves_the_original_tool() {
    let executed = Arc::new(Mutex::new(Vec::new()));
    let mut registry = registry(Arc::clone(&executed));
    assert!(
        registry
            .register(Probe {
                executed,
                risk: ToolRisk::Destructive,
            })
            .is_err()
    );
    assert_eq!(
        registry.get("probe").unwrap().spec().risk,
        ToolRisk::ReadOnly
    );
}
