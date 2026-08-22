use std::{
    collections::VecDeque,
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use spine_heart::{AgentId, ThymosConfig};
use spine_runtime::{
    CompletionRequest, Harness, HarnessConfig, MessageRole, ModelProvider, ModelTurn, Result,
    SubagentHarnessFactory, Tool, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult,
    ToolRisk, ToolSpec,
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
