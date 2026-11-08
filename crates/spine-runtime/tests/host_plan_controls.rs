use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use spine_runtime::{
    CompletionRequest, ControlPlane, Harness, HarnessCheckpoint, HarnessConfig, HarnessJournal,
    HarnessProgress, MessageRole, ModelProvider, ModelTurn, Result, Tool, ToolCall, ToolCategory,
    ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};
use tokio::sync::Notify;

const PLAN: &str = "```plan\n1. Inspect the first item\n2. Inspect the second item\n```";

fn answer(content: &str) -> ModelTurn {
    ModelTurn {
        content: content.into(),
        ..Default::default()
    }
}

fn tool_turn(content: &str, calls: &[usize]) -> ModelTurn {
    ModelTurn {
        content: content.into(),
        tool_calls: calls
            .iter()
            .map(|id| ToolCall {
                id: format!("call-{id}"),
                name: "probe".into(),
                arguments: serde_json::json!({"fail": id % 2 == 1}),
            })
            .collect(),
        ..Default::default()
    }
}

struct Script {
    turns: Mutex<VecDeque<ModelTurn>>,
    requests: Mutex<Vec<CompletionRequest>>,
}

impl Script {
    fn new(turns: Vec<ModelTurn>) -> Arc<Self> {
        Arc::new(Self {
            turns: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl ModelProvider for Script {
    async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
        self.requests.lock().unwrap().push(request);
        Ok(self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected model continuation"))
    }
}

struct Probe {
    result: Option<ToolResult>,
    completed: Arc<AtomicUsize>,
    gate: Option<(Arc<Notify>, Arc<Notify>)>,
}

#[async_trait]
impl Tool for Probe {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "probe".into(),
            description: "Inspect one item".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({"type":"object"}),
        }
    }

    async fn execute(&self, call: &ToolCall, _: &ToolContext) -> Result<ToolResult> {
        if let Some((started, release)) = &self.gate {
            started.notify_one();
            release.notified().await;
        }
        self.completed.fetch_add(1, Ordering::SeqCst);
        Ok(self.result.clone().unwrap_or_else(|| {
            if call.arguments["fail"] == true {
                ToolResult::failure("read failed")
            } else {
                ToolResult::success("item inspected")
            }
        }))
    }
}

fn registry(result: Option<ToolResult>, completed: &Arc<AtomicUsize>) -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    registry
        .register(Probe {
            result,
            completed: completed.clone(),
            gate: None,
        })
        .unwrap();
    registry
}

fn has_prompt(request: &CompletionRequest, text: &str) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.role == MessageRole::User && m.content.contains(text))
}

#[derive(Default)]
struct Journal(Mutex<Vec<HarnessProgress>>);
impl HarnessJournal for Journal {
    fn save(&self, progress: &HarnessProgress) -> Result<()> {
        self.0.lock().unwrap().push(progress.clone());
        Ok(())
    }
}

#[tokio::test]
async fn ordinary_instructions_return_after_one_model_call() {
    for prose in [
        "For the website, follow these instructions yourself:\n1. Open account settings.\n2. **Permissions:** select **Read and write**.",
        "- **Permissions:** choose **Read** and write\n- **Repository:** **Open** access is disabled",
        "```text\n1. Read the page\n2. Open settings\n```",
    ] {
        let provider = Script::new(vec![answer(prose)]);
        let harness = Harness::new(
            provider.clone(),
            ToolRegistry::default(),
            HarnessConfig::default(),
        )
        .unwrap();
        let outcome = harness
            .run(
                "Explain settings without taking action.",
                "How do permissions work?",
            )
            .await
            .unwrap();
        assert_eq!(outcome.response, prose);
        assert!(outcome.host_plan.is_none());
        assert_eq!(outcome.completed_tool_calls, 0);
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn failed_internal_results_do_not_advance_and_stalled_tools_are_bounded() {
    for result in [
        ToolResult::failure("inspection failed"),
        ToolResult {
            success: true,
            output: "partial output".into(),
            error: Some("inspection failed".into()),
        },
    ] {
        let completed = Arc::new(AtomicUsize::new(0));
        let provider = Script::new(vec![
            tool_turn(PLAN, &[1]),
            tool_turn("Retrying the inspection.", &[2]),
            answer("The inspection failed; both steps remain open."),
        ]);
        let harness = Harness::new(
            provider.clone(),
            registry(Some(result), &completed),
            HarnessConfig {
                max_empty_plan_continuations: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let outcome = harness.run("Inspect", "Inspect both items").await.unwrap();
        let plan = outcome.host_plan.unwrap();
        assert_eq!(plan.cursor, 0);
        assert!(plan.steps.iter().all(|step| step.evidence.is_empty()));
        assert_eq!(completed.load(Ordering::SeqCst), 2);
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(has_prompt(requests.last().unwrap(), "[HOST PLAN CEILING]"));
        assert!(!requests.last().unwrap().allow_tool_calls);
        assert!(outcome.policy.max_tool_rounds.is_none());
    }
}

#[tokio::test]
async fn successful_tools_that_keep_promising_without_step_progress_are_bounded() {
    let completed = Arc::new(AtomicUsize::new(0));
    let provider = Script::new(vec![
        tool_turn(&format!("{PLAN}\nI'll check one more item."), &[1]),
        tool_turn("I'll check one more item.", &[2]),
        answer("Stopped with the first step open."),
    ]);
    let harness = Harness::new(
        provider.clone(),
        registry(Some(ToolResult::success("observed")), &completed),
        HarnessConfig {
            max_empty_plan_continuations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let outcome = harness.run("Inspect", "Inspect both items").await.unwrap();
    assert_eq!(outcome.host_plan.unwrap().cursor, 0);
    assert_eq!(completed.load(Ordering::SeqCst), 2);
    assert!(has_prompt(
        provider.requests.lock().unwrap().last().unwrap(),
        "[HOST PLAN CEILING]"
    ));
}

#[tokio::test]
async fn successful_step_progress_resets_the_stall_budget() {
    let completed = Arc::new(AtomicUsize::new(0));
    let provider = Script::new(vec![
        tool_turn(
            "```plan\n1. Inspect one\n2. Inspect two\n3. Inspect three\n```",
            &[1],
        ),
        tool_turn("Observed item one.", &[2]),
        tool_turn("Retry item two.", &[3]),
        tool_turn("Observed item two.", &[4]),
        tool_turn("Retry item three.", &[5]),
        tool_turn("Observed item three.", &[6]),
        answer("All three items inspected."),
    ]);
    let harness = Harness::new(
        provider.clone(),
        registry(None, &completed),
        HarnessConfig {
            max_empty_plan_continuations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let outcome = harness.run("Inspect", "Inspect three items").await.unwrap();
    assert!(outcome.host_plan.unwrap().done());
    assert_eq!(completed.load(Ordering::SeqCst), 6);
    assert_eq!(provider.requests.lock().unwrap().len(), 7);
}

#[tokio::test]
async fn unplanned_tool_rounds_remain_unlimited() {
    let completed = Arc::new(AtomicUsize::new(0));
    let mut turns = (1..=12)
        .map(|id| tool_turn("Checking.", &[id]))
        .collect::<Vec<_>>();
    turns.push(answer("The requested checks finished."));
    let provider = Script::new(turns);
    let harness = Harness::new(
        provider.clone(),
        registry(Some(ToolResult::failure("read failed")), &completed),
        HarnessConfig {
            max_empty_plan_continuations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let outcome = harness.run("Inspect", "Check these items").await.unwrap();
    assert!(outcome.host_plan.is_none());
    assert!(outcome.policy.max_tool_rounds.is_none());
    assert_eq!(completed.load(Ordering::SeqCst), 12);
    assert_eq!(provider.requests.lock().unwrap().len(), 13);
}

#[tokio::test]
async fn discard_cancels_generation_and_prevents_reinstallation_or_repair() {
    struct Generating {
        started: Arc<Notify>,
        requests: Mutex<Vec<CompletionRequest>>,
    }
    #[async_trait]
    impl ModelProvider for Generating {
        async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
            let index = {
                let mut requests = self.requests.lock().unwrap();
                requests.push(request);
                requests.len()
            };
            if index == 2 {
                self.started.notify_one();
                return std::future::pending().await;
            }
            // Repeating the old fence must not install it after cancellation.
            Ok(answer(PLAN))
        }
    }
    let started = Arc::new(Notify::new());
    let provider = Arc::new(Generating {
        started: started.clone(),
        requests: Mutex::new(Vec::new()),
    });
    let journal = Arc::new(Journal::default());
    let harness = Arc::new(
        Harness::new(
            provider.clone(),
            ToolRegistry::default(),
            HarnessConfig::default(),
        )
        .unwrap()
        .with_journal(journal.clone()),
    );
    let task_harness = harness.clone();
    let task = tokio::spawn(async move { task_harness.run("Inspect", "Inspect both items").await });
    started.notified().await;
    assert!(!harness.controls().request_plan_discard("   "));
    assert!(
        harness
            .controls()
            .request_plan_discard("These were instructions for the user.")
    );
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(outcome.host_plan.is_none());
    assert_eq!(outcome.completed_tool_calls, 0);
    {
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(has_prompt(&requests[2], "[HOST PLAN DISCARDED]"));
        assert!(has_prompt(
            &requests[2],
            "These were instructions for the user."
        ));
    }
    let repaired = harness
        .repair(&outcome, "Provide a clear explanation.")
        .await
        .unwrap();
    assert!(repaired.host_plan.is_none());
    assert_eq!(provider.requests.lock().unwrap().len(), 4);
    let states = journal.0.lock().unwrap();
    assert!(states.last().unwrap().host_plan.is_none());
    assert!(states.last().unwrap().finished);
}

#[tokio::test]
async fn discard_waits_for_the_tool_boundary_and_survives_checkpoint_resume() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let completed = Arc::new(AtomicUsize::new(0));
    let provider = Script::new(vec![
        tool_turn(PLAN, &[1, 2]),
        answer("Stopped after the first item."),
        answer(PLAN),
    ]);
    let mut registry = ToolRegistry::default();
    registry
        .register(Probe {
            result: Some(ToolResult::success("inspected")),
            completed: completed.clone(),
            gate: Some((started.clone(), release.clone())),
        })
        .unwrap();
    let harness =
        Arc::new(Harness::new(provider.clone(), registry, HarnessConfig::default()).unwrap());
    let task_harness = harness.clone();
    let mut task =
        tokio::spawn(async move { task_harness.run("Inspect", "Inspect both items").await });
    started.notified().await;
    harness
        .controls()
        .request_plan_discard("Do not perform the second item.");
    harness.controls().request_graceful_stop();
    assert!(
        tokio::time::timeout(Duration::from_millis(30), &mut task)
            .await
            .is_err(),
        "an in-flight side effect must reach its boundary"
    );
    assert_eq!(completed.load(Ordering::SeqCst), 0);
    release.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(completed.load(Ordering::SeqCst), 1);
    assert!(outcome.host_plan.is_none());
    assert!(outcome.stopped_gracefully);
    assert!(outcome.messages.iter().any(|m| m.tool_call_id.as_deref() == Some("call-2") && m.tool_succeeded == Some(false)));
    let checkpoint = outcome.checkpoint.unwrap();
    checkpoint.validate().unwrap();
    let checkpoint: HarnessCheckpoint =
        serde_json::from_str(&serde_json::to_string(&checkpoint).unwrap()).unwrap();
    assert!(checkpoint.host_plan.is_none());
    let history = checkpoint.messages.clone();
    let resumed = harness.resume(checkpoint).await.unwrap();
    assert!(resumed.host_plan.is_none());
    assert_eq!(provider.requests.lock().unwrap().len(), 3);

    // Cancellation belongs to the old task, even if its transcript is supplied as history.
    let fresh = Script::new(vec![
        tool_turn(PLAN, &[3]),
        tool_turn("Observed item two.", &[4]),
        answer("Finished."),
    ]);
    let fresh_harness = Harness::new(
        fresh.clone(),
        registry_for_success(&completed),
        HarnessConfig::default(),
    )
    .unwrap();
    let fresh_outcome = fresh_harness
        .run_with_history("Inspect", &history, "A new task")
        .await
        .unwrap();
    assert!(fresh_outcome.host_plan.unwrap().done());
    assert!(
        fresh.requests.lock().unwrap()[0]
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .all(|m| !m.content.contains("[HOST PLAN DISCARDED]"))
    );
}

fn registry_for_success(completed: &Arc<AtomicUsize>) -> ToolRegistry {
    registry(Some(ToolResult::success("inspected")), completed)
}

#[tokio::test]
async fn discard_during_final_summary_is_saved_in_the_checkpoint() {
    struct SummaryProvider {
        started: Arc<Notify>,
    }
    #[async_trait]
    impl ModelProvider for SummaryProvider {
        async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
            if request.allow_tool_calls {
                return Ok(tool_turn(PLAN, &[1]));
            }
            self.started.notify_one();
            std::future::pending().await
        }
    }
    let started = Arc::new(Notify::new());
    let completed = Arc::new(AtomicUsize::new(0));
    let harness = Arc::new(
        Harness::new(
            Arc::new(SummaryProvider {
                started: started.clone(),
            }),
            registry_for_success(&completed),
            HarnessConfig::default(),
        )
        .unwrap(),
    );
    harness.controls().request_graceful_stop();
    let task_harness = harness.clone();
    let task = tokio::spawn(async move { task_harness.run("Inspect", "Inspect both items").await });
    started.notified().await;
    harness
        .controls()
        .request_plan_discard("The remaining plan is no longer needed.");
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(outcome.host_plan.is_none());
    let checkpoint = outcome.checkpoint.unwrap();
    checkpoint.validate().unwrap();
    assert!(checkpoint.host_plan.is_none());
    assert!(checkpoint.messages.iter().any(|m| {
        m.content
            .contains("The remaining plan is no longer needed.")
    }));
}

#[tokio::test]
async fn discard_wins_races_with_provider_errors_stale_tool_calls_and_final_summaries() {
    struct RacingProvider {
        controls: Mutex<Option<ControlPlane>>,
        calls: AtomicUsize,
        mode: u8,
    }
    #[async_trait]
    impl ModelProvider for RacingProvider {
        async fn complete(&self, request: CompletionRequest) -> Result<ModelTurn> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                return Ok(if self.mode == 1 {
                    tool_turn(PLAN, &[1])
                } else {
                    answer(PLAN)
                });
            }
            if call == 1 {
                self.controls
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .request_plan_discard("Discard wins the provider race.");
                if self.mode == 0 {
                    return Err(spine_runtime::RuntimeError::Provider(
                        "fixture request failed".into(),
                    ));
                }
                if self.mode == 1 {
                    assert!(!request.allow_tool_calls);
                    return Ok(answer("STALE SUMMARY: all plan steps completed."));
                }
                return Ok(tool_turn("stale plan tool call", &[1]));
            }
            Ok(answer("The explanation is complete."))
        }
    }
    for mode in [0, 1, 2] {
        let provider = Arc::new(RacingProvider {
            controls: Mutex::new(None),
            calls: AtomicUsize::new(0),
            mode,
        });
        let completed = Arc::new(AtomicUsize::new(0));
        let harness = Harness::new(
            provider.clone(),
            registry_for_success(&completed),
            HarnessConfig::default(),
        )
        .unwrap();
        *provider.controls.lock().unwrap() = Some(harness.controls());
        if mode == 1 {
            harness.controls().request_graceful_stop();
        }
        let outcome = harness.run("Inspect", "Inspect both items").await.unwrap();
        assert!(outcome.host_plan.is_none());
        assert!(!outcome.response.contains("STALE SUMMARY"));
        assert!(
            outcome
                .messages
                .iter()
                .any(|m| m.content.contains("Discard wins the provider race."))
        );
        if mode == 1 {
            let checkpoint = outcome.checkpoint.unwrap();
            checkpoint.validate().unwrap();
            assert!(checkpoint.host_plan.is_none());
        } else {
            assert_eq!(completed.load(Ordering::SeqCst), 0);
        }
    }
}
