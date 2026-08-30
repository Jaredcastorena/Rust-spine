use std::{
    collections::VecDeque,
    num::NonZeroU64,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::{
    CompletionRequest, HostPlan, Message, MessageRole, ModelProvider, Result, RuntimeError,
    TokenUsage, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk,
    parse_plan_steps, promised_more_work,
};

const PLAN_GUIDANCE_MARKER: &str = "[HOST PLAN CONTRACT]";
const MAX_EMPTY_MODEL_RETRIES: u64 = 1;

#[derive(Clone, Debug)]
pub struct HarnessConfig {
    pub max_tool_rounds: Option<NonZeroU64>,
    pub max_tool_result_chars: usize,
    pub allow_destructive_tools: bool,
    pub enforce_host_plans: bool,
    pub max_empty_plan_continuations: u64,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: None,
            max_tool_result_chars: 16_384,
            allow_destructive_tools: false,
            enforce_host_plans: true,
            max_empty_plan_continuations: 8,
        }
    }
}

#[derive(Default)]
struct PendingControls {
    guidance: VecDeque<String>,
}

#[derive(Clone, Default)]
pub struct ControlPlane {
    pending: Arc<Mutex<PendingControls>>,
    stop: Arc<AtomicBool>,
}

impl ControlPlane {
    pub fn queue_guidance(&self, message: impl Into<String>) -> bool {
        let message = message.into();
        let message = message.trim();
        if message.is_empty() {
            return false;
        }
        self.pending
            .lock()
            .expect("operator control lock poisoned")
            .guidance
            .push_back(message.to_owned());
        true
    }

    pub fn request_graceful_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn drain(&self) -> OperatorControls {
        let guidance = self
            .pending
            .lock()
            .expect("operator control lock poisoned")
            .guidance
            .drain(..)
            .collect();
        let stop = self.stop.swap(false, Ordering::AcqRel);
        OperatorControls { guidance, stop }
    }
}

struct OperatorControls {
    guidance: Vec<String>,
    stop: bool,
}

impl OperatorControls {
    fn is_empty(&self) -> bool {
        self.guidance.is_empty() && !self.stop
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarnessCheckpoint {
    pub schema: u32,
    pub harness_id: String,
    pub messages: Vec<Message>,
    pub completed_tool_calls: u64,
    pub completed_tool_rounds: u64,
    pub pending_task: String,
    #[serde(default)]
    pub host_plan: Option<HostPlan>,
}

impl HarnessCheckpoint {
    pub fn to_interaction(
        &self,
        agent_id: spine_heart::AgentId,
        thread_id: spine_heart::ThreadId,
    ) -> Result<spine_heart::InteractionInput> {
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("record_type".into(), "harness_checkpoint".into());
        metadata.insert("harness_id".into(), self.harness_id.clone());
        Ok(spine_heart::InteractionInput {
            agent_id,
            thread_id,
            role: spine_heart::ParticipantRole::Operator,
            kind: spine_heart::EventKind::Control,
            content: spine_heart::Content::Inline(serde_json::to_string(self)?),
            causal_parents: Vec::new(),
            provenance: spine_heart::Provenance {
                metadata,
                ..spine_heart::Provenance::default()
            },
            tool: None,
            attachments: Vec::new(),
            outcome: Some("graceful_stop_checkpoint".into()),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HarnessEvent {
    ModelTurnCompleted,
    ToolStarted {
        id: String,
        name: String,
    },
    ToolCompleted {
        id: String,
        name: String,
        success: bool,
    },
    GuidanceInjected {
        messages: Vec<String>,
    },
    GracefulStopBoundary,
}

#[derive(Clone, Debug)]
pub struct RunOutcome {
    pub response: String,
    pub stopped_gracefully: bool,
    pub checkpoint: Option<HarnessCheckpoint>,
    pub completed_tool_calls: u64,
    pub completed_tool_rounds: u64,
    pub usage: TokenUsage,
    pub messages: Vec<Message>,
    pub host_plan: Option<HostPlan>,
}

pub struct Harness {
    id: String,
    provider: Arc<dyn ModelProvider>,
    registry: ToolRegistry,
    config: HarnessConfig,
    controls: ControlPlane,
    events: broadcast::Sender<HarnessEvent>,
    agent_id: Option<spine_heart::AgentId>,
}

impl Harness {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        registry: ToolRegistry,
        config: HarnessConfig,
    ) -> Result<Self> {
        if config.max_tool_result_chars == 0
            || (config.enforce_host_plans && config.max_empty_plan_continuations == 0)
        {
            return Err(RuntimeError::InvalidConfig(
                "tool result length and enabled plan continuation limit must be positive".into(),
            ));
        }
        static NEXT_HARNESS: AtomicU64 = AtomicU64::new(1);
        let id = format!("harness-{}", NEXT_HARNESS.fetch_add(1, Ordering::Relaxed));
        let (events, _) = broadcast::channel(256);
        Ok(Self {
            id,
            provider,
            registry,
            config,
            controls: ControlPlane::default(),
            events,
            agent_id: None,
        })
    }

    pub fn with_agent_id(mut self, agent_id: spine_heart::AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn controls(&self) -> ControlPlane {
        self.controls.clone()
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn subscribe(&self) -> broadcast::Receiver<HarnessEvent> {
        self.events.subscribe()
    }

    pub async fn run(
        &self,
        system: impl Into<String>,
        task: impl Into<String>,
    ) -> Result<RunOutcome> {
        let task = task.into();
        self.run_messages(
            vec![
                Message::new(MessageRole::System, system),
                Message::new(MessageRole::User, task.clone()),
            ],
            task,
            0,
            0,
            None,
        )
        .await
    }

    pub async fn run_with_history(
        &self,
        system: impl Into<String>,
        history: &[Message],
        task: impl Into<String>,
    ) -> Result<RunOutcome> {
        let task = task.into();
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(Message::new(MessageRole::System, system));
        messages.extend_from_slice(history);
        messages.push(Message::new(MessageRole::User, task.clone()));
        self.run_messages(messages, task, 0, 0, None).await
    }

    pub async fn resume(&self, checkpoint: HarnessCheckpoint) -> Result<RunOutcome> {
        if checkpoint.schema != 1 {
            return Err(RuntimeError::InvalidConfig(
                "unsupported harness checkpoint schema".into(),
            ));
        }
        let mut messages = checkpoint.messages;
        messages.push(Message::new(
            MessageRole::User,
            "[RESUME FROM SAFE CHECKPOINT] Continue the open task from the retained tool results and obligations.",
        ));
        let host_plan = checkpoint.host_plan;
        self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
        self.run_messages(
            messages,
            checkpoint.pending_task,
            checkpoint.completed_tool_calls,
            checkpoint.completed_tool_rounds,
            host_plan,
        )
        .await
    }

    async fn run_messages(
        &self,
        mut messages: Vec<Message>,
        task: String,
        mut completed_tool_calls: u64,
        mut completed_tool_rounds: u64,
        mut host_plan: Option<HostPlan>,
    ) -> Result<RunOutcome> {
        let mut usage = TokenUsage::default();
        let mut empty_plan_continuations = 0_u64;
        let mut empty_model_retries = 0_u64;
        self.ensure_plan_guidance(&mut messages);
        loop {
            let turn = self
                .provider
                .complete(CompletionRequest {
                    messages: messages.clone(),
                    tools: self.registry.specs(),
                    allow_tool_calls: true,
                })
                .await?;
            usage.add(turn.usage);
            let _ = self.events.send(HarnessEvent::ModelTurnCompleted);
            if turn.tool_calls.is_empty() {
                self.maybe_install_plan(&mut host_plan, &task, &turn.content);
                let controls = self.controls.drain();
                messages.push(Message::assistant(
                    &turn.content,
                    turn.reasoning,
                    Vec::new(),
                ));
                if !controls.is_empty() {
                    self.append_controls(&mut messages, &controls);
                }
                if controls.stop {
                    return self
                        .finish_gracefully(
                            messages,
                            task,
                            completed_tool_calls,
                            completed_tool_rounds,
                            usage,
                            host_plan,
                        )
                        .await;
                }
                if !controls.guidance.is_empty() {
                    self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
                    continue;
                }
                if turn.content.trim().is_empty() {
                    if empty_model_retries < MAX_EMPTY_MODEL_RETRIES {
                        empty_model_retries = empty_model_retries.saturating_add(1);
                        messages.push(Message::new(
                            MessageRole::User,
                            "[HOST EMPTY TURN] Return a concise answer now, or call the next necessary tool if work remains. Never return an empty message.",
                        ));
                        self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
                        continue;
                    }
                    let response = "The provider returned repeated empty responses. Completed tool results were retained; please retry the request.".to_owned();
                    messages.push(Message::new(MessageRole::Assistant, &response));
                    return Ok(RunOutcome {
                        response,
                        stopped_gracefully: false,
                        checkpoint: None,
                        completed_tool_calls,
                        completed_tool_rounds,
                        usage,
                        messages,
                        host_plan,
                    });
                }
                empty_model_retries = 0;

                if self.config.enforce_host_plans {
                    if let Some(plan) = host_plan.as_mut().filter(|plan| !plan.done()) {
                        let is_last = plan.cursor + 1 == plan.steps.len();
                        if is_last
                            && turn.content.trim().chars().count() >= 200
                            && !promised_more_work(&turn.content)
                        {
                            plan.mark_current_done("writeup");
                        }
                    }
                    if host_plan.as_ref().is_some_and(|plan| !plan.done()) {
                        empty_plan_continuations = empty_plan_continuations.saturating_add(1);
                        if empty_plan_continuations > self.config.max_empty_plan_continuations {
                            messages.push(Message::new(
                                MessageRole::User,
                                "[HOST PLAN CEILING] No evidence was produced for the open step. Return a concise progress report and explicitly state what remains, without claiming that work already completed.",
                            ));
                            return self
                                .finish_without_tools(
                                    messages,
                                    task,
                                    false,
                                    completed_tool_calls,
                                    completed_tool_rounds,
                                    usage,
                                    host_plan,
                                )
                                .await;
                        }
                        self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
                        continue;
                    }
                    if host_plan.is_none() && promised_more_work(&turn.content) {
                        empty_plan_continuations = empty_plan_continuations.saturating_add(1);
                        if empty_plan_continuations <= self.config.max_empty_plan_continuations {
                            messages.push(Message::new(
                                MessageRole::User,
                                "[HOST CONTINUATION: NEED PLAN] You promised more work but emitted no tools. Emit a ```plan block with 2-8 short steps, then call tools for step 1. Do not claim you will continue later.",
                            ));
                            continue;
                        }
                    }
                }
                return Ok(RunOutcome {
                    response: turn.content,
                    stopped_gracefully: false,
                    checkpoint: None,
                    completed_tool_calls,
                    completed_tool_rounds,
                    usage,
                    messages,
                    host_plan,
                });
            }

            empty_model_retries = 0;
            self.maybe_install_plan(&mut host_plan, &task, &turn.content);

            if self
                .config
                .max_tool_rounds
                .is_some_and(|ceiling| completed_tool_rounds >= ceiling.get())
            {
                messages.push(Message::new(MessageRole::Assistant, turn.content));
                messages.push(Message::new(
                    MessageRole::User,
                    "[HOST TOOL CEILING] Return a complete progress summary without calling tools.",
                ));
                return self
                    .finish_without_tools(
                        messages,
                        task,
                        false,
                        completed_tool_calls,
                        completed_tool_rounds,
                        usage,
                        host_plan,
                    )
                    .await;
            }

            completed_tool_rounds = completed_tool_rounds.saturating_add(1);
            let promised_more = promised_more_work(&turn.content);
            messages.push(Message::assistant(
                turn.content,
                turn.reasoning,
                turn.tool_calls.clone(),
            ));
            let mut boundary_controls = None;
            let mut plan_evidence = false;
            let calls = turn.tool_calls;
            for (index, call) in calls.iter().enumerate() {
                let result = self.execute_tool(call, &task).await;
                plan_evidence |= self.registry.get(&call.name).is_some_and(|tool| {
                    tool.spec().category != ToolCategory::Action || result.success
                });
                completed_tool_calls = completed_tool_calls.saturating_add(1);
                messages.push(Message::tool(
                    &call.id,
                    result.model_text(self.config.max_tool_result_chars),
                ));
                let controls = self.controls.drain();
                if !controls.is_empty() {
                    for skipped in &calls[index + 1..] {
                        messages.push(Message::tool(
                            &skipped.id,
                            "[skipped after an operator control at the completed-tool boundary]",
                        ));
                    }
                    self.append_controls(&mut messages, &controls);
                    boundary_controls = Some(controls);
                    break;
                }
            }
            if plan_evidence
                && !promised_more
                && let Some(plan) = host_plan.as_mut().filter(|plan| !plan.done())
            {
                plan.mark_current_done("tools");
            }
            if boundary_controls
                .as_ref()
                .is_some_and(|controls| controls.stop)
            {
                return self
                    .finish_gracefully(
                        messages,
                        task,
                        completed_tool_calls,
                        completed_tool_rounds,
                        usage,
                        host_plan,
                    )
                    .await;
            }
            self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
        }
    }

    fn ensure_plan_guidance(&self, messages: &mut [Message]) {
        if !self.config.enforce_host_plans {
            return;
        }
        let Some(system) = messages
            .iter_mut()
            .find(|message| message.role == MessageRole::System)
        else {
            return;
        };
        if !system.content.contains(PLAN_GUIDANCE_MARKER) {
            system.content.push_str(
                "\n\n[HOST PLAN CONTRACT]\nIf the task needs multiple actions, include a ```plan block with 2-8 short numbered steps and call tools for the current step only. The host advances the cursor only after tool evidence. Never end a response by promising work for later; call the needed tool now.",
            );
        }
    }

    fn maybe_install_plan(&self, host_plan: &mut Option<HostPlan>, task: &str, response: &str) {
        if !self.config.enforce_host_plans || host_plan.is_some() {
            return;
        }
        let steps = parse_plan_steps(response);
        if let Some(plan) = HostPlan::new(task.chars().take(200).collect::<String>(), steps) {
            *host_plan = Some(plan);
        }
    }

    fn append_open_plan_prompt(&self, messages: &mut Vec<Message>, host_plan: Option<&HostPlan>) {
        if !self.config.enforce_host_plans {
            return;
        }
        let Some(plan) = host_plan.filter(|plan| !plan.done()) else {
            return;
        };
        let Some(step) = plan.current() else {
            return;
        };
        messages.push(Message::new(
            MessageRole::User,
            format!(
                "[HOST PLAN — CURRENT STEP]\n{}\nDo only step {}/{}: {}\nCall the needed tools now. Do not skip ahead or promise to continue later.",
                plan.progress(),
                plan.cursor + 1,
                plan.steps.len(),
                step.text
            ),
        ));
    }

    async fn execute_tool(&self, call: &ToolCall, task: &str) -> ToolResult {
        let Some(tool) = self.registry.get(&call.name) else {
            return ToolResult::failure(format!("tool {:?} is unavailable", call.name));
        };
        let _ = self.events.send(HarnessEvent::ToolStarted {
            id: call.id.clone(),
            name: call.name.clone(),
        });
        let call_risk = tool.risk_for_call(call);
        let result = if call_risk == ToolRisk::Destructive && !self.config.allow_destructive_tools {
            ToolResult::failure("destructive tool call blocked by this harness")
        } else {
            let mut metadata = std::collections::BTreeMap::new();
            metadata.insert("task".into(), task.into());
            tool.execute(
                call,
                &ToolContext {
                    harness_id: self.id.clone(),
                    agent_id: self.agent_id.clone(),
                    metadata,
                },
            )
            .await
            .unwrap_or_else(|error| ToolResult::failure(error.to_string()))
        };
        let _ = self.events.send(HarnessEvent::ToolCompleted {
            id: call.id.clone(),
            name: call.name.clone(),
            success: result.success,
        });
        result
    }

    fn append_controls(&self, messages: &mut Vec<Message>, controls: &OperatorControls) {
        if !controls.guidance.is_empty() {
            let _ = self.events.send(HarnessEvent::GuidanceInjected {
                messages: controls.guidance.clone(),
            });
            messages.push(Message::new(
                MessageRole::User,
                format!(
                    "[OPERATOR INJECTION — AFTER TOOL BOUNDARY]\n{}\nThis supplements the active task. Briefly answer or acknowledge it, then continue unless it requests a stop.",
                    controls.guidance.join("\n")
                ),
            ));
        }
        if controls.stop {
            let _ = self.events.send(HarnessEvent::GracefulStopBoundary);
            messages.push(Message::new(
                MessageRole::User,
                "[OPERATOR GRACEFUL STOP] Preserve completed work, state open obligations, and return a concise resumable summary without more tools.",
            ));
        }
    }

    async fn finish_gracefully(
        &self,
        messages: Vec<Message>,
        task: String,
        completed_tool_calls: u64,
        completed_tool_rounds: u64,
        usage: TokenUsage,
        host_plan: Option<HostPlan>,
    ) -> Result<RunOutcome> {
        self.finish_without_tools(
            messages,
            task,
            true,
            completed_tool_calls,
            completed_tool_rounds,
            usage,
            host_plan,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_without_tools(
        &self,
        mut messages: Vec<Message>,
        task: String,
        stopped_gracefully: bool,
        completed_tool_calls: u64,
        completed_tool_rounds: u64,
        mut usage: TokenUsage,
        host_plan: Option<HostPlan>,
    ) -> Result<RunOutcome> {
        let turn = self
            .provider
            .complete(CompletionRequest {
                messages: messages.clone(),
                tools: Vec::new(),
                allow_tool_calls: false,
            })
            .await?;
        usage.add(turn.usage);
        let response = if turn.content.trim().is_empty() {
            "Stopped at a safe tool boundary; completed results are retained in the checkpoint."
                .into()
        } else {
            turn.content
        };
        messages.push(Message::new(MessageRole::Assistant, &response));
        let checkpoint = stopped_gracefully.then(|| HarnessCheckpoint {
            schema: 1,
            harness_id: self.id.clone(),
            messages: messages.clone(),
            completed_tool_calls,
            completed_tool_rounds,
            pending_task: task,
            host_plan: host_plan.clone(),
        });
        Ok(RunOutcome {
            response,
            stopped_gracefully,
            checkpoint,
            completed_tool_calls,
            completed_tool_rounds,
            usage,
            messages,
            host_plan,
        })
    }
}
