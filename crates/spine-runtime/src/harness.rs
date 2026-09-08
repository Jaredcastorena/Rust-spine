use std::{
    collections::VecDeque,
    num::{NonZeroU64, NonZeroUsize},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::{
    CompletionRequest, HostPlan, Message, MessageRole, ModelProvider, PlanStepStatus, Result,
    RuntimeError, TokenUsage, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult,
    ToolRisk, parse_plan_steps, promised_more_work,
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

/// Per-run controls selected by host policy from committed cognitive state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessPolicy {
    pub temperature: Option<f32>,
    pub max_action_calls: Option<NonZeroUsize>,
    /// Effective ceiling for this run; `None` explicitly means unlimited.
    pub max_tool_rounds: Option<NonZeroU64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompletedWork {
    tool_calls: u64,
    tool_rounds: u64,
    action_calls: usize,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessCheckpoint {
    pub schema: u32,
    pub harness_id: String,
    pub messages: Vec<Message>,
    pub completed_tool_calls: u64,
    pub completed_tool_rounds: u64,
    #[serde(default)]
    pub completed_action_calls: usize,
    pub pending_task: String,
    #[serde(default)]
    pub host_plan: Option<HostPlan>,
    #[serde(default)]
    pub policy: HarnessPolicy,
}

impl HarnessCheckpoint {
    pub const RECORD_TYPE: &'static str = "harness_checkpoint";
    pub const OUTCOME: &'static str = "graceful_stop_checkpoint";

    pub fn to_interaction(
        &self,
        agent_id: spine_heart::AgentId,
        thread_id: spine_heart::ThreadId,
    ) -> Result<spine_heart::InteractionInput> {
        self.validate()?;
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("record_type".into(), Self::RECORD_TYPE.into());
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
            outcome: Some(Self::OUTCOME.into()),
        })
    }

    pub fn from_interaction(interaction: &spine_heart::InteractionInput) -> Result<Self> {
        if interaction.role != spine_heart::ParticipantRole::Operator
            || interaction.kind != spine_heart::EventKind::Control
            || interaction.outcome.as_deref() != Some(Self::OUTCOME)
            || interaction
                .provenance
                .metadata
                .get("record_type")
                .map(String::as_str)
                != Some(Self::RECORD_TYPE)
            || interaction.tool.is_some()
            || !interaction.attachments.is_empty()
        {
            return Err(invalid_checkpoint(
                "persisted record does not have the checkpoint envelope",
            ));
        }
        let spine_heart::Content::Inline(content) = &interaction.content else {
            return Err(invalid_checkpoint("checkpoint content is not inline JSON"));
        };
        let checkpoint: Self = serde_json::from_str(content)
            .map_err(|error| invalid_checkpoint(format!("invalid checkpoint JSON: {error}")))?;
        checkpoint.validate()?;
        if interaction
            .provenance
            .metadata
            .get("harness_id")
            .map(String::as_str)
            != Some(checkpoint.harness_id.as_str())
        {
            return Err(invalid_checkpoint(
                "checkpoint metadata does not match its payload",
            ));
        }
        Ok(checkpoint)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != 1 {
            return Err(invalid_checkpoint("unsupported checkpoint schema"));
        }
        if self.harness_id.trim().is_empty()
            || self.harness_id.trim() != self.harness_id
            || self.harness_id.len() > 256
        {
            return Err(invalid_checkpoint(
                "harness id must contain 1..=256 trimmed bytes",
            ));
        }
        if self.pending_task.trim().is_empty() {
            return Err(invalid_checkpoint("pending task is empty"));
        }
        if self
            .policy
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self.completed_action_calls as u128 > u128::from(self.completed_tool_calls)
            || self
                .policy
                .max_action_calls
                .is_some_and(|limit| self.completed_action_calls > limit.get())
        {
            return Err(invalid_checkpoint(
                "checkpoint action policy or counters are invalid",
            ));
        }
        if self.messages.len() < 3
            || self.messages.first().map(|message| message.role) != Some(MessageRole::System)
            || self.messages.last().map(|message| message.role) != Some(MessageRole::Assistant)
        {
            return Err(invalid_checkpoint(
                "checkpoint transcript is not a complete system/user/assistant exchange",
            ));
        }

        let mut outstanding_calls = std::collections::BTreeSet::new();
        let mut tool_rounds = 0_u64;
        let mut tool_results = 0_u64;
        let mut saw_user = false;
        for message in &self.messages {
            if !outstanding_calls.is_empty() && message.role != MessageRole::Tool {
                return Err(invalid_checkpoint(
                    "assistant tool calls are missing adjacent tool results",
                ));
            }
            match message.role {
                MessageRole::System | MessageRole::User => {
                    if message.role == MessageRole::User {
                        saw_user = true;
                    }
                    if message.tool_call_id.is_some()
                        || !message.tool_calls.is_empty()
                        || message.reasoning.is_some()
                    {
                        return Err(invalid_checkpoint(
                            "system or user message contains assistant/tool metadata",
                        ));
                    }
                }
                MessageRole::Assistant => {
                    if message.tool_call_id.is_some() {
                        return Err(invalid_checkpoint(
                            "assistant message contains a tool result id",
                        ));
                    }
                    if !message.tool_calls.is_empty() {
                        tool_rounds = tool_rounds.saturating_add(1);
                        for call in &message.tool_calls {
                            if call.id.trim().is_empty()
                                || call.name.trim().is_empty()
                                || !outstanding_calls.insert(call.id.as_str())
                            {
                                return Err(invalid_checkpoint(
                                    "assistant tool calls have empty or duplicate identities",
                                ));
                            }
                        }
                    }
                }
                MessageRole::Tool => {
                    if message.reasoning.is_some() || !message.tool_calls.is_empty() {
                        return Err(invalid_checkpoint(
                            "tool result contains assistant metadata",
                        ));
                    }
                    let Some(call_id) = message.tool_call_id.as_deref() else {
                        return Err(invalid_checkpoint("tool result is missing its call id"));
                    };
                    if !outstanding_calls.remove(call_id) {
                        return Err(invalid_checkpoint(
                            "tool result does not match an outstanding assistant call",
                        ));
                    }
                    tool_results = tool_results.saturating_add(1);
                }
            }
        }
        if !saw_user || !outstanding_calls.is_empty() {
            return Err(invalid_checkpoint(
                "checkpoint transcript has no user task or ends inside a tool batch",
            ));
        }
        let final_message = self.messages.last().expect("length checked");
        if final_message.content.trim().is_empty()
            || !final_message.tool_calls.is_empty()
            || final_message.tool_call_id.is_some()
        {
            return Err(invalid_checkpoint(
                "checkpoint transcript does not end with a safe assistant summary",
            ));
        }
        if self.completed_tool_rounds > tool_rounds || self.completed_tool_calls > tool_results {
            return Err(invalid_checkpoint(
                "checkpoint tool counters do not match its transcript",
            ));
        }
        if let Some(plan) = &self.host_plan {
            validate_checkpoint_plan(plan)?;
        }
        Ok(())
    }
}

fn validate_checkpoint_plan(plan: &HostPlan) -> Result<()> {
    if plan.goal.trim().is_empty()
        || !(2..=8).contains(&plan.steps.len())
        || plan.cursor > plan.steps.len()
    {
        return Err(invalid_checkpoint("host plan shape is invalid"));
    }
    for (offset, step) in plan.steps.iter().enumerate() {
        let expected_status = if offset < plan.cursor {
            PlanStepStatus::Done
        } else if offset == plan.cursor {
            PlanStepStatus::Active
        } else {
            PlanStepStatus::Pending
        };
        if step.index != offset + 1
            || step.text.trim().is_empty()
            || step.text.chars().count() > 160
            || step.status != expected_status
            || (step.status == PlanStepStatus::Done) != !step.evidence.trim().is_empty()
        {
            return Err(invalid_checkpoint("host plan state is inconsistent"));
        }
    }
    Ok(())
}

fn invalid_checkpoint(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidConfig(format!("invalid harness checkpoint: {}", message.into()))
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
    pub completed_action_calls: usize,
    pub policy: HarnessPolicy,
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
            CompletedWork::default(),
            None,
            self.default_policy(),
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
        self.run_messages(
            messages,
            task,
            CompletedWork::default(),
            None,
            self.default_policy(),
        )
        .await
    }

    pub async fn run_with_history_policy(
        &self,
        system: impl Into<String>,
        history: &[Message],
        task: impl Into<String>,
        policy: HarnessPolicy,
    ) -> Result<RunOutcome> {
        let task = task.into();
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(Message::new(MessageRole::System, system));
        messages.extend_from_slice(history);
        messages.push(Message::new(MessageRole::User, task.clone()));
        self.run_messages(messages, task, CompletedWork::default(), None, policy)
            .await
    }

    fn default_policy(&self) -> HarnessPolicy {
        HarnessPolicy {
            max_tool_rounds: self.config.max_tool_rounds,
            ..HarnessPolicy::default()
        }
    }

    /// Continue a completed draft for host verification without refreshing its budgets.
    /// Graceful stops require an explicit checkpoint resume instead.
    pub async fn repair(
        &self,
        previous: &RunOutcome,
        task: impl Into<String>,
    ) -> Result<RunOutcome> {
        if previous.stopped_gracefully || previous.checkpoint.is_some() {
            return Err(RuntimeError::InvalidConfig(
                "a stopped outcome requires explicit checkpoint resume".into(),
            ));
        }
        let task = task.into();
        let mut messages = previous.messages.clone();
        messages.push(Message::new(MessageRole::User, task.clone()));
        let mut outcome = self
            .run_messages(
                messages,
                task,
                CompletedWork {
                    tool_calls: previous.completed_tool_calls,
                    tool_rounds: previous.completed_tool_rounds,
                    action_calls: previous.completed_action_calls,
                },
                previous.host_plan.clone(),
                previous.policy,
            )
            .await?;
        outcome.usage.add(previous.usage);
        Ok(outcome)
    }

    pub async fn resume(&self, checkpoint: HarnessCheckpoint) -> Result<RunOutcome> {
        checkpoint.validate()?;
        let mut messages = checkpoint.messages;
        messages.push(Message::new(
            MessageRole::User,
            "[RESUME FROM SAFE CHECKPOINT] Continue the open task from the retained tool results and obligations.",
        ));
        let host_plan = checkpoint.host_plan;
        let policy = checkpoint.policy;
        self.append_open_plan_prompt(&mut messages, host_plan.as_ref());
        self.run_messages(
            messages,
            checkpoint.pending_task,
            CompletedWork {
                tool_calls: checkpoint.completed_tool_calls,
                tool_rounds: checkpoint.completed_tool_rounds,
                action_calls: checkpoint.completed_action_calls,
            },
            host_plan,
            policy,
        )
        .await
    }

    async fn run_messages(
        &self,
        mut messages: Vec<Message>,
        task: String,
        mut completed: CompletedWork,
        mut host_plan: Option<HostPlan>,
        policy: HarnessPolicy,
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
                    temperature: policy.temperature,
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
                        .finish_gracefully(messages, task, completed, usage, host_plan, policy)
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
                        completed_tool_calls: completed.tool_calls,
                        completed_tool_rounds: completed.tool_rounds,
                        completed_action_calls: completed.action_calls,
                        policy,
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
                                    messages, task, false, completed, usage, host_plan, policy,
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
                    completed_tool_calls: completed.tool_calls,
                    completed_tool_rounds: completed.tool_rounds,
                    completed_action_calls: completed.action_calls,
                    policy,
                    usage,
                    messages,
                    host_plan,
                });
            }

            empty_model_retries = 0;
            self.maybe_install_plan(&mut host_plan, &task, &turn.content);

            if policy
                .max_tool_rounds
                .is_some_and(|ceiling| completed.tool_rounds >= ceiling.get())
            {
                messages.push(Message::new(MessageRole::Assistant, turn.content));
                messages.push(Message::new(
                    MessageRole::User,
                    "[HOST TOOL CEILING] Return a complete progress summary without calling tools.",
                ));
                return self
                    .finish_without_tools(
                        messages, task, false, completed, usage, host_plan, policy,
                    )
                    .await;
            }

            completed.tool_rounds = completed.tool_rounds.saturating_add(1);
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
                let is_action = self.registry.get(&call.name).is_some_and(|tool| {
                    matches!(
                        tool.spec().category,
                        ToolCategory::Action | ToolCategory::Both
                    )
                });
                if is_action
                    && policy
                        .max_action_calls
                        .is_some_and(|ceiling| completed.action_calls >= ceiling.get())
                {
                    messages.push(Message::tool(
                        &call.id,
                        "[skipped: host action budget for this run was reached]",
                    ));
                } else {
                    if is_action {
                        completed.action_calls = completed.action_calls.saturating_add(1);
                    }
                    let result = self.execute_tool(call, &task).await;
                    plan_evidence |= self.registry.get(&call.name).is_some_and(|tool| {
                        tool.spec().category != ToolCategory::Action || result.success
                    });
                    completed.tool_calls = completed.tool_calls.saturating_add(1);
                    messages.push(Message::tool(
                        &call.id,
                        result.model_text(self.config.max_tool_result_chars),
                    ));
                }
                // A rejected action is also a safe boundary. Otherwise an
                // exhausted budget could starve queued guidance/stop forever.
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
                    .finish_gracefully(messages, task, completed, usage, host_plan, policy)
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
        completed: CompletedWork,
        usage: TokenUsage,
        host_plan: Option<HostPlan>,
        policy: HarnessPolicy,
    ) -> Result<RunOutcome> {
        self.finish_without_tools(messages, task, true, completed, usage, host_plan, policy)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_without_tools(
        &self,
        mut messages: Vec<Message>,
        task: String,
        stopped_gracefully: bool,
        completed: CompletedWork,
        mut usage: TokenUsage,
        host_plan: Option<HostPlan>,
        policy: HarnessPolicy,
    ) -> Result<RunOutcome> {
        let turn = self
            .provider
            .complete(CompletionRequest {
                messages: messages.clone(),
                tools: Vec::new(),
                allow_tool_calls: false,
                temperature: policy.temperature,
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
            completed_tool_calls: completed.tool_calls,
            completed_tool_rounds: completed.tool_rounds,
            completed_action_calls: completed.action_calls,
            pending_task: task,
            host_plan: host_plan.clone(),
            policy,
        });
        Ok(RunOutcome {
            response,
            stopped_gracefully,
            checkpoint,
            completed_tool_calls: completed.tool_calls,
            completed_tool_rounds: completed.tool_rounds,
            completed_action_calls: completed.action_calls,
            policy,
            usage,
            messages,
            host_plan,
        })
    }
}
