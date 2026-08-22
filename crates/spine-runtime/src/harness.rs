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
    CompletionRequest, Message, MessageRole, ModelProvider, Result, RuntimeError, TokenUsage,
    ToolCall, ToolContext, ToolRegistry, ToolResult, ToolRisk,
};

#[derive(Clone, Debug)]
pub struct HarnessConfig {
    pub max_tool_rounds: Option<NonZeroU64>,
    pub max_tool_result_chars: usize,
    pub allow_destructive_tools: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: None,
            max_tool_result_chars: 16_384,
            allow_destructive_tools: false,
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
        if config.max_tool_result_chars == 0 {
            return Err(RuntimeError::InvalidConfig(
                "maximum tool result length must be positive".into(),
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
        self.run_messages(messages, task, 0, 0).await
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
        self.run_messages(
            messages,
            checkpoint.pending_task,
            checkpoint.completed_tool_calls,
            checkpoint.completed_tool_rounds,
        )
        .await
    }

    async fn run_messages(
        &self,
        mut messages: Vec<Message>,
        task: String,
        mut completed_tool_calls: u64,
        mut completed_tool_rounds: u64,
    ) -> Result<RunOutcome> {
        let mut usage = TokenUsage::default();
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
                let controls = self.controls.drain();
                if controls.is_empty() {
                    messages.push(Message::assistant(
                        &turn.content,
                        turn.reasoning,
                        Vec::new(),
                    ));
                    return Ok(RunOutcome {
                        response: turn.content,
                        stopped_gracefully: false,
                        checkpoint: None,
                        completed_tool_calls,
                        completed_tool_rounds,
                        usage,
                        messages,
                    });
                }
                messages.push(Message::assistant(turn.content, turn.reasoning, Vec::new()));
                self.append_controls(&mut messages, &controls);
                if controls.stop {
                    return self
                        .finish_gracefully(
                            messages,
                            task,
                            completed_tool_calls,
                            completed_tool_rounds,
                            usage,
                        )
                        .await;
                }
                continue;
            }

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
                    )
                    .await;
            }

            completed_tool_rounds = completed_tool_rounds.saturating_add(1);
            messages.push(Message::assistant(
                turn.content,
                turn.reasoning,
                turn.tool_calls.clone(),
            ));
            let mut boundary_controls = None;
            for call in turn.tool_calls {
                let result = self.execute_tool(&call, &task).await;
                completed_tool_calls = completed_tool_calls.saturating_add(1);
                messages.push(Message::tool(
                    &call.id,
                    result.model_text(self.config.max_tool_result_chars),
                ));
                let controls = self.controls.drain();
                if !controls.is_empty() {
                    self.append_controls(&mut messages, &controls);
                    boundary_controls = Some(controls);
                    break;
                }
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
                    )
                    .await;
            }
        }
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
    ) -> Result<RunOutcome> {
        self.finish_without_tools(
            messages,
            task,
            true,
            completed_tool_calls,
            completed_tool_rounds,
            usage,
        )
        .await
    }

    async fn finish_without_tools(
        &self,
        mut messages: Vec<Message>,
        task: String,
        stopped_gracefully: bool,
        completed_tool_calls: u64,
        completed_tool_rounds: u64,
        mut usage: TokenUsage,
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
        });
        Ok(RunOutcome {
            response,
            stopped_gracefully,
            checkpoint,
            completed_tool_calls,
            completed_tool_rounds,
            usage,
            messages,
        })
    }
}
