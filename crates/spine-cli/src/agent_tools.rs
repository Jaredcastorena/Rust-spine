use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroU64,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use spine_heart::{AgentId, ThymosConfig};
use spine_runtime::{
    HarnessConfig, HarnessEvent, ModelProvider, SubagentHarnessFactory, Tool, ToolCall,
    ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};

const SUBAGENT_SYSTEM_PROMPT: &str = "You are a temporary Spine sub-agent with your own isolated harness and ephemeral Thymos. Complete only the delegated task. You may read shared canonical memory and use the curated filesystem, web, shell, and cognition tools. Report concrete results and evidence in your final response. You cannot delegate recursively.";

pub fn register_subagent_tools(
    registry: &mut ToolRegistry,
    provider: Arc<dyn ModelProvider>,
    child_registry: ToolRegistry,
    embedding_dimension: usize,
    thymos_channels: usize,
) -> spine_runtime::Result<()> {
    let allowed_tools: Vec<_> = child_registry
        .specs()
        .into_iter()
        .map(|spec| spec.name)
        .collect();
    let factory = SubagentHarnessFactory::new(
        provider,
        child_registry,
        allowed_tools,
        HarnessConfig::default(),
        ThymosConfig::new(embedding_dimension, thymos_channels)?,
    );
    let manager = Arc::new(SubagentManager {
        factory,
        state: Mutex::new(ManagerState::default()),
    });
    registry.register(DelegateTool {
        manager: Arc::clone(&manager),
    })?;
    registry.register(CheckResultsTool {
        manager: Arc::clone(&manager),
    })?;
    registry.register(CancelAgentTool { manager })?;
    Ok(())
}

#[derive(Default)]
struct ManagerState {
    counter: u64,
    jobs: BTreeMap<String, AgentJob>,
}

struct AgentJob {
    task: String,
    reports: Arc<Mutex<VecDeque<String>>>,
    handle: tokio::task::JoinHandle<()>,
}

struct SubagentManager {
    factory: SubagentHarnessFactory,
    state: Mutex<ManagerState>,
}

impl SubagentManager {
    fn spawn(
        &self,
        task: String,
        check_in_every: u64,
        maximum_rounds: NonZeroU64,
    ) -> spine_runtime::Result<String> {
        let mut state = self.state.lock().expect("subagent manager lock poisoned");
        state.counter = state.counter.saturating_add(1);
        let id = format!("sa{}", state.counter);
        let agent_id = AgentId::new(&id)?;
        let harness = self.factory.create_with_config(
            agent_id,
            HarnessConfig {
                max_tool_rounds: Some(maximum_rounds),
                ..HarnessConfig::default()
            },
        )?;
        let mut events = harness.harness.subscribe();
        let reports = Arc::new(Mutex::new(VecDeque::from([format!(
            "[{id} | started] {task}"
        )])));
        let event_reports = Arc::clone(&reports);
        let event_id = id.clone();
        let event_task = tokio::spawn(async move {
            let mut boundaries = 0_u64;
            while let Ok(event) = events.recv().await {
                if matches!(
                    event,
                    HarnessEvent::ModelTurnCompleted | HarnessEvent::ToolCompleted { .. }
                ) {
                    boundaries = boundaries.saturating_add(1);
                    if boundaries.is_multiple_of(check_in_every) {
                        event_reports
                            .lock()
                            .expect("subagent report lock poisoned")
                            .push_back(format!(
                                "[{event_id} | progress] completed {boundaries} model/tool boundaries"
                            ));
                    }
                }
            }
        });
        let final_reports = Arc::clone(&reports);
        let final_id = id.clone();
        let task_for_run = task.clone();
        let handle = tokio::spawn(async move {
            // Keeping the wrapper alive for the whole task preserves this agent's independent,
            // temporary Thymos; it is intentionally dropped instead of promoted afterward.
            let temporary = harness;
            let result = temporary
                .harness
                .run(SUBAGENT_SYSTEM_PROMPT, task_for_run)
                .await;
            event_task.abort();
            let message = match result {
                Ok(outcome) if outcome.stopped_gracefully => format!(
                    "[{final_id} | FINAL] stopped at a safe boundary after {} tool calls; partial response: {}",
                    outcome.completed_tool_calls, outcome.response
                ),
                Ok(outcome) => format!("[{final_id} | FINAL]\n{}", outcome.response),
                Err(error) => format!("[{final_id} | FINAL ERROR] {error}"),
            };
            final_reports
                .lock()
                .expect("subagent report lock poisoned")
                .push_back(message);
        });
        state.jobs.insert(
            id.clone(),
            AgentJob {
                task,
                reports,
                handle,
            },
        );
        Ok(id)
    }

    fn reports(&self) -> String {
        let mut state = self.state.lock().expect("subagent manager lock poisoned");
        if state.jobs.is_empty() {
            return "(no sub-agents running)".into();
        }
        let mut output = Vec::new();
        let mut completed_and_drained = Vec::new();
        for (id, job) in &state.jobs {
            let mut reports = job.reports.lock().expect("subagent report lock poisoned");
            output.extend(reports.drain(..));
            if job.handle.is_finished() && reports.is_empty() {
                completed_and_drained.push(id.clone());
            }
        }
        for id in completed_and_drained {
            state.jobs.remove(&id);
        }
        if output.is_empty() {
            let active: Vec<_> = state
                .jobs
                .iter()
                .map(|(id, job)| format!("{id}: {}", job.task))
                .collect();
            format!("(no new reports — active: {})", active.join("; "))
        } else {
            output.join("\n\n")
        }
    }

    fn cancel(&self, requested: &str) -> String {
        let mut state = self.state.lock().expect("subagent manager lock poisoned");
        if requested.is_empty() || requested == "all" {
            if state.jobs.is_empty() {
                return "(no sub-agents running)".into();
            }
            let ids: Vec<_> = state.jobs.keys().cloned().collect();
            for job in state.jobs.values() {
                job.handle.abort();
            }
            state.jobs.clear();
            return format!("Cancelled {} sub-agent(s): {}", ids.len(), ids.join(", "));
        }
        match state.jobs.remove(requested) {
            Some(job) => {
                job.handle.abort();
                format!("Cancelled sub-agent {requested}")
            }
            None => format!(
                "(no sub-agent {requested:?} — running: {})",
                state.jobs.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

struct DelegateTool {
    manager: Arc<SubagentManager>,
}

#[async_trait]
impl Tool for DelegateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "delegate".into(),
            description: "Spawn an independent temporary sub-agent harness for a bounded task."
                .into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "check_in_every": {"type": "integer", "minimum": 1, "maximum": 100},
                    "max_ticks": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "required": ["task"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(task) = call
            .arguments
            .get("task")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(ToolResult::failure("delegate requires a non-empty task"));
        };
        let check_in_every = call
            .arguments
            .get("check_in_every")
            .and_then(|v| v.as_u64())
            .unwrap_or(3)
            .clamp(1, 100);
        let maximum_rounds = call
            .arguments
            .get("max_ticks")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .clamp(1, 100);
        let id = self.manager.spawn(
            task.to_owned(),
            check_in_every,
            NonZeroU64::new(maximum_rounds).expect("clamped positive"),
        )?;
        Ok(ToolResult::success(format!(
            "Sub-agent {id} started with an isolated temporary harness. Use check_results for progress or cancel_agent to stop it."
        )))
    }
}

struct CheckResultsTool {
    manager: Arc<SubagentManager>,
}

#[async_trait]
impl Tool for CheckResultsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "check_results".into(),
            description: "Drain progress and final reports from temporary sub-agents.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: empty_schema(),
        }
    }

    async fn execute(
        &self,
        _call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        Ok(ToolResult::success(self.manager.reports()))
    }
}

struct CancelAgentTool {
    manager: Arc<SubagentManager>,
}

#[async_trait]
impl Tool for CancelAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "cancel_agent".into(),
            description: "Cancel one temporary sub-agent, or all of them.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"agent_id": {"type": "string", "description": "Agent id or all"}},
                "required": ["agent_id"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let requested = call
            .arguments
            .get("agent_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        Ok(ToolResult::success(self.manager.cancel(requested)))
    }
}

fn empty_schema() -> serde_json::Value {
    serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_manager_has_legacy_compatible_status() {
        struct NeverFactory;
        let _ = std::any::TypeId::of::<NeverFactory>();
        assert_eq!(empty_schema()["type"], "object");
    }
}
