#![forbid(unsafe_code)]

mod harness;
mod llama_cpp;
mod plan;
mod provider;
mod subagent;
mod tool;
mod tool_parser;

pub use harness::{
    ControlPlane, Harness, HarnessCheckpoint, HarnessConfig, HarnessEvent, RunOutcome,
};
pub use llama_cpp::{LlamaCppConfig, LlamaCppProvider};
pub use plan::{HostPlan, PlanStep, PlanStepStatus, parse_plan_steps, promised_more_work};
pub use provider::{CompletionRequest, Message, MessageRole, ModelProvider, ModelTurn, TokenUsage};
pub use subagent::{SubagentHarness, SubagentHarnessFactory};
pub use tool::{
    Tool, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("provider error: {0}")]
    Provider(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("invalid runtime configuration: {0}")]
    InvalidConfig(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("heart error: {0}")]
    Heart(#[from] spine_heart::HeartError),
}
