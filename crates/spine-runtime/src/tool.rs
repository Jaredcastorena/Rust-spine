use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ToolCategory {
    Internal,
    Action,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ToolRisk {
    ReadOnly,
    Mutating,
    Destructive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub risk: ToolRisk,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }

    pub fn model_text(&self, maximum_chars: usize) -> String {
        let text = self.error.as_ref().unwrap_or(&self.output);
        if text.chars().count() <= maximum_chars {
            text.clone()
        } else {
            text.chars().take(maximum_chars).collect::<String>() + "\n[truncated]"
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolContext {
    pub harness_id: String,
    pub agent_id: Option<spine_heart::AgentId>,
    pub metadata: BTreeMap<String, String>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn risk_for_call(&self, _call: &ToolCall) -> ToolRisk {
        self.spec().risk
    }
    async fn execute(&self, call: &ToolCall, context: &ToolContext) -> Result<ToolResult>;
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        self.register_shared(Arc::new(tool))
    }

    pub fn register_shared(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.spec().name;
        if self.tools.contains_key(&name) {
            return Err(RuntimeError::InvalidConfig(format!(
                "tool {name:?} is already registered"
            )));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|tool| tool.spec()).collect()
    }

    pub fn curated(&self, allowed: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Self> {
        let mut registry = Self::default();
        for name in allowed {
            let name = name.as_ref();
            let tool = self.tools.get(name).ok_or_else(|| {
                RuntimeError::InvalidConfig(format!("curated tool {name:?} is unavailable"))
            })?;
            registry.register_shared(Arc::clone(tool))?;
        }
        Ok(registry)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
