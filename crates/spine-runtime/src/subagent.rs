use std::{collections::BTreeSet, sync::Arc};

use spine_heart::{AgentId, Thymos, ThymosConfig};

use crate::{Harness, HarnessConfig, ModelProvider, Result, ToolRegistry};

pub struct SubagentHarness {
    pub harness: Harness,
    pub thymos: Thymos,
    pub persistent: bool,
}

impl SubagentHarness {
    pub fn promote(mut self) -> Self {
        self.persistent = true;
        self
    }
}

pub struct SubagentHarnessFactory {
    provider: Arc<dyn ModelProvider>,
    parent_registry: ToolRegistry,
    allowed_tools: BTreeSet<String>,
    config: HarnessConfig,
    thymos_config: ThymosConfig,
}

impl SubagentHarnessFactory {
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        parent_registry: ToolRegistry,
        allowed_tools: impl IntoIterator<Item = impl Into<String>>,
        config: HarnessConfig,
        thymos_config: ThymosConfig,
    ) -> Self {
        Self {
            provider,
            parent_registry,
            allowed_tools: allowed_tools.into_iter().map(Into::into).collect(),
            config,
            thymos_config,
        }
    }

    pub fn create(&self, agent_id: AgentId) -> Result<SubagentHarness> {
        self.create_with_config(agent_id, self.config.clone())
    }

    pub fn create_with_config(
        &self,
        agent_id: AgentId,
        config: HarnessConfig,
    ) -> Result<SubagentHarness> {
        let registry = self.parent_registry.curated(&self.allowed_tools)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"spine-ephemeral-subagent-v1");
        hasher.update(agent_id.as_str().as_bytes());
        let thymos = Thymos::with_seed(self.thymos_config.clone(), *hasher.finalize().as_bytes())?;
        let harness =
            Harness::new(Arc::clone(&self.provider), registry, config)?.with_agent_id(agent_id);
        Ok(SubagentHarness {
            harness,
            thymos,
            persistent: false,
        })
    }
}
