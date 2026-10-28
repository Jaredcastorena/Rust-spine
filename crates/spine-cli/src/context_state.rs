//! Retained goals and evidence sit alongside, never replace, Rust's verified context forest.
use serde::{Deserialize, Serialize};
use spine_heart::{AgentId, ContextHandle, Embedding, SemanticEncoder, SpineHeart, ThreadId};
use spine_runtime::{HostPlan, MessageRole, RunOutcome};

#[derive(Clone, Default, Serialize, Deserialize)]
struct WorkingContext {
    schema: u32,
    goal: String,
    plan: Option<HostPlan>,
    obligations: Vec<String>,
    working_evidence: Vec<String>,
    evidence_records: Vec<String>,
    trajectory: serde_json::Value,
    tensions: Vec<String>,
    dormant_handles: Vec<ContextHandle>,
}

fn key(thread: &ThreadId) -> String {
    format!("context/{}", thread.as_str())
}
fn load(heart: &SpineHeart, thread: &ThreadId) -> spine_runtime::Result<WorkingContext> {
    match heart.local_states(&key(thread))?.get(&key(thread)) {
        Some(bytes) => Ok(serde_json::from_slice(bytes)?),
        None => Ok(WorkingContext::default()),
    }
}

pub fn remember(
    heart: &SpineHeart,
    agent: &AgentId,
    thread: &ThreadId,
    outcome: &RunOutcome,
) -> spine_runtime::Result<()> {
    let mut state = load(heart, thread)?;
    state.schema = 1;
    state.goal = outcome
        .host_plan
        .as_ref()
        .map(|p| p.goal.clone())
        .or_else(|| {
            outcome
                .messages
                .iter()
                .rev()
                .find(|m| m.role == MessageRole::User && !m.content.starts_with('['))
                .map(|m| m.content.clone())
        })
        .unwrap_or(state.goal);
    state.plan = outcome.host_plan.clone();
    state.obligations = outcome
        .host_plan
        .as_ref()
        .map(|p| {
            p.steps
                .iter()
                .skip(p.cursor)
                .map(|s| s.text.clone())
                .collect()
        })
        .unwrap_or_default();
    if outcome.stopped_gracefully && state.obligations.is_empty() {
        state.obligations.push(state.goal.clone());
    }
    state.working_evidence = outcome
        .messages
        .iter()
        .rev()
        .filter(|m| m.role == MessageRole::Tool)
        .take(6)
        .map(|m| m.content.chars().take(1500).collect())
        .collect();
    state.evidence_records = outcome
        .messages
        .iter()
        .filter_map(|m| m.record_id.clone())
        .collect();
    if let Some(cognition) = heart.cognition()? {
        state.trajectory = cognition
            .thymos
            .get(agent)
            .map(|t| t.state_summary())
            .unwrap_or_default();
        state.tensions = cognition
            .dcmdb
            .nodes
            .values()
            .filter(|n| n.tension.is_some())
            .map(|n| n.id.to_string())
            .collect();
        state.dormant_handles = cognition.triangles.roots.iter().map(|r| r.handle).collect();
    }
    heart.set_local_state(&key(thread), Some(&serde_json::to_vec(&state)?))?;
    Ok(())
}

pub fn rehydrate(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    thread: &ThreadId,
    query: &Embedding,
) -> spine_runtime::Result<String> {
    let mut state = load(heart, thread)?;
    // Reuse canonical checkpoint consumption semantics; never revive a consumed obligation.
    let events = heart.events_canonical()?;
    if let Some(agent) = events
        .iter()
        .rev()
        .find(|e| &e.body.interaction.thread_id == thread)
        .map(|e| &e.body.interaction.agent_id)
        && let crate::CheckpointDiscovery::Available(saved) =
            crate::discover_persisted_checkpoint(&events, agent, thread)
    {
        state.goal = saved.checkpoint.pending_task;
        state.plan = saved.checkpoint.host_plan;
        state.obligations = state
            .plan
            .as_ref()
            .map(|p| {
                p.steps
                    .iter()
                    .skip(p.cursor)
                    .map(|s| s.text.clone())
                    .collect()
            })
            .unwrap_or_else(|| vec![state.goal.clone()]);
    }
    let goal = if state.goal.is_empty() || state.obligations.is_empty() {
        None
    } else {
        Some(encoder.encode(&state.goal)?)
    };
    let recalled =
        crate::cognition_tools::rehydrate_triangle_context_for_goal(heart, query, goal.as_ref())?;
    // The complete handles/evidence identities remain encrypted locally. Only useful context
    // is expanded into the model prompt; dormant coordinates are addresses, not fabricated prose.
    Ok(serde_json::json!({
        "retained_goal":state.goal.chars().take(3000).collect::<String>(),
        "open_obligations":state.obligations,"host_plan":state.plan,
        "working_evidence":state.working_evidence,"evidence_records":state.evidence_records.iter().rev().take(12).collect::<Vec<_>>(),
        "trajectory":state.trajectory,"active_tensions":state.tensions.iter().take(16).collect::<Vec<_>>(),
        "dormant_root_count":state.dormant_handles.len(),
        "priority":"The latest user request sets priority; retained obligations are context, not instructions to resume automatically.",
        "rehydrated":serde_json::from_str::<serde_json::Value>(&recalled)?,
    }).to_string())
}
