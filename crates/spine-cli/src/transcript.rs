//! Durable device-local tool journals and idempotent canonical transcript projection.
use serde::{Deserialize, Serialize};
use spine_heart::{
    AgentId, Content, ContextLeaf, EventKind, InteractionInput, ParticipantRole, Provenance,
    SemanticEncoder, SpineHeart, ThreadId, ToolExchange,
};
use spine_runtime::{
    HarnessCheckpoint, HarnessJournal, HarnessProgress, Message, MessageRole, ToolCall,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

type Result<T> = spine_runtime::Result<T>;

#[derive(Serialize, Deserialize)]
struct Envelope {
    schema: u32,
    agent: AgentId,
    thread: ThreadId,
    progress: HarnessProgress,
}

pub struct HeartJournal {
    heart: Arc<SpineHeart>,
    agent: AgentId,
    thread: ThreadId,
}

impl HeartJournal {
    pub fn new(heart: Arc<SpineHeart>, agent: AgentId, thread: ThreadId) -> Self {
        Self {
            heart,
            agent,
            thread,
        }
    }
}

impl HarnessJournal for HeartJournal {
    fn save(&self, progress: &HarnessProgress) -> Result<()> {
        let envelope = Envelope {
            schema: 1,
            agent: self.agent.clone(),
            thread: self.thread.clone(),
            progress: progress.clone(),
        };
        self.heart.set_local_state(
            &format!("harness/{}", progress.run_id),
            Some(&serde_json::to_vec(&envelope)?),
        )?;
        Ok(())
    }
}

/// Call before replaying chat history. Unknown actions are disclosed, never dispatched.
pub fn recover(heart: &SpineHeart, encoder: &dyn SemanticEncoder) -> Result<usize> {
    recover_thread(heart, encoder, None)
}

pub fn recover_thread(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    thread: Option<&ThreadId>,
) -> Result<usize> {
    let records = heart.local_states("harness/")?;
    let mut recovered = 0;
    for (key, bytes) in records {
        let mut saved: Envelope = serde_json::from_slice(&bytes)?;
        if thread.is_some_and(|thread| thread != &saved.thread) {
            continue;
        }
        if saved.schema != 1 || saved.progress.start_offset > saved.progress.messages.len() {
            return Err(spine_runtime::RuntimeError::InvalidConfig(
                "Invalid saved tool journal; retained for inspection".into(),
            ));
        }
        let p = &mut saved.progress;
        if !p.finished {
            let mut pending = BTreeMap::new();
            for message in &p.messages {
                for call in &message.tool_calls {
                    pending.insert(call.id.clone(), call.clone());
                }
                if let Some(id) = &message.tool_call_id {
                    pending.remove(id);
                }
            }
            for id in pending.keys() {
                let mut result = Message::tool(
                    id,
                    "[interrupted: no durable result exists. This action may or may not have executed. Inspect its effects before retrying; never automatically replay it.]",
                );
                result.record_id = Some(format!("{}:unknown:{}", p.run_id, p.messages.len()));
                p.messages.push(result);
            }
            let mut summary = Message::new(
                MessageRole::Assistant,
                format!(
                    "Recovered interrupted work: {}. Completed results are retained. {} action(s) have unknown completion; inspect their effects before retrying. No action was replayed. Resume explicitly to continue open obligations.",
                    p.pending_task,
                    pending.len()
                ),
            );
            summary.record_id = Some(format!("{}:recovery", p.run_id));
            p.messages.push(summary);
        }
        persist(
            heart,
            encoder,
            &saved.agent,
            &saved.thread,
            &p.messages[p.start_offset..],
        )?;
        let checkpoint_saved = heart.events_canonical()?.iter().any(|event| {
            spine_runtime::HarnessCheckpoint::from_interaction(&event.body.interaction)
                .ok()
                .is_some_and(|checkpoint| {
                    checkpoint.messages.iter().any(|m| {
                        m.record_id
                            .as_ref()
                            .is_some_and(|id| id.starts_with(&format!("{}:", p.run_id)))
                    })
                })
        });
        if (!p.finished || p.stopped) && !checkpoint_saved {
            let checkpoint = HarnessCheckpoint {
                schema: 1,
                harness_id: p.run_id.clone(),
                messages: p.messages.clone(),
                completed_tool_calls: p.completed_tool_calls,
                completed_tool_rounds: p.completed_tool_rounds,
                completed_action_calls: p.completed_action_calls,
                pending_task: p.pending_task.clone(),
                host_plan: p.host_plan.clone(),
                policy: Some(p.policy),
            };
            let input = checkpoint.to_interaction(saved.agent, saved.thread)?;
            let content = match &input.content {
                Content::Inline(text) => text,
                _ => unreachable!(),
            };
            heart.commit_embedded(input.clone(), encoder.encode(content)?)?;
            recovered += 1;
        }
        heart.set_local_state(&key, None)?;
    }
    Ok(recovered)
}

/// A child explicitly reported final; its safe-boundary stop is successful completion.
pub fn finish_child_checkin(heart: &SpineHeart, thread: &ThreadId) -> Result<()> {
    for (key, bytes) in heart.local_states("harness/")? {
        let mut saved: Envelope = serde_json::from_slice(&bytes)?;
        if &saved.thread == thread && saved.progress.finished {
            saved.progress.stopped = false;
            heart.set_local_state(&key, Some(&serde_json::to_vec(&saved)?))?;
        }
    }
    Ok(())
}

pub fn persist(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent: &AgentId,
    thread: &ThreadId,
    messages: &[Message],
) -> Result<Vec<ContextLeaf>> {
    let mut recorded = heart
        .events_canonical()?
        .into_iter()
        .filter_map(|event| {
            event
                .body
                .interaction
                .provenance
                .metadata
                .get("harness_record")
                .cloned()
        })
        .collect::<BTreeSet<_>>();
    let mut calls = BTreeMap::<String, ToolCall>::new();
    let mut leaves = Vec::new();
    for message in messages {
        let mut items = Vec::<(
            String,
            ParticipantRole,
            EventKind,
            String,
            Option<ToolExchange>,
        )>::new();
        match message.role {
            MessageRole::System | MessageRole::User => continue,
            MessageRole::Assistant => {
                if let Some(reasoning) = message.reasoning.as_ref().filter(|r| !r.trim().is_empty())
                {
                    items.push((
                        "thought".into(),
                        ParticipantRole::Assistant,
                        EventKind::Control,
                        reasoning.clone(),
                        None,
                    ));
                }
                if !message.content.trim().is_empty() {
                    items.push((
                        "answer".into(),
                        ParticipantRole::Assistant,
                        EventKind::Message,
                        message.content.clone(),
                        None,
                    ));
                }
                for (index, call) in message.tool_calls.iter().enumerate() {
                    calls.insert(call.id.clone(), call.clone());
                    let args = call.arguments.to_string();
                    items.push((
                        format!("call{index}"),
                        ParticipantRole::Assistant,
                        EventKind::ToolCall,
                        format!("{}({args})", call.name),
                        Some(ToolExchange {
                            operation_id: call.id.clone(),
                            tool_name: call.name.clone(),
                            arguments: Content::Inline(args),
                            result: None,
                            succeeded: None,
                            background: false,
                        }),
                    ));
                }
            }
            MessageRole::Tool => {
                let id = message.tool_call_id.clone().unwrap_or_default();
                let call = calls.get(&id);
                items.push((
                    "result".into(),
                    ParticipantRole::Tool,
                    EventKind::ToolResult,
                    message.content.clone(),
                    Some(ToolExchange {
                        operation_id: id,
                        tool_name: call.map_or_else(|| "unknown".into(), |c| c.name.clone()),
                        arguments: Content::Inline(
                            call.map_or_else(|| "{}".into(), |c| c.arguments.to_string()),
                        ),
                        result: Some(Content::Inline(message.content.clone())),
                        succeeded: None,
                        background: false,
                    }),
                ));
            }
        }
        for (part, role, kind, text, tool) in items {
            let identity = message.record_id.as_ref().map(|id| format!("{id}/{part}"));
            if identity.as_ref().is_some_and(|id| recorded.contains(id)) {
                continue;
            }
            let mut metadata = BTreeMap::new();
            if part == "thought" {
                metadata.insert("model_reasoning".into(), "true".into());
            }
            if let Some(id) = &identity {
                metadata.insert("harness_record".into(), id.clone());
            }
            let embedded = encoder.encode(&text)?;
            let (commit, memory) = heart.commit_embedded(
                InteractionInput {
                    agent_id: agent.clone(),
                    thread_id: thread.clone(),
                    role,
                    kind,
                    content: Content::Inline(text),
                    causal_parents: Vec::new(),
                    provenance: Provenance {
                        provider: Some("llama.cpp".into()),
                        metadata,
                        ..Provenance::default()
                    },
                    tool,
                    attachments: Vec::new(),
                    outcome: None,
                },
                embedded,
            )?;
            if let Some(id) = identity {
                recorded.insert(id);
            }
            if part != "thought" {
                leaves.push(ContextLeaf {
                    node_id: memory.node_id,
                    chronology: commit.event.body.device_sequence,
                });
            }
        }
    }
    // Completed ordinary runs can be dropped only after every display/result record is durable.
    for (key, bytes) in heart.local_states("harness/")? {
        let saved: Envelope = serde_json::from_slice(&bytes)?;
        if !saved.progress.finished || saved.progress.stopped {
            continue;
        }
        let required = saved.progress.messages[saved.progress.start_offset..]
            .iter()
            .flat_map(|m| {
                let mut ids = Vec::new();
                if let Some(id) = &m.record_id {
                    if m.role == MessageRole::Assistant {
                        if m.reasoning.as_ref().is_some_and(|r| !r.trim().is_empty()) {
                            ids.push(format!("{id}/thought"));
                        }
                        if !m.content.trim().is_empty() {
                            ids.push(format!("{id}/answer"));
                        }
                        for i in 0..m.tool_calls.len() {
                            ids.push(format!("{id}/call{i}"));
                        }
                    } else if m.role == MessageRole::Tool {
                        ids.push(format!("{id}/result"));
                    }
                }
                ids
            })
            .collect::<Vec<_>>();
        if required.iter().all(|id| recorded.contains(id)) {
            heart.set_local_state(&key, None)?;
        }
    }
    Ok(leaves)
}
