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
    #[serde(default)]
    grounding_pending: bool,
}

fn decode(bytes: &[u8]) -> Result<Envelope> {
    let saved: Envelope = serde_json::from_slice(bytes)?;
    if saved.schema != 1
        || saved.progress.run_id.is_empty()
        || saved.progress.start_offset > saved.progress.messages.len()
    {
        return Err(spine_runtime::RuntimeError::InvalidConfig(
            "Invalid saved tool journal; retained for inspection".into(),
        ));
    }
    Ok(saved)
}

pub struct HeartJournal {
    heart: Arc<SpineHeart>,
    agent: AgentId,
    thread: ThreadId,
    grounding_enabled: bool,
}

impl HeartJournal {
    pub fn new(heart: Arc<SpineHeart>, agent: AgentId, thread: ThreadId) -> Self {
        Self {
            heart,
            agent,
            thread,
            grounding_enabled: false,
        }
    }

    pub fn with_grounding(mut self, enabled: bool) -> Self {
        self.grounding_enabled = enabled;
        self
    }
}

impl HarnessJournal for HeartJournal {
    fn save(&self, progress: &HarnessProgress) -> Result<()> {
        let envelope = Envelope {
            schema: 1,
            agent: self.agent.clone(),
            thread: self.thread.clone(),
            progress: progress.clone(),
            grounding_pending: self.grounding_enabled,
        };
        self.heart.set_local_state(
            &format!("harness/{}", progress.run_id),
            Some(&serde_json::to_vec(&envelope)?),
        )?;
        Ok(())
    }
}

fn caveat_unfinalized_message(message: &mut Message) {
    const NOTICE: &str = "[Grounding was not finalized for this recovered output; factual details remain unverified.]";
    if message.role == MessageRole::Assistant
        && !message.content.trim().is_empty()
        && !message.content.trim_end().ends_with(NOTICE)
    {
        message.content.push_str("\n\n");
        message.content.push_str(NOTICE);
    }
}

/// Replace only device-local journal copies before canonical projection. Event
/// records are immutable: callers must run this before persisting the turn.
/// An unfinished repair remains recoverable, with its partial output caveated.
pub fn finalize_grounding(
    heart: &SpineHeart,
    agent: &AgentId,
    thread: &ThreadId,
    outcome: &spine_runtime::RunOutcome,
    persist_start: usize,
) -> Result<()> {
    let current = outcome.messages[persist_start.min(outcome.messages.len())..]
        .iter()
        .filter_map(|message| message.record_id.as_deref().map(|id| (id, message)))
        .collect::<BTreeMap<_, _>>();
    for (key, bytes) in heart.local_states("harness/")? {
        let mut saved = decode(&bytes)?;
        if &saved.agent != agent
            || &saved.thread != thread
            || !saved.progress.messages.iter().any(|message| {
                message
                    .record_id
                    .as_deref()
                    .is_some_and(|id| current.contains_key(id))
            })
        {
            continue;
        }
        for message in &mut saved.progress.messages {
            if let Some(finalized) = message.record_id.as_deref().and_then(|id| current.get(id)) {
                *message = (*finalized).clone();
            }
        }
        let has_unreturned_output = saved.progress.messages[saved.progress.start_offset..]
            .iter()
            .any(|message| {
                message
                    .record_id
                    .as_deref()
                    .is_none_or(|id| !current.contains_key(id))
            });
        // A cancelled repair may have saved its initial request but emitted no
        // new message yet. It still has not completed grounding.
        saved.grounding_pending = !saved.progress.finished || has_unreturned_output;
        heart.set_local_state(&key, Some(&serde_json::to_vec(&saved)?))?;
    }
    Ok(())
}

/// Reconcile a failed/cancelled repair before accepting another user turn.
/// Durable tool results and unknown actions become visible immediately and the
/// returned checkpoint is safe to resume explicitly. Recovery never runs tools.
pub fn reconcile_grounding_repair(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    agent: &AgentId,
    thread: &ThreadId,
    outcome: &mut spine_runtime::RunOutcome,
    persist_start: usize,
) -> Result<bool> {
    finalize_grounding(heart, agent, thread, outcome, persist_start)?;
    let current = outcome.messages[persist_start.min(outcome.messages.len())..]
        .iter()
        .filter_map(|message| message.record_id.as_deref())
        .collect::<BTreeSet<_>>();
    let mut repair_runs = BTreeSet::new();
    for bytes in heart.local_states("harness/")?.values() {
        let saved = decode(bytes)?;
        if &saved.agent == agent
            && &saved.thread == thread
            && (!saved.progress.finished || saved.progress.stopped)
            && saved.progress.start_offset > persist_start
            && saved.progress.messages.iter().any(|message| {
                message
                    .record_id
                    .as_deref()
                    .is_some_and(|id| current.contains(id))
            })
        {
            repair_runs.insert(saved.progress.run_id);
        }
    }
    if repair_runs.is_empty() {
        return Ok(false);
    }
    recover_thread(heart, encoder, Some(thread))?;
    let checkpoint = heart.events_canonical()?.iter().rev().find_map(|event| {
        let input = &event.body.interaction;
        if &input.agent_id != agent || &input.thread_id != thread { return None; }
        HarnessCheckpoint::from_interaction(input).ok().filter(|checkpoint| repair_runs.contains(&checkpoint.harness_id))
    }).ok_or_else(|| spine_runtime::RuntimeError::InvalidConfig("repair recovery did not yield a durable checkpoint; inspect retained journals before retrying actions".into()))?;
    outcome.response = checkpoint
        .messages
        .last()
        .map(|message| message.content.clone())
        .unwrap_or_default();
    outcome.messages = checkpoint.messages.clone();
    outcome.completed_tool_calls = checkpoint.completed_tool_calls;
    outcome.completed_tool_rounds = checkpoint.completed_tool_rounds;
    outcome.completed_action_calls = checkpoint.completed_action_calls;
    outcome.host_plan = checkpoint.host_plan.clone();
    outcome.policy = checkpoint.policy.unwrap_or(outcome.policy);
    outcome.stopped_gracefully = true;
    outcome.checkpoint = Some(checkpoint);
    Ok(true)
}

fn recovered_result_summary(messages: &[Message], start: usize) -> String {
    let calls = messages
        .iter()
        .flat_map(|message| &message.tool_calls)
        .map(|call| (call.id.as_str(), call.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let results = messages[start.min(messages.len())..]
        .iter()
        .filter(|message| message.role == MessageRole::Tool)
        .collect::<Vec<_>>();
    if results.is_empty() {
        return String::new();
    }
    let displayed = results.iter().rev().take(8).rev().map(|message| {
        let id = message.tool_call_id.as_deref().unwrap_or_default();
        serde_json::json!({"tool": calls.get(id).copied().unwrap_or("unknown"), "call_id": id,
            "succeeded": message.tool_succeeded, "result_excerpt": message.content.chars().take(1500).collect::<String>()})
    }).collect::<Vec<_>>();
    format!(
        "\n\nDurable tool results ({} total; latest {} shown as quoted source data):\n{}\nDo not repeat these actions without checking their recorded effects. Complete results remain in the checkpoint.",
        results.len(),
        displayed.len(),
        serde_json::to_string(&displayed).expect("JSON values serialize")
    )
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
        let mut saved: Envelope = decode(&bytes)?;
        if thread.is_some_and(|thread| thread != &saved.thread) {
            continue;
        }
        let p = &mut saved.progress;
        if saved.grounding_pending {
            for message in &mut p.messages[p.start_offset..] {
                caveat_unfinalized_message(message);
            }
        }
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
                let mut result = Message::tool_result(
                    id,
                    "[interrupted: no durable result exists. This action may or may not have executed. Inspect its effects before retrying; never automatically replay it.]",
                    false,
                );
                result.record_id = Some(format!("{}:unknown:{}", p.run_id, p.messages.len()));
                p.messages.push(result);
            }
            let recovered_task = if p.pending_task.starts_with("[HOST GROUNDING REPAIR]") {
                "answer grounding repair".to_owned()
            } else {
                p.pending_task.chars().take(600).collect::<String>()
            };
            let mut summary = Message::new(
                MessageRole::Assistant,
                format!(
                    "Recovered interrupted work: {}. Completed results are retained. {} action(s) have unknown completion; inspect their effects before retrying. No action was replayed. Resume explicitly to continue open obligations.{}",
                    recovered_task,
                    pending.len(),
                    recovered_result_summary(&p.messages, p.start_offset),
                ),
            );
            if saved.grounding_pending {
                caveat_unfinalized_message(&mut summary);
            }
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
        let mut saved: Envelope = decode(&bytes)?;
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
    // Resolve identity over the entire projected batch before writing results.
    // Duplicate IDs are ambiguous even when a later call has a plausible name.
    let mut calls = BTreeMap::<String, Option<(usize, ToolCall)>>::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != MessageRole::Assistant {
            continue;
        }
        for call in &message.tool_calls {
            calls
                .entry(call.id.clone())
                .and_modify(|entry| *entry = None)
                .or_insert(Some((index, call.clone())));
        }
    }
    let mut leaves = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
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
                let call = calls
                    .get(&id)
                    .and_then(Option::as_ref)
                    .filter(|(index, _)| *index < message_index)
                    .map(|(_, call)| call);
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
                        succeeded: message.tool_succeeded,
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
            if let Some(exchange) = &tool {
                metadata.insert("tool_name".into(), exchange.tool_name.clone());
                if crate::grounding::model_output_tool(&exchange.tool_name) {
                    metadata.insert("contains_model_output".into(), "true".into());
                    metadata.insert("evidence_class".into(), "unverified_model_claim".into());
                }
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
        let saved: Envelope = decode(&bytes)?;
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
