use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use async_trait::async_trait;
use spine_heart::{
    AgentId, Content, Embedding, EventKind, FactAggregation, FactValue, InteractionInput,
    ParticipantRole, Provenance, RehydrateBudget, SemanticEncoder, SpineHeart, ThreadId,
};
use spine_runtime::{
    Tool, ToolCall, ToolCategory, ToolContext, ToolRegistry, ToolResult, ToolRisk, ToolSpec,
};

pub fn register_cognition_tools<E: SemanticEncoder + 'static>(
    registry: &mut ToolRegistry,
    heart: Arc<SpineHeart>,
    encoder: Arc<E>,
    allow_model_memory_writes: bool,
) -> spine_runtime::Result<()> {
    let encoder: Arc<dyn SemanticEncoder> = encoder;
    for name in ["heart_stats", "memory_stats"] {
        registry.register(MemoryStatsTool {
            name,
            heart: Arc::clone(&heart),
        })?;
    }
    for name in ["heart_recall", "search_memory"] {
        registry.register(MemoryRecallTool {
            name,
            heart: Arc::clone(&heart),
            encoder: Arc::clone(&encoder),
        })?;
    }
    registry.register(FeelTool {
        heart: Arc::clone(&heart),
        encoder: Arc::clone(&encoder),
    })?;
    if allow_model_memory_writes {
        registry.register(SaveMemoryTool {
            heart: Arc::clone(&heart),
            encoder,
        })?;
    }
    registry.register(FactSearchTool {
        heart: Arc::clone(&heart),
    })?;
    registry.register(FactAggregateTool {
        heart: Arc::clone(&heart),
    })?;
    registry.register(MaintainMemoryTool { heart })?;
    Ok(())
}

struct SaveMemoryTool {
    heart: Arc<SpineHeart>,
    encoder: Arc<dyn SemanticEncoder>,
}

#[async_trait]
impl Tool for SaveMemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "save_memory".into(),
            description: "Store an explicitly enabled, unverified model-authored memory record."
                .into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: object_schema(serde_json::json!({"text": {"type": "string"}}), &["text"]),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(value) = call
            .arguments
            .get("text")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(ToolResult::failure("save_memory requires non-empty text"));
        };
        let mut metadata = BTreeMap::new();
        metadata.insert("record_type".into(), "model-requested-memory".into());
        metadata.insert("evidence_class".into(), "unverified_model_claim".into());
        metadata.insert("contains_model_output".into(), "true".into());
        let interaction = InteractionInput {
            agent_id: context.agent_id.clone().unwrap_or(AgentId::new("main")?),
            thread_id: ThreadId::new("model-memory")?,
            role: ParticipantRole::Assistant,
            kind: EventKind::Message,
            content: Content::Inline(value.to_owned()),
            causal_parents: Vec::new(),
            provenance: Provenance {
                provider: Some("model-requested-save".into()),
                metadata,
                ..Provenance::default()
            },
            tool: None,
            attachments: Vec::new(),
            outcome: Some("unverified".into()),
        };
        let (commit, memory) = self
            .heart
            .commit_embedded(interaction, self.encoder.encode(value)?)?;
        Ok(ToolResult::success(format!(
            "Saved unverified model memory event={} node={}",
            commit.event.id, memory.node_id
        )))
    }
}

pub fn recall_context(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    query: &str,
    top_k: usize,
) -> spine_runtime::Result<String> {
    hybrid_recall_json(heart, encoder, query, top_k.clamp(1, 16))
}

pub fn rehydrate_triangle_context(
    heart: &SpineHeart,
    query: &Embedding,
) -> spine_runtime::Result<String> {
    let Some(state) = heart.cognition()? else {
        return Ok("[]".into());
    };
    let canonical = heart.events_canonical()?;
    let events: BTreeMap<_, _> = canonical
        .into_iter()
        .map(|event| (event.id, event))
        .collect();
    let mut seen_nodes = BTreeSet::new();
    let mut result = Vec::new();
    for root in state.triangles.roots.iter().rev().take(3) {
        let rehydrated = heart.rehydrate_context(
            root.handle,
            Some(query),
            RehydrateBudget {
                max_depth: 2,
                max_fanout: 3,
                max_nodes: 8,
                max_tokens: 1_024,
            },
        )?;
        for coordinate in rehydrated.coordinates {
            if !seen_nodes.insert(coordinate.node_id) {
                continue;
            }
            let Some(node) = state.dcmdb.node(coordinate.node_id) else {
                continue;
            };
            for event_id in node.event_ids.iter().rev().take(2) {
                let Some(event) = events.get(event_id) else {
                    continue;
                };
                if let Content::Inline(text) = &event.body.interaction.content {
                    result.push(serde_json::json!({
                        "node_id": coordinate.node_id.to_string(),
                        "role": format!("{:?}", coordinate.role),
                        "depth": coordinate.depth,
                        "event_id": event.id.to_string(),
                        "text": bounded_text(text, 1_000),
                    }));
                }
                if result.len() >= 6 {
                    return Ok(serde_json::to_string(&result)?);
                }
            }
        }
    }
    Ok(serde_json::to_string(&result)?)
}

struct MemoryStatsTool {
    name: &'static str,
    heart: Arc<SpineHeart>,
}

#[async_trait]
impl Tool for MemoryStatsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.into(),
            description: "Return exact encrypted-heart and cognitive-memory statistics.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(serde_json::json!({}), &[]),
        }
    }

    async fn execute(
        &self,
        _call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let store = self.heart.stats()?;
        let cognition = self.heart.cognition()?;
        let cognitive = cognition.map(|state| {
            let confidences: Vec<_> = state.dcmdb.nodes.values().map(|node| node.confidence).collect();
            let mean_confidence = if confidences.is_empty() {
                0.0
            } else {
                confidences.iter().sum::<f32>() / confidences.len() as f32
            };
            let hierarchy_depth = state
                .dcmdb
                .nodes
                .values()
                .map(|node| node.level)
                .max()
                .unwrap_or_default();
            serde_json::json!({
                "current": state.is_current(&self.heart.sync_frontier().map(|f| f.devices).unwrap_or_default()),
                "active_nodes": state.dcmdb.nodes.len(),
                "absorbed_nodes": state.dcmdb.absorbed.len(),
                "hierarchy_depth": hierarchy_depth,
                "mean_confidence": mean_confidence,
                "fact_count": state.facts.facts().count(),
                "active_fact_count": state.facts.active().count(),
                "agents_with_thymos": state.thymos.len(),
                "triangle_roots": state.triangles.roots.len(),
                "projected_events": state.projected_events,
            })
        });
        Ok(ToolResult::success(
            serde_json::json!({
                "events": store.events,
                "blobs": store.blobs,
                "snapshots": store.snapshots,
                "tombstones": store.tombstones,
                "cognition": cognitive,
            })
            .to_string(),
        ))
    }
}

struct MemoryRecallTool {
    name: &'static str,
    heart: Arc<SpineHeart>,
    encoder: Arc<dyn SemanticEncoder>,
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.into(),
            description:
                "Hybrid semantic and exact-text recall over canonical encrypted-heart memories."
                    .into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(
                serde_json::json!({
                    "query": {"type": "string", "description": "Memory query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 16}
                }),
                &["query"],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(query) = call.arguments.get("query").and_then(|value| value.as_str()) else {
            return Ok(ToolResult::failure(format!(
                "{} requires a string query",
                self.name
            )));
        };
        let top_k = call
            .arguments
            .get("top_k")
            .and_then(|value| value.as_u64())
            .unwrap_or(5)
            .clamp(1, 16) as usize;
        Ok(ToolResult::success(hybrid_recall_json(
            &self.heart,
            self.encoder.as_ref(),
            query,
            top_k,
        )?))
    }
}

fn hybrid_recall_json(
    heart: &SpineHeart,
    encoder: &dyn SemanticEncoder,
    query: &str,
    top_k: usize,
) -> spine_runtime::Result<String> {
    let embedding = encoder.encode(query)?;
    let dense = heart.recall_memories(&embedding, f64::MAX, top_k.saturating_mul(2), 6)?;

    let mut results = Vec::new();
    let mut seen = BTreeSet::new();
    for (score, event) in lexical_events(heart, query, top_k)? {
        if let Content::Inline(text) = event.body.interaction.content
            && seen.insert(event.id)
        {
            results.push(serde_json::json!({
                "retrieval": "lexical",
                "score": score,
                "event_id": event.id.to_string(),
                "role": format!("{:?}", event.body.interaction.role),
                "source": event.body.interaction.provenance.source_uri,
                "text": bounded_text(&text, 3_000),
            }));
        }
    }
    for memory in dense {
        for event in memory.events {
            if results.len() >= top_k.saturating_mul(3) {
                break;
            }
            if let Content::Inline(text) = event.body.interaction.content
                && seen.insert(event.id)
            {
                results.push(serde_json::json!({
                    "retrieval": "semantic",
                    "score": memory.hit.score,
                    "semantic_score": memory.hit.semantic_score,
                    "node_id": memory.hit.node_id.to_string(),
                    "event_id": event.id.to_string(),
                    "role": format!("{:?}", event.body.interaction.role),
                    "source": event.body.interaction.provenance.source_uri,
                    "text": bounded_text(&text, 3_000),
                }));
            }
        }
    }
    results.truncate(top_k.saturating_mul(2));
    Ok(serde_json::to_string(&results)?)
}

fn lexical_events(
    heart: &SpineHeart,
    query: &str,
    top_k: usize,
) -> spine_runtime::Result<Vec<(f32, spine_heart::SignedEvent)>> {
    let query_normalized = query.to_lowercase();
    let query_terms: BTreeSet<_> = terms(query).into_iter().collect();
    let mut scored = Vec::new();
    for event in heart.events_canonical()? {
        let Content::Inline(text) = &event.body.interaction.content else {
            continue;
        };
        let lower = text.to_lowercase();
        let document_terms = terms(text);
        if document_terms.is_empty() {
            continue;
        }
        let matches = document_terms
            .iter()
            .filter(|term| query_terms.contains(*term))
            .count() as f32;
        let unique_matches = document_terms
            .iter()
            .filter(|term| query_terms.contains(*term))
            .collect::<BTreeSet<_>>()
            .len() as f32;
        let phrase_bonus = if !query_normalized.is_empty() && lower.contains(&query_normalized) {
            8.0
        } else {
            0.0
        };
        let score = phrase_bonus
            + unique_matches * 2.0
            + matches / (document_terms.len() as f32).sqrt().max(1.0);
        if score > 0.0 {
            scored.push((score, event));
        }
    }
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.1.body.timestamp.cmp(&left.1.body.timestamp))
    });
    scored.truncate(top_k);
    Ok(scored)
}

struct FeelTool {
    heart: Arc<SpineHeart>,
    encoder: Arc<dyn SemanticEncoder>,
}

#[async_trait]
impl Tool for FeelTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "feel".into(),
            description: "Introspect the current agent's Thymos feeling vector for a context."
                .into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(
                serde_json::json!({"context": {"type": "string", "description": "Context to feel; defaults to the active task"}}),
                &[],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let text = call
            .arguments
            .get("context")
            .and_then(|value| value.as_str())
            .or_else(|| context.metadata.get("task").map(String::as_str))
            .unwrap_or("current context");
        let agent = context.agent_id.clone().unwrap_or(AgentId::new("main")?);
        let feeling = self.heart.feel(&agent, &self.encoder.encode(text)?)?;
        Ok(ToolResult::success(match feeling {
            Some(feeling) => serde_json::to_string(&feeling)?,
            None => serde_json::json!({"available": false, "reason": "agent has no Thymos observations yet"}).to_string(),
        }))
    }
}

struct FactSearchTool {
    heart: Arc<SpineHeart>,
}

#[async_trait]
impl Tool for FactSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fact_search".into(),
            description:
                "Search extracted facts, respecting newer facts that supersede older state.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(
                serde_json::json!({
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "include_superseded": {"type": "boolean"}
                }),
                &["query"],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(query) = call.arguments.get("query").and_then(|value| value.as_str()) else {
            return Ok(ToolResult::failure("fact_search requires a string query"));
        };
        let top_k = call
            .arguments
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(10)
            .clamp(1, 50) as usize;
        let include = call
            .arguments
            .get("include_superseded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let hits: Vec<_> = self
            .heart
            .search_facts(query, top_k, include)?
            .into_iter()
            .map(|hit| {
                serde_json::json!({
                    "score": hit.score,
                    "entity": hit.fact.entity,
                    "attribute": hit.fact.attribute,
                    "value": fact_value(hit.fact.value),
                    "slot": hit.fact.slot_key,
                    "excerpt": hit.fact.excerpt,
                    "event_time": hit.fact.event_time,
                    "superseded": hit.fact.superseded_by.is_some(),
                    "confidence": hit.fact.confidence,
                })
            })
            .collect();
        Ok(ToolResult::success(serde_json::to_string(&hits)?))
    }
}

struct FactAggregateTool {
    heart: Arc<SpineHeart>,
}

#[async_trait]
impl Tool for FactAggregateTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fact_aggregate".into(),
            description: "Sum, count, or select the latest fact for a slot prefix.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(
                serde_json::json!({
                    "slot_prefix": {"type": "string"},
                    "operation": {"type": "string", "enum": ["sum", "count", "latest"]}
                }),
                &["slot_prefix", "operation"],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let Some(prefix) = call.arguments.get("slot_prefix").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure("fact_aggregate requires slot_prefix"));
        };
        let Some(operation) = call.arguments.get("operation").and_then(|v| v.as_str()) else {
            return Ok(ToolResult::failure("fact_aggregate requires operation"));
        };
        let value = match self.heart.aggregate_facts(prefix, operation)? {
            FactAggregation::Sum(value) => serde_json::json!({"operation": "sum", "value": value}),
            FactAggregation::Count(value) => {
                serde_json::json!({"operation": "count", "value": value})
            }
            FactAggregation::Latest(fact) => match fact {
                Some(fact) => serde_json::json!({
                    "operation": "latest",
                    "entity": fact.entity,
                    "attribute": fact.attribute,
                    "value": fact_value(fact.value),
                    "slot": fact.slot_key,
                    "excerpt": fact.excerpt,
                    "event_time": fact.event_time,
                }),
                None => serde_json::json!({"operation": "latest", "value": null}),
            },
        };
        Ok(ToolResult::success(value.to_string()))
    }
}

struct MaintainMemoryTool {
    heart: Arc<SpineHeart>,
}

#[async_trait]
impl Tool for MaintainMemoryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "maintain_memory".into(),
            description: "Run bounded DCMDb consolidation, pruning, and dream maintenance.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::Mutating,
            parameters: object_schema(
                serde_json::json!({"maximum_rounds": {"type": "integer", "minimum": 0, "maximum": 32}}),
                &[],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let rounds = call
            .arguments
            .get("maximum_rounds")
            .and_then(|v| v.as_u64())
            .unwrap_or(4)
            .min(32) as usize;
        let report = self.heart.maintain_cognition(rounds)?;
        Ok(ToolResult::success(
            serde_json::json!({
                "merges": report.merges,
                "pruned": report.pruned,
                "walks_completed": report.walks_completed,
                "nodes_reactivated": report.nodes_reactivated,
            })
            .to_string(),
        ))
    }
}

fn object_schema(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": false,
    });
    if !required.is_empty() {
        schema["required"] = serde_json::json!(required);
    }
    schema
}

fn fact_value(value: FactValue) -> serde_json::Value {
    match value {
        FactValue::Text(value) => serde_json::Value::String(value),
        FactValue::Integer(value) => serde_json::json!(value),
        FactValue::Amount(value) => serde_json::json!(value),
    }
}

fn terms(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.len() > 1)
        .map(str::to_lowercase)
        .collect()
}

fn bounded_text(text: &str, maximum_chars: usize) -> String {
    if text.chars().count() <= maximum_chars {
        text.to_owned()
    } else {
        text.chars().take(maximum_chars).collect::<String>() + "\n[truncated]"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_are_case_insensitive_and_punctuation_safe() {
        assert_eq!(terms("Hello, HELLO-world!"), ["hello", "hello", "world"]);
    }

    #[test]
    fn bounded_text_respects_unicode_characters() {
        assert_eq!(bounded_text("a🦀bc", 2), "a🦀\n[truncated]");
    }
}
