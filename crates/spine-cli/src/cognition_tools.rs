use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use async_trait::async_trait;
use spine_heart::{
    AgentId, Content, Embedding, EventId, EventKind, Fact, FactAggregation, FactQueryAggregation,
    FactSlotType, FactValue, InteractionInput, ParticipantRole, Provenance, RecallHit,
    RehydrateBudget, SemanticEncoder, SpineHeart, ThreadId,
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

#[derive(Clone, Debug)]
pub struct AutomaticRecall {
    pub context: String,
    pub hits: Vec<RecallHit>,
    pub hierarchy_depth: u32,
    pub total_nodes: usize,
    pub retrieval_stat_dimensions: usize,
}

impl AutomaticRecall {
    pub fn empty_for_layout(retrieval_stat_dimensions: usize) -> Self {
        Self {
            context: "[]".into(),
            hits: Vec::new(),
            hierarchy_depth: 0,
            total_nodes: 0,
            retrieval_stat_dimensions,
        }
    }

    pub fn tension_count(&self) -> usize {
        self.hits.iter().filter(|hit| hit.tensioned).count()
    }

    /// Python's six-feature oracle plus a released legacy segment when needed.
    pub fn risk_stats(&self) -> Vec<f32> {
        let oracle = self.oracle_risk_stats();
        if self.retrieval_stat_dimensions == 10 {
            let count = serde_json::from_str::<Vec<serde_json::Value>>(&self.context)
                .map_or(0, |events| events.len());
            let mut features = oracle.to_vec();
            features.extend([
                (count as f32 / 10.0).min(1.0),
                if count == 0 { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ]);
            features
        } else {
            oracle.to_vec()
        }
    }

    fn oracle_risk_stats(&self) -> [f32; 6] {
        if self.hits.is_empty() {
            return [0.0; 6];
        }
        let count = self.hits.len() as f32;
        let top_score = self
            .hits
            .iter()
            .map(|hit| hit.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_score = self.hits.iter().map(|hit| hit.score).sum::<f32>() / count;
        let mean_confidence = self.hits.iter().map(|hit| hit.confidence).sum::<f32>() / count;
        let tension_fraction = self.tension_count() as f32 / count;
        [
            top_score,
            top_score - mean_score,
            mean_confidence,
            tension_fraction,
            self.hierarchy_depth as f32 / 5.0,
            ((self.total_nodes as f64 + 1.0).ln() / 10.0) as f32,
        ]
    }
}

/// Select the reflection policy before committing an observation. Keeping this
/// value in event provenance makes the policy stable across replay and sync.
pub fn reflection_multiplier(
    heart: &SpineHeart,
    agent: &AgentId,
    embedding: &Embedding,
) -> spine_runtime::Result<f32> {
    let Some(mut thymos) = heart
        .cognition()?
        .and_then(|state| state.thymos.get(agent).cloned())
    else {
        return Ok(1.0);
    };
    let surprise = thymos.step(embedding.as_slice())?.surprise;
    let policy =
        spine_runtime::ModulationConfig::default().compute(spine_runtime::ModulationInput {
            surprise,
            valence: 0.0,
            arousal: 0.0,
            risk: 0.0,
            tensions: 0,
            base_temperature: 0.7,
            configured_tool_rounds: None,
        });
    Ok(policy.reflect_eta_multiplier)
}

/// Recall context for a just-committed user turn without recalling that turn as
/// its own evidence. The returned node hits drive host modulation directly.
pub fn automatic_recall_context(
    heart: &SpineHeart,
    query: &Embedding,
    lexical_query: &str,
    top_k: usize,
    excluded_event: EventId,
    expansion_depth: usize,
) -> spine_runtime::Result<AutomaticRecall> {
    hybrid_recall(
        heart,
        query,
        lexical_query,
        top_k.clamp(1, 16),
        Some(excluded_event),
        Some(expansion_depth.min(3)),
    )
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
            let mut level_distribution = BTreeMap::<u32, usize>::new();
            let mut source_counts = BTreeMap::<String, f32>::new();
            for node in state.dcmdb.nodes.values() {
                *level_distribution.entry(node.level).or_default() += 1;
                for (source, count) in &node.source_counts {
                    *source_counts.entry(source.clone()).or_default() += count;
                }
            }
            serde_json::json!({
                "current": state.is_current(&self.heart.sync_frontier().map(|f| f.devices).unwrap_or_default()),
                "active_nodes": state.dcmdb.nodes.len(),
                "absorbed_nodes": state.dcmdb.absorbed.len(),
                "hierarchy_depth": hierarchy_depth,
                "mean_confidence": mean_confidence,
                "min_confidence": confidences.iter().copied().reduce(f32::min),
                "max_confidence": confidences.iter().copied().reduce(f32::max),
                "level_distribution": level_distribution,
                "source_counts": source_counts,
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
    Ok(hybrid_recall(heart, &embedding, query, top_k, None, None)?.context)
}

fn hybrid_recall(
    heart: &SpineHeart,
    embedding: &Embedding,
    query: &str,
    top_k: usize,
    excluded_event: Option<EventId>,
    expansion_depth: Option<usize>,
) -> spine_runtime::Result<AutomaticRecall> {
    let dense_limit = top_k
        .saturating_mul(2)
        .saturating_add(usize::from(excluded_event.is_some()));
    let max_events_per_node = 6 + usize::from(excluded_event.is_some());
    let dense = heart.recall_memories_with_expansion(
        embedding,
        f64::MAX,
        dense_limit,
        max_events_per_node,
        expansion_depth,
    )?;
    let mut dense = dense
        .into_iter()
        .filter_map(|mut memory| {
            memory
                .events
                .retain(|event| excluded_event.is_none_or(|excluded| event.id != excluded));
            (!memory.events.is_empty()).then_some(memory)
        })
        .collect::<Vec<_>>();
    let hits = dense
        .iter()
        .take(top_k)
        .map(|memory| memory.hit.clone())
        .collect();

    let mut results = Vec::new();
    let mut seen = BTreeSet::new();
    for (score, event) in lexical_events(heart, query, top_k, excluded_event.as_ref())? {
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
    for memory in dense.drain(..) {
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
    let cognition = heart
        .cognition()?
        .ok_or(spine_heart::HeartError::NotFound)?;
    Ok(AutomaticRecall {
        context: serde_json::to_string(&results)?,
        hits,
        hierarchy_depth: cognition
            .dcmdb
            .nodes
            .values()
            .map(|node| node.level)
            .max()
            .unwrap_or(0),
        total_nodes: cognition.dcmdb.nodes.len(),
        ..AutomaticRecall::empty_for_layout(cognition.config.retrieval_stat_dimensions)
    })
}

fn lexical_events(
    heart: &SpineHeart,
    query: &str,
    top_k: usize,
    excluded_event: Option<&EventId>,
) -> spine_runtime::Result<Vec<(f32, spine_heart::SignedEvent)>> {
    let query_normalized = query.to_lowercase();
    let query_terms: BTreeSet<_> = terms(query).into_iter().collect();
    let mut scored = Vec::new();
    for event in heart.events_canonical()? {
        if excluded_event.is_some_and(|excluded| event.id == *excluded) {
            continue;
        }
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
            description: "Introspect the current agent's feeling vector, grid state, and host-reported trajectory, modulation, and risk policy."
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
        let mut output = match feeling {
            Some(feeling) => serde_json::to_value(&feeling)?,
            None => {
                serde_json::json!({"available": false, "reason": "agent has no Thymos observations yet"})
            }
        };
        if output.get("available").is_none() {
            output["available"] = serde_json::json!(true);
        }
        for (key, field) in [
            ("spine_trajectory", "trajectory"),
            ("spine_modulation", "modulation"),
            ("spine_risk_policy", "risk_policy"),
        ] {
            output[field] = context
                .metadata
                .get(key)
                .filter(|value| value.len() <= 8_192)
                .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
                .filter(serde_json::Value::is_object)
                .unwrap_or(serde_json::Value::Null);
        }
        output["grid_state"] = self
            .heart
            .thymos_diagnostics(&agent)?
            .unwrap_or(serde_json::Value::Null);
        Ok(ToolResult::success(output.to_string()))
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
                let mut fact = fact_json(&hit.fact);
                fact["score"] = serde_json::json!(hit.score);
                fact
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
            description: "Deterministically aggregate event facts with a natural-language query, or use the legacy slot_prefix plus operation interface. Natural queries route to sum/count/diff/max/min; the legacy operations remain sum/count/latest. Results include evidence provenance.".into(),
            category: ToolCategory::Internal,
            risk: ToolRisk::ReadOnly,
            parameters: object_schema(
                serde_json::json!({
                    "query": {
                        "type": "string",
                        "description": "Natural-language aggregation query; provide this alone"
                    },
                    "slot_prefix": {
                        "type": "string",
                        "description": "Legacy explicit route; requires operation and cannot be combined with query"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["sum", "count", "latest"],
                        "description": "Legacy explicit route; requires slot_prefix and cannot be combined with query"
                    }
                }),
                &[],
            ),
        }
    }

    async fn execute(
        &self,
        call: &ToolCall,
        _context: &ToolContext,
    ) -> spine_runtime::Result<ToolResult> {
        let query = call
            .arguments
            .get("query")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let prefix = call.arguments.get("slot_prefix").and_then(|v| v.as_str());
        let operation = call.arguments.get("operation").and_then(|v| v.as_str());

        if query.is_some() && (prefix.is_some() || operation.is_some()) {
            return Ok(ToolResult::failure(
                "fact_aggregate query cannot be combined with slot_prefix or operation",
            ));
        }
        if let Some(query) = query {
            let aggregation = self.heart.aggregate_fact_query(query)?;
            return Ok(ToolResult::success(
                query_aggregation_json(aggregation).to_string(),
            ));
        }

        let Some(prefix) = prefix else {
            return Ok(ToolResult::failure("fact_aggregate requires slot_prefix"));
        };
        let Some(operation) = operation else {
            return Ok(ToolResult::failure("fact_aggregate requires operation"));
        };
        let (aggregation, fact_evidence) = self
            .heart
            .aggregate_facts_with_evidence(prefix, operation)?;
        let aggregable_evidence: Vec<_> = fact_evidence
            .iter()
            .filter(|fact| {
                matches!(
                    fact.slot_type,
                    FactSlotType::EventAmount | FactSlotType::EventCount
                )
            })
            .collect();
        let evidence_total = aggregable_evidence.len();
        let evidence: Vec<_> = aggregable_evidence
            .into_iter()
            .take(40)
            .map(fact_json)
            .collect();
        let evidence_truncated = evidence_total.saturating_sub(evidence.len());
        let value = match aggregation {
            FactAggregation::Sum(value) => {
                serde_json::json!({
                    "operation": "sum",
                    "value": value,
                    "evidence": evidence,
                    "evidence_total": evidence_total,
                    "evidence_truncated": evidence_truncated,
                })
            }
            FactAggregation::Count(value) => {
                serde_json::json!({
                    "operation": "count",
                    "value": value,
                    "evidence": evidence,
                    "evidence_total": evidence_total,
                    "evidence_truncated": evidence_truncated,
                })
            }
            FactAggregation::Latest(fact) => match fact {
                Some(fact) => {
                    let mut value = fact_json(&fact);
                    value["operation"] = serde_json::json!("latest");
                    value
                }
                None => serde_json::json!({"operation": "latest", "value": null}),
            },
        };
        Ok(ToolResult::success(value.to_string()))
    }
}

fn query_aggregation_json(aggregation: Option<FactQueryAggregation>) -> serde_json::Value {
    match aggregation {
        Some(FactQueryAggregation::Sum {
            value,
            evidence,
            money,
        }) => {
            let (evidence, evidence_total, evidence_truncated) = evidence_json(&evidence, 40);
            serde_json::json!({
                "operation": "sum",
                "value": value,
                "money": money,
                "evidence": evidence,
                "evidence_total": evidence_total,
                "evidence_truncated": evidence_truncated,
            })
        }
        Some(FactQueryAggregation::Count { value, evidence }) => {
            let (evidence, evidence_total, evidence_truncated) = evidence_json(&evidence, 10);
            serde_json::json!({
                "operation": "count",
                "value": value,
                "evidence": evidence,
                "evidence_total": evidence_total,
                "evidence_truncated": evidence_truncated,
            })
        }
        Some(FactQueryAggregation::Diff {
            value,
            highest_value,
            highest,
            lowest_value,
            lowest,
            money,
        }) => {
            let highest = fact_json(&highest);
            let lowest = fact_json(&lowest);
            serde_json::json!({
                "operation": "diff",
                "value": value,
                "money": money,
                "highest": {"value": highest_value, "fact": highest.clone()},
                "lowest": {"value": lowest_value, "fact": lowest.clone()},
                "evidence": [highest, lowest],
                "evidence_total": 2,
                "evidence_truncated": 0,
            })
        }
        Some(FactQueryAggregation::Max { value, fact, money }) => {
            let fact = fact_json(&fact);
            serde_json::json!({
                "operation": "max",
                "value": value,
                "money": money,
                "fact": fact.clone(),
                "evidence": [fact],
                "evidence_total": 1,
                "evidence_truncated": 0,
            })
        }
        Some(FactQueryAggregation::Min { value, fact, money }) => {
            let fact = fact_json(&fact);
            serde_json::json!({
                "operation": "min",
                "value": value,
                "money": money,
                "fact": fact.clone(),
                "evidence": [fact],
                "evidence_total": 1,
                "evidence_truncated": 0,
            })
        }
        None => serde_json::json!({
            "operation": "none",
            "value": null,
            "evidence": [],
            "evidence_total": 0,
            "evidence_truncated": 0,
        }),
    }
}

fn evidence_json(facts: &[Fact], limit: usize) -> (Vec<serde_json::Value>, usize, usize) {
    let total = facts.len();
    let evidence = facts.iter().take(limit).map(fact_json).collect::<Vec<_>>();
    let truncated = total.saturating_sub(evidence.len());
    (evidence, total, truncated)
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

fn fact_value(value: &FactValue) -> serde_json::Value {
    match value {
        FactValue::Text(value) => serde_json::Value::String(value.clone()),
        FactValue::Integer(value) => serde_json::json!(value),
        FactValue::Amount(value) => serde_json::json!(value),
    }
}

fn fact_json(fact: &Fact) -> serde_json::Value {
    let slot_type = match fact.slot_type {
        FactSlotType::State => "state",
        FactSlotType::StateQuantity => "state.quantity",
        FactSlotType::Frequency => "frequency",
        FactSlotType::Event => "event",
        FactSlotType::EventAmount => "event.amount",
        FactSlotType::EventCount => "event.count",
        FactSlotType::Preference => "preference",
        FactSlotType::Entity => "entity",
    };
    serde_json::json!({
        "fact_id": fact.id.to_string(),
        "event_id": fact.event_id.to_string(),
        "node_id": fact.node_id.to_string(),
        "entity": fact.entity,
        "attribute": fact.attribute,
        "value": fact_value(&fact.value),
        "slot_type": slot_type,
        "slot": fact.slot_key,
        "excerpt": fact.excerpt,
        "event_time": fact.event_time,
        "session_time": fact.session_time,
        "session_id": fact.metadata.get("session_id"),
        "time_source": format!("{:?}", fact.time_source).to_ascii_lowercase(),
        "arrival_order": fact.arrival_order,
        "source_role": fact.source_role,
        "superseded": fact.superseded_by.is_some(),
        "confidence": fact.confidence,
    })
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

    #[test]
    fn automatic_recall_builds_python_compatible_risk_features() {
        let recall = AutomaticRecall {
            context: "[]".into(),
            hierarchy_depth: 3,
            total_nodes: 99,
            retrieval_stat_dimensions: 6,
            hits: vec![
                RecallHit {
                    node_id: spine_heart::NodeId::from_bytes([1; 32]),
                    score: 0.9,
                    semantic_score: 0.9,
                    graph_score: 0.0,
                    freshness: 0.0,
                    confidence: 0.8,
                    tensioned: true,
                },
                RecallHit {
                    node_id: spine_heart::NodeId::from_bytes([2; 32]),
                    score: 0.5,
                    semantic_score: 0.5,
                    graph_score: 0.0,
                    freshness: 0.0,
                    confidence: 0.6,
                    tensioned: false,
                },
            ],
        };
        let stats = recall.risk_stats();
        assert!((stats[0] - 0.9).abs() < f32::EPSILON);
        assert!((stats[1] - 0.2).abs() < f32::EPSILON);
        assert!((stats[2] - 0.7).abs() < f32::EPSILON);
        assert!((stats[3] - 0.5).abs() < f32::EPSILON);
        assert!((stats[4] - 0.6).abs() < f32::EPSILON);
        assert!((stats[5] - 0.46051702).abs() < f32::EPSILON);
        let mut legacy = recall;
        legacy.retrieval_stat_dimensions = 10;
        legacy.context = "[{}, {}, {}]".into();
        assert_eq!(&legacy.risk_stats()[..6], stats.as_slice());
        assert_eq!(&legacy.risk_stats()[6..], &[0.3, 0.0, 0.0, 0.0]);
        legacy.context = "[]".into();
        legacy.hits.clear();
        assert_eq!(
            legacy.risk_stats(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            AutomaticRecall::empty_for_layout(6).risk_stats(),
            vec![0.0; 6]
        );
        assert_eq!(
            AutomaticRecall::empty_for_layout(10).risk_stats(),
            legacy.risk_stats()
        );
    }

    #[test]
    fn automatic_recall_excludes_the_just_committed_event() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let created = SpineHeart::create(
            spine_heart::HeartConfig::new(directory.path().join("recall.spine")),
            "recall-test-passphrase",
        )
        .expect("create heart");
        let manifest = spine_heart::ModelManifest {
            schema: 1,
            model_name: "fixed-test-encoder".into(),
            artifact_hash: [1; 32],
            tokenizer_hash: [2; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        };
        created
            .heart
            .initialize_cognition(
                spine_heart::CognitiveConfig::new(1, manifest, 2).expect("cognitive config"),
            )
            .expect("initialize cognition");
        let agent = AgentId::new("main").expect("agent");
        let thread = ThreadId::new("recall-test").expect("thread");
        let embedding = Embedding::normalized(vec![1.0, 0.0, 0.0], 3).expect("embedding");
        let interaction = |text: &str| InteractionInput {
            agent_id: agent.clone(),
            thread_id: thread.clone(),
            role: ParticipantRole::User,
            kind: EventKind::Message,
            content: Content::Inline(text.into()),
            causal_parents: Vec::new(),
            provenance: Provenance::default(),
            tool: None,
            attachments: Vec::new(),
            outcome: None,
        };
        created
            .heart
            .commit_embedded(interaction("older canonical evidence"), embedding.clone())
            .expect("commit older event");
        let (_, current) = created
            .heart
            .commit_embedded(
                interaction("new prompt must not recall itself"),
                embedding.clone(),
            )
            .expect("commit current event");

        let recalled = automatic_recall_context(
            &created.heart,
            &embedding,
            "new prompt must not recall itself",
            5,
            current.event_id,
            1,
        )
        .expect("automatic recall");

        assert!(recalled.context.contains("older canonical evidence"));
        assert!(
            !recalled
                .context
                .contains("new prompt must not recall itself")
        );
        assert_eq!(recalled.hits.len(), 1);
        let before = created.heart.cognition().unwrap().unwrap().thymos;
        let surprising = Embedding::normalized(vec![0.0, 1.0, 0.0], 3).unwrap();
        assert_eq!(
            reflection_multiplier(&created.heart, &agent, &embedding).unwrap(),
            1.0
        );
        assert_eq!(
            reflection_multiplier(&created.heart, &agent, &surprising).unwrap(),
            2.0
        );
        assert_eq!(created.heart.cognition().unwrap().unwrap().thymos, before);
    }
}
