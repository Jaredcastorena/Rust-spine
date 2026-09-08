use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, CognitiveConfig, CognitiveState, Content, Embedding, EncryptedDelta, EventBody,
    FeelingVector, HeartError, ImportReceipt, InteractionInput, KeySource, MemoryReceipt,
    RecoveryPhrase, Result, SemanticEncoder, SignedEvent, Snapshot, SnapshotId, StoreStats,
    SyncFrontier, Tombstone, TombstoneId, TombstoneTarget,
    store::{CreatedStore, Store},
    sync,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeartConfig {
    pub path: PathBuf,
    pub projection_generation: u64,
    pub model_manifest_hash: Option<[u8; 32]>,
}

impl HeartConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            projection_generation: 1,
            model_manifest_hash: None,
        }
    }
}

pub struct CreatedHeart {
    pub heart: SpineHeart,
    pub recovery_phrase: RecoveryPhrase,
}

#[derive(Clone)]
pub struct SpineHeart {
    config: HeartConfig,
    store: Store,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitReceipt {
    pub event: SignedEvent,
    pub inserted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecalledMemory {
    pub hit: crate::RecallHit,
    pub events: Vec<SignedEvent>,
}

pub struct ReadOnlyHeart {
    heart: SpineHeart,
    snapshot: Snapshot,
}

impl SpineHeart {
    pub fn create(config: HeartConfig, passphrase: &str) -> Result<CreatedHeart> {
        let CreatedStore {
            store,
            recovery_phrase,
        } = Store::create(&config.path, passphrase)?;
        Ok(CreatedHeart {
            heart: Self { config, store },
            recovery_phrase,
        })
    }

    pub fn open(config: HeartConfig, keys: KeySource) -> Result<Self> {
        let store = Store::open(&config.path, keys)?;
        Ok(Self { config, store })
    }

    pub fn create_replica(
        config: HeartConfig,
        recovery_phrase: &str,
        passphrase: &str,
    ) -> Result<Self> {
        let store = Store::create_replica(&config.path, recovery_phrase, passphrase)?;
        Ok(Self { config, store })
    }

    pub fn path(&self) -> &Path {
        &self.config.path
    }

    pub fn device_id(&self) -> crate::DeviceId {
        self.store.device_id()
    }

    pub fn commit_interaction(&self, interaction: InteractionInput) -> Result<CommitReceipt> {
        validate_interaction(&interaction)?;
        let now = now_millis()?;
        let (device_sequence, timestamp) = self.store.reserve_clock(now)?;
        let body = EventBody {
            schema: 1,
            device_id: self.store.device_id(),
            authorization_epoch: self.store.current_authorization_epoch()?,
            device_sequence,
            timestamp,
            interaction,
        };
        let event = self.store.sign_event(body)?;
        let inserted = self.store.put_event(&event)?;
        Ok(CommitReceipt { event, inserted })
    }

    pub fn initialize_cognition(&self, config: CognitiveConfig) -> Result<()> {
        if config.generation != self.config.projection_generation {
            return Err(HeartError::InvalidInput(
                "cognitive generation does not match HeartConfig".into(),
            ));
        }
        if self
            .store
            .get_projection::<CognitiveState>(config.generation)?
            .is_some()
        {
            return Err(HeartError::InvalidInput(
                "cognitive projection generation already exists".into(),
            ));
        }
        if !self.store.events_canonical()?.is_empty() {
            return Err(HeartError::ProjectionStale);
        }
        self.store
            .put_projection(config.generation, &CognitiveState::new(config)?)
    }

    pub fn cognition(&self) -> Result<Option<CognitiveState>> {
        self.store.get_projection(self.config.projection_generation)
    }

    pub fn cognition_is_current(&self) -> Result<bool> {
        let Some(state) = self.cognition()? else {
            return Ok(false);
        };
        Ok(state.is_current(&self.store.frontier()?))
    }

    /// Backfills historical typed facts once after the fact extractor upgrade.
    ///
    /// Call during startup after stale-frontier recovery. This requires no model
    /// inference and preserves DCMDb, affect, learned risk, context triangles and
    /// event frontiers exactly. A failed or concurrent upgrade installs nothing.
    pub fn upgrade_fact_projection(&self) -> Result<bool> {
        let previous = self.current_cognition()?;
        match previous.schema {
            CognitiveState::CURRENT_SCHEMA => return Ok(false),
            1 => {}
            found => {
                return Err(HeartError::UnsupportedSchema {
                    found,
                    expected: CognitiveState::CURRENT_SCHEMA,
                });
            }
        }
        let events = self.store.events_canonical()?;
        let mut frontier = std::collections::BTreeMap::new();
        for event in &events {
            frontier
                .entry(event.body.device_id)
                .and_modify(|sequence: &mut u64| {
                    *sequence = (*sequence).max(event.body.device_sequence)
                })
                .or_insert(event.body.device_sequence);
        }
        if !previous.is_current(&frontier) {
            return Err(HeartError::ProjectionStale);
        }
        let mut replacement = previous.clone();
        replacement.upgrade_facts(&events)?;
        let event_ids: Vec<_> = events.iter().map(|event| event.id).collect();
        self.store.replace_projection_if_unchanged(
            self.config.projection_generation,
            &previous,
            &event_ids,
            &replacement,
        )?;
        Ok(true)
    }

    pub fn commit_embedded(
        &self,
        interaction: InteractionInput,
        embedding: Embedding,
    ) -> Result<(CommitReceipt, MemoryReceipt)> {
        let mut state = self.cognition()?.ok_or(HeartError::NotFound)?;
        if !state.is_current(&self.store.frontier()?) {
            return Err(HeartError::ProjectionStale);
        }
        let receipt = self.commit_interaction(interaction)?;
        let events = self.store.events_canonical()?;
        if events.last().map(|event| event.id) != Some(receipt.event.id) {
            return Err(HeartError::ProjectionStale);
        }
        let memory = state.observe(&receipt.event, embedding)?;
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok((receipt, memory))
    }

    /// Commits a pre-embedded import as one cognitive projection update.
    ///
    /// Events remain individually durable and signed. If another writer changes canonical
    /// ordering during the import, the raw events are retained and the projection is reported
    /// stale so it can be rebuilt rather than installing an incorrectly ordered projection.
    pub fn commit_embedded_batch(
        &self,
        items: Vec<(InteractionInput, Embedding)>,
    ) -> Result<Vec<(CommitReceipt, MemoryReceipt)>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let mut state = self.cognition()?.ok_or(HeartError::NotFound)?;
        if !state.is_current(&self.store.frontier()?) {
            return Err(HeartError::ProjectionStale);
        }
        for (interaction, embedding) in &items {
            validate_interaction(interaction)?;
            if embedding.as_slice().len() != state.config.model.dimension {
                return Err(HeartError::InvalidInput(
                    "embedding dimension does not match projection model".into(),
                ));
            }
        }

        let mut commits = Vec::with_capacity(items.len());
        for (interaction, _) in &items {
            commits.push(self.commit_interaction(interaction.clone())?);
        }

        let canonical = self.store.events_canonical()?;
        if canonical.len() < commits.len()
            || !canonical[canonical.len() - commits.len()..]
                .iter()
                .zip(&commits)
                .all(|(event, commit)| event.id == commit.event.id)
        {
            return Err(HeartError::ProjectionStale);
        }

        let mut results = Vec::with_capacity(items.len());
        for (commit, (_, embedding)) in commits.into_iter().zip(items) {
            let memory = state.observe(&commit.event, embedding)?;
            results.push((commit, memory));
        }
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(results)
    }

    pub fn rebuild_cognition(
        &self,
        config: CognitiveConfig,
        encoder: &dyn SemanticEncoder,
    ) -> Result<CognitiveState> {
        if config.generation != self.config.projection_generation {
            return Err(HeartError::InvalidInput(
                "cognitive generation does not match HeartConfig".into(),
            ));
        }
        if encoder.manifest() != &config.model {
            return Err(HeartError::InvalidInput(
                "encoder manifest does not match cognitive projection".into(),
            ));
        }
        let mut state = CognitiveState::new(config)?;
        for event in self.store.events_canonical()? {
            if let Some(text) = CognitiveState::inline_text(&event) {
                state.observe(&event, encoder.encode(text)?)?;
            } else {
                state.acknowledge_unembedded(&event);
            }
        }
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(state)
    }

    /// Apply only canonical events missing from the persisted projection.
    ///
    /// Unlike a full rebuild, this preserves projection-only learned state such
    /// as risk updates, maintenance results, and context triangles.
    pub fn catch_up_cognition(&self, encoder: &dyn SemanticEncoder) -> Result<CognitiveState> {
        let mut state = self
            .cognition()?
            .ok_or_else(|| HeartError::InvalidInput("heart has no cognitive projection".into()))?;
        if encoder.manifest() != &state.config.model {
            return Err(HeartError::InvalidInput(
                "encoder manifest does not match cognitive projection".into(),
            ));
        }
        state.triangles.verify(&state.dcmdb)?;
        let canonical_frontier = self.store.frontier()?;
        if state.event_frontier.iter().any(|(device, projected)| {
            canonical_frontier.get(device).copied().unwrap_or_default() < *projected
        }) {
            return Err(HeartError::ProjectionStale);
        }
        let canonical = self.store.events_canonical()?;
        let projected_events =
            usize::try_from(state.projected_events).map_err(|_| HeartError::ProjectionStale)?;
        if projected_events > canonical.len()
            || canonical.iter().enumerate().any(|(index, event)| {
                let projected = state
                    .event_frontier
                    .get(&event.body.device_id)
                    .is_some_and(|sequence| *sequence >= event.body.device_sequence);
                (index < projected_events) != projected
            })
        {
            return Err(HeartError::ProjectionStale);
        }
        for event in canonical.into_iter().skip(projected_events) {
            let projected = state
                .event_frontier
                .get(&event.body.device_id)
                .is_some_and(|sequence| *sequence >= event.body.device_sequence);
            debug_assert!(
                !projected,
                "suffix validation rejects projected tail events"
            );
            if let Some(text) = CognitiveState::inline_text(&event) {
                state.observe(&event, encoder.encode(text)?)?;
            } else {
                state.acknowledge_unembedded(&event);
            }
        }
        if !state.is_current(&canonical_frontier) {
            return Err(HeartError::ProjectionStale);
        }
        state.triangles.verify(&state.dcmdb)?;
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(state)
    }

    pub fn recall(
        &self,
        query: &Embedding,
        now: f64,
        top_k: usize,
    ) -> Result<Vec<crate::RecallHit>> {
        let state = self.current_cognition()?;
        state.dcmdb.query(query.as_slice(), now, top_k)
    }

    pub fn recall_memories(
        &self,
        query: &Embedding,
        now: f64,
        top_k: usize,
        max_events_per_node: usize,
    ) -> Result<Vec<RecalledMemory>> {
        self.recall_memories_with_expansion(query, now, top_k, max_events_per_node, None)
    }

    /// As recall_memories, with optional bounded DCMDb evidence expansion.
    /// This walks absorbed memory coordinates, never context triangles.
    pub fn recall_memories_with_expansion(
        &self,
        query: &Embedding,
        now: f64,
        top_k: usize,
        max_events_per_node: usize,
        max_depth: Option<usize>,
    ) -> Result<Vec<RecalledMemory>> {
        let state = self.current_cognition()?;
        let hits = state.dcmdb.query(query.as_slice(), now, top_k)?;
        let canonical = self.store.events_canonical()?;
        let positions: std::collections::BTreeMap<_, _> = canonical
            .iter()
            .enumerate()
            .map(|(index, event)| (event.id, index))
            .collect();
        let by_id: std::collections::BTreeMap<_, _> = canonical
            .into_iter()
            .map(|event| (event.id, event))
            .collect();
        let mut result = Vec::with_capacity(hits.len());
        for hit in hits {
            let node = state.dcmdb.node(hit.node_id).ok_or(HeartError::NotFound)?;
            let mut ids = max_depth.map_or_else(
                || node.event_ids.clone(),
                |depth| {
                    state.dcmdb.subtree_event_ids(
                        hit.node_id,
                        depth.min(3),
                        32,
                        max_events_per_node,
                    )
                },
            );
            if max_depth.is_none() {
                ids.sort_by_key(|event_id| positions.get(event_id).copied().unwrap_or_default());
                ids.reverse();
            }
            ids.truncate(max_events_per_node);
            let events = ids
                .into_iter()
                .filter_map(|event_id| by_id.get(&event_id).cloned())
                .collect();
            result.push(RecalledMemory { hit, events });
        }
        Ok(result)
    }

    pub fn feel(&self, agent: &AgentId, context: &Embedding) -> Result<Option<FeelingVector>> {
        self.current_cognition()?.feel(agent, context)
    }

    pub fn predict_risk(
        &self,
        agent: &AgentId,
        context: &Embedding,
        retrieval_stats: &[f32],
    ) -> Result<f32> {
        let state = self.current_cognition()?;
        let feeling = state.feel(agent, context)?.map_or_else(
            || vec![0.0; state.config.thymos_channels],
            |item| item.activated,
        );
        state
            .risk
            .predict(context.as_slice(), &feeling, retrieval_stats)
    }

    pub fn update_risk(
        &self,
        agent: &AgentId,
        context: &Embedding,
        retrieval_stats: &[f32],
        tension: f32,
    ) -> Result<f32> {
        let mut state = self.current_cognition()?;
        let feeling = state.feel(agent, context)?.map_or_else(
            || vec![0.0; state.config.thymos_channels],
            |item| item.activated,
        );
        let previous = state
            .risk
            .update(context.as_slice(), &feeling, retrieval_stats, tension)?;
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(previous)
    }

    pub fn search_facts(
        &self,
        query: &str,
        top_k: usize,
        include_superseded: bool,
    ) -> Result<Vec<crate::FactHit>> {
        Ok(self
            .current_cognition()?
            .facts
            .search(query, top_k, include_superseded))
    }

    pub fn aggregate_facts(
        &self,
        slot_prefix: &str,
        operation: &str,
    ) -> Result<crate::FactAggregation> {
        self.current_cognition()?
            .facts
            .aggregate(slot_prefix, operation)
    }

    pub fn aggregate_facts_with_evidence(
        &self,
        slot_prefix: &str,
        operation: &str,
    ) -> Result<(crate::FactAggregation, Vec<crate::Fact>)> {
        let state = self.current_cognition()?;
        let evidence = state.facts.active_for_slot_prefix(slot_prefix);
        let aggregation = state.facts.aggregate(slot_prefix, operation)?;
        Ok((aggregation, evidence))
    }

    pub fn aggregate_fact_query(&self, query: &str) -> Result<Option<crate::FactQueryAggregation>> {
        self.current_cognition()?.facts.aggregate_query(query)
    }

    /// Run bounded DCMDb consolidation, pruning, and dream maintenance and persist the result.
    pub fn maintain_cognition(&self, maximum_rounds: usize) -> Result<crate::MaintenanceReport> {
        let mut state = self.current_cognition()?;
        state.triangles.verify(&state.dcmdb)?;
        let now = now_millis()? as f64 / 1_000.0;
        let protected = state.triangles.referenced_nodes();
        let report = state
            .dcmdb
            .maintain_protected(now, maximum_rounds, &protected);
        let invariant_errors = state.dcmdb.check_invariants();
        if !invariant_errors.is_empty() {
            return Err(HeartError::InvalidInput(format!(
                "DCMDb maintenance violated invariants: {}",
                invariant_errors.join("; ")
            )));
        }
        state.triangles.verify(&state.dcmdb)?;
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(report)
    }

    pub fn compact_context(
        &self,
        leaves: impl IntoIterator<Item = crate::ContextLeaf>,
        target_roots: usize,
    ) -> Result<Vec<crate::ContextBranch>> {
        let mut state = self.current_cognition()?;
        state
            .triangles
            .compact(leaves, &state.dcmdb, target_roots)?;
        let roots = state.triangles.roots.clone();
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(roots)
    }

    pub fn rehydrate_context(
        &self,
        root: crate::ContextHandle,
        query: Option<&Embedding>,
        budget: crate::RehydrateBudget,
    ) -> Result<crate::RehydratedContext> {
        let state = self.current_cognition()?;
        state.triangles.rehydrate(root, query, &state.dcmdb, budget)
    }

    pub fn reflect(
        &self,
        agent: &AgentId,
        context: &Embedding,
        expected: &Embedding,
        actual: &Embedding,
    ) -> Result<FeelingVector> {
        let mut state = self.current_cognition()?;
        let feeling = state.learn_experience(agent, context, expected, actual)?;
        self.store
            .put_projection(self.config.projection_generation, &state)?;
        Ok(feeling)
    }

    pub fn promote_agent_thymos(&self, agent: AgentId, thymos: crate::Thymos) -> Result<()> {
        let mut state = self.current_cognition()?;
        state.promote_thymos(agent, thymos)?;
        self.store
            .put_projection(self.config.projection_generation, &state)
    }

    pub fn event(&self, id: crate::EventId) -> Result<Option<SignedEvent>> {
        self.store.get_event(id)
    }

    pub fn put_blob(&self, media_type: &str, bytes: &[u8]) -> Result<crate::ColdBlobRef> {
        if media_type.trim().is_empty() {
            return Err(HeartError::InvalidInput(
                "cold blob media type must not be empty".into(),
            ));
        }
        self.store.put_blob(media_type, bytes, 1_048_576)
    }

    pub fn blob(&self, id: crate::BlobId) -> Result<Option<crate::ColdBlob>> {
        self.store.get_blob(id)
    }

    pub fn events_canonical(&self) -> Result<Vec<SignedEvent>> {
        self.store.events_canonical()
    }

    pub fn snapshot(&self, label: Option<String>) -> Result<SnapshotId> {
        let wall = now_millis()?;
        let frontier = self.store.frontier()?;
        let mut snapshot = Snapshot {
            id: SnapshotId::default(),
            label,
            created_wall_millis: wall,
            event_frontier: frontier,
            projection_generation: self.config.projection_generation,
            model_manifest_hash: self.config.model_manifest_hash,
        };
        snapshot.id = self.store.snapshot_id(&snapshot)?;
        self.store.put_snapshot(&snapshot)?;
        Ok(snapshot.id)
    }

    pub fn checkout(&self, id: SnapshotId) -> Result<ReadOnlyHeart> {
        let snapshot = self.store.get_snapshot(id)?.ok_or(HeartError::NotFound)?;
        Ok(ReadOnlyHeart {
            heart: self.clone(),
            snapshot,
        })
    }

    pub fn sync_frontier(&self) -> Result<SyncFrontier> {
        Ok(SyncFrontier {
            devices: self.store.frontier()?,
            snapshots: self
                .store
                .snapshots()?
                .into_iter()
                .map(|item| item.id)
                .collect(),
            tombstones: self
                .store
                .tombstones()?
                .into_iter()
                .map(|item| item.id)
                .collect(),
            blobs: self
                .store
                .blobs()?
                .into_iter()
                .map(|item| item.reference.id)
                .collect(),
            authorizations: self.store.authorizations()?.into_iter().fold(
                std::collections::BTreeMap::new(),
                |mut epochs, item| {
                    epochs
                        .entry(item.device_id)
                        .and_modify(|epoch| *epoch = (*epoch).max(item.epoch))
                        .or_insert(item.epoch);
                    epochs
                },
            ),
        })
    }

    pub fn export_delta(&self, remote: &SyncFrontier) -> Result<EncryptedDelta> {
        sync::export_delta(&self.store, remote)
    }

    pub fn import_delta(&self, delta: EncryptedDelta) -> Result<ImportReceipt> {
        sync::import_delta(&self.store, delta)
    }

    pub fn redact(&self, target: TombstoneTarget, reason: Option<String>) -> Result<TombstoneId> {
        let wall = now_millis()?;
        let (device_sequence, _) = self.store.reserve_clock(wall)?;
        let mut tombstone = Tombstone {
            id: TombstoneId::default(),
            target,
            device_id: self.store.device_id(),
            authorization_epoch: self.store.current_authorization_epoch()?,
            device_sequence,
            wall_millis: wall,
            reason,
            signer_public_key: [0; 32],
            signature: Vec::new(),
        };
        tombstone.id = self.store.tombstone_id(&tombstone)?;
        let signing_bytes = postcard::to_allocvec(&(
            &tombstone.target,
            tombstone.device_id,
            tombstone.authorization_epoch,
            tombstone.device_sequence,
            tombstone.wall_millis,
            &tombstone.reason,
        ))?;
        let (public, signature) = self.store.sign_bytes(&signing_bytes);
        tombstone.signer_public_key = public;
        tombstone.signature = signature;
        self.store.put_tombstone(&tombstone)?;
        Ok(tombstone.id)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        self.store.stats()
    }

    fn current_cognition(&self) -> Result<CognitiveState> {
        let state = self.cognition()?.ok_or(HeartError::NotFound)?;
        if !state.is_current(&self.store.frontier()?) {
            return Err(HeartError::ProjectionStale);
        }
        Ok(state)
    }
}

impl ReadOnlyHeart {
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    pub fn events_canonical(&self) -> Result<Vec<SignedEvent>> {
        let frontier = &self.snapshot.event_frontier;
        Ok(self
            .heart
            .events_canonical()?
            .into_iter()
            .filter(|event| {
                event.body.device_sequence
                    <= frontier
                        .get(&event.body.device_id)
                        .copied()
                        .unwrap_or_default()
            })
            .collect())
    }
}

fn validate_interaction(interaction: &InteractionInput) -> Result<()> {
    if let Content::Inline(text) = &interaction.content
        && text.len() > 16 * 1024 * 1024
    {
        return Err(HeartError::InvalidInput(
            "inline interaction exceeds 16 MiB; use a cold blob".into(),
        ));
    }
    Ok(())
}

fn now_millis() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| HeartError::InvalidInput("system clock precedes Unix epoch".into()))?;
    Ok(duration.as_millis().min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod fact_upgrade_tests {
    use super::*;
    use crate::{
        ContextLeaf, EventKind, FactStore, FactValue, ModelManifest, ParticipantRole, Provenance,
        ThreadId,
    };

    fn input(text: &str) -> InteractionInput {
        InteractionInput {
            agent_id: AgentId::new("main").unwrap(),
            thread_id: ThreadId::new("facts-upgrade").unwrap(),
            role: ParticipantRole::User,
            kind: EventKind::Message,
            content: Content::Inline(text.into()),
            causal_parents: Vec::new(),
            provenance: Provenance::default(),
            tool: None,
            attachments: Vec::new(),
            outcome: None,
        }
    }

    fn heart(path: &Path) -> SpineHeart {
        let heart = SpineHeart::create(HeartConfig::new(path), "upgrade-pass")
            .unwrap()
            .heart;
        let manifest = ModelManifest {
            schema: 1,
            model_name: "upgrade-test".into(),
            artifact_hash: [3; 32],
            tokenizer_hash: [4; 32],
            dimension: 3,
            normalized: true,
            quantization: None,
        };
        heart
            .initialize_cognition(CognitiveConfig::new(1, manifest, 2).unwrap())
            .unwrap();
        heart
    }

    #[test]
    fn current_legacy_heart_backfills_facts_once_without_replaying_cognition() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("upgrade.spine");
        let heart = heart(&path);
        assert!(!heart.upgrade_fact_projection().unwrap());
        let vector = Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap();
        let (commit, memory) = heart.commit_embedded(
            input("[date: 2025/02/03] [user]: I attended 3 weddings.\n[assistant]: I attended 99 weddings."),
            vector.clone(),
        ).unwrap();
        heart
            .update_risk(&AgentId::new("main").unwrap(), &vector, &[0.0; 4], 0.8)
            .unwrap();
        heart
            .compact_context(
                [ContextLeaf {
                    node_id: memory.node_id,
                    chronology: 1,
                }],
                1,
            )
            .unwrap();
        let mut legacy = heart.cognition().unwrap().unwrap();
        legacy.schema = 1;
        legacy.facts = FactStore::default();
        heart.store.put_projection(1, &legacy).unwrap();
        assert!(heart.cognition_is_current().unwrap());
        drop(heart);

        let reopened = SpineHeart::open(
            HeartConfig::new(&path),
            KeySource::Passphrase("upgrade-pass".into()),
        )
        .unwrap();
        assert!(reopened.upgrade_fact_projection().unwrap());
        let upgraded = reopened.cognition().unwrap().unwrap();
        assert_eq!(upgraded.schema, 2);
        let facts: Vec<_> = upgraded.facts.facts().collect();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].event_id, commit.event.id);
        assert_eq!(facts[0].node_id, memory.node_id);
        assert_eq!(facts[0].value, FactValue::Integer(3));
        assert_eq!(facts[0].event_time.as_deref(), Some("2025-02-03"));
        let mut expected = legacy;
        expected.schema = 2;
        expected.facts = upgraded.facts.clone();
        assert_eq!(
            upgraded, expected,
            "all non-fact cognitive state must survive exactly"
        );
        assert!(!reopened.upgrade_fact_projection().unwrap());
        assert_eq!(reopened.cognition().unwrap().unwrap(), upgraded);
        assert_eq!(reopened.events_canonical().unwrap(), vec![commit.event]);
    }

    #[test]
    fn fact_upgrade_without_retained_provenance_installs_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let heart = heart(&temp.path().join("missing.spine"));
        heart
            .commit_embedded(
                input("I attended 3 weddings."),
                Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap(),
            )
            .unwrap();
        let mut legacy = heart.cognition().unwrap().unwrap();
        legacy.schema = 1;
        legacy.facts = FactStore::default();
        legacy.dcmdb.nodes.clear();
        legacy.dcmdb.absorbed.clear();
        heart.store.put_projection(1, &legacy).unwrap();
        assert!(
            matches!(heart.upgrade_fact_projection(), Err(HeartError::InvalidInput(message)) if message.contains("no retained DCMDb provenance"))
        );
        assert_eq!(heart.cognition().unwrap().unwrap(), legacy);
        heart
            .commit_interaction(input("new unprojected event"))
            .unwrap();
        assert!(matches!(
            heart.upgrade_fact_projection(),
            Err(HeartError::ProjectionStale)
        ));
        assert_eq!(heart.cognition().unwrap().unwrap(), legacy);
    }

    #[test]
    fn fact_upgrade_compare_and_swap_rejects_concurrent_state_and_event_changes() {
        let temp = tempfile::tempdir().unwrap();
        let heart = heart(&temp.path().join("concurrent.spine"));
        let previous = heart.cognition().unwrap().unwrap();
        let mut concurrent = previous.clone();
        concurrent.config.retrieval_stat_dimensions = 5;
        heart.store.put_projection(1, &concurrent).unwrap();
        assert!(matches!(
            heart
                .store
                .replace_projection_if_unchanged(1, &previous, &[], &previous),
            Err(HeartError::ProjectionStale)
        ));
        assert_eq!(heart.cognition().unwrap().unwrap(), concurrent);
        heart.commit_interaction(input("new event")).unwrap();
        assert!(matches!(
            heart
                .store
                .replace_projection_if_unchanged(1, &concurrent, &[], &previous),
            Err(HeartError::ProjectionStale)
        ));
        assert_eq!(heart.cognition().unwrap().unwrap(), concurrent);
    }
}
