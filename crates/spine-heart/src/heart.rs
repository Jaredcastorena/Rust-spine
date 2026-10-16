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
    diagnostics::FeelingObservation,
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
        let heart = Self { config, store };
        heart.upgrade_risk_projection()?;
        Ok(heart)
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

    /// Add all Python retrieval features while retaining released legacy weights
    /// in a distinct segment. CAS prevents overwriting concurrent learned state.
    pub fn upgrade_risk_projection(&self) -> Result<bool> {
        let Some(previous) = self.cognition()? else {
            return Ok(false);
        };
        let mut replacement = previous.clone();
        if !replacement.upgrade_risk_layout()? {
            return Ok(false);
        }
        let event_ids: Vec<_> = self
            .store
            .events_canonical()?
            .iter()
            .map(|event| event.id)
            .collect();
        self.store.replace_projection_if_unchanged(
            self.config.projection_generation,
            &previous,
            &event_ids,
            &replacement,
        )?;
        Ok(true)
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
            1..=3 => {}
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
        let previous = state.clone();
        let receipt = self.commit_interaction(interaction)?;
        let events = self.store.events_canonical()?;
        if events.last().map(|event| event.id) != Some(receipt.event.id) {
            return Err(HeartError::ProjectionStale);
        }
        let before_updates =
            agent_observation_frontier(&state, &receipt.event.body.interaction.agent_id)?;
        let memory = state.observe(&receipt.event, embedding)?;
        let samples = observed_feeling(&state, &receipt.event, &memory.feeling, before_updates)?;
        self.store.put_projection_with_history(
            self.config.projection_generation,
            Some(&previous),
            &state,
            &samples,
            false,
            &events.iter().map(|event| event.id).collect::<Vec<_>>(),
        )?;
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
        let previous = state.clone();

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
        let mut samples = Vec::new();
        for (commit, (_, embedding)) in commits.into_iter().zip(items) {
            let before_updates =
                agent_observation_frontier(&state, &commit.event.body.interaction.agent_id)?;
            let memory = state.observe(&commit.event, embedding)?;
            samples.extend(observed_feeling(
                &state,
                &commit.event,
                &memory.feeling,
                before_updates,
            )?);
            results.push((commit, memory));
        }
        self.store.put_projection_with_history(
            self.config.projection_generation,
            Some(&previous),
            &state,
            &samples,
            false,
            &canonical.iter().map(|event| event.id).collect::<Vec<_>>(),
        )?;
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
        let previous = self.cognition()?;
        let canonical = self.store.events_canonical()?;
        let mut state = CognitiveState::new(config)?;
        let mut samples = Vec::new();
        for event in &canonical {
            if let Some(text) = CognitiveState::inline_text(event) {
                let before_updates =
                    agent_observation_frontier(&state, &event.body.interaction.agent_id)?;
                let memory = state.observe(event, encoder.encode(text)?)?;
                samples.extend(observed_feeling(
                    &state,
                    event,
                    &memory.feeling,
                    before_updates,
                )?);
            } else {
                state.acknowledge_unembedded(event);
            }
        }
        self.store.put_projection_with_history(
            self.config.projection_generation,
            previous.as_ref(),
            &state,
            &samples,
            true,
            &canonical.iter().map(|event| event.id).collect::<Vec<_>>(),
        )?;
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
        let previous = state.clone();
        let mut samples = Vec::new();
        for event in canonical.iter().skip(projected_events) {
            let projected = state
                .event_frontier
                .get(&event.body.device_id)
                .is_some_and(|sequence| *sequence >= event.body.device_sequence);
            debug_assert!(
                !projected,
                "suffix validation rejects projected tail events"
            );
            if let Some(text) = CognitiveState::inline_text(event) {
                let before_updates =
                    agent_observation_frontier(&state, &event.body.interaction.agent_id)?;
                let memory = state.observe(event, encoder.encode(text)?)?;
                samples.extend(observed_feeling(
                    &state,
                    event,
                    &memory.feeling,
                    before_updates,
                )?);
            } else {
                state.acknowledge_unembedded(event);
            }
        }
        if !state.is_current(&canonical_frontier) {
            return Err(HeartError::ProjectionStale);
        }
        state.triangles.verify(&state.dcmdb)?;
        self.store.put_projection_with_history(
            self.config.projection_generation,
            Some(&previous),
            &state,
            &samples,
            false,
            &canonical.iter().map(|event| event.id).collect::<Vec<_>>(),
        )?;
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

    /// Grid diagnostics and the last ten known observed learning samples. Old
    /// hearts have unknown history until new samples are observed; querying this
    /// summary never replays or changes the live learned tensor.
    pub fn thymos_diagnostics(&self, agent: &AgentId) -> Result<Option<serde_json::Value>> {
        let state = self.current_cognition()?;
        let Some(thymos) = state.thymos.get(agent) else {
            return Ok(None);
        };
        let events = self.events_canonical()?;
        if state.projected_events != events.len() as u64 {
            return Err(HeartError::ProjectionStale);
        }
        let live_events = events.into_iter().map(|event| event.id).collect();
        let history = self
            .store
            .diagnostic_history(self.config.projection_generation)?;
        let history_summary = history.summary(
            agent,
            thymos.update_count(),
            &live_events,
            crate::diagnostics::thymos_hash(thymos)?,
        )?;
        let mut summary = thymos.state_summary();
        summary
            .as_object_mut()
            .expect("grid summary object")
            .extend(
                history_summary
                    .as_object()
                    .expect("history summary object")
                    .clone(),
            );
        Ok(Some(summary))
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
        let feeling = state.risk.affect_features(&feeling)?;
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
        let feeling = state.risk.affect_features(&feeling)?;
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
        let previous = state.clone();
        let canonical = self.events_canonical()?;
        let predecessor_hash = agent_observation_frontier(&state, agent)?.1;
        let feeling = state.learn_experience(agent, context, expected, actual)?;
        let samples = canonical
            .iter()
            .rev()
            .find(|event| &event.body.interaction.agent_id == agent)
            .map(|event| {
                FeelingObservation::new(
                    agent.clone(),
                    event.id,
                    &state.thymos[agent],
                    &feeling,
                    predecessor_hash,
                )
            })
            .transpose()?
            .into_iter()
            .collect::<Vec<_>>();
        self.store.put_projection_with_history(
            self.config.projection_generation,
            Some(&previous),
            &state,
            &samples,
            false,
            &canonical.iter().map(|event| event.id).collect::<Vec<_>>(),
        )?;
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

fn agent_update_count(state: &CognitiveState, agent: &AgentId) -> u64 {
    state
        .thymos
        .get(agent)
        .map_or(0, crate::Thymos::update_count)
}

fn agent_observation_frontier(state: &CognitiveState, agent: &AgentId) -> Result<(u64, [u8; 32])> {
    let hash = state
        .thymos
        .get(agent)
        .map(crate::diagnostics::thymos_hash)
        .transpose()?
        .unwrap_or([0; 32]);
    Ok((agent_update_count(state, agent), hash))
}

fn observed_feeling(
    state: &CognitiveState,
    event: &SignedEvent,
    feeling: &FeelingVector,
    before_updates: (u64, [u8; 32]),
) -> Result<Vec<FeelingObservation>> {
    let agent = &event.body.interaction.agent_id;
    let updates = agent_update_count(state, agent);
    if updates <= before_updates.0 {
        return Ok(Vec::new());
    }
    Ok(vec![FeelingObservation::new(
        agent.clone(),
        event.id,
        &state.thymos[agent],
        feeling,
        before_updates.1,
    )?])
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
            .update_risk(&AgentId::new("main").unwrap(), &vector, &[0.0; 6], 0.8)
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
        assert_eq!(upgraded.schema, CognitiveState::CURRENT_SCHEMA);
        let facts: Vec<_> = upgraded.facts.facts().collect();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].event_id, commit.event.id);
        assert_eq!(facts[0].node_id, memory.node_id);
        assert_eq!(facts[0].value, FactValue::Integer(3));
        assert_eq!(facts[0].event_time.as_deref(), Some("2025-02-03"));
        let mut expected = legacy;
        expected.schema = CognitiveState::CURRENT_SCHEMA;
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

    #[test]
    fn document_first_person_claims_are_excluded_and_removed_by_fact_backfill() {
        let temp = tempfile::tempdir().unwrap();
        let heart = heart(&temp.path().join("document-facts.spine"));
        let vector = Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap();
        let (user, _) = heart
            .commit_embedded(input("I attended 2 weddings."), vector.clone())
            .unwrap();
        let mut documents = Vec::new();
        for provider_marker in [true, false] {
            let text = "[user]: I attended 99 weddings.";
            let mut document = input(text);
            if provider_marker {
                document.provenance.provider = Some("spine-document-ingest".into());
            } else {
                document
                    .provenance
                    .metadata
                    .insert("record_schema".into(), "spine-document-chunk".into());
            }
            document.provenance.source_uri = Some(format!("file://document-{provider_marker}.md"));
            documents.push(heart.commit_embedded(document, vector.clone()).unwrap());
        }
        let mut legacy = heart.cognition().unwrap().unwrap();
        assert_eq!(legacy.facts.facts().count(), 1);
        assert_eq!(legacy.facts.facts().next().unwrap().event_id, user.event.id);
        assert_eq!(
            heart.events_canonical().unwrap().len(),
            3,
            "documents remain in canonical memory"
        );
        legacy.schema = 2;
        let extractor = crate::FactExtractor::new().unwrap();
        for (commit, memory) in &documents {
            legacy.facts.add_candidates(
                commit.event.id,
                memory.node_id,
                extractor.extract(
                    CognitiveState::inline_text(&commit.event).unwrap(),
                    None,
                    None,
                    1,
                    [0, 1],
                ),
            );
        }
        assert_eq!(legacy.facts.facts().count(), 3);
        heart.store.put_projection(1, &legacy).unwrap();
        assert!(heart.upgrade_fact_projection().unwrap());
        let upgraded = heart.cognition().unwrap().unwrap();
        assert_eq!(upgraded.facts.facts().count(), 1);
        assert_eq!(
            upgraded.facts.facts().next().unwrap().event_id,
            user.event.id
        );
        assert_eq!(
            upgraded
                .facts
                .aggregate("attended.weddings", "count")
                .unwrap(),
            crate::FactAggregation::Count(2)
        );
        let mut expected = legacy;
        expected.schema = CognitiveState::CURRENT_SCHEMA;
        expected.facts = upgraded.facts.clone();
        assert_eq!(upgraded, expected);
        assert!(!heart.upgrade_fact_projection().unwrap());
    }

    #[test]
    fn schema_three_fact_upgrade_repairs_partner_and_age_without_resetting_cognition() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("schema-three-facts.spine");
        let heart = heart(&path);
        let vector = Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap();
        let (partner, partner_memory) = heart
            .commit_embedded(input("My wife is named Ana."), vector.clone())
            .unwrap();
        let (ordinary, ordinary_memory) = heart
            .commit_embedded(input("My wife is feeling better."), vector.clone())
            .unwrap();
        let (age, age_memory) = heart
            .commit_embedded(input("I am 9 years old."), vector.clone())
            .unwrap();
        heart
            .update_risk(&AgentId::new("main").unwrap(), &vector, &[0.0; 6], 0.8)
            .unwrap();
        heart
            .compact_context(
                [ContextLeaf {
                    node_id: partner_memory.node_id,
                    chronology: 1,
                }],
                1,
            )
            .unwrap();

        // Recreate schema 3's stored mistakes without depending on the fixed
        // extractor to continue producing them: a newer false name supersedes
        // Ana, and the single-digit age is absent from the projection.
        let extractor = crate::FactExtractor::new().unwrap();
        let mut legacy = heart.cognition().unwrap().unwrap();
        legacy.schema = 3;
        legacy.facts = FactStore::default();
        let correct = extractor.extract("My wife is named Ana.", None, None, 1, [0, 1]);
        legacy
            .facts
            .add_candidates(partner.event.id, partner_memory.node_id, correct.clone());
        let mut false_name = correct[0].clone();
        false_name.value = FactValue::Text("feeling".into());
        false_name.excerpt = "My wife is feeling better.".into();
        false_name.ingest_millis = 2;
        false_name.arrival_order = [0, 2];
        legacy
            .facts
            .add_candidates(ordinary.event.id, ordinary_memory.node_id, vec![false_name]);
        assert!(legacy.facts.facts().any(|fact| {
            fact.value == FactValue::Text("feeling".into()) && fact.superseded_by.is_none()
        }));
        assert!(legacy.facts.facts().any(|fact| {
            fact.value == FactValue::Text("Ana".into()) && fact.superseded_by.is_some()
        }));
        heart.store.put_projection(1, &legacy).unwrap();
        let canonical = heart.events_canonical().unwrap();
        drop(heart);

        let reopened = SpineHeart::open(
            HeartConfig::new(&path),
            KeySource::Passphrase("upgrade-pass".into()),
        )
        .unwrap();
        assert!(reopened.upgrade_fact_projection().unwrap());
        let upgraded = reopened.cognition().unwrap().unwrap();
        assert_eq!(upgraded.schema, 4);
        let facts: Vec<_> = upgraded.facts.facts().collect();
        assert_eq!(facts.len(), 2);
        let partner_fact = facts
            .iter()
            .find(|fact| fact.attribute == "partner_name")
            .unwrap();
        assert_eq!(partner_fact.value, FactValue::Text("Ana".into()));
        assert_eq!(partner_fact.event_id, partner.event.id);
        assert_eq!(partner_fact.node_id, partner_memory.node_id);
        assert!(partner_fact.superseded_by.is_none());
        let age_fact = facts.iter().find(|fact| fact.attribute == "age").unwrap();
        assert_eq!(age_fact.value, FactValue::Integer(9));
        assert_eq!(age_fact.event_id, age.event.id);
        assert_eq!(age_fact.node_id, age_memory.node_id);

        let mut expected = legacy;
        expected.schema = 4;
        expected.facts = upgraded.facts.clone();
        assert_eq!(
            upgraded, expected,
            "all non-fact state must survive exactly"
        );
        assert_eq!(reopened.events_canonical().unwrap(), canonical);
        assert!(!reopened.upgrade_fact_projection().unwrap());
        drop(reopened);
        let reopened = SpineHeart::open(
            HeartConfig::new(&path),
            KeySource::Passphrase("upgrade-pass".into()),
        )
        .unwrap();
        assert_eq!(reopened.cognition().unwrap().unwrap(), upgraded);
        assert!(!reopened.upgrade_fact_projection().unwrap());
    }

    #[test]
    fn diagnostic_history_survives_reopen_and_old_missing_history_stays_unknown() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("diagnostics.spine");
        let heart = heart(&path);
        let agent = AgentId::new("main").unwrap();
        let mut observed = Vec::new();
        for index in 0..15 {
            let vector =
                Embedding::normalized(vec![1.0, index as f32 / 10.0 + 0.1, 0.2], 3).unwrap();
            let before = agent_update_count(&heart.cognition().unwrap().unwrap(), &agent);
            let (_, memory) = heart
                .commit_embedded(input(&format!("observation {index}")), vector)
                .unwrap();
            let after = agent_update_count(&heart.cognition().unwrap().unwrap(), &agent);
            if after > before {
                observed.push(memory.feeling);
            }
        }
        assert!(observed.len() >= 10);
        let before = heart.cognition().unwrap().unwrap();
        let summary = heart.thymos_diagnostics(&agent).unwrap().unwrap();
        let expected = observed
            .iter()
            .rev()
            .take(10)
            .map(|feeling| f64::from(feeling.valence))
            .sum::<f64>()
            / 10.0;
        assert!((summary["valence_trend"].as_f64().unwrap() - expected).abs() < 1e-9);
        assert_eq!(summary["history_length"], observed.len());
        assert_eq!(summary["trend_window_complete"], true);
        assert_eq!(heart.cognition().unwrap().unwrap(), before);
        drop(heart);
        let reopened = SpineHeart::open(
            HeartConfig::new(&path),
            KeySource::Passphrase("upgrade-pass".into()),
        )
        .unwrap();
        assert_eq!(
            reopened.thymos_diagnostics(&agent).unwrap().unwrap(),
            summary
        );
        let canonical = reopened.events_canonical().unwrap();
        let ids: Vec<_> = canonical.iter().map(|event| event.id).collect();
        reopened
            .store
            .put_projection_with_history(1, Some(&before), &before, &[], true, &ids)
            .unwrap();
        let unknown = reopened.thymos_diagnostics(&agent).unwrap().unwrap();
        assert!(unknown["valence_trend"].is_null());
        assert_eq!(unknown["history_length"], 0);
        assert_eq!(reopened.cognition().unwrap().unwrap(), before);
        reopened
            .commit_embedded(
                input("new known observation"),
                Embedding::normalized(vec![0.4, 1.0, 0.3], 3).unwrap(),
            )
            .unwrap();
        let known = reopened.thymos_diagnostics(&agent).unwrap().unwrap();
        assert_eq!(known["history_length"], 1);
        assert!(known["valence_trend"].is_number());
        assert_eq!(known["trend_window_complete"], false);
        reopened
            .redact(TombstoneTarget::Event(canonical[0].id), None)
            .unwrap();
        assert!(matches!(
            reopened.thymos_diagnostics(&agent),
            Err(HeartError::ProjectionStale)
        ));
    }

    #[test]
    fn suffix_history_retry_is_atomic_and_preserves_learned_context_after_reopen() {
        struct SuffixEncoder {
            manifest: ModelManifest,
            fail_second: bool,
        }
        impl SemanticEncoder for SuffixEncoder {
            fn manifest(&self) -> &ModelManifest {
                &self.manifest
            }
            fn encode(&self, text: &str) -> Result<Embedding> {
                if self.fail_second && text == "suffix-two" {
                    return Err(HeartError::InvalidInput(
                        "test suffix encoder failure".into(),
                    ));
                }
                Embedding::normalized(
                    if text == "suffix-one" {
                        vec![0.2, 1.0, 0.1]
                    } else {
                        vec![1.0, 0.2, 0.4]
                    },
                    3,
                )
            }
        }

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("suffix-diagnostics.spine");
        let heart = heart(&path);
        let agent = AgentId::new("main").unwrap();
        let mut feelings = Vec::new();
        let mut leaves = Vec::new();
        for index in 0..12 {
            let vector = Embedding::normalized(
                if index % 2 == 0 {
                    vec![1.0, 0.0, 0.0]
                } else {
                    vec![0.82, 0.57, 0.0]
                },
                3,
            )
            .unwrap();
            let before = agent_update_count(&heart.cognition().unwrap().unwrap(), &agent);
            let (commit, memory) = heart
                .commit_embedded(input(&format!("prefix-{index}")), vector)
                .unwrap();
            if agent_update_count(&heart.cognition().unwrap().unwrap(), &agent) > before {
                feelings.push(memory.feeling);
            }
            if index < 2 {
                leaves.push(ContextLeaf {
                    node_id: memory.node_id,
                    chronology: commit.event.body.device_sequence,
                });
            }
        }
        heart.compact_context(leaves, 1).unwrap();
        heart
            .update_risk(
                &agent,
                &Embedding::normalized(vec![0.0, 0.0, 1.0], 3).unwrap(),
                &[0.5; 6],
                0.9,
            )
            .unwrap();
        let previous = heart.cognition().unwrap().unwrap();
        assert!(!previous.triangles.triangles.is_empty());
        let old_summary = heart.thymos_diagnostics(&agent).unwrap().unwrap();
        let previous_history =
            postcard::to_allocvec(&heart.store.diagnostic_history(1).unwrap()).unwrap();
        heart.commit_interaction(input("suffix-one")).unwrap();
        let mut assistant = input("suffix-assistant");
        assistant.role = ParticipantRole::Assistant;
        heart.commit_interaction(assistant).unwrap();
        heart.commit_interaction(input("suffix-two")).unwrap();
        let mut encoder = SuffixEncoder {
            manifest: previous.config.model.clone(),
            fail_second: true,
        };

        assert!(heart.catch_up_cognition(&encoder).is_err());
        assert_eq!(heart.cognition().unwrap().unwrap(), previous);
        assert_eq!(
            postcard::to_allocvec(&heart.store.diagnostic_history(1).unwrap()).unwrap(),
            previous_history
        );

        encoder.fail_second = false;
        let mut expected = previous.clone();
        for event in heart
            .events_canonical()
            .unwrap()
            .iter()
            .skip(previous.projected_events as usize)
        {
            let before = agent_update_count(&expected, &agent);
            let memory = expected
                .observe(
                    event,
                    encoder
                        .encode(CognitiveState::inline_text(event).unwrap())
                        .unwrap(),
                )
                .unwrap();
            if agent_update_count(&expected, &agent) > before {
                feelings.push(memory.feeling);
            }
        }
        let recovered = heart.catch_up_cognition(&encoder).unwrap();
        assert_eq!(recovered, expected);
        assert_eq!(recovered.risk, previous.risk);
        assert_eq!(recovered.triangles, previous.triangles);
        let summary = heart.thymos_diagnostics(&agent).unwrap().unwrap();
        assert_eq!(
            summary["history_length"].as_u64().unwrap(),
            old_summary["history_length"].as_u64().unwrap() + 2
        );
        assert_eq!(summary["history_current"], true);
        assert_eq!(summary["trend_window_complete"], true);
        let expected_trend = feelings
            .iter()
            .rev()
            .take(10)
            .map(|feeling| f64::from(feeling.valence))
            .sum::<f64>()
            / 10.0;
        assert!((summary["valence_trend"].as_f64().unwrap() - expected_trend).abs() < 1e-9);
        heart.catch_up_cognition(&encoder).unwrap();
        assert_eq!(heart.thymos_diagnostics(&agent).unwrap().unwrap(), summary);
        drop(heart);
        let reopened = SpineHeart::open(
            HeartConfig::new(&path),
            KeySource::Passphrase("upgrade-pass".into()),
        )
        .unwrap();
        assert_eq!(reopened.cognition().unwrap().unwrap(), recovered);
        assert_eq!(
            reopened.thymos_diagnostics(&agent).unwrap().unwrap(),
            summary
        );
    }

    #[test]
    fn diagnostic_history_install_is_atomic_and_rejects_concurrent_projection_changes() {
        let temp = tempfile::tempdir().unwrap();
        let heart = heart(&temp.path().join("diagnostic-atomic.spine"));
        let previous = heart.cognition().unwrap().unwrap();
        let mut proposed = previous.clone();
        proposed.schema = 99;
        let invalid = FeelingObservation {
            agent: AgentId::new("main").unwrap(),
            event_id: crate::EventId::from_bytes([1; 32]),
            update_count: 1,
            valence: f32::NAN,
            arousal: 0.0,
            state_hash: [0; 32],
            predecessor_hash: [0; 32],
        };
        assert!(
            heart
                .store
                .put_projection_with_history(1, Some(&previous), &proposed, &[invalid], false, &[])
                .is_err()
        );
        assert_eq!(heart.cognition().unwrap().unwrap(), previous);
        heart.store.put_projection(1, &proposed).unwrap();
        assert!(matches!(
            heart
                .store
                .put_projection_with_history(1, Some(&previous), &previous, &[], true, &[]),
            Err(HeartError::ProjectionStale)
        ));
        assert_eq!(heart.cognition().unwrap().unwrap(), proposed);
        heart
            .commit_interaction(input("concurrent canonical event"))
            .unwrap();
        assert!(matches!(
            heart
                .store
                .put_projection_with_history(1, Some(&proposed), &proposed, &[], true, &[]),
            Err(HeartError::ProjectionStale)
        ));
        assert_eq!(heart.cognition().unwrap().unwrap(), proposed);
    }
}

#[cfg(test)]
mod risk_upgrade_tests {
    use super::*;

    fn legacy_config() -> CognitiveConfig {
        let mut config = CognitiveConfig::new(
            1,
            crate::ModelManifest {
                schema: 1,
                model_name: "risk-upgrade-test".into(),
                artifact_hash: [1; 32],
                tokenizer_hash: [2; 32],
                dimension: 3,
                normalized: true,
                quantization: None,
            },
            2,
        )
        .unwrap();
        config.retrieval_stat_dimensions = 4;
        config
    }

    #[test]
    fn risk_upgrade_cas_rejects_concurrent_learning_or_new_events() {
        for event_conflict in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let heart = SpineHeart::create(
                HeartConfig::new(directory.path().join("cas.spine")),
                "cas-pass",
            )
            .unwrap()
            .heart;
            heart.initialize_cognition(legacy_config()).unwrap();
            let previous = heart.cognition().unwrap().unwrap();
            let mut replacement = previous.clone();
            replacement.upgrade_risk_layout().unwrap();
            let agent = AgentId::new("main").unwrap();
            if event_conflict {
                heart
                    .commit_interaction(InteractionInput {
                        agent_id: agent,
                        thread_id: crate::ThreadId::new("test").unwrap(),
                        role: crate::ParticipantRole::User,
                        kind: crate::EventKind::Message,
                        content: Content::Inline("concurrent observation".into()),
                        causal_parents: Vec::new(),
                        provenance: crate::Provenance::default(),
                        tool: None,
                        attachments: Vec::new(),
                        outcome: None,
                    })
                    .unwrap();
            } else {
                heart
                    .update_risk(
                        &agent,
                        &Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap(),
                        &[0.3, 0.0, 0.0, 0.0],
                        1.0,
                    )
                    .unwrap();
            }
            let concurrent = heart.cognition().unwrap().unwrap();
            assert!(matches!(
                heart
                    .store
                    .replace_projection_if_unchanged(1, &previous, &[], &replacement),
                Err(HeartError::ProjectionStale)
            ));
            assert_eq!(heart.cognition().unwrap().unwrap(), concurrent);
        }
    }

    #[test]
    fn unknown_persisted_risk_layout_is_rejected_without_resetting_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unknown.spine");
        let heart = SpineHeart::create(HeartConfig::new(&path), "unknown-pass")
            .unwrap()
            .heart;
        let mut config = legacy_config();
        config.retrieval_stat_dimensions = 5;
        heart.initialize_cognition(config).unwrap();
        let before = heart.cognition().unwrap().unwrap();
        assert!(heart.upgrade_risk_projection().is_err());
        assert_eq!(heart.cognition().unwrap().unwrap(), before);
        drop(heart);
        assert!(
            SpineHeart::open(
                HeartConfig::new(&path),
                KeySource::Passphrase("unknown-pass".into())
            )
            .is_err()
        );
    }
}
