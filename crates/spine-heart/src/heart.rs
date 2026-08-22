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
            let mut ids = node.event_ids.clone();
            ids.sort_by_key(|event_id| positions.get(event_id).copied().unwrap_or_default());
            ids.reverse();
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

    /// Run bounded DCMDb consolidation, pruning, and dream maintenance and persist the result.
    pub fn maintain_cognition(&self, maximum_rounds: usize) -> Result<crate::MaintenanceReport> {
        let mut state = self.current_cognition()?;
        let now = now_millis()? as f64 / 1_000.0;
        let report = state.dcmdb.maintain(now, maximum_rounds);
        let invariant_errors = state.dcmdb.check_invariants();
        if !invariant_errors.is_empty() {
            return Err(HeartError::InvalidInput(format!(
                "DCMDb maintenance violated invariants: {}",
                invariant_errors.join("; ")
            )));
        }
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
