use std::collections::{BTreeMap, btree_map::Entry};

use serde::{Deserialize, Serialize};

use crate::{
    AgentId, Content, ContextForest, Dcmdb, DcmdbConfig, Embedding, EventId, FactExtractor,
    FactStore, FeelingVector, HeartError, MemoryObservation, ModelManifest, NodeId,
    ParticipantRole, Result, RiskField, SignedEvent, Thymos, ThymosConfig, TrajectoryStep,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveConfig {
    pub generation: u64,
    pub model: ModelManifest,
    pub thymos_channels: usize,
    pub retrieval_stat_dimensions: usize,
}

impl CognitiveConfig {
    pub fn new(generation: u64, model: ModelManifest, thymos_channels: usize) -> Result<Self> {
        if generation == 0 {
            return Err(HeartError::InvalidInput(
                "projection generation must be positive".into(),
            ));
        }
        if model.dimension < 2 || thymos_channels == 0 {
            return Err(HeartError::InvalidInput(
                "cognitive projection requires dimension >= 2 and Thymos channels > 0".into(),
            ));
        }
        Ok(Self {
            generation,
            model,
            thymos_channels,
            retrieval_stat_dimensions: 4,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveState {
    pub schema: u32,
    pub config: CognitiveConfig,
    pub event_frontier: BTreeMap<crate::DeviceId, u64>,
    pub dcmdb: Dcmdb,
    pub thymos: BTreeMap<AgentId, Thymos>,
    pub risk: RiskField,
    pub facts: FactStore,
    pub triangles: ContextForest,
    pub projected_events: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryReceipt {
    pub event_id: EventId,
    pub node_id: NodeId,
    pub feeling: FeelingVector,
    pub trajectory: TrajectoryStep,
}

impl CognitiveState {
    pub fn new(config: CognitiveConfig) -> Result<Self> {
        let dcmdb = Dcmdb::new(DcmdbConfig::dense(config.model.dimension))?;
        let risk = RiskField::new(
            config.model.dimension,
            config.thymos_channels,
            config.retrieval_stat_dimensions,
        );
        Ok(Self {
            schema: 1,
            config,
            event_frontier: BTreeMap::new(),
            dcmdb,
            thymos: BTreeMap::new(),
            risk,
            facts: FactStore::default(),
            triangles: ContextForest::default(),
            projected_events: 0,
        })
    }

    pub fn is_current(&self, frontier: &BTreeMap<crate::DeviceId, u64>) -> bool {
        &self.event_frontier == frontier
    }

    pub fn observe(&mut self, event: &SignedEvent, embedding: Embedding) -> Result<MemoryReceipt> {
        if embedding.as_slice().len() != self.config.model.dimension {
            return Err(HeartError::InvalidInput(
                "embedding dimension does not match projection model".into(),
            ));
        }
        let already_projected = self
            .event_frontier
            .get(&event.body.device_id)
            .is_some_and(|sequence| *sequence >= event.body.device_sequence);
        if already_projected {
            return Err(HeartError::InvalidInput(
                "event is already represented by this projection frontier".into(),
            ));
        }

        let interaction = &event.body.interaction;
        let mut metadata = BTreeMap::new();
        metadata.insert("event_id".into(), event.id.to_string());
        metadata.insert("agent_id".into(), interaction.agent_id.to_string());
        metadata.insert("thread_id".into(), interaction.thread_id.to_string());
        metadata.insert("role".into(), format!("{:?}", interaction.role));
        let time = event.body.timestamp.wall_millis as f64 / 1_000.0;
        let node_id = self.dcmdb.update(MemoryObservation {
            event_id: event.id,
            vector: embedding.as_slice().to_vec(),
            time,
            source: interaction.provenance.source_uri.clone(),
            metadata,
        })?;
        if interaction.role == ParticipantRole::User
            && let Some(text) = Self::inline_text(event)
        {
            let candidates = FactExtractor::new()?.extract(
                text,
                None,
                None,
                event.body.timestamp.wall_millis,
                [
                    event.body.timestamp.wall_millis,
                    u64::from(event.body.timestamp.counter),
                ],
            );
            self.facts.add_candidates(event.id, node_id, candidates);
        }

        let channels = self.config.thymos_channels;
        let dimension = self.config.model.dimension;
        let model_hash = self.config.model.artifact_hash;
        let thymos = match self.thymos.entry(interaction.agent_id.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"spine-thymos-agent-v1");
                hasher.update(&model_hash);
                hasher.update(interaction.agent_id.as_str().as_bytes());
                let seed = *hasher.finalize().as_bytes();
                entry.insert(Thymos::with_seed(
                    ThymosConfig::new(dimension, channels)?,
                    seed,
                )?)
            }
        };
        let (feeling, trajectory) = if interaction.role == ParticipantRole::User {
            let feeling = thymos
                .learn_predicted_next(embedding.as_slice())?
                .unwrap_or(thymos.query(embedding.as_slice())?);
            let trajectory = thymos.step(embedding.as_slice())?;
            (feeling, trajectory)
        } else {
            (
                thymos.query(embedding.as_slice())?,
                TrajectoryStep::default(),
            )
        };

        self.event_frontier
            .entry(event.body.device_id)
            .and_modify(|sequence| *sequence = (*sequence).max(event.body.device_sequence))
            .or_insert(event.body.device_sequence);
        self.projected_events = self.projected_events.saturating_add(1);
        Ok(MemoryReceipt {
            event_id: event.id,
            node_id,
            feeling,
            trajectory,
        })
    }

    pub(crate) fn acknowledge_unembedded(&mut self, event: &SignedEvent) {
        self.event_frontier
            .entry(event.body.device_id)
            .and_modify(|sequence| *sequence = (*sequence).max(event.body.device_sequence))
            .or_insert(event.body.device_sequence);
        self.projected_events = self.projected_events.saturating_add(1);
    }

    pub fn feel(&self, agent: &AgentId, embedding: &Embedding) -> Result<Option<FeelingVector>> {
        self.thymos
            .get(agent)
            .map(|thymos| thymos.query(embedding.as_slice()))
            .transpose()
    }

    pub fn learn_experience(
        &mut self,
        agent: &AgentId,
        context: &Embedding,
        expected: &Embedding,
        actual: &Embedding,
    ) -> Result<FeelingVector> {
        let thymos = self.thymos.get_mut(agent).ok_or(HeartError::NotFound)?;
        thymos.update_from_experience(context.as_slice(), expected.as_slice(), actual.as_slice())
    }

    pub fn promote_thymos(&mut self, agent: AgentId, thymos: Thymos) -> Result<()> {
        if thymos.config.dimension != self.config.model.dimension
            || thymos.config.channels != self.config.thymos_channels
        {
            return Err(HeartError::InvalidInput(
                "promoted Thymos dimensions do not match the cognitive projection".into(),
            ));
        }
        self.thymos.insert(agent, thymos);
        Ok(())
    }

    pub fn inline_text(event: &SignedEvent) -> Option<&str> {
        match &event.body.interaction.content {
            Content::Inline(text) => Some(text),
            Content::ColdBlob(_) | Content::Redacted => None,
        }
    }
}
