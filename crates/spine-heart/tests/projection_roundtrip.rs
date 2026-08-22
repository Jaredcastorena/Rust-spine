use spine_heart::{
    AgentId, CognitiveConfig, Content, Embedding, EventKind, HeartConfig, InteractionInput,
    KeySource, ModelManifest, ParticipantRole, Provenance, Result, SemanticEncoder, SpineHeart,
    ThreadId,
};

#[derive(Clone)]
struct TinyEncoder {
    manifest: ModelManifest,
}

impl TinyEncoder {
    fn new() -> Self {
        Self {
            manifest: ModelManifest {
                schema: 1,
                model_name: "test-hash-encoder".into(),
                artifact_hash: [3; 32],
                tokenizer_hash: [4; 32],
                dimension: 3,
                normalized: true,
                quantization: None,
            },
        }
    }
}

impl SemanticEncoder for TinyEncoder {
    fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn encode(&self, text: &str) -> Result<Embedding> {
        let bytes = blake3::hash(text.as_bytes());
        Embedding::normalized(
            vec![
                f32::from(bytes.as_bytes()[0]) + 1.0,
                f32::from(bytes.as_bytes()[1]) + 1.0,
                f32::from(bytes.as_bytes()[2]) + 1.0,
            ],
            3,
        )
    }
}

fn interaction(text: &str) -> InteractionInput {
    InteractionInput {
        agent_id: AgentId::new("main").unwrap(),
        thread_id: ThreadId::new("projection-test").unwrap(),
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

#[test]
fn encrypted_cognition_survives_reopen_and_detects_staleness() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("cognition.spine");
    let encoder = TinyEncoder::new();
    let created = SpineHeart::create(HeartConfig::new(&path), "projection-pass").unwrap();
    created
        .heart
        .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
        .unwrap();
    let embedding = encoder.encode("the triangle keeps addresses").unwrap();
    let (_, memory) = created
        .heart
        .commit_embedded(
            interaction("the triangle keeps addresses"),
            embedding.clone(),
        )
        .unwrap();
    assert!(created.heart.cognition_is_current().unwrap());
    assert_eq!(
        created
            .heart
            .feel(&AgentId::new("main").unwrap(), &embedding)
            .unwrap()
            .unwrap()
            .raw
            .len(),
        2
    );
    assert_eq!(
        created.heart.recall(&embedding, 0.0, 1).unwrap()[0].node_id,
        memory.node_id
    );
    drop(created);

    let reopened = SpineHeart::open(
        HeartConfig::new(&path),
        KeySource::Passphrase("projection-pass".into()),
    )
    .unwrap();
    assert_eq!(reopened.cognition().unwrap().unwrap().projected_events, 1);
    reopened
        .commit_interaction(interaction("unprojected event"))
        .unwrap();
    assert!(!reopened.cognition_is_current().unwrap());
    let rebuilt = reopened
        .rebuild_cognition(
            CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap(),
            &encoder,
        )
        .unwrap();
    assert_eq!(rebuilt.projected_events, 2);
    assert!(reopened.cognition_is_current().unwrap());
}

#[test]
fn embedded_batch_commits_every_event_and_projection_together() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("batch.spine");
    let encoder = TinyEncoder::new();
    let created = SpineHeart::create(HeartConfig::new(&path), "batch-pass").unwrap();
    created
        .heart
        .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
        .unwrap();

    let texts = [
        "first bulk memory",
        "second bulk memory",
        "third bulk memory",
    ];
    let items = texts
        .iter()
        .map(|text| (interaction(text), encoder.encode(text).unwrap()))
        .collect();
    let receipts = created.heart.commit_embedded_batch(items).unwrap();

    assert_eq!(receipts.len(), texts.len());
    assert!(receipts.iter().all(|(commit, _)| commit.inserted));
    assert_eq!(created.heart.stats().unwrap().events, texts.len() as u64);
    assert_eq!(
        created.heart.cognition().unwrap().unwrap().projected_events,
        texts.len() as u64
    );
    assert!(created.heart.cognition_is_current().unwrap());
}
