use spine_heart::{
    AgentId, CognitiveConfig, Content, ContextLeaf, Embedding, EventKind, HeartConfig, HeartError,
    InteractionInput, KeySource, ModelManifest, ParticipantRole, Provenance, Result,
    SemanticEncoder, SpineHeart, ThreadId,
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
fn suffix_catch_up_preserves_projection_only_learning_and_context() {
    let temp = tempfile::tempdir().unwrap();
    let encoder = TinyEncoder::new();
    let created = SpineHeart::create(
        HeartConfig::new(temp.path().join("suffix.spine")),
        "suffix-pass",
    )
    .unwrap();
    created
        .heart
        .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
        .unwrap();
    let (first_event, first_memory) = created
        .heart
        .commit_embedded(
            interaction("first context coordinate"),
            Embedding::normalized(vec![1.0, 0.0, 0.0], 3).unwrap(),
        )
        .unwrap();
    let (second_event, second_memory) = created
        .heart
        .commit_embedded(
            interaction("second context coordinate"),
            Embedding::normalized(vec![0.82, 0.57, 0.0], 3).unwrap(),
        )
        .unwrap();
    created
        .heart
        .compact_context(
            [
                ContextLeaf {
                    node_id: first_memory.node_id,
                    chronology: first_event.event.body.device_sequence,
                },
                ContextLeaf {
                    node_id: second_memory.node_id,
                    chronology: second_event.event.body.device_sequence,
                },
            ],
            1,
        )
        .unwrap();
    let risk_context = Embedding::normalized(vec![0.0, 0.0, 1.0], 3).unwrap();
    created
        .heart
        .update_risk(
            &AgentId::new("main").unwrap(),
            &risk_context,
            &[0.8, 0.2, 0.1, 0.3],
            0.9,
        )
        .unwrap();
    let learned = created.heart.cognition().unwrap().unwrap();
    assert!(!learned.triangles.roots.is_empty());
    assert!(!learned.triangles.triangles.is_empty());

    created
        .heart
        .commit_interaction(interaction("canonical suffix event"))
        .unwrap();
    let caught_up = created.heart.catch_up_cognition(&encoder).unwrap();

    assert_eq!(caught_up.projected_events, learned.projected_events + 1);
    assert_eq!(caught_up.risk, learned.risk);
    assert_eq!(caught_up.triangles, learned.triangles);
    assert!(created.heart.cognition_is_current().unwrap());
}

#[test]
fn catch_up_rejects_imports_that_reorder_the_projected_prefix() {
    let temp = tempfile::tempdir().unwrap();
    let first_path = temp.path().join("first.spine");
    let second_path = temp.path().join("second.spine");
    let created = SpineHeart::create(HeartConfig::new(&first_path), "first-pass").unwrap();
    let recovery_phrase = created.recovery_phrase.expose().to_owned();
    let first = created.heart;
    let second = SpineHeart::create_replica(
        HeartConfig::new(&second_path),
        &recovery_phrase,
        "second-pass",
    )
    .unwrap();
    let encoder = TinyEncoder::new();

    second
        .commit_interaction(interaction("older offline event"))
        .unwrap();
    first
        .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
        .unwrap();
    first
        .commit_embedded(
            interaction("already projected local event"),
            encoder.encode("already projected local event").unwrap(),
        )
        .unwrap();
    let before = first.cognition().unwrap().unwrap();
    let delta = second
        .export_delta(&first.sync_frontier().unwrap())
        .unwrap();
    assert_eq!(first.import_delta(delta).unwrap().inserted, 1);
    assert!(!first.cognition_is_current().unwrap());

    assert!(matches!(
        first.catch_up_cognition(&encoder),
        Err(HeartError::ProjectionStale)
    ));
    assert_eq!(first.cognition().unwrap().unwrap(), before);
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
