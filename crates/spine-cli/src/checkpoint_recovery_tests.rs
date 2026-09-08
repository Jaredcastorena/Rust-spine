use super::*;
use spine_heart::{HeartError, ModelManifest};

struct TestEncoder {
    manifest: ModelManifest,
    dimension: usize,
    fail: bool,
}

impl TestEncoder {
    fn new() -> Self {
        Self {
            manifest: ModelManifest {
                schema: 1,
                model_name: "checkpoint-recovery-test".into(),
                artifact_hash: [21; 32],
                tokenizer_hash: [22; 32],
                dimension: 3,
                normalized: true,
                quantization: None,
            },
            dimension: 3,
            fail: false,
        }
    }
}

impl SemanticEncoder for TestEncoder {
    fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    fn encode(&self, _: &str) -> spine_heart::Result<Embedding> {
        if self.fail {
            return Err(HeartError::InvalidInput("test encoder unavailable".into()));
        }
        Embedding::normalized(vec![1.0; self.dimension], self.dimension)
    }
}

fn seed(heart: &SpineHeart, encoder: &TestEncoder) -> PersistedHarnessCheckpoint {
    let checkpoint = HarnessCheckpoint {
        schema: 1,
        harness_id: "recovery-test".into(),
        messages: vec![
            Message::new(MessageRole::System, "system"),
            Message::new(MessageRole::User, "continue"),
            Message::new(MessageRole::Assistant, "paused"),
        ],
        completed_tool_calls: 0,
        completed_tool_rounds: 0,
        completed_action_calls: 0,
        pending_task: "continue".into(),
        host_plan: None,
        policy: Some(HarnessPolicy::default()),
    };
    let (receipt, _) = heart
        .commit_embedded(
            checkpoint
                .to_interaction(
                    AgentId::new("main").unwrap(),
                    ThreadId::new("test").unwrap(),
                )
                .unwrap(),
            encoder.encode("checkpoint").unwrap(),
        )
        .unwrap();
    PersistedHarnessCheckpoint {
        checkpoint,
        event_id: receipt.event.id,
    }
}

fn setup() -> (
    tempfile::TempDir,
    SpineHeart,
    TestEncoder,
    Option<PersistedHarnessCheckpoint>,
) {
    let temp = tempfile::tempdir().unwrap();
    let created = SpineHeart::create(
        HeartConfig::new(temp.path().join("checkpoint.spine")),
        "test-pass",
    )
    .unwrap();
    let encoder = TestEncoder::new();
    created
        .heart
        .initialize_cognition(CognitiveConfig::new(1, encoder.manifest.clone(), 2).unwrap())
        .unwrap();
    let checkpoint = seed(&created.heart, &encoder);
    (temp, created.heart, encoder, Some(checkpoint))
}

#[test]
fn durable_consumption_after_projection_failure_resumes_once_and_opens_memory_circuit() {
    let (_temp, heart, mut encoder, mut checkpoint) = setup();
    let checkpoint_id = checkpoint.as_ref().unwrap().event_id;
    // The event is durably signed before projection observation rejects this dimension.
    encoder.dimension = 2;
    let mut breaker = CircuitBreaker::default();
    let agent = AgentId::new("main").unwrap();
    let thread = ThreadId::new("test").unwrap();
    let resumed = prepare_checkpoint_resume(
        &heart,
        &encoder,
        &agent,
        &thread,
        &mut checkpoint,
        &mut breaker,
    )
    .unwrap();
    assert_eq!(resumed.pending_task, "continue");
    assert!(checkpoint.is_none());
    assert_eq!(heart.stats().unwrap().events, 2);
    assert!(!heart.cognition_is_current().unwrap());
    assert!(
        exact_checkpoint_consumed(
            &heart.events_canonical().unwrap(),
            &agent,
            &thread,
            checkpoint_id
        )
        .unwrap()
    );
    assert_eq!(
        breaker.status(ResilienceChannel::Dcmdb).state,
        resilience::BreakerState::Open
    );
    assert!(
        prepare_checkpoint_resume(
            &heart,
            &encoder,
            &agent,
            &thread,
            &mut checkpoint,
            &mut breaker
        )
        .is_err()
    );
    assert_eq!(heart.stats().unwrap().events, 2);
    assert_eq!(
        discover_persisted_checkpoint(&heart.events_canonical().unwrap(), &agent, &thread),
        CheckpointDiscovery::None
    );
}

#[test]
fn absent_consumption_after_encoder_failure_restores_the_same_checkpoint() {
    let (_temp, heart, mut encoder, mut checkpoint) = setup();
    let original = checkpoint.clone();
    encoder.fail = true;
    let mut breaker = CircuitBreaker::default();
    let error = prepare_checkpoint_resume(
        &heart,
        &encoder,
        &AgentId::new("main").unwrap(),
        &ThreadId::new("test").unwrap(),
        &mut checkpoint,
        &mut breaker,
    )
    .unwrap_err();
    assert!(error.contains("no consumption marker was committed"));
    assert_eq!(checkpoint, original);
    assert_eq!(heart.stats().unwrap().events, 1);
}

#[test]
fn malformed_canonical_consumption_disables_local_resume() {
    let (_temp, heart, encoder, mut checkpoint) = setup();
    let agent = AgentId::new("main").unwrap();
    let thread = ThreadId::new("test").unwrap();
    let mut marker = checkpoint_consumption_interaction(
        checkpoint.as_ref().unwrap(),
        agent.clone(),
        thread.clone(),
    )
    .unwrap();
    marker.content = Content::Inline("{broken".into());
    heart.commit_interaction(marker).unwrap();
    let mut breaker = CircuitBreaker::default();
    let error = prepare_checkpoint_resume(
        &heart,
        &encoder,
        &agent,
        &thread,
        &mut checkpoint,
        &mut breaker,
    )
    .unwrap_err();
    assert!(error.contains("indeterminate"));
    assert!(!error.contains("checkpoint remains available"));
    assert!(checkpoint.is_none());
    assert_eq!(heart.stats().unwrap().events, 2);
}

#[test]
fn consumption_of_another_exact_id_does_not_consume_the_selected_checkpoint() {
    let (_temp, heart, encoder, mut checkpoint) = setup();
    let agent = AgentId::new("main").unwrap();
    let thread = ThreadId::new("test").unwrap();
    let selected_id = checkpoint.as_ref().unwrap().event_id;
    let other = seed(&heart, &encoder);
    heart
        .commit_interaction(
            checkpoint_consumption_interaction(&other, agent.clone(), thread.clone()).unwrap(),
        )
        .unwrap();
    let mut breaker = CircuitBreaker::default();
    let error = prepare_checkpoint_resume(
        &heart,
        &encoder,
        &agent,
        &thread,
        &mut checkpoint,
        &mut breaker,
    )
    .unwrap_err();
    assert!(error.contains("no consumption marker was committed"));
    assert_eq!(checkpoint.as_ref().unwrap().event_id, selected_id);
    assert!(
        !exact_checkpoint_consumed(
            &heart.events_canonical().unwrap(),
            &agent,
            &thread,
            selected_id
        )
        .unwrap()
    );
}
