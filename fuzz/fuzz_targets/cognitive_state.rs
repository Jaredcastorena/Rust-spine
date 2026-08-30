#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use spine_heart::{
    ContextForest, ContextLeaf, Dcmdb, DcmdbConfig, EventId, MemoryObservation, RehydrateBudget,
    Thymos, ThymosConfig,
};

fuzz_target!(|data: &[u8]| {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.999_999;
    config.max_merges_per_pass = Some(8);
    config.dream_walks = 3;
    config.dream_walk_length = 3;
    let mut memory = Dcmdb::new(config).expect("fixed DCMDb config is valid");
    let mut thymos = Thymos::with_seed(
        ThymosConfig::new(3, 4).expect("fixed Thymos config is valid"),
        [11; 32],
    )
    .expect("fixed Thymos seed initializes");
    let mut leaves = Vec::new();

    for (index, chunk) in data.chunks(8).take(64).enumerate() {
        let component =
            |offset: usize| f32::from(chunk.get(offset).copied().unwrap_or(128)) / 127.5 - 1.0;
        let vector = [component(0), component(1), component(2)];
        let mut event = [0_u8; 32];
        event[..8].copy_from_slice(&(index as u64).to_be_bytes());
        if let Ok(node_id) = memory.update(MemoryObservation {
            event_id: EventId::from_bytes(event),
            vector: vector.to_vec(),
            time: f64::from(chunk.get(3).copied().unwrap_or_default()),
            source: Some(format!(
                "source-{}",
                chunk.get(4).copied().unwrap_or_default() % 4
            )),
            metadata: BTreeMap::from([(
                "token_count".into(),
                (usize::from(chunk.get(5).copied().unwrap_or_default()) + 1).to_string(),
            )]),
        }) {
            leaves.push(ContextLeaf {
                node_id,
                chronology: u64::from(chunk.get(6).copied().unwrap_or_default()),
            });
        }
        let eligibility = [component(4), component(5), component(6), component(7)];
        thymos
            .update(&vector, &eligibility)
            .expect("finite shaped update");
        let trajectory = thymos.step(&vector).expect("finite shaped trajectory");
        assert!(trajectory.surprise.is_finite());
        assert!(trajectory.speed.is_finite());
        assert!(trajectory.heading_norm.is_finite());
    }

    assert!(memory.check_invariants().is_empty());
    memory.maintain(memory.logical_time() + 1.0, 3);
    assert!(memory.check_invariants().is_empty());
    assert!(memory.nodes.values().all(|node| {
        node.centroid.iter().all(|value| value.is_finite())
            && node.weight.is_finite()
            && node.kappa.is_finite()
            && node.tau.is_finite()
    }));
    assert!(thymos.tensor().iter().all(|value| value.is_finite()));

    let mut forest = ContextForest::default();
    forest
        .compact(leaves, &memory, 1)
        .expect("valid memory leaves compact");
    forest.verify(&memory).expect("compacted forest verifies");
    for root in forest.roots.clone() {
        let result = forest
            .rehydrate(
                root.handle,
                None,
                &memory,
                RehydrateBudget {
                    max_depth: u32::from(data.first().copied().unwrap_or_default() % 8),
                    max_fanout: usize::from(data.get(1).copied().unwrap_or_default() % 8),
                    max_nodes: usize::from(data.get(2).copied().unwrap_or_default() % 16),
                    max_tokens: usize::from(data.get(3).copied().unwrap_or_default()),
                },
            )
            .expect("verified root rehydrates");
        assert!(result.consumed_tokens <= usize::from(data.get(3).copied().unwrap_or_default()));
    }
});
