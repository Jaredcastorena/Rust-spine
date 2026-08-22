use std::{collections::BTreeMap, str::FromStr};

use proptest::prelude::*;
use spine_heart::{
    AgentId, ClaimExtractor, ContextForest, ContextLeaf, Dcmdb, DcmdbConfig, Embedding, EventId,
    MemoryObservation, NodeId, RehydrateBudget, ThreadId, Thymos, ThymosConfig,
};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        max_shrink_iters: 10_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn hash_ids_roundtrip_and_arbitrary_text_never_panics(bytes in any::<[u8; 32]>(), text in any::<String>()) {
        let id = EventId::from_bytes(bytes);
        prop_assert_eq!(EventId::from_str(&id.to_string()).unwrap(), id);
        let _ = EventId::from_str(&text);
        let _ = NodeId::from_str(&text);
    }

    #[test]
    fn named_ids_enforce_their_trimmed_byte_boundaries(text in any::<String>()) {
        for accepted in [
            AgentId::new(text.clone()).map(|id| id.as_str().to_owned()),
            ThreadId::new(text.clone()).map(|id| id.as_str().to_owned()),
        ].into_iter().flatten() {
            prop_assert!(!accepted.is_empty());
            prop_assert!(accepted.len() <= 256);
            prop_assert_eq!(accepted, text.trim());
        }
    }

    #[test]
    fn normalized_embeddings_are_finite_unit_vectors(values in prop::collection::vec(-1.0e6_f32..1.0e6_f32, 2..65)) {
        let dimension = values.len();
        if let Ok(embedding) = Embedding::normalized(values, dimension) {
            let norm = embedding
                .as_slice()
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            prop_assert!(embedding.as_slice().iter().all(|value| value.is_finite()));
            prop_assert!((norm - 1.0).abs() < 1.0e-4, "norm was {norm}");
        }
    }

    #[test]
    fn claim_extraction_is_deterministic_bounded_and_deduplicated(text in any::<String>()) {
        let extractor = ClaimExtractor::new().unwrap();
        let first = extractor.extract(&text);
        let second = extractor.extract(&text);
        prop_assert_eq!(&first, &second);
        prop_assert!(first.iter().all(|claim| claim.excerpt.chars().count() <= 120));
        for (index, claim) in first.iter().enumerate() {
            prop_assert!(!first[..index].contains(claim));
        }
    }

    #[test]
    fn dcmdb_random_update_query_and_maintenance_preserve_invariants(
        observations in prop::collection::vec(
            ((-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32), 0_u16..2_000_u16),
            1..96,
        ),
        query in ((-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32)),
    ) {
        let mut config = DcmdbConfig::dense(3);
        config.max_merges_per_pass = Some(8);
        config.dream_walks = 4;
        config.dream_walk_length = 4;
        let mut memory = Dcmdb::new(config).unwrap();
        for (index, (x, y, z, time)) in observations.into_iter().enumerate() {
            let mut event = [0_u8; 32];
            event[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let result = memory.update(MemoryObservation {
                event_id: EventId::from_bytes(event),
                vector: vec![x, y, z],
                time: f64::from(time),
                source: Some(format!("source-{}", index % 3)),
                metadata: BTreeMap::new(),
            });
            if x != 0.0 || y != 0.0 || z != 0.0 {
                prop_assert!(result.is_ok());
            }
            prop_assert!(memory.check_invariants().is_empty());
        }
        let hits = memory.query(&[query.0, query.1, query.2], memory.logical_time(), 12);
        if query.0 != 0.0 || query.1 != 0.0 || query.2 != 0.0 {
            for hit in hits.unwrap() {
                prop_assert!(hit.score.is_finite());
                prop_assert!(hit.semantic_score.is_finite());
                prop_assert!(hit.graph_score.is_finite());
            }
        }
        memory.maintain(memory.logical_time() + 1.0, 4);
        prop_assert!(memory.check_invariants().is_empty());
        for node in memory.nodes.values() {
            prop_assert!(node.centroid.iter().all(|value| value.is_finite()));
            prop_assert!(node.weight.is_finite());
            prop_assert!(node.kappa.is_finite());
            prop_assert!(node.tau.is_finite());
        }
    }

    #[test]
    fn thymos_random_updates_and_trajectory_remain_finite(
        inputs in prop::collection::vec(
            ((-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32), (-10.0_f32..10.0_f32)),
            1..128,
        ),
        eligibility in prop::array::uniform4(-2.0_f32..2.0_f32),
    ) {
        let config = ThymosConfig::new(3, 4).unwrap();
        let mut thymos = Thymos::with_seed(config, [7; 32]).unwrap();
        for (x, y, z) in inputs {
            let input = [x, y, z];
            thymos.update(&input, &eligibility).unwrap();
            let trajectory = thymos.step(&input).unwrap();
            let feeling = thymos.query(&input).unwrap();
            prop_assert!(trajectory.surprise.is_finite());
            prop_assert!(trajectory.speed.is_finite());
            prop_assert!(trajectory.heading_norm.is_finite());
            prop_assert!(feeling.raw.iter().all(|value| value.is_finite()));
            prop_assert!(feeling.activated.iter().all(|value| value.is_finite()));
            prop_assert!(feeling.valence.is_finite());
            prop_assert!(feeling.arousal.is_finite());
            prop_assert!(thymos.tensor().iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn triangle_compaction_and_rehydration_respect_all_budgets(
        vectors in prop::collection::vec(
            ((-1.0_f32..1.0_f32), (-1.0_f32..1.0_f32), (-1.0_f32..1.0_f32)),
            1..32,
        ),
        target_roots in 1_usize..8,
        max_depth in 0_u32..8,
        max_fanout in 0_usize..8,
        max_nodes in 0_usize..16,
        max_tokens in 0_usize..64,
    ) {
        let mut config = DcmdbConfig::dense(3);
        config.theta_similarity = 0.999_999;
        let mut memory = Dcmdb::new(config).unwrap();
        let mut leaves = Vec::new();
        for (index, (x, y, z)) in vectors.into_iter().enumerate() {
            let mut event = [0_u8; 32];
            event[..8].copy_from_slice(&(index as u64).to_be_bytes());
            if let Ok(node_id) = memory.update(MemoryObservation {
                event_id: EventId::from_bytes(event),
                vector: vec![x, y, z],
                time: index as f64,
                source: None,
                metadata: BTreeMap::from([("token_count".into(), ((index % 9) + 1).to_string())]),
            }) {
                leaves.push(ContextLeaf { node_id, chronology: index as u64 });
            }
        }
        let mut forest = ContextForest::default();
        let roots = forest.compact(leaves, &memory, target_roots);
        prop_assert!(roots.is_ok(), "compaction failed: {roots:?}");
        forest.verify(&memory).unwrap();
        let budget = RehydrateBudget { max_depth, max_fanout, max_nodes, max_tokens };
        for root in forest.roots.clone() {
            let result = forest.rehydrate(root.handle, None, &memory, budget).unwrap();
            prop_assert!(result.coordinates.len() <= max_nodes);
            prop_assert!(result.alternate_routes.len() <= max_fanout);
            prop_assert!(result.consumed_tokens <= max_tokens);
            prop_assert!(result.coordinates.iter().all(|item| item.depth <= max_depth));
        }
    }
}

#[test]
fn malformed_thymos_configs_are_rejected_before_allocation_or_updates() {
    macro_rules! reject {
        ($field:ident, $value:expr) => {{
            let mut config = ThymosConfig::new(3, 4).unwrap();
            config.$field = $value;
            assert!(
                Thymos::with_seed(config, [3; 32]).is_err(),
                stringify!($field)
            );
        }};
    }

    reject!(learning_rate, f32::NAN);
    reject!(decay, -0.1);
    reject!(decay, 1.1);
    reject!(valence_temperature, 0.0);
    reject!(trajectory_alpha, f32::INFINITY);
    reject!(eps, 0.0);

    let mut oversized = ThymosConfig::new(3, 4).unwrap();
    oversized.dimension = usize::MAX;
    oversized.channels = 2;
    assert!(Thymos::with_seed(oversized, [3; 32]).is_err());
}
