use std::collections::BTreeMap;

use spine_heart::{
    ContextForest, ContextHandle, ContextLeaf, Dcmdb, DcmdbConfig, Embedding, EventId,
    MemoryObservation, RehydrateBudget,
};

fn add(memory: &mut Dcmdb, byte: u8, vector: [f32; 3], time: f64) -> spine_heart::NodeId {
    memory
        .update(MemoryObservation {
            event_id: EventId::from_bytes([byte; 32]),
            vector: vector.to_vec(),
            time,
            source: None,
            metadata: BTreeMap::new(),
        })
        .unwrap()
}

#[test]
fn compaction_folds_only_pairs_with_a_real_coherent_apex() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.9999;
    config.tension_promote_count = 99;
    let mut memory = Dcmdb::new(config).unwrap();
    let a = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let b = add(&mut memory, 2, [0.99, 0.1, 0.0], 2.0);
    let c = add(&mut memory, 3, [0.0, 1.0, 0.0], 3.0);
    assert_ne!(a, b);
    assert!(memory.coherent_apex(a, b, 0.8).is_some());

    let mut forest = ContextForest::default();
    let roots = forest
        .compact(
            [
                ContextLeaf {
                    node_id: a,
                    chronology: 1,
                },
                ContextLeaf {
                    node_id: b,
                    chronology: 2,
                },
                ContextLeaf {
                    node_id: c,
                    chronology: 3,
                },
            ],
            &memory,
            1,
        )
        .unwrap();

    // A and B fold. C stays separate because no real DCMDb coordinate is
    // coherent with both clusters; no scaffold node is manufactured.
    assert_eq!(roots.len(), 2);
    assert_eq!(forest.triangles.len(), 1);
    let triangle = forest.triangles.values().next().unwrap();
    assert!(memory.node(triangle.apex).is_some());
    assert!(triangle.apex == a || triangle.apex == b);
    assert_eq!(memory.nodes.len(), 3);
    forest.verify(&memory).unwrap();
}

#[test]
fn rehydration_reads_apex_and_preferred_path_but_keeps_alternate_route() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.9999;
    let mut memory = Dcmdb::new(config).unwrap();
    let a = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let b = add(&mut memory, 2, [0.99, 0.1, 0.0], 2.0);
    let mut forest = ContextForest::default();
    forest
        .compact(
            [
                ContextLeaf {
                    node_id: a,
                    chronology: 1,
                },
                ContextLeaf {
                    node_id: b,
                    chronology: 2,
                },
            ],
            &memory,
            1,
        )
        .unwrap();
    let root = forest.roots[0].handle;
    assert!(matches!(root, ContextHandle::Triangle(_)));
    let query = Embedding::normalized(vec![0.99, 0.1, 0.0], 3).unwrap();
    let result = forest
        .rehydrate(
            root,
            Some(&query),
            &memory,
            RehydrateBudget {
                max_depth: 4,
                max_fanout: 2,
                max_nodes: 4,
                max_tokens: 4,
            },
        )
        .unwrap();
    assert!(!result.coordinates.is_empty());
    assert_eq!(result.alternate_routes.len(), 1);
    assert!(
        result
            .coordinates
            .iter()
            .all(|coordinate| memory.node(coordinate.node_id).is_some())
    );
}

#[test]
fn duplicate_node_handles_coalesce_without_self_referential_triangles() {
    let mut memory = Dcmdb::new(DcmdbConfig::dense(3)).unwrap();
    let first = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let second = add(&mut memory, 2, [1.0, 0.0, 0.0], 2.0);
    assert_eq!(first, second);
    let mut forest = ContextForest::default();

    let roots = forest
        .compact(
            [
                ContextLeaf {
                    node_id: first,
                    chronology: 2,
                },
                ContextLeaf {
                    node_id: second,
                    chronology: 9,
                },
            ],
            &memory,
            1,
        )
        .unwrap();

    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].handle, ContextHandle::Node(first));
    assert_eq!((roots[0].chronology_start, roots[0].chronology_end), (2, 9));
    assert!(forest.triangles.is_empty());
}

#[test]
fn nested_triangle_identity_uses_the_full_overlapping_chronology_range() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.999_999;
    let mut memory = Dcmdb::new(config).unwrap();
    let a = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let b = add(&mut memory, 2, [0.999, 0.02, 0.0], 2.0);
    let c = add(&mut memory, 3, [0.98, 0.19, 0.0], 3.0);
    let mut forest = ContextForest::default();

    let roots = forest
        .compact(
            [
                ContextLeaf {
                    node_id: a,
                    chronology: 10,
                },
                ContextLeaf {
                    node_id: b,
                    chronology: 20,
                },
                ContextLeaf {
                    node_id: c,
                    chronology: 15,
                },
            ],
            &memory,
            1,
        )
        .unwrap();

    assert_eq!(roots.len(), 1);
    assert_eq!(
        (roots[0].chronology_start, roots[0].chronology_end),
        (10, 20)
    );
    forest.verify(&memory).unwrap();
}
