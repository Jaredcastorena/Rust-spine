use std::collections::{BTreeMap, BTreeSet};

use spine_heart::{Dcmdb, DcmdbConfig, EventId, MemoryObservation};

fn add(memory: &mut Dcmdb, byte: u8, vector: [f32; 3], time: f64) -> spine_heart::NodeId {
    memory
        .update(MemoryObservation {
            event_id: EventId::from_bytes([byte; 32]),
            vector: vector.to_vec(),
            time,
            source: Some("test".into()),
            metadata: BTreeMap::new(),
        })
        .unwrap()
}

#[test]
fn consolidation_preserves_hierarchy_events_and_internal_invariants() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.9999;
    config.merge_angle = 15_f32.to_radians();
    config.merge_penalty = 0.0;
    let mut memory = Dcmdb::new(config).unwrap();
    let first = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let second = add(&mut memory, 2, [0.99, 0.1, 0.0], 2.0);
    assert_ne!(first, second);
    assert_eq!(memory.consolidate_pass(), 1);
    assert_eq!(memory.nodes.len(), 1);
    assert_eq!(memory.absorbed.len(), 1);
    let active = memory.nodes.values().next().unwrap();
    assert_eq!(active.event_ids.len(), 2);
    assert_eq!(active.children.len(), 1);
    assert!(memory.check_invariants().is_empty());
}

#[test]
fn absorbed_subtree_expansion_uses_depth_and_node_event_budgets() {
    let mut memory = Dcmdb::new(DcmdbConfig::dense(3)).unwrap();
    let ids = [
        add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0),
        add(&mut memory, 2, [0.0, 1.0, 0.0], 2.0),
        add(&mut memory, 3, [0.0, 0.0, 1.0], 3.0),
        add(&mut memory, 4, [-1.0, 0.0, 0.0], 4.0),
    ];
    // Model the same aggregated-event contract as real DCMDb consolidation,
    // retaining an absorbed chain to make the Python depth oracle explicit.
    for i in (0..3).rev() {
        let inherited = memory.nodes[&ids[i + 1]].event_ids.clone();
        let node = memory.nodes.get_mut(&ids[i]).unwrap();
        node.children.push(ids[i + 1]);
        node.event_ids.extend(inherited);
    }
    for id in &ids[1..] {
        let node = memory.nodes.remove(id).unwrap();
        memory.absorbed.insert(*id, node);
    }
    let events = |count: u8| {
        (1..=count)
            .map(|n| EventId::from_bytes([n; 32]))
            .collect::<Vec<_>>()
    };
    assert_eq!(memory.subtree_event_ids(ids[0], 1, 32, 20), events(2));
    assert_eq!(memory.subtree_event_ids(ids[0], 2, 32, 20), events(3));
    assert_eq!(memory.subtree_event_ids(ids[0], 3, 32, 20), events(4));
    assert_eq!(memory.subtree_event_ids(ids[0], 3, 2, 20), events(2));
    assert_eq!(memory.subtree_event_ids(ids[0], 3, 32, 1), events(1));
    assert!(memory.subtree_event_ids(ids[0], 3, 0, 20).is_empty());
    // Repeated/cyclic coordinates cannot trigger unbounded traversal.
    memory
        .absorbed
        .get_mut(&ids[3])
        .unwrap()
        .children
        .push(ids[0]);
    assert!(memory.subtree_event_ids(ids[0], usize::MAX, 32, 2).len() <= 2);
}

#[test]
fn pruning_cleans_all_graph_and_count_references() {
    let mut config = DcmdbConfig::dense(3);
    config.prune_weight_threshold = 2.0;
    let mut memory = Dcmdb::new(config).unwrap();
    add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    add(&mut memory, 2, [0.0, 1.0, 0.0], 2.0);
    assert_eq!(memory.prune_pass(2.0), 2);
    assert!(memory.nodes.is_empty());
    assert!(memory.graph.is_empty());
    assert!(memory.check_invariants().is_empty());
}

#[test]
fn pruning_preserves_nodes_addressed_by_context_triangles() {
    let mut config = DcmdbConfig::dense(3);
    config.prune_weight_threshold = 2.0;
    let mut memory = Dcmdb::new(config).unwrap();
    let protected = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let disposable = add(&mut memory, 2, [0.0, 1.0, 0.0], 2.0);

    assert_eq!(
        memory.prune_pass_protected(2.0, &BTreeSet::from([protected])),
        1
    );
    assert!(memory.node(protected).is_some());
    assert!(memory.node(disposable).is_none());
    assert!(memory.check_invariants().is_empty());
}

#[test]
fn mmr_can_choose_diversity_over_a_near_duplicate() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.9999;
    config.semantic_weight = 1.0;
    config.graph_weight = 0.0;
    config.freshness_weight = 0.0;
    config.confidence_weight = 0.0;
    config.mmr_weight = 0.8;
    let mut memory = Dcmdb::new(config).unwrap();
    let first = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let duplicate = add(&mut memory, 2, [0.99, 0.1, 0.0], 2.0);
    let diverse = add(&mut memory, 3, [0.0, 1.0, 0.0], 3.0);
    let hits = memory.query(&[1.0, 0.0, 0.0], 3.0, 2).unwrap();
    assert_eq!(hits[0].node_id, first);
    assert_eq!(hits[1].node_id, diverse);
    assert_ne!(hits[1].node_id, duplicate);
}

#[test]
fn dreaming_is_deterministic_and_reactivates_walked_nodes() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.9999;
    config.merge_angle = 1_f32.to_radians();
    config.dream_walks = 4;
    config.dream_walk_length = 3;
    let mut memory = Dcmdb::new(config).unwrap();
    let first = add(&mut memory, 1, [1.0, 0.0, 0.0], 1.0);
    let second = add(&mut memory, 2, [0.0, 1.0, 0.0], 2.0);
    memory.graph.entry(first).or_default().insert(second, 1.0);
    memory.graph.entry(second).or_default().insert(first, 1.0);
    let mut replica = memory.clone();
    let first_tau = memory.nodes[&first].tau;

    let report = memory.dream_pass(3.0);
    let replica_report = replica.dream_pass(3.0);

    assert_eq!(report, replica_report);
    assert_eq!(memory, replica);
    assert_eq!(report.walks_completed, 4);
    assert_eq!(report.nodes_reactivated, 2);
    assert!(memory.nodes[&first].tau > first_tau);
    assert!(memory.check_invariants().is_empty());
}

#[test]
fn invalid_configuration_is_rejected_before_it_can_create_nan_state() {
    macro_rules! reject {
        ($field:ident, $value:expr) => {{
            let mut config = DcmdbConfig::dense(3);
            config.$field = $value;
            assert!(Dcmdb::new(config).is_err(), stringify!($field));
        }};
    }

    reject!(theta_similarity, f32::NAN);
    reject!(theta_similarity, -0.1);
    reject!(gamma0, f32::INFINITY);
    reject!(tau0, -1.0);
    reject!(tau_min, 101.0);
    reject!(dream_temperature, 0.0);
    reject!(pmi_learning_rate, 1.1);
    reject!(pmi_angle_max, -0.1);
    reject!(merge_angle, std::f32::consts::PI);
    reject!(pagerank_restart, -0.1);
    reject!(mmr_weight, 1.1);
    reject!(split_probability, f32::NAN);
    reject!(eps, 0.0);

    let mut config = DcmdbConfig::dense(3);
    config.neighbors = 0;
    assert!(Dcmdb::new(config).is_err());
}
