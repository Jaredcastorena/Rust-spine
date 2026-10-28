use spine_heart::{CorpusMode, Dcmdb, DcmdbConfig, DcmdbPolicy, EventId, MemoryObservation};
use std::collections::{BTreeMap, BTreeSet};

fn vector(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn close(actual: f32, expected: f64, label: &str) {
    let tolerance = 2e-5 + expected.abs() as f32 * 0.003;
    assert!(
        (actual - expected as f32).abs() <= tolerance,
        "{label}: {actual} != {expected}"
    );
}

#[test]
fn update_pmi_variants_source_priors_and_mmr_match_python() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dcmdb-python.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let policy: DcmdbPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
        let mut config = DcmdbConfig::dense(3);
        config.tension_promote_count = 99;
        config.graph_weight = 0.2;
        config.semantic_weight = 0.7;
        config.mmr_weight = 0.25;
        let mut db = Dcmdb::new(config).unwrap();
        for (index, input) in fixture["observations"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            db.update_with_policy(
                MemoryObservation {
                    event_id: EventId::from_bytes([index as u8; 32]),
                    vector: vector(input),
                    time: (index + 1) as f64,
                    source: Some(if index % 3 == 0 { "archive" } else { "live" }.into()),
                    metadata: BTreeMap::new(),
                },
                &policy,
            )
            .unwrap();
        }
        let expected_nodes = case["nodes"].as_array().unwrap();
        assert_eq!(db.nodes.len(), expected_nodes.len(), "policy {policy:?}");
        let mut mapping = BTreeMap::new();
        let mut matched = BTreeSet::new();
        for expected in expected_nodes {
            let centroid = vector(&expected["centroid"]);
            let actual = db
                .nodes
                .values()
                .filter(|n| !matched.contains(&n.id))
                .min_by(|a, b| {
                    let distance = |v: &[f32]| {
                        v.iter()
                            .zip(&centroid)
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                    };
                    distance(&a.centroid).total_cmp(&distance(&b.centroid))
                })
                .unwrap();
            mapping.insert(expected["id"].as_u64().unwrap(), actual.id);
            matched.insert(actual.id);
            for (name, values) in [
                ("centroid", &actual.centroid),
                ("sufficient_sum", &actual.sufficient_sum),
            ] {
                for (a, b) in values.iter().zip(expected[name].as_array().unwrap()) {
                    close(*a, b.as_f64().unwrap(), name);
                }
            }
            for (name, value) in [
                ("effective_count", actual.effective_count),
                ("weight", actual.weight),
                ("kappa", actual.kappa),
                ("confidence", actual.confidence),
                ("tau", actual.tau),
                ("visits", actual.visits),
            ] {
                close(value, expected[name].as_f64().unwrap(), name);
            }
            assert_eq!(
                actual.tension.is_some(),
                expected["tensioned"].as_bool().unwrap()
            );
            for (source, count) in expected["source_counts"].as_object().unwrap() {
                close(
                    actual.source_counts[source],
                    count.as_f64().unwrap(),
                    source,
                );
            }
        }
        for (source, edges) in case["graph"].as_object().unwrap() {
            let actual = &db.graph[&mapping[&source.parse::<u64>().unwrap()]];
            assert_eq!(actual.len(), edges.as_object().unwrap().len());
            for (target, score) in edges.as_object().unwrap() {
                close(
                    actual[&mapping[&target.parse::<u64>().unwrap()]],
                    score.as_f64().unwrap(),
                    "graph",
                );
            }
        }
        for query in case["queries"].as_array().unwrap() {
            let hits = db
                .query_with_policy(
                    &vector(&query["vector"]),
                    15.,
                    3,
                    query["source"].as_str(),
                    &policy,
                )
                .unwrap();
            let expected = query["hits"].as_array().unwrap();
            assert_eq!(hits.len(), expected.len());
            for (hit, expected) in hits.iter().zip(expected) {
                assert_eq!(
                    hit.node_id,
                    mapping[&expected[0].as_u64().unwrap()],
                    "recall order for {policy:?}"
                );
                close(hit.score, expected[1].as_f64().unwrap(), "recall score");
            }
        }
        assert!(db.check_invariants().is_empty());
        for (name, value) in db.stats() {
            close(value as f32, case["stats"][&name].as_f64().unwrap(), &name);
        }
        for (name, value) in db.merge_angle_diagnostics() {
            close(value as f32, case["angles"][&name].as_f64().unwrap(), &name);
        }
        assert_eq!(
            db.query(&[1., 0., 0.], 15., usize::MAX).unwrap().len(),
            db.nodes.len()
        );
        let mut calibrated_policy = policy.clone();
        db.calibrate_weights_equalize(Some("archive"), &mut calibrated_policy)
            .unwrap();
        for (actual, expected) in [
            db.config.semantic_weight,
            db.config.graph_weight,
            db.config.freshness_weight,
            db.config.confidence_weight,
            calibrated_policy.source_score_weight,
        ]
        .into_iter()
        .zip(case["calibrated"].as_array().unwrap())
        {
            close(actual, expected.as_f64().unwrap(), "calibrated weight");
        }
    }
}

#[test]
fn corpus_presets_and_sample_tuning_match_python_without_random_global_state() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dcmdb-python.json")).unwrap();
    let sample: Vec<_> = fixture["sample"]
        .as_array()
        .unwrap()
        .iter()
        .map(vector)
        .collect();
    for (mode, name) in [
        (CorpusMode::Archival, "archival"),
        (CorpusMode::Summary, "summary"),
    ] {
        let (config, policy) = DcmdbConfig::from_sample(&sample, mode, 500).unwrap();
        close(
            config.theta_similarity,
            fixture["tuned"][name]["theta"].as_f64().unwrap(),
            "tuned threshold",
        );
        close(
            config.merge_angle,
            fixture["tuned"][name]["angle"].as_f64().unwrap(),
            "tuned angle",
        );
        assert_eq!(config.source_divergence_penalty, 3.);
        assert_eq!(policy.source_prior_weight, 0.1);
        assert_eq!(policy.source_score_weight, 0.05);
    }
    assert!(DcmdbConfig::from_sample(&sample[..1], CorpusMode::Summary, 500).is_err());
    assert!(
        DcmdbConfig::from_sample(&[vec![0.; 3], vec![0.; 3]], CorpusMode::Summary, 500).is_err()
    );
    assert_eq!(DcmdbConfig::hashing(3).theta_similarity, 0.55);
    assert_eq!(DcmdbConfig::book_corpus(3).0.prune_weight_threshold, 0.);
}

#[test]
fn adaptive_merges_and_decayed_evidence_match_python_and_preserve_hierarchy() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dcmdb-python.json")).unwrap();
    let inputs = fixture["merge_inputs"].as_array().unwrap();
    for case in fixture["merges"].as_array().unwrap() {
        let mut config = DcmdbConfig::dense(3);
        config.theta_similarity = 0.995;
        config.tension_promote_count = 99;
        config.max_merges_per_pass = Some(1);
        let policy = DcmdbPolicy {
            adaptive_merge_angle: case["adaptive"].as_bool().unwrap(),
            merge_angle_margin: case["margin"].as_f64().unwrap() as f32,
            use_decayed_effective_samples: case["decayed"].as_bool().unwrap(),
            ..DcmdbPolicy::default()
        };
        let mut db = Dcmdb::new(config).unwrap();
        for (i, input) in inputs.iter().enumerate() {
            db.update(MemoryObservation {
                event_id: EventId::from_bytes([i as u8; 32]),
                vector: vector(input),
                time: if i + 1 == inputs.len() {
                    500.
                } else {
                    (i + 1) as f64
                },
                source: Some("synthetic".into()),
                metadata: BTreeMap::new(),
            })
            .unwrap();
        }
        assert_eq!(
            db.consolidate_pass_with_policy(&policy).unwrap(),
            case["count"].as_u64().unwrap() as usize
        );
        let expected_nodes = case["nodes"].as_array().unwrap();
        assert_eq!(db.nodes.len(), expected_nodes.len());
        let mut seen = BTreeSet::new();
        for expected in expected_nodes {
            let centroid = vector(&expected["centroid"]);
            let actual = db
                .nodes
                .values()
                .filter(|n| !seen.contains(&n.id))
                .min_by(|a, b| {
                    let distance = |v: &[f32]| {
                        v.iter()
                            .zip(&centroid)
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                    };
                    distance(&a.centroid).total_cmp(&distance(&b.centroid))
                })
                .unwrap();
            seen.insert(actual.id);
            for (a, b) in actual.centroid.iter().zip(centroid) {
                close(*a, f64::from(b), "merged centroid");
            }
            for (name, value) in [
                ("count", actual.effective_count),
                ("kappa", actual.kappa),
                ("confidence", actual.confidence),
            ] {
                close(value, expected[name].as_f64().unwrap(), name);
            }
            assert_eq!(actual.level, expected["level"].as_u64().unwrap() as u32);
            assert_eq!(
                db.children(actual.id).len(),
                expected["children"].as_u64().unwrap() as usize
            );
            assert_eq!(db.subtree(actual.id).len(), actual.children.len() + 1);
        }
        assert_eq!(
            db.nodes_at_level(1).len(),
            case["count"].as_u64().unwrap() as usize
        );
        assert_eq!(
            db.hierarchy_depth(),
            u32::from(case["count"].as_u64().unwrap() != 0)
        );
        assert!(db.check_invariants().is_empty());
    }
}

#[test]
fn invalid_policy_and_time_do_not_mutate_state() {
    let mut db = Dcmdb::new(DcmdbConfig::dense(3)).unwrap();
    let original = db.clone();
    let observation = MemoryObservation {
        event_id: EventId::from_bytes([1; 32]),
        vector: vec![1., 0., 0.],
        time: 1.,
        source: None,
        metadata: BTreeMap::new(),
    };
    let policy = DcmdbPolicy {
        source_prior_weight: f32::NAN,
        ..DcmdbPolicy::default()
    };
    assert!(db.update_with_policy(observation.clone(), &policy).is_err());
    assert!(
        db.update(MemoryObservation {
            time: f64::INFINITY,
            ..observation
        })
        .is_err()
    );
    assert_eq!(db, original);
}

#[test]
fn counter_and_bayesian_tension_promotion_match_python() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/dcmdb-python.json")).unwrap();
    for case in fixture["promotions"].as_array().unwrap() {
        let mut config = DcmdbConfig::dense(3);
        if case["bayesian"].as_bool().unwrap() {
            config.tension_promote_count = 99;
            config.minimum_tension_evidence = 2;
            config.split_probability = 0.;
            config.clear_probability = 0.;
        }
        let mut db = Dcmdb::new(config).unwrap();
        let mut ids = Vec::new();
        for (i, value) in case["observations"].as_array().unwrap().iter().enumerate() {
            ids.push(
                db.update(MemoryObservation {
                    event_id: EventId::from_bytes([i as u8; 32]),
                    vector: vector(value),
                    time: (i + 1) as f64,
                    source: Some("synthetic".into()),
                    metadata: BTreeMap::new(),
                })
                .unwrap(),
            );
        }
        assert_ne!(ids[0], *ids.last().unwrap());
        let parent = db.node(ids[0]).unwrap();
        let child = db.node(*ids.last().unwrap()).unwrap();
        for (actual, name) in [
            (&parent.centroid, "parent_centroid"),
            (&child.centroid, "child_centroid"),
        ] {
            for (a, b) in actual.iter().zip(vector(&case[name])) {
                close(*a, f64::from(b), name);
            }
        }
        close(
            parent.kappa,
            case["parent_kappa"].as_f64().unwrap(),
            "parent kappa after split",
        );
        close(
            parent.confidence,
            case["parent_confidence"].as_f64().unwrap(),
            "parent confidence after split",
        );
        close(
            parent.effective_count,
            case["parent_count"].as_f64().unwrap(),
            "parent evidence after split",
        );
    }
}
