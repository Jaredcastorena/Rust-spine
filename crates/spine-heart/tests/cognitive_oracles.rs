use std::collections::BTreeMap;

use spine_heart::{
    Dcmdb, DcmdbConfig, EventId, MemoryObservation, RiskField, Thymos, ThymosConfig,
};

fn close(actual: f32, expected: f64, tolerance: f32) {
    assert!(
        (actual - expected as f32).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn dcmdb_matches_python_two_observation_oracle() {
    let mut config = DcmdbConfig::dense(3);
    config.theta_similarity = 0.8;
    config.tension_promote_count = 99;
    config.minimum_tension_evidence = 3;
    let mut memory = Dcmdb::new(config).unwrap();

    let node_id = memory
        .update(MemoryObservation {
            event_id: EventId::from_bytes([1; 32]),
            vector: vec![1.0, 0.0, 0.0],
            time: 1.0,
            source: Some("oracle".into()),
            metadata: BTreeMap::new(),
        })
        .unwrap();
    let updated_id = memory
        .update(MemoryObservation {
            event_id: EventId::from_bytes([2; 32]),
            vector: vec![0.99, 0.1, 0.0],
            time: 2.0,
            source: Some("oracle".into()),
            metadata: BTreeMap::new(),
        })
        .unwrap();
    assert_eq!(updated_id, node_id);

    let node = memory.node(node_id).unwrap();
    close(node.centroid[0], 0.9978156842886754, 2e-6);
    close(node.centroid[1], 0.06605952003702781, 2e-6);
    close(node.sufficient_sum[0], 1.640102976199661, 2e-6);
    close(node.sufficient_sum[1], 0.06566193358085788, 2e-6);
    close(node.effective_count, 1.6434108193699335, 2e-6);
    close(node.weight, 1.9849870227716662, 2e-6);
    // The Rust projection intentionally stores f32 vectors; concentration is
    // particularly sensitive near a unit resultant length.
    close(node.kappa, 824.6839990229377, 1e-2);
    close(node.confidence, 0.2054263524212417, 2e-6);
    close(node.tau, 105.0, 1e-4);
    let tension = node
        .tension
        .as_ref()
        .expect("atypical update creates tension");
    close(tension.log_bayes_factor, 0.0, 1e-6);

    let hit = &memory.query(&[1.0, 0.0, 0.0], 2.0, 1).unwrap()[0];
    assert_eq!(hit.node_id, node_id);
    close(hit.score, 0.85830543348087, 2e-6);
}

#[test]
fn thymos_matches_python_fixed_tensor_oracle() {
    let mut config = ThymosConfig::new(3, 2).unwrap();
    config.learning_rate = 0.01;
    config.decay = 0.001;
    config.normalize_rows = false;
    config.trajectory_alpha = 0.3;
    let mut thymos = Thymos::with_tensor(
        config,
        vec![0.2, -0.1, 0.05, -0.3, 0.4, 0.1],
        Some(vec![0.3, 0.5]),
    )
    .unwrap();

    let feeling = thymos.query(&[1.0, 2.0, 3.0]).unwrap();
    close(feeling.raw[0], 0.04008918628686366, 2e-6);
    close(feeling.raw[1], 0.21380899352993954, 2e-6);
    close(feeling.valence, 0.1269490899084016, 2e-6);
    close(feeling.arousal, 0.21753489046915803, 2e-6);

    let eligibility = thymos
        .compute_valence(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])
        .unwrap();
    close(eligibility[0], -0.1, 2e-6);
    close(eligibility[1], 0.4, 2e-6);
    thymos.update(&[1.0, 2.0, 3.0], &eligibility).unwrap();
    let expected_tensor = [
        0.19978928571428572,
        -0.09992142857142858,
        0.049917857142857146,
        -0.29947142857142856,
        0.4000571428571429,
        0.10058571428571429,
    ];
    for (actual, expected) in thymos.tensor().iter().zip(expected_tensor) {
        close(*actual, expected, 2e-6);
    }
    close(thymos.channel_mass()[0], 0.2997400891862868, 2e-6);
    close(thymos.channel_mass()[1], 0.5003552359741198, 2e-6);

    assert_eq!(thymos.step(&[1.0, 0.0, 0.0]).unwrap().speed, 0.0);
    let step = thymos.step(&[0.0, 1.0, 0.0]).unwrap();
    close(step.speed, std::f64::consts::FRAC_PI_2, 2e-6);
    close(step.surprise, std::f64::consts::FRAC_PI_2, 2e-6);
    close(step.heading_norm, 0.0, 2e-6);
    let prediction = thymos.predict_next().unwrap();
    close(prediction[0], 0.0, 2e-6);
    close(prediction[1], 1.0, 2e-6);
    close(prediction[2], 0.0, 2e-6);
}

#[test]
fn risk_field_learns_toward_positive_outcomes() {
    let mut risk = RiskField::new(3, 0, 0);
    close(
        risk.update(&[1.0, 0.0, 0.0], &[], &[], 1.0).unwrap(),
        0.5,
        1e-6,
    );
    assert!(risk.predict(&[1.0, 0.0, 0.0], &[], &[]).unwrap() > 0.5);
}
