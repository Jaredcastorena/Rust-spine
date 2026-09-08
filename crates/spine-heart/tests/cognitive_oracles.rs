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
fn thymos_state_summary_matches_grid_equations_without_mutation() {
    let thymos = Thymos::with_tensor(
        ThymosConfig::new(2, 2).unwrap(),
        vec![1.0, 0.0, 0.0, 2.0],
        Some(vec![2.0, 4.0]),
    )
    .unwrap();
    let before = thymos.clone();
    let summary = thymos.state_summary();
    assert_eq!(summary["channel_norms"], serde_json::json!([1.0, 2.0]));
    assert_eq!(
        summary["mean_resultant_length"],
        serde_json::json!([0.5, 0.5])
    );
    assert_eq!(summary["mean_cross_similarity"], 0.0);
    assert_eq!(summary["config_K"], 2);
    assert_eq!(summary["config_d"], 2);
    assert_eq!(summary["has_trajectory"], false);
    assert_eq!(thymos, before);
    let one_channel =
        Thymos::with_tensor(ThymosConfig::new(2, 1).unwrap(), vec![0.0, 0.0], None).unwrap();
    assert_eq!(one_channel.state_summary()["mean_cross_similarity"], 0.0);
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
fn per_observation_learning_multiplier_matches_scaled_python_eta_without_scaling_decay() {
    let config = ThymosConfig::new(3, 2).unwrap();
    let tensor = vec![0.2, -0.1, 0.05, -0.3, 0.4, 0.1];
    for multiplier in [1.0, 1.5, 2.0] {
        let mut actual = Thymos::with_tensor(config.clone(), tensor.clone(), None).unwrap();
        let mut scaled_config = config.clone();
        scaled_config.learning_rate *= multiplier;
        let mut oracle = Thymos::with_tensor(scaled_config, tensor.clone(), None).unwrap();
        let eligibility = actual
            .compute_valence(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])
            .unwrap();
        actual
            .update_scaled(&[1.0, 2.0, 3.0], &eligibility, multiplier)
            .unwrap();
        oracle.update(&[1.0, 2.0, 3.0], &eligibility).unwrap();
        assert_eq!(actual.tensor(), oracle.tensor());
        assert_eq!(actual.channel_mass(), oracle.channel_mass());
        assert_eq!(actual.config.learning_rate, config.learning_rate);
        let before = actual.clone();
        assert!(
            actual
                .update_scaled(&[1.0, 2.0, 3.0], &eligibility, f32::NAN)
                .is_err()
        );
        assert_eq!(actual, before);
    }
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
