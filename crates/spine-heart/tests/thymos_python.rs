use spine_heart::{ActivationNonlinearity, Thymos, ThymosConfig, ValenceMode};

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn close(actual: &[f32], expected: &serde_json::Value) {
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-5,
            "coordinate {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn all_activation_and_valence_modes_match_python_across_learning_and_trajectory() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/thymos-python.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut config = ThymosConfig::new(3, 3).unwrap();
        config.learning_rate = 0.02;
        config.decay = 0.003;
        config.valence_temperature = 0.7;
        config.normalize_rows = case["normalize_rows"].as_bool().unwrap();
        config.channel_labels = Some(vec!["first".into(), "second".into(), "third".into()]);
        config.valence_mode = match case["mode"].as_str().unwrap() {
            "cosine_error" => ValenceMode::CosineError,
            "spherical_residual" => ValenceMode::SphericalResidual,
            "direct_diff" => ValenceMode::DirectDifference,
            other => panic!("unknown oracle mode {other}"),
        };
        config.nonlinearity = match case["activation"].as_str().unwrap() {
            "none" => ActivationNonlinearity::None,
            "tanh" => ActivationNonlinearity::Tanh,
            "relu" => ActivationNonlinearity::Relu,
            "softmax" => ActivationNonlinearity::Softmax,
            other => panic!("unknown oracle activation {other}"),
        };
        let mut thymos = Thymos::with_tensor(
            config,
            floats(&fixture["initial_tensor"]),
            Some(floats(&fixture["initial_mass"])),
        )
        .unwrap();
        for step in case["steps"].as_array().unwrap() {
            let input = floats(&step["input"]);
            let query = thymos.query(&input).unwrap();
            close(&query.raw, &step["query"]["raw"]);
            close(&query.activated, &step["query"]["activated"]);
            let unnormalized = thymos.query_with_normalization(&input, false).unwrap();
            close(&unnormalized.raw, &step["unnormalized_query"]["raw"]);
            close(
                &unnormalized.activated,
                &step["unnormalized_query"]["activated"],
            );
            assert_eq!(
                query.dominant_channel,
                step["query"]["dominant_channel"].as_u64().unwrap() as usize
            );
            let eligibility = thymos
                .compute_valence(&floats(&step["expected"]), &input)
                .unwrap();
            close(&eligibility, &step["eligibility"]);
            thymos.update(&input, &eligibility).unwrap();
            close(thymos.tensor(), &step["tensor"]);
            close(thymos.channel_mass(), &step["mass"]);
            let trajectory = thymos.step(&input).unwrap();
            for (name, value) in [
                ("surprise", trajectory.surprise),
                ("speed", trajectory.speed),
                ("heading_norm", trajectory.heading_norm),
            ] {
                assert!(
                    (value - step["trajectory"][name].as_f64().unwrap() as f32).abs() < 2e-5,
                    "{name}: {value} versus {}",
                    step["trajectory"][name]
                );
            }
            match thymos.predict_next() {
                Some(prediction) => close(&prediction, &step["prediction"]),
                None => assert!(step["prediction"].is_null()),
            }
        }
    }
}

#[test]
fn tied_channels_choose_first_like_numpy_argmax() {
    let thymos = Thymos::with_tensor(
        ThymosConfig::new(2, 3).unwrap(),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        None,
    )
    .unwrap();
    assert_eq!(thymos.query(&[1.0, 0.0]).unwrap().dominant_channel, 0);
    assert_eq!(thymos.query(&[0.0, 0.0]).unwrap().dominant_channel, 0);
    assert_eq!(
        thymos
            .channel_affinities(&[1.0, 0.0])
            .unwrap()
            .iter()
            .map(|x| x.0)
            .collect::<Vec<_>>(),
        vec![0, 2, 1]
    );
}

#[test]
fn absolute_learning_rate_presets_and_trajectory_position_preserve_python_controls() {
    let mut config = ThymosConfig::minimal(2).unwrap();
    config.learning_rate = 0.;
    config.decay = 0.1;
    let mut thymos =
        Thymos::with_tensor(config, vec![1., 0., 0., 1., 0., 0.], Some(vec![1., 1., 0.])).unwrap();
    thymos
        .update_with_learning_rate(&[1., 0.], &[2., 3., 1.], 0.5)
        .unwrap();
    assert_eq!(thymos.tensor(), &[1.9, 0., 0., 0.9, 0., 0.]);
    assert_eq!(thymos.channel_mass(), &[1.9, 0.9, 0.]);
    assert_eq!(thymos.config.learning_rate, 0.);
    assert!(thymos.trajectory_position().is_none());
    thymos.step(&[1., 0.]).unwrap();
    assert_eq!(thymos.trajectory_position(), Some([1., 0.].as_slice()));
    assert_eq!(ThymosConfig::for_minilm(8).unwrap().dimension, 384);
    assert_eq!(
        ThymosConfig::for_large_embed(768, 16)
            .unwrap()
            .learning_rate,
        0.005
    );
}
