use spine_heart::NliModel;
use spine_models::{MiniLmNli, NliAssets};

#[test]
fn native_nli_matches_reference_and_rejects_incomplete_pairs() {
    let Ok(directory) = std::env::var("SPINE_NLI_DIR") else {
        eprintln!("Set SPINE_NLI_DIR to run the pinned local-model reference check");
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/nli_reference.json")).unwrap();
    let model = MiniLmNli::load(NliAssets::from_directory(directory)).unwrap();
    let artifact_hash: [u8; 32] =
        serde_json::from_value(reference["artifact_hash"].clone()).unwrap();
    let tokenizer_hash: [u8; 32] =
        serde_json::from_value(reference["tokenizer_hash"].clone()).unwrap();
    assert_eq!(model.manifest().artifact_hash, artifact_hash);
    assert_eq!(model.manifest().tokenizer_hash, tokenizer_hash);
    let mut pairs: Vec<(String, String)> =
        serde_json::from_value(reference["pairs"].clone()).unwrap();
    // The ninth pair crosses the native inference batch boundary and changes
    // padding. Both paths must preserve the reference interpretation.
    pairs.push(pairs[0].clone());
    let mut expected: Vec<[f32; 3]> = serde_json::from_value(reference["logits"].clone()).unwrap();
    expected.push(expected[0]);
    let logits = model.predict_logits(&pairs).unwrap();
    assert_eq!(logits.len(), expected.len());
    for (index, (actual, reference)) in logits.iter().zip(&expected).enumerate() {
        for (actual, reference) in actual.iter().zip(reference) {
            assert!(actual.is_finite());
            assert!(
                (actual - reference).abs() < 0.0005,
                "pair {index}: native {actual}, reference {reference}"
            );
        }
    }
    let too_many_tokens = "evidence ".repeat(700);
    let error = model
        .predict_logits(&[(too_many_tokens, "The final claim differs.".into())])
        .unwrap_err();
    assert!(error.to_string().contains("512 tokens"));
    let error = model
        .predict_logits(&[("x".repeat(128 * 1024 + 1), "claim".into())])
        .unwrap_err();
    assert!(error.to_string().contains("input size limit"));
}
