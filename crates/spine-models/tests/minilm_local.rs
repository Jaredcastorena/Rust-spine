use spine_heart::SemanticEncoder;
use spine_models::{MiniLmAssets, MiniLmEncoder};

#[test]
fn local_minilm_assets_produce_normalized_384d_embeddings() {
    let Ok(directory) = std::env::var("SPINE_MINILM_DIR") else {
        return;
    };
    let encoder = MiniLmEncoder::load(MiniLmAssets::from_directory(directory), 256).unwrap();
    let values = encoder
        .encode("Spine keeps durable semantic memory.")
        .unwrap();
    assert_eq!(values.as_slice().len(), 384);
    let norm = values
        .as_slice()
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}
