use spine_heart::{NliLabelOrder, NliModel};
use spine_models::{MiniLmNli, NliAssets};

#[test]
fn local_nli_baseline_preserves_label_order_and_separates_examples() {
    let Ok(directory) = std::env::var("SPINE_NLI_DIR") else {
        return;
    };
    let model = MiniLmNli::load(NliAssets::from_directory(directory)).unwrap();
    assert_eq!(model.label_order(), NliLabelOrder::CROSS_ENCODER_MINILM);
    let logits = model
        .predict_logits(&[
            (
                "A person is outdoors.".into(),
                "A person is outdoors.".into(),
            ),
            ("A person is outdoors.".into(), "Nobody is outdoors.".into()),
        ])
        .unwrap();
    assert_eq!(logits.len(), 2);
    assert_eq!(
        logits[0]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0,
        1
    );
    assert_eq!(
        logits[1]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0,
        0
    );
}
