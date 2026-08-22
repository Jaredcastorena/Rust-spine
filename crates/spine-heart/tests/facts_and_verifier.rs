use std::collections::BTreeMap;

use spine_heart::{
    AtomicClaim, ClaimExtractor, ClaimRelation, EventId, FactAggregation, FactCandidate,
    FactExtractor, FactSlotType, FactStore, FactValue, NliLabelOrder, NliModel, NliVerifier,
    NodeId, Result, TimeSource,
};

#[test]
fn typed_facts_supersede_state_but_keep_preferences_additive() {
    let extractor = FactExtractor::new().unwrap();
    let mut store = FactStore::default();
    let old = extractor.extract("I'm 31 years old. I love jazz.", None, None, 10, [1, 1]);
    store.add_candidates(
        EventId::from_bytes([1; 32]),
        NodeId::from_bytes([11; 32]),
        old,
    );
    let new = extractor.extract(
        "Actually, I'm now 32 years old. I love hiking.",
        None,
        None,
        20,
        [2, 1],
    );
    store.add_candidates(
        EventId::from_bytes([2; 32]),
        NodeId::from_bytes([22; 32]),
        new,
    );

    let active_ages: Vec<_> = store
        .active()
        .filter(|fact| fact.slot_key == "profile.age")
        .collect();
    assert_eq!(active_ages.len(), 1);
    assert_eq!(active_ages[0].value, FactValue::Integer(32));
    assert_eq!(
        store
            .active()
            .filter(|fact| fact.slot_type == FactSlotType::Preference)
            .count(),
        2
    );
    let hits = store.search("what is my age", 3, false);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fact.value, FactValue::Integer(32));
}

#[test]
fn fact_aggregation_uses_active_typed_values() {
    let mut store = FactStore::default();
    let candidates = [12.5, 7.5]
        .into_iter()
        .enumerate()
        .map(|(index, value)| FactCandidate {
            entity: "USER".into(),
            attribute: format!("amount_purchase_{index}"),
            value: FactValue::Amount(value),
            slot_type: FactSlotType::EventAmount,
            slot_key: format!("expense.purchase_{index}"),
            excerpt: "purchase".into(),
            event_time: None,
            session_time: None,
            ingest_millis: index as u64,
            time_source: TimeSource::Inferred,
            arrival_order: [index as u64 + 1, 0],
            source_role: "user".into(),
            confidence: 0.9,
            has_update_cue: false,
            metadata: BTreeMap::new(),
        })
        .collect();
    store.add_candidates(
        EventId::from_bytes([7; 32]),
        NodeId::from_bytes([8; 32]),
        candidates,
    );
    assert_eq!(
        store.aggregate("expense.", "sum").unwrap(),
        FactAggregation::Sum(20.0)
    );
    assert_eq!(
        store.aggregate("expense.", "count").unwrap(),
        FactAggregation::Count(2)
    );
}

struct OracleNli;

impl NliModel for OracleNli {
    fn label_order(&self) -> NliLabelOrder {
        NliLabelOrder::CROSS_ENCODER_MINILM
    }

    fn predict_logits(&self, pairs: &[(String, String)]) -> Result<Vec<[f32; 3]>> {
        Ok(pairs
            .iter()
            .map(|(hypothesis, evidence)| {
                if hypothesis.contains("located in Denver") && evidence.contains("Denver") {
                    [-3.0, 4.0, -1.0]
                } else if evidence.contains("Boulder") {
                    [4.0, -3.0, -1.0]
                } else {
                    [-1.0, -1.0, 3.0]
                }
            })
            .collect())
    }
}

#[test]
fn verifier_preserves_three_way_label_contract_and_aggregates_maxima() {
    let claim = AtomicClaim {
        subject: "the user".into(),
        relation: ClaimRelation::LocatedIn,
        value: "Denver".into(),
        time: None,
        excerpt: String::new(),
    };
    assert_eq!(claim.hypothesis(), "the user was located in Denver.");
    let verifier = NliVerifier::new(OracleNli, 3).unwrap();
    let report = verifier
        .verify(
            &[claim],
            &[
                "The user lives in Denver.".into(),
                "An obsolete note says Boulder.".into(),
            ],
        )
        .unwrap();
    assert!(report.coverage > 0.98);
    assert!(report.contradiction > 0.98);
    assert_eq!(report.evaluated_pairs, 2);
}

#[test]
fn claim_extractor_is_deterministic_typed_and_deduplicated() {
    let extractor = ClaimExtractor::new().unwrap();
    let text = "I had 3 sessions on 2026-08-22. I had 3 sessions on 2026-08-22. I moved to Denver.";
    let first = extractor.extract(text);
    let second = extractor.extract(text);
    assert_eq!(first, second);
    assert_eq!(
        first
            .iter()
            .filter(|claim| claim.relation == ClaimRelation::HasCount)
            .count(),
        1
    );
    assert!(
        first
            .iter()
            .any(|claim| { claim.relation == ClaimRelation::LocatedIn && claim.value == "Denver" })
    );
}
