use std::collections::BTreeMap;

use spine_heart::{
    EventId, FactAggregation, FactExtractor, FactSlotType, FactStore, FactValue, NodeId, TimeSource,
};

fn one(extractor: &FactExtractor, text: &str) -> spine_heart::FactCandidate {
    let facts = extractor.extract(text, None, None, 42, [3, 4]);
    assert_eq!(facts.len(), 1, "expected one fact for {text:?}: {facts:?}");
    facts.into_iter().next().unwrap()
}

#[test]
fn golden_oracle_extraction_families() {
    let extractor = FactExtractor::new().unwrap();
    let cases = [
        (
            "My age is 32.",
            "age",
            FactValue::Integer(32),
            FactSlotType::State,
            "profile.age",
        ),
        (
            "I moved to New York.",
            "location",
            FactValue::Text("New York".into()),
            FactSlotType::State,
            "profile.location",
        ),
        (
            "My job is a software engineer.",
            "occupation",
            FactValue::Text("software engineer".into()),
            FactSlotType::State,
            "profile.occupation",
        ),
        (
            "I'm vegan.",
            "diet",
            FactValue::Text("vegan".into()),
            FactSlotType::State,
            "profile.diet",
        ),
        (
            "My wife is named Ana.",
            "partner_name",
            FactValue::Text("Ana".into()),
            FactSlotType::State,
            "profile.partner",
        ),
        (
            "I've read 12 pages.",
            "count_pages",
            FactValue::Integer(12),
            FactSlotType::StateQuantity,
            "activity.reading.pages.count",
        ),
        (
            "I go to yoga twice a week.",
            "frequency_yoga",
            FactValue::Integer(2),
            FactSlotType::Frequency,
            "activity.yoga.frequency",
        ),
        (
            "I went to New York.",
            "event",
            FactValue::Text("New York".into()),
            FactSlotType::Event,
            "event.travel",
        ),
        (
            "I spent $45 on bike tires.",
            "amount_bike_tires",
            FactValue::Amount(45.0),
            FactSlotType::EventAmount,
            "expense.bike_tires",
        ),
        (
            "I attended 3 weddings.",
            "count_weddings",
            FactValue::Integer(3),
            FactSlotType::EventCount,
            "attended.weddings.count",
        ),
        (
            "My favorite ice cream is Rocky Road.",
            "favorite_ice_cream",
            FactValue::Text("Rocky Road".into()),
            FactSlotType::Preference,
            "preference.favorite.ice_cream",
        ),
        (
            "I prefer coffee over tea.",
            "preference",
            FactValue::Text("coffee".into()),
            FactSlotType::Preference,
            "preference.general",
        ),
        (
            "I like hiking.",
            "preference_like",
            FactValue::Text("hiking".into()),
            FactSlotType::Preference,
            "preference.like",
        ),
        (
            "I can't eat tree nuts.",
            "diet_avoid",
            FactValue::Text("tree nuts".into()),
            FactSlotType::Preference,
            "preference.food.avoid",
        ),
        (
            "My hobby is Formula 1 racing.",
            "hobby",
            FactValue::Text("Formula 1 racing".into()),
            FactSlotType::Preference,
            "preference.hobby",
        ),
        (
            "My daughter is named Mia.",
            "child_name",
            FactValue::Text("Mia".into()),
            FactSlotType::Entity,
            "entity.family.child",
        ),
        (
            "My dog is named Rover.",
            "pet_dog",
            FactValue::Text("Rover".into()),
            FactSlotType::Entity,
            "entity.pet.dog",
        ),
    ];

    for (text, attribute, value, slot_type, slot_key) in cases {
        let fact = one(&extractor, text);
        assert_eq!(fact.attribute, attribute, "{text}");
        assert_eq!(fact.value, value, "{text}");
        assert_eq!(fact.slot_type, slot_type, "{text}");
        assert_eq!(fact.slot_key, slot_key, "{text}");
        assert_eq!(fact.source_role, "user", "{text}");
    }
}

#[test]
fn paired_chunks_extract_only_user_text_and_preserve_dates() {
    let extractor = FactExtractor::new().unwrap();
    let text = "[2026-04-05 12:00] [date: 2025/02/03] [user]: Actually, I spent $45 on bike tires.\n[assistant]: I spent $900 on an invented purchase.";
    let first = extractor.extract(text, None, Some("2025/02/03".into()), 99, [7, 4]);
    let second = extractor.extract(text, None, Some("2025/02/03".into()), 99, [7, 4]);

    assert_eq!(first, second);
    assert_eq!(first.len(), 1, "assistant text must not become a user fact");
    let fact = &first[0];
    assert_eq!(fact.value, FactValue::Amount(45.0));
    assert_eq!(fact.event_time.as_deref(), Some("2025-02-03"));
    assert_eq!(fact.session_time.as_deref(), Some("2025-02-03"));
    assert_eq!(fact.time_source, TimeSource::Explicit);
    assert_eq!(fact.arrival_order, [7, 4]);
    assert!(fact.has_update_cue);
    assert!(!fact.excerpt.contains("assistant"));

    assert!(
        extractor
            .extract(
                "[date: 2025/02/03] [assistant]: I'm 99 years old.",
                None,
                Some("2025-02-03".into()),
                99,
                [7, 5],
            )
            .is_empty()
    );
}

#[test]
fn event_count_aggregation_sums_occurrences_not_fact_rows() {
    let extractor = FactExtractor::new().unwrap();
    let mut store = FactStore::default();
    for (index, text) in ["I attended 3 weddings.", "I attended 2 weddings."]
        .into_iter()
        .enumerate()
    {
        let candidates = extractor.extract(text, None, None, index as u64, [0, index as u64 + 1]);
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            candidates,
        );
    }

    assert_eq!(
        store.aggregate("attended.weddings", "count").unwrap(),
        FactAggregation::Count(5)
    );
    assert_eq!(
        store.aggregate("attended.weddings", "sum").unwrap(),
        FactAggregation::Sum(5.0)
    );

    let manual = spine_heart::FactCandidate {
        entity: "USER".into(),
        attribute: "amount_ticket".into(),
        value: FactValue::Amount(25.0),
        slot_type: FactSlotType::EventAmount,
        slot_key: "expense.ticket".into(),
        excerpt: "I spent $25 on a ticket.".into(),
        event_time: None,
        session_time: None,
        ingest_millis: 3,
        time_source: TimeSource::Inferred,
        arrival_order: [0, 3],
        source_role: "user".into(),
        confidence: 0.9,
        has_update_cue: false,
        metadata: BTreeMap::new(),
    };
    store.add_candidates(
        EventId::from_bytes([3; 32]),
        NodeId::from_bytes([13; 32]),
        vec![manual],
    );
    assert_eq!(
        store.aggregate("expense.", "count").unwrap(),
        FactAggregation::Count(1)
    );

    let current_quantity = spine_heart::FactCandidate {
        entity: "USER".into(),
        attribute: "count_pages".into(),
        value: FactValue::Integer(80),
        slot_type: FactSlotType::StateQuantity,
        slot_key: "activity.reading.pages.count".into(),
        excerpt: "I've read 80 pages.".into(),
        event_time: None,
        session_time: None,
        ingest_millis: 4,
        time_source: TimeSource::Inferred,
        arrival_order: [0, 4],
        source_role: "user".into(),
        confidence: 0.85,
        has_update_cue: false,
        metadata: BTreeMap::new(),
    };
    store.add_candidates(
        EventId::from_bytes([4; 32]),
        NodeId::from_bytes([14; 32]),
        vec![current_quantity],
    );
    assert_eq!(
        store.aggregate("activity.reading", "sum").unwrap(),
        FactAggregation::Sum(0.0),
        "supersedable state quantities must never be summed as events"
    );
}
