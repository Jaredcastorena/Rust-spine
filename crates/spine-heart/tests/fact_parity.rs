use std::collections::BTreeMap;

use spine_heart::{
    EventId, FactAggregation, FactAggregationIntent, FactAggregationRouter, FactExtractor,
    FactQueryAggregation, FactSlotType, FactStore, FactValue, NodeId, TimeSource,
};

fn one(extractor: &FactExtractor, text: &str) -> spine_heart::FactCandidate {
    let facts = extractor.extract(text, None, None, 42, [3, 4]);
    assert_eq!(facts.len(), 1, "expected one fact for {text:?}: {facts:?}");
    facts.into_iter().next().unwrap()
}

#[test]
fn partner_names_require_explicit_naming_statements() {
    let extractor = FactExtractor::new().unwrap();
    for (text, name) in [
        ("My wife is named Ana.", "Ana"),
        ("My husband is called Alex.", "Alex"),
        ("MY PARTNER IS NAMED Sam.", "Sam"),
        ("My partner is named Ana", "Ana"),
    ] {
        let fact = one(&extractor, text);
        assert_eq!(fact.attribute, "partner_name");
        assert_eq!(fact.value, FactValue::Text(name.into()));
    }
    for text in [
        "My wife is feeling better.",
        "My wife loves hiking.",
        "My husband is going to work.",
        "My partner called yesterday.",
        "MY WIFE IS FEELING BETTER.",
        "My girlfriend is a teacher.",
        "My wife is named after her grandmother.",
        "My wife is called every evening.",
        "My wife is called daily.",
        "My wife is called Often when someone needs help.",
    ] {
        assert!(
            extractor
                .extract(text, None, None, 1, [0, 1])
                .iter()
                .all(|fact| fact.slot_key != "profile.partner"),
            "ordinary predicate became a partner name: {text}"
        );
    }
}

#[test]
fn ordinary_partner_updates_do_not_supersede_a_known_name() {
    let extractor = FactExtractor::new().unwrap();
    let mut store = FactStore::default();
    for (index, text) in [
        "My wife is named Ana.",
        "My wife is feeling better.",
        "My wife loves hiking.",
        "My wife is named after her grandmother.",
        "My wife is called every evening.",
        "My wife is called daily.",
    ]
    .into_iter()
    .enumerate()
    {
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            extractor.extract(text, None, None, index as u64, [0, index as u64]),
        );
    }
    let partners: Vec<_> = store
        .active()
        .filter(|fact| fact.slot_key == "profile.partner")
        .collect();
    assert_eq!(partners.len(), 1);
    assert_eq!(partners[0].value, FactValue::Text("Ana".into()));
    assert_eq!(partners[0].event_id, EventId::from_bytes([1; 32]));
}

#[test]
fn single_and_double_digit_ages_work_across_all_age_forms() {
    let extractor = FactExtractor::new().unwrap();
    for age in [9, 32] {
        for text in [
            format!("I'm {age} years old."),
            format!("I am {age} years old."),
            format!("I am currently {age} years old."),
            format!("I'm now {age}."),
            format!("I just turned {age}."),
            format!("I turned {age}."),
            format!("My age is {age}."),
        ] {
            let fact = one(&extractor, &text);
            assert_eq!(fact.value, FactValue::Integer(age), "{text}");
            assert_eq!(fact.slot_key, "profile.age");
        }
    }
}

fn amount_candidate(label: &str, value: f64, arrival: u64) -> spine_heart::FactCandidate {
    spine_heart::FactCandidate {
        entity: "USER".into(),
        attribute: format!("amount_{label}_{arrival}"),
        value: FactValue::Amount(value),
        slot_type: FactSlotType::EventAmount,
        slot_key: format!("expense.{label}"),
        excerpt: format!("I spent ${value} on {label} item {arrival}."),
        event_time: None,
        session_time: None,
        ingest_millis: arrival,
        time_source: TimeSource::Inferred,
        arrival_order: [0, arrival],
        source_role: "user".into(),
        confidence: 0.9,
        has_update_cue: false,
        metadata: BTreeMap::new(),
    }
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

#[test]
fn aggregation_intent_router_matches_oracle_priority() {
    let router = FactAggregationRouter::new().unwrap();
    let cases = [
        (
            "What is the difference between what I spent and what I paid?",
            FactAggregationIntent::Diff,
        ),
        (
            "What was the most expensive thing I spent money on?",
            FactAggregationIntent::Max,
        ),
        (
            "Which purchase was cheapest overall?",
            FactAggregationIntent::Min,
        ),
        (
            "How much did I spend across all trips?",
            FactAggregationIntent::Sum,
        ),
        (
            "How many times did I attend a wedding?",
            FactAggregationIntent::Count,
        ),
        (
            "How many times did I spend money?",
            FactAggregationIntent::Sum,
        ),
        ("Remind me about my bicycle.", FactAggregationIntent::None),
    ];

    for (query, expected) in cases {
        assert_eq!(router.detect(query), expected, "{query}");
    }
}

#[test]
fn natural_query_aggregation_matches_oracle_modes_and_exclusions() {
    let extractor = FactExtractor::new().unwrap();
    let mut store = FactStore::default();
    let facts = [
        ("I spent $45 on bike tires.", "2025-02-01"),
        ("I spent $135 on bike accessories.", "2025-02-03"),
        ("I attended 3 weddings.", "2025-03-01"),
        ("I attended 2 weddings.", "2025-03-02"),
        ("I've read 80 Sapiens pages.", "2025-04-01"),
    ];
    for (index, (text, event_time)) in facts.into_iter().enumerate() {
        let candidates = extractor.extract(
            text,
            Some(event_time.into()),
            None,
            index as u64,
            [0, index as u64],
        );
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            candidates,
        );
    }

    let sum = store
        .aggregate_query("How much total did I spend on bike gear?")
        .unwrap()
        .unwrap();
    let FactQueryAggregation::Sum {
        value,
        evidence,
        money,
    } = sum
    else {
        panic!("expected sum, got {sum:?}");
    };
    assert_eq!(value, 180.0);
    assert!(money);
    assert_eq!(evidence.len(), 2);
    assert_eq!(evidence[0].value, FactValue::Amount(45.0));
    assert_eq!(evidence[1].value, FactValue::Amount(135.0));

    let default_sum = store.aggregate_query("bike").unwrap().unwrap();
    assert!(matches!(
        default_sum,
        FactQueryAggregation::Sum { value: 180.0, .. }
    ));

    let count = store
        .aggregate_query("How many weddings did I attend?")
        .unwrap()
        .unwrap();
    assert!(matches!(
        count,
        FactQueryAggregation::Count {
            value: 5,
            ref evidence,
        } if evidence.len() == 2
    ));

    let diff = store
        .aggregate_query("What was the difference between my bike expenses?")
        .unwrap()
        .unwrap();
    assert!(matches!(
        diff,
        FactQueryAggregation::Diff {
            value: 90.0,
            highest_value: 135.0,
            lowest_value: 45.0,
            money: true,
            ..
        }
    ));

    let max = store
        .aggregate_query("What was the most expensive bike purchase?")
        .unwrap()
        .unwrap();
    assert!(matches!(
        max,
        FactQueryAggregation::Max {
            value: 135.0,
            money: true,
            ..
        }
    ));

    let min = store
        .aggregate_query("What was the cheapest bike purchase?")
        .unwrap()
        .unwrap();
    assert!(matches!(
        min,
        FactQueryAggregation::Min {
            value: 45.0,
            money: true,
            ..
        }
    ));

    assert!(
        store
            .aggregate_query("How much total Sapiens reading?")
            .unwrap()
            .is_none(),
        "state.quantity facts must never enter deterministic aggregation"
    );
}

#[test]
fn diff_and_extrema_preserve_python_tie_order() {
    let mut store = FactStore::default();
    for (index, value) in [100.0, 50.0, 100.0, 50.0].into_iter().enumerate() {
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            vec![amount_candidate("tiecase", value, index as u64)],
        );
    }

    let incoming = store.search("tiecase", 40, false);
    let first_high = incoming
        .iter()
        .find(|hit| hit.fact.value == FactValue::Amount(100.0))
        .unwrap()
        .fact
        .id;
    let last_high = incoming
        .iter()
        .rfind(|hit| hit.fact.value == FactValue::Amount(100.0))
        .unwrap()
        .fact
        .id;
    let first_low = incoming
        .iter()
        .find(|hit| hit.fact.value == FactValue::Amount(50.0))
        .unwrap()
        .fact
        .id;

    let maximum = store
        .aggregate_query_with_intent("tiecase", FactAggregationIntent::Max)
        .unwrap();
    let FactQueryAggregation::Max { fact, .. } = maximum else {
        panic!("expected max, got {maximum:?}");
    };
    assert_eq!(fact.id, first_high, "max keeps the first BM25-order tie");

    let minimum = store
        .aggregate_query_with_intent("tiecase", FactAggregationIntent::Min)
        .unwrap();
    let FactQueryAggregation::Min { fact, .. } = minimum else {
        panic!("expected min, got {minimum:?}");
    };
    assert_eq!(fact.id, first_low, "min keeps the first BM25-order tie");

    let difference = store
        .aggregate_query_with_intent("tiecase", FactAggregationIntent::Diff)
        .unwrap();
    let FactQueryAggregation::Diff {
        highest, lowest, ..
    } = difference
    else {
        panic!("expected diff, got {difference:?}");
    };
    assert_eq!(highest.id, last_high, "diff takes the last tied maximum");
    assert_eq!(lowest.id, first_low, "diff takes the first tied minimum");
}

#[test]
fn natural_aggregation_is_capped_at_forty_active_search_hits() {
    let mut store = FactStore::default();
    for index in 0_u8..45 {
        store.add_candidates(
            EventId::from_bytes([index + 1; 32]),
            NodeId::from_bytes([index + 101; 32]),
            vec![amount_candidate("capstone", 1.0, u64::from(index))],
        );
    }
    assert_eq!(store.active().count(), 45);

    let aggregation = store
        .aggregate_query("How much total capstone spending?")
        .unwrap()
        .unwrap();
    let FactQueryAggregation::Sum {
        value, evidence, ..
    } = aggregation
    else {
        panic!("expected sum, got {aggregation:?}");
    };
    assert_eq!(value, 40.0);
    assert_eq!(evidence.len(), 40);
}

#[test]
fn adjacent_purchase_sentences_do_not_contaminate_aggregation_evidence() {
    let extractor = FactExtractor::new().unwrap();
    let candidates = extractor.extract(
        "I spent $45 on bike tires. I spent $135 on concert tickets.",
        None,
        None,
        1,
        [0, 1],
    );
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].excerpt, "I spent $45 on bike tires.");
    assert_eq!(candidates[1].excerpt, "I spent $135 on concert tickets.");
    let mut store = FactStore::default();
    store.add_candidates(
        EventId::from_bytes([1; 32]),
        NodeId::from_bytes([2; 32]),
        candidates,
    );
    assert!(matches!(
        store.aggregate_query("concert").unwrap(),
        Some(FactQueryAggregation::Sum { value: 135.0, ref evidence, .. })
            if evidence.len() == 1
    ));
}

#[test]
fn fact_search_keeps_structured_metadata_and_oracle_stopwords() {
    let mut candidate = amount_candidate("purchase", 45.0, 1);
    candidate.metadata = BTreeMap::from([
        ("over".into(), "rejected_choice".into()),
        ("per".into(), "weekly_interval".into()),
        ("category".into(), "outdoor_equipment".into()),
        ("source_uri".into(), "private_source_location".into()),
    ]);
    candidate.excerpt = "I could buy this, but I would not. No, I should wait.".into();
    let mut store = FactStore::default();
    store.add_candidates(
        EventId::from_bytes([1; 32]),
        NodeId::from_bytes([2; 32]),
        vec![candidate],
    );
    for query in ["rejected choice", "weekly interval", "outdoor equipment"] {
        assert_eq!(store.search(query, 1, false).len(), 1, "{query}");
    }
    assert!(store.search("private source location", 1, false).is_empty());
    assert!(
        store
            .search("could would not no should", 1, false)
            .is_empty()
    );
    assert!(store.aggregate_query("could unicorn").unwrap().is_none());
}

#[test]
fn amount_search_scores_match_python_fact_documents() {
    let extractor = FactExtractor::new().unwrap();
    let mut store = FactStore::default();
    for (index, text) in [
        "I spent $45 on bike tires.",
        "I spent $135 on bike accessories.",
    ]
    .into_iter()
    .enumerate()
    {
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            extractor.extract(text, None, None, 1, [0, index as u64]),
        );
    }
    let hits = store.search("bike tires", 2, false);
    assert_eq!(hits[0].fact.slot_key, "expense.bike_tires");
    // Python oracle: length 16, bike/tires frequency 4, BM25 k1=1.5, b=.75.
    assert!((hits[0].score - 1.591_761_4).abs() < 1e-6);
    assert!((hits[1].score - 0.331_493_74).abs() < 1e-6);
}

#[test]
fn tied_fact_search_preserves_arrival_order_before_confidence_or_hashes() {
    let mut store = FactStore::default();
    for (index, confidence) in [0.5, 0.9].into_iter().enumerate() {
        let mut candidate = amount_candidate("tie", 45.0, 0);
        candidate.arrival_order = [0, index as u64];
        candidate.confidence = confidence;
        store.add_candidates(
            EventId::from_bytes([index as u8 + 1; 32]),
            NodeId::from_bytes([index as u8 + 11; 32]),
            vec![candidate],
        );
    }
    let hits = store.search("tie", 2, false);
    assert_eq!(hits[0].score, hits[1].score);
    assert_eq!(hits[0].fact.event_id, EventId::from_bytes([1; 32]));
    let Some(FactQueryAggregation::Max { fact, .. }) =
        store.aggregate_query_with_intent("tie", FactAggregationIntent::Max)
    else {
        panic!("expected a tied maximum");
    };
    assert_eq!(fact.event_id, EventId::from_bytes([1; 32]));
}
