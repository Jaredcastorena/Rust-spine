use spine_heart::{
    AgentId, Content, EventKind, HeartConfig, InteractionInput, KeySource, ParticipantRole,
    Provenance, SpineHeart, SyncFrontier, ThreadId, TombstoneTarget,
};

fn interaction(text: &str) -> InteractionInput {
    InteractionInput {
        agent_id: AgentId::new("main").unwrap(),
        thread_id: ThreadId::new("test-thread").unwrap(),
        role: ParticipantRole::User,
        kind: EventKind::Message,
        content: Content::Inline(text.to_owned()),
        causal_parents: Vec::new(),
        provenance: Provenance::default(),
        tool: None,
        attachments: Vec::new(),
        outcome: None,
    }
}

#[test]
fn create_commit_reopen_and_checkout() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("owner.spine");
    let created =
        SpineHeart::create(HeartConfig::new(&path), "correct horse battery staple").unwrap();
    let phrase = created.recovery_phrase.expose().to_owned();
    let receipt = created
        .heart
        .commit_interaction(interaction("remember the triangle boundary"))
        .unwrap();
    let snapshot = created.heart.snapshot(Some("first".into())).unwrap();
    drop(created);

    let reopened = SpineHeart::open(
        HeartConfig::new(&path),
        KeySource::Passphrase("correct horse battery staple".into()),
    )
    .unwrap();
    assert_eq!(
        reopened.event(receipt.event.id).unwrap(),
        Some(receipt.event)
    );
    assert_eq!(
        reopened
            .checkout(snapshot)
            .unwrap()
            .events_canonical()
            .unwrap()
            .len(),
        1
    );
    drop(reopened);

    let recovered =
        SpineHeart::open(HeartConfig::new(&path), KeySource::RecoveryPhrase(phrase)).unwrap();
    assert_eq!(recovered.stats().unwrap().events, 1);
}

#[test]
fn wrong_passphrase_fails() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("owner.spine");
    let _created = SpineHeart::create(HeartConfig::new(&path), "right").unwrap();
    let opened = SpineHeart::open(
        HeartConfig::new(&path),
        KeySource::Passphrase("wrong".into()),
    );
    assert!(opened.is_err());
}

#[test]
fn empty_frontier_exports_all_events() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("owner.spine");
    let created = SpineHeart::create(HeartConfig::new(&path), "passphrase").unwrap();
    created
        .heart
        .commit_interaction(interaction("one"))
        .unwrap();
    created
        .heart
        .commit_interaction(interaction("two"))
        .unwrap();
    let delta = created
        .heart
        .export_delta(&SyncFrontier::default())
        .unwrap();
    assert!(!delta.ciphertext.is_empty());
}

#[test]
fn tampered_sync_delta_is_rejected_without_partial_import() {
    let temp = tempfile::tempdir().unwrap();
    let first_path = temp.path().join("first.spine");
    let second_path = temp.path().join("second.spine");
    let created = SpineHeart::create(HeartConfig::new(&first_path), "first-pass").unwrap();
    let phrase = created.recovery_phrase.expose().to_owned();
    let first = created.heart;
    let second =
        SpineHeart::create_replica(HeartConfig::new(&second_path), &phrase, "second-pass").unwrap();
    first
        .commit_interaction(interaction("authenticated delta"))
        .unwrap();
    let before = second.stats().unwrap();
    let delta = first
        .export_delta(&second.sync_frontier().unwrap())
        .unwrap();
    for case in 0..512 {
        let mut corrupted = delta.clone();
        let index = (case * 7_919) % corrupted.ciphertext.len();
        corrupted.ciphertext[index] ^= 1 << (case % 8);
        assert!(second.import_delta(corrupted).is_err());
    }
    for byte in 0..delta.nonce.len() {
        for bit in 0..8 {
            let mut corrupted = delta.clone();
            corrupted.nonce[byte] ^= 1 << bit;
            assert!(second.import_delta(corrupted).is_err());
        }
    }
    for length in [0, 1, delta.ciphertext.len() / 2, delta.ciphertext.len() - 1] {
        let mut truncated = delta.clone();
        truncated.ciphertext.truncate(length);
        assert!(second.import_delta(truncated).is_err());
    }
    let mut wrong_schema = delta;
    wrong_schema.schema = u32::MAX;
    assert!(second.import_delta(wrong_schema).is_err());
    assert_eq!(second.stats().unwrap(), before);
}

#[test]
fn two_offline_writers_converge() {
    let temp = tempfile::tempdir().unwrap();
    let first_path = temp.path().join("first.spine");
    let second_path = temp.path().join("second.spine");
    let created = SpineHeart::create(HeartConfig::new(&first_path), "first-pass").unwrap();
    let phrase = created.recovery_phrase.expose().to_owned();
    let first = created.heart;
    let second =
        SpineHeart::create_replica(HeartConfig::new(&second_path), &phrase, "second-pass").unwrap();

    let first_event = first.commit_interaction(interaction("from first")).unwrap();
    second
        .commit_interaction(interaction("from second"))
        .unwrap();
    first.snapshot(Some("shared snapshot".into())).unwrap();
    first
        .redact(
            TombstoneTarget::Event(first_event.event.id),
            Some("owner requested".into()),
        )
        .unwrap();

    let to_second = first
        .export_delta(&second.sync_frontier().unwrap())
        .unwrap();
    let imported_second = second.import_delta(to_second).unwrap();
    assert_eq!(imported_second.inserted, 0);
    assert_eq!(imported_second.snapshots, 1);
    assert_eq!(imported_second.tombstones, 1);
    assert_eq!(imported_second.authorizations, 1);

    let to_first = second
        .export_delta(&first.sync_frontier().unwrap())
        .unwrap();
    let imported_first = first.import_delta(to_first).unwrap();
    assert_eq!(imported_first.inserted, 1);
    assert_eq!(imported_first.authorizations, 1);

    let first_ids: Vec<_> = first
        .events_canonical()
        .unwrap()
        .into_iter()
        .map(|event| event.id)
        .collect();
    let second_ids: Vec<_> = second
        .events_canonical()
        .unwrap()
        .into_iter()
        .map(|event| event.id)
        .collect();
    assert_eq!(first_ids, second_ids);
    assert_eq!(first.stats().unwrap(), second.stats().unwrap());
}
