use spine_heart::{HeartConfig, SpineHeart, TombstoneTarget};

#[test]
fn inline_and_external_blobs_roundtrip_sync_and_crypto_shred() {
    let temp = tempfile::tempdir().unwrap();
    let first_path = temp.path().join("first.spine");
    let second_path = temp.path().join("second.spine");
    let created = SpineHeart::create(HeartConfig::new(&first_path), "first").unwrap();
    let phrase = created.recovery_phrase.expose().to_owned();
    let first = created.heart;
    let second =
        SpineHeart::create_replica(HeartConfig::new(&second_path), &phrase, "second").unwrap();

    let small = first
        .put_blob("text/plain", b"small encrypted body")
        .unwrap();
    let large_bytes = vec![0x5a; 1_048_576];
    let large = first
        .put_blob("application/octet-stream", &large_bytes)
        .unwrap();
    assert_eq!(
        first.blob(small.id).unwrap().unwrap().bytes,
        b"small encrypted body"
    );
    assert_eq!(first.blob(large.id).unwrap().unwrap().bytes, large_bytes);
    let external_dir = std::path::PathBuf::from(format!("{}.blobs", first_path.display()));
    assert!(external_dir.join(format!("{}.blob", large.id)).exists());

    let delta = first
        .export_delta(&second.sync_frontier().unwrap())
        .unwrap();
    let receipt = second.import_delta(delta).unwrap();
    assert_eq!(receipt.blobs, 2);
    assert_eq!(
        second.blob(small.id).unwrap().unwrap().bytes,
        b"small encrypted body"
    );
    assert_eq!(
        second.blob(large.id).unwrap().unwrap().bytes,
        vec![0x5a; 1_048_576]
    );

    first
        .redact(
            TombstoneTarget::Blob(large.id),
            Some("remove attachment".into()),
        )
        .unwrap();
    assert!(first.blob(large.id).unwrap().is_none());
    assert!(!external_dir.join(format!("{}.blob", large.id)).exists());
    let deletion = first
        .export_delta(&second.sync_frontier().unwrap())
        .unwrap();
    second.import_delta(deletion).unwrap();
    assert!(second.blob(large.id).unwrap().is_none());
    assert_eq!(first.stats().unwrap().blobs, 1);
    assert_eq!(second.stats().unwrap().blobs, 1);
}
