use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

/// Source-aware identity and provenance for one exact document version.
///
/// The identity deliberately includes both the canonical source path and the
/// original bytes. Equal bytes at different sources remain independently
/// attributable, while reading an unchanged source again produces the same ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DocumentIdentity {
    pub(crate) path: PathBuf,
    pub(crate) path_text: String,
    pub(crate) source_uri: String,
    pub(crate) document_id: String,
    pub(crate) content_sha256: String,
}

#[derive(Debug, Default)]
pub(crate) struct DocumentIngestHistory {
    document_chunks: BTreeMap<String, BTreeMap<usize, BTreeMap<usize, BTreeSet<String>>>>,
    legacy_chunks_by_source: BTreeMap<String, Vec<(usize, String)>>,
}

impl DocumentIngestHistory {
    pub(crate) fn record_document_chunk(
        &mut self,
        document_id: String,
        chunk_index: usize,
        chunk_count: usize,
        chunk_sha256: String,
    ) {
        if chunk_count == 0 || chunk_index >= chunk_count {
            return;
        }
        self.document_chunks
            .entry(document_id)
            .or_default()
            .entry(chunk_count)
            .or_default()
            .entry(chunk_index)
            .or_default()
            .insert(chunk_sha256);
    }

    pub(crate) fn record_legacy_chunk(
        &mut self,
        source_uri: String,
        chunk_index: usize,
        chunk_hash: String,
    ) {
        // Preserve the URI exactly as recorded. Resolving it now could follow a
        // retargeted symlink and incorrectly attribute old chunks to a new source.
        self.legacy_chunks_by_source
            .entry(source_uri)
            .or_default()
            .push((chunk_index, chunk_hash));
    }

    pub(crate) fn contains(&self, identity: &DocumentIdentity, legacy_hashes: &[String]) -> bool {
        self.document_is_complete(&identity.document_id)
            || self
                .legacy_chunks_by_source
                .get(&identity.source_uri)
                .is_some_and(|stored| {
                    stored.len() == legacy_hashes.len()
                        && stored.iter().zip(legacy_hashes).enumerate().all(
                            |(expected_index, ((stored_index, stored_hash), expected_hash))| {
                                *stored_index == expected_index && stored_hash == expected_hash
                            },
                        )
                })
    }

    pub(crate) fn contains_document_chunk(
        &self,
        document_id: &str,
        chunk_index: usize,
        chunk_count: usize,
        chunk_sha256: &str,
    ) -> bool {
        self.document_chunks
            .get(document_id)
            .and_then(|counts| counts.get(&chunk_count))
            .and_then(|chunks| chunks.get(&chunk_index))
            .is_some_and(|hashes| hashes.contains(chunk_sha256))
    }

    fn document_is_complete(&self, document_id: &str) -> bool {
        self.document_chunks.get(document_id).is_some_and(|counts| {
            counts.iter().any(|(count, chunks)| {
                *count > 0
                    && chunks.len() == *count
                    && (0..*count).all(|index| chunks.contains_key(&index))
            })
        })
    }
}

impl DocumentIdentity {
    pub(crate) fn from_canonical_path(path: PathBuf, raw: &[u8]) -> Result<Self, String> {
        let path_text = path
            .to_str()
            .ok_or_else(|| format!("document path is not valid UTF-8: {}", path.display()))?
            .to_owned();
        let content_sha256 = sha256_hex(raw);
        let mut identity = Sha256::new();
        identity.update(path_text.as_bytes());
        identity.update([0]);
        identity.update(raw);
        let identity_sha256 = hex::encode(identity.finalize());
        let document_id = format!("doc-{}", &identity_sha256[..20]);
        let source_uri = file_source_uri(&path_text);
        Ok(Self {
            path,
            path_text,
            source_uri,
            document_id,
            content_sha256,
        })
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn canonicalize_source(path: &Path) -> Result<PathBuf, String> {
    path.canonicalize()
        .map_err(|error| format!("could not canonicalize {}: {error}", path.display()))
}

fn file_source_uri(path_text: &str) -> String {
    format!("file://{path_text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matches_python_oracle_and_is_source_aware() {
        let first =
            DocumentIdentity::from_canonical_path(PathBuf::from("/tmp/example.txt"), b"same bytes")
                .expect("identity");
        let repeated =
            DocumentIdentity::from_canonical_path(PathBuf::from("/tmp/example.txt"), b"same bytes")
                .expect("repeated identity");
        let other_source =
            DocumentIdentity::from_canonical_path(PathBuf::from("/tmp/other.txt"), b"same bytes")
                .expect("other identity");
        let changed = DocumentIdentity::from_canonical_path(
            PathBuf::from("/tmp/example.txt"),
            b"changed bytes",
        )
        .expect("changed identity");

        assert_eq!(first, repeated);
        assert_eq!(
            first.content_sha256,
            "58100dc8fc06562ce3e578231dc948e083520ee49c4b4ee5a5a28bb4b4003feb"
        );
        assert_eq!(first.document_id, "doc-2e9dcb89f97259738fe2");
        assert_eq!(first.source_uri, "file:///tmp/example.txt");
        assert_ne!(first.document_id, other_source.document_id);
        assert_eq!(first.content_sha256, other_source.content_sha256);
        assert_ne!(first.document_id, changed.document_id);
        assert_ne!(first.content_sha256, changed.content_sha256);
    }

    #[test]
    fn legacy_fallback_is_scoped_to_the_canonical_source() {
        let first =
            DocumentIdentity::from_canonical_path(PathBuf::from("/tmp/example.txt"), b"same bytes")
                .expect("first identity");
        let second =
            DocumentIdentity::from_canonical_path(PathBuf::from("/tmp/other.txt"), b"same bytes")
                .expect("second identity");
        let legacy_hashes = vec!["legacy-chunk-hash".to_owned()];
        let mut history = DocumentIngestHistory::default();
        history.record_legacy_chunk(first.source_uri.clone(), 0, legacy_hashes[0].clone());

        assert!(history.contains(&first, &legacy_hashes));
        assert!(!history.contains(&second, &legacy_hashes));

        history.record_document_chunk(second.document_id.clone(), 0, 1, "sha256".into());
        assert!(history.contains(&second, &legacy_hashes));
    }

    #[test]
    fn new_document_identity_is_known_only_after_every_chunk_is_present() {
        let identity = DocumentIdentity::from_canonical_path(
            PathBuf::from("/tmp/example.txt"),
            b"versioned bytes",
        )
        .expect("identity");
        let mut history = DocumentIngestHistory::default();
        history.record_document_chunk(identity.document_id.clone(), 0, 2, "first".into());

        assert!(!history.contains(&identity, &[]));
        assert!(history.contains_document_chunk(&identity.document_id, 0, 2, "first"));
        assert!(!history.contains_document_chunk(&identity.document_id, 1, 2, "second"));

        history.record_document_chunk(identity.document_id.clone(), 1, 2, "second".into());
        assert!(history.contains(&identity, &[]));
    }

    #[test]
    fn legacy_fallback_requires_the_exact_indexed_chunk_sequence() {
        let identity = DocumentIdentity::from_canonical_path(
            PathBuf::from("/tmp/example.txt"),
            b"versioned bytes",
        )
        .expect("identity");
        let mut history = DocumentIngestHistory::default();
        history.record_legacy_chunk(identity.source_uri.clone(), 0, "first".into());
        history.record_legacy_chunk(identity.source_uri.clone(), 1, "second".into());

        assert!(history.contains(&identity, &["first".into(), "second".into()]));
        assert!(!history.contains(&identity, &["second".into(), "first".into()]));
        assert!(!history.contains(&identity, &["first".into()]));
    }

    #[cfg(unix)]
    #[test]
    fn legacy_file_uri_is_not_rebound_through_the_current_filesystem() {
        use std::{fs, os::unix::fs::symlink};

        let directory = tempfile::tempdir().expect("temporary directory");
        let document = directory.path().join("source.md");
        let alias = directory.path().join("alias.md");
        fs::write(&document, "same bytes").expect("document");
        symlink(&document, &alias).expect("document symlink");
        let identity = DocumentIdentity::from_canonical_path(
            document.canonicalize().expect("canonical source"),
            b"same bytes",
        )
        .expect("identity");
        let mut history = DocumentIngestHistory::default();
        history.record_legacy_chunk(format!("file://{}", alias.display()), 0, "hash".into());

        assert!(!history.contains(&identity, &["hash".into()]));
    }
}
