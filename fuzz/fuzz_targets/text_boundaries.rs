#![no_main]

use std::{str::FromStr, sync::OnceLock};

use libfuzzer_sys::fuzz_target;
use spine_heart::{AgentId, ClaimExtractor, EventId, NodeId, ThreadId};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    static EXTRACTOR: OnceLock<ClaimExtractor> = OnceLock::new();
    let extractor =
        EXTRACTOR.get_or_init(|| ClaimExtractor::new().expect("static claim regexes compile"));
    let first = extractor.extract(&text);
    let second = extractor.extract(&text);
    assert_eq!(first, second);
    assert!(
        first
            .iter()
            .all(|claim| claim.excerpt.chars().count() <= 120)
    );
    for (index, claim) in first.iter().enumerate() {
        assert!(!first[..index].contains(claim));
    }
    let _ = EventId::from_str(&text);
    let _ = NodeId::from_str(&text);
    let _ = AgentId::new(text.to_string());
    let _ = ThreadId::new(text.to_string());
});
