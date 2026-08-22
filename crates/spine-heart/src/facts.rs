use std::{cmp::Ordering, collections::BTreeMap};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{EventId, FactId, HeartError, NodeId, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FactSlotType {
    State,
    StateQuantity,
    Frequency,
    Event,
    EventAmount,
    EventCount,
    Preference,
    Entity,
}

impl FactSlotType {
    fn supersedable(self) -> bool {
        matches!(self, Self::State | Self::StateQuantity)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FactValue {
    Text(String),
    Integer(i64),
    Amount(f64),
}

impl FactValue {
    fn normalized(&self) -> String {
        match self {
            Self::Text(value) => value.trim().to_lowercase(),
            Self::Integer(value) => value.to_string(),
            Self::Amount(value) => format!("{value:.6}"),
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Amount(value) => Some(*value),
            Self::Text(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TimeSource {
    Explicit,
    Relative,
    Inferred,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactCandidate {
    pub entity: String,
    pub attribute: String,
    pub value: FactValue,
    pub slot_type: FactSlotType,
    pub slot_key: String,
    pub excerpt: String,
    pub event_time: Option<String>,
    pub session_time: Option<String>,
    pub ingest_millis: u64,
    pub time_source: TimeSource,
    pub arrival_order: [u64; 2],
    pub source_role: String,
    pub confidence: f32,
    pub has_update_cue: bool,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    pub event_id: EventId,
    pub node_id: NodeId,
    pub entity: String,
    pub attribute: String,
    pub value: FactValue,
    pub value_normalized: String,
    pub slot_type: FactSlotType,
    pub slot_key: String,
    pub excerpt: String,
    pub event_time: Option<String>,
    pub session_time: Option<String>,
    pub ingest_millis: u64,
    pub time_source: TimeSource,
    pub arrival_order: [u64; 2],
    pub source_role: String,
    pub confidence: f32,
    pub has_update_cue: bool,
    pub superseded_by: Option<FactId>,
    pub supersedes: Option<FactId>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FactHit {
    pub fact: Fact,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FactAggregation {
    Sum(f64),
    Count(u64),
    Latest(Option<Box<Fact>>),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FactStore {
    facts: BTreeMap<FactId, Fact>,
}

impl FactStore {
    pub fn facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.values()
    }

    pub fn active(&self) -> impl Iterator<Item = &Fact> {
        self.facts
            .values()
            .filter(|fact| fact.superseded_by.is_none())
    }

    pub fn add_candidates(
        &mut self,
        event_id: EventId,
        node_id: NodeId,
        candidates: Vec<FactCandidate>,
    ) -> usize {
        let count = candidates.len();
        for (index, candidate) in candidates.into_iter().enumerate() {
            let id = fact_id(event_id, index as u64, &candidate.slot_key);
            let normalized = candidate.value.normalized();
            self.facts.entry(id).or_insert(Fact {
                id,
                event_id,
                node_id,
                entity: candidate.entity,
                attribute: candidate.attribute,
                value: candidate.value,
                value_normalized: normalized,
                slot_type: candidate.slot_type,
                slot_key: candidate.slot_key,
                excerpt: candidate.excerpt,
                event_time: candidate.event_time,
                session_time: candidate.session_time,
                ingest_millis: candidate.ingest_millis,
                time_source: candidate.time_source,
                arrival_order: candidate.arrival_order,
                source_role: candidate.source_role,
                confidence: candidate.confidence,
                has_update_cue: candidate.has_update_cue,
                superseded_by: None,
                supersedes: None,
                metadata: candidate.metadata,
            });
        }
        self.rebuild_supersession();
        count
    }

    pub fn rebuild_supersession(&mut self) -> usize {
        for fact in self.facts.values_mut() {
            fact.superseded_by = None;
            fact.supersedes = None;
        }
        let mut groups: BTreeMap<(String, String), Vec<FactId>> = BTreeMap::new();
        for fact in self.facts.values() {
            if fact.slot_type.supersedable() {
                groups
                    .entry((fact.entity.clone(), fact.slot_key.clone()))
                    .or_default()
                    .push(fact.id);
            }
        }
        let mut edges = 0;
        for ids in groups.values_mut() {
            ids.sort_by(|left, right| {
                let left = &self.facts[left];
                let right = &self.facts[right];
                recency_cmp(left, right).then_with(|| left.id.cmp(&right.id))
            });
            for pair in ids.windows(2) {
                let older = self.facts[&pair[0]].clone();
                let newer = self.facts[&pair[1]].clone();
                if recency_cmp(&older, &newer) == Ordering::Equal
                    || older.value_normalized == newer.value_normalized
                {
                    continue;
                }
                self.facts
                    .get_mut(&older.id)
                    .expect("fact exists")
                    .superseded_by = Some(newer.id);
                let newer = self.facts.get_mut(&newer.id).expect("fact exists");
                if newer.supersedes.is_none() {
                    newer.supersedes = Some(older.id);
                }
                edges += 1;
            }
        }
        edges
    }

    pub fn search(&self, query: &str, top_k: usize, include_superseded: bool) -> Vec<FactHit> {
        let documents: Vec<&Fact> = self.facts.values().collect();
        if documents.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }
        let document_terms: Vec<Vec<String>> =
            documents.iter().map(|fact| fact_tokens(fact)).collect();
        let average_length =
            document_terms.iter().map(Vec::len).sum::<usize>() as f32 / documents.len() as f32;
        let mut document_frequency = BTreeMap::<String, usize>::new();
        for terms in &document_terms {
            let unique: std::collections::BTreeSet<_> = terms.iter().collect();
            for term in unique {
                *document_frequency.entry(term.clone()).or_default() += 1;
            }
        }
        let mut hits = Vec::new();
        for (fact, terms) in documents.into_iter().zip(document_terms) {
            if !include_superseded && fact.superseded_by.is_some() {
                continue;
            }
            let mut frequencies = BTreeMap::<&str, usize>::new();
            for term in &terms {
                *frequencies.entry(term).or_default() += 1;
            }
            let mut score = 0.0;
            for query_term in &query_terms {
                let Some(&frequency) = frequencies.get(query_term.as_str()) else {
                    continue;
                };
                let document_count = self.facts.len() as f32;
                let containing = document_frequency
                    .get(query_term)
                    .copied()
                    .unwrap_or_default() as f32;
                let inverse = ((document_count - containing + 0.5) / (containing + 0.5) + 1.0).ln();
                let frequency = frequency as f32;
                let denominator = frequency
                    + 1.5 * (1.0 - 0.75 + 0.75 * terms.len() as f32 / average_length.max(1.0));
                score += inverse * frequency * 2.5 / denominator;
            }
            if score > 0.0 {
                hits.push(FactHit {
                    fact: fact.clone(),
                    score,
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.fact.confidence.total_cmp(&left.fact.confidence))
                .then_with(|| left.fact.id.cmp(&right.fact.id))
        });
        hits.truncate(top_k);
        hits
    }

    pub fn aggregate(&self, slot_prefix: &str, operation: &str) -> Result<FactAggregation> {
        let matching: Vec<_> = self
            .active()
            .filter(|fact| fact.slot_key.starts_with(slot_prefix))
            .collect();
        match operation {
            "sum" => Ok(FactAggregation::Sum(
                matching
                    .iter()
                    .filter_map(|fact| fact.value.as_number())
                    .sum(),
            )),
            "count" => Ok(FactAggregation::Count(matching.len() as u64)),
            "latest" => Ok(FactAggregation::Latest(
                matching
                    .into_iter()
                    .max_by(|left, right| recency_cmp(left, right))
                    .cloned()
                    .map(Box::new),
            )),
            _ => Err(HeartError::InvalidInput(
                "fact aggregation must be sum, count, or latest".into(),
            )),
        }
    }
}

pub struct FactExtractor {
    rules: Vec<ExtractionRule>,
    update_cue: Regex,
}

struct ExtractionRule {
    regex: Regex,
    attribute: &'static str,
    slot_type: FactSlotType,
    slot_key: &'static str,
    confidence: f32,
    numeric: bool,
}

impl FactExtractor {
    pub fn new() -> Result<Self> {
        let specs = [
            (
                r"(?i)\bI(?:'m| am)(?: currently| now)? (\d{1,3}) years? old\b",
                "age",
                FactSlotType::State,
                "profile.age",
                0.95,
                true,
            ),
            (
                r"(?i)\bI (?:live|moved|relocated) (?:in|to) ([A-Za-z][A-Za-z -]+?)(?:[.,;!?]|$)",
                "location",
                FactSlotType::State,
                "profile.location",
                0.95,
                false,
            ),
            (
                r"(?i)\bI work as (?:a |an )?([A-Za-z][A-Za-z -]+?)(?:[.,;!?]|$)",
                "occupation",
                FactSlotType::State,
                "profile.occupation",
                0.95,
                false,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: a| an)? (vegetarian|vegan|pescatarian|omnivore|flexitarian)\b",
                "diet",
                FactSlotType::State,
                "profile.diet",
                0.95,
                false,
            ),
            (
                r"(?i)\bI (?:really )?(?:love|adore) ([A-Za-z0-9][A-Za-z0-9 _-]+?)(?:[.,;!?]|$)",
                "preference_love",
                FactSlotType::Preference,
                "preference.love",
                0.80,
                false,
            ),
            (
                r"(?i)\bI (?:hate|dislike|can't stand|cannot stand) ([A-Za-z0-9][A-Za-z0-9 _-]+?)(?:[.,;!?]|$)",
                "preference_hate",
                FactSlotType::Preference,
                "preference.hate",
                0.80,
                false,
            ),
            (
                r"(?i)\bmy hobby is ([A-Za-z0-9][A-Za-z0-9 _-]+?)(?:[.,;!?]|$)",
                "hobby",
                FactSlotType::Preference,
                "preference.hobby",
                0.90,
                false,
            ),
        ];
        let mut rules = Vec::new();
        for (pattern, attribute, slot_type, slot_key, confidence, numeric) in specs {
            rules.push(ExtractionRule {
                regex: Regex::new(pattern)
                    .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
                attribute,
                slot_type,
                slot_key,
                confidence,
                numeric,
            });
        }
        Ok(Self {
            rules,
            update_cue: Regex::new(r"(?i)\b(actually|changed|moved|relocated|switched|no longer|now|updated|instead|correction|left|quit|got a new|started working|prefer now)\b")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
        })
    }

    pub fn extract(
        &self,
        text: &str,
        event_time: Option<String>,
        session_time: Option<String>,
        ingest_millis: u64,
        arrival_order: [u64; 2],
    ) -> Vec<FactCandidate> {
        let mut facts = Vec::new();
        for rule in &self.rules {
            for capture in rule.regex.captures_iter(text) {
                let Some(value) = capture.get(1) else {
                    continue;
                };
                let raw = value.as_str().trim().trim_matches(['.', ',', ';']);
                if raw.is_empty() {
                    continue;
                }
                let value = if rule.numeric {
                    raw.parse::<i64>()
                        .map(FactValue::Integer)
                        .unwrap_or_else(|_| FactValue::Text(raw.into()))
                } else {
                    FactValue::Text(raw.into())
                };
                facts.push(FactCandidate {
                    entity: "USER".into(),
                    attribute: rule.attribute.into(),
                    value,
                    slot_type: rule.slot_type,
                    slot_key: rule.slot_key.into(),
                    excerpt: capture
                        .get(0)
                        .map_or(raw, |matched| matched.as_str())
                        .into(),
                    event_time: event_time.clone(),
                    session_time: session_time.clone(),
                    ingest_millis,
                    time_source: if event_time.is_some() {
                        TimeSource::Explicit
                    } else {
                        TimeSource::Inferred
                    },
                    arrival_order,
                    source_role: "user".into(),
                    confidence: rule.confidence,
                    has_update_cue: self.update_cue.is_match(text),
                    metadata: BTreeMap::new(),
                });
            }
        }
        facts
    }
}

fn fact_id(event_id: EventId, index: u64, slot_key: &str) -> FactId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"spine-fact-v1");
    hasher.update(event_id.as_bytes());
    hasher.update(&index.to_be_bytes());
    hasher.update(slot_key.as_bytes());
    FactId::from_bytes(*hasher.finalize().as_bytes())
}

fn recency_cmp(left: &Fact, right: &Fact) -> Ordering {
    recency_key(left).cmp(&recency_key(right))
}

fn recency_key(fact: &Fact) -> (u8, String, u64, u64) {
    if matches!(
        fact.time_source,
        TimeSource::Explicit | TimeSource::Relative
    ) && let Some(event_time) = &fact.event_time
    {
        return (0, event_time.clone(), 0, 0);
    }
    if let Some(session_time) = &fact.session_time {
        return (1, session_time.clone(), 0, 0);
    }
    if fact.arrival_order != [0, 0] {
        return (
            2,
            String::new(),
            fact.arrival_order[0],
            fact.arrival_order[1],
        );
    }
    (3, fact.ingest_millis.to_string(), 0, 0)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .filter_map(|term| {
            let term = term.to_lowercase();
            (term.len() > 1 && !STOPWORDS.contains(&term.as_str())).then_some(term)
        })
        .collect()
}

fn fact_tokens(fact: &Fact) -> Vec<String> {
    tokenize(&format!(
        "{} {} {} {} {}",
        fact.entity, fact.attribute, fact.slot_key, fact.value_normalized, fact.excerpt
    ))
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "my", "i", "me", "you", "your", "we",
    "our", "they", "their", "in", "on", "at", "to", "for", "of", "and", "or", "it", "its", "that",
    "this", "do", "does", "did", "have", "has", "had",
];
