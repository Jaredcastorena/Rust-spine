use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

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

    fn aggregable(self) -> bool {
        matches!(self, Self::EventAmount | Self::EventCount)
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FactAggregationIntent {
    None,
    Sum,
    Count,
    Diff,
    Max,
    Min,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FactQueryAggregation {
    Sum {
        value: f64,
        evidence: Vec<Fact>,
        money: bool,
    },
    Count {
        value: i64,
        evidence: Vec<Fact>,
    },
    Diff {
        value: f64,
        highest_value: f64,
        highest: Box<Fact>,
        lowest_value: f64,
        lowest: Box<Fact>,
        money: bool,
    },
    Max {
        value: f64,
        fact: Box<Fact>,
        money: bool,
    },
    Min {
        value: f64,
        fact: Box<Fact>,
        money: bool,
    },
}

pub struct FactAggregationRouter {
    sum: Regex,
    count: Regex,
    diff: Regex,
    max: Regex,
    min: Regex,
}

impl FactAggregationRouter {
    pub fn new() -> Result<Self> {
        fn pattern(value: &str) -> Result<Regex> {
            Regex::new(value).map_err(|error| HeartError::InvalidInput(error.to_string()))
        }

        Ok(Self {
            sum: pattern(
                r"(?i)\b(total|how much|combined|altogether|in all|sum|overall|spent|raised|donated|paid|earned|expenses?|spending|costs?|money|donations?|across all|in total|altogether)\b",
            )?,
            count: pattern(
                r"(?i)\b(how many|count|number of|times|occasions|how often|instances?|attended|visited|participated|frequency)\b",
            )?,
            diff: pattern(
                r"(?i)\b(difference|more than|less than|compared to|versus|vs\.?|between .{1,30} and|more expensive|cheaper than|pricier|cost more|cost less|higher than|lower than)\b",
            )?,
            max: pattern(
                r"(?i)\b(most|highest|maximum|most expensive|most often|biggest|largest|most times|greatest)\b",
            )?,
            min: pattern(
                r"(?i)\b(least|lowest|minimum|cheapest|fewest|smallest|least expensive|least often)\b",
            )?,
        })
    }

    pub fn detect(&self, query: &str) -> FactAggregationIntent {
        if self.diff.is_match(query) {
            FactAggregationIntent::Diff
        } else if self.max.is_match(query) {
            FactAggregationIntent::Max
        } else if self.min.is_match(query) {
            FactAggregationIntent::Min
        } else if self.sum.is_match(query) {
            FactAggregationIntent::Sum
        } else if self.count.is_match(query) {
            FactAggregationIntent::Count
        } else {
            FactAggregationIntent::None
        }
    }
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

    pub fn active_for_slot_prefix(&self, slot_prefix: &str) -> Vec<Fact> {
        let mut matching: Vec<_> = self
            .active()
            .filter(|fact| fact.slot_key.starts_with(slot_prefix))
            .cloned()
            .collect();
        matching
            .sort_by(|left, right| recency_cmp(left, right).then_with(|| left.id.cmp(&right.id)));
        matching
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
                // Python's stable BM25 ordering preserves ingestion order on a
                // tie. Hash-map/ID order and confidence must not change which
                // source is selected by a tied max/min or the forty-hit cap.
                .then_with(|| left.fact.arrival_order.cmp(&right.fact.arrival_order))
                .then_with(|| left.fact.ingest_millis.cmp(&right.fact.ingest_millis))
                .then_with(|| left.fact.id.cmp(&right.fact.id))
        });
        hits.truncate(top_k);
        hits
    }

    pub fn aggregate(&self, slot_prefix: &str, operation: &str) -> Result<FactAggregation> {
        let matching = self.active_for_slot_prefix(slot_prefix);
        match operation {
            "sum" => Ok(FactAggregation::Sum(
                matching
                    .iter()
                    .filter(|fact| fact.slot_type.aggregable())
                    .filter_map(|fact| fact.value.as_number())
                    .sum(),
            )),
            "count" => Ok(FactAggregation::Count(
                matching
                    .iter()
                    .filter(|fact| fact.slot_type.aggregable())
                    .map(|fact| match (&fact.slot_type, &fact.value) {
                        (FactSlotType::EventCount, FactValue::Integer(value)) => {
                            u64::try_from(*value).unwrap_or(0)
                        }
                        (FactSlotType::EventCount, FactValue::Amount(value))
                            if value.is_finite() && *value >= 0.0 =>
                        {
                            *value as u64
                        }
                        (FactSlotType::EventCount, FactValue::Text(value)) => value
                            .parse::<f64>()
                            .ok()
                            .filter(|value| value.is_finite() && *value >= 0.0)
                            .map_or(1, |value| value as u64),
                        _ => 1,
                    })
                    .sum(),
            )),
            "latest" => Ok(FactAggregation::Latest(
                matching.into_iter().max_by(recency_cmp).map(Box::new),
            )),
            _ => Err(HeartError::InvalidInput(
                "fact aggregation must be sum, count, or latest".into(),
            )),
        }
    }

    pub fn aggregate_query(&self, query: &str) -> Result<Option<FactQueryAggregation>> {
        let detected = FactAggregationRouter::new()?.detect(query);
        let intent = match detected {
            FactAggregationIntent::None => FactAggregationIntent::Sum,
            intent => intent,
        };
        Ok(self.aggregate_query_with_intent(query, intent))
    }

    pub fn aggregate_query_with_intent(
        &self,
        query: &str,
        intent: FactAggregationIntent,
    ) -> Option<FactQueryAggregation> {
        let facts = self
            .search(query, 40, false)
            .into_iter()
            .map(|hit| hit.fact)
            .filter(|fact| fact.slot_type.aggregable())
            .collect();
        aggregate_facts_for_intent(facts, intent)
    }
}

fn aggregate_facts_for_intent(
    facts: Vec<Fact>,
    intent: FactAggregationIntent,
) -> Option<FactQueryAggregation> {
    if facts.is_empty() || intent == FactAggregationIntent::None {
        return None;
    }

    let money = facts
        .iter()
        .any(|fact| fact.slot_type == FactSlotType::EventAmount);
    match intent {
        FactAggregationIntent::None => None,
        FactAggregationIntent::Sum => {
            let mut numeric: Vec<_> = facts
                .into_iter()
                .filter_map(|fact| query_numeric_value(&fact).map(|value| (value, fact)))
                .collect();
            if numeric.is_empty() {
                return None;
            }
            let value = numeric.iter().map(|(value, _)| value).sum();
            numeric.sort_by(|(_, left), (_, right)| recency_cmp(left, right));
            Some(FactQueryAggregation::Sum {
                value,
                evidence: numeric.into_iter().map(|(_, fact)| fact).collect(),
                money,
            })
        }
        FactAggregationIntent::Count => {
            let value = facts
                .iter()
                .map(query_count_contribution)
                .fold(0_i64, i64::saturating_add);
            (value != 0).then_some(FactQueryAggregation::Count {
                value,
                evidence: facts,
            })
        }
        FactAggregationIntent::Diff => {
            let mut numeric: Vec<_> = facts
                .into_iter()
                .filter_map(|fact| query_numeric_value(&fact).map(|value| (value, fact)))
                .collect();
            if numeric.len() < 2 {
                return None;
            }
            numeric.sort_by(|(left, _), (right, _)| left.total_cmp(right));
            let (lowest_value, lowest) = numeric.first().cloned().expect("two numeric facts");
            let (highest_value, highest) = numeric.last().cloned().expect("two numeric facts");
            Some(FactQueryAggregation::Diff {
                value: highest_value - lowest_value,
                highest_value,
                highest: Box::new(highest),
                lowest_value,
                lowest: Box::new(lowest),
                money,
            })
        }
        FactAggregationIntent::Max | FactAggregationIntent::Min => {
            let mut numeric = facts
                .into_iter()
                .filter_map(|fact| query_numeric_value(&fact).map(|value| (value, fact)));
            let (mut selected_value, mut selected_fact) = numeric.next()?;
            for (value, fact) in numeric {
                let replace = match intent {
                    FactAggregationIntent::Max => value > selected_value,
                    FactAggregationIntent::Min => value < selected_value,
                    _ => unreachable!("only max/min reach this branch"),
                };
                if replace {
                    selected_value = value;
                    selected_fact = fact;
                }
            }
            let fact = Box::new(selected_fact);
            match intent {
                FactAggregationIntent::Max => Some(FactQueryAggregation::Max {
                    value: selected_value,
                    fact,
                    money,
                }),
                FactAggregationIntent::Min => Some(FactQueryAggregation::Min {
                    value: selected_value,
                    fact,
                    money,
                }),
                _ => unreachable!("only max/min reach this branch"),
            }
        }
    }
}

fn query_numeric_value(fact: &Fact) -> Option<f64> {
    match &fact.value {
        FactValue::Text(_) => fact.value_normalized.parse().ok(),
        FactValue::Integer(value) => Some(*value as f64),
        FactValue::Amount(value) => Some(*value),
    }
}

fn query_count_contribution(fact: &Fact) -> i64 {
    if fact.slot_type != FactSlotType::EventCount {
        return 1;
    }
    match &fact.value {
        FactValue::Integer(value) => *value,
        FactValue::Amount(value) if value.is_finite() => *value as i64,
        FactValue::Text(_) => fact
            .value_normalized
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .map_or(1, |value| value as i64),
        FactValue::Amount(_) => 1,
    }
}

pub struct FactExtractor {
    rules: Vec<ExtractionRule>,
    update_cue: Regex,
    event_date: Regex,
    user_section: Regex,
    role_marker: Regex,
    leading_article: Regex,
    trailing_time: Regex,
}

struct ExtractionRule {
    regex: Regex,
    kind: ExtractionKind,
    confidence: f32,
}

#[derive(Clone, Copy)]
enum ExtractionKind {
    Single {
        attribute: &'static str,
        slot_type: FactSlotType,
        slot_key: &'static str,
        numeric: bool,
    },
    Favorite {
        slot_key_base: &'static str,
    },
    Prefer {
        slot_key: &'static str,
    },
    CountItem {
        slot_type: FactSlotType,
        slot_key_base: &'static str,
    },
    Frequency {
        slot_key_base: &'static str,
    },
    Event {
        slot_key: &'static str,
    },
    Amount {
        slot_key_base: &'static str,
    },
    Pet {
        reversed: bool,
        slot_key_base: &'static str,
    },
}

impl FactExtractor {
    pub fn new() -> Result<Self> {
        use ExtractionKind::{Amount, CountItem, Event, Favorite, Frequency, Pet, Prefer, Single};

        let specs = [
            (
                r"(?i)\bI(?:'m| am)(?: currently| now)? (\d{1,3}) years? old\b",
                Single {
                    attribute: "age",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.age",
                    numeric: true,
                },
                0.95,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: currently| now)? (\d{1,3})\b",
                Single {
                    attribute: "age",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.age",
                    numeric: true,
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:just )?turned (\d{1,3})\b",
                Single {
                    attribute: "age",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.age",
                    numeric: true,
                },
                0.90,
            ),
            (
                r"(?i)\bmy age is (\d{1,3})\b",
                Single {
                    attribute: "age",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.age",
                    numeric: true,
                },
                0.95,
            ),
            (
                r"(?i)\bI live in ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: currently)? living in ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI(?:'ve)? (?:just )?moved? to ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bI relocated to ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: currently)? based in ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI(?:'m| am) from ([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|because)\b|$)",
                Single {
                    attribute: "location",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.location",
                    numeric: false,
                },
                0.80,
            ),
            (
                r"(?i)\bI work as (?:a |an )?([A-Za-z]+(?:\s[A-Za-z]+){0,3}?)(?:\.|,|\n|and |at |$)",
                Single {
                    attribute: "occupation",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.occupation",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bmy job is (?:a |an )?([A-Za-z]+(?:\s[A-Za-z]+){0,2}?)(?:\.|,|\n|$)",
                Single {
                    attribute: "occupation",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.occupation",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: a| an) ([A-Za-z]+(?:\s[A-Za-z]+){0,2}?) (?:by profession|for a living|professionally)\b",
                Single {
                    attribute: "occupation",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.occupation",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI (?:got|started|landed) (?:a |an )?(?:new )?job as (?:a |an )?([A-Za-z]+(?:\s[A-Za-z]+){0,2}?)\b",
                Single {
                    attribute: "occupation",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.occupation",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: now)? (?:a |an )?(nurse|doctor|teacher|engineer|developer|programmer|designer|manager|lawyer|chef|pilot|therapist|accountant|architect|scientist|researcher|writer|journalist|artist|musician|student|professor|analyst|consultant|director|coordinator|supervisor|technician|mechanic|electrician|plumber|carpenter|driver|officer|agent|administrator)\b",
                Single {
                    attribute: "occupation",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.occupation",
                    numeric: false,
                },
                0.85,
            ),
            (
                r"(?i)\bI(?:'m| am)(?: a| an)? (vegetarian|vegan|pescatarian|omnivore|carnivore|flexitarian|halal|kosher|gluten.free|dairy.free|lactose.intolerant|nut.free)\b",
                Single {
                    attribute: "diet",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.diet",
                    numeric: false,
                },
                0.95,
            ),
            (
                r"(?i)\bI(?:'m| am) on (?:a |an )?([\w-]+ diet)\b",
                Single {
                    attribute: "diet",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.diet",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bmy (?:wife|husband|partner|girlfriend|boyfriend|fiancee?|spouse) (?:is(?:\s+named|\s+called)?\s+)?([A-Z][a-z]+)\b",
                Single {
                    attribute: "partner_name",
                    slot_type: FactSlotType::State,
                    slot_key: "profile.partner",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI (?:now )?have (\d+) ([\w-]+(?:\s[\w-]+){0,3}?)(?:\s*(?:now|total|in total|so far))?\s*(?:\.|,|!|\n|$)",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "inventory",
                },
                0.80,
            ),
            (
                r"(?i)\bI(?:'ve| have) (?:now |already )?(?:written|completed|finished|published) (\d+) ([\w-]+(?:\s[\w-]+){0,3}?)\b",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "activity",
                },
                0.85,
            ),
            (
                r"(?i)\bI(?:'ve| have) (?:now )?read (\d+) (pages?|chapters?)\b",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "activity.reading",
                },
                0.85,
            ),
            (
                r"(?i)\bI(?:'ve| have) (?:been to|tried|visited|attended) (\d+) ([\w-]+(?:\s[\w-]+){0,3}?)\b",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "activity",
                },
                0.80,
            ),
            (
                r"(?i)\bI(?:'ve| have) (?:done|run|completed|participated in) (\d+) ([\w-]+(?:\s[\w-]+){0,3}?)\b",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "activity",
                },
                0.80,
            ),
            (
                r"(?i)\b(?:only )?(\d+) more ([\w-]+(?:\s[\w-]+){0,2}?) (?:to go|left|until|needed)\b",
                CountItem {
                    slot_type: FactSlotType::StateQuantity,
                    slot_key_base: "target",
                },
                0.75,
            ),
            (
                r"(?i)\bI (?:go to|attend|do|practice|play|visit) ([\w-]+(?:\s[\w-]+){0,3}?) (\d+) times? (?:a |per )(week|month|year|day)\b",
                Frequency {
                    slot_key_base: "activity",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:go to|attend|do|practice|play|visit) ([\w-]+(?:\s[\w-]+){0,3}?) (once|twice|three times|four times|five times) (?:a |per )(week|month|year|day)\b",
                Frequency {
                    slot_key_base: "activity",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:went|traveled|flew|visited|took a trip) to ([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)\b",
                Event {
                    slot_key: "event.travel",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:ran|completed|finished|did) (?:a |the |my )?([\w\s-]+?(?:5k|10k|marathon|half.marathon|race|run|sprint|triathlon))\b",
                Event {
                    slot_key: "event.fitness",
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:participated|volunteered|joined|took part) (?:in |at )?(?:a |the |my )?([\w\s-]+?(?:charity|fundraiser|walkathon|run|drive|campaign))\b",
                Event {
                    slot_key: "event.charity",
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:attended|went to|saw) (?:a |the |my )?([\w\s-]+?(?:concert|party|wedding|festival|show|event|gathering))\b",
                Event {
                    slot_key: "event.social",
                },
                0.75,
            ),
            (
                r"(?i)\bI (?:started|got|landed|accepted|left|quit|was promoted) (?:a |the |my )?([\w\s-]+?(?:job|position|role|promotion|offer))\b",
                Event {
                    slot_key: "event.work",
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:had|underwent|started|finished) (?:a |the |my )?([\w\s-]+?(?:surgery|procedure|treatment|therapy|diagnosis|medication|appointment))\b",
                Event {
                    slot_key: "event.health",
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:\w+ )?(?:spent|paid|shelled out|forked out) \$(\d[\d,]*(?:\.\d{1,2})?) (?:on|for) ([\w\s-]+?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Amount {
                    slot_key_base: "expense",
                },
                0.90,
            ),
            (
                r"(?i)\bI (?:\w+ )?(?:spent|paid) (\d[\d,]*(?:\.\d{1,2})?) dollars? (?:on|for) ([\w\s-]+?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Amount {
                    slot_key_base: "expense",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:raised|donated|contributed|gave) \$(\d[\d,]*(?:\.\d{1,2})?) (?:to|for|at|toward) ([\w\s-]+?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Amount {
                    slot_key_base: "charity",
                },
                0.90,
            ),
            (
                r"(?i)\bI (?:earned|made|received|got) \$(\d[\d,]*(?:\.\d{1,2})?) (?:from|at|selling) ([\w\s-]+?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Amount {
                    slot_key_base: "income",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:attended|went to|participated in|joined) (\d+) ([\w-]+(?:\s[\w-]+){0,3}?)\b(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                CountItem {
                    slot_type: FactSlotType::EventCount,
                    slot_key_base: "attended",
                },
                0.85,
            ),
            (
                r"(?i)\bmy favou?rite ([A-Za-z]+(?:\s[A-Za-z]+){0,2}?) is ([A-Za-z]+(?:\s[A-Za-z]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Favorite {
                    slot_key_base: "preference.favorite",
                },
                0.90,
            ),
            (
                r"(?i)\bI prefer ([A-Za-z]+(?:\s[A-Za-z]+){0,3}?) (?:over|to|rather than|instead of) ([A-Za-z]+(?:\s[A-Za-z]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Prefer {
                    slot_key: "preference.general",
                },
                0.85,
            ),
            (
                r"(?i)\bI (?:really )?(?:love|adore) ([\w-]+(?:\s[\w-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Single {
                    attribute: "preference_love",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.love",
                    numeric: false,
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:hate|dislike|can't stand|cannot stand) ([\w-]+(?:\s[\w-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Single {
                    attribute: "preference_hate",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.hate",
                    numeric: false,
                },
                0.80,
            ),
            (
                r"(?i)\bI (?:enjoy|like) ([\w-]+(?:\s[\w-]+){0,3}?)(?:[.,;!?\n]|\s+(?:and|but|or|so|when|then|though|while)\b|$)",
                Single {
                    attribute: "preference_like",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.like",
                    numeric: false,
                },
                0.75,
            ),
            (
                r"(?i)\bI (?:(?:don't|do not|stopped|avoid) eat(?:ing)?|can't eat|cannot eat) ([A-Za-z]+(?:\s[A-Za-z]+){0,2}?)\b(?:\.|,|\n|$)",
                Single {
                    attribute: "diet_avoid",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.food.avoid",
                    numeric: false,
                },
                0.85,
            ),
            (
                r"(?i)\bmy hobby is ([A-Za-z0-9][A-Za-z0-9 _-]+?)(?:\.|,|\n|$)",
                Single {
                    attribute: "hobby",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.hobby",
                    numeric: false,
                },
                0.90,
            ),
            (
                r"(?i)\bI(?:'m| am) (?:a |an )?([\w\s]+?) (?:enthusiast|fan|aficionado)\b",
                Single {
                    attribute: "interest",
                    slot_type: FactSlotType::Preference,
                    slot_key: "preference.interest",
                    numeric: false,
                },
                0.80,
            ),
            (
                r"(?i)\bmy (?:son|daughter|child|kid|baby) (?:is(?:\s+named|\s+called)?\s+)?([A-Z][a-z]+)\b",
                Single {
                    attribute: "child_name",
                    slot_type: FactSlotType::Entity,
                    slot_key: "entity.family.child",
                    numeric: false,
                },
                0.85,
            ),
            (
                r"(?i)\bmy (?:brother|sister) (?:is(?:\s+named|\s+called)?\s+)?([A-Z][a-z]+)\b",
                Single {
                    attribute: "sibling_name",
                    slot_type: FactSlotType::Entity,
                    slot_key: "entity.family.sibling",
                    numeric: false,
                },
                0.80,
            ),
            (
                r"(?i)\bmy (dog|cat|rabbit|hamster|bird|fish|horse|parrot|turtle|lizard|snake|guinea pig) (?:is (?:named |called )?)?([A-Z][a-z]+)\b",
                Pet {
                    reversed: false,
                    slot_key_base: "entity.pet",
                },
                0.90,
            ),
            (
                r"(?i)\bI have (?:a |an )?([A-Z][a-z]+),? (?:who is (?:a |an )?)?(dog|cat|rabbit|hamster|bird|fish|horse|parrot|turtle|lizard|snake|guinea pig)\b",
                Pet {
                    reversed: true,
                    slot_key_base: "entity.pet",
                },
                0.85,
            ),
        ];
        let mut rules = Vec::with_capacity(specs.len());
        for (pattern, kind, confidence) in specs {
            rules.push(ExtractionRule {
                regex: Regex::new(pattern)
                    .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
                kind,
                confidence,
            });
        }
        Ok(Self {
            rules,
            update_cue: Regex::new(r"(?i)\b(actually|changed|moved|relocated|switched|no longer|now|updated|instead|correction|left|quit|got a new|started working|prefer now)\b")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
            event_date: Regex::new(r"(?i)\[date:\s*(\d{4})[/-](\d{2})[/-](\d{2})")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
            user_section: Regex::new(r"(?is)\[user\]\s*:\s*(.*?)(?:\[assistant\]\s*:|$)")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
            role_marker: Regex::new(r"(?i)\[(?:user|assistant|system|tool)\]\s*:")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
            leading_article: Regex::new(r"(?i)^(?:a|an|the)\s+")
                .map_err(|error| HeartError::InvalidInput(error.to_string()))?,
            trailing_time: Regex::new(
                r"(?i)\s+(?:last|this|past|next|the past|the last|recent)\s+(?:week|month|year|few\s+\w+|couple\s+\w+|two\s+\w+|three\s+\w+|\d+\s+\w+)|\s+(?:yesterday|today|tonight|recently|lately|now|so far|yet|already|ever)\b",
            )
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
        let event_time = self
            .event_date
            .captures(text)
            .and_then(|capture| {
                Some(format!(
                    "{}-{}-{}",
                    capture.get(1)?.as_str(),
                    capture.get(2)?.as_str(),
                    capture.get(3)?.as_str()
                ))
            })
            .or_else(|| event_time.and_then(normalize_date));
        let session_time = session_time.and_then(normalize_date);
        let Some(user_text) = self.user_text(text) else {
            return Vec::new();
        };
        let mut facts = Vec::new();
        for rule in &self.rules {
            for capture in rule.regex.captures_iter(user_text) {
                let Some(matched) = capture.get(0) else {
                    continue;
                };
                // Rust regex consumes the terminator used in Python's lookahead.
                // Start the sentence-end search at the captured value instead,
                // so a consumed period cannot pull in the following purchase.
                let value_end = capture
                    .iter()
                    .skip(1)
                    .flatten()
                    .map(|value| value.end())
                    .max()
                    .unwrap_or(matched.end());
                let excerpt = sentence_containing(user_text, matched.start(), value_end);
                let mut push = |attribute: String,
                                value: FactValue,
                                slot_type: FactSlotType,
                                slot_key: String,
                                metadata: BTreeMap<String, String>| {
                    facts.push(FactCandidate {
                        entity: "USER".into(),
                        attribute,
                        value,
                        slot_type,
                        slot_key,
                        excerpt: excerpt.clone(),
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
                        has_update_cue: self.update_cue.is_match(&excerpt),
                        metadata,
                    });
                };
                match rule.kind {
                    ExtractionKind::Single {
                        attribute,
                        slot_type,
                        slot_key,
                        numeric,
                    } => {
                        let Some(raw) = capture.get(1).map(|value| clean(value.as_str())) else {
                            continue;
                        };
                        if raw.chars().count() < 2 {
                            continue;
                        }
                        let value = if numeric {
                            raw.parse::<i64>()
                                .map(FactValue::Integer)
                                .unwrap_or_else(|_| FactValue::Text(raw.clone()))
                        } else {
                            FactValue::Text(raw)
                        };
                        push(
                            attribute.into(),
                            value,
                            slot_type,
                            slot_key.into(),
                            BTreeMap::new(),
                        );
                    }
                    ExtractionKind::Favorite { slot_key_base } => {
                        let Some(category) = capture.get(1).map(|value| clean(value.as_str()))
                        else {
                            continue;
                        };
                        let Some(value) = capture.get(2).map(|value| clean(value.as_str())) else {
                            continue;
                        };
                        let category_key = self.normalize_item(&category);
                        if category_key.is_empty() || value.is_empty() {
                            continue;
                        }
                        push(
                            format!("favorite_{category_key}"),
                            FactValue::Text(value),
                            FactSlotType::Preference,
                            format!("{slot_key_base}.{category_key}"),
                            BTreeMap::new(),
                        );
                    }
                    ExtractionKind::Prefer { slot_key } => {
                        let Some(preferred) = capture.get(1).map(|value| clean(value.as_str()))
                        else {
                            continue;
                        };
                        let Some(over) = capture.get(2).map(|value| clean(value.as_str())) else {
                            continue;
                        };
                        if preferred.is_empty() {
                            continue;
                        }
                        push(
                            "preference".into(),
                            FactValue::Text(preferred),
                            FactSlotType::Preference,
                            slot_key.into(),
                            BTreeMap::from([("over".into(), over)]),
                        );
                    }
                    ExtractionKind::CountItem {
                        slot_type,
                        slot_key_base,
                    } => {
                        let Some(number) = capture
                            .get(1)
                            .and_then(|value| value.as_str().parse::<i64>().ok())
                        else {
                            continue;
                        };
                        let Some(item) = capture.get(2).map(|value| clean(value.as_str())) else {
                            continue;
                        };
                        let item_key = self.normalize_item(&item);
                        if item_key.chars().count() < 2 {
                            continue;
                        }
                        push(
                            format!("count_{item_key}"),
                            FactValue::Integer(number),
                            slot_type,
                            format!("{slot_key_base}.{item_key}.count"),
                            BTreeMap::new(),
                        );
                    }
                    ExtractionKind::Frequency { slot_key_base } => {
                        let Some(activity) = capture.get(1).map(|value| clean(value.as_str()))
                        else {
                            continue;
                        };
                        let Some(raw_frequency) =
                            capture.get(2).map(|value| value.as_str().to_lowercase())
                        else {
                            continue;
                        };
                        let Some(period) =
                            capture.get(3).map(|value| value.as_str().to_lowercase())
                        else {
                            continue;
                        };
                        let frequency = match raw_frequency.as_str() {
                            "once" => Some(1),
                            "twice" => Some(2),
                            "three times" => Some(3),
                            "four times" => Some(4),
                            "five times" => Some(5),
                            value => value.parse::<i64>().ok(),
                        };
                        let Some(frequency) = frequency else {
                            continue;
                        };
                        let activity_key = self.normalize_item(&activity);
                        if activity_key.is_empty() {
                            continue;
                        }
                        push(
                            format!("frequency_{activity_key}"),
                            FactValue::Integer(frequency),
                            FactSlotType::Frequency,
                            format!("{slot_key_base}.{activity_key}.frequency"),
                            BTreeMap::from([("per".into(), period)]),
                        );
                    }
                    ExtractionKind::Event { slot_key } => {
                        let Some(description) = capture.get(1).map(|value| clean(value.as_str()))
                        else {
                            continue;
                        };
                        if description.chars().count() < 2 {
                            continue;
                        }
                        push(
                            "event".into(),
                            FactValue::Text(description),
                            FactSlotType::Event,
                            slot_key.into(),
                            BTreeMap::new(),
                        );
                    }
                    ExtractionKind::Amount { slot_key_base } => {
                        let Some(amount) = capture
                            .get(1)
                            .and_then(|value| value.as_str().replace(',', "").parse::<f64>().ok())
                        else {
                            continue;
                        };
                        let Some(category) = capture.get(2).map(|value| clean(value.as_str()))
                        else {
                            continue;
                        };
                        if !amount.is_finite() || amount <= 0.0 {
                            continue;
                        }
                        let category_key = self.normalize_item(&category);
                        let category_key = if category_key.is_empty() {
                            "general".into()
                        } else {
                            category_key
                        };
                        push(
                            format!("amount_{category_key}"),
                            FactValue::Amount(amount),
                            FactSlotType::EventAmount,
                            format!("{slot_key_base}.{category_key}"),
                            BTreeMap::from([("category".into(), category)]),
                        );
                    }
                    ExtractionKind::Pet {
                        reversed,
                        slot_key_base,
                    } => {
                        let (animal, name) = if reversed {
                            (capture.get(2), capture.get(1))
                        } else {
                            (capture.get(1), capture.get(2))
                        };
                        let (Some(animal), Some(name)) = (animal, name) else {
                            continue;
                        };
                        let animal = clean(animal.as_str()).to_lowercase();
                        let name = clean(name.as_str());
                        if animal.is_empty() || name.is_empty() {
                            continue;
                        }
                        push(
                            format!("pet_{animal}"),
                            FactValue::Text(name),
                            FactSlotType::Entity,
                            format!("{slot_key_base}.{animal}"),
                            BTreeMap::new(),
                        );
                    }
                }
            }
        }
        let mut seen_state = BTreeSet::new();
        facts.retain(|fact| {
            !fact.slot_type.supersedable()
                || seen_state.insert((
                    fact.entity.clone(),
                    fact.slot_key.clone(),
                    fact.value.normalized(),
                ))
        });
        facts
    }

    fn user_text<'a>(&self, text: &'a str) -> Option<&'a str> {
        if let Some(capture) = self.user_section.captures(text) {
            return capture.get(1).map(|value| value.as_str().trim());
        }
        (!self.role_marker.is_match(text)).then_some(text)
    }

    fn normalize_item(&self, raw: &str) -> String {
        let cleaned = clean(raw);
        let cleaned = self.leading_article.replace(&cleaned, "");
        let cleaned = self.trailing_time.replace_all(&cleaned, "");
        cleaned
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("_")
            .to_lowercase()
    }
}

fn clean(raw: &str) -> String {
    raw.trim()
        .trim_matches(['.', ',', ';', ':', '!', '?'])
        .to_owned()
}

fn normalize_date(raw: String) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() || raw == "?" {
        return None;
    }
    let normalized = raw.replace('/', "-");
    let mut parts = normalized.split('-');
    let valid = matches!(parts.next(), Some(year) if year.len() == 4 && year.bytes().all(|byte| byte.is_ascii_digit()))
        && matches!(parts.next(), Some(month) if month.len() == 2 && month.bytes().all(|byte| byte.is_ascii_digit()))
        && matches!(parts.next(), Some(day) if day.len() == 2 && day.bytes().all(|byte| byte.is_ascii_digit()))
        && parts.next().is_none();
    valid.then_some(normalized)
}

fn sentence_containing(text: &str, match_start: usize, match_end: usize) -> String {
    let start = text[..match_start]
        .rfind('.')
        .map_or(0, |index| index.saturating_add(1));
    let end = text[match_end..]
        .find('.')
        .map_or(text.len(), |index| match_end + index + 1);
    text[start..end].trim().chars().take(120).collect()
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
    (
        2,
        String::new(),
        fact.arrival_order[0],
        fact.arrival_order[1],
    )
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
    // Match the oracle's raw value + normalized value search document. Amounts
    // display as currency while their normalized value remains numeric.
    let (display, normalized) = match &fact.value {
        FactValue::Text(value) => (value.clone(), fact.value_normalized.clone()),
        FactValue::Integer(value) => (value.to_string(), value.to_string()),
        FactValue::Amount(value) => (format!("${value:.2}"), format!("{value:?}")),
    };
    let mut document = format!(
        "{} {} {} {} {} {}",
        fact.entity, fact.attribute, fact.slot_key, display, normalized, fact.excerpt
    );
    for key in ["over", "per", "category"] {
        if let Some(value) = fact.metadata.get(key) {
            document.push(' ');
            document.push_str(value);
        }
    }
    tokenize(&document)
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "my", "i", "me", "you", "your", "we",
    "our", "they", "their", "in", "on", "at", "to", "for", "of", "and", "or", "it", "its", "that",
    "this", "do", "does", "did", "have", "has", "had", "will", "would", "could", "should", "can",
    "not", "no",
];
