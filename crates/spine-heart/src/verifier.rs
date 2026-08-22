use std::collections::BTreeSet;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{HeartError, Result};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ClaimRelation {
    Equals,
    GreaterThan,
    LessThan,
    HasCount,
    HasAmount,
    OccurredOn,
    StartedOn,
    EndedOn,
    DurationIs,
    IsA,
    PartOf,
    LocatedIn,
    AuthoredBy,
    Possesses,
}

pub struct ClaimExtractor {
    money: Regex,
    count: Regex,
    equals: Regex,
    date: Regex,
    duration: Regex,
    is_a: Regex,
    location: Regex,
    possession: Regex,
    author: Regex,
    proper: Regex,
    first_person: Regex,
}

impl ClaimExtractor {
    pub fn new() -> Result<Self> {
        fn pattern(value: &str) -> Result<Regex> {
            Regex::new(value).map_err(|error| HeartError::InvalidInput(error.to_string()))
        }
        Ok(Self {
            money: pattern(r"(?i)\$\s*(\d[\d,]*(?:\.\d{1,2})?)")?,
            count: pattern(
                r"(?i)\b(\d+)\s+(?:years?\s+old|sessions?|events?|times?|days?|weeks?|months?|items?|books?|movies?|songs?|pages?|chapters?|restaurants?|weddings?|projects?|tasks?|fish|stars?|points?|miles?|km|steps?|reps?|sets?|minutes?|hours?|pounds?|kilograms?|kg|lbs?)\b",
            )?,
            equals: pattern(r"(?i)(?:is|was|equals?|=)\s+(\d[\d,]*(?:\.\d+)?)\b")?,
            date: pattern(r"\b(\d{4}-\d{2}-\d{2})\b")?,
            duration: pattern(
                r"(?i)lasted\s+(\d+\s+(?:days?|hours?|minutes?|weeks?|months?|years?))",
            )?,
            is_a: pattern(
                r"\b([A-Z][a-zA-Z\s]{2,30}?)\s+(?:is|was|are|were)\s+(?:a|an)\s+([a-zA-Z\s]{3,40}?)(?:\.|,|;|$)",
            )?,
            location: pattern(
                r"(?i)\b(?:lives?\s+in|located\s+in|based\s+in|situated\s+in|resides?\s+in|moved?\s+to|in\s+the\s+city\s+of)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
            )?,
            possession: pattern(r"\b([A-Z][a-zA-Z\s]{2,25}?)'s\s+([a-zA-Z\s]{3,30}?)\b")?,
            author: pattern(
                r"(?i)\b(?:written|authored|created|composed|directed|published)\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
            )?,
            proper: pattern(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")?,
            first_person: pattern(r"(?i)\b(I|me|my|the user|the person)\b")?,
        })
    }

    pub fn extract(&self, text: &str) -> Vec<AtomicClaim> {
        let mut claims = Vec::new();
        for sentence in sentence_spans(text) {
            let date = self
                .date
                .captures(sentence)
                .and_then(|captures| captures.get(1))
                .map(|value| value.as_str().to_owned());
            let subject = self.subject(sentence);
            let excerpt: String = sentence.chars().take(120).collect();
            let money = self
                .money
                .captures(sentence)
                .and_then(|captures| captures.get(1));
            let count = self
                .count
                .captures(sentence)
                .and_then(|captures| captures.get(1));
            if let Some(value) = money {
                claims.push(AtomicClaim {
                    subject: subject.clone(),
                    relation: ClaimRelation::HasAmount,
                    value: format!("${}", value.as_str()),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            } else if let Some(value) = count {
                claims.push(AtomicClaim {
                    subject: subject.clone(),
                    relation: ClaimRelation::HasCount,
                    value: value.as_str().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            } else if let Some(value) = self
                .equals
                .captures(sentence)
                .and_then(|captures| captures.get(1))
            {
                claims.push(AtomicClaim {
                    subject: subject.clone(),
                    relation: ClaimRelation::Equals,
                    value: value.as_str().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            }
            let lowered = sentence.to_lowercase();
            for (words, relation) in [
                (
                    ["occurred", "happened", "took place", "was held"].as_slice(),
                    ClaimRelation::OccurredOn,
                ),
                (
                    ["started", "began", "commenced", "launched"].as_slice(),
                    ClaimRelation::StartedOn,
                ),
                (
                    ["ended", "finished", "concluded", "completed", "stopped"].as_slice(),
                    ClaimRelation::EndedOn,
                ),
            ] {
                if let Some(date) = &date
                    && words.iter().any(|word| lowered.contains(word))
                {
                    claims.push(AtomicClaim {
                        subject: subject.clone(),
                        relation,
                        value: date.clone(),
                        time: None,
                        excerpt: excerpt.clone(),
                    });
                }
            }
            if let Some(value) = self
                .duration
                .captures(sentence)
                .and_then(|captures| captures.get(1))
            {
                claims.push(AtomicClaim {
                    subject: subject.clone(),
                    relation: ClaimRelation::DurationIs,
                    value: value.as_str().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            }
            if let Some(captures) = self.is_a.captures(sentence)
                && let (Some(entity), Some(value)) = (captures.get(1), captures.get(2))
            {
                claims.push(AtomicClaim {
                    subject: entity.as_str().trim().into(),
                    relation: ClaimRelation::IsA,
                    value: value.as_str().trim().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            }
            if let Some(value) = self
                .location
                .captures(sentence)
                .and_then(|captures| captures.get(1))
            {
                claims.push(AtomicClaim {
                    subject: subject.clone(),
                    relation: ClaimRelation::LocatedIn,
                    value: value.as_str().trim().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            }
            if let Some(captures) = self.possession.captures(sentence)
                && let (Some(entity), Some(value)) = (captures.get(1), captures.get(2))
            {
                claims.push(AtomicClaim {
                    subject: entity.as_str().trim().into(),
                    relation: ClaimRelation::Possesses,
                    value: value.as_str().trim().into(),
                    time: date.clone(),
                    excerpt: excerpt.clone(),
                });
            }
            if let Some(value) = self
                .author
                .captures(sentence)
                .and_then(|captures| captures.get(1))
            {
                claims.push(AtomicClaim {
                    subject,
                    relation: ClaimRelation::AuthoredBy,
                    value: value.as_str().trim().into(),
                    time: date,
                    excerpt,
                });
            }
        }
        let mut seen = BTreeSet::new();
        claims.retain(|claim| {
            seen.insert((
                claim.subject.to_lowercase(),
                claim.relation,
                claim.value.to_lowercase(),
            ))
        });
        claims
    }

    fn subject(&self, sentence: &str) -> String {
        if self.first_person.is_match(sentence) {
            return "the user".into();
        }
        self.proper
            .captures(sentence)
            .and_then(|captures| captures.get(1))
            .map_or_else(|| "the subject".into(), |value| value.as_str().into())
    }
}

fn sentence_spans(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    for (index, character) in text.char_indices() {
        if matches!(character, '.' | '!' | '?') {
            let end = index + character.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                result.push(sentence);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        result.push(tail);
    }
    result
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtomicClaim {
    pub subject: String,
    pub relation: ClaimRelation,
    pub value: String,
    pub time: Option<String>,
    pub excerpt: String,
}

impl AtomicClaim {
    pub fn hypothesis(&self) -> String {
        let prefix = self
            .time
            .as_ref()
            .map_or_else(String::new, |time| format!("At time {time}, "));
        match self.relation {
            ClaimRelation::Equals => {
                format!("{prefix}{} was equal to {}.", self.subject, self.value)
            }
            ClaimRelation::GreaterThan => {
                format!("{prefix}{} was greater than {}.", self.subject, self.value)
            }
            ClaimRelation::LessThan => {
                format!("{prefix}{} was less than {}.", self.subject, self.value)
            }
            ClaimRelation::HasCount => {
                format!("{prefix}{} had a count of {}.", self.subject, self.value)
            }
            ClaimRelation::HasAmount => {
                format!("{prefix}{} had an amount of {}.", self.subject, self.value)
            }
            ClaimRelation::OccurredOn => {
                format!("The event {} occurred on {}.", self.subject, self.value)
            }
            ClaimRelation::StartedOn => {
                format!("The event {} started on {}.", self.subject, self.value)
            }
            ClaimRelation::EndedOn => {
                format!("The event {} ended on {}.", self.subject, self.value)
            }
            ClaimRelation::DurationIs => {
                format!("The duration of {} was {}.", self.subject, self.value)
            }
            ClaimRelation::IsA => {
                format!("{prefix}{} was a {}.", self.subject, self.value)
            }
            ClaimRelation::PartOf => {
                format!("{prefix}{} was part of {}.", self.subject, self.value)
            }
            ClaimRelation::LocatedIn => {
                format!("{prefix}{} was located in {}.", self.subject, self.value)
            }
            ClaimRelation::AuthoredBy => {
                format!("{prefix}{} was authored by {}.", self.subject, self.value)
            }
            ClaimRelation::Possesses => {
                format!("{prefix}{} possessed {}.", self.subject, self.value)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NliLabelOrder {
    pub contradiction: usize,
    pub entailment: usize,
    pub neutral: usize,
}

impl NliLabelOrder {
    pub const CROSS_ENCODER_MINILM: Self = Self {
        contradiction: 0,
        entailment: 1,
        neutral: 2,
    };

    fn validate(self) -> Result<()> {
        let mut labels = [self.contradiction, self.entailment, self.neutral];
        labels.sort_unstable();
        if labels == [0, 1, 2] {
            Ok(())
        } else {
            Err(HeartError::InvalidInput(
                "NLI label order must be a permutation of 0, 1, 2".into(),
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NliProbabilities {
    pub entailment: f32,
    pub neutral: f32,
    pub contradiction: f32,
}

pub trait NliModel: Send + Sync {
    fn label_order(&self) -> NliLabelOrder;
    fn predict_logits(&self, pairs: &[(String, String)]) -> Result<Vec<[f32; 3]>>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NliReport {
    pub coverage: f32,
    pub contradiction: f32,
    pub claim_support: Vec<f32>,
    pub claim_contradiction: Vec<f32>,
    pub evaluated_pairs: usize,
}

pub struct NliVerifier<M> {
    model: M,
    top_m: usize,
}

impl<M: NliModel> NliVerifier<M> {
    pub fn new(model: M, top_m: usize) -> Result<Self> {
        model.label_order().validate()?;
        if top_m == 0 {
            return Err(HeartError::InvalidInput(
                "NLI evidence fanout must be positive".into(),
            ));
        }
        Ok(Self { model, top_m })
    }

    pub fn verify(&self, claims: &[AtomicClaim], evidence: &[String]) -> Result<NliReport> {
        if claims.is_empty() || evidence.is_empty() {
            return Ok(NliReport::default());
        }
        let mut pairs = Vec::new();
        let mut owners = Vec::new();
        for (claim_index, claim) in claims.iter().enumerate() {
            let hypothesis = claim.hypothesis();
            for span in select_evidence(&hypothesis, evidence, self.top_m) {
                pairs.push((hypothesis.clone(), span.clone()));
                owners.push(claim_index);
            }
        }
        let logits = self.model.predict_logits(&pairs)?;
        if logits.len() != pairs.len() {
            return Err(HeartError::InvalidInput(
                "NLI model returned the wrong batch length".into(),
            ));
        }
        let order = self.model.label_order();
        let mut support = vec![0.0_f32; claims.len()];
        let mut contradiction = vec![0.0_f32; claims.len()];
        for (claim_index, logits) in owners.into_iter().zip(logits) {
            let probabilities = probabilities(logits, order);
            support[claim_index] = support[claim_index].max(probabilities.entailment);
            contradiction[claim_index] =
                contradiction[claim_index].max(probabilities.contradiction);
        }
        Ok(NliReport {
            coverage: support.iter().sum::<f32>() / claims.len() as f32,
            contradiction: contradiction.iter().copied().fold(0.0, f32::max),
            claim_support: support,
            claim_contradiction: contradiction,
            evaluated_pairs: pairs.len(),
        })
    }
}

pub fn probabilities(logits: [f32; 3], order: NliLabelOrder) -> NliProbabilities {
    let maximum = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let values = logits.map(|value| (value - maximum).exp());
    let total = values.iter().sum::<f32>().max(1e-12);
    NliProbabilities {
        entailment: values[order.entailment] / total,
        neutral: values[order.neutral] / total,
        contradiction: values[order.contradiction] / total,
    }
}

fn select_evidence<'a>(hypothesis: &str, evidence: &'a [String], top_m: usize) -> Vec<&'a String> {
    if evidence.len() <= top_m {
        return evidence.iter().collect();
    }
    let hypothesis_tokens = tokens(hypothesis);
    let mut scored: Vec<_> = evidence
        .iter()
        .enumerate()
        .map(|(index, span)| {
            let span_tokens = tokens(span);
            let overlap = hypothesis_tokens
                .iter()
                .filter(|token| span_tokens.contains(token.as_str()))
                .count();
            (overlap, index, span)
        })
        .collect();
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    scored
        .into_iter()
        .take(top_m)
        .map(|(_, _, span)| span)
        .collect()
}

fn tokens(text: &str) -> std::collections::BTreeSet<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}
