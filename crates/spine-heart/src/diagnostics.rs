use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{AgentId, EventId, FeelingVector, HeartError, Result, Thymos};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FeelingObservation {
    pub agent: AgentId,
    pub event_id: EventId,
    pub update_count: u64,
    pub valence: f32,
    pub arousal: f32,
    pub state_hash: [u8; 32],
    pub predecessor_hash: [u8; 32],
}

impl FeelingObservation {
    pub fn new(
        agent: AgentId,
        event_id: EventId,
        thymos: &Thymos,
        feeling: &FeelingVector,
        predecessor_hash: [u8; 32],
    ) -> Result<Self> {
        Ok(Self {
            agent,
            event_id,
            update_count: thymos.update_count(),
            valence: feeling.valence,
            arousal: feeling.arousal,
            state_hash: thymos_hash(thymos)?,
            predecessor_hash,
        })
    }
}

/// Separate derived projection: adding diagnostics must not change the nested
/// postcard layout of existing cognitive or Thymos state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DiagnosticHistory {
    schema: u32,
    agents: BTreeMap<AgentId, VecDeque<FeelingObservation>>,
}

impl Default for DiagnosticHistory {
    fn default() -> Self {
        Self {
            schema: 1,
            agents: BTreeMap::new(),
        }
    }
}

impl DiagnosticHistory {
    pub fn add(&mut self, observations: &[FeelingObservation]) -> Result<()> {
        self.validate_schema()?;
        for sample in observations {
            if sample.update_count == 0
                || !sample.valence.is_finite()
                || !sample.arousal.is_finite()
            {
                return Err(HeartError::InvalidInput(
                    "invalid diagnostic feeling sample".into(),
                ));
            }
            let samples = self.agents.entry(sample.agent.clone()).or_default();
            if samples.iter().any(|known| {
                known.update_count == sample.update_count
                    && known.state_hash == sample.state_hash
                    && known.event_id == sample.event_id
                    && known.valence == sample.valence
                    && known.arousal == sample.arousal
            }) {
                continue;
            }
            if samples.back().is_some_and(|last| {
                last.state_hash != sample.predecessor_hash
                    || last.update_count.checked_add(1) != Some(sample.update_count)
            }) {
                // Promotion, replacement, or an unobserved learning update starts
                // a new known segment; never join unrelated tensor histories.
                samples.clear();
            }
            samples.push_back(sample.clone());
            while samples.len() > 100 {
                samples.pop_front();
            }
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<()> {
        if self.schema != 1 {
            return Err(HeartError::UnsupportedSchema {
                found: self.schema,
                expected: 1,
            });
        }
        Ok(())
    }

    pub fn summary(
        &self,
        agent: &AgentId,
        update_count: u64,
        live_events: &BTreeSet<EventId>,
        current_hash: [u8; 32],
    ) -> Result<serde_json::Value> {
        self.validate_schema()?;
        let known: Vec<_> = self
            .agents
            .get(agent)
            .into_iter()
            .flatten()
            .filter(|sample| {
                sample.update_count <= update_count && live_events.contains(&sample.event_id)
            })
            .collect();
        let recent: Vec<_> = known.iter().rev().take(10).collect();
        let current = recent.first().is_some_and(|sample| {
            sample.update_count == update_count && sample.state_hash == current_hash
        });
        let contiguous = recent.iter().enumerate().all(|(index, sample)| {
            sample.update_count == update_count.saturating_sub(index as u64)
        });
        let trend = |value: fn(&FeelingObservation) -> f32| -> Option<f64> {
            (!recent.is_empty() && current).then(|| {
                recent
                    .iter()
                    .map(|sample| f64::from(value(sample)))
                    .sum::<f64>()
                    / recent.len() as f64
            })
        };
        Ok(serde_json::json!({
            "valence_trend": trend(|sample| sample.valence),
            "arousal_trend": trend(|sample| sample.arousal),
            "history_length": known.len(),
            "trend_sample_count": recent.len(),
            "trend_window_complete": current && contiguous && recent.len() == 10,
            "history_current": current,
            "history_source": "known_observed_updates",
        }))
    }
}

pub(crate) fn thymos_hash(thymos: &Thymos) -> Result<[u8; 32]> {
    Ok(*blake3::hash(&postcard::to_allocvec(thymos)?).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_identity_includes_observed_evidence_and_feeling() {
        let agent = AgentId::new("main").unwrap();
        let mut history = DiagnosticHistory::default();
        let first = FeelingObservation {
            agent: agent.clone(),
            event_id: EventId::from_bytes([1; 32]),
            update_count: 5,
            valence: 0.1,
            arousal: 0.2,
            state_hash: [2; 32],
            predecessor_hash: [1; 32],
        };
        let mut replacement = first.clone();
        replacement.event_id = EventId::from_bytes([3; 32]);
        history.add(&[first, replacement.clone()]).unwrap();
        let live = BTreeSet::from([replacement.event_id]);
        assert_eq!(
            history.summary(&agent, 5, &live, [2; 32]).unwrap()["history_length"],
            1
        );
        replacement.valence = 0.75;
        history.add(&[replacement.clone()]).unwrap();
        assert_eq!(
            history.summary(&agent, 5, &live, [2; 32]).unwrap()["valence_trend"],
            0.75
        );
        replacement.arousal = 0.5;
        history.add(&[replacement]).unwrap();
        assert_eq!(
            history.summary(&agent, 5, &live, [2; 32]).unwrap()["arousal_trend"],
            0.5
        );
    }

    #[test]
    fn bounded_history_matches_ten_sample_mean_and_rejects_stale_evidence() {
        let agent = AgentId::new("main").unwrap();
        let mut history = DiagnosticHistory::default();
        let samples: Vec<_> = (1_u8..=105)
            .map(|index| FeelingObservation {
                agent: agent.clone(),
                event_id: EventId::from_bytes([index; 32]),
                update_count: u64::from(index),
                valence: f32::from(index),
                arousal: f32::from(index) * 2.0,
                state_hash: [0; 32],
                predecessor_hash: [0; 32],
            })
            .collect();
        let mut events: BTreeSet<_> = samples.iter().map(|sample| sample.event_id).collect();
        history.add(&samples).unwrap();
        history.add(&samples[100..]).unwrap();
        let summary = history.summary(&agent, 105, &events, [0; 32]).unwrap();
        assert_eq!(summary["history_length"], 100);
        assert_eq!(summary["valence_trend"], 100.5);
        assert_eq!(summary["arousal_trend"], 201.0);
        assert_eq!(summary["trend_window_complete"], true);
        assert!(
            history.summary(&agent, 105, &events, [1; 32]).unwrap()["valence_trend"].is_null(),
            "same count with changed tensor is stale"
        );
        assert!(
            history.summary(&agent, 106, &events, [0; 32]).unwrap()["valence_trend"].is_null(),
            "unrecorded catch-up update is unknown"
        );
        events.remove(&samples.last().unwrap().event_id);
        let summary = history.summary(&agent, 105, &events, [0; 32]).unwrap();
        assert_eq!(summary["history_current"], false);
        assert!(summary["valence_trend"].is_null());
        assert_eq!(summary["trend_window_complete"], false);
        let replacement = FeelingObservation {
            agent: agent.clone(),
            event_id: samples[0].event_id,
            update_count: 2,
            valence: 0.5,
            arousal: 0.25,
            state_hash: [2; 32],
            predecessor_hash: [1; 32],
        };
        history.add(&[replacement]).unwrap();
        let summary = history.summary(&agent, 2, &events, [2; 32]).unwrap();
        assert_eq!(summary["history_length"], 1);
        assert_eq!(summary["valence_trend"], 0.5);
        assert_eq!(summary["trend_window_complete"], false);
    }
}
