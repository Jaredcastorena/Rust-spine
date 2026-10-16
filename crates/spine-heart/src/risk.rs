use serde::{Deserialize, Serialize};

use crate::{HeartError, Result, vector};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RiskField {
    d: usize,
    channels: usize,
    retrieval_stats: usize,
    weights: Vec<f32>,
    bias: f32,
    learning_rate: f32,
    updates: u64,
}

impl RiskField {
    pub fn new(d: usize, channels: usize, retrieval_stats: usize) -> Self {
        let channels = if retrieval_stats == 10 {
            channels.saturating_mul(2)
        } else {
            channels
        };
        Self {
            d,
            channels,
            retrieval_stats,
            weights: vec![0.0; d + channels + retrieval_stats],
            bias: 0.0,
            learning_rate: 0.01,
            updates: 0,
        }
    }

    pub fn predict(&self, x: &[f32], feeling: &[f32], stats: &[f32]) -> Result<f32> {
        let features = self.features(x, feeling, stats)?;
        Ok(vector::sigmoid(
            vector::dot(&self.weights, &features) + self.bias,
        ))
    }

    pub fn update(
        &mut self,
        x: &[f32],
        feeling: &[f32],
        stats: &[f32],
        tension: f32,
    ) -> Result<f32> {
        if !tension.is_finite() || !(0.0..=1.0).contains(&tension) {
            return Err(HeartError::InvalidInput(
                "tension label must be finite and in [0, 1]".into(),
            ));
        }
        let features = self.features(x, feeling, stats)?;
        let prediction = vector::sigmoid(vector::dot(&self.weights, &features) + self.bias);
        let error = tension - prediction;
        let gradient = error * prediction * (1.0 - prediction);
        for (weight, feature) in self.weights.iter_mut().zip(features) {
            *weight += self.learning_rate * gradient * feature;
        }
        self.bias += self.learning_rate * gradient;
        self.updates += 1;
        Ok(prediction)
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Upgrade released four-feature fields without reinterpreting their learned
    /// count/empty weights as the Python score/margin features, or raw affect as
    /// normalized affect. Compatibility stores both affect segments and both
    /// retrieval segments; the newly introduced oracle coordinates start at zero.
    pub fn upgrade_retrieval_layout(
        &mut self,
        d: usize,
        channels: usize,
        declared: usize,
    ) -> Result<bool> {
        let stored_channels = if declared == 10 {
            channels.checked_mul(2)
        } else {
            Some(channels)
        }
        .ok_or_else(|| HeartError::InvalidInput("risk dimensions overflow".into()))?;
        let prefix = d
            .checked_add(stored_channels)
            .ok_or_else(|| HeartError::InvalidInput("risk dimensions overflow".into()))?;
        let expected = prefix
            .checked_add(declared)
            .ok_or_else(|| HeartError::InvalidInput("risk dimensions overflow".into()))?;
        if self.d != d
            || self.channels != stored_channels
            || self.retrieval_stats != declared
            || self.weights.len() != expected
            || !matches!(declared, 4 | 6 | 10)
            || self.weights.iter().any(|value| !value.is_finite())
            || !self.bias.is_finite()
            || !self.learning_rate.is_finite()
            || self.learning_rate < 0.0
        {
            return Err(HeartError::InvalidInput(
                "unsupported or inconsistent persisted risk feature layout".into(),
            ));
        }
        if declared != 4 {
            return Ok(false);
        }
        let upgraded_channels = channels
            .checked_mul(2)
            .ok_or_else(|| HeartError::InvalidInput("risk dimensions overflow".into()))?;
        let legacy_stats = self.weights.split_off(prefix);
        let legacy_affect = self.weights.split_off(d);
        self.weights.extend(vec![0.0; channels]);
        self.weights.extend(legacy_affect);
        self.weights.extend([0.0; 6]);
        self.weights.extend(legacy_stats);
        self.channels = upgraded_channels;
        self.retrieval_stats = 10;
        Ok(true)
    }

    pub fn retrieval_stat_dimensions(&self) -> usize {
        self.retrieval_stats
    }

    /// Python uses activated / (norm + 1e-8). Released fields keep their raw
    /// affect contribution as a second segment until online learning replaces it.
    pub fn affect_features(&self, activated: &[f32]) -> Result<Vec<f32>> {
        let expected = if self.retrieval_stats == 10 {
            self.channels / 2
        } else {
            self.channels
        };
        vector::validate_dimension(activated, expected)?;
        if self.retrieval_stats == 4 {
            return Ok(activated.to_vec());
        }
        let norm = vector::norm(activated);
        let mut features: Vec<_> = activated
            .iter()
            .map(|value| value / (norm + 1e-8))
            .collect();
        if self.retrieval_stats == 10 {
            features.extend_from_slice(activated);
        }
        Ok(features)
    }

    fn features(&self, x: &[f32], feeling: &[f32], stats: &[f32]) -> Result<Vec<f32>> {
        vector::validate_dimension(x, self.d)?;
        let compatible_feeling;
        let feeling =
            if self.retrieval_stats == 10 && feeling.len().checked_mul(2) == Some(self.channels) {
                compatible_feeling = self.affect_features(feeling)?;
                compatible_feeling.as_slice()
            } else {
                feeling
            };
        vector::validate_dimension(feeling, self.channels)?;
        // Existing four-value API callers keep their original feature contract.
        // Six-value input cannot recover legacy evidence count on its own, so a
        // migrated field requires either all ten values or its four legacy ones.
        let compatible;
        let stats = if self.retrieval_stats == 10 && stats.len() == 4 {
            compatible = [vec![0.0; 6], stats.to_vec()].concat();
            compatible.as_slice()
        } else {
            stats
        };
        vector::validate_dimension(stats, self.retrieval_stats)?;
        let mut features = Vec::with_capacity(self.weights.len());
        features.extend_from_slice(x);
        features.extend_from_slice(feeling);
        features.extend_from_slice(stats);
        Ok(features)
    }
}
