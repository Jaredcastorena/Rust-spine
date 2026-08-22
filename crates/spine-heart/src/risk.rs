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

    fn features(&self, x: &[f32], feeling: &[f32], stats: &[f32]) -> Result<Vec<f32>> {
        vector::validate_dimension(x, self.d)?;
        vector::validate_dimension(feeling, self.channels)?;
        vector::validate_dimension(stats, self.retrieval_stats)?;
        let mut features = Vec::with_capacity(self.weights.len());
        features.extend_from_slice(x);
        features.extend_from_slice(feeling);
        features.extend_from_slice(stats);
        Ok(features)
    }
}
