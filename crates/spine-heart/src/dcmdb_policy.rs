use serde::{Deserialize, Serialize};

use crate::{HeartError, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PmiMode {
    Coupled,
    #[default]
    Decoupled,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PmiVariant {
    #[default]
    Positive,
    Normalized,
    Shifted,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum PmiSignificance {
    #[default]
    None,
    G2,
    ChiSquared,
}

/// Optional Python DCMDb algorithm controls. Pass the same policy to updates,
/// queries and maintenance. It is deliberately separate from the released
/// projection layout: existing encrypted hearts keep their exact wire format.
/// Applications persisting a custom policy should serialize it alongside their
/// operation log, rather than changing it implicitly during replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DcmdbPolicy {
    pub maximum_secondary_updates: Option<usize>,
    pub source_prior_weight: f32,
    pub source_score_weight: f32,
    pub pmi_mode: PmiMode,
    pub pmi_variant: PmiVariant,
    pub pmi_shift: f32,
    pub pmi_significance: PmiSignificance,
    pub significance_power: f32,
    pub adaptive_merge_angle: bool,
    pub merge_angle_margin: f32,
    pub use_decayed_effective_samples: bool,
}

impl Default for DcmdbPolicy {
    fn default() -> Self {
        Self {
            maximum_secondary_updates: None,
            source_prior_weight: 0.0,
            source_score_weight: 0.0,
            pmi_mode: PmiMode::Decoupled,
            pmi_variant: PmiVariant::Positive,
            pmi_shift: 0.0,
            pmi_significance: PmiSignificance::None,
            significance_power: 0.5,
            adaptive_merge_angle: false,
            merge_angle_margin: 0.0,
            use_decayed_effective_samples: false,
        }
    }
}

impl DcmdbPolicy {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("source_prior_weight", self.source_prior_weight),
            ("source_score_weight", self.source_score_weight),
            ("pmi_shift", self.pmi_shift),
            ("significance_power", self.significance_power),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(HeartError::InvalidInput(format!(
                    "DCMDb {name} must be finite and nonnegative"
                )));
            }
        }
        if !self.merge_angle_margin.is_finite()
            || self.merge_angle_margin.abs() > std::f32::consts::PI
        {
            return Err(HeartError::InvalidInput(
                "DCMDb merge angle margin must be finite and in [-pi, pi]".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn quantile(values: &mut [f32], probability: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let position = probability * (values.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    values[low] + (values[high] - values[low]) * position.fract()
}
