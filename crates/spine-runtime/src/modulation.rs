use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

/// Host-owned policy constants mirrored from the Python Spine modulation oracle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModulationConfig {
    pub recall_default_top_k: usize,
    pub plan_default_max_actions: usize,
    pub surprise_moderate: f32,
    pub surprise_high: f32,
    pub urgency_arousal_scale: f32,
    pub urgency_surprise_scale: f32,
    pub tension_moderate: usize,
    pub tension_high: usize,
    pub risk_delta_k: usize,
    pub risk_theta0: f32,
    pub risk_delta_theta: f32,
    pub risk_moderate: f32,
    pub risk_high: f32,
    pub high_risk_tool_rounds: u64,
}

impl Default for ModulationConfig {
    fn default() -> Self {
        Self {
            recall_default_top_k: 5,
            plan_default_max_actions: 3,
            surprise_moderate: 0.5,
            surprise_high: 1.0,
            urgency_arousal_scale: 2.0,
            urgency_surprise_scale: 2.0,
            tension_moderate: 1,
            tension_high: 2,
            risk_delta_k: 5,
            risk_theta0: 0.5,
            risk_delta_theta: 0.3,
            risk_moderate: 0.4,
            risk_high: 0.7,
            high_risk_tool_rounds: 8,
        }
    }
}

/// Signals available to the host after a user turn has been committed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModulationInput {
    pub surprise: f32,
    pub valence: f32,
    pub arousal: f32,
    pub risk: f32,
    pub tensions: usize,
    pub base_temperature: f32,
    /// `None` means unlimited, matching Spine's `reason_max_tool_rounds = 0`.
    pub configured_tool_rounds: Option<NonZeroU64>,
}

/// Effective controls enforced by the host for one user-initiated run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostModulation {
    pub recall_top_k: usize,
    pub provider_temperature: f32,
    pub max_actions: usize,
    pub reflect_eta_multiplier: f32,
    pub urgency: f32,
    pub risk: f32,
    pub coverage_threshold: f32,
    pub expansion_probability: f32,
    /// `None` deliberately preserves the operator's unlimited-tool-round setting.
    pub max_tool_rounds: Option<NonZeroU64>,
}

impl HostModulation {
    /// Python's deterministic absorbed-subtree expansion depth, bounded to 1..3.
    pub fn recall_expansion_depth(&self) -> usize {
        1 + (2.0 * finite_or(self.expansion_probability, 0.3).clamp(0.0, 1.0)) as usize
    }
}

impl ModulationConfig {
    /// Compute the deterministic surprise, tension, and risk policy.
    ///
    /// Surprise thresholds are strict while risk and tension thresholds are
    /// inclusive, matching `spine/modulation.py`. Tension and risk are MAX-gated,
    /// so they can only make an already selected policy more conservative.
    pub fn compute(&self, input: ModulationInput) -> HostModulation {
        let defaults = Self::default();
        let default_top_k = self.recall_default_top_k.max(1);
        let default_actions = self.plan_default_max_actions.max(1);
        let surprise_moderate = finite_or(self.surprise_moderate, defaults.surprise_moderate);
        let surprise_high = finite_or(self.surprise_high, defaults.surprise_high);
        let risk_moderate = finite_or(self.risk_moderate, defaults.risk_moderate);
        let risk_high = finite_or(self.risk_high, defaults.risk_high);
        let surprise = finite_or(input.surprise, 0.0).max(0.0);
        let valence = finite_or(input.valence, 0.0);
        let arousal = finite_or(input.arousal, 0.0).max(0.0);
        let risk = finite_or(input.risk, 0.0).clamp(0.0, 1.0);

        let mut recall_top_k = default_top_k;
        let mut temperature_modifier = 0.0_f32;
        let mut max_actions = default_actions;
        let reflect_eta_multiplier;

        if surprise > surprise_high {
            recall_top_k = default_top_k.saturating_mul(2);
            temperature_modifier = 0.3;
            max_actions = 1;
            reflect_eta_multiplier = 2.0;
        } else if surprise > surprise_moderate {
            recall_top_k = default_top_k.saturating_mul(3) / 2;
            temperature_modifier = 0.15;
            if valence < -0.3 {
                max_actions = (default_actions / 2).max(1);
            }
            reflect_eta_multiplier = 1.5;
        } else {
            reflect_eta_multiplier = 1.0;
        }

        if input.tensions >= self.tension_high {
            recall_top_k = recall_top_k.max(default_top_k.saturating_mul(2));
            temperature_modifier = temperature_modifier.max(0.2);
        } else if input.tensions >= self.tension_moderate {
            temperature_modifier = temperature_modifier.max(0.1);
        }

        let extra_k = ((self.risk_delta_k as f32) * risk).floor() as usize;
        if extra_k > 0 {
            recall_top_k = recall_top_k.max(default_top_k.saturating_add(extra_k));
        }
        if risk >= risk_high {
            temperature_modifier = temperature_modifier.max(0.25);
            max_actions = max_actions.min(1);
        } else if risk >= risk_moderate {
            temperature_modifier = temperature_modifier.max(0.1);
            max_actions = max_actions.min((default_actions / 2).max(1));
        }

        let arousal_scale =
            positive_finite_or(self.urgency_arousal_scale, defaults.urgency_arousal_scale);
        let surprise_scale =
            positive_finite_or(self.urgency_surprise_scale, defaults.urgency_surprise_scale);
        let urgency = (0.5 * (arousal / arousal_scale).min(1.0)
            + 0.5 * (surprise / surprise_scale).min(1.0))
        .clamp(0.0, 1.0);
        let theta0 = finite_or(self.risk_theta0, defaults.risk_theta0);
        let delta_theta = finite_or(self.risk_delta_theta, defaults.risk_delta_theta);
        let coverage_threshold = (theta0 + delta_theta * risk).clamp(0.0, 1.0);
        let expansion_probability = (0.3 + 0.7 * risk).clamp(0.0, 1.0);
        let base_temperature =
            if input.base_temperature.is_finite() && input.base_temperature >= 0.0 {
                input.base_temperature
            } else {
                0.7
            };
        let max_tool_rounds = input.configured_tool_rounds.map(|configured| {
            let mut rounds = configured.get();
            if risk >= risk_high {
                rounds = rounds.max(self.high_risk_tool_rounds.max(1));
            } else if risk >= risk_moderate {
                rounds = rounds.saturating_add(2);
            }
            NonZeroU64::new(rounds).expect("a nonzero tool ceiling remains nonzero")
        });

        HostModulation {
            recall_top_k: recall_top_k.max(1),
            provider_temperature: (base_temperature - temperature_modifier).max(0.1),
            max_actions: max_actions.max(1),
            reflect_eta_multiplier,
            urgency,
            risk,
            coverage_threshold,
            expansion_probability,
            max_tool_rounds,
        }
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn positive_finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(surprise: f32, valence: f32, arousal: f32, risk: f32) -> ModulationInput {
        ModulationInput {
            surprise,
            valence,
            arousal,
            risk,
            tensions: 0,
            base_temperature: 0.7,
            configured_tool_rounds: None,
        }
    }

    #[test]
    fn surprise_oracle_uses_strict_thresholds() {
        let config = ModulationConfig::default();
        let low = config.compute(input(0.5, -1.0, 0.0, 0.0));
        assert_eq!(
            (low.recall_top_k, low.provider_temperature, low.max_actions),
            (5, 0.7, 3)
        );

        let moderate = config.compute(input(0.75, -0.4, 1.0, 0.0));
        assert_eq!(moderate.recall_top_k, 7);
        assert!((moderate.provider_temperature - 0.55).abs() < f32::EPSILON);
        assert_eq!(moderate.max_actions, 1);
        assert_eq!(moderate.reflect_eta_multiplier, 1.5);
        assert!((moderate.urgency - 0.4375).abs() < f32::EPSILON);

        let boundary = config.compute(input(1.0, -1.0, 0.0, 0.0));
        assert_eq!(boundary.recall_top_k, 7);
        let high = config.compute(input(1.01, 1.0, 3.0, 0.0));
        assert_eq!((high.recall_top_k, high.max_actions), (10, 1));
        assert!((high.provider_temperature - 0.4).abs() < f32::EPSILON);
        assert_eq!(high.reflect_eta_multiplier, 2.0);
    }

    #[test]
    fn tension_oracle_is_max_gated() {
        let config = ModulationConfig::default();
        let mut moderate = input(0.0, 0.0, 0.0, 0.0);
        moderate.tensions = 1;
        let moderate = config.compute(moderate);
        assert_eq!(moderate.recall_top_k, 5);
        assert!((moderate.provider_temperature - 0.6).abs() < f32::EPSILON);

        let mut high = input(0.75, 0.0, 0.0, 0.0);
        high.tensions = 2;
        let high = config.compute(high);
        assert_eq!(high.recall_top_k, 10);
        assert!((high.provider_temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn risk_oracle_schedules_depth_coverage_expansion_and_caution() {
        let config = ModulationConfig::default();
        let low = config.compute(input(0.0, 0.0, 0.0, 0.0));
        assert_eq!(low.recall_top_k, 5);
        assert!((low.coverage_threshold - 0.5).abs() < f32::EPSILON);
        assert!((low.expansion_probability - 0.3).abs() < f32::EPSILON);
        assert_eq!(low.recall_expansion_depth(), 1);

        let moderate = config.compute(input(0.0, 0.0, 0.0, 0.5));
        assert_eq!((moderate.recall_top_k, moderate.max_actions), (7, 1));
        assert!((moderate.coverage_threshold - 0.65).abs() < f32::EPSILON);
        assert!((moderate.expansion_probability - 0.65).abs() < f32::EPSILON);
        assert_eq!(moderate.recall_expansion_depth(), 2);
        assert!((moderate.provider_temperature - 0.6).abs() < f32::EPSILON);

        let high = config.compute(input(0.0, 0.0, 0.0, 1.0));
        assert_eq!((high.recall_top_k, high.max_actions), (10, 1));
        assert!((high.coverage_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(high.expansion_probability, 1.0);
        assert_eq!(high.recall_expansion_depth(), 3);
        assert!((high.provider_temperature - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn unlimited_tool_rounds_stay_unlimited_while_positive_budgets_expand() {
        let config = ModulationConfig::default();
        assert_eq!(
            config.compute(input(0.0, 0.0, 0.0, 1.0)).max_tool_rounds,
            None
        );

        let mut moderate = input(0.0, 0.0, 0.0, 0.5);
        moderate.configured_tool_rounds = NonZeroU64::new(3);
        assert_eq!(
            config
                .compute(moderate)
                .max_tool_rounds
                .map(NonZeroU64::get),
            Some(5)
        );
        let mut high = moderate;
        high.risk = 0.8;
        assert_eq!(
            config.compute(high).max_tool_rounds.map(NonZeroU64::get),
            Some(8)
        );
    }

    #[test]
    fn malformed_floats_never_escape_as_nan_or_zero_budgets() {
        let config = ModulationConfig {
            risk_theta0: f32::NAN,
            urgency_arousal_scale: 0.0,
            ..ModulationConfig::default()
        };
        let policy = config.compute(input(f32::NAN, f32::NAN, f32::INFINITY, f32::NAN));
        assert!(policy.provider_temperature.is_finite());
        assert!(policy.urgency.is_finite());
        assert!(policy.coverage_threshold.is_finite());
        assert!(policy.expansion_probability.is_finite());
        assert!(policy.recall_top_k > 0);
        assert!(policy.max_actions > 0);
    }
}
