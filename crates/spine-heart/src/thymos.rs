use serde::{Deserialize, Serialize};

use crate::{HeartError, Result, vector};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ValenceMode {
    CosineError,
    SphericalResidual,
    DirectDifference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActivationNonlinearity {
    None,
    Tanh,
    Softmax,
    Relu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThymosConfig {
    pub dimension: usize,
    pub channels: usize,
    pub learning_rate: f32,
    pub decay: f32,
    pub normalize_rows: bool,
    pub valence_temperature: f32,
    pub valence_mode: ValenceMode,
    pub nonlinearity: ActivationNonlinearity,
    pub channel_labels: Option<Vec<String>>,
    pub trajectory_alpha: f32,
    pub eps: f32,
}

impl ThymosConfig {
    pub fn new(dimension: usize, channels: usize) -> Result<Self> {
        let config = Self {
            dimension,
            channels,
            learning_rate: 0.01,
            decay: 0.001,
            normalize_rows: false,
            valence_temperature: 1.0,
            valence_mode: ValenceMode::CosineError,
            nonlinearity: ActivationNonlinearity::None,
            channel_labels: None,
            trajectory_alpha: 0.3,
            eps: 1e-8,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<usize> {
        if self.dimension < 2 || self.channels == 0 {
            return Err(HeartError::InvalidInput(
                "Thymos requires dimension >= 2 and at least one channel".into(),
            ));
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate < 0.0
            || !self.decay.is_finite()
            || !(0.0..=1.0).contains(&self.decay)
            || !self.valence_temperature.is_finite()
            || self.valence_temperature <= 0.0
            || !self.trajectory_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.trajectory_alpha)
            || !self.eps.is_finite()
            || self.eps <= 0.0
        {
            return Err(HeartError::InvalidInput(
                "Thymos learning, decay, temperature, trajectory, or epsilon is invalid".into(),
            ));
        }
        if self
            .channel_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != self.channels)
        {
            return Err(HeartError::InvalidInput(
                "channel label count does not match Thymos channels".into(),
            ));
        }
        let elements = self
            .dimension
            .checked_mul(self.channels)
            .filter(|elements| *elements <= 16_777_216)
            .ok_or_else(|| {
                HeartError::InvalidInput(
                    "Thymos tensor dimensions overflow or exceed 16,777,216 elements".into(),
                )
            })?;
        Ok(elements)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeelingVector {
    pub raw: Vec<f32>,
    pub activated: Vec<f32>,
    pub valence: f32,
    pub arousal: f32,
    pub dominant_channel: usize,
    pub dominant_label: Option<String>,
    pub input_norm: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub surprise: f32,
    pub speed: f32,
    pub heading_norm: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thymos {
    pub config: ThymosConfig,
    tensor: Vec<f32>,
    channel_mass: Vec<f32>,
    logical_time: f64,
    update_count: u64,
    previous_position: Option<Vec<f32>>,
    heading: Option<Vec<f32>>,
    heading_norm: f32,
}

impl Thymos {
    pub fn with_tensor(
        config: ThymosConfig,
        tensor: Vec<f32>,
        channel_mass: Option<Vec<f32>>,
    ) -> Result<Self> {
        let expected = config.validate()?;
        vector::validate_dimension(&tensor, expected)?;
        let channel_mass = match channel_mass {
            Some(channel_mass) => {
                vector::validate_dimension(&channel_mass, config.channels)?;
                channel_mass
            }
            None => (0..config.channels)
                .map(|channel| {
                    let row = &tensor[channel * config.dimension..(channel + 1) * config.dimension];
                    vector::norm(row)
                })
                .collect(),
        };
        Ok(Self {
            config,
            tensor,
            channel_mass,
            logical_time: 0.0,
            update_count: 0,
            previous_position: None,
            heading: None,
            heading_norm: 0.0,
        })
    }

    pub fn with_seed(config: ThymosConfig, seed: [u8; 32]) -> Result<Self> {
        let elements = config.validate()?;
        let mut reader = blake3::Hasher::new_keyed(&seed).finalize_xof();
        let byte_count = elements.checked_mul(4).ok_or_else(|| {
            HeartError::InvalidInput("Thymos tensor byte length overflowed".into())
        })?;
        let mut bytes = vec![0_u8; byte_count];
        reader.fill(&mut bytes);
        let scale = (config.dimension as f32).sqrt();
        let mut tensor = Vec::with_capacity(config.dimension * config.channels);
        for chunk in bytes.as_chunks::<4>().0 {
            let raw = u32::from_le_bytes(*chunk);
            tensor.push((raw as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32);
        }
        for row in tensor.chunks_mut(config.dimension) {
            let normalized = vector::unit(row, config.eps);
            for (value, normalized) in row.iter_mut().zip(normalized) {
                *value = normalized / scale;
            }
        }
        Self::with_tensor(config, tensor, None)
    }

    pub fn new(config: ThymosConfig) -> Result<Self> {
        let mut seed = [0_u8; 32];
        getrandom::fill(&mut seed).map_err(|_| HeartError::Crypto)?;
        Self::with_seed(config, seed)
    }

    pub fn tensor(&self) -> &[f32] {
        &self.tensor
    }

    pub fn channel_mass(&self) -> &[f32] {
        &self.channel_mass
    }

    pub fn logical_time(&self) -> f64 {
        self.logical_time
    }

    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    pub fn query(&self, input: &[f32]) -> Result<FeelingVector> {
        vector::validate_dimension(input, self.config.dimension)?;
        let input_norm = vector::norm(input);
        let input = vector::unit(input, self.config.eps);
        let raw = self.multiply(&input);
        let activated = self.activate(&raw);
        let valence = activated.iter().sum::<f32>() / activated.len() as f32;
        let arousal = vector::norm(&activated);
        let dominant_channel = activated
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or_default();
        let dominant_label = self
            .config
            .channel_labels
            .as_ref()
            .and_then(|labels| labels.get(dominant_channel))
            .cloned();
        Ok(FeelingVector {
            raw,
            activated,
            valence,
            arousal,
            dominant_channel,
            dominant_label,
            input_norm,
        })
    }

    pub fn compute_valence(&self, expected: &[f32], actual: &[f32]) -> Result<Vec<f32>> {
        vector::validate_dimension(expected, self.config.dimension)?;
        vector::validate_dimension(actual, self.config.dimension)?;
        let expected = vector::unit(expected, self.config.eps);
        let actual = vector::unit(actual, self.config.eps);
        let residual = match self.config.valence_mode {
            ValenceMode::CosineError => {
                let similarity = vector::clamp_unit(vector::dot(&expected, &actual));
                actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| actual - similarity * expected)
                    .collect()
            }
            ValenceMode::SphericalResidual => {
                let projection = vector::dot(&expected, &actual);
                let residual: Vec<f32> = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| actual - projection * expected)
                    .collect();
                vector::unit(&residual, self.config.eps)
            }
            ValenceMode::DirectDifference => actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| actual - expected)
                .collect(),
        };
        Ok(self.multiply(&residual))
    }

    pub fn update(&mut self, context: &[f32], eligibility: &[f32]) -> Result<()> {
        vector::validate_dimension(context, self.config.dimension)?;
        vector::validate_dimension(eligibility, self.config.channels)?;
        let context = vector::unit(context, self.config.eps);
        let activation = self.multiply(&context);
        let rho = 1.0 - self.config.decay;
        for channel in 0..self.config.channels {
            let signal = self.config.learning_rate * eligibility[channel] * activation[channel];
            let row_start = channel * self.config.dimension;
            let row_end = row_start + self.config.dimension;
            let row = &mut self.tensor[row_start..row_end];
            for (weight, context) in row.iter_mut().zip(&context) {
                *weight = *weight * rho + signal * context;
            }
            self.channel_mass[channel] = self.channel_mass[channel] * rho + signal.abs();
            if self.config.normalize_rows {
                let normalized = vector::unit(row, self.config.eps);
                row.copy_from_slice(&normalized);
            }
        }
        self.logical_time += 1.0;
        self.update_count = self.update_count.saturating_add(1);
        Ok(())
    }

    pub fn update_from_experience(
        &mut self,
        context: &[f32],
        expected: &[f32],
        actual: &[f32],
    ) -> Result<FeelingVector> {
        let eligibility = self.compute_valence(expected, actual)?;
        self.update(context, &eligibility)?;
        self.query(actual)
    }

    /// Learn from the trajectory's prior prediction and a newly observed incoming context.
    /// Returns `None` until both a previous position and heading exist.
    pub fn learn_predicted_next(&mut self, actual: &[f32]) -> Result<Option<FeelingVector>> {
        vector::validate_dimension(actual, self.config.dimension)?;
        let Some(context) = self.previous_position.clone() else {
            return Ok(None);
        };
        let Some(expected) = self.predict_next() else {
            return Ok(None);
        };
        self.update_from_experience(&context, &expected, actual)
            .map(Some)
    }

    pub fn step(&mut self, input: &[f32]) -> Result<TrajectoryStep> {
        vector::validate_dimension(input, self.config.dimension)?;
        let input = vector::unit(input, self.config.eps);
        let Some(previous) = self.previous_position.clone() else {
            self.previous_position = Some(input);
            return Ok(TrajectoryStep::default());
        };
        let predicted = self.predict_next();
        let delta = vector::log_map(&previous, &input, self.config.eps);
        let speed = vector::norm(&delta);
        let surprise = predicted
            .map(|predicted| vector::clamp_unit(vector::dot(&predicted, &input)).acos())
            .unwrap_or(speed);
        self.heading = Some(match self.heading.take() {
            None => delta,
            Some(old_heading) => {
                let old_norm = vector::norm(&old_heading);
                if old_norm > self.config.eps && speed > self.config.eps {
                    let old_direction: Vec<f32> =
                        old_heading.iter().map(|value| value / old_norm).collect();
                    let new_direction: Vec<f32> = delta.iter().map(|value| value / speed).collect();
                    let direction = vector::slerp(
                        &old_direction,
                        &new_direction,
                        self.config.trajectory_alpha,
                        self.config.eps,
                    );
                    let magnitude = (1.0 - self.config.trajectory_alpha) * old_norm
                        + self.config.trajectory_alpha * speed;
                    direction.iter().map(|value| value * magnitude).collect()
                } else {
                    old_heading
                        .iter()
                        .zip(delta)
                        .map(|(old, delta)| {
                            (1.0 - self.config.trajectory_alpha) * old
                                + self.config.trajectory_alpha * delta
                        })
                        .collect()
                }
            }
        });
        if let Some(heading) = &mut self.heading {
            let original_norm = vector::norm(heading);
            let radial = vector::dot(heading, &input);
            let projected: Vec<f32> = heading
                .iter()
                .zip(&input)
                .map(|(heading, input)| heading - radial * input)
                .collect();
            let projected_norm = vector::norm(&projected);
            if projected_norm > self.config.eps && original_norm > self.config.eps {
                for (heading, projected) in heading.iter_mut().zip(projected) {
                    *heading = projected * original_norm / projected_norm;
                }
            } else {
                heading.fill(0.0);
            }
            self.heading_norm = vector::norm(heading);
        }
        self.previous_position = Some(input);
        self.logical_time += 1.0;
        Ok(TrajectoryStep {
            surprise,
            speed,
            heading_norm: self.heading_norm,
        })
    }

    pub fn predict_next(&self) -> Option<Vec<f32>> {
        let position = self.previous_position.as_ref()?;
        let heading = self.heading.as_ref()?;
        if self.heading_norm < self.config.eps {
            return Some(position.clone());
        }
        Some(vector::exp_map(position, heading, self.config.eps))
    }

    fn multiply(&self, input: &[f32]) -> Vec<f32> {
        self.tensor
            .chunks_exact(self.config.dimension)
            .map(|row| vector::dot(row, input))
            .collect()
    }

    fn activate(&self, raw: &[f32]) -> Vec<f32> {
        match self.config.nonlinearity {
            ActivationNonlinearity::None => raw.to_vec(),
            ActivationNonlinearity::Tanh => raw.iter().map(|value| value.tanh()).collect(),
            ActivationNonlinearity::Relu => raw.iter().map(|value| value.max(0.0)).collect(),
            ActivationNonlinearity::Softmax => {
                let temperature = self.config.valence_temperature.max(1e-8);
                let maximum = raw
                    .iter()
                    .map(|value| value / temperature)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exponentials: Vec<f32> = raw
                    .iter()
                    .map(|value| ((value / temperature) - maximum).exp())
                    .collect();
                let total: f32 = exponentials.iter().sum();
                exponentials.iter().map(|value| value / total).collect()
            }
        }
    }
}
