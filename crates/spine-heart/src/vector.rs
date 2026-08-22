use crate::{HeartError, Result};

pub(crate) fn validate_dimension(vector: &[f32], expected: usize) -> Result<()> {
    if vector.len() != expected {
        return Err(HeartError::InvalidInput(format!(
            "vector dimension {} does not match expected {expected}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(HeartError::InvalidInput(
            "vector contains non-finite values".into(),
        ));
    }
    Ok(())
}

pub(crate) fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

pub(crate) fn unit(vector: &[f32], eps: f32) -> Vec<f32> {
    let magnitude = norm(vector);
    if magnitude < eps {
        return vector.to_vec();
    }
    vector.iter().map(|value| value / magnitude).collect()
}

pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn clamp_unit(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

pub(crate) fn slerp(left: &[f32], right: &[f32], amount: f32, eps: f32) -> Vec<f32> {
    let left = unit(left, eps);
    let right = unit(right, eps);
    let cosine = clamp_unit(dot(&left, &right));
    let theta = cosine.acos();
    if theta < eps {
        let blended: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(left, right)| (1.0 - amount) * left + amount * right)
            .collect();
        return unit(&blended, eps);
    }
    let denominator = theta.sin();
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            (((1.0 - amount) * theta).sin() * left + (amount * theta).sin() * right) / denominator
        })
        .collect()
}

pub(crate) fn log_map(base: &[f32], target: &[f32], eps: f32) -> Vec<f32> {
    let cosine = dot(base, target).clamp(-1.0 + eps, 1.0 - eps);
    let theta = cosine.acos();
    if theta < eps {
        let difference: Vec<f32> = target.iter().zip(base).map(|(t, b)| t - b).collect();
        let radial = dot(&difference, base);
        return difference
            .iter()
            .zip(base)
            .map(|(difference, base)| difference - radial * base)
            .collect();
    }
    let scale = theta / theta.sin();
    target
        .iter()
        .zip(base)
        .map(|(target, base)| scale * (target - cosine * base))
        .collect()
}

pub(crate) fn exp_map(base: &[f32], tangent: &[f32], eps: f32) -> Vec<f32> {
    let magnitude = norm(tangent);
    if magnitude < eps {
        return base.to_vec();
    }
    base.iter()
        .zip(tangent)
        .map(|(base, tangent)| magnitude.cos() * base + magnitude.sin() * tangent / magnitude)
        .collect()
}
