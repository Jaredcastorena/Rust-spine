//! Python's optional tuning and inspection surface, without changing stored layouts.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Dcmdb, DcmdbConfig, DcmdbPolicy, HeartError, MemoryNode, NodeId, Result,
    dcmdb_policy::quantile, vector,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorpusMode {
    Archival,
    Summary,
}

impl DcmdbConfig {
    pub fn hashing(dimension: usize) -> Self {
        let mut config = Self::dense(dimension);
        config.theta_similarity = 0.55;
        config.merge_angle = 50_f32.to_radians();
        config.gamma0 = 0.5;
        config
    }

    /// Both values form the preset; pass its policy to updates, queries and maintenance.
    pub fn book_corpus(dimension: usize) -> (Self, DcmdbPolicy) {
        let mut config = Self::dense(dimension);
        config.theta_similarity = 0.85;
        config.merge_angle = 30_f32.to_radians();
        config.gamma0 = 0.5;
        config.source_divergence_penalty = 5.0;
        config.prune_weight_threshold = 0.0;
        (
            config,
            DcmdbPolicy {
                source_prior_weight: 0.1,
                source_score_weight: 0.1,
                ..DcmdbPolicy::default()
            },
        )
    }

    /// Tune thresholds from normalized samples. Large inputs use evenly spaced
    /// samples instead of Python's random subset, making tuning reproducible.
    pub fn from_sample(
        embeddings: &[Vec<f32>],
        mode: CorpusMode,
        maximum_sample: usize,
    ) -> Result<(Self, DcmdbPolicy)> {
        let dimension = embeddings.first().map_or(0, Vec::len);
        if dimension < 2 || embeddings.len() < 2 || maximum_sample < 2 {
            return Err(HeartError::InvalidInput(
                "DCMDb tuning requires at least two vectors, dimensions and sample slots".into(),
            ));
        }
        for embedding in embeddings {
            vector::validate_dimension(embedding, dimension)?;
        }
        let sample: Vec<_> = sample_indices(embeddings.len(), maximum_sample.min(500))
            .into_iter()
            .map(|i| vector::unit(&embeddings[i], 1e-8))
            .collect();
        if sample.iter().any(|row| vector::norm(row) < 1e-8) {
            return Err(HeartError::InvalidInput(
                "DCMDb tuning requires nonzero vectors".into(),
            ));
        }
        let mut pairs = Vec::new();
        let mut nearest = vec![f32::NEG_INFINITY; sample.len()];
        for i in 0..sample.len() {
            for j in i + 1..sample.len() {
                let similarity = vector::clamp_unit(vector::dot(&sample[i], &sample[j]));
                pairs.push(similarity);
                nearest[i] = nearest[i].max(similarity);
                nearest[j] = nearest[j].max(similarity);
            }
        }
        let (mut config, mut policy) = Self::book_corpus(dimension);
        (config.theta_similarity, config.merge_angle) = match mode {
            CorpusMode::Archival => (
                quantile(&mut nearest, 0.75)
                    .max(quantile(&mut pairs, 0.95))
                    .max(0.5),
                20_f32.to_radians(),
            ),
            CorpusMode::Summary => (
                quantile(&mut nearest, 0.5)
                    .max(quantile(&mut pairs, 0.75))
                    .max(0.3),
                40_f32.to_radians(),
            ),
        };
        config.source_divergence_penalty = 3.0;
        policy.source_score_weight = 0.05;
        Ok((config, policy))
    }
}

impl Dcmdb {
    pub fn children(&self, id: NodeId) -> Vec<&MemoryNode> {
        self.node(id)
            .into_iter()
            .flat_map(|node| &node.children)
            .filter_map(|id| self.absorbed.get(id))
            .collect()
    }

    /// Full diagnostic traversal, cycle-safe. Runtime retrieval uses the separate
    /// host-bounded subtree_event_ids API instead of this unbounded inspector.
    pub fn subtree(&self, id: NodeId) -> Vec<&MemoryNode> {
        let mut pending = vec![id];
        let mut seen = BTreeSet::new();
        let mut result = Vec::new();
        while let Some(id) = pending.pop() {
            if seen.insert(id)
                && let Some(node) = self.node(id)
            {
                pending.extend(node.children.iter().rev().copied());
                result.push(node);
            }
        }
        result
    }

    pub fn hierarchy_depth(&self) -> u32 {
        self.nodes
            .values()
            .map(|n| n.level)
            .max()
            .unwrap_or_default()
    }

    pub fn nodes_at_level(&self, level: u32) -> Vec<&MemoryNode> {
        self.nodes.values().filter(|n| n.level == level).collect()
    }

    pub fn stats(&self) -> BTreeMap<String, f64> {
        let n = self.nodes.len();
        let mut result = BTreeMap::from([("n_nodes".into(), n as f64)]);
        if n != 0 {
            for (name, sum) in [
                (
                    "kappa_mean",
                    self.nodes.values().map(|n| f64::from(n.kappa)).sum::<f64>(),
                ),
                (
                    "conf_mean",
                    self.nodes.values().map(|n| f64::from(n.confidence)).sum(),
                ),
                (
                    "w_mean",
                    self.nodes.values().map(|n| f64::from(n.weight)).sum(),
                ),
            ] {
                result.insert(name.into(), sum / n as f64);
            }
        }
        result
    }

    pub fn merge_angle_diagnostics(&self) -> BTreeMap<String, f64> {
        let mut result = BTreeMap::from([
            ("n_nodes".into(), self.nodes.len() as f64),
            (
                "phi_merge_deg".into(),
                f64::from(self.config.merge_angle.to_degrees()),
            ),
            ("frac_under_gate".into(), 1.0),
        ]);
        if self.nodes.len() < 2 {
            return result;
        }
        let mut angles: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, n)| {
                let best = self
                    .nodes
                    .iter()
                    .filter(|(other, _)| *other != id)
                    .map(|(_, m)| vector::dot(&n.centroid, &m.centroid))
                    .fold(f32::NEG_INFINITY, f32::max);
                vector::clamp_unit(best).acos()
            })
            .collect();
        result.insert(
            "frac_under_gate".into(),
            angles
                .iter()
                .filter(|a| **a <= self.config.merge_angle)
                .count() as f64
                / angles.len() as f64,
        );
        for (name, p) in [
            ("best_deg_min", 0.),
            ("best_deg_p50", 0.5),
            ("best_deg_p90", 0.9),
            ("best_deg_p99", 0.99),
        ] {
            result.insert(
                name.into(),
                f64::from(quantile(&mut angles, p).to_degrees()),
            );
        }
        result
    }

    /// Equalize scoring-channel scales with Python's inverse-magnitude heuristic.
    /// Sampling is deterministic; save the resulting config and policy together.
    pub fn calibrate_weights_equalize(
        &mut self,
        source: Option<&str>,
        policy: &mut DcmdbPolicy,
    ) -> Result<()> {
        policy.validate()?;
        let nodes: Vec<_> = self.nodes.values().collect();
        if nodes.is_empty() {
            return Ok(());
        }
        let count = nodes.len() as f64;
        let indices = sample_indices(nodes.len(), 64);
        let semantic = if nodes.len() == 1 {
            1.0
        } else {
            indices
                .iter()
                .flat_map(|i| {
                    nodes.iter().map(|other| {
                        f64::from(vector::dot(&nodes[*i].centroid, &other.centroid).abs())
                    })
                })
                .sum::<f64>()
                / (indices.len() as f64 * count)
        };
        let graph = nodes
            .iter()
            .map(|node| {
                self.graph
                    .get(&node.id)
                    .filter(|row| !row.is_empty())
                    .map_or(0.0, |row| {
                        row.values().map(|v| f64::from(*v)).sum::<f64>() / row.len() as f64
                    })
            })
            .sum::<f64>()
            / count;
        let freshness = nodes
            .iter()
            .map(|node| {
                (-(self.logical_time() - node.last_seen).max(0.0) / f64::from(node.tau.max(1e-6)))
                    .exp()
            })
            .sum::<f64>()
            / count;
        let confidence = nodes
            .iter()
            .map(|node| f64::from(node.confidence))
            .sum::<f64>()
            / count;
        let prior = nodes
            .iter()
            .map(|node| f64::from(self.source_probability(node.id, source)))
            .sum::<f64>()
            / count;
        let weights = [
            1.0 / semantic.max(1e-6),
            1.0 / graph.max(1e-6),
            1.0 / freshness.max(1e-6),
            1.0 / confidence.max(1e-6),
            if source.is_some() {
                1.0 / prior.max(1e-6)
            } else {
                0.0
            },
        ];
        let sum: f64 = weights.iter().sum();
        self.config.semantic_weight = (weights[0] / sum) as f32;
        self.config.graph_weight = (weights[1] / sum) as f32;
        self.config.freshness_weight = (weights[2] / sum) as f32;
        self.config.confidence_weight = (weights[3] / sum) as f32;
        policy.source_score_weight = (weights[4] / sum) as f32;
        Ok(())
    }
}

fn sample_indices(length: usize, maximum: usize) -> Vec<usize> {
    let size = length.min(maximum);
    if size <= 1 {
        return (0..size).collect();
    }
    (0..size).map(|i| i * (length - 1) / (size - 1)).collect()
}
