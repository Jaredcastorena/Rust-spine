use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{EventId, HeartError, NodeId, Result, vector};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DcmdbConfig {
    pub dimension: usize,
    pub theta_similarity: f32,
    pub neighbors: usize,
    pub gamma0: f32,
    pub lambda_age: f32,
    pub lambda_uncertainty: f32,
    pub delta0: f32,
    pub max_update_angle: f32,
    pub zeta: f32,
    pub tension_promote_count: u32,
    pub merge_angle: f32,
    pub merge_bonus: f32,
    pub merge_penalty: f32,
    pub max_merges_per_pass: Option<usize>,
    pub kappa_pseudocount: f32,
    pub maximum_kappa_drop: f32,
    pub kappa_drop_penalty: f32,
    pub source_divergence_penalty: f32,
    pub concentration_scale: f32,
    pub concentration_bias: f32,
    pub confidence_mass_minimum: f32,
    pub tau0: f32,
    pub tau_min: f32,
    pub tau_max: f32,
    pub dream_stickiness: f32,
    pub dream_walks: usize,
    pub dream_walk_length: usize,
    pub dream_temperature: f32,
    pub dream_exploration_bonus: f32,
    pub tau_edge: f32,
    pub pmi_epsilon: f32,
    pub pmi_learning_rate: f32,
    pub pmi_neighbors: usize,
    pub pmi_angle_max: f32,
    pub top_edges: usize,
    pub pagerank_restart: f32,
    pub semantic_weight: f32,
    pub graph_weight: f32,
    pub freshness_weight: f32,
    pub confidence_weight: f32,
    pub tension_penalty: f32,
    pub mmr_weight: f32,
    pub split_probability: f32,
    pub clear_probability: f32,
    pub split_penalty: f32,
    pub minimum_tension_evidence: u32,
    pub effective_sample_minimum: f32,
    pub kappa_regularization: f32,
    pub eps: f32,
    pub initial_weight: f32,
    pub prune_weight_threshold: f32,
    pub minimum_prune_age: f32,
}

impl DcmdbConfig {
    pub fn dense(dimension: usize) -> Self {
        Self {
            dimension,
            theta_similarity: 0.80,
            neighbors: 8,
            gamma0: 1.0,
            lambda_age: 1.0,
            lambda_uncertainty: 1.0,
            delta0: 1.0,
            max_update_angle: 10_f32.to_radians(),
            zeta: 0.05,
            tension_promote_count: 1,
            merge_angle: 60_f32.to_radians(),
            merge_bonus: 0.1,
            merge_penalty: 1.5,
            max_merges_per_pass: None,
            kappa_pseudocount: 1.0,
            maximum_kappa_drop: 0.0,
            kappa_drop_penalty: 0.0,
            source_divergence_penalty: 0.0,
            concentration_scale: 0.1,
            concentration_bias: 1.0,
            confidence_mass_minimum: 8.0,
            tau0: 100.0,
            tau_min: 10.0,
            tau_max: 10_000.0,
            dream_stickiness: 0.05,
            dream_walks: 10,
            dream_walk_length: 5,
            dream_temperature: 1.0,
            dream_exploration_bonus: 0.1,
            tau_edge: 1_000.0,
            pmi_epsilon: 1.0,
            pmi_learning_rate: 0.1,
            pmi_neighbors: 3,
            pmi_angle_max: 80_f32.to_radians(),
            top_edges: 16,
            pagerank_restart: 0.15,
            semantic_weight: 0.90,
            graph_weight: 0.0,
            freshness_weight: 0.05,
            confidence_weight: 0.05,
            tension_penalty: 0.1,
            mmr_weight: 0.0,
            split_probability: 0.9,
            clear_probability: 0.1,
            split_penalty: 1.5,
            minimum_tension_evidence: 3,
            effective_sample_minimum: 8.0,
            kappa_regularization: 1.0,
            eps: 1e-8,
            initial_weight: 1.0,
            prune_weight_threshold: 0.01,
            minimum_prune_age: 0.0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.dimension < 2 {
            return Err(HeartError::InvalidInput(
                "DCMDb dimension must be at least two".into(),
            ));
        }
        for (name, value) in [
            ("theta_similarity", self.theta_similarity),
            ("gamma0", self.gamma0),
            ("lambda_age", self.lambda_age),
            ("lambda_uncertainty", self.lambda_uncertainty),
            ("delta0", self.delta0),
            ("max_update_angle", self.max_update_angle),
            ("zeta", self.zeta),
            ("merge_angle", self.merge_angle),
            ("merge_bonus", self.merge_bonus),
            ("merge_penalty", self.merge_penalty),
            ("kappa_pseudocount", self.kappa_pseudocount),
            ("maximum_kappa_drop", self.maximum_kappa_drop),
            ("kappa_drop_penalty", self.kappa_drop_penalty),
            ("source_divergence_penalty", self.source_divergence_penalty),
            ("concentration_scale", self.concentration_scale),
            ("concentration_bias", self.concentration_bias),
            ("confidence_mass_minimum", self.confidence_mass_minimum),
            ("tau0", self.tau0),
            ("tau_min", self.tau_min),
            ("tau_max", self.tau_max),
            ("dream_stickiness", self.dream_stickiness),
            ("dream_temperature", self.dream_temperature),
            ("dream_exploration_bonus", self.dream_exploration_bonus),
            ("tau_edge", self.tau_edge),
            ("pmi_epsilon", self.pmi_epsilon),
            ("pmi_learning_rate", self.pmi_learning_rate),
            ("pmi_angle_max", self.pmi_angle_max),
            ("pagerank_restart", self.pagerank_restart),
            ("semantic_weight", self.semantic_weight),
            ("graph_weight", self.graph_weight),
            ("freshness_weight", self.freshness_weight),
            ("confidence_weight", self.confidence_weight),
            ("tension_penalty", self.tension_penalty),
            ("mmr_weight", self.mmr_weight),
            ("split_probability", self.split_probability),
            ("clear_probability", self.clear_probability),
            ("split_penalty", self.split_penalty),
            ("effective_sample_minimum", self.effective_sample_minimum),
            ("kappa_regularization", self.kappa_regularization),
            ("eps", self.eps),
            ("initial_weight", self.initial_weight),
            ("prune_weight_threshold", self.prune_weight_threshold),
            ("minimum_prune_age", self.minimum_prune_age),
        ] {
            if !value.is_finite() {
                return Err(HeartError::InvalidInput(format!(
                    "DCMDb {name} must be finite"
                )));
            }
        }
        let similarity = |value: f32| (0.0..=1.0).contains(&value);
        let probability = |value: f32| (0.0..=1.0).contains(&value);
        let angle = |value: f32| (0.0..=std::f32::consts::PI).contains(&value);
        if !similarity(self.theta_similarity)
            || !angle(self.max_update_angle)
            || !(0.0..=std::f32::consts::FRAC_PI_2).contains(&self.merge_angle)
            || !angle(self.pmi_angle_max)
            || !probability(self.pmi_learning_rate)
            || !probability(self.pagerank_restart)
            || !probability(self.mmr_weight)
            || !probability(self.split_probability)
            || !probability(self.clear_probability)
        {
            return Err(HeartError::InvalidInput(
                "DCMDb similarity, angle, or probability is out of range".into(),
            ));
        }
        if self.neighbors == 0
            || self.pmi_neighbors == 0
            || self.top_edges == 0
            || self.tension_promote_count == 0
            || self.minimum_tension_evidence == 0
        {
            return Err(HeartError::InvalidInput(
                "DCMDb count limits must be positive".into(),
            ));
        }
        if self.gamma0 < 0.0
            || self.lambda_age < 0.0
            || self.lambda_uncertainty < 0.0
            || self.delta0 < 0.0
            || self.zeta < 0.0
            || self.merge_bonus < 0.0
            || self.merge_penalty < 0.0
            || self.kappa_pseudocount < 0.0
            || self.maximum_kappa_drop < 0.0
            || self.kappa_drop_penalty < 0.0
            || self.source_divergence_penalty < 0.0
            || self.concentration_scale < 0.0
            || self.dream_stickiness < 0.0
            || self.dream_exploration_bonus < 0.0
            || self.semantic_weight < 0.0
            || self.graph_weight < 0.0
            || self.freshness_weight < 0.0
            || self.confidence_weight < 0.0
            || self.tension_penalty < 0.0
            || self.split_penalty < 0.0
            || self.kappa_regularization < 0.0
            || self.initial_weight < 0.0
            || self.prune_weight_threshold < 0.0
            || self.minimum_prune_age < 0.0
        {
            return Err(HeartError::InvalidInput(
                "DCMDb nonnegative parameter was negative".into(),
            ));
        }
        if self.confidence_mass_minimum <= 0.0
            || self.tau_min <= 0.0
            || self.tau0 < self.tau_min
            || self.tau0 > self.tau_max
            || self.tau_max < self.tau_min
            || self.dream_temperature <= 0.0
            || self.tau_edge <= 0.0
            || self.pmi_epsilon <= 0.0
            || self.effective_sample_minimum <= 0.0
            || self.eps <= 0.0
        {
            return Err(HeartError::InvalidInput(
                "DCMDb positive scales or tau ordering are invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensionInfo {
    pub center: Vec<f32>,
    pub vector: Vec<f32>,
    pub uncertainty_radius: f32,
    pub trajectory: Vec<Vec<f32>>,
    pub sufficient_sum: Vec<f32>,
    pub effective_count: f32,
    pub log_bayes_factor: f32,
    pub evidence_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: NodeId,
    pub event_ids: Vec<EventId>,
    pub centroid: Vec<f32>,
    pub sufficient_sum: Vec<f32>,
    pub effective_count: f32,
    pub weight: f32,
    pub kappa: f32,
    pub confidence: f32,
    pub tau: f32,
    pub last_decay: f64,
    pub last_seen: f64,
    pub visits: f32,
    pub atypical_count: u32,
    pub tension: Option<TensionInfo>,
    pub source_counts: BTreeMap<String, f32>,
    pub level: u32,
    pub children: Vec<NodeId>,
    pub metadata: BTreeMap<String, String>,
}

impl MemoryNode {
    fn apply_decay(&mut self, now: f64, eps: f32) {
        let elapsed = (now - self.last_decay).max(0.0) as f32;
        if elapsed <= 0.0 {
            return;
        }
        let rho = (-elapsed / self.tau.max(eps)).exp();
        for value in &mut self.sufficient_sum {
            *value *= rho;
        }
        self.effective_count *= rho;
        self.weight *= rho;
        for count in self.source_counts.values_mut() {
            *count *= rho;
        }
        self.last_decay = now;
    }
}

#[derive(Clone, Debug)]
pub struct MemoryObservation {
    pub event_id: EventId,
    pub vector: Vec<f32>,
    pub time: f64,
    pub source: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallHit {
    pub node_id: NodeId,
    pub score: f32,
    pub semantic_score: f32,
    pub graph_score: f32,
    pub freshness: f32,
    pub confidence: f32,
    pub tensioned: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaintenanceReport {
    pub merges: usize,
    pub pruned: usize,
    pub walks_completed: usize,
    pub nodes_reactivated: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dcmdb {
    pub config: DcmdbConfig,
    pub nodes: BTreeMap<NodeId, MemoryNode>,
    pub absorbed: BTreeMap<NodeId, MemoryNode>,
    pub graph: BTreeMap<NodeId, BTreeMap<NodeId, f32>>,
    visit_counts: BTreeMap<NodeId, f32>,
    coactivations: BTreeMap<(NodeId, NodeId), f32>,
    logical_time: f64,
    last_edge_decay: f64,
    spawn_counter: u64,
    dream_counter: u64,
}

impl Dcmdb {
    pub fn new(config: DcmdbConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            nodes: BTreeMap::new(),
            absorbed: BTreeMap::new(),
            graph: BTreeMap::new(),
            visit_counts: BTreeMap::new(),
            coactivations: BTreeMap::new(),
            logical_time: 0.0,
            last_edge_decay: 0.0,
            spawn_counter: 0,
            dream_counter: 0,
        })
    }

    pub fn logical_time(&self) -> f64 {
        self.logical_time
    }

    pub fn update(&mut self, observation: MemoryObservation) -> Result<NodeId> {
        vector::validate_dimension(&observation.vector, self.config.dimension)?;
        let input = vector::unit(&observation.vector, self.config.eps);
        if vector::norm(&input) < self.config.eps {
            return Err(HeartError::InvalidInput(
                "DCMDb observation must have nonzero magnitude".into(),
            ));
        }
        let now = observation.time.max(self.logical_time);
        self.logical_time = now;
        self.decay_edges(now);
        for node in self.nodes.values_mut() {
            node.apply_decay(now, self.config.eps);
        }

        let knn = self.nearest(&input, self.config.neighbors);
        let mut candidates: Vec<(NodeId, f32)> = knn
            .iter()
            .copied()
            .filter(|(_, similarity)| *similarity >= self.config.theta_similarity)
            .collect();
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let pmi_ids: Vec<NodeId> = knn
            .iter()
            .filter(|(_, similarity)| *similarity >= self.config.pmi_angle_max.cos())
            .take(self.config.pmi_neighbors)
            .map(|(id, _)| *id)
            .collect();

        if candidates.is_empty() {
            self.record_coactivation(&pmi_ids);
            let id = self.spawn(
                observation.event_id,
                &input,
                now,
                observation.source.as_deref(),
                observation.metadata,
            );
            *self.visit_counts.entry(id).or_default() += 1.0;
            self.refresh_edges();
            return Ok(id);
        }

        let (mut winner, winner_similarity) = candidates[0];
        self.update_node(
            winner,
            observation.event_id,
            &input,
            winner_similarity,
            now,
            observation.source.as_deref(),
        );
        if let Some(node) = self.nodes.get_mut(&winner) {
            node.metadata.extend(observation.metadata);
        }
        if let Some(promoted) = self.promote_if_tension(
            winner,
            observation.event_id,
            &input,
            now,
            observation.source.as_deref(),
        ) {
            winner = promoted;
        }
        for (id, similarity) in candidates.into_iter().skip(1) {
            self.update_node(
                id,
                observation.event_id,
                &input,
                similarity,
                now,
                observation.source.as_deref(),
            );
        }
        self.record_coactivation(&pmi_ids);
        self.refresh_edges();
        Ok(winner)
    }

    pub fn query(&self, query: &[f32], now: f64, top_k: usize) -> Result<Vec<RecallHit>> {
        vector::validate_dimension(query, self.config.dimension)?;
        let query = vector::unit(query, self.config.eps);
        if self.nodes.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let mut semantic: Vec<(NodeId, f32)> = self
            .nodes
            .iter()
            .map(|(id, node)| (*id, vector::dot(&node.centroid, &query)))
            .collect();
        semantic.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let seeds: Vec<(NodeId, f32)> = semantic
            .iter()
            .take(self.config.neighbors)
            .map(|(id, score)| (*id, score.max(0.0)))
            .collect();
        let pagerank = self.personalized_pagerank(&seeds);
        let mut hits = Vec::with_capacity(self.nodes.len());
        for (id, semantic_score) in semantic {
            let node = &self.nodes[&id];
            let graph_score = pagerank.get(&id).copied().unwrap_or_default();
            let freshness =
                (-(now - node.last_seen).max(0.0) as f32 / node.tau.max(self.config.eps)).exp();
            let tensioned = node.tension.is_some();
            let score = self.config.semantic_weight * semantic_score
                + self.config.graph_weight * graph_score
                + self.config.freshness_weight * freshness
                + self.config.confidence_weight * node.confidence
                - if tensioned {
                    self.config.tension_penalty
                } else {
                    0.0
                };
            hits.push(RecallHit {
                node_id: id,
                score,
                semantic_score,
                graph_score,
                freshness,
                confidence: node.confidence,
                tensioned,
            });
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        if self.config.mmr_weight > 0.0 && top_k > 1 && hits.len() > 1 {
            let weight = self.config.mmr_weight;
            let pool_length = (top_k * 3).min(hits.len());
            let mut remaining = hits[..pool_length].to_vec();
            let mut selected = vec![remaining.remove(0)];
            while selected.len() < top_k && !remaining.is_empty() {
                let (best_index, _) = remaining
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| {
                        let maximum_similarity = selected
                            .iter()
                            .map(|chosen| {
                                vector::dot(
                                    &self.nodes[&candidate.node_id].centroid,
                                    &self.nodes[&chosen.node_id].centroid,
                                )
                            })
                            .fold(f32::NEG_INFINITY, f32::max);
                        let mmr = (1.0 - weight) * candidate.score - weight * maximum_similarity;
                        (index, mmr)
                    })
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| right.0.cmp(&left.0))
                    })
                    .expect("remaining MMR candidates are nonempty");
                selected.push(remaining.remove(best_index));
            }
            return Ok(selected);
        }
        hits.truncate(top_k);
        Ok(hits)
    }

    pub fn node(&self, id: NodeId) -> Option<&MemoryNode> {
        self.nodes.get(&id).or_else(|| self.absorbed.get(&id))
    }

    pub fn relationship_score(&self, left: NodeId, right: NodeId) -> Option<f32> {
        let left_node = self.node(left)?;
        let right_node = self.node(right)?;
        let semantic = vector::dot(&left_node.centroid, &right_node.centroid);
        let edge = self
            .graph
            .get(&left)
            .and_then(|neighbors| neighbors.get(&right))
            .copied()
            .or_else(|| {
                self.graph
                    .get(&right)
                    .and_then(|neighbors| neighbors.get(&left))
                    .copied()
            })
            .unwrap_or_default();
        Some(0.9 * semantic + 0.1 * edge.tanh())
    }

    pub fn coherent_apex(
        &self,
        left: NodeId,
        right: NodeId,
        minimum_similarity: f32,
    ) -> Option<(NodeId, f32)> {
        let left_node = self.node(left)?;
        let right_node = self.node(right)?;
        self.nodes
            .iter()
            .filter_map(|(candidate_id, candidate)| {
                let left_similarity = vector::dot(&candidate.centroid, &left_node.centroid);
                let right_similarity = vector::dot(&candidate.centroid, &right_node.centroid);
                let coherence = left_similarity.min(right_similarity);
                (coherence >= minimum_similarity).then_some((*candidate_id, coherence))
            })
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
    }

    pub fn consolidate_pass(&mut self) -> usize {
        if self.nodes.len() < 2 {
            return 0;
        }
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let nearest: Vec<(NodeId, NodeId, f32)> = ids
            .iter()
            .filter_map(|left| {
                let left_node = &self.nodes[left];
                ids.iter()
                    .filter(|right| *right != left)
                    .map(|right| {
                        (
                            *right,
                            vector::dot(&left_node.centroid, &self.nodes[right].centroid),
                        )
                    })
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| right.0.cmp(&left.0))
                    })
                    .map(|(right, similarity)| (*left, right, similarity))
            })
            .collect();
        let maximum = self.config.max_merges_per_pass.unwrap_or(self.nodes.len());
        let mut merges = 0;
        let mut cool = BTreeSet::new();
        for (left, right, similarity) in nearest {
            if merges >= maximum
                || cool.contains(&left)
                || cool.contains(&right)
                || !self.nodes.contains_key(&left)
                || !self.nodes.contains_key(&right)
                || vector::clamp_unit(similarity).acos() > self.config.merge_angle
            {
                continue;
            }
            let left_node = self.nodes[&left].clone();
            let right_node = self.nodes[&right].clone();
            let sum: Vec<f32> = left_node
                .sufficient_sum
                .iter()
                .zip(&right_node.sufficient_sum)
                .map(|(left, right)| left + right)
                .collect();
            let left_resultant = vector::norm(&left_node.sufficient_sum);
            let right_resultant = vector::norm(&right_node.sufficient_sum);
            let merged_resultant = vector::norm(&sum);
            let left_count = left_node.effective_count.max(self.config.eps);
            let right_count = right_node.effective_count.max(self.config.eps);
            let merged_count = left_count + right_count;
            let pseudo = self.config.kappa_pseudocount.max(0.0);
            let left_kappa = kappa_from_m(
                left_resultant / (left_count + pseudo),
                self.config.dimension,
                self.config.eps,
            );
            let right_kappa = kappa_from_m(
                right_resultant / (right_count + pseudo),
                self.config.dimension,
                self.config.eps,
            );
            let merged_kappa = kappa_from_m(
                merged_resultant / (merged_count + pseudo),
                self.config.dimension,
                self.config.eps,
            );
            let left_ll = left_count * log_c_vmf(left_kappa, self.config.dimension)
                + left_kappa * left_resultant;
            let right_ll = right_count * log_c_vmf(right_kappa, self.config.dimension)
                + right_kappa * right_resultant;
            let merged_ll = merged_count * log_c_vmf(merged_kappa, self.config.dimension)
                + merged_kappa * merged_resultant;
            let delta_ll = merged_ll - left_ll - right_ll;
            let effective_samples =
                (left_node.visits + right_node.visits).max(self.config.effective_sample_minimum);
            let bic = effective_samples.max(self.config.eps).ln();
            let kappa_drop = (left_kappa.min(right_kappa) - merged_kappa).max(0.0);
            if self.config.maximum_kappa_drop > 0.0 && kappa_drop > self.config.maximum_kappa_drop {
                continue;
            }
            let source_divergence = source_total_variation(&left_node, &right_node);
            let score = delta_ll + self.config.merge_penalty * bic
                - self.config.kappa_drop_penalty * kappa_drop
                - self.config.source_divergence_penalty * source_divergence;
            let no_penalties = self.config.merge_penalty <= 0.0
                && self.config.maximum_kappa_drop <= 0.0
                && self.config.kappa_drop_penalty <= 0.0
                && self.config.source_divergence_penalty <= 0.0;
            if score <= 0.0 && !no_penalties {
                continue;
            }
            self.merge_nodes(left, right, similarity, sum, merged_count, merged_kappa);
            cool.insert(left);
            cool.insert(right);
            merges += 1;
        }
        merges
    }

    pub fn prune_pass(&mut self, now: f64) -> usize {
        let to_remove: Vec<_> = self
            .nodes
            .iter_mut()
            .filter_map(|(id, node)| {
                node.apply_decay(now, self.config.eps);
                (node.weight < self.config.prune_weight_threshold
                    && (now - node.last_seen).max(0.0) as f32 >= self.config.minimum_prune_age)
                    .then_some(*id)
            })
            .collect();
        for id in &to_remove {
            self.nodes.remove(id);
            self.graph.remove(id);
            for row in self.graph.values_mut() {
                row.remove(id);
            }
            self.visit_counts.remove(id);
            self.coactivations
                .retain(|(left, right), _| left != id && right != id);
        }
        to_remove.len()
    }

    pub fn dream_pass(&mut self, now: f64) -> MaintenanceReport {
        let ids: Vec<_> = self.nodes.keys().copied().collect();
        if ids.len() < 2 || self.config.dream_walks == 0 {
            return MaintenanceReport::default();
        }
        self.logical_time = self.logical_time.max(now);
        self.decay_edges(now);

        let uniform_seeds: Vec<_> = ids.iter().map(|id| (*id, 1.0)).collect();
        let importance = self.personalized_pagerank(&uniform_seeds);
        let seed_weights: Vec<_> = ids
            .iter()
            .map(|id| importance.get(id).copied().unwrap_or_default().max(0.0))
            .collect();
        let cycle = self.dream_counter;
        self.dream_counter = self.dream_counter.saturating_add(1);
        let mut visited_global = BTreeSet::new();
        let mut walk_coactivations: BTreeMap<(NodeId, NodeId), f32> = BTreeMap::new();
        let mut walks_completed = 0;

        for walk in 0..self.config.dream_walks {
            let mut current = ids
                [deterministic_choice(&seed_weights, dream_nonce(cycle, walk, usize::MAX, None))];
            let mut walk_visited = BTreeSet::from([current]);

            for step in 0..self.config.dream_walk_length {
                let Some(row) = self.graph.get(&current) else {
                    break;
                };
                let neighbors: Vec<_> = row
                    .iter()
                    .filter(|(id, weight)| self.nodes.contains_key(id) && **weight > 0.0)
                    .map(|(id, weight)| (*id, *weight))
                    .collect();
                if neighbors.is_empty() {
                    break;
                }
                let maximum_visits = neighbors
                    .iter()
                    .map(|(id, _)| self.nodes[id].visits)
                    .fold(1.0_f32, f32::max);
                let temperature = self.config.dream_temperature.max(1e-6);
                let logits: Vec<_> = neighbors
                    .iter()
                    .map(|(id, weight)| {
                        weight / temperature
                            + self.config.dream_exploration_bonus
                                * (1.0 - self.nodes[id].visits / maximum_visits)
                    })
                    .collect();
                let maximum_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<_> = logits
                    .iter()
                    .map(|logit| (*logit - maximum_logit).exp())
                    .collect();
                let index =
                    deterministic_choice(&weights, dream_nonce(cycle, walk, step, Some(current)));
                current = neighbors[index].0;
                walk_visited.insert(current);
            }

            let unique: Vec<_> = walk_visited.into_iter().collect();
            for (index, left) in unique.iter().enumerate() {
                visited_global.insert(*left);
                for right in unique.iter().skip(index + 1) {
                    *walk_coactivations.entry((*left, *right)).or_default() += 1.0;
                }
            }
            walks_completed += 1;
        }

        for ((left, right), count) in walk_coactivations {
            *self.visit_counts.entry(left).or_default() += count;
            *self.visit_counts.entry(right).or_default() += count;
            *self.coactivations.entry((left, right)).or_default() += count;
            *self.coactivations.entry((right, left)).or_default() += count;
        }
        self.refresh_edges();
        for id in &visited_global {
            let node = self.nodes.get_mut(id).expect("visited dream node exists");
            node.tau = (node.tau * (1.0 + self.config.dream_stickiness))
                .clamp(self.config.tau_min, self.config.tau_max);
        }
        let merges = self.consolidate_pass();
        MaintenanceReport {
            merges,
            pruned: 0,
            walks_completed,
            nodes_reactivated: visited_global.len(),
        }
    }

    pub fn maintain(&mut self, now: f64, maximum_rounds: usize) -> MaintenanceReport {
        let mut report = MaintenanceReport::default();
        for _ in 0..maximum_rounds {
            let merged = self.consolidate_pass();
            report.merges += merged;
            if merged == 0 {
                break;
            }
        }
        report.pruned = self.prune_pass(now);
        let dream = self.dream_pass(now);
        report.merges += dream.merges;
        report.walks_completed = dream.walks_completed;
        report.nodes_reactivated = dream.nodes_reactivated;
        for _ in 0..maximum_rounds {
            let merged = self.consolidate_pass();
            report.merges += merged;
            if merged == 0 {
                break;
            }
        }
        report
    }

    pub fn check_invariants(&self) -> Vec<String> {
        let active: BTreeSet<_> = self.nodes.keys().copied().collect();
        let mut errors = Vec::new();
        for (source, row) in &self.graph {
            if !active.contains(source) {
                errors.push(format!("graph has stale row {source}"));
            }
            for target in row.keys() {
                if !active.contains(target) {
                    errors.push(format!("graph row {source} has stale target {target}"));
                }
            }
        }
        for id in self.visit_counts.keys() {
            if !active.contains(id) {
                errors.push(format!("visit counts contain stale node {id}"));
            }
        }
        for (left, right) in self.coactivations.keys() {
            if !active.contains(left) || !active.contains(right) {
                errors.push(format!("coactivation contains stale pair {left}, {right}"));
            }
        }
        errors
    }

    fn merge_nodes(
        &mut self,
        left: NodeId,
        right: NodeId,
        similarity: f32,
        sufficient_sum: Vec<f32>,
        effective_count: f32,
        kappa: f32,
    ) {
        let absorbed = self.nodes.remove(&right).expect("merge target exists");
        let visits = self.visit_counts.remove(&right).unwrap_or_default()
            + self.visit_counts.get(&left).copied().unwrap_or_default();
        let left_node = self.nodes.get_mut(&left).expect("merge source exists");
        left_node.centroid = vector::unit(&sufficient_sum, self.config.eps);
        left_node.sufficient_sum = sufficient_sum;
        left_node.effective_count = effective_count;
        left_node.weight += absorbed.weight
            + self.config.merge_bonus * left_node.weight.min(absorbed.weight) * similarity.max(0.0);
        left_node.kappa = kappa;
        left_node.confidence = vector::sigmoid(
            self.config.concentration_scale * kappa - self.config.concentration_bias,
        ) * (effective_count
            / self.config.confidence_mass_minimum.max(self.config.eps))
        .min(1.0);
        left_node.tau = 0.5 * (left_node.tau + absorbed.tau);
        left_node.last_seen = left_node.last_seen.max(absorbed.last_seen);
        left_node.last_decay = left_node.last_decay.max(absorbed.last_decay);
        left_node.visits += absorbed.visits;
        for event_id in &absorbed.event_ids {
            if !left_node.event_ids.contains(event_id) {
                left_node.event_ids.push(*event_id);
            }
        }
        for (source, count) in &absorbed.source_counts {
            *left_node.source_counts.entry(source.clone()).or_default() += count;
        }
        for (key, value) in &absorbed.metadata {
            left_node
                .metadata
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
        left_node.level = if left_node.level == absorbed.level {
            left_node.level.saturating_add(1)
        } else {
            left_node.level.max(absorbed.level)
        };
        left_node.children.push(right);
        left_node.children.extend(absorbed.children.iter().copied());
        self.visit_counts.insert(left, visits);

        let right_edges = self.graph.remove(&right).unwrap_or_default();
        for (neighbor, weight) in right_edges {
            if neighbor == left {
                continue;
            }
            self.graph
                .entry(left)
                .or_default()
                .entry(neighbor)
                .and_modify(|existing| *existing = existing.max(weight))
                .or_insert(weight);
            let reverse = self
                .graph
                .get(&neighbor)
                .and_then(|row| row.get(&right))
                .copied()
                .unwrap_or_default();
            self.graph
                .entry(neighbor)
                .or_default()
                .entry(left)
                .and_modify(|existing| *existing = existing.max(reverse))
                .or_insert(reverse);
        }
        for row in self.graph.values_mut() {
            row.remove(&right);
        }
        if let Some(row) = self.graph.get_mut(&left) {
            row.remove(&left);
            let mut edges: Vec<_> = row.iter().map(|(id, weight)| (*id, *weight)).collect();
            edges.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            edges.truncate(self.config.top_edges);
            *row = edges.into_iter().collect();
        }

        let active: Vec<_> = self.nodes.keys().copied().collect();
        for neighbor in active {
            if neighbor == left {
                continue;
            }
            let combined = self
                .coactivations
                .get(&(left, neighbor))
                .copied()
                .unwrap_or_default()
                + self
                    .coactivations
                    .get(&(right, neighbor))
                    .copied()
                    .unwrap_or_default();
            if combined > 0.0 {
                self.coactivations.insert((left, neighbor), combined);
                self.coactivations.insert((neighbor, left), combined);
            }
        }
        self.coactivations
            .retain(|(source, target), _| source != &right && target != &right);
        self.absorbed.insert(right, absorbed);
    }

    fn spawn(
        &mut self,
        event_id: EventId,
        input: &[f32],
        now: f64,
        source: Option<&str>,
        metadata: BTreeMap<String, String>,
    ) -> NodeId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"spine-dcmdb-node-v1");
        hasher.update(event_id.as_bytes());
        hasher.update(&self.spawn_counter.to_be_bytes());
        self.spawn_counter = self.spawn_counter.saturating_add(1);
        let id = NodeId::from_bytes(*hasher.finalize().as_bytes());
        let kappa = kappa_from_m(1.0, self.config.dimension, self.config.eps);
        let confidence = vector::sigmoid(
            self.config.concentration_scale * kappa - self.config.concentration_bias,
        ) * (1.0 / self.config.confidence_mass_minimum.max(self.config.eps))
            .min(1.0);
        let mut source_counts = BTreeMap::new();
        let mut visits = 0.0;
        if let Some(source) = source {
            source_counts.insert(source.to_owned(), 1.0);
            visits = 1.0;
        }
        self.nodes.insert(
            id,
            MemoryNode {
                id,
                event_ids: vec![event_id],
                centroid: input.to_vec(),
                sufficient_sum: input.to_vec(),
                effective_count: 1.0,
                weight: self.config.initial_weight,
                kappa,
                confidence,
                tau: self.config.tau0,
                last_decay: now,
                last_seen: now,
                visits,
                atypical_count: 0,
                tension: None,
                source_counts,
                level: 0,
                children: Vec::new(),
                metadata,
            },
        );
        self.graph.entry(id).or_default();
        self.visit_counts.entry(id).or_default();
        id
    }

    fn nearest(&self, input: &[f32], count: usize) -> Vec<(NodeId, f32)> {
        let mut scores: Vec<_> = self
            .nodes
            .iter()
            .map(|(id, node)| (*id, vector::dot(&node.centroid, input)))
            .collect();
        scores.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        scores.truncate(count);
        scores
    }

    fn update_node(
        &mut self,
        id: NodeId,
        event_id: EventId,
        input: &[f32],
        similarity: f32,
        now: f64,
        source: Option<&str>,
    ) {
        let config = &self.config;
        let node = self.nodes.get_mut(&id).expect("candidate node exists");
        if !node.event_ids.contains(&event_id) {
            node.event_ids.push(event_id);
        }
        let elapsed = (now - node.last_seen).max(0.0) as f32;
        let gamma = config.gamma0
            * (1.0 + config.lambda_age * (elapsed / node.tau.max(config.eps)).min(1.0))
            * (1.0 + config.lambda_uncertainty * (1.0 - node.confidence));
        let beta = gamma / (node.weight + gamma);
        let theta = vector::clamp_unit(similarity).acos();
        node.centroid = if theta < 1e-8 {
            input.to_vec()
        } else {
            let beta_hat = beta.min(config.max_update_angle / theta);
            vector::slerp(&node.centroid, input, beta_hat, config.eps)
        };
        let eta = beta * similarity;
        for (sum, input) in node.sufficient_sum.iter_mut().zip(input) {
            *sum += eta * input;
        }
        node.effective_count += eta;
        let mean_resultant =
            vector::norm(&node.sufficient_sum) / node.effective_count.max(config.eps);
        node.kappa = kappa_from_m(mean_resultant, config.dimension, config.eps);
        node.confidence =
            vector::sigmoid(config.concentration_scale * node.kappa - config.concentration_bias)
                * (node.effective_count / config.confidence_mass_minimum.max(config.eps)).min(1.0);
        node.weight += config.delta0 * similarity;
        node.last_seen = now;
        node.visits += 1.0;
        node.tau =
            (node.tau * (1.0 + config.dream_stickiness)).clamp(config.tau_min, config.tau_max);
        if let Some(source) = source {
            *node.source_counts.entry(source.to_owned()).or_default() += 1.0;
        }

        let angle = vector::clamp_unit(similarity).acos();
        let expected_angle = vector::clamp_unit(mean_resultant).acos();
        let atypical =
            similarity >= config.theta_similarity && angle > expected_angle + config.zeta;
        if atypical {
            let had_tension = node.tension.is_some();
            node.atypical_count = node.atypical_count.saturating_add(1);
            let tension_vector: Vec<f32> = input
                .iter()
                .zip(&node.centroid)
                .map(|(input, centroid)| input - centroid)
                .collect();
            if let Some(tension) = &mut node.tension {
                for (sum, input) in tension.sufficient_sum.iter_mut().zip(input) {
                    *sum += input;
                }
                tension.effective_count += 1.0;
                tension.evidence_count = tension.evidence_count.saturating_add(1);
                tension.vector = tension_vector;
                tension.uncertainty_radius = angle;
                tension.trajectory.push(input.to_vec());
                tension.center = vector::unit(&tension.sufficient_sum, config.eps);
            } else {
                let combined: Vec<f32> = node
                    .centroid
                    .iter()
                    .zip(input)
                    .map(|(centroid, input)| centroid + input)
                    .collect();
                node.tension = Some(TensionInfo {
                    center: vector::unit(&combined, config.eps),
                    vector: tension_vector,
                    uncertainty_radius: angle,
                    trajectory: vec![input.to_vec()],
                    sufficient_sum: input.to_vec(),
                    effective_count: 1.0,
                    log_bayes_factor: 0.0,
                    evidence_count: 1,
                });
            }
            // The Python oracle begins Bayes-factor accumulation with the
            // second observation assigned to an existing tension cluster.
            if had_tension {
                update_split_bayes_factor(node, config);
            }
        } else if node.tension.is_some() {
            if let Some(tension) = &mut node.tension {
                tension.evidence_count = tension.evidence_count.saturating_add(1);
            }
            update_split_bayes_factor(node, config);
            let clear = node.tension.as_ref().is_some_and(|tension| {
                tension.evidence_count >= config.minimum_tension_evidence
                    && vector::sigmoid(tension.log_bayes_factor) < config.clear_probability
            });
            if clear {
                node.atypical_count = 0;
                node.tension = None;
            }
        } else {
            node.atypical_count = node.atypical_count.saturating_sub(1);
        }
    }

    fn promote_if_tension(
        &mut self,
        node_id: NodeId,
        event_id: EventId,
        input: &[f32],
        now: f64,
        source: Option<&str>,
    ) -> Option<NodeId> {
        let tension = self.nodes.get(&node_id)?.tension.clone()?;
        let should_spawn = if tension.evidence_count < self.config.minimum_tension_evidence {
            self.nodes[&node_id].atypical_count >= self.config.tension_promote_count
        } else {
            vector::sigmoid(tension.log_bayes_factor) > self.config.split_probability
        };
        if should_spawn {
            let spawn_vector = if vector::norm(&tension.sufficient_sum) > self.config.eps {
                vector::unit(&tension.sufficient_sum, self.config.eps)
            } else {
                input.to_vec()
            };
            let node = self.nodes.get_mut(&node_id)?;
            if tension.evidence_count >= self.config.minimum_tension_evidence {
                for (sum, tension_sum) in
                    node.sufficient_sum.iter_mut().zip(&tension.sufficient_sum)
                {
                    *sum -= tension_sum;
                }
                node.effective_count =
                    (node.effective_count - tension.effective_count).max(self.config.eps);
                node.centroid = vector::unit(&node.sufficient_sum, self.config.eps);
            }
            node.atypical_count = 0;
            node.tension = None;
            return Some(self.spawn(event_id, &spawn_vector, now, source, BTreeMap::new()));
        }
        let should_clear = tension.evidence_count >= self.config.minimum_tension_evidence
            && vector::sigmoid(tension.log_bayes_factor) < self.config.clear_probability;
        if should_clear {
            let node = self.nodes.get_mut(&node_id)?;
            node.atypical_count = 0;
            node.tension = None;
        }
        None
    }

    fn decay_edges(&mut self, now: f64) {
        let elapsed = (now - self.last_edge_decay).max(0.0) as f32;
        if elapsed <= 0.0 {
            return;
        }
        let rho = (-elapsed / self.config.tau_edge.max(self.config.eps)).exp();
        for count in self.visit_counts.values_mut() {
            *count *= rho;
        }
        for count in self.coactivations.values_mut() {
            *count *= rho;
        }
        for row in self.graph.values_mut() {
            for weight in row.values_mut() {
                *weight *= rho;
            }
        }
        self.last_edge_decay = now;
    }

    fn record_coactivation(&mut self, ids: &[NodeId]) {
        for id in ids {
            *self.visit_counts.entry(*id).or_default() += 1.0;
        }
        for (index, left) in ids.iter().enumerate() {
            for right in ids.iter().skip(index + 1) {
                *self.coactivations.entry((*left, *right)).or_default() += 1.0;
                *self.coactivations.entry((*right, *left)).or_default() += 1.0;
            }
        }
    }

    fn refresh_edges(&mut self) {
        let total: f32 = self.visit_counts.values().sum::<f32>() + self.config.pmi_epsilon;
        if total <= 0.0 {
            return;
        }
        for ((left, right), joint_count) in &self.coactivations {
            let left_count =
                self.visit_counts.get(left).copied().unwrap_or_default() + self.config.pmi_epsilon;
            let right_count =
                self.visit_counts.get(right).copied().unwrap_or_default() + self.config.pmi_epsilon;
            let joint = (joint_count + self.config.pmi_epsilon) / total;
            let independent = (left_count / total) * (right_count / total);
            let ppmi = (joint.max(1e-12) / independent.max(1e-12)).ln().max(0.0);
            let previous = self
                .graph
                .entry(*left)
                .or_default()
                .get(right)
                .copied()
                .unwrap_or_default();
            self.graph.entry(*left).or_default().insert(
                *right,
                (1.0 - self.config.pmi_learning_rate) * previous
                    + self.config.pmi_learning_rate * ppmi,
            );
        }
        for row in self.graph.values_mut() {
            let mut edges: Vec<_> = row.iter().map(|(id, weight)| (*id, *weight)).collect();
            edges.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            edges.truncate(self.config.top_edges);
            let retained: BTreeSet<_> = edges.into_iter().map(|(id, _)| id).collect();
            row.retain(|id, weight| retained.contains(id) && *weight > 0.0);
        }
    }

    fn personalized_pagerank(&self, seeds: &[(NodeId, f32)]) -> BTreeMap<NodeId, f32> {
        let ids: Vec<_> = self.nodes.keys().copied().collect();
        if ids.is_empty() {
            return BTreeMap::new();
        }
        let mut restart = BTreeMap::new();
        let seed_total: f32 = seeds.iter().map(|(_, weight)| *weight).sum();
        if seed_total > 0.0 {
            for (id, weight) in seeds {
                restart.insert(*id, *weight / seed_total);
            }
        } else {
            for id in &ids {
                restart.insert(*id, 1.0 / ids.len() as f32);
            }
        }
        let mut rank = restart.clone();
        for _ in 0..100 {
            let mut next: BTreeMap<NodeId, f32> = ids
                .iter()
                .map(|id| {
                    (
                        *id,
                        self.config.pagerank_restart * restart.get(id).copied().unwrap_or_default(),
                    )
                })
                .collect();
            for source in &ids {
                let source_rank = rank.get(source).copied().unwrap_or_default();
                let row = self.graph.get(source);
                let total = row
                    .map(|row| row.values().map(|weight| weight.max(0.0)).sum())
                    .unwrap_or(0.0);
                if total <= 0.0 {
                    *next.entry(*source).or_default() +=
                        (1.0 - self.config.pagerank_restart) * source_rank;
                } else if let Some(row) = row {
                    for (target, weight) in row {
                        if self.nodes.contains_key(target) {
                            *next.entry(*target).or_default() += (1.0
                                - self.config.pagerank_restart)
                                * source_rank
                                * weight.max(0.0)
                                / total;
                        }
                    }
                }
            }
            let change: f32 = ids
                .iter()
                .map(|id| {
                    (next.get(id).copied().unwrap_or_default()
                        - rank.get(id).copied().unwrap_or_default())
                    .abs()
                })
                .sum();
            rank = next;
            if change < 1e-6 {
                break;
            }
        }
        rank
    }
}

fn dream_nonce(cycle: u64, walk: usize, step: usize, current: Option<NodeId>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"spine-dcmdb-dream-v1");
    hasher.update(&cycle.to_be_bytes());
    hasher.update(&(walk as u64).to_be_bytes());
    hasher.update(&(step as u64).to_be_bytes());
    if let Some(current) = current {
        hasher.update(current.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn deterministic_choice(weights: &[f32], nonce: [u8; 32]) -> usize {
    debug_assert!(!weights.is_empty());
    let total: f64 = weights
        .iter()
        .map(|weight| f64::from(weight.max(0.0)))
        .sum();
    if total <= 0.0 || !total.is_finite() {
        let raw = u64::from_be_bytes(nonce[..8].try_into().expect("eight-byte nonce prefix"));
        return (raw as usize) % weights.len();
    }
    let raw = u64::from_be_bytes(nonce[..8].try_into().expect("eight-byte nonce prefix"));
    let unit = raw as f64 / (u64::MAX as f64 + 1.0);
    let target = unit * total;
    let mut cumulative = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += f64::from(weight.max(0.0));
        if target < cumulative {
            return index;
        }
    }
    weights.len() - 1
}

fn update_split_bayes_factor(node: &mut MemoryNode, config: &DcmdbConfig) {
    let Some(tension) = &mut node.tension else {
        return;
    };
    if tension.effective_count < config.eps {
        tension.log_bayes_factor = 0.0;
        return;
    }
    let resultant = vector::norm(&tension.sufficient_sum);
    let mean = resultant / (tension.effective_count + config.kappa_regularization);
    let kappa = kappa_from_m(mean, config.dimension, config.eps);
    let main_projection = vector::dot(&node.centroid, &tension.sufficient_sum);
    let likelihood_difference = kappa * (resultant - main_projection);
    let effective_samples = config
        .effective_sample_minimum
        .max(tension.evidence_count as f32);
    let bic_penalty = effective_samples.max(config.eps).ln();
    tension.log_bayes_factor = likelihood_difference - config.split_penalty * bic_penalty;
}

pub fn kappa_from_m(mean_resultant: f32, dimension: usize, eps: f32) -> f32 {
    let mean = mean_resultant.clamp(0.0, 1.0 - 1e-9);
    if mean <= eps {
        return 0.0;
    }
    (mean * (dimension as f32 - mean * mean) / (1.0 - mean * mean + eps)).max(0.0)
}

fn log_c_vmf(kappa: f32, dimension: usize) -> f32 {
    let kappa = kappa.max(0.0);
    if kappa < 1e-3 {
        // This constant is multiplied by N and cancels exactly in merge
        // deltas because N_m = N_i + N_j.
        0.0
    } else {
        let coefficient = 0.5 * (dimension as f32 - 1.0);
        coefficient * kappa.max(1e-12).ln()
            - coefficient * (2.0 * std::f32::consts::PI).ln()
            - kappa
    }
}

fn source_total_variation(left: &MemoryNode, right: &MemoryNode) -> f32 {
    let keys: BTreeSet<_> = left
        .source_counts
        .keys()
        .chain(right.source_counts.keys())
        .collect();
    let left_total: f32 = keys
        .iter()
        .map(|key| {
            left.source_counts
                .get(*key)
                .copied()
                .unwrap_or_default()
                .max(0.0)
        })
        .sum();
    let right_total: f32 = keys
        .iter()
        .map(|key| {
            right
                .source_counts
                .get(*key)
                .copied()
                .unwrap_or_default()
                .max(0.0)
        })
        .sum();
    if left_total <= 0.0 || right_total <= 0.0 {
        return 0.0;
    }
    0.5 * keys
        .iter()
        .map(|key| {
            let left = left
                .source_counts
                .get(*key)
                .copied()
                .unwrap_or_default()
                .max(0.0)
                / left_total;
            let right = right
                .source_counts
                .get(*key)
                .copied()
                .unwrap_or_default()
                .max(0.0)
                / right_total;
            (left - right).abs()
        })
        .sum::<f32>()
}
