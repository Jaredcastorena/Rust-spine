use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Dcmdb, Embedding, HeartError, NodeId, Result, TriangleId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ContextHandle {
    Node(NodeId),
    Triangle(TriangleId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContextLeaf {
    pub node_id: NodeId,
    pub chronology: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContextBranch {
    pub handle: ContextHandle,
    pub chronology_start: u64,
    pub chronology_end: u64,
}

impl ContextBranch {
    pub fn leaf(leaf: ContextLeaf) -> Self {
        Self {
            handle: ContextHandle::Node(leaf.node_id),
            chronology_start: leaf.chronology,
            chronology_end: leaf.chronology,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextTriangle {
    pub id: TriangleId,
    pub apex: NodeId,
    pub left: ContextHandle,
    pub right: ContextHandle,
    pub chronology_start: u64,
    pub chronology_end: u64,
    pub relationship_strength: f32,
    pub depth: u32,
    pub leaf_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriangleConfig {
    pub minimum_coherence: f32,
}

impl Default for TriangleConfig {
    fn default() -> Self {
        Self {
            minimum_coherence: 0.80,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextForest {
    pub schema: u32,
    pub config: TriangleConfig,
    pub triangles: BTreeMap<TriangleId, ContextTriangle>,
    pub roots: Vec<ContextBranch>,
}

impl Default for ContextForest {
    fn default() -> Self {
        Self {
            schema: 1,
            config: TriangleConfig::default(),
            triangles: BTreeMap::new(),
            roots: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RehydrateBudget {
    pub max_depth: u32,
    pub max_fanout: usize,
    pub max_nodes: usize,
    pub max_tokens: usize,
}

impl Default for RehydrateBudget {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_fanout: 8,
            max_nodes: 32,
            max_tokens: 4_096,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateRole {
    Apex,
    PreferredLeaf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedCoordinate {
    pub node_id: NodeId,
    pub role: CoordinateRole,
    pub depth: u32,
    pub estimated_tokens: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RehydratedContext {
    pub coordinates: Vec<ResolvedCoordinate>,
    pub alternate_routes: Vec<ContextHandle>,
    pub consumed_tokens: usize,
    pub truncated: bool,
}

impl ContextForest {
    pub fn with_config(config: TriangleConfig) -> Result<Self> {
        if !config.minimum_coherence.is_finite()
            || !(-1.0..=1.0).contains(&config.minimum_coherence)
        {
            return Err(HeartError::InvalidInput(
                "triangle coherence must be finite and in [-1, 1]".into(),
            ));
        }
        Ok(Self {
            config,
            ..Self::default()
        })
    }

    pub fn compact(
        &mut self,
        leaves: impl IntoIterator<Item = ContextLeaf>,
        dcmdb: &Dcmdb,
        target_roots: usize,
    ) -> Result<&[ContextBranch]> {
        if target_roots == 0 {
            return Err(HeartError::InvalidInput(
                "triangle target root count must be positive".into(),
            ));
        }
        let mut branches = self.roots.clone();
        for leaf in leaves {
            if dcmdb.node(leaf.node_id).is_none() {
                return Err(HeartError::NotFound);
            }
            branches.push(ContextBranch::leaf(leaf));
        }
        branches = coalesce_branches(branches);
        branches.sort_by(branch_order);
        while branches.len() > target_roots {
            let Some(candidate) = self.strongest_pair(&branches, dcmdb) else {
                break;
            };
            let (first_index, second_index) = if candidate.left_index < candidate.right_index {
                (candidate.left_index, candidate.right_index)
            } else {
                (candidate.right_index, candidate.left_index)
            };
            let right_removed = branches.remove(second_index);
            let left_removed = branches.remove(first_index);
            let (left, right) = if branch_order(&left_removed, &right_removed).is_le() {
                (left_removed, right_removed)
            } else {
                (right_removed, left_removed)
            };
            let depth = 1 + self.branch_depth(left).max(self.branch_depth(right));
            let leaf_count = self.branch_leaf_count(left) + self.branch_leaf_count(right);
            let chronology_start = left.chronology_start.min(right.chronology_start);
            let chronology_end = left.chronology_end.max(right.chronology_end);
            let triangle = ContextTriangle {
                id: triangle_id(
                    candidate.apex,
                    left.handle,
                    right.handle,
                    chronology_start,
                    chronology_end,
                ),
                apex: candidate.apex,
                left: left.handle,
                right: right.handle,
                chronology_start,
                chronology_end,
                relationship_strength: candidate.strength,
                depth,
                leaf_count,
            };
            self.verify_triangle(&triangle, dcmdb)?;
            let branch = ContextBranch {
                handle: ContextHandle::Triangle(triangle.id),
                chronology_start: triangle.chronology_start,
                chronology_end: triangle.chronology_end,
            };
            self.triangles.insert(triangle.id, triangle);
            branches.push(branch);
            branches.sort_by(branch_order);
        }
        self.roots = branches;
        self.verify(dcmdb)?;
        Ok(&self.roots)
    }

    pub fn verify(&self, dcmdb: &Dcmdb) -> Result<()> {
        for triangle in self.triangles.values() {
            self.verify_triangle(triangle, dcmdb)?;
        }
        for root in &self.roots {
            self.verify_handle(root.handle, dcmdb)?;
        }
        Ok(())
    }

    pub fn rehydrate(
        &self,
        root: ContextHandle,
        query: Option<&Embedding>,
        dcmdb: &Dcmdb,
        budget: RehydrateBudget,
    ) -> Result<RehydratedContext> {
        self.verify_handle(root, dcmdb)?;
        let mut result = RehydratedContext::default();
        let mut visited = BTreeSet::new();
        self.rehydrate_inner(root, query, dcmdb, budget, 0, &mut visited, &mut result)?;
        Ok(result)
    }

    fn strongest_pair(&self, branches: &[ContextBranch], dcmdb: &Dcmdb) -> Option<PairCandidate> {
        let mut best: Option<PairCandidate> = None;
        for left_index in 0..branches.len() {
            for right_index in left_index + 1..branches.len() {
                let left_coordinate = self.coordinate(branches[left_index].handle)?;
                let right_coordinate = self.coordinate(branches[right_index].handle)?;
                let Some((apex, coherence)) = dcmdb.coherent_apex(
                    left_coordinate,
                    right_coordinate,
                    self.config.minimum_coherence,
                ) else {
                    continue;
                };
                let relationship = dcmdb
                    .relationship_score(left_coordinate, right_coordinate)
                    .unwrap_or(coherence);
                let candidate = PairCandidate {
                    left_index,
                    right_index,
                    apex,
                    strength: relationship.min(coherence),
                    chronology: (
                        branches[left_index].chronology_start,
                        branches[right_index].chronology_start,
                    ),
                    handles: (branches[left_index].handle, branches[right_index].handle),
                };
                if best.as_ref().is_none_or(|best| candidate.better_than(best)) {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    fn coordinate(&self, handle: ContextHandle) -> Option<NodeId> {
        match handle {
            ContextHandle::Node(node) => Some(node),
            ContextHandle::Triangle(id) => self.triangles.get(&id).map(|triangle| triangle.apex),
        }
    }

    fn branch_depth(&self, branch: ContextBranch) -> u32 {
        match branch.handle {
            ContextHandle::Node(_) => 0,
            ContextHandle::Triangle(id) => self.triangles.get(&id).map_or(0, |item| item.depth),
        }
    }

    fn branch_leaf_count(&self, branch: ContextBranch) -> u64 {
        match branch.handle {
            ContextHandle::Node(_) => 1,
            ContextHandle::Triangle(id) => {
                self.triangles.get(&id).map_or(0, |item| item.leaf_count)
            }
        }
    }

    fn verify_triangle(&self, triangle: &ContextTriangle, dcmdb: &Dcmdb) -> Result<()> {
        if dcmdb.node(triangle.apex).is_none() {
            return Err(HeartError::NotFound);
        }
        self.verify_handle(triangle.left, dcmdb)?;
        self.verify_handle(triangle.right, dcmdb)?;
        if triangle.left == triangle.right
            || triangle.id
                != triangle_id(
                    triangle.apex,
                    triangle.left,
                    triangle.right,
                    triangle.chronology_start,
                    triangle.chronology_end,
                )
        {
            return Err(HeartError::InvalidInput(
                "invalid context triangle identity or branches".into(),
            ));
        }
        Ok(())
    }

    fn verify_handle(&self, handle: ContextHandle, dcmdb: &Dcmdb) -> Result<()> {
        match handle {
            ContextHandle::Node(node) if dcmdb.node(node).is_some() => Ok(()),
            ContextHandle::Triangle(id) if self.triangles.contains_key(&id) => Ok(()),
            ContextHandle::Node(_) | ContextHandle::Triangle(_) => Err(HeartError::NotFound),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rehydrate_inner(
        &self,
        handle: ContextHandle,
        query: Option<&Embedding>,
        dcmdb: &Dcmdb,
        budget: RehydrateBudget,
        depth: u32,
        visited: &mut BTreeSet<ContextHandle>,
        result: &mut RehydratedContext,
    ) -> Result<()> {
        if !visited.insert(handle) {
            return Err(HeartError::InvalidInput(
                "cycle detected in context triangle".into(),
            ));
        }
        if depth > budget.max_depth || result.coordinates.len() >= budget.max_nodes {
            result.truncated = true;
            return Ok(());
        }
        match handle {
            ContextHandle::Node(node_id) => {
                self.push_coordinate(
                    node_id,
                    CoordinateRole::PreferredLeaf,
                    depth,
                    dcmdb,
                    budget,
                    result,
                );
            }
            ContextHandle::Triangle(id) => {
                let triangle = self.triangles.get(&id).ok_or(HeartError::NotFound)?;
                if !self.push_coordinate(
                    triangle.apex,
                    CoordinateRole::Apex,
                    depth,
                    dcmdb,
                    budget,
                    result,
                ) {
                    return Ok(());
                }
                let (preferred, alternate) = self.preferred_branches(triangle, query, dcmdb);
                if result.alternate_routes.len() < budget.max_fanout {
                    result.alternate_routes.push(alternate);
                } else {
                    result.truncated = true;
                }
                self.rehydrate_inner(preferred, query, dcmdb, budget, depth + 1, visited, result)?;
            }
        }
        Ok(())
    }

    fn push_coordinate(
        &self,
        node_id: NodeId,
        role: CoordinateRole,
        depth: u32,
        dcmdb: &Dcmdb,
        budget: RehydrateBudget,
        result: &mut RehydratedContext,
    ) -> bool {
        if result
            .coordinates
            .iter()
            .any(|item| item.node_id == node_id)
        {
            return true;
        }
        let estimated_tokens = dcmdb
            .node(node_id)
            .and_then(|node| node.metadata.get("token_count"))
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        if result.coordinates.len() >= budget.max_nodes
            || result.consumed_tokens.saturating_add(estimated_tokens) > budget.max_tokens
        {
            result.truncated = true;
            return false;
        }
        result.coordinates.push(ResolvedCoordinate {
            node_id,
            role,
            depth,
            estimated_tokens,
        });
        result.consumed_tokens += estimated_tokens;
        true
    }

    fn preferred_branches(
        &self,
        triangle: &ContextTriangle,
        query: Option<&Embedding>,
        dcmdb: &Dcmdb,
    ) -> (ContextHandle, ContextHandle) {
        let Some(query) = query else {
            return (triangle.right, triangle.left);
        };
        let left = self
            .coordinate(triangle.left)
            .and_then(|node| dcmdb.node(node))
            .map_or(f32::NEG_INFINITY, |node| {
                dot(node.centroid.as_slice(), query.as_slice())
            });
        let right = self
            .coordinate(triangle.right)
            .and_then(|node| dcmdb.node(node))
            .map_or(f32::NEG_INFINITY, |node| {
                dot(node.centroid.as_slice(), query.as_slice())
            });
        if left > right {
            (triangle.left, triangle.right)
        } else {
            (triangle.right, triangle.left)
        }
    }
}

fn coalesce_branches(branches: Vec<ContextBranch>) -> Vec<ContextBranch> {
    let mut unique = BTreeMap::<ContextHandle, ContextBranch>::new();
    for branch in branches {
        unique
            .entry(branch.handle)
            .and_modify(|existing| {
                existing.chronology_start = existing.chronology_start.min(branch.chronology_start);
                existing.chronology_end = existing.chronology_end.max(branch.chronology_end);
            })
            .or_insert(branch);
    }
    unique.into_values().collect()
}

#[derive(Clone, Copy)]
struct PairCandidate {
    left_index: usize,
    right_index: usize,
    apex: NodeId,
    strength: f32,
    chronology: (u64, u64),
    handles: (ContextHandle, ContextHandle),
}

impl PairCandidate {
    fn better_than(&self, other: &Self) -> bool {
        self.strength
            .total_cmp(&other.strength)
            .then_with(|| other.chronology.cmp(&self.chronology))
            .then_with(|| other.handles.cmp(&self.handles))
            .is_gt()
    }
}

fn branch_order(left: &ContextBranch, right: &ContextBranch) -> std::cmp::Ordering {
    left.chronology_start
        .cmp(&right.chronology_start)
        .then_with(|| left.chronology_end.cmp(&right.chronology_end))
        .then_with(|| left.handle.cmp(&right.handle))
}

fn triangle_id(
    apex: NodeId,
    left: ContextHandle,
    right: ContextHandle,
    chronology_start: u64,
    chronology_end: u64,
) -> TriangleId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"spine-context-triangle-v1");
    hasher.update(apex.as_bytes());
    hash_handle(&mut hasher, left);
    hash_handle(&mut hasher, right);
    hasher.update(&chronology_start.to_be_bytes());
    hasher.update(&chronology_end.to_be_bytes());
    TriangleId::from_bytes(*hasher.finalize().as_bytes())
}

fn hash_handle(hasher: &mut blake3::Hasher, handle: ContextHandle) {
    match handle {
        ContextHandle::Node(id) => {
            hasher.update(&[0]);
            hasher.update(id.as_bytes());
        }
        ContextHandle::Triangle(id) => {
            hasher.update(&[1]);
            hasher.update(id.as_bytes());
        }
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
