use std::collections::hash_map::Values;
use std::collections::{HashMap, HashSet};

use crate::isotopic_fit::IsotopicFit;
use crate::peaks::PeakKey;

use super::cluster::{DependenceCluster, SubgraphSelection, SubgraphSolverMethod};

pub type FitKey = usize;

#[derive(Debug)]
pub struct FitNode {
    pub key: FitKey,
    pub edges: HashSet<FitKey>,
    pub overlap_edges: HashSet<FitKey>,
    pub peak_indices: HashSet<PeakKey>,
    pub score: f64,
    pub start: f64,
    pub end: f64,
}

impl FitNode {
    pub fn from_fit(fit: &IsotopicFit, key: FitKey, start: f64, end: f64) -> Self {
        let mut indices = HashSet::new();
        indices.extend(fit.experimental.iter().copied());
        Self {
            key,
            score: fit.score,
            overlap_edges: HashSet::new(),
            edges: HashSet::new(),
            peak_indices: indices,
            start,
            end,
        }
    }

    pub fn is_disjoint(&self, other: &FitNode) -> bool {
        self.peak_indices.is_disjoint(&other.peak_indices)
    }

    pub fn overlaps(&self, other: &FitNode) -> bool {
        !self.is_disjoint(other)
    }

    pub fn visit(&mut self, other: &mut FitNode) {
        if self.is_disjoint(other) {
            self.edges.insert(other.key);
            other.edges.insert(self.key);
        } else {
            self.overlap_edges.insert(other.key);
            other.overlap_edges.insert(self.key);
        }
    }

    pub fn peak_iter(&self) -> std::collections::hash_set::Iter<PeakKey> {
        self.peak_indices.iter()
    }

    pub fn create_ref(&self) -> FitRef {
        FitRef::new(self.key, self.score, self.start, self.end)
    }
}

impl PartialEq for FitNode {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for FitNode {}
impl std::hash::Hash for FitNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for FitNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FitRef {
    pub key: FitKey,
    pub score: f64,
    pub start: f64,
    pub end: f64,
}

impl FitRef {
    pub fn new(key: FitKey, score: f64, start: f64, end: f64) -> Self {
        Self {
            key,
            score,
            start,
            end,
        }
    }
}

impl PartialEq for FitRef {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for FitRef {}
impl std::hash::Hash for FitRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for FitRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub(crate) type FitNodeGraphInner = HashMap<FitKey, FitNode>;

#[derive(Debug, Default)]
pub struct FitGraph {
    pub fit_nodes: FitNodeGraphInner,
    pub dependencies: HashMap<FitKey, IsotopicFit>,
}

impl FitGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &FitKey) -> Option<&FitNode> {
        self.fit_nodes.get(key)
    }

    pub fn get_mut(&mut self, key: &FitKey) -> Option<&mut FitNode> {
        self.fit_nodes.get_mut(key)
    }

    pub fn add_fit(&mut self, fit: IsotopicFit, start: f64, end: f64) -> &FitNode {
        let key = self.fit_nodes.len();
        let node = FitNode::from_fit(&fit, key, start, end);
        self.fit_nodes.insert(key.clone(), node);
        self.dependencies.insert(key, fit);
        self.fit_nodes.get(&key).unwrap()
    }

    pub fn remove(&mut self, key: &FitKey) -> Option<FitNode> {
        self.fit_nodes.remove(key)
    }

    pub fn values(&self) -> Values<FitKey, FitNode> {
        self.fit_nodes.values()
    }

    pub fn replace_nodes(&mut self, nodes: HashMap<FitKey, FitNode>) {
        self.fit_nodes = nodes;
    }

    pub fn len(&self) -> usize {
        self.fit_nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fit_nodes.is_empty()
    }

    /// Put [`DependenceCluster`] next to the [`FitNode`]s that references it
    fn split_nodes<'a>(
        &mut self,
        clusters: &'a [DependenceCluster],
    ) -> Vec<(&'a DependenceCluster, Vec<FitNode>)> {
        let mut result = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            let nodes = cluster
                .dependencies
                .iter()
                .map(|f| self.fit_nodes.remove(&f.key))
                .flatten();
            result.push((cluster, nodes.collect()));
        }
        result
    }

    pub fn solve_subgraphs<'a>(&mut self, clusters: &'a [DependenceCluster], method: SubgraphSolverMethod) -> Vec<Vec<FitRef>> {
        let clusters_and_nodes = self.split_nodes(clusters);

        let mut solutions = Vec::with_capacity(clusters.len());

        for (cluster, nodes) in clusters_and_nodes {
            let subgraph = SubgraphSelection::from_nodes(nodes, cluster.score_ordering);

            let (solution, mut nodes) = subgraph.solve(method);
            // Re-absorb solved sub-graph
            self.fit_nodes.extend(nodes.drain());
            solutions.push(solution);
        }
        solutions
    }
}

impl std::ops::Index<&FitKey> for FitGraph {
    type Output = FitNode;

    fn index(&self, index: &FitKey) -> &Self::Output {
        &self.fit_nodes[index]
    }
}
