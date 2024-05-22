use std::cmp;
use std::collections::hash_map::{HashMap, Values};
use std::collections::hash_set::{HashSet, Iter};
use std::fmt::Display;
use std::hash::Hash;

use fnv::FnvBuildHasher;
use identity_hash::IdentityHashable;

use crate::isotopic_fit::IsotopicFit;
use crate::peaks::PeakKey;
use crate::scorer::ScoreType;

use super::cluster::{
    DependenceCluster, SubgraphSelection, SubgraphSolution, SubgraphSolverMethod,
};
use super::graph::FitEvictionReason;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FitKey(usize);

impl Hash for FitKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.0);
    }
}

impl IdentityHashable for FitKey {}

pub type BuildIdentityHasherFitKey = identity_hash::BuildIdentityHasher<FitKey>;

impl Display for FitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct FitNode {
    pub key: FitKey,
    pub edges: HashSet<FitKey, BuildIdentityHasherFitKey>,
    pub overlap_edges: HashSet<FitKey, BuildIdentityHasherFitKey>,
    pub peak_indices: HashSet<PeakKey, FnvBuildHasher>,
    pub score: ScoreType,
    pub start: f64,
    pub end: f64,
}

impl FitNode {
    pub fn from_fit(fit: &IsotopicFit, key: FitKey, start: f64, end: f64) -> Self {
        let mut peak_indices =
            HashSet::with_capacity_and_hasher(fit.experimental.len(), FnvBuildHasher::default());
        peak_indices.extend(fit.experimental.iter().copied());
        Self {
            key,
            score: fit.score,
            overlap_edges: HashSet::default(),
            edges: HashSet::default(),
            peak_indices,
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

    pub fn peak_iter(&self) -> Iter<PeakKey> {
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
impl Hash for FitNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitNode {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitNode {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.score.partial_cmp(&other.score) {
            Some(ord) => match ord {
                cmp::Ordering::Less => ord,
                cmp::Ordering::Equal => self.peak_indices.len().cmp(&other.peak_indices.len()),
                cmp::Ordering::Greater => ord,
            },
            None => panic!(
                "FitNode scores were not compare-able: {} <=> {}",
                self.score, other.score
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FitRef {
    pub key: FitKey,
    pub score: ScoreType,
    pub start: f64,
    pub end: f64,
}

impl FitRef {
    pub fn new(key: FitKey, score: ScoreType, start: f64, end: f64) -> Self {
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
impl Hash for FitRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitRef {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitRef {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

pub(crate) type FitNodeGraphInner = HashMap<FitKey, FitNode, FnvBuildHasher>;

#[derive(Debug, Default)]
pub struct FitGraph {
    pub nodes: FitNodeGraphInner,
    pub dependencies: HashMap<FitKey, IsotopicFit>,
}

impl FitGraph {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.dependencies.clear();
    }

    #[inline(always)]
    pub fn get(&self, key: &FitKey) -> Option<&FitNode> {
        self.nodes.get(key)
    }

    #[inline(always)]
    pub fn get_mut(&mut self, key: &FitKey) -> Option<&mut FitNode> {
        self.nodes.get_mut(key)
    }

    pub fn add_fit(&mut self, fit: IsotopicFit, start: f64, end: f64) -> &FitNode {
        let key = FitKey(self.nodes.len() + 1);
        let node = FitNode::from_fit(&fit, key, start, end);
        self.nodes.insert(key, node);
        self.dependencies.insert(key, fit);
        self.nodes.get(&key).unwrap()
    }

    pub fn remove(&mut self, key: FitEvictionReason) -> Option<FitNode> {
        let key = match &key {
            FitEvictionReason::Superceded(k) => k,
            FitEvictionReason::NotBestFit(k) => k,
        };
        let found = self.nodes.remove(key);
        if found.is_some() {
            self.dependencies.remove(key);
        }
        found
    }

    pub fn values(&self) -> Values<FitKey, FitNode> {
        self.nodes.values()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Put [`DependenceCluster`] next to the [`FitNode`]s that references it
    fn split_nodes(
        &mut self,
        clusters: Vec<DependenceCluster>,
    ) -> Vec<(DependenceCluster, Vec<FitNode>)> {
        let mut result = Vec::with_capacity(clusters.len());

        clusters.into_iter().for_each(|mut cluster| {
            let mut deps = Vec::new();
            let mut nodes = Vec::new();
            cluster.dependencies.into_iter().for_each(|f| {
                if let Some(node) = self.nodes.remove(&f.key) {
                    deps.push(f);
                    nodes.push(node)
                } else {
                    tracing::warn!("Failed to find a node for fit {f:?}");
                }
            });
            cluster.dependencies = deps;
            if !nodes.is_empty() {
                result.push((cluster, nodes))
            }
        });
        result
    }

    pub fn solve_subgraphs(
        &mut self,
        clusters: Vec<DependenceCluster>,
        method: SubgraphSolverMethod,
    ) -> Vec<(DependenceCluster, SubgraphSolution)> {
        let n = clusters.len();
        let clusters_and_nodes = self.split_nodes(clusters);

        let mut solutions = Vec::with_capacity(n);

        for (mut cluster, nodes) in clusters_and_nodes {
            let subgraph = SubgraphSelection::from_nodes(nodes, cluster.score_ordering);

            let (solution, mut nodes) = subgraph.solve(method);
            // Prevent dependencies that are omitted from solution from being considered
            cluster.dependencies.retain(|f| solution.contains(f));
            // Re-absorb solved sub-graph
            self.nodes.extend(nodes.drain());
            solutions.push((cluster, solution));
        }
        solutions
    }
}

impl std::ops::Index<&FitKey> for FitGraph {
    type Output = FitNode;

    fn index(&self, index: &FitKey) -> &Self::Output {
        &self.nodes[index]
    }
}

#[derive(Debug, Clone)]
pub struct FitSubgraph {
    pub cluster: DependenceCluster,
    pub fit_nodes: Vec<FitNode>,
}

#[allow(unused)]
impl FitSubgraph {
    pub fn new(cluster: DependenceCluster, fit_nodes: Vec<FitNode>) -> Self {
        Self { cluster, fit_nodes }
    }

    pub fn solve(self, method: SubgraphSolverMethod) -> SolvedSubgraph {
        let subgraph = SubgraphSelection::from_nodes(self.fit_nodes, self.cluster.score_ordering);
        let (solution, fit_nodes) = subgraph.solve(method);
        SolvedSubgraph::new(self.cluster, fit_nodes, solution)
    }
}

#[derive(Debug, Clone)]
pub struct SolvedSubgraph {
    pub cluster: DependenceCluster,
    pub fit_nodes: HashMap<FitKey, FitNode, FnvBuildHasher>,
    pub solution: Vec<FitRef>,
}

impl SolvedSubgraph {
    pub fn new(
        cluster: DependenceCluster,
        fit_nodes: HashMap<FitKey, FitNode, FnvBuildHasher>,
        solution: Vec<FitRef>,
    ) -> Self {
        Self {
            cluster,
            fit_nodes,
            solution,
        }
    }
}
