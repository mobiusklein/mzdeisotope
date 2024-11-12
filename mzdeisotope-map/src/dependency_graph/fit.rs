use std::{
    cmp,
    collections::{hash_map::Values, hash_set::Iter, HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use identity_hash::{BuildIdentityHasher, IdentityHashable};
use mzdeisotope::scorer::ScoreType;
use mzpeaks::coordinate::Span2D;

use crate::{FeatureSetFit, MapCoordinate};

use super::{
    cluster::{DependenceCluster, SubgraphSelection, SubgraphSolution, SubgraphSolverMethod},
    feature::FeatureKey,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FitKey(usize);

impl Hash for FitKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.0);
    }
}

impl IdentityHashable for FitKey {}

pub type BuildIdentityHasherFitKey = BuildIdentityHasher<FitKey>;

impl Display for FitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct FitNode {
    pub key: FitKey,
    pub edges: HashSet<FitKey, BuildIdentityHasherFitKey>,
    pub feature_indices: HashSet<FeatureKey, BuildIdentityHasher<usize>>,
    pub score: ScoreType,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}

impl Span2D for FitNode {
    type DimType1 = f64;

    type DimType2 = f64;

    fn start(&self) -> (Self::DimType1, Self::DimType2) {
        (self.start.coord, self.start.time)
    }

    fn end(&self) -> (Self::DimType1, Self::DimType2) {
        (self.end.coord, self.end.time)
    }
}

impl FitNode {
    pub fn from_fit(fit: &FeatureSetFit, key: FitKey) -> Self {
        let start = fit.start;
        let end = fit.end;

        let mut feature_indices = HashSet::with_capacity_and_hasher(
            fit.features.len(),
            BuildIdentityHasher::<_>::default(),
        );
        feature_indices.extend(
            fit.features
                .iter()
                .flatten()
                .copied()
                .map(|i| FeatureKey(i)),
        );
        Self {
            key,
            score: fit.score,
            edges: HashSet::default(),
            feature_indices,
            start,
            end,
        }
    }

    pub fn is_disjoint(&self, other: &FitNode) -> bool {
        // Can't use coordinate overlaps here, bounding box isn't precise
        self.feature_indices.is_disjoint(&other.feature_indices)
    }

    pub fn intersects(&self, other: &FitNode) -> bool {
        !self.is_disjoint(other)
    }

    pub fn visit(&mut self, other: &mut FitNode) {
        if self.is_disjoint(other) {
            self.edges.insert(other.key);
            other.edges.insert(self.key);
        }
    }

    pub fn feature_iter(&self) -> Iter<FeatureKey> {
        self.feature_indices.iter()
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
                cmp::Ordering::Equal => {
                    self.feature_indices.len().cmp(&other.feature_indices.len())
                }
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
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}

impl FitRef {
    pub fn new(key: FitKey, score: ScoreType, start: MapCoordinate, end: MapCoordinate) -> Self {
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

#[derive(Debug, Clone, Copy)]
pub enum FitEvictionReason {
    Superceded(FitKey),
    NotBestFit(FitKey),
}

pub(crate) type FitNodeGraphInner = HashMap<FitKey, FitNode, BuildIdentityHasherFitKey>;

#[derive(Debug, Default)]
pub struct FitGraph {
    pub nodes: FitNodeGraphInner,
    pub dependencies: HashMap<FitKey, FeatureSetFit>,
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

    pub fn add_fit(&mut self, fit: FeatureSetFit) -> &FitNode {
        let key = FitKey(self.nodes.len() + 1);
        let node = FitNode::from_fit(&fit, key);
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
    pub fit_nodes: HashMap<FitKey, FitNode, BuildIdentityHasherFitKey>,
    pub solution: Vec<FitRef>,
}

impl SolvedSubgraph {
    pub fn new(
        cluster: DependenceCluster,
        fit_nodes: HashMap<FitKey, FitNode, BuildIdentityHasherFitKey>,
        solution: Vec<FitRef>,
    ) -> Self {
        Self {
            cluster,
            fit_nodes,
            solution,
        }
    }
}
