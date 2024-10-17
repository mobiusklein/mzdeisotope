use identity_hash::BuildIdentityHasher;

use crate::scorer::ScoreInterpretation;
use super::fit::{FitKey, FitNode, FitNodeGraphInner, FitRef};

pub type SubgraphSolution = Vec<FitRef>;

/// A collection of inter-dependent isotopic fits
#[derive(Debug, Clone)]
pub struct DependenceCluster {
    /// A collection of isotopic fit references stored in a [`PeakDependenceGraph`](crate::peak_graph::PeakDependenceGraph)
    pub dependencies: Vec<FitRef>,
    /// How to sort fits by score
    pub score_ordering: ScoreInterpretation,
    /// The minimum m/z of the isotopic peaks in the cluster
    pub start: f64,
    /// The maximum m/z of the isotopic peaks in the cluster
    pub end: f64,
}

impl DependenceCluster {
    pub fn new(
        dependencies: Vec<FitRef>,
        score_ordering: ScoreInterpretation,
    ) -> DependenceCluster {
        let mut cluster = DependenceCluster {
            dependencies,
            score_ordering,
            start: 0.0,
            end: 0.0,
        };
        cluster.reset();
        cluster
    }

    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    pub fn contains(&self, fit: &FitRef) -> bool {
        self.dependencies.iter().any(|f| f == fit)
    }

    fn mz_bounds(&self) -> (f64, f64) {
        let mut start = f64::INFINITY;
        let mut end = -f64::INFINITY;
        for f in self.dependencies.iter() {
            if start > f.start {
                start = f.start;
            }
            if end < f.end {
                end = f.end;
            }
        }
        (start, end)
    }

    fn reset(&mut self) {
        match self.score_ordering {
            ScoreInterpretation::HigherIsBetter => {
                self.dependencies
                    .sort_by(|a, b| a.score.total_cmp(&b.score).reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                self.dependencies
                    .sort_by(|a, b| a.score.total_cmp(&b.score));
            }
        }
        let bounds = self.mz_bounds();
        self.start = bounds.0;
        self.end = bounds.1;
    }

    pub fn best_fit(&self) -> Option<&FitRef> {
        self.dependencies.first()
    }

    pub fn add(&mut self, fit: FitRef) {
        self.dependencies.push(fit);
        self.reset()
    }

    pub fn iter(&self) -> std::slice::Iter<FitRef> {
        self.dependencies.iter()
    }
}

impl std::ops::Index<usize> for DependenceCluster {
    type Output = FitRef;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dependencies[index]
    }
}

impl IntoIterator for DependenceCluster {
    type Item = FitRef;

    type IntoIter = <std::vec::Vec<FitRef> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.dependencies.into_iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgraphSolverMethod {
    Greedy,
}

#[derive(Debug)]
pub struct SubgraphSelection {
    pub nodes: FitNodeGraphInner,
    pub score_ordering: ScoreInterpretation,
}

impl SubgraphSelection {
    pub fn from_nodes(nodes: Vec<FitNode>, score_ordering: ScoreInterpretation) -> Self {
        let mut node_map =
            FitNodeGraphInner::with_capacity_and_hasher(nodes.len(), BuildIdentityHasher::<FitKey>::default());
        for node in nodes {
            node_map.insert(node.key, node);
        }
        Self::new(node_map, score_ordering)
    }

    pub fn new(nodes: FitNodeGraphInner, score_ordering: ScoreInterpretation) -> Self {
        Self {
            nodes,
            score_ordering,
        }
    }

    pub fn build_edges(&mut self) {
        let mut nodes: Vec<&mut FitNode> = self.nodes.values_mut().collect();

        let n = nodes.len();
        let nodes = &mut nodes[0..n];

        for i in 0..n {
            for j in (i + 1)..n {
                unsafe {
                    let node_i = nodes.get_unchecked_mut(i) as *mut &mut FitNode;
                    let node_j = nodes.get_unchecked_mut(j) as *mut &mut FitNode;
                    (*node_i).visit(*node_j);
                }
            }
        }
    }

    pub fn greedy(&self) -> SubgraphSolution {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        let mut layers: Vec<Vec<&FitNode>> = Vec::new();

        layers.push(Vec::new());
        match self.score_ordering {
            ScoreInterpretation::HigherIsBetter => {
                nodes.sort_by(|a, b| a.cmp(b).reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                nodes.sort();
            }
        }

        for node in nodes {
            let mut placed = false;
            for layer in layers.iter_mut() {
                let mut collision = false;
                for member in layer.iter() {
                    if node.overlaps(member) {
                        collision = true;
                        break;
                    }
                }
                if collision {
                    layer.push(node);
                    placed = true;
                    break;
                }
            }
            if !placed {
                layers.push(vec![node]);
            }
        }
        layers
            .iter()
            .filter_map(|layer| layer.first())
            .map(|node| node.create_ref())
            .collect()
    }

    pub fn solve(mut self, method: SubgraphSolverMethod) -> (SubgraphSolution, FitNodeGraphInner) {
        self.build_edges();
        match method {
            SubgraphSolverMethod::Greedy => (self.greedy(), self.nodes),
        }
    }
}
