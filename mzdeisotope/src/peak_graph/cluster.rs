use crate::scorer::ScoreInterpretation;
use identity_hash::BuildIdentityHasher;
use itertools::Itertools;

use super::fit::{FitKey, FitNode, FitNodeGraphInner, FitRef};

pub type SubgraphSolution = Vec<FitRef>;

#[derive(Debug, Clone)]
pub struct DependenceCluster {
    pub dependencies: Vec<FitRef>,
    pub score_ordering: ScoreInterpretation,
    pub start: f64,
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

    pub fn mz_bounds(&self) -> (f64, f64) {
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
                    .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                self.dependencies
                    .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
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

        (0..nodes.len()).tuple_combinations().for_each(|(i, j)| {
            let mut it = nodes.iter_mut().skip(i);
            let node_i = it.next().unwrap();
            let node_j = it.nth(j - (i + 1)).unwrap();
            node_i.visit(node_j);
        });
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
