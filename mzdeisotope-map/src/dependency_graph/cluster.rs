use identity_hash::BuildIdentityHasher;
use itertools::Itertools;

use mzdeisotope::scorer::ScoreInterpretation;

// use super::fit::{FitKey, FitNode, FitNodeGraphInner, FitRef};
use super::fit::{FitKey, FitNode, FitNodeGraphInner, FitRef};
use crate::feature_fit::MapCoordinate;

pub type SubgraphSolution = Vec<FitRef>;

#[derive(Debug, Clone)]
pub struct DependenceCluster {
    pub dependencies: Vec<FitRef>,
    pub score_ordering: ScoreInterpretation,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}


impl DependenceCluster {
    pub fn new(
        dependencies: Vec<FitRef>,
        score_ordering: ScoreInterpretation,
    ) -> DependenceCluster {
        let mut cluster = DependenceCluster {
            dependencies,
            score_ordering,
            start: MapCoordinate::default(),
            end: MapCoordinate::default(),
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

    pub fn bounds(&self) -> (MapCoordinate, MapCoordinate) {
        let mut start = MapCoordinate::new(f64::INFINITY, f64::INFINITY);
        let mut end = MapCoordinate::new(-f64::INFINITY, -f64::INFINITY);

        for f in self.dependencies.iter() {
            if start.coord > f.start.coord {
                start.coord = f.start.coord;
            }
            if start.time > f.start.time {
                start.time = f.start.time;
            }

            if end.coord < f.end.coord {
                end.coord = f.end.coord;
            }
            if end.time < f.end.time {
                end.time = f.end.time;
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
        (self.start, self.end) = self.bounds();
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
                    if node.intersects(member) {
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
        let node_refs = layers
            .iter()
            .filter_map(|layer| layer.first())
            .map(|node| node.create_ref())
            .collect();

        node_refs
    }

    pub fn solve(mut self, method: SubgraphSolverMethod) -> (SubgraphSolution, FitNodeGraphInner) {
        self.build_edges();
        match method {
            SubgraphSolverMethod::Greedy => (self.greedy(), self.nodes),
        }
    }
}
