use crate::scorer::ScoreInterpretation;

use super::fit::{FitNode, FitNodeGraphInner, FitRef};

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
        self.dependencies.iter().position(|f| f == fit).is_some()
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
        let mut node_map = FitNodeGraphInner::with_capacity(nodes.len());
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
        let nodes: Vec<_> = self.nodes.values_mut().collect();
        let iterator = NodeCombinationIterMut::new(nodes);
        for (node1, node2) in iterator {
            node1.visit(node2);
        }
    }

    pub fn greedy(&self) -> Vec<FitRef> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        let mut layers: Vec<Vec<&FitNode>> = Vec::new();

        layers.push(Vec::new());
        match self.score_ordering {
            ScoreInterpretation::HigherIsBetter => {
                nodes.sort_by(|a, b| a.cmp(&b).reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                nodes.sort_by(|a, b| a.cmp(&b));
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
                if !collision {
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
            .map(|layer| layer.first())
            .flatten()
            .map(|node| node.create_ref())
            .collect()
    }

    pub fn solve(self, method: SubgraphSolverMethod) -> (Vec<FitRef>, FitNodeGraphInner) {
        match method {
            SubgraphSolverMethod::Greedy => (self.greedy(), self.nodes),
        }
    }
}

#[derive(Debug)]
struct NodeCombinationIterMut<'a> {
    pub nodes: Vec<&'a mut FitNode>,
    i: usize,
    j: usize,
    pub n: usize,
}

impl<'a> NodeCombinationIterMut<'a> {
    pub fn new(nodes: Vec<&'a mut FitNode>) -> Self {
        let n = nodes.len();
        Self {
            nodes,
            i: 0,
            j: 0,
            n,
        }
    }

    pub fn advance(&mut self) -> Option<(&'a mut FitNode, &'a mut FitNode)> {
        let n = self.n;
        if self.i >= n {
            return None;
        }
        if self.j >= n {
            self.i = self.j;
            self.j = self.i + 1;
        }
        if self.i >= n {
            None
        } else {
            assert_ne!(self.i, self.j);
            unsafe {
                let n1 = *self.nodes.get_mut(self.i).unwrap() as *mut FitNode;
                let n2 = *self.nodes.get_mut(self.i).unwrap() as *mut FitNode;
                self.j += 1;
                Some((&mut *n1, &mut *n2))
            }
        }
    }
}

impl<'a> Iterator for NodeCombinationIterMut<'a> {
    type Item = (&'a mut FitNode, &'a mut FitNode);

    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }
}
