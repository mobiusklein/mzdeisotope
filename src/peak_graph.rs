use std::collections::hash_map::{Iter, IterMut, Keys};
use std::collections::{HashMap, HashSet};

use crate::isotopic_fit::IsotopicFit;
use crate::peaks::PeakKey;
use crate::scorer::ScoreInterpretation;

pub type FitKey = usize;

#[derive(Debug)]
pub struct PeakNode {
    pub key: PeakKey,
    pub links: HashMap<FitKey, f64>,
}

impl PeakNode {
    pub fn new(key: PeakKey) -> Self {
        Self {
            key,
            links: HashMap::new(),
        }
    }
    pub fn contains(&self, fit: &FitKey) -> bool {
        self.links.contains_key(fit)
    }
}

impl PartialEq for PeakNode {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for PeakNode {}
impl std::hash::Hash for PeakNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

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

#[derive(Debug)]
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

#[derive(Debug, Default)]
pub struct PeakGraph {
    pub peak_nodes: HashMap<PeakKey, PeakNode>,
}

impl PeakGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_peak(&mut self, key: PeakKey) {
        self.peak_nodes
            .entry(key)
            .or_insert_with(|| PeakNode::new(key));
    }

    pub fn get(&self, key: &PeakKey) -> Option<&PeakNode> {
        self.peak_nodes.get(key)
    }

    pub fn get_mut(&mut self, key: &PeakKey) -> Option<&mut PeakNode> {
        self.peak_nodes.get_mut(key)
    }

    pub fn len(&self) -> usize {
        self.peak_nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peak_nodes.is_empty()
    }

    pub fn keys(&self) -> Keys<PeakKey, PeakNode> {
        self.peak_nodes.keys()
    }

    pub fn iter(&self) -> Iter<PeakKey, PeakNode> {
        self.peak_nodes.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<PeakKey, PeakNode> {
        self.peak_nodes.iter_mut()
    }

    pub fn drop_fit_dependence(&mut self, fit: &FitNode) {
        for p in fit.peak_indices.iter() {
            match self.peak_nodes.get_mut(p) {
                Some(p) => {
                    p.links.remove(&fit.key);
                }
                None => {}
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct FitGraph {
    pub fit_nodes: HashMap<FitKey, FitNode>,
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

    pub fn add_fit(&mut self, node: FitNode) {
        self.fit_nodes.insert(node.key, node);
    }

    pub fn len(&self) -> usize {
        self.fit_nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fit_nodes.is_empty()
    }

    pub fn arrange_layers(
        &self,
        mut envelopes: Vec<FitRef>,
        ordering: ScoreInterpretation,
    ) -> Vec<Vec<FitRef>> {
        let mut layers: Vec<Vec<FitRef>> = Vec::new();
        layers.push(Vec::new());
        match ordering {
            ScoreInterpretation::HigherIsBetter => {
                envelopes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
            }
            ScoreInterpretation::LowerIsBetter => {
                envelopes.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            }
        }
        for envelope in envelopes {
            let mut placed = false;
            for layer in layers.iter_mut() {
                let mut collision = false;
                for member in layer.iter() {
                    if self.fit_nodes[&member.key].overlaps(&self.fit_nodes[&envelope.key]) {
                        collision = true;
                        break;
                    }
                }
                if !collision {
                    layer.push(envelope);
                    placed = true;
                    break;
                }
            }
            if !placed {
                layers.push(vec![envelope]);
            }
        }
        layers
    }
}

#[derive(Debug, Clone)]
pub struct SubgraphSelection {
    pub layers: Vec<Vec<FitRef>>,
    pub score_ordering: ScoreInterpretation,
}

impl SubgraphSelection {}

#[derive(Debug)]
pub struct PeakDependenceGraph {
    pub peak_nodes: PeakGraph,
    pub dependencies: Vec<IsotopicFit>,
    pub fit_nodes: HashMap<FitKey, FitNode>,
    pub score_ordering: ScoreInterpretation,
    pub clusters: Vec<DependenceCluster>,
}

impl PeakDependenceGraph {
    pub fn new(score_ordering: ScoreInterpretation) -> Self {
        Self {
            score_ordering,
            peak_nodes: PeakGraph::new(),
            fit_nodes: HashMap::new(),
            dependencies: Vec::new(),
            clusters: Vec::new(),
        }
    }

    pub fn add_peak(&mut self, key: PeakKey) {
        self.peak_nodes.add_peak(key)
    }

    pub fn add_fit(&mut self, fit: IsotopicFit, start: f64, end: f64) {
        let key = self.fit_nodes.len();
        let node = FitNode::from_fit(&fit, key, start, end);
        for key in node.peak_indices.iter() {
            let pn = self.peak_nodes.get_mut(key).unwrap();
            pn.links.insert(node.key, node.score);
        }
        self.fit_nodes.insert(key, node);
        self.dependencies.push(fit)
    }

    pub fn nodes_for(
        &self,
        fit: &FitNode,
        cache: Option<&mut HashMap<FitKey, Vec<PeakKey>>>,
    ) -> Vec<PeakKey> {
        match cache {
            Some(cache) => {
                let node_keys = cache
                    .entry(fit.key)
                    .or_insert_with(|| self.nodes_for(fit, None));
                let result = node_keys.iter().copied().collect();
                result
            }
            None => {
                let fit = &self.dependencies[fit.key];
                let mut result =
                    Vec::with_capacity(fit.experimental.len() - fit.missed_peaks as usize);
                for p in fit.experimental.iter() {
                    match p {
                        PeakKey::Matched(_k) => {
                            result.push(*p);
                        }
                        _ => {}
                    }
                }
                result
            }
        }
    }

    pub fn drop_fit_dependence(&mut self, fit: &FitNode) {
        for p in fit.peak_indices.iter() {
            match self.peak_nodes.get_mut(p) {
                Some(p) => {
                    p.links.remove(&fit.key);
                }
                None => {}
            }
        }
    }

    pub fn select_best_exact_fits(&mut self) {
        let mut by_peaks: HashMap<Vec<PeakKey>, Vec<FitRef>> = HashMap::new();
        let ordering = self.score_ordering;
        for fit_node in self.fit_nodes.values() {
            let key = self.dependencies[fit_node.key]
                .experimental
                .iter()
                .filter(|p| p.is_matched())
                .copied()
                .collect();
            by_peaks.entry(key).or_default().push(fit_node.create_ref());
        }
        let mut best_fits = HashMap::with_capacity(by_peaks.len());
        match ordering {
            ScoreInterpretation::HigherIsBetter => {
                for (_key, mut bucket) in by_peaks.drain() {
                    if bucket.len() == 1 {
                        best_fits.insert(
                            bucket[0].key,
                            self.fit_nodes.remove(&bucket[0].key).unwrap(),
                        );
                        continue;
                    }
                    bucket.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
                    let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        self.peak_nodes.drop_fit_dependence(&self.fit_nodes[&f.key]);
                    }
                    best_fits.insert(best.key, self.fit_nodes.remove(&best.key).unwrap());
                }
            }
            ScoreInterpretation::LowerIsBetter => {
                for (_key, mut bucket) in by_peaks.drain() {
                    if bucket.len() == 1 {
                        best_fits.insert(
                            bucket[0].key,
                            self.fit_nodes.remove(&bucket[0].key).unwrap(),
                        );
                        continue;
                    }
                    bucket.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().reverse());
                    let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        self.peak_nodes.drop_fit_dependence(&self.fit_nodes[&f.key]);
                    }
                    best_fits.insert(best.key, self.fit_nodes.remove(&best.key).unwrap());
                }
            }
        }

        self.fit_nodes = best_fits;
    }

    pub fn drop_superceded_fits(&mut self) {
        let mut suppressed = Vec::new();
        let mut kept = Vec::new();
        let score_ordering = self.score_ordering;
        for (key, fit) in self.dependencies.iter().enumerate() {
            if fit.is_empty() {
                suppressed.push(key);
                continue;
            }
            let mono = fit.experimental.first().unwrap();
            if mono.is_placeholder() {
                continue;
            }
            let mono_peak_node = self.peak_nodes.get(mono).unwrap();
            let mut suppress = false;
            for (candidate_key, score) in mono_peak_node.links.iter() {
                if self.dependencies[*candidate_key].charge == fit.charge {
                    // Is this search really necessary?
                    match self.dependencies[*candidate_key]
                        .experimental
                        .iter()
                        .position(|k| k == mono)
                    {
                        Some(_) => match score_ordering {
                            ScoreInterpretation::HigherIsBetter => {
                                if fit.score < *score {
                                    suppress = true;
                                    break;
                                }
                            }
                            ScoreInterpretation::LowerIsBetter => {
                                if fit.score > *score {
                                    suppress = true;
                                    break;
                                }
                            }
                        },
                        None => {}
                    }
                }
            }
            if suppress {
                suppressed.push(key);
            } else {
                kept.push(key);
            }
        }
        for drop in suppressed {
            let fit_node = self.fit_nodes.remove(&drop).unwrap();
            self.drop_fit_dependence(&fit_node);
        }
    }

    fn gather_independent_clusters(&mut self) -> Vec<HashSet<FitKey>> {
        // A cache for `nodes_for`
        let mut node_cache: HashMap<FitKey, Vec<PeakKey>> = HashMap::new();
        // Map peak key to the set of other keys that share fits over them, indirectly
        // pointing into `dependency_history`.
        let mut clusters: HashMap<PeakKey, usize> = HashMap::new();

        // A running log of previous dependency sets.
        let mut dependency_history: HashMap<usize, HashSet<FitKey>> = HashMap::new();

        for (_key, seed_node) in self.peak_nodes.iter() {
            let mut dependencies = HashSet::new();
            dependencies.extend(seed_node.links.keys().copied());

            if dependencies.is_empty() {
                continue;
            }

            let deps: Vec<usize> = dependencies.iter().copied().collect();
            for dep_key in deps {
                let fit_node = self.fit_nodes.get(&dep_key).unwrap();
                for peak_key in self.nodes_for(fit_node, Some(&mut node_cache)).iter() {
                    if let Some(key) = clusters.get(peak_key) {
                        let members = dependency_history.get(key).unwrap();
                        dependencies.extend(members.iter());
                    }
                }
            }

            let key = dependency_history.len();
            dependency_history.insert(key, dependencies);
            for dep_key in dependency_history.get(&key).unwrap().iter() {
                let fit_node = self.fit_nodes.get(&dep_key).unwrap();
                for peak_key in self.nodes_for(fit_node, Some(&mut node_cache)).iter() {
                    clusters.insert(*peak_key, key);
                }
            }
        }
        let mut result = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for val in clusters.into_values() {
            if seen.contains(&val) {
                continue;
            } else {
                seen.insert(val);
                result.push(dependency_history.remove(&val).unwrap());
            }
        }
        result
    }

    pub fn find_non_overlapping_intervals(&mut self) {
        self.select_best_exact_fits();
        self.drop_superceded_fits();
        let clusters = self.gather_independent_clusters();
        for keys in clusters {
            let mut fits = Vec::new();
            for k in keys {
                fits.push(self.fit_nodes[&k].create_ref());
            }
            let cluster = DependenceCluster::new(fits, self.score_ordering);
            self.clusters.push(cluster);
        }
        self.clusters
            .sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    }
}
