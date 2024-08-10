use std::collections::{HashMap, HashSet, VecDeque};
use std::mem;

use identity_hash::BuildIdentityHasher;

use crate::isotopic_fit::IsotopicFit;
use crate::peaks::PeakKey;
use crate::scorer::ScoreInterpretation;

use super::cluster::{DependenceCluster, SubgraphSolverMethod};
use super::fit::{FitGraph, FitKey, FitRef};
use super::peak::PeakGraph;


/// A graph relating experimental peaks to isotopic pattern fits, constructing two levels
/// of dependence, from peaks to isotopic fits and later isotopic fits which depend upon
/// the same experimental peaks.
#[derive(Debug)]
pub struct PeakDependenceGraph {
    peak_nodes: PeakGraph,
    fit_nodes: FitGraph,
    pub score_ordering: ScoreInterpretation,
    pub clusters: Vec<DependenceCluster>,
}

#[derive(Debug, Clone, Copy)]
pub enum FitEvictionReason {
    Superceded(FitKey),
    NotBestFit(FitKey),
}

impl PeakDependenceGraph {
    pub fn new(score_ordering: ScoreInterpretation) -> Self {
        Self {
            score_ordering,
            peak_nodes: PeakGraph::new(),
            fit_nodes: FitGraph::new(),
            clusters: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize, score_ordering: ScoreInterpretation) -> Self {
        Self {
            score_ordering,
            peak_nodes: PeakGraph::with_capacity(capacity),
            fit_nodes: FitGraph::with_capacity(capacity * 2),
            clusters: Vec::with_capacity(capacity / 2),
        }
    }

    pub fn reset(&mut self) {
        self.peak_nodes.reset();
        self.fit_nodes.reset();
        self.clusters.clear();
    }

    pub fn add_peak(&mut self, key: PeakKey) {
        self.peak_nodes.add_peak(key)
    }

    pub fn add_fit(&mut self, fit: IsotopicFit, start: f64, end: f64) {
        let node = self.fit_nodes.add_fit(fit, start, end);
        for key in node.peak_iter() {
            let pn = self.peak_nodes.get_or_create_mute(*key);
            pn.links.insert(node.key, node.score);
        }
    }

    fn select_best_exact_fits(&mut self) {
        let mut by_peaks: HashMap<Vec<PeakKey>, Vec<FitRef>> = HashMap::new();
        let ordering = self.score_ordering;
        self.fit_nodes.values().for_each(|fit_node| {
            let key = self.fit_nodes.dependencies[&fit_node.key]
                .experimental
                .iter()
                .filter(|p| p.is_matched())
                .copied()
                .collect();
            by_peaks.entry(key).or_default().push(fit_node.create_ref());
        });

        match ordering {
            ScoreInterpretation::HigherIsBetter => {
                for (_key, mut bucket) in by_peaks.drain() {
                    if bucket.len() == 1 {
                        continue;
                    }
                    bucket.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
                    // let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        let fit = &self.fit_nodes[&f.key];
                        self.peak_nodes.drop_fit_dependence(fit.peak_iter(), &f.key);
                        self.fit_nodes.remove(FitEvictionReason::NotBestFit(f.key));
                    }
                }
            }
            ScoreInterpretation::LowerIsBetter => {
                for (_key, mut bucket) in by_peaks.drain() {
                    if bucket.len() == 1 {
                        continue;
                    }
                    bucket.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().reverse());
                    // let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        let fit = &self.fit_nodes[&f.key];
                        self.peak_nodes.drop_fit_dependence(fit.peak_iter(), &f.key);
                        self.fit_nodes.remove(FitEvictionReason::NotBestFit(f.key));
                    }
                }
            }
        }
    }

    fn drop_superceded_fits(&mut self) {
        let mut suppressed: HashSet<FitKey> = HashSet::new();
        let score_ordering = self.score_ordering;
        for (key, fit) in self.fit_nodes.dependencies.iter() {
            if fit.is_empty() {
                suppressed.insert(*key);
                continue;
            }
            let mono = fit.experimental.first().unwrap();
            if mono.is_placeholder() {
                continue;
            }
            let mono_peak_node = self.peak_nodes.get(mono).unwrap();
            let mut suppress = false;

            for (candidate_key, score) in mono_peak_node.links.iter() {
                if self.fit_nodes.dependencies[candidate_key].charge == fit.charge {
                    // Is this search really necessary?
                    if self.fit_nodes.dependencies[candidate_key]
                        .experimental
                        .iter()
                        .any(|k| k == mono)
                    {
                        match score_ordering {
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
                        }
                    }
                }
            }
            if suppress {
                suppressed.insert(*key);
            }
        }

        for drop in suppressed {
            let fit_node = self
                .fit_nodes
                .remove(FitEvictionReason::Superceded(drop))
                .unwrap();
            self.peak_nodes
                .drop_fit_dependence(fit_node.peak_iter(), &fit_node.key);
        }
    }

    fn gather_independent_clusters(&mut self) -> Vec<HashSet<FitKey, BuildIdentityHasher<FitKey>>> {
        let traversal = BreadFirstTraversal::new(self);
        traversal.collect()
    }

    pub fn find_non_overlapping_intervals(&mut self) {
        self.select_best_exact_fits();
        self.drop_superceded_fits();
        let clusters = self.gather_independent_clusters();
        for keys in clusters {
            let mut fits = Vec::with_capacity(keys.len());
            for k in keys {
                fits.push(self.fit_nodes[&k].create_ref());
            }
            let cluster = DependenceCluster::new(fits, self.score_ordering);
            self.clusters.push(cluster);
        }
        self.clusters
            .sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    }

    pub fn solutions(
        &mut self,
        method: SubgraphSolverMethod,
    ) -> Vec<(DependenceCluster, Vec<(FitRef, IsotopicFit)>)> {
        self.find_non_overlapping_intervals();
        let clusters = mem::take(&mut self.clusters);
        let solutions = self.fit_nodes.solve_subgraphs(clusters, method);

        let accepted_fits: Vec<(DependenceCluster, Vec<(FitRef, IsotopicFit)>)> = solutions
            .into_iter()
            .map(|(cluster, sols)| {
                let fits_of: Vec<(FitRef, IsotopicFit)> = sols
                    .into_iter()
                    .map(
                        |fit_ref| match self.fit_nodes.dependencies.remove(&fit_ref.key) {
                            Some(fit) => {
                                debug_assert!((fit_ref.score - fit.score).abs() < 1e-6);
                                (fit_ref, fit)
                            }
                            None => {
                                panic!("Failed to locate fit for {:?}", fit_ref);
                            }
                        },
                    )
                    .collect();
                (cluster, fits_of)
            })
            .collect();
        accepted_fits
    }
}


struct BreadFirstTraversal<'a> {
    graph: &'a PeakDependenceGraph,
    /// The fit node keys that have not yet been visited
    nodes: HashSet<FitKey>,
    /// A record of the peaks that have been visited already
    peak_mask: Vec<bool>,
}

impl<'a> Iterator for BreadFirstTraversal<'a> {
    type Item = HashSet<FitKey, BuildIdentityHasher<FitKey>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_component()
    }
}

impl<'a> BreadFirstTraversal<'a> {
    fn new(graph: &'a PeakDependenceGraph) -> Self {
        let nodes = graph.fit_nodes.nodes.keys().copied().collect();

        // Create a mask of equal length to the peak list initialized with false values
        let peaks = if let Some(peak_count) = graph
            .peak_nodes
            .keys()
            .filter(|p| p.is_matched())
            .map(|p| p.to_index_unchecked() as usize)
            .max()
        {
            let mut peaks: Vec<bool> = Vec::with_capacity(peak_count + 1);
            peaks.resize(peak_count + 1, false);
            peaks
        } else {
            Vec::new()
        };
        Self {
            graph,
            nodes,
            peak_mask: peaks,
        }
    }

    fn edges_from(&mut self, node: FitKey) -> HashSet<FitKey, BuildIdentityHasher<FitKey>> {
        let fit_node = self.graph.fit_nodes.get(&node).unwrap();
        let mut next_keys = HashSet::default();
        for peak in fit_node.peak_iter() {
            match *peak {
                // If the peak has already been visited, skip it, otherwise
                // add it to the mask and traverse the peak node.
                PeakKey::Matched(i) => {
                    if self.peak_mask[i as usize] {
                        continue;
                    } else {
                        self.peak_mask[i as usize] = true;
                    }
                }
                PeakKey::Placeholder(_) => {
                    continue;
                }
            }
            let peak_node = self.graph.peak_nodes.get(peak).unwrap();
            next_keys.extend(
                peak_node
                    .links
                    .keys()
                    // Skip fit nodes that we've already visited, which means that
                    // they will have been removed from `self.nodes`.
                    .filter(|i| self.nodes.contains(i))
                    .copied(),
            );
        }
        next_keys
    }

    fn visit(&mut self, node: FitKey) -> HashSet<FitKey, BuildIdentityHasher<FitKey>> {
        let mut component: HashSet<FitKey, BuildIdentityHasher<FitKey>> = HashSet::default();

        let mut nodes = VecDeque::from(vec![node]);

        while !nodes.is_empty() {
            let node = nodes.pop_front().unwrap();
            // If we've already visited this node, it's not there to be removed
            // so we can skip it.
            if !self.nodes.remove(&node) {
                continue;
            }
            component.insert(node);
            nodes.extend(self.edges_from(node));
        }
        component
    }

    fn next_component(&mut self) -> Option<HashSet<FitKey, BuildIdentityHasher<FitKey>>> {
        if let Some(node) = self.nodes.iter().next().copied() {
            Some(self.visit(node))
        } else {
            None
        }
    }
}
