use std::collections::{HashMap, HashSet};
use std::mem;

use crate::isotopic_fit::IsotopicFit;
use crate::peaks::PeakKey;
use crate::scorer::ScoreInterpretation;

use super::cluster::{DependenceCluster, SubgraphSolverMethod};
use super::fit::{FitGraph, FitKey, FitRef};
use super::peak::PeakGraph;

#[derive(Debug)]
pub struct PeakDependenceGraph {
    pub peak_nodes: PeakGraph,
    pub fit_nodes: FitGraph,
    pub score_ordering: ScoreInterpretation,
    pub clusters: Vec<DependenceCluster>,
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

    pub fn add_peak(&mut self, key: PeakKey) {
        self.peak_nodes.add_peak(key)
    }

    pub fn add_fit(&mut self, fit: IsotopicFit, start: f64, end: f64) {
        let node = self.fit_nodes.add_fit(fit, start, end);
        for key in node.peak_iter() {
            let pn = self.peak_nodes.get_mut(key).unwrap();
            pn.links.insert(node.key, node.score);
        }
    }

    pub fn select_best_exact_fits(&mut self) {
        let mut by_peaks: HashMap<Vec<PeakKey>, Vec<FitRef>> = HashMap::new();
        let ordering = self.score_ordering;
        for fit_node in self.fit_nodes.values() {
            let key = self.fit_nodes.dependencies[&fit_node.key]
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
                        let fit = &self.fit_nodes[&f.key];
                        self.peak_nodes.drop_fit_dependence(fit.peak_iter(), &f.key);
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
                        self.peak_nodes
                            .drop_fit_dependence(self.fit_nodes[&f.key].peak_iter(), &f.key);
                    }
                    best_fits.insert(best.key, self.fit_nodes.remove(&best.key).unwrap());
                }
            }
        }
        self.fit_nodes.replace_nodes(best_fits);
    }

    pub fn drop_superceded_fits(&mut self) {
        let mut suppressed = Vec::new();
        let mut kept = Vec::new();
        let score_ordering = self.score_ordering;
        for (key, fit) in self.fit_nodes.dependencies.values().enumerate() {
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
                if self.fit_nodes.dependencies[candidate_key].charge == fit.charge {
                    // Is this search really necessary?
                    match self.fit_nodes.dependencies[candidate_key]
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
            self.peak_nodes
                .drop_fit_dependence(fit_node.peak_iter(), &fit_node.key);
        }
    }

    fn gather_independent_clusters(&mut self) -> Vec<HashSet<FitKey>> {
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

            // Get all the `FitKey` referenced by this cluster of dependencies
            let deps: Vec<FitKey> = dependencies.iter().copied().collect();
            for dep_key in deps {
                let fit_node = self.fit_nodes.get(&dep_key).unwrap();
                for peak_key in fit_node.peak_iter().filter(|p| p.is_matched()) {
                    if let Some(key) = clusters.get(peak_key) {
                        let members = dependency_history.get(key).unwrap();
                        dependencies.extend(members.iter());
                    }
                }
            }

            // The generation that holds this revised dependency group
            let key = dependency_history.len();
            dependency_history.insert(key, dependencies);

            for dep_key in dependency_history.get(&key).unwrap().iter() {
                let fit_node = self.fit_nodes.get(&dep_key).unwrap();
                for peak_key in fit_node.peak_iter().filter(|p| p.is_matched()) {
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

    pub fn solutions(&mut self, method: SubgraphSolverMethod) -> Vec<(DependenceCluster, Vec<(FitRef, IsotopicFit)>)> {
        self.find_non_overlapping_intervals();
        let clusters = mem::take(&mut self.clusters);
        let solutions = self.fit_nodes.solve_subgraphs(clusters, method);
        // let total_size: usize = solutions.iter().map(|(_, f)| f.len()).sum();
        // let mut accepted_fits = Vec::with_capacity(total_size);
        let accepted_fits: Vec<_> = solutions.into_iter().map(|(c, sols)| {

            let fits_of: Vec<_> = sols.into_iter().map(|fit_ref| {
                match self.fit_nodes.dependencies.remove(&fit_ref.key) {
                    Some(fit) => {
                        return (fit_ref, fit)
                    },
                    None => {
                        panic!("Failed to locate fit for {:?}", fit_ref);
                    }
                }
            }).collect();
            (c, fits_of)
        }).collect();
        accepted_fits
    }
}
