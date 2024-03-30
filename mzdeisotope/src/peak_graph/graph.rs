use std::collections::{HashMap, HashSet};
use std::mem;

use fnv::FnvBuildHasher;

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
                        .position(|k| k == mono)
                        .is_some()
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

    fn gather_independent_clusters(&mut self) -> Vec<HashSet<FitKey, FnvBuildHasher>> {
        // Map peak key to the set of other keys that share fits over them, indirectly
        // pointing into `dependency_history`.
        let mut clusters: HashMap<PeakKey, usize, FnvBuildHasher> =
            HashMap::with_capacity_and_hasher(self.peak_nodes.len() / 4, Default::default());

        // A running log of previous dependency sets.
        let mut dependency_history: HashMap<
            usize,
            HashSet<FitKey, FnvBuildHasher>,
            FnvBuildHasher,
        > = HashMap::default();

        for (_peak_key, seed_node) in self.peak_nodes.iter() {
            if seed_node.links.is_empty() {
                continue;
            }

            let mut dependencies: HashSet<FitKey, FnvBuildHasher> =
                HashSet::with_capacity_and_hasher(
                    seed_node.links.len() * 2,
                    FnvBuildHasher::default(),
                );
            dependencies.extend(seed_node.links.keys().copied());

            // Get all the `FitKey` referenced by this cluster of dependencies
            for dep_key in seed_node.links.keys() {
                let fit_node = self.fit_nodes.get(dep_key).unwrap();
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
                let fit_node = self.fit_nodes.get(dep_key).unwrap();
                for peak_key in fit_node.peak_iter().filter(|p| p.is_matched()) {
                    clusters.insert(*peak_key, key);
                }
            }
        }
        let mut result = Vec::with_capacity(clusters.len() / 3);
        let mut seen: HashSet<usize, FnvBuildHasher> =
            HashSet::with_capacity_and_hasher(clusters.len(), Default::default());
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
