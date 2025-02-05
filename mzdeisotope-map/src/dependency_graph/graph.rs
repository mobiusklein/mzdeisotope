use std::collections::{HashMap, HashSet, VecDeque};
use std::mem;

use identity_hash::BuildIdentityHasher;

use crate::feature_fit::FeatureSetFit;
use super::feature::FeatureKey;
use mzdeisotope::scorer::ScoreInterpretation;

use super::cluster::{DependenceCluster, SubgraphSolverMethod};
use super::fit::{FitEvictionReason, FitGraph, FitKey, FitRef};
use super::feature::FeatureGraph;

#[derive(Debug)]
pub struct FeatureDependenceGraph {
    pub feature_nodes: FeatureGraph,
    pub fit_nodes: FitGraph,
    pub score_ordering: ScoreInterpretation,
    pub clusters: Vec<DependenceCluster>,
}

impl FeatureDependenceGraph {
    pub fn new(score_ordering: ScoreInterpretation) -> Self {
        Self {
            score_ordering,
            feature_nodes: FeatureGraph::new(),
            fit_nodes: FitGraph::new(),
            clusters: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.feature_nodes.reset();
        self.fit_nodes.reset();
        self.clusters.clear();
    }

    pub fn add_feature(&mut self, key: FeatureKey) {
        self.feature_nodes.add_feature(key)
    }

    pub fn add_fit(&mut self, fit: FeatureSetFit) {
        let node = self.fit_nodes.add_fit(fit);
        for key in node.feature_iter() {
            let pn = self.feature_nodes.get_or_create_mute(*key);
            pn.links.insert(node.key, node.score);
        }
    }

    fn select_best_exact_fits(&mut self) {
        let mut by_features: HashMap<Vec<FeatureKey>, Vec<FitRef>> = HashMap::new();
        let ordering = self.score_ordering;
        self.fit_nodes.values().for_each(|fit_node| {
            let key = self.fit_nodes.dependencies[&fit_node.key]
                .features
                .iter()
                .filter(|p| p.is_some())
                .copied()
                .map(|i| FeatureKey(i.unwrap()))
                .collect();
            by_features.entry(key).or_default().push(fit_node.create_ref());
        });

        match ordering {
            ScoreInterpretation::HigherIsBetter => {
                for (_key, mut bucket) in by_features.drain() {
                    if bucket.len() == 1 {
                        continue;
                    }
                    bucket.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().reverse());
                    // let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        let fit = &self.fit_nodes[&f.key];
                        self.feature_nodes.drop_fit_dependence(fit.feature_iter(), &f.key);
                        self.fit_nodes.remove(FitEvictionReason::NotBestFit(f.key));
                    }
                }
            }
            ScoreInterpretation::LowerIsBetter => {
                for (_key, mut bucket) in by_features.drain() {
                    if bucket.len() == 1 {
                        continue;
                    }
                    bucket.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().reverse());
                    // let best = bucket[0];
                    let rest = &bucket[1..];
                    for f in rest {
                        let fit = &self.fit_nodes[&f.key];
                        self.feature_nodes.drop_fit_dependence(fit.feature_iter(), &f.key);
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
            let mono = fit.features.first().unwrap();
            if mono.is_none() {
                continue;
            }
            let mono = FeatureKey(mono.unwrap());
            let mono_peak_node = self.feature_nodes.get(&mono).unwrap();
            let mut suppress = false;

            for (candidate_key, score) in mono_peak_node.links.iter() {
                if self.fit_nodes.dependencies[candidate_key].charge == fit.charge {
                    // Is this search really necessary?
                    if self.fit_nodes.dependencies[candidate_key]
                        .features
                        .iter().flatten()
                        .any(|k| FeatureKey(*k) == mono)
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
            self.feature_nodes
                .drop_fit_dependence(fit_node.feature_iter(), &fit_node.key);
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
    ) -> Vec<(DependenceCluster, Vec<(FitRef, FeatureSetFit)>)> {
        self.find_non_overlapping_intervals();
        let clusters = mem::take(&mut self.clusters);
        let solutions = self.fit_nodes.solve_subgraphs(clusters, method);

        let accepted_fits: Vec<(DependenceCluster, Vec<(FitRef, FeatureSetFit)>)> = solutions
            .into_iter()
            .map(|(cluster, sols)| {
                let fits_of: Vec<(FitRef, FeatureSetFit)> = sols
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
                let mut features_used = HashMap::new();
                for (r, f) in fits_of.iter() {
                    for i in f.features.iter().flatten() {
                        if features_used.contains_key(i) {
                            tracing::warn!("Feature {i} was already used in {:?}, used again in {r:?}", features_used.get(i).unwrap())
                        }
                        features_used.insert(*i, r);
                    }
                }
                (cluster, fits_of)
            })
            .collect();
        accepted_fits
    }
}


struct BreadFirstTraversal<'a> {
    graph: &'a FeatureDependenceGraph,
    /// The fit node keys that have not yet been visited
    nodes: HashSet<FitKey>,
    /// A record of the peaks that have been visited already
    feature_mask: Vec<bool>,
}

impl Iterator for BreadFirstTraversal<'_> {
    type Item = HashSet<FitKey, BuildIdentityHasher<FitKey>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_component()
    }
}

impl<'a> BreadFirstTraversal<'a> {
    fn new(graph: &'a FeatureDependenceGraph) -> Self {
        let nodes = graph.fit_nodes.nodes.keys().copied().collect();

        // Create a mask of equal length to the peak list initialized with false values
        let features = if let Some(peak_count) = graph
            .feature_nodes
            .keys()
            .map(|p| p.0)
            .max()
        {
            let mut features: Vec<bool> = Vec::with_capacity(peak_count + 1);
            features.resize(peak_count + 1, false);
            features
        } else {
            Vec::new()
        };
        Self {
            graph,
            nodes,
            feature_mask: features,
        }
    }

    fn edges_from(&mut self, node: FitKey) -> HashSet<FitKey, BuildIdentityHasher<FitKey>> {
        let fit_node = self.graph.fit_nodes.get(&node).unwrap();
        let mut next_keys = HashSet::default();
        for i in fit_node.feature_iter().copied() {
            // If the peak has already been visited, skip it, otherwise
            // add it to the mask and traverse the peak node.
            if self.feature_mask[i.0] {
                continue;
            } else {
                self.feature_mask[i.0] = true;
            }
            let peak_node = self.graph.feature_nodes.get(&i).unwrap();
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
        self.nodes.iter().next().copied().map(|node| self.visit(node))
    }
}
