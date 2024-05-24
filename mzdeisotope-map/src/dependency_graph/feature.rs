use std::collections::{hash_map::{Entry, Iter, IterMut, Keys}, HashMap};

use mzdeisotope::scorer::ScoreType;

use super::fit::{BuildIdentityHasherFitKey, FitKey};



#[derive(Debug)]
pub struct FeatureNode {
    pub key: usize,
    pub links: HashMap<FitKey, ScoreType, BuildIdentityHasherFitKey>,
}


impl FeatureNode {
    pub fn new(key: usize) -> Self {
        Self {
            key,
            links: HashMap::default(),
        }
    }
    pub fn contains(&self, fit: &FitKey) -> bool {
        self.links.contains_key(fit)
    }

    pub fn remove(&mut self, fit: &FitKey) -> Option<f32> {
        self.links.remove(fit)
    }
}

impl PartialEq for FeatureNode {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for FeatureNode {}
impl std::hash::Hash for FeatureNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

#[derive(Debug, Default)]
pub struct FeatureGraph {
    pub peak_nodes: HashMap<usize, FeatureNode>,
}

impl FeatureGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.peak_nodes.clear();
    }

    pub fn add_peak(&mut self, key: usize) {
        self.peak_nodes
            .entry(key)
            .or_insert_with(|| FeatureNode::new(key));
    }

    pub fn get(&self, key: &usize) -> Option<&FeatureNode> {
        self.peak_nodes.get(key)
    }

    pub fn get_mut(&mut self, key: &usize) -> Option<&mut FeatureNode> {
        self.peak_nodes.get_mut(key)
    }

    pub fn get_or_create_mute(&mut self, key: usize) -> &mut FeatureNode {
        match self.peak_nodes.entry(key) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(FeatureNode::new(key)),
        }
    }

    pub fn len(&self) -> usize {
        self.peak_nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peak_nodes.is_empty()
    }

    pub fn keys(&self) -> Keys<usize, FeatureNode> {
        self.peak_nodes.keys()
    }

    pub fn iter(&self) -> Iter<usize, FeatureNode> {
        self.peak_nodes.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<usize, FeatureNode> {
        self.peak_nodes.iter_mut()
    }

    pub fn drop_fit_dependence<'a, I: Iterator<Item = &'a usize>>(
        &mut self,
        peak_iter: I,
        fit_key: &FitKey,
    ) {
        for p in peak_iter {
            if let Some(p) = self.peak_nodes.get_mut(p) {
                p.remove(fit_key);
            } else {
                tracing::warn!("Failed to remove {fit_key} for {p:?}");
            }
        }
    }
}
