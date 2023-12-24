use std::collections::hash_map::{Iter, IterMut, Keys, Entry};
use std::collections::HashMap;

use crate::peaks::PeakKey;
use crate::scorer::ScoreType;

use super::fit::FitKey;

#[derive(Debug)]
pub struct PeakNode {
    pub key: PeakKey,
    pub links: HashMap<FitKey, ScoreType>,
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

    pub fn remove(&mut self, fit: &FitKey) -> Option<f32> {
        self.links.remove(fit)
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

#[derive(Debug, Default)]
pub struct PeakGraph {
    pub peak_nodes: HashMap<PeakKey, PeakNode>,
}

impl PeakGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.peak_nodes.clear();
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

    pub fn get_or_create_mute(&mut self, key: PeakKey) -> &mut PeakNode {
        match self.peak_nodes.entry(key) {
            Entry::Occupied(o) => {
                o.into_mut()
            },
            Entry::Vacant(v) => {
                v.insert(PeakNode::new(key))
            },
        }
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

    pub fn drop_fit_dependence<'a, I: Iterator<Item = &'a PeakKey>>(
        &mut self,
        peak_iter: I,
        fit_key: &FitKey,
    ) {
        for p in peak_iter {
            if let Some(p) = self.peak_nodes.get_mut(p) {
                p.remove(fit_key);
            }
        }
    }
}
