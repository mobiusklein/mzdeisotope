use std::collections::{hash_map::{Entry, Iter, IterMut, Keys}, HashMap};

use identity_hash::{BuildIdentityHasher, IdentityHashable};
use mzdeisotope::scorer::ScoreType;

use super::fit::{BuildIdentityHasherFitKey, FitKey};


#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FeatureKey(pub usize);

impl IdentityHashable for FeatureKey {}

impl std::hash::Hash for FeatureKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.0 as u64)
    }
}

impl From<usize> for FeatureKey {
    fn from(value: usize) -> Self {
        FeatureKey(value)
    }
}

#[derive(Debug)]
pub struct FeatureNode {
    pub key: FeatureKey,
    pub links: HashMap<FitKey, ScoreType, BuildIdentityHasherFitKey>,
}


impl FeatureNode {
    pub fn new<T: Into<FeatureKey>>(key: T) -> Self {
        Self {
            key: key.into(),
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
    pub nodes: HashMap<FeatureKey, FeatureNode, BuildIdentityHasher<FeatureKey>>,
}

impl FeatureGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
    }

    pub fn add_feature(&mut self, key: FeatureKey) {
        self.nodes
            .entry(key)
            .or_insert_with(|| FeatureNode::new(key));
    }

    pub fn get(&self, key: &FeatureKey) -> Option<&FeatureNode> {
        self.nodes.get(key)
    }

    pub fn get_mut(&mut self, key: &FeatureKey) -> Option<&mut FeatureNode> {
        self.nodes.get_mut(key)
    }

    pub fn get_or_create_mute(&mut self, key: FeatureKey) -> &mut FeatureNode {
        match self.nodes.entry(key) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(FeatureNode::new(key)),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn keys(&self) -> Keys<FeatureKey, FeatureNode> {
        self.nodes.keys()
    }

    pub fn iter(&self) -> Iter<FeatureKey, FeatureNode> {
        self.nodes.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<FeatureKey, FeatureNode> {
        self.nodes.iter_mut()
    }

    pub fn drop_fit_dependence<'a, I: Iterator<Item = &'a FeatureKey>>(
        &mut self,
        feat_iter: I,
        fit_key: &FitKey,
    ) {
        for p in feat_iter {
            if let Some(p) = self.nodes.get_mut(p) {
                p.remove(fit_key);
            } else {
                tracing::warn!("Failed to remove {fit_key} for {p:?}");
            }
        }
    }
}
