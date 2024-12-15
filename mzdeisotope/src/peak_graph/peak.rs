use std::collections::hash_map::{Entry, Keys};
use std::collections::HashMap;

use identity_hash::BuildIdentityHasher;

use crate::peaks::PeakKey;
use crate::scorer::ScoreType;

use super::fit::FitKey;

#[derive(Debug)]
pub struct PeakNode {
    #[allow(unused)]
    pub key: PeakKey,
    pub links: HashMap<FitKey, ScoreType, BuildIdentityHasher<usize>>,
}

impl PeakNode {
    pub fn new(key: PeakKey) -> Self {
        Self {
            key,
            links: HashMap::default(),
        }
    }

    #[allow(unused)]
    pub fn contains(&self, fit: &FitKey) -> bool {
        self.links.contains_key(fit)
    }

    pub fn remove(&mut self, fit: &FitKey) -> Option<f32> {
        self.links.remove(fit)
    }
}


#[derive(Debug, Default)]
pub(crate) struct PeakGraph {
    pub peak_nodes: HashMap<PeakKey, PeakNode, BuildIdentityHasher<PeakKey>>,
}

#[allow(unused)]
impl PeakGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            peak_nodes: HashMap::with_capacity_and_hasher(capacity, Default::default())
        }
    }

    pub fn reset(&mut self) {
        self.peak_nodes.clear();
    }

    pub fn get(&self, key: &PeakKey) -> Option<&PeakNode> {
        self.peak_nodes.get(key)
    }

    pub fn get_or_create_mut(&mut self, key: PeakKey) -> &mut PeakNode {
        match self.peak_nodes.entry(key) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(PeakNode::new(key)),
        }
    }

    pub fn keys(&self) -> Keys<PeakKey, PeakNode> {
        self.peak_nodes.keys()
    }

    pub fn drop_fit_dependence<'a, I: Iterator<Item = &'a PeakKey>>(
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
