use std::{
    cmp,
    collections::{hash_set::Iter, HashSet},
    fmt::Display,
    hash::Hash,
};

use identity_hash::{BuildIdentityHasher, IdentityHashable};
use mzdeisotope::scorer::ScoreType;

use crate::{FeatureSetFit, MapCoordinate};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FitKey(usize);

impl Hash for FitKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.0);
    }
}

impl IdentityHashable for FitKey {}

pub type BuildIdentityHasherFitKey = BuildIdentityHasher<FitKey>;

impl Display for FitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone)]
pub struct FitNode {
    pub key: FitKey,
    pub edges: HashSet<FitKey, BuildIdentityHasherFitKey>,
    pub overlap_edges: HashSet<FitKey, BuildIdentityHasherFitKey>,
    pub feature_indices: HashSet<usize, BuildIdentityHasher<usize>>,
    pub score: ScoreType,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}

impl FitNode {
    pub fn from_fit(fit: &FeatureSetFit, key: FitKey) -> Self {
        let start = fit.start;
        let end = fit.end;

        let mut peak_indices = HashSet::with_capacity_and_hasher(
            fit.features.len(),
            BuildIdentityHasher::<_>::default(),
        );
        peak_indices.extend(fit.features.iter().flatten().copied());
        Self {
            key,
            score: fit.score,
            overlap_edges: HashSet::default(),
            edges: HashSet::default(),
            feature_indices: peak_indices,
            start,
            end,
        }
    }

    pub fn is_disjoint(&self, other: &FitNode) -> bool {
        self.feature_indices.is_disjoint(&other.feature_indices)
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

    pub fn peak_iter(&self) -> Iter<usize> {
        self.feature_indices.iter()
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
impl Hash for FitNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitNode {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitNode {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.score.partial_cmp(&other.score) {
            Some(ord) => match ord {
                cmp::Ordering::Less => ord,
                cmp::Ordering::Equal => {
                    self.feature_indices.len().cmp(&other.feature_indices.len())
                }
                cmp::Ordering::Greater => ord,
            },
            None => panic!(
                "FitNode scores were not compare-able: {} <=> {}",
                self.score, other.score
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FitRef {
    pub key: FitKey,
    pub score: ScoreType,
    pub start: MapCoordinate,
    pub end: MapCoordinate,
}

impl FitRef {
    pub fn new(key: FitKey, score: ScoreType, start: MapCoordinate, end: MapCoordinate) -> Self {
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
impl Hash for FitRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl PartialOrd for FitRef {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitRef {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}
