use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::prelude::*;
use mzpeaks::{CentroidPeak, IndexType, MZPeakSetType, MassErrorType};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakKey {
    Matched(u32),
    Placeholder(i64),
}

impl PeakKey {
    pub fn is_matched(&self) -> bool {
        match self {
            PeakKey::Matched(_) => true,
            _ => false,
        }
    }

    pub fn is_placeholder(&self) -> bool {
        match self {
            PeakKey::Placeholder(_) => true,
            _ => false,
        }
    }
}

impl std::hash::Hash for PeakKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            PeakKey::Matched(i) => i.hash(state),
            PeakKey::Placeholder(i) => i.hash(state),
        }
    }
}

#[derive(Debug)]
pub struct PlaceholderCache<C: CentroidLike + Clone + From<CentroidPeak>> {
    placeholders: HashMap<i64, C>,
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> Default for PlaceholderCache<C> {
    fn default() -> Self {
        Self {
            placeholders: Default::default(),
        }
    }
}

pub trait MZCaching {
    fn key_for(&self, mz: f64) -> i64 {
        let key = (mz * 1000.0).round() as i64;
        key
    }
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> PlaceholderCache<C> {
    pub fn create(&mut self, mz: f64) -> i64 {
        let key = self.key_for(mz);
        self.placeholders
            .entry(key.clone())
            .or_insert_with(|| CentroidPeak::new(mz, 1.0, IndexType::MAX).into());
        return key;
    }

    pub fn create_and_get<'a>(&'a mut self, mz: f64) -> &'a C {
        let key = self.key_for(mz);
        self.create(mz);
        self.placeholders.get(&key).unwrap()
    }

    pub fn get<'a>(&'a self, mz: f64) -> Option<&'a C> {
        let key = self.key_for(mz);
        self.placeholders.get(&key)
    }

    pub fn get_key(&self, key: &i64) -> &C {
        self.placeholders.get(key).unwrap()
    }

    pub fn clear(&mut self) {
        self.placeholders.clear()
    }
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> MZCaching for PlaceholderCache<C> {}

#[derive(Debug, Clone, Default)]
pub struct SliceCache {
    range_to_indices: HashMap<(i64, i64), Range<usize>>,
}

impl MZCaching for SliceCache {}

impl SliceCache {
    pub fn clear(&mut self) {
        self.range_to_indices.clear()
    }

    pub fn entry(&mut self, m1: f64, m2: f64) -> Entry<(i64, i64), Range<usize>> {
        let key = self.key_from(m1, m2);
        self.range_to_indices.entry(key)
    }

    pub fn key_from(&mut self, m1: f64, m2: f64) -> (i64, i64) {
        let i1 = self.key_for(m1);
        let i2 = self.key_for(m2);
        (i1, i2)
    }
}

#[derive(Debug)]
pub struct WorkingPeakSet<C: CentroidLike + Clone + From<CentroidPeak>> {
    pub peaks: MZPeakSetType<C>,
    pub placeholders: PlaceholderCache<C>,
    pub slice_cache: SliceCache,
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> WorkingPeakSet<C> {
    pub fn new(peaks: MZPeakSetType<C>) -> Self {
        Self {
            peaks,
            placeholders: PlaceholderCache::default(),
            slice_cache: SliceCache::default(),
        }
    }

    pub fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        match self.slice_cache.entry(m1, m2) {
            Entry::Occupied(r) => r.get().clone(),
            Entry::Vacant(v) => {
                let ivs = self.peaks.between(m1, m2, 0.001, MassErrorType::Absolute);
                let r = if ivs.is_empty() {
                    0..0
                } else {
                    let start = ivs.first().unwrap().get_index() as usize;
                    let end = (ivs.last().unwrap().get_index() + 1) as usize;
                    start..end
                };
                v.insert(r.clone());
                r
            }
        }
    }

    pub fn match_theoretical(
        &mut self,
        tid: &TheoreticalIsotopicPattern,
        error_tolerance: f64,
    ) -> (Vec<PeakKey>, usize) {
        let mut peaks = Vec::with_capacity(tid.len());
        let mut missed = 0;
        for peak in tid.iter() {
            let (key, missed_peak) = self.has_peak(peak.mz(), error_tolerance, MassErrorType::PPM);
            peaks.push(key);
            missed += missed_peak as usize;
        }
        (peaks, missed)
    }

    pub fn has_peak(
        &mut self,
        mz: f64,
        error_tolerance: f64,
        error_type: MassErrorType,
    ) -> (PeakKey, bool) {
        match self.peaks.has_peak(mz, error_tolerance, error_type) {
            Some(peak) => (PeakKey::Matched(peak.get_index()), false),
            None => (PeakKey::Placeholder(self.placeholders.create(mz)), true),
        }
    }

    pub fn collect_for(&self, keys: &Vec<PeakKey>) -> Vec<&C> {
        let mut result = Vec::with_capacity(keys.len());
        for key in keys.iter() {
            result.push(self.get(key))
        }
        result
    }

    pub fn get(&self, key: &PeakKey) -> &C {
        match key {
            PeakKey::Matched(i) => &self.peaks[*i as usize],
            PeakKey::Placeholder(i) => self.placeholders.get_key(i),
        }
    }

    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    pub fn key_iter(&self) -> PeakKeyIter {
        PeakKeyIter::descending(self.peaks.len())
    }
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> std::ops::Index<usize> for WorkingPeakSet<C> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        &self.peaks[index]
    }
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> std::ops::Index<PeakKey> for WorkingPeakSet<C> {
    type Output = C;

    fn index(&self, index: PeakKey) -> &Self::Output {
        match index {
            PeakKey::Matched(i) => self.peaks.index(i as usize),
            PeakKey::Placeholder(i) => self.placeholders.get_key(&i),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeakKeyIter {
    total: usize,
    current: usize,
    sign: i8,
}

impl PeakKeyIter {
    pub fn new(total: usize, current: usize, sign: i8) -> Self {
        Self {
            total,
            current,
            sign,
        }
    }

    pub fn descending(total: usize) -> Self {
        let current = if total == 0 { usize::MAX } else { total - 1 };

        Self {
            total,
            current,
            sign: -1,
        }
    }

    pub fn next_key(&mut self) -> Option<PeakKey> {
        if self.sign > 0 {
            if self.current < self.total - 1 {
                let res = PeakKey::Matched(self.current as u32);
                self.current += 1;
                Some(res)
            } else {
                None
            }
        } else {
            if self.current > 0 && self.current < self.total {
                let res = PeakKey::Matched(self.current as u32);
                self.current -= 1;
                Some(res)
            } else if self.current == 0 {
                let res = PeakKey::Matched(self.current as u32);
                self.current = usize::MAX;
                Some(res)
            } else {
                None
            }
        }
    }
}

impl Iterator for PeakKeyIter {
    type Item = PeakKey;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_key()
    }
}