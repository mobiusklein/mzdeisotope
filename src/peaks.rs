use std::collections::HashMap;
use chemical_elements::isotopic_pattern::{TheoreticalIsotopicPattern};
use mzpeaks::prelude::*;
use mzpeaks::{CentroidPeak, MZPeakSetType, IndexType, MassErrorType};


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeakKey {
    Matched(u32),
    Placeholder(i64)
}


#[derive(Debug)]
pub struct PlaceholderCache<C: CentroidLike + Clone + From<CentroidPeak>> {
    placeholders: HashMap<i64, C>
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> Default for PlaceholderCache<C> {
    fn default() -> Self {
        Self { placeholders: Default::default() }
    }
}

impl<C: CentroidLike + Clone + From<CentroidPeak>> PlaceholderCache<C> {

    pub fn create(&mut self, mz: f64) -> i64 {
        let key = self.key_for(mz);
        self.placeholders.entry(key.clone()).or_insert_with(
            ||CentroidPeak::new(mz, 1.0, IndexType::MAX).into());
        return key
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

    pub fn key_for(&self, mz: f64) -> i64 {
        let key = (mz * 1000.0).round() as i64;
        key
    }

}


#[derive(Debug)]
pub struct WorkingPeakSet<C: CentroidLike + Clone + From<CentroidPeak>> {
    pub peaks: MZPeakSetType<C>,
    pub placeholders: PlaceholderCache<C>
}


impl<C: CentroidLike + Clone + From<CentroidPeak>> WorkingPeakSet<C> {
    pub fn new(peaks: MZPeakSetType<C>) -> Self {
        Self {
            peaks,
            placeholders: PlaceholderCache::default()
        }
    }

    pub fn match_theoretical(&mut self, tid: &TheoreticalIsotopicPattern, error_tolerance: f64) -> (Vec<PeakKey>, usize) {
        let mut peaks = Vec::with_capacity(tid.len());
        let mut missed = 0;
        for peak in tid.iter() {
            if let Some(hit) = self.peaks.has_peak(peak.mz(), error_tolerance,  MassErrorType::PPM) {
                peaks.push(PeakKey::Matched(hit.get_index()));
            } else {
                let k = self.placeholders.create(peak.mz());
                peaks.push(PeakKey::Placeholder(k));
                missed += 1;
            }
        }
        (peaks, missed)
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
            PeakKey::Matched(i) => {
                &self.peaks[*i as usize]
            },
            PeakKey::Placeholder(i) => {
                self.placeholders.get_key(i)
            }
        }
    }
}

