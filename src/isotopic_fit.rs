use std::hash::{Hash, Hasher};

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;

use crate::peaks::PeakKey;

#[derive(Debug, Clone)]
pub struct IsotopicFit {
    pub experimental: Vec<PeakKey>,
    pub theoretical: TheoreticalIsotopicPattern,
    pub seed_peak: PeakKey,
    pub charge: i32,
    pub score: f64,
    pub missed_peaks: u16,
}

impl PartialEq for IsotopicFit {
    fn eq(&self, other: &Self) -> bool {
        let val = (self.score - other.score).abs() < 1e-6;
        if !val {
            return false;
        }
        let val = self.charge == other.charge;
        if !val {
            return false;
        }
        for (a, b) in self.experimental.iter().zip(other.experimental.iter()) {
            if a != b {
                return false;
            }
        }
        if self.theoretical != other.theoretical {
            return false;
        }
        true
    }
}

impl PartialOrd for IsotopicFit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        return self.score.partial_cmp(&other.score);
    }
}

impl Eq for IsotopicFit {}

impl Ord for IsotopicFit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hash for IsotopicFit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(key) = self.experimental.first() {
            key.hash(state);
        }
        self.charge.hash(state);
    }
}

impl IsotopicFit {
    pub fn new(
        experimental: Vec<PeakKey>,
        seed_peak: PeakKey,
        theoretical: TheoreticalIsotopicPattern,
        charge: i32,
        score: f64,
        missed_peaks: u16,
    ) -> Self {
        Self {
            experimental,
            seed_peak,
            theoretical,
            charge,
            score,
            missed_peaks,
        }
    }

    pub fn len(&self) -> usize {
        self.experimental.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
