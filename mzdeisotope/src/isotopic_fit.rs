/*! Description of isotopic pattern fits */
use std::hash::{Hash, Hasher};

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;

use crate::{peaks::PeakKey, scorer::ScoreType};

/// Describes an isotopic pattern fit produced during the deconvolution of a
/// spectrum.
#[derive(Debug, Clone)]
pub struct IsotopicFit {
    /// The set of references to observed peaks
    pub experimental: Vec<PeakKey>,
    /// The associated theoretical isotopic pattern
    pub theoretical: TheoreticalIsotopicPattern,
    /// The peak reference that was used to seed the isotopic pattern
    pub seed_peak: PeakKey,
    /// The charge state of the isotopic pattern fitted
    pub charge: i32,
    /// The quality of the isotopic fit calculated by some algorithm
    pub score: ScoreType,
    /// The number of experimental peaks that were *missing* in `experimental`
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
        Some(self.cmp(other))
    }
}

impl Eq for IsotopicFit {}

impl Ord for IsotopicFit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score)
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
    #[inline]
    pub fn new(
        experimental: Vec<PeakKey>,
        seed_peak: PeakKey,
        theoretical: TheoreticalIsotopicPattern,
        charge: i32,
        score: ScoreType,
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

    #[inline]
    pub fn len(&self) -> usize {
        self.experimental.len()
    }

    #[inline]
    pub fn num_missed_peaks(&self) -> usize {
        self.missed_peaks as usize
    }

    #[inline]
    pub fn num_matched_peaks(&self) -> usize {
        self.len() - self.num_missed_peaks()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
