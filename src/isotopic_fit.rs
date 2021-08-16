use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::prelude::*;

#[derive(Debug, Clone)]
pub struct IsotopicFit<'peaks, C: CentroidLike + Clone> {
    pub experimental: Vec<C>,
    pub theoretical: TheoreticalIsotopicPattern,
    pub seed_peak: &'peaks C,
    pub monoisotopic_peak: &'peaks C,
    pub charge: i32,
    pub score: f64,
    pub missed_peaks: u16,
}

impl<'peaks, C: CentroidLike + Clone> PartialEq for IsotopicFit<'peaks, C> {
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

impl<'peaks, C: CentroidLike + Clone> PartialOrd for IsotopicFit<'peaks, C> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        return self.score.partial_cmp(&other.score);
    }
}

impl<'peaks, C: CentroidLike + Clone> Eq for IsotopicFit<'peaks, C> {}

impl<'peaks, C: CentroidLike + Clone> Ord for IsotopicFit<'peaks, C> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'peaks, C: CentroidLike + Clone> std::hash::Hash for IsotopicFit<'peaks, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.experimental.first() {
            Some(peak) => (peak.mz().round() as i64).hash(state),
            None => {}
        }
        self.charge.hash(state);
    }
}

impl<'peaks, C: CentroidLike + Clone> IsotopicFit<'peaks, C> {
    pub fn new(
        experimental: Vec<C>,
        seed_peak: &'peaks C,
        monoisotopic_peak: &'peaks C,
        theoretical: TheoreticalIsotopicPattern,
        charge: i32,
        score: f64,
        missed_peaks: u16,
    ) -> Self {
        Self {
            experimental,
            seed_peak,
            monoisotopic_peak,
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
