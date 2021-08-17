use crate::isotopic_fit::{IsotopicFit};
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::isotopic_model::{CachingIsotopicModel, IsotopicPatternGenerator, PROTON};
use crate::scorer::{IsotopicPatternScorer, MSDeconvScorer};
use mzpeaks::CentroidPeak;
use mzpeaks::{prelude::*};
use mzpeaks::{MZPeakSetType};

#[derive(Debug)]
pub struct DeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak>,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
> {
    pub peaks: WorkingPeakSet<C>,
    pub isotopic_model: I,
    pub scorer: S,
}


impl<C: CentroidLike + Clone + From<CentroidPeak>, I: IsotopicPatternGenerator, S: IsotopicPatternScorer>
    DeconvoluterType<C, I, S>
{
    pub fn new(peaks: MZPeakSetType<C>, isotopic_model: I, scorer: S) -> Self {
        Self {
            peaks: WorkingPeakSet::new(peaks),
            isotopic_model,
            scorer,
        }
    }

    pub fn fit_theoretical_isotopic_pattern(&mut self, peak: PeakKey, charge: i32) -> IsotopicFit {
        let mz = self.peaks.get(&peak).mz();
        let tid = self.isotopic_model.isotopic_cluster(mz, charge, PROTON, 0.95, 0.001);
        let (keys, missed_peaks) = self.peaks.match_theoretical(&tid, 10.0);
        let exp = self.peaks.collect_for(&keys);
        // TODO scale `tid` here
        let score = self.scorer.score(&exp, &tid);
        IsotopicFit::new(keys, peak, tid, charge, score, missed_peaks as u16)
    }
}


pub type AveragineDeconvoluter<'lifespan> = DeconvoluterType<CentroidPeak, CachingIsotopicModel<'lifespan>, MSDeconvScorer>;


#[cfg(test)]
mod test {
    use crate::isotopic_model::IsotopicModels;

    use super::*;

    #[test]
    fn test_mut() {
        let peaks = vec![CentroidPeak::new(300.0, 150.0, 0), CentroidPeak::new(301.007, 5.0, 1)];
        let peaks = MZPeakSetType::new(peaks);
        let mut task = AveragineDeconvoluter::new(peaks, IsotopicModels::Peptide.into(), MSDeconvScorer::default());
        let p = PeakKey::Matched(0);
        let fit1 = task.fit_theoretical_isotopic_pattern(p, 1);
        let fit2 = task.fit_theoretical_isotopic_pattern(p, 2);
        assert!(fit1.score > fit2.score);
    }
}