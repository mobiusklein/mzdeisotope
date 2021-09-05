use std::ops::Range;

use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    CachingIsotopicModel, IsotopicPatternGenerator, PROTON,
    TIDScalingMethod
};
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter,
};

use crate::deconv_traits::{ExhaustivePeakSearch, IsotopicPatternFitter, RelativePeakSearch};


use mzpeaks::prelude::*;
use mzpeaks::{CentroidPeak, MZPeakSetType, MassErrorType};


#[derive(Debug)]
pub struct DeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak>,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub peaks: WorkingPeakSet<C>,
    pub isotopic_model: I,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TIDScalingMethod,
    pub max_missed_peaks: u16,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        if fit.missed_peaks > self.max_missed_peaks {
            return false;
        }
        self.fit_filter.test(fit)
    }
}


impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > DeconvoluterType<C, I, S, F>
{
    pub fn new(
        peaks: MZPeakSetType<C>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        max_missed_peaks: u16,
    ) -> Self {
        Self {
            peaks: WorkingPeakSet::new(peaks),
            isotopic_model,
            scorer,
            fit_filter,
            scaling_method: TIDScalingMethod::default(),
            max_missed_peaks,
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak>,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for DeconvoluterType<C, I, S, F>
{
    fn fit_theoretical_isotopic_pattern(&mut self, peak: PeakKey, charge: i32) -> IsotopicFit {
        let mz = self.peaks.get(&peak).mz();
        let mut tid = self
            .isotopic_model
            .isotopic_cluster(mz, charge, PROTON, 0.95, 0.001);
        let (keys, missed_peaks) = self.peaks.match_theoretical(&tid, 10.0);
        let exp = self.peaks.collect_for(&keys);
        self.scaling_method.scale(&exp, &mut tid);
        let score = self.scorer.score(&exp, &tid);
        IsotopicFit::new(keys, peak, tid, charge, score, missed_peaks as u16)
    }

    fn has_peak(&mut self, mz: f64, error_tolerance: f64) -> PeakKey {
        let (peak, _missed) = self.peaks.has_peak(mz, error_tolerance, MassErrorType::PPM);
        peak
    }

    fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        self.peaks.between(m1, m2)
    }

    fn get_peak(&self, key: PeakKey) -> &C {
        self.peaks.get(&key)
    }

    fn create_key(&mut self, mz: f64) -> PeakKey {
        let i = self.peaks.placeholders.create(mz);
        PeakKey::Placeholder(i)
    }

    fn peak_count(&self) -> usize {
        self.peaks.len()
    }
}


pub type AveragineDeconvoluter<'lifespan> = DeconvoluterType<
    CentroidPeak,
    CachingIsotopicModel<'lifespan>,
    MSDeconvScorer,
    MaximizingFitFilter,
>;

#[cfg(test)]
mod test {
    use crate::isotopic_model::IsotopicModels;

    use super::*;

    #[test]
    fn test_mut() {
        let peaks = vec![
            CentroidPeak::new(300.0, 150.0, 0),
            CentroidPeak::new(301.007, 5.0, 1),
        ];
        let peaks = MZPeakSetType::new(peaks);
        let mut task = AveragineDeconvoluter::new(
            peaks,
            IsotopicModels::Peptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::default(),
            1,
        );
        let p = PeakKey::Matched(0);
        let fit1 = task.fit_theoretical_isotopic_pattern(p, 1);
        let fit2 = task.fit_theoretical_isotopic_pattern(p, 2);
        assert!(fit1.score > fit2.score);
    }

    #[test]
    fn test_fit_all() {
        let peaks = vec![
            CentroidPeak::new(300.0 - 1.007, 3.0, 0),
            CentroidPeak::new(300.0, 150.0, 2),
            CentroidPeak::new(301.007, 5.0, 3),
        ];
        let peaks = MZPeakSetType::new(peaks);
        let mut task = AveragineDeconvoluter::new(
            peaks,
            IsotopicModels::Peptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::default(),
            1,
        );
        let solution_space = task.find_all_peak_charge_pairs(300.0, 10.0, (1, 8), 1, 1, true);
        assert_eq!(solution_space.len(), 10);
    }
}
