use std::collections::HashSet;
use std::ops::Range;

use crate::charge::{ChargeIterator, ChargeRangeIter};
use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    isotopic_shift, CachingIsotopicModel, IsotopicPatternGenerator, PROTON,
};
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter,
};

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::prelude::*;
use mzpeaks::{CentroidPeak, MZPeakSetType, MassErrorType};

#[derive(Debug, Clone, Copy)]
pub enum TIDScalingMethod {
    Sum,
    Max,
    Top3,
}

impl Default for TIDScalingMethod {
    fn default() -> Self {
        TIDScalingMethod::Sum
    }
}

impl TIDScalingMethod {
    pub fn scale<C: CentroidLike>(
        &self,
        experimental: &[C],
        theoretical: &mut TheoreticalIsotopicPattern,
    ) {
        if theoretical.len() == 0 {
            return;
        }
        match self {
            Self::Sum => {
                let total: f32 = experimental.iter().map(|p| p.intensity()).sum();
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= total as f64);
            }
            Self::Max => {
                let (index, peak) = experimental
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.intensity().partial_cmp(&b.1.intensity()).unwrap())
                    .unwrap();
                let scale = peak.intensity() / theoretical[index].intensity();
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
            Self::Top3 => {
                let mut t1_index: usize = 0;
                let mut t2_index: usize = 0;
                let mut t3_index: usize = 0;
                let mut t1 = 0.0f32;
                let mut t2 = 0.0f32;
                let mut t3 = 0.0f32;

                for (i, p) in experimental.iter().enumerate() {
                    let y = p.intensity();
                    if y > t1 {
                        t3 = t2;
                        t2 = t1;
                        t1 = y;
                        t3_index = t2_index;
                        t2_index = t1_index;
                        t1_index = i;
                    } else if y > t2 {
                        t3_index = t2_index;
                        t3 = t2;
                        t2 = y;
                        t2_index = i;
                    } else if y > t3 {
                        t3_index = i;
                        t3 = y;
                    }
                }

                let mut scale = experimental[t1_index].intensity() / t1;
                scale += experimental[t2_index].intensity() / t2;
                scale += experimental[t3_index].intensity() / t3;
                theoretical
                    .iter_mut()
                    .for_each(|p| p.intensity *= scale as f64);
            }
        }
    }
}

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

pub trait IsotopicPatternFitter<C: CentroidLike> {
    fn fit_theoretical_isotopic_pattern(&mut self, peak: PeakKey, charge: i32) -> IsotopicFit;

    fn has_peak(&mut self, mz: f64, error_tolerance: f64) -> PeakKey;
    fn between(&mut self, m1: f64, m2: f64) -> Range<usize>;
    fn get_peak(&self, key: PeakKey) -> &C;
    fn create_key(&mut self, mz: f64) -> PeakKey;
    fn peak_count(&self) -> usize;
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

pub trait RelativePeakSearch<C: CentroidLike>: IsotopicPatternFitter<C> {
    fn has_previous_peak_at_charge(
        &mut self,
        mz: f64,
        charge: i32,
        step: i8,
        error_tolerance: f64,
    ) -> Option<PeakKey> {
        let prev = mz - isotopic_shift(charge) * step as f64;
        match self.has_peak(prev, error_tolerance) {
            PeakKey::Matched(i) => Some(PeakKey::Matched(i)),
            PeakKey::Placeholder(_i) => None,
        }
    }

    fn has_successor_peak_at_charge(
        &mut self,
        mz: f64,
        charge: i32,
        step: i8,
        error_tolerance: f64,
    ) -> Option<PeakKey> {
        let next = mz + isotopic_shift(charge) * step as f64;
        match self.has_peak(next, error_tolerance) {
            PeakKey::Matched(i) => Some(PeakKey::Matched(i)),
            PeakKey::Placeholder(_i) => None,
        }
    }

    fn find_forward_putative_peak_inplace(
        &mut self,
        mz: f64,
        charge: i32,
        result: &mut HashSet<(PeakKey, i32)>,
        step: i8,
        error_tolerance: f64,
    ) -> usize {
        let shift = isotopic_shift(charge);
        let next_peak = mz + (shift * step as f64);
        let iv = self.between(
            MassErrorType::PPM.lower_bound(next_peak, error_tolerance),
            MassErrorType::PPM.upper_bound(next_peak, error_tolerance),
        );
        for i in iv.clone() {
            let forward = self.get_peak(PeakKey::Matched(i as u32)).mz();
            let prev_peak_mz = forward - (shift * step as f64);
            let key = self.create_key(prev_peak_mz);
            result.insert((key, charge));
        }
        iv.end - iv.start
    }

    fn find_previous_putative_peak_inplace(
        &mut self,
        mz: f64,
        charge: i32,
        result: &mut HashSet<(PeakKey, i32)>,
        step: i8,
        error_tolerance: f64,
    ) -> usize {
        let shift = isotopic_shift(charge);
        let prev_peak = mz - shift;
        let iv = self.between(
            MassErrorType::PPM.lower_bound(prev_peak, error_tolerance),
            MassErrorType::PPM.upper_bound(prev_peak, error_tolerance),
        );
        for i in iv.clone() {
            let prev_peak_mz = self.get_peak(PeakKey::Matched(i as u32)).mz();
            if step == 1 {
                self.find_forward_putative_peak_inplace(
                    prev_peak_mz,
                    charge,
                    result,
                    1,
                    error_tolerance,
                );
            } else if step > 0 {
                self.find_previous_putative_peak_inplace(
                    prev_peak_mz,
                    charge,
                    result,
                    step - 1,
                    error_tolerance,
                );
            }
        }
        iv.end - iv.start
    }
}

pub trait ExhaustivePeakSearch<C: CentroidLike>:
    IsotopicPatternFitter<C> + RelativePeakSearch<C>
{
    fn fit_peaks_at_charge(
        &mut self,
        peak_charge_set: HashSet<(PeakKey, i32)>,
    ) -> HashSet<IsotopicFit> {
        let mut solutions = HashSet::new();
        for (key, charge) in peak_charge_set {
            let fit = self.fit_theoretical_isotopic_pattern(key, charge);
            if self.check_isotopic_fit(&fit) {
                solutions.insert(fit);
            }
        }
        solutions
    }

    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        fit.score >= 0.0
    }

    fn _find_all_peak_charge_pairs_iter<I: ChargeIterator>(
        &mut self,
        mz: f64,
        error_tolerance: f64,
        charge_iter: I,
        left_search_limit: i8,
        right_search_limit: i8,
        recalculate_starting_peak: bool,
    ) -> HashSet<(PeakKey, i32)> {
        let mut solutions = HashSet::new();
        for charge in charge_iter {
            let key = self.has_peak(mz, error_tolerance);
            solutions.insert((key, charge));
            for i in 1..left_search_limit + 1 {
                if let Some(key) = self.has_previous_peak_at_charge(mz, charge, i, error_tolerance)
                {
                    solutions.insert((key, charge));
                }
                if recalculate_starting_peak {
                    self.find_previous_putative_peak_inplace(
                        mz,
                        charge,
                        &mut solutions,
                        i,
                        error_tolerance,
                    );
                }
            }
            for i in 1..right_search_limit + 1 {
                if let Some(key) = self.has_successor_peak_at_charge(mz, charge, i, error_tolerance)
                {
                    solutions.insert((key, charge));
                    if recalculate_starting_peak {
                        self.find_forward_putative_peak_inplace(
                            mz,
                            charge,
                            &mut solutions,
                            i,
                            error_tolerance,
                        );
                    }
                }
            }
            if recalculate_starting_peak {
                for i in 0..(std::cmp::min(left_search_limit, 2)) {
                    self.find_previous_putative_peak_inplace(
                        mz,
                        charge,
                        &mut solutions,
                        i,
                        error_tolerance,
                    );
                }
            }
        }

        solutions
    }

    fn find_all_peak_charge_pairs(
        &mut self,
        mz: f64,
        error_tolerance: f64,
        charge_range: (i32, i32),
        left_search_limit: i8,
        right_search_limit: i8,
        recalculate_starting_peak: bool,
    ) -> HashSet<(PeakKey, i32)> {
        let charge_iter = ChargeRangeIter::from(charge_range);
        self._find_all_peak_charge_pairs_iter(
            mz,
            error_tolerance,
            charge_iter,
            left_search_limit,
            right_search_limit,
            recalculate_starting_peak,
        )
    }

    fn skip_peak(&self, peak: &C) -> bool {
        peak.mz() <= 0.0 || peak.intensity() <= 0.0
    }

    fn step_deconvolve(
        &mut self,
        error_tolerance: f64,
        charge_range: (i32, i32),
        left_search_limit: i8,
        right_search_limit: i8,
    ) -> HashSet<IsotopicFit> {
        let mut solutions: HashSet<IsotopicFit> = HashSet::new();
        let n = self.peak_count();
        if n == 0 {
            return solutions;
        }
        for i in (0..n).rev() {
            let peak = self.get_peak(PeakKey::Matched(i as u32));
            if self.skip_peak(peak) {
                continue;
            }
            let mz = peak.mz();
            let peak_charge_set = self.find_all_peak_charge_pairs(
                mz,
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                true,
            );
            let fits = self.fit_peaks_at_charge(peak_charge_set);
            solutions.extend(fits);
        }
        solutions
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
