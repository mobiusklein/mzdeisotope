use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Range;

use chemical_elements::neutral_mass;
use mzpeaks::peak::MZPoint;
use mzpeaks::{CentroidLike, MZLocated, MassPeakSetType, Tolerance};

use crate::charge::{ChargeIterator, ChargeRange, ChargeRangeIter};
use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{isotopic_shift, IsotopicPatternParams, PROTON};
use crate::peaks::PeakKey;
use crate::solution::DeconvolvedSolutionPeak;

pub trait IsotopicPatternFitter<C: CentroidLike> {
    fn fit_theoretical_isotopic_pattern(&mut self, peak: PeakKey, charge: i32) -> IsotopicFit {
        self.fit_theoretical_isotopic_pattern_with_params(
            peak,
            charge,
            IsotopicPatternParams::default(),
        )
    }

    fn fit_theoretical_isotopic_pattern_with_params(
        &mut self,
        peak: PeakKey,
        charge: i32,
        params: IsotopicPatternParams,
    ) -> IsotopicFit;

    fn make_solution_from_fit(
        &self,
        fit: &IsotopicFit,
        error_tolerance: Tolerance,
    ) -> DeconvolvedSolutionPeak {
        let first_peak = self.get_peak(*fit.experimental.first().unwrap());
        let first_peak_mz = if !error_tolerance.test(first_peak.mz(), fit.theoretical.origin) {
            match self.has_peak_direct(fit.theoretical.origin, error_tolerance) {
                Some(p) => p.mz(),
                None => fit.theoretical.origin,
            }
        } else {
            first_peak.mz()
        };

        let neutral_mass = neutral_mass(first_peak_mz, fit.charge, PROTON);
        let envelope: Vec<_> = fit
            .experimental
            .iter()
            .zip(fit.theoretical.iter())
            .map(|(ek, t)| {
                let e = self.get_peak(*ek);
                MZPoint::new(e.mz(), e.intensity().min(t.intensity()))
            })
            .collect();
        let intensity = envelope.iter().map(|p| p.intensity).sum();

        DeconvolvedSolutionPeak::new(
            neutral_mass,
            intensity,
            fit.charge,
            0,
            fit.score,
            Box::new(envelope),
        )
    }

    fn has_peak_direct(&self, mz: f64, error_tolerance: Tolerance) -> Option<&C>;
    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey;
    fn between(&mut self, m1: f64, m2: f64) -> Range<usize>;
    fn get_peak(&self, key: PeakKey) -> &C;
    fn create_key(&mut self, mz: f64) -> PeakKey;
    fn peak_count(&self) -> usize;

    fn merge_isobaric_peaks(&self, mut peaks: Vec<DeconvolvedSolutionPeak>) -> Vec<DeconvolvedSolutionPeak> {
        let mut acc = Vec::with_capacity(peaks.len());
        if peaks.len() == 0 {
            return acc
        }
        peaks.sort_unstable_by(|a, b| {
            match a.charge.cmp(&b.charge) {
                Ordering::Equal => a.neutral_mass.total_cmp(&b.neutral_mass),
                x => x,
            }
        });

        let mut it = peaks.into_iter();
        let first = it.next().unwrap();
        let last = it.fold(first, |mut prev, current| {
            if prev.charge == current.charge && Tolerance::Da(1e-3).test(current.neutral_mass, prev.neutral_mass) {
                if current.index != 0 {
                    prev.index = current.index;
                }
                prev.intensity += current.intensity;
                prev.envelope.iter_mut().zip(current.envelope.iter()).for_each(|(p, c)| {
                    p.intensity += c.intensity;
                });
                prev
            } else {
                acc.push(prev);
                current
            }
        });
        acc.push(last);

        acc
    }

    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit);
}

pub trait RelativePeakSearch<C: CentroidLike>: IsotopicPatternFitter<C> {
    fn has_previous_peak_at_charge(
        &mut self,
        mz: f64,
        charge: i32,
        step: i8,
        error_tolerance: Tolerance,
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
        error_tolerance: Tolerance,
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
        error_tolerance: Tolerance,
    ) -> usize {
        let shift = isotopic_shift(charge);
        let next_peak = mz + (shift * step as f64);
        let mass_iv = error_tolerance.bounds(next_peak);
        let iv = self.between(mass_iv.0, mass_iv.1);
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
        error_tolerance: Tolerance,
    ) -> usize {
        let shift = isotopic_shift(charge);
        let prev_peak = mz - shift;
        let mass_iv = error_tolerance.bounds(prev_peak);
        let iv = self.between(mass_iv.0, mass_iv.1);
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
        isotopic_params: IsotopicPatternParams,
    ) -> HashSet<IsotopicFit> {
        let mut solutions = HashSet::new();
        for (key, charge) in peak_charge_set {
            let fit =
                self.fit_theoretical_isotopic_pattern_with_params(key, charge, isotopic_params);
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
        error_tolerance: Tolerance,
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
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
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
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
    ) -> Vec<IsotopicFit> {
        let mut solutions: Vec<IsotopicFit> = Vec::new();
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
            let fits = self.fit_peaks_at_charge(peak_charge_set, isotopic_params);
            solutions.extend(fits);
        }
        solutions
    }
}

pub trait GraphDependentSearch<C: CentroidLike>: ExhaustivePeakSearch<C> {
    fn add_fit_dependence(&mut self, fit: IsotopicFit);

    fn select_best_disjoint_subgraphs(&mut self, fit_accumulator: &mut Vec<IsotopicFit>);

    fn _explore_local(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
    ) -> usize {
        let peak = self.get_peak(peak);
        if self.skip_peak(peak) {
            0
        } else {
            let mz = peak.mz();
            let peak_charge_set = self.find_all_peak_charge_pairs(
                mz,
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                true,
            );
            let fits = self.fit_peaks_at_charge(peak_charge_set, isotopic_params);
            let k = fits.len();
            fits.into_iter().for_each(|f| {
                if f.charge.abs() > 1 && f.num_matched_peaks() == 1 {
                    return;
                }
                self.add_fit_dependence(f)
            });
            k
        }
    }

    fn populate_graph(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
    ) -> usize {
        let n = self.peak_count();
        if n == 0 {
            return 0;
        }

        (0..n)
            .rev()
            .map(|i| {
                // let peak = self.get_peak();
                self._explore_local(PeakKey::Matched(i as u32), error_tolerance, charge_range, left_search_limit, right_search_limit, isotopic_params)
            })
            .sum()
    }

    fn graph_step_deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: (i32, i32),
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
    ) -> Vec<IsotopicFit> {
        let mut fit_accumulator = Vec::new();
        self.populate_graph(
            error_tolerance,
            charge_range,
            left_search_limit,
            right_search_limit,
            isotopic_params,
        );
        self.select_best_disjoint_subgraphs(&mut fit_accumulator);
        fit_accumulator
    }
}

pub trait TargetedDeconvolution<C: CentroidLike>:
    IsotopicPatternFitter<C> + RelativePeakSearch<C>
{
    type TargetSolution;

    fn targeted_deconvolution(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        params: IsotopicPatternParams,
    ) -> Self::TargetSolution;

    fn resolve_target<'a, 'b: 'a>(
        &self,
        solution: &'b MassPeakSetType<DeconvolvedSolutionPeak>,
        target: &'a Self::TargetSolution,
    ) -> Option<&'b DeconvolvedSolutionPeak>;
}

pub trait IsotopicDeconvolutionAlgorithm<C: CentroidLike> {
    fn deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
        convergence: f32,
        max_iterations: u32,
    ) -> MassPeakSetType<DeconvolvedSolutionPeak>;
}
