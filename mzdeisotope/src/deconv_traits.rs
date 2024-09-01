/*! Traits that implement infrastructure for the deconvolution process */
use std::cmp::Ordering;
use std::collections::HashSet;
use std::ops::Range;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use chemical_elements::neutral_mass;
use mzpeaks::peak::MZPoint;
use mzpeaks::{CentroidLike, MZLocated, MassPeakSetType, Tolerance};
use thiserror::Error;

use crate::charge::{ChargeIterator, ChargeListIter, ChargeRange, ChargeRangeIter};
use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{isotopic_shift, IsotopicPatternParams, PROTON};
use crate::peak_graph::FitRef;
use crate::peaks::PeakKey;
use crate::scorer::ScoreType;
use crate::solution::DeconvolvedSolutionPeak;

pub type QuerySet = HashSet<(PeakKey, i32)>;

/// An error that might occur during deconvolution
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DeconvolutionError {
    #[error("Failed to resolve a deconvolution solution")]
    FailedToResolveSolution,
    #[error("Failed to resolve a fit reference {0:?}")]
    FailedToResolveFit(FitRef),
}


/// A set of behaviors for fitting theoretical peaks to experimentally detected
/// peak lists.
pub trait IsotopicPatternFitter<C: CentroidLike> {
    fn collect_for(&self, keys: &[PeakKey]) -> Vec<&C>;

    fn fit_theoretical_isotopic_pattern(
        &mut self,
        peak: PeakKey,
        charge: i32,
        error_tolerance: Tolerance,
    ) -> IsotopicFit {
        self.fit_theoretical_isotopic_pattern_with_params(
            peak,
            charge,
            error_tolerance,
            IsotopicPatternParams::default(),
        )
    }

    fn create_isotopic_pattern(
        &mut self,
        mz: f64,
        charge: i32,
        params: IsotopicPatternParams,
    ) -> TheoreticalIsotopicPattern;

    fn score_isotopic_fit(
        &self,
        experimental: &[&C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType;

    fn fit_theoretical_isotopic_pattern_with_params(
        &mut self,
        peak: PeakKey,
        charge: i32,
        error_tolerance: Tolerance,
        params: IsotopicPatternParams,
    ) -> IsotopicFit;

    fn fit_theoretical_isotopic_pattern_incremental_from_seed(
        &self,
        seed_fit: &IsotopicFit,
        experimental: &[&C],
        params: IsotopicPatternParams,
        lower_bound: f64,
    ) -> Vec<IsotopicFit>;

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

    /// Search for an experimental peak satisfying the m/z error tolerance limits
    fn has_peak_direct(&self, mz: f64, error_tolerance: Tolerance) -> Option<&C>;
    /// Search for an experimental peak satisfying the m/z error tolerance limits,
    /// but return a [`PeakKey`] which records the index of the peak if it is found
    /// or a unique identifier for a marker for a common missing peak.
    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey;
    fn between(&mut self, m1: f64, m2: f64) -> Range<usize>;
    /// Translate a [`PeakKey`] into a reference to `C`. Placeholders are created
    /// and cached on-the-fly.
    fn get_peak(&self, key: PeakKey) -> &C;
    /// Translate an m/z into a placeholder
    fn create_key(&mut self, mz: f64) -> PeakKey;
    /// The number of peaks in the experimental peak list
    fn peak_count(&self) -> usize;

    #[tracing::instrument(skip_all, level = "trace")]
    fn merge_isobaric_peaks(
        &self,
        mut peaks: Vec<DeconvolvedSolutionPeak>,
    ) -> Vec<DeconvolvedSolutionPeak> {
        let mut acc = Vec::with_capacity(peaks.len());
        if peaks.is_empty() {
            return acc;
        }
        peaks.sort_by(|a, b| match a.charge.cmp(&b.charge) {
            Ordering::Equal => {
                if (a.neutral_mass - b.neutral_mass).abs() <= 1e-3 {
                    if (a.intensity - b.intensity).abs() <= 1e-3 {
                        Ordering::Equal
                    } else {
                        a.intensity.total_cmp(&b.intensity)
                    }
                } else {
                    a.neutral_mass.total_cmp(&b.neutral_mass)
                }
            }
            x => x,
        });

        let mut it = peaks.into_iter();
        let first = it.next().unwrap();
        let last = it.fold(first, |mut prev, current| {
            if prev.charge == current.charge
                && Tolerance::Da(1e-3).test(current.neutral_mass, prev.neutral_mass)
            {
                if current.index != 0 {
                    prev.index = current.index;
                }
                prev.intensity += current.intensity;
                prev.envelope
                    .iter_mut()
                    .zip(current.envelope.iter())
                    .for_each(|(p, c)| {
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

    /// Subtract the scaled theoretical intensities from experimental peaks
    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit);
}

/// A collection of common behaviors for searching for peaks +/- isotopic peaks
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
        result: &mut QuerySet,
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
        result: &mut QuerySet,
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


/// A collection of behaviors to exhaustively try to fit each experimental peak
/// as a theoretical isotopic distribution.
pub trait ExhaustivePeakSearch<C: CentroidLike>:
    IsotopicPatternFitter<C> + RelativePeakSearch<C>
{
    /// The minimum intensity below which an experimental peak is ignored.
    const MINIMUM_INTENSITY: f32 = 5.0;

    /// Evaluate a set of local isotopic pattern seeds and return the set of [`IsotopicFit`]
    /// that pass quality thresholds.
    ///
    /// # Arguments
    /// - `peak_charge_set`: The ([`PeakKey`], charge) pairs to fit in this batch
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `params`: The isotopic pattern generation parameters to use
    fn fit_peaks_at_charge(
        &mut self,
        peak_charge_set: QuerySet,
        error_tolerance: Tolerance,
        isotopic_params: IsotopicPatternParams,
    ) -> HashSet<IsotopicFit> {
        let mut solutions = HashSet::new();
        for (key, charge) in peak_charge_set {
            let fit = self.fit_theoretical_isotopic_pattern_with_params(
                key,
                charge,
                error_tolerance,
                isotopic_params,
            );
            if let Some(incremental_truncation) = isotopic_params.incremental_truncation {
                let experimental = self.collect_for(&fit.experimental);
                solutions.extend(
                    self.fit_theoretical_isotopic_pattern_incremental_from_seed(
                        &fit,
                        &experimental,
                        isotopic_params,
                        incremental_truncation,
                    )
                    .into_iter(),
                )
            }
            if self.check_isotopic_fit(&fit) {
                solutions.insert(fit);
            }
        }
        solutions
    }

    /// Verify an [`IsotopicFit`] passes some quality threshold for
    /// further consideration.
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool;

    /// Find all the charge states for all peaks suggested by the locale of `mz`
    ///
    /// # Arguments
    /// - `mz`: The query m/z value to search from
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `recalculate_starting_peak`: Whether or not to use adjacent isotopic peaks to recalculate the
    ///   query m/z
    fn _find_all_peak_charge_pairs_iter<I: ChargeIterator>(
        &mut self,
        mz: f64,
        error_tolerance: Tolerance,
        charge_iter: I,
        left_search_limit: i8,
        right_search_limit: i8,
        recalculate_starting_peak: bool,
    ) -> QuerySet {
        let mut solutions = QuerySet::new();
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

    fn find_all_peak_charge_pairs<I: ChargeIterator>(
        &mut self,
        mz: f64,
        error_tolerance: Tolerance,
        charge_range: I,
        left_search_limit: i8,
        right_search_limit: i8,
        recalculate_starting_peak: bool,
    ) -> QuerySet {
        self._find_all_peak_charge_pairs_iter(
            mz,
            error_tolerance,
            charge_range,
            left_search_limit,
            right_search_limit,
            recalculate_starting_peak,
        )
    }

    /// Skip a peak if it isn't fails simple quality requirements like
    /// intensity being too low or have a non-positive m/z.
    fn skip_peak(&self, peak: &C) -> bool {
        peak.mz() <= 0.0
            || peak.intensity() < Self::MINIMUM_INTENSITY
            || (peak.intensity() - Self::MINIMUM_INTENSITY).abs() <= 1e-3
    }

    /// A predicate whether or not to use the [QuickCharge](crate::charge::quick_charge) algorithm for charge
    /// state selection.
    fn use_quick_charge(&self) -> bool;

    /// Invoke [`quick_charge`](crate::charge::quick_charge) on the peak
    /// at `index`.
    fn quick_charge(&self, index: usize, charge_range: ChargeRange) -> ChargeListIter;

    /// Visit all of the peaks in the spectrum and fit each of them in m/z descending order
    /// using [`ExhaustivePeakSearch::find_all_peak_charge_pairs`] and [`ExhaustivePeakSearch::fit_peaks_at_charge`].
    ///
    /// # Arguments
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `params`: The isotopic pattern generation parameters to use
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
            let peak_charge_set = if self.use_quick_charge() {
                let charge_iter = self.quick_charge(i, charge_range);
                self.find_all_peak_charge_pairs(
                    mz,
                    error_tolerance,
                    charge_iter,
                    left_search_limit,
                    right_search_limit,
                    true,
                )
            } else {
                self.find_all_peak_charge_pairs(
                    mz,
                    error_tolerance,
                    ChargeRangeIter::from(charge_range),
                    left_search_limit,
                    right_search_limit,
                    true,
                )
            };
            let fits = self.fit_peaks_at_charge(peak_charge_set, error_tolerance, isotopic_params);
            solutions.extend(fits);
        }
        solutions
    }
}

/// In addition to the behavior of [`ExhaustivePeakSearch`], maintain a graph of
/// overlapping solutions to deconvolve complex spectra.
pub trait GraphDependentSearch<C: CentroidLike>: ExhaustivePeakSearch<C> {

    /// Register `fit`'s dependencies and store it for later extraction
    fn add_fit_dependence(&mut self, fit: IsotopicFit);

    /// Select the set of solutions for this iteration, pushing them into
    /// `fit_accumulator`.
    fn select_best_disjoint_subgraphs(
        &mut self,
        fit_accumulator: &mut Vec<IsotopicFit>,
    ) -> Result<(), DeconvolutionError>;

    fn _explore_local<I: ChargeIterator>(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: I,
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
            let fits = self.fit_peaks_at_charge(peak_charge_set, error_tolerance, isotopic_params);
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

    /// Visit all peaks in the spectrum and add their plausible solutions to the
    /// dependence graph.
    ///
    /// # Arguments
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `params`: The isotopic pattern generation parameters to use
    #[tracing::instrument(level="debug", skip_all)]
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
                if self.use_quick_charge() {
                    let charge_list = self.quick_charge(i, charge_range);
                    self._explore_local(
                        PeakKey::Matched(i as u32),
                        error_tolerance,
                        charge_list,
                        left_search_limit,
                        right_search_limit,
                        isotopic_params,
                    )
                } else {
                    self._explore_local(
                        PeakKey::Matched(i as u32),
                        error_tolerance,
                        ChargeRangeIter::from(charge_range),
                        left_search_limit,
                        right_search_limit,
                        isotopic_params,
                    )
                }
            })
            .sum()
    }

    /// Populate the graph with the current peaks and then select the best solutions and
    /// return them.
    ///
    /// # Arguments
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `params`: The isotopic pattern generation parameters to use
    fn graph_step_deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: (i32, i32),
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
    ) -> Result<Vec<IsotopicFit>, DeconvolutionError> {
        let mut fit_accumulator = Vec::new();
        self.populate_graph(
            error_tolerance,
            charge_range,
            left_search_limit,
            right_search_limit,
            isotopic_params,
        );
        self.select_best_disjoint_subgraphs(&mut fit_accumulator)?;
        Ok(fit_accumulator)
    }
}

/// Behavior for registering deconvolution solutions for specific experimental peaks
pub trait TargetedDeconvolution<C: CentroidLike>:
    IsotopicPatternFitter<C> + RelativePeakSearch<C>
{
    /// A place to store a solution.
    type TargetSolution;

    /// Deconvolve the specific target peak and register a solution handle that can be used
    /// to retrieve the final solution later.
    /// # Arguments
    /// - `peak`: The peak key to seed the targeted solution with
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `params`: The isotopic pattern generation parameters to use
    fn targeted_deconvolution(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        params: IsotopicPatternParams,
    ) -> Self::TargetSolution;

    /// Retrieve a solution by its handle
    fn resolve_target<'a, 'b: 'a>(
        &self,
        deconvoluted_peaks: &'b MassPeakSetType<DeconvolvedSolutionPeak>,
        target: &'a Self::TargetSolution,
    ) -> Option<&'b DeconvolvedSolutionPeak>;
}


/// Deconvolve the entire spectrum iteratively
pub trait IsotopicDeconvolutionAlgorithm<C: CentroidLike> {
    /// Process the spectrum peak list, extracting solutions, subtracting
    /// used signal, and repeat the process until `max_iterations` have elapsed
    /// or the ratio of change in remaining intensity falls below `convergence`.
    ///
    /// # Arguments
    /// - `error_tolerance`: The mass error tolerance for matching isotopic peaks
    /// - `charge_range`: The range of charge states to consider for each peak
    /// - `left_search_limit`: The number of isotopic shifts lower from each peak m/z to probe
    /// - `right_search_limit`: The number of isotopic shifts up from the each peak m/z to probe
    /// - `convergence`: The ratio of change in total intensity over remaining intensity below which
    ///   subsequent iterations are skipped.
    /// - `max_iterations`: The maximum number of iterations to run through.
    ///
    /// # Returns
    /// The set of all solved [`DeconvolvedSolutionPeak`] in a [`MassPeakSetType`] if successful. [`DeconvolutionError`]
    /// otherwise.
    ///
    fn deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
        convergence: f32,
        max_iterations: u32,
    ) -> Result<MassPeakSetType<DeconvolvedSolutionPeak>, DeconvolutionError>;
}
