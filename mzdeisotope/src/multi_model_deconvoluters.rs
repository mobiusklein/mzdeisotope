/*! An **experimental** variant of the the deconvolution machinery using multiple
 * isotopic models concurrently */
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::charge::ChargeRangeIter;
use crate::deconvoluter::{TrivialTargetLink, PeakDepenceGraphTargetLink};
use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    IsotopicPatternGenerator, IsotopicPatternParams, TheoreticalIsotopicDistributionScalingMethod,
};
use crate::peak_graph::{DependenceCluster, FitRef, PeakDependenceGraph, SubgraphSolverMethod};
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, ScoreType,
};

use crate::deconv_traits::{
    ExhaustivePeakSearch, GraphDependentSearch, IsotopicDeconvolutionAlgorithm,
    IsotopicPatternFitter, RelativePeakSearch, TargetedDeconvolution, QuerySet, DeconvolutionError,
};
use crate::solution::DeconvolvedSolutionPeak;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::{prelude::*, IntensityMeasurementMut, MassPeakSetType};
use mzpeaks::{CentroidPeak, Tolerance};


#[derive(Debug)]
pub struct MultiDeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub peaks: WorkingPeakSet<C>,
    pub isotopic_models: Vec<I>,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub max_missed_peaks: u16,
    pub use_quick_charge: bool,
    pub current_model_index: usize,
    targets: Vec<TrivialTargetLink>,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for MultiDeconvoluterType<C, I, S, F>
{
    fn fit_theoretical_isotopic_pattern_with_params(
        &mut self,
        peak: PeakKey,
        charge: i32,
        error_tolerance: Tolerance,
        params: IsotopicPatternParams,
    ) -> IsotopicFit {
        let mz = self.peaks.get(&peak).mz();
        let mut tid = self.create_isotopic_pattern(mz, charge, params);
        let (keys, missed_peaks) = self.peaks.match_theoretical(&tid, error_tolerance);
        let exp = self.peaks.collect_for(&keys);
        self.scaling_method.scale(&exp, &mut tid);
        let score = self.score_isotopic_fit(exp.as_slice(), &tid);
        let fit = IsotopicFit::new(keys, peak, tid, charge, score, missed_peaks as u16);
        #[cfg(feature="verbose")]
        self.write_log(&format!(
            "fit\t{:?}\t{}\t{}\t{}\t{}\t{}\n",
            peak, mz, charge, score, fit.theoretical, fit.theoretical.origin
        ))
        .unwrap();
        fit
    }

    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey {
        let (peak, _missed) = self.peaks.has_peak(mz, error_tolerance);
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

    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit) {
        self.peaks.subtract_theoretical_intensity(fit)
    }

    fn has_peak_direct(&self, mz: f64, error_tolerance: Tolerance) -> Option<&C> {
        self.peaks.has_peak_direct(mz, error_tolerance)
    }

    fn create_isotopic_pattern(
        &mut self,
        mz: f64,
        charge: i32,
        params: IsotopicPatternParams,
    ) -> TheoreticalIsotopicPattern {
        let tid = self.isotopic_models[self.current_model_index].isotopic_cluster(
            mz,
            charge,
            params.charge_carrier,
            params.truncate_after,
            params.ignore_below,
        );
        tid
    }

    fn score_isotopic_fit(
        &self,
        experimental: &[&C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        self.scorer.score(experimental, theoretical)
    }

    fn fit_theoretical_isotopic_pattern_incremental_from_seed(
        &self,
        seed_fit: &IsotopicFit,
        experimental: &[&C],
        _: IsotopicPatternParams,
        lower_bound: f64,
    ) -> Vec<IsotopicFit> {
        seed_fit
            .theoretical
            .clone()
            .incremental_truncation(lower_bound)
            .skip(1)
            .map(|mut tid| {
                let n = tid.len();
                let subset = &experimental[..n];
                self.scaling_method.scale(subset, &mut tid);
                let score = self.score_isotopic_fit(subset, &tid);
                let subset_keys = seed_fit.experimental[..n].to_vec();
                let missed_keys = subset_keys.iter().map(|k| k.is_placeholder() as u16).sum();
                IsotopicFit::new(
                    subset_keys,
                    seed_fit.seed_peak,
                    tid,
                    seed_fit.charge,
                    score,
                    missed_keys,
                )
            })
            .filter(|fit| self.check_isotopic_fit(fit))
            .collect()
    }

    fn collect_for(&self, keys: &[PeakKey]) -> Vec<&C> {
        self.peaks.collect_for(keys)
    }
}


impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for MultiDeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for MultiDeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        if fit.missed_peaks > self.max_missed_peaks {
            return false;
        }
        self.fit_filter.test(fit)
    }

    fn fit_peaks_at_charge(
        &mut self,
        peak_charge_set: QuerySet,
        error_tolerance: Tolerance,
        isotopic_params: IsotopicPatternParams,
    ) -> HashSet<IsotopicFit> {
        let mut solutions = HashSet::new();
        for i in 0..self.isotopic_models.len() {
            self.current_model_index = i;
            for (key, charge) in peak_charge_set.iter().copied() {
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
                            incremental_truncation
                        ).into_iter()
                    )
                }
                if self.check_isotopic_fit(&fit) {
                    solutions.insert(fit);
                }
            }
        }
        self.current_model_index = 0;
        solutions
    }

    fn quick_charge(&self, index: usize, charge_range: crate::charge::ChargeRange) -> crate::charge::ChargeListIter {
        self.peaks.quick_charge(index, charge_range)
    }

    fn use_quick_charge(&self) -> bool {
        self.use_quick_charge
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > TargetedDeconvolution<C> for MultiDeconvoluterType<C, I, S, F>
{
    type TargetSolution = TrivialTargetLink;

    fn targeted_deconvolution(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: crate::charge::ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        params: IsotopicPatternParams,
    ) -> Self::TargetSolution {
        let mz = self.peaks.get(&peak).mz();

        let peak_charge_set = self.find_all_peak_charge_pairs::<ChargeRangeIter>(
            mz,
            error_tolerance,
            charge_range.into(),
            left_search_limit,
            right_search_limit,
            true,
        );

        let fits = self.fit_peaks_at_charge(peak_charge_set, error_tolerance, params);
        let solution = if let Some(best_fit) = self.fit_filter.select(fits.into_iter()) {
            let mut dpeak = self.make_solution_from_fit(&best_fit, error_tolerance);
            dpeak.index = peak.to_index_unchecked();
            Some(dpeak)
        } else {
            None
        };
        let link = TrivialTargetLink::new(peak, solution);
        self.targets.push(link.clone());
        link
    }

    fn resolve_target<'a, 'b: 'a>(
        &self,
        solution: &'b MassPeakSetType<DeconvolvedSolutionPeak>,
        target: &'a Self::TargetSolution,
    ) -> Option<&'b DeconvolvedSolutionPeak> {
        if let Some(target) = target.link.as_ref() {
            solution
                .all_peaks_for(target.neutral_mass, Tolerance::PPM(1.0))
                .into_iter()
                .filter(|p| *p == target)
                .next()
        } else {
            None
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicDeconvolutionAlgorithm<C> for MultiDeconvoluterType<C, I, S, F>
{
    fn deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: (i32, i32),
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
        convergence: f32,
        max_iterations: u32,
    ) -> Result<MassPeakSetType<DeconvolvedSolutionPeak>, DeconvolutionError> {
        let mut before_tic = self.peaks.tic();
        let mut deconvoluted_peaks = Vec::with_capacity(32);
        deconvoluted_peaks.extend(self.targets.iter().flat_map(|t| t.link.as_ref()).cloned());
        for _ in 0..max_iterations {
            let fits = self.step_deconvolve(
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                isotopic_params,
            );
            deconvoluted_peaks.extend(fits.into_iter().map(|fit| {
                let peak = self.make_solution_from_fit(&fit, error_tolerance);
                self.peaks.subtract_theoretical_intensity(&fit);
                peak
            }));
            let after_tic = self.peaks.tic();
            if (before_tic - after_tic) / after_tic <= convergence {
                break;
            } else {
                before_tic = after_tic;
            }
        }
        deconvoluted_peaks = self.merge_isobaric_peaks(deconvoluted_peaks);
        deconvoluted_peaks
            .iter()
            .filter(|p| p.index > 0)
            .for_each(|p| {
                self.targets
                    .iter_mut()
                    .filter(|t| {
                        let k = t.query.to_index_unchecked();
                        k == 0 && p.index == u32::MAX || p.index == k
                    })
                    .next()
                    .and_then(|t| -> Option<i8> {
                        t.link = Some(p.clone());
                        None
                    });
            });
        Ok(MassPeakSetType::new(deconvoluted_peaks))
    }
}


#[derive(Debug)]
pub struct GraphMultiDeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub inner: MultiDeconvoluterType<C, I, S, F>,
    pub peak_graph: PeakDependenceGraph,
    solutions: Vec<PeakDepenceGraphTargetLink>,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphMultiDeconvoluterType<C, I, S, F>
{
    fn solve_subgraph_top(
        &mut self,
        cluster: DependenceCluster,
        fits: Vec<(FitRef, IsotopicFit)>,
        peak_accumulator: &mut Vec<IsotopicFit>,
    ) -> Result<(), DeconvolutionError> {
        if let Some(best_fit_key) = cluster.best_fit() {
            if let Some((_, fit)) = fits
                .into_iter()
                .find(|(k, _)| k.key == best_fit_key.key) {
                    peak_accumulator.push(fit);
                Ok(())
            }
            else {
                Err(DeconvolutionError::FailedToResolveFit(best_fit_key.clone()))
            }
        } else {
            Ok(())
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
    fn fit_theoretical_isotopic_pattern_with_params(
        &mut self,
        peak: PeakKey,
        charge: i32,
        error_tolerance: Tolerance,
        params: IsotopicPatternParams,
    ) -> IsotopicFit {
        self.inner.fit_theoretical_isotopic_pattern_with_params(
            peak,
            charge,
            error_tolerance,
            params,
        )
    }

    fn has_peak(&mut self, mz: f64, error_tolerance: Tolerance) -> PeakKey {
        self.inner.has_peak(mz, error_tolerance)
    }

    fn between(&mut self, m1: f64, m2: f64) -> Range<usize> {
        self.inner.between(m1, m2)
    }

    fn get_peak(&self, key: PeakKey) -> &C {
        self.inner.get_peak(key)
    }

    fn create_key(&mut self, mz: f64) -> PeakKey {
        self.inner.create_key(mz)
    }

    fn peak_count(&self) -> usize {
        self.inner.peak_count()
    }

    fn subtract_theoretical_intensity(&mut self, fit: &IsotopicFit) {
        self.inner.subtract_theoretical_intensity(fit)
    }

    fn has_peak_direct(&self, mz: f64, error_tolerance: Tolerance) -> Option<&C> {
        self.inner.has_peak_direct(mz, error_tolerance)
    }

    fn create_isotopic_pattern(
        &mut self,
        mz: f64,
        charge: i32,
        params: IsotopicPatternParams,
    ) -> TheoreticalIsotopicPattern {
        self.inner.create_isotopic_pattern(mz, charge, params)
    }

    fn score_isotopic_fit(
        &self,
        experimental: &[&C],
        theoretical: &TheoreticalIsotopicPattern,
    ) -> ScoreType {
        self.inner.score_isotopic_fit(experimental, theoretical)
    }

    fn fit_theoretical_isotopic_pattern_incremental_from_seed(
        &self,
        seed_fit: &IsotopicFit,
        experimental: &[&C],
        params: IsotopicPatternParams,
        lower_bound: f64,
    ) -> Vec<IsotopicFit> {
        self.inner
            .fit_theoretical_isotopic_pattern_incremental_from_seed(
                seed_fit,
                experimental,
                params,
                lower_bound,
            )
    }

    fn collect_for(&self, keys: &[PeakKey]) -> Vec<&C> {
        self.inner.collect_for(keys)
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        self.inner.check_isotopic_fit(fit)
    }

    fn quick_charge(&self, index: usize, charge_range: crate::charge::ChargeRange) -> crate::charge::ChargeListIter {
        self.inner.quick_charge(index, charge_range)
    }

    fn use_quick_charge(&self) -> bool {
        self.inner.use_quick_charge
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphDependentSearch<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
    fn add_fit_dependence(&mut self, fit: IsotopicFit) {
        if fit.experimental.is_empty() || !self.check_isotopic_fit(&fit) {
            return;
        }
        let start = self.get_peak(*fit.experimental.first().unwrap()).mz();
        let end = self.get_peak(*fit.experimental.last().unwrap()).mz();
        self.peak_graph.add_fit(fit, start, end)
    }

    fn select_best_disjoint_subgraphs(&mut self, fit_accumulator: &mut Vec<IsotopicFit>) -> Result<(), DeconvolutionError> {
        let solutions = self.peak_graph.solutions(SubgraphSolverMethod::Greedy);
        let res: Result<(), DeconvolutionError> = solutions.into_iter().map(|(cluster, fits)| {
            self.solve_subgraph_top(cluster, fits, fit_accumulator)
        }).collect();
        res
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > TargetedDeconvolution<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
    type TargetSolution = PeakDepenceGraphTargetLink;

    fn targeted_deconvolution(
        &mut self,
        peak: PeakKey,
        error_tolerance: Tolerance,
        charge_range: crate::charge::ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        params: IsotopicPatternParams,
    ) -> Self::TargetSolution {
        self._explore_local::<ChargeRangeIter>(
            peak,
            error_tolerance,
            charge_range.into(),
            left_search_limit,
            right_search_limit,
            params,
        );
        let link = PeakDepenceGraphTargetLink {
            query: peak,
            link: None,
        };
        self.solutions.push(link.clone());
        link
    }

    fn resolve_target<'a, 'b: 'a>(
        &self,
        solution: &'b MassPeakSetType<DeconvolvedSolutionPeak>,
        target: &'a Self::TargetSolution,
    ) -> Option<&'b DeconvolvedSolutionPeak> {
        let populated = self.solutions.iter().find(|t| t.query == target.query);

        if let Some(target) = populated {
            if let Some(target) = target.link.as_ref() {
                solution
                    .all_peaks_for(target.neutral_mass, Tolerance::PPM(1.0))
                    .into_iter()
                    .filter(|p| *p == target)
                    .next()
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicDeconvolutionAlgorithm<C> for GraphMultiDeconvoluterType<C, I, S, F>
{
    fn deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: crate::charge::ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        isotopic_params: IsotopicPatternParams,
        convergence: f32,
        max_iterations: u32,
    ) -> Result<MassPeakSetType<DeconvolvedSolutionPeak>, DeconvolutionError> {
        let mut before_tic = self.inner.peaks.tic();
        let ref_tick = before_tic;
        let mut deconvoluted_peaks = Vec::new();
        let mut converged = false;
        let mut convergence_check = f32::MAX;
        for i in 0..max_iterations {
            tracing::debug!(
                "Starting iteration {i} with remaining TIC {before_tic:0.4e} ({:0.3}%), {} peaks fit",
                before_tic / ref_tick * 100.0,
                deconvoluted_peaks.len()
            );
            let fits = self.graph_step_deconvolve(
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                isotopic_params,
            )?;
            deconvoluted_peaks.extend(fits.into_iter().map(|fit| {
                let mut peak = self.make_solution_from_fit(&fit, error_tolerance);
                self.inner.peaks.subtract_theoretical_intensity(&fit);
                if i == 0 {
                    self.solutions
                        .iter_mut()
                        .filter(|t| t.link.is_none())
                        .for_each(|t| {
                            if fit.experimental.contains(&t.query) {
                                // t.link = Some(peak.clone());
                                peak.index = match t.query {
                                    PeakKey::Matched(k) => {
                                        if k == 0 {
                                            u32::MAX
                                        } else {
                                            k
                                        }
                                    }
                                    _ => 0,
                                }
                            }
                        });
                }
                peak
            }));
            let after_tic = self.inner.peaks.tic();
            convergence_check = (before_tic - after_tic) / after_tic;
            if convergence_check <= convergence {
                tracing::debug!(
                    "Converged at on iteration {i} with remaining TIC {before_tic:0.4e} - {after_tic:0.4e} = {:0.4e} ({convergence_check}), {} peaks fit",
                    before_tic - after_tic,
                    deconvoluted_peaks.len()
                );
                converged = true;
                break;
            } else {
                before_tic = after_tic;
            }
            self.peak_graph.reset();
        }
        if !converged {
            tracing::debug!(
                "Failed to converge after {max_iterations} iterations with remaining TIC {before_tic:0.4e} ({convergence_check}), {} peaks fit",
                deconvoluted_peaks.len()
            );
        }

        let link_table: HashMap<PeakKey, C> = self
            .solutions
            .iter()
            .map(|link| {
                let c = self.get_peak(link.query);
                (link.query, c.clone())
            })
            .collect();

        let mut mask = HashSet::new();
        deconvoluted_peaks = self.merge_isobaric_peaks(deconvoluted_peaks);
        deconvoluted_peaks
            .iter()
            .filter(|p| p.index > 0)
            .for_each(|p| {
                self.solutions
                    .iter_mut()
                    .filter(|t| match t.query {
                        PeakKey::Matched(k) => k == 0 && p.index == u32::MAX || p.index == k,
                        PeakKey::Placeholder(j) => {
                            if !mask.contains(&j) {
                                let c = link_table.get(&t.query).unwrap();
                                tracing::debug!("Query peak {} is a placeholder", c.mz());
                                mask.insert(j);
                            }
                            false
                        }
                    })
                    .next()
                    .and_then(|t| -> Option<i8> {
                        t.link = Some(p.clone());
                        None
                    });
            });
        Ok(MassPeakSetType::new(deconvoluted_peaks))
    }
}
