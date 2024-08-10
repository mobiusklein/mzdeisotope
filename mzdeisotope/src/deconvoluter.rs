/*! Concrete deconvolution machinery implementations of [`deconv_traits`](crate::deconv_traits) */
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::charge::ChargeRangeIter;
use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    CachingIsotopicModel, IsotopicPatternGenerator, IsotopicPatternParams,
    TheoreticalIsotopicDistributionScalingMethod,
};
use crate::peak_graph::{DependenceCluster, FitRef, PeakDependenceGraph, SubgraphSolverMethod};
use crate::peaks::{PeakKey, PeakLike, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter, ScoreType,
};

use crate::deconv_traits::{
    DeconvolutionError, ExhaustivePeakSearch, GraphDependentSearch, IsotopicDeconvolutionAlgorithm,
    IsotopicPatternFitter, RelativePeakSearch, TargetedDeconvolution,
};
use crate::solution::DeconvolvedSolutionPeak;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::{prelude::*, MassPeakSetType};
use mzpeaks::{CentroidPeak, MZPeakSetType, Tolerance};

/// A targeted deconvolution solution for use with [`TargetedDeconvolution`]
#[derive(Debug, Clone, PartialEq)]
pub struct TargetLink {
    pub query: PeakKey,
    pub link: Option<DeconvolvedSolutionPeak>,
}

impl TargetLink {
    pub fn new(query: PeakKey, link: Option<DeconvolvedSolutionPeak>) -> Self {
        Self { query, link }
    }
}

/// A builder pattern for [`DeconvoluterType`] or [`GraphDeconvoluterType`]
#[derive(Debug, Default)]
pub struct DeconvoluterBuilder<
    C: PeakLike + Default,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    max_missed_peaks: Option<u16>,
    fit_filter: Option<F>,
    scorer: Option<S>,
    isotopic_model: Option<I>,
    peaks: Option<MZPeakSetType<C>>,
    use_quick_charge: bool,
}

impl<
        C: PeakLike + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > DeconvoluterBuilder<C, I, S, F>
{
    pub fn new() -> Self {
        Self {
            max_missed_peaks: Some(1),
            scorer: None,
            isotopic_model: None,
            fit_filter: None,
            peaks: None,
            use_quick_charge: false,
        }
    }

    /// Pre-specify `max_missed_peaks`
    pub fn missed_peaks(mut self, value: u16) -> Self {
        self.max_missed_peaks = Some(value);
        self
    }

    /// Pre-specify the [`IsotopicPatternScorer`] implementation
    pub fn scoring(mut self, value: S) -> Self {
        self.scorer = Some(value);
        self
    }

    /// Pre-specify the [`IsotopicFitFilter`] implementation
    pub fn filter(mut self, value: F) -> Self {
        self.fit_filter = Some(value);
        self
    }

    /// Pre-specify the [`IsotopicPatternGenerator`] implementation
    pub fn isotopic_model(mut self, value: I) -> Self {
        self.isotopic_model = Some(value);
        self
    }

    pub fn peaks(mut self, value: MZPeakSetType<C>) -> Self {
        self.peaks = Some(value);
        self
    }

    pub fn use_quick_charge(mut self, value: bool) -> Self {
        self.use_quick_charge = value;
        self
    }

    /// Create a [`DeconvoluterType`] instance from the builder's properties
    pub fn create(self) -> DeconvoluterType<C, I, S, F> {
        DeconvoluterType::new(
            self.peaks.unwrap_or_default(),
            self.isotopic_model
                .expect("An isotopic pattern generator must be specified"),
            self.scorer.expect("An isotopic scorer must be specified"),
            self.fit_filter
                .expect("An isotopic fit filter must be specified"),
            self.max_missed_peaks.unwrap(),
            self.use_quick_charge,
        )
    }

    /// Create a [`GraphDeconvoluterType`] instance from the builder's properties
    pub fn create_graph(self) -> GraphDeconvoluterType<C, I, S, F> {
        GraphDeconvoluterType::new(
            self.peaks.unwrap_or_default(),
            self.isotopic_model
                .expect("An isotopic pattern generator must be specified"),
            self.scorer.expect("An isotopic scorer must be specified"),
            self.fit_filter
                .expect("An isotopic fit filter must be specified"),
            self.max_missed_peaks.unwrap(),
            self.use_quick_charge,
        )
    }
}

/// A basic deconvolution that greedily takes the best solution for an isotopic peak in a search window.
///
/// This algorithm is not suitable for complex mass spectra.
#[derive(Debug)]
pub struct DeconvoluterType<
    C: PeakLike,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub(crate) peaks: WorkingPeakSet<C>,
    pub(crate) isotopic_model: I,
    pub(crate) scorer: S,
    pub(crate) fit_filter: F,
    pub(crate) scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub(crate) max_missed_peaks: u16,
    pub(crate) use_quick_charge: bool,
    targets: Vec<TargetLink>,
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    RelativePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    ExhaustivePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        if fit.missed_peaks > self.max_missed_peaks {
            return false;
        }
        self.fit_filter.test(fit)
    }

    fn quick_charge(
        &self,
        index: usize,
        charge_range: crate::charge::ChargeRange,
    ) -> crate::charge::ChargeListIter {
        self.peaks.quick_charge(index, charge_range)
    }

    #[inline(always)]
    fn use_quick_charge(&self) -> bool {
        self.use_quick_charge
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    DeconvoluterType<C, I, S, F>
{
    pub fn new(
        peaks: MZPeakSetType<C>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        max_missed_peaks: u16,
        use_quick_charge: bool,
    ) -> Self {
        Self {
            peaks: WorkingPeakSet::new(peaks),
            isotopic_model,
            scorer,
            fit_filter,
            scaling_method: TheoreticalIsotopicDistributionScalingMethod::default(),
            max_missed_peaks,
            targets: Vec::new(),
            use_quick_charge,
        }
    }

    /// Pre-compute the isotopic patterns for a range of m/z values and a
    /// range of charge states for a specific set of truncation rules.
    ///
    /// # See also
    /// [`IsotopicPatternGenerator::populate_cache`]
    pub fn populate_isotopic_model_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) {
        self.isotopic_model.populate_cache(
            min_mz,
            max_mz,
            min_charge,
            max_charge,
            charge_carrier,
            truncate_after,
            ignore_below,
        )
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    IsotopicPatternFitter<C> for DeconvoluterType<C, I, S, F>
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
        IsotopicFit::new(keys, peak, tid, charge, score, missed_peaks as u16)
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
        self.isotopic_model.isotopic_cluster(
            mz,
            charge,
            params.charge_carrier,
            params.truncate_after,
            params.ignore_below,
        )
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

/// A pre-specified [`DeconvoluterType`] with its type parameters fixed for ease of use.
pub type AveragineDeconvoluter<'lifespan> = DeconvoluterType<
    CentroidPeak,
    CachingIsotopicModel<'lifespan>,
    MSDeconvScorer,
    MaximizingFitFilter,
>;

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    TargetedDeconvolution<C> for DeconvoluterType<C, I, S, F>
{
    type TargetSolution = TargetLink;

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
        let link = TargetLink::new(peak, solution);
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
                .iter()
                .find(|p| *p == target)
        } else {
            None
        }
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    IsotopicDeconvolutionAlgorithm<C> for DeconvoluterType<C, I, S, F>
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
                    .find(|t| {
                        let k = t.query.to_index_unchecked();
                        k == 0 && p.index == u32::MAX || p.index == k
                    })
                    .and_then(|t| -> Option<i8> {
                        t.link = Some(p.clone());
                        None
                    });
            });
        Ok(MassPeakSetType::new(deconvoluted_peaks))
    }
}

/// The preferred algorithm, uses a dependencies graph tracks fit-to-peak and fit-to-fit
/// relationships to correctly deconvolve a complex spectrum.
///
/// The dependency tracking method by not double-counting or pre-emptively consuming an
/// isotopic peaks.
#[derive(Debug)]
pub struct GraphDeconvoluterType<
    C: PeakLike,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub(crate) inner: DeconvoluterType<C, I, S, F>,
    pub(crate) peak_graph: PeakDependenceGraph,
    solutions: Vec<TargetLink>,
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    GraphDeconvoluterType<C, I, S, F>
{
    pub fn new(
        peaks: MZPeakSetType<C>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        max_missed_peaks: u16,
        use_quick_charge: bool,
    ) -> Self {
        let inner = DeconvoluterType::new(
            peaks,
            isotopic_model,
            scorer,
            fit_filter,
            max_missed_peaks,
            use_quick_charge,
        );
        let peak_graph =
            PeakDependenceGraph::with_capacity(inner.peak_count(), inner.scorer.interpretation());

        Self {
            inner,
            peak_graph,
            solutions: Vec::new(),
        }
    }

    pub fn populate_isotopic_model_cache(
        &mut self,
        min_mz: f64,
        max_mz: f64,
        min_charge: i32,
        max_charge: i32,
        charge_carrier: f64,
        truncate_after: f64,
        ignore_below: f64,
    ) {
        self.inner.populate_isotopic_model_cache(
            min_mz,
            max_mz,
            min_charge,
            max_charge,
            charge_carrier,
            truncate_after,
            ignore_below,
        )
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    GraphDeconvoluterType<C, I, S, F>
{
    fn solve_subgraph_top(
        &mut self,
        cluster: DependenceCluster,
        fits: Vec<(FitRef, IsotopicFit)>,
        peak_accumulator: &mut Vec<IsotopicFit>,
    ) -> Result<(), DeconvolutionError> {
        if let Some(best_fit_key) = cluster.best_fit() {
            if let Some((_, fit)) = fits.into_iter().find(|(k, _)| k.key == best_fit_key.key) {
                peak_accumulator.push(fit);
                Ok(())
            } else {
                Err(DeconvolutionError::FailedToResolveFit(*best_fit_key))
            }
        } else {
            Ok(())
        }
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    IsotopicPatternFitter<C> for GraphDeconvoluterType<C, I, S, F>
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

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    RelativePeakSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    ExhaustivePeakSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        self.inner.check_isotopic_fit(fit)
    }

    fn quick_charge(
        &self,
        index: usize,
        charge_range: crate::charge::ChargeRange,
    ) -> crate::charge::ChargeListIter {
        self.inner.quick_charge(index, charge_range)
    }

    #[inline(always)]
    fn use_quick_charge(&self) -> bool {
        self.inner.use_quick_charge()
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    GraphDependentSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
    fn add_fit_dependence(&mut self, fit: IsotopicFit) {
        if fit.experimental.is_empty() || !self.check_isotopic_fit(&fit) {
            return;
        }
        let start = self.get_peak(*fit.experimental.first().unwrap()).mz();
        let end = self.get_peak(*fit.experimental.last().unwrap()).mz();
        self.peak_graph.add_fit(fit, start, end)
    }

    fn select_best_disjoint_subgraphs(
        &mut self,
        fit_accumulator: &mut Vec<IsotopicFit>,
    ) -> Result<(), DeconvolutionError> {
        let solutions = self.peak_graph.solutions(SubgraphSolverMethod::Greedy);
        let res: Result<(), DeconvolutionError> =
            solutions.into_iter().try_for_each(|(cluster, fits)| {
                self.solve_subgraph_top(cluster, fits, fit_accumulator)
            });
        res
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    TargetedDeconvolution<C> for GraphDeconvoluterType<C, I, S, F>
{
    type TargetSolution = TargetLink;

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
        let link = TargetLink {
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
                    .iter()
                    .find(|p| *p == target)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<C: PeakLike, I: IsotopicPatternGenerator, S: IsotopicPatternScorer, F: IsotopicFitFilter>
    IsotopicDeconvolutionAlgorithm<C> for GraphDeconvoluterType<C, I, S, F>
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
                    .find(|t| match t.query {
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
                    .and_then(|t| -> Option<i8> {
                        t.link = Some(p.clone());
                        None
                    });
            });
        Ok(MassPeakSetType::new(deconvoluted_peaks))
    }
}

/// A pre-specified [`GraphDeconvoluterType`] with its type parameters fixed for ease of use.
pub type GraphAveragineDeconvoluter<'lifespan, C> =
    GraphDeconvoluterType<C, CachingIsotopicModel<'lifespan>, MSDeconvScorer, MaximizingFitFilter>;

#[cfg(test)]
mod test {
    use std::fs;
    use std::io;

    use flate2::bufread::GzDecoder;

    use mzdata::MzMLReader;
    use mzpeaks::prelude::*;

    use crate::isotopic_model::IsotopicModels;
    use crate::isotopic_model::PROTON;
    use crate::scorer::MSDeconvScorer;

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
            false,
        );
        let p = PeakKey::Matched(0);
        let tol = Tolerance::PPM(10.0);
        let fit1 = task.fit_theoretical_isotopic_pattern(p, 1, tol);
        let fit2 = task.fit_theoretical_isotopic_pattern(p, 2, tol);
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
            false,
        );
        let solution_space = task.find_all_peak_charge_pairs::<ChargeRangeIter>(
            300.0,
            Tolerance::PPM(10.0),
            (1, 8).into(),
            1,
            1,
            true,
        );
        assert_eq!(solution_space.len(), 8);
        let n_matched = solution_space
            .iter()
            .map(|(k, _)| k.is_matched() as i32)
            .sum::<i32>();
        assert_eq!(n_matched, 7);
        let n_placeholders = solution_space
            .iter()
            .map(|(k, _)| k.is_placeholder() as i32)
            .sum::<i32>();
        assert_eq!(n_placeholders, 1);
    }

    #[test_log::test]
    fn test_file_graph_step() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();

        let mut deconvoluter = GraphAveragineDeconvoluter::new(
            centroided.peaks,
            IsotopicModels::Glycopeptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::new(10.0),
            3,
            false,
        );

        let fits = deconvoluter
            .graph_step_deconvolve(
                Tolerance::PPM(10.0),
                (1, 8),
                1,
                1,
                IsotopicPatternParams::default(),
            )
            .unwrap();

        let best_fit = fits.iter().max().unwrap();
        assert_eq!(best_fit.charge, 4);
        assert_eq!(best_fit.missed_peaks, 0);
        assert!(
            (best_fit.score - 3127.7483).abs() < 1e-3,
            "{}",
            best_fit.score
        );
        assert_eq!(fits.len(), 297);

        Ok(())
    }

    #[test_log::test]
    fn test_full_process() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();

        let mut deconvoluter = GraphAveragineDeconvoluter::new(
            centroided.peaks,
            IsotopicModels::Glycopeptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::new(10.0),
            3,
            false,
        );

        let isotopic_params = IsotopicPatternParams::default();
        let dpeaks = deconvoluter
            .deconvolve(
                Tolerance::PPM(10.0),
                (1, 8),
                1,
                1,
                isotopic_params,
                1e-3,
                10,
            )
            .unwrap();

        assert_eq!(dpeaks.len(), 558);
        let best_fit = dpeaks
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(best_fit.charge, 4);
        assert!(
            (best_fit.score - 3127.7483).abs() < 1e-3,
            "{}",
            best_fit.score
        );

        let expected_intensity = 1770737.8;
        assert!(
            (best_fit.intensity - expected_intensity).abs() < 1e-6,
            "Expected intensity {expected_intensity}, got {} delta {}",
            best_fit.intensity,
            best_fit.intensity - expected_intensity
        );
        eprintln!("intensity {}", best_fit.intensity);

        Ok(())
    }

    #[test_log::test]
    fn test_full_process_quick_charge() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();

        let mut deconvoluter = GraphAveragineDeconvoluter::new(
            centroided.peaks,
            IsotopicModels::Glycopeptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::new(10.0),
            3,
            true,
        );

        // If the cache is not populated, this test is not stable
        deconvoluter.populate_isotopic_model_cache(80.0, 3000.0, 1, 8, PROTON, 0.95, 0.001);

        let isotopic_params = IsotopicPatternParams::default();
        let dpeaks = deconvoluter
            .deconvolve(
                Tolerance::PPM(10.0),
                (1, 8),
                1,
                1,
                isotopic_params,
                1e-3,
                10,
            )
            .unwrap();

        let best_fit = dpeaks
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(best_fit.charge, 4);
        assert!(
            (best_fit.score - 3127.7483).abs() < 1e-3,
            "{}",
            best_fit.score
        );

        let expected_intensity = 1770737.8;
        assert!(
            (best_fit.intensity - expected_intensity).abs() < 1e-6,
            "Expected intensity {expected_intensity}, got {} delta {}",
            best_fit.intensity,
            best_fit.intensity - expected_intensity
        );
        eprintln!("intensity {}", best_fit.intensity);
        assert_eq!(dpeaks.len(), 567);
        Ok(())
    }

    #[test_log::test]
    fn test_file_step() -> io::Result<()> {
        let decoder = GzDecoder::new(io::BufReader::new(fs::File::open(
            "./tests/data/20150710_3um_AGP_001_29_30.mzML.gz",
        )?));
        let mut reader = MzMLReader::new(decoder);
        let scan = reader.next().unwrap();
        let centroided = scan.into_centroid().unwrap();
        let mzs: Vec<_> = centroided.peaks.iter().map(|p| p.mz).collect();

        let mut deconvoluter = AveragineDeconvoluter::new(
            centroided.peaks,
            IsotopicModels::Glycopeptide.into(),
            MSDeconvScorer::default(),
            MaximizingFitFilter::new(0.0),
            3,
            false,
        );
        let isotopic_params = IsotopicPatternParams::default();
        deconvoluter.isotopic_model.populate_cache(
            *mzs.first().unwrap(),
            *mzs.last().unwrap(),
            1,
            8,
            isotopic_params.charge_carrier,
            isotopic_params.truncate_after,
            isotopic_params.ignore_below,
        );
        let tol = Tolerance::PPM(10.0);
        let (total_combos, total_fits, total_missed) = mzs
            .iter()
            .map(|mz| {
                let combos = deconvoluter.find_all_peak_charge_pairs::<ChargeRangeIter>(
                    *mz,
                    Tolerance::PPM(10.0),
                    (1, 8).into(),
                    1,
                    1,
                    true,
                );
                let n_combos = combos.len();
                let fits = deconvoluter.fit_peaks_at_charge(combos, tol, isotopic_params);
                let n_missed_peaks: usize = fits.iter().map(|fit| fit.num_missed_peaks()).sum();
                let n_fits = fits.len();
                (n_combos, n_fits, n_missed_peaks)
            })
            .reduce(
                |(combo_acc, fit_acc, missed_acc), (n_combos, n_fits, n_missed)| {
                    (
                        combo_acc + n_combos,
                        fit_acc + n_fits,
                        missed_acc + n_missed,
                    )
                },
            )
            .unwrap();

        let isotopic_patterns = deconvoluter.isotopic_model.len();

        let (total_combos2, total_fits2, total_missed2) = mzs
            .iter()
            .map(|mz| {
                let combos = deconvoluter.find_all_peak_charge_pairs::<ChargeRangeIter>(
                    *mz,
                    Tolerance::PPM(10.0),
                    (1, 8).into(),
                    1,
                    1,
                    true,
                );
                let n_combos = combos.len();
                let fits = deconvoluter.fit_peaks_at_charge(combos, tol, isotopic_params);
                let n_missed_peaks: usize = fits.iter().map(|fit| fit.num_missed_peaks()).sum();
                let n_fits = fits.len();
                (n_combos, n_fits, n_missed_peaks)
            })
            .reduce(
                |(combo_acc, fit_acc, missed_acc), (n_combos, n_fits, n_missed)| {
                    (
                        combo_acc + n_combos,
                        fit_acc + n_fits,
                        missed_acc + n_missed,
                    )
                },
            )
            .unwrap();

        eprintln!(
            "{} {total_combos} {total_missed} {total_fits}",
            deconvoluter.isotopic_model.len()
        );

        assert_eq!(deconvoluter.isotopic_model.len(), isotopic_patterns);
        // assert_eq!(deconvoluter.isotopic_model.len(), 109788);

        assert_eq!(total_combos, total_combos2);
        assert_eq!(total_missed, total_missed2);
        assert_eq!(total_fits, total_fits2);
        assert_eq!(total_combos, 19142);
        assert_eq!(total_missed, 21125);
        assert_eq!(total_fits, 10788);

        let fits =
            deconvoluter.step_deconvolve(Tolerance::PPM(10.0), (1, 8), 1, 1, isotopic_params);

        let best_fit = fits.iter().max().unwrap();
        assert_eq!(best_fit.charge, 4);
        assert_eq!(best_fit.missed_peaks, 0);
        assert!(
            (best_fit.score - 3127.7483).abs() < 1e-3,
            "{}",
            best_fit.score
        );
        assert_eq!(fits.len(), 10281);
        Ok(())
    }
}
