use std::collections::{HashMap, HashSet};

#[cfg(feature="verbose")]
use std::fs::File;
#[cfg(feature="verbose")]
use std::io::{self, BufWriter, Write};
use std::ops::Range;

use crate::isotopic_fit::IsotopicFit;
use crate::isotopic_model::{
    CachingIsotopicModel, IsotopicPatternGenerator, IsotopicPatternParams, TheoreticalIsotopicDistributionScalingMethod,
};
use crate::peak_graph::{DependenceCluster, FitRef, PeakDependenceGraph, SubgraphSolverMethod};
use crate::peaks::{PeakKey, WorkingPeakSet};
use crate::scorer::{
    IsotopicFitFilter, IsotopicPatternScorer, MSDeconvScorer, MaximizingFitFilter, ScoreType,
};

use crate::deconv_traits::{
    ExhaustivePeakSearch, GraphDependentSearch, IsotopicDeconvolutionAlgorithm,
    IsotopicPatternFitter, RelativePeakSearch, TargetedDeconvolution,
};
use crate::solution::DeconvolvedSolutionPeak;

use chemical_elements::isotopic_pattern::TheoreticalIsotopicPattern;
use mzpeaks::{prelude::*, IntensityMeasurementMut, MassPeakSetType};
use mzpeaks::{CentroidPeak, MZPeakSetType, Tolerance};

#[derive(Debug, Clone, PartialEq)]
pub struct TrivialTargetLink {
    pub query: PeakKey,
    pub link: Option<DeconvolvedSolutionPeak>,
}

impl TrivialTargetLink {
    pub fn new(query: PeakKey, link: Option<DeconvolvedSolutionPeak>) -> Self {
        Self { query, link }
    }
}

#[derive(Debug, Default)]
pub struct DeconvoluterBuilder<
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    max_missed_peaks: Option<u16>,
    fit_filter: Option<F>,
    scorer: Option<S>,
    isotopic_model: Option<I>,
    peaks: Option<MZPeakSetType<C>>,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
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
        }
    }

    pub fn missed_peaks(mut self, value: u16) -> Self {
        self.max_missed_peaks = Some(value);
        self
    }

    pub fn scoring(mut self, value: S) -> Self {
        self.scorer = Some(value);
        self
    }

    pub fn filter(mut self, value: F) -> Self {
        self.fit_filter = Some(value);
        self
    }

    pub fn isotopic_model(mut self, value: I) -> Self {
        self.isotopic_model = Some(value);
        self
    }

    pub fn peaks(mut self, value: MZPeakSetType<C>) -> Self {
        self.peaks = Some(value);
        self
    }

    pub fn create(self) -> DeconvoluterType<C, I, S, F> {
        DeconvoluterType::new(
            self.peaks.unwrap(),
            self.isotopic_model.unwrap(),
            self.scorer.unwrap(),
            self.fit_filter.unwrap(),
            self.max_missed_peaks.unwrap(),
        )
    }

    pub fn create_graph(self) -> GraphDeconvoluterType<C, I, S, F> {
        GraphDeconvoluterType::new(
            self.peaks.unwrap(),
            self.isotopic_model.unwrap(),
            self.scorer.unwrap(),
            self.fit_filter.unwrap(),
            self.max_missed_peaks.unwrap(),
        )
    }
}

#[derive(Debug)]
pub struct DeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub peaks: WorkingPeakSet<C>,
    pub isotopic_model: I,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub max_missed_peaks: u16,
    targets: Vec<TrivialTargetLink>,
    #[cfg(feature="verbose")]
    log: Option<BufWriter<File>>,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > RelativePeakSearch<C> for DeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
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
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
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
            scaling_method: TheoreticalIsotopicDistributionScalingMethod::default(),
            max_missed_peaks,
            targets: Vec::new(),
            #[cfg(feature="verbose")]
            log: None,
        }
    }

    #[cfg(feature="verbose")]
    pub(crate) fn set_log_file(&mut self, sink: File) {
        self.log = Some(BufWriter::new(sink));
    }

    #[cfg(feature="verbose")]
    pub(crate) fn write_log(&mut self, message: &str) -> io::Result<()> {
        match self.log.as_mut() {
            Some(sink) => sink.write_all(message.as_bytes()),
            None => Ok(()),
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

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for DeconvoluterType<C, I, S, F>
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
        let tid = self.isotopic_model.isotopic_cluster(
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

pub type AveragineDeconvoluter<'lifespan> = DeconvoluterType<
    CentroidPeak,
    CachingIsotopicModel<'lifespan>,
    MSDeconvScorer,
    MaximizingFitFilter,
>;

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > TargetedDeconvolution<C> for DeconvoluterType<C, I, S, F>
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

        let peak_charge_set = self.find_all_peak_charge_pairs(
            mz,
            error_tolerance,
            charge_range,
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
    > IsotopicDeconvolutionAlgorithm<C> for DeconvoluterType<C, I, S, F>
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
    ) -> MassPeakSetType<DeconvolvedSolutionPeak> {
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
        MassPeakSetType::new(deconvoluted_peaks)
    }
}

#[derive(Debug, Clone)]
pub struct PeakDepenceGraphTargetLink {
    pub query: PeakKey,
    pub link: Option<DeconvolvedSolutionPeak>,
}

/// Graph deconvolution tracks fit dependencies so that whole peak list deconvolution doesn't re-use peaks
#[derive(Debug)]
pub struct GraphDeconvoluterType<
    C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub inner: DeconvoluterType<C, I, S, F>,
    pub peak_graph: PeakDependenceGraph,
    solutions: Vec<PeakDepenceGraphTargetLink>,
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphDeconvoluterType<C, I, S, F>
{
    pub fn new(
        peaks: MZPeakSetType<C>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        max_missed_peaks: u16,
    ) -> Self {
        let inner =
            DeconvoluterType::new(peaks, isotopic_model, scorer, fit_filter, max_missed_peaks);
        let peak_graph = PeakDependenceGraph::new(inner.scorer.interpretation());

        Self {
            inner,
            peak_graph,
            solutions: Vec::new(),
        }
    }

    #[cfg(feature="verbose")]
    pub(crate) fn set_log_file(&mut self, sink: File) {
        self.inner.set_log_file(sink);
    }

    #[cfg(feature="verbose")]
    pub(crate) fn write_log(&mut self, message: &str) -> io::Result<()> {
        self.inner.write_log(message)
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

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphDeconvoluterType<C, I, S, F>
{
    fn solve_subgraph_top(
        &mut self,
        cluster: DependenceCluster,
        fits: Vec<(FitRef, IsotopicFit)>,
        peak_accumulator: &mut Vec<IsotopicFit>,
    ) {
        if let Some(best_fit_key) = cluster.best_fit() {
            let (_, fit) = fits
                .into_iter()
                .find(|(k, _)| k.key == best_fit_key.key)
                .unwrap_or_else(|| {
                    panic!("Failed to locate a solution {:?}", best_fit_key);
                });
            #[cfg(feature="verbose")]
            self.write_log(&format!(
                "selected\t{:?}\t{:?}\t{}\t{}\t{}\n",
                fit.seed_peak,
                fit.experimental.first(),
                fit.charge,
                fit.score,
                fit.theoretical
            ))
            .unwrap();
            peak_accumulator.push(fit)
        }
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > IsotopicPatternFitter<C> for GraphDeconvoluterType<C, I, S, F>
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
    > RelativePeakSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > ExhaustivePeakSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
    fn check_isotopic_fit(&self, fit: &IsotopicFit) -> bool {
        self.inner.check_isotopic_fit(fit)
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphDependentSearch<C> for GraphDeconvoluterType<C, I, S, F>
{
    fn add_fit_dependence(&mut self, fit: IsotopicFit) {
        if fit.experimental.is_empty() || !self.check_isotopic_fit(&fit) {
            return;
        }
        let start = self.get_peak(*fit.experimental.first().unwrap()).mz();
        let end = self.get_peak(*fit.experimental.last().unwrap()).mz();
        self.peak_graph.add_fit(fit, start, end)
    }

    fn select_best_disjoint_subgraphs(&mut self, fit_accumulator: &mut Vec<IsotopicFit>) {
        let solutions = self.peak_graph.solutions(SubgraphSolverMethod::Greedy);
        solutions.into_iter().for_each(|(cluster, fits)| {
            self.solve_subgraph_top(cluster, fits, fit_accumulator);
        });
    }
}

impl<
        C: CentroidLike + Clone + From<CentroidPeak> + IntensityMeasurementMut,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > TargetedDeconvolution<C> for GraphDeconvoluterType<C, I, S, F>
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
        self._explore_local(
            peak,
            error_tolerance,
            charge_range,
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
    > IsotopicDeconvolutionAlgorithm<C> for GraphDeconvoluterType<C, I, S, F>
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
    ) -> MassPeakSetType<DeconvolvedSolutionPeak> {
        let mut before_tic = self.inner.peaks.tic();
        let ref_tick = before_tic;
        let mut deconvoluted_peaks = Vec::new();
        let mut converged = false;
        let mut convergence_check = f32::MAX;
        for i in 0..max_iterations {
            log::debug!(
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
            );
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
                log::debug!(
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
            log::debug!(
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
                                log::debug!("Query peak {} is a placeholder", c.mz());
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
        MassPeakSetType::new(deconvoluted_peaks)
    }
}

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
        );
        let solution_space =
            task.find_all_peak_charge_pairs(300.0, Tolerance::PPM(10.0), (1, 8), 1, 1, true);
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
        );

        let fits = deconvoluter.graph_step_deconvolve(
            Tolerance::PPM(10.0),
            (1, 8),
            1,
            1,
            IsotopicPatternParams::default(),
        );

        let best_fit = fits.iter().max().unwrap();
        assert_eq!(best_fit.charge, 4);
        assert_eq!(best_fit.missed_peaks, 0);
        assert!(
            (best_fit.score - 3131.769).abs() < 1e-3,
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
        );

        let isotopic_params = IsotopicPatternParams::default();
        let dpeaks = deconvoluter.deconvolve(
            Tolerance::PPM(10.0),
            (1, 8),
            1,
            1,
            isotopic_params,
            1e-3,
            10,
        );

        assert_eq!(dpeaks.len(), 557);
        let best_fit = dpeaks
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(best_fit.charge, 4);
        assert!(
            (best_fit.score - 3131.769).abs() < 1e-3,
            "{}",
            best_fit.score
        );

        let expected_intensity = 1771624.4;
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
                let combos = deconvoluter.find_all_peak_charge_pairs(
                    *mz,
                    Tolerance::PPM(10.0),
                    (1, 8),
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
                let combos = deconvoluter.find_all_peak_charge_pairs(
                    *mz,
                    Tolerance::PPM(10.0),
                    (1, 8),
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
        assert_eq!(total_missed, 21124);
        assert_eq!(total_fits, 10789);

        let fits =
            deconvoluter.step_deconvolve(Tolerance::PPM(10.0), (1, 8), 1, 1, isotopic_params);

        let best_fit = fits.iter().max().unwrap();
        assert_eq!(best_fit.charge, 4);
        assert_eq!(best_fit.missed_peaks, 0);
        assert!(
            (best_fit.score - 3127.5288).abs() < 1e-3,
            "{}",
            best_fit.score
        );
        assert_eq!(fits.len(), 10282);
        Ok(())
    }
}
