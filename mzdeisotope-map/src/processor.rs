use std::{collections::HashSet, mem};

use chemical_elements::{isotopic_pattern::TheoreticalIsotopicPattern, neutral_mass};

use identity_hash::BuildIdentityHasher;
use itertools::Itertools;
use mzpeaks::{
    coordinate::{BoundingBox, CoordinateRange, QuadTree, Span1D},
    feature::{ChargedFeature, Feature, TimeInterval},
    feature_map::FeatureMap,
    prelude::*,
    CoordinateLikeMut, Mass, MZ,
};

use mzdeisotope::{
    charge::ChargeRange,
    isotopic_model::{
        IsotopicPatternGenerator, TheoreticalIsotopicDistributionScalingMethod, PROTON,
    },
    scorer::{IsotopicFitFilter, IsotopicPatternScorer, ScoreInterpretation, ScoreType},
};
use mzsignal::feature_mapping::graph::FeatureGraphBuilder;
use tracing::{debug, trace};

use crate::{
    dependency_graph::FeatureDependenceGraph, feature_fit::{FeatureSetFit, MapCoordinate}, fmap::IndexedFeature, solution::{reflow_feature, DeconvolvedSolutionFeature, FeatureMerger, MZPointSeries}, traits::{
        DeconvolutionError, FeatureIsotopicFitter, FeatureMapMatch, FeatureMapType,
        FeatureSearchParams, FeatureType, GraphFeatureDeconvolution, GraphStepResult,
    }, FeatureSetIter
};

#[derive(Debug)]
pub struct EnvelopeConformer {
    pub minimum_theoretical_abundance: f64,
}

impl EnvelopeConformer {
    pub fn new(minimum_theoretical_abundance: f64) -> Self {
        Self {
            minimum_theoretical_abundance,
        }
    }

    pub fn conform_into<
        C: CentroidLike + Default + CoordinateLikeMut<MZ> + IntensityMeasurementMut,
    >(
        &self,
        experimental: Vec<Option<C>>,
        theoretical: &mut TheoreticalIsotopicPattern,
        cleaned_eid: &mut Vec<C>,
    ) -> usize {
        let mut n_missing: usize = 0;
        let mut total_intensity = 0.0f32;

        for (observed_peak, peak) in experimental.into_iter().zip(theoretical.iter()) {
            if observed_peak.is_none() {
                let mut templated_peak = C::default();
                *templated_peak.coordinate_mut() = peak.mz();
                *templated_peak.intensity_mut() = 1.0;
                if peak.intensity > self.minimum_theoretical_abundance {
                    n_missing += 1;
                }
                total_intensity += 1.0;
                cleaned_eid.push(templated_peak)
            } else {
                let observed_peak = observed_peak.unwrap();
                total_intensity += observed_peak.intensity();
                cleaned_eid.push(observed_peak)
            }
        }

        let total_intensity = total_intensity as f64;
        theoretical.iter_mut().for_each(|p| {
            p.intensity *= total_intensity;
        });
        n_missing
    }

    pub fn conform<C: CentroidLike + Default + CoordinateLikeMut<MZ> + IntensityMeasurementMut>(
        &self,
        experimental: Vec<Option<C>>,
        theoretical: &mut TheoreticalIsotopicPattern,
    ) -> (Vec<C>, usize) {
        let mut cleaned_eid = Vec::with_capacity(experimental.len());
        let n_missing = self.conform_into(experimental, theoretical, &mut cleaned_eid);
        (cleaned_eid, n_missing)
    }
}

const MAX_COMBINATIONS: usize = 1000;

#[derive(Debug)]
pub struct FeatureProcessorBuilder<
    Y: Clone + Default,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub feature_map: FeatureMap<MZ, Y, Feature<MZ, Y>>,
    pub isotopic_model: Option<I>,
    pub scorer: Option<S>,
    pub fit_filter: Option<F>,
    pub scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub prefer_multiply_charged: bool,
    pub minimum_size: usize,
    pub maximum_time_gap: f64,
    pub minimum_intensity: f32,
}

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > FeatureProcessorBuilder<Y, I, S, F>
{
    pub fn feature_map(&mut self, feature_map: FeatureMap<MZ, Y, Feature<MZ, Y>>) -> &mut Self {
        self.feature_map = feature_map;
        self
    }

    pub fn isotopic_model(&mut self, isotopic_model: I) -> &mut Self {
        self.isotopic_model = Some(isotopic_model);
        self
    }

    pub fn fit_filter(&mut self, fit_filter: F) -> &mut Self {
        self.fit_filter = Some(fit_filter);
        self
    }

    pub fn scorer(&mut self, scorer: S) -> &mut Self {
        self.scorer = Some(scorer);
        self
    }

    pub fn build(self) -> FeatureProcessor<Y, I, S, F> {
        FeatureProcessor::new(
            self.feature_map,
            self.isotopic_model.unwrap(),
            self.scorer.unwrap(),
            self.fit_filter.unwrap(),
            self.minimum_size,
            self.maximum_time_gap,
            self.minimum_intensity,
            self.prefer_multiply_charged,
        )
    }
}

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > Default for FeatureProcessorBuilder<Y, I, S, F>
{
    fn default() -> Self {
        Self {
            feature_map: Default::default(),
            isotopic_model: Default::default(),
            scorer: Default::default(),
            fit_filter: Default::default(),
            scaling_method: Default::default(),
            prefer_multiply_charged: true,
            minimum_size: 3,
            maximum_time_gap: 0.25,
            minimum_intensity: 5.0,
        }
    }
}

#[derive(Debug)]
pub struct FeatureProcessor<
    Y: Clone + Default,
    I: IsotopicPatternGenerator,
    S: IsotopicPatternScorer,
    F: IsotopicFitFilter,
> {
    pub feature_map: FeatureMapType<Y>,
    pub isotopic_model: I,
    pub scorer: S,
    pub fit_filter: F,
    pub scaling_method: TheoreticalIsotopicDistributionScalingMethod,
    pub prefer_multiply_charged: bool,
    pub minimum_size: usize,
    pub maximum_time_gap: f64,
    pub minimum_intensity: f32,
    feature_buffer: Vec<IndexedFeature<Y>>,
    envelope_conformer: EnvelopeConformer,
    dependency_graph: FeatureDependenceGraph,
}

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > FeatureIsotopicFitter<Y> for FeatureProcessor<Y, I, S, F>
{
    fn make_isotopic_pattern(
        &mut self,
        mz: f64,
        charge: i32,
        search_params: &FeatureSearchParams,
    ) -> TheoreticalIsotopicPattern {
        self.isotopic_model.isotopic_cluster(
            mz,
            charge,
            PROTON,
            search_params.truncate_after,
            search_params.ignore_below,
        )
    }

    fn fit_theoretical_distribution_on_features(
        &self,
        mz: f64,
        error_tolerance: Tolerance,
        charge: i32,
        base_tid: TheoreticalIsotopicPattern,
        max_missed_peaks: usize,
        threshold_scale: f32,
        feature: &Option<CoordinateRange<Y>>,
    ) -> Vec<FeatureSetFit> {
        let feature_groups =
            self.match_theoretical_isotopic_distribution(&base_tid, error_tolerance, feature);
        let n_real = feature_groups
            .iter()
            .flatten()
            .map(|s| s.len())
            .sum::<usize>();

        if n_real == 0 {
            return Vec::new();
        }
        let fgi = feature_groups
            .into_iter()
            .map(|g| {
                if g.is_none() {
                    vec![None].into_iter()
                } else {
                    let v: Vec<_> = g.unwrap().into_iter().map(Some).collect();
                    v.into_iter()
                }
            })
            .multi_cartesian_product()
            .enumerate();
        let mut snapped_tid: TheoreticalIsotopicPattern;
        let mut fits = Vec::new();
        for (i, features) in fgi.take(MAX_COMBINATIONS) {
            if i % 100 == 0 && i > 0 {
                debug!("... Considering combination {i} of {n_real} features for {mz}@{charge}");
            }

            if features.iter().all(|f| f.is_none()) {
                continue;
            }

            if let Some((_, f)) = features.first().unwrap() {
                snapped_tid = base_tid.clone().shift(mz - f.mz());
            } else {
                snapped_tid = base_tid.clone();
            };

            let mut counter = 0;
            let mut score_max: ScoreType = 0.0;

            let mut score_vec: Vec<ScoreType> = Vec::new();
            let mut time_vec: Vec<f64> = Vec::new();

            let mut features_vec: Vec<_> = Vec::with_capacity(features.len());
            let mut indices_vec: Vec<_> = Vec::with_capacity(features.len());
            for opt in features.into_iter() {
                if let Some((j, f)) = opt {
                    indices_vec.push(Some(j));
                    features_vec.push(Some(f.as_ref()));
                } else {
                    features_vec.push(None);
                    indices_vec.push(None);
                }
            }

            let it = FeatureSetIter::new(&features_vec);
            let mut cleaned_eid = Vec::with_capacity(snapped_tid.len());
            let mut tid = snapped_tid.clone();
            for (time, eid) in it {
                counter += 1;
                if counter % 500 == 0 && counter > 0 {
                    debug!("Step {counter} at {time} for feature combination {i} of {mz}@{charge}");
                }
                tid.clone_from(&snapped_tid);

                cleaned_eid.clear();
                let n_missing =
                    self.envelope_conformer
                        .conform_into(eid, &mut tid, &mut cleaned_eid);
                if n_missing > max_missed_peaks {
                    continue;
                }

                let score = self.scorer.score(&cleaned_eid, &tid);
                if score.is_nan() {
                    continue;
                }
                score_max = score_max.max(score);
                score_vec.push(score);
                time_vec.push(time);
            }

            if score_vec.is_empty() {
                continue;
            }

            let final_score = self.find_thresholded_score(&score_vec, score_max, threshold_scale);

            if !self.fit_filter.test_score(final_score) {
                continue;
            }

            let missing_features: usize = features_vec.iter().map(|f| f.is_none() as usize).sum();

            let (start, end) = features_vec
                .iter()
                .flatten()
                .map(|f| {
                    let pt = f.iter().next().unwrap();
                    let start = MapCoordinate::new(pt.0, pt.1);
                    let pt = f.iter().last().unwrap();
                    let end = MapCoordinate::new(pt.0, pt.1);
                    (start, end)
                })
                .next()
                .unwrap();

            let neutral_mass = neutral_mass(snapped_tid.origin, charge, PROTON);
            let fit = FeatureSetFit::new(
                indices_vec,
                snapped_tid,
                start,
                end,
                final_score,
                charge,
                missing_features,
                neutral_mass,
                counter,
                score_vec,
                time_vec,
            );

            fits.push(fit);
        }

        fits
    }
}

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > FeatureMapMatch<Y> for FeatureProcessor<Y, I, S, F>
{
    #[inline(always)]
    fn feature_map(&self) -> &FeatureMapType<Y> {
        &self.feature_map
    }

    fn feature_map_mut(&mut self) -> &mut FeatureMapType<Y> {
        &mut self.feature_map
    }
}

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > GraphFeatureDeconvolution<Y> for FeatureProcessor<Y, I, S, F>
{
    #[inline(always)]
    fn score_interpretation(&self) -> ScoreInterpretation {
        self.scorer.interpretation()
    }

    #[inline(always)]
    fn add_fit_to_graph(&mut self, fit: FeatureSetFit) {
        self.dependency_graph.add_fit(fit)
    }

    #[inline(always)]
    fn prefer_multiply_charged(&self) -> bool {
        self.prefer_multiply_charged
    }

    fn skip_feature(&self, feature: &FeatureType<Y>) -> bool {
        if self.minimum_size > feature.len() {
            debug!(
                "Skipping feature {} with {} points",
                feature.mz(),
                feature.len()
            );
            true
        } else {
            false
        }
    }

    fn dependency_graph_mut(&mut self) -> &mut FeatureDependenceGraph {
        &mut self.dependency_graph
    }
}

pub(crate) type IndexSet = HashSet<usize, BuildIdentityHasher<usize>>;

impl<
        Y: Clone + Default,
        I: IsotopicPatternGenerator,
        S: IsotopicPatternScorer,
        F: IsotopicFitFilter,
    > FeatureProcessor<Y, I, S, F>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_map: FeatureMap<MZ, Y, Feature<MZ, Y>>,
        isotopic_model: I,
        scorer: S,
        fit_filter: F,
        minimum_size: usize,
        maximum_time_gap: f64,
        minimum_intensity: f32,
        prefer_multiply_charged: bool,
    ) -> Self {
        let dependency_graph = FeatureDependenceGraph::new(scorer.interpretation());
        Self {
            feature_map: feature_map.into_iter().map(FeatureType::from).collect(),
            isotopic_model,
            scorer,
            fit_filter,
            scaling_method: TheoreticalIsotopicDistributionScalingMethod::default(),
            envelope_conformer: EnvelopeConformer::new(0.05),
            minimum_size,
            maximum_time_gap,
            minimum_intensity: minimum_intensity.max(1.001),
            prefer_multiply_charged,
            dependency_graph,
            feature_buffer: Vec::new(),
        }
    }

    pub fn builder() -> FeatureProcessorBuilder<Y, I, S, F> {
        FeatureProcessorBuilder::default()
    }

    fn find_thresholded_score(
        &self,
        scores: &[ScoreType],
        maximum_score: ScoreType,
        percentage: ScoreType,
    ) -> ScoreType {
        let threshold = maximum_score * percentage;
        let (acc, count) = scores.iter().fold((0.0, 0usize), |(total, count), val| {
            if *val > threshold {
                (total + val, count + 1)
            } else {
                (total, count)
            }
        });
        if count == 0 {
            0.0
        } else {
            acc / count as ScoreType
        }
    }

    pub fn finalize_fit(
        &mut self,
        fit: &FeatureSetFit,
        detection_threshold: f32,
        max_missed_peaks: usize,
    ) -> DeconvolvedSolutionFeature<Y> {
        let (time_range, _segments) = fit.find_separation(&self.feature_map, detection_threshold);
        let features: Vec<_> = fit
            .features
            .iter()
            .map(|i| i.map(|i| self.feature_map.get_item(i).as_ref()))
            .collect();
        let feat_iter = FeatureSetIter::new_with_time_interval(
            &features,
            time_range.start().unwrap(),
            time_range.end().unwrap(),
        );

        let base_tid = &fit.theoretical;
        let charge = fit.charge;
        let abs_charge = charge.abs();

        let mut scores = Vec::new();

        // Accumulators for the experimental envelope features
        let mut envelope_features = Vec::with_capacity(base_tid.len());
        // Accumulators for the residual intensities for features
        let mut residuals = Vec::with_capacity(base_tid.len());

        // Initialize the accumulators
        for _ in base_tid.iter() {
            envelope_features.push(MZPointSeries::default());
            residuals.push(Vec::new());
        }

        let mut deconv_feature = ChargedFeature::empty(charge);
        for (time, eid) in feat_iter {
            let mut tid = base_tid.clone();
            let (cleaned_eid, n_missing) = self.envelope_conformer.conform(eid, &mut tid);
            let n_real_peaks = cleaned_eid.len() - n_missing;
            if n_real_peaks == 0
                || (n_real_peaks == 1 && abs_charge > 1)
                || n_missing > max_missed_peaks
            {
                continue;
            }

            let score = self.scorer.score(&cleaned_eid, &tid);
            if score < 0.0 || score.is_nan() {
                continue;
            }
            scores.push(score);

            // Collect all the properties at this time point
            let mut total_intensity = 0.0;
            cleaned_eid
                .iter()
                .zip(tid.iter())
                .enumerate()
                .for_each(|(i, (e, t))| {
                    let intens = e.intensity().min(t.intensity());
                    total_intensity += intens;

                    // Update the envelope for this time point
                    envelope_features
                        .get_mut(i)
                        .unwrap()
                        .push_raw(e.mz(), intens);

                    // Update the residual for this time point
                    if e.intensity() * 0.7 < t.intensity() {
                        residuals[i].push(1.0);
                    } else {
                        residuals[i].push((e.intensity() - intens).max(1.0));
                    }
                    if tracing::enabled!(tracing::Level::TRACE) {
                        trace!(
                            "Resid({}): at time {time} slot {i} e:{}/t:{} -> {intens} with residual {}",
                            fit.features[i].map(|x| x.to_string()).unwrap_or("?".to_string()),
                            e.intensity(),
                            t.intensity(),
                            *residuals[i].last().unwrap()
                        );
                    }
                    debug_assert!(
                        e.intensity() >= *residuals[i].last().unwrap(),
                        "{} < {}",
                        e.intensity(),
                        *residuals[i].last().unwrap()
                    );
                });
            let neutral_mass = neutral_mass(tid[0].mz, charge, PROTON);
            deconv_feature.push_raw(neutral_mass, time, total_intensity);
        }

        drop(features);

        self.do_subtraction(fit, &residuals, &deconv_feature);

        DeconvolvedSolutionFeature::new(
            deconv_feature,
            fit.score,
            scores,
            envelope_features.into_boxed_slice(),
        )
    }

    fn do_subtraction(
        &mut self,
        fit: &FeatureSetFit,
        residuals: &[Vec<f32>],
        deconv_feature: &ChargedFeature<Mass, Y>,
    ) {
        // Do the subtraction along each feature now that the wide reads are done
        for (i, fidx) in fit.features.iter().enumerate() {
            if fidx.is_none() {
                continue;
            }
            let fidx = fidx.unwrap();

            // let time_vec: Vec<_> = self.feature_map[fidx].iter_time().collect();
            let tstart = self.feature_map[fidx].start_time().unwrap();
            let tend = self.feature_map[fidx].end_time().unwrap();

            let feature_to_reduce = &mut self.feature_map[fidx];
            let n_of = feature_to_reduce.len();
            let residual_intensity = &residuals[i];
            let n_residual = residual_intensity.len();
            let mz_of = feature_to_reduce.mz();

            for (time, res_int) in deconv_feature
                .iter_time()
                .zip(residual_intensity.iter().copied())
            {
                if let (Some(j), terr) = feature_to_reduce.find_time(time) {
                    if let Some((mz_at, time_at, int_at)) = feature_to_reduce.at_mut(j) {
                        if terr.abs() > 1e-3 {
                            let terr = time_at - time;
                            trace!(
                                "Did not find a coordinate {mz_of} for {time} ({time_at} {terr} {j}) in {i} ({tstart:0.3}-{tend:0.3})",
                            );
                        } else {
                            trace!(
                                "Residual({fidx}) {int_at} => {res_int} @ {mz_at}|{time}|{time_at} {n_residual}/{n_of}",
                            );
                            // debug_assert!(*int_at >= res_int, "Expectation failed: {int_at} >= {res_int}, delta {}", (*int_at - res_int));
                            *int_at = res_int.min(*int_at);
                        }
                    } else {
                        debug!("{i} unable to update {time}");
                    }
                } else {
                    debug!("{i} unable to update {time}");
                }
            }

            feature_to_reduce.invalidate();
        }
    }

    fn find_unused_features(
        &self,
        fits: &[FeatureSetFit],
        min_width_mz: f64,
        min_width_time: f64,
    ) -> Vec<usize> {
        let quads: QuadTree<f64, f64, BoundingBox<f64, f64>> = fits
            .iter()
            .map(|f| {
                BoundingBox::new(
                    (f.start.coord - min_width_mz, f.start.time - min_width_time),
                    (f.end.coord + min_width_mz, f.end.time + min_width_time),
                )
            })
            .collect();
        self.feature_map
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let bb_f = BoundingBox::new(
                    (f.mz(), f.start_time().unwrap()),
                    (f.mz(), f.end_time().unwrap()),
                );
                quads.overlaps(&bb_f).is_empty()
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn mask_features_at(&mut self, indices_to_mask: &[usize]) {
        for i in indices_to_mask.iter().copied() {
            let f = &mut self.feature_map[i];
            for (_, _, z) in f.iter_mut() {
                *z = 1.0;
            }
        }
    }

    fn remove_dead_points(&mut self, indices_to_mask: Option<&IndexSet>) {
        let n_before = self.feature_map.len();
        let n_points_before: usize = self.feature_map.iter().map(|f| f.len()).sum();

        let mut tmp = FeatureMapType::empty();

        mem::swap(&mut tmp, &mut self.feature_map);

        if self.feature_buffer.capacity() < tmp.len() {
            self.feature_buffer.reserve(tmp.len() - self.feature_buffer.capacity());
        }
        let mut features_acc = mem::take(&mut self.feature_buffer);

        let mut inner: Vec<_> = tmp.into_inner().into_inner();
        for (i, f) in inner.drain(..).enumerate() {
            let process = indices_to_mask
                .map(|mask| mask.contains(&i))
                .unwrap_or(true);
            if process {
                let parts: Vec<_> = f
                    .feature
                    .split_when(|_, (_, _, cur_int)| cur_int <= self.minimum_intensity)
                    .into_iter()
                    .filter(|s| s.len() >= self.minimum_size)
                    .collect();
                if parts.len() == 1 {
                    if parts[0].len() == f.len() {
                        features_acc.push(f)
                    } else {
                        let p: FeatureType<Y> = parts[0].to_owned().into();
                        features_acc.push(p)
                    }
                } else {
                    for s in parts {
                        let p: FeatureType<Y> = s.to_owned().into();
                        features_acc.push(p);
                    }
                }
            } else if f.len() >= self.minimum_size {
                features_acc.push(f);
            }
        }
        self.feature_buffer = inner;
        self.feature_map = FeatureMapType::new(features_acc);
        let n_after = self.feature_map.len();
        let n_points_after: usize = self.feature_map.iter().map(|f| f.len()).sum();
        debug!("{n_before} features, {n_points_before} points before, {n_after} features, {n_points_after} points after");
    }

    #[allow(clippy::too_many_arguments)]
    pub fn deconvolve(
        &mut self,
        error_tolerance: Tolerance,
        charge_range: ChargeRange,
        left_search_limit: i8,
        right_search_limit: i8,
        search_params: &FeatureSearchParams,
        convergence: f32,
        max_iterations: u32,
    ) -> Result<FeatureMap<Mass, Y, DeconvolvedSolutionFeature<Y>>, DeconvolutionError> {
        let mut before_tic: f32 = self.feature_map.iter().map(|f| f.total_intensity()).sum();
        let ref_tic = before_tic;
        if ref_tic == 0.0 || self.feature_map.is_empty() {
            debug!("The TIC of the feature map was zero or the feature map was empty. Skipping processing.");
            return Ok(Default::default());
        }
        let mut deconvoluted_features = Vec::new();
        let mut converged = false;
        let mut convergence_check = f32::MAX;

        let mut indices_modified = IndexSet::default();

        for i in 0..max_iterations {
            if i > 0 {
                self.remove_dead_points(Some(&indices_modified));
                indices_modified.clear();
            } else {
                self.remove_dead_points(None);
            }
            debug!(
                "Starting iteration {i} with remaining TIC {before_tic:0.4e} ({:0.3}%), {} feature fit",
                before_tic / ref_tic * 100.0,
                deconvoluted_features.len()
            );

            let GraphStepResult {
                selected_fits: fits,
            } = self.graph_step_deconvolve(
                error_tolerance,
                charge_range,
                left_search_limit,
                right_search_limit,
                search_params,
            )?;

            debug!("Found {} fits", fits.len());

            let minimum_size = self.minimum_size;
            let max_gap_size = self.maximum_time_gap;
            let n_before = deconvoluted_features.len();
            deconvoluted_features.extend(
                fits.iter()
                    .filter(|fit| fit.n_points >= minimum_size)
                    .map(|fit| {
                        let solution = self.finalize_fit(
                            fit,
                            search_params.detection_threshold,
                            search_params.max_missed_peaks,
                        );
                        indices_modified.extend(fit.features.iter().flatten().copied());

                        solution
                    })
                    .filter(|f| !f.is_empty())
                    .flat_map(|f| f.split_sparse(max_gap_size))
                    .filter(|fit| fit.len() >= minimum_size),
            );

            let n_new_features = deconvoluted_features.len() - n_before;

            if n_new_features == 0 {
                debug!("No new features were extracted on iteration {i} with remaining TIC {before_tic:0.4e}, {n_before} features fit");
                break;
            }

            if i == 0 {
                let min_width_mz = self.isotopic_model.largest_isotopic_width();
                let indices_to_mask = self.find_unused_features(&fits, min_width_mz, max_gap_size);
                debug!("{} features unused", indices_to_mask.len());
                self.mask_features_at(&indices_to_mask);
                indices_modified.extend(indices_to_mask);
            }

            let after_tic = self
                .feature_map
                .iter()
                .map(|f| f.as_ref().total_intensity())
                .sum();

            convergence_check = (before_tic - after_tic) / after_tic;
            if convergence_check <= convergence {
                debug!(
                    "Converged on iteration {i} with remaining TIC {before_tic:0.4e} - {after_tic:0.4e} = {:0.4e} ({convergence_check}), {} features fit",
                    before_tic - after_tic,
                    deconvoluted_features.len()
                );
                converged = true;
                break;
            } else {
                before_tic = after_tic;
            }
            self.dependency_graph.reset();
        }
        if !converged && !deconvoluted_features.is_empty() {
            debug!(
                "Failed to converge after {max_iterations} iterations with remaining TIC {before_tic:0.4e} ({convergence_check}), {} features fit",
                deconvoluted_features.len()
            );
        }

        let map: FeatureMap<_, _, _> = deconvoluted_features
            .into_iter()
            .filter(|f| self.fit_filter.test_score(f.score))
            .collect();

        if map.is_empty() {
            return Ok(map);
        }

        debug!("Building merge graph over {} features", map.len());
        let merger = FeatureMerger::<Y>::default();
        let map_merged = merger
            .bridge_feature_gaps(map, Tolerance::PPM(2.0), self.maximum_time_gap)
            .features
            .into_iter()
            .map(reflow_feature)
            .collect();
        Ok(map_merged)
    }
}
